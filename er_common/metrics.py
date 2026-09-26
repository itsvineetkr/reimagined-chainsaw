"""Entity-level F-beta exactly as the leaderboard computes it, plus a vectorised evaluator.

Per Source 1 entity with true set T and predicted set P (beta = 0.5):
    T = {}, P = {}   -> 1.0
    T = {}, P != {}  -> 0.0
    T != {}, P = {}  -> 0.0
    otherwise        -> (1 + b^2) |P n T| / (|P| + b^2 |T|)
and the score is the unweighted mean over entities.

Note that ``|T|`` always counts *all* ground-truth matches, including those the blocking stage
never generated -- recall lost in candidate generation is charged to the final score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from er_common.progress import progress

BETA = 0.5

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]


def f_beta_counts(n_tp: int, n_pred: int, n_true: int, beta: float = BETA) -> float:
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or n_tp == 0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * n_tp / (n_pred + b2 * n_true)


@dataclass(frozen=True)
class EvalReport:
    f05: float
    macro_precision: float  # over entities with >= 1 prediction
    macro_recall: float  # over entities with >= 1 true match
    pair_precision: float
    pair_recall: float
    singleton_accuracy: float  # share of true singletons predicted empty
    matched_entity_f05: float  # F0.5 restricted to entities with >= 1 true match
    false_merge_entities: int  # entities with >= 1 wrong id predicted
    n_entities: int
    n_singletons: int
    n_pred_pairs: int
    n_true_pairs: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"F0.5={self.f05:.4f}  P(macro)={self.macro_precision:.4f}  R(macro)={self.macro_recall:.4f}  "
            f"pairP={self.pair_precision:.4f}  pairR={self.pair_recall:.4f}  "
            f"singletonAcc={self.singleton_accuracy:.4f}  matchedF0.5={self.matched_entity_f05:.4f}  "
            f"falseMergeEntities={self.false_merge_entities}  n={self.n_entities}"
        )


def evaluate(
    pred: Mapping[str, Sequence[str]],
    gt: Mapping[str, Sequence[str]],
    s1_ids: Sequence[str],
    beta: float = BETA,
) -> EvalReport:
    f_all: list[float] = []
    f_matched: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    tp_tot = pred_tot = true_tot = 0
    n_single = single_ok = false_merge = 0
    for s1 in progress(s1_ids, "score entities", len(s1_ids), "S1"):
        p = set(pred.get(s1, ()))
        t = set(gt.get(s1, ()))
        tp = len(p & t)
        f = f_beta_counts(tp, len(p), len(t), beta)
        f_all.append(f)
        tp_tot += tp
        pred_tot += len(p)
        true_tot += len(t)
        if p:
            precs.append(tp / len(p))
            false_merge += int(tp < len(p))
        if t:
            recs.append(tp / len(t))
            f_matched.append(f)
        else:
            n_single += 1
            single_ok += int(not p)
    return EvalReport(
        f05=float(np.mean(f_all)) if f_all else 0.0,
        macro_precision=float(np.mean(precs)) if precs else 1.0,
        macro_recall=float(np.mean(recs)) if recs else 1.0,
        pair_precision=tp_tot / pred_tot if pred_tot else 1.0,
        pair_recall=tp_tot / true_tot if true_tot else 1.0,
        singleton_accuracy=single_ok / n_single if n_single else 1.0,
        matched_entity_f05=float(np.mean(f_matched)) if f_matched else 1.0,
        false_merge_entities=false_merge,
        n_entities=len(s1_ids),
        n_singletons=n_single,
        n_pred_pairs=pred_tot,
        n_true_pairs=true_tot,
    )


def candidate_recall(
    candidates: Mapping[str, Sequence[str]], gt: Mapping[str, Sequence[str]], s1_ids: Sequence[str]
) -> dict[str, float]:
    """Blocking diagnostics: pair recall, share of entities whose full true set was retrieved,
    and the F0.5 ceiling an oracle matcher could reach with these candidates."""
    hit = total = full = 0
    ceiling: list[float] = []
    sizes: list[int] = []
    for s1 in progress(s1_ids, "score entities", len(s1_ids), "S1"):
        c = set(candidates.get(s1, ()))
        t = set(gt.get(s1, ()))
        sizes.append(len(c))
        found = len(c & t)
        hit += found
        total += len(t)
        full += int(found == len(t))
        ceiling.append(f_beta_counts(found, found, len(t)))
    n = max(len(s1_ids), 1)
    return {
        "pair_recall": hit / total if total else 1.0,
        "entity_full_recall": full / n,
        "oracle_f05_ceiling": float(np.mean(ceiling)) if ceiling else 1.0,
        "mean_candidates": float(np.mean(sizes)) if sizes else 0.0,
        "max_candidates": float(max(sizes, default=0)),
    }


class GroupedEvaluator:
    """Fast macro F-beta for boolean masks over a fixed pair table (used inside tuning loops).

    ``group``   : int group id per pair (S1 index 0..G-1)
    ``label``   : bool per pair
    ``n_true``  : int per group -- full ground-truth count (includes matches missed by blocking)
    ``active``  : bool per group -- which S1 entities count towards the score
    """

    def __init__(self, group: IntArr, label: BoolArr, n_true: IntArr, active: BoolArr, beta: float = BETA):
        self.group = np.asarray(group, dtype=np.int64)
        self.label = np.asarray(label, dtype=bool)
        self.n_true = np.asarray(n_true, dtype=np.float64)
        self.active = np.asarray(active, dtype=bool)
        self.n_groups = len(self.n_true)
        self.b2 = beta * beta
        if self.active.sum() == 0:
            raise ValueError("GroupedEvaluator needs at least one active group")

    def per_group(self, mask: BoolArr) -> FloatArr:
        g = self.group[mask]
        k = np.bincount(g, minlength=self.n_groups).astype(np.float64)
        tp = np.bincount(g, weights=self.label[mask].astype(np.float64), minlength=self.n_groups)
        denom = k + self.b2 * self.n_true
        f = np.where(tp > 0, (1.0 + self.b2) * tp / np.where(denom > 0, denom, 1.0), 0.0)
        f = np.where(self.n_true == 0, (k == 0).astype(np.float64), f)
        return f

    def score(self, mask: BoolArr) -> float:
        return float(self.per_group(mask)[self.active].mean())
