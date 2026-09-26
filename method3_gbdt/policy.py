"""Decision policy on top of calibrated pair probabilities.

    raw P(match)  --Platt-->  calibrated p  --exclusivity-->  q  --strategy-->  match set

Exclusivity (an S2/S3 record belongs to at most one S1 entity):
    none  q = p
    hard  q = p for the S1 with the highest p for that record, 0 for the others
    soft  treat ownership of a record as one categorical choice among the S1 records competing
          for it plus "no owner":  q_i = o_i / (1 + sum_j o_j),  o = p / (1 - p).
          A record claimed by one S1 keeps q = p; two strong claims (0.9, 0.9) both drop to 0.47,
          i.e. ambiguous records are withheld -- exactly what a precision-weighted metric wants.

Strategy:
    expected_f  per S1, predict the top-k maximising E[F0.5] (Bayes-optimal for independent,
                calibrated labels; handles singletons and multi-matches without a threshold)
    threshold   q >= tau and q >= best_q_of_S1 - delta

Every combination is scored on out-of-fold probabilities; the best is kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import numpy.typing as npt

from er_common.decision import PlattCalibrator, exclusive_mask, expected_fbeta_topk, threshold_margin
from er_common.metrics import GroupedEvaluator
from er_common.pairs import PairTable

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]
IntArr = npt.NDArray[np.int64]

EXCLUSIVITY = ("none", "hard", "soft")
STRATEGIES = ("expected_f", "threshold")
TAUS = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2).tolist())
DELTAS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


@dataclass(frozen=True)
class Policy:
    calib_a: float = 1.0
    calib_b: float = 0.0
    exclusivity: str = "soft"
    strategy: str = "expected_f"
    tau: float = 0.5
    delta: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def apply_exclusivity(p: FloatArr, cand: IntArr, mode: str) -> FloatArr:
    if mode == "none":
        return p
    if mode == "hard":
        return np.where(exclusive_mask(cand, p), p, 0.0)
    if mode == "soft":
        o = np.clip(p, 0.0, 1 - 1e-6)
        o = o / (1.0 - o)
        total = np.bincount(cand, weights=o, minlength=int(cand.max()) + 1 if len(cand) else 0)
        return o / (1.0 + total[cand])
    raise ValueError(f"unknown exclusivity mode {mode!r}")


def decide(policy: Policy, q: FloatArr, pt: PairTable) -> BoolArr:
    if policy.strategy == "expected_f":
        return expected_fbeta_topk(pt.s1_idx, q, pt.n_s1)
    if policy.strategy == "threshold":
        return threshold_margin(pt.s1_idx, q, pt.n_s1, policy.tau, policy.delta)
    raise ValueError(f"unknown strategy {policy.strategy!r}")


def apply_policy(policy: Policy, raw: FloatArr, pt: PairTable) -> tuple[BoolArr, FloatArr]:
    p = PlattCalibrator(policy.calib_a, policy.calib_b)(raw)
    q = apply_exclusivity(p, pt.c_idx, policy.exclusivity)
    return decide(policy, q, pt), q


def fit_policy(
    raw: FloatArr,
    pt: PairTable,
    label: BoolArr,
    n_true: IntArr,
    active_s1: BoolArr,
    exclusivity: tuple[str, ...] = EXCLUSIVITY,
    strategies: tuple[str, ...] = STRATEGIES,
    fixed_threshold: float | None = None,
) -> tuple[Policy, float]:
    """Fit calibration on active S1 pairs and pick the best (exclusivity, strategy, tau, delta)."""
    active_pairs = active_s1[pt.s1_idx]
    cal = PlattCalibrator.fit(raw[active_pairs], label[active_pairs])
    p = cal(raw)
    ev = GroupedEvaluator(pt.s1_idx, label, n_true, active_s1)
    best: tuple[float, Policy] | None = None

    def consider(score: float, pol: Policy) -> None:
        nonlocal best
        if best is None or score > best[0] + 1e-9:
            best = (score, pol)

    for excl in exclusivity:
        q = apply_exclusivity(p, pt.c_idx, excl)
        base = Policy(calib_a=cal.a, calib_b=cal.b, exclusivity=excl)
        if "expected_f" in strategies:
            pol = replace(base, strategy="expected_f")
            consider(ev.score(decide(pol, q, pt)), pol)
        if "threshold" in strategies:
            grid = [(fixed_threshold, 1.0)] if fixed_threshold is not None else [(t, d) for t in TAUS for d in DELTAS]
            for tau, delta in grid:
                pol = replace(base, strategy="threshold", tau=tau, delta=delta)
                consider(ev.score(threshold_margin(pt.s1_idx, q, pt.n_s1, tau, delta)), pol)
    assert best is not None
    return best[1], best[0]
