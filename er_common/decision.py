"""Decision layer: turn pair scores into per-entity match sets.

Three building blocks, all vectorised over a flat pair table (``group`` = S1 index):

1. ``exclusive_mask``      -- an S2/S3 record can belong to at most one S1 entity (S1 is
                              deduplicated). Keep only the best-scoring S1 per candidate.
2. ``threshold_margin``    -- absolute threshold ``tau`` + relative margin ``delta`` to the
                              entity's best candidate + optional ``max_k``.
3. ``expected_fbeta_topk`` -- Bayes-optimal set under calibrated, independent probabilities:
                              for each entity pick k maximising E[F_beta(top-k)]. Singletons
                              fall out naturally (k = 0 wins when P(no match) dominates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from er_common.metrics import BETA

FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]


def group_max(group: IntArr, score: FloatArr, n_groups: int) -> FloatArr:
    out = np.full(n_groups, -np.inf)
    np.maximum.at(out, group, score)
    return out


def group_rank(group: IntArr, score: FloatArr) -> IntArr:
    """0-based rank of each pair inside its group by descending score (ties broken by position)."""
    order = np.lexsort((np.arange(len(score)), -score, group))
    g_sorted = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(len(order)) - run_start
    return rank


def exclusive_mask(cand: IntArr, score: FloatArr, eligible: BoolArr | None = None) -> BoolArr:
    """True for pairs holding the highest score for their candidate record among eligible pairs."""
    n = len(score)
    elig = np.ones(n, dtype=bool) if eligible is None else eligible
    idx = np.flatnonzero(elig)
    keep = np.zeros(n, dtype=bool)
    if idx.size == 0:
        return keep
    order = idx[np.lexsort((idx, -score[idx], cand[idx]))]
    c_sorted = cand[order]
    first = np.r_[True, c_sorted[1:] != c_sorted[:-1]]
    keep[order[first]] = True
    return keep


def threshold_margin(
    group: IntArr,
    score: FloatArr,
    n_groups: int,
    tau: float,
    delta: float,
    max_k: int = 0,
    base: BoolArr | None = None,
) -> BoolArr:
    """Pairs with ``score >= tau`` and ``score >= best_in_group - delta`` (and rank < max_k if > 0)."""
    eligible = np.ones(len(score), dtype=bool) if base is None else base
    s = np.where(eligible, score, -np.inf)
    gmax = group_max(group, s, n_groups)
    mask = eligible & (s >= tau) & (s >= gmax[group] - delta)
    if max_k > 0:
        mask &= group_rank(group, s) < max_k
    return mask


def _pb_pmf(p: FloatArr) -> FloatArr:
    pmf = np.ones(1)
    for x in p:
        pmf = np.convolve(pmf, np.array([1.0 - x, x]))
    return pmf


def expected_fbeta_best_k(p_sorted: FloatArr, beta: float = BETA, k_max: int = 20) -> tuple[int, float]:
    """Best k (and its expected F) for candidate probabilities sorted in descending order."""
    b2 = beta * beta
    m = len(p_sorted)
    best_k, best_e = 0, float(np.prod(1.0 - p_sorted))
    if m == 0:
        return 0, 1.0
    suffix: list[FloatArr] = [np.ones(1)] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = np.convolve(suffix[i + 1], np.array([1.0 - p_sorted[i], p_sorted[i]]))
    pin = np.ones(1)
    for k in range(1, min(m, k_max) + 1):
        pin = np.convolve(pin, np.array([1.0 - p_sorted[k - 1], p_sorted[k - 1]]))
        pout = suffix[k]
        a = np.arange(k + 1, dtype=np.float64)[:, None]
        b = np.arange(len(pout), dtype=np.float64)[None, :]
        val = (1.0 + b2) * a / (k + b2 * (a + b))
        e = float(pin @ val @ pout)
        if e > best_e + 1e-12:
            best_k, best_e = k, e
    return best_k, best_e


def expected_fbeta_topk(
    group: IntArr,
    prob: FloatArr,
    n_groups: int,
    beta: float = BETA,
    min_prob: float = 1e-3,
    k_max: int = 20,
    base: BoolArr | None = None,
) -> BoolArr:
    """Per group, predict the top-k pairs maximising expected F-beta (labels assumed independent)."""
    eligible = np.ones(len(prob), dtype=bool) if base is None else base
    p = np.where(eligible, prob, 0.0)
    mask = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(p >= min_prob)
    if idx.size == 0:
        return mask
    order = idx[np.lexsort((idx, -p[idx], group[idx]))]
    g_sorted = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    ends = np.r_[starts[1:], len(order)]
    for s, e in zip(starts, ends, strict=True):
        rows = order[s:e]
        if p[rows[0]] < 0.05:  # E[F|k>=1] <= p_top < P(no match) for any realistic list
            continue
        k, _ = expected_fbeta_best_k(p[rows], beta, k_max)
        mask[rows[:k]] = True
    return mask


@dataclass(frozen=True)
class PlattCalibrator:
    """p' = sigmoid(a * logit(p) + b), fitted by logistic regression on out-of-fold scores."""

    a: float = 1.0
    b: float = 0.0

    @staticmethod
    def _logit(p: FloatArr) -> FloatArr:
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(q / (1 - q))

    @classmethod
    def fit(cls, prob: FloatArr, label: BoolArr) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        x = cls._logit(np.asarray(prob, dtype=np.float64)).reshape(-1, 1)
        y = np.asarray(label, dtype=int)
        if y.min() == y.max():
            return cls()
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        lr.fit(x, y)
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))

    def __call__(self, prob: FloatArr) -> FloatArr:
        z = self.a * self._logit(np.asarray(prob, dtype=np.float64)) + self.b
        return 1.0 / (1.0 + np.exp(-z))
