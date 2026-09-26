"""Deterministic S1-level fold assignment.

Folds are over Source 1 entities (the unit the metric averages over), stratified by
(country, number-of-true-matches bucket) so each fold sees the same singleton / multi-match mix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def stratified_s1_folds(
    s1_ids: Sequence[str],
    countries: Sequence[str],
    gt: Mapping[str, Sequence[str]],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    strata: dict[tuple[str, int], list[str]] = {}
    for s1, country in zip(s1_ids, countries, strict=True):
        key = (country, min(len(gt.get(s1, ())), 2))
        strata.setdefault(key, []).append(s1)
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    offset = 0
    for key in sorted(strata):
        members = sorted(strata[key])
        perm = rng.permutation(len(members))
        for i, j in enumerate(perm):
            folds[members[j]] = (i + offset) % n_folds
        offset += len(members)
    return folds
