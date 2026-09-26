"""Candidate pair table shared by all methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.progress import progress

IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]


@dataclass
class PairTable:
    """Flat (S1 record, S2/S3 record) candidate list plus per-pair features.

    ``s1`` / ``s23`` are the normalised record frames; ``s1_idx`` / ``c_idx`` index their rows.
    Pairs are unique and sorted by (s1_idx, c_idx).
    """

    s1: pd.DataFrame
    s23: pd.DataFrame
    s1_idx: IntArr
    c_idx: IntArr
    feats: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if len(self.s1_idx) != len(self.c_idx):
            raise ValueError("s1_idx and c_idx must be aligned")
        if len(self.feats) and len(self.feats) != len(self.s1_idx):
            raise ValueError("feature frame not aligned with pairs")

    @property
    def n_pairs(self) -> int:
        return len(self.s1_idx)

    @property
    def n_s1(self) -> int:
        return len(self.s1)

    @property
    def s1_ids(self) -> list[str]:
        return self.s1["entity_id"].tolist()

    @property
    def c_ids(self) -> npt.NDArray[np.str_]:
        return self.s23["entity_id"].to_numpy()

    def labels(self, gt: Mapping[str, Sequence[str]]) -> BoolArr:
        s1_ids = self.s1["entity_id"].to_numpy()
        c_ids = self.c_ids
        true_pairs = {(s, c) for s, cs in gt.items() for c in cs}
        return np.fromiter(
            (
                (s1_ids[i], c_ids[j]) in true_pairs
                for i, j in progress(zip(self.s1_idx, self.c_idx, strict=True), "label pairs", self.n_pairs, "pair")
            ),
            dtype=bool,
            count=self.n_pairs,
        )

    def n_true(self, gt: Mapping[str, Sequence[str]]) -> IntArr:
        return np.array([len(gt.get(s, ())) for s in self.s1["entity_id"]], dtype=np.int64)

    def id_lists(self, mask: BoolArr | None = None, score: FloatArr | None = None) -> dict[str, list[str]]:
        """{s1_id: [cand ids]} for pairs in ``mask`` (all pairs if None), best score first."""
        sel = np.ones(self.n_pairs, dtype=bool) if mask is None else mask
        idx = np.flatnonzero(sel)
        key = -score[idx] if score is not None else np.zeros(len(idx))
        order = idx[np.lexsort((self.c_idx[idx], key, self.s1_idx[idx]))]
        s1_ids = self.s1["entity_id"].to_numpy()
        c_ids = self.c_ids
        out: dict[str, list[str]] = {s: [] for s in s1_ids}
        for i in progress(order, "collect id lists", len(order), "pair"):
            out[s1_ids[self.s1_idx[i]]].append(c_ids[self.c_idx[i]])
        return out

    def subset(self, mask: BoolArr) -> PairTable:
        return PairTable(
            s1=self.s1,
            s23=self.s23,
            s1_idx=self.s1_idx[mask],
            c_idx=self.c_idx[mask],
            feats=self.feats.loc[mask].reset_index(drop=True) if len(self.feats) else self.feats,
        )


def unique_pairs(s1_idx: IntArr, c_idx: IntArr, n_c: int) -> tuple[IntArr, IntArr]:
    code = np.unique(s1_idx.astype(np.int64) * max(n_c, 1) + c_idx.astype(np.int64))
    return code // max(n_c, 1), code % max(n_c, 1)
