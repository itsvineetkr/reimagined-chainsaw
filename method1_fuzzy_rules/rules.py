"""Method 1 similarity features and the rule-based matcher.

Rule set (all thresholds tuned on entity-level F0.5, never hand-fixed):

    veto        country labels present and different
    strong      name >= t_name_strong  AND address >= t_addr_strong
    addr_rule   address >= t_addr_only AND name >= t_name_weak     (same premises, noisy name)
    score_rule  evidence >= tau
    evidence  = w_name * name + (1 - w_name) * address
                + b_postal [postal equal]  - p_postal [postal conflict]
                + b_house  [house equal]   - p_house  [house conflict]

A missing address (empty on either side) contributes a tuned neutral value ``v_addr_missing``
instead of counting as a mismatch. Candidate records claimed by several S1 entities go to the highest
evidence (``exclusive``); per entity only candidates within ``delta`` of the best are kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.decision import exclusive_mask, threshold_margin
from er_common.pairs import PairTable
from er_common.progress import stages
from er_common.similarity import acronym_match, paired, tri_state
from er_common.tuning import Dim

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def fuzzy_features(pt: PairTable) -> pd.DataFrame:
    a, b = pt.s1, pt.s23
    ia, ib = pt.s1_idx, pt.c_idx

    def col(df: pd.DataFrame, name: str, idx: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        return df[name].to_numpy()[idx]

    f: dict[str, npt.NDArray[np.float32]] = {}
    with stages(f"M1 features ({len(ia):,} pairs)", 5) as step:
        step("name fuzzy")
        na, nb = col(a, "name_core", ia), col(b, "name_core", ib)
        f["name_tsort"] = paired(na, nb, "tsort")
        f["name_tset"] = paired(na, nb, "tset")
        f["name_jw"] = paired(na, nb, "jw")
        step("name skeleton / acronym")
        f["name_skel"] = paired(col(a, "name_skel", ia), col(b, "name_skel", ib), "ratio")
        f["name_acronym"] = acronym_match(na, col(a, "name_acronym", ia), nb, col(b, "name_acronym", ib))
        step("trade names")
        alt_a, alt_b = col(a, "name_alt", ia), col(b, "name_alt", ib)
        alt = np.fmax(paired(alt_a, nb, "tsort"), paired(na, alt_b, "tsort"))
        f["name_alt"] = np.fmax(alt, paired(alt_a, alt_b, "tsort"))
        step("address fuzzy")
        aa, ab = col(a, "addr_core", ia), col(b, "addr_core", ib)
        f["addr_tset"] = paired(aa, ab, "tset")
        f["addr_tsort"] = paired(aa, ab, "tsort")
        step("postal / house / country")
        f["postal"] = tri_state(col(a, "postal", ia), col(b, "postal", ib))
        f["house"] = tri_state(col(a, "house", ia), col(b, "house", ib))
        f["country"] = tri_state(col(a, "country_norm", ia), col(b, "country_norm", ib))
    df = pd.DataFrame(f)
    name = np.nanmax(
        np.vstack(
            [
                0.5 * df["name_tsort"] + 0.5 * df["name_jw"],
                0.95 * df["name_skel"],
                df["name_alt"],
                0.9 * df["name_acronym"],
            ]
        ).astype(np.float64),
        axis=0,
    )
    df["name_sim"] = np.nan_to_num(name, nan=0.0)
    df["addr_sim"] = 0.5 * df["addr_tset"] + 0.5 * df["addr_tsort"]
    return df


@dataclass(frozen=True)
class RuleParams:
    w_name: float = 0.5
    v_addr_missing: float = 0.5
    b_postal: float = 0.05
    p_postal: float = 0.10
    b_house: float = 0.05
    p_house: float = 0.10
    t_name_strong: float = 0.90
    t_addr_strong: float = 0.80
    t_addr_only: float = 0.95
    t_name_weak: float = 0.60
    tau: float = 0.85
    delta: float = 0.15
    exclusive: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


SPACE: dict[str, Dim] = {
    "v_addr_missing": Dim(0.0, 1.0),
    "w_name": Dim(0.3, 0.8),
    "b_postal": Dim(0.0, 0.2),
    "p_postal": Dim(0.0, 0.3),
    "b_house": Dim(0.0, 0.15),
    "p_house": Dim(0.0, 0.3),
    "t_name_strong": Dim(0.7, 1.0),
    "t_addr_strong": Dim(0.6, 1.0),
    "t_addr_only": Dim(0.8, 1.01),
    "t_name_weak": Dim(0.3, 0.8),
    "tau": Dim(0.6, 1.1),
    "delta": Dim(0.0, 0.5),
    "exclusive": Dim(0.0, 1.0, integer=True),
}


class RuleEngine:
    """Pre-extracts float64 arrays once so each rule evaluation is a handful of vector ops."""

    def __init__(self, pt: PairTable):
        f = pt.feats
        self.name = f["name_sim"].to_numpy(np.float64)
        self.addr = f["addr_sim"].to_numpy(np.float64)
        self.postal = f["postal"].to_numpy(np.float64)
        self.house = f["house"].to_numpy(np.float64)
        self.country = f["country"].to_numpy(np.float64)
        self.group = pt.s1_idx
        self.cand = pt.c_idx
        self.n_groups = pt.n_s1

    def evidence(self, p: RuleParams) -> FloatArr:
        addr = np.where(np.isnan(self.addr), p.v_addr_missing, self.addr)
        out: FloatArr = (
            p.w_name * self.name
            + (1 - p.w_name) * addr
            + p.b_postal * (self.postal == 1)
            - p.p_postal * (self.postal == 0)
            + p.b_house * (self.house == 1)
            - p.p_house * (self.house == 0)
        )
        return out

    def decide(self, p: RuleParams) -> tuple[BoolArr, FloatArr]:
        ev = self.evidence(p)
        with np.errstate(invalid="ignore"):
            strong = (self.name >= p.t_name_strong) & (self.addr >= p.t_addr_strong)
            addr_rule = (self.addr >= p.t_addr_only) & (self.name >= p.t_name_weak)
        raw = ~(self.country == 0) & (strong | addr_rule | (ev >= p.tau))
        if p.exclusive >= 0.5:
            raw &= exclusive_mask(self.cand, ev, eligible=raw)
        mask = threshold_margin(self.group, ev, self.n_groups, tau=-np.inf, delta=p.delta, base=raw)
        return mask, ev
