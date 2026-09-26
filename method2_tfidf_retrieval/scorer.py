"""Method 2 pair features and weighted similarity scorer.

    name_score    = mean(name char-TF-IDF cos, name word-TF-IDF cos, token_sort, Jaro-Winkler)
    address_score = mean(addr char-TF-IDF cos, addr word-TF-IDF cos, token_set)
    score         = w_name * name_score + (1 - w_name) * address_score
                    + postal / house-number bonus-penalty - country-conflict penalty

TF-IDF weighting is what separates this from Method 1: generic tokens ("hospital", "traders",
"pvt") carry little weight, distinctive ones ("balaji", "dupont") carry most of it.
A missing address contributes a tuned neutral value; weights, threshold ``tau``, margin ``delta``
and the exclusivity switch are tuned on entity-level F0.5.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.decision import exclusive_mask, threshold_margin
from er_common.pairs import PairTable
from er_common.similarity import TfidfSpace, TokenSets, cosine_pairs, paired, tri_state
from er_common.tuning import Dim
from method2_tfidf_retrieval.retrieval import RetrievalResult

FloatArr = npt.NDArray[np.float64]
BoolArr = npt.NDArray[np.bool_]


def _col(df: pd.DataFrame, name: str, idx: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
    return df[name].to_numpy()[idx]


def tfidf_features(pt: PairTable, ret: RetrievalResult) -> pd.DataFrame:
    a, b, ia, ib = pt.s1, pt.s23, pt.s1_idx, pt.c_idx
    nq, nd = ret.spaces["name_char"]
    aq, ad = ret.spaces["addr_char"]
    wn_q, wn_d = TfidfSpace("word", (1, 1)).fit_transform(a["name_core"].tolist(), b["name_core"].tolist())
    wa_q, wa_d = TfidfSpace("word", (1, 1)).fit_transform(a["addr_core"].tolist(), b["addr_core"].tolist())
    na, nb = _col(a, "name_core", ia), _col(b, "name_core", ib)
    aa, ab = _col(a, "addr_core", ia), _col(b, "addr_core", ib)
    nums = TokenSets(a["nums"].tolist(), b["nums"].tolist()).features(ia, ib)
    f = {
        "name_char_cos": cosine_pairs(nq, nd, ia, ib),
        "name_word_cos": cosine_pairs(wn_q, wn_d, ia, ib),
        "name_tsort": paired(na, nb, "tsort"),
        "name_jw": paired(na, nb, "jw"),
        "addr_char_cos": cosine_pairs(aq, ad, ia, ib),
        "addr_word_cos": cosine_pairs(wa_q, wa_d, ia, ib),
        "addr_tset": paired(aa, ab, "tset"),
        "postal": tri_state(_col(a, "postal", ia), _col(b, "postal", ib)),
        "house": tri_state(_col(a, "house", ia), _col(b, "house", ib)),
        "nums_jaccard": nums["jaccard"],
        "country": tri_state(_col(a, "country_norm", ia), _col(b, "country_norm", ib)),
    }
    df = pd.DataFrame(f)
    with np.errstate(all="ignore"):
        name_cols = ["name_char_cos", "name_word_cos", "name_tsort", "name_jw"]
        addr_cols = ["addr_char_cos", "addr_word_cos", "addr_tset"]
        df["name_score"] = df[name_cols].mean(axis=1, skipna=True).fillna(0.0)
        df["addr_score"] = df[addr_cols].mean(axis=1, skipna=True)
    return df


@dataclass(frozen=True)
class ScoreParams:
    w_name: float = 0.5
    v_addr_missing: float = 0.5
    b_postal: float = 0.05
    p_postal: float = 0.10
    b_house: float = 0.05
    p_house: float = 0.10
    p_country: float = 0.30
    tau: float = 0.70
    delta: float = 0.15
    exclusive: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


SPACE: dict[str, Dim] = {
    "v_addr_missing": Dim(0.0, 1.0),
    "w_name": Dim(0.2, 0.8),
    "b_postal": Dim(0.0, 0.2),
    "p_postal": Dim(0.0, 0.3),
    "b_house": Dim(0.0, 0.2),
    "p_house": Dim(0.0, 0.3),
    "p_country": Dim(0.0, 1.0),
    "tau": Dim(0.3, 1.1),
    "delta": Dim(0.0, 0.5),
    "exclusive": Dim(0.0, 1.0, integer=True),
}


class Scorer:
    def __init__(self, pt: PairTable):
        f = pt.feats
        self.name = f["name_score"].to_numpy(np.float64)
        self.addr = f["addr_score"].to_numpy(np.float64)
        self.postal = f["postal"].to_numpy(np.float64)
        self.house = f["house"].to_numpy(np.float64)
        self.country = f["country"].to_numpy(np.float64)
        self.group, self.cand, self.n_groups = pt.s1_idx, pt.c_idx, pt.n_s1

    def score(self, p: ScoreParams) -> FloatArr:
        addr = np.where(np.isnan(self.addr), p.v_addr_missing, self.addr)
        out: FloatArr = (
            p.w_name * self.name
            + (1 - p.w_name) * addr
            + p.b_postal * (self.postal == 1)
            - p.p_postal * (self.postal == 0)
            + p.b_house * (self.house == 1)
            - p.p_house * (self.house == 0)
            - p.p_country * (self.country == 0)
        )
        return out

    def decide(self, p: ScoreParams) -> tuple[BoolArr, FloatArr]:
        s = self.score(p)
        base = s >= p.tau
        if p.exclusive >= 0.5:
            base &= exclusive_mask(self.cand, s, eligible=base)
        return threshold_margin(self.group, s, self.n_groups, tau=p.tau, delta=p.delta, base=base), s
