"""Method 3 pair features (~70 columns), all language- and country-agnostic.

Groups
  retrieval   per-channel ranks / scores, RRF, #channels, fused rank, source flag (from Method 2)
  name        TF-IDF char/word cosine, 6 fuzzy scorers, skeleton ratio, IDF-weighted Jaccard,
              containment, exact flags, acronym, trade-name (dba), legal-form agreement,
              first/last-token agreement, length deltas
  address     TF-IDF char/word cosine, fuzzy scorers on core and full address, landmark
              similarity, IDF-weighted Jaccard, postal equality / prefix, house number,
              numeric-token overlap and conflicts
  frequency   how common the name is on each side (chains: many branches share a name)
  context     per-S1: rank / gap of this candidate vs the S1's best; per-candidate: how many S1
              records compete for it and the gap to the best competing S1 (reverse rank)
There is deliberately no country one-hot: the only country signal is label agreement, so an
unseen country (France in test) is handled by the same features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.pairs import PairTable
from er_common.similarity import TokenSets, acronym_match, paired, prefix_match, tri_state
from method2_tfidf_retrieval.retrieval import RetrievalResult
from method2_tfidf_retrieval.scorer import tfidf_features

IntArr = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]

# features whose partial dependence must be non-decreasing in P(match)
MONOTONE_INCREASING = (
    "name_char_cos",
    "name_word_cos",
    "name_tsort",
    "name_tset",
    "name_jw",
    "name_ratio",
    "name_wjaccard",
    "addr_char_cos",
    "addr_word_cos",
    "addr_tset",
    "addr_tsort",
    "addr_wjaccard",
    "postal",
    "house",
    "country",
)


def _col(df: pd.DataFrame, name: str, idx: IntArr) -> npt.NDArray[Any]:
    return df[name].to_numpy()[idx]


def _eq(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> F32:
    out = np.fromiter((x == y for x, y in zip(a, b, strict=True)), dtype=np.float32, count=len(a))
    empty = np.fromiter((not x or not y for x, y in zip(a, b, strict=True)), dtype=bool, count=len(a))
    out[empty] = np.nan
    return out


def _token_pos_eq(a: npt.NDArray[Any], b: npt.NDArray[Any], pos: int) -> F32:
    out = np.full(len(a), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        tx, ty = x.split(), y.split()
        if tx and ty:
            out[i] = float(tx[pos] == ty[pos])
    return out


def name_features(pt: PairTable) -> dict[str, F32]:
    a, b, ia, ib = pt.s1, pt.s23, pt.s1_idx, pt.c_idx
    na, nb = _col(a, "name_core", ia), _col(b, "name_core", ib)
    alt_a, alt_b = _col(a, "name_alt", ia), _col(b, "name_alt", ib)
    toks = TokenSets(a["name_core"].tolist(), b["name_core"].tolist()).features(ia, ib)
    alt = np.fmax(np.fmax(paired(alt_a, nb, "tsort"), paired(na, alt_b, "tsort")), paired(alt_a, alt_b, "tsort"))
    len_a = np.fromiter((len(x) for x in na), np.float32, len(na))
    len_b = np.fromiter((len(x) for x in nb), np.float32, len(nb))
    ntok_a = np.fromiter((len(x.split()) for x in na), np.float32, len(na))
    ntok_b = np.fromiter((len(x.split()) for x in nb), np.float32, len(nb))
    return {
        "name_ratio": paired(na, nb, "ratio"),
        "name_partial": paired(na, nb, "partial"),
        "name_tset": paired(na, nb, "tset"),
        "name_lev": paired(na, nb, "lev"),
        "name_norm_tsort": paired(_col(a, "name_norm", ia), _col(b, "name_norm", ib), "tsort"),
        "name_skel_ratio": paired(_col(a, "name_skel", ia), _col(b, "name_skel", ib), "ratio"),
        "name_skel_tset": paired(_col(a, "name_skel", ia), _col(b, "name_skel", ib), "tset"),
        "name_core_exact": _eq(na, nb),
        "name_skel_exact": _eq(_col(a, "name_skel", ia), _col(b, "name_skel", ib)),
        "name_acronym": acronym_match(na, _col(a, "name_acronym", ia), nb, _col(b, "name_acronym", ib)),
        "name_alt_best": alt,
        "name_legal_eq": tri_state(_col(a, "name_legal", ia), _col(b, "name_legal", ib)),
        "name_first_tok_eq": _token_pos_eq(na, nb, 0),
        "name_last_tok_eq": _token_pos_eq(na, nb, -1),
        "name_jaccard": toks["jaccard"],
        "name_wjaccard": toks["wjaccard"],
        "name_containment": toks["containment"],
        "name_only_s1": toks["only_left"],
        "name_only_cand": toks["only_right"],
        "name_len_diff": np.abs(len_a - len_b),
        "name_ntok_diff": np.abs(ntok_a - ntok_b),
        "name_len_min": np.minimum(len_a, len_b),
    }


def address_features(pt: PairTable) -> dict[str, F32]:
    a, b, ia, ib = pt.s1, pt.s23, pt.s1_idx, pt.c_idx
    aa, ab = _col(a, "addr_core", ia), _col(b, "addr_core", ib)
    toks = TokenSets(a["addr_core"].tolist(), b["addr_core"].tolist()).features(ia, ib)
    nums = TokenSets(a["nums"].tolist(), b["nums"].tolist()).features(ia, ib)
    len_a = np.fromiter((len(x.split()) for x in aa), np.float32, len(aa))
    len_b = np.fromiter((len(x.split()) for x in ab), np.float32, len(ab))
    return {
        "addr_tsort": paired(aa, ab, "tsort"),
        "addr_ratio": paired(aa, ab, "ratio"),
        "addr_partial": paired(aa, ab, "partial"),
        "addr_jw": paired(aa, ab, "jw"),
        "addr_full_tset": paired(_col(a, "addr_norm", ia), _col(b, "addr_norm", ib), "tset"),
        "landmark_tset": paired(_col(a, "addr_landmark", ia), _col(b, "addr_landmark", ib), "tset"),
        "addr_jaccard": toks["jaccard"],
        "addr_wjaccard": toks["wjaccard"],
        "addr_containment": toks["containment"],
        "addr_only_s1": toks["only_left"],
        "addr_only_cand": toks["only_right"],
        "addr_ntok_s1": len_a,
        "addr_ntok_cand": len_b,
        "postal_prefix3": prefix_match(_col(a, "postal", ia), _col(b, "postal", ib), 3),
        "nums_containment": nums["containment"],
        "nums_common": nums["common"],
        "nums_only_s1": nums["only_left"],
        "nums_only_cand": nums["only_right"],
    }


def frequency_features(pt: PairTable) -> dict[str, F32]:
    s1_counts = pt.s1["name_core"].value_counts()
    c_counts = pt.s23["name_core"].value_counts()
    n1 = pt.s1["name_core"].map(s1_counts).to_numpy(np.float32)
    n23_of_s1 = pt.s1["name_core"].map(c_counts).fillna(0).to_numpy(np.float32)
    nc = pt.s23["name_core"].map(c_counts).to_numpy(np.float32)
    return {
        "name_freq_s1": np.log1p(n1[pt.s1_idx]),
        "name_freq_s1_in_cands": np.log1p(n23_of_s1[pt.s1_idx]),
        "name_freq_cand": np.log1p(nc[pt.c_idx]),
    }


def _group_top2(g: IntArr, v: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-pair (max, second max) of ``v`` within its group ``g`` (second = -inf if singleton group)."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(gs)])
    top = vs[starts]
    second = np.where(sizes > 1, vs[np.minimum(starts + 1, len(vs) - 1)], -np.inf)
    grp_id = np.cumsum(first) - 1
    out_top, out_second = np.empty(len(v)), np.empty(len(v))
    out_top[order] = top[grp_id]
    out_second[order] = second[grp_id]
    return out_top, out_second


def context_features(pt: PairTable, df: pd.DataFrame) -> dict[str, F32]:
    name = df["name_score"].to_numpy(np.float64)
    addr = df["addr_score"].to_numpy(np.float64)
    combo = 0.5 * name + 0.5 * np.nan_to_num(addr, nan=0.5)
    g, c = pt.s1_idx, pt.c_idx
    s1_max, s1_second = _group_top2(g, combo)
    c_max, c_second = _group_top2(c, combo)
    best_other_cand = np.where(combo >= s1_max, s1_second, s1_max)
    best_other_s1 = np.where(combo >= c_max, c_second, c_max)
    n_strong_name = np.bincount(g, weights=(name >= 0.9), minlength=pt.n_s1)
    n_strong_addr = np.bincount(g, weights=(np.nan_to_num(addr) >= 0.9), minlength=pt.n_s1)
    n_comp = np.bincount(c, minlength=len(pt.s23))
    rank_s1 = pd.Series(combo).groupby(g).rank(ascending=False, method="min").to_numpy()
    rank_c = pd.Series(combo).groupby(c).rank(ascending=False, method="min").to_numpy()

    def clip(x: npt.NDArray[np.float64]) -> F32:  # gaps are -inf when there is no competitor
        return np.clip(x, -2.0, 2.0).astype(np.float32)

    return {
        "combo": combo.astype(np.float32),
        "ctx_rank_in_s1": rank_s1.astype(np.float32),
        "ctx_gap_to_s1_best": clip(combo - s1_max),
        "ctx_margin_over_other_cand": clip(combo - best_other_cand),
        "ctx_n_strong_names": n_strong_name[g].astype(np.float32),
        "ctx_n_strong_addrs": n_strong_addr[g].astype(np.float32),
        "ctx_rank_in_cand": rank_c.astype(np.float32),
        "ctx_n_s1_competing": n_comp[c].astype(np.float32),
        "ctx_margin_over_other_s1": clip(combo - best_other_s1),
    }


def build_features(pt: PairTable, ret: RetrievalResult) -> pd.DataFrame:
    base = tfidf_features(pt, ret)
    df = pd.concat([ret.feats.reset_index(drop=True), base], axis=1)
    for block in (name_features(pt), address_features(pt), frequency_features(pt)):
        for k, v in block.items():
            df[k] = v
    df["name_x_addr"] = (df["name_score"] * df["addr_score"]).astype(np.float32)
    df["name_min_addr"] = np.fmin(df["name_score"], df["addr_score"]).astype(np.float32)
    df["addr_missing"] = df["addr_score"].isna().astype(np.float32)
    for k, v in context_features(pt, df).items():
        df[k] = v
    return df.astype(np.float32)
