"""Method 2 candidate generation: multi-channel retrieval + reciprocal-rank fusion.

Channels (each run separately against S2 and against S3, so a source with many near-duplicate
records cannot crowd the other one out of the top-k):

    name_char  char_wb 2-4-gram TF-IDF cosine on the core name (+ trade name)
    addr_char  char_wb 3-4-gram TF-IDF cosine on the core address
    bm25       Okapi BM25 over word tokens of name + address + postal code
    postal     exact postal-code block, ranked by name_char cosine inside the block

Fusion: RRF(pair) = sum_channels 1 / (rrf_k + rank); the ``k_final`` best pairs per
(S1, source) survive. The per-channel ranks / scores are kept as features for later methods.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from er_common.progress import progress, stages
from er_common.similarity import TfidfSpace, sparse_topk

IntArr = npt.NDArray[np.int64]
F32 = npt.NDArray[np.float32]

CHANNELS = ("name_char", "addr_char", "bm25", "postal")


@dataclass(frozen=True)
class RetrievalConfig:
    k_name: int = 15
    k_addr: int = 15
    k_bm25: int = 15
    k_postal: int = 10
    k_final: int = 25
    rrf_k: float = 20.0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


def name_text(df: pd.DataFrame) -> list[str]:
    core = df["name_core"].to_numpy()
    alt = df["name_alt"].to_numpy()
    return [f"{c} {a}".strip() for c, a in zip(core, alt, strict=True)]


def bm25_text(df: pd.DataFrame) -> list[str]:
    return [f"{n} {a} {p}".strip() for n, a, p in zip(df["name_core"], df["addr_core"], df["postal"], strict=True)]


class BM25:
    def __init__(self, k1: float, b: float):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32)

    def fit_transform(self, queries: list[str], docs: list[str]) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        self.vec.fit(queries + docs)
        tf = self.vec.transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) if n_docs else 1.0
        norm = self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1e-6))
        w = tf.copy()
        rows = np.repeat(np.arange(n_docs), np.diff(w.indptr))
        w.data = idf[w.indices] * w.data * (self.k1 + 1) / (w.data + norm[rows])
        q = self.vec.transform(queries).tocsr().astype(np.float32)
        q.data[:] = 1.0
        return q, w.tocsr()


@dataclass
class RetrievalResult:
    s1_idx: IntArr
    c_idx: IntArr
    feats: pd.DataFrame
    spaces: dict[str, tuple[sp.csr_matrix, sp.csr_matrix]] = field(default_factory=dict)


def _postal_channel(
    s1: pd.DataFrame, s23: pd.DataFrame, src_rows: IntArr, q: sp.csr_matrix, d: sp.csr_matrix, k: int
) -> tuple[list[int], list[int], list[float]]:
    blocks: dict[str, list[int]] = defaultdict(list)
    postal23 = s23["postal"].to_numpy()
    for j in src_rows:
        if postal23[j]:
            blocks[postal23[j]].append(int(j))
    out_i: list[int] = []
    out_j: list[int] = []
    out_r: list[float] = []
    codes = s1["postal"].to_numpy()
    for i, code in enumerate(progress(codes, "postal block", len(codes), "rec")):
        members = blocks.get(code) if code else None
        if not members:
            continue
        m = np.asarray(members, dtype=np.int64)
        sims = np.asarray((d[m] @ q[i].T).todense()).ravel()
        order = np.lexsort((m, -sims))[:k]
        out_i += [i] * len(order)
        out_j += m[order].tolist()
        out_r += list(range(1, len(order) + 1))
    return out_i, out_j, out_r


def retrieve(s1: pd.DataFrame, s23: pd.DataFrame, cfg: RetrievalConfig) -> RetrievalResult:
    with stages("fit retrieval spaces", 3) as step:
        step("name char TF-IDF")
        name_q, name_d = TfidfSpace("char_wb", (2, 4)).fit_transform(name_text(s1), name_text(s23))
        step("address char TF-IDF")
        addr_q, addr_d = TfidfSpace("char_wb", (3, 4)).fit_transform(
            s1["addr_core"].tolist(), s23["addr_core"].tolist()
        )
        step("BM25")
        bm_q, bm_d = BM25(cfg.bm25_k1, cfg.bm25_b).fit_transform(bm25_text(s1), bm25_text(s23))
    channels = {
        "name_char": (name_q, name_d, cfg.k_name),
        "addr_char": (addr_q, addr_d, cfg.k_addr),
        "bm25": (bm_q, bm_d, cfg.k_bm25),
    }
    source = s23["source"].to_numpy()
    rows: list[pd.DataFrame] = []
    sources = sorted(set(source))
    for src, (ch, (q, d, k)) in progress(
        [(src, item) for src in sources for item in channels.items()], "retrieval channels", len(sources) * 3, "channel"
    ):
        src_rows = np.flatnonzero(source == src).astype(np.int64)
        idx, sc = sparse_topk(q, d[src_rows], k)
        n_q, kk = idx.shape
        ok = sc.ravel() > 0
        rows.append(
            pd.DataFrame(
                {
                    "s1_idx": np.repeat(np.arange(n_q), kk)[ok],
                    "c_idx": src_rows[idx.ravel()][ok],
                    "channel": ch,
                    "rank": np.tile(np.arange(1, kk + 1), n_q)[ok],
                    "score": sc.ravel()[ok],
                }
            )
        )
    for src in sources:
        src_rows = np.flatnonzero(source == src).astype(np.int64)
        pi, pj, pr = _postal_channel(s1, s23, src_rows, name_q, name_d, cfg.k_postal)
        rows.append(
            pd.DataFrame(
                {
                    "s1_idx": pi,
                    "c_idx": pj,
                    "channel": "postal",
                    "rank": pr,
                    "score": np.ones(len(pi), dtype=np.float32),
                }
            )
        )

    hits = pd.concat(rows, ignore_index=True)
    hits["rrf"] = 1.0 / (cfg.rrf_k + hits["rank"])
    wide_rank = hits.pivot_table(index=["s1_idx", "c_idx"], columns="channel", values="rank", aggfunc="min")
    wide_score = hits.pivot_table(index=["s1_idx", "c_idx"], columns="channel", values="score", aggfunc="max")
    fused = hits.groupby(["s1_idx", "c_idx"], sort=True)["rrf"].sum()
    wide = pd.DataFrame(index=fused.index)
    for ch in CHANNELS:
        wide[f"rank_{ch}"] = wide_rank[ch] if ch in wide_rank else np.nan
        if ch != "postal":
            wide[f"ret_{ch}"] = wide_score[ch] if ch in wide_score else np.nan
    wide["rrf"] = fused
    wide["n_channels"] = wide[[f"rank_{c}" for c in CHANNELS]].notna().sum(axis=1)
    wide = wide.reset_index()
    wide["is_s3"] = (source[wide["c_idx"].to_numpy()] == "S3").astype(np.int64)

    if cfg.k_final > 0:
        wide = wide.sort_values(["s1_idx", "is_s3", "rrf", "c_idx"], ascending=[True, True, False, True])
        wide["rank_fused"] = wide.groupby(["s1_idx", "is_s3"]).cumcount() + 1
        wide = wide[wide["rank_fused"] <= cfg.k_final]
    else:
        wide["rank_fused"] = wide.groupby(["s1_idx", "is_s3"])["rrf"].rank(ascending=False, method="first")
    wide = wide.sort_values(["s1_idx", "c_idx"]).reset_index(drop=True)

    kmax = {"name_char": cfg.k_name, "addr_char": cfg.k_addr, "bm25": cfg.k_bm25, "postal": cfg.k_postal}
    for ch in CHANNELS:
        wide[f"rank_{ch}"] = wide[f"rank_{ch}"].fillna(kmax[ch] + 1)
    feats = wide.drop(columns=["s1_idx", "c_idx"]).astype(np.float32)
    return RetrievalResult(
        s1_idx=wide["s1_idx"].to_numpy(np.int64),
        c_idx=wide["c_idx"].to_numpy(np.int64),
        feats=feats.reset_index(drop=True),
        spaces={"name_char": (name_q, name_d), "addr_char": (addr_q, addr_d), "bm25": (bm_q, bm_d)},
    )
