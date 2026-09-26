"""Vectorised pairwise similarity primitives over aligned (left[i], right[i]) string pairs.

Everything returns float32 arrays in [0, 1] (NaN where a field is missing on either side, so
downstream models can distinguish "different" from "unknown").
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

F32 = npt.NDArray[np.float32]
IntArr = npt.NDArray[np.int64]
Texts = Sequence[str] | npt.NDArray[Any]  # lists or pandas-derived object arrays of str

WORKERS = int(os.environ.get("ER_WORKERS", "-1"))

SCORERS: dict[str, tuple[Callable[..., float], float]] = {
    "ratio": (fuzz.ratio, 100.0),
    "partial": (fuzz.partial_ratio, 100.0),
    "tsort": (fuzz.token_sort_ratio, 100.0),
    "tset": (fuzz.token_set_ratio, 100.0),
    "jw": (JaroWinkler.normalized_similarity, 1.0),
    "lev": (Levenshtein.normalized_similarity, 1.0),
}


def paired(left: Texts, right: Texts, scorer: str, missing_nan: bool = True) -> F32:
    fn, scale = SCORERS[scorer]
    out = process.cpdist(list(left), list(right), scorer=fn, workers=WORKERS, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32) / np.float32(scale)
    if missing_nan:
        empty = np.fromiter((not a or not b for a, b in zip(left, right, strict=True)), bool, len(left))
        out[empty] = np.nan
    return out


def rowwise_dot(a: sp.csr_matrix, b: sp.csr_matrix, ia: IntArr, ib: IntArr, chunk: int = 200_000) -> F32:
    """sum_k a[ia[n], k] * b[ib[n], k] for every n, chunked to bound memory."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        e = min(s + chunk, len(ia))
        prod = a[ia[s:e]].multiply(b[ib[s:e]])
        out[s:e] = np.asarray(prod.sum(axis=1)).ravel()
    return out


class TfidfSpace:
    """TF-IDF space fitted on the union of both sides (label-free, so fitting on test text is fine)."""

    def __init__(self, analyzer: str, ngram_range: tuple[int, int], min_df: int = 1):
        extra = {"token_pattern": r"(?u)\b\w+\b"} if analyzer == "word" else {}
        self.vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
            lowercase=False,
            **extra,
        )

    def fit_transform(self, left: Texts, right: Texts) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        self.vec.fit(list(left) + list(right))
        return self.vec.transform(list(left)).tocsr(), self.vec.transform(list(right)).tocsr()


def cosine_pairs(a: sp.csr_matrix, b: sp.csr_matrix, ia: IntArr, ib: IntArr) -> F32:
    """Cosine for L2-normalised TF-IDF rows; NaN where either row is empty."""
    out = rowwise_dot(a, b, ia, ib)
    empty = (np.diff(a.indptr)[ia] == 0) | (np.diff(b.indptr)[ib] == 0)
    out[empty] = np.nan
    return out


class TokenSets:
    """Binary token incidence for two sides + IDF weights -> set similarities as sparse algebra."""

    def __init__(self, left: Texts, right: Texts):
        vocab: dict[str, int] = {}
        rows_l = [self._ids(t, vocab) for t in left]
        rows_r = [self._ids(t, vocab) for t in right]
        self.left = self._matrix(rows_l, len(vocab))
        self.right = self._matrix(rows_r, len(vocab))
        df = np.asarray((self.left > 0).sum(axis=0)).ravel() + np.asarray((self.right > 0).sum(axis=0)).ravel()
        n_docs = len(rows_l) + len(rows_r)
        self.idf = np.log((n_docs + 1) / (df + 1)).astype(np.float32) + np.float32(1.0)
        self.df = df
        self.left_w = (self.left @ sp.diags(self.idf)).tocsr()
        self.right_w = (self.right @ sp.diags(self.idf)).tocsr()
        self.len_l = np.asarray(self.left.sum(axis=1)).ravel().astype(np.float32)
        self.len_r = np.asarray(self.right.sum(axis=1)).ravel().astype(np.float32)
        self.wlen_l = np.asarray(self.left_w.sum(axis=1)).ravel().astype(np.float32)
        self.wlen_r = np.asarray(self.right_w.sum(axis=1)).ravel().astype(np.float32)

    @staticmethod
    def _ids(text: str, vocab: dict[str, int]) -> list[int]:
        return sorted({vocab.setdefault(t, len(vocab)) for t in text.split()})

    @staticmethod
    def _matrix(rows: list[list[int]], n_cols: int) -> sp.csr_matrix:
        indptr = np.r_[0, np.cumsum([len(r) for r in rows])]
        indices = np.fromiter((i for r in rows for i in r), dtype=np.int64, count=int(indptr[-1]))
        data = np.ones(len(indices), dtype=np.float32)
        return sp.csr_matrix((data, indices, indptr), shape=(len(rows), max(n_cols, 1)))

    def features(self, ia: IntArr, ib: IntArr) -> dict[str, F32]:
        inter = rowwise_dot(self.left, self.right, ia, ib)
        winter = rowwise_dot(self.left_w, self.right, ia, ib)
        la, lb = self.len_l[ia], self.len_r[ib]
        wa, wb = self.wlen_l[ia], self.wlen_r[ib]
        union = la + lb - inter
        wunion = wa + wb - winter
        missing = (la == 0) | (lb == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = {
                "jaccard": np.where(missing, np.nan, inter / np.maximum(union, 1)),
                "wjaccard": np.where(missing, np.nan, winter / np.maximum(wunion, 1e-6)),
                "containment": np.where(missing, np.nan, inter / np.maximum(np.minimum(la, lb), 1)),
                "common": inter,
                "only_left": la - inter,
                "only_right": lb - inter,
            }
        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def sparse_topk(queries: sp.csr_matrix, docs: sp.csr_matrix, k: int, max_cells: int = 20_000_000) -> tuple[IntArr, F32]:
    """Exact top-k by dot product for every query row; returns (indices, scores) of shape (n_q, k').

    Chunked dense scoring keeps memory at ``max_cells`` float32 cells regardless of corpus size.
    Rows are ordered by descending score, ties by ascending doc index (deterministic).
    """
    n_q, n_d = queries.shape[0], docs.shape[0]
    k = min(k, n_d)
    idx_out = np.zeros((n_q, k), dtype=np.int64)
    sc_out = np.zeros((n_q, k), dtype=np.float32)
    if k == 0 or n_q == 0:
        return idx_out, sc_out
    docs_t = docs.T.tocsc()
    chunk = max(1, min(n_q, max_cells // max(n_d, 1)))
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        dense = (queries[s:e] @ docs_t).toarray().astype(np.float32, copy=False)
        part = np.argpartition(-dense, kth=k - 1, axis=1)[:, :k] if k < n_d else np.tile(np.arange(n_d), (e - s, 1))
        part_scores = np.take_along_axis(dense, part, axis=1)
        order = np.lexsort((part, -part_scores), axis=1)
        idx_out[s:e] = np.take_along_axis(part, order, axis=1)
        sc_out[s:e] = np.take_along_axis(part_scores, order, axis=1)
    return idx_out, sc_out


def acronym_match(names_a: Texts, acr_a: Texts, names_b: Texts, acr_b: Texts) -> F32:
    """1 if one side's whole core name is a token equal to the other side's initials (SBI vs State Bank of India)."""
    out = np.zeros(len(names_a), dtype=np.float32)
    for i, (na, aa, nb, ab) in enumerate(zip(names_a, acr_a, names_b, acr_b, strict=True)):
        ta, tb = set(na.split()), set(nb.split())
        if (ab and len(ab) >= 2 and ab in ta) or (aa and len(aa) >= 2 and aa in tb):
            out[i] = 1.0
    return out


def tri_state(left: Texts, right: Texts) -> F32:
    """1 equal, 0 different, NaN if missing on either side."""
    out = np.full(len(left), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if a and b:
            out[i] = 1.0 if a == b else 0.0
    return out


def prefix_match(left: Texts, right: Texts, n: int) -> F32:
    out = np.full(len(left), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if a and b:
            out[i] = 1.0 if a[:n] == b[:n] else 0.0
    return out
