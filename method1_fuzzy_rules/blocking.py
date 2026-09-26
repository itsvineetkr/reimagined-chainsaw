"""Method 1 blocking: exact-key blocks (no retrieval index).

Keys per record
  p:<postal>                 postal / ZIP / PIN code
  pn:<postal>:<initial>      postal + first letter of the name skeleton (sub-block for dense codes)
  t:<skeleton token>         each name token (transliteration-tolerant skeleton)
  b:<tok1>_<tok2>            consecutive name-token bigram (rarer than single tokens)
  a:<acronym>                initials of multi-token names / whole single-token names (SBI)
  h:<house>|<street token>   house number + first alphabetic address token

Keys whose S1 x S2/S3 product exceeds ``max_block_pairs`` are skipped (too generic); an S1
record left without any candidate falls back to its smallest block. Surviving pairs are then
cut to the ``max_cands_per_source`` best per S1 per source by a cheap token-set pre-score.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.pairs import unique_pairs
from er_common.similarity import paired

IntArr = npt.NDArray[np.int64]


@dataclass(frozen=True)
class BlockingConfig:
    max_block_pairs: int = 4000
    max_cands_per_source: int = 25
    prescore_name_weight: float = 0.6


KEY_COLUMNS = ("name_skel", "name_core", "name_acronym", "postal", "house", "addr_core")


def record_keys(name_skel: str, name_core: str, acronym: str, postal: str, house: str, addr_core: str) -> list[str]:
    keys: list[str] = []
    skel = name_skel.split()
    if postal:
        keys.append(f"p:{postal}")
        if skel:
            keys.append(f"pn:{postal}:{skel[0][:1]}")
    keys += [f"t:{t}" for t in skel if len(t) >= 2]
    keys += [f"b:{a}_{b}" for a, b in itertools.pairwise(skel)]
    core = name_core.split()
    if acronym:
        keys.append(f"a:{acronym}")
    if len(core) == 1 and len(core[0]) >= 2:
        keys.append(f"a:{core[0]}")
    if house:
        alpha = [t for t in addr_core.split() if t.isalpha() and len(t) > 1]
        if alpha:
            keys.append(f"h:{house}|{alpha[0]}")
    return keys


def _keys(df: pd.DataFrame) -> list[list[str]]:
    return [record_keys(*row) for row in zip(*(df[c].tolist() for c in KEY_COLUMNS), strict=True)]


def block(s1: pd.DataFrame, s23: pd.DataFrame, cfg: BlockingConfig) -> tuple[IntArr, IntArr, IntArr]:
    """Return (s1_idx, c_idx, n_shared_keys) for the retained candidate pairs."""
    left: dict[str, list[int]] = defaultdict(list)
    right: dict[str, list[int]] = defaultdict(list)
    s1_keys = _keys(s1)
    for i, ks in enumerate(s1_keys):
        for k in set(ks):
            left[k].append(i)
    for j, ks in enumerate(_keys(s23)):
        for k in set(ks):
            right[k].append(j)

    parts_i: list[IntArr] = []
    parts_j: list[IntArr] = []
    covered = np.zeros(len(s1), dtype=bool)
    for key in sorted(left):
        li, rj = left[key], right.get(key)
        if not rj or len(li) * len(rj) > cfg.max_block_pairs:
            continue
        a = np.repeat(np.asarray(li, dtype=np.int64), len(rj))
        b = np.tile(np.asarray(rj, dtype=np.int64), len(li))
        parts_i.append(a)
        parts_j.append(b)
        covered[li] = True

    for i in np.flatnonzero(~covered):  # fallback: smallest non-empty block for this record
        sizes = [(len(right[k]), k) for k in set(s1_keys[i]) if k in right]
        if sizes:
            _, k = min(sizes)
            rj = right[k]
            parts_i.append(np.full(len(rj), i, dtype=np.int64))
            parts_j.append(np.asarray(rj, dtype=np.int64))

    if not parts_i:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    all_i, all_j = np.concatenate(parts_i), np.concatenate(parts_j)
    n_c = len(s23)
    code, counts = np.unique(all_i * n_c + all_j, return_counts=True)
    ii, jj = code // n_c, code % n_c

    w = cfg.prescore_name_weight
    name_s = paired(s1["name_core"].to_numpy()[ii], s23["name_core"].to_numpy()[jj], "tset", missing_nan=False)
    addr_s = paired(s1["addr_core"].to_numpy()[ii], s23["addr_core"].to_numpy()[jj], "tset", missing_nan=False)
    pre = w * name_s + (1 - w) * addr_s + 0.01 * np.minimum(counts, 10)

    src = (s23["source"].to_numpy()[jj] == "S3").astype(np.int64)
    order = np.lexsort((jj, -pre, src, ii))
    grp = ii[order] * 2 + src[order]
    starts = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = order[rank < cfg.max_cands_per_source]
    ki, kj = unique_pairs(ii[keep], jj[keep], n_c)
    shared = dict(zip(code.tolist(), counts.tolist(), strict=True))
    n_keys = np.array([shared[int(a) * n_c + int(b)] for a, b in zip(ki, kj, strict=True)], dtype=np.int64)
    return ki, kj, n_keys
