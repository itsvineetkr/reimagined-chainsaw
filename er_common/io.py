"""Data loading, output writing and submission-format validation.

All challenge files are TSV. Two traps handled here:
  * pandas turns strings such as "NA", "None", "null" into NaN by default -> a business
    literally named "NA" would silently vanish. We read everything as ``str`` with
    ``keep_default_na=False``.
  * Business names can contain double quotes. The organisers' snippet uses pandas' default
    (QUOTE_MINIMAL) parsing, so we use it too, but verify the parsed row count against the raw
    line count and fall back to ``QUOTE_NONE`` if a stray quote swallowed lines.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


@dataclass(frozen=True)
class SplitData:
    """One split (train or test). ``s23`` is S2 and S3 stacked, with a ``source`` column."""

    split: str
    s1: pd.DataFrame
    s23: pd.DataFrame
    gt: dict[str, tuple[str, ...]] | None

    @property
    def s1_ids(self) -> list[str]:
        return self.s1["entity_id"].tolist()


def _count_data_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        n = sum(1 for line in fh if line.strip("\r\n"))
    return max(n - 1, 0)


def read_tsv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _count_data_lines(path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")
    if len(df) != expected:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
        )
    if len(df) != expected:
        raise ValueError(f"{path}: parsed {len(df)} rows but file has {expected} data lines")
    return df.rename(columns=lambda c: str(c).strip())


def _load_records(path: Path, prefix: str) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)}")
    df = df.loc[:, list(RECORD_COLUMNS)].copy()
    for col in RECORD_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    if df["entity_id"].duplicated().any():
        dups = df.loc[df["entity_id"].duplicated(), "entity_id"].head(5).tolist()
        raise ValueError(f"{path}: duplicate entity_id values, e.g. {dups}")
    bad = df.loc[~df["entity_id"].str.startswith(prefix), "entity_id"].head(5).tolist()
    if bad:
        raise ValueError(f"{path}: entity_id without expected prefix {prefix!r}: {bad}")
    return df.reset_index(drop=True)


def parse_id_list(value: str) -> tuple[str, ...]:
    """Comma-separated ID list -> de-duplicated tuple preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in value.split(","):
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return tuple(out)


def load_ground_truth(path: str | Path) -> dict[str, tuple[str, ...]]:
    df = read_tsv(path)
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    gt: dict[str, tuple[str, ...]] = {}
    for s1_id, ids in zip(df["source1_entity_id"], df["matched_entity_ids"], strict=True):
        gt[s1_id.strip()] = parse_id_list(ids)
    return gt


def load_split(data_dir: str | Path, split: str) -> SplitData:
    """Load ``<data_dir>/<split>/<split>_source{1,2,3}.tsv`` (+ ground truth for train)."""
    base = Path(data_dir) / split
    s1 = _load_records(base / f"{split}_source1.tsv", "S1-")
    s2 = _load_records(base / f"{split}_source2.tsv", "S2-")
    s3 = _load_records(base / f"{split}_source3.tsv", "S3-")
    s2["source"] = "S2"
    s3["source"] = "S3"
    s23 = pd.concat([s2, s3], ignore_index=True)
    s1["source"] = "S1"

    gt_path = base / f"{split}_ground_truth.tsv"
    gt = load_ground_truth(gt_path) if gt_path.is_file() else None
    if gt is not None:
        _check_gt(gt, s1, s23, gt_path)
    return SplitData(split=split, s1=s1, s23=s23, gt=gt)


def _check_gt(gt: Mapping[str, Sequence[str]], s1: pd.DataFrame, s23: pd.DataFrame, path: Path) -> None:
    s1_ids = set(s1["entity_id"])
    s23_ids = set(s23["entity_id"])
    unknown_s1 = [k for k in gt if k not in s1_ids]
    unknown_c = sorted({c for v in gt.values() for c in v if c not in s23_ids})
    if unknown_s1 or unknown_c:
        raise ValueError(f"{path}: ground truth references unknown ids: s1={unknown_s1[:5]} s2/s3={unknown_c[:5]}")


def gt_for(gt: Mapping[str, Sequence[str]], s1_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """S1 entities absent from the GT file are singletons (empty match set)."""
    return {s: tuple(gt.get(s, ())) for s in s1_ids}


def candidate_owner_multiplicity(gt: Mapping[str, Sequence[str]]) -> int:
    """Max number of S1 entities a single S2/S3 record is matched to in the GT.

    1 means every S2/S3 record belongs to at most one S1 entity -> the exclusivity constraint
    used by the decision layer is consistent with the labels.
    """
    counts: dict[str, int] = {}
    for ids in gt.values():
        for c in ids:
            counts[c] = counts.get(c, 0) + 1
    return max(counts.values(), default=0)


def write_id_lists(
    path: str | Path, header: Sequence[str], rows: Mapping[str, Sequence[str]], order: Sequence[str]
) -> None:
    """Write ``id<TAB>comma,list`` rows in ``order``; entities without a list get an empty cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(header) + "\n")
        for s1_id in order:
            ids = rows.get(s1_id, ())
            fh.write(f"{s1_id}\t{','.join(ids)}\n")


def validate_submission(
    matches: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
    s1_ids: Sequence[str],
    s23_ids: Iterable[str],
) -> list[str]:
    """Return a list of rule violations (empty list == valid)."""
    errors: list[str] = []
    valid_c = set(s23_ids)
    s1_set = set(s1_ids)
    if len(s1_set) != len(s1_ids):
        errors.append("duplicate Source 1 ids in output order")
    for name, table in (("matching_results", matches), ("candidate_pairs", candidates)):
        extra = set(table) - s1_set
        if extra:
            errors.append(f"{name}: rows for unknown S1 ids, e.g. {sorted(extra)[:3]}")
        for s1_id, ids in table.items():
            if len(set(ids)) != len(ids):
                errors.append(f"{name}: duplicate ids for {s1_id}")
            bad = [c for c in ids if c not in valid_c or not c.startswith(("S2-", "S3-"))]
            if bad:
                errors.append(f"{name}: invalid ids for {s1_id}: {bad[:3]}")
    for s1_id, ids in matches.items():
        missing = set(ids) - set(candidates.get(s1_id, ()))
        if missing:
            errors.append(f"match not in candidate list for {s1_id}: {sorted(missing)[:3]}")
    return errors


def write_submission(
    out_dir: str | Path,
    matches: Mapping[str, Sequence[str]],
    candidates: Mapping[str, Sequence[str]],
    s1_ids: Sequence[str],
    s23_ids: Iterable[str],
) -> None:
    errors = validate_submission(matches, candidates, s1_ids, s23_ids)
    if errors:
        raise ValueError("submission violates challenge constraints:\n  " + "\n  ".join(errors[:20]))
    out = Path(out_dir)
    write_id_lists(out / "matching_results.tsv", MATCH_HEADER, matches, s1_ids)
    write_id_lists(out / "candidate_pairs.tsv", CANDIDATE_HEADER, candidates, s1_ids)
