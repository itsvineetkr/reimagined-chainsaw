"""Shared command-line driver: cross-validated evaluation on train + inference on test.

    python -m <method_package> validate --data-dir dataset --out-dir outputs/<method>
    python -m <method_package> predict  --data-dir dataset --out-dir outputs/<method>
    python -m <method_package> all      --data-dir dataset --out-dir outputs/<method>

``validate`` scores every train S1 entity with a model that never saw its labels (K-fold over
S1 entities, all tuning nested inside the fold), then writes metrics and an error dump.
``predict`` fits on all train labels and writes the two submission files for the test split.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm

from er_common.folds import stratified_s1_folds
from er_common.io import (
    SplitData,
    candidate_owner_multiplicity,
    gt_for,
    load_ground_truth,
    load_split,
    write_id_lists,
    write_submission,
)
from er_common.metrics import candidate_recall, evaluate
from er_common.pairs import PairTable
from er_common.progress import progress, set_enabled

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
GT = Mapping[str, Sequence[str]]

log = logging.getLogger("er")


class Method(Protocol):
    name: str
    config: Any

    def build_pairs(self, data: SplitData) -> PairTable: ...

    def fit(self, pt: PairTable, gt: GT, active_s1: BoolArr) -> Any: ...

    def predict(self, model: Any, pt: PairTable) -> tuple[BoolArr, FloatArr]: ...

    def model_summary(self, model: Any) -> dict[str, Any]: ...


def generic_cross_validate(
    method: Method, pt: PairTable, gt: GT, folds: IntArr, n_folds: int
) -> tuple[BoolArr, FloatArr, list[dict[str, Any]]]:
    pred = np.zeros(pt.n_pairs, dtype=bool)
    score = np.zeros(pt.n_pairs, dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for k in progress(range(n_folds), "cv folds", n_folds, "fold"):
        t0 = time.perf_counter()
        model = method.fit(pt, gt, folds != k)
        mask, sc = method.predict(model, pt)
        in_fold = folds[pt.s1_idx] == k
        pred[in_fold] = mask[in_fold]
        score[in_fold] = sc[in_fold]
        summaries.append({"fold": k, "seconds": round(time.perf_counter() - t0, 2), **method.model_summary(model)})
        log.info("fold %d/%d done in %.1fs", k + 1, n_folds, time.perf_counter() - t0)
    return pred, score, summaries


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "lightgbm"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _manifest(method: Method, data_dir: Path, split: str, extra: dict[str, Any]) -> dict[str, Any]:
    files = sorted((data_dir / split).glob(f"{split}_*.tsv"))
    return {
        "method": method.name,
        "config": (
            dataclasses.asdict(method.config)
            if dataclasses.is_dataclass(method.config) and not isinstance(method.config, type)
            else {}
        ),
        "git_commit": _git_commit(),
        "versions": _versions(),
        "platform": platform.platform(),
        "data": {f.name: _sha256(f) for f in files},
        **extra,
    }


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")


def _error_dump(pt: PairTable, gt: GT, pred: BoolArr, score: FloatArr, path: Path) -> None:
    label = pt.labels(gt)
    s1_ids = pt.s1["entity_id"].to_numpy()
    rows = []
    for kind, sel in (("false_positive", pred & ~label), ("false_negative", ~pred & label)):
        idx = np.flatnonzero(sel)
        for i in progress(idx, f"error dump ({kind})", len(idx), "pair"):
            a, b = pt.s1.iloc[pt.s1_idx[i]], pt.s23.iloc[pt.c_idx[i]]
            rows.append(
                {
                    "kind": kind,
                    "score": round(float(score[i]), 4),
                    "s1_id": a["entity_id"],
                    "cand_id": b["entity_id"],
                    "s1_name": a["business_name"],
                    "cand_name": b["business_name"],
                    "s1_address": a["business_address"],
                    "cand_address": b["business_address"],
                    "s1_country": a["country"],
                    "cand_country": b["country"],
                }
            )
    retrieved = {(s1_ids[i], pt.c_ids[j]) for i, j in zip(pt.s1_idx, pt.c_idx, strict=True)}
    c_lookup = pt.s23.set_index("entity_id")
    s1_lookup = pt.s1.set_index("entity_id")
    for s1, cs in progress(gt.items(), "error dump (blocking misses)", len(gt), "S1"):
        for c in cs:
            if (s1, c) not in retrieved:
                a, b = s1_lookup.loc[s1], c_lookup.loc[c]
                rows.append(
                    {
                        "kind": "missed_by_blocking",
                        "score": float("nan"),
                        "s1_id": s1,
                        "cand_id": c,
                        "s1_name": a["business_name"],
                        "cand_name": b["business_name"],
                        "s1_address": a["business_address"],
                        "cand_address": b["business_address"],
                        "s1_country": a["country"],
                        "cand_country": b["country"],
                    }
                )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["kind", "score"], ascending=[True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _per_country(pt: PairTable, pred: Mapping[str, Sequence[str]], gt: GT) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for country, grp in pt.s1.groupby("country_norm", sort=True):
        ids = grp["entity_id"].tolist()
        rep = evaluate(pred, gt, ids)
        out[str(country) or "<empty>"] = {
            "n": len(ids),
            "f05": round(rep.f05, 4),
            "singleton_acc": round(rep.singleton_accuracy, 4),
            "pair_precision": round(rep.pair_precision, 4),
            "pair_recall": round(rep.pair_recall, 4),
        }
    return out


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------


def load_train(method: Method, data_dir: Path) -> tuple[PairTable, dict[str, tuple[str, ...]]]:
    t0 = time.perf_counter()
    train = load_split(data_dir, "train")
    if train.gt is None:
        raise FileNotFoundError(f"{data_dir}/train/train_ground_truth.tsv is required")
    gt = gt_for(train.gt, train.s1_ids)
    log.info(
        "train: %d S1, %d S2+S3 records; max S1 owners per S2/S3 record in GT = %d",
        len(train.s1),
        len(train.s23),
        candidate_owner_multiplicity(gt),
    )
    pt = method.build_pairs(train)
    log.info("train pairs built in %.1fs", time.perf_counter() - t0)
    return pt, gt


def run_validate(
    method: Method,
    data_dir: Path,
    out_dir: Path,
    n_folds: int,
    seed: int,
    train: tuple[PairTable, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    pt, gt = train or load_train(method, data_dir)
    t_pairs = time.perf_counter() - t0
    blocking = candidate_recall(pt.id_lists(), gt, pt.s1_ids)
    log.info(
        "candidates: %d pairs (%.1f / S1)  pair recall %.4f  oracle F0.5 ceiling %.4f  [%.1fs]",
        pt.n_pairs,
        pt.n_pairs / max(pt.n_s1, 1),
        blocking["pair_recall"],
        blocking["oracle_f05_ceiling"],
        t_pairs,
    )

    folds_map = stratified_s1_folds(pt.s1_ids, pt.s1["country_norm"].tolist(), gt, n_folds, seed)
    folds = np.array([folds_map[s] for s in pt.s1_ids], dtype=np.int64)
    cv = getattr(method, "cross_validate", None)
    if callable(cv):
        pred_mask, score, fold_info = cv(pt, gt, folds, n_folds)
    else:
        pred_mask, score, fold_info = generic_cross_validate(method, pt, gt, folds, n_folds)

    pred = pt.id_lists(pred_mask, score)
    report = evaluate(pred, gt, pt.s1_ids)
    per_fold = []
    for k in range(n_folds):
        ids = [s for s in pt.s1_ids if folds_map[s] == k]
        per_fold.append(round(evaluate(pred, gt, ids).f05, 4))
    log.info("CV %s", report.summary())
    log.info("per-fold F0.5: %s  (std %.4f)", per_fold, float(np.std(per_fold)))

    vdir = out_dir / "validation"
    write_id_lists(vdir / "oof_matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), pred, pt.s1_ids)
    write_id_lists(
        vdir / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), pt.id_lists(), pt.s1_ids
    )
    _error_dump(pt, gt, pred_mask, score, vdir / "errors.tsv")
    result = {
        "cv": report.as_dict(),
        "per_fold_f05": per_fold,
        "per_country": _per_country(pt, pred, gt),
        "blocking": blocking,
        "n_pairs": pt.n_pairs,
        "folds": fold_info,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _json_dump(_manifest(method, data_dir, "train", {"validation": result}), vdir / "report.json")
    return result


def run_predict(
    method: Method,
    data_dir: Path,
    out_dir: Path,
    test_gt: Path | None,
    train: tuple[PairTable, dict[str, tuple[str, ...]]] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    pt_train, gt = train or load_train(method, data_dir)
    model = method.fit(pt_train, gt, np.ones(pt_train.n_s1, dtype=bool))
    log.info("fitted on all train labels: %s", json.dumps(method.model_summary(model), default=str)[:400])

    test = load_split(data_dir, "test")
    pt = method.build_pairs(test)
    mask, score = method.predict(model, pt)
    matches = pt.id_lists(mask, score)
    candidates = pt.id_lists(None, score)
    tdir = out_dir / "test"
    write_submission(tdir, matches, candidates, pt.s1_ids, pt.s23["entity_id"].tolist())
    n_nonempty = sum(1 for v in matches.values() if v)
    log.info(
        "test: %d S1, %d candidate pairs, %d predicted pairs, %d S1 with >=1 match -> %s",
        pt.n_s1,
        pt.n_pairs,
        int(mask.sum()),
        n_nonempty,
        tdir,
    )
    extra: dict[str, Any] = {
        "test": {
            "n_s1": pt.n_s1,
            "n_candidate_pairs": pt.n_pairs,
            "n_pred_pairs": int(mask.sum()),
            "n_s1_with_match": n_nonempty,
            "countries": pt.s1["country_norm"].value_counts().to_dict(),
        },
        "model": method.model_summary(model),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if test_gt is not None and test_gt.is_file():
        tgt = gt_for(load_ground_truth(test_gt), pt.s1_ids)
        rep = evaluate(matches, tgt, pt.s1_ids)
        extra["test_scored_against"] = str(test_gt)
        extra["test_eval"] = rep.as_dict()
        extra["test_per_country"] = _per_country(pt, matches, tgt)
        extra["test_blocking"] = candidate_recall(candidates, tgt, pt.s1_ids)
        log.info("TEST %s", rep.summary())
        log.info("TEST per country: %s", extra["test_per_country"])
    _json_dump(_manifest(method, data_dir, "test", extra), tdir / "manifest.json")
    return extra


def main(method: Method, argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog=f"python -m {method.name}", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=("validate", "predict", "all"))
    ap.add_argument("--data-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None, help="overrides config.seed")
    ap.add_argument("--config", type=Path, default=None, help="JSON file with config overrides")
    ap.add_argument(
        "--test-gt", type=Path, default=None, help="optional ground truth for the test split (synthetic data only)"
    )
    ap.add_argument("--no-progress", action="store_true", help="disable progress bars (e.g. when logging to a file)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    set_enabled(not args.no_progress)
    if args.config is not None:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        method.config = dataclasses.replace(method.config, **overrides)
    if args.seed is not None:
        method.config = dataclasses.replace(method.config, seed=args.seed)
    out_dir = args.out_dir or Path("outputs") / method.name
    log.info("%s config: %s", method.name, method.config)

    with logging_redirect_tqdm():  # log lines print above the bars instead of breaking them
        train = load_train(method, args.data_dir)
        if args.command in ("validate", "all"):
            run_validate(method, args.data_dir, out_dir, args.folds, method.config.seed, train)
        if args.command in ("predict", "all"):
            run_predict(method, args.data_dir, out_dir, args.test_gt, train)
