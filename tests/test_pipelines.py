"""End-to-end smoke test of all three methods on a small synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from er_common.io import load_split, read_tsv
from er_common.synth import write_dataset


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("data"), n_train=300, n_test=150, seed=11)


def _cfg(tmp: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("module", "overrides"),
    [
        ("method1_fuzzy_rules", {"n_random": 40, "n_rounds": 1}),
        ("method2_tfidf_retrieval", {"n_random": 40, "n_rounds": 1}),
        ("method3_gbdt", {"inner_folds": 3, "loco": True}),
    ],
)
def test_method_end_to_end(module: str, overrides: dict[str, object], data_dir: Path, tmp_path: Path) -> None:
    main = __import__(f"{module}.__main__", fromlist=["main"])
    method_cls = {
        "method1_fuzzy_rules": "FuzzyRulesMethod",
        "method2_tfidf_retrieval": "TfidfRetrievalMethod",
        "method3_gbdt": "GbdtMethod",
    }[module]
    method = getattr(__import__(f"{module}.method", fromlist=[method_cls]), method_cls)()
    out = tmp_path / module
    main.main(
        method,
        [
            "all",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out),
            "--folds",
            "3",
            "--config",
            str(_cfg(tmp_path, module, overrides)),
            "--test-gt",
            str(data_dir / "test" / "_hidden_test_ground_truth.tsv"),
        ],
    )

    test = load_split(data_dir, "test")
    matches = read_tsv(out / "test" / "matching_results.tsv")
    cands = read_tsv(out / "test" / "candidate_pairs.tsv")
    assert list(matches.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(cands.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert matches["source1_entity_id"].tolist() == test.s1_ids
    assert cands["source1_entity_id"].tolist() == test.s1_ids
    valid = set(test.s23["entity_id"])
    for m, c in zip(matches["matched_entity_ids"], cands["candidate_entity_ids"], strict=True):
        m_ids = [x for x in m.split(",") if x]
        c_ids = [x for x in c.split(",") if x]
        assert set(m_ids) <= set(c_ids) <= valid
        assert len(set(m_ids)) == len(m_ids)
    assert "fr" in set(test.s1["country"].str.lower().str[:2])  # unseen country present in test

    report = json.loads((out / "validation" / "report.json").read_text(encoding="utf-8"))
    assert 0.5 < report["validation"]["cv"]["f05"] <= 1.0
    manifest = json.loads((out / "test" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_eval"]["f05"] > 0.5


def test_method3_predict_alone_matches_all(data_dir: Path, tmp_path: Path) -> None:
    """``predict`` on its own must rebuild the same folds / OOF / policy as ``all``."""
    from er_common.runner import main
    from method3_gbdt.method import GbdtMethod

    cfg = _cfg(tmp_path, "m3", {"inner_folds": 3, "loco": False})
    common = ["--data-dir", str(data_dir), "--folds", "3", "--config", str(cfg)]
    main(GbdtMethod(), ["all", "--out-dir", str(tmp_path / "a"), *common])
    main(GbdtMethod(), ["predict", "--out-dir", str(tmp_path / "b"), *common])
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (tmp_path / "a" / "test" / f).read_bytes() == (tmp_path / "b" / "test" / f).read_bytes()
