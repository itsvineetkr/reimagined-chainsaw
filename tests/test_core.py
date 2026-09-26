from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from er_common.decision import (
    exclusive_mask,
    expected_fbeta_best_k,
    expected_fbeta_topk,
    group_rank,
    threshold_margin,
)
from er_common.folds import stratified_s1_folds
from er_common.io import parse_id_list, read_tsv, validate_submission, write_id_lists
from er_common.metrics import GroupedEvaluator, evaluate, f_beta_counts
from er_common.normalize import normalize_address, normalize_country, normalize_name, skeleton
from er_common.similarity import TokenSets, sparse_topk
from er_common.tuning import Dim, search
from method3_gbdt.policy import apply_exclusivity

# ---------------------------------------------------------------------------------------- metric


def test_fbeta_examples_from_challenge() -> None:
    assert f_beta_counts(2, 2, 2) == pytest.approx(1.0)
    assert f_beta_counts(2, 3, 2) == pytest.approx(0.714, abs=1e-3)


def test_fbeta_singletons_and_empty_predictions() -> None:
    assert f_beta_counts(0, 0, 0) == 1.0
    assert f_beta_counts(0, 1, 0) == 0.0
    assert f_beta_counts(0, 0, 3) == 0.0


def test_evaluate_macro_average() -> None:
    gt = {"S1-1": ["S2-1", "S3-1"], "S1-2": [], "S1-3": ["S2-9"]}
    pred = {"S1-1": ["S2-1", "S2-2", "S3-1"], "S1-2": [], "S1-3": []}
    rep = evaluate(pred, gt, ["S1-1", "S1-2", "S1-3"])
    assert rep.f05 == pytest.approx((2.5 / 3.5 + 1.0 + 0.0) / 3)
    assert rep.singleton_accuracy == 1.0
    assert rep.false_merge_entities == 1


def test_grouped_evaluator_matches_reference() -> None:
    rng = np.random.default_rng(0)
    group = np.repeat(np.arange(20), 5)
    label = rng.random(100) < 0.2
    n_true = np.bincount(group, weights=label, minlength=20).astype(int) + (rng.random(20) < 0.2)
    mask = rng.random(100) < 0.3
    ev = GroupedEvaluator(group, label, n_true, np.ones(20, bool))
    ref = []
    for g in range(20):
        sel = group == g
        ref.append(f_beta_counts(int((mask & label & sel).sum()), int((mask & sel).sum()), int(n_true[g])))
    assert ev.score(mask) == pytest.approx(np.mean(ref))


# ---------------------------------------------------------------------------------------- decision


def _brute_expected_f(p: np.ndarray, k: int, beta: float = 0.5) -> float:
    total = 0.0
    for labels in itertools.product([0, 1], repeat=len(p)):
        y = np.array(labels)
        prob = float(np.prod(np.where(y == 1, p, 1 - p)))
        total += prob * f_beta_counts(int(y[:k].sum()), k, int(y.sum()), beta)
    return total


@pytest.mark.parametrize("seed", range(5))
def test_expected_fbeta_matches_bruteforce(seed: int) -> None:
    p = np.sort(np.random.default_rng(seed).random(6))[::-1]
    k, e = expected_fbeta_best_k(p)
    brute = [_brute_expected_f(p, kk) for kk in range(len(p) + 1)]
    assert e == pytest.approx(max(brute), abs=1e-9)
    assert brute[k] == pytest.approx(max(brute), abs=1e-9)


def test_expected_fbeta_singleton_threshold() -> None:
    assert expected_fbeta_best_k(np.array([0.55]))[0] == 1
    assert expected_fbeta_best_k(np.array([0.45]))[0] == 0


def test_expected_fbeta_topk_mask() -> None:
    group = np.array([0, 0, 0, 1, 1])
    prob = np.array([0.97, 0.94, 0.05, 0.3, 0.2])
    mask = expected_fbeta_topk(group, prob, 2)
    assert mask.tolist() == [True, True, False, False, False]


def test_exclusive_mask_keeps_best_owner() -> None:
    cand = np.array([7, 7, 8, 8, 9])
    score = np.array([0.9, 0.8, 0.1, 0.4, 0.5])
    assert exclusive_mask(cand, score).tolist() == [True, False, False, True, True]


def test_soft_exclusivity() -> None:
    cand = np.array([1, 1, 2])
    q = apply_exclusivity(np.array([0.9, 0.9, 0.7]), cand, "soft")
    assert q[0] == pytest.approx(9 / 19)
    assert q[2] == pytest.approx(0.7)


def test_threshold_margin_and_rank() -> None:
    group = np.array([0, 0, 0, 1])
    score = np.array([0.9, 0.85, 0.5, 0.3])
    assert group_rank(group, score).tolist() == [0, 1, 2, 0]
    mask = threshold_margin(group, score, 2, tau=0.4, delta=0.1)
    assert mask.tolist() == [True, True, False, False]
    assert threshold_margin(group, score, 2, tau=0.4, delta=1.0, max_k=1).tolist() == [True, False, False, False]


# ---------------------------------------------------------------------------------------- normalisation


def test_name_normalisation_examples() -> None:
    assert normalize_name("ABC Pvt. Ltd.").norm == "abc pvt ltd"
    assert normalize_name("ABC Pvt. Ltd.").core == "abc"
    assert normalize_name("Shree Balaji Hospital").core == normalize_name("Sri Balaji Hosp.").core
    assert normalize_name("Saint Mary's Medical Center").core == normalize_name("St Marys Medical Centre").core
    assert normalize_name("S.B.I. Life").core.startswith("sbi")
    dba = normalize_name("Joe's Pizza LLC dba Joe's Place")
    assert dba.core == "joes pizza" and dba.alt == "joes place"


def test_address_normalisation_examples() -> None:
    assert normalize_address("12, MG Rd., Lucknow").norm == "12 mg rd lucknow"
    a = normalize_address("12 Mahatma Gandhi Road, Lucknow, Uttar Pradesh 226 001")
    assert a.postal == "226001" and a.house == "12" and "mg rd" in a.core
    b = normalize_address("Shop No. 4, Near SBI ATM, Sector 15, Gurgaon")
    assert b.landmark == "near sbi atm" and "gurugram" in b.core and "sbi" not in b.core
    assert normalize_address("10250 Santa Monica Blvd, Los Angeles, CA").postal == ""
    assert normalize_address("33005, 50 Impasse Voltaire, Bordeaux").postal == "33005"
    assert normalize_address("Near Clock Tower Mumbai").core == "mumbai"


def test_country_is_open_set() -> None:
    assert normalize_country("USA") == normalize_country("United States") == "us"
    assert normalize_country("France") == "fr"
    assert normalize_country("Germany") == "germany"


def test_skeleton_transliteration() -> None:
    assert skeleton("shree") == skeleton("sri")
    assert skeleton("mohammed") == skeleton("muhammad")


# ---------------------------------------------------------------------------------------- similarity / io / folds


def test_sparse_topk_matches_dense() -> None:
    rng = np.random.default_rng(1)
    q = sp.csr_matrix(rng.random((7, 12)) * (rng.random((7, 12)) < 0.4))
    d = sp.csr_matrix(rng.random((30, 12)) * (rng.random((30, 12)) < 0.4))
    _, sc = sparse_topk(q, d, 5, max_cells=40)
    dense = (q @ d.T).toarray()
    for i in range(7):
        np.testing.assert_allclose(sc[i], np.sort(dense[i])[::-1][:5], rtol=1e-5)


def test_token_sets() -> None:
    ts = TokenSets(["a b c", ""], ["a b", "a"])
    f = ts.features(np.array([0, 1]), np.array([0, 1]))
    assert f["jaccard"][0] == pytest.approx(2 / 3)
    assert np.isnan(f["jaccard"][1])


def test_tsv_roundtrip_keeps_na_strings(tmp_path: Path) -> None:
    path = tmp_path / "x.tsv"
    path.write_text('entity_id\tbusiness_name\nS1-1\tNA\nS1-2\tJoe\'s "Best" Pizza\n', encoding="utf-8")
    df = read_tsv(path)
    assert df["business_name"].tolist()[0] == "NA"
    assert len(df) == 2


def test_write_and_validate_outputs(tmp_path: Path) -> None:
    rows = {"S1-1": ["S2-1", "S3-4"], "S1-2": []}
    write_id_lists(tmp_path / "m.tsv", ("source1_entity_id", "matched_entity_ids"), rows, ["S1-1", "S1-2"])
    text = (tmp_path / "m.tsv").read_text(encoding="utf-8").splitlines()
    assert text == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-1,S3-4", "S1-2\t"]
    errs = validate_submission({"S1-1": ["S2-1", "S2-1"]}, {"S1-1": ["S2-1"]}, ["S1-1"], ["S2-1"])
    assert any("duplicate" in e for e in errs)
    errs = validate_submission({"S1-1": ["S2-2"]}, {"S1-1": ["S2-1"]}, ["S1-1"], ["S2-1", "S2-2"])
    assert any("not in candidate" in e for e in errs)
    assert parse_id_list(" S2-1, S2-1 ,,S3-2") == ("S2-1", "S3-2")


def test_folds_are_deterministic_and_balanced() -> None:
    ids = [f"S1-{i}" for i in range(100)]
    countries = ["us" if i % 2 else "in" for i in range(100)]
    gt = {s: (["x"] if i % 3 else []) for i, s in enumerate(ids)}
    a = stratified_s1_folds(ids, countries, gt, 5, seed=3)
    assert a == stratified_s1_folds(ids, countries, gt, 5, seed=3)
    assert sorted(np.bincount(list(a.values())).tolist()) == [20] * 5


def test_search_improves_objective() -> None:
    space = {"x": Dim(-2, 2), "y": Dim(-2, 2)}
    best, score = search(
        lambda p: -((p["x"] - 0.7) ** 2) - (p["y"] + 0.3) ** 2, space, {"x": 0, "y": 0}, n_random=50, seed=0
    )
    assert score > -0.01
    assert best["x"] == pytest.approx(0.7, abs=0.1)
