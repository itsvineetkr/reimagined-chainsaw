"""Method 3 -- supervised pairwise GBDT on retrieval candidates (hard negatives by construction).

Training pairs are exactly the candidates Method 2's retrieval produces for the training S1
records, labelled with the ground truth. Every negative therefore already looks like a match to
at least one retrieval channel (same-name branches, same-building neighbours, shared postal
codes, near-identical names) -- the hard negatives the model must learn to reject -- and train
and inference see the same pair distribution, so probabilities stay calibrated.

Validation: K-fold over S1 entities -> out-of-fold P(match) for every train pair; the decision
policy (calibration, exclusivity, strategy, thresholds) is then fitted on K-1 folds' OOF
probabilities and scored on the remaining fold (nested), so the reported F0.5 is out-of-sample
for both the model and the decision layer.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from er_common.folds import stratified_s1_folds
from er_common.io import SplitData
from er_common.metrics import GroupedEvaluator
from er_common.pairs import PairTable
from er_common.progress import progress
from method2_tfidf_retrieval.method import build_retrieval_pairs
from method2_tfidf_retrieval.retrieval import RetrievalConfig
from method3_gbdt import model as gbdt
from method3_gbdt.features import build_features
from method3_gbdt.model import GbdtParams
from method3_gbdt.policy import Policy, apply_policy, fit_policy

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]
IntArr = npt.NDArray[np.int64]
GT = Mapping[str, Sequence[str]]

log = logging.getLogger("er.method3")

# decision-layer variants reported in the ablation table: (exclusivity modes, strategies)
ABLATIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "threshold_tuned": (("none",), ("threshold",)),
    "threshold_tuned+exclusivity": (("hard", "soft"), ("threshold",)),
    "expected_f": (("none",), ("expected_f",)),
    "expected_f+soft_exclusivity": (("soft",), ("expected_f",)),
    "auto (selected)": (("none", "hard", "soft"), ("expected_f", "threshold")),
}


@dataclass(frozen=True)
class Method3Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gbdt: GbdtParams = field(default_factory=GbdtParams)
    inner_folds: int = 5
    loco: bool = True
    seed: int = 42


@dataclass
class FittedGbdt:
    booster: lgb.Booster
    policy: Policy
    features: list[str]
    n_rounds: int
    oof_policy_f05: float
    importance: dict[str, float]


def _folds_for(pt: PairTable, gt: GT, active_s1: BoolArr, n_folds: int, seed: int) -> IntArr:
    """Stratified S1-level fold ids (-1 for inactive S1). Identical to the runner's validation
    folds when all S1 are active, so ``predict`` alone and ``all`` produce the same model."""
    ids = [s for s, a in zip(pt.s1_ids, active_s1, strict=True) if a]
    countries = pt.s1["country_norm"].to_numpy()[active_s1].tolist()
    fmap = stratified_s1_folds(ids, countries, gt, n_folds, seed)
    return np.array([fmap.get(s, -1) for s in pt.s1_ids], dtype=np.int64)


class GbdtMethod:
    name = "method3_gbdt"

    def __init__(self, config: Method3Config | None = None):
        self.config = config or Method3Config()
        self._oof_cache: dict[str, Any] | None = None

    # -------------------------------------------------------------------------------- pairs
    def build_pairs(self, data: SplitData) -> PairTable:
        pt, ret = build_retrieval_pairs(data, self.config.retrieval)
        t0 = time.perf_counter()
        pt.feats = build_features(pt, ret)
        log.info("features: %d pairs x %d columns [%.1fs]", pt.n_pairs, pt.feats.shape[1], time.perf_counter() - t0)
        return pt

    # -------------------------------------------------------------------------------- helpers
    def _params(self) -> GbdtParams:
        p = self.config.gbdt
        return GbdtParams(**p) if isinstance(p, dict) else p

    def _oof(self, pt: PairTable, label: BoolArr, folds: IntArr, n_folds: int) -> tuple[FloatArr, list[int]]:
        oof = np.full(pt.n_pairs, np.nan)
        iters: list[int] = []
        pair_fold = folds[pt.s1_idx]
        for k in progress(range(n_folds), "out-of-fold models", n_folds, "fold"):
            t0 = time.perf_counter()
            tr = (pair_fold != k) & (pair_fold >= 0)
            te = pair_fold == k
            booster, it = gbdt.train(
                pt.feats.loc[tr], label[tr], pt.s1_idx[tr], self._params(), seed=self.config.seed + k
            )
            oof[te] = gbdt.predict(booster, pt.feats.loc[te])
            iters.append(it)
            log.info("  oof fold %d/%d: %d rounds [%.1fs]", k + 1, n_folds, it, time.perf_counter() - t0)
        return oof, iters

    # -------------------------------------------------------------------------------- validation
    def cross_validate(
        self, pt: PairTable, gt: GT, folds: IntArr, n_folds: int
    ) -> tuple[BoolArr, FloatArr, list[dict[str, Any]]]:
        label = pt.labels(gt)
        n_true = pt.n_true(gt)
        oof, iters = self._oof(pt, label, folds, n_folds)
        self._oof_cache = {"pt": id(pt), "folds": folds.tobytes(), "oof": oof, "iters": iters}
        pair_fold = folds[pt.s1_idx]

        info: list[dict[str, Any]] = []
        results: dict[str, float] = {}
        final_pred = np.zeros(pt.n_pairs, dtype=bool)
        final_q = np.zeros(pt.n_pairs)
        for variant, (excl, strat) in progress(ABLATIONS.items(), "decision-layer ablation", len(ABLATIONS), "variant"):
            pred = np.zeros(pt.n_pairs, dtype=bool)
            q_all = np.zeros(pt.n_pairs)
            for k in range(n_folds):
                policy, _ = fit_policy(oof, pt, label, n_true, folds != k, excl, strat)
                mask, q = apply_policy(policy, oof, pt)
                sel = pair_fold == k
                pred[sel], q_all[sel] = mask[sel], q[sel]
                if variant.startswith("auto"):
                    info.append({"fold": k, "rounds": iters[k], "policy": policy.as_dict()})
            results[variant] = GroupedEvaluator(pt.s1_idx, label, n_true, np.ones(pt.n_s1, bool)).score(pred)
            if variant.startswith("auto"):
                final_pred, final_q = pred, q_all
        raw_half = GroupedEvaluator(pt.s1_idx, label, n_true, np.ones(pt.n_s1, bool)).score(oof >= 0.5)
        ablation = {"raw_p>=0.5": round(raw_half, 4), **{k: round(v, 4) for k, v in results.items()}}
        log.info("decision-layer ablation (nested CV F0.5): %s", ablation)
        info.append({"decision_ablation_nested_cv_f05": ablation})

        auc_like = float(np.mean(oof[label])) - float(np.mean(oof[~label])) if label.any() else 0.0
        info.append({"oof_mean_p_pos_minus_neg": round(auc_like, 4), "oof_rounds": iters})
        if self.config.loco:
            info.append({"leave_one_country_out": self._loco(pt, label, n_true, oof, final_pred)})
        return final_pred, final_q, info

    def _loco(self, pt: PairTable, label: BoolArr, n_true: IntArr, oof: FloatArr, cv_pred: BoolArr) -> dict[str, Any]:
        """Train without one country, score it: a proxy for the unseen-country (France) shift."""
        countries = pt.s1["country_norm"].to_numpy()
        out: dict[str, Any] = {}
        uniq = [c for c in sorted(set(countries)) if (countries == c).sum() >= 50]
        if len(uniq) < 2:
            return {"skipped": "fewer than two countries with >= 50 S1 records"}
        for c in progress(uniq, "leave-one-country-out", len(uniq), "country"):
            held = countries == c
            tr = ~held[pt.s1_idx]
            booster, _ = gbdt.train(pt.feats.loc[tr], label[tr], pt.s1_idx[tr], self._params(), seed=self.config.seed)
            p = oof.copy()
            p[~tr] = gbdt.predict(booster, pt.feats.loc[~tr])
            policy, _ = fit_policy(oof, pt, label, n_true, ~held)
            mask, _ = apply_policy(policy, p, pt)
            ev = GroupedEvaluator(pt.s1_idx, label, n_true, held)
            out[c] = {
                "f05_trained_without": round(ev.score(mask), 4),
                "f05_in_distribution_cv": round(ev.score(cv_pred), 4),
            }
            log.info(
                "LOCO %s: F0.5 %.4f trained without it vs %.4f in-distribution CV",
                c,
                out[c]["f05_trained_without"],
                out[c]["f05_in_distribution_cv"],
            )
        return out

    # -------------------------------------------------------------------------------- fit / predict
    def fit(self, pt: PairTable, gt: GT, active_s1: BoolArr) -> FittedGbdt:
        label = pt.labels(gt)
        n_true = pt.n_true(gt)
        folds = _folds_for(pt, gt, active_s1, self.config.inner_folds, self.config.seed)
        cache = self._oof_cache
        if cache is not None and cache["pt"] == id(pt) and cache["folds"] == folds.tobytes():
            oof, iters = cache["oof"], cache["iters"]
            log.info("reusing out-of-fold predictions from validation (same folds)")
        else:
            oof, iters = self._oof(pt, label, folds, self.config.inner_folds)
        policy, score = fit_policy(np.nan_to_num(oof, nan=0.0), pt, label, n_true, active_s1)
        n_rounds = int(np.median(iters) / (1.0 - self._params().es_fraction))
        tr = active_s1[pt.s1_idx]
        booster, _ = gbdt.train(
            pt.feats.loc[tr], label[tr], pt.s1_idx[tr], self._params(), seed=self.config.seed, n_rounds=n_rounds
        )
        gain = booster.feature_importance(importance_type="gain")
        total = float(gain.sum()) or 1.0
        importance = {
            f: round(float(g) / total, 4)
            for f, g in sorted(zip(booster.feature_name(), gain, strict=True), key=lambda x: -x[1])
        }
        log.info("final model: %d rounds; policy %s (OOF F0.5 %.4f)", n_rounds, policy.as_dict(), score)
        return FittedGbdt(booster, policy, list(pt.feats.columns), n_rounds, score, importance)

    def predict(self, model: FittedGbdt, pt: PairTable) -> tuple[BoolArr, FloatArr]:
        missing = [f for f in model.features if f not in pt.feats.columns]
        if missing:
            raise ValueError(f"feature mismatch between train and inference: {missing[:5]}")
        raw = gbdt.predict(model.booster, pt.feats.loc[:, model.features])
        return apply_policy(model.policy, raw, pt)

    def model_summary(self, model: FittedGbdt) -> dict[str, Any]:
        top = dict(list(model.importance.items())[:25])
        return {
            "n_rounds": model.n_rounds,
            "policy": model.policy.as_dict(),
            "oof_policy_f05": round(model.oof_policy_f05, 4),
            "top_feature_gain_share": top,
            "gbdt_params": asdict(self._params()),
        }
