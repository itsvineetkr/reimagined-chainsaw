"""Method 1 -- multi-field fuzzy matching + tuned rules."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from er_common.io import SplitData
from er_common.metrics import GroupedEvaluator
from er_common.normalize import normalize_records
from er_common.pairs import PairTable
from er_common.tuning import search
from method1_fuzzy_rules.blocking import BlockingConfig, block
from method1_fuzzy_rules.rules import SPACE, RuleEngine, RuleParams, fuzzy_features

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]

log = logging.getLogger("er.method1")


@dataclass(frozen=True)
class Method1Config:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    n_random: int = 600
    n_rounds: int = 3
    seed: int = 42


class FuzzyRulesMethod:
    name = "method1_fuzzy_rules"

    def __init__(self, config: Method1Config | None = None):
        self.config = config or Method1Config()

    def build_pairs(self, data: SplitData) -> PairTable:
        s1 = normalize_records(data.s1)
        s23 = normalize_records(data.s23)
        cfg = self.config.blocking
        if isinstance(cfg, dict):  # JSON override
            cfg = BlockingConfig(**cfg)
        ii, jj, n_keys = block(s1, s23, cfg)
        pt = PairTable(s1=s1, s23=s23, s1_idx=ii, c_idx=jj)
        feats = fuzzy_features(pt)
        feats["n_shared_keys"] = n_keys.astype(np.float32)
        pt.feats = feats
        return pt

    def fit(self, pt: PairTable, gt: Mapping[str, Sequence[str]], active_s1: BoolArr) -> RuleParams:
        engine = RuleEngine(pt)
        evaluator = GroupedEvaluator(pt.s1_idx, pt.labels(gt), pt.n_true(gt), active_s1)

        def objective(p: dict[str, float]) -> float:
            mask, _ = engine.decide(RuleParams(**p))
            return evaluator.score(mask)

        best, score = search(
            objective,
            SPACE,
            RuleParams().as_dict(),
            n_random=self.config.n_random,
            n_rounds=self.config.n_rounds,
            seed=self.config.seed,
        )
        log.info("tuned rules: in-sample F0.5 %.4f", score)
        return RuleParams(**best)

    def predict(self, model: RuleParams, pt: PairTable) -> tuple[BoolArr, FloatArr]:
        return RuleEngine(pt).decide(model)

    def model_summary(self, model: RuleParams) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in dataclasses.asdict(model).items()}
