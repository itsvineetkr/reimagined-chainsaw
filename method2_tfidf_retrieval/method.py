"""Method 2 -- TF-IDF / BM25 retrieval + weighted similarity scoring."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from er_common.io import SplitData
from er_common.metrics import GroupedEvaluator
from er_common.normalize import normalize_records
from er_common.pairs import PairTable
from er_common.tuning import search
from method2_tfidf_retrieval.retrieval import RetrievalConfig, RetrievalResult, retrieve
from method2_tfidf_retrieval.scorer import SPACE, ScoreParams, Scorer, tfidf_features

BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]

log = logging.getLogger("er.method2")


@dataclass(frozen=True)
class Method2Config:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    n_random: int = 600
    n_rounds: int = 3
    seed: int = 42


def build_retrieval_pairs(data: SplitData, cfg: RetrievalConfig) -> tuple[PairTable, RetrievalResult]:
    """Normalise + retrieve; shared with Method 3."""
    if isinstance(cfg, dict):  # JSON override
        cfg = RetrievalConfig(**cfg)
    s1 = normalize_records(data.s1)
    s23 = normalize_records(data.s23)
    ret = retrieve(s1, s23, cfg)
    pt = PairTable(s1=s1, s23=s23, s1_idx=ret.s1_idx, c_idx=ret.c_idx)
    return pt, ret


class TfidfRetrievalMethod:
    name = "method2_tfidf_retrieval"

    def __init__(self, config: Method2Config | None = None):
        self.config = config or Method2Config()

    def build_pairs(self, data: SplitData) -> PairTable:
        pt, ret = build_retrieval_pairs(data, self.config.retrieval)
        pt.feats = pd.concat([ret.feats, tfidf_features(pt, ret)], axis=1)
        return pt

    def fit(self, pt: PairTable, gt: Mapping[str, Sequence[str]], active_s1: BoolArr) -> ScoreParams:
        scorer = Scorer(pt)
        evaluator = GroupedEvaluator(pt.s1_idx, pt.labels(gt), pt.n_true(gt), active_s1)

        def objective(p: dict[str, float]) -> float:
            mask, _ = scorer.decide(ScoreParams(**p))
            return evaluator.score(mask)

        best, score = search(
            objective,
            SPACE,
            ScoreParams().as_dict(),
            n_random=self.config.n_random,
            n_rounds=self.config.n_rounds,
            seed=self.config.seed,
        )
        log.info("tuned scorer: in-sample F0.5 %.4f", score)
        return ScoreParams(**best)

    def predict(self, model: ScoreParams, pt: PairTable) -> tuple[BoolArr, FloatArr]:
        return Scorer(pt).decide(model)

    def model_summary(self, model: ScoreParams) -> dict[str, Any]:
        return {k: round(v, 4) for k, v in dataclasses.asdict(model).items()}
