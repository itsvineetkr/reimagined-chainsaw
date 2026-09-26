"""LightGBM training with S1-grouped early stopping and deterministic settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from method3_gbdt.features import MONOTONE_INCREASING

IntArr = npt.NDArray[np.int64]
BoolArr = npt.NDArray[np.bool_]
FloatArr = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GbdtParams:
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_rounds: int = 3000
    early_stopping: int = 150
    es_fraction: float = 0.15
    monotone: bool = True
    num_threads: int = 4


def lgb_params(p: GbdtParams, seed: int, features: Sequence[str]) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": p.learning_rate,
        "num_leaves": p.num_leaves,
        "min_child_samples": p.min_child_samples,
        "feature_fraction": p.feature_fraction,
        "bagging_fraction": p.bagging_fraction,
        "bagging_freq": p.bagging_freq,
        "lambda_l2": p.lambda_l2,
        "num_threads": p.num_threads,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    if p.monotone:
        params["monotone_constraints"] = [1 if f in MONOTONE_INCREASING else 0 for f in features]
        params["monotone_constraints_method"] = "advanced"
    return params


def _es_split(groups: IntArr, fraction: float, seed: int) -> BoolArr:
    """Boolean mask of rows whose S1 group is held out for early stopping."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    held = rng.choice(uniq, size=max(1, round(fraction * len(uniq))), replace=False)
    return np.isin(groups, held)


def train(
    x: pd.DataFrame,
    y: BoolArr,
    groups: IntArr,
    p: GbdtParams,
    seed: int,
    n_rounds: int | None = None,
) -> tuple[lgb.Booster, int]:
    """With ``n_rounds=None``: early-stop on a held-out slice of S1 groups, return (model, best_iter).
    Otherwise train on everything for exactly ``n_rounds``."""
    features = list(x.columns)
    params = lgb_params(p, seed, features)
    if n_rounds is None:
        es = _es_split(groups, p.es_fraction, seed)
        dtrain = lgb.Dataset(x.loc[~es], label=y[~es].astype(int), feature_name=features, free_raw_data=True)
        dvalid = lgb.Dataset(x.loc[es], label=y[es].astype(int), reference=dtrain)
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=p.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(p.early_stopping, verbose=False)],
        )
        return booster, int(booster.best_iteration or p.max_rounds)
    dtrain = lgb.Dataset(x, label=y.astype(int), feature_name=features, free_raw_data=True)
    booster = lgb.train(params, dtrain, num_boost_round=max(1, n_rounds))
    return booster, n_rounds


def predict(booster: lgb.Booster, x: pd.DataFrame) -> FloatArr:
    it = booster.best_iteration if booster.best_iteration and booster.best_iteration > 0 else None
    return np.asarray(booster.predict(x, num_iteration=it), dtype=np.float64)
