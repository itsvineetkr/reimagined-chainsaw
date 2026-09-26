"""Derivative-free parameter search against the (non-differentiable) entity-level F0.5.

Random search over the box, seeded with the hand-set defaults, followed by coordinate
refinement on a shrinking grid. Fully deterministic given ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from er_common.progress import bar

Params = dict[str, float]


@dataclass(frozen=True)
class Dim:
    low: float
    high: float
    integer: bool = False

    def clip(self, x: float) -> float:
        x = min(max(x, self.low), self.high)
        return float(round(x)) if self.integer else float(x)

    def sample(self, rng: np.random.Generator) -> float:
        return self.clip(rng.uniform(self.low, self.high))


def search(
    objective: Callable[[Params], float],
    space: Mapping[str, Dim],
    start: Params,
    n_random: int = 400,
    n_rounds: int = 3,
    grid: int = 9,
    seed: int = 0,
) -> tuple[Params, float]:
    rng = np.random.default_rng(seed)
    best = {k: space[k].clip(v) for k, v in start.items()}
    total = 1 + n_random + n_rounds * len(space) * grid
    with bar("tune parameters", total, "eval") as pb:
        best_score = objective(best)
        pb.update(1)
        for _ in range(n_random):
            cand = {k: d.sample(rng) for k, d in space.items()}
            s = objective(cand)
            if s > best_score + 1e-12:
                best, best_score = cand, s
            pb.update(1)
            pb.set_postfix_str(f"best F0.5={best_score:.4f}", refresh=False)
        width = 0.5
        for _ in range(n_rounds):
            for k in sorted(space):
                d = space[k]
                span = (d.high - d.low) * width
                for v in np.linspace(best[k] - span / 2, best[k] + span / 2, grid):
                    pb.update(1)
                    cand = dict(best)
                    cand[k] = d.clip(float(v))
                    if cand[k] == best[k]:
                        continue
                    s = objective(cand)
                    if s > best_score + 1e-12:
                        best, best_score = cand, s
                        pb.set_postfix_str(f"best F0.5={best_score:.4f}", refresh=False)
            width *= 0.5
    return best, best_score
