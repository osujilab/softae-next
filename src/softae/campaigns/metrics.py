"""Per-step metrics and convergence/stopping rules for campaigns.

Metrics track two notions of "converged":

* **optimization** — how close the best-found objective is to the dataset's true
  optimum (simple/cumulative regret, best-so-far).
* **model accuracy** — how well the surrogate predicts the *whole* pool
  (RMSE over the pool, calibration coverage).

Conventions: ``s = +1`` for maximize, ``−1`` for minimize; ``y*`` is the true
optimum objective value; ``y_best_t`` is the best revealed value through step
``t``.  ``simple_regret_t = s·(y* − y_best_t) ≥ 0``.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class StepMetrics:
    """Metrics recorded after one campaign step."""

    iteration: int
    sampled_value: float          # objective value revealed this step
    best_so_far: float            # best revealed value through this step
    simple_regret: float          # s·(y* − best_so_far), ≥ 0
    cumulative_regret: float      # running sum of per-step instantaneous regret
    pool_fraction: float          # n_sampled / pool_size
    surrogate_rmse: float = float("nan")   # RMSE of mu over full pool (nan if no model)
    coverage: float = float("nan")          # frac. of pool within mu ± 1.96σ (nan if no model)


def best_so_far(values: list[float], maximize: bool) -> float:
    """Best objective among revealed *values* (ignoring NaNs)."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return max(finite) if maximize else min(finite)


def surrogate_pool_metrics(
    mu: np.ndarray, sigma: np.ndarray, y_true: np.ndarray
) -> tuple[float, float]:
    """RMSE and 95% coverage of surrogate predictions over the pool."""
    valid = np.isfinite(y_true) & np.isfinite(mu)
    if not np.any(valid):
        return float("nan"), float("nan")
    err = mu[valid] - y_true[valid]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    lo = mu[valid] - 1.96 * sigma[valid]
    hi = mu[valid] + 1.96 * sigma[valid]
    coverage = float(np.mean((y_true[valid] >= lo) & (y_true[valid] <= hi)))
    return rmse, coverage


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------

class StoppingRule(abc.ABC):
    """Decides, from the metrics history, whether a campaign has converged."""

    @abc.abstractmethod
    def should_stop(self, history: list[StepMetrics]) -> bool: ...


class OptimizationStoppingRule(StoppingRule):
    """Stop when best-found is within tolerance of the true optimum *and* plateaus.

    Fires when both hold at the latest step:

    * ``simple_regret ≤ max(abs_tol, rel_tol · |y*|)`` — close enough to optimum;
    * ``best_so_far`` has varied by ≤ that same tolerance over the last
      ``patience`` steps — and not improved — i.e. it has plateaued.
    """

    def __init__(
        self,
        y_optimum: float,
        *,
        rel_tol: float = 1e-2,
        abs_tol: float = 0.0,
        patience: int = 5,
    ) -> None:
        self.y_optimum = float(y_optimum)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.patience = int(patience)

    @property
    def tolerance(self) -> float:
        return max(self.abs_tol, self.rel_tol * abs(self.y_optimum))

    def should_stop(self, history: list[StepMetrics]) -> bool:
        if not history:
            return False
        tol = self.tolerance
        if history[-1].simple_regret > tol:
            return False
        if len(history) < self.patience:
            return False
        recent = [m.best_so_far for m in history[-self.patience:]]
        recent = [v for v in recent if math.isfinite(v)]
        if len(recent) < self.patience:
            return False
        return (max(recent) - min(recent)) <= tol


class ModelAccuracyStoppingRule(StoppingRule):
    """Stop when surrogate RMSE over the pool is low and calibration is good."""

    def __init__(self, *, rmse_tol: float = 0.25, coverage_tol: float = 0.9) -> None:
        self.rmse_tol = float(rmse_tol)
        self.coverage_tol = float(coverage_tol)

    def should_stop(self, history: list[StepMetrics]) -> bool:
        if not history:
            return False
        m = history[-1]
        if not (math.isfinite(m.surrogate_rmse) and math.isfinite(m.coverage)):
            return False
        return m.surrogate_rmse <= self.rmse_tol and m.coverage >= self.coverage_tol
