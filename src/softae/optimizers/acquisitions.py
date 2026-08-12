"""Acquisition strategies for Bayesian optimization (pool-based and continuous).

Two families:

* **optimization** — drive toward the objective optimum: :class:`UcbAcquisition`,
  :class:`EiAcquisition`.
* **active_learning** — reduce surrogate uncertainty: :class:`MaxVarianceAcquisition`
  (MacKay), :class:`IntegratedVarianceAcquisition` (ALC surrogate), and
  :class:`UncertaintyWeightedAcquisition`.

Strategies score every candidate; the optimizer selects the argmax.  The
active-learning strategies work in **epistemic** (model) variance, subtracting
the known **aleatoric** observation noise (the candidate's ``alpha``) so they
reduce reducible uncertainty rather than chasing intrinsically noisy points.

A strategy reads what it needs from the optimizer via a small, duck-typed
interface: ``_objective`` ("maximize"/"minimize"), ``_kappa``, ``history``,
``candidate_variance`` (per-candidate aleatoric variance aligned to ``cand_X``),
and ``backend`` (for cross-covariance).  Both
:class:`~softae.optimizers.bayesian.BayesianOptimizer` (continuous) and
:class:`~softae.optimizers.pooled_bayesian.PooledBayesianOptimizer` (pool-based)
satisfy this interface.
"""

from __future__ import annotations

import abc

import numpy as np
from scipy.stats import norm

from softae.errors import CampaignError


class AcquisitionStrategy(abc.ABC):
    """Scores candidates; higher score ⇒ more desirable (argmax selected)."""

    name: str = "acquisition"
    family: str = "optimization"  # "optimization" | "active_learning"

    @abc.abstractmethod
    def score(
        self,
        optimizer,
        cand_X: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        history: list[tuple[dict, float]],
    ) -> np.ndarray:
        """Return a score per candidate row of ``cand_X``."""

    # ── shared helpers ───────────────────────────────────────────────────

    @staticmethod
    def _sign(optimizer) -> float:
        return 1.0 if getattr(optimizer, "_objective", "maximize") == "maximize" else -1.0

    @staticmethod
    def _candidate_variance(optimizer, n: int) -> np.ndarray:
        var = getattr(optimizer, "candidate_variance", None)
        if var is None:
            return np.zeros(n, dtype=float)
        var = np.asarray(var, dtype=float)
        return var if var.shape == (n,) else np.zeros(n, dtype=float)

    @classmethod
    def _epistemic_sigma(cls, optimizer, sigma: np.ndarray) -> np.ndarray:
        """sqrt(max(sigma² − aleatoric_var, 0)): reducible uncertainty only."""
        alpha = cls._candidate_variance(optimizer, sigma.shape[0])
        return np.sqrt(np.clip(sigma ** 2 - alpha, 0.0, None))


# ── optimization family ──────────────────────────────────────────────────

class UcbAcquisition(AcquisitionStrategy):
    """Upper Confidence Bound: ``sign·mu + kappa·sigma``."""

    name = "ucb"
    family = "optimization"

    def score(self, optimizer, cand_X, mu, sigma, history):
        kappa = getattr(optimizer, "_kappa", 2.0)
        return self._sign(optimizer) * mu + kappa * sigma


class EiAcquisition(AcquisitionStrategy):
    """Expected Improvement over the best observed value."""

    name = "ei"
    family = "optimization"

    def score(self, optimizer, cand_X, mu, sigma, history):
        if not history:
            return sigma  # nothing observed yet → pure exploration
        sign = self._sign(optimizer)
        values = [v for _, v in history]
        best_y = max(values) if sign > 0 else min(values)
        improvement = sign * (mu - best_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.where(sigma > 1e-12, improvement / sigma, 0.0)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return np.where(sigma > 1e-12, ei, 0.0)


# ── active-learning family ─────────────────────────────────────────────────

class MaxVarianceAcquisition(AcquisitionStrategy):
    """MacKay (ALM): select the point of maximum epistemic posterior variance."""

    name = "max_variance"
    family = "active_learning"

    def score(self, optimizer, cand_X, mu, sigma, history):
        return self._epistemic_sigma(optimizer, sigma)


class IntegratedVarianceAcquisition(AcquisitionStrategy):
    """ALC surrogate: favour points whose covariance reduces variance broadly.

    ``score_j = Σ_k k(x_j, x_k)² / (sigma_j² + alpha_j)`` over remaining
    candidates, using the fitted kernel.  Falls back to :class:`MaxVarianceAcquisition`
    (with a one-time warning) when the backend exposes no cross-covariance.
    """

    name = "integrated_variance"
    family = "active_learning"

    def __init__(self) -> None:
        self._warned = False

    def score(self, optimizer, cand_X, mu, sigma, history):
        backend = getattr(optimizer, "backend", None)
        K = backend.cross_cov(cand_X, cand_X) if backend is not None else None
        if K is None:
            if not self._warned:
                import structlog

                structlog.get_logger(__name__).warning(
                    "integrated_variance: backend has no cross_cov; "
                    "falling back to max_variance"
                )
                self._warned = True
            return MaxVarianceAcquisition().score(optimizer, cand_X, mu, sigma, history)
        alpha = self._candidate_variance(optimizer, sigma.shape[0])
        denom = sigma ** 2 + alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.sum(K ** 2, axis=1) / np.where(denom > 1e-12, denom, 1e-12)
        return scores


class UncertaintyWeightedAcquisition(AcquisitionStrategy):
    """Explore by epistemic variance, but down-weight intrinsically noisy points.

    ``score = sigma_epistemic · w``, where ``w = 1 / (1 + alpha / alpha_ref)``
    (``alpha_ref`` = median candidate variance) penalises candidates the dataset
    measures noisily — so budget is not spent re-measuring high-noise formulations.
    """

    name = "uncertainty_weighted"
    family = "active_learning"

    def score(self, optimizer, cand_X, mu, sigma, history):
        alpha = self._candidate_variance(optimizer, sigma.shape[0])
        ref = np.median(alpha[alpha > 0]) if np.any(alpha > 0) else 1.0
        w = 1.0 / (1.0 + alpha / ref)
        return self._epistemic_sigma(optimizer, sigma) * w


#: Registry mapping config strings → strategy classes.
ACQUISITIONS: dict[str, type[AcquisitionStrategy]] = {
    UcbAcquisition.name: UcbAcquisition,
    EiAcquisition.name: EiAcquisition,
    MaxVarianceAcquisition.name: MaxVarianceAcquisition,
    IntegratedVarianceAcquisition.name: IntegratedVarianceAcquisition,
    UncertaintyWeightedAcquisition.name: UncertaintyWeightedAcquisition,
}


def make_acquisition(acquisition: str | AcquisitionStrategy) -> AcquisitionStrategy:
    """Resolve an acquisition name or instance to a strategy."""
    if isinstance(acquisition, AcquisitionStrategy):
        return acquisition
    if acquisition not in ACQUISITIONS:
        raise CampaignError(
            f"unknown acquisition '{acquisition}'; available: {sorted(ACQUISITIONS)}"
        )
    return ACQUISITIONS[acquisition]()
