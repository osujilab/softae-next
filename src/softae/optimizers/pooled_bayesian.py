"""Pool-based Bayesian optimizer for simulated campaigns.

Unlike :class:`~softae.optimizers.bayesian.BayesianOptimizer`, which proposes
points over a continuous box via random candidates, this optimizer selects the
next point from a **finite, explicit candidate pool** (the dataset's points) and
never repeats one — the textbook pool-based / benchmark BO formulation.  It is:

* **backend-agnostic** — fits through a :class:`~softae.optimizers.surrogates.SurrogateBackend`,
  so sklearn / BoTorch / GPyTorch / GPCAM are swappable.
* **noise-aware** — accepts a per-candidate observation-noise variance
  (``pool_variance``); the variances of *observed* points become the GP ``alpha``
  array (heteroscedastic / fixed-noise GP), and the variances of *candidate*
  points are exposed to the acquisition for epistemic/aleatoric reasoning.
* **acquisition-pluggable** — any :class:`~softae.optimizers.acquisitions.AcquisitionStrategy`.

Encoding is the shared :class:`~softae.optimizers.encoding.OneHotEncoder`:
floats/ints pass through, categoricals are one-hot encoded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from softae.errors import CampaignError, OptimizerError
from softae.optimizers.acquisitions import make_acquisition
from softae.optimizers.base import BaseOptimizer
from softae.optimizers.encoding import OneHotEncoder
from softae.optimizers.surrogates import SurrogateBackend, make_backend


class PooledBayesianOptimizer(BaseOptimizer):
    """GP-based BO restricted to a finite candidate pool.

    Parameters
    ----------
    parameter_space
        Search-space definition (see :class:`BaseOptimizer`).  For pooled runs
        this is typically :meth:`GroundTruthDataset.parameter_space`.
    pool
        List of candidate parameter dicts to choose from.
    pool_variance
        Optional ``{point_key → variance}`` per-candidate observation-noise
        variance.  When omitted, the backend learns a homoscedastic noise level.
    backend
        A :class:`SurrogateBackend` or its registry name (default ``"sklearn"``).
    acquisition
        An :class:`AcquisitionStrategy` or its registry name (default ``"ucb"``).
    n_initial
        Random warm-up draws (from the pool) before the surrogate is used.
    kappa
        UCB exploration weight.
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
        *,
        pool: list[dict[str, Any]],
        pool_variance: dict[tuple, float] | None = None,
        backend: SurrogateBackend | str = "sklearn",
        acquisition=None,
        n_initial: int = 5,
        kappa: float = 2.0,
        use_alpha: bool = True,
    ) -> None:
        super().__init__(parameter_space, objective, seed)
        if not pool:
            raise CampaignError("pool must be a non-empty list of candidate points")
        if n_initial < 1:
            raise OptimizerError("n_initial must be >= 1")

        self._kappa = kappa
        self._n_initial = n_initial
        self._rng = np.random.RandomState(seed)
        self.backend = make_backend(backend, seed=seed)
        self._strategy = make_acquisition(acquisition if acquisition is not None else "ucb")
        self._encoder = OneHotEncoder(self._parameter_space)

        # Pool bookkeeping, keyed by encoded-vector rounded tuple.
        self._pool: list[dict[str, Any]] = [dict(p) for p in pool]
        self._pool_keys: list[tuple] = [self._encoder.key(p) for p in self._pool]
        self._remaining: set[tuple] = set(self._pool_keys)
        self._pool_variance = dict(pool_variance) if pool_variance else None
        #: When False, per-point variances are withheld from the GP ``alpha``
        #: (the GP learns homoscedastic noise) but still reach the acquisition
        #: via ``candidate_variance`` — i.e. noise_channel="acquisition_weight".
        self.use_alpha = use_alpha

        #: Per-candidate aleatoric variance aligned to the last scored candidate
        #: matrix; read by acquisition strategies.  Set during :meth:`suggest`.
        self.candidate_variance: np.ndarray | None = None

    # ── encoding (delegates to the shared OneHotEncoder) ─────────────────

    def _encode(self, params: dict[str, Any]) -> list[float]:
        return self._encoder.encode(params)

    def _key(self, params: dict[str, Any]) -> tuple:
        """Float-robust hashable identity for a pool point."""
        return self._encoder.key(params)

    # ── pool helpers ─────────────────────────────────────────────────────

    def _remaining_points(self) -> list[dict[str, Any]]:
        return [p for p, k in zip(self._pool, self._pool_keys) if k in self._remaining]

    def _alpha_for_history(self) -> np.ndarray | float | None:
        """Per-observation variance array aligned to ``_history`` (or None)."""
        if self._pool_variance is None or not self.use_alpha:
            return None
        return np.array(
            [self._pool_variance.get(self._encoder.key(p), 0.0) for p, _ in self._history],
            dtype=float,
        )

    def _candidate_variance(self, points: list[dict[str, Any]]) -> np.ndarray:
        if self._pool_variance is None:
            return np.zeros(len(points), dtype=float)
        return np.array(
            [self._pool_variance.get(self._encoder.key(p), 0.0) for p in points], dtype=float
        )

    # ── BaseOptimizer interface ──────────────────────────────────────────

    def suggest(self) -> dict[str, Any] | None:
        remaining = self._remaining_points()
        if not remaining:
            return None  # pool exhausted

        # Warm-up: random draw from the remaining pool.
        if len(self._history) < self._n_initial:
            idx = self._rng.randint(len(remaining))
            return remaining[idx]

        # Fit surrogate on observed (params, value) with heteroscedastic alpha.
        X = np.array([self._encoder.encode(p) for p, _ in self._history], dtype=float)
        y = np.array([v for _, v in self._history], dtype=float)
        self.backend.fit(X, y, self._alpha_for_history())

        cand_X = np.array([self._encoder.encode(p) for p in remaining], dtype=float)
        mu, sigma = self.backend.predict(cand_X)

        # Expose candidate aleatoric variance for the acquisition, then score.
        self.candidate_variance = self._candidate_variance(remaining)
        scores = self._strategy.score(self, cand_X, mu, sigma, self._history)
        return remaining[int(np.argmax(scores))]

    def tell(self, params: dict[str, Any], result: float) -> None:
        key = self._encoder.key(params)
        if key not in self._remaining:
            raise CampaignError(
                f"told a point not in the remaining pool (already sampled or "
                f"not a pool point): {params}"
            )
        self._remaining.discard(key)
        self._history.append((params, result))

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._find_best()

    @property
    def n_remaining(self) -> int:
        return len(self._remaining)
