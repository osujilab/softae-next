"""Observation-noise models — the explicit menu for feeding uncertainty to the optimizer.

A :class:`NoiseModel` maps a candidate's per-replicate statistics
(:class:`CellStats`) to an **observation-noise variance in objective space**
(the same space as the GP target, e.g. ``log10 σ``).  That variance is what the
campaign feeds to the surrogate's ``alpha`` (a fixed-noise / heteroscedastic GP)
and/or to an uncertainty-aware acquisition.

The design exposes three orthogonal, independently configurable choices:

1. **Which sources** estimate the variance — replicate spread
   (:class:`ReplicateNoise`), EIS fit quality (:class:`FitQualityNoise`), their
   combination (:class:`CompositeNoise`), or a single global level
   (:class:`HomoscedasticNoise`).
2. **Mean vs single draw** — ``target_is_mean`` decides whether the variance is
   that of the aggregated replicate mean (SEM², divide by ``n_rep``) or of one
   revealed replicate (full ``std²``).
3. **Where the noise enters the optimizer** — handled downstream by
   ``noise_channel`` (``"alpha"`` / ``"acquisition_weight"`` / ``"both"``); this
   module only produces the variance.

Variance propagates from raw-σ space to objective space via the active
transform's delta-method derivative (:meth:`ObjectiveTransform.dydx`), so the
models work for any transform, not just ``log10``.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from softae.campaigns.objectives import ObjectiveTransform
from softae.errors import CampaignError

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Per-candidate statistics (input to every NoiseModel)
# ---------------------------------------------------------------------------

@dataclass
class CellStats:
    """Aggregated statistics for one candidate (a distinct condition).

    A "cell" is one ``point_id`` — all its replicates collapsed into summary
    stats plus the objective-space mean.
    """

    point_id: str
    params: dict[str, Any]               # composition + environment coordinates
    sigma_values: list[float]            # raw replicate σ (S/cm)
    sigma_mean: float                    # mean of finite, positive σ
    sigma_std: float                     # std (ddof=1); nan if < 2 valid replicates
    n_rep: int                           # number of valid replicates
    fitted_Z: float = float("nan")
    adjusted_Z: float = float("nan")
    fit_residual: float = float("nan")   # alt fit-quality source (raw-EIS path)
    y: float = float("nan")              # objective-space mean (transform of sigma_mean)
    is_rail: bool = False                # flagged sensor/fit rail (very low trust)


def relative_fit_discrepancy(cell: CellStats) -> float:
    """Relative impedance discrepancy ``|fitted - adjusted| / max(|adjusted|, eps)``.

    A proxy for EIS fit quality: since σ ∝ 1/R₁, a relative error in the fitted
    impedance is approximately a relative error in σ.  Returns ``nan`` when
    impedance columns are absent.
    """
    f, a = cell.fitted_Z, cell.adjusted_Z
    if not (math.isfinite(f) and math.isfinite(a)):
        return float("nan")
    return abs(f - a) / max(abs(a), _EPS)


def fit_quality_weight(cell: CellStats) -> float:
    """Acquisition-channel weight ``1 / (1 + disc)`` ∈ (0, 1]; 1.0 if no fit info."""
    disc = relative_fit_discrepancy(cell)
    if not math.isfinite(disc):
        return 1.0
    return 1.0 / (1.0 + disc)


# ---------------------------------------------------------------------------
# NoiseModel ABC
# ---------------------------------------------------------------------------

class NoiseModel(abc.ABC):
    """Maps a :class:`CellStats` to an objective-space observation-noise variance."""

    name: str = "noise"
    #: When True, the campaign should hand the surrogate *no* fixed variances and
    #: let it learn a single homoscedastic noise level (sklearn ``WhiteKernel``).
    learn_noise: bool = False

    def __init__(self, *, target_is_mean: bool = True, var_floor: float = 1e-6) -> None:
        if var_floor <= 0:
            raise CampaignError("var_floor must be > 0")
        self.target_is_mean = target_is_mean
        self.var_floor = float(var_floor)

    def prepare(self, cells: list[CellStats], transform: ObjectiveTransform) -> None:
        """Optional one-shot fit over all cells (e.g. compute pooled variance)."""

    @abc.abstractmethod
    def _raw_variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        """Component-specific variance *before* flooring (may be nan/0)."""

    def variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        """Objective-space variance for *cell*, floored to be strictly positive."""
        v = self._raw_variance(cell, transform)
        if not math.isfinite(v) or v < 0:
            v = 0.0
        return v + self.var_floor


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------

class ReplicateNoise(NoiseModel):
    """Variance from replicate spread, propagated to objective space.

    ``var_sigma = std²`` (or ``std²/n_rep`` when ``target_is_mean``), then mapped
    to objective space with the delta method ``var_obj = dydx(mean)² · var_sigma``.

    With only n=2 replicates the empirical std is itself very uncertain, so
    ``shrinkage="pooled"`` blends the per-cell variance toward a global pooled
    variance: ``var = λ·pooled + (1−λ)·cell``.  Single-replicate cells (std=NaN)
    fall back fully to the pooled value.
    """

    name = "replicate"

    def __init__(
        self,
        *,
        target_is_mean: bool = True,
        var_floor: float = 1e-6,
        shrinkage: str = "pooled",
        lam: float = 0.5,
    ) -> None:
        super().__init__(target_is_mean=target_is_mean, var_floor=var_floor)
        if shrinkage not in ("none", "pooled"):
            raise CampaignError("shrinkage must be 'none' or 'pooled'")
        if not 0.0 <= lam <= 1.0:
            raise CampaignError("lam must be in [0, 1]")
        self.shrinkage = shrinkage
        self.lam = float(lam)
        self._pooled_var: float = 0.0

    def _cell_var_obj(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        if cell.n_rep < 2 or not math.isfinite(cell.sigma_std):
            return float("nan")
        var_sigma = cell.sigma_std ** 2
        if self.target_is_mean:
            var_sigma /= cell.n_rep
        jac = transform.dydx(cell.sigma_mean)
        if not math.isfinite(jac):
            return float("nan")
        return jac * jac * var_sigma

    def prepare(self, cells: list[CellStats], transform: ObjectiveTransform) -> None:
        vals = [self._cell_var_obj(c, transform) for c in cells]
        finite = [v for v in vals if math.isfinite(v)]
        # Pooled estimate = mean of per-cell variances (robust enough for a
        # benchmark; median is an easy swap if outliers dominate).
        self._pooled_var = float(np.mean(finite)) if finite else 0.0

    def _raw_variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        cell_var = self._cell_var_obj(cell, transform)
        if self.shrinkage == "none":
            return cell_var if math.isfinite(cell_var) else 0.0
        # pooled shrinkage
        if not math.isfinite(cell_var):
            return self._pooled_var  # single-replicate → fully pooled
        return self.lam * self._pooled_var + (1.0 - self.lam) * cell_var


class FitQualityNoise(NoiseModel):
    """Variance from EIS fit quality (impedance discrepancy).

    ``var_obj = (k_fit · dydx(mean) · disc · mean)²`` where
    ``disc = |fitted_Z − adjusted_Z| / max(|adjusted_Z|, eps)``.  For ``log10``
    this reduces to ``(k_fit · disc / ln10)²``; for raw σ it is the squared
    absolute error ``(k_fit · disc · mean)²``.

    When fitting from raw EIS instead of the aggregated file, set the alternate
    source via ``fit_residual`` on the cell and ``source="residual"`` here.
    """

    name = "fit_quality"

    def __init__(
        self,
        *,
        target_is_mean: bool = True,
        var_floor: float = 1e-6,
        k_fit: float = 1.0,
        source: str = "impedance",
    ) -> None:
        super().__init__(target_is_mean=target_is_mean, var_floor=var_floor)
        if k_fit < 0:
            raise CampaignError("k_fit must be >= 0")
        if source not in ("impedance", "residual"):
            raise CampaignError("source must be 'impedance' or 'residual'")
        self.k_fit = float(k_fit)
        self.source = source

    def _raw_variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        if self.source == "residual":
            disc = cell.fit_residual
        else:
            disc = relative_fit_discrepancy(cell)
        if not math.isfinite(disc):
            return 0.0
        jac = transform.dydx(cell.sigma_mean)
        if not math.isfinite(jac):
            return 0.0
        abs_err_obj = self.k_fit * jac * disc * cell.sigma_mean
        return abs_err_obj * abs_err_obj


class HomoscedasticNoise(NoiseModel):
    """A single global noise level.

    ``level=None`` (default) sets :attr:`learn_noise` so the campaign lets the
    surrogate learn one noise level (sklearn ``WhiteKernel``) — the baseline to
    benchmark per-point models against.  A float fixes the variance.
    """

    name = "homoscedastic"

    def __init__(
        self,
        *,
        level: float | None = None,
        target_is_mean: bool = True,
        var_floor: float = 1e-6,
    ) -> None:
        super().__init__(target_is_mean=target_is_mean, var_floor=var_floor)
        if level is not None and level < 0:
            raise CampaignError("level must be >= 0 or None")
        self.level = level
        self.learn_noise = level is None

    def _raw_variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        return 0.0 if self.level is None else float(self.level)


class CompositeNoise(NoiseModel):
    """Combine several noise sources.

    ``combine="sum"`` adds variances (independent error sources — the default);
    ``combine="max"`` takes the worst source (conservative).
    """

    name = "composite"

    def __init__(
        self,
        sources: list[NoiseModel],
        *,
        combine: str = "sum",
        var_floor: float = 1e-6,
    ) -> None:
        # target_is_mean is governed by the child sources, not the composite.
        super().__init__(target_is_mean=True, var_floor=var_floor)
        if not sources:
            raise CampaignError("CompositeNoise requires at least one source")
        if combine not in ("sum", "max"):
            raise CampaignError("combine must be 'sum' or 'max'")
        if any(s.learn_noise for s in sources):
            raise CampaignError("cannot compose a learn-noise (homoscedastic None) source")
        self.sources = sources
        self.combine = combine

    def prepare(self, cells: list[CellStats], transform: ObjectiveTransform) -> None:
        for s in self.sources:
            s.prepare(cells, transform)

    def _raw_variance(self, cell: CellStats, transform: ObjectiveTransform) -> float:
        # Use each source's floored variance minus its own floor, so floors do
        # not stack; the composite applies its own floor once in variance().
        parts = [max(s.variance(cell, transform) - s.var_floor, 0.0) for s in self.sources]
        return sum(parts) if self.combine == "sum" else max(parts)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

#: Single-source registry (composite is built via :func:`build_noise_model`).
NOISE_MODELS: dict[str, type[NoiseModel]] = {
    ReplicateNoise.name: ReplicateNoise,
    FitQualityNoise.name: FitQualityNoise,
    HomoscedasticNoise.name: HomoscedasticNoise,
    CompositeNoise.name: CompositeNoise,
}


def build_noise_model(
    name: str = "composite",
    *,
    sources: list[str] | None = None,
    combine: str = "sum",
    target_is_mean: bool = True,
    var_floor: float = 1e-6,
    k_fit: float = 1.0,
    replicate_shrinkage: str = "pooled",
    lam: float = 0.5,
    homoscedastic_level: float | None = None,
) -> NoiseModel:
    """Construct a :class:`NoiseModel` from config-style primitives.

    ``name="composite"`` builds a :class:`CompositeNoise` over ``sources``
    (default ``["replicate", "fit_quality"]``); any other name builds that single
    source directly.
    """

    def _make(src: str) -> NoiseModel:
        if src == ReplicateNoise.name:
            return ReplicateNoise(
                target_is_mean=target_is_mean,
                var_floor=var_floor,
                shrinkage=replicate_shrinkage,
                lam=lam,
            )
        if src == FitQualityNoise.name:
            return FitQualityNoise(
                target_is_mean=target_is_mean, var_floor=var_floor, k_fit=k_fit
            )
        if src == HomoscedasticNoise.name:
            return HomoscedasticNoise(
                level=homoscedastic_level,
                target_is_mean=target_is_mean,
                var_floor=var_floor,
            )
        raise CampaignError(
            f"unknown noise source '{src}'; available: "
            f"{sorted(k for k in NOISE_MODELS if k != 'composite')}"
        )

    if name == CompositeNoise.name:
        src_names = sources if sources is not None else ["replicate", "fit_quality"]
        return CompositeNoise(
            [_make(s) for s in src_names], combine=combine, var_floor=var_floor
        )
    if name not in NOISE_MODELS:
        raise CampaignError(
            f"unknown noise model '{name}'; available: {sorted(NOISE_MODELS)}"
        )
    return _make(name)
