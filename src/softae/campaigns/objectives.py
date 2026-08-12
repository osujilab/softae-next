"""Objective transforms: map measured conductivity → optimization objective.

Conductivity spans many orders of magnitude, so the default objective is
``log10(σ)``.  A transform also exposes its first derivative (:meth:`dydx`) so a
:class:`~softae.campaigns.noise.NoiseModel` can propagate a raw-σ replicate
spread into objective-space variance via the delta method — keeping the noise
model agnostic to which transform is active.

Transforms are *direction-free*: maximise/minimise is handled separately by the
optimizer's ``objective`` argument.  ``ArrheniusDerivedObjective`` is a phase-P3
seam and is intentionally left unimplemented.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from softae.errors import CampaignError

_LN10 = math.log(10.0)


class ObjectiveTransform(abc.ABC):
    """Maps conductivity (S/cm) to an objective value, with a delta-method hook."""

    name: str = "transform"

    @abc.abstractmethod
    def apply(self, sigma: np.ndarray) -> np.ndarray:
        """Transform an array of σ values; invalid entries become ``NaN``."""

    @abc.abstractmethod
    def dydx(self, sigma_mean: float) -> float:
        """|d(objective)/dσ| at ``sigma_mean``, for variance propagation.

        Returns ``nan`` when the derivative is undefined (e.g. σ ≤ 0 for log).
        """

    def apply_scalar(self, sigma: float) -> float:
        """Convenience scalar wrapper around :meth:`apply`."""
        return float(self.apply(np.asarray([sigma], dtype=float))[0])


class Log10Sigma(ObjectiveTransform):
    """``y = log10(σ)``.  σ ≤ 0 → NaN (mirrors ``ArrheniusFitter``'s valid mask)."""

    name = "log10_sigma"

    def apply(self, sigma: np.ndarray) -> np.ndarray:
        s = np.asarray(sigma, dtype=float)
        out = np.full_like(s, np.nan, dtype=float)
        valid = np.isfinite(s) & (s > 0)
        out[valid] = np.log10(s[valid])
        return out

    def dydx(self, sigma_mean: float) -> float:
        if not math.isfinite(sigma_mean) or sigma_mean <= 0:
            return float("nan")
        return 1.0 / (sigma_mean * _LN10)


class RawSigma(ObjectiveTransform):
    """Identity transform: ``y = σ`` (use when σ does not span decades)."""

    name = "raw_sigma"

    def apply(self, sigma: np.ndarray) -> np.ndarray:
        s = np.asarray(sigma, dtype=float)
        out = s.astype(float, copy=True)
        out[~np.isfinite(s)] = np.nan
        return out

    def dydx(self, sigma_mean: float) -> float:
        return 1.0


class ArrheniusDerivedObjective(ObjectiveTransform):
    """Seam (P3): objective derived from an Arrhenius/VFT fit of σ(T).

    Will group tidy rows by composition, fit σ(T) with the existing
    :class:`softae.analysis.arrhenius.ArrheniusFitter`, and return e.g. ``Ea`` or
    σ at a target temperature.  Not implemented in phase P0/P1.
    """

    name = "arrhenius_derived"

    def apply(self, sigma: np.ndarray) -> np.ndarray:  # pragma: no cover - seam
        raise CampaignError(
            "ArrheniusDerivedObjective is a phase-P3 seam and is not implemented yet."
        )

    def dydx(self, sigma_mean: float) -> float:  # pragma: no cover - seam
        raise CampaignError(
            "ArrheniusDerivedObjective is a phase-P3 seam and is not implemented yet."
        )


#: Registry mapping config strings → transform classes.
OBJECTIVES: dict[str, type[ObjectiveTransform]] = {
    Log10Sigma.name: Log10Sigma,
    RawSigma.name: RawSigma,
    ArrheniusDerivedObjective.name: ArrheniusDerivedObjective,
}


def get_transform(name: str) -> ObjectiveTransform:
    """Instantiate a transform by its registry name."""
    try:
        return OBJECTIVES[name]()
    except KeyError:
        raise CampaignError(
            f"unknown objective transform '{name}'; available: {sorted(OBJECTIVES)}"
        ) from None
