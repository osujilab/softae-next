"""Temperature-derived objectives: fold a σ(T) series into a scalar objective.

When temperature is an *axis the dataset sweeps* rather than a knob to tune, the
campaign objective is often a parameter of the σ(T) curve — activation energy
``Ea`` (Arrhenius), the VFT ``B``/``T₀``, or σ extrapolated to a target
temperature — not a single conductivity.  A :class:`DerivedObjective` maps the
``(T, σ)`` points of one candidate composition to ``(value, variance)``, which
:meth:`softae.campaigns.datasets.GroundTruthDataset.from_tidy_derived` turns into
the candidate's objective.

These reuse the existing fitters (:class:`softae.analysis.arrhenius.ArrheniusFitter`,
:class:`softae.analysis.vft.VftFitter`).  The observation variance is a
transparent fit-quality heuristic: ``var = var_floor + var_scale · (1 − R²)`` —
a worse fit yields a noisier (less trusted) objective.
"""

from __future__ import annotations

import abc
import math

import numpy as np

from softae.errors import CampaignError

_KB_EV = 8.617333e-5
_LN10 = math.log(10.0)


class DerivedObjective(abc.ABC):
    """Maps a candidate's ``(temps_C, sigmas)`` series to ``(value, variance)``."""

    name: str = "derived"
    #: Natural optimisation direction hint (for documentation / defaults).
    direction_hint: str = "maximize"

    def __init__(self, *, var_scale: float = 0.1, var_floor: float = 1e-6) -> None:
        if var_scale < 0 or var_floor <= 0:
            raise CampaignError("var_scale must be >= 0 and var_floor > 0")
        self.var_scale = var_scale
        self.var_floor = var_floor

    @abc.abstractmethod
    def _value_and_quality(
        self, temps_C: np.ndarray, sigmas: np.ndarray
    ) -> tuple[float, float]:
        """Return ``(value, R_squared)``; value NaN if the fit failed."""

    def compute(self, temps_C, sigmas) -> tuple[float, float, bool]:
        """Return ``(value, variance, ok)`` for one candidate's T-series."""
        value, r2 = self._value_and_quality(
            np.asarray(temps_C, dtype=float), np.asarray(sigmas, dtype=float)
        )
        if not math.isfinite(value):
            return float("nan"), float("inf"), False
        r2 = r2 if math.isfinite(r2) else 0.0
        variance = self.var_floor + self.var_scale * max(0.0, 1.0 - r2)
        return float(value), float(variance), True


class ArrheniusEa(DerivedObjective):
    """Activation energy ``Ea`` (eV) from an Arrhenius fit (minimise for fast transport)."""

    name = "arrhenius_ea"
    direction_hint = "minimize"

    def _value_and_quality(self, temps_C, sigmas):
        from softae.analysis.arrhenius import ArrheniusFitter

        res = ArrheniusFitter().fit(list(temps_C), list(sigmas))
        if not res.fit_success:
            return float("nan"), float("nan")
        return res.Ea_eV, res.R_squared


class ArrheniusLnA(DerivedObjective):
    """Pre-exponential ``ln A`` from an Arrhenius fit."""

    name = "arrhenius_ln_a"
    direction_hint = "maximize"

    def _value_and_quality(self, temps_C, sigmas):
        from softae.analysis.arrhenius import ArrheniusFitter

        res = ArrheniusFitter().fit(list(temps_C), list(sigmas))
        if not res.fit_success:
            return float("nan"), float("nan")
        return res.ln_A, res.R_squared


class ArrheniusSigmaAtT(DerivedObjective):
    """log₁₀ σ extrapolated to a target temperature via the Arrhenius fit."""

    name = "arrhenius_sigma_at_T"
    direction_hint = "maximize"

    def __init__(self, *, target_temp_C: float = 25.0, **kw) -> None:
        super().__init__(**kw)
        self.target_temp_C = target_temp_C

    def _value_and_quality(self, temps_C, sigmas):
        from softae.analysis.arrhenius import ArrheniusFitter

        res = ArrheniusFitter().fit(list(temps_C), list(sigmas))
        if not res.fit_success:
            return float("nan"), float("nan")
        # ln σ = ln_A − (Ea/kB)·(1/T)  →  log10 σ at the target T.
        T = self.target_temp_C + 273.15
        ln_sigma = res.ln_A - (res.Ea_eV / _KB_EV) * (1.0 / T)
        return ln_sigma / _LN10, res.R_squared


class VftB(DerivedObjective):
    """VFT pseudo-activation parameter ``B`` (K) (minimise for fast transport)."""

    name = "vft_B"
    direction_hint = "minimize"

    def _value_and_quality(self, temps_C, sigmas):
        from softae.analysis.vft import VftFitter

        res = VftFitter().fit(list(temps_C), list(sigmas))
        if not res.fit_success:
            return float("nan"), float("nan")
        return res.B, res.R_squared


class VftSigmaAtT(DerivedObjective):
    """log₁₀ σ extrapolated to a target temperature via the VFT fit."""

    name = "vft_sigma_at_T"
    direction_hint = "maximize"

    def __init__(self, *, target_temp_C: float = 25.0, **kw) -> None:
        super().__init__(**kw)
        self.target_temp_C = target_temp_C

    def _value_and_quality(self, temps_C, sigmas):
        from softae.analysis.vft import VftFitter

        res = VftFitter().fit(list(temps_C), list(sigmas))
        if not res.fit_success:
            return float("nan"), float("nan")
        T = self.target_temp_C + 273.15
        ln_sigma = res.ln_A - res.B / (T - res.T0_K)
        return ln_sigma / _LN10, res.R_squared


#: Registry mapping config strings → derived-objective classes.
DERIVED_OBJECTIVES: dict[str, type[DerivedObjective]] = {
    ArrheniusEa.name: ArrheniusEa,
    ArrheniusLnA.name: ArrheniusLnA,
    ArrheniusSigmaAtT.name: ArrheniusSigmaAtT,
    VftB.name: VftB,
    VftSigmaAtT.name: VftSigmaAtT,
}


def build_derived_objective(
    name: str, *, target_temp_C: float = 25.0, var_scale: float = 0.1, var_floor: float = 1e-6
) -> DerivedObjective:
    """Construct a derived objective by registry name."""
    if name not in DERIVED_OBJECTIVES:
        raise CampaignError(
            f"unknown temperature_objective '{name}'; available: {sorted(DERIVED_OBJECTIVES)}"
        )
    cls = DERIVED_OBJECTIVES[name]
    if name in ("arrhenius_sigma_at_T", "vft_sigma_at_T"):
        return cls(target_temp_C=target_temp_C, var_scale=var_scale, var_floor=var_floor)
    return cls(var_scale=var_scale, var_floor=var_floor)
