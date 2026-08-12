"""Vogel–Fulcher–Tammann (VFT) analysis for temperature-dependent conductivity.

Provides:

* :class:`VftResult`  — per-series VFT fit output
* :class:`VftFitter`  — non-linear least-squares fit of ln(σ) vs T

Physical model (activation-energy formalism)::

    σ(T) = σ_∞ · exp(−Eₐ / (k_B · (T − T₀)))

with T in Kelvin, ``Eₐ`` the (VFT) activation energy, ``T₀`` the Vogel
temperature (K), and ``σ_∞`` (= ``A``) the high-temperature prefactor.  The
Vogel parameter ``B = Eₐ / k_B`` (units K) is the quantity actually fitted
(it is well-scaled relative to ``T₀``); ``Eₐ = B · k_B`` is reported as the
primary parameter so VFT and Arrhenius are directly comparable.

Unlike Arrhenius (a straight line in 1/T), VFT is non-linear in its three
parameters, so we fit ``ln σ = ln σ_∞ − B / (T − T₀)`` with
:func:`scipy.optimize.curve_fit`.

Mirrors the structure and conventions of
:mod:`softae.analysis.arrhenius` (dataclass result + fitter, NaN-safe valid
mask, ``R_squared`` quality metric).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_KELVIN = 273.15
_KB_EV = 8.617333e-5          # Boltzmann constant (eV/K)
_EV_TO_KJ_PER_MOL = 96.485    # 1 eV = 96.485 kJ/mol


@dataclass
class VftResult:
    """Per-series VFT fit output.

    Attributes
    ----------
    channel, run_id
        Optional labels (mirrors :class:`~softae.analysis.arrhenius.ArrheniusResult`).
    temperatures_C, conductivities
        Parallel input arrays (°C, S/cm); NaN / non-positive σ excluded from the fit.
    A
        Pre-exponential factor σ_∞ (S/cm).
    ln_A
        Natural log of *A* (= ln σ_∞).
    Ea_eV, Ea_kJ_per_mol
        VFT activation energy (``Eₐ = B · k_B``) — the primary reported parameter.
    B
        Vogel parameter (K), ``B = Eₐ / k_B``; the quantity directly fitted.
    T0_K, T0_C
        Vogel temperature (K and °C).
    R_squared
        Coefficient of determination of the ln(σ) fit.
    n_points
        Number of valid points used.
    fit_success, error_msg
        Status of the non-linear fit.
    """

    channel: int = 0
    run_id: str = ""
    temperatures_C: list[float] = field(default_factory=list)
    conductivities: list[float] = field(default_factory=list)
    A: float = float("nan")
    ln_A: float = float("nan")
    Ea_eV: float = float("nan")
    Ea_kJ_per_mol: float = float("nan")
    B: float = float("nan")
    T0_K: float = float("nan")
    T0_C: float = float("nan")
    R_squared: float = float("nan")
    T_min_C: float = float("nan")
    T_max_C: float = float("nan")
    n_points: int = 0
    fit_success: bool = False
    error_msg: str = ""
    model: str = "vft"
    """Thermal model tag, for unified storage alongside Arrhenius fits."""


class VftFitter:
    """Fit σ = A · exp(−B / (T − T₀)) to (T, σ) data.

    Fits the linearised-in-output form ``ln σ = ln A − B / (T − T₀)`` with
    :func:`scipy.optimize.curve_fit` (the model is still non-linear in ``T₀``).
    Points with NaN or non-positive σ are excluded.
    """

    def fit(
        self,
        temperatures_C: list[float],
        conductivities: list[float],
        *,
        channel: int = 0,
        run_id: str = "",
    ) -> VftResult:
        temps = np.asarray(temperatures_C, dtype=float)
        sigmas = np.asarray(conductivities, dtype=float)
        valid = np.isfinite(sigmas) & (sigmas > 0) & np.isfinite(temps)
        n_valid = int(valid.sum())

        result = VftResult(
            channel=channel,
            run_id=run_id,
            temperatures_C=list(temperatures_C),
            conductivities=list(conductivities),
            n_points=n_valid,
        )

        # Three parameters (ln A, B, T₀) → need at least 3 distinct points.
        if n_valid < 3:
            result.error_msg = f"Insufficient valid points ({n_valid}); need ≥ 3 for VFT"
            return result

        T_K = temps[valid] + _KELVIN
        ln_sigma = np.log(sigmas[valid])

        if np.ptp(T_K) < 1e-9:
            result.error_msg = "All temperatures identical; cannot fit VFT"
            return result

        def _model(T, ln_A, B, T0):
            return ln_A - B / (T - T0)

        # Initial guesses: ln_A near the high-T value; B a few hundred K;
        # T₀ comfortably below the lowest measured temperature.
        T_min = float(T_K.min())
        p0 = (float(ln_sigma.max()), 500.0, T_min - 50.0)
        # Bounds: B ≥ 0, T₀ strictly below the lowest T to keep (T − T₀) > 0.
        bounds = ([-np.inf, 0.0, -np.inf], [np.inf, np.inf, T_min - 1e-6])

        try:
            from scipy.optimize import curve_fit

            popt, _ = curve_fit(
                _model, T_K, ln_sigma, p0=p0, bounds=bounds, maxfev=10000
            )
        except Exception as exc:  # convergence failure, etc.
            result.error_msg = f"VFT fit failed: {exc}"
            return result

        ln_A, B, T0 = (float(popt[0]), float(popt[1]), float(popt[2]))
        pred = _model(T_K, ln_A, B, T0)
        ss_res = float(np.sum((ln_sigma - pred) ** 2))
        ss_tot = float(np.sum((ln_sigma - ln_sigma.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        result.ln_A = ln_A
        result.A = float(np.exp(ln_A))
        result.B = B
        # Activation-energy formalism: Eₐ = B · k_B  (B is Eₐ/k_B in Kelvin).
        result.Ea_eV = float(B * _KB_EV)
        result.Ea_kJ_per_mol = float(result.Ea_eV * _EV_TO_KJ_PER_MOL)
        result.T0_K = T0
        result.T0_C = T0 - _KELVIN
        result.R_squared = r_squared
        result.T_min_C = float(temps[valid].min())
        result.T_max_C = float(temps[valid].max())
        result.fit_success = True
        return result
