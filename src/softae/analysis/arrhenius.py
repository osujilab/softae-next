"""Arrhenius analysis for temperature-stepped EIS experiments.

Provides:

* :class:`ArrheniusSweepConfig` — configuration dataclass for a temperature sweep
* :class:`ArrheniusResult`      — per-channel Arrhenius fit output
* :class:`ArrheniusFitter`      — ln(σ) vs 1/T linear regression

Physical model::

    σ = A · exp(−Eₐ / (k_B · T))

Linearised::

    ln(σ) = ln(A) − (Eₐ / k_B) · (1/T)

where T is in Kelvin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_KB_EV: float = 8.617333e-5          # Boltzmann constant (eV/K)
_EV_TO_KJ_PER_MOL: float = 96.485    # 1 eV = 96.485 kJ/mol


# ---------------------------------------------------------------------------
# ArrheniusSweepConfig
# ---------------------------------------------------------------------------

@dataclass
class ArrheniusSweepConfig:
    """Parameters that define a temperature-stepped EIS sweep.

    Temperature specification (mutually exclusive modes):

    * **Step mode** (default): generate from *T_start*, *T_stop*, *T_step*.
    * **Explicit mode**: provide *temperatures* list — overrides step mode.

    Attributes
    ----------
    channels : list[int]
        EIS channels to measure (1-indexed, upper bound determined by PCB config).
    T_start : float
        Sweep start temperature (°C). Ignored if *temperatures* is given.
    T_stop : float
        Sweep stop temperature (°C, inclusive). Ignored if *temperatures* given.
    T_step : float
        Step between temperatures (°C, must be > 0). Ignored if *temperatures* given.
    temperatures : list[float] or None
        Explicit temperature list (°C). Overrides step-mode parameters.
    dwell_s : float
        Hold time (seconds) after equilibration, before EIS measurement.
    tolerance_C : float
        Maximum |PV − SP| before the dwell begins (°C).
    wait_timeout_s : float
        Maximum wait time for a setpoint to be reached (seconds).
    eis_model : str
        Circuit model for fitting (must be a key in ``CIRCUIT_MODELS``).
    electrode_geometry : dict or None
        Keys: ``L_cm``, ``t_cm``, ``w_cm``.  Required to compute σ; if
        ``None``, σ is set to ``NaN`` in the results.
    """

    channels: list[int]
    T_start: float = 25.0
    T_stop: float = 75.0
    T_step: float = 10.0
    temperatures: list[float] | None = None
    dwell_s: float = 60.0
    tolerance_C: float = 0.5
    wait_timeout_s: float = 1800.0
    eis_model: str = "simpleSalt"
    thermal_model: str = "arrhenius"
    """Temperature-dependence model fitted to σ(T): ``"arrhenius"`` (σ = A·exp(−Eₐ/k_BT))
    or ``"vft"`` (σ = A·exp(−B/(T−T₀))).  VFT needs ≥ 3 temperatures."""
    electrode_geometry: dict[str, float] | None = None
    eis_params: dict[str, int | float] | None = None
    """EIS measurement parameters.  Keys: ``f_hi`` (Hz), ``f_lo_mHz`` (mHz),
    ``npts``, ``mv_ac`` (mV), ``mv_dc`` (mV).  When ``None``, the HT-preset
    defaults are used when building the sweep .mscr files."""

    rh_setpoints: list[float] | None = None
    """Optional RH sweep (%RH).  When provided, the T sweep is repeated at each
    setpoint (outer loop = RH, inner loop = T).  ``None`` disables RH sweep."""
    rh_dwell_s: float = 30.0
    """Additional hold time (seconds) *after* RH stabilises at the new setpoint.
    Analogous to ``dwell_s`` for temperature."""
    rh_tolerance: float = 2.0
    """Maximum |RH_PV − RH_SP| considered stable (% RH).
    Analogous to ``tolerance_C`` for temperature."""
    rh_wait_timeout_s: float = 600.0
    """Maximum time to wait for RH to reach the setpoint (seconds).
    Analogous to ``wait_timeout_s`` for temperature."""
    rh_instrument: str = "rh_controller"
    """Name of the RH controller instrument in the manager."""
    sweep_order: dict[str, int] = field(
        default_factory=lambda: {"RH": 1, "T": 2, "channels": 3}
    )
    """Nesting order for the three sweep axes.  Keys: ``"RH"``, ``"T"``,
    ``"channels"``.  Values must be a permutation of ``{1, 2, 3}`` where
    1 = outermost loop (changes slowest) and 3 = innermost loop (swept
    continuously).  Default reproduces the original behaviour: RH outermost,
    temperature middle, channels innermost."""

    def resolved_temperatures(self) -> list[float]:
        """Return the explicit list, or auto-generate from start/stop/step.

        The stop temperature is included if it falls on a step boundary
        (within floating-point tolerance).  When ``T_start == T_stop`` a
        single-point list is returned immediately, bypassing ``T_step``.
        """
        if self.temperatures is not None:
            return list(self.temperatures)
        if self.T_start == self.T_stop:
            return [float(self.T_start)]
        if self.T_step <= 0:
            raise ValueError("T_step must be > 0")
        n = int(round((self.T_stop - self.T_start) / self.T_step)) + 1
        return [round(self.T_start + i * self.T_step, 6) for i in range(n)]

    def validate(self) -> None:
        """Raise :class:`ValueError` for invalid configurations."""
        if not self.channels:
            raise ValueError("channels must not be empty")
        for ch in self.channels:
            if ch < 1:
                raise ValueError(f"Channel {ch} is outside valid range (must be >= 1)")
        temps = self.resolved_temperatures()
        if not temps:
            raise ValueError("resolved_temperatures() returned an empty list")
        if self.thermal_model not in ("arrhenius", "vft"):
            raise ValueError(
                f"thermal_model must be 'arrhenius' or 'vft', got '{self.thermal_model}'"
            )
        if self.thermal_model == "vft" and len(temps) < 3:
            raise ValueError("VFT fitting requires at least 3 temperatures")
        if self.dwell_s < 0:
            raise ValueError("dwell_s must be >= 0")
        if self.tolerance_C <= 0:
            raise ValueError("tolerance_C must be > 0")
        if self.T_step <= 0 and self.temperatures is None and self.T_start != self.T_stop:
            raise ValueError("T_step must be > 0")
        if self.rh_dwell_s < 0:
            raise ValueError("rh_dwell_s must be >= 0")
        if self.rh_tolerance <= 0:
            raise ValueError("rh_tolerance must be > 0")
        if self.rh_wait_timeout_s <= 0:
            raise ValueError("rh_wait_timeout_s must be > 0")

    # ── Serialisation ────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialise to a pretty-printed JSON string.

        All fields are plain Python primitives so :func:`dataclasses.asdict`
        produces a directly JSON-serialisable dict.
        """
        import dataclasses
        import json as _json
        return _json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "ArrheniusSweepConfig":
        """Deserialise from a JSON string produced by :meth:`to_json`.

        Unknown keys are silently ignored so that configs saved by a newer
        version of the code can be loaded by an older one.
        """
        import dataclasses
        import json as _json
        d = _json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# ArrheniusResult
# ---------------------------------------------------------------------------

@dataclass
class ArrheniusResult:
    """Per-channel Arrhenius fit output.

    Attributes
    ----------
    channel : int
    run_id : str
    temperatures_C : list[float]
        Sweep temperatures (°C), one per measurement point.
    conductivities : list[float]
        σ (S/cm) at each temperature; ``NaN`` when fit failed or σ unavailable.
    Ea_eV : float
        Activation energy (eV).
    Ea_kJ_per_mol : float
        Activation energy (kJ/mol).
    ln_A : float
        Natural log of the pre-exponential factor.
    R_squared : float
        Coefficient of determination for the ln(σ) vs 1/T linear fit.
    T_min_C : float
        Minimum temperature used in the fit (°C).
    T_max_C : float
        Maximum temperature used in the fit (°C).
    n_points : int
        Number of valid σ points used in the fit.
    fit_success : bool
    error_msg : str
    """

    channel: int
    run_id: str
    temperatures_C: list[float] = field(default_factory=list)
    conductivities: list[float] = field(default_factory=list)
    Ea_eV: float = float("nan")
    Ea_kJ_per_mol: float = float("nan")
    ln_A: float = float("nan")
    R_squared: float = float("nan")
    T_min_C: float = float("nan")
    T_max_C: float = float("nan")
    n_points: int = 0
    fit_success: bool = False
    error_msg: str = ""
    model: str = "arrhenius"
    """Thermal model tag, for unified storage alongside VFT fits."""


# ---------------------------------------------------------------------------
# ArrheniusFitter
# ---------------------------------------------------------------------------

class ArrheniusFitter:
    """Fits the Arrhenius equation: σ = A · exp(−Eₐ / (k_B · T)).

    Uses ``scipy.stats.linregress`` on the linearised form:
    ln(σ) = ln(A) − (Eₐ / k_B) · (1/T).

    Points with NaN or non-positive conductivity are excluded.
    """

    KB_EV: float = _KB_EV

    def fit(
        self,
        temperatures_C: list[float],
        conductivities: list[float],
        *,
        channel: int = 0,
        run_id: str = "",
    ) -> ArrheniusResult:
        """Perform Arrhenius fit for one channel.

        Parameters
        ----------
        temperatures_C : list[float]
            Measurement temperatures (°C).
        conductivities : list[float]
            σ (S/cm) at each temperature; NaN values are excluded.
        channel : int
            Channel number for labelling.
        run_id : str
            Run identifier for labelling.

        Returns
        -------
        ArrheniusResult
        """
        temps = np.array(temperatures_C, dtype=float)
        sigmas = np.array(conductivities, dtype=float)

        valid = np.isfinite(sigmas) & (sigmas > 0) & np.isfinite(temps)
        n_valid = int(valid.sum())

        result = ArrheniusResult(
            channel=channel,
            run_id=run_id,
            temperatures_C=list(temperatures_C),
            conductivities=list(conductivities),
            n_points=n_valid,
        )

        if n_valid < 2:
            result.error_msg = f"Insufficient valid data points ({n_valid}); need ≥ 2"
            return result

        from scipy.stats import linregress  # type: ignore

        T_K = temps[valid] + 273.15
        inv_T = 1.0 / T_K
        ln_sigma = np.log(sigmas[valid])

        slope, intercept, r_value, _p, _se = linregress(inv_T, ln_sigma)

        # slope = −Eₐ / k_B  →  Eₐ = −slope · k_B
        Ea_eV = float(-slope * self.KB_EV)
        Ea_kJ = float(Ea_eV * _EV_TO_KJ_PER_MOL)

        result.Ea_eV = Ea_eV
        result.Ea_kJ_per_mol = Ea_kJ
        result.ln_A = float(intercept)
        result.R_squared = float(r_value ** 2)
        result.T_min_C = float(temps[valid].min())
        result.T_max_C = float(temps[valid].max())
        result.fit_success = True
        return result
