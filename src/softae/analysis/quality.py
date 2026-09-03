"""Is this measurement trustworthy enough to optimize against? (P4)

An unattended campaign turns every measurement into a decision: the objective it
produces steers the next suggestion, and nobody is watching. Two failures matter
more than the rest, because both produce numbers that *look* fine:

* a **bad trace** — a dead channel, a saturated amplifier, an open circuit — still
  yields floats, and averaging them gives a plausible objective; and
* a **bad fit** — a circuit model that did not converge onto the data — still
  reports an ``R1``, and ``σ = L/(R·w·t)`` still evaluates.

Nothing here throws data away on its own. The gate's verdict feeds the *existing*
unmeasured path (P0.1): a rejected measurement is reported as **no value**, so it
is never told to the optimizer, while the well is still recorded as cast. That
reuse matters — a second, parallel rejection route would be a second place for
"unmeasured" to be mishandled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ── Thresholds (overridable from ``[quality]`` config) ───────────────────────

#: Below this coefficient of determination the fit did not describe the data.
DEFAULT_MIN_R_SQUARED = 0.95
#: Above this RMS residual (%) the fit tracks the data too loosely to trust.
DEFAULT_MAX_RESIDUAL_PCT = 15.0
#: A trace with fewer usable points than this cannot support a 5-parameter fit.
DEFAULT_MIN_POINTS = 8
#: |Z| at or below this is an implausible short — a dead or shorted channel.
DEFAULT_MIN_ABS_Z = 1e-3
#: |Z| at or above this is an implausible open circuit.
DEFAULT_MAX_ABS_Z = 1e12
#: Instrument-path stray capacitance, F. **A last-resort fallback, not a second home
#: for the number** — ``[eis.instrument] stray_C_instrument_F`` is authoritative and
#: :func:`quality_config` reads it. This constant is reached only when the config
#: cannot be loaded at all, and it is kept equal to the shipped value so that path is
#: not silently a different instrument.
#:
#: **Two-electrode, and that is the correct basis rather than a limitation.** A
#: three-electrode figure for a two-terminal load is a floating-divider artefact
#: (overhaul F17) — the same ch17 blank reads 246 pF that way against 10.2 pF tied.
#:
#: It is a **board median** over seven tied open blanks spanning 10.2–24.7 pF, a real
#: 2.4× per-channel variation repeatable to 1%. Measured across that whole spread the
#: screen below flags 1–3 of 296 stored spectra, with exactly one stable either way —
#: so the median's imprecision does not decide the outcome here.
DEFAULT_STRAY_C_F = 18.5e-12


def open_circuit_z_ohm(freq: Any, stray_c_f: float = DEFAULT_STRAY_C_F) -> float:
    """|Z| an **unbridged** cell presents at this sweep's geometric-mid frequency.

    An open circuit is not "very large |Z|" — it is a specific physical object: the
    fixture's stray capacitance with nothing across it. So the threshold is *derived*
    from that capacitance and the band actually swept, rather than chosen::

        Z_open = 1 / (2 pi f_geo C_stray)

    **The band matters, which is why this is a function and not a constant.** Over
    3.9 Hz – 200 kHz the same 18.5 pF presents 9.7e6 Ω at the geometric mid and 2.2e9 Ω
    at the bottom of the sweep — so a scalar threshold silently encodes one sweep's
    geometry as though it were a property of the hardware. Measured on 296 stored
    spectra, a fixed 9.7e6 flags four and this per-spectrum form flags one: the
    difference is not cosmetic.

    ``f_geo`` is the geometric mean of the swept extremes because the comparand is a
    *median* over a logarithmically spaced sweep, and the geometric mean is where that
    median sits. Returns ``nan`` when the band or the capacitance is unusable, which
    callers must treat as "no opinion" rather than as a passing threshold.
    """
    f = np.asarray(freq, dtype=float)
    f = f[np.isfinite(f) & (f > 0.0)]
    if f.size < 2 or not np.isfinite(stray_c_f) or stray_c_f <= 0.0:
        return float("nan")
    f_geo = float(np.sqrt(f.min() * f.max()))
    if not np.isfinite(f_geo) or f_geo <= 0.0:
        return float("nan")
    return 1.0 / (2.0 * np.pi * f_geo * float(stray_c_f))


class Verdict(str, Enum):
    """What to do with a measurement."""

    ACCEPT = "accept"    # use it as an objective
    SUSPECT = "suspect"  # use it, but flag — degraded, not disqualifying
    REJECT = "reject"    # do not tell the optimizer; report as unmeasured


@dataclass
class QualityReport:
    """Why a measurement was accepted, flagged, or rejected."""

    verdict: Verdict
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the measurement may be used as an objective."""
        return self.verdict is not Verdict.REJECT

    def summary(self) -> str:
        if not self.issues:
            return f"{self.verdict.value}: no issues"
        return f"{self.verdict.value}: " + "; ".join(self.issues)


# ── 4.2 Raw-trace validation ────────────────────────────────────────────────

def validate_eis_trace(
    eis_result: Any,
    *,
    min_points: int = DEFAULT_MIN_POINTS,
    min_abs_z: float = DEFAULT_MIN_ABS_Z,
    max_abs_z: float = DEFAULT_MAX_ABS_Z,
    stray_c_f: float = DEFAULT_STRAY_C_F,
) -> QualityReport:
    """Check a raw impedance trace before anything is fitted to it.

    Rejects only what is *physically impossible or unusable* — non-finite data, a
    trace too short to fit, an implausible short or open. Everything else that
    merely looks unusual is reported as ``SUSPECT``: a soft material genuinely can
    produce an odd-looking spectrum, and a gate that rejects unfamiliar physics
    would quietly bias the campaign toward the samples the checker expected.
    """
    issues: list[str] = []
    metrics: dict[str, float] = {}

    try:
        freq = np.asarray(eis_result.frequency, dtype=float)
        z_real = np.asarray(eis_result.z_real, dtype=float)
        z_imag = np.asarray(eis_result.z_imag_neg, dtype=float)
    except Exception as exc:
        return QualityReport(Verdict.REJECT, [f"trace unreadable: {exc}"])

    n = int(min(freq.size, z_real.size, z_imag.size))
    metrics["n_points"] = float(n)

    if n == 0:
        return QualityReport(Verdict.REJECT, ["empty trace"], metrics)

    finite = np.isfinite(freq[:n]) & np.isfinite(z_real[:n]) & np.isfinite(z_imag[:n])
    n_finite = int(finite.sum())
    metrics["n_finite"] = float(n_finite)

    if n_finite == 0:
        return QualityReport(Verdict.REJECT, ["no finite points"], metrics)
    if n_finite < n:
        issues.append(f"{n - n_finite} of {n} points non-finite")

    if n_finite < int(min_points):
        issues.append(f"only {n_finite} usable points (need {int(min_points)})")
        return QualityReport(Verdict.REJECT, issues, metrics)

    mag = np.hypot(z_real[:n][finite], z_imag[:n][finite])
    z_med = float(np.median(mag))
    metrics["z_median"] = z_med
    metrics["z_min"] = float(np.min(mag))
    metrics["z_max"] = float(np.max(mag))

    if z_med <= float(min_abs_z):
        issues.append(f"|Z| median {z_med:.3g} Ω — shorted or dead channel")
        return QualityReport(Verdict.REJECT, issues, metrics)
    if z_med >= float(max_abs_z):
        issues.append(f"|Z| median {z_med:.3g} Ω — open circuit")
        return QualityReport(Verdict.REJECT, issues, metrics)

    # The *derived* open-circuit reading, and deliberately a SCREEN rather than a
    # refusal. `max_abs_z` above is an absolute backstop and on this rig it cannot
    # fire — the whole stored corpus has a median |Z| below 5.4e7 against its 1e12.
    # This one is keyed to what an unbridged cell physically presents, so it can.
    #
    # It appends an issue and falls through, which this function's own contract turns
    # into SUSPECT: "Rejects only what is physically impossible or unusable …
    # everything else that merely looks unusual is reported as SUSPECT". Rejecting
    # here would be wrong on the measured population — the spectra it flags are
    # five-week-old dried films, and a film too resistive to measure is an upper
    # BOUND on sigma, which is a result. It cannot distinguish that from a genuinely
    # empty well, and no threshold on |Z| can: both read as a near-pure capacitance.
    # Telling them apart needs provenance, which is the open admissibility question.
    z_open = open_circuit_z_ohm(freq[:n][finite], stray_c_f)
    if np.isfinite(z_open):
        metrics["z_open_circuit"] = z_open
        if z_med >= z_open:
            issues.append(
                f"|Z| median {z_med:.3g} Ω at or above the {float(stray_c_f) * 1e12:.1f} pF "
                f"open-circuit reading {z_open:.3g} Ω — nothing may be bridging the "
                "electrodes, or the film is beyond the measurable range"
            )

    if float(np.min(mag)) <= 0.0:
        issues.append("non-positive |Z| present")

    # A stuck instrument returns the same number at every frequency. Real
    # spectra always vary across a decade sweep.
    if n_finite > 2 and np.allclose(mag, mag[0], rtol=1e-9, atol=0.0):
        issues.append("|Z| identical at every frequency — instrument may be stuck")
        return QualityReport(Verdict.REJECT, issues, metrics)

    # Frequencies should be monotonic in one direction; interleaving points from
    # two sweeps would silently mix measurements.
    f_ok = freq[:n][finite]
    if f_ok.size > 2:
        d = np.diff(f_ok)
        if not (np.all(d > 0) or np.all(d < 0)):
            issues.append("frequency axis is not monotonic")

    verdict = Verdict.SUSPECT if issues else Verdict.ACCEPT
    return QualityReport(verdict, issues, metrics)


@dataclass
class _RawTrace:
    """Minimal trace view over a raw instrument array (no EISResult available)."""

    frequency: Any
    z_real: Any
    z_imag_neg: Any


def validate_raw_eis(raw: Any, **kw: Any) -> QualityReport:
    """Validate the raw array an EIS step returns, without building an EISResult.

    The autonomous path never materialises an ``EISResult`` — it reads the
    instrument's array directly — so the gate has to meet the data where it is
    rather than forcing a conversion that could itself fail. Column convention
    matches the objective extractor: the last two columns are Z' and -Z''.
    """
    if raw is None:
        return QualityReport(Verdict.REJECT, ["no measurement returned"])
    try:
        arr = np.asarray(
            raw[0] if isinstance(raw, (list, tuple)) else raw, dtype=float)
    except Exception as exc:
        return QualityReport(Verdict.REJECT, [f"unreadable measurement: {exc}"])

    if arr.size == 0:
        return QualityReport(Verdict.REJECT, ["empty measurement"])
    if arr.ndim < 2 or arr.shape[1] < 2:
        # A 1-D result carries no impedance structure to check; let the
        # extractor's own finite-check decide rather than inventing a verdict.
        return QualityReport(Verdict.SUSPECT, ["measurement has no Z' / Z'' columns"])

    z_real, z_imag = arr[:, -2], arr[:, -1]
    freq = arr[:, 0] if arr.shape[1] >= 3 else np.arange(len(z_real), dtype=float)
    return validate_eis_trace(_RawTrace(freq, z_real, z_imag), **kw)


# ── 4.1 Fit-quality metrics ─────────────────────────────────────────────────

def compute_fit_quality(
    eis_result: Any, z_fit: Any, n_params: int = 0
) -> dict[str, float]:
    """Goodness-of-fit metrics for a fitted circuit model.

    Residuals were previously computed for *plotting only*, so nothing recorded
    whether a fit actually described the data — a non-converged fit still
    reported an ``R1``, and the conductivity derived from it looked like every
    other number in the table.

    Returns ``chi2``, ``chi2_reduced``, ``r_squared`` (on the complex trace),
    ``residual_rms_pct``, and ``residual_max_pct``. Empty when the inputs cannot
    support a comparison.
    """
    if z_fit is None:
        return {}
    try:
        measured = np.asarray(eis_result.z_real, dtype=float) - 1j * np.asarray(
            eis_result.z_imag_neg, dtype=float)
        fitted = np.asarray(z_fit, dtype=complex)
    except Exception:
        return {}

    n = int(min(measured.size, fitted.size))
    if n == 0:
        return {}
    measured, fitted = measured[:n], fitted[:n]

    good = np.isfinite(measured) & np.isfinite(fitted) & (np.abs(measured) > 0)
    if int(good.sum()) < 2:
        return {}
    measured, fitted = measured[good], fitted[good]

    resid = measured - fitted
    # Modulus-weighted chi-square: impedance spans decades, so an unweighted sum
    # would be dominated by the low-frequency end regardless of fit quality.
    weight = np.abs(measured) ** 2
    chi2 = float(np.sum((np.abs(resid) ** 2) / weight))

    dof = max(1, int(measured.size) - int(n_params))
    ss_res = float(np.sum(np.abs(resid) ** 2))
    ss_tot = float(np.sum(np.abs(measured - measured.mean()) ** 2))

    pct = np.abs(resid) / np.abs(measured) * 100.0
    return {
        "chi2": chi2,
        "chi2_reduced": chi2 / dof,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "residual_rms_pct": float(np.sqrt(np.mean(pct ** 2))),
        "residual_max_pct": float(np.max(pct)),
    }


def grade_fit(
    metrics: dict[str, float],
    *,
    success: bool = True,
    min_r_squared: float = DEFAULT_MIN_R_SQUARED,
    max_residual_pct: float = DEFAULT_MAX_RESIDUAL_PCT,
) -> QualityReport:
    """Turn fit metrics into a verdict.

    A fit that did not converge is rejected outright. A converged fit that does
    not describe the data is also rejected — its ``R1`` is a fitting artefact, and
    the conductivity computed from it would enter the campaign as a real
    observation. **Missing metrics are not a rejection**: an older record or an
    unavailable ``z_fit`` means unknown quality, not bad quality, and inventing a
    failure would discard good data.
    """
    if not success:
        return QualityReport(Verdict.REJECT, ["fit did not converge"], dict(metrics))
    if not metrics:
        return QualityReport(Verdict.SUSPECT, ["fit quality unknown"], {})

    issues: list[str] = []
    r2 = metrics.get("r_squared")
    rms = metrics.get("residual_rms_pct")

    if r2 is not None and r2 < float(min_r_squared):
        issues.append(f"R²={r2:.3f} below {float(min_r_squared):.3f}")
    if rms is not None and rms > float(max_residual_pct):
        issues.append(f"RMS residual {rms:.1f}% above {float(max_residual_pct):.1f}%")

    verdict = Verdict.REJECT if issues else Verdict.ACCEPT
    return QualityReport(verdict, issues, dict(metrics))


# ── 4.3 Combined gate ───────────────────────────────────────────────────────

def quality_config(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Resolve thresholds from ``[quality]`` — the single parse point."""
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("quality", {}) or {}
        except Exception:
            config = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "min_r_squared": _f("min_r_squared", DEFAULT_MIN_R_SQUARED),
        "max_residual_pct": _f("max_residual_pct", DEFAULT_MAX_RESIDUAL_PCT),
        "min_points": _f("min_points", DEFAULT_MIN_POINTS),
        "min_abs_z": _f("min_abs_z", DEFAULT_MIN_ABS_Z),
        "max_abs_z": _f("max_abs_z", DEFAULT_MAX_ABS_Z),
        # Deliberately NOT read from `[quality]`. The stray capacitance is a fact about
        # the instrument, it already has a home in `[eis.instrument]`, and copying it
        # into a second section is how two numbers for one quantity start to drift.
        "stray_c_f": _stray_c_from_config(),
    }


def _stray_c_from_config() -> float:
    """``[eis.instrument] stray_C_instrument_F``, or the fallback if unreadable.

    Read from the instrument section rather than ``[quality]`` because that is where
    the measurement lives — the same value the EIS engine uses, with the blank-sweep
    provenance recorded beside it. A ``[quality]`` copy would be a second number for
    one physical quantity, free to drift from the one the blanks actually produced.
    """
    try:
        from softae.config import loader

        section = loader.load().get("eis", {}).get("instrument", {}) or {}
        value = float(section.get("stray_C_instrument_F", DEFAULT_STRAY_C_F))
    except Exception:
        return DEFAULT_STRAY_C_F
    return value if np.isfinite(value) and value > 0.0 else DEFAULT_STRAY_C_F


def gate_raw_measurement(
    raw: Any, *, config: dict[str, Any] | None = None
) -> QualityReport:
    """Gate a raw EIS step result — the autonomous path's entry point.

    Honours ``[quality] enabled``: while disabled, every check still runs and a
    would-be rejection is logged, but the measurement is used. That lets the gate
    be observed against real campaigns before it is allowed to discard data.
    """
    cfg_all: dict[str, Any]
    if config is None:
        try:
            from softae.config import loader

            cfg_all = loader.load().get("quality", {}) or {}
        except Exception:
            cfg_all = {}
    else:
        cfg_all = config

    enabled = bool(cfg_all.get("enabled", False))
    cfg = quality_config(cfg_all)

    report = validate_raw_eis(
        raw,
        min_points=int(cfg["min_points"]),
        min_abs_z=cfg["min_abs_z"],
        max_abs_z=cfg["max_abs_z"],
        stray_c_f=cfg["stray_c_f"],
    )

    if report.verdict is Verdict.REJECT and not enabled:
        logger.warning(
            "quality_gate_would_reject", issues=report.issues,
            metrics=report.metrics,
            msg="gate disabled — measurement used despite failing checks",
        )
        return QualityReport(
            Verdict.SUSPECT, report.issues + ["gate disabled"], report.metrics)

    if report.verdict is Verdict.REJECT:
        logger.warning("quality_gate_reject", issues=report.issues,
                       metrics=report.metrics)
    return report


def gate_measurement(
    eis_result: Any,
    fit_result: Any = None,
    *,
    config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> QualityReport:
    """Full accept/suspect/reject decision for one measurement.

    The trace is checked first: if it is unusable, the fit derived from it cannot
    rescue it. ``enabled=False`` short-circuits to ACCEPT while still reporting
    metrics, so the gate can be observed on real runs before it is given
    authority over data.
    """
    cfg = quality_config(config)

    trace = validate_eis_trace(
        eis_result,
        min_points=int(cfg["min_points"]),
        min_abs_z=cfg["min_abs_z"],
        max_abs_z=cfg["max_abs_z"],
        stray_c_f=cfg["stray_c_f"],
    )
    issues = list(trace.issues)
    metrics = dict(trace.metrics)
    verdict = trace.verdict

    if verdict is not Verdict.REJECT and fit_result is not None:
        fit_metrics = getattr(fit_result, "quality", None) or {}
        fit = grade_fit(
            fit_metrics,
            success=bool(getattr(fit_result, "success", True)),
            min_r_squared=cfg["min_r_squared"],
            max_residual_pct=cfg["max_residual_pct"],
        )
        issues.extend(fit.issues)
        metrics.update(fit.metrics)
        if fit.verdict is Verdict.REJECT:
            verdict = Verdict.REJECT
        elif fit.verdict is Verdict.SUSPECT and verdict is Verdict.ACCEPT:
            verdict = Verdict.SUSPECT

    if not enabled and verdict is Verdict.REJECT:
        logger.warning(
            "quality_gate_would_reject", issues=issues, metrics=metrics,
            msg="gate disabled — measurement used despite failing checks",
        )
        return QualityReport(Verdict.SUSPECT, issues + ["gate disabled"], metrics)

    if verdict is Verdict.REJECT:
        logger.warning("quality_gate_reject", issues=issues, metrics=metrics)
    elif issues:
        logger.info("quality_gate_suspect", issues=issues)

    return QualityReport(verdict, issues, metrics)
