"""What may be said about this spectrum — a value, a bound, or nothing.

Overhaul §3.3 is the failure this module exists to prevent. A dry, salt-dilute film
produced a spectrum whose conduction was absent at every frequency, and the defensible
output was ``σ ≲ 4×10⁻⁷ S/cm`` — an upper bound, not a fitted value. A pipeline that
cannot express the difference will report the bound as a measurement, and a campaign
optimising against it will chase instrument noise.

So conductivity is reported through a small vocabulary rather than as a bare float:

``value``
    Measured, with an uncertainty.
``bound``
    Resolution-limited, with a *qualified* ceiling — phase accuracy was measured.
``bound_unqualified``
    Resolution-limited, ceiling *provisional* — phase accuracy is still an estimate.
``unavailable``
    No per-sample thickness, so no cell constant, so no conductivity. Not an error.

.. warning::
   The switch **degrades toward caution**, but the reason has changed. An earlier
   version fell back to a magnitude proxy against ``Z_φ ≈ 5×10⁷ Ω``; that ceiling is
   **withdrawn** (it was a floating-reference-electrode artefact), and nothing here
   uses it.

   What replaces it is narrower and defensible. Phase noise *has* now been measured —
   0.149° on a 9.9 kΩ resistive load, giving a ``tan δ`` floor of 0.0026 — so a
   genuine headroom comparison is possible. But films sit at 10⁶–10⁸ Ω and are
   capacitive, three decades from where that number was taken. Inside the calibrated
   band a bound is **qualified**; outside it the same bound is **provisional**, because
   extrapolating an instrument constant across three decades without saying so is
   exactly how the withdrawn ceiling came to be believed in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Fraction of surviving points above ``Z_φ`` beyond which an unqualified envelope
#: forces a bound. Half is deliberate: it is the point past which the *typical* point
#: in the spectrum is untrustworthy, not merely the tail.
DEFAULT_ABOVE_CEILING_FRAC = 0.5


@dataclass(frozen=True)
class SigmaReport:
    """Conductivity, the resistance it came from, and what may be claimed about it."""

    mode: str = "unavailable"
    value: float = float("nan")
    upper_bound: float = float("nan")
    rel_uncertainty: float = float("nan")
    provisional: bool = False

    R_reported_ohm: float = float("nan")
    R_reported_se_ohm: float = float("nan")
    R_basis: str = "split_bulk"       # "sum" | "split_bulk"
    rho: float = float("nan")

    K_per_cm: float = float("nan")
    K_route: str = "geometric"
    thickness_method: str = "unavailable"
    #: Electrode configuration behind ``K_config_factor`` (framework §1.1, R20).
    #: A wiring fact, recorded independently of whether the factor was confirmed.
    electrode_config: str = "unverified"
    k_config_factor: float = 1.0
    #: False ⇒ the *absolute scale* of σ is unqualified: F16 is a clean ~2× error with
    #: no other symptom. **Relative trends stay valid regardless**, since the factor is
    #: constant across a series — so a campaign ranking formulations is unaffected.
    config_factor_verified: bool = False
    #: Whether this sample had a verified ionic path to the reference stripe (R26).
    #: Reported separately from :attr:`config_factor_verified` because it is the one
    #: precondition that varies sample to sample on an otherwise verified board.
    re_contact_verified: bool = False

    model_free_R_ohm: float = float("nan")
    cross_check_pct: float = float("nan")
    phase_headroom: float = float("nan")

    @property
    def is_bound(self) -> bool:
        return self.mode in ("bound", "bound_unqualified")

    @property
    def is_value(self) -> bool:
        return self.mode == "value"

    def as_text(self) -> str:
        """Operator-facing rendering — ``1.2e-04 ±8%`` or ``≲ 4.0e-07 (provisional)``."""
        if self.mode == "unavailable":
            return "σ unavailable"
        tag = " (provisional)" if self.provisional else ""
        if self.is_bound:
            return f"σ ≲ {self.upper_bound:.2g} S/cm{tag}"
        unc = (
            f" ±{self.rel_uncertainty * 100:.0f}%"
            if self.rel_uncertainty == self.rel_uncertainty else ""
        )
        return f"{self.value:.3g} S/cm{unc}{tag}"

    def describe(self) -> str:
        basis = "R_series+R_bulk" if self.R_basis == "sum" else "R_bulk"
        if self.config_factor_verified:
            cfg = f", {self.electrode_config} ÷{self.k_config_factor:g}"
        else:
            why = ("no verified RE contact"
                   if self.electrode_config == "3-electrode"
                   and not self.re_contact_verified else "unverified")
            cfg = (f", {self.electrode_config}, K_config_factor {why} — "
                   f"absolute scale unqualified (relative trends unaffected)")
        return (
            f"{self.as_text()} from {basis} = {self.R_reported_ohm:.4g} Ω, "
            f"K = {self.K_per_cm:.1f} /cm [{self.K_route}], "
            f"t from {self.thickness_method}{cfg}"
        )


@dataclass(frozen=True)
class SpectrumReport:
    """One spectrum's complete analysis — the single return shape of both engines.

    Both engines return this so that DataStore, the analysis tab and the browser each
    learn exactly one new type and never branch on which engine produced it. Flipping
    ``[eis] engine`` is then the whole cutover, and it is reversible per run.
    """

    engine: str
    fit: Any = None                                  # circuit_fitting.FitResult
    sigma: SigmaReport = field(default_factory=SigmaReport)
    quality: Any = None                              # quality.QualityReport
    gate_log: tuple[dict[str, Any], ...] = ()
    mask: np.ndarray | None = None
    cell: Any = None                                 # geometry.CellConstant
    envelope: Any = None
    #: What was subtracted as fixture, and what that did (E3). ``None`` on the legacy
    #: engine, which corrects nothing — and a ``mode = "none"`` correction on the gated
    #: one, which is a different statement: *considered, and deliberately not applied*.
    correction: Any = None           # fixture.FixtureCorrection
    correction_outcome: Any = None   # fixture.CorrectionOutcome

    @property
    def corrected(self) -> bool:
        """Whether fixture impedance was actually removed from *this* spectrum.

        Reads the **outcome**, not the correction. ``correction.applies`` says only
        that the constants exist and are usable — a spectrum rejected at admission
        carries exactly such a correction and was never corrected by it, because §6
        places the subtraction downstream of the gates that rejected it.
        """
        return bool(getattr(self.correction_outcome, "applied", False))

    @property
    def ok(self) -> bool:
        """True when the measurement may be used at all (mirrors ``QualityReport.ok``)."""
        return bool(getattr(self.quality, "ok", True))

    @property
    def n_dropped(self) -> int:
        return int(sum(e.get("n_dropped", 0) for e in self.gate_log))

    def gate_summary(self) -> str:
        """The one-cell rendering for the analysis tab's ``Gate`` column."""
        if self.engine != "gated":
            return "—"
        for entry in self.gate_log:
            if not entry.get("passed", True) and entry.get("severity") in (
                "block_spectrum", "block_session"
            ):
                return f"REJECTED: {entry.get('gate', '?')}"
        dropped = self.n_dropped
        return f"{dropped} dropped" if dropped else "pass"

    def describe(self) -> str:
        return f"[{self.engine}] {self.gate_summary()} — {self.sigma.describe()}"


def decide_report_mode(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
    tand_headroom_mult: float = 3.0,
) -> tuple[str, bool, float]:
    """Decide value vs bound. Returns ``(mode, provisional, phase_headroom)``.

    The comparison is always the same one (framework §4.8) —
    ``headroom = tan δ_measured / tan(ε)``, and below ``tand_headroom_mult`` the
    extracted conductance sits at or under the instrument's resolution. What varies is
    whether ``ε`` may be *trusted at this impedance*:

    * ``ε`` measured **and** the spectrum inside its calibrated band → ``value`` or a
      qualified ``bound``.
    * ``ε`` measured but the spectrum decades away from where it was characterised →
      the same decision, marked **provisional**. Films sit at 10⁶–10⁸ Ω against a
      10⁴ Ω resistive characterisation, and an instrument constant carried three
      decades without comment is how the withdrawn ``Z_φ`` ceiling was born.
    * ``ε`` unmeasured → provisional bound. Never a value.
    """
    if cell is None:
        return "unavailable", False, float("nan")

    from softae.analysis.eis.admittance import loss_tangent

    Z = np.asarray(Z, dtype=complex)
    mag = np.abs(Z)
    finite = np.isfinite(mag)
    z_med = float(np.median(mag[finite])) if finite.any() else float("nan")

    measured = bool(getattr(envelope, "phase_noise_measured", False))
    in_band = bool(
        getattr(envelope, "phase_noise_valid_at", lambda _z: False)(z_med))
    floor = getattr(envelope, "tand_floor", float("nan"))

    if not measured or not (floor == floor) or floor <= 0:
        logger.info("eis_reported_as_bound",
                    reason="phase noise unmeasured", provisional=True)
        return "bound_unqualified", True, float("nan")

    tand = loss_tangent(Z)
    tand = tand[np.isfinite(tand) & (tand > 0)]
    if tand.size == 0:
        return "bound_unqualified", True, float("nan")

    headroom = float(np.median(tand)) / floor
    resolution_limited = headroom < float(tand_headroom_mult)

    if resolution_limited:
        logger.info(
            "eis_reported_as_bound", reason="loss tangent below the phase floor",
            phase_headroom=headroom, z_median_ohm=z_med, provisional=not in_band,
        )
        return ("bound" if in_band else "bound_unqualified"), not in_band, headroom

    if not in_band:
        logger.info(
            "eis_phase_floor_extrapolated", z_median_ohm=z_med,
            phase_headroom=headroom,
            msg="value reported, but the phase floor it cleared is extrapolated",
        )
    return "value", not in_band, headroom


def sigma_upper_bound(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
) -> float:
    """The defensible ceiling ``σ ≲ ε·ω·C·K`` when the loss is below the phase floor.

    A spectrum whose loss sits under the instrument's resolution still yields a
    rigorous statement, and refusing to make it would throw away a real result.

    Evaluated at the **lowest** usable frequency in the spectrum, not the median: the
    bound scales with ω, so the tightest defensible ceiling is the one at the bottom of
    the sweep. That is the same reason the updated envelope favours low-frequency
    sweeps — ``σ_min`` runs from ~1e-9 S/cm at 1 Hz to ~1e-7 S/cm at 100 Hz.

    Falls back to what the magnitude ceiling licenses (``K/Z_max``) only when ε is
    unavailable — never to the withdrawn ``K/Z_φ``.
    """
    if cell is None:
        return float("nan")
    K = getattr(cell, "K_per_cm", float("nan"))
    if not (K == K):
        return float("nan")

    eps = getattr(envelope, "eps_rad", float("nan"))
    if eps == eps:
        C = float("nan")
        try:
            from softae.analysis.eis.admittance import par_capacitance_estimate

            C = par_capacitance_estimate(freq, Z)
        except Exception:
            C = float("nan")
        f = np.asarray(freq, dtype=float)
        f = f[np.isfinite(f) & (f > 0)]
        if C == C and f.size:
            omega = 2.0 * math.pi * float(np.min(f))
            return float(eps * omega * C * K)

    z_max = float(getattr(envelope, "z_max_ohm", float("nan")))
    return K / z_max if z_max == z_max and z_max > 0 else float("nan")
