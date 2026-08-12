"""Subtracting what the fixture contributes, and refusing to when it cannot be known.

A measurement made through a mux, a ribbon and a PCB trace records the cell *and* the
path to it. Framework §4 places fixture correction at step 4 of the pipeline, between
admission and Kramers–Kronig, and R8 restricts it to what the artifacts license.

**Series-only, deliberately.** The obvious richer correction — open/short/load, "OSL" —
is *not implemented here*, and its absence is a finding rather than a gap. Overhaul F6
records OSL corrupting every spectrum on this fixture: a mean error of 32 %, with one
channel reading 1.26 MΩ against a true ~840 Ω. The open blank is still worth measuring,
because an *unusable* open is the positive evidence that shunt admittance is negligible
— which is precisely the condition under which short-only series correction is exact
(§3.9). So the open's job here is to **select** the fallback, not to be applied.

That produces one hazard worth naming, because two nearby things disagree on purpose.
:meth:`~softae.analysis.eis.calibration.CalibrationSet.capabilities` reports
``correction_mode = "osl"`` when the open is usable — truthfully, because it describes
what the *artifacts* support. This module answers a different question: what the
*engine* will apply. :func:`resolve_mode` therefore clamps ``auto`` to ``series``
whenever a short exists, records why in ``declined``, and logs it once. A usable open
must never silently select a correction that does not exist.

**The correction is allowed to fail loudly.** A subtraction cannot be validated by
looking at its output — a wrong ``R_short`` yields a spectrum that is merely shifted,
and shifted spectra look fine. What it *can* do is announce when the subtraction did
something a fixture cannot do: move a point out of the physical quadrant it was in
beforehand. That check costs nothing and turns the plan's acceptance criterion — "a
deliberately corrupted correction produces a visible failure rather than a plausible
wrong result" — into something the log actually shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: What may be configured. ``auto`` resolves from the calibration's capability ladder,
#: which is the point: correction switches on the moment a short blank exists, without
#: an operator remembering to flip a key, and stays off when one does not.
CORRECTION_MODES = ("auto", "none", "series")

#: What may actually be *applied*. ``osl`` is deliberately absent — see the module
#: docstring. Keeping the two tuples separate is what makes the clamp explicit rather
#: than an oversight.
APPLICABLE_MODES = ("none", "series")

#: Corrected-vs-raw disagreement (%) above which the correction is worth a second look.
#: Not an error: a short blank of a few ohms against a film of megohms *should* shift
#: nothing, so a large shift means either a low-impedance sample or a wrong constant.
DEFAULT_SHIFT_NOTABLE_PCT = 5.0

#: End-to-end load-blank error (%) beyond which the correction is not trusted (R9).
DEFAULT_LOAD_TOLERANCE_PCT = 5.0


@dataclass(frozen=True)
class FixtureCorrection:
    """The constants that will be subtracted from one channel's spectrum.

    Constructed once per channel and reused across its spectra. ``mode = "none"`` is a
    valid, fully-formed correction meaning *subtract nothing* — an uncorrected spectrum
    is honest, and representing "no correction" as ``None`` everywhere would push that
    decision into every call site.
    """

    mode: str = "none"
    channel: int = -1
    fixture_id: str = ""
    R_short_ohm: float = float("nan")
    L_lead_H: float = float("nan")
    #: Constants inherited from a representative channel rather than measured here.
    inherited: bool = False
    #: Why a richer mode than ``mode`` was not applied. Empty when nothing was declined.
    declined: str = ""
    #: Provenance for the ``fixture_corrections`` row (R17/§7.5).
    source_measurement_id: int | None = None

    @property
    def applies(self) -> bool:
        """Whether this correction changes any number at all."""
        return self.mode == "series" and math.isfinite(self.R_short_ohm)

    def describe(self) -> str:
        if not self.applies:
            base = f"fixture correction: none (ch{self.channel})"
            return f"{base} — {self.declined}" if self.declined else base
        via = " inherited" if self.inherited else ""
        text = (
            f"fixture correction: series (ch{self.channel}{via}), "
            f"R = {self.R_short_ohm:.4g} Ω, L = {self.L_lead_H:.4g} H"
        )
        return f"{text} — {self.declined}" if self.declined else text


@dataclass
class CorrectionOutcome:
    """What the subtraction actually did to this spectrum."""

    applied: bool = False
    n_points: int = 0
    max_shift_pct: float = float("nan")
    median_shift_pct: float = float("nan")
    #: Points the correction pushed *out* of the quadrant they occupied beforehand.
    #: Non-zero means the subtracted constants are inconsistent with this measurement.
    induced_nonphysical: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def suspect(self) -> bool:
        return self.induced_nonphysical > 0


# ── The subtraction itself ───────────────────────────────────────────────────

def fixture_impedance(f: np.ndarray, *, R_short_ohm: float, L_lead_H: float) -> np.ndarray:
    """``Z_fixture(f) = R + jωL``, in the physics convention (``Im Z < 0`` capacitive).

    The sign matches :func:`~softae.analysis.eis.calibration.derive_short`, which fits
    ``L`` as the slope of ``Im Z`` against ``ω`` with no intercept. A lead is inductive,
    so its ``Im Z`` is positive — the opposite sign to the film it is in series with,
    which is why leaving it in inflates the apparent HF response rather than cancelling.
    """
    omega = 2.0 * math.pi * np.asarray(f, dtype=float)
    L = 0.0 if not math.isfinite(L_lead_H) else float(L_lead_H)
    return np.full(omega.shape, float(R_short_ohm), dtype=complex) + 1j * omega * L


def apply_series_correction(
    f: np.ndarray, Z: np.ndarray, correction: FixtureCorrection
) -> tuple[np.ndarray, CorrectionOutcome]:
    """Subtract the fixture's series contribution. Returns ``(Z_corrected, outcome)``.

    A correction that does not apply returns the input array untouched — not a copy
    with the same numbers, the same array — so "uncorrected" is genuinely a no-op and
    cannot introduce a rounding difference into the legacy comparison.
    """
    Zc = np.asarray(Z, dtype=complex)
    if not correction.applies:
        return Zc, CorrectionOutcome(applied=False, n_points=int(Zc.size))

    Z_fix = fixture_impedance(
        f, R_short_ohm=correction.R_short_ohm, L_lead_H=correction.L_lead_H)
    Z_out = Zc - Z_fix

    mag = np.abs(Zc)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(mag > 0, np.abs(Z_out - Zc) / mag * 100.0, np.nan)
    finite = np.isfinite(shift)

    # The one check a subtraction can honestly make about itself. A fixture is a small
    # series impedance; removing it cannot turn a capacitive point resistive-inductive
    # or drive Re Z negative. If it did, the constants do not belong to this spectrum.
    was_physical = np.isfinite(Zc) & (Zc.real > 0)
    now_broken = was_physical & (Z_out.real <= 0)
    induced = int(np.count_nonzero(now_broken))

    outcome = CorrectionOutcome(
        applied=True,
        n_points=int(Zc.size),
        max_shift_pct=float(np.max(shift[finite])) if np.any(finite) else float("nan"),
        median_shift_pct=(float(np.median(shift[finite])) if np.any(finite)
                          else float("nan")),
        induced_nonphysical=induced,
    )

    if induced:
        msg = (f"fixture correction drove Re Z ≤ 0 at {induced} point(s) that were "
               f"physical before it — R_short = {correction.R_short_ohm:.4g} Ω is "
               f"not consistent with this spectrum")
        outcome.issues.append(msg)
        logger.warning(
            "eis_fixture_correction_suspect", channel=correction.channel,
            fixture=correction.fixture_id, n_points=induced,
            R_short_ohm=correction.R_short_ohm, msg=msg,
        )
    elif outcome.max_shift_pct == outcome.max_shift_pct and \
            outcome.max_shift_pct >= DEFAULT_SHIFT_NOTABLE_PCT:
        logger.info(
            "eis_fixture_corrected", channel=correction.channel,
            fixture=correction.fixture_id,
            max_shift_pct=outcome.max_shift_pct,
            median_shift_pct=outcome.median_shift_pct,
        )
    return Z_out, outcome


# ── Choosing the mode ────────────────────────────────────────────────────────

def resolve_mode(configured: str, capabilities: Any) -> tuple[str, str]:
    """``(mode_to_apply, why_something_richer_was_declined)``.

    The clamp lives here and nowhere else. ``capabilities.correction_mode`` may say
    ``"osl"``; this function never returns it, because OSL is not implemented and
    silently downgrading without saying so is how a spectrum ends up labelled
    "corrected" by a correction that never ran.
    """
    want = (configured or "auto").strip().lower()
    if want not in CORRECTION_MODES:
        logger.warning(
            "eis_fixture_mode_unknown", mode=want, known=CORRECTION_MODES,
            msg="falling back to auto",
        )
        want = "auto"

    if want == "none":
        return "none", "disabled by configuration"

    available = getattr(capabilities, "correction_mode", "none") if capabilities else "none"

    if available == "none":
        return "none", "no short blank — run softae-commission run blank_short"

    if available == "osl":
        # A usable open licenses OSL, which this rig has evidence against (F6).
        return "series", (
            "OSL is licensed by the artifacts but not applied: it corrupted every "
            "spectrum on this fixture (overhaul F6, mean error 32 %)")

    if available == "series":
        return "series", ""

    # A capability this module does not implement. Unreachable against today's ladder,
    # which yields only none/series/osl — kept so that adding a fourth mode there
    # fails closed here rather than falling through to a correction that never runs.
    return "none", f"correction mode '{available}' is not implemented"


def correction_for_channel(
    channel: int,
    *,
    calibration: Any,
    configured: str = "auto",
) -> FixtureCorrection:
    """Build the correction for one channel from a resolved calibration.

    A ``None`` calibration, a stale one (whose constants ``resolve_calibration`` has
    already dropped), or a channel with a non-finite constant all produce a
    ``mode = "none"`` correction carrying the reason — never a silent no-op, and never
    a ``NaN`` subtracted into every point of a spectrum.
    """
    ch = int(channel)
    if calibration is None:
        return FixtureCorrection(
            mode="none", channel=ch,
            declined="no calibration for this fixture — run softae-commission")

    caps = calibration.capabilities()
    mode, declined = resolve_mode(configured, caps)
    fixture_id = str(getattr(calibration, "fixture_id", ""))

    if mode == "none":
        return FixtureCorrection(
            mode="none", channel=ch, fixture_id=fixture_id, declined=declined)

    constants = calibration.for_channel(ch)
    R = float(constants.get("R_short_ohm", float("nan")))
    if not math.isfinite(R):
        # The short blank exists but not for this channel and not by inheritance.
        return FixtureCorrection(
            mode="none", channel=ch, fixture_id=fixture_id,
            declined=(f"no short-blank constant for ch{ch} — measure it, or re-derive "
                      f"with --channels and --representative to inherit one"))

    return FixtureCorrection(
        mode="series",
        channel=ch,
        fixture_id=fixture_id,
        R_short_ohm=R,
        L_lead_H=float(constants.get("L_lead_H", float("nan"))),
        inherited=ch in tuple(getattr(calibration, "channels_assumed", ())),
        declined=declined,
        source_measurement_id=(getattr(calibration, "sources", {}) or {}).get(
            "blank_short"),
    )


# ── Resolving one, from configuration, without re-reading the disk 32 times ──

#: ``(fixture_id, hardware_hash) -> CalibrationSet | None``. Keyed by the hash so a
#: hardware change invalidates the entry on its own — the cache cannot serve constants
#: from a board that is no longer installed, which is the failure it would otherwise
#: introduce into the very mechanism designed to prevent it.
_CAL_CACHE: dict[tuple[str, str], Any] = {}


def clear_calibration_cache() -> None:
    """Drop the cached calibrations. For tests, and for a re-derive within a session."""
    _CAL_CACHE.clear()


def resolve_correction(
    channel: int,
    *,
    settings: Any = None,
    calibration: Any = None,
) -> FixtureCorrection:
    """The correction for *channel*, resolved from configuration and the calibration.

    Pass *calibration* to bypass the lookup entirely; otherwise the canonical TOML is
    loaded through :func:`~softae.analysis.eis.calibration.resolve_calibration`, which
    has already dropped the constants if the hardware has changed.
    """
    from softae.analysis.eis.settings import eis_settings

    cfg = settings if settings is not None else eis_settings().fixture

    if calibration is None:
        from softae.analysis.eis.calibration import hardware_hash, resolve_calibration

        key = (cfg.fixture_id, hardware_hash())
        if key not in _CAL_CACHE:
            _CAL_CACHE[key] = resolve_calibration(cfg.fixture_id)
        calibration = _CAL_CACHE[key]

    return correction_for_channel(
        channel, calibration=calibration, configured=cfg.mode)


# ── Validating it end to end ─────────────────────────────────────────────────

def validate_load_blank(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    nominal_ohm: float,
    correction: FixtureCorrection,
    tolerance_pct: float = DEFAULT_LOAD_TOLERANCE_PCT,
) -> tuple[bool, float, str]:
    """Push a known resistor through the correction and see what comes out (R9).

    Returns ``(within_tolerance, error_pct, message)``. This is the *only* end-to-end
    check the correction has: the short blank determines the constants, so measuring
    the short again proves nothing. A third component whose value is known independently
    is what turns "the arithmetic ran" into "the arithmetic was right".

    An error far outside tolerance is a setup fact, not a numerical one — a wrong
    resistor in the socket, a channel mismatch, or the correction being applied to a
    fixture it was not taken on.
    """
    from softae.analysis.eis.calibration import derive_reference_r

    nominal = float(nominal_ohm)
    if not (nominal > 0):
        return False, float("nan"), "no marked value — nothing to validate against"

    Z_corr, outcome = apply_series_correction(f, Z, correction)
    R, err, _noise = derive_reference_r(f, Z_corr, nominal_ohm=nominal)

    if not math.isfinite(err):
        return False, err, "corrected load blank yielded no resistance"

    ok = abs(err) <= float(tolerance_pct)
    verb = "within" if ok else "outside"
    msg = (f"corrected load reads {R:.4g} Ω against a marked {nominal:.4g} Ω "
           f"({err:+.2f} %), {verb} ±{tolerance_pct:g} %")
    if outcome.suspect:
        msg += f"; {outcome.issues[0]}"
    logger.info("eis_fixture_load_validated", ok=ok, error_pct=err,
                measured_ohm=R, nominal_ohm=nominal,
                mode=correction.mode, channel=correction.channel)
    return ok, err, msg
