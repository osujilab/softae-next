"""Commissioning constants: acquired once, derived, persisted, reused across campaigns.

Every "blocked on bench data" item in the overhaul reduces to the same sentence: *run
this workflow once*. This module owns what that workflow produces — the fixture's short
resistance and lead inductance, whether its open is usable at all, the measured phase
accuracy, and the impedance window this particular fixture actually reproduces.

**A calibration is a durable asset, not a per-run chore.** Fixture electronics drift is
minimal, so a set stays valid until the *hardware* changes. That is why staleness is
keyed on a :func:`hardware_hash` rather than a clock: a set is good indefinitely and
worthless the instant the routing changes, and no elapsed-time rule expresses that.

**Blanks are measurements, not a parallel type.** A blank is an EIS spectrum with the
same columns, the same file format and the same conditions row as a sample; only its
``role`` differs. So acquisition reuses ``eis_measure_step``, ``EISResult.save`` and
``record_measurement`` unchanged, and this module reads back what they wrote.

.. warning::
   **``Z_φ`` is not here, deliberately.** Earlier drafts had this module derive a
   "phase-reliable ceiling" ≈ 5×10⁷ Ω. That ceiling is **withdrawn** — it was an
   artefact of a floating reference electrode (overhaul §3.7, F13), and with RE
   correctly connected no negative ``Re Z`` occurs anywhere in the band. What replaces
   it is :class:`PhaseAccuracyTable`: ``ε`` as a function of ``|Z|`` and frequency,
   measured where the samples actually sit, with **no** extrapolation past the
   impedances that were characterised.

.. note::
   Calibration is **incremental, never all-or-nothing**. Each artifact unlocks specific
   capabilities and :class:`CalibrationCapabilities` reports which — and, more usefully,
   names the one artifact that would unblock each thing still missing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Roles a measurement row may carry. ``sample`` is the schema default, so every
#: pre-existing row is already correct without a backfill.
MEASUREMENT_ROLES = (
    "sample",
    "drift_repeat",
    "blank_open",
    "blank_short",
    "blank_load",
    "reference_cap",
    "reference_r",
)

#: Roles that are commissioning artifacts rather than data.
#:
#: Listed explicitly rather than as "everything except ``sample``". That subtraction
#: was correct until ``drift_repeat`` existed and would have quietly enrolled it —
#: putting a re-measured film into ``softae-commission``'s prompts and its capability
#: ladder, where it means nothing.
COMMISSIONING_ROLES = (
    "blank_open",
    "blank_short",
    "blank_load",
    "reference_cap",
    "reference_r",
)

#: A ``drift_repeat`` is one series member measured **a second time at the end of the
#: session**, against which its own first measurement is differenced (framework §5.6's
#: protocol). It exists because the geometry route is immune to fixture error but not
#: to *session* drift: 16 channels at ~40 s each is ~11 minutes during which the films
#: keep equilibrating with the chamber, and if measurement order correlates with
#: thickness that equilibration becomes a false slope.
#:
#: It is a distinct role rather than a flag on ``sample`` for one practical reason: the
#: geometry-series fit selects spectra with ``role = 'sample'`` and takes the most
#: recent per channel, so a repeat left tagged ``sample`` would **replace** the
#: in-sequence measurement and silently move that channel to the wrong point in time.
#: A separate role excludes it from the regression by the filter that already exists.
DRIFT_REPEAT_ROLE = "drift_repeat"

#: Fixture-correction modes, least to most presumptuous.
CORRECTION_MODES = ("none", "series", "osl")

#: Roles whose *marked* value must be supplied alongside the measurement, and its unit.
#:
#: Required rather than optional because the marking and the measurement disagreeing is
#: the check that catches an unusable part — overhaul §3.7's capacitor marked "102"
#: (1 nF) measured ~150 nF with tan δ = 0.18. With only one of the two numbers, nothing
#: would have flagged it.
ARTIFACT_NOMINAL_UNITS = {
    "blank_load": "ohms",
    "reference_r": "ohms",
    "reference_cap": "farads",
}

#: How the cell was sensed for a spectrum. ``unknown`` is the honest default for any
#: row recorded before the mode was asked for — never assume it was correct.
ELECTRODE_MODES = ("unknown", "two", "three")

#: Roles that are **two-terminal loads with no ionic path to the reference stripe**,
#: and therefore must be measured with RE tied to CE (two-electrode).
#:
#: Overhaul §3.10 / F17. In three-electrode mode the reference floats onto a capacitive
#: divider between WE and CE, and the instrument reports only a fraction of the true
#: WE–CE impedance. The fraction is **not a constant**: measured on one instrument in a
#: single session it ranged 2.2 → 23, because a floating RE draws current through its
#: own stray-coupling network and so competes with whatever load is present.
#:
#: A load-dependent divider cannot be corrected for — there is no α to divide out and
#: none may be fitted. So a reference measured this way is *uncalibratable in principle*
#: rather than merely uncalibrated, and R24 forbids the value entering ``INSTRUMENT``.
#:
#: This is also what resolves a long run of "impossible" results: parts reading 8–9×
#: their markings and blanks at 123–250 pF. Re-measured with RE tied to CE, a part
#: marked "101" gave 94.5 pF and one marked "102" gave 986.6 pF — both within 6 % of
#: their EIA codes. **The components were correct all along**, and two earlier drafts
#: blaming mismarked parts are withdrawn.
TWO_TERMINAL_ROLES = frozenset({
    "blank_short", "blank_load", "blank_open", "reference_cap", "reference_r",
})


def electrode_mode_ok(role: str, mode: str) -> tuple[bool, str]:
    """Whether *role* measured in *mode* may enter the instrument envelope (R24).

    Returns ``(ok, reason)``. A sample is unconstrained — a conductive film in contact
    with the reference stripe *does* establish an ionic path, which is exactly the
    condition under which three-electrode sensing is valid and ``K_config_factor = 2``
    is exact. The restriction applies only to loads that cannot establish one.
    """
    r, m = str(role or "sample"), str(mode or "unknown")
    if r not in TWO_TERMINAL_ROLES:
        return True, ""
    if m == "two":
        return True, ""
    if m == "three":
        return False, (
            f"'{r}' was measured in three-electrode mode. A two-terminal load has no "
            f"ionic path to RE, so the reference floats onto a load-dependent "
            f"capacitive divider (F17: alpha 2.2-23). The value is uncalibratable, not "
            f"merely uncalibrated — re-measure with RE tied to CE."
        )
    return False, (
        f"'{r}' has no recorded electrode mode. It predates the check, and assuming it "
        f"was two-electrode would invent the one fact that decides whether it is usable "
        f"(F17). Re-measure with RE tied to CE, or re-import declaring the mode."
    )

#: Largest resistance (Ω) a *short* blank may plausibly show. The measured fixture is
#: ≈5.4 Ω; three orders of headroom above that still leaves anything above this
#: unambiguously not a short.
MAX_PLAUSIBLE_SHORT_OHM = 1.0e3
#: Largest lead inductance (H) a short blank may plausibly show. The measured value is
#: 4.18 µH, and F5's *fitted* 400–500 µH was already an artifact — so a millihenry is
#: far beyond anything real on this fixture.
MAX_PLAUSIBLE_L_LEAD_H = 1.0e-3


def short_is_plausible(R_ohm: float, L_H: float) -> tuple[bool, str]:
    """Whether a short blank's derived constants can describe an actual short.

    Returns ``(ok, reason)``. Added after a mock run wrote ``R = 3.2 MΩ`` and
    ``L = −1.47 H`` — a *negative inductance* — into a version-controlled
    calibration file without a word of complaint.

    A short blank is the one commissioning artifact whose correct answer is known in
    advance: a few ohms and a few microhenries. Anything else means the jumper is not
    installed, the wrong channel was swept, or the run was mislabelled — all operator
    setup errors, all recoverable, and all invisible later if the numbers are simply
    recorded. R9 makes a failed *load* check block the session; this is the same
    discipline one artifact earlier, where it is cheaper.
    """
    if R_ohm != R_ohm or L_H != L_H:
        return False, "derivation produced NaN — the trace is unusable"
    if R_ohm < 0:
        return False, f"negative resistance ({R_ohm:.4g} Ω) is not physical"
    if R_ohm > MAX_PLAUSIBLE_SHORT_OHM:
        return False, (
            f"{R_ohm:.4g} Ω is not a short (expected a few ohms) — is the jumper "
            f"installed on this channel?"
        )
    if L_H < 0:
        return False, (
            f"negative inductance ({L_H:.4g} H) is not physical — this trace is not "
            f"a short blank"
        )
    if L_H > MAX_PLAUSIBLE_L_LEAD_H:
        return False, (
            f"{L_H * 1e6:.4g} µH exceeds anything this fixture can carry (measured "
            f"4.18 µH); suspect an HF phase artifact rather than real inductance"
        )
    return True, ""

#: Config sections whose contents define "the same hardware". A change to any of them
#: invalidates a calibration; a change to anything else (thresholds, objectives, GUI
#: preferences) does not, which is why this is a curated list rather than the whole file.
HARDWARE_SECTIONS = ("pcb", "channels", "instruments", "eis.instrument")


def hardware_hash(config: Mapping[str, Any] | None = None) -> str:
    """A stable digest of the hardware a calibration was taken on.

    Staleness is by hardware identity, not by clock (framework §8.5). Drift is
    minimal, so there is no expiry — but a short blank taken before a board swap must
    never be silently applied after one.

    Only :data:`HARDWARE_SECTIONS` contribute. Hashing the whole config would make a
    gate-threshold edit invalidate a perfectly good calibration, which trains operators
    to ignore staleness warnings — the failure mode this is meant to prevent.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load()
        except Exception:
            logger.warning("hardware_hash_config_unreadable", exc_info=True)
            config = {}

    material: dict[str, Any] = {}
    for section in HARDWARE_SECTIONS:
        node: Any = config
        for part in section.split("."):
            node = (node or {}).get(part) if isinstance(node, Mapping) else None
        if node is not None:
            material[section] = node

    blob = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ── Phase accuracy, measured rather than assumed ─────────────────────────────

@dataclass(frozen=True)
class FixtureConductance:
    """``Re(Y_fixture)`` as a function of frequency, for one channel.

    A table rather than a scalar for a measured reason. Seven tied open blanks on this
    fixture give ``d ln G / d ln f`` between **+0.87 and +1.04** — the fixture's real
    part is **dielectric loss** (``G = ωC·tan δ``, tan δ ≈ 0.05), not an ohmic leak,
    which would sit at 0. A single number therefore describes the fixture at exactly
    one frequency and is wrong everywhere else, and *which* frequency it came from is
    precisely what a scalar does not record.

    This is the ``G_fixture`` that framework §5.6 puts in the geometry-series intercept
    and that :meth:`~softae.analysis.eis.geometry_series.GeometrySeriesFit.dead_height_cm`
    needs. It matters most where it is largest relative to the sample: at 1 kHz the
    measured fixture is ~406% of the conductance of a σ = 10⁻⁷ S/cm film, and at 10 Hz
    about 30% — because ``G_fixture ∝ ω`` while ``G_bulk`` is flat. That ratio is the
    quantitative case for measuring low.
    """

    freq_hz: tuple[float, ...] = ()
    G_S: tuple[float, ...] = ()
    #: ``d ln G / d ln f``. ≈1 is dielectric loss, ≈0 an ohmic leak. NaN if unfitted.
    exponent: float = float("nan")

    @property
    def is_empty(self) -> bool:
        return not self.freq_hz or not self.G_S

    @property
    def is_dielectric(self) -> bool:
        """Whether the loss looks dielectric rather than ohmic."""
        return self.exponent == self.exponent and self.exponent > 0.5

    def at(self, freq_hz: float) -> float:
        """``G`` at a frequency, log-interpolated. **Never extrapolates** — NaN outside.

        Same refusal as :class:`PhaseAccuracyTable`: carrying a fixture constant beyond
        the band it was measured in is how the withdrawn ``Z_φ`` came to be believed.
        """
        if self.is_empty or not (freq_hz > 0):
            return float("nan")
        lf = math.log10(freq_hz)
        pts = sorted((math.log10(f), g) for f, g in zip(self.freq_hz, self.G_S)
                     if f > 0 and g == g)
        if not pts or lf < pts[0][0] or lf > pts[-1][0]:
            return float("nan")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return float(np.interp(lf, xs, ys))

    def as_mapping(self) -> dict[float, float]:
        return {f: g for f, g in zip(self.freq_hz, self.G_S)}

    def describe(self) -> str:
        if self.is_empty:
            return "G_fixture: not measured"
        kind = ("dielectric loss" if self.is_dielectric
                else "ohmic leak" if self.exponent == self.exponent else "unclassified")
        return (f"G_fixture: {min(self.G_S):.2e}–{max(self.G_S):.2e} S over "
                f"{min(self.freq_hz):g}–{max(self.freq_hz):g} Hz, "
                f"d ln G/d ln f = {self.exponent:+.2f} ({kind})")


@dataclass(frozen=True)
class PhaseAccuracyTable:
    """``ε`` (degrees) as a function of ``|Z|`` and frequency.

    Framework §5.3 asks for a *table*, not a scalar, and the reason is the whole point
    of E2: the one phase-noise number this rig has (0.149°) was taken on a 9.9 kΩ
    **resistive** load, while films sit at 10⁶–10⁸ Ω and are **capacitive**. Carrying
    that number three decades without saying so is precisely how the withdrawn ``Z_φ``
    ceiling came to be believed.

    So :meth:`epsilon_deg` refuses to extrapolate beyond the impedance decades actually
    characterised, returning ``NaN`` instead. A caller that gets ``NaN`` must report a
    *provisional* bound — never a value.
    """

    #: Impedance magnitudes (Ω) at which ε was measured, ascending.
    z_ohm: tuple[float, ...] = ()
    #: Phase accuracy (degrees) at each of those impedances.
    eps_deg: tuple[float, ...] = ()
    #: Load type each point was taken on — ``resistive`` or ``capacitive``.
    load: str = "resistive"
    #: Decades either side of a measured point where the value may be trusted.
    #:
    #: A **trust radius for interpolation**, not a count of how many decades the table
    #: covers — that is :attr:`decades_spanned`. The two were once conflated in review
    #: ("``valid_decades = 1.0`` while the sweep spans five"), and they answer different
    #: questions: this one says how far from a measured point ε may still be quoted,
    #: and it stays 1.0 because that is a property of the instrument, not of how many
    #: points happened to survive gating.
    valid_decades: float = 1.0

    @property
    def is_empty(self) -> bool:
        return not self.z_ohm or not self.eps_deg

    @property
    def decades_spanned(self) -> float:
        """How many decades of ``|Z|`` the tabulated points actually cover.

        The coverage number an operator wants when asking "does this calibration reach
        where my films sit?" — distinct from :attr:`valid_decades`, which is the
        per-point trust radius. NaN when nothing was tabulated.
        """
        zs = [z for z in self.z_ohm if z > 0]
        if len(zs) < 2:
            return float("nan")
        return float(math.log10(max(zs) / min(zs)))

    def covers(self, z_ohm: float) -> bool:
        """Whether *z_ohm* is within ``valid_decades`` of a characterised point."""
        if self.is_empty or not (z_ohm > 0):
            return False
        lz = math.log10(z_ohm)
        return any(
            abs(lz - math.log10(z)) <= self.valid_decades
            for z in self.z_ohm if z > 0
        )

    def epsilon_deg(self, z_ohm: float) -> float:
        """Phase accuracy at *z_ohm*, or ``NaN`` where it was never characterised.

        Log-interpolates between measured points. **Does not extrapolate** — outside
        the covered range the honest answer is "unknown", and a NaN forces the caller
        onto the provisional path rather than letting a fabricated ε qualify a bound.
        """
        if self.is_empty or not (z_ohm > 0) or not self.covers(z_ohm):
            return float("nan")
        zs = np.log10(np.asarray(self.z_ohm, dtype=float))
        es = np.asarray(self.eps_deg, dtype=float)
        order = np.argsort(zs)
        return float(np.interp(math.log10(z_ohm), zs[order], es[order]))

    def describe(self) -> str:
        if self.is_empty:
            return "phase accuracy: not measured"
        lo, hi = min(self.z_ohm), max(self.z_ohm)
        span = self.decades_spanned
        covered = f", spanning {span:.1f} decade(s)" if span == span else ""
        return (
            f"phase accuracy: {len(self.z_ohm)} point(s) on a {self.load} load, "
            f"{lo:.3g}–{hi:.3g} Ω{covered}, max ε {max(self.eps_deg):.2f}°, "
            f"±{self.valid_decades:g} decade(s) either side of each point"
        )


# ── What a partially-populated calibration can actually do ───────────────────

@dataclass(frozen=True)
class CalibrationCapabilities:
    """What this calibration unlocks, and what would unlock the rest.

    ``blocked`` is the operator-facing payoff. "Upper bounds are provisional" is a
    dead end; *"run the reference capacitor to qualify them"* is a task. The
    difference is what makes an incremental calibration worth shipping at all.
    """

    correction_mode: str = "none"
    phase_floor_measured: bool = False
    magnitude_window_measured: bool = False
    can_validate_correction: bool = False
    open_is_usable: bool = False
    #: capability → the single artifact that would unblock it.
    blocked: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        parts = [f"fixture correction: {self.correction_mode}"]
        if self.phase_floor_measured:
            parts.append("phase floor measured")
        if self.magnitude_window_measured:
            parts.append("|Z| window measured")
        if self.can_validate_correction:
            parts.append("correction validated against a load")
        if self.blocked:
            parts.append(
                "blocked — " + "; ".join(
                    f"{cap}: {fix}" for cap, fix in sorted(self.blocked.items())
                )
            )
        return "; ".join(parts)


# ── The set itself ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationSet:
    """Everything commissioning derived, for one fixture, at one hardware state."""

    fixture_id: str = "default"
    hardware_hash: str = ""
    created_at: str = ""
    channels_measured: tuple[int, ...] = ()
    #: Channels inheriting a representative channel's constants. Listed explicitly so
    #: using one can *log the assumption* — never a silent extrapolation.
    channels_assumed: tuple[int, ...] = ()

    R_short_ohm: dict[int, float] = field(default_factory=dict)
    L_lead_H: dict[int, float] = field(default_factory=dict)
    C_stray_F: dict[int, float] = field(default_factory=dict)
    open_usable: dict[int, bool] = field(default_factory=dict)
    #: Per-channel ``Re(Y_fixture)`` over frequency, from the open blank. A table
    #: rather than a scalar because on this fixture it is dielectric loss
    #: (``d ln G/d ln f`` ≈ 1), so one number describes one frequency only.
    G_fixture: dict[int, FixtureConductance] = field(default_factory=dict)

    phase_acc: PhaseAccuracyTable = field(default_factory=PhaseAccuracyTable)
    z_min_ohm: float = float("nan")
    z_max_ohm: float = float("nan")
    load_error_pct: float = float("nan")

    #: role → measurement_id, so every derived number can be traced to its spectrum (R17).
    sources: dict[str, int] = field(default_factory=dict)

    # ── Interrogation ────────────────────────────────────────────────────────

    @property
    def has_short(self) -> bool:
        return bool(self.R_short_ohm)

    @property
    def has_load(self) -> bool:
        return self.load_error_pct == self.load_error_pct

    @property
    def has_open(self) -> bool:
        return bool(self.open_usable)

    def open_is_usable(self, channel: int | None = None) -> bool:
        """Whether the open blank is a measurement rather than noise.

        An unusable open is **not a missing artifact** — it is the positive evidence
        that shunt admittance is negligible, which is exactly the condition under
        which short-only series correction is *exact* (framework §3.9). One free
        bare-board pass therefore answers "is OSL legitimate here?" with a defensible
        no, which is worth more than a missing answer.
        """
        if not self.open_usable:
            return False
        if channel is None:
            return any(self.open_usable.values())
        return bool(self.open_usable.get(int(channel), False))

    def capabilities(self) -> CalibrationCapabilities:
        """The ladder: what is unlocked, and the one artifact that unblocks each rest."""
        blocked: dict[str, str] = {}

        if self.has_short:
            mode = "osl" if self.open_is_usable() else "series"
        else:
            mode = "none"
            blocked["fixture correction"] = "run the short blank"

        if mode == "series" and self.has_open:
            # Not a limitation worth reporting: an unmeasurable open *selects* this.
            pass
        elif mode == "series":
            blocked["OSL correction"] = (
                "run the open blank (an unusable open is itself a valid answer)")

        phase = not self.phase_acc.is_empty
        if not phase:
            blocked["qualified upper bounds"] = (
                "run the reference capacitor — bounds stay provisional without it")

        window = self.z_min_ohm == self.z_min_ohm and self.z_max_ohm == self.z_max_ohm
        if not window:
            blocked["measured |Z| window"] = (
                "run the reference resistors (≥1 per decade)")

        if not self.has_load:
            blocked["correction validation"] = "run the load blank"

        return CalibrationCapabilities(
            correction_mode=mode,
            phase_floor_measured=phase,
            magnitude_window_measured=window,
            can_validate_correction=self.has_load,
            open_is_usable=self.open_is_usable(),
            blocked=blocked,
        )

    def is_stale(self, *, current_hash: str | None = None) -> bool:
        """Whether this set belongs to different hardware than the rig now has.

        An unknown hash on either side reads as **stale**: "we do not know what this
        was taken on" must degrade capabilities, not pass silently.
        """
        current = current_hash if current_hash is not None else hardware_hash()
        if not self.hardware_hash or not current:
            return True
        return self.hardware_hash != current

    def measured_spread(self, field_name: str = "C_stray_F") -> float:
        """max/min of a per-channel constant over the channels actually measured.

        The empirical size of what :attr:`channels_assumed` is assuming away. NaN when
        fewer than two channels were measured — with one channel there is nothing to
        compare, and reporting 1.0 would read as "no variation" rather than "unknown".
        """
        mapping = getattr(self, field_name, None)
        if not isinstance(mapping, Mapping):
            return float("nan")
        vals = [float(v) for ch, v in mapping.items()
                if ch in self.channels_measured
                and float(v) == float(v) and float(v) > 0]
        if len(vals) < 2:
            return float("nan")
        return float(max(vals) / min(vals))

    def for_channel(self, channel: int) -> dict[str, float]:
        """This channel's constants, logging when they are inherited rather than its own.

        The inheritance is **not** a formality on this fixture. Seven tied open blanks
        across nominally identical stripes (ch17–23, 2026-08-06) gave ``C_stray``
        spanning 10.2–24.7 pF — a 2.4× spread — while repeating to 1% on any single
        channel. So channel-to-channel variation is real and roughly an order of
        magnitude larger than the measurement error it would otherwise be mistaken for.
        The warning carries that number, because "inherited" without a magnitude reads
        as bookkeeping and gets skimmed.
        """
        ch = int(channel)
        if ch in self.channels_assumed:
            logger.warning(
                "eis_calibration_channel_assumed", channel=ch,
                measured=self.channels_measured,
                measured_spread=self.measured_spread("C_stray_F"),
                msg="constants inherited from a representative channel — measured "
                    "channel-to-channel C_stray spread on this fixture is 2.4x "
                    "(10.2-24.7 pF over 7 identical stripes), so this is a real "
                    "uncertainty, not a formality",
            )
        return {
            "R_short_ohm": float(self.R_short_ohm.get(ch, float("nan"))),
            "L_lead_H": float(self.L_lead_H.get(ch, float("nan"))),
            "C_stray_F": float(self.C_stray_F.get(ch, float("nan"))),
        }

    def envelope(self, base: Any = None) -> Any:
        """An :class:`InstrumentEnvelope` with whatever this set actually measured.

        Unmeasured quantities keep the configured estimate *and its
        ``*_measured = False`` flag*, so a partial calibration never silently
        promotes a guess to a measurement.
        """
        from softae.analysis.eis.envelope import InstrumentEnvelope, instrument_envelope

        env = base if base is not None else instrument_envelope()
        updates: dict[str, Any] = {}

        if not self.phase_acc.is_empty:
            # The lowest characterised impedance is the conservative headline value.
            idx = int(np.argmin(np.asarray(self.phase_acc.z_ohm, dtype=float)))
            updates["phase_noise_deg"] = float(self.phase_acc.eps_deg[idx])
            updates["phase_noise_at_ohm"] = float(self.phase_acc.z_ohm[idx])
            updates["phase_noise_load"] = self.phase_acc.load
            updates["phase_noise_valid_decades"] = float(self.phase_acc.valid_decades)
            updates["phase_noise_measured"] = True

        if self.z_min_ohm == self.z_min_ohm and self.z_max_ohm == self.z_max_ohm:
            updates["z_min_ohm"] = float(self.z_min_ohm)
            updates["z_max_ohm"] = float(self.z_max_ohm)
            updates["magnitude_window_measured"] = True

        if self.created_at:
            updates["measured_at"] = self.created_at

        if not updates:
            return env
        if isinstance(env, InstrumentEnvelope):
            return replace(env, **updates)
        return env

    def describe(self) -> str:
        caps = self.capabilities()
        n = len(self.channels_measured)
        assumed = (f" (+{len(self.channels_assumed)} assumed)"
                   if self.channels_assumed else "")
        return (
            f"calibration '{self.fixture_id}' @{self.hardware_hash or '?'} "
            f"[{self.created_at or 'undated'}], {n} channel(s){assumed}: "
            f"{caps.describe()}"
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """A plain mapping, keys stringified for TOML/JSON round-tripping."""
        return {
            "fixture_id": self.fixture_id,
            "hardware_hash": self.hardware_hash,
            "created_at": self.created_at,
            "channels_measured": list(self.channels_measured),
            "channels_assumed": list(self.channels_assumed),
            "R_short_ohm": {str(k): float(v) for k, v in self.R_short_ohm.items()},
            "L_lead_H": {str(k): float(v) for k, v in self.L_lead_H.items()},
            "C_stray_F": {str(k): float(v) for k, v in self.C_stray_F.items()},
            "open_usable": {str(k): bool(v) for k, v in self.open_usable.items()},
            "G_fixture": {
                str(k): {"freq_hz": list(v.freq_hz), "G_S": list(v.G_S),
                         "exponent": float(v.exponent)}
                for k, v in self.G_fixture.items()
            },
            "phase_acc": {
                "z_ohm": list(self.phase_acc.z_ohm),
                "eps_deg": list(self.phase_acc.eps_deg),
                "load": self.phase_acc.load,
                "valid_decades": self.phase_acc.valid_decades,
            },
            "z_min_ohm": self.z_min_ohm,
            "z_max_ohm": self.z_max_ohm,
            "load_error_pct": self.load_error_pct,
            "sources": {str(k): int(v) for k, v in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationSet":
        """Rebuild from :meth:`to_dict`, tolerating a partial or older mapping."""
        def _imap(key: str) -> dict[int, float]:
            out: dict[int, float] = {}
            for k, v in (data.get(key) or {}).items():
                try:
                    out[int(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out

        def _f(key: str) -> float:
            try:
                return float(data.get(key, float("nan")))
            except (TypeError, ValueError):
                return float("nan")

        pa = data.get("phase_acc") or {}
        phase = PhaseAccuracyTable(
            z_ohm=tuple(float(z) for z in (pa.get("z_ohm") or [])),
            eps_deg=tuple(float(e) for e in (pa.get("eps_deg") or [])),
            load=str(pa.get("load", "resistive")),
            valid_decades=float(pa.get("valid_decades", 1.0) or 1.0),
        )
        opens: dict[int, bool] = {}
        for k, v in (data.get("open_usable") or {}).items():
            try:
                opens[int(k)] = bool(v)
            except (TypeError, ValueError):
                continue
        gfix: dict[int, FixtureConductance] = {}
        for k, v in (data.get("G_fixture") or {}).items():
            try:
                gfix[int(k)] = FixtureConductance(
                    freq_hz=tuple(float(x) for x in (v.get("freq_hz") or [])),
                    G_S=tuple(float(x) for x in (v.get("G_S") or [])),
                    exponent=float(v.get("exponent", float("nan"))),
                )
            except (TypeError, ValueError, AttributeError):
                continue
        sources: dict[str, int] = {}
        for k, v in (data.get("sources") or {}).items():
            try:
                sources[str(k)] = int(v)
            except (TypeError, ValueError):
                continue

        return cls(
            fixture_id=str(data.get("fixture_id", "default")),
            hardware_hash=str(data.get("hardware_hash", "")),
            created_at=str(data.get("created_at", "")),
            channels_measured=tuple(
                int(c) for c in (data.get("channels_measured") or [])),
            channels_assumed=tuple(
                int(c) for c in (data.get("channels_assumed") or [])),
            R_short_ohm=_imap("R_short_ohm"),
            L_lead_H=_imap("L_lead_H"),
            C_stray_F=_imap("C_stray_F"),
            open_usable=opens,
            G_fixture=gfix,
            phase_acc=phase,
            z_min_ohm=_f("z_min_ohm"),
            z_max_ohm=_f("z_max_ohm"),
            load_error_pct=_f("load_error_pct"),
            sources=sources,
        )


# ── Derivation from acquired spectra ─────────────────────────────────────────

def derive_short(f: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """``(R_fixture, L_lead)`` from a shorted channel.

    ``R`` is the median ``Re Z`` — median rather than mean because a single railed
    point should not move a fixture constant. ``L`` comes from the slope of
    ``Im Z`` against ``ω``, which is what a series inductance *is*.

    R11 exists because F5 recorded fitted inductances of 400–500 µH against a short
    blank's true 4.18 µH: a HF phase artifact absorbed as inductance. Measuring L here
    is what licenses pinning it to ≈0 in the sample fit rather than letting the
    optimizer discover a fictitious one.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    good = np.isfinite(freq) & np.isfinite(Zc.real) & np.isfinite(Zc.imag) & (freq > 0)
    if not np.any(good):
        return float("nan"), float("nan")

    R = float(np.median(Zc.real[good]))
    omega = 2.0 * math.pi * freq[good]
    im = Zc.imag[good]
    if omega.size < 2:
        return R, float("nan")
    # Slope through the origin: L = <ωX>/<ω²>, least squares with no intercept, since
    # a short has no reason to carry one and fitting one absorbs real inductance.
    denom = float(np.sum(omega ** 2))
    L = float(np.sum(omega * im) / denom) if denom > 0 else float("nan")
    return R, L


def derive_open(
    f: np.ndarray, Z: np.ndarray, *, envelope: Any = None, gates: Any = None
) -> tuple[bool, float, float]:
    """``(usable, over_range_frac, im_flip_frac)`` for an open blank.

    Two orthogonal signatures, per framework §3.9: how much of the band sits above
    the magnitude ceiling, and how often ``Im Z`` changes sign. A smooth physical
    blank flips rarely; noise flips constantly.

    ⚠️ **An open cell inherently floats the reference electrode** (overhaul §3.7),
    which is the same condition that produced the withdrawn ``Z_φ``. A bare-board open
    on a three-electrode fixture is therefore a measurement of inter-stripe geometry,
    **not** a fixture open — a genuine one needs RE tied to CE at the connector. The
    verdict here is still meaningful (an unusable open selects series-only, which is
    the right answer either way), but ``usable = False`` on this hardware should be
    read as "not yet attempted properly" rather than "the fixture has no open".
    """
    from softae.analysis.eis.envelope import instrument_envelope
    from softae.analysis.eis.settings import eis_settings

    env = envelope if envelope is not None else instrument_envelope()
    cfg = gates if gates is not None else eis_settings().gates

    Zc = np.asarray(Z, dtype=complex)
    mag = np.abs(Zc)
    finite = np.isfinite(mag)
    if not np.any(finite):
        return False, 1.0, 1.0

    over = float(np.mean(mag[finite] > env.z_max_ohm))
    signs = np.sign(Zc.imag[finite])
    flips = float(np.mean(np.diff(signs) != 0)) if signs.size > 1 else 1.0
    usable = (over < cfg.blank_over_frac) and (flips < cfg.blank_flip_frac)
    return usable, over, flips


def derive_open_constants(
    f: np.ndarray, Z: np.ndarray, *, lo_hz: float = 1e2, hi_hz: float = 1e4
) -> tuple[float, FixtureConductance]:
    """``(C_stray, G_fixture(f))`` from an open blank.

    :func:`derive_open` returns only a *verdict* — usable or not. That left
    ``CalibrationSet.C_stray_F`` declared, serialised, read by ``for_channel``, and
    **written by nothing**: the fixture's two shunt constants were derivable from an
    artifact the module already collected and were simply never extracted. This is the
    missing producer.

    ``C_stray`` is the median of ``Im(Y)/ω`` over *lo_hz–hi_hz*, a band chosen to sit
    above the low-frequency phase floor (where ``Re Z`` goes negative on a near-ideal
    blank) and below the top of the sweep. ``G_fixture`` is kept as a **table over the
    whole band**, because on this fixture it is dielectric loss and varies with ω;
    see :class:`FixtureConductance`.

    Both are per channel. The measured channel-to-channel spread is 2.4×, so this is
    not a quantity to derive once and share — which is what ``channels_assumed`` warns
    about when it must be.
    """
    f = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = np.where(np.abs(Zc) > 0, 1.0 / np.where(np.abs(Zc) > 0, Zc, 1.0),
                     np.nan + 0j)
    omega = 2.0 * np.pi * f

    band = (f >= lo_hz) & (f <= hi_hz) & np.isfinite(np.abs(Y)) & (omega > 0)
    if not np.any(band):
        band = np.isfinite(np.abs(Y)) & (omega > 0)
    C = float(np.nanmedian((np.imag(Y) / omega)[band])) if np.any(band) else float("nan")

    # G is tabulated only where it is positive and finite. A negative Re(Y) is the
    # phase floor showing through on a near-ideal blank -- real, and meaningless as a
    # conductance, so it is dropped rather than clipped to zero: a floor of zero would
    # read as "no loss here" when the truth is "below what this instrument resolves".
    G_ok = np.isfinite(np.real(Y)) & (np.real(Y) > 0) & (f > 0)
    freqs = tuple(float(v) for v in f[G_ok])
    gs = tuple(float(v) for v in np.real(Y)[G_ok])

    exponent = float("nan")
    if len(freqs) >= 3:
        try:
            exponent = float(np.polyfit(np.log10(freqs), np.log10(gs), 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            exponent = float("nan")

    return C, FixtureConductance(freq_hz=freqs, G_S=gs, exponent=exponent)


@dataclass(frozen=True)
class ReferenceLoad:
    """What a reference component's own physics says its spectrum must look like.

    The gates in :func:`derive_phase_table` are only as good as the expectation they
    compare against, and that expectation is a property of the *part*, not of the
    function. A capacitor falls at ``d log|Z| / d log f = −1`` and lives in the fourth
    quadrant; a resistor is flat and on the real axis. Naming the expectation rather
    than hardcoding "capacitor" is what lets a resistive reference ladder use the same
    tabulation later without a second copy of it.
    """

    #: Name recorded on :attr:`PhaseAccuracyTable.load`.
    name: str
    #: Expected ``d log|Z| / d log f``. −1 for a capacitor, 0 for a resistor.
    log_slope: float
    #: Required sign of ``Im Z``: −1 capacitive, +1 inductive, 0 unconstrained.
    im_sign: int
    #: Whether ``Re Z`` must be positive. A passive part dissipates; it never sources.
    positive_real: bool = True


CAPACITIVE_REFERENCE = ReferenceLoad("capacitive", log_slope=-1.0, im_sign=-1)
RESISTIVE_REFERENCE = ReferenceLoad("resistive", log_slope=0.0, im_sign=0)

REFERENCE_LOADS = {
    "capacitive": CAPACITIVE_REFERENCE,
    "resistive": RESISTIVE_REFERENCE,
}

#: How far a point's local ``d log|Z| / d log f`` may sit from its load's expectation
#: before it is treated as saturation or a range-switch artifact rather than a
#: measurement.
#:
#: 0.5 is chosen from the measured sweeps rather than by taste. On this rig's reference
#: capacitor the *good* band holds |slope + 1| ≲ 0.2, the instrument's ~1.0147 GΩ input
#: rail shows as a plateau at slope ≈ 0 (deviation 1.0), and a range-switch step between
#: adjacent points reads |slope| ≳ 2 (deviation ≳ 1.0). 0.5 therefore sits in the empty
#: middle: it admits every genuine point with margin and excludes both failure shapes.
PHASE_TABLE_SLOPE_TOL = 0.5


def _log_log_slope(freq: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """``d log|Z| / d log f`` at each point, by central differences across the sweep.

    Points are sorted by frequency first, because the instrument reports descending and
    a gradient taken in report order would come back sign-flipped.
    """
    slope = np.full(freq.shape, np.nan, dtype=float)
    if freq.size < 3:
        return slope
    order = np.argsort(freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope[order] = np.gradient(np.log10(mag[order]), np.log10(freq[order]))
    return slope


def phase_table_gate(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    load: "str | ReferenceLoad" = CAPACITIVE_REFERENCE,
    slope_tol: float = PHASE_TABLE_SLOPE_TOL,
) -> np.ndarray:
    """Mask of points a *load* reference may legitimately contribute to a phase table.

    Three gates, in order of how badly the ungated version failed on real data:

    **Finiteness.** ``|Z|`` and the loss angle must exist and be positive. This was the
    only gate the function had.

    **Quadrant.** ``tan δ = |Re Z| / |Im Z|`` takes absolute values, so a point in the
    wrong quadrant — a passive capacitor reading ``Re Z < 0``, or ``Im Z > 0`` — does
    not fail, it produces a *large* ε and is then averaged in as though it were a
    measured loss. On the mux16 reference capacitor that turned instrument-noise points
    at the top of the sweep into 34–45° "phase accuracy". A quadrant violation is not a
    lossy measurement; it is not a measurement, and it is dropped.

    **Saturation.** A reference capacitor obeys ``|Z| = 1/(2πfC)``, i.e. a log-log slope
    of exactly −1. Where the instrument rails at its input-impedance ceiling the sweep
    stops following that law and *plateaus* — slope → 0 — which is what the mux16 record
    shows at ~1.0147 GΩ, entering the table as both a 44.96° point and a 0.45° one from
    the same railed magnitude. Detecting the departure from the part's own power law
    needs no ceiling constant, so it also catches a rail at a different level, on a
    different instrument, or a mid-sweep range switch.

    .. note::
       **Failure directions are deliberately asymmetric.** Over-dropping costs table
       coverage — fewer decades characterised, ``epsilon_deg`` returning NaN more often,
       and callers pushed onto the provisional-bound path. Under-dropping puts a
       non-measurement into the phase *floor*, which silently qualifies spectra that
       should have stayed provisional. The first is visible and recoverable; the second
       is neither, so ``slope_tol`` is set to over-drop.

       Two known over-drops: the two points either side of a genuine range switch lose
       their local slope to it, and a sweep of fewer than three points gets no slope at
       all — there the saturation gate abstains rather than dropping everything, since
       with no neighbours there is no plateau to see.
    """
    if isinstance(load, str):
        # Refused rather than defaulted: a mistyped load name would silently apply a
        # capacitor's expectation to a resistor and empty the table, which reads as
        # "nothing survived gating" — a plausible result, and the wrong one.
        if load not in REFERENCE_LOADS:
            raise ValueError(
                f"unknown reference load {load!r}; expected one of "
                f"{sorted(REFERENCE_LOADS)}")
        spec = REFERENCE_LOADS[load]
    else:
        spec = load

    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    mag = np.abs(Zc)

    ok = np.isfinite(mag) & (mag > 0) & np.isfinite(freq) & (freq > 0)
    ok &= np.isfinite(Zc.real) & np.isfinite(Zc.imag)

    if spec.positive_real:
        ok &= Zc.real > 0
    if spec.im_sign < 0:
        ok &= Zc.imag < 0
    elif spec.im_sign > 0:
        ok &= Zc.imag > 0

    slope = _log_log_slope(freq, mag)
    railed = np.isfinite(slope) & (np.abs(slope - spec.log_slope) > float(slope_tol))
    return ok & ~railed


def derive_phase_table(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    per_decade: bool = True,
    load: "str | ReferenceLoad" = CAPACITIVE_REFERENCE,
    slope_tol: float = PHASE_TABLE_SLOPE_TOL,
) -> tuple[list[float], list[float]]:
    """``(|Z| points, phase-error bounds in degrees)`` from one reference component.

    **A single capacitor populates the whole table.** Swept 4 Hz–200 kHz, a 1 nF part
    traverses ``|Z|`` from ~800 Ω to ~40 MΩ — four and a half decades, which is most of
    the working range. R25 asks for a table over ``|Z|`` "populated from reference
    components spanning the working decades", and one component spans them by virtue of
    the sweep. Reducing that sweep to a single number throws the span away.

    The statistic per decade is the **median** loss angle, not the minimum.

    That distinction is the whole correctness of this function. The measured loss angle
    is an *upper bound* on the instrument's phase error, because it also contains the
    reference part's own loss — which is the conservative direction a gate wants. But
    the **minimum** across a sweep is not a bound on anything: it is the single luckiest
    point, where noise happened to cancel. Taking it on this rig's 1 nF C0G yields
    ``tan δ = 7e-5``, i.e. 0.004°, roughly a hundred times tighter than the 0.2–0.5°
    the same data supports per decade — and a phase floor that small would qualify
    almost any spectrum as a measured value rather than a bound, which is precisely the
    §3.3 failure the value-vs-bound machinery exists to prevent.

    **A median is only as good as what it is a median of**, which is why
    :func:`phase_table_gate` runs first. The median defends against one unlucky point;
    it does not defend against a *population* of railed or wrong-quadrant points, and on
    the mux16 record there were enough of both to move whole decades — 7 of 24 tabulated
    points sat above 30°, and the two extremes of the table were the instrument's input
    rail rather than the capacitor. Gating first and taking the median second are
    complementary, not alternatives.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    mag = np.abs(Zc)
    with np.errstate(divide="ignore", invalid="ignore"):
        tand = np.abs(Zc.real) / np.abs(Zc.imag)
    eps = np.degrees(np.arctan(tand))

    ok = phase_table_gate(freq, Zc, load=load, slope_tol=slope_tol) & np.isfinite(eps)
    if not np.any(ok):
        return [], []
    if int(np.size(ok)) - int(np.count_nonzero(ok)):
        logger.info("eis_phase_table_gated", kept=int(np.count_nonzero(ok)),
                    total=int(np.size(ok)))
    mag, eps = mag[ok], eps[ok]

    if not per_decade:
        return [float(np.median(mag))], [float(np.median(eps))]

    decade = np.floor(np.log10(mag)).astype(int)
    z_pts: list[float] = []
    e_pts: list[float] = []
    for d in sorted(set(decade.tolist())):
        m = decade == d
        if not np.any(m):
            continue
        z_pts.append(float(np.median(mag[m])))
        e_pts.append(float(np.median(eps[m])))
    return z_pts, e_pts


@dataclass(frozen=True)
class ReferenceCapResult:
    """What one reference-capacitor sweep says, raw and stray-corrected.

    Unpacks as the ``(C, tand_min, z_at_tand_min)`` triple :func:`derive_reference_cap`
    has always returned — the same affordance
    :class:`~softae.workflows.commissioning.AcquiredSpectrum` uses — so every existing
    caller is unaffected. The extra fields exist so the marked-value check can *show its
    working*: "149.8 pF disagrees with a 100 pF marking" and "96.6 pF agrees with it,
    once the fixture's 53 pF shunt is taken off" are the same measurement, and only the
    second is a statement about the part.
    """

    #: Median ``1/(ω|Im Z|)`` over the sweep — the part **plus** whatever shunts it.
    C_raw_F: float
    tand_min: float
    z_at_tand_min_ohm: float
    #: The fixture's stray shunt, or NaN when none was measured for this channel.
    C_stray_F: float = float("nan")

    @property
    def corrected(self) -> bool:
        """Whether a usable stray was supplied — i.e. whether the correction ran."""
        return self.C_stray_F == self.C_stray_F

    @property
    def C_corrected_F(self) -> float:
        """The part alone: raw minus the parallel stray. NaN without a stray."""
        return self.C_raw_F - self.C_stray_F

    @property
    def C_F(self) -> float:
        """The value the marked-value check judged — corrected where one was possible."""
        return self.C_corrected_F if self.corrected else self.C_raw_F

    def __iter__(self) -> Iterator[float]:
        return iter((self.C_F, self.tand_min, self.z_at_tand_min_ohm))


def derive_reference_cap(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    nominal_F: float | None = None,
    C_stray_F: float | None = None,
) -> ReferenceCapResult:
    """What a reference capacitor measured, checked against what it is marked.

    The single most valuable and most-skipped commissioning artifact (§7.4): it is the
    only route to a *measured* ``ε`` where the samples actually sit.

    Overhaul §3.7 is also the cautionary tale. The capacitor marked "102" (decoding to
    1 nF) measured ~150 nF with a minimum ``tan δ`` of 0.18 — 70× above the instrument
    floor. Whatever it was, it was unusable as a phase reference. So this returns the
    *measured* capacitance alongside the loss, and comparing them against the marking is
    exactly the check that would have caught it.

    **The check is run against the corrected value when there is one.** The fixture's
    stray capacitance sits in **parallel** with the part, so what the sweep sees is
    ``C_part + C_stray`` and the part's own capacitance is
    ``Im(Y)/ω − C_stray``. On this rig the stray is ~53 pF against 100 pF parts, so the
    uncorrected check reads 1.50× and flags two perfectly good C0G capacitors while
    telling the operator to re-read a part code that was right all along. Pass
    *C_stray_F* — the same per-channel number :func:`derive_open_constants` produces —
    and the check judges the part rather than the part plus the fixture. Omit it and
    the behaviour is exactly as before: the raw value is checked, and the report says so.

    Subtracting the stray from the **median** rather than from each point is not an
    approximation of the per-point correction, it is identical to it: the stray is one
    constant, and ``median(x_i − c) = median(x_i) − c`` for any constant. Per-point
    would matter only if the correction varied across the sweep, which a fixed shunt
    capacitance does not.

    A stray that is NaN, zero or negative is treated as **absent**, not as zero. Those
    are the shapes :func:`derive_open_constants` returns from a trace it could not read,
    and a fixture with literally no shunt is not a thing this hardware produces —
    silently subtracting nothing would report "corrected" for a correction that never
    happened.

    .. note::
       This correction is **local to the marked-value check**, deliberately. The
       production fixture correction is series-only by design (see
       ``analysis/eis/fixture.py``): it has no shunt term to carry this, and giving it
       one would reopen the OSL path that corrupted whole spectra. Nothing outside this
       function's report is corrected here.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    good = (
        np.isfinite(freq) & (freq > 0)
        & np.isfinite(Zc.real) & np.isfinite(Zc.imag) & (Zc.imag < 0)
    )

    stray = float(C_stray_F) if C_stray_F is not None else float("nan")
    if not (stray > 0):
        stray = float("nan")

    if not np.any(good):
        return ReferenceCapResult(float("nan"), float("nan"), float("nan"), stray)

    omega = 2.0 * math.pi * freq[good]
    C = 1.0 / (omega * np.abs(Zc.imag[good]))
    tand = np.abs(Zc.real[good]) / np.abs(Zc.imag[good])

    k = int(np.argmin(tand))
    result = ReferenceCapResult(
        C_raw_F=float(np.median(C)),
        tand_min=float(tand[k]),
        z_at_tand_min_ohm=float(np.abs(Zc[good][k])),
        C_stray_F=stray,
    )

    judged = result.C_F
    if nominal_F is not None and nominal_F > 0 and judged == judged:
        ratio = judged / float(nominal_F)
        if ratio > 2.0 or ratio < 0.5:
            logger.warning(
                "eis_reference_cap_mismatch", measured_F=judged,
                C_raw_F=result.C_raw_F,
                C_corrected_F=result.C_corrected_F if result.corrected else None,
                C_stray_F=result.C_stray_F if result.corrected else None,
                stray_corrected=result.corrected,
                nominal_F=float(nominal_F), ratio=ratio,
                msg="measured capacitance disagrees with the marking — re-read the "
                    "part code and confirm on a meter before trusting it as a "
                    "phase reference"
                    + ("" if result.corrected else
                       " (no open blank on this channel, so this is the RAW value: "
                       "the fixture's parallel stray is still in it)"),
            )
    return result


def derive_reference_r(
    f: np.ndarray, Z: np.ndarray, *, nominal_ohm: float
) -> tuple[float, float, float]:
    """``(R_measured, error_pct, phase_noise_deg)`` from a reference resistor.

    A resistor's phase is 0° by definition, so the *scatter* of its measured phase is
    a direct read of ``ε`` at that impedance — which is what makes a resistor ladder
    the cheap route to a magnitude window and a phase floor at the same time.
    """
    Zc = np.asarray(Z, dtype=complex)
    good = np.isfinite(Zc.real) & np.isfinite(Zc.imag)
    if not np.any(good):
        return float("nan"), float("nan"), float("nan")

    R = float(np.median(Zc.real[good]))
    err = ((R - nominal_ohm) / nominal_ohm * 100.0) if nominal_ohm else float("nan")
    phase = np.degrees(np.angle(Zc[good]))
    return R, err, float(np.std(phase))


# ── Persistence ──────────────────────────────────────────────────────────────

def calibration_path(fixture_id: str, *, root: Path | None = None) -> Path:
    """Where a fixture's canonical calibration lives.

    Version-controlled beside the code, because framework §8.5 requires commissioning
    data to travel with the software and be re-validated after a hardware change. The
    database keeps history; *this* file is what a checkout reproduces.
    """
    base = root if root is not None else Path("calibration") / "eis"
    return base / f"{fixture_id}.toml"


def save_calibration(
    calibration: CalibrationSet, *, root: Path | None = None
) -> Path:
    """Write the canonical TOML. Returns the path written."""
    path = calibration_path(calibration.fixture_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_to_toml(calibration.to_dict()), encoding="utf-8")
    logger.info("eis_calibration_saved", path=str(path),
                fixture=calibration.fixture_id,
                hardware_hash=calibration.hardware_hash)
    return path


def load_calibration(
    fixture_id: str = "default",
    *,
    root: Path | None = None,
    path: Path | None = None,
) -> CalibrationSet | None:
    """Load a calibration, or ``None`` when there is none to load.

    ``None`` is a legitimate state, not an error: the envelope then falls back to
    ``[eis.instrument]``'s estimates *with their ``measured = False`` flags intact*,
    which is exactly how an uncalibrated rig should behave.
    """
    target = path if path is not None else calibration_path(fixture_id, root=root)
    if not target.exists():
        logger.info("eis_calibration_absent", path=str(target),
                    msg="falling back to configured estimates, flagged unmeasured")
        return None
    try:
        import tomllib

        data = tomllib.loads(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("eis_calibration_unreadable", path=str(target), exc_info=True)
        return None
    return CalibrationSet.from_dict(data)


def resolve_calibration(
    fixture_id: str = "default",
    *,
    root: Path | None = None,
    path: Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> CalibrationSet | None:
    """Load a calibration and **degrade it if the hardware has changed**.

    A stale set is not applied. Returning it unchanged would let a short blank from a
    different board correct today's spectra — silently, and with every number looking
    plausible. Instead the fixture constants are dropped and the capability ladder
    falls back to "run the short blank", which is the truth.
    """
    cal = load_calibration(fixture_id, root=root, path=path)
    if cal is None:
        return None

    current = hardware_hash(config)
    if not cal.is_stale(current_hash=current):
        return cal

    logger.warning(
        "eis_calibration_stale", fixture=fixture_id,
        recorded=cal.hardware_hash or "(none)", current=current,
        msg="hardware changed since this calibration — constants dropped rather "
            "than applied to a different fixture",
    )
    return replace(
        cal, R_short_ohm={}, L_lead_H={}, C_stray_F={}, open_usable={},
        phase_acc=PhaseAccuracyTable(), z_min_ohm=float("nan"),
        z_max_ohm=float("nan"), load_error_pct=float("nan"),
    )


def _to_toml(data: Mapping[str, Any]) -> str:
    """Minimal TOML writer — the stdlib reads TOML but does not write it.

    Deliberately not a dependency, but it must handle **nested** tables. The first
    version emitted one level and fell through to ``json.dumps(str(v))`` for anything
    deeper, so ``G_fixture`` — a mapping of per-channel tables — serialised as quoted
    Python reprs and was silently discarded on load. The file looked fine and the
    calibration came back missing a field, which is the worst shape a persistence bug
    can take. Recursion costs four lines and removes the whole class of it.
    """
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            if v != v:
                return "nan"
            return repr(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_fmt(x) for x in v) + "]"
        return json.dumps(str(v))

    def _emit(node: Mapping[str, Any], prefix: str, tables: list[str]) -> list[str]:
        """Return this table's scalar lines, appending nested tables to *tables*."""
        own: list[str] = []
        nested: list[tuple[str, Mapping[str, Any]]] = []
        for k, v in node.items():
            if isinstance(v, Mapping):
                nested.append((str(k), v))
            else:
                own.append(f"{k} = {_fmt(v)}")
        for k, v in nested:
            # TOML permits digit-only bare keys, which is what channel numbers are.
            path = f"{prefix}.{k}" if prefix else str(k)
            body = _emit(v, path, tables)
            tables.append(f"\n[{path}]\n" + "\n".join(body))
        return own

    tables: list[str] = []
    scalars = _emit(data, "", tables)

    header = (
        "# EIS commissioning calibration — generated, but version-controlled.\n"
        "# Framework §8.5: commissioning data travels with the code and is\n"
        "# re-validated after any hardware change. Staleness is by hardware_hash,\n"
        "# not by date: fixture drift is minimal, so there is no expiry.\n"
    )
    return header + "\n".join(scalars) + "\n" + "\n".join(tables) + "\n"


def describe_or_absent(calibration: CalibrationSet | None) -> str:
    """One startup line, whether or not a calibration exists."""
    if calibration is None:
        return (
            "EIS calibration: none — using configured estimates (flagged unmeasured). "
            "Run commissioning to measure the fixture: short blank → load blank → "
            "reference capacitor, in that order of value per hour of bench time."
        )
    return calibration.describe()
