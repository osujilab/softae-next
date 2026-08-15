"""Commissioning contracts: the roles, the guards, and the shapes everything shares.

Every "blocked on bench data" item in the overhaul reduces to the same sentence: *run
this workflow once*. Three modules own what that workflow produces, and this one owns
the vocabulary the other two are written in:

``calibration`` (here)
    Which roles a measurement may carry, which of them are commissioning artifacts,
    which must be sensed two-terminal, whether a short blank is plausible at all, and
    :func:`hardware_hash` — the identity a calibration is keyed to. Then the three
    frozen contracts every consumer passes around: :class:`FixtureConductance`,
    :class:`PhaseAccuracyTable`, :class:`CalibrationCapabilities`.
``calibration_derive``
    Turning an acquired spectrum into a number: the short, the open, the phase table,
    the reference capacitor and resistor.
``calibration_set``
    :class:`~softae.analysis.eis.calibration_set.CalibrationSet` — everything derived,
    for one fixture at one hardware state — and its persistence.

**Every name in all three is importable from here.** See the re-export block at the
foot of this module: ``from softae.analysis.eis.calibration import X`` is the published
spelling and the split is invisible to callers.

**A calibration is a durable asset, not a per-run chore.** Fixture electronics drift is
minimal, so a set stays valid until the *hardware* changes. That is why staleness is
keyed on a :func:`hardware_hash` rather than a clock: a set is good indefinitely and
worthless the instant the routing changes, and no elapsed-time rule expresses that.

**Blanks are measurements, not a parallel type.** A blank is an EIS spectrum with the
same columns, the same file format and the same conditions row as a sample; only its
``role`` differs. So acquisition reuses ``eis_measure_step``, ``EISResult.save`` and
``record_measurement`` unchanged, and this module reads back what they wrote.

.. warning::
   **``Z_phi`` is not here, deliberately.** Earlier drafts had this module derive a
   "phase-reliable ceiling" ~ 5x10^7 ohm. That ceiling is **withdrawn** - it was an
   artefact of a floating reference electrode (overhaul 3.7, F13), and with RE
   correctly connected no negative ``Re Z`` occurs anywhere in the band. What replaces
   it is :class:`PhaseAccuracyTable`: ``eps`` as a function of ``|Z|`` and frequency,
   measured where the samples actually sit, with **no** extrapolation past the
   impedances that were characterised.

.. note::
   Calibration is **incremental, never all-or-nothing**. Each artifact unlocks specific
   capabilities and :class:`CalibrationCapabilities` reports which - and, more usefully,
   names the one artifact that would unblock each thing still missing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
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


# ── The re-export contract ───────────────────────────────────────────────────
#
# ``calibration`` is the published name for all three modules. Consumers import from
# here in single statements spanning every cluster (``commissioning.py`` takes nine
# names at once), and ``tests/test_eis_calibration.py`` monkeypatches
# ``calmod.derive_reference_cap`` on *this* module while the consumer imports it
# function-locally from *this* module — so the re-export is what makes both work.
#
# It is **lazy** rather than a bottom-of-module ``from .calibration_set import ...``.
# The eager form only survives being imported hub-first: import a sibling first and it
# closes a cycle on a half-built module, ``cannot import name ... from partially
# initialized module``. PEP 562 removes the cycle from the import graph entirely — this
# module imports neither sibling until a name is actually asked for, so the arrows run
# one way (sibling → hub) in every order. The first lookup caches into ``globals()``,
# after which the name is an ordinary module attribute: ``monkeypatch.setattr`` and its
# undo behave exactly as they would have with an eager binding.

_REEXPORTS: dict[str, str] = {
    # ── calibration_derive: acquired spectrum → number ────────────────────────
    "CAPACITIVE_REFERENCE": "calibration_derive",
    "PHASE_TABLE_SLOPE_TOL": "calibration_derive",
    "REFERENCE_LOADS": "calibration_derive",
    "RESISTIVE_REFERENCE": "calibration_derive",
    "ReferenceCapResult": "calibration_derive",
    "ReferenceLoad": "calibration_derive",
    "_log_log_slope": "calibration_derive",
    "derive_open": "calibration_derive",
    "derive_open_constants": "calibration_derive",
    "derive_phase_table": "calibration_derive",
    "derive_reference_cap": "calibration_derive",
    "derive_reference_r": "calibration_derive",
    "derive_short": "calibration_derive",
    "phase_table_gate": "calibration_derive",
    # ── calibration_set: the set itself, and its persistence ──────────────────
    "CalibrationSet": "calibration_set",
    "_to_toml": "calibration_set",
    "calibration_path": "calibration_set",
    "describe_or_absent": "calibration_set",
    "load_calibration": "calibration_set",
    "resolve_calibration": "calibration_set",
    "save_calibration": "calibration_set",
}

__all__ = [
    "ARTIFACT_NOMINAL_UNITS",
    "COMMISSIONING_ROLES",
    "CORRECTION_MODES",
    "DRIFT_REPEAT_ROLE",
    "ELECTRODE_MODES",
    "HARDWARE_SECTIONS",
    "MAX_PLAUSIBLE_L_LEAD_H",
    "MAX_PLAUSIBLE_SHORT_OHM",
    "MEASUREMENT_ROLES",
    "TWO_TERMINAL_ROLES",
    "CalibrationCapabilities",
    "FixtureConductance",
    "PhaseAccuracyTable",
    "electrode_mode_ok",
    "hardware_hash",
    "short_is_plausible",
    *sorted(_REEXPORTS),
]


def __getattr__(name: str) -> Any:
    """Resolve a re-exported name from its owning sibling, once, then cache it."""
    module = _REEXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(f"{__package__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_REEXPORTS))
