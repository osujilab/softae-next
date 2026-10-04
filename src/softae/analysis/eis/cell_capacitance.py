"""C_cell — the empty-electrode capacitance a regime-B film arc is identified against.

Slice 2, lane L2 (``docs/SubAgent docs/regime_b_slice2_spec.md`` §2.2, operator rulings Q1–Q2
of 2026-10-02). Pure functions, no I/O: the operator-run ``softae.tools.c_cell`` reads the
store and writes the accepted value; nothing here does either.

**The estimator.** On fixture-corrected, screened points, C_cell is the median of the
apparent capacitance ``−1/(2πf·Im Z)`` over 100 Hz – 1 kHz. It is immune to the series
element, which only adds to Re Z.

**Insulating films only** (Q2). A spectrum is a source only when every rule holds:

==========================  ==============================================================
Reason it is refused        Rule
==========================  ==============================================================
``well_empty``              the occupancy record says nothing was cast when it was read
``occupancy_unrecorded``    no occupancy record at all — unknown is not "occupied"
``regime_bad_data``         U for a bad-data reason, or unclassified (the empty-well and
                            unclosed-RE signature: incoherent, mostly non-physical)
``regime_a_film``           A: a conducting film's arc, by classification
``band_unresolved``         fewer than 3 screened points in the band
``phase_not_capacitive``    median phase in the band above −80°
``apparent_c_not_flat``     apparent C max/min above 1.3 in the band
``lf_dropped``              lowest screened point above 10 Hz: the LF was lost, so the
                            reference electrode is suspect (spec: 3b ch7 at 165 Hz)
``film_arc_present``        tan δ at the foot ≥ 1: an arc closed below the valley, i.e. a
                            film, whose arc capacitance is not C_cell
==========================  ==============================================================

**An empty well is never a source** (09-30 Q2): its reference electrode floats. It is caught
twice — by its occupancy record, and by its data (U/incoherent, or the LF dropped).

**Absent is unavailable, never a default.** A group with no qualifying spectrum has no
entry; ``[eis.instrument] c_cell_F`` is never substituted (it is 3–6× the measured value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from softae.analysis.eis.observation import is_classified
from softae.analysis.eis.regime import A, classify_regime

#: Stored as ``cell_capacitance.method``; bump the suffix if any rule below changes.
METHOD = "apparent_c_100_1000Hz_v1"
BAND_HZ = (100.0, 1000.0)
PHASE_MAX_DEG = -80.0
FLAT_MAX = 1.3
F_LO_MAX_HZ = 10.0
#: The partner's existence test (spec §1.3): R_mf exists iff tan δ(foot) ≥ 1.
FOOT_TAND_MAX = 1.0
#: A median that one outlier cannot move needs three points.
MIN_BAND_POINTS = 3
CHANNEL_GROUPS = ((1, 8), (9, 16), (17, 24), (25, 32))

OCCUPIED, EMPTY, UNKNOWN = "occupied", "empty", "unknown"

__all__ = ["METHOD", "CHANNEL_GROUPS", "OCCUPIED", "EMPTY", "UNKNOWN", "InsulatorCheck",
           "CellCapacitance", "channel_group", "apparent_capacitance", "foot_tand",
           "check_insulator", "assess_spectrum", "aggregate_groups", "preferred_record"]


def channel_group(channel: int) -> tuple[int, int] | None:
    """The mux group ``(lo, hi)`` holding *channel*, or ``None`` off the board."""
    return next((g for g in CHANNEL_GROUPS if g[0] <= int(channel) <= g[1]), None)


def apparent_capacitance(f: Any, Z: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(f, −1/(2πf·Im Z))`` for the points inside :data:`BAND_HZ`, ascending in f."""
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1]) & (Z.imag < 0)
    order = np.argsort(f[band])
    fb = f[band][order]
    return fb, -1.0 / (2 * np.pi * fb * Z.imag[band][order])


def foot_tand(f: Any, Z: Any) -> float:
    """tan δ at the foot: its maximum *below* the valley (the tan δ minimum).

    0.0 when the valley is the lowest point — capacitive all the way down, so no arc
    closed in band. Raw, not smoothed: one high point refuses a read, never admits one.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    order = np.argsort(f)
    tand = (Z.real / -Z.imag)[order]
    valley = int(np.argmin(tand))
    return float(tand[:valley].max()) if valley > 0 else 0.0


@dataclass(frozen=True)
class InsulatorCheck:
    """One spectrum's C estimate and every rule it failed (empty = a source)."""

    channel: int
    c_cell_F: float | None
    reasons: tuple[str, ...]
    measurement_id: int | None = None
    n_band: int = 0
    phase_med_deg: float = math.nan
    flat_ratio: float = math.nan
    f_lo_hz: float = math.nan
    foot_tand: float = math.nan
    regime: str = ""
    regime_reason: str = ""
    occupancy: str = UNKNOWN

    @property
    def qualifies(self) -> bool:
        return not self.reasons and self.c_cell_F is not None


def _occupancy_reasons(occupancy: str) -> list[str]:
    if occupancy == OCCUPIED:
        return []
    return ["well_empty" if occupancy == EMPTY else "occupancy_unrecorded"]


def _regime_reasons(regime: str, reason: str) -> list[str]:
    if not is_classified(regime, reason):
        return ["regime_bad_data"]
    return ["regime_a_film"] if regime == A else []


def check_insulator(f: Any, Z: Any, *, channel: int, regime: str, regime_reason: str,
                    occupancy: str, measurement_id: int | None = None) -> InsulatorCheck:
    """Apply every rule to *screened*, fixture-corrected points; never raises on data."""
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    reasons = _occupancy_reasons(occupancy) + _regime_reasons(regime, regime_reason)
    fb, cb = apparent_capacitance(f, Z)
    n = int(cb.size)
    c = float(np.median(cb)) if n else None
    flat = float(cb.max() / cb.min()) if n else math.nan
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    phase = float(np.median(np.degrees(np.angle(Z[band])))) if band.any() else math.nan
    f_lo = float(f.min()) if f.size else math.nan
    foot = foot_tand(f, Z) if f.size else math.nan
    if n < MIN_BAND_POINTS:
        reasons.append("band_unresolved")
    if not phase <= PHASE_MAX_DEG:
        reasons.append("phase_not_capacitive")
    if not flat <= FLAT_MAX:
        reasons.append("apparent_c_not_flat")
    if not f_lo <= F_LO_MAX_HZ:
        reasons.append("lf_dropped")
    if not foot < FOOT_TAND_MAX:
        reasons.append("film_arc_present")
    return InsulatorCheck(channel=int(channel), c_cell_F=c, reasons=tuple(reasons),
                          measurement_id=measurement_id, n_band=n, phase_med_deg=phase,
                          flat_ratio=flat, f_lo_hz=f_lo, foot_tand=foot, regime=regime,
                          regime_reason=regime_reason, occupancy=occupancy)


def assess_spectrum(f: Any, Z: Any, *, channel: int, occupancy: str,
                    measurement_id: int | None = None) -> InsulatorCheck:
    """Classify *all* fixture-corrected points, then check the classifier's screened ones.

    The same order the engine uses: the regime verdict reads every point, and the
    estimator reads only the points the screen kept.
    """
    v = classify_regime(f, Z)
    return check_insulator(v.screen.f, v.screen.Z, channel=channel, regime=v.label,
                           regime_reason=v.reason, occupancy=occupancy,
                           measurement_id=measurement_id)


@dataclass(frozen=True)
class CellCapacitance:
    """One group's C_cell: the median of its channels' values, with its support."""

    board_id: int
    group: tuple[int, int]
    c_cell_F: float
    spread_dec: float                  # log10(max/min) over the channel values
    source_measurement_ids: tuple[int, ...]
    channel_values: tuple[tuple[int, float], ...]
    method: str = METHOD

    @property
    def n_spectra(self) -> int:
        return len(self.source_measurement_ids)


def aggregate_groups(checks: Iterable[InsulatorCheck], *,
                     board_id: int) -> dict[tuple[int, int], CellCapacitance]:
    """Per group: median over channels of each channel's median over its sources.

    A channel read thirteen times must not outvote a channel read once. Only qualifying
    checks count; a group with none is absent — never a default.
    """
    per_channel: dict[int, list[InsulatorCheck]] = {}
    for c in checks:
        if c.qualifies and channel_group(c.channel) is not None:
            per_channel.setdefault(c.channel, []).append(c)
    out: dict[tuple[int, int], CellCapacitance] = {}
    for group in CHANNEL_GROUPS:
        chans = sorted(ch for ch in per_channel if channel_group(ch) == group)
        if not chans:
            continue
        values = [(ch, float(np.median([c.c_cell_F for c in per_channel[ch]])))
                  for ch in chans]
        v = np.array([x for _, x in values])
        ids = tuple(sorted(c.measurement_id for ch in chans for c in per_channel[ch]
                           if c.measurement_id is not None))
        out[group] = CellCapacitance(
            board_id=int(board_id), group=group, c_cell_F=float(np.median(v)),
            spread_dec=float(np.log10(v.max() / v.min())), source_measurement_ids=ids,
            channel_values=tuple(values))
    return out


def preferred_record(channel: int, records: Iterable[Any]) -> Any | None:
    """Spec §2.2's preference order over one board's current (non-superseded) records.

    The channel's own record first, then the group record covering it, else ``None`` —
    the caller then reports ``c_cell_unmeasured``. A record exposes ``channel`` (``None``
    for a group row), ``group_lo`` and ``group_hi``, as attributes or mapping keys.
    """
    def get(r: Any, key: str) -> Any:
        return r.get(key) if isinstance(r, dict) else getattr(r, key, None)

    records = list(records)
    own = [r for r in records if get(r, "channel") == int(channel)]
    if own:
        return own[-1]
    group = [r for r in records if get(r, "channel") is None
             and (get(r, "group_lo") or 0) <= int(channel) <= (get(r, "group_hi") or -1)]
    return group[-1] if group else None
