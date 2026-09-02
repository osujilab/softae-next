"""Derivation from acquired spectra: one blank or reference part, one number.

Each function here takes the ``(f, Z)`` of a single commissioning sweep and returns what
that artifact is evidence *for* — the fixture's series constants from a short, its shunt
constants and usability verdict from an open, a phase-accuracy table from a reference
capacitor, and a magnitude window from a reference resistor. Nothing here reads or
writes a :class:`~softae.analysis.eis.calibration_set.CalibrationSet`; assembling the
numbers into one is ``calibration_set``'s job and choosing which spectra to hand over is
``softae.workflows.commissioning``'s.

Every name is re-exported from :mod:`softae.analysis.eis.calibration`, which is the
published spelling — import from there.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.calibration import FixtureConductance

logger = structlog.get_logger(__name__)


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


@dataclass(frozen=True)
class ReferenceRResult:
    """What one reference-resistor sweep says: a magnitude, and a phase bound.

    Unpacks as the ``(R_measured, error_pct, eps_deg)`` triple
    :func:`derive_reference_r` has always returned — the same affordance
    :class:`ReferenceCapResult` uses — so the two call sites that discard the third
    element (``commissioning``'s ``blank_load`` pass, ``fixture.validate_load``) are
    unaffected. The extra fields exist because the third element's *meaning* changed:
    it is now a bound measured at a particular ``|Z|`` over a particular subset of the
    sweep, and a consumer that files it in a table over ``|Z|`` needs both of those.
    """

    R_ohm: float
    error_pct: float
    #: Median angular deviation from a resistor's ideal 0°, over gated points.
    eps_deg: float
    #: Gated median ``|Z|`` — the impedance :attr:`eps_deg` was actually measured at,
    #: which is not ``|R|`` once the fixture's stray shunts the part (see below).
    z_at_eps_ohm: float = float("nan")
    #: Gate accounting, so an over-drop is visible rather than inferred.
    n_kept: int = 0
    n_total: int = 0

    def __iter__(self) -> Iterator[float]:
        return iter((self.R_ohm, self.error_pct, self.eps_deg))


def derive_reference_r(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    nominal_ohm: float,
    per_decade: bool = True,
    slope_tol: float = PHASE_TABLE_SLOPE_TOL,
) -> ReferenceRResult:
    """A reference resistor's measured value, and the phase bound it supports.

    **The phase statistic is an angular deviation, not a scatter.** It is the exact
    mirror of the loss angle :func:`derive_phase_table` computes for a capacitor —
    the same question asked about a different ideal, with the numerator and the
    denominator swapped::

        capacitive   ideal −90°   eps = degrees(arctan(|Re Z| / |Im Z|))
        resistive    ideal   0°   eps = degrees(arctan(|Im Z| / |Re Z|))

    so both read "median angular deviation from what this part should look like", and
    both are conservative in the same direction — **upward**, because the measured
    deviation contains the part's own reactance as well as the instrument's phase
    error, which is what a floor wants.

    This replaced ``std(phase)``, which was a **scatter** filed in a table of bounds.
    The two are not the same kind of number and no interpolation between them means
    anything. A scatter also cannot see a *bias*: an instrument reading every point of
    a resistor a consistent 5° off ideal has ``std(phase) = 0`` and certifies a perfect
    phase floor. It is not reliably optimistic either — a frequency-varying systematic
    enters ``std`` as dispersion, and on a 9.9 kΩ reference shunted by this fixture's
    own 24.7 pF stray it reports ~4° where the angular deviation reports <0.1°: a ~27×
    disagreement with ``DEFAULT_PHASE_NOISE_DEG = 0.149`` on the very load that constant
    was taken on. Its direction of error is undefined, which is worse than wrong.

    **Gated, by the same gate the capacitive branch uses**, with
    :data:`RESISTIVE_REFERENCE` as the expectation: flat ``|Z|``, ``Re Z > 0``, and
    ``Im Z`` of *either* sign, since a resistor's ideal reactance is zero and whichever
    parasitic dominates sets the sign — series lead inductance low on the ladder, the
    board's stray shunt high on it. The saturation test is not a formality here: above
    ~10⁵ Ω this fixture's 10–25 pF stray turns a reference resistor into a parallel RC
    whose top half rolls off at slope −1, so on the deployed 4 Hz–200 kHz sweep an
    ungated ε at 1 MΩ reads ~8° — the *board* — against ~1.3° once the roll-off is
    dropped. Below ~10 kΩ the gate keeps every point, so it costs no coverage on the
    half of the ladder the historical 0.149° figure lives on.

    The **median** is taken for the reason :func:`derive_phase_table` argues at length:
    the minimum across a sweep is the single luckiest point, not a bound. Per-decade
    binning comes along for the same reason it exists there, though on this fixture it
    is a no-op — what survives gating spans a small fraction of a decade, so one part
    contributes one ``(|Z|, ε)`` point rather than a table. Where a part *did* span two
    decades the bins are collapsed by a median of the per-decade medians, which is an
    identity in the single-bin case this hardware actually produces.

    *R_ohm* and *error_pct* are unchanged and deliberately ungated: the magnitude
    accuracy is a separate concern with its own consumer (``fixture.validate_load``).
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    n_total = int(Zc.size)
    good = np.isfinite(Zc.real) & np.isfinite(Zc.imag)
    if not np.any(good):
        return ReferenceRResult(float("nan"), float("nan"), float("nan"),
                                float("nan"), 0, n_total)

    R = float(np.median(Zc.real[good]))
    err = ((R - nominal_ohm) / nominal_ohm * 100.0) if nominal_ohm else float("nan")

    mag = np.abs(Zc)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.degrees(np.arctan(np.abs(Zc.imag) / np.abs(Zc.real)))

    ok = phase_table_gate(freq, Zc, load=RESISTIVE_REFERENCE,
                          slope_tol=slope_tol) & np.isfinite(eps)
    n_kept = int(np.count_nonzero(ok))
    if not n_kept:
        return ReferenceRResult(R, err, float("nan"), float("nan"), 0, n_total)
    if n_kept < n_total:
        logger.info("eis_reference_r_gated", kept=n_kept, total=n_total)

    mag, eps = mag[ok], eps[ok]
    if per_decade:
        decade = np.floor(np.log10(mag)).astype(int)
        bins = sorted(set(decade.tolist()))
        z_bins = [float(np.median(mag[decade == d])) for d in bins]
        e_bins = [float(np.median(eps[decade == d])) for d in bins]
    else:
        z_bins, e_bins = [float(np.median(mag))], [float(np.median(eps))]

    return ReferenceRResult(R, err, float(np.median(e_bins)),
                            float(np.median(z_bins)), n_kept, n_total)
