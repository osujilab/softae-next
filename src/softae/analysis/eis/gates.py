"""Does this spectrum contain the physics we are about to extract from it?

The expensive failure in EIS is not a fit that fails — it is a fit that *succeeds*
on a spectrum containing no parallel conduction, reports an ``R1``, and hands a
conductivity to a campaign that then optimises against it. Overhaul §3.3 is that
failure: a dry film reduced to a pure series RC, tan δ rising with frequency, the
sample's conduction absent from the data at every frequency — and the correct output
was an upper bound, not a fitted value.

So the gates run **before** any optimiser. They are cheap, they need no fit, and they
are decisive.

Two severities and a flag, per framework §2.1:

``block_point``
    Marks surviving points; the spectrum continues with fewer of them.
``block_spectrum``
    The spectrum is rejected outright and the detail carries the reason.
``flag``
    Advisory. Recorded, never removes data.

**Nothing is ever dropped silently.** Every removed point and every rejected spectrum
carries a gate name and a human-readable reason. Overhaul §3.2 records silent masking
as the root cause of the hardest-to-diagnose failure in that campaign — a correction
that quietly corrupted data and produced plausible-looking but wrong results
downstream. The log is the mitigation and it is not optional.

.. note::
   Two deliberate deviations from the printed specification, both explained at their
   gate: the topology triad runs as a **group** rather than stopping at the first
   failure (§3.5.3 wants two independent formulations to confirm each other, which
   the printed runner's ``break`` makes impossible), and
   :func:`gate_stuck_instrument` has no counterpart in the framework at all — the
   spec assumes a working instrument, and this rig has produced a stuck one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.admittance import (
    apparent_capacitance,
    log_slope,
    loss_tangent,
    parallel_branch_window,
    top_decade_window,
)

logger = structlog.get_logger(__name__)

#: Severity levels, exactly as the framework names them.
BLOCK_POINT = "block_point"
BLOCK_SPECTRUM = "block_spectrum"
FLAG = "flag"

#: Fourth severity, used by series-level gates (framework §5.1/§5.2 assign it but
#: §2.1's enum omits it). Ships **unarmed** — nothing raises it in E0.
BLOCK_SESSION = "block_session"

SEVERITIES = (BLOCK_POINT, BLOCK_SPECTRUM, FLAG, BLOCK_SESSION)


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict on one spectrum.

    A frozen dataclass rather than the specification's ``namedtuple``, to match house
    style — but with the spec's exact field names and severity strings, so the
    printed code stays checkable against this one line for line.
    """

    name: str
    severity: str
    passed: bool
    detail: str
    mask: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_dropped(self) -> int:
        """How many points this gate removed (always 0 unless ``block_point``)."""
        if self.severity != BLOCK_POINT:
            return 0
        return int((~np.asarray(self.mask, dtype=bool)).sum())

    def as_log_entry(self) -> dict[str, Any]:
        """The runner's log shape — R17's 'named gate and reason', untranslated."""
        return {
            "gate": self.name,
            "severity": self.severity,
            "passed": bool(self.passed),
            "detail": self.detail,
            "n_dropped": self.n_dropped,
        }

    def describe(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        drop = f" (−{self.n_dropped} pts)" if self.n_dropped else ""
        return f"[{mark}] {self.name}{drop}: {self.detail}"


def _all_pass(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=bool)


def _ctx_get(ctx: dict[str, Any], section: str, key: str, default: Any) -> Any:
    """Read ``ctx[section].key`` whether the section is an object or a mapping."""
    obj = ctx.get(section)
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── §3.3 Housekeeping ────────────────────────────────────────────────────────

def gate_finiteness(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Drop non-finite values, duplicate frequencies, and ``f ≤ 0``.

    Non-finite values and repeated frequencies break both the Kramers–Kronig basis
    construction and the fit Jacobian.

    ``f ≤ 0`` is dropped here for a reason the framework does not state: the topology
    triad at step 8 takes ``np.polyfit(np.log10(f), …)``, so a single non-positive
    frequency reaching that far divides by zero and poisons all three slopes at once.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    n = int(min(f.size, Z.size))
    ok = np.isfinite(f[:n]) & np.isfinite(Z[:n]) & (f[:n] > 0)

    seen: set[float] = set()
    for i in range(n):
        if not ok[i]:
            continue
        key = float(f[i])
        if key in seen:
            ok[i] = False
        else:
            seen.add(key)

    n_bad = int((~ok).sum())
    return GateResult(
        "finiteness", BLOCK_POINT, n_bad == 0,
        f"{n_bad} non-finite, non-positive, or duplicate-frequency pts"
        if n_bad else "all points finite, positive-frequency, unique",
        ok,
    )


def gate_monotonic_frequency(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Flag a frequency axis that reverses direction.

    Interleaving points from two sweeps would silently mix measurements. Advisory
    rather than blocking, because a re-ordered file is still usable data.
    """
    f = np.asarray(f, dtype=float)
    ok = _all_pass(f.size)
    if f.size <= 2:
        return GateResult("monotonic_frequency", FLAG, True,
                          "too few points to judge", ok)
    d = np.diff(f)
    good = bool(np.all(d > 0) or np.all(d < 0))
    return GateResult(
        "monotonic_frequency", FLAG, good,
        "frequency axis is monotonic" if good
        else "frequency axis reverses — points may come from two sweeps",
        ok,
    )


# ── §3.1 Unphysical quadrant ─────────────────────────────────────────────────

#: Fraction of points violating the quadrant beyond which a floating reference
#: electrode is the leading suspicion rather than the instrument (R19).
RE_SUSPICION_FRAC = 0.2


def gate_quadrant(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Reject points outside the passive capacitive quadrant — and say *why*.

    For a passive two-terminal cell ``Re Z > 0`` always, and a blocking cell's phase is
    confined to ``[−90°, 0°]``. Points outside that region are not measurements.

    **Three causes on this board, and they call for different responses (R19/F13).**

    The reference electrode here is a **stripe between CE and WE**, connected to the
    cell only through whatever spans the coplanar gap. So the control loop is closed by
    the *sample*, and what a quadrant violation means depends on what is on the board:

    1. **Open by geometry** (``open_by_geometry``) — a bare board has only air between
       the stripes, so the RE floats *by construction*. The violation is structural and
       expected; there is no wiring to repair and nothing to attribute to the
       instrument. Such a spectrum is a legitimate measurement of the inter-stripe
       geometry, but it is **not** a fixture open and cannot serve as an OSL term.
    2. **Loop closed** (``bridged_by_sample`` / ``tied_to_ce`` / ``connected``) — a cast
       film bridges the stripes, or RE is jumpered at the connector. The loop is intact,
       so a violation genuinely is instrument-side and may justify a bound.
    3. **Unverified** — the honest default. A widespread violation is then a *suspicion*
       about the loop, not a finding about the instrument.

    The distinction is not academic. Attributing a floating loop to the instrument is
    precisely how the withdrawn ``Z_φ ≈ 5×10⁷ Ω`` ceiling came to be believed: the
    ``Re Z`` error from a floating RE follows a clean ``|Z''|²`` law that mimics a
    genuine constant conductance offset.

    Points are dropped in every case — they are artefact either way — but only the
    detail string differs, because the correct response ranges from "expected, ignore"
    to "repeat with RE tied to CE" to "this really is the instrument".
    """
    Z = np.asarray(Z, dtype=complex)
    with np.errstate(invalid="ignore"):
        phase = np.degrees(np.angle(Z))
        # The phase clause is no second criterion — for finite Z it is implied by
        # `Re > 0`. It guards direct callers that bypass `gate_finiteness`'s mask.
        ok = (Z.real > 0) & (np.abs(phase) <= 90.0 + 1e-9)
    ok = np.asarray(ok, dtype=bool)
    n_bad = int((~ok).sum())

    if n_bad == 0:
        return GateResult("quadrant", BLOCK_POINT, True,
                          "all points in the passive capacitive quadrant", ok)

    from softae.analysis.eis.policy import RE_CLOSED_LOOP

    re_state = str(_ctx_get(ctx, "meta", "re_connection", "unverified"))
    widespread = n_bad > RE_SUSPICION_FRAC * max(ok.size, 1)
    detail = f"{n_bad} pts with Re Z < 0 or |phase| > 90°"

    if re_state == "open_by_geometry":
        detail += (
            " — expected: nothing bridges the RE stripe, so the loop is open by "
            "construction. Not an instrument limit and not a repairable fault"
        )
    elif widespread and re_state == "unverified":
        detail += (
            " — RE integrity UNVERIFIED; suspect an open control loop "
            "(is anything bridging the stripes? tie RE to CE and repeat "
            "before attributing this to the instrument)"
        )
        logger.warning(
            "eis_quadrant_re_unverified", n_bad=n_bad, n_total=int(ok.size),
            re_connection=re_state,
            msg="widespread quadrant violation with unverified reference electrode",
        )
    elif widespread and re_state in RE_CLOSED_LOOP:
        detail += (
            f" — RE reported '{re_state}', so the loop was closed and this is "
            f"genuinely instrument-side"
        )

    return GateResult(
        "quadrant", BLOCK_POINT, False, detail, ok,
        {"frac_quadrant_violation": float(n_bad) / max(ok.size, 1)},
    )


# ── §3.2 Magnitude window ────────────────────────────────────────────────────

def gate_magnitude(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Drop points outside the instrument's reproducible ``|Z|`` range.

    Outside ``[Z_MIN, Z_MAX]`` the accuracy specification does not apply, so residuals
    from those points are meaningless — yet a least-squares fit weights them equally.

    Pointwise, unlike the legacy median-based check in
    :func:`softae.analysis.quality.validate_eis_trace`. That is stricter and correct:
    a spectrum can be perfectly usable over most of its band while running over range
    at one end, and rejecting or keeping the whole trace on a median is the wrong
    granularity.
    """
    Z = np.asarray(Z, dtype=complex)
    mag = np.abs(Z)
    z_min = float(_ctx_get(ctx, "envelope", "z_min_ohm", 0.0))
    z_max = float(_ctx_get(ctx, "envelope", "z_max_ohm", np.inf))
    ok = np.isfinite(mag) & (mag >= z_min) & (mag <= z_max)
    n_bad = int((~ok).sum())
    return GateResult(
        "magnitude_window", BLOCK_POINT, n_bad == 0,
        f"{n_bad} pts outside [{z_min:.3g}, {z_max:.3g}] Ω"
        if n_bad else f"all points within [{z_min:.3g}, {z_max:.3g}] Ω",
        ok,
        {"z_min_seen": float(np.min(mag)) if mag.size else float("nan"),
         "z_max_seen": float(np.max(mag)) if mag.size else float("nan")},
    )


def gate_phase_noise_extrapolated(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Flag a spectrum sitting far from where the phase noise was characterised.

    Replaces an earlier ``above_phase_ceiling`` gate built on ``Z_φ ≈ 5×10⁷ Ω``, which
    is **withdrawn** — that ceiling was an artefact of a floating reference electrode
    and no evidence supports a phase-reliable limit below the magnitude ceiling.

    What remains is narrower and true: ``ε`` was measured on a *resistive* load at
    10⁴ Ω, films sit at 10⁶–10⁸ Ω and are capacitive, and a loss tangent compared
    against a floor extrapolated three decades is not a qualified comparison. Advisory,
    because the consequence is a *reporting mode* — the bound becomes provisional
    rather than the spectrum being rejected.
    """
    Z = np.asarray(Z, dtype=complex)
    mag = np.abs(Z)
    ok = _all_pass(mag.size)
    finite = np.isfinite(mag)
    if not finite.any():
        return GateResult("phase_noise_extrapolated", FLAG, True,
                          "no finite points to judge", ok)

    env = ctx.get("envelope")
    z_med = float(np.median(mag[finite]))
    valid = bool(getattr(env, "phase_noise_valid_at", lambda _z: True)(z_med))
    at = float(getattr(env, "phase_noise_at_ohm", float("nan")))

    return GateResult(
        "phase_noise_extrapolated", FLAG, valid,
        f"median |Z| {z_med:.3g} Ω within the band where phase noise was measured"
        if valid else
        f"median |Z| {z_med:.3g} Ω is far from the {at:.3g} Ω at which phase noise "
        f"was characterised — any loss-tangent floor here is extrapolated",
        ok,
        {"z_median": z_med, "phase_noise_valid": float(valid)},
    )


# ── Instrument health (no counterpart in the framework) ──────────────────────

def gate_stuck_instrument(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Reject a trace whose ``|Z|`` is identical at every frequency.

    **Not in the framework**, which assumes a working instrument throughout. This rig
    has produced one that was not: the check is preserved verbatim from
    :func:`softae.analysis.quality.validate_eis_trace`, where it was added because a
    stuck instrument returns plausible floats that average to a plausible objective.
    Real spectra always vary across a decade sweep.
    """
    Z = np.asarray(Z, dtype=complex)
    mag = np.abs(Z)
    ok = _all_pass(mag.size)
    finite = mag[np.isfinite(mag)]
    if finite.size <= 2:
        return GateResult("stuck_instrument", BLOCK_SPECTRUM, True,
                          "too few points to judge", ok)
    stuck = bool(np.allclose(finite, finite[0], rtol=1e-9, atol=0.0))
    return GateResult(
        "stuck_instrument", BLOCK_SPECTRUM, not stuck,
        "|Z| identical at every frequency — instrument may be stuck" if stuck
        else "|Z| varies across the sweep",
        ok,
    )


# ── §3.4 Blocking-cell high-frequency inductive rejection ────────────────────

def gate_hf_inductive(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Truncate a contiguous inductive run at the top of the band, on a blocking cell.

    If the cell is ionically blocking and no faradaic process exists, the sample
    response is capacitive at every frequency. After fixture series correction any
    residual ``Im Z > 0`` at the top of the band is therefore either genuine
    uncorrected lead inductance or instrument phase error near ``F_MAX``.

    Left in, these points force the fit to allocate a large and physically absurd
    ``L`` — overhaul F5 records fitted values of 400–500 µH against a short blank's
    measured 4.18 µH — which then distorts the high-frequency end where ``R_series``
    is determined, propagating a phase artefact into the resistance split.

    Safe to remove: the plateau, and hence ``R_bulk``, lies at or below the ``−Z''``
    minimum, below the artefact in frequency.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    n = int(min(f.size, Z.size))
    ok = _all_pass(n)

    if not bool(_ctx_get(ctx, "cell", "blocking", True)):
        return GateResult("hf_inductive", BLOCK_POINT, True,
                          "skipped — non-blocking cell, inductance may be real", ok)

    # Walk down from the highest frequency; stop at the first capacitive point, so
    # only the contiguous run at the top is removed.
    for i in np.argsort(f[:n])[::-1]:
        if np.isfinite(Z[i]) and Z.imag[i] > 0:
            ok[i] = False
        else:
            break

    n_bad = int((~ok).sum())
    return GateResult(
        "hf_inductive", BLOCK_POINT, n_bad == 0,
        f"{n_bad} HF inductive pts truncated — blocking cell, so Im Z > 0 is artefact"
        if n_bad else "no inductive run at the top of the band",
        ok,
    )


# ── §3.5 Topology admission — the decisive triad ─────────────────────────────

def gate_tand_slope(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Does a conductance sit in *parallel* with the capacitance at all?

    ``tan δ = G/(ωC) ∝ f⁻¹`` for parallel conduction; ``tan δ = ωC·R_s ∝ f⁺¹`` for a
    series parasitic. The **sign of the slope** is the discriminator.

    A positive slope means the measured dissipation is series — contact, lead, or
    electrode resistance in line with the cell capacitance. No amount of fitting
    recovers a parallel conductance from such a spectrum, because none is present.
    The physical remedy is to raise ``G`` (higher conductivity, higher temperature) or
    lower ``C_par`` until the parallel term re-enters the measurable range.

    The slope is fitted over :func:`~softae.analysis.eis.admittance.parallel_branch_window`
    rather than the whole sweep — see that function for why the printed global fit
    rejects well-formed blocking-cell spectra.
    """
    f = np.asarray(f, dtype=float)
    tand = loss_tangent(Z)
    window = parallel_branch_window(f, Z)
    slope = log_slope(f[window], tand[window])
    ok = _all_pass(f.size)
    threshold = float(_ctx_get(ctx, "gates", "tand_slope_max", -0.3))

    if slope != slope:
        return GateResult("tand_slope", BLOCK_SPECTRUM, False,
                          "insufficient valid tanδ points to take a slope", ok,
                          {"tand_slope": float("nan")})

    passed = slope <= threshold
    kind = ("parallel conduction present" if passed
            else "SERIES parasitic — no conductivity content at any frequency")
    return GateResult(
        "tand_slope", BLOCK_SPECTRUM, passed,
        f"d log tanδ/d log f = {slope:+.2f} over {int(window.sum())} pts ({kind})", ok,
        {"tand_slope": slope, "tand_window_pts": float(window.sum())},
    )


def gate_cap_flatness(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Is the parallel capacitance ideal, or a dispersive lossy dielectric?

    A geometric or fixture capacitance is frequency-flat. A falling ``C_app(f)``
    indicates a dispersive lossy dielectric — which carries its own parallel
    conductance ``G ≈ ωC·tanδ_mat`` occupying the *same quadrature component* as the
    sample's ionic conduction. When that term is comparable to the measurand, the
    extracted ``G`` is not the sample's.

    Overhaul §3.2 found exactly this on the fixture blanks (FR-4 plus moisture), which
    is why blank subtraction there was untenable: detecting dispersion early tells you
    that subtracting a scalar stray capacitance will not be enough.

    Advisory. The framework conditions escalation to ``block_spectrum`` on the sample
    conductance being small *compared with the inferred dielectric loss*, and that
    comparison needs the substrate loss quantified — overhaul open question #3.

    Fitted over :func:`~softae.analysis.eis.admittance.top_decade_window`, because
    ``C_app`` only *is* the geometric capacitance above the relaxation corner. Over the
    full band a well-formed spectrum reads −0.66 and would be called dispersive; over
    the top decade the same spectrum reads −0.07 and a genuinely dispersive one −1.30.
    """
    f = np.asarray(f, dtype=float)
    C = apparent_capacitance(f, Z)
    window = top_decade_window(f)
    slope = log_slope(f[window], C[window])
    ok = _all_pass(f.size)
    threshold = float(_ctx_get(ctx, "gates", "cap_flatness_max", 0.15))

    if slope != slope:
        return GateResult("cap_flatness", FLAG, True,
                          "insufficient valid C_app points to take a slope", ok,
                          {"cap_slope": float("nan")})

    passed = abs(slope) <= threshold
    return GateResult(
        "cap_flatness", FLAG, passed,
        f"d log C_app/d log f = {slope:+.2f} "
        f"({'ideal capacitance' if passed else 'DISPERSIVE — lossy dielectric present'})",
        ok, {"cap_slope": slope},
    )


def gate_series_rc(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """The same failure as :func:`gate_tand_slope`, seen from magnitude not phase.

    A pure series RC shows ``Z'`` flat with frequency while ``−Z''`` falls as ``f⁻¹``.
    A parallel RC shows the opposite: ``Z'`` falls at high frequency while the
    low-frequency plateau is resistive.

    Redundant with the loss-tangent slope **by design** — framework §3.5.3 calls
    agreement between two independent formulations the confirmation. See
    :func:`run_gates` for why that redundancy needs the triad run as a group.
    """
    Z = np.asarray(Z, dtype=complex)
    ok = _all_pass(np.asarray(f).size)
    sr = log_slope(f, np.abs(np.real(Z)))
    si = log_slope(f, np.abs(np.imag(Z)))

    if sr != sr or si != si:
        return GateResult("series_rc_topology", BLOCK_SPECTRUM, True,
                          "insufficient points to take Z′/Z″ slopes", ok,
                          {"zreal_slope": sr, "zimag_slope": si})

    series_like = (abs(sr) < 0.15) and (si < -0.85)
    return GateResult(
        "series_rc_topology", BLOCK_SPECTRUM, not series_like,
        f"slope Z′={sr:+.2f}, Z″={si:+.2f}"
        + (" — pure series RC, no bulk arc" if series_like else ""),
        ok, {"zreal_slope": sr, "zimag_slope": si},
    )


#: The topology-admission triad. Run as a group — see :func:`run_gates`.
TOPOLOGY_TRIAD: tuple[Callable[..., GateResult], ...] = (
    gate_tand_slope,
    gate_cap_flatness,
    gate_series_rc,
)


# ── §3.7b R_sol feature identification ───────────────────────────────────────

def gate_valley_feature(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Require a resolvable valley, and record how far it sits from the |Z| minimum.

    A blocking-cell spectrum has **two** minima that are easy to confuse:

    ===============================  ==============================  ============
    feature                          what it is                      band
    ===============================  ==============================  ============
    minimum of ``|Z|``               HF intercept ≈ ``R_series``     upper
    interior local min of ``−Z''``   the valley ≈ ``R_s + R_bulk``   mid
    ===============================  ==============================  ============

    Overhaul §3.9 records taking the wrong one **twice**, and it reached a published
    comparison before being caught. The two differed by more than an order of
    magnitude on the same file (7.4×10⁴ vs 1.06×10⁶). What makes F15 so dangerous is
    that it has no other symptom: the spectrum looks fine, the fit looks fine, the
    residuals look fine, and σ is simply wrong by 10×.

    **Strict interior is the whole point.** A naive ``argmin`` over a search window
    returns the window *edge* whenever the true minimum lies outside it, which is
    exactly how the error happened. Endpoints are therefore never candidates here.

    .. note::
       ``circuit_fitting._local_minima`` — the legacy initial-guess helper — *does*
       admit endpoints, since its window comparison at ``i = 0`` sees only the points
       above it. That is left alone under the legacy-untouched rule; this gate is what
       catches the consequence, and it runs before any fit is attempted.

    Severity is ``block_spectrum`` when no interior minimum exists (there is nothing to
    extract, and falling back to the ``|Z|`` minimum is the error itself) and ``flag``
    otherwise, recording both features so the confusion is visible in the log.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    keep = _all_pass(freq.size)

    order = np.argsort(freq)
    fs = freq[order]
    zi = -Zc.imag[order]          # −Z'' > 0 in the capacitive quadrant
    zr = Zc.real[order]

    # Strict interior: an endpoint is a window artifact, never a valley.
    cand = [
        k for k in range(1, len(fs) - 1)
        if np.isfinite(zi[k]) and zi[k] < zi[k - 1] and zi[k] < zi[k + 1]
    ]
    if not cand:
        return GateResult(
            "valley_feature", BLOCK_SPECTRUM, False,
            "no interior −Z'' local minimum: no resolvable valley — do NOT fall "
            "back to the |Z| minimum, that is R_series and differs by ~10×",
            keep,
        )

    k = cand[0]
    r_valley = float(zr[k])

    z_abs = np.abs(Zc)
    z_min = float(np.min(z_abs[np.isfinite(z_abs)])) if np.any(np.isfinite(z_abs)) else float("nan")
    ratio = r_valley / z_min if z_min > 0 else float("nan")
    return GateResult(
        "valley_feature", FLAG, True,
        f"valley Z'={r_valley:.3e} Ω at {fs[k]:.4g} Hz; |Z|min={z_min:.3e} Ω "
        f"(ratio {ratio:.1f}× — these are different features)",
        keep,
        {"R_sol_valley": r_valley, "f_valley": float(fs[k]),
         "valley_over_zmin": ratio},
    )


# ── §3.8 Minimum surviving points ────────────────────────────────────────────

def gate_min_points(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Reject a spectrum gated down below what a fit can be supported by.

    Fitting a 5-parameter model to 5 surviving points produces a number with no
    support. A spectrum reduced this far is unmeasurable, not merely noisy.
    """
    n = int(min(np.asarray(f).size, np.asarray(Z).size))
    ok = _all_pass(n)
    need = int(_ctx_get(ctx, "gates", "min_fit_pts", 8))
    passed = n >= need
    return GateResult(
        "min_points", BLOCK_SPECTRUM, passed,
        f"{n} surviving points (need {need})", ok, {"n_surviving": float(n)},
    )


# ── Front 2: how well determined is the answer? ──────────────────────────────
#
# These run *after* a fit and read it from ``ctx["fit"]``. They do not remove points —
# by the time a fit exists the data has already been admitted — so every one is a
# ``flag``, with the single exception of a residual norm so large that the model
# plainly does not describe the data at all.

def gate_pegged_parameters(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Flag parameters resting on a box constraint (framework §4.1).

    A pegged parameter is unidentified: the data pushed it as far as the optimiser
    allowed, so its value is a property of the *bound* rather than of the sample, and
    the standard error reported beside it is meaningless.
    """
    ok = _all_pass(np.asarray(f).size)
    cov = getattr(ctx.get("fit"), "covariance", None)
    if cov is None:
        return GateResult("pegged_parameters", FLAG, True,
                          "no covariance available", ok)

    tol = float(_ctx_get(ctx, "gates", "bound_tol", 1e-3))
    pegged = cov.pegged(tol)
    return GateResult(
        "pegged_parameters", FLAG, not pegged,
        f"pegged at a bound: {', '.join(pegged)} — value set by the constraint, "
        f"not the data" if pegged else "no parameter rests on a bound",
        ok, {"n_pegged": float(len(pegged))},
    )


def gate_relative_standard_error(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Flag a poorly determined *measurand* (framework §4.2).

    Nuisance parameters may legitimately be loose; the resistance the conductivity is
    computed from may not. Only the reported resistance is checked — flagging a badly
    determined CPE exponent on every spectrum would be noise.
    """
    ok = _all_pass(np.asarray(f).size)
    fit = ctx.get("fit")
    cov = getattr(fit, "covariance", None)
    if cov is None:
        return GateResult("relative_standard_error", FLAG, True,
                          "no covariance available", ok)

    from softae.analysis.eis.models import roles_for

    roles = roles_for(getattr(fit, "model_name", "")) or {}
    a, b = roles.get("R_series", "R0"), roles.get("R_bulk", "R1")
    rho = cov.rho(a, b)
    degenerate = cov.singular or (rho == rho and rho <= float(
        _ctx_get(ctx, "gates", "rho_degenerate", -0.95)))

    if degenerate:
        total = cov.sum_value(a, b)
        rel = abs(cov.sum_se(a, b) / total) if total else float("nan")
        label = "R_series+R_bulk"
    else:
        rel = cov.rel_se(b)
        label = b

    limit = float(_ctx_get(ctx, "gates", "max_rel_se", 0.10))
    passed = not (rel == rel and rel > limit)
    return GateResult(
        "relative_standard_error", FLAG, passed,
        f"{label} determined to {rel * 100:.1f}%"
        + ("" if passed else f", above the {limit * 100:.0f}% limit"),
        ok, {"rel_se_measurand": rel},
    )


def gate_degeneracy(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Flag an unidentifiable series/bulk split (framework §4.3, R2).

    Advisory *here* only because the behaviour has already happened: the engine
    selects sum-vs-split from ``ρ`` before this runs, so the gate's job is to record
    that the choice was made and why, not to ask anyone to make it.
    """
    ok = _all_pass(np.asarray(f).size)
    fit = ctx.get("fit")
    cov = getattr(fit, "covariance", None)
    if cov is None:
        return GateResult("degeneracy", FLAG, True, "no covariance available", ok)

    from softae.analysis.eis.models import roles_for

    roles = roles_for(getattr(fit, "model_name", "")) or {}
    a, b = roles.get("R_series", "R0"), roles.get("R_bulk", "R1")
    rho = cov.rho(a, b)
    threshold = float(_ctx_get(ctx, "gates", "rho_degenerate", -0.95))

    if cov.singular:
        return GateResult("degeneracy", FLAG, False,
                          "covariance singular — the split is unidentifiable", ok,
                          {"rho": float("nan")})
    if not (rho == rho):
        return GateResult("degeneracy", FLAG, True, "ρ unavailable", ok)

    passed = rho > threshold
    return GateResult(
        "degeneracy", FLAG, passed,
        f"ρ(R_series, R_bulk) = {rho:+.3f}"
        + ("" if passed else
           " — relaxation corner out of band; reporting the sum, not the split"),
        ok, {"rho": rho},
    )


def gate_model_free_crosscheck(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Compare the fit against ``1/max(Re Y)``, which uses no circuit model (§4.4).

    Agreement says the model is not mis-specified at this operating point. Divergence
    localises the problem — but note the estimator degrades as the plateau leaves the
    band (see :func:`~softae.analysis.eis.admittance.model_free_r_bulk`), so a large
    disagreement often means "the plateau is marginal here" rather than "the fit is
    wrong". Advisory for exactly that reason.
    """
    from softae.analysis.eis.admittance import model_free_r_bulk

    ok = _all_pass(np.asarray(f).size)
    fit = ctx.get("fit")
    if fit is None or not getattr(fit, "success", False):
        return GateResult("model_free_crosscheck", FLAG, True, "no fit to compare", ok)

    fitted = float(getattr(fit, "R1", float("nan")))
    free = model_free_r_bulk(Z)
    if not (fitted == fitted and free == free) or fitted <= 0:
        return GateResult("model_free_crosscheck", FLAG, True,
                          "cross-check unavailable", ok)

    pct = abs(free - fitted) / fitted * 100.0
    passed = pct <= 25.0
    return GateResult(
        "model_free_crosscheck", FLAG, passed,
        f"model-free 1/max(Re Y) differs from the fit by {pct:.0f}%"
        + ("" if passed else " — plateau may be marginal, or the model mis-specified"),
        ok, {"cross_check_pct": pct},
    )


def gate_residual_structure(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Flag *structure* in the residuals, not merely their size (framework §4.5).

    A fit can pass on norm and still be systematically wrong: long runs of one sign
    mean the model lacks a term the data contains. A runs test compares the observed
    number of sign changes against the ``n/2`` expected of random residuals.

    .. note::
       This assumes the residuals contain **noise**. A noise-free spectrum — a
       synthetic, or a heavily averaged one — leaves only systematic fit error, which
       is structured by construction and trips the test every time. That is the gate
       behaving correctly on an unphysical input, which is part of why it can only
       ever flag.
    """
    ok = _all_pass(np.asarray(f).size)
    fit = ctx.get("fit")
    z_fit = getattr(fit, "z_fit", None)
    if fit is None or z_fit is None:
        return GateResult("residual_structure", FLAG, True,
                          "no fitted curve to take residuals from", ok)

    measured = np.asarray(Z, dtype=complex)
    fitted = np.asarray(z_fit, dtype=complex)
    n = int(min(measured.size, fitted.size))
    if n < 8:
        return GateResult("residual_structure", FLAG, True,
                          "too few points for a runs test", ok)

    resid = np.real(measured[:n] - fitted[:n])
    good = np.isfinite(resid) & (resid != 0)
    signs = np.sign(resid[good])
    n_tot = int(signs.size)
    n_pos = int((signs > 0).sum())
    n_neg = n_tot - n_pos
    if n_tot < 8 or n_pos == 0 or n_neg == 0:
        return GateResult("residual_structure", FLAG, False,
                          "every residual has the same sign — the model is offset "
                          "from the data", ok, {"runs_z": float("nan")})

    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    expected = 2.0 * n_pos * n_neg / n_tot + 1.0
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_tot)) / (
        n_tot ** 2 * (n_tot - 1))
    if var <= 0:
        return GateResult("residual_structure", FLAG, True,
                          "runs-test variance undefined", ok)

    z = (runs - expected) / float(np.sqrt(var))
    passed = abs(z) <= 3.0
    return GateResult(
        "residual_structure", FLAG, passed,
        f"runs test z = {z:+.1f} ({runs} runs, {expected:.1f} expected)"
        + ("" if passed else " — residuals are structured; the model lacks a term"),
        ok, {"runs_z": float(z)},
    )


def gate_residual_norm(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Reject a fit whose residuals are so large the model does not describe the data.

    The only blocking Front-2 gate. Overhaul F11 is a fit with 10²–10³ % residuals
    still reporting an ``R1`` that ``σ = K/R`` happily consumes; the threshold sits far
    above the *grading* limit in :func:`softae.analysis.quality.grade_fit` so that this
    catches catastrophe rather than duplicating that judgement.
    """
    ok = _all_pass(np.asarray(f).size)
    fit = ctx.get("fit")
    metrics = getattr(fit, "quality", None) or {}
    rms = metrics.get("residual_rms_pct")
    if rms is None:
        return GateResult("residual_norm", BLOCK_SPECTRUM, True,
                          "no residual metrics available", ok)

    limit = float(_ctx_get(ctx, "gates", "residual_hard_pct", 100.0))
    passed = not (rms == rms and rms > limit)
    return GateResult(
        "residual_norm", BLOCK_SPECTRUM, passed,
        f"RMS residual {rms:.0f}%"
        + ("" if passed else f" — above the {limit:.0f}% ceiling; the model does not "
                             f"describe this data"),
        ok, {"residual_rms_pct": float(rms)},
    )


#: Front-2 gates. Run after fitting, with the fit supplied as ``ctx["fit"]``.
FRONT2_GATES: tuple[Callable[..., GateResult], ...] = (
    gate_residual_norm,
    gate_residual_structure,
    gate_pegged_parameters,
    gate_relative_standard_error,
    gate_degeneracy,
    gate_model_free_crosscheck,
)


# ── §3.6 Kramers–Kronig stationarity truncation ──────────────────────────────

def gate_kk_truncation(f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]) -> GateResult:
    """Remove the low-frequency run no causal linear system could have produced.

    Framework §3.6, and the asymmetry is the point: only the **contiguous failing run
    at the low-frequency end** is removed, because that is where the sweep is slow
    enough for the sample to drift and where ``R_bulk`` does not live. An isolated
    mid-band failure is a noisy point, not drift, so it is flagged and kept — removing
    it would be discarding data on a criterion that does not apply to it.

    Three outcomes, not two. A ladder that will not fit **passes with a note**: a test
    that could not run is an absence of evidence, and treating it as failure would let
    an unrelated numerical problem reject good spectra. A failing run that reaches past
    :data:`~softae.analysis.eis.kk.DEFAULT_KK_MAX_TRUNCATE_FRAC` of the band is
    *rejected rather than truncated*, because §3.6's licence to cut depends on the cut
    staying clear of the arc.
    """
    from softae.analysis.eis.kk import (
        DEFAULT_KK_C,
        DEFAULT_KK_MAX_M,
        DEFAULT_KK_MAX_TRUNCATE_FRAC,
        lin_kk,
        low_frequency_run,
    )
    from softae.analysis.eis.settings import DEFAULT_KK_RESID_PCT

    n = int(np.asarray(f).size)
    ok = _all_pass(n)
    limit = float(_ctx_get(ctx, "gates", "kk_resid_pct", DEFAULT_KK_RESID_PCT))

    result = lin_kk(f, Z,
                    blocking=bool(_ctx_get(ctx, "cell", "blocking", True)),
                    c=float(_ctx_get(ctx, "gates", "kk_c", DEFAULT_KK_C)),
                    max_M=int(_ctx_get(ctx, "gates", "kk_max_M", DEFAULT_KK_MAX_M)))
    if not result.ok:
        return GateResult("kk_truncation", FLAG, True,
                          f"K–K test did not run: {result.error}", ok)

    resid = np.asarray(result.resid_pct, dtype=float)
    failing = resid > limit
    finite = resid[np.isfinite(resid)]
    # The median is the statistic ladder-order selection minimises, so logging it is
    # what lets an operator tell "this spectrum is noisy" from "the ladder under-fit".
    metrics = {"kk_max_resid_pct": result.max_resid_pct,
               "kk_median_resid_pct": float(np.median(finite)) if finite.size
               else float("nan"),
               "kk_order_M": float(result.M), "kk_mu": float(result.mu)}

    if not failing.any():
        return GateResult(
            "kk_truncation", BLOCK_POINT, True,
            f"K–K compliant to {result.max_resid_pct:.2f}% (order M={result.M})",
            ok, metrics)

    run = low_frequency_run(f, failing)
    isolated = int((failing & ~run).sum())
    n_run = int(run.sum())
    metrics.update({"kk_truncated": float(n_run), "kk_isolated": float(isolated)})

    max_frac = float(_ctx_get(ctx, "gates", "kk_max_truncate_frac",
                              DEFAULT_KK_MAX_TRUNCATE_FRAC))
    if n and n_run / n > max_frac:
        return GateResult(
            "kk_truncation", BLOCK_SPECTRUM, False,
            f"{n_run}/{n} points fail K–K in one run from the low-frequency end "
            f"(> {max_frac:.0%}) — non-stationary across the band, not a drifting "
            f"tail; truncating this far would remove the arc that carries R_bulk",
            ok, metrics)

    detail = (f"truncated {n_run} low-f point(s) above {limit:g}% K–K residual"
              if n_run else f"no low-f run above {limit:g}% K–K residual")
    if isolated:
        detail += (f"; {isolated} isolated mid-band failure(s) flagged, not removed "
                   f"(noise or an outlier, not drift)")
    return GateResult("kk_truncation", BLOCK_POINT, n_run == 0, detail,
                      ~run, metrics)


# ── §3.7 Plateau-in-band ─────────────────────────────────────────────────────

#: Fewest points that may constitute a plateau. Below three, "flat" is a statement
#: about how finely the sweep was sampled rather than about the sample.
MIN_PLATEAU_POINTS = 3


def gate_plateau_in_band(
    f: np.ndarray, Z: np.ndarray, ctx: dict[str, Any]
) -> GateResult:
    """Is there actually a resistive plateau inside the measured band?

    Framework §3.7. If the plateau lies outside the band, the optimiser's ``R_bulk``
    is an **extrapolation, not a measurement**, and may only be reported flagged as
    such.

    **Deviation from the printed check, and why.** §3.7 estimates the plateau window
    analytically — ``f_c`` from ``R_bulk`` and ``C_par``, ``f_lo`` from the blocking
    onset — and tests that window against ``[F_MIN, F_MAX]``. On this fixture ``C_par``
    is *fixture-dominated* (≈0.35 nF, the board and mux rather than the film), so a
    predicted window would largely describe the hardware and move with it.

    So this measures the plateau that is **actually present**: the widest contiguous
    run in log-frequency over which ``Re Z`` is flat to within a tolerance of *its own
    median*. Self-referential on purpose. An earlier version compared each point
    against :func:`~softae.analysis.eis.admittance.model_free_r_bulk`, which fails on
    exactly the spectra this gate matters for — that estimator is documented as 41 %
    low at ``R_bulk ≈ 50 kΩ`` because the plateau is being squeezed, so a gate keyed to
    it reported "no plateau" on a spectrum with nearly two clean decades of one.
    Measuring flatness against the data itself has no such bias and needs no constant.

    A useful side effect: the plateau median is a better ``R`` estimate than the
    model-free one precisely where the model-free one degrades, and it is reported.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    ok = _all_pass(freq.size)
    need = float(_ctx_get(ctx, "gates", "plateau_min_decades", 0.5))
    tol = float(_ctx_get(ctx, "gates", "plateau_tol_pct", 10.0)) / 100.0

    usable = np.isfinite(freq) & (freq > 0) & np.isfinite(Zc.real)
    good = usable & (Zc.real > 0)
    if int(usable.sum()) < 2:
        # Nothing measurable — an absence of evidence, so a flag rather than a verdict.
        return GateResult("plateau_in_band", FLAG, True,
                          "too few finite points to measure a plateau", ok)
    if int(good.sum()) < 2:
        # Measurable, but with no positive resistive component anywhere. That is not
        # missing evidence, it is evidence of absence: a purely reactive response has
        # no plateau to find, and any R_bulk from it would be invented.
        return GateResult(
            "plateau_in_band", BLOCK_SPECTRUM, False,
            "no point has a positive Re Z — the response is purely reactive across "
            "the band, so there is no resistive plateau to measure", ok,
            {"plateau_decades": 0.0})

    order = np.argsort(freq[good])
    fs = freq[good][order]
    zr = Zc.real[good][order]

    # Widest contiguous window whose Re Z stays within ±tol of the window median.
    #
    # At least MIN_PLATEAU_POINTS wide, and that minimum is load-bearing rather than
    # defensive. A 41-point sweep over four decades samples every 0.1 decade, and *any*
    # smooth curve is flat to 10 % across one such gap — a two-point "plateau" measures
    # the sweep density, not the sample. Measured on a synthetic with the plateau pushed
    # far above the band (R_bulk = 1 GΩ, no plateau present at all), the flattest
    # adjacent pair still agreed to 8.7 %, so a two-point window would have certified a
    # plateau on a spectrum that has none.
    best = (0, 0, 0.0)                      # (i, j, decades)
    for i in range(fs.size):
        for j in range(fs.size - 1, i + MIN_PLATEAU_POINTS - 2, -1):
            seg = zr[i:j + 1]
            med = float(np.median(seg))
            if med <= 0:
                continue
            if float(np.max(np.abs(seg - med))) <= tol * med:
                span = float(np.log10(fs[j] / fs[i])) if fs[i] > 0 else 0.0
                if span > best[2]:
                    best = (i, j, span)
                break                        # longer j for this i cannot be narrower

    i, j, decades = best
    R = float(np.median(zr[i:j + 1])) if j > i else float("nan")
    metrics = {"plateau_decades": decades, "plateau_R_ohm": R,
               "plateau_n_points": float(j - i + 1 if j > i else 0)}

    if decades <= 0.0:
        return GateResult(
            "plateau_in_band", BLOCK_SPECTRUM, False,
            f"no {MIN_PLATEAU_POINTS} consecutive points share a Re Z flat to "
            f"{tol:.0%} — there is no resistive plateau anywhere in the measured "
            f"band, so any R_bulk would be pure extrapolation", ok, metrics)

    passed = decades >= need
    return GateResult(
        "plateau_in_band", FLAG, passed,
        f"resistive plateau spans {decades:.2f} decade(s) "
        f"({fs[i]:.3g}-{fs[j]:.3g} Hz) at R = {R:.3e} ohm"
        + ("" if passed else f" - under the {need:g} decade minimum; R_bulk is "
                             f"substantially extrapolated"),
        ok, metrics)


#: Front-1 gates that run on the **raw instrument record**, before any correction
#: (framework §6 steps 1–3, plus this rig's two additions).
#:
#: These ask one question: *did the instrument record something real?* Correcting
#: first would let a subtraction rescue a spectrum the measurement itself failed —
#: a railed point is railed, and removing a few ohms of lead does not un-rail it.
FRONT1_PRE_CORRECTION: tuple[Callable[..., GateResult], ...] = (
    gate_finiteness,
    gate_monotonic_frequency,
    gate_quadrant,
    gate_magnitude,
    gate_phase_noise_extrapolated,
    gate_stuck_instrument,
)

#: Front-1 gates that run **after** fixture correction (framework §6 steps 5–9).
#:
#: These ask a different question: *does this spectrum contain the physics being
#: extracted?* — and that is only answerable once the fixture's own contribution is
#: gone. §6 is explicit that the topology triad "must run on *corrected, truncated*
#: data — an uncorrected series parasitic or an uncorrected HF artifact can invert
#: the very slopes the triad tests." A fixture ``R_short`` **is** a series parasitic,
#: which makes :func:`gate_tand_slope` the sharp case rather than a hypothetical one.
FRONT1_POST_CORRECTION: tuple[Callable[..., GateResult], ...] = (
    gate_hf_inductive,        # §6 step 5
    gate_kk_truncation,       # §6 step 6 — K–K sees corrected, HF-truncated data
    gate_min_points,          # §6 step 7
    *TOPOLOGY_TRIAD,          # §6 step 8
    gate_valley_feature,      # §6 step 8b
    gate_plateau_in_band,     # §6 step 9
)

#: Every Front-1 gate in framework §6 execution order.
#:
#: Retained as the default for :func:`run_gates` and for callers with nothing to
#: correct: with no correction applied the split is a no-op, and running the two
#: halves back to back is identical to running this tuple once.
FRONT1_GATES: tuple[Callable[..., GateResult], ...] = (
    *FRONT1_PRE_CORRECTION,
    *FRONT1_POST_CORRECTION,
)


def blocked_by(results: Sequence[GateResult]) -> GateResult | None:
    """The first gate that rejected the spectrum outright, or ``None``.

    Lets a caller split :func:`run_gates` into stages without losing the runner's
    short-circuit: a spectrum the admission gates rejected must not go on to be
    corrected and re-gated, because every downstream verdict would then describe a
    measurement that was already inadmissible.
    """
    for r in results:
        if r.severity in (BLOCK_SPECTRUM, BLOCK_SESSION) and not r.passed:
            return r
    return None


# ── §3.7c Excitation saturation (series-level) ───────────────────────────────

#: Relative tolerance at which two independently measured points count as identical.
#: Deliberately at the float-comparison floor: this gate exists to catch a *rail*,
#: where the two spectra agree to all 15 significant figures, not to flag samples
#: that merely measure alike.
DUPLICATE_RTOL = 1e-12


def gate_cross_spectrum_duplicates(
    spectra: Sequence[tuple[Any, np.ndarray, np.ndarray]],
    ctx: dict[str, Any] | None = None,
) -> GateResult:
    """Bitwise-identical points between independent spectra prove an instrument rail.

    Framework §3.7c. An overdriven front end clips, and clipping shows three
    signatures in increasing order of certainty: two amplitudes disagreeing most at
    the **impedance minimum** (where current peaks); low-frequency points collapsing
    onto a few discrete values; and identical values between spectra of *different
    samples*, which no physical measurement produces.

    Overhaul §3.8 caught exactly this at 100 mV — two channels returning identical
    |Z| to 15 significant figures at two frequencies, 13 low-frequency points
    collapsing onto 8–9 discrete values, and a |Z| ratio deviating up to 2.3× across
    6–30 kHz.

    **The remedy is a higher current range, not a lower amplitude**, and conflating
    this with interfacial nonlinearity gets that backwards. Interfacial nonlinearity
    scales with the voltage *across the interface*, which a divider argument shows is
    small in exactly the high-impedance regime where more amplitude is wanted;
    saturation scales with *current*, so it bites hardest at the impedance minimum.
    Opposite frequency signatures, opposite fixes.

    Series-level: it needs two or more independent spectra and so cannot run inside
    :func:`run_gates`, which sees one at a time. ``block_spectrum`` applies to every
    spectrum in the affected set — a rail is a property of the acquisition, not of
    the unlucky pair that happened to collide.
    """
    del ctx  # thresholds are physical, not configurable — see DUPLICATE_RTOL
    items = [
        (label, np.asarray(fv, dtype=float), np.abs(np.asarray(zv, dtype=complex)))
        for label, fv, zv in spectra
    ]
    hits: list[tuple[Any, Any, float, float]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            li, fi, ai = items[i]
            lj, fj, aj = items[j]
            common = np.intersect1d(fi, fj)
            for fv in common:
                a = ai[fi == fv]
                b = aj[fj == fv]
                if a.size == 0 or b.size == 0:
                    continue
                if np.isclose(a[0], b[0], rtol=DUPLICATE_RTOL, atol=0.0):
                    hits.append((li, lj, float(fv), float(a[0])))

    detail = f"{len(hits)} bitwise-identical point(s) between independent spectra"
    if hits:
        li, lj, fv, val = hits[0]
        detail += (
            f" — e.g. {li} and {lj} both {val:.6e} Ω at {fv:.4g} Hz. "
            f"This is an instrument rail, not a measurement: raise the current "
            f"range (not the amplitude) and re-run the two-amplitude check."
        )
    # No per-point mask: the verdict is on the acquisition, and the affected spectra
    # have different lengths, so there is no single array this could describe.
    return GateResult(
        "cross_spectrum_duplicates", BLOCK_SPECTRUM, not hits, detail,
        np.ones(0, dtype=bool),
        {"n_duplicates": float(len(hits)), "n_spectra": float(len(items))},
    )


# ── The runner ───────────────────────────────────────────────────────────────

def run_gates(
    f: np.ndarray,
    Z: np.ndarray,
    ctx: dict[str, Any],
    gates: Sequence[Callable[..., GateResult]] = FRONT1_GATES,
    initial_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[GateResult], list[dict[str, Any]]]:
    """Apply gates in order, threading the surviving mask forward.

    Returns ``(mask, results, log)`` where *mask* indexes the original arrays, so a
    caller can always recover which of its own points survived.

    *initial_mask* continues an earlier stage's mask rather than starting from all-
    pass. Framework §6 interleaves fixture correction *between* Front-1 gates, so the
    runner has to be resumable — without this the post-correction stage would
    resurrect points the admission gates had already dropped.

    **Deviation from the printed runner.** The specification ``break``s on the first
    failed ``block_spectrum``. That makes §3.5.3's deliberate redundancy impossible —
    it calls agreement between the loss-tangent and series-RC formulations "the
    confirmation", but a spectrum failing the first would never reach the second. So
    the topology triad is run as a *group*: every member evaluates, and only then
    does the chain stop. Non-triad blocking gates still short-circuit, because there
    is no diagnostic value in running a topology test on a stuck instrument.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    n = int(min(f.size, Z.size))
    if initial_mask is None:
        mask = _all_pass(n)
    else:
        mask = np.asarray(initial_mask, dtype=bool).copy()
        if mask.size != n:
            raise ValueError(
                f"initial_mask has {mask.size} entries for {n} points")

    results: list[GateResult] = []
    log: list[dict[str, Any]] = []

    for gate in gates:
        idx = np.where(mask)[0]
        if idx.size == 0:
            # Every point has been dropped. Stopping here silently would leave the
            # spectrum with no surviving points *and* no block_spectrum entry
            # explaining why — the one outcome R17 forbids. Record the rejection
            # through the real gate rather than a synthetic message, then stop.
            r = gate_min_points(f[:0], Z[:0], ctx)
            results.append(r)
            log.append(r.as_log_entry())
            logger.warning("eis_gate_rejected", gate=r.name, detail=r.detail)
            break

        try:
            r = gate(f[idx], Z[idx], ctx)
        except Exception as exc:  # a broken gate must not discard a measurement
            logger.warning("eis_gate_raised", gate=getattr(gate, "__name__", "?"),
                           exc_info=True)
            r = GateResult(getattr(gate, "__name__", "unknown"), FLAG, True,
                           f"gate raised and was skipped: {exc}", _all_pass(idx.size))

        results.append(r)
        log.append(r.as_log_entry())

        if r.severity == BLOCK_POINT and not r.passed:
            local = np.asarray(r.mask, dtype=bool)
            if local.size == idx.size:
                mask[idx[~local]] = False
            logger.info("eis_gate_points_dropped", gate=r.name, n=r.n_dropped,
                        detail=r.detail)

        if r.severity in (BLOCK_SPECTRUM, BLOCK_SESSION) and not r.passed:
            logger.warning("eis_gate_rejected", gate=r.name, detail=r.detail)
            # The triad is evaluated in full so its members can confirm each other;
            # any other blocking failure ends the chain immediately.
            if gate not in TOPOLOGY_TRIAD:
                break

    return mask, results, log


def gate_metrics(results: Sequence[GateResult]) -> dict[str, float]:
    """Collect every gate's scalar metrics into one flat dict for reporting."""
    out: dict[str, float] = {}
    for r in results:
        out.update(r.metrics)
    out["n_dropped_total"] = float(sum(r.n_dropped for r in results))
    return out
