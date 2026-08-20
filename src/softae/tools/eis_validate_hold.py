"""The condition: approach it, prove the material stopped moving, then hold it.

This is the half of the validation harness that actuates **the temperature
controller and the RH controller**, and holds both for hours. Every unit it
needs was already built and shipped -- ``approach_setpoint``, ``SettleTracker``,
``monitored_hold``, ``RHHoldWatch``, ``classify_rh_hold``, ``rh_watchdog_config``
-- and every one of them is reused **unmodified**.

**Two policy inversions live here and in no shared module.** Both are one-line
decisions at this call site, and both are written down rather than inherited,
because the shipped behaviour is correct for its own caller and this caller is
not that one:

===================================  ==============================  ====================
shipped                              why it is right there           what this does
===================================  ==============================  ====================
``approach_setpoint`` returns        RH-unmet at a temperature is a  **refuses to start**
``reached=False`` and the caller     primary experimental *result*   -- one bounded retry,
continues                            of the characterization run     then park, exit non-zero
``SettleTracker`` ``ceiling`` /      a campaign must keep going; a   **refuses to start.**
``not_evaluable`` is recorded and    single settle phase must never  An uninterpretable
the campaign continues               park it                         comparison is worse
                                                                     than no comparison
===================================  ==============================  ====================

A validation run on an unequilibrated cell measures the drying transient, not
the material, and no amount of downstream statistics recovers from that.

**One inherited default is deliberately not inherited.**
``DEFAULT_RH_APPROACH_TIMEOUT_S`` is 1800 s, which is shorter than a *measured*
RH approach: the 2026-08-11 run took **~5000 s** to descend from ~22 % to a
commanded 15 % at 85 C. Inheriting 1800 s blindly would make the harness refuse
to start on exactly the conditions that most need a hold. The default here is
:data:`DEFAULT_RH_APPROACH_TIMEOUT_S` = 5400 s, and the projection prints why.

**The settle rounds pay for themselves twice.** They are the stability gate
*and* the arc-capture watch: running ``arc_closure`` over spectra that were
going to be taken anyway yields an apex histogram for the whole strip **before a
single reference sweep is spent**, which is what makes the setpoint -- the lever
that actually decides whether the run produces evidence -- decidable in advance.

**The settle gate proves the RIG stopped moving; the soak is for the SAMPLE.**
Chamber air, the stage's thermal mass and the RH sensor settle on one timescale;
a polymer film taking up or shedding water at a new RH settles on its own, far
longer one. ``SettleTracker`` watches sigma across rounds and so does see the
sample -- but it certifies *the trailing window is flat*, which a slow uptake can
satisfy locally while the film is still hours from equilibrium. Measuring there
confounds the scout-vs-reference comparison this harness exists to make, and it
would surface as *trend* in the drift re-checks rather than as noise, which is
the one failure mode paired differences cannot absorb. :func:`soak_phase` closes
that gap: a floor on **continuous time at condition** before the first spectrum.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

#: NOT the shipped 1800 s. See the module docstring.
DEFAULT_RH_APPROACH_TIMEOUT_S = 5400.0
DEFAULT_TEMP_APPROACH_TIMEOUT_S = 1800.0
#: A descent takes far longer than a climb, so a cooling leg gets the long bound.
DEFAULT_TEMP_DESCENT_TIMEOUT_S = 5400.0
DEFAULT_SETTLE_MAX_HOLD_S = 5400.0
DEFAULT_MIN_TREATMENT = 6
DEFAULT_DRIFT_CHECK = 3

#: No soak unless one is asked for. Every invocation and every test written
#: before the soak existed must behave exactly as it did, and 0 is the only
#: default that guarantees it.
DEFAULT_SOAK_S = 0.0
#: How much wall-clock the soak may spend *beyond* the soak itself, recovering
#: from excursion restarts, before it refuses. Not a knob: it is
#: :func:`_approach_one`'s "one bounded retry, then refuse" policy expressed in
#: time rather than in attempts, because excursions come in blips of unequal
#: length and counting them would refuse a run over three harmless ones while
#: tolerating one long enough to matter.
SOAK_CEILING_FACTOR = 2.0
#: Between-poll interval during a soak. Matches ``approach_condition``'s own
#: default, and must stay well under the excursion grace windows (120 s
#: temperature, 600 s RH) or a sustained excursion could never accumulate the
#: two consecutive samples ``sustained_above`` needs to span one.
SOAK_POLL_INTERVAL_S = 30.0
#: One progress line per this many polls -- 5 min at the default interval. A
#: four-hour soak printing every poll is 480 lines of nothing happening, which
#: is how an operator learns to stop reading the console.
SOAK_PRINT_EVERY_N_POLLS = 10

TEMP_CONTROLLER = "temp_controller"
RH_CONTROLLER = "rh_controller"


class VirtualClock:
    """A clock that advances only when the code under it waits.

    ``approach_setpoint`` and the settle loop both bound themselves on elapsed
    wall-clock, so a mock run with the sleeps removed would spin for the real
    timeout -- 1800 s of tight loop -- while measuring nothing. Handing both the
    same object as ``now`` *and* ``sleep`` makes the timeout mean "this many
    seconds of simulated waiting", which is the thing under test.

    Never used on a real run: there the clock is the wall's, because the rig's
    thermal mass is.
    """

    __slots__ = ("t",)

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


def _observe(hook: Any, event: str, *args: Any) -> None:
    """Call an observer hook and **swallow whatever it does**.

    Both long phases publish themselves through injected hooks rather than
    imported sidecars, so this module keeps knowing nothing about narration --
    and a monitoring convenience must never be the reason a phase refuses. One
    implementation, because two copies of "never raise" is one copy too many for
    the guarantee to stay true.
    """
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:                                     # pragma: no cover
        logger.warning(event, exc_info=True)


class RefuseToStart(RuntimeError):
    """The condition was never established, so no sweep may be taken.

    Deliberately not a warning and not a recorded-and-continued outcome: this is
    the class of failure whose whole point is that it must stop the run *before*
    any reference sweep, because a measurement taken now would look exactly like
    a measurement taken correctly.
    """


# ── The plan ─────────────────────────────────────────────────────────────────

@dataclass
class ValidationPlan:
    """What was asked. Fingerprinted, so a resume cannot silently change it."""

    validation_name: str
    channels: tuple[int, ...]
    rh_setpoint_pct: float
    temp_setpoint_c: float
    baseline_preset: str
    baseline_source: str
    reference_preset: str
    drift_check: int = DEFAULT_DRIFT_CHECK
    min_treatment: int = DEFAULT_MIN_TREATMENT
    order: str = "ref-first"
    max_follow_ups: int = 1
    rh_tolerance_pct: float = 2.0
    tolerance_c: float = 2.0
    rh_approach_timeout_s: float = DEFAULT_RH_APPROACH_TIMEOUT_S
    temp_approach_timeout_s: float = DEFAULT_TEMP_APPROACH_TIMEOUT_S
    settle: bool = True
    settle_max_hold_s: float = DEFAULT_SETTLE_MAX_HOLD_S
    #: Seconds of **continuous time at condition** required before the first
    #: spectrum. Set from ``--soak-h``, stored in seconds so it is uniform with
    #: every other duration on this plan. The settle phase runs at condition and
    #: therefore counts against it -- see :func:`soak_phase`.
    soak_s: float = DEFAULT_SOAK_S
    end_state: str = "park"
    retries: int = 1
    max_consecutive_failures: int = 3
    visit: int = 1
    mock: bool = False

    #: The two frequencies the whole partition turns on, resolved from the real
    #: presets rather than hard-coded so an edited preset moves them with it.
    ref_close_hz: float = 0.0
    baseline_ok_hz: float = 0.0
    #: ``[eis.scout] band_below_apex_min_decades`` as it stood for this run.
    band_min_decades: float = 1.0

    def cell_key(self, channel: int) -> str:
        return (f"{int(channel)}:{self.rh_setpoint_pct:g}:"
                f"{self.temp_setpoint_c:g}:{self.visit}")

    def fingerprint(self) -> str:
        """What a resume must match.

        Only the fields that change *what is being measured* -- not timeouts,
        not retry counts, not the end state. Continuing a validation whose
        channel set or condition moved would append one experiment's
        observations to another's, which corrupts both and is not visible in
        the resulting data.

        **``soak_s`` is in, and every other duration is out.** The line is not
        seconds-versus-not, it is ceiling-versus-floor. ``settle_max_hold_s`` and
        the approach timeouts are *ceilings on waiting for a criterion*: the
        criterion decides the sample's state and the ceiling only decides how
        long the harness is willing to wait for it, so moving one cannot move
        what was measured. ``soak_s`` is a **floor that sets the state
        directly** -- a film 30 min into an RH step and the same film 6 h in are
        different specimens, and pooling their cells is the exact corruption
        this hash exists to refuse. The operator-facing cost is stated rather
        than hidden: ``--resume`` must repeat the soak it started with, and a
        resume that simply forgets the flag is refused instead of quietly
        measuring an unsoaked sample into a soaked dataset.
        """
        import hashlib

        parts = "|".join(str(p) for p in (
            self.validation_name, self.channels, self.rh_setpoint_pct,
            self.temp_setpoint_c, self.baseline_preset, self.reference_preset,
            self.order, self.max_follow_ups, self.visit, self.soak_s,
        ))
        return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(self).items()
        }
        payload["fingerprint"] = self.fingerprint()
        return payload


def validate_plan(plan: ValidationPlan) -> None:
    """Refuse impossible plans **before** anything is heated.

    Two checks today, and neither is hypothetical: ``settle_check`` requires at
    least ``DEFAULT_SETTLE_MIN_CHANNELS`` participating channels, so a run on
    fewer than that can never return ``settled`` -- it runs every round to the
    ceiling and then refuses. Caught late that costs the full
    ``--settle-max-hold-s`` (90 minutes by default) at temperature, and the
    operator's first evidence is a refusal that names the wrong cause.
    """
    from softae.analysis.equilibration import DEFAULT_SETTLE_MIN_CHANNELS

    # A negative soak would sail through `soak_phase` as "already satisfied" and
    # read on the projection as a *shortened* run, so it is refused rather than
    # clamped: the operator who typed it meant something, and silently meaning 0
    # is the reading least likely to be it.
    if plan.soak_s < 0:
        raise RefuseToStart(
            f"--soak-h is negative ({plan.soak_s / 3600:g} h). A soak is a floor "
            "on time at condition; there is no such thing as a negative one."
        )
    if plan.settle and len(plan.channels) < DEFAULT_SETTLE_MIN_CHANNELS:
        raise RefuseToStart(
            f"{len(plan.channels)} channel(s) is below the settle gate's "
            f"minimum of {DEFAULT_SETTLE_MIN_CHANNELS}: the criterion could "
            "never be evaluated, so the run would hold to its ceiling and then "
            "refuse anyway. Add channels, or state --settle off and accept a "
            "withheld outcome."
        )


def population_thresholds(
    baseline_f_lo_hz: float, reference_f_lo_hz: float, band_decades: float
) -> tuple[float, float]:
    """``(ref_close_hz, baseline_ok_hz)`` -- the edges of the resolving window.

    A preset closes an arc only when its floor sits ``band_decades`` below the
    apex, so the lowest apex it can close is ``f_lo * 10 ** band_decades``. With
    the shipped ``Quick`` baseline and ``Extended`` reference and the shipped
    1.0-decade cut, that is 64.75 Hz and 13.51 Hz: a resolving window **0.68
    decades wide**, which is the design's main risk and the reason the
    arc-capture watch exists.
    """
    factor = 10.0 ** float(band_decades)
    return float(reference_f_lo_hz) * factor, float(baseline_f_lo_hz) * factor


def classify_apex(apex_hz: float, plan: ValidationPlan) -> str:
    from softae.tools.eis_validate_report import CONTROL, TREATMENT, UNRESOLVED

    if not (math.isfinite(apex_hz) and apex_hz > 0):
        return UNRESOLVED
    if apex_hz >= plan.baseline_ok_hz:
        return CONTROL
    if apex_hz >= plan.ref_close_hz:
        return TREATMENT
    return UNRESOLVED


# ── Projection ───────────────────────────────────────────────────────────────

@dataclass
class Projection:
    n_channels: int
    reference_s: float
    scout_s: float
    follow_up_s: float
    drift_s: float
    settle_round_s: float

    @property
    def measurement_low_s(self) -> float:
        return self.reference_s + self.scout_s + self.drift_s

    @property
    def measurement_high_s(self) -> float:
        return self.measurement_low_s + self.follow_up_s


def project(plan: ValidationPlan) -> Projection:
    """From ``preflight``'s measured anchors, quoted as a range.

    How many cells extend is **not knowable in advance** -- that is the whole
    question -- so the follow-up term is a bound, not an estimate.
    """
    from softae.core.eis_scripts import EISParams
    from softae.core.preflight import estimate_eis_duration
    from softae.workflows.equilibration import default_round_period_s

    n = len(plan.channels)
    reference = estimate_eis_duration(EISParams.from_preset(plan.reference_preset))
    baseline = estimate_eis_duration(EISParams.from_preset(plan.baseline_preset))
    follow_up = estimate_eis_duration(EISParams.from_preset(_next_rung(plan)))
    return Projection(
        n_channels=n,
        reference_s=reference * n,
        scout_s=baseline * n,
        follow_up_s=follow_up * n,
        drift_s=reference * max(0, int(plan.drift_check)),
        settle_round_s=default_round_period_s(plan.baseline_preset, n),
    )


def _next_rung(plan: ValidationPlan) -> str:
    """The preset a blind one-rung follow-up would land on, for the projection."""
    from softae.config.loader import eis_presets
    from softae.core.eis_scripts import EISParams

    try:
        current = EISParams.from_preset(plan.baseline_preset).f_lo_mHz
        wider = [
            (name, EISParams.from_preset(name).f_lo_mHz)
            for name in (eis_presets() or {})
        ]
        candidates = [(n, f) for n, f in wider if 0 < f < current]
        if candidates:
            return max(candidates, key=lambda pair: pair[1])[0]
    except Exception:                                     # pragma: no cover
        pass
    return plan.baseline_preset


def render_projection(plan: ValidationPlan, projection: Projection) -> str:
    """The block printed before the confirmation prompt. ASCII only."""
    lines: list[str] = []
    add = lines.append
    n = projection.n_channels
    add("")
    add(f"Projected run -- {n} channels, RH {plan.rh_setpoint_pct:g} %, "
        f"T {plan.temp_setpoint_c:g} C")
    add(f"                 baseline {plan.baseline_preset}, reference "
        f"{plan.reference_preset}, drift-check {plan.drift_check}")
    add("")
    add(f"  {'phase':<44}{'time':>14}{'bound':>10}")
    add("  " + "-" * 68)
    add(f"  {'approach   temperature':<44}{'0 - 30 min':>14}"
        f"{plan.temp_approach_timeout_s / 60:>7.0f} min")
    add(f"  {'approach   RH':<44}{'0 - 30 min':>14}"
        f"{plan.rh_approach_timeout_s / 60:>7.0f} min")
    settle_label = (f"settle     {plan.baseline_preset} rounds "
                    f"(~{projection.settle_round_s:.0f} s/round)")
    add(f"  {settle_label:<44}{'25 - 45 min':>14}"
        f"{plan.settle_max_hold_s / 60:>7.0f} min")
    # Its own row, always, including at 0. The operator reads this table and
    # types "yes" against it, so a soak that did not appear here would be hours
    # the projection lied about -- and an absent soak is itself a decision worth
    # seeing. `time` is what the soak will *add* (settle time already counts
    # against it, so a settle longer than the soak adds nothing); `bound` is the
    # ceiling the excursion restarts are held under.
    soak_add = ("0 min" if plan.soak_s <= 0
                else f"0 - {plan.soak_s / 60:.0f} min")
    add(f"  {'soak       hold at condition (--soak-h)':<44}{soak_add:>14}"
        f"{plan.soak_s * SOAK_CEILING_FACTOR / 60:>7.0f} min")
    add("  " + "-" * 68)
    add(f"  R   reference   {plan.reference_preset:<12}"
        f"{projection.reference_s / 60:>8.1f} min")
    add(f"  B0  scout       {plan.baseline_preset:<12}"
        f"{projection.scout_s / 60:>8.1f} min")
    add(f"  B1  follow-up   {_next_rung(plan):<12}"
        f"{0.0:>8.1f} - {projection.follow_up_s / 60:.1f} min "
        "(extending cells only)")
    add(f"  drift re-check  {plan.reference_preset:<12}"
        f"{projection.drift_s / 60:>8.1f} min   x {plan.drift_check}")
    add("  " + "-" * 68)
    add(f"  {'measurement':<44}"
        f"{projection.measurement_low_s / 60:.0f} - "
        f"{projection.measurement_high_s / 60:.0f} min")
    add("")
    add("  LOWER BOUND on the measurement block: preflight counts declared dwells")
    add("  and ignores comms overhead, script build, and analysis. A segmented")
    add("  follow-up is MODELLED, never stopwatched -- this grid is generated per")
    add("  sample, so no timing anchor can ever match it.")
    add("")
    add(f"  RH approach is bounded at {plan.rh_approach_timeout_s:.0f} s, not the")
    add("  shipped default of 1800 s: an RH descent from ~22 % to a commanded 15 %")
    add("  at 85 C was MEASURED at ~5000 s (2026-08-11). The shipped default would")
    add("  refuse to start on exactly the conditions that most need a hold.")
    add("")
    if plan.soak_s > 0:
        add(f"  SOAK {plan.soak_s / 3600:.2f} h: the settle gate proves the RIG "
            "stopped moving; this")
        add("  is the SAMPLE's own equilibration. The clock starts when the "
            "condition is")
        add("  ESTABLISHED -- i.e. when the approach completes -- so the settle "
            "rounds,")
        add("  which run at condition, count against it and only the remainder "
            "is waited.")
        add("")
    add(f"  RESOLVING WINDOW: apex in [{plan.ref_close_hz:.2f}, "
        f"{plan.baseline_ok_hz:.2f}) Hz -- "
        f"{math.log10(plan.baseline_ok_hz / plan.ref_close_hz):.2f} decades.")
    add("  Outside it the two arms are identical (above) or the reference is not")
    add("  a reference (below). The SETPOINT is the lever that decides how many")
    add("  cells land inside it; the sample size is not.")
    return "\n".join(lines)


# ── Approach ─────────────────────────────────────────────────────────────────

@dataclass
class ApproachReport:
    axis: str
    target: float
    reached: bool
    elapsed_s: float
    pv_final: float
    attempts: int


def approach_condition(
    manager: Any,
    plan: ValidationPlan,
    *,
    sleep: Any = None,
    now: Any = None,
    poll_interval_s: float = 30.0,
) -> list[ApproachReport]:
    """Temperature first, then RH. Refuses rather than proceeding unequilibrated.

    **Order matters.** The attainable RH floor *rises with temperature* -- 15 %RH
    commanded gave 16.9-23.2 % PV at 65-85 C -- so approaching RH before the
    block is at temperature is approaching a target that is about to move.
    """
    from softae.workflows.equilibration import approach_setpoint

    temp = manager.get(TEMP_CONTROLLER)
    rh = manager.get(RH_CONTROLLER)
    reports: list[ApproachReport] = []

    temp.write_sp(float(plan.temp_setpoint_c))
    reports.append(_approach_one(
        approach_setpoint, lambda: float(temp.get_pv(1)),
        plan.temp_setpoint_c, axis="temperature", instrument=TEMP_CONTROLLER,
        tolerance=plan.tolerance_c, timeout_s=plan.temp_approach_timeout_s,
        poll_interval_s=poll_interval_s, sleep=sleep, now=now,
    ))

    rh.set_setpoint(float(plan.rh_setpoint_pct))
    rh.start()
    reports.append(_approach_one(
        approach_setpoint, lambda: float(rh.get_H()),
        plan.rh_setpoint_pct, axis="rh", instrument=RH_CONTROLLER,
        tolerance=plan.rh_tolerance_pct, timeout_s=plan.rh_approach_timeout_s,
        poll_interval_s=poll_interval_s, sleep=sleep, now=now,
    ))
    return reports


def _approach_one(
    approach_setpoint: Any, read_pv: Any, target: float, *, axis: str,
    instrument: str, tolerance: float, timeout_s: float,
    poll_interval_s: float, sleep: Any, now: Any,
) -> ApproachReport:
    """One axis, one bounded retry, then refuse. **The first policy inversion.**"""
    elapsed = 0.0
    for attempt in (1, 2):
        outcome = approach_setpoint(
            read_pv, float(target), axis=axis, instrument=instrument,
            tolerance=float(tolerance), timeout_s=float(timeout_s),
            poll_interval_s=float(poll_interval_s), sleep=sleep, now=now,
        )
        elapsed += float(outcome.elapsed_s)
        if outcome.reached:
            logger.info("eis_validate_approach_reached", axis=axis,
                        target=float(target), pv=float(outcome.pv_final),
                        elapsed_s=elapsed, attempts=attempt)
            return ApproachReport(axis, float(target), True, elapsed,
                                  float(outcome.pv_final), attempt)
        logger.warning("eis_validate_approach_timeout", axis=axis,
                       target=float(target), pv=float(outcome.pv_final),
                       attempt=attempt, timeout_s=float(timeout_s))
    raise RefuseToStart(
        f"{axis} never reached {target:g} within {tolerance:g} after two "
        f"attempts of {timeout_s:.0f} s (last PV {outcome.pv_final:g}). "
        "A validation run on an unequilibrated cell measures the drying "
        "transient, not the material -- refusing to start."
    )


# ── Settle, and the arc-capture watch ────────────────────────────────────────

@dataclass
class SettleOutcome:
    verdict: str
    n_rounds: int
    elapsed_s: float
    apex_by_channel: dict[int, float] = field(default_factory=dict)
    projected: dict[str, int] = field(default_factory=dict)
    rh_median_pct: float = float("nan")

    @property
    def certified(self) -> bool:
        return self.verdict == "settled"


def settle_phase(
    manager: Any,
    plan: ValidationPlan,
    measure: Callable[[int], Any],
    *,
    min_hold_first_s: float | None = None,
    round_period_s: float | None = None,
    sleep: Any = None,
    now: Any = None,
    on_round: Callable[[dict[str, Any]], None] | None = None,
) -> SettleOutcome:
    """Quick rounds until the *material* stops moving -- and an apex histogram.

    The condition that licenses the run is not that the room's PV is steady, it
    is that the **sample** has stopped changing, which is precisely what
    :class:`~softae.analysis.equilibration.SettleTracker` decides. It is reused
    unmodified, including its self-referential RH clause: the trailing window's
    RH spread is compared against *itself*, never against the setpoint, so the
    gate still works when RH is floor-limited-but-steady.

    *measure* takes a channel and returns an ``EISResult``; the caller owns
    acquisition so this stays testable without a rig.

    *on_round* ``(payload)`` fires once per round with the gate's own state --
    the same hook shape :func:`soak_phase` uses, injected for the same reason
    and **swallowed** for the same one. Two runs (2026-08-20) died in this gate
    without acquiring a single spectrum, and the only place the per-round
    numbers ever existed was console scrollback: the settle sweeps are not
    persisted, so *which* channel held the gate was unrecoverable afterwards.
    :mod:`softae.tools.eis_validate` turns this into the run's published stream.
    """
    from softae.analysis.eis.arc import arc_closure
    from softae.analysis.equilibration import (
        DEFAULT_MIN_HOLD_FIRST_S,
        DEFAULT_RH_STABILITY_PCT,
        SETTLE_DISABLED,
        SettleTracker,
    )
    from softae.workflows.equilibration import default_round_period_s

    if not plan.settle:
        return SettleOutcome(SETTLE_DISABLED, 0, 0.0, {}, {}, float("nan"))

    sleep = sleep or time.sleep
    now = now or time.monotonic
    floor_s = (DEFAULT_MIN_HOLD_FIRST_S if min_hold_first_s is None
               else float(min_hold_first_s))
    period_s = (default_round_period_s(plan.baseline_preset, len(plan.channels))
                if round_period_s is None else float(round_period_s))

    tracker = SettleTracker(
        enabled=True, rh_stability_pct=DEFAULT_RH_STABILITY_PCT)
    rh = manager.get(RH_CONTROLLER)
    apexes: dict[int, float] = {}
    start = float(now())
    rounds = 0
    stopped_early = False

    while True:
        fits: list[Any] = []
        for channel in plan.channels:
            try:
                eis = measure(channel)
            except Exception as exc:
                logger.warning("eis_validate_settle_sweep_failed",
                               channel=channel, error=str(exc))
                continue
            fits.append(_round_fit(channel, eis))
            closure = arc_closure(eis.frequency, eis.z_imag_neg,
                                  getattr(eis, "phase", None))
            apex = float(closure.f_apex_interior_hz)
            if math.isfinite(apex) and apex > 0:
                apexes[int(channel)] = apex
        rounds += 1

        rh_median = _read_rh(rh)
        check = tracker.observe(fits, rh_median_pct=rh_median)
        elapsed = float(now()) - start
        deviations = settle_deviations(
            tracker.rounds[-tracker.n_rounds:],
            check.participating if check is not None else [])
        worst_channel = (max(deviations, key=deviations.__getitem__)
                         if deviations else None)
        drift = (check.max_deviation_rel
                 if check is not None and check.max_deviation_rel is not None
                 else float("nan"))
        rh_text = "  n/a" if rh_median is None else f"{rh_median:5.1f}"
        state = "SETTLED" if tracker.settled else "not yet"
        n_in = (f"{len(check.participating)}" if check is not None
                else "-")   # no trailing window yet: not zero channels, no verdict
        # Named, bounded and attributed. `spread 0.130` was read as %RH by an
        # operator who then nearly loosened the tolerance to 1.0 -- which is
        # 100 % relative deviation, i.e. accepting a film whose conductivity
        # doubled between rounds. It is a relative deviation of sigma from its
        # own window mean, the threshold belongs beside it, and a single bad
        # cell holds all fifteen (`settle_check` takes the max), so the cell is
        # named. `settle_check`'s own reason string already formats it this way.
        drift_text = "    n/a" if math.isnan(drift) else f"{drift * 100:6.2f}%"
        worst_text = "n/a " if worst_channel is None else f"ch{worst_channel:<3}"
        print(f"[settle] round {rounds:<3} RH {rh_text} %RH  "
              f"sigma drift {drift_text} (tol {tracker.tol_rel * 100:.2f}%)  "
              f"worst {worst_text} channels {n_in}/{len(plan.channels)}  "
              f"-> {state}", flush=True)
        if check is not None and not check.evaluable:
            print(f"         not evaluable: {check.reason}", flush=True)
        _observe(on_round, "eis_validate_settle_observer_failed", {
            "round": rounds,
            "elapsed_s": round(elapsed, 1),
            "rh_median_pct": None if rh_median is None else round(rh_median, 2),
            "rh_spread_pct": (None if tracker.rh_spread_pct is None
                              else round(tracker.rh_spread_pct, 3)),
            "tol_rel": tracker.tol_rel,
            "worst_deviation_rel": None if math.isnan(drift) else round(drift, 5),
            "worst_channel": worst_channel,
            "deviation_rel_by_channel": {str(ch): round(value, 5)
                                         for ch, value in sorted(deviations.items())},
            "participating": list(check.participating) if check is not None else [],
            "n_channels": len(plan.channels),
            "evaluable": None if check is None else bool(check.evaluable),
            "settled": bool(tracker.settled),
            "reason": "" if check is None else check.reason,
        })

        if tracker.settled and elapsed >= floor_s:
            stopped_early = True
            break
        if elapsed >= plan.settle_max_hold_s:
            break
        sleep(max(0.0, period_s))

    verdict = tracker.outcome(stopped_early=stopped_early)
    projected = _project_populations(apexes, plan)
    return SettleOutcome(
        verdict=verdict, n_rounds=rounds, elapsed_s=float(now()) - start,
        apex_by_channel=apexes, projected=projected,
        rh_median_pct=_read_rh(rh) if rh is not None else float("nan"),
    )


def settle_deviations(window: Any, participating: Any) -> dict[int, float]:
    """Per-channel relative deviation over *window* -- the max of which is the
    gate's own ``max_deviation_rel``.

    :class:`~softae.analysis.equilibration.SettleCheck` reports the **max**
    across participating channels and not which channel produced it, so one bad
    cell holding all fifteen is invisible in the verdict. The arithmetic is
    restated here rather than asked for there because
    :mod:`softae.analysis.equilibration` is shared with the equilibration
    workflow, and it is three lines: ``max|sigma - mean| / |mean|`` over the
    channel's own window, exactly as ``settle_check`` computes it. A test pins
    the two against each other, which is what keeps the restatement honest.

    Only *participating* channels are considered -- and they are the only ones
    for which the quantity exists, since participation is precisely the
    guarantee that every round in the window carries a finite, non-railed sigma.
    """
    wanted = {int(channel) for channel in participating}
    if not wanted:
        return {}
    series: dict[int, list[float]] = {channel: [] for channel in wanted}
    for fits in window:
        for fit in fits:
            channel = int(fit.channel)
            if channel in wanted and fit.sigma is not None:
                series[channel].append(float(fit.sigma))
    deviations: dict[int, float] = {}
    for channel, sigmas in series.items():
        if not sigmas:
            continue
        mean = sum(sigmas) / len(sigmas)
        if math.isfinite(mean) and mean != 0.0:
            deviations[channel] = max(abs(s - mean) for s in sigmas) / abs(mean)
    return deviations


def band_by_channel(
    apexes: dict[int, float], plan: ValidationPlan
) -> dict[int, str]:
    """Which side of the resolving window each channel projects onto.

    One implementation for the three callers that need it -- the projected
    counts, the console histogram, and the run's narration -- so a channel
    cannot be CONTROL in one of them and TREATMENT in another.
    """
    return {int(channel): classify_apex(
                apexes.get(int(channel), float("nan")), plan)
            for channel in plan.channels}


def _project_populations(
    apexes: dict[int, float], plan: ValidationPlan
) -> dict[str, int]:
    from softae.tools.eis_validate_report import CONTROL, TREATMENT, UNRESOLVED

    counts = {CONTROL: 0, TREATMENT: 0, UNRESOLVED: 0}
    for band in band_by_channel(apexes, plan).values():
        counts[band] += 1
    return counts


def render_arc_watch(outcome: SettleOutcome, plan: ValidationPlan, *,
                     max_per_band: int = 8) -> str:
    """The histogram, from spectra that were going to be taken anyway.

    **Printed on every path, including the refusals**, because it is built from
    sweeps already taken and it is the single most useful thing to know when the
    settle gate fails: whether the cells are even in the resolving window. The
    per-channel listing is here for the same reason -- an operator deciding
    whether to move the setpoint needs to know *which* cells are where, not only
    how many -- and is bounded at *max_per_band* entries so fifteen channels do
    not become an unreadable line.
    """
    from softae.tools.eis_validate_report import CONTROL, TREATMENT, UNRESOLVED

    counts = outcome.projected
    bands: dict[str, list[str]] = {UNRESOLVED: [], TREATMENT: [], CONTROL: []}
    for channel, band in band_by_channel(outcome.apex_by_channel, plan).items():
        apex = outcome.apex_by_channel.get(channel, float("nan"))
        shown = f"{apex:.1f}" if (math.isfinite(apex) and apex > 0) else "n/a"
        bands[band].append(f"ch{channel} {shown}")

    lines = [
        f"[watch ] apex histogram: <{plan.ref_close_hz:.1f} Hz: "
        f"{counts.get(UNRESOLVED, 0)} | {plan.ref_close_hz:.1f}-"
        f"{plan.baseline_ok_hz:.1f} Hz: {counts.get(TREATMENT, 0)} | "
        f">{plan.baseline_ok_hz:.1f} Hz: {counts.get(CONTROL, 0)}",
        f"         projected  UNRESOLVED {counts.get(UNRESOLVED, 0)}  "
        f"TREATMENT {counts.get(TREATMENT, 0)}  CONTROL {counts.get(CONTROL, 0)}"
        f"   (no extra sweeps were taken to build this)",
        "         apex (Hz) by channel:",
    ]
    for band in (UNRESOLVED, TREATMENT, CONTROL):
        entries = bands[band]
        if not entries:
            continue
        head, rest = entries[:max_per_band], entries[max_per_band:]
        tail = f"  (+{len(rest)} more)" if rest else ""
        lines.append(f"         {band:<11}" + "  ".join(head) + tail)
    return "\n".join(lines)


def assert_settle_licensed(outcome: SettleOutcome) -> None:
    """**The second policy inversion.** ``ceiling`` and ``not_evaluable`` refuse.

    The campaign path records either and continues, correctly -- a campaign's
    job is to keep going. This harness must not produce an uninterpretable
    comparison, so it stops instead. ``disabled`` is allowed through, because
    ``--settle off`` is a stated choice, but every row it produces is stamped
    ``hold_certified = "disabled"`` and the outcome is withheld: never a silent
    proceed.
    """
    from softae.analysis.equilibration import (
        SETTLE_CEILING,
        SETTLE_DISABLED,
        SETTLE_NOT_EVALUABLE,
    )

    if outcome.verdict in (SETTLE_CEILING, SETTLE_NOT_EVALUABLE):
        raise RefuseToStart(
            f"the settle gate returned `{outcome.verdict}` after "
            f"{outcome.n_rounds} rounds ({outcome.elapsed_s / 60:.1f} min). "
            "The material was never shown to have stopped moving, and "
            "'undeclared is unknown, never empty' -- refusing to start."
        )
    if outcome.verdict == SETTLE_DISABLED:
        print("  ! --settle off: every row is stamped hold_certified=disabled "
              "and the decision-rule outcome will be WITHHELD.")


# ── The soak ─────────────────────────────────────────────────────────────────

@dataclass
class SoakOutcome:
    """What the soak actually delivered, as opposed to what was asked."""

    #: Continuous time at condition credited when the first spectrum was
    #: licensed. Never less than ``plan.soak_s`` on a return.
    soaked_s: float
    #: Wall-clock spent inside :func:`soak_phase`. Zero when the settle phase
    #: already covered the soak.
    waited_s: float
    #: Time at condition inherited from the settle phase, which runs there.
    settle_credit_s: float
    #: How many times a warn-grade excursion reset the continuity clock.
    restarts: int = 0


def soak_phase(
    plan: ValidationPlan,
    watch: Any,
    *,
    established_at: float,
    sleep: Any = None,
    now: Any = None,
    poll_interval_s: float = SOAK_POLL_INTERVAL_S,
    on_poll: Callable[[int, float, float, int], None] | None = None,
    on_restart: Callable[[int, float, float], None] | None = None,
) -> SoakOutcome:
    """Hold the established condition for ``plan.soak_s`` before any spectrum.

    **The clock starts when the condition is ESTABLISHED, not when this is
    called.** *established_at* is the instant ``approach_condition`` returned, so
    the settle rounds -- which run at condition, for 25-45 minutes -- count
    against the soak, and only the remainder is waited. The alternative, starting
    the clock where this function sits in the sequence, was rejected: it charges
    the operator twice for time the sample has already spent at the new RH, and
    the quantity the soak asserts is *time at condition*, which is indifferent to
    whether a Quick round was running during it. Time spent *approaching*
    setpoint is excluded for the mirror-image reason -- it is not time at
    condition at all.

    **It runs with ``--settle off`` too, and that is when it matters most.**
    Disabling the settle gate removes the only evidence that anything stopped
    moving; the soak is then the sole thing standing between the approach and the
    first sweep, and it is the instrument by which an operator who has taken the
    equilibration judgement into their own hands actually exercises it. Skipping
    the soak because the outcome will be withheld anyway was rejected on those
    grounds. It falls out of the clock rule with no special case: a disabled
    settle returns immediately, so the credit is ~0 and the full soak is waited.

    **The soak watches; it does not sit idle.** A soak that drifted out of
    tolerance and then measured anyway would be worse than no soak at all -- it
    would attach a *claim* of equilibration to a sample that had been moved. So
    ``watch.poll()`` runs on the same cadence the approach uses, which gives the
    two graded verdicts their existing consequences for free:

    ``fault``
        :class:`~softae.errors.SafetyError` propagates out of ``poll`` and the
        runner parks and exits non-zero. No new path.
    ``warn``
        **restarts the continuity clock.** The soak asserts *continuous* time at
        condition and an excursion is the negation of continuity, so an elapsed
        count that survived one would be a false certificate. Recorded and
        continued -- the shipped posture, and the one this harness keeps
        elsewhere -- was rejected here for that reason, and aborting outright was
        rejected because ``HoldWatch``'s temperature warn is an *instantaneous*
        test and a single blip is not evidence that the condition cannot be held.
        Restarts are bounded by :data:`SOAK_CEILING_FACTOR`; a condition that
        cannot hold itself for the soak will not hold for the measurement block
        either, so exceeding the ceiling refuses rather than proceeding.

    **The two observers, and why they are hooks rather than prints.**
    *on_poll* ``(polls, soaked_s, target_s, restarts)`` fires after every poll and
    *on_restart* ``(restart, lost_s, target_s)`` after every excursion reset.
    They exist because a soak is hours of a process saying nothing to anyone not
    standing at the terminal, and
    :mod:`softae.tools.eis_validate_narrate` turns them into the run's published
    stream and its ``conditions.json``. Injected rather than imported so this
    module keeps knowing nothing about sidecars, and **swallowed** rather than
    propagated: an observer is a monitoring convenience, and a monitoring
    convenience must never be the reason a soak refuses.
    """
    sleep = sleep or time.sleep
    now = now or time.monotonic

    def observe(hook: Any, *args: Any) -> None:
        _observe(hook, "eis_validate_soak_observer_failed", *args)

    target = float(plan.soak_s)
    entered = float(now())
    credit = entered - float(established_at)
    if target <= 0:
        return SoakOutcome(soaked_s=credit, waited_s=0.0, settle_credit_s=credit)

    ceiling = entered + target * SOAK_CEILING_FACTOR
    clock_start = float(established_at)
    restarts = 0
    polls = 0
    print(f"[soak  ] holding at condition for {target / 60:.0f} min; "
          f"{max(0.0, credit) / 60:.1f} min of it already spent at condition "
          f"during approach-to-settle.", flush=True)

    while True:
        elapsed = float(now()) - clock_start
        if elapsed >= target:
            break
        if float(now()) >= ceiling:
            raise RefuseToStart(
                f"the soak could not accumulate {target / 60:.0f} min of "
                f"unbroken time at condition within "
                f"{target * SOAK_CEILING_FACTOR / 60:.0f} min of waiting "
                f"({restarts} excursion restart(s), best run "
                f"{elapsed / 60:.1f} min). A condition that cannot hold itself "
                "through the soak will not hold through the measurement block "
                "-- refusing to start."
            )
        sleep(max(0.0, min(float(poll_interval_s), target - elapsed)))
        polls += 1

        if watch is not None:
            watch.poll()                      # a fault raises; the runner parks
            if watch.excursion:
                restarts += 1
                clock_start = float(now())
                print(f"[soak  ] EXCURSION at {elapsed / 60:.1f} min -- the soak "
                      f"clock restarts (restart {restarts}); continuity is the "
                      "quantity being asserted.", flush=True)
                observe(on_restart, restarts, elapsed, target)
                continue
        observe(on_poll, polls, float(now()) - clock_start, target, restarts)
        if polls % SOAK_PRINT_EVERY_N_POLLS == 0:
            print(f"[soak  ] {(float(now()) - clock_start) / 60:6.1f} / "
                  f"{target / 60:.0f} min at condition", flush=True)

    soaked = float(now()) - clock_start
    print(f"[soak  ] complete: {soaked / 60:.1f} min of unbroken time at "
          f"condition, {restarts} restart(s).", flush=True)
    return SoakOutcome(
        soaked_s=soaked, waited_s=float(now()) - entered,
        settle_credit_s=credit, restarts=restarts,
    )


# ── The hold ─────────────────────────────────────────────────────────────────

@dataclass
class HoldWatch:
    """Graded excursion watch on both axes, sampled **between** sweeps.

    A sweep is never interrupted mid-script, which is the requirement; polling
    from a background thread would satisfy it too, but would also put a second
    caller on the temperature and RH drivers while a third drives the
    potentiostat, and nothing in this tree demonstrates those drivers are
    re-entrant. Between-sweep sampling costs at most one sweep of latency
    (17-120 s) against grace windows of 120 s (temperature) and 600 s (RH), and
    both verdicts are *sustained*-excursion verdicts, so the grade an operator
    acts on is unchanged.

    ``warn`` -> record, continue, and stamp every row taken inside the window.
    ``fault`` on either axis -> raise, which the runner turns into park +
    disconnect + non-zero exit. A PV that cannot be read for the whole grace
    window is a fault, matching ``monitored_hold``'s own posture.
    """

    manager: Any
    plan: ValidationPlan
    now: Any = time.monotonic
    temp_series: list[tuple[float, float]] = field(default_factory=list)
    rh_series: list[tuple[float, float]] = field(default_factory=list)
    warn_c: float = 2.0
    fault_c: float = 5.0
    grace_s: float = 120.0
    rh_thresholds: dict[str, float] = field(default_factory=dict)
    excursion: bool = False

    def __post_init__(self) -> None:
        if not self.rh_thresholds:
            from softae.drivers.contracts import rh_watchdog_config

            self.rh_thresholds = rh_watchdog_config()

    def poll(self) -> None:
        """Sample both axes and grade. Raises ``SafetyError`` on a fault."""
        from softae.drivers.contracts import (
            RH_CONVERGING,
            RH_FAULT,
            classify_rh_hold,
            sustained_above,
            sustained_below,
        )
        from softae.errors import SafetyError

        t = float(self.now())
        temp_pv = _read_pv(self.manager, TEMP_CONTROLLER, "get_pv")
        rh_pv = _read_pv(self.manager, RH_CONTROLLER, "get_H")
        self.temp_series.append((t, temp_pv))
        self.rh_series.append((t, rh_pv))

        target = float(self.plan.temp_setpoint_c)
        if sustained_above(self.temp_series, target, self.fault_c, self.grace_s) or \
                sustained_below(self.temp_series, target, self.fault_c, self.grace_s):
            raise SafetyError(
                f"temperature sustained more than {self.fault_c:g} C from "
                f"{target:g} C for {self.grace_s:g} s",
                instrument=TEMP_CONTROLLER, requested=target, limit=self.fault_c,
            )

        # `.state`, not the verdict object: `classify_rh_hold` returns an
        # `RHHoldVerdict`, so comparing it to the string constant is always
        # unequal and would grade every single poll as an excursion -- which
        # would then exclude every row from the accuracy tables and quietly
        # empty the report.
        verdict = classify_rh_hold(
            self.rh_series, float(self.plan.rh_setpoint_pct),
            warn_pct=self.rh_thresholds["warn_pct"],
            fault_pct=self.rh_thresholds["fault_pct"],
            grace_s=self.rh_thresholds["grace_s"],
            temperature_C=temp_pv,
        ).state
        if verdict == RH_FAULT:
            raise SafetyError(
                f"RH sustained more than {self.rh_thresholds['fault_pct']:g} %RH "
                f"from {self.plan.rh_setpoint_pct:g} % for "
                f"{self.rh_thresholds['grace_s']:g} s",
                instrument=RH_CONTROLLER,
                requested=float(self.plan.rh_setpoint_pct),
                limit=self.rh_thresholds["fault_pct"],
            )

        warn_temp = (
            math.isfinite(temp_pv) and abs(temp_pv - target) > self.warn_c
        )
        warn_rh = verdict != RH_CONVERGING
        self.excursion = bool(warn_temp or warn_rh)
        if self.excursion:
            logger.warning("eis_validate_hold_excursion", temp_pv=temp_pv,
                           rh_pv=rh_pv, rh_verdict=verdict)


def _read_pv(manager: Any, instrument: str, method: str) -> float:
    try:
        return float(getattr(manager.get(instrument), method)())
    except Exception:
        return float("nan")


def _read_rh(rh: Any) -> float | None:
    """``None`` is the *unreadable* sentinel ``SettleTracker`` expects.

    A settle window with a missing RH reading is ``not_evaluable``, never
    silently 'settled', and that only works if the absence is passed through
    as an absence rather than as a NaN that compares false against everything.
    """
    try:
        value = float(rh.get_H())
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _round_fit(channel: int, eis: Any) -> Any:
    """One channel's contribution to a settle round, **with a sigma**.

    ``settle_check`` excludes any channel carrying a NULL sigma from a round --
    correctly, because "constant because nothing measured it" passes a stability
    test perfectly -- so a caller that supplies only ``r1_ohms`` makes every
    window ``not_evaluable`` and runs to its ceiling for nothing.

    The sigma supplied here is ``1/R``, i.e. sigma with the cell constant set to
    1. That is exact rather than approximate for this gate's purpose: the check
    is a **per-channel relative deviation from that channel's own mean**, so the
    channel's real ``K`` cancels out of it entirely -- the same reason the
    harness's whole comparison is geometry-free. No geometry is resolved, and
    none is needed.

    ``R`` is the low-frequency real part, not a circuit fit. The gate compares a
    quantity against itself across rounds, so what it needs is something monotone
    in the arc size and available on every round -- and a fit on every channel of
    every round is where a settle phase's cost would otherwise go, at 2.6 s per
    open-arc fit.
    """
    from softae.analysis.equilibration import RoundFit

    try:
        r = float(eis.z_real[-1])
    except Exception:
        return RoundFit(channel=int(channel), sigma=None, r1_ohms=None)
    sigma = (1.0 / r) if (math.isfinite(r) and r > 0) else None
    return RoundFit(channel=int(channel), sigma=sigma, r1_ohms=r)


__all__ = [
    "DEFAULT_DRIFT_CHECK", "DEFAULT_MIN_TREATMENT",
    "DEFAULT_RH_APPROACH_TIMEOUT_S", "DEFAULT_SETTLE_MAX_HOLD_S",
    "DEFAULT_SOAK_S", "DEFAULT_TEMP_APPROACH_TIMEOUT_S",
    "DEFAULT_TEMP_DESCENT_TIMEOUT_S", "SOAK_CEILING_FACTOR",
    "SOAK_POLL_INTERVAL_S", "SOAK_PRINT_EVERY_N_POLLS",
    "ApproachReport", "HoldWatch", "Projection", "RefuseToStart",
    "SettleOutcome", "SoakOutcome", "ValidationPlan", "VirtualClock",
    "approach_condition", "assert_settle_licensed", "band_by_channel",
    "classify_apex", "population_thresholds", "project", "render_arc_watch",
    "render_projection", "settle_deviations", "settle_phase", "soak_phase",
    "validate_plan",
]
