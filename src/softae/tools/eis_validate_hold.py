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

**Actuating and judging are two different orderings, and only the second one
carries the evidence.** The RH loop is commanded at the *start* of
:func:`approach_condition`, beside the temperature setpoint, so the humidifier
works through the heat rather than waiting it out; RH *arrival* is still judged
only after temperature is satisfied, because the attainable RH floor rises with
temperature and a reading accepted against a floor about to move is not
evidence. That docstring carries the whole argument -- including why drying
through the heat is *permitted* rather than impossible, and what takes the loop
down when the temperature approach refuses.

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
from collections.abc import Sequence
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

#: The settle band this harness starts from, and the value ``--settle-tol-rel``
#: falls back to. A **relative deviation of sigma from its own window mean** --
#: dimensionless, and in particular not %RH.
#:
#: **Restated rather than imported, and pinned by a test.** This module holds no
#: module-level ``softae`` imports at all: ``softae-eis-validate --help`` costs
#: 0.3 s and loads neither numpy nor scipy, and importing
#: ``analysis.equilibration`` for one float would put both on every invocation
#: including ``--help``. So the number is written here and
#: ``test_settle_tol_rel_default_is_the_shipped_criterions_own`` fails the moment
#: it diverges from :data:`softae.analysis.equilibration.DEFAULT_SETTLE_TOL_REL`,
#: which remains the criterion's home.
DEFAULT_SETTLE_TOL_REL = 0.10

#: At or above this band, the run says out loud what the operator just bought.
#: Twice the shipped default: on the board that motivated the flag the median
#: scatter was 12.5-14 %, so the honest fix there (~0.16-0.18) sits below this
#: line, and everything above it is a deliberate widening that deserves to be
#: audible rather than silent.
SETTLE_TOL_REL_LOOSE = 0.20

#: Above this band the run refuses to start. At 0.5 a channel's sigma may span
#: ``[0.5, 1.5] x`` its own window mean -- a threefold spread between the
#: extremes -- and still certify as *settled*; there is no film state that
#: reading describes, so this is not a loosened gate but the absence of one. The
#: ceiling exists because the two directions are not symmetric: a band set too
#: LOW is self-correcting -- :func:`~softae.analysis.equilibration.endorse_tolerance`
#: announces it as unachievable and the run refuses at its ceiling -- while a
#: band set too HIGH silently certifies a film that moved, and the run that
#: follows looks exactly like a correct one. A comment in :func:`settle_phase`
#: records an operator who read ``spread 0.130`` as %RH and nearly typed ``1.0``
#: here, which is 100 % relative deviation: a film whose conductivity doubled
#: between rounds, accepted. Exposing the number on the CLI reopens that trap, so
#: the trap is closed at the far end. Deliberately **not** overridable: a
#: ``--yes-i-really-mean-it`` escape would restore exactly the value it exists to
#: refuse.
SETTLE_TOL_REL_MAX = 0.50

#: Which criterion the settle gate routes on. Restated here rather than imported
#: for the reason :data:`DEFAULT_SETTLE_TOL_REL` is, and pinned to
#: :data:`softae.analysis.equilibration.SETTLE_CRITERION_DEVIATION` by a test.
DEFAULT_SETTLE_CRITERION = "deviation"
#: **Unset**, not "no drift permitted". A rate criterion with no tolerance has
#: nothing to compare against, so 0 is refused at :func:`validate_plan` rather
#: than taken literally -- a literal zero would make every cell moving and the
#: run would blame the film for the flag.
DEFAULT_SETTLE_RATE_TOL_DEC_PER_H = 0.0

#: Above this rate band the run refuses to start, on exactly
#: :data:`SETTLE_TOL_REL_MAX`'s argument and for exactly its reason. At 0.5 dec/h
#: a cell whose conductivity changes by a factor of **3.16 every hour** still
#: certifies as having stopped moving; there is no film state that reading
#: describes. The two directions are again not symmetric: a band set too TIGHT is
#: self-correcting -- :data:`~softae.analysis.equilibration.RATE_SPAN_TOO_SHORT`
#: and :data:`~softae.analysis.equilibration.RATE_UNDETECTABLE` are both
#: non-evaluable, so the phase runs to its ceiling and refuses -- while a band set
#: too LOOSE certifies, and the run it produces is indistinguishable from a
#: correct one.
#:
#: The number is calibrated against what the shipped deviation band already
#: amounts to: ``--settle-tol-rel 0.10`` over the measured 562.5 s round is
#: 0.610 ln/h = **0.265 dec/h**, so nothing presently defensible is refused here.
#: For scale in the other direction, H3 asks the whole hold to stay inside 0.05
#: dec. Deliberately **not** overridable, for the reason
#: :data:`SETTLE_TOL_REL_MAX` is not.
SETTLE_RATE_TOL_DEC_PER_H_MAX = 0.5

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

#: The circuit model both the settle gate and the measurement block fit with.
#: **One name, because two would be a silent disagreement**: the gate excludes a
#: channel whose R₁ rests on *the model's* lower bound
#: (:func:`~softae.analysis.equilibration.r1_lower_bound_ohms`), so a gate fitting
#: one model while the run reports another would rail against a bound that does
#: not describe the number anybody reads. Carried on the plan so a future
#: ``--circuit-model`` moves both together; the campaign path names its own copy
#: at :data:`softae.core.autonomous_wiring.SETTLE_CIRCUIT_MODEL` for the same
#: reason.
SETTLE_CIRCUIT_MODEL = "simpleSalt"


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
    #: Which equivalent circuit the settle gate fits to get its R₁, and which the
    #: measurement block reports through. Trailing, with a default, so every
    #: existing positional construction is unchanged. **Deliberately not in
    #: :meth:`fingerprint`** -- it changes how a spectrum is *read*, not which
    #: specimen was measured, and the hash's stated line is ceiling-versus-floor
    #: on the sample's state. Recorded in :meth:`as_dict` instead, which is where
    #: an analysis choice belongs.
    circuit_model: str = SETTLE_CIRCUIT_MODEL
    #: The settle gate's band, from ``--settle-tol-rel``. Trailing, with a
    #: default, so every existing positional construction is unchanged. Until
    #: this field existed :func:`settle_phase` built its ``SettleTracker``
    #: without a ``tol_rel`` and so silently took the shipped 0.10 -- which
    #: 20260821T173111Z_eis_validate could not satisfy at all, its own median
    #: scatter being 12.5-14 %, with no way for the operator to say so.
    #:
    #: **In** :meth:`fingerprint`, unlike ``circuit_model`` and unlike every
    #: other duration on this plan -- see that method for why.
    settle_tol_rel: float = DEFAULT_SETTLE_TOL_REL
    #: Which of the two sibling criteria the settle gate ROUTES on --
    #: ``deviation`` (shipped), ``rate``, or ``both`` (deviation routes, the rate
    #: is reported). **In** :meth:`fingerprint` when it is not the default, on
    #: ``settle_tol_rel``'s argument: this is the criterion, not a ceiling on
    #: waiting for one.
    settle_criterion: str = DEFAULT_SETTLE_CRITERION
    #: The rate band, in **decades per hour** -- the operator's unit, converted
    #: to the gate's ln-units at the one boundary that needs it
    #: (:func:`~softae.analysis.equilibration.rate_tol_ln_per_hour`). Decades
    #: because H3 is in decades and the spec derives this number from it as
    #: ``H3_MAX_HOLD_DRIFT_DEC / T_meas``; an operator who computes it computes
    #: decades. 0 means unset and is refused for the criteria that need it.
    settle_rate_tol_dec_per_h: float = DEFAULT_SETTLE_RATE_TOL_DEC_PER_H
    #: At the ceiling, partition rather than fail: proceed on the cells the
    #: criterion certified quiet and record the rest with their reasons.
    #: **Off by default**, so every existing verdict is byte-identical, and
    #: available only here -- the campaign path does not get it, because
    #: conditioning a BO objective on settling biases it toward materials that
    #: equilibrate fast, which is a material property correlated with the thing
    #: being optimised.
    survivors: bool = False

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

        **``settle_tol_rel`` is in, and ``circuit_model`` stays out.** They look
        alike -- both trailing, both added late, both about how the run treats a
        spectrum -- and they are not. ``circuit_model`` changes how a spectrum is
        *read*; the settle band decides **which sample states are admitted into
        the dataset at all**, which puts it on ``soak_s``'s side of the line, not
        on the ceilings' side. ``settle_max_hold_s`` is a ceiling on *waiting for
        a criterion*; this is the criterion. And it is a **pre-registered gate
        parameter** in the sense :mod:`softae.tools.eis_validate_rule` opens
        with -- *"a validation whose success criterion is chosen after the data
        arrives proves nothing"* -- so a ``--resume`` that continues under a
        widened band is precisely the thing pre-registration forbids, and it is
        the resume that would be *tempting*: the operator who hits a ceiling has
        a widened tolerance in hand and a half-finished run on disk.

        **It enters the hash only when it is not the default, and that is not a
        hack.** The guarantee wanted is *"a resume cannot change the criterion"*,
        and conditional inclusion delivers it in full: default -> the ten-part
        string exactly as before; anything else -> an eleventh, labelled part. A
        run started at 0.15 and resumed without the flag mismatches, a run
        started at the default and resumed at 0.15 mismatches, and two different
        non-default bands mismatch. What it buys is that every checkpoint written
        before this field existed keeps its fingerprint, so introducing an
        operator knob does not invalidate an in-flight run's ``--resume`` --
        including the run whose unsatisfiable band is why the knob exists.

        **The criterion selector and the survivor flag join it, on the same
        argument and by the same conditional mechanism.** The spec that proposed
        them argued they were *out* -- ceilings on waiting for a criterion -- and
        that argument does not survive contact with the paragraph above: they are
        not ceilings on waiting for the criterion, they ARE the criterion, and
        ``survivors`` decides which cells are admitted to the population at all,
        which is ``settle_tol_rel``'s side of the ceiling-versus-floor line
        rather than ``settle_max_hold_s``'s. A ``--resume`` that switched
        ``--settle-criterion`` mid-run would be exactly the after-the-fact
        criterion choice :mod:`softae.tools.eis_validate_rule` opens by
        forbidding. Each enters only when it is not the default, so every
        checkpoint written before these fields existed keeps its fingerprint.
        """
        import hashlib

        parts = [str(p) for p in (
            self.validation_name, self.channels, self.rh_setpoint_pct,
            self.temp_setpoint_c, self.baseline_preset, self.reference_preset,
            self.order, self.max_follow_ups, self.visit, self.soak_s,
        )]
        if float(self.settle_tol_rel) != float(DEFAULT_SETTLE_TOL_REL):
            # Labelled, not bare: a self-describing token cannot be confused
            # with a future appended field, and reads in a debugger.
            parts.append(f"settle_tol_rel={float(self.settle_tol_rel)!r}")
        if str(self.settle_criterion) != DEFAULT_SETTLE_CRITERION:
            parts.append(f"settle_criterion={str(self.settle_criterion)!r}")
            # Only under a criterion that reads it: a rate band typed beside
            # `deviation` changes nothing about what was measured, and hashing
            # it would refuse a resume over a number the run never used.
            parts.append("settle_rate_tol_dec_per_h="
                         f"{float(self.settle_rate_tol_dec_per_h)!r}")
        if bool(self.survivors):
            parts.append("survivors=True")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(self).items()
        }
        payload["fingerprint"] = self.fingerprint()
        return payload


def validate_plan(plan: ValidationPlan) -> None:
    """Refuse impossible plans **before** anything is heated.

    Three checks today, and none is hypothetical: ``settle_check`` requires at
    least ``DEFAULT_SETTLE_MIN_CHANNELS`` participating channels, so a run on
    fewer than that can never return ``settled`` -- it runs every round to the
    ceiling and then refuses. Caught late that costs the full
    ``--settle-max-hold-s`` (90 minutes by default) at temperature, and the
    operator's first evidence is a refusal that names the wrong cause.

    The third is ``--settle-tol-rel``, and it is refused in **one** direction
    only. Too tight is left alone on purpose: it is self-correcting, because
    :func:`~softae.analysis.equilibration.endorse_tolerance` announces it as
    unachievable at the first judged window and the phase then refuses at its
    ceiling rather than certifying anything. Too loose has no such backstop --
    it *certifies*, and the run it produces is indistinguishable from a correct
    one -- so :data:`SETTLE_TOL_REL_MAX` is where it stops, before anything is
    heated.
    """
    from softae.analysis.equilibration import (
        DEFAULT_SETTLE_MIN_CHANNELS,
        settle_tol_rel_refusal,
    )

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
    if not plan.settle:
        # Gated on the same condition the channel-count check is: with no gate
        # there is nothing to certify, so the band is never read and a refusal
        # here would be about an unused number.
        return
    # One rule, shared with `run_plan.SettlePlan` rather than restated here, so
    # a band the campaign path would refuse cannot be accepted by this tool.
    if (refusal := settle_tol_rel_refusal(plan.settle_tol_rel)) is not None:
        raise RefuseToStart(f"--settle-tol-rel {plan.settle_tol_rel:g}: {refusal}")
    if not math.isfinite(plan.settle_tol_rel):
        # `nan > MAX` is False, so the ceiling below would pass it through, and a
        # NaN band makes every comparison in `settle_check` False -- a gate that
        # can never certify, arriving as "the film never settled".
        raise RefuseToStart(
            f"--settle-tol-rel {plan.settle_tol_rel} is not a real number. The "
            "settle band is a relative deviation of sigma from its own window "
            "mean; a non-finite one makes every round's comparison false, so "
            "the run would hold to its ceiling and blame the film."
        )
    if plan.settle_tol_rel > SETTLE_TOL_REL_MAX:
        raise RefuseToStart(
            f"--settle-tol-rel {plan.settle_tol_rel:g} is above the maximum "
            f"{SETTLE_TOL_REL_MAX:g}. This number is a RELATIVE DEVIATION OF "
            f"SIGMA from its own window mean, not %RH: "
            f"{plan.settle_tol_rel:g} certifies a cell whose conductivity "
            f"varies by {plan.settle_tol_rel * 100:.0f}% across the judged "
            f"window as having stopped moving, and 1.0 would accept a film "
            f"whose conductivity doubled between rounds. If the gate cannot be "
            f"cleared at a defensible band, the cells are the problem and a "
            f"wider band only hides it."
        )
    _validate_criterion(plan)


def _validate_criterion(plan: ValidationPlan) -> None:
    """The criterion selector and the survivor flag, refused before any heat.

    Four refusals, and every one of them is a run that would otherwise spend its
    whole ceiling at temperature before failing for a reason the operator could
    have been told at the prompt.
    """
    from softae.analysis.equilibration import (
        DEFAULT_SETTLE_MIN_CHANNELS,
        SETTLE_CRITERIA,
        SETTLE_CRITERION_DEVIATION,
    )

    criterion = str(plan.settle_criterion)
    if criterion not in SETTLE_CRITERIA:
        raise RefuseToStart(
            f"--settle-criterion {criterion!r} is not one of "
            f"{', '.join(SETTLE_CRITERIA)}.")
    rate_tol = float(plan.settle_rate_tol_dec_per_h)
    if criterion != SETTLE_CRITERION_DEVIATION:
        if not math.isfinite(rate_tol) or rate_tol <= 0:
            raise RefuseToStart(
                f"--settle-criterion {criterion} needs "
                f"--settle-rate-tol-dec-per-h, and {rate_tol:g} is not a usable "
                "band. It is a DRIFT RATE in decades per hour, not a relative "
                "deviation and not %RH: 0.025 means a cell whose conductivity "
                "moves by 0.025 decades in an hour is still called still. The "
                "spec derives it from H3 as 0.05 dec / (measurement block "
                "hours), so a 2 h block wants 0.025. A zero or negative band "
                "would make every cell moving and the run would blame the film."
            )
        if rate_tol > SETTLE_RATE_TOL_DEC_PER_H_MAX:
            raise RefuseToStart(
                f"--settle-rate-tol-dec-per-h {rate_tol:g} is above the maximum "
                f"{SETTLE_RATE_TOL_DEC_PER_H_MAX:g}. This number is DECADES PER "
                f"HOUR: {rate_tol:g} certifies a cell whose conductivity changes "
                f"by a factor of {10.0 ** rate_tol:.3g} every hour as having "
                f"stopped moving. For scale, the shipped --settle-tol-rel 0.10 "
                f"over a 562.5 s round is 0.265 dec/h, and H3 asks the whole "
                f"hold to stay inside 0.05 dec. If the gate cannot be cleared at "
                f"a defensible band, the cells are the problem."
            )
    if not plan.survivors:
        return
    if criterion == SETTLE_CRITERION_DEVIATION:
        # The partition rests on telling "this cell is MOVING" from "this cell
        # cannot be judged", and the deviation criterion is precisely the one
        # that cannot: for a 3-round window its statistic IS the window noise
        # floor to within 13 %. Partitioning on it would drop the moving cells
        # along with the noisy ones -- the exact failure the criterion exists to
        # prevent, wearing the feature's name.
        raise RefuseToStart(
            "--survivors on needs --settle-criterion rate or both. The "
            "partition drops cells the gate could not JUDGE and keeps refusing "
            "on cells it proved were MOVING, and the deviation criterion cannot "
            "tell those apart -- its statistic is a scatter estimate compared "
            "against a drift tolerance. Dropping on it would drop the moving "
            "cells too."
        )
    # A survivor set must clear `min_channels` AND carry `min_treatment`
    # TREATMENT cells AND at least one CONTROL. TREATMENT survivors are
    # survivors, so the binding floor is the larger of the two and not their
    # sum, plus the one CONTROL that keeps D3 computable.
    floor = max(DEFAULT_SETTLE_MIN_CHANNELS, int(plan.min_treatment) + 1)
    if len(plan.channels) < floor:
        raise RefuseToStart(
            f"--survivors on with {len(plan.channels)} channel(s): a survivor "
            f"set must carry at least {int(plan.min_treatment)} TREATMENT "
            f"cell(s), at least one CONTROL, and at least "
            f"{DEFAULT_SETTLE_MIN_CHANNELS} cells in total -- {floor} channels "
            "before a single one is dropped. Partitioning a board with no "
            "headroom refuses at the same ceiling it would have refused at, "
            "having spent the whole hold to get there."
        )


def loose_band_notice(tol_rel: float) -> str:
    """The line printed when an admissible band is still a generous one.

    Between :data:`SETTLE_TOL_REL_LOOSE` and :data:`SETTLE_TOL_REL_MAX` the run
    proceeds -- a board scattering at 12-14 % has no honest alternative -- but it
    says what was bought, in the unit the number is actually in. ``""`` below the
    line, so the common case prints nothing.
    """
    if not (math.isfinite(tol_rel) and tol_rel >= SETTLE_TOL_REL_LOOSE):
        return ""
    return (f"[settle] WIDE BAND: --settle-tol-rel {tol_rel:g} certifies a cell "
            f"whose sigma varies by {tol_rel * 100:.0f}% about its own window "
            f"mean as settled. That is a RELATIVE DEVIATION, not %RH. Default "
            f"is {DEFAULT_SETTLE_TOL_REL:g}; the run refuses above "
            f"{SETTLE_TOL_REL_MAX:g}.")


def suggested_settle_tol_rel(floor_rel: float) -> float | None:
    """The narrowest admissible band this board's own scatter could clear.

    ``floor_rel * 1.2``, rounded up to a whole percent: the floor itself is the
    boundary case and a band exactly on it certifies nothing, so the suggestion
    carries margin. ``None`` when the answer would be above
    :data:`SETTLE_TOL_REL_MAX` -- a value the run would then refuse is not a
    suggestion, it is a second trap, and at that scatter the cells are the
    finding rather than the flag.
    """
    if not (math.isfinite(floor_rel) and floor_rel > 0):
        return None
    suggestion = math.ceil(float(floor_rel) * 1.2 * 100.0) / 100.0
    return None if suggestion > SETTLE_TOL_REL_MAX else suggestion


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
    # The RH loop is commanded WITH the temperature setpoint, so this row is
    # what is left to wait for AFTER the heat -- not a fresh descent. It is
    # labelled rather than re-costed: how much of the descent the heat absorbs
    # is the chamber's business, and a smaller number printed here would be a
    # saving this table cannot promise. The operator types "yes" against it.
    add(f"  {'approach   RH  (commanded with temperature)':<44}{'0 - 30 min':>14}"
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
    add("  BOTH SETPOINTS ARE COMMANDED AT THE START. The humidifier loop runs")
    add("  through the heat instead of waiting it out, because drying is the slow")
    add("  axis. RH ARRIVAL is still judged only AFTER temperature is satisfied:")
    add("  the attainable RH floor RISES with temperature (15 % commanded gave")
    add("  16.9-23.2 % PV at 65-85 C), so a reading accepted before the block is")
    add("  hot is accepted against a floor about to move. The RH row above is")
    add("  therefore the REMAINDER after the heat, and its bound is measured from")
    add("  the moment judging starts, not from the setpoint write. How much of")
    add("  the descent the heat absorbs is the chamber's to decide; this table")
    add("  projects no saving for it.")
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
    #: Seconds spent **judging arrival** -- and so the quantity ``timeout_s``
    #: bounds. Not the time the axis has been under command; see :attr:`lead_s`.
    elapsed_s: float
    pv_final: float
    attempts: int
    #: Seconds this axis was already being driven **before judging began**. Zero
    #: for temperature, which is judged straight off its own setpoint write; the
    #: whole temperature approach for RH, whose loop is commanded at the start
    #: and judged after temperature. Appended with a default rather than
    #: inserted, so every positional construction of this record still reads the
    #: same and the printing loop's shape is unchanged.
    lead_s: float = 0.0

    @property
    def driven_s(self) -> float:
        """Total seconds under command: the lead plus the judged window."""
        return self.lead_s + self.elapsed_s


def approach_condition(
    manager: Any,
    plan: ValidationPlan,
    *,
    sleep: Any = None,
    now: Any = None,
    poll_interval_s: float = 30.0,
    on_command: Callable[[str, float], None] | None = None,
) -> list[ApproachReport]:
    """Both setpoints commanded at once; temperature judged first, then RH.

    **The order that carries the evidence is the order of the two JUDGEMENTS,
    and it is unchanged.** The attainable RH floor *rises with temperature* --
    15 %RH commanded gave 16.9-23.2 % PV at 65-85 C -- so *declaring* RH reached
    before the block is at temperature is declaring it against a floor that is
    about to move. No RH reading is compared against tolerance until the
    temperature approach has returned ``reached``.

    **What was welded to that ordering, and is now separated from it, is when the
    loop starts ACTUATING.** Drying is the slow axis -- a measured ~5000 s
    descent against a ~13 min heat -- and it had no reason to sit idle through
    the heat. The RH setpoint is written and the loop started immediately after
    the temperature setpoint write, so the humidifier is under closed-loop
    control at *this run's own target* for the whole approach, and the RH
    approach that follows the heat is whatever is left of the descent.

    **Drying through the heat ramp is PERMITTED -- operator ruling, 2026-08-21 --
    so no clamp is needed and none is added.** It is permitted, not impossible,
    and the difference matters because this axis has two actuators, not one.
    ``scripts/trinket_firmware/dac0_rh/code.py`` drives an Aalborg PSV pair from
    a single ``ctrl``: humid air over ``V0_range`` scaled by ``ctrl``, and **dry
    air over ``V1_range`` -- "dry air signal range" -- scaled by ``1 - ctrl``**.
    So ``ctrl`` = 1 is fully humid, ``ctrl`` near 0 is dry air at nearly full
    flow, and ``ctrl`` == 0 *exactly* is the firmware's auto-shutoff, which
    closes both valves. The PID's ``output_limits`` of
    ``(out_min, out_max) = (0.01, 1.0)`` therefore bound the loop at the
    **driest flowing state**, not at a humidification trickle. An early-started
    loop can and does dry the chamber during the heat.

    **The alternative to it is not "no drying" -- it is uncontrolled drift.**
    RH moves while the block heats whether or not anything is commanding the
    axis. The sequential form sent a temperature setpoint, waited on it, and did
    not care where RH went in the meantime; the excursion it tolerated could be
    more extreme than anything a loop aimed at the target would produce. The
    early loop is closed-loop at *this run's own target*, so the heat is spent
    moving toward the setpoint rather than away from it. **Active control is
    bounded by the setpoint; absence of control is bounded by nothing.** And the
    state it displaces is not even "no command": ``AsyncRHController.safe_off``
    records that a process which sets a setpoint and never calls ``start``
    "writes **nothing** -- leaving the Trinket at whatever duty a previous
    session left it at". The sequential form did not merely decline to drive the
    axis through the heat -- it left it *unsupervised* at a stale duty, pointed
    wherever the last run pointed it.

    **The variance in that stale duty is the honest limit on what the overlap
    buys.** If the previous session left the Trinket near ``out_min``, dry air
    was already flowing and the descent was already underway -- open-loop and
    aimed at no particular target, but in the right direction -- so the overlap
    saves close to nothing. If it left duty 0, both valves were shut and the
    chamber was drifting back toward room air throughout. If it left a real
    humidifying duty, the sequential form spent the whole heat driving *away*
    from the target and the descent only began afterwards -- and there the
    overlap is worth the entire approach. Which of the three it is is a fact
    about the chamber's history, not about this harness, which is why
    :func:`render_projection` declares the overlap and projects no saving for it.

    **A temperature refusal takes the early loop down with it** -- see
    :func:`_release_rh`, which is the arm that closes what this one opened.

    *on_command* ``(axis, target)`` fires as each setpoint is written, distinct
    from the arrival that the returned report describes, so a watcher can see
    that the RH loop is live rather than inferring it from a silence. Injected
    and **swallowed**, for the reasons :func:`_observe` exists.
    """
    from softae.workflows.equilibration import approach_setpoint

    temp = manager.get(TEMP_CONTROLLER)
    rh = manager.get(RH_CONTROLLER)
    clock = now or time.monotonic
    reports: list[ApproachReport] = []

    # Temperature setpoint first, and RH's "immediately after" rather than
    # "before": `set_setpoint` raises `SafetyError` on an over-max target, and
    # that refusal now lands ~13 min earlier than it used to. Keeping the heater
    # write ahead of it means the state a rejected RH setpoint leaves is
    # byte-for-byte the state it left before -- heater commanded, RH loop never
    # started, park in `cmd_run`'s `finally` -- only sooner.
    temp.write_sp(float(plan.temp_setpoint_c))
    _observe(on_command, _COMMAND_OBSERVER_FAILED,
             "temperature", float(plan.temp_setpoint_c))

    rh.set_setpoint(float(plan.rh_setpoint_pct))
    rh.start()
    rh_commanded_at = float(clock())
    _observe(on_command, _COMMAND_OBSERVER_FAILED,
             "rh", float(plan.rh_setpoint_pct))
    logger.info("eis_validate_rh_commanded_early",
                rh_target=float(plan.rh_setpoint_pct),
                temp_target=float(plan.temp_setpoint_c))
    print(f"[approach] rh loop commanded at {plan.rh_setpoint_pct:g} % now, and "
          f"judged after temperature: the floor rises with T.", flush=True)

    try:
        reports.append(_approach_one(
            approach_setpoint, lambda: float(temp.get_pv(1)),
            plan.temp_setpoint_c, axis="temperature", instrument=TEMP_CONTROLLER,
            tolerance=plan.tolerance_c, timeout_s=plan.temp_approach_timeout_s,
            poll_interval_s=poll_interval_s, sleep=sleep, now=now,
        ))
    except RefuseToStart:
        _release_rh(rh)
        raise

    reports.append(_approach_one(
        approach_setpoint, lambda: float(rh.get_H()),
        plan.rh_setpoint_pct, axis="rh", instrument=RH_CONTROLLER,
        tolerance=plan.rh_tolerance_pct, timeout_s=plan.rh_approach_timeout_s,
        poll_interval_s=poll_interval_s, sleep=sleep, now=now,
        lead_s=max(0.0, float(clock()) - rh_commanded_at),
    ))
    return reports


#: One string, because both command observations are the same failure.
_COMMAND_OBSERVER_FAILED = "eis_validate_approach_command_observer_failed"


def _release_rh(rh: Any) -> None:
    """Take the early-started RH loop down when the temperature approach refuses.

    The loop is started early *so that the RH approach which follows is short*.
    A temperature refusal means no RH approach follows: the humidifier would be
    left driving a setpoint nothing will ever judge, and under ``--end-state
    hold`` -- the one exit with no park -- nothing would take it down either. So
    the arm that opened it closes it, and the change is a no-worse-than-today
    guarantee on every path rather than only on the parking ones.

    ``safe_dry``, not ``safe_off``, and neither of them ``stop``. All three stop
    the loop; only the latter two write anything, because ``stop`` returns
    cleanly having sent nothing when the PID thread is wedged in an I2C read.
    Between the two that write, the choice is which end state the chamber is left
    in, and this call site wants the dry one.

    **Why the dry purge belongs here specifically.** This arm fires when the
    temperature approach refused, i.e. after the RH loop has been drying the
    chamber for as long as the heat took -- often the whole descent. ``safe_off``
    writes duty 0, which the Trinket treats as a special case and shuts *both*
    Aalborg PSVs, so there is no flow at all and room air wins: the chamber the
    run just spent an hour drying goes back to ~50 %RH in tens of seconds, and
    the operator's retry pays for the whole descent again. ``safe_dry`` leaves
    ``out_min`` -- dry air -- on the wire and lets the firmware's own ~25 s
    deadman close the valves, so nothing stays energised and the dry state
    survives the refusal.

    This matters most on the path that made the release necessary at all:
    ``--end-state hold`` has no park behind it, so this call is the *only* thing
    that decides what the chamber does next.

    Best-effort through :func:`_observe`, whose guarantee ("call it, never
    raise") is exactly the one wanted here even though the callee is a driver
    rather than an observer: the refusal is the news, and a humidifier that
    cannot be released must not become a *different* exception on the way out of
    one. ``cmd_run``'s park tries again and ``SafeParkResult`` is where that
    failure is meant to be reported -- except under ``--end-state hold``, which
    is why the outcome is also printed here rather than only logged.

    **Only the temperature arm is wrapped.** An RH refusal leaves the loop
    running, which is byte-for-byte what the sequential form did; changing it
    here would be an unrelated behaviour change smuggled in beside this one.
    """
    dry = getattr(rh, "safe_dry", None)
    _observe(dry or getattr(rh, "safe_off", None) or getattr(rh, "stop", None),
             "eis_validate_rh_release_failed")
    if dry is not None:
        _observe(_report_dry_purge, "eis_validate_rh_release_report_failed", rh)


def _report_dry_purge(rh: Any) -> None:
    """Say on the console which end state the chamber was left in.

    Routed through :func:`_observe` by its caller for the reason ``_observe``
    exists: this is narration, and narration must never be why a refusal turns
    into a different exception.
    """
    from softae.core.safe_park import RH_DEADMAN_S

    err = getattr(rh, "last_safe_dry_error", "")
    if isinstance(err, str) and err:
        logger.warning("eis_validate_rh_dry_purge_failed", error=err)
        print(f"[approach] rh DRY-PURGE PARK FAILED: {err}", flush=True)
        return
    duty = float(getattr(rh, "last_safe_dry_duty", 0.0))
    logger.info("eis_validate_rh_dry_purge", duty=duty)
    print(f"[approach] rh loop released to a DRY PURGE at duty {duty:g}: dry air "
          f"keeps flowing, then the Trinket's deadman shuts both valves after "
          f"~{RH_DEADMAN_S:g} s. The chamber holds its dry state -- a retry does "
          f"not pay for the descent twice.", flush=True)


def _approach_one(
    approach_setpoint: Any, read_pv: Any, target: float, *, axis: str,
    instrument: str, tolerance: float, timeout_s: float,
    poll_interval_s: float, sleep: Any, now: Any, lead_s: float = 0.0,
) -> ApproachReport:
    """One axis, one bounded retry, then refuse. **The first policy inversion.**

    **``timeout_s`` bounds the judging, and ``elapsed_s`` measures the same
    window**, so the number the operator reads and the number the refusal quotes
    are the same clock. For RH the loop has already been driving for *lead_s*
    when this is called, and charging that lead against the timeout was rejected
    on two grounds. The bound is calibrated from a descent measured *at 85 C*
    (~5000 s, 2026-08-11) -- i.e. against the floor that exists once the block is
    hot -- so time spent at some other temperature is not the quantity it bounds.
    And the consequence would be perverse: two full temperature attempts is
    3600 s out of a 5400 s budget, so a slow heat would refuse an RH approach
    that was descending exactly as measured. The lead is *reported* rather than
    charged -- :attr:`ApproachReport.lead_s`, with
    :attr:`ApproachReport.driven_s` for the total time under command.
    """
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
                        elapsed_s=elapsed, lead_s=float(lead_s),
                        attempts=attempt)
            return ApproachReport(axis, float(target), True, elapsed,
                                  float(outcome.pv_final), attempt,
                                  float(lead_s))
        logger.warning("eis_validate_approach_timeout", axis=axis,
                       target=float(target), pv=float(outcome.pv_final),
                       attempt=attempt, timeout_s=float(timeout_s),
                       lead_s=float(lead_s))
    # The lead is named in the refusal because it changes what the refusal
    # means: a chamber that missed the band having had the whole heat as a head
    # start is a different diagnosis from one that missed it from a cold write.
    head_start = (f", after already driving for {lead_s / 60:.0f} min during the "
                  "temperature approach" if lead_s > 0 else "")
    raise RefuseToStart(
        f"{axis} never reached {target:g} within {tolerance:g} after two "
        f"attempts of {timeout_s:.0f} s (last PV {outcome.pv_final:g})"
        f"{head_start}. "
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
    #: Could the configured tolerance be met **at all**, on this board's own
    #: measured scatter? ``None`` when the question was never answerable. The
    #: gate computed this every round through
    #: :meth:`~softae.analysis.equilibration.SettleTracker.endorsement` and no
    #: caller here ever asked for it, so a run could spend its whole ceiling
    #: chasing a tolerance its own noise floor forbade -- and did, on
    #: ``20260820T183625Z_eis_validate``. Trailing, with a default, because this
    #: class is constructed positionally.
    tolerance_achievable: bool | None = None
    #: The sentence :func:`~softae.analysis.equilibration.endorse_tolerance`
    #: produced. Carried verbatim rather than rebuilt, so the refusal an operator
    #: reads is the one the rule wrote.
    endorsement: str = ""
    #: Median relative scatter across the participating channels of the last
    #: judged window -- the number the endorsement was decided against.
    noise_floor_rel: float | None = None
    #: The cells the rate criterion certified quiet at the last judged window.
    #: Populated whenever a rate was computed, under ``--survivors`` on OR off:
    #: recording is not routing, and the denominator is worth having either way.
    survivors: list[int] = field(default_factory=list)
    #: ``{channel: why}`` for every cell that did not survive, in the criterion's
    #: own refusal vocabulary. **This is the denominator.** Without it "11 of 13
    #: settled" is unrecoverable after the fact, which is exactly what made
    #: 20260821T173111Z and 20260821T192508Z impossible to diagnose.
    dropped: dict[int, str] = field(default_factory=dict)
    #: The band census recomputed over :attr:`survivors` -- what
    #: ``--min-treatment`` must be judged against once cells have been dropped.
    #: Empty when nothing was partitioned.
    survivor_projected: dict[str, int] = field(default_factory=dict)
    #: Which post-drop floor broke, in words, or ``""``. Non-empty means the
    #: partition was attempted and refused, and the verdict stayed a refusal.
    survivor_refusal: str = ""
    #: Mean rate across the cells that produced a fit, ln-units per hour.
    #: Reported and never routed on -- pooling certifies the population and this
    #: gate's endpoints are per cell.
    pooled_rate_per_hour: float | None = None

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
        SETTLE_CRITERION_DEVIATION,
        SETTLE_DISABLED,
        SettleTracker,
        r1_lower_bound_ohms,
        rate_tol_ln_per_hour,
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

    # The bound the gate had no way to use until `_round_fit` started reporting
    # the quantity it describes. `is_railed` returned False unconditionally while
    # `r1_ohms` held a raw low-frequency real part -- correctly, since the model's
    # R1 floor says nothing about that number -- so `EXCLUDED_RAILED` was
    # structurally unreachable here and a channel whose fit came to rest on the
    # floor reported the SAME number every round, which is what a stability
    # criterion is least able to refuse. 325 of 1440 fits in the reference run
    # sat on it while reporting success.
    #
    # `tol_rel` comes from the plan. It used to be omitted, which took the
    # shipped 0.10 silently -- correct for the board it was measured on and
    # arithmetically unsatisfiable on 20260821T173111Z_eis_validate, whose own
    # median scatter was 12.5-14 %.
    #
    # `criterion` selects which of the two sibling gates routes; `deviation` is
    # the shipped default and `both` routes on it while reporting the rate, so
    # the only configuration that changes a verdict is the one an operator asked
    # for by name. The rate band arrives in DECADES per hour -- the unit H3 is in
    # and the unit the spec derives it in -- and is converted once, here.
    criterion = str(plan.settle_criterion)
    rate_tol = (None if criterion == SETTLE_CRITERION_DEVIATION
                else rate_tol_ln_per_hour(plan.settle_rate_tol_dec_per_h))
    tracker = SettleTracker(
        enabled=True, tol_rel=plan.settle_tol_rel,
        rh_stability_pct=DEFAULT_RH_STABILITY_PCT,
        r1_bound_ohms=r1_lower_bound_ohms(plan.circuit_model),
        criterion=criterion, rate_tol_per_hour=rate_tol)
    if (wide := loose_band_notice(plan.settle_tol_rel)):
        print(wide, flush=True)
    rh = manager.get(RH_CONTROLLER)
    apexes: dict[int, float] = {}
    start = float(now())
    rounds = 0
    stopped_early = False
    announced: dict[str, Any] = {}
    endorsed: bool | None = None
    endorsement = ""
    floor_rel: float | None = None

    while True:
        fits: list[Any] = []
        for channel in plan.channels:
            try:
                eis = measure(channel)
            except Exception as exc:
                logger.warning("eis_validate_settle_sweep_failed",
                               channel=channel, error=str(exc))
                continue
            fits.append(_round_fit(channel, eis, plan.circuit_model))
            closure = arc_closure(eis.frequency, eis.z_imag_neg,
                                  getattr(eis, "phase", None))
            apex = float(closure.f_apex_interior_hz)
            if math.isfinite(apex) and apex > 0:
                apexes[int(channel)] = apex
        rounds += 1

        rh_median = _read_rh(rh)
        # Read before the round is recorded, because the round is what it stamps:
        # `elapsed` is the trailing window's time axis, and it must be aligned
        # with `tracker.rounds` the way `rh_medians` already is. A duration since
        # the phase began -- never a target, and no setpoint enters here.
        elapsed = float(now()) - start
        check = tracker.observe(fits, rh_median_pct=rh_median, t_s=elapsed)
        endorsed, endorsement, floor_rel = tracker.endorsement()
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
        _announce_rate(tracker, plan)
        _announce_endorsement(tracker, endorsed, endorsement, announced,
                              floor_rel=floor_rel)
        _announce_basis(fits, check, tracker.min_channels, plan.circuit_model,
                        announced)
        # Routed through `_observe` for the reason `_observe` exists: the table
        # is a monitoring convenience and must never be why a gate refuses.
        _observe(_print_trend, "eis_validate_trend_render_failed",
                 tracker.rounds, plan, check, apexes, rounds)
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
            # Whether the tolerance was reachable AT ALL on this board's own
            # scatter -- gate state, and the question a reader of a `ceiling`
            # asks first. A median relative deviation, which is the same
            # dimensionless, geometry-free ratio the deviations already are.
            "tolerance_achievable": endorsed,
            "endorsement": endorsement,
            "noise_floor_rel": None if floor_rel is None else round(floor_rel, 5),
            # How many channels this round produced a number the circuit model
            # stands behind, and why the others left the window. Both are gate
            # state and neither is an observable -- a count and a reason word,
            # not a resistance -- which is the same line `deviation_rel_by_
            # channel` sits on. Without them a run that stalls because its
            # spectra stopped fitting is indistinguishable, in the only record
            # that survives, from a run that stalled because the film moved.
            "n_modelled": _n_modelled(fits),
            "excluded_by_channel": _narrated_exclusions(check),
            # The criterion that ROUTED this round. One word, and without it a
            # reader of a `settled` cannot tell which of two gates said so.
            "settle_criterion": criterion,
            **_rate_payload(tracker),
        })

        if tracker.settled and elapsed >= floor_s:
            stopped_early = True
            break
        if elapsed >= plan.settle_max_hold_s:
            break
        sleep(max(0.0, period_s))

    verdict = tracker.outcome(stopped_early=stopped_early)
    projected = _project_populations(apexes, plan)
    outcome = SettleOutcome(
        verdict=verdict, n_rounds=rounds, elapsed_s=float(now()) - start,
        apex_by_channel=apexes, projected=projected,
        rh_median_pct=_read_rh(rh) if rh is not None else float("nan"),
        tolerance_achievable=endorsed, endorsement=endorsement,
        noise_floor_rel=floor_rel,
    )
    _apply_survivors(outcome, tracker, plan)
    return outcome


def _apply_survivors(
    outcome: SettleOutcome, tracker: Any, plan: ValidationPlan
) -> None:
    """Record the partition always; let it change the verdict only if asked.

    Two separable things, and collapsing them is how a feature that is "off by
    default" ends up changing a default. **Recording** the survivor set and the
    per-cell reason happens whenever a rate was computed at all, because the
    denominator costs nothing and its absence is what made both of the 2026-08-21
    runs undiagnosable. **Routing** on it happens only under ``--survivors on``.

    A cell proven MOVING keeps the run refusing, whatever the flag says. That is
    the locked distinction the whole criterion exists to draw: "this cell is
    moving" is evidence about the sample and blocks; "this cell cannot be judged"
    is an absence and may be dropped, recorded, and swept anyway.
    """
    from softae.analysis.equilibration import (
        SETTLE_CEILING,
        SETTLE_NOT_EVALUABLE,
        SETTLE_SURVIVORS,
    )

    rate = tracker.last_rate
    if rate is None:
        return
    outcome.pooled_rate_per_hour = rate.pooled_rate_per_hour
    outcome.survivors, outcome.dropped = survivor_partition(rate, plan.channels)
    if not plan.survivors:
        return
    bands = band_by_channel(outcome.apex_by_channel, plan)
    outcome.survivor_projected = {
        band: sum(1 for ch in outcome.survivors if bands.get(ch) == band)
        for band in set(bands.values())
    }
    _announce_survivors(outcome, tracker.min_channels)
    if outcome.verdict not in (SETTLE_CEILING, SETTLE_NOT_EVALUABLE):
        return
    if rate.moving:
        outcome.survivor_refusal = (
            f"{len(rate.moving)} cell(s) were PROVEN to be still moving, which "
            f"no partition may drop: "
            + " ".join(f"ch{ch}" for ch in rate.moving)
            + ". A moving cell invalidates the paired difference the run exists "
              "to make, so this is evidence about the sample and not an absence "
              "of it.")
        return
    outcome.survivor_refusal = survivor_floors(
        outcome.survivors, bands, min_channels=tracker.min_channels)
    if not outcome.survivor_refusal:
        outcome.verdict = SETTLE_SURVIVORS


def _announce_endorsement(
    tracker: Any, endorsed: bool | None, endorsement: str, announced: dict[str, Any],
    *, floor_rel: float | None = None,
) -> None:
    """Say, at the FIRST judged window, whether the tolerance is reachable at all.

    ``SettleTracker.endorsement`` has existed since the criterion did, is called
    by the campaign path and by the equilibration workflow, and was never called
    here -- so the one consumer that refuses on a ceiling was also the one that
    never asked whether any hold length could have cleared it. On
    ``20260820T183625Z_eis_validate`` the answer at round 3 was no, twenty-eight
    minutes before the ceiling it then ran to.

    Two scopes, because they answer different questions and disagree by design.
    The board's endorsement is taken over the **median** participant, so that one
    noisy cell does not condemn the setpoint; the criterion aggregates with
    **max**, so one noisy cell is exactly what holds the board. Only the
    per-channel view can name it, and naming it is the whole point -- an operator
    told "ch25 can never satisfy 10 %" fixes a cell, while one told "not yet"
    waits another hour.

    Announced on **change**, not every round: the first judged window carries the
    news, and a later line means the answer actually moved.

    An UNACHIEVABLE board also gets **the flag and a number**. "No hold length
    can satisfy it" is a complete diagnosis and an incomplete instruction: it
    tells an operator that waiting is not the fix without saying what is, and
    until this run there was nothing they could have changed anyway. The value
    is derived from the floor this board just measured (see
    :func:`suggested_settle_tol_rel`), never hardcoded, and it is deliberately
    *not* attached to the per-channel UNSETTLEABLE lines below: when one cell in
    fifteen is unsettleable the fix is that cell, and offering a wider band there
    would be advice to hide it.
    """
    if endorsed is not None and announced.get("board") is not endorsed:
        announced["board"] = endorsed
        mark = "ACHIEVABLE" if endorsed else "UNACHIEVABLE"
        print(f"[settle] tolerance {mark}: {endorsement}", flush=True)
        if endorsed is False and floor_rel is not None:
            suggestion = suggested_settle_tol_rel(floor_rel)
            if suggestion is None:
                print(f"         no admissible band clears a "
                      f"{floor_rel * 100:.2f}% floor (--settle-tol-rel stops at "
                      f"{SETTLE_TOL_REL_MAX:g}), so the cells are the finding "
                      f"here, not the flag", flush=True)
            else:
                print(f"         to hold this board to a band it can meet, "
                      f"restart with --settle-tol-rel {suggestion:g} (a RELATIVE "
                      f"sigma deviation, not %RH -- {suggestion * 100:.0f}%). "
                      f"--resume will refuse a changed band, by design: it is "
                      f"the criterion this run was registered under",
                      flush=True)
    named: set[int] = announced.setdefault("channels", set())
    for channel, (ok, _why, floor) in tracker.per_channel_endorsement().items():
        if ok is False and floor is not None and channel not in named:
            named.add(channel)
            print(f"[settle] ch{channel} UNSETTLEABLE: its own scatter "
                  f"{floor * 100:.1f}% exceeds the tolerance "
                  f"{tracker.tol_rel * 100:.2f}% -- no hold length can satisfy "
                  f"it, so waiting is not the fix", flush=True)


def _announce_rate(tracker: Any, plan: ValidationPlan) -> None:
    """The rate line, under the round's ``[settle]`` line. Silent by default.

    Printed on every round that produced a rate, unlike the endorsement lines
    above, because under ``both`` this IS the shadow measurement -- the two
    criteria read the same window and the operator is being asked to compare
    them, which cannot be done from a line that prints once.

    The tolerance is quoted in the unit the operator typed it in, and the rate
    beside it in the same one, because the gate's own arithmetic is in ln-units
    and a number printed in one unit next to a threshold in another is exactly
    the trap ``spread 0.130`` was.
    """
    from softae.analysis.equilibration import (
        LN_PER_DECADE,
        SETTLE_CRITERION_BOTH,
    )

    rate = tracker.last_rate
    if rate is None:
        return
    worst = rate.max_upper_bound_per_hour
    worst_text = "  n/a" if worst is None else f"{worst / LN_PER_DECADE:+7.4f}"
    pooled = rate.pooled_rate_per_hour
    pooled_text = ("  n/a" if pooled is None
                   else f"{pooled / LN_PER_DECADE:+7.4f}")
    shadow = (" (SHADOW -- deviation is what routed this round)"
              if tracker.criterion == SETTLE_CRITERION_BOTH else "")
    print(f"         rate: worst 95% bound {worst_text} dec/h  pooled "
          f"{pooled_text} dec/h  (tol "
          f"{plan.settle_rate_tol_dec_per_h:g} dec/h)  quiet "
          f"{len(rate.quiet)} moving {len(rate.moving)} unjudgeable "
          f"{len(rate.undetectable) + len(rate.unsettleable)}{shadow}",
          flush=True)
    if rate.moving:
        print("         still MOVING: "
              + " ".join(f"ch{ch}" for ch in rate.moving)
              + " -- a proven slope, which no partition may drop", flush=True)


def _rate_payload(tracker: Any) -> dict[str, Any]:
    """The rate's share of the ``on_round`` record -- **empty by default**.

    Absent rather than null under ``deviation``, so a reader of the stream can
    tell "this run did not compute a rate" from "this round's rate was
    unavailable", and so the shipped payload is byte-identical to today's.

    Every value here is a rate or a count -- dimensionless per hour, or a channel
    number -- which is the same line the deviations already sit on: gate state,
    never the observable behind it.
    """
    from softae.analysis.equilibration import LN_PER_DECADE

    rate = tracker.last_rate
    if rate is None:
        return {}
    per_channel = {
        str(ch): round(judged.rate_per_hour / LN_PER_DECADE, 6)
        for ch, judged in sorted(rate.by_channel.items())
        if judged.rate_per_hour is not None
    }
    return {
        "rate_evaluable": bool(rate.evaluable),
        "rate_settled": bool(rate.settled),
        "rate_dec_per_h_by_channel": per_channel,
        "pooled_rate_dec_per_h": (
            None if rate.pooled_rate_per_hour is None
            else round(rate.pooled_rate_per_hour / LN_PER_DECADE, 6)),
        "rate_quiet": list(rate.quiet),
        "rate_moving": list(rate.moving),
        "rate_unjudgeable": sorted(rate.undetectable + rate.unsettleable),
        "rate_span_s": round(rate.span_s, 1),
        "rate_reason": rate.reason,
    }


def _announce_survivors(outcome: SettleOutcome, min_channels: int) -> None:
    """The partition, at the drop, with the bias it introduces named out loud."""
    from softae.tools.eis_validate_report import CONTROL, TREATMENT, UNRESOLVED

    total = len(outcome.survivors) + len(outcome.dropped)
    census = outcome.survivor_projected
    detail = ", ".join(f"ch{ch} ({why})"
                       for ch, why in sorted(outcome.dropped.items()))
    print(f"[settle] SURVIVORS {len(outcome.survivors)}/{total}"
          + (f" -- dropped {detail}" if detail else " -- nothing dropped")
          + f". {CONTROL} {census.get(CONTROL, 0)} {TREATMENT} "
            f"{census.get(TREATMENT, 0)} {UNRESOLVED} "
            f"{census.get(UNRESOLVED, 0)}; the gate's minimum is "
            f"{int(min_channels)}.", flush=True)
    print("         Every number this run reports is now CONDITIONAL ON "
          "SETTLING. Dropped cells are still swept and are stamped so the "
          "population filter excludes them, and the reason for each is in the "
          "run's event stream -- read the survivor set as a subset, never as "
          "the board.", flush=True)


def _announce_basis(
    fits: Any, check: Any, min_channels: int, circuit_model: str,
    announced: dict[str, Any],
) -> None:
    """Say, on the round it happens, when the gate's number was not the fit's.

    The gate now reads a fitted R1 (see :func:`_round_fit`), and a fit can fail.
    The failure is **a property of the cell, not of the round**, so it will not
    arrive as scattered singletons -- it arrives as a population. Measured on
    ``20260820T164634Z_eis_validate``: ch22 carried non-finite points in all
    three of its sweeps and refused all three, while ch25 and ch32 carried none
    and refused none. A cell that drops points drops them every round.

    That population leaving the window silently is the failure this exists to
    prevent: below ``min_channels`` participants the criterion is *not
    evaluable*, which runs to the ceiling and then refuses, and an operator
    reading "not yet" every round for ninety minutes has no way to tell that
    from a film that is genuinely still moving. The two want opposite responses,
    exactly as :func:`_announce_endorsement`'s two do.

    Announced once per channel and once per stall, not once per round: a line
    every round is a line nobody reads.
    """
    from softae.analysis.equilibration import (
        BASIS_FITTED,
        EXCLUDED_RAILED,
        EXCLUDED_SIGMA_NULL,
    )

    named: set[int] = announced.setdefault("basis", set())
    for fit in fits:
        if fit.basis in ("", BASIS_FITTED) or int(fit.channel) in named:
            continue
        named.add(int(fit.channel))
        raw = ("and the sweep carried no readable Z' either"
               if fit.r_raw_ohms is None else
               f"its raw low-frequency Z' was {fit.r_raw_ohms:.4g} ohm, recorded "
               "as a diagnostic but NOT substituted -- that number is noisiest "
               "on exactly the cells whose fit fails")
        # "no USABLE R1", because a fit that railed on the model's bound also
        # lands here now: the route demotes it to `success=False` with a NaN R1
        # rather than reporting the bound as a measurement. See `_fitted_r1`.
        print(f"[settle] ch{int(fit.channel)} NO FIT: {circuit_model} produced no "
              f"usable R1, so this round carries no sigma and the channel drops "
              f"out of the window ({raw})", flush=True)

    if check is None or check.evaluable or announced.get("starved"):
        return
    blamed = sorted(int(ch) for ch, why in check.excluded.items()
                    if why in (EXCLUDED_SIGMA_NULL, EXCLUDED_RAILED))
    if not blamed or len(check.participating) >= int(min_channels):
        return
    announced["starved"] = True
    print(f"[settle] NOT EVALUABLE because of the FITS, not the sample: "
          f"{len(check.participating)} channel(s) participate against a minimum "
          f"of {int(min_channels)}; "
          f"{', '.join(f'ch{ch}' for ch in blamed)} carry no usable fitted R1. "
          f"Holding longer cannot fix this -- the gate will run to its ceiling "
          f"and refuse.", flush=True)


def _n_modelled(fits: Any) -> int:
    from softae.analysis.equilibration import BASIS_FITTED

    return sum(1 for fit in fits if fit.basis == BASIS_FITTED)


def _narrated_exclusions(check: Any) -> dict[str, str]:
    """``{channel: why}`` in words the **event stream** is allowed to carry.

    ``events.jsonl`` is narration and a test asserts over its raw bytes that no
    observable's vocabulary appears in it -- ``sigma``, ``r1``, ``fit`` and
    ``ohms`` are all forbidden substrings. The ``EXCLUDED_*`` constants spell two
    of those (``sigma_null``, ``railed_R1``), so the reason is narrated in words
    that say the same thing without naming the quantity. Nothing is lost: the
    console line beside it carries the number, and the constants stay unchanged
    for every caller that is not a stream.
    """
    if check is None:
        return {}
    return {str(ch): _exclusion_word(why)
            for ch, why in sorted(check.excluded.items())}


def _exclusion_word(why: str) -> str:
    """One ``EXCLUDED_*`` constant, in the vocabulary a stream may carry."""
    from softae.analysis.equilibration import (
        EXCLUDED_ABSENT,
        EXCLUDED_RAILED,
        EXCLUDED_SIGMA_NULL,
        EXCLUDED_ZERO_MEAN,
    )

    return {EXCLUDED_ABSENT: "absent", EXCLUDED_SIGMA_NULL: "no_value",
            EXCLUDED_RAILED: "railed",
            EXCLUDED_ZERO_MEAN: "zero_mean"}.get(why, "excluded")


# ── Surviving-channel mode ───────────────────────────────────────────────────

def survivor_partition(
    rate: Any, channels: Sequence[int]
) -> tuple[list[int], dict[int, str]]:
    """``(survivors, {channel: why})`` -- who the rate criterion can speak for.

    **Survivorship bias lives here, and a reader meets it at this function.**
    Selecting cells *for having settled* conditions every number downstream on
    settling. For THIS harness that is defensible and arguably desirable: the
    question is whether an acquisition strategy resolves the arc, and comparing
    two arms on a cell that is still drying is precisely what H3 exists to
    forbid, so restricting to quiet cells removes a confound rather than adding
    one. For a campaign objective it is **not** defensible -- dropping cells that
    never settle biases the objective toward materials that equilibrate fast,
    which is a material property correlated with the thing being optimised --
    which is why ``--survivors`` exists only in this tool. For equilibration
    characterization it is worse still: the dropped cells are the ones whose hold
    time such a run exists to measure. Anyone reading a survivor set as a board
    is reading a conditional distribution as a marginal one.

    Every channel is accounted for, including the ones that never reached the
    window: an unexplained absence from both lists is how a denominator goes
    missing.
    """
    if rate is None:
        return [], {}
    quiet = {int(ch) for ch in rate.quiet}
    survivors = sorted(ch for ch in map(int, channels) if ch in quiet)
    dropped: dict[int, str] = {}
    for channel in sorted(map(int, channels)):
        if channel in quiet:
            continue
        judged = rate.by_channel.get(channel)
        if judged is not None and judged.refusal:
            dropped[channel] = str(judged.refusal)
        elif channel in rate.excluded:
            dropped[channel] = _exclusion_word(rate.excluded[channel])
        else:
            dropped[channel] = "absent"
    return survivors, dropped


def survivor_row_stamp(why: str) -> str:
    """The per-row ``hold_certified`` word for one drop reason.

    Coarse where :func:`survivor_partition` is fine-grained, because a row stamp
    is read by a population filter and the artifact is read by a person. The
    filter needs one bit -- was this cell certified -- and the person needs the
    refusal.
    """
    from softae.analysis.equilibration import (
        DROPPED_STILL_MOVING,
        DROPPED_UNEVALUABLE,
        RATE_MOVING,
    )

    return DROPPED_STILL_MOVING if why == RATE_MOVING else DROPPED_UNEVALUABLE


def survivor_floors(
    survivors: Sequence[int], bands: dict[int, str], *, min_channels: int
) -> str:
    """Why this survivor set is not evidence, or ``""`` if it is.

    **Checked AFTER the drop, never before**, which is the whole discipline of
    the feature: a floor cleared by the board says nothing about the set that
    remains, and a run whose survivors are all CONTROL has proven nothing at all.
    Two floors here; ``--min-treatment`` is the third and stays with its owner in
    :mod:`softae.tools.eis_validate`, recomputed there against
    :attr:`SettleOutcome.survivor_projected` rather than restated here.

    Returns a sentence rather than raising, on
    :func:`~softae.analysis.equilibration.settle_tol_rel_refusal`'s precedent:
    the caller decides that a broken floor means the verdict stays a refusal, and
    the refusal it raises is its own.
    """
    from softae.tools.eis_validate_report import CONTROL, TREATMENT

    kept = sorted(int(ch) for ch in survivors)
    needed = max(1, int(min_channels))
    if len(kept) < needed:
        return (f"{len(kept)} channel(s) survived the drop, below the settle "
                f"gate's minimum of {needed}. A survivor set that small is not "
                f"evidence, and falling back to the whole board would be "
                f"treating an absence of evidence as evidence.")
    census = {band: sum(1 for ch in kept if bands.get(ch) == band)
              for band in (CONTROL, TREATMENT)}
    if not census[CONTROL] or not census[TREATMENT]:
        empty = CONTROL if not census[CONTROL] else TREATMENT
        return (f"the survivors carry no {empty} cell "
                f"({CONTROL} {census[CONTROL]}, {TREATMENT} "
                f"{census[TREATMENT]}). D1, D2 and D4 are {TREATMENT} "
                f"statistics and D3 is the {CONTROL} noise floor, so a run with "
                f"one arm missing cannot evaluate its own decision rule.")
    return ""


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


def _print_trend(
    history: list[Any], plan: ValidationPlan, check: Any,
    apexes: dict[int, float], round_index: int,
) -> None:
    """The per-channel signed table, under the round's ``[settle]`` line.

    The gate's line reports a **magnitude** -- how far the worst channel sits
    from its own recent mean -- which cannot distinguish a film still taking up
    water from one merely jittering around a settled value, and those two want
    opposite decisions from the operator. :mod:`softae.tools.eis_validate_trend`
    renders the same rounds signed, against a baseline that excludes the current
    reading.

    **Console only, deliberately.** Not added to the ``on_round`` payload: the
    published stream carries the *gate's state*, and per-channel sigma was kept
    out of it by an earlier decision that this view does not reopen. The band is
    passed through because the operator has been correlating drift against it by
    hand -- and it is marked provisional in the legend, because ``apexes`` is
    read off pre-equilibration sweeps.
    """
    from softae.tools.eis_validate_trend import (
        render_trend_legend,
        render_trend_table,
        trend_rows,
    )

    rows = trend_rows(
        history, plan.channels,
        bands=band_by_channel(apexes, plan),
        excluded=None if check is None else check.excluded,
        participating=None if check is None else check.participating,
    )
    if round_index <= 1:
        print(render_trend_legend(), flush=True)
    print(render_trend_table(rows), flush=True)


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


def _tolerance_clause(outcome: SettleOutcome) -> str:
    """The sentence that decides what an operator does about a ceiling.

    ``ceiling`` says the criterion was evaluable and said no. It does not say
    whether **any** hold length could have said yes, and those two want opposite
    responses: wait longer, or stop asking for a tolerance the board's own
    scatter forbids. :func:`~softae.analysis.equilibration.endorse_tolerance`
    already answers it from the run's own measurement; the refusal quotes that
    answer rather than restating the rule.

    Silent when the question was never answerable -- an absent endorsement is
    not an endorsement, and appending "unknown" to a refusal adds nothing an
    operator can act on.
    """
    if outcome.tolerance_achievable is None or not outcome.endorsement:
        return ""
    verdict = ("The tolerance WAS achievable on this run's own scatter, so more "
               "rounds were the missing ingredient: "
               if outcome.tolerance_achievable else
               "The tolerance was NEVER achievable on this run's own scatter, "
               "so no hold length would have cleared it: ")
    return f" {verdict}{outcome.endorsement}"


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
        SETTLE_SURVIVORS,
    )

    if outcome.verdict in (SETTLE_CEILING, SETTLE_NOT_EVALUABLE):
        # `survivor_refusal` names the floor the partition broke, and it is
        # appended rather than substituted: the ceiling is still WHY the run is
        # stopping, and the broken floor is why the escape hatch did not save it.
        survivors = (f" --survivors was on and did not rescue it: "
                     f"{outcome.survivor_refusal}"
                     if outcome.survivor_refusal else "")
        raise RefuseToStart(
            f"the settle gate returned `{outcome.verdict}` after "
            f"{outcome.n_rounds} rounds ({outcome.elapsed_s / 60:.1f} min). "
            "The material was never shown to have stopped moving, and "
            "'undeclared is unknown, never empty' -- refusing to start."
            + _tolerance_clause(outcome) + survivors
        )
    if outcome.verdict == SETTLE_SURVIVORS:
        # Allowed through, and never silently: this is a weaker claim about a
        # smaller board, every row it produces is stamped with which side of the
        # partition its cell landed on, and the outcome is conditional on
        # settling. `certified` stays False, so nothing downstream reads it as a
        # clean hold.
        print(f"  ! --survivors: proceeding on {len(outcome.survivors)} of "
              f"{len(outcome.survivors) + len(outcome.dropped)} cells. Every "
              f"row is stamped hold_certified=survivors or dropped_*, and every "
              f"result is CONDITIONAL ON SETTLING.")
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


def _round_fit(
    channel: int, eis: Any, circuit_model: str = SETTLE_CIRCUIT_MODEL
) -> Any:
    """One channel's contribution to a settle round, **with a sigma**.

    ``settle_check`` excludes any channel carrying a NULL sigma from a round --
    correctly, because "constant because nothing measured it" passes a stability
    test perfectly -- so a caller that supplies only ``r1_ohms`` makes every
    window ``not_evaluable`` and runs to its ceiling for nothing.

    The sigma supplied here is ``1/R1``, i.e. sigma with the cell constant set to
    1. That is exact rather than approximate for this gate's purpose: the check
    is a **per-channel relative deviation from that channel's own mean**, so the
    channel's real ``K`` cancels out of it entirely -- the same reason the
    harness's whole comparison is geometry-free. No geometry is resolved, and
    none is needed.

    **``R1`` is the circuit fit's, and this reverses an earlier decision made on
    cost.** What stood here said the gate wanted "something monotone in the arc
    size and available on every round", took the low-frequency real part, and
    priced a fit at "2.6 s per open-arc fit". The cost was the whole argument and
    it was wrong twice over:

    - *Measured* (2026-08-21) through the sanctioned route,
      :func:`~softae.analysis.eis.engine.analyze_spectrum` with ``engine`` unset
      and ``[eis] engine = "legacy"``, on the ten real sweeps of run
      ``20260820T164634Z_eis_validate``: **16.5 - 120.3 ms** where the fit
      converges and **0.7 - 1.2 ms** where it refuses -- against the 2.6 s
      quoted, i.e. **20x to 150x** cheaper. Fifteen channels is ~1 s of a 562 s
      round even at the slow end. The pacing budget does not notice this. See
      :func:`_fitted_r1` for the direct-fitter comparison and for the one config
      under which this stops being true.
    - The number it bought was defective. ``z_real[-1]`` is the real part **at the
      lowest swept frequency**, which on a cell whose arc has not closed there is
      `R + arc contribution` and moves with anything that moves the arc. On run
      ``20260820T183625Z_eis_validate`` ch25 sat at 87/88/92/87 % relative
      deviation across four consecutive windows -- flat scatter about a stable
      mean, not drift -- while thirteen of fifteen channels converged to 6-22 %,
      and it held the whole board at the ceiling for an hour. The gate takes the
      MAX across channels, so one such cell is enough.

    A future reader must not re-derive the raw point as an optimisation: it is
    not that it costs nothing, it is that what it saves is measured in
    milliseconds and what it costs is measured in hours at temperature.

    **No fallback to the raw point when the fit fails, and the reason is not the
    one first written here.** That reason was "the fit refuses when the data put
    its R1 guess outside the model's bounds, which is what an unclosed arc does".
    That mechanism is real -- it is what a synthetic open arc produces -- but it
    is **not** what the real refusals are. On run
    ``20260820T164634Z_eis_validate``, measured 2026-08-21, all three of ch22's
    sweeps refuse with ``array must not contain infs or NaNs`` and none of them
    ever reaches a bounds check:

    ==============================  ========  =============  ==================
    sweep                           points    non-finite     ``z_real[-1]``
    ==============================  ========  =============  ==================
    ``ch22_001_reference``          53        **64** / 265   1.664e+08 (finite)
    ``ch22_002_adaptive_scout``     27        **16** / 135   3.361e+07 (finite)
    ``ch22_003_reference_end``      53        **4** / 265    1.682e+08 (finite)
    every ``ch25`` and ``ch32``     27-53     **0**          finite
    ==============================  ========  =============  ==================

    (Cells are points x five stored arrays -- ``frequency``, ``z_magnitude``,
    ``phase``, ``z_real``, ``z_imag_neg``; the frequency axis is always clean, so
    the non-finite count is four per dropped point.)

    **A single-point read cannot notice that a quarter of the spectrum is NaN.**
    ``z_real[-1]`` is one array cell, and on all three ch22 sweeps it is finite
    and perfectly plausible -- so the old path counted a corrupt spectrum as
    evidence and let ch22 sit at 47-55 % deviation for an hour without ever
    saying anything was wrong with the data. The fit reads all 53 points and
    refuses; the raw point reads one and cannot. Falling back would therefore
    restore, under the fitted path's name, exactly the read that could not see
    the defect that caused the refusal -- and it would do so on the cells where
    the defect is densest. ``sigma=None`` excludes the channel and
    :func:`_announce_basis` says so on the round it happens. The raw value is
    still carried, in ``r_raw_ohms``, so a shadow run can measure what the fit
    actually bought without a second bench run.

    ``r1_ohms`` is NaN rather than ``None`` on a failed fit: an all-``None``
    ``RoundFit`` reads to ``_exclusion`` as *absent* -- a sweep that never
    happened -- and that is a different finding from a sweep that happened and
    could not be fitted.
    """
    from softae.analysis.equilibration import (
        BASIS_ABSENT,
        BASIS_FIT_FAILED,
        BASIS_FITTED,
        RoundFit,
    )

    raw = _low_frequency_real(eis)
    r1 = _fitted_r1(eis, circuit_model)
    if r1 is None:
        return RoundFit(
            channel=int(channel), sigma=None,
            r1_ohms=None if raw is None else float("nan"),
            basis=BASIS_ABSENT if raw is None else BASIS_FIT_FAILED,
            r_raw_ohms=raw)
    return RoundFit(channel=int(channel), sigma=1.0 / r1, r1_ohms=r1,
                    basis=BASIS_FITTED, r_raw_ohms=raw)


def _low_frequency_real(eis: Any) -> float | None:
    """``Z'`` at the lowest swept frequency -- a diagnostic now, not the gate."""
    try:
        value = float(eis.z_real[-1])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _fitted_r1(eis: Any, circuit_model: str) -> float | None:
    """The route's R1 in ohms, or ``None`` if it produced no usable number.

    **Through** :func:`~softae.analysis.eis.engine.analyze_spectrum`, **with
    ``engine`` left unset.** An earlier version of this function called
    ``fit_circuit`` directly and argued the gate "wants a resistance and nothing
    else" -- true, and not a licence to open a second route. User ruling ``[a23]``
    is that one resolver decides which physics runs, "as nothing changes about
    the casting nor measurement between them", and a settle round is the same
    casting and the same measurement as the reference sweep taken forty minutes
    later. ``SpectrumReport.fit`` is the fitter's own ``FitResult``, so R1 is
    reachable through the sanctioned route without asking the route for anything
    it does not already return.

    **The cost argument that justified the shortcut does not survive
    measurement.** Re-measured warm on the ten stored sweeps of run
    ``20260820T164634Z_eis_validate`` (2026-08-21), median of five, config
    ``[eis] engine = "legacy"``:

    ===================  ==================  ====================
    per spectrum         ``fit_circuit``     ``analyze_spectrum``
    ===================  ==================  ====================
    successful fit       15.5 - 123.5 ms     16.5 - 120.3 ms
    refused fit          0.9 - 1.2 ms        0.7 - 1.2 ms
    ===================  ==================  ====================

    The wrapper's arc annotation, quality grading and (withheld) sigma are
    sub-millisecond against a fit that is tens of milliseconds; the difference is
    inside the run-to-run noise. Fifteen channels is ~1 s of a ~562 s round.

    **``cell=None``, deliberately, and no sigma is taken from the report.** The
    gate compares a channel against *itself*, so the cell constant cancels out of
    every statistic here -- see :func:`_round_fit`. There is no thickness and no
    area to supply, and supplying a nominal one to make the report carry a sigma
    would put a fabricated geometry into the one phase that is geometry-free by
    construction. ``report.sigma`` is therefore ``mode="unavailable"`` on every
    round, and nothing reads it: ``1/R1`` is built here instead.

    **``blocking=True``, stated rather than inherited.** It is the default, and
    on the legacy engine it is inert -- ``_legacy_report`` never builds a gate
    context -- so leaving it off would look identical today and would be a
    silent bet that the flag stays inert. It does not: under
    ``[eis] engine = "gated"`` ``blocking`` reaches ``build_context`` and decides
    whether ``gate_hf_inductance`` treats a high-frequency ``Im Z > 0`` run as
    artefact and truncates it, and whether the Lin-KK ladder is fitted with a
    series capacitance. These are ionically **blocking** coplanar cells --
    polymer electrolyte on inert stripes, no faradaic couple, which is why the
    low-frequency tail is capacitive and why the arcs do not close -- and a
    settle round measures the same cells the rest of the harness does. The value
    is a fact about the hardware, not about this phase, so it is written down
    where a reader can check it against the board.

    **A railed fit now arrives here as no fit, and that is a behaviour change.**
    ``analyze_spectrum`` runs ``_demote_if_railed``, which clears ``success`` and
    NaNs ``R1`` on a fit resting on the model's R1 bound -- naming the settle
    criterion, by name, as one of the consumers that were taking a property of
    ``CIRCUIT_MODELS`` for an observation. So the exclusion this phase used to
    make itself, via the ``r1_bound_ohms`` it hands ``SettleTracker``, is now
    made one level upstream and arrives as ``BASIS_FIT_FAILED`` /
    ``EXCLUDED_SIGMA_NULL`` rather than ``EXCLUDED_RAILED``. The channel leaves
    the window on the same round either way and the verdict is unchanged; what
    moves is the word in ``excluded_by_channel`` -- and the console gains a line,
    because a railed fit used to count as ``BASIS_FITTED`` and so was never named
    by :func:`_announce_basis`. The bound stays wired: it is a second line of
    defence for a caller that reaches ``SettleTracker`` with a railed R1 in hand,
    which is still every path that reads a stored fit.
    """
    from softae.analysis.eis.engine import analyze_spectrum

    try:
        report = analyze_spectrum(eis, cell=None, model_name=str(circuit_model),
                                  blocking=True)
    except Exception as exc:
        logger.warning("eis_validate_settle_fit_raised",
                       model=str(circuit_model), error=str(exc))
        return None
    fit = report.fit
    if fit is None:
        # The gated engine withholds the fit entirely on a spectrum its
        # admission gates reject. Never reached while `[eis] engine` is legacy.
        logger.warning("eis_validate_settle_fit_withheld",
                       model=str(circuit_model), engine=str(report.engine))
        return None
    if not fit.success:
        logger.warning("eis_validate_settle_fit_failed",
                       model=str(circuit_model), error=str(fit.error_msg))
        return None
    r1 = float(fit.R1) if fit.R1 is not None else float("nan")
    return r1 if math.isfinite(r1) and r1 > 0 else None


__all__ = [
    "DEFAULT_DRIFT_CHECK", "DEFAULT_MIN_TREATMENT",
    "DEFAULT_RH_APPROACH_TIMEOUT_S", "DEFAULT_SETTLE_MAX_HOLD_S",
    "DEFAULT_SOAK_S", "DEFAULT_TEMP_APPROACH_TIMEOUT_S",
    "DEFAULT_TEMP_DESCENT_TIMEOUT_S", "SETTLE_CIRCUIT_MODEL",
    "SOAK_CEILING_FACTOR",
    "SOAK_POLL_INTERVAL_S", "SOAK_PRINT_EVERY_N_POLLS",
    "ApproachReport", "HoldWatch", "Projection", "RefuseToStart",
    "SettleOutcome", "SoakOutcome", "ValidationPlan", "VirtualClock",
    "approach_condition", "assert_settle_licensed", "band_by_channel",
    "classify_apex", "population_thresholds", "project", "render_arc_watch",
    "render_projection", "settle_deviations", "settle_phase", "soak_phase",
    "validate_plan",
]
