"""P.22 — the equilibration characterization run: σ(t) while the chamber is brought
to condition.

Records EIS **repeatedly over time** while humidity is established at a low
setpoint and temperature is swept up and then back down, to obtain per-sample
equilibration-time statistics. It answers three open questions in one bench
session: how long each sample takes to equilibrate, whether the rig can *hold*
the RH setpoint at each temperature, and whether σ(T) retraces up vs down.

Why a sibling module rather than a fourth builder on ``ArrheniusSweep``
-----------------------------------------------------------------------
``temp_eis_sweep.py`` is 1249 lines against a ~400-line house rule, already
carries three DAG builders and two RH wait loops, and hands **one whole-sweep
DAG** to the executor. This run needs the opposite shape: a Python outer loop
that holds, keeps a ``HoldReport`` *as a value*, decides whether to continue, and
submits a small measurement-only DAG per round. Primitives are reused by import
(``eis_measure_step``, ``monitored_hold``, ``estimate_eis_duration``); nothing is
copied and ``ArrheniusSweep`` is not touched.

Two philosophies, and this run deliberately takes the other one
---------------------------------------------------------------
In a campaign, failing to hold a condition must refuse: the sample is no longer
being made as specified. **Here, discovering that the rig cannot hold 15 %RH at
85 °C is a primary result.** Same ``monitored_hold`` call, different consequence,
recorded either way. Two things stay genuine aborts and are told apart from an
unmet setpoint by **this module's own recorded series**, never by the exception
text — see :func:`watch_hold`.

No ``Longest`` anchor rounds — retired, and here is why
-------------------------------------------------------
The spec budgeted two ``Longest`` rounds per setpoint "where the low-frequency
information matters". That contradicts its own physics: the phase-reliable floor
on this fixture is **~9 Hz**, and ``Longest`` sweeps to **0.2 Hz**. The two most
expensive rounds of every setpoint — ~14.6 min each at the measured per-channel
overhead — were therefore buying points *below* the trustworthy phase floor, in a
design that elsewhere argues ``Quick`` (20 Hz) is the only clean preset. Most
paid for, least usable.

Dropping the preset to ``Quick`` would have made an "anchor round" byte-identical
to a series round, so the concept is removed rather than repriced. **Do not
reintroduce it** without first moving the phase floor.

The idea it was conflated with — a settled block at the start and another at the
end of the down leg, their agreement being the session-drift evidence — is
**kept**, at analysis time and at zero instrument cost: those blocks are the first
and last settled series blocks and already exist. See
:func:`softae.analysis.equilibration.session_drift`.

``rounds_per_setpoint`` is a CEILING, not a count
-------------------------------------------------
**This changed what an existing ``--rounds`` means.** A setpoint now stops as soon
as σ has settled and the hold floor has elapsed, and only runs the full
``rounds_per_setpoint`` when it has not. The 2026-08-11 production run is the
argument: the σ swing was 1600–2800 % at the first setpoint, 57–1370 % at the
second, and 0.5–8.5 % and 0.8–3.1 % at the third and fourth — flat to within a
5.98 % measured noise floor. Seven of eight setpoints were held for 45 minutes
each to re-measure a number that had stopped moving.

The criterion, its participation rule (a fit railed on the model's R₁ bound is
**not** evidence, however constant it looks) and its refusal to treat an absence
of evidence as settling all live in
:mod:`softae.analysis.equilibration` — they are a pure function over a window of
fits and want no rig to test. ``settle_enabled = False`` restores the old
fixed-count behaviour exactly.

The acquisition floor is **coupled to the fit minimum, where a τ is wanted**
-----------------------------------------------------------------------------
A criterion that can stop a setpoint after three rounds can produce a series the
offline τ fitter refuses: :data:`~softae.analysis.equilibration.MIN_POINTS_FOR_TAU`
is 5, because σ(t) = σ_∞ + (σ₀ − σ_∞)·exp(−t/τ) has three free parameters. How
many rounds a *time* floor buys depends entirely on the sampling interval: at the
660 s ``round_period_s`` that a ``Standard`` default once forced, **every** floor
bought fewer than five — ``ceil(1500/660) = 3`` at the first setpoint and
``ceil(600/660) = 1`` after — so a run could have stopped every setpoint short of
the fit minimum and produced no τ at all, which is the entire purpose of the run.
The interval is 200 s now and the first setpoint's floor alone buys 8, but that is
arithmetic working out rather than a guarantee, which is why the coupling below
stands.

So the fewest rounds a setpoint may run is

    ``max(settle_n_rounds, MIN_POINTS_FOR_TAU, ceil(min_hold_s / round_period_s))``

for the **first** ``tau_setpoints`` setpoints of the run, and

    ``max(settle_n_rounds, ceil(min_hold_s / round_period_s))``

after them. Where it applies it is a **self-consistency property, not a
tunable**: the acquisition side must not be able to emit a series the analysis
side will refuse. The constant is *imported* from
:mod:`softae.analysis.equilibration` rather than restated, so the two cannot
drift, and a ``rounds_per_setpoint`` below it is refused outright by
:meth:`EquilibrationConfig.validate` — a ceiling under the fit minimum is a design
in which none of those setpoints can ever yield a τ. That refusal is conditional
on ``tau_setpoints > 0``: with the window closed there is no τ to preserve, and a
settle-only sweep is a legitimate thing to ask for.

**Where it applies is the operator's call, and it is narrow.** The films dry once,
at the start of the session, and stay dry: measured per-setpoint σ swing on the up
leg of the 2026-08-11 run was 1600–2800 % at S0, 57–1370 % at S1, then 0.5–8.5 %
at S2 and 0.8–3.1 % at S3, against a 5.98 % noise floor. Past S1 there is no
relaxation left to fit, so forcing five rounds there spends instrument time buying
a τ that would be fitted to noise and that nobody can use. ``tau_setpoints``
defaults to 2 for that reason; ``0`` removes the floor everywhere and a value at
or above :attr:`EquilibrationConfig.n_setpoints` restores it everywhere.

Never calls ``wait()``
----------------------
Both driver ``wait()`` primitives fail open on timeout (they log and return), and
``temp_eis_sweep._abortable_wait_rh`` does the same. This module reaches none of
them: the approach *and* the hold are both ``monitored_hold``, which refuses. The
driver fix is a separate coordinated task and this run is correct without it.

Progress is **events, not printing**
------------------------------------
A run is ~9.3 h typical and 15.3 h worst case, and an operator standing at the rig
cannot otherwise tell a working run from a hung one. This module therefore emits
:class:`ProgressEvent` through :attr:`EquilibrationRun.on_progress` and renders
**nothing**; ``tools/equilibration.py`` owns every character that reaches a
console. That is the house pattern (``ArrheniusSweep`` exposes
``on_step_complete`` / ``on_step_error`` / ``on_eis_point`` and ``tab_arrhenius``
renders them), and it means a future GUI tab consumes the same stream with no
change here.

Every emission is wrapped: **a formatting bug or a closed pipe must not abort a
nine-hour experiment.** A cosmetic failure degrades to silence and increments
:attr:`EquilibrationRun.progress_failures`; it never propagates.
"""

from __future__ import annotations

import asyncio
import json
import math
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import structlog

from softae.analysis.eis.geometry import THICKNESS_METHODS
from softae.analysis.equilibration import (
    DEFAULT_MIN_HOLD_FIRST_S,
    DEFAULT_MIN_HOLD_S,
    DEFAULT_SETTLE_MIN_CHANNELS,
    DEFAULT_SETTLE_N_ROUNDS,
    DEFAULT_SETTLE_TOL_REL,
    MIN_POINTS_FOR_TAU,
    SETTLE_DISABLED,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    RoundFit,
    SettleTracker,
    load_round_fits,
    r1_lower_bound_ohms,
)
from softae.core.conditions_capture import ENV_KEYS, read_environment
from softae.core.deposition_steps import eis_measure_step
from softae.drivers.contracts import monitored_hold
from softae.errors import SafetyError
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)

#: Legs of the sweep, in execution order. ``"down"`` revisits every temperature.
LEGS = ("up", "down")
#: The one round kind. It stays as an explicit coordinate axis — in the step name,
#: in the sidecar point, and in the analysis key — because that key is persisted
#: and a constant column costs nothing next to a join that has to be re-derived.
#: There is no second value; see the module docstring on why the ``Longest``
#: anchors were retired.
KIND_SERIES = "series"
#: The three terms of ``sigma = L / (R * t * w)``, in that order. **All three or
#: none** — see :meth:`EquilibrationConfig._validate_geometry`.
GEOMETRY_TERMS = ("L_cm", "t_cm", "w_cm")
#: Fraction of a timeout a typical approach actually takes. A stated guess used
#: only for the *typical* projection column; the worst case uses the timeouts.
TYPICAL_APPROACH_FRACTION = 0.25

# ── Chamber defaults, named once ─────────────────────────────────────────────
#
# Declared here rather than inline in the dataclass because the CLI now exposes
# every one of them and quotes the default in its own ``--help``. Two spellings
# of a default is how a plan file comes to disagree with the run it describes.

#: %RH held at every temperature. **20, not 15, and not a controls fault.** The
#: flush basin holds water *inside* the heated enclosure, so warming the chamber
#: humidifies it with surplus moisture: the 2026-08-11 run commanded 15 %RH and
#: measured a PV of 16.9–20.4 at 65 °C and 19.5–23.2 at 85 °C. 15 % is below what
#: this enclosure can deliver hot, and commanding it produces an unmet setpoint at
#: every hot condition — a graded failure that is an artefact of the basin, not a
#: measurement of the rig. **Do not "optimise" this back down** without first
#: taking the water out of the enclosure.
DEFAULT_RH_SETPOINT_PCT = 20.0
#: Band that decides whether temperature was *held*. **2.0 °C, not 0.5.** At 0.5 a
#: 0.6 °C dip (PV 64.4 against a 65.0 setpoint) graded the whole down-leg S1 as
#: "hold not met" on a chamber that wanders a few tenths — an unmet verdict that
#: is not a failure, on a run where an unmet verdict is a primary result and has
#: to mean something. It stays well inside :data:`DEFAULT_WARN_C` (3.0) and far
#: inside :data:`DEFAULT_FAULT_C` (10.0), so the excursion warning and the runaway
#: guard both still fire strictly before and after this band, in that order.
DEFAULT_TOLERANCE_C = 2.0
DEFAULT_RH_TOLERANCE_PCT = 2.0
#: Allowance for driving temperature into band on the **ascending** leg, where the
#: heater is doing the work.
DEFAULT_APPROACH_TIMEOUT_S = 1800.0
#: The same allowance on the **descending** leg, where nothing is doing the work.
#: Cooling here is passive and asymptotic, and 1800 s is not enough: measured
#: down-leg approach times were 0.5 min at 85 °C, 12.0 at 65, 22.5 at 45 and 30.0
#: at 27.5 — where it hit the timeout **without reaching tolerance**. The stage was
#: still at 34.1 °C when measurement began and had only fallen to 29.0 °C by the
#: end of the 44-minute series, so those 15 rounds span a 5 °C ramp while labelled
#: 27.5 °C.
#:
#: 5400 s = the 1800 s already spent plus ~60 min. At the measured end-of-series
#: rate (~5 °C per 44 min) falling from 34.1 °C to 27.5 °C needs ~58 min, and to
#: the edge of the 2.0 °C band ~40 min. A separate timeout rather than a multiplier
#: on the ascending one: an operator who has to extend it needs to see the number
#: it is being extended from.
DEFAULT_DOWN_APPROACH_TIMEOUT_S = 5400.0
DEFAULT_RH_APPROACH_TIMEOUT_S = 1800.0
DEFAULT_WARN_C = 3.0
DEFAULT_FAULT_C = 10.0
DEFAULT_GRACE_S = 120.0
#: How many setpoints **of the run** the :data:`MIN_POINTS_FOR_TAU` round floor
#: applies to. See the module docstring: the films dry once, so a τ exists to be
#: fitted at the first setpoint or two and nowhere after.
DEFAULT_TAU_SETPOINTS = 2

#: Channels a default run measures — 1-16, and the count the sampling interval
#: below is derived at. Named because the derivation needs it and a second
#: spelling of "16" is how a derived default stops matching what it describes.
DEFAULT_N_CHANNELS = 16

#: EIS preset for the σ(t) series. **Quick, and this is what makes the shipped
#: defaults able to measure their own subject.**
#:
#: The sampling interval sets the shortest resolvable τ at roughly twice itself,
#: and the interval is floored by what an all-channel round costs. At ``Standard``
#: (40.85 s/channel measured) 16 channels cost 654 s, forcing a ≥660 s interval and
#: a τ floor of ~22 min — against a τ of ~500 s (8.3 min) measured at the first
#: setpoint. The stock configuration was sampling ~2.6x too coarsely to see the
#: transient this tool exists to characterise; it was a *feasible* default that
#: could not answer the question.
#:
#: ``Quick`` costs 10.47 s/channel measured, which is a 200 s interval and a ~6.7
#: min τ floor — under the measured τ, so the relaxation is resolved. It is also
#: what the operator's own production runs and ``scripts/equilibration_run.ps1``
#: already use, so this aligns the default with practice rather than inventing one.
#:
#: The counter-argument is arc closure: 33 % of the 1440 spectra in run
#: ``20260811T023757Z`` showed an unclosed arc at ``Quick`` (peak of -Z'' at the
#: lowest measured frequency), and ``Standard``'s 4 Hz floor would close some of
#: them. It does not justify a 4x cost on **every** round: ``Standard`` does not
#: close them at the cold end either, where ch1 sits at 6.5e7 Ω. The right answer
#: is to detect and report an unclosed arc, not to buy a slower sweep everywhere
#: in the hope of avoiding one.
DEFAULT_EIS_PRESET = "Quick"


@lru_cache(maxsize=1)
def model_underestimate_frac() -> float:
    """How far the sweep model may fall *under* a real round, as a fraction of itself.

    Read off the model's own calibration rather than chosen.
    ``estimate_eis_duration`` is fitted to three presets timed on this rig
    (:data:`~softae.core.preflight.EIS_MEASURED_S_PER_CHANNEL`) and reproduces each
    within ~8 %; on the one it under-counts — ``Standard``, ~37.7 s/channel modelled
    against 40.85 s measured — it is 8.2 % low.

    Computed here instead of written down so a re-fit of those constants moves
    everything sized against them. The threshold this replaced was a literal, and
    it outlived the model it was sized against by exactly one recalibration.

    Zero if the anchors are unavailable or the model never under-counts them; every
    caller then degrades to trusting the model as-is, which is the honest response
    to having no measurement of its error.
    """
    from softae.core.eis_scripts import EISParams
    from softae.core.preflight import EIS_MEASURED_S_PER_CHANNEL, estimate_eis_duration

    worst = 0.0
    for preset, measured in EIS_MEASURED_S_PER_CHANNEL.items():
        modelled = estimate_eis_duration(EISParams.from_preset(preset))
        if modelled > 0:
            worst = max(worst, float(measured) / modelled - 1.0)
    return worst


#: Per-round buffer added on top of the measured channel cost. **Chosen, not
#: derived, and deliberately kept as its own term** so the next person can see
#: which half of the default is measurement and which is judgement.
#:
#: It is *not* a statistical bound and must not be presented as one: the measured
#: per-channel spread across the anchors is ~±0.1 s, so ±1.6 s over 16 channels,
#: and 30 s is well above that. It is headroom for the per-round work no anchor
#: covers — executor construction, the mscr rebuild, DAG setup — plus operating
#: margin for a run nobody is watching.
ROUND_BUFFER_S = 30.0


def default_round_period_s(preset: str = DEFAULT_EIS_PRESET,
                           n_channels: int = DEFAULT_N_CHANNELS) -> float:
    """The shipped σ(t) sampling interval: a measured round plus a chosen buffer.

    Two terms, kept apart on purpose::

        n_channels × EIS_MEASURED_S_PER_CHANNEL[preset]   DERIVED  167.5 s
        + ROUND_BUFFER_S                                  CHOSEN    30.0 s
        rounded up to a typable ten                                200.0 s

    Rounded by the same rule :func:`minimum_feasible_period_s` uses, because a
    default an operator cannot retype from memory is a default they will type
    wrongly. 200 rather than 198 for that reason alone.

    Two properties worth checking against, neither of which drove the number:

    * The modelled round at ``Quick``/16 is ~180.5 s, and its worst case under
      :func:`model_underestimate_frac` is ~195.4 s — under 200, so an operator's
      first ``plan`` neither refuses the shipped defaults nor cautions on them. A
      period derived from the measurement alone (170 s) would have done both, the
      model over-counting ``Quick`` by 7.8 %.
    * 200 s resolves τ no shorter than ~6.7 min, against the ~500 s (8.3 min) τ
      measured at the first setpoint. The interval this replaced was 660 s — a
      ~22 min τ floor, ~2.6x too coarse to see the transient the run exists to
      characterise.

    Falls back to the model alone for a preset with no anchor, buffer included: an
    untimed preset gets a period that is honest about resting on an extrapolation
    rather than one that silently reads as measured.
    """
    from softae.core.eis_scripts import EISParams
    from softae.core.preflight import EIS_MEASURED_S_PER_CHANNEL, estimate_eis_duration

    n = max(1, int(n_channels))
    per_channel = EIS_MEASURED_S_PER_CHANNEL.get(preset)
    if per_channel is None:
        per_channel = (estimate_eis_duration(EISParams.from_preset(preset))
                       * (1.0 + model_underestimate_frac()))
    return math.ceil((float(per_channel) * n + ROUND_BUFFER_S) / 10.0) * 10.0


#: σ(t) sampling interval for a run that types nothing. See
#: :func:`default_round_period_s` — 200 s at the shipped preset and channel count.
DEFAULT_ROUND_PERIOD_S = default_round_period_s()

_ABORT_UNREADABLE = "unreadable_pv"
_ABORT_OVERSHOOT = "sustained_overshoot"


# ── Progress events ──────────────────────────────────────────────────────────

EV_RUN_STARTED = "run_started"
EV_LEG_STARTED = "leg_started"
EV_LEG_FINISHED = "leg_finished"
EV_SETPOINT_STARTED = "setpoint_started"
EV_SETPOINT_FINISHED = "setpoint_finished"
EV_APPROACH_STARTED = "approach_started"
EV_APPROACH_PROGRESS = "approach_progress"
EV_APPROACH_FINISHED = "approach_finished"
EV_ROUND_STARTED = "round_started"
EV_ROUND_FINISHED = "round_finished"
EV_CHANNEL_MEASURED = "channel_measured"
EV_HOLD_VERDICT = "hold_verdict"
EV_HEARTBEAT = "heartbeat"
EV_COST_WARNING = "cost_warning"
#: Why a setpoint's σ(t) series ended: it SETTLED, it hit the CEILING, or the
#: criterion was NOT EVALUABLE. A first-class result, not a log line — an
#: operator looking at a short setpoint has to be able to tell one that stopped
#: because σ stopped moving from one that stopped because nobody could tell.
EV_SETTLE_VERDICT = "settle_verdict"
EV_RUN_FINISHED = "run_finished"
#: Whether the chamber actually came back down. Emitted on **every** exit path,
#: successful or not, because a silent restore attempt and a successful one look
#: identical — and the difference is an operator who walks away from a heater
#: still commanded at 85 °C and one who does not.
EV_AMBIENT_RESTORED = "ambient_restored"

#: Events durable enough to be worth a ``structlog`` line each. The high-rate
#: ones (heartbeat, approach ticks, per-channel completions) are deliberately
#: absent: over 15 h they would be tens of thousands of lines and would bury the
#: verdicts, which are the primary result.
MILESTONE_EVENTS = frozenset({
    EV_RUN_STARTED, EV_LEG_STARTED, EV_LEG_FINISHED, EV_SETPOINT_STARTED,
    EV_SETPOINT_FINISHED, EV_APPROACH_STARTED, EV_APPROACH_FINISHED,
    EV_HOLD_VERDICT, EV_COST_WARNING, EV_SETTLE_VERDICT, EV_RUN_FINISHED,
    EV_AMBIENT_RESTORED,
})

VERDICT_MET = "met"
VERDICT_UNMET = "unmet"
VERDICT_ABORTED = "aborted"

#: ``env`` was never attempted on this event kind.
ENV_ABSENT = ""
#: ``env`` carries a fresh five-value snapshot. Individual values may still be
#: ``None`` — an unreadable or stale PV, which must render as unavailable.
ENV_OK = "ok"
#: Telemetry was **deliberately not read**, because an instrument was in use. A
#: distinct state from "read and failed": the controls are fine, we simply did
#: not interrupt to ask.
ENV_SKIPPED = "skipped"


@dataclass
class ProgressEvent:
    """One thing that happened, said once, with no opinion about how it looks.

    The whole hierarchy an operator needs is on every event — ``leg`` →
    ``setpoint_index`` → ``phase`` → ``round_index`` → ``channel`` — because a
    renderer that has to remember state across events cannot recover from a
    dropped one, and a nine-hour run drops events the moment a pipe closes.

    ``fraction`` is *this run's own* view of how far along it is, not a
    projection: a consumer reconciles the two (see the CLI's ``reconciled_eta_s``).
    It counts completed setpoints plus completed rounds inside the current one and
    ignores the approach, so it is a **lower bound** — which makes any ETA derived
    from it err long, the safe direction for an unattended run.
    """

    kind: str
    leg: str = ""
    setpoint_index: int = -1
    n_setpoints: int = 0
    setpoints_done: int = 0
    temperature_C: float = float("nan")
    rh_setpoint_pct: float = float("nan")
    #: The third rung of the hierarchy: ``"approach"``, ``"hold"``, or a round kind.
    phase: str = ""
    axis: str = ""
    pv: float = float("nan")
    target: float = float("nan")
    round_index: int = -1
    n_rounds: int = 0
    round_kind: str = ""
    channel: int = 0
    #: Wall-clock the round actually took, and that divided by the channel count.
    #: ``estimate_eis_duration`` models the frequency sweep, so these are the only
    #: numbers in the system that include mux switching, script upload, data
    #: retrieval and the file write as well.
    round_duration_s: float = float("nan")
    per_channel_s: float = float("nan")
    #: :data:`VERDICT_MET` / :data:`VERDICT_UNMET` / :data:`VERDICT_ABORTED`, or "".
    verdict: str = ""
    fraction: float = 0.0
    elapsed_s: float = 0.0
    #: Wall clock, because ``elapsed_s`` runs on the injected monotonic clock and
    #: an operator reading a log the next morning needs to know *when*.
    wall_clock: str = ""
    detail: str = ""
    #: The five values :func:`~softae.core.conditions_capture.read_environment`
    #: returns — temp SP, chamber PV, stage PV, RH SP, RH PV — so the progress
    #: stream doubles as a headless controls monitor and nobody has to open the
    #: GUI (and contend for the rig lock) to see whether control is working.
    #: ``None`` for any value that could not be read: a stale or NaN PV must reach
    #: a renderer as *absent*, never as 0.0 and never as the last good number.
    env: dict[str, Any] = field(default_factory=dict)
    #: :data:`ENV_ABSENT` / :data:`ENV_OK` / :data:`ENV_SKIPPED`.
    env_status: str = ENV_ABSENT

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "leg": self.leg,
                "setpoint_index": self.setpoint_index,
                "n_setpoints": self.n_setpoints, "phase": self.phase,
                "axis": self.axis, "pv": _num(self.pv), "target": _num(self.target),
                "round_index": self.round_index, "round_kind": self.round_kind,
                "channel": self.channel, "verdict": self.verdict,
                "round_duration_s": _num(self.round_duration_s),
                "per_channel_s": _num(self.per_channel_s),
                "fraction": round(self.fraction, 4),
                "elapsed_s": round(self.elapsed_s, 1),
                "wall_clock": self.wall_clock, "detail": self.detail,
                "env_status": self.env_status,
                **{key: _num(self.env.get(key)) for key in ENV_KEYS}}


class EquilibrationAbort(RuntimeError):
    """A condition that is **not** a result — the run stops and restores ambient.

    Carries ``kind`` (:data:`_ABORT_UNREADABLE` or :data:`_ABORT_OVERSHOOT`) so a
    caller can distinguish a dead sensor from a runaway heater without parsing
    the message. Everything else — RH short of setpoint, T short of setpoint,
    excursions inside the fault band — is recorded with ``met=False`` and the run
    continues.
    """

    def __init__(self, message: str, *, kind: str, axis: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.axis = axis


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class EquilibrationConfig:
    """Everything decided before the chamber is touched.

    ``ArrheniusSweepConfig`` is deliberately **not** reused: its ``sweep_order``
    describes a three-axis nest this run does not have, and
    ``resolved_temperatures()`` is monotonic only in step mode.
    """

    channels: list[int] = field(default_factory=lambda: list(range(1, 17)))
    temperatures_C: list[float] = field(
        default_factory=lambda: [27.5, 45.0, 65.0, 85.0])
    legs: tuple[str, ...] = LEGS
    #: See :data:`DEFAULT_RH_SETPOINT_PCT` — 20 %, because the flush basin
    #: humidifies the enclosure as it warms and 15 % is unreachable hot.
    rh_setpoint_pct: float = DEFAULT_RH_SETPOINT_PCT
    #: **A CEILING, not a count.** Every setpoint runs *at most* this many σ(t)
    #: rounds and stops earlier the moment σ has settled (see ``settle_*`` below
    #: and :class:`~softae.analysis.equilibration.SettleTracker`). This changed
    #: meaning: before, ``--rounds 15`` meant fifteen rounds everywhere, and the
    #: 2026-08-11 run spent 45 minutes at each of seven setpoints whose σ was
    #: already flat to within the noise. Set ``settle_enabled = False`` to get the
    #: old fixed-count behaviour back exactly.
    #:
    #: Wherever a τ is wanted it is a ceiling **over** :data:`MIN_POINTS_FOR_TAU`,
    #: never under it: those setpoints could produce no τ at all, so
    #: :meth:`validate` refuses rather than execute a night that cannot be
    #: analysed. With ``tau_setpoints = 0`` there is no τ to preserve and the
    #: ceiling may go as low as 1 — a settle-only sweep is a legitimate design.
    rounds_per_setpoint: int = 15
    #: Budgeted wall-clock per σ(t) round: the modelled sweep, plus the mux
    #: switching and file writes the model does not carry, plus room for the
    #: model's own ~8 % residual. Sets the shortest resolvable τ (≈ 2 × this), which
    #: is why :func:`default_round_period_s` is derived and not picked.
    round_period_s: float = DEFAULT_ROUND_PERIOD_S
    #: Stop a setpoint when σ settles, instead of always running to the ceiling.
    settle_enabled: bool = True
    #: Relative half-width of the settle band. Must exceed the run's own measured
    #: noise floor or nothing can ever satisfy it — the run says so once per
    #: setpoint and lets the ceiling stand (:meth:`SettleTracker.endorsement`).
    settle_tol_rel: float = DEFAULT_SETTLE_TOL_REL
    #: Consecutive rounds that must all sit inside the band. A *detection window*,
    #: which is a different question from how many points a τ needs: configuring
    #: it below :data:`MIN_POINTS_FOR_TAU` narrows the window as asked and does
    #: **not** widen it behind the operator — inside the :attr:`tau_setpoints`
    #: window the round floor still guarantees :data:`MIN_POINTS_FOR_TAU` rounds,
    #: and past it this is the floor. See :func:`settle_floor_rounds`.
    settle_n_rounds: int = DEFAULT_SETTLE_N_ROUNDS
    #: Fewest channels that must carry usable evidence for the criterion to be
    #: *evaluable at all*. Below it the setpoint runs to its ceiling and records
    #: ``not_evaluable`` — never "settled", which is what a board of railed
    #: channels would otherwise report on round three.
    settle_min_channels: int = DEFAULT_SETTLE_MIN_CHANNELS
    #: Floor on the hold at the **first setpoint of the run** — ~3 τ, with τ =
    #: 425–575 s measured while the films dry from ambient to 15 %RH. The whole
    #: transient lives here; the settle criterion cannot shorten it below this.
    min_hold_first_s: float = DEFAULT_MIN_HOLD_FIRST_S
    #: Floor on every later setpoint, first of a leg or not. The films are already
    #: dry, but the chamber still has to re-establish RH at the new temperature.
    min_hold_s: float = DEFAULT_MIN_HOLD_S
    #: How many setpoints **of the run** are held long enough to fit a τ, counted
    #: from the run's first — not from each leg's. Inside that window the round
    #: floor carries :data:`MIN_POINTS_FOR_TAU`; outside it the floor is the
    #: settle window and the time floor alone. ``0`` disables the τ floor
    #: everywhere; anything ≥ :attr:`n_setpoints` applies it everywhere.
    tau_setpoints: int = DEFAULT_TAU_SETPOINTS
    eis_preset: str = DEFAULT_EIS_PRESET
    eis_model: str = "simpleSalt"
    electrode_geometry: dict[str, float] | None = None
    #: How ``electrode_geometry["t_cm"]`` was obtained, in the analysis layer's own
    #: vocabulary (:data:`~softae.analysis.eis.geometry.THICKNESS_METHODS`). It is
    #: **not** re-spelled here, so the tier this run records and the tier a fit
    #: reports cannot fork into two vocabularies. ``"target"`` is the default
    #: because the shipped case is a hand-computed digital-twin target, not a
    #: measurement — see :meth:`EquilibrationRun.thickness_provenance`.
    thickness_method: str = "target"
    #: See :data:`DEFAULT_TOLERANCE_C` — 2.0 °C, wide enough that a chamber
    #: wandering a few tenths is not graded as a failure to hold.
    tolerance_C: float = DEFAULT_TOLERANCE_C
    rh_tolerance_pct: float = DEFAULT_RH_TOLERANCE_PCT
    #: Ascending leg only. The descending one has its own, because cooling here is
    #: passive — see :attr:`down_approach_timeout_s` and
    #: :meth:`temperature_approach_timeout_s`.
    approach_timeout_s: float = DEFAULT_APPROACH_TIMEOUT_S
    #: See :data:`DEFAULT_DOWN_APPROACH_TIMEOUT_S`. Applies to the temperature axis
    #: on the ``"down"`` leg and to nothing else: the humidifier is actively driven
    #: in both directions and is not what runs out of time here.
    down_approach_timeout_s: float = DEFAULT_DOWN_APPROACH_TIMEOUT_S
    #: **Not** the 120 s ``AsyncRHController.wait`` default: holding a low %RH from
    #: 27.5 → 85 °C moves the absolute water content by ~9.6×, so RH must be
    #: re-established at every temperature and 120 s would time out routinely.
    rh_approach_timeout_s: float = DEFAULT_RH_APPROACH_TIMEOUT_S
    warn_C: float = DEFAULT_WARN_C
    fault_C: float = DEFAULT_FAULT_C
    rh_warn_pct: float = 5.0
    #: Wide **on purpose**, and it is not a slack tolerance — ``rh_tolerance_pct``
    #: is what decides ``met``. The fault band is only the runaway guard, and the
    #: overshoot abort fires on ``pv > target + fault`` regardless of axis. With a
    #: 15 %RH target and a narrow band, a chamber that simply cannot dry below the
    #: ~40 %RH of the room would abort — converting *the answer to question 2*
    #: into a crash. 50 puts the trip at 65 %RH, which is a humidifier running
    #: away rather than a rig that will not dry down.
    rh_fault_pct: float = 50.0
    grace_s: float = DEFAULT_GRACE_S
    poll_interval_s: float = 30.0
    #: Floor on the interval between the *throttled* progress events — the
    #: approach ticks and the idle heartbeat. Milestones are never throttled.
    progress_interval_s: float = 60.0
    ambient_C: float = 27.5
    temp_instrument: str = "temp_controller"
    rh_instrument: str = "rh_controller"
    eis_instrument: str | None = None

    def validate(self) -> None:
        """Refuse a design that cannot be executed or cannot be analysed."""
        if not self.channels:
            raise ValueError("no channels selected")
        if not self.temperatures_C:
            raise ValueError("no temperatures selected")
        bad_legs = [leg for leg in self.legs if leg not in LEGS]
        if bad_legs or not self.legs:
            raise ValueError(f"legs must be drawn from {LEGS}; got {self.legs!r}")
        if self.rounds_per_setpoint < 1:
            raise ValueError(
                f"rounds_per_setpoint {self.rounds_per_setpoint} would acquire "
                f"nothing at any setpoint")
        # Conditional on a tau being wanted. The guard refuses a ceiling too low
        # for the fitter, and that is only an unsatisfiable design while the run
        # is trying to fit something: `--tau-setpoints 0` says no setpoint carries
        # the tau floor, and refusing `--tau-setpoints 0 --rounds 4` was refusing
        # to preserve a tau nobody asked for. A settle-only sweep -- "how long
        # does this chamber take to stop moving" -- is a legitimate design and
        # `tau_setpoints` is how it is spelled.
        if self.tau_setpoints > 0 and self.rounds_per_setpoint < MIN_POINTS_FOR_TAU:
            raise ValueError(
                f"rounds_per_setpoint {self.rounds_per_setpoint} is below "
                f"MIN_POINTS_FOR_TAU {MIN_POINTS_FOR_TAU}: sigma(t) = sigma_inf + "
                f"(sigma_0 - sigma_inf)*exp(-t/tau) has three free parameters, so "
                f"fit_equilibration refuses any series shorter than "
                f"{MIN_POINTS_FOR_TAU} points. A CEILING below that is an "
                f"unsatisfiable design while tau_setpoints is "
                f"{self.tau_setpoints} -- none of those setpoints could ever yield "
                f"a tau. Pass --tau-setpoints 0 if no tau is wanted anywhere.")
        self._validate_settle()
        if not 0.0 <= self.rh_setpoint_pct <= 100.0:
            raise ValueError(f"rh_setpoint_pct {self.rh_setpoint_pct} is not a %RH")
        if self.tolerance_C <= 0 or self.rh_tolerance_pct <= 0:
            raise ValueError("tolerances must be positive")
        if self.thickness_method not in THICKNESS_METHODS:
            raise ValueError(
                f"thickness_method '{self.thickness_method}' is not one of "
                f"{list(THICKNESS_METHODS)}; the analysis layer's vocabulary is "
                f"used verbatim so the two cannot fork")
        self._validate_geometry()

    def _validate_settle(self) -> None:
        """Refuse a settle criterion that could only ever answer wrongly.

        A one-round window is not a window (any single value is within tolerance
        of itself), a zero or negative tolerance can never be met, and a
        ``min_channels`` of zero would let an empty board declare equilibrium —
        the exact failure the participation rule exists to prevent.
        """
        if self.settle_n_rounds < 2:
            raise ValueError(
                "settle_n_rounds must be >= 2; a one-round window is always "
                "within tolerance of itself and would settle every setpoint on "
                "its first round")
        if self.settle_tol_rel <= 0:
            raise ValueError("settle_tol_rel must be positive")
        if self.settle_min_channels < 1:
            raise ValueError(
                "settle_min_channels must be >= 1; zero participating channels "
                "is an absence of evidence, never a settled setpoint")
        if self.min_hold_first_s < 0 or self.min_hold_s < 0:
            raise ValueError("hold floors must be >= 0")
        if self.tau_setpoints < 0:
            raise ValueError(
                "tau_setpoints must be >= 0; it counts setpoints of the run that "
                "carry the MIN_POINTS_FOR_TAU round floor, and a negative count "
                "states nothing. 0 is the way to remove the floor everywhere")

    def _validate_geometry(self) -> None:
        """``electrode_geometry`` is all three terms or nothing at all.

        A **partial** dict is not a smaller geometry: ``build_round_workflow``
        copies each term into the EIS step params, so a missing ``t_cm`` reaches
        ``router.handle`` as ``None`` rather than being dropped, and every σ in
        the run is NULL. A **non-positive** term is worse still — it is a stated,
        wrong value that σ = L/(R·t·w) divides by, and a truthiness test cannot
        tell ``t_cm = 0`` from ``t_cm`` never having been given.

        Refused here rather than only in the CLI because the config *is* the
        contract: a GUI tab or a script that builds one directly gets the same
        answer as an operator at the command line.
        """
        geom = self.electrode_geometry
        if geom is None:
            return
        missing = [term for term in GEOMETRY_TERMS if geom.get(term) is None]
        if missing:
            supplied = [f"{k}={geom[k]!r}" for k in GEOMETRY_TERMS
                        if geom.get(k) is not None]
            raise ValueError(
                f"electrode_geometry has {', '.join(supplied) or 'no terms'} but is "
                f"MISSING {', '.join(missing)}; sigma = L/(R*t*w) needs all three, "
                f"and a partial dict reaches the EIS step params as None rather "
                f"than being dropped")
        bad = [f"{k}={geom[k]!r}" for k in GEOMETRY_TERMS if float(geom[k]) <= 0]
        if bad:
            raise ValueError(
                f"electrode_geometry has non-positive term(s) {', '.join(bad)}; "
                f"that is a stated wrong value, not an absent one, and sigma "
                f"divides by t and w")

    def temperature_approach_timeout_s(self, leg: str) -> float:
        """How long the temperature axis may take to reach band, on *leg*.

        The two directions are not symmetric and one number cannot describe both.
        Going up, the heater drives the stage and 1800 s is generous. Coming down
        nothing drives it: the approach is passive and asymptotic, and the
        2026-08-11 run hit the 1800 s timeout at the last down-leg setpoint
        without ever reaching tolerance — so fifteen "isothermal" rounds were
        taken across a 5 °C ramp.

        A ``leg`` this run does not recognise takes the ascending allowance:
        :meth:`validate` has already refused any such leg, so this is unreachable
        rather than a policy.
        """
        return (float(self.down_approach_timeout_s) if leg == "down"
                else float(self.approach_timeout_s))

    def leg_temperatures(self, leg: str) -> list[float]:
        """The setpoint order for *leg* — ``"down"`` retraces the up leg."""
        temps = [float(t) for t in self.temperatures_C]
        return temps if leg == "up" else list(reversed(temps))

    @property
    def peak_temperature_C(self) -> float:
        return max(float(t) for t in self.temperatures_C)

    @property
    def n_setpoints(self) -> int:
        return len(self.legs) * len(self.temperatures_C)


# ── Outcomes ─────────────────────────────────────────────────────────────────

@dataclass
class ApproachOutcome:
    """Result of driving one axis into its band. ``reached`` is derived from the
    recorded series, **never** from ``HoldReport.aborted``.

    That flag is inverted here: ``should_abort`` firing on band entry returns
    ``aborted=True`` meaning *reached*, while a genuine timeout returns
    ``aborted=False``. Read backwards, every timed-out approach reads as a
    success — so the flag never leaves :func:`approach_setpoint`.
    """

    axis: str
    target: float
    reached: bool
    elapsed_s: float
    pv_final: float
    n_samples: int
    series: list[tuple[float, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "target": self.target, "reached": self.reached,
                "elapsed_s": round(self.elapsed_s, 3), "pv_final": _num(self.pv_final),
                "n_samples": self.n_samples}


@dataclass
class HoldOutcome:
    """What the chamber actually did during one watched window.

    ``met=False`` is **data**, not an error: it is the answer to "can the rig hold
    this setpoint here?". The summary carries it per axis per setpoint so a
    downstream reader can never mistake a measured-off-target point for a
    measured-on-target one — which is exactly what the σ(T) data on disk cannot
    do today.
    """

    axis: str
    target: float
    met: bool
    pv_min: float | None = None
    pv_max: float | None = None
    pv_final: float = float("nan")
    n_samples: int = 0
    n_warn: int = 0
    held_s: float = 0.0
    safety_message: str = ""
    series: list[tuple[float, float]] = field(default_factory=list)

    @property
    def units(self) -> str:
        return "%RH" if self.axis == "humidity" else "C"

    def describe(self) -> str:
        """The operator-facing sentence, **rendered by this module**.

        ``monitored_hold``'s warn and fault strings are written for an anneal and
        are degrees-Celsius throughout; handing one to an operator to explain a
        humidity excursion would mislead. This module owns both text sites anyway
        — its own ``on_warn``, its own ``except SafetyError`` — so saying it
        correctly costs no driver edit.
        """
        span = (f"{self.pv_min:g}..{self.pv_max:g}{self.units}"
                if self.pv_min is not None else "no readings")
        if self.met:
            return (f"{self.axis} held {self.target:g}{self.units} for "
                    f"{self.held_s:.0f}s ({span}, {self.n_warn} excursions)")
        return (f"{self.axis} did NOT hold {self.target:g}{self.units} over "
                f"{self.held_s:.0f}s: {span}, {self.n_warn} excursion(s). "
                f"Recorded as a result; the run continues.")

    def as_dict(self) -> dict[str, Any]:
        return {"axis": self.axis, "target": self.target, "met": self.met,
                "pv_min": _num(self.pv_min), "pv_max": _num(self.pv_max),
                "pv_final": _num(self.pv_final), "n_samples": self.n_samples,
                "n_warn": self.n_warn, "held_s": round(self.held_s, 3),
                "detail": self.describe(), "safety_message": self.safety_message}

    @classmethod
    def merge(cls, outcomes: Sequence["HoldOutcome"]) -> "HoldOutcome":
        """Collapse the per-round windows of one axis into a per-setpoint verdict.

        ``met`` is conjunctive: one round out of band means the setpoint was not
        held for the series, which is the claim the analysis needs to be able to
        trust.
        """
        if not outcomes:
            return cls(axis="", target=float("nan"), met=False)
        mins = [o.pv_min for o in outcomes if o.pv_min is not None]
        maxs = [o.pv_max for o in outcomes if o.pv_max is not None]
        return cls(
            axis=outcomes[0].axis,
            target=outcomes[0].target,
            met=all(o.met for o in outcomes),
            pv_min=min(mins) if mins else None,
            pv_max=max(maxs) if maxs else None,
            pv_final=outcomes[-1].pv_final,
            n_samples=sum(o.n_samples for o in outcomes),
            n_warn=sum(o.n_warn for o in outcomes),
            held_s=sum(o.held_s for o in outcomes),
            safety_message="; ".join(o.safety_message for o in outcomes
                                     if o.safety_message),
        )


@dataclass
class SeriesOutcome:
    """Why one setpoint's σ(t) series ended, and on what evidence.

    A **first-class result**, recorded in the sidecar and announced on the
    console, because ``rounds_per_setpoint`` is now a ceiling and a four-round
    setpoint is otherwise ambiguous in the worst possible way: it looks identical
    whether σ settled, whether the criterion said no until the rounds ran out, or
    whether nothing on the board could carry the criterion at all. Those are a
    result, a result, and a fault, and only the last one wants a human.
    """

    outcome: str
    n_rounds: int
    ceiling: int
    floor_s: float
    held_s: float
    participating: list[int] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)
    max_deviation_rel: float | None = None
    #: ``None`` when it could not be judged — never ``True`` by default.
    tolerance_achievable: bool | None = None
    endorsement: str = ""
    noise_floor_rel: float | None = None
    temp_holds: list[HoldOutcome] = field(default_factory=list)
    rh_holds: list[HoldOutcome] = field(default_factory=list)

    @property
    def settled(self) -> bool:
        return self.outcome == SETTLE_SETTLED

    def describe(self) -> str:
        """One line, and it must distinguish a short setpoint that settled from a
        short setpoint that gave up."""
        head = (f"{self.n_rounds}/{self.ceiling} rounds in {self.held_s:.0f}s "
                f"(floor {self.floor_s:.0f}s)")
        who = (f"ch {','.join(str(c) for c in self.participating)}"
               if self.participating else "NO participating channel")
        if self.outcome == SETTLE_SETTLED:
            return f"SETTLED after {head}; {who}"
        if self.outcome == SETTLE_DISABLED:
            return f"settle criterion OFF: {head}"
        if self.outcome == SETTLE_NOT_EVALUABLE:
            excluded = ", ".join(f"ch{c}={why}" for c, why in
                                 sorted(self.excluded.items()))
            return (f"NOT EVALUABLE -- ran to the ceiling: {head}; too few usable "
                    f"channels ({excluded or 'none reported'})")
        return f"CEILING reached without settling: {head}; {who}. {self.endorsement}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "settle_outcome": self.outcome,
            "settle_rounds_run": self.n_rounds,
            "settle_rounds_ceiling": self.ceiling,
            "settle_participating": list(self.participating),
            "settle_excluded": {str(k): v for k, v in sorted(self.excluded.items())},
            "settle_floor_s": round(float(self.floor_s), 3),
            "settle_held_s": round(float(self.held_s), 3),
            "settle_max_deviation_rel": _num(self.max_deviation_rel),
            "settle_noise_floor_rel": _num(self.noise_floor_rel),
            "settle_tolerance_achievable": self.tolerance_achievable,
            "settle_detail": self.describe(),
        }


def _num(value: Any) -> Any:
    """JSON-safe number: ``NaN`` becomes ``null`` rather than invalid JSON."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


# ── The watched primitives (both axes, one implementation) ───────────────────

def _observe(hook: Any, stamp: float, value: float) -> None:
    """Hand a sample to a progress observer, and **never** let it break the poll.

    This runs inside the PV read of a watched hold. An exception escaping here
    would propagate into ``monitored_hold`` as an unreadable PV and, one layer
    up, abort a nine-hour run as a dead sensor — a formatting bug indistinguishable
    from a hardware failure. Silence is the only acceptable degradation.
    """
    if hook is None:
        return
    try:
        hook(stamp, value)
    except Exception:
        pass


def _recording_reader(read_pv: Any, series: list[tuple[float, float]], now: Any,
                      on_sample: Any = None):
    """Wrap a PV reader so **every** sample is kept by this module.

    ``monitored_hold`` builds its ``HoldReport`` on the return path and discards
    it entirely when it raises — the ``SafetyError`` carries ``instrument`` /
    ``requested`` / ``limit`` and puts ``held_s`` and ``pv`` in message text only.
    The most informative failures would therefore lose their data, which is
    unacceptable when an unmet setpoint is the primary result. Recording here
    costs nothing and the full PV trace is wanted anyway: it *is* the evidence.

    ``on_sample(t, pv)`` is the progress tap. Every poll of both axes passes
    through here, which makes this the one place a heartbeat can fire while the
    chamber is doing nothing an operator can see.
    """

    def _read() -> float:
        try:
            value = float(read_pv())
        except Exception:
            # Recorded as an attempted-and-failed sample, then re-raised so
            # ``monitored_hold``'s own "cannot verify the hold" path still runs.
            # Without the record, "the sensor is dead" and "the hold was never
            # entered" would be the same empty series.
            stamp = float(now())
            series.append((stamp, float("nan")))
            _observe(on_sample, stamp, float("nan"))
            raise
        stamp = float(now())
        series.append((stamp, value))
        _observe(on_sample, stamp, value)
        return value

    return _read


def _finite(series: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(t, v) for t, v in series if isinstance(v, float) and math.isfinite(v)]


def _sustained_overshoot(
    series: Sequence[tuple[float, float]], target: float, fault: float, grace_s: float
) -> bool:
    """True when the trailing run of samples above ``target + fault`` spans *grace_s*.

    ``monitored_hold`` grades ``abs(pv − target)``, so a stage that cannot *reach*
    85 °C and a heater that *runs past* it produce the same ``SafetyError``. The
    first is data; the second is a hazard. The sign is checked here, from this
    run's own series.
    """
    above: list[tuple[float, float]] = []
    for t, v in reversed(list(series)):
        if isinstance(v, float) and math.isfinite(v) and v > target + fault:
            above.append((t, v))
        else:
            break
    if len(above) < 2:
        return False
    return (above[0][0] - above[-1][0]) >= float(grace_s)


def approach_setpoint(
    read_pv: Any,
    target: float,
    *,
    axis: str,
    instrument: str,
    tolerance: float,
    timeout_s: float,
    poll_interval_s: float = 30.0,
    should_abort: Any = None,
    sleep: Any = None,
    now: Any = None,
    on_sample: Any = None,
) -> ApproachOutcome:
    """Drive one axis into its band — a ``monitored_hold`` with a non-binding fault.

    Calling the watched hold immediately after ``write_sp`` would grade a
    legitimate ramp as a fault and abort after ``grace_s``. The guard needs no new
    code: run the approach as a hold of ``timeout_s`` with an unreachable fault
    band and a ``should_abort`` that fires once the PV enters the band. That is a
    clean early exit which **returns** rather than raising, and hands back
    ``held_s`` (= approach time) and the sample count for free.

    No fourth polling loop is written. Two already exist and disagree.
    """
    _now = now or time.monotonic
    series: list[tuple[float, float]] = []

    def _in_band_or_stop() -> bool:
        if should_abort is not None and should_abort():
            return True
        if not series:
            return False
        pv = series[-1][1]
        return math.isfinite(pv) and abs(pv - float(target)) <= float(tolerance)

    report = monitored_hold(
        timeout_s,
        read_pv=_recording_reader(read_pv, series, _now, on_sample),
        target_C=float(target),
        instrument=instrument,
        warn_C=float("inf"),
        fault_C=float("inf"),
        grace_s=float(timeout_s) + 1.0,
        poll_interval_s=poll_interval_s,
        should_abort=_in_band_or_stop,
        sleep=sleep,
        now=_now,
    )
    # `report.aborted` is deliberately not consulted: True means *reached* here.
    finite = _finite(series)
    pv_final = finite[-1][1] if finite else float("nan")
    reached = bool(finite) and abs(pv_final - float(target)) <= float(tolerance)
    outcome = ApproachOutcome(
        axis=axis, target=float(target), reached=reached, elapsed_s=report.held_s,
        pv_final=pv_final, n_samples=len(series), series=list(series),
    )
    logger.info("equilibration_approach", axis=axis, target=target, reached=reached,
                elapsed_s=round(outcome.elapsed_s, 1), pv_final=_num(pv_final))
    return outcome


def watch_hold(
    read_pv: Any,
    target: float,
    *,
    hold_time_s: float,
    axis: str,
    instrument: str,
    tolerance: float,
    warn: float,
    fault: float,
    grace_s: float = 120.0,
    poll_interval_s: float = 30.0,
    should_abort: Any = None,
    sleep: Any = None,
    now: Any = None,
    on_sample: Any = None,
) -> HoldOutcome:
    """Hold one axis and **record what happened**, met or not.

    The same ``monitored_hold`` the anneal path uses, with the opposite
    consequence: its ``SafetyError`` is caught and turned into
    ``HoldOutcome(met=False)``, and the run continues. Two guards keep that from
    being the dangerous ``except SafetyError: continue``:

    * zero finite samples in the window → :class:`EquilibrationAbort`
      (``unreadable_pv``); "we cannot verify" is not an answer to "can it hold?".
    * a sustained overshoot past ``target + fault`` → :class:`EquilibrationAbort`
      (``sustained_overshoot``), regardless of axis.

    ``met`` is computed from the recorded extremes, not from whether the call
    returned: the setpoint was held only if every finite sample stayed inside
    ``tolerance``.
    """
    _now = now or time.monotonic
    series: list[tuple[float, float]] = []
    n_warn_seen = 0
    safety_message = ""

    def _on_warn(pv: float, deviation: float, _message: str) -> None:
        nonlocal n_warn_seen
        n_warn_seen += 1
        logger.warning("equilibration_excursion", axis=axis, instrument=instrument,
                       pv=pv, target=target, deviation=round(deviation, 2))

    try:
        report = monitored_hold(
            hold_time_s,
            read_pv=_recording_reader(read_pv, series, _now, on_sample),
            target_C=float(target),
            instrument=instrument,
            warn_C=float(warn),
            fault_C=float(fault),
            grace_s=float(grace_s),
            poll_interval_s=poll_interval_s,
            on_warn=_on_warn,
            should_abort=should_abort,
            sleep=sleep,
            now=_now,
        )
        held_s, n_samples = report.held_s, report.n_samples
    except SafetyError as exc:
        # The report is lost on this path, which is why the reader records.
        safety_message = str(exc)
        held_s = (series[-1][0] - series[0][0]) if len(series) > 1 else 0.0
        n_samples = len(_finite(series))

    finite = _finite(series)
    if not finite:
        raise EquilibrationAbort(
            f"{axis} PV was unreadable for the whole {hold_time_s:.0f}s window "
            f"({len(series)} attempted samples). That is a dead sensor, not a "
            f"measurement of what the rig can hold — continuing would run the rig "
            f"blind.",
            kind=_ABORT_UNREADABLE, axis=axis,
        )
    if _sustained_overshoot(series, float(target), float(fault), grace_s):
        raise EquilibrationAbort(
            f"{axis} PV stayed above {float(target) + float(fault):.1f} for more than "
            f"{float(grace_s):.0f}s (peak {max(v for _, v in finite):.1f}). An "
            f"overshoot is not an undershoot: this is a runaway, not a limit.",
            kind=_ABORT_OVERSHOOT, axis=axis,
        )

    values = [v for _, v in finite]
    pv_min, pv_max = min(values), max(values)
    met = (safety_message == ""
           and pv_min >= float(target) - float(tolerance)
           and pv_max <= float(target) + float(tolerance))
    outcome = HoldOutcome(
        axis=axis, target=float(target), met=met, pv_min=pv_min, pv_max=pv_max,
        pv_final=values[-1], n_samples=n_samples, n_warn=n_warn_seen,
        held_s=held_s, safety_message=safety_message, series=list(series),
    )
    if not met:
        logger.warning("equilibration_setpoint_unmet", axis=axis, target=target,
                       pv_min=pv_min, pv_max=pv_max, detail=safety_message,
                       note="recorded as a result; the run continues")
    return outcome


# ── Step naming and the per-round DAG ────────────────────────────────────────

def measurement_step_name(
    channel: int, leg: str, setpoint_index: int, round_index: int,
    *, kind: str = KIND_SERIES,
) -> str:
    """The full coordinate, in the name — ``eq_ch3_Lup_S2_R7``.

    Three layers collide on a repeated point and only one of them is loud:
    ``workflow_executor._build_dag`` **refuses** duplicate step names; but
    ``router.py``'s ``file_stem = step.name`` would silently overwrite the earlier
    ``.txt``, and ``temp_eis_sweep._capture`` resolves ``t_idx`` by
    first-match-with-default-0, silently attributing an unmatched temperature to
    the first setpoint. The last two become live the moment someone makes names
    unique to get past the first without also fixing the keying.

    So the leg and the setpoint **index** are in the name — not the temperature
    value, because ``round(27.5)`` is 28 and the value was never the identity —
    and this module keys its own results by the same tuple. It deliberately does
    not reuse or extend ``_EIS_STEP_RE``.
    """
    suffix = "" if kind == KIND_SERIES else f"_{kind}"
    return f"eq_ch{int(channel)}_L{leg}_S{int(setpoint_index)}_R{int(round_index)}{suffix}"


#: Stands in for ``run_id`` when a round is built outside a recorded run (a test,
#: a projection, a caller predating the argument). Drawn once per process, so
#: rounds built here still share a directory with each other and with **nothing
#: else** — never with another process, and never with a recorded run.
_UNATTRIBUTED_RUN_TOKEN = uuid.uuid4().hex[:8]


def _run_token(run_id: str | None) -> str:
    return str(run_id) if run_id else _UNATTRIBUTED_RUN_TOKEN


def mscr_path(channel: int, kind: str = KIND_SERIES,
              run_id: str | None = None) -> str:
    """Per-(run, channel, kind) script path — **isolated, like the output dir**.

    The kind and the channel were already in the name, so two presets and two
    channels could not clobber each other. The *run* was not, and that is the one
    axis with another process on the other end of it: the path was a fixed
    ``<tmp>/softae_eq_ch{N}_{kind}.mscr``, and the test suite writes to exactly
    those paths. On 2026-08-10 a test run rewrote channels 1–4's scripts at 19:31
    while a real run's scripts from 17:50 sat in the same directory. Twenty
    minutes earlier and the rig would have measured a live run's channels with
    the test's sweep parameters, while every log line reported success — the
    ``.mscr`` is uploaded per round, so the swap would have taken effect
    mid-flight and left no trace in the spectra.

    Same reasoning and same fallback as :func:`round_outdir`: a caller with no
    ``run_id`` gets a per-process token rather than the shared name, so an
    unattributed round still collides with nothing.
    """
    token = _run_token(run_id)
    return str(Path(tempfile.gettempdir())
               / f"softae_eq_{token}_ch{int(channel)}_{kind}.mscr")


def round_outdir(run_id: str | None = None) -> str:
    """Where this run's raw EIS files land — **its own directory, not the shared one**.

    ``eis_measure_step`` points every step in the system at one
    ``<tmp>/softae_eis_output``, and ``async_espico.sendscript_getdata`` names the
    file from a timestamp alone. A file left there by an earlier run is therefore
    indistinguishable from one this run wrote, and a stale spectrum picked up here
    would enter σ(t) as a real point at a coordinate it never occupied.
    """
    return str(Path(tempfile.gettempdir()) / f"softae_eq_out_{_run_token(run_id)}")


def build_round_workflow(
    config: EquilibrationConfig,
    *,
    leg: str,
    setpoint_index: int,
    round_index: int,
    kind: str = KIND_SERIES,
    temperature_C: float = float("nan"),
    run_id: str | None = None,
) -> Workflow:
    """One round: measure every channel once, nothing else.

    ``circuit_model`` and the electrode geometry are set explicitly.
    ``eis_measure_step`` sets neither, and ``router.handle`` auto-fits **only**
    when ``circuit_model`` is present and takes L/t/w from params — without both
    there is no ``fit_results`` row, no σ, and therefore no σ(t) series at all.
    """
    geom = config.electrode_geometry or {}
    steps: list[WorkflowStep] = []
    for channel in config.channels:
        step = eis_measure_step(
            channel,
            name=measurement_step_name(channel, leg, setpoint_index, round_index,
                                       kind=kind),
        )
        params: dict[str, Any] = {
            "circuit_model": config.eis_model,
            "mscrpath": mscr_path(channel, kind, run_id),
            # Overridden here rather than in `eis_measure_step`, which is shared by
            # every workflow in the system: see `round_outdir` for what the shared
            # default would let a stale file do to this run's sigma(t).
            "outdir": round_outdir(run_id),
        }
        if geom:
            params["electrode_L_cm"] = geom.get("L_cm")
            params["electrode_t_cm"] = geom.get("t_cm")
            params["electrode_w_cm"] = geom.get("w_cm")
        step = step.with_params(**params)
        if config.eis_instrument:
            step = WorkflowStep(
                name=step.name, instrument=config.eis_instrument, method=step.method,
                params=dict(step.params), timeout_s=step.timeout_s, retry=step.retry,
                tags=dict(step.tags),
            )
        # `role` stays 'sample'. Tagging these `drift_repeat` would enrol the run
        # in the commissioning capability ladder, where it means nothing, and the
        # geometry-series fit depends on there being exactly ONE repeat.
        # `thermal_history` is not written: it is pre-run provenance, and a
        # within-run coordinate in it would be a second vocabulary in one column.
        steps.append(step.with_tags(
            leg=leg, setpoint_index=str(setpoint_index),
            round_index=str(round_index), kind=kind,
        ))
    return Workflow(
        name=f"equilibration_{leg}_S{setpoint_index}_R{round_index}_{kind}",
        description=f"P.22 {kind} round at setpoint {setpoint_index} ({leg} leg)",
        setup=steps,
        metadata={"source": "equilibration", "leg": leg,
                  "setpoint_index": setpoint_index, "round_index": round_index,
                  "kind": kind, "temperature_C": temperature_C,
                  "rh_setpoint_pct": config.rh_setpoint_pct},
    )


# ── Time budget — the run projects itself before it runs ─────────────────────

@dataclass
class DurationProjection:
    """Both columns, and now both **ends**.

    The worst case uses the **timeouts**, not an assumed ramp rate, so it is an
    upper bound rather than a guess. The length is no longer a single number
    either: ``rounds_per_setpoint`` is a ceiling, so how long the run takes
    depends on how fast σ settles, which is data. Every total is therefore a
    floor-to-ceiling pair, and a caller that prints one number is printing the
    wrong one.
    """

    n_setpoints: int
    rounds_per_setpoint: int
    per_setpoint_worst_s: float
    per_setpoint_typical_s: float
    worst_case_s: float
    typical_s: float
    breakdown_worst: dict[str, float] = field(default_factory=dict)
    breakdown_typical: dict[str, float] = field(default_factory=dict)
    #: ``"modelled"`` or ``"measured"`` — which round cost the numbers rest on.
    #: Never collapsed into one figure: the *gap* between the two is the finding.
    basis: str = "modelled"
    series_round_s: float = 0.0
    #: Fewest rounds the settle criterion can stop each setpoint after, in run
    #: order. The **source of truth** for every floor below: the three named
    #: figures are read off it and the floor totals are its sum, so a regime
    #: nobody named cannot be silently dropped from the budget. Equal to the
    #: ceiling throughout when the criterion is disabled, which collapses every
    #: range below.
    floor_rounds: tuple[int, ...] = ()
    #: The run's first setpoint — the only one with ``min_hold_first_s``.
    min_rounds_first: int = 0
    #: A later setpoint still inside the τ window (``tau_setpoints``). Equal to
    #: :attr:`min_rounds_later` when the window does not extend past the first.
    min_rounds_tau: int = 0
    #: A setpoint past the τ window: ``settle_n_rounds`` and the time floor alone.
    min_rounds_later: int = 0
    #: How far the τ floor reaches, echoed so a reader of the projection alone can
    #: tell which of the three figures above applies where.
    tau_setpoints: int = 0
    #: The two temperature-approach allowances, kept apart. ``breakdown_worst``
    #: carries their per-setpoint mean, which is the right term in the totals and
    #: the wrong number to quote at an operator deciding whether to extend one.
    temp_approach_timeout_up_s: float = 0.0
    temp_approach_timeout_down_s: float = 0.0
    #: The same totals at the floor. ``*_floor_s <= *_s`` always. The per-setpoint
    #: pair uses ``min_rounds_later``: it describes a *typical* setpoint, and only
    #: one setpoint in the run is the first.
    per_setpoint_typical_floor_s: float = 0.0
    per_setpoint_worst_floor_s: float = 0.0
    typical_floor_s: float = 0.0
    worst_floor_s: float = 0.0

    @property
    def adaptive(self) -> bool:
        """Is there a range at all, or did the criterion collapse it to a point?"""
        return any(floor != self.rounds_per_setpoint for floor in self.floor_rounds)

    @property
    def min_rounds(self) -> int:
        """The shortest a setpoint anywhere in this run may be."""
        return min(self.floor_rounds) if self.floor_rounds else self.min_rounds_first

    def rounds_span(self) -> str:
        """``"15"`` or ``"3-15"`` — what a setpoint may cost, in rounds."""
        if not self.adaptive:
            return str(self.rounds_per_setpoint)
        return f"{self.min_rounds}-{self.rounds_per_setpoint}"

    def describe(self) -> str:
        if not self.adaptive:
            return (f"{self.n_setpoints} setpoints x {self.rounds_per_setpoint} "
                    f"rounds: typical {self.typical_s / 3600:.1f} h, worst case "
                    f"{self.worst_case_s / 3600:.1f} h ({self.basis})")
        return (f"{self.n_setpoints} setpoints x {self.rounds_span()} rounds "
                f"(ceiling): typical {self.typical_floor_s / 3600:.1f}-"
                f"{self.typical_s / 3600:.1f} h, worst case "
                f"{self.worst_floor_s / 3600:.1f}-{self.worst_case_s / 3600:.1f} h "
                f"({self.basis})")


def eis_round_cost_s(config: EquilibrationConfig, preset: str) -> float:
    """Modelled cost of one all-channel EIS round at *preset*.

    ``estimate_eis_duration`` is correct and is used. ``estimate_workflow_duration``
    is **not**: ``estimate_step_duration`` returns ``0.0`` (not ``None``) for
    ``method == "wait"`` unless the params carry ``duration_s``/``seconds``, and
    temperature waits carry ``within``/``equilibration_time``/``timeout`` — so
    every hold would project as free *and* ``DurationEstimate.is_complete`` would
    still report ``True``. A projection that is confidently wrong is worse than
    one that admits a gap, so the holds are computed from this config instead.
    """
    from softae.core.eis_scripts import EISParams
    from softae.core.preflight import estimate_eis_duration

    return estimate_eis_duration(EISParams.from_preset(preset)) * len(config.channels)


def inter_round_gap_s(config: EquilibrationConfig,
                      measured_round_s: float | None = None) -> float:
    """Watched dead time between two σ(t) rounds, so the round period is honoured.

    The period is honoured **from the round's start**, so the gap is the period
    minus what that round cost. With *measured_round_s* the executor supplies what
    the round really took; without it the modelled cost stands in, for the callers
    that have nothing measured yet (``plan``, the pre-run projection).

    Which cost is subtracted is the whole defect this argument exists to fix.
    Subtracting the modelled cost produces a cycle of
    ``real_cost + (period - modelled_cost)``, which overruns the period by exactly
    however much the model is low. When the defect was found the model ran ~10x
    low, so at 12 channels / 240 s / ``Quick`` the cycle was ~353 s against a
    configured 240 s: σ(t) sampled at an interval nobody asked for, and the fitter
    reading the series as evenly spaced at the *configured* period. The 2026-08
    recalibration shrank that error to ~8 % but did not remove it, and it never
    removes the unmodelled per-channel overhead, so the measured cost is still the
    only correct thing to subtract.

    ``project_duration`` never had that bug — it computes a cycle as the period or
    the round cost, whichever is longer — so the executor was disagreeing with the
    projection printed to the operator. Both now derive the cycle from this one
    function.

    The floor is ``poll_interval_s``, **not zero**: both axes are graded by
    sampling this gap, and a zero gap would leave temperature and humidity
    ungraded for the entire series — precisely the fail-open ``wait()`` behaviour
    this module was written to replace. An overrunning round therefore still pays
    one poll interval, and :meth:`EquilibrationRun._warn_if_round_overruns` tells
    the operator the configured period is unachievable.
    """
    cost = (eis_round_cost_s(config, config.eis_preset)
            if measured_round_s is None else float(measured_round_s))
    return max(config.poll_interval_s, config.round_period_s - cost)


def round_cost_s(config: EquilibrationConfig, *,
                 measured_per_channel_s: float | None = None) -> float:
    """Cost of one all-channel round — **measured whenever the operator has one**.

    One function decides which cost a plan rests on, because the round cost, the
    inter-round gap, the headroom and the whole-run duration all have to rest on
    the *same* one. Mixing a modelled round into a measured projection is wrong in
    a way no single printed figure reveals.

    ``eis_round_cost_s`` models the frequency sweep, and it is fitted to three
    presets timed on this rig rather than assumed — 12 channels on ``Standard``
    measured 40.7 s/channel against ~37.7 s/channel modelled. It ran ~10x low until
    2026-08, which is why this argument exists at all; it is now within ~8 %, so
    the modelled figure is a usable estimate rather than a floor. A caller holding
    a measured per-channel number should still pass it: no fit beats a stopwatch,
    and the fit says nothing about a preset that was never timed.
    """
    if measured_per_channel_s is not None and float(measured_per_channel_s) > 0:
        return float(measured_per_channel_s) * len(config.channels)
    return eis_round_cost_s(config, config.eis_preset)


def minimum_feasible_period_s(round_cost: float) -> float:
    """The shortest ``--round-period-s`` a round of this cost can be sampled at.

    The round cost itself, rounded up to a whole ten seconds so the answer is a
    number an operator can type. Below it the round does not fit inside the period
    at all — and nothing adapts: the executor never shortens a round and never
    resamples σ(t), so the series is simply taken at whatever the round costs plus
    the poll floor, while the fitter still reads it as evenly spaced at the
    configured period.

    Deliberately **not** the same as ``_warn_if_round_overruns``'s suggestion,
    which adds 10 % margin because it is proposing a period for a *re-run*. This
    is the floor, and it is presented as one.
    """
    return math.ceil(float(round_cost) / 10.0) * 10.0


def round_headroom_s_per_channel(config: EquilibrationConfig,
                                 measured_round_s: float | None = None) -> float:
    """Seconds per channel the configured period leaves over the round cost.

    On the modelled cost this is the margin available to absorb two things: the
    per-channel mux switch, script upload, data retrieval and file write that
    ``estimate_eis_duration`` does not model, and the ~8 % residual of the fit
    itself. Since the 2026-08 recalibration the second dominates and it is
    *proportional* to the sweep, which is why the caution built on this number
    (``softae.tools.equilibration.model_underestimate_frac``) is a fraction rather
    than a flat per-channel reserve.

    It remains a *headroom*, not an estimate of the overhead: no per-channel
    overhead constant lives in this module, because that number belongs to one rig
    and would be wrong the moment it was written down. The run measures its own —
    see :meth:`EquilibrationRun.measured_cost_summary`.

    With *measured_round_s* the remainder is no longer margin for anything: the
    overhead is already inside the measurement, and what is left is idle time. The
    argument exists so a plan given a measured cost reports one consistent set of
    numbers rather than a measured duration beside a modelled headroom.
    """
    channels = max(1, len(config.channels))
    cost = (eis_round_cost_s(config, config.eis_preset)
            if measured_round_s is None else float(measured_round_s))
    return (float(config.round_period_s) - cost) / channels


def tau_floor_rounds(config: EquilibrationConfig, setpoint_ordinal: int) -> int:
    """:data:`MIN_POINTS_FOR_TAU` where a τ is wanted, and ``1`` where it is not.

    *setpoint_ordinal* counts setpoints of the **run**, not of the leg: the down
    leg revisits temperatures the films have already dried at, so its first
    setpoint is not a fresh transient and must not be treated as one.

    Returning ``1`` rather than ``0`` outside the window keeps this a real floor
    on rounds — a setpoint always runs at least once — so callers can take a plain
    ``max`` over the three bounds without a special case for "no τ here".
    """
    if int(setpoint_ordinal) < int(config.tau_setpoints):
        return MIN_POINTS_FOR_TAU
    return 1


def hold_floor_s(config: EquilibrationConfig, setpoint_ordinal: int) -> float:
    """The time floor at this setpoint: the first of the run gets its own."""
    return float(config.min_hold_first_s if int(setpoint_ordinal) == 0
                 else config.min_hold_s)


def settle_floor_rounds(config: EquilibrationConfig, series_round_s: float, *,
                        setpoint_ordinal: int) -> int:
    """Fewest rounds a setpoint can stop after, given the criterion and the floors.

    Three things bound it from below: the criterion needs ``settle_n_rounds``
    rounds before it has a window at all; the hold floor (``min_hold_first_s`` at
    the run's first setpoint, ``min_hold_s`` after) must have elapsed; and, **for
    the first ``tau_setpoints`` setpoints only**, the series must be long enough
    for the offline fitter to accept it — :data:`MIN_POINTS_FOR_TAU`, imported
    rather than restated so the acquisition side cannot drift away from the
    analysis side that will refuse its output.

    That third bound is not a nicety where it applies. At the shipped
    ``round_period_s`` of 660 s the time floors buy 3 rounds at the first setpoint
    and 1 after, so without it a run could stop the transient short of the fit
    minimum and end with no τ at all. But it is confined to the setpoints that
    have a transient to fit: see :func:`tau_floor_rounds` and the module docstring
    for the measured swing that decides where that is.

    The ceiling bounds it from above, and :meth:`EquilibrationConfig.validate`
    guarantees the ceiling is itself at least :data:`MIN_POINTS_FOR_TAU`, so the
    clamp can never reintroduce an unanalysable setpoint inside the τ window. With
    the criterion disabled the answer is the ceiling, which is what makes every
    projected range collapse to the old single number rather than needing a second
    code path.
    """
    ceiling = int(config.rounds_per_setpoint)
    if not config.settle_enabled:
        return ceiling
    floor_s = hold_floor_s(config, setpoint_ordinal)
    by_time = (math.ceil(floor_s / series_round_s) if series_round_s > 0 else 1)
    return max(1, min(ceiling, max(int(config.settle_n_rounds),
                                   tau_floor_rounds(config, setpoint_ordinal),
                                   int(by_time))))


def project_duration(
    config: EquilibrationConfig,
    *,
    measured_series_round_s: float | None = None,
) -> DurationProjection:
    """Upper and typical bounds for the whole run.

    From the config alone by default. Once a run has *measured* what a round
    actually costs, that value is passed in and the projection is redone on it —
    but the modelled projection is never overwritten in place. The gap between the
    two is a finding about ``estimate_eis_duration``, not an error to correct
    quietly: everything downstream (``preflight.project_campaign`` included)
    inherits that model, so a run whose measured round drifts from it is the first
    place the drift can be seen. It once stood at ~10x and was refitted from the
    bench; keeping both numbers is how the next such gap gets caught earlier.
    """
    series_cost = eis_round_cost_s(config, config.eis_preset)
    basis = "modelled"
    if measured_series_round_s is not None and float(measured_series_round_s) > 0:
        series_cost, basis = float(measured_series_round_s), "measured"

    # The cycle the executor actually performs: the round, then the gap
    # `inter_round_gap_s` hands it for that same cost. Derived from that one
    # function rather than restated as `max(period, cost)` so the projection
    # cannot drift away from the loop again — including the poll-interval floor,
    # which an overrunning round really does spend and which a bare `max` would
    # drop (one hour, over a shipped 8-setpoint x 15-round run).
    series_round_s = series_cost + inter_round_gap_s(config, series_cost)
    series_s = config.rounds_per_setpoint * series_round_s
    n = config.n_setpoints

    worst = {"temperature_approach": _mean_temp_approach_s(config),
             "rh_approach": config.rh_approach_timeout_s,
             "sigma_series": series_s}
    typical = {k: (v if k == "sigma_series" else v * TYPICAL_APPROACH_FRACTION)
               for k, v in worst.items()}
    per_worst, per_typical = sum(worst.values()), sum(typical.values())

    # The floor. Only the series shortens — the approaches are driven by the
    # chamber and the settle criterion has no opinion about them — so the fixed
    # part is carried across unchanged and only `sigma_series` is replaced. Every
    # setpoint is asked for its own floor rather than two being extrapolated: the
    # first has `min_hold_first_s`, the next `tau_setpoints - 1` still carry the
    # τ floor, and the rest carry neither.
    floors = tuple(settle_floor_rounds(config, series_round_s, setpoint_ordinal=i)
                   for i in range(n))
    fixed_worst = per_worst - series_s
    fixed_typical = per_typical - series_s
    # `min_rounds_later` describes the regime MOST setpoints are in, which is what
    # the per-setpoint floor row should quote; the last setpoint is always in it.
    later = floors[-1] if floors else 0
    return DurationProjection(
        n_setpoints=n, rounds_per_setpoint=config.rounds_per_setpoint,
        per_setpoint_worst_s=per_worst, per_setpoint_typical_s=per_typical,
        worst_case_s=per_worst * n, typical_s=per_typical * n,
        breakdown_worst=worst, breakdown_typical=typical, basis=basis,
        series_round_s=series_round_s,
        floor_rounds=floors,
        min_rounds_first=floors[0] if floors else 0,
        min_rounds_tau=floors[1] if len(floors) > 1 else (floors[0] if floors else 0),
        min_rounds_later=later,
        tau_setpoints=int(config.tau_setpoints),
        temp_approach_timeout_up_s=config.temperature_approach_timeout_s("up"),
        temp_approach_timeout_down_s=config.temperature_approach_timeout_s("down"),
        per_setpoint_typical_floor_s=fixed_typical + later * series_round_s,
        per_setpoint_worst_floor_s=fixed_worst + later * series_round_s,
        typical_floor_s=fixed_typical * n + sum(floors) * series_round_s,
        worst_floor_s=fixed_worst * n + sum(floors) * series_round_s,
    )


def _mean_temp_approach_s(config: EquilibrationConfig) -> float:
    """Per-setpoint temperature-approach allowance, averaged over the legs.

    The two legs no longer share a timeout, so no single per-setpoint figure is
    *the* allowance. The mean is the one that keeps the arithmetic honest —
    ``per_setpoint * n_setpoints`` still equals the run total — and the two real
    numbers are carried separately on the projection for anything that quotes an
    allowance to an operator rather than summing it.
    """
    n = max(1, config.n_setpoints)
    total = sum(config.temperature_approach_timeout_s(leg)
                for leg in config.legs for _temp in config.temperatures_C)
    return total / n


# ── The run ──────────────────────────────────────────────────────────────────

class EquilibrationRun:
    """Execute the characterization run and write its sidecar.

    Persistence is **Stage 1**: the σ(t) series itself needs no schema change (it
    is a join over ``measurements × fit_results × conditions``, all already
    written by the router), and the two things that are genuinely not recoverable
    — the coordinate and the hold facts — go to a JSON sidecar at
    ``db/runs/<run_id>/equilibration.json``, exactly as
    ``ArrheniusSweep._write_json_sidecar`` already does. ``core/data_store.py`` is
    not opened. Stage 2 adds two tables after a posted-and-acked claim.
    """

    def __init__(
        self,
        config: EquilibrationConfig,
        manager: Any,
        *,
        data_store: Any = None,
        run_id: str | None = None,
        sleep: Any = None,
        now: Any = None,
        executor_factory: Any = None,
        fit_reader: Any = None,
    ) -> None:
        config.validate()
        self.config = config
        self.manager = manager
        self.data_store = data_store
        self.run_id = run_id
        self._sleep = sleep
        self._now = now or time.monotonic
        self._executor_factory = executor_factory or _default_executor_factory
        #: ``(step_names: dict[channel, str]) -> Sequence[RoundFit]``. Injected so
        #: the settle criterion can be exercised without a database; ``None``
        #: reads this run's own fits back out of the store.
        self._fit_reader = fit_reader
        self._abort_flag = threading.Event()
        self._rh_started_by_run = False
        #: ``on_progress(event: ProgressEvent) -> None``. Set by the CLI, or by a
        #: GUI tab, or left ``None``. Called from the run's own thread *and* from
        #: the worker threads the watched holds run on, so a consumer that touches
        #: a widget must marshal; a consumer that prints need not.
        self.on_progress: Any = None
        #: How many progress emissions were swallowed. Nonzero means the renderer
        #: is broken, not the run.
        self.progress_failures = 0
        #: How many telemetry reads were skipped to avoid contending with the rig.
        self.telemetry_skips = 0
        self._measuring = False
        self._t0 = float(self._now())
        self._last_tick_at = self._t0
        self._setpoints_done = 0
        self._rounds_done = 0
        self._context: dict[str, Any] = {"n_setpoints": config.n_setpoints,
                                         "n_rounds": config.rounds_per_setpoint}
        self._cost_warned = False
        self._fits_warned = False
        #: One row per executed round: what it *actually* cost.
        self.round_costs: list[dict[str, Any]] = []
        self.points: list[dict[str, Any]] = []
        self.holds: list[dict[str, Any]] = []
        self.approaches: list[dict[str, Any]] = []
        self.setpoints: list[dict[str, Any]] = []
        self.aborted = False
        self.abort_reason = ""
        self.restored_ambient = False
        #: Why the teardown could not bring the chamber down, or ``""``. Non-empty
        #: is the one state in this class that needs a human at the rig, so it is
        #: kept as a value rather than left in a log line: the CLI reads it.
        self.restore_error = ""
        #: The temperature this run last **commanded**, recorded before the write
        #: rather than after — a ``write_sp`` that raises halfway may still have
        #: landed, and a restore-failure message that names no setpoint tells an
        #: operator nothing they can act on. ``NaN`` until the first setpoint.
        self.last_commanded_C = float("nan")
        #: Did the sidecar reach disk? The spectra survive in the database without
        #: it, but the hold verdicts — the evidence that the chamber was at
        #: condition when each spectrum was taken — exist nowhere else.
        self.sidecar_written = False
        self.sidecar_error = ""

    def abort(self) -> None:
        """Request an orderly stop; the watched holds check this every poll."""
        self._abort_flag.set()

    # ── progress ─────────────────────────────────────────────────────────

    def _fraction(self) -> float:
        """Completed setpoints plus completed rounds inside the current one.

        The approach is deliberately not counted: its duration is exactly the
        thing this run cannot predict (that is why the worst case uses the
        timeouts), so folding a guess at it into the fraction would put a guess
        into every ETA. The cost is a fraction that reads low early in a setpoint
        and an ETA that errs long — the safe direction when nobody is watching.
        """
        n = max(1, int(self.config.n_setpoints))
        per = 1.0 / n
        rounds = max(1, int(self.config.rounds_per_setpoint))
        within = min(1.0, self._rounds_done / rounds) * per
        return min(0.999, self._setpoints_done * per + within)

    def _instrument_busy(self) -> bool:
        """Is either controller's lock held right now? Asked, never waited on.

        ``InstrumentManager.acquire`` is an ``async`` context manager and this is
        called from the watched-hold worker thread, so awaiting is not available
        and blocking on a synchronous handle would be worse than useless: a
        telemetry read that waits is a telemetry read that delays the next EIS
        shot, and the spacing of those shots *is* the σ(t) series. A missed
        monitor line costs nothing. So the lock is inspected, and anything
        ambiguous counts as busy.
        """
        for name in (self.config.temp_instrument, self.config.rh_instrument):
            try:
                lock = getattr(self.manager.get(name), "_lock", None)
            except Exception:
                continue
            locked = getattr(lock, "locked", None)
            if locked is None:
                continue
            try:
                if bool(locked()):
                    return True
            except Exception:
                return True
        return False

    def _env_fields(self) -> dict[str, Any]:
        """A telemetry snapshot for the progress stream, or an honest skip.

        This exists so an operator can watch a headless run's controls without
        opening the GUI — which would contend for the rig lock and the serial
        ports of the very run being checked on. It reuses
        :func:`~softae.core.conditions_capture.read_environment` verbatim rather
        than reading the drivers again: one reader, one mapping of which
        controller owns which value, and no second place for that mapping to
        drift.

        Two things it will not do. It will not block: an in-flight measurement
        means the snapshot is skipped, because a delayed EIS shot distorts the
        series this run exists to produce and a missing monitor line does not. And
        it will not invent: ``read_environment`` already maps an unreadable or
        NaN PV to ``None``, and that ``None`` is passed through for the renderer
        to show as unavailable.
        """
        if self._measuring or self._instrument_busy():
            self.telemetry_skips += 1
            return {"env": {}, "env_status": ENV_SKIPPED}
        try:
            return {"env": dict(read_environment(self.manager)), "env_status": ENV_OK}
        except Exception:
            self.telemetry_skips += 1
            return {"env": {}, "env_status": ENV_SKIPPED}

    def _emit(self, kind: str, **fields: Any) -> None:
        """Publish one progress event. **Cannot fail the run.**

        Three separate guards, because they fail for different reasons: building
        the event (a bad keyword or an unformattable value), the durable log (a
        broken structlog processor), and the consumer (a formatting bug, a closed
        pipe, a dead GUI). Each degrades to silence and a counter; none of them
        reaches the caller, which is mid-experiment with a heater at 85 °C.
        """
        try:
            event = ProgressEvent(kind=kind, setpoints_done=self._setpoints_done,
                                  fraction=self._fraction(),
                                  elapsed_s=float(self._now()) - self._t0,
                                  wall_clock=time.strftime("%Y-%m-%d %H:%M:%S"),
                                  **{**self._context, **fields})
        except Exception:
            self.progress_failures += 1
            return
        self._last_tick_at = float(event.elapsed_s) + self._t0
        try:
            if kind in MILESTONE_EVENTS:
                logger.info("equilibration_progress", run_id=self.run_id,
                            **event.as_dict())
            else:
                logger.debug("equilibration_progress", run_id=self.run_id,
                             **event.as_dict())
        except Exception:
            self.progress_failures += 1
        hook = self.on_progress
        if hook is None:
            return
        try:
            hook(event)
        except Exception:
            self.progress_failures += 1

    def _tick(self, kind: str, **fields: Any) -> None:
        """Emit at most one throttled event per ``progress_interval_s``.

        Used for the approach ticks and the idle heartbeat, which are driven by
        the PV poll and would otherwise arrive at whatever rate the hardware is
        read at. Any un-throttled milestone resets the clock, so a heartbeat never
        follows a real line by less than the interval.
        """
        now = float(self._now())
        if now - self._last_tick_at < float(self.config.progress_interval_s):
            return
        # Read after the throttle, never before: this is real instrument I/O and
        # it must happen at the monitor's cadence, not the PV poll's.
        self._emit(kind, **{**self._env_fields(), **fields})

    async def run(self) -> dict[str, Any]:
        """Execute every leg and setpoint, then write and return the sidecar.

        **Every** way out of this method goes through :meth:`_teardown` — normal
        completion, :class:`EquilibrationAbort`, ``KeyboardInterrupt``,
        cancellation, and any bug — because the chamber is only brought back to
        ambient by code that is still running. An operator who interrupts, and a
        driver that stops answering, are the two commonest exits and neither is an
        ``EquilibrationAbort``: before this, both left the stage heater commanded
        at up to 85 °C with the process gone and the drivers disconnected, in an
        occupied building.
        """
        projection = project_duration(self.config)
        logger.info("equilibration_run_start", run_id=self.run_id,
                    channels=self.config.channels, legs=list(self.config.legs),
                    projection=projection.describe())
        self._t0 = float(self._now())
        self._last_tick_at = self._t0
        self._emit(EV_RUN_STARTED, detail=projection.describe())
        try:
            for leg in self.config.legs:
                self._context["leg"] = leg
                self._emit(EV_LEG_STARTED)
                for sp_idx, temp in enumerate(self.config.leg_temperatures(leg)):
                    await self._run_setpoint(leg, sp_idx, temp)
                self._emit(EV_LEG_FINISHED)
        # `BaseException`, not `Exception`: `KeyboardInterrupt` and
        # `asyncio.CancelledError` both derive from `BaseException`, so
        # `except Exception` would miss precisely the two exits that matter most
        # here. A cancelled run is torn down like any other — cancellation is a
        # request to stop the *work*, not permission to abandon the chamber; the
        # teardown's awaits are best-effort in that case, since a task whose
        # cancellation is still pending may re-raise on the next await, but the
        # attempt costs nothing and usually succeeds.
        #
        # It re-raises unconditionally. This is a teardown, NOT a handler: the
        # traceback and the exit code must stay exactly as truthful as they were.
        except BaseException as exc:
            self._record_failure(exc)
            await self._teardown()
            raise
        self._emit(EV_RUN_FINISHED, detail=f"{len(self.points)} spectra, "
                                           f"{len(self.holds)} watched windows")
        return await self._teardown()

    # ── teardown: the part that must run whatever happened ───────────────

    def _record_failure(self, exc: BaseException) -> None:
        """Mark the run aborted and announce it, whatever the class of failure.

        :class:`EquilibrationAbort` keeps its own vocabulary — ``kind`` and
        ``axis`` are what the sidecar and the report already read — and is
        deliberately not re-spelled through the generic path. Everything else is
        recorded as ``<ExceptionType>: <message>`` so ``abort_reason`` can never
        be misread as a watched hold's verdict when it was a broken serial port
        or a Ctrl-C.
        """
        self.aborted = True
        if isinstance(exc, EquilibrationAbort):
            self.abort_reason = f"{exc.kind}: {exc}"
            logger.error("equilibration_run_aborted", kind=exc.kind, axis=exc.axis,
                         reason=str(exc))
            self._emit(EV_RUN_FINISHED, verdict=VERDICT_ABORTED, axis=exc.axis,
                       detail=f"{exc.kind}: {exc}")
            return
        self.abort_reason = f"{type(exc).__name__}: {exc}"
        logger.error("equilibration_run_failed", run_id=self.run_id,
                     error_type=type(exc).__name__, reason=str(exc),
                     detail="not an EquilibrationAbort; the teardown still runs")
        self._emit(EV_RUN_FINISHED, verdict=VERDICT_ABORTED, detail=self.abort_reason)

    async def _teardown(self) -> dict[str, Any]:
        """Bring the chamber down, say whether it came down, and persist.

        On most of its paths this runs with an exception already in flight, so
        **nothing here may raise**: an exception raised inside an ``except`` block
        replaces the original, and the operator would be handed a "restore
        failed" traceback in place of the driver fault that caused it — the one
        piece of information that explains the night. Both halves are guarded and
        both record their failure as a value instead.

        ``BaseException`` is deliberately *not* caught around the restore: a
        second Ctrl-C during teardown is an explicit instruction to stop now, and
        swallowing it would leave no way out of a hung driver at all.
        """
        try:
            await self._restore_ambient()
        except Exception as exc:
            self.restored_ambient = False
            self.restore_error = f"{type(exc).__name__}: {exc}"
            logger.warning("equilibration_restore_failed", error=str(exc))
        self._announce_ambient()
        return self._safe_write_sidecar()

    def last_commanded_description(self) -> str:
        """The setpoint the chamber was last driven to, for a failure message.

        Named rather than left implicit: an operator told only that the restore
        failed has nothing to act on, and the config's peak is not the answer —
        the run may have failed on the first setpoint of the up leg.
        """
        temp = self.last_commanded_C
        if temp != temp:            # NaN — no setpoint was ever commanded
            return "its power-on setpoint (this run commanded none)"
        rh = (f" and the humidifier at {self.config.rh_setpoint_pct:g} %RH"
              if self._rh_started_by_run else "")
        return f"{temp:g} C{rh}"

    def _announce_ambient(self) -> None:
        """State the outcome of the restore, in the operator's own stream.

        A silent restore attempt is indistinguishable from a successful one, so
        both are said. The failure line names the setpoint the chamber is still
        commanded to and tells the operator to check it by hand, because at that
        point nothing in software is going to.
        """
        if not self.restore_error:
            self._emit(EV_AMBIENT_RESTORED, verdict=VERDICT_MET,
                       detail=f"ambient restored: {self.config.temp_instrument} "
                              f"commanded to {self.config.ambient_C:g} C")
            logger.info("equilibration_ambient_restored", run_id=self.run_id,
                        ambient_C=self.config.ambient_C)
            return
        # The instruction and the setpoint come FIRST, before the driver's own
        # message. `ProgressRenderer._line` truncates at 234 characters and the
        # error text is the one part of this sentence with no length bound; put it
        # in front and the only thing an operator is guaranteed to read is the
        # driver's complaint, not what to do about it. ASCII throughout, for the
        # same reason the rest of this module's details are.
        detail = (f"CHECK THE CHAMBER MANUALLY: the ambient restore FAILED and it "
                  f"may still be commanded to {self.last_commanded_description()}. "
                  f"({self.restore_error})")
        self._emit(EV_AMBIENT_RESTORED, verdict=VERDICT_UNMET, detail=detail)
        logger.error("equilibration_ambient_restore_failed", run_id=self.run_id,
                     error=self.restore_error,
                     last_commanded=self.last_commanded_description(),
                     detail="nothing further will bring the chamber down")

    # ── one setpoint ─────────────────────────────────────────────────────

    async def _run_setpoint(self, leg: str, sp_idx: int, temperature_C: float) -> None:
        cfg = self.config
        temp_ctl = self.manager.get(cfg.temp_instrument)
        rh_ctl = self._rh_controller()

        self._rounds_done = 0
        self._context.update(leg=leg, setpoint_index=sp_idx,
                             temperature_C=float(temperature_C),
                             rh_setpoint_pct=float(cfg.rh_setpoint_pct),
                             phase="approach", round_index=-1, round_kind="")
        self._emit(EV_SETPOINT_STARTED, **self._env_fields())

        # Recorded BEFORE the write, not after: a `write_sp` that raises partway
        # may still have landed the setpoint, and the teardown message has to name
        # what the chamber might be sitting at.
        self.last_commanded_C = float(temperature_C)
        await asyncio.to_thread(temp_ctl.write_sp, temperature_C, 0)
        t_approach = await self._approach(
            temp_ctl.get_pv, temperature_C, axis="temperature",
            instrument=cfg.temp_instrument, tolerance=cfg.tolerance_C,
            # Leg-dependent: coming down nothing drives the stage, and the
            # ascending allowance timed out mid-descent on 2026-08-11.
            timeout_s=cfg.temperature_approach_timeout_s(leg),
            leg=leg, setpoint_index=sp_idx)

        rh_approach = None
        if rh_ctl is not None:
            await self._start_rh(rh_ctl)
            rh_approach = await self._approach(
                rh_ctl.get_H, cfg.rh_setpoint_pct, axis="humidity",
                instrument=cfg.rh_instrument, tolerance=cfg.rh_tolerance_pct,
                timeout_s=cfg.rh_approach_timeout_s, leg=leg, setpoint_index=sp_idx)

        # `_setpoints_done` is still 0 here — it is incremented once this setpoint
        # finishes — so it is this setpoint's ordinal in the *run*, not in the leg.
        # The down leg's first setpoint is a re-visit of a temperature the films
        # have already seen, and it gets neither the extra time floor nor the τ
        # floor for that reason.
        ordinal = self._setpoints_done
        series = await self._run_series(
            leg, sp_idx, temperature_C, temp_ctl, rh_ctl,
            floor_s=hold_floor_s(cfg, ordinal),
            min_rounds=tau_floor_rounds(cfg, ordinal))

        temp_verdict = HoldOutcome.merge(series.temp_holds)
        rh_verdict = HoldOutcome.merge(series.rh_holds)
        self._setpoints_done += 1
        self._context.update(phase="", round_index=-1, round_kind="")
        self._announce_settle(series)
        # The per-setpoint verdict is the primary result of question 2 — whether
        # the rig can hold 15 %RH at this temperature — so it is announced when it
        # resolves, not saved for a report the operator reads in the morning.
        held = [o for o in (temp_verdict, rh_verdict) if o.axis]
        self._emit(
            EV_SETPOINT_FINISHED,
            verdict=(VERDICT_MET if held and all(o.met for o in held)
                     else VERDICT_UNMET),
            detail="; ".join(o.describe() for o in held) or "no watched window",
            **self._env_fields(),
        )
        self.setpoints.append({
            "leg": leg, "setpoint_index": sp_idx, "temperature_C": temperature_C,
            "rh_setpoint_pct": cfg.rh_setpoint_pct,
            # `n_rounds` is now what was RUN; the ceiling is beside it, because a
            # reader comparing two setpoints needs to know that 4 vs 15 was the
            # criterion working and not a shorter configuration.
            "n_rounds": series.n_rounds,
            "temp_approach_reached": t_approach.reached,
            "rh_approach_reached": None if rh_approach is None else rh_approach.reached,
            "hold_met": temp_verdict.met if series.temp_holds else None,
            "rh_hold_met": rh_verdict.met if series.rh_holds else None,
            "hold_pv_min": _num(temp_verdict.pv_min),
            "hold_pv_max": _num(temp_verdict.pv_max),
            "hold_n_warn": temp_verdict.n_warn,
            "hold_held_s": round(temp_verdict.held_s, 3),
            "rh_hold_pv_min": _num(rh_verdict.pv_min),
            "rh_hold_pv_max": _num(rh_verdict.pv_max),
            "rh_hold_n_warn": rh_verdict.n_warn,
            **series.as_dict(),
        })
        # Persist here, not only at the end. A power cut or a `kill -9` catches no
        # handler at all, and the hold verdicts are the one thing this run produces
        # that cannot be reconstructed afterwards — the coordinate is recoverable
        # from the step names embedded in `payload_path`, the verdicts are not.
        # One small JSON write against a setpoint tens of minutes long, and it
        # cannot raise into the loop.
        self._safe_write_sidecar()

    async def _run_series(self, leg: str, sp_idx: int, temperature_C: float,
                          temp_ctl: Any, rh_ctl: Any, *,
                          floor_s: float, min_rounds: int) -> "SeriesOutcome":
        """The σ(t) series at one setpoint: rounds until it settles, or the ceiling.

        The ceiling is unconditional. Stopping *early* needs three things at once,
        and they answer different questions:

        * the criterion has a settled window (``settle_n_rounds`` rounds of it);
        * the time floor has elapsed — a series that looks flat two rounds into a
          setpoint the chamber has not finished reaching is flat for the wrong
          reason;
        * ``min_rounds`` rounds exist. Inside the τ window that is
          :data:`MIN_POINTS_FOR_TAU`, because a shorter series is one
          :func:`~softae.analysis.equilibration.fit_equilibration` refuses
          outright and the run must not acquire a setpoint it cannot analyse; past
          the window it is 1, because there is no τ left to protect and the fifth
          round would re-measure a settled number. :func:`tau_floor_rounds` owns
          which of the two this is — the loop is handed the answer, not the rule.
        """
        cfg = self.config
        hold_start = float(self._now())
        self._context["phase"] = KIND_SERIES
        tracker = SettleTracker(
            enabled=cfg.settle_enabled, tol_rel=cfg.settle_tol_rel,
            n_rounds=cfg.settle_n_rounds, min_channels=cfg.settle_min_channels,
            r1_bound_ohms=r1_lower_bound_ohms(cfg.eis_model))
        temp_holds: list[HoldOutcome] = []
        rh_holds: list[HoldOutcome] = []
        n_rounds = 0
        settled_early = False

        for round_index in range(cfg.rounds_per_setpoint):
            measured_s = await self._measure_round(
                leg, sp_idx, round_index, KIND_SERIES, hold_start, temperature_C)
            n_rounds += 1
            tracker.observe(self._round_fits(leg, sp_idx, round_index))
            elapsed_s = float(self._now()) - hold_start
            if (tracker.settled and elapsed_s >= float(floor_s)
                    and n_rounds >= int(min_rounds)):
                settled_early = True
                break
            if round_index == cfg.rounds_per_setpoint - 1:
                break             # no watched gap after the final round
            await self._watched_gap(leg, sp_idx, round_index, measured_s,
                                    temperature_C, temp_ctl, rh_ctl,
                                    temp_holds, rh_holds)

        endorsed, endorsement, noise_floor_rel = tracker.endorsement()
        return SeriesOutcome(
            outcome=tracker.outcome(stopped_early=settled_early),
            n_rounds=n_rounds, ceiling=cfg.rounds_per_setpoint,
            participating=tracker.participating,
            excluded=dict(tracker.last.excluded) if tracker.last else {},
            floor_s=float(floor_s), held_s=float(self._now()) - hold_start,
            max_deviation_rel=(tracker.last.max_deviation_rel
                               if tracker.last else None),
            tolerance_achievable=endorsed, endorsement=endorsement,
            noise_floor_rel=noise_floor_rel,
            temp_holds=temp_holds, rh_holds=rh_holds)

    async def _watched_gap(self, leg: str, sp_idx: int, round_index: int,
                           measured_s: float, temperature_C: float,
                           temp_ctl: Any, rh_ctl: Any,
                           temp_holds: list[HoldOutcome],
                           rh_holds: list[HoldOutcome]) -> None:
        """The graded dead time between two rounds, so the round period is honoured."""
        cfg = self.config
        # The MEASURED round, not the modelled one: the period is honoured from
        # this round's start, and the model under-counts a real round by whatever
        # its fit residual and the unmodelled per-channel overhead come to (see
        # `inter_round_gap_s`). Passing the modelled cost here is what made every
        # cycle overrun --round-period-s.
        gap = inter_round_gap_s(cfg, measured_s)
        # The gap is split so BOTH axes are graded by the same primitive.
        # `monitored_hold` grades one target, and leaving either axis ungraded for
        # the whole series is what the fail-open `wait()` already does.
        share = gap / 2.0 if rh_ctl is not None else gap
        temp_holds.append(await self._hold(
            temp_ctl.get_pv, temperature_C, share, axis="temperature",
            instrument=cfg.temp_instrument, tolerance=cfg.tolerance_C,
            warn=cfg.warn_C, fault=cfg.fault_C, leg=leg, setpoint_index=sp_idx,
            round_index=round_index))
        if rh_ctl is not None:
            rh_holds.append(await self._hold(
                rh_ctl.get_H, cfg.rh_setpoint_pct, share, axis="humidity",
                instrument=cfg.rh_instrument, tolerance=cfg.rh_tolerance_pct,
                warn=cfg.rh_warn_pct, fault=cfg.rh_fault_pct, leg=leg,
                setpoint_index=sp_idx, round_index=round_index))

    def _round_fits(self, leg: str, sp_idx: int, round_index: int) -> list[RoundFit]:
        """This round's σ per channel, read back from the store. **Never raises.**

        A store that cannot be read yields an all-``None`` fit per channel, which
        is zero participating channels, which is ``not_evaluable`` and a run to
        the ceiling. That is the safe direction and the only one: a read failure
        must never be able to shorten a hold.
        """
        if not self.config.settle_enabled:
            return []
        names = {ch: measurement_step_name(ch, leg, sp_idx, round_index,
                                           kind=KIND_SERIES)
                 for ch in self.config.channels}
        reader = self._fit_reader or (
            lambda step_names: load_round_fits(self.data_store, self.run_id or "",
                                               step_names))
        try:
            return list(reader(names))
        except Exception as exc:
            # Said once per run, like the period-overrun warning: the cause is a
            # store, not a round, so 120 identical lines over a night would bury
            # the verdicts without adding a fact.
            if not self._fits_warned:
                self._fits_warned = True
                logger.warning("equilibration_round_fits_unavailable",
                               run_id=self.run_id, leg=leg, setpoint_index=sp_idx,
                               round_index=round_index, error=str(exc),
                               detail="the settle criterion has no evidence to read; "
                                      "every setpoint will run to its ceiling")
            return [RoundFit(channel=ch) for ch in self.config.channels]

    def _announce_settle(self, series: "SeriesOutcome") -> None:
        """Say why this setpoint stopped, once, in the operator's own stream."""
        self._emit(EV_SETTLE_VERDICT,
                   verdict=VERDICT_MET if series.settled else VERDICT_UNMET,
                   detail=series.describe())
        logger.info("equilibration_settle_verdict", run_id=self.run_id,
                    **series.as_dict())

    async def _approach(self, read_pv, target, **kw) -> ApproachOutcome:
        leg, sp_idx = kw.pop("leg"), kw.pop("setpoint_index")
        axis = kw["axis"]
        self._context["phase"] = "approach"
        self._emit(EV_APPROACH_STARTED, axis=axis, target=float(target))

        def _sample(_stamp: float, pv: float) -> None:
            # The approach is the long silent part — up to 30 min per axis per
            # setpoint with nothing else to print. This is what makes it visible.
            self._tick(EV_APPROACH_PROGRESS, axis=axis, pv=pv, target=float(target),
                       phase="approach")

        outcome = await asyncio.to_thread(
            approach_setpoint, read_pv, target,
            poll_interval_s=self.config.poll_interval_s,
            should_abort=self._abort_flag.is_set, sleep=self._sleep, now=self._now,
            on_sample=_sample, **kw)
        self._emit(EV_APPROACH_FINISHED, axis=axis, target=float(target),
                   pv=outcome.pv_final, phase="approach",
                   verdict=VERDICT_MET if outcome.reached else VERDICT_UNMET,
                   detail=f"{axis} {'reached' if outcome.reached else 'DID NOT reach'} "
                          f"{float(target):g} in {outcome.elapsed_s:.0f}s")
        self.approaches.append({**outcome.as_dict(), "leg": leg,
                                "setpoint_index": sp_idx})
        return outcome

    async def _hold(self, read_pv, target, hold_time_s, **kw) -> HoldOutcome:
        leg = kw.pop("leg")
        sp_idx, round_index = kw.pop("setpoint_index"), kw.pop("round_index")
        axis = kw["axis"]
        self._context["phase"] = "hold"

        def _sample(_stamp: float, pv: float) -> None:
            # Nothing changes during a watched gap, which is exactly when a silent
            # console is indistinguishable from a hung one.
            self._tick(EV_HEARTBEAT, axis=axis, pv=pv, target=float(target),
                       phase="hold")

        outcome = await asyncio.to_thread(
            watch_hold, read_pv, target, hold_time_s=hold_time_s,
            grace_s=self.config.grace_s, poll_interval_s=self.config.poll_interval_s,
            should_abort=self._abort_flag.is_set, sleep=self._sleep, now=self._now,
            on_sample=_sample, **kw)
        self._emit(EV_HOLD_VERDICT, axis=axis, target=float(target),
                   pv=outcome.pv_final, round_index=round_index, phase="hold",
                   verdict=VERDICT_MET if outcome.met else VERDICT_UNMET,
                   detail=outcome.describe())
        self.holds.append({**outcome.as_dict(), "leg": leg, "setpoint_index": sp_idx,
                           "round_index": round_index})
        return outcome

    # ── one round ────────────────────────────────────────────────────────

    async def _measure_round(
        self, leg: str, sp_idx: int, round_index: int, kind: str,
        hold_start: float, temperature_C: float,
    ) -> float:
        """Measure every channel once; return the round's **measured** wall clock.

        Returned rather than stashed on ``self`` because the caller subtracts it
        from ``round_period_s`` to size the watched gap — a value that decides how
        long the rig is left unwatched has no business being read back off shared
        state.
        """
        self._build_mscr(kind)
        workflow = build_round_workflow(
            self.config, leg=leg, setpoint_index=sp_idx, round_index=round_index,
            kind=kind, temperature_C=temperature_C, run_id=self.run_id)
        executor = self._executor_factory(self.manager, self.data_store, self.run_id)
        self._executor = executor

        completed: dict[int, float] = {}
        self._context.update(phase=kind, round_index=round_index, round_kind=kind)
        self._emit(EV_ROUND_STARTED, **self._env_fields())

        def _on_complete(step, index, total, raw, elapsed=0.0):
            try:
                channel = int(step.params.get("chan", 0))
            except (TypeError, ValueError):
                return
            completed[channel] = float(self._now()) - hold_start
            # A 16-channel round is minutes long at the measured per-channel cost;
            # without this the operator sees one line per round and nothing between.
            self._emit(EV_CHANNEL_MEASURED, channel=channel,
                       detail=f"{index + 1}/{total}")

        executor.on_step_complete = _on_complete
        # No telemetry read while the DAG owns the rig: the monitor must never be
        # the reason a spectrum is late.
        self._measuring = True
        started = float(self._now())
        try:
            await executor.run(workflow)
        finally:
            self._measuring = False
        duration = float(self._now()) - started
        per_channel = duration / max(1, len(self.config.channels))
        self.round_costs.append({
            "leg": leg, "setpoint_index": sp_idx, "round_index": round_index,
            "kind": kind, "n_channels": len(self.config.channels),
            "n_completed": len(completed), "duration_s": round(duration, 3),
            "per_channel_s": round(per_channel, 3),
            "modelled_s": round(eis_round_cost_s(
                self.config, self.config.eis_preset), 3),
        })
        if kind == KIND_SERIES:
            self._rounds_done += 1
            self._warn_if_round_overruns(duration, per_channel)
        self._emit(EV_ROUND_FINISHED, **self._env_fields(),
                   round_duration_s=duration, per_channel_s=per_channel,
                   detail=f"{len(completed)}/{len(self.config.channels)} channels "
                          f"in {duration:.0f}s ({per_channel:.1f}s/ch)")

        for channel in self.config.channels:
            if channel not in completed:
                # A step that did not complete has no spectrum; a fabricated point
                # would enter the fit as a real one.
                continue
            self.points.append({
                "step_name": measurement_step_name(channel, leg, sp_idx, round_index,
                                                   kind=kind),
                "channel": channel, "leg": leg, "setpoint_index": sp_idx,
                "round_index": round_index, "kind": kind,
                "t_since_hold_s": round(completed[channel], 3),
                "temperature_C": temperature_C,
                "rh_setpoint_pct": self.config.rh_setpoint_pct,
            })
        return duration

    # ── what a round really costs ────────────────────────────────────────

    def _warn_if_round_overruns(self, duration_s: float, per_channel_s: float) -> None:
        """Say it once, early, and **do not fix it**.

        ``round_period_s`` is an experimental parameter: it is the sampling
        interval of σ(t), it sets the shortest resolvable τ, and the fitter reads
        the series as evenly spaced. Silently stretching it mid-run to accommodate
        a slow round would make the series inhomogeneous in a way nothing
        downstream can see — a worse defect than the overrun itself, because the
        overrun is at least visible in the timestamps.

        So the operator is told, with the measured number and the flag that would
        fix it, and the run continues exactly as configured.
        """
        period = float(self.config.round_period_s)
        if self._cost_warned or duration_s <= period:
            return
        self._cost_warned = True
        suggested = math.ceil(duration_s * 1.1 / 10.0) * 10.0
        modelled = eis_round_cost_s(self.config, self.config.eis_preset)
        # What the loop will actually do from here on. The gap floors at one poll
        # interval so both axes stay graded, so the cycle is the round plus that
        # floor — stated outright, because "the period is unachievable" is only
        # actionable next to the number that replaces it.
        cycle = duration_s + inter_round_gap_s(self.config, duration_s)
        # Kept tight on purpose: `ProgressRenderer._line` truncates a milestone at
        # 234 characters, and the `--round-period-s` suggestion is at the end —
        # the first thing an over-long detail would amputate. The modelled figure
        # and the sampling-interval rationale live in the log record below, where
        # there is no width limit, and in `plan`'s period caution.
        self._emit(
            EV_COST_WARNING, round_duration_s=duration_s, per_channel_s=per_channel_s,
            verdict=VERDICT_UNMET,
            detail=(f"a round took {duration_s:.0f}s ({per_channel_s:.1f}s/ch) but "
                    f"--round-period-s is {period:.0f}s: UNACHIEVABLE at "
                    f"{len(self.config.channels)} ch on '{self.config.eis_preset}'. "
                    f"The cycle is now {cycle:.0f}s. sigma(t) is not resampled to fit "
                    f"-- re-run with --round-period-s {suggested:.0f}"),
        )
        logger.warning(
            "equilibration_round_overruns_period", run_id=self.run_id,
            measured_round_s=round(duration_s, 1),
            measured_per_channel_s=round(per_channel_s, 2),
            modelled_round_s=round(modelled, 1), round_period_s=period,
            actual_cycle_s=round(cycle, 1), n_channels=len(self.config.channels),
            preset=self.config.eis_preset, suggested_round_period_s=suggested,
            detail="the sampling interval is an experimental parameter and is not "
                   "auto-adjusted; the modelled cost beside it is a fitted sweep "
                   "estimate, not a measurement",
        )

    def measured_round_cost_s(self, kind: str = KIND_SERIES) -> float | None:
        """Median wall-clock of the executed rounds of *kind*, or ``None``.

        Median rather than mean: one round that caught a retry should not move the
        number the rest of the system will inherit.
        """
        values = sorted(float(row["duration_s"]) for row in self.round_costs
                        if row["kind"] == kind)
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    def measured_cost_summary(self) -> dict[str, Any]:
        """The run's own answer to "what does a round actually cost?".

        ``estimate_eis_duration`` models the frequency sweep, and every consumer of
        it — this projection, ``preflight.project_campaign``, the campaign check —
        inherits both its fit residual and the per-channel overhead it omits. This
        is the measurement they are answerable to, recorded beside the modelled
        figure rather than in place of it so the size of the gap survives. It read
        ~10x once; the ratio below is what would show that happening again.
        """
        out: dict[str, Any] = {"rounds": list(self.round_costs),
                               "round_period_s": self.config.round_period_s,
                               "period_overrun_warned": self._cost_warned}
        for kind in (KIND_SERIES,):
            measured = self.measured_round_cost_s(kind)
            if measured is None:
                continue
            preset = self.config.eis_preset
            modelled = eis_round_cost_s(self.config, preset)
            channels = max(1, len(self.config.channels))
            out[kind] = {
                "n_rounds": sum(1 for r in self.round_costs if r["kind"] == kind),
                "preset": preset,
                "measured_round_s": round(measured, 3),
                "measured_per_channel_s": round(measured / channels, 3),
                "modelled_round_s": round(modelled, 3),
                "modelled_per_channel_s": round(modelled / channels, 3),
                "ratio_measured_over_modelled": (round(measured / modelled, 3)
                                                 if modelled > 0 else None),
                "unmodelled_per_channel_s": round((measured - modelled) / channels, 3),
            }
        return out

    def _build_mscr(self, kind: str) -> None:
        """Write the per-channel scripts for this round's preset."""
        try:
            from softae.core.eis_scripts import EISParams
            from softae.drivers.mscr_library import eis_run_mscrbuild
        except ImportError:
            return
        params = EISParams.from_preset(self.config.eis_preset)
        for channel in self.config.channels:
            try:
                eis_run_mscrbuild(
                    mscr_path(channel, kind, self.run_id),
                    mux_ch=channel, mVac=params.mv_ac,
                    f_hi=params.f_hi, f_lo=params.f_lo_mHz, npts=params.npts,
                    mVdc=params.mv_dc)
            except Exception as exc:
                logger.warning("equilibration_mscr_build_failed", channel=channel,
                               kind=kind, error=str(exc))

    # ── environment lifecycle ────────────────────────────────────────────

    def _rh_controller(self) -> Any:
        try:
            return self.manager.get(self.config.rh_instrument)
        except Exception:
            logger.warning("equilibration_rh_absent", name=self.config.rh_instrument,
                           detail="the humidity axis will be neither driven nor graded")
            return None

    async def _start_rh(self, rh_ctl: Any) -> None:
        await asyncio.to_thread(rh_ctl.set_setpoint, self.config.rh_setpoint_pct)
        try:
            running = bool((rh_ctl.status() or {}).get("running", False))
        except Exception:
            running = False
        if not running:
            await asyncio.to_thread(rh_ctl.start)
            self._rh_started_by_run = True

    async def _restore_ambient(self) -> None:
        """Best effort, and it runs on **every** exit path — see :meth:`_teardown`.

        Often called with a driver already broken, since that is frequently why we
        got here, so each half is guarded separately: a dead humidifier must not
        stop the heater being commanded down, and a heater that will not answer
        must not stop the humidifier being stopped. Both failures are folded into
        :attr:`restore_error`, because a humidifier left running unattended is the
        same class of problem as a heater left hot.
        """
        self.restore_error = ""
        problems: list[str] = []
        try:
            temp_ctl = self.manager.get(self.config.temp_instrument)
            await asyncio.to_thread(temp_ctl.write_sp, self.config.ambient_C, 0)
            self.restored_ambient = True
        except Exception as exc:
            logger.warning("equilibration_restore_failed", error=str(exc))
            self.restored_ambient = False
            problems.append(f"stage heater not returned to ambient: "
                            f"{type(exc).__name__}: {exc}")
        if self._rh_started_by_run:
            try:
                await asyncio.to_thread(self.manager.get(self.config.rh_instrument).stop)
            except Exception as exc:
                logger.warning("equilibration_rh_stop_failed", error=str(exc))
                problems.append(f"humidifier not stopped: {type(exc).__name__}: {exc}")
        self.restore_error = "; ".join(problems)

    # ── Stage 1 persistence ──────────────────────────────────────────────

    def thickness_provenance(self) -> dict[str, Any]:
        """The thickness this run divides by, **and where it came from**.

        The geometry reaches ``fit_results.electrode_t_cm`` through the EIS step
        params, so the stored σ is already divided by it — but
        ``fit_results.thickness_method`` is populated only from a
        ``SpectrumReport``, and this run passes none. The column therefore stays
        ``NULL``: a σ on disk whose thickness is a **hand-computed digital-twin
        target** is indistinguishable from one measured by profilometry. That is
        the P.7 / P.11 provenance gap resurfacing in a new place.

        Closing it in ``fit_results`` is Stage 2 and needs a ``data_store.py``
        claim. Closing it *here* costs nothing and makes the run self-describing:
        whoever reads this sidecar knows which tier the σ rests on. The vocabulary
        is :data:`~softae.analysis.eis.geometry.THICKNESS_METHODS` verbatim, never
        a new string, so this record and a later ``fit_results`` row cannot
        disagree about what ``'target'`` means.

        ``None`` when no geometry was supplied — in which case σ is NULL anyway
        and there is no thickness to have provenance for.
        """
        geom = self.config.electrode_geometry or {}
        t_cm = geom.get("t_cm")
        if t_cm is None:
            return {}
        return {
            "t_cm": _num(t_cm),
            "value_um": _num(float(t_cm) * 1.0e4),
            "units": "um",
            "thickness_method": self.config.thickness_method,
            "vocabulary": list(THICKNESS_METHODS),
            "recorded_in_fit_results": False,
            "note": ("fit_results.thickness_method is NULL for this run: that column "
                     "is populated only from a SpectrumReport and none is passed. "
                     "This sidecar is the only record of the tier, until Stage 2."),
        }

    def sidecar_payload(self) -> dict[str, Any]:
        """The two things no existing table can carry: the coordinate, and the holds.

        ``arrhenius_results`` is one row per (run, channel) with no time axis;
        ``measurements`` is per spectrum, so a per-(channel, setpoint, leg) τ would
        be stored fifteen times; ``conditions`` samples only at measurement time
        and cannot see a hold window; ``fit_results`` is a circuit fit and τ is not
        a circuit parameter.
        """
        cfg = self.config
        projection = project_duration(cfg)
        return {
            "schema": "equilibration/1",
            "run_id": self.run_id or "",
            "config": {
                "channels": list(cfg.channels),
                "temperatures_C": list(cfg.temperatures_C),
                "legs": list(cfg.legs),
                "rh_setpoint_pct": cfg.rh_setpoint_pct,
                # A CEILING since the settle criterion landed. Recorded under the
                # same key so an old reader still finds it, with the criterion
                # beside it so a new one can tell what it meant.
                "rounds_per_setpoint": cfg.rounds_per_setpoint,
                "rounds_per_setpoint_is_ceiling": True,
                "settle_enabled": cfg.settle_enabled,
                "settle_tol_rel": cfg.settle_tol_rel,
                "settle_n_rounds": cfg.settle_n_rounds,
                "settle_min_channels": cfg.settle_min_channels,
                "min_hold_first_s": cfg.min_hold_first_s,
                "min_hold_s": cfg.min_hold_s,
                # How far the MIN_POINTS_FOR_TAU floor reached. Without it a short
                # late setpoint in this run and a short late setpoint in a run
                # that predates the window are indistinguishable on disk.
                "tau_setpoints": cfg.tau_setpoints,
                "round_period_s": cfg.round_period_s,
                "eis_preset": cfg.eis_preset,
                "eis_model": cfg.eis_model,
                "electrode_geometry": cfg.electrode_geometry,
                "tolerance_C": cfg.tolerance_C,
                "rh_tolerance_pct": cfg.rh_tolerance_pct,
                # The bands and allowances a verdict in this file was graded
                # against. `hold_met` is meaningless without the tolerance that
                # produced it, and both moved on 2026-08-12.
                "warn_C": cfg.warn_C,
                "fault_C": cfg.fault_C,
                "grace_s": cfg.grace_s,
                "approach_timeout_s": cfg.approach_timeout_s,
                "down_approach_timeout_s": cfg.down_approach_timeout_s,
                "rh_approach_timeout_s": cfg.rh_approach_timeout_s,
            },
            "thickness": self.thickness_provenance(),
            "projection": {"typical_s": projection.typical_s,
                           "worst_case_s": projection.worst_case_s,
                           "typical_floor_s": projection.typical_floor_s,
                           "worst_floor_s": projection.worst_floor_s,
                           "min_rounds_first": projection.min_rounds_first,
                           "min_rounds_tau": projection.min_rounds_tau,
                           "min_rounds_later": projection.min_rounds_later,
                           "floor_rounds": list(projection.floor_rounds),
                           "basis": projection.basis},
            "measured_cost": self.measured_cost_summary(),
            "points": list(self.points),
            "holds": list(self.holds),
            "approaches": list(self.approaches),
            "setpoints": list(self.setpoints),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            # Recorded because the sidecar may be the *only* trace of an
            # interrupted night: whoever reads it in the morning needs to know
            # whether the chamber was left hot, not just that the run stopped.
            "restored_ambient": self.restored_ambient,
            "restore_error": self.restore_error,
            "last_commanded_C": _num(self.last_commanded_C),
        }

    def sidecar_path(self) -> Path | None:
        if not self.run_id:
            return None
        base = Path(getattr(self.data_store, "project_dir", "db"))
        return base / "runs" / self.run_id / "equilibration.json"

    def _write_sidecar(self) -> dict[str, Any]:
        payload = self.sidecar_payload()
        path = self.sidecar_path()
        if path is None:
            return payload
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self.sidecar_error = f"{type(exc).__name__}: {exc}"
            logger.warning("equilibration_sidecar_failed", error=str(exc))
            return payload
        self.sidecar_written = True
        self.sidecar_error = ""
        logger.info("equilibration_sidecar_written", path=str(path),
                    n_points=len(self.points), n_holds=len(self.holds))
        return payload

    def _safe_write_sidecar(self) -> dict[str, Any]:
        """:meth:`_write_sidecar`, but it cannot raise into a teardown or a loop.

        ``_write_sidecar`` already guards the file write; this guards everything
        before it. ``sidecar_payload`` walks the whole recorded run, and a bad
        value in there would otherwise replace the exception in flight — turning
        "the potentiostat stopped answering" into a JSON error and losing the only
        explanation of what went wrong.
        """
        try:
            return self._write_sidecar()
        except Exception as exc:
            self.sidecar_written = False
            self.sidecar_error = f"{type(exc).__name__}: {exc}"
            logger.error("equilibration_sidecar_failed", run_id=self.run_id,
                         error=str(exc),
                         detail="the hold verdicts exist nowhere else")
            return {}


def _default_executor_factory(manager: Any, data_store: Any, run_id: Any) -> Any:
    from softae.workflows.workflow_executor import WorkflowExecutor

    return WorkflowExecutor(manager=manager, data_store=data_store, run_id=run_id)


def load_sidecar(project_dir: Any, run_id: str) -> dict[str, Any] | None:
    """Read back a recorded run's sidecar, or ``None`` if it was never written."""
    path = Path(project_dir) / "runs" / str(run_id) / "equilibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
