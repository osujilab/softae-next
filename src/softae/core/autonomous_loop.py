"""Closed-loop autonomous experimentation engine.

Implements the ``suggest → execute → analyze → tell`` cycle that drives
optimizer-guided experiment campaigns.

The loop is decoupled from the GUI: it reports progress through plain
callbacks so that callers can bridge into Qt signals, logging, or CLI
output without any widget dependency.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from enum import Enum, auto
from typing import Any, Awaitable, Callable

import structlog

from softae.core.data_store import DataStore
from softae.core.electrode_allocator import ElectrodeAllocator
from softae.errors import AbortedError, OptimizerError
from softae.optimizers.base import BaseOptimizer
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)

#: Default ceiling on any human-in-the-loop gate (approval, board exchange).
#: A gate with no ceiling is the single most likely way an arbitrary-duration
#: unattended run hangs silently — it looks identical to "still working".
#: Generous enough that an operator at the rig is never cut off mid-answer;
#: pass ``gate_timeout_s=None`` to opt back into waiting forever.
DEFAULT_GATE_TIMEOUT_S = 3600.0

#: Outcomes of a campaign control request (:meth:`AutonomousLoop.pause`,
#: :meth:`~AutonomousLoop.resume`, :meth:`~AutonomousLoop.abort`).
#:
#: Every request returns one of these and **none of them is a silent no-op**: a
#: control an operator pressed and heard nothing back from is worse than no
#: control, so "I did nothing" still has to say which nothing.
CONTROL_APPLIED = "applied"
CONTROL_ALREADY_PAUSED = "already_paused"
CONTROL_NOT_PAUSED = "not_paused"
#: The run has already ended (converged, stopped, errored, or previously
#: aborted). Deliberately **not** a park: a converged run must not have its
#: setpoint dropped by a control request that arrived after it finished.
CONTROL_ENDED = "ended"


# ---------------------------------------------------------------------------
# Loop state machine
# ---------------------------------------------------------------------------

class LoopState(Enum):
    """Lifecycle states of the autonomous loop."""

    IDLE = auto()
    SUGGESTING = auto()
    AWAITING_APPROVAL = auto()
    EXECUTING = auto()
    ANALYZING = auto()
    AWAITING_BOARD = auto()
    CONVERGED = auto()
    STOPPED = auto()
    ERROR = auto()


#: States the loop may not leave once entered.
#:
#: These are exactly the three states the main ``while`` in :meth:`AutonomousLoop.run`
#: already treats as the end of the run, and exactly the three that
#: ``autonomous_wiring`` maps to durable run statuses ("converged" / "stopped" /
#: "error").  The guard in :meth:`AutonomousLoop._set_state` therefore introduces no
#: new vocabulary — it enforces the one the loop had already written down in two
#: places and relied on in neither.
TERMINAL_STATES = frozenset({
    LoopState.CONVERGED,
    LoopState.STOPPED,
    LoopState.ERROR,
})


class BoardDecision(Enum):
    """Outcome of a board-exchange prompt when an electrode plate fills."""

    PROCEED = auto()   # a fresh board is in place — continue the campaign
    CANCEL = auto()    # manual override — stop the run (e.g. unintended overflow)


class BoardCheck(Enum):
    """Operator's answer when a resumed campaign finds recorded well occupancy.

    Drop-cast wells are single-use, so casting into an occupied well would waste
    the sample.  At campaign start (across GUI sessions) the current board may
    already have recorded casts; the operator confirms whether the plate is
    fresh or the same one still in the machine.
    """

    FRESH = auto()     # a new/replaced board — start clean on the next board id
    RESUME = auto()    # same board — continue past the already-used electrodes
    CANCEL = auto()    # abort before casting


# ---------------------------------------------------------------------------
# Convergence criteria
# ---------------------------------------------------------------------------

def default_convergence_check(
    history: list[tuple[dict[str, Any], float]],
    *,
    patience: int = 5,
    rel_tol: float = 1e-3,
) -> bool:
    """Return ``True`` when the objective has plateaued.

    Checks whether the best objective value has not improved by more
    than *rel_tol* (relative) over the last *patience* observations.
    """
    if len(history) < patience + 1:
        return False
    recent = [v for _, v in history[-patience:]]
    best_overall = max(abs(v) for _, v in history)
    if best_overall == 0:
        return True
    spread = max(recent) - min(recent)
    return spread / best_overall < rel_tol


# ---------------------------------------------------------------------------
# Autonomous loop
# ---------------------------------------------------------------------------

class AutonomousLoop:
    """Async closed-loop optimization engine.

    Parameters
    ----------
    optimizer : BaseOptimizer
        Optimizer instance to drive the search.
    workflow_template : Workflow or None
        A single-iteration workflow used as the template for each trial. The
        loop injects suggested parameters into the workflow's ``variables`` and
        interpolates ``$var`` placeholders before execution. Mutually exclusive
        with ``workflow_builder`` — supply exactly one.
    workflow_builder : callable or None
        ``(params) -> Workflow`` — builds a *concrete* single-iteration workflow
        for each suggestion (no ``$var`` interpolation needed). This is the
        unified path: the caller builds the trial via the shared deposition
        engine with the suggested per-pump volumes already resolved.
    manager : InstrumentManager
        Hardware / mock instrument registry.
    data_store : DataStore
        Experiment database for recording DOE observations.
    run_id : str
        Active experiment run identifier.
    objective_extractor : callable
        ``(step_results: dict[str, Any]) -> float`` — extracts a scalar
        objective from the results produced by one loop iteration.
    auto_approve : bool
        If ``True``, skip the approval gate and execute immediately.
    convergence_fn : callable or None
        ``(history) -> bool``.  Defaults to :func:`default_convergence_check`.
    """

    def __init__(
        self,
        optimizer: BaseOptimizer,
        workflow_template: Workflow | None,
        manager: InstrumentManager,
        data_store: DataStore,
        run_id: str,
        # May return None to mean "not measured" (never told to the optimizer).
        objective_extractor: Callable[[dict[str, Any]], "float | None"],
        *,
        workflow_builder: Callable[[dict[str, Any]], Workflow] | None = None,
        auto_approve: bool = False,
        convergence_fn: Callable[
            [list[tuple[dict[str, Any], float]]], bool
        ] | None = None,
        max_iterations: int | None = None,
        batch_size: int = 1,
        batch_workflow_builder: Callable[[list[dict[str, Any]]], Workflow] | None = None,
        batch_objective_extractor: Callable[[dict[str, Any], int, dict[str, Any]], "float | None"] | None = None,
        batch_channels: list[int] | None = None,
        electrode_allocator: ElectrodeAllocator | None = None,
        placement_workflow_builder: Callable[[list[dict[str, Any]], list[int]], Workflow] | None = None,
        placement_objective_extractor: Callable[[dict[str, Any], int, dict[str, Any]], "float | None"] | None = None,
        on_board_exchange: Callable[[int], "BoardDecision | bool | Awaitable[BoardDecision | bool]"] | None = None,
        equilibration_builder: Callable[[], Workflow] | None = None,
        track_occupancy: bool = False,
        # `(channel) -> sample_uuid | None` for the round currently being cast.
        # A *lookup*, not a minter: identity is minted by whoever builds the
        # trial (that layer is the one that knows a well is being consumed), and
        # this loop only needs to copy it onto the occupancy row it writes — the
        # one row in the sample-identity spine the builder cannot write itself,
        # because the allocator's board index lives here. Defaulting to None
        # keeps the loop usable with no notion of samples at all, which T2.7's
        # analysis-only modalities depend on.
        sample_uuid_for: Callable[[int], str | None] | None = None,
        continue_on_error: bool = True,
        max_channel_retries: int = 1,
        park_after_failed_trials: int = 3,
        gate_timeout_s: float | None = DEFAULT_GATE_TIMEOUT_S,
    ) -> None:
        if workflow_template is None and workflow_builder is None:
            raise ValueError(
                "AutonomousLoop needs either workflow_template or workflow_builder"
            )
        if batch_size > 1 and electrode_allocator is None and (
            batch_workflow_builder is None or batch_objective_extractor is None
        ):
            raise ValueError(
                "batch_size > 1 requires batch_workflow_builder and "
                "batch_objective_extractor"
            )
        if electrode_allocator is not None and (
            placement_workflow_builder is None or placement_objective_extractor is None
        ):
            raise ValueError(
                "electrode_allocator requires placement_workflow_builder and "
                "placement_objective_extractor"
            )
        self._optimizer = optimizer
        self._template = workflow_template
        self._builder = workflow_builder
        self._manager = manager
        self._data_store = data_store
        self._run_id = run_id
        self._extract_objective = objective_extractor
        self._auto_approve = auto_approve
        self._convergence_fn = convergence_fn or default_convergence_check
        # Hard trial budget. Needed for open-ended optimizers (e.g. Bayesian)
        # whose ``suggest()`` never returns None; grid/random self-exhaust.
        self._max_iterations = max_iterations

        # q-batch BO: propose q distinct points per round, one per channel, in a
        # single physical run; each is scored against its own channel's result.
        self._batch_size = max(1, int(batch_size))
        self._batch_builder = batch_workflow_builder
        self._batch_extract = batch_objective_extractor
        self._batch_channels = batch_channels

        # Electrode/board management: single-use electrodes allocated sequentially;
        # a full board triggers a prompted exchange (+ equilibration) mid-round.
        self._allocator = electrode_allocator
        self._placement_builder = placement_workflow_builder
        self._placement_extract = placement_objective_extractor
        self._on_board_exchange = on_board_exchange
        self._equilibration_builder = equilibration_builder
        # Persist single-use well occupancy (board_id, electrode) so a resumed
        # session can warn before re-casting into an already-used well.
        self._track_occupancy = track_occupancy
        self._sample_uuid_for = sample_uuid_for

        # --- Fault tolerance (retry, then park) ---
        # The executor already implements per-channel replay/skip recovery; it was
        # simply never enabled for campaigns (only the HT tab opted in), so any
        # step failure ended the run. Enabling it makes a single bad channel
        # survivable — and ``park_after_failed_trials`` is what keeps that from
        # becoming the opposite failure mode, silently burning a whole board on a
        # systematic fault.
        #
        # **Named to pair with ``[safety] rh_ceiling_park_after_trials``, and to
        # be unmistakable for ``[instruments.rh_controller].max_consecutive_failures``**,
        # which is a *sensor soft-reset* threshold of 5 and has nothing to do with
        # trials or parking. Under the old shared spelling, wiring this limit up
        # by the obvious config key would have silently given a campaign five
        # trials' rope instead of three, with nothing in either file to say so.
        # This limit is deliberately **not** config-resolved at all today; if that
        # ever changes it belongs in ``[safety]`` beside its sibling.
        self._continue_on_error = continue_on_error
        self._max_channel_retries = max_channel_retries
        self._park_after_failed_trials = park_after_failed_trials
        self._consecutive_failures = 0
        self._park_reason: str | None = None
        self._gate_timeout_s = gate_timeout_s

        self._state = LoopState.IDLE
        self._pending_params: dict[str, Any] | None = None
        self._iteration = 0
        self._approval_event = asyncio.Event()

        # --- Campaign controls (stage 4) ------------------------------------
        # The trial currently in flight, so a control request has something to
        # reach. It was a local in `_run_workflow` and dropped on return, which
        # meant that even in-process there was nothing to pause or abort: the
        # loop could decline the *next* trial and nothing else.
        self._executor: Any | None = None
        # Pause is deliberately **not** a `LoopState`. The three terminal states
        # are decisions the run has already taken and `_set_state` latches them;
        # a pause is by definition leaveable, so encoding it as a state would
        # either need a hole in that latch or would overwrite `EXECUTING` with
        # something the trial then has to restore. It is an orthogonal axis.
        self._pause_requested = False
        self._paused = False
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._abort_requested = False
        self._abort_reason: str | None = None
        # True only between entering and leaving `run()`. An Abort that arrives
        # outside that window has nothing to stop and must not park — see
        # :meth:`abort`.
        self._running = False
        # Fired as `(phase, detail)` with *phase* one of "requested",
        # "holding", "deferred", "resumed" — so an operator learns not only that
        # a Pause was heard but *where* it came to rest.
        self.on_pause_change: Callable[[str, str], Any] | None = None

        # --- Callbacks ---
        self.on_state_change: Callable[[LoopState, LoopState], Any] | None = None
        self.on_suggestion: Callable[[int, dict[str, Any]], Any] | None = None
        self.on_result: Callable[[int, dict[str, Any], float], Any] | None = None
        self.on_converged: Callable[[int, tuple[dict[str, Any], float] | None], Any] | None = None
        # Fired when a board fills and an exchange is requested (board_index,
        # remaining_samples) — a notification hook distinct from the decision
        # gate ``on_board_exchange`` (which returns PROCEED/CANCEL).
        self.on_board_exchange_requested: Callable[[int, int], Any] | None = None
        # Recovery visibility: a step was retried (step_name, error, attempt) or a
        # channel was abandoned (step_name, reason). Without these, graceful
        # recovery is invisible — the run looks healthy while silently losing wells.
        self.on_step_recovered: Callable[[str, str, int], Any] | None = None
        self.on_step_skipped: Callable[[str, str], Any] | None = None
        # Fired when the loop parks (reason). The caller drives safe_park + alerts.
        self.on_park: Callable[[str], Any] | None = None
        # Fired after every completed iteration (P3.2). The caller serializes the
        # spec + optimizer and writes the resume point; the loop deliberately
        # knows nothing about how a checkpoint is stored.
        self.on_checkpoint: Callable[[int], Any] | None = None
        # Run alongside any step tagged as a purge window (P8), and joined before
        # the run proceeds. The loop just forwards it to each trial's executor so
        # the wiring lives in one place rather than at every host.
        self.on_purge_window: Callable[[Any], Any] | None = None
        # Waste container ledger (P5.4). Forwarded to each trial's executor so
        # flushes book themselves as they run rather than the level only moving
        # when an operator edits it — waste is a wall-clock cap on unattended
        # time, and a drifting number is one nobody trusts.
        self.waste_ledger: Any | None = None
        # Fired after a workflow's steps complete and **before** the objective is
        # extracted, with the raw ``{step_name: result}`` in hand; returns the
        # (possibly augmented) results. Async because its one production use —
        # T3.1's confirmation sweeps — has to await a follow-up workflow, which is
        # impossible from the objective extractor (a synchronous callable invoked
        # from a non-awaited site).
        #
        # The loop stays measurement-agnostic: it knows only that something may
        # want to look at raw results and may hand back more of them. What counts
        # as a bad measurement, and what to do about it, lives in the wiring.
        self.on_trial_measured: (
            Callable[[dict[str, Any]], Any] | None) = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def consecutive_failures(self) -> int:
        """Trials failed back-to-back, for the checkpoint that must persist it."""
        return self._consecutive_failures

    # ── Iteration advance + checkpoint (P3.2) ───────────────────────────

    def _advance_iteration(self) -> None:
        """Close out an iteration and write its resume point.

        Every path that finishes a trial goes through here — success, aborted,
        errored, and unmeasured alike — because the well was consumed in all of
        them. A resume that replayed a failed trial would re-cast a used well.
        Routing all advances through one helper is what keeps a future code path
        from silently skipping the checkpoint.

        Failure paths reach it **through** :meth:`_note_trial_failure`, which
        calls it after incrementing the streak so the checkpoint carries the new
        count; see that method for why the order is load-bearing.
        """
        self._iteration += 1
        self._checkpoint()

    def _checkpoint(self) -> None:
        """Invoke the checkpoint handler; never let it break a run.

        A campaign that is otherwise healthy must not die because the resume
        point could not be written — the failure costs resumability, not the
        experiment, so it is logged and swallowed.
        """
        if self.on_checkpoint is None:
            return
        try:
            self.on_checkpoint(self._iteration)
        except Exception:
            logger.warning(
                "checkpoint_failed", iteration=self._iteration, exc_info=True
            )

    # ── State management ────────────────────────────────────────────────

    def _set_state(self, new: LoopState) -> None:
        """Transition to *new*, **refusing to leave a terminal state**.

        A terminal state is a decision the loop has already taken — stopped,
        converged, or errored — and every route into one is something that
        wanted the run to end.  Letting an ordinary transition overwrite it is
        how :meth:`stop` used to be discarded: the request lands while the loop
        is ``EXECUTING``, and the next statement on every round path is
        ``_set_state(ANALYZING)``, so the ``while`` never saw ``STOPPED`` and the
        run carried on with no error and nothing in the log but an ordinary
        state-change line.  Every ``_park`` site survived only because a
        hand-placed ``break`` followed it; the guard makes that structural rather
        than a property each call site has to remember.

        The refusal is **logged at warning**, never silent.  A swallowed state
        change is the hardest kind of defect to see from a log file, and a quiet
        guard would merely trade one invisible discard for another.  Use
        :meth:`_clear_terminal` to reopen the machine deliberately.
        """
        old = self._state
        if old in TERMINAL_STATES and new is not old:
            logger.warning(
                "loop_state_change_refused",
                current=old.name,
                attempted=new.name,
                msg="loop is in a terminal state — transition discarded",
            )
            return
        self._state = new
        logger.info("loop_state_change", old=old.name, new=new.name)
        if self.on_state_change:
            self.on_state_change(old, new)

    def _clear_terminal(self, new: LoopState = LoopState.IDLE) -> None:
        """Deliberately reopen a loop that has reached a terminal state.

        The guard in :meth:`_set_state` is a latch, and a latch with no release
        is a trap for whoever first tries to restart a loop object: their
        transition would simply vanish, which is the exact failure the guard
        exists to end.  Nothing in production restarts a loop today — each
        campaign builds a fresh one — so this is the explicit, logged door for
        the restartable loop rather than a facility in current use.

        ``park_reason`` is cleared with the state: a reopened loop that still
        reported why it parked would make the *next* run look parked, and the
        CLI's exit code reads exactly that pair (state plus ``park_reason``).
        """
        old = self._state
        self._state = new
        self._park_reason = None
        logger.warning("loop_terminal_cleared", old=old.name, new=new.name)
        if self.on_state_change:
            self.on_state_change(old, new)

    # ── Control ─────────────────────────────────────────────────────────

    def approve(self) -> None:
        """Signal approval for the pending suggestion."""
        self._approval_event.set()

    def stop(self) -> None:
        """Request the loop to stop after the current cycle.

        ``STOPPED`` is terminal (see :meth:`_set_state`), so the request holds
        even when it lands mid-trial: the round finishes analysing what it
        already cast, and no further suggestion is made.
        """
        self._set_state(LoopState.STOPPED)

    # ── Campaign controls: Pause / Resume / Abort (stage 4) ─────────────
    #
    # Three scopes exist and only two of them are here. **E-Stop is rig-scale
    # and is not a campaign control** — it lives on the main toolbar and stops
    # everything. Abort and Pause are campaign-scoped, and the distinction
    # between them is the whole design:
    #
    #   Pause  — *"stop issuing new steps, then hold in a safe state"*.
    #            Resumable, keeps the anneal, touches **no** setpoint, no lamp,
    #            no head, and never reaches `safe_park` or `_stop_wait`.
    #            Step-granularity latency is the specification, not a shortfall.
    #   Abort  — terminal for this campaign, **and it parks**. It is the one
    #            that may cut into an eight-hour hold.
    #
    # They are separate methods rather than one `halt(kind)` because the two
    # differ in every axis that matters — retract policy, checkpoint policy,
    # whether the rig is made safe — and a single entry point would have to
    # infer those from an argument. `halt_and_park_scope.md` records the same
    # conclusion for the four stop events that already existed.

    @property
    def is_paused(self) -> bool:
        """Whether a pause has been *requested*. Not the same as *held*."""
        return self._pause_requested

    def pause(self, reason: str = "") -> str:
        """Stop issuing new steps and hold at the next safe interruption.

        Returns one of the ``CONTROL_*`` outcomes; never raises.

        **What this does not do.** It does not call ``safe_park``, does not
        write a setpoint, does not touch the lamp and does not move the head.
        ``safe_park`` drops the setpoint to ``DEFAULT_SAFE_TEMP_C = 10.0`` and
        turns the lamp off — routing a Pause through it would destroy the anneal
        the Pause exists to preserve. It does not set the temperature
        controller's ``_stop_wait`` either, for the same reason: that flag ends
        a hold, and Pause exists to keep one.

        **Where the hold lands.** Not here. This records the request; the hold
        is taken at the first step boundary whose rig pose
        (:func:`~softae.core.rig_pose.safe_to_interrupt`) a purge would also
        accept, and failing that at the top of the next cycle — which always
        qualifies, because a trial's teardown has run by then. So a Pause during
        an eight-hour anneal takes effect when the anneal ends. That is the
        specified behaviour: an operator who wants a long hold cut short wants
        Abort.

        Unbounded on purpose. Every other wait in this loop has a ceiling
        because "nobody answered" is indistinguishable from "still working"; a
        pause is an explicit operator act, and auto-resuming a rig somebody
        deliberately stopped would be the worse failure. The heartbeat keeps
        ticking throughout, so a paused campaign never looks wedged.
        """
        if self._abort_requested or self._state in TERMINAL_STATES:
            return CONTROL_ENDED
        if self._pause_requested:
            return CONTROL_ALREADY_PAUSED
        self._pause_requested = True
        self._paused = False
        self._resume_event.clear()
        logger.warning("loop_pause_requested", iteration=self._iteration,
                       reason=reason or "operator request")
        self._notify_pause("requested", reason or "operator request")
        return CONTROL_APPLIED

    def resume(self) -> str:
        """Leave a pause. The exact inverse of :meth:`pause`.

        Nothing is re-driven and nothing is re-initialised: the trial resumes at
        the step it was holding before, or the loop proceeds to the next cycle.
        The iteration counter and the checkpoint are untouched by the round
        trip, which is what makes "resume" mean *continue* rather than *restart*.
        """
        if not self._pause_requested:
            return CONTROL_NOT_PAUSED
        self._pause_requested = False
        self._paused = False
        self._resume_event.set()
        executor = self._executor
        if executor is not None:
            try:
                executor.resume()
            except Exception:
                logger.warning("executor_resume_failed", exc_info=True)
        logger.warning("loop_resumed", iteration=self._iteration)
        self._notify_pause("resumed", "operator request")
        return CONTROL_APPLIED

    def abort(self, reason: str = "operator abort") -> str:
        """End this campaign and park the rig. Terminal, and it cuts in.

        Three stages, in this order, none of which blocks:

        1. ``executor.abort()`` — the next step is refused;
        2. every instrument's ``_stop_wait`` is set — a watched hold returns
           within one poll instead of at the end of eight hours;
        3. every gate is released — an abort issued while paused, or while
           waiting on approval, must not wait for the thing it is ending.

        **The park is deliberately not stage 4 of this method.** It happens on
        the loop's own thread of control, in :meth:`run`, once the trial has
        actually stopped — see the park site there for why parking a rig whose
        current step is still mid-flight is not the same as parking a stopped
        one.

        **Stage 1 precedes stage 2, which reverses the order the spec gave**,
        because the executor must already be ``ABORTED`` at the moment the hold
        breaks. In the other order the interrupted anneal returns first, the
        executor is still ``RUNNING``, and a step that raises on the way out
        (the real temp controller refuses every command while ``_stop_wait`` is
        set) can enter the channel-recovery path before the abort is seen.

        Returns one of the ``CONTROL_*`` outcomes; never raises.
        """
        if self._abort_requested:
            return CONTROL_ENDED
        if self._state in TERMINAL_STATES or not self._running:
            # The run is over. **Do not park.** A converged or already-stopped
            # campaign has left the rig in a state somebody chose — possibly a
            # deliberately head-down rest with a wet tip — and dropping the
            # setpoint and killing the lamp underneath that would be a control
            # request doing harm after the thing it controls has gone.
            logger.warning("loop_abort_after_end", state=self._state.name,
                           reason=reason)
            return CONTROL_ENDED

        self._abort_requested = True
        self._abort_reason = reason
        logger.error("loop_abort_requested", iteration=self._iteration,
                     reason=reason)

        executor = self._executor
        if executor is not None:
            try:
                executor.abort()
            except Exception:
                logger.warning("executor_abort_failed", exc_info=True)

        self._set_stop_wait()

        # Release everything that could be blocking. `_resume_event` frees a
        # paused loop; `_approval_event` frees an approval gate, whose caller
        # then re-checks the abort flag rather than executing the trial it was
        # holding.
        self._resume_event.set()
        self._approval_event.set()
        return CONTROL_APPLIED

    # ── Control internals ───────────────────────────────────────────────

    def _notify_pause(self, phase: str, detail: str) -> None:
        if self.on_pause_change is None:
            return
        try:
            self.on_pause_change(phase, detail)
        except Exception:
            logger.warning("on_pause_change_failed", exc_info=True)

    #: The mid-hold abort flags this system already has, by attribute name.
    #:
    #: **Both, not just the thermal one.** ``TempEISSweep.abort`` — the shipped
    #: abort this is modelled on — sets ``_stop_wait`` on the temperature
    #: controller *and* ``_wait_abort`` on the RH controller, because a
    #: synchronous ``rh_controller.wait()`` is just as uninterruptible as a
    #: thermal hold and is running in the same thread pool. An abort wired to
    #: only one of them is an abort that works on the branch somebody tested.
    HOLD_ABORT_FLAGS = ("_stop_wait", "_wait_abort")

    def _hold_abort_events(self) -> list[Any]:
        """Every instrument's mid-hold abort flag, by duck type.

        Enumerated across the manager rather than reaching for
        ``"temp_controller"`` by name: the temperature instrument is
        configurable (``temp_eis_sweep`` and ``equilibration`` both take a
        ``temp_instrument`` name), and an abort that missed a renamed controller
        would be an eight-hour abort that looked like a working one.
        """
        events: list[Any] = []
        try:
            names = list(self._manager.names)
        except Exception:
            return events
        for name in names:
            try:
                inst = self._manager.get(name)
            except Exception:
                continue
            for attribute in self.HOLD_ABORT_FLAGS:
                event = getattr(inst, attribute, None)
                if event is not None and callable(getattr(event, "set", None)):
                    events.append(event)
        return events

    def _set_stop_wait(self) -> None:
        """Interrupt any watched hold, within one poll of its own cadence.

        ``run_anneal_hold`` already derives ``monitored_hold``'s ``should_abort``
        from this flag, and ``monitored_hold`` tests it at the top of every poll
        before sleeping. The mechanism is fully built and tested; its only
        consumers were the Arrhenius tab and the temperature sweep, and the
        campaign path never picked it up. This is that wiring — **not a second
        abort path**.
        """
        for event in self._hold_abort_events():
            try:
                event.set()
            except Exception:
                logger.warning("stop_wait_set_failed", exc_info=True)

    def _release_stop_wait(self) -> None:
        """Clear the flag, and clear it **before** the park, never after.

        The real controller's ``_with_retry`` raises ``CommunicationError`` on
        every command while this is set, and ``safe_park``'s whole contribution
        on the thermal axis is one ``write_sp(10 °C)``. Parking with the flag
        still set would therefore record ``temperature: ...`` in the park's
        errors and leave the heater exactly where the abort found it — a park
        that reports itself incomplete and leaves the rig hot overnight.

        Safe to clear here and nowhere earlier: this runs on the loop's own
        thread after the trial has already stopped, so the abort edge it exists
        to deliver has demonstrably been delivered. Clearing it from
        :meth:`abort` instead — set then immediately cleared — could fall
        entirely inside one poll interval and be missed.
        """
        for event in self._hold_abort_events():
            try:
                event.clear()
            except Exception:
                logger.warning("stop_wait_clear_failed", exc_info=True)

    def _hold_executor_if_quiescent(self) -> None:
        """Take a requested pause at this step boundary, if the pose allows.

        Called from ``on_step_complete``, which is the loop's existing view of a
        step boundary — so the pause lands *between* steps by construction
        rather than by a second mechanism agreeing to. The executor's own pause
        loop sits at the top of the next tier/step, so pausing from here holds
        before anything else runs.
        """
        if not self._pause_requested or self._paused:
            return
        executor = self._executor
        if executor is None:
            return
        from softae.core.rig_pose import classify_pose, safe_to_interrupt

        if not safe_to_interrupt(self._manager):
            # Not a refusal — a deferral. The top-of-cycle gate always
            # qualifies, so the worst case is that the pause lands one trial
            # later rather than one step later.
            self._notify_pause(
                "deferred",
                f"pose {classify_pose(self._manager).value} — holding later",
            )
            return
        try:
            executor.pause()
        except Exception:
            logger.warning("executor_pause_failed", exc_info=True)
            return
        self._paused = True
        logger.warning("loop_paused_at_step_boundary", iteration=self._iteration)
        self._notify_pause("holding", "at a step boundary")

    async def _pause_gate(self) -> None:
        """Hold at the top of a cycle while a pause is outstanding.

        *"Before next cycle/loop start"*, the second half of the operator's
        definition — and the backstop that makes the pose gate above a
        deferral rather than a refusal. Between cycles the previous trial's
        teardown has run, so this boundary is quiescent by construction and
        needs no pose read.

        An ``asyncio.Event`` rather than a poll: there is nothing to sample, and
        a spin loop here would be a busy wait for however long an operator takes
        to come back. Both :meth:`resume` and :meth:`abort` set it, which is why
        an abort can never be trapped behind a pause.
        """
        if not self._pause_requested:
            return
        if not self._paused:
            self._paused = True
            logger.warning("loop_paused_between_cycles", iteration=self._iteration)
            self._notify_pause("holding", "before the next cycle")
        while (
            self._pause_requested
            and not self._abort_requested
            and self._state not in TERMINAL_STATES
        ):
            await self._resume_event.wait()

    # ── Main loop ───────────────────────────────────────────────────────

    async def run(self) -> tuple[dict[str, Any], float] | None:
        """Execute the full suggest-execute-analyze-tell cycle.

        Returns the best ``(params, objective)`` or ``None`` on abort.
        """
        self._running = True
        try:
            return await self._run()
        finally:
            self._running = False

    async def _run(self) -> tuple[dict[str, Any], float] | None:
        self._set_state(LoopState.SUGGESTING)

        while self._state not in TERMINAL_STATES:
            # 0a. CONTROL — "before next cycle/loop start". Both halves of the
            # operator's definition of Pause meet here: this is the boundary
            # that always qualifies as safe, and it is where an Abort that
            # landed between trials leaves the loop.
            await self._pause_gate()
            if self._abort_requested:
                break

            # 0. BUDGET
            if self._max_iterations is not None and self._iteration >= self._max_iterations:
                logger.info("loop_budget_reached", iteration=self._iteration)
                break

            # BOARD-AWARE: single-use electrodes allocated sequentially; a round
            # (q>=1) may straddle a board exchange. Covers single-point (q=1) too.
            if self._allocator is not None:
                if not await self._run_board_aware_round(self._batch_size):
                    break
                continue

            # BATCH (q-BO): a whole round proposes/executes/tells q points.
            if self._batch_size > 1:
                if not await self._run_batch_round():
                    break
                continue

            # 1. SUGGEST
            self._set_state(LoopState.SUGGESTING)
            params = self._optimizer.suggest()
            if params is None:
                logger.info("optimizer_exhausted", iteration=self._iteration)
                break

            self._pending_params = params
            logger.info("loop_suggest", iteration=self._iteration, params=params)
            if self.on_suggestion:
                self.on_suggestion(self._iteration, params)

            # Record DOE row (objective_value=None until analyzed)
            doe_id = self._data_store.record_doe_parameter(
                run_id=self._run_id,
                channel=0,
                iteration=self._iteration,
                parameters=params,
            )

            # 2. APPROVAL GATE
            if not self._auto_approve:
                self._set_state(LoopState.AWAITING_APPROVAL)
                self._approval_event.clear()
                if not await self._await_gate(self._approval_event, "approval"):
                    break
                if self._state is LoopState.STOPPED:
                    break

            # 3. EXECUTE
            self._set_state(LoopState.EXECUTING)
            try:
                step_results = await self._execute_trial(params)
            except AbortedError:
                logger.warning("trial_aborted", iteration=self._iteration)
                self._set_state(LoopState.STOPPED)
                break
            except Exception as exc:
                logger.error("trial_error", iteration=self._iteration, error=str(exc))
                if self._is_hard_fault(exc):
                    self._park(f"hard fault: {type(exc).__name__}: {exc}")
                    break
                if self._note_trial_failure(f"execute: {type(exc).__name__}"):
                    break
                continue

            # 4. ANALYZE
            self._set_state(LoopState.ANALYZING)
            try:
                objective = self._extract_objective(step_results)
            except Exception as exc:
                logger.error("objective_extraction_error", iteration=self._iteration, error=str(exc))
                if self._note_trial_failure(f"analyze: {type(exc).__name__}"):
                    break
                continue

            # 5. TELL — never fabricate an observation for an unmeasured trial.
            if self._is_unmeasured(objective, params):
                if self._note_trial_failure("unmeasured"):
                    break
                continue
            self._note_trial_success()
            self._optimizer.tell(params, objective)
            self._data_store.update_doe_objective(doe_id, objective)
            logger.info(
                "loop_tell",
                iteration=self._iteration,
                objective=objective,
                best=self._optimizer.best(),
            )
            if self.on_result:
                self.on_result(self._iteration, params, objective)

            self._advance_iteration()

            # 6. CONVERGENCE CHECK
            if self._convergence_fn(self._optimizer.history):
                logger.info("loop_converged", iteration=self._iteration)
                self._set_state(LoopState.CONVERGED)
                if self.on_converged:
                    self.on_converged(self._iteration, self._optimizer.best())
                break

        # ── The park half of Abort ──────────────────────────────────────────
        # Sited here, and not inside `abort()`, because *here* the trial has
        # actually stopped: the executor has raised through, teardown has run,
        # and nothing is mid-command. `abort()` is called from the control
        # watcher's task while a step may still be in flight, and parking a rig
        # whose current step is still writing setpoints is a race — the park's
        # `write_sp(10 °C)` and the anneal's `write_sp(original)` would be two
        # writers with no defined winner.
        #
        # `_park` rather than a bare STOPPED, deliberately: it fires `on_park`,
        # which is what runs `safe_park` and raises the durable CRITICAL alert.
        # `stop()` only sets state and makes nothing safe.
        #
        # `park_reason` also decides the checkpoint's fate one layer up: a run
        # with a park reason keeps it. That is right for an operator Abort — the
        # campaign ended because somebody said so, not because it finished, and
        # being able to resume it is exactly why the checkpoint exists.
        if self._abort_requested and self._park_reason is None:
            self._release_stop_wait()
            self._park(self._abort_reason or "operator abort")

        best = self._optimizer.best()
        # Only a non-terminal exit (budget reached, optimizer exhausted) needs a
        # closing state. Asking for STOPPED from ERROR would now be refused, and
        # the refusal would be logged as a defect it is not.
        if self._state not in TERMINAL_STATES:
            self._set_state(LoopState.STOPPED)
        return best

    # ── Trial execution ─────────────────────────────────────────────────

    async def _execute_trial(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a single-iteration workflow for this suggestion and run it.

        Two paths produce the concrete trial workflow:

        * ``workflow_builder`` — the caller builds a fully concrete workflow from
          the suggested params (the unified engine path: per-pump volumes are
          already resolved, so no ``$var`` interpolation is needed);
        * ``workflow_template`` — suggested params are merged into the template's
          variables and interpolated into every step's ``$var`` placeholders.

        Either way, this is where a suggestion actually reaches the hardware.
        Returns a dict mapping step names to their raw results.
        """
        if self._builder is not None:
            base = self._builder(params)
            trial_wf = Workflow(
                name=f"{base.name}_trial{self._iteration}",
                description=base.description,
                variables={**base.variables, **params},
                setup=base.setup,
                loop_steps=base.loop_steps,
                teardown=base.teardown,
                iterate_over=base.iterate_over,
                iterations=1,
                metadata={
                    **base.metadata,
                    "autonomous_iteration": self._iteration,
                    "suggested_params": params,
                },
            )
        else:
            from softae.workflows.workflow_parser import interpolate_params

            variables = {**self._template.variables, **params}

            def _resolve(steps: list[WorkflowStep]) -> list[WorkflowStep]:
                return [
                    s.with_params(**interpolate_params(s.params, variables))
                    for s in steps
                ]

            trial_wf = Workflow(
                name=f"{self._template.name}_trial{self._iteration}",
                description=self._template.description,
                variables=variables,
                setup=_resolve(self._template.setup),
                loop_steps=_resolve(self._template.loop_steps),
                teardown=_resolve(self._template.teardown),
                iterate_over=self._template.iterate_over,
                iterations=1,
                metadata={
                    **self._template.metadata,
                    "autonomous_iteration": self._iteration,
                    "suggested_params": params,
                },
            )

        return await self._run_workflow(trial_wf)

    async def _run_workflow(self, trial_wf: Workflow) -> dict[str, Any]:
        """Execute a concrete workflow and collect step results by name.

        Runs with the executor's **graceful channel recovery** enabled: a
        recoverable fault (e.g. a wedged-stage timeout) replays that channel from
        its precondition when no elution was committed, or abandons just that
        channel — instead of killing the campaign. Steps that never complete are
        simply absent from ``results``, so the objective extractor returns
        ``None`` and :meth:`_is_unmeasured` skips that trial rather than telling
        the optimizer a fabricated value.
        """
        results: dict[str, Any] = {}

        executor = WorkflowExecutor(
            self._manager,
            data_store=self._data_store,
            run_id=self._run_id,
            continue_on_error=self._continue_on_error,
            max_channel_retries=self._max_channel_retries,
        )
        # The trial's executor, reachable for as long as the trial lasts. Held
        # on `self` rather than only as a local because a control request
        # arrives on another task and has to reach *this* executor — without
        # this, Pause and Abort could decline the next trial and nothing else.
        self._executor = executor

        # Executor passes (step, idx, total, result, elapsed); accept extras so
        # this stays robust to callback-signature growth.
        def _step_done(step, idx, total, result, *_) -> None:
            results[step.name] = result
            # A step boundary is the only place a Pause may take hold inside a
            # trial, and this is the loop's existing view of one.
            self._hold_executor_if_quiescent()

        executor.on_step_complete = _step_done
        # Anti-clog purge alongside co-runnable steps (P8). Set from the loop so
        # every campaign gets it, GUI or headless, without each host wiring it.
        executor.on_purge_window = self.on_purge_window
        executor.waste_ledger = self.waste_ledger

        def _recovered(step, error, attempt, *_):
            logger.warning(
                "step_recovered", iteration=self._iteration,
                step=getattr(step, "name", str(step)), attempt=attempt,
                error=str(error),
            )
            if self.on_step_recovered:
                self.on_step_recovered(getattr(step, "name", str(step)), str(error), attempt)

        def _skipped(step, index, total, reason, *_):
            logger.warning(
                "step_skipped", iteration=self._iteration,
                step=getattr(step, "name", str(step)), reason=str(reason),
            )
            if self.on_step_skipped:
                self.on_step_skipped(getattr(step, "name", str(step)), str(reason))

        executor.on_step_recover = _recovered
        executor.on_step_skipped = _skipped

        try:
            # An Abort that landed in the window between building this executor
            # and starting it would otherwise be lost: `abort()` reached the
            # previous trial's executor (or none), and this one starts fresh.
            if self._abort_requested:
                executor.abort()
            await executor.run(trial_wf)
        finally:
            # Dropped as soon as the trial ends. A stale handle would let the
            # *next* trial be paused by a request aimed at this one — or, worse,
            # let `resume()` release an executor that no longer exists.
            self._executor = None
        return await self._post_measure(results)

    async def _post_measure(self, results: dict[str, Any]) -> dict[str, Any]:
        """Let the caller inspect raw results (and add to them) before scoring.

        Sited here rather than at the three trial paths because every one of them
        — single-point, batched and placement — reaches results through
        :meth:`_run_workflow`, so one insertion covers all three and they cannot
        drift apart. The board-exchange equilibration also passes through, and
        correctly does nothing: it produces no measurement steps for a handler to
        recognise.

        **Never fatal, and never destructive.** A handler that raises, or returns
        something that is not a results mapping, leaves the trial exactly as it
        was: this is an observation seam, and a campaign must not die because the
        thing watching it did.
        """
        if self.on_trial_measured is None:
            return results
        try:
            out = self.on_trial_measured(results)
            if inspect.isawaitable(out):
                out = await out
            return out if isinstance(out, dict) else results
        except Exception:
            logger.warning("trial_measured_hook_failed",
                           iteration=self._iteration, exc_info=True)
            return results

    # ── Batch (q-BO) round ──────────────────────────────────────────────

    async def _run_batch_round(self) -> bool:
        """Run one batched round: suggest q → one workflow → tell q.

        Proposes q distinct points, casts one per channel in a single physical
        workflow, then scores and tells each against its own channel's result.
        Sets a terminal state and returns ``False`` when the loop should stop;
        ``True`` to continue.

        **The final round narrows to the budget** rather than overrunning it. This
        used to round ``max_iterations`` up to the next multiple of q, spending up
        to q-1 electrodes and their anneal time beyond what the operator asked for.
        """
        q = self._round_q(self._batch_size)
        if q < 1:
            logger.info("loop_budget_reached", iteration=self._iteration)
            return False
        if q < self._batch_size:
            logger.info("round_narrowed", requested=self._batch_size, actual=q)

        # 1. SUGGEST (batch)
        self._set_state(LoopState.SUGGESTING)
        batch = self._optimizer.suggest_batch(q)
        if not batch:
            logger.info("optimizer_exhausted", iteration=self._iteration)
            return False

        for k, params in enumerate(batch):
            if self.on_suggestion:
                self.on_suggestion(self._iteration + k, params)
        logger.info("loop_suggest_batch", iteration=self._iteration, q=len(batch))

        # Record a DOE row per batch member (tagged with its electrode channel).
        doe_ids = [
            self._data_store.record_doe_parameter(
                run_id=self._run_id,
                channel=(self._batch_channels[k] if self._batch_channels else 0),
                iteration=self._iteration + k,
                parameters=params,
            )
            for k, params in enumerate(batch)
        ]

        # 2. APPROVAL GATE (once for the whole round)
        if not self._auto_approve:
            self._set_state(LoopState.AWAITING_APPROVAL)
            self._approval_event.clear()
            if not await self._await_gate(self._approval_event, "approval"):
                return False
            if self._state is LoopState.STOPPED:
                return False

        # 3. EXECUTE (one physical run for the q-point batch)
        self._set_state(LoopState.EXECUTING)
        try:
            base = self._batch_builder(batch)
            trial_wf = Workflow(
                name=f"{base.name}_batch{self._iteration}",
                description=base.description,
                variables=dict(base.variables),
                setup=base.setup,
                loop_steps=base.loop_steps,
                teardown=base.teardown,
                iterate_over=base.iterate_over,
                iterations=1,
                metadata={**base.metadata, "autonomous_iteration": self._iteration},
            )
            step_results = await self._run_workflow(trial_wf)
        except AbortedError:
            logger.warning("trial_aborted", iteration=self._iteration)
            self._set_state(LoopState.STOPPED)
            return False
        except Exception as exc:
            logger.error("trial_error", iteration=self._iteration, error=str(exc))
            if self._is_hard_fault(exc):
                self._park(f"hard fault: {type(exc).__name__}: {exc}")
                return False
            # A whole round failed to execute; count it once and move on unless
            # the failures look systematic. Every well in the round was still
            # consumed, so each advances through _advance_iteration — a bare
            # ``+= len(batch)`` here skipped the checkpoint, and a resume from
            # the stale point would have re-cast the whole round's used wells.
            return not self._note_trial_failure(
                f"execute: {type(exc).__name__}", wells=len(batch))

        # 4/5. ANALYZE + TELL each batch member against its own channel result.
        self._set_state(LoopState.ANALYZING)
        for k, params in enumerate(batch):
            try:
                objective = self._batch_extract(step_results, k, params)
            except Exception as exc:
                logger.error(
                    "objective_extraction_error",
                    iteration=self._iteration, error=str(exc),
                )
                if self._note_trial_failure(f"analyze: {type(exc).__name__}"):
                    return False
                continue
            if self._is_unmeasured(objective, params):
                if self._note_trial_failure("unmeasured"):
                    return False
                continue
            self._note_trial_success()
            self._optimizer.tell(params, objective)
            self._data_store.update_doe_objective(doe_ids[k], objective)
            if self.on_result:
                self.on_result(self._iteration, params, objective)
            self._advance_iteration()

            # 6. CONVERGENCE (per evaluation, so a round can converge mid-batch)
            if self._convergence_fn(self._optimizer.history):
                logger.info("loop_converged", iteration=self._iteration)
                self._set_state(LoopState.CONVERGED)
                if self.on_converged:
                    self.on_converged(self._iteration, self._optimizer.best())
                return False

        return True

    # ── Board-aware round (sequential electrodes + board exchange) ───────

    def _round_q(self, requested: int) -> int:
        """Clamp a round's width to the budget still unspent.

        A round is atomic in the sense that every point it suggests is cast, so
        suggesting more than the budget allows would *overrun* it — the loop used
        to round ``max_iterations`` up to the next multiple of q, spending up to
        q-1 extra electrodes and their anneal time on a campaign the operator had
        already bounded. Shrinking the final round spends exactly the budget.
        """
        if self._max_iterations is None:
            return max(0, requested)
        return max(0, min(requested, self._max_iterations - self._iteration))

    async def _exchange_board(self) -> bool:
        """Swap to a fresh plate. ``False`` means the run is over.

        Advances the allocator first but persists the pointer only after the
        operator confirms: persisting before the prompt would leave the pointer on
        a plate that was never installed if they cancel, and the next session would
        then cast into the old board's occupied wells.
        """
        new_board = self._allocator.swap_board()
        self._set_state(LoopState.AWAITING_BOARD)
        logger.info("board_full", board=new_board)

        decision = await self._request_board_exchange(new_board, 0)
        if decision is BoardDecision.CANCEL:
            logger.warning("board_exchange_cancelled", board=new_board)
            self._set_state(LoopState.STOPPED)
            return False

        if self._track_occupancy:
            try:
                self._data_store.set_active_board(new_board)
            except Exception:
                logger.warning("active_board_persist_failed", exc_info=True)

        # Equilibrate the fresh board before casting on it (best-effort).
        if self._equilibration_builder is not None:
            try:
                self._set_state(LoopState.EXECUTING)
                await self._run_workflow(self._equilibration_builder())
            except Exception:
                logger.warning("equilibration_failed", exc_info=True)
        return True

    def _sample_uuid(self, channel: int) -> str | None:
        """The identity of the sample just cast into *channel*, if any.

        Guarded **separately** from the occupancy write it feeds. Occupancy is a
        safety record — a well not marked occupied can be re-cast into, ruining
        both samples — so a broken identity lookup must cost the uuid and not the
        row. Wrapping both together would let the more decorative failure
        suppress the more important write.
        """
        if self._sample_uuid_for is None:
            return None
        try:
            return self._sample_uuid_for(int(channel))
        except Exception:
            logger.warning("sample_uuid_lookup_failed", channel=channel,
                           exc_info=True)
            return None

    async def _run_board_aware_round(self, q: int) -> bool:
        """Run one round of q suggestions on **one** board. Returns ``False`` to stop.

        The round is sized *before* anything is suggested, to the smallest of the
        requested q, the electrodes still free on the current board, and the budget
        still unspent. A board with nothing left is exchanged first. A round is
        therefore always cast, measured and told on a single plate.

        **Sizing down beats splitting.** The previous design suggested a full q and
        then straddled the exchange, casting what fit and holding those wells
        through an operator prompt of unbounded duration — wet films sitting on a
        plate half-measured while the rig waits for someone to walk over. It also
        made the batch's own diversification a fiction: constant-liar picks q points
        to be informative *together*, and a round split across a swap tells them
        after an arbitrary gap during which the plate, the humidity and the operator
        have all changed. A narrower round that completes is worth more than a wide
        one that is interrupted.
        """
        # 0. Budget first — an exhausted campaign must not prompt for a fresh plate
        # it will never cast on. Only then exchange a full board, before suggesting,
        # so no cast is ever stranded by the prompt.
        wanted = self._round_q(q)
        if wanted < 1:
            logger.info("loop_budget_reached", iteration=self._iteration)
            return False
        if self._allocator.board_full and not await self._exchange_board():
            return False

        q_round = min(wanted, self._allocator.remaining)
        if q_round < 1:
            logger.info("optimizer_exhausted", iteration=self._iteration)
            return False
        if q_round < q:
            logger.info("round_narrowed", requested=q, actual=q_round,
                        board_remaining=self._allocator.remaining,
                        board=self._allocator.board_index)

        # 1. SUGGEST (q>=1; q=1 is the single-point case)
        self._set_state(LoopState.SUGGESTING)
        batch = self._optimizer.suggest_batch(q_round)
        if not batch:
            logger.info("optimizer_exhausted", iteration=self._iteration)
            return False
        for k, params in enumerate(batch):
            if self.on_suggestion:
                self.on_suggestion(self._iteration + k, params)

        # 2. APPROVAL GATE (once for the round)
        if not self._auto_approve:
            self._set_state(LoopState.AWAITING_APPROVAL)
            self._approval_event.clear()
            if not await self._await_gate(self._approval_event, "approval"):
                return False
            if self._state is LoopState.STOPPED:
                return False

        # 3. PLACE — one chunk, which fits on this board by construction.
        alloc = self._allocator.allocate(len(batch))
        batch = batch[: len(alloc.channels)]
        if not batch:
            logger.info("optimizer_exhausted", iteration=self._iteration)
            return False
        self._set_state(LoopState.EXECUTING)
        try:
            wf = self._placement_builder(batch, alloc.channels)
            round_cache = await self._run_workflow(wf)
        except AbortedError:
            logger.warning("trial_aborted", iteration=self._iteration)
            self._set_state(LoopState.STOPPED)
            return False
        except Exception as exc:
            logger.error("trial_error", iteration=self._iteration, error=str(exc))
            if self._is_hard_fault(exc):
                self._park(f"hard fault: {type(exc).__name__}: {exc}")
                return False
            # The allocator already spent these wells (allocate() runs before the
            # cast), so the budget and the resume point must account for them
            # even though nothing was measured — mirroring the batch round. The
            # failure itself is still counted once for the whole round.
            return not self._note_trial_failure(
                f"execute: {type(exc).__name__}", wells=len(alloc.channels))

        # Persist single-use occupancy for the wells just cast (keyed by this
        # board's id, so a later session can detect re-casts). The sample uuid,
        # when the builder minted one, makes the row say *which* sample occupies
        # the well rather than only that something does.
        if self._track_occupancy:
            for ch in alloc.channels:
                try:
                    self._data_store.record_electrode_cast(
                        alloc.board_index, ch,
                        run_id=self._run_id, iteration=self._iteration,
                        sample_uuid=self._sample_uuid(ch),
                    )
                except Exception:
                    logger.warning("occupancy_record_failed", exc_info=True)

        # 4. ANALYZE + TELL the whole round.
        self._set_state(LoopState.ANALYZING)
        should_stop = self._tell_placed(round_cache, list(zip(batch, alloc.channels)))
        return not should_stop

    async def _request_board_exchange(self, board_index: int, remaining: int) -> BoardDecision:
        """Ask the caller to swap the board; **CANCEL when no handler is supplied.**

        Replacing an electrode plate is a physical act that no software can
        perform, so "no handler" cannot mean "assume it happened".  It previously
        defaulted to ``PROCEED``, which made a headless run continue casting onto
        a board that was still full — overwriting occupied, single-use wells and
        ruining the plate.  Stopping cleanly at the board boundary is the only
        safe default; a caller that can genuinely service an exchange must say so
        by passing a handler.
        """
        if self.on_board_exchange_requested:
            self.on_board_exchange_requested(board_index, remaining)
        if self._on_board_exchange is None:
            logger.warning(
                "board_exchange_no_handler",
                board=board_index, remaining=remaining,
                msg="no exchange handler — stopping instead of assuming a fresh plate",
            )
            return BoardDecision.CANCEL
        result = self._on_board_exchange(board_index)
        if hasattr(result, "__await__"):
            # Bound the wait: nobody may be at the rig to swap the plate.
            # NOTE: only an *awaitable* handler can be bounded here. A handler
            # that blocks synchronously (e.g. a GUI modal waiting on a
            # threading.Event) stalls the event loop, so it must impose its own
            # timeout — see the Live BO tab's gates.
            if self._gate_timeout_s is None:
                result = await result
            else:
                try:
                    result = await asyncio.wait_for(result, timeout=self._gate_timeout_s)
                except asyncio.TimeoutError:
                    self._park(
                        f"board-exchange gate timed out after "
                        f"{self._gate_timeout_s:.0f}s with no response"
                    )
                    return BoardDecision.CANCEL
        if isinstance(result, BoardDecision):
            return result
        return BoardDecision.PROCEED if result else BoardDecision.CANCEL

    # ── Fault handling (retry, then park) ───────────────────────────────

    def _is_hard_fault(self, exc: BaseException) -> bool:
        """``True`` for fault classes where retrying cannot help or would harm.

        Safety violations and an unarmed interlock are refusals, not glitches:
        the same command will be refused again, and for a reservoir hard-stop
        (a mechanical dead-end) retrying is actively dangerous.  These park
        immediately without consuming the retry budget.
        """
        from softae.core.hardware_safety import HardwareNotArmedError
        from softae.errors import SafetyError

        return isinstance(exc, (SafetyError, HardwareNotArmedError))

    def _park(self, reason: str) -> None:
        """Enter the terminal parked state, recording *why*.

        Hardware is made safe by the caller's ``on_park`` handler (which calls
        :func:`softae.core.safe_park.safe_park`) — the loop's job is to stop and
        say why, not to drive instruments directly.
        """
        self._park_reason = reason
        logger.error("loop_parked", iteration=self._iteration, reason=reason)
        if self.on_park:
            try:
                self.on_park(reason)
            except Exception:
                logger.warning("on_park_failed", exc_info=True)
        self._set_state(LoopState.STOPPED)

    def _note_trial_failure(self, what: str, *, wells: int = 1) -> bool:
        """Count a failed/unmeasured trial, close it out; ``True`` → park.

        Only failures that already survived the executor's own retries reach
        here, so a single flaky step never parks a run — but a systematic fault
        stops burning wells after ``park_after_failed_trials``.

        **The three steps are in this order on purpose, and the ordering is why
        the advance lives here** rather than at the nine call sites that used to
        do it themselves:

        1. *count* — so that
        2. *checkpoint* (via :meth:`_advance_iteration`, once per well the failed
           trial consumed) persists the **incremented** streak. Checkpointing
           first would write ``n-1`` and a crash-restart loop would then restore
           the counter to the value it had before every failure, reproducing
           exactly the bug persistence exists to fix; and
        3. *park* last, so the resume point is on disk before the run stops.

        **This is one of three consecutive-failure counters, deliberately.**
        ``RHCeilingEscalation`` (``core/autonomous_wiring.py``) counts RH-decided
        equilibrate phases at campaign-wiring level, and
        ``_consecutive_channel_failures`` (``workflows/workflow_executor.py``)
        counts channel faults inside one workflow. They are **not** to be merged:
        they sit at three layers, escalate differently (two park directly, the
        executor's prompts-and-holds first), and one abstraction across loop /
        wiring / executor would couple layers that are meant to stay separate.
        **That decision stands; do not merge them.**

        They were also once expected to share their *treatment on resume*, and
        they no longer do. This counter is restored only when the previous run
        stopped in a way somebody was told about, and cleared otherwise; the RH
        ceiling streak is restored unconditionally. The reasoning for both, and
        for why they diverge, is at the restore site in ``autonomous_wiring`` —
        which is the one place that can see both counters at once, and therefore
        the only place the comparison can honestly live.
        """
        self._consecutive_failures += 1
        logger.warning(
            "trial_failed",
            iteration=self._iteration, what=what,
            consecutive=self._consecutive_failures,
            limit=self._park_after_failed_trials,
        )
        for _ in range(max(1, int(wells))):
            self._advance_iteration()
        if self._consecutive_failures >= self._park_after_failed_trials:
            self._park(
                f"{self._consecutive_failures} consecutive trial failures "
                f"({what}) — treating as a systematic fault"
            )
            return True
        return False

    def _note_trial_success(self) -> None:
        """Reset the consecutive-failure counter after a measured trial."""
        self._consecutive_failures = 0

    async def _await_gate(self, event: asyncio.Event, what: str) -> bool:
        """Wait on a human-in-the-loop gate, bounded by ``gate_timeout_s``.

        Returns ``True`` if the gate was released, ``False`` if it timed out (in
        which case the loop has already parked) **or if an Abort released it**.
        An unbounded wait here is what turns "nobody answered the prompt" into a
        run that appears to be working all night; parking instead makes the rig
        safe and records the reason.

        The abort check is on the return path rather than at the call sites
        because :meth:`abort` releases this gate by *setting* its event — which
        is indistinguishable from an operator answering it. Without the check,
        an Abort issued at an approval prompt would be read as approval and
        would execute the trial it was meant to stop.
        """
        if self._gate_timeout_s is None:
            await event.wait()
            return not self._abort_requested
        try:
            await asyncio.wait_for(event.wait(), timeout=self._gate_timeout_s)
            return not self._abort_requested
        except asyncio.TimeoutError:
            self._park(
                f"{what} gate timed out after {self._gate_timeout_s:.0f}s "
                "with no response"
            )
            return False

    @property
    def park_reason(self) -> str | None:
        """Why the loop parked, or ``None`` if it did not."""
        return self._park_reason

    def _is_unmeasured(self, objective: "float | None", params: dict[str, Any]) -> bool:
        """``True`` when a trial produced no usable measurement.

        An objective of ``None`` means *not measured*, and it must never reach
        :meth:`optimizer.tell`.  Coercing it to a number (historically ``0.0``)
        told the optimizer a fabricated observation, making the surrogate
        confident about a composition that was never actually measured — a
        silent corruption of the whole campaign's model.

        The DOE row is deliberately left with ``objective_value`` NULL, which is
        the honest record of "suggested, cast, but not measured".
        """
        if objective is None:
            logger.warning(
                "trial_unmeasured",
                iteration=self._iteration,
                params=params,
                msg="no usable measurement — not told to the optimizer",
            )
            return True
        return False

    def _tell_placed(
        self, cache: dict[str, Any], placed: list[tuple[dict[str, Any], int]]
    ) -> bool:
        """Extract per-electrode objectives from *cache* and tell each.

        Returns ``True`` when the loop should **stop** — either convergence was
        reached (state ``CONVERGED``) or the failure run-length triggered a park
        (state ``STOPPED``). Either way the caller ends the round.
        """
        for params, channel in placed:
            try:
                objective = self._placement_extract(cache, channel, params)
            except Exception as exc:
                logger.error(
                    "objective_extraction_error",
                    iteration=self._iteration, error=str(exc),
                )
                if self._note_trial_failure(f"analyze: {type(exc).__name__}"):
                    return True   # parked — stop the round
                continue
            # Record the row even when unmeasured: the well *was* cast, so the
            # provenance matters; only objective_value stays NULL.
            doe_id = self._data_store.record_doe_parameter(
                run_id=self._run_id, channel=channel,
                iteration=self._iteration, parameters=params,
            )
            if self._is_unmeasured(objective, params):
                if self._note_trial_failure("unmeasured"):
                    return True   # parked — stop the round
                continue
            self._note_trial_success()
            self._optimizer.tell(params, objective)
            self._data_store.update_doe_objective(doe_id, objective)
            if self.on_result:
                self.on_result(self._iteration, params, objective)
            self._advance_iteration()
            if self._convergence_fn(self._optimizer.history):
                logger.info("loop_converged", iteration=self._iteration)
                self._set_state(LoopState.CONVERGED)
                if self.on_converged:
                    self.on_converged(self._iteration, self._optimizer.best())
                return True
        return False
