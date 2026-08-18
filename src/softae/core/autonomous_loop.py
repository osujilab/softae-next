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
        max_consecutive_failures: int = 3,
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
        # survivable — and ``max_consecutive_failures`` is what keeps that from
        # becoming the opposite failure mode, silently burning a whole board on a
        # systematic fault.
        self._continue_on_error = continue_on_error
        self._max_channel_retries = max_channel_retries
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._park_reason: str | None = None
        self._gate_timeout_s = gate_timeout_s

        self._state = LoopState.IDLE
        self._pending_params: dict[str, Any] | None = None
        self._iteration = 0
        self._approval_event = asyncio.Event()

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

    # ── Iteration advance + checkpoint (P3.2) ───────────────────────────

    def _advance_iteration(self) -> None:
        """Close out an iteration and write its resume point.

        Every path that finishes a trial goes through here — success, aborted,
        errored, and unmeasured alike — because the well was consumed in all of
        them. A resume that replayed a failed trial would re-cast a used well.
        Routing all advances through one helper is what keeps a future code path
        from silently skipping the checkpoint.
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

    # ── Main loop ───────────────────────────────────────────────────────

    async def run(self) -> tuple[dict[str, Any], float] | None:
        """Execute the full suggest-execute-analyze-tell cycle.

        Returns the best ``(params, objective)`` or ``None`` on abort.
        """
        self._set_state(LoopState.SUGGESTING)

        while self._state not in TERMINAL_STATES:
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
                self._advance_iteration()
                if self._note_trial_failure(f"execute: {type(exc).__name__}"):
                    break
                continue

            # 4. ANALYZE
            self._set_state(LoopState.ANALYZING)
            try:
                objective = self._extract_objective(step_results)
            except Exception as exc:
                logger.error("objective_extraction_error", iteration=self._iteration, error=str(exc))
                self._advance_iteration()
                if self._note_trial_failure(f"analyze: {type(exc).__name__}"):
                    break
                continue

            # 5. TELL — never fabricate an observation for an unmeasured trial.
            if self._is_unmeasured(objective, params):
                self._advance_iteration()
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
        # Executor passes (step, idx, total, result, elapsed); accept extras so
        # this stays robust to callback-signature growth.
        executor.on_step_complete = lambda step, idx, total, result, *_: results.update(
            {step.name: result}
        )
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

        await executor.run(trial_wf)
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
            for _ in batch:
                self._advance_iteration()
            return not self._note_trial_failure(f"execute: {type(exc).__name__}")

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
                self._advance_iteration()
                if self._note_trial_failure(f"analyze: {type(exc).__name__}"):
                    return False
                continue
            if self._is_unmeasured(objective, params):
                self._advance_iteration()
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
            for _ in alloc.channels:
                self._advance_iteration()
            return not self._note_trial_failure(f"execute: {type(exc).__name__}")

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

    def _note_trial_failure(self, what: str) -> bool:
        """Count a failed/unmeasured trial; ``True`` when the loop should park.

        Only failures that already survived the executor's own retries reach
        here, so a single flaky step never parks a run — but a systematic fault
        stops burning wells after ``max_consecutive_failures``.
        """
        self._consecutive_failures += 1
        logger.warning(
            "trial_failed",
            iteration=self._iteration, what=what,
            consecutive=self._consecutive_failures,
            limit=self._max_consecutive_failures,
        )
        if self._consecutive_failures >= self._max_consecutive_failures:
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
        which case the loop has already parked).  An unbounded wait here is what
        turns "nobody answered the prompt" into a run that appears to be working
        all night; parking instead makes the rig safe and records the reason.
        """
        if self._gate_timeout_s is None:
            await event.wait()
            return True
        try:
            await asyncio.wait_for(event.wait(), timeout=self._gate_timeout_s)
            return True
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
                self._advance_iteration()
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
                self._advance_iteration()
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
