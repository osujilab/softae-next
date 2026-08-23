"""Workflow executor — runs a :class:`Workflow` against an :class:`InstrumentManager`.

Lifecycle states::

    idle → running ⟷ paused → completed
                  ↘ aborted

The executor drives setup → loop × N → teardown.  Teardown always runs
(even after abort) so the hardware is left in a safe state.

Signals
-------
Callbacks are used instead of Qt signals so the executor has no GUI
dependency.  Pass callables via:

* ``on_step_start(step, index, total)``
* ``on_step_complete(step, index, total, result)``
* ``on_step_error(step, index, total, error)``
* ``on_state_change(old_state, new_state)``
* ``on_pause_hold(held)`` — the run has actually come to rest in a pause wait
  (``True``), or is leaving one (``False``).  Not the same event as
  ``on_state_change(_, PAUSED)``: see :meth:`WorkflowExecutor._pause_hold`.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from enum import Enum, auto
from typing import Any, Callable

import structlog

# Result routing is a plug-in seam, not executor knowledge: the EIS method
# names, routing params and DataStore persistence all live with the modality
# (spec §4 Tier 1, "Result-router registry"). workflows/ importing analysis/eis/
# follows the direction data_store.py already established.
from softae.analysis.eis.router import (
    EISResultRouter,
    ResultRouter,
    RouterContext,
    SweepCounter,
)

# The generic half of the same seam: `MeasurementResult` is modality-agnostic by
# construction, so the executor may name the type it collects without learning
# any modality's vocabulary (spec §4 Tier 2 component 1).
from softae.analysis.measurement_result import MeasurementResult
from softae.errors import AbortedError, StepTimeoutError, ValidationError_, WorkflowError
from softae.server.manager import InstrumentManager
from softae.workflows.experiment_logger import ExperimentLogger
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)

#: Virtual instrument name for built-in, hardware-agnostic control steps (e.g.
#: ``wait``). Steps targeting it are handled by the executor without a manager
#: lookup, so they run on any session (mock or real) without a driver.
_CONTROL_INSTRUMENT = "control"

#: Fallback ceiling (s) for a step that declares no ``timeout_s``, used when
#: ``[safety] step_timeout_s`` is absent. Matches the value that section ships.
DEFAULT_STEP_TIMEOUT_S = 900.0

#: Step tag declaring that an anti-clog purge may run **concurrently** with this
#: step. Set by the step's author, never inferred: whether a step's dead time is
#: usable depends on physics the executor cannot see (does dispensing disturb
#: this measurement? does the step need the syringe itself?), so it has to be
#: asserted deliberately rather than guessed from a phase name.
PURGE_WINDOW_TAG = "purge_window"

#: How often a purge window re-offers the opportunity while its step runs. The
#: purge itself decides whether one is actually owed, and that check is cheap,
#: so this only needs to be short relative to the purge interval.
PURGE_WINDOW_POLL_S = 30.0

#: Consecutive abandoned channels after which the run stops and asks a human.
#:
#: Derived from the retry arithmetic, not by analogy. A wedged stage raises
#: ``CommunicationError`` / ``StepTimeoutError``, both of which
#: :meth:`WorkflowExecutor._recoverable_cause` calls retryable, so with the
#: shipped ``max_channel_retries = 1`` each channel costs *initial + 1 replay* =
#: **2** stage attempts. Unbounded, a 32-channel plate therefore drives an
#: obstruction **64** times.
#:
#: At 3 the ceiling costs 3 × 2 = **6** attempts — under a tenth of that — while
#: still leaving 29 channels unattempted, so a plate is salvageable if the
#: operator frees the stage. Below 3 the ceiling would fight the policy it sits
#: inside: one bad well must not cost a plate, and two bad wells on a 32-channel
#: board is ordinary. Above 3 it buys nothing but attempts, since every one of
#: these channels already had its stage session reset by ``on_step_recover``
#: between attempts — three consecutive channels failing *after* a reset each is
#: no longer a story about wells.
#:
#: ``0`` disables the ceiling, restoring the previous unbounded behaviour — and
#: ``0`` is the **constructor default**, so this is opt-in per host rather than
#: on for everyone. The other host of the recovery path is
#: ``AutonomousLoop._run_workflow``, which builds its executor with
#: ``continue_on_error=True`` and wires **no** prompt callback: defaulting the
#: ceiling on would give an unattended campaign a silent hour-long stall in place
#: of a question nobody is there to hear — the exact "appears to be working all
#: night" failure this mechanism exists to prevent — and it would do so on top of
#: the campaign's own fault classification and park path, which already covers it.
#: The HT tab passes this value explicitly; that is the caller prompt-and-hold was
#: specified for.
DEFAULT_MAX_CONSECUTIVE_CHANNEL_FAILURES = 3

#: Ceiling (s) on the prompt-and-hold wait before the run parks itself.
#:
#: Modelled on ``autonomous_loop.DEFAULT_GATE_TIMEOUT_S`` (3600.0) and on the
#: reasoning ``_await_gate`` records: an unbounded wait is what turns "nobody
#: answered the prompt" into a run that appears to be working all night. "HT is
#: attended" and "HT is attended at hour four" are different claims; a plate runs
#: for hours and a failing channel surfaces only as a coloured table row.
DEFAULT_CHANNEL_HOLD_TIMEOUT_S = 3600.0

#: Poll interval (s) of the hold loop. Matches the pause loop's, so a resume or
#: an abort is picked up just as fast from a hold as from an ordinary pause.
CHANNEL_HOLD_POLL_S = 0.05


def _extract_iter_suffix(name: str) -> int | None:
    """Extract iteration index from a step name like ``'deposit__iter2'``."""
    parts = name.rsplit("__iter", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


class ExecutorState(Enum):
    """Lifecycle states of the :class:`WorkflowExecutor`."""

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ABORTED = auto()
    ERROR = auto()


class WorkflowExecutor:
    """Async engine that executes a :class:`Workflow`.

    Parameters
    ----------
    manager : InstrumentManager
        The instrument registry to dispatch method calls against.
    experiment_logger : ExperimentLogger | None
        If provided, every step invocation is written as a structured log
        record to the experiment log file.
    routers : list[ResultRouter] | None
        Result routers consulted after every successful step (Tier 1 seam,
        spec §4). ``None`` — the default — wires the EIS router, preserving
        the historical auto-routing; future modalities register here instead
        of editing the executor (spec Tier 2 component 5 lands on this seam).
    """

    def __init__(
        self,
        manager: InstrumentManager,
        experiment_logger: ExperimentLogger | None = None,
        data_store: Any | None = None,
        run_id: str | None = None,
        continue_on_error: bool = False,
        max_channel_retries: int = 1,
        routers: list[ResultRouter] | None = None,
        max_consecutive_channel_failures: int = 0,
        channel_hold_timeout_s: float = DEFAULT_CHANNEL_HOLD_TIMEOUT_S,
    ) -> None:
        self.manager = manager
        self.experiment_logger = experiment_logger
        self.data_store = data_store
        self._run_id = run_id

        # `None` means "the defaults", not "no routing" — every existing caller
        # keeps EIS auto-routing without change. Pass `[]` to disable routing.
        self._routers: list[ResultRouter] = (
            list(routers) if routers is not None else [EISResultRouter()]
        )
        # Union of every router's routing-only params, computed once: the
        # executor still strips them before instrument calls, but no longer
        # knows any modality's vocabulary itself.
        self._routing_params: frozenset[str] = frozenset().union(
            *(getattr(r, "consumed_params", frozenset()) for r in self._routers)
        )

        # Public, per-run: every MeasurementResult the routers returned during
        # the most recent `run()`, in acquisition order. Reset at the start of
        # each run, so it always describes one run rather than the executor's
        # lifetime (unlike `_sweep_counter`, whose whole purpose is to span).
        #
        # Tier 2 component 2 (spec §4) populates this and NOTHING reads it yet —
        # persistence still happens inside each router, unchanged. Component 3
        # is the first consumer. Routers that return None (skipped or failed)
        # contribute no entry, so presence here means "recorded", never merely
        # "attempted".
        self.measurement_results: list[MeasurementResult] = []

        # Acquisition position within this run, for §6's drift metadata. Counted from
        # what was actually recorded, so a retry or a skipped channel shifts every
        # later position — which is correct: `sweep_order` answers "when in the
        # sequence was this taken", not "where was it planned". Owned here (not by
        # a router) so the count spans the whole run whatever routers handle it.
        self._sweep_counter = SweepCounter()

        # --- Graceful stage-timeout recovery policy ---
        # When enabled (HT/AE campaigns), a recoverable comms/timeout failure on
        # a channel-tagged step does not abort the run. Instead the channel is
        # conditionally replayed from its precondition step (only if no elution
        # was committed — see `dispense_committed`) up to `max_channel_retries`
        # times, then skipped so the campaign proceeds to the next channel.
        self.continue_on_error = continue_on_error
        self.max_channel_retries = max_channel_retries

        # --- Mechanical ceiling on abandoned channels ---
        # Skipping is the intended behaviour and stays; what had no bound was how
        # many channels may be abandoned in a row, which is the difference between
        # "one bad well" and "driving the stage into an obstruction all afternoon".
        self.max_consecutive_channel_failures = max_consecutive_channel_failures
        self.channel_hold_timeout_s = channel_hold_timeout_s
        self._consecutive_channel_failures = 0

        # Public, per-run: every channel abandoned by `_skip_channel`, in the
        # order they were given up on. Reset at the start of each run, so it
        # always describes one run. Its purpose is durable: the host writes it to
        # the run row so that "which wells on this plate are real?" is answerable
        # next month, when the results table that showed it live is long gone.
        self.skipped_channels: list[str] = []

        self._state = ExecutorState.IDLE
        self._current_step: WorkflowStep | None = None
        self._current_index: int = 0
        self._workflow: Workflow | None = None

        # --- Callbacks (plain callables, not Qt signals) ---
        self.on_step_start: Callable[..., Any] | None = None
        self.on_step_complete: Callable[..., Any] | None = None
        self.on_step_error: Callable[..., Any] | None = None
        self.on_state_change: Callable[..., Any] | None = None
        # Fired before a channel replay so the host can reset the wedged stage
        # (``on_step_recover(step, error, attempt)``), and when a channel is
        # abandoned (``on_step_skipped(step, index, total, reason)``).
        self.on_step_recover: Callable[..., Any] | None = None
        self.on_step_skipped: Callable[..., Any] | None = None
        # Fired when consecutive channel failures hit the ceiling, immediately
        # *after* the executor has paused itself
        # (``on_channel_failure_hold(channel, consecutive, timeout_s)``). The
        # host's job is to make the question audible — the executor only holds.
        # The run is already stopped when this fires, so a host that answers
        # synchronously is answering a hold that exists.
        self.on_channel_failure_hold: Callable[..., Any] | None = None

        # --- Pause holds ---
        # ``on_pause_hold(held: bool)`` fires ``True`` when the run actually
        # comes to rest in a pause wait and ``False`` when it leaves one. It is
        # deliberately **not** ``on_state_change(_, PAUSED)``, which is the
        # obvious hook and is wrong for a reason invisible from the signature:
        # ``pause()`` only sets the flag, and the executor keeps driving until
        # the top of the next tier or step. Between the button press and the
        # wait lies the remainder of the current step — a dispense, a stage
        # move, an EIS sweep — so a host that hands the instruments back on the
        # state change hands the syringe back mid-dispense.
        #
        # ``None`` by default; with it unset nothing about the executor changes.
        self.on_pause_hold: Callable[..., Any] | None = None
        # Nested pause holds collapse to the outermost. ``RigActivity.unsuspend``
        # is membership, not a counter — its own docstring says an unsuspend from
        # an inner hold clears an outer pause's suspension — and a *suspended*
        # owner is the one that PERMITS manual control. So an inner hold firing
        # ``False`` would take manual control away in the middle of the
        # operator's own pause, with nothing reporting it. Counting here means
        # the host sees one True and one False per hold and needs no nesting
        # logic of its own. Touched only from the executor's event loop, so a
        # plain int is enough.
        self._pause_hold_depth = 0

        # --- Concurrent purge windows (P8) ---
        # ``on_purge_window(step) -> None`` is run as a task *alongside* a step
        # that declares itself co-runnable, and joined before the run moves on.
        #
        # This is the second of two in-run purge mechanisms, and it exists
        # because the first does not generalise. A long anneal offers repeated
        # opportunities through ``monitored_hold``'s poll loop; an EIS sweep is a
        # single opaque blocking read (``readlines_until_end``) with no interior
        # yield point, so the only way to use that dead time is to run the purge
        # *concurrently* on disjoint instruments.
        #
        # Safe because sync driver calls are dispatched to a thread pool (so the
        # event loop keeps running) and every instrument has its own asyncio
        # lock (so a collision blocks rather than corrupts). The join is what
        # guarantees the next step never starts mid-purge.
        self.on_purge_window: Callable[..., Any] | None = None

        # --- Waste accrual (P5.4) ---
        # Optional WasteLedger. Set by the host so flushes book themselves as
        # they run; without it the container level only ever moves when an
        # operator edits it by hand, which is how it silently drifts.
        self.waste_ledger: Any | None = None

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> ExecutorState:
        return self._state

    @property
    def current_step(self) -> WorkflowStep | None:
        return self._current_step

    @property
    def progress(self) -> tuple[int, int]:
        """``(current_index, total_steps)``."""
        total = self._workflow.total_steps if self._workflow else 0
        return (self._current_index, total)

    # ── Control ─────────────────────────────────────────────────────────

    async def run(self, workflow: Workflow) -> None:
        """Execute *workflow* from start to finish.

        Raises
        ------
        AbortedError
            If :meth:`abort` was called during execution.
        WorkflowError
            On unrecoverable step failure.
        """
        # Safety interlock: refuse to drive real motion hardware unless the
        # operator has deliberately armed it. No-op for mock managers, so
        # simulation and tests are unaffected. This is the single choke point
        # protecting every execution path (CLI, autonomous, GUI, agent).
        from softae.core.hardware_safety import assert_hardware_armed

        assert_hardware_armed(self.manager, action=f"execute workflow '{workflow.name}'")

        # The rig lock goes here for the same reason the arming check does: this is the
        # one place every execution path passes through. Wiring it into each CLI
        # instead would need `softae-campaign`, `softae-commission`, the GUI and any
        # future entry point to each remember — and the one that forgets does not fail,
        # it silently defeats the lock for everyone else.
        #
        # Acquisition is re-entrant per process, so a GUI that runs several workflows
        # in a session, or a workflow that nests another, is unaffected. Simulated rigs
        # are exempt: a mock run holding the lock would turn a dry run into an outage.
        from softae.core.run_lock import (
            RunLockHeld,
            acquire_run_lock,
            read_run_lock,
            release_run_lock,
            rig_is_simulated,
        )

        self._lock_taken = False
        if not rig_is_simulated(self.manager):
            # Whether the rig was *already* ours decides who gets to give it back.
            # `acquire_run_lock` is re-entrant: asked by the process that already
            # holds the lock it hands back the existing claim rather than raising.
            # Releasing that in the `finally` below would free a claim this call
            # never made — and a campaign now holds one for its whole length, one
            # lock across many trials, so the first trial's teardown would have
            # dropped it and left every later trial running on a rig the lock file
            # said was free. Same `mine_already` discipline as `held_run_lock`
            # (run_lock.py) and for exactly the same reason.
            before = read_run_lock()
            mine_already = before is not None and before.is_mine()
            try:
                acquire_run_lock(what=f"workflow '{workflow.name}'")
                self._lock_taken = not mine_already
            except RunLockHeld as exc:
                raise WorkflowError(
                    f"refusing to execute '{workflow.name}': {exc}\n"
                    f"Two processes cannot drive the rig at once. Wait for that run, "
                    f"or take the rig over from the calibration launcher if its owner "
                    f"is genuinely gone."
                ) from exc

        self._workflow = workflow
        self._current_index = 0
        self.measurement_results = []
        self.skipped_channels = []
        self._consecutive_channel_failures = 0
        self._set_state(ExecutorState.RUNNING)

        # Build the step list WITHOUT teardown — teardown runs in `finally`.
        main_steps: list[WorkflowStep] = list(workflow.setup)
        for i in range(workflow.iterations):
            for step in workflow.loop_steps:
                expanded = step.with_tags(iteration=str(i))
                expanded = WorkflowStep(
                    name=f"{step.name}__iter{i}",
                    instrument=expanded.instrument,
                    method=expanded.method,
                    params=dict(expanded.params),
                    depends_on=list(expanded.depends_on),
                    timeout_s=expanded.timeout_s,
                    retry=expanded.retry,
                    tags=dict(expanded.tags),
                )
                main_steps.append(expanded)

        total = len(main_steps) + len(workflow.teardown)

        logger.info(
            "workflow_start",
            name=workflow.name,
            total_steps=total,
            iterations=workflow.iterations,
        )

        try:
            dag = self._build_dag(main_steps)
            tiers = self._topological_tiers(main_steps, dag)

            # The graceful-recovery path (channel replay/skip) is only meaningful
            # for a linear workflow — HT/AE campaigns are inherently sequential
            # (one stage, one syringe). If parallelism is present we fall back to
            # the standard fail-fast tier executor even when continue_on_error is
            # set, since replaying a channel across concurrent tiers is undefined.
            linear = all(len(t) == 1 for t in tiers)
            if self.continue_on_error and linear:
                await self._run_linear_with_recovery(
                    [t[0] for t in tiers], total
                )
            else:
                if self.continue_on_error and not linear:
                    logger.warning(
                        "continue_on_error_ignored_nonlinear", name=workflow.name
                    )
                await self._run_tiers(tiers, total)

            self._set_state(ExecutorState.COMPLETED)
            logger.info("workflow_completed", name=workflow.name)

        except AbortedError:
            logger.warning("workflow_aborted", name=workflow.name, at_step=self._current_index)
            raise

        except Exception as exc:
            self._set_state(ExecutorState.ERROR)
            logger.error("workflow_error", name=workflow.name, error=str(exc))
            raise

        finally:
            # Teardown always runs (best-effort, even after abort/error)
            await self._run_teardown(workflow.teardown, total)
            # Released after teardown, not before: teardown drives the hardware too
            # (parking the head, halting pumps), and handing the rig away mid-park is
            # exactly the window the lock exists to close.
            if getattr(self, "_lock_taken", False):
                release_run_lock()
                self._lock_taken = False

    def pause(self) -> None:
        """Pause execution after the current step completes."""
        if self._state is ExecutorState.RUNNING:
            self._set_state(ExecutorState.PAUSED)
            logger.info("workflow_paused")

    def resume(self) -> None:
        """Resume a paused workflow."""
        if self._state is ExecutorState.PAUSED:
            self._set_state(ExecutorState.RUNNING)
            logger.info("workflow_resumed")

    @asynccontextmanager
    async def _pause_hold(self):
        """Announce, for the duration of the block, that the run is *held*.

        One definition, three call sites — the three ``while self._state is
        ExecutorState.PAUSED`` waits. It cannot be collapsed to one invocation:
        the consecutive-failure hold carries a deadline that parks and aborts,
        and is not structurally the same wait.

        **Entered only when the run is already paused.** Wrapping the waits
        unconditionally would announce a hold at every tier and step boundary of
        every run, handing the instruments back for the instant between entering
        and leaving — a real window, since the host that consumes this callback
        is read from the GUI thread. Guarding at the call site rather than in
        here is what makes the announcement cover the *whole* wait: either the
        guard passes and the block covers it, or there is no wait at all.

        Nested holds collapse to the outermost, for the reason
        :attr:`_pause_hold_depth` records. No such nesting is reachable today
        (``_hold_for_operator`` runs from the body of the linear loop, after that
        loop's own wait has exited, and the tier and linear strategies are
        alternatives chosen once per run) — the counter is what keeps it
        unreachable *by construction* rather than by that argument staying true.
        """
        self._pause_hold_depth += 1
        if self._pause_hold_depth == 1:
            self._fire_pause_hold(True)
        try:
            yield
        finally:
            self._pause_hold_depth -= 1
            if self._pause_hold_depth == 0:
                self._fire_pause_hold(False)

    def _fire_pause_hold(self, held: bool) -> None:
        """Tell the host the run is (no longer) held. Never raises.

        The hold is the safety property; telling the host about it is a
        courtesy, and a courtesy that fails must not break a run that has
        already stopped — the same reasoning ``_hold_for_operator`` applies to
        a prompt that fails to draw.
        """
        if self.on_pause_hold is None:
            return
        try:
            self.on_pause_hold(held)
        except Exception:
            logger.warning("on_pause_hold_failed", held=held, exc_info=True)

    def abort(self) -> None:
        """Request workflow abort.  Takes effect before the next step."""
        old = self._state
        if old in {ExecutorState.RUNNING, ExecutorState.PAUSED}:
            self._set_state(ExecutorState.ABORTED)
            logger.warning("workflow_abort_requested")

    # ── DAG helpers ─────────────────────────────────────────────────────

    def _build_dag(
        self, steps: list[WorkflowStep]
    ) -> dict[str, set[str]]:
        """Build adjacency dict: step_name -> set of dependency names.

        Steps without explicit ``depends_on`` implicitly depend on the
        preceding step, preserving sequential ordering for backward
        compatibility.
        """
        # Step names are identifiers: dependencies reference them and this DAG is
        # keyed by them (as are step results).  Two steps sharing a name collapse
        # into one DAG entry, which silently drops a step and can fabricate a
        # false dependency cycle — so reject duplicates up front with a clear
        # message instead of failing later as an inscrutable "cycle".
        names = [s.name for s in steps]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValidationError_(
                f"Duplicate step name(s) {dupes} — step names must be unique "
                f"within a phase (they key dependencies and results)."
            )

        step_names = {s.name for s in steps}
        dag: dict[str, set[str]] = {}

        for i, step in enumerate(steps):
            resolved_deps: set[str] = set()

            if step.depends_on:
                # Explicit dependencies
                for dep in step.depends_on:
                    if dep in step_names:
                        resolved_deps.add(dep)
                    else:
                        iter_idx = _extract_iter_suffix(step.name)
                        if iter_idx is not None:
                            candidate = f"{dep}__iter{iter_idx}"
                            if candidate in step_names:
                                resolved_deps.add(candidate)
                            else:
                                raise ValidationError_(
                                    f"Step '{step.name}' depends on '{dep}', "
                                    f"which does not exist (tried '{candidate}')"
                                )
                        else:
                            raise ValidationError_(
                                f"Step '{step.name}' depends on '{dep}', "
                                f"which does not exist"
                            )
            elif i > 0:
                # Implicit sequential dependency on previous step
                resolved_deps.add(steps[i - 1].name)

            dag[step.name] = resolved_deps

        return dag

    def _topological_tiers(
        self,
        steps: list[WorkflowStep],
        dag: dict[str, set[str]],
    ) -> list[list[WorkflowStep]]:
        """Group steps into tiers via Kahn's algorithm.

        All steps in a tier can run concurrently.  Raises
        ``ValidationError_`` on cycles.
        """
        step_map = {s.name: s for s in steps}
        in_degree = {name: len(deps) for name, deps in dag.items()}

        dependents: dict[str, list[str]] = {name: [] for name in dag}
        for name, deps in dag.items():
            for dep in deps:
                dependents[dep].append(name)

        tiers: list[list[WorkflowStep]] = []
        ready = [n for n, d in in_degree.items() if d == 0]
        visited = 0

        while ready:
            tier = [step_map[n] for n in ready]
            tiers.append(tier)
            visited += len(ready)

            next_ready: list[str] = []
            for name in ready:
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        next_ready.append(dep_name)
            ready = next_ready

        if visited != len(dag):
            cycle_members = sorted(
                n for n, d in in_degree.items() if d > 0
            )
            raise ValidationError_(
                f"Dependency cycle detected among steps: {cycle_members}"
            )

        return tiers

    # ── Execution strategies ────────────────────────────────────────────

    async def _run_tiers(
        self, tiers: list[list[WorkflowStep]], total: int
    ) -> None:
        """Standard fail-fast executor: run tiers in order, concurrently within
        a tier. The first unrecoverable step error aborts the run."""
        step_index = 0
        for tier in tiers:
            # --- Pause loop, THEN check abort ---
            # Order matters: `abort()` accepts PAUSED, so an executor held here is
            # released by the very state change that must stop it. Checking abort
            # before the pause loop let the released iteration fall straight into
            # `_run_step` — one more dispense or stage move after the operator
            # aborted a rig they believed quiescent. That order is unchanged: the
            # hold wraps the wait only, and the abort check still follows it.
            if self._state is ExecutorState.PAUSED:
                async with self._pause_hold():
                    while self._state is ExecutorState.PAUSED:
                        await asyncio.sleep(0.05)

            if self._state is ExecutorState.ABORTED:
                raise AbortedError(
                    f"Workflow aborted before step '{tier[0].name}'"
                )

            if len(tier) == 1:
                # Single-step tier — sequential fast path
                step = tier[0]
                self._current_index = step_index
                self._current_step = step
                await self._run_step(step, step_index, total)
                step_index += 1
            else:
                # Parallel tier — run all steps concurrently
                base_index = step_index

                async def _run_in_tier(s: WorkflowStep, idx: int) -> None:
                    self._current_step = s
                    await self._run_step(s, idx, total)

                tasks = [
                    _run_in_tier(s, base_index + i)
                    for i, s in enumerate(tier)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Re-raise the first real exception
                for r in results:
                    if isinstance(r, BaseException):
                        raise r

                step_index += len(tier)

    async def _run_linear_with_recovery(
        self, steps: list[WorkflowStep], total: int
    ) -> None:
        """Sequential executor with graceful stage-timeout recovery.

        On a recoverable comms/timeout failure of a channel-tagged step:

        * if the failure is safe to retry (a ``precondition`` step, or a
          ``deposit`` step whose elution was **not** yet committed), fire
          ``on_step_recover`` (host resets the stage) and **replay the channel
          from its first step** — up to ``max_channel_retries`` times;
        * otherwise (elution committed, or retries exhausted) **skip the rest of
          that channel** and continue with the next one.

        Campaign-level steps (no ``channel`` tag — e.g. the startup flush) and
        non-recoverable errors still hard-fail, since a bad line prime is unsafe
        to run through.
        """
        channel_retries: dict[str, int] = {}
        i = 0
        while i < len(steps):
            # Pause first, abort second — see `_run_tiers` for why the order is
            # the fix: an abort issued while paused must not release one step.
            # The hold wraps the wait only; the order it protects is untouched.
            if self._state is ExecutorState.PAUSED:
                async with self._pause_hold():
                    while self._state is ExecutorState.PAUSED:
                        await asyncio.sleep(0.05)

            if self._state is ExecutorState.ABORTED:
                raise AbortedError(
                    f"Workflow aborted before step '{steps[i].name}'"
                )

            step = steps[i]
            self._current_index = i
            self._current_step = step
            try:
                await self._run_step(step, i, total)
                if self._channel_ends_here(steps, i):
                    # A channel that finished every one of its steps breaks the
                    # streak. Counting whole channels rather than steps is what
                    # makes the ceiling mean "the rig is wedged" instead of "some
                    # steps worked": a stage that fails only on the move still
                    # completes the pump steps before it.
                    self._consecutive_channel_failures = 0
                i += 1
                continue
            except WorkflowError as exc:
                channel = step.tags.get("channel")
                cause = self._recoverable_cause(exc)
                if channel is None or cause is None:
                    raise  # campaign-level or non-recoverable → hard fail

                phase = step.tags.get("phase", "")
                committed = bool(getattr(cause, "dispense_committed", False)) or \
                    bool(getattr(exc, "dispense_committed", False))
                # Deposit is only safe to replay before elution is committed;
                # every other channel phase (precondition, piezo, eis) can be
                # re-entered from the channel's precondition step.
                retriable = (phase != "deposit") or (not committed)
                used = channel_retries.get(channel, 0)

                if retriable and used < self.max_channel_retries:
                    channel_retries[channel] = used + 1
                    logger.warning(
                        "channel_retry",
                        channel=channel,
                        step=step.name,
                        phase=phase,
                        attempt=used + 1,
                        max=self.max_channel_retries,
                    )
                    if self.on_step_recover:
                        try:
                            self.on_step_recover(step, exc, used + 1)
                        except Exception:
                            logger.warning("on_step_recover_failed", exc_info=True)
                    i = self._channel_run_start(steps, i)
                    continue

                # Give up on this channel — skip its remaining steps.
                reason = "elution committed" if committed else "retries exhausted"
                logger.error(
                    "channel_skipped",
                    channel=channel,
                    step=step.name,
                    phase=phase,
                    reason=reason,
                )
                i = self._skip_channel(steps, channel, i, total, reason)
                await self._count_channel_failure(channel)
                continue

    @staticmethod
    def _recoverable_cause(error: BaseException) -> BaseException | None:
        """Return the underlying recoverable comms/timeout error, or ``None``.

        A stage that exhausts its own self-heal cascade raises
        :class:`CommunicationError`; a step that blew its ``timeout_s`` raises
        :class:`StepTimeoutError`. Both are wrapped in a ``WorkflowError`` by
        :meth:`_run_step`, so we look through ``__cause__`` as well.
        """
        from softae.errors import CommunicationError, StepTimeoutError

        for e in (error, getattr(error, "__cause__", None)):
            if isinstance(e, (CommunicationError, StepTimeoutError)):
                return e
        msg = str(error)
        if "VI_ERROR_TMO" in msg or "Timeout expired" in msg:
            return error
        return None

    @staticmethod
    def _channel_run_start(steps: list[WorkflowStep], i: int) -> int:
        """Start index of the *contiguous* channel run containing step ``i``.

        Replay rewinds to the first step of the failing step's contiguous block
        of same-channel steps — **not** the channel's global-first occurrence.
        With the legacy interleaved layout a channel's steps form a single block,
        so this equals the old behaviour; but when measurement is **deferred**
        into a later per-batch block (batch runs), a recoverable EIS failure
        replays only that EIS step rather than jumping back and re-casting the
        drop (a double-dispense).
        """
        channel = steps[i].tags.get("channel")
        j = i
        while j > 0 and steps[j - 1].tags.get("channel") == channel:
            j -= 1
        return j

    @staticmethod
    def _channel_ends_here(steps: list[WorkflowStep], i: int) -> bool:
        """Is step ``i`` the last step of its channel's contiguous block?

        Mirrors :meth:`_channel_run_start` at the other end of the block, so both
        the replay rewind and the streak reset agree on where a channel stops —
        including under the deferred-measurement layout, where one channel owns
        two blocks.
        """
        channel = steps[i].tags.get("channel")
        if channel is None:
            return False
        return i + 1 >= len(steps) or steps[i + 1].tags.get("channel") != channel

    def _skip_channel(
        self,
        steps: list[WorkflowStep],
        channel: str,
        from_i: int,
        total: int,
        reason: str,
    ) -> int:
        """Skip the contiguous remaining steps of *channel* from *from_i*.

        The step at ``from_i`` already reported its own error via
        :meth:`_run_step`; the trailing steps of the channel are surfaced as
        skipped. Returns the index of the next non-channel step.
        """
        # Recorded once per channel, not once per skipped step: the question this
        # answers is "is this well real?", which is asked of channels. A channel
        # can be abandoned twice under the deferred-measurement layout (its cast
        # block and its measure block are separate), and it is no less one well
        # for it.
        if channel not in self.skipped_channels:
            self.skipped_channels.append(channel)

        j = from_i + 1
        while j < len(steps) and steps[j].tags.get("channel") == channel:
            skipped = steps[j]
            logger.info("step_skipped", step=skipped.name, channel=channel, reason=reason)
            if self.experiment_logger:
                self.experiment_logger.log_step(
                    workflow=self._workflow.name if self._workflow else "",
                    step=skipped,
                    duration_s=0.0,
                    result=f"skipped: {reason}",
                )
            if self.on_step_skipped:
                try:
                    self.on_step_skipped(skipped, j, total, reason)
                except Exception:
                    logger.warning("on_step_skipped_failed", exc_info=True)
            j += 1
        return j

    # ── Mechanical ceiling: prompt-and-hold ──────────────────────────────

    async def _count_channel_failure(self, channel: str) -> None:
        """Book one abandoned channel and hold the run if the streak hits the ceiling."""
        self._consecutive_channel_failures += 1
        ceiling = self.max_consecutive_channel_failures
        if not ceiling or self._consecutive_channel_failures < ceiling:
            return
        await self._hold_for_operator(channel, self._consecutive_channel_failures)

    async def _hold_for_operator(self, channel: str, consecutive: int) -> None:
        """Pause and ask, rather than park — the operator is the better classifier.

        Deliberately **not** an automatic park. ``_is_hard_fault`` recognises only
        ``SafetyError`` and ``HardwareNotArmedError``, and a wedged stage is
        neither; an operator standing at the rig can see in one glance what no
        exception class encodes. Taking that decision away from them would be a
        downgrade, so this stops the motion and hands them the question.

        **A hold is never a lockout.** Resuming continues the plate, aborting
        stops it, and both are picked up within :data:`CHANNEL_HOLD_POLL_S`. The
        only thing this refuses is *silence*: if nobody answers within
        ``channel_hold_timeout_s`` the run was unattended whatever the design
        assumed, and it parks.
        """
        logger.error(
            "channel_failure_ceiling",
            channel=channel,
            consecutive=consecutive,
            ceiling=self.max_consecutive_channel_failures,
            attempts_per_channel=1 + self.max_channel_retries,
            timeout_s=self.channel_hold_timeout_s,
        )
        # Paused BEFORE the prompt is raised, not after. The prompt is an
        # arbitrary host callable, and a host that answers synchronously — a test
        # harness, a headless driver — would otherwise resume a run that had not
        # yet paused, and its answer would be silently dropped by the pause that
        # followed it. Pausing first makes both answers land whenever they arrive.
        self.pause()
        if self.on_channel_failure_hold is None:
            # The hold still happens — stopping is the safety property and the
            # dialog is only how it is asked — but a host that enabled the ceiling
            # and wired no prompt has built a silent stall, and that is worth
            # saying out loud in the one place that can see it.
            logger.warning("channel_failure_hold_unprompted", channel=channel)
        else:
            try:
                self.on_channel_failure_hold(
                    channel, consecutive, self.channel_hold_timeout_s
                )
            except Exception:
                # A prompt that fails to draw must not also cancel the hold: the
                # stop is the safety property, the dialog is only how it is asked.
                logger.warning("on_channel_failure_hold_failed", exc_info=True)

        deadline = time.monotonic() + self.channel_hold_timeout_s
        unanswered = False
        # The guard is not redundant with the loop: `pause()` above is a no-op
        # unless the state was RUNNING, so an abort that landed a moment earlier
        # leaves nothing to hold and nothing to announce.
        if self._state is ExecutorState.PAUSED:
            async with self._pause_hold():
                while self._state is ExecutorState.PAUSED:
                    if time.monotonic() >= deadline:
                        unanswered = True
                        break
                    await asyncio.sleep(CHANNEL_HOLD_POLL_S)

        # The park is deliberately OUTSIDE the hold, which is why the timeout
        # breaks the loop rather than acting inside it. A hold hands the
        # instruments back — that is the whole point of `on_pause_hold` — and
        # `_park_unattended` drives them: parking while the hold was still
        # announced would move the head and halt the pumps at the one moment a
        # manual jog was permitted. Leaving the hold re-guards the rig first.
        if unanswered:
            await self._park_unattended(channel, consecutive)
            self.abort()
            return

        # Answered (resumed, or aborted). A clean slate rather than a hair
        # trigger: an operator who has just looked at the rig and said "carry on"
        # should not be asked again by the very next failure.
        self._consecutive_channel_failures = 0

    async def _park_unattended(self, channel: str, consecutive: int) -> None:
        """Nobody answered the hold — make the hardware safe.

        Safe to call from here specifically: the executor is paused *between*
        steps, so this is the one moment a park cannot interleave its writes with
        a step's own serial I/O.

        Never raises. A park that fails must still leave the abort to happen —
        the alternative is a run that carries on because its stop failed.
        """
        from softae.core.safe_park import safe_park_async

        reason = (
            f"HT: {consecutive} consecutive channels abandoned (last: {channel}); "
            f"no operator answered within {self.channel_hold_timeout_s:.0f}s"
        )
        try:
            result = await safe_park_async(self.manager, reason=reason)
            logger.error(
                "channel_failure_park",
                channel=channel,
                consecutive=consecutive,
                summary=result.summary(),
            )
        except Exception:
            logger.error("channel_failure_park_failed", channel=channel,
                         exc_info=True)

    # ── Internal ────────────────────────────────────────────────────────

    async def _run_step(
        self,
        step: WorkflowStep,
        index: int,
        total: int,
    ) -> None:
        """Execute a single step with retries, timeout, and logging."""
        if self.on_step_start:
            self.on_step_start(step, index, total)

        logger.info(
            "step_start",
            step=step.name,
            instrument=step.instrument,
            method=step.method,
            index=index,
        )

        last_error: Exception | None = None
        attempts = 1 + step.retry

        for attempt in range(attempts):
            t0 = time.monotonic()
            # Launched alongside the step when it declares itself co-runnable.
            # Joined in the `finally` below on EVERY exit — success, timeout,
            # error, or retry — so the next step can never begin while fluid is
            # still moving.
            purge_task = self._open_purge_window(step, attempt)
            try:
                try:
                    result = await self._dispatch(step)
                    elapsed = time.monotonic() - t0

                    # --- Result routing (e.g. EIS → DataStore) ---
                    await self._route_result(step, result)

                    # --- Waste accrual (P5.4) ---
                    # Booked from steps that actually *ran*, not from the plan:
                    # a channel skipped by error recovery never reached the
                    # container, and a waste level that drifts high makes the
                    # operator distrust the one number that caps unattended time.
                    self._accrue_waste(step)

                    # Log success
                    if self.experiment_logger:
                        self.experiment_logger.log_step(
                            workflow=self._workflow.name if self._workflow else "",
                            step=step,
                            duration_s=elapsed,
                            result="ok",
                        )
                    if self.on_step_complete:
                        self.on_step_complete(step, index, total, result, elapsed)

                    logger.info("step_complete", step=step.name,
                                duration_s=round(elapsed, 3))
                    return

                except asyncio.TimeoutError:
                    elapsed = time.monotonic() - t0
                    last_error = StepTimeoutError(step.name, step.timeout_s or 0)
                    logger.warning(
                        "step_timeout",
                        step=step.name,
                        attempt=attempt + 1,
                        timeout_s=step.timeout_s,
                    )

                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    last_error = exc
                    logger.warning(
                        "step_failed",
                        step=step.name,
                        attempt=attempt + 1,
                        error=str(exc),
                    )

                # Log failure attempt
                if self.experiment_logger:
                    self.experiment_logger.log_step(
                        workflow=self._workflow.name if self._workflow else "",
                        step=step,
                        duration_s=elapsed,
                        result=f"error: {last_error}",
                    )
            finally:
                await self._close_purge_window(purge_task, step)

        # All retries exhausted
        if self.on_step_error:
            self.on_step_error(step, index, total, last_error)
        raise WorkflowError(
            f"Step '{step.name}' failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    # ── Waste accrual (P5.4) ─────────────────────────────────────────────

    def _accrue_waste(self, step: WorkflowStep) -> None:
        """Book this step's flushed volume against the waste container.

        Uses the *same* traversal the preflight projection uses
        (:func:`~softae.core.preflight.step_waste_uL`), so the projected and
        actual fill rates cannot drift apart — two classifiers over the same
        step shapes would be two things to keep in step with the engine.

        Anti-clog purges are **not** double-counted: they dispense directly
        through the syringe rather than as executor steps, and book themselves.
        """
        ledger = getattr(self, "waste_ledger", None)
        if ledger is None:
            return
        try:
            from softae.core.preflight import step_waste_uL

            volume = step_waste_uL(step)
            if volume > 0:
                ledger.add(volume)
        except Exception:
            # Bookkeeping must never fail a step that physically succeeded.
            logger.warning("waste_accrual_failed", step=step.name, exc_info=True)

    # ── Concurrent purge windows (P8) ────────────────────────────────────

    def _open_purge_window(self, step: WorkflowStep, attempt: int):
        """Start a purge loop alongside *step* if it declares itself co-runnable.

        A **loop**, not a single shot, which is what lets one mechanism cover
        every shape of dead time. A 3-minute EIS sweep affords one opportunity;
        a five-hour anneal affords twenty, and a one-shot launch would purge
        once and let the line stagnate for the remaining 4h45m. Re-asking on a
        cadence covers both without the executor knowing which it is dealing
        with — and without any driver having to cooperate.

        Only on the first attempt: a retry means the step already failed once,
        and stacking unprompted actuation onto a recovering step is the wrong
        instinct.

        Returns an opaque handle, or ``None`` if no window was opened.
        """
        if self.on_purge_window is None or attempt != 0:
            return None
        if not (step.tags or {}).get(PURGE_WINDOW_TAG):
            return None
        stop = threading.Event()
        try:
            task = asyncio.ensure_future(
                asyncio.to_thread(self._purge_window_loop, step, stop)
            )
        except Exception:
            logger.warning("purge_window_start_failed", step=step.name,
                           exc_info=True)
            return None
        return (task, stop)

    def _purge_window_loop(self, step: WorkflowStep, stop: threading.Event) -> None:
        """Offer a purge repeatedly until the step finishes. Runs off-loop.

        Offers **before** testing the stop flag, deliberately. Checking first
        would let a short step finish before this thread is even scheduled,
        yielding zero attempts — so a fast-but-frequent step would advertise a
        purge window and then never actually provide one.
        """
        while True:
            try:
                self.on_purge_window(step)
            except Exception:
                logger.warning("purge_window_attempt_failed", step=step.name,
                               exc_info=True)
            # Interruptible: `stop.set()` wakes this immediately, so the join
            # never waits out a full interval — only an in-flight purge.
            if stop.wait(PURGE_WINDOW_POLL_S):
                return

    async def _close_purge_window(self, handle, step: WorkflowStep) -> None:
        """Stop the purge loop and join it before the run proceeds.

        **The join is the safety property**, not a tidiness one: without it a
        cast could begin while the purge is still dispensing. Waiting is
        explicitly acceptable — a purge that outlasts its window holds up the
        next step rather than being abandoned half-done.

        The flag is checked *between* purges rather than cancelling the task, so
        a purge in progress is always allowed to finish. Never propagates: a
        purge failure must not fail the step it ran beside.
        """
        if handle is None:
            return
        task, stop = handle
        stop.set()
        try:
            await task
        except Exception:
            logger.warning("purge_window_failed", step=step.name, exc_info=True)

    async def _dispatch(self, step: WorkflowStep) -> Any:
        """Call the instrument method via ``execute()`` (which handles locking).

        We intentionally use ``inst.execute()`` rather than wrapping in
        ``manager.acquire()`` because ``execute()`` already acquires the
        per-instrument ``asyncio.Lock``.  Double-acquiring would deadlock
        (``asyncio.Lock`` is not reentrant).

        The virtual ``control`` instrument (see :data:`_CONTROL_INSTRUMENT`) is
        handled here without touching the manager — its steps (e.g. ``wait``) are
        general-purpose and not tied to any hardware.
        """
        method_params = {k: v for k, v in step.params.items() if k not in self._routing_params}
        if step.instrument == _CONTROL_INSTRUMENT:
            coro = self._run_control_step(step.method, method_params)
        else:
            inst = self.manager.get(step.instrument)
            coro = inst.execute(step.method, **method_params)
        timeout = step.timeout_s
        if timeout is None:
            timeout = self._default_step_timeout_s()
            if timeout:
                logger.debug("step_default_timeout_applied", step=step.name,
                             timeout_s=timeout)
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    def _default_step_timeout_s(self) -> float:
        """``[safety] step_timeout_s`` — the ceiling for a step that declares none.

        Cached per executor: this is read once per step dispatch, and re-parsing the
        config file on every step of a 32-channel run is a cost with no benefit.

        A step with no timeout runs forever, which is precisely the hazard an
        unattended campaign cannot absorb — it is indistinguishable from a hung
        instrument and no gate fires. Steps that legitimately run for hours already
        declare their own (``deposition_recipe.anneal_timeout_s``), so the default
        only reaches steps nobody bounded.

        ``0`` means unbounded, and is the deliberate escape hatch: a rig with a
        genuinely open-ended step can say so rather than raising the number until it
        stops mattering.
        """
        cached = getattr(self, "_default_timeout_cache", None)
        if cached is not None:
            return cached
        value = DEFAULT_STEP_TIMEOUT_S
        try:
            from softae.config.loader import safety

            value = float((safety() or {}).get("step_timeout_s",
                                               DEFAULT_STEP_TIMEOUT_S))
        except Exception:
            pass
        self._default_timeout_cache = value
        return value

    async def _run_control_step(self, method: str, params: dict[str, Any]) -> Any:
        """Built-in, instrument-free ``control`` steps (currently ``wait``)."""
        if method == "wait":
            seconds = float(params.get("seconds", 0.0))
            if seconds < 0:
                raise ValidationError_("wait 'seconds' must be non-negative")
            await asyncio.sleep(seconds)
            return {"waited_s": seconds}
        raise ValidationError_(f"Unknown control method '{method}'")

    async def _run_teardown(
        self,
        teardown_steps: list[WorkflowStep],
        total: int,
    ) -> None:
        """Best-effort teardown — errors are logged but not raised."""
        for step in teardown_steps:
            try:
                logger.info("teardown_step", step=step.name)
                if step.instrument == _CONTROL_INSTRUMENT:
                    method_params = {
                        k: v for k, v in step.params.items()
                        if k not in self._routing_params
                    }
                    await self._run_control_step(step.method, method_params)
                else:
                    # KNOWN INCONSISTENCY (pre-existing, preserved through the
                    # Tier-1 router extraction): unlike _dispatch, this branch
                    # passes step.params UNFILTERED — a teardown instrument step
                    # carrying routing-only params would forward them to the
                    # driver. No teardown step does today; do not fix silently,
                    # a fix changes observable driver kwargs.
                    inst = self.manager.get(step.instrument)
                    await inst.execute(step.method, **step.params)
            except Exception as exc:
                logger.error("teardown_step_failed", step=step.name, error=str(exc))

    async def _route_result(self, step: WorkflowStep, result: Any) -> None:
        """Offer *step*'s raw result to every matching registered router.

        The Tier-1 seam (spec §4): the executor no longer knows what an EIS
        step is — it only iterates routers. The context is rebuilt per call
        rather than cached because ``data_store``/``run_id`` are public-ish
        attributes a host (or a test) may swap after construction, and a stale
        snapshot here would silently route into the wrong run.

        Returned :class:`MeasurementResult` objects accumulate in
        :attr:`measurement_results` (Tier 2 component 2). The executor still
        does not interpret them — collecting is not consuming — so this stays a
        pure addition beside an unchanged persistence path.
        """
        ctx = RouterContext(
            data_store=self.data_store,
            run_id=self._run_id,
            manager=self.manager,
            sweep_counter=self._sweep_counter,
        )
        for router in self._routers:
            if router.matches(step):
                # Routers own their error handling (persistence must never fail
                # a step that physically succeeded — see EISResultRouter).
                measurement = await router.handle(step, result, ctx)
                if measurement is not None:
                    self.measurement_results.append(measurement)

    def _set_state(self, new: ExecutorState) -> None:
        old = self._state
        self._state = new
        if self.on_state_change:
            self.on_state_change(old, new)
