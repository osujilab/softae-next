"""The hardware-run harness shared by every autonomous surface (P2.4).

`BOTabBase` already shares the *generic GUI* machinery between the two BO tabs —
worker lifecycle, convergence plot, log pane, config save/load. What it never
covered is the machinery for actually driving the rig unattended: the
head-position start-gate, the two board gates, OS shutdown blocking, the overflow
pre-flight, and campaign execution. All of that lived inside
``LiveBOCampaignTab``.

That made Live BO the *owner* of autonomous execution rather than one instance of
it, so a second surface — the general Autonomous tab, or a headless CLI — would
have had to re-implement the gates, and any one of them could silently acquire a
weaker safety posture than the others. These gates are not incidental UI: the
board-freshness prompt is what stops a resumed campaign re-casting into used
wells, and the bounded waits are what stop an unanswered modal from holding a
multi-day campaign open forever.

This mixin holds that harness so no surface owns it. It deliberately stops short
of anything Bayesian — no parameter space, optimizer, or convergence plotting —
because the general façade must be able to run a spec that has none of those.

Host requirements: a ``QWidget`` (Qt signals resolve through the mixin once the
concrete class inherits ``QObject``), plus ``self._manager``, ``self._data_store``,
and a ``_sig_log`` signal for worker-thread log lines. Call
:meth:`_init_autonomous_run` from ``__init__``.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMessageBox

from softae.core.autonomous_loop import (
    DEFAULT_GATE_TIMEOUT_S,
    BoardCheck,
    BoardDecision,
)

logger = structlog.get_logger(__name__)


class AutonomousRunMixin:
    """Head gate, board gates, shutdown blocking, pre-flight, and execution."""

    #: Worker → GUI modal marshalling. Declared here so every host gets them.
    _sig_board_prompt = Signal(int)          # board index
    _sig_board_check = Signal(int, object)   # board id, occupied set

    def _init_autonomous_run(self) -> None:
        """Wire the gate marshalling. Call from the host's ``__init__``."""
        self._board_decision: BoardDecision | None = None
        self._board_event = threading.Event()
        self._sig_board_prompt.connect(self._on_board_prompt)

        self._board_check_decision: BoardCheck | None = None
        self._board_check_event = threading.Event()
        self._sig_board_check.connect(self._on_board_check_prompt)

        self._shutdown_hwnd: int = 0

    # ── Head-position start-gate ─────────────────────────────────────────────

    def _verify_head_position(self, context: str = "starting the campaign") -> bool:
        """Confirm the dispenser-head position before a run. ``False`` aborts.

        The loop drives the head with *conditional* commands, so a stale belief
        causes exactly one wrong flip. See
        :func:`softae.gui.widgets.head_check_dialog.verify_head_before_run`.
        """
        from softae.gui.widgets.head_check_dialog import verify_head_before_run

        return verify_head_before_run(self, self._manager, context=context)

    # ── Single occupancy — one campaign owns the rig at a time (S5.I) ────────

    def _refuse_if_rig_busy(self, spec: Any = None) -> bool:
        """``True`` when a live foreign run lock refuses this launch.

        **Outright, never queued and never a takeover prompt.** A second
        automated run on one rig is the collision the lock exists to prevent,
        and offering to take the rig from the button that starts a campaign is
        how a live overnight run gets killed by someone who only meant to start
        theirs. Taking the rig stays where it already is: a deliberate,
        separately-confirmed act (``break_run_lock``).

        The predicate is the CLI's, imported rather than re-derived — ``softae-
        campaign run`` refuses on exactly this pair (``tools/campaign.py``) — so
        the two surfaces cannot come to disagree about what "busy" means. That
        includes the simulation exemption: a mock manager claims no lock and
        moves nothing, so it is not refused over hardware it will never touch.

        ``foreign_run_lock`` is called **unwrapped**, not through
        ``rig_owner.foreign_rig_lock``: the never-raises wrapper decorates a
        *view*, and a launch that cannot find out who owns the rig must not
        start blind. Raising here refuses too, and loudly.

        A refusal costs the operator nothing — see
        :func:`softae.core.rejected_launch.preserve_rejected_launch`.
        """
        from softae.core.run_lock import (
            busy_rig_message,
            foreign_run_lock,
            rig_is_simulated,
        )

        holder = foreign_run_lock()
        if holder is None or rig_is_simulated(self._manager):
            return False

        preserved = self._preserve_rejected_launch(spec)
        self._sig_log.emit(
            f"✗ Not started — {holder.describe().splitlines()[0]}")
        QMessageBox.warning(
            self, "The rig is already running a campaign",
            busy_rig_message(holder, action="This campaign")
            + "\n\n" + preserved.describe(),
        )
        return True

    def _preserve_rejected_launch(self, spec: Any = None):
        """Write the refused configuration where the operator can get it back.

        The **panel state** is the lossless format and is what is always
        written; the spec is offered to the completeness check, which decides
        whether a terminal command may be handed over at all.
        """
        from softae.core.rejected_launch import preserve_rejected_launch

        panel_state_fn = getattr(self, "_panel_state", None)
        try:
            panel_state = panel_state_fn() if callable(panel_state_fn) else None
        except Exception:
            logger.warning("panel_state_unavailable", exc_info=True)
            panel_state = None
        return preserve_rejected_launch(
            project_dir=self._project_dir(), panel_state=panel_state, spec=spec)

    def _project_dir(self):
        """Where this surface's files live — the store's project, or the data root."""
        from pathlib import Path

        project = getattr(getattr(self, "_data_store", None), "project_dir", None)
        if project:
            return Path(project)
        from softae.config import loader

        return Path(loader.data_root())

    # ── Board-exchange gate (worker thread blocks on the GUI decision) ───────

    def _board_exchange_gate(self, board_index: int) -> BoardDecision:
        """``on_board_exchange`` callback — runs on the WORKER thread."""
        # Don't ask for a plate that does not exist (P5.4). When the inventory is
        # declared and empty, cancel without prompting — an operator woken at
        # 3 a.m. to install a board they do not have is a worse outcome than a
        # clean stop with the reason recorded.
        inventory = getattr(self, "_board_inventory", None)
        if inventory is not None and inventory.is_managed:
            if (inventory.remaining() or 0) <= 0:
                self._sig_log.emit(
                    "No spare electrode boards declared as remaining — "
                    "cancelling instead of prompting for a plate that is not there."
                )
                return BoardDecision.CANCEL

        self._board_decision = None
        self._board_event.clear()
        self._sig_board_prompt.emit(int(board_index))
        # Bounded: this blocks the worker thread synchronously, so the loop's own
        # asyncio gate timeout cannot reach it — the wait must bound itself or an
        # unanswered modal hangs the campaign indefinitely. Timing out is
        # equivalent to declining, which stops safely without casting.
        if not self._board_event.wait(timeout=DEFAULT_GATE_TIMEOUT_S):
            self._sig_log.emit(
                f"Board-exchange prompt unanswered after "
                f"{DEFAULT_GATE_TIMEOUT_S:.0f}s - cancelling the run."
            )
            return BoardDecision.CANCEL
        return self._board_decision or BoardDecision.CANCEL

    def _on_board_prompt(self, board_index: int) -> None:
        """GUI-thread slot: ask the operator to swap the board or cancel."""
        resp = QMessageBox.question(
            self,
            "Electrode board full",
            f"The current electrode board is full.\n\n"
            f"Insert fresh board #{board_index} and click Yes to continue the "
            f"campaign (the new board will be equilibrated first), or No to "
            f"cancel the run (e.g. if this overflow was unintended).",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        self._board_decision = (
            BoardDecision.PROCEED
            if resp == QMessageBox.StandardButton.Yes
            else BoardDecision.CANCEL
        )
        # A confirmed swap consumes a plate. Decremented only on PROCEED, so a
        # cancelled prompt does not silently spend inventory.
        if self._board_decision is BoardDecision.PROCEED:
            inventory = getattr(self, "_board_inventory", None)
            if inventory is not None and inventory.is_managed:
                left = inventory.consume()
                self._sig_log.emit(f"Board installed; {left} spare board(s) left.")
        self._board_event.set()

    # ── Board-freshness gate (resume safety) ─────────────────────────────────

    def _board_check_gate(self, board_id: int, occupied: set[int]) -> BoardCheck:
        """``on_board_check`` callback (WORKER thread) — resume-safety prompt."""
        self._board_check_decision = None
        self._board_check_event.clear()
        self._sig_board_check.emit(int(board_id), set(occupied))
        # Bounded for the same reason as the exchange gate; an unanswered
        # freshness prompt must not hold the campaign open forever. Timing out
        # cancels, which is the safe answer (never re-cast into used wells).
        if not self._board_check_event.wait(timeout=DEFAULT_GATE_TIMEOUT_S):
            self._sig_log.emit(
                f"Board-freshness prompt unanswered after "
                f"{DEFAULT_GATE_TIMEOUT_S:.0f}s - cancelling the run."
            )
            return BoardCheck.CANCEL
        return self._board_check_decision or BoardCheck.CANCEL

    def _on_board_check_prompt(self, board_id: int, occupied: set[int]) -> None:
        """GUI-thread slot: is the electrode board fresh, the same, or cancel?"""
        n = len(occupied)
        box = QMessageBox(self)
        box.setWindowTitle("Electrode board — recorded occupancy")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"Board #{board_id} already has {n} recorded cast(s) "
            f"(single-use wells).\n\nIs a freshly replaced board loaded?"
        )
        box.setInformativeText(
            "Fresh board — start clean on a new board.\n"
            "Same board — resume, skipping the already-used wells.\n"
            "Cancel — stop (e.g. to avoid an accidental re-cast)."
        )
        b_fresh = box.addButton("Fresh board", QMessageBox.ButtonRole.AcceptRole)
        b_same = box.addButton("Same board (resume)", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_fresh:
            self._board_check_decision = BoardCheck.FRESH
        elif clicked is b_same:
            self._board_check_decision = BoardCheck.RESUME
        else:
            self._board_check_decision = BoardCheck.CANCEL
        self._board_check_event.set()

    def _release_board_gates(self) -> None:
        """Unblock any pending gate so the worker unwinds on abort.

        Both are released as CANCEL: an abort must never be interpreted as
        consent to keep casting.
        """
        self._board_decision = BoardDecision.CANCEL
        self._board_event.set()
        self._board_check_decision = BoardCheck.CANCEL
        self._board_check_event.set()

    # ── OS shutdown blocking (best-effort, Windows) ──────────────────────────

    def _acquire_shutdown_block(self, campaign_name: str) -> None:
        """Ask the OS not to restart under a running campaign.

        Best-effort — policy can override it — which is why it is one layer
        alongside park-on-exit and unclean-shutdown detection at next launch.
        """
        try:
            from softae.gui.shutdown_guard import block_shutdown

            win = self.window()
            hwnd = int(win.winId()) if win is not None else 0
            self._shutdown_hwnd = hwnd
            block_shutdown(hwnd, f"SoftAE is running the '{campaign_name}' campaign.")
        except Exception:
            logger.warning("shutdown_block_acquire_failed", exc_info=True)

    def _release_shutdown_block(self) -> None:
        try:
            from softae.gui.shutdown_guard import unblock_shutdown

            hwnd = getattr(self, "_shutdown_hwnd", 0)
            if hwnd:
                unblock_shutdown(hwnd)
        except Exception:
            logger.warning("shutdown_block_release_failed", exc_info=True)

    # ── Overflow pre-flight ─────────────────────────────────────────────────

    def _preflight_overflow_ok(self, spec) -> bool:
        """Scan the whole parameter space for well overflow before starting.

        Advisory: warns up-front rather than surfacing an infeasible suggestion
        mid-run. **A pre-flight failure never blocks the run** — a scan that
        cannot run is not evidence of a problem.
        """
        from softae.core.autonomous_wiring import preflight_overflow

        try:
            result = preflight_overflow(spec)
        except Exception as exc:  # advisory only — never block on a scan failure
            logger.warning("preflight_overflow_failed", error=str(exc))
            return True
        if not result.any_overflow:
            return True

        worst_point, worst = result.worst
        pct = 100.0 * result.overflow_fraction
        msg = (
            f"{result.n_overflow} of {result.n_points} sampled composition points "
            f"({pct:.0f}% of the search space) exceed the "
            f"{result.capacity_uL:.1f} µL well capacity at this deposition volume.\n\n"
            f"Worst: {self._fmt_params(worst_point)} → {worst.total_uL:.1f} µL "
            f"({-worst.headroom_uL:.1f} µL over).\n\n"
            f"Lower the target deposition volume (or narrow the ranges) to avoid "
            f"infeasible casts mid-run. Proceed anyway?"
        )
        resp = QMessageBox.warning(
            self, "Overflow pre-flight", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    def _preflight_projection_ok(self, spec, *, catalog=None) -> bool:
        """Show projected duration and stock runway; let the operator decide.

        Reports a **rate with bounds**, not an ETA — a BO campaign stops on a
        convergence criterion, so the iteration count is unknown in advance.
        Only *blocks* when the declared stock cannot cover the budget, since
        that is a predicted hard-stop mid-run rather than a matter of taste;
        undeclared stock is unknown, not insufficient, and never blocks.
        """
        from softae.core.preflight import project_campaign

        try:
            if catalog is None:
                from softae.config import loader
                from softae.core.task_catalog import TaskCatalog

                catalog = TaskCatalog.load_toml(loader.tasks_toml_path())
            ledger = getattr(self._manager.get("syringe"), "reservoir_ledger", None)
            # Purge consumption accrues with elapsed time, so a multi-day
            # campaign's purge bill can exceed its trial draw entirely (P8).
            from softae.core.purge import load_purge_settings

            purge = load_purge_settings(self._data_store).uL_per_day()
            projection = project_campaign(
                spec, catalog=catalog, ledger=ledger, purge_uL_per_day=purge)
        except Exception:
            # Advisory: a projection that cannot run is not a reason to refuse
            # a campaign the operator has asked for.
            logger.warning("preflight_projection_failed", exc_info=True)
            return True

        summary = projection.describe()
        self._sig_log.emit(summary.replace("\n", " | "))

        if projection.stock_sufficient is not False:
            return True

        resp = QMessageBox.warning(
            self, "Not enough stock for this campaign",
            summary + "\n\nStart anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    @staticmethod
    def _fmt_params(params: dict[str, Any]) -> str:
        """Fallback formatter; hosts may override (the Live BO tab does).

        Matches the Live BO tab's formatting so a pre-flight warning reads the
        same on every surface.
        """
        return ", ".join(f"{k}={float(v):g}" for k, v in params.items())

    # ── Execution ───────────────────────────────────────────────────────────

    def _execute_campaign(self, spec, *, on_event, aborted_exc: type[Exception]):
        """Worker thread: run the autonomous loop in a fresh event loop.

        Returns the :class:`CampaignResult`, or ``None`` if the run aborted or
        raised — in which case ``_sig_done`` has already been emitted, so the
        caller has nothing left to report.
        """
        import asyncio

        from softae.core.autonomous_wiring import run_autonomous_campaign

        # Rebind the manager's per-instrument locks to THIS thread's event loop.
        try:
            self._manager.reset_locks()
        except Exception:
            pass

        self._acquire_shutdown_block(spec.name)
        # Claims the rig for the whole run (so the purge timer defers rather
        # than competing) and returns it to idle rest on every exit path.
        try:
            with self._rig_run(f"campaign:{spec.name}"):
                result = asyncio.run(
                    run_autonomous_campaign(
                        spec,
                        manager=self._manager,
                        data_store=self._data_store,
                        objective_extractor=None,
                        on_event=on_event,
                        approval_fn=None,
                        on_board_exchange=self._board_exchange_gate,
                        on_board_check=self._board_check_gate,
                    )
                )
            best = result.best_objective
            best_txt = f"; best objective {best:.4g}" if best is not None else ""
            self._sig_done.emit(
                True,
                f"{result.final_state.lower()} after {result.n_trials} trials{best_txt}",
            )
            return result
        except aborted_exc:
            self._sig_done.emit(True, "aborted")
            return None
        except Exception as exc:
            logger.exception("autonomous_campaign_error", error=str(exc))
            self._sig_done.emit(False, str(exc))
            return None
        finally:
            self._release_shutdown_block()

    def _rig_run(self, owner: str):
        """The window's run wrapper — ownership + return-to-idle-rest.

        Falls back to a no-op context if the host window does not provide it, so
        the mixin stays usable in isolation (and in tests) without the GUI shell.
        """
        from contextlib import nullcontext

        factory = getattr(self.window(), "rig_run", None)
        return factory(owner) if callable(factory) else nullcontext()
