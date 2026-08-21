"""The dialog an attached window's E-Stop opens — the ladder, rung by rung.

All the rules live in :mod:`softae.gui.estop_ladder`, which has no Qt in it. This
is the surface: it drives :meth:`~softae.gui.estop_ladder.EstopLadder.poll` on a
one-second timer, renders what the ladder says, and turns the two operator acts —
*keep waiting* and *take the rig* — into buttons.

**Nothing here decides anything.** The one property the whole step turns on is
that no timer reaches the kill, and it is enforced in the ladder rather than in
this dialog's wiring: the timer's only call is ``poll()``, and ``poll()`` cannot
enter rung 3 or perform rung 4. This file could not auto-advance if it tried.

What the operator sees while waiting is **elapsed seconds against the rung's
budget, and the newest line of ``events.jsonl``**, refreshed every second. That
pairing is the whole reason the wait is watchable: a rising elapsed clock beside
a changing event line is a campaign working through a step, and the same clock
beside a frozen line is a campaign that has stopped answering. CLAUDE.md §5 asks
a human deciding whether to kill a process to make exactly that distinction
before they do; this puts the evidence in front of them.

The typed confirmation
----------------------
Rung 4 terminates a process. The confirmation is not a Yes/No — it asks the
operator to type the PID back, with ``lock.describe()`` above it, because PID
reuse is an acknowledged and unmitigated limit of the rig lock: the number may
belong to something unrelated that the operating system handed it after the
campaign died. ``started_at`` and ``what`` are the only evidence that it does
not, so they are shown, and the prompt says what they are for.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import structlog
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from softae.gui.estop_ladder import (
    OFFERED_STATES,
    STATE_OFFERED_WAIT,
    TERMINAL_STATES,
    EstopLadder,
    TakeoverResult,
    holder_noun,
)

logger = structlog.get_logger(__name__)

#: How often the ladder is polled. Matched to the campaign's own control poll
#: (``DEFAULT_CONTROL_POLL_S``): reading faster cannot make an answer arrive
#: sooner, and the elapsed clock is what the operator is actually watching.
POLL_MS = 1000

#: The button text for each act. Named so a test can assert which act is on
#: offer without matching prose, and so the takeover's label can never read as
#: the mild one.
ACT_WAIT = "Keep waiting for the park"
ACT_TAKE_OVER = "TAKE THE RIG — break the lock and terminate the campaign"


class EstopLadderDialog(QDialog):
    """Drive one E-Stop ladder to its end, and report what it reached.

    Parameters
    ----------
    ladder
        The :class:`~softae.gui.estop_ladder.EstopLadder`. Already knows which
        rungs it may reach.
    schedule
        How to run rung 4's coroutine. The GUI runs on **qasync**, so a loop is
        already driving Qt and the instruments' ``asyncio.Lock``s are bound to
        it; making a second loop here would either raise "already running" or
        hand ``connect_all()`` locks from the wrong one. Injectable so a test can
        drive the takeover with no loop at all — the same seam, and for the same
        reason, as :class:`~softae.gui.widgets.calibration_launcher.CalibrationLauncherDialog`.
    confirm
        ``(pid, evidence) -> bool``. Replaced in tests; never bypassed.
    """

    #: Emitted the moment rung 4 is confirmed and **before** it runs, so the
    #: window's park latch closes over the whole takeover rather than only over
    #: its result. Same discipline as the owner-mode button's ``parked``.
    takeover_started = Signal(str)

    #: Emitted with the :class:`~softae.gui.estop_ladder.TakeoverResult`.
    takeover_done = Signal(object)

    def __init__(
        self,
        ladder: EstopLadder,
        *,
        parent: Any = None,
        schedule: Callable[[Any], Any] | None = None,
        confirm: Callable[[int, str], bool] | None = None,
        poll_ms: int = POLL_MS,
    ) -> None:
        super().__init__(parent)
        self._ladder = ladder
        self._schedule = schedule or self._default_schedule
        self._confirm = confirm or self._ask_to_confirm
        self.setWindowTitle("Emergency Stop — the campaign owns the rig")
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)

        self._lbl_head = QLabel(self._header())
        self._lbl_head.setWordWrap(True)
        self._lbl_head.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._lbl_head)

        self._lbl_note = QLabel("")
        self._lbl_note.setWordWrap(True)
        layout.addWidget(self._lbl_note)

        # The wedged-versus-working evidence: elapsed clock and newest event.
        self._lbl_watch = QLabel("")
        self._lbl_watch.setWordWrap(True)
        self._lbl_watch.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._lbl_watch.setStyleSheet("font-family: Consolas, monospace;")
        layout.addWidget(self._lbl_watch)

        self._btn_act = QPushButton("")
        self._btn_act.clicked.connect(self._on_act)
        layout.addWidget(self._btn_act)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_ms))
        # The timer's *only* call. Rung 3 is entered by `advance()` and rung 4 by
        # `take_over()`, and `poll()` reaches neither — see the module docstring.
        self._timer.timeout.connect(self._tick)

    # ── Rung 1, on open ──────────────────────────────────────────────────────

    def begin(self) -> bool:
        """Write the abort and start the clock. ``False`` if it could not be written.

        Separate from ``__init__`` so a caller can construct the dialog without
        writing anything — and so the one failure the operator must see (the
        write itself) is reported before the wait begins rather than inside it.
        """
        if self._ladder.run_dir is not None:
            try:
                self._ladder.start()
            except Exception as exc:
                logger.warning("estop_abort_write_failed", exc_info=True)
                QMessageBox.critical(
                    self, "Emergency Stop",
                    f"The abort could NOT be written ({exc.__class__.__name__}: "
                    f"{exc}).\n\nThe campaign has not been asked to stop. If the "
                    "rig must stop now, stop the campaign at its own terminal.")
                self._render()
                return False
        self._render()
        self._timer.start()
        return True

    # ── The wait ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        state = self._ladder.poll()
        if state in TERMINAL_STATES:
            self._timer.stop()
        self._render()

    def _render(self) -> None:
        ladder = self._ladder
        self._lbl_note.setText(ladder.note)
        self._lbl_watch.setText(
            f"{self._elapsed_text()}\nlast event: {ladder.last_event_line}")
        self._apply_act_button()

    def _elapsed_text(self) -> str:
        budget = self._ladder.budget_s
        elapsed = self._ladder.elapsed_s
        if budget is None:
            return f"waiting {elapsed:.0f}s"
        return f"waiting {elapsed:.0f}s of {budget:.0f}s"

    def _apply_act_button(self) -> None:
        if self._ladder.may_take_over:
            self._btn_act.setText(ACT_TAKE_OVER)
            self._btn_act.setStyleSheet(
                "QPushButton { background-color: #d32f2f; color: white; "
                "font-weight: bold; padding: 8px; }")
            self._btn_act.setEnabled(True)
            self._btn_act.setVisible(True)
            return
        if self._ladder.may_advance:
            self._btn_act.setText(ACT_WAIT)
            self._btn_act.setStyleSheet("")
            self._btn_act.setEnabled(True)
            self._btn_act.setVisible(True)
            return
        self._btn_act.setVisible(False)

    def _header(self) -> str:
        ladder = self._ladder
        name = (ladder.campaign or (None, ""))[0] or holder_noun(ladder.holder_kind)
        rungs = ladder.reachable_rungs
        reach = (
            " → ".join(str(rung) for rung in rungs) if rungs else "none")
        limit = ""
        if ladder.cross_host:
            limit = (
                "<br><b>This campaign is on another machine.</b> Its rig lock "
                "reads as alive by design — there is no process id here to check "
                "and none to stop — so this window can ask it to abort and "
                "nothing more."
            )
        return (
            f"<b>{name}</b> holds the rig. This window opened no instrument "
            f"session, so it cannot park anything itself; the stop has to reach "
            f"the process that can.<br>Rungs reachable from here: <b>{reach}</b>."
            f"{limit}"
        )

    # ── The two operator acts ────────────────────────────────────────────────

    def _on_act(self) -> None:
        state = self._ladder.state
        if state == STATE_OFFERED_WAIT:
            self._ladder.advance()
            self._render()
            return
        if self._ladder.may_take_over:
            self._begin_takeover()
            return
        logger.info("estop_ladder_act_ignored", state=state,
                    offered=state in OFFERED_STATES)

    def _begin_takeover(self) -> None:
        pid = self._ladder.takeover_pid
        if not self._confirm(pid, self._takeover_evidence()):
            return
        self._timer.stop()
        self._btn_act.setEnabled(False)
        self._btn_act.setText("Taking the rig…")
        self.takeover_started.emit("operator E-Stop — takeover")
        self._schedule(self._run_takeover())

    async def _run_takeover(self) -> None:
        result = await self._ladder.take_over(confirmed=True)
        self._on_takeover_done(result)

    def _on_takeover_done(self, result: TakeoverResult) -> None:
        self._render()
        self.takeover_done.emit(result)
        self.accept()

    def _takeover_evidence(self) -> str:
        lock = self._ladder.lock
        try:
            described = lock.describe()
        except Exception:
            described = "the rig lock could not be described."
        return (
            f"{described}\n\n"
            "Before you confirm, read the two facts above: what that process "
            "said it was running, and when it started. They are the only "
            "evidence that the process id still belongs to the campaign — the "
            "operating system reuses process ids, so a number left by a run that "
            "already died can point at something else entirely.\n\n"
            "This will terminate that one process id and nothing else, then open "
            "the instruments here and park them."
        )

    def _ask_to_confirm(self, pid: int, evidence: str) -> bool:
        """Type the PID back. A Yes/No would not carry the evidence."""
        typed, ok = QInputDialog.getText(
            self, "Take the rig from the campaign?",
            f"{evidence}\n\nType {pid} to confirm:")
        if not ok:
            return False
        return str(typed).strip() == str(pid)

    def _default_schedule(self, coro: Any) -> Any:
        task = asyncio.ensure_future(coro)
        task.add_done_callback(self._on_scheduled_done)
        return task

    def _on_scheduled_done(self, task: Any) -> None:
        try:
            task.result()
        except Exception:
            logger.warning("estop_takeover_failed", exc_info=True)
            self._on_takeover_done(
                TakeoverResult(refused="the takeover raised — see the log."))

    def closeEvent(self, event) -> None:                    # noqa: N802 (Qt)
        self._timer.stop()
        super().closeEvent(event)
