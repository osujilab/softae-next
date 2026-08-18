"""Safe Exit — the deliberate way out, with a say over the dispenser head.

Sits opposite the Emergency Stop in the toolbar and reads as its calmer sibling:
same shape, same weight, amber instead of red. The colour is the whole message —
this is a *planned* stop, not an emergency, and pressing it is not an admission
that something went wrong.

**Why it exists.** Closing the window parks the rig and raises the head, which is
the correct default when someone walks away: nobody is present to decide, and a
raised head cannot be sitting in a drop or pressed against a board. But retracting
is not universally the safe act. A head left deliberately lowered is *holding a
position* — an anneal hold in the flush basin, a mid-cast pause, an in-drop mix —
and raising it drags the tip out of the drop it is in. That judgement belongs to
the person standing at the rig, so this button asks them, once, and only when the
head is actually down. Everything else about the park is identical.

The prompt is deliberately three-way. "Leave it down" and "raise it" are both
legitimate outcomes, so neither is hidden behind the other, and Cancel is there
because discovering the head is down is itself sometimes the news that stops you
exiting at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox, QPushButton

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


def head_is_down(manager: "InstrumentManager") -> bool:
    """Whether the dispenser head is known to be lowered.

    Several states collapse to ``False`` on purpose, and the collapse is toward *not
    asking*: no syringe, a disconnected one, a driver that does not track the head, or
    a state that cannot be read. The same posture as
    :func:`softae.drivers.contracts.check_head_clear` — "do not invent a belief" — and
    here it means the operator is never asked a question about hardware whose state
    nothing actually knows.

    **Each of those paths logs its reason.** They are indistinguishable from the
    outside — the prompt simply does not appear — and "why didn't it ask me this time?"
    is otherwise unanswerable after the fact. The head genuinely being up is by far the
    most common reason, and it is the only one that is not worth investigating.
    """
    try:
        syringe = manager.get("syringe")
    except Exception:
        logger.debug("safe_exit_no_head_prompt", reason="no syringe registered")
        return False
    if syringe is None:
        logger.debug("safe_exit_no_head_prompt", reason="no syringe registered")
        return False
    if not getattr(syringe, "is_connected", False):
        logger.debug("safe_exit_no_head_prompt", reason="syringe not connected")
        return False
    is_up = getattr(syringe, "is_head_up", None)
    if not callable(is_up):
        logger.debug("safe_exit_no_head_prompt",
                     reason="driver does not track head state")
        return False
    try:
        up = bool(is_up())
    except Exception:
        logger.debug("safe_exit_no_head_prompt", reason="head state unreadable",
                     exc_info=True)
        return False
    if up:
        logger.debug("safe_exit_no_head_prompt", reason="head is already up")
    return not up


class _SafeExitWorker(QThread):
    """Run the park off the GUI thread, so a slow serial port cannot freeze the exit.

    Drives the same :func:`softae.core.safe_park.safe_park` as the E-Stop button and
    an unattended campaign — there must not be two stop sequences that can drift.
    """

    done = Signal(list)

    def __init__(self, manager: "InstrumentManager", *, retract_head: bool,
                 parent=None):
        super().__init__(parent)
        self._manager = manager
        self._retract_head = retract_head

    def run(self) -> None:
        from softae.core.safe_park import safe_park

        result = safe_park(
            self._manager,
            reason="operator safe exit",
            retract_head=self._retract_head,
        )
        self.done.emit(list(result.errors))


class SafeExitButton(QPushButton):
    """Amber companion to the Emergency Stop: park deliberately, then close."""

    #: Emitted with the park reason the moment the exit is confirmed — *before* the
    #: sequence runs, so nothing can start actuating in the window between the press
    #: and its completion. Same latch contract as the E-Stop's ``parked``.
    parked = Signal(str)

    #: Emitted once the park has finished and the window should close. Separate from
    #: ``parked`` because the close must wait for the hardware, not for the click.
    exit_requested = Signal()

    LABEL = "⏻  SAFE EXIT"

    def __init__(self, manager: "InstrumentManager", parent=None):
        super().__init__(self.LABEL, parent)
        self._manager = manager
        self._worker: _SafeExitWorker | None = None
        self.setToolTip(
            "Park the rig and close. If the dispenser head is down you will be "
            "asked whether to raise it."
        )
        # Amber, matched to the E-Stop's geometry so the pair reads as one control
        # surface. Deliberately not red: this is a planned stop.
        self.setStyleSheet(
            "QPushButton {"
            "  background-color: #f57c00;"
            "  color: white;"
            "  font-size: 16px;"
            "  font-weight: bold;"
            "  padding: 10px 24px;"
            "  border-radius: 6px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #e65100;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f7f7f;"
            "}"
        )
        self.setMinimumHeight(44)
        self.clicked.connect(self._on_clicked)

    # ── The prompt ───────────────────────────────────────────────────────────

    def _ask_head_choice(self) -> bool | None:
        """``True`` raise, ``False`` leave lowered, ``None`` cancel the exit."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Safe Exit")
        box.setText("The dispenser head is recorded as **lowered**.")
        box.setInformativeText(
            "Recorded, not sensed — the head has no position feedback, so this is "
            "the last thing the software was told. Look at it before answering.\n\n"
            "Raise it before exiting, or leave it where it is?\n\n"
            "Leave it down if it is holding a position — an anneal hold, a paused "
            "cast, or a drop it is sitting in. Raising it will pull the tip clear."
        )
        raise_btn = box.addButton("Raise head, then exit",
                                  QMessageBox.ButtonRole.AcceptRole)
        leave_btn = box.addButton("Leave head down, then exit",
                                  QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        # Raising is the safer of the two exits, so it is what Enter and Escape reach.
        box.setDefaultButton(raise_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is cancel_btn:
            return None
        return clicked is not leave_btn

    # ── The sequence ─────────────────────────────────────────────────────────

    def _on_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # already parking — ignore re-click

        retract = True
        if head_is_down(self._manager):
            choice = self._ask_head_choice()
            if choice is None:
                return  # cancelled: nothing touched, window stays open
            retract = choice

        self.setEnabled(False)
        self.setText("⏻  PARKING…")
        self.parked.emit("operator safe exit")

        self._worker = _SafeExitWorker(self._manager, retract_head=retract,
                                       parent=self)
        self._worker.done.connect(self._on_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_done(self, errors: list) -> None:
        self._worker = None
        self.setEnabled(True)
        self.setText(self.LABEL)

        if errors:
            # A partial park is the one case where exiting anyway is the wrong
            # default: something refused to go safe, and closing the window removes
            # the operator's easiest way to see what.
            answer = QMessageBox.warning(
                self, "Safe Exit",
                "Some subsystems did not park:\n\n" + "\n".join(errors)
                + "\n\nClose anyway?",
                QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Close:
                return

        self.exit_requested.emit()
