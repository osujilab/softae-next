"""Emergency Stop button — always visible in the toolbar.

Sends stop commands to all instruments when pressed, then tells the operator
**what was commanded and what was checked**, which are not the same list.

Two things this deliberately does not do:

* It does not move the dispenser head as part of the stop. The head has no
  position feedback, so ``head_retract()`` is a conditional flip on a belief that
  may be wrong in either direction — and wrong in one of them it drives the head
  *down* onto the board as the response to an emergency. Instead the operator is
  asked, afterwards, which way the head is pointing; the answer is written with
  ``set_head_state`` (which issues no motion), and only then is a retract
  offered. **The operator is the sensor.**
* It does not claim the rig is safe. Every axis is a command that was sent;
  nothing reads back to confirm the hardware obeyed.

The stop itself is never delayed by a question — the pumps, heater and lamp are
commanded the moment the button is pressed, and every prompt happens after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QPushButton, QMessageBox

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


class _EStopWorker(QThread):
    """Run the full E-Stop sequence off the GUI thread.

    The sequence itself lives in :func:`softae.core.safe_park.safe_park` so the
    button and an unattended campaign drive the rig safe by *exactly* the same
    path — there must not be two stop sequences that can drift apart.

    Emits ``done(SafeParkResult)``. The whole result travels, not just the error
    strings: what the dialog has to say now depends on the commanded/verified/
    unverifiable split, and flattening it here would have thrown that away at the
    thread boundary.
    """

    done = Signal(object)

    def __init__(self, manager: "InstrumentManager", parent=None):
        super().__init__(parent)
        self._manager = manager

    def run(self) -> None:
        from softae.core.safe_park import safe_park

        # retract_head is left at its default (None): the stop issues no head
        # motion. The operator is asked afterwards — see _ask_head_state.
        result = safe_park(self._manager, reason="operator emergency stop")
        self.done.emit(result)


class EmergencyStopButton(QPushButton):
    """Large red emergency-stop button.

    When clicked, disables itself and runs the full stop sequence on a
    background thread so the GUI stays responsive.  Re-enables and reports what
    was commanded once the sequence completes.
    """

    #: Emitted with the park reason the moment the stop is requested — *before*
    #: the sequence runs, so nothing else can start actuating while it is in
    #: flight. The anti-clog purge timer and the window's park latch take this.
    parked = Signal(str)

    def __init__(self, manager: "InstrumentManager", parent=None):
        super().__init__("⛔  EMERGENCY STOP", parent)
        self._manager = manager
        self._worker: _EStopWorker | None = None
        self.setStyleSheet(
            "QPushButton {"
            "  background-color: #d32f2f;"
            "  color: white;"
            "  font-size: 16px;"
            "  font-weight: bold;"
            "  padding: 10px 24px;"
            "  border-radius: 6px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #b71c1c;"
            "}"
            "QPushButton:disabled {"
            "  background-color: #7f7f7f;"
            "}"
        )
        self.setMinimumHeight(44)
        self.clicked.connect(self._on_stop)

    def _on_stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return  # already in progress — ignore re-click

        self.setEnabled(False)
        self.setText("⛔  STOPPING…")
        # Latch first: the sequence runs on a worker thread, and nothing must be
        # able to actuate during the window between the press and its completion.
        self.parked.emit("operator emergency stop")

        self._worker = _EStopWorker(self._manager, parent=self)
        self._worker.done.connect(self._on_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    # ── After the stop: report honestly, then ask about the head ─────────────

    def _on_done(self, result) -> None:
        self._worker = None
        self.setEnabled(True)
        self.setText("⛔  EMERGENCY STOP")

        self._report(result)
        try:
            self._ask_head_state()
        except Exception:       # a prompt must never be what fails after a stop
            logger.warning("estop_head_prompt_failed", exc_info=True)

    def _report(self, result) -> None:
        """Say what was commanded versus what was checked.

        This replaces *"All instruments stopped / safe."* — a sentence the system
        is not in a position to assert about any axis, and can never assert about
        the head. ``result.ok`` is still what picks the icon, because a refusal is
        the louder finding; it is no longer what picks the *words*.
        """
        text = ("Stop commands were issued." if result.ok
                else "PARTIAL STOP — something refused to go safe.")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning if not result.ok
                    else QMessageBox.Icon.Information)
        box.setWindowTitle("Emergency Stop")
        box.setText(text)
        box.setInformativeText(result.describe())
        box.exec()

    def _ask_head_state(self) -> None:
        """Ask which way the head is pointing, record it, then offer a retract.

        Three answers, and "Not sure" is a real one: it records nothing and moves
        nothing, which is strictly better than a guess written into the belief
        that later paths will act on. Nothing here is a gate — the operator can
        dismiss all of it and go to Manual Control.
        """
        syringe = self._syringe()
        if syringe is None:
            return

        is_up = self._ask_head_choice()
        if is_up is None:
            logger.warning("estop_head_state_unknown",
                           msg="operator did not confirm head position")
            return

        try:
            syringe.set_head_state(is_up)      # records; issues no motion
        except Exception:
            logger.warning("estop_set_head_state_failed", exc_info=True)
            return
        logger.info("estop_head_state_recorded", is_up=is_up)

        self._offer_retract(syringe, is_up)

    def _ask_head_choice(self) -> bool | None:
        """``True`` up, ``False`` down, ``None`` the operator did not say."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Emergency Stop — dispenser head")
        box.setText("Where is the dispenser head?")
        box.setInformativeText(
            "The stop did NOT move it. There is no sensor on the head, so the "
            "software cannot tell — look at it and answer from what you see, not "
            "from memory.\n\nYour answer only records the position; it moves "
            "nothing by itself."
        )
        up_btn = box.addButton("Head is UP (retracted)",
                               QMessageBox.ButtonRole.AcceptRole)
        down_btn = box.addButton("Head is DOWN (lowered)",
                                 QMessageBox.ButtonRole.DestructiveRole)
        unsure_btn = box.addButton("Not sure", QMessageBox.ButtonRole.RejectRole)
        box.setEscapeButton(unsure_btn)
        box.exec()

        clicked = box.clickedButton()
        if clicked is up_btn:
            return True
        if clicked is down_btn:
            return False
        return None      # "Not sure", Escape, or dismissed

    def _offer_retract(self, syringe, is_up: bool) -> None:
        """Offer a retract now that the belief is truthful.

        Offered in **both** branches, not only when the head is down: the offer is
        unconditional, and it is ``head_retract()`` that is conditional — which is
        now safe, because the operator has just told it the truth. From "head is
        UP" it correctly does nothing.
        """
        if not self._confirm_retract(is_up):
            return
        try:
            syringe.head_retract()
            logger.warning("estop_head_retract_commanded",
                           msg="commanded on the operator's instruction; "
                               "not confirmed — no sensor")
        except Exception as exc:
            QMessageBox.warning(self, "Emergency Stop",
                                f"Retract failed: {exc}")

    def _confirm_retract(self, is_up: bool) -> bool:
        answer = QMessageBox.question(
            self, "Emergency Stop — dispenser head",
            ("Head recorded as " + ("UP." if is_up else "DOWN.")
             + "\n\nRetract it now?"
             + ("\n\nIt is already recorded as up, so this will do nothing."
                if is_up else
                "\n\nLeave it down if it is holding a position — an anneal hold, "
                "a paused cast, or a drop it is sitting in.")),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _syringe(self):
        """The syringe, or ``None`` — absent, disconnected or untracked head."""
        try:
            syringe = self._manager.get("syringe")
        except Exception:
            return None
        if syringe is None or not getattr(syringe, "is_connected", False):
            return None
        if not callable(getattr(syringe, "set_head_state", None)):
            return None
        return syringe
