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

Attached mode: the button says what it can reach, before it is pressed
-----------------------------------------------------------------------
All of the above describes a window that **opened the sessions**. A window
attached to a campaign holds none, so pressing this cannot park anything: the
park would file every subsystem under ``skipped`` and report success having sent
nothing to the rig. The stop must instead reach the campaign, by the escalation
ladder in :mod:`softae.gui.estop_ladder` — and how far that ladder can go depends
on where the campaign is running.

So the label and the tooltip are decided **at construction**, from the launch
mode, and never on the press. A red button whose guarantee silently varies by
launch location is worse than one that admits its limit, and the limit is real:
:attr:`~softae.core.run_lock.RunLock.is_alive` reports a lock from another host
as always alive by design, so against a cross-host campaign there is no process
id here to check and none to stop.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any, Callable

import structlog
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QPushButton, QMessageBox

from softae.gui.estop_ladder import reachable_rungs
from softae.gui.launch_mode import OWNER_MODE, LaunchMode

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: The label for every press that can actually stop the rig — either by
#: commanding this process's own sessions, or by reaching the campaign that owns
#: them. Unchanged from the owner-mode button, deliberately: the operator's
#: muscle memory for the red rectangle is not something to redesign.
LABEL_STOP = "⛔  EMERGENCY STOP"

#: Cross-host. The press writes an abort and waits for the acknowledgement; it
#: cannot verify a park, cannot check a PID and cannot open an instrument.
LABEL_REQUEST_ONLY = "⛔  E-STOP — REQUEST ONLY"

#: Something on another machine holds the rig and it is not a campaign: no
#: control channel to write to, and no reachable process. Nothing this button
#: presses changes anything, and it says so rather than looking armed.
LABEL_UNREACHABLE = "⛔  E-STOP — UNAVAILABLE"

TOOLTIP_OWNER = (
    "Commands every instrument this window opened to a safe state: pumps "
    "halted, heater to its safe setpoint, lamp off. The dispenser head is not "
    "moved — you are asked about it afterwards."
)


def _cross_host(holder: Any) -> bool:
    """Whether *holder* is on another machine.

    The **same comparison** :attr:`~softae.core.run_lock.RunLock.is_alive` makes,
    and it has to be: that property is why the rungs differ. A lock whose host is
    not ours reads as permanently alive, so the liveness check that would decide
    whether a PID is worth stopping never runs — and the button must reflect the
    decision the lock has already taken, not a second opinion about it.

    An absent host is *not* foreign. A lock written before the field existed, or
    by something that left it blank, is treated as local — the same reading
    ``is_alive`` gives it.
    """
    host = str(getattr(holder, "host", "") or "")
    return bool(host and host != socket.gethostname())


def _label_for(mode: LaunchMode, rungs: tuple[int, ...]) -> str:
    """The button's text. Decided once, from the mode, before any press."""
    if not mode.attached:
        return LABEL_STOP
    if not rungs:
        return LABEL_UNREACHABLE
    if 4 in rungs:
        # Rungs 1-3 may or may not exist, but a confirmed takeover ends in a real
        # park issued from this process, so the plain label is not a promise this
        # button cannot keep.
        return LABEL_STOP
    return LABEL_REQUEST_ONLY


def _campaign_name(campaign: tuple[str, str] | None) -> str:
    if not campaign:
        return "another process"
    name, run_id = campaign
    return f"{name or 'a campaign'} (run {run_id})" if run_id else (name or "a campaign")


def attached_tooltip(
    campaign: tuple[str, str] | None,
    *,
    cross_host: bool,
    rungs: tuple[int, ...],
) -> str:
    """What this press can reach, named before it happens.

    Every branch names the holder and then states the *limit*, in that order. The
    cross-host sentence is the one that matters: it is the difference between a
    button that stops a rig and a button that sends a message, and an operator
    must not learn which they have by pressing it.
    """
    who = _campaign_name(campaign)
    if not rungs:
        return (
            f"The rig is held by {who} on another machine, and it is not a "
            "campaign — it reads no control file. This window can neither ask it "
            "to stop nor reach its instruments. Stop it at that machine."
        )
    if cross_host:
        return (
            f"Asks {who} to abort, and waits for it to acknowledge.\n\n"
            "That campaign is on ANOTHER MACHINE. Its rig lock reads as alive by "
            "design, so there is no process id here to check and none to stop, "
            "and its instruments are on a computer this one cannot reach. This "
            "button can request a stop and confirm the request was read — it "
            "cannot make one happen."
        )
    if rungs == (4,):
        return (
            f"The rig is held by {who}, which is not a campaign: it publishes no "
            "event stream and reads no control file, so there is nothing to "
            "request.\n\nThe only act left is taking the rig — breaking the lock, "
            "stopping that one process id and parking from here. It is manual, it "
            "is confirmed, and this window will never do it on its own."
        )
    return (
        f"Asks {who} to abort, then follows what it does.\n\n"
        "This window opened no instrument session, so it cannot park the rig "
        "itself. It writes the abort, waits for the acknowledgement (15 s), then "
        "waits for the park (120 s), showing the elapsed time and the campaign's "
        "newest event throughout. If the park never arrives it OFFERS to take the "
        "rig — it never does so on a timer."
    )


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

    In **owner mode** a click disables the button and runs the full stop sequence
    on a background thread so the GUI stays responsive, then reports what was
    commanded.

    In **attached mode** there is nothing to command, so a click opens the
    escalation ladder instead (:mod:`softae.gui.estop_ladder`). Which mode this
    is, and how far the ladder can climb, is fixed here in the constructor and
    written into the label and the tooltip — see the module docstring.

    Parameters
    ----------
    launch_mode
        The window's :class:`~softae.gui.launch_mode.LaunchMode`. ``None`` means
        owner mode, which is the historical behaviour and the right default for a
        process that opened its own sessions.
    ladder_factory, dialog_factory
        The ladder and its dialog, injected for tests. Nothing in a test may
        construct a real ladder by accident: rung 4 terminates a process.
    """

    #: Emitted with the park reason the moment the stop is requested — *before*
    #: the sequence runs, so nothing else can start actuating while it is in
    #: flight. The anti-clog purge timer and the window's park latch take this.
    #:
    #: In attached mode it fires only when the operator confirms rung 4, because
    #: that is the only press in that mode which commands anything: rungs 1-3 are
    #: requests, and latching the window's park on a request would announce a park
    #: that has not happened and may not.
    parked = Signal(str)

    def __init__(
        self,
        manager: "InstrumentManager",
        parent=None,
        *,
        launch_mode: LaunchMode | None = None,
        ladder_factory: Callable[[], Any] | None = None,
        dialog_factory: Callable[..., Any] | None = None,
    ):
        mode = launch_mode if launch_mode is not None else OWNER_MODE
        rungs = (
            reachable_rungs(run_dir=mode.run_dir, cross_host=_cross_host(mode.holder))
            if mode.attached else ()
        )
        super().__init__(_label_for(mode, rungs), parent)
        self._manager = manager
        self._launch_mode = mode
        self._cross_host = _cross_host(mode.holder)
        self._rungs = rungs
        self._ladder_factory = ladder_factory or self._build_ladder
        self._dialog_factory = dialog_factory or self._build_dialog
        self._dialog: Any = None
        self.setToolTip(
            attached_tooltip(mode.campaign, cross_host=self._cross_host, rungs=rungs)
            if mode.attached else TOOLTIP_OWNER
        )
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

    # ── What this button can reach, answerable without pressing it ───────────

    @property
    def attached(self) -> bool:
        """Whether this button belongs to a window that opened no sessions."""
        return self._launch_mode.attached

    @property
    def reachable_rungs(self) -> tuple[int, ...]:
        """The ladder rungs a press can enter. ``()`` in owner mode — no ladder."""
        return self._rungs

    def _on_stop(self) -> None:
        if self.attached:
            self._open_ladder()
            return

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

    # ── Attached mode: the ladder, not a park ────────────────────────────────

    def _open_ladder(self) -> None:
        """Open the escalation ladder. One at a time, and never modeless.

        The dialog is what waits; this method neither waits nor decides. The
        press that follows a press — the ladder's rung 4 — happens inside it,
        behind a confirmation, and reaches this object again only through
        :meth:`_on_takeover_done`.
        """
        if self._dialog is not None:
            return                      # already climbing — ignore re-click
        dialog = self._dialog_factory(self._ladder_factory(), parent=self)
        dialog.takeover_started.connect(self.parked.emit)
        dialog.takeover_done.connect(self._on_takeover_done)
        self._dialog = dialog
        try:
            if dialog.begin():
                dialog.exec()
        finally:
            self._dialog = None

    def _build_ladder(self):
        from softae.gui.estop_ladder import EstopLadder, default_ladder_collaborators

        mode = self._launch_mode
        return EstopLadder(
            mode.run_dir, lock=mode.holder, cross_host=self._cross_host,
            campaign=mode.campaign, manager=self._manager,
            **default_ladder_collaborators(),
        )

    def _build_dialog(self, ladder, *, parent=None):
        from softae.gui.widgets.estop_ladder_dialog import EstopLadderDialog

        return EstopLadderDialog(ladder, parent=parent)

    def _on_takeover_done(self, result) -> None:
        """Rung 4 finished. Report it the same way a park is always reported.

        Through :meth:`SafeParkResult.headline` — reached via
        :meth:`TakeoverResult.headline` — so a takeover whose ``connect_all``
        failed is headed *"NOTHING WAS COMMANDED"* rather than reported as a
        stop. The head prompt follows, because this process now holds the
        sessions and the question is finally one it can act on.
        """
        self._report(result)
        try:
            self._ask_head_state()
        except Exception:       # a prompt must never be what fails after a stop
            logger.warning("estop_head_prompt_failed", exc_info=True)

    # ── After the stop: report honestly, then ask about the head ─────────────

    def _on_done(self, result) -> None:
        self._worker = None
        self.setEnabled(True)
        self.setText(LABEL_STOP)

        self._report(result)
        try:
            self._ask_head_state()
        except Exception:       # a prompt must never be what fails after a stop
            logger.warning("estop_head_prompt_failed", exc_info=True)

    def _report(self, result) -> None:
        """Say what was commanded versus what was checked.

        This replaces *"All instruments stopped / safe."* — a sentence the system
        is not in a position to assert about any axis, and can never assert about
        the head.

        The headline is **not** derived here. It used to be, from ``result.ok``
        alone, which over-read it: ``ok`` means *nothing raised*, and a park
        against a disconnected rig raises nothing while commanding nothing, so
        the press of a red button could answer "Stop commands were issued." with
        an empty ``commanded`` list and a reassuring blue icon.
        :meth:`SafeParkResult.headline` now makes that choice once, for every
        surface that reports a park.
        """
        text, severe = result.headline()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning if severe
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
