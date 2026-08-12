"""Emergency Stop button — always visible in the toolbar.

Sends stop commands to all instruments when pressed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QPushButton, QMessageBox

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class _EStopWorker(QThread):
    """Run the full E-Stop sequence off the GUI thread.

    The sequence itself lives in :func:`softae.core.safe_park.safe_park` so the
    button and an unattended campaign drive the rig safe by *exactly* the same
    path — there must not be two stop sequences that can drift apart.

    Emits ``done(list)`` with any error strings (empty list = all clean).
    Signals are delivered back to the main thread via Qt queued connections.
    """

    done = Signal(list)

    def __init__(self, manager: "InstrumentManager", parent=None):
        super().__init__(parent)
        self._manager = manager


    def run(self) -> None:
        from softae.core.safe_park import safe_park

        result = safe_park(self._manager, reason="operator emergency stop")
        self.done.emit(list(result.errors))


class EmergencyStopButton(QPushButton):
    """Large red emergency-stop button.

    When clicked, disables itself and runs the full stop sequence on a
    background thread so the GUI stays responsive.  Re-enables and reports
    any errors once the sequence completes.
    """

    #: Emitted with the park reason the moment the stop is requested — *before*
    #: the sequence runs, so nothing else can start actuating while it is in
    #: flight. The anti-clog purge timer latches this.
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

    def _on_done(self, errors: list) -> None:
        self._worker = None
        self.setEnabled(True)
        self.setText("⛔  EMERGENCY STOP")

        if errors:
            QMessageBox.warning(
                self, "Emergency Stop",
                "Partial stop — some errors:\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(self, "Emergency Stop", "All instruments stopped / safe.")
