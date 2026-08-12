"""Instrument status bar — sits at the bottom of the main window.

Shows a row of coloured dots indicating instrument connection state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QStatusBar, QWidget

from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class _StatusWorker(StoppableWorker):
    """Background thread that polls manager.status_all() every 2 s.

    Emits ``status_ready(dict)`` on the main thread via Qt's
    queued-connection mechanism.  Stops cleanly when
    ``requestInterruption()`` is called.
    """

    status_ready = Signal(dict)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager

    def run(self) -> None:
        while not self.isInterruptionRequested():
            if self._manager is not None:
                try:
                    statuses = self._manager.status_all()
                    self.status_ready.emit(statuses)
                except Exception:
                    pass
            self.msleep(2000)


_COLORS = {
    "CONNECTED": "#4CAF50",     # green
    "CONNECTING": "#FF9800",    # orange
    "DISCONNECTED": "#9E9E9E",  # grey
    "ERROR": "#f44336",         # red
}


class InstrumentStatusBar(QStatusBar):
    """Status bar with per-instrument connection indicators.

    Polls :meth:`InstrumentManager.status_all` every 2 seconds.
    Accepts an optional shared *poller* (:class:`~softae.gui.widgets.instrument_poller.InstrumentPoller`);
    when provided, the local ``_StatusWorker`` is not started and all polling
    is delegated to the shared thread, eliminating redundant serial I/O.
    """

    def __init__(self, manager: InstrumentManager, *, poller=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._indicators: dict[str, QLabel] = {}

        self._container = QWidget()
        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(4, 0, 4, 0)
        self.addPermanentWidget(self._container, 1)

        if poller is not None:
            poller.status_ready.connect(self._apply_statuses)
            self._status_worker = None  # shared poller used; no local worker needed
        else:
            self._status_worker = _StatusWorker(manager, parent=self)
            self._status_worker.status_ready.connect(self._apply_statuses)
            self._status_worker.start()

    def _apply_statuses(self, statuses: dict) -> None:
        for name, info in statuses.items():
            if name not in self._indicators:
                lbl = QLabel()
                self._layout.addWidget(lbl)
                self._indicators[name] = lbl

            state = info.get("state", "DISCONNECTED")
            color = _COLORS.get(state, "#9E9E9E")
            self._indicators[name].setText(
                f'<span style="color:{color}; font-size:18px;">●</span> {name}'
            )

    def closeEvent(self, event) -> None:
        if self._status_worker is not None:
            self._status_worker.stop_worker()
        super().closeEvent(event)
