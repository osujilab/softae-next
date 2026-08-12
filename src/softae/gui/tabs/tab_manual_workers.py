"""One-shot command worker for ManualControlTab button handlers.

Kept in a separate module because tab_manual.py already exceeds 400 lines
(contains _ManualPollingWorker and _ManualEisWorker).
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class _CommandWorker(QThread):
    """Run a single callable on a background thread.

    Emits ``completed(object)`` with the callable's return value on
    success, or ``failed(str)`` with the exception message on error.
    Both signals are delivered back to the main thread via the normal Qt
    queued-connection mechanism.
    """

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
