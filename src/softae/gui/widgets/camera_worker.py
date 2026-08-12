"""Dedicated QThread worker for non-blocking camera frame acquisition.

The ThorLabs TSI SDK has strict thread affinity — a camera opened on
one thread cannot be used from another (error 1004).  This worker keeps
the camera open on a **single persistent thread** and processes frame
requests communicated via :meth:`request_frame`.

Usage::

    worker = CameraWorker(manager)
    worker.frame_ready.connect(my_display_slot)
    worker.error_occurred.connect(my_error_slot)
    worker.start()                        # opens camera on worker thread
    worker.request_frame(exposure=0.045)  # non-blocking
    # ... later ...
    worker.stop_worker()                  # closes camera, joins thread
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PySide6.QtCore import QMutex, QWaitCondition, Signal

from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


class CameraWorker(StoppableWorker):
    """Persistent camera thread — opens once, acquires frames on demand.

    Signals
    -------
    frame_ready : object
        Emitted with a ``numpy.ndarray`` of shape ``(H, W, 3)`` after
        each successful acquisition.
    error_occurred : str
        Emitted when the camera cannot be opened or a frame fails.
    """

    frame_ready = Signal(object)
    error_occurred = Signal(str)

    _default_stop_timeout_ms = 5000

    def __init__(self, manager: "InstrumentManager", parent=None):
        super().__init__(parent)
        self._manager = manager
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._exposure: float = 0.045
        self._pending: bool = False
        self._abort: bool = False

    # ── Public API (called from the main / GUI thread) ───────────────────

    def request_frame(self, exposure: float = 0.045) -> None:
        """Queue a single frame acquisition at *exposure* seconds.

        This method returns immediately.  The result is delivered
        asynchronously via the :pyqt:`frame_ready` signal.

        If a previous request is still pending (worker busy acquiring),
        the exposure is updated to the latest value — requests are
        coalesced, never queued.
        """
        self._mutex.lock()
        self._exposure = exposure
        self._pending = True
        self._condition.wakeOne()
        self._mutex.unlock()

    def _request_stop(self) -> None:
        """Signal the acquisition loop to abort and wake it (flag family)."""
        self._mutex.lock()
        self._abort = True
        self._condition.wakeOne()
        self._mutex.unlock()

    # ── Thread body (runs entirely on the worker thread) ─────────────────

    def run(self) -> None:  # noqa: D401
        """Open the camera, loop acquiring frames, close on exit."""
        self._abort = False  # allow restart after a previous stop

        cam = self._manager.get("camera")
        try:
            cam.open()
        except Exception as exc:
            self.error_occurred.emit(f"Failed to open camera: {exc}")
            return

        logger.info("camera_worker_started")

        while True:
            self._mutex.lock()
            while not self._pending and not self._abort:
                self._condition.wait(self._mutex)

            if self._abort:
                self._mutex.unlock()
                break

            exposure = self._exposure
            self._pending = False
            self._mutex.unlock()

            try:
                arr = cam.acquire_n_frames(1, exposure)
                self.frame_ready.emit(arr)
            except Exception as exc:
                self.error_occurred.emit(str(exc))

        try:
            cam.close()
        except Exception:
            pass
        logger.info("camera_worker_stopped")
