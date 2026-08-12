"""Dedicated QThread worker for USB webcam frame acquisition via OpenCV."""

from __future__ import annotations

import numpy as np
import structlog
from PySide6.QtCore import QMutex, QWaitCondition, Signal

from softae.gui.widgets.worker_thread import StoppableWorker

logger = structlog.get_logger(__name__)

# cv2 is optional — gracefully degrade if not installed
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class WebcamWorker(StoppableWorker):
    """USB webcam acquisition thread using OpenCV cv2.VideoCapture.

    Signals
    -------
    frame_ready : object
        Emitted with numpy.ndarray (H, W, 3) uint8 RGB.
    error_occurred : str
        Emitted on camera open/read failure.
    """

    frame_ready = Signal(object)
    error_occurred = Signal(str)

    _default_stop_timeout_ms = 5000

    def __init__(
        self,
        camera_index: int = 0,
        target_width: int = 1280,
        target_height: int = 720,
        default_exposure: float = -7,
        parent=None,
    ):
        super().__init__(parent)
        self._camera_index = camera_index
        self._target_width = target_width
        self._target_height = target_height
        self._exposure: float = default_exposure
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._pending: bool = False
        self._abort: bool = False

    # ── Public API (called from GUI thread) ──────────────────────────────

    def request_frame(self, exposure: float | None = None) -> None:
        """Queue a single frame acquisition. Returns immediately."""
        self._mutex.lock()
        if exposure is not None:
            self._exposure = exposure
        self._pending = True
        self._condition.wakeOne()
        self._mutex.unlock()

    def set_exposure(self, exposure: float) -> None:
        """Update exposure value for next frame."""
        self._mutex.lock()
        self._exposure = exposure
        self._mutex.unlock()

    def _request_stop(self) -> None:
        """Signal the acquisition loop to abort and wake it (flag family)."""
        self._mutex.lock()
        self._abort = True
        self._condition.wakeOne()
        self._mutex.unlock()

    # ── Thread body ──────────────────────────────────────────────────────

    def run(self) -> None:
        """Open camera, acquire frames on demand, close on exit."""
        self._abort = False

        if not _HAS_CV2:
            self.error_occurred.emit("OpenCV (cv2) not installed")
            return

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.error_occurred.emit(
                f"Failed to open webcam at index {self._camera_index}"
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
        logger.info("webcam_worker_started", camera_idx=self._camera_index)

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
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                ret, frame = cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to read frame")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                self.frame_ready.emit(frame_rgb)
            except Exception as exc:
                self.error_occurred.emit(f"Frame error: {exc}")

        try:
            cap.release()
        except Exception:
            pass
        logger.info("webcam_worker_stopped")
