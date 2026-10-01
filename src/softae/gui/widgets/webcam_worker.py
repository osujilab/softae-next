"""Dedicated QThread worker for USB webcam frame acquisition via OpenCV."""

from __future__ import annotations

import numpy as np
import structlog
from PySide6.QtCore import QDeadlineTimer, QMutex, QWaitCondition, Signal

from softae.gui.widgets.worker_thread import StoppableWorker

logger = structlog.get_logger(__name__)

# cv2 is optional — gracefully degrade if not installed
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

#: consecutive failed frames (exception or ret=False) before the capture is reopened
REOPEN_AFTER_FAILURES = 3
#: first reopen backoff; doubles per attempt up to BACKOFF_MAX_S
BACKOFF_BASE_S = 0.5
BACKOFF_MAX_S = 5.0


class WebcamWorker(StoppableWorker):
    """USB webcam acquisition thread using OpenCV cv2.VideoCapture.

    Signals
    -------
    frame_ready : object
        Emitted with numpy.ndarray (H, W, 3) uint8 RGB.
    error_occurred : str
        Emitted on open failure, and on failure-state transitions (first failed
        frame of an episode, each reopen that did not recover) — not per frame.
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
        self._applied_exposure: float | None = None
        self._reopen_attempts = 0
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
        """Open camera, acquire frames on demand, reopen on repeated failure."""
        self._abort = False
        self._reopen_attempts = 0
        if not _HAS_CV2:
            self.error_occurred.emit("OpenCV (cv2) not installed")
            return
        cap = self._open()
        if cap is None:
            self.error_occurred.emit(f"Failed to open webcam at index {self._camera_index}")
            return
        logger.info("webcam_worker_started", camera_idx=self._camera_index)

        failures = 0
        while cap is not None and (exposure := self._wait_for_request()) is not None:
            try:
                frame = self._grab(cap, exposure)
            except Exception as exc:  # cv2.error ("Unknown C++ exception") is an Exception
                error = exc
            else:
                if failures or self._reopen_attempts:
                    logger.info("webcam_recovered", failures=failures,
                                reopen_attempts=self._reopen_attempts)
                failures = self._reopen_attempts = 0
                self.frame_ready.emit(frame)
                continue

            failures += 1
            if failures == 1 and not self._reopen_attempts:
                logger.warning("webcam_frame_failed", exc_info=error,
                               camera_idx=self._camera_index)
                self.error_occurred.emit(f"Frame error: {error}")
            if failures >= REOPEN_AFTER_FAILURES:
                if self._reopen_attempts:  # the previous reopen opened but did not recover
                    self.error_occurred.emit(
                        f"Frame error (reopening, attempt {self._reopen_attempts + 1}): {error}")
                cap = self._reopen(cap)
                failures = 0

        self._release(cap)
        logger.info("webcam_worker_stopped")

    # ── Helpers (worker thread) ──────────────────────────────────────────

    def _wait_for_request(self) -> float | None:
        """Block until a frame is requested; the exposure to use, or None on stop."""
        self._mutex.lock()
        try:
            while not self._pending and not self._abort:
                self._condition.wait(self._mutex)
            if self._abort:
                return None
            self._pending = False
            return self._exposure
        finally:
            self._mutex.unlock()

    def _grab(self, cap, exposure: float) -> np.ndarray:
        # Each property set can renegotiate the DSHOW filter graph, so only on change.
        if exposure != self._applied_exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self._applied_exposure = exposure
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(frame_rgb, dtype=np.uint8)

    def _open(self):
        """A configured, opened capture, or None."""
        try:
            cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                self._release(cap)
                return None
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        except Exception:
            logger.warning("webcam_open_failed", exc_info=True, camera_idx=self._camera_index)
            return None
        self._applied_exposure = None  # a fresh device has not seen our exposure yet
        return cap

    def _reopen(self, cap):
        """Release, then retry opening with backoff; None if stopped meanwhile."""
        self._release(cap)
        while True:
            self._reopen_attempts += 1
            attempt = self._reopen_attempts
            delay = min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)
            logger.warning("webcam_reopen", attempt=attempt, delay_s=delay,
                           camera_idx=self._camera_index)
            if not self._sleep_unless_stopped(delay):
                return None
            cap = self._open()
            if cap is not None:
                return cap
            self.error_occurred.emit(
                f"Frame error (reopening, attempt {attempt}): "
                f"could not reopen webcam at index {self._camera_index}")

    def _sleep_unless_stopped(self, seconds: float) -> bool:
        """Wait out *seconds* unless stop is requested; True if still running."""
        deadline = QDeadlineTimer(int(seconds * 1000))
        self._mutex.lock()
        try:
            # request_frame also wakes this condition, so loop to the deadline
            while not self._abort and not deadline.hasExpired():
                self._condition.wait(self._mutex, deadline)
            return not self._abort
        finally:
            self._mutex.unlock()

    @staticmethod
    def _release(cap) -> None:
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            logger.warning("webcam_release_failed", exc_info=True)
