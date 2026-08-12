"""Shared base class for background polling workers.

``StoppableWorker`` codifies the uniform, idempotent ``stop_worker()`` contract
that every polling ``QThread`` in the GUI previously re-derived by hand.  The
stop sequence is expressed via three template points so each worker keeps its
own ``run()`` loop and wake mechanism while sharing the stop boilerplate:

- ``stop_worker(timeout_ms=None)`` — public entry point (unchanged name), guards
  on ``isRunning()``, resolves ``None`` to the per-class default timeout, then
  ``_request_stop()`` + ``wait(timeout_ms)``.
- ``_request_stop()`` — how to signal the run loop to exit.  Default =
  ``requestInterruption()`` + ``_wake()`` (the interruption family).  Flag-based
  workers override this to set their flag.
- ``_wake()`` — how to break a timed ``QWaitCondition`` sleep so stop is prompt.
  Default no-op (plain ``msleep`` loops need nothing).
"""

from __future__ import annotations

from PySide6.QtCore import QThread


class StoppableWorker(QThread):
    """Base for background workers with a uniform, idempotent stop_worker().

    The default stop is the interruption idiom used by the polling workers:
    ``requestInterruption()`` + an optional ``_wake()`` (to break a timed
    ``QWaitCondition`` sleep) + ``wait(timeout_ms)``.  Subclasses whose run
    loop checks a *flag* rather than ``isInterruptionRequested()`` override
    ``_request_stop()`` entirely to set that flag (and wake any condition).
    Subclasses that merely need to wake a wait-condition on the default
    interruption path override ``_wake()``.
    """

    #: default join timeout when stop_worker() is called with no explicit value
    _default_stop_timeout_ms: int = 2000

    def stop_worker(self, timeout_ms: int | None = None) -> None:
        """Request the worker to stop and join it (idempotent, no-op if idle)."""
        if not self.isRunning():
            return
        if timeout_ms is None:
            timeout_ms = self._default_stop_timeout_ms
        self._request_stop()
        self.wait(timeout_ms)

    # ── template methods ────────────────────────────────────────────────
    def _request_stop(self) -> None:
        """Signal the run loop to exit.  Default = interruption + wake."""
        self.requestInterruption()
        self._wake()

    def _wake(self) -> None:
        """Wake a timed wait so stop is prompt.  Default no-op (plain msleep)."""
        pass
