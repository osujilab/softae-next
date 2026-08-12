"""Unit tests for the extracted shutdown bases.

Covers ``StoppableWorker`` (uniform ``stop_worker`` via ``_request_stop`` /
``_wake`` template methods) and ``DaemonRunnerMixin`` (``abort_run`` /
``cleanup`` via ``_abort_run_impl`` / ``_runner_thread`` hooks).  The existing
per-worker / per-tab shutdown tests remain the primary behavior-preserving
regression guard; these exercise the shared bases directly.
"""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QMutex, QWaitCondition

from softae.gui._shutdown import DAEMON_JOIN_TIMEOUT
from softae.gui.daemon_runner import DaemonRunnerMixin
from softae.gui.widgets.worker_thread import StoppableWorker

# ── StoppableWorker ─────────────────────────────────────────────────────────


class _InterruptionWorker(StoppableWorker):
    """Interruption-family worker: plain msleep loop, default stop sequence."""

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.msleep(20)


class _WakeWorker(StoppableWorker):
    """Interruption-family worker with a long timed wait broken by _wake()."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def _wake(self) -> None:
        self._condition.wakeOne()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self._mutex.lock()
            self._condition.wait(self._mutex, 60000)  # 60 s — only _wake() breaks it
            self._mutex.unlock()


class _FlagWorker(StoppableWorker):
    """Flag-family worker: run loops on a flag, _request_stop sets it."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._stop = False

    def _request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        # Note: no ``self._stop = False`` reset here — resetting inside run()
        # races the loop-wait below (isRunning() flips True before the body
        # executes), which would clobber a stop requested in that window.  The
        # base mechanism under test is the same either way.
        while not self._stop:
            self.msleep(20)


class _TimeoutProbeWorker(StoppableWorker):
    """Never really runs; records the timeout passed to wait()."""

    _default_stop_timeout_ms = 5000

    def __init__(self) -> None:
        super().__init__()
        self.waited_with: int | None = None

    def isRunning(self) -> bool:  # noqa: N802 - Qt override name
        return True

    def _request_stop(self) -> None:
        pass

    def wait(self, timeout_ms: int) -> bool:  # type: ignore[override]
        self.waited_with = timeout_ms
        return True


def test_stoppable_worker_base_stop_worker_joins(qapp):
    w = _InterruptionWorker()
    w.start()
    assert w.wait(1000) is False or True  # allow scheduler to spin up
    # ensure it is actually running before we stop it
    for _ in range(50):
        if w.isRunning():
            break
        time.sleep(0.01)
    assert w.isRunning()

    t0 = time.monotonic()
    w.stop_worker()
    elapsed = time.monotonic() - t0

    assert not w.isRunning()
    assert elapsed < 2.0  # returned well under the 2000 ms default timeout


def test_stoppable_worker_stop_worker_noop_when_idle(qapp):
    w = _InterruptionWorker()
    assert not w.isRunning()
    # never started — must return immediately and raise nothing
    w.stop_worker()
    assert not w.isRunning()


def test_stoppable_worker_wake_hook_called(qapp):
    w = _WakeWorker()
    w.start()
    for _ in range(50):
        if w.isRunning():
            break
        time.sleep(0.01)
    assert w.isRunning()

    t0 = time.monotonic()
    w.stop_worker()  # only _wake() (wakeOne) can break the 60 s condition wait
    elapsed = time.monotonic() - t0

    assert not w.isRunning()
    assert elapsed < 2.0  # proves _wake() fired (far under the 60 s sleep)


def test_stoppable_worker_flag_subclass_via_request_stop(qapp):
    w = _FlagWorker()
    w.start()
    for _ in range(50):
        if w.isRunning():
            break
        time.sleep(0.01)
    assert w.isRunning()

    w.stop_worker()
    assert not w.isRunning()
    assert w._stop is True


def test_stoppable_worker_per_class_default_timeout(qapp):
    w = _TimeoutProbeWorker()
    w.stop_worker()
    assert w.waited_with == 5000  # per-class default

    w.stop_worker(100)
    assert w.waited_with == 100  # explicit arg overrides


# ── DaemonRunnerMixin ───────────────────────────────────────────────────────


class _StubDaemonTab(DaemonRunnerMixin):
    def __init__(self) -> None:
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def _abort_run_impl(self) -> None:
        self._event.set()

    def _runner_thread(self):
        return self._thread


class _RaisingAbortTab(DaemonRunnerMixin):
    def _abort_run_impl(self) -> None:
        raise RuntimeError("boom")

    def _runner_thread(self):
        return None


def test_daemon_runner_mixin_cleanup_aborts():
    tab = _StubDaemonTab()

    def _spin() -> None:
        while not tab._event.is_set():
            time.sleep(0.005)

    tab._thread = threading.Thread(target=_spin, daemon=True)
    tab._thread.start()
    assert tab._thread.is_alive()

    tab.cleanup()

    assert tab._event.is_set()
    assert not tab._thread.is_alive()  # joined within DAEMON_JOIN_TIMEOUT


def test_daemon_runner_mixin_cleanup_noop_when_idle():
    tab = _StubDaemonTab()
    assert tab._runner_thread() is None
    t0 = time.monotonic()
    tab.cleanup()  # no thread to join — must return promptly, raise nothing
    assert time.monotonic() - t0 < DAEMON_JOIN_TIMEOUT


def test_daemon_runner_mixin_abort_run_swallows_exceptions():
    tab = _RaisingAbortTab()
    # blanket try/except in abort_run must swallow the raise
    tab.abort_run()
