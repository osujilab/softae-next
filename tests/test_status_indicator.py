"""Tests for Fix 4 — InstrumentStatusBar status poll off main thread.

Confirms:
  - _StatusWorker emits status_ready(dict) with the data returned by
    manager.status_all() (not called on main thread).
  - _apply_statuses slot populates / updates the indicator labels correctly.
  - _StatusWorker stops cleanly when requestInterruption() is called.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.status_indicator import InstrumentStatusBar, _StatusWorker


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager():
    return create_mock_manager(config={})


# ── Fix 4a: main thread not blocked — status_all runs in worker ───────────────


class TestStatusWorkerRunsOffMainThread:
    def test_status_all_not_called_synchronously_on_construction(self, qapp):
        """InstrumentStatusBar.__init__ must NOT call status_all() directly."""
        mock_mgr = MagicMock()
        mock_mgr.status_all.return_value = {}

        bar = InstrumentStatusBar(mock_mgr)
        # status_all should NOT have been called synchronously during __init__
        # (it was called by the old _timer code; the new code schedules a worker)
        mock_mgr.status_all.assert_not_called()
        # Stop the background worker before closing (prevent Qt teardown crash)
        if hasattr(bar, "_status_worker") and bar._status_worker.isRunning():
            bar._status_worker.requestInterruption()
            bar._status_worker.wait(2000)
        bar.close()

    def test_worker_calls_status_all_in_background(self, qapp):
        """_StatusWorker must call manager.status_all() and emit status_ready."""
        received = []
        fake_statuses = {"stage": {"state": "CONNECTED"}}
        mock_mgr = MagicMock()
        mock_mgr.status_all.return_value = fake_statuses

        worker = _StatusWorker(mock_mgr)
        worker.status_ready.connect(lambda d: received.append(d))
        worker.start()

        timeout = 4.0
        t0 = time.monotonic()
        while not received and (time.monotonic() - t0) < timeout:
            time.sleep(0.1)
            QApplication.processEvents()

        worker.requestInterruption()
        worker.wait(2000)
        assert received, "_StatusWorker did not emit status_ready within 4 s"
        assert "stage" in received[0]


# ── Fix 4b: _apply_statuses slot updates labels correctly ────────────────────


class TestApplyStatusesUpdatesLabels:
    def test_apply_statuses_creates_indicator_for_new_instrument(self, qapp, manager):
        bar = InstrumentStatusBar(manager)
        # Stop the background worker so it doesn't interfere
        bar._status_worker.requestInterruption()
        bar._status_worker.wait(2000)

        test_statuses = {"test_inst": {"state": "CONNECTED"}}
        bar._apply_statuses(test_statuses)
        QApplication.processEvents()

        assert "test_inst" in bar._indicators
        label_text = bar._indicators["test_inst"].text()
        assert "test_inst" in label_text
        assert "#4CAF50" in label_text  # green for CONNECTED
        bar.close()

    def test_apply_statuses_updates_existing_indicator_color(self, qapp, manager):
        bar = InstrumentStatusBar(manager)
        bar._status_worker.requestInterruption()
        bar._status_worker.wait(2000)

        bar._apply_statuses({"dev": {"state": "CONNECTED"}})
        QApplication.processEvents()
        assert "#4CAF50" in bar._indicators["dev"].text()

        bar._apply_statuses({"dev": {"state": "ERROR"}})
        QApplication.processEvents()
        assert "#f44336" in bar._indicators["dev"].text()  # red for ERROR
        bar.close()

    def test_apply_statuses_handles_unknown_state_gracefully(self, qapp, manager):
        bar = InstrumentStatusBar(manager)
        bar._status_worker.requestInterruption()
        bar._status_worker.wait(2000)

        bar._apply_statuses({"dev": {"state": "UNKNOWN_XYZ"}})
        QApplication.processEvents()
        assert "dev" in bar._indicators
        bar.close()


# ── Fix 4c: worker stops cleanly ─────────────────────────────────────────────


class TestStatusWorkerStopsCleanly:
    def test_worker_stops_on_interruption(self, qapp):
        mock_mgr = MagicMock()
        mock_mgr.status_all.return_value = {}

        worker = _StatusWorker(mock_mgr)
        worker.start()
        time.sleep(0.1)
        worker.requestInterruption()
        stopped = worker.wait(3000)
        assert stopped, "_StatusWorker did not stop within 3 s after requestInterruption()"
        assert not worker.isRunning()

    def test_worker_handles_none_manager_safely(self, qapp):
        """_StatusWorker must not crash if manager is None."""
        worker = _StatusWorker(None)
        worker.start()
        time.sleep(0.2)
        worker.requestInterruption()
        assert worker.wait(2000)

    def test_worker_handles_status_all_exception_safely(self, qapp):
        """_StatusWorker must not crash if status_all() raises."""
        mock_mgr = MagicMock()
        mock_mgr.status_all.side_effect = RuntimeError("comms error")

        worker = _StatusWorker(mock_mgr)
        worker.start()
        time.sleep(0.1)
        worker.requestInterruption()
        assert worker.wait(3000), "Worker should stop cleanly even when status_all() raises"


class TestStatusWorkerStopWorker:
    def test_status_worker_stop_worker_joins(self, qapp, manager):
        worker = _StatusWorker(manager)
        worker.start()
        try:
            assert worker.isRunning()
            worker.stop_worker()
            assert not worker.isRunning()
        finally:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)

    def test_stop_worker_noop_when_not_running(self, qapp, manager):
        worker = _StatusWorker(manager)
        worker.stop_worker()  # never started — returns immediately, no raise
        assert not worker.isRunning()
