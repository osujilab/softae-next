"""Uniform ``stop_worker()`` contract on the shared/standalone polling workers.

Covers the workers that had no dedicated test module: ``InstrumentPoller``
(shared poller) and ``_SidebarPollWorker`` (standalone sidebar path).  Each must
expose an idempotent ``stop_worker()`` that requests interruption, wakes any
wait-condition, and joins promptly.
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.instrument_poller import InstrumentPoller
from softae.gui.widgets.monitor_sidebar import _SidebarPollWorker


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager():
    return create_mock_manager(config={})


class TestInstrumentPollerStopWorker:
    def test_instrument_poller_stop_worker_joins(self, qapp, manager):
        worker = InstrumentPoller(manager)
        worker.start()
        try:
            assert worker.isRunning()
            t0 = time.monotonic()
            worker.stop_worker()
            elapsed = time.monotonic() - t0
            assert not worker.isRunning()
            # poke() wakes the 2 s timed wait, so shutdown returns well under a cycle.
            assert elapsed < 1.5
        finally:
            if worker.isRunning():
                worker.requestInterruption()
                worker.poke()
                worker.wait(2000)

    def test_stop_worker_noop_when_not_running(self, qapp, manager):
        worker = InstrumentPoller(manager)
        worker.stop_worker()  # never started — returns immediately, no raise
        assert not worker.isRunning()


class TestSidebarWorkerStopWorker:
    def test_sidebar_worker_stop_worker_joins(self, qapp, manager):
        worker = _SidebarPollWorker(manager)
        worker.start()
        try:
            assert worker.isRunning()
            worker.stop_worker()
            assert not worker.isRunning()
        finally:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)
