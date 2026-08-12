"""Tests for Fix 3 — InitCalibrationTab stage calibration handlers off main thread.

Confirms:
  - _on_go_home and _on_go_dep1 disable their buttons immediately on the main thread
    (hardware call has NOT been made yet at that point).
  - The asyncio coroutine delivers the correct position text to _lbl_stage_pos.
  - Buttons are re-enabled even when stage raises an exception.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_init import InitCalibrationTab


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_coro(coro):
    """Run a coroutine synchronously in a fresh event loop (test utility)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


@pytest.fixture
def tab(qapp, manager):
    widget = InitCalibrationTab(manager)
    yield widget
    # Explicitly stop background workers before widget destruction to avoid
    # "QThread destroyed while running" native crash on teardown.
    pos_map = getattr(widget, "_pos_map", None)
    if pos_map is not None:
        worker = getattr(pos_map, "_pos_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(2000)
    widget.close()


# ── Fix 3a: main thread not blocked ──────────────────────────────────────────


class TestInitCalibButtonsDisabledImmediately:
    def test_go_home_disables_button_before_hardware_call(self, tab):
        """_btn_go_home must be disabled synchronously inside _on_go_home."""
        # Patch ensure_future to prevent the coroutine from running on the
        # test event loop (avoids interacting with pytest-asyncio bookkeeping).
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_go_home()
        assert not tab._btn_go_home.isEnabled(), \
            "_btn_go_home must be disabled synchronously"
        # Close the captured coroutine to silence "RuntimeWarning: never awaited"
        for c in captured:
            c.close()
        # Re-enable for subsequent tests
        tab._btn_go_home.setEnabled(True)

    def test_go_dep1_disables_button_before_hardware_call(self, tab):
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_go_dep1()
        assert not tab._btn_go_dep1.isEnabled()
        for c in captured:
            c.close()
        tab._btn_go_dep1.setEnabled(True)

    def test_set_home_disables_button_before_hardware_call(self, tab):
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_set_home()
        assert not tab._btn_set_home.isEnabled()
        for c in captured:
            c.close()
        tab._btn_set_home.setEnabled(True)

    def test_set_dep1_disables_button_before_hardware_call(self, tab):
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_set_dep1()
        assert not tab._btn_set_dep1.isEnabled()
        for c in captured:
            c.close()
        tab._btn_set_dep1.setEnabled(True)


# ── Fix 3b: slot receives correct data ───────────────────────────────────────


class TestInitCalibPositionLabelUpdated:
    def test_go_home_updates_label_on_success(self, tab):
        """After coro completes, _lbl_stage_pos shows 'At home'."""
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_go_home()
        assert captured
        _run_coro(captured[0])
        QApplication.processEvents()
        text = tab._lbl_stage_pos.text()
        assert "home" in text.lower() or text.startswith("At"), \
            f"Unexpected label: {text}"
        tab._btn_go_home.setEnabled(True)

    def test_go_dep1_updates_label_on_success(self, tab):
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_go_dep1()
        assert captured
        _run_coro(captured[0])
        QApplication.processEvents()
        text = tab._lbl_stage_pos.text()
        assert "dep" in text.lower() or "At" in text, f"Unexpected label: {text}"
        tab._btn_go_dep1.setEnabled(True)


# ── Fix 3c: button re-enabled on error ───────────────────────────────────────


class TestInitCalibButtonRenabledOnError:
    def test_go_home_reenables_button_on_stage_error(self, tab):
        stage = tab._manager.get("stage")
        original = stage.move_to

        def bad_move(x, y):
            raise OSError("motor fault")

        stage.move_to = bad_move
        captured = []
        with patch("asyncio.ensure_future", lambda c: captured.append(c) or MagicMock()):
            tab._on_go_home()
        assert captured
        _run_coro(captured[0])
        QApplication.processEvents()
        assert tab._btn_go_home.isEnabled(), "Button must be re-enabled after error"
        assert "Error" in tab._lbl_stage_pos.text()
        stage.move_to = original


class TestSyringeConfigPanel:
    def test_spinboxes_default_use_loader_helper(self, qapp, manager):
        with patch("softae.config.loader.syringe_parallel_counts", return_value={0: 2, 1: 1, 2: 1}):
            widget = InitCalibrationTab(manager)
        try:
            assert widget._spin_syr_parallel_by_pump[0].value() == 2
        finally:
            pos_map = getattr(widget, "_pos_map", None)
            if pos_map is not None:
                worker = getattr(pos_map, "_pos_worker", None)
                if worker is not None and worker.isRunning():
                    worker.requestInterruption()
                    worker.wait(2000)
            widget.close()

    def test_syringe_panel_widgets_exist(self, tab):
        assert hasattr(tab, "_spin_syr_parallel_by_pump")
        assert hasattr(tab, "_btn_apply_syr")
        assert len(tab._spin_syr_parallel_by_pump) == 3

    def test_apply_save_calls_loader_and_live_driver(self, tab):
        syr = tab._manager.get("syringe")
        original = syr.set_parallel_syringes
        syr.set_parallel_syringes = MagicMock()
        try:
            tab._spin_syr_parallel_by_pump[0].setValue(2)
            tab._spin_syr_parallel_by_pump[1].setValue(1)
            tab._spin_syr_parallel_by_pump[2].setValue(2)
            with patch("softae.config.loader.save_syringe_parallel_counts") as save_mock:
                tab._on_apply_syringe_parallel()
            save_mock.assert_called_once_with({0: 2, 1: 1, 2: 2})
            assert syr.set_parallel_syringes.call_count == 3
            assert "Saved per-pump syringe counts" in tab._lbl_syr_status.text()
        finally:
            syr.set_parallel_syringes = original


class TestInitTabNoLiquidPanel:
    def test_liquid_model_widgets_not_present(self, tab):
        assert not hasattr(tab, "_chk_liq_enabled")
        assert not hasattr(tab, "_line_liq_widgets")
        assert not hasattr(tab, "_btn_apply_liq")


class TestInitRefreshCompatibility:
    def test_refresh_table_accepts_no_args(self, tab):
        tab._refresh_table()
        assert tab._table.rowCount() >= 0


class TestInitTabCleanup:
    def test_init_tab_cleanup_stops_poll_worker(self, tab):
        assert tab._poll_worker.isRunning()
        tab.cleanup()
        assert not tab._poll_worker.isRunning()

    def test_init_tab_cleanup_idempotent_when_not_running(self, tab):
        tab.cleanup()
        tab.cleanup()  # second call must not raise
        assert not tab._poll_worker.isRunning()


class TestTablePollWorkerStopWorker:
    def test_table_poll_worker_stop_worker_joins(self, qapp, manager):
        from softae.gui.tabs.tab_init import _TablePollWorker

        worker = _TablePollWorker(manager)
        worker.start()
        try:
            assert worker.isRunning()
            worker.stop_worker()
            assert not worker.isRunning()
        finally:
            if worker.isRunning():
                worker.requestInterruption()
                worker.poke()
                worker.wait(2000)

    def test_stop_worker_noop_when_not_running(self, qapp, manager):
        from softae.gui.tabs.tab_init import _TablePollWorker

        worker = _TablePollWorker(manager)
        # Never started — must return immediately without raising.
        worker.stop_worker()
        assert not worker.isRunning()
