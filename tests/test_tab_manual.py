"""Tests for Fix 1 — ManualControlTab button handlers off main thread.

Confirms:
  - Each blocking handler disables its button(s) immediately on the main thread.
  - The _CommandWorker delivers results via signal to the correct slot.
  - _CommandWorker stops cleanly after run() completes (one-shot semantics).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_manual import ManualControlTab, _ManualPollingWorker
from softae.gui.tabs.tab_manual_workers import _CommandWorker


# ── Fixtures ─────────────────────────────────────────────────────────────────


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
    widget = ManualControlTab(manager)
    yield widget
    # belt-and-suspenders: stop embedded workers that might still be running
    pos_map = getattr(widget, "_pos_map", None)
    if pos_map is not None:
        worker = getattr(pos_map, "_pos_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(2000)
    pv_worker = getattr(widget, "_pv_worker", None)
    if pv_worker is not None and pv_worker.isRunning():
        pv_worker.stop_worker()
    widget.close()


# ── Fix 1a: main thread not blocked ──────────────────────────────────────────


class TestButtonsDisabledDuringRun:
    def test_goto_disables_button_immediately(self, tab):
        """_on_goto must disable _btn_goto before hardware call returns."""
        stage = tab._manager.get("stage")
        original_move = stage.move_to

        def slow_move(x, y):
            time.sleep(0.05)
            original_move(x, y)

        stage.move_to = slow_move
        tab._on_goto()
        # Button must be disabled synchronously — before the worker thread finishes
        assert not tab._btn_goto.isEnabled(), "_btn_goto should be disabled while worker runs"
        # Wait for the worker to finish and button to re-enable
        QApplication.processEvents()
        timeout = 3.0
        t0 = time.monotonic()
        while tab._btn_goto.isEnabled() is False and (time.monotonic() - t0) < timeout:
            time.sleep(0.05)
            QApplication.processEvents()
        stage.move_to = original_move  # restore

    def test_ramp_disables_button_immediately(self, tab):
        """_on_ramp must disable _btn_ramp before ramp_linear call returns."""
        tc = tab._manager.get("temp_controller")
        original_ramp = getattr(tc, "ramp_linear", None)

        def slow_ramp(**kwargs):
            time.sleep(0.05)

        tc.ramp_linear = slow_ramp
        tab._on_ramp()
        assert not tab._btn_ramp.isEnabled(), "_btn_ramp should be disabled while ramp runs"
        timeout = 3.0
        t0 = time.monotonic()
        while tab._btn_ramp.isEnabled() is False and (time.monotonic() - t0) < timeout:
            time.sleep(0.05)
            QApplication.processEvents()
        if original_ramp is not None:
            tc.ramp_linear = original_ramp

    def test_head_retract_disables_button_immediately(self, tab):
        syr = tab._manager.get("syringe")
        original = syr.head_retract

        def slow_retract():
            time.sleep(0.05)

        syr.head_retract = slow_retract
        tab._on_head_retract()
        assert not tab._btn_head_retract.isEnabled()
        syr.head_retract = original

    def test_infuse_disables_button_immediately(self, tab):
        syr = tab._manager.get("syringe")
        original = syr.single_pump

        def slow_pump(**kwargs):
            time.sleep(0.05)

        syr.single_pump = slow_pump
        tab._on_infuse(0)
        assert not tab._pump_widgets[0]["btn"].isEnabled()
        syr.single_pump = original

    def test_jog_disables_all_jog_buttons(self, tab):
        stage = tab._manager.get("stage")
        original = stage.move_by

        def slow_move(dx, dy):
            time.sleep(0.05)
            original(dx, dy)

        stage.move_by = slow_move
        tab._on_jog(1, 0)
        assert all(not b.isEnabled() for b in tab._jog_buttons), \
            "All jog buttons should be disabled while jogging"
        stage.move_by = original


# ── Fix 1b: slot receives correct data ───────────────────────────────────────


class TestCommandWorkerSignals:
    def test_completed_signal_delivers_return_value(self, qapp):
        """_CommandWorker.completed carries the fn() return value."""
        received = []
        w = _CommandWorker(lambda: 42)
        w.completed.connect(lambda v: received.append(v))
        w.start()
        assert w.wait(2000), "Worker did not finish in time"
        QApplication.processEvents()
        assert received == [42]

    def test_failed_signal_delivers_exception_message(self, qapp):
        """_CommandWorker.failed carries str(exc) when fn() raises."""
        errors = []

        def bad_fn():
            raise RuntimeError("boom")

        w = _CommandWorker(bad_fn)
        w.failed.connect(lambda e: errors.append(e))
        w.start()
        assert w.wait(2000), "Worker did not finish in time"
        QApplication.processEvents()
        assert len(errors) == 1
        assert "boom" in errors[0]

    def test_goto_updates_position_label_via_signal(self, tab):
        """After _on_goto worker completes, _lbl_pos shows the new position."""
        stage = tab._manager.get("stage")
        # Move to known position then call _on_goto to a different coord
        tab._spin_x.setValue(5.0)
        tab._spin_y.setValue(3.0)
        tab._on_goto()
        # Wait for worker
        timeout = 3.0
        t0 = time.monotonic()
        while not tab._btn_goto.isEnabled() and (time.monotonic() - t0) < timeout:
            time.sleep(0.05)
            QApplication.processEvents()
        assert tab._btn_goto.isEnabled(), "Button should be re-enabled after worker finishes"
        assert "Position:" in tab._lbl_pos.text()


# ── Fix 1c: worker stops cleanly (one-shot) ──────────────────────────────────


class TestCommandWorkerLifecycle:
    def test_worker_is_not_running_after_completion(self, qapp):
        """_CommandWorker naturally finishes — not an infinite loop."""
        w = _CommandWorker(lambda: None)
        w.start()
        finished = w.wait(2000)
        assert finished, "Worker thread should exit cleanly after run()"
        assert not w.isRunning()

    def test_worker_btn_reenabled_on_error(self, tab):
        """_btn_goto is re-enabled even when the stage raises an error."""
        stage = tab._manager.get("stage")
        original = stage.move_to

        def raising_move(x, y):
            raise OSError("stage offline")

        stage.move_to = raising_move
        # Suppress the QMessageBox.warning dialog that fires on the failed signal
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            tab._on_goto()
            timeout = 3.0
            t0 = time.monotonic()
            while not tab._btn_goto.isEnabled() and (time.monotonic() - t0) < timeout:
                time.sleep(0.05)
                QApplication.processEvents()
        assert tab._btn_goto.isEnabled(), "Button should be re-enabled even after error"
        stage.move_to = original


class TestManualCorrectionAndSyringeReadout:
    def _wait_button_enabled(self, button, timeout_s: float = 3.0) -> None:
        t0 = time.monotonic()
        while not button.isEnabled() and (time.monotonic() - t0) < timeout_s:
            time.sleep(0.05)
            QApplication.processEvents()

    def test_infuse_toggle_off_sends_raw_volume(self, tab):
        syr = tab._manager.get("syringe")
        calls = []

        def record_call(**kwargs):
            calls.append(kwargs)

        original = syr.single_pump
        syr.single_pump = record_call
        try:
            tab._chk_apply_correction.setChecked(False)
            tab._pump_widgets[0]["vol"].setValue(12.0)
            tab._on_infuse(0)
            self._wait_button_enabled(tab._pump_widgets[0]["btn"])
            assert calls
            assert calls[0]["dispense_vol"] == pytest.approx(12.0)
            assert "correction off" in tab._lbl_last_command.text()
        finally:
            syr.single_pump = original

    def test_infuse_toggle_on_sends_corrected_volume(self, tab):
        syr = tab._manager.get("syringe")
        calls = []

        def record_call(**kwargs):
            calls.append(kwargs)

        original = syr.single_pump
        syr.single_pump = record_call
        try:
            tab._chk_apply_correction.setChecked(True)
            tab._pump_widgets[0]["vol"].setValue(10.0)
            with patch("softae.gui.tabs.tab_manual.liquid_handling_config", return_value={
                "enabled": True,
                "valves_in_series": 2,
                "beta": 0.30,
                "eta_ref_mpas": 1.0,
                "alpha_growth_per_run": 0.0,
                "pump_line": {"0": 0},
                "line": {
                    "0": {
                        "cracking_kpa_per_valve": 8.0,
                        "compliance_uL_per_kpa": 0.55,
                        "alpha_base": 0.2,
                        "viscosity_mpas": 1.0,
                    }
                },
            }):
                tab._on_infuse(0)
            self._wait_button_enabled(tab._pump_widgets[0]["btn"])
            assert calls
            assert calls[0]["dispense_vol"] > 10.0
            assert "correction on" in tab._lbl_last_command.text()
            assert "target 10.00 uL" in tab._lbl_last_command.text()
        finally:
            syr.single_pump = original

    def test_syringe_count_readout_updates_from_poll_data(self, tab):
        tab._on_pv_poll_done({"parallel_syringes_by_pump": {0: 1, 1: 2, 2: 3}})
        assert tab._pump_widgets[0]["count_lbl"].text().endswith("1")
        assert tab._pump_widgets[1]["count_lbl"].text().endswith("2")
        assert tab._pump_widgets[2]["count_lbl"].text().endswith("3")

    def test_poll_worker_propagates_parallel_map_from_status(self, manager):
        worker = _ManualPollingWorker(manager)
        emitted: list[dict] = []
        worker.poll_done.connect(lambda payload: emitted.append(payload))

        syr = manager.get("syringe")
        original_status = syr.status
        original_msleep = worker.msleep

        syr.status = MagicMock(return_value={
            "parallel_syringes": 1,
            "parallel_syringes_by_pump": {0: 2, 1: 1, 2: 2},
        })
        worker.msleep = lambda _ms: setattr(worker, "_stop", True)

        try:
            worker.run()
        finally:
            syr.status = original_status
            worker.msleep = original_msleep

        assert emitted
        payload = emitted[-1]
        assert payload["parallel_syringes"] == 1
        assert payload["parallel_syringes_by_pump"] == {0: 2, 1: 1, 2: 2}


class TestManualTabCleanup:
    def test_manual_tab_cleanup_stops_pv_worker(self, tab):
        assert tab._pv_worker.isRunning()
        tab.cleanup()
        assert not tab._pv_worker.isRunning()

    def test_manual_tab_cleanup_safe_with_no_eis_thread(self, tab):
        assert tab._eis_thread is None
        tab.cleanup()  # must not raise when _eis_thread is None
        tab.cleanup()  # idempotent
        assert not tab._pv_worker.isRunning()


def _eis_result(channel):
    import numpy as np

    from softae.analysis.eis_data import EISResult

    freq = np.geomspace(1e4, 1.0, 15)
    z = 5e4 + 1e6 / (1 + 1j * 2 * np.pi * freq * 1e-4)
    return EISResult.from_arrays(channel=channel, f=freq, z_real=z.real, z_imag_neg=-z.imag)


class TestManualEisMultiChannel:
    """Channel(s) spec field + multi-channel finished handling."""

    def test_pico_label_valid_for_channel_spec(self, tab):
        tab._edit_eis_ch.setText("2,4,5-10")
        tab._update_eis_pico_label()
        assert tab._lbl_eis_pico.text() != "—"      # a real pico (or "mixed")

    def test_pico_label_dash_for_invalid_spec(self, tab):
        tab._edit_eis_ch.setText("not-a-channel")
        tab._update_eis_pico_label()
        assert tab._lbl_eis_pico.text() == "—"

    def test_multi_channel_finished_shows_series_window_and_status(self, tab):
        payloads = [
            {"channel": ch, "eis_result": _eis_result(ch), "fit_result": None,
             "auto_fit": False, "fit_model": "simpleSalt", "pico_name": "pico1"}
            for ch in (2, 4, 6)
        ]
        tab._on_eis_finished(payloads)
        assert tab._eis_series_window is not None
        assert tab._lbl_eis_status.text() == (
            "Multiple channels run: see Analysis tab for measurement details.")
        tab._eis_series_window.close()

    def test_empty_finished_reports_nothing_measured(self, tab):
        tab._on_eis_finished([])
        assert tab._lbl_eis_status.text() == "No channels measured."
