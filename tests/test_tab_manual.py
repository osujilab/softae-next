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
import structlog
from PySide6.QtWidgets import QApplication, QWidget

from softae.core.rig_activity import RigActivity
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_manual import (
    MANUAL_INSTRUMENTS,
    ManualControlTab,
    _ManualPollingWorker,
)
from softae.gui.tabs.tab_manual_workers import _CommandWorker
from softae.gui.widgets import occupancy_guard


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


def _shutdown_tab(widget) -> None:
    """Belt-and-suspenders: stop embedded workers that might still be running."""
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


@pytest.fixture
def tab(qapp, manager):
    widget = ManualControlTab(manager)
    yield widget
    _shutdown_tab(widget)


@pytest.fixture
def rig_activity():
    """The claim registry a window owns, standing in for ``MainWindow``'s."""
    return RigActivity()


@pytest.fixture
def hosted_tab(qapp, manager, rig_activity):
    """A Manual tab *inside a window that runs things* — the case with a claim.

    The `tab` fixture above is the windowless one, which most of the suite uses
    and which must stay fully usable; this one gives ``self.window()`` something
    carrying ``_rig_activity``, the same attribute ``MainWindow`` holds it on and
    the same one `tests/test_rig_claim.py`'s host double publishes.
    """
    host = QWidget()
    host._rig_activity = rig_activity
    widget = ManualControlTab(manager, parent=host)
    yield widget
    widget.cleanup()
    host.close()


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
        """One poll, asserted directly.

        This used to run the worker's whole loop on the test thread and escape it
        by patching ``msleep`` to raise the stop flag — so the assertion about
        what a poll *reports* was pinned to how the loop *waits*, and changing the
        wait to something interruptible turned the test into an infinite loop.
        ``poll_once`` is the reading; the loop is not part of this question.
        """
        worker = _ManualPollingWorker(manager)

        syr = manager.get("syringe")
        original_status = syr.status
        syr.status = MagicMock(return_value={
            "parallel_syringes": 1,
            "parallel_syringes_by_pump": {0: 2, 1: 1, 2: 2},
        })
        try:
            payload = worker.poll_once()
        finally:
            syr.status = original_status

        assert payload["parallel_syringes"] == 1
        assert payload["parallel_syringes_by_pump"] == {0: 2, 1: 1, 2: 2}


class TestManualTabCleanup:
    def test_manual_tab_cleanup_stops_pv_worker(self, tab):
        assert tab._pv_worker.isRunning()
        tab.cleanup()
        assert not tab._pv_worker.isRunning()

    def test_manual_tab_cleanup_wakes_the_poll_wait_instead_of_waiting_it_out(
            self, tab):
        """``cleanup()`` runs on the main thread, so its cost is a GUI freeze.

        The poll used to wait with ``msleep``, which cannot be interrupted, so a
        stop arriving just after a poll joined for the remaining ~2 s — every tab
        close, and every window close.

        This asserted a 0.7 s wall bound with the message "the poll sleep was not
        woken", and the assertion could not tell those two things apart: on a
        machine saturated by the suite, exceeding a wall bound is a scheduling
        fact, not a wait-condition fact, so the test failed for reasons that had
        nothing to do with the guarantee it names. The mechanism is now recorded
        by the worker (:attr:`_ManualPollingWorker.interval_end`, written under
        the same mutex the stop is set under), so promptness can be asserted
        directly and is independent of load and of test order. Restoring an
        uninterruptible sleep, or dropping the ``wakeAll`` from ``_request_stop``,
        leaves ``"elapsed"`` here and fails.
        """
        worker = tab._pv_worker
        assert worker.isRunning()

        # Wait for the loop to reach its first wait, so what is asserted below is
        # the outcome of an interval that really was in flight. Without this, a
        # thread that had not been scheduled yet would leave `interval_end` None
        # and the assertion would pass vacuously.
        deadline = time.monotonic() + 5.0
        while worker.interval_end is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.interval_end is not None, "the poll loop never reached its wait"

        tab.cleanup()

        assert not worker.isRunning(), "cleanup() did not join the poll worker"
        assert worker.interval_end in worker.PROMPT_STOPS, (
            f"the poll wait ended as {worker.interval_end!r}: the stop did not "
            f"wake it, so cleanup() sat out the rest of the poll interval")

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


# ── A run in this window holds the rig ───────────────────────────────────────
#
# The standing ruling is that manual control is never refused, and it stands for
# the owner it was made about: a campaign in *another process*, which the
# operator at the bench cannot pause and whose refusal would therefore leave them
# with nothing but the E-Stop.
#
# A run started from this very window is the other case. It has a Pause beside
# it, pausing suspends its rig claim, and a suspended owner is skipped by
# `RigActivity.conflicts` — so the refusal is addressed to someone who can lift
# it in one click. Both directions are pinned below, because a feature like this
# can ship inverted and look green with only one of them tested.

#: One entry per actuating family reachable on a mock rig. Piezo is absent
#: deliberately: `[piezo] enabled` ships false, so `_on_piezo_a_on` returns
#: before the ownership note and the parametrised case would prove nothing.
_ACTUATING = [
    ("stage jog", "stage", "move_by", lambda t: t._on_jog(1, 0)),
    ("stage go-to", "stage", "move_to", lambda t: t._on_goto()),
    ("temperature setpoint", "temp_controller", "write_sp", lambda t: t._on_set_temp()),
    ("temperature ramp", "temp_controller", "ramp_linear", lambda t: t._on_ramp()),
    ("humidity setpoint", "rh_controller", "set_setpoint", lambda t: t._on_set_rh()),
    ("dispenser head descend", "syringe", "head_descend", lambda t: t._on_head_descend()),
    ("dispenser head retract", "syringe", "head_retract", lambda t: t._on_head_retract()),
    ("pump 0 dispense", "syringe", "single_pump", lambda t: t._on_infuse(0)),
    ("lamp on", "lamp", "on", lambda t: t._on_lamp_on()),
]


class TestRefusedWhileARunInThisWindowDrivesTheRig:
    def test_a_live_claim_refuses_the_jog_and_names_the_run(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        rig_activity.acquire("ht:cast_series", {"stage"})
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get("stage"), "move_by",
                            lambda dx, dy: calls.append((dx, dy)))

        hosted_tab._on_jog(1, 0)
        settle_qt(hosted_tab)

        assert calls == []
        text = hosted_tab._lbl_last_command.text()
        assert "Refused" in text
        assert "stage jog" in text
        assert "ht:cast_series" in text, "a refusal that does not name the run"
        assert "Pause" in text, "a refusal that does not name the way out"
        # Refused before the slot disables anything, so the control is still
        # there to press once the run is paused.
        assert all(b.isEnabled() for b in hosted_tab._jog_buttons)

    def test_the_same_claim_suspended_permits_the_jog(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        """The pause ruling, and the direction that would ship silently inverted.

        Same claim, same instruments, same control — only suspended, which is
        what the executor does when it holds a run at a pause. Nothing in the tab
        tests for suspension; `conflicts` already skips a suspended owner, and
        this is what proves the tab inherited that answer rather than deriving a
        second one.
        """
        rig_activity.acquire("ht:cast_series", {"stage"})
        rig_activity.suspend("ht:cast_series", reason="paused by operator")
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get("stage"), "move_by",
                            lambda dx, dy: calls.append((dx, dy)))

        hosted_tab._spin_jog_step.setValue(1.0)
        hosted_tab._on_jog(1, 0)
        settle_qt(hosted_tab)

        assert calls == [(1.0, 0.0)]
        assert "Refused" not in hosted_tab._lbl_last_command.text()

    def test_unsuspending_refuses_again(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        """A resumed run takes the controls back — the pause is not one-way."""
        rig_activity.acquire("ht:cast_series", {"stage"})
        rig_activity.suspend("ht:cast_series")
        rig_activity.unsuspend("ht:cast_series")
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get("stage"), "move_by",
                            lambda dx, dy: calls.append((dx, dy)))

        hosted_tab._on_jog(1, 0)
        settle_qt(hosted_tab)

        assert calls == []
        assert "Refused" in hosted_tab._lbl_last_command.text()

    def test_a_claim_on_other_instruments_does_not_refuse(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        """Scoped, not blanket: an anneal holds the heater, not the stage."""
        rig_activity.acquire("ht:anneal", {"temp_controller"})
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get("stage"), "move_by",
                            lambda dx, dy: calls.append((dx, dy)))

        hosted_tab._spin_jog_step.setValue(1.0)
        hosted_tab._on_jog(1, 0)
        settle_qt(hosted_tab)

        assert calls == [(1.0, 0.0)]

    def test_a_liquid_handler_claim_refuses_the_stage_and_the_pump(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        """`liquid_handler` is one step driving stage *and* syringe internally.

        `deposition_steps` casts with `instrument="liquid_handler"`, so a claim
        derived from a cast workflow names only that. A manual scope of bare
        `{"stage"}` would miss it and permit a jog into a live cast — the exact
        under-refusal that widening on doubt exists to prevent.
        """
        rig_activity.acquire("ht:dropcast", {"liquid_handler"})
        moves, pumps = [], []
        monkeypatch.setattr(hosted_tab._manager.get("stage"), "move_by",
                            lambda dx, dy: moves.append((dx, dy)))
        monkeypatch.setattr(hosted_tab._manager.get("syringe"), "single_pump",
                            lambda **kw: pumps.append(kw))

        hosted_tab._on_jog(1, 0)
        hosted_tab._chk_apply_correction.setChecked(False)
        hosted_tab._on_infuse(0)
        settle_qt(hosted_tab)

        assert moves == []
        assert pumps == [], "fluid was commanded into a live cast"

    @pytest.mark.parametrize(
        "action,instrument,method,press", _ACTUATING, ids=[a[0] for a in _ACTUATING])
    def test_a_whole_rig_claim_refuses_every_actuating_family(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt,
            action, instrument, method, press):
        """`instruments=None` is what a campaign takes, and it conflicts with all."""
        rig_activity.acquire("campaign:phase_map:run-7")
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get(instrument), method,
                            lambda *a, **k: calls.append((a, k)))

        press(hosted_tab)
        settle_qt(hosted_tab)

        assert calls == []
        text = hosted_tab._lbl_last_command.text()
        assert action in text and "Refused" in text
        assert "campaign:phase_map:run-7" in text

    def test_manual_eis_is_refused_by_a_claim_on_the_pico_it_routes_to(
            self, hosted_tab, rig_activity, monkeypatch):
        """The acquisition worker is booby-trapped rather than merely asserted absent.

        Letting this one regress into actually running is not a red test, it is a
        hung one: the sweep runs on its own ``QThread`` against the mock
        potentiostat and finishes by opening a matplotlib window, and the first
        attempt at this mutation check wedged for the full ten-minute bound
        instead of failing. A refusal test must not be able to start the thing it
        is refusing.
        """
        from softae.config.loader import pico_for_channel

        def boom(*a, **k):
            raise AssertionError("an EIS acquisition was built despite the refusal")

        monkeypatch.setattr("softae.gui.tabs.tab_manual._ManualEisWorker", boom)
        hosted_tab._edit_eis_ch.setText("1")
        rig_activity.acquire("ht:sweep", {pico_for_channel(1)})

        hosted_tab._on_eis_run()

        assert hosted_tab._eis_thread is None, "an EIS sweep was started anyway"
        assert "manual EIS" in hosted_tab._lbl_last_command.text()
        assert "Refused" in hosted_tab._lbl_last_command.text()

    def test_manual_eis_is_not_refused_by_a_claim_on_the_other_pico(
            self, hosted_tab, rig_activity):
        """Asserted at the seam rather than through the slot, deliberately.

        Driving the permitted branch of `_on_eis_run` would start a real
        acquisition thread against the mock potentiostat; the property under test
        is the *scope* the slot hands the seam, and that scope is
        `{pico_for_channel(ch) for ch in channels}` either way.
        """
        from softae.config.loader import pico_for_channel

        others = {pico_for_channel(ch) for ch in range(1, 33)} - {pico_for_channel(1)}
        if not others:
            pytest.skip("this rig routes every channel to one potentiostat")
        rig_activity.acquire("ht:sweep", others)

        assert hosted_tab._note_manual_actuation(
            "manual EIS", {pico_for_channel(1)}) is None

    def test_the_refusal_leaves_a_log_line_naming_the_owner(
            self, hosted_tab, rig_activity):
        rig_activity.acquire("ht:cast_series", {"syringe"})

        with structlog.testing.capture_logs() as logs:
            hosted_tab._note_manual_actuation("pump 0 dispense", {"syringe"})

        entry = next(e for e in logs
                     if e["event"] == "manual_actuation_refused_while_running")
        assert entry["action"] == "pump 0 dispense"
        assert entry["owner"] == "ht:cast_series"
        assert entry["instruments"] == ["syringe"]

    def test_the_refusal_opens_no_modal_dialog(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        """A queued `failed` signal can reach these slots from a worker thread; a
        modal there blocks the event loop and wedges a headless run."""
        def boom(*a, **k):
            raise AssertionError("the refusal must not open a dialog")

        monkeypatch.setattr("softae.gui.tabs.tab_manual.QMessageBox.warning", boom)
        monkeypatch.setattr("softae.gui.tabs.tab_manual.QMessageBox.information", boom)
        # Stubbed so a regression here calls a harmless lambda rather than the
        # driver, and `settle_qt` drains the worker's signals before monkeypatch
        # puts the real modal back (see the fixture's own note).
        monkeypatch.setattr(hosted_tab._manager.get("syringe"), "head_descend",
                            lambda: None)
        rig_activity.acquire("campaign:phase_map:run-7")

        hosted_tab._on_head_descend()
        settle_qt(hosted_tab)


class TestARefusedDispenseWritesNothingAndAsksNothing:
    """The occupancy prompt and its store write must sit *behind* the refusal.

    `_on_infuse` deliberately notes the actuation late, so that the log line
    means "fluid was commanded" rather than "a button was pressed". Everything
    ahead of that note, though, includes a modal asking whether the electrode
    board has been replaced — and a "fresh board" answer is persisted
    immediately, on purpose, so a failed dispense cannot lose it.

    Late refusal plus early persistence is the defect: the operator is asked a
    question about a dispense that was never going to happen, answers it, and
    the store is left asserting a fresh board nothing was ever cast on. Board
    occupancy is what gates re-casting, so that is a persisted false statement
    about the physical rig, not a cosmetic one.

    Both directions are pinned. The refused case asserts the store is untouched
    *and* that the refusal was actually reached; the permitted case runs the
    identical setup with no claim and asserts the prompt, the persistence and
    the actuation line all still happen — which is what stops the first from
    passing because the slot returned somewhere else entirely.
    """

    @pytest.fixture
    def store_tab(self, qapp, manager, rig_activity, tmp_path, monkeypatch):
        """A hosted tab with a DataStore, parked head-down over electrode 1.

        `hosted_tab` carries no store, so `_pending_cast_target` returns None
        there and the occupancy branch under test is never entered.
        """
        from softae.core.data_store import DataStore
        from softae.core.deposition_steps import deposition_positions
        from softae.gui.widgets import rig_owner

        # Keep these tests off the machine's real `~/.softae/rig.lock`: the
        # permitted path calls `refresh_rig_owner()`, and a live GUI on this
        # bench would otherwise decide what the status line says.
        monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: None)
        ds = DataStore(tmp_path / "proj")
        host = QWidget()
        host._rig_activity = rig_activity
        widget = ManualControlTab(manager, parent=host, data_store=ds)

        manager.get("syringe").set_head_state(False)          # head DOWN → a cast
        ox, oy = deposition_positions().origin
        manager.get("stage").live_position = lambda: (ox, oy, 0.0)
        ds.record_electrode_cast(0, 1)                        # E1 already used
        widget._chk_apply_correction.setChecked(False)

        yield widget, ds

        widget.cleanup()
        host.close()
        ds.close()

    def test_a_refused_infuse_asks_nothing_and_leaves_occupancy_untouched(
            self, store_tab, rig_activity, monkeypatch, settle_qt):
        tab, ds = store_tab
        rig_activity.acquire("ht:cast_series", {"syringe"})

        def no_prompt(*a, **k):
            raise AssertionError(
                "the operator was asked about the board for a refused dispense")

        def no_pump(*a, **k):
            raise AssertionError("fluid was commanded despite the refusal")

        monkeypatch.setattr(occupancy_guard, "prompt_board_replaced", no_prompt)
        monkeypatch.setattr(tab._manager.get("syringe"), "single_pump", no_pump)

        tab._on_infuse(0)
        settle_qt(tab)

        # The path was entered and *refused* — without this the assertions below
        # would pass just as well if the slot had returned before reaching them.
        text = tab._lbl_last_command.text()
        assert "Refused" in text
        assert "pump 0 dispense" in text
        assert "ht:cast_series" in text

        assert ds.current_board_id() == 0, "a board swap was persisted for a refusal"
        assert ds.occupied_electrodes(0) == {1}, "occupancy changed under a refusal"
        assert ds.occupied_electrodes(1) == set(), "a fresh board was invented"

    def test_a_permitted_infuse_still_prompts_persists_and_records(
            self, store_tab, monkeypatch, settle_qt):
        """The same setup with no claim — the behaviour the refusal must not cost.

        This is also what proves the refused case above is not vacuous: the
        prompt this asserts *did* fire is the one the refused case asserts was
        never reached, from an identical rig state.
        """
        from softae.gui.widgets.occupancy_guard import BoardReplacedDecision

        tab, ds = store_tab
        asked = []
        monkeypatch.setattr(
            occupancy_guard, "prompt_board_replaced",
            lambda *a, **k: (asked.append(a), BoardReplacedDecision.FRESH)[1])

        tab._on_infuse(0)
        settle_qt(tab)

        assert asked, "the occupied-well prompt was not shown"
        assert ds.current_board_id() == 1, "the fresh board was not persisted"
        assert ds.occupied_electrodes(1) == {1}, "the cast was not recorded"
        assert "Last command: Pump 0" in tab._lbl_last_command.text()


class TestTheBusyRigMessageStaysTrue:
    """`busy_rig_message` ended with "manual control at the rig is never
    refused", which the refusal above makes false.

    Tested here because this tab is what made it false. The message varies only
    by ``action`` and by the lock it describes — the closing paragraph is a
    constant — so one case covers every caller of it (`gui/app.py`,
    `tab_arrhenius`, `_autonomous_run`, `tools/campaign.py`, `tools/env_hold.py`,
    `tools/eis_validate.py`).
    """

    def _message(self) -> str:
        from softae.core.run_lock import RunLock, busy_rig_message

        return busy_rig_message(
            RunLock(pid=4321, what="campaign:overnight:run-7",
                    started_at="2026-08-23T14:02:00+00:00", host="other-host"),
            action="This campaign")

    def test_it_no_longer_says_manual_control_is_never_refused(self):
        assert "never refused" not in self._message()

    def test_it_still_draws_the_line_and_draws_it_at_the_right_mechanism(self):
        """Dropping the sentence would be the other way to make it not-false, and
        would leave the message reading as the blanket lockout the ruling
        forbids. It has to keep saying what this refusal is *not*."""
        text = self._message()

        assert "Manual control" in text
        assert "this lock" in text, "the exemption must be scoped to the lock"
        assert "paus" in text.lower(), "the other refusal's way out is unnamed"


class TestNonActuatingControlsAreNeverRefused:
    """A read, a render and a bookkeeping dialog are not actuations.

    The refusal is scoped to the slots that put a command on a bus. Everything
    that only *shows* something has to keep working through a whole-rig claim,
    or the tab stops being the thing an operator watches a run with.
    """

    @pytest.fixture
    def busy_tab(self, hosted_tab, rig_activity):
        rig_activity.acquire("campaign:phase_map:run-7")
        return hosted_tab

    def test_readouts_still_update_from_the_polling_worker(self, busy_tab):
        busy_tab._on_pv_poll_done({"temp_sp": 40.0, "temp_pv": 39.2, "rh": 33.0,
                                   "pos": (1.0, 2.0)})

        assert "40.0" in busy_tab._lbl_temp_sp.text()
        assert "39.2" in busy_tab._lbl_temp_pv.text()
        assert "33.0" in busy_tab._lbl_rh.text()
        assert "1.00" in busy_tab._lbl_pos.text()

    def test_the_polling_worker_still_reads_the_instruments(self, busy_tab):
        """A poll is a read, not an actuation — and the tab is how a run is watched."""
        assert busy_tab._pv_worker is not None
        # A key the mock rig really produces (see
        # `test_poll_worker_propagates_parallel_map_from_status`), so an empty
        # dict cannot pass this.
        assert "parallel_syringes" in busy_tab._pv_worker.poll_once()

    def test_display_only_controls_still_respond(self, busy_tab):
        busy_tab._on_eis_preset_changed("Quick")
        busy_tab._edit_eis_ch.setText("2,4,5-10")
        busy_tab._update_eis_pico_label()
        busy_tab._on_sigma_mode_toggled(False)
        busy_tab.refresh_head_label()
        busy_tab.refresh_stock_labels()

        assert busy_tab._lbl_eis_pico.text() != "—"
        assert busy_tab._spin_eis_K.isEnabled()

    def test_the_camera_is_not_gated(self, busy_tab):
        """Snap and live preview never routed through the ownership note and do
        not start now: a frame is light, and nothing on the bus receives it."""
        busy_tab._on_snap()
        busy_tab._on_live_toggle(True)
        busy_tab._on_live_toggle(False)

        assert "Refused" not in busy_tab._lbl_last_command.text()

    def test_declaring_syringe_stock_is_not_gated(self, busy_tab, monkeypatch):
        """Bookkeeping the operator does *while* a run is held, not a command."""
        shown = []
        monkeypatch.setattr(
            "softae.gui.tabs.tab_manual.QMessageBox.information",
            lambda *a, **k: shown.append(a))

        busy_tab._on_report_stock()

        assert shown, "the stock dialog was not reached"
        assert "Refused" not in busy_tab._lbl_last_command.text()


class TestAWindowlessTabIsFullyUsable:
    """Most of the suite builds this tab with no parent; `window()` then returns
    the tab itself, which carries no registry. Same shape and same reason as
    `rig_claim.rig_run`'s null context."""

    def test_no_window_means_no_owner_and_no_refusal(
            self, tab, monkeypatch, settle_qt):
        assert tab._rig_run_owner(MANUAL_INSTRUMENTS) is None

        calls = []
        monkeypatch.setattr(tab._manager.get("stage"), "move_by",
                            lambda dx, dy: calls.append((dx, dy)))
        tab._spin_jog_step.setValue(1.0)
        tab._on_jog(1, 0)
        settle_qt(tab)

        assert calls == [(1.0, 0.0)]

    def test_another_windows_claim_cannot_reach_a_windowless_tab(
            self, tab, monkeypatch, settle_qt):
        """The registry is per-window, so a claim held elsewhere is not this
        tab's business — and looking one up globally is what would make the
        windowless path raise."""
        elsewhere = RigActivity()
        elsewhere.acquire("campaign:somewhere_else")
        assert elsewhere.conflicts({"lamp"}) == "campaign:somewhere_else"
        calls = []
        monkeypatch.setattr(tab._manager.get("lamp"), "on",
                            lambda: calls.append("on"))

        tab._on_lamp_on()

        assert calls == ["on"]


class _FakeLedger:
    """Enough ledger to be *attached*; the dialog itself is recorded, not built."""

    hard_stop_uL = 100.0
    soft_warn_uL = 500.0

    def remaining_uL(self, pump_id):    # drives `refresh_stock_labels`
        return None


class TestTheStockDialogIsOpenedWithItsStore:
    """`_on_report_stock` opened ``ReservoirDialog(ledger, parent=self)`` with no
    store at all, which disables the stock dropdown and turns
    ``save_loadout(None, ...)`` into a silent no-op — the operator's declaration
    discarded without a word. The store is what makes that combo writable."""

    @staticmethod
    def _open(widget, monkeypatch) -> list:
        """Press Report Stock with a ledger attached, recording the dialog."""
        calls: list[dict] = []

        class _Recorder:
            def __init__(self, ledger, parent=None, **kwargs):
                calls.append({"ledger": ledger, "parent": parent, **kwargs})

            def exec(self):
                return 0

        # The handler imports the dialog at call time, so the module attribute
        # is the binding it will resolve.
        monkeypatch.setattr(
            "softae.gui.widgets.reservoir_dialog.ReservoirDialog", _Recorder)
        monkeypatch.setattr(widget._manager.get("syringe"), "reservoir_ledger",
                            _FakeLedger(), raising=False)

        widget._on_report_stock()
        return calls

    def test_report_stock_hands_the_dialog_the_tabs_data_store(
            self, qapp, manager, monkeypatch):
        store = object()
        widget = ManualControlTab(manager, data_store=store)
        try:
            calls = self._open(widget, monkeypatch)
        finally:
            _shutdown_tab(widget)

        assert len(calls) == 1, "the stock dialog was not opened"
        assert calls[0]["data_store"] is store

    def test_report_stock_passes_no_store_when_the_tab_has_none(
            self, tab, monkeypatch):
        """The store is *threaded from the tab*, not conjured at the call site —
        so a tab built without one still says so, and the dialog's own note
        about an unsaveable pick is what the operator sees."""
        calls = self._open(tab, monkeypatch)

        assert len(calls) == 1
        assert calls[0]["data_store"] is None

    def test_report_stock_passes_no_catalog_and_relies_on_the_fallback(
            self, tab, monkeypatch):
        """This tab is never handed a solution catalog; the dialog falls back to
        ``catalogs_from_data_root()``, which reads the same two CSVs off the same
        data root that ``MainWindow._load_catalogs_from_root`` does."""
        calls = self._open(tab, monkeypatch)

        assert "sol_catalog" not in calls[0]
        assert not hasattr(tab, "_sol_catalog")


# ── Manual ramp: progress line, Abort, and the abort_ramp contract ───────────
#
# Spec: docs/SubAgent docs/manual_temp_ramp_progress_abort.md §3. The mock
# controller's ramp is instant, so a ramp is held "running" by gating one of its
# setpoint writes on an Event. Every gate is released at teardown, before the
# tab fixture's close joins the worker.

import threading as _threading

from structlog.testing import capture_logs

from softae.drivers.temp_ramp import ramp_schedule, ramp_span_s
from softae.gui.tabs.tab_manual_workers import (
    RAMP_ABORTING_TEXT,
    RAMP_IDLE_TEXT,
    RAMP_REFUSED_TEXT,
    _RampWorker,
)

_GATE_BOUND_S = 5.0


class _WriteGate:
    """Holds the mock controller's *nth* ``write_sp`` call until released."""

    def __init__(self, tc, nth: int) -> None:
        self.entered, self.release = _threading.Event(), _threading.Event()
        self.calls: list[float] = []
        self._orig, self._nth = tc.write_sp, nth

    def __call__(self, T_SP, print_flag=1):
        self.calls.append(T_SP)
        if len(self.calls) == self._nth:
            self.entered.set()
            assert self.release.wait(_GATE_BOUND_S), "write gate never released"
        self._orig(T_SP, print_flag)


@pytest.fixture
def gate_writes(tab, monkeypatch):
    # A gate that times out fails the worker, and `failed` opens a modal that
    # would wedge the run; a failing test must fail, not hang.
    monkeypatch.setattr("softae.gui.tabs.tab_manual.QMessageBox.warning",
                        lambda *a, **k: None)
    gates: list[_WriteGate] = []

    def _make(nth: int = 1) -> _WriteGate:
        tc = tab._manager.get("temp_controller")
        gate = _WriteGate(tc, nth)
        monkeypatch.setattr(tc, "write_sp", gate)
        gates.append(gate)
        return gate

    yield _make
    for gate in gates:
        gate.release.set()


def _wait_for(predicate, timeout_s: float = _GATE_BOUND_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start_held_ramp(tab, gate_writes, nth: int = 1, end: float = 30.0) -> _WriteGate:
    """Start a 25 -> *end* ramp held inside its *nth* setpoint write."""
    gate = gate_writes(nth)
    tab._spin_ramp_end.setValue(end)
    tab._on_ramp()
    assert gate.entered.wait(_GATE_BOUND_S), "the ramp never reached its write"
    return gate


class TestManualRampProgressAndAbort:
    def test_ramp_progress_label_updates_from_signal(self, tab, gate_writes):
        gate = _start_held_ramp(tab, gate_writes, nth=2)   # step 1 written
        assert _wait_for(lambda: tab._lbl_ramp_progress.text().startswith("Step 1/"))
        text = tab._lbl_ramp_progress.text()
        assert "SP 25.0 °C → 30.0 °C" in text and "ETA" in text
        assert "cooling is passive" not in text             # upward ramp
        gate.release.set()

    def test_ramp_progress_downward_appends_passive_cooling_note(self, tab, gate_writes):
        gate = _start_held_ramp(tab, gate_writes, nth=2, end=20.0)
        assert _wait_for(lambda: "cooling is passive" in tab._lbl_ramp_progress.text())
        assert tab._lbl_ramp_progress.text().count("cooling is passive") == 1
        gate.release.set()

    def test_ramp_abort_button_enabled_only_while_running(self, tab, gate_writes, settle_qt):
        assert not tab._btn_ramp_abort.isEnabled()
        assert tab._lbl_ramp_progress.text() == RAMP_IDLE_TEXT
        gate = _start_held_ramp(tab, gate_writes)
        assert tab._btn_ramp_abort.isEnabled() and not tab._btn_ramp.isEnabled()
        gate.release.set()
        settle_qt(tab)
        assert _wait_for(lambda: tab._ramp_worker is None)
        assert not tab._btn_ramp_abort.isEnabled() and tab._btn_ramp.isEnabled()
        assert tab._lbl_ramp_progress.text().startswith("Ramp complete — SP 30.0 °C")

    def test_ramp_abort_sets_cancel_and_reports_holding(self, tab, gate_writes, settle_qt):
        gate = _start_held_ramp(tab, gate_writes)            # step 1 in flight
        worker = tab._ramp_worker
        tab._btn_ramp_abort.click()
        assert worker.cancel.is_set()
        assert tab._lbl_ramp_progress.text() == RAMP_ABORTING_TEXT
        assert not tab._btn_ramp_abort.isEnabled()
        gate.release.set()
        settle_qt(tab)
        assert _wait_for(lambda: "holding SP" in tab._lbl_ramp_progress.text())
        total = len(ramp_schedule(25.0, 30.0, ramp_span_s(25.0, 30.0, 1.0))[0])
        assert tab._lbl_ramp_progress.text() == (
            f"Ramp aborted at step 1/{total} — holding SP 25.0 °C")
        assert gate.calls == [25.0], "the ramp wrote a step after the abort"

    def test_ramp_zero_span_shows_already_at_target(self, tab, settle_qt):
        tab._spin_ramp_end.setValue(25.0)                    # the mock SP
        tab._on_ramp()
        settle_qt(tab)
        assert _wait_for(lambda: tab._ramp_worker is None)
        assert tab._lbl_ramp_progress.text() == "Already at 25.0 °C — nothing to ramp"

    def test_ramp_refused_shows_refusal_in_progress_label(
            self, hosted_tab, rig_activity, monkeypatch, settle_qt):
        rig_activity.acquire("campaign:phase_map:run-7")
        calls = []
        monkeypatch.setattr(hosted_tab._manager.get("temp_controller"), "ramp_linear",
                            lambda *a, **k: calls.append(k))
        hosted_tab._on_ramp()
        settle_qt(hosted_tab)
        assert calls == []
        assert hosted_tab._lbl_ramp_progress.text() == RAMP_REFUSED_TEXT
        assert hosted_tab._btn_ramp.isEnabled() and not hosted_tab._btn_ramp_abort.isEnabled()

    def test_set_temp_disabled_while_ramp_runs(self, tab, gate_writes, settle_qt):
        gate = _start_held_ramp(tab, gate_writes)
        assert not tab._btn_set_temp.isEnabled()
        assert tab._btn_set_temp.toolTip() == "Abort the ramp first"
        gate.release.set()
        settle_qt(tab)
        assert _wait_for(lambda: tab._btn_set_temp.isEnabled())
        assert tab._btn_set_temp.toolTip() == ""

    def test_ramp_failure_shows_failed_text_and_reenables(self, tab, monkeypatch, settle_qt):
        monkeypatch.setattr("softae.gui.tabs.tab_manual.QMessageBox.warning",
                            lambda *a, **k: None)

        def boom(**kwargs):
            raise RuntimeError("SP above the safety limit")

        monkeypatch.setattr(tab._manager.get("temp_controller"), "ramp_linear", boom)
        tab._spin_ramp_end.setValue(30.0)
        tab._on_ramp()
        settle_qt(tab)
        assert _wait_for(lambda: tab._ramp_worker is None)
        assert tab._lbl_ramp_progress.text() == (
            "Ramp failed before its first step: SP above the safety limit")
        assert tab._btn_ramp.isEnabled() and tab._btn_set_temp.isEnabled()

    def test_cleanup_mid_ramp_cancels_and_joins_worker(self, tab, gate_writes):
        gate = _start_held_ramp(tab, gate_writes)
        worker = tab._ramp_worker
        reports = []
        worker.completed.connect(reports.append, type=_direct())
        # cleanup() joins on this thread, so the in-flight write is released from another.
        _threading.Timer(0.2, gate.release.set).start()
        tab.cleanup()
        assert worker.cancel.is_set()
        assert not worker.isRunning(), "cleanup() returned with the ramp still running"
        assert reports and reports[0].cancelled and reports[0].last_sp == 25.0
        assert gate.calls == [25.0]

    def test_cleanup_ramp_join_timeout_logs_warning(self, tab, gate_writes, monkeypatch):
        gate = _start_held_ramp(tab, gate_writes)
        worker = tab._ramp_worker
        monkeypatch.setattr(worker, "stop_worker",
                            lambda timeout_ms=3000: (worker.cancel.set(), False)[1])
        with capture_logs() as logs:
            tab.cleanup()
        assert "manual_ramp_join_timeout" in [e["event"] for e in logs]
        gate.release.set()
        worker.wait(int(_GATE_BOUND_S * 1000))


def _direct():
    from PySide6.QtCore import Qt

    return Qt.ConnectionType.DirectConnection


class TestAbortRampContract:
    """``abort_ramp`` is called by ``MainWindow.notify_parked`` on every park."""

    def test_abort_ramp_no_ramp_running_is_noop(self, tab):
        tab.abort_ramp()
        assert tab._lbl_ramp_progress.text() == RAMP_IDLE_TEXT
        assert not tab._btn_ramp_abort.isEnabled() and tab._btn_ramp.isEnabled()

    def test_abort_ramp_after_ramp_finished_is_noop(self, tab, settle_qt):
        tab._spin_ramp_end.setValue(30.0)
        tab._on_ramp()
        settle_qt(tab)
        assert _wait_for(lambda: tab._ramp_worker is None)
        done = tab._lbl_ramp_progress.text()
        assert done.startswith("Ramp complete")
        tab.abort_ramp()
        assert tab._lbl_ramp_progress.text() == done
        assert tab._btn_ramp.isEnabled()

    def test_abort_ramp_second_call_is_noop(self, tab, gate_writes, settle_qt):
        gate = _start_held_ramp(tab, gate_writes)
        with capture_logs() as logs:
            tab.abort_ramp()
            tab.abort_ramp()
        assert [e["event"] for e in logs].count("manual_ramp_abort_requested") == 1
        assert tab._lbl_ramp_progress.text() == RAMP_ABORTING_TEXT
        gate.release.set()
        settle_qt(tab)

    def test_abort_ramp_while_write_blocked_returns_immediately(
            self, tab, gate_writes, settle_qt):
        """It sets the event and returns; it never joins. The ramp's write is
        held for the whole call, so a join would block until the gate bound."""
        gate = _start_held_ramp(tab, gate_writes)
        worker = tab._ramp_worker
        t0 = time.monotonic()
        tab.abort_ramp()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"abort_ramp blocked the GUI thread for {elapsed:.2f} s"
        assert worker.cancel.is_set() and worker.isRunning()
        assert not gate.release.is_set()
        gate.release.set()
        settle_qt(tab)

    def test_ramp_worker_is_a_command_worker_joined_by_settle_qt(self):
        assert issubclass(_RampWorker, _CommandWorker)


# ── Manual EIS: the saved header and the conditions row share one snapshot ───
#
# eis-acq [e147]: `_measure_one` saved each spectrum BEFORE reading the
# environment, so every manual `.txt` header read T_SP/T_PV/RH_SP/RH_PV = nan
# while the DB `conditions` row held the real values. The header is asserted by
# reading the file back, because the file is what an operator actually opens.

_ENV_STAGE_SP, _ENV_STAGE_PV = 60.0, 58.4
_ENV_RH_SP, _ENV_RH_PV, _ENV_AIR = 22.0, 23.7, 26.1


def _env_manager(pico):
    """A manager whose temp/RH controllers answer known, distinct values."""
    from types import SimpleNamespace

    temp = SimpleNamespace(get_sp=lambda: _ENV_STAGE_SP, get_pv=lambda: _ENV_STAGE_PV)
    rh = SimpleNamespace(get_T=lambda: _ENV_AIR, get_H=lambda: _ENV_RH_PV,
                         status=lambda: {"setpoint": _ENV_RH_SP})
    instruments = {"pico1": pico, "temp_controller": temp, "rh_controller": rh}
    return SimpleNamespace(names=list(instruments), get=instruments.__getitem__)


def test_manual_eis_measure_one_saved_header_carries_environment(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    import softae.core.conditions_capture as conditions_capture
    from softae.analysis.eis_data import EISResult
    from softae.core.data_store import DataStore
    from softae.gui.tabs.tab_manual import _ManualEisWorker

    spectrum = _eis_result(3)
    monkeypatch.setattr("softae.drivers.mscr_library.eis_run_mscrbuild",
                        lambda *a, **k: None)
    monkeypatch.setattr(EISResult, "from_raw",
                        classmethod(lambda cls, raw, **kw: spectrum))
    reads = []
    real_read = conditions_capture.read_environment
    monkeypatch.setattr(conditions_capture, "read_environment",
                        lambda m: reads.append(m) or real_read(m))

    pico = SimpleNamespace(sendscript_getdata=lambda *a: object(), _output_dir=".")
    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("manual_eis", mode="manual",
                                 campaign="manual", quality="explore")
        worker = _ManualEisWorker(
            _env_manager(pico), store, channels=[3], eis_params={},
            auto_fit=False, fit_model="simpleSalt", auto_save=True)
        payload = worker._measure_one(3, "pico1", run_id)
        rows = store.query_conditions(measurement_id=payload["measurement_id"])

    saved = EISResult.load(spectrum.raw_file_path)
    header = (saved.T_sp, saved.T_pv, saved.rh_sp, saved.rh_pv)
    assert header == (_ENV_STAGE_SP, _ENV_STAGE_PV, _ENV_RH_SP, _ENV_RH_PV)
    assert len(reads) == 1, "one snapshot per measurement feeds header and row"
    (row,) = rows
    assert (row["stage_temp_sp_C"], row["stage_temp_pv_C"],
            row["rh_sp_pct"], row["rh_pv_pct"]) == header
    assert row["chamber_air_C"] == _ENV_AIR
