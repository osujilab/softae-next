"""Tests for the Experiment Builder tab (tab_experiment.py).

Consolidated: removed trivial construction checks, merged duplicate signal slot
tests, and parameterised channel-spec / pico-routing tables. Each test function
now exercises a distinct behaviour.
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_experiment import ExperimentBuilderTab
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow

# ── Fixtures ─────────────────────────────────────────────────────────────


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
    widget = ExperimentBuilderTab(manager)
    # Deterministic baseline: pin the deposition recipe to 'single_drop' regardless
    # of the operator's [dropcast].default_recipe. Tests that need two_phase opt in
    # via _select_recipe(tab, "two_phase").
    i = widget._combo_deposit_recipe.findData("single_drop")
    if i >= 0:
        widget._combo_deposit_recipe.setCurrentIndex(i)
    yield widget
    widget.close()



# ── Formulation table ─────────────────────────────────────────────────────


class TestFormulationTable:
    """PCB-driven formulation table behaviour."""

    def test_read_formulation_matrix(self, tab: ExperimentBuilderTab):
        # Columns 2–5 are Pump 0 / Pump 1 / Pump 2 / Total.
        tab._form_table.setItem(0, 2, QTableWidgetItem("10.5"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("5.0"))
        tab._form_table.setItem(0, 4, QTableWidgetItem("2.5"))
        tab._form_table.setItem(0, 5, QTableWidgetItem("18.0"))
        matrix = tab._read_formulation_matrix()
        assert isinstance(matrix, list)
        assert matrix[0] == [10.5, 5.0, 2.5, 18.0]
        # Non-numeric falls back to 0.0
        tab._form_table.setItem(0, 2, QTableWidgetItem("abc"))
        assert tab._read_formulation_matrix()[0][0] == 0.0

    def test_formulation_table_has_three_pump_columns(self, tab: ExperimentBuilderTab):
        # Checkbox + Channel + Pump 0/1/2 + Total = 6 columns.
        assert tab._form_table.columnCount() == 6
        headers = [
            tab._form_table.horizontalHeaderItem(c).text()
            for c in range(tab._form_table.columnCount())
        ]
        assert headers[2:] == ["Pump 0 (µL)", "Pump 1 (µL)", "Pump 2 (µL)", "Total (µL)"]

    def test_selected_channels_partial(self, tab: ExperimentBuilderTab):
        tab._form_table.cellWidget(0, 0).setChecked(False)
        if tab._form_table.rowCount() > 2:
            tab._form_table.cellWidget(2, 0).setChecked(False)
        selected = tab._selected_channels()
        assert 0 not in selected
        if tab._form_table.rowCount() > 2:
            assert 2 not in selected


# ── Workflow generation ───────────────────────────────────────────────────


class TestWorkflowGeneration:
    """Building a Workflow from the UI state."""

    @pytest.fixture(autouse=True)
    def _disable_piezo(self):
        """Pin piezo config to disabled for this class.

        The base workflow contract exercised here (1 + 4*n + 1 steps, no piezo
        setup/teardown steps) only holds when piezo liquid-events are disabled.
        The real ``softae_config.toml`` has piezo ENABLED by default, and the
        global config cache can carry either state depending on test-collection
        order, so pin the input explicitly to make these tests deterministic
        regardless of ordering. Patched where tab_experiment.py calls it.
        """
        with patch(
            "softae.gui.tabs.tab_experiment.piezo_config",
            return_value={"enabled": False, "liquid_events": {"enabled": False}},
        ):
            yield

    def test_generates_workflow(self, tab: ExperimentBuilderTab):
        # Default engine recipe (single_drop) — startup flush, deposit, final flush.
        wf = tab._generate_workflow()
        assert isinstance(wf, Workflow)
        assert wf.name.startswith("ht_")
        assert wf.metadata["source"] == "deposition_engine"
        assert wf.setup[0].name == "startup_flush"
        assert wf.teardown[0].name == "final_flush"

    def test_workflow_has_loop_steps(self, tab: ExperimentBuilderTab):
        # Per-channel steps are flattened into setup; no loop.
        wf = tab._generate_workflow()
        setup_names = [s.name for s in wf.setup]
        assert any(n.startswith("deposit_ch") for n in setup_names)
        assert any(n.startswith("measure_eis_ch") for n in setup_names)
        assert wf.loop_steps == []

    def test_total_steps_calculated(self, tab: ExperimentBuilderTab):
        wf = tab._generate_workflow()
        n = len(tab._selected_channels())
        # single_drop: startup + (deposit + EIS) per channel + final flush.
        assert wf.total_steps == 1 + 2 * n + 1
        assert len(wf.resolve_steps()) == wf.total_steps

    def test_deposit_uses_ui_rate(self, tab: ExperimentBuilderTab):
        tab._spin_rate.setValue(123.4)
        wf = tab._generate_workflow()
        deposit = [s for s in wf.setup if s.name.startswith("deposit_ch")][0]
        assert deposit.params["disp_rate"] == 123.4   # single_drop = flat rate

    def test_generate_workflow_uses_formulation_matrix_values(self, tab: ExperimentBuilderTab):
        # Columns 2/3/4 are Pump 0 / Pump 1 / Pump 2.
        tab._form_table.setItem(0, 2, QTableWidgetItem("11.0"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("7.5"))
        tab._form_table.setItem(0, 4, QTableWidgetItem("4.0"))
        wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch1"))
        assert dep.params["vols"] == [pytest.approx(11.0), pytest.approx(7.5), pytest.approx(4.0)]
        assert dep.params["ids"] == [0, 1, 2]

    def test_dispense_plan_carries_three_pumps(self, tab: ExperimentBuilderTab):
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)  # channel 1
        tab._form_table.setItem(0, 2, QTableWidgetItem("6.0"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("3.0"))
        tab._form_table.setItem(0, 4, QTableWidgetItem("1.0"))
        plan, _enabled, prime = tab._build_dispense_plan([0], tab._read_formulation_matrix())
        row = plan[0]
        assert row["commanded_uL"] == [pytest.approx(6.0), pytest.approx(3.0), pytest.approx(1.0)]
        assert row["pump2_commanded_uL"] == pytest.approx(1.0)
        assert row["total_commanded_uL"] == pytest.approx(10.0)
        # A prime estimate exists for every connected pump.
        assert set(prime.keys()) == set(tab.PUMP_IDS)

    def test_generate_workflow_applies_correction_when_enabled(self, tab: ExperimentBuilderTab):
        tab._form_table.setItem(0, 2, QTableWidgetItem("10.0"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("5.0"))
        with patch("softae.gui.tabs.tab_experiment.liquid_handling_config", return_value={
            "enabled": True,
            "beta": 0.30,
            "eta_ref_mpas": 1.0,
            "alpha_growth_per_run": 0.0,
            "pump_line": {"0": 0, "1": 1},
            "line": {
                "0": {
                    "cracking_kpa_per_valve": 8.0,
                    "compliance_uL_per_kpa": 0.55,
                    "alpha_base": 0.2,
                    "viscosity_mpas": 1.0,
                },
                "1": {
                    "cracking_kpa_per_valve": 8.0,
                    "compliance_uL_per_kpa": 0.55,
                    "alpha_base": 0.2,
                    "viscosity_mpas": 1.0,
                },
            },
        }), patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
            "enabled": False,
            "liquid_events": {"enabled": False},
        }):
            wf = tab._generate_workflow()
            tab._on_generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch1"))
        # Commanded volumes carry the dead-volume correction (> targets).
        assert dep.params["vols"][0] > 10.0
        assert dep.params["vols"][1] > 5.0
        assert wf.metadata["liquid_handling_enabled"] is True
        preview = tab._txt_preview.toPlainText()
        assert "liquid_correction: enabled" in preview
        assert "prime estimates:" in preview

    def test_measure_auto_routes_pico(self, tab: ExperimentBuilderTab):
        wf = tab._generate_workflow()
        eis = [s for s in wf.setup if s.name.startswith("measure_eis_ch")][0]
        assert eis.instrument == "pico1"

    def test_measure_only_mode(self, tab: ExperimentBuilderTab):
        """Measure Only: no syringe steps, all EIS steps in setup, no loop."""
        tab._combo_mode.setCurrentIndex(1)
        wf = tab._generate_workflow()
        n = len(tab._selected_channels())
        assert len(wf.setup) == n
        assert len(wf.teardown) == 0
        assert wf.loop_steps == []
        assert all(s.name.startswith("measure_eis_ch") for s in wf.setup)
        # Each step targets a distinct channel
        channels = [s.params["chan"] for s in wf.setup]
        assert channels == sorted(set(channels))
        assert wf.metadata["mode"] == "measure_only"

    def test_no_channels_selected_raises(self, tab: ExperimentBuilderTab):
        for r in range(tab._form_table.rowCount()):
            tab._form_table.cellWidget(r, 0).setChecked(False)
        with pytest.raises(ValueError, match="No channels selected"):
            tab._generate_workflow()

    def test_mode_change_hides_dispense(self, tab: ExperimentBuilderTab):
        tab._combo_mode.setCurrentIndex(1)
        assert tab._disp_grp.isHidden()
        tab._combo_mode.setCurrentIndex(0)
        assert not tab._disp_grp.isHidden()

    def test_preview_includes_liquid_correction_visibility(self, tab: ExperimentBuilderTab):
        tab._on_generate_workflow()
        text = tab._txt_preview.toPlainText()
        assert "liquid_correction:" in text
        assert "Dispense plan sample:" in text

    def test_preview_shows_target_vs_commanded_when_enabled(self, tab: ExperimentBuilderTab):
        tab._form_table.setItem(0, 2, QTableWidgetItem("10.0"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("5.0"))
        with patch("softae.gui.tabs.tab_experiment.liquid_handling_config", return_value={
            "enabled": True,
            "beta": 0.30,
            "eta_ref_mpas": 1.0,
            "alpha_growth_per_run": 0.0,
            "pump_line": {"0": 0, "1": 1},
            "line": {
                "0": {
                    "cracking_kpa_per_valve": 8.0,
                    "compliance_uL_per_kpa": 0.55,
                    "alpha_base": 0.2,
                    "viscosity_mpas": 1.0,
                },
                "1": {
                    "cracking_kpa_per_valve": 8.0,
                    "compliance_uL_per_kpa": 0.55,
                    "alpha_base": 0.2,
                    "viscosity_mpas": 1.0,
                },
            },
        }), patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
            "enabled": False,
            "liquid_events": {"enabled": False},
        }):
            tab._on_generate_workflow()
        text = tab._txt_preview.toPlainText()
        assert "liquid_correction: enabled" in text
        assert "target=" in text
        assert "commanded=" in text
        assert tab._lbl_liquid_correction.text() == "Enabled"


# ── Run controls ──────────────────────────────────────────────────────────


class TestRunControls:
    """Start / Pause / Abort button interactions."""

    def test_start_disables_start_button(self, tab: ExperimentBuilderTab):
        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True):
            tab._on_start()
        assert not tab._btn_start.isEnabled()
        assert tab._btn_pause.isEnabled()
        assert tab._btn_abort.isEnabled()
        assert isinstance(tab._executor, WorkflowExecutor)

    def test_start_aborts_when_head_declined(self, tab: ExperimentBuilderTab):
        """A declined head-position gate stops the run before any work."""
        with patch.object(tab, "_verify_head_position", return_value=False), \
                patch.object(tab, "_generate_workflow") as gen, \
                patch.object(tab, "_run_workflow_thread") as run:
            tab._on_start()
        gen.assert_not_called()
        run.assert_not_called()
        assert tab._btn_start.isEnabled()  # run never started

    def test_pause_toggle(self, tab: ExperimentBuilderTab):
        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True):
            tab._on_start()
        tab._executor._state = ExecutorState.RUNNING
        tab._on_pause()
        assert "Resume" in tab._btn_pause.text()
        tab._executor._state = ExecutorState.PAUSED
        tab._on_pause()
        assert "Pause" in tab._btn_pause.text()

    def test_abort_calls_executor_abort(self, tab: ExperimentBuilderTab):
        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True):
            tab._on_start()
        tab._executor.abort = MagicMock()
        tab._on_abort()
        tab._executor.abort.assert_called_once()


def _pump_until(qapp, predicate, *, timeout_s: float = 15.0) -> bool:
    """Run the Qt event loop from the main thread until *predicate* holds.

    The hold is entered and left on the executor's asyncio thread, so nothing it
    announces reaches a widget until the main thread pumps. Driving the tab
    synchronously would exercise a path production never takes.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return bool(predicate())


class TestCeilingHoldLegibility:
    """The Pause control during a consecutive-failure ceiling hold.

    The executor never self-resumes, so the run sits in ``_hold_for_operator``
    until the host answers or the deadline parks it. Through all of that the
    ordinary reading of the button — "Pause / Resume" — describes a state the
    run is not in: it is not waiting to be paused, it is already held, and the
    reason is a stack of failed channels the operator has not seen yet.
    """

    @staticmethod
    def _wire(tab, manager, *, claim=None, timeout_s: float = 60.0):
        """Wire a real executor's hold callbacks to the tab, as a run does."""
        from softae.gui.rig_claim import NULL_RIG_CLAIM

        executor = WorkflowExecutor(
            manager, max_consecutive_channel_failures=1,
            channel_hold_timeout_s=timeout_s,
        )
        executor._state = ExecutorState.RUNNING
        executor.on_state_change = tab._cb_state_change
        executor.on_channel_failure_hold = tab._cb_channel_hold
        executor.on_pause_hold = tab._pause_hold_callback(
            NULL_RIG_CLAIM if claim is None else claim
        )
        tab._executor = executor
        tab._btn_pause.setEnabled(True)
        return executor

    @staticmethod
    def _hold_on_a_thread(executor) -> threading.Thread:
        import asyncio

        def _drive() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(executor._hold_for_operator("7", 3))
            finally:
                loop.close()

        thread = threading.Thread(target=_drive, daemon=True)
        thread.start()
        return thread

    def test_the_control_says_the_run_is_already_held(self, qapp, tab, manager):
        executor = self._wire(tab, manager)
        thread = self._hold_on_a_thread(executor)
        try:
            assert _pump_until(
                qapp, lambda: "Continue plate" in tab._btn_pause.text()
            ), f"button never relabelled (text={tab._btn_pause.text()!r})"

            tip = tab._btn_pause.toolTip()
            assert "already held" in tip
            assert "nothing to pause" in tip
            assert "3 channels" in tip and "7" in tip   # the reason, in words
            # Relabelled, not disabled: continuing the plate is one of the two
            # answers, and after the non-modal prompt is dismissed this button
            # is the only way left to give it.
            assert tab._btn_pause.isEnabled()
        finally:
            executor.abort()
            thread.join(timeout=10.0)
            _pump_until(qapp, lambda: not thread.is_alive())
            if tab._hold_box is not None:
                tab._hold_box.close()
        assert not thread.is_alive()

    def test_the_control_returns_to_normal_once_the_hold_ends(
        self, qapp, tab, manager
    ):
        executor = self._wire(tab, manager)
        thread = self._hold_on_a_thread(executor)
        try:
            assert _pump_until(
                qapp, lambda: "Continue plate" in tab._btn_pause.text()
            )
            executor.resume()                       # the operator answers
            assert _pump_until(qapp, lambda: not thread.is_alive())
        finally:
            thread.join(timeout=10.0)
            if tab._hold_box is not None:
                tab._hold_box.close()

        assert _pump_until(qapp, lambda: tab._btn_pause.toolTip() == "")
        assert "Pause" in tab._btn_pause.text()
        assert "Continue plate" not in tab._btn_pause.text()
        assert tab._ceiling_hold is None

    def test_the_hold_callback_touches_no_widget_on_its_own_thread(
        self, qapp, tab, manager
    ):
        """The callback fires on the executor's asyncio thread.

        A direct widget call from there is undefined behaviour that mostly
        looks like it works, so the crossing is asserted as a *delay*: nothing
        may change until the main thread pumps. Wiring ``on_pause_hold``
        straight to ``_ui_pause_hold`` fails this and passes everything else.
        """
        self._wire(tab, manager)
        tab._ceiling_hold = ("7", 3, 60.0)
        before = tab._btn_pause.text()
        fire = tab._executor.on_pause_hold

        thread = threading.Thread(target=fire, args=(True,), daemon=True)
        thread.start()
        thread.join(timeout=10.0)
        assert not thread.is_alive()

        assert tab._btn_pause.text() == before, "the widget was touched directly"
        assert _pump_until(
            qapp, lambda: "Continue plate" in tab._btn_pause.text()
        ), "the announcement never crossed to the GUI thread"

    def test_an_ordinary_operator_pause_is_left_alone(self, qapp, tab, manager):
        """Only the ceiling hold is special.

        A run held at the operator's own Pause must keep the button that gets it
        back; relabelling every hold would strand them.
        """
        self._wire(tab, manager)
        tab._executor._state = ExecutorState.PAUSED
        tab._ui_state_change("RUNNING", "PAUSED")
        assert "Resume" in tab._btn_pause.text()

        tab._ui_pause_hold(True)                    # no ceiling hold recorded

        assert "Resume" in tab._btn_pause.text()
        assert tab._btn_pause.toolTip() == ""

    def test_the_hold_still_suspends_the_runs_claim(self, qapp, tab, manager):
        """The safety half of the callback survived being wrapped.

        Asserted on the registry rather than on the wiring: a wrapper that
        forgot ``set_held`` would still relabel the button and still look right.
        """
        from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity
        from softae.gui.rig_claim import RigRunClaim

        activity = RigActivity()
        owner = "ht:plate"
        activity.acquire(owner, None)
        executor = self._wire(tab, manager, claim=RigRunClaim(activity, owner))

        thread = self._hold_on_a_thread(executor)
        try:
            assert _pump_until(
                qapp, lambda: "Continue plate" in tab._btn_pause.text()
            )
            # Suspended: manual control is permitted, but the rig is not idle.
            assert activity.conflicts(PURGE_INSTRUMENTS) is None
            assert activity.suspended_conflict(PURGE_INSTRUMENTS) == owner
            executor.resume()
            assert _pump_until(qapp, lambda: not thread.is_alive())
        finally:
            thread.join(timeout=10.0)
            if tab._hold_box is not None:
                tab._hold_box.close()

        assert activity.conflicts(PURGE_INSTRUMENTS) == owner   # driving again

    def test_record_formulation_called_for_selected_channels(self, tab: ExperimentBuilderTab):
        tab._data_store = MagicMock()
        tab._data_store.start_run.return_value = "run123"
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)
        tab._form_table.cellWidget(1, 0).setChecked(True)
        tab._form_table.setItem(0, 2, QTableWidgetItem("9.0"))
        tab._form_table.setItem(0, 3, QTableWidgetItem("1.0"))
        tab._form_table.setItem(1, 2, QTableWidgetItem("2.0"))
        tab._form_table.setItem(1, 3, QTableWidgetItem("3.0"))
        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True), \
                patch.object(tab, "_occupancy_gate", return_value=True):
            tab._on_start()
        assert tab._data_store.record_formulation.call_count == 2


# ── Single-use well occupancy (guard + record on deposit) ──────────────────


class TestOccupancy:
    """HT casts record occupancy; a pre-run guard warns on re-cast."""

    def _wf(self, mode="deposition"):
        wf = MagicMock()
        wf.metadata = {"mode": mode}
        return wf

    def test_gate_no_conflict_uses_current_board(self, tab, tmp_path):
        from softae.core.data_store import DataStore
        ds = DataStore(tmp_path / "proj")
        tab._data_store = ds
        with patch.object(tab, "_selected_channels", return_value=[0, 1]):
            assert tab._occupancy_gate(self._wf()) is True
        assert tab._active_board_id == 0
        ds.close()

    def test_gate_measure_only_skips(self, tab, tmp_path):
        from softae.core.data_store import DataStore
        ds = DataStore(tmp_path / "proj")
        ds.record_electrode_cast(0, 1)  # would conflict if it were checked
        tab._data_store = ds
        with patch.object(tab, "_selected_channels", return_value=[0]):
            assert tab._occupancy_gate(self._wf(mode="measure_only")) is True
        ds.close()

    def test_gate_conflict_fresh_advances_board(self, tab, tmp_path, monkeypatch):
        from softae.core.data_store import DataStore
        from softae.gui.widgets import occupancy_guard
        from softae.gui.widgets.occupancy_guard import BoardReplacedDecision
        ds = DataStore(tmp_path / "proj")
        ds.record_electrode_cast(0, 2)  # channel 2 already used
        tab._data_store = ds
        monkeypatch.setattr(
            occupancy_guard, "prompt_board_replaced",
            lambda *a, **k: BoardReplacedDecision.FRESH,
        )
        with patch.object(tab, "_selected_channels", return_value=[0, 1]):  # ch 1,2
            assert tab._occupancy_gate(self._wf()) is True
        assert tab._active_board_id == 1  # advanced to a fresh board
        # Durable: the swap survives even if the run is aborted before a deposit.
        assert ds.current_board_id() == 1
        ds.close()

    def test_gate_conflict_cast_anyway_keeps_same_board(self, tab, tmp_path, monkeypatch):
        from softae.core.data_store import DataStore
        from softae.gui.widgets import occupancy_guard
        from softae.gui.widgets.occupancy_guard import BoardReplacedDecision
        ds = DataStore(tmp_path / "proj")
        ds.record_electrode_cast(0, 2)
        tab._data_store = ds
        monkeypatch.setattr(
            occupancy_guard, "prompt_board_replaced",
            lambda *a, **k: BoardReplacedDecision.CAST_ANYWAY,
        )
        with patch.object(tab, "_selected_channels", return_value=[0, 1]):
            assert tab._occupancy_gate(self._wf()) is True
        assert tab._active_board_id == 0  # same board — deliberate re-cast
        ds.close()

    def test_gate_conflict_cancel_aborts(self, tab, tmp_path, monkeypatch):
        from softae.core.data_store import DataStore
        from softae.gui.widgets import occupancy_guard
        from softae.gui.widgets.occupancy_guard import BoardReplacedDecision
        ds = DataStore(tmp_path / "proj")
        ds.record_electrode_cast(0, 1)
        tab._data_store = ds
        monkeypatch.setattr(
            occupancy_guard, "prompt_board_replaced",
            lambda *a, **k: BoardReplacedDecision.CANCEL,
        )
        with patch.object(tab, "_selected_channels", return_value=[0]):  # ch 1
            assert tab._occupancy_gate(self._wf()) is False
        ds.close()

    def test_deposit_step_records_occupancy(self, tab, tmp_path):
        from softae.core.data_store import DataStore
        ds = DataStore(tmp_path / "proj")
        tab._data_store = ds
        tab._active_board_id = 0
        tab._ui_step_complete("deposit_ch3", 0, 10, None, 0.0)
        assert ds.occupied_electrodes(0) == {3}
        ds.close()

    def test_non_deposit_steps_record_nothing(self, tab, tmp_path):
        from softae.core.data_store import DataStore
        ds = DataStore(tmp_path / "proj")
        tab._data_store = ds
        tab._active_board_id = 0
        tab._ui_step_complete("precondition_ch3", 0, 10, None, 0.0)
        tab._ui_step_complete("startup_flush", 1, 10, None, 0.0)
        assert ds.occupied_electrodes(0) == set()
        ds.close()


# ── UI signal slots ───────────────────────────────────────────────────────


class TestUISignalSlots:
    """Thread-safe signal → UI update slot behaviour."""

    @pytest.mark.parametrize(
        "step_name,expected_channel,expected_display",
        [
            ("deposit__iter2", "2", "deposit"),
            ("measure_eis_ch12", "12", "measure_eis_ch12"),
            ("startup_flush", "", "startup_flush"),
        ],
    )
    def test_parse_step_name_patterns(
        self,
        tab: ExperimentBuilderTab,
        step_name: str,
        expected_channel: str,
        expected_display: str,
    ):
        channel, display = tab._parse_step_name(step_name)
        assert channel == expected_channel
        assert display == expected_display

    def test_step_start_updates_progress(self, tab: ExperimentBuilderTab):
        tab._progress.setRange(0, 10)
        tab._ui_step_start("test_step", 3, 10)
        assert tab._progress.value() == 3
        assert "test_step" in tab._lbl_status.text()

    def test_step_complete(self, tab: ExperimentBuilderTab):
        """Step complete updates table row and _results list."""
        tab._ui_step_complete("deposit__iter2", 5, 10, "mock_result")
        assert tab._results_table.rowCount() == 1
        assert tab._results_table.item(0, 2).text() == "✓"
        assert tab._results[0]["status"] == "ok"

    def test_step_error(self, tab: ExperimentBuilderTab):
        """Step error updates table row and _results list."""
        tab._ui_step_error("deposit__iter0", 2, 10, "timeout exceeded")
        assert tab._results_table.item(0, 2).text() == "✗"
        assert tab._results[0]["status"] == "error"

    @pytest.mark.parametrize("executor_state,exit_code,expected_text", [
        (None, 0, "Completed"),
        (ExecutorState.ERROR, 1, "Failed"),
        (ExecutorState.ABORTED, 1, "Aborted"),
    ])
    def test_workflow_done(
        self, tab, executor_state, exit_code, expected_text
    ):
        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True):
            tab._on_start()
        if executor_state is not None:
            tab._executor._state = executor_state
        tab._ui_workflow_done(exit_code)
        assert tab._btn_start.isEnabled()
        assert expected_text in tab._lbl_status.text()

    def test_non_iter_step_complete(self, tab: ExperimentBuilderTab):
        tab._ui_step_complete("startup_flush", 0, 5, None)
        assert tab._results_table.item(0, 0).text() == ""
        assert tab._results_table.item(0, 1).text() == "startup_flush"


# ── CSV export ────────────────────────────────────────────────────────────


class TestCSVExport:

    def test_csv_export_writes_file(self, tab: ExperimentBuilderTab, tmp_path: Path):
        tab._results = [
            {"channel": "0", "step": "deposit", "status": "ok", "result": "done"},
            {"channel": "1", "step": "eis", "status": "error", "error": "timeout"},
        ]
        csv_path = str(tmp_path / "test_results.csv")
        with patch(
            "softae.gui.tabs.tab_experiment.QFileDialog.getSaveFileName",
            return_value=(csv_path, "CSV Files (*.csv)"),
        ), patch("softae.gui.tabs.tab_experiment.QMessageBox.information"):
            tab._on_save_csv()
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        assert len(rows) == 2
        assert rows[0]["channel"] == "0"
        assert rows[1]["status"] == "error"


# ── Channel selection helpers ─────────────────────────────────────────────


class TestChannelSelectionHelpers:

    @pytest.mark.parametrize("spec,max_ch,expected", [
        ("3", 16, [3]),
        ("5-8", 16, [5, 6, 7, 8]),
        ("1,3,5-8,12", 32, [1, 3, 5, 6, 7, 8, 12]),
        ("0,1,99", 16, [1]),       # out-of-range clamped
        ("", 16, []),
    ])
    def test_parse_channel_spec(self, spec, max_ch, expected):
        result = ExperimentBuilderTab._parse_channel_spec(spec, max_ch)
        assert result == expected

    def test_channel_entry_applies(self, tab: ExperimentBuilderTab):
        tab._on_deselect_all()
        tab._edit_channels.setText("1,3")
        tab._on_channel_entry()
        assert tab._form_table.cellWidget(0, 0).isChecked()
        if tab._form_table.rowCount() > 2:
            assert tab._form_table.cellWidget(2, 0).isChecked()
        if tab._form_table.rowCount() > 1:
            assert not tab._form_table.cellWidget(1, 0).isChecked()


# ── Config-driven pico routing ────────────────────────────────────────────


class TestPicoRouting:

    @pytest.mark.parametrize("channel,expected_pico", [
        (1, "pico1"),
        (16, "pico1"),
        (17, "pico2"),
        (32, "pico2"),
    ])
    def test_pico_for_channel(self, channel, expected_pico):
        from softae.config.loader import pico_for_channel
        assert pico_for_channel(channel) == expected_pico

    @pytest.mark.parametrize("channel", [0, 33])
    def test_pico_channel_out_of_range_raises(self, channel):
        from softae.config.loader import pico_for_channel
        with pytest.raises(ValueError):
            pico_for_channel(channel)


# ── Daemon shutdown seam (abort_run / cleanup) ─────────────────────────────


class _StubExecutor:
    """Executor stub whose abort() sets a threading.Event (the run's abort signal)."""

    def __init__(self) -> None:
        self.ev = threading.Event()

    def abort(self) -> None:
        self.ev.set()


def _spin_on_event(ev: threading.Event) -> threading.Thread:
    def run() -> None:
        while not ev.is_set():
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestDaemonShutdown:
    def test_experiment_cleanup_aborts_running_thread(self, tab: ExperimentBuilderTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        assert tab._run_thread.is_alive()
        tab.cleanup()
        assert ex.ev.is_set()
        assert not tab._run_thread.is_alive()

    def test_experiment_cleanup_is_noop_when_idle(self, tab: ExperimentBuilderTab):
        assert tab._executor is None
        assert tab._run_thread is None
        tab.cleanup()  # must not raise / block

    def test_experiment_cleanup_is_idempotent(self, tab: ExperimentBuilderTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        tab.cleanup()
        tab.cleanup()
        assert not tab._run_thread.is_alive()

    def test_experiment_abort_run_signals_without_joining(self, tab: ExperimentBuilderTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        tab.abort_run()
        assert ex.ev.is_set()
        assert tab._run_thread.is_alive()  # signal-only: not joined
        tab.cleanup()  # teardown join


def _select_recipe(tab, key: str) -> None:
    """Select a deposition recipe by its userData key in the HT recipe dropdown."""
    combo = tab._combo_deposit_recipe
    idx = combo.findData(key)
    assert idx >= 0, f"recipe '{key}' not offered"
    combo.setCurrentIndex(idx)


class TestRecipeEngineOptIn:
    """The engine's single-drop recipe (electrode + per-channel volumes injected)."""

    @pytest.fixture(autouse=True)
    def _disable_piezo(self):
        with patch(
            "softae.gui.tabs.tab_experiment.piezo_config",
            return_value={"enabled": False, "liquid_events": {"enabled": False}},
        ):
            yield

    @staticmethod
    def _select_ch1(tab):
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)  # channel 1

    def test_baseline_is_single_drop_engine(self, tab: ExperimentBuilderTab):
        # The fixture pins single_drop; every deposition run is the engine now.
        self._select_ch1(tab)
        assert tab._selected_recipe() == "single_drop"
        wf = tab._generate_workflow()
        names = [s.name for s in wf.setup]
        assert wf.metadata.get("source") == "deposition_engine"
        assert any(n.startswith("deposit_ch") for n in names)
        assert not any(n.startswith("deposit_p0_ch") for n in names)  # no legacy path

    def test_single_drop_recipe_uses_engine(self, tab: ExperimentBuilderTab):
        self._select_ch1(tab)
        _select_recipe(tab, "single_drop")
        wf = tab._generate_workflow()
        assert wf.metadata.get("source") == "deposition_engine"
        assert wf.metadata.get("recipe") == "single_drop"
        names = [s.name for s in wf.setup]
        assert wf.setup[0].name == "startup_flush"
        assert any(n.startswith("deposit_ch") for n in names)      # engine deposit step
        assert any(n.startswith("measure_eis_ch") for n in names)  # EIS reused (Full mode)
        assert not any(n.startswith("deposit_p0_ch") for n in names)
        assert not any(n.startswith("precondition_ch") for n in names)  # 1 phase
        assert wf.teardown and wf.teardown[0].name == "final_flush"

    def test_engine_injects_electrode_and_per_channel_volumes(self, tab: ExperimentBuilderTab):
        self._select_ch1(tab)
        _select_recipe(tab, "single_drop")
        wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch"))
        assert "x" in dep.params and "y" in dep.params
        assert isinstance(dep.params.get("vols"), list) and len(dep.params["vols"]) == 3

    def test_engine_formulate_only_has_no_eis(self, tab: ExperimentBuilderTab):
        self._select_ch1(tab)
        _select_recipe(tab, "single_drop")
        tab._combo_mode.setCurrentIndex(2)  # Formulate-Only
        wf = tab._generate_workflow()
        assert not any(s.name.startswith("measure_eis") for s in wf.setup)

    def test_single_drop_uses_ui_rate_and_zeroes_deadvols(self, tab: ExperimentBuilderTab):
        self._select_ch1(tab)
        tab._spin_rate.setValue(321.0)
        _select_recipe(tab, "single_drop")
        wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch"))
        assert dep.params["disp_rate"] == 321.0      # flat rate = UI dispense rate
        assert dep.params["ids"] == [0, 1, 2]         # aligned to the 3-pump matrix
        assert dep.params["deadvols"] == [0.0, 0.0, 0.0]
        assert len(dep.params["vols"]) == 3
        assert "disp_rates" not in dep.params         # single-drop uses a flat rate


class TestDepositMethodSelector:
    """Process Studio → HT: the engine's deposit-phase method comes from the catalog."""

    @pytest.fixture(autouse=True)
    def _disable_piezo(self):
        with patch(
            "softae.gui.tabs.tab_experiment.piezo_config",
            return_value={"enabled": False, "liquid_events": {"enabled": False}},
        ):
            yield

    @staticmethod
    def _select_ch1(tab):
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)

    def test_selector_lists_deposit_methods_and_defaults(self, tab: ExperimentBuilderTab):
        names = [tab._combo_deposit_method.itemData(i)
                 for i in range(tab._combo_deposit_method.count())]
        assert "single_drop_simul" in names
        # star_mix / startup_flush are not electrode-deposit methods (no x/y/vols).
        assert "star_mix" not in names and "startup_flush_full" not in names
        assert tab._selected_deposit_method() == "single_drop_simul"

    def test_selector_enabled_for_engine_recipe(self, tab: ExperimentBuilderTab):
        # Every recipe is an engine recipe now → the deposit-method selector is
        # always usable.
        assert tab._combo_deposit_method.isEnabled() is True
        _select_recipe(tab, "two_phase")
        assert tab._combo_deposit_method.isEnabled() is True

    def test_maturity_readout_populated(self, tab: ExperimentBuilderTab):
        # single_drop_simul is 'tested' → warning readout (below validated).
        assert tab._lbl_deposit_maturity.text()

    def test_engine_uses_selected_method(self, tab: ExperimentBuilderTab):
        self._select_ch1(tab)
        _select_recipe(tab, "two_phase")
        # Force the selection explicitly (default is already single_drop_simul).
        idx = tab._combo_deposit_method.findData("single_drop_simul")
        tab._combo_deposit_method.setCurrentIndex(idx)
        wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch"))
        # The chosen method's instrument/method drive the deposit step.
        assert dep.instrument == "liquid_handler"
        assert dep.method == "single_drop_simul"


class TestTwoPhaseCast:
    """The two-phase deposition recipe: precondition_flush → single_drop per channel."""

    @pytest.fixture(autouse=True)
    def _disable_piezo(self):
        with patch(
            "softae.gui.tabs.tab_experiment.piezo_config",
            return_value={"enabled": False, "liquid_events": {"enabled": False}},
        ):
            yield

    @staticmethod
    def _select_ch1_with_volumes(tab, v0=10.0, v1=30.0, v2=0.0):
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)  # channel 1
        tab._form_table.setItem(0, 2, QTableWidgetItem(str(v0)))
        tab._form_table.setItem(0, 3, QTableWidgetItem(str(v1)))
        tab._form_table.setItem(0, 4, QTableWidgetItem(str(v2)))

    def test_two_phase_selectable(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        _select_recipe(tab, "two_phase")
        assert tab._selected_recipe() == "two_phase"
        wf = tab._generate_workflow()
        assert wf.metadata.get("source") == "deposition_engine"
        assert wf.metadata.get("recipe") == "two_phase"

    def test_recipe_emits_precondition_then_deposit(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        _select_recipe(tab, "two_phase")
        wf = tab._generate_workflow()
        assert wf.metadata.get("source") == "deposition_engine"
        assert wf.metadata.get("recipe") == "two_phase"
        names = [s.name for s in wf.setup]
        # Campaign start flush, then precondition → deposit → EIS for channel 1.
        assert names[0] == "startup_flush"
        assert names.index("precondition_ch1") < names.index("deposit_ch1")
        assert any(n.startswith("measure_eis_ch") for n in names)  # Full mode
        assert wf.teardown and wf.teardown[0].name == "final_flush"

    def test_deposit_uses_per_pump_proportional_rates(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab, 10.0, 30.0, 0.0)  # total 40 µL
        tab._spin_rate.setValue(100.0)      # dispense rate → split 25/75/0
        _select_recipe(tab, "two_phase")
        # Pin the settle config so the derived wait is deterministic regardless of
        # the live [dropcast] section.
        with patch(
            "softae.gui.tabs.tab_experiment.dropcast_config",
            return_value={"settle_factor": 2.0, "settle_base_s": 0.0},
        ):
            wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch"))
        assert dep.params["disp_rates"] == [
            pytest.approx(25.0), pytest.approx(75.0), pytest.approx(0.0)]
        assert dep.params["vols"] == [pytest.approx(10.0), pytest.approx(30.0), pytest.approx(0.0)]
        assert dep.params["deadvols"] == [0.0, 0.0, 0.0]
        # Settling wait derived: 40/100 min = 0.4 min = 24 s × settle_factor 2 = 48 s.
        assert dep.params["elution_wait_s"] == pytest.approx(48.0)

    def test_precondition_uses_flush_rate_split_and_factor(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab, 10.0, 30.0, 0.0)
        tab._spin_flush_rate.setValue(500.0)   # line flush rate → split 125/375/0
        tab._spin_flush_factor.setValue(3.0)
        _select_recipe(tab, "two_phase")
        wf = tab._generate_workflow()
        pre = next(s for s in wf.setup if s.name.startswith("precondition_ch"))
        assert pre.method == "precondition_flush"
        assert pre.params["rate_list"] == [
            pytest.approx(125.0), pytest.approx(375.0), pytest.approx(0.0)]
        assert pre.params["vol_list"] == [pytest.approx(10.0), pytest.approx(30.0), pytest.approx(0.0)]
        assert pre.params["flush_factor"] == 3.0

    def test_start_flush_vector_feeds_startup(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        tab._edit_start_flush.setText("10, 20, 30")
        tab._spin_flush_rate.setValue(400.0)
        _select_recipe(tab, "two_phase")
        wf = tab._generate_workflow()
        start = wf.setup[0]
        assert start.name == "startup_flush"
        assert start.params["disp_vols"] == [pytest.approx(10.0), pytest.approx(20.0), pytest.approx(30.0)]
        assert start.params["disp_rate"] == pytest.approx(400.0)

    def test_start_flush_single_value_broadcasts(self, tab: ExperimentBuilderTab):
        tab._edit_start_flush.setText("50")
        assert tab._start_flush_volumes() == [50.0, 50.0, 50.0]

    def test_start_flush_empty_is_zeros(self, tab: ExperimentBuilderTab):
        tab._edit_start_flush.setText("")
        assert tab._start_flush_volumes() == [0.0, 0.0, 0.0]

    def test_formulate_only_has_no_eis(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        _select_recipe(tab, "two_phase")
        tab._combo_mode.setCurrentIndex(2)  # Formulate-Only
        wf = tab._generate_workflow()
        assert not any(s.name.startswith("measure_eis") for s in wf.setup)
        assert any(s.name.startswith("precondition_ch") for s in wf.setup)
        assert any(s.name.startswith("deposit_ch") for s in wf.setup)

    def test_preview_notes_engine_recipe(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        _select_recipe(tab, "two_phase")
        tab._on_generate_workflow()
        text = tab._txt_preview.toPlainText()
        assert "Deposition engine" in text and "two_phase" in text

    def test_dispense_controls_init_from_config(self, tab: ExperimentBuilderTab):
        # Controls initialise from the live [dropcast] config section (assert the
        # wiring, not hardcoded numbers, so lab-tuned config values don't break this).
        from softae.config.loader import dropcast_config

        dc = dropcast_config()
        assert tab._spin_flush_rate.value() == pytest.approx(dc["line_flush_rate_uL_min"])
        assert tab._spin_flush_factor.value() == pytest.approx(dc["flush_factor"])
        expected_start = [pytest.approx(v) for v in dc["start_flush_uL"][: len(tab.PUMP_IDS)]]
        assert tab._start_flush_volumes() == expected_start

    def test_settle_factor_from_config_drives_wait(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab, 10.0, 30.0, 0.0)  # total 40 µL
        tab._spin_rate.setValue(100.0)  # 0.4 min duration
        _select_recipe(tab, "two_phase")
        with patch(
            "softae.gui.tabs.tab_experiment.dropcast_config",
            return_value={"settle_factor": 1.0, "settle_base_s": 0.0},
        ):
            wf = tab._generate_workflow()
        dep = next(s for s in wf.setup if s.name.startswith("deposit_ch"))
        # 0.4 min = 24 s × settle_factor 1.0 = 24 s (vs 48 s at the default 2.0).
        assert dep.params["elution_wait_s"] == pytest.approx(24.0)

    def test_preview_shows_recipe_maturity(self, tab: ExperimentBuilderTab):
        self._select_ch1_with_volumes(tab)
        _select_recipe(tab, "two_phase")
        tab._on_generate_workflow()
        assert "recipe maturity:" in tab._txt_preview.toPlainText()

    def test_engine_recipe_enables_deposit_selector(self, tab: ExperimentBuilderTab):
        # The deposit-method selector is usable for any (engine) recipe.
        _select_recipe(tab, "two_phase")
        assert tab._combo_deposit_method.isEnabled() is True
        _select_recipe(tab, "single_drop")
        assert tab._combo_deposit_method.isEnabled() is True

    def test_default_recipe_config_selects_recipe(self, qapp, manager):
        # Config default_recipe drives the initial dropdown selection.
        full_cfg = {
            "dispense_rate_uL_min": 75.0, "line_flush_rate_uL_min": 500.0,
            "flush_factor": 3.0, "settle_factor": 2.0, "settle_base_s": 0.0,
            "start_flush_uL": [80.0, 80.0, 80.0], "default_recipe": "two_phase",
        }
        with patch(
            "softae.gui.tabs.tab_experiment.dropcast_config", return_value=full_cfg
        ):
            w = ExperimentBuilderTab(manager)
        try:
            assert w._selected_recipe() == "two_phase"
            assert w._combo_deposit_method.isEnabled() is True
        finally:
            w.close()


class TestOverflowGuard:
    """Run-start overflow guard: commanded volume vs the board's well capacity."""

    def test_under_capacity_proceeds_without_dialog(self, tab: ExperimentBuilderTab):
        tab._active_pcb_config = lambda: {"well_capacity_uL": 50.0}
        plan = [{"channel": 1, "total_commanded_uL": 30.0}]
        assert tab._overflow_check(plan) is True  # no dialog, proceeds

    def test_board_without_capacity_is_noop(self, tab: ExperimentBuilderTab):
        tab._active_pcb_config = lambda: {}  # board declares no cap → nothing enforced
        plan = [{"channel": 1, "total_commanded_uL": 9999.0}]
        assert tab._overflow_check(plan) is True

    def test_overflow_warns_and_respects_choice(self, tab: ExperimentBuilderTab, monkeypatch):
        from softae.gui.tabs import tab_experiment as te

        tab._active_pcb_config = lambda: {"well_capacity_uL": 20.0}
        plan = [{"channel": 3, "total_commanded_uL": 55.0}]  # 35 µL over

        seen = {}

        def fake_warning(parent, title, text, *a, **k):
            seen["text"] = text
            return te.QMessageBox.StandardButton.No

        monkeypatch.setattr(te.QMessageBox, "warning", staticmethod(fake_warning))
        assert tab._overflow_check(plan) is False           # cancel → do not run
        assert "channel 3" in seen["text"]
        assert "20.0 µL well capacity" in seen["text"]

        monkeypatch.setattr(
            te.QMessageBox, "warning",
            staticmethod(lambda *a, **k: te.QMessageBox.StandardButton.Yes),
        )
        assert tab._overflow_check(plan) is True             # proceed anyway


class TestRunSequence:
    """Run-sequence controls → RunPlan (anneal insertion, pointwise/batch scope)."""

    def test_default_is_pointwise_with_measure(self, tab: ExperimentBuilderTab):
        from softae.core.run_plan import PhaseKind, PhaseScope

        plan = tab._build_run_plan(formulate_only=False)
        assert [p.kind for p in plan.phases] == [PhaseKind.FORMULATE, PhaseKind.MEASURE]
        assert all(p.scope is PhaseScope.PER_SAMPLE for p in plan.phases)

    def test_formulate_only_omits_measure(self, tab: ExperimentBuilderTab):
        assert not tab._build_run_plan(formulate_only=True).has_measure

    def test_anneal_toggle_inserts_anneal_with_params(self, tab: ExperimentBuilderTab):
        tab._chk_anneal.setChecked(True)
        tab._spin_anneal_temp.setValue(120.0)
        tab._spin_anneal_hold.setValue(10.0)
        plan = tab._build_run_plan(formulate_only=False)
        assert plan.has_anneal
        anneal = next(p for p in plan.phases if p.kind.name == "ANNEAL")
        assert anneal.anneal_params["target_temp_C"] == 120.0
        assert anneal.anneal_params["hold_time_s"] == 600.0  # 10 min → 600 s

    def test_batch_scope_defers_measurement(self, tab: ExperimentBuilderTab):
        tab._combo_measure_scope.setCurrentIndex(1)  # Batch
        assert tab._build_run_plan(formulate_only=False).defers_measurement

    def test_sequence_preview_reflects_controls(self, tab: ExperimentBuilderTab):
        tab._chk_anneal.setChecked(True)
        tab._update_sequence_preview()
        text = tab._lbl_sequence.text()
        assert "Formulate" in text and "Anneal" in text and "Measure" in text


class TestTableCopyPaste:
    """Spreadsheet-style copy/paste on the formulation table."""

    @staticmethod
    def _set(tab, r, c, text):
        tab._form_table.setItem(r, c, QTableWidgetItem(text))

    def test_copy_selection_serialises_to_tsv(self, tab: ExperimentBuilderTab):
        from PySide6.QtWidgets import QApplication, QTableWidgetSelectionRange

        t = tab._form_table
        for (r, c), v in {(0, 2): "1", (0, 3): "2", (1, 2): "3", (1, 3): "4"}.items():
            self._set(tab, r, c, v)
        t.setRangeSelected(QTableWidgetSelectionRange(0, 2, 1, 3), True)
        t._copy_selection()
        assert QApplication.clipboard().text() == "1\t2\n3\t4"

    def test_paste_tsv_from_external_spreadsheet(self, tab: ExperimentBuilderTab):
        from PySide6.QtWidgets import QApplication

        t = tab._form_table
        QApplication.clipboard().setText("10\t20\t30\n40\t50\t60")
        t.setCurrentCell(0, 2)  # start at Pump 0
        t._paste_selection()
        assert [t.item(0, c).text() for c in (2, 3, 4)] == ["10", "20", "30"]
        assert [t.item(1, c).text() for c in (2, 3, 4)] == ["40", "50", "60"]

    def test_paste_skips_readonly_and_widget_cells(self, tab: ExperimentBuilderTab):
        from PySide6.QtWidgets import QApplication

        t = tab._form_table
        channel_before = t.item(0, 1).text()          # channel col is read-only
        QApplication.clipboard().setText("X\tY\tZ")     # → cols 1(channel), 2, 3
        t.setCurrentCell(0, 1)
        t._paste_selection()
        assert t.item(0, 1).text() == channel_before   # read-only untouched
        assert t.item(0, 2).text() == "Y"
        assert t.item(0, 3).text() == "Z"

    def test_paste_clamps_to_table_bounds(self, tab: ExperimentBuilderTab):
        from PySide6.QtWidgets import QApplication

        t = tab._form_table
        rows = t.rowCount()
        QApplication.clipboard().setText("\n".join(["9"] * (rows + 5)))
        t.setCurrentCell(rows - 1, 2)
        t._paste_selection()                            # only the last row is writable
        assert t.item(rows - 1, 2).text() == "9"
        assert t.rowCount() == rows                     # no rows added, no crash


class TestChannelShiftClick:
    """Shift-click a channel checkbox to fill the range from the anchor."""

    def test_plain_click_sets_anchor_only(self, tab: ExperimentBuilderTab, monkeypatch):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        tab._on_deselect_all()
        monkeypatch.setattr(QApplication, "keyboardModifiers",
                            staticmethod(lambda: Qt.KeyboardModifier.NoModifier))
        tab._on_channel_check_clicked(3)
        assert tab._chk_anchor_row == 3
        assert tab._selected_channels() == []  # handler alone checks nothing

    def test_shift_click_fills_range_with_anchor_state(self, tab: ExperimentBuilderTab, monkeypatch):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        assert tab._form_table.rowCount() >= 6
        tab._on_deselect_all()
        # A plain click on row 2 (which also checks it) becomes the anchor.
        tab._form_table.cellWidget(2, 0).setChecked(True)
        tab._on_channel_check_clicked(2)
        assert tab._chk_anchor_row == 2

        monkeypatch.setattr(QApplication, "keyboardModifiers",
                            staticmethod(lambda: Qt.KeyboardModifier.ShiftModifier))
        tab._on_channel_check_clicked(5)
        assert tab._selected_channels() == [2, 3, 4, 5]  # range filled to anchor's state
        assert tab._chk_anchor_row == 2                  # anchor unchanged (re-extendable)

    def test_repopulating_clears_anchor(self, tab: ExperimentBuilderTab):
        tab._chk_anchor_row = 4
        tab._populate_formulation_table(tab._form_table.rowCount())
        assert tab._chk_anchor_row is None


# ── "Fit All EIS": model choice and honest σ (P.14) ───────────────────────


def _eis(channel: int):
    """A minimal EISResult — the fit itself is stubbed in these tests."""
    import numpy as np

    from softae.analysis.eis_data import EISResult

    f = np.array([1.0, 10.0, 100.0])
    return EISResult(
        channel=channel,
        frequency=f,
        z_magnitude=np.array([3e3, 2e3, 1e3]),
        phase=np.array([-10.0, -20.0, -30.0]),
        z_real=np.array([3e3, 2e3, 1e3]),
        z_imag_neg=np.array([1e2, 2e2, 3e2]),
    )


def _fit(R0: float = 50.0, R1: float = 1000.0, model: str = "simpleSalt"):
    """A converged FitResult with known R0/R1, standing in for a real fit."""
    import numpy as np

    from softae.analysis.circuit_fitting import FitResult

    return FitResult(
        model_name=model,
        parameters=np.array([R0, 1e-7, 0.7, R1, 1e-10]),
        R0=R0, R1=R1, R0_guess=R0, R1_guess=R1,
        z_indices=[0, 3],
        success=True,
    )


class TestFitAllEIS:
    """The HT tab's post-run circuit fit: operator-chosen model, honest σ."""

    def test_the_default_model_is_named_not_the_first_key_so_reordering_cannot_change_it(
        self, qapp, manager
    ):
        """Reordering CIRCUIT_MODELS must not silently move what the rig fits with.

        The old code took ``list(CIRCUIT_MODELS)[0]``, so a dict edit made for
        unrelated reasons would change every σ the HT tab reports, with no test
        and nothing visible in the UI to say it had happened.
        """
        from softae.analysis import circuit_fitting

        reordered = {
            name: circuit_fitting.CIRCUIT_MODELS[name]
            for name in reversed(list(circuit_fitting.CIRCUIT_MODELS))
        }
        assert next(iter(reordered)) != "simpleSalt"  # the reorder actually moved it

        with patch.object(circuit_fitting, "CIRCUIT_MODELS", reordered):
            widget = ExperimentBuilderTab(manager)
        try:
            assert widget._combo_fit_model.currentData() == "simpleSalt"
        finally:
            widget.close()

    def test_the_model_combo_offers_every_circuit_model_with_its_description(
        self, tab: ExperimentBuilderTab
    ):
        """The operator picks from the whole registry, and sees what each model is.

        A bare model name ("flexSalt") does not tell an operator what circuit it
        fits; the description is the only thing in the UI that does.
        """
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        combo = tab._combo_fit_model
        offered = [combo.itemData(i) for i in range(combo.count())]
        assert offered == list(CIRCUIT_MODELS)
        for i, (name, info) in enumerate(CIRCUIT_MODELS.items()):
            assert name in combo.itemText(i)
            assert info["description"] in combo.itemText(i)

    def test_the_fit_uses_the_operator_selected_model(self, tab: ExperimentBuilderTab):
        """Selecting a model must actually change the circuit that is fitted.

        A selector wired to nothing is worse than no selector: it reports a model
        in the header that the numbers below it were not produced with.
        """
        tab._eis_results = [_eis(1)]
        idx = tab._combo_fit_model.findData("flexSalt")
        assert idx >= 0
        tab._combo_fit_model.setCurrentIndex(idx)

        with patch("softae.analysis.circuit_fitting.fit_circuit",
                   return_value=_fit()) as fake:
            tab._on_fit_all_eis()

        assert fake.call_args.args[1] == "flexSalt"
        assert "flexSalt" in tab._txt_fit_output.toPlainText()

    def test_sigma_is_dashed_when_no_thickness_was_recorded_rather_than_computed_from_a_placeholder(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """No thickness must read as "unknown", never as a number.

        The old path divided by a hard-coded t = 0.175 cm — ten times any
        drop-cast film — and printed the result as though it had been measured.
        A thickness whose ``deposit_area_mm2`` was never recorded is equally
        unusable (P.11): the quotient has no known denominator.
        """
        from softae.core.data_store import DataStore

        tab._eis_results = [_eis(3)]

        # (a) nothing recorded at all
        with patch("softae.analysis.circuit_fitting.fit_circuit", return_value=_fit()):
            tab._on_fit_all_eis()
        assert "σ=—" in tab._txt_fit_output.toPlainText()
        assert "S/cm" not in tab._txt_fit_output.toPlainText()

        # (b) a thickness exists, but the area it was divided by does not
        ds = DataStore(tmp_path / "proj")
        run_id = ds.start_run("ht")
        ds.record_formulation(run_id, 3, predicted_thickness_um=20.0,
                              deposit_area_mm2=None)
        tab._data_store = ds
        tab._run_id_by_channel[3] = run_id
        with patch("softae.analysis.circuit_fitting.fit_circuit", return_value=_fit()):
            tab._on_fit_all_eis()
        assert "σ=—" in tab._txt_fit_output.toPlainText()
        ds.close()

    def test_sigma_uses_the_boards_geometry_not_hard_coded_literals(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """σ must move when the board changes, because physically it does.

        L and w are properties of the selected PCB, which this tab already reads
        for the overflow check; freezing them at 0.2/0.2 made σ wrong for every
        board that is not the one those literals were typed for.
        """
        from softae.core.data_store import DataStore

        ds = DataStore(tmp_path / "proj")
        run_id = ds.start_run("ht")
        ds.record_formulation(run_id, 3, predicted_thickness_um=20.0,
                              deposit_area_mm2=4.0)
        tab._data_store = ds
        tab._run_id_by_channel[3] = run_id
        tab._eis_results = [_eis(3)]

        # L=0.5, w=0.1, t=20 µm=0.002 cm, R1=1000 Ω → σ = 0.5/(1000·0.002·0.1) = 2.5
        with patch.object(tab, "_active_pcb_config",
                          return_value={"electrode_L_cm": 0.5, "electrode_w_cm": 0.1}), \
             patch("softae.analysis.circuit_fitting.fit_circuit",
                   return_value=_fit(R1=1000.0)):
            tab._on_fit_all_eis()

        text = tab._txt_fit_output.toPlainText()
        assert "σ=2.500e+00 S/cm" in text
        assert "5.714e-03" not in text  # what the 0.2/0.175/0.2 literals gave
        ds.close()

    def test_r0_and_r1_still_report_without_any_geometry_because_they_are_measured(
        self, tab: ExperimentBuilderTab
    ):
        """Withholding σ must not withhold the resistances it was derived from.

        R0 and R1 come straight out of the fit and need no board geometry and no
        thickness; suppressing them alongside σ would hide the one part of the
        result that is unambiguously trustworthy.
        """
        tab._eis_results = [_eis(7)]
        with patch.object(tab, "_active_pcb_config", return_value={}), \
             patch("softae.analysis.circuit_fitting.fit_circuit",
                   return_value=_fit(R0=42.0, R1=1234.0)):
            tab._on_fit_all_eis()

        text = tab._txt_fit_output.toPlainText()
        assert "R0=42.0Ω" in text
        assert "R1=1234.0Ω" in text
        assert "σ=—" in text

    def test_the_run_id_survives_workflow_completion_so_a_post_run_fit_can_resolve_thickness(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """"Fit All EIS" is clicked *after* the run ends, when ``_ds_run_id`` is None.

        ``predicted_thickness_record`` is keyed on ``(run_id, channel)`` and
        ``EISResult`` carries no run_id, so unless the pairing is captured at
        capture time it is gone by the time the button is reachable. The active-run
        marker must still be cleared — other code reads ``is not None`` as "running".
        """
        from softae.core.data_store import DataStore

        ds = DataStore(tmp_path / "proj")
        run_id = ds.start_run("ht")
        ds.record_formulation(run_id, 3, predicted_thickness_um=20.0,
                              deposit_area_mm2=4.0)
        tab._data_store = ds
        tab._ds_run_id = run_id

        with patch.object(tab, "_raw_to_eis_result", return_value=_eis(3)):
            tab._ui_step_complete("measure_eis_ch3", 0, 1, object(), 0.0)
        tab._ui_workflow_done(0)

        assert tab._ds_run_id is None                 # the run is no longer active
        assert tab._run_id_by_channel == {3: run_id}  # but its identity survived
        assert tab._recorded_thickness_cm(3) == pytest.approx(20e-4)
        ds.close()

    def test_the_fit_output_does_not_overwrite_the_workflow_preview(
        self, tab: ExperimentBuilderTab
    ):
        """Fitting must not destroy the generated-workflow text beside it.

        Both used to write ``_txt_preview``, so clicking "Fit All EIS" wiped the
        step list the operator had just generated and was reading against.
        """
        tab._txt_preview.setPlainText("PREVIEW SENTINEL")
        tab._eis_results = [_eis(1)]

        with patch("softae.analysis.circuit_fitting.fit_circuit", return_value=_fit()):
            tab._on_fit_all_eis()

        assert tab._txt_preview.toPlainText() == "PREVIEW SENTINEL"
        assert "Circuit Fitting" in tab._txt_fit_output.toPlainText()


# ── Predicted dry thickness from the loaded stocks (P.12) ─────────────────


#: The shipped stock the payoff is measured against.
_PEO_LICL = "5wt% 20kDa PEO + 11.35mM LiCl (100:1 EO:Li)"
#: The manual fallback `t` in the Analysis tab, in µm. Not changed by P.12 —
#: it stays the seed for channels with no basis; P.12 gives HT channels a basis.
_MANUAL_SEED_UM = 1750.0


def _plan(channel: int = 1, **pumps: float) -> dict:
    """One `dispense_plan` entry with the per-pump commanded volumes given."""
    entry = {"channel": channel, "total_commanded_uL": sum(pumps.values())}
    for p in ExperimentBuilderTab.PUMP_IDS:
        entry[f"pump{p}_commanded_uL"] = float(pumps.get(f"p{p}", 0.0))
    return entry


def _loaded(tab, tmp_path, by_pump: dict):
    """Give *tab* a real project DB with *by_pump* declared as the loadout."""
    from softae.core.data_store import DataStore
    from softae.core.stock_assignment import PumpLoadout, save_loadout

    ds = DataStore(tmp_path / "proj")
    if by_pump:
        save_loadout(ds, PumpLoadout(dict(by_pump)))
    tab._data_store = ds
    return ds


def _select_pcb(tab, name: str) -> None:
    idx = tab._combo_pcb.findData(name)
    assert idx >= 0, f"{name} is not in the PCB combo"
    tab._combo_pcb.setCurrentIndex(idx)


class TestPredictedThickness:
    """An HT cast gets its dry thickness from the stocks actually on the pumps.

    Before this, HT channels recorded volumes and nothing else, so the Analysis
    tab's Fit-All fell back to a hand-typed t = 0.175 cm for every one of them —
    not a missing sigma but a **silently wrong** one, computed against a
    placeholder.
    """

    def test_the_real_peo_stock_predicts_a_film_two_decades_thinner_than_the_manual_seed(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """Characterization, against the *shipped* catalogue — this is the payoff.

        30 µL of 5wt% 20kDa PEO + LiCl over the 4-stripe board's 18.704 mm² well
        is a 1604 µm wet film that dries to ~74 µm at 4.64 % retained. The manual
        seed is 1750 µm: 23.5x too thick, and that factor is a property of the
        formulation, not a constant — the shipped catalogue alone spans 2.2x
        (50-50 glycerol-water) to 143x (2 wt% silica). That span is why no
        re-tuned constant would do, and why this test pins the number rather than
        a bound.

        It also guards the three inputs behind it: a regression in
        ``dep_fraction``, in the deposit area, or in the evaporation assumption
        each move this and nothing else in the suite.
        """
        ds = _loaded(tab, tmp_path, {0: _PEO_LICL})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")

        um, area_mm2, method = tab._predicted_cast(_plan(p0=30.0), tab._cast_stocks())

        assert method == "predicted"
        assert um == pytest.approx(74.36, rel=0.01)
        assert area_mm2 == pytest.approx(18.7038, rel=1e-3)
        assert _MANUAL_SEED_UM / um == pytest.approx(23.5, rel=0.05)
        ds.close()

    def test_an_ht_cast_records_a_predicted_thickness_from_its_loaded_stocks(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """End-to-end through the real run-start path, into the real DB.

        The unit-level helpers can be right while the persistence call site still
        drops the three new columns on the floor, which is exactly the failure
        that left campaign rows thickness-less before P7.6.
        """
        ds = _loaded(tab, tmp_path, {0: _PEO_LICL})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")
        tab._on_deselect_all()
        tab._form_table.cellWidget(0, 0).setChecked(True)
        tab._form_table.setItem(0, 2, QTableWidgetItem("30.0"))

        with patch.object(tab, "_run_workflow_thread"), \
                patch.object(tab, "_verify_head_position", return_value=True), \
                patch.object(tab, "_occupancy_gate", return_value=True):
            tab._on_start()

        record = ds.predicted_thickness_record(tab._executor._run_id, 1)
        assert record is not None
        assert record.method == "predicted"
        assert record.area_mm2 == pytest.approx(18.7038, rel=1e-3)
        assert 0 < record.um < 200          # a drop-cast film, not the 1750 seed
        ds.close()

    def test_the_wet_dispensed_thickness_never_seeds_the_analysis_t(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """The recorded t is the *dried* film, not volume / area.

        The wet number is what the deposition panel calls ``dispensed`` and it
        says nothing about drying, so using it as sigma's denominator would
        over-state by 1/retained-fraction — reproducing, in the other direction,
        the very error this task removes. For this stock the two are 1604 µm and
        74 µm.
        """
        ds = _loaded(tab, tmp_path, {0: _PEO_LICL})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")

        um, _area, _method = tab._predicted_cast(_plan(p0=30.0), tab._cast_stocks())
        wet_um = 30.0 / 18.703786 * 1000.0
        assert wet_um == pytest.approx(1604, rel=0.01)      # the number not used
        assert um < wet_um / 20

        run_id = ds.start_run("ht")
        ds.record_formulation(run_id, 1, predicted_thickness_um=um,
                              deposit_area_mm2=18.703786, thickness_method="predicted")
        tab._run_id_by_channel[1] = run_id
        assert tab._recorded_thickness_cm(1) == pytest.approx(um * 1e-4)
        ds.close()

    def test_an_undeclared_loadout_records_unavailable_rather_than_a_guessed_thickness(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """The loadout is project-scoped, so a fresh project starts empty.

        That is the "nobody has said what is in the syringes" state, and no
        thickness can be attributed to it. ``'unavailable'`` rather than NULL: the
        twin *was* asked. The area survives, because the board declares one
        regardless of what is loaded.
        """
        ds = _loaded(tab, tmp_path, {})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")

        assert tab._cast_stocks() is None
        um, area_mm2, method = tab._predicted_cast(_plan(p0=30.0), tab._cast_stocks())
        assert um is None
        assert method == "unavailable"
        assert area_mm2 == pytest.approx(18.7038, rel=1e-3)
        ds.close()

    def test_a_stock_missing_from_the_catalog_refuses_rather_than_dropping_it_silently(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """``stocks_from_loadout`` *skips* an unknown stock — P.12 must not inherit it.

        Skipping is right for a composition search, which simply cannot solve for
        a stock it has no chemistry for. For a thickness it is not: the pump still
        dispenses, so the skip deletes part of the film and yields a confident,
        too-thin number with nothing in the row to say so. A partial answer here
        is worse than no answer, because only the second one is visible.
        """
        from softae.core.composition_axes import stocks_from_loadout
        from softae.core.stock_assignment import catalogs_from_data_root, load_loadout

        ds = _loaded(tab, tmp_path, {0: _PEO_LICL, 1: "Unobtainium suspension"})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")

        # The premise: the skip really happens one layer down.
        _chem, sol = catalogs_from_data_root()
        stocks, assignment = stocks_from_loadout(load_loadout(ds), sol)
        assert set(stocks) == {_PEO_LICL} and set(assignment.values()) == {0}

        context = tab._cast_stocks()
        um, area_mm2, method = tab._predicted_cast(_plan(p0=20.0, p1=10.0), context)
        assert um is None
        assert method == "unavailable"
        assert area_mm2 == pytest.approx(18.7038, rel=1e-3)

        # …but a channel that casts nothing from the unknown pump is unaffected:
        # the missing stock contributes no film, so nothing is being hidden.
        um_ok, _area, method_ok = tab._predicted_cast(_plan(p0=30.0), context)
        assert method_ok == "predicted" and um_ok == pytest.approx(74.36, rel=0.01)
        ds.close()

    def test_a_sessile_board_records_a_null_area_on_the_ht_path_too(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """``SoftAE_IDE_EIS`` declares ``cast_confinement = "sessile"``.

        No wells, so the wetted footprint is set by volume and contact angle — an
        *observation*, which nothing on the board predicts. NULL area, not the
        inter-electrode rectangle, and not the string ``'unavailable'``: P.11's
        read guard keys off that column, so a value there would hand the objective
        a thickness with an invented basis. The campaign path already behaves this
        way; the HT path must not diverge.
        """
        ds = _loaded(tab, tmp_path, {0: _PEO_LICL})
        _select_pcb(tab, "SoftAE_IDE_EIS")

        um, area_mm2, method = tab._predicted_cast(_plan(p0=30.0), tab._cast_stocks())
        assert area_mm2 is None
        assert um is None
        assert method == "unavailable"
        ds.close()

    def test_an_all_carrier_stock_reports_unavailable_rather_than_a_zero_thickness_sigma_divides_by(
        self, tab: ExperimentBuilderTab, tmp_path
    ):
        """The shipped ``Water`` stock has ``dep_fraction == 0``: it dries to nothing.

        The twin answers 0.0 µm, which is arithmetically correct and unusable —
        sigma = L/(R1*t*w) divides by t. Zero is not a thin film, it is the
        absence of one, so it must read as unavailable all the way through to the
        sigma column.
        """
        from softae.core.autonomous_wiring import simulate_cast
        from softae.core.stock_assignment import catalogs_from_data_root

        chem, sol = catalogs_from_data_root()
        water = sol.get("Water")
        assert water.dep_fraction(chem) == 0.0            # the premise
        bare = simulate_cast({"Water": 30.0}, {"Water": water}, chem,
                             pcb_name="SoftAE_EIS_4Stripe")
        assert bare.final_thickness_um == 0.0             # the twin does answer 0

        ds = _loaded(tab, tmp_path, {0: "Water"})
        _select_pcb(tab, "SoftAE_EIS_4Stripe")
        um, _area, method = tab._predicted_cast(_plan(p0=30.0), tab._cast_stocks())
        assert um is None
        assert method == "unavailable"

        run_id = ds.start_run("ht")
        ds.record_formulation(run_id, 1, predicted_thickness_um=um,
                              deposit_area_mm2=18.703786, thickness_method=method)
        tab._run_id_by_channel[1] = run_id
        assert tab._recorded_thickness_cm(1) is None      # so sigma cannot divide
        ds.close()

    def test_a_component_marked_particulate_but_with_no_dep_role_is_reported_not_silently_excluded(
        self, tab: ExperimentBuilderTab
    ):
        """``is_particulate`` and the deposit roles are separate axes.

        ``is_particulate`` drives molar-mass availability and anti-clog purge
        routing; the dry film is decided by ``role`` / ``counts_as_deposit``. A
        CSV row that marks a solid via ``is_particulate`` alone, leaving ``role``
        blank, contributes nothing to the thickness — too thin, the safe
        direction, but still wrong, and invisible in the resulting number. It is
        surfaced instead.
        """
        from softae.core.formulation import (
            Chemical,
            ChemicalCatalog,
            Solution,
            SolutionComponent,
        )

        cat = ChemicalCatalog()
        cat.add(Chemical("Nanosilica", density_g_per_mL=2.65,
                         molar_mass_g_per_mol=60.08, is_particulate=True))
        cat.add(Chemical("IPA", density_g_per_mL=0.786))
        blank_role = Solution("mislabelled", [
            SolutionComponent("Nanosilica", "", 1.0, "g"),      # role omitted
            SolutionComponent("IPA", "carrier", 9.0, "mL"),
        ])
        declared = Solution("correct", [
            SolutionComponent("Nanosilica", "dep", 1.0, "g"),
            SolutionComponent("IPA", "carrier", 9.0, "mL"),
        ])

        assert blank_role.dep_fraction(cat) == 0.0         # silently absent…
        assert tab._uncounted_particulates({"mislabelled": blank_role}, cat) == [
            ("mislabelled", "Nanosilica")]                 # …but not silently
        assert tab._uncounted_particulates({"correct": declared}, cat) == []

    def test_the_shipped_catalogue_has_no_uncounted_particulate(
        self, tab: ExperimentBuilderTab
    ):
        """The audit the spec asks for, run against the real data files.

        The only particulate shipped is ``Fumed silica``, and both silica stocks
        give it ``role = dep``. If that ever stops being true, every silica
        prediction quietly loses its entire solids loading.
        """
        from softae.core.stock_assignment import catalogs_from_data_root

        chem, sol = catalogs_from_data_root()
        stocks = {n: sol.get(n) for n in sol.list_names()}
        assert tab._uncounted_particulates(stocks, chem) == []
