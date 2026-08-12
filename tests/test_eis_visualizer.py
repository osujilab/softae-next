"""Tests for EISVisualizerWidget and its supporting types.

Covers:
  * EISEntry construction
  * ListEISSource / DataStoreSource
  * PollableSource timer behaviour
  * EISVisualizerWidget instantiation and mode-switching
  * _OverviewPane grid/dirty-flag logic
  * _InspectionPane list population and selection preservation
  * _ConductivityPane rendering
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QApplication

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis_data import EISResult
from softae.gui.widgets.eis_visualizer_widget import (
    EISEntry,
    EISVisualizerWidget,
    EISVisualizerWindow,
    ListEISSource,
    DataStoreSource,
    PollableSource,
    _ConductivityPane,
    _InspectionPane,
    _OverviewPane,
)


# ---------------------------------------------------------------------------
# QApplication singleton
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_eis() -> EISResult:
    """Minimal synthetic EISResult (50 pts, RC-like)."""
    f = np.logspace(1, 5, 50)
    tau = 2 * np.pi * f * 1e-6
    z_real = 100 + 50 / (1 + tau**2)
    z_imag_neg = np.abs(50 * tau / (1 + tau**2))
    return EISResult.from_arrays(channel=1, f=f, z_real=z_real, z_imag_neg=z_imag_neg)


@pytest.fixture
def sample_fit() -> FitResult:
    return FitResult(
        model_name="simpleSalt",
        parameters=np.array([100.0, 1e-7, 0.7, 500.0, 1e-10]),
        R0=100.0,
        R1=500.0,
        R0_guess=100.0,
        R1_guess=500.0,
        z_indices=[0, 3],
        success=True,
    )


@pytest.fixture
def sample_entry(sample_eis, sample_fit) -> EISEntry:
    return EISEntry(
        label="Ch01 \u2014 run_test001",
        eis=sample_eis,
        fit=sample_fit,
        sigma=CellConstant.from_legacy(0.2, 0.175, 0.2).sigma(500.0),
        run_id="run_test001",
    )


# ---------------------------------------------------------------------------
# TestEISEntry
# ---------------------------------------------------------------------------


class TestEISEntry:
    def test_entry_sigma_with_fit(self, sample_entry):
        assert sample_entry.sigma is not None
        assert sample_entry.sigma > 0

    def test_entry_sigma_none_no_fit(self, sample_eis):
        e = EISEntry(label="x", eis=sample_eis, fit=None, sigma=None)
        assert e.sigma is None
        assert e.fit is None


# ---------------------------------------------------------------------------
# TestListEISSource
# ---------------------------------------------------------------------------


class TestListEISSource:
    def test_get_entries_returns_list(self, sample_entry):
        src = ListEISSource([sample_entry])
        assert len(src.get_entries()) == 1

    def test_get_entries_empty(self):
        src = ListEISSource([])
        assert len(src.get_entries()) == 0


# ---------------------------------------------------------------------------
# TestDataStoreSource
# ---------------------------------------------------------------------------


class TestDataStoreSource:
    def test_data_store_source_skips_missing_eis_path(
        self, tmp_path, sample_eis
    ):
        from softae.core.data_store import DataStore

        # Ensure raw_file_path is None so eis_file_path is NULL in DB
        sample_eis.raw_file_path = None
        with DataStore(tmp_path / "ds") as store:
            run_id = store.start_run(
                "test", mode="manual", campaign="test", quality="explore"
            )
            store.record_measurement(run_id, sample_eis)
            entries = DataStoreSource(store).get_entries()
        assert len(entries) == 0

    def test_data_store_source_loads_eis_file(self, tmp_path, sample_eis):
        from softae.core.data_store import DataStore

        # Save EIS file outside the DataStore directory so the path is
        # stored as absolute in the DB.
        eis_file = tmp_path / "sample_eis.txt"
        sample_eis.save(eis_file, study_name="test")
        sample_eis.raw_file_path = str(eis_file)

        with DataStore(tmp_path / "ds") as store:
            run_id = store.start_run(
                "test", mode="manual", campaign="test", quality="explore"
            )
            store.record_measurement(run_id, sample_eis)
            entries = DataStoreSource(store).get_entries()

        assert len(entries) == 1
        assert entries[0].eis.channel == 1


# ---------------------------------------------------------------------------
# TestWidgetInstantiation
# ---------------------------------------------------------------------------


class TestWidgetInstantiation:
    def test_mode_switch_to_inspection(self, qapp, sample_entry):
        widget = EISVisualizerWidget(ListEISSource([sample_entry]))
        widget._set_mode(1)
        assert widget._stack.currentIndex() == 1

    def test_mode_switch_to_conductivity(self, qapp, sample_entry):
        widget = EISVisualizerWidget(ListEISSource([sample_entry]))
        widget._set_mode(2)
        assert widget._stack.currentIndex() == 2

    def test_refresh_updates_entries(self, qapp, sample_entry):
        widget = EISVisualizerWidget(ListEISSource([]))
        assert len(widget._entries) == 0
        widget.set_source(ListEISSource([sample_entry]))
        widget.refresh()
        assert len(widget._entries) == 1


# ---------------------------------------------------------------------------
# TestOverviewPane
# ---------------------------------------------------------------------------


class TestOverviewPane:
    def test_grid_rebuild_does_not_rerender(self, qapp, sample_entry):
        pane = _OverviewPane()
        pane.refresh([sample_entry])
        # After refresh, dirty set must be empty (all entries rendered)
        assert len(pane._dirty) == 0
        # Changing columns must not mark anything dirty
        pane.set_columns(3)
        assert len(pane._dirty) == 0

    def test_dirty_flag_set_after_new_entry(self, qapp, sample_entry, sample_eis):
        pane = _OverviewPane()
        entries = [sample_entry]
        pane.refresh(entries)
        assert len(pane._dirty) == 0

        entry2 = EISEntry(
            label="Ch02 \u2014 run_test001",
            eis=sample_eis,
            fit=None,
            sigma=None,
            run_id="run_test001",
        )
        entries2 = [sample_entry, entry2]
        pane.refresh(entries2)
        # Dirty should be empty after render
        assert len(pane._dirty) == 0


# ---------------------------------------------------------------------------
# TestPollableSource
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestConductivityPane
# ---------------------------------------------------------------------------


class TestConductivityPane:
    def test_renders_without_exception(self, qapp, sample_entry):
        pane = _ConductivityPane(on_pick_jump=lambda i: None)
        pane.refresh([sample_entry])  # must not raise

    def test_no_fit_entry_renders_placeholder(self, qapp, sample_eis):
        entry = EISEntry(
            label="Ch01 \u2014 run_test",
            eis=sample_eis,
            fit=None,
            sigma=None,
        )
        pane = _ConductivityPane(on_pick_jump=lambda i: None)
        pane.refresh([entry])
        ax = pane._fig.axes[0]
        # Placeholder point plotted via ax.plot → at least one line
        assert len(ax.lines) > 0 or len(ax.collections) > 0


# ---------------------------------------------------------------------------
# TestInspectionPane
# ---------------------------------------------------------------------------


class TestInspectionPane:
    def test_list_populated_after_refresh(self, qapp, sample_entry):
        pane = _InspectionPane()
        pane.refresh([sample_entry])
        assert pane._list.count() == 1

    def test_selection_preserved_after_append(
        self, qapp, sample_entry, sample_eis
    ):
        pane = _InspectionPane()
        entry1 = sample_entry
        pane.refresh([entry1])
        pane._list.setCurrentRow(0)
        assert pane._list.currentRow() == 0

        entry2 = EISEntry(
            label="Ch02 \u2014 run_test001",
            eis=sample_eis,
            fit=None,
            sigma=None,
            run_id="run_test001",
        )
        pane.refresh([entry1, entry2])
        assert pane._list.currentRow() == 0


# ---------------------------------------------------------------------------
# TestEISVisualizerWindow
# ---------------------------------------------------------------------------


class TestEISVisualizerWindow:
    def test_window_creates_with_empty_source(self, qapp):
        win = EISVisualizerWindow(ListEISSource([]))
        assert win is not None

    def test_window_central_widget_is_viewer(self, qapp, sample_entry):
        win = EISVisualizerWindow(ListEISSource([sample_entry]))
        assert isinstance(win.centralWidget(), EISVisualizerWidget)

    def test_window_default_size(self, qapp):
        win = EISVisualizerWindow(ListEISSource([]))
        assert win.width() >= 800
        assert win.height() >= 600

    def test_window_title(self, qapp):
        win = EISVisualizerWindow(ListEISSource([]), title="Test Title")
        assert win.windowTitle() == "Test Title"

    def test_window_set_source_delegates(self, qapp, sample_entry):
        win = EISVisualizerWindow(ListEISSource([]))
        win.set_source(ListEISSource([sample_entry]))
        assert len(win._viewer._entries) == 1


# ---------------------------------------------------------------------------
# TestManualTabChannelRouting
# ---------------------------------------------------------------------------


class TestManualTabChannelRouting:
    """Verify that the Manual tab derives pico name from channel number."""

    @pytest.fixture
    def manual_tab(self, qapp):
        from unittest.mock import MagicMock
        from softae.gui.tabs.tab_manual import ManualControlTab

        mgr = MagicMock()
        mgr.get.side_effect = Exception("no instrument")
        tab = ManualControlTab(mgr)
        yield tab
        if tab._pv_worker is not None:
            tab._pv_worker.stop_worker()

    @pytest.mark.parametrize("channel,expected_pico", [
        (1, "pico1"), (16, "pico1"), (17, "pico2"), (32, "pico2"),
    ])
    def test_pico_label(self, manual_tab, channel, expected_pico):
        # The channel selector is a free-text QLineEdit (supports "2,4,5-10");
        # setText fires textChanged → _update_eis_pico_label synchronously.
        manual_tab._edit_eis_ch.setText(str(channel))
        assert manual_tab._lbl_eis_pico.text() == expected_pico


# ---------------------------------------------------------------------------
# TestAnalysisTabVisualizer
# ---------------------------------------------------------------------------


class TestAnalysisTabVisualizer:
    """Verify EISVisualizerWidget is embedded in the Analysis tab."""

    @pytest.fixture
    def analysis_tab(self, qapp):
        from unittest.mock import MagicMock
        from softae.gui.tabs.tab_analysis import AnalysisTab

        mgr = MagicMock()
        return AnalysisTab(mgr, data_store=None)

    def test_tab_has_two_sub_tabs(self, analysis_tab):
        """Analysis tab must expose at least two sub-tabs (Fit & Export + EIS Browser)."""
        assert analysis_tab._tab_widget.count() >= 2

    def test_vis_widget_present(self, analysis_tab):
        """An EISVisualizerWidget must be created during tab construction."""
        from softae.gui.widgets.eis_visualizer_widget import EISVisualizerWidget
        assert isinstance(analysis_tab._vis_widget, EISVisualizerWidget)

    def test_reload_without_datastore_shows_message(self, analysis_tab, qtbot):
        """Reload with no DataStore must not crash — it shows an info dialog."""
        from unittest.mock import patch
        from PySide6.QtWidgets import QMessageBox

        with patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok) as mock_info:
            analysis_tab._on_vis_reload()
            mock_info.assert_called_once()

    def test_popout_creates_window(self, analysis_tab, qtbot):
        """Pop-out creates an EISVisualizerWindow and stores a reference."""
        analysis_tab._on_vis_popout()
        assert len(analysis_tab._popout_windows) == 1
        from softae.gui.widgets.eis_visualizer_widget import EISVisualizerWindow
        assert isinstance(analysis_tab._popout_windows[0], EISVisualizerWindow)

    def test_reload_datastore_browser_action_sets_list_source(self, qapp, sample_eis):
        from softae.gui.tabs.tab_analysis import AnalysisTab

        mgr = MagicMock()
        tab = AnalysisTab(mgr, data_store=MagicMock())
        try:
            class _FakeDialog:
                action = "browser"

                def __init__(self, *_args, **_kwargs):
                    pass

                def exec(self):
                    return QDialog.DialogCode.Accepted

                def selected_rows(self):
                    return [{"run_id": "run_1", "channel": 1, "eis_file_path": "dummy.txt"}]

            entry = EISEntry(label="Ch01 — run_1", eis=sample_eis, fit=None, sigma=None, run_id="run_1")
            with patch("softae.gui.tabs.tab_analysis._DataStoreSelectionDialog", _FakeDialog):
                with patch.object(tab, "_entries_from_measurement_rows", return_value=([entry], 0)):
                    with patch("softae.gui.tabs.tab_analysis.QMessageBox.information"):
                        tab._on_vis_reload()

            entries = tab._vis_source.get_entries()
            assert len(entries) == 1
            assert entries[0].fit is None
            assert tab._loaded == []
        finally:
            tab.close()

    def test_reload_datastore_fit_export_imports_without_fitting(self, qapp, sample_eis):
        from softae.gui.tabs.tab_analysis import AnalysisTab

        mgr = MagicMock()
        tab = AnalysisTab(mgr, data_store=MagicMock())
        try:
            class _FakeDialog:
                action = "fit_export"

                def __init__(self, *_args, **_kwargs):
                    pass

                def exec(self):
                    return QDialog.DialogCode.Accepted

                def selected_rows(self):
                    return [{"run_id": "run_2", "channel": 1, "eis_file_path": "dummy.txt"}]

            entry = EISEntry(label="Ch01 — run_2", eis=sample_eis, fit=None, sigma=None, run_id="run_2")
            with patch("softae.gui.tabs.tab_analysis._DataStoreSelectionDialog", _FakeDialog):
                with patch.object(tab, "_entries_from_measurement_rows", return_value=([entry], 0)):
                    with patch("softae.gui.tabs.tab_analysis.QMessageBox.information"):
                        tab._on_vis_reload()

            assert len(tab._loaded) == 1
            assert len(tab._fits) == 0
            assert tab._results_table.rowCount() == 0
        finally:
            tab.close()
