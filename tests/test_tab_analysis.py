"""Tests for Fix 5 — AnalysisTab canvas.draw_idle() instead of draw().

Confirms:
  - _update_plots calls draw_idle(), not draw().
  - _on_clear_all calls draw_idle(), not draw().
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_analysis import AnalysisTab, _ArrhFitWorker, _FitAllWorker

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
    widget = AnalysisTab(manager)
    yield widget
    widget.close()


# ── Fix 5a: _update_plots uses draw_idle ─────────────────────────────────────


class TestUpdatePlotsUsesDrawIdle:
    def test_update_plots_calls_draw_idle_not_draw(self, tab):
        """_update_plots must call canvas.draw_idle(), never canvas.draw()."""
        if tab._canvas is None:
            pytest.skip("matplotlib not available")

        draw_calls = []
        draw_idle_calls = []
        tab._canvas.draw = lambda: draw_calls.append(True)
        tab._canvas.draw_idle = lambda: draw_idle_calls.append(True)

        # Inject a minimal fake EISResult so _update_plots has something to draw
        fake = MagicMock()
        fake.z_real = [1.0, 2.0]
        fake.z_imag_neg = [1.0, 2.0]
        fake.z_magnitude = [1414.0, 707.0]
        fake.frequency = [1000.0, 100.0]
        fake.channel = 1
        tab._loaded = [fake]

        tab._update_plots()

        assert not draw_calls, "_update_plots must NOT call canvas.draw()"
        assert draw_idle_calls, "_update_plots must call canvas.draw_idle()"

    def test_clear_all_calls_draw_idle_not_draw(self, tab):
        """_on_clear_all must call canvas.draw_idle(), never canvas.draw()."""
        if tab._canvas is None:
            pytest.skip("matplotlib not available")

        draw_calls = []
        draw_idle_calls = []
        tab._canvas.draw = lambda: draw_calls.append(True)
        tab._canvas.draw_idle = lambda: draw_idle_calls.append(True)

        # Ensure there is something to clear so the branch is taken
        fake = MagicMock()
        tab._loaded = [fake]
        tab._fits = [MagicMock()]

        tab._on_clear_all()

        assert not draw_calls, "_on_clear_all must NOT call canvas.draw()"
        assert draw_idle_calls, "_on_clear_all must call canvas.draw_idle()"


class TestBodeSplit:
    def test_bode_split_into_magnitude_over_components(self, tab):
        if tab._canvas is None:
            pytest.skip("matplotlib not available")
        import numpy as np

        from softae.analysis.eis_data import EISResult

        f = np.geomspace(1e4, 1.0, 15)
        tab._loaded = [EISResult.from_arrays(
            channel=1, f=f,
            z_real=np.full(15, 980.0), z_imag_neg=np.full(15, 170.0))]
        tab._update_plots()

        axes = tab._fig.axes
        assert len(axes) == 3  # Nyquist + |Z| + Z'/-Z''
        titles = [a.get_title() for a in axes]
        assert any("Nyquist" in t for t in titles)
        assert any("|Z|" in t for t in titles)          # dedicated magnitude plot
        assert any("Z′" in t or "Z'" in t for t in titles)  # real/imag plot
        # The magnitude plot carries the |Z| y-axis; the components plot doesn't.
        assert any(a.get_ylabel() == "|Z| (Ω)" for a in axes)


# ── Fix 5b: verify no regression in basic tab construction ───────────────────


class TestAnalysisTabConstruction:
    def test_tab_constructs_without_error(self, tab):
        assert tab is not None

    def test_initial_loaded_list_is_empty(self, tab):
        assert tab._loaded == []

    def test_initial_fits_list_is_empty(self, tab):
        assert tab._fits == []


# ── Filename parsers ──────────────────────────────────────────────────────────


class TestGuessTemperature:
    def test_new_format_T_RH(self):
        assert AnalysisTab._guess_temperature("eis_ch1_T35_RH60.txt") == 35.0

    def test_new_format_T_only(self):
        assert AnalysisTab._guess_temperature("eis_ch2_T120_RH80.txt") == 120.0

    def test_legacy_T_prefix(self):
        assert AnalysisTab._guess_temperature("sample_T45.txt") == 45.0

    def test_no_temperature_returns_none(self):
        assert AnalysisTab._guess_temperature("sample_RH60.txt") is None


class TestGuessRH:
    def test_standard_format(self):
        assert AnalysisTab._guess_rh("eis_ch1_T35_RH60.txt") == 60.0

    def test_lowercase_rh(self):
        assert AnalysisTab._guess_rh("sample_rh35.csv") == 35.0

    def test_rh_with_underscore(self):
        assert AnalysisTab._guess_rh("file_RH_80.txt") == 80.0

    def test_decimal_rh(self):
        assert AnalysisTab._guess_rh("file_RH52.5.txt") == 52.5

    def test_no_rh_returns_none(self):
        assert AnalysisTab._guess_rh("eis_ch1_T35.txt") is None


class TestNaturalSortKey:
    def test_ch10_after_ch9(self):
        paths = ["eis_ch10_T35.txt", "eis_ch2_T35.txt", "eis_ch1_T35.txt", "eis_ch9_T35.txt"]
        ordered = sorted(paths, key=AnalysisTab._natural_sort_key)
        assert ordered == [
            "eis_ch1_T35.txt",
            "eis_ch2_T35.txt",
            "eis_ch9_T35.txt",
            "eis_ch10_T35.txt",
        ]

    def test_full_paths_sorted_by_filename(self):
        paths = [
            r"C:\data\eis_ch10_T35.txt",
            r"C:\data\eis_ch2_T35.txt",
        ]
        ordered = sorted(paths, key=AnalysisTab._natural_sort_key)
        assert ordered[0].endswith("ch2_T35.txt")
        assert ordered[1].endswith("ch10_T35.txt")


# ── Conductivity-vs-sample aggregate plot ────────────────────────────────────


class _Fit:
    def __init__(self, R1: float, success: bool = True):
        self.R1 = R1
        self.success = success


def _add_fit_row(tab, channel: int, *, checked: bool = True, t: float = 0.175) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidgetItem

    from softae.gui.tabs.tab_analysis import (
        _RCOL_CHANNEL, _RCOL_T, _editable_item, _ro_item, _sample_id_item,
    )

    tab._results_table.blockSignals(True)
    row = tab._results_table.rowCount()
    tab._results_table.insertRow(row)
    chk = QTableWidgetItem()
    chk.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    chk.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    tab._results_table.setItem(row, 0, chk)
    tab._results_table.setItem(row, 1, _sample_id_item(row + 1))  # col 1 = Sample ID
    tab._results_table.setItem(row, _RCOL_CHANNEL, _ro_item(str(channel)))
    tab._results_table.setItem(row, _RCOL_T, _editable_item(f"{t:g}"))  # per-row thickness
    tab._results_table.blockSignals(False)


def _sigma_axis(tab):
    return tab._sigma_fig.axes[0]


class TestConductivityVsSamplePlot:
    def test_plots_one_point_per_checked_successful_fit(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(1000.0), _Fit(2000.0), _Fit(0.0, success=False)]
        for ch in (1, 2, 3):
            _add_fit_row(tab, ch)

        tab._update_sigma_plot()
        ax = _sigma_axis(tab)
        lines = ax.get_lines()
        assert lines, "expected a σ-vs-sample line"
        assert len(lines[0].get_xdata()) == 2  # only the two successful fits

    def test_unchecked_all_falls_back_to_all_rows(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(1000.0), _Fit(2000.0)]
        for ch in (1, 2):
            _add_fit_row(tab, ch, checked=False)

        tab._update_sigma_plot()
        assert len(_sigma_axis(tab).get_lines()[0].get_xdata()) == 2

    def test_no_successful_fits_shows_placeholder(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(0.0, success=False)]
        _add_fit_row(tab, 1)

        tab._update_sigma_plot()
        ax = _sigma_axis(tab)
        assert not ax.get_lines()
        assert ax.texts and "No conductivities" in ax.texts[0].get_text()

    def test_geometry_change_rescales_sigma(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(1000.0)]
        _add_fit_row(tab, 1)

        tab._spin_L.setValue(0.2)
        tab._spin_t.setValue(0.1)
        tab._spin_w.setValue(0.2)
        tab._update_sigma_plot()
        y1 = _sigma_axis(tab).get_lines()[0].get_ydata()[0]

        tab._spin_L.setValue(0.4)  # double L -> double σ (valueChanged also fires update)
        tab._update_sigma_plot()
        y2 = _sigma_axis(tab).get_lines()[0].get_ydata()[0]
        assert y2 == pytest.approx(2 * y1, rel=1e-6)


# ── Per-row thickness → conductivity ─────────────────────────────────────────


class TestPerRowThickness:
    def test_editing_thickness_recomputes_row_sigma(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_SIGMA, _RCOL_T

        tab._fits = [_Fit(1000.0)]
        _add_fit_row(tab, 1, t=0.175)
        s1 = tab._row_sigma(0)

        tab._results_table.item(0, _RCOL_T).setText("0.35")  # double thickness
        s2 = tab._row_sigma(0)
        assert s2 == pytest.approx(s1 / 2, rel=1e-9)  # σ ∝ 1/t

        # …and recomputing refreshes the displayed σ cell.
        tab._recompute_row_sigma(0)
        assert tab._results_table.item(0, _RCOL_SIGMA).text() not in ("", "—")

    def test_two_rows_same_fit_different_thickness_differ(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(1000.0), _Fit(1000.0)]  # identical fits
        _add_fit_row(tab, 3, t=0.1)
        _add_fit_row(tab, 3, t=0.2)  # same channel, different thickness

        tab._update_sigma_plot()
        ys = list(_sigma_axis(tab).get_lines()[0].get_ydata())
        assert ys[0] == pytest.approx(2 * ys[1], rel=1e-6)  # thinner → higher σ

    def test_apply_thickness_to_selected(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_T

        tab._fits = [_Fit(1000.0), _Fit(1000.0)]
        _add_fit_row(tab, 1, t=0.175)
        _add_fit_row(tab, 2, t=0.175)

        tab._spin_t.setValue(0.05)
        tab._results_table.selectRow(1)  # only the second row
        tab._on_apply_thickness_to_selected()

        assert tab._results_table.item(1, _RCOL_T).text() == "0.05"
        assert tab._results_table.item(0, _RCOL_T).text() == "0.175"  # untouched

    def test_thickness_column_is_editable(self, tab):
        from PySide6.QtCore import Qt

        from softae.gui.tabs.tab_analysis import _RCOL_T

        _add_fit_row(tab, 1)
        flags = tab._results_table.item(0, _RCOL_T).flags()
        assert flags & Qt.ItemFlag.ItemIsEditable


# ── Clipboard paste of a thickness column (F1) ───────────────────────────────


def _add_result_row(tab, *, channel: int = 1, r1: float = 1000.0, t: float = 0.175):
    """A full-width results row, as `_populate_results` builds one.

    Wider than `_add_fit_row` on purpose: the paste tests need the read-only R0/R1
    and σ cells present to prove a paste steps over them.
    """
    from softae.gui.tabs.tab_analysis import (
        _RCOL_ERROR, _RCOL_FILE, _RCOL_GATE, _RCOL_MODEL, _RCOL_R0, _RCOL_R1,
        _RCOL_SIGMA, _RCOL_STATUS, _ro_item,
    )

    _add_fit_row(tab, channel, t=t)
    row = tab._results_table.rowCount() - 1
    tab._results_table.blockSignals(True)
    for col, text in ((_RCOL_FILE, "f.txt"), (_RCOL_MODEL, "randles"),
                      (_RCOL_R0, "10.00"), (_RCOL_R1, f"{r1:.2f}"),
                      (_RCOL_SIGMA, "—"), (_RCOL_STATUS, "✓"),
                      (_RCOL_GATE, "pass"), (_RCOL_ERROR, "")):
        tab._results_table.setItem(row, col, _ro_item(text))
    tab._results_table.blockSignals(False)
    tab._recompute_row_sigma(row)
    return row


def _paste_into_results(tab, text: str, *, row: int = 0, col: int | None = None):
    from softae.gui.tabs.tab_analysis import _RCOL_T

    QApplication.clipboard().setText(text)
    tab._results_table.setCurrentCell(row, _RCOL_T if col is None else col)
    tab._results_table._paste_selection()


def _sigmas(tab):
    from softae.gui.tabs.tab_analysis import _RCOL_SIGMA

    return [tab._results_table.item(r, _RCOL_SIGMA).text()
            for r in range(tab._results_table.rowCount())]


class TestResultsTablePaste:
    def test_paste_writes_thickness_column_for_every_clipboard_row(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_T

        for _ in range(3):
            _add_result_row(tab)

        _paste_into_results(tab, "0.01\r\n0.02\r\n0.03\r\n")  # Excel's CRLF shape

        assert [tab._results_table.item(r, _RCOL_T).text() for r in range(3)] == [
            "0.01", "0.02", "0.03"]

    def test_paste_recomputes_sigma_for_every_pasted_row(self, tab):
        """TRAP 1 — the defect that would ship new t beside stale σ.

        Mutation-checked: with the `pasteCompleted` connection removed, this test
        fails (σ stays at the pre-paste value for all three rows).
        """
        tab._fits = []  # force `_row_r1` onto the R1 cell, not a live fit
        for _ in range(3):
            _add_result_row(tab, r1=1000.0, t=0.175)
        before = _sigmas(tab)
        assert all(s not in ("", "—") for s in before)

        _paste_into_results(tab, "0.35\n0.35\n0.35")  # doubled thickness

        after = _sigmas(tab)
        assert all(a != b for a, b in zip(after, before)), (
            "every pasted row's σ must be recomputed, not just the first")
        for a, b in zip(after, before):
            # rel 1e-4: the cells carry the displayed '%.4e', not full precision.
            assert float(a) == pytest.approx(float(b) / 2, rel=1e-4)  # σ ∝ 1/t

    def test_paste_refreshes_the_sigma_plot_once(self, tab):
        calls = []
        tab._update_sigma_plot = lambda: calls.append(True)
        for _ in range(4):
            _add_result_row(tab)

        _paste_into_results(tab, "0.01\n0.02\n0.03\n0.04")

        assert len(calls) == 1  # one batched refresh, not one per row

    def test_paste_steps_over_readonly_cells_preserving_column_alignment(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_R0, _RCOL_R1, _RCOL_T

        _add_result_row(tab)
        # Three columns wide starting at R0: R0 and R1 are read-only, t is not.
        _paste_into_results(tab, "999\t888\t0.02", col=_RCOL_R0)

        assert tab._results_table.item(0, _RCOL_R0).text() == "10.00"   # untouched
        assert tab._results_table.item(0, _RCOL_R1).text() == "1000.00"  # untouched
        assert tab._results_table.item(0, _RCOL_T).text() == "0.02"      # aligned

    def test_paste_cannot_write_the_checkbox_column(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_CHECK

        _add_result_row(tab)
        tab._results_table.item(0, _RCOL_CHECK).setCheckState(Qt.CheckState.Checked)

        _paste_into_results(tab, "x", col=_RCOL_CHECK)

        item = tab._results_table.item(0, _RCOL_CHECK)
        assert item.text() == ""
        assert item.checkState() == Qt.CheckState.Checked

    def test_paste_longer_than_the_table_clamps_without_adding_rows(self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_T

        for _ in range(2):
            _add_result_row(tab)

        _paste_into_results(tab, "\n".join(f"0.0{i}" for i in range(1, 8)))

        assert tab._results_table.rowCount() == 2
        assert [tab._results_table.item(r, _RCOL_T).text() for r in range(2)] == [
            "0.01", "0.02"]

    def test_paste_that_writes_nothing_is_reported_and_names_the_column(self, tab):
        """TRAP 2 — the everyday case: the cursor is not in the t column."""
        from softae.gui.tabs.tab_analysis import _RCOL_CHANNEL, _RCOL_T

        _add_result_row(tab)
        _paste_into_results(tab, "0.01", col=_RCOL_CHANNEL)

        assert tab._results_table.item(0, _RCOL_T).text() == "0.175"  # unchanged
        status = tab._lbl_db_status.text()
        assert "nothing" in status.lower()
        assert "t (cm)" in status

    def test_paste_does_not_relocate_itself_to_the_thickness_column(self, tab):
        """Refusing is the contract; silently moving the operator's paste is not."""
        from softae.gui.tabs.tab_analysis import _RCOL_FILE, _RCOL_T

        for _ in range(2):
            _add_result_row(tab)

        _paste_into_results(tab, "0.01\n0.02", col=_RCOL_FILE)

        assert [tab._results_table.item(r, _RCOL_T).text() for r in range(2)] == [
            "0.175", "0.175"]

    def test_paste_with_non_numeric_values_counts_them_and_leaves_them_visible(
            self, tab):
        from softae.gui.tabs.tab_analysis import _RCOL_SIGMA, _RCOL_T

        tab._fits = []
        for _ in range(3):
            _add_result_row(tab)

        _paste_into_results(tab, "t (cm)\n0.02\n")  # a spreadsheet header row

        status = tab._lbl_db_status.text()
        assert "2" in status                          # two rows written
        assert "1 not numeric" in status              # one of them unusable
        assert tab._results_table.item(0, _RCOL_T).text() == "t (cm)"  # left visible
        assert tab._results_table.item(0, _RCOL_SIGMA).text() == "—"
        assert tab._results_table.item(1, _RCOL_SIGMA).text() not in ("", "—")

    def test_clean_paste_reports_without_a_dialog(self, tab, monkeypatch):
        import softae.gui.tabs.tab_analysis as mod

        popped = []
        monkeypatch.setattr(mod.QMessageBox, "information",
                            lambda *a, **k: popped.append(a))
        _add_result_row(tab)

        _paste_into_results(tab, "0.02")

        assert popped == []
        assert "Pasted 1" in tab._lbl_db_status.text()


# ── DataStore selection dialog: Shift-click range (F2) ───────────────────────


class _FakeStore:
    """Minimal store surface `_DataStoreSelectionDialog` actually calls."""

    def __init__(self, n: int = 6):
        self._n = n

    def query_runs(self):
        return [{"run_id": "run1"}]

    def query_measurements(self, run_id=None, channel=None, limit=None,
                           descending=True):
        return [{"measurement_id": i, "timestamp": f"2026-08-21T00:0{i}",
                 "run_id": "run1", "channel": i + 1, "workflow_name": "wf",
                 "eis_file_path": f"eis_ch{i + 1}.txt"} for i in range(self._n)]

    def query_conditions(self, measurement_id=None, stage=None):
        return []


class TestDataStoreSelectionDialogRange:
    def _dialog(self, qapp, n=6):
        from softae.gui.tabs.tab_analysis import _DataStoreSelectionDialog

        return _DataStoreSelectionDialog(_FakeStore(n))

    def test_shift_click_range_fills_checkboxes_between_anchor_and_row(self, qapp):
        dlg = self._dialog(qapp)
        try:
            dlg._table.item(1, 0).setCheckState(Qt.CheckState.Checked)
            dlg._table._check_anchor_row = 1

            dlg._table._apply_check_range(1, 4)

            checked = [dlg._table.item(r, 0).checkState() == Qt.CheckState.Checked
                       for r in range(6)]
            assert checked == [False, True, True, True, True, False]
        finally:
            dlg.close()

    def test_selected_rows_returns_exactly_the_shift_filled_range(self, qapp):
        """`selected_rows()` reads checkbox state, so the range must reach it."""
        dlg = self._dialog(qapp)
        try:
            dlg._table.item(1, 0).setCheckState(Qt.CheckState.Checked)
            dlg._table._check_anchor_row = 1
            dlg._table._apply_check_range(1, 4)

            assert [r["channel"] for r in dlg.selected_rows()] == [2, 3, 4, 5]
        finally:
            dlg.close()

    def test_reload_rows_resets_the_range_anchor(self, qapp):
        """A filter change must not let a range be filled from stale rows.

        `_reset_check_anchor` hangs off the *model's* row signals, so the widget's
        own `blockSignals` cannot suppress it — this asserts that rather than
        arguing it.
        """
        dlg = self._dialog(qapp)
        try:
            dlg._table._check_anchor_row = 3
            dlg._reload_rows()
            assert dlg._table._check_anchor_row is None
        finally:
            dlg.close()

    def test_checkable_column_is_the_checkbox_column(self, qapp):
        dlg = self._dialog(qapp)
        try:
            assert dlg._table.checkable_column == 0
        finally:
            dlg.close()


# ── Excel-style sample IDs ───────────────────────────────────────────────────


class TestExcelColumnName:
    def test_single_and_double_letters(self):
        from softae.gui.tabs.tab_analysis import _excel_column_name

        cases = {1: "A", 26: "Z", 27: "AA", 28: "AB", 52: "AZ",
                 53: "BA", 702: "ZZ", 703: "AAA"}
        for n, expected in cases.items():
            assert _excel_column_name(n) == expected


class TestSampleIdColumn:
    def test_duplicate_channels_get_distinct_ids(self, tab):
        # Two measurements on the SAME channel must still be uniquely identified.
        _add_fit_row(tab, channel=3)
        _add_fit_row(tab, channel=3)
        assert tab._results_table.item(0, 1).text() == "A"
        assert tab._results_table.item(1, 1).text() == "B"
        # channel column (now col 2) is identical for both
        assert tab._results_table.item(0, 2).text() == "3"
        assert tab._results_table.item(1, 2).text() == "3"

    def test_sample_id_is_not_editable(self, tab):
        from PySide6.QtCore import Qt

        _add_fit_row(tab, channel=1)
        flags = tab._results_table.item(0, 1).flags()
        assert not (flags & Qt.ItemFlag.ItemIsEditable)

    def test_renumber_after_removal_keeps_ids_contiguous(self, tab):
        for ch in (1, 2, 3):
            _add_fit_row(tab, ch)
        tab._results_table.removeRow(1)  # drop "B"
        tab._renumber_sample_ids()
        assert [tab._results_table.item(r, 1).text() for r in range(2)] == ["A", "B"]

    def test_sigma_plot_labels_use_sample_ids(self, tab):
        if tab._sigma_canvas is None:
            pytest.skip("matplotlib not available")
        tab._fits = [_Fit(1000.0), _Fit(2000.0)]
        _add_fit_row(tab, channel=3)
        _add_fit_row(tab, channel=3)  # same channel — IDs disambiguate

        tab._update_sigma_plot()
        ax = _sigma_axis(tab)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["A", "B"]


# ── Fit-worker parenting (closeEvent sweep reachability) ─────────────────────


class TestFitWorkersParentedToTab:
    def test_fit_workers_parented_to_tab(self, tab):
        """Both fit workers are parented to the tab so findChildren(QThread) reaches them."""
        fit_all = _FitAllWorker([], "randles", 1.0, 1.0, 1.0, parent=tab)
        arrh = _ArrhFitWorker([], "randles", 1.0, 1.0, 1.0, parent=tab)

        assert fit_all.parent() is tab
        assert arrh.parent() is tab

        children = tab.findChildren(QThread)
        assert fit_all in children
        assert arrh in children


# ── P.11: the table's `t` obeys the same area guard as the campaign objective ─


class TestRecordedThicknessAreaGuard:
    """`_recorded_thickness_cm` is the *second* consumer of the twin's thickness.

    The campaign objective is the first. Both divide σ by the same number, and the
    number is a quotient whose denominator moved by 4.676× on 2026-08-07 without
    the row recording which side it fell on. Guarding only the campaign path would
    leave the Fit & Export table quietly using a thickness the campaign had just
    refused -- and the table's value is the one an operator exports and publishes.
    """

    def _prepare(self, tab, store):
        tab._data_store = store
        tab._run_id_by_channel[7] = "run1"

    def _record_store(self, *, um, area):
        from softae.core.data_store import PredictedThicknessRecord

        class _Store:
            def predicted_thickness_um(self, run_id, channel):
                return um

            def predicted_thickness_record(self, run_id, channel):
                if um is None:
                    return None
                return PredictedThicknessRecord(um=um, area_mm2=area,
                                                method="predicted")

        return _Store()

    def test_the_gui_thickness_path_applies_the_same_guard_as_the_campaign_path(
            self, tab):
        """A NULL area withholds the thickness here exactly as it does there.

        The two paths must not disagree about whether a row is usable: an operator
        comparing an exported σ against a campaign's objective would otherwise find
        one populated and the other blank for the same cast, with nothing to
        explain the difference.
        """
        self._prepare(tab, self._record_store(um=150.0, area=None))
        assert tab._recorded_thickness_cm(7) is None

    def test_a_thickness_with_its_area_still_converts_from_micrometres_to_centimetres(
            self, tab):
        """The guard must not disturb the unit conversion it wraps.

        The table works in cm because it feeds σ = L/(R·w·t); the twin records µm.
        A guard that returned the raw µm would be a 10 000× error in σ -- far worse
        than the 4.676× it was added to prevent.
        """
        self._prepare(tab, self._record_store(um=150.0, area=18.7038))
        assert tab._recorded_thickness_cm(7) == pytest.approx(150.0 * 1e-4)

    def test_a_store_predating_the_record_reader_keeps_the_old_behaviour(self, tab):
        """Reached by the same `getattr` fallback as the campaign path.

        A store with no `predicted_thickness_record` has no area concept at all, so
        there is nothing to guard against; refusing its thickness would break the
        table for stores that never had the ambiguity.
        """
        class _Old:
            def predicted_thickness_um(self, run_id, channel):
                return 150.0

        self._prepare(tab, _Old())
        assert tab._recorded_thickness_cm(7) == pytest.approx(150.0 * 1e-4)

    def test_an_unknown_channel_is_still_none_rather_than_an_error(self, tab):
        """The pre-existing None-on-anything-missing contract is unchanged.

        The caller reads `None` as "fall back to the manual default"; an exception
        escaping here would instead abort building the row.
        """
        self._prepare(tab, self._record_store(um=150.0, area=18.7038))
        assert tab._recorded_thickness_cm(99) is None
