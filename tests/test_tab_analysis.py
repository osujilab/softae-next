"""Tests for Fix 5 — AnalysisTab canvas.draw_idle() instead of draw().

Confirms:
  - _update_plots calls draw_idle(), not draw().
  - _on_clear_all calls draw_idle(), not draw().
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QThread
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
