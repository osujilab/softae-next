"""Tests for DepositionPanel / WellSketch and the standalone launcher helpers."""

from __future__ import annotations

import csv

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog

from softae.core.deposition import (
    DepositionSummary,
    WellGeometry,
    carrier_component_keys,
    simulate_plate_deposition,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
    compute_elution_volumes,
)
from softae.gui.deposition_app import load_catalogs
from softae.gui.widgets.deposition_panel import (
    _SUM_ERROR_STYLE,
    _SUM_OK_STYLE,
    _SUM_WARN_STYLE,
    DepositionPanel,
    WellSketch,
    build_deposition_csv_rows,
)


@pytest.fixture
def catalog():
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    cat.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    return cat


@pytest.fixture
def sol_catalog():
    """Dep-fraction-0.25 stock: target 20 uL -> 80 uL eluted (dep 20 / carrier 60)."""
    cat = SolutionCatalog()
    cat.add(Solution("s25", [
        SolutionComponent("Ethanol", "dep", 1.0, "mL"),
        SolutionComponent("Water", "carrier", 3.0, "mL"),
    ]))
    return cat


@pytest.fixture
def panel(qtbot, catalog, sol_catalog):
    p = DepositionPanel(catalog, sol_catalog)
    qtbot.addWidget(p)
    return p


def _rich_catalogs():
    """A carrier-only stock + three dep-bearing stocks spanning dep_fraction."""
    chem = ChemicalCatalog()
    chem.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    chem.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    sol = SolutionCatalog()
    # dep_fraction: carrier_only 0.0, half_dep 0.5, low_dep 0.1, mid_dep 0.3
    sol.add(Solution("carrier_only", [SolutionComponent("Water", "carrier", 2.0, "mL")]))
    sol.add(Solution("half_dep", [SolutionComponent("Ethanol", "dep", 1.0, "mL"),
                                  SolutionComponent("Water", "carrier", 1.0, "mL")]))
    sol.add(Solution("low_dep", [SolutionComponent("Ethanol", "dep", 0.2, "mL"),
                                 SolutionComponent("Water", "carrier", 1.8, "mL")]))
    sol.add(Solution("mid_dep", [SolutionComponent("Ethanol", "dep", 0.6, "mL"),
                                 SolutionComponent("Water", "carrier", 1.4, "mL")]))
    return chem, sol


@pytest.fixture
def rich_panel(qtbot):
    chem, sol = _rich_catalogs()
    p = DepositionPanel(chem, sol)
    qtbot.addWidget(p)
    return p


def _row_of(p: DepositionPanel, name: str) -> int:
    for r in range(p._table_stocks.rowCount()):
        if p._table_stocks.item(r, 2).text() == name:
            return r
    raise KeyError(name)


def _set_auto(p: DepositionPanel, name: str, on: bool) -> None:
    p._table_stocks.item(_row_of(p, name), 1).setCheckState(
        Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
    )


def _set_fraction(p: DepositionPanel, name: str, value: float) -> None:
    p._spin_fractions[_row_of(p, name)].setValue(value)


def _set_used(p: DepositionPanel, name: str, on: bool) -> None:
    p._table_stocks.item(_row_of(p, name), 0).setCheckState(
        Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
    )


def _parse_csv_sections(path) -> dict:
    """Split a sectioned deposition CSV into {name: {"header": [...], "rows": [...]}}."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return _parse_csv_sections_from_rows(rows)


def _parse_csv_sections_from_rows(rows) -> dict:
    """Section-split already-materialised CSV rows (list[list[str]])."""
    sections: dict = {}
    current = None
    header_pending = False
    for row in rows:
        if not row:
            current = None
            continue
        if row[0].startswith("# "):
            current = row[0][2:]
            sections[current] = {"header": None, "rows": []}
            header_pending = True
            continue
        if current is None:
            continue
        if header_pending:
            sections[current]["header"] = row
            header_pending = False
        else:
            sections[current]["rows"].append(row)
    return sections


def _auto_balance_all(panel: DepositionPanel) -> None:
    """Turn Auto ON for every checked row (explicit-first default opens Auto OFF).

    Restores the historical equal-split first-open behaviour these worked-example
    assertions were written against.
    """
    for row in range(panel._table_stocks.rowCount()):
        panel._table_stocks.item(row, 1).setCheckState(Qt.CheckState.Checked)


def _apply_worked_example(panel: DepositionPanel) -> None:
    """Spec §3 inputs: target 20 µL, Ø5 × 2 mm, n=2, fixed 40 µL/well, 95 % evap."""
    panel._spin_target.setValue(20.0)
    panel._spin_diameter.setValue(5.0)
    panel._spin_depth.setValue(2.0)
    panel._spin_n_wells.setValue(2)
    panel._combo_mode.setCurrentIndex(1)  # Fixed µL per well
    panel._spin_per_well.setValue(40.0)
    panel._spin_evap.setValue(95.0)
    _auto_balance_all(panel)  # equal-split dep share (was the old "auto" default)


# ── load_catalogs ──


class TestLoadCatalogs:
    def test_load_catalogs_none_config_returns_empty_catalogs_and_message(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        chem, sol, status = load_catalogs(None)
        assert len(chem) == 0
        assert len(sol) == 0
        assert "empty" in status

    def test_load_catalogs_missing_csvs_returns_empty_catalogs_and_message(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        chem, sol, status = load_catalogs({"paths": {"data_root": str(tmp_path)}})
        assert len(chem) == 0
        assert len(sol) == 0
        assert "empty" in status

    def test_load_catalogs_valid_csvs_returns_populated_catalogs(
        self, tmp_path, monkeypatch, catalog, sol_catalog
    ):
        monkeypatch.chdir(tmp_path)
        catalog.save_csv(tmp_path / "chemicals.csv")
        sol_catalog.save_csv(tmp_path / "solutions.csv")
        chem, sol, status = load_catalogs({"paths": {"data_root": str(tmp_path)}})
        assert chem.list_names() == catalog.list_names()
        assert sol.list_names() == sol_catalog.list_names()
        assert "2 chemicals" in status
        assert "1 solutions" in status

    def test_load_catalogs_partial_csvs_loads_present_file_only(
        self, tmp_path, monkeypatch, catalog
    ):
        monkeypatch.chdir(tmp_path)
        catalog.save_csv(tmp_path / "chemicals.csv")
        chem, sol, status = load_catalogs({"paths": {"data_root": str(tmp_path)}})
        assert len(chem) == 2
        assert len(sol) == 0
        assert "2 chemicals" in status
        assert "solutions empty" in status

    def test_load_catalogs_malformed_csv_reports_parse_error_and_empty(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        # solutions.csv with a wrong header (`role_type` instead of `role`)
        # -> load_csv raises KeyError on the required `role` column.
        (tmp_path / "solutions.csv").write_text(
            "solution_name,component_name,role_type,quantity,unit\n"
            "s25,Ethanol,dep,1.0,mL\n",
            encoding="utf-8",
        )
        chem, sol, status = load_catalogs({"paths": {"data_root": str(tmp_path)}})
        # never raised; malformed catalog degraded to empty
        assert len(sol) == 0
        # error surfaced, and NOT reported as the plain not-found message
        assert "could not be read" in status
        assert "solutions.csv" in status
        assert "starting empty" not in status

    def test_load_catalogs_falls_back_to_loader_data_root_when_dict_lacks_it(
        self, tmp_path, monkeypatch, catalog, sol_catalog
    ):
        import softae.gui.deposition_app as app

        catalog.save_csv(tmp_path / "chemicals.csv")
        sol_catalog.save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(app.cfg, "data_root", lambda: tmp_path)
        # config dict present but WITHOUT a paths.data_root → loader.data_root() used
        chem, sol, status = load_catalogs({"paths": {}})
        assert chem.list_names() == catalog.list_names()
        assert sol.list_names() == sol_catalog.list_names()
        assert "2 chemicals" in status


# ── panel construction ──


class TestPanelConstruction:
    def test_panel_constructs_with_synthetic_catalogs_without_io(
        self, qtbot, catalog, sol_catalog, monkeypatch
    ):
        import softae.config.loader as loader

        def _forbidden(*args, **kwargs):
            raise AssertionError("DepositionPanel must not touch the config loader")

        monkeypatch.setattr(loader, "load", _forbidden)
        p = DepositionPanel(catalog, sol_catalog)
        qtbot.addWidget(p)
        assert p._table_stocks.rowCount() == 1

    def test_panel_lists_solutions_checked_explicit_zero_fraction(self, panel):
        assert panel._table_stocks.rowCount() == 1
        assert panel._table_stocks.item(0, 2).text() == "s25"
        assert panel._table_stocks.item(0, 0).checkState() == Qt.CheckState.Checked
        # OVERRIDE: rows open Auto OFF with an editable spin at literal 0.00.
        assert panel._table_stocks.item(0, 1).checkState() == Qt.CheckState.Unchecked
        spin = panel._table_stocks.cellWidget(0, 3)
        assert spin.value() == 0.0
        assert spin.specialValueText() == ""
        assert spin.isEnabled()

    def test_panel_default_inputs_match_formulation_conventions(self, panel):
        assert panel._spin_target.value() == pytest.approx(20.0)
        assert panel._spin_target.minimum() == pytest.approx(0.1)
        assert panel._spin_target.maximum() == pytest.approx(1000.0)
        # whatever the evap default is, slider and spinbox must agree
        assert panel._slider_evap.value() / 2.0 == pytest.approx(panel._spin_evap.value())


# ── compute ──


class TestPanelCompute:
    def test_compute_worked_example_matches_hand_calc(self, panel):
        _apply_worked_example(panel)
        assert panel._table_wells.rowCount() == 2
        assert float(panel._table_wells.item(0, 2).text()) == pytest.approx(11.50)
        assert float(panel._table_wells.item(0, 3).text()) == pytest.approx(585.7, abs=0.05)

    def test_compute_worked_example_mass_balance_strip_shows_totals(self, panel):
        _apply_worked_example(panel)
        text = panel._lbl_balance.text()
        assert "Eluted 80.00" in text
        assert "Dispensed 80.00" in text
        assert "Undeposited 0.00" in text
        assert "Evaporated 57.00" in text
        assert "Final 23.00" in text

    def test_compute_overfilled_well_shows_overflow_warning(self, panel, qtbot):
        _apply_worked_example(panel)
        panel.show()
        assert panel._lbl_overflow.isVisible()
        assert panel._table_wells.item(0, 5).text() == "YES"

    def test_compute_slider_change_updates_final_thickness(self, panel):
        _apply_worked_example(panel)
        panel._slider_evap.setValue(200)  # 95 % -> 100 %
        assert panel._spin_evap.value() == pytest.approx(100.0)
        assert float(panel._table_wells.item(0, 2).text()) == pytest.approx(10.00)
        assert float(panel._table_wells.item(0, 3).text()) == pytest.approx(509.3, abs=0.05)

    def test_compute_equal_split_mode_divides_eluted_volume(self, panel):
        _apply_worked_example(panel)
        panel._combo_mode.setCurrentIndex(0)  # Equal split
        assert float(panel._table_wells.item(0, 1).text()) == pytest.approx(40.00)
        assert float(panel._table_wells.item(1, 1).text()) == pytest.approx(40.00)
        assert "Undeposited 0.00" in panel._lbl_balance.text()


# ── validation ──


class TestPanelValidation:
    def test_validation_dispense_exceeding_eluted_shows_error_without_raising(self, panel):
        _apply_worked_example(panel)
        panel._spin_per_well.setValue(45.0)  # 2 x 45 > 80 eluted
        assert "exceeds eluted volume" in panel._lbl_status.text()
        assert panel._table_wells.rowCount() == 0
        assert panel._lbl_balance.text() == "—"

    def test_validation_no_solution_checked_shows_error_without_raising(self, panel):
        _apply_worked_example(panel)
        panel._table_stocks.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        assert panel._lbl_status.text() == "Select at least one solution."
        assert panel._table_wells.rowCount() == 0

    def test_validation_error_state_recovers_when_input_valid_again(self, panel):
        _apply_worked_example(panel)
        panel._spin_per_well.setValue(45.0)
        assert panel._table_wells.rowCount() == 0
        panel._spin_per_well.setValue(35.0)
        assert panel._table_wells.rowCount() == 2
        assert "Undeposited 10.00" in panel._lbl_balance.text()
        assert panel._lbl_overflow.isHidden()

    def test_validation_error_does_not_emit_signal(self, panel, qtbot):
        _apply_worked_example(panel)
        with qtbot.assertNotEmitted(panel.deposition_computed):
            panel._spin_per_well.setValue(45.0)


# ── signal ──


class TestPanelSignal:
    def test_signal_emits_deposition_summary_on_valid_compute(self, panel, qtbot):
        _apply_worked_example(panel)
        panel._spin_evap.setValue(90.0)
        with qtbot.waitSignal(panel.deposition_computed, timeout=1000) as blocker:
            panel._spin_evap.setValue(95.0)
        summary = blocker.args[0]
        assert isinstance(summary, DepositionSummary)
        assert summary.total_final_uL == pytest.approx(23.0)

    def test_slider_and_spinbox_stay_in_sync_both_directions(self, panel):
        panel._slider_evap.setValue(120)
        assert panel._spin_evap.value() == pytest.approx(60.0)
        panel._spin_evap.setValue(72.5)
        assert panel._slider_evap.value() == 145


# ── WellSketch ──


class TestWellSketch:
    def test_sketch_set_state_stores_levels_and_schedules_update(self, qtbot):
        sketch = WellSketch()
        qtbot.addWidget(sketch)
        well = WellGeometry(5.0, 2.0)
        sketch.set_state(well, 1.02, 0.29)
        assert sketch._well is well
        assert sketch._wet_fill == pytest.approx(1.02)
        assert sketch._final_fill == pytest.approx(0.29)

    def test_sketch_paints_without_error_via_grab(self, qtbot):
        sketch = WellSketch()
        qtbot.addWidget(sketch)
        sketch.set_state(WellGeometry(5.0, 2.0), 1.02, 0.29)
        assert not sketch.grab().isNull()
        sketch.set_state(None, 0.0, 0.0)
        assert not sketch.grab().isNull()


# ── set_catalogs / live reload ──


def _two_stock_catalogs():
    chem = ChemicalCatalog()
    chem.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    chem.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    sol = SolutionCatalog()
    sol.add(Solution("a", [SolutionComponent("Ethanol", "dep", 1.0, "mL"),
                           SolutionComponent("Water", "carrier", 3.0, "mL")]))
    sol.add(Solution("b", [SolutionComponent("Ethanol", "dep", 1.0, "mL"),
                           SolutionComponent("Water", "carrier", 1.0, "mL")]))
    return chem, sol


def _three_stock_catalogs():
    chem = ChemicalCatalog()
    chem.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    chem.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    sol = SolutionCatalog()
    for nm in ("b", "c", "d"):  # note "b" survives from the 2-stock set
        sol.add(Solution(nm, [SolutionComponent("Ethanol", "dep", 1.0, "mL"),
                              SolutionComponent("Water", "carrier", 2.0, "mL")]))
    return chem, sol


class TestSetCatalogs:
    def test_populate_stocks_is_idempotent(self, panel):
        panel._populate_stocks()
        panel._populate_stocks()
        n = len(panel._sol_catalog.list_names())
        assert panel._table_stocks.rowCount() == n
        assert len(panel._spin_fractions) == panel._table_stocks.rowCount()

    def test_set_catalogs_repopulates_stock_table(self, qtbot):
        chemA, solA = _two_stock_catalogs()
        p = DepositionPanel(chemA, solA)
        qtbot.addWidget(p)
        chemB, solB = _three_stock_catalogs()
        p.set_catalogs(chemB, solB)
        names = [p._table_stocks.item(r, 2).text() for r in range(p._table_stocks.rowCount())]
        assert names == solB.list_names()

    def test_set_catalogs_preserves_check_and_fraction_for_surviving_names(self, qtbot):
        chemA, solA = _two_stock_catalogs()
        p = DepositionPanel(chemA, solA)
        qtbot.addWidget(p)
        # solA rows are alphabetical: row 0 == "a", row 1 == "b"; "b" survives.
        b_row = [r for r in range(p._table_stocks.rowCount())
                 if p._table_stocks.item(r, 2).text() == "b"][0]
        p._table_stocks.item(b_row, 0).setCheckState(Qt.CheckState.Unchecked)
        p._spin_fractions[b_row].setValue(0.5)

        chemB, solB = _three_stock_catalogs()
        p.set_catalogs(chemB, solB)
        new_b = [r for r in range(p._table_stocks.rowCount())
                 if p._table_stocks.item(r, 2).text() == "b"][0]
        assert p._table_stocks.item(new_b, 0).checkState() == Qt.CheckState.Unchecked
        assert p._spin_fractions[new_b].value() == pytest.approx(0.5)

    def test_set_catalogs_empty_solution_catalog_shows_error_without_raising(self, qtbot):
        chemA, solA = _two_stock_catalogs()
        p = DepositionPanel(chemA, solA)
        qtbot.addWidget(p)
        p.set_catalogs(chemA, SolutionCatalog())
        assert p._table_stocks.rowCount() == 0
        assert p._lbl_status.text() == "Select at least one solution."
        assert p._table_wells.rowCount() == 0

    def test_set_catalogs_triggers_recompute_and_signal(self, qtbot):
        chemA, solA = _two_stock_catalogs()
        p = DepositionPanel(chemA, solA)
        qtbot.addWidget(p)
        chemB, solB = _three_stock_catalogs()
        with qtbot.waitSignal(p.deposition_computed, timeout=1000):
            p.set_catalogs(chemB, solB)


# ── explicit fractions / Auto toggle ──


class TestExplicitFractions:
    def test_explicit_fractions_auto_off_includes_literal_zero(self, rich_panel):
        # Default is Auto OFF at 0.00 -> every checked row contributes a literal 0.0.
        fractions = rich_panel._explicit_fractions()
        assert fractions["half_dep"] == pytest.approx(0.0)
        assert "half_dep" in fractions  # not omitted despite value 0.0

    def test_explicit_fractions_auto_on_row_omitted(self, rich_panel):
        _set_auto(rich_panel, "half_dep", True)
        fractions = rich_panel._explicit_fractions()
        assert "half_dep" not in fractions
        assert "low_dep" in fractions

    def test_explicit_fractions_all_auto_returns_none(self, rich_panel):
        for name in ("carrier_only", "half_dep", "low_dep", "mid_dep"):
            _set_auto(rich_panel, name, True)
        assert rich_panel._explicit_fractions() is None

    def test_explicit_fractions_unchecked_row_excluded(self, rich_panel):
        _set_fraction(rich_panel, "half_dep", 0.4)
        rich_panel._table_stocks.item(_row_of(rich_panel, "half_dep"), 0).setCheckState(
            Qt.CheckState.Unchecked
        )
        assert "half_dep" not in (rich_panel._explicit_fractions() or {})


class TestAutoState:
    def test_populate_defaults_auto_off_and_spin_enabled(self, rich_panel):
        for row in range(rich_panel._table_stocks.rowCount()):
            assert rich_panel._table_stocks.item(row, 1).checkState() == Qt.CheckState.Unchecked
            spin = rich_panel._table_stocks.cellWidget(row, 3)
            assert spin.isEnabled()
            assert spin.value() == pytest.approx(0.0)

    def test_toggle_auto_off_enables_spin(self, rich_panel):
        row = _row_of(rich_panel, "half_dep")
        _set_auto(rich_panel, "half_dep", True)
        assert not rich_panel._spin_fractions[row].isEnabled()
        _set_auto(rich_panel, "half_dep", False)
        assert rich_panel._spin_fractions[row].isEnabled()
        assert not rich_panel._spin_fractions[row].isReadOnly()

    def test_auto_on_row_displays_resolved_share_after_recompute(self, rich_panel):
        # Enable Auto on all dep-bearing rows -> each of 3 resolves to ~1/3.
        for name in ("half_dep", "low_dep", "mid_dep"):
            _set_auto(rich_panel, name, True)
        row = _row_of(rich_panel, "half_dep")
        assert rich_panel._spin_fractions[row].value() == pytest.approx(1 / 3, abs=0.01)

    def test_set_catalogs_preserves_auto_state_for_surviving_names(self, rich_panel):
        _set_auto(rich_panel, "half_dep", True)
        _set_fraction(rich_panel, "low_dep", 0.5)  # low_dep stays Auto off
        chem, sol = _rich_catalogs()  # same names survive
        rich_panel.set_catalogs(chem, sol)
        assert rich_panel._table_stocks.item(
            _row_of(rich_panel, "half_dep"), 1
        ).checkState() == Qt.CheckState.Checked
        low_row = _row_of(rich_panel, "low_dep")
        assert rich_panel._table_stocks.item(low_row, 1).checkState() == Qt.CheckState.Unchecked
        assert rich_panel._spin_fractions[low_row].value() == pytest.approx(0.5)
        assert rich_panel._spin_fractions[low_row].isEnabled()


class TestAutoBalance:
    def test_auto_balance_all_sets_auto_on_for_checked_rows(self, rich_panel):
        # Uncheck one row; it must remain Auto off after balancing.
        low_row = _row_of(rich_panel, "low_dep")
        rich_panel._table_stocks.item(low_row, 0).setCheckState(Qt.CheckState.Unchecked)
        rich_panel._on_auto_balance()
        for row in range(rich_panel._table_stocks.rowCount()):
            checked = rich_panel._table_stocks.item(row, 0).checkState() == Qt.CheckState.Checked
            auto = rich_panel._table_stocks.item(row, 1).checkState() == Qt.CheckState.Checked
            assert auto == checked

    def test_auto_balance_all_triggers_single_recompute(self, rich_panel, qtbot):
        calls: list[object] = []
        rich_panel.deposition_computed.connect(calls.append)
        rich_panel._on_auto_balance()
        assert len(calls) == 1


class TestSumIndicator:
    def test_indicator_default_explicit_zero_amber_prompt(self, rich_panel):
        assert rich_panel._lbl_fraction_sum.text() == "Σ = 0.00 — set fractions or enable Auto"
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_WARN_STYLE

    def test_indicator_all_auto_green_sum_one(self, rich_panel):
        rich_panel._on_auto_balance()
        assert "Σ = 1.00 (3 auto-balanced)" == rich_panel._lbl_fraction_sum.text()
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_OK_STYLE

    def test_indicator_explicit_below_one_amber_short(self, rich_panel):
        _set_fraction(rich_panel, "half_dep", 0.3)
        _set_fraction(rich_panel, "low_dep", 0.4)
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_WARN_STYLE
        assert "Σ = 0.70 < 1" in rich_panel._lbl_fraction_sum.text()
        assert "short by 6.00 µL" in rich_panel._lbl_fraction_sum.text()

    def test_indicator_explicit_exceeds_one_with_auto_red(self, rich_panel):
        _set_auto(rich_panel, "low_dep", True)  # one Auto absorber remains
        _set_fraction(rich_panel, "half_dep", 0.8)
        _set_fraction(rich_panel, "mid_dep", 0.4)  # E = 1.2 with an Auto row present
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_ERROR_STYLE
        assert "Σ_explicit = 1.20" in rich_panel._lbl_fraction_sum.text()
        # The Auto row is clamped to 0 share -> elutes 0 µL.
        low_row = _row_of(rich_panel, "low_dep")
        assert rich_panel._table_stocks.item(low_row, 4).text() == "0.00"

    def test_indicator_carrier_only_excluded_and_flagged(self, rich_panel):
        _set_fraction(rich_panel, "half_dep", 0.6)
        _set_fraction(rich_panel, "low_dep", 0.4)  # E = 1.00 over dep-bearing
        _set_fraction(rich_panel, "carrier_only", 0.5)  # bulk-only, excluded from Σ
        text = rich_panel._lbl_fraction_sum.text()
        assert text.startswith("Σ = 1.00")
        assert text.endswith("(+1 carrier-only bulk share)")


class TestElutedBreakdown:
    def test_stock_output_cells_match_elution_result(self, rich_panel):
        rich_panel._on_auto_balance()
        elution = compute_elution_volumes(
            rich_panel._selected_solutions(),
            rich_panel._chem_catalog,
            rich_panel._spin_target.value(),
            rich_panel._explicit_fractions(),
        )
        for name in ("half_dep", "low_dep", "mid_dep"):
            row = _row_of(rich_panel, name)
            assert float(rich_panel._table_stocks.item(row, 4).text()) == pytest.approx(
                elution.per_solution[name], abs=0.01
            )
            assert float(rich_panel._table_stocks.item(row, 5).text()) == pytest.approx(
                elution.dep_vol_uL[name], abs=0.01
            )
            assert float(rich_panel._table_stocks.item(row, 6).text()) == pytest.approx(
                elution.carrier_vol_uL[name], abs=0.01
            )

    def test_seeded_spread_visible_in_eluted_column(self, rich_panel):
        rich_panel._on_auto_balance()
        eluted = [
            float(rich_panel._table_stocks.item(_row_of(rich_panel, n), 4).text())
            for n in ("half_dep", "low_dep", "mid_dep")
        ]
        assert min(eluted) < max(eluted)  # unequal eluted totals
        deps = [
            float(rich_panel._table_stocks.item(_row_of(rich_panel, n), 5).text())
            for n in ("half_dep", "low_dep", "mid_dep")
        ]
        assert deps[0] == pytest.approx(deps[1], abs=0.01)  # equal dep share
        assert deps[1] == pytest.approx(deps[2], abs=0.01)

    def test_unchecked_and_error_state_show_dash(self, rich_panel):
        rich_panel._on_auto_balance()
        low_row = _row_of(rich_panel, "low_dep")
        rich_panel._table_stocks.item(low_row, 0).setCheckState(Qt.CheckState.Unchecked)
        assert rich_panel._table_stocks.item(low_row, 4).text() == "—"
        # empty-catalog error path -> all output cells dashed, no exception
        rich_panel.set_catalogs(rich_panel._chem_catalog, SolutionCatalog())
        assert rich_panel._table_stocks.rowCount() == 0


class TestComponentBreakdown:
    def test_component_table_hidden_by_default(self, rich_panel):
        assert not rich_panel._table_components.isVisible()

    def test_toggle_shows_and_populates_component_table(self, rich_panel, qtbot):
        rich_panel.show()
        rich_panel._chk_show_components.setChecked(True)
        assert rich_panel._table_components.isVisible()
        # carrier_only(1) + 3 dep stocks x 2 components = 7 (solution, chemical) rows
        assert rich_panel._table_components.rowCount() == 7

    def test_component_role_classified_by_carrier_keys(self, rich_panel):
        rich_panel._chk_show_components.setChecked(True)
        roles = {}
        for r in range(rich_panel._table_components.rowCount()):
            sol = rich_panel._table_components.item(r, 0).text()
            chem = rich_panel._table_components.item(r, 1).text()
            roles[(sol, chem)] = rich_panel._table_components.item(r, 2).text()
        assert roles[("half_dep", "Ethanol")] == "dep"
        assert roles[("half_dep", "Water")] == "carrier"

    def test_component_eluted_matches_component_vol_uL(self, rich_panel):
        rich_panel._on_auto_balance()
        rich_panel._chk_show_components.setChecked(True)
        elution = compute_elution_volumes(
            rich_panel._selected_solutions(),
            rich_panel._chem_catalog,
            rich_panel._spin_target.value(),
            rich_panel._explicit_fractions(),
        )
        for r in range(rich_panel._table_components.rowCount()):
            key = (
                rich_panel._table_components.item(r, 0).text(),
                rich_panel._table_components.item(r, 1).text(),
            )
            assert float(rich_panel._table_components.item(r, 3).text()) == pytest.approx(
                elution.component_vol_uL[key], abs=0.01
            )


class TestStability:
    def test_recompute_not_reentered_on_programmatic_fill(self, rich_panel):
        rich_panel._on_auto_balance()  # ensure a valid cached result
        calls: list[object] = []
        rich_panel.deposition_computed.connect(calls.append)
        rich_panel._spin_target.setValue(30.0)  # exactly one user edit
        assert len(calls) == 1  # filling cells / auto-spin display did not re-enter

    def test_populate_stocks_idempotent_with_new_columns(self, rich_panel):
        rich_panel._populate_stocks()
        rich_panel._populate_stocks()
        n = len(rich_panel._sol_catalog.list_names())
        assert rich_panel._table_stocks.rowCount() == n
        assert len(rich_panel._spin_fractions) == rich_panel._table_stocks.rowCount()
        assert rich_panel._table_stocks.columnCount() == 7


class TestNormalize:
    def test_normalize_makes_explicit_sum_one_for_explicit_only(self, rich_panel):
        _set_used(rich_panel, "carrier_only", False)
        _set_used(rich_panel, "low_dep", False)
        _set_fraction(rich_panel, "half_dep", 0.30)
        _set_fraction(rich_panel, "mid_dep", 0.20)
        assert rich_panel._btn_normalize.isEnabled()
        rich_panel._btn_normalize.click()
        assert rich_panel._spin_fractions[_row_of(rich_panel, "half_dep")].value() == (
            pytest.approx(0.60)
        )
        assert rich_panel._spin_fractions[_row_of(rich_panel, "mid_dep")].value() == (
            pytest.approx(0.40)
        )
        assert rich_panel._lbl_fraction_sum.text() == "Σ = 1.00"
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_OK_STYLE

    def test_normalize_over_one_scales_down_to_one(self, rich_panel):
        _set_used(rich_panel, "carrier_only", False)
        _set_used(rich_panel, "low_dep", False)
        _set_fraction(rich_panel, "half_dep", 0.80)
        _set_fraction(rich_panel, "mid_dep", 0.60)
        rich_panel._btn_normalize.click()
        half = rich_panel._spin_fractions[_row_of(rich_panel, "half_dep")].value()
        mid = rich_panel._spin_fractions[_row_of(rich_panel, "mid_dep")].value()
        assert half < 0.80
        assert mid < 0.60
        assert half + mid == pytest.approx(1.00)
        assert rich_panel._lbl_fraction_sum.styleSheet() == _SUM_OK_STYLE

    def test_normalize_noop_disabled_when_already_sum_one(self, rich_panel):
        _set_used(rich_panel, "carrier_only", False)
        _set_used(rich_panel, "low_dep", False)
        _set_fraction(rich_panel, "half_dep", 0.60)
        _set_fraction(rich_panel, "mid_dep", 0.40)
        assert not rich_panel._btn_normalize.isEnabled()

    def test_normalize_disabled_when_no_explicit_positive(self, rich_panel):
        # first-open: every dep row is at 0.00
        assert not rich_panel._btn_normalize.isEnabled()
        rich_panel._on_auto_balance()  # all-Auto -> still nothing explicit
        assert not rich_panel._btn_normalize.isEnabled()

    def test_normalize_enabled_on_amber_short_and_overshoot(self, rich_panel):
        _set_fraction(rich_panel, "half_dep", 0.30)
        _set_fraction(rich_panel, "mid_dep", 0.20)  # Σ = 0.50 amber short
        assert rich_panel._btn_normalize.isEnabled()
        _set_fraction(rich_panel, "half_dep", 0.80)
        _set_fraction(rich_panel, "mid_dep", 0.60)  # Σ = 1.40 amber overshoot
        assert rich_panel._btn_normalize.isEnabled()

    def test_normalize_leaves_auto_rows_untouched(self, rich_panel):
        _set_auto(rich_panel, "low_dep", True)
        _set_fraction(rich_panel, "half_dep", 0.80)
        _set_fraction(rich_panel, "mid_dep", 0.60)  # red E>1 with an Auto row
        low_row = _row_of(rich_panel, "low_dep")
        before = rich_panel._spin_fractions[low_row].value()
        assert rich_panel._btn_normalize.isEnabled()
        rich_panel._btn_normalize.click()
        # Auto flag untouched; its spin value not rescaled by Normalize.
        assert rich_panel._table_stocks.item(low_row, 1).checkState() == Qt.CheckState.Checked
        assert rich_panel._spin_fractions[low_row].value() == pytest.approx(before)
        half = rich_panel._spin_fractions[_row_of(rich_panel, "half_dep")].value()
        mid = rich_panel._spin_fractions[_row_of(rich_panel, "mid_dep")].value()
        assert half + mid == pytest.approx(1.00)

    def test_normalize_rounding_residual_keeps_sum_exactly_one(self, rich_panel):
        _set_used(rich_panel, "carrier_only", False)
        _set_fraction(rich_panel, "half_dep", 0.40)
        _set_fraction(rich_panel, "low_dep", 0.40)
        _set_fraction(rich_panel, "mid_dep", 0.40)  # E = 1.20, three explicit rows
        rich_panel._btn_normalize.click()
        total = sum(
            rich_panel._spin_fractions[_row_of(rich_panel, n)].value()
            for n in ("half_dep", "low_dep", "mid_dep")
        )
        assert round(total, 2) == pytest.approx(1.00)

    def test_normalize_triggers_single_recompute(self, rich_panel):
        _set_fraction(rich_panel, "half_dep", 0.30)
        _set_fraction(rich_panel, "mid_dep", 0.20)
        calls: list[object] = []
        rich_panel.deposition_computed.connect(calls.append)
        rich_panel._btn_normalize.click()
        assert len(calls) == 1


class TestExportCsv:
    def test_export_button_disabled_until_valid_result(self, panel):
        _set_used(panel, "s25", False)  # no solution -> error state
        assert not panel._btn_export_csv.isEnabled()
        assert panel._last_summary is None
        _set_used(panel, "s25", True)  # valid recompute
        assert panel._btn_export_csv.isEnabled()
        assert panel._last_summary is not None

    def test_export_csv_writes_expected_rows_for_worked_example(
        self, panel, tmp_path, monkeypatch
    ):
        _apply_worked_example(panel)
        out = tmp_path / "out.csv"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
        )
        elution = panel._last_elution
        summary = panel._last_summary
        panel._btn_export_csv.click()
        assert out.exists()
        sections = _parse_csv_sections(out)
        assert set(sections) == {"CONFIG", "STOCKS", "MASS_BALANCE", "WELLS"}
        # per-well values match summary
        for i, w in enumerate(summary.wells):
            row = sections["WELLS"]["rows"][i]
            assert float(row[1]) == pytest.approx(w.dispensed_uL)
            assert float(row[6]) == pytest.approx(w.final_volume_uL)
        # per-stock values match elution
        stock_row = sections["STOCKS"]["rows"][0]
        name = stock_row[0]
        assert float(stock_row[5]) == pytest.approx(elution.per_solution[name])
        assert float(stock_row[6]) == pytest.approx(elution.dep_vol_uL[name])
        assert float(stock_row[7]) == pytest.approx(elution.carrier_vol_uL[name])

    def test_export_csv_disabled_on_error_state(self, panel, tmp_path, monkeypatch):
        _set_used(panel, "s25", False)  # error: no solution
        assert panel._last_summary is None
        assert not panel._btn_export_csv.isEnabled()
        out = tmp_path / "should_not_exist.csv"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
        )
        panel._on_export_csv()  # guarded no-op when cache is empty
        assert not out.exists()

    def test_export_csv_uses_cached_result(self, panel, tmp_path, monkeypatch):
        _apply_worked_example(panel)  # 2-well result cached
        # Swap the cache for a distinct 3-well result; the table still says 2.
        well = WellGeometry(panel._spin_diameter.value(), panel._spin_depth.value())
        solutions = panel._selected_solutions()
        alt_elution = compute_elution_volumes(
            solutions, panel._chem_catalog, panel._spin_target.value(),
            panel._explicit_fractions(),
        )
        alt_summary = simulate_plate_deposition(
            alt_elution, well, panel._spin_evap.value(), 3, None,
            carrier_keys=carrier_component_keys(solutions),
        )
        panel._last_elution = alt_elution
        panel._last_summary = alt_summary
        out = tmp_path / "cached.csv"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "")
        )
        panel._on_export_csv()
        sections = _parse_csv_sections(out)
        # 3 well rows prove the cache (not the 2-well table) was the source.
        assert len(sections["WELLS"]["rows"]) == 3

    def test_export_csv_write_error_shows_status_without_raising(
        self, panel, tmp_path, monkeypatch
    ):
        _apply_worked_example(panel)
        # A directory path makes open() raise OSError.
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path), "")
        )
        panel._on_export_csv()  # must not raise
        assert "CSV export failed" in panel._lbl_status.text()
        assert panel._lbl_status.styleSheet() == _SUM_ERROR_STYLE

    def test_export_csv_cancel_is_noop(self, panel, tmp_path, monkeypatch):
        _apply_worked_example(panel)
        before = panel._lbl_status.text()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        panel._on_export_csv()
        assert panel._lbl_status.text() == before
        assert list(tmp_path.iterdir()) == []


class TestBuildDepositionCsvRows:
    def test_build_csv_rows_has_all_four_sections(self, panel):
        _apply_worked_example(panel)
        rows = build_deposition_csv_rows(
            panel._last_elution, panel._last_summary, panel._export_config()
        )
        markers = {r[0] for r in rows if r}
        assert {"# CONFIG", "# STOCKS", "# MASS_BALANCE", "# WELLS"} <= markers

    def test_build_csv_rows_per_well_matches_summary(self, panel):
        _apply_worked_example(panel)
        summary = panel._last_summary
        sections = _parse_csv_sections_from_rows(
            build_deposition_csv_rows(panel._last_elution, summary, panel._export_config())
        )
        data = sections["WELLS"]["rows"]
        assert len(data) == len(summary.wells)
        for i, w in enumerate(summary.wells):
            assert float(data[i][1]) == pytest.approx(w.dispensed_uL)
            assert float(data[i][2]) == pytest.approx(w.dep_uL)
            assert float(data[i][6]) == pytest.approx(w.final_volume_uL)
            assert float(data[i][8]) == pytest.approx(w.final_thickness_um)

    def test_build_csv_rows_per_stock_matches_elution(self, panel):
        _apply_worked_example(panel)
        elution = panel._last_elution
        sections = _parse_csv_sections_from_rows(
            build_deposition_csv_rows(elution, panel._last_summary, panel._export_config())
        )
        row = sections["STOCKS"]["rows"][0]
        name = row[0]
        assert float(row[4]) == pytest.approx(elution.solution_fractions[name])
        assert float(row[5]) == pytest.approx(elution.per_solution[name])
        assert float(row[6]) == pytest.approx(elution.dep_vol_uL[name])
        assert float(row[7]) == pytest.approx(elution.carrier_vol_uL[name])


def test_toggle_all_stocks(panel):
    """Single button toggles every stock's Use box (Deposition tab)."""
    assert panel._stocks_all_checked()                    # default: all checked
    assert panel._btn_toggle_all_stocks.text() == "Deselect All"
    panel._on_toggle_all_stocks()                         # → deselect all
    assert not panel._stocks_all_checked()
    assert panel._btn_toggle_all_stocks.text() == "Select All"
    panel._on_toggle_all_stocks()                         # → select all
    assert panel._stocks_all_checked()
    assert panel._btn_toggle_all_stocks.text() == "Deselect All"


class TestOverflowSweep:
    """Range-sweep overflow map on the deposition digital twin."""

    def test_sweepable_axes_includes_deposition_and_stock(self, panel):
        axes = panel.sweepable_axes()
        assert "deposition_uL" in axes            # always sweepable
        assert "x_s25" in axes                    # manual mode → per-stock fraction

    def test_overflow_sweep_flags_high_deposition(self, panel):
        # Single stock s25 (dep_fraction 0.25): elution = 4 × deposition µL.
        # Default well 5 mm × 2 mm → capacity ≈ 39.27 µL.
        result = panel.overflow_sweep({"deposition_uL": (5.0, 20.0)}, steps=4)
        assert result.n_points == 4
        # deposition {5,10,15,20} → elution {20,40,60,80}; only 5 µL fits.
        assert result.n_overflow == 3
        worst_point, worst = result.worst
        assert worst_point["deposition_uL"] == pytest.approx(20.0)
        assert worst.total_uL == pytest.approx(80.0)
        assert worst.capacity_uL == pytest.approx(WellGeometry(5.0, 2.0).capacity_uL)

    def test_overflow_sweep_all_clear_for_small_deposition(self, panel):
        result = panel.overflow_sweep({"deposition_uL": (1.0, 5.0)}, steps=3)
        assert not result.any_overflow           # elution ≤ 20 µL, well ≈ 39 µL
