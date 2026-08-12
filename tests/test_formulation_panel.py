"""Tests for FormulationPanel round-trip / canonical-save / change-signal (spec §2-3)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)

from softae.config import loader
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.gui.widgets import formulation_io as fio
from softae.gui.widgets.formulation_panel import INVALID_BG, FormulationPanel


def _synthetic_chem() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(
        Chemical("Water", "O", density_g_per_mL=1.001, molar_mass_g_per_mol=18.015,
                 notes="water", viscosity_mPa_s=1.0021, is_particulate=False)
    )
    cat.add(
        Chemical("Isopropanol", "CC(O)C", density_g_per_mL=0.786,
                 molar_mass_g_per_mol=60.096, notes="ipa", viscosity_mPa_s=2.0)
    )
    cat.add(
        Chemical("Fumed silica", "O=[Si]=O", density_g_per_mL=2.65,
                 molar_mass_g_per_mol=60.08, notes="np",
                 viscosity_mPa_s=None, is_particulate=True)
    )
    return cat


def _synthetic_sol() -> SolutionCatalog:
    cat = SolutionCatalog()
    cat.add(Solution("Silica solution", [
        SolutionComponent("Fumed silica", "dep", 1.0, "g", calc_mode="Mass-based"),
        SolutionComponent("Isopropanol", "carrier", 9.0, "mL", calc_mode="Mass-based"),
    ]))
    return cat


def _chem_row_for(panel: FormulationPanel, name: str) -> int:
    for row in range(panel._chem_table.rowCount()):
        item = panel._chem_table.item(row, 0)
        if item is not None and item.text() == name:
            return row
    raise AssertionError(f"chem row {name!r} not found")


def _select_solution(panel: FormulationPanel, name: str) -> None:
    """Select a solution in the checkable menu list by name (drives the comp table)."""
    for i in range(panel._list_solutions.count()):
        item = panel._list_solutions.item(i)
        if item is not None and item.text() == name:
            panel._list_solutions.setCurrentItem(item)
            panel._on_solution_selected(name)  # ensure comp table repopulates
            return
    raise AssertionError(f"solution {name!r} not found")


@pytest.fixture
def panel(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
    p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
    qtbot.addWidget(p)
    return p


class TestFieldRoundTrip:
    def test_build_chem_catalog_preserves_viscosity_and_particulate(self, panel):
        cat = panel._build_chem_catalog()
        assert cat.get("Fumed silica").is_particulate is True
        assert cat.get("Fumed silica").viscosity_mPa_s is None
        assert cat.get("Water").viscosity_mPa_s == pytest.approx(1.0021)
        assert cat.get("Water").is_particulate is False

    def test_build_solution_preserves_calc_mode(self, panel):
        _select_solution(panel, "Silica solution")
        sol = panel._build_current_solution()
        assert sol is not None
        modes = {c.chemical_name: c.calc_mode for c in sol.components}
        assert modes["Fumed silica"] == "Mass-based"
        assert modes["Isopropanol"] == "Mass-based"

    def test_save_load_round_trip_matches_core(self, panel, tmp_path):
        # Panel save → canonical data_root (== tmp_path via the fixture monkeypatch).
        panel._on_save_canonical()
        panel_chem = ChemicalCatalog.load_csv(tmp_path / "chemicals.csv")
        panel_sol = SolutionCatalog.load_csv(tmp_path / "solutions.csv")

        # Direct core round-trip of the same catalogs.
        core_dir = tmp_path / "core"
        core_dir.mkdir()
        _synthetic_chem().save_csv(core_dir / "chemicals.csv")
        _synthetic_sol().save_csv(core_dir / "solutions.csv")
        core_chem = ChemicalCatalog.load_csv(core_dir / "chemicals.csv")
        core_sol = SolutionCatalog.load_csv(core_dir / "solutions.csv")

        assert panel_chem.list_names() == core_chem.list_names()
        for name in core_chem.list_names():
            a, b = panel_chem.get(name), core_chem.get(name)
            assert (a.name, a.formula, a.density_g_per_mL, a.molar_mass_g_per_mol,
                    a.notes, a.viscosity_mPa_s, a.is_particulate) == (
                b.name, b.formula, b.density_g_per_mL, b.molar_mass_g_per_mol,
                b.notes, b.viscosity_mPa_s, b.is_particulate)

        def _tuples(comps):
            return [(c.chemical_name, c.role, c.quantity, c.unit, c.calc_mode) for c in comps]

        assert panel_sol.list_names() == core_sol.list_names()
        for name in core_sol.list_names():
            assert _tuples(panel_sol.get(name).components) == _tuples(core_sol.get(name).components)

    def test_blank_viscosity_cell_parses_to_none(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 4).setText("")
        assert panel._build_chem_catalog().get("Water").viscosity_mPa_s is None

    def test_invalid_numeric_cells_default_without_raising(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("abc")  # density
        panel._chem_table.item(row, 3).setText("abc")  # mw
        _select_solution(panel, "Silica solution")
        panel._comp_table.item(0, 2).setText("abc")  # quantity
        chem = panel._build_chem_catalog().get("Water")
        assert chem.density_g_per_mL == pytest.approx(1.0)
        assert chem.molar_mass_g_per_mol == pytest.approx(0.0)
        sol = panel._build_current_solution()
        assert sol.components[0].quantity == pytest.approx(0.0)


class TestCanonicalPersistence:
    def test_ctor_auto_loads_from_data_root_when_no_catalogs_given(
        self, qtbot, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)

        def _forbidden(*args, **kwargs):
            raise AssertionError("auto-load must not open a folder dialog")

        monkeypatch.setattr(QFileDialog, "getExistingDirectory", _forbidden)
        p = FormulationPanel()
        qtbot.addWidget(p)
        assert p._chem_table.rowCount() == 3
        assert p._list_solutions.count() == 1

    def test_ctor_uses_injected_catalogs_when_provided(self, qtbot, monkeypatch):
        called: list[int] = []
        monkeypatch.setattr(loader, "data_root", lambda: called.append(1) or "x")
        p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
        qtbot.addWidget(p)
        assert p._chem_table.rowCount() == 3
        assert called == []  # data_root never read on the injection path

    def test_ctor_missing_files_degrade_to_empty_without_raising(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)  # empty dir
        p = FormulationPanel()
        qtbot.addWidget(p)
        assert p._chem_table.rowCount() == 0
        assert p._list_solutions.count() == 0

    def test_save_canonical_writes_to_data_root_and_creates_dir(
        self, qtbot, monkeypatch, tmp_path
    ):
        target = tmp_path / "nested" / "data"  # does not exist yet
        monkeypatch.setattr(loader, "data_root", lambda: target)
        p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
        qtbot.addWidget(p)
        p._on_save_canonical()
        assert (target / "chemicals.csv").is_file()
        assert (target / "solutions.csv").is_file()


class TestCatalogsChangedSignal:
    def test_catalogs_changed_emitted_on_canonical_save(self, panel, qtbot):
        with qtbot.waitSignal(panel.catalogs_changed, timeout=1000):
            panel._on_save_canonical()

    def test_catalogs_changed_not_emitted_on_save_failure(self, panel, qtbot, monkeypatch):
        warnings: list[tuple] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

        def _boom(self, path):
            raise OSError("read-only directory")

        monkeypatch.setattr(ChemicalCatalog, "save_csv", _boom)
        with qtbot.assertNotEmitted(panel.catalogs_changed):
            panel._on_save_canonical()
        assert warnings  # error surfaced via QMessageBox.warning


class TestComponentChemicalCombo:
    def test_component_chemical_cell_is_combo_listing_catalog_names(self, panel):
        _select_solution(panel, "Silica solution")
        combo = panel._comp_table.cellWidget(0, 0)
        assert isinstance(combo, QComboBox)
        items = {combo.itemText(i) for i in range(combo.count())}
        assert set(panel._current_chem_names()).issubset(items)

    def test_component_combo_preserves_unknown_chemical_value(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        sol = SolutionCatalog()
        sol.add(Solution("Dye mix", [SolutionComponent("Ghost dye", "dep", 1.0, "mL")]))
        p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=sol)
        qtbot.addWidget(p)
        _select_solution(p, "Dye mix")
        combo = p._comp_table.cellWidget(0, 0)
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "Ghost dye" in items
        assert combo.currentText() == "Ghost dye"

    def test_component_combo_refreshes_when_chemical_added(self, panel):
        _select_solution(panel, "Silica solution")
        panel._on_add_chemical()
        row = panel._chem_table.rowCount() - 1
        panel._chem_table.setItem(row, 0, QTableWidgetItem("PEG 400"))
        for r in range(panel._comp_table.rowCount()):
            combo = panel._comp_table.cellWidget(r, 0)
            items = [combo.itemText(i) for i in range(combo.count())]
            assert "PEG 400" in items

    def test_component_combo_refresh_preserves_current_selection(self, panel):
        _select_solution(panel, "Silica solution")
        combo = panel._comp_table.cellWidget(1, 0)  # Isopropanol row
        combo.setCurrentText("Isopropanol")
        panel._on_add_chemical()
        row = panel._chem_table.rowCount() - 1
        panel._chem_table.setItem(row, 0, QTableWidgetItem("PEG 400"))
        assert combo.currentText() == "Isopropanol"

    def test_build_solution_reads_combo_chemical(self, panel):
        _select_solution(panel, "Silica solution")
        combo = panel._comp_table.cellWidget(0, 0)
        combo.setCurrentText("Water")
        sol = panel._build_current_solution()
        assert sol is not None
        assert "Water" in [c.chemical_name for c in sol.components]

    def test_add_component_row_combo_lists_current_names(self, panel):
        panel._on_add_component()
        row = panel._comp_table.rowCount() - 1
        combo = panel._comp_table.cellWidget(row, 0)
        items = {combo.itemText(i) for i in range(combo.count())}
        assert items == set(panel._current_chem_names())


class TestValidation:
    def test_validate_returns_issue_for_blank_density(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        issues = panel._validate_entries()
        assert any("Water" in i and "density" in i for i in issues)

    def test_validate_returns_issue_for_nonpositive_quantity(self, panel):
        _select_solution(panel, "Silica solution")
        panel._comp_table.item(0, 2).setText("0")
        issues = panel._validate_entries()
        assert any("quantity" in i for i in issues)

    def test_validate_returns_issue_for_unknown_chemical(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        sol = SolutionCatalog()
        sol.add(Solution("Dye mix", [SolutionComponent("Ghost dye", "dep", 1.0, "mL")]))
        p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=sol)
        qtbot.addWidget(p)
        _select_solution(p, "Dye mix")
        issues = p._validate_entries()
        assert any("unknown chemical" in i and "Ghost dye" in i for i in issues)

    def test_validate_no_issue_for_blank_mw(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 3).setText("")
        issues = panel._validate_entries()
        assert not any("MW" in i or "molar" in i.lower() for i in issues)

    def test_validate_no_issue_for_blank_viscosity(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 4).setText("")
        issues = panel._validate_entries()
        assert not any("viscosity" in i.lower() for i in issues)

    def test_validate_returns_issue_for_data_row_with_blank_name(self, panel):
        panel._on_add_chemical()
        row = panel._chem_table.rowCount() - 1
        panel._chem_table.setItem(row, 2, QTableWidgetItem("1.0"))  # density, no name
        issues = panel._validate_entries()
        assert any("has data but no name" in i for i in issues)

    def test_validate_returns_empty_for_clean_catalog(self, panel):
        assert panel._validate_entries() == []

    def test_save_cancel_aborts_write_and_signal(self, panel, qtbot, monkeypatch, tmp_path):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
        )
        with qtbot.assertNotEmitted(panel.catalogs_changed):
            panel._on_save_canonical()
        assert not (tmp_path / "chemicals.csv").exists()

    def test_save_proceed_writes_and_emits(self, panel, qtbot, monkeypatch, tmp_path):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        with qtbot.waitSignal(panel.catalogs_changed, timeout=1000):
            panel._on_save_canonical()
        assert (tmp_path / "chemicals.csv").is_file()
        assert (tmp_path / "solutions.csv").is_file()
        loaded = ChemicalCatalog.load_csv(tmp_path / "chemicals.csv")
        assert loaded.get("Water").density_g_per_mL == 1.0

    def test_save_no_issues_saves_silently(self, panel, qtbot, monkeypatch, tmp_path):
        def _fail(*a, **k):
            raise AssertionError("question must not be called on a clean save")

        monkeypatch.setattr(QMessageBox, "question", _fail)
        with qtbot.waitSignal(panel.catalogs_changed, timeout=1000):
            panel._on_save_canonical()
        assert (tmp_path / "chemicals.csv").is_file()

    def test_save_as_applies_same_validation_gate(self, panel, qtbot, monkeypatch, tmp_path):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
        )
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
        )
        with qtbot.assertNotEmitted(panel.catalogs_changed):
            panel._on_save_as()
        assert not (tmp_path / "chemicals.csv").exists()


class TestInvalidCellHighlight:
    def test_highlight_applied_to_blank_density_cell(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        panel._highlight_invalid_cells()
        assert panel._chem_table.item(row, 2).background().color() == INVALID_BG

    def test_highlight_cleared_when_cell_becomes_valid(self, panel):
        row = _chem_row_for(panel, "Water")
        panel._chem_table.item(row, 2).setText("")
        panel._highlight_invalid_cells()
        panel._chem_table.item(row, 2).setText("1.0")
        panel._highlight_invalid_cells()
        assert panel._chem_table.item(row, 2).background().color() != INVALID_BG

    def test_highlight_applied_to_nonpositive_quantity_cell(self, panel):
        _select_solution(panel, "Silica solution")
        panel._comp_table.item(0, 2).setText("0")
        panel._highlight_invalid_cells()
        assert panel._comp_table.item(0, 2).background().color() == INVALID_BG

    def test_highlight_does_not_recurse_on_item_changed(self, panel):
        _select_solution(panel, "Silica solution")
        calls: list[int] = []
        orig = panel._highlight_invalid_cells

        def counting() -> None:
            calls.append(1)
            orig()

        panel._highlight_invalid_cells = counting
        panel._comp_table.item(0, 2).setText("3")
        assert 0 < len(calls) <= 5


def _unknown_ref_panel(qtbot, monkeypatch, tmp_path, unit: str) -> FormulationPanel:
    """A panel whose only solution references a chemical missing from the catalog."""
    monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
    sol = SolutionCatalog()
    sol.add(Solution("Dye mix", [SolutionComponent("Ghost dye", "dep", 1.0, unit)]))
    p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=sol)
    qtbot.addWidget(p)
    _select_solution(p, "Dye mix")
    return p


def _rename_chem(panel: FormulationPanel, old: str, new: str) -> None:
    """Edit a chem name cell old→new; the connected itemChanged drives the cascade."""
    row = _chem_row_for(panel, old)
    panel._chem_table.item(row, 0).setText(new)  # fires itemChanged


class TestComputeValidation:
    def test_calculate_with_unknown_chem_shows_gate_not_keyerror(
        self, qtbot, monkeypatch, tmp_path
    ):
        p = _unknown_ref_panel(qtbot, monkeypatch, tmp_path, "g")
        questions: list[tuple] = []
        warnings: list[tuple] = []
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: questions.append(a) or QMessageBox.StandardButton.Cancel,
        )
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
        p._on_calculate()  # must not raise the KeyError
        assert questions, "compute gate dialog was not shown"
        assert not any("Calculation Error" in a[1] for a in warnings)

    def test_calculate_cancel_aborts_no_result_no_emit(self, qtbot, monkeypatch, tmp_path):
        p = _unknown_ref_panel(qtbot, monkeypatch, tmp_path, "g")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
        )
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
        with qtbot.assertNotEmitted(p.volumes_calculated):
            p._on_calculate()
        assert p._last_result is None
        assert p._lbl_result.text() == ""

    def test_calculate_proceed_with_volume_unit_unknown_ref_computes(
        self, qtbot, monkeypatch, tmp_path
    ):
        p = _unknown_ref_panel(qtbot, monkeypatch, tmp_path, "mL")
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
        )
        p._on_calculate()
        assert isinstance(p._last_result, list) and p._last_result
        assert p._lbl_result.text() != ""

    def test_calculate_clean_computes_without_dialog(self, panel, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError("question must not be called on a clean compute")

        monkeypatch.setattr(QMessageBox, "question", _fail)
        panel._on_calculate()
        assert isinstance(panel._last_result, list) and panel._last_result

    def test_calculate_clean_then_apply_emits_volumes(self, panel, qtbot):
        panel._on_calculate()
        with qtbot.waitSignal(panel.volumes_calculated, timeout=1000) as blocker:
            panel._on_apply()
        assert blocker.args[0] == panel._last_result

    def test_calculate_blank_density_gate_lists_issue(self, panel, monkeypatch):
        row = _chem_row_for(panel, "Fumed silica")
        panel._chem_table.item(row, 2).setText("")  # blank density
        bodies: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *a, **k: bodies.append(a[2]) or QMessageBox.StandardButton.Cancel,
        )
        panel._on_calculate()
        assert bodies, "gate dialog not shown for blank density"
        assert "Fumed silica" in bodies[0] and "density" in bodies[0]
        assert panel._last_result is None


class TestChemicalRenameCascade:
    def test_rename_updates_component_refs_in_catalog(self, panel):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        _rename_chem(panel, "Fumed silica", "Fumed silica HS")
        comps = panel._sol_catalog.get("Silica solution").components
        names = [c.chemical_name for c in comps]
        assert "Fumed silica HS" in names
        assert "Fumed silica" not in names

    def test_rename_updates_onscreen_combos(self, panel):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        _rename_chem(panel, "Fumed silica", "Fumed silica HS")
        combo = None
        for r in range(panel._comp_table.rowCount()):
            c = panel._comp_table.cellWidget(r, 0)
            if isinstance(c, QComboBox) and (
                c.currentData() or c.currentText()
            ) == "Fumed silica HS":
                combo = c
                break
        assert combo is not None
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "Fumed silica" not in items

    def test_rename_cascades_to_non_displayed_solution(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        sol = SolutionCatalog()
        sol.add(Solution("Sol A", [
            SolutionComponent("Fumed silica", "dep", 1.0, "g", calc_mode="Mass-based"),
        ]))
        sol.add(Solution("Sol B", [
            SolutionComponent("Fumed silica", "dep", 2.0, "g", calc_mode="Mass-based"),
        ]))
        p = FormulationPanel(chem_catalog=_synthetic_chem(), sol_catalog=sol)
        qtbot.addWidget(p)
        _select_solution(p, "Sol A")
        _rename_chem(p, "Fumed silica", "Fumed silica HS")
        b_names = [c.chemical_name for c in p._sol_catalog.get("Sol B").components]
        assert b_names == ["Fumed silica HS"]

    def test_rename_to_existing_name_blocked_and_reverted(self, panel, monkeypatch):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        warnings: list[tuple] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
        _rename_chem(panel, "Fumed silica", "Isopropanol")  # collides
        row = _chem_row_for(panel, "Fumed silica")
        item = panel._chem_table.item(row, 0)
        assert item.text() == "Fumed silica"
        assert item.data(Qt.UserRole) == "Fumed silica"
        names = [c.chemical_name for c in panel._sol_catalog.get("Silica solution").components]
        assert "Fumed silica" in names and "Isopropanol" in names
        assert len(warnings) == 1

    def test_clearing_name_not_cascaded(self, panel):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        row = _chem_row_for(panel, "Fumed silica")
        panel._chem_table.item(row, 0).setText("")  # clear
        names = [c.chemical_name for c in panel._sol_catalog.get("Silica solution").components]
        assert "Fumed silica" in names  # ref left pointing at old name
        issues = panel._validate_entries()
        assert any("unknown chemical" in i and "Fumed silica" in i for i in issues)

    def test_new_row_first_name_not_treated_as_rename(self, panel, monkeypatch):
        warnings: list[tuple] = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))
        before = {
            name: [c.chemical_name for c in panel._sol_catalog.get(name).components]
            for name in panel._sol_catalog.list_names()
        }
        panel._on_add_chemical()
        row = panel._chem_table.rowCount() - 1
        panel._chem_table.setItem(row, 0, QTableWidgetItem("PEG 400"))  # fires itemChanged
        after = {
            name: [c.chemical_name for c in panel._sol_catalog.get(name).components]
            for name in panel._sol_catalog.list_names()
        }
        assert after == before  # no ref rewritten
        assert not warnings
        assert "PEG 400" in panel._current_chem_names()

    def test_chained_renames_land_refs_at_final_name(self, panel):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        _rename_chem(panel, "Fumed silica", "A")
        _rename_chem(panel, "A", "B")
        names = [c.chemical_name for c in panel._sol_catalog.get("Silica solution").components]
        assert "B" in names and "A" not in names and "Fumed silica" not in names
        row = _chem_row_for(panel, "B")
        assert panel._chem_table.item(row, 0).data(Qt.UserRole) == "B"

    def test_rename_does_not_recurse_on_item_changed(self, panel):
        _select_solution(panel, "Silica solution")
        panel._build_current_solution()
        _rename_chem(panel, "Fumed silica", "Fumed silica HS")  # must not RecursionError
        row = _chem_row_for(panel, "Fumed silica HS")
        item = panel._chem_table.item(row, 0)
        assert item.text() == "Fumed silica HS"
        assert item.data(Qt.UserRole) == "Fumed silica HS"

    def test_add_chem_row_stamps_userrole(self):
        table = QTableWidget(0, 7)
        chem = Chemical("Fumed silica", "O=[Si]=O", density_g_per_mL=2.65)
        fio.add_chem_row(table, chem)
        assert table.item(0, 0).data(Qt.UserRole) == "Fumed silica"
        fio.add_chem_row(table, None)
        assert table.item(1, 0) is None  # blank row creates no col-0 item


def test_counts_as_deposit_column_roundtrips(qtbot):
    """The 'Deposits?' combo round-trips None/True/False through build_solution."""
    table = QTableWidget(0, 6)
    qtbot.addWidget(table)
    names = ["PEO", "LiCl", "SiO2"]
    fio.add_comp_row(table, SolutionComponent("PEO", "dep", 1.0, "mL"), names)  # None
    fio.add_comp_row(
        table, SolutionComponent("LiCl", "solute", 1.0, "mL", counts_as_deposit=False), names
    )
    fio.add_comp_row(
        table, SolutionComponent("SiO2", "dep", 1.0, "mL", counts_as_deposit=True), names
    )
    by = {c.chemical_name: c.counts_as_deposit for c in fio.build_solution("s", table).components}
    assert by["PEO"] is None
    assert by["LiCl"] is False
    assert by["SiO2"] is True


def _targets_panel(qtbot):
    chem = ChemicalCatalog()
    chem.add(Chemical("PolyA", density_g_per_mL=1.2, molar_mass_g_per_mol=100.0))
    chem.add(Chemical("PolyB", density_g_per_mL=1.5, molar_mass_g_per_mol=120.0))
    chem.add(Chemical("Water", density_g_per_mL=1.0, molar_mass_g_per_mol=18.0))
    sol = SolutionCatalog()
    sol.add(Solution("A", [SolutionComponent("PolyA", "dep", 2.0, "mL"),
                           SolutionComponent("Water", "carrier", 8.0, "mL")]))
    sol.add(Solution("B", [SolutionComponent("PolyB", "dep", 3.0, "mL"),
                           SolutionComponent("Water", "carrier", 7.0, "mL")]))
    p = FormulationPanel(chem_catalog=chem, sol_catalog=sol)
    qtbot.addWidget(p)
    for i in range(p._list_solutions.count()):
        p._list_solutions.item(i).setCheckState(Qt.CheckState.Checked)
    p._on_solution_set_changed()  # rebuild pump rows for both stocks
    return p


def test_formulation_panel_targets_mode_disables_fraction_spins(qtbot):
    p = _targets_panel(qtbot)
    p._combo_form_mode.setCurrentIndex(1)
    assert p._form_mode() == "targets"
    assert not p._targets_editor.isHidden()
    assert all(not s.isEnabled() for s in p._frac_spins.values())
    p._combo_form_mode.setCurrentIndex(0)
    assert all(s.isEnabled() for s in p._frac_spins.values())


def test_formulation_panel_targets_solve_emits_pump_vector(qtbot, monkeypatch):
    p = _targets_panel(qtbot)
    monkeypatch.setattr(p, "_validate_entries", lambda: [])  # no validation dialog
    p._combo_form_mode.setCurrentIndex(1)
    # distinct pumps for A and B so the vector has two nonzero entries
    combos = list(p._pump_combos.values())
    combos[1].setCurrentIndex(1)
    p._targets_editor.add_target("Dried fraction", a="PolyB", value="0.4")
    p._spin_target.setValue(5.0)

    p._on_calculate()
    assert p._last_result is not None
    assert len(p._last_result) == 3            # [pump0, pump1, total]
    assert p._last_result[-1] == pytest.approx(sum(p._last_result[:-1]), rel=1e-9)

    received: list[list[float]] = []
    p.volumes_calculated.connect(received.append)
    p._on_apply()
    assert received and received[0] == p._last_result  # same vector the HT tab consumes


# ── Solution select/deselect toggle + rename (CatalogEditorMixin) ─────────────


def test_toggle_all_solutions(panel):
    assert panel._solutions_all_checked()                 # default: all checked
    assert panel._btn_toggle_all_sol.text() == "Deselect All"
    panel._on_toggle_all_solutions()                      # → deselect all
    assert not panel._solutions_all_checked()
    assert panel._btn_toggle_all_sol.text() == "Select All"
    panel._on_toggle_all_solutions()                      # → select all
    assert panel._solutions_all_checked()
    assert panel._btn_toggle_all_sol.text() == "Deselect All"


def test_rename_solution(panel, monkeypatch):
    _select_solution(panel, "Silica solution")
    monkeypatch.setattr(
        "softae.gui.widgets.catalog_editor_base.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("PEO solution", True)),
    )
    panel._on_rename_solution()
    names = panel._sol_catalog.list_names()
    assert "PEO solution" in names and "Silica solution" not in names
    assert any(panel._list_solutions.item(i).text() == "PEO solution"
               for i in range(panel._list_solutions.count()))
    assert len(panel._sol_catalog.get("PEO solution").components) == 2  # components kept


def test_rename_solution_duplicate_blocked(panel, monkeypatch):
    panel._sol_catalog.add(Solution("Other", []))
    _select_solution(panel, "Silica solution")
    monkeypatch.setattr(
        "softae.gui.widgets.catalog_editor_base.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("Other", True)),
    )
    warned: list = []
    monkeypatch.setattr(
        "softae.gui.widgets.catalog_editor_base.QMessageBox.warning",
        staticmethod(lambda *a, **k: warned.append(a)),
    )
    panel._on_rename_solution()
    assert "Silica solution" in panel._sol_catalog.list_names()  # unchanged
    assert warned                                                # duplicate flagged
