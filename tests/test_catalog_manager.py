"""Tests for the slim CatalogManager CRUD editor (shared CatalogEditorMixin)."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QInputDialog, QMessageBox

from softae.config import loader
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.gui.widgets.catalog_manager import CatalogManager


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


def _select_solution(mgr: CatalogManager, name: str) -> None:
    for i in range(mgr._list_solutions.count()):
        item = mgr._list_solutions.item(i)
        if item is not None and item.text() == name:
            mgr._list_solutions.setCurrentItem(item)
            mgr._on_solution_selected(name)
            return
    raise AssertionError(f"solution {name!r} not found")


@pytest.fixture
def manager(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
    m = CatalogManager(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
    qtbot.addWidget(m)
    return m


class TestCatalogManagerCRUD:
    def test_add_chemical_appends_chem_row_expected(self, manager):
        before = manager._chem_table.rowCount()
        manager._on_add_chemical()
        assert manager._chem_table.rowCount() == before + 1

    def test_remove_chemical_drops_selected_row_expected(self, manager):
        before = manager._chem_table.rowCount()
        manager._chem_table.setCurrentCell(0, 0)
        manager._on_remove_chemical()
        assert manager._chem_table.rowCount() == before - 1

    def test_copy_chemical_prepopulates_unique_named_row_expected(self, manager):
        manager._chem_table.setCurrentCell(0, 0)  # Water
        before = manager._chem_table.rowCount()
        manager._on_copy_chemical()
        assert manager._chem_table.rowCount() == before + 1
        last = manager._chem_table.item(before, 0).text()
        assert "copy" in last.lower()
        assert last != "Water"

    def test_new_solution_adds_list_item_expected(self, manager, monkeypatch):
        monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("Brine", True))
        before = manager._list_solutions.count()
        manager._on_new_solution()
        assert manager._list_solutions.count() == before + 1

    def test_delete_solution_removes_list_item_and_clears_components_expected(self, manager):
        _select_solution(manager, "Silica solution")
        before = manager._list_solutions.count()
        manager._on_delete_solution()
        assert manager._list_solutions.count() == before - 1
        assert manager._comp_table.rowCount() == 0

    def test_add_component_row_lists_current_chem_names_expected(self, manager):
        manager._on_add_component()
        row = manager._comp_table.rowCount() - 1
        combo = manager._comp_table.cellWidget(row, 0)
        items = {combo.itemText(i) for i in range(combo.count())}
        assert items == set(manager._current_chem_names())

    def test_component_combo_preserves_unknown_chemical_ref_expected(
        self, qtbot, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        sol = SolutionCatalog()
        sol.add(Solution("Dye mix", [SolutionComponent("Ghost dye", "dep", 1.0, "mL")]))
        m = CatalogManager(chem_catalog=_synthetic_chem(), sol_catalog=sol)
        qtbot.addWidget(m)
        _select_solution(m, "Dye mix")
        combo = m._comp_table.cellWidget(0, 0)
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "Ghost dye" in items
        assert combo.currentText() == "Ghost dye"

    def test_solution_set_changed_hook_is_noop_no_pump_widgets_expected(self, manager):
        assert not hasattr(manager, "_pump_combos")
        assert not hasattr(manager, "_refresh_pump_assignments")
        # The base hook is a no-op that never raises.
        assert manager._on_solution_set_changed() is None


class TestCatalogManagerSave:
    def test_save_writes_both_csvs_under_data_root_and_creates_dir_expected(
        self, qtbot, monkeypatch, tmp_path
    ):
        target = tmp_path / "nested" / "data"  # does not exist yet
        monkeypatch.setattr(loader, "data_root", lambda: target)
        m = CatalogManager(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
        qtbot.addWidget(m)
        m._on_save_canonical()
        assert (target / "chemicals.csv").is_file()
        assert (target / "solutions.csv").is_file()

    def test_save_round_trip_matches_core_preserves_particulate_and_calc_mode_expected(
        self, manager, tmp_path
    ):
        _select_solution(manager, "Silica solution")
        manager._on_save_canonical()
        chem = ChemicalCatalog.load_csv(tmp_path / "chemicals.csv")
        sol = SolutionCatalog.load_csv(tmp_path / "solutions.csv")
        assert chem.get("Fumed silica").is_particulate is True
        assert chem.get("Fumed silica").viscosity_mPa_s is None
        modes = {c.chemical_name: c.calc_mode for c in sol.get("Silica solution").components}
        assert modes["Fumed silica"] == "Mass-based"
        assert modes["Isopropanol"] == "Mass-based"

    def test_catalogs_changed_emitted_on_successful_save_expected(self, manager, qtbot):
        with qtbot.waitSignal(manager.catalogs_changed, timeout=1000):
            manager._on_save_canonical()

    def test_save_cancel_on_validation_aborts_write_and_signal_expected(
        self, manager, qtbot, monkeypatch, tmp_path
    ):
        row = None
        for r in range(manager._chem_table.rowCount()):
            if manager._chem_table.item(r, 0).text() == "Water":
                row = r
                break
        manager._chem_table.item(row, 2).setText("")  # blank density
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel
        )
        with qtbot.assertNotEmitted(manager.catalogs_changed):
            manager._on_save_canonical()
        assert not (tmp_path / "chemicals.csv").exists()

    def test_ctor_auto_loads_from_data_root_when_no_catalogs_given_expected(
        self, qtbot, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        m = CatalogManager()
        qtbot.addWidget(m)
        assert m._chem_table.rowCount() == 3
        assert m._list_solutions.count() == 1

    def test_ctor_uses_injected_catalogs_without_reading_data_root_expected(
        self, qtbot, monkeypatch
    ):
        called: list[int] = []
        monkeypatch.setattr(loader, "data_root", lambda: called.append(1) or "x")
        m = CatalogManager(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
        qtbot.addWidget(m)
        assert m._chem_table.rowCount() == 3
        assert called == []  # data_root never read on the injection path
