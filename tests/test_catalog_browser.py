"""Tests for the read-only live CatalogBrowser widget."""

from __future__ import annotations

import pytest

from softae.config import loader
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.gui.widgets.catalog_browser import CatalogBrowser


def _synthetic_chem() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.0))
    cat.add(Chemical("Isopropanol", "CC(O)C", density_g_per_mL=0.786))
    cat.add(Chemical("Fumed silica", "O=[Si]=O", density_g_per_mL=2.65, is_particulate=True))
    return cat


def _synthetic_sol() -> SolutionCatalog:
    cat = SolutionCatalog()
    cat.add(Solution("Silica solution", [
        SolutionComponent("Fumed silica", "dep", 1.0, "g"),
        SolutionComponent("Isopropanol", "carrier", 9.0, "mL"),
    ]))
    return cat


@pytest.fixture
def browser(qtbot):
    b = CatalogBrowser(chem_catalog=_synthetic_chem(), sol_catalog=_synthetic_sol())
    qtbot.addWidget(b)
    return b


class TestCatalogBrowser:
    def test_browser_lists_all_chemicals_expected(self, browser):
        assert browser._chem_table.rowCount() == len(_synthetic_chem())

    def test_browser_lists_solutions_with_dep_fraction_and_components_expected(self, browser):
        assert browser._sol_table.rowCount() == 1
        assert browser._sol_table.item(0, 0).text() == "Silica solution"
        # Dep fraction formatted .2f (non-empty, parseable).
        float(browser._sol_table.item(0, 1).text())
        comps = browser._sol_table.item(0, 2).text()
        assert "Fumed silica" in comps and "Isopropanol" in comps

    def test_reload_with_catalogs_repopulates_without_disk_read_expected(
        self, browser, monkeypatch
    ):
        called: list[int] = []
        monkeypatch.setattr(loader, "data_root", lambda: called.append(1) or "x")
        chem = ChemicalCatalog()
        chem.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789))
        browser.reload(chem, SolutionCatalog())
        assert browser._chem_table.rowCount() == 1
        assert called == []  # data_root never read on the injected path

    def test_reload_no_args_reads_data_root_expected(self, browser, monkeypatch, tmp_path):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        browser.reload()
        assert browser._chem_table.rowCount() == 3
        assert browser._sol_table.rowCount() == 1

    def test_edit_button_emits_edit_requested_expected(self, browser, qtbot):
        with qtbot.waitSignal(browser.edit_requested, timeout=1000):
            browser._btn_edit.click()

    def test_reload_missing_files_degrades_to_empty_without_raising_expected(
        self, browser, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)  # empty dir
        browser.reload()  # must not raise
        assert browser._chem_table.rowCount() == 0
        assert browser._sol_table.rowCount() == 0

    def test_reload_is_idempotent_expected(self, browser):
        chem, sol = _synthetic_chem(), _synthetic_sol()
        browser.reload(chem, sol)
        first_rows = browser._chem_table.rowCount()
        browser.reload(chem, sol)
        assert browser._chem_table.rowCount() == first_rows

    def test_browser_loads_from_data_root_when_no_catalogs_injected_expected(
        self, qtbot, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        b = CatalogBrowser()  # no injected catalogs → reads data_root
        qtbot.addWidget(b)
        assert b._chem_table.rowCount() == 3
