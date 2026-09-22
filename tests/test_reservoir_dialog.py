"""Tests for the syringe-stock declaration dialog."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from softae.core.data_store import DataStore
from softae.core.formulation import ChemicalCatalog, SolutionCatalog, Solution
from softae.core.reservoir import ReservoirLedger
from softae.gui.widgets.reservoir_dialog import ReservoirDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def test_untracked_pumps_read_as_unknown_not_empty(qapp, store):
    """An undeclared stock must never look like a depleted one."""
    dlg = ReservoirDialog(ReservoirLedger(store))
    assert dlg._labels[0].text() == "not tracked"


def test_declare_brings_a_pump_under_management_and_persists(qapp, store):
    ledger = ReservoirLedger(store)
    dlg = ReservoirDialog(ledger)

    dlg._spins[1].setValue(4000.0)
    dlg._declare(1)

    assert ledger.remaining_uL(1) == 4000.0
    assert store.reservoir_level_uL(1) == 4000.0
    assert "4000" in dlg._labels[1].text()


def _catalogs_named(*names):
    """A ``(chemicals, solutions)`` pair standing in for the data root's CSVs."""
    sol = SolutionCatalog()
    for name in names:
        sol.add(Solution(name=name))
    return ChemicalCatalog(), sol


def _combo_items(combo) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def test_reservoir_dialog_without_catalog_falls_back_to_data_root(
    qapp, store, monkeypatch
):
    """Handed no catalog, the dialog reads the data root instead of offering nothing."""
    monkeypatch.setattr(
        "softae.core.stock_assignment.catalogs_from_data_root",
        lambda: _catalogs_named("LiCl 1M", "PEO 5wt%"))

    dlg = ReservoirDialog(ReservoirLedger(store), data_store=store)

    items = _combo_items(dlg._stock_boxes[0])
    assert "LiCl 1M" in items and "PEO 5wt%" in items


def test_reservoir_dialog_given_a_catalog_does_not_reach_the_data_root(
    qapp, store, monkeypatch
):
    """The fallback is a fallback: a supplied catalog is what the combo shows."""
    monkeypatch.setattr(
        "softae.core.stock_assignment.catalogs_from_data_root",
        lambda: _catalogs_named("from the data root"))
    _, supplied = _catalogs_named("from the caller")

    dlg = ReservoirDialog(ReservoirLedger(store), data_store=store,
                          sol_catalog=supplied)

    items = _combo_items(dlg._stock_boxes[0])
    assert "from the caller" in items and "from the data root" not in items


def test_reservoir_dialog_without_store_disables_stock_selection(qapp, store):
    """Nothing may look writable when the pick cannot be persisted."""
    dlg = ReservoirDialog(ReservoirLedger(store))

    assert all(not box.isEnabled() for box in dlg._stock_boxes.values())
    assert dlg._stock_note.isVisibleTo(dlg)
    assert dlg._stock_note.text().strip()


def test_reservoir_dialog_with_a_store_keeps_stock_selection_live(qapp, store):
    """The disable is conditional — a dialog that can persist must stay usable."""
    dlg = ReservoirDialog(ReservoirLedger(store), data_store=store)

    assert all(box.isEnabled() for box in dlg._stock_boxes.values())
    assert not dlg._stock_note.isVisibleTo(dlg)


def test_depleted_stock_is_shown_distinctly_from_low_stock(qapp, store):
    ledger = ReservoirLedger(store, soft_warn_uL=1000.0, hard_stop_uL=250.0)
    ledger.refill(0, 200.0)    # below the hard stop
    ledger.refill(1, 600.0)    # below the soft warn only
    ledger.refill(2, 5000.0)   # healthy

    dlg = ReservoirDialog(ledger)
    below_stop = dlg._labels[0].styleSheet()
    low = dlg._labels[1].styleSheet()

    assert below_stop and low and below_stop != low
    assert dlg._labels[2].styleSheet() == ""
