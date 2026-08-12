"""Tests for the syringe-stock declaration dialog."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from softae.core.data_store import DataStore
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
