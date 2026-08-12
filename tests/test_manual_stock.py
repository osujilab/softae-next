"""Reporting syringe stock from the Manual Control tab.

The interlock is only as good as the operator's ability to declare what is
actually loaded, and the bench-side place to do that is next to the pump
controls — not buried in a menu.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from softae.core.data_store import DataStore
from softae.core.reservoir import ReservoirLedger, attach_reservoir_ledger
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_manual import ManualControlTab


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


@pytest.fixture
def tab(qapp, store):
    mgr = create_mock_manager(config={})
    t = ManualControlTab(mgr, data_store=store)
    yield t, mgr
    t.cleanup()


def _stock_text(tab, pump_id: int) -> str:
    return tab._pump_widgets[pump_id]["stock_lbl"].text()


def test_untracked_stock_says_so_rather_than_showing_a_number(tab):
    """An undeclared stock must never read as a quantity the operator can trust."""
    t, _ = tab
    assert "not tracked" in _stock_text(t, 0)


def test_declared_stock_is_displayed_per_pump(tab, store):
    t, mgr = tab
    ledger = attach_reservoir_ledger(mgr, store, config={})
    ledger.refill(0, 4000.0)

    t.refresh_stock_labels()

    assert "4000" in _stock_text(t, 0)
    assert "not tracked" in _stock_text(t, 1)   # untouched pumps stay unmanaged


def test_low_and_depleted_are_visually_distinct(tab, store):
    t, mgr = tab
    ledger = attach_reservoir_ledger(
        mgr, store,
        config={"reservoir_soft_warn_uL": 1000.0, "reservoir_hard_stop_uL": 250.0},
    )
    ledger.refill(0, 200.0)    # below the hard stop
    ledger.refill(1, 600.0)    # low only
    ledger.refill(2, 5000.0)   # healthy

    t.refresh_stock_labels()

    depleted = t._pump_widgets[0]["stock_lbl"].styleSheet()
    low = t._pump_widgets[1]["stock_lbl"].styleSheet()
    healthy = t._pump_widgets[2]["stock_lbl"].styleSheet()

    assert depleted and low and depleted != low
    assert healthy == ""


def test_readout_follows_a_dispense(tab, store):
    t, mgr = tab
    ledger = attach_reservoir_ledger(mgr, store, config={})
    ledger.refill(0, 5000.0)

    mgr.get("syringe").single_pump(res_vol=1000, ID=0, rate=100.0, dispense_vol=500.0)
    t.refresh_stock_labels()

    assert "4500" in _stock_text(t, 0)


def test_tab_works_without_a_ledger_attached(tab):
    """A rig that never declared stock must still be fully operable."""
    t, _ = tab
    assert t._reservoir_ledger() is None
    t.refresh_stock_labels()          # must not raise
    assert "not tracked" in _stock_text(t, 2)
