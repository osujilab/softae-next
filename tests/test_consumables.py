"""Waste capacity and spare-plate inventory (P5.4).

Both cap unattended time from the opposite direction to stock: you can have
plenty of solution and still be stopped by a full waste bottle or an empty
plate drawer. Neither is sensable, so both are operator assertions the platform
remembers and refuses to walk past.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.consumables import (
    BoardInventory,
    WasteLedger,
    attach_consumables,
)
from softae.core.data_store import DataStore
from softae.errors import SafetyError


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


# ── Waste ────────────────────────────────────────────────────────────────────

class TestWaste:
    def test_undeclared_capacity_tracks_without_enforcing(self, store):
        """Adding this must not break a bench that never declared a capacity."""
        w = WasteLedger(store)
        w.check(1_000_000.0)          # must not raise
        assert w.add(500.0).level_uL == 500.0
        assert w.capacity_uL is None

    def test_accumulates_across_transfers(self, store):
        w = WasteLedger(store, capacity_uL=10_000.0)
        w.add(300.0)
        w.add(200.0)
        assert w.level_uL() == 500.0

    def test_overflow_is_refused(self, store):
        w = WasteLedger(store, capacity_uL=1000.0)
        w.add(900.0)
        with pytest.raises(SafetyError, match="overflow"):
            w.check(200.0)

    def test_a_transfer_that_exactly_fills_is_allowed(self, store):
        w = WasteLedger(store, capacity_uL=1000.0)
        w.check(1000.0)               # boundary is not an overflow
        w.add(1000.0)
        assert w.level_uL() == 1000.0

    def test_warns_once_on_crossing_not_every_transfer(self, store):
        seen: list[float] = []
        w = WasteLedger(store, capacity_uL=1000.0, warn_fraction=0.8,
                        on_warn=lambda lvl, cap: seen.append(lvl))
        w.add(700.0)
        assert seen == []
        w.add(150.0)                  # crosses 800
        w.add(50.0)                   # already above — must not warn again
        assert len(seen) == 1

    def test_emptying_resets_the_level(self, store):
        w = WasteLedger(store, capacity_uL=1000.0)
        w.add(900.0)
        w.empty()
        assert w.level_uL() == 0.0
        w.check(900.0)                # room again

    def test_level_survives_reopen(self, tmp_path: Path):
        """A waste bottle does not empty itself when the GUI closes."""
        ds = DataStore(tmp_path / "proj")
        WasteLedger(ds).add(1234.0)
        ds.close()

        reopened = DataStore(tmp_path / "proj")
        try:
            assert WasteLedger(reopened).level_uL() == 1234.0
        finally:
            reopened.close()

    def test_headroom_and_fraction_report_state(self, store):
        w = WasteLedger(store, capacity_uL=1000.0)
        status = w.add(250.0)
        assert status.fraction == pytest.approx(0.25)
        assert status.headroom_uL == pytest.approx(750.0)

    def test_a_zero_capacity_is_rejected_as_a_config_error(self, store):
        with pytest.raises(ValueError):
            WasteLedger(store, capacity_uL=0.0)

    def test_a_broken_store_does_not_break_tracking(self):
        class _Broken:
            def waste_level_uL(self):
                raise OSError("db gone")

            def set_waste_level(self, v):
                raise OSError("db gone")

        w = WasteLedger(_Broken(), capacity_uL=1000.0)
        assert w.add(100.0).level_uL == 100.0    # in-memory still correct


# ── Spare boards ─────────────────────────────────────────────────────────────

class TestBoardInventory:
    def test_undeclared_is_unknown_not_zero(self, store):
        """Claiming zero plates would stop campaigns on no evidence."""
        inv = BoardInventory(store)
        assert inv.remaining() is None
        assert not inv.is_managed
        inv.check()                   # must not raise

    def test_declared_count_is_used(self, store):
        inv = BoardInventory(store)
        inv.declare(3)
        assert inv.remaining() == 3
        assert inv.is_managed

    def test_consuming_decrements(self, store):
        inv = BoardInventory(store)
        inv.declare(2)
        assert inv.consume() == 1
        assert inv.consume() == 0

    def test_exhausted_inventory_refuses(self, store):
        inv = BoardInventory(store)
        inv.declare(0)
        with pytest.raises(SafetyError, match="No spare electrode boards"):
            inv.check()

    def test_count_never_goes_negative(self, store):
        inv = BoardInventory(store)
        inv.declare(1)
        inv.consume()
        inv.consume()
        assert inv.remaining() == 0

    def test_count_survives_reopen(self, tmp_path: Path):
        ds = DataStore(tmp_path / "proj")
        BoardInventory(ds).declare(4)
        ds.close()

        reopened = DataStore(tmp_path / "proj")
        try:
            assert BoardInventory(reopened).remaining() == 4
        finally:
            reopened.close()

    def test_unmanaged_consume_is_a_no_op(self, store):
        assert BoardInventory(store).consume() is None


# ── Wiring ───────────────────────────────────────────────────────────────────

class TestAttach:
    def test_attach_builds_both_from_one_call(self, store):
        waste, boards = attach_consumables(store, config={})
        assert isinstance(waste, WasteLedger)
        assert isinstance(boards, BoardInventory)

    def test_capacity_comes_from_config(self, store):
        waste, _ = attach_consumables(store, config={"waste_capacity_uL": 5000.0})
        assert waste.capacity_uL == 5000.0

    def test_absent_capacity_leaves_it_unmanaged(self, store):
        waste, _ = attach_consumables(store, config={})
        assert waste.capacity_uL is None

    def test_bad_capacity_value_degrades_to_unmanaged(self, store):
        """A config typo must not turn into a phantom limit."""
        waste, _ = attach_consumables(store, config={"waste_capacity_uL": "lots"})
        assert waste.capacity_uL is None


# ── Board-exchange integration ───────────────────────────────────────────────

class TestExchangeGate:
    def test_gate_cancels_rather_than_prompting_with_no_plates(self, store):
        """Waking someone at 3 a.m. for a plate they lack is the worse outcome."""
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from softae.core.autonomous_loop import BoardDecision
        from tests.test_autonomous_run_mixin import _BareHost  # noqa: PLC0415

        app = QApplication.instance() or QApplication([])
        host = _BareHost(_DummyManager())
        try:
            inv = BoardInventory(store)
            inv.declare(0)
            host._board_inventory = inv

            assert host._board_exchange_gate(2) is BoardDecision.CANCEL
            assert any("No spare" in m for m in host.logs)
        finally:
            host.deleteLater()

    def test_gate_proceeds_normally_when_inventory_is_undeclared(self, store):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication, QMessageBox

        from softae.core.autonomous_loop import BoardDecision
        from tests.test_autonomous_run_mixin import _BareHost  # noqa: PLC0415

        app = QApplication.instance() or QApplication([])
        host = _BareHost(_DummyManager())
        try:
            host._board_inventory = BoardInventory(store)   # undeclared
            original = QMessageBox.question
            QMessageBox.question = staticmethod(
                lambda *a, **k: QMessageBox.StandardButton.Yes)
            try:
                assert host._board_exchange_gate(2) is BoardDecision.PROCEED
            finally:
                QMessageBox.question = original
        finally:
            host.deleteLater()


class _DummyManager:
    def get(self, name):
        raise KeyError(name)

    def reset_locks(self):
        pass
