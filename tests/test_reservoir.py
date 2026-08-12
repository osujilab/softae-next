"""Stock-volume ledger — the mechanical-hazard interlock (P5.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.data_store import DataStore
from softae.core.reservoir import (
    DEFAULT_HARD_STOP_UL,
    DEFAULT_SOFT_WARN_UL,
    ReservoirLedger,
    attach_reservoir_ledger,
)
from softae.drivers.mock_factory import create_mock_manager
from softae.errors import SafetyError


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


class TestUnmanagedPumps:
    def test_unknown_pump_is_not_treated_as_empty(self):
        """Adding the ledger must not break a rig with undeclared reservoirs."""
        led = ReservoirLedger()
        assert led.remaining_uL(0) is None
        led.check(0, 10_000.0)              # must not raise
        assert led.debit(0, 10_000.0) is None


class TestHardStop:
    def test_refuses_a_dispense_that_would_breach_the_stop(self):
        led = ReservoirLedger()
        led.refill(0, 400.0)                # 400 µL left, hard stop at 250
        with pytest.raises(SafetyError, match="hard stop"):
            led.check_and_debit(0, 200.0)   # would leave 200 < 250

    def test_allows_a_dispense_that_stays_above_the_stop(self):
        led = ReservoirLedger()
        led.refill(0, 400.0)
        st = led.check_and_debit(0, 100.0)  # leaves 300 >= 250
        assert st is not None and st.remaining_uL == 300.0

    def test_refusal_leaves_the_level_untouched(self):
        """A refused dispense must not be debited."""
        led = ReservoirLedger()
        led.refill(0, 400.0)
        with pytest.raises(SafetyError):
            led.check_and_debit(0, 200.0)
        assert led.remaining_uL(0) == 400.0

    def test_error_is_a_hard_fault_for_the_loop(self):
        """SafetyError ⇒ AutonomousLoop parks immediately, no retries."""
        from softae.core.autonomous_loop import AutonomousLoop

        led = ReservoirLedger()
        led.refill(0, 300.0)
        try:
            led.check(0, 100.0)
        except SafetyError as exc:
            assert AutonomousLoop._is_hard_fault(None, exc) is True
        else:
            pytest.fail("expected SafetyError")


class TestSoftWarn:
    def test_warns_once_on_crossing_not_every_dispense(self):
        seen: list[tuple[int, float]] = []
        led = ReservoirLedger(on_warn=lambda p, r: seen.append((p, r)))
        led.refill(0, 1200.0)               # above the 1000 µL soft warn

        led.debit(0, 100.0)                 # 1100 — still above
        assert seen == []
        led.debit(0, 200.0)                 # 900 — crosses
        assert len(seen) == 1
        led.debit(0, 100.0)                 # 800 — already warned
        assert len(seen) == 1

    def test_soft_warn_does_not_block(self):
        led = ReservoirLedger()
        led.refill(0, 1100.0)
        st = led.check_and_debit(0, 300.0)  # 800: warns, still proceeds
        assert st.warned is True
        assert st.remaining_uL == 800.0


class TestPersistence:
    def test_levels_survive_reopen(self, tmp_path: Path):
        with DataStore(tmp_path / "p") as ds:
            ReservoirLedger(ds).refill(0, 5000.0)
        with DataStore(tmp_path / "p") as ds2:
            assert ReservoirLedger(ds2).remaining_uL(0) == 5000.0

    def test_debit_is_persisted(self, store):
        led = ReservoirLedger(store)
        led.refill(1, 5000.0)
        led.debit(1, 1500.0)
        assert store.reservoir_level_uL(1) == 3500.0
        assert ReservoirLedger(store).remaining_uL(1) == 3500.0   # fresh instance

    def test_a_broken_store_does_not_break_dispensing(self, store):
        class Broken:
            def reservoir_level_uL(self, pid):
                raise RuntimeError("db gone")

            def set_reservoir_level(self, pid, v):
                raise RuntimeError("db gone")

        led = ReservoirLedger(Broken())
        assert led.remaining_uL(0) is None
        led.check(0, 999.0)                 # unmanaged → passes through


class TestThresholdValidation:
    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            ReservoirLedger(soft_warn_uL=100.0, hard_stop_uL=500.0)

    def test_defaults_match_the_agreed_policy(self):
        assert DEFAULT_SOFT_WARN_UL == 1000.0
        assert DEFAULT_HARD_STOP_UL == 250.0


class TestDriverChokePoint:
    """Every dispense path — HT, campaign, manual, CLI — passes single_pump."""

    def test_depleted_stock_blocks_the_pump_despite_generous_res_vol(self):
        """The ledger is independent of ``res_vol``, and must stay that way.

        ``res_vol`` is the syringe volume declared to the pump firmware, padded
        by convention to exceed the command so the pump's own limit logic never
        trips; it says nothing about stock on hand.  Here a generous 10 mL sails
        past the firmware sanity check while the ledger still refuses, because
        only 300 µL is actually left.  If this test ever starts passing for the
        wrong reason — or someone seeds the ledger from ``res_vol`` — the hard
        stop protecting the plunger is gone.
        """
        mgr = create_mock_manager(config={})
        syr = mgr.get("syringe")
        led = ReservoirLedger()
        led.refill(0, 300.0)
        syr.reservoir_ledger = led

        with pytest.raises(SafetyError, match="hard stop"):
            syr.single_pump(res_vol=10, ID=0, rate=100.0, dispense_vol=100.0)

    def test_normal_dispense_debits_the_ledger(self):
        mgr = create_mock_manager(config={})
        syr = mgr.get("syringe")
        led = ReservoirLedger()
        led.refill(0, 5000.0)
        syr.reservoir_ledger = led

        syr.single_pump(res_vol=10, ID=0, rate=100.0, dispense_vol=250.0)
        assert led.remaining_uL(0) == 4750.0

    def test_per_pump_isolation(self):
        mgr = create_mock_manager(config={})
        syr = mgr.get("syringe")
        led = ReservoirLedger()
        led.refill(0, 5000.0)
        led.refill(1, 5000.0)
        syr.reservoir_ledger = led

        syr.single_pump(res_vol=10, ID=1, rate=100.0, dispense_vol=400.0)
        assert led.remaining_uL(0) == 5000.0     # untouched
        assert led.remaining_uL(1) == 4600.0

    def test_noop_dispense_does_not_debit(self):
        """A zeroed formulation component must not consume stock."""
        mgr = create_mock_manager(config={})
        syr = mgr.get("syringe")
        led = ReservoirLedger()
        led.refill(0, 5000.0)
        syr.reservoir_ledger = led

        syr.single_pump(res_vol=10, ID=0, rate=100.0, dispense_vol=0.0)
        assert led.remaining_uL(0) == 5000.0

    def test_no_ledger_attached_is_unchanged_behaviour(self):
        mgr = create_mock_manager(config={})
        syr = mgr.get("syringe")
        syr.single_pump(res_vol=10, ID=0, rate=100.0, dispense_vol=100.0)  # no raise


class TestAttach:
    """The one wiring path shared by the GUI and the headless CLI."""

    def test_attach_makes_the_interlock_live(self, store):
        mgr = create_mock_manager(config={})
        ledger = attach_reservoir_ledger(mgr, store, config={})
        assert ledger is not None
        assert mgr.get("syringe").reservoir_ledger is ledger

        ledger.refill(0, 300.0)
        with pytest.raises(SafetyError, match="hard stop"):
            mgr.get("syringe").single_pump(
                res_vol=10, ID=0, rate=100.0, dispense_vol=100.0
            )

    def test_attach_reads_thresholds_from_safety_config(self, store):
        mgr = create_mock_manager(config={})
        ledger = attach_reservoir_ledger(
            mgr, store,
            config={"reservoir_soft_warn_uL": 2000.0, "reservoir_hard_stop_uL": 500.0},
        )
        assert ledger.soft_warn_uL == 2000.0
        assert ledger.hard_stop_uL == 500.0

    def test_inverted_thresholds_fall_back_rather_than_crash(self, store):
        """A config typo must not leave the rig with no interlock at all."""
        mgr = create_mock_manager(config={})
        ledger = attach_reservoir_ledger(
            mgr, store,
            config={"reservoir_soft_warn_uL": 100.0, "reservoir_hard_stop_uL": 900.0},
        )
        assert ledger.soft_warn_uL == DEFAULT_SOFT_WARN_UL
        assert ledger.hard_stop_uL == DEFAULT_HARD_STOP_UL

    def test_levels_survive_reattachment(self, store):
        """Stock is a physical fact — it must outlive the process that declared it."""
        mgr = create_mock_manager(config={})
        attach_reservoir_ledger(mgr, store, config={}).refill(1, 4000.0)

        mgr2 = create_mock_manager(config={})
        assert attach_reservoir_ledger(mgr2, store, config={}).remaining_uL(1) == 4000.0

    def test_soft_warn_raises_a_durable_alert(self, store):
        mgr = create_mock_manager(config={})
        ledger = attach_reservoir_ledger(mgr, store, config={})
        ledger.refill(0, 1100.0)

        mgr.get("syringe").single_pump(res_vol=10, ID=0, rate=100.0, dispense_vol=200.0)

        alerts = [a for a in store.query_alerts() if a["kind"] == "reservoir"]
        assert len(alerts) == 1
        assert "low" in alerts[0]["message"]

    def test_no_syringe_is_not_an_error(self, store):
        class _Empty:
            def get(self, name):
                raise KeyError(name)

        assert attach_reservoir_ledger(_Empty(), store, config={}) is None
