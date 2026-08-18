"""``halt_pump`` — the pump stop, and the constants that keep it reachable.

The chain has no stop command. What stops it is a fresh program, and the park has
always sent one — expressed as a 0.001 µL dispense, which meant it travelled
through the dispense validator and the stock interlock. This file pins the three
properties that separate a halt from a dispense, plus the two undocumented
constants that a config edit could otherwise delete in silence.
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from softae.core.reservoir import ReservoirLedger
from softae.drivers.async_syringe import (
    HALT_RATE_UL_PER_MIN,
    HALT_SVOLUME_ML,
    HALT_VOLUME_UL,
)
from softae.drivers.contracts import PUMP_NOOP_VOLUME_UL
from softae.drivers.mock_syringe import MockSyringe
from softae.errors import SafetyError


@pytest.fixture()
def real_syringe():
    """An :class:`AsyncSyringe` connected to a mocked VISA resource."""
    pv = types.ModuleType("pyvisa")
    mock_inst = MagicMock()
    mock_rm = MagicMock()
    mock_rm.open_resource.return_value = mock_inst
    pv.ResourceManager = MagicMock(return_value=mock_rm)

    with patch.dict("sys.modules", {"pyvisa": pv}):
        from softae.drivers.async_syringe import AsyncSyringe

        syr = AsyncSyringe(config={"diameter": 14.4, "min_rate": 0.05})
        import asyncio

        asyncio.run(syr.connect())
    return syr, mock_inst


def _writes(mock_inst) -> list[str]:
    return [call.args[0] for call in mock_inst.write.call_args_list]


# ── The bytes ────────────────────────────────────────────────────────────────

class TestTheHaltWritesTheSameProgram:
    def test_it_writes_the_five_command_program(self, real_syringe):
        syr, inst = real_syringe
        syr.halt_pump(1)

        assert _writes(inst) == [
            "1 svolume 1000 ml",
            "1diameter 14.4",
            "1irate 0.1 ul/min",
            "1tvolume 0.001 ul",
            "1irun",
        ]

    def test_it_ends_on_irun_because_that_is_what_countermands(self, real_syringe):
        """The override *is* the fresh trigger; a program without it stops nothing."""
        syr, inst = real_syringe
        syr.halt_pump(0)
        assert _writes(inst)[-1] == "0irun"

    def test_the_bytes_match_what_the_park_used_to_send_as_a_dispense(
        self, real_syringe
    ):
        """Same program, different path. Only the path was broken."""
        syr, inst = real_syringe
        syr.halt_pump(2)
        halt_writes = _writes(inst)

        inst.write.reset_mock()
        syr.single_pump(HALT_SVOLUME_ML, 2, HALT_RATE_UL_PER_MIN, HALT_VOLUME_UL)
        assert _writes(inst) == halt_writes


# ── The constants that could silently delete the halt ────────────────────────

class TestTheUndocumentedCouplings:
    def test_the_halt_rate_no_longer_depends_on_min_rate(self, real_syringe):
        """A ``min_rate`` above the halt rate used to convert every park into
        three ``SafetyError``\\ s. ``halt_pump`` does not validate, so it cannot."""
        syr, inst = real_syringe
        syr._min_rate = 10.0          # far above HALT_RATE_UL_PER_MIN
        syr.halt_pump(0)              # must not raise
        assert _writes(inst)[-1] == "0irun"

    def test_the_halt_volume_no_longer_depends_on_the_noop_floor(self, real_syringe):
        """``_is_noop_pump_command`` uses ``<=``, so raising the floor off zero
        used to make the halt a logged no-op with nothing reporting a failure."""
        syr, inst = real_syringe
        with patch("softae.drivers.contracts.PUMP_NOOP_VOLUME_UL", 1.0):
            syr.halt_pump(0)
        assert _writes(inst)[-1] == "0irun"

    def test_the_shipped_floor_is_still_exactly_zero(self):
        """Records *why* the old path survived at all, so a change is deliberate."""
        assert PUMP_NOOP_VOLUME_UL == 0.0
        assert HALT_VOLUME_UL > PUMP_NOOP_VOLUME_UL

    def test_the_halt_program_is_declared_with_an_integer_svolume(self):
        """``f"{ID} svolume {HALT_SVOLUME_ML} ml"`` — 1000.0 would change the bytes."""
        assert isinstance(HALT_SVOLUME_ML, int)


# ── The interlock, and the kwarg that was rejected ───────────────────────────

class TestTheLedgerCannotRefuseAHalt:
    def test_a_halt_at_the_hard_stop_is_not_refused(self, real_syringe):
        syr, inst = real_syringe
        ledger = ReservoirLedger(soft_warn_uL=500.0, hard_stop_uL=200.0)
        ledger.refill(0, 200.0)
        syr.reservoir_ledger = ledger

        with pytest.raises(SafetyError):      # the dispense path still refuses
            syr.single_pump(1000, 0, 0.1, 0.001)

        syr.halt_pump(0)                      # the halt path does not
        assert _writes(inst)[-1] == "0irun"

    def test_a_halt_debits_nothing(self, real_syringe):
        syr, _ = real_syringe
        ledger = ReservoirLedger(soft_warn_uL=5_000.0, hard_stop_uL=200.0)
        ledger.refill(0, 1_000.0)
        syr.reservoir_ledger = ledger

        syr.halt_pump(0)
        assert ledger.remaining_uL(0) == 1_000.0

    def test_single_pump_has_no_bypass_ledger_kwarg(self, real_syringe):
        """The rejected design, pinned.

        A bypass flag would punch a hole in an interlock every other caller — HT,
        campaign, manual, CLI — depends on, to serve one caller. The separate
        method is also the honest signature: a halt takes no volume and no rate.
        """
        syr, _ = real_syringe
        with pytest.raises(TypeError):
            syr.single_pump(1000, 0, 0.1, 0.001, bypass_ledger=True)

    def test_halt_pump_takes_only_a_pump_id(self, real_syringe):
        syr, _ = real_syringe
        with pytest.raises(TypeError):
            syr.halt_pump(0, 0.1)


# ── Mutual exclusion ─────────────────────────────────────────────────────────

class TestTheVisaWritesAreSerialised:
    def test_two_threads_cannot_interleave_a_pump_program(self, real_syringe):
        """A park lands mid-block otherwise: one caller's ``tvolume`` against the
        other's ``irate``, then whichever ``irun`` arrives first."""
        syr, inst = real_syringe
        seen: list[str] = []
        inst.write.side_effect = lambda cmd: seen.append(cmd)

        def dispense():
            syr.single_pump(1000, 0, 5.0, 10.0)

        t = threading.Thread(target=dispense)
        t.start()
        syr.halt_pump(1)
        t.join()

        # Each program is a contiguous run of five writes for one pump ID.
        assert len(seen) == 10
        first_ids = [cmd[0] for cmd in seen[:5]]
        second_ids = [cmd[0] for cmd in seen[5:]]
        assert len(set(first_ids)) == 1
        assert len(set(second_ids)) == 1
        assert set(first_ids) != set(second_ids)

    def test_the_lock_is_a_threading_primitive_not_an_asyncio_one(self, real_syringe):
        """The contention is between OS threads — a ``QThread``, the event loop
        thread, and the shared I/O pool. An ``asyncio.Lock`` cannot exclude any
        of them from each other."""
        syr, _ = real_syringe
        assert isinstance(syr._visa_lock, type(threading.RLock()))


# ── Mock/real parity ─────────────────────────────────────────────────────────

class TestMockMirrorsReal:
    def test_the_mock_records_halts_without_dispensing(self):
        syr = MockSyringe()
        syr.halt_pump(2)
        assert syr._halted == [2]
        assert syr._dispensed[2] == 0.0

    def test_the_mock_halt_is_not_refusable_either(self):
        """A mock that could refuse would let the fix pass against a broken driver."""
        syr = MockSyringe(config={"min_rate": 0.05})
        ledger = ReservoirLedger(soft_warn_uL=500.0, hard_stop_uL=200.0)
        ledger.refill(0, 200.0)
        syr.reservoir_ledger = ledger

        with pytest.raises(SafetyError):
            syr.single_pump(1000, 0, 0.1, 0.001)

        syr.halt_pump(0)
        assert syr._halted == [0]
        assert ledger.remaining_uL(0) == 200.0

    def test_the_mock_halt_ignores_injected_faults_of_the_dispense_path(self):
        """Fault injection targets ``single_pump``; a halt is a different command."""
        syr = MockSyringe(config={"fail_next_n": 3})
        syr.halt_pump(0)
        assert syr._halted == [0]
