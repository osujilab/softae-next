"""Smoke tests for the mock instrument suite and InstrumentManager."""

import asyncio

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.errors import InstrumentError, SafetyError
from softae.server.manager import InstrumentManager


@pytest.fixture
def manager() -> InstrumentManager:
    return create_mock_manager(config={})


@pytest.mark.asyncio
async def test_connect_all(manager: InstrumentManager):
    results = await manager.connect_all()
    assert all(results.values()), f"Some instruments failed to connect: {results}"


@pytest.mark.asyncio
async def test_status_all(manager: InstrumentManager):
    await manager.connect_all()
    statuses = manager.status_all()
    assert len(statuses) > 0
    for name, s in statuses.items():
        assert s["connected"] is True, f"{name} not connected"


@pytest.mark.asyncio
async def test_stage_move(manager: InstrumentManager):
    await manager.connect_all()
    stage = manager.get("stage")
    stage.move_to(10, 20)
    pos = stage.live_position()
    # Mock adds tiny noise, so check approximate
    assert abs(float(pos[0]) - 10) < 0.1
    assert abs(float(pos[1]) - 20) < 0.1


@pytest.mark.asyncio
async def test_temp_setpoint(manager: InstrumentManager):
    await manager.connect_all()
    tc = manager.get("temp_controller")
    tc.write_sp(50, print_flag=0)
    assert tc.get_sp() == 50.0


@pytest.mark.asyncio
async def test_eis_synthetic(manager: InstrumentManager):
    await manager.connect_all()
    pico = manager.get("pico1")
    curves = pico.sendscript_getdata("fake.mscr", "fake_out", 1)
    data = pico.eis_extractdata(curves)
    # Unified contract (mirrors AsyncESPico): list of five 1-D arrays
    # [f, |Z|, phase, Z', -Z'']
    assert len(data) == 5
    assert all(col.ndim == 1 for col in data)
    assert len(data[0]) > 0
    assert all(len(col) == len(data[0]) for col in data)


@pytest.mark.asyncio
async def test_disconnect_all(manager: InstrumentManager):
    await manager.connect_all()
    await manager.disconnect_all()
    for s in manager.list_instruments():
        assert s["connected"] is False


# --- acquire_multiple tests ---------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_multiple_two_instruments(manager: InstrumentManager):
    await manager.connect_all()
    async with manager.acquire_multiple("stage", "syringe") as insts:
        assert set(insts.keys()) == {"stage", "syringe"}
        insts["stage"].move_to(5, 10)
        pos = insts["stage"].live_position()
        assert abs(float(pos[0]) - 5) < 0.1
        insts["syringe"].single_pump(1.0, 0, 0.5, 0.1)


@pytest.mark.asyncio
async def test_acquire_multiple_sorted_order(manager: InstrumentManager):
    await manager.connect_all()
    # Provide names in reverse‐alpha order; should still acquire without deadlock
    async with manager.acquire_multiple("temp_controller", "syringe", "stage") as insts:
        assert set(insts.keys()) == {"stage", "syringe", "temp_controller"}
        insts["stage"].move_to(1, 1)
        insts["syringe"].single_pump(1.0, 0, 0.5, 0.1)
        insts["temp_controller"].write_sp(30, print_flag=0)


@pytest.mark.asyncio
async def test_acquire_multiple_unknown_instrument(manager: InstrumentManager):
    await manager.connect_all()
    with pytest.raises(InstrumentError):
        async with manager.acquire_multiple("stage", "nonexistent"):
            pass


@pytest.mark.asyncio
async def test_acquire_multiple_disconnected(manager: InstrumentManager):
    await manager.connect_all()
    await manager.disconnect("syringe")
    with pytest.raises(InstrumentError):
        async with manager.acquire_multiple("stage", "syringe"):
            pass


@pytest.mark.asyncio
async def test_acquire_multiple_concurrent_no_deadlock(manager: InstrumentManager):
    await manager.connect_all()

    async def task_a():
        async with manager.acquire_multiple("stage", "syringe"):
            await asyncio.sleep(0.01)

    async def task_b():
        async with manager.acquire_multiple("syringe", "stage"):
            await asyncio.sleep(0.01)

    # Both tasks request overlapping instruments in different order;
    # sorted acquisition prevents deadlock.
    await asyncio.wait_for(
        asyncio.gather(task_a(), task_b()),
        timeout=5,
    )


@pytest.mark.asyncio
async def test_acquire_multiple_single_instrument(manager: InstrumentManager):
    await manager.connect_all()
    async with manager.acquire_multiple("stage") as insts:
        assert list(insts.keys()) == ["stage"]
        insts["stage"].move_to(7, 3)
        pos = insts["stage"].live_position()
        assert abs(float(pos[0]) - 7) < 0.1


@pytest.mark.asyncio
async def test_acquire_multiple_releases_on_exception(manager: InstrumentManager):
    await manager.connect_all()
    with pytest.raises(RuntimeError):
        async with manager.acquire_multiple("stage", "syringe"):
            raise RuntimeError("deliberate")
    # Locks should be released — re-acquiring must succeed
    async with manager.acquire_multiple("stage", "syringe") as insts:
        assert "stage" in insts
        assert "syringe" in insts

# --- anneal tests -------------------------------------------------------------


@pytest.mark.parametrize("ramp_rate", [None, 10.0])
@pytest.mark.asyncio
async def test_anneal(manager: InstrumentManager, ramp_rate):
    """Anneal restores original SP whether or not a ramp_rate is given."""
    await manager.connect_all()
    tc = manager.get("temp_controller")
    tc.write_sp(25.0, print_flag=0)
    kwargs = {"target_temp_C": 50.0, "hold_time_s": 0, "tolerance": 2.0}
    if ramp_rate is not None:
        kwargs["ramp_rate"] = ramp_rate
    tc.anneal(**kwargs)
    assert tc.get_sp() == pytest.approx(25.0)


@pytest.mark.asyncio
async def test_anneal_restores_sp_on_error(manager: InstrumentManager):
    await manager.connect_all()
    tc = manager.get("temp_controller")
    tc.write_sp(20.0, print_flag=0)
    # Patch wait() to raise so we exercise the finally block
    original_wait = tc.wait
    def failing_wait(*a, **kw):
        raise RuntimeError("simulated hardware error")
    tc.wait = failing_wait
    try:
        with pytest.raises(RuntimeError):
            tc.anneal(target_temp_C=80.0, hold_time_s=0)
        assert tc.get_sp() == pytest.approx(20.0)
    finally:
        tc.wait = original_wait


# --- Safety interlock tests (A1) ──────────────────────────────────────────────


@pytest.mark.parametrize("sp,match", [(999.0, "exceeds max"), (-10.0, "below min")])
@pytest.mark.asyncio
async def test_temp_setpoint_safety(manager: InstrumentManager, sp, match):
    await manager.connect_all()
    tc = manager.get("temp_controller")
    with pytest.raises(SafetyError, match=match):
        tc.write_sp(sp)


@pytest.mark.parametrize("x,y,match", [(999, 0, "X="), (0, 999, "Y="), (None, None, None)])
@pytest.mark.asyncio
async def test_stage_move_bounds(manager: InstrumentManager, x, y, match):
    """Out-of-bounds moves raise SafetyError; valid moves succeed."""
    await manager.connect_all()
    stage = manager.get("stage")
    if match is not None:
        with pytest.raises(SafetyError, match=match):
            stage.move_to(x, y)
    else:
        stage.move_to(5, 5)
        pos = stage.live_position()
        assert abs(float(pos[0]) - 5) < 0.1


@pytest.mark.parametrize("rate,match", [(99999, "exceeds max"), (0.0001, "below min")])
@pytest.mark.asyncio
async def test_syringe_rate_safety(manager: InstrumentManager, rate, match):
    await manager.connect_all()
    syr = manager.get("syringe")
    with pytest.raises(SafetyError, match=match):
        syr.single_pump(10.0, 0, rate, 1.0)


@pytest.mark.parametrize("zero_vol", [0.0, 0, -1.0])
@pytest.mark.asyncio
async def test_syringe_zero_volume_is_noop(manager: InstrumentManager, zero_vol):
    """A 0 µL command is a no-op: no dispense and no min-rate SafetyError.

    Even paired with a 0 rate (as a zeroed formulation component yields), the
    call must return quietly rather than trip the ``min_rate`` limit.
    """
    await manager.connect_all()
    syr = manager.get("syringe")
    before = dict(syr._dispensed)
    syr.single_pump(10.0, 0, 0.0, zero_vol)  # rate 0.0 would otherwise raise
    assert syr._dispensed == before           # nothing changed


@pytest.mark.asyncio
async def test_syringe_volume_exceeds_declared_syringe_volume(manager: InstrumentManager):
    """The pump-firmware sanity check on ``res_vol``.

    ``res_vol`` is the syringe volume declared to the pump, not the stock on
    hand — see ``ReservoirLedger`` for the actual consumables interlock.
    """
    await manager.connect_all()
    syr = manager.get("syringe")
    with pytest.raises(SafetyError, match="exceeds declared syringe volume"):
        syr.single_pump(1.0, 0, 100, 5000)  # 5000 µL > 1 mL = 1000 µL


@pytest.mark.asyncio
async def test_stage_move_by_out_of_bounds(manager: InstrumentManager):
    """Relative move that would leave the stage out of bounds raises SafetyError."""
    await manager.connect_all()
    stage = manager.get("stage")
    stage.move_to(0, 0)
    with pytest.raises(SafetyError):
        stage.move_by(999, 0)



# -- Parallel-syringe volume split ----------------------------------------

@pytest.mark.asyncio
async def test_syringe_parallel_halves_volume():
    """When parallel_syringes=2, each syringe receives half the volume."""
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={"parallel_syringes": 2})
    await syr.connect()
    syr.single_pump(10.0, 0, 100, 200)  # request 200 µL
    assert syr._dispensed[0] == pytest.approx(100.0)  # half


@pytest.mark.asyncio
async def test_syringe_parallel_default_no_split():
    """Default parallel_syringes=1 dispenses the full volume."""
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={})
    await syr.connect()
    syr.single_pump(10.0, 0, 100, 200)
    assert syr._dispensed[0] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_syringe_effective_per_syringe_volume_helper():
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={"parallel_syringes": 1})
    await syr.connect()
    assert syr.effective_per_syringe_volume(100.0) == pytest.approx(100.0)
    syr.set_parallel_syringes(2)
    assert syr.effective_per_syringe_volume(100.0) == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_syringe_per_pump_counts_are_independent():
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={"parallel_syringes": 1, "parallel_syringes_pump1": 2})
    await syr.connect()
    status = syr.status()
    assert status["parallel_syringes_by_pump"][0] == 1
    assert status["parallel_syringes_by_pump"][1] == 2
    assert syr.effective_per_syringe_volume(100.0, pump_id=1) == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_syringe_set_parallel_syringes_rejects_invalid():
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={})
    await syr.connect()
    with pytest.raises(ValueError):
        syr.set_parallel_syringes(0)


@pytest.mark.asyncio
async def test_syringe_single_pump_uses_helper():
    from softae.drivers.mock_syringe import MockSyringe

    syr = MockSyringe(config={"parallel_syringes": 2})
    await syr.connect()
    original = syr.effective_per_syringe_volume
    syr.effective_per_syringe_volume = lambda *args, **kwargs: 12.5
    try:
        syr.single_pump(10.0, 0, 100.0, 200.0)
        assert syr._dispensed[0] == pytest.approx(12.5)
    finally:
        syr.effective_per_syringe_volume = original
