"""Vertical-slice tests for the liquid_handler coordinator instrument.

Covers the three links of the reconciliation chain:
1. geometry — channel → electrode stage position;
2. the composite routines drive stage + syringe correctly against mocks;
3. the routines run end-to-end through the WorkflowExecutor with a per-channel
   electrode position injected into the step params (as the HT tab does).
"""

from __future__ import annotations

import pytest

from softae.core.geometry import electrode_positions, electrode_xy_for_channel
from softae.drivers.mock_factory import create_mock_manager
from softae.workflows.workflow_executor import WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

# All waits scaled to ~0 so routines complete instantly.
FAST = 0.0

PCB_4x4 = {"grid": [4, 4], "spacing_mm": [15, 15]}


# ── Geometry ─────────────────────────────────────────────────────────────────

def test_channel_1_is_the_origin():
    x, y = electrode_xy_for_channel(PCB_4x4, 1, origin_x=43.5, origin_y=50.0)
    assert (x, y) == (43.5, 50.0)


def test_row_major_decreasing_coords():
    # ch2 is one column right-to-left (X decreases); ch5 starts the next row
    # (Y decreases, X back to origin).
    assert electrode_xy_for_channel(PCB_4x4, 2, 43.5, 50.0) == (28.5, 50.0)
    assert electrode_xy_for_channel(PCB_4x4, 5, 43.5, 50.0) == (43.5, 35.0)


def test_channel_out_of_range_raises():
    with pytest.raises(ValueError):
        electrode_xy_for_channel(PCB_4x4, 17, 43.5, 50.0)


def test_positions_length_matches_grid():
    xs, ys = electrode_positions(PCB_4x4, 0.0, 0.0)
    assert len(xs) == len(ys) == 16


# ── Coordinator routines against mocks ───────────────────────────────────────

@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    return mgr


@pytest.mark.asyncio
async def test_liquid_handler_registered(connected):
    assert "liquid_handler" in connected.names
    lh = connected.get("liquid_handler")
    assert lh.manager is connected


@pytest.mark.asyncio
async def test_startup_flush_dispenses_and_ends_at_wick(connected):
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    result = await lh.execute(
        "startup_flush",
        flush_x=-50.0, flush_y=50.0, wick_x=-50.0, wick_y=-25.0,
        disp_rate=200, disp_vol=80, ids=[0, 1], time_scale=FAST,
    )
    assert result["flushed_ids"] == [0, 1]
    # Both pumps dispensed
    assert syringe._dispensed.get(0, 0) > 0
    assert syringe._dispensed.get(1, 0) > 0
    # Stage finished at the wick position
    s = connected.get("stage").status()
    assert s["x"] == pytest.approx(-50.0, abs=0.05)
    assert s["y"] == pytest.approx(-25.0, abs=0.05)


@pytest.mark.asyncio
async def test_single_drop_moves_to_electrode_then_wicks(connected):
    lh = connected.get("liquid_handler")
    ex, ey = electrode_xy_for_channel(PCB_4x4, 6, origin_x=43.5, origin_y=50.0)
    result = await lh.execute(
        "single_drop_simul",
        x=ex, y=ey, wick_x=-50.0, wick_y=-25.0,
        ids=[0, 1], disp_rate=75, vols=[21, 21], deadvols=[10, 30],
        elution_wait_s=240, time_scale=FAST,
    )
    assert result["electrode_xy"] == [ex, ey]
    # Ends at wick (drop happened in between, verified via dispense below)
    s = connected.get("stage").status()
    assert s["x"] == pytest.approx(-50.0, abs=0.05)
    assert s["y"] == pytest.approx(-25.0, abs=0.05)
    # vols + deadvols were commanded
    assert connected.get("syringe")._dispensed.get(0, 0) > 0


@pytest.mark.asyncio
async def test_single_drop_zero_volume_leaves_pump_alone(connected):
    """A zeroed component skips its pump entirely — no command, no min-rate error.

    Casting a two-of-three formulation where the middle stock is 0 µL must not
    trip the pump's min-rate limit (its proportional rate is also 0). The zero
    pump is left untouched; the other two dispense normally.
    """
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    commanded: list[int] = []
    orig = syringe.single_pump

    def spy(res_vol, ID, rate, dispense_vol):
        commanded.append(int(ID))
        return orig(res_vol=res_vol, ID=ID, rate=rate, dispense_vol=dispense_vol)

    syringe.single_pump = spy
    # disp_rate_total split by [10, 0, 30] → pump 1's rate is 0.0.
    result = await lh.execute(
        "single_drop_simul",
        x=0, y=0, wick_x=0, wick_y=0,
        ids=[0, 1, 2], disp_rate=75, vols=[10.0, 0.0, 30.0], deadvols=[0, 0, 0],
        disp_rate_total=100.0, elution_wait_s=0, time_scale=FAST,
    )
    assert result["electrode_xy"] == [0, 0]
    assert commanded == [0, 2]                       # pump 1 (0 µL) left alone
    assert syringe._dispensed.get(1, 0.0) == 0.0     # nothing dispensed from it
    assert syringe._dispensed.get(0, 0.0) > 0
    assert syringe._dispensed.get(2, 0.0) > 0


@pytest.mark.asyncio
async def test_single_drop_length_mismatch_raises(connected):
    lh = connected.get("liquid_handler")
    with pytest.raises(Exception):
        await lh.execute(
            "single_drop_simul",
            x=0, y=0, wick_x=0, wick_y=0,
            ids=[0, 1], disp_rate=75, vols=[21], deadvols=[10, 30],
            time_scale=FAST,
        )


@pytest.mark.asyncio
async def test_single_drop_per_pump_rates(connected):
    """Each pump extrudes at its own rate when disp_rates is supplied."""
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    calls: list[tuple[int, float]] = []
    orig = syringe.single_pump

    def spy(res_vol, ID, rate, dispense_vol):
        calls.append((int(ID), float(rate)))
        return orig(res_vol=res_vol, ID=ID, rate=rate, dispense_vol=dispense_vol)

    syringe.single_pump = spy
    await lh.execute(
        "single_drop_simul",
        x=0, y=0, wick_x=0, wick_y=0,
        ids=[0, 1], disp_rate=999, vols=[10, 30], deadvols=[0, 0],
        disp_rates=[25.0, 75.0], elution_wait_s=0, time_scale=FAST,
    )
    rate_by_id = dict(calls)
    # Per-pump rates win over the scalar disp_rate (999 must not appear).
    assert rate_by_id[0] == pytest.approx(25.0)
    assert rate_by_id[1] == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_single_drop_scalar_rate_still_works(connected):
    """Omitting disp_rates falls back to the scalar disp_rate for every pump."""
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    calls: list[float] = []
    orig = syringe.single_pump

    def spy(res_vol, ID, rate, dispense_vol):
        calls.append(float(rate))
        return orig(res_vol=res_vol, ID=ID, rate=rate, dispense_vol=dispense_vol)

    syringe.single_pump = spy
    await lh.execute(
        "single_drop_simul",
        x=0, y=0, wick_x=0, wick_y=0,
        ids=[0, 1], disp_rate=75, vols=[10, 30], deadvols=[0, 0],
        elution_wait_s=0, time_scale=FAST,
    )
    assert calls == [pytest.approx(75.0), pytest.approx(75.0)]


@pytest.mark.asyncio
async def test_single_drop_rates_length_mismatch_raises(connected):
    lh = connected.get("liquid_handler")
    with pytest.raises(Exception):
        await lh.execute(
            "single_drop_simul",
            x=0, y=0, wick_x=0, wick_y=0,
            ids=[0, 1], disp_rate=75, vols=[10, 30], deadvols=[0, 0],
            disp_rates=[25.0], time_scale=FAST,
        )


@pytest.mark.asyncio
async def test_single_drop_total_rate_splits_per_pump(connected):
    """disp_rate_total is split across pumps in proportion to volume (autonomous)."""
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    calls: list[tuple[int, float]] = []
    orig = syringe.single_pump

    def spy(res_vol, ID, rate, dispense_vol):
        calls.append((int(ID), float(rate)))
        return orig(res_vol=res_vol, ID=ID, rate=rate, dispense_vol=dispense_vol)

    syringe.single_pump = spy
    await lh.execute(
        "single_drop_simul",
        x=0, y=0, wick_x=0, wick_y=0,
        ids=[0, 1], disp_rate=999, vols=[10, 30], deadvols=[0, 0],
        disp_rate_total=100.0, settle_factor=2.0, elution_wait_s=240,
        time_scale=FAST,
    )
    rate_by_id = dict(calls)
    # 100 split by [10,30] → [25,75]; the scalar 999 must not appear.
    assert rate_by_id[0] == pytest.approx(25.0)
    assert rate_by_id[1] == pytest.approx(75.0)


@pytest.mark.asyncio
async def test_precondition_rate_total_splits_per_pump(connected):
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    calls: list[tuple[int, float]] = []
    orig = syringe.single_pump

    def spy(res_vol, ID, rate, dispense_vol):
        calls.append((int(ID), float(rate)))
        return orig(res_vol=res_vol, ID=ID, rate=rate, dispense_vol=dispense_vol)

    syringe.single_pump = spy
    await lh.execute(
        "precondition_flush",
        flush_x=-50.0, flush_y=50.0, wick_x=-50.0, wick_y=-25.0,
        ids=[0, 1], vol_list=[10.0, 30.0], rate_total=500.0,
        flush_factor=2.0, plug_ids=[0, 2], plug_rate=150.0, plug_vol=30.0,
        time_scale=FAST,
    )
    rate_by_id = dict(calls)  # preload dispenses come last → win over the plug
    # 500 split by [10,30] → [125,375]; pump 1 is preload-only (unambiguous).
    assert rate_by_id[1] == pytest.approx(375.0)
    assert rate_by_id[0] == pytest.approx(125.0)


@pytest.mark.asyncio
async def test_precondition_requires_rate_list_or_total(connected):
    lh = connected.get("liquid_handler")
    with pytest.raises(Exception):
        await lh.execute(
            "precondition_flush", flush_x=0, flush_y=0, wick_x=0, wick_y=0,
            ids=[0, 1], vol_list=[10.0, 30.0], time_scale=FAST)


# ── End-to-end through the executor (as the HT tab wires it) ──────────────────

@pytest.mark.asyncio
async def test_composite_workflow_runs_end_to_end(connected):
    """startup_flush (setup) → per-channel drop-cast with injected electrode xy."""
    channels = [1, 6]
    drop_steps = []
    for ch in channels:
        ex, ey = electrode_xy_for_channel(PCB_4x4, ch, origin_x=43.5, origin_y=50.0)
        drop_steps.append(
            WorkflowStep(
                name=f"deposit_drop_ch{ch}",
                instrument="liquid_handler",
                method="single_drop_simul",
                params={
                    "x": ex, "y": ey, "wick_x": -50.0, "wick_y": -25.0,
                    "ids": [0, 1], "disp_rate": 75, "vols": [21, 21],
                    "deadvols": [10, 30], "elution_wait_s": 5, "time_scale": FAST,
                },
                tags={"position": "electrode", "channel": str(ch)},
            )
        )

    wf = Workflow(
        name="composite_slice",
        setup=[
            WorkflowStep(
                name="startup_flush",
                instrument="liquid_handler",
                method="startup_flush",
                params={
                    "flush_x": -50.0, "flush_y": 50.0, "wick_x": -50.0,
                    "wick_y": -25.0, "disp_rate": 200, "disp_vol": 80,
                    "ids": [0, 1], "time_scale": FAST,
                },
            ),
            *drop_steps,
        ],
    )

    executor = WorkflowExecutor(connected)
    await executor.run(wf)

    # Every pump saw dispense volume; stage ended at wick after the last drop.
    assert connected.get("syringe")._dispensed.get(0, 0) > 0
    s = connected.get("stage").status()
    assert s["x"] == pytest.approx(-50.0, abs=0.05)


# ── New legacy ports: per-pump startup flush, precondition flush, star mix ────

@pytest.mark.asyncio
async def test_startup_flush_per_pump_volumes(connected):
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    result = await lh.execute(
        "startup_flush",
        flush_x=-50.0, flush_y=50.0, wick_x=-50.0, wick_y=-25.0,
        disp_rate=200, disp_vol=0,       # scalar ignored when disp_vols given
        ids=[0, 1], disp_vols=[10.0, 40.0], time_scale=FAST,
    )
    assert result["dispensed_uL"] == [10.0, 40.0]     # per-pump commanded volumes
    assert syringe._dispensed.get(1, 0) > syringe._dispensed.get(0, 0)


@pytest.mark.asyncio
async def test_startup_flush_length_mismatch_raises(connected):
    lh = connected.get("liquid_handler")
    with pytest.raises(Exception):
        await lh.execute(
            "startup_flush", flush_x=0, flush_y=0, wick_x=0, wick_y=0,
            disp_rate=200, disp_vol=0, ids=[0, 1], disp_vols=[10.0], time_scale=FAST)


@pytest.mark.asyncio
async def test_precondition_flush_plug_then_preload(connected):
    lh = connected.get("liquid_handler")
    syringe = connected.get("syringe")
    result = await lh.execute(
        "precondition_flush",
        flush_x=-50.0, flush_y=50.0, wick_x=-50.0, wick_y=-25.0,
        ids=[0, 1], rate_list=[75, 75], vol_list=[10.0, 20.0],
        flush_factor=2.0, plug_ids=[0, 2], plug_rate=150, plug_vol=30,
        time_scale=FAST,
    )
    # Pre-load is vol_list * flush_factor.
    assert result["preload_uL"] == [20.0, 40.0]
    # Plug pumps (0,2) and preload pumps (0,1) all dispensed.
    for pid in (0, 1, 2):
        assert syringe._dispensed.get(pid, 0) > 0
    # Ends at the wick.
    s = connected.get("stage").status()
    assert s["x"] == pytest.approx(-50.0, abs=0.05)
    assert s["y"] == pytest.approx(-25.0, abs=0.05)


@pytest.mark.asyncio
async def test_precondition_flush_length_mismatch_raises(connected):
    lh = connected.get("liquid_handler")
    with pytest.raises(Exception):
        await lh.execute(
            "precondition_flush", flush_x=0, flush_y=0, wick_x=0, wick_y=0,
            ids=[0, 1], rate_list=[75], vol_list=[10, 20], time_scale=FAST)


@pytest.mark.asyncio
async def test_star_mix_traces_star_and_retracts(connected):
    lh = connected.get("liquid_handler")
    result = await lh.execute(
        "star_mix", x=10.0, y=5.0, r_extent=1.5, n_points=6, dwell_s=0.0,
        time_scale=FAST,
    )
    assert result["center"] == [10.0, 5.0]
    assert result["vertices"] == 14   # (n_points+1) * 2
    # Returns to the drop centre and retracts the head.
    s = connected.get("stage").status()
    assert s["x"] == pytest.approx(10.0, abs=0.05)
    assert s["y"] == pytest.approx(5.0, abs=0.05)
    assert connected.get("syringe")._is_up is True
