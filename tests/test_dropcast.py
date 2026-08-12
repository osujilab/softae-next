"""Tests for the legacy-derived drop-cast sweep loop (dropcast.py)."""

from __future__ import annotations

import pytest

from softae.config.loader import pico_for_channel
from softae.core.dropcast import (
    DropcastFormulation,
    DropcastPreflightError,
    build_dropcast_sweep_workflow,
    preflight_dropcast,
    run_dropcast_sweep,
)
from softae.drivers.mock_factory import create_mock_manager

WELLS = (21, 22, 23, 24)
PCB = "SoftAE_EIS_4Stripe"


def _formulation(**over) -> DropcastFormulation:
    base = dict(ids=(0, 1, 2), vols=(0.1, 0.1, 0.1), disp_rate=1000.0, time_scale=0.0)
    base.update(over)
    return DropcastFormulation(**base)


# ── Formulation validation ───────────────────────────────────────────────────

def test_ids_vols_mismatch_rejected():
    with pytest.raises(ValueError):
        DropcastFormulation(ids=(0, 1), vols=(0.1,))


def test_deadvols_default_to_zeros():
    f = _formulation()
    assert f.deadvols == (0.0, 0.0, 0.0)


# ── Workflow shape ───────────────────────────────────────────────────────────

def test_sweep_has_prime_then_one_cast_per_well_then_flush():
    wf = build_dropcast_sweep_workflow(WELLS, _formulation(), pcb_name=PCB)
    names = [s.name for s in wf.setup]
    assert names[0] == "startup_flush"
    assert [n for n in names if n.startswith("dropcast_ch")] == [
        f"dropcast_ch{ch}" for ch in WELLS
    ]
    assert wf.teardown[0].name == "final_flush"


def test_wells_21_to_24_are_one_row_stripe():
    # On the 8x4 board, channels 21-24 share a row: same Y, X steps by spacing.
    wf = build_dropcast_sweep_workflow(WELLS, _formulation(), pcb_name=PCB)
    casts = [s for s in wf.setup if s.name.startswith("dropcast_ch")]
    ys = {c.params["y"] for c in casts}
    xs = [c.params["x"] for c in casts]
    assert len(ys) == 1  # all same row
    assert xs == sorted(xs, reverse=True)  # X decreases across the row


def test_no_eis_by_default_and_eis_routes_to_pico2_when_enabled():
    wf_no = build_dropcast_sweep_workflow(WELLS, _formulation(), pcb_name=PCB)
    assert not any(s.name.startswith("measure_eis") for s in wf_no.setup)

    wf_eis = build_dropcast_sweep_workflow(
        WELLS, _formulation(), pcb_name=PCB, measure_eis=True
    )
    for ch in WELLS:
        m = next(s for s in wf_eis.setup if s.name == f"measure_eis_ch{ch}")
        assert m.instrument == pico_for_channel(ch)  # 21-24 -> pico2


def test_three_pump_vols_carried_through():
    wf = build_dropcast_sweep_workflow(WELLS, _formulation(), pcb_name=PCB)
    cast = next(s for s in wf.setup if s.name == "dropcast_ch21")
    assert cast.params["ids"] == [0, 1, 2]
    assert cast.params["vols"] == [0.1, 0.1, 0.1]


# ── End-to-end drive ─────────────────────────────────────────────────────────

@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


@pytest.mark.asyncio
async def test_run_dropcast_sweep_drives_all_wells(connected):
    events: list[dict] = []
    result = await run_dropcast_sweep(
        WELLS, _formulation(), manager=connected, pcb_name=PCB, on_event=events.append,
    )
    # prime + one cast per well executed.
    assert result.steps_run == 1 + len(WELLS)
    assert result.channels == WELLS
    # All three pumps actually dispensed (no literal '$var' leaked to hardware).
    for pid in (0, 1, 2):
        assert connected.get("syringe")._dispensed.get(pid, 0.0) > 0
    # Event stream announced the sweep and every step.
    assert any(e["type"] == "sweep_started" for e in events)
    assert sum(e["type"] == "step_start" for e in events) == 1 + len(WELLS)
    assert any(e["type"] == "sweep_finished" for e in events)


@pytest.mark.asyncio
async def test_electrode_positions_reported(connected):
    result = await run_dropcast_sweep(
        WELLS, _formulation(), manager=connected, pcb_name=PCB,
    )
    assert set(result.electrode_xy) == set(WELLS)
    # Well 21 sits at the dep1 origin X (row 5, col 0 of the 8x4 grid).
    assert result.electrode_xy[21][0] == pytest.approx(43.5, abs=0.01)


# ── Preflight safety gate ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preflight_passes_for_valid_sweep(connected):
    rep = preflight_dropcast(
        WELLS, _formulation(vols=(5.0, 5.0, 5.0)), manager=connected, pcb_name=PCB,
    )
    assert rep.ok
    assert not rep.errors


@pytest.mark.asyncio
async def test_preflight_flags_subresolution_stroke(connected):
    # 0.05 µL is below the reliable per-syringe threshold even for one syringe.
    rep = preflight_dropcast(
        (21,), _formulation(ids=(0,), vols=(0.05,)), manager=connected, pcb_name=PCB,
    )
    assert rep.ok  # a warning, not a blocking error
    assert any("reliable-dispense threshold" in w for w in rep.warnings)


@pytest.mark.asyncio
async def test_preflight_blocks_bad_rate(connected):
    rep = preflight_dropcast(
        WELLS, _formulation(disp_rate=5000.0), manager=connected, pcb_name=PCB,
    )
    assert not rep.ok
    assert any("disp_rate" in e for e in rep.errors)


@pytest.mark.asyncio
async def test_preflight_blocks_out_of_range_channel(connected):
    rep = preflight_dropcast(
        (999,), _formulation(), manager=connected, pcb_name=PCB,
    )
    assert not rep.ok
    assert any("outside" in e and "grid" in e for e in rep.errors)


@pytest.mark.asyncio
async def test_dry_run_executes_no_motion(connected):
    before = dict(connected.get("syringe")._dispensed)
    result = await run_dropcast_sweep(
        WELLS, _formulation(), manager=connected, pcb_name=PCB, dry_run=True,
    )
    assert result.executed is False
    assert result.steps_run == 0
    # No dispense happened.
    assert dict(connected.get("syringe")._dispensed) == before


@pytest.mark.asyncio
async def test_failing_preflight_blocks_execution(connected):
    with pytest.raises(DropcastPreflightError):
        await run_dropcast_sweep(
            WELLS, _formulation(disp_rate=5000.0), manager=connected, pcb_name=PCB,
        )


@pytest.mark.asyncio
async def test_confirm_fn_veto_aborts(connected):
    before = dict(connected.get("syringe")._dispensed)
    result = await run_dropcast_sweep(
        WELLS, _formulation(), manager=connected, pcb_name=PCB,
        confirm_fn=lambda report: False,
    )
    assert result.executed is False
    assert dict(connected.get("syringe")._dispensed) == before
