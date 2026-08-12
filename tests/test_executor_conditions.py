"""The EIS autoroute captures temp/humidity conditions at measurement time.

Routing now lives in :class:`softae.analysis.eis.router.EISResultRouter`
(Tier-1 extraction); these tests drive it through the executor's
``_route_result`` seam so router matching and the executor-owned sweep counter
are both exercised. The E6 defaulting semantics asserted here are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.workflows.workflow_executor import WorkflowExecutor
from softae.workflows.workflow_model import WorkflowStep


def _raw_eis(npts: int = 12) -> np.ndarray:
    """A 5-column [f, |Z|, phase, Z', -Z''] array as the pico driver emits."""
    f = np.geomspace(1e5, 1.0, npts)
    zreal = np.full(npts, 980.0)
    zimg_neg = np.full(npts, 170.0)  # -Z''
    zmag = np.hypot(zreal, zimg_neg)
    phase = np.degrees(np.arctan2(-zimg_neg, zreal))
    return np.column_stack([f, zmag, phase, zreal, zimg_neg])


@pytest.mark.asyncio
async def test_autoroute_records_conditions(tmp_path):
    manager = create_mock_manager()
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("ht_experiment")

    ex = WorkflowExecutor(manager, data_store=store, run_id=run_id)
    step = WorkflowStep(
        name="eis_ch3",
        instrument="pico0",
        method="eis_extractdata",
        params={"channel": 3},
    )

    await ex._route_result(step, _raw_eis())

    measurements = store.query_measurements(run_id=run_id)
    assert len(measurements) == 1
    conds = store.query_conditions(
        measurement_id=measurements[0]["measurement_id"], stage="measurement"
    )
    assert len(conds) == 1
    c = conds[0]
    # Mock drivers seed finite values for every SP/PV.
    for key in ("stage_temp_sp_C", "chamber_air_C", "stage_temp_pv_C", "rh_sp_pct", "rh_pv_pct"):
        assert c[key] is not None

    store.close()


@pytest.mark.asyncio
async def test_autoroute_numbers_the_acquisition_sequence(tmp_path):
    """`sweep_order` is counted from what was recorded, not from the plan (E6/§6).

    Position within the sweep is drift metadata, so it has to describe the order the
    instrument actually saw. A planner-supplied index would describe the intended
    order, which is the wrong one after a retry, a skip or a channel replay -- and
    those are precisely the runs where drift is worth looking for.
    """
    manager = create_mock_manager()
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("ht_experiment")
    ex = WorkflowExecutor(manager, data_store=store, run_id=run_id)

    for ch in (3, 7, 11):
        step = WorkflowStep(name=f"eis_ch{ch}", instrument="pico0",
                            method="eis_extractdata", params={"channel": ch})
        await ex._route_result(step, _raw_eis())

    rows = store._conn.execute(
        "SELECT channel, sweep_order, re_connection, re_contact_verified "
        "FROM measurements WHERE run_id = ? ORDER BY measurement_id", (run_id,)
    ).fetchall()
    assert [r[1] for r in rows] == [1, 2, 3]

    # A sample's loop is closed by the cast film -- a fact of the workflow, not a
    # guess -- so defaulting it silences a warning that would otherwise fire on every
    # spectrum the rig produces. Contact is NOT defaulted with it (R26).
    assert all(r[2] == "bridged_by_sample" for r in rows)
    assert all(r[3] == 0 for r in rows)

    store.close()


@pytest.mark.asyncio
async def test_a_commissioning_role_does_not_inherit_the_samples_re_default(tmp_path):
    """A blank has no film, so nothing bridges the stripes for it.

    Defaulting `bridged_by_sample` by role would assert a closed loop for exactly the
    measurements where F13's open-loop diagnostic matters most.
    """
    manager = create_mock_manager()
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("ht_experiment")
    ex = WorkflowExecutor(manager, data_store=store, run_id=run_id)

    step = WorkflowStep(name="eis_ch1", instrument="pico0",
                        method="eis_extractdata", params={"channel": 1},
                        tags={"role": "blank_open", "fixture_id": "mux16"})
    await ex._route_result(step, _raw_eis())

    row = store._conn.execute(
        "SELECT role, re_connection FROM measurements WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert row[0] == "blank_open"
    assert row[1] == "unverified"

    store.close()
