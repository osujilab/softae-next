"""The sample-identity spine (T2.6, spec Tier 2 component 6).

One physical sample is described by three rows written by three different layers
at three different moments — the formulation the wiring records at build time,
the occupancy the loop records after the cast, and the measurement the router
records after the spectrum — plus a payload file written beside the last of
them. Nothing they share identifies the sample: ``(run_id, channel)`` is
board-relative and reused, and a re-cast well produces a second formulation the
first one's spectra would silently join to.

``sample_uuid`` is that missing key. These tests pin the two properties that make
it worth having:

1. **It is the same value in all four places** for a well that was actually cast,
   end to end through a real campaign.
2. **It is absent, not blank, everywhere else.** A step with no tag yields no
   ``attrs`` key and a NULL column — because an anonymous identity would join to
   every other unidentified sample, which is worse than no identity at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import xarray as xr

from softae.config import loader
from softae.core.autonomous_wiring import (
    SAMPLE_UUID_PREFIX,
    CampaignSpec,
    build_placement_workflow,
    mint_sample_uuid,
    mint_sample_uuids,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.core.task_catalog import TaskCatalog
from softae.drivers.mock_factory import create_mock_manager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="identity_campaign",
        channels=(21, 22, 23, 24),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=4,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
def catalog() -> TaskCatalog:
    return TaskCatalog.load_toml(loader.tasks_toml_path())


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


def _column_names(store: DataStore, table: str) -> set[str]:
    return {
        row[1]
        for row in store._conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _occupancy_uuids(store: DataStore) -> dict[tuple[int, int], str | None]:
    return {
        (int(r[0]), int(r[1])): r[2]
        for r in store._conn.execute(
            "SELECT board_id, electrode, sample_uuid FROM electrode_occupancy"
        ).fetchall()
    }


def _formulation_uuids(store: DataStore, run_id: str) -> dict[int, str | None]:
    """``{channel: sample_uuid}`` for a run's formulation rows, latest per channel."""
    rows = store._conn.execute(
        "SELECT channel, sample_uuid FROM formulations WHERE run_id = ? "
        "ORDER BY formulation_id",
        (run_id,),
    ).fetchall()
    return {int(r[0]): r[1] for r in rows}


# ── Minting ──────────────────────────────────────────────────────────────────


def test_mint_sample_uuid_carries_the_kind_in_the_value():
    minted = mint_sample_uuid()
    assert minted.startswith(SAMPLE_UUID_PREFIX)
    # The suffix is a real uuid4, not a counter — two rigs writing into one
    # dataset must not be able to collide.
    assert len(minted) == len(SAMPLE_UUID_PREFIX) + 36


def test_mint_sample_uuids_gives_every_well_its_own_identity():
    """One well is one sample, even when four wells get the same formulation."""
    minted = mint_sample_uuids([21, 22, 23, 24])
    assert set(minted) == {21, 22, 23, 24}
    assert len(set(minted.values())) == 4


# ── Storage columns + migration ──────────────────────────────────────────────


def test_sample_uuid_exists_on_all_three_spine_tables(tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    for table in ("formulations", "electrode_occupancy", "measurements"):
        assert "sample_uuid" in _column_names(store, table), table
    store.close()


def test_the_migration_is_idempotent_across_reopens(tmp_path: Path):
    """Re-opening a store must not attempt a duplicate ALTER (it would raise)."""
    project = tmp_path / "proj"
    for _ in range(3):
        store = DataStore(project)
        cols = _column_names(store, "formulations")
        assert [c for c in cols if c == "sample_uuid"] == ["sample_uuid"]
        store.close()


def test_a_legacy_database_gains_the_columns_without_backfill(tmp_path: Path):
    """Pre-T2.6 rows keep NULL: we cannot know which sample they described.

    Minting one for them retroactively would assert that a formulation row and
    the spectra taken off the same well are different samples — the precise
    false statement the column exists to refute. Same no-backfill rule as [8p]'s
    ``deposit_area_mm2``.
    """
    project = tmp_path / "proj"
    store = DataStore(project)
    run_id = store.start_run("legacy")
    store.close()

    # Simulate a database written before the migration existed.
    conn = sqlite3.connect(project / "db" / "softae.db")
    conn.execute("ALTER TABLE formulations DROP COLUMN sample_uuid")
    conn.execute("ALTER TABLE electrode_occupancy DROP COLUMN sample_uuid")
    conn.execute(
        "INSERT INTO formulations (run_id, channel, total_uL) VALUES (?, 3, 20.0)",
        (run_id,),
    )
    conn.execute(
        "INSERT INTO electrode_occupancy (board_id, electrode) VALUES (0, 3)"
    )
    conn.commit()
    conn.close()

    store = DataStore(project)
    assert "sample_uuid" in _column_names(store, "formulations")
    assert _formulation_uuids(store, run_id) == {3: None}
    assert _occupancy_uuids(store) == {(0, 3): None}
    store.close()


def test_record_writers_normalise_a_blank_identity_to_null(tmp_path: Path):
    """``''`` is not an identity — it would join to every other blank row."""
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("blank_identity")
    store.record_formulation(run_id, 5, total_uL=10.0, sample_uuid="")
    store.record_electrode_cast(0, 5, sample_uuid="")
    assert _formulation_uuids(store, run_id) == {5: None}
    assert _occupancy_uuids(store) == {(0, 5): None}
    store.close()


def test_a_recast_well_replaces_the_identity_rather_than_keeping_the_old_one(
    tmp_path: Path,
):
    """The new sample occupies the well; the old one's rows keep the old uuid."""
    store = DataStore(tmp_path / "proj")
    first, second = mint_sample_uuid(), mint_sample_uuid()
    store.record_electrode_cast(0, 7, sample_uuid=first)
    store.record_electrode_cast(0, 7, sample_uuid=second)
    assert _occupancy_uuids(store) == {(0, 7): second}
    store.close()


# ── Step tagging ─────────────────────────────────────────────────────────────


def test_every_channel_bearing_step_carries_its_own_sample_uuid(catalog):
    spec = _spec()
    channels = [21, 22]
    minted = mint_sample_uuids(channels)
    wf = build_placement_workflow(
        spec, [{"vol_p0": 20.0, "vol_p1": 10.0}] * 2, channels,
        catalog=catalog, sample_uuid_by_channel=minted,
    )

    seen: dict[str, set[str]] = {}
    for step in wf.resolve_steps():
        channel = step.tags.get("channel")
        if channel is None:
            # Board-wide steps belong to no single sample and must stay untagged.
            assert "sample_uuid" not in step.tags, step.name
            continue
        seen.setdefault(channel, set()).add(step.tags["sample_uuid"])

    assert set(seen) == {"21", "22"}
    assert seen["21"] == {minted[21]}
    assert seen["22"] == {minted[22]}
    assert seen["21"] != seen["22"]


def test_an_unminted_build_leaves_every_step_untagged(catalog):
    """The default path is unchanged: no map, no tag, no empty string."""
    wf = build_placement_workflow(
        _spec(), [{"vol_p0": 20.0, "vol_p1": 10.0}], [21], catalog=catalog,
    )
    assert all("sample_uuid" not in s.tags for s in wf.resolve_steps())


# ── Router stamping ──────────────────────────────────────────────────────────


def _eis_step(name: str, chan: int, **tags: str) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        instrument="pico1",
        method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": chan},
        tags=dict(tags),
    )


@pytest.mark.asyncio
async def test_a_tagged_step_stamps_the_row_the_meta_and_the_payload(
    connected, tmp_path: Path
):
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("router_identity")
    sample_uuid = mint_sample_uuid()

    executor = WorkflowExecutor(connected, data_store=store, run_id=run_id)
    await executor.run(Workflow(
        name="stamped",
        setup=[_eis_step("measure", 1, channel="1", sample_uuid=sample_uuid)],
    ))
    assert executor.state == ExecutorState.COMPLETED

    row = store.query_measurements(run_id=run_id)[0]
    assert row["sample_uuid"] == sample_uuid

    measurement = executor.measurement_results[0]
    assert measurement.meta["sample_uuid"] == sample_uuid
    # The payload must self-describe: a file lifted out of the project and read
    # somewhere else still says which sample it came off.
    assert measurement.data.attrs["sample_uuid"] == sample_uuid
    assert isinstance(measurement.data.attrs["sample_uuid"], str)

    on_disk = Path(store.project_dir) / row["payload_path"]
    with xr.open_dataset(on_disk, engine="h5netcdf") as ds:
        assert ds.attrs["sample_uuid"] == sample_uuid
    store.close()


@pytest.mark.asyncio
async def test_an_untagged_step_omits_the_key_rather_than_writing_a_blank(
    connected, tmp_path: Path
):
    """The absence convention, at the seam where it is easiest to break.

    This is also the reasoning behind ``test_result_router_golden.py`` staying
    green untouched: it drives the executor directly with hand-built steps that
    carry no ``sample_uuid`` tag, so its pinned ``sample_uuid: None`` is still
    what the path produces.
    """
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("router_anonymous")

    executor = WorkflowExecutor(connected, data_store=store, run_id=run_id)
    await executor.run(Workflow(name="bare", setup=[_eis_step("measure", 1)]))
    assert executor.state == ExecutorState.COMPLETED

    assert store.query_measurements(run_id=run_id)[0]["sample_uuid"] is None
    measurement = executor.measurement_results[0]
    assert "sample_uuid" not in measurement.meta
    assert "sample_uuid" not in measurement.data.attrs
    store.close()


@pytest.mark.asyncio
async def test_one_sample_carries_arbitrarily_many_measurements(
    connected, tmp_path: Path
):
    """``sample_uuid`` is a *grouping* key, never a unique one.

    A sample is measured repeatedly by design — before and after an anneal, at
    each step of a temperature sweep, on a drift re-check — and each of those is
    an independent event with its own row, its own timestamp, its own acquisition
    position and its own payload file. The column carries no ``UNIQUE``
    constraint and is not part of any primary key precisely so that the
    relationship stays one sample → many measurements.

    This is the invariant a per-measurement identity would have destroyed: minting
    at measurement time (rather than at well consumption) would have given three
    spectra off one film three different identities.
    """
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("repeat_measurement")
    sample_uuid = mint_sample_uuid()

    executor = WorkflowExecutor(connected, data_store=store, run_id=run_id)
    await executor.run(Workflow(
        name="sweep",
        setup=[
            _eis_step(f"eis_ch1_T{t}_RH40", 1, channel="1",
                      sample_uuid=sample_uuid)
            for t in (25, 40, 55)
        ],
    ))
    assert executor.state == ExecutorState.COMPLETED

    rows = store.query_measurements(run_id=run_id)
    assert len(rows) == 3
    # One identity, three events.
    assert {r["sample_uuid"] for r in rows} == {sample_uuid}
    assert len({r["measurement_id"] for r in rows}) == 3
    # Each is individually timestamped and individually positioned in the sweep.
    assert all(r["timestamp"] for r in rows)
    assert sorted(r["sweep_order"] for r in rows) == [1, 2, 3]
    # …and separately stored: three payloads, three conditions snapshots.
    assert len({r["payload_path"] for r in rows}) == 3
    conditions = store.query_conditions(run_id=run_id)
    assert {c["measurement_id"] for c in conditions} == {
        r["measurement_id"] for r in rows
    }
    store.close()


@pytest.mark.asyncio
async def test_a_blank_tag_is_treated_as_no_tag(connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("router_blank")

    executor = WorkflowExecutor(connected, data_store=store, run_id=run_id)
    await executor.run(Workflow(
        name="blank", setup=[_eis_step("measure", 1, sample_uuid="")],
    ))

    assert store.query_measurements(run_id=run_id)[0]["sample_uuid"] is None
    assert "sample_uuid" not in executor.measurement_results[0].data.attrs
    store.close()


# ── End-to-end: the spine holds across all four writers ──────────────────────


@pytest.mark.asyncio
async def test_every_consumed_well_carries_one_identity_end_to_end(
    connected, tmp_path: Path
):
    """A board-aware campaign: formulation, occupancy, measurement and payload
    all name the same sample, per well, with no two wells sharing one."""
    from softae.core.autonomous_loop import BoardDecision

    store = DataStore(tmp_path / "proj")
    spec = _spec(budget=3, electrode_capacity=8, equilibration_s=0.0)
    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda b: BoardDecision.PROCEED,
    )
    assert result.n_trials == 3

    cast_channels = sorted(store.occupied_electrodes(0))
    assert cast_channels == [1, 2, 3]

    occupancy = _occupancy_uuids(store)
    formulations = _formulation_uuids(store, result.run_id)
    measurements = {
        int(r["channel"]): r["sample_uuid"]
        for r in store.query_measurements(run_id=result.run_id)
    }
    payload_by_channel = {
        int(r["channel"]): r["payload_path"]
        for r in store.query_measurements(run_id=result.run_id)
    }

    identities = set()
    for channel in cast_channels:
        sample_uuid = occupancy[(0, channel)]
        assert sample_uuid, f"channel {channel} has no occupancy identity"
        assert sample_uuid.startswith(SAMPLE_UUID_PREFIX)
        assert formulations[channel] == sample_uuid
        assert measurements[channel] == sample_uuid
        with xr.open_dataset(
            Path(store.project_dir) / payload_by_channel[channel],
            engine="h5netcdf",
        ) as ds:
            assert ds.attrs["sample_uuid"] == sample_uuid
        identities.add(sample_uuid)

    # One well, one sample: three wells, three identities.
    assert len(identities) == len(cast_channels)
    store.close()


@pytest.mark.asyncio
async def test_a_batch_round_mints_q_distinct_identities(connected, tmp_path: Path):
    """q suggestions cast onto q wells in one round are q separate samples.

    The failure this guards against is a *round*-scoped uuid: the four wells of a
    q=4 round are cast in one workflow, and one identity per workflow would read
    as correct everywhere except the one place it matters — telling the four
    apart afterwards.
    """
    store = DataStore(tmp_path / "proj")
    spec = _spec(batch=True, budget=4)  # 4 channels → one round of 4
    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
    )
    assert result.n_trials == 4

    measurements = {
        int(r["channel"]): r["sample_uuid"]
        for r in store.query_measurements(run_id=result.run_id)
    }
    assert set(measurements) == {21, 22, 23, 24}
    assert all(v for v in measurements.values())
    assert len(set(measurements.values())) == 4

    # And the formulation rows for the round agree, channel by channel.
    formulations = _formulation_uuids(store, result.run_id)
    for channel, sample_uuid in measurements.items():
        assert formulations[channel] == sample_uuid
    store.close()
