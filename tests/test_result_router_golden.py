"""Golden-run characterization of the EIS result-routing path.

Guards the Tier-1 router extraction (docs/SubAgent docs/
afl_comparison_and_restructuring_spec.md §4, Tier 1 "Result-router registry" and
the Tier-2 sequencing note: "A golden-run characterization test comes first: the
EIS path must produce byte-identical rows and files across the router
extraction").

This test was written against the PRE-refactor executor (the `_EIS_METHODS`
branch + `_route_eis_to_datastore`) and passed there; it must keep passing
UNCHANGED after the routing moves to ``softae.analysis.eis.router``. It
deliberately drives the public entry point (``WorkflowExecutor.run``) rather
than any private hook, so it survives the private surface being renamed.

What is pinned, and why it can be pinned exactly:

* MockESPico seeds its RNG from the channel number, so the spectrum — and
  therefore npts / f_min / f_max / the saved file's numeric table — is
  deterministic per channel.
* Every step-tag key the routing reads (role, fixture_id, electrode_mode,
  nominal, thermal_history, re_connection, re_contact_verified) is exercised,
  on one commissioning-tagged step, alongside one untagged sample step that
  pins the E6 defaulting semantics.
* Timestamps and mock temp/RH process values are genuinely nondeterministic,
  so those columns get shape matchers (ISO-parseable / finite float) — still
  compared for every column, just not to a literal.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep


# ── Matchers for the columns that cannot be literal ─────────────────────────


class _Finite:
    """Equal to any finite real number (mock SP/PVs are seeded but noisy)."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and math.isfinite(float(other))

    def __repr__(self) -> str:  # pragma: no cover - repr for assert diffs
        return "<finite float>"


class _IsoTimestamp:
    """Equal to any ISO-8601 timestamp string (wall-clock, so not literal)."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, str):
            return False
        try:
            datetime.fromisoformat(other)
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:  # pragma: no cover - repr for assert diffs
        return "<iso timestamp>"


FINITE = _Finite()
ISO_TS = _IsoTimestamp()


# ── Fixtures (same shapes as test_eis_autoroute.py) ─────────────────────────


@pytest.fixture
def manager() -> InstrumentManager:
    return create_mock_manager(config={})


@pytest.fixture
async def connected_manager(manager: InstrumentManager):
    await manager.connect_all()
    return manager


@pytest.fixture
def data_store(tmp_path: Path):
    ds = DataStore(tmp_path / "project")
    yield ds
    ds.close()


# ── The golden run ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_golden_eis_routing_rows_and_files(connected_manager, data_store):
    run_id = data_store.start_run("golden_eis_route")

    # Step 1: plain sample with routing-only params. The electrode_* / circuit
    # params MUST be stripped before the instrument call — MockESPico's
    # sendscript_getdata(mscrpath, outdir, chan) would raise on unexpected
    # kwargs, so the workflow completing at all is the filter's proof.
    sample_step = WorkflowStep(
        name="measure",
        instrument="pico1",
        method="sendscript_getdata",
        params={
            "mscrpath": "f.mscr", "outdir": "out", "chan": 1,
            "electrode_x_mm": 12.5, "electrode_y_mm": -3.0,
            "circuit_model": "simpleSalt",
            "electrode_L_cm": 0.5, "electrode_t_cm": 0.001, "electrode_w_cm": 0.1,
        },
    )
    # Step 2: commissioning blank carrying every tag key the routing reads,
    # with a new-style name that exercises the eis_ch<N>_T<t>_RH<rh> file-stem
    # branch (no _ch suffix appended).
    blank_step = WorkflowStep(
        name="eis_ch2_T25_RH40",
        instrument="pico1",
        method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": 2},
        tags={
            "role": "blank_open",
            "fixture_id": "mux16",
            "electrode_mode": "2T",
            "nominal": "100.0",
            "thermal_history": "as_received",
            "re_connection": "tied_to_ce",
            "re_contact_verified": True,
        },
    )

    wf = Workflow(name="golden", setup=[sample_step, blank_step])
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    await executor.run(wf)
    assert executor.state == ExecutorState.COMPLETED

    # ── Measurements: every column, exact where deterministic ────────────
    rows = data_store.query_measurements(run_id=run_id)
    expected_measurements = [
        {
            "measurement_id": 1,
            "run_id": run_id,
            "channel": 1,
            "electrode_x_mm": 12.5,
            "electrode_y_mm": -3.0,
            "timestamp": ISO_TS,
            "npts": 41,
            "f_min_hz": 1.0,
            "f_max_hz": 50000.0,
            "measurement_time_s": 0.0,
            "eis_file_path": str(Path("runs") / run_id / "eis" / "measure_ch1.txt"),
            "eis_params_json": "{}",
            # E6 defaults: role='sample' implies the cast film closes the RE
            # loop; contact is never defaulted true (R26).
            "role": "sample",
            "fixture_id": None,
            "nominal_value": None,
            "electrode_mode": "unknown",
            "thermal_history": "",
            "sweep_order": 1,
            "re_connection": "bridged_by_sample",
            "re_contact_verified": 0,
            # Tier 2 component 3. `query_measurements` selects `m.*`, so these
            # appear in every row dict — pinned here rather than excluded, so
            # that a future column cannot slip in unasserted.
            "modality": "eis",
            "payload_path": str(
                Path("runs") / run_id / "data" / "eis" / "measure_ch1.nc"
            ),
            "payload_format": "netcdf4",
            # Minting is T2.6; until then the honest value is NULL.
            "sample_uuid": None,
            "workflow_name": "golden_eis_route",
            "pcb_name": None,
        },
        {
            "measurement_id": 2,
            "run_id": run_id,
            "channel": 2,
            "electrode_x_mm": None,
            "electrode_y_mm": None,
            "timestamp": ISO_TS,
            "npts": 41,
            "f_min_hz": 1.0,
            "f_max_hz": 50000.0,
            "measurement_time_s": 0.0,
            "eis_file_path": str(
                Path("runs") / run_id / "eis" / "eis_ch2_T25_RH40.txt"
            ),
            "eis_params_json": "{}",
            "role": "blank_open",
            "fixture_id": "mux16",
            "nominal_value": 100.0,
            "electrode_mode": "2T",
            "thermal_history": "as_received",
            "sweep_order": 2,
            "re_connection": "tied_to_ce",
            "re_contact_verified": 1,
            "modality": "eis",
            "payload_path": str(
                Path("runs") / run_id / "data" / "eis" / "eis_ch2_T25_RH40.nc"
            ),
            "payload_format": "netcdf4",
            "sample_uuid": None,
            "workflow_name": "golden_eis_route",
            "pcb_name": None,
        },
    ]
    assert rows == expected_measurements

    # ── Conditions: one 'measurement' snapshot per row, every column ─────
    conds = data_store.query_conditions(run_id=run_id)
    conds.sort(key=lambda c: c["condition_id"])
    expected_conditions = [
        {
            "condition_id": i + 1,
            "measurement_id": i + 1,
            "run_id": run_id,
            "stage": "measurement",
            "timestamp": ISO_TS,
            "stage_temp_sp_C": FINITE,
            "chamber_air_C": FINITE,
            "stage_temp_pv_C": FINITE,
            "rh_sp_pct": FINITE,
            "rh_pv_pct": FINITE,
            # Schema epoch 4: `record_conditions` resolves the sample's
            # temperature at write time. The fixture supplies a stage PV, so the
            # best source wins — pinned as a literal because a *source* that
            # drifted to the air probe is the failure this whole module exists
            # to catch, and a sentinel would hide it.
            "temperature_C": FINITE,
            "temperature_source": "stage_pv",
            "notes": "",
        }
        for i in range(2)
    ]
    assert conds == expected_conditions

    # The derived column is not merely present and finite — it is *the stage PV*.
    for cond in conds:
        assert cond["temperature_C"] == cond["stage_temp_pv_C"]
        assert cond["temperature_C"] != cond["chamber_air_C"]

    # ── Auto-fit: routed only for the step that declared circuit_model ───
    fits = data_store.query_fits(run_id=run_id)
    assert len(fits) == 1
    assert fits[0]["measurement_id"] == 1
    assert fits[0]["model_name"] == "simpleSalt"

    # ── Files on disk: numeric tables byte-equivalent to the mock spectra ─
    # MockESPico re-seeds from the channel on every call, so re-invoking it
    # reproduces the exact spectrum the run recorded — no fixture data to
    # drift out of date. |Z|/phase are recomputed from Z'/-Z'' exactly the way
    # the routing's EISResult.from_raw does.
    pico = connected_manager.get("pico1")
    eis_dir = data_store.eis_dir(run_id)
    payload_dir = data_store.payload_dir(run_id, "eis")
    xr = pytest.importorskip("xarray")

    for chan, stem in ((1, "measure_ch1"), (2, "eis_ch2_T25_RH40")):
        raw = np.asarray(pico.sendscript_getdata("f.mscr", "out", chan)[0])
        z = raw[:, 3] + 1j * (-raw[:, 4])
        expected_table = np.column_stack(
            [raw[:, 0], np.abs(z), np.angle(z, deg=True), z.real, -z.imag]
        )
        loaded = np.loadtxt(eis_dir / f"{stem}.txt")
        np.testing.assert_allclose(loaded, expected_table, rtol=1e-12, atol=0.0)

        # ── Tier 2 component 3: the netCDF payload beside the .txt ───────
        # Compared against the *same* expected table, not against the .txt —
        # so this asserts the payload is independently correct rather than
        # merely self-consistent with the file it is meant to replace.
        payload = payload_dir / f"{stem}.nc"
        assert payload.exists(), f"no payload written for {stem}"
        with xr.open_dataset(payload) as ds:
            np.testing.assert_allclose(
                np.column_stack([
                    ds["frequency_hz"].values, ds["z_mag"].values,
                    ds["phase"].values, ds["z_real"].values,
                    ds["z_imag_neg"].values,
                ]),
                expected_table, rtol=1e-12, atol=0.0,
            )
            # The payload self-describes: `attrs` alone name the row it belongs
            # to, which is what makes a file on disk usable without the database.
            assert ds.attrs["run_id"] == run_id
            assert ds.attrs["step_name"] == (
                "measure" if chan == 1 else "eis_ch2_T25_RH40")

    # The database's paths resolve against the project dir, so a stored row can
    # actually find its file — a relative path that does not is worse than none.
    for row in rows:
        assert (data_store.project_dir / row["payload_path"]).exists()


@pytest.mark.asyncio
async def test_a_failed_payload_write_leaves_nulls_and_costs_nothing_else(
    connected_manager, data_store, monkeypatch
):
    """The payload is an optional second copy; the measurement is not.

    A full disk, a locked file or a missing HDF5 backend must not retract a
    measurement that physically happened. The row is committed *before* the
    payload is written, so a failure here can only leave the payload columns
    NULL — which is exactly what NULL there means: no payload was written.

    Everything the golden test pins must survive unchanged, which is why this
    re-asserts the `.txt` and the original columns rather than only the NULLs:
    a payload failure that quietly cost the auto-fit would pass a narrower test.
    """
    import xarray as xr

    def _explode(self, *args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _explode)

    run_id = data_store.start_run("payload_failure")
    step = WorkflowStep(
        name="measure",
        instrument="pico1",
        method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": 1,
                "circuit_model": "simpleSalt",
                "electrode_L_cm": 0.5, "electrode_t_cm": 0.001,
                "electrode_w_cm": 0.1},
    )

    executor = WorkflowExecutor(connected_manager, data_store=data_store,
                               run_id=run_id)
    await executor.run(Workflow(name="payload_failure", setup=[step]))

    # The run did not fail, and the routing did not report a failed measurement.
    assert executor.state == ExecutorState.COMPLETED
    assert len(executor.measurement_results) == 1
    assert executor.measurement_results[0].modality == "eis"

    rows = data_store.query_measurements(run_id=run_id)
    assert len(rows) == 1
    assert rows[0]["payload_path"] is None
    assert rows[0]["payload_format"] is None
    # `modality` is set on the INSERT, so it survives a payload failure entirely.
    assert rows[0]["modality"] == "eis"

    # The transitional .txt and the fit are untouched by the payload's failure.
    assert (data_store.eis_dir(run_id) / "measure_ch1.txt").exists()
    assert len(data_store.query_fits(run_id=run_id)) == 1
    # No half-written payload left behind for a later reader to trust.
    assert list(data_store.payload_dir(run_id, "eis").glob("*.nc")) == []
