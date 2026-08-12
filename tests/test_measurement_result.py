"""Tier 2 components 1-2: the MeasurementResult contract and the router seam.

Covers (spec §4, ``afl_comparison_and_restructuring_spec.md``):

* the **lossless** ``EISResult`` <-> ``MeasurementResult`` bridge, including the
  gate branch with the mask present *and* absent, and the mask-without-reasons
  case the two flags' independence exists for;
* the contract module staying **modality-agnostic** — asserted by import
  inspection, because the spec names ``EISResult``'s shape leaking into
  ``MeasurementResult`` as this tier's feared failure mode;
* the router stamping step-side electrode geometry (SESSION_MAIL #6) without
  fabricating anything;
* the executor accumulating results across a routed run.

The golden-run test (``test_result_router_golden.py``) is the guard for the
*unchanged* half and is deliberately not touched by any of this.
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from softae.analysis.eis.router import EISResultRouter, RouterContext
from softae.analysis.eis_data import EIS_MODALITY, FREQ_DIM, EISResult
from softae.analysis.measurement_result import MeasurementResult
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

# ── Fixtures (same shapes as test_result_router_golden.py) ──────────────────


@pytest.fixture
def manager() -> InstrumentManager:
    # create_mock_manager(), never a bare stub: hardware_safety fails CLOSED on
    # an unreadable manager (TASKS.md P.4), so a stub would raise, not skip.
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


# ── Builders ────────────────────────────────────────────────────────────────


def _spectrum(n: int = 6) -> dict[str, np.ndarray]:
    f = np.logspace(0, 4, n)
    return {
        "frequency": f,
        "z_real": np.linspace(120.0, 5.0, n),
        "z_imag_neg": np.linspace(3.0, 88.0, n),
        "z_magnitude": np.linspace(121.0, 89.0, n),
        "phase": np.linspace(-1.5, -84.0, n),
    }


def _minimal_result() -> EISResult:
    """Only the required fields — every optional one left at its default."""
    return EISResult(channel=3, **_spectrum())


def _full_result(n: int = 6) -> EISResult:
    """Every one of the 17 fields populated, gates included."""
    return EISResult(
        channel=7,
        **_spectrum(n),
        residual_real_pct=np.linspace(-0.4, 0.9, n),
        residual_imag_pct=np.linspace(0.2, -1.1, n),
        timestamp=datetime(2026, 8, 7, 15, 4, 5, 123456),
        measurement_time_s=42.75,
        eis_params={"npts": n, "f_hi": 50000.0, "f_lo_mHz": 1000.0,
                    "mv_ac": 10.0, "mv_dc": 0, "preset": "quick"},
        raw_file_path="runs/r1/eis/eis_ch7.txt",
        T_sp=25.0, T_pv=24.87, rh_sp=40.0, rh_pv=41.2,
        mask=np.array([True, True, False, True, False, True][:n]),
        drop_gate=np.array(["", "", "hf_inductive", "", "lf_drift", ""][:n],
                           dtype=object),
    )


def assert_eis_identical(a: EISResult, b: EISResult) -> None:
    """Every field of *a* survives into *b*, exactly."""
    for name in ("frequency", "z_real", "z_imag_neg", "z_magnitude", "phase"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name),
                                      err_msg=f"array {name} diverged")

    assert a.channel == b.channel
    assert a.timestamp == b.timestamp
    assert a.measurement_time_s == b.measurement_time_s
    assert a.eis_params == b.eis_params
    assert a.raw_file_path == b.raw_file_path

    # The four condition scalars default to NaN, and NaN != NaN.
    for name in ("T_sp", "T_pv", "rh_sp", "rh_pv"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name),
                                   equal_nan=True, err_msg=f"scalar {name}")

    for name in ("residual_real_pct", "residual_imag_pct", "mask", "drop_gate"):
        original, restored = getattr(a, name), getattr(b, name)
        if original is None:
            assert restored is None, f"{name}: None became {restored!r}"
        else:
            assert restored is not None, f"{name}: lost in the round trip"
            np.testing.assert_array_equal(original, restored, err_msg=name)


# ── T2.1: round-trip fidelity ───────────────────────────────────────────────


def test_bridge_roundtrip_minimal_result_identical():
    original = _minimal_result()
    assert_eis_identical(original, EISResult.from_measurement(original.to_measurement()))


def test_bridge_roundtrip_full_result_identical():
    original = _full_result()
    assert_eis_identical(original, EISResult.from_measurement(original.to_measurement()))


def test_bridge_roundtrip_without_gates_omits_gate_variables():
    """The gate branch, absent: no mask/drop_gate vars, and None stays None."""
    original = _minimal_result()
    assert original.mask is None and original.drop_gate is None

    measurement = original.to_measurement()
    assert "mask" not in measurement.data.data_vars
    assert "drop_gate" not in measurement.data.data_vars

    restored = EISResult.from_measurement(measurement)
    assert restored.mask is None
    assert restored.drop_gate is None


def test_bridge_roundtrip_with_gates_restores_object_dtype_strings():
    """The gate branch, present: names come back as plain `str` in an object array."""
    original = _full_result()
    restored = EISResult.from_measurement(original.to_measurement())

    assert restored.drop_gate.dtype == object
    assert all(type(v) is str for v in restored.drop_gate)
    assert list(restored.drop_gate) == ["", "", "hf_inductive", "", "lf_drift", ""]

    assert restored.mask.dtype == bool
    np.testing.assert_array_equal(restored.mask, original.mask)


def test_bridge_roundtrip_mask_without_drop_gate_preserved():
    """A survivor mask carrying no per-point reasons is a real EISResult state.

    `_encode_drop_gates` synthesises generic names for exactly this case, so the
    two flags must attach independently — pairing them would drop the mask.
    """
    original = _minimal_result()
    original.mask = np.array([True, False, True, True, False, True])

    measurement = original.to_measurement()
    assert "mask" in measurement.data.data_vars
    assert "drop_gate" not in measurement.data.data_vars

    restored = EISResult.from_measurement(measurement)
    np.testing.assert_array_equal(restored.mask, original.mask)
    assert restored.drop_gate is None


def test_bridge_roundtrip_preserves_nan_condition_scalars():
    """The NaN defaults must survive as NaN, not become 0.0 or None."""
    restored = EISResult.from_measurement(_minimal_result().to_measurement())
    for name in ("T_sp", "T_pv", "rh_sp", "rh_pv"):
        assert math.isnan(getattr(restored, name)), f"{name} lost its NaN"


def test_bridge_roundtrip_of_empty_spectrum_is_identical():
    empty = EISResult(
        channel=1, frequency=np.array([]), z_magnitude=np.array([]),
        phase=np.array([]), z_real=np.array([]), z_imag_neg=np.array([]),
    )
    restored = EISResult.from_measurement(empty.to_measurement())
    assert restored.npts == 0
    assert_eis_identical(empty, restored)


# ── T2.1: Dataset shape, attrs, meta, summary ───────────────────────────────


def test_to_measurement_has_frequency_coord_and_spec_named_variables():
    measurement = _minimal_result().to_measurement()

    assert measurement.modality == EIS_MODALITY == "eis"
    assert isinstance(measurement.data, xr.Dataset)
    assert FREQ_DIM in measurement.data.coords
    assert set(measurement.data.data_vars) == {"z_real", "z_imag_neg", "z_mag", "phase"}
    for name in measurement.data.data_vars:
        assert measurement.data[name].dims == (FREQ_DIM,)


def test_to_measurement_attrs_are_netcdf_encodable():
    """attrs must reconstruct the result alone, so they must be encodable.

    netCDF attributes hold only strings, numbers and arrays of those — no
    nested dicts and no None. This pins the constraint directly rather than
    via a backend, so it holds even where no writer is installed.
    """
    attrs = _full_result().to_measurement().data.attrs
    for key, value in attrs.items():
        assert isinstance(value, (str, int, float, np.ndarray)), (
            f"attr {key!r} is {type(value).__name__}, not netCDF-encodable"
        )
    # The nested dict specifically is JSON-encoded rather than stored raw.
    assert json.loads(attrs["eis_params_json"])["preset"] == "quick"


def test_to_measurement_omits_raw_file_path_when_unset():
    """Absent stays absent — netCDF has no null, so a None key must not appear."""
    assert "raw_file_path" not in _minimal_result().to_measurement().data.attrs
    assert "raw_file_path" in _full_result().to_measurement().data.attrs


def test_to_measurement_meta_mirrors_values_unencoded():
    original = _full_result()
    meta = original.to_measurement().meta

    assert meta["channel"] == original.channel
    assert meta["timestamp"] == original.timestamp        # datetime, not a string
    assert meta["eis_params"] == original.eis_params      # dict, not JSON
    assert meta["raw_file_path"] == original.raw_file_path
    assert meta["npts"] == original.npts


def test_to_measurement_summary_is_a_derived_digest():
    original = _full_result()
    summary = original.to_measurement().summary

    assert summary == {
        "npts": float(original.npts),
        "f_min_hz": float(original.frequency.min()),
        "f_max_hz": float(original.frequency.max()),
    }
    # Derived only: an empty spectrum has no min/max to report.
    assert _minimal_result().to_measurement().summary is not None
    empty = EISResult(channel=1, frequency=np.array([]), z_magnitude=np.array([]),
                      phase=np.array([]), z_real=np.array([]), z_imag_neg=np.array([]))
    assert empty.to_measurement().summary is None


def test_from_measurement_rejects_foreign_modality():
    payload = MeasurementResult(modality="image", data=xr.Dataset())
    with pytest.raises(ValueError, match="image"):
        EISResult.from_measurement(payload)


def test_from_measurement_rejects_payload_missing_variables():
    measurement = _minimal_result().to_measurement()
    del measurement.data["z_mag"]
    with pytest.raises(ValueError, match="z_mag"):
        EISResult.from_measurement(measurement)


# ── T2.1: the contract itself ───────────────────────────────────────────────


def test_measurement_result_is_frozen():
    measurement = _minimal_result().to_measurement()
    with pytest.raises(FrozenInstanceError):
        measurement.modality = "image"             # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        measurement.data = xr.Dataset()            # type: ignore[misc]


def test_measurement_result_defaults_meta_to_empty_dict():
    payload = MeasurementResult(modality="image", data=xr.Dataset())
    assert payload.meta == {}
    assert payload.summary is None
    # Default must not be shared between instances.
    payload.meta["a"] = 1
    assert MeasurementResult(modality="image", data=xr.Dataset()).meta == {}


def test_measurement_result_equality_is_by_identity_and_hashable():
    """`==` must answer, not raise — Dataset's elementwise `==` would."""
    data = xr.Dataset()
    a = MeasurementResult(modality="eis", data=data)
    b = MeasurementResult(modality="eis", data=data)
    assert a == a
    assert a != b            # identity, not structural — would have raised on ==
    assert len({a, b}) == 2  # hashable, so usable as keys


def test_measurement_result_module_imports_no_modality():
    """The contract module must not import any modality (spec's feared leak).

    Inspected via the AST rather than by catching ImportError, so an import
    that merely *happens* to be installed still fails the check.
    """
    source = Path(
        __import__("softae.analysis.measurement_result", fromlist=["_"]).__file__
    ).read_text(encoding="utf-8")

    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(f"{node.module or ''}.{a.name}" for a in node.names)

    offenders = [name for name in imported
                 if "eis" in name.lower() or "softae." in name]
    assert not offenders, f"modality-agnostic module imports: {offenders}"


@pytest.mark.parametrize("engine", ["h5netcdf"])
def test_eis_payload_survives_a_netcdf_roundtrip(tmp_path: Path, engine: str):
    """The self-describing claim, end to end — dormant until a writer is installed.

    ``h5netcdf`` is declared, but its ``h5py`` backend is not installed as of
    2026-08-07, so this skips rather than failing. It goes live the moment T2.3
    adds the writer, which is exactly when the guarantee starts to matter.
    """
    pytest.importorskip("h5py", reason="h5netcdf backend not installed (T2.3)")

    original = _full_result()
    path = tmp_path / "payload.nc"
    original.to_measurement().data.to_netcdf(path, engine=engine)

    reloaded = MeasurementResult(modality=EIS_MODALITY,
                                 data=xr.load_dataset(path, engine=engine))
    assert_eis_identical(original, EISResult.from_measurement(reloaded))


# ── T2.2: the router seam ───────────────────────────────────────────────────


def _eis_step(name: str = "measure", chan: int = 1, **params) -> WorkflowStep:
    return WorkflowStep(
        name=name, instrument="pico1", method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": chan, **params},
    )


@pytest.mark.asyncio
async def test_router_handle_returns_measurement_result(connected_manager, data_store):
    run_id = data_store.start_run("router_returns")
    step = _eis_step()
    raw = connected_manager.get("pico1").sendscript_getdata("f.mscr", "out", 1)

    returned = await EISResultRouter().handle(
        step, raw,
        RouterContext(data_store=data_store, run_id=run_id,
                      manager=connected_manager),
    )

    assert isinstance(returned, MeasurementResult)
    assert returned.modality == "eis"
    assert returned.meta["step_name"] == "measure"
    assert returned.meta["run_id"] == run_id
    assert returned.meta["measurement_id"] == 1


@pytest.mark.asyncio
async def test_router_handle_returns_none_without_datastore(connected_manager):
    """No store is a skip, and a skip yields no payload — never a half-recorded one."""
    step = _eis_step()
    raw = connected_manager.get("pico1").sendscript_getdata("f.mscr", "out", 1)

    assert await EISResultRouter().handle(
        step, raw, RouterContext(data_store=None, run_id=None)
    ) is None


@pytest.mark.asyncio
async def test_router_stamps_declared_electrode_geometry(connected_manager, data_store):
    """SESSION_MAIL #6: geometry the step declared travels with the payload."""
    run_id = data_store.start_run("router_geometry")
    step = _eis_step(electrode_L_cm=0.5, electrode_w_cm=0.1, electrode_x_mm=12.5)
    raw = connected_manager.get("pico1").sendscript_getdata("f.mscr", "out", 1)

    returned = await EISResultRouter().handle(
        step, raw,
        RouterContext(data_store=data_store, run_id=run_id,
                      manager=connected_manager),
    )

    # Present-only: the two undeclared keys are absent, not zero and not None.
    assert returned.meta["electrode_geometry"] == {
        "electrode_L_cm": 0.5, "electrode_w_cm": 0.1, "electrode_x_mm": 12.5,
    }
    assert "electrode_t_cm" not in returned.meta["electrode_geometry"]
    # And it reaches the payload's own attrs, so the file self-describes.
    assert returned.data.attrs["electrode_L_cm"] == 0.5
    assert "electrode_t_cm" not in returned.data.attrs
    # Never fabricated: area resolution is T2.3's, via core/geometry.py.
    assert "deposit_area_mm2" not in returned.data.attrs


@pytest.mark.asyncio
async def test_router_omits_geometry_key_when_step_declares_none(
    connected_manager, data_store
):
    run_id = data_store.start_run("router_no_geometry")
    raw = connected_manager.get("pico1").sendscript_getdata("f.mscr", "out", 1)

    returned = await EISResultRouter().handle(
        _eis_step(), raw,
        RouterContext(data_store=data_store, run_id=run_id,
                      manager=connected_manager),
    )
    assert "electrode_geometry" not in returned.meta


@pytest.mark.asyncio
async def test_executor_accumulates_measurement_results_during_routed_run(
    connected_manager, data_store
):
    run_id = data_store.start_run("executor_accumulates")
    wf = Workflow(name="accumulate", setup=[
        _eis_step("measure", chan=1, electrode_L_cm=0.5),
        _eis_step("eis_ch2_T25_RH40", chan=2),
    ])
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)

    assert executor.measurement_results == []
    await executor.run(wf)
    assert executor.state == ExecutorState.COMPLETED

    results = executor.measurement_results
    assert len(results) == 2
    assert all(isinstance(m, MeasurementResult) for m in results)
    # Acquisition order, and each payload rebuilds into its own channel.
    assert [m.meta["channel"] for m in results] == [1, 2]
    assert [m.meta["step_name"] for m in results] == ["measure", "eis_ch2_T25_RH40"]
    assert EISResult.from_measurement(results[0]).channel == 1
    # Only the step that declared geometry carries it.
    assert "electrode_geometry" in results[0].meta
    assert "electrode_geometry" not in results[1].meta


@pytest.mark.asyncio
async def test_executor_measurement_results_reset_between_runs(
    connected_manager, data_store
):
    """Per-run, not per-executor: a second run must not append to the first."""
    run_id = data_store.start_run("executor_resets")
    executor = WorkflowExecutor(connected_manager, data_store=data_store, run_id=run_id)
    wf = Workflow(name="once", setup=[_eis_step("measure", chan=1)])

    await executor.run(wf)
    await executor.run(wf)
    assert len(executor.measurement_results) == 1


@pytest.mark.asyncio
async def test_executor_collects_nothing_when_routing_is_disabled(
    connected_manager, data_store
):
    run_id = data_store.start_run("executor_no_routers")
    executor = WorkflowExecutor(connected_manager, data_store=data_store,
                                run_id=run_id, routers=[])
    await executor.run(Workflow(name="unrouted", setup=[_eis_step()]))

    assert executor.state == ExecutorState.COMPLETED
    assert executor.measurement_results == []
