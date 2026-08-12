"""Tests for the Arrhenius sweep module (Project 2).

Covers:
* ArrheniusSweepConfig  (tests 1–3)
* ArrheniusFitter       (tests 4–8)
* DataStore migration   (tests 9–10)
* ArrheniusSweep.build_workflow  (tests 11–13)
* ArrheniusSweep.run (integration) (tests 14–16)
* ArrheniusSweep.from_yaml  (tests 17–18)
* JSON sidecar export   (tests 19–20)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.arrhenius import ArrheniusFitter, ArrheniusResult, ArrheniusSweepConfig
from softae.core.data_store import DataStore
from softae.workflows.temp_eis_sweep import ArrheniusSweep


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_config() -> ArrheniusSweepConfig:
    return ArrheniusSweepConfig(
        channels=[1, 2],
        T_start=25.0,
        T_stop=55.0,
        T_step=10.0,
        dwell_s=0.0,
        electrode_geometry={"L_cm": 0.2, "t_cm": 0.175, "w_cm": 0.2},
    )


@pytest.fixture
def data_store(tmp_path: Path):
    ds = DataStore(tmp_path / "project")
    yield ds
    ds.close()


@pytest.fixture
def run_id(data_store: DataStore) -> str:
    return data_store.start_run("arrhenius_test")


@pytest.fixture
def mock_manager():
    """Unconnected manager for pure-sync tests."""
    from softae.drivers.mock_factory import create_mock_manager
    return create_mock_manager(config={})


@pytest.fixture
async def connected_manager():
    from softae.drivers.mock_factory import create_mock_manager

    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    return mgr


# ── ArrheniusSweepConfig ──────────────────────────────────────────────────────


def test_config_resolved_temperatures_step_mode():
    cfg = ArrheniusSweepConfig(channels=[1], T_start=25.0, T_stop=55.0, T_step=10.0)
    temps = cfg.resolved_temperatures()
    assert temps == [25.0, 35.0, 45.0, 55.0]


def test_config_resolved_temperatures_explicit():
    cfg = ArrheniusSweepConfig(
        channels=[1],
        temperatures=[20.0, 30.0, 50.0],
    )
    assert cfg.resolved_temperatures() == [20.0, 30.0, 50.0]


def test_config_validate_empty_channels_raises():
    cfg = ArrheniusSweepConfig(channels=[])
    with pytest.raises(ValueError, match="channels must not be empty"):
        cfg.validate()


def test_config_validate_out_of_range_channel_raises():
    cfg = ArrheniusSweepConfig(channels=[0])
    with pytest.raises(ValueError, match="outside valid range"):
        cfg.validate()


def test_config_validate_negative_dwell_raises():
    cfg = ArrheniusSweepConfig(channels=[1], dwell_s=-1.0)
    with pytest.raises(ValueError, match="dwell_s"):
        cfg.validate()


def test_config_validate_passes_good_config(simple_config):
    simple_config.validate()  # should not raise


# ── ArrheniusFitter ───────────────────────────────────────────────────────────


def _arrhenius_sigmas(temps_C: list[float], Ea_eV: float, ln_A: float) -> list[float]:
    """Generate synthetic σ values from ground-truth Arrhenius parameters."""
    KB = 8.617333e-5
    return [math.exp(ln_A - Ea_eV / (KB * (T + 273.15))) for T in temps_C]


def test_fitter_basic_fit():
    """Round-trip: generate synthetic data, fit, recover Eₐ within 5%."""
    temps = [25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    true_Ea = 0.40  # eV
    true_lnA = 10.0
    sigmas = _arrhenius_sigmas(temps, true_Ea, true_lnA)
    fitter = ArrheniusFitter()
    result = fitter.fit(temps, sigmas, channel=1, run_id="test")
    assert result.fit_success
    assert abs(result.Ea_eV - true_Ea) / true_Ea < 0.05
    assert result.R_squared > 0.999


def test_fitter_nan_conductivity_excluded():
    """NaN σ points should be silently excluded from the fit."""
    temps = [25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    sigmas = _arrhenius_sigmas(temps, 0.38, 9.5)
    sigmas[2] = float("nan")   # inject one bad point
    result = ArrheniusFitter().fit(temps, sigmas, channel=1, run_id="y")
    assert result.fit_success
    assert result.n_points == 5  # one excluded


def test_fitter_insufficient_points_returns_failed_result():
    result = ArrheniusFitter().fit([25.0], [1e-4], channel=1, run_id="z")
    assert not result.fit_success
    assert result.n_points < 2
    assert result.error_msg != ""


def test_fitter_all_nan_conductivities():
    temps = [25.0, 35.0, 45.0]
    sigmas = [float("nan")] * 3
    result = ArrheniusFitter().fit(temps, sigmas, channel=1, run_id="nan_test")
    assert not result.fit_success


# ── DataStore CRUD ────────────────────────────────────────────────────────────


def test_datastore_record_and_query_arrhenius(data_store, run_id):
    result = ArrheniusResult(
        channel=1,
        run_id=run_id,
        temperatures_C=[25.0, 35.0, 45.0],
        conductivities=[1e-4, 2e-4, 4e-4],
        Ea_eV=0.40,
        Ea_kJ_per_mol=38.6,
        ln_A=10.0,
        R_squared=0.999,
        T_min_C=25.0,
        T_max_C=45.0,
        n_points=3,
        fit_success=True,
        error_msg="",
    )
    row_id = data_store.record_arrhenius(run_id, result)
    assert isinstance(row_id, int)

    rows = data_store.query_arrhenius(run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["channel"] == 1
    assert abs(row["Ea_eV"] - 0.40) < 1e-6
    assert row["fit_success"] == 1
    assert row["temperatures_C"] == [25.0, 35.0, 45.0]
    assert row["conductivities"] == [1e-4, 2e-4, 4e-4]


def test_datastore_query_arrhenius_filter_by_channel(data_store, run_id):
    for ch in [1, 2, 3]:
        r = ArrheniusResult(channel=ch, run_id=run_id, n_points=0, fit_success=False)
        data_store.record_arrhenius(run_id, r)

    ch2_rows = data_store.query_arrhenius(run_id=run_id, channel=2)
    assert len(ch2_rows) == 1
    assert ch2_rows[0]["channel"] == 2


def test_datastore_arrhenius_table_created_on_init(tmp_path):
    """Opening a new DataStore must create the arrhenius_results table."""
    ds = DataStore(tmp_path / "new_project")
    tables = {
        row[0]
        for row in ds._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    ds.close()
    assert "arrhenius_results" in tables


# ── ArrheniusSweep.build_workflow ──────────────────────────────────────────────


def test_build_workflow_step_count(simple_config, mock_manager):
    """Verify the correct number of steps are generated."""
    sweep = ArrheniusSweep(simple_config, mock_manager)
    wf = sweep.build_workflow()
    # 4 temps × (set_temp + wait_temp + 2 eis) = 16 setup steps
    assert len(wf.setup) == 16
    # 1 teardown (restore_ambient)
    assert len(wf.teardown) == 1


def test_build_workflow_dag_depends_on(simple_config, mock_manager):
    """EIS steps at T0 must depend only on wait_temp_T0."""
    sweep = ArrheniusSweep(simple_config, mock_manager)
    wf = sweep.build_workflow()
    steps = {s.name: s for s in wf.setup}
    # T_start=25 → step name encodes rounded temperature and RH=0 (no RH sweep)
    eis_ch1_T0 = steps["eis_ch1_T25_RH0"]
    assert "wait_temp_T0" in eis_ch1_T0.depends_on


def test_build_workflow_temperature_chain(simple_config, mock_manager):
    """set_temp_T1 must depend on all EIS steps at T0."""
    sweep = ArrheniusSweep(simple_config, mock_manager)
    wf = sweep.build_workflow()
    steps = {s.name: s for s in wf.setup}
    set_T1 = steps["set_temp_T1"]
    assert "eis_ch1_T25_RH0" in set_T1.depends_on
    assert "eis_ch2_T25_RH0" in set_T1.depends_on


# ── ArrheniusSweep.run (integration) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_run_returns_one_result_per_channel(simple_config, connected_manager):
    sweep = ArrheniusSweep(simple_config, connected_manager)
    results = await sweep.run()
    assert len(results) == 2  # channels [1, 2]
    for r in results:
        assert isinstance(r, ArrheniusResult)
        assert r.channel in [1, 2]


@pytest.mark.asyncio
async def test_sweep_run_captures_eis_data(simple_config, connected_manager):
    """After run(), internal EIS cache must hold measurements for all (ch, t) pairs."""
    sweep = ArrheniusSweep(simple_config, connected_manager)
    await sweep.run()
    temps = simple_config.resolved_temperatures()
    for t_idx in range(len(temps)):
        for ch in simple_config.channels:
            assert (ch, t_idx) in sweep._eis_results, f"missing (ch={ch}, t_idx={t_idx})"


@pytest.mark.asyncio
async def test_sweep_run_stores_results_in_datastore(
    simple_config, connected_manager, data_store, run_id
):
    sweep = ArrheniusSweep(
        simple_config, connected_manager, data_store=data_store, run_id=run_id
    )
    await sweep.run()
    rows = data_store.query_arrhenius(run_id=run_id)
    assert len(rows) == len(simple_config.channels)
    assert all(r["model"] == "arrhenius" for r in rows)


def test_sweep_selects_vft_fitter(simple_config, mock_manager):
    from softae.analysis.vft import VftFitter

    simple_config.thermal_model = "vft"  # simple_config has 4 temps → valid
    sweep = ArrheniusSweep(simple_config, mock_manager)
    assert isinstance(sweep._fitter, VftFitter)


@pytest.mark.asyncio
async def test_sweep_run_stores_vft_rows(connected_manager, data_store, run_id):
    config = ArrheniusSweepConfig(
        channels=[1, 2],
        T_start=25.0, T_stop=55.0, T_step=10.0,  # 4 temps ≥ 3 for VFT
        dwell_s=0.0,
        thermal_model="vft",
        electrode_geometry={"L_cm": 0.2, "t_cm": 0.175, "w_cm": 0.2},
    )
    sweep = ArrheniusSweep(
        config, connected_manager, data_store=data_store, run_id=run_id
    )
    await sweep.run()
    rows = data_store.query_arrhenius(run_id=run_id)
    assert len(rows) == 2
    assert all(r["model"] == "vft" for r in rows)
    # VFT rows carry B/T0 and, via the activation-energy formalism, Eₐ too.
    for r in rows:
        assert r["B"] is not None
        if r["fit_success"]:
            assert r["Ea_eV"] is not None


# ── from_yaml ────────────────────────────────────────────────────────────────


@pytest.fixture
def arrhenius_yaml(tmp_path: Path) -> Path:
    content = """\
name: arrhenius_test
metadata:
  experiment_type: arrhenius_sweep
setup:
  - name: noop
    instrument: stage
    method: get_position

variables:
  channels: [1]
  T_start: 25.0
  T_stop: 45.0
  T_step: 10.0
  dwell_s: 0.0
  temp_tolerance: 0.5
  wait_timeout_s: 300.0
  eis_model: simpleSalt
  electrode_L_cm: 0.2
  electrode_t_cm: 0.175
  electrode_w_cm: 0.2
  eis_instrument: pico1
  temp_instrument: temp_controller
"""
    p = tmp_path / "arrhenius_test.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_from_yaml_builds_correct_config(arrhenius_yaml, mock_manager):
    sweep = ArrheniusSweep.from_yaml(arrhenius_yaml, mock_manager)
    assert sweep.config.channels == [1]
    assert sweep.config.T_start == 25.0
    assert sweep.config.T_stop == 45.0
    assert sweep.config.eis_model == "simpleSalt"
    geom = sweep.config.electrode_geometry
    assert geom is not None
    assert geom["L_cm"] == 0.2


def test_from_yaml_wrong_experiment_type_raises(tmp_path, mock_manager):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\n"
        "metadata:\n  experiment_type: standard_eis\n"
        "setup:\n  - name: x\n    instrument: stage\n    method: get_position\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="arrhenius_sweep"):
        ArrheniusSweep.from_yaml(bad, mock_manager)


# ── JSON sidecar ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_json_sidecar_written(simple_config, connected_manager, data_store):
    run_id = data_store.start_run("sidecar_test")
    sweep = ArrheniusSweep(
        simple_config, connected_manager, data_store=data_store, run_id=run_id
    )
    await sweep.run()

    sidecar = (
        Path(data_store.project_dir) / "runs" / run_id / "arrhenius_results.json"
    )
    assert sidecar.exists(), f"sidecar not found: {sidecar}"
    payload = json.loads(sidecar.read_text())
    assert isinstance(payload, list)
    assert len(payload) == len(simple_config.channels)
    for entry in payload:
        assert "channel" in entry
        assert "Ea_eV" in entry
        assert "temperatures_C" in entry


@pytest.mark.asyncio
async def test_json_sidecar_schema(simple_config, connected_manager, data_store):
    run_id = data_store.start_run("schema_test")
    sweep = ArrheniusSweep(
        simple_config, connected_manager, data_store=data_store, run_id=run_id
    )
    await sweep.run()

    sidecar = (
        Path(data_store.project_dir) / "runs" / run_id / "arrhenius_results.json"
    )
    payload = json.loads(sidecar.read_text())
    required_keys = {
        "channel", "run_id", "temperatures_C", "conductivities",
        "Ea_eV", "Ea_kJ_per_mol", "ln_A", "R_squared",
        "n_points", "fit_success", "error_msg",
    }
    for entry in payload:
        assert required_keys <= set(entry.keys()), (
            f"Missing keys: {required_keys - set(entry.keys())}"
        )
