"""P3 tests: VFT fitter, temperature-derived objectives, constraints, DataStore adapter, Pareto."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from softae.analysis.arrhenius import ArrheniusFitter
from softae.analysis.vft import VftFitter
from softae.campaigns import (
    BOCampaignConfig,
    GroundTruthDataset,
    build_campaign,
    pareto_indices,
    pareto_mask,
)
from softae.campaigns.adapters import DataStoreAdapter
from softae.campaigns.derived import ArrheniusEa, build_derived_objective
from softae.core.data_store import DataStore
from softae.errors import CampaignError

# ── VFT fitter ─────────────────────────────────────────────────────────────

def test_vft_fitter_recovers_known_params():
    A, B, T0 = 1.0, 600.0, 180.0   # σ = A exp(-B/(T-T0))
    temps_C = np.array([20, 40, 60, 80, 100], dtype=float)
    T_K = temps_C + 273.15
    sigmas = A * np.exp(-B / (T_K - T0))
    res = VftFitter().fit(list(temps_C), list(sigmas))
    assert res.fit_success
    assert res.B == pytest.approx(B, rel=0.05)
    assert res.T0_K == pytest.approx(T0, rel=0.05)
    assert res.R_squared > 0.999


def test_vft_insufficient_points():
    res = VftFitter().fit([25.0, 50.0], [1e-5, 2e-5])  # only 2 points
    assert not res.fit_success
    assert "Insufficient" in res.error_msg


# ── Temperature-derived objectives ─────────────────────────────────────────

def _arrhenius_tidy(compositions, Ea_by_comp, temps_C=(25, 45, 65, 85)):
    """Synthetic multi-temperature tidy frame with a known Ea per composition."""
    kB = 8.617333e-5
    rows = []
    for x in compositions:
        Ea = Ea_by_comp[x]
        for T in temps_C:
            T_K = T + 273.15
            sigma = math.exp(-Ea / (kB * T_K)) * 1e3  # arbitrary prefactor
            rows.append({
                "x": float(x), "rh_pct": 30.0, "temp_C": float(T),
                "replicate": 0, "conductivity": sigma,
                "point_id": f"x{x:g}_T{T:g}", "source": "synthetic",
            })
    return pd.DataFrame(rows)


def test_derived_dataset_recovers_per_composition_ea():
    Ea_by_comp = {1: 0.30, 2: 0.20, 3: 0.10}
    df = _arrhenius_tidy([1, 2, 3], Ea_by_comp)
    ds = GroundTruthDataset.from_tidy_derived(
        df, derived_objective=ArrheniusEa(), rail_sigma_ceiling=None
    )
    assert ds.size == 3                      # one candidate per composition (T folded in)
    assert ds.param_columns == ["x", "rh_pct"]  # temp_C removed
    # Minimising Ea → optimum is composition x=3 (Ea 0.10).
    best_params, best_val = ds.true_optimum(maximize=False)
    assert best_params["x"] == 3.0
    assert best_val == pytest.approx(0.10, abs=1e-3)


def test_derived_campaign_runs_end_to_end(tmp_path):
    df = _arrhenius_tidy([1, 2, 3, 4], {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1})
    ds = GroundTruthDataset.from_tidy_derived(df, derived_objective=ArrheniusEa(),
                                              rail_sigma_ceiling=None)
    cfg = BOCampaignConfig(
        objective_direction="minimize", temperature_objective="arrhenius_ea",
        n_initial=2, noiseless_oracle=True, patience=2,
    )
    res = build_campaign(cfg, dataset=ds).run()
    assert res.best_params["x"] == 4.0       # lowest Ea
    assert res.best_value == pytest.approx(0.1, abs=1e-3)


def test_build_derived_objective_unknown_raises():
    with pytest.raises(CampaignError, match="unknown temperature_objective"):
        build_derived_objective("nope")


def test_arrhenius_derived_matches_fitter():
    # Sanity: derived ArrheniusEa equals a direct ArrheniusFitter on the series.
    Ea = 0.25
    kB = 8.617333e-5
    temps = [25.0, 50.0, 75.0]
    sig = [math.exp(-Ea / (kB * (t + 273.15))) for t in temps]
    val, var, ok = ArrheniusEa().compute(np.array(temps), np.array(sig))
    direct = ArrheniusFitter().fit(temps, sig)
    assert ok and var > 0
    assert val == pytest.approx(direct.Ea_eV, rel=1e-6)


# ── Constraint / feasibility hook ───────────────────────────────────────────

def test_feasible_filters_pool():
    df = pd.DataFrame([
        {"x": float(x), "rh_pct": 30.0, "replicate": 0,
         "conductivity": 10.0 ** (-x), "point_id": f"x{x}", "source": "s"}
        for x in range(1, 6)
    ])
    full = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    filtered = GroundTruthDataset.from_tidy(
        df, rail_sigma_ceiling=None, feasible=lambda p: p["x"] <= 3
    )
    assert full.size == 5
    assert filtered.size == 3
    assert all(c.params["x"] <= 3 for c in filtered.cells)


# ── DataStore adapter ───────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path):
    with DataStore(tmp_path / "proj") as ds:
        yield ds


def _seed_store_run(store) -> str:
    """Populate a run with two channels × two replicate measurements + fits.

    ``measurements`` and ``fit_results`` are seeded directly — their ``record_*``
    helpers require full EISResult/FitResult objects — but conditions go through
    :meth:`DataStore.record_conditions`, because since schema epoch 4 that writer
    is where a row's temperature is resolved. A raw INSERT here would produce
    rows no production writer can produce, and the fixture would then be testing
    the adapter's fallback path in every test that uses it. One test below does
    exactly that, deliberately and alone.
    """
    run_id = store.start_run("ht", campaign="dev")
    conn = store._conn
    ts = "2026-01-01T00:00:00"
    for ch, sigma in ((1, 1e-5), (2, 5e-5)):
        store.record_doe_parameter(run_id, ch, 0, {"x": float(ch), "rh_pct": 30.0})
        for rep in range(2):
            cur = conn.execute(
                "INSERT INTO measurements (run_id, channel, timestamp, npts) "
                "VALUES (?, ?, ?, ?)",
                (run_id, ch, ts, 10),
            )
            mid = cur.lastrowid
            store.record_conditions(
                mid, "measurement", chamber_air_C=35.0, rh_pv_pct=30.0)
            conn.execute(
                "INSERT INTO fit_results (measurement_id, run_id, model_name, R0, R1, "
                "sigma_S_per_cm, success, fitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, run_id, "simpleSalt", 10.0, 1.0 / sigma,
                 sigma * (1 + 0.01 * rep), 1, ts),
            )
    conn.commit()
    return run_id


def test_datastore_adapter_to_tidy(store):
    run_id = _seed_store_run(store)
    df = DataStoreAdapter(store, run_id).to_tidy()
    assert len(df) == 4                       # 2 channels × 2 replicates
    assert df["point_id"].nunique() == 2      # composition x=1, x=2
    assert {"x", "rh_pct", "temp_C", "conductivity"} <= set(df.columns)
    # adapter feeds the standard pipeline
    ds = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    assert ds.size == 2


def test_datastore_adapter_empty_run_raises(store):
    run_id = store.start_run("empty", campaign="dev")
    with pytest.raises(CampaignError, match="no fitted conductivity"):
        DataStoreAdapter(store, run_id).to_tidy()


# ── Which thermometer stamps a tidy row ─────────────────────────────────────
#
# This adapter consulted the chamber-air probe FIRST and never read the stage PV
# at all — `resolve_temperature_C`'s precedence exactly inverted, on every row it
# emitted, and worth up to 42 C on the run that motivated the fix.

def _seed_one_measurement(store, **conditions) -> str:
    run_id = store.start_run("thermometry", campaign="dev")
    conn = store._conn
    ts = "2026-01-01T00:00:00"
    cur = conn.execute(
        "INSERT INTO measurements (run_id, channel, timestamp, npts) VALUES (?, 1, ?, 10)",
        (run_id, ts),
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO fit_results (measurement_id, run_id, model_name, R0, R1, "
        "sigma_S_per_cm, success, fitted_at) VALUES (?, ?, 'simpleSalt', 10.0, 1e5, 1e-5, 1, ?)",
        (mid, run_id, ts),
    )
    store.record_conditions(mid, "measurement", **conditions)
    conn.commit()
    return run_id


def test_datastore_adapter_temp_c_prefers_the_stage_pv_and_names_it(store):
    run_id = _seed_one_measurement(
        store, stage_temp_pv_C=85.0, stage_temp_sp_C=85.0, chamber_air_C=42.8)
    row = DataStoreAdapter(store, run_id).to_tidy().iloc[0]
    assert row["temp_C"] == pytest.approx(85.0)
    assert row["temp_source"] == "stage_pv"


def test_datastore_adapter_temp_c_falls_back_to_the_setpoint_then_the_air(store):
    run_id = _seed_one_measurement(
        store, stage_temp_sp_C=65.0, chamber_air_C=36.6)
    row = DataStoreAdapter(store, run_id).to_tidy().iloc[0]
    assert row["temp_C"] == pytest.approx(65.0)
    assert row["temp_source"] == "stage_sp"

    air_run = _seed_one_measurement(store, chamber_air_C=29.1)
    air_row = DataStoreAdapter(store, air_run).to_tidy().iloc[0]
    assert air_row["temp_C"] == pytest.approx(29.1)
    assert air_row["temp_source"] == "chamber_air"


def test_datastore_adapter_temp_source_is_carried_but_is_not_a_coordinate(store):
    """The label travels with the number without becoming part of identity."""
    run_id = _seed_store_run(store)
    df = DataStoreAdapter(store, run_id).to_tidy()
    assert "temp_source" in df.columns
    assert set(df["temp_source"]) == {"chamber_air"}   # the fixture writes air only
    assert not df["point_id"].str.contains("temp_source").any()
    # And it must not be inferred as a GP coordinate downstream.
    ds = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    assert "temp_source" not in ds.param_columns


def test_datastore_adapter_no_thermometer_at_all_is_nan_not_a_number(store):
    run_id = _seed_one_measurement(store, rh_pv_pct=30.0)
    row = DataStoreAdapter(store, run_id).to_tidy().iloc[0]
    assert math.isnan(row["temp_C"])
    assert row["temp_source"] == "unavailable"


def test_datastore_adapter_reads_the_stored_resolution_not_its_own(store):
    """Epoch 4: the adapter consumes the row's answer instead of re-deriving it.

    Pinned by writing a stored pair the *source* columns cannot produce. If the
    adapter still resolved from the raw reads it would return 42.8/chamber_air;
    reading the stored column, it returns what the row says. No production
    writer creates this disagreement — that is why it makes a clean probe.
    """
    run_id = _seed_one_measurement(store, chamber_air_C=42.8)
    store._conn.execute(
        "UPDATE conditions SET temperature_C = 85.0, temperature_source = 'stage_pv' "
        "WHERE run_id = ?", (run_id,)
    )
    store._conn.commit()

    row = DataStoreAdapter(store, run_id).to_tidy().iloc[0]
    assert row["temp_C"] == pytest.approx(85.0)
    assert row["temp_source"] == "stage_pv"


def test_datastore_adapter_falls_back_to_the_resolver_when_the_row_has_no_source(store):
    """A row with no stored label — raw INSERT, or a pre-epoch-4 binary.

    The fallback is the same authority the writer would have used, so the answer
    must be identical to the stored-column path's: stage PV over the air probe.
    """
    run_id = store.start_run("stale_writer", campaign="dev")
    conn = store._conn
    ts = "2026-01-01T00:00:00"
    cur = conn.execute(
        "INSERT INTO measurements (run_id, channel, timestamp, npts) VALUES (?, 1, ?, 10)",
        (run_id, ts),
    )
    mid = cur.lastrowid
    conn.execute(
        "INSERT INTO fit_results (measurement_id, run_id, model_name, R0, R1, "
        "sigma_S_per_cm, success, fitted_at) VALUES (?, ?, 'simpleSalt', 10.0, 1e5, 1e-5, 1, ?)",
        (mid, run_id, ts),
    )
    conn.execute(
        "INSERT INTO conditions (measurement_id, run_id, stage, timestamp, "
        "stage_temp_pv_C, chamber_air_C) VALUES (?, ?, 'measurement', ?, 85.0, 42.8)",
        (mid, run_id, ts),
    )
    conn.commit()

    row = DataStoreAdapter(store, run_id).to_tidy().iloc[0]
    assert row["temp_C"] == pytest.approx(85.0)
    assert row["temp_source"] == "stage_pv"


# ── Pareto utilities ────────────────────────────────────────────────────────

def test_pareto_mask_minimize():
    pts = np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mask = pareto_mask(pts, maximize=[False, False])
    # (1,2) and (2,1) are non-dominated; (2,2) and (3,3) are dominated.
    assert mask.tolist() == [True, True, False, False]


def test_pareto_indices_mixed_directions():
    # maximize first, minimize second
    pts = np.array([[10.0, 5.0], [10.0, 8.0], [12.0, 9.0]])
    idx = pareto_indices(pts, maximize=[True, False])
    assert 0 in idx           # (10,5) best on obj2 among the 10s
    assert 1 not in idx       # dominated by (10,5)
