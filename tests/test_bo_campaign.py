"""Tests for the simulated BO campaign suite: parser, dataset, rails, runner, config."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from softae.campaigns import (
    AggregatedTxtAdapter,
    BOCampaignConfig,
    GroundTruthDataset,
    Log10Sigma,
    Oracle,
    build_campaign,
    detect_rails,
    run_campaign,
)
from softae.errors import CampaignError
from tests.campaign_helpers import SEED_DATASET, make_tidy, write_small_aggregated

_seed_missing = not os.path.exists(SEED_DATASET)
needs_seed = pytest.mark.skipif(_seed_missing, reason="seed dataset file not present")


# ── Parser (synthetic, always runs) ───────────────────────────────────────

def test_parser_synthetic_shape(tmp_path):
    p = tmp_path / "small.txt"
    layout = write_small_aggregated(p)
    adapter = AggregatedTxtAdapter(
        p,
        rh_levels=layout["rh_levels"],
        eo_li_levels=layout["eo_li_levels"],
        silica_levels=layout["silica_levels"],
        n_replicates=layout["n_replicates"],
    )
    df = adapter.to_tidy()
    assert len(df) == 16
    assert df["point_id"].nunique() == 8  # 2 RH x 2 EO x 2 silica
    # conductivity matches the flat block we wrote
    assert df["conductivity"].tolist() == layout["conductivity"]


def test_parser_index_decode(tmp_path):
    p = tmp_path / "small.txt"
    layout = write_small_aggregated(p)
    adapter = AggregatedTxtAdapter(
        p,
        rh_levels=layout["rh_levels"],
        eo_li_levels=layout["eo_li_levels"],
        silica_levels=layout["silica_levels"],
        n_replicates=layout["n_replicates"],
    )
    df = adapter.to_tidy()
    # Flat index 0 → first RH, first EO, first silica, replicate 0.
    row0 = df.iloc[0]
    assert (row0["rh_pct"], row0["eo_li_ratio"], row0["silica_vol_frac"], row0["replicate"]) == (
        10.0, 40.0, 0.0, 0,
    )
    # Index 8 → second RH block starts (2 EO x 2 silica x 2 rep = 8 per RH).
    row8 = df.iloc[8]
    assert row8["rh_pct"] == 30.0


def test_parser_wrong_count_raises(tmp_path):
    p = tmp_path / "small.txt"
    write_small_aggregated(p)
    # Declare a layout requiring more values than the file holds.
    adapter = AggregatedTxtAdapter(p, n_replicates=4)  # expects 64, file has 16
    with pytest.raises(CampaignError, match="values, expected"):
        adapter.to_tidy()


def test_parser_missing_file_raises(tmp_path):
    adapter = AggregatedTxtAdapter(tmp_path / "nope.txt")
    with pytest.raises(CampaignError, match="not found"):
        adapter.to_tidy()


# ── Seed-dataset specifics ────────────────────────────────────────────────

@needs_seed
def test_seed_parser_yields_64_rows():
    df = AggregatedTxtAdapter(SEED_DATASET).to_tidy()
    assert len(df) == 64
    assert df["point_id"].nunique() == 32


@needs_seed
def test_seed_rail_rows_detected():
    df = AggregatedTxtAdapter(SEED_DATASET).to_tidy()
    rails = detect_rails(df)
    # Exactly the two σ≈0.1 / fitted_Z=100 rows (flat indices 22 and 54).
    assert rails.sum() == 2
    assert bool(rails.iloc[22]) and bool(rails.iloc[54])
    assert df.iloc[22]["fitted_Z"] == pytest.approx(100.0)


@needs_seed
def test_seed_rail_excluded_from_optimum_toggle():
    df = AggregatedTxtAdapter(SEED_DATASET).to_tidy()
    ds_excl = GroundTruthDataset.from_tidy(df, exclude_rails_from_optimum=True)
    ds_incl = GroundTruthDataset.from_tidy(df, exclude_rails_from_optimum=False)
    _, y_excl = ds_excl.true_optimum(maximize=True)
    pt_incl, y_incl = ds_incl.true_optimum(maximize=True)
    # Including rails inflates the optimum toward the σ≈0.1 rail value.
    assert y_incl > y_excl
    # And the inflated optimum is a rail-containing cell.
    pid = ds_incl.point_id_for(pt_incl)
    assert ds_incl._by_id[pid].is_rail


# ── Dataset aggregation / oracle ──────────────────────────────────────────

def test_dataset_aggregates_replicates():
    df = make_tidy([1, 2, 3], lambda x, r: 10.0 ** (-x))
    ds = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    assert ds.size == 3
    pv = ds.pool_variance()
    assert all(v > 0 and math.isfinite(v) for v in pv.values())


def test_oracle_noiseless_returns_mean():
    df = make_tidy([1, 2, 3], lambda x, r: 10.0 ** (-x))
    ds = GroundTruthDataset.from_tidy(df, transform=Log10Sigma(), rail_sigma_ceiling=None)
    oracle = Oracle(ds, noiseless=True)
    rng = np.random.RandomState(0)
    for params in ds.pool_points():
        obs = oracle.reveal(params, rng)
        assert obs.value == pytest.approx(ds.true_value(obs.point_id))


def test_oracle_noise_is_seeded():
    df = make_tidy([1, 2, 3], lambda x, r: 10.0 ** (-x) * (1 + 0.3 * r))
    ds = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    oracle = Oracle(ds, noiseless=False)
    p = ds.pool_points()[0]
    v1 = oracle.reveal(p, np.random.RandomState(7)).value
    v2 = oracle.reveal(p, np.random.RandomState(7)).value
    assert v1 == v2


def test_oracle_unknown_point_raises():
    df = make_tidy([1, 2], lambda x, r: 10.0 ** (-x))
    ds = GroundTruthDataset.from_tidy(df, rail_sigma_ceiling=None)
    oracle = Oracle(ds, noiseless=True)
    with pytest.raises(CampaignError, match="not in dataset pool"):
        oracle.reveal({"x": 999.0, "rh_pct": 30.0}, np.random.RandomState(0))


# ── Runner / convergence ──────────────────────────────────────────────────

def test_runner_converges_to_known_optimum():
    # Monotone surface: log10 σ = x, so the optimum is the largest x.
    df = make_tidy([1, 2, 3, 4, 5], lambda x, r: 10.0 ** x)
    ds = GroundTruthDataset.from_tidy(df, transform=Log10Sigma(), rail_sigma_ceiling=None)
    cfg = BOCampaignConfig(n_initial=2, seed=0, rel_tol=1e-2, patience=2,
                           noiseless_oracle=True)
    runner = build_campaign(cfg, dataset=ds)
    res = runner.run()
    assert res.converged
    assert res.steps_to_tolerance is not None and res.steps_to_tolerance <= ds.size
    assert res.best_value == pytest.approx(5.0)
    # Simple regret is monotone non-increasing.
    curve = res.regret_curve()
    assert all(curve[i + 1] <= curve[i] + 1e-9 for i in range(len(curve) - 1))


@needs_seed
def test_runner_on_seed_dataset():
    cfg = BOCampaignConfig(
        dataset_path=SEED_DATASET, n_initial=5, seed=1, noiseless_oracle=True
    )
    res = run_campaign(cfg)
    assert res.n_steps >= 1
    # The reported optimum must not be a rail (default excludes rails).
    assert res.true_optimum_value < -1.0  # rails sit at log10(0.1) = -1.0


# ── Config ────────────────────────────────────────────────────────────────

def test_config_roundtrip():
    cfg = BOCampaignConfig(acquisition="ei", n_initial=3, seed=42, k_fit=2.0)
    cfg2 = BOCampaignConfig.from_json(cfg.to_json())
    assert cfg2.acquisition == "ei"
    assert cfg2.n_initial == 3
    assert cfg2.k_fit == 2.0


def test_config_from_json_ignores_unknown_keys():
    cfg = BOCampaignConfig.from_json('{"acquisition": "ucb", "totally_new_field": 5}')
    assert cfg.acquisition == "ucb"


@pytest.mark.parametrize("kwargs,match", [
    ({"backend": "nope"}, "backend"),
    ({"acquisition": "nope"}, "acquisition"),
    ({"transform": "nope"}, "transform"),
    ({"objective_direction": "sideways"}, "objective_direction"),
    ({"noise_channel": "nope"}, "noise_channel"),
    ({"rel_tol": -1.0}, "rel_tol"),
    ({"coverage_tol": 2.0}, "coverage_tol"),
    ({"n_initial": 0}, "n_initial"),
])
def test_config_validate_rejects_bad_values(kwargs, match):
    cfg = BOCampaignConfig(**kwargs)
    with pytest.raises(CampaignError, match=match):
        cfg.validate()


def test_config_validate_n_initial_vs_pool():
    cfg = BOCampaignConfig(n_initial=10)
    with pytest.raises(CampaignError, match="must be <"):
        cfg.validate(pool_size=5)
