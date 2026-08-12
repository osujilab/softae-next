"""P1 tests: benchmark harness, persistence, model-accuracy stopping, optional backends."""

from __future__ import annotations

import importlib.util

import pytest

from softae.campaigns import (
    BOCampaignConfig,
    GroundTruthDataset,
    Log10Sigma,
    build_campaign,
    record_campaign,
    run_grid,
)
from softae.core.data_store import DataStore
from softae.optimizers.surrogates import make_backend
from tests.campaign_helpers import make_tidy

# ── shared synthetic dataset ───────────────────────────────────────────────

def _monotone_dataset(n=6):
    df = make_tidy(range(1, n + 1), lambda x, r: 10.0 ** x)
    return GroundTruthDataset.from_tidy(
        df, transform=Log10Sigma(), rail_sigma_ceiling=None
    )


# ── Benchmark harness ──────────────────────────────────────────────────────

def test_benchmark_grid_row_count():
    ds = _monotone_dataset()
    base = BOCampaignConfig(n_initial=2, noiseless_oracle=True, patience=2)
    result = run_grid(base, seeds=[0, 1, 2], acquisitions=["ucb", "ei"], dataset=ds)
    # 3 seeds x 2 acquisitions = 6 rows.
    assert len(result.rows) == 6
    assert set(result.rows["acquisition"]) == {"ucb", "ei"}
    assert (result.rows["error"] == "").all()


def test_benchmark_aggregate_groups():
    ds = _monotone_dataset()
    base = BOCampaignConfig(n_initial=2, noiseless_oracle=True, patience=2)
    result = run_grid(base, seeds=[0, 1, 2, 3], acquisitions=["ucb", "ei"], dataset=ds)
    agg = result.aggregate()
    # One row per (acquisition, backend, transform, direction) = 2 here.
    assert len(agg) == 2
    assert {"frac_converged", "stt_mean", "regret_auc_mean"} <= set(agg.columns)
    assert (agg["n_campaigns"] == 4).all()


def test_benchmark_records_uninstalled_backend_as_error():
    ds = _monotone_dataset()
    base = BOCampaignConfig(n_initial=2, noiseless_oracle=True)
    result = run_grid(base, seeds=[0], backends=["sklearn", "botorch"], dataset=ds)
    by_backend = {r["backend"]: r for r in result.rows.to_dict("records")}
    assert by_backend["sklearn"]["error"] == ""
    if importlib.util.find_spec("botorch") is None:
        assert by_backend["botorch"]["error"] != ""  # gracefully recorded, not raised


# ── Persistence ────────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path):
    with DataStore(tmp_path / "proj") as ds:
        yield ds


def test_record_campaign_writes_doe_rows(store):
    ds = _monotone_dataset()
    cfg = BOCampaignConfig(n_initial=2, noiseless_oracle=True, patience=2,
                           acquisition="ucb")
    result = build_campaign(cfg, dataset=ds).run()

    run_id = record_campaign(store, result, config=cfg)
    rows = store.query_doe_parameters(run_id=run_id)
    assert len(rows) == result.n_steps
    assert all(r["acquisition_fn"] == "ucb" for r in rows)
    # iterations are 0-based and contiguous
    assert sorted(r["iteration"] for r in rows) == list(range(result.n_steps))


def test_record_campaign_writes_sidecar_and_finishes(store):
    ds = _monotone_dataset()
    cfg = BOCampaignConfig(n_initial=2, noiseless_oracle=True, patience=2)
    result = build_campaign(cfg, dataset=ds).run()
    run_id = record_campaign(store, result, config=cfg)

    sidecar = store.run_dir(run_id) / "bo_campaign_result.json"
    assert sidecar.exists()
    # run is registered under the bo_campaign campaign and marked finished
    runs = {r["run_id"]: r for r in store.query_runs(campaign="bo_campaign")}
    assert run_id in runs
    assert runs[run_id]["status"] in ("done", "stopped")


# ── Model-accuracy stopping mode ───────────────────────────────────────────

def test_model_accuracy_stopping_mode_runs():
    ds = _monotone_dataset(n=6)
    cfg = BOCampaignConfig(
        n_initial=2, noiseless_oracle=True,
        stopping_mode="model_accuracy", rmse_tol=10.0, coverage_tol=0.0,
    )
    # Loose tolerances → should converge on model accuracy quickly.
    result = build_campaign(cfg, dataset=ds).run()
    assert result.converged
    assert result.steps_to_tolerance is not None


# ── Optional backends (skip when absent) ───────────────────────────────────

@pytest.mark.skipif(
    importlib.util.find_spec("botorch") is None, reason="botorch not installed"
)
def test_botorch_backend_fit_predict():
    import numpy as np
    be = make_backend("botorch")
    X = np.linspace(0, 1, 6).reshape(-1, 1)
    y = X[:, 0] ** 2
    be.fit(X, y, alpha=1e-6)
    mu, sigma = be.predict(X)
    assert mu.shape == (6,) and sigma.shape == (6,)
    assert np.all(sigma >= 0)


@pytest.mark.skipif(
    importlib.util.find_spec("gpytorch") is None, reason="gpytorch not installed"
)
def test_gpytorch_backend_fit_predict():
    import numpy as np
    be = make_backend("gpytorch")
    X = np.linspace(0, 1, 6).reshape(-1, 1)
    y = X[:, 0] ** 2
    be.fit(X, y, alpha=np.full(6, 1e-6))
    mu, sigma = be.predict(X)
    assert mu.shape == (6,) and sigma.shape == (6,)
