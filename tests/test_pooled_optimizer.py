"""Tests for PooledBayesianOptimizer (pool-restricted, noise-aware BO)."""

from __future__ import annotations

import numpy as np
import pytest

from softae.errors import CampaignError, OptimizerError
from softae.optimizers import PooledBayesianOptimizer

SPACE = {"x": {"type": "float", "low": 1.0, "high": 5.0}}
POOL = [{"x": float(i)} for i in range(1, 6)]  # 5 candidates


def _opt(**kw):
    return PooledBayesianOptimizer(SPACE, pool=POOL, n_initial=2, seed=0, **kw)


def test_pool_never_repeats():
    opt = _opt()
    seen = []
    while (p := opt.suggest()) is not None:
        opt.tell(p, p["x"])
        seen.append(p["x"])
    assert sorted(seen) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(seen) == len(set(seen))


def test_pool_exhaustion_returns_none():
    opt = _opt()
    for _ in range(5):
        p = opt.suggest()
        opt.tell(p, p["x"])
    assert opt.suggest() is None
    assert opt.suggest() is None  # stays None


def test_tell_unknown_point_raises():
    opt = _opt()
    with pytest.raises(CampaignError, match="not in the remaining pool"):
        opt.tell({"x": 99.0}, 1.0)


def test_tell_twice_same_point_raises():
    opt = _opt()
    opt.tell({"x": 1.0}, 1.0)
    with pytest.raises(CampaignError, match="not in the remaining pool"):
        opt.tell({"x": 1.0}, 1.0)


def test_pool_finds_known_optimum():
    # Objective increases with x; the maximum pool point is x=5.
    opt = PooledBayesianOptimizer(
        SPACE, pool=POOL, objective="maximize", n_initial=2, seed=0
    )
    for _ in range(len(POOL)):
        p = opt.suggest()
        if p is None:
            break
        opt.tell(p, p["x"])  # noiseless monotone objective
    best_params, best_val = opt.best()
    assert best_params["x"] == 5.0
    assert best_val == 5.0


def test_heteroscedastic_alpha_aligned_to_history():
    opt = PooledBayesianOptimizer(SPACE, pool=POOL, n_initial=2, seed=0)
    opt._pool_variance = {opt._key(p): 0.01 * p["x"] for p in POOL}
    for _ in range(3):
        p = opt.suggest()
        opt.tell(p, p["x"])
    alpha = opt._alpha_for_history()
    assert alpha is not None
    assert alpha.shape[0] == opt.n_trials
    # values correspond to the per-point variance of the sampled points
    expected = [0.01 * p["x"] for p, _ in opt.history]
    assert np.allclose(alpha, expected)


def test_use_alpha_false_withholds_from_gp():
    opt = PooledBayesianOptimizer(SPACE, pool=POOL, n_initial=2, seed=0, use_alpha=False)
    opt._pool_variance = {opt._key(p): 0.5 for p in POOL}
    opt.tell({"x": 1.0}, 1.0)
    assert opt._alpha_for_history() is None  # GP gets no per-point alpha


def test_invalid_parameter_space_still_raises():
    with pytest.raises(OptimizerError, match="low.*must be < high"):
        PooledBayesianOptimizer(
            {"x": {"type": "float", "low": 5.0, "high": 1.0}}, pool=POOL
        )


def test_empty_pool_raises():
    with pytest.raises(CampaignError, match="non-empty"):
        PooledBayesianOptimizer(SPACE, pool=[])
