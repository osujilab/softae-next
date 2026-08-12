"""Tests for the optimizer subsystem (C1)."""

import pytest

from softae.errors import OptimizerError, SoftAEError
from softae.optimizers import BaseOptimizer, GridSearchOptimizer, RandomSearchOptimizer


# ── Helpers ──────────────────────────────────────────────────────────────────

SIMPLE_SPACE = {
    "x": {"type": "float", "low": 0.0, "high": 10.0},
    "y": {"type": "float", "low": -5.0, "high": 5.0},
}

MIXED_SPACE = {
    "temp": {"type": "float", "low": 25.0, "high": 80.0},
    "rh":   {"type": "int",   "low": 30,   "high": 90},
    "solvent": {"type": "categorical", "choices": ["water", "DMSO", "DMF"]},
}


# ── ABC tests ────────────────────────────────────────────────────────────────


def test_base_optimizer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseOptimizer(SIMPLE_SPACE)


@pytest.mark.parametrize("space,match", [
    ({}, "non-empty"),
    ({"x": {"low": 0, "high": 1}}, "missing 'type'"),
    ({"x": {"type": "complex", "low": 0, "high": 1}}, "unknown type"),
    ({"x": {"type": "float", "low": 10, "high": 5}}, "low.*must be < high"),
    ({"s": {"type": "categorical", "choices": []}}, "non-empty 'choices'"),
])
def test_invalid_parameter_space(space, match):
    with pytest.raises(OptimizerError, match=match):
        GridSearchOptimizer(space)


def test_invalid_objective():
    with pytest.raises(OptimizerError, match="maximize.*minimize"):
        GridSearchOptimizer(SIMPLE_SPACE, objective="something")


# ── GridSearchOptimizer tests ────────────────────────────────────────────────


def test_grid_total_points():
    opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=5)
    count = 0
    while opt.suggest() is not None:
        count += 1
    assert count == 25  # 5 × 5


def test_grid_suggest_tell_cycle():
    opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=3)
    for i in range(9):  # 3 × 3
        p = opt.suggest()
        assert p is not None
        opt.tell(p, float(i))
    assert opt.n_trials == 9
    assert len(opt.history) == 9


def test_grid_exhaustion_returns_none():
    opt = GridSearchOptimizer(SIMPLE_SPACE, n_points=2)
    for _ in range(4):  # 2 × 2
        opt.suggest()
    assert opt.suggest() is None
    assert opt.suggest() is None  # stays None


@pytest.mark.parametrize("objective,best_fn", [
    ("maximize", max),
    ("minimize", min),
])
def test_grid_best(objective, best_fn):
    opt = GridSearchOptimizer(SIMPLE_SPACE, objective=objective, n_points=2)
    for _ in range(4):
        p = opt.suggest()
        opt.tell(p, p["x"] + p["y"])
    _, best_val = opt.best()
    assert best_val == best_fn(v for _, v in opt.history)


def test_grid_int_dedup():
    space = {"n": {"type": "int", "low": 1, "high": 3}}
    opt = GridSearchOptimizer(space, n_points=10)
    points = []
    while (p := opt.suggest()) is not None:
        points.append(p["n"])
    assert sorted(set(points)) == [1, 2, 3]
    assert len(points) == 3  # capped, not 10


def test_grid_categorical_ignores_n_points():
    space = {"s": {"type": "categorical", "choices": ["a", "b"]}}
    opt = GridSearchOptimizer(space, n_points=100)
    points = []
    while (p := opt.suggest()) is not None:
        points.append(p["s"])
    assert set(points) == {"a", "b"}
    assert len(points) == 2  # not 100


# ── RandomSearchOptimizer tests ──────────────────────────────────────────────


def test_random_reproducibility():
    opt1 = RandomSearchOptimizer(MIXED_SPACE, seed=42, n_trials=5)
    opt2 = RandomSearchOptimizer(MIXED_SPACE, seed=42, n_trials=5)
    for _ in range(5):
        assert opt1.suggest() == opt2.suggest()


def test_random_budget_exhaustion():
    opt = RandomSearchOptimizer(SIMPLE_SPACE, seed=0, n_trials=3)
    for _ in range(3):
        p = opt.suggest()
        assert p is not None
        opt.tell(p, 0.0)
    assert opt.suggest() is None


def test_random_suggest_within_bounds():
    opt = RandomSearchOptimizer(MIXED_SPACE, seed=7, n_trials=50)
    for _ in range(50):
        p = opt.suggest()
        assert 25.0 <= p["temp"] <= 80.0
        assert 30 <= p["rh"] <= 90
        assert isinstance(p["rh"], int)
        assert p["solvent"] in ("water", "DMSO", "DMF")

def test_random_best_tracking():
    opt = RandomSearchOptimizer(SIMPLE_SPACE, objective="maximize", seed=1, n_trials=10)
    for _ in range(10):
        p = opt.suggest()
        opt.tell(p, p["x"] * 2)
    best_params, best_val = opt.best()
    assert best_val == max(v for _, v in opt.history)


def test_random_history_and_n_trials():
    opt = RandomSearchOptimizer(SIMPLE_SPACE, seed=0, n_trials=5)
    for i in range(5):
        p = opt.suggest()
        opt.tell(p, float(i))
        assert opt.n_trials == i + 1
        assert len(opt.history) == i + 1


# ── Error hierarchy test ────────────────────────────────────────────────────


def test_optimizer_error_is_softae_error():
    err = OptimizerError("test")
    assert isinstance(err, SoftAEError)
