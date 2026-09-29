"""Tests for the optimizer subsystem (C1)."""

import itertools
import json

import pytest

from softae.errors import OptimizerError, SoftAEError
from softae.optimizers import BaseOptimizer, GridSearchOptimizer, RandomSearchOptimizer


# ── Helpers ──────────────────────────────────────────────────────────────────

SIMPLE_SPACE = {
    "x": {"type": "float", "low": 0.0, "high": 10.0},
    "y": {"type": "float", "low": -5.0, "high": 5.0},
}

#: Rung 3c's shape: a handful of EXPLICIT compositions, enumerated as a
#: categorical product rather than sampled from a continuous box.
FOUR_COMPOSITIONS = {
    "polymer": {"type": "categorical", "choices": ["PEO", "PVA"]},
    "salt": {"type": "categorical", "choices": ["LiCl", "LiTFSI"]},
}


def _walk(opt):
    """Every point an optimizer yields, in order, to exhaustion."""
    out = []
    while (p := opt.suggest()) is not None:
        out.append(p)
    return out

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


# ── Grid shuffle (rung 3c: explicit compositions, random cast order) ─────────


def test_grid_shuffle_default_off_is_byte_identical():
    """Off by default: the walk is still `itertools.product`, seed or no seed."""
    product = [{"polymer": p, "salt": s}
               for p in ("PEO", "PVA") for s in ("LiCl", "LiTFSI")]
    assert _walk(GridSearchOptimizer(FOUR_COMPOSITIONS)) == product
    assert _walk(GridSearchOptimizer(FOUR_COMPOSITIONS, seed=7)) == product
    assert _walk(GridSearchOptimizer(FOUR_COMPOSITIONS, shuffle=False)) == product
    assert GridSearchOptimizer(SIMPLE_SPACE, n_points=5)._grid == \
        GridSearchOptimizer(SIMPLE_SPACE, n_points=5, shuffle=True)._grid


def test_grid_shuffle_same_seed_same_order():
    """Two optimizers on one seed walk one order; the order is not the product."""
    a = GridSearchOptimizer(SIMPLE_SPACE, seed=17, n_points=5, shuffle=True)
    b = GridSearchOptimizer(SIMPLE_SPACE, seed=17, n_points=5, shuffle=True)
    walked = _walk(a)
    assert walked == _walk(b)
    assert walked != a._grid


def test_grid_shuffle_is_a_permutation_of_the_product():
    """Every product point exactly once — reordered, never resampled."""
    opt = GridSearchOptimizer(SIMPLE_SPACE, seed=3, n_points=5, shuffle=True)
    walked = _walk(opt)
    assert len(walked) == opt.n_grid_points == 25
    assert sorted(map(repr, walked)) == sorted(map(repr, opt._grid))
    assert walked != opt._grid


def test_grid_shuffle_survives_serialization_round_trip():
    """A resume walks the SAME permutation: the order rides in the checkpoint,
    so it holds even under ``seed=None``, where a redraw would differ."""
    opt = GridSearchOptimizer(FOUR_COMPOSITIONS, seed=11, shuffle=True)
    twin = GridSearchOptimizer(FOUR_COMPOSITIONS, seed=11, shuffle=True)
    for _ in range(2):
        opt.suggest()
        twin.suggest()
    restored = BaseOptimizer.from_dict(json.loads(json.dumps(opt.to_dict())))
    assert isinstance(restored, GridSearchOptimizer)
    assert _walk(restored) == _walk(twin)

    unseeded = GridSearchOptimizer(SIMPLE_SPACE, seed=None, n_points=5, shuffle=True)
    intended = [unseeded._grid[i] for i in unseeded._order]
    for _ in range(2):
        unseeded.suggest()
    resumed = BaseOptimizer.from_dict(json.loads(json.dumps(unseeded.to_dict())))
    assert _walk(resumed) == intended[2:]


def test_grid_shuffle_precedes_replication_layout():
    """Shuffling the grid does not disturb the interleaved replicate layout."""
    from softae.optimizers.replicated import ReplicatingOptimizer

    inner = GridSearchOptimizer(FOUR_COMPOSITIONS, seed=5, shuffle=True)
    batch = ReplicatingOptimizer(inner, 2, layout="interleaved").suggest_batch(8)
    assert len(batch) == 8
    assert all(batch[i] == batch[i + 4] for i in range(4))
    product = [dict(zip(("polymer", "salt"), c)) for c in
               itertools.product(("PEO", "PVA"), ("LiCl", "LiTFSI"))]
    assert sorted(map(repr, batch[:4])) == sorted(map(repr, product))


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
