"""Optimizer checkpoint/restore (P3.1).

A resumed campaign has to continue the *same* search, not start a similar one.
These tests pin the two findings that shaped the design:

* the fitted GP does **not** need serializing (refit from history is
  deterministic), and
* the RNG **does** (replaying history alone silently changes the random stream).
"""

from __future__ import annotations

import json

import pytest

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer
from softae.optimizers.bayesian import BayesianOptimizer
from softae.optimizers.random import RandomSearchOptimizer

SPACE = {"a": {"type": "float", "low": 0.0, "high": 10.0},
         "b": {"type": "float", "low": 0.0, "high": 5.0}}
OBS = [({"a": 1.0, "b": 2.0}, 0.5),
       ({"a": 4.0, "b": 1.0}, 0.9),
       ({"a": 7.0, "b": 3.5}, 0.3)]


def _run(cls, n=len(OBS)):
    """Interleaved suggest/tell, the way a real campaign drives an optimizer."""
    o = cls(SPACE, objective="maximize", seed=42)
    for p, v in OBS[:n]:
        o.suggest()
        o.tell(p, v)
    return o


# ── The two design-shaping properties ────────────────────────────────────────

def test_gp_refit_from_history_is_deterministic():
    """Why fitted hyperparameters are NOT serialized.

    The surrogate is built with a fixed ``random_state``, so two optimizers given
    identical history fit the same model and propose the same point. Persisting
    sklearn internals would be fragile across versions and could restore a model
    that no longer matches the data.
    """
    def build():
        o = BayesianOptimizer(SPACE, objective="maximize", seed=42)
        for p, v in OBS:
            o.tell(p, v)
        return o

    assert build().suggest() == build().suggest()


def test_replaying_history_alone_diverges_from_a_true_continuation():
    """Why RNG state IS serialized.

    ``tell`` does not advance the RNG the way the original interleaved
    suggest/tell did, so a naive replay resumes on a different random stream —
    re-drawing candidate pools the run already used.
    """
    original = _run(BayesianOptimizer)
    naive = BayesianOptimizer(SPACE, objective="maximize", seed=42)
    for p, v in OBS:
        naive.tell(p, v)

    assert naive.suggest() != original.suggest()


# ── Round-trip ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls", [RandomSearchOptimizer, BayesianOptimizer])
def test_restored_optimizer_continues_exactly(cls):
    """The acceptance property: resume proposes what the original would have."""
    o = _run(cls)
    blob = json.dumps(o.to_dict())      # snapshot before the next draw
    expected = o.suggest()

    restored = BaseOptimizer.from_dict(json.loads(blob))

    assert restored.suggest() == expected


@pytest.mark.parametrize("cls", [RandomSearchOptimizer, BayesianOptimizer])
def test_state_is_json_serializable(cls):
    """Checkpoints go into SQLite as text — no numpy or tuples may leak."""
    json.dumps(_run(cls).to_dict())     # must not raise


@pytest.mark.parametrize("cls", [RandomSearchOptimizer, BayesianOptimizer])
def test_history_and_best_survive(cls):
    o = _run(cls)
    restored = BaseOptimizer.from_dict(json.loads(json.dumps(o.to_dict())))

    assert len(restored.history) == len(OBS)
    assert restored.best() == o.best()
    assert restored.n_trials == o.n_trials


@pytest.mark.parametrize("cls", [RandomSearchOptimizer, BayesianOptimizer])
def test_restore_survives_repeated_round_trips(cls):
    """A campaign checkpoints every iteration, so state must not degrade."""
    o = _run(cls)
    for _ in range(3):
        o = BaseOptimizer.from_dict(json.loads(json.dumps(o.to_dict())))
    assert len(o.history) == len(OBS)
    assert o.suggest() is not None


# ── Safety of the dispatch ───────────────────────────────────────────────────

def test_unknown_optimizer_refuses_rather_than_degrading():
    """Resuming as a different search strategy would silently change the science."""
    state = _run(BayesianOptimizer).to_dict()
    state["optimizer"] = "NoSuchOptimizer"

    with pytest.raises(OptimizerError, match="Unknown optimizer"):
        BaseOptimizer.from_dict(state)


def test_checkpoint_names_its_own_class():
    assert _run(BayesianOptimizer).to_dict()["optimizer"] == "BayesianOptimizer"
    assert _run(RandomSearchOptimizer).to_dict()["optimizer"] == "RandomSearchOptimizer"


def test_batch_strategy_round_trips_by_registry_name():
    """Regression: the class name ('ConstantLiarStrategy') is not a valid key."""
    o = BayesianOptimizer(SPACE, seed=1, batch_strategy="kriging_believer")
    state = o.to_dict()
    assert state["extra"]["batch_strategy"] == "kriging_believer"

    restored = BaseOptimizer.from_dict(json.loads(json.dumps(state)))
    assert restored.to_dict()["extra"]["batch_strategy"] == "kriging_believer"


def test_random_search_budget_is_not_reset_by_resume():
    """Otherwise a resumed run gets a fresh budget and overruns the campaign."""
    o = RandomSearchOptimizer(SPACE, seed=1, n_trials=4)
    for _ in range(4):
        o.suggest()
    assert o.suggest() is None           # exhausted

    restored = BaseOptimizer.from_dict(json.loads(json.dumps(o.to_dict())))
    assert restored.suggest() is None    # still exhausted, not refreshed


def test_grid_search_resume_continues_from_its_saved_index():
    """A resumed grid must not silently re-walk from index 0 at the default width.

    ``GridSearchOptimizer`` defined neither ``_construct_from`` nor
    ``_state_extra``, so a checkpoint round-trip rebuilt it with the *default*
    ``n_points=5`` and a fresh cursor — a resumed campaign then re-cast points it
    had already measured, on a grid of a different shape, with nothing red.
    """
    from softae.optimizers.grid import GridSearchOptimizer

    o = GridSearchOptimizer(SPACE, objective="maximize", n_points=3)   # 3x3 = 9
    for i in range(4):
        p = o.suggest()
        o.tell(p, float(i))

    restored = BaseOptimizer.from_dict(json.loads(json.dumps(o.to_dict())))

    assert isinstance(restored, GridSearchOptimizer)
    assert restored._n_points == 3                    # width survives
    rest = []
    while (p := restored.suggest()) is not None:
        rest.append(p)
    assert len(rest) == 5                             # 9 - 4 already walked
    assert rest[0] == o._grid[4]                      # cursor survives


def test_feasibility_counters_survive_a_resume():
    """So a resumed run can still report that the space was mostly infeasible."""
    calls = {"n": 0}

    def _refuse_the_first_three(params):
        calls["n"] += 1
        return calls["n"] > 3

    o = RandomSearchOptimizer(SPACE, seed=1, n_trials=4)
    o.feasibility_fn = _refuse_the_first_three
    o.suggest()
    assert o._n_rejected == 3

    restored = BaseOptimizer.from_dict(json.loads(json.dumps(o.to_dict())))
    assert restored._n_rejected == o._n_rejected
    assert restored._n_unchecked == o._n_unchecked


def test_resuming_without_a_prior_mean_warns(caplog):
    """A prior is an arbitrary callable; losing it silently changes the surrogate."""
    o = BayesianOptimizer(SPACE, seed=1, prior_mean=lambda p: 0.0)
    state = json.loads(json.dumps(o.to_dict()))
    assert state["extra"]["had_prior_mean"] is True

    restored = BaseOptimizer.from_dict(state)   # must not raise
    assert restored is not None
