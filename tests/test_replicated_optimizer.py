"""ReplicatingOptimizer: one inner proposal, k consecutive wells.

The wrapper is a view onto the inner optimizer, so every test here asserts that
some piece of state the campaign relies on reaches the inner rather than
accumulating a second copy out on the wrapper.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from structlog.testing import capture_logs

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer, Feasibility
from softae.optimizers.random import RandomSearchOptimizer
from softae.optimizers.replicated import ReplicatingOptimizer

SPACE = {
    "eo_li": {"type": "float", "low": 5.0, "high": 50.0},
    "silica": {"type": "float", "low": 0.0, "high": 0.15},
}


def _inner(n_trials: int = 4, seed: int = 42) -> RandomSearchOptimizer:
    return RandomSearchOptimizer(
        SPACE, objective="maximize", seed=seed, n_trials=n_trials
    )


def _distinct(batch) -> int:
    return len({tuple(sorted(p.items())) for p in batch})


def test_replicating_optimizer_suggest_batch_repeats_each_proposal_k_times():
    """Each proposal is emitted k times consecutively, so a pair lands on
    adjacent channels."""
    batch = ReplicatingOptimizer(_inner(), 2).suggest_batch(4)

    assert len(batch) == 4
    assert batch[0] == batch[1] and batch[2] == batch[3]
    assert batch[0] != batch[2]


def test_replicating_optimizer_inner_budget_counts_suggestions_not_wells():
    """budget counts distinct compositions: n_trials=4 with k=2 fills 8 wells."""
    batch = ReplicatingOptimizer(_inner(n_trials=4), 2).suggest_batch(8)

    assert len(batch) == 8
    assert _distinct(batch) == 4


def test_replicating_optimizer_tell_forwards_both_replicates_to_inner_history():
    """Both wells of a pair are told at the same x, and nothing de-duplicates."""
    inner = _inner()
    opt = ReplicatingOptimizer(inner, 2)
    pair = opt.suggest_batch(2)
    opt.tell(pair[0], 1.0)
    opt.tell(pair[1], 3.0)

    assert len(inner.history) == 2
    assert inner.history[0][0] == inner.history[1][0]
    assert opt.history == inner.history
    assert opt.n_trials == 2
    assert opt.best() == inner.best() == (pair[1], 3.0)
    # Wrapping an optimizer must not reset the one being wrapped.
    assert ReplicatingOptimizer(inner, 2).history == inner.history


def test_replicating_optimizer_partial_round_warns_and_truncates():
    """q=7 with k=2 yields 7 wells and warns; a whole round warns not at all."""
    with capture_logs() as logs:
        batch = ReplicatingOptimizer(_inner(n_trials=4), 2).suggest_batch(7)

    assert len(batch) == 7
    assert _distinct(batch) == 4
    assert batch[4] == batch[5] and batch[6] != batch[5]
    assert [log["event"] for log in logs] == ["replicate_round_not_whole"]

    with capture_logs() as whole_logs:
        ReplicatingOptimizer(_inner(n_trials=4), 2).suggest_batch(8)

    assert "replicate_round_not_whole" not in [log["event"] for log in whole_logs]


def test_replicating_optimizer_roundtrips_inner_state_through_to_dict():
    """A resumed wrapper proposes what its un-serialized twin would have."""
    opt = ReplicatingOptimizer(_inner(n_trials=6), 2)
    opt.tell(opt.suggest_batch(2)[0], 0.7)
    blob = json.dumps(opt.to_dict())
    expected = opt.suggest_batch(4)

    restored = BaseOptimizer.from_dict(json.loads(blob))

    assert isinstance(restored, ReplicatingOptimizer)
    assert restored.replicates == 2
    assert isinstance(restored.inner, RandomSearchOptimizer)
    assert restored.history == opt.history
    assert restored.suggest_batch(4) == expected


def test_replicating_optimizer_interleaved_layout_separates_each_pair():
    """Interleaved emits the whole proposal block k times, so a pair is q/k apart."""
    batch = ReplicatingOptimizer(
        _inner(), 2, layout="interleaved").suggest_batch(8)

    assert len(batch) == 8
    assert _distinct(batch) == 4
    assert all(batch[i] == batch[i + 4] for i in range(4))
    assert all(batch[i] != batch[i + 1] for i in range(3))


def test_replicating_optimizer_interleaved_partial_round_truncates_to_q():
    """A partial round still yields q points and says the round is not whole."""
    with capture_logs() as logs:
        batch = ReplicatingOptimizer(
            _inner(n_trials=4), 2, layout="interleaved").suggest_batch(7)

    assert len(batch) == 7
    assert _distinct(batch) == 4
    assert [log["event"] for log in logs] == ["replicate_round_not_whole"]


def test_replicating_optimizer_unknown_layout_is_refused():
    """An unrecognised layout names the two that exist rather than guessing."""
    with pytest.raises(OptimizerError) as exc:
        ReplicatingOptimizer(_inner(), 2, layout="spiral")

    assert "adjacent" in str(exc.value) and "interleaved" in str(exc.value)
    assert ReplicatingOptimizer.LAYOUTS == ("adjacent", "interleaved")


def test_replicating_optimizer_layout_roundtrips_through_to_dict():
    """Layout is search state: a resume must not silently re-adjacent the pairs."""
    opt = ReplicatingOptimizer(_inner(n_trials=6), 2, layout="interleaved")
    opt.tell(opt.suggest_batch(2)[0], 0.4)
    state = json.loads(json.dumps(opt.to_dict()))
    expected = opt.suggest_batch(4)

    restored = BaseOptimizer.from_dict(state)

    assert state["extra"]["layout"] == "interleaved"
    assert restored.layout == "interleaved"
    assert restored.suggest_batch(4) == expected

    state["extra"].pop("layout")          # a checkpoint predating the option
    assert BaseOptimizer.from_dict(state).layout == "adjacent"


def test_replicating_optimizer_from_dict_resolves_in_a_fresh_interpreter():
    """A checkpoint resumes in a process that imported only the base module —
    which is all ``campaign_resume`` imports."""
    opt = ReplicatingOptimizer(_inner(), 2)
    opt.tell(opt.suggest_batch(2)[0], 0.5)

    proc = subprocess.run(
        [sys.executable, "-c",
         "import json, sys\n"
         "from softae.optimizers.base import BaseOptimizer\n"
         "o = BaseOptimizer.from_dict(json.loads(sys.stdin.read()))\n"
         "print(type(o).__name__, o.replicates, o.n_trials)"],
        input=json.dumps(opt.to_dict()), capture_output=True, text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ReplicatingOptimizer", "2", "1"]


def test_replicating_optimizer_feasibility_fn_reaches_the_inner_optimizer():
    """Positive control: a filter set on the OUTER that refuses everything must
    empty the batch, or the property forwarding is not under test."""
    inner = _inner()
    inner._feasibility_max_tries = 5     # the loop that redraws lives in the inner
    opt = ReplicatingOptimizer(inner, 2)
    opt.check_feasible_fn = lambda params: Feasibility.INFEASIBLE

    assert opt.check_feasible_fn is inner.check_feasible_fn
    assert opt.suggest_batch(4) == []
    assert opt.n_rejected == inner.n_rejected > 0
    # Re-wrapping keeps the hook the campaign installed.
    assert ReplicatingOptimizer(inner, 2).check_feasible_fn is not None

    legacy_inner = _inner()
    legacy_inner._feasibility_max_tries = 5
    legacy = ReplicatingOptimizer(legacy_inner, 2)
    legacy.feasibility_fn = lambda params: False

    assert legacy.feasibility_fn is legacy_inner.feasibility_fn
    assert legacy.suggest_batch(4) == []


def test_replicating_optimizer_replicates_below_one_is_refused():
    """k >= 1; k == 1 is legal, though the wiring will not wrap for it."""
    for bad in (0, -1):
        with pytest.raises(OptimizerError):
            ReplicatingOptimizer(_inner(), bad)

    assert len(ReplicatingOptimizer(_inner(), 1).suggest_batch(3)) == 3


def test_replicating_optimizer_suggest_single_delegates_unchanged():
    """The single-point path is the inner's: one draw per call, not k."""
    opt = ReplicatingOptimizer(_inner(), 3)
    twin = _inner()

    assert opt.suggest() == twin.suggest()
    assert opt.suggest() == twin.suggest()     # still in step: one draw was spent
    assert opt.n_trials == 0                   # suggest records nothing
