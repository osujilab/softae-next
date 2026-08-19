"""Resuming an interrupted campaign (P3.3).

Acceptance: kill a campaign mid-run, resume, and it reproduces the optimizer
state and continues on the correct board and wells — without re-casting a used
well and without abandoning free ones.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from softae.core.autonomous_wiring import (
    CampaignSpec,
    campaign_spec_fingerprint,
    serialize_campaign_spec,
)
from softae.core.campaign_resume import (
    ResumeMismatchError,
    ResumePlan,
    describe_resume,
    load_resume_plan,
)
from softae.core.data_store import DataStore
from softae.core.electrode_allocator import ElectrodeAllocator
from softae.core.measurement_spec import MeasurementSpec
from softae.optimizers.bayesian import BayesianOptimizer

SPACE = {"a": {"type": "float", "low": 0.0, "high": 10.0}}


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def _spec(**kw) -> CampaignSpec:
    base = dict(name="c", channels=(1, 2), parameter_space=dict(SPACE), budget=10)
    base.update(kw)
    return CampaignSpec(**base)


def _checkpoint(store, spec, *, iteration=3, board_id=1, optimizer=None):
    opt = optimizer or _optimizer(iteration)
    store.save_campaign_checkpoint(
        spec.name, iteration=iteration, run_id="r1", loop_state="EXECUTING",
        board_id=board_id, spec_json=serialize_campaign_spec(spec),
        optimizer_json=json.dumps(opt.to_dict()))
    return opt


def _optimizer(n=3) -> BayesianOptimizer:
    o = BayesianOptimizer(dict(SPACE), objective="maximize", seed=42)
    for i in range(n):
        o.suggest()
        o.tell({"a": float(i)}, 0.1 * (i + 1))
    return o


# ── Rebuilding the search ────────────────────────────────────────────────────

class TestLoad:
    def test_no_checkpoint_returns_none(self, store):
        assert load_resume_plan(store, _spec()) is None

    def test_optimizer_history_is_restored(self, store):
        spec = _spec()
        original = _checkpoint(store, spec, iteration=3)

        plan = load_resume_plan(store, spec)

        assert plan.optimizer.n_trials == original.n_trials
        assert plan.optimizer.best() == original.best()

    def test_resume_continues_the_same_search(self, store):
        """The acceptance property: the next point matches the original run's."""
        spec = _spec()
        original = _checkpoint(store, spec, iteration=3)
        expected = original.suggest()

        plan = load_resume_plan(store, spec)

        assert plan.optimizer.suggest() == expected

    def test_iteration_and_board_are_carried(self, store):
        spec = _spec()
        _checkpoint(store, spec, iteration=4, board_id=7)

        plan = load_resume_plan(store, spec)

        assert plan.iteration == 4
        assert plan.board_id == 7
        assert plan.remaining_budget == 10 - 4

    def test_exhausted_budget_is_reported_not_silently_run(self, store):
        spec = _spec(budget=3)
        _checkpoint(store, spec, iteration=3)

        plan = load_resume_plan(store, spec)

        assert plan.is_exhausted
        assert any("budget" in w for w in plan.warnings)


# ── Refusing to resume the wrong thing ───────────────────────────────────────

class TestMismatch:
    def test_changed_parameter_space_is_refused(self, store):
        """Grafting one search's observations onto another corrupts both."""
        _checkpoint(store, _spec())
        changed = _spec(parameter_space={"a": {"type": "float", "low": 0.0, "high": 99.0}})

        with pytest.raises(ResumeMismatchError, match="different search"):
            load_resume_plan(store, changed)

    def test_changed_objective_is_refused(self, store):
        _checkpoint(store, _spec())
        with pytest.raises(ResumeMismatchError):
            load_resume_plan(store, _spec(objective="minimize"))

    def test_non_strict_mode_reports_instead_of_raising(self, store):
        """For inspection only — never to force a resume."""
        _checkpoint(store, _spec())
        changed = _spec(parameter_space={"a": {"type": "float", "low": 0.0, "high": 99.0}})

        plan = load_resume_plan(store, changed, strict=False)

        assert plan is not None
        assert any("different search" in w for w in plan.warnings)

    def test_missing_optimizer_state_is_refused(self, store):
        """Resuming without it would silently restart the search from scratch."""
        spec = _spec()
        store.save_campaign_checkpoint(
            spec.name, iteration=2, spec_json=serialize_campaign_spec(spec),
            optimizer_json=None)

        with pytest.raises(ResumeMismatchError, match="no optimizer state"):
            load_resume_plan(store, spec)

    def test_corrupt_optimizer_state_is_refused(self, store):
        spec = _spec()
        store.save_campaign_checkpoint(
            spec.name, iteration=2, spec_json=serialize_campaign_spec(spec),
            optimizer_json='{"optimizer": "NotAnOptimizer"}')

        with pytest.raises(ResumeMismatchError):
            load_resume_plan(store, spec)

    def test_retuned_rates_still_resume(self, store):
        """Timings may legitimately change between sessions."""
        _checkpoint(store, _spec())
        assert load_resume_plan(store, _spec(disp_rate=999.0)) is not None

    def test_a_lost_prior_mean_warns_rather_than_failing(self, store):
        spec_with = _spec(prior_mean=lambda p: 0.0)
        _checkpoint(store, spec_with)

        plan = load_resume_plan(store, _spec())     # same search, no prior

        assert any("prior-mean" in w for w in plan.warnings)


# ── T2.4: the measurement block must not break an in-flight resume ───────────

class TestMeasurementBlockCompatibility:
    """The guard on T2.4: a campaign interrupted before it must resume after it.

    The rig runs multi-day campaigns. A checkpoint written by the pre-T2.4 code
    stores the fingerprint that code computed, and it is the only record of what
    was being searched — so if the block changed the hash, every unfinished run
    would refuse to continue and blame a parameter space nobody touched.
    """

    def _legacy(self, **kw) -> CampaignSpec:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return _spec(**kw)

    def test_a_checkpoint_written_in_the_legacy_spelling_resumes(self, store):
        legacy = self._legacy(eis_preset="Extended", eis_overrides={"npts": 33})
        original = _checkpoint(store, legacy, iteration=3)

        plan = load_resume_plan(store, legacy)

        assert plan.optimizer.n_trials == original.n_trials

    def test_a_legacy_checkpoint_resumes_onto_a_measurement_block_spec(self, store):
        """The migration case: the file is rewritten, the campaign continues."""
        legacy = self._legacy(eis_preset="Extended", eis_overrides={"npts": 33})
        _checkpoint(store, legacy, iteration=3)
        migrated = _spec(measurement=MeasurementSpec(
            preset="Extended", overrides={"npts": 33}))

        plan = load_resume_plan(store, migrated)     # must not raise

        assert plan.iteration == 3

    def test_a_checkpoint_predating_the_block_entirely_still_verifies(self, store):
        """Synthesised as the old code wrote it: no measurement key anywhere.

        Nothing on the resume path may *require* the new key, or checkpoints
        already on disk would be rejected for lacking a field they could not
        have had.
        """
        spec = _spec()
        opt = _optimizer(2)
        pre_t24 = {
            "fingerprint": campaign_spec_fingerprint(spec),
            "fields": {"name": "c", "eis_preset": "Quick"},
            "requires": {"prior_mean": False, "formulation": False},
        }
        store.save_campaign_checkpoint(
            spec.name, iteration=2, run_id="r1", board_id=1,
            spec_json=json.dumps(pre_t24, sort_keys=True),
            optimizer_json=json.dumps(opt.to_dict()))

        plan = load_resume_plan(store, spec)

        assert plan is not None and plan.iteration == 2

    def test_re_measuring_with_a_different_preset_still_resumes(self, store):
        """A preset is a setting, not the search — as it was before T2.4."""
        _checkpoint(store, _spec())

        assert load_resume_plan(
            store, _spec(measurement=MeasurementSpec(preset="Longest"))) is not None

    def test_changing_the_modality_refuses_the_resume(self, store):
        """Different instrument, different objective — not the same experiment."""
        _checkpoint(store, _spec())

        with pytest.raises(ResumeMismatchError, match="different search"):
            load_resume_plan(store, _spec(
                measurement=MeasurementSpec(modality="image")))


# ── Board / well reconciliation ──────────────────────────────────────────────

class TestElectrodeReconciliation:
    def test_occupied_wells_are_never_reallocated(self):
        alloc = ElectrodeAllocator(capacity=8, start=1, occupied=frozenset({1, 2, 3}))
        assert alloc.allocate(3).channels == [4, 5, 6]

    def test_free_wells_in_gaps_are_used_not_skipped(self):
        """A channel skipped by error recovery leaves a usable well behind.

        The old ``max(occupied) + 1`` rule abandoned wells 2 and 4 here — with 32
        single-use wells a board and hours of anneal already invested, that is
        expensive waste.
        """
        alloc = ElectrodeAllocator(capacity=6, start=1, occupied=frozenset({1, 3, 5}))
        assert alloc.allocate(3).channels == [2, 4, 6]

    def test_remaining_counts_gaps(self):
        alloc = ElectrodeAllocator(capacity=6, start=1, occupied=frozenset({1, 3, 5}))
        assert alloc.remaining == 3

    def test_a_fully_occupied_board_reports_full(self):
        alloc = ElectrodeAllocator(capacity=3, start=1, occupied=frozenset({1, 2, 3}))
        assert alloc.board_full
        assert alloc.allocate(1).overflow == 1

    def test_swapping_boards_clears_occupancy(self):
        """A fresh plate has no history; carrying it over would blank live wells."""
        alloc = ElectrodeAllocator(capacity=4, start=1, occupied=frozenset({1, 2}))
        alloc.swap_board()

        assert alloc.remaining == 4
        assert alloc.allocate(2).channels == [1, 2]

    def test_no_occupancy_behaves_exactly_as_before(self):
        """The default path must be unchanged by the gap-filling addition."""
        alloc = ElectrodeAllocator(capacity=4, start=2)
        assert alloc.allocate(3).channels == [2, 3, 4]
        assert alloc.allocate(1).overflow == 1


# ── Operator-facing summary ──────────────────────────────────────────────────

def test_describe_resume_states_progress_and_notes(store):
    spec = _spec()
    _checkpoint(store, spec, iteration=4, board_id=2)

    text = describe_resume(load_resume_plan(store, spec))

    assert "4 iteration" in text
    assert "board 2" in text
    assert "remain" in text


def test_resume_plan_exposes_exhaustion():
    plan = ResumePlan(campaign="c", optimizer=_optimizer(1), iteration=5,
                      board_id=1, run_id="r", remaining_budget=0)
    assert plan.is_exhausted


# ── How the previous run stopped: the resume-time discriminator ──────────────

class TestPreviousExitWasAcknowledged:
    """The rule that decides whether a resume clears the failure streak.

    The unit-level companion to the end-to-end arms in
    ``test_campaign_resume_e2e.py``. Those prove the counter behaves; these
    enumerate the vocabulary, which is where a status added later would
    otherwise slip through unclassified.
    """

    @staticmethod
    def _run(store: DataStore, status: str, *, finished: bool = True) -> str:
        run_id = store.start_run("campaign", mode="autonomous")
        if finished:
            store.finish_run(run_id, status)
        else:
            store._conn.execute(
                "UPDATE experiments SET status = ? WHERE run_id = ?",
                (status, run_id))
            store._conn.commit()
        return run_id

    @pytest.mark.parametrize(
        "status", ["stopped", "converged", "done", "partial", "aborted"])
    def test_a_status_written_by_an_unwinding_exit_is_acknowledged(
        self, tmp_path: Path, status: str
    ) -> None:
        """Each of these means the process lived long enough to say how it ended.

        ``stopped`` covers a park (which raises a CRITICAL alert) and an
        operator's own stop; ``aborted`` is an operator's deliberate abort.
        """
        from softae.core.autonomous_wiring import _previous_exit_was_acknowledged

        with DataStore(tmp_path / "proj") as store:
            ok, why = _previous_exit_was_acknowledged(
                store, self._run(store, status))
            assert ok is True
            assert status in why

    @pytest.mark.parametrize("status", ["error", "interrupted"])
    def test_a_crash_status_is_not_acknowledged(
        self, tmp_path: Path, status: str
    ) -> None:
        """``error`` is the campaign catch-all; ``interrupted`` is the recovery sweep."""
        from softae.core.autonomous_wiring import _previous_exit_was_acknowledged

        with DataStore(tmp_path / "proj") as store:
            ok, why = _previous_exit_was_acknowledged(
                store, self._run(store, status))
            assert ok is False
            assert status in why

    def test_an_unfinalized_row_is_not_acknowledged_whatever_its_status_says(
        self, tmp_path: Path
    ) -> None:
        """``finished_at`` beats the status string, and must.

        A hard kill leaves the row at its ``running`` default; the sweep that
        rewrites it to ``interrupted`` runs at the *next* launch and may not have
        run yet. Trusting the status alone would read a killed run as a live one.
        """
        from softae.core.autonomous_wiring import _previous_exit_was_acknowledged

        with DataStore(tmp_path / "proj") as store:
            # Even a status from the acknowledged set cannot rescue an open row.
            ok, why = _previous_exit_was_acknowledged(
                store, self._run(store, "stopped", finished=False))
            assert ok is False
            assert "never closed" in why

    def test_an_unknown_status_is_not_acknowledged(self, tmp_path: Path) -> None:
        """Unrecognised defaults to preserving the streak, not clearing it.

        A status added later should fail towards an early park rather than
        towards a chronic fault that can never escalate.
        """
        from softae.core.autonomous_wiring import _previous_exit_was_acknowledged

        with DataStore(tmp_path / "proj") as store:
            ok, _ = _previous_exit_was_acknowledged(
                store, self._run(store, "quiesced"))
            assert ok is False

    @pytest.mark.parametrize("run_id", [None, "", "no-such-run"])
    def test_an_unresolvable_run_id_is_not_acknowledged(
        self, tmp_path: Path, run_id
    ) -> None:
        """A checkpoint predating the column, or whose run row is gone."""
        from softae.core.autonomous_wiring import _previous_exit_was_acknowledged

        with DataStore(tmp_path / "proj") as store:
            ok, why = _previous_exit_was_acknowledged(store, run_id)
            assert ok is False
            assert why
