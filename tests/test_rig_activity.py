"""Claim registry, suspension, and workflow scope derivation.

The registry's existing behaviour (re-entrant claims, scoped conflicts) is
exercised incidentally by ``test_purge_runner.py``; what is pinned here is the
part that is easy to get wrong — **suspension**, the third state that keeps a
paused run distinguishable from an idle rig.

The trap this file exists to close: implementing pause as "release the claim,
re-acquire on resume". That passes any test that only asks whether manual
control is re-enabled, and it silently destroys the purge's ability to tell a
paused rig from a free one.
"""

from __future__ import annotations

from softae.core.rig_activity import (
    PURGE_INSTRUMENTS,
    RigActivity,
    workflow_instruments,
)
from softae.workflows.workflow_model import Workflow, WorkflowStep


def _step(name: str, instrument: str) -> WorkflowStep:
    return WorkflowStep(name=name, instrument=instrument, method="noop")


class TestSuspendVisibility:
    """A suspended owner is skipped by ``conflicts`` and kept by the other."""

    def test_conflicts_suspended_owner_returns_none(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage", "syringe"})
        assert activity.conflicts(PURGE_INSTRUMENTS) == "ht:cast"

        activity.suspend("ht:cast")
        assert activity.conflicts(PURGE_INSTRUMENTS) is None

    def test_suspended_conflict_suspended_owner_returns_owner(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage", "syringe"})
        activity.suspend("ht:cast")
        assert activity.suspended_conflict(PURGE_INSTRUMENTS) == "ht:cast"

    def test_suspended_conflict_active_owner_returns_none(self):
        """The two predicates partition the owners; neither double-counts."""
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage", "syringe"})
        assert activity.suspended_conflict(PURGE_INSTRUMENTS) is None

    def test_suspended_conflict_out_of_scope_owner_returns_none(self):
        """Suspension does not widen a scope — overlap is still required."""
        activity = RigActivity()
        activity.acquire("eis:sweep", {"espico"})
        activity.suspend("eis:sweep")
        assert activity.suspended_conflict(PURGE_INSTRUMENTS) is None

    def test_conflicts_second_active_owner_still_blocks(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage"})
        activity.acquire("anti-clog purge", {"syringe"})
        activity.suspend("ht:cast")
        assert activity.conflicts(PURGE_INSTRUMENTS) == "anti-clog purge"

    def test_unsuspend_owner_conflicts_again(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage", "syringe"})
        activity.suspend("ht:cast")
        activity.unsuspend("ht:cast")
        assert activity.conflicts(PURGE_INSTRUMENTS) == "ht:cast"
        assert activity.suspended_conflict(PURGE_INSTRUMENTS) is None

    def test_busy_suspended_owner_remains_true(self):
        """A paused run still *holds* the rig; it merely is not driving it."""
        activity = RigActivity()
        activity.acquire("ht:cast")
        activity.suspend("ht:cast")
        assert activity.busy is True


class TestSuspendDepth:
    """Depth survives the round trip — the reason it is not a release."""

    def test_suspend_unsuspend_claim_depth_preserved(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage"})
        activity.acquire("ht:cast", {"syringe"})

        activity.suspend("ht:cast")
        activity.unsuspend("ht:cast")

        assert activity.owners() == ("ht:cast",)
        activity.release("ht:cast")
        assert activity.conflicts({"stage"}) == "ht:cast"
        activity.release("ht:cast")
        assert activity.busy is False

    def test_suspend_round_trip_owner_entry_not_duplicated(self):
        activity = RigActivity()
        activity.acquire("ht:cast")
        for _ in range(3):
            activity.suspend("ht:cast")
            activity.unsuspend("ht:cast")
        assert activity.owners() == ("ht:cast",)

    def test_suspend_repeated_is_idempotent(self):
        """The executor's failure-ceiling pause nests inside an operator pause."""
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage"})
        activity.suspend("ht:cast", reason="operator pause")
        activity.suspend("ht:cast", reason="channel-failure hold")

        activity.unsuspend("ht:cast")
        # One unsuspend clears it: the owner goes back to conflicting, which is
        # the guarded direction, and the claim itself is untouched.
        assert activity.conflicts({"stage"}) == "ht:cast"
        assert activity.owners() == ("ht:cast",)

    def test_suspend_unknown_owner_is_noop(self):
        activity = RigActivity()
        activity.suspend("never-claimed")
        assert activity.suspended_conflict(PURGE_INSTRUMENTS) is None
        assert activity.busy is False

    def test_unsuspend_unknown_owner_is_noop(self):
        activity = RigActivity()
        activity.acquire("ht:cast")
        activity.unsuspend("someone-else")
        assert activity.conflicts({"stage"}) == "ht:cast"


class TestSuspensionLifetime:
    """Suspension dies with the claim, never outlives it."""

    def test_release_final_claim_drops_suspension(self):
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage", "syringe"})
        activity.suspend("ht:cast")
        activity.release("ht:cast")

        assert activity.suspended_conflict(PURGE_INSTRUMENTS) is None
        assert activity.conflicts(PURGE_INSTRUMENTS) is None
        assert activity.busy is False

    def test_release_nested_claim_keeps_suspension(self):
        """Only the *final* release drops it — an inner pop is still paused."""
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage"})
        activity.acquire("ht:cast", {"syringe"})
        activity.suspend("ht:cast")

        activity.release("ht:cast")
        assert activity.conflicts({"stage"}) is None
        assert activity.suspended_conflict({"stage"}) == "ht:cast"

    def test_acquire_after_suspended_release_is_not_suspended(self):
        """No phantom: the next run under the same owner starts driving."""
        activity = RigActivity()
        activity.acquire("ht:cast", {"stage"})
        activity.suspend("ht:cast")
        activity.release("ht:cast")

        activity.acquire("ht:cast", {"stage"})
        assert activity.conflicts({"stage"}) == "ht:cast"
        assert activity.suspended_conflict({"stage"}) is None

    def test_claimed_block_suspended_inside_leaves_nothing_behind(self):
        activity = RigActivity()
        with activity.claimed("ht:cast", {"stage"}):
            activity.suspend("ht:cast")
        assert activity.busy is False
        assert activity.suspended_conflict({"stage"}) is None

    def test_claimed_block_raising_while_suspended_leaves_nothing_behind(self):
        activity = RigActivity()
        try:
            with activity.claimed("ht:cast", {"stage"}):
                activity.suspend("ht:cast")
                raise RuntimeError("run failed while paused")
        except RuntimeError:
            pass
        assert activity.busy is False
        assert activity.suspended_conflict({"stage"}) is None


class TestDescribe:
    def test_describe_idle_registry_returns_idle(self):
        assert RigActivity().describe() == "idle"

    def test_describe_active_owner_is_unmarked(self):
        activity = RigActivity()
        activity.acquire("ht:cast_series")
        assert activity.describe() == "ht:cast_series"

    def test_describe_suspended_owner_is_marked_paused(self):
        activity = RigActivity()
        activity.acquire("ht:cast_series")
        activity.suspend("ht:cast_series")
        assert activity.describe() == "ht:cast_series (paused)"

    def test_describe_suspended_owner_with_reason_shows_reason(self):
        activity = RigActivity()
        activity.acquire("ht:cast_series")
        activity.suspend("ht:cast_series", reason="channel-failure hold")
        assert activity.describe() == "ht:cast_series (channel-failure hold)"

    def test_describe_mixed_owners_marks_only_the_suspended_one(self):
        activity = RigActivity()
        activity.acquire("anti-clog purge")
        activity.acquire("ht:cast_series")
        activity.suspend("ht:cast_series")
        assert activity.describe() == "anti-clog purge, ht:cast_series (paused)"


class TestWorkflowInstruments:
    """Scope derivation widens on doubt; it never narrows."""

    def test_workflow_instruments_all_phases_are_unioned(self):
        wf = Workflow(
            name="cast_anneal_eis",
            setup=[_step("flush", "syringe")],
            loop_steps=[_step("cast", "stage"), _step("measure", "espico")],
            teardown=[_step("cool", "temp_controller")],
            iterations=4,
        )
        assert workflow_instruments(wf) == frozenset(
            {"syringe", "stage", "espico", "temp_controller"}
        )

    def test_workflow_instruments_zero_iterations_still_covers_loop_steps(self):
        """Wider than ``resolve_steps()`` on purpose — widening is the safe bias."""
        wf = Workflow(name="none", loop_steps=[_step("cast", "stage")],
                      iterations=0)
        assert workflow_instruments(wf) == frozenset({"stage"})

    def test_workflow_instruments_empty_workflow_falls_back_to_whole_rig(self):
        assert workflow_instruments(Workflow(name="empty")) is None

    def test_workflow_instruments_blank_names_fall_back_to_whole_rig(self):
        wf = Workflow(name="blank", setup=[_step("noop", "   ")])
        assert workflow_instruments(wf) is None

    def test_workflow_instruments_blank_names_are_dropped_not_kept(self):
        wf = Workflow(name="mixed",
                      setup=[_step("noop", ""), _step("cast", "stage")])
        assert workflow_instruments(wf) == frozenset({"stage"})

    def test_workflow_instruments_flat_steps_attribute_is_read(self):
        class _Flat:
            steps = [_step("cast", "stage"), _step("dispense", "syringe")]

        assert workflow_instruments(_Flat()) == frozenset({"stage", "syringe"})

    def test_workflow_instruments_unreadable_workflow_falls_back_to_whole_rig(self):
        class _Hostile:
            @property
            def setup(self):
                raise RuntimeError("no scope for you")

        assert workflow_instruments(_Hostile()) is None

    def test_workflow_instruments_non_workflow_object_falls_back_to_whole_rig(self):
        assert workflow_instruments(object()) is None
        assert workflow_instruments(None) is None

    def test_workflow_instruments_result_is_accepted_as_a_claim_scope(self):
        """The derived scope is exactly what ``acquire`` takes."""
        wf = Workflow(name="cast", loop_steps=[_step("cast", "stage")])
        activity = RigActivity()
        activity.acquire("ht:cast", workflow_instruments(wf))
        assert activity.conflicts({"stage"}) == "ht:cast"
        assert activity.conflicts({"espico"}) is None
