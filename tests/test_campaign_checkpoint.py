"""Campaign resume checkpoints (P3.2).

The checkpoint is what makes an interrupted multi-day campaign recoverable. Two
properties matter more than the storage details:

* it is written **after** the observation reaches the optimizer, so a crash costs
  a data point rather than fabricating one for a well that was never cast; and
* it survives exactly the situations it exists for — a park, a crash — while
  being cleared when a campaign ends on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from softae.core.autonomous_wiring import (
    CampaignSpec,
    campaign_spec_fingerprint,
    serialize_campaign_spec,
)
from softae.core.data_store import DataStore
from softae.core.measurement_spec import MeasurementSpec


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def _spec(**kw) -> CampaignSpec:
    base = dict(
        name="c", channels=(1, 2),
        parameter_space={"a": {"type": "float", "low": 0.0, "high": 1.0}},
        budget=5,
    )
    base.update(kw)
    return CampaignSpec(**base)


class TestStore:
    def test_save_and_read_back(self, store):
        store.save_campaign_checkpoint(
            "c", iteration=3, run_id="r1", loop_state="EXECUTING", board_id=2,
            spec_json="{}", optimizer_json='{"optimizer": "X"}')

        cp = store.campaign_checkpoint("c")
        assert cp["iteration"] == 3
        assert cp["run_id"] == "r1"
        assert cp["board_id"] == 2
        assert cp["loop_state"] == "EXECUTING"

    def test_missing_campaign_reads_as_none(self, store):
        assert store.campaign_checkpoint("nope") is None

    def test_one_row_per_campaign_so_it_cannot_grow_unbounded(self, store):
        for i in range(1, 6):
            store.save_campaign_checkpoint("c", iteration=i)

        assert len(store.campaign_checkpoints()) == 1
        assert store.campaign_checkpoint("c")["iteration"] == 5

    def test_campaigns_are_independent(self, store):
        store.save_campaign_checkpoint("a", iteration=1)
        store.save_campaign_checkpoint("b", iteration=9)

        assert store.campaign_checkpoint("a")["iteration"] == 1
        assert store.campaign_checkpoint("b")["iteration"] == 9

    def test_survives_reopen(self, tmp_path: Path):
        """A checkpoint that died with the process would be useless."""
        ds = DataStore(tmp_path / "proj")
        ds.save_campaign_checkpoint("c", iteration=7, optimizer_json='{"k": 1}')
        ds.close()

        reopened = DataStore(tmp_path / "proj")
        try:
            cp = reopened.campaign_checkpoint("c")
            assert cp["iteration"] == 7
            assert json.loads(cp["optimizer_json"]) == {"k": 1}
        finally:
            reopened.close()

    def test_clear_removes_only_the_named_campaign(self, store):
        store.save_campaign_checkpoint("a", iteration=1)
        store.save_campaign_checkpoint("b", iteration=1)

        store.clear_campaign_checkpoint("a")

        assert store.campaign_checkpoint("a") is None
        assert store.campaign_checkpoint("b") is not None

    def test_clearing_an_absent_campaign_is_harmless(self, store):
        store.clear_campaign_checkpoint("never-existed")


class TestSpecSnapshot:
    def test_fingerprint_is_stable_for_an_identical_spec(self):
        assert campaign_spec_fingerprint(_spec()) == campaign_spec_fingerprint(_spec())

    def test_changing_the_search_space_changes_the_fingerprint(self):
        """Resuming onto a different space would graft one search onto another."""
        other = _spec(parameter_space={"a": {"type": "float", "low": 0.0, "high": 9.0}})
        assert campaign_spec_fingerprint(_spec()) != campaign_spec_fingerprint(other)

    def test_changing_the_objective_changes_the_fingerprint(self):
        assert campaign_spec_fingerprint(_spec()) != campaign_spec_fingerprint(
            _spec(objective="minimize"))

    def test_retuning_rates_does_not_change_identity(self):
        """Timings may legitimately be re-tuned between sessions."""
        assert campaign_spec_fingerprint(_spec()) == campaign_spec_fingerprint(
            _spec(disp_rate=999.0, settle_factor=7.0))

    def test_raising_the_budget_does_not_change_identity(self):
        """Extending the budget is *the* normal way to continue a stopped run.

        Treating it as a different experiment would make the resume path
        contradict its own advice to raise the budget when exhausted.
        """
        assert campaign_spec_fingerprint(_spec(budget=5)) == campaign_spec_fingerprint(
            _spec(budget=50))

    def test_snapshot_is_json_and_carries_the_fingerprint(self):
        payload = json.loads(serialize_campaign_spec(_spec()))
        assert payload["fingerprint"] == campaign_spec_fingerprint(_spec())
        assert payload["fields"]["budget"] == 5

    def test_snapshot_flags_what_must_be_re_supplied(self):
        """Silently resuming without a prior or formulation would change the science."""
        plain = json.loads(serialize_campaign_spec(_spec()))
        assert plain["requires"]["prior_mean"] is False

        with_prior = json.loads(
            serialize_campaign_spec(_spec(prior_mean=lambda p: 0.0)))
        assert with_prior["requires"]["prior_mean"] is True

    def test_snapshot_handles_unserializable_fields_without_raising(self):
        """A live Python object in the spec must not break checkpointing."""
        class _Weird:
            pass

        serialize_campaign_spec(_spec(prior_mean=_Weird()))   # must not raise


class TestMeasurementBlockFingerprint:
    """T2.4: the measurement block must not move any existing fingerprint.

    An in-flight campaign's checkpoint stores the fingerprint the *old* code
    computed. If T2.4 changed the hash of the same spec, every unfinished
    multi-day run on the rig would become unresumable — with a message blaming a
    changed parameter space that nobody changed.
    """

    #: Recorded from `_pinned_spec()` against the code as it stood BEFORE T2.4
    #: (2026-08-07, run before the edit). Hardcoded on purpose: a test that
    #: recomputed the expected value from the current code could not detect the
    #: very drift it exists to catch.
    LEGACY_FINGERPRINT = "acf4c6e1058568ae"

    @staticmethod
    def _pinned_spec(**kw) -> CampaignSpec:
        return CampaignSpec(name="fp", parameter_space={
            "x": {"type": "float", "low": 0, "high": 1}}, **kw)

    def test_the_default_spec_hashes_to_its_pre_t2_4_value(self):
        assert campaign_spec_fingerprint(self._pinned_spec()) == self.LEGACY_FINGERPRINT

    def test_a_legacy_eis_spec_hashes_to_the_same_pre_t2_4_value(self):
        """The eis_* fields were never identity, and must not become identity."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = self._pinned_spec(eis_preset="Extended",
                                       eis_overrides={"npts": 33},
                                       measure_eis=False)

        assert campaign_spec_fingerprint(legacy) == self.LEGACY_FINGERPRINT

    def test_legacy_and_new_spellings_fingerprint_identically(self):
        """The T2.4 guard: a resume must not care which spelling was used."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = _spec(eis_preset="Extended", eis_overrides={"npts": 33})
        new = _spec(measurement=MeasurementSpec(preset="Extended",
                                                overrides={"npts": 33}))

        assert campaign_spec_fingerprint(legacy) == campaign_spec_fingerprint(new)

    def test_changing_the_modality_does_change_identity(self):
        """A different measurement is a different experiment, unlike a preset."""
        other = _spec(measurement=MeasurementSpec(modality="image"))
        assert campaign_spec_fingerprint(_spec()) != campaign_spec_fingerprint(other)

    def test_the_snapshot_spells_out_the_canonical_block(self):
        payload = json.loads(serialize_campaign_spec(_spec()))

        assert payload["fields"]["measurement"] == {
            "modality": "eis", "preset": "Quick", "overrides": {}, "enabled": True}

    def test_the_snapshot_still_carries_the_old_eis_preset_key(self):
        """A snapshot written after T2.4 must still read like one written before."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            payload = json.loads(serialize_campaign_spec(_spec(eis_preset="Longest")))

        assert payload["fields"]["eis_preset"] == "Longest"


class TestLoopIntegration:
    def test_every_iteration_advance_checkpoints(self):
        """All completion paths consume a well, so all must move the resume point."""
        from softae.core.autonomous_loop import AutonomousLoop

        seen: list[int] = []
        loop = AutonomousLoop.__new__(AutonomousLoop)   # bypass heavy __init__
        loop._iteration = 0
        loop.on_checkpoint = seen.append

        for _ in range(3):
            loop._advance_iteration()

        assert seen == [1, 2, 3]
        assert loop.iteration == 3

    def test_a_failing_checkpoint_never_breaks_the_run(self):
        """Losing resumability must not end an otherwise healthy campaign."""
        from softae.core.autonomous_loop import AutonomousLoop

        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0

        def boom(_i):
            raise RuntimeError("disk full")

        loop.on_checkpoint = boom
        loop._advance_iteration()          # must not raise
        assert loop.iteration == 1
