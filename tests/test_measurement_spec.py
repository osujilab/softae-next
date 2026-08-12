"""The measurement block and its legacy ``eis_*`` shim (T2.4).

Tier 2 makes the measurement side data-agnostic. This is the spec-level half:
one :class:`MeasurementSpec` naming a modality replaces the three EIS-shaped
:class:`CampaignSpec` fields, and the old fields become a transitional shim that
is canonicalized *into* the block rather than read alongside it.

Two properties carry the risk:

* **A spec has one authority.** Whichever spelling a caller used, exactly one
  block decides what gets measured — so no consumer can read a stale mirror.
* **Two spellings that disagree are refused.** Silently preferring one would run
  a different measurement from one of the two descriptions the caller wrote, and
  nothing in the data would show which lost.

Fingerprint stability is tested next door in ``test_campaign_checkpoint.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import pytest

from softae.core.autonomous_wiring import CampaignSpec
from softae.core.measurement_spec import (
    MeasurementSpec,
    canonicalize_measurement,
    measurement_identity,
)

SPACE = {"a": {"type": "float", "low": 0.0, "high": 10.0}}


def _spec(**kw) -> CampaignSpec:
    base = dict(name="c", channels=(1, 2), parameter_space=dict(SPACE), budget=10)
    base.update(kw)
    return CampaignSpec(**base)


def _legacy_spec(**kw) -> CampaignSpec:
    """Construct with the deprecated spelling without failing on the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _spec(**kw)


# ── The block itself ─────────────────────────────────────────────────────────

class TestMeasurementSpec:
    def test_defaults_describe_todays_behaviour(self):
        """The default block must be what every existing campaign already did."""
        m = MeasurementSpec()
        assert (m.modality, m.preset, m.overrides, m.enabled) == ("eis", "Quick", {}, True)

    def test_overrides_are_copied_so_a_frozen_block_stays_frozen(self):
        source = {"npts": 41}
        m = MeasurementSpec(overrides=source)
        source["npts"] = 999

        assert m.overrides == {"npts": 41}

    def test_an_empty_modality_is_refused(self):
        """A blank modality would resolve to no step builder at all."""
        with pytest.raises(ValueError, match="modality"):
            MeasurementSpec(modality="")

    def test_from_dict_refuses_an_unknown_key_rather_than_ignoring_it(self):
        """A typo'd key would otherwise silently take its default."""
        with pytest.raises(ValueError, match="unknown measurement key"):
            MeasurementSpec.from_dict({"presett": "Extended"})

    def test_as_dict_round_trips_through_from_dict(self):
        m = MeasurementSpec(modality="image", preset="wide", overrides={"n": 2},
                            enabled=False)
        assert MeasurementSpec.from_dict(m.as_dict()) == m


# ── Canonicalization: which spelling wins, and when neither may ──────────────

class TestCanonicalization:
    def test_no_spelling_supplied_gives_the_default_block(self):
        assert _spec().measurement == MeasurementSpec()

    def test_legacy_fields_populate_the_block(self):
        spec = _legacy_spec(eis_preset="Extended", eis_overrides={"npts": 33},
                            measure_eis=False)

        assert spec.measurement == MeasurementSpec(
            modality="eis", preset="Extended", overrides={"npts": 33}, enabled=False)

    def test_legacy_fields_warn_that_they_are_deprecated(self):
        with pytest.warns(DeprecationWarning, match="measurement block"):
            _spec(eis_preset="Extended")

    def test_the_new_spelling_alone_warns_about_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _spec(measurement=MeasurementSpec(preset="Extended"))

    def test_the_old_attributes_still_read_from_a_new_form_spec(self):
        """Consumers outside this task's reach still do `spec.eis_preset`."""
        spec = _spec(measurement=MeasurementSpec(
            preset="Extended", overrides={"npts": 33}, enabled=False))

        assert spec.eis_preset == "Extended"
        assert spec.eis_overrides == {"npts": 33}
        assert spec.measure_eis is False

    def test_agreeing_spellings_are_accepted(self):
        spec = _legacy_spec(eis_preset="Extended",
                            measurement=MeasurementSpec(preset="Extended"))
        assert spec.measurement.preset == "Extended"

    def test_disagreeing_spellings_are_refused_not_silently_resolved(self):
        """Precedence would run a different experiment from one of the two."""
        with pytest.raises(ValueError, match="disagree"):
            _spec(eis_preset="Quick", measurement=MeasurementSpec(preset="Extended"))

    def test_the_conflict_message_names_both_values(self):
        with pytest.raises(ValueError) as exc:
            _spec(measure_eis=True, measurement=MeasurementSpec(enabled=False))

        assert "measure_eis=True" in str(exc.value)
        assert "measurement.enabled=False" in str(exc.value)

    def test_a_non_block_measurement_is_refused(self):
        with pytest.raises(ValueError, match="must be a MeasurementSpec"):
            _spec(measurement="Quick")

    def test_a_mapping_is_accepted_as_a_block(self):
        """The TOML loader hands over a table; so may a GUI reading JSON."""
        assert canonicalize_measurement(
            measurement={"preset": "Longest"}) == MeasurementSpec(preset="Longest")


# ── Staying canonical under copying ──────────────────────────────────────────

class TestReplace:
    def test_replace_of_an_unrelated_field_does_not_trip_the_conflict_rule(self):
        """`build_optimizer` does exactly this to resolve the direction."""
        spec = _legacy_spec(eis_preset="Extended")

        assert replace(spec, objective="minimize").measurement.preset == "Extended"

    def test_replace_of_an_unrelated_field_does_not_re_warn(self):
        """Otherwise every internal copy would warn about the caller's spelling."""
        spec = _legacy_spec(eis_preset="Extended")

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            replace(spec, objective="minimize")

    def test_with_measurement_swaps_the_block_and_the_mirrors_follow(self):
        spec = _legacy_spec(eis_preset="Extended")

        swapped = spec.with_measurement(MeasurementSpec(preset="Longest"))

        assert swapped.measurement.preset == "Longest"
        assert swapped.eis_preset == "Longest"

    def test_replacing_the_block_directly_is_refused_with_a_way_forward(self):
        """It would silently keep the old preset, so it points at the helper."""
        spec = _legacy_spec(eis_preset="Extended")

        with pytest.raises(ValueError, match="with_measurement"):
            replace(spec, measurement=MeasurementSpec(preset="Longest"))


# ── Identity contribution (the fingerprint's input) ──────────────────────────

class TestIdentity:
    def test_the_default_modality_contributes_nothing(self):
        """Adding a key would rehash every campaign ever checkpointed."""
        assert measurement_identity(MeasurementSpec()) is None

    def test_settings_alone_never_become_identity(self):
        """Preset/overrides/enabled were never identity when spelled eis_*."""
        assert measurement_identity(MeasurementSpec(
            preset="Longest", overrides={"npts": 5}, enabled=False)) is None

    def test_a_different_modality_is_a_different_experiment(self):
        assert measurement_identity(MeasurementSpec(modality="image")) == {
            "modality": "image"}

    def test_an_absent_block_contributes_nothing(self):
        assert measurement_identity(None) is None


# ── Refusing what T2.5 has not built yet ─────────────────────────────────────

@pytest.mark.asyncio
async def test_a_campaign_naming_an_unbuilt_modality_refuses_to_start():
    """Better an explicit refusal than EIS steps recorded as an image run."""
    from softae.core.autonomous_wiring import run_autonomous_campaign

    spec = _spec(measurement=MeasurementSpec(modality="image"))

    with pytest.raises(NotImplementedError, match="image"):
        await run_autonomous_campaign(spec)
