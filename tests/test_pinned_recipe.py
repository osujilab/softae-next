"""The pinned-recipe route: one fully-specified composition, cast `budget` times.

Two halves, and they are opposites that share one mechanism.

*Pinning* — every composition axis at ``low == high`` — already runs with no
source change: ``build_targets_from_axes`` substitutes the constant, so the
suggestion cannot perturb the recipe, and an ``int`` ``replicate`` axis under
``grid`` supplies exactly ``budget`` wells to cast it into. The offline proof of
that is here, with no DataStore, no manager and no rig.

*The evil twin* — an axis declared **searched** whose name never reaches
``parameter_space`` — runs identically, at the axis's lower bound, on a warning
nobody reads. :func:`check_pinning` is the only thing that tells the two apart
for a file-loaded campaign.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.autonomous_wiring import CampaignSpec, GeneralFormulation
from softae.core.campaign_spec_io import load_campaign_spec
from softae.core.composition_axes import (
    CompositionAxis,
    axes_parameter_space,
    build_targets_from_axes,
)
from softae.core.pinned_recipe import (
    PinnedRecipeError,
    check_pinning,
    composition_axes,
    describe_pinning,
    is_pinned_recipe,
    pinned_axes,
    searched_axes,
)
from softae.optimizers.grid import GridSearchOptimizer

BENCH_INSTANCE = (Path(__file__).resolve().parents[1]
                  / "examples" / "bench_instance.toml")


def _spec(axes, space=None, **kw):
    """A composition spec over *axes*, searching *space* (default: the axes)."""
    from softae.core.formulation import ChemicalCatalog, Solution

    return CampaignSpec(
        name="pin",
        parameter_space=(axes_parameter_space(axes) if space is None else space),
        general_formulation=GeneralFormulation(
            stocks={"S": Solution(name="S")},
            catalog=ChemicalCatalog(),
            pump_assignment={"S": 0},
            target_deposition_uL=4.5,
            axes=axes,
        ),
        **kw,
    )


RATIO = CompositionAxis(kind="molar_ratio", a="EO", b="Li", low=20.0, high=20.0)
SILICA = CompositionAxis(kind="dried_fraction", a="SiO2", low=0.1, high=0.1)
SEARCHED_RATIO = CompositionAxis(kind="molar_ratio", a="EO", b="Li",
                                 low=5.0, high=40.0)


# ── The worked example ───────────────────────────────────────────────────────

class TestBenchInstance:
    """`examples/bench_instance.toml` is the file the bench run starts from.

    Loaded through the real loader and the real solution catalog on purpose: a
    stubbed catalog would let a stock name the rig cannot resolve sit in the
    example indefinitely, and the example's whole job is to be runnable.
    """

    @pytest.fixture(scope="class")
    def spec(self):
        return load_campaign_spec(BENCH_INSTANCE)

    def test_bench_instance_decodes_as_a_fully_pinned_recipe(self, spec):
        assert is_pinned_recipe(spec)
        assert searched_axes(spec) == ()
        assert len(pinned_axes(spec)) == len(composition_axes(spec))

    def test_bench_instance_describes_every_axis_as_pinned(self, spec):
        line = describe_pinning(spec)

        assert line.startswith("recipe PINNED")
        for axis in composition_axes(spec):
            assert axis.describe() in line
        assert "searched:" not in line     # the section header, not the count

    def test_bench_instance_searches_exactly_one_non_composition_parameter(
        self, spec
    ):
        """The replicate axis is what makes it run `budget` times at all."""
        axis_names = {a.name for a in composition_axes(spec)}

        assert set(spec.parameter_space) - axis_names == {"replicate"}
        assert spec.parameter_space["replicate"]["type"] == "int"
        assert spec.parameter_space["replicate"]["high"] == spec.budget

    def test_bench_instance_passes_the_pinning_check(self, spec):
        check_pinning(spec)      # raises on failure; nothing to assert

    def test_grid_over_the_replicate_axis_exhausts_at_budget(self, spec):
        """Offline: no DataStore, no manager, no rig — just the optimizer."""
        opt = GridSearchOptimizer(spec.parameter_space, "maximize",
                                  n_points=spec.budget)

        batch = opt.suggest_batch(spec.budget)

        assert len(batch) == spec.budget
        assert [p["replicate"] for p in batch] == list(range(1, spec.budget + 1))
        assert opt.suggest() is None

    def test_every_grid_suggestion_maps_to_identical_composition_targets(
        self, spec
    ):
        """The pinned recipe cannot be perturbed by which suggestion arrives."""
        build = build_targets_from_axes(composition_axes(spec))
        opt = GridSearchOptimizer(spec.parameter_space, "maximize",
                                  n_points=spec.budget)

        produced = [build(p) for p in opt.suggest_batch(spec.budget)]

        assert len(produced) == spec.budget
        assert all(t == produced[0] for t in produced)


# ── is_pinned_recipe / describe_pinning ──────────────────────────────────────

class TestDescription:

    def test_a_spec_with_one_searched_axis_is_not_a_pinned_recipe(self):
        assert not is_pinned_recipe(_spec((SEARCHED_RATIO, SILICA)))

    def test_a_volume_mode_spec_is_not_a_pinned_recipe(self):
        """No composition context = no recipe to pin.

        "Nothing to pin" must not be spelled with the same token as "pinned
        deliberately", or every legacy campaign reads as a declared replay.
        """
        legacy = CampaignSpec(
            name="vol",
            parameter_space={"vol_p0": {"type": "float", "low": 5.0,
                                        "high": 30.0}})

        assert not is_pinned_recipe(legacy)
        assert composition_axes(legacy) == ()

    def test_describe_pinning_reports_a_searched_axis_as_searched(self):
        line = describe_pinning(_spec((SEARCHED_RATIO, SILICA)))

        assert "SEARCHED" in line
        assert SEARCHED_RATIO.describe() in line
        assert SILICA.describe() in line

    def test_describe_pinning_states_the_absence_of_composition_targets(self):
        legacy = CampaignSpec(
            name="vol",
            parameter_space={"vol_p0": {"type": "float", "low": 5.0,
                                        "high": 30.0}})

        assert "no composition targets" in describe_pinning(legacy)


# ── check_pinning ────────────────────────────────────────────────────────────

class TestCheckPinning:

    def test_a_searched_axis_present_in_parameter_space_is_accepted(self):
        check_pinning(_spec((SEARCHED_RATIO, SILICA)))

    def test_a_fully_pinned_recipe_is_accepted(self):
        """The check must not refuse what it exists to permit.

        `validate_axes` would refuse exactly this — *"Every target is pinned …
        nothing to search"* — which is why it is not the tool used here.
        """
        check_pinning(_spec((RATIO, SILICA),
                            space={"replicate": {"type": "int", "low": 1,
                                                 "high": 4}}))

    def test_a_searched_axis_absent_from_parameter_space_is_refused(self):
        spec = _spec((SEARCHED_RATIO, SILICA),
                     space={"replicate": {"type": "int", "low": 1, "high": 4}})

        with pytest.raises(PinnedRecipeError) as exc:
            check_pinning(spec)

        assert SEARCHED_RATIO.name in str(exc.value)
        assert "lower bound" in str(exc.value)

    def test_the_refusal_names_every_missing_axis_not_only_the_first(self):
        other = CompositionAxis(kind="concentration", a="LiCl", low=0.1,
                                high=0.9)
        spec = _spec((SEARCHED_RATIO, other),
                     space={"replicate": {"type": "int", "low": 1, "high": 4}})

        with pytest.raises(PinnedRecipeError) as exc:
            check_pinning(spec)

        assert SEARCHED_RATIO.name in str(exc.value)
        assert other.name in str(exc.value)

    def test_a_spec_with_no_composition_axes_is_accepted(self):
        """No false positive on the legacy volume-mode shape every old spec has."""
        check_pinning(CampaignSpec(
            name="vol",
            parameter_space={"vol_p0": {"type": "float", "low": 5.0,
                                        "high": 30.0}}))


# ── What the check is NOT, stated so nobody assumes it ───────────────────────

def test_the_defect_the_check_prevents_is_silent_without_it():
    """Positive control for the *premise*, not for the check.

    Without the refusal there is no exception to catch and no NaN to notice:
    `build_targets_from_axes` substitutes the axis's `low` and the campaign
    casts a fixed recipe at the corner of the declared box. This asserts that
    substitution directly, so that if the fallback is ever changed to raise, the
    justification written into `check_pinning` fails loudly rather than reading
    as folklore.
    """
    build = build_targets_from_axes((SEARCHED_RATIO,))

    targets = build({"replicate": 3})       # the suggestion lacks the axis

    assert targets[0].value == SEARCHED_RATIO.low
