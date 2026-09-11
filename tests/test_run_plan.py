"""Tests for the run-plan phase abstraction (softae.core.run_plan)."""

from __future__ import annotations

import pytest

from softae.core.measurement_spec import MeasurementSpec
from softae.core.phase_setpoints import PhaseSetpoints
from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)


def _settle() -> SettlePlan:
    return SettlePlan(round_period_s=240.0, min_hold_s=1500.0, max_hold_s=14400.0)


# ── factories ────────────────────────────────────────────────────────────────

def test_pointwise_default_is_formulate_then_measure_all_per_sample():
    plan = RunPlan.pointwise()
    kinds = [p.kind for p in plan.phases]
    assert kinds == [PhaseKind.FORMULATE, PhaseKind.MEASURE]
    assert all(p.scope is PhaseScope.PER_SAMPLE for p in plan.phases)
    assert plan.has_measure and not plan.has_anneal
    assert not plan.defers_measurement


def test_pointwise_can_omit_measure_and_insert_anneal():
    plan = RunPlan.pointwise(measure=False, anneal=True)
    kinds = [p.kind for p in plan.phases]
    assert kinds == [PhaseKind.FORMULATE, PhaseKind.ANNEAL]
    assert plan.has_anneal and not plan.has_measure


def test_batch_formulate_persample_anneal_and_measure_perbatch():
    plan = RunPlan.batch(anneal=True)
    scopes = {p.kind: p.scope for p in plan.phases}
    assert scopes[PhaseKind.FORMULATE] is PhaseScope.PER_SAMPLE
    assert scopes[PhaseKind.ANNEAL] is PhaseScope.PER_BATCH
    assert scopes[PhaseKind.MEASURE] is PhaseScope.PER_BATCH
    assert plan.defers_measurement


# ── validation ───────────────────────────────────────────────────────────────

def test_plan_requires_a_formulate_phase():
    with pytest.raises(ValueError, match="FORMULATE"):
        RunPlan((RunPhase(PhaseKind.MEASURE),))


def test_formulate_must_be_per_sample():
    with pytest.raises(ValueError, match="per-sample"):
        RunPlan((RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_BATCH),))


def test_arrhenius_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        RunPlan((
            RunPhase(PhaseKind.FORMULATE),
            RunPhase(PhaseKind.ARRHENIUS, PhaseScope.PER_BATCH),
        ))


# ── segmentation (how the engine groups phases) ──────────────────────────────

def test_pointwise_is_one_per_sample_segment():
    segs = RunPlan.pointwise(anneal=True).segments()
    assert len(segs) == 1
    scope, phases = segs[0]
    assert scope is PhaseScope.PER_SAMPLE
    assert [p.kind for p in phases] == [
        PhaseKind.FORMULATE, PhaseKind.ANNEAL, PhaseKind.MEASURE,
    ]


def test_batch_splits_into_per_sample_then_per_batch_segments():
    segs = RunPlan.batch(anneal=True).segments()
    assert [s[0] for s in segs] == [PhaseScope.PER_SAMPLE, PhaseScope.PER_BATCH]
    assert [p.kind for p in segs[0][1]] == [PhaseKind.FORMULATE]
    assert [p.kind for p in segs[1][1]] == [PhaseKind.ANNEAL, PhaseKind.MEASURE]


# ── describe / labels (GUI visibility) ───────────────────────────────────────

def test_describe_lists_phases_in_order_with_scope():
    text = RunPlan.batch(anneal=True).describe()
    assert "Formulate [per sample]" in text
    assert "Anneal" in text and "[per batch]" in text
    assert "Measure EIS [per batch]" in text
    # Ordered left-to-right.
    assert text.index("Formulate") < text.index("Anneal") < text.index("Measure")


def test_anneal_label_reflects_explicit_params():
    phase = RunPhase(
        PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
        anneal_params={"target_temp_C": 120, "hold_time_s": 600},
    )
    assert phase.label() == "Anneal (120°C/10min) [per batch]"


# ── conditions / measurement (the bench-instance contract) ───────────────────
#
# Both fields are trailing and defaulted, which is the whole reason they are
# trailing and defaulted: every construction that existed before them has to
# mean exactly what it meant.

def test_runphase_positional_construction_is_unchanged_by_the_new_fields():
    """The five positional arguments that existed keep their meaning and order.

    A field inserted anywhere but the end would rebind one of these silently —
    ``RunPhase(kind, scope, task, params, settle)`` would start passing ``settle``
    to something else — and nothing would fail at the call site.
    """
    settle = _settle()
    phase = RunPhase(
        PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH, "anneal_85C_8h",
        {"hold_time_s": 600}, settle,
    )
    assert phase.kind is PhaseKind.EQUILIBRATE
    assert phase.scope is PhaseScope.PER_BATCH
    assert phase.anneal_task == "anneal_85C_8h"
    assert phase.anneal_params == {"hold_time_s": 600}
    assert phase.settle is settle
    assert phase.conditions is None
    assert phase.measurement is None


def test_measurement_on_a_non_measure_phase_is_refused():
    """Only MEASURE acquires data, so a preset elsewhere could never take effect."""
    with pytest.raises(ValueError, match="measurement block"):
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 measurement=MeasurementSpec(preset="Extended"))


def test_measurement_on_a_measure_phase_is_accepted():
    phase = RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                     measurement=MeasurementSpec(preset="Extended"))
    assert phase.measurement.preset == "Extended"


@pytest.mark.parametrize("kind", [
    PhaseKind.FORMULATE, PhaseKind.ANNEAL, PhaseKind.EQUILIBRATE, PhaseKind.MEASURE,
])
def test_conditions_are_legal_on_every_phase_kind(kind):
    """A cast, a cure, a hold and a read each have their own environment."""
    phase = RunPhase(kind, PhaseScope.PER_BATCH,
                     conditions=PhaseSetpoints("casting", 25.0, 40.0))
    assert phase.conditions.name == "casting"


def test_phase_label_renders_the_commanded_conditions():
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"target_temp_C": 85, "hold_time_s": 28800},
                     conditions=PhaseSetpoints("anneal", 85.0, 20.0))
    assert phase.label() == "Anneal (85°C/480min) @ anneal (85 °C, 20 %RH) [per batch]"


def test_phase_label_renders_the_measurement_override():
    phase = RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                     measurement=MeasurementSpec(preset="Extended"))
    assert phase.label() == "Measure EIS (Extended) [per batch]"


def test_describe_renders_conditions_and_measurement_for_every_phase():
    plan = RunPlan.batch(
        anneal=True,
        conditions={
            PhaseKind.FORMULATE: PhaseSetpoints("casting", 25.0, 40.0),
            PhaseKind.ANNEAL: PhaseSetpoints("anneal", 85.0, 20.0),
        },
        measurement=MeasurementSpec(preset="Extended"),
    )
    text = plan.describe()
    assert "Formulate @ casting (25 °C, 40 %RH) [per sample]" in text
    assert "@ anneal (85 °C, 20 %RH) [per batch]" in text
    assert "Measure EIS (Extended) [per batch]" in text
    assert text.index("casting") < text.index("anneal") < text.index("Extended")


def test_factories_thread_conditions_onto_the_phase_of_that_kind():
    casting = PhaseSetpoints("casting", 25.0, 40.0)
    plan = RunPlan.pointwise(conditions={PhaseKind.FORMULATE: casting})
    by_kind = {p.kind: p for p in plan.phases}
    assert by_kind[PhaseKind.FORMULATE].conditions is casting
    assert by_kind[PhaseKind.MEASURE].conditions is None


def test_factories_refuse_a_measurement_override_with_no_measure_phase():
    """Otherwise the override is accepted and silently dropped on the floor."""
    with pytest.raises(ValueError, match="measure=False"):
        RunPlan.batch(measure=False, anneal=True,
                      measurement=MeasurementSpec(preset="Extended"))


def test_factory_defaults_leave_both_new_fields_unset():
    """No caller that did not ask for them gets them."""
    for plan in (RunPlan.pointwise(anneal=True), RunPlan.batch(anneal=True)):
        assert all(p.conditions is None for p in plan.phases)
        assert all(p.measurement is None for p in plan.phases)
