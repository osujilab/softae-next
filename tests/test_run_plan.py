"""Tests for the run-plan phase abstraction (softae.core.run_plan)."""

from __future__ import annotations

import pytest

from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
)


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
