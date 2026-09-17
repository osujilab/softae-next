"""Tests for the run-plan phase abstraction (softae.core.run_plan)."""

from __future__ import annotations

import pytest

from softae.analysis.equilibration import (
    SETTLE_CRITERION_BOTH,
    SETTLE_CRITERION_DEVIATION,
    SETTLE_CRITERION_RATE,
)
from softae.core.measurement_spec import MeasurementSpec
from softae.core.phase_setpoints import PhaseSetpoints
from softae.core.run_plan import (
    DEFAULT_ANNEAL_TASK,
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)
from softae.core.task_catalog import Task, TaskCatalog


def _settle() -> SettlePlan:
    return SettlePlan(round_period_s=240.0, min_hold_s=1500.0, max_hold_s=14400.0)


def _catalog(temp_C: float | None = 85.0) -> TaskCatalog:
    """A one-task catalog, built in memory rather than read from ``data/``.

    ``data/tasks.toml`` is gitignored, so a label test that read the real
    catalog would assert on a machine-local file and could not be green in a
    fresh checkout. ``temp_C=None`` is the task that states no cure temperature.
    """
    params: dict = {"hold_time_s": 28800, "ramp_rate": 5, "tolerance": 1.0}
    if temp_C is not None:
        params["target_temp_C"] = temp_C
    catalog = TaskCatalog()
    catalog.add(Task(name="anneal_85C_8h", instrument="temp_controller",
                     method="anneal", params=params, timeout_s=36900))
    return catalog


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
    assert phase.label() == f"Anneal ({DEFAULT_ANNEAL_TASK}: 120°C/10min) [per batch]"


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
    assert phase.hold_s is None


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
    assert phase.label() == (f"Anneal ({DEFAULT_ANNEAL_TASK}: 85°C/8h) → rests at 85 °C "
                             "@ anneal (85 °C, 20 %RH) [per batch]")


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


# ── hold_s: one authority for the cure's duration ────────────────────────────
#
# Operator ruling D1 = option (d) in `docs/SubAgent docs/anneal_phase_duration.md`:
# the catalog task stays the sole authority for the hardware command, and this
# is the one typed per-run override of its hold.

def test_anneal_hold_s_and_anneal_params_hold_time_refused():
    """Said twice, in the wording ``settle_plan()`` already uses."""
    with pytest.raises(ValueError, match="say it once"):
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 anneal_params={"hold_time_s": 600}, hold_s=3600.0)


def test_hold_s_on_non_anneal_phase_refused():
    """Only ANNEAL holds at temperature, so a hold elsewhere never reaches it."""
    with pytest.raises(ValueError, match="never reach the chamber"):
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH, hold_s=3600.0)


@pytest.mark.parametrize("hold", [0.0, -1.0])
def test_non_positive_hold_s_refused(hold):
    """``None`` is how "no hold stated" is spelled; zero is a different claim."""
    with pytest.raises(ValueError, match="must be positive"):
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, hold_s=hold)


def test_anneal_params_may_still_override_the_temperature_beside_hold_s():
    """Only the duration is said twice; the rest of the task is untouched."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"target_temp_C": 85}, hold_s=3600.0)
    assert phase.hold_s == 3600.0
    assert phase.anneal_params == {"target_temp_C": 85}


def test_anneal_label_names_cure_and_restore_temperatures():
    """Both numbers **and their roles** — the point of the ruling, made visible."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"target_temp_C": 85}, hold_s=28800.0,
                     conditions=PhaseSetpoints("cooldown", 25.0))

    label = phase.label()

    assert "85" in label and "25" in label
    assert label.startswith(f"Anneal ({DEFAULT_ANNEAL_TASK}: 85°C/8h) → rests at 25 °C")


def test_anneal_label_without_conditions_names_no_restore_temperature():
    """The silent half: no conditions, nothing to say about the resting state."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, hold_s=1800.0)

    assert phase.label() == f"Anneal ({DEFAULT_ANNEAL_TASK}: 30min) [per batch]"


# ── T11.21: the label keeps the anneal's own identity ────────────────────────
#
# The defect: any typed field displaced the task name entirely, so the most
# ordinary phase there is — a bare `hold_s` on a catalogued cure — rendered
# "Anneal (8h) → rests at 25 °C". The only temperature on that line is the
# RESTORE target, and an operator reading it has nothing telling them so.

def test_anneal_label_hold_only_keeps_the_task_name():
    """`hold_s` alone no longer displaces the task; both survive, in one line."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h", hold_s=28800.0,
                     conditions=PhaseSetpoints("cooldown", 25.0))

    label = phase.label()

    assert label == ("Anneal (anneal_85C_8h: 8h) → rests at 25 °C "
                     "@ cooldown (25 °C) [per batch]")
    # The point of the task name being there: the cure's temperature is
    # *findable*, and the only number shown is the one whose role is stated.
    assert "25 °C" in label and "rests at" in label


def test_anneal_label_with_catalog_renders_cure_and_rest_temperatures():
    """Both temperatures, each with its role — the cure from the task, the rest
    from ``conditions``. Neither is inferred from the other."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h", hold_s=28800.0,
                     conditions=PhaseSetpoints("cooldown", 25.0))

    label = phase.label(_catalog())

    assert label.startswith("Anneal (anneal_85C_8h: 85°C/8h) → rests at 25 °C")
    # 85.0 from the catalog and 85 from an override are one anneal, not two.
    assert "85.0°C" not in label


def test_anneal_label_with_unknown_task_invents_no_temperature():
    """A task the catalog does not hold degrades to its name — never a default.

    The honest failure and the dangerous one differ by one number: printing a
    cure temperature nothing supports would be read off the screen as the
    commanded hold.
    """
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_not_in_this_catalog", hold_s=1800.0)

    label = phase.label(_catalog())

    assert label == "Anneal (anneal_not_in_this_catalog: 30min) [per batch]"
    assert "°C" not in label


def test_anneal_label_with_silent_task_invents_no_temperature():
    """A catalogued task stating no ``target_temp_C`` is 'nobody said', not 0 °C."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h", hold_s=1800.0)

    assert phase.label(_catalog(temp_C=None)) == "Anneal (anneal_85C_8h: 30min) [per batch]"


def test_anneal_label_override_outranks_the_catalog_temperature():
    """The emitter's precedence, rendered: the run's override wins the hold, so
    it must win the label — a label naming 85 while the rig ramps to 120 is
    worse than one naming neither."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h",
                     anneal_params={"target_temp_C": 120}, hold_s=600.0)

    assert phase.label(_catalog()) == "Anneal (anneal_85C_8h: 120°C/10min) [per batch]"


def test_anneal_label_with_no_typed_fields_is_the_task_alone():
    """The pre-existing bare case is unchanged apart from gaining no parameters."""
    phase = RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h")

    assert phase.label() == "Anneal (anneal_85C_8h) [per batch]"


def test_describe_forwards_the_catalog_to_every_anneal_phase():
    """`describe()` is the only caller of `label()` in production, so the
    catalog has to reach the phase through it or the capability is unreachable."""
    plan = RunPlan.batch(anneal=True, anneal_task="anneal_85C_8h", hold_s=28800.0)

    assert "Anneal (anneal_85C_8h: 8h)" in plan.describe()
    assert "Anneal (anneal_85C_8h: 85°C/8h)" in plan.describe(_catalog())


def test_describe_without_a_catalog_opens_no_file(monkeypatch):
    """The label never loads the catalog itself: it feeds a resume digest, and a
    read of gitignored ``data/tasks.toml`` would make that digest machine-local."""
    def _refuse(*args, **kwargs):
        raise AssertionError("describe() opened a file")

    monkeypatch.setattr("builtins.open", _refuse)
    monkeypatch.setattr(TaskCatalog, "load_toml", _refuse)

    assert "Anneal" in RunPlan.batch(anneal=True, hold_s=3600.0).describe()


def test_factories_thread_hold_s_onto_the_anneal_phase():
    for plan in (RunPlan.pointwise(anneal=True, hold_s=3600.0),
                 RunPlan.batch(anneal=True, hold_s=3600.0)):
        by_kind = {p.kind: p for p in plan.phases}
        assert by_kind[PhaseKind.ANNEAL].hold_s == 3600.0
        assert by_kind[PhaseKind.MEASURE].hold_s is None


def test_factories_refuse_a_hold_with_no_anneal_phase():
    """Otherwise the cure time is accepted and silently dropped on the floor.

    The same shape as ``measurement=`` with ``measure=False`` above.
    """
    with pytest.raises(ValueError, match="anneal=False"):
        RunPlan.batch(anneal=False, hold_s=3600.0)


# ── the settle criterion and its band (T11.2) ───────────────────────────────

def _settle_with(**kwargs) -> SettlePlan:
    """``_settle()`` plus whichever criterion keywords are under test."""
    return SettlePlan(round_period_s=240.0, min_hold_s=1500.0,
                      max_hold_s=14400.0, **kwargs)


def test_settle_plan_rate_criterion_and_band_are_accepted():
    """The positive control: this raises ``TypeError`` until the fields exist."""
    plan = _settle_with(criterion=SETTLE_CRITERION_RATE, rate_tol_dec_per_h=0.05)

    assert plan.criterion == SETTLE_CRITERION_RATE
    assert plan.rate_tol_dec_per_h == 0.05


def test_settle_plan_defaults_are_the_deviation_criterion_with_no_band():
    """Every plan written before this change keeps every verdict it had."""
    plan = _settle()

    assert plan.criterion == SETTLE_CRITERION_DEVIATION
    assert plan.rate_tol_dec_per_h is None


def test_settle_plan_unknown_criterion_word_is_refused():
    with pytest.raises(ValueError, match="criterion"):
        _settle_with(criterion="slope", rate_tol_dec_per_h=0.05)


def test_settle_plan_rate_criterion_without_a_band_is_refused():
    """A rate gate with no band certifies nothing and burns to the ceiling."""
    with pytest.raises(ValueError, match="rate_tol_dec_per_h"):
        _settle_with(criterion=SETTLE_CRITERION_RATE)


def test_settle_plan_both_criterion_without_a_band_is_refused():
    """The silent one: deviation still routes, and the shadow never computes.

    ``SettleTracker._rate_verdict`` returns ``None`` — never a verdict — with no
    tolerance configured, which is indistinguishable from "the window is not
    full yet", so ``both`` would look exactly like a working shadow run.
    """
    with pytest.raises(ValueError, match="rate_tol_dec_per_h"):
        _settle_with(criterion=SETTLE_CRITERION_BOTH)


@pytest.mark.parametrize("band", [0.0, -0.05])
def test_settle_plan_non_positive_rate_band_is_refused(band):
    """``None`` is how "no band" is spelled; zero is a band nothing satisfies."""
    with pytest.raises(ValueError, match="rate_tol_dec_per_h"):
        _settle_with(criterion=SETTLE_CRITERION_RATE, rate_tol_dec_per_h=band)


def test_settle_plan_label_omits_the_default_criterion():
    assert _settle().label() == "≤4h, ≥25min, every 4min"


@pytest.mark.parametrize(
    "criterion", [SETTLE_CRITERION_RATE, SETTLE_CRITERION_BOTH])
def test_settle_plan_label_names_the_criterion_and_band(criterion):
    label = _settle_with(criterion=criterion, rate_tol_dec_per_h=0.05).label()

    assert label.startswith("≤4h, ≥25min, every 4min, ")
    assert label.endswith(f"{criterion} ≤0.05 dec/h")
