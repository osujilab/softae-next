"""Named condition sets — ``[condition_sets.<name>]`` and ``conditions = "casting"``.

A campaign declares named sets of chamber setpoints at the top level, and each
run-plan phase either names one or states an inline table. The two spellings
must mean the same thing, an undeclared name must be refused at load, and — the
property the whole design exists to keep — a phase that says nothing about
conditions must drive no axis at all.

Several of these tests compile a real ``Workflow`` through
``build_trial_workflow``, the builder the campaign loop itself runs. Stopping at
"the spec decoded" would leave the seam half-proved: setpoints on a phase are
not commanded setpoints until they are ``temp_controller.write_sp {"T_SP": …}``
and ``rh_controller.set_setpoint {"val": …}`` on the rig.
"""

from __future__ import annotations

import pytest

from softae.core.campaign_spec_io import (
    SpecLoadError,
    load_campaign_spec,
    spec_from_dict,
    spec_to_dict,
    spec_toml_completeness,
)
from softae.core.campaign_spec_run_plan import (
    decode_baseline_conditions,
    decode_condition_sets,
    encode_condition_sets,
)
from softae.core.phase_setpoints import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_TOLERANCE_C,
    PhaseSetpoints,
)
from tests.support import fixture_catalog
from tests.support.fixture_catalog import (
    EXAMPLES,
    example_path,
    use_default_chemistry,
)

#: Bound, not redefined: the fixture lives once in the helper module, and pytest
#: finds a fixture by name in whatever namespace it is bound to.
task_catalog = fixture_catalog.task_catalog

#: A campaign with everything the loader requires and nothing this file is about.
BASE_SPEC: dict = {
    "name": "condition_sets_probe",
    "channels": [1],
    "parameter_space": {"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
    "vol_params": ["vol_p0"],
}

CASTING = {"temp_setpoint_C": 25.0, "rh_setpoint_pct": 22.0,
           "rh_approach_timeout_s": 14400.0}


def _spec(**overrides):
    return spec_from_dict({**BASE_SPEC, **overrides}, source="<test>")


def _plan(*phases) -> dict:
    return {"phases": list(phases)}


def _formulate(**extra) -> dict:
    return {"kind": "formulate", "scope": "per_sample", **extra}


MEASURE = {"kind": "measure", "scope": "per_sample"}


@pytest.fixture
def default_chemistry(monkeypatch):
    """Chemistry seam pointed at the shipped catalogs for the whole test."""
    use_default_chemistry(monkeypatch)


def condition_steps(spec, catalog) -> list[tuple]:
    """Every step the compiled trial workflow tags ``phase = "conditions"``."""
    from softae.core.autonomous_wiring import build_trial_workflow

    workflow = build_trial_workflow(spec, {"vol_p0": 10.0}, catalog=catalog)
    return [
        (step.name, step.instrument, step.method, dict(step.params))
        for group in ("setup", "loop_steps", "teardown")
        for step in (getattr(workflow, group, ()) or ())
        if (step.tags or {}).get("phase") == "conditions"
    ]


# -- decode -------------------------------------------------------------------

def test_decode_condition_sets_takes_the_table_key_as_the_name():
    sets = decode_condition_sets({"casting": CASTING})
    assert sets["casting"].name == "casting"
    assert sets["casting"].rh_setpoint_pct == 22.0


def test_decode_condition_sets_admits_the_approach_timeouts_a_phase_waits_for():
    """A named set is a PHASE's conditions and a phase waits; the baseline in
    the second half refuses the same keys, because nothing waits for it.
    """
    sets = decode_condition_sets(
        {"anneal_rest": {"rh_setpoint_pct": 22.0,
                         "rh_approach_timeout_s": 14400.0,
                         "approach_timeout_s": 10800.0,
                         "rh_tolerance_pct": 3.0}})
    rest = sets["anneal_rest"]
    assert rest.rh_approach_timeout_s == 14400.0
    assert rest.approach_timeout_s == 10800.0
    assert rest.rh_tolerance_pct == 3.0

    with pytest.raises(ValueError, match="NOTHING WAITS HERE"):
        decode_baseline_conditions(
            {"name": "baseline", "rh_setpoint_pct": 22.0,
             "rh_approach_timeout_s": 14400.0})


def test_decode_condition_sets_leaves_an_unstated_axis_undriven():
    """An omitted axis inside a set is still *not driven* — sets do not inherit."""
    only_rh = decode_condition_sets({"dry": {"rh_setpoint_pct": 5.0}})["dry"]
    assert only_rh.temp_setpoint_C is None
    assert only_rh.drives_humidity and not only_rh.drives_temperature
    assert only_rh.tolerance_C == DEFAULT_TOLERANCE_C
    assert only_rh.approach_timeout_s == DEFAULT_APPROACH_TIMEOUT_S


def test_decode_condition_sets_takes_arbitrary_n_with_operator_chosen_names():
    """No vocabulary, no enum, no count: any number of operator-chosen names
    decode, and the names below are deliberately not the humidity ones.
    """
    names = [f"stage_{i}" for i in range(9)] + ["wet", "dry", "ambient-2"]
    sets = decode_condition_sets({n: {"rh_setpoint_pct": 10.0} for n in names})
    assert sorted(sets) == sorted(names)
    assert all(sets[n].name == n for n in names)


def test_decode_condition_sets_refuses_a_name_inside_a_set():
    with pytest.raises(ValueError, match="The table key IS the name"):
        decode_condition_sets({"casting": {"name": "something_else",
                                           "rh_setpoint_pct": 22.0}})


def test_decode_condition_sets_refuses_a_non_table_body():
    with pytest.raises(ValueError, match="must be a table of setpoints"):
        decode_condition_sets({"casting": 22.0})


def test_decode_condition_sets_refuses_a_non_table():
    with pytest.raises(ValueError, match=r"\[condition_sets\.casting\]"):
        decode_condition_sets(["casting"])


# -- resolution ---------------------------------------------------------------

def test_a_phase_referencing_a_set_by_name_carries_that_sets_setpoints():
    spec = _spec(condition_sets={"casting": CASTING},
                 run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    assert spec.run_plan.phases[0].conditions == spec.condition_sets["casting"]
    assert spec.run_plan.phases[0].conditions.rh_approach_timeout_s == 14400.0


def test_two_phases_may_reference_one_set():
    spec = _spec(
        condition_sets={"rest": {"rh_setpoint_pct": 22.0}},
        run_plan=_plan(
            _formulate(conditions="rest"),
            {"kind": "anneal", "scope": "per_batch", "conditions": "rest"},
            MEASURE))
    first, second = spec.run_plan.phases[0], spec.run_plan.phases[1]
    assert first.conditions == second.conditions == spec.condition_sets["rest"]


def test_an_inline_conditions_table_is_still_legal_and_means_the_same_thing():
    """Back-compatibility, stated as an equality rather than as "it still loads"."""
    referenced = _spec(condition_sets={"casting": CASTING},
                       run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    inline = _spec(run_plan=_plan(
        _formulate(conditions={"name": "casting", **CASTING}), MEASURE))
    assert (referenced.run_plan.phases[0].conditions
            == inline.run_plan.phases[0].conditions)


def test_a_declared_set_nothing_references_is_kept_not_refused():
    """An unreferenced set warns but is never refused: keeping a spare around
    while A/B-ing two chamber environments is ordinary use.
    """
    spec = _spec(
        condition_sets={"casting": CASTING, "spare": {"rh_setpoint_pct": 40.0}},
        run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    assert sorted(spec.condition_sets) == ["casting", "spare"]
    assert spec.run_plan.phases[0].conditions.rh_setpoint_pct == 22.0


def test_condition_sets_may_be_declared_with_no_run_plan_at_all():
    spec = _spec(condition_sets={"casting": CASTING})
    assert spec.run_plan is None
    assert spec.condition_sets["casting"].rh_setpoint_pct == 22.0


# -- refusals -----------------------------------------------------------------

def test_an_undeclared_name_is_refused_and_lists_what_is_declared():
    """A typo fails at load, and the message lists the declared names: one that
    named only the typo would not say what was probably meant.
    """
    with pytest.raises(SpecLoadError) as exc:
        _spec(condition_sets={"casting": CASTING, "anneal_rest": CASTING},
              run_plan=_plan(_formulate(conditions="castng"), MEASURE))
    message = str(exc.value)
    assert "'castng'" in message
    assert "'anneal_rest'" in message and "'casting'" in message


def test_a_reference_with_no_condition_sets_table_at_all_is_refused():
    with pytest.raises(SpecLoadError, match="no \\[condition_sets.casting\\] declares"):
        _spec(run_plan=_plan(_formulate(conditions="casting"), MEASURE))


def test_conditions_that_is_neither_a_name_nor_a_table_is_refused():
    """Neither a name nor a table is refused, naming both legal forms; both at
    once is unreachable, since TOML and a dict each hold one such key.
    """
    for bad in (22.0, ["casting"], True, None):
        with pytest.raises(SpecLoadError) as exc:
            _spec(condition_sets={"casting": CASTING},
                  run_plan=_plan(_formulate(conditions=bad), MEASURE))
        assert "inline" in str(exc.value) and "NAME" in str(exc.value)


def test_both_forms_on_one_phase_cannot_be_written_in_toml_at_all():
    """Documented, not asserted against our code: ``tomllib`` refuses the file
    first, which the next reader would otherwise have to rediscover.
    """
    import tomllib

    with pytest.raises(tomllib.TOMLDecodeError):
        tomllib.loads(
            '[[run_plan.phases]]\nkind="formulate"\nconditions="casting"\n'
            '[run_plan.phases.conditions]\nname="x"\n')


def test_a_malformed_set_is_refused_before_any_phase_resolves():
    with pytest.raises(SpecLoadError, match="unknown key"):
        _spec(condition_sets={"casting": {"rh_setpoint_pctt": 22.0}},
              run_plan=_plan(_formulate(conditions="casting"), MEASURE))


# -- round trip ---------------------------------------------------------------

def test_condition_sets_round_trip_through_a_written_file():
    spec = _spec(condition_sets={"casting": CASTING, "spare": {"rh_setpoint_pct": 40.0}},
                 run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    written = spec_to_dict(spec)
    assert written["condition_sets"] == {"casting": CASTING,
                                         "spare": {"rh_setpoint_pct": 40.0}}
    assert spec_to_dict(spec_from_dict(written)) == written


def test_spec_toml_completeness_does_not_report_condition_sets_as_unwritable():
    spec = _spec(condition_sets={"casting": CASTING},
                 run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    completeness = spec_toml_completeness(spec)
    assert completeness.complete, completeness.reasons
    assert "condition_sets" not in completeness.missing


def test_a_set_whose_key_and_name_disagree_is_unrepresentable_not_written_wrong():
    """Only reachable from Python. The encoder answers ``UNREPRESENTABLE``
    rather than picking a spelling that would reload as a different campaign.
    """
    from softae.core.campaign_spec_fields import UNREPRESENTABLE

    assert encode_condition_sets(
        {"casting": PhaseSetpoints(name="not_casting", rh_setpoint_pct=22.0)}
    ) is UNREPRESENTABLE
    assert encode_condition_sets(
        {"casting": PhaseSetpoints(name="casting", rh_setpoint_pct=22.0)}
    ) == {"casting": {"rh_setpoint_pct": 22.0}}


def test_a_spec_with_no_condition_sets_writes_no_condition_sets_key():
    assert "condition_sets" not in spec_to_dict(_spec())


# -- end to end: the compiled workflow ----------------------------------------

def test_a_referenced_set_reaches_the_compiled_workflow_as_commanded_setpoints(
        task_catalog):
    """The seam end to end: a name becomes ``T_SP`` and ``val`` on the rig.
    Asserted on params — a count stays green if the setpoints arrive ``None``.
    """
    spec = _spec(condition_sets={"casting": CASTING},
                 run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    steps = {(instrument, method): params
             for _, instrument, method, params in condition_steps(spec, task_catalog)}

    assert steps[("temp_controller", "write_sp")]["T_SP"] == 25.0
    assert steps[("rh_controller", "set_setpoint")]["val"] == 22.0
    assert steps[("rh_controller", "wait")] == {
        "target": 22.0, "tol": 2.0, "timeout": 14400.0, "raise_on_timeout": True}
    assert steps[("temp_controller", "wait")]["timeout"] == 1800.0


def test_a_reference_and_an_inline_block_compile_to_identical_steps(task_catalog):
    referenced = _spec(condition_sets={"casting": CASTING},
                       run_plan=_plan(_formulate(conditions="casting"), MEASURE))
    inline = _spec(run_plan=_plan(
        _formulate(conditions={"name": "casting", **CASTING}), MEASURE))
    assert (condition_steps(referenced, task_catalog)
            == condition_steps(inline, task_catalog))


def test_a_phase_that_omits_conditions_emits_no_condition_steps(task_catalog):
    """Omission drives no axis, proved as steps at the chamber; a declared set
    sits unreferenced, so omission gaining a second meaning goes red here.
    """
    spec = _spec(condition_sets={"casting": CASTING},
                 run_plan=_plan(_formulate(), MEASURE))
    assert spec.run_plan.phases[0].conditions is None
    assert condition_steps(spec, task_catalog) == []


# -- back-compatibility against the committed examples ------------------------

@pytest.mark.parametrize("example", EXAMPLES)
def test_every_committed_example_still_loads_and_declares_no_condition_sets(
        example, default_chemistry):
    """Adopting named sets is per file and optional, and no example has yet;
    the second assertion is what stops an inventing loader passing.
    """
    spec = load_campaign_spec(example_path(example))
    assert spec.condition_sets == {}


def test_the_examples_that_drive_the_chamber_still_compile_condition_steps(
        task_catalog, default_chemistry):
    """A blanket "it still loads" would pass on a loader that dropped every
    block; ``shadow_campaign`` drives neither axis and is the negative control.
    """
    counts = {
        example: len(condition_steps(
            load_campaign_spec(example_path(example)), task_catalog))
        for example in ("bench_instance", "rung3a_fake_cast", "shadow_campaign")
    }
    assert counts["bench_instance"] > 0
    assert counts["rung3a_fake_cast"] > 0
    assert counts["shadow_campaign"] == 0
