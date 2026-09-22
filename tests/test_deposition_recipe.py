"""Tests for the recipe-driven deposition engine."""

from __future__ import annotations

import pytest

from softae.config.loader import pico_for_channel
from softae.core.deposition_recipe import (
    ANNEAL_RAMP_ALLOWANCE_S,
    ANNEAL_TIMEOUT_MARGIN,
    BUILTIN_DEPOSITION_RECIPES,
    DepositionSlots,
    PiezoPlan,
    build_recipe_deposition_workflow,
    build_slotted_deposition_workflow,
    deposition_recipe_names,
    get_deposition_recipe,
)
from softae.workflows.workflow_model import WorkflowStep
from softae.core.task_catalog import Task, TaskCatalog

PCB = {"grid": [8, 4], "spacing_mm": [10, 10]}


def _deposit_task() -> Task:
    return Task(
        name="single_drop_simul", instrument="liquid_handler", method="single_drop_simul",
        params={"x": 0, "y": 0, "vols": [1, 1], "disp_rate": 75},
    )


# ── Recipe-driven deposition engine ──────────────────────────────────────────

def test_builds_one_deposit_per_channel_with_injection():
    wf = build_slotted_deposition_workflow(
        _deposit_task(),
        channels=[21, 22],
        formulation_by_channel={21: [3.0, 1.0], 22: [2.0, 2.0]},
        pcb=PCB, origin_xy=(43.5, 50.0),
    )
    names = [s.name for s in wf.setup]
    assert "deposit_ch21" in names and "deposit_ch22" in names

    dep21 = next(s for s in wf.setup if s.name == "deposit_ch21")
    # Electrode injected from geometry (ch21 = row5,col0 → x=43.5, y=0).
    assert dep21.params["x"] == pytest.approx(43.5)
    assert dep21.params["y"] == pytest.approx(0.0)
    # Per-channel volumes injected.
    assert dep21.params["vols"] == [3.0, 1.0]
    assert dep21.tags["channel"] == "21"


def test_per_channel_volumes_differ():
    wf = build_slotted_deposition_workflow(
        _deposit_task(), channels=[21, 22],
        formulation_by_channel={21: [3.0, 1.0], 22: [2.0, 2.0]},
        pcb=PCB, origin_xy=(43.5, 50.0),
    )
    v21 = next(s for s in wf.setup if s.name == "deposit_ch21").params["vols"]
    v22 = next(s for s in wf.setup if s.name == "deposit_ch22").params["vols"]
    assert v21 == [3.0, 1.0] and v22 == [2.0, 2.0]


def test_eis_routes_per_channel_and_can_be_disabled():
    wf = build_slotted_deposition_workflow(
        _deposit_task(), channels=[21], formulation_by_channel={21: [1, 1]},
        pcb=PCB, origin_xy=(43.5, 50.0), measure_eis=True,
    )
    m = next(s for s in wf.setup if s.name == "measure_eis_ch21")
    assert m.instrument == pico_for_channel(21)
    assert m.params["chan"] == 21

    wf2 = build_slotted_deposition_workflow(
        _deposit_task(), channels=[21], formulation_by_channel={21: [1, 1]},
        pcb=PCB, origin_xy=(43.5, 50.0), measure_eis=False,
    )
    assert not any(s.name.startswith("measure_eis") for s in wf2.setup)


def test_custom_slot_names_respected():
    slots = DepositionSlots(electrode_x="ex", electrode_y="ey", volumes="v")
    task = Task(name="d", instrument="lh", method="drop", params={"ex": 0, "ey": 0, "v": []})
    wf = build_slotted_deposition_workflow(
        task, channels=[21], formulation_by_channel={21: [5.0]},
        pcb=PCB, origin_xy=(43.5, 50.0), slots=slots, measure_eis=False,
    )
    dep = wf.setup[0]
    assert dep.params["ex"] == pytest.approx(43.5)
    assert dep.params["v"] == [5.0]


# ── Deposition recipes + unified engine ──────────────────────────────────────

def _engine_catalog() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(name="startup_flush_full", instrument="liquid_handler", method="startup_flush",
                 params={"flush_x": -50, "flush_y": 50, "wick_x": -50, "wick_y": -25,
                         "disp_rate": 1500, "disp_vol": 150, "ids": [0, 1, 2]}))
    cat.add(Task(name="precondition_flush", instrument="liquid_handler", method="precondition_flush",
                 params={"flush_x": -50, "flush_y": 50, "wick_x": -50, "wick_y": -25,
                         "ids": [0, 1], "rate_list": [75, 75], "vol_list": [21, 21],
                         "flush_factor": 3.0}))
    cat.add(Task(name="single_drop_simul", instrument="liquid_handler", method="single_drop_simul",
                 params={"x": 0, "y": 0, "wick_x": -50, "wick_y": -25, "ids": [0, 1, 2],
                         "disp_rate": 75, "vols": [21, 21, 21], "deadvols": [20, 20, 20]}))
    cat.add(Task(name="alt_drop", instrument="liquid_handler", method="single_drop_simul",
                 params={"x": 0, "y": 0, "ids": [0, 1, 2], "disp_rate": 75, "vols": [1, 1, 1]}))
    cat.add(Task(name="final_flush", instrument="syringe", method="single_pump",
                 params={"res_vol": 1000, "ID": 0, "rate": 200, "dispense_vol": 80}))
    cat.add(Task(name="piezo_channel_a_on", instrument="piezo", method="set_channel",
                 params={"channel": "A", "enabled": True}))
    cat.add(Task(name="piezo_channel_a_off", instrument="piezo", method="set_channel",
                 params={"channel": "A", "enabled": False}))
    cat.add(Task(name="piezo_standby", instrument="piezo", method="standby", params={}))
    cat.add(Task(name="piezo_liquid_event", instrument="piezo", method="apply_profile",
                 params={"frequency_hz": 525, "on_s": 2.0, "rest_s": 3.0}))
    return cat


def _eis(ch: int) -> WorkflowStep:
    return WorkflowStep(name=f"measure_eis_ch{ch}", instrument=pico_for_channel(ch),
                        method="sendscript_getdata", params={"chan": ch})


def _build(recipe_name, **over):
    kw = dict(
        catalog=_engine_catalog(), pump_ids=[0, 1, 2],
        dispense_rate=100.0, flush_rate=500.0, flush_factor=3.0,
        settle_factor=2.0, settle_base_s=0.0, start_flush_uL=[80, 80, 80],
        pcb=PCB, origin_xy=(43.5, 50.0),
    )
    kw.update(over)
    return build_recipe_deposition_workflow(
        get_deposition_recipe(recipe_name), [21],
        {21: [10.0, 30.0, 0.0]}, **kw)


def test_builtin_recipes_registered():
    assert set(deposition_recipe_names()) == {"single_drop", "two_phase"}
    assert "single_drop" in BUILTIN_DEPOSITION_RECIPES


def test_two_phase_method_deps_roll_up():
    deps = get_deposition_recipe("two_phase").method_deps()
    assert set(deps) == {
        "startup_flush_full", "precondition_flush", "single_drop_simul", "final_flush"}


def test_single_drop_recipe_flat_rate_no_precondition():
    wf = _build("single_drop")
    names = [s.name for s in wf.setup]
    assert names[0] == "startup_flush"
    assert "deposit_ch21" in names
    assert not any(n.startswith("precondition") for n in names)
    dep = next(s for s in wf.setup if s.name == "deposit_ch21")
    assert dep.params["disp_rate"] == pytest.approx(100.0)   # flat
    assert "disp_rates" not in dep.params
    assert dep.params["vols"] == [10.0, 30.0, 0.0]
    assert dep.params["deadvols"] == [0.0, 0.0, 0.0]
    assert dep.params["ids"] == [0, 1, 2]
    assert wf.teardown[0].name == "final_flush"
    assert wf.metadata["recipe"] == "single_drop"


def test_two_phase_recipe_precondition_then_split_deposit():
    wf = _build("two_phase")
    names = [s.name for s in wf.setup]
    assert names.index("precondition_ch21") < names.index("deposit_ch21")
    pre = next(s for s in wf.setup if s.name == "precondition_ch21")
    assert pre.params["rate_list"] == [pytest.approx(125.0), pytest.approx(375.0), pytest.approx(0.0)]
    assert pre.params["vol_list"] == [10.0, 30.0, 0.0]
    assert pre.params["flush_factor"] == 3.0
    dep = next(s for s in wf.setup if s.name == "deposit_ch21")
    assert dep.params["disp_rates"] == [pytest.approx(25.0), pytest.approx(75.0), pytest.approx(0.0)]
    assert dep.params["elution_wait_s"] == pytest.approx(48.0)   # 24 s × 2


def test_eis_interleaved_after_deposit():
    wf = _build("two_phase", eis_step_by_channel={21: _eis(21)})
    names = [s.name for s in wf.setup]
    assert names.index("deposit_ch21") + 1 == names.index("measure_eis_ch21")


def test_no_eis_when_omitted():
    wf = _build("single_drop")
    assert not any(s.name.startswith("measure_eis") for s in wf.setup)


def test_deposit_method_override_applies_to_deposit_phase_only():
    wf = _build("two_phase", deposit_method="alt_drop")
    dep = next(s for s in wf.setup if s.name == "deposit_ch21")
    assert dep.method == "single_drop_simul"  # alt_drop's driver method
    # Precondition phase is untouched by the deposit override.
    pre = next(s for s in wf.setup if s.name == "precondition_ch21")
    assert pre.method == "precondition_flush"


# ── Piezo wiring ─────────────────────────────────────────────────────────────

def test_no_piezo_steps_by_default():
    wf = _build("two_phase")
    assert not any("piezo" in s.name for s in wf.setup + wf.teardown)
    assert wf.metadata["piezo"] == "not_applied"


def test_piezo_wraps_deposit_and_returns_to_standby():
    piezo = PiezoPlan(enabled=True, event_task="piezo_liquid_event",
                      event_params={"frequency_hz": 700})
    wf = _build("two_phase", piezo=piezo)
    names = [s.name for s in wf.setup]
    # Event profile once in setup; on wraps the deposit; off after the deposit.
    assert "piezo_event" in names
    assert names.index("piezo_on_ch21") < names.index("deposit_ch21")
    assert names.index("deposit_ch21") < names.index("piezo_off_ch21")
    # Piezo-on comes after precondition (it wraps the *deposit*, not the flush).
    assert names.index("precondition_ch21") < names.index("piezo_on_ch21")
    # Standby last in teardown.
    assert wf.teardown[-1].name == "piezo_standby"
    assert wf.metadata["piezo"] == "applied"
    # Event params overridden.
    ev = next(s for s in wf.setup if s.name == "piezo_event")
    assert ev.params["frequency_hz"] == 700


def test_piezo_off_follows_eis_when_present():
    piezo = PiezoPlan(enabled=True)
    wf = _build("single_drop", piezo=piezo, eis_step_by_channel={21: _eis(21)})
    names = [s.name for s in wf.setup]
    assert names.index("measure_eis_ch21") < names.index("piezo_off_ch21")


def test_piezo_all_elution_wraps_every_elution_event():
    """all_elution scope brackets the startup flush, precondition, deposit, final flush."""
    piezo = PiezoPlan(enabled=True, elution_scope="all_elution")
    wf = _build("two_phase", piezo=piezo, eis_step_by_channel={21: _eis(21)})
    names = [s.name for s in wf.setup + wf.teardown]
    for evt in ("startup_flush", "precondition_ch21", "deposit_ch21", "final_flush"):
        assert f"piezo_on_{evt}" in names and f"piezo_off_{evt}" in names
    assert "piezo_on_ch21" not in names          # no legacy deposit-only naming
    assert "piezo_standby" in names
    # The (non-elution) EIS falls outside the deposit's piezo bracket.
    assert names.index("piezo_off_deposit_ch21") < names.index("measure_eis_ch21")


# ── Run-plan-driven phase ordering (anneal + pointwise/batch measurement) ─────

from softae.core.run_plan import RunPlan  # noqa: E402


def _anneal_catalog() -> TaskCatalog:
    cat = _engine_catalog()
    cat.add(Task(name="anneal_150C_5min", instrument="temp_controller", method="anneal",
                 params={"target_temp_C": 150, "hold_time_s": 300,
                         "ramp_rate": 5, "tolerance": 1.0}))
    return cat


def _build_plan(recipe_name, channels, formulation, run_plan, *, catalog=None, eis=True, **over):
    cat = catalog or _anneal_catalog()
    eis_by = {ch: _eis(ch) for ch in channels} if eis else None
    kw = dict(
        catalog=cat, pump_ids=[0, 1, 2], dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, settle_base_s=0.0,
        start_flush_uL=[80, 80, 80], pcb=PCB, origin_xy=(43.5, 50.0),
        eis_step_by_channel=eis_by, run_plan=run_plan,
    )
    kw.update(over)
    return build_recipe_deposition_workflow(
        get_deposition_recipe(recipe_name), channels, formulation, **kw)


def test_default_run_plan_interleaves_deposit_and_eis_per_channel():
    """No run_plan → today's per-channel deposit-then-EIS layout, unchanged."""
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]}, None)
    names = [s.name for s in wf.setup]
    assert names == ["startup_flush",
                     "deposit_ch21", "measure_eis_ch21",
                     "deposit_ch22", "measure_eis_ch22"]
    assert wf.metadata["deferred_measurement"] is False


def test_pointwise_anneal_interleaves_per_channel():
    wf = _build_plan("single_drop", [21], {21: [10.0, 30.0, 0.0]},
                     RunPlan.pointwise(anneal=True))
    names = [s.name for s in wf.setup]
    # The anneal is bracketed to the flush basin: the tip is protected for the
    # whole hold, and an anti-clog purge during it costs no motion.
    assert names == ["startup_flush", "deposit_ch21",
                     "anneal_to_flush_ch21", "anneal_rest_ch21",
                     "anneal_ch21", "anneal_leave_rest_ch21",
                     "measure_eis_ch21"]
    anneal = next(s for s in wf.setup if s.name == "anneal_ch21")
    assert anneal.instrument == "temp_controller"
    assert anneal.method == "anneal"
    assert anneal.tags["phase"] == "anneal"
    assert anneal.tags["channel"] == "21"


def test_two_phase_pointwise_anneal_order():
    wf = _build_plan("two_phase", [21], {21: [10.0, 30.0, 0.0]},
                     RunPlan.pointwise(anneal=True))
    names = [s.name for s in wf.setup]
    assert names == ["startup_flush", "precondition_ch21", "deposit_ch21",
                     "anneal_to_flush_ch21", "anneal_rest_ch21",
                     "anneal_ch21", "anneal_leave_rest_ch21",
                     "measure_eis_ch21"]


def test_batch_formulate_all_then_anneal_all_then_measure_all():
    wf = _build_plan("single_drop", [21, 22], {21: [10.0, 30.0, 0.0], 22: [5.0, 5.0, 5.0]},
                     RunPlan.batch(anneal=True))
    names = [s.name for s in wf.setup]
    assert names == ["startup_flush",
                     "deposit_ch21", "deposit_ch22",
                     "anneal_to_flush_all", "anneal_rest_all",
                     "anneal_all", "anneal_leave_rest_all",
                     "measure_eis_ch21", "measure_eis_ch22"]
    anneal = next(s for s in wf.setup if s.name == "anneal_all")
    assert "channel" not in anneal.tags       # whole-plate → campaign-level
    assert anneal.tags["phase"] == "anneal"
    assert wf.metadata["deferred_measurement"] is True


def test_the_anneal_bracket_parks_the_tip_in_the_flush_basin():
    """The hold is the longest stretch of a run; the tip must not sit in air.

    It is also the only stretch where no pump moves *and* where a purge costs
    nothing, which is why the rig is parked exactly where it purges.
    """
    from softae.core.deposition_steps import deposition_positions

    wf = _build_plan("single_drop", [21], {21: [10.0, 30.0, 0.0]},
                     RunPlan.pointwise(anneal=True))
    by_name = {s.name: s for s in wf.setup}

    travel = by_name["anneal_to_flush_ch21"]
    assert (travel.instrument, travel.method) == ("stage", "move_to")
    assert (travel.params["x"], travel.params["y"]) == deposition_positions().flush

    assert by_name["anneal_rest_ch21"].method == "head_descend"
    # Retracting afterwards is mandatory: the head guard refuses stage motion
    # while lowered, so leaving it down would block the next phase outright.
    assert by_name["anneal_leave_rest_ch21"].method == "head_retract"


def test_the_bracket_travels_before_it_lowers():
    """Reversed, the move would be refused by the head guard."""
    wf = _build_plan("single_drop", [21], {21: [1.0, 1.0, 1.0]},
                     RunPlan.pointwise(anneal=True))
    names = [s.name for s in wf.setup]
    assert names.index("anneal_to_flush_ch21") < names.index("anneal_rest_ch21")


def test_batch_without_anneal_is_formulate_all_then_measure_all():
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
                     RunPlan.batch(anneal=False))
    names = [s.name for s in wf.setup]
    assert names == ["startup_flush", "deposit_ch21", "deposit_ch22",
                     "measure_eis_ch21", "measure_eis_ch22"]


def test_anneal_params_override_task_defaults():
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [1, 1, 1]},
                     RunPlan.batch(anneal=True, anneal_params={"target_temp_C": 120,
                                                               "hold_time_s": 600}))
    anneal = next(s for s in wf.setup if s.name == "anneal_all")
    assert anneal.params["target_temp_C"] == 120
    assert anneal.params["hold_time_s"] == 600


def test_long_anneal_gets_a_timeout_that_outlasts_its_hold():
    """P1.6: a multi-hour hold must not inherit the task's short ceiling.

    The catalogued task declares 600 s. Overriding the hold to 4 h via
    ``anneal_params`` used to leave that 600 s in place, so the executor aborted
    the anneal partway — and with graceful recovery enabled the campaign absorbed
    it as a channel skip, silently producing wrongly-annealed samples.
    """
    four_hours = 4 * 3600
    cat = _anneal_catalog()
    cat.get("anneal_150C_5min").timeout_s = 600.0

    wf = _build_plan(
        "single_drop", [21], {21: [10.0, 30.0, 0.0]},
        RunPlan.batch(anneal=True, anneal_params={"hold_time_s": four_hours}),
        catalog=cat,
    )
    anneal = next(s for s in wf.setup if s.name == "anneal_all")
    assert anneal.params["hold_time_s"] == four_hours
    assert anneal.timeout_s > four_hours, (
        "anneal ceiling must outlast the hold it was asked to perform"
    )


def test_short_anneal_keeps_a_sane_ceiling():
    """The shipped 5-minute anneal is unaffected by the derivation."""
    wf = _build_plan("single_drop", [21], {21: [10.0, 30.0, 0.0]},
                     RunPlan.batch(anneal=True))
    anneal = next(s for s in wf.setup if s.name == "anneal_all")
    assert anneal.timeout_s is not None
    assert anneal.timeout_s >= 300.0


# ── Commanded conditions at phase boundaries (B1) ────────────────────────────
#
# `RunPhase.conditions` says what the chamber is *told* to hold for a phase. The
# engine's job is narrow and entirely about timing: emit the establish steps at
# the boundary, and only when the commanded environment actually changes. The
# steps themselves are `PhaseSetpoints.establish_steps`' business and are pinned
# in `test_phase_setpoints.py`; nothing here re-tests them.

from softae.core.phase_setpoints import PhaseSetpoints  # noqa: E402
from softae.core.run_plan import (  # noqa: E402
    PhaseKind,
    PhaseScope,
    RunPhase,
    SettlePlan,
)

CASTING = PhaseSetpoints(name="casting", temp_setpoint_C=25.0, rh_setpoint_pct=40.0)
CURING = PhaseSetpoints(name="curing", temp_setpoint_C=85.0, rh_setpoint_pct=20.0)
HOLDING = PhaseSetpoints(name="holding", temp_setpoint_C=25.0, rh_setpoint_pct=50.0)

#: Five steps per establish: temp sp, temp wait, rh sp, rh start, rh wait.
_STEPS_PER_CONDITION = 5


def _settle() -> SettlePlan:
    return SettlePlan(round_period_s=240.0, min_hold_s=1500.0, max_hold_s=14400.0,
                      rh_stability_pct=None)


def _batch_plan_with_conditions() -> RunPlan:
    """cast @casting → cure @curing → equilibrate @holding → measure @holding."""
    return RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE, conditions=CASTING),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, conditions=CURING),
        RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                 settle=_settle(), conditions=HOLDING),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH, conditions=HOLDING),
    ))


def _condition_names(wf) -> list[str]:
    return [s.name for s in wf.setup if s.tags.get("phase") == "conditions"]


def test_conditions_emitted_at_each_boundary_in_phase_order():
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
                     _batch_plan_with_conditions())
    names = [s.name for s in wf.setup]
    assert names == [
        "startup_flush",
        "conditions_casting_temp_sp_ch21", "conditions_casting_temp_wait_ch21",
        "conditions_casting_rh_sp_ch21", "conditions_casting_rh_start_ch21",
        "conditions_casting_rh_wait_ch21",
        "deposit_ch21",
        "deposit_ch22",
        "conditions_curing_temp_sp_all", "conditions_curing_temp_wait_all",
        "conditions_curing_rh_sp_all", "conditions_curing_rh_start_all",
        "conditions_curing_rh_wait_all",
        "anneal_to_flush_all", "anneal_rest_all",
        "anneal_all", "anneal_leave_rest_all",
        "conditions_holding_temp_sp_all", "conditions_holding_temp_wait_all",
        "conditions_holding_rh_sp_all", "conditions_holding_rh_start_all",
        "conditions_holding_rh_wait_all",
        "measure_eis_ch21", "measure_eis_ch22",
    ]


def test_conditions_per_sample_phase_emits_once_not_once_per_channel():
    """The chamber is one chamber: four casts under one casting environment.

    The per-sample segment loops channels, so a naive emitter would establish the
    casting conditions four times and wait out four approaches.
    """
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
                     _batch_plan_with_conditions())
    casting = [n for n in _condition_names(wf) if "_casting_" in n]
    assert len(casting) == _STEPS_PER_CONDITION
    assert all(n.endswith("_ch21") for n in casting), (
        "established on the first channel of the segment, not re-established")


def test_conditions_repeat_setpoint_emits_no_second_establish():
    """EQUILIBRATE and MEASURE share `holding`; the second phase costs nothing.

    Re-emitting would not merely be idle: the RH wait carries
    ``raise_on_timeout=True``, so a needless second approach is a new way for a
    correct run to abort.
    """
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
                     _batch_plan_with_conditions())
    holding = [n for n in _condition_names(wf) if "_holding_" in n]
    assert len(holding) == _STEPS_PER_CONDITION


def test_conditions_change_back_is_established_again():
    """A → B → A is three establishes, not two: the chamber is at B when A returns."""
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE, conditions=CASTING),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, conditions=CURING),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH, conditions=CASTING),
    ))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan)
    assert len(_condition_names(wf)) == 3 * _STEPS_PER_CONDITION


def test_conditions_steps_carry_the_phase_and_condition_tags():
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     _batch_plan_with_conditions())
    curing = [s for s in wf.setup if s.name.startswith("conditions_curing_")]
    assert curing, "the cure phase's conditions were not emitted"
    for step in curing:
        assert step.tags["phase"] == "conditions"
        assert step.tags["condition"] == "curing"


def test_conditions_same_name_different_setpoints_get_distinct_step_names():
    """Two phases may label different environments alike; `depends_on` is by name."""
    hot = PhaseSetpoints(name="hold", temp_setpoint_C=85.0)
    cold = PhaseSetpoints(name="hold", temp_setpoint_C=25.0)
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE, conditions=hot),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH, conditions=cold),
    ))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan)
    names = [s.name for s in wf.setup]
    assert len(names) == len(set(names))


def test_equilibrate_per_batch_emits_its_conditions_and_no_measurement_steps():
    """The phase terminates on evidence, so the engine lays out no steps for it.

    Everything after the cure is asserted, not just the window before the first
    measure step: an arm that wrongly emitted a sweep of its own would sit
    *inside* that window and move the boundary with it.
    """
    wf = _build_plan("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
                     _batch_plan_with_conditions())
    names = [s.name for s in wf.setup]
    assert names[names.index("anneal_leave_rest_all") + 1:] == [
        "conditions_holding_temp_sp_all", "conditions_holding_temp_wait_all",
        "conditions_holding_rh_sp_all", "conditions_holding_rh_start_all",
        "conditions_holding_rh_wait_all",
        "measure_eis_ch21", "measure_eis_ch22",
    ]


def test_equilibrate_per_sample_emits_its_conditions_and_no_measurement_steps():
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_SAMPLE,
                 settle=_settle(), conditions=HOLDING),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_SAMPLE),
    ))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan)
    names = [s.name for s in wf.setup]
    assert names == [
        "startup_flush", "deposit_ch21",
        "conditions_holding_temp_sp_ch21", "conditions_holding_temp_wait_ch21",
        "conditions_holding_rh_sp_ch21", "conditions_holding_rh_start_ch21",
        "conditions_holding_rh_wait_ch21",
        "measure_eis_ch21",
    ]


def test_equilibrate_without_conditions_emits_nothing_at_all():
    """The no-op `run_plan`'s module warning describes — now stated, not silent."""
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH, settle=_settle()),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
    ))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan)
    assert [s.name for s in wf.setup] == [
        "startup_flush", "deposit_ch21", "measure_eis_ch21"]


def test_conditions_free_plan_emits_the_pre_conditions_workflow_unchanged():
    """Byte-identity pin: a plan with no `conditions` is untouched by B1.

    The full ordered shape of every step — name, instrument, method, tags — plus
    the metadata, for the batch plan the engine has always built. Params are
    pinned by the recipe tests above and are not reachable from the conditions
    walk, which only ever *appends* steps.
    """
    wf = _build_plan("single_drop", [21, 22],
                     {21: [10.0, 30.0, 0.0], 22: [5.0, 5.0, 5.0]},
                     RunPlan.batch(anneal=True))
    shape = [(s.name, s.instrument, s.method, sorted(s.tags.items()))
             for s in wf.setup]
    assert shape == [
        ("startup_flush", "liquid_handler", "startup_flush", []),
        ("deposit_ch21", "liquid_handler", "single_drop_simul",
         [("channel", "21"), ("phase", "deposit")]),
        ("deposit_ch22", "liquid_handler", "single_drop_simul",
         [("channel", "22"), ("phase", "deposit")]),
        ("anneal_to_flush_all", "stage", "move_to", [("phase", "anneal")]),
        ("anneal_rest_all", "syringe", "head_descend", [("phase", "anneal")]),
        ("anneal_all", "temp_controller", "anneal",
         [("phase", "anneal"), ("purge_window", "concurrent")]),
        ("anneal_leave_rest_all", "syringe", "head_retract", [("phase", "anneal")]),
        ("measure_eis_ch21", pico_for_channel(21), "sendscript_getdata",
         [("channel", "21"), ("purge_window", "concurrent")]),
        ("measure_eis_ch22", pico_for_channel(22), "sendscript_getdata",
         [("channel", "22"), ("purge_window", "concurrent")]),
    ]
    assert [(s.name, s.instrument, s.method) for s in wf.teardown] == [
        ("final_flush", "syringe", "single_pump")]
    assert wf.metadata["deferred_measurement"] is True
    assert not _condition_names(wf)


# ─────────────────────────── The campaign-level baseline (T11.28) ───────────────────────────
#
# Both observations behind the spec are about the stretch NOBODY drives: the
# pumps run before any axis is commanded, and a campaign following a `safe_park`
# starts from 10 degC and a zeroed humidity setpoint. The baseline is commanded
# at the very top of `setup`, waits for nothing, and is overridden by the first
# phase that speaks.

BASELINE = PhaseSetpoints(name="baseline", temp_setpoint_C=25.0,
                          rh_setpoint_pct=40.0)

#: Three steps per commanded baseline: temp sp, rh sp, rh start. No waits.
_STEPS_PER_BASELINE = 3


def _baseline_names(wf) -> list[str]:
    return [s.name for s in wf.setup if s.tags.get("baseline") == "true"]


def _step_shape(wf) -> list[tuple]:
    return [(s.name, s.instrument, s.method, s.params, s.timeout_s,
             sorted(s.tags.items())) for s in wf.setup]


def test_recipe_workflow_baseline_precedes_the_startup_flush():
    """The test the spec exists for: the axes are driven before the pumps run."""
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     _batch_plan_with_conditions(), baseline=BASELINE)
    names = [s.name for s in wf.setup]
    flush = names.index("startup_flush")
    baseline = [names.index(n) for n in _baseline_names(wf)]
    assert len(baseline) == _STEPS_PER_BASELINE
    assert max(baseline) < flush
    assert names[:_STEPS_PER_BASELINE] == [
        "conditions_baseline_temp_sp_baseline",
        "conditions_baseline_rh_sp_baseline",
        "conditions_baseline_rh_start_baseline",
    ]


def test_recipe_workflow_baseline_precedes_the_piezo_event_step():
    """Ahead of the piezo profile too: the top of `setup` means the top."""
    piezo = PiezoPlan(enabled=True, event_task="piezo_liquid_event")
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, None,
                     baseline=BASELINE, piezo=piezo)
    names = [s.name for s in wf.setup]
    assert names.index("piezo_event") == _STEPS_PER_BASELINE


def test_recipe_workflow_baseline_emits_no_wait_step():
    """Commanded, not gated. A wait here is a new refusal before the first cast."""
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     _batch_plan_with_conditions(), baseline=BASELINE)
    baseline_steps = [s for s in wf.setup if s.tags.get("baseline") == "true"]
    assert baseline_steps
    assert not any(s.method == "wait" for s in baseline_steps)


def test_recipe_workflow_baseline_absent_leaves_setup_unchanged():
    """`baseline=None` reproduces today's step list exactly, params and tags."""
    args = ("single_drop", [21, 22], {21: [1, 1, 1], 22: [2, 2, 2]},
            _batch_plan_with_conditions())
    without = _build_plan(*args)
    default = _build_plan(*args, baseline=None)
    assert _step_shape(default) == _step_shape(without)
    assert not _baseline_names(default)


def test_recipe_workflow_first_phase_matching_baseline_emits_no_conditions():
    """The tracker is seeded, so the cast phase re-establishes nothing.

    Re-emitting would not merely be idle: `establish_steps` gives the RH wait
    `raise_on_timeout=True`, so a needless approach is a fresh way for a correct
    run to abort, and this one would sit right before the first cast.
    """
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE, conditions=CASTING),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, conditions=CURING),
    ))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan, baseline=CASTING)
    established = [n for n in _condition_names(wf) if n not in _baseline_names(wf)]
    assert not [n for n in established if "_casting_" in n]
    assert len(_baseline_names(wf)) == _STEPS_PER_BASELINE
    assert len([n for n in established if "_curing_" in n]) == _STEPS_PER_CONDITION


def test_recipe_workflow_first_phase_differing_from_baseline_still_waits():
    """A floor, not a policy: the first phase that speaks overrides it in full."""
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     _batch_plan_with_conditions(), baseline=BASELINE)
    casting = [s for s in wf.setup if "_casting_" in s.name]
    assert len(casting) == _STEPS_PER_CONDITION
    rh_wait = next(s for s in casting
                   if s.method == "wait" and s.instrument == "rh_controller")
    assert rh_wait.params["raise_on_timeout"] is True


def test_recipe_workflow_baseline_never_returns_after_a_phase_overrides_it():
    """It is the earliest setpoint and only that.

    A baseline that re-asserted itself would fight the ANNEAL phase's rest state
    at the one boundary this project has reasoned hardest about.
    """
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     _batch_plan_with_conditions(), baseline=BASELINE)
    assert _baseline_names(wf) == [s.name for s in wf.setup[:_STEPS_PER_BASELINE]]


def test_recipe_workflow_baseline_drives_only_the_axes_it_names():
    """An omitted axis is not driven: the same rule the phases obey."""
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, None,
                     baseline=PhaseSetpoints("baseline", rh_setpoint_pct=22.0))
    assert [(s.instrument, s.method) for s in wf.setup[:2]] == [
        ("rh_controller", "set_setpoint"), ("rh_controller", "start")]
    assert wf.setup[0].params["val"] == 22.0


# ── An anneal's duration and its temperature: one authority each ─────────────
#
# `docs/SubAgent docs/anneal_phase_duration.md` §4, operator ruling D1 = (d):
# the catalog task stays the sole authority for the hardware command, and
# `RunPhase.hold_s` is the one typed per-run spelling of its hold.


def _anneal_plan(**phase_kw) -> RunPlan:
    """FORMULATE per-sample → ANNEAL per-batch carrying *phase_kw*."""
    return RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 anneal_task="anneal_150C_5min", **phase_kw),
    ))


def _emitted_anneal(plan: RunPlan, **build_kw):
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan, eis=False,
                     **build_kw)
    return next(s for s in wf.setup if s.name == "anneal_all")


def test_anneal_hold_s_reaches_the_emitted_step():
    """THE CONTROL. The second assertion is the safety claim, not the first.

    A `hold_s` written into the step *after* `anneal_timeout_s` derived the
    ceiling would satisfy the param assertion and still leave the executor
    aborting the hold partway — the `36d574a`-shaped near-miss. The catalog
    fixture's `anneal_150C_5min` declares no `timeout_s`, so the derived floor
    `3600 × 1.25 + 900` is the maximum and can be asserted exactly.
    """
    anneal = _emitted_anneal(_anneal_plan(hold_s=3600.0))

    assert anneal.params["hold_time_s"] == 3600.0
    assert anneal.timeout_s == 3600.0 * ANNEAL_TIMEOUT_MARGIN \
        + ANNEAL_RAMP_ALLOWANCE_S
    assert anneal.timeout_s == 5400.0


def test_anneal_hold_s_and_the_tasks_own_hold_are_not_both_on_the_wire():
    """The override replaces the task's hold rather than sitting beside it."""
    anneal = _emitted_anneal(_anneal_plan(hold_s=3600.0))

    assert anneal.params["hold_time_s"] == 3600.0      # task declares 300
    assert anneal.params["target_temp_C"] == 150       # the rest is the task's


def test_no_hold_s_leaves_the_catalog_task_untouched():
    """The control for the control: absence changes nothing."""
    anneal = _emitted_anneal(_anneal_plan())

    assert anneal.params["hold_time_s"] == 300
    assert anneal.timeout_s == 300 * ANNEAL_TIMEOUT_MARGIN + ANNEAL_RAMP_ALLOWANCE_S


def _rest_warnings(plan):
    """The emitted anneal step, and every rest-at-cure warning the build logged."""
    import structlog

    with structlog.testing.capture_logs() as logs:
        anneal = _emitted_anneal(plan)
    return anneal, [e for e in logs
                    if e.get("event") == "anneal_rests_at_cure_temperature"]


def test_conditions_equal_to_cure_temperature_warns():
    """BOTH HALVES, or the check cannot fail.

    Equal is the one combination that is certainly a mis-read: the phase
    pre-ramps to the cure temperature and then "restores" to it, so the chamber
    rests hot. Unequal is the ordinary case and must stay silent — a warning
    that fires on everything is a log line, not a check.
    """
    step, equal = _rest_warnings(
        _anneal_plan(conditions=PhaseSetpoints("cure", 150.0, 20.0)))
    _, unequal = _rest_warnings(
        _anneal_plan(conditions=PhaseSetpoints("cooldown", 65.0, 20.0)))

    assert len(equal) == 1
    assert equal[0]["temp_C"] == 150.0
    assert equal[0]["task"] == "anneal_150C_5min"
    assert equal[0]["channel"] == "all"
    assert "rest at 150 °C" in equal[0]["detail"]
    assert "hot film" in equal[0]["detail"]
    assert unequal == []
    # Never a refusal: the plan still builds, and builds the same step.
    assert step.params["target_temp_C"] == 150


def test_conditions_disagreeing_leaves_task_temperature_on_the_wire():
    """CHARACTERIZATION — green before this change and green after. Not a control.

    Pins the behaviour measured in ``anneal_phase_duration.md`` §2: the task
    wins the hold command, ``conditions`` wins the resting state afterwards, and
    neither is discarded. A refactor that let ``conditions`` reach
    ``target_temp_C`` would cure four samples at the wrong temperature.
    """
    plan = _anneal_plan(conditions=PhaseSetpoints("cooldown", 65.0, 20.0))
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan, eis=False)

    anneal = next(s for s in wf.setup if s.name == "anneal_all")
    write_sp = next(s for s in wf.setup
                    if s.tags.get("phase") == "conditions" and s.method == "write_sp")

    assert anneal.params["target_temp_C"] == 150       # the cure, from the task
    assert write_sp.params["T_SP"] == 65.0             # the restore, from the phase


# ── Refusing a named task the catalog does not hold ──────────────────────────
#
# The engine used to test membership and drop the step. A plan naming a flush
# task the catalog lacks therefore compiled without it, so the rig cast through
# unflushed lines and nothing in the log said so. Declaring no task at all is a
# different thing and stays a legal no-op.

from dataclasses import replace  # noqa: E402

from softae.core.deposition_recipe import PlanCompileError  # noqa: E402


def _catalog_without(task_name: str) -> TaskCatalog:
    """The full engine catalog minus one task."""
    cat = _anneal_catalog()
    cat.remove(task_name)
    return cat


def _build_recipe(recipe, catalog, **over):
    """Compile *recipe* directly, for the cases that alter the recipe itself."""
    kw = dict(
        catalog=catalog, pump_ids=[0, 1, 2], dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, start_flush_uL=[80, 80, 80],
        pcb=PCB, origin_xy=(43.5, 50.0),
    )
    kw.update(over)
    return build_recipe_deposition_workflow(
        recipe, [21], {21: [10.0, 30.0, 0.0]}, **kw)


def test_engine_missing_startup_flush_refuses():
    """A named campaign-start flush the catalog lacks stops the compile.

    Skipped silently, the rig casts onto a board through unflushed lines.
    """
    with pytest.raises(PlanCompileError, match="startup_flush_full"):
        _build("single_drop", catalog=_catalog_without("startup_flush_full"))


def test_engine_missing_final_flush_refuses():
    """The teardown flush too: named and absent is a refusal, not a skip."""
    with pytest.raises(PlanCompileError, match="final_flush"):
        _build("single_drop", catalog=_catalog_without("final_flush"))


def test_engine_missing_anneal_task_refuses():
    """A cure phase whose task is absent would leave every sample uncured."""
    with pytest.raises(PlanCompileError, match="anneal_150C_5min"):
        _build_plan("single_drop", [21], {21: [1, 1, 1]},
                    RunPlan.pointwise(anneal=True), catalog=_engine_catalog())


def test_engine_enabled_piezo_missing_task_refuses():
    """An enabled piezo that cannot resolve its task would actuate nothing."""
    with pytest.raises(PlanCompileError, match="piezo_channel_a_on"):
        _build("two_phase", catalog=_catalog_without("piezo_channel_a_on"),
               piezo=PiezoPlan(enabled=True))


def test_engine_disabled_piezo_missing_task_compiles():
    """A disabled piezo asks for nothing, so its task names are never resolved."""
    wf = _build("two_phase", catalog=_catalog_without("piezo_channel_a_on"),
                piezo=PiezoPlan(enabled=False))
    assert not any("piezo" in s.name for s in wf.setup + wf.teardown)


def test_engine_undeclared_flush_methods_compiles():
    """A recipe naming no flush asks for nothing, which stays legal.

    Only a name the catalog cannot resolve is a refusal.
    """
    recipe = replace(get_deposition_recipe("single_drop"),
                     startup_method="", final_method="")
    wf = _build_recipe(recipe, _engine_catalog())
    assert [s.name for s in wf.setup] == ["deposit_ch21"]
    assert wf.teardown == []


def test_engine_refusal_names_task_phase_and_near_miss():
    """The message has to be actionable away from the code: what, where, and did-you-mean."""
    recipe = replace(get_deposition_recipe("single_drop"),
                     startup_method="startup_flush_fll")
    with pytest.raises(PlanCompileError) as refusal:
        _build_recipe(recipe, _engine_catalog())
    message = str(refusal.value)
    assert "startup_flush_fll" in message
    assert "campaign-start flush" in message
    assert "startup_flush_full" in message


def test_engine_missing_deposit_method_override_refuses():
    """An operator-selected deposit method the catalog lacks stops the compile.

    The one lookup that still raised a bare KeyError: an unattended run would
    die on a stack trace naming neither the method nor the channel.
    """
    with pytest.raises(PlanCompileError) as refusal:
        _build("single_drop", deposit_method="single_drop_simool")
    message = str(refusal.value)
    assert "single_drop_simool" in message
    assert "deposit-method override on channel 21" in message
    assert "single_drop_simul" in message          # the near miss


def test_engine_missing_recipe_phase_method_refuses():
    """A recipe phase whose own method is absent refuses, naming the phase.

    Dropped instead, the channel would be measured without ever being cast.
    """
    recipe = replace(get_deposition_recipe("single_drop"),
                     phases=(replace(get_deposition_recipe("single_drop").phases[0],
                                     method="single_drop_absent"),))
    with pytest.raises(PlanCompileError) as refusal:
        _build_recipe(recipe, _engine_catalog())
    message = str(refusal.value)
    assert "single_drop_absent" in message
    assert "deposit phase on channel 21" in message


def test_engine_anneal_phase_naming_no_task_refuses():
    """An ANNEAL phase that names no task refuses rather than curing nothing.

    Unlike every other role, "nothing was asked for" cannot be a legal no-op
    here: an uncured sample is measured and told, not obviously missing.
    """
    plan = RunPlan((RunPhase(PhaseKind.FORMULATE),
                    RunPhase(PhaseKind.ANNEAL, anneal_task="")))
    with pytest.raises(PlanCompileError) as refusal:
        _build_plan("single_drop", [21], {21: [1, 1, 1]}, plan, eis=False)
    message = str(refusal.value)
    assert "ANNEAL phase over channel 21" in message
    assert "names no task" in message
    # Not the near-miss message for the empty string, which would suggest
    # catalog names for a name nobody wrote.
    assert "nearest names" not in message
