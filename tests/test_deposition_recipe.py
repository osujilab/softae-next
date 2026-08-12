"""Tests for the recipe-driven deposition engine."""

from __future__ import annotations

import pytest

from softae.config.loader import pico_for_channel
from softae.core.deposition_recipe import (
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


def test_piezo_skipped_when_tasks_absent():
    # A catalog without piezo tasks + piezo enabled → no piezo steps, no crash.
    from softae.core.task_catalog import TaskCatalog
    cat = TaskCatalog()
    for n in ("startup_flush_full", "precondition_flush", "single_drop_simul", "final_flush"):
        cat.add(Task(name=n, instrument="lh", method=n, params={}))
    wf = build_recipe_deposition_workflow(
        get_deposition_recipe("two_phase"), [21], {21: [10.0, 30.0, 0.0]},
        catalog=cat, pump_ids=[0, 1, 2], dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, start_flush_uL=[80, 80, 80],
        piezo=PiezoPlan(enabled=True), pcb=PCB, origin_xy=(43.5, 50.0))
    assert not any("piezo" in s.name for s in wf.setup + wf.teardown)


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


def test_missing_anneal_task_is_skipped_not_fatal():
    # Catalog WITHOUT the anneal task → anneal phase emits nothing, no crash.
    wf = _build_plan("single_drop", [21], {21: [1, 1, 1]},
                     RunPlan.pointwise(anneal=True), catalog=_engine_catalog())
    names = [s.name for s in wf.setup]
    assert not any(n.startswith("anneal") for n in names)
    assert "deposit_ch21" in names and "measure_eis_ch21" in names


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
