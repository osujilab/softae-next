"""What each committed campaign example compiles to, written out in full.

For every spec under ``examples/``: the ordered steps of the workflow's three
lists, the phase each step is tagged with, the startup and teardown flushes,
the volumes the solver hands each channel, and the setpoints the condition
steps command. Compiled through ``build_trial_workflow`` — the builder the
campaign loop itself runs — against the placeholder catalogs the package
ships, so the result is the same on every checkout rather than on one bench.

The expectations are Python literals rather than generated snapshots: a reader
can see what the rig is asked to do without running anything, and a compiler
change shows up as a diff of prose. The trade is that this is coarser than a
byte-for-byte comparison, so the assertions are specific about what a wrong
plan gets wrong — which list a step is in, and in what order.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from tests.support import fixture_catalog
from tests.support.fixture_catalog import compile_example

#: Bound, not redefined: the fixture lives once in the helper module, and pytest
#: finds a fixture by name in whatever namespace it is bound to.
task_catalog = fixture_catalog.task_catalog

#: One entry per step: ``(name, instrument, method)``.
Step = tuple[str, str, str]

#: The sequence a four-well plate campaign compiles to: baseline conditions,
#: startup flush, casting conditions, four casts, the cure, the settle
#: conditions, then the four reads.
_PLATE_SETUP: tuple[Step, ...] = (
    ("conditions_baseline_temp_sp_baseline", "temp_controller", "write_sp"),
    ("conditions_baseline_rh_sp_baseline", "rh_controller", "set_setpoint"),
    ("conditions_baseline_rh_start_baseline", "rh_controller", "start"),
    ("startup_flush", "liquid_handler", "startup_flush"),
    ("conditions_casting_temp_sp_ch1", "temp_controller", "write_sp"),
    ("conditions_casting_temp_wait_ch1", "temp_controller", "wait"),
    ("conditions_casting_rh_sp_ch1", "rh_controller", "set_setpoint"),
    ("conditions_casting_rh_start_ch1", "rh_controller", "start"),
    ("conditions_casting_rh_wait_ch1", "rh_controller", "wait"),
    ("deposit_ch1", "liquid_handler", "single_drop_simul"),
    ("deposit_ch2", "liquid_handler", "single_drop_simul"),
    ("deposit_ch3", "liquid_handler", "single_drop_simul"),
    ("deposit_ch4", "liquid_handler", "single_drop_simul"),
    ("conditions_anneal_temp_sp_all", "temp_controller", "write_sp"),
    ("conditions_anneal_temp_wait_all", "temp_controller", "wait"),
    ("conditions_anneal_rh_sp_all", "rh_controller", "set_setpoint"),
    ("conditions_anneal_rh_start_all", "rh_controller", "start"),
    ("conditions_anneal_rh_wait_all", "rh_controller", "wait"),
    ("anneal_to_flush_all", "stage", "move_to"),
    ("anneal_rest_all", "syringe", "head_descend"),
    ("anneal_all", "temp_controller", "anneal"),
    ("anneal_leave_rest_all", "syringe", "head_retract"),
    ("conditions_equilibrate_temp_sp_all", "temp_controller", "write_sp"),
    ("conditions_equilibrate_temp_wait_all", "temp_controller", "wait"),
    ("conditions_equilibrate_rh_sp_all", "rh_controller", "set_setpoint"),
    ("conditions_equilibrate_rh_start_all", "rh_controller", "start"),
    ("conditions_equilibrate_rh_wait_all", "rh_controller", "wait"),
    ("measure_eis_ch1", "pico1", "sendscript_getdata"),
    ("measure_eis_ch2", "pico1", "sendscript_getdata"),
    ("measure_eis_ch3", "pico1", "sendscript_getdata"),
    ("measure_eis_ch4", "pico1", "sendscript_getdata"),
)

#: The same plate sequence, with no campaign baseline and no casting conditions
#: (that example states neither) and aimed at four non-contiguous channels.
_PRECAST_SETUP: tuple[Step, ...] = (
    ("startup_flush", "liquid_handler", "startup_flush"),
    ("deposit_ch1", "liquid_handler", "single_drop_simul"),
    ("deposit_ch11", "liquid_handler", "single_drop_simul"),
    ("deposit_ch13", "liquid_handler", "single_drop_simul"),
    ("deposit_ch14", "liquid_handler", "single_drop_simul"),
    ("conditions_anneal_temp_sp_all", "temp_controller", "write_sp"),
    ("conditions_anneal_temp_wait_all", "temp_controller", "wait"),
    ("conditions_anneal_rh_sp_all", "rh_controller", "set_setpoint"),
    ("conditions_anneal_rh_start_all", "rh_controller", "start"),
    ("conditions_anneal_rh_wait_all", "rh_controller", "wait"),
    ("anneal_to_flush_all", "stage", "move_to"),
    ("anneal_rest_all", "syringe", "head_descend"),
    ("anneal_all", "temp_controller", "anneal"),
    ("anneal_leave_rest_all", "syringe", "head_retract"),
    ("conditions_equilibrate_temp_sp_all", "temp_controller", "write_sp"),
    ("conditions_equilibrate_temp_wait_all", "temp_controller", "wait"),
    ("conditions_equilibrate_rh_sp_all", "rh_controller", "set_setpoint"),
    ("conditions_equilibrate_rh_start_all", "rh_controller", "start"),
    ("conditions_equilibrate_rh_wait_all", "rh_controller", "wait"),
    ("measure_eis_ch1", "pico1", "sendscript_getdata"),
    ("measure_eis_ch11", "pico1", "sendscript_getdata"),
    ("measure_eis_ch13", "pico1", "sendscript_getdata"),
    ("measure_eis_ch14", "pico1", "sendscript_getdata"),
)

#: No run plan at all: cast one well, read it, move to the next.
_POINTWISE_SETUP: tuple[Step, ...] = (
    ("startup_flush", "liquid_handler", "startup_flush"),
    ("deposit_ch1", "liquid_handler", "single_drop_simul"),
    ("measure_eis_ch1", "pico1", "sendscript_getdata"),
    ("deposit_ch2", "liquid_handler", "single_drop_simul"),
    ("measure_eis_ch2", "pico1", "sendscript_getdata"),
    ("deposit_ch3", "liquid_handler", "single_drop_simul"),
    ("measure_eis_ch3", "pico1", "sendscript_getdata"),
    ("deposit_ch4", "liquid_handler", "single_drop_simul"),
    ("measure_eis_ch4", "pico1", "sendscript_getdata"),
)

_TEARDOWN: tuple[Step, ...] = (("final_flush", "syringe", "single_pump"),)

#: ``phase`` tag values across ``setup``, consecutive repeats collapsed. ``-``
#: is a step carrying no phase tag: the flushes and the EIS reads.
_PLATE_PHASES = ("conditions", "-", "conditions", "deposit", "conditions",
                 "anneal", "conditions", "-")
_PRECAST_PHASES = ("-", "deposit", "conditions", "anneal", "conditions", "-")
_POINTWISE_PHASES = ("-", "deposit", "-", "deposit", "-", "deposit", "-",
                     "deposit", "-")


@dataclasses.dataclass(frozen=True)
class Expected:
    """Everything one example is expected to compile to."""

    setup: tuple[Step, ...]
    loop_steps: tuple[Step, ...]
    teardown: tuple[Step, ...]
    phases: tuple[str, ...]
    vols: dict[str, list[float]]
    setpoints: dict[str, float]


#: A pinned three-stock recipe: the same solved volumes in every well.
_PINNED_VOLS = [85.56919852682813, 11.182499999999951, 0.5936853979030321]

EXPECTED: dict[str, Expected] = {
    "bench_instance": Expected(
        setup=_PLATE_SETUP,
        loop_steps=(),
        teardown=_TEARDOWN,
        phases=_PLATE_PHASES,
        vols={f"deposit_ch{n}": _PINNED_VOLS for n in (1, 2, 3, 4)},
        setpoints={
            "conditions_baseline_temp_sp_baseline": 25.0,
            "conditions_baseline_rh_sp_baseline": 22.0,
            "conditions_casting_temp_sp_ch1": 25.0,
            "conditions_casting_rh_sp_ch1": 22.0,
            "conditions_anneal_temp_sp_all": 25.0,
            "conditions_anneal_rh_sp_all": 22.0,
            "conditions_equilibrate_temp_sp_all": 25.0,
            "conditions_equilibrate_rh_sp_all": 22.0,
        },
    ),
    # Same plate sequence as above; the searched recipe solves to its own
    # volumes at the parameter-space midpoint.
    "bo_init_bench": Expected(
        setup=_PLATE_SETUP,
        loop_steps=(),
        teardown=_TEARDOWN,
        phases=_PLATE_PHASES,
        vols={f"deposit_ch{n}": [83.19227634552729, 13.978125000000015,
                                 0.5771941368501707]
              for n in (1, 2, 3, 4)},
        setpoints={
            "conditions_baseline_temp_sp_baseline": 25.0,
            "conditions_baseline_rh_sp_baseline": 22.0,
            "conditions_casting_temp_sp_ch1": 25.0,
            "conditions_casting_rh_sp_ch1": 22.0,
            "conditions_anneal_temp_sp_all": 25.0,
            "conditions_anneal_rh_sp_all": 22.0,
            "conditions_equilibrate_temp_sp_all": 25.0,
            "conditions_equilibrate_rh_sp_all": 22.0,
        },
    ),
    "rung3a_fake_cast": Expected(
        setup=_PRECAST_SETUP,
        loop_steps=(),
        teardown=_TEARDOWN,
        phases=_PRECAST_PHASES,
        vols={f"deposit_ch{n}": _PINNED_VOLS for n in (1, 11, 13, 14)},
        setpoints={
            "conditions_anneal_temp_sp_all": 25.0,
            "conditions_anneal_rh_sp_all": 22.0,
            "conditions_equilibrate_temp_sp_all": 25.0,
            "conditions_equilibrate_rh_sp_all": 22.0,
        },
    ),
    # Two raw volume parameters rather than a formulation, and no run plan:
    # nothing drives the chamber anywhere in this one.
    "shadow_campaign": Expected(
        setup=_POINTWISE_SETUP,
        loop_steps=(),
        teardown=_TEARDOWN,
        phases=_POINTWISE_PHASES,
        vols={f"deposit_ch{n}": [27.5, 47.5] for n in (1, 2, 3, 4)},
        setpoints={},
    ),
}

STEP_LISTS = ("setup", "loop_steps", "teardown")


# -- extraction ---------------------------------------------------------------

def steps_of(workflow: Any, list_name: str) -> list[Step]:
    """``(name, instrument, method)`` for one of the workflow's step lists."""
    return [(s.name, s.instrument, s.method)
            for s in (getattr(workflow, list_name, ()) or ())]


def first_mismatch(expected: tuple[Step, ...], actual: list[Step],
                   list_name: str) -> str | None:
    """A legible account of the first way *actual* departs from *expected*.

    ``None`` when they agree. Names the index and the step, because "the plan
    changed" is not something a reader can act on.
    """
    for index in range(min(len(expected), len(actual))):
        if expected[index] != actual[index]:
            return (f"{list_name}[{index}]: expected {expected[index]!r}, "
                    f"compiled {actual[index]!r}")
    if len(expected) != len(actual):
        longer, extra = (("compiled", actual[len(expected):])
                         if len(actual) > len(expected)
                         else ("expected", list(expected[len(actual):])))
        return (f"{list_name} has {len(actual)} steps, expected "
                f"{len(expected)}; only in {longer}: {[s[0] for s in extra]}")
    return None


def phase_runs(workflow: Any) -> tuple[str, ...]:
    """The ``phase`` tags across ``setup``, consecutive repeats collapsed."""
    out: list[str] = []
    for step in (workflow.setup or ()):
        tag = (step.tags or {}).get("phase", "-")
        if not out or out[-1] != tag:
            out.append(tag)
    return tuple(out)


def tagged_params(workflow: Any, phase: str) -> dict[str, dict[str, Any]]:
    """Params of every step in any list tagged with *phase*, keyed by step name."""
    return {
        step.name: dict(step.params)
        for name in STEP_LISTS
        for step in (getattr(workflow, name, ()) or ())
        if (step.tags or {}).get("phase") == phase
    }


@pytest.fixture
def compiled(request, task_catalog, monkeypatch):
    """The compiled workflow for the parametrised example."""
    return compile_example(request.param, task_catalog, monkeypatch)


# -- the assertions -----------------------------------------------------------

@pytest.mark.parametrize("compiled", list(EXPECTED), indirect=True)
def test_compiled_example_step_lists_match_the_recorded_sequence(compiled, request):
    """Each of the three lists, compared separately and in order: flattening
    them would let a step move from setup to teardown unnoticed.
    """
    expected = EXPECTED[request.node.callspec.params["compiled"]]
    for list_name in STEP_LISTS:
        detail = first_mismatch(getattr(expected, list_name),
                                steps_of(compiled, list_name), list_name)
        assert detail is None, detail


@pytest.mark.parametrize("compiled", list(EXPECTED), indirect=True)
def test_compiled_example_phase_tags_run_in_the_recorded_order(compiled, request):
    """The phases the executor sees, in order, with repeats collapsed."""
    expected = EXPECTED[request.node.callspec.params["compiled"]]
    assert phase_runs(compiled) == expected.phases


@pytest.mark.parametrize("compiled", list(EXPECTED), indirect=True)
def test_compiled_example_brackets_the_run_with_both_flushes(compiled, request):
    """Both flushes, asserted apart from the sequence above: a missing one
    compiles in silence and reaches the lines.
    """
    setup = steps_of(compiled, "setup")
    teardown = steps_of(compiled, "teardown")
    assert ("startup_flush", "liquid_handler", "startup_flush") in setup, (
        f"no startup flush compiled; setup begins {setup[:3]}")
    assert ("final_flush", "syringe", "single_pump") in teardown, (
        f"no teardown flush compiled; teardown is {teardown}")


@pytest.mark.parametrize("compiled", list(EXPECTED), indirect=True)
def test_compiled_example_deposits_the_solved_volumes_per_channel(compiled, request):
    """The volumes the solver hands each well, per pump — asserted on values,
    because a wrong formulation emits as many steps as a right one.
    """
    expected = EXPECTED[request.node.callspec.params["compiled"]]
    deposits = tagged_params(compiled, "deposit")
    assert sorted(deposits) == sorted(expected.vols)
    for name, want in expected.vols.items():
        assert deposits[name]["vols"] == pytest.approx(want, rel=1e-9), name


@pytest.mark.parametrize("compiled", list(EXPECTED), indirect=True)
def test_compiled_example_commands_the_recorded_setpoints(compiled, request):
    """``T_SP`` and ``val`` on every condition step that commands an axis —
    the numbers the chamber is actually driven to.
    """
    expected = EXPECTED[request.node.callspec.params["compiled"]]
    commanded = {
        name: params.get("T_SP", params.get("val"))
        for name, params in tagged_params(compiled, "conditions").items()
        if "T_SP" in params or "val" in params
    }
    assert commanded == expected.setpoints


def test_first_mismatch_rejects_a_changed_dropped_or_moved_step(
        task_catalog, monkeypatch):
    """The control: three mutations, each of which must be reported. A
    comparator answering ``None`` to everything would leave this file vacuous.
    """
    expected = EXPECTED["shadow_campaign"].setup

    def fresh():
        return compile_example("shadow_campaign", task_catalog, monkeypatch)

    def detail_for(workflow, list_name="setup"):
        return first_mismatch(
            expected if list_name == "setup" else _TEARDOWN,
            steps_of(workflow, list_name), list_name)

    assert detail_for(fresh()) is None, "the comparator rejects a correct plan"

    changed = fresh()
    index = next(i for i, s in enumerate(changed.setup) if s.name == "deposit_ch2")
    changed.setup[index] = dataclasses.replace(changed.setup[index],
                                               method="not_a_method")
    detail = detail_for(changed)
    assert detail is not None and f"setup[{index}]" in detail, detail
    assert "not_a_method" in detail, detail

    shortened = fresh()
    dropped = shortened.setup.pop()
    detail = detail_for(shortened)
    assert detail is not None and dropped.name in detail, detail

    moved = fresh()
    moved.teardown = [moved.setup.pop()] + list(moved.teardown)
    assert detail_for(moved) is not None, "a step leaving setup went unreported"
    assert detail_for(moved, "teardown") is not None, (
        "a step arriving in teardown went unreported")
