"""Tests for the commanded-environment noun (softae.core.phase_setpoints).

The type exists so a phase can say *what the chamber is told to do*, as against
the ``conditions`` table's record of what it did. Three things are load-bearing
and each is pinned below: ``None`` on an axis emits nothing, the five steps come
out in the one order that works, and the RH wait raises rather than warns.
"""

from __future__ import annotations

import pytest

from softae.core.phase_setpoints import (
    BASELINE_ABSENT_EVENT,
    BASELINE_EVENT,
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    DEFAULT_RH_TOLERANCE_PCT,
    DEFAULT_TOLERANCE_C,
    PhaseSetpoints,
    baseline_event_payload,
    first_phase_axes,
)


def _names(steps):
    return [s.name for s in steps]


def _by_method(steps):
    return [(s.instrument, s.method) for s in steps]


# ── the five-step sequence ───────────────────────────────────────────────────

class TestEstablishStepsOrder:
    """``temp write_sp → temp wait → rh set_setpoint → rh start → rh wait``.

    ``rh_start`` sits between the setpoint and the wait because the setpoint
    alone actuates nothing — a wait placed before it polls a stationary reading
    until it times out, which under ``raise_on_timeout`` is a failed run.
    """

    def test_establish_steps_both_axes_emits_the_five_step_sequence_in_order(self):
        sp = PhaseSetpoints("anneal", temp_setpoint_C=85.0, rh_setpoint_pct=20.0)
        assert _by_method(sp.establish_steps("all")) == [
            ("temp_controller", "write_sp"),
            ("temp_controller", "wait"),
            ("rh_controller", "set_setpoint"),
            ("rh_controller", "start"),
            ("rh_controller", "wait"),
        ]

    def test_establish_steps_temperature_only_emits_no_humidity_steps(self):
        sp = PhaseSetpoints("cure", temp_setpoint_C=85.0, rh_setpoint_pct=None)
        steps = sp.establish_steps("all")
        assert _by_method(steps) == [
            ("temp_controller", "write_sp"),
            ("temp_controller", "wait"),
        ]

    def test_establish_steps_humidity_only_emits_no_temperature_steps(self):
        sp = PhaseSetpoints("dry", temp_setpoint_C=None, rh_setpoint_pct=15.0)
        steps = sp.establish_steps("all")
        assert _by_method(steps) == [
            ("rh_controller", "set_setpoint"),
            ("rh_controller", "start"),
            ("rh_controller", "wait"),
        ]

    def test_establish_steps_no_axes_driven_emits_nothing(self):
        """``None`` on both axes is a phase that inherits the environment."""
        assert PhaseSetpoints("inherit").establish_steps("all") == []

    def test_establish_steps_suffix_distinguishes_per_sample_from_per_batch(self):
        sp = PhaseSetpoints("anneal", temp_setpoint_C=85.0)
        assert _names(sp.establish_steps("all")) != _names(sp.establish_steps("ch3"))
        assert all(n.endswith("_ch3") for n in _names(sp.establish_steps("ch3")))


# ── the refusal the plan depends on ──────────────────────────────────────────

class TestTheRHWaitIsAGate:
    """A step that cannot fail is not a check.

    ``raise_on_timeout`` is what stops a run whose humidity never arrived from
    proceeding into a multi-hour cure at an unknown RH. Losing it would turn
    every RH gate into a log line, and nothing else in the emitted workflow
    would look different.
    """

    def test_rh_wait_step_carries_raise_on_timeout_true(self):
        sp = PhaseSetpoints("anneal", rh_setpoint_pct=20.0)
        wait = sp.establish_steps("all")[-1]
        assert wait.method == "wait"
        assert wait.params["raise_on_timeout"] is True

    def test_rh_wait_step_ceiling_exceeds_its_own_driver_timeout(self):
        """``rh_wait``'s own rule: the step ceiling must outlast the wait it wraps.

        Otherwise the executor aborts the step while the driver is still inside
        its poll loop, and the failure is reported as a timeout rather than as
        the humidity refusal it would have become.
        """
        sp = PhaseSetpoints("anneal", rh_setpoint_pct=20.0,
                            rh_approach_timeout_s=5400.0)
        wait = sp.establish_steps("all")[-1]
        assert wait.params["timeout"] == pytest.approx(5400.0)
        assert wait.timeout_s > wait.params["timeout"]

    def test_temp_wait_step_ceiling_exceeds_its_own_driver_timeout(self):
        sp = PhaseSetpoints("anneal", temp_setpoint_C=85.0,
                            approach_timeout_s=2400.0)
        wait = sp.establish_steps("all")[1]
        assert wait.params["timeout"] == pytest.approx(2400.0)
        assert wait.timeout_s > wait.params["timeout"]


# ── the steps must be executable, i.e. match the driver signatures ───────────

class TestEmittedStepsMatchTheDriverSignatures:
    """``BaseInstrument.execute`` forwards ``step.params`` as **kwargs, unfiltered.

    So a param name that is not the driver's raises ``TypeError`` at the rig
    rather than at build time. These bind the emitted params against the real
    signatures, which is the only check that can catch a rename.
    """

    def test_temp_steps_bind_against_the_temperature_driver(self):
        import inspect

        from softae.drivers.async_temp_controller import AsyncTempController

        steps = PhaseSetpoints("anneal", temp_setpoint_C=85.0).establish_steps("all")
        for step in steps:
            method = getattr(AsyncTempController, step.method)
            inspect.signature(method).bind(None, **step.params)

    def test_rh_steps_bind_against_the_humidity_driver(self):
        import inspect

        from softae.drivers.async_rh_controller import AsyncRHController

        steps = PhaseSetpoints("anneal", rh_setpoint_pct=20.0).establish_steps("all")
        for step in steps:
            method = getattr(AsyncRHController, step.method)
            inspect.signature(method).bind(None, **step.params)

    def test_emitted_steps_are_tagged_as_the_conditions_phase(self):
        steps = PhaseSetpoints("anneal", temp_setpoint_C=85.0,
                               rh_setpoint_pct=20.0).establish_steps("all")
        assert {s.tags["phase"] for s in steps} == {"conditions"}
        assert {s.tags["condition"] for s in steps} == {"anneal"}
        assert {s.tags["axis"] for s in steps} == {"temperature", "humidity"}


# ── label ────────────────────────────────────────────────────────────────────

class TestLabel:
    def test_label_renders_name_and_both_axes(self):
        sp = PhaseSetpoints("anneal", temp_setpoint_C=85.0, rh_setpoint_pct=20.0)
        assert sp.label() == "anneal (85 °C, 20 %RH)"

    def test_label_omits_an_axis_that_is_not_driven(self):
        assert PhaseSetpoints("cure", temp_setpoint_C=85.0).label() == "cure (85 °C)"
        assert PhaseSetpoints("dry", rh_setpoint_pct=20.0).label() == "dry (20 %RH)"

    def test_label_says_so_when_neither_axis_is_driven(self):
        assert PhaseSetpoints("inherit").label() == "inherit (not driven)"


# ── validation ───────────────────────────────────────────────────────────────

class TestValidation:
    def test_a_nameless_setpoint_is_refused(self):
        with pytest.raises(ValueError, match="name"):
            PhaseSetpoints("   ", temp_setpoint_C=25.0)

    def test_a_nan_setpoint_is_refused_because_none_is_how_undriven_is_spelled(self):
        with pytest.raises(ValueError, match="finite"):
            PhaseSetpoints("anneal", temp_setpoint_C=float("nan"))

    @pytest.mark.parametrize("field_name", [
        "tolerance_C", "rh_tolerance_pct",
        "approach_timeout_s", "rh_approach_timeout_s",
    ])
    def test_a_non_positive_tolerance_or_timeout_is_refused(self, field_name):
        with pytest.raises(ValueError, match="positive"):
            PhaseSetpoints("anneal", temp_setpoint_C=25.0, **{field_name: 0.0})

    def test_zero_rh_setpoint_is_a_command_not_an_absence(self):
        """``0.0`` drives the axis dry; ``None`` leaves it alone. Different steps."""
        assert PhaseSetpoints("purge", rh_setpoint_pct=0.0).drives_humidity
        assert not PhaseSetpoints("purge", rh_setpoint_pct=None).drives_humidity
        assert len(PhaseSetpoints("purge", rh_setpoint_pct=0.0).establish_steps("all")) == 3
        assert PhaseSetpoints("purge", rh_setpoint_pct=None).establish_steps("all") == []


# ── the defaults must not fork from the module that measured them ────────────

def test_defaults_match_the_equilibration_run_that_measured_them():
    """These four are restated, not imported — so a retune there must fail here.

    ``workflows.equilibration`` is a 2 800-line async workflow that pulls the
    driver contracts in, and this type sits on the deposition engine's import
    path, which is why the numbers are copied rather than imported. This test is
    the price of that copy: it is the thing that makes the fork loud.
    """
    from softae.workflows import equilibration as eq

    assert DEFAULT_TOLERANCE_C == eq.DEFAULT_TOLERANCE_C
    assert DEFAULT_RH_TOLERANCE_PCT == eq.DEFAULT_RH_TOLERANCE_PCT
    assert DEFAULT_APPROACH_TIMEOUT_S == eq.DEFAULT_APPROACH_TIMEOUT_S
    assert DEFAULT_RH_APPROACH_TIMEOUT_S == eq.DEFAULT_RH_APPROACH_TIMEOUT_S


def test_rh_approach_timeout_is_a_phase_parameter_not_the_driver_default():
    """The driver's ``timeout=120.0`` is a monitoring poll, not an approach gate.

    ONE observed descent to ~20 %RH at 85 °C ran on the order of 5 000 s
    (2026-08-11 — a single measurement, not a rig constant), so a phase that
    inherited the driver default would fail its approach every time while the
    chamber worked normally — which under ``raise_on_timeout`` aborts a run with
    the samples already cast. The value has to be settable per phase, and the
    emitted step has to carry the phase's value rather than the driver's. It is a
    ceiling either way: the approach ends when the chamber arrives.
    """
    import inspect

    from softae.drivers.async_rh_controller import AsyncRHController

    driver_default = inspect.signature(AsyncRHController.wait).parameters["timeout"].default
    sp = PhaseSetpoints("anneal", rh_setpoint_pct=20.0, rh_approach_timeout_s=6000.0)
    wait = sp.establish_steps("all")[-1]
    assert wait.params["timeout"] == pytest.approx(6000.0)
    assert wait.params["timeout"] != driver_default


# -- the campaign-level baseline: commanded, never gated (T11.28) -------------

#: ``establish_steps("ch3")`` as HEAD built it, read out of
#: ``git show HEAD:src/softae/core/phase_setpoints.py`` and run *before* the
#: ``wait=`` factoring existed. That provenance is the whole value of the pin:
#: a literal regenerated from the refactored code would agree with itself no
#: matter what the refactor did.
_HEAD_ESTABLISH_STEPS = [
    ("conditions_anneal_temp_sp_ch3", "temp_controller", "write_sp",
     {"T_SP": 85.0, "print_flag": 0}, 30.0,
     {"axis": "temperature", "condition": "anneal", "phase": "conditions"}),
    ("conditions_anneal_temp_wait_ch3", "temp_controller", "wait",
     {"timeout": 2400.0, "within": 1.5}, 3060.0,
     {"axis": "temperature", "condition": "anneal", "phase": "conditions"}),
    ("conditions_anneal_rh_sp_ch3", "rh_controller", "set_setpoint",
     {"val": 20.0}, 30.0,
     {"axis": "humidity", "condition": "anneal", "phase": "conditions"}),
    ("conditions_anneal_rh_start_ch3", "rh_controller", "start",
     {}, 30.0,
     {"axis": "humidity", "condition": "anneal", "phase": "conditions"}),
    ("conditions_anneal_rh_wait_ch3", "rh_controller", "wait",
     {"raise_on_timeout": True, "target": 20.0, "timeout": 6000.0, "tol": 3.0},
     7560.0,
     {"axis": "humidity", "condition": "anneal", "phase": "conditions"}),
]


def _tuned() -> PhaseSetpoints:
    """Every field non-default, so the pin below can see any of them move."""
    return PhaseSetpoints("anneal", temp_setpoint_C=85.0, rh_setpoint_pct=20.0,
                          tolerance_C=1.5, rh_tolerance_pct=3.0,
                          approach_timeout_s=2400.0, rh_approach_timeout_s=6000.0)


def _rows(steps):
    return [(s.name, s.instrument, s.method, s.params, s.timeout_s, s.tags)
            for s in steps]


def test_phase_setpoints_establish_steps_output_unchanged_by_refactor():
    """The positive control on the ``wait=`` factoring.

    ``command_steps`` exists because ``establish_steps`` was split in two, and a
    split that quietly moved a param, a ceiling or a tag would leave every other
    test in this file green -- they assert instruments and methods, not numbers.
    """
    assert _rows(_tuned().establish_steps("ch3")) == _HEAD_ESTABLISH_STEPS


def test_phase_setpoints_command_steps_emits_no_wait():
    """The same steps, in the same order, minus the two ``wait`` steps."""
    steps = _tuned().command_steps("ch3")
    assert [s.method for s in steps] == ["write_sp", "set_setpoint", "start"]
    assert _rows(steps) == [
        (name, instrument, method, params, ceiling, {**tags, "baseline": "true"})
        for (name, instrument, method, params, ceiling, tags)
        in _HEAD_ESTABLISH_STEPS if method != "wait"
    ]


def test_phase_setpoints_command_steps_match_establish_apart_from_the_waits():
    """Name, instrument, method, params and ceiling identical; only tags differ.

    A baseline that reached the chamber differently from a phase's conditions
    would be a second way to command the same two axes, which is the thing
    ``PhaseSetpoints`` exists to prevent.
    """
    sp = _tuned()
    commanded = {s.name: s for s in sp.command_steps("ch3")}
    for step in sp.establish_steps("ch3"):
        if step.method == "wait":
            assert step.name not in commanded
            continue
        twin = commanded[step.name]
        assert (twin.instrument, twin.method, twin.params, twin.timeout_s) == (
            step.instrument, step.method, step.params, step.timeout_s)
        assert twin.tags == {**step.tags, "baseline": "true"}


def test_phase_setpoints_command_steps_carry_the_conditions_and_baseline_tags():
    steps = PhaseSetpoints("baseline", temp_setpoint_C=25.0,
                           rh_setpoint_pct=40.0).command_steps("baseline")
    assert {s.tags["phase"] for s in steps} == {"conditions"}
    assert {s.tags["baseline"] for s in steps} == {"true"}
    assert all("baseline" not in s.tags
               for s in PhaseSetpoints("anneal", temp_setpoint_C=25.0,
                                       rh_setpoint_pct=40.0).establish_steps("all"))


def test_phase_setpoints_command_steps_single_axis_emits_only_that_axis():
    """``None`` contributes no steps here either -- the rule is unchanged."""
    temp_only = PhaseSetpoints("warm", temp_setpoint_C=25.0).command_steps("baseline")
    assert [(s.instrument, s.method) for s in temp_only] == [
        ("temp_controller", "write_sp")]
    rh_only = PhaseSetpoints("damp", rh_setpoint_pct=22.0).command_steps("baseline")
    assert [(s.instrument, s.method) for s in rh_only] == [
        ("rh_controller", "set_setpoint"), ("rh_controller", "start")]
    assert PhaseSetpoints("inherit").command_steps("baseline") == []


def test_phase_setpoints_command_steps_rh_zero_is_a_commanded_dry_purge():
    """Section 4's direction rule, pinned on the wait-free path as well.

    Bench-verified 2026-08-21 (``async_rh_controller.safe_dry``): ``ctrl`` near
    zero is dry air and ``ctrl == 0`` exactly shuts both Aalborg PSVs. So a
    baseline cannot spell *leave it alone* as ``rh_setpoint_pct = 0.0`` -- that
    commands a dry purge, which after a park is the state the chamber is already
    stuck in. Absence is the only spelling of "not driven".
    """
    purge = PhaseSetpoints("purge", rh_setpoint_pct=0.0).command_steps("baseline")
    assert [(s.instrument, s.method) for s in purge] == [
        ("rh_controller", "set_setpoint"), ("rh_controller", "start")]
    assert purge[0].params["val"] == 0.0
    assert PhaseSetpoints("quiet", rh_setpoint_pct=None).command_steps("baseline") == []


# -- the run-start transcript -------------------------------------------------

class _Spec:
    def __init__(self, conditions=None, run_plan=None):
        self.conditions = conditions
        self.run_plan = run_plan


class _Plan:
    def __init__(self, *phases):
        self.phases = phases


class _Phase:
    def __init__(self, kind, conditions=None):
        self.kind = kind
        self.conditions = conditions


def test_baseline_event_payload_declared_reports_the_setpoints_and_waited_false():
    sp = PhaseSetpoints("baseline", temp_setpoint_C=25.0, rh_setpoint_pct=40.0)
    assert baseline_event_payload(_Spec(conditions=sp)) == (
        BASELINE_EVENT,
        {"name": "baseline", "temp_setpoint_C": 25.0, "rh_setpoint_pct": 40.0,
         "waited": False},
    )


def test_baseline_event_payload_waited_is_stated_not_implied():
    """``command_steps`` emits no wait, so nothing in the run checked arrival.

    A transcript that merely omitted the key would read as *arrived* to anyone
    who did not know the emission shape.
    """
    sp = PhaseSetpoints("baseline", rh_setpoint_pct=22.0)
    _, payload = baseline_event_payload(_Spec(conditions=sp))
    assert payload["waited"] is False
    assert payload["temp_setpoint_C"] is None


def test_baseline_event_payload_absent_names_the_first_phase_and_its_axes():
    """The T11.28 case itself, recorded rather than inferred from silence."""
    plan = _Plan(_Phase("formulate"), _Phase("anneal"))
    assert baseline_event_payload(_Spec(run_plan=plan)) == (
        BASELINE_ABSENT_EVENT,
        {"first_phase_kind": "formulate", "first_phase_drives": []},
    )


def test_baseline_event_payload_absent_lists_a_speaking_first_phases_axes():
    """The check can fail *and* pass: a first phase that speaks reports its axes."""
    plan = _Plan(_Phase("formulate",
                        PhaseSetpoints("casting", temp_setpoint_C=25.0,
                                       rh_setpoint_pct=40.0)))
    event, payload = baseline_event_payload(_Spec(run_plan=plan))
    assert event == BASELINE_ABSENT_EVENT
    assert payload["first_phase_drives"] == ["temperature", "humidity"]


def test_first_phase_axes_reports_each_driven_axis_separately():
    assert first_phase_axes(_Plan(_Phase(
        "formulate", PhaseSetpoints("warm", temp_setpoint_C=25.0)))) == ["temperature"]
    assert first_phase_axes(_Plan(_Phase(
        "formulate", PhaseSetpoints("damp", rh_setpoint_pct=0.0)))) == ["humidity"]
    assert first_phase_axes(_Plan()) == []
    assert first_phase_axes(None) == []
