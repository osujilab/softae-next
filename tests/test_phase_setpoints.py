"""Tests for the commanded-environment noun (softae.core.phase_setpoints).

The type exists so a phase can say *what the chamber is told to do*, as against
the ``conditions`` table's record of what it did. Three things are load-bearing
and each is pinned below: ``None`` on an axis emits nothing, the five steps come
out in the one order that works, and the RH wait raises rather than warns.
"""

from __future__ import annotations

import pytest

from softae.core.phase_setpoints import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    DEFAULT_RH_TOLERANCE_PCT,
    DEFAULT_TOLERANCE_C,
    PhaseSetpoints,
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

    The observed descent to ~20 %RH at 85 °C is on the order of 5 000 s, so a
    phase that inherited the driver default would fail its approach every time
    while the chamber worked normally — which under ``raise_on_timeout`` aborts a
    run with the samples already cast. The value has to be settable per phase,
    and the emitted step has to carry the phase's value rather than the driver's.
    """
    import inspect

    from softae.drivers.async_rh_controller import AsyncRHController

    driver_default = inspect.signature(AsyncRHController.wait).parameters["timeout"].default
    sp = PhaseSetpoints("anneal", rh_setpoint_pct=20.0, rh_approach_timeout_s=6000.0)
    wait = sp.establish_steps("all")[-1]
    assert wait.params["timeout"] == pytest.approx(6000.0)
    assert wait.params["timeout"] != driver_default
