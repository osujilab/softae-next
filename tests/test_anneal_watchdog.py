"""The anneal hold watchdog.

Soft-material anneals run for hours inside a single step. An unwatched
``time.sleep`` has two failure modes that both look like success: a heater that
drifts, sticks, or loses its setpoint yields a wrongly-annealed sample that is
then measured and recorded as valid; and an abort cannot interrupt the sleep.

Time is injected throughout, so these run instantly while exercising multi-hour
holds.
"""

from __future__ import annotations

import pytest

from softae.drivers.contracts import (
    DEFAULT_ANNEAL_FAULT_C,
    anneal_watchdog_config,
    monitored_hold,
)
from softae.errors import SafetyError


class _Clock:
    """Virtual clock: ``sleep`` advances time instead of blocking."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def _hold(pv_sequence, *, hold_time_s=3600.0, target=120.0, **kw):
    """Run a hold where each poll reads the next value from *pv_sequence*.

    A callable is used as-is; a list is consumed and then repeats its last value
    so a short list can describe a long hold.
    """
    clock = _Clock()
    if callable(pv_sequence):
        read = pv_sequence
    else:
        values = list(pv_sequence)

        def read():
            return values.pop(0) if len(values) > 1 else values[0]

    params = dict(
        read_pv=read, target_C=target, instrument="temp_controller",
        warn_C=3.0, fault_C=10.0, grace_s=120.0, poll_interval_s=30.0,
        sleep=clock.sleep, now=clock.now,
    )
    params.update(kw)
    return monitored_hold(hold_time_s, **params), clock


# ── The happy path ───────────────────────────────────────────────────────────

def test_a_stable_hold_runs_its_full_duration():
    report, clock = _hold([120.0], hold_time_s=3600.0)

    assert clock.t == pytest.approx(3600.0)
    assert report.held_s == pytest.approx(3600.0)
    assert report.n_warn == 0
    assert not report.aborted


def test_the_hold_is_actually_sampled_not_slept_through():
    """The point of the watchdog: it looks at the PV while it waits."""
    report, _ = _hold([120.0], hold_time_s=3600.0, poll_interval_s=30.0)
    assert report.n_samples == 3600 / 30


def test_small_wobble_within_tolerance_is_not_reported():
    report, _ = _hold([120.0, 121.5, 119.0, 120.5], hold_time_s=600.0)
    assert report.n_warn == 0


def test_excursion_range_is_recorded_for_the_run_record():
    report, _ = _hold([120.0, 124.0, 118.0, 120.0], hold_time_s=600.0)
    assert report.pv_min == 118.0
    assert report.pv_max == 124.0
    assert report.excursion_C == pytest.approx(6.0)


# ── Warn band: report, keep holding ──────────────────────────────────────────

def test_deviation_beyond_warn_band_warns_but_continues():
    """A 5 °C wobble is worth knowing about; it is not worth binning the sample."""
    seen: list[float] = []
    report, clock = _hold(
        [120.0, 126.0, 126.0, 120.0], hold_time_s=600.0,
        on_warn=lambda pv, dev, msg: seen.append(dev))

    assert seen, "excursion was not reported"
    assert report.n_warn == 1          # one excursion, not one per sample
    assert clock.t == pytest.approx(600.0)   # ran to completion


def test_a_second_excursion_warns_again():
    report, _ = _hold(
        [126.0, 120.0, 126.0, 120.0, 120.0], hold_time_s=150.0)
    assert report.n_warn == 2


def test_a_failing_warn_hook_does_not_break_the_hold():
    def boom(*_a):
        raise RuntimeError("notifier down")

    report, clock = _hold([126.0], hold_time_s=300.0, on_warn=boom)
    assert clock.t == pytest.approx(300.0)


# ── Fault band: abort a hold that is no longer annealing ─────────────────────

def test_sustained_fault_aborts_the_hold():
    """A dead or runaway heater must not silently produce a bad sample."""
    with pytest.raises(SafetyError, match="no longer being annealed"):
        _hold([120.0, 60.0], hold_time_s=7200.0)


def test_the_abort_names_the_measured_value_and_the_target():
    with pytest.raises(SafetyError) as exc:
        _hold([120.0, 60.0], hold_time_s=7200.0)
    msg = str(exc.value)
    assert "60.0" in msg and "120.0" in msg


def test_a_momentary_fault_excursion_does_not_abort():
    """A lid opened for one poll must not kill a four-hour anneal."""
    # One bad sample, then recovered — grace never elapses.
    report, clock = _hold([120.0, 60.0, 120.0, 120.0], hold_time_s=600.0)
    assert clock.t == pytest.approx(600.0)
    assert not report.aborted


def test_grace_period_must_elapse_before_aborting():
    """Deviation alone is not a fault; *sustained* deviation is."""
    # poll 30 s, grace 120 s → needs 5 consecutive bad samples.
    with pytest.raises(SafetyError):
        _hold([60.0], hold_time_s=7200.0, grace_s=120.0, poll_interval_s=30.0)

    # A shorter hold ends before the grace period elapses.
    report, _ = _hold([60.0], hold_time_s=60.0, grace_s=120.0, poll_interval_s=30.0)
    assert not report.aborted


def test_intermittent_faults_reset_the_timer_rather_than_accumulating():
    """Alternating bad/good never sustains, so it must never abort."""
    state = {"n": 0}

    def alternating():
        state["n"] += 1
        return 60.0 if state["n"] % 2 else 120.0

    report, clock = _hold(alternating, hold_time_s=600.0)

    assert not report.aborted
    assert clock.t == pytest.approx(600.0)


# ── Loss of visibility is itself a fault ─────────────────────────────────────

def test_unreadable_pv_aborts_after_the_grace_period():
    """"We cannot verify the hold" is not a safe state for an unattended run."""
    def dead():
        raise OSError("controller not responding")

    with pytest.raises(SafetyError, match="unreadable"):
        _hold(dead, hold_time_s=7200.0)


def test_a_brief_comms_blip_is_tolerated():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("transient")
        return 120.0

    report, clock = _hold(flaky, hold_time_s=300.0)
    assert clock.t == pytest.approx(300.0)
    assert not report.aborted


# ── Abort responsiveness ─────────────────────────────────────────────────────

def test_abort_interrupts_a_long_hold_promptly():
    """Previously an abort during an anneal waited out the full hold."""
    state = {"stop": False}

    def should_abort():
        return state["stop"]

    clock = _Clock()

    def read():
        if clock.t >= 90.0:
            state["stop"] = True
        return 120.0

    report = monitored_hold(
        14400.0, read_pv=read, target_C=120.0, instrument="t",
        warn_C=3.0, fault_C=10.0, grace_s=120.0, poll_interval_s=30.0,
        should_abort=should_abort, sleep=clock.sleep, now=clock.now)

    assert report.aborted
    assert clock.t < 200.0            # not the full four hours


# ── Config plumbing ──────────────────────────────────────────────────────────

def test_thresholds_come_from_safety_config():
    cfg = anneal_watchdog_config({
        "anneal_deviation_warn_C": 1.0,
        "anneal_deviation_fault_C": 4.0,
        "anneal_deviation_grace_s": 60.0,
        "anneal_poll_interval_s": 5.0,
    })
    assert cfg == {"warn_C": 1.0, "fault_C": 4.0, "grace_s": 60.0,
                   "poll_interval_s": 5.0}


def test_missing_config_falls_back_to_defaults():
    assert anneal_watchdog_config({})["fault_C"] == DEFAULT_ANNEAL_FAULT_C


def test_a_zero_length_hold_is_a_no_op():
    report, clock = _hold([120.0], hold_time_s=0.0)
    assert report.n_samples == 0
    assert clock.t == 0.0
