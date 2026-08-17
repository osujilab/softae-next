"""Watching RH through a long hold — and refusing to stop the hold for it.

The RH loop is PID-controlled and effective, so the risk through a multi-hour hot
hold is *not* that humidity wanders. It is that the PV sits away from the command
for hours with nothing broken. Measured 2026-08-11: 15 %RH commanded returned
16.9–20.4 at 65 °C and 19.5–23.2 at 85 °C. At least two explanations fit that
record — an attainable floor that rises with chamber temperature, and a flush
basin still evaporating — and the distinguishing measurement was never taken, so
the classifier grades the **observation** rather than a cause.

**The central behaviour asserted here is that no humidity verdict stops a hold.**
Not the warn state and not the fault: an elevated-temperature cure is dominated by
its thermal history, and killing an 8 h cure at 3 a.m. on a secondary variable
trades a certain loss for an uncertain one. A fault therefore neither aborts nor
raises — it alerts at ``CRITICAL`` and reaches the record via ``HoldReport.rh``.
Temperature stays blocking, and there is a test here that proves the demotion is
scoped to humidity.

Thresholds are read from ``rh_watchdog_config()`` rather than written as literals,
so a future band change cannot silently reclassify a series and leave a test
passing under a name that says otherwise.

No rig: fabricated series and ``MockRHController`` throughout, with time injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.analysis.conditions import TEMPERATURE_MIXED
from softae.analysis.rh_floor import (
    describe_floor,
    resolve_db_path,
    rh_floor_by_temperature,
)
from softae.core.alerts import (
    CRITICAL,
    WARNING,
    Alert,
    clear_alert_sinks,
    register_alert_sink,
)
from softae.core.data_store import DataStore
from softae.drivers.contracts import (
    ALERT_RH_FAULT,
    ALERT_RH_OFF_SETPOINT,
    DEFAULT_RH_FAULT_PCT,
    DEFAULT_RH_GRACE_S,
    DEFAULT_RH_WARN_PCT,
    RH_CONVERGING,
    RH_FAULT,
    RH_OFF_SETPOINT_SUSTAINED,
    RHHoldWatch,
    classify_rh_hold,
    rh_watchdog_config,
    run_anneal_hold,
    sustained_above,
    sustained_below,
)
from softae.drivers.mock_rh_controller import MockRHController
from softae.errors import SafetyError

CFG = rh_watchdog_config()
WARN = CFG["warn_pct"]
FAULT = CFG["fault_pct"]
GRACE = CFG["grace_s"]
POLL = CFG["poll_interval_s"]
SP = 15.0

#: Strictly between the two bands, so it is the warn state on either sign and
#: stays so under any future band change that keeps warn nested inside fault.
OFF_SP_PV = SP + (WARN + FAULT) / 2.0
#: Comfortably beyond the fault band on either sign.
FAULT_PV = SP + FAULT * 2.0
UNDER_FAULT_PV = SP - FAULT * 2.0

THRESHOLDS = dict(CFG)


def _series(values, *, dt: float = POLL, t0: float = 0.0):
    """A ``(t, pv)`` series sampled every *dt* seconds."""
    return [(t0 + i * dt, float(v)) for i, v in enumerate(values)]


#: Samples needed for a trailing run to span the grace window, +1 for the margin
#: `sustained_above` needs (a run of N samples spans only (N-1)·dt).
N_SUSTAINED = int(GRACE / POLL) + 2


def _classify(values, setpoint=SP, *, dt: float = POLL, **kw):
    params = dict(warn_pct=WARN, fault_pct=FAULT, grace_s=GRACE,
                  temperature_C=85.0)
    params.update(kw)
    return classify_rh_hold(_series(values, dt=dt), setpoint, **params)


class _Clock:
    """Virtual clock: ``sleep`` advances time instead of blocking."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class _TempController:
    """The minimum surface :func:`run_anneal_hold` pulls off a controller."""

    name = "temp_controller"

    def __init__(self, pv: float = 85.0) -> None:
        self._pv = pv

    def get_pv(self) -> float:
        return self._pv


@pytest.fixture(autouse=True)
def _no_sink_leakage():
    clear_alert_sinks()
    yield
    clear_alert_sinks()


@pytest.fixture
def alerts() -> list[Alert]:
    seen: list[Alert] = []
    register_alert_sink(seen.append)
    return seen


# ── Change 1: one excursion test, lifted, not a second implementation ────────

class TestSharedExcursionTest:
    """The temperature call site's expectations, re-pinned on the shared helper.

    ``tests/test_equilibration_workflow.py`` exercises the same function through
    ``watch_hold`` and is deliberately unmodified; these assert the helper itself
    behaves identically at its new home for both callers.
    """

    def test_sustained_above_needs_the_trailing_run_to_span_the_grace_window(self):
        assert sustained_above(_series([120.0] * 11), 85.0, 10.0, GRACE) is True
        assert sustained_above(_series([120.0] * 5), 85.0, 10.0, GRACE) is False

    def test_a_single_sample_can_never_span_an_interval(self):
        assert sustained_above(_series([120.0]), 85.0, 10.0, 0.0) is False

    def test_only_the_trailing_run_counts_so_a_ramp_through_the_band_is_not_a_fault(self):
        ramp = [120.0] * 20 + [85.0] + [86.0] * 20
        assert sustained_above(_series(ramp), 85.0, 10.0, GRACE) is False

    def test_an_unreadable_sample_breaks_the_run_rather_than_extending_it(self):
        broken = [120.0] * 20 + [float("nan")] + [120.0, 120.0]
        assert sustained_above(_series(broken), 85.0, 10.0, GRACE) is False

    def test_sustained_below_is_the_mirror_and_ignores_an_overshoot(self):
        assert sustained_below(_series([5.0] * 11), 15.0, 3.0, GRACE) is True
        assert sustained_below(_series([25.0] * 11), 15.0, 3.0, GRACE) is False


# ── Change 2: the thresholds resolve from [safety] ───────────────────────────

class TestWatchdogConfig:
    def test_thresholds_come_from_safety_config(self):
        cfg = rh_watchdog_config({
            "rh_deviation_warn_pct": 1.0,
            "rh_deviation_fault_pct": 4.0,
            "rh_deviation_grace_s": 60.0,
            "rh_poll_interval_s": 5.0,
        })
        assert cfg == {"warn_pct": 1.0, "fault_pct": 4.0, "grace_s": 60.0,
                       "poll_interval_s": 5.0}

    def test_missing_config_falls_back_to_defaults(self):
        assert rh_watchdog_config({})["fault_pct"] == DEFAULT_RH_FAULT_PCT

    def test_an_unparseable_value_falls_back_rather_than_raising(self):
        assert rh_watchdog_config({"rh_deviation_grace_s": "soon"})["grace_s"] == 600.0

    def test_the_shipped_grace_is_long_because_the_loop_is_slow(self):
        """A short grace produces alerts an operator learns to ignore."""
        assert rh_watchdog_config()["grace_s"] >= 300.0

    def test_the_shipped_keys_are_actually_under_safety_and_not_merely_defaulted(self):
        """The anneal keys sat under `[purge]` for weeks, unread, and because they
        held values identical to the code defaults nothing looked wrong. These
        values coincide with the defaults too, so only key presence can tell an
        editable setting from a dead one."""
        from softae.config.loader import safety

        shipped = safety()
        for key in ("rh_deviation_warn_pct", "rh_deviation_fault_pct",
                    "rh_deviation_grace_s", "rh_poll_interval_s"):
            assert key in shipped, f"{key} is not under [safety]; editing it is a no-op"


# ── Changes 3-4: three states, none of which stops an anneal ─────────────────

class TestClassification:
    def test_a_series_approaching_the_setpoint_within_grace_is_converging(self):
        verdict = _classify([45.0, 38.0, 30.0, 24.0, 19.0, 16.0, 15.2])
        assert verdict.state == RH_CONVERGING
        assert verdict.is_fault is False

    def test_a_hold_still_inside_the_band_is_converging(self):
        assert _classify([15.5, 14.8, 15.1] * 8).state == RH_CONVERGING

    def test_sustained_above_the_setpoint_with_low_variance_is_off_setpoint(self):
        """The measured case: 15 %RH commanded, 19.5-23.2 delivered at 85 C.

        Kept on its literal values because it is a record of real data. Under the
        ±5 bands this series straddles the fault band, and it is the *trailing*
        run — broken by the 19.9 near the end — that keeps it out of the fault
        state. The parametric tests below are what pin the bands themselves.
        """
        verdict = _classify([20.4, 19.5, 20.1, 19.8, 20.4, 20.0, 19.7, 20.2,
                             19.9, 20.3, 20.1])
        assert verdict.state == RH_OFF_SETPOINT_SUSTAINED

    def test_an_off_setpoint_series_is_not_a_fault(self):
        """The central behaviour: this is information, not damage."""
        assert _classify([OFF_SP_PV] * N_SUSTAINED).is_fault is False

    def test_the_off_setpoint_sentence_carries_setpoint_pv_and_temperature(self):
        """Any one of the three alone is not actionable."""
        verdict = _classify([OFF_SP_PV] * N_SUSTAINED, temperature_C=85.0)
        detail = verdict.describe()
        assert verdict.state == RH_OFF_SETPOINT_SUSTAINED, "must test the warn branch"
        assert f"{SP:.1f}" in detail
        assert f"{OFF_SP_PV:.1f}" in detail
        assert "85.0" in detail

    def test_the_off_setpoint_sentence_names_the_side_the_pv_sat_on(self):
        """Above and below the command are different findings to an operator even
        when they are one state to the classifier."""
        assert "above" in _classify([OFF_SP_PV] * N_SUSTAINED).describe()
        assert "below" in _classify([SP - (WARN + FAULT) / 2.0]
                                    * N_SUSTAINED).describe()

    def test_an_excursion_beyond_the_fault_band_is_a_fault(self):
        verdict = _classify([FAULT_PV] * N_SUSTAINED)
        assert verdict.state == RH_FAULT
        assert verdict.is_fault is True

    def test_sustained_below_beyond_the_fault_band_is_a_fault(self):
        """Symmetric with the overshoot: the reason cites the fault band and makes
        no claim about a mechanism."""
        verdict = _classify([UNDER_FAULT_PV] * N_SUSTAINED)
        assert verdict.state == RH_FAULT
        assert "BELOW" in verdict.reason
        assert f"{FAULT:g}" in verdict.reason
        assert "basin" not in verdict.reason

    def test_a_transient_crossing_of_the_band_does_not_trip(self):
        """The ramp hazard, re-pinned on the humidity side."""
        assert _classify([40.0] * 20 + [15.1, 15.0, 14.9]).state == RH_CONVERGING

    def test_a_ramp_that_has_only_just_saturated_is_still_converging(self):
        """Sustained means sustained: a few samples is not ten minutes."""
        short = [SP + FAULT * 4, SP + FAULT * 3, SP + FAULT * 2, OFF_SP_PV,
                 OFF_SP_PV]
        assert len(short) * POLL < GRACE, "the run must be too short to sustain"
        assert _classify(short).state == RH_CONVERGING

    def test_no_readable_pv_at_all_is_a_fault(self):
        verdict = _classify([float("nan")] * N_SUSTAINED)
        assert verdict.state == RH_FAULT
        assert verdict.n_samples == 0

    def test_an_empty_window_is_a_fault_rather_than_a_silent_pass(self):
        assert classify_rh_hold([], SP).state == RH_FAULT

    def test_the_config_resolver_output_is_accepted_verbatim(self):
        """`rh_watchdog_config()` must be splattable into the classifier."""
        verdict = classify_rh_hold(_series([OFF_SP_PV] * N_SUSTAINED), SP,
                                   **rh_watchdog_config())
        assert verdict.state == RH_OFF_SETPOINT_SUSTAINED


# ── Change 8: the bands are flat +/-5 %RH, symmetric and nested ──────────────

class TestSymmetricBands:
    """Over- and undershoot are graded identically. Before Change 8 an undershoot
    faulted at the *warn* band, so these signs were not comparable at all."""

    def test_overshoot_beyond_the_fault_band_is_a_fault(self):
        assert _classify([SP + FAULT + 1.0] * N_SUSTAINED).state == RH_FAULT

    def test_undershoot_beyond_the_fault_band_is_a_fault(self):
        assert _classify([SP - FAULT - 1.0] * N_SUSTAINED).state == RH_FAULT

    def test_undershoot_between_warn_and_fault_warns_rather_than_faults(self):
        """The branch that did not exist before Change 8: below `warn_pct` used to
        return RH_FAULT outright, pre-empting the position a below-warn warning
        would occupy."""
        verdict = _classify([SP - (WARN + FAULT) / 2.0] * N_SUSTAINED)
        assert verdict.state == RH_OFF_SETPOINT_SUSTAINED
        assert verdict.state != RH_FAULT

    @pytest.mark.parametrize("deviation", [WARN / 2.0, (WARN + FAULT) / 2.0,
                                           FAULT + 1.0])
    def test_the_bands_are_symmetric_in_both_directions(self, deviation):
        """One test that fails on any future reintroduction of an asymmetry."""
        above = _classify([SP + deviation] * N_SUSTAINED)
        below = _classify([SP - deviation] * N_SUSTAINED)
        assert above.state == below.state

    def test_the_warn_band_is_inside_the_fault_band(self):
        """A real invariant only since Change 8: `warn_pct` used to double as the
        undershoot fault band, so the two were not ordered on that sign."""
        cfg = rh_watchdog_config()
        assert cfg["warn_pct"] < cfg["fault_pct"]

    def test_the_rh_defaults_match_the_shipped_config(self):
        """Two spellings of one threshold is a defect. `rh_watchdog_config`
        swallows a config-load exception and proceeds with `{}`, so a disagreement
        means a config-load failure silently restores a superseded band — exactly
        the live defect on the anneal axis (`DEFAULT_ANNEAL_WARN_C`/`_FAULT_C` vs
        `[safety]`). Key *presence* is checked separately; this checks the values.
        """
        from softae.config.loader import safety

        shipped = safety()
        assert shipped["rh_deviation_warn_pct"] == DEFAULT_RH_WARN_PCT
        assert shipped["rh_deviation_fault_pct"] == DEFAULT_RH_FAULT_PCT
        assert shipped["rh_deviation_grace_s"] == DEFAULT_RH_GRACE_S

    def test_measured_ripple_trips_neither_band(self):
        """Measured per-round `rh_pv_pct` spread over the 2026-08-11 run: median
        1.42 %RH, p90 2.16, max 2.99. At the p90 spread, centred on the command,
        the peak excursion is ~1.07 %RH — well inside the warn band, let alone the
        fault band. A band change has to re-argue this against data.
        """
        ripple = [SP + 1.07 if i % 2 else SP - 1.07
                  for i in range(int(2 * GRACE / POLL))]
        assert _classify(ripple).state == RH_CONVERGING

    def test_a_slow_approach_reaches_the_fault_band(self):
        """The expected consequence of the tightening, made visible rather than
        left as a surprise. The 85 C block descends 22.3 -> 14.9 %RH at about
        0.0015 %RH/s against a commanded 15. At the old +/-10 band the PV never
        got beyond 25 and so never faulted; at +/-5 it spends ~1500 s beyond 20,
        which outlasts the 600 s grace.

        Read beside `test_a_humidity_fault_does_not_park_the_hold`, the pair is one
        statement: this now faults, and faulting now costs nothing.
        """
        start, rate = 22.3, 0.0015          # %RH, %RH/s

        def descent_while_beyond(band: float) -> list[float]:
            out: list[float] = []
            for i in range(500):
                pv = start - rate * (i * POLL)
                if pv <= SP + band:
                    break
                out.append(pv)
            return out

        beyond_fault = descent_while_beyond(FAULT)
        assert (len(beyond_fault) - 1) * POLL >= GRACE, "the approach outlasts grace"
        assert _classify(beyond_fault).state == RH_FAULT

        # And the same descent under the superseded 10.0 band: never even close.
        assert descent_while_beyond(10.0) == []


def _drive(watch: RHHoldWatch, clock: _Clock, pv_seq) -> None:
    """Feed *pv_seq* through *watch*, one sample per poll interval."""
    for _ in pv_seq:
        watch.sample()
        clock.sleep(POLL)


class _StepReader:
    """A `get_TH`-shaped reader that returns each %RH in turn, then holds the last."""

    def __init__(self, values) -> None:
        self._values = list(values)
        self._i = 0

    def __call__(self) -> tuple[float, float]:
        pv = self._values[min(self._i, len(self._values) - 1)]
        self._i += 1
        return 85.0, pv


class TestAlerting:
    def test_off_setpoint_raises_exactly_one_alert_carrying_the_three_numbers(
            self, alerts):
        clock = _Clock()
        watch = RHHoldWatch(lambda: (85.0, OFF_SP_PV), SP, thresholds=THRESHOLDS,
                            now=clock.now)
        _drive(watch, clock, range(N_SUSTAINED))

        assert watch.verdict.state == RH_OFF_SETPOINT_SUSTAINED
        assert len(alerts) == 1, "a verdict true for hours is one finding, not many"
        assert alerts[0].kind == ALERT_RH_OFF_SETPOINT
        assert alerts[0].severity == WARNING
        details = alerts[0].details
        assert details["rh_setpoint_pct"] == pytest.approx(SP)
        assert details["rh_pv_pct"] == pytest.approx(OFF_SP_PV)
        assert details["temperature_C"] == pytest.approx(85.0)

    def test_a_fault_raises_exactly_one_critical_alert_carrying_the_three_numbers(
            self, alerts):
        """Nothing parks on an RH fault any more, so if this alert does not fire an
        8 h cure at a badly wrong humidity runs to completion and reports clean."""
        clock = _Clock()
        watch = RHHoldWatch(lambda: (85.0, FAULT_PV), SP, thresholds=THRESHOLDS,
                            now=clock.now)
        _drive(watch, clock, range(N_SUSTAINED))

        assert watch.verdict.state == RH_FAULT
        assert len(alerts) == 1
        assert alerts[0].kind == ALERT_RH_FAULT
        assert alerts[0].severity == CRITICAL
        details = alerts[0].details
        assert details["rh_setpoint_pct"] == pytest.approx(SP)
        assert details["rh_pv_pct"] == pytest.approx(FAULT_PV)
        assert details["temperature_C"] == pytest.approx(85.0)

    def test_a_hold_that_degrades_from_off_setpoint_to_fault_raises_both(self, alerts):
        """The throttle is per state, not per hold. With a single bool the fault
        alert is swallowed by the earlier warn — the silent 8 h cure this change
        exists to prevent, reintroduced by the throttle."""
        clock = _Clock()
        reader = _StepReader([OFF_SP_PV] * N_SUSTAINED + [FAULT_PV] * N_SUSTAINED)
        watch = RHHoldWatch(reader, SP, thresholds=THRESHOLDS, now=clock.now)
        _drive(watch, clock, range(2 * N_SUSTAINED))

        assert [(a.kind, a.severity) for a in alerts] == [
            (ALERT_RH_OFF_SETPOINT, WARNING),
            (ALERT_RH_FAULT, CRITICAL),
        ]

    def test_a_converging_hold_never_alerts(self, alerts):
        clock = _Clock()
        watch = RHHoldWatch(lambda: (85.0, SP + 0.1), SP, thresholds=THRESHOLDS,
                            now=clock.now)
        _drive(watch, clock, range(N_SUSTAINED))
        assert alerts == []

    def test_the_alert_is_persisted_so_the_reason_outlives_the_process(self, tmp_path):
        clock = _Clock()
        with DataStore(tmp_path / "proj") as store:
            watch = RHHoldWatch(lambda: (85.0, OFF_SP_PV), SP, thresholds=THRESHOLDS,
                                data_store=store, run_id="r1", now=clock.now)
            _drive(watch, clock, range(N_SUSTAINED))
            rows = store.query_alerts(run_id="r1")
        assert len(rows) == 1
        assert rows[0]["kind"] == ALERT_RH_OFF_SETPOINT

    def test_the_fault_alert_is_persisted_so_the_reason_outlives_the_process(
            self, tmp_path):
        """The alert row is the durable half. A *thermal* fault mid-hold raises out
        of `monitored_hold` before any `HoldReport` exists, so on that path the row
        is the only surviving evidence that humidity was also wrong."""
        clock = _Clock()
        with DataStore(tmp_path / "proj") as store:
            watch = RHHoldWatch(lambda: (85.0, FAULT_PV), SP, thresholds=THRESHOLDS,
                                data_store=store, run_id="r1", now=clock.now)
            _drive(watch, clock, range(N_SUSTAINED))
            rows = store.query_alerts(run_id="r1")
        assert len(rows) == 1
        assert rows[0]["kind"] == ALERT_RH_FAULT
        assert rows[0]["severity"] == CRITICAL

    def test_the_persisted_alert_kind_value_is_frozen(self):
        """`alerts.kind` is persisted. Change 7 renamed the constant from
        ALERT_RH_FLOOR_LIMITED to ALERT_RH_OFF_SETPOINT; aligning the *value* with
        the new identifier would silently orphan every historical row from any
        query written afterwards. The next tidy-up must trip over this."""
        assert ALERT_RH_OFF_SETPOINT == "rh_floor_limited"
        assert ALERT_RH_FAULT == "rh_fault"

    def test_the_sample_reads_humidity_from_the_mock_controllers_get_TH(self):
        """`_read_rh` must accept the shipped driver's own return shape."""
        rh = MockRHController()
        clock = _Clock()
        watch = RHHoldWatch(rh.get_TH, SP, thresholds=THRESHOLDS, now=clock.now)
        watch.sample()
        assert watch.series and watch.series[0][1] == pytest.approx(rh.get_H(), abs=5.0)

    def test_a_failing_rh_read_is_not_reported_as_a_dead_thermocouple(self):
        def boom():
            raise OSError("sensor bus down")

        watch = RHHoldWatch(boom, SP, thresholds=THRESHOLDS, now=_Clock().now)
        wrapped = watch.wrap_reader(lambda: 85.0)
        assert wrapped() == 85.0            # the temperature read still succeeds


class TestPollThrottling:
    def test_samples_are_not_taken_faster_than_the_configured_interval(self):
        clock = _Clock()
        watch = RHHoldWatch(lambda: (85.0, 20.0), SP, thresholds=THRESHOLDS,
                            now=clock.now)
        wrapped = watch.wrap_reader(lambda: 85.0)
        for _ in range(6):                  # six temperature polls, 30 s apart
            clock.sleep(30.0)
            wrapped()
        assert len(watch.series) == 3       # 60 s RH cadence, not 30 s


# ── The hold continues regardless of the verdict; the verdict is reported ────

class TestRunAnnealHold:
    """`run_anneal_hold` is the anneal's polling site and now watches both axes.

    Humidity is announced and recorded here, never acted on. Both stop routes are
    gone together, and deliberately so: removing only the raise would leave a hold
    that truncated at the fault and returned a clean-looking `HoldReport` — a cure
    that stopped at hour 3 of 8 and reported success, strictly worse than nothing.
    """

    @staticmethod
    def _run(rh_pv: float, *, hold_time_s: float = 7200.0, temp_pv: float = 85.0):
        clock = _Clock()
        return run_anneal_hold(
            _TempController(temp_pv), hold_time_s, 85.0,
            rh_reader=lambda: (85.0, rh_pv), rh_setpoint_pct=SP,
            sleep=clock.sleep, now=clock.now,
        ), clock

    def test_an_off_setpoint_hold_runs_to_completion(self, alerts):
        """An 8 h cure is not parked because the humidity sat off command."""
        report, clock = self._run(OFF_SP_PV)

        assert clock.t == pytest.approx(7200.0)      # the whole hold, not a park
        assert report.aborted is False
        assert len(alerts) == 1
        assert alerts[0].kind == ALERT_RH_OFF_SETPOINT
        assert report.rh.state == RH_OFF_SETPOINT_SUSTAINED

    def test_a_humidity_fault_does_not_park_the_hold(self, alerts):
        """THE assertion. A fault neither raises nor aborts; it alerts and is
        carried out on the report."""
        report, clock = self._run(FAULT_PV)

        assert clock.t == pytest.approx(7200.0)
        assert report.aborted is False
        assert report.rh is not None and report.rh.state == RH_FAULT
        assert [(a.kind, a.severity) for a in alerts] == [
            (ALERT_RH_FAULT, CRITICAL)]

    def test_a_humidity_fault_waits_out_the_whole_hold_rather_than_stopping_early(
            self):
        """Inverted by the demotion: this used to assert the hold stopped early.
        A cure that truncates at hour 3 of 8 is the failure, not the safeguard."""
        report, clock = self._run(FAULT_PV, hold_time_s=28800.0)
        assert clock.t == pytest.approx(28800.0)
        assert report.aborted is False

    def test_temperature_stays_blocking_even_when_humidity_also_faults(self, alerts):
        """The test that proves the demotion is scoped to humidity: same hold, one
        axis demoted and one not. The raise is the *thermal* message."""
        with pytest.raises(SafetyError, match="no longer being annealed"):
            self._run(FAULT_PV, temp_pv=140.0)

    def test_an_rh_alert_already_raised_survives_a_later_thermal_park(self, alerts):
        """The alert row is the durable half of the record, and on this path it is
        the *only* half: a thermal fault raises out of `monitored_hold` before any
        `HoldReport` exists, so `report.rh` never reaches the caller.

        The two graces are far apart — thermal 120 s, RH 600 s — so a heater that
        dies at the same moment humidity goes wrong parks before the RH verdict can
        even form. The RH alert survives only when it was raised first, which is
        what this drives.
        """
        clock = _Clock()
        # Good temperature until the RH fault has been announced, then a dead heater.
        controller = _TempController(85.0)
        reads = {"n": 0}

        def get_pv() -> float:
            reads["n"] += 1
            return 85.0 if reads["n"] <= 2 * N_SUSTAINED else 140.0

        controller.get_pv = get_pv

        with pytest.raises(SafetyError, match="no longer being annealed"):
            run_anneal_hold(
                controller, 28800.0, 85.0,
                rh_reader=lambda: (85.0, FAULT_PV), rh_setpoint_pct=SP,
                sleep=clock.sleep, now=clock.now,
            )

        assert [a.kind for a in alerts] == [ALERT_RH_FAULT]

    def test_a_hold_at_condition_is_untouched(self, alerts):
        report, clock = self._run(SP)
        assert clock.t == pytest.approx(7200.0)
        assert alerts == []
        assert report.rh.state == RH_CONVERGING

    def test_with_no_rh_reader_the_hold_is_exactly_the_thermal_one(self, alerts):
        """The RH watch is opt-in; nothing changes for a caller that does not ask."""
        clock = _Clock()
        report = run_anneal_hold(_TempController(85.0), 3600.0, 85.0,
                                 sleep=clock.sleep, now=clock.now)
        assert clock.t == pytest.approx(3600.0)
        assert report.n_samples > 0
        assert report.rh is None, "no reader, no verdict — not a manufactured one"
        assert alerts == []


# ── Change 5: the floor is measurable from data already on disk ──────────────

def _eis_result(channel: int = 1):
    """The minimum a measurement row needs; the spectrum itself is irrelevant here."""
    from datetime import datetime

    import numpy as np

    from softae.analysis.eis_data import EISResult

    n = 4
    return EISResult(
        channel=channel, frequency=np.logspace(2, 5, n),
        z_magnitude=np.full(n, 1000.0), phase=np.full(n, -10.0),
        z_real=np.full(n, 980.0), z_imag_neg=np.full(n, 170.0),
        timestamp=datetime(2026, 8, 11, 2, 37, 57), measurement_time_s=1.0,
        eis_params={"npts": n},
    )


def _record_condition(store: DataStore, run_id: str, rh_sp, rh_pv, **temps):
    mid = store.record_measurement(run_id, _eis_result())
    store.record_conditions(mid, "anneal", rh_sp_pct=rh_sp, rh_pv_pct=rh_pv, **temps)


@pytest.fixture
def floor_store(tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    run_id = store.start_run("rh_floor")
    # 65 C: asked for 15, delivered 16.9-20.4 -> saturated, the floor is real.
    for pv in (16.9, 18.2, 20.4):
        _record_condition(store, run_id, 15.0, pv, stage_temp_pv_C=65.0)
    # 85 C: asked for 15, delivered 19.5-23.2.
    for pv in (19.5, 21.0, 23.2):
        _record_condition(store, run_id, 15.0, pv, stage_temp_pv_C=85.0)
    # 27.5 C: asked for 40 and got it -> the floor here was never probed.
    for pv in (40.1, 39.8):
        _record_condition(store, run_id, 40.0, pv, stage_temp_pv_C=27.5)
    store.run_id = run_id
    yield store
    store.close()


class TestFloorReporter:
    def test_the_floor_is_the_minimum_pv_in_each_temperature_bin(self, floor_store):
        bins = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir)}
        assert bins[70.0].rh_floor_pct == pytest.approx(16.9)
        assert bins[90.0].rh_floor_pct == pytest.approx(19.5)

    def test_bins_are_returned_coldest_first(self, floor_store):
        temps = [b.temperature_C for b in
                 rh_floor_by_temperature(floor_store.project_dir)]
        assert temps == sorted(temps)

    def test_a_temperature_with_no_rows_is_absent_rather_than_zero(self, floor_store):
        """An unvisited temperature has no floor; a zero would read as bone dry."""
        temps = {b.temperature_C for b in
                 rh_floor_by_temperature(floor_store.project_dir)}
        assert 50.0 not in temps and 0.0 not in temps

    def test_a_bin_asked_for_less_than_it_delivered_is_marked_saturated(self,
                                                                       floor_store):
        bins = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir)}
        assert bins[70.0].saturated is True
        assert bins[90.0].saturated is True

    def test_a_bin_that_met_its_setpoint_is_not_a_measured_floor(self, floor_store):
        """MIN(rh_pv) where nobody asked for less says what was asked, not the limit."""
        bins = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir)}
        assert bins[30.0].saturated is False
        assert "not probed" in bins[30.0].describe()

    def test_row_counts_are_carried_so_a_one_sample_bin_is_visible(self, floor_store):
        bins = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir)}
        assert bins[70.0].n_rows == 3

    def test_each_bin_names_the_thermometer_it_was_binned_on(self, floor_store):
        bins = rh_floor_by_temperature(floor_store.project_dir)
        assert all(b.temperature_source == "stage_pv" for b in bins)

    def test_a_bin_fed_by_two_thermometers_is_labelled_mixed(self, floor_store):
        """The defect `analysis.conditions` warns about must stay visible."""
        _record_condition(floor_store, floor_store.run_id, 15.0, 17.5,
                          chamber_air_C=66.0)
        bins = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir)}
        assert bins[70.0].temperature_source == TEMPERATURE_MIXED

    def test_the_query_can_be_scoped_to_one_run(self, floor_store):
        assert rh_floor_by_temperature(floor_store.project_dir, run_id="nope") == []

    def test_the_bin_width_is_configurable(self, floor_store):
        wide = {b.temperature_C: b for b in
                rh_floor_by_temperature(floor_store.project_dir, bin_width_C=100.0)}
        # 65 C and 85 C now share one bin; its floor is the lower of the two.
        assert wide[100.0].n_rows == 6
        assert wide[100.0].rh_floor_pct == pytest.approx(16.9)

    def test_a_non_positive_bin_width_is_refused(self, floor_store):
        with pytest.raises(ValueError, match="bin_width_C"):
            rh_floor_by_temperature(floor_store.project_dir, bin_width_C=0.0)

    def test_an_empty_project_reports_absence_rather_than_a_floor_of_zero(self,
                                                                         tmp_path):
        with DataStore(tmp_path / "empty") as store:
            assert rh_floor_by_temperature(store.project_dir) == []
        assert "never been observed" in describe_floor([])

    def test_the_summary_names_the_worst_probed_floor(self, floor_store):
        text = describe_floor(rh_floor_by_temperature(floor_store.project_dir))
        assert "19.5 %RH at 90 C" in text

    def test_the_reporter_never_writes(self, floor_store):
        """`mode=ro` makes SQLite itself refuse; nothing to audit."""
        import sqlite3

        from softae.analysis.rh_floor import _connect_ro

        conn = _connect_ro(resolve_db_path(floor_store.project_dir))
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO conditions (measurement_id, run_id, stage, "
                             "timestamp) VALUES (1, 'run1', 's', 't')")
        finally:
            conn.close()

    def test_a_project_dir_and_a_db_file_resolve_to_the_same_database(self,
                                                                     floor_store):
        direct = rh_floor_by_temperature(
            resolve_db_path(floor_store.project_dir))
        assert direct == rh_floor_by_temperature(floor_store.project_dir)
