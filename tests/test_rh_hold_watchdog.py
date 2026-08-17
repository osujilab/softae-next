"""Watching RH through a long hold: saturation, not drift.

The RH loop is PID-controlled and effective, so the risk through a multi-hour hot
hold is *not* that humidity wanders. It is that the **attainable RH floor rises
with chamber temperature** — the flush basin holds water inside the heated
enclosure — so a setpoint below that floor saturates the controller and the PV
sits above the command indefinitely with nothing broken. Measured 2026-08-11:
15 %RH commanded returned 16.9–20.4 at 65 °C and 19.5–23.2 at 85 °C.

The central behaviour asserted here is that this state **does not park a run**.
Parking an 8 h unattended cure at 3 a.m. on a plumbing fact is the failure mode
the autonomy work exists to prevent.

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
from softae.core.alerts import Alert, clear_alert_sinks, register_alert_sink
from softae.core.data_store import DataStore
from softae.drivers.contracts import (
    ALERT_RH_FLOOR_LIMITED,
    DEFAULT_RH_FAULT_PCT,
    RH_CONVERGING,
    RH_FAULT,
    RH_FLOOR_LIMITED,
    RHHoldWatch,
    classify_rh_hold,
    rh_watchdog_config,
    run_anneal_hold,
    sustained_above,
    sustained_below,
)
from softae.drivers.mock_rh_controller import MockRHController
from softae.errors import SafetyError

WARN = 3.0
FAULT = 10.0
GRACE = 600.0
SP = 15.0

THRESHOLDS = {"warn_pct": WARN, "fault_pct": FAULT, "grace_s": GRACE,
              "poll_interval_s": 60.0}


def _series(values, *, dt: float = 60.0, t0: float = 0.0):
    """A ``(t, pv)`` series sampled every *dt* seconds."""
    return [(t0 + i * dt, float(v)) for i, v in enumerate(values)]


def _classify(values, setpoint=SP, *, dt: float = 60.0, **kw):
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


# ── Changes 3-4: three states, and only one of them parks ────────────────────

class TestClassification:
    def test_a_series_approaching_the_setpoint_within_grace_is_converging(self):
        verdict = _classify([45.0, 38.0, 30.0, 24.0, 19.0, 16.0, 15.2])
        assert verdict.state == RH_CONVERGING
        assert verdict.parks is False

    def test_a_hold_still_inside_the_band_is_converging(self):
        assert _classify([15.5, 14.8, 15.1] * 8).state == RH_CONVERGING

    def test_sustained_above_the_setpoint_with_low_variance_is_floor_limited(self):
        """The measured case: 15 %RH commanded, 19.5-23.2 delivered at 85 C."""
        verdict = _classify([20.4, 19.5, 20.1, 19.8, 20.4, 20.0, 19.7, 20.2,
                             19.9, 20.3, 20.1])
        assert verdict.state == RH_FLOOR_LIMITED

    def test_floor_limited_does_not_park_the_run(self):
        """The central behaviour: this is information, not damage."""
        verdict = _classify([20.0] * 11)
        assert verdict.parks is False

    def test_the_floor_limited_sentence_carries_setpoint_pv_and_temperature(self):
        """Any one of the three alone is not actionable."""
        verdict = _classify([20.4] * 11, temperature_C=85.0)
        detail = verdict.describe()
        assert "15.0" in detail and "20.4" in detail and "85.0" in detail

    def test_an_excursion_beyond_the_fault_band_is_a_fault(self):
        verdict = _classify([70.0] * 11)
        assert verdict.state == RH_FAULT
        assert verdict.parks is True

    def test_sustained_below_the_setpoint_is_a_fault_not_floor_limited(self):
        """The basin can only push humidity UP, so an undershoot is not that effect."""
        verdict = _classify([5.0] * 11)
        assert verdict.state == RH_FAULT
        assert "BELOW" in verdict.reason

    def test_a_transient_crossing_of_the_band_does_not_trip(self):
        """The ramp hazard, re-pinned on the humidity side."""
        assert _classify([40.0] * 20 + [15.1, 15.0, 14.9]).state == RH_CONVERGING

    def test_a_ramp_that_has_only_just_saturated_is_still_converging(self):
        """Sustained means sustained: three samples is not ten minutes."""
        assert _classify([40.0, 30.0, 22.0, 20.1, 20.0]).state == RH_CONVERGING

    def test_no_readable_pv_at_all_is_a_fault(self):
        verdict = _classify([float("nan")] * 11)
        assert verdict.state == RH_FAULT
        assert verdict.n_samples == 0

    def test_an_empty_window_is_a_fault_rather_than_a_silent_pass(self):
        assert classify_rh_hold([], SP).state == RH_FAULT

    def test_the_config_resolver_output_is_accepted_verbatim(self):
        """`rh_watchdog_config()` must be splattable into the classifier."""
        verdict = classify_rh_hold(_series([20.0] * 11), SP, **rh_watchdog_config())
        assert verdict.state == RH_FLOOR_LIMITED


class TestAlerting:
    def test_floor_limited_raises_exactly_one_alert_carrying_the_three_numbers(
            self, alerts):
        watch = RHHoldWatch(lambda: (85.0, 20.4), SP, thresholds=THRESHOLDS,
                            now=_Clock().now)
        # A fresh timestamp per sample, so the trailing run spans the grace window.
        clock = _Clock()
        watch._now = clock.now
        for _ in range(12):
            watch.sample()
            clock.sleep(60.0)

        assert watch.verdict.state == RH_FLOOR_LIMITED
        assert len(alerts) == 1, "a verdict true for hours is one finding, not many"
        details = alerts[0].details
        assert details["rh_setpoint_pct"] == pytest.approx(SP)
        assert details["rh_pv_pct"] == pytest.approx(20.4)
        assert details["temperature_C"] == pytest.approx(85.0)

    def test_a_converging_hold_never_alerts(self, alerts):
        clock = _Clock()
        watch = RHHoldWatch(lambda: (85.0, 15.1), SP, thresholds=THRESHOLDS,
                            now=clock.now)
        for _ in range(12):
            watch.sample()
            clock.sleep(60.0)
        assert alerts == []

    def test_the_alert_is_persisted_so_the_reason_outlives_the_process(self, tmp_path):
        clock = _Clock()
        with DataStore(tmp_path / "proj") as store:
            watch = RHHoldWatch(lambda: (85.0, 20.4), SP, thresholds=THRESHOLDS,
                                data_store=store, run_id="r1", now=clock.now)
            for _ in range(12):
                watch.sample()
                clock.sleep(60.0)
            rows = store.query_alerts(run_id="r1")
        assert len(rows) == 1
        assert rows[0]["kind"] == ALERT_RH_FLOOR_LIMITED

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


# ── The hold itself continues, or parks, per the verdict ─────────────────────

class TestRunAnnealHold:
    """`run_anneal_hold` is the anneal's polling site and now watches both axes."""

    @staticmethod
    def _run(rh_pv: float, *, hold_time_s: float = 7200.0):
        clock = _Clock()
        return run_anneal_hold(
            _TempController(85.0), hold_time_s, 85.0,
            rh_reader=lambda: (85.0, rh_pv), rh_setpoint_pct=SP,
            sleep=clock.sleep, now=clock.now,
        ), clock

    def test_a_floor_limited_hold_runs_to_completion(self, alerts):
        """THE assertion: an 8 h cure is not parked on a plumbing fact."""
        cfg = rh_watchdog_config()
        floor_pv = SP + (cfg["warn_pct"] + cfg["fault_pct"]) / 2.0

        report, clock = self._run(floor_pv)

        assert clock.t == pytest.approx(7200.0)      # the whole hold, not a park
        assert report.aborted is False
        assert len(alerts) == 1
        assert alerts[0].kind == ALERT_RH_FLOOR_LIMITED

    def test_a_humidity_fault_parks_the_hold(self):
        cfg = rh_watchdog_config()
        with pytest.raises(SafetyError, match="humidity"):
            self._run(SP + cfg["fault_pct"] * 3.0)

    def test_a_humidity_fault_stops_early_rather_than_waiting_out_the_hold(self):
        cfg = rh_watchdog_config()
        clock = _Clock()
        with pytest.raises(SafetyError):
            run_anneal_hold(
                _TempController(85.0), 28800.0, 85.0,
                rh_reader=lambda: (85.0, SP + cfg["fault_pct"] * 3.0),
                rh_setpoint_pct=SP, sleep=clock.sleep, now=clock.now,
            )
        assert clock.t < 28800.0

    def test_a_hold_at_condition_is_untouched(self, alerts):
        report, clock = self._run(SP)
        assert clock.t == pytest.approx(7200.0)
        assert alerts == []

    def test_with_no_rh_reader_the_hold_is_exactly_the_thermal_one(self, alerts):
        """The RH watch is opt-in; nothing changes for a caller that does not ask."""
        clock = _Clock()
        report = run_anneal_hold(_TempController(85.0), 3600.0, 85.0,
                                 sleep=clock.sleep, now=clock.now)
        assert clock.t == pytest.approx(3600.0)
        assert report.n_samples > 0
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
