"""The P.22 run: the watched hold on both axes, the naming, and the projection.

Time is injected throughout, so a nine-hour run is exercised instantly. Nothing
here touches an instrument — the fakes below are the whole hardware surface this
module uses.

Three traps are pinned here because each of them reads as a success when it is
not: ``HoldReport.aborted`` means *reached* on the approach path; the hold report
is **discarded** on the ``SafetyError`` path; and a repeated point collides
silently one layer below the executor's loud duplicate-name check.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from softae.analysis.equilibration import (
    MIN_POINTS_FOR_TAU,
    SETTLE_CEILING,
    SETTLE_DISABLED,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    RoundFit,
)
from softae.workflows.equilibration import (
    ENV_OK,
    ENV_SKIPPED,
    EV_AMBIENT_RESTORED,
    EV_CHANNEL_MEASURED,
    EV_COST_WARNING,
    EV_HEARTBEAT,
    EV_LEG_FINISHED,
    EV_LEG_STARTED,
    EV_ROUND_FINISHED,
    EV_ROUND_STARTED,
    EV_RUN_FINISHED,
    EV_RUN_STARTED,
    EV_SETPOINT_FINISHED,
    EV_SETPOINT_STARTED,
    EV_SETTLE_VERDICT,
    KIND_SERIES,
    VERDICT_MET,
    VERDICT_UNMET,
    EquilibrationAbort,
    EquilibrationConfig,
    EquilibrationRun,
    approach_setpoint,
    build_round_workflow,
    inter_round_gap_s,
    measurement_step_name,
    project_duration,
    round_headroom_s_per_channel,
    watch_hold,
)


class _Clock:
    """Virtual clock: ``sleep`` advances time instead of blocking."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


def _reader(values):
    """A PV reader driven by *values*; the last value repeats forever.

    An entry of ``None`` raises, standing in for an unreadable sensor.
    """
    remaining = list(values)

    def _read():
        value = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if value is None:
            raise RuntimeError("sensor unreadable")
        return value

    return _read


# ── The approach ─────────────────────────────────────────────────────────────

class TestApproach:
    def test_the_approach_phase_reports_reached_from_the_final_sample_not_from_the_aborted_flag(self):
        clock = _Clock()
        outcome = approach_setpoint(
            _reader([50.0, 60.0, 70.0, 85.0]), 85.0, axis="temperature",
            instrument="temp_controller", tolerance=0.5, timeout_s=1800.0,
            poll_interval_s=30.0, sleep=clock.sleep, now=clock.now)

        assert outcome.reached is True
        assert outcome.pv_final == pytest.approx(85.0)
        # It exited early rather than burning the whole timeout.
        assert outcome.elapsed_s < 1800.0

    def test_an_approach_that_times_out_is_not_reported_as_reached(self):
        # `monitored_hold` returns aborted=False on a genuine timeout and
        # aborted=True on the band-entry exit. Read from the flag, this is the
        # case that would come back as a success.
        clock = _Clock()
        outcome = approach_setpoint(
            _reader([50.0]), 85.0, axis="temperature", instrument="temp_controller",
            tolerance=0.5, timeout_s=600.0, poll_interval_s=30.0,
            sleep=clock.sleep, now=clock.now)

        assert outcome.reached is False
        assert outcome.elapsed_s == pytest.approx(600.0)
        assert outcome.n_samples == 600 / 30

    def test_the_approach_never_faults_on_a_legitimate_ramp(self):
        # A watched hold started right after write_sp would grade the ramp as a
        # fault and abort after grace_s. The approach uses a non-binding band.
        clock = _Clock()
        ramp = [25.0 + 2.0 * i for i in range(40)]
        outcome = approach_setpoint(
            _reader(ramp), 85.0, axis="temperature", instrument="temp_controller",
            tolerance=1.0, timeout_s=3600.0, poll_interval_s=30.0,
            sleep=clock.sleep, now=clock.now)
        assert outcome.reached is True

    def test_an_unreadable_approach_is_not_reached_rather_than_silently_fine(self):
        clock = _Clock()
        outcome = approach_setpoint(
            _reader([None]), 85.0, axis="temperature", instrument="temp_controller",
            tolerance=0.5, timeout_s=300.0, poll_interval_s=30.0,
            sleep=clock.sleep, now=clock.now)
        assert outcome.reached is False


# ── The watched hold, on both axes ───────────────────────────────────────────

def _hold(values, target, **kw):
    clock = _Clock()
    params = dict(
        hold_time_s=600.0, axis="temperature", instrument="temp_controller",
        tolerance=0.5, warn=3.0, fault=10.0, grace_s=120.0, poll_interval_s=30.0,
        sleep=clock.sleep, now=clock.now,
    )
    params.update(kw)
    return watch_hold(_reader(values), target, **params), clock


class TestWatchedHold:
    def test_the_watched_hold_grades_humidity_when_given_the_rh_reader(self):
        # `monitored_hold` is reader-agnostic in its logic: it calls read_pv(),
        # coerces to float and grades abs(pv - target). Passing rh.get_H makes it
        # a humidity watchdog with NO driver edit at all.
        outcome, _ = _hold([15.2, 14.9, 15.1], 15.0, axis="humidity",
                           instrument="rh_controller", tolerance=2.0, warn=5.0,
                           fault=15.0)
        assert outcome.met is True
        assert outcome.axis == "humidity"
        assert outcome.n_samples == 600 / 30

    def test_a_humidity_fault_message_reaches_the_operator_rendered_by_this_module_not_as_an_anneal(self):
        # The chamber cannot reach a 40 %RH setpoint and sits at 20 %. An
        # UNDERshoot, so it is recorded rather than aborted.
        outcome, _ = _hold([20.0], 40.0, axis="humidity", instrument="rh_controller",
                           tolerance=2.0, warn=5.0, fault=15.0, hold_time_s=600.0)
        detail = outcome.describe()

        assert outcome.met is False
        assert "%RH" in detail
        assert "anneal" not in detail.lower()
        assert "°C" not in detail and " C" not in detail
        # The driver's own sentence is kept for provenance, not shown as the
        # explanation: it is written for a temperature anneal.
        assert "Anneal hold aborted" in outcome.safety_message

    def test_the_hold_statistics_survive_a_safety_error_because_the_reader_records_every_sample(self):
        # monitored_hold raises without constructing a HoldReport, and the
        # SafetyError carries held_s and pv in message text only. The most
        # informative failures would lose their data.
        outcome, _ = _hold([50.0], 100.0, hold_time_s=3600.0, grace_s=120.0)

        assert outcome.met is False
        assert outcome.n_samples > 0
        assert outcome.pv_min == pytest.approx(50.0)
        assert outcome.pv_max == pytest.approx(50.0)
        assert outcome.held_s > 0
        assert outcome.safety_message

    def test_an_unmet_setpoint_short_of_target_is_recorded_and_does_not_abort(self):
        # Discovering the rig cannot hold this setpoint IS the result.
        outcome, _ = _hold([80.0], 85.0, hold_time_s=600.0, fault=10.0)
        assert outcome.met is False
        assert outcome.pv_max == pytest.approx(80.0)

    def test_an_unreadable_pv_aborts_rather_than_being_recorded_as_an_unmet_setpoint(self):
        # `monitored_hold` collapses "PV outside the band" and "PV unreadable"
        # into one SafetyError, distinguished only by message text. The first is
        # the answer to question 2; the second is a dead sensor, and continuing
        # runs the rig blind for nine hours. Decided from the series, not the text.
        with pytest.raises(EquilibrationAbort) as excinfo:
            _hold([None], 85.0, hold_time_s=600.0)
        assert excinfo.value.kind == "unreadable_pv"

    def test_a_sustained_overshoot_beyond_the_fault_band_aborts(self):
        # monitored_hold grades abs(pv - target): a stage that cannot REACH 85 and
        # a heater running past it are the same exception. The sign is checked here.
        with pytest.raises(EquilibrationAbort) as excinfo:
            _hold([120.0], 85.0, hold_time_s=600.0, fault=10.0, grace_s=120.0)
        assert excinfo.value.kind == "sustained_overshoot"

    def test_a_momentary_overshoot_is_not_a_runaway(self):
        outcome, _ = _hold([120.0, 85.0], 85.0, hold_time_s=600.0, fault=10.0,
                           grace_s=120.0)
        assert outcome.met is False        # it did leave the tolerance band
        assert outcome.pv_max == pytest.approx(120.0)

    def test_the_excursion_count_is_this_modules_own_because_it_passes_its_own_on_warn(self):
        outcome, _ = _hold([85.0, 92.0, 85.0, 92.0], 85.0, hold_time_s=600.0,
                           warn=3.0, fault=30.0, poll_interval_s=30.0)
        assert outcome.n_warn >= 1


# ── Naming: three collisions, one of them loud ───────────────────────────────

def _design_config(**kw) -> EquilibrationConfig:
    params = dict(channels=[1, 2], temperatures_C=[27.5, 45.0], rounds_per_setpoint=5,
                  electrode_geometry={"L_cm": 0.2, "t_cm": 0.0175, "w_cm": 0.2})
    params.update(kw)
    return EquilibrationConfig(**params)


def _every_step_name(config: EquilibrationConfig) -> list[str]:
    names: list[str] = []
    for leg in config.legs:
        for sp_idx, temp in enumerate(config.leg_temperatures(leg)):
            for round_index in range(config.rounds_per_setpoint):
                wf = build_round_workflow(
                    config, leg=leg, setpoint_index=sp_idx,
                    round_index=round_index, temperature_C=temp)
                names.extend(step.name for step in wf.setup)
    return names


class TestStepNaming:
    def test_every_repeat_at_one_setpoint_gets_a_distinct_step_name_and_a_distinct_eis_file(self):
        # Layer 1 (_build_dag) refuses duplicates loudly. Layer 2 (router's
        # file_stem = step.name) overwrites the earlier .txt SILENTLY, which is the
        # trap that becomes live the moment names are made unique carelessly.
        names = _every_step_name(_design_config())
        assert len(names) == len(set(names))

        stems = [f"{name}_ch1" for name in names]   # router's non-Arrhenius stem
        assert len(stems) == len(set(stems))

    def test_the_up_leg_and_down_leg_points_at_the_same_temperature_are_separately_addressable(self):
        config = _design_config()
        up = measurement_step_name(1, "up", 0, 0)
        down = measurement_step_name(1, "down", 1, 0)
        assert up != down
        # 45 C is setpoint 1 on the up leg and setpoint 0 on the down leg; the
        # temperature VALUE was never the identity (round(27.5) is 28).
        assert config.leg_temperatures("down")[0] == config.leg_temperatures("up")[-1]

    def test_no_step_written_by_this_run_matches_the_arrhenius_sweep_step_name_regex(self):
        from softae.workflows.temp_eis_sweep import _EIS_STEP_RE, _EIS_STEP_RE_LEGACY

        for name in _every_step_name(_design_config()):
            assert _EIS_STEP_RE.match(name) is None
            assert _EIS_STEP_RE_LEGACY.match(name) is None
            # router.py's own literal, which would otherwise take file_stem = name.
            assert re.match(r"^eis_ch\d+_T\d+_RH\d+$", name) is None

    def test_the_executor_accepts_a_whole_round_because_the_names_are_unique(self):
        from softae.workflows.workflow_executor import WorkflowExecutor

        wf = build_round_workflow(_design_config(channels=list(range(1, 17))),
                                  leg="up", setpoint_index=0, round_index=0)
        dag = WorkflowExecutor._build_dag(None, wf.setup)
        assert len(dag) == 16

    def test_the_script_path_is_namespaced_per_channel_and_kind(self):
        from softae.workflows.equilibration import mscr_path

        assert mscr_path(3, KIND_SERIES) != mscr_path(4, KIND_SERIES)
        assert mscr_path(3) == mscr_path(3, KIND_SERIES)

    def test_the_output_directory_is_per_run_and_never_the_shared_eis_output(self):
        # eis_measure_step points every workflow in the system at one
        # <tmp>/softae_eis_output, and the driver names the file from a timestamp
        # alone -- so a file left there by an earlier run would be attributed to
        # this one and enter sigma(t) at a coordinate it never occupied.
        from softae.workflows.equilibration import round_outdir

        first = round_outdir("20260810T010203Z_equilibration_characterization")
        second = round_outdir("20260811T010203Z_equilibration_characterization")

        assert first != second
        assert "softae_eis_output" not in first
        assert "softae_eis_output" not in second

    def test_every_step_in_a_round_writes_into_that_runs_own_output_directory(self):
        wf = build_round_workflow(_design_config(), leg="up", setpoint_index=0,
                                  round_index=0, run_id="RUN_A")
        other = build_round_workflow(_design_config(), leg="up", setpoint_index=0,
                                     round_index=0, run_id="RUN_B")

        outdirs = {step.params["outdir"] for step in wf.setup}
        assert len(outdirs) == 1
        assert "softae_eis_output" not in outdirs.pop()
        assert wf.setup[0].params["outdir"] != other.setup[0].params["outdir"]

    def test_a_round_built_without_a_run_id_still_avoids_the_shared_directory(self):
        # The default keeps existing callers working; it must not put them back on
        # the path every other workflow writes to.
        wf = build_round_workflow(_design_config(), leg="up", setpoint_index=0,
                                  round_index=0)
        assert "softae_eis_output" not in wf.setup[0].params["outdir"]


class TestStepContent:
    def test_the_measurement_steps_carry_circuit_model_and_geometry_so_a_fit_row_is_written(self):
        # router.handle auto-fits ONLY when step.params["circuit_model"] is
        # present, and takes L/t/w from params. Without both there is no
        # fit_results row, no sigma, and therefore no sigma(t) series at all --
        # and eis_measure_step sets neither.
        wf = build_round_workflow(_design_config(), leg="up", setpoint_index=0,
                                  round_index=0)
        for step in wf.setup:
            assert step.params["circuit_model"] == "simpleSalt"
            assert step.params["electrode_L_cm"] == pytest.approx(0.2)
            assert step.params["electrode_t_cm"] == pytest.approx(0.0175)
            assert step.params["electrode_w_cm"] == pytest.approx(0.2)

    def test_this_run_records_no_thermal_history_and_tags_no_measurement_as_a_drift_repeat(self):
        from softae.analysis.eis.calibration import DRIFT_REPEAT_ROLE

        wf = build_round_workflow(_design_config(), leg="down", setpoint_index=1,
                                  round_index=2)
        for step in wf.setup:
            assert "thermal_history" not in step.tags
            assert step.tags.get("role", "sample") == "sample"
            assert step.tags.get("role") != DRIFT_REPEAT_ROLE

    def test_the_coordinate_is_on_the_step_even_though_the_router_will_drop_it(self):
        # router.handle reads a fixed list of tags; the coordinate is persisted by
        # the sidecar, not by these tags. They are carried anyway so a live log
        # line and a step name cannot disagree.
        wf = build_round_workflow(_design_config(), leg="down", setpoint_index=1,
                                  round_index=2)
        assert wf.setup[0].tags["leg"] == "down"
        assert wf.setup[0].tags["setpoint_index"] == "1"
        assert wf.setup[0].tags["round_index"] == "2"
        assert wf.setup[0].tags["kind"] == KIND_SERIES


# ── The projection ───────────────────────────────────────────────────────────

class TestProjection:
    def test_the_projection_does_not_route_holds_through_estimate_workflow_duration(self,
                                                                                   monkeypatch):
        # estimate_step_duration returns 0.0 (not None) for method == "wait" unless
        # the params carry duration_s/seconds -- and temperature waits carry
        # within/equilibration_time/timeout. Every hold would project as free AND
        # DurationEstimate.is_complete would still say True.
        import softae.core.preflight as preflight

        def _boom(*_a, **_kw):
            raise AssertionError("holds must not be projected through the wait branch")

        monkeypatch.setattr(preflight, "estimate_workflow_duration", _boom)
        monkeypatch.setattr(preflight, "estimate_step_duration", _boom)

        projection = project_duration(_design_config())
        assert projection.worst_case_s > 0

    def test_the_projected_worst_case_uses_the_timeouts_rather_than_an_assumed_ramp_rate(self):
        base = _design_config(approach_timeout_s=1800.0, rh_approach_timeout_s=1800.0)
        doubled = _design_config(approach_timeout_s=3600.0, rh_approach_timeout_s=3600.0)

        delta = (project_duration(doubled).worst_case_s
                 - project_duration(base).worst_case_s)
        assert delta == pytest.approx(3600.0 * base.n_setpoints)
        assert project_duration(base).breakdown_worst["temperature_approach"] == 1800.0

    def test_the_shipped_design_projects_the_overnight_run_the_spec_budgeted(self):
        config = EquilibrationConfig()          # 16 channels, 4 temps, 2 legs, 15 rounds
        projection = project_duration(config)
        assert projection.n_setpoints == 8
        # The CEILING, which is what the old fixed-count projection was. It grew
        # with the round-period default (120 s -> 660 s, derived from 16 channels
        # x the measured 40.7 s/channel on 'Standard'), because the old default
        # projected a night the rig could not actually run at that channel count.
        assert projection.typical_s / 3600 == pytest.approx(24.0, abs=0.05)
        assert projection.worst_case_s / 3600 == pytest.approx(30.0, abs=0.05)
        # And the FLOOR, which the settle criterion makes reachable. At the 660 s
        # default period the time floors buy only ceil(1500/660) = 3 rounds at the
        # first setpoint and ceil(600/660) = 1 after, and settle_n_rounds is 3 --
        # so MIN_POINTS_FOR_TAU is what binds at EVERY setpoint, first included.
        assert (projection.min_rounds_first, projection.min_rounds_later) == (
            MIN_POINTS_FOR_TAU, MIN_POINTS_FOR_TAU)
        assert projection.typical_floor_s / 3600 == pytest.approx(9.3, abs=0.05)
        assert projection.worst_floor_s / 3600 == pytest.approx(15.3, abs=0.05)
        assert projection.adaptive is True
        assert "anchor_rounds" not in projection.breakdown_typical

    def test_the_round_gap_leaves_room_for_the_measurement_inside_the_round_period(self):
        config = EquilibrationConfig()
        gap = inter_round_gap_s(config)
        assert 0 < gap < config.round_period_s


# ── The run loop ─────────────────────────────────────────────────────────────

class _FakeTemp:
    def __init__(self, pv_values):
        self._read = _reader(pv_values)
        self.setpoints: list[float] = []
        self.reads = 0

    def write_sp(self, T_SP, print_flag=1):
        self.setpoints.append(float(T_SP))

    def get_pv(self, n_avg=1):
        self.reads += 1
        return self._read()

    def get_sp(self):
        return self.setpoints[-1] if self.setpoints else float("nan")


class _FakeRH:
    def __init__(self, pv_values):
        self._read = _reader(pv_values)
        self.setpoint = None
        self.started = False
        self.stopped = False
        #: Stands in for ``AsyncRHController``'s ``max_stale_s`` behaviour: a
        #: reading held past its shelf life comes back NaN, not last-good.
        self.air_temp_C = float("nan")

    def set_setpoint(self, val):
        self.setpoint = float(val)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def status(self):
        return {"running": self.started, "setpoint": self.setpoint}

    def get_H(self):
        return self._read()

    def get_T(self):
        return self.air_temp_C


class _FakeManager:
    def __init__(self, temp, rh):
        self._instruments = {"temp_controller": temp, "rh_controller": rh}

    @property
    def names(self):
        return list(self._instruments)

    def get(self, name):
        return self._instruments[name]


class _HeldLock:
    """A lock that is held and would block forever if anyone waited on it."""

    def locked(self):
        return True

    def acquire(self, *_a, **_kw):        # pragma: no cover - must never run
        raise AssertionError("telemetry must ask, never wait")


class _FakeExecutor:
    """Completes every step immediately, as the real executor does on success.

    ``raises`` makes it fault mid-setpoint instead — a driver that stops
    answering, or a Ctrl-C landing between two channels. ``counter`` is shared
    across the per-round executors so the fault can be aimed at a given round of
    the whole run.
    """

    def __init__(self, clock=None, per_step_s=0.0, raises=None, counter=None):
        self.on_step_complete = None
        self.workflows = []
        self._clock = clock
        self._per_step_s = float(per_step_s)
        self._raises = raises
        self._counter = counter

    async def run(self, workflow):
        if self._raises is not None:
            self._counter["rounds"] += 1
            if self._counter["rounds"] >= self._counter["fail_on"]:
                raise self._raises
        self.workflows.append(workflow)
        steps = list(workflow.setup)
        for index, step in enumerate(steps):
            if self._clock is not None and self._per_step_s:
                self._clock.t += self._per_step_s
            if self.on_step_complete is not None:
                self.on_step_complete(step, index, len(steps), {"raw": True}, 1.0)


def _runner(temp_pv, rh_pv, tmp_path, *, per_step_s=0.0, raises=None,
            fail_on_round=1, **cfg_kw):
    clock = _Clock()
    executors: list[_FakeExecutor] = []
    counter = {"rounds": 0, "fail_on": int(fail_on_round)}

    def _factory(*_a, **_kw):
        ex = _FakeExecutor(clock=clock, per_step_s=per_step_s, raises=raises,
                           counter=counter)
        executors.append(ex)
        return ex

    params = dict(channels=[1], temperatures_C=[45.0], legs=("up",),
                  rounds_per_setpoint=5, approach_timeout_s=300.0,
                  rh_approach_timeout_s=300.0, poll_interval_s=10.0, grace_s=30.0,
                  electrode_geometry={"L_cm": 0.2, "t_cm": 0.0175, "w_cm": 0.2})
    params.update(cfg_kw)
    config = EquilibrationConfig(**params)
    temp, rh = _FakeTemp(temp_pv), _FakeRH(rh_pv)

    class _Store:
        project_dir = str(tmp_path)

    run = EquilibrationRun(config, _FakeManager(temp, rh), data_store=_Store(),
                           run_id="RUN1", sleep=clock.sleep, now=clock.now,
                           executor_factory=_factory)
    return run, temp, rh, executors


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_an_unmet_setpoint_is_recorded_with_hold_met_false_and_the_run_continues(
            self, tmp_path):
        # 40 C against a 45 C setpoint: inside the fault band, so monitored_hold
        # never raises -- but it is not held, and that is the primary result.
        run, temp, _rh, _ex = _runner([40.0], [15.0], tmp_path)
        payload = await run.run()

        assert payload["setpoints"][0]["hold_met"] is False
        assert payload["setpoints"][0]["rh_hold_met"] is True
        assert payload["setpoints"][0]["temp_approach_reached"] is False
        assert not payload["aborted"]
        # It ran to completion: every round measured, and nothing else -- the
        # Longest anchor rounds are retired (they sampled below the ~9 Hz phase
        # floor), so a setpoint is exactly `rounds_per_setpoint` rounds.
        assert len(payload["points"]) == 5
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)

    @pytest.mark.asyncio
    async def test_a_chamber_that_cannot_dry_below_ambient_is_recorded_not_aborted(
            self, tmp_path):
        # THE answer to question 2. The overshoot abort fires regardless of axis,
        # so a narrow RH fault band would turn "the rig cannot hold 15 %RH" into a
        # crash instead of the primary result this run exists to obtain.
        run, _temp, _rh, _ex = _runner([45.0], [42.0], tmp_path)
        payload = await run.run()

        assert not payload["aborted"]
        assert payload["setpoints"][0]["rh_hold_met"] is False
        assert payload["setpoints"][0]["rh_hold_pv_max"] == pytest.approx(42.0)

    @pytest.mark.asyncio
    async def test_a_humidifier_running_away_still_aborts(self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [90.0], tmp_path)
        with pytest.raises(EquilibrationAbort) as excinfo:
            await run.run()
        assert excinfo.value.kind == "sustained_overshoot"
        assert excinfo.value.axis == "humidity"

    @pytest.mark.asyncio
    async def test_a_held_setpoint_records_met_and_writes_the_sidecar(self, tmp_path):
        run, _temp, rh, _ex = _runner([45.0], [15.0], tmp_path)
        payload = await run.run()

        assert payload["setpoints"][0]["hold_met"] is True
        assert rh.setpoint == pytest.approx(15.0) and rh.started
        path = Path(tmp_path) / "runs" / "RUN1" / "equilibration.json"
        assert path.exists()
        assert payload["schema"] == "equilibration/1"
        assert payload["points"][0]["t_since_hold_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_an_unreadable_pv_aborts_the_run_rather_than_being_recorded_as_an_unmet_setpoint(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([None], [15.0], tmp_path)
        with pytest.raises(EquilibrationAbort) as excinfo:
            await run.run()
        assert excinfo.value.kind == "unreadable_pv"
        assert run.aborted

    @pytest.mark.asyncio
    async def test_a_sustained_overshoot_beyond_the_fault_band_aborts_and_restores_ambient(
            self, tmp_path):
        run, temp, _rh, _ex = _runner([60.0], [15.0], tmp_path, fault_C=10.0)
        with pytest.raises(EquilibrationAbort) as excinfo:
            await run.run()

        assert excinfo.value.kind == "sustained_overshoot"
        assert run.restored_ambient is True
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)
        # The partial run is still on disk: an abort must not cost the evidence.
        assert (Path(tmp_path) / "runs" / "RUN1" / "equilibration.json").exists()

    @pytest.mark.asyncio
    async def test_the_sidecar_carries_the_coordinate_the_database_cannot(self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       temperatures_C=[27.5, 45.0], legs=("up", "down"),
                                       fault_C=50.0)
        payload = await run.run()

        keys = {(p["leg"], p["setpoint_index"], p["round_index"], p["kind"])
                for p in payload["points"]}
        assert ("up", 0, 0, KIND_SERIES) in keys
        assert ("down", 1, 2, KIND_SERIES) in keys
        assert len({p["step_name"] for p in payload["points"]}) == len(payload["points"])

    @pytest.mark.asyncio
    async def test_both_axes_are_graded_by_the_same_watched_primitive(self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)
        payload = await run.run()

        axes = {hold["axis"] for hold in payload["holds"]}
        assert axes == {"temperature", "humidity"}
        assert {a["axis"] for a in payload["approaches"]} == {"temperature", "humidity"}


class TestInterruptSafeTeardown:
    """Every exit path brings the chamber down and persists what was recorded.

    ``run()`` used to catch **only** ``EquilibrationAbort``. A Ctrl-C, a driver
    ``CommunicationError`` or any bug propagated past both the ambient restore and
    the sidecar write — leaving the stage heater commanded at up to 85 °C with the
    process gone and the drivers disconnected, and losing the hold verdicts, which
    are the one thing in this run that no reconstruction from the database can
    recover.
    """

    @pytest.mark.asyncio
    async def test_run_keyboard_interrupt_restores_ambient_and_writes_the_sidecar(
            self, tmp_path):
        run, temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                      raises=KeyboardInterrupt())
        with pytest.raises(KeyboardInterrupt):
            await run.run()

        assert run.restored_ambient is True
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)
        assert run.sidecar_written is True
        assert (Path(tmp_path) / "runs" / "RUN1" / "equilibration.json").exists()

    @pytest.mark.asyncio
    async def test_run_generic_exception_restores_ambient_and_writes_the_sidecar(
            self, tmp_path):
        # A driver CommunicationError, a SafetyError or a plain bug: none is an
        # EquilibrationAbort, and all of them used to walk past the teardown.
        run, temp, _rh, _ex = _runner(
            [45.0], [15.0], tmp_path,
            raises=RuntimeError("the potentiostat stopped answering"))
        with pytest.raises(RuntimeError, match="stopped answering"):
            await run.run()

        assert run.restored_ambient is True
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)
        assert run.sidecar_written is True
        # Recorded as what it was. `abort_reason` must not read as if a watched
        # hold had decided it.
        assert run.aborted is True
        assert run.abort_reason.startswith("RuntimeError: ")

    @pytest.mark.asyncio
    async def test_run_equilibration_abort_keeps_its_recorded_kind_and_teardown(
            self, tmp_path):
        # Regression pin. The abort path predates this fix and its semantics --
        # `kind:` prefixed reason, ambient restored, sidecar on disk, exception
        # re-raised with its `kind` and `axis` -- must not have moved.
        run, temp, _rh, _ex = _runner([60.0], [15.0], tmp_path, fault_C=10.0)
        with pytest.raises(EquilibrationAbort) as excinfo:
            await run.run()

        assert excinfo.value.kind == "sustained_overshoot"
        assert excinfo.value.axis == "temperature"
        assert run.aborted is True
        assert run.abort_reason.startswith("sustained_overshoot: ")
        assert run.restored_ambient is True
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)
        assert (Path(tmp_path) / "runs" / "RUN1" / "equilibration.json").exists()

    @pytest.mark.asyncio
    async def test_run_restore_failure_does_not_mask_the_original_exception(
            self, tmp_path):
        # The restore runs in a context where a driver may ALREADY be broken --
        # often that is exactly why we are here. Its own failure must not become
        # the exception the operator is shown.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       raises=RuntimeError("the original fault"))

        async def _restore_explodes():
            raise OSError("the serial port is gone")

        run._restore_ambient = _restore_explodes

        with pytest.raises(RuntimeError, match="the original fault"):
            await run.run()
        assert "OSError" in run.restore_error

    @pytest.mark.asyncio
    async def test_run_sidecar_failure_does_not_mask_the_original_exception(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       raises=RuntimeError("the original fault"))

        def _write_explodes():
            raise OSError("the disk is full")

        run._write_sidecar = _write_explodes

        with pytest.raises(RuntimeError, match="the original fault"):
            await run.run()
        assert run.sidecar_written is False
        assert "OSError" in run.sidecar_error

    @pytest.mark.asyncio
    async def test_run_restore_failure_is_reported_naming_the_last_commanded_setpoint(
            self, tmp_path):
        # A silent restore attempt is indistinguishable from a successful one, and
        # the number the operator has to walk to the rig with is the setpoint the
        # chamber is still commanded to -- not the config's peak, and not ambient.
        run, temp, _rh, _ex = _runner([85.0], [15.0], tmp_path,
                                      temperatures_C=[85.0], fault_C=50.0)
        events = _collect(run)
        accepted = temp.write_sp

        def _refuse_ambient(T_SP, print_flag=1):
            # Only the RESTORE write fails. The run's own 85 C setpoint already
            # landed -- which is precisely the state that leaves a heater hot.
            if abs(float(T_SP) - run.config.ambient_C) < 1e-9:
                raise OSError("the temperature controller is not answering")
            accepted(T_SP, print_flag)

        temp.write_sp = _refuse_ambient
        payload = await run.run()

        assert run.restored_ambient is False
        assert "not answering" in run.restore_error
        notices = [e for e in events if e.kind == EV_AMBIENT_RESTORED]
        assert len(notices) == 1
        assert notices[0].verdict == VERDICT_UNMET
        assert "85 C" in notices[0].detail
        # The instruction leads, because the unbounded driver message trails and
        # the renderer truncates a long milestone.
        assert notices[0].detail.startswith("CHECK THE CHAMBER MANUALLY")
        assert payload["restored_ambient"] is False
        assert payload["last_commanded_C"] == pytest.approx(85.0)

    @pytest.mark.asyncio
    async def test_run_successful_restore_is_stated_rather_than_left_to_be_assumed(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)
        events = _collect(run)
        await run.run()

        notices = [e for e in events if e.kind == EV_AMBIENT_RESTORED]
        assert len(notices) == 1
        assert notices[0].verdict == VERDICT_MET
        assert "ambient restored" in notices[0].detail

    @pytest.mark.asyncio
    async def test_run_writes_the_sidecar_after_each_setpoint_not_only_at_the_end(
            self, tmp_path):
        # A power cut or a `kill -9` catches no handler at all, so the record of
        # setpoint 1 has to already be on disk while setpoint 2 is still running.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       temperatures_C=[40.0, 45.0], fault_C=50.0)
        path = Path(tmp_path) / "runs" / "RUN1" / "equilibration.json"
        on_disk: list = []

        def _peek(event):
            if event.kind == EV_SETPOINT_STARTED and event.setpoint_index == 1:
                on_disk.append(json.loads(path.read_text(encoding="utf-8"))
                               if path.exists() else None)

        run.on_progress = _peek
        await run.run()

        assert on_disk and on_disk[0] is not None, \
            "nothing was on disk when the second setpoint began"
        assert len(on_disk[0]["setpoints"]) == 1


class TestAgainstTheRealExecutor:
    """The one test that uses the real ``WorkflowExecutor`` and the real router.

    The fakes above cannot show that a spectrum actually lands with a σ attached,
    and that is the whole chain this run depends on: ``circuit_model`` + geometry
    on the step → ``router.handle`` auto-fits → a ``fit_results`` row → a σ(t)
    series. Mock drivers only; nothing here is armed and nothing moves.
    """

    @pytest.mark.asyncio
    async def test_a_mock_round_lands_a_sigma_that_the_join_reads_back(self, tmp_path):
        from softae.analysis.equilibration import load_sigma_series
        from softae.core.data_store import DataStore
        from softae.drivers.factory import create_manager

        manager = create_manager(mock=True)
        await manager.connect_all()
        store = DataStore(str(tmp_path / "proj"), db_filename="test.db")
        run_id = store.start_run("equilibration_characterization",
                                 mode="characterization")
        clock = _Clock()
        config = EquilibrationConfig(
            channels=[1], temperatures_C=[30.0], legs=("up",), rounds_per_setpoint=5,
            approach_timeout_s=60.0, rh_approach_timeout_s=60.0, poll_interval_s=10.0,
            grace_s=60.0, fault_C=50.0, rh_fault_pct=60.0,
            electrode_geometry={"L_cm": 0.2, "t_cm": 0.0175, "w_cm": 0.2})
        run = EquilibrationRun(config, manager, data_store=store, run_id=run_id,
                               sleep=clock.sleep, now=clock.now)
        try:
            payload = await run.run()
        finally:
            await manager.disconnect_all()

        try:
            # 5 series rounds (MIN_POINTS_FOR_TAU), one channel. No anchors.
            assert len(payload["points"]) == 5
            series = load_sigma_series(store, run_id, payload)
            points = series[(1, "up", 0, KIND_SERIES)]
            assert len(points) == 5
            assert all(p["sigma"] is not None for p in points), \
                "no sigma means circuit_model or the geometry never reached the router"
            assert [p["t_since_hold_s"] for p in points] == sorted(
                p["t_since_hold_s"] for p in points)
            # The conditions snapshot rides along on every shot, unchanged.
            assert points[0]["rh_pv_pct"] is not None
        finally:
            store.close()


# ── Progress: the nine-hour silence ──────────────────────────────────────────

def _collect(run):
    """Attach a recording hook and hand back the event list."""
    events = []
    run.on_progress = events.append
    return events


class TestProgressEvents:
    @pytest.mark.asyncio
    async def test_the_progress_events_fire_in_the_hierarchy_order_an_operator_reads(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)
        events = _collect(run)
        await run.run()
        kinds = [e.kind for e in events]

        assert kinds[0] == EV_RUN_STARTED
        # The ambient verdict is deliberately the LAST thing said, after the run's
        # own summary: whether the chamber came down is what an operator needs to
        # read before walking away, so nothing is allowed to follow it.
        assert kinds[-1] == EV_AMBIENT_RESTORED
        assert kinds[-2] == EV_RUN_FINISHED
        assert kinds.index(EV_LEG_STARTED) < kinds.index(EV_SETPOINT_STARTED)
        assert kinds.index(EV_SETPOINT_STARTED) < kinds.index(EV_ROUND_STARTED)
        assert kinds.index(EV_ROUND_STARTED) < kinds.index(EV_CHANNEL_MEASURED)
        assert kinds.index(EV_CHANNEL_MEASURED) < kinds.index(EV_ROUND_FINISHED)
        assert kinds.index(EV_SETPOINT_FINISHED) < kinds.index(EV_LEG_FINISHED)
        assert kinds.index(EV_LEG_FINISHED) < kinds.index(EV_RUN_FINISHED)

    @pytest.mark.asyncio
    async def test_every_event_carries_the_whole_coordinate_so_a_dropped_one_costs_nothing(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)
        events = _collect(run)
        await run.run()

        rounds = [e for e in events if e.kind == EV_CHANNEL_MEASURED]
        assert rounds, "no per-channel event was emitted"
        for event in rounds:
            assert event.leg == "up"
            assert event.setpoint_index == 0
            assert event.round_index >= 0
            assert event.channel == 1
            assert event.n_setpoints == 1

    @pytest.mark.asyncio
    async def test_a_heartbeat_fires_while_the_chamber_is_idle_in_a_watched_hold(
            self, tmp_path):
        # The watched gap between rounds is where nothing changes and nothing
        # prints -- exactly where a silent console reads as a hung run.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       progress_interval_s=20.0)
        events = _collect(run)
        await run.run()

        beats = [e for e in events if e.kind == EV_HEARTBEAT]
        assert beats, "no heartbeat during the idle hold"
        assert {e.phase for e in beats} == {"hold"}
        assert all(e.pv == e.pv for e in beats)     # a real reading, not NaN

    @pytest.mark.asyncio
    async def test_the_hold_verdict_reaches_the_operator_when_it_resolves(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([40.0], [15.0], tmp_path)
        events = _collect(run)
        await run.run()

        verdicts = [e for e in events if e.kind == EV_SETPOINT_FINISHED]
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "unmet"
        assert "did NOT hold" in verdicts[0].detail

    @pytest.mark.asyncio
    async def test_a_broken_renderer_cannot_abort_a_nine_hour_experiment(
            self, tmp_path):
        # A formatting bug or a closed pipe is a cosmetic failure. Letting it
        # propagate would cost the night AND the data.
        run, temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)

        def _explode(_event):
            raise RuntimeError("closed pipe")

        run.on_progress = _explode
        payload = await run.run()

        assert not payload["aborted"]
        assert len(payload["points"]) == 5
        assert run.progress_failures > 0
        assert temp.setpoints[-1] == pytest.approx(run.config.ambient_C)

    @pytest.mark.asyncio
    async def test_the_shipped_renderer_consumes_a_real_event_stream_intact(
            self, tmp_path):
        # The events and the renderer live in different modules on purpose, which
        # is exactly how a field gets renamed on one side only. Nothing else pins
        # the two together.
        from softae.tools.equilibration import ProgressRenderer

        class _Sink:
            text = ""

            def isatty(self):
                return False

            def write(self, text):
                type(self).text += text

            def flush(self):
                pass

        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       progress_interval_s=20.0)
        renderer = ProgressRenderer(run.config, stream=_Sink(),
                                    milestone_interval_s=1.0)
        run.on_progress = renderer
        await run.run()

        assert renderer.failures == 0
        assert run.progress_failures == 0
        assert "VERDICT" in _Sink.text
        assert "env  T sp" in _Sink.text
        assert "DONE" in _Sink.text

    @pytest.mark.asyncio
    async def test_the_abort_path_still_announces_before_it_raises(self, tmp_path):
        run, _temp, _rh, _ex = _runner([None], [15.0], tmp_path)
        events = _collect(run)
        with pytest.raises(EquilibrationAbort):
            await run.run()

        finished = [e for e in events if e.kind == EV_RUN_FINISHED]
        assert finished and finished[0].verdict == "aborted"


# ── Progress doubles as a headless controls monitor ──────────────────────────

class TestTelemetry:
    @pytest.mark.asyncio
    async def test_the_periodic_emission_carries_the_five_environment_values(
            self, tmp_path):
        # The point of this is that nobody has to open the GUI on a running
        # headless job -- which would contend for the rig lock and the serial
        # ports of the very run being checked on.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       progress_interval_s=20.0)
        events = _collect(run)
        await run.run()

        with_env = [e for e in events if e.env_status == ENV_OK]
        assert with_env, "no telemetry reached the progress stream"
        env = with_env[-1].env
        assert env["stage_temp_sp_C"] == pytest.approx(45.0)
        assert env["stage_temp_pv_C"] == pytest.approx(45.0)
        assert env["rh_sp_pct"] == pytest.approx(15.0)
        assert env["rh_pv_pct"] == pytest.approx(15.0)
        assert all(e.wall_clock for e in with_env)

    @pytest.mark.asyncio
    async def test_an_unreadable_value_is_absent_never_zero_and_never_last_good(
            self, tmp_path):
        # _FakeRH.get_T returns NaN, standing in for AsyncRHController's
        # max_stale_s behaviour. read_environment maps it to None, and None must
        # survive to the renderer as "unavailable".
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       progress_interval_s=20.0)
        events = _collect(run)
        await run.run()

        env = [e for e in events if e.env_status == ENV_OK][-1].env
        assert env["chamber_air_C"] is None
        assert env["chamber_air_C"] != 0.0

    def test_telemetry_is_skipped_rather_than_blocked_when_an_instrument_is_busy(
            self, tmp_path):
        # A missed monitor line is free. A telemetry read that WAITS delays the
        # next EIS shot, and the spacing of those shots is the sigma(t) series
        # this run exists to produce.
        run, temp, rh, _ex = _runner([45.0], [15.0], tmp_path)
        temp._lock = _HeldLock()
        before = temp.reads

        fields = run._env_fields()

        assert fields["env_status"] == ENV_SKIPPED
        assert fields["env"] == {}
        assert temp.reads == before, "a busy instrument was polled anyway"
        assert run.telemetry_skips == 1

    def test_telemetry_is_skipped_while_a_measurement_dag_owns_the_rig(self, tmp_path):
        run, temp, _rh, _ex = _runner([45.0], [15.0], tmp_path)
        run._measuring = True
        before = temp.reads

        assert run._env_fields()["env_status"] == ENV_SKIPPED
        assert temp.reads == before

    @pytest.mark.asyncio
    async def test_telemetry_is_emitted_during_the_approach_not_only_at_boundaries(
            self, tmp_path):
        # The approach is up to 30 min per axis with nothing else to print.
        run, _temp, _rh, _ex = _runner([30.0, 30.0, 30.0, 45.0], [15.0], tmp_path,
                                       progress_interval_s=10.0)
        events = _collect(run)
        await run.run()

        approach_env = [e for e in events
                        if e.phase == "approach" and e.env_status == ENV_OK]
        assert approach_env, "the approach phase emitted no telemetry"


# ── What a round really costs ────────────────────────────────────────────────

class TestMeasuredRoundCost:
    @pytest.mark.asyncio
    async def test_the_run_measures_what_a_round_actually_cost(self, tmp_path):
        # estimate_eis_duration models the frequency sweep only. This is the
        # first number in the system that includes the mux switch, the script
        # upload, the retrieval and the file write.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=11.0,
                                       channels=[1, 2, 3, 4])
        payload = await run.run()

        cost = payload["measured_cost"]
        assert cost["series"]["measured_round_s"] == pytest.approx(44.0)
        assert cost["series"]["measured_per_channel_s"] == pytest.approx(11.0)
        assert cost["series"]["measured_round_s"] > cost["series"]["modelled_round_s"]
        assert cost["series"]["ratio_measured_over_modelled"] > 1.0
        assert cost["series"]["unmodelled_per_channel_s"] > 0.0
        assert len(cost["rounds"]) == 5          # one preset, no anchor rounds
        assert "anchor" not in cost

    @pytest.mark.asyncio
    async def test_a_round_that_overruns_the_period_is_announced_loudly_and_early(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=200.0,
                                       round_period_s=120.0)
        events = _collect(run)
        await run.run()

        warnings = [e for e in events if e.kind == EV_COST_WARNING]
        assert len(warnings) == 1, "said once, not once per round"
        assert warnings[0].round_duration_s == pytest.approx(200.0)
        assert "--round-period-s" in warnings[0].detail
        # Fired on the FIRST series round, not at the end of the run.
        assert events.index(warnings[0]) < len(events) - 5

    @pytest.mark.asyncio
    async def test_the_sampling_interval_is_never_auto_adjusted_mid_run(self, tmp_path):
        # round_period_s is an experimental parameter: it sets the shortest
        # resolvable tau and the fitter reads the series as evenly spaced.
        # Stretching it silently would make sigma(t) inhomogeneous invisibly.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=200.0,
                                       round_period_s=120.0)
        payload = await run.run()

        assert run.config.round_period_s == pytest.approx(120.0)
        assert payload["config"]["round_period_s"] == pytest.approx(120.0)
        assert payload["measured_cost"]["period_overrun_warned"] is True

    def test_a_measured_cost_reprojects_the_run_without_erasing_the_modelled_one(self):
        # The measured round has to exceed the period for the reprojection to
        # lengthen the run, so the period is stated: the shipped default is
        # derived from the 16-channel measured cost and a 170 s round fits inside
        # it with room to spare.
        config = EquilibrationConfig(round_period_s=120.0)
        modelled = project_duration(config)
        measured = project_duration(config, measured_series_round_s=170.0)

        assert modelled.basis == "modelled" and measured.basis == "measured"
        assert measured.typical_s > modelled.typical_s
        assert project_duration(config).typical_s == pytest.approx(modelled.typical_s)

    def test_the_headroom_says_what_the_period_leaves_for_unmodelled_overhead(self):
        # 16 channels x Standard models ~62 s of sweep, and the model covers the
        # frequency sweep ONLY -- the mux switch, the upload, the retrieval and
        # the file write are what the remainder has to absorb.
        from softae.workflows.equilibration import eis_round_cost_s

        config = EquilibrationConfig(round_period_s=120.0)
        headroom = round_headroom_s_per_channel(config)
        expected = (120.0 - eis_round_cost_s(config, "Standard")) / 16

        assert headroom == pytest.approx(expected)
        assert 0.0 < headroom < 5.0, "a 120 s period leaves almost no headroom"

    def test_the_default_round_period_holds_a_measured_round_at_the_default_channels(
            self):
        # The regression the default now closes: 120 s against 16 channels could
        # not contain a round on ANY preset (~168 s on the fastest, 651 s on the
        # default one), so a run accepting every default could not honour its own
        # sampling interval and sigma(t) was not evenly spaced as the fitter reads
        # it. Headroom is now positive on the MEASURED cost, not merely on the
        # model.
        from softae.tools.equilibration import MEASURED_PER_CHANNEL_S_STANDARD
        from softae.workflows.equilibration import round_cost_s

        config = EquilibrationConfig()          # 16 ch, 'Standard'
        measured = round_cost_s(
            config, measured_per_channel_s=MEASURED_PER_CHANNEL_S_STANDARD)
        assert measured <= config.round_period_s
        assert round_headroom_s_per_channel(config, measured) >= 0.0


# ── The round period is honoured from the round's START ──────────────────────

def _operator_config(**kw):
    """The settings of the real bench run: 12 channels, 240 s, Quick."""
    params = dict(channels=list(range(1, 13)), round_period_s=240.0,
                  eis_preset="Quick", poll_interval_s=30.0)
    params.update(kw)
    return EquilibrationConfig(**params)


class TestRoundPeriodIsHonoured:
    def test_the_gap_uses_the_measured_round_rather_than_the_modelled_one(self):
        # The defect: the model covers the frequency sweep only (~17 s for this
        # config), so subtracting it left a 223 s gap after a 130 s round -- a
        # 353 s cycle against a configured 240 s.
        config = _operator_config()
        assert inter_round_gap_s(config, 130.0) == pytest.approx(110.0)
        assert inter_round_gap_s(config) > 220.0, "the modelled path is the old bug"

    def test_the_gap_with_no_measurement_still_returns_the_modelled_value(self):
        # Pins the projection path: `plan` and `project_duration` have nothing
        # measured and must keep the behaviour they had.
        from softae.workflows.equilibration import eis_round_cost_s

        config = _operator_config()
        assert inter_round_gap_s(config) == pytest.approx(
            config.round_period_s - eis_round_cost_s(config, config.eis_preset))

    @pytest.mark.parametrize("measured_s", [1.0, 60.0, 130.0, 209.9])
    def test_the_round_plus_the_gap_equals_the_configured_period(self, measured_s):
        config = _operator_config()
        cycle = measured_s + inter_round_gap_s(config, measured_s)
        assert cycle == pytest.approx(config.round_period_s)

    def test_a_round_that_overruns_the_period_floors_the_gap_at_one_poll_interval(self):
        # Not zero: both axes are graded by sampling this gap, and a zero gap
        # would leave temperature and humidity ungraded for the whole series.
        config = _operator_config()
        gap = inter_round_gap_s(config, 400.0)
        assert gap == pytest.approx(config.poll_interval_s)
        assert gap > 0.0

    def test_the_projection_describes_the_same_cycle_the_executor_performs(self):
        config = _operator_config(rounds_per_setpoint=10)
        projection = project_duration(config, measured_series_round_s=130.0)
        assert projection.series_round_s == pytest.approx(
            130.0 + inter_round_gap_s(config, 130.0))
        assert projection.series_round_s == pytest.approx(config.round_period_s)

    @pytest.mark.asyncio
    async def test_the_run_loop_spaces_rounds_by_the_period_when_the_round_fits(
            self, tmp_path):
        # One channel at 130 s/step, so a round costs 130 s inside a 240 s period.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=130.0,
                                       round_period_s=240.0, poll_interval_s=10.0,
                                       rounds_per_setpoint=5)
        await run.run()

        stamps = [p["t_since_hold_s"] for p in run.points]
        assert len(stamps) == 5
        spacings = [b - a for a, b in zip(stamps, stamps[1:])]
        assert spacings == pytest.approx([240.0] * 4)

    @pytest.mark.asyncio
    async def test_the_run_loop_pays_the_poll_floor_and_warns_when_a_round_overruns(
            self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=200.0,
                                       round_period_s=120.0, poll_interval_s=10.0,
                                       rounds_per_setpoint=5)
        events = _collect(run)
        await run.run()

        stamps = [p["t_since_hold_s"] for p in run.points]
        spacings = [b - a for a, b in zip(stamps, stamps[1:])]
        # The round cannot be shortened, so the cycle is the round plus the floor.
        assert spacings == pytest.approx([210.0] * 4)
        warnings = [e for e in events if e.kind == EV_COST_WARNING]
        assert len(warnings) == 1
        assert "UNACHIEVABLE" in warnings[0].detail
        assert "210s" in warnings[0].detail, "the actual cycle time is not stated"
        # The `--round-period-s` suggestion sits at the end of the detail and
        # `ProgressRenderer._line` truncates a milestone at 234 characters, so an
        # over-long detail silently amputates the one actionable number.
        assert len(warnings[0].detail) <= 192, "this line will be truncated on screen"

    @pytest.mark.asyncio
    async def test_both_axes_are_still_graded_when_the_gap_falls_to_the_floor(
            self, tmp_path):
        # The floor exists for this: a zero gap would produce no watched window
        # at all, which is the fail-open behaviour this module replaced.
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path, per_step_s=200.0,
                                       round_period_s=120.0, poll_interval_s=10.0,
                                       rounds_per_setpoint=5)
        await run.run()

        axes = {h["axis"] for h in run.holds}
        assert axes == {"temperature", "humidity"}
        assert all(h["n_samples"] >= 1 for h in run.holds)


# ── Planning from a measured cost ────────────────────────────────────────────

class TestMeasuredRoundCostBasis:
    """A plan whose only cost basis is the model told the operator a 240 s period
    was fine when a real round was 488 s. The measured number has to be able to
    reach every figure the plan prints."""

    def test_round_cost_with_no_measurement_returns_the_modelled_cost(self):
        from softae.workflows.equilibration import eis_round_cost_s, round_cost_s

        config = _operator_config(eis_preset="Standard")
        assert round_cost_s(config) == pytest.approx(
            eis_round_cost_s(config, "Standard"))

    def test_round_cost_with_a_measurement_scales_it_by_the_channel_count(self):
        from softae.workflows.equilibration import round_cost_s

        config = _operator_config(eis_preset="Standard")     # 12 channels
        assert round_cost_s(config, measured_per_channel_s=40.7) == pytest.approx(488.4)

    @pytest.mark.parametrize("value", [None, 0.0, -1.0])
    def test_round_cost_treats_a_nonpositive_measurement_as_absent(self, value):
        from softae.workflows.equilibration import eis_round_cost_s, round_cost_s

        config = _operator_config(eis_preset="Standard")
        assert round_cost_s(config, measured_per_channel_s=value) == pytest.approx(
            eis_round_cost_s(config, "Standard"))

    def test_the_minimum_feasible_period_is_the_round_cost_rounded_up_to_ten(self):
        # The operator's real numbers: 12 channels at 40.7 s/channel.
        from softae.workflows.equilibration import minimum_feasible_period_s

        config = _operator_config(eis_preset="Standard")
        assert minimum_feasible_period_s(488.4) == pytest.approx(490.0)

    def test_the_headroom_uses_the_measured_round_when_one_is_supplied(self):
        config = _operator_config(eis_preset="Standard")     # 12 ch, 240 s period
        # 240 - 488.4 over 12 channels: negative, which is the honest answer. The
        # modelled headroom for the same config is positive and reassuring.
        assert round_headroom_s_per_channel(config, 488.4) == pytest.approx(-20.7)
        assert round_headroom_s_per_channel(config) > 0.0


# ── Thickness provenance ─────────────────────────────────────────────────────

class TestThicknessProvenance:
    @pytest.mark.asyncio
    async def test_the_sidecar_records_the_thickness_and_that_it_is_only_a_target(
            self, tmp_path):
        # The geometry reaches fit_results.electrode_t_cm through the step params,
        # but fit_results.thickness_method stays NULL because that column is only
        # filled from a SpectrumReport and this run passes none. So the stored
        # sigma would carry a hand-computed target with nothing saying so.
        run, _temp, _rh, _ex = _runner(
            [45.0], [15.0], tmp_path,
            electrode_geometry={"L_cm": 0.2, "t_cm": 0.02, "w_cm": 0.2})
        payload = await run.run()

        thickness = payload["thickness"]
        assert thickness["thickness_method"] == "target"
        assert thickness["t_cm"] == pytest.approx(0.02)
        assert thickness["value_um"] == pytest.approx(200.0)
        assert thickness["units"] == "um"
        assert thickness["recorded_in_fit_results"] is False

    def test_the_method_is_the_analysis_layers_vocabulary_not_a_new_string(self):
        from softae.analysis.eis.geometry import THICKNESS_METHODS

        config = EquilibrationConfig()
        assert config.thickness_method in THICKNESS_METHODS
        with pytest.raises(ValueError, match="thickness_method"):
            EquilibrationConfig(thickness_method="digital_twin").validate()

    @pytest.mark.asyncio
    async def test_no_geometry_means_no_thickness_claim_at_all(self, tmp_path):
        run, _temp, _rh, _ex = _runner([45.0], [15.0], tmp_path,
                                       electrode_geometry=None)
        payload = await run.run()
        assert payload["thickness"] == {}


class TestGeometryContract:
    """``electrode_geometry`` is all three terms or nothing — in the config, not
    only in the CLI that usually builds it.

    A partial dict is not a smaller geometry: ``build_round_workflow`` copies each
    term into the EIS step params, so a missing ``t_cm`` reaches ``router.handle``
    as ``None``. And a truthiness test cannot tell ``t_cm = 0`` — a stated, wrong
    value that σ = L/(R·t·w) divides by — from a term nobody supplied.
    """

    def test_a_partial_geometry_is_refused_naming_the_missing_terms(self):
        with pytest.raises(ValueError, match=r"MISSING t_cm"):
            EquilibrationConfig(
                electrode_geometry={"L_cm": 0.2, "w_cm": 0.2}).validate()

    def test_a_zero_thickness_is_refused_as_a_stated_value(self):
        with pytest.raises(ValueError, match="non-positive"):
            EquilibrationConfig(electrode_geometry={"L_cm": 0.2, "t_cm": 0.0,
                                                    "w_cm": 0.2}).validate()

    def test_all_three_terms_reach_the_eis_step_params(self):
        config = EquilibrationConfig(
            channels=[1], electrode_geometry={"L_cm": 0.2, "t_cm": 0.02,
                                              "w_cm": 0.2})
        config.validate()
        params = build_round_workflow(config, leg="up", setpoint_index=0,
                                      round_index=0).setup[0].params
        assert params["electrode_L_cm"] == pytest.approx(0.2)
        assert params["electrode_t_cm"] == pytest.approx(0.02)
        assert params["electrode_w_cm"] == pytest.approx(0.2)


class TestNoFailOpenWait:
    def test_no_call_site_in_this_module_reaches_either_controllers_fail_open_wait(self):
        # Both driver wait() primitives log a timeout and RETURN, and
        # temp_eis_sweep._abortable_wait_rh does the same. This run is correct
        # without the driver fix precisely because it never calls any of them.
        import softae.workflows.equilibration as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.strip().startswith("#"))
        assert ".wait(" not in code
        assert "_abortable_wait_rh(" not in code


# ── rounds_per_setpoint is a CEILING ─────────────────────────────────────────
#
# The 2026-08-11 run held all eight setpoints for the full 15 rounds. Only the
# first needed it: the sigma swing was 1600-2800% there, 57-1370% at the second,
# and 0.5-8.5% / 0.8-3.1% at the third and fourth -- flat inside a 5.98% measured
# noise floor. Seven setpoints x 45 min were spent re-measuring a settled number.
#
# The fits come from an injected reader rather than a database: the criterion is
# what is under test, and a real store would only add a join between the assertion
# and the thing asserted. `load_round_fits` itself is pinned in
# test_equilibration_analysis.py against a real DataStore.

def _fits(spec):
    """``{channel: (sigma, R1)}`` -> the round's fits."""
    return [RoundFit(channel=ch, sigma=s, r1_ohms=r1)
            for ch, (s, r1) in sorted(spec.items())]


def _fit_stream(per_round):
    """A fit reader driven by a list of per-round specs; the last repeats."""
    calls = {"n": 0}

    def _read(_step_names):
        index = min(calls["n"], len(per_round) - 1)
        calls["n"] += 1
        return _fits(per_round[index])

    _read.calls = calls
    return _read


def _flat_round(channels, sigma=2.0e-4, r1=5.0e3):
    return {ch: (sigma, r1) for ch in channels}


def _settle_runner(tmp_path, fit_reader, **cfg_kw):
    """A one-setpoint runner with the settle criterion wired to *fit_reader*."""
    clock = _Clock()

    def _factory(*_a, **_kw):
        return _FakeExecutor(clock=clock, per_step_s=1.0)

    params = dict(channels=[1, 2, 3], temperatures_C=[45.0], legs=("up",),
                  rounds_per_setpoint=8, approach_timeout_s=300.0,
                  rh_approach_timeout_s=300.0, poll_interval_s=10.0, grace_s=30.0,
                  round_period_s=100.0, settle_n_rounds=3, settle_min_channels=3,
                  min_hold_first_s=0.0, min_hold_s=0.0,
                  electrode_geometry={"L_cm": 0.2, "t_cm": 0.0175, "w_cm": 0.2})
    params.update(cfg_kw)
    config = EquilibrationConfig(**params)

    class _Store:
        project_dir = str(tmp_path)

    run = EquilibrationRun(config, _FakeManager(_FakeTemp([45.0]), _FakeRH([15.0])),
                           data_store=_Store(), run_id="RUNSETTLE",
                           sleep=clock.sleep, now=clock.now,
                           executor_factory=_factory, fit_reader=fit_reader)
    return run


def _setpoint(payload, index=0):
    return payload["setpoints"][index]


class TestAdaptiveSettleHold:
    @pytest.mark.asyncio
    async def test_a_settled_series_stops_early_with_outcome_settled(self, tmp_path):
        run = _settle_runner(tmp_path, _fit_stream([_flat_round([1, 2, 3])]))
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_SETTLED
        # Three rounds is the window width, but MIN_POINTS_FOR_TAU is the floor:
        # the criterion is satisfied on round 3 and the setpoint keeps going to 5,
        # because a 3-point series is one `fit_equilibration` refuses for tau.
        assert row["settle_rounds_run"] == MIN_POINTS_FOR_TAU
        assert row["settle_rounds_ceiling"] == 8
        assert row["settle_participating"] == [1, 2, 3]
        assert row["n_rounds"] == MIN_POINTS_FOR_TAU
        assert len(payload["points"]) == 3 * MIN_POINTS_FOR_TAU

    @pytest.mark.asyncio
    async def test_a_never_settling_series_runs_to_the_ceiling_with_outcome_ceiling(
            self, tmp_path):
        # sigma still moving 40% a round: evaluable, and the answer is no.
        moving = [_flat_round([1, 2, 3], sigma=1.0e-4 * (1.4 ** i)) for i in range(8)]
        run = _settle_runner(tmp_path, _fit_stream(moving))
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_CEILING
        assert row["settle_rounds_run"] == 8
        assert row["settle_max_deviation_rel"] > 0.10

    @pytest.mark.asyncio
    async def test_a_board_of_railed_channels_never_settles_and_is_not_evaluable(
            self, tmp_path):
        # THE critical case, end to end. A fit railed at the circuit model's R1
        # bound returns sigma = 0.5 S/cm every round with success = 1 -- a perfect
        # constant, and a constant is trivially "settled". If it participated,
        # four dead channels would shorten every setpoint of the run.
        railed = [{ch: (0.5, 100.0) for ch in (1, 2, 3)}]
        run = _settle_runner(tmp_path, _fit_stream(railed))
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_NOT_EVALUABLE
        assert row["settle_rounds_run"] == 8, "it must run to the ceiling"
        assert row["settle_participating"] == []
        assert set(row["settle_excluded"].values()) == {"railed_R1"}

    @pytest.mark.asyncio
    async def test_too_few_participating_channels_is_not_evaluable_and_hits_the_ceiling(
            self, tmp_path):
        # Two good channels against settle_min_channels = 3. Perfectly flat, and
        # still not enough evidence to shorten a hold.
        thin = [{1: (2.0e-4, 5.0e3), 2: (2.0e-4, 5.0e3), 3: (None, None)}]
        run = _settle_runner(tmp_path, _fit_stream(thin))
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_NOT_EVALUABLE
        assert row["settle_rounds_run"] == 8
        # The two that did participate are still named -- the setpoint could not
        # be judged on two channels, which is not the same as nothing being there.
        assert row["settle_participating"] == [1, 2]
        assert row["settle_excluded"] == {"3": "absent"}

    @pytest.mark.asyncio
    async def test_the_floor_is_respected_even_when_the_window_is_settled_at_once(
            self, tmp_path):
        # A 100 s cycle and a 450 s floor: the criterion is satisfied on round 3
        # and the setpoint still holds until the floor has elapsed.
        run = _settle_runner(tmp_path, _fit_stream([_flat_round([1, 2, 3])]),
                             min_hold_first_s=450.0)
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_SETTLED
        assert row["settle_rounds_run"] > MIN_POINTS_FOR_TAU
        assert row["settle_held_s"] >= 450.0

    @pytest.mark.asyncio
    async def test_the_first_setpoint_of_the_run_uses_the_first_setpoint_floor(
            self, tmp_path):
        # "First" is first of the RUN, not first of each leg: the down leg's
        # opening setpoint re-visits a temperature the films have already seen.
        run = _settle_runner(
            tmp_path, _fit_stream([_flat_round([1, 2, 3])]),
            temperatures_C=[45.0, 65.0], legs=("up", "down"),
            min_hold_first_s=450.0, min_hold_s=0.0)
        payload = await run.run()

        floors = [row["settle_floor_s"] for row in payload["setpoints"]]
        assert floors == [450.0, 0.0, 0.0, 0.0]
        assert payload["setpoints"][0]["settle_rounds_run"] > MIN_POINTS_FOR_TAU
        assert payload["setpoints"][1]["settle_rounds_run"] == MIN_POINTS_FOR_TAU

    @pytest.mark.asyncio
    async def test_a_tolerance_below_the_noise_floor_is_reported_as_unachievable(
            self, tmp_path):
        # The measured floor was 5.98% median with 22 of 96 series above 20%, so
        # a 2% tolerance is unsatisfiable. Say so and let the ceiling stand --
        # widening the tolerance behind the operator would be the other option.
        noisy = [_flat_round([1, 2, 3], sigma=2.0e-4 * (1.0 + 0.10 * (i % 2)))
                 for i in range(8)]
        run = _settle_runner(tmp_path, _fit_stream(noisy), settle_tol_rel=0.02)
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_CEILING
        assert row["settle_tolerance_achievable"] is False
        assert row["settle_noise_floor_rel"] > 0.02

    @pytest.mark.asyncio
    async def test_the_settle_verdict_reaches_the_progress_stream_once_per_setpoint(
            self, tmp_path):
        run = _settle_runner(tmp_path, _fit_stream([_flat_round([1, 2, 3])]))
        events = _collect(run)
        await run.run()

        verdicts = [e for e in events if e.kind == EV_SETTLE_VERDICT]
        assert len(verdicts) == 1
        assert verdicts[0].verdict == VERDICT_MET
        assert "SETTLED" in verdicts[0].detail

    @pytest.mark.asyncio
    async def test_a_setpoint_that_gave_up_reads_differently_from_one_that_settled(
            self, tmp_path):
        railed = [{ch: (0.5, 100.0) for ch in (1, 2, 3)}]
        run = _settle_runner(tmp_path, _fit_stream(railed))
        events = _collect(run)
        await run.run()

        detail = [e for e in events if e.kind == EV_SETTLE_VERDICT][0].detail
        assert "NOT EVALUABLE" in detail
        assert "SETTLED" not in detail

    @pytest.mark.asyncio
    async def test_settle_disabled_runs_exactly_rounds_per_setpoint(self, tmp_path):
        # The regression: `settle_enabled = False` must reproduce the old
        # fixed-count behaviour exactly, flat sigma or not.
        run = _settle_runner(tmp_path, _fit_stream([_flat_round([1, 2, 3])]),
                             settle_enabled=False,
                             rounds_per_setpoint=MIN_POINTS_FOR_TAU)
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_DISABLED
        assert row["settle_rounds_run"] == MIN_POINTS_FOR_TAU
        assert len(payload["points"]) == 3 * MIN_POINTS_FOR_TAU

    @pytest.mark.asyncio
    async def test_an_unreadable_store_runs_to_the_ceiling_rather_than_settling(
            self, tmp_path):
        # A read failure must never be able to SHORTEN a hold. The default reader
        # is used here and the fake store has no connection at all.
        run = _settle_runner(tmp_path, None,
                             rounds_per_setpoint=MIN_POINTS_FOR_TAU)
        payload = await run.run()
        row = _setpoint(payload)

        assert row["settle_outcome"] == SETTLE_NOT_EVALUABLE
        assert row["settle_rounds_run"] == MIN_POINTS_FOR_TAU

    def test_a_one_round_settle_window_is_refused_by_the_config(self):
        # Any single value is within tolerance of itself, so a one-round window
        # would settle every setpoint on its first round.
        with pytest.raises(ValueError, match="settle_n_rounds"):
            EquilibrationConfig(settle_n_rounds=1).validate()

    def test_zero_participating_channels_is_refused_as_a_configuration(self):
        with pytest.raises(ValueError, match="settle_min_channels"):
            EquilibrationConfig(settle_min_channels=0).validate()


class TestTauFitMinimumIsTheRoundFloor:
    """The acquisition side must not be able to emit a series the analysis side
    refuses.

    Measured against the shipped defaults, the earliest a setpoint could stop was
    ``max(settle_n_rounds, ceil(min_hold/round_period))``, and at
    ``DEFAULT_ROUND_PERIOD_S = 660`` that is 3 rounds at the first setpoint and 3
    at every later one -- below :data:`MIN_POINTS_FOR_TAU` at *every* setpoint of
    a default run. A run that stopped there produced spectra nobody could fit a
    tau to, which is the one number the run exists to measure.
    """

    @pytest.mark.asyncio
    async def test_later_floor_at_660s_period_still_runs_the_tau_fit_minimum(
            self, tmp_path):
        # ceil(600 / 660) = 1 round of time floor, and the detection window is 3.
        # sigma is flat from the first round, so nothing but the fit minimum can
        # keep this setpoint going -- and it must.
        run = _settle_runner(
            tmp_path, _fit_stream([_flat_round([1, 2, 3])]),
            temperatures_C=[45.0, 65.0], legs=("up",), rounds_per_setpoint=15,
            round_period_s=660.0, min_hold_first_s=0.0, min_hold_s=600.0)
        payload = await run.run()
        later = _setpoint(payload, 1)

        assert later["settle_floor_s"] == 600.0
        assert later["settle_outcome"] == SETTLE_SETTLED
        assert later["settle_rounds_run"] == MIN_POINTS_FOR_TAU

    @pytest.mark.asyncio
    async def test_first_floor_at_660s_period_still_runs_the_tau_fit_minimum(
            self, tmp_path):
        # ceil(1500 / 660) = 3 rounds, which is also the detection window -- so
        # before the coupling the FIRST setpoint, the one carrying essentially the
        # whole transient, could stop two rounds short of a fittable series.
        run = _settle_runner(
            tmp_path, _fit_stream([_flat_round([1, 2, 3])]),
            rounds_per_setpoint=15, round_period_s=660.0,
            min_hold_first_s=1500.0, min_hold_s=600.0)
        payload = await run.run()
        first = _setpoint(payload)

        assert first["settle_floor_s"] == 1500.0
        assert first["settle_outcome"] == SETTLE_SETTLED
        assert first["settle_rounds_run"] == MIN_POINTS_FOR_TAU

    @pytest.mark.asyncio
    async def test_a_three_round_window_stays_three_rounds_wide_and_still_cannot_stop_early(
            self, tmp_path):
        # `settle_n_rounds` is a DETECTION WINDOW; how many points a tau needs is
        # a different question, and the operator's window is not rewritten to
        # answer it. The series moves for two rounds and is flat thereafter, so a
        # 3-round window is satisfied at round 4 and a silently-widened 5-round
        # one would not be satisfied at round 5 at all. Settling at exactly
        # MIN_POINTS_FOR_TAU is therefore only possible if the window stayed 3.
        drifting = [_flat_round([1, 2, 3], sigma=s) for s in
                    (1.0e-4, 2.0e-4, 3.0e-4, 3.0e-4, 3.0e-4)]
        run = _settle_runner(tmp_path, _fit_stream(drifting), settle_n_rounds=3,
                             rounds_per_setpoint=15)
        payload = await run.run()
        row = _setpoint(payload)

        assert run.config.settle_n_rounds == 3, "the operator's window was rewritten"
        assert row["settle_outcome"] == SETTLE_SETTLED
        assert row["settle_rounds_run"] == MIN_POINTS_FOR_TAU

    def test_a_ceiling_below_the_fit_minimum_is_refused_naming_both_numbers(self):
        # Unsatisfiable by construction: no setpoint in such a run could ever
        # yield a tau, so it is refused before the chamber is touched rather than
        # discovered the next morning.
        with pytest.raises(ValueError) as excinfo:
            EquilibrationConfig(rounds_per_setpoint=MIN_POINTS_FOR_TAU - 1).validate()

        message = str(excinfo.value)
        assert str(MIN_POINTS_FOR_TAU - 1) in message
        assert str(MIN_POINTS_FOR_TAU) in message
        assert "MIN_POINTS_FOR_TAU" in message
        EquilibrationConfig(rounds_per_setpoint=MIN_POINTS_FOR_TAU).validate()

    @pytest.mark.asyncio
    async def test_a_settled_series_is_long_enough_for_the_real_fitter_to_accept(
            self, tmp_path):
        # The test that proves the coupling achieves its purpose: the series a
        # settled setpoint actually produced is handed to `fit_equilibration`
        # itself, not to a restatement of its rule. One round shorter -- what the
        # run would have stopped at before -- is refused for exactly that reason.
        from softae.analysis.equilibration import (
            REFUSAL_TOO_FEW_POINTS,
            fit_equilibration,
        )

        decaying = [_flat_round([1, 2, 3],
                                sigma=3.0e-4 + 1.0e-4 * math.exp(-i / 1.2))
                    for i in range(15)]
        run = _settle_runner(tmp_path, _fit_stream(decaying),
                             rounds_per_setpoint=15)
        payload = await run.run()
        row = _setpoint(payload)
        assert row["settle_outcome"] == SETTLE_SETTLED

        times = [p["t_since_hold_s"] for p in payload["points"] if p["channel"] == 1]
        sigmas = [3.0e-4 + 1.0e-4 * math.exp(-i / 1.2) for i in range(len(times))]
        assert len(times) == row["settle_rounds_run"]

        accepted = fit_equilibration("exponential", times, sigmas, channel=1)
        assert accepted.refusal != REFUSAL_TOO_FEW_POINTS
        assert accepted.n_points >= MIN_POINTS_FOR_TAU

        one_round_shorter = fit_equilibration("exponential", times[:-1], sigmas[:-1],
                                              channel=1)
        assert one_round_shorter.refusal == REFUSAL_TOO_FEW_POINTS


class TestMscrIsolation:
    def test_two_runs_do_not_share_a_script_path(self):
        # On 2026-08-10 a test run rewrote channels 1-4's scripts at 19:31 while a
        # real run's scripts from 17:50 sat at the same fixed paths. The .mscr is
        # uploaded per round, so the swap would have taken effect mid-flight and
        # the rig would have measured with the test's sweep parameters while every
        # log line reported success.
        from softae.workflows.equilibration import mscr_path

        assert mscr_path(3, KIND_SERIES, "RUN_A") != mscr_path(3, KIND_SERIES, "RUN_B")
        assert "RUN_A" in mscr_path(3, KIND_SERIES, "RUN_A")

    def test_an_unattributed_caller_still_gets_its_own_path_not_the_shared_one(self):
        from softae.workflows.equilibration import mscr_path, round_outdir

        anonymous = mscr_path(3, KIND_SERIES)
        assert anonymous != mscr_path(3, KIND_SERIES, "RUN_A")
        assert anonymous.endswith("_ch3_series.mscr")
        # The same per-process token `round_outdir` falls back to, so an
        # unattributed round's script and its output land under one identity.
        token = Path(round_outdir()).name.replace("softae_eq_out_", "")
        assert token in Path(anonymous).name

    def test_the_round_workflow_points_each_step_at_this_runs_own_script(self):
        from softae.workflows.equilibration import mscr_path

        config = EquilibrationConfig(channels=[1, 2], temperatures_C=[45.0])
        workflow = build_round_workflow(config, leg="up", setpoint_index=0,
                                        round_index=0, run_id="RUN_A")
        paths = [step.params["mscrpath"] for step in workflow.setup]
        assert paths == [mscr_path(1, KIND_SERIES, "RUN_A"),
                         mscr_path(2, KIND_SERIES, "RUN_A")]
