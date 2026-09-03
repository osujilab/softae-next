"""The ``softae-equilibration`` CLI surface.

``run --execute`` drives the stage heater to 85 °C and the humidifier for an
unattended overnight run, and **nothing in the shipped configuration refuses
it** — 85 °C is far inside ``temp_max_C = 200.0``, ``validate_rh_setpoint`` is a
cap with no floor, and ``assert_hardware_armed`` covers
``("stage", "syringe", "piezo")`` only. The confirmation prompt this module
builds is therefore the only barrier, and these tests are what keep it one.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
import structlog

from softae.analysis.equilibration import MIN_POINTS_FOR_TAU
from softae.core.hardware_safety import HardwareNotArmedError
from softae.tools.equilibration import (
    EXIT_DECLINED,
    EXIT_FAILED,
    EXIT_OK,
    PLAN_SCHEMA,
    ProgressRenderer,
    _cmd_fit,
    _cmd_plan,
    _cmd_report,
    _cmd_run,
    _endorse,
    _open_store,
    build_config,
    build_parser,
    configure_logging,
    confirm_no_geometry,
    confirm_thermal,
    hms,
    main,
    reconciled_eta_s,
)
from softae.workflows.equilibration import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_DOWN_APPROACH_TIMEOUT_S,
    DEFAULT_RH_SETPOINT_PCT,
    DEFAULT_TAU_SETPOINTS,
    DEFAULT_TOLERANCE_C,
    ENV_OK,
    ENV_SKIPPED,
    EV_AMBIENT_RESTORED,
    EV_COST_WARNING,
    EV_HEARTBEAT,
    EV_HOLD_VERDICT,
    EV_ROUND_FINISHED,
    EV_SETPOINT_FINISHED,
    VERDICT_MET,
    VERDICT_UNMET,
    EquilibrationConfig,
    ProgressEvent,
    project_duration,
)


@pytest.fixture()
def project(monkeypatch, tmp_path):
    """Point the default store at a temporary project directory."""
    from softae.config import loader

    monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
    return tmp_path


def _args(*argv):
    return build_parser().parse_args(list(argv))


#: A complete electrode geometry, spliced into any ``run --execute`` invocation
#: that is testing something *else*. Without it ``confirm_no_geometry`` prompts
#: before anything is opened, and every such test would decline at a gate it was
#: not written to exercise.
GEOMETRY = ("--electrode-l-cm", "0.2", "--electrode-t-cm", "0.0175",
            "--electrode-w-cm", "0.2")


# ── Argument surface ─────────────────────────────────────────────────────────

class TestArgumentSurface:
    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_the_verbs_are_in_operator_order_and_read_only_ones_need_nothing(self):
        assert _args("plan").func is _cmd_plan
        assert _args("run").execute is False       # dry run is the DEFAULT
        assert _args("fit", "--run", "R1").model == "exponential"
        assert _args("report", "--run", "R1").tol_rel == pytest.approx(0.02)

    def test_fit_and_report_require_a_run_because_they_read_one(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["fit"])

    def test_the_shipped_design_is_the_default_so_the_operator_types_one_word(self):
        config = build_config(_args("plan"))
        assert config.channels == list(range(1, 17))
        assert config.temperatures_C == [27.5, 45.0, 65.0, 85.0]
        assert config.legs == ("up", "down")
        # 20 %RH, not 15: the flush basin holds water inside the heated
        # enclosure, so warming the chamber humidifies it and 15 is below what
        # the enclosure can deliver hot (measured PV 16.9-20.4 at 65 C, 19.5-23.2
        # at 85 C on 2026-08-11). Not a controls fault and not re-tunable.
        assert config.rh_setpoint_pct == pytest.approx(20.0)
        assert config.rh_setpoint_pct == pytest.approx(DEFAULT_RH_SETPOINT_PCT)
        assert config.rounds_per_setpoint == 15

    def test_the_shipped_chamber_defaults_are_the_ones_the_bench_ruled(self):
        config = build_config(_args("plan"))
        # 2.0 C, not 0.5: at 0.5 a 0.6 C dip graded a whole setpoint "hold not
        # met" on a chamber that wanders a few tenths.
        assert config.tolerance_C == pytest.approx(2.0) == DEFAULT_TOLERANCE_C
        # And it still sits strictly inside the excursion warning, which sits
        # strictly inside the runaway guard. Ordering these wrongly would either
        # warn on every in-band sample or abort before warning once.
        assert config.tolerance_C < config.warn_C < config.fault_C
        # Cooling is passive; the descending allowance is the longer one.
        assert config.approach_timeout_s == pytest.approx(DEFAULT_APPROACH_TIMEOUT_S)
        assert config.down_approach_timeout_s == pytest.approx(
            DEFAULT_DOWN_APPROACH_TIMEOUT_S)
        assert config.down_approach_timeout_s > config.approach_timeout_s
        assert config.tau_setpoints == DEFAULT_TAU_SETPOINTS == 2

    def test_an_unknown_relaxation_model_is_rejected_by_the_parser_itself(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["fit", "--run", "R1", "--model", "magic"])


# ── The store ────────────────────────────────────────────────────────────────

class TestStoreSelection:
    def test_a_mock_run_writes_to_an_isolated_store_and_never_to_the_project_store(
            self, project):
        args = _args("run", "--mock")
        store, path = _open_store(args)
        try:
            assert Path(path) == project / "real" / "mock"
        finally:
            store.close()

    def test_a_real_run_writes_to_the_store_everything_else_uses(self, project):
        store, path = _open_store(_args("run"))
        try:
            assert Path(path) == project / "real"
        finally:
            store.close()


# ── run: the gates ───────────────────────────────────────────────────────────

class TestRunGates:
    def test_run_without_execute_prints_the_plan_and_touches_no_instrument(
            self, monkeypatch, capsys):
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", _never("create_manager"))
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))

        assert _cmd_run(_args("run", "--channels", "1-16")) == EXIT_OK
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "WHOLE RUN" in out          # the projection is printed, not hidden

    def test_run_asserts_the_arming_interlock_before_a_store_is_opened(
            self, monkeypatch):
        # Reaching the executor first would leave an empty experiments row behind
        # on every declined attempt.
        import softae.core.hardware_safety as safety
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", lambda **_kw: object())
        monkeypatch.setattr(safety, "assert_hardware_armed", _raise_not_armed)
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))
        monkeypatch.setattr(tool, "confirm_thermal", _never("confirm_thermal"))

        assert _cmd_run(_args("run", "--channels", "1-16", *GEOMETRY, "--execute")) \
            == EXIT_DECLINED

    def test_the_thermal_confirmation_is_asked_separately_because_the_arming_interlock_covers_motion_only(
            self, monkeypatch):
        from softae.core.hardware_safety import MOTION_INSTRUMENTS

        # The mechanical fact the CLI is built around: probe_motion sees neither
        # controller, so the assert passes unconditionally on this run.
        assert "temp_controller" not in MOTION_INSTRUMENTS
        assert "rh_controller" not in MOTION_INSTRUMENTS

        import softae.core.hardware_safety as safety
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", lambda **_kw: object())
        monkeypatch.setattr(safety, "assert_hardware_armed", lambda *_a, **_kw: None)
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))
        monkeypatch.setattr("builtins.input", lambda *_a: "y")

        # The interlock passed. The confirmation is what stops it.
        assert _cmd_run(_args("run", "--channels", "1-16", *GEOMETRY, "--execute")) \
            == EXIT_DECLINED

    def test_the_confirmation_states_the_peak_temperature_and_the_projected_duration(
            self, capsys):
        config = build_config(_args("run", "--channels", "1-16"))
        assert confirm_thermal(config, reader=lambda _p: "yes") is True
        out = capsys.readouterr().out
        assert "85 C" in out
        assert "hours, unattended" in out

    def test_a_reflex_keypress_does_not_start_a_nine_hour_heat(self, capsys):
        config = build_config(_args("run", "--channels", "1-16"))
        assert confirm_thermal(config, reader=lambda _p: "y") is False
        assert confirm_thermal(config, reader=lambda _p: "") is False
        assert "Declined" in capsys.readouterr().out

    def test_yes_skips_the_prompt_for_a_scripted_start(self, monkeypatch):
        monkeypatch.setattr("builtins.input", _never("input"))
        config = build_config(_args("run", "--channels", "1-16"))
        assert confirm_thermal(config, assume_yes=True) is True

    def test_confirm_thermal_states_the_channel_selection_before_the_chamber_moves(
            self, capsys):
        # The last thing a human reads before the heater is driven. A wrong
        # --channels is otherwise invisible until the wrong samples are measured.
        config = build_config(_args("run", "--channels", "1-16"))
        confirm_thermal(config, assume_yes=True)
        out = capsys.readouterr().out
        assert "CHANNELS DRIVEN: 1-16 (16)" in out

    def test_confirm_thermal_renders_a_subset_as_the_subset_not_as_the_full_board(
            self, capsys):
        # `channels[0]-channels[-1]` would print "1-16" for this selection and tell
        # the operator four excluded samples are in the run.
        config = build_config(_args("run", "--channels", "1-3,8-16"))
        confirm_thermal(config, assume_yes=True)
        out = capsys.readouterr().out
        assert "CHANNELS DRIVEN: 1-3,8-16 (12)" in out
        assert "1-16" not in out


# ── run: the channel selection cannot be inherited from `plan` ───────────────

class TestRunRequiresChannels:
    def test_run_without_channels_refuses_and_names_the_flag(
            self, monkeypatch, capsys):
        # `plan` and `run` are separate processes. A default here would restore
        # the channels the operator removed at plan time -- and drive them.
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", _never("create_manager"))
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))
        monkeypatch.setattr(tool, "confirm_thermal", _never("confirm_thermal"))

        assert _cmd_run(_args("run", "--execute")) == EXIT_FAILED
        captured = capsys.readouterr()
        assert "--channels" in captured.err
        assert "share no state" in captured.err
        assert "WHOLE RUN" not in captured.out    # nothing was planned or printed

    def test_run_without_channels_refuses_on_the_dry_run_too(
            self, monkeypatch, capsys):
        # A dry run that models a different channel set than the real run is worse
        # than no dry run: it is read as confirmation.
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", _never("create_manager"))
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))

        assert _cmd_run(_args("run")) == EXIT_FAILED
        captured = capsys.readouterr()
        assert "--channels" in captured.err
        assert "Dry run" not in captured.out

    def test_run_declares_no_default_channel_set_but_plan_keeps_one(self):
        # The single declared difference between the two design surfaces.
        assert _args("run").channels is None
        assert _args("plan").channels == "1-16"

    def test_build_config_on_run_without_channels_raises_rather_than_attribute_errors(
            self):
        from softae.tools.equilibration import ChannelsNotStated

        with pytest.raises(ChannelsNotStated):
            build_config(_args("run"))

    def test_plan_without_channels_still_designs_the_whole_board(
            self, project, capsys):
        # `plan` opens nothing and heats nothing; a zero-argument plan is how the
        # shipped design is read.
        assert _cmd_plan(_args("plan")) == EXIT_OK
        out = capsys.readouterr().out
        assert "channels:     1-16 (16)" in out

    def test_a_subset_on_run_survives_into_the_round_workflow_unchanged(self):
        # The plumbing the refusal protects: whatever `run` is given is what the
        # DAG measures -- these channels and no others.
        from softae.workflows.equilibration import build_round_workflow

        config = build_config(_args("run", "--channels", "1-3,8-16"))
        wf = build_round_workflow(config, leg="up", setpoint_index=0, round_index=0)

        assert len(wf.setup) == 12
        assert [step.params["chan"] for step in wf.setup] == \
            [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def _never(what):
    def _boom(*_a, **_kw):
        raise AssertionError(f"{what} must not be reached here")
    return _boom


# ── run: what the interrupt message is allowed to claim ──────────────────────

class _InterruptedRunner:
    """Stands in for :class:`EquilibrationRun`: a Ctrl-C, and a teardown outcome.

    The workflow tests pin that the teardown *runs*; these pin that the CLI tells
    the operator what it did. So the outcome is set as plain state here rather
    than provoked through a fake rig — the two facts the message rests on are
    ``restore_error`` and ``sidecar_written``, and both are read, not inferred.
    """

    def __init__(self, config, _manager, *, data_store=None, run_id=None, **_kw):
        self.config = config
        self.run_id = run_id
        self.on_progress = None
        self.points: list = []
        self.holds: list = []
        self.progress_failures = 0
        self.restored_ambient = True
        self.restore_error = ""
        self.last_commanded_C = 85.0
        self.sidecar_written = True
        self.sidecar_error = ""

    def last_commanded_description(self) -> str:
        return f"{self.last_commanded_C:g} C"

    def sidecar_path(self):
        return Path("runs") / str(self.run_id) / "equilibration.json"

    def measured_cost_summary(self) -> dict:
        return {}

    async def run(self):
        raise KeyboardInterrupt


def _arrange_interrupt(monkeypatch, **outcome):
    """Wire ``_cmd_run`` up to a runner that is interrupted with *outcome*."""
    import softae.core.hardware_safety as safety
    import softae.drivers.factory as factory
    import softae.tools.equilibration as tool

    class _Manager:
        async def connect_all(self):
            return None

        async def disconnect_all(self):
            return None

    monkeypatch.setattr(factory, "create_manager", lambda **_kw: _Manager())
    monkeypatch.setattr(safety, "assert_hardware_armed", lambda *_a, **_kw: None)
    monkeypatch.setattr(tool, "confirm_thermal", lambda *_a, **_kw: True)

    def _make(config, manager, **kw):
        runner = _InterruptedRunner(config, manager, **kw)
        for key, value in outcome.items():
            setattr(runner, key, value)
        return runner

    monkeypatch.setattr(tool, "EquilibrationRun", _make)


class TestInterruptMessage:
    """The Ctrl-C branch says what happened, not what is usually true.

    It used to print ``"Interrupted — partial rounds are recorded."`` on a path
    where neither the ambient restore nor the sidecar write had run at all.
    """

    def test_the_interrupt_message_reports_a_chamber_that_did_come_down(
            self, monkeypatch, project, capsys):
        _arrange_interrupt(monkeypatch)
        assert _cmd_run(_args("run", "--channels", "1-2", *GEOMETRY, "--execute",
                              "--mock")) == EXIT_FAILED
        err = capsys.readouterr().err

        assert "Interrupted" in err
        assert "Ambient restored" in err
        assert "Sidecar written" in err
        assert "partial rounds are recorded" not in err

    def test_the_interrupt_message_reports_a_chamber_that_did_not_come_down(
            self, monkeypatch, project, capsys):
        _arrange_interrupt(
            monkeypatch, restored_ambient=False, sidecar_written=False,
            restore_error="stage heater not returned to ambient: OSError: no reply",
            sidecar_error="OSError: the disk is full")
        assert _cmd_run(_args("run", "--channels", "1-2", *GEOMETRY, "--execute",
                              "--mock")) == EXIT_FAILED
        err = capsys.readouterr().err

        # It must be on the console, not only in structlog: this is the one line
        # that decides whether the operator walks away from a hot chamber.
        assert "AMBIENT WAS NOT RESTORED" in err
        assert "85 C" in err                      # the setpoint it is still at
        assert "CHECK IT AT THE RIG" in err
        assert "THE SIDECAR WAS NOT WRITTEN" in err
        assert "exist nowhere else" in err
        assert "Ambient restored" not in err


def _raise_not_armed(*_a, **_kw):
    raise HardwareNotArmedError("not armed")


# ── plan ─────────────────────────────────────────────────────────────────────

class TestPlan:
    def test_plan_is_read_only_and_prints_both_duration_columns(self, project, capsys):
        assert _cmd_plan(_args("plan")) == EXIT_OK
        out = capsys.readouterr().out
        assert "typical" in out and "worst case" in out
        # Every figure is a RANGE. `--rounds` is a ceiling, so a single number
        # would be a promise the run cannot keep in either direction: the floor
        # is every setpoint settling at its earliest and the ceiling is none of
        # them settling at all.
        #
        # Read off the projection rather than written as literals. The hours here
        # rest on `estimate_eis_duration`, which lives in core/preflight.py and is
        # recalibrated from time to time (it ran ~10x low until 2026-08); pinning
        # a number would make this test a hostage to that constant instead of a
        # statement about what `plan` prints.
        projection = project_duration(build_config(_args("plan")))
        assert (f"{projection.per_setpoint_typical_floor_s / 3600:.2f}-"
                f"{projection.per_setpoint_typical_s / 3600:.2f} h") in out
        assert (f"{projection.typical_floor_s / 3600:.2f}-"
                f"{projection.typical_s / 3600:.2f} h") in out
        assert (f"{projection.worst_floor_s / 3600:.2f}-"
                f"{projection.worst_case_s / 3600:.2f} h") in out
        assert projection.typical_floor_s < projection.typical_s
        assert "anchor" not in out.lower()

    def test_plan_prints_the_effective_minimum_rounds_for_every_regime(
            self, project, capsys):
        # Three floors now, not two, and they are printed rather than their
        # minimum: the run's first setpoint has `min_hold_first_s`, the rest of
        # the --tau-setpoints window has the fitter's MIN_POINTS_FOR_TAU, and
        # everything after has neither. An operator reading a 3-round setpoint
        # beside a 5-round one must be able to see which regime each was in.
        assert _cmd_plan(_args("plan")) == EXIT_OK
        out = capsys.readouterr().out

        # Read off the projection rather than written as literals. These floors
        # are ceil(hold / period), and the period is DERIVED from the preset's
        # cost -- it moved 200 -> 370 s when `Quick`'s floor moved to 7 Hz, which
        # took the first setpoint from 8 rounds down to 5. A literal here would
        # pin the arithmetic of one preset instead of the property being tested:
        # that all three regimes are named and distinguishable.
        projection = project_duration(build_config(_args("plan")))
        assert (f"effective minimum {projection.min_rounds_first} rounds at "
                f"setpoint 1 of the run") in out
        assert f"The first 2 setpoint(s) may not stop under " \
               f"{MIN_POINTS_FOR_TAU} rounds" in out
        assert "MIN_POINTS_FOR_TAU" in out
        # Past the tau window the floor is --settle-n-rounds alone.
        assert f"{projection.min_rounds_later} after that" in out
        assert (f"a setpoint runs {projection.min_rounds_later}-15 rounds") in out
        assert "/".join(str(r) for r in projection.floor_rounds) + \
               " rounds in run order, of 15" in out
        # The tau floor still binds SOMEWHERE, which is the only reason the
        # middle regime exists: at the 370 s period the first setpoint's 1500 s
        # hold buys ceil(1500/370) = 5 rounds and no longer overtakes the fit
        # minimum, so the two coincide rather than the hold dominating.
        assert projection.min_rounds_first >= MIN_POINTS_FOR_TAU
        assert projection.min_rounds_tau == MIN_POINTS_FOR_TAU

    def test_plan_states_which_setpoints_the_tau_floor_applies_to(
            self, project, capsys):
        # `--tau-setpoints 0` removes it everywhere, and the plan must say so
        # rather than quietly printing a shorter floor: a run that guarantees no
        # tau anywhere is a legitimate thing to ask for and a terrible thing to
        # get by accident.
        assert _cmd_plan(_args("plan", "--tau-setpoints", "0")) == EXIT_OK
        out = capsys.readouterr().out

        assert "--tau-setpoints 0" in out
        assert "applies NOWHERE" in out
        assert "a setpoint runs 3-15 rounds" in out

    def test_plan_warns_that_a_missing_geometry_means_every_sigma_is_null(
            self, project, capsys):
        _cmd_plan(_args("plan"))
        out = capsys.readouterr().out
        assert "NO electrode geometry" in out
        assert "sigma(t) at all" in out

    def test_plan_names_the_next_most_valuable_action(self, project, capsys):
        _cmd_plan(_args("plan", *GEOMETRY))
        out = capsys.readouterr().out
        assert "Next most valuable action" in out
        # It points at a saved plan rather than at a hand-copied flag list: the
        # flag list is what silently lost --preset and the geometry.
        assert "--save equilibration_plan.toml" in out
        assert "run --from-plan equilibration_plan.toml --execute" in out

    def test_plan_notes_the_absent_recorded_thickness_without_blaming_it_for_null_sigma(
            self, project, capsys):
        # Without geometry sigma IS null -- but because record_fit is handed no
        # t_cm, not because a `measured_thickness` row is missing. P.11 governs
        # make_thickness_lookup and tab_analysis; neither is in this path.
        _cmd_plan(_args("plan", "--channels", "1-4"))
        out = capsys.readouterr().out
        assert "no recorded thickness" in out
        assert "1-4" in out
        assert "sigma(t) at all" in out          # the real NULL-sigma message stands
        assert "P.11" not in out
        assert "refused" not in out

    def test_plan_with_geometry_does_not_claim_sigma_is_refused_for_manually_cast_films(
            self, project, capsys):
        # The defect this replaces: `plan --electrode-t-cm ...` printed "sigma will
        # be refused for these (P.11)" for every channel with no `measured_thickness`
        # row. record_fit computes sigma from the L/t/w it is handed by
        # router.handle and consults no table, so sigma is computed for all of them
        # -- and the warning would have sent the operator to reconfigure a run that
        # was going to work.
        _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2",
                        "--electrode-t-cm", "0.0175", "--electrode-w-cm", "0.2"))
        out = capsys.readouterr().out

        assert "refused" not in out
        assert "P.11" not in out
        assert "sigma IS computed for every one of them" in out

    def test_plan_with_geometry_says_the_thickness_is_uniform_and_operator_attributed(
            self, project, capsys):
        # One number divided into 16 channels: per-channel variation is not
        # captured, and 'target' is a hand-computed twin figure, not a measurement.
        _cmd_plan(_args("plan", "--electrode-l-cm", "0.2", "--electrode-t-cm",
                        "0.0175", "--electrode-w-cm", "0.2"))
        out = capsys.readouterr().out

        assert "175 um" in out
        assert "OPERATOR-SUPPLIED and UNIFORM across all 16 channel(s)" in out
        assert "per-channel variation" in out and "is not captured" in out
        assert "--thickness-method 'target'" in out

    def test_plan_says_when_the_supplied_thickness_overrides_a_recorded_one(
            self, project, capsys):
        # An operator must know when their flag is silently winning over stored
        # data -- record_fit takes the params and never looks the row up.
        from softae.config import loader
        from softae.core.data_store import DataStore

        store = DataStore(str(project / "real"), db_filename=loader.data_db_filename())
        store.record_thickness(2, 210.0, instrument="profilometer")
        store.record_thickness(3, 205.0, instrument="profilometer")
        store.close()

        _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2",
                        "--electrode-t-cm", "0.0175", "--electrode-w-cm", "0.2"))
        out = capsys.readouterr().out

        assert "OVERRIDES the recorded thickness on file for channel(s) 2,3" in out
        assert "refused" not in out

    def test_plan_with_geometry_says_there_is_nothing_to_compare_when_no_row_exists(
            self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2",
                        "--electrode-t-cm", "0.0175", "--electrode-w-cm", "0.2"))
        out = capsys.readouterr().out
        assert "No channel here has a recorded thickness to compare it against" in out
        assert "OVERRIDES" not in out

    def test_a_malformed_channel_spec_is_reported_not_raised(self, project, capsys):
        assert _cmd_plan(_args("plan", "--channels", "banana")) == EXIT_FAILED

    def test_plan_prints_the_channels_that_are_actually_in_the_run(
            self, project, capsys):
        # `channels[0]-channels[-1]` renders 1-3,8-16 as "1-16" -- a display that
        # tells the operator four samples are in the run when they are not.
        _cmd_plan(_args("plan", "--channels", "1-3,8-16"))
        out = capsys.readouterr().out
        assert "1-3,8-16" in out
        assert "channels:     1-16" not in out
        assert "(12)" in out

    def test_plan_cautions_when_the_period_fits_the_model_but_not_the_models_error(
            self, project, capsys):
        # The regime the caution exists for, and the only one: 16 channels on
        # 'Standard' model ~604 s, so a 620 s period DOES contain the modelled
        # round -- but not the ~8 % the model may be under it by, and the bench
        # number for that preset is 651 s. `plan` must not let 620 read as safe.
        _cmd_plan(_args("plan", "--preset", "Standard", "--round-period-s", "620"))
        out = capsys.readouterr().out
        assert "FITTED to presets" in out
        assert "s/channel of margin" in out
        assert "CAUTION" in out
        assert "--round-period-s" in out
        # Not the harder statement -- the round does fit the model.
        assert "UNACHIEVABLE" not in out

    def test_a_round_that_does_not_fit_at_all_is_not_also_hedged_about(
            self, project, capsys):
        # 604 s of modelled round against a 120 s period. Two verdicts of
        # different strength on one question ("may not fit" directly above
        # "UNACHIEVABLE") teaches the reader that neither is meant literally, so
        # the margin caution stands down once the round plainly does not fit.
        _cmd_plan(_args("plan", "--preset", "Standard", "--round-period-s", "120"))
        out = capsys.readouterr().out
        assert "UNACHIEVABLE" in out
        assert "CAUTION" not in out
        assert "Minimum feasible --round-period-s 600" in out

    def test_the_default_round_period_contains_a_real_round_at_the_default_channels(
            self, project, capsys):
        # The defect this default closes: --round-period-s was 120 s while
        # --channels defaulted to all 16, which costs ~168 s even on the fastest
        # preset. Someone accepting every default got a run that could not honour
        # its own sampling interval.
        from softae.tools.equilibration import DEFAULT_ROUND_PERIOD_S
        from softae.workflows.equilibration import (
            ROUND_BUFFER_S,
            EquilibrationConfig,
            round_cost_s,
        )

        config = EquilibrationConfig()
        assert config.round_period_s == DEFAULT_ROUND_PERIOD_S
        # Against the cost the period is DERIVED from, whichever branch supplied
        # it. It used to be indexed straight out of EIS_MEASURED_S_PER_CHANNEL,
        # which broke the moment `Quick`'s reading was retired with its 20 Hz
        # floor -- and the invariant was never about that dict, it is that the
        # shipped period contains the shipped round with the buffer to spare.
        assert round_cost_s(config) <= config.round_period_s - ROUND_BUFFER_S

        _cmd_plan(_args("plan"))
        out = capsys.readouterr().out
        # SILENT on both counts, and that is the property being pinned: `plan` is
        # the screen an operator reads before committing a night, and a tool that
        # cautions on its own shipped configuration teaches them to ignore it.
        #
        # It briefly did caution here, because the period was derived from the
        # measurement alone (16 x 40.7 s -> 660 s) and left 8.8 s of slack, less
        # than the sweep model's own ~8 % error. The period is now the measured
        # round plus a stated buffer, which clears that error at the shipped
        # preset. The threshold that fires the caution is derived from the model's
        # calibration rather than being the flat 10 s/channel it was when the
        # model ran ~10x low -- see `model_underestimate_frac`.
        assert "UNACHIEVABLE" not in out
        assert "CAUTION" not in out

    def test_plan_does_not_caution_when_the_period_is_generous(self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", "--round-period-s", "600"))
        out = capsys.readouterr().out
        assert "CAUTION" not in out

    def test_the_anchor_preset_flag_is_gone_because_the_concept_is(self):
        # An anchor round at the series preset is byte-identical to a series
        # round; at Longest it cost ~503 s/channel modelled -- never timed here --
        # to reach 0.2 Hz. Cost is the live reason it is gone: the ~9 Hz phase
        # floor once cited beside it rested on a Z_phi ceiling that
        # analysis/eis/envelope.py has withdrawn.
        with pytest.raises(SystemExit):
            build_parser().parse_args(["plan", "--anchor-preset", "Longest"])


class TestPlanningFromAMeasuredCost:
    """``--measured-per-channel-s`` is how a bench number gets into the projection,
    and the modelled path must not read like a prediction beside it.

    Written when the model was ~10x low and a plan resting on it told the operator
    a 240 s period was fine for a round that really took 488 s. The model has since
    been refitted and is ~8 % out, so what these tests now pin is that the two
    paths stay distinguishable and stay labelled -- not that the gap is large."""

    #: The operator's real run: 12 channels, 240 s period, 'Standard' at the
    #: 40.7 s/channel it measured. ``--preset`` is now typed rather than defaulted:
    #: the default is 'Quick', and 40.7 s/channel is a 'Standard' number.
    OPERATOR = ("plan", "--channels", "1-3,8-16", "--round-period-s", "240",
                "--preset", "Standard")

    def test_a_measured_per_channel_cost_changes_the_projected_total_duration(
            self, project, capsys):
        _cmd_plan(_args(*self.OPERATOR))
        modelled = _whole_run_h(capsys.readouterr().out)

        _cmd_plan(_args(*self.OPERATOR, "--measured-per-channel-s", "40.7"))
        out = capsys.readouterr().out
        measured = _whole_run_h(out)

        # 15 rounds x 518 s against 15 x 483 s: still a different night, but a
        # much smaller difference than when this was written. The modelled round
        # was ~47 s then and is ~453 s now, because the sweep model was
        # recalibrated against the bench; what is left is the ~3 s/channel of
        # mux/upload/retrieval the model still does not carry, not an order of
        # magnitude. An hour is well past a rounding difference at this scale.
        assert measured > modelled + 1.0
        assert "MEASURED" in out
        assert "40.7s/channel" in out

    def test_the_measured_cost_reaches_the_round_cost_the_gap_and_the_interval(
            self, project, capsys):
        _cmd_plan(_args(*self.OPERATOR, "--measured-per-channel-s", "40.7"))
        out = capsys.readouterr().out

        assert "Standard      8.1 min (40.7s/channel)" in out   # the round cost
        assert "vs MEASURED 488s leaves -20.7s/channel" in out  # the headroom
        assert "one round costs 488s" in out
        assert "Sampling interval 518s" in out    # the round + the 30 s gap floor
        # The modelled per-channel figure for this config is 3.9 s; it must not
        # appear anywhere once a measured one was given.
        assert "3.9s/channel" not in out

    def test_plan_states_the_configured_period_is_unachievable_at_the_measured_cost(
            self, project, capsys):
        _cmd_plan(_args(*self.OPERATOR, "--measured-per-channel-s", "40.7"))
        out = capsys.readouterr().out

        assert "UNACHIEVABLE" in out
        assert "--round-period-s 490" in out, "the minimum feasible period is not given"
        # The operator-facing consequence: the sampling interval sets the shortest
        # resolvable equilibration time constant.
        assert "resolves tau no shorter than ~17 min" in out

    def test_plan_does_not_call_a_period_unachievable_when_the_round_fits(
            self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", "--round-period-s", "600",
                        "--measured-per-channel-s", "40.7"))
        out = capsys.readouterr().out
        assert "UNACHIEVABLE" not in out
        assert "Sampling interval 600s" in out

    def test_a_modelled_plan_is_labelled_modelled_and_says_the_bench_is_higher(
            self, project, capsys):
        # Requirement: never print a modelled number bare, as if it were a
        # prediction. This is the label that stops that.
        #
        # Reaching the modelled branch now takes an off-grid sweep. Since the
        # 2026-08-17 bench run every shipped preset is anchored, so `--f-lo-mHz`
        # is what leaves the timed grids behind -- which is also the honest
        # shape of the remaining risk: custom sweeps, not stock ones.
        _cmd_plan(_args(*self.OPERATOR, "--f-lo-mHz", "700"))
        out = capsys.readouterr().out

        assert "basis: MODELLED" in out
        # No longer "a FLOOR" and no longer "SEVERAL TIMES higher": both were true
        # of a model that ran ~10x low and are false of one refitted to the bench.
        # What must survive is the label and the size of the doubt -- a modelled
        # figure is never printed bare, and the note quotes the model's own fitted
        # error rather than leaving the reader to guess at it.
        assert "fitted to the" in out
        assert "UNDER a real round" in out
        assert "10%" in out
        # Derived, not "37.19": the literal here is what let the advice text go on
        # quoting 40.7 after the retune moved the grid that number was timed at.
        from softae.tools.equilibration import MEASURED_PER_CHANNEL_S_STANDARD

        assert f"--measured-per-channel-s {MEASURED_PER_CHANNEL_S_STANDARD:g}" in out

    def test_an_anchored_preset_reads_as_measured_with_no_flag_typed(
            self, project, capsys):
        """The other half of the 2026-08-17 change, and the reason ``_print_basis``
        grew a third branch: the cost really did come from a stopwatch, so calling
        it MODELLED would understate it and push the operator into re-supplying a
        number the system already holds."""
        _cmd_plan(_args(*self.OPERATOR))
        out = capsys.readouterr().out

        assert "basis: MEASURED" in out
        assert "timed on this rig" in out
        assert "basis: MODELLED" not in out

    def test_a_measured_plan_drops_the_modelled_caveats_rather_than_stacking_them(
            self, project, capsys):
        _cmd_plan(_args(*self.OPERATOR, "--measured-per-channel-s", "40.7"))
        out = capsys.readouterr().out
        assert "basis: MODELLED" not in out
        assert "NOTE: the model is FITTED" not in out

    def test_the_measured_cost_also_reaches_the_thermal_confirmation_prompt(
            self, capsys):
        # The last screen before an unattended heat must not quote a modelled
        # duration when a measured one was given on the same command line.
        config = build_config(_args("run", "--channels", "1-3,8-16",
                                    "--round-period-s", "240"))
        confirm_thermal(config, assume_yes=True)
        modelled = capsys.readouterr().out
        confirm_thermal(config, assume_yes=True, measured_per_channel_s=40.7)
        measured = capsys.readouterr().out

        # Floor-to-worst, not typical-to-worst: `--rounds` is a ceiling, so the
        # low end of what the operator is committing to on this screen is every
        # setpoint settling at its earliest, not a typical approach time.
        #
        # Both spans come from `project_duration` rather than being written out:
        # the modelled one rests on core/preflight.py's sweep calibration, and a
        # literal here would break every time that is re-fitted while saying
        # nothing about the banner.
        modelled_span = _span_h(project_duration(config))
        measured_span = _span_h(project_duration(
            config, measured_series_round_s=40.7 * len(config.channels)))
        assert modelled_span != measured_span
        assert modelled_span in modelled
        assert modelled_span not in measured
        assert measured_span in measured

    def test_a_nonpositive_measurement_falls_back_to_the_systems_own_basis(
            self, project, capsys):
        # The flag is an input to a projection, not a gate: a typo must not stop a
        # read-only command, but it must not be silently believed either.
        #
        # "falls back to the model" is now "falls back to whatever the system
        # would have said unaided", and for an anchored preset that is its own
        # stopwatch rather than the model. What is pinned is that the typed 0 is
        # discarded -- which the per-channel figure proves, since believing it
        # would print 0.0s/channel.
        assert _cmd_plan(_args(*self.OPERATOR, "--measured-per-channel-s", "0")) \
            == EXIT_OK
        out = capsys.readouterr().out
        assert "basis: MEASURED 37.2s/channel" in out
        assert "timed on this rig" in out
        assert "0.0s/channel" not in out


def _whole_run_h(out: str) -> float:
    """The typical whole-run CEILING, in hours, out of the projection table.

    The cell is a ``floor-ceiling`` range now that ``--rounds`` is a ceiling, so
    the ceiling is taken explicitly rather than by position: it is the figure that
    changes with the round cost in the way this comparison is about.
    """
    for line in out.splitlines():
        if "WHOLE RUN" in line:
            return float(line.split()[2].split("-")[-1])
    raise AssertionError("no WHOLE RUN row in the printed design")


def _span_h(projection) -> str:
    """The floor-to-worst hours ``confirm_thermal`` commits the operator to."""
    return (f"{projection.typical_floor_s / 3600:.1f}-"
            f"{projection.worst_case_s / 3600:.1f}")


class TestChannelSpecFormatting:
    def test_a_non_contiguous_selection_round_trips_through_the_formatter(self):
        from softae.core.channel_spec import format_channel_spec, parse_channel_spec

        for spec in ("1-3,8-16", "2,4,5-10", "7", "3,9"):
            channels = parse_channel_spec(spec)
            rendered = format_channel_spec(channels)
            assert parse_channel_spec(rendered) == channels

    def test_the_compact_form_is_the_one_a_reader_expects(self):
        from softae.core.channel_spec import format_channel_spec

        assert format_channel_spec([1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]) \
            == "1-3,8-16"
        # "2,4,5-10" parses to a contiguous 4..10, so the canonical form is
        # tighter than what was typed -- and still re-parses to the same set.
        assert format_channel_spec([2, 4, 5, 6, 7, 8, 9, 10]) == "2,4-10"
        assert format_channel_spec([7]) == "7"
        assert format_channel_spec([3, 9]) == "3,9"
        assert format_channel_spec([4, 5]) == "4,5"      # a run of two, written out
        assert format_channel_spec([]) == ""


# ── report ───────────────────────────────────────────────────────────────────

def _recorded_run(project_dir: Path, stats: list[dict],
                  session_drift: list[dict] | None = None) -> str:
    from softae.config import loader
    from softae.core.data_store import DataStore

    store = DataStore(str(project_dir), db_filename=loader.data_db_filename())
    run_id = store.start_run("equilibration_characterization", mode="characterization")
    store.close()
    path = project_dir / "runs" / run_id / "equilibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": "equilibration/1", "run_id": run_id, "points": [], "holds": [],
        "setpoints": [{"leg": "up", "setpoint_index": 0, "temperature_C": 27.5,
                       "hold_met": True, "rh_hold_met": False,
                       "hold_pv_min": 27.4, "hold_pv_max": 27.6,
                       "rh_hold_pv_min": 24.0, "rh_hold_pv_max": 26.0}],
        "aborted": False, "abort_reason": "", "stats": stats,
        "session_drift": session_drift or [],
        "thickness": {"t_cm": 0.02, "value_um": 200.0, "units": "um",
                      "thickness_method": "target",
                      "recorded_in_fit_results": False,
                      "note": "fit_results.thickness_method is NULL for this run"},
    }, indent=2), encoding="utf-8")
    return run_id


def _stats(noise_floor_rel):
    return [{"channel": 1, "leg": "up", "setpoint_index": 0, "model": "exponential",
             "tau_s": 640.0, "tau_stderr_s": 12.0, "t_tol_s": 2400.0, "tol_rel": 0.02,
             "sigma_settled": 1e-4, "noise_floor_rel": noise_floor_rel,
             "noise_floor_is_upper_bound": True, "r_squared": 0.999, "n_points": 15,
             "fit_success": True, "refusal": ""}]


class TestReport:
    def test_report_refuses_to_endorse_a_conditioning_tolerance_below_the_measured_noise_floor(
            self, project, capsys):
        run_id = _recorded_run(project / "real", _stats(0.05))
        code = _cmd_report(_args("report", "--run", run_id, "--tol-rel", "0.01"))
        out = capsys.readouterr().out

        assert code == EXIT_FAILED, "a script must not be able to ignore this"
        assert "REFUSED" in out
        assert "can never be satisfied" in out
        assert "No hold time is printed for these" in out

    def test_report_endorses_a_tolerance_above_the_floor(self, project, capsys):
        run_id = _recorded_run(project / "real", _stats(0.005))
        assert _cmd_report(_args("report", "--run", run_id, "--tol-rel", "0.02")) \
            == EXIT_OK
        assert "endorsed" in capsys.readouterr().out

    def test_report_prints_the_hold_verdicts_because_an_unmet_setpoint_is_the_result(
            self, project, capsys):
        run_id = _recorded_run(project / "real", _stats(0.005))
        _cmd_report(_args("report", "--run", run_id))
        out = capsys.readouterr().out
        assert "T held=yes" in out
        assert "RH held=NO" in out

    def test_report_labels_the_noise_floor_an_upper_bound(self, project, capsys):
        run_id = _recorded_run(project / "real", _stats(0.005))
        _cmd_report(_args("report", "--run", run_id))
        assert "UPPER BOUND" in capsys.readouterr().out

    def test_a_run_with_no_sidecar_is_refused_because_the_coordinate_lives_there(
            self, project, capsys):
        assert _cmd_report(_args("report", "--run", "NOPE")) == EXIT_FAILED

    def test_an_unmeasured_noise_floor_is_not_an_endorsement(self, capsys):
        assert _endorse(_stats(None), 0.02) == EXIT_FAILED

    def test_report_says_the_thickness_is_a_target_not_a_measurement(
            self, project, capsys):
        run_id = _recorded_run(project / "real", _stats(0.005))
        _cmd_report(_args("report", "--run", run_id))
        out = capsys.readouterr().out
        assert "method 'target'" in out
        assert "200.0 um" in out
        assert "fit_results.thickness_method is NULL" in out

    def test_report_says_the_r1_cross_check_was_not_available_rather_than_fine(
            self, project, capsys):
        # 'not checked' is not 'checked and fine'.
        run_id = _recorded_run(project / "real", _stats(0.005))
        _cmd_report(_args("report", "--run", run_id))
        out = capsys.readouterr().out
        assert "tau(R1) cross-check: NOT AVAILABLE" in out

    def test_report_presents_session_drift_from_data_the_series_already_produced(
            self, project, capsys):
        # This is what the retired Longest anchors were meant to buy. It now
        # costs no instrument time at all.
        drift = [{"channel": 1, "start": {"leg": "up", "setpoint_index": 0,
                                          "sigma_settled": 1.0e-4},
                  "end": {"leg": "down", "setpoint_index": 3,
                          "sigma_settled": 1.2e-4},
                  "drift_rel": 0.1818, "noise_floor_rel": 0.005, "tol_rel": 0.02,
                  "significant": True}]
        run_id = _recorded_run(project / "real", _stats(0.005), session_drift=drift)
        _cmd_report(_args("report", "--run", run_id))
        out = capsys.readouterr().out

        assert "Session drift" in out
        assert "DRIFTED" in out
        assert "retrace evidence" in out


# ── Progress rendering ───────────────────────────────────────────────────────

class _Sink:
    """A stream that can pretend to be a terminal, or to have closed."""

    def __init__(self, tty=False, explode=False):
        self.text = ""
        self._tty = tty
        self._explode = explode

    def isatty(self):
        return self._tty

    def write(self, text):
        if self._explode:
            raise ValueError("I/O operation on closed file")
        self.text += text

    def flush(self):
        pass


def _event(kind="heartbeat", **kw):
    params = dict(leg="up", setpoint_index=1, n_setpoints=8, temperature_C=65.0,
                  rh_setpoint_pct=15.0, phase="hold", axis="temperature",
                  pv=64.8, target=65.0, n_rounds=15, fraction=0.2,
                  elapsed_s=3600.0, wall_clock="2026-08-10 22:00:00")
    params.update(kw)
    return ProgressEvent(kind=kind, **params)


def _renderer(**kw):
    params = dict(config=EquilibrationConfig(), stream=None, quiet=False)
    params.update(kw)
    config = params.pop("config")
    if params.get("stream") is None:
        params["stream"] = _Sink()
    return ProgressRenderer(config, **params)


class TestProgressRendering:
    def test_a_terminal_redraws_in_place_and_a_redirected_stream_never_does(self):
        # A 15 h run is overwhelmingly likely to be `> run.log`. An in-place
        # redraw at poll cadence would be megabytes of carriage returns in it.
        tty, pipe = _Sink(tty=True), _Sink(tty=False)
        on_tty, redirected = _renderer(stream=tty), _renderer(stream=pipe)

        for i in range(6):
            event = _event(elapsed_s=60.0 * i)
            on_tty(event)
            redirected(event)

        assert "\r" in tty.text
        assert "\r" not in pipe.text
        assert tty.text != pipe.text

    def test_a_redirected_stream_gets_periodic_lines_not_one_per_poll(self):
        pipe = _Sink(tty=False)
        renderer = _renderer(stream=pipe, milestone_interval_s=300.0)
        for i in range(20):
            renderer(_event(elapsed_s=60.0 * i))

        lines = [ln for ln in pipe.text.splitlines() if ln.strip()]
        assert 0 < len(lines) < 20

    def test_the_status_line_shows_the_whole_hierarchy(self):
        tty = _Sink(tty=True)
        renderer = _renderer(stream=tty)
        renderer(_event(kind=EV_HEARTBEAT, phase="series", round_index=6,
                        round_kind="series", channel=12))

        assert "up" in tty.text and "S2/8" in tty.text
        assert "r7/15" in tty.text and "ch12" in tty.text
        assert "ETA" in tty.text

    def test_the_ambient_verdict_prints_on_a_line_of_its_own_either_way(self):
        # Both outcomes are announced. A silent restore attempt and a successful
        # one look identical, and only one of them means the rig is safe to leave.
        for stream, verdict, detail, mark in (
                (_Sink(tty=True), VERDICT_MET, "ambient restored: temp_controller "
                                               "commanded to 27.5 C", " ok "),
                (_Sink(tty=True), VERDICT_UNMET, "CHECK THE CHAMBER MANUALLY: the "
                                                 "ambient restore FAILED", " !!!! ")):
            renderer = _renderer(stream=stream)
            renderer(_event(kind=EV_AMBIENT_RESTORED, verdict=verdict, detail=detail))
            assert mark in stream.text
            assert detail in stream.text
            assert "\n" in stream.text, "a milestone, not a redrawn status line"

    def test_the_status_line_fits_a_narrow_terminal_and_uses_no_unicode(self):
        tty = _Sink(tty=True)
        renderer = _renderer(stream=tty)
        renderer(_event(kind=EV_HEARTBEAT, phase="series", round_index=6,
                        round_kind="series", channel=16, **_env_kw()))

        drawn = tty.text.split("\r")[-1]
        assert len(drawn) <= 78, drawn
        assert drawn.rstrip() == drawn.rstrip(), "padded, not truncated mid-word"
        assert tty.text.isascii()
        # Nothing was cut off: the last thing on the line is still the ETA.
        assert drawn.rstrip().endswith("h")

    def test_a_rendering_failure_degrades_to_silence_and_is_counted(self):
        # The run wraps this too; both guards exist because the cost of the second
        # is nothing and the cost of a crash at hour seven is the whole night.
        renderer = _renderer(stream=_Sink(tty=True, explode=True))
        renderer(_event())
        renderer(_event(kind=EV_SETPOINT_FINISHED, verdict="met", detail="x"))
        assert renderer.failures == 2

    def test_a_malformed_event_cannot_reach_the_caller_as_an_exception(self):
        renderer = _renderer()
        renderer(object())            # not an event at all
        assert renderer.failures == 1

    def test_quiet_drops_the_live_line_but_keeps_verdicts(self):
        loud, quiet = _Sink(tty=True), _Sink(tty=True)
        _renderer(stream=loud)(_event())
        _renderer(stream=quiet, quiet=True)(_event())
        assert loud.text and not quiet.text

        verdict = _event(kind=EV_SETPOINT_FINISHED, verdict="unmet",
                         detail="humidity did NOT hold 15%RH")
        sink = _Sink(tty=True)
        _renderer(stream=sink, quiet=True)(verdict)
        assert "NOT HELD" in sink.text

    def test_an_unmet_hold_window_is_announced_but_a_met_one_is_not(self):
        # ~240 met windows in a shipped run would bury the setpoint verdicts.
        met, unmet = _Sink(tty=True), _Sink(tty=True)
        _renderer(stream=met, quiet=True)(
            _event(kind=EV_HOLD_VERDICT, verdict="met", detail="held"))
        _renderer(stream=unmet, quiet=True)(
            _event(kind=EV_HOLD_VERDICT, verdict="unmet", detail="did NOT hold"))
        assert not met.text
        assert "did NOT hold" in unmet.text


class TestETA:
    def test_the_eta_is_the_projection_before_anything_has_happened(self):
        assert reconciled_eta_s(0.0, 0.0, 9.3 * 3600) == pytest.approx(9.3 * 3600)

    def test_the_eta_is_reconciled_against_actual_elapsed_not_the_projection_alone(self):
        # Half done in 8 h against a 9.3 h projection: the run is demonstrably
        # slower than modelled and the ETA must say so.
        eta = reconciled_eta_s(8 * 3600, 0.5, 9.3 * 3600)
        naive = 9.3 * 3600 - 8 * 3600
        assert eta > naive
        assert eta == pytest.approx(0.5 * 16 * 3600 + 0.5 * 9.3 * 3600 - 8 * 3600)

    def test_the_eta_never_goes_negative(self):
        assert reconciled_eta_s(50 * 3600, 0.99, 9.3 * 3600) >= 0.0

    def test_hms_is_fixed_width_ascii_and_survives_rubbish(self):
        assert hms(3661) == "1:01:01"
        assert hms(float("nan")) == "0:00:00"
        assert hms(None) == "0:00:00"

    def test_a_measured_round_cost_reprojects_the_eta_and_keeps_the_model_visible(self):
        # The gap between projected and actual IS the finding, so the modelled
        # figure stays on the line rather than being silently replaced.
        # A cycle is the period or the round cost, whichever is longer, so the
        # measured round has to overrun BOTH for the reprojection to lengthen the
        # run. 16 channels of `Standard` now model ~603 s (recalibrated against
        # the bench), so 170 s no longer overruns anything -- it is *shorter*
        # than the model. 700 s is a round that genuinely overruns.
        tty = _Sink(tty=True)
        renderer = _renderer(stream=tty,
                             config=EquilibrationConfig(round_period_s=120.0))
        modelled_total = renderer.projected_total_s

        renderer(_event(kind=EV_ROUND_FINISHED, round_kind="series",
                        round_duration_s=700.0, per_channel_s=43.75))

        assert renderer.measured_total_s is not None
        assert renderer.measured_total_s > modelled_total
        assert renderer.projected_total_s == pytest.approx(modelled_total)

        renderer(_event(kind=EV_HEARTBEAT))
        assert "model" in tty.text

    def test_a_round_overrun_warning_is_rendered_as_its_own_loud_line(self):
        sink = _Sink(tty=True)
        _renderer(stream=sink, quiet=True)(
            _event(kind=EV_COST_WARNING, verdict="unmet", round_duration_s=170.0,
                   detail="a round took 170s but --round-period-s is 120s"))
        assert "ROUND OVERRUNS THE PERIOD" in sink.text
        assert "--round-period-s" in sink.text


# ── Log level ────────────────────────────────────────────────────────────────

@pytest.fixture()
def structlog_state():
    """Save and restore the process-wide log configuration.

    ``configure_logging`` deliberately mutates global state, so a test that
    exercises it would otherwise leave every later test running at whatever level
    it chose.
    """
    saved = structlog.get_config()
    root_level = logging.getLogger().level
    try:
        yield
    finally:
        structlog.configure(**saved)
        logging.getLogger().setLevel(root_level)


class TestLogLevel:
    """The headless CLI configured nothing, so structlog's default ``PrintLogger``
    printed **every** level — including ``rh_duty_sent``, logged on each RH control
    update. Six hours of that buries the run's own reporting."""

    def test_configure_logging_default_honours_the_configured_level(
            self, structlog_state, monkeypatch):
        from softae.config import loader

        monkeypatch.setattr(loader, "log_level", lambda: "WARNING")
        assert configure_logging() == logging.WARNING
        monkeypatch.setattr(loader, "log_level", lambda: "INFO")
        assert configure_logging() == logging.INFO

    def test_configure_logging_default_filters_debug_but_keeps_the_runs_own_output(
            self, structlog_state, monkeypatch, capsys):
        from softae.config import loader
        from softae.workflows import equilibration as workflow

        monkeypatch.setattr(loader, "log_level", lambda: "INFO")
        configure_logging()
        capsys.readouterr()

        # The exact line the operator named, from the module that emits it.
        structlog.get_logger("softae.drivers.async_rh_controller").debug(
            "rh_duty_sent", duty=0.42)
        # The run's own milestone, through the workflow's real module logger.
        workflow.logger.info("equilibration_run_start", run_id="R1")
        out = capsys.readouterr().out

        assert "rh_duty_sent" not in out
        assert "equilibration_run_start" in out, "the run's milestones were silenced"

    def test_configure_logging_verbose_restores_debug(
            self, structlog_state, monkeypatch, capsys):
        from softae.config import loader

        monkeypatch.setattr(loader, "log_level", lambda: "INFO")
        assert configure_logging(verbose=True) == logging.DEBUG
        capsys.readouterr()

        structlog.get_logger("softae.drivers.async_rh_controller").debug(
            "rh_duty_sent", duty=0.42)
        assert "rh_duty_sent" in capsys.readouterr().out

    def test_the_progress_renderer_is_untouched_by_the_default_level(
            self, structlog_state, monkeypatch):
        # The fix is worthless if it also silences the milestone lines, the hold
        # verdicts and the live status line -- none of which go through structlog.
        from softae.config import loader

        monkeypatch.setattr(loader, "log_level", lambda: "WARNING")
        configure_logging()

        sink = _Sink(tty=True)
        renderer = _renderer(stream=sink)
        renderer(_event(kind=EV_SETPOINT_FINISHED, verdict="met",
                        detail="temperature held"))
        renderer(_event(kind=EV_HOLD_VERDICT, verdict="unmet", detail="RH drifted"))
        renderer(_event(kind=EV_HEARTBEAT))

        assert "VERDICT" in sink.text and "temperature held" in sink.text
        assert "RH drifted" in sink.text
        assert "ETA" in sink.text                     # the live status line

    @pytest.mark.parametrize("argv", [["-v", "plan"], ["plan", "-v"]])
    def test_main_applies_the_verbose_flag_from_either_side_of_the_subcommand(
            self, structlog_state, monkeypatch, capsys, project, argv):
        # A subparser copies its own defaults over the outer namespace, so `-v`
        # before the subcommand is exactly the spelling that silently breaks.
        from softae.config import loader

        monkeypatch.setattr(loader, "log_level", lambda: "INFO")
        assert main([*argv, "--channels", "1-2"]) == EXIT_OK
        capsys.readouterr()

        structlog.get_logger("softae.drivers.async_rh_controller").debug(
            "rh_duty_sent", duty=0.42)
        assert "rh_duty_sent" in capsys.readouterr().out

    def test_main_configures_the_level_before_any_subcommand_runs(
            self, structlog_state, monkeypatch, capsys, project):
        from softae.config import loader

        monkeypatch.setattr(loader, "log_level", lambda: "INFO")
        assert main(["plan", "--channels", "1-2"]) == EXIT_OK
        capsys.readouterr()

        structlog.get_logger("softae.drivers.async_rh_controller").debug(
            "rh_duty_sent", duty=0.42)
        assert "rh_duty_sent" not in capsys.readouterr().out


def _env_kw(**overrides):
    env = {"stage_temp_sp_C": 65.0, "chamber_air_C": 64.2, "stage_temp_pv_C": 64.8,
           "rh_sp_pct": 15.0, "rh_pv_pct": 14.6}
    env.update(overrides)
    return {"env": env, "env_status": ENV_OK}


class TestControlsMonitor:
    def test_the_periodic_line_shows_both_setpoints_and_both_process_values(self):
        # The whole point: check a headless run's controls without opening the GUI
        # on it and contending for the rig lock.
        sink = _Sink(tty=False)
        renderer = _renderer(stream=sink, quiet=True)
        renderer(_event(**_env_kw()))

        assert "T sp 65.0 pv 64.2" in sink.text
        assert "stage 64.8C" in sink.text
        assert "RH sp 15.0 pv 14.6%" in sink.text
        assert "2026-08-10 22:00:00" in sink.text
        assert "[1:00:00]" in sink.text

    def test_an_unreadable_value_renders_as_unavailable_never_as_zero(self):
        # AsyncRHController turns a reading held past max_stale_s into NaN.
        # Flattening that to 0.0 or to the last good number would let a dead
        # sensor read as a working one for nine hours.
        sink = _Sink(tty=False)
        _renderer(stream=sink, quiet=True)(
            _event(**_env_kw(rh_pv_pct=None, chamber_air_C=float("nan"))))

        assert "pv --" in sink.text
        assert "0.0" not in sink.text.split("RH")[-1]

    def test_a_skipped_read_says_so_rather_than_looking_like_a_failed_one(self):
        sink = _Sink(tty=False)
        _renderer(stream=sink, quiet=True)(
            _event(env={}, env_status=ENV_SKIPPED))
        assert "not read: rig in use" in sink.text

    def test_the_monitor_line_is_rate_limited_but_the_first_one_is_immediate(self):
        sink = _Sink(tty=False)
        renderer = _renderer(stream=sink, quiet=True, milestone_interval_s=300.0)
        for i in range(10):
            renderer(_event(elapsed_s=60.0 * i, **_env_kw()))

        env_lines = [ln for ln in sink.text.splitlines() if "env" in ln]
        assert len(env_lines) == 2      # t=0 and t=300 s
        assert env_lines[0].startswith("[0:00:00]")

    def test_an_event_without_telemetry_prints_no_monitor_line_at_all(self):
        sink = _Sink(tty=False)
        _renderer(stream=sink, quiet=True)(_event())     # env_status defaults absent
        assert "env" not in sink.text

    def test_the_compact_form_rides_the_live_line_too(self):
        tty = _Sink(tty=True)
        _renderer(stream=tty)(_event(**_env_kw()))
        assert "T65/65" in tty.text and "RH15/15" in tty.text


# ── fit ──────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_reconstructs_the_series_from_the_db_and_writes_stats_to_the_sidecar(
            self, project, capsys):
        import math

        from softae.config import loader
        from softae.core.data_store import DataStore

        project_dir = project / "real"
        store = DataStore(str(project_dir), db_filename=loader.data_db_filename())
        run_id = store.start_run("equilibration_characterization",
                                 mode="characterization")
        points = []
        for i in range(15):
            t = 120.0 * i
            sigma = 1.0e-4 + 1.0e-4 * math.exp(-t / 400.0)
            stem = f"eq_ch1_Lup_S0_R{i}"
            cur = store._conn.execute(
                "INSERT INTO measurements (run_id, channel, timestamp, eis_file_path) "
                "VALUES (?, 1, ?, ?)",
                (run_id, f"2026-08-10T01:{i:02d}:00", f"eis/{stem}_ch1.txt"))
            store._conn.execute(
                "INSERT INTO fit_results (measurement_id, run_id, model_name, "
                "sigma_S_per_cm, fitted_at) VALUES (?, ?, 'simpleSalt', ?, 'now')",
                (cur.lastrowid, run_id, sigma))
            points.append({"step_name": stem, "channel": 1, "leg": "up",
                           "setpoint_index": 0, "round_index": i, "kind": "series",
                           "t_since_hold_s": t})
        store._conn.commit()
        store.close()

        path = project_dir / "runs" / run_id / "equilibration.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": "equilibration/1", "run_id": run_id,
                                    "points": points, "holds": [], "setpoints": []}),
                        encoding="utf-8")

        assert _cmd_fit(_args("fit", "--run", run_id)) == EXIT_OK
        assert "1 fitted, 0 refused" in capsys.readouterr().out

        stats = json.loads(path.read_text(encoding="utf-8"))["stats"]
        assert len(stats) == 1
        assert stats[0]["tau_s"] == pytest.approx(400.0, rel=1e-3)
        assert stats[0]["channel"] == 1 and stats[0]["leg"] == "up"

    def test_fit_refuses_a_run_with_no_sidecar_because_the_coordinate_lives_there(
            self, project, capsys):
        assert _cmd_fit(_args("fit", "--run", "NOPE")) == EXIT_FAILED
        assert "coordinate" in capsys.readouterr().err


# ── The plan as an executable artifact ───────────────────────────────────────

def _write_plan_file(path, *design):
    """``plan --save``, run for its file. Returns the path it wrote."""
    assert _cmd_plan(_args("plan", *design, "--save", str(path))) == EXIT_OK
    return path


def _corrupt_plan(path, old, new):
    """Rewrite one saved plan by literal substitution, to break it deliberately."""
    text = Path(path).read_text(encoding="utf-8")
    assert old in text, f"the saved plan no longer contains {old!r}"
    Path(path).write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def _offline_run(monkeypatch):
    """Neither an instrument nor a store may be opened by the calls under test."""
    import softae.drivers.factory as factory
    import softae.tools.equilibration as tool

    monkeypatch.setattr(factory, "create_manager", _never("create_manager"))
    monkeypatch.setattr(tool, "_open_store", _never("_open_store"))


class TestPlanArtifact:
    """``plan --save`` / ``run --from-plan``: the design travels as a file.

    The two processes share no state, so on 2026-08-10 ``--preset`` reverted to
    ``Standard`` and the geometry was dropped whole on ``run`` — ~40 min of rig
    time and every sigma NULL. A file carrying every resolved value, defaults
    included, is what closes that.
    """

    DESIGN = ("--channels", "1-3,8-16", "--preset", "Quick", "--rounds", "12",
              "--round-period-s", "240", "--rh", "20", "--temperatures", "30,60",
              "--legs", "up", "--measured-per-channel-s", "10.47", *GEOMETRY)

    def test_plan_save_then_run_from_plan_reproduces_every_design_field(
            self, project, tmp_path):
        from softae.tools.equilibration import PLAN_DESIGN_KEYS, _seat_plan

        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)

        planned = _args("plan", *self.DESIGN)
        executed = _args("run", "--from-plan", str(path))
        assert _seat_plan(executed) == []

        for key in PLAN_DESIGN_KEYS:
            assert getattr(executed, key) == getattr(planned, key), key
        # And the thing that actually matters: the same run, not merely the same
        # namespace.
        assert build_config(executed) == build_config(planned)

    def test_a_saved_plan_carries_the_defaults_nobody_typed(self, project, tmp_path):
        # The whole point. `--model`, `--preset` and `--thickness-method` were
        # never on the command line, and a plan omitting them lets them revert.
        from softae.tools.equilibration import load_plan

        design = load_plan(_write_plan_file(tmp_path / "plan.toml",
                                            "--channels", "1-4", *GEOMETRY))

        from softae.workflows.equilibration import DEFAULT_EIS_PRESET

        assert design["model"] == "simpleSalt"
        assert design["thickness_method"] == "target"
        # 'Quick', and read from the constant rather than spelled: the value is
        # what the plan artifact must not lose, and pinning the spelling here
        # would make this a test of the default instead of a test of the file.
        assert design["preset"] == DEFAULT_EIS_PRESET == "Quick"
        assert design["rounds"] == 15
        # The RH setpoint among them: 20, and it is in the file rather than left
        # to whatever the `run` process happens to default to.
        assert design["rh"] == pytest.approx(20.0)

    def test_every_newly_exposed_chamber_flag_reaches_the_config_and_the_plan(
            self, project, tmp_path):
        # None of these was settable without editing source, and the operator hit
        # the approach timeout on a real run with no flag to extend it. Each is
        # given a value distinct from its default so a flag silently dropped on
        # the floor cannot pass by coincidence.
        from softae.tools.equilibration import load_plan

        typed = ("--tolerance-c", "1.25", "--rh-tolerance-pct", "3.5",
                 "--warn-c", "4.5", "--fault-c", "12.0", "--grace-s", "90",
                 "--approach-timeout-s", "2400", "--down-approach-timeout-s", "7200",
                 "--rh-approach-timeout-s", "3000", "--tau-setpoints", "3")
        expected = {"tolerance_C": 1.25, "rh_tolerance_pct": 3.5, "warn_C": 4.5,
                    "fault_C": 12.0, "grace_s": 90.0, "approach_timeout_s": 2400.0,
                    "down_approach_timeout_s": 7200.0,
                    "rh_approach_timeout_s": 3000.0}

        args = _args("plan", "--channels", "1-4", *GEOMETRY, *typed)
        config = build_config(args)
        for field, value in expected.items():
            assert getattr(config, field) == pytest.approx(value), field
        assert config.tau_setpoints == 3

        design = load_plan(_write_plan_file(tmp_path / "plan.toml",
                                            "--channels", "1-4", *GEOMETRY, *typed))
        for field, value in expected.items():
            assert design[field] == pytest.approx(value), field
        assert design["tau_setpoints"] == 3

    def test_the_chamber_flags_survive_the_round_trip_into_the_run(
            self, project, tmp_path):
        # A plan that records a tolerance the run does not read would be worse
        # than not recording it: the file would state a design nobody executed.
        from softae.tools.equilibration import _seat_plan

        typed = ("--tolerance-c", "1.25", "--down-approach-timeout-s", "7200",
                 "--tau-setpoints", "3")
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4",
                                *GEOMETRY, *typed)

        executed = _args("run", "--from-plan", str(path))
        assert _seat_plan(executed) == []
        config = build_config(executed)

        assert config.tolerance_C == pytest.approx(1.25)
        assert config.down_approach_timeout_s == pytest.approx(7200.0)
        assert config.tau_setpoints == 3

    def test_a_plan_from_the_previous_schema_is_refused_rather_than_defaulted(
            self, project, tmp_path, monkeypatch, capsys):
        # A `/2` plan was written when tolerance_C was 0.5, --rh was 15 and the
        # descending leg had the ascending leg's allowance. Reading one here with
        # the new keys defaulted would silently change what "held" means on a file
        # that states neither value.
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4",
                                *GEOMETRY)
        _corrupt_plan(path, f'schema = "{PLAN_SCHEMA}"',
                      'schema = "equilibration-plan/2"')
        _offline_run(monkeypatch)
        capsys.readouterr()

        assert _cmd_run(_args("run", "--from-plan", str(path),
                              "--execute")) == EXIT_FAILED
        err = capsys.readouterr().err
        assert "equilibration-plan/2" in err
        assert PLAN_SCHEMA in err
        assert "does NOT fall back to the built-in defaults" in err

    def test_a_saved_plan_names_its_writer_and_stamps_a_timestamp(
            self, project, tmp_path):
        import tomllib

        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4", *GEOMETRY)
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))

        # `/4` since the sweep floor joined [design]. A `/3` plan is refused
        # rather than read with it defaulted: those were written while `Quick`
        # ended at 20 Hz, and executing one here would take it to the preset's
        # 7 Hz -- roughly twice the cost and a different set of samples whose arcs
        # close -- on a file that names no floor at all. `/3` itself was the
        # chamber joining [design], on the same rule: a `/2` plan was written when
        # tolerance_C was 0.5 and --rh was 15, so executing one here would
        # silently change what "held" means.
        assert data["schema"] == PLAN_SCHEMA == "equilibration-plan/4"
        # Never `__name__`, which is "__main__" under `python -m` and names
        # nothing a reader could open.
        assert data["written_by"] == "softae.tools.equilibration"
        # The same wall-clock spelling ProgressEvent stamps the resulting run with.
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
                            data["written_at"])

    def test_a_saved_plan_says_how_to_execute_it_in_the_file_itself(
            self, project, tmp_path):
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4", *GEOMETRY)
        text = Path(path).read_text(encoding="utf-8")
        assert f"run --from-plan {path} --execute" in text
        assert "sharing no state" in text

    def test_run_from_plan_does_not_require_channels(
            self, project, tmp_path, monkeypatch, capsys):
        # The mandatory --channels refusal must not fire when the plan supplies
        # the selection: the two mechanisms exist for the same reason.
        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)
        _offline_run(monkeypatch)
        capsys.readouterr()

        assert _cmd_run(_args("run", "--from-plan", str(path))) == EXIT_OK
        captured = capsys.readouterr()
        assert "Refusing to run" not in captured.err
        assert "channels:     1-3,8-16 (12)" in captured.out

    def test_run_from_plan_applies_an_override_and_prints_it_as_a_diff(
            self, project, tmp_path, monkeypatch, capsys):
        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)
        _offline_run(monkeypatch)
        capsys.readouterr()

        assert _cmd_run(_args("run", "--from-plan", str(path),
                              "--preset", "Standard")) == EXIT_OK
        out = capsys.readouterr().out

        assert "preset: Quick -> Standard" in out           # the diff
        assert "(Standard)" in out                           # and it was applied
        assert "NOT the saved design" in out

    def test_run_from_plan_reports_nothing_overridden_when_nothing_was(
            self, project, tmp_path, monkeypatch, capsys):
        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)
        _offline_run(monkeypatch)
        capsys.readouterr()

        _cmd_run(_args("run", "--from-plan", str(path)))
        assert "executed exactly as saved" in capsys.readouterr().out

    def test_a_flag_retyped_at_the_planned_value_is_not_reported_as_an_override(
            self, project, tmp_path, monkeypatch, capsys):
        # An operator repeating the design on the command line changed nothing,
        # and a diff that fires anyway teaches them to ignore diffs.
        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)
        _offline_run(monkeypatch)
        capsys.readouterr()

        _cmd_run(_args("run", "--from-plan", str(path), "--preset", "Quick"))
        assert "executed exactly as saved" in capsys.readouterr().out

    def test_an_override_equal_to_the_builtin_default_is_still_an_override(
            self, project, tmp_path, monkeypatch, capsys):
        # The trap: 'Standard' is also --preset's default, so comparing the
        # namespace against the default would read a typed --preset Standard as
        # "nobody chose one" and silently restore Quick.
        path = _write_plan_file(tmp_path / "plan.toml", *self.DESIGN)
        _offline_run(monkeypatch)
        capsys.readouterr()

        _cmd_run(_args("run", "--from-plan", str(path), "--preset", "Standard"))
        out = capsys.readouterr().out
        assert "preset: Quick -> Standard" in out
        assert "rounds:       up to 12 x 240s per setpoint (Standard)" in out

    def test_every_override_is_repeated_in_the_thermal_confirmation(self, capsys):
        # The banner is the last screen before a nine-hour heat; an override that
        # scrolled past twenty lines ago is an override nobody confirmed.
        config = build_config(_args("run", "--channels", "1-4"))
        confirm_thermal(config, assume_yes=True,
                        plan_overrides=[("preset", "Quick", "Standard"),
                                        ("rounds", 12, 30)])
        out = capsys.readouterr().out
        assert "OVERRIDDEN vs the saved plan: preset Quick -> Standard" in out
        assert "OVERRIDDEN vs the saved plan: rounds 12 -> 30" in out


class TestPlanArtifactRefusals:
    """A plan that cannot be executed refuses. It never falls back to the
    defaults — those are a ``Standard`` preset and no geometry, i.e. the run that
    failed."""

    def _refused(self, monkeypatch, capsys, path):
        _offline_run(monkeypatch)
        capsys.readouterr()
        code = _cmd_run(_args("run", "--from-plan", str(path), "--execute"))
        captured = capsys.readouterr()
        assert code == EXIT_FAILED
        assert "does NOT fall back to the built-in defaults" in captured.err
        # Nothing was designed, so nothing can have been executed.
        assert "WHOLE RUN" not in captured.out
        return captured.err

    def test_a_missing_plan_file_refuses_rather_than_defaulting(
            self, project, tmp_path, monkeypatch, capsys):
        err = self._refused(monkeypatch, capsys, tmp_path / "absent.toml")
        assert "no such plan file" in err

    def test_an_unparseable_plan_file_refuses_rather_than_defaulting(
            self, project, tmp_path, monkeypatch, capsys):
        path = tmp_path / "broken.toml"
        path.write_text("this is not [ toml", encoding="utf-8")
        assert "not readable as TOML" in self._refused(monkeypatch, capsys, path)

    def test_an_unrecognised_schema_refuses_rather_than_defaulting(
            self, project, tmp_path, monkeypatch, capsys):
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4", *GEOMETRY)
        _corrupt_plan(path, f'schema = "{PLAN_SCHEMA}"',
                      'schema = "equilibration-plan/99"')
        assert "declares schema" in self._refused(monkeypatch, capsys, path)

    def test_an_unknown_design_key_refuses_rather_than_taking_its_default(
            self, project, tmp_path, monkeypatch, capsys):
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4", *GEOMETRY)
        _corrupt_plan(path, "[design]", '[design]\npresett = "Quick"')
        assert "unknown design key(s) ['presett']" in \
            self._refused(monkeypatch, capsys, path)

    def test_a_plan_missing_a_required_key_refuses_rather_than_defaulting(
            self, project, tmp_path, monkeypatch, capsys):
        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4", *GEOMETRY)
        _corrupt_plan(path, "rounds = 15", "")
        assert "missing required design key(s) ['rounds']" in \
            self._refused(monkeypatch, capsys, path)

    def test_a_plan_with_no_design_table_refuses(
            self, project, tmp_path, monkeypatch, capsys):
        path = tmp_path / "empty.toml"
        path.write_text(f'schema = "{PLAN_SCHEMA}"\n', encoding="utf-8")
        assert "no [design] table" in self._refused(monkeypatch, capsys, path)

    def test_an_absent_geometry_in_a_plan_is_a_value_not_a_corruption(
            self, project, tmp_path):
        # The four nullable keys: a plan that simply has no geometry must load.
        from softae.tools.equilibration import load_plan

        design = load_plan(_write_plan_file(tmp_path / "plan.toml",
                                            "--channels", "1-4"))
        assert design["electrode_t_cm"] is None
        assert design["measured_per_channel_s"] is None

    def test_a_plan_that_cannot_be_written_fails_loudly_rather_than_silently(
            self, project, tmp_path, capsys):
        # A plan believed saved and not saved is worse than one never asked for:
        # the next command reads from it.
        occupied = tmp_path / "not_a_dir"
        occupied.write_text("occupied", encoding="utf-8")
        assert _cmd_plan(_args("plan", "--channels", "1-4", *GEOMETRY,
                               "--save", str(occupied / "plan.toml"))) == EXIT_FAILED
        assert "Could not save the plan" in capsys.readouterr().err


# ── The printed command must work, and must reproduce the printed design ─────

class TestSuggestedInvocations:
    """``plan`` printed ``softae-equilibration run --channels ... --execute``.

    Two defects in one line: the console script was declared in ``pyproject.toml``
    but generated by no install in the working venv, so the name did not resolve;
    and the command dropped ``--preset`` and all three geometry flags, so an
    operator following the tool's own advice reproduced the NULL-sigma run.
    """

    def test_the_invocation_the_tool_prints_is_one_that_resolves(self):
        # The premise of the fix, pinned as a property of the TOOL rather than of
        # the venv. This test used to assert `shutil.which(CONSOLE_SCRIPT) is
        # None` -- true on the day it was written, false the moment someone ran
        # `pip install -e .` (2026-08-11), and never a statement about anything
        # the tool does. What the printed suggestions actually rest on is the
        # module form working whether or not the entry point was ever generated,
        # so that is what is asserted: the exact string the tool hands the
        # operator, run.
        import subprocess
        import sys

        from softae.tools.equilibration import CLI, MODULE

        assert CLI == f"python -m {MODULE}"
        completed = subprocess.run([sys.executable, "-m", MODULE, "--help"],
                                   capture_output=True, text=True, timeout=120)

        assert completed.returncode == EXIT_OK, completed.stderr
        assert "{plan,run,fit,report}" in completed.stdout

    def test_no_command_plan_prints_names_the_unresolvable_console_script(
            self, project, capsys):
        from softae.tools.equilibration import CONSOLE_SCRIPT

        _cmd_plan(_args("plan", "--channels", "1-4", "--preset", "Quick", *GEOMETRY))
        assert CONSOLE_SCRIPT not in capsys.readouterr().out

    def test_no_command_the_channel_refusal_prints_names_the_console_script(
            self, project, capsys):
        from softae.tools.equilibration import CONSOLE_SCRIPT

        _cmd_run(_args("run"))
        captured = capsys.readouterr()
        assert CONSOLE_SCRIPT not in captured.err
        assert "python -m softae.tools.equilibration run --channels" in captured.err

    def test_the_next_action_command_carries_the_preset_and_the_geometry(
            self, project, capsys):
        # Without --save there is no file to point at, so the flag list itself
        # must be complete -- these are the four flags the old line dropped.
        _cmd_plan(_args("plan", "--channels", "1-4", "--preset", "Quick", *GEOMETRY))
        out = capsys.readouterr().out.split("Next most valuable action")[1]

        assert "--preset Quick" in out
        assert "--electrode-l-cm 0.2" in out
        assert "--electrode-t-cm 0.0175" in out
        assert "--electrode-w-cm 0.2" in out

    def test_the_next_action_points_at_the_saved_plan_once_one_exists(
            self, project, tmp_path, capsys):
        path = tmp_path / "plan.toml"
        _cmd_plan(_args("plan", "--channels", "1-4", "--preset", "Quick", *GEOMETRY,
                        "--save", str(path)))
        out = capsys.readouterr().out.split("Next most valuable action")[1]

        assert f"run --from-plan {path} --execute" in out
        assert "python -m softae.tools.equilibration" in out

    def test_the_geometry_suggestion_keeps_the_rest_of_the_design(
            self, project, capsys):
        # The other branch had the same defect: it suggested a bare `plan` with
        # three geometry flags and silently dropped the preset the operator chose.
        _cmd_plan(_args("plan", "--channels", "1-4", "--preset", "Quick"))
        out = capsys.readouterr().out.split("Next most valuable action")[1]

        assert "--preset Quick" in out
        assert "--channels 1-4" in out
        assert "--electrode-l-cm 0.2" in out


# ── Geometry: no silent partial, and 0.0 is not absence ──────────────────────

class TestGeometryResolution:
    def test_partial_geometry_is_refused_naming_the_supplied_and_missing_terms(
            self, project, capsys):
        # It used to be discarded whole by a three-way `and`, with nothing said.
        assert _cmd_plan(_args("plan", "--channels", "1-4",
                               "--electrode-l-cm", "0.2")) == EXIT_FAILED
        err = capsys.readouterr().err

        assert "partial electrode geometry" in err
        assert "L_cm=0.2" in err                       # what was supplied
        assert "--electrode-t-cm" in err               # and what is missing
        assert "--electrode-w-cm" in err

    def test_a_zero_thickness_is_rejected_as_a_stated_value_not_as_an_omission(
            self, project, capsys):
        # A truthiness test made `--electrode-t-cm 0` indistinguishable from
        # omitting it -- in the denominator of sigma = L/(R*t*w).
        assert _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2",
                               "--electrode-t-cm", "0", "--electrode-w-cm", "0.2")) \
            == EXIT_FAILED
        err = capsys.readouterr().err

        assert "non-positive electrode geometry: --electrode-t-cm 0" in err
        assert "STATED value, not an absent one" in err
        assert "partial" not in err                    # told apart from omission

    def test_a_negative_width_is_rejected_the_same_way(self, project, capsys):
        assert _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2",
                               "--electrode-t-cm", "0.02",
                               "--electrode-w-cm", "-0.2")) == EXIT_FAILED
        assert "--electrode-w-cm -0.2" in capsys.readouterr().err

    def test_the_geometry_warning_names_the_missing_terms(self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4"))
        out = capsys.readouterr().out
        assert "--electrode-l-cm" in out
        assert "--electrode-t-cm" in out
        assert "--electrode-w-cm" in out

    def test_the_warning_does_not_claim_no_geometry_when_one_term_was_supplied(
            self, project, capsys):
        # The claim was false the moment one term was given -- and that is exactly
        # the case where the operator most needs to be told which are missing.
        _cmd_plan(_args("plan", "--channels", "1-4", "--electrode-l-cm", "0.2"))
        captured = capsys.readouterr()
        assert "NO electrode geometry" not in captured.out
        assert "MISSING --electrode-t-cm, --electrode-w-cm" in captured.err

    def test_all_three_terms_build_the_geometry_the_run_measures_with(self):
        config = build_config(_args("run", "--channels", "1-4", *GEOMETRY))
        assert config.electrode_geometry == {"L_cm": 0.2, "t_cm": 0.0175,
                                             "w_cm": 0.2}
        # The config enforces the same contract for callers that never see this
        # CLI -- see TestGeometryContract in test_equilibration_workflow.py.


# ── run --execute re-asks when the geometry is absent ────────────────────────

class TestGeometryConfirmation:
    """Not a refusal and not an opt-out flag: the operator's own call, re-asked.

    A run with no L/t/w is a legitimate thing to want — R₁ is still recorded.
    Discovering after nine hours that σ is NULL because three flags were not
    repeated on ``run`` is not.
    """

    NO_GEOMETRY = ("run", "--channels", "1-4", "--execute")

    def test_an_absent_geometry_prompts_and_states_the_consequence(self, capsys):
        config = build_config(_args("run", "--channels", "1-4"))
        assert confirm_no_geometry(config, reader=lambda _p: "yes") is True
        out = capsys.readouterr().out
        assert "sigma_S_per_cm will be NULL for EVERY measurement" in out
        assert "R1 is recorded" in out

    def test_a_supplied_geometry_is_not_re_asked(self, capsys):
        config = build_config(_args("run", "--channels", "1-4", *GEOMETRY))
        assert confirm_no_geometry(config, reader=_never("reader")) is True
        assert capsys.readouterr().out == ""

    def test_yes_skips_the_geometry_confirmation(self, monkeypatch):
        # Otherwise no equilibration run could be scripted or run unattended.
        monkeypatch.setattr("builtins.input", _never("input"))
        config = build_config(_args("run", "--channels", "1-4"))
        assert confirm_no_geometry(config, assume_yes=True) is True

    def test_a_non_tty_declines_the_geometry_confirmation(self):
        def _eof(_prompt):
            raise EOFError

        config = build_config(_args("run", "--channels", "1-4"))
        assert confirm_no_geometry(config, reader=_eof) is False

    def test_a_reflex_keypress_does_not_start_a_run_with_no_geometry(self):
        config = build_config(_args("run", "--channels", "1-4"))
        assert confirm_no_geometry(config, reader=lambda _p: "y") is False

    def test_declining_opens_no_instrument_and_heats_nothing(
            self, monkeypatch, capsys):
        import softae.tools.equilibration as tool

        _offline_run(monkeypatch)
        monkeypatch.setattr(tool, "confirm_thermal", _never("confirm_thermal"))
        monkeypatch.setattr("builtins.input", lambda *_a: "no")

        assert _cmd_run(_args(*self.NO_GEOMETRY)) == EXIT_DECLINED
        assert "Declined" in capsys.readouterr().out

    def test_the_geometry_question_is_asked_before_the_thermal_banner(
            self, monkeypatch, capsys):
        # The thermal gate stays the LAST thing read before the chamber moves.
        import softae.core.hardware_safety as safety
        import softae.drivers.factory as factory
        import softae.tools.equilibration as tool

        monkeypatch.setattr(factory, "create_manager", lambda **_kw: object())
        monkeypatch.setattr(safety, "assert_hardware_armed", lambda *_a, **_kw: None)
        monkeypatch.setattr(tool, "_open_store", _never("_open_store"))
        replies = iter(["yes", "no"])
        monkeypatch.setattr("builtins.input", lambda *_a: next(replies))

        assert _cmd_run(_args(*self.NO_GEOMETRY)) == EXIT_DECLINED
        # The prompt string itself goes to `input`, which is stubbed here, so the
        # anchor is the consequence the question states.
        out = capsys.readouterr().out
        assert out.index("sigma_S_per_cm will be NULL") \
            < out.index("DRIVES THE STAGE HEATER")

    def test_a_dry_run_without_geometry_never_prompts(self, monkeypatch, capsys):
        _offline_run(monkeypatch)
        monkeypatch.setattr("builtins.input", _never("input"))

        assert _cmd_run(_args("run", "--channels", "1-4")) == EXIT_OK
        assert "Dry run" in capsys.readouterr().out


# ── `--model` meant two different things ─────────────────────────────────────

class TestModelFlagVocabularies:
    """``--model`` was the EIS CIRCUIT model on ``plan``/``run`` and the
    RELAXATION model on ``fit``/``report``. One spelling, two vocabularies, both
    plausible-looking strings -- so ``fit --model simpleSalt`` failed confusingly
    rather than obviously, and ``plan --model exponential`` would have recorded a
    circuit model nothing can fit."""

    def test_circuit_model_and_the_model_alias_reach_the_same_destination(self):
        for flag in ("--circuit-model", "--model"):
            args = build_parser().parse_args(
                ["plan", "--channels", "1-4", flag, "flexSalt"])
            assert args.model == "flexSalt"

    def test_relaxation_model_and_the_model_alias_reach_the_same_destination(self):
        for flag in ("--relaxation-model", "--model"):
            args = build_parser().parse_args(
                ["fit", "--run", "R1", flag, "none"])
            assert args.model == "none"

    def test_each_name_is_still_validated_against_its_own_vocabulary(self):
        # The confusion is named, not removed: a circuit model on `fit` is still
        # a refusal, and now the flag it was typed under says which is which.
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["fit", "--run", "R1", "--relaxation-model", "simpleSalt"])

    def test_the_help_on_each_name_points_at_the_other_one(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["plan", "--help"])
        assert "--relaxation-model" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            build_parser().parse_args(["fit", "--help"])
        assert "--circuit-model" in capsys.readouterr().out

    def test_the_saved_plan_prints_the_unambiguous_spelling(self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", *GEOMETRY))
        out = capsys.readouterr().out
        assert "--circuit-model simpleSalt" in out


# ── the settle criterion on the shared design surface ────────────────────────

class TestSettleFlags:
    def test_the_settle_criterion_reaches_the_config_from_the_command_line(self):
        config = build_config(_args(
            "run", "--channels", "1-4", "--settle-tol-rel", "0.25",
            "--settle-n-rounds", "4", "--settle-min-channels", "2",
            "--min-hold-first-s", "900", "--min-hold-s", "300"))

        assert config.settle_enabled is True
        assert config.settle_tol_rel == pytest.approx(0.25)
        assert config.settle_n_rounds == 4
        assert config.settle_min_channels == 2
        assert config.min_hold_first_s == pytest.approx(900.0)
        assert config.min_hold_s == pytest.approx(300.0)

    def test_settle_off_restores_the_fixed_round_count(self):
        config = build_config(_args("run", "--channels", "1-4", "--settle", "off"))
        assert config.settle_enabled is False

    def test_the_shipped_defaults_are_the_ones_the_evidence_supports(self):
        # 10% clears the 5.98% median noise floor the run measured; 1500 s is ~3
        # tau at the first setpoint (tau = 425-575 s while the films dry).
        config = build_config(_args("run", "--channels", "1-4"))
        assert config.settle_tol_rel == pytest.approx(0.10)
        assert config.settle_n_rounds == 3
        assert config.settle_min_channels == 3
        assert config.min_hold_first_s == pytest.approx(1500.0)
        assert config.min_hold_s == pytest.approx(600.0)

    def test_the_criterion_travels_in_a_saved_plan_rather_than_reverting(
            self, project, tmp_path, monkeypatch, capsys):
        # How long a setpoint is held is as much the experiment as which
        # temperatures it visits: a criterion that reverted to its default
        # between `plan` and `run` would be the 2026-08-10 defect in a new field.
        from softae.tools.equilibration import load_plan

        path = _write_plan_file(tmp_path / "plan.toml", "--channels", "1-4",
                                "--settle", "off", "--settle-tol-rel", "0.2",
                                *GEOMETRY)
        design = load_plan(path)
        assert design["settle"] == "off"
        assert design["settle_tol_rel"] == pytest.approx(0.2)

    def test_plan_says_that_rounds_is_a_ceiling_before_a_night_is_budgeted(
            self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", *GEOMETRY))
        out = capsys.readouterr().out
        assert "--rounds is a CEILING, not a count" in out
        assert "railed" in out

    def test_plan_prints_the_duration_as_a_floor_to_ceiling_range(
            self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", *GEOMETRY))
        out = capsys.readouterr().out
        whole = [line for line in out.splitlines() if "WHOLE RUN" in line][0]
        floor, _, ceiling = whole.split()[2].partition("-")
        assert float(floor) < float(ceiling)
        assert "a RANGE, not an estimate" in out

    def test_settle_off_collapses_the_range_to_the_single_old_number(
            self, project, capsys):
        _cmd_plan(_args("plan", "--channels", "1-4", "--settle", "off", *GEOMETRY))
        out = capsys.readouterr().out
        whole = [line for line in out.splitlines() if "WHOLE RUN" in line][0]
        assert "-" not in whole.split()[2]
        assert "every setpoint runs exactly 15 rounds" in out


# ── The run row is closed, on every exit path ────────────────────────────────

def _arrange_outcome(monkeypatch, outcome: BaseException | None = None):
    """``_arrange_interrupt``, but the runner's ending is chosen per test.

    The existing helper only ever raises ``KeyboardInterrupt``, and what the run
    row is entitled to say differs per exit path, so each one has to be reachable.
    """
    _arrange_interrupt(monkeypatch)

    import softae.tools.equilibration as tool

    class _Runner(_InterruptedRunner):
        async def run(self):
            if outcome is not None:
                raise outcome
            return None

    monkeypatch.setattr(tool, "EquilibrationRun", _Runner)


def _the_only_outcome(project: Path) -> dict:
    """``run_outcome`` for the single run the command wrote."""
    from softae.core.data_store import DataStore

    with DataStore(project) as ds:
        run_ids = [r[0] for r in ds._conn.execute(
            "SELECT run_id FROM experiments ORDER BY started_at")]
        assert len(run_ids) == 1, run_ids
        return ds.run_outcome(run_ids[0])


class TestRunRowFinalization:
    """``start_run`` had no matching ``finish_run`` anywhere in this tool.

    Neither the CLI nor ``WorkflowExecutor`` closed the row, so a nine-hour
    characterization that finished cleanly left ``finished_at`` NULL — which is
    byte-for-byte what a killed process leaves behind. The next GUI launch read
    it as an unclean shutdown and offered to park the rig over a run that had
    completed, which is how a real crash report gets trained out of an operator.
    """

    def _run(self, monkeypatch, project: Path, outcome=None):
        _arrange_outcome(monkeypatch, outcome)
        return _cmd_run(_args("run", "--channels", "1-2", *GEOMETRY, "--execute",
                              "--project", str(project)))

    def test_a_completed_run_closes_its_row_done(self, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        assert self._run(monkeypatch, project) == EXIT_OK
        assert _the_only_outcome(project) == {"status": "done", "finished": True}

    def test_a_completed_run_is_not_reported_as_an_unclean_shutdown(
            self, monkeypatch, tmp_path):
        """The defect as the operator met it, pinned at its own surface."""
        from softae.core.data_store import DataStore

        project = tmp_path / "proj"
        self._run(monkeypatch, project)
        with DataStore(project) as ds:
            assert ds.unfinished_runs() == []

    def test_a_ctrl_c_closes_its_row_interrupted(self, monkeypatch, tmp_path):
        project = tmp_path / "proj"
        assert self._run(monkeypatch, project, KeyboardInterrupt()) == EXIT_FAILED
        assert _the_only_outcome(project)["status"] == "interrupted"

    def test_a_dead_sensor_abort_closes_its_row_aborted(self, monkeypatch, tmp_path):
        from softae.workflows.equilibration import EquilibrationAbort

        project = tmp_path / "proj"
        abort = EquilibrationAbort("the RH probe stopped replying", kind="unreadable")
        assert self._run(monkeypatch, project, abort) == EXIT_FAILED
        assert _the_only_outcome(project)["status"] == "aborted"

    def test_an_unnamed_failure_still_closes_its_row_error(
            self, monkeypatch, tmp_path):
        """The ``finally`` catch-all: no ``except`` names a bare RuntimeError."""
        project = tmp_path / "proj"
        with pytest.raises(RuntimeError):
            self._run(monkeypatch, project, RuntimeError("the mux stopped replying"))
        assert _the_only_outcome(project)["status"] == "error"

    def test_a_finalization_failure_does_not_fail_the_run(
            self, monkeypatch, tmp_path):
        """Recording *how* a run ended must not decide *whether* it succeeded."""
        from softae.core.data_store import DataStore

        def _boom(self, *_a, **_kw):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(DataStore, "finish_run", _boom)
        assert self._run(monkeypatch, tmp_path / "proj") == EXIT_OK

    def test_the_row_is_closed_before_the_store_is(self, monkeypatch, tmp_path):
        """The finalizer and ``store.close()`` share one ``finally``, in order.

        A closed connection can record nothing, so the ordering inside that
        block is the whole of the fix on the failure paths.
        """
        from softae.core.data_store import DataStore

        events: list[str] = []
        real_finish, real_close = DataStore.finish_run, DataStore.close
        monkeypatch.setattr(
            DataStore, "finish_run",
            lambda self, *a, **k: events.append("finish") or real_finish(self, *a, **k))
        monkeypatch.setattr(
            DataStore, "close",
            lambda self, *a, **k: events.append("close") or real_close(self, *a, **k))

        self._run(monkeypatch, tmp_path / "proj", KeyboardInterrupt())
        assert events[:2] == ["finish", "close"]
