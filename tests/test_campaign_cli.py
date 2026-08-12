"""Headless campaign CLI and its spec loader (P6).

The point of P6 is that an unattended run should not require a GUI to stay open.
The point of *these tests* is that it must not require a second implementation
either: the CLI calls the same `run_autonomous_campaign`, and the loader refuses
anything it cannot represent faithfully rather than quietly running a different
experiment from the one the file describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.campaign_spec_io import (
    SpecLoadError,
    load_campaign_spec,
    spec_from_dict,
    spec_to_dict,
)
from softae.tools import campaign as cli

MINIMAL = {
    "name": "c",
    "parameter_space": {"a": {"type": "float", "low": 0.0, "high": 1.0}},
}


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "spec.toml"
    p.write_text(body, encoding="utf-8")
    return p


# ── Loader ───────────────────────────────────────────────────────────────────

class TestLoader:
    def test_loads_a_minimal_spec(self, tmp_path):
        p = _write(tmp_path, """
name = "demo"
[parameter_space.a]
type = "float"
low = 0.0
high = 1.0
""")
        spec = load_campaign_spec(p)
        assert spec.name == "demo"
        assert "a" in spec.parameter_space

    def test_lists_become_tuples(self, tmp_path):
        p = _write(tmp_path, """
name = "demo"
channels = [21, 22]
pump_ids = [0, 1]
[parameter_space.a]
type = "float"
low = 0.0
high = 1.0
""")
        spec = load_campaign_spec(p)
        assert spec.channels == (21, 22)
        assert spec.pump_ids == (0, 1)

    def test_a_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(SpecLoadError, match="No such campaign file"):
            load_campaign_spec(tmp_path / "nope.toml")

    def test_malformed_toml_is_reported(self, tmp_path):
        with pytest.raises(SpecLoadError, match="Could not parse"):
            load_campaign_spec(_write(tmp_path, "name = [unclosed"))

    def test_name_is_required(self):
        with pytest.raises(SpecLoadError, match="'name' is required"):
            spec_from_dict({"parameter_space": MINIMAL["parameter_space"]})

    def test_a_campaign_with_nothing_to_search_is_refused(self):
        with pytest.raises(SpecLoadError, match="parameter_space"):
            spec_from_dict({"name": "c"})

    def test_an_unknown_field_is_refused_not_ignored(self):
        """A typo would otherwise silently take the default."""
        with pytest.raises(SpecLoadError, match="unknown field"):
            spec_from_dict({**MINIMAL, "budgett": 40})

    @pytest.mark.parametrize(
        "field", ["prior_mean", "formulation", "run_plan", "piezo",
                  "general_formulation", "seed_observations"])
    def test_unrepresentable_fields_are_refused_with_a_reason(self, field):
        """Loading one partially would run a different experiment."""
        with pytest.raises(SpecLoadError, match="cannot be set from a file"):
            spec_from_dict({**MINIMAL, field: "whatever"})

    def test_a_bad_parameter_is_caught_at_load_not_in_the_optimizer(self):
        with pytest.raises(SpecLoadError, match="needs 'low' and 'high'"):
            spec_from_dict({"name": "c",
                            "parameter_space": {"a": {"type": "float"}}})

    def test_inverted_bounds_are_caught(self):
        with pytest.raises(SpecLoadError, match="low >= high"):
            spec_from_dict({"name": "c", "parameter_space": {
                "a": {"type": "float", "low": 5.0, "high": 1.0}}})

    def test_a_categorical_needs_choices(self):
        with pytest.raises(SpecLoadError, match="needs 'choices'"):
            spec_from_dict({"name": "c", "parameter_space": {
                "a": {"type": "categorical"}}})

    def test_an_unknown_parameter_type_is_caught(self):
        with pytest.raises(SpecLoadError, match="unknown type"):
            spec_from_dict({"name": "c", "parameter_space": {
                "a": {"type": "quaternion"}}})


class TestRoundTrip:
    def test_a_written_spec_reloads_identically(self):
        original = spec_from_dict({**MINIMAL, "budget": 17, "channels": [3, 4],
                                   "two_phase": True})
        reloaded = spec_from_dict(spec_to_dict(original))

        assert reloaded.budget == 17
        assert reloaded.channels == (3, 4)
        assert reloaded.two_phase is True

    def test_writing_omits_unrepresentable_fields_rather_than_faking_them(self):
        spec = spec_from_dict(MINIMAL)
        spec.prior_mean = lambda p: 0.0
        assert "prior_mean" not in spec_to_dict(spec)

    def test_defaults_are_not_written_out(self):
        """A written file should show what was chosen, not the whole dataclass."""
        written = spec_to_dict(spec_from_dict(MINIMAL))
        assert "name" in written and "parameter_space" in written
        assert "kappa" not in written


# ── [measurement] and the legacy eis_* keys (T2.4) ───────────────────────────

class TestMeasurementBlock:
    def test_a_measurement_table_is_loaded(self, tmp_path):
        p = _write(tmp_path, """
name = "demo"
[parameter_space.a]
type = "float"
low = 0.0
high = 1.0

[measurement]
modality = "eis"
preset = "Extended"
enabled = true

[measurement.overrides]
npts = 41
""")
        spec = load_campaign_spec(p)

        assert spec.measurement.preset == "Extended"
        assert spec.measurement.overrides == {"npts": 41}

    def test_the_legacy_keys_still_load_and_populate_the_block(self, tmp_path):
        """Files written before the block existed must keep running."""
        p = _write(tmp_path, """
name = "demo"
eis_preset = "Longest"
measure_eis = false
[parameter_space.a]
type = "float"
low = 0.0
high = 1.0
""")
        with pytest.warns(DeprecationWarning):
            spec = load_campaign_spec(p)

        assert spec.measurement.preset == "Longest"
        assert spec.measurement.enabled is False

    def test_both_spellings_disagreeing_is_a_spec_error(self):
        """Neither can be preferred silently — the file says two things."""
        with pytest.raises(SpecLoadError, match="disagree"):
            spec_from_dict({**MINIMAL, "eis_preset": "Quick",
                            "measurement": {"preset": "Extended"}})

    def test_both_spellings_agreeing_is_accepted_silently(self):
        """No warning here: the block is already present, so there is nothing to
        migrate to. It is also what `dataclasses.replace` supplies internally,
        and warning on that would fire on copies the caller never made."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            spec = spec_from_dict({**MINIMAL, "eis_preset": "Extended",
                                   "measurement": {"preset": "Extended"}})

        assert spec.measurement.preset == "Extended"

    def test_a_misspelled_measurement_key_is_refused(self):
        with pytest.raises(SpecLoadError, match="unknown measurement key"):
            spec_from_dict({**MINIMAL, "measurement": {"presett": "Extended"}})

    def test_a_non_table_measurement_is_refused(self):
        with pytest.raises(SpecLoadError, match="measurement"):
            spec_from_dict({**MINIMAL, "measurement": "Extended"})

    def test_writing_emits_the_block_and_drops_the_legacy_mirrors(self):
        """Emitting both would deprecation-warn on reload of our own output."""
        with pytest.warns(DeprecationWarning):
            spec = spec_from_dict({**MINIMAL, "eis_preset": "Extended"})

        written = spec_to_dict(spec)

        assert written["measurement"]["preset"] == "Extended"
        assert "eis_preset" not in written

    def test_a_written_measurement_block_reloads_without_warning(self):
        original = spec_from_dict({**MINIMAL,
                                   "measurement": {"preset": "Longest",
                                                   "overrides": {"npts": 12}}})
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            reloaded = spec_from_dict(spec_to_dict(original))

        assert reloaded.measurement == original.measurement

    def test_a_default_block_is_not_written_out(self):
        written = spec_to_dict(spec_from_dict(MINIMAL))
        assert "measurement" not in written


# ── CLI ──────────────────────────────────────────────────────────────────────

DEMO = """
name = "cli_test"
channels = [21, 22]
pcb_name = "SoftAE_EIS_4Stripe"
budget = 2
two_phase = true
vol_params = ["vol_p0", "vol_p1"]
pump_ids = [0, 1]
time_scale = 0.0

[parameter_space.vol_p0]
type = "float"
low = 5.0
high = 30.0

[parameter_space.vol_p1]
type = "float"
low = 5.0
high = 30.0
"""


class TestCLI:
    def test_check_parses_and_projects_without_running(self, tmp_path, capsys):
        rc = cli.main(["check", str(_write(tmp_path, DEMO))])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "cli_test" in out
        assert "per iteration" in out
        assert "sooner" in out          # bound, not an ETA

    def test_a_bad_spec_exits_with_a_usage_code(self, tmp_path, capsys):
        rc = cli.main(["check", str(_write(tmp_path, 'name = "x"'))])
        assert rc == cli.EXIT_USAGE
        assert "Spec error" in capsys.readouterr().err

    def test_run_executes_the_campaign_end_to_end(self, tmp_path, capsys):
        rc = cli.main([
            "run", str(_write(tmp_path, DEMO)), "--mock", "--yes", "--head-up",
            "--project", str(tmp_path / "proj"),
        ])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "head registered as raised" in out
        assert "STOPPED: 2 trial(s)" in out

    def test_head_position_must_be_stated_not_assumed(self, tmp_path, capsys,
                                                      monkeypatch):
        """A wrong head belief costs one wrong flip; it is never guessed."""
        monkeypatch.setattr(cli.sys, "stdin", None)
        rc = cli.main(["run", str(_write(tmp_path, DEMO)), "--mock", "--yes"])

        assert rc == cli.EXIT_DECLINED
        assert "Head position unknown" in capsys.readouterr().out

    def test_resume_is_an_alias_for_run_resume(self, tmp_path):
        args = cli.build_parser().parse_args(
            ["resume", str(_write(tmp_path, DEMO))])
        assert args.command == "resume"

    def test_the_parser_rejects_contradictory_head_flags(self, tmp_path):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(
                ["run", "s.toml", "--head-up", "--head-down"])


class TestHeadlessPurgeAttachment:
    """The headless path must purge, or must not bill for purging (T5.4).

    `attach_purge_scheduler` used to be called only from `gui/main_window.py`,
    so a headless campaign resolved a `NullPurgeRunner` and never purged — while
    the CLI's projection billed `purge_uL_per_day` to the runway regardless.
    """

    def test_the_headless_path_resolves_a_real_purge_runner(self, tmp_path):
        from softae.core.autonomous_wiring import _resolve_purge_runner
        from softae.core.purge_runner import NullPurgeRunner
        from softae.drivers.mock_factory import create_mock_manager

        manager = create_mock_manager(config={})
        cli._attach_purge(manager, None)
        runner = _resolve_purge_runner(manager)

        assert not isinstance(runner, NullPurgeRunner)
        assert runner.performs_purges is True

    def test_a_disabled_schedule_attaches_nothing(self, monkeypatch):
        """`[purge] enabled = false` bills nothing, so it must stand up nothing."""
        from softae.core import purge as purge_mod
        from softae.drivers.mock_factory import create_mock_manager

        monkeypatch.setattr(
            purge_mod, "load_purge_settings",
            lambda store=None: purge_mod.PurgeSettings(enabled=False))
        manager = create_mock_manager(config={})

        assert cli._attach_purge(manager, None) is None
        assert cli._purge_is_attached(manager) is False

    def test_an_unattached_purge_is_not_billed_to_the_runway(self, monkeypatch):
        """The projection reads attachment reality, not the configured rate."""
        seen = {}

        def _fake_project(spec, *, catalog, ledger=None, purge_uL_per_day=None):
            seen["purge"] = purge_uL_per_day
            raise RuntimeError("stop here — only the argument is under test")

        import softae.core.preflight as preflight
        from softae.drivers.mock_factory import create_mock_manager

        monkeypatch.setattr(preflight, "project_campaign", _fake_project)
        manager = create_mock_manager(config={})  # nothing attached

        cli._project(spec_from_dict(MINIMAL), manager, assume_yes=True)

        assert seen["purge"] == {}

    def test_an_attached_purge_is_billed_to_the_runway(self, monkeypatch):
        """Positive control: the empty bill above is attachment, not a no-op."""
        seen = {}

        def _fake_project(spec, *, catalog, ledger=None, purge_uL_per_day=None):
            seen["purge"] = purge_uL_per_day
            raise RuntimeError("stop here — only the argument is under test")

        import softae.core.preflight as preflight
        from softae.drivers.mock_factory import create_mock_manager

        monkeypatch.setattr(preflight, "project_campaign", _fake_project)
        manager = create_mock_manager(config={})
        cli._attach_purge(manager, None)

        cli._project(spec_from_dict(MINIMAL), manager, assume_yes=True)

        assert seen["purge"], "an attached scheduler must bill its consumption"

    def test_check_attaches_so_it_projects_the_run_it_describes(self, tmp_path,
                                                                capsys):
        rc = cli.main(["check", str(_write(tmp_path, DEMO))])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "not attached" not in out


class TestCalibrationAdvisory:
    """Preflight names uncommissioned channels — and never stops for them."""

    def test_check_reports_calibration_state_for_the_declared_channels(
            self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(
            "softae.analysis.eis.calibration.load_calibration",
            lambda *a, **k: None)

        rc = cli.main(["check", str(_write(tmp_path, DEMO))])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "EIS calibration [advisory]" in out
        assert "uncalibrated channels: 21, 22" in out

    def test_the_advisory_says_it_is_advisory_and_returns_ok(self, tmp_path,
                                                             capsys):
        rc = cli.main(["check", str(_write(tmp_path, DEMO))])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "[advisory]" in out

    def test_a_measured_channel_is_not_reported_as_uncalibrated(self, capsys,
                                                                monkeypatch):
        from softae.analysis.eis.calibration import CalibrationSet

        monkeypatch.setattr(
            "softae.analysis.eis.calibration.resolve_calibration",
            lambda *a, **k: CalibrationSet(fixture_id="default",
                                           channels_measured=(21,),
                                           channels_assumed=(22,)))

        cli._calibration_advisory(
            spec_from_dict({**MINIMAL, "channels": [21, 22]}))
        out = capsys.readouterr().out

        assert "uncalibrated channels" not in out
        assert "22" in out          # named as inheriting, not as measured

    def test_an_unreadable_calibration_never_breaks_preflight(self, capsys,
                                                              monkeypatch):
        def _boom(*_a, **_k):
            raise RuntimeError("calibration store is on fire")

        monkeypatch.setattr(
            "softae.analysis.eis.calibration.resolve_calibration", _boom)

        cli._calibration_advisory(spec_from_dict(MINIMAL))

        assert "calibration state unavailable" in capsys.readouterr().out


class TestSafeDefaults:
    def test_confirm_refuses_without_a_terminal_rather_than_guessing(self,
                                                                     monkeypatch):
        """A cron-launched run that assumed 'yes' is the failure P6 must avoid."""
        monkeypatch.setattr(cli.sys, "stdin", None)
        assert cli._confirm("proceed?", assume_yes=False) is False

    def test_confirm_honours_pre_approval(self):
        assert cli._confirm("proceed?", assume_yes=True) is True

    def test_the_cli_passes_no_board_handlers(self):
        """Headless board gates must fall back to the safe built-in defaults.

        `on_board_exchange=None` cancels (a plate swap is physical and nobody is
        there); `on_board_check=None` resumes past used wells, never re-casting.
        """
        import inspect

        source = inspect.getsource(cli._cmd_run)
        assert "on_board_exchange=None" in source
        assert "on_board_check=None" in source
