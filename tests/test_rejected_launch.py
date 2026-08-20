"""A refused launch costs a file, not a setup (S5.I).

Only one campaign may own the rig, and a second attempt is refused outright.
These tests pin the other half of that ruling: the refusal must hand back the
configuration, and it must hand back a *relaunch command* only when a file can
be proved to carry the whole campaign.

The proof is the point. Writing a composition campaign as TOML drops
``general_formulation``; the reloaded spec then has no ``vol_params`` either, so
``resolved_vol_params()`` reads its composition axes as raw µL volumes and the
"same" campaign runs a different experiment without raising anything. A command
that does that is worse than no command.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from softae.core.campaign_spec_io import spec_from_dict
from softae.core.rejected_launch import preserve_rejected_launch

MINIMAL = {"name": "phase_map",
           "parameter_space": {"a": {"type": "float", "low": 0.0, "high": 1.0}}}

PANEL = {"name": "phase_map", "channels": "2, 4", "search_mode": "composition",
         "composition_axes": [{"kind": "Molar ratio", "a": "PEO", "b": "LiCl"}],
         "seed_observations": [[{"a": 0.5}, 0.7]]}

WHEN = datetime(2026, 8, 19, 14, 3, 9, tzinfo=timezone.utc)


def _plain_spec():
    return spec_from_dict({**MINIMAL, "budget": 17, "channels": [3, 4]})


def _composition_spec():
    spec = spec_from_dict(MINIMAL)
    spec.general_formulation = object()      # what the GUI's composition mode sets
    return spec


class TestPanelStateIsAlwaysWritten:
    def test_preserve_writes_the_panel_state_under_the_project(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=_plain_spec(), now=WHEN)
        assert out.panel_state_path is not None
        assert out.panel_state_path.parent == tmp_path / "rejected"
        assert out.panel_state_path.name == "phase_map_20260819T140309Z.json"

    def test_preserve_writes_the_panel_state_for_an_unwritable_campaign(self, tmp_path):
        """The lossless format is unconditional; only the TOML is earned."""
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=_composition_spec(), now=WHEN)
        assert out.panel_state_path is not None and out.panel_state_path.exists()
        assert out.toml_path is None and out.command is None

    def test_preserve_json_round_trips_the_panel_state_verbatim(self, tmp_path):
        import json

        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=None, now=WHEN)
        assert json.loads(out.panel_state_path.read_text(encoding="utf-8")) == PANEL

    def test_preserve_without_a_panel_state_still_reports_the_spec(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, spec=_plain_spec(),
                                       now=WHEN)
        assert out.panel_state_path is None
        assert out.command is not None


class TestTheCommandIsEarned:
    def test_preserve_composition_spec_offers_no_cli_command(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=_composition_spec(), now=WHEN)
        assert out.command is None
        assert out.completeness is not None and not out.completeness.complete
        assert "general_formulation" in out.completeness.missing
        assert "general_formulation" in out.describe()

    def test_preserve_volume_spec_offers_a_runnable_command(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=_plain_spec(), now=WHEN)
        assert out.toml_path is not None and out.toml_path.exists()
        assert out.command is not None
        assert out.command.startswith("softae-campaign run ")
        assert str(out.toml_path) in out.command
        assert "--project" in out.command and str(tmp_path) in out.command
        assert "--yes" in out.command
        # Positional spec — `build_parser` has no --spec flag to pass.
        assert "--spec" not in out.command

    def test_preserve_written_toml_reloads_to_the_same_campaign(self, tmp_path):
        from softae.core.campaign_spec_io import load_campaign_spec

        out = preserve_rejected_launch(project_dir=tmp_path, spec=_plain_spec(),
                                       now=WHEN)
        reloaded = load_campaign_spec(out.toml_path)
        assert reloaded.name == "phase_map"
        assert reloaded.budget == 17
        assert reloaded.channels == (3, 4)

    def test_preserve_omits_head_flags_when_the_position_is_unknown(self, tmp_path):
        """The refusal precedes the head gate, so nothing has verified it."""
        out = preserve_rejected_launch(project_dir=tmp_path, spec=_plain_spec(),
                                       now=WHEN)
        assert "--head-up" not in out.command and "--head-down" not in out.command
        assert "--head-up" in out.describe()      # said, rather than guessed

    def test_preserve_emits_head_flag_when_the_position_is_known(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, spec=_plain_spec(),
                                       head_up=False, now=WHEN)
        assert out.command.endswith("--head-down")

    def test_preserve_marks_a_mock_relaunch_as_mock(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, spec=_plain_spec(),
                                       mock=True, now=WHEN)
        assert "--mock" in out.command

    def test_preserve_quotes_a_project_path_containing_spaces(self, tmp_path):
        spaced = tmp_path / "my project"
        out = preserve_rejected_launch(project_dir=spaced, spec=_plain_spec(),
                                       now=WHEN)
        assert f'"{spaced}"' in out.command


class TestItNeverCostsMoreThanItSaves:
    def test_preserve_reports_an_unwritable_directory_rather_than_raising(
        self, tmp_path, monkeypatch
    ):
        def no_write(*a, **k):
            raise OSError("read-only volume")

        monkeypatch.setattr("pathlib.Path.write_text", no_write)
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state=PANEL,
                                       spec=None, now=WHEN)
        assert out.panel_state_path is None
        assert out.errors and "read-only volume" in out.errors[0]
        assert "read-only volume" in out.describe()

    def test_preserve_names_the_file_after_the_campaign_and_the_time(self, tmp_path):
        out = preserve_rejected_launch(
            project_dir=tmp_path, panel_state={"name": "a b/c*d"}, now=WHEN)
        assert out.panel_state_path.name == "a_b_c_d_20260819T140309Z.json"
        assert out.panel_state_path.parent == tmp_path / "rejected"

    def test_preserve_without_a_name_still_writes_somewhere_findable(self, tmp_path):
        out = preserve_rejected_launch(project_dir=tmp_path, panel_state={})
        assert out.panel_state_path.name.startswith("campaign_")


@pytest.mark.parametrize("field", ["seed", "rh_stability_pct"])
def test_preserve_an_explicitly_disabled_setting_survives_into_the_command(
    tmp_path, field
):
    """``rh_stability_pct = None`` switches the RH gate OFF; the default switches
    it back on.

    TOML has no null, so the writer used to drop the ``None`` and this offered no
    command rather than one that re-enabled the gate. The file names its explicit
    nothings now, so the command *is* offered — and the reloaded spec still has
    the gate off, which is the thing that actually had to be true.
    """
    from softae.core.campaign_spec_io import load_campaign_spec

    spec = _plain_spec()
    setattr(spec, field, None)
    out = preserve_rejected_launch(project_dir=tmp_path, spec=spec, now=WHEN)

    assert out.command is not None
    assert out.completeness.complete
    assert getattr(load_campaign_spec(out.toml_path), field) is None
