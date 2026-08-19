"""Tests for the config-relative ``loader.data_root()`` resolver (spec §1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.config import loader


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    """Reset the module-level config cache around every case."""
    loader._config = None
    loader._config_path = None
    loader._config_hash = None
    yield
    loader._config = None
    loader._config_path = None
    loader._config_hash = None


def _write_cfg(tmp_path: Path, data_root_value: str) -> Path:
    toml = tmp_path / "softae_config.toml"
    toml.write_text(f"[paths]\ndata_root = '{data_root_value}'\n", encoding="utf-8")
    loader.load(path=toml, reload=True)
    return toml


class TestDefaultPCB:
    def test_default_pcb_name_reads_config_key(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text(
            'default_pcb = "SoftAE_EIS_4Stripe"\n'
            "[pcb.SoftAE_IDE_EIS]\nchannels = 16\ngrid = [4, 4]\n"
            "[pcb.SoftAE_EIS_4Stripe]\nchannels = 32\ngrid = [8, 4]\n",
            encoding="utf-8",
        )
        loader.load(path=toml, reload=True)
        assert loader.default_pcb_name() == "SoftAE_EIS_4Stripe"

    def test_default_pcb_name_falls_back_to_first_sorted(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text(
            "[pcb.Bravo]\nchannels = 8\n[pcb.Alpha]\nchannels = 4\n",
            encoding="utf-8",
        )
        loader.load(path=toml, reload=True)
        assert loader.default_pcb_name() == "Alpha"

    def test_default_pcb_name_none_when_no_pcbs(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[paths]\ndata_root = './data'\n", encoding="utf-8")
        loader.load(path=toml, reload=True)
        assert loader.default_pcb_name() is None

    def test_electrode_count_from_channels_then_grid(self):
        from softae.core.geometry import electrode_count
        assert electrode_count({"channels": 32, "grid": [8, 4]}) == 32
        assert electrode_count({"grid": [8, 4]}) == 32  # no channels → grid product
        assert electrode_count({}) == 16                # default 4×4


class TestDropcastConfig:
    def test_defaults_when_section_absent(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[paths]\ndata_root = './data'\n", encoding="utf-8")
        loader.load(path=toml, reload=True)
        dc = loader.dropcast_config()
        assert dc["dispense_rate_uL_min"] == 75.0
        assert dc["line_flush_rate_uL_min"] == 500.0
        assert dc["flush_factor"] == 3.0
        assert dc["settle_factor"] == 2.0
        assert dc["start_flush_uL"] == [80.0, 80.0, 80.0]

    def test_section_overrides_defaults(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text(
            "[dropcast]\n"
            "dispense_rate_uL_min = 120.0\n"
            "settle_factor = 3.5\n"
            "start_flush_uL = [10.0, 20.0, 30.0]\n",
            encoding="utf-8",
        )
        loader.load(path=toml, reload=True)
        dc = loader.dropcast_config()
        assert dc["dispense_rate_uL_min"] == 120.0
        assert dc["settle_factor"] == 3.5
        assert dc["start_flush_uL"] == [10.0, 20.0, 30.0]
        # Unspecified keys keep their defaults.
        assert dc["flush_factor"] == 3.0

    def test_bad_start_flush_falls_back_to_default(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text(
            "[dropcast]\nstart_flush_uL = 42.0\n", encoding="utf-8"
        )
        loader.load(path=toml, reload=True)
        assert loader.dropcast_config()["start_flush_uL"] == [80.0, 80.0, 80.0]

    def test_set_default_recipe_replaces_line(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text(
            '[dropcast]\ndefault_recipe = "legacy"\nflush_factor = 3.0\n',
            encoding="utf-8")
        loader.load(path=toml, reload=True)
        loader.set_dropcast_default_recipe("two_phase")
        dc = loader.dropcast_config()
        assert dc["default_recipe"] == "two_phase"
        assert dc["flush_factor"] == 3.0   # other keys preserved

    def test_set_default_recipe_appends_when_missing(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[dropcast]\nflush_factor = 3.0\n", encoding="utf-8")
        loader.load(path=toml, reload=True)
        loader.set_dropcast_default_recipe("single_drop")
        assert loader.dropcast_config()["default_recipe"] == "single_drop"

    def test_set_default_recipe_creates_section(self, tmp_path):
        toml = tmp_path / "softae_config.toml"
        toml.write_text('[paths]\ndata_root = "./data"\n', encoding="utf-8")
        loader.load(path=toml, reload=True)
        loader.set_dropcast_default_recipe("two_phase")
        assert loader.dropcast_config()["default_recipe"] == "two_phase"


class TestCampaignCadences:
    """``[campaign]`` — the two run-directory sidecar cadences (stage 5, S5.F).

    The section is **optional and absent from the shipped config**, so the
    absent case is the shipped case: a headless campaign must publish at the
    documented defaults with nothing in the file at all.
    """

    def _cfg(self, tmp_path: Path, body: str = "") -> None:
        toml = tmp_path / "softae_config.toml"
        toml.write_text(f"[paths]\ndata_root = './data'\n{body}", encoding="utf-8")
        loader.load(path=toml, reload=True)

    def test_campaign_section_absent_returns_documented_defaults(self, tmp_path):
        from softae.core.campaign_events import (
            DEFAULT_CONDITIONS_POLL_S,
            DEFAULT_HEARTBEAT_S,
        )

        self._cfg(tmp_path)
        assert loader.campaign_config() == {}
        assert loader.campaign_conditions_poll_s() == DEFAULT_CONDITIONS_POLL_S == 5.0
        assert loader.campaign_heartbeat_s() == DEFAULT_HEARTBEAT_S == 30.0

    def test_campaign_section_present_overrides_both_cadences(self, tmp_path):
        self._cfg(tmp_path, "[campaign]\nconditions_poll_s = 12\nheartbeat_s = 45\n")
        assert loader.campaign_conditions_poll_s() == 12.0
        assert loader.campaign_heartbeat_s() == 45.0

    def test_campaign_zero_survives_as_disabled_rather_than_defaulting(self, tmp_path):
        """``0`` is a value, not an absence — it is how each sidecar is switched
        off, so a truthiness test here would make the off switch unreachable."""
        self._cfg(tmp_path, "[campaign]\nconditions_poll_s = 0\nheartbeat_s = 0\n")
        assert loader.campaign_conditions_poll_s() == 0.0
        assert loader.campaign_heartbeat_s() == 0.0

    def test_campaign_unparsable_cadence_falls_back_to_the_default(self, tmp_path):
        self._cfg(tmp_path, "[campaign]\nconditions_poll_s = 'soon'\n")
        assert loader.campaign_conditions_poll_s() == 5.0

    def test_campaign_section_of_the_wrong_type_is_ignored(self, tmp_path):
        self._cfg(tmp_path, "campaign = 'yes please'\n")
        assert loader.campaign_config() == {}
        assert loader.campaign_heartbeat_s() == 30.0


class TestDataRoot:
    def test_data_root_relative_anchors_at_config_dir_not_cwd(self, tmp_path, monkeypatch):
        _write_cfg(tmp_path, "./data")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert loader.data_root() == (tmp_path / "data").resolve()

    def test_data_root_absolute_returned_as_is(self, tmp_path):
        abs_target = tmp_path / "abs"
        _write_cfg(tmp_path, abs_target.as_posix())
        assert loader.data_root() == abs_target.resolve()

    def test_data_root_expanduser_expands_tilde(self, tmp_path):
        _write_cfg(tmp_path, "~/cat")
        result = loader.data_root()
        assert result.is_absolute()
        assert result == Path("~/cat").expanduser().resolve()

    def test_data_root_no_config_falls_back_to_cwd_without_raising(self, tmp_path, monkeypatch):
        def _raise(*args, **kwargs):
            raise FileNotFoundError("no config anywhere")

        monkeypatch.setattr(loader, "_find_config_file", _raise)
        monkeypatch.chdir(tmp_path)
        result = loader.data_root()  # must not raise
        assert result == (Path.cwd() / "data").resolve()

    def test_data_root_does_not_create_directory(self, tmp_path):
        _write_cfg(tmp_path, "./data")
        result = loader.data_root()
        assert not result.exists()  # resolver has no filesystem side effect
