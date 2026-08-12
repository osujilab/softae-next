"""Tests for B1 — config hash per workflow run."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from softae.config import loader as cfg
from softae.core.data_store import DataStore


@pytest.fixture(autouse=True)
def _restore_config_cache():
    """Prevent temp-config leakage into the global loader cache.

    These tests reset ``cfg._config`` and load throw-away temp configs. Without
    cleanup the last temp config stays cached in the module-level globals and
    leaks into unrelated tests (making the suite order-dependent). Snapshot the
    cache before each test and restore it afterward so downstream tests re-read
    the real ``softae_config.toml``.
    """
    snapshot = (cfg._config, cfg._config_path, cfg._config_hash)
    try:
        yield
    finally:
        cfg._config, cfg._config_path, cfg._config_hash = snapshot


# ── Config hash / path tests ────────────────────────────────────────────────


class TestConfigHash:
    def test_hash_is_64_hex(self, tmp_path: Path) -> None:
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[paths]\ndata_root = './data'\n", encoding="utf-8")
        cfg._config = None  # reset cache
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        h = cfg.config_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[paths]\ndata_root = './data'\n", encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        h1 = cfg.config_hash()
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        h2 = cfg.config_hash()
        assert h1 == h2

    def test_hash_changes_with_content(self, tmp_path: Path) -> None:
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[a]\nk = 1\n", encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        h1 = cfg.config_hash()

        toml.write_text("[a]\nk = 2\n", encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        h2 = cfg.config_hash()
        assert h1 != h2

    def test_config_path_returns_path(self, tmp_path: Path) -> None:
        toml = tmp_path / "softae_config.toml"
        toml.write_text("[x]\n", encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        assert cfg.config_path() == toml


# ── DataStore config_hash column tests ──────────────────────────────────────


class TestDataStoreConfigHash:
    def test_start_run_stores_hash(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj") as ds:
            run_id = ds.start_run("wf", "{}", config_hash="abc123")
            rows = ds.query_runs()
            assert rows[0]["config_hash"] == "abc123"

    def test_start_run_default_empty_hash(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj") as ds:
            ds.start_run("wf", "{}")
            rows = ds.query_runs()
            assert rows[0]["config_hash"] == ""

    def test_migration_adds_column(self, tmp_path: Path) -> None:
        """Simulate an old database without the config_hash column."""
        import sqlite3

        proj = tmp_path / "proj" / "db"
        proj.mkdir(parents=True)
        db = proj / "softae.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE experiments ("
            "  run_id TEXT PRIMARY KEY,"
            "  started_at TEXT NOT NULL,"
            "  finished_at TEXT,"
            "  workflow_name TEXT NOT NULL,"
            "  workflow_mode TEXT NOT NULL DEFAULT 'unknown',"
            "  campaign TEXT NOT NULL DEFAULT 'dev',"
            "  quality TEXT NOT NULL DEFAULT 'explore',"
            "  pcb_name TEXT,"
            "  eis_preset TEXT,"
            "  config_snapshot_json TEXT NOT NULL DEFAULT '{}',"
            "  status TEXT NOT NULL DEFAULT 'running'"
            ")"
        )
        conn.commit()
        conn.close()

        # Open DataStore — migration should add config_hash
        (tmp_path / "proj" / "runs").mkdir(exist_ok=True)
        (tmp_path / "proj" / "formulations").mkdir(exist_ok=True)
        with DataStore(tmp_path / "proj") as ds:
            run_id = ds.start_run("wf", "{}", config_hash="migrated")
            rows = ds.query_runs()
            assert rows[0]["config_hash"] == "migrated"


# ── save_pico_ports tests ────────────────────────────────────────────────────


class TestSavePicoPorts:
    _TOML_TEMPLATE = textwrap.dedent("""\
        [instruments.pico1]
        driver   = "espico"
        port     = "auto"          # auto-detect via PalmSens SDK

        [instruments.pico2]
        driver   = "espico"
        port     = "auto"
    """)

    def _load(self, tmp_path: Path, content: str) -> Path:
        toml = tmp_path / "softae_config.toml"
        toml.write_text(content, encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        return toml

    def test_save_pico_ports_updates_file(self, tmp_path: Path) -> None:
        toml = self._load(tmp_path, self._TOML_TEMPLATE)
        cfg.save_pico_ports("COM3", "COM5")
        text = toml.read_text(encoding="utf-8")
        assert 'port     = "COM3"' in text
        assert 'port     = "COM5"' in text
        # 'auto' in comments is fine; the port value line must not say "auto"
        assert 'port     = "auto"' not in text

    def test_save_pico_ports_reloads_cache(self, tmp_path: Path) -> None:
        self._load(tmp_path, self._TOML_TEMPLATE)
        cfg.save_pico_ports("COM7", "COM9")
        instr = cfg.instruments()
        assert instr["pico1"]["port"] == "COM7"
        assert instr["pico2"]["port"] == "COM9"

    def test_save_pico_ports_preserves_comments(self, tmp_path: Path) -> None:
        toml = self._load(tmp_path, self._TOML_TEMPLATE)
        cfg.save_pico_ports("COM3", "COM5")
        text = toml.read_text(encoding="utf-8")
        assert "espico" in text  # driver line intact
        assert "[instruments.pico1]" in text
        assert "[instruments.pico2]" in text

    def test_save_pico_ports_roundtrip(self, tmp_path: Path) -> None:
        self._load(tmp_path, self._TOML_TEMPLATE)
        cfg.save_pico_ports("COM11", "COM13")
        cfg.save_pico_ports("COM3", "COM5")
        instr = cfg.instruments()
        assert instr["pico1"]["port"] == "COM3"
        assert instr["pico2"]["port"] == "COM5"


# ── Piezo config canonical/legacy compatibility tests ──────────────────────


class TestPiezoConfigCanonicalization:
    def _load(self, tmp_path: Path, content: str) -> Path:
        toml = tmp_path / "softae_config.toml"
        toml.write_text(content, encoding="utf-8")
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None
        cfg.load(toml)
        return toml

    def test_piezo_enabled_prefers_instruments_section(self, tmp_path: Path) -> None:
        self._load(
            tmp_path,
            textwrap.dedent(
                """\
                [instruments.piezo]
                enabled = true

                [piezo]
                enabled = false
                frequency_hz = 900
                """
            ),
        )
        section = cfg.piezo_config()
        assert section["enabled"] is True
        assert section["frequency_hz"] == 900

    def test_piezo_enabled_legacy_fallback_when_canonical_missing(self, tmp_path: Path) -> None:
        self._load(
            tmp_path,
            textwrap.dedent(
                """\
                [piezo]
                enabled = true
                """
            ),
        )
        section = cfg.piezo_config()
        assert section["enabled"] is True

    def test_save_piezo_config_syncs_enabled_flags(self, tmp_path: Path) -> None:
        toml = self._load(
            tmp_path,
            textwrap.dedent(
                """\
                [instruments.piezo]
                enabled = false

                [piezo]
                enabled = true
                channel = "A"
                """
            ),
        )
        cfg.save_piezo_config({"frequency_hz": 777})

        text = toml.read_text(encoding="utf-8")
        assert "[instruments.piezo]" in text
        assert "enabled = false" in text

        cfg.save_piezo_config({"enabled": True})
        text = toml.read_text(encoding="utf-8")
        assert "[instruments.piezo]" in text
        assert "enabled = true" in text


# ── CLI config hash logging test ────────────────────────────────────────────


class TestCLIConfigHash:
    def test_dry_run_shows_hash(self, tmp_path: Path, capsys, monkeypatch) -> None:
        import yaml

        # Create a config file in tmp_path so the loader picks it up
        config_file = tmp_path / "softae_config.toml"
        config_file.write_text("[paths]\ndata_root = './data'\n", encoding="utf-8")
        monkeypatch.setenv("SOFTAE_CONFIG", str(config_file))

        # Reset loader cache
        cfg._config = None
        cfg._config_path = None
        cfg._config_hash = None

        # Create a minimal workflow
        wf = {
            "name": "hash_test",
            "description": "test",
            "setup": [
                {"name": "s1", "instrument": "stage", "method": "move_to",
                 "params": {"x": 0, "y": 0}},
            ],
        }
        wf_path = tmp_path / "wf.yaml"
        wf_path.write_text(yaml.dump(wf), encoding="utf-8")

        from softae.workflows.__main__ import main

        rc = main(["--dry-run", str(wf_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Config hash:" in out
