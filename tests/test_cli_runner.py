"""Tests for the CLI workflow runner (``softae-run``)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from softae.workflows.__main__ import build_parser, main


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_workflow_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid workflow YAML and return its path."""
    wf = {
        "name": "test_cli_workflow",
        "description": "A tiny workflow for CLI tests",
        "variables": {"greeting": "hello"},
        "setup": [
            {
                "name": "set_temp",
                "instrument": "temp_controller",
                "method": "write_sp",
                "params": {"T_SP": 25, "print_flag": 0},
            },
        ],
        "loop": {
            "iterate_over": None,
            "steps": [
                {
                    "name": "read_temp",
                    "instrument": "temp_controller",
                    "method": "get_pv",
                    "params": {},
                },
            ],
        },
        "teardown": [
            {
                "name": "cooldown",
                "instrument": "temp_controller",
                "method": "write_sp",
                "params": {"T_SP": 10, "print_flag": 0},
            },
        ],
    }
    p = tmp_path / "test_wf.yaml"
    p.write_text(yaml.dump(wf, default_flow_style=False), encoding="utf-8")
    return p


@pytest.fixture()
def simple_workflow_json(tmp_path: Path) -> Path:
    """Write a minimal valid workflow JSON and return its path."""
    wf = {
        "name": "test_json_workflow",
        "description": "JSON format CLI test",
        "setup": [
            {
                "name": "move",
                "instrument": "stage",
                "method": "move_to",
                "params": {"x": 1.0, "y": 2.0},
            },
        ],
    }
    p = tmp_path / "test_wf.json"
    p.write_text(json.dumps(wf), encoding="utf-8")
    return p


# ── Parser tests ────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_basic_args(self):
        p = build_parser()
        args = p.parse_args(["my_workflow.yaml"])
        assert args.workflow == "my_workflow.yaml"
        assert args.mock is False
        assert args.real is False
        assert args.dry_run is False
        assert args.verbose is False
        assert args.log_dir == "./logs"

    def test_mock_and_real_mutually_exclusive(self):
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["w.yaml", "--mock", "--real"])

    def test_dry_run_flag(self):
        p = build_parser()
        args = p.parse_args(["w.yaml", "--dry-run", "--verbose"])
        assert args.dry_run is True
        assert args.verbose is True

    def test_log_dir(self):
        p = build_parser()
        args = p.parse_args(["w.yaml", "--log-dir", "/tmp/logs"])
        assert args.log_dir == "/tmp/logs"


# ── Dry-run tests ───────────────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_yaml(self, simple_workflow_yaml: Path, capsys):
        rc = main(["--dry-run", str(simple_workflow_yaml)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "test_cli_workflow" in out
        assert "dry-run" in out.lower()
        assert "Validation passed" in out

    def test_dry_run_json(self, simple_workflow_json: Path, capsys):
        rc = main(["--dry-run", str(simple_workflow_json)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "test_json_workflow" in out
        assert "Validation passed" in out

    def test_dry_run_lists_steps(self, simple_workflow_yaml: Path, capsys):
        main(["--dry-run", str(simple_workflow_yaml)])
        out = capsys.readouterr().out
        assert "set_temp" in out
        assert "read_temp" in out
        assert "cooldown" in out

    def test_dry_run_missing_file(self, capsys):
        rc = main(["--dry-run", "nonexistent.yaml"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "Error" in err


# ── Execution tests (mock instruments) ──────────────────────────────────────


class TestMockExecution:
    def test_run_mock_yaml(self, simple_workflow_yaml: Path, tmp_path: Path, capsys):
        log_dir = str(tmp_path / "logs")
        rc = main(["--mock", "--verbose", "--log-dir", log_dir,
                    str(simple_workflow_yaml)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "completed" in out.lower()

        # Check log file was created
        log_files = list(Path(log_dir).glob("*.jsonl"))
        assert len(log_files) == 1
        # Verify log content
        content = log_files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(content) >= 2
        first = json.loads(content[0])
        assert first["event"] == "config_hash"
        assert "hash" in first
        second = json.loads(content[1])
        assert "step" in second
        assert "result" in second

    def test_verbose_output(self, simple_workflow_yaml: Path, tmp_path: Path, capsys):
        log_dir = str(tmp_path / "logs")
        main(["--mock", "--verbose", "--log-dir", log_dir,
              str(simple_workflow_yaml)])
        out = capsys.readouterr().out
        # Verbose mode should show step names and state transitions
        assert "Starting:" in out or "✓" in out

    def test_state_transitions(self, simple_workflow_yaml: Path, tmp_path: Path, capsys):
        log_dir = str(tmp_path / "logs")
        main(["--mock", "--verbose", "--log-dir", log_dir,
              str(simple_workflow_yaml)])
        out = capsys.readouterr().out
        assert "IDLE" in out and "RUNNING" in out


# ── Validation tests ────────────────────────────────────────────────────────


class TestValidation:
    def test_validate_bad_instrument(self, tmp_path: Path, capsys):
        wf = {
            "name": "bad_inst",
            "description": "test",
            "setup": [
                {"name": "s1", "instrument": "nonexistent_device",
                 "method": "do_thing", "params": {}},
            ],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(wf), encoding="utf-8")
        rc = main(["--validate", str(p)])
        assert rc == 3
        err = capsys.readouterr().err
        assert "nonexistent_device" in err

    def test_validate_bad_method(self, tmp_path: Path, capsys):
        wf = {
            "name": "bad_method",
            "description": "test",
            "setup": [
                {"name": "s1", "instrument": "stage",
                 "method": "nonexistent_method", "params": {}},
            ],
        }
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump(wf), encoding="utf-8")
        rc = main(["--validate", str(p)])
        assert rc == 3
        err = capsys.readouterr().err
        assert "nonexistent_method" in err

    def test_validate_flag_parsed(self):
        p = build_parser()
        args = p.parse_args(["w.yaml", "--validate"])
        assert args.validate is True


# ── Error handling tests ────────────────────────────────────────────────────


class TestErrorHandling:
    def test_bad_workflow_file(self, capsys):
        rc = main(["nonexistent_file.yaml"])
        assert rc == 2

    def test_invalid_yaml(self, tmp_path: Path, capsys):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: bad\n", encoding="utf-8")  # no steps
        rc = main(["--dry-run", str(bad)])
        assert rc == 2
