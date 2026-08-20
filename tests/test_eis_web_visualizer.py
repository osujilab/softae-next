"""Tests for the softae.web EIS visualizer package.

Covers data adapters, figure builders, app construction, and CLI entry.
No Qt dependency (these are pure-Python / Dash tests).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis_data import EISResult
from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis_entry import EISEntry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_eis() -> EISResult:
    freq = np.geomspace(1e4, 1.0, 21)
    omega = 2 * np.pi * freq
    z = 5e4 + 1 / (1e-7 * (1j * omega) ** 0.7) + 1e6 / (1 + 1j * omega * 1e6 * 1e-10)
    return EISResult.from_arrays(
        channel=1,
        f=freq,
        z_real=z.real,
        z_imag_neg=-z.imag,
    )


@pytest.fixture()
def synthetic_fit() -> FitResult:
    return FitResult(
        model_name="simpleSalt",
        parameters=np.array([5e4, 1e-7, 0.7, 1e6, 1e-10]),
        R0=5e4,
        R1=1e6,
        R0_guess=5e4,
        R1_guess=1e6,
        z_indices=[0, 3],
        success=True,
    )


@pytest.fixture()
def single_entry(synthetic_eis, synthetic_fit) -> EISEntry:
    return EISEntry(
        label="Ch01 — test_run",
        eis=synthetic_eis,
        fit=synthetic_fit,
        sigma=1.23e-4,
        run_id="test_run_A",
    )


@pytest.fixture()
def entry_list(synthetic_eis, synthetic_fit) -> list[EISEntry]:
    entries = []
    for ch in range(1, 5):
        fit = FitResult(
            model_name="simpleSalt",
            parameters=np.array([5e4, 1e-7, 0.7, 1e6, 1e-10]),
            R0=5e4, R1=float(1e6 * ch),
            R0_guess=5e4, R1_guess=1e6,
            z_indices=[0, 3], success=True,
        )
        entries.append(
            EISEntry(
                label=f"Ch{ch:02d} — test_run_A",
                eis=synthetic_eis,
                fit=fit,
                sigma=1e-3 / ch,
                run_id="test_run_A",
            )
        )
    return entries


@pytest.fixture()
def tmp_db(tmp_path) -> Path:
    """Create a minimal softae DataStore db with one run and one measurement."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    runs_dir = tmp_path / "runs"

    db_path = db_dir / "softae.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE experiments (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            workflow_name TEXT NOT NULL,
            workflow_mode TEXT NOT NULL DEFAULT 'unknown',
            campaign TEXT NOT NULL DEFAULT 'dev',
            quality TEXT NOT NULL DEFAULT 'explore',
            pcb_name TEXT,
            eis_preset TEXT,
            config_snapshot_json TEXT NOT NULL DEFAULT '{}',
            config_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running'
        );
        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            channel INTEGER NOT NULL,
            electrode_x_mm REAL,
            electrode_y_mm REAL,
            timestamp TEXT NOT NULL,
            npts INTEGER,
            f_min_hz REAL,
            f_max_hz REAL,
            measurement_time_s REAL,
            eis_file_path TEXT,
            eis_params_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE fit_results (
            fit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            measurement_id INTEGER NOT NULL,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            R0 REAL,
            R1 REAL,
            sigma_S_per_cm REAL,
            electrode_L_cm REAL,
            electrode_t_cm REAL,
            electrode_w_cm REAL,
            success INTEGER NOT NULL DEFAULT 1,
            error_msg TEXT NOT NULL DEFAULT '',
            parameters_json TEXT NOT NULL DEFAULT '{}',
            fitted_at TEXT NOT NULL
        );
    """)
    # Insert a run
    conn.execute(
        "INSERT INTO experiments (run_id, started_at, workflow_name) VALUES (?, ?, ?)",
        ("run_2026_test", "2026-01-01T00:00:00", "test_workflow"),
    )
    # Save an EIS file
    freq = np.geomspace(1e4, 1, 21)
    omega = 2 * np.pi * freq
    z = 5e4 + 1 / (1e-7 * (1j * omega) ** 0.7) + 1e6 / (1 + 1j * omega * 1e6 * 1e-10)
    eis = EISResult.from_arrays(channel=1, f=freq, z_real=z.real, z_imag_neg=-z.imag)

    run_eis = runs_dir / "run_2026_test" / "eis"
    run_eis.mkdir(parents=True)
    eis_path = run_eis / "ch01.txt"
    eis.save(str(eis_path))
    rel_path = eis_path.relative_to(tmp_path)

    conn.execute(
        """INSERT INTO measurements
           (run_id, channel, timestamp, npts, f_min_hz, f_max_hz, eis_file_path)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("run_2026_test", 1, "2026-01-01T00:01:00", 21, 1.0, 1e4, str(rel_path)),
    )
    conn.execute(
        """INSERT INTO fit_results
           (measurement_id, run_id, model_name, R0, R1, sigma_S_per_cm, success,
            parameters_json, fitted_at)
           VALUES (1, 'run_2026_test', 'simpleSalt', 50000, 1000000, 1.23e-4, 1,
                   '[50000, 1e-7, 0.7, 1000000, 1e-10]', '2026-01-01T00:02:00')""",
    )
    conn.commit()
    conn.close()
    return tmp_path  # return project_dir (parent of db/)


# ---------------------------------------------------------------------------
# Tests: app factory
# ---------------------------------------------------------------------------

def test_create_app_returns_dash_app():
    from softae.web.app import create_app
    import dash
    app = create_app()
    assert isinstance(app, dash.Dash)


def test_create_app_with_db_path(tmp_db):
    from softae.web.app import create_app
    app = create_app(db_path=str(tmp_db / "db" / "softae.db"))
    assert app is not None


# ---------------------------------------------------------------------------
# Tests: DBAdapter
# ---------------------------------------------------------------------------

def test_db_adapter_list_runs(tmp_db):
    from softae.web.data_adapter import DBAdapter
    adapter = DBAdapter(tmp_db / "db" / "softae.db")
    runs = adapter.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_2026_test"
    assert runs[0]["n_measurements"] == 1


def test_db_adapter_list_runs_empty_path():
    from softae.web.data_adapter import DBAdapter
    adapter = DBAdapter("/nonexistent/softae.db")
    runs = adapter.list_runs()
    assert runs == []


def test_db_adapter_get_entries_returns_entries(tmp_db):
    from softae.web.data_adapter import DBAdapter
    adapter = DBAdapter(tmp_db / "db" / "softae.db")
    entries = adapter.get_entries(run_ids=["run_2026_test"])
    assert len(entries) == 1
    assert entries[0].eis.channel == 1


def test_db_adapter_get_entries_filters_channels(tmp_db):
    from softae.web.data_adapter import DBAdapter
    adapter = DBAdapter(tmp_db / "db" / "softae.db")
    # Channel 2 doesn't exist in fixture
    entries = adapter.get_entries(run_ids=["run_2026_test"], channels=[2])
    assert entries == []
    # Channel 1 exists
    entries = adapter.get_entries(run_ids=["run_2026_test"], channels=[1])
    assert len(entries) == 1


def test_db_adapter_accepts_project_dir(tmp_db):
    """DBAdapter should accept the project directory (not just the .db file)."""
    from softae.web.data_adapter import DBAdapter
    adapter = DBAdapter(tmp_db)
    runs = adapter.list_runs()
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# Tests: FileAdapter
# ---------------------------------------------------------------------------

def test_file_adapter_loads_eis_txt(tmp_path, synthetic_eis):
    p = tmp_path / "test_ch01.txt"
    synthetic_eis.save(str(p))
    from softae.web.data_adapter import FileAdapter
    entries = FileAdapter([p]).get_entries()
    assert len(entries) == 1
    assert entries[0].eis.npts == synthetic_eis.npts


def test_file_adapter_missing_file_skipped(tmp_path):
    from softae.web.data_adapter import FileAdapter
    entries = FileAdapter([tmp_path / "nonexistent.txt"]).get_entries()
    assert entries == []


def test_file_adapter_multiple_files(tmp_path, synthetic_eis):
    paths = []
    for i in range(3):
        p = tmp_path / f"ch{i:02d}.txt"
        synthetic_eis.save(str(p))
        paths.append(p)
    from softae.web.data_adapter import FileAdapter
    entries = FileAdapter(paths).get_entries()
    assert len(entries) == 3


# ---------------------------------------------------------------------------
# Tests: LiveAdapter
# ---------------------------------------------------------------------------

def test_live_adapter_returns_most_recent_run(tmp_db):
    from softae.web.data_adapter import LiveAdapter
    adapter = LiveAdapter(tmp_db / "db" / "softae.db")
    run_id = adapter.active_run_id()
    assert run_id == "run_2026_test"


def test_live_adapter_get_entries_nonempty(tmp_db):
    from softae.web.data_adapter import LiveAdapter
    adapter = LiveAdapter(tmp_db / "db" / "softae.db")
    entries = adapter.get_entries()
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Tests: figure builders
# ---------------------------------------------------------------------------

def test_overview_figure_built(entry_list):
    from softae.web.components import build_overview_figure
    import plotly.graph_objects as go
    fig = build_overview_figure(entry_list, cols=2)
    assert isinstance(fig, go.Figure)
    # Should have at least one trace per entry
    assert len(fig.data) >= len(entry_list)



def test_inspection_figure_built(single_entry):
    from softae.web.components import build_inspection_figure
    import plotly.graph_objects as go
    fig = build_inspection_figure(single_entry)
    assert isinstance(fig, go.Figure)
    assert fig.layout.height == 640


def test_inspection_figure_no_fit(synthetic_eis):
    from softae.web.components import build_inspection_figure
    import plotly.graph_objects as go
    entry = EISEntry(label="no fit", eis=synthetic_eis, fit=None, sigma=None)
    fig = build_inspection_figure(entry)
    assert isinstance(fig, go.Figure)


def test_conductivity_figure_built(entry_list):
    from softae.web.components import build_conductivity_figure
    import plotly.graph_objects as go
    fig = build_conductivity_figure(entry_list)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1



def test_arrhenius_figure_with_data():
    from softae.web.components import build_arrhenius_figure
    import plotly.graph_objects as go
    data = [
        {
            "channel": 1,
            "temperatures_C": [25.0, 40.0, 55.0, 70.0],
            "conductivities_S_per_cm": [1e-4, 2e-4, 4e-4, 8e-4],
            "Ea_eV": 0.42,
            "Ea_kJ_per_mol": 40.5,
            "ln_A": 3.7,
            "R_squared": 0.997,
            "fit_success": True,
        }
    ]
    fig = build_arrhenius_figure(data, unit="eV")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# Tests: layering — web must not drag in Qt / matplotlib-QtAgg
# ---------------------------------------------------------------------------

def test_web_imports_are_qt_free():
    """Importing the whole web layer must not pull in PySide6 or the QtAgg
    backend. Run in a clean subprocess so an already-imported Qt (from GUI
    tests in the same session) can't mask a real leak."""
    code = (
        "import sys;"
        "import softae.web.callbacks, softae.web.data_adapter,"
        " softae.web.components.overview, softae.web.components.inspection,"
        " softae.web.components.conductivity;"
        "assert 'PySide6' not in sys.modules, 'PySide6 leaked into web';"
        "assert 'matplotlib.backends.backend_qtagg' not in sys.modules,"
        " 'QtAgg backend leaked into web';"
        "print('web is Qt-free')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert "web is Qt-free" in result.stdout


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------

def test_cli_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "softae.web", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0
    assert "softae" in result.stdout.lower() or "port" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Tests: the package survives a base install (dash is an optional extra)
# ---------------------------------------------------------------------------

#: Prepended to a subprocess's source to make dash *look* uninstalled, which is
#: the base-install case the dev venv (which has ``[web]``) cannot otherwise
#: reproduce.  A meta_path finder is the honest simulation: it fails at exactly
#: the point a real missing distribution would, with the same exception type
#: and the same ``.name``.
_BLOCK_DASH = """
import sys


class _NoDash:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("dash", "dash_bootstrap_components"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _NoDash())
for _m in [m for m in sys.modules if m.split(".")[0].startswith("dash")]:
    del sys.modules[_m]
"""


def _run_without_dash(argv: list[str], code: str | None = None):
    """Run a subprocess with dash blocked — either ``-c code`` or ``argv``."""
    cmd = [sys.executable]
    if code is not None:
        cmd += ["-c", _BLOCK_DASH + code]
    else:
        cmd += ["-c", _BLOCK_DASH + (
            "import runpy, sys\n"
            f"sys.argv = {argv!r}\n"
            "runpy.run_module('softae.web', run_name='__main__')\n"
        )]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def test_web_package_imports_without_dash():
    """``import softae.web`` must not require the optional [web] extra.

    It is imported by ``__main__`` after argument parsing, so a module-scope
    dash import here turns a base install into a raw ModuleNotFoundError
    traceback the moment anyone runs ``softae-web``.
    """
    result = _run_without_dash(
        [],
        code=(
            "import softae.web, sys\n"
            "assert 'dash' not in sys.modules, 'dash leaked into softae.web'\n"
            "print('softae.web is dash-free')\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    assert "softae.web is dash-free" in result.stdout


def test_web_cli_without_dash_exits_nonzero_with_install_hint():
    result = _run_without_dash(["softae.web", "--no-browser"])
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    assert "softae[web]" in result.stderr
    assert "Traceback" not in result.stderr


def test_create_app_still_reachable_from_package():
    """PEP 562 laziness must not amputate the documented public name."""
    from softae.web import create_app

    assert callable(create_app)
    assert "create_app" in dir(sys.modules["softae.web"])
