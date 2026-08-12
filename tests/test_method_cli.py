"""Tests for the `softae-method` lifecycle CLI (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.config import loader
from softae.core.data_store import DataStore
from softae.core.recipe_registry import Recipe, RecipeRegistry
from softae.core.task_catalog import Task, TaskCatalog
from softae.tools import method as mcli


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    """Point the CLI's loader paths at throwaway files, seeded with a catalog."""
    tasks = tmp_path / "tasks.toml"
    recipes = tmp_path / "recipes.toml"

    cat = TaskCatalog()
    cat.add(Task(name="fresh", instrument="liquid_handler", method="single_drop_simul"))
    cat.add(Task(
        name="ready", instrument="liquid_handler", method="startup_flush",
        maturity="tested", evidence={"tests": ["tests/x.py::t"]},
    ))
    cat.save_toml(tasks)

    reg = RecipeRegistry()
    reg.add(Recipe(name="sweep", maturity="tested", methods=["fresh", "ready"]))
    reg.save_toml(recipes)

    monkeypatch.setattr(loader, "tasks_toml_path", lambda: tasks)
    monkeypatch.setattr(loader, "recipes_toml_path", lambda: recipes)
    monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "proj"))
    monkeypatch.setattr(loader, "data_db_filename", lambda: "softae.db")
    return tmp_path, tasks, recipes


def _reload(tasks: Path) -> TaskCatalog:
    return TaskCatalog.load_toml(tasks)


# ── status ───────────────────────────────────────────────────────────────────

def test_status_table_ok(env, capsys):
    assert mcli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "fresh" in out and "sweep" in out and "recipe" in out


def test_status_unknown_returns_3(env):
    assert mcli.main(["status", "nope"]) == 3


def test_status_recipe_effective_capped(env, capsys):
    # sweep own=tested, but 'fresh' is draft -> effective draft.
    mcli.main(["status", "sweep"])
    out = capsys.readouterr().out
    assert "effective=draft" in out


# ── promote ──────────────────────────────────────────────────────────────────

def test_promote_forward_ok(env):
    _, tasks, _ = env
    assert mcli.main(["promote", "fresh", "prototype"]) == 0
    assert _reload(tasks).get("fresh").maturity == "prototype"


def test_promote_to_tested_without_tests_blocked(env):
    assert mcli.main(["promote", "fresh", "tested"]) == 2  # no linked tests


def test_promote_unknown_stage(env):
    assert mcli.main(["promote", "fresh", "bogus"]) == 2


# ── test (linked pytest) ─────────────────────────────────────────────────────

def test_test_promotes_on_green(env, monkeypatch):
    _, tasks, _ = env

    class _Proc:
        returncode = 0
    monkeypatch.setattr(mcli.subprocess, "run", lambda *a, **k: _Proc())

    assert mcli.main(["test", "ready"]) == 0  # 'ready' already tested → stays
    # A method with tests linked, below tested, gets promoted:
    cat = _reload(tasks)
    cat.get("fresh").evidence["tests"] = ["tests/x.py::t"]
    cat.save_toml(tasks)
    assert mcli.main(["test", "fresh"]) == 0
    assert _reload(tasks).get("fresh").maturity == "tested"


def test_test_fails_does_not_promote(env, monkeypatch):
    _, tasks, _ = env
    cat = _reload(tasks)
    cat.get("fresh").evidence["tests"] = ["tests/x.py::t"]
    cat.save_toml(tasks)

    class _Proc:
        returncode = 1
    monkeypatch.setattr(mcli.subprocess, "run", lambda *a, **k: _Proc())

    assert mcli.main(["test", "fresh"]) == 1
    assert _reload(tasks).get("fresh").maturity == "draft"


def test_test_no_linked_tests(env):
    assert mcli.main(["test", "fresh"]) == 2  # 'fresh' has no evidence.tests


# ── sign-off ─────────────────────────────────────────────────────────────────

def test_sign_off_validates(env):
    tmp_path, tasks, _ = env
    ds = DataStore(project_dir=str(tmp_path / "proj"), db_filename="softae.db")
    run_id = ds.start_run("cast", config_hash="cfg99")
    ds.close()

    rc = mcli.main(["sign-off", "ready", "--run", run_id, "--by", "pshaps",
                    "--note", "wells 21-24"])
    assert rc == 0
    t = _reload(tasks).get("ready")
    assert t.maturity == "validated"
    assert t.evidence["validated_run_id"] == run_id
    assert t.evidence["config_hash"] == "cfg99"


def test_sign_off_rejects_missing_run(env):
    assert mcli.main(["sign-off", "ready", "--run", "no_such_run", "--by", "p"]) == 2


# ── versions ─────────────────────────────────────────────────────────────────

def test_versions_lists_chain(env, capsys):
    _, tasks, _ = env
    from softae.core import lifecycle as lc
    cat = _reload(tasks)
    lc.supersede("fresh", Task(name="fresh", instrument="i", method="mm2"), cat)
    cat.save_toml(tasks)

    assert mcli.main(["versions", "fresh"]) == 0
    out = capsys.readouterr().out
    assert "fresh@v1" in out and "current" in out


def test_versions_unknown(env):
    assert mcli.main(["versions", "nope"]) == 3
