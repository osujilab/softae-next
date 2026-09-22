"""``softae-catalog init`` — seeding a data root, and refusing to overwrite one.

Covers the bootstrap CLI and the packaged defaults it copies: that every
shipped file arrives, byte for byte; that an existing catalog is never
overwritten and a refused run writes nothing at all; and that the loader
refuses a catalog that came back empty instead of returning it.

Every test writes into ``tmp_path``. The real data root is an instance's own
catalog, and this command's whole job is not to touch one.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from softae.catalog_defaults import (
    CATALOG_FILES,
    PLACEHOLDER_NOTICE,
    default_file,
    load_defaults,
)
from softae.tools.catalog_init import main, seed_catalog


def shipped_bytes(name: str) -> bytes:
    """The bytes of one packaged catalog file."""
    with default_file(name) as path:
        return path.read_bytes()


def test_seed_catalog_writes_every_shipped_file_byte_for_byte(tmp_path: Path):
    """An empty destination receives all four files unmodified, and the staging
    directory they were copied through is gone.
    """
    written = seed_catalog(tmp_path / "data")

    assert [p.name for p in written] == list(CATALOG_FILES)
    for name in CATALOG_FILES:
        assert (tmp_path / "data" / name).read_bytes() == shipped_bytes(name)
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


def test_seed_catalog_creates_a_destination_that_does_not_exist(tmp_path: Path):
    """A clone has no ``data/`` at all, so the directory is made on the way."""
    dest = tmp_path / "nested" / "data"
    seed_catalog(dest)
    assert (dest / "tasks.toml").is_file()


def test_seed_catalog_refuses_an_existing_file_and_writes_nothing(tmp_path: Path):
    """One pre-existing target aborts the whole copy, not just that file: a
    half-seeded root mixes the instance's catalog with the placeholder.
    """
    dest = tmp_path / "data"
    dest.mkdir()
    (dest / "chemicals.csv").write_text("mine,not,yours\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="chemicals.csv"):
        seed_catalog(dest)

    assert (dest / "chemicals.csv").read_text(encoding="utf-8") == "mine,not,yours\n"
    assert sorted(p.name for p in dest.iterdir()) == ["chemicals.csv"]


def failing_on_call(primitive, stop_at: int):
    """*primitive*, but raising ``OSError`` on its *stop_at*-th call.

    Stands in for the disk filling, an antivirus lock or a permission change
    partway through the four copies — the cases no pre-flight check can rule
    out and the only way the rollback path is reachable at all.
    """
    calls = {"n": 0}

    def wrapped(src, dst):
        calls["n"] += 1
        if calls["n"] == stop_at:
            raise OSError(f"injected failure on call {stop_at}")
        return primitive(src, dst)

    return wrapped


def test_seed_catalog_copy_failure_partway_leaves_the_destination_empty(
        tmp_path: Path, monkeypatch):
    """A copy dying on the third file writes none of the four, and stages nowhere
    a later reader could mistake for a catalog.
    """
    dest = tmp_path / "data"
    monkeypatch.setattr(shutil, "copyfile", failing_on_call(shutil.copyfile, 3))

    with pytest.raises(OSError, match="injected failure"):
        seed_catalog(dest)

    assert list(dest.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


def test_seed_catalog_move_failure_removes_the_files_it_had_already_placed(
        tmp_path: Path, monkeypatch):
    """The rollback arm: two files are in place when the third move fails, and
    both are taken back out — the destination directory itself is left alone.
    """
    dest = tmp_path / "data"
    monkeypatch.setattr(os, "replace", failing_on_call(os.replace, 3))

    with pytest.raises(OSError, match="injected failure"):
        seed_catalog(dest)

    assert list(dest.iterdir()) == []
    assert [p.name for p in tmp_path.iterdir()] == ["data"]


def test_seed_catalog_writes_a_catalog_that_loads_non_empty(tmp_path: Path):
    """The seeded files parse; a copy that arrived unreadable would not show up
    in a file listing, only in a catalog that silently holds nothing.
    """
    from softae.core.formulation import ChemicalCatalog, SolutionCatalog
    from softae.core.task_catalog import TaskCatalog

    dest = tmp_path / "data"
    seed_catalog(dest)

    assert TaskCatalog.load_toml(dest / "tasks.toml").list_names()
    assert ChemicalCatalog.load_csv(dest / "chemicals.csv").list_names()
    assert SolutionCatalog.load_csv(dest / "solutions.csv").list_names()


def test_catalog_init_cli_seeds_once_then_refuses(tmp_path: Path, capsys):
    """Exit 0 the first time, non-zero the second, with the file named."""
    dest = tmp_path / "data"

    assert main(["init", "--dest", str(dest)]) == 0
    first = capsys.readouterr().out
    assert PLACEHOLDER_NOTICE in first
    assert "tasks.toml" in first

    assert main(["init", "--dest", str(dest)]) == 2
    second = capsys.readouterr()
    assert "refusing to overwrite" in second.err
    assert "tasks.toml" in second.err


def test_catalog_init_cli_without_dest_uses_the_configured_data_root(
        tmp_path: Path, monkeypatch):
    """``--dest`` is a test affordance; the default is the configured root."""
    from softae.config import loader

    root = tmp_path / "configured"
    monkeypatch.setattr(loader, "data_root", lambda: root)

    assert main(["init"]) == 0
    assert (root / "solutions.csv").is_file()


def test_load_defaults_returns_the_three_catalogs_populated():
    """The packaged catalogs load, and hold the names the examples reference."""
    defaults = load_defaults()

    assert "single_drop_simul" in defaults.tasks.list_names()
    assert "Lithium chloride" in defaults.chemicals.list_names()
    assert "LiCl 10M" in defaults.solutions.list_names()


def test_load_defaults_refuses_a_catalog_that_came_back_empty(monkeypatch):
    """Every loader here turns a missing file into an empty catalog rather than
    raising, so without this refusal "nothing shipped" would read as "fine".
    """
    from softae.core.task_catalog import TaskCatalog

    monkeypatch.setattr(TaskCatalog, "load_toml",
                        classmethod(lambda cls, path: TaskCatalog()))

    with pytest.raises(RuntimeError, match="tasks.toml"):
        load_defaults()


def test_default_file_refuses_a_name_the_package_does_not_ship():
    """A typo names the file it wanted, rather than resolving to nothing."""
    with pytest.raises(KeyError, match="softae_config.toml"):
        with default_file("softae_config.toml"):
            pass
