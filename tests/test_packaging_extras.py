"""The GUI's dependencies are an *extra*, and these tests are the only guard.

A metadata edit is invisible to every other tier of the suite: pip does not
uninstall a distribution because it left a requirement list, so the developer's
editable venv keeps importing PySide6, opencv and qasync exactly as before, and
a mistake here only surfaces in a *fresh* environment built without ``[gui]``.
There is no fresh environment to build inside a test run — so what can be
asserted is the metadata itself, read straight out of ``pyproject.toml``.

The property being protected: ``pip install softae`` on a headless box gives a
working CLI (the workflow runner, the eleven ``softae-*`` tools, the web
visualizer), and ``pip install softae[gui]`` adds the desktop application on
top.  Qt-free and tomllib-only, so this file costs nothing to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Console scripts that legitimately need Qt, so are allowed a softae.gui target.
GUI_SCRIPTS = {"softae-gui", "softae-deposition"}

#: The three distributions the [gui] extra owns.
GUI_PACKAGES = ("PySide6", "opencv-python", "qasync")


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


PROJECT = _pyproject()["project"]


def _names(requirements) -> set[str]:
    """Distribution names from PEP 508 strings, lowercased and normalised.

    ``"PySide6>=6.6"`` -> ``"pyside6"``; ``"softae[gui]"`` -> ``"softae"``.
    """
    out = set()
    for req in requirements:
        head = req.split(";")[0].strip()
        for sep in ("<", ">", "=", "!", "~", " ", "["):
            head = head.split(sep)[0]
        out.add(head.strip().lower().replace("_", "-"))
    return out


def test_gui_extra_exists_and_lists_the_three_gui_packages():
    gui = PROJECT["optional-dependencies"]["gui"]
    assert _names(GUI_PACKAGES) <= _names(gui), gui


def test_gui_dependencies_are_optional_not_required():
    """None of the three may appear in the unconditional dependency list.

    This is the assertion that makes ``pip install softae`` headless-clean;
    PySide6 alone is a ~100 MB download that a rig-less analysis box, a CI
    runner or a headless workflow host has no use for.
    """
    required = _names(PROJECT["dependencies"])
    leaked = _names(GUI_PACKAGES) & required
    assert not leaked, f"{sorted(leaked)} must live only in the [gui] extra"


def test_nest_asyncio_stays_unconditional():
    """nest-asyncio is the *server* layer's, not the GUI's.

    ``softae/server/base_instrument.py:140`` imports it inside ``__enter__`` —
    the sync context-manager convenience every driver inherits and every
    headless script uses.  Moving it into ``[gui]`` would break the CLI tools
    on a base install, which is the exact regression this extra is meant to
    avoid, so it is pinned here rather than left to a reviewer's memory.
    """
    assert "nest-asyncio" in _names(PROJECT["dependencies"])
    assert "nest-asyncio" not in _names(PROJECT["optional-dependencies"]["gui"])


def test_dev_extra_pulls_in_the_gui_extra():
    """``pip install -e ".[dev]"`` must keep working unchanged.

    ~50 test files import PySide6 and pytest-qt cannot run without it, so the
    dev extra self-references ``softae[gui]`` rather than the documented
    install string changing under everyone.
    """
    dev = PROJECT["optional-dependencies"]["dev"]
    assert any(req.replace(" ", "").lower().startswith("softae[") and "gui" in req
               for req in dev), dev


@pytest.mark.parametrize(
    "name,target",
    sorted(PROJECT["scripts"].items()),
    ids=sorted(PROJECT["scripts"]),
)
def test_cli_console_scripts_do_not_target_the_gui_package(name, target):
    """Every non-GUI entry point must live outside ``softae.gui``.

    Parametrized off the real table, so a new tool that reaches into the GUI
    package — the way a headless install would discover it, i.e. by crashing —
    fails here instead.
    """
    if name in GUI_SCRIPTS:
        assert target.startswith("softae.gui"), f"{name} was expected to be a GUI script"
        return
    assert not target.startswith("softae.gui"), (
        f"{name} -> {target} would need the [gui] extra to run"
    )
