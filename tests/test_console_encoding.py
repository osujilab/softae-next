"""Every console entrypoint reconfigures its console before it prints anything.

The rig runs on Windows, where ``sys.stdout`` defaults to the ANSI code page and any
of ``⚠ σ δ Ω ≲ →`` raises ``UnicodeEncodeError`` on the way out. The characters are not
decoration: they are in the gates' own rejection text, so an unguarded entrypoint dies
*on its own warning*, part-way through, with a traceback that names the ``print`` rather
than the problem. That is exactly how the first shadow probe run ended — a ``tan δ``
inside ``run_gates`` (see ``test_tool_shadow_rehearse.py``'s tan-δ test).

Two tests, and they prove different things:

* the **adoption** test reads source and cannot prove the call runs. It pins that no
  entrypoint can be added — or an existing one rewritten — without the rule being
  noticed, which is the failure mode that actually happened: six of ten tools had the
  guard and four did not, silently.
* the **behavioural** test proves ``use_utf8_console`` does what its name says, against
  a real cp1252 stream, with a control half that shows the stream would have raised.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

from softae.tools import use_utf8_console

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Characters from real gate/report output, one per family: prose δ, the warning marker,
#: the conductivity symbol, and the upper-bound sign the σ demotion prints.
NON_CP1252 = "tan δ ⚠ σ ≲"


def _console_scripts() -> list[tuple[str, str]]:
    """``[(script-name, "module:function"), …]`` from ``[project.scripts]``.

    Read from ``pyproject.toml`` rather than listed here, so a new entrypoint is
    covered the moment it is declared instead of when someone remembers this file.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        scripts = tomllib.load(fh)["project"]["scripts"]
    return sorted(scripts.items())


SCRIPTS = _console_scripts()


def test_the_script_table_is_read_and_not_empty():
    # Guards the parametrization itself: a rename of [project.scripts] would otherwise
    # turn every adoption test below into a silent zero-case pass.
    assert len(SCRIPTS) >= 10


@pytest.mark.parametrize("name,target", SCRIPTS, ids=[n for n, _ in SCRIPTS])
def test_console_script_main_calls_use_utf8_console(name, target):
    module_path, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_path), func_name)
    assert "use_utf8_console" in inspect.getsource(func), (
        f"{name} ({target}) prints without reconfiguring the console first"
    )


def test_a_cp1252_stdout_raises_before_the_guard_and_not_after(tmp_path, monkeypatch):
    """A real file opened cp1252/strict is the honest double for a Windows console.

    A ``StringIO`` would accept every character regardless, and a mock would only
    replay whatever this test asserted; a genuine ``TextIOWrapper`` has the same
    ``reconfigure`` the real stream has, and refuses the same characters.
    """
    path = tmp_path / "cp1252.out"
    stream = open(path, "w", encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    try:
        # Control: without the guard this is the crash the probe run hit.
        with pytest.raises(UnicodeEncodeError):
            print(NON_CP1252)
            stream.flush()

        use_utf8_console()
        print(NON_CP1252)  # must not raise
        stream.flush()
    finally:
        stream.close()

    assert NON_CP1252 in path.read_text(encoding="utf-8")
