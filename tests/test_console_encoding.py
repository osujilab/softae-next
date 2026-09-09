"""The console can carry SoftAE's own output, and nobody has to remember to ask.

The rig runs on Windows, where ``sys.stdout`` defaults to the ANSI code page and any
of ``⚠ σ δ Ω ≲ →`` raises ``UnicodeEncodeError`` on the way out. The characters are not
decoration: they are in the gates' own rejection text, so an unguarded caller dies
*on its own warning*, part-way through, with a traceback that names the ``print`` rather
than the problem. That is exactly how the first shadow probe run ended — a ``tan δ``
inside ``run_gates`` (see ``test_tool_shadow_rehearse.py``'s tan-δ test).

The guarantee used to be fifteen remembered ``main()`` calls, which left two whole
classes of caller outside it — a bare library import, and pytest. It is now automatic:
``softae/__init__.py`` widens the console at import, and only when the console's codec
cannot carry the text. Four kinds of test, and they prove different things:

* the **adoption** tests read source and cannot prove anything runs. They pin that no
  entrypoint can be added — or an existing one rewritten — without the rule being
  noticed, which is the failure mode that actually happened: six of ten tools had the
  guard and four did not, silently. They are now belt to the automatic braces.
* the **behavioural** tests prove ``use_utf8_console`` does what its name says, against
  a real cp1252 stream, with a control half that shows the stream would have raised —
  and that ``only_if_needed`` leaves a capable stream completely alone.
* the **gap** tests run a **subprocess** with a genuine cp1252 stdout, because that is
  the only way to be immune to pytest's own capture. Gap 1 is a bare library import;
  gap 2 is pytest itself, whose default capture *swallowed* the crash and reported
  green while ``pytest -s`` reproduced it.
* each gap test carries its **positive control** — the same subprocess with
  ``SOFTAE_NO_CONSOLE_UTF8=1`` — which must fail. A test that is green both ways is
  vacuous, and the opt-out makes the control exact rather than simulated.
"""

from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from softae import OPT_OUT_ENV, stream_can_carry_any_text, use_utf8_console

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Characters from real gate/report output, one per family: prose δ, the warning marker,
#: the conductivity symbol, and the upper-bound sign the σ demotion prints.
NON_CP1252 = "tan δ ⚠ σ ≲"


def _entry_points() -> list[tuple[str, str]]:
    """``[(script-name, "module:function"), …]`` from ``pyproject.toml``.

    Both ``[project.scripts]`` and ``gui_scripts``: the GUI launcher prints the same
    characters and is no less an entrypoint for being windowed. Read from the file
    rather than listed here, so a new entrypoint is covered the moment it is declared
    instead of when someone remembers this file.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    targets = dict(data["project"]["scripts"])
    targets.update(data["project"].get("entry-points", {}).get("gui_scripts", {}))
    return sorted(targets.items())


def _modules_with_a_main_block() -> list[str]:
    """Every ``src/softae`` module that can be run as a script.

    ``[project.scripts]`` is not the whole population: ``tools/measurability_sweep.py``
    has a ``__main__`` block, is run as ``python -m``, and is declared nowhere — so it
    was outside the adoption test entirely. Sourced from the tree for the same reason
    the table above is sourced from the file.
    """
    out = []
    for path in sorted((REPO_ROOT / "src" / "softae").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if '__name__ == "__main__"' in text:
            rel = path.relative_to(REPO_ROOT / "src").with_suffix("")
            out.append(".".join(rel.parts))
    return out


SCRIPTS = _entry_points()
MAIN_MODULES = _modules_with_a_main_block()


def test_the_script_table_is_read_and_not_empty():
    # Guards the parametrization itself: a rename of [project.scripts] would otherwise
    # turn every adoption test below into a silent zero-case pass.
    assert len(SCRIPTS) >= 13


def test_the_main_module_scan_is_not_empty():
    # Same guard for the tree scan: a moved package would zero it out silently.
    assert len(MAIN_MODULES) >= 14


@pytest.mark.parametrize("name,target", SCRIPTS, ids=[n for n, _ in SCRIPTS])
def test_console_script_main_calls_use_utf8_console(name, target):
    module_path, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_path), func_name)
    assert "use_utf8_console" in inspect.getsource(func), (
        f"{name} ({target}) prints without reconfiguring the console first"
    )


@pytest.mark.parametrize("dotted", MAIN_MODULES, ids=MAIN_MODULES)
def test_runnable_module_calls_use_utf8_console(dotted):
    """A ``__main__`` block that no ``[project.scripts]`` entry names is still an
    entrypoint. ``measurability_sweep`` is the one that was already in this position:
    it calls the guard, and nothing said it had to."""
    source = inspect.getsource(importlib.import_module(dotted))
    assert "use_utf8_console" in source, (
        f"{dotted} is runnable as a script but never reconfigures the console"
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


def test_only_if_needed_still_widens_a_stream_that_cannot_carry_the_text(tmp_path,
                                                                        monkeypatch):
    """The conditional mode is not a no-op: on the rig's own console it fires."""
    stream = open(tmp_path / "cp1252.out", "w", encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    try:
        use_utf8_console(only_if_needed=True)
        assert stream.encoding.lower().replace("-", "") == "utf8"
        assert stream.errors == "replace"
        print(NON_CP1252)  # must not raise
    finally:
        stream.close()


def test_only_if_needed_leaves_a_capable_stream_completely_alone(tmp_path, monkeypatch):
    """A host that chose UTF-8 keeps exactly what it chose — errors included.

    This is what makes the import-time mutation defensible rather than a library
    helping itself to the host's streams: on every stream that can already carry the
    text, nothing happens at all.
    """
    stream = open(tmp_path / "utf8.out", "w", encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    try:
        use_utf8_console(only_if_needed=True)
        assert stream.errors == "strict", "a capable stream was reconfigured anyway"
    finally:
        stream.close()


@pytest.mark.parametrize(
    "encoding,expected",
    [("utf-8", True), ("utf8", True), ("UTF-8", True), ("utf-16", True),
     ("cp1252", False), ("latin-1", False), ("ascii", False), ("cp932", False),
     ("no-such-codec", False), ("", False), (None, False)],
)
def test_stream_can_carry_any_text_answers_by_codec_not_by_spelling(encoding, expected):
    # The predicate reads exactly one attribute, so a namespace is the honest double;
    # a real TextIOWrapper cannot be made to claim an arbitrary codec.
    assert stream_can_carry_any_text(SimpleNamespace(encoding=encoding)) is expected


def test_a_stream_with_no_encoding_attribute_is_unknown_not_safe():
    """Unknown must not be spelled with the same token as checked-and-clean."""
    assert stream_can_carry_any_text(object()) is False


# --------------------------------------------------------------------------- #
# The two gaps. Subprocess, because pytest's own capture hides both of them.
# --------------------------------------------------------------------------- #

#: A log call through the same sink every one of the ~67 non-ASCII sites uses:
#: structlog's default ``PrintLoggerFactory`` ``print()``s to ``sys.stdout``. Written
#: as escapes so this snippet survives being handed to a cp1252 subprocess argv.
LOG_A_SIGMA = (
    "import softae, structlog\n"
    "structlog.get_logger('probe').info('eis_k_config_unarmed', "
    "msg='absolute \u03c3 is unqualified')\n"
    "print('\u2265 1 per decade')\n"
)


def _run(code_or_args, *, opted_out: bool, cwd=None, module=False):
    """Run a subprocess whose stdout/stderr are a genuine cp1252 console.

    ``PYTHONIOENCODING`` is the honest double for the rig's console: it produces a real
    ``TextIOWrapper`` at cp1252/strict, which is what ``sys.stdout`` is on that machine.
    """
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    if opted_out:
        env[OPT_OUT_ENV] = "1"
    else:
        env.pop(OPT_OUT_ENV, None)
    argv = [sys.executable] + (list(code_or_args) if module else ["-c", code_or_args])
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, timeout=180)


def test_gap1_a_bare_library_import_no_longer_dies_on_its_own_log_text():
    """No ``main()`` runs here — this is the script/notebook/REPL caller.

    The real instance is ``cell_config`` on the *shipped* three-electrode config,
    which logs a ``σ``; it sits in the per-sample autonomous path.
    """
    done = _run(LOG_A_SIGMA, opted_out=False)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in done.stderr


def test_gap1_positive_control_the_same_import_dies_with_the_guard_opted_out():
    """Without this, the test above is green whether or not anything guards."""
    done = _run(LOG_A_SIGMA, opted_out=True)
    assert done.returncode != 0
    assert b"UnicodeEncodeError" in done.stderr


def test_gap2_a_log_call_under_pytest_dash_s_no_longer_crashes(tmp_path):
    """pytest was the second gap, and the worse one: default capture replaces
    ``sys.stdout`` with a UTF-8 file, so the crash was *swallowed* and the suite
    reported green. ``-s`` is where the real console shows through — which is why
    this runs a nested pytest rather than asserting inside the current one."""
    probe = tmp_path / "test_probe_sigma.py"
    probe.write_text("def test_probe():\n    " + LOG_A_SIGMA.replace("\n", "\n    "),
                     encoding="utf-8")
    args = ["-m", "pytest", str(probe), "-p", "no:cacheprovider", "-q", "-s"]
    done = _run(args, opted_out=False, cwd=str(tmp_path), module=True)
    assert done.returncode == 0, done.stdout.decode("utf-8", "replace")

    control = _run(args, opted_out=True, cwd=str(tmp_path), module=True)
    assert control.returncode != 0, "pytest -s no longer reproduces the gap at all"
    assert b"UnicodeEncodeError" in control.stdout + control.stderr
