"""SoftAE — Soft-matter Autonomous Experimentation platform.

Importing this package makes the process's console able to carry SoftAE's own
output. See :func:`use_utf8_console`; the *why* is in
:func:`_widen_console_if_it_cannot_carry_our_output`.
"""

from __future__ import annotations

import codecs
import os
import sys
from typing import Any

__version__ = "0.1.0"

#: Set to a truthy value to suppress the automatic widening performed at import.
#: It does **not** disable an explicit ``use_utf8_console()`` call — that is the
#: caller asking for it, and the fifteen console entrypoints still ask.
OPT_OUT_ENV = "SOFTAE_NO_CONSOLE_UTF8"

_FALSEY = {"", "0", "false", "no", "off"}


def _opted_out() -> bool:
    return os.environ.get(OPT_OUT_ENV, "").strip().lower() not in _FALSEY


def stream_can_carry_any_text(stream: Any) -> bool:
    """Is *stream*'s codec one that can encode every character we might print?

    Only the UTF family qualifies, and the test is by *normalised codec name*
    rather than by string comparison, so ``UTF-8``/``utf8``/``u8`` all answer the
    same. A stream whose encoding cannot be determined answers ``False``: unknown
    is not the same token as safe, and spelling it as safe is how a guard becomes
    decorative.
    """
    encoding = getattr(stream, "encoding", None)
    if not isinstance(encoding, str) or not encoding:
        return False
    try:
        return codecs.lookup(encoding).name.startswith("utf")
    except LookupError:
        return False


def use_utf8_console(*, only_if_needed: bool = False) -> None:
    """Make stdout/stderr survive non-ASCII output on a Windows console.

    SoftAE prints characters outside cp1252 in the ordinary course of reporting —
    ``⚠`` in warnings, ``σ``/``Ω``/``δ`` throughout the EIS reporting, ``≲`` for an
    upper bound, ``→`` in the "run this next" hints, and ``≥`` inside
    ``CalibrationSet.describe()``. The rig runs on Windows, where ``sys.stdout``
    defaults to the ANSI code page, and printing any of them raises
    ``UnicodeEncodeError``.

    That is not cosmetic. It surfaced as ``softae-commission derive`` **crashing with a
    traceback** part-way through a real derivation, after the artifacts had been read
    and before the calibration was written — the command appeared to fail at the
    analysis, when in fact it had failed at the ``print``. A tool that dies on its own
    warning text is worse than one that cannot warn.

    ``errors="replace"`` rather than a strict re-encode: a console that genuinely
    cannot render a glyph should show a substitute, never abort the command that was
    trying to tell the operator something.

    ``reconfigure`` mutates the ``TextIOWrapper`` **in place** rather than replacing
    it, so anything already holding a reference — a ``logging.StreamHandler`` bound by
    an earlier ``basicConfig``, a ``structlog`` ``PrintLogger`` constructed at import —
    follows along. Ordering against logging setup is therefore a non-issue.

    With *only_if_needed*, streams whose codec can already carry any text are left
    completely untouched. That is the mode the automatic import-time call uses, so a
    host process that deliberately chose UTF-8 is not silently re-encoded.
    """
    for stream in (sys.stdout, sys.stderr):
        if only_if_needed and stream_can_carry_any_text(stream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # A redirected or wrapped stream may refuse. Losing the nicer encoding is
            # acceptable; failing to start the tool is not.
            pass


def _widen_console_if_it_cannot_carry_our_output() -> None:
    """Run the guard once, at import, so no caller has to remember it.

    The guarantee used to live at fifteen remembered ``main()`` call sites, and the
    remembering is the defect. Two whole classes of caller never reach any of them:

    * a **library call** — ``from softae.analysis.eis.geometry import cell_config``
      from a script, a notebook or another tool. ``cell_config`` logs a ``σ`` on the
      shipped three-electrode config, so this crashed on a bare import.
    * **pytest** — no ``main()`` runs, and default capture replaces ``sys.stdout``
      with a UTF-8 file, so the crash is *swallowed* and the suite reports green.
      ``pytest -s`` reproduced it. A suite that cannot see the fault is worse than no
      test at all, because it is read as evidence.

    Editing the ~67 non-ASCII messages was the alternative and is the wrong shape: it
    treats an open-ended set of symptoms, and it provably misses cases — a log call
    whose text arrives from ``cal.describe()`` carries a ``≥`` that no literal scan can
    see. The characters are correct; the console is what is narrow.

    A library mutating the host's ``sys.stdout`` is a real smell, so the mutation is
    held to the narrowest form that still closes the gap:

    * it fires **only** when the stream's codec cannot carry the text — a UTF-8 host,
      a notebook, and pytest's own captured stdout are all left untouched;
    * when it does fire it only ever **widens** (utf-8 + ``errors="replace"``), and
      ``errors="replace"`` cannot itself raise;
    * it is suppressible with ``SOFTAE_NO_CONSOLE_UTF8=1`` for a host that means it.

    Deliberately *not* a structlog ``logger_factory``: that covers log calls only, and
    ``tools/commission.py`` reaches the same ``≥`` through a bare ``print``. It is also
    not reliably in force — ``tools/shadow_rehearse_report.py`` installs its own
    ``PrintLoggerFactory`` and tees to ``sys.stdout`` for the duration of a rehearsal,
    which is exactly the σ-heavy run that most needs the guarantee.
    """
    if _opted_out():
        return
    use_utf8_console(only_if_needed=True)


_widen_console_if_it_cannot_carry_our_output()

__all__ = ["OPT_OUT_ENV", "__version__", "stream_can_carry_any_text",
           "use_utf8_console"]
