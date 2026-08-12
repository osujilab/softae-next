"""Command-line tools for SoftAE (method lifecycle, commissioning, thickness, …)."""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """Make stdout/stderr survive non-ASCII output on a Windows console.

    Every CLI here prints characters outside cp1252 — ``⚠`` in warnings, ``σ``/``Ω``/
    ``δ`` throughout the EIS reporting, ``≲`` for an upper bound, ``→`` in the "run
    this next" hints. The rig runs on Windows, where ``sys.stdout`` defaults to the
    ANSI code page, and printing any of them raises ``UnicodeEncodeError``.

    That is not cosmetic. It surfaced as ``softae-commission derive`` **crashing with a
    traceback** part-way through a real derivation, after the artifacts had been read
    and before the calibration was written — the command appeared to fail at the
    analysis, when in fact it had failed at the ``print``. A tool that dies on its own
    warning text is worse than one that cannot warn.

    ``errors="replace"`` rather than a strict re-encode: a console that genuinely
    cannot render a glyph should show a substitute, never abort the command that was
    trying to tell the operator something.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # A redirected or wrapped stream may refuse. Losing the nicer encoding is
            # acceptable; failing to start the tool is not.
            pass


__all__ = ["use_utf8_console"]
