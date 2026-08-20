"""Command-line tools for SoftAE (method lifecycle, commissioning, thickness, …)."""

from __future__ import annotations

import sys
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


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


def run_finalizer(store: Any, run_id: str) -> Callable[[str], None]:
    """Return a one-shot, never-raising closer for *run_id*.

    Every ``DataStore.start_run`` needs a matching ``finish_run`` on **every**
    exit path, and neither these CLIs nor ``WorkflowExecutor`` used to provide
    one. A row left with ``finished_at`` NULL is byte-for-byte what a *crashed*
    run looks like (``DataStore.unfinished_runs``), and the consequence is not
    cosmetic: ``gui/widgets/unclean_shutdown.py`` reads those rows at the next
    launch, offers the operator a recovery park of the rig over a run that
    finished perfectly, and **relabels the row** ``interrupted`` — an ``UPDATE``
    with no unset. A tool that does not close its own row therefore rewrites its
    own history the next time the GUI starts.

    Idempotent, so the caller can put ``finalize("error")`` in a ``finally`` as
    the catch-all for paths no ``except`` names without it overwriting the status
    a handler already recorded — and, on the success path, so the explicit
    ``finalize("done")`` wins. Never raises, for the same reason
    ``autonomous_wiring._finalize_run`` does not: failing to record how a run
    ended must not turn a successful run into a failed one.

    Statuses come from ``finish_run``'s documented vocabulary only —
    ``done`` / ``partial`` / ``aborted`` / ``error`` / ``interrupted``.
    """
    finalized = False

    def _finalize(status: str) -> None:
        nonlocal finalized
        if finalized:
            return
        finalized = True
        try:
            store.finish_run(run_id, status)
        except Exception:
            logger.warning("finish_run_failed", run_id=run_id, status=status,
                           exc_info=True)

    return _finalize


__all__ = ["run_finalizer", "use_utf8_console"]
