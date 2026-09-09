"""Command-line tools for SoftAE (method lifecycle, commissioning, thickness, …)."""

from __future__ import annotations

import argparse
import logging
from typing import Any, Callable

import structlog

from softae import use_utf8_console

logger = structlog.get_logger(__name__)


# ``use_utf8_console`` is defined in ``softae/__init__.py`` and re-exported here.
# It lives at the package root because that is the only module every caller imports:
# the guarantee is now automatic (widening at import, only when the console's codec
# cannot carry the text), and the fifteen explicit calls below in the CLI ``main()``s
# import it from here as they always have. Re-exported rather than moved so those
# fifteen ``from softae.tools import use_utf8_console`` lines keep working unchanged.


def add_verbosity_flag(parser: argparse.ArgumentParser) -> None:
    """Add ``-v`` / ``--verbose`` — on the top-level parser *and* on every subcommand.

    ``default=argparse.SUPPRESS`` is load-bearing, not tidiness: a subparser
    copies its own defaults over the outer namespace after it parses, so a plain
    ``default=False`` here would silently discard
    ``python -m softae.tools.equilibration -v run ...`` — the exact spelling an
    operator reaches for when a run is misbehaving. Callers therefore read the
    flag as ``getattr(args, "verbose", False)``, since a namespace that nobody
    supplied it to does not carry the attribute at all.
    """
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS,
                        help="DEBUG logging, overriding [logging] level. Noisy: "
                             "the RH controller logs a duty cycle on every update.")


def configure_logging(verbose: bool = False) -> int:
    """Filter the log stream once, and return the level applied.

    Nothing else configures structlog on a headless path. The GUI does it at
    ``gui/app.py``; a headless entry point that skips it inherits structlog's
    default ``PrintLogger``, which emits **every** level — including
    ``rh_duty_sent``, logged on each RH control update. Over a six-hour
    unattended run that buries the run's own reporting in DEBUG. It is shared
    here rather than owned by one tool because the RH controller is shared: any
    CLI that brings the chamber to condition inherits the same flood.

    It does not touch that reporting. ``ProgressRenderer`` and its siblings write
    the milestones, hold verdicts, telemetry lines and the live status line
    straight to stdout, and the workflows' milestone log calls are
    ``info``/``warning``. Filtering at INFO leaves every one of them visible,
    which is the whole point: the operator loses the spam and keeps the run.

    Safe to call twice, and safe after the GUI has already configured: both
    ``structlog.configure`` and the explicit ``setLevel`` are last-writer-wins
    rather than additive. The ``setLevel`` is not redundant with
    ``basicConfig`` — ``basicConfig`` returns early once the root logger has a
    handler, so on any second call it would apply no level at all.
    """
    from softae.config import loader

    level = (logging.DEBUG if verbose
             else getattr(logging, loader.log_level(), logging.INFO))
    logging.basicConfig(level=level, format="%(message)s")
    logging.getLogger().setLevel(level)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    return level


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


__all__ = ["add_verbosity_flag", "configure_logging", "run_finalizer",
           "use_utf8_console"]
