"""Application bootstrap — creates QApplication, InstrumentManager, and MainWindow."""

from __future__ import annotations

import asyncio
import logging
import sys

import qasync
import structlog
from PySide6.QtWidgets import QApplication

from softae.drivers.factory import create_manager
from softae.gui.main_window import MainWindow
from softae.config import loader as cfg
from softae.core.data_store import DataStore

log = logging.getLogger(__name__)


async def _connect_and_refresh(
    manager,
    window: MainWindow,
) -> None:
    """Connect instruments in background, then refresh the Init tab."""
    try:
        await manager.connect_all()
    except Exception as exc:
        log.error("Background instrument connection failed: %s", exc)
    # Trigger a near-immediate Init tab poll refresh when available.
    worker = getattr(window._tab_init, "_poll_worker", None)
    if worker is not None:
        worker.poke()
    else:
        window._tab_init._refresh_table()
    # Re-sync the head label now that every driver is up.  Connection happens
    # after the launch prompt, so this is the last point at which the displayed
    # state could drift from the driver's belief.
    manual_tab = getattr(window, "_tab_manual", None)
    if manual_tab is not None and hasattr(manual_tab, "refresh_head_label"):
        manual_tab.refresh_head_label()


def run_app(*, mock: bool | None = None) -> int:
    """Create and run the SoftAE desktop application.

    Parameters
    ----------
    mock : bool or None
        ``True`` forces all mock drivers.  ``None`` auto-detects
        real hardware, falling back to mocks when unavailable.

    Returns the Qt exit code (0 = normal).
    """
    # Configure log level from softae_config.toml [logging] level (default INFO)
    _level = getattr(logging, cfg.log_level(), logging.INFO)
    logging.basicConfig(level=_level, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(_level),
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SoftAE")
    app.setOrganizationName("OsujiLab")

    # --- Instrument manager (auto-detect real vs mock) ---
    manager = create_manager(mock=mock)

    # Launching the desktop GUI is a deliberate, human-driven act, so it arms
    # the hardware interlock for this process when real motion instruments are
    # present. Headless/agent paths never reach here and stay gated on the
    # SOFTAE_ALLOW_HARDWARE env var. See softae.core.hardware_safety.
    from softae.core.hardware_safety import arm_hardware, real_motion_instruments

    if real_motion_instruments(manager):
        arm_hardware(True)
        log.warning("Real motion hardware detected — GUI armed for this session.")

    # --- qasync event loop (combined Qt + asyncio) ---
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # --- Project data store ---
    data_store = DataStore(
        project_dir=cfg.data_project_dir(),
        db_filename=cfg.data_db_filename(),
    )

    # --- Main window ---
    window = MainWindow(manager, data_store=data_store)
    window.show()

    from softae.gui.widgets.head_check_dialog import ask_head_state, register_head_state
    from softae.gui.widgets.unclean_shutdown import check_unclean_shutdown

    # Did the last session end cleanly?  Every terminal path finalizes its run
    # row, so an unfinished one means the process died mid-experiment (crash,
    # power cut, or an OS-forced update restart).  This check never races the
    # shutdown — it runs afterwards — which makes it the most reliable layer of
    # the shutdown story.
    #
    # Runs BEFORE the head prompt on purpose: the head is a motor flipper that
    # holds position without power, so an unclean stop can leave it lowered over
    # an electrode.  Asking "is the head up or down?" first would make the
    # operator answer from memory before being told to go and look.
    check_unclean_shutdown(window, manager, data_store)

    # Register the physical dispenser-head position at launch.  The head has no
    # position feedback and may have been flipped manually while the app was
    # closed, so the operator confirms it up front; the answer is recorded into
    # the syringe driver's belief (no motion — there is no imminent stage
    # travel at launch).  Re-verified at each HT-experiment / campaign start.
    _head_state = ask_head_state(window, context="starting the session")
    register_head_state(manager, _head_state)
    _manual_tab = getattr(window, "_tab_manual", None)
    if _manual_tab is not None and hasattr(_manual_tab, "refresh_head_label"):
        _manual_tab.refresh_head_label()

    # Schedule instrument connection in background (non-blocking)
    asyncio.ensure_future(_connect_and_refresh(manager, window))

    with loop:
        rc = loop.run_forever()
        # Cleanup: disconnect instruments before exit
        loop.run_until_complete(manager.disconnect_all())
        data_store.close()

    return rc
