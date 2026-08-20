"""Main application window with tabbed interface and emergency stop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from contextlib import contextmanager
from typing import Callable

import structlog

from softae.config import loader
from softae.gui.campaign_stream import CampaignStreamView
from softae.gui.launch_mode import OWNER_MODE, LaunchMode
from softae.core.formulation import ChemicalCatalog, SolutionCatalog
from softae.gui.tabs.tab_analysis import AnalysisTab
from softae.gui.tabs.tab_arrhenius import ArrheniusTab
from softae.gui.tabs.tab_autonomous import AutonomousTab
from softae.gui.tabs.tab_bo_live import LiveBOCampaignTab
from softae.gui.tabs.tab_bo_simulator import BOSimulatorTab
from softae.gui.tabs.tab_experiment import ExperimentBuilderTab
from softae.gui.tabs.tab_init import InitCalibrationTab
from softae.gui.tabs.tab_liquid_model import LiquidModelTab
from softae.gui.tabs.tab_manual import ManualControlTab
from softae.gui.tabs.tab_monitor import MonitoringTab
from softae.gui.tabs.tab_process_studio import ProcessStudioTab
from softae.gui.widgets.camera_worker import CameraWorker
from softae.gui.widgets.catalog_browser import CatalogBrowser
from softae.gui.widgets.catalog_manager import CatalogManager
from softae.gui.widgets.conditions_source import ConditionsFileSource
from softae.gui.widgets.deposition_panel import DepositionPanel
from softae.gui.widgets.emergency_stop import EmergencyStopButton
from softae.gui.widgets.instrument_poller import InstrumentPoller
from softae.gui.widgets.monitor_sidebar import MonitorSidebar
from softae.gui.widgets.rig_owner import foreign_rig_lock, owner_status_line
from softae.gui.widgets.safe_exit import SafeExitButton
from softae.gui.widgets.status_indicator import InstrumentStatusBar
from softae.gui.widgets.webcam_worker import WebcamWorker

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: How often the idle purge timer *asks* whether a purge is owed. Much shorter
#: than the purge interval itself: a tick that finds the rig busy must not push
#: the purge out by a whole period, and the runner is cheap when nothing is due.
_PURGE_POLL_MS = 30_000

#: How often the window re-reads who owns the rig, and — when attached — the
#: campaign's event stream.
#:
#: The number is a real cost, not a preference. ``read_events`` deliberately
#: keeps no byte offset (an offset is meaningless across a rotation) and does not
#: hold the handle between polls (a held handle makes ``os.replace`` fail on
#: Windows and silently turns the 32 MB cap off), so **every poll reads the whole
#: live stream** — ~2.5 MB after a week, 32 MB at the cap.
#:
#: 5 s is the floor below which the extra reads buy nothing: the heartbeat is
#: 30 s and the conditions sidecar republishes at 5 s, so a 1 s poll re-reads
#: megabytes to redisplay the same strings. It is also six times *more*
#: responsive than what it replaces — the park indicator's only periodic caller
#: was the 30 s purge tick, which an attached window does not have.
_CAMPAIGN_POLL_MS = 5_000


def _park_nothing() -> None:
    """The exit park of a window that opened no instrument session.

    A process may command only the sessions it opened, so an attached window has
    nothing to park — and the ruling this implements is that the park path must
    then be *absent* rather than a conditional that evaluates false. This is what
    "absent" is made of: :class:`MainWindow` binds its real exit park onto the
    instance only in owner mode, and an attached window falls back to the
    class-level default below, which is this. ``closeEvent`` therefore reads no
    lock, tests no flag, and cannot be one stale boolean away from parking a rig
    another process is driving — nor from skipping a park it owed.
    """


class MainWindow(QMainWindow):
    """Top-level window containing the full tabbed application.

    Parameters
    ----------
    manager : InstrumentManager
        Pre-configured instrument registry (mock or real).
    data_store : DataStore or None
        Project-scoped SQLite backend for experiment data.
    launch_mode : LaunchMode or None
        Owner or attached, decided **once** before the window exists (see
        :mod:`softae.gui.launch_mode`). ``None`` means owner mode — the
        historical behaviour, and what a bare ``MainWindow(manager)`` in a test
        or a script gets. It is a constructor argument rather than an attribute
        set afterwards because the window's construction *differs* by mode: an
        attached window is built without an exit park and without the purge
        timer, and neither can be un-built by a later assignment.
    """

    #: The exit park, chosen once at construction. Owner mode binds the real one
    #: onto the instance; an attached window is constructed without it and
    #: inherits this, which does nothing. See :func:`_park_nothing`.
    _exit_park: Callable[[], None] = staticmethod(_park_nothing)

    def __init__(
        self,
        manager: InstrumentManager,
        *,
        data_store: DataStore | None = None,
        launch_mode: LaunchMode | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._launch_mode = launch_mode if launch_mode is not None else OWNER_MODE

        # The park path, installed here or not at all. Owner mode parks what it
        # opened; an attached window opened nothing, so there is nothing to park
        # and no branch on the way out to get wrong.
        if self._launch_mode.owner:
            self._exit_park = self._safe_park_on_exit

        # Stock interlock — attached before any tab can command a pump, via the
        # shared core helper so the GUI and the headless CLI enforce identically.
        from softae.core.reservoir import attach_reservoir_ledger

        self._reservoir_ledger = attach_reservoir_ledger(manager, data_store)

        # Waste capacity + spare plates (P5.4) — the limits that cap unattended
        # time from the other direction. Undeclared means unknown, not exhausted.
        self._waste_ledger = None
        self._board_inventory = None
        if data_store is not None:
            from softae.core.consumables import attach_consumables

            self._waste_ledger, self._board_inventory = attach_consumables(data_store)

        # Anti-clog purge (P8). Attached at the same choke point as the stock
        # ledger so every dispense — HT, campaign, manual — resets that line's
        # timer; without this the harness would purge at the full idle rate
        # during an active run, for lines the run had just used itself.
        from softae.core.hardware_safety import attach_head_guard
        from softae.core.purge import attach_purge_scheduler
        from softae.core.purge_runner import IdleRestState, PurgeRunner
        from softae.core.rig_activity import RigActivity

        # Let the stage see head state so it refuses to translate while lowered.
        # Load-bearing now that head-down is the RESTING state rather than a
        # brief interval inside a known sequence.
        attach_head_guard(manager)

        self._purge_scheduler = attach_purge_scheduler(manager, data_store=data_store)
        #: Who currently owns the hardware. The purge asks and defers; it never
        #: waits, because a background timer that can block is one that can
        #: deadlock a run.
        self._rig_activity = RigActivity()
        #: Idle-rest *intent*. The authority on whether a purge may happen is
        #: classify_pose(), read from the hardware — the head is down at idle
        #: rest and down mid-cast alike, so a flag alone cannot be trusted.
        self._idle_rest = IdleRestState()
        #: Held until an operator clears it; see :meth:`notify_parked`.
        self._park_latch: str | None = None
        self._purge_runner = None
        self._purge_timer = None
        if self._purge_scheduler is not None:
            self._purge_runner = PurgeRunner(
                manager, self._purge_scheduler,
                waste_ledger=self._waste_ledger,
                park_reason=self._park_reason,
                idle_rest=self._idle_rest,
                activity=self._rig_activity,
                data_store=data_store,
            )
            # In-run purging is entirely executor-driven (see PURGE_WINDOW_TAG):
            # any step that declares itself co-runnable gets a purge loop beside
            # it, joined before the run moves on. No driver knows purging exists.
            #
            # The background timer cannot serve the in-run case at all — a run
            # holds the rig claim for its whole duration, so the timer correctly
            # defers.
            #
            # Published on the syringe so the campaign path finds *this* runner
            # rather than constructing a second one against the same scheduler.
            try:
                manager.get("syringe").purge_runner = self._purge_runner
            except Exception:
                pass
            # The idle purge is the only thing on this rig that actuates with
            # nobody asking — so an attached window has no timer at all, for the
            # same reason it has no exit park: it would be commanding sessions
            # another process owns. Not merely stopped; never created, so there
            # is nothing here a later edit can .start().
            if self._launch_mode.owner:
                self._purge_timer = QTimer(self)
                # Polls far more often than the purge interval: the timer only
                # asks "is one owed and may it happen now", and a tick that
                # finds the rig busy must not push the purge out by a whole
                # period.
                self._purge_timer.setInterval(_PURGE_POLL_MS)
                self._purge_timer.timeout.connect(self._on_purge_tick)
                self._purge_timer.start()

        # Cached catalogs refreshed by _on_catalogs_changed (no live list widget yet).
        self._chem_catalog = None
        self._sol_catalog = None

        self.setWindowTitle("SoftAE — Soft-matter Autonomous Experimentation")
        self.setMinimumSize(1200, 800)

        # What an attached window knows about the run. `None` in owner mode, and
        # also when the holder is not a campaign — something owns the rig but
        # publishes no stream, so there is nothing to follow and the window says
        # only that it is occupied.
        self._attach_stream: CampaignStreamView | None = None
        if self._launch_mode.attached and self._launch_mode.run_dir:
            self._attach_stream = CampaignStreamView(
                self._launch_mode.run_dir, campaign=self._launch_mode.campaign
            )

        # Shared instrument poller — single background thread that replaces the
        # three independent polling workers previously running in tab_monitor,
        # monitor_sidebar, and status_indicator. Its *source* is chosen with the
        # launch mode; the three consumer widgets do not know there is a choice.
        self._poller = InstrumentPoller(
            self._manager, source=self._instrument_source(), parent=self
        )

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()

        # The window's other periodic tick, and in attach mode its only one. See
        # `_CAMPAIGN_POLL_MS` for the cadence and `_on_campaign_tick` for the
        # three jobs it does.
        self._campaign_timer = QTimer(self)
        self._campaign_timer.setInterval(_CAMPAIGN_POLL_MS)
        self._campaign_timer.timeout.connect(self._on_campaign_tick)
        self._campaign_timer.start()
        self._on_campaign_tick()   # paint once rather than start blank

        self._poller.start()

    def _instrument_source(self):
        """Where the poller's readings come from — decided once, with the mode.

        Owner mode reads the instruments it opened. Attached mode reads the
        campaign's ``conditions.json`` and **no instrument at all**: a read is a
        serial transaction on a bus the owning process is using, and the four
        ``manager.get(...)`` calls would in any case fail one by one into their
        own exception handlers, leaving the operator a rig that renders as
        disconnected while it runs an eight-hour anneal.

        A run directory of ``None`` — occupied by something that is not a
        campaign — is the honest degenerate case: the instruments are reported
        as the holder's and every value is unknown.
        """
        if self._launch_mode.owner:
            return None
        return ConditionsFileSource(self._launch_mode.run_dir, manager=self._manager)

    @property
    def launch_mode(self) -> LaunchMode:
        """Owner or attached — read-only, because it is decided before this exists.

        Deliberately has no setter. Construction *branches* on the mode (no exit
        park and no purge timer in attached mode), so a window that could be
        re-moded afterwards would carry a mode its own wiring did not match —
        which is the drift the launch-time decision exists to rule out. The one
        way out of attach mode is the operator act that also acquires the
        sessions: Init tab → Connect All.
        """
        return self._launch_mode

    # --- UI construction ------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        # Override the aggregated minimumSizeHint that Qt calculates from all
        # tab children.  Without this the body_splitter sash is stuck because
        # the QTabWidget's minimum == (almost) the full window width.
        self._tabs.setMinimumWidth(600)

        # --- Full tab stack ---
        self._tab_init = InitCalibrationTab(self._manager, data_store=self._data_store)
        self._tab_liquid_model = LiquidModelTab(self._manager)
        # The launch mode is passed, not looked up: Manual Control's controls are
        # the ones that would command sessions this window may not have opened,
        # and the tab branches on the answer at construction (it starts no
        # polling thread when attached).
        self._tab_manual = ManualControlTab(
            self._manager, data_store=self._data_store, launch_mode=self._launch_mode
        )
        self._tab_monitor = MonitoringTab(self._manager, poller=self._poller)
        self._tab_experiment = ExperimentBuilderTab(self._manager, data_store=self._data_store)
        self._tab_arrhenius = ArrheniusTab(self._manager, data_store=self._data_store)
        self._tab_autonomous = AutonomousTab(self._manager)

        self._tabs.addTab(self._tab_init, "1. Init && Calibration")
        self._tabs.addTab(self._tab_liquid_model, "2. Liquid Model")
        self._tabs.addTab(self._tab_manual, "3. Manual Control")
        self._tabs.addTab(self._tab_monitor, "4. Monitoring")
        self._tabs.addTab(self._tab_experiment, "5. HT Experiment")
        self._tabs.addTab(self._tab_arrhenius, "6. Arrhenius Sweep")
        self._tab_analysis = AnalysisTab(self._manager, data_store=self._data_store)
        # Offline simulation tab — no InstrumentManager needed.
        self._tab_bo_simulator = BOSimulatorTab(data_store=self._data_store)
        # Live, hardware-in-the-loop BO campaign (mock manager by default).
        self._tab_bo_live = LiveBOCampaignTab(self._manager, data_store=self._data_store)

        self._tabs.addTab(self._tab_autonomous, "7. Autonomous")
        self._tabs.addTab(self._tab_analysis, "8. Analysis")
        # (Former "9. Process Configuration" retired — the standalone SandboxTab was
        # redundant with Process Studio's embedded Builder, which is a superset.)
        self._tabs.addTab(self._tab_bo_simulator, "9. BO Simulator")
        self._tabs.addTab(self._tab_bo_live, "10. Live BO Campaign")

        # Shared catalogs, loaded once from data_root (single source of truth).
        self._chem_catalog, self._sol_catalog = self._load_catalogs_from_root()

        self._catalog_browser = CatalogBrowser(
            chem_catalog=self._chem_catalog, sol_catalog=self._sol_catalog
        )
        self._deposition_panel = DepositionPanel(self._chem_catalog, self._sol_catalog)

        self._tabs.addTab(self._catalog_browser, "11. Catalogs")
        self._tabs.addTab(self._deposition_panel, "12. Deposition")

        # Method/recipe maturity library + builder (reads tasks.toml / recipes.toml).
        self._tab_process_studio = ProcessStudioTab(
            data_store=self._data_store, manager=self._manager
        )
        self._tabs.addTab(self._tab_process_studio, "13. Process Studio")

        # Wire catalog-editor reach + live refresh (mirrors deposition_app, in-window).
        self._catalog_browser.edit_requested.connect(self._open_catalog_manager)
        self._deposition_panel.manage_catalogs_requested.connect(self._open_catalog_manager)
        self._deposition_panel.reload_catalogs_requested.connect(self._on_catalogs_changed)

        # Camera worker — single persistent thread shared between tabs.
        # All ThorLabs SDK calls happen on this thread (SDK thread affinity).
        self._cam_worker = CameraWorker(self._manager, parent=self)
        self._tab_manual.set_camera_worker(self._cam_worker)

        # Webcam worker — USB webcam feed wired to the sidebar only.
        # Only started when enabled in [webcam] config section (default: true)
        # because OpenCV CAP_DSHOW can block 500 ms–3 s on Windows if no
        # USB webcam is present.
        self._webcam_worker = WebcamWorker(parent=self)
        try:
            from softae.config.loader import load as _cfg_load
            _webcam_enabled = _cfg_load().get("webcam", {}).get("enabled", True)
        except Exception:
            _webcam_enabled = True
        if _webcam_enabled:
            self._webcam_worker.start()

        # Monitoring sidebar — compact right-hand panel (~15 % width)
        self._sidebar = MonitorSidebar(self._manager, poller=self._poller, parent=self)
        self._sidebar.set_camera_worker(self._cam_worker)
        self._sidebar.set_webcam_worker(self._webcam_worker)

        # Splitter: tabs (~85 %) | sidebar (~15 %)
        # Stored as an instance attribute so Python's GC never collects the
        # wrapper while the window is alive.
        self._body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._body_splitter.addWidget(self._tabs)
        self._body_splitter.addWidget(self._sidebar)
        self._body_splitter.setChildrenCollapsible(False)
        self._body_splitter.setHandleWidth(12)
        # setSizes is deferred to showEvent so it is not overridden by Qt's
        # first layout pass (stretch factors would fight an early setSizes).
        body_splitter = self._body_splitter  # alias for layout.addWidget below

        # Bridge: Arrhenius sweep status → Monitor tab + sidebar
        self._tab_arrhenius.sweep_status_changed.connect(
            self._tab_monitor.update_sweep_status
        )
        self._tab_arrhenius.sweep_status_changed.connect(
            self._sidebar.update_arrhenius_status
        )

        # Bridge: PCB selection → Arrhenius tab channel limit
        self._tab_init.pcb_channel_count_changed.connect(
            self._tab_arrhenius.set_pcb_channel_count
        )

        # Bridges: HT Experiment + Autonomous workflow status → sidebar.
        #
        # These two slots are the window's campaign-status line, and in attach
        # mode the campaign is in another process — so the *source* is swapped
        # and the slots are driven from the event stream in `_on_campaign_tick`
        # instead. Same two slots, same kind of string; the sidebar never learns
        # that attach exists. Nothing in-process can drive them in that mode
        # anyway: every workflow goes through `WorkflowExecutor`, which refuses
        # while another process holds the rig lock.
        if self._attach_stream is None:
            self._tab_experiment.workflow_status_changed.connect(
                self._sidebar.update_ht_status
            )
            self._tab_autonomous.workflow_status_changed.connect(
                self._sidebar.update_auto_status
            )

        # Bridge: catalog edits (via the shared FormulationPanel opener) → refresh hook
        self._tab_experiment.catalogs_changed.connect(self._on_catalogs_changed)

        layout.addWidget(body_splitter)
        self.setCentralWidget(central)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        for i in range(self._tabs.count()):
            act = QAction(self._tabs.tabText(i), self)
            tab_idx = i
            act.triggered.connect(
                lambda checked=False, idx=tab_idx: self._tabs.setCurrentIndex(idx)
            )
            view_menu.addAction(act)

        instr_menu = menu.addMenu("&Instruments")
        if self._reservoir_ledger is not None:
            self._reservoir_action = QAction("Syringe Stock…", self)
            self._reservoir_action.setStatusTip(
                "Declare loaded syringe volumes for the stock interlock"
            )
            self._reservoir_action.triggered.connect(self._open_reservoir_dialog)
            instr_menu.addAction(self._reservoir_action)

        if self._waste_ledger is not None:
            self._bench_action = QAction("Bench Consumables…", self)
            self._bench_action.setStatusTip(
                "Waste level, spare electrode plates, and anti-clog purge settings"
            )
            self._bench_action.triggered.connect(self._open_bench_dialog)
            instr_menu.addAction(self._bench_action)

        self._board_swap_action = QAction("Log Board Swap…", self)
        self._board_swap_action.setStatusTip(
            "Record a fresh electrode board and reset the occupancy map"
        )
        self._board_swap_action.triggered.connect(self._log_board_swap)
        instr_menu.addAction(self._board_swap_action)

        catalog_menu = menu.addMenu("&Catalogs")
        self._edit_catalogs_action = QAction("Edit Catalogs…", self)
        self._edit_catalogs_action.setStatusTip("Open the chemical/solution catalog editor")
        self._edit_catalogs_action.triggered.connect(self._open_catalog_manager)
        catalog_menu.addAction(self._edit_catalogs_action)

    def _log_board_swap(self) -> None:
        """Operator-initiated board replacement — advance the board, reset the map."""
        from softae.gui.widgets.occupancy_guard import prompt_log_board_swap

        if self._data_store is None:
            QMessageBox.information(
                self, "Log Board Swap",
                "Board occupancy needs a project data store; none is open.")
            return

        new_id = prompt_log_board_swap(self, self._data_store)
        if new_id is None:
            return
        self._tab_init.refresh_occupancy(new_id)
        self.statusBar().showMessage(
            f"Board swap logged — now on board {new_id}; all wells available.", 8000)

    def _open_bench_dialog(self) -> None:
        """Declare waste level and spare-plate count."""
        from softae.gui.widgets.bench_dialog import BenchDialog

        BenchDialog(self._waste_ledger, self._board_inventory, parent=self,
                    data_store=self._data_store).exec()

    def _open_reservoir_dialog(self) -> None:
        """Show remaining stock and let the operator declare refills."""
        from softae.gui.widgets.reservoir_dialog import ReservoirDialog

        ReservoirDialog(
            self._reservoir_ledger, parent=self,
            data_store=self._data_store,
            sol_catalog=self._sol_catalog,
        ).exec()
        # A changed loadout changes which lines the purge treats as particulate.
        self._resync_particulate_pumps()

    def _resync_particulate_pumps(self) -> None:
        """Re-derive particulate lines after the operator edits the loadout.

        Without this the new declaration would not take effect until the next
        GUI start — and a purge harness that quietly ignores what the operator
        just told it is worse than one that was never configured.
        """
        scheduler = getattr(self, "_purge_scheduler", None)
        if scheduler is None:
            return
        try:
            from softae.core.stock_assignment import resolve_particulate_pumps

            derived = resolve_particulate_pumps(
                scheduler.settings, data_store=self._data_store,
                sol_catalog=self._sol_catalog,
            )
            if tuple(derived) != tuple(scheduler.settings.particulate_pumps):
                scheduler.settings.particulate_pumps = tuple(derived)
                logger.info("particulate_pumps_resynced", pumps=list(derived))
        except Exception:
            logger.warning("particulate_resync_failed", exc_info=True)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        # Emergency stop — always visible, prominent. The launch mode goes in
        # because the button's *guarantee* depends on it: an attached window
        # holds no sessions, so a press cannot park and must instead climb the
        # escalation ladder to the campaign that can. Which rungs that ladder can
        # reach is decided in the constructor and written into the label, so the
        # operator reads it before they press rather than discovering it after.
        self._estop = EmergencyStopButton(
            self._manager, launch_mode=self._launch_mode)
        self._estop.parked.connect(self.notify_parked)
        toolbar.addWidget(self._estop)

        # The way out of a park. Hidden until there is one to clear — a control
        # that is always visible is one an operator learns to press.
        #
        # It ships in the same change that routes more surfaces through the latch,
        # deliberately: ``clear_park`` previously had no production caller at all,
        # so the first automatic park would have stranded the rig with nothing to
        # clear it and no indication that anything was latched.
        self._clear_park_btn = QPushButton("⚠  PARKED — CLEAR")
        self._clear_park_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #6a1b9a;"
            "  color: white;"
            "  font-weight: bold;"
            "  padding: 8px 16px;"
            "  border-radius: 6px;"
            "}"
            "QPushButton:hover { background-color: #4a148c; }"
        )
        self._clear_park_btn.setMinimumHeight(44)
        self._clear_park_btn.clicked.connect(self._on_clear_park)
        # Visibility is driven through the QWidgetAction the toolbar wraps it in,
        # not through the widget: the toolbar re-shows its own child widgets on
        # layout, so a bare setVisible(False) does not survive.
        self._clear_park_action = toolbar.addWidget(self._clear_park_btn)
        self._clear_park_action.setVisible(False)

        # Catalog editor — same QAction as the Catalogs menu entry.
        toolbar.addAction(self._edit_catalogs_action)

        # Push Safe Exit to the far right. The two stop controls then bracket the
        # toolbar rather than sitting side by side, which is the point: pressing the
        # wrong one of two adjacent red-ish buttons is exactly the mistake to design
        # out, and an emergency stop is not something to hit by accident on the way
        # to closing the application.
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Safe Exit is a park path — the same safe_park() call as the exit park,
        # on the same manager, at the same moment — so it is absent in attached
        # mode for the same reason and by the same decision. Left present it
        # would either warn on *every* close (a park that commanded nothing is
        # correctly severe, and in attach mode that is not a fault but the
        # normal state) or sit disabled with a live park one setEnabled away.
        # Nothing is lost: closing is still X / Alt-F4 / File → Quit, the head
        # question it exists to ask is meaningless about a head this process
        # does not drive, and the stop control in attach mode is the E-Stop.
        if self._launch_mode.owner:
            self._safe_exit = SafeExitButton(self._manager)
            self._safe_exit.parked.connect(self.notify_parked)
            self._safe_exit.exit_requested.connect(self._on_safe_exit)
            toolbar.addWidget(self._safe_exit)
        else:
            toolbar.addWidget(self._build_attach_notice())

    def _build_attach_notice(self) -> QWidget:
        """The statement that stands where Safe Exit does in owner mode.

        A control that vanishes without saying why reads as a bug. This says the
        mode, names the holder, and carries the launch decision's own sentence as
        its tooltip — so the missing park control and the reason for it arrive
        together.

        The headline itself comes from :func:`~softae.gui.widgets.rig_owner.attached_owner_line`,
        which the Manual Control tab's refusal banner also starts from: two
        surfaces describing one fact must not describe it in two phrasings.
        """
        from PySide6.QtWidgets import QLabel

        from softae.gui.widgets.rig_owner import attached_owner_line

        mode = self._launch_mode
        label = QLabel(attached_owner_line(mode.campaign))
        label.setToolTip(
            mode.reason
            + "\n\nNothing here can park the rig, so Safe Exit is not offered. "
            "Close the window normally; use Emergency Stop to request a stop."
        )
        label.setStyleSheet("QLabel { color: #6a1b9a; font-weight: bold; padding: 0 12px; }")
        return label

    def _on_safe_exit(self) -> None:
        """Close the window after the Safe Exit park has finished.

        ``_safe_park_on_exit`` still runs from ``closeEvent`` — it is best-effort and
        idempotent, and its head retraction is skipped when the head is already up.
        The one case that matters is an operator who chose to *leave the head down*:
        that choice is latched here so the close path cannot quietly undo it.
        """
        self._skip_exit_retract = not self._safe_exit_retracted()
        self.close()

    def _safe_exit_retracted(self) -> bool:
        """Whether the head ended up raised, however it got there."""
        from softae.gui.widgets.safe_exit import head_is_down

        return not head_is_down(self._manager)

    def _build_statusbar(self) -> None:
        self._status_bar = InstrumentStatusBar(self._manager, poller=self._poller)
        self.setStatusBar(self._status_bar)

    # --- Catalogs --------------------------------------------------------------

    def _load_catalogs_from_root(self) -> tuple[ChemicalCatalog, SolutionCatalog]:
        """Read the shared catalogs from ``data_root()``; degrade to empty on failure."""
        try:
            root = loader.data_root()
            return (
                ChemicalCatalog.load_csv(root / "chemicals.csv"),
                SolutionCatalog.load_csv(root / "solutions.csv"),
            )
        except Exception:
            return ChemicalCatalog(), SolutionCatalog()

    def _open_catalog_manager(self) -> None:
        """Menu/toolbar/browser/deposition entry — open the slim CatalogManager.

        The full ``FormulationPanel`` (elution/pump/apply-to-channels) stays
        reachable via the tab-5 "Formulation Manager…" button.
        """
        dlg = CatalogManager(parent=self)  # auto-loads from data_root()
        dlg.catalogs_changed.connect(self._on_catalogs_changed)
        dlg.exec()
        self._on_catalogs_changed()  # final safety refresh after close

    def _on_catalogs_changed(self) -> None:
        """Re-read ``data_root()`` once and live-refresh the browser + deposition panel.

        Guarded so it never raises even if fired before the tabs exist (e.g. an
        early tab-5 signal).
        """
        try:
            root = loader.data_root()
            self._chem_catalog = ChemicalCatalog.load_csv(root / "chemicals.csv")
            self._sol_catalog = SolutionCatalog.load_csv(root / "solutions.csv")
            if getattr(self, "_catalog_browser", None) is not None:
                self._catalog_browser.reload(self._chem_catalog, self._sol_catalog)
            if getattr(self, "_deposition_panel", None) is not None:
                self._deposition_panel.set_catalogs(self._chem_catalog, self._sol_catalog)
            self.statusBar().showMessage(
                f"Catalogs updated: {len(self._chem_catalog)} chemicals, "
                f"{len(self._sol_catalog)} solutions",
                5000,
            )
        except Exception:
            self.statusBar().showMessage("Catalogs updated", 5000)

    def showEvent(self, event) -> None:
        """Apply initial splitter proportions after the window is first shown."""
        super().showEvent(event)
        if not getattr(self, "_splitter_sized", False):
            self._splitter_sized = True
            # Defer to the next event-loop tick so Qt's layout pass has
            # resolved final geometry before we call setSizes.
            QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_initial_splitter_sizes(self) -> None:
        total = self._body_splitter.width()
        if total > 0:
            sidebar_w = max(180, min(320, round(total * 0.15)))
            self._body_splitter.setSizes([total - sidebar_w, sidebar_w])

    # ── Park state ───────────────────────────────────────────────────────────

    def notify_parked(self, reason: str) -> None:
        """Latch that the rig has been parked, and why.

        Held until an operator explicitly clears it. A park is not something the
        software gets to decide is over — it means a human may have reached in.
        """
        self._park_latch = str(reason) or "parked"
        # A parked rig is not at idle rest, whatever it was doing before.
        self._idle_rest.mark_left()
        logger.info("gui_park_latched", reason=self._park_latch)
        self._refresh_park_indicator()

    def clear_park(self) -> None:
        """Operator has resolved the fault; unattended actuation may resume."""
        if getattr(self, "_park_latch", None):
            logger.info("gui_park_cleared", reason=self._park_latch)
        self._park_latch = None
        self._refresh_park_indicator()

    def _refresh_park_indicator(self) -> None:
        """Show the clear-park control exactly while a park is outstanding.

        Reads :meth:`_park_reason`, not the latch, so a campaign that parked
        itself overnight surfaces the same control — that is the case with nobody
        watching, and it is the one where an invisible latch is worst.

        Guarded because the latch exists before the toolbar does.
        """
        button = getattr(self, "_clear_park_btn", None)
        action = getattr(self, "_clear_park_action", None)
        if button is None or action is None:
            return
        reason = self._park_reason()
        action.setVisible(bool(reason))
        if reason:
            button.setToolTip(
                f"The rig is parked: {reason}\n\n"
                "Unattended actuation (idle rest, anti-clog purging) stays "
                "refused until this is cleared. Manual control is never blocked."
            )

    def _on_clear_park(self) -> None:
        """Confirm, then clear. **Only the window's own latch** is clearable here.

        A campaign's park belongs to the campaign; clearing it from the toolbar
        would tell a running loop that a fault it detected is resolved, on no
        evidence. If the reason survives this, the button stays visible — which is
        the honest outcome, not a bug.
        """
        reason = self._park_reason()
        if not reason:
            self._refresh_park_indicator()
            return
        answer = QMessageBox.question(
            self, "Clear park",
            f"The rig parked: {reason}\n\n"
            "Clear it only once you have checked the hardware — in particular "
            "where the dispenser head actually is, which nothing on this rig can "
            "sense.\n\nClear the park?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.clear_park()
        still = self._park_reason()
        self.statusBar().showMessage(
            f"Park still outstanding: {still}" if still else "Park cleared", 8000
        )

    def _park_reason(self) -> str | None:
        """Aggregate park state — the window latch, or a running campaign's.

        A campaign that parked itself overnight is the case this exists for, and
        *where* that campaign is decides where the fact comes from: in attach
        mode it is another process's park, which arrives as a ``park`` record on
        the event stream; in owner mode it is the in-process tab's.
        """
        latched = getattr(self, "_park_latch", None)
        if latched:
            return latched
        stream = getattr(self, "_attach_stream", None)
        if stream is not None:
            return stream.park_reason
        tab = getattr(self, "_tab_bo_live", None)
        try:
            reason = getattr(tab, "park_reason", None)
            return reason() if callable(reason) else reason
        except Exception:
            return None

    # ── Who owns the rig, and what it is doing (attach render) ───────────────

    def _on_campaign_tick(self) -> None:
        """The window's campaign-facing tick. Never raises.

        Three jobs, and the third is why it exists in *both* modes:

        1. poll the campaign's event stream (attached only) and put its two
           strings into the sidebar's existing status slots;
        2. re-read who holds the rig and render the owner line — a foreign lock
           can appear at any time, including under an owner-mode window;
        3. refresh the park indicator. ``_on_purge_tick`` was its **only**
           periodic caller and the purge timer is not created in attach mode, so
           without this the "PARKED — CLEAR" control would never appear for
           precisely the unattended, nobody-watching case it exists for.

        Swallowed like the purge tick, and for the same reason: a background
        timer that can kill the GUI is worse than a stale label.
        """
        try:
            stream = self._attach_stream
            if stream is not None:
                stream.poll()
                self._sidebar.update_ht_status(stream.ht_status())
                self._sidebar.update_auto_status(stream.auto_status())
            self._refresh_rig_owner()
            self._refresh_park_indicator()
        except Exception:
            logger.warning("campaign_tick_failed", exc_info=True)

    def _refresh_rig_owner(self) -> None:
        """Render the sidebar's owner line from the lock plus the stream.

        The lock is re-read every tick because this is a *view*: it is what makes
        the line appear when a campaign starts underneath an idle window and
        disappear when one ends. It decides nothing — the launch decision is
        made once and is never re-derived from this read.
        """
        sidebar = getattr(self, "_sidebar", None)
        if sidebar is None:
            return
        stream = self._attach_stream
        sidebar.update_rig_owner(owner_status_line(
            foreign_rig_lock(),
            phase=None if stream is None else stream.phase,
            phase_age_s=None if stream is None else stream.phase_age_s,
        ))

    # ── Anti-clog purge (P8) ─────────────────────────────────────────────────

    def enter_idle_rest(self) -> bool:
        """Rest the rig at the flush station, head down — tip protected.

        Works from any pose (it retracts before travelling), so it is safe to
        call at the end of *any* run regardless of where that run left things.
        """
        from softae.core.purge_runner import enter_idle_rest

        result = enter_idle_rest(
            self._manager, park_reason=self._park_reason, state=self._idle_rest
        )
        if not result.entered:
            logger.info("idle_rest_not_entered", reason=result.reason)
        return result.entered

    def leave_idle_rest(self) -> bool:
        """Retract the head before a run moves anything."""
        from softae.core.purge_runner import leave_idle_rest

        return leave_idle_rest(self._manager, state=self._idle_rest)

    @contextmanager
    def rig_run(self, owner: str, *, rest_after: bool = True):
        """Own the hardware for the duration of a run, then return it to rest.

        **This is the convention every run path should use.** It replaces the
        previous situation where each workflow simply left the head retracted
        wherever it finished — fine when head-up was the resting state, wrong now
        that the tip is meant to sit in flush between runs.

        Ownership is claimed for the whole block so the background purge timer
        defers instead of competing, and idle rest is restored on **every** exit
        path including abort and error: a rig left dry overnight clogs exactly as
        badly as one left stagnant. A parked rig refuses to rest, which is
        correct — a park should stay visible.
        """
        self._rig_activity.acquire(owner)
        self.leave_idle_rest()
        try:
            yield
        finally:
            try:
                if rest_after:
                    self.enter_idle_rest()
            finally:
                # Released last, and unconditionally: a leaked claim would
                # silently disable purging for the rest of the session, which
                # looks exactly like "the harness is switched off".
                self._rig_activity.release(owner)

    def _on_purge_tick(self) -> None:
        """Purge if one is owed and the rig is genuinely idle.

        Every precondition is re-checked inside the runner; this only decides
        *when to ask*. Exceptions are swallowed because a background timer that
        can kill the GUI is worse than a missed purge.
        """
        # Cheap, and the only periodic tick the window has: a campaign park sets
        # no latch of its own, so without this the clear-park control would never
        # appear for the unattended case it exists for.
        self._refresh_park_indicator()
        if self._purge_runner is None:
            return
        try:
            outcome = self._purge_runner.maybe_purge(context="idle")
        except Exception:
            logger.warning("purge_tick_failed", exc_info=True)
            return
        if outcome.performed or outcome.dry_run:
            self.statusBar().showMessage(outcome.summary(), 8000)

    def closeEvent(self, event) -> None:
        """Stop ALL worker threads before closing.

        Order: (0) signal cooperative abort to every daemon runner up front so
        long/hardware runs stop issuing work immediately and wind down in
        parallel; (1) QThread tab cleanups (prior spec); (2) own QThread
        workers (prior spec); (3) daemon-tab cleanups that join the now-aborted
        runners; (4) defensive findChildren(QThread) sweep (prior spec).
        """
        # The Process Studio Builder (embedded SandboxTab) is the runnable sandbox
        # now that the standalone tab is retired; include it so its daemon runs
        # wind down on close.
        daemon_tabs = [
            getattr(self, "_tab_experiment", None),
            getattr(getattr(self, "_tab_process_studio", None), "_sandbox", None),
            getattr(self, "_tab_bo_simulator", None),
            getattr(self, "_tab_bo_live", None),
            getattr(self, "_tab_arrhenius", None),
        ]

        # 0a. Stop the anti-clog purge timer before anything else winds down, so
        #     it cannot fire a dispense into a half-torn-down rig. P1.7 requires
        #     it to stop cleanly on close and not resurrect during recovery.
        if getattr(self, "_purge_timer", None) is not None:
            try:
                self._purge_timer.stop()
            except Exception:
                pass

        # 0b. …and the campaign tick, which reads files: a poll that fires
        #     during teardown would be reading a stream to update widgets that
        #     are being destroyed.
        if getattr(self, "_campaign_timer", None) is not None:
            try:
                self._campaign_timer.stop()
            except Exception:
                pass

        # 0. Signal abort to every daemon runner FIRST (parallel wind-down; no join).
        for tab in daemon_tabs:
            abort = getattr(tab, "abort_run", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass

        # 1. Ask each tab that owns worker threads to stop them (idempotent).
        for tab in (
            getattr(self, "_tab_init", None),
            getattr(self, "_tab_manual", None),
            getattr(self, "_tab_monitor", None),
        ):
            cleanup = getattr(tab, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass  # never let one tab's cleanup block the others / the close

        # 2. Stop MainWindow's own long-lived workers (existing logic, unchanged).
        if hasattr(self, "_cam_worker") and self._cam_worker.isRunning():
            self._cam_worker.stop_worker()
        if hasattr(self, "_webcam_worker") and self._webcam_worker.isRunning():
            self._webcam_worker.stop_worker()
        if hasattr(self, "_poller") and self._poller.isRunning():
            self._poller.stop_worker()   # was inlined requestInterruption+poke+wait

        # 3. Daemon-tab cleanups — abort already fired above, so these just join
        #    the (already winding-down) daemon threads with a bounded timeout.
        for tab in daemon_tabs + [getattr(self, "_web_launcher", None)]:
            cleanup = getattr(tab, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

        # 4. Defensive belt-and-braces: stop any QThread still running anywhere
        #    under the window (transient workers mid-flight, future tabs, etc.).
        from PySide6.QtCore import QThread
        for thread in self.findChildren(QThread):
            if not thread.isRunning():
                continue
            stop = getattr(thread, "stop_worker", None)
            if callable(stop):
                stop()
            else:
                thread.requestInterruption()
                poke = getattr(thread, "poke", None)
                if callable(poke):
                    poke()
                thread.wait(5000)

        # 5. Leave the rig in its safe resting state.  Runs are aborted above, but
        #    aborting only stops *issuing* work — it does not retract the head,
        #    halt the pumps, or drop the temperature setpoint.  Without this, a
        #    normal close left the head down and the heater at setpoint.
        #    Runs last, after the workers are stopped, so nothing re-commands the
        #    hardware afterwards; before the manager disconnects (see gui/app.py).
        #
        #    Calls what it was given at construction — no lock read, no ownership
        #    test, no flag. Park follows the instrument session, not the campaign:
        #    a window that opened nothing was built with nothing to call here.
        self._exit_park()

        super().closeEvent(event)

    def _safe_park_on_exit(self) -> None:
        """Best-effort park during shutdown — never blocks the close.

        Installed as :attr:`_exit_park` only in owner mode; an attached window is
        constructed without it.

        Retracts the head **unless** the operator just chose otherwise via Safe Exit.
        Closing the window is otherwise an unattended act — nobody is left to decide
        — so raising it stays the default for every other route out, including the
        window's X button.
        """
        import structlog

        log = structlog.get_logger(__name__)
        try:
            from softae.core.safe_park import safe_park

            retract = not getattr(self, "_skip_exit_retract", False)
            result = safe_park(self._manager, reason="application closing",
                               retract_head=retract)
            # ``severe``, not ``not ok``. This close is unattended by definition,
            # so the log line is the only account of it — and ``ok`` is true of a
            # park that raised nothing *because it reached nothing*, which on the
            # way out of an owner-mode window means the sessions were already
            # gone. That used to log nothing at all.
            text, severe = result.headline()
            if severe:
                log.warning("safe_park_on_exit_incomplete", headline=text,
                            summary=result.summary(), errors=result.errors)
        except Exception:
            log.warning("safe_park_on_exit_failed", exc_info=True)
