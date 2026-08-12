"""Tests for MainWindow catalog menu reach + catalogs_changed refresh hook (Fix 3)."""

from __future__ import annotations

import threading
import time

import pytest
from PySide6.QtWidgets import QApplication, QMenu

from softae.config import loader
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_experiment import ExperimentBuilderTab
from softae.gui.widgets.catalog_browser import CatalogBrowser
from softae.gui.widgets.catalog_manager import CatalogManager
from softae.gui.widgets.deposition_panel import DepositionPanel
from softae.gui.widgets.formulation_panel import FormulationPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _synthetic_chem() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.0))
    cat.add(Chemical("Isopropanol", "CC(O)C", density_g_per_mL=0.786))
    cat.add(Chemical("Fumed silica", "O=[Si]=O", density_g_per_mL=2.65, is_particulate=True))
    return cat


def _synthetic_sol() -> SolutionCatalog:
    cat = SolutionCatalog()
    cat.add(Solution("Silica solution", [
        SolutionComponent("Fumed silica", "dep", 1.0, "g"),
        SolutionComponent("Isopropanol", "carrier", 9.0, "mL"),
    ]))
    return cat


@pytest.fixture
def mock_manager():
    return create_mock_manager(config={})


class _StubExecutor:
    """Executor stub whose abort() sets a threading.Event (the run's abort signal)."""

    def __init__(self) -> None:
        self.ev = threading.Event()

    def abort(self) -> None:
        self.ev.set()


class _StubSweep:
    """Sweep stub whose abort() sets a threading.Event (the run's abort signal)."""

    def __init__(self) -> None:
        self.ev = threading.Event()

    def abort(self) -> None:
        self.ev.set()


def _spin_on_event(ev: threading.Event) -> threading.Thread:
    def run() -> None:
        while not ev.is_set():
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


@pytest.fixture
def main_window(qapp, qtbot, monkeypatch, mock_manager):
    # Keep the webcam thread from starting during the test.
    monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
    from softae.gui.main_window import MainWindow

    mw = MainWindow(mock_manager)
    qtbot.addWidget(mw)
    yield mw

    # Deterministic teardown — the product MainWindow.closeEvent now stops every
    # worker QThread (tab cleanups + own workers + defensive findChildren sweep),
    # so the fixture just closes the window and lets deleteLater() run while the
    # session QApplication is still alive.  Nothing survives to interpreter
    # shutdown where GC would destroy Qt objects in a bad order (Windows
    # STATUS_STACK_BUFFER_OVERRUN segfault at exit).
    mw.close()
    qapp.processEvents()
    mw.deleteLater()
    qapp.processEvents()


class TestCatalogMenu:
    def test_catalogs_menu_action_exists(self, main_window):
        texts = [
            a.text()
            for menu in main_window.menuBar().findChildren(QMenu)
            for a in menu.actions()
        ]
        assert "Edit Catalogs…" in texts

    def test_edit_catalogs_action_opens_catalog_manager_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        # Spec change: the menu/toolbar action now opens the SLIM CatalogManager,
        # NOT the full FormulationPanel (which stays on the tab-5 button).
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        made_mgr: list[int] = []
        made_panel: list[int] = []
        orig_mgr_init = CatalogManager.__init__
        orig_panel_init = FormulationPanel.__init__

        def spy_mgr(self, *args, **kwargs):
            made_mgr.append(1)
            orig_mgr_init(self, *args, **kwargs)

        def spy_panel(self, *args, **kwargs):
            made_panel.append(1)
            orig_panel_init(self, *args, **kwargs)

        monkeypatch.setattr(CatalogManager, "__init__", spy_mgr)
        monkeypatch.setattr(CatalogManager, "exec", lambda self: 0)
        monkeypatch.setattr(FormulationPanel, "__init__", spy_panel)
        monkeypatch.setattr(FormulationPanel, "exec", lambda self: 0)
        main_window._edit_catalogs_action.trigger()
        assert made_mgr  # a CatalogManager was constructed
        assert not made_panel  # NOT a FormulationPanel

    def test_browser_edit_requested_opens_catalog_manager_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        made: list[int] = []
        orig_init = CatalogManager.__init__

        def spy_init(self, *args, **kwargs):
            made.append(1)
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(CatalogManager, "__init__", spy_init)
        monkeypatch.setattr(CatalogManager, "exec", lambda self: 0)
        main_window._catalog_browser.edit_requested.emit()
        assert made

    def test_tab5_formulation_button_still_opens_formulation_panel_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        made: list[int] = []
        orig_init = FormulationPanel.__init__

        def spy_init(self, *args, **kwargs):
            made.append(1)
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(FormulationPanel, "__init__", spy_init)
        monkeypatch.setattr(FormulationPanel, "exec", lambda self: 0)
        main_window._tab_experiment._btn_formulation.click()
        assert made  # the full FormulationPanel path is preserved

    def test_on_catalogs_changed_refreshes_both_browser_and_deposition_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        reloads: list[int] = []
        set_cats: list[int] = []
        monkeypatch.setattr(
            main_window._catalog_browser, "reload", lambda *a, **k: reloads.append(1)
        )
        monkeypatch.setattr(
            main_window._deposition_panel, "set_catalogs", lambda *a, **k: set_cats.append(1)
        )
        main_window._on_catalogs_changed()
        assert len(reloads) == 1
        assert len(set_cats) == 1

    def test_tab_button_delegates_to_shared_opener(self, main_window, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(
            ExperimentBuilderTab,
            "open_formulation_manager",
            lambda self: calls.append(1),
        )
        main_window._tab_experiment._btn_formulation.click()
        assert len(calls) == 1

    def test_catalogs_changed_bridged_to_refresh_hook(
        self, main_window, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        main_window._tab_experiment.catalogs_changed.emit()
        assert main_window._chem_catalog is not None
        assert len(main_window._chem_catalog) == 3


class TestCatalogTab:
    def test_main_window_has_catalogs_tab_expected(self, main_window):
        idx = None
        for i in range(main_window._tabs.count()):
            if main_window._tabs.tabText(i) == "11. Catalogs":
                idx = i
                break
        assert idx is not None
        assert isinstance(main_window._tabs.widget(idx), CatalogBrowser)

    def test_catalogs_changed_reloads_browser_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        main_window._tab_experiment.catalogs_changed.emit()
        assert main_window._catalog_browser._chem_table.rowCount() == 3


class TestDepositionTab:
    def test_main_window_has_deposition_tab_expected(self, main_window):
        idx = None
        for i in range(main_window._tabs.count()):
            if main_window._tabs.tabText(i) == "12. Deposition":
                idx = i
                break
        assert idx is not None
        assert isinstance(main_window._tabs.widget(idx), DepositionPanel)

    def test_deposition_panel_constructed_with_injected_catalogs_expected(self, main_window):
        assert main_window._deposition_panel._sol_catalog is main_window._sol_catalog
        assert main_window._deposition_panel._chem_catalog is main_window._chem_catalog

    def test_catalogs_changed_calls_set_catalogs_on_deposition_panel_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        _synthetic_chem().save_csv(tmp_path / "chemicals.csv")
        _synthetic_sol().save_csv(tmp_path / "solutions.csv")
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        calls: list[int] = []
        monkeypatch.setattr(
            main_window._deposition_panel, "set_catalogs", lambda *a, **k: calls.append(1)
        )
        main_window._tab_experiment.catalogs_changed.emit()
        assert len(calls) == 1

    def test_deposition_manage_catalogs_requested_opens_catalog_manager_expected(
        self, main_window, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(loader, "data_root", lambda: tmp_path)
        made: list[int] = []
        orig_init = CatalogManager.__init__

        def spy_init(self, *args, **kwargs):
            made.append(1)
            orig_init(self, *args, **kwargs)

        monkeypatch.setattr(CatalogManager, "__init__", spy_init)
        monkeypatch.setattr(CatalogManager, "exec", lambda self: 0)
        main_window._deposition_panel.manage_catalogs_requested.emit()
        assert made


class TestThreadShutdown:
    """MainWindow.closeEvent stops every worker QThread it or its tabs started."""

    def test_close_stops_all_worker_threads(self, main_window):
        from PySide6.QtCore import QThread

        mw = main_window
        # The active init tab keeps its poll worker + the shared poller running.
        assert mw._tab_init._poll_worker.isRunning()
        assert mw._poller.isRunning()
        # The manual tab's _pv_worker is stopped by its hideEvent while it is not
        # the current tab; ensure it is running so close() genuinely stops it.
        if not mw._tab_manual._pv_worker.isRunning():
            mw._tab_manual._pv_worker.start()
        assert mw._tab_manual._pv_worker.isRunning()

        mw.close()

        assert all(not t.isRunning() for t in mw.findChildren(QThread))

    def test_close_stops_tab_init_position_worker(self, main_window):
        from PySide6.QtCore import QThread

        mw = main_window
        pos_worker = mw._tab_init._pos_map._pos_worker
        assert pos_worker.isRunning()

        mw.close()

        # The embedded PositionMapWidget worker is found via the child sweep.
        matches = [
            t for t in mw.findChildren(QThread) if t is pos_worker
        ]
        assert matches, "position worker should still be a child of the window"
        assert not pos_worker.isRunning()

    def test_cleanup_is_idempotent(self, main_window):
        mw = main_window
        # Two calls to each tab cleanup must not raise and must leave workers stopped.
        mw._tab_init.cleanup()
        mw._tab_init.cleanup()
        mw._tab_manual.cleanup()
        mw._tab_manual.cleanup()

        assert not mw._tab_init._poll_worker.isRunning()
        assert not mw._tab_manual._pv_worker.isRunning()

    def test_closeevent_runs_without_running_threads(self, main_window):
        mw = main_window
        mw.close()
        # Second close: everything already stopped; closeEvent must be idempotent.
        mw.close()

    # ── Daemon-runner cooperative abort on close ────────────────────────

    def test_close_aborts_running_daemon_runner(self, main_window):
        mw = main_window
        sweep = _StubSweep()
        mw._tab_arrhenius._sweep = sweep
        mw._tab_arrhenius._sweep_thread = _spin_on_event(sweep.ev)
        assert mw._tab_arrhenius._sweep_thread.is_alive()

        mw.close()

        assert sweep.ev.is_set()
        assert not mw._tab_arrhenius._sweep_thread.is_alive()

    def test_close_signals_all_daemon_runners_before_join(self, main_window):
        mw = main_window
        abort_ts: dict[str, float] = {}
        finished: list[tuple[str, float]] = []

        def _record_abort(tab, name):
            orig = tab.abort_run

            def wrapped():
                abort_ts.setdefault(name, time.perf_counter())
                orig()

            tab.abort_run = wrapped

        def _spin_record(is_aborted, name):
            def run():
                while not is_aborted():
                    time.sleep(0.02)
                finished.append((name, time.perf_counter()))

            t = threading.Thread(target=run, daemon=True)
            t.start()
            return t

        ex_e = _StubExecutor()
        mw._tab_experiment._executor = ex_e
        mw._tab_experiment._run_thread = _spin_record(ex_e.ev.is_set, "experiment")

        ex_s = _StubExecutor()
        sandbox = mw._tab_process_studio._sandbox  # embedded Builder (tab 9 retired)
        sandbox._executor = ex_s
        sandbox._run_thread = _spin_record(ex_s.ev.is_set, "sandbox")

        mw._tab_bo_simulator._abort_requested = False
        mw._tab_bo_simulator._thread = _spin_record(
            lambda: mw._tab_bo_simulator._abort_requested, "bo_campaign"
        )

        sweep = _StubSweep()
        mw._tab_arrhenius._sweep = sweep
        mw._tab_arrhenius._sweep_thread = _spin_record(sweep.ev.is_set, "arrhenius")

        for tab, name in (
            (mw._tab_experiment, "experiment"),
            (sandbox, "sandbox"),
            (mw._tab_bo_simulator, "bo_campaign"),
            (mw._tab_arrhenius, "arrhenius"),
        ):
            _record_abort(tab, name)

        mw.close()

        # Every runner was signalled and joined.
        assert set(abort_ts) == {"experiment", "sandbox", "bo_campaign", "arrhenius"}
        assert {n for n, _ in finished} == {
            "experiment",
            "sandbox",
            "bo_campaign",
            "arrhenius",
        }
        assert not mw._tab_experiment._run_thread.is_alive()
        assert not mw._tab_arrhenius._sweep_thread.is_alive()
        # Signal-first ordering: all four aborts fired before any thread finished.
        assert max(abort_ts.values()) < min(ts for _, ts in finished)

    def test_close_is_noop_when_no_daemon_run_active(self, main_window):
        mw = main_window
        # No injected daemon runs — close must return promptly without blocking
        # on any join (all daemon threads are None).
        start = time.perf_counter()
        mw.close()
        assert time.perf_counter() - start < 2.0

    def test_close_stops_started_webcam_worker(
        self, qapp, qtbot, monkeypatch, mock_manager
    ):
        from softae.gui.widgets import webcam_worker as _wcmod

        if not getattr(_wcmod, "_HAS_CV2", False):
            pytest.skip("cv2 not available — webcam worker cannot start")

        # Replace run() with an idle-until-abort loop so the worker stays running
        # deterministically without real webcam hardware.  stop_worker() sets the
        # abort flag and wakes the wait condition (unchanged), so close() stops it.
        def _idle_run(self):
            self._abort = False
            self._mutex.lock()
            while not self._abort:
                self._condition.wait(self._mutex, 100)
            self._mutex.unlock()

        monkeypatch.setattr(_wcmod.WebcamWorker, "run", _idle_run)
        monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": True}})

        from softae.gui.main_window import MainWindow

        mw = MainWindow(mock_manager)
        qtbot.addWidget(mw)
        try:
            if not mw._webcam_worker.isRunning():
                pytest.skip("webcam worker did not start")
            mw.close()
            assert not mw._webcam_worker.isRunning()
        finally:
            mw.close()
            qapp.processEvents()
            mw.deleteLater()
            qapp.processEvents()


class TestSafeParkOnExit:
    """Closing the app must leave the rig in its safe resting state.

    Aborting a run only stops *issuing* work; before this, a normal close left
    the head down and the heater at setpoint.
    """

    def test_close_parks_the_rig(self, main_window, monkeypatch):
        import softae.core.safe_park as sp

        calls: list[str] = []
        monkeypatch.setattr(
            sp, "safe_park",
            lambda mgr, **kw: calls.append(kw.get("reason", "")) or sp.SafeParkResult(),
        )
        main_window.close()
        assert calls, "closeEvent must drive the rig safe"
        assert "clos" in calls[0].lower()

    def test_close_still_succeeds_if_park_fails(self, main_window, monkeypatch):
        """A refusing instrument must never block the application from closing."""
        import softae.core.safe_park as sp

        def boom(*a, **k):
            raise RuntimeError("serial gone")

        monkeypatch.setattr(sp, "safe_park", boom)
        main_window.close()          # must not raise


# ── Anti-clog purge wiring (P8) ──────────────────────────────────────────────

class TestPurgeWiring:
    """The idle purge is the only thing that actuates with nobody asking."""

    def test_the_scheduler_is_attached_to_the_syringe(self, main_window, mock_manager):
        """Attached at the choke point so every dispense resets a line's timer."""
        assert getattr(mock_manager.get("syringe"), "purge_scheduler", None) is not None

    def test_a_fresh_window_is_not_at_idle_rest(self, main_window):
        """Nothing has put the rig at the flush station yet."""
        assert main_window._idle_rest.at_rest is False

    def test_the_idle_tick_does_not_purge_a_rig_that_is_not_at_rest(self, main_window):
        main_window._on_purge_tick()        # must not raise, must not dispense
        outcome = main_window._purge_runner.maybe_purge()
        assert not outcome.performed

    def test_an_estop_latches_a_park_reason(self, main_window):
        main_window.notify_parked("operator emergency stop")
        assert main_window._park_reason() == "operator emergency stop"

    def test_parking_drops_the_rig_out_of_idle_rest(self, main_window):
        """Whatever it was doing, a parked rig is not a safe purge target."""
        main_window._idle_rest.mark_entered()
        main_window.notify_parked("hard fault")
        assert main_window._idle_rest.at_rest is False

    def test_a_park_is_only_cleared_explicitly(self, main_window):
        main_window.notify_parked("hard fault")
        assert main_window._park_reason()
        main_window.clear_park()
        assert main_window._park_reason() is None

    def test_a_parked_rig_refuses_to_enter_idle_rest(self, main_window):
        main_window.notify_parked("hard fault")
        assert main_window.enter_idle_rest() is False
        assert main_window._idle_rest.at_rest is False

    def test_the_estop_button_emits_before_the_sequence_runs(
        self, main_window, qtbot, monkeypatch
    ):
        """Latch first — nothing may actuate while the stop is in flight.

        The real worker is stubbed out: this asserts the *ordering*, and running
        the actual sequence would pop the modal completion dialog.
        """
        import softae.gui.widgets.emergency_stop as estop_mod
        from PySide6.QtCore import QObject, Signal

        events: list[str] = []

        class _StubWorker(QObject):
            done = Signal(list)
            finished = Signal()

            def __init__(self, manager, parent=None):
                super().__init__(parent)

            def start(self):
                events.append("worker-started")

        monkeypatch.setattr(estop_mod, "_EStopWorker", _StubWorker)
        main_window._estop.parked.connect(lambda r: events.append(f"parked:{r}"))

        main_window._estop._on_stop()

        assert events == ["parked:operator emergency stop", "worker-started"]
        assert main_window._park_reason() == "operator emergency stop"

    def test_closing_stops_the_purge_timer(self, qapp, qtbot, monkeypatch,
                                           mock_manager):
        monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
        from softae.gui.main_window import MainWindow

        mw = MainWindow(mock_manager)
        qtbot.addWidget(mw)
        assert mw._purge_timer.isActive()
        mw.close()
        qapp.processEvents()
        assert not mw._purge_timer.isActive()
        mw.deleteLater()
