"""Every in-process run kind claims the rig, so the purge defers for all three.

The defect this file pins is one of omission. ``MainWindow.rig_run`` — the
context manager that takes the ``RigActivity`` claim — had exactly one caller,
the HT tab. An Arrhenius sweep and a Sandbox run each ran on a bare daemon
thread and claimed nothing, so the background anti-clog purge did **not** defer
during either: it was free to travel the stage to the flush basin and fire the
syringe in the middle of a live sweep. Latent only because ``[purge] actuate``
ships ``false``.

The assertions are therefore written as *purge* assertions wherever they can be.
"Did the tab call something" is a test of the edit; "does the purge decline, and
name the run" is a test of the defect.

``_HostWindow`` binds the **real** ``MainWindow.rig_run`` rather than restating
it. A stand-in that accepted keywords the real one does not would let every test
here pass while production raised ``TypeError`` on the first sweep — the exact
shape of failure this arc has already hit once.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from softae.core.purge import PurgeScheduler, PurgeSettings
from softae.core.purge_runner import IdleRestState, PurgeRunner
from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.main_window import MainWindow
from softae.gui.rig_claim import rig_run
from softae.gui.tabs.tab_arrhenius import ArrheniusTab
from softae.gui.tabs.tab_experiment import ExperimentBuilderTab
from softae.gui.tabs.tab_sandbox import SandboxTab
from softae.workflows.workflow_model import Workflow, WorkflowStep


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ── The rig, faked just far enough to answer "may I purge?" ─────────────────

FLUSH = (-50.0, 50.0)


class _Syringe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []
        self.head_up = False

    def is_head_up(self) -> bool:
        return self.head_up

    def head_descend(self) -> None:
        self.head_up = False

    def head_retract(self) -> None:
        self.head_up = True

    def single_pump(self, *, res_vol, ID, rate, dispense_vol) -> None:
        self.calls.append((int(ID), float(dispense_vol)))


class _Stage:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []

    def live_position(self):
        return FLUSH

    def move_to(self, x, y, *, head_may_be_down: bool = False) -> None:
        self.moves.append((float(x), float(y)))


class _PurgeManager:
    def __init__(self) -> None:
        self.syringe = _Syringe()
        self.stage = _Stage()

    def get(self, name):
        return {"syringe": self.syringe, "stage": self.stage}[name]


def _due_runner(activity: RigActivity) -> PurgeRunner:
    """A purge that is owed right now, on a rig resting at the flush basin."""
    settings = PurgeSettings(enabled=True, actuate=True, interval_s=900.0,
                             particulate_uL=20.0, other_uL=10.0,
                             particulate_pumps=(1,), pumps=(0, 1, 2))
    clock = SimpleNamespace(t=0.0)
    sched = PurgeScheduler(settings, now=lambda: clock.t)
    clock.t = 1000.0                       # past the interval → due
    return PurgeRunner(_PurgeManager(), sched, activity=activity,
                       idle_rest=IdleRestState(True), flush_xy=FLUSH)


def _purge_deferral(activity: RigActivity) -> str | None:
    """Why the purge declined right now, or ``None`` if it went ahead."""
    return _due_runner(activity).maybe_purge(context="idle").skipped_reason


# ── The host window: the real adapter, minimal surroundings ─────────────────

class _HostWindow(QWidget):
    """A top-level window offering the genuine ``MainWindow.rig_run``.

    Only three attributes of ``MainWindow`` are reachable from that method, and
    all three are here. Constructing a real ``MainWindow`` per test would cost
    seconds and prove nothing extra about the claim.
    """

    rig_run = MainWindow.rig_run

    def __init__(self) -> None:
        super().__init__()
        self._rig_activity = RigActivity()
        self.rest_events: list[str] = []
        self.setLayout(QVBoxLayout())

    def leave_idle_rest(self) -> bool:
        self.rest_events.append("leave")
        return True

    def enter_idle_rest(self) -> bool:
        self.rest_events.append("enter")
        return True


@pytest.fixture
def host(qapp):
    w = _HostWindow()
    yield w
    w.close()


def _attach(host: _HostWindow, tab: QWidget) -> None:
    """Parent *tab* so ``tab.window()`` resolves to *host*, as the shell does."""
    host.layout().addWidget(tab)


# ── Workflows and sweeps whose runs observe the rig from inside ─────────────

def _step(name: str, instrument: str) -> WorkflowStep:
    return WorkflowStep(name=name, instrument=instrument, method="noop")


def _cast_workflow(name: str = "cast_series") -> Workflow:
    return Workflow(
        name=name,
        setup=[_step("flush", "syringe")],
        loop_steps=[_step("goto", "stage"), _step("measure", "espico")],
        iterations=2,
    )


class _ObservingExecutor:
    """An executor whose only job is to look at the rig while it 'runs'."""

    def __init__(self, activity: RigActivity) -> None:
        self._activity = activity
        self.conflict_during_run: str | None = None
        self.deferral_during_run: str | None = None
        self.ran = False

    async def run(self, wf) -> None:
        self.ran = True
        self.conflict_during_run = self._activity.conflicts(PURGE_INSTRUMENTS)
        self.deferral_during_run = _purge_deferral(self._activity)


class _FailingExecutor(_ObservingExecutor):
    async def run(self, wf) -> None:
        await super().run(wf)
        raise RuntimeError("step 3 exploded")


class _ObservingSweep:
    """A stand-in for ``ArrheniusSweep`` with the attributes the thread reads."""

    def __init__(self, activity: RigActivity, *, run_id="20260820T000000Z_arr",
                 fail: bool = False) -> None:
        self._activity = activity
        self.run_id = run_id
        self.temp_instrument = "temp_controller"
        self.eis_instrument = "pico1"
        self.config = SimpleNamespace(rh_instrument="rh_controller",
                                      thermal_model="arrhenius")
        self.conflict_during_run: str | None = None
        self.deferral_during_run: str | None = None
        self.ran = False
        self._fail = fail

    async def run(self) -> list:
        self.ran = True
        self.conflict_during_run = self._activity.conflicts(PURGE_INSTRUMENTS)
        self.deferral_during_run = _purge_deferral(self._activity)
        if self._fail:
            raise RuntimeError("temperature controller stopped answering")
        return []

    def abort(self) -> None:
        pass


def _mute(tab, *signal_names: str) -> None:
    """Detach a tab's run-completion slots.

    What is under test is the run thread's claim, not the UI epilogue that
    follows it — and that epilogue re-reads a real ``WorkflowExecutor.state``
    and exports plot files to disk. The production thread queues these signals;
    pytest-qt drains the queue at teardown, which is where they would otherwise
    land.
    """
    for name in signal_names:
        try:
            getattr(tab, name).disconnect()
        except (RuntimeError, TypeError):
            pass


def _run_off_thread(fn, *args) -> None:
    """Run a tab's run-thread body the way the tab does: on its own thread.

    Off-thread so the tab's Qt signals queue rather than dispatching into slots
    that would export plots and write files on the way past.
    """
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    t.join(timeout=20.0)
    assert not t.is_alive(), "run thread did not finish"


# ── HT: unchanged in every respect but the scope it now passes ──────────────

class TestHTClaim:
    @pytest.fixture
    def tab(self, qapp, host):
        widget = ExperimentBuilderTab(create_mock_manager(config={}))
        _attach(host, widget)
        widget._exp_logger = None
        _mute(widget, "_sig_workflow_done")
        yield widget
        widget.close()

    def test_ht_run_purge_defers_naming_the_workflow(self, tab, host):
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_workflow_thread, _cast_workflow())

        assert tab._executor.ran
        assert tab._executor.conflict_during_run == "ht:cast_series"
        assert "ht:cast_series" in (tab._executor.deferral_during_run or "")

    def test_ht_run_claim_scoped_to_the_workflow_steps(self, tab, host):
        """The one thing that changes for HT: the claim is no longer blanket."""
        captured: dict = {}
        real = host.rig_run

        def _spy(owner, **kw):
            captured.update(owner=owner, **kw)
            return real(owner, **kw)

        host.rig_run = _spy
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_workflow_thread, _cast_workflow())

        assert captured["owner"] == "ht:cast_series"
        assert captured["instruments"] == frozenset({"syringe", "stage", "espico"})
        assert captured.get("manage_rest", True) is True

    def test_ht_run_still_returns_the_rig_to_idle_rest(self, tab, host):
        """HT's idle-rest convention is untouched by the hoist."""
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_workflow_thread, _cast_workflow())
        assert host.rest_events == ["leave", "enter"]

    def test_ht_run_completed_leaves_the_registry_empty(self, tab, host):
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_workflow_thread, _cast_workflow())

        assert host._rig_activity.owners() == ()
        assert host._rig_activity.busy is False
        assert _purge_deferral(host._rig_activity) is None

    def test_ht_run_raised_releases_the_claim(self, tab, host):
        tab._executor = _FailingExecutor(host._rig_activity)
        _run_off_thread(tab._run_workflow_thread, _cast_workflow())

        assert tab._executor.conflict_during_run == "ht:cast_series"
        assert host._rig_activity.busy is False


# ── Sandbox: claimed for the first time ─────────────────────────────────────

class TestSandboxClaim:
    @pytest.fixture
    def tab(self, qapp, host):
        widget = SandboxTab(create_mock_manager(config={}))
        _attach(host, widget)
        _mute(widget, "_sig_done")
        yield widget
        widget.close()

    def test_sandbox_run_purge_defers_naming_the_workflow(self, tab, host):
        """The X1 regression for the second uncovered path."""
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, _cast_workflow("bench_test"))

        assert tab._executor.ran
        assert tab._executor.conflict_during_run == "sandbox:bench_test"
        assert "sandbox:bench_test" in (tab._executor.deferral_during_run or "")

    def test_sandbox_run_claim_scoped_to_the_workflow_steps(self, tab, host):
        captured: dict = {}
        real = host.rig_run

        def _spy(owner, **kw):
            captured.update(owner=owner, **kw)
            return real(owner, **kw)

        host.rig_run = _spy
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, _cast_workflow("bench_test"))

        assert captured["owner"] == "sandbox:bench_test"
        assert captured["instruments"] == frozenset({"syringe", "stage", "espico"})
        assert captured["manage_rest"] is False

    def test_sandbox_run_moves_no_fluidics_of_its_own(self, tab, host):
        """Step B claims; it does not add motion the run never asked for."""
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, _cast_workflow("bench_test"))
        assert host.rest_events == []

    def test_sandbox_run_empty_workflow_claims_the_whole_rig(self, tab, host):
        """A workflow naming no instrument widens, and the purge still defers."""
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, Workflow(name="empty"))

        assert tab._executor.conflict_during_run == "sandbox:empty"

    def test_sandbox_run_completed_leaves_the_registry_empty(self, tab, host):
        tab._executor = _ObservingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, _cast_workflow("bench_test"))

        assert host._rig_activity.owners() == ()
        assert _purge_deferral(host._rig_activity) is None

    def test_sandbox_run_raised_releases_the_claim(self, tab, host):
        tab._executor = _FailingExecutor(host._rig_activity)
        _run_off_thread(tab._run_thread_fn, _cast_workflow("bench_test"))

        assert tab._executor.conflict_during_run == "sandbox:bench_test"
        assert host._rig_activity.busy is False


# ── Arrhenius: claimed for the first time, and whole-rig by ruling ──────────

class TestArrheniusClaim:
    @pytest.fixture
    def tab(self, qapp, host):
        widget = ArrheniusTab(create_mock_manager(config={}))
        _attach(host, widget)
        widget._run_id = None
        _mute(widget, "_sig_sweep_done")
        yield widget
        widget.close()

    def test_arrhenius_sweep_purge_defers_naming_the_sweep(self, tab, host):
        """The X1 regression, stated as a purge assertion.

        Before step B this returned ``None``: the sweep claimed nothing, so a
        due purge would have travelled the stage and dispensed mid-sweep.
        """
        sweep = _ObservingSweep(host._rig_activity)
        _run_off_thread(tab._run_sweep_thread, sweep)

        assert sweep.ran
        assert sweep.deferral_during_run is not None
        assert "arrhenius:20260820T000000Z_arr" in sweep.deferral_during_run

    def test_arrhenius_sweep_claims_the_whole_rig_not_its_three_instruments(
        self, tab, host
    ):
        """Commanded is not occupied.

        ``{temp_controller, rh_controller, pico1}`` is what the sweep *drives*,
        and it is exactly the scope that would leave the purge free to move the
        stage. Whole rig is the ruling, and this is the assertion that fails if
        someone narrows it to the derivable-looking set.
        """
        captured: dict = {}
        real = host.rig_run

        def _spy(owner, **kw):
            captured.update(owner=owner, **kw)
            return real(owner, **kw)

        host.rig_run = _spy
        _run_off_thread(tab._run_sweep_thread, _ObservingSweep(host._rig_activity))

        assert captured["owner"] == "arrhenius:20260820T000000Z_arr"
        assert captured["instruments"] is None
        assert captured["manage_rest"] is False

    def test_arrhenius_sweep_without_a_run_id_still_claims(self, tab, host):
        sweep = _ObservingSweep(host._rig_activity, run_id=None)
        _run_off_thread(tab._run_sweep_thread, sweep)
        assert sweep.conflict_during_run == "arrhenius:sweep"

    def test_arrhenius_sweep_moves_no_fluidics_of_its_own(self, tab, host):
        """The tip stays resting in flush for the hours the sweep lasts."""
        _run_off_thread(tab._run_sweep_thread, _ObservingSweep(host._rig_activity))
        assert host.rest_events == []

    def test_arrhenius_sweep_completed_leaves_the_registry_empty(self, tab, host):
        _run_off_thread(tab._run_sweep_thread, _ObservingSweep(host._rig_activity))

        assert host._rig_activity.owners() == ()
        assert host._rig_activity.busy is False
        assert _purge_deferral(host._rig_activity) is None

    def test_arrhenius_sweep_raised_releases_the_claim(self, tab, host):
        sweep = _ObservingSweep(host._rig_activity, fail=True)
        _run_off_thread(tab._run_sweep_thread, sweep)

        assert sweep.conflict_during_run is not None
        assert host._rig_activity.busy is False


# ── The adapter itself ──────────────────────────────────────────────────────

class TestRigRunAdapter:
    def test_rig_run_windowless_host_is_a_noop(self, qapp):
        """A tab with no shell around it stays usable — most of the suite."""
        orphan = ExperimentBuilderTab(create_mock_manager(config={}))
        try:
            with rig_run(orphan, "ht:x", instruments={"stage"}):
                pass
        finally:
            orphan.close()

    def test_rig_run_non_widget_host_is_a_noop(self):
        with rig_run(object(), "ht:x"):
            pass

    def test_rig_run_hosted_claims_and_releases(self, host, qapp):
        tab = QWidget()
        _attach(host, tab)
        with rig_run(tab, "probe:one", instruments={"stage"}):
            assert host._rig_activity.conflicts({"stage"}) == "probe:one"
        assert host._rig_activity.busy is False

    def test_rig_run_hosted_releases_on_exception(self, host, qapp):
        tab = QWidget()
        _attach(host, tab)
        with pytest.raises(RuntimeError):
            with rig_run(tab, "probe:one"):
                raise RuntimeError("boom")
        assert host._rig_activity.busy is False

    def test_rig_run_is_the_only_adapter(self):
        """One home. A second copy is what this hoist exists to prevent.

        ``tab_experiment`` carried its own ``_rig_run``; growing a second and a
        third in the other two tabs is the duplication the review sweep exists
        to catch. The absence is asserted rather than merely intended.
        """
        for tab_cls in (ExperimentBuilderTab, SandboxTab, ArrheniusTab):
            assert not hasattr(tab_cls, "_rig_run"), tab_cls.__name__

    def test_rig_run_default_scope_is_the_whole_rig(self, host, qapp):
        """The conservative default survives; opting in is what narrows."""
        tab = QWidget()
        _attach(host, tab)
        with rig_run(tab, "probe:one"):
            assert host._rig_activity.conflicts(PURGE_INSTRUMENTS) == "probe:one"
            assert host._rig_activity.conflicts({"anything_at_all"}) == "probe:one"
