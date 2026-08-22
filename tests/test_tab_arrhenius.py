"""Tests for the Arrhenius sweep tab (tab_arrhenius.py).

Three concerns live here.

**Daemon shutdown** — hardware safety: cleanup()/abort_run() must signal the
sweep's abort (which stops issuing temp-controller / potentiostat commands)
before any join.

**Arbitration** — a sweep must be guarded on *both* axes: a ``RigActivity``
claim so nothing else in this process drives the rig (and so the anti-clog purge
defers), and the cross-process rig lock so no headless tool can start underneath
it. The claim's own coverage lives in ``tests/test_rig_claim.py`` alongside the
other two run kinds; what is asserted here is the **lock**, and the pairing —
that the two are held *together*, for the whole sweep, and released together on
every exit.

The lock is the subtler half. ``WorkflowExecutor.run`` already takes it, so a
sweep was never wholly unlocked — but ``ArrheniusSweep`` runs one executor *per
phase*, and between phases ``_run_rh_sweep`` starts the RH controller, writes its
setpoint and polls it outside any executor. ``_lock_between_phases`` below
reproduces exactly the executor's acquire/release dance so those gaps are
visible: without the tab-level hold, the lock file is genuinely free in them.

**Export paths** — an export must land under the DataStore root regardless of
where the process was started, so the CWD tests deliberately drive from a
directory that is neither the repo nor the store.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QVBoxLayout, QWidget

from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity
from softae.core.run_lock import RunLock, read_run_lock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.main_window import MainWindow
from softae.gui.tabs.tab_arrhenius import ArrheniusTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


@pytest.fixture
def tab(qapp, manager):
    widget = ArrheniusTab(manager)
    yield widget
    widget.close()


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


class TestDaemonShutdown:
    def test_arrhenius_cleanup_aborts_running_thread(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        assert tab._sweep_thread.is_alive()
        tab.cleanup()
        assert sw.ev.is_set()
        assert tab._abort_requested is True
        assert not tab._sweep_thread.is_alive()

    def test_arrhenius_cleanup_is_noop_when_idle(self, tab: ArrheniusTab):
        assert getattr(tab, "_sweep", None) is None
        assert tab._sweep_thread is None
        tab.cleanup()  # must not raise / block

    def test_arrhenius_cleanup_is_idempotent(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        tab.cleanup()
        tab.cleanup()
        assert not tab._sweep_thread.is_alive()

    def test_arrhenius_abort_run_signals_without_joining(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        tab.abort_run()
        assert sw.ev.is_set()
        assert tab._abort_requested is True
        assert tab._sweep_thread.is_alive()  # signal-only: not joined
        tab.cleanup()  # teardown join


class TestChannelParsing:
    """The channels field feeds ``ArrheniusSweepConfig(channels=...)`` directly.

    Order matters here in a way it does not elsewhere: for a sweep, channel
    order *is* measurement order, so these pin order preservation as hard as
    they pin validation.  ``_parse_channels`` delegates to the shared parser,
    which raises ``ChannelSpecError`` — a ``ValueError`` subclass, which is
    what the ``except (ValueError, TypeError)`` at the save-config call site
    catches.
    """

    def test_arrhenius_parse_channels_preserves_entry_order(self, tab: ArrheniusTab):
        tab.set_pcb_channel_count(32)
        tab._le_channels.setText("10, 2, 3-5")
        assert tab._parse_channels() == [10, 2, 3, 4, 5]

    def test_arrhenius_parse_channels_dedups_first_wins(self, tab: ArrheniusTab):
        tab.set_pcb_channel_count(32)
        tab._le_channels.setText("3, 1-3")
        assert tab._parse_channels() == [3, 1, 2]

    def test_arrhenius_parse_channels_bad_token_raises_a_value_error(
        self, tab: ArrheniusTab
    ):
        from softae.core.channel_spec import ChannelSpecError

        tab._le_channels.setText("1, x")
        with pytest.raises(ValueError):          # the call site's except clause
            tab._parse_channels()
        tab._le_channels.setText("1, x")
        with pytest.raises(ChannelSpecError):    # ...and it is the specific one
            tab._parse_channels()

    def test_arrhenius_parse_channels_empty_raises(self, tab: ArrheniusTab):
        tab._le_channels.setText("   ")
        with pytest.raises(ValueError):
            tab._parse_channels()

    def test_arrhenius_parse_channels_enforces_the_board_channel_bound(
        self, tab: ArrheniusTab
    ):
        tab.set_pcb_channel_count(8)
        tab._le_channels.setText("1, 9")
        with pytest.raises(ValueError):
            tab._parse_channels()
        tab.set_pcb_channel_count(16)
        assert tab._parse_channels() == [1, 9]   # same text, wider board


# ── Arbitration: the claim and the lock, together ───────────────────────────


class _HostWindow(QWidget):
    """A top-level window offering the genuine ``MainWindow.rig_run``.

    Bound from the real class rather than restated: a stand-in accepting
    keywords the real one does not would let every test here pass while
    production raised ``TypeError`` on the first sweep.
    """

    rig_run = MainWindow.rig_run

    def __init__(self) -> None:
        super().__init__()
        self._rig_activity = RigActivity()
        self.setLayout(QVBoxLayout())

    def leave_idle_rest(self) -> bool:      # pragma: no cover - manage_rest=False
        return True

    def enter_idle_rest(self) -> bool:      # pragma: no cover - manage_rest=False
        return True


@pytest.fixture
def host(qapp):
    w = _HostWindow()
    yield w
    w.close()


@pytest.fixture
def hosted_tab(qapp, manager, host):
    """A tab parented into a window, with its completion slots detached.

    What is under test is the run thread's arbitration, not the UI epilogue that
    follows it — and that epilogue exports plot files to disk on the way past.
    """
    widget = ArrheniusTab(manager)
    host.layout().addWidget(widget)
    widget._run_id = None
    try:
        widget._sig_sweep_done.disconnect()
    except (RuntimeError, TypeError):       # pragma: no cover - defensive
        pass
    yield widget
    widget.close()


@pytest.fixture
def real_rig(monkeypatch, tmp_path):
    """Make the tab believe the rig is real, and put its lock file in *tmp_path*.

    ``rig_is_simulated`` is patched on ``softae.core.run_lock`` — the tab imports
    it inside the method, so the module attribute is what it resolves. Without
    this the mock manager is (correctly) simulated and no lock is taken at all,
    which is its own test below.
    """
    import softae.core.run_lock as rl

    monkeypatch.setattr(rl, "DEFAULT_SCOPE", tmp_path)
    monkeypatch.setattr(rl, "rig_is_simulated", lambda _m: False)
    return tmp_path


def _lock_between_phases() -> RunLock | None:
    """Run the executor's own lock dance once, and report what it left behind.

    Copied in shape from ``WorkflowExecutor.run``: read, decide whether the lock
    was already ours, acquire, and release **only** what this call created. That
    ``mine_already`` discipline is the whole reason a tab-level hold nests safely
    — and the reason the gap between two phases exists without one.
    """
    from softae.core.run_lock import (
        acquire_run_lock,
        read_run_lock,
        release_run_lock,
    )

    before = read_run_lock()
    mine_already = before is not None and before.is_mine()
    acquire_run_lock(what="workflow 'temp_eis_sweep'")
    if not mine_already:
        release_run_lock()
    return read_run_lock()


class _ObservingSweep:
    """A stand-in for ``ArrheniusSweep`` carrying the attributes the thread reads.

    Its ``run`` is where the assertions are taken from: the point of the whole
    exercise is what the rig looks like *while the sweep is running*, which is
    unobservable from outside the thread.
    """

    def __init__(self, activity: RigActivity, *, run_id="20260821T000000Z_arr",
                 raise_exc: BaseException | None = None) -> None:
        self._activity = activity
        self.run_id = run_id
        self.temp_instrument = "temp_controller"
        self.eis_instrument = "pico1"
        self.config = SimpleNamespace(rh_instrument="rh_controller",
                                      thermal_model="arrhenius")
        self.ran = False
        self.claim_during_run: str | None = None
        self.lock_during_run: RunLock | None = None
        self.lock_between_phases: RunLock | None = None
        self._raise = raise_exc

    async def run(self) -> list:
        self.ran = True
        self.claim_during_run = self._activity.conflicts(PURGE_INSTRUMENTS)
        self.lock_during_run = read_run_lock()
        self.lock_between_phases = _lock_between_phases()
        if self._raise is not None:
            raise self._raise
        return []

    def abort(self) -> None:                # pragma: no cover - not exercised
        pass


def _run_off_thread(fn, *args) -> BaseException | None:
    """Run the tab's run-thread body the way the tab does: on its own thread.

    Off-thread so the tab's Qt signals queue rather than dispatching into slots.
    Returns anything that escaped, so a ``BaseException`` (an abort's
    ``CancelledError``) can be asserted on rather than lost.
    """
    escaped: list[BaseException] = []

    def _target():
        try:
            fn(*args)
        except BaseException as exc:        # noqa: BLE001 - that is the point
            escaped.append(exc)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=20.0)
    assert not t.is_alive(), "sweep thread did not finish"
    return escaped[0] if escaped else None


class TestSweepArbitration:
    def test_arrhenius_sweep_holds_both_the_claim_and_the_run_lock(
        self, hosted_tab, host, real_rig
    ):
        """The pairing. Either half missing and this sweep is unarbitrated.

        This is the load-bearing assertion: the sweep used to run on a bare
        daemon thread holding neither, making it the one actuating GUI path
        guarded on no axis at all.
        """
        sweep = _ObservingSweep(host._rig_activity)
        assert _run_off_thread(hosted_tab._run_sweep_thread, sweep) is None

        assert sweep.ran
        assert sweep.claim_during_run == "arrhenius:20260821T000000Z_arr"
        assert sweep.lock_during_run is not None
        assert sweep.lock_during_run.is_mine()
        assert "Arrhenius sweep" in sweep.lock_during_run.what

    def test_arrhenius_sweep_lock_survives_a_phase_boundary(
        self, hosted_tab, host, real_rig
    ):
        """The defect the tab-level hold exists for.

        ``ArrheniusSweep`` runs one ``WorkflowExecutor`` per phase; an RH sweep
        runs several, and between them it drives the RH controller directly. Each
        executor releases the lock it took, so without an outer hold the rig is
        unlocked in exactly those windows — while a board sits at setpoint.
        """
        sweep = _ObservingSweep(host._rig_activity)
        _run_off_thread(hosted_tab._run_sweep_thread, sweep)

        assert sweep.lock_between_phases is not None, (
            "the rig lock was free between two executor phases"
        )
        assert sweep.lock_between_phases.is_mine()

    def test_arrhenius_sweep_releases_both_on_completion(
        self, hosted_tab, host, real_rig
    ):
        _run_off_thread(hosted_tab._run_sweep_thread,
                        _ObservingSweep(host._rig_activity))

        assert host._rig_activity.busy is False
        assert read_run_lock() is None

    def test_arrhenius_sweep_releases_both_on_failure(
        self, hosted_tab, host, real_rig
    ):
        sweep = _ObservingSweep(
            host._rig_activity,
            raise_exc=RuntimeError("temperature controller stopped answering"),
        )
        _run_off_thread(hosted_tab._run_sweep_thread, sweep)

        assert sweep.claim_during_run is not None
        assert host._rig_activity.busy is False
        assert read_run_lock() is None

    def test_arrhenius_sweep_releases_both_on_abort(
        self, hosted_tab, host, real_rig
    ):
        """An abort cancels the executor, and ``CancelledError`` is a
        ``BaseException`` — so it escapes the thread's ``except Exception``
        entirely. The ``with`` blocks are what release, and they still do.
        """
        import asyncio

        sweep = _ObservingSweep(host._rig_activity,
                                raise_exc=asyncio.CancelledError("abort"))
        escaped = _run_off_thread(hosted_tab._run_sweep_thread, sweep)

        assert isinstance(escaped, asyncio.CancelledError)
        assert host._rig_activity.busy is False
        assert read_run_lock() is None

    def test_arrhenius_sweep_refused_while_another_process_holds_the_rig(
        self, hosted_tab, host, real_rig, monkeypatch
    ):
        """Refused before anything is claimed, and the holder is named."""
        import json

        import softae.core.run_lock as rl

        monkeypatch.setattr(rl, "_pid_alive", lambda _pid: True)
        (real_rig / rl.LOCK_FILENAME).write_text(
            json.dumps({"pid": os.getpid() + 100000,
                        "what": "commissioning blank_short",
                        "started_at": "2026-08-21T14:02:00+00:00", "host": ""}),
            encoding="utf-8",
        )

        lines: list[str] = []
        hosted_tab._sig_log_line.connect(lines.append)
        sweep = _ObservingSweep(host._rig_activity)
        assert _run_off_thread(hosted_tab._run_sweep_thread, sweep) is None

        # The refusal is emitted from the sweep thread; drain the queue so a
        # queued connection is delivered before the message is read.
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.processEvents()

        assert sweep.ran is False, "the sweep ran despite a foreign rig lock"
        assert host._rig_activity.busy is False
        message = "\n".join(lines)
        assert "commissioning blank_short" in message
        assert "Calibration Launcher" in message   # a named exit, not a bare busy

    def test_arrhenius_sweep_on_a_simulated_rig_takes_no_lock(
        self, hosted_tab, host, monkeypatch, tmp_path
    ):
        """A mock run holding the lock turns a dry run into an outage for a real
        one. The exemption is the executor's own predicate, not a second notion.
        """
        import softae.core.run_lock as rl

        monkeypatch.setattr(rl, "DEFAULT_SCOPE", tmp_path)
        sweep = _ObservingSweep(host._rig_activity)
        _run_off_thread(hosted_tab._run_sweep_thread, sweep)

        assert sweep.ran
        assert sweep.claim_during_run is not None    # the claim is not exempt
        assert sweep.lock_during_run is None
        assert not (tmp_path / rl.LOCK_FILENAME).exists()


# ── Export paths: under the store, never under the CWD ──────────────────────


class _StubStore:
    """Just the two accessors the tab asks a DataStore for."""

    def __init__(self, root: Path) -> None:
        self.project_dir = root

    def run_dir(self, run_id: str) -> Path:
        return self.project_dir / "runs" / run_id


def _elsewhere(monkeypatch, tmp_path) -> Path:
    """Move the process somewhere that is neither the repo nor the store.

    Driving from a foreign CWD is the whole point: the defect was a relative
    ``Path("softae_data")``, which is invisible when the test runs from the
    repo root.
    """
    cwd = tmp_path / "launched_from_here"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


class TestExportPaths:
    def test_arrhenius_images_dir_is_absolute_with_no_store_or_run_id(
        self, tab, monkeypatch, tmp_path
    ):
        """The worst case — no store, no run id — is still not CWD-relative."""
        cwd = _elsewhere(monkeypatch, tmp_path)
        images_dir = tab._images_dir(None)

        assert images_dir.is_absolute()
        assert cwd not in images_dir.parents
        assert images_dir.parts[-3:] == ("runs", "unknown", "images")

    def test_arrhenius_images_dir_is_the_same_from_two_different_cwds(
        self, tab, monkeypatch, tmp_path
    ):
        """A path that moves with the launch directory is the defect itself.

        Resolved **in each directory**, deliberately: a relative result compares
        equal to itself no matter where it was produced, so comparing the raw
        objects is a test that cannot fail.
        """
        _elsewhere(monkeypatch, tmp_path)
        first = tab._images_dir(None).resolve()
        second_cwd = tmp_path / "and_now_from_here"
        second_cwd.mkdir()
        monkeypatch.chdir(second_cwd)

        assert tab._images_dir(None).resolve() == first

    def test_arrhenius_images_dir_defers_to_the_store_run_dir(
        self, tab, monkeypatch, tmp_path
    ):
        """``DataStore.run_dir`` owns the layout; the tab does not restate it."""
        _elsewhere(monkeypatch, tmp_path)
        store = _StubStore(tmp_path / "store")
        tab._data_store = store

        assert tab._images_dir("R1") == store.run_dir("R1") / "images"

    def test_arrhenius_images_dir_without_a_run_id_stays_under_the_store(
        self, tab, monkeypatch, tmp_path
    ):
        """A store with no run id must not fall back out of the store."""
        _elsewhere(monkeypatch, tmp_path)
        tab._data_store = _StubStore(tmp_path / "store")

        assert (tab._images_dir(None)
                == tmp_path / "store" / "runs" / "unknown" / "images")

    def test_arrhenius_export_plot_writes_under_the_store_not_the_cwd(
        self, tab, monkeypatch, tmp_path
    ):
        """End to end, from a foreign CWD. A stray tree in the launch directory
        is exactly what was found in the repo root."""
        cwd = _elsewhere(monkeypatch, tmp_path)
        tab._data_store = _StubStore(tmp_path / "store")
        tab._run_id = "R1"

        tab._on_export_plot()
        out = tmp_path / "store" / "runs" / "R1" / "images" / "arrhenius_R1.png"
        deadline = time.time() + 10.0
        while not out.exists() and time.time() < deadline:
            time.sleep(0.05)

        assert out.exists()
        assert list(cwd.iterdir()) == []

    def test_arrhenius_export_plot_with_no_store_writes_under_the_config_root(
        self, tab, monkeypatch, tmp_path
    ):
        """The exact shape of the stray artifact: no store, no run id.

        This is the path that produced a ``softae_data/`` tree in the repo root.
        The configured project dir is redirected so the assertion is about
        *which root* the export chose, not about the operator's real store.
        """
        import softae.config.loader as loader

        cwd = _elsewhere(monkeypatch, tmp_path)
        root = tmp_path / "configured_store"
        monkeypatch.setattr(loader, "data_project_dir", lambda: str(root))
        tab._data_store = None
        tab._run_id = None

        tab._on_export_plot()
        out = root / "runs" / "unknown" / "images" / "arrhenius_plot.png"
        deadline = time.time() + 10.0
        while not out.exists() and time.time() < deadline:
            time.sleep(0.05)

        assert out.exists()
        assert list(cwd.iterdir()) == []

    def test_arrhenius_3d_export_writes_under_the_store_not_the_cwd(
        self, tab, monkeypatch, tmp_path
    ):
        """The RH-3D fallback carried the same relative shape as the main one."""
        cwd = _elsewhere(monkeypatch, tmp_path)
        tab._data_store = _StubStore(tmp_path / "store")
        tab._run_id = "R1"
        rh_results = {
            30.0: [SimpleNamespace(channel=1, temperatures_C=[25.0, 45.0],
                                   conductivities=[1e-5, 2e-5])],
        }

        tab._save_3d_plots_to_images(rh_results)

        out = tmp_path / "store" / "runs" / "R1" / "images" / "rh_3d_ch1_R1.png"
        assert out.exists()
        assert list(cwd.iterdir()) == []
