"""The autonomous run harness is surface-independent (P2.4).

Live BO must be *one instance* of an autonomous run, not the owner of the
machinery. These tests drive the harness through a minimal host that has no BO
parts at all — if any of them start needing a parameter space, an optimizer, or a
convergence plot, the harness has leaked back into being BO-specific.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from softae.core.autonomous_loop import BoardCheck, BoardDecision
from softae.core.run_lock import RunLock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs._autonomous_run import AutonomousRunMixin


def _foreign_lock(**over) -> RunLock:
    base = dict(pid=4242, what="campaign:phase_map:20260817T090000Z_phase_map",
                started_at="2026-08-17T09:00:00+00:00", host="another-host",
                log_path=r"C:\proj\runs\20260817T090000Z_phase_map")
    base.update(over)
    return RunLock(**base)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class _BareHost(AutonomousRunMixin, QWidget):
    """A surface with none of the Bayesian apparatus — the general façade shape."""

    _sig_log = Signal(str)
    _sig_done = Signal(bool, str)

    def __init__(self, manager, data_store=None):
        super().__init__()
        self._manager = manager
        self._data_store = data_store
        self.logs: list[str] = []
        self.done: list[tuple[bool, str]] = []
        self._sig_log.connect(self.logs.append)
        self._sig_done.connect(lambda ok, msg: self.done.append((ok, msg)))
        self._init_autonomous_run()


class _PanelHost(_BareHost):
    """A surface that *does* carry a config panel, as the BO tabs do."""

    def _panel_state(self) -> dict:
        return {"name": "phase_map", "budget": 8}


@pytest.fixture
def host(qapp):
    h = _BareHost(create_mock_manager(config={}))
    yield h
    h.deleteLater()


@pytest.fixture
def panel_host(qapp, tmp_path):
    """A panel-carrying host whose files land in ``tmp_path``, never the data root."""
    h = _PanelHost(create_mock_manager(config={}),
                   data_store=SimpleNamespace(project_dir=tmp_path))
    yield h
    h.deleteLater()


def test_a_non_bo_surface_can_host_the_whole_harness(host):
    """The point of P2.4: no BO apparatus required to drive the rig."""
    for name in (
        "_verify_head_position", "_board_exchange_gate", "_board_check_gate",
        "_refuse_if_rig_busy", "_preflight_overflow_ok",
        "_hand_over_to_a_detached_campaign", "_release_board_gates",
    ):
        assert hasattr(host, name), name


def test_the_harness_no_longer_runs_a_campaign_in_this_process(host):
    """Step J's whole point, pinned introspectively.

    ``_execute_campaign`` drove the loop on a daemon thread over this process's
    own instrument sessions, which cost the run its lock identity and put two
    processes on one serial bus. Its absence is what the handover replaced it
    with; a later "simplification" that restores it would undo the step, so the
    absence is asserted rather than merely intended. Same for the OS-shutdown
    block, which hung on *this* window's HWND and so protected nothing once the
    campaign stopped living here.
    """
    for gone in ("_execute_campaign", "_acquire_shutdown_block",
                 "_release_shutdown_block", "_rig_run"):
        assert not hasattr(host, gone), gone


class TestBoardGates:
    def test_exchange_gate_returns_the_operator_decision(self, host, monkeypatch):
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        assert host._board_exchange_gate(2) is BoardDecision.PROCEED

    def test_exchange_gate_declining_cancels(self, host, monkeypatch):
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
        assert host._board_exchange_gate(2) is BoardDecision.CANCEL

    def test_unanswered_exchange_prompt_cancels_rather_than_hanging(
        self, host, monkeypatch
    ):
        """A modal nobody answers must not hold a multi-day campaign open."""
        monkeypatch.setattr(
            "softae.gui.tabs._autonomous_run.DEFAULT_GATE_TIMEOUT_S", 0.05)
        # No slot fires, so the event is never set — the wait must bound itself.
        host._sig_board_prompt.disconnect()

        assert host._board_exchange_gate(2) is BoardDecision.CANCEL
        assert any("unanswered" in m for m in host.logs)

    def test_unanswered_freshness_prompt_cancels(self, host, monkeypatch):
        """Timing out must never be read as 'yes, the board is fresh'."""
        monkeypatch.setattr(
            "softae.gui.tabs._autonomous_run.DEFAULT_GATE_TIMEOUT_S", 0.05)
        host._sig_board_check.disconnect()

        assert host._board_check_gate(1, {1, 2}) is BoardCheck.CANCEL

    def test_abort_releases_both_gates_as_cancel(self, host):
        """An abort must never be interpreted as consent to keep casting."""
        host._release_board_gates()
        assert host._board_decision is BoardDecision.CANCEL
        assert host._board_check_decision is BoardCheck.CANCEL
        assert host._board_event.is_set() and host._board_check_event.is_set()

    def test_a_blocked_worker_is_freed_by_the_gui_answer(self, host, monkeypatch):
        """The gate really does marshal across threads, not just return inline."""
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
        result: list[BoardDecision] = []

        def worker():
            result.append(host._board_exchange_gate(3))

        t = threading.Thread(target=worker)
        t.start()
        for _ in range(200):            # pump the GUI thread so the slot runs
            QApplication.processEvents()
            if not t.is_alive():
                break
        t.join(timeout=5)
        assert result == [BoardDecision.PROCEED]


class TestSingleOccupancy:
    """One campaign owns the rig, and being refused costs nothing (S5.I).

    The refusal is outright — never a queue, never a takeover offer — and its
    words are the CLI's, so the two surfaces cannot come to say different things
    about one lock file.
    """

    @staticmethod
    def _hold_rig(monkeypatch, lock):
        """Install a foreign holder and make the rig read as real hardware.

        Both halves are needed: the mock manager reads as *simulated*, and a
        simulated rig is deliberately not refused — the same exemption
        ``softae-campaign run`` grants a ``--mock`` run.
        """
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *a: lock)
        monkeypatch.setattr("softae.core.run_lock.rig_is_simulated", lambda m: False)

    @staticmethod
    def _capture_dialog(monkeypatch) -> list[str]:
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda parent, title, text, *a, **k: shown.append(text)))
        return shown

    def test_refuse_if_rig_busy_with_a_free_rig_permits_the_launch(
        self, host, monkeypatch
    ):
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *a: None)
        assert host._refuse_if_rig_busy(None) is False

    def test_refuse_if_rig_busy_with_a_foreign_holder_refuses(
        self, panel_host, monkeypatch
    ):
        self._hold_rig(monkeypatch, _foreign_lock())
        self._capture_dialog(monkeypatch)
        assert panel_host._refuse_if_rig_busy(None) is True

    def test_refuse_if_rig_busy_on_a_simulated_rig_permits_the_launch(
        self, panel_host, monkeypatch
    ):
        """Parity with the CLI: a run that claims nothing is not refused."""
        monkeypatch.setattr(
            "softae.core.run_lock.foreign_run_lock", lambda *a: _foreign_lock())
        assert panel_host._refuse_if_rig_busy(None) is False

    def test_refuse_if_rig_busy_reuses_the_cli_refusal_wording(
        self, panel_host, monkeypatch
    ):
        """One sentence, one place. A second wording is a second policy."""
        from softae.core.run_lock import busy_rig_message

        lock = _foreign_lock()
        self._hold_rig(monkeypatch, lock)
        shown = self._capture_dialog(monkeypatch)
        panel_host._refuse_if_rig_busy(None)
        assert busy_rig_message(lock, action="This campaign") in shown[0]

    def test_refuse_if_rig_busy_offers_no_takeover_or_queue(
        self, panel_host, monkeypatch
    ):
        """Taking the rig stays a separate, deliberate act — never a Run button."""
        self._hold_rig(monkeypatch, _foreign_lock())
        shown = self._capture_dialog(monkeypatch)
        broken: list[int] = []
        monkeypatch.setattr("softae.core.run_lock.break_run_lock",
                            lambda *a, **k: broken.append(1))
        panel_host._refuse_if_rig_busy(None)
        assert broken == []
        assert "queue" not in shown[0].lower()

    def test_refuse_if_rig_busy_writes_the_panel_state(self, panel_host, monkeypatch):
        self._hold_rig(monkeypatch, _foreign_lock())
        shown = self._capture_dialog(monkeypatch)
        panel_host._refuse_if_rig_busy(None)

        written = sorted((panel_host._project_dir() / "rejected").glob("*.json"))
        assert len(written) == 1
        assert str(written[0]) in shown[0]

    def test_refuse_if_rig_busy_without_a_panel_state_still_refuses(
        self, host, monkeypatch, tmp_path
    ):
        """A surface with no config panel is still refused, and still says why."""
        host._data_store = SimpleNamespace(project_dir=tmp_path)
        self._hold_rig(monkeypatch, _foreign_lock())
        shown = self._capture_dialog(monkeypatch)
        assert host._refuse_if_rig_busy(None) is True
        assert "Nothing was started" in shown[0]

    def test_refuse_if_rig_busy_logs_the_holder_to_the_campaign_log(
        self, panel_host, monkeypatch
    ):
        self._hold_rig(monkeypatch, _foreign_lock())
        self._capture_dialog(monkeypatch)
        panel_host._refuse_if_rig_busy(None)
        assert any("4242" in line for line in panel_host.logs)


class TestPreflight:
    def test_a_scan_failure_never_blocks_the_run(self, host, monkeypatch):
        """Advisory only — an unavailable scan is not evidence of a problem."""
        import softae.core.autonomous_wiring as wiring

        def boom(spec):
            raise RuntimeError("scan unavailable")

        monkeypatch.setattr(wiring, "preflight_overflow", boom)
        assert host._preflight_overflow_ok(object()) is True


class TestHandover:
    """The campaign is prepared here and *run somewhere else* (S5.J).

    These replace ``TestExecution`` and ``TestShutdownBlock``, which asserted a
    contract that no longer exists: the harness ran the loop in this process on a
    daemon thread and blocked OS shutdown on this window's HWND. Both were
    written around ``_execute_campaign``, so they are rewritten rather than
    deleted — the behaviour they described moved, it did not stop mattering.
    """

    @staticmethod
    def _spec():
        from softae.core.autonomous_wiring import CampaignSpec

        return CampaignSpec(
            name="handover", channels=(1,),
            parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
            vol_params=("vol_p0",), pump_ids=(0,), budget=2)

    @staticmethod
    def _arm(host, monkeypatch, *, released=None, spawned=None, lock=None):
        """Drive the handover synchronously and let nothing reach the rig.

        The scheduler is replaced because the real one needs the qasync loop the
        GUI runs on; the lock reader and the spawn are replaced because this test
        must not read the operator's live rig lock or start a process.
        """
        import asyncio

        import softae.gui.campaign_launch as launch

        released = [] if released is None else released
        spawned = [] if spawned is None else spawned
        monkeypatch.setattr(
            host, "_schedule",
            lambda coro, done: done(asyncio.run(coro) is None))
        monkeypatch.setattr("softae.core.rig_session.release_rig_session",
                            lambda *a, **k: released.append(True) or True)
        monkeypatch.setattr("softae.core.run_lock.read_run_lock",
                            lambda *a, **k: lock)
        monkeypatch.setattr(
            launch, "spawn_campaign",
            lambda argv, *, log_file: spawned.append((argv, log_file)) or 31337)
        monkeypatch.setattr(
            QMessageBox, "information",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

    def test_handover_writes_a_spec_file_and_spawns_a_child(
        self, panel_host, monkeypatch
    ):
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)

        assert panel_host._hand_over_to_a_detached_campaign(self._spec()) is True

        written = sorted((panel_host._project_dir() / "launched").glob("*.toml"))
        assert len(written) == 1
        assert spawned and str(written[0]) in spawned[0][0]

    def test_handover_argv_is_the_cli_command_a_terminal_would_run(
        self, panel_host, monkeypatch
    ):
        """One entry point. A GUI-only launch path is one that can drift."""
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)
        panel_host._hand_over_to_a_detached_campaign(self._spec())

        argv = spawned[0][0]
        assert argv[:3] == ["-m", "softae.tools.campaign", "run"]
        assert "--project" in argv
        # The child's stdin is DEVNULL, so an un-pre-approved prompt is a refusal
        # and an unstated head position is a refusal to start at all.
        assert "--yes" in argv
        assert "--head-up" in argv or "--head-down" in argv

    def test_handover_releases_the_rig_before_it_spawns(
        self, panel_host, monkeypatch
    ):
        """Disconnect *and* give the claim back — the child acquires its own."""
        order: list[str] = []
        import asyncio

        import softae.gui.campaign_launch as launch

        async def _disconnect():
            order.append("disconnect")

        monkeypatch.setattr(panel_host._manager, "disconnect_all", _disconnect)
        monkeypatch.setattr(
            panel_host, "_schedule",
            lambda coro, done: done(asyncio.run(coro) is None))
        monkeypatch.setattr("softae.core.rig_session.release_rig_session",
                            lambda *a, **k: order.append("release") or True)
        monkeypatch.setattr("softae.core.run_lock.read_run_lock",
                            lambda *a, **k: None)
        monkeypatch.setattr(launch, "spawn_campaign",
                            lambda argv, *, log_file: order.append("spawn") or 7)
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        panel_host._hand_over_to_a_detached_campaign(self._spec())
        assert order == ["disconnect", "release", "spawn"]

    def test_handover_refuses_to_spawn_while_an_instrument_is_still_connected(
        self, panel_host, monkeypatch
    ):
        """Two processes on one set of ports is the collision the lock prevents."""
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)
        monkeypatch.setattr(
            panel_host._manager, "list_instruments",
            lambda: [{"name": "syringe", "connected": True}])
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "critical",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))

        panel_host._hand_over_to_a_detached_campaign(self._spec())
        assert spawned == []
        assert "NOT started" in shown[0]

    def test_handover_refuses_to_spawn_while_this_process_still_claims_the_rig(
        self, panel_host, monkeypatch
    ):
        """A surviving claim means the child's own acquire would be refused."""
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned,
                  lock=_foreign_lock(pid=1, what="gui:desktop"))
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "critical",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))

        panel_host._hand_over_to_a_detached_campaign(self._spec())
        assert spawned == []
        assert "still claimed" in shown[0]

    def test_handover_asks_before_disconnecting_and_a_refusal_starts_nothing(
        self, panel_host, monkeypatch
    ):
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)
        monkeypatch.setattr(
            panel_host._manager, "list_instruments",
            lambda: [{"name": "syringe", "connected": True}])
        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))

        assert panel_host._hand_over_to_a_detached_campaign(self._spec()) is False
        assert spawned == []

    def test_a_spec_a_file_cannot_carry_is_refused_rather_than_launched(
        self, panel_host, monkeypatch
    ):
        """The cost of shelling out, stated where the operator meets it.

        A composition campaign written to TOML reloads with no
        ``general_formulation`` *and* no ``vol_params``, so the child would search
        the composition axes as raw µL volumes and raise nothing. The launch is
        refused, and the panel state — which is lossless — is written.
        """
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))

        spec = self._spec()
        object.__setattr__(spec, "prior_mean", lambda params: 0.0)

        assert panel_host._hand_over_to_a_detached_campaign(spec) is False
        assert spawned == []
        assert "prior_mean" in shown[0]
        assert list((panel_host._project_dir() / "launched").glob("*")) == []
        assert len(list((panel_host._project_dir() / "rejected").glob("*.json"))) == 1

    def test_the_unwritable_refusal_does_not_tell_the_operator_to_retry(
        self, panel_host, monkeypatch
    ):
        """The rig-busy paragraph would; this refusal is not about the rig.

        ``PreservedLaunch.describe`` says "press Run again once the rig is free"
        and "relaunching it through this tab is the only way" — both written for
        a *busy* rig, and both false here. Pressing Run will refuse forever, and
        the tab runs nothing.
        """
        self._arm(panel_host, monkeypatch)
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))

        spec = self._spec()
        object.__setattr__(spec, "prior_mean", lambda params: 0.0)
        panel_host._hand_over_to_a_detached_campaign(spec)

        assert "once the rig is free" not in shown[0]
        assert "only way to run the campaign" not in shown[0]
        assert "Waiting will not help" in shown[0]
        # …but the setup is still preserved and the file is still named.
        saved = next((panel_host._project_dir() / "rejected").glob("*.json"))
        assert str(saved) in shown[0]

    def test_an_unreadable_head_belief_refuses_rather_than_guessing(
        self, panel_host, monkeypatch
    ):
        """The loop drives the head conditionally; a guess costs one wrong flip."""
        spawned: list = []
        self._arm(panel_host, monkeypatch, spawned=spawned)
        monkeypatch.setattr(panel_host, "_head_state_after_gate", lambda: None)
        monkeypatch.setattr(QMessageBox, "critical",
                            staticmethod(lambda *a, **k: None))

        assert panel_host._hand_over_to_a_detached_campaign(self._spec()) is False
        assert spawned == []

    def test_a_failed_spawn_says_the_instruments_were_released(
        self, panel_host, monkeypatch
    ):
        """The one state the operator cannot see for themselves: nothing runs,
        and the ports this window used to hold are now open to nobody."""
        import softae.gui.campaign_launch as launch

        self._arm(panel_host, monkeypatch)

        def boom(argv, *, log_file):
            raise OSError("no such interpreter")

        monkeypatch.setattr(launch, "spawn_campaign", boom)
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "critical",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))

        panel_host._hand_over_to_a_detached_campaign(self._spec())
        assert "released" in shown[0] and "Init tab" in shown[0]
