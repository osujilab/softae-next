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
        "_acquire_shutdown_block", "_release_shutdown_block",
        "_preflight_overflow_ok", "_execute_campaign", "_release_board_gates",
    ):
        assert hasattr(host, name), name


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


class TestShutdownBlock:
    def test_acquire_and_release_never_raise(self, host):
        """Best-effort: a failure here must not take down a campaign."""
        host._acquire_shutdown_block("c")
        host._release_shutdown_block()

    def test_release_without_acquire_is_a_no_op(self, host):
        host._release_shutdown_block()


class TestExecution:
    def test_execute_reports_the_result_and_releases_the_block(
        self, host, monkeypatch
    ):
        import softae.core.autonomous_wiring as wiring
        from softae.core.autonomous_wiring import CampaignResult

        released: list[bool] = []
        monkeypatch.setattr(
            host, "_release_shutdown_block", lambda: released.append(True))

        async def fake_run(spec, **kw):
            return CampaignResult(
                run_id="r1", best_params={}, best_objective=1.25, n_trials=3,
                final_state="CONVERGED", converged=True, history=[])

        monkeypatch.setattr(wiring, "run_autonomous_campaign", fake_run)

        class _Spec:
            name = "c"

        result = host._execute_campaign(
            _Spec(), on_event=lambda e: None, aborted_exc=RuntimeError)

        assert result is not None and result.n_trials == 3
        assert host.done and host.done[0][0] is True
        assert released == [True]

    def test_a_failing_campaign_reports_failure_not_a_crash(self, host, monkeypatch):
        import softae.core.autonomous_wiring as wiring

        async def boom(spec, **kw):
            raise ValueError("instrument exploded")

        monkeypatch.setattr(wiring, "run_autonomous_campaign", boom)

        class _Spec:
            name = "c"

        assert host._execute_campaign(
            _Spec(), on_event=lambda e: None, aborted_exc=RuntimeError) is None
        assert host.done and host.done[0][0] is False
        assert "exploded" in host.done[0][1]

    def test_abort_is_reported_as_success_not_error(self, host, monkeypatch):
        """An operator-requested stop is not a failure."""
        import softae.core.autonomous_wiring as wiring

        class _Aborted(Exception):
            pass

        async def aborted(spec, **kw):
            raise _Aborted()

        monkeypatch.setattr(wiring, "run_autonomous_campaign", aborted)

        class _Spec:
            name = "c"

        assert host._execute_campaign(
            _Spec(), on_event=lambda e: None, aborted_exc=_Aborted) is None
        assert host.done == [(True, "aborted")]
