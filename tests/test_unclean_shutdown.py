"""Surviving a host shutdown: detection at restart, park on exit, OS blocking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from softae.core.alerts import clear_alert_sinks
from softae.core.data_store import DataStore
from softae.gui.shutdown_guard import ShutdownBlocker, block_shutdown, unblock_shutdown
from softae.gui.widgets.unclean_shutdown import check_unclean_shutdown


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_sink_leakage():
    clear_alert_sinks()
    yield
    clear_alert_sinks()


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def _patch_dialog(monkeypatch, click: str | None):
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

    def _clicked(self):
        if click is None:
            return None
        for b in self.buttons():
            if b.text().replace("&", "") == click:
                return b
        return None

    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked)


# ── Detection ──────────────────────────────────────────────────────────────


class TestDetection:
    def test_unfinished_run_is_detected(self, store):
        store.start_run("wf")                       # never finished
        assert len(store.unfinished_runs()) == 1

    def test_finished_run_is_not_flagged(self, store):
        rid = store.start_run("wf")
        store.finish_run(rid, "done")
        assert store.unfinished_runs() == []

    def test_detection_survives_reopen(self, tmp_path: Path):
        """The whole point: the evidence outlives the process that died."""
        with DataStore(tmp_path / "p") as ds:
            ds.start_run("interrupted_wf")
        with DataStore(tmp_path / "p") as ds2:
            assert len(ds2.unfinished_runs()) == 1


# ── Start-up handling ──────────────────────────────────────────────────────


class TestStartupCheck:
    def test_clean_start_does_nothing(self, qapp, store, monkeypatch):
        called = []
        monkeypatch.setattr(QMessageBox, "exec", lambda self: called.append(1) or 0)
        assert check_unclean_shutdown(None, MagicMock(), store) is False
        assert called == []

    def test_no_store_is_a_noop(self, qapp):
        assert check_unclean_shutdown(None, MagicMock(), None) is False

    def test_park_when_operator_accepts(self, qapp, store, monkeypatch):
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Park now")
        mgr = MagicMock()
        for name in ("syringe", "temp_controller", "lamp"):
            mgr.get.return_value.is_connected = True

        assert check_unclean_shutdown(None, mgr, store) is True
        assert mgr.get.return_value.halt_pump.call_count == 3

    def test_the_recovery_park_does_not_move_the_head(self, qapp, store,
                                                      monkeypatch):
        """The sharpest case for the policy, and this dialog already argues it.

        The belief here comes from a session that was *killed*: the dialog's own
        text says the head "may have been left LOWERED over an electrode". A
        conditional flip on that belief is a coin toss, and one face of it drives
        the head down. So the recovery park commands no head motion and the
        operator's own inspection — which the dialog demands — is the sensor.
        """
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Park now")
        mgr = MagicMock()
        mgr.get.return_value.is_connected = True

        assert check_unclean_shutdown(None, mgr, store) is True
        mgr.get.return_value.head_retract.assert_not_called()
        mgr.get.return_value.head_flip.assert_not_called()

    def test_no_park_when_declined(self, qapp, store, monkeypatch):
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Skip")
        mgr = MagicMock()
        assert check_unclean_shutdown(None, mgr, store) is False
        mgr.get.return_value.head_retract.assert_not_called()

    def test_records_a_durable_alert(self, qapp, store, monkeypatch):
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Skip")
        check_unclean_shutdown(None, MagicMock(), store)

        alerts = store.query_alerts()
        assert len(alerts) == 1
        assert alerts[0]["kind"] == "unclean_shutdown"

    def test_alert_reports_the_unknown_head_position(self, qapp, store, monkeypatch):
        """The head does not self-retract, so reporting IS the mitigation.

        Operator decision (2026-07-30): rather than race the OS with a
        best-effort park — which could strand the head mid-travel — an unclean
        stop is accepted as a known state and reported. The durable alert has to
        carry it, because a dialog can be dismissed and forgotten.
        """
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Skip")
        check_unclean_shutdown(None, MagicMock(), store)

        alert = store.query_alerts()[0]
        assert "lowered" in alert["message"]
        assert "electrode" in alert["message"]

    def test_reported_once_not_every_launch(self, qapp, store, monkeypatch):
        """Stale runs are marked, so the warning does not repeat forever."""
        store.start_run("wf")
        _patch_dialog(monkeypatch, "Skip")

        check_unclean_shutdown(None, MagicMock(), store)
        assert store.unfinished_runs() == []          # now marked interrupted

        # A second launch sees nothing to report.
        assert check_unclean_shutdown(None, MagicMock(), store) is False
        assert len(store.query_alerts()) == 1

    def test_query_failure_does_not_block_startup(self, qapp):
        broken = MagicMock()
        broken.unfinished_runs.side_effect = RuntimeError("db locked")
        assert check_unclean_shutdown(None, MagicMock(), broken) is False


# ── What the recovery park is allowed to claim ─────────────────────────────


class TestTheRecoveryParkReport:
    """This path read ``result.ok`` — *nothing raised* — and warned only on a
    refusal. A park at start-up against instruments that are not connected yet
    skips all three, raises nothing, and used to pass in silence, leaving the
    operator of a session that died with the head possibly down believing the
    rig had just been made safe.
    """

    def _accept_and_capture(self, monkeypatch, store, mgr):
        _patch_dialog(monkeypatch, "Park now")
        warned: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warned.append(a[2])))
        store.start_run("wf")
        check_unclean_shutdown(None, mgr, store)
        return warned

    def test_a_recovery_park_that_commanded_something_warns_about_nothing(
        self, qapp, store, monkeypatch
    ):
        mgr = MagicMock()
        mgr.get.return_value.is_connected = True
        assert self._accept_and_capture(monkeypatch, store, mgr) == []

    def test_a_recovery_park_that_commanded_nothing_warns_that_nothing_was_sent(
        self, qapp, store, monkeypatch
    ):
        from softae.core.safe_park import HEADLINE_NOTHING

        mgr = MagicMock()
        mgr.get.return_value.is_connected = False

        warned = self._accept_and_capture(monkeypatch, store, mgr)

        assert warned and HEADLINE_NOTHING in warned[0]
        assert "not connected" in warned[0]      # describe() names each one

    def test_a_recovery_park_that_refused_still_says_partial_stop(
        self, qapp, store, monkeypatch
    ):
        from softae.core.safe_park import HEADLINE_PARTIAL

        mgr = MagicMock()
        mgr.get.return_value.is_connected = True
        mgr.get.return_value.off.side_effect = RuntimeError("lamp: no reply")

        warned = self._accept_and_capture(monkeypatch, store, mgr)

        assert warned and HEADLINE_PARTIAL in warned[0]
        assert "no reply" in warned[0]


# ── OS shutdown blocking ───────────────────────────────────────────────────


class TestShutdownGuard:
    def test_is_a_safe_noop_off_windows(self):
        """Must never raise on any platform, whatever the host is."""
        assert block_shutdown(0, "reason") is False
        assert unblock_shutdown(0) is False

    def test_blocker_is_reference_counted(self, monkeypatch):
        """Overlapping runs must not release each other's block."""
        import softae.gui.shutdown_guard as sg

        events: list[str] = []
        monkeypatch.setattr(sg, "block_shutdown", lambda h, r: events.append("block") or True)
        monkeypatch.setattr(sg, "unblock_shutdown", lambda h: events.append("unblock") or True)

        b = ShutdownBlocker(1234, "running")
        b.acquire()
        b.acquire()                       # second run starts
        assert b.active
        b.release()                       # first finishes — block must persist
        assert b.active
        assert events == ["block"]
        b.release()                       # last one out
        assert not b.active
        assert events == ["block", "unblock"]

    def test_release_without_acquire_is_harmless(self):
        ShutdownBlocker(0, "r").release()
