"""Tests for the dispenser-head verification dialog and start-gate helper."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets import head_check_dialog as hcd
from softae.gui.widgets.head_check_dialog import (
    HeadState,
    ask_head_state,
    register_head_state,
    verify_head_before_run,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager():
    return create_mock_manager(config={})


def _patch_click(monkeypatch, text: str | None) -> None:
    """Make the modal return without showing; click the button labelled ``text``.

    ``text=None`` simulates a dismissal (no button clicked).  Selecting by label
    avoids depending on Qt's role-based ``buttons()`` ordering.
    """
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

    def _clicked(self):
        if text is None:
            return None
        for b in self.buttons():
            if b.text().replace("&", "") == text:
                return b
        return None

    monkeypatch.setattr(QMessageBox, "clickedButton", _clicked)


# ── Driver contract (registration is pure state, no motion) ────────────────


class TestDriverHeadState:
    def test_set_and_get_head_state(self, manager):
        syr = manager.get("syringe")
        syr.set_head_state(False)
        assert syr.is_head_up() is False
        syr.set_head_state(True)
        assert syr.is_head_up() is True

    def test_set_head_state_does_not_move(self, manager):
        """Registration must not call head_flip (no physical motion)."""
        syr = manager.get("syringe")
        flips = []
        orig = syr.head_flip
        syr.head_flip = lambda *a, **k: (flips.append(1), orig(*a, **k))  # type: ignore[assignment]
        syr.set_head_state(False)
        syr.set_head_state(True)
        assert flips == []


# ── ask_head_state button mapping ──────────────────────────────────────────


class TestAskHeadState:
    def test_raised(self, qapp, monkeypatch):
        _patch_click(monkeypatch, "Raised")
        assert ask_head_state(None) is HeadState.RAISED

    def test_lowered(self, qapp, monkeypatch):
        _patch_click(monkeypatch, "Lowered")
        assert ask_head_state(None) is HeadState.LOWERED

    def test_cancel_button(self, qapp, monkeypatch):
        _patch_click(monkeypatch, "Cancel")
        assert ask_head_state(None) is HeadState.CANCELLED

    def test_dismissed(self, qapp, monkeypatch):
        _patch_click(monkeypatch, None)
        assert ask_head_state(None) is HeadState.CANCELLED


# ── register_head_state ────────────────────────────────────────────────────


class TestRegisterHeadState:
    def test_register_raised(self, manager):
        manager.get("syringe").set_head_state(False)
        assert register_head_state(manager, HeadState.RAISED) is True
        assert manager.get("syringe").is_head_up() is True

    def test_register_lowered(self, manager):
        assert register_head_state(manager, HeadState.LOWERED) is True
        assert manager.get("syringe").is_head_up() is False

    def test_register_cancelled_is_noop(self, manager):
        manager.get("syringe").set_head_state(True)
        assert register_head_state(manager, HeadState.CANCELLED) is False
        assert manager.get("syringe").is_head_up() is True


# ── verify_head_before_run (start-gate policy) ─────────────────────────────


class TestVerifyBeforeRun:
    def test_cancel_aborts(self, qapp, manager, monkeypatch):
        monkeypatch.setattr(hcd, "ask_head_state", lambda *a, **k: HeadState.CANCELLED)
        assert verify_head_before_run(None, manager) is False

    def test_raised_registers_and_proceeds(self, qapp, manager, monkeypatch):
        manager.get("syringe").set_head_state(False)
        monkeypatch.setattr(hcd, "ask_head_state", lambda *a, **k: HeadState.RAISED)
        assert verify_head_before_run(None, manager) is True
        assert manager.get("syringe").is_head_up() is True

    def test_lowered_registers_then_auto_retracts(self, qapp, manager, monkeypatch):
        # Reported Lowered → belief set False → safety retract flips it back up.
        monkeypatch.setattr(hcd, "ask_head_state", lambda *a, **k: HeadState.LOWERED)
        assert verify_head_before_run(None, manager) is True
        assert manager.get("syringe").is_head_up() is True  # retracted for safety

    def test_lowered_registration_precedes_retract(self, qapp, manager, monkeypatch):
        """head_retract must actually fire — proving belief was set to down first.

        If registration were skipped, the default (up) belief would make
        head_retract a no-op and the head would never physically retract.
        """
        syr = manager.get("syringe")
        syr.set_head_state(True)  # stale "up" belief
        flips = []
        orig = syr.head_flip
        syr.head_flip = lambda *a, **k: (flips.append(1), orig(*a, **k))  # type: ignore[assignment]
        monkeypatch.setattr(hcd, "ask_head_state", lambda *a, **k: HeadState.LOWERED)
        verify_head_before_run(None, manager)
        assert flips == [1]  # exactly one flip: the safety retract

    def test_retract_failure_aborts(self, qapp, manager, monkeypatch):
        monkeypatch.setattr(hcd, "ask_head_state", lambda *a, **k: HeadState.LOWERED)
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)

        def boom():
            raise RuntimeError("pneumatics offline")

        manager.get("syringe").head_retract = boom  # type: ignore[assignment]
        assert verify_head_before_run(None, manager) is False
