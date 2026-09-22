"""The GUI's Final-Check modal: what it shows, and what it refuses to let through."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog

from softae.core.final_check import BLOCK, OK, WARN, FinalCheck, Finding, Section
from softae.gui.widgets.final_check_dialog import (
    FinalCheckDialog,
    show_final_check_dialog,
)


def _digest(*severities: str) -> FinalCheck:
    findings = tuple(Finding(sev, f"a {sev} finding about pump 0")
                     for sev in severities)
    return FinalCheck(
        title="rung3b",
        sections=(Section("Stock on the pumps",
                          (("pump 0", "LiCl 1M · 4,000 µL"),), findings),))


def _trap_dialog_loop(monkeypatch, *, result: int) -> list[bool]:
    """Record — rather than enter — the modal loop, and answer it with *result*."""
    entered: list[bool] = []
    monkeypatch.setattr(
        QDialog, "exec",
        lambda self: (entered.append(True), result)[1], raising=False)
    return entered


class TestDialog:
    def test_final_check_dialog_blocking_digest_disables_proceed(self, qapp):
        """A block is not answerable at the prompt, so the prompt must not offer it."""
        dlg = FinalCheckDialog(None, _digest(BLOCK, WARN))

        assert dlg._btn_proceed.isEnabled() is False
        assert dlg._lbl_block.isVisibleTo(dlg)
        assert "change the spec or the bench" in dlg._lbl_block.text()

    def test_final_check_dialog_clean_digest_enables_proceed(self, qapp):
        """The disable is conditional — one that always fired would gate nothing."""
        dlg = FinalCheckDialog(None, _digest(WARN, OK))

        assert dlg._btn_proceed.isEnabled() is True
        assert not dlg._lbl_block.isVisibleTo(dlg)

    def test_final_check_dialog_pane_carries_the_rendered_digest(self, qapp):
        """One renderer for terminal, log and window; a re-layout could disagree."""
        from softae.core.final_check import render_final_check

        digest = _digest(WARN)
        dlg = FinalCheckDialog(None, digest)

        assert dlg._pane.toPlainText() == render_final_check(digest, width=88)
        assert dlg._pane.isReadOnly() is True


class TestShow:
    def test_show_final_check_dialog_accepted_returns_true(self, qapp, monkeypatch):
        entered = _trap_dialog_loop(monkeypatch,
                                    result=int(QDialog.DialogCode.Accepted))

        assert show_final_check_dialog(None, _digest(WARN)) is True
        assert entered == [True]

    def test_show_final_check_dialog_rejected_returns_false(self, qapp, monkeypatch):
        _trap_dialog_loop(monkeypatch, result=int(QDialog.DialogCode.Rejected))

        assert show_final_check_dialog(None, _digest(WARN)) is False

    def test_show_final_check_dialog_blocking_digest_refuses_an_acceptance(
        self, qapp, monkeypatch
    ):
        """Belt and braces: a caller reaching the loop another way gets the same no."""
        _trap_dialog_loop(monkeypatch, result=int(QDialog.DialogCode.Accepted))

        assert show_final_check_dialog(None, _digest(BLOCK)) is False
