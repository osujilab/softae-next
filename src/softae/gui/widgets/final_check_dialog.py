"""The Final-Check digest as a modal, for the launch that cannot be prompted.

:func:`~softae.gui.campaign_launch.campaign_run_argv` passes ``--yes`` and gives
the child ``DEVNULL`` stdin, so the CLI's own digest prompt is auto-approved by
construction on the GUI path. This dialog is where that path asks instead.

**A block is not overridable here**, for the reason
:func:`~softae.core.final_check.confirm_final_check` gives at the prompt: a block
is the file and the bench contradicting each other, and a click is not evidence
the contradiction was resolved. The Proceed button is disabled and says why —
and :func:`show_final_check_dialog` refuses a blocked digest again on the way
out, so a caller that reaches the dialog by another route gets the same answer.

The pane is a read-only monospace copy of
:func:`~softae.core.final_check.render_final_check` rather than a re-layout: one
renderer serves the terminal, the log and this window, so the three surfaces
cannot come to say different things about one launch.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Wrap width handed to the renderer. Wider than the terminal's 80 because the
#: pane is sized here rather than by somebody's console.
PANE_WIDTH = 88


def blocking_reason(fc: Any) -> str:
    """Why Proceed is dead — the fix is a changed spec or bench, not a click."""
    return (
        f"{fc.n_block} blocking finding(s): the campaign file and the bench "
        f"record contradict each other. Answering here cannot resolve that — "
        f"change the spec or the bench, then launch again."
    )


class FinalCheckDialog(QDialog):
    """The rendered digest, read-only, with Proceed disabled by any block."""

    def __init__(self, parent: QWidget | None, fc: Any, *,
                 width: int = PANE_WIDTH) -> None:
        super().__init__(parent)
        from softae.core.final_check import render_final_check

        self._fc = fc
        self.setWindowTitle(f"Final check — {getattr(fc, 'title', 'campaign')}")
        self.setMinimumSize(820, 620)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Read this before the rig changes hands. The campaign runs in its "
            "own process and will not ask again."))

        self._pane = QPlainTextEdit(render_final_check(fc, width=width))
        self._pane.setReadOnly(True)
        self._pane.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._pane.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._pane, 1)

        self._lbl_block = QLabel(blocking_reason(fc))
        self._lbl_block.setWordWrap(True)
        self._lbl_block.setStyleSheet("color: #c0392b; font-weight: bold;")
        self._lbl_block.setVisible(bool(fc.has_block))
        layout.addWidget(self._lbl_block)

        buttons = QDialogButtonBox()
        self._btn_proceed: QPushButton = buttons.addButton(
            "Proceed", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self._btn_proceed.setEnabled(not fc.has_block)
        if fc.has_block:
            self._btn_proceed.setToolTip(blocking_reason(fc))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def show_final_check_dialog(parent: QWidget | None, fc: Any) -> bool:
    """Whether to launch: the operator accepted *and* nothing blocks."""
    dialog = FinalCheckDialog(parent, fc)
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    return accepted and not fc.has_block
