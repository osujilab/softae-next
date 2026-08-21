"""Table widgets with spreadsheet-style clipboard support.

* :class:`CopyableTableWidget` — Ctrl+C copies the selected rectangle to
  Excel-compatible TSV (tabs between columns, newlines between rows).  Read-only,
  so it suits results/export tables.
* :class:`PasteableTableWidget` — adds Ctrl+V, writing clipboard TSV (from
  another region *or* an external spreadsheet) into editable cells only, starting
  at the current cell.  Suits editable input tables (the formulation matrix).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QTableWidget


class CopyableTableWidget(QTableWidget):
    """A ``QTableWidget`` whose selection can be copied (Ctrl+C) as TSV.

    Copies the bounding rectangle of the current selection so the block can be
    pasted straight into a spreadsheet.  Cells with no item serialise as empty.

    Optional Shift-click range toggling: set :attr:`checkable_column` to the
    index of a column of checkable items and a Shift-click fills every checkbox
    between the last plainly-clicked row (the anchor) and the clicked row with
    the anchor's check state.  :attr:`checkRangeToggled` fires once per range so
    hosts can refresh in a single pass.
    """

    #: Emitted after a Shift-click fills a range of checkboxes (batched refresh).
    checkRangeToggled = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: Column of checkable items to enable Shift-click range toggling on.
        self.checkable_column: int | None = None
        self._check_anchor_row: int | None = None
        # Any row insert/remove invalidates the anchor — a repopulated or edited
        # table shouldn't fill a range from a row that no longer means the same.
        self.model().rowsInserted.connect(self._reset_check_anchor)
        self.model().rowsRemoved.connect(self._reset_check_anchor)

    def _reset_check_anchor(self, *_args) -> None:
        self._check_anchor_row = None

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt override name
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override name
        col = self.checkable_column
        if col is not None and event.button() == Qt.MouseButton.LeftButton:
            idx = self.indexAt(event.position().toPoint())
            if idx.isValid() and idx.column() == col:
                row = idx.row()
                if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                        and self._check_anchor_row is not None):
                    self._apply_check_range(self._check_anchor_row, row)
                    event.accept()
                    return
                # Plain click on a checkbox: it becomes the new range anchor.
                self._check_anchor_row = row
        super().mousePressEvent(event)

    def _apply_check_range(self, anchor: int, row: int) -> None:
        col = self.checkable_column
        if col is None:
            return
        anchor_item = self.item(anchor, col)
        if anchor_item is None:
            return
        state = anchor_item.checkState()
        lo, hi = sorted((anchor, row))
        self.blockSignals(True)
        try:
            for r in range(lo, hi + 1):
                item = self.item(r, col)
                if item is not None:
                    item.setCheckState(state)
        finally:
            self.blockSignals(False)
        self.checkRangeToggled.emit()

    def _copy_selection(self) -> None:
        ranges = self.selectedRanges()
        if not ranges:
            return
        top = min(r.topRow() for r in ranges)
        bottom = max(r.bottomRow() for r in ranges)
        left = min(r.leftColumn() for r in ranges)
        right = max(r.rightColumn() for r in ranges)
        lines = []
        for row in range(top, bottom + 1):
            cells = []
            for col in range(left, right + 1):
                item = self.item(row, col)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))


class PasteableTableWidget(CopyableTableWidget):
    """A :class:`CopyableTableWidget` that also pastes TSV (Ctrl+V).

    Paste writes clipboard TSV starting at the current cell.  Only editable cells
    are written; read-only cells and widget (checkbox) cells are stepped over so
    column alignment is preserved, and the paste is clamped to the existing rows
    and columns (it never grows the table).

    The write loop runs with the widget's signals blocked, so ``itemChanged`` does
    *not* fire per cell.  :attr:`pasteCompleted` is the replacement: it fires once
    per paste, after the block is lifted, carrying the rows actually written — so a
    host that derives values from pasted cells can refresh them in a single pass
    instead of receiving nothing at all.
    """

    #: Emitted once after a paste, with the list of rows written (batched refresh).
    #:
    #: The list is **empty when nothing was written** — the paste started outside
    #: any editable column — because a paste that does nothing is an event the host
    #: needs to report, not an absence of one.  Not emitted for an empty clipboard:
    #: nothing was attempted there.
    pasteCompleted = Signal(list)

    def keyPressEvent(self, event) -> None:  # noqa: N802 — Qt override name
        if event.matches(QKeySequence.StandardKey.Paste):
            self._paste_selection()
            event.accept()
            return
        super().keyPressEvent(event)  # CopyableTableWidget handles Copy

    def _paste_selection(self) -> None:
        text = QApplication.clipboard().text()
        if not text:
            return
        rows = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        while rows and rows[-1] == "":
            rows.pop()                      # ignore Excel's trailing newline
        start_row = max(0, self.currentRow())
        start_col = max(0, self.currentColumn())
        written: list[int] = []
        self.blockSignals(True)
        try:
            for dr, line in enumerate(rows):
                row = start_row + dr
                if row >= self.rowCount():
                    break                   # never grow the table past its rows
                for dc, value in enumerate(line.split("\t")):
                    col = start_col + dc
                    if col >= self.columnCount():
                        break
                    item = self.item(row, col)
                    if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEditable):
                        continue            # read-only / widget cell → step over
                    item.setText(value.strip())
                    if not written or written[-1] != row:
                        written.append(row)
        finally:
            self.blockSignals(False)
        # After the unblock, never inside it: an emit under blockSignals is exactly
        # the silent no-op this signal exists to prevent.
        self.pasteCompleted.emit(written)
