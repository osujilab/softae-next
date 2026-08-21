"""Tests for the shared copyable/pasteable table widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
)

from softae.gui.widgets.copyable_table import (
    CopyableTableWidget,
    PasteableTableWidget,
)


def _check_grid(rows, qtbot):
    t = _grid(CopyableTableWidget, rows, 1, qtbot)
    for r in range(rows):
        it = QTableWidgetItem()
        it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        it.setCheckState(Qt.CheckState.Unchecked)
        t.setItem(r, 0, it)
    t.checkable_column = 0
    return t


def _checked(t):
    return [t.item(r, 0).checkState() == Qt.CheckState.Checked
            for r in range(t.rowCount())]


def _grid(cls, rows, cols, qtbot):
    t = cls()
    t.setRowCount(rows)
    t.setColumnCount(cols)
    qtbot.addWidget(t)
    return t


def test_copy_serialises_selected_rectangle_to_tsv(qtbot):
    t = _grid(CopyableTableWidget, 3, 3, qtbot)
    for (r, c), v in {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}.items():
        t.setItem(r, c, QTableWidgetItem(v))
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)
    t._copy_selection()
    assert QApplication.clipboard().text() == "a\tb\nc\td"


def test_copy_empty_cells_serialise_as_blank(qtbot):
    t = _grid(CopyableTableWidget, 2, 2, qtbot)
    t.setItem(0, 0, QTableWidgetItem("x"))
    t.setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), True)
    t._copy_selection()
    assert QApplication.clipboard().text() == "x\t\n\t"


def test_pasteable_is_a_copyable():
    assert issubclass(PasteableTableWidget, CopyableTableWidget)


def test_paste_writes_editable_cells(qtbot):
    t = _grid(PasteableTableWidget, 2, 2, qtbot)
    for r in range(2):
        for c in range(2):
            t.setItem(r, c, QTableWidgetItem("0"))
    QApplication.clipboard().setText("1\t2\n3\t4")
    t.setCurrentCell(0, 0)
    t._paste_selection()
    assert [t.item(r, c).text() for r in range(2) for c in range(2)] == ["1", "2", "3", "4"]


def _ro(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return it


def _mixed_grid(rows, qtbot):
    """3 columns: read-only, editable, read-only — the results-table shape."""
    t = _grid(PasteableTableWidget, rows, 3, qtbot)
    for r in range(rows):
        t.setItem(r, 0, _ro("ro"))
        t.setItem(r, 1, QTableWidgetItem("0"))
        t.setItem(r, 2, _ro("ro"))
    return t


def _paste_at(t, text, row, col):
    QApplication.clipboard().setText(text)
    t.setCurrentCell(row, col)
    emitted = []
    t.pasteCompleted.connect(emitted.append)
    t._paste_selection()
    return emitted


def test_paste_completed_emits_once_with_rows_written(qtbot):
    t = _mixed_grid(4, qtbot)
    emitted = _paste_at(t, "1\n2\n3", 0, 1)
    assert emitted == [[0, 1, 2]]  # one batched emission, not one per cell
    assert [t.item(r, 1).text() for r in range(4)] == ["1", "2", "3", "0"]


def test_paste_completed_emits_empty_list_when_nothing_writable(qtbot):
    """TRAP 2: a cursor outside the editable column writes nothing at all."""
    t = _mixed_grid(4, qtbot)
    emitted = _paste_at(t, "1\n2\n3", 0, 0)  # cursor on the read-only column
    assert emitted == [[]]                   # reported, not silent
    assert [t.item(r, 1).text() for r in range(4)] == ["0"] * 4


def test_paste_completed_not_emitted_for_empty_clipboard(qtbot):
    t = _mixed_grid(2, qtbot)
    emitted = _paste_at(t, "", 0, 1)
    assert emitted == []  # nothing was attempted, so there is nothing to report


def test_paste_completed_fires_after_signals_unblocked(qtbot):
    """TRAP 1: an emit inside blockSignals would be swallowed silently."""
    t = _mixed_grid(2, qtbot)
    seen = []
    t.pasteCompleted.connect(lambda rows: seen.append(
        (t.signalsBlocked(), [t.item(r, 1).text() for r in rows])))
    QApplication.clipboard().setText("7\n8")
    t.setCurrentCell(0, 1)
    t._paste_selection()
    assert seen == [(False, ["7", "8"])]  # receiver sees the written text


def test_paste_steps_over_readonly_cells_preserving_column_alignment(qtbot):
    t = _mixed_grid(2, qtbot)
    QApplication.clipboard().setText("a\tb\tc")  # 3 wide, starting at col 0
    t.setCurrentCell(0, 0)
    t._paste_selection()
    # Only the editable middle column takes a value, and it takes the *middle* one.
    assert [t.item(0, c).text() for c in range(3)] == ["ro", "b", "ro"]


def test_paste_clamps_at_row_count_and_never_grows_table(qtbot):
    t = _mixed_grid(2, qtbot)
    emitted = _paste_at(t, "1\n2\n3\n4\n5", 0, 1)
    assert t.rowCount() == 2
    assert emitted == [[0, 1]]
    assert [t.item(r, 1).text() for r in range(2)] == ["1", "2"]


def test_paste_drops_trailing_excel_newline(qtbot):
    t = _mixed_grid(3, qtbot)
    emitted = _paste_at(t, "1\r\n2\r\n", 0, 1)  # Excel's CRLF + trailing newline
    assert emitted == [[0, 1]]                  # not a third, empty row
    assert t.item(2, 1).text() == "0"


# ── Shift-click range toggling ───────────────────────────────────────────────


def test_shift_range_fills_from_anchor_state(qtbot):
    t = _check_grid(5, qtbot)
    t.item(1, 0).setCheckState(Qt.CheckState.Checked)  # anchor row is checked
    t._check_anchor_row = 1
    emitted = []
    t.checkRangeToggled.connect(lambda: emitted.append(True))

    t._apply_check_range(1, 3)  # extend to row 3

    assert _checked(t) == [False, True, True, True, False]
    assert emitted == [True]  # one batched refresh, not one per row


def test_shift_range_works_upward_and_uses_anchor_state(qtbot):
    t = _check_grid(5, qtbot)
    # Anchor row 3 left unchecked → filling a range clears it.
    for r in range(5):
        t.item(r, 0).setCheckState(Qt.CheckState.Checked)
    t._check_anchor_row = 3
    t.item(3, 0).setCheckState(Qt.CheckState.Unchecked)

    t._apply_check_range(3, 1)  # anchor below the clicked row

    assert _checked(t) == [True, False, False, False, True]


def test_anchor_resets_on_row_insert_and_remove(qtbot):
    t = _check_grid(4, qtbot)
    t._check_anchor_row = 2
    t.insertRow(0)
    assert t._check_anchor_row is None
    t._check_anchor_row = 1
    t.removeRow(0)
    assert t._check_anchor_row is None


def test_range_toggle_disabled_when_no_checkable_column(qtbot):
    t = _grid(CopyableTableWidget, 3, 1, qtbot)  # checkable_column stays None
    # _apply_check_range is a no-op guard when the feature isn't enabled.
    t._apply_check_range(0, 2)  # must not raise
