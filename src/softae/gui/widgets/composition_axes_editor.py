"""Composition **axes** editor: the twin's targets table with a range per row.

The sibling of :class:`~softae.gui.widgets.targets_editor.TargetsEditor`. That one
fixes a composition — "EO:Li = 20" — and the twin solves it. This one gives each
target a **Low/High** pair so a campaign can search it, which is what turns the Live
BO tab from "explore raw pump volumes" into "explore compositions".

Deliberately the same vocabulary, the same A/B dropdowns and the same
:meth:`set_available` contract as the targets editor, so an operator who has set up
the twin already knows how to drive this. A row with ``Low == High`` is *pinned*:
held constant, and kept out of the optimizer's parameter space entirely rather than
handed over as a zero-width dimension.

The table is the view; :mod:`softae.core.composition_axes` is the model, and every
question worth asking (what does the optimizer search? what targets does a
suggestion become? is this set determinate?) is answered there, without Qt.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.core.composition_axes import (
    AXIS_LABELS,
    CompositionAxis,
    axes_parameter_space,
    build_targets_from_axes,
    validate_axes,
)

_TYPES = [AXIS_LABELS["molar_ratio"], AXIS_LABELS["dried_fraction"],
          AXIS_LABELS["concentration"]]
_KIND_FOR_LABEL = {label: kind for kind, label in AXIS_LABELS.items()}
_BASES = ["volume", "mass", "mole"]
_ISSUE_STYLE = "color: #f9a825; font-weight: bold;"  # amber, as in TargetsEditor

# Column indices.
_C_TYPE, _C_A, _C_B, _C_LOW, _C_HIGH, _C_BASIS = range(6)


class CompositionAxesEditor(QWidget):
    """Rows → :class:`CompositionAxis` objects. Emits :attr:`changed` on any edit."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._species: list[str] = []      # dissolved solutes (ratio / concentration)
        self._components: list[str] = []   # deposited components (dried fraction)
        self._n_stocks: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Type", "A", "B", "Low", "High", "Basis"])
        self._table.setToolTip(
            "Each row is a composition target the campaign searches between Low and "
            "High.\n"
            "Molar ratio: A/B species.  Dried fraction: component A on Basis.  "
            "Concentration: species A in mol/L.\n\n"
            "Set Low == High to pin a target instead of searching it — pinned rows "
            "are held constant and are not given to the optimizer.\n"
            "The solver needs one target per stock; the deposition-µL box counts as "
            "one of them."
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(160)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(_C_TYPE, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_C_A, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_C_B, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_C_LOW, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_C_HIGH, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_C_BASIS, QHeaderView.ResizeMode.Interactive)
        self._table.setColumnWidth(_C_TYPE, 120)
        self._table.setColumnWidth(_C_LOW, 70)
        self._table.setColumnWidth(_C_HIGH, 70)
        self._table.setColumnWidth(_C_BASIS, 80)
        self._table.itemChanged.connect(self.changed)
        lay.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add target")
        self._btn_add.clicked.connect(lambda: self.add_axis())
        btn_row.addWidget(self._btn_add)
        self._btn_del = QPushButton("Remove")
        self._btn_del.clicked.connect(self._on_del)
        btn_row.addWidget(self._btn_del)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._lbl_issues = QLabel("")
        self._lbl_issues.setWordWrap(True)
        self._lbl_issues.setStyleSheet(_ISSUE_STYLE)
        lay.addWidget(self._lbl_issues)

        self.changed.connect(self._refresh_issues)

    # -- host wiring ----------------------------------------------------------

    def set_available(
        self, species: list[str], components: list[str], n_stocks: int = 0
    ) -> None:
        """Feed the names from the checked stocks — same contract as TargetsEditor."""
        self._species = list(species)
        self._components = list(components)
        self._n_stocks = int(n_stocks)
        for row in range(self._table.rowCount()):
            self._repopulate_row(row)
        self._refresh_issues()

    # -- editing --------------------------------------------------------------

    def _make_name_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)   # settable programmatically, and typeable
        combo.currentTextChanged.connect(self.changed)
        return combo

    def _fill_combo(self, combo: QComboBox, items: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        combo.addItems(items)
        combo.setCurrentText(current)
        combo.blockSignals(False)

    def _repopulate_row(self, row: int) -> None:
        """Point A/B at the right name list for this row's Type."""
        type_combo = self._table.cellWidget(row, _C_TYPE)
        if not isinstance(type_combo, QComboBox):
            return
        label = type_combo.currentText()
        a_items = (self._components if label == AXIS_LABELS["dried_fraction"]
                   else self._species)
        a_combo = self._table.cellWidget(row, _C_A)
        if isinstance(a_combo, QComboBox):
            self._fill_combo(a_combo, a_items)
        b_enabled = label == AXIS_LABELS["molar_ratio"]
        b_combo = self._table.cellWidget(row, _C_B)
        if isinstance(b_combo, QComboBox):
            self._fill_combo(b_combo, self._species if b_enabled else [])
            b_combo.setEnabled(b_enabled)
        basis_combo = self._table.cellWidget(row, _C_BASIS)
        if isinstance(basis_combo, QComboBox):
            basis_combo.setEnabled(label == AXIS_LABELS["dried_fraction"])

    def _on_type_changed(self) -> None:
        combo = self.sender()
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, _C_TYPE) is combo:
                self._repopulate_row(row)
                break
        self.changed.emit()

    def add_axis(
        self,
        type_text: str | None = None,
        *,
        a: str = "",
        b: str = "",
        low: str = "0",
        high: str = "1",
        basis: str = "volume",
    ) -> int:
        """Append an axis row (optionally pre-filled). Returns the row index."""
        self._table.blockSignals(True)
        row = self._table.rowCount()
        self._table.insertRow(row)
        type_combo = QComboBox()
        type_combo.addItems(_TYPES)
        if type_text in _TYPES:
            type_combo.setCurrentText(type_text)
        type_combo.currentIndexChanged.connect(self._on_type_changed)
        self._table.setCellWidget(row, _C_TYPE, type_combo)
        self._table.setCellWidget(row, _C_A, self._make_name_combo())
        self._table.setCellWidget(row, _C_B, self._make_name_combo())
        self._table.setItem(row, _C_LOW, QTableWidgetItem(low))
        self._table.setItem(row, _C_HIGH, QTableWidgetItem(high))
        basis_combo = QComboBox()
        basis_combo.addItems(_BASES)
        if basis in _BASES:
            basis_combo.setCurrentText(basis)
        basis_combo.currentIndexChanged.connect(self.changed)
        self._table.setCellWidget(row, _C_BASIS, basis_combo)
        self._table.blockSignals(False)

        self._repopulate_row(row)
        self._table.cellWidget(row, _C_A).setCurrentText(a)
        self._table.cellWidget(row, _C_B).setCurrentText(b)
        self.changed.emit()
        return row

    def _on_del(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            row = self._table.rowCount() - 1
        if row >= 0:
            self._table.removeRow(row)
            self.changed.emit()

    def clear(self) -> None:
        self._table.setRowCount(0)
        self.changed.emit()

    # -- read-out -------------------------------------------------------------

    def _combo_text(self, row: int, col: int) -> str:
        w = self._table.cellWidget(row, col)
        return w.currentText().strip() if isinstance(w, QComboBox) else ""

    def _number(self, row: int, col: int) -> float | None:
        item = self._table.item(row, col)
        try:
            return float(item.text().strip()) if item is not None else None
        except ValueError:
            return None

    def axes(self) -> list[CompositionAxis]:
        """Build the axis list from the table, skipping blank/unparseable rows.

        Skipping rather than raising matches the targets editor: a half-typed row
        must not blank the whole preview while the operator is still typing. The
        inline issue line, not an exception, is what tells them a row is ignored.
        """
        out: list[CompositionAxis] = []
        for row in range(self._table.rowCount()):
            label = self._combo_text(row, _C_TYPE)
            kind = _KIND_FOR_LABEL.get(label)
            a = self._combo_text(row, _C_A)
            low = self._number(row, _C_LOW)
            high = self._number(row, _C_HIGH)
            if kind is None or not a or low is None or high is None:
                continue
            b = self._combo_text(row, _C_B)
            if kind == "molar_ratio" and not b:
                continue
            if high < low:
                low, high = high, low     # a transposed pair is a typo, not an error
            try:
                out.append(CompositionAxis(
                    kind=kind, a=a, b=b, low=low, high=high,
                    basis=self._combo_text(row, _C_BASIS) or "volume"))
            except ValueError:
                continue
        return out

    def parameter_space(self) -> dict[str, dict[str, Any]]:
        """The searchable axes as a campaign ``parameter_space``."""
        return axes_parameter_space(self.axes())

    def build_targets_fn(self):
        """``params -> targets`` for :class:`GeneralFormulation`."""
        return build_targets_from_axes(self.axes())

    def issues(self) -> list[str]:
        return validate_axes(self.axes(), n_stocks=self._n_stocks)

    # -- persistence ----------------------------------------------------------

    def to_state(self) -> list[dict[str, Any]]:
        return [
            {"kind": ax.kind, "a": ax.a, "b": ax.b,
             "low": ax.low, "high": ax.high, "basis": ax.basis}
            for ax in self.axes()
        ]

    def from_state(self, rows: Any) -> None:
        self._table.setRowCount(0)
        for row in rows or []:
            try:
                self.add_axis(
                    AXIS_LABELS.get(str(row.get("kind")), _TYPES[0]),
                    a=str(row.get("a", "")), b=str(row.get("b", "")),
                    low=str(row.get("low", 0)), high=str(row.get("high", 1)),
                    basis=str(row.get("basis", "volume")),
                )
            except (AttributeError, TypeError, ValueError):
                continue
        self._refresh_issues()

    # -- inline validation ----------------------------------------------------

    def _refresh_issues(self) -> None:
        issues = self.issues()
        self._lbl_issues.setText(" ".join(issues))
        self._lbl_issues.setVisible(bool(issues))
