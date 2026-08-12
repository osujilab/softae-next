"""Reusable composition-targets editor: a small table → solve_formulation targets.

Shared by the Deposition twin and the Formulation Manager so both speak the same
target vocabulary (molar ratio / dried fraction / concentration) and build the
exact same :class:`~softae.core.formulation.FormulationTarget` list.  The
``TotalDepositTarget`` (the deposition-µL scale) is owned by the host panel, not
this editor — it is added alongside ``targets()`` at solve time.

A/B are **dropdowns** populated by the host from the checked stocks
(:meth:`set_available`): *species* (dissolved solutes) for ratio/concentration,
*deposited components* for dried-fraction.  An inline warning flags an
under/over-determined target set or a name that isn't in the stocks — the two
mistakes that make the composition not match the intent.
"""

from __future__ import annotations

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

from softae.core.formulation import (
    Basis,
    ConcentrationTarget,
    DriedFractionTarget,
    FormulationTarget,
    MolarRatioTarget,
)

_TYPES = ["Molar ratio", "Dried fraction", "Concentration"]
_BASES = ["volume", "mass", "mole"]
_BASIS_MAP = {"volume": Basis.VOLUME, "mass": Basis.MASS, "mole": Basis.MOLE}
_ISSUE_STYLE = "color: #f9a825; font-weight: bold;"  # amber

# Column indices.
_C_TYPE, _C_A, _C_B, _C_VALUE, _C_BASIS = range(5)


class TargetsEditor(QWidget):
    """Compact editor whose rows become ``FormulationTarget`` objects.

    Column semantics by Type — Molar ratio: A/B species = Value.  Dried fraction:
    component A = Value on Basis.  Concentration: species A = Value mol/L.
    Emits :attr:`changed` on any edit so the host can recompute live.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._species: list[str] = []      # dissolved solutes (ratio / concentration A,B)
        self._components: list[str] = []   # deposited components (dried-fraction A)
        self._n_stocks: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Type", "A", "B", "Value", "Basis"])
        self._table.setToolTip(
            "A/B are chemical names from the checked stocks.  Molar ratio: A/B species "
            "= Value.  Dried fraction: component A = Value (Basis).  Concentration: "
            "species A = Value mol/L.  You need one target per checked stock (the "
            "target-µL box counts as one)."
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setMinimumHeight(180)  # legible, with room for a few rows + scroll
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(_C_TYPE, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_C_A, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_C_B, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_C_VALUE, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_C_BASIS, QHeaderView.ResizeMode.Interactive)
        self._table.setColumnWidth(_C_TYPE, 120)
        self._table.setColumnWidth(_C_VALUE, 70)
        self._table.setColumnWidth(_C_BASIS, 80)
        self._table.itemChanged.connect(self.changed)
        lay.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add target")
        self._btn_add.clicked.connect(lambda: self.add_target())
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
        """Feed the names available from the checked stocks (repopulates combos).

        ``species`` fill A/B for ratio/concentration; ``components`` fill A for
        dried-fraction.  ``n_stocks`` drives the determinacy check.
        """
        self._species = list(species)
        self._components = list(components)
        self._n_stocks = int(n_stocks)
        for row in range(self._table.rowCount()):
            self._repopulate_row(row)
        self._refresh_issues()

    # -- editing --------------------------------------------------------------

    def _make_name_combo(self) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)  # editable so a name can be set programmatically / typed
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
        """Point A/B combos at the right name list for the row's current Type."""
        type_combo = self._table.cellWidget(row, _C_TYPE)
        a_combo = self._table.cellWidget(row, _C_A)
        b_combo = self._table.cellWidget(row, _C_B)
        if not isinstance(type_combo, QComboBox):
            return
        ttype = type_combo.currentText()
        a_items = self._components if ttype == "Dried fraction" else self._species
        if isinstance(a_combo, QComboBox):
            self._fill_combo(a_combo, a_items)
        b_enabled = ttype == "Molar ratio"
        if isinstance(b_combo, QComboBox):
            self._fill_combo(b_combo, self._species if b_enabled else [])
            b_combo.setEnabled(b_enabled)

    def _on_type_changed(self) -> None:
        combo = self.sender()
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, _C_TYPE) is combo:
                self._repopulate_row(row)
                break
        self.changed.emit()

    def add_target(
        self,
        type_text: str | None = None,
        *,
        a: str = "",
        b: str = "",
        value: str = "0",
        basis: str = "volume",
    ) -> int:
        """Append a target row (optionally pre-filled). Returns the row index."""
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
        self._table.setItem(row, _C_VALUE, QTableWidgetItem(value))
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

    # -- read-out -------------------------------------------------------------

    def _combo_text(self, row: int, col: int) -> str:
        w = self._table.cellWidget(row, col)
        return w.currentText().strip() if isinstance(w, QComboBox) else ""

    def _value_text(self, row: int) -> str:
        item = self._table.item(row, _C_VALUE)
        return item.text().strip() if item is not None else ""

    def targets(self) -> list[FormulationTarget]:
        """Build the target list from the table (skips blank / unparseable rows)."""
        out: list[FormulationTarget] = []
        for row in range(self._table.rowCount()):
            ttype = self._combo_text(row, _C_TYPE)
            a = self._combo_text(row, _C_A)
            value_text = self._value_text(row)
            if not a or value_text == "":
                continue
            try:
                value = float(value_text)
            except ValueError:
                continue
            if ttype == "Molar ratio":
                b = self._combo_text(row, _C_B)
                if not b:
                    continue
                out.append(MolarRatioTarget(a, b, value))
            elif ttype == "Dried fraction":
                basis = _BASIS_MAP.get(self._combo_text(row, _C_BASIS), Basis.VOLUME)
                out.append(DriedFractionTarget(a, value, basis))
            elif ttype == "Concentration":
                out.append(ConcentrationTarget(a, value))
        return out

    # -- validation -----------------------------------------------------------

    def issues(self) -> list[str]:
        """Human-readable problems with the current target set (empty if clean)."""
        targets = self.targets()
        out: list[str] = []
        # Determinacy: the host always adds one TotalDepositTarget (the µL box).
        n_constraints = len(targets) + 1
        if self._n_stocks:
            if n_constraints < self._n_stocks:
                out.append(
                    f"Under-determined: {self._n_stocks} checked stocks need "
                    f"{self._n_stocks} targets (incl. the target-µL box) — add "
                    f"{self._n_stocks - n_constraints} more."
                )
            elif n_constraints > self._n_stocks:
                out.append(
                    f"Over-determined: {n_constraints} targets for {self._n_stocks} "
                    f"stocks — remove {n_constraints - self._n_stocks}."
                )
        for t in targets:
            if isinstance(t, MolarRatioTarget):
                for sp in (t.numerator, t.denominator):
                    if self._species and sp not in self._species:
                        out.append(f"'{sp}' is not a dissolved species in the checked stocks.")
            elif isinstance(t, ConcentrationTarget):
                if self._species and t.species not in self._species:
                    out.append(f"'{t.species}' is not a dissolved species in the checked stocks.")
            elif isinstance(t, DriedFractionTarget):
                if self._components and t.component not in self._components:
                    out.append(f"'{t.component}' is not a deposited component of the checked stocks.")
        return out

    def _refresh_issues(self) -> None:
        issues = self.issues()
        self._lbl_issues.setText("  ".join(f"⚠ {m}" for m in issues))
