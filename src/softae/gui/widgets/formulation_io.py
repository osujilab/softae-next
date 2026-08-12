"""Table↔catalog round-trip helpers for :class:`FormulationPanel`.

Extracted from ``formulation_panel.py`` so the widget module stays lean.  These
are the CSV/table field round-trip functions whose parse semantics must match
``softae.core.formulation`` ``load_csv`` / ``_opt_float`` (see the catalog
management spec §2): blank/invalid numeric cells degrade to the same defaults the
core uses, and no function here raises on malformed cell text.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QTableWidget, QTableWidgetItem

from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
)

CALC_MODES = ["Volume-based", "Mass-based"]


def _text(table: QTableWidget, row: int, col: int) -> str:
    item = table.item(row, col)
    return item.text() if item is not None else ""


def _parse_float(text: str, default: float) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _opt_float(text: str | None) -> float | None:
    """Blank → ``None``; unparseable → ``None``; else ``float`` (matches ``_opt_float``)."""
    if text is None or str(text).strip() == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def make_check_item(checked: bool) -> QTableWidgetItem:
    """Checkable, textless item for the Particulate column."""
    item = QTableWidgetItem()
    item.setFlags(
        Qt.ItemFlag.ItemIsUserCheckable
        | Qt.ItemFlag.ItemIsEnabled
        | Qt.ItemFlag.ItemIsSelectable
    )
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


def make_calc_mode_combo(calc_mode: str = "Volume-based") -> QComboBox:
    combo = QComboBox()
    combo.addItems(CALC_MODES)
    combo.setCurrentText(calc_mode if calc_mode in CALC_MODES else "Volume-based")
    return combo


# Deposition-accounting axis (orthogonal to Role identity): does this component's
# retained volume count toward the deposited-film target?  "(role default)" ->
# None (inherit from solute identity); the two explicit choices override it.
_DEPOSITS_CHOICES = ["(role default)", "Deposits", "Excluded"]


def make_deposits_combo(counts_as_deposit: bool | None = None) -> QComboBox:
    """Combo for ``SolutionComponent.counts_as_deposit`` (None/True/False)."""
    combo = QComboBox()
    combo.addItems(_DEPOSITS_CHOICES)
    combo.setCurrentIndex(0 if counts_as_deposit is None else (1 if counts_as_deposit else 2))
    combo.setToolTip(
        "Whether this component's volume counts toward the deposited-film target. "
        "'(role default)' inherits from Role (solutes deposit, solvents don't); set "
        "'Excluded' for a solute whose volume is negligible (e.g. a dissolved salt)."
    )
    return combo


def _deposits_from_combo(combo: object) -> bool | None:
    """Read a deposits combo back to None/True/False (defaults to None)."""
    if not isinstance(combo, QComboBox):
        return None
    idx = combo.currentIndex()
    return None if idx == 0 else (True if idx == 1 else False)


UNKNOWN_MARK = "  (unknown)"  # optional visual marker suffix for a preserved legacy ref


def make_chemical_combo(names: list[str], current: str = "") -> QComboBox:
    """Non-editable combo of chemical names for a component's Chemical cell.

    ``names`` are the current chemical names (from the live chem table).  If
    ``current`` is non-blank and not in ``names`` (a legacy / unknown ref), it is
    prepended as a preserved, selectable entry so loading a catalog never silently
    drops or rewrites a component's chemical.  The combo is NOT editable, so a NEW
    selection is constrained to valid names while a preexisting unknown value is
    still displayed.  Each item stores its raw name in ``Qt.UserRole`` so
    ``build_solution`` can read ``currentData()`` and never depend on display text.
    """
    combo = QComboBox()
    combo.setEditable(False)
    ordered = list(names)
    if current and current not in ordered:
        ordered.insert(0, current)  # preserve unknown ref, first
    for name in ordered:
        combo.addItem(name, name)  # display == Qt.UserRole
    if current:
        combo.setCurrentText(current)
    return combo


def add_chem_row(table: QTableWidget, chem: Chemical | None = None) -> int:
    """Append one 7-column chemical row; a blank row still gets a check item."""
    row = table.rowCount()
    table.insertRow(row)
    if chem is None:
        table.setItem(row, 5, make_check_item(False))
        return row
    name_item = QTableWidgetItem(chem.name)
    name_item.setData(Qt.UserRole, chem.name)  # pre-edit baseline for rename detection
    table.setItem(row, 0, name_item)
    table.setItem(row, 1, QTableWidgetItem(chem.formula))
    table.setItem(row, 2, QTableWidgetItem(str(chem.density_g_per_mL)))
    table.setItem(row, 3, QTableWidgetItem(str(chem.molar_mass_g_per_mol)))
    visc = "" if chem.viscosity_mPa_s is None else str(chem.viscosity_mPa_s)
    table.setItem(row, 4, QTableWidgetItem(visc))
    table.setItem(row, 5, make_check_item(chem.is_particulate))
    table.setItem(row, 6, QTableWidgetItem(chem.notes))
    return row


def add_comp_row(
    table: QTableWidget,
    comp: SolutionComponent | None = None,
    chem_names: list[str] | None = None,
) -> int:
    """Append one 6-column component row (Chemical / Calc-mode / Deposits combos)."""
    row = table.rowCount()
    table.insertRow(row)
    names = chem_names or []
    current = comp.chemical_name if comp else ""
    table.setCellWidget(row, 0, make_chemical_combo(names, current))
    table.setItem(row, 1, QTableWidgetItem(comp.role if comp else "dep"))
    table.setItem(row, 2, QTableWidgetItem(str(comp.quantity) if comp else "0"))
    table.setItem(row, 3, QTableWidgetItem(comp.unit if comp else "mL"))
    table.setCellWidget(row, 4, make_calc_mode_combo(comp.calc_mode if comp else "Volume-based"))
    table.setCellWidget(
        row, 5, make_deposits_combo(comp.counts_as_deposit if comp else None)
    )
    return row


def populate_chem_table(table: QTableWidget, catalog: ChemicalCatalog) -> None:
    table.setRowCount(0)
    for name in catalog.list_names():
        add_chem_row(table, catalog.get(name))


def populate_comp_table(
    table: QTableWidget,
    solution: Solution,
    chem_names: list[str] | None = None,
) -> None:
    table.setRowCount(0)
    for comp in solution.components:
        add_comp_row(table, comp, chem_names)


def build_chem_catalog(table: QTableWidget) -> ChemicalCatalog:
    """Rebuild a :class:`ChemicalCatalog` from the 7-column chem table.

    Round-trips every field; blank-name rows are skipped; numeric cells default
    per the core (density→1.0, mw→0.0, viscosity→None).  Never raises.
    """
    cat = ChemicalCatalog()
    for row in range(table.rowCount()):
        name = _text(table, row, 0).strip()
        if not name:
            continue
        part_item = table.item(row, 5)
        is_particulate = (
            part_item is not None and part_item.checkState() == Qt.CheckState.Checked
        )
        cat.add(
            Chemical(
                name=name,
                formula=_text(table, row, 1),
                density_g_per_mL=_parse_float(_text(table, row, 2), 1.0),
                molar_mass_g_per_mol=_parse_float(_text(table, row, 3), 0.0),
                notes=_text(table, row, 6),
                viscosity_mPa_s=_opt_float(_text(table, row, 4)),
                is_particulate=is_particulate,
            )
        )
    return cat


def build_solution(name: str, table: QTableWidget) -> Solution:
    """Rebuild a :class:`Solution` (name + components) from the 5-column table.

    Preserves each component's ``calc_mode`` from its combo; blank-name rows are
    skipped; quantity defaults to 0.0.  Never raises.
    """
    components: list[SolutionComponent] = []
    for row in range(table.rowCount()):
        combo0 = table.cellWidget(row, 0)
        if isinstance(combo0, QComboBox):
            chem = (combo0.currentData() or combo0.currentText()).strip()
        else:  # defensive: legacy text cell
            chem = _text(table, row, 0).strip()
        if not chem:
            continue
        combo = table.cellWidget(row, 4)
        calc_mode = combo.currentText() if isinstance(combo, QComboBox) else "Volume-based"
        components.append(
            SolutionComponent(
                chemical_name=chem,
                role=_text(table, row, 1) or "dep",
                quantity=_parse_float(_text(table, row, 2), 0.0),
                unit=_text(table, row, 3) or "mL",
                calc_mode=calc_mode or "Volume-based",
                counts_as_deposit=_deposits_from_combo(table.cellWidget(row, 5)),
            )
        )
    return Solution(name=name, components=components)
