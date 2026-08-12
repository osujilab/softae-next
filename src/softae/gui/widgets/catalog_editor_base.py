"""Shared catalog-editing logic for the formulation/catalog editors.

``CatalogEditorMixin`` is a plain-object mixin (NOT a ``QObject`` subclass) that
carries the data-loss-safe table↔catalog round-trip, the save-time validation
gate + invalid-cell highlighting, the constrained chemical dropdown, and the
chemical rename-cascade.  It is shared verbatim by both
:class:`~softae.gui.widgets.formulation_panel.FormulationPanel` and
:class:`~softae.gui.widgets.catalog_manager.CatalogManager` so any future fix
lands in one place.

Design rules (see the catalog/deposition spec §Extraction):

* Declare the mixin FIRST in the MRO (``class X(CatalogEditorMixin, QDialog)``)
  so its methods win and ``QDialog`` provides the ``QObject`` machinery.
* Qt requires ``Signal(...)`` to be declared on the concrete ``QObject``
  subclass, so ``catalogs_changed`` (and ``FormulationPanel``'s
  ``volumes_calculated``) STAY declared on each subclass.  The mixin only
  *emits* ``self.catalogs_changed`` — valid because every subclass declares it.
* The base must not know about pumps/calculators.  Every place that previously
  refreshed the pump assignments now calls the overridable hook
  :meth:`_on_solution_set_changed` (a base no-op).  ``FormulationPanel``
  overrides it to refresh pump rows; ``CatalogManager`` leaves it a no-op.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from softae.config import loader
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.gui.widgets import formulation_io as fio

INVALID_BG = QColor("#ffcdd2")  # light red tint for offending cells


class CatalogEditorMixin:
    """Shared chemical/solution catalog-editing behaviour (see module docstring)."""

    # -- Overridable hook ------------------------------------------------------

    def _on_solution_set_changed(self) -> None:
        """Hook: the solution set (or a table reload) changed.  Base no-op.

        ``FormulationPanel`` overrides this to refresh its pump-assignment rows;
        ``CatalogManager`` leaves it a no-op (it has no pump/calculator UI).
        """

    # -- Catalog resolution / refresh ------------------------------------------

    def _resolve_catalogs(
        self,
        chem_catalog: ChemicalCatalog | None,
        sol_catalog: SolutionCatalog | None,
    ) -> None:
        """Resolve catalogs: injected (caller owns them) or auto-loaded.

        Either path degrades to empty without a dialog.
        """
        if chem_catalog is None and sol_catalog is None:
            self._chem_catalog, self._sol_catalog = self._auto_load_catalogs()
        else:
            self._chem_catalog = chem_catalog if chem_catalog is not None else ChemicalCatalog()
            self._sol_catalog = sol_catalog if sol_catalog is not None else SolutionCatalog()

    @staticmethod
    def _auto_load_catalogs() -> tuple[ChemicalCatalog, SolutionCatalog]:
        """Load catalogs from ``loader.data_root()``; degrade to empty on failure."""
        try:
            root = loader.data_root()
            return (
                ChemicalCatalog.load_csv(root / "chemicals.csv"),
                SolutionCatalog.load_csv(root / "solutions.csv"),
            )
        except Exception:
            return ChemicalCatalog(), SolutionCatalog()

    def _refresh_tables_from_catalogs(self) -> None:
        """(Re)populate chem table, solution list, component table, and hook."""
        self._chem_table.blockSignals(True)
        fio.populate_chem_table(self._chem_table, self._chem_catalog)
        self._chem_table.blockSignals(False)

        self._list_solutions.blockSignals(True)
        self._list_solutions.clear()
        for name in self._sol_catalog.list_names():
            self._add_solution_list_item(name, checked=True)
        self._list_solutions.blockSignals(False)
        if self._list_solutions.count():
            self._list_solutions.setCurrentRow(0)
        self._update_toggle_all_label()

        self._on_solution_selected(self._current_solution_name())
        self._on_solution_set_changed()

    # -- Widget builders -------------------------------------------------------

    def _build_chem_group(self) -> QGroupBox:
        """Build the Chemical Catalog group (7-col table + Add/Copy/Remove)."""
        chem_grp = QGroupBox("Chemical Catalog")
        chem_lay = QVBoxLayout(chem_grp)

        self._chem_table = QTableWidget(0, 7)
        self._chem_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Formula",
                "Density (g/mL)",
                "MW (g/mol)",
                "Viscosity (mPa·s)",
                "Particulate",
                "Notes",
            ]
        )
        chem_lay.addWidget(self._chem_table)

        chem_btn_row = QHBoxLayout()
        self._btn_add_chem = QPushButton("Add Chemical")
        self._btn_add_chem.clicked.connect(self._on_add_chemical)
        chem_btn_row.addWidget(self._btn_add_chem)

        self._btn_copy_chem = QPushButton("Copy to New")
        self._btn_copy_chem.setToolTip(
            "Prepopulate a new chemical row from the selected row's data"
        )
        self._btn_copy_chem.clicked.connect(self._on_copy_chemical)
        chem_btn_row.addWidget(self._btn_copy_chem)

        self._btn_remove_chem = QPushButton("Remove Selected")
        self._btn_remove_chem.clicked.connect(self._on_remove_chemical)
        chem_btn_row.addWidget(self._btn_remove_chem)

        chem_btn_row.addStretch()
        chem_lay.addLayout(chem_btn_row)
        return chem_grp

    def _build_solution_group(self) -> QGroupBox:
        """Build the Stock Solutions group (list + 5-col component table + CRUD)."""
        sol_grp = QGroupBox("Stock Solutions")
        sol_lay = QVBoxLayout(sol_grp)

        sol_body = QHBoxLayout()

        # Left: checkable menu list of solutions. The selected row drives the
        # component table on the right; the checkbox controls whether the
        # solution appears in the Volume Calculator below (auto-updating).
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Solutions (✓ = include in calculator):"))
        self._list_solutions = QListWidget()
        self._list_solutions.setMaximumWidth(240)
        self._list_solutions.currentTextChanged.connect(self._on_solution_selected)
        self._list_solutions.itemChanged.connect(self._on_solution_item_changed)
        left_col.addWidget(self._list_solutions)

        self._btn_toggle_all_sol = QPushButton("Deselect All")
        self._btn_toggle_all_sol.setToolTip(
            "Check or uncheck every solution's include box at once."
        )
        self._btn_toggle_all_sol.clicked.connect(self._on_toggle_all_solutions)
        left_col.addWidget(self._btn_toggle_all_sol)

        sol_btn_row = QHBoxLayout()
        self._btn_new_sol = QPushButton("New")
        self._btn_new_sol.clicked.connect(self._on_new_solution)
        sol_btn_row.addWidget(self._btn_new_sol)

        self._btn_copy_sol = QPushButton("Copy to New")
        self._btn_copy_sol.setToolTip(
            "Create a new solution prepopulated with the selected solution's components"
        )
        self._btn_copy_sol.clicked.connect(self._on_copy_solution)
        sol_btn_row.addWidget(self._btn_copy_sol)

        self._btn_rename_sol = QPushButton("Rename")
        self._btn_rename_sol.setToolTip("Rename the selected solution.")
        self._btn_rename_sol.clicked.connect(self._on_rename_solution)
        sol_btn_row.addWidget(self._btn_rename_sol)

        self._btn_del_sol = QPushButton("Delete")
        self._btn_del_sol.clicked.connect(self._on_delete_solution)
        sol_btn_row.addWidget(self._btn_del_sol)
        left_col.addLayout(sol_btn_row)
        sol_body.addLayout(left_col)

        # Right: the solution-contents (component) table + its buttons.
        right_col = QVBoxLayout()
        self._comp_table = QTableWidget(0, 6)
        self._comp_table.setHorizontalHeaderLabels(
            ["Chemical", "Role", "Quantity", "Unit", "Calc mode", "Deposits?"]
        )
        right_col.addWidget(self._comp_table)

        comp_btn_row = QHBoxLayout()
        self._btn_add_comp = QPushButton("Add Component")
        self._btn_add_comp.clicked.connect(self._on_add_component)
        comp_btn_row.addWidget(self._btn_add_comp)

        self._btn_remove_comp = QPushButton("Remove Component")
        self._btn_remove_comp.clicked.connect(self._on_remove_component)
        comp_btn_row.addWidget(self._btn_remove_comp)

        comp_btn_row.addStretch()
        right_col.addLayout(comp_btn_row)
        sol_body.addLayout(right_col)
        sol_body.setStretch(0, 0)
        sol_body.setStretch(1, 1)

        sol_lay.addLayout(sol_body)
        return sol_grp

    def _wire_table_signals(self) -> None:
        """Connect live validation/highlight + component-combo sync on cell edits."""
        self._chem_table.itemChanged.connect(self._on_chem_item_changed)
        self._comp_table.itemChanged.connect(self._on_comp_item_changed)

    # -- Solution list helpers (checkable menu list) ---------------------------

    def _add_solution_list_item(
        self, name: str, *, checked: bool = True
    ) -> QListWidgetItem:
        """Append a checkable item to the solution menu list."""
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._list_solutions.addItem(item)
        if hasattr(self, "_btn_toggle_all_sol"):  # covers New / Copy-to-New / reload
            self._update_toggle_all_label()
        return item

    def _current_solution_name(self) -> str:
        """Text of the currently selected solution (empty string if none)."""
        item = self._list_solutions.currentItem()
        return item.text() if item is not None else ""

    def _checked_solution_names(self) -> list[str]:
        """Names of solutions whose include-checkbox is ticked."""
        names: list[str] = []
        for i in range(self._list_solutions.count()):
            it = self._list_solutions.item(i)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                names.append(it.text())
        return names

    def _on_solution_item_changed(self, _item: QListWidgetItem | None = None) -> None:
        """A solution's include-checkbox toggled → notify the subclass hook."""
        self._update_toggle_all_label()
        self._on_solution_set_changed()

    # -- Chemical catalog slots ------------------------------------------------

    def _on_add_chemical(self) -> None:
        fio.add_chem_row(self._chem_table, None)
        self._refresh_component_chemical_options()

    def _on_copy_chemical(self) -> None:
        """Prepopulate a new chemical row from the selected row's data."""
        row = self._chem_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Copy to New", "Select a chemical row to copy first.")
            return
        name_item = self._chem_table.item(row, 0)
        name_text = name_item.text().strip() if name_item is not None else ""
        if not name_text:
            QMessageBox.warning(
                self, "Copy to New", "Selected row has no chemical name to copy."
            )
            return
        cat = self._build_chem_catalog()
        src = cat._chemicals.get(name_text)
        if src is None:
            QMessageBox.warning(self, "Copy to New", "Selected row is not a valid chemical.")
            return

        base = f"{name_text} (copy)"
        new_name = base
        n = 2
        while new_name in cat._chemicals:
            new_name = f"{base} {n}"
            n += 1
        copy = Chemical(
            name=new_name,
            formula=src.formula,
            density_g_per_mL=src.density_g_per_mL,
            molar_mass_g_per_mol=src.molar_mass_g_per_mol,
            notes=src.notes,
            viscosity_mPa_s=src.viscosity_mPa_s,
            is_particulate=src.is_particulate,
        )
        fio.add_chem_row(self._chem_table, copy)
        self._refresh_component_chemical_options()

    def _on_remove_chemical(self) -> None:
        row = self._chem_table.currentRow()
        if row >= 0:
            self._chem_table.removeRow(row)
            self._refresh_component_chemical_options()

    # -- Solution slots --------------------------------------------------------

    def _on_new_solution(self) -> None:
        name, ok = QInputDialog.getText(self, "New Solution", "Solution name:")
        if not (ok and name.strip()):
            return
        name = name.strip()
        if name in self._sol_catalog._solutions:
            QMessageBox.warning(
                self, "Duplicate", f"A solution named '{name}' already exists."
            )
            return
        self._sol_catalog.add(Solution(name=name))
        item = self._add_solution_list_item(name, checked=True)
        self._list_solutions.setCurrentItem(item)
        self._on_solution_set_changed()

    def _on_copy_solution(self) -> None:
        """Create a new solution prepopulated with the selected solution's components."""
        src_name = self._current_solution_name()
        if not src_name:
            QMessageBox.warning(self, "Copy to New", "Select a solution to copy first.")
            return
        self._build_current_solution()  # flush on-screen component edits first
        src = self._sol_catalog.get(src_name)

        new_name = self._unique_solution_name(f"{src_name} (copy)")
        new_name, ok = QInputDialog.getText(
            self, "Copy to New", "New solution name:", text=new_name
        )
        if not (ok and new_name.strip()):
            return
        new_name = new_name.strip()
        if new_name in self._sol_catalog._solutions:
            QMessageBox.warning(
                self, "Duplicate", f"A solution named '{new_name}' already exists."
            )
            return

        components = [
            SolutionComponent(
                chemical_name=c.chemical_name,
                role=c.role,
                quantity=c.quantity,
                unit=c.unit,
                calc_mode=c.calc_mode,
            )
            for c in src.components
        ]
        self._sol_catalog.add(Solution(name=new_name, components=components))
        item = self._add_solution_list_item(new_name, checked=True)
        self._list_solutions.setCurrentItem(item)
        self._on_solution_set_changed()

    def _unique_solution_name(self, base: str) -> str:
        """Return ``base``, or ``base 2`` / ``base 3`` … if it is already taken."""
        if base not in self._sol_catalog._solutions:
            return base
        n = 2
        while f"{base} {n}" in self._sol_catalog._solutions:
            n += 1
        return f"{base} {n}"

    def _on_delete_solution(self) -> None:
        item = self._list_solutions.currentItem()
        if item is None:
            return
        name = item.text()
        self._sol_catalog.remove(name)
        self._list_solutions.takeItem(self._list_solutions.row(item))
        self._comp_table.setRowCount(0)
        self._update_toggle_all_label()
        self._on_solution_set_changed()

    def _on_rename_solution(self) -> None:
        """Rename the selected solution, re-keying it in the catalog."""
        old = self._current_solution_name()
        if not old:
            QMessageBox.warning(self, "Rename Solution", "Select a solution to rename first.")
            return
        new, ok = QInputDialog.getText(
            self, "Rename Solution", "New name:", text=old
        )
        if not (ok and new.strip()):
            return
        new = new.strip()
        if new == old:
            return
        if new in self._sol_catalog._solutions:
            QMessageBox.warning(self, "Duplicate", f"A solution named '{new}' already exists.")
            return
        self._build_current_solution()  # flush pending component edits into `old`
        sol = self._sol_catalog.get(old)
        self._sol_catalog.remove(old)
        sol.name = new
        self._sol_catalog.add(sol)  # re-keyed under the new name (add uses sol.name)
        item = self._list_solutions.currentItem()
        if item is not None:
            self._list_solutions.blockSignals(True)
            item.setText(new)
            self._list_solutions.blockSignals(False)
        self._on_solution_set_changed()

    def _solutions_all_checked(self) -> bool:
        n = self._list_solutions.count()
        return n > 0 and all(
            self._list_solutions.item(i).checkState() == Qt.CheckState.Checked
            for i in range(n)
        )

    def _update_toggle_all_label(self) -> None:
        self._btn_toggle_all_sol.setText(
            "Deselect All" if self._solutions_all_checked() else "Select All"
        )

    def _on_toggle_all_solutions(self) -> None:
        """One button: uncheck every include box if all are checked, else check all."""
        n = self._list_solutions.count()
        if n == 0:
            return
        new_state = (
            Qt.CheckState.Unchecked if self._solutions_all_checked() else Qt.CheckState.Checked
        )
        self._list_solutions.blockSignals(True)
        for i in range(n):
            self._list_solutions.item(i).setCheckState(new_state)
        self._list_solutions.blockSignals(False)
        self._update_toggle_all_label()
        self._on_solution_set_changed()  # one refresh, not one per item

    def _on_solution_selected(self, name: str) -> None:
        self._comp_table.blockSignals(True)
        self._comp_table.setRowCount(0)
        if name and name in self._sol_catalog._solutions:
            fio.populate_comp_table(
                self._comp_table, self._sol_catalog.get(name), self._current_chem_names()
            )
        self._comp_table.blockSignals(False)

    def _on_add_component(self) -> None:
        fio.add_comp_row(self._comp_table, None, self._current_chem_names())

    def _on_remove_component(self) -> None:
        row = self._comp_table.currentRow()
        if row >= 0:
            self._comp_table.removeRow(row)

    # -- Component chemical dropdown sync --------------------------------------

    def _current_chem_names(self) -> list[str]:
        """Chemical names read LIVE from col 0 of the chem table, blanks skipped.

        Reads the table (not ``self._chem_catalog``) so a chemical the user just
        added — even unsaved — is offered in component dropdowns.
        """
        names: list[str] = []
        for row in range(self._chem_table.rowCount()):
            item = self._chem_table.item(row, 0)
            text = item.text().strip() if item is not None else ""
            if text and text not in names:
                names.append(text)
        return names

    def _refresh_component_chemical_options(self) -> None:
        """Rebuild each component combo's option list from the current chem names,
        preserving each combo's current selection (incl. a preserved unknown)."""
        names = self._current_chem_names()
        for row in range(self._comp_table.rowCount()):
            combo = self._comp_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            current = combo.currentData() or combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            ordered = list(names)
            if current and current not in ordered:
                ordered.insert(0, current)
            for n in ordered:
                combo.addItem(n, n)  # display == UserRole
            combo.setCurrentText(current if current else "")
            combo.blockSignals(False)

    def _on_chem_item_changed(self, item=None) -> None:
        """Chem-table edit → cascade rename + resync combos + re-highlight."""
        if item is not None and item.column() == 0:
            self._maybe_cascade_rename(item)  # may cascade old→new, or block+revert
        self._refresh_component_chemical_options()
        self._highlight_invalid_cells()

    # -- Chemical rename cascade (referential integrity) -----------------------

    def _maybe_cascade_rename(self, item: QTableWidgetItem) -> None:
        """Detect a chemical rename on the col-0 name cell and cascade it.

        Uses the pre-edit name stored on ``Qt.UserRole`` to tell a rename from a
        brand-new name or a clear.  A rename onto an existing name is blocked and
        reverted; an accepted rename repoints every component reference.
        """
        new = item.text().strip()
        old_data = item.data(Qt.UserRole)
        old = (old_data or "").strip()

        # Not a rename: brand-new name (old blank) or a clear (new blank).
        if not old or not new or old == new:
            self._stamp_name_role(item, new)  # keep baseline current
            return

        # Collision: renaming onto an existing chemical name → block + revert.
        if self._name_collides(item, new):
            QMessageBox.warning(
                self,
                "Rename Chemical",
                f"A chemical named '{new}' already exists. Choose a different name.",
            )
            self._revert_name_cell(item, old)  # restore old text + UserRole
            return

        # Accepted rename old → new: cascade, then re-stamp the baseline.
        self._cascade_component_rename(old, new)
        self._stamp_name_role(item, new)

    def _name_collides(self, item: QTableWidgetItem, new: str) -> bool:
        """True if any OTHER chem row's col-0 text equals ``new`` (stripped)."""
        own_row = self._chem_table.row(item)
        for row in range(self._chem_table.rowCount()):
            if row == own_row:
                continue
            other = self._chem_table.item(row, 0)
            if other is not None and other.text().strip() == new:
                return True
        return False

    def _stamp_name_role(self, item: QTableWidgetItem, name: str) -> None:
        """Update the pre-edit baseline on ``Qt.UserRole`` without re-entering itemChanged."""
        self._chem_table.blockSignals(True)
        try:
            item.setData(Qt.UserRole, name)
        finally:
            self._chem_table.blockSignals(False)

    def _revert_name_cell(self, item: QTableWidgetItem, old: str) -> None:
        """Restore a blocked rename's cell text + baseline, guarded against recursion."""
        self._chem_table.blockSignals(True)
        try:
            item.setText(old)
            item.setData(Qt.UserRole, old)
        finally:
            self._chem_table.blockSignals(False)

    def _cascade_component_rename(self, old: str, new: str) -> None:
        """Point every component reference from ``old`` to ``new`` — stored + on-screen."""
        # (a) Stored catalog refs — all solutions, including non-displayed ones.
        for sol_name in self._sol_catalog.list_names():
            for comp in self._sol_catalog.get(sol_name).components:
                if comp.chemical_name == old:
                    comp.chemical_name = new

        # (b) On-screen component combos of the currently displayed solution.
        #     Retarget any combo still selecting ``old`` to ``new`` BEFORE the
        #     option-list refresh, or the refresh would preserve ``old`` as an
        #     orphaned "unknown".
        for row in range(self._comp_table.rowCount()):
            combo = self._comp_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            current = combo.currentData() or combo.currentText()
            if current == old:
                combo.blockSignals(True)
                if combo.findData(new) < 0:
                    combo.addItem(new, new)  # ensure ``new`` is selectable
                combo.setCurrentIndex(combo.findData(new))
                combo.blockSignals(False)

    def _on_comp_item_changed(self, _item=None) -> None:
        """Component-table edit → re-highlight offending cells."""
        self._highlight_invalid_cells()

    # -- Save-time validation + highlighting -----------------------------------

    def _validate_entries(self) -> list[str]:
        """Return human-readable issue strings for the on-screen tables (empty = clean).

        Reads raw cell text (not the built catalog) so it surfaces the same safe
        fallbacks the build helpers apply.  Never raises.
        """
        issues: list[str] = []

        # --- Chemical table ---
        for row in range(self._chem_table.rowCount()):
            name = self._chem_table.item(row, 0)
            name_text = name.text().strip() if name is not None else ""
            density_text = self._chem_table.item(row, 2)
            density_text = density_text.text().strip() if density_text is not None else ""
            if name_text:
                bad, reason = self._numeric_issue(density_text)
                if bad:
                    issues.append(
                        f"Chemical '{name_text}': {reason} density — will assume 1.0 g/mL"
                    )
            elif self._chem_row_has_data(row):
                issues.append(f"Chemical row {row + 1} has data but no name — will be skipped")

        # --- Component table (currently shown solution) ---
        sol_name = self._current_solution_name()
        known = set(self._current_chem_names())
        for row in range(self._comp_table.rowCount()):
            combo = self._comp_table.cellWidget(row, 0)
            if isinstance(combo, QComboBox):
                chem = (combo.currentData() or combo.currentText()).strip()
            else:
                item = self._comp_table.item(row, 0)
                chem = item.text().strip() if item is not None else ""
            qty_item = self._comp_table.item(row, 2)
            qty_text = qty_item.text().strip() if qty_item is not None else ""
            if not chem:
                if self._comp_row_has_data(row):
                    issues.append(
                        f"Component row {row + 1} has data but no name — will be skipped"
                    )
                continue
            bad, _reason = self._numeric_issue(qty_text)
            if bad:
                issues.append(f"Solution '{sol_name}' component '{chem}': quantity <= 0")
            if chem not in known:
                issues.append(
                    f"Solution '{sol_name}' component '{chem}': "
                    f"references unknown chemical '{chem}'"
                )

        # --- All stored solutions (belt-and-braces for non-selected ones) ---
        for name in self._sol_catalog.list_names():
            if name == sol_name:
                continue  # already validated on-screen
            for comp in self._sol_catalog.get(name).components:
                if comp.quantity <= 0:
                    issues.append(
                        f"Solution '{name}' component '{comp.chemical_name}': quantity <= 0"
                    )
                if comp.chemical_name not in known:
                    issues.append(
                        f"Solution '{name}' component '{comp.chemical_name}': "
                        f"references unknown chemical '{comp.chemical_name}'"
                    )

        return issues

    @staticmethod
    def _numeric_issue(text: str) -> tuple[bool, str]:
        """Classify a numeric cell: (is_bad, reason) where reason ∈ blank/invalid/non-positive."""
        if text == "":
            return True, "blank"
        try:
            value = float(text)
        except (TypeError, ValueError):
            return True, "invalid"
        if value <= 0:
            return True, "non-positive"
        return False, ""

    def _chem_row_has_data(self, row: int) -> bool:
        """True if any non-name chem cell holds data (so a blank name is a mistake)."""
        for col in (1, 2, 3, 4, 6):
            item = self._chem_table.item(row, col)
            if item is not None and item.text().strip():
                return True
        part = self._chem_table.item(row, 5)
        if part is not None and part.checkState() == Qt.CheckState.Checked:
            return True
        return False

    def _comp_row_has_data(self, row: int) -> bool:
        """True if any non-name component cell differs from its blank-row default."""
        for col in (1, 2, 3):
            item = self._comp_table.item(row, col)
            if item is not None and item.text().strip():
                return True
        return False

    def _issue_dialog_body(self, issues: list[str], closing: str) -> str:
        """Shared body builder for the Save and Compute validation dialogs."""
        body = "The formulation has the following issues:\n\n• " + "\n• ".join(issues[:20])
        if len(issues) > 20:
            body += f"\n… and {len(issues) - 20} more."
        return body + "\n\n" + closing

    def _confirm_save_with_issues(self, issues: list[str]) -> bool:
        body = self._issue_dialog_body(issues, "Save anyway with the noted defaults?")
        reply = QMessageBox.question(
            self,
            "Catalog Validation",
            body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _set_cell_invalid(self, table: QTableWidget, row: int, col: int, invalid: bool) -> None:
        item = table.item(row, col)
        created = item is None
        if created:
            item = QTableWidgetItem("")
        item.setBackground(INVALID_BG if invalid else QBrush())
        if created:
            table.setItem(row, col, item)

    def _highlight_invalid_cells(self) -> None:
        """Colour offending numeric/name cell backgrounds; clear valid ones.

        Cheap, non-recursive: guarded by ``blockSignals`` so ``setBackground`` does
        not re-trigger ``itemChanged``.
        """
        self._chem_table.blockSignals(True)
        self._comp_table.blockSignals(True)
        try:
            # Chem table: name (col 0) for data-bearing blank names; density (col 2).
            for row in range(self._chem_table.rowCount()):
                name = self._chem_table.item(row, 0)
                name_text = name.text().strip() if name is not None else ""
                density = self._chem_table.item(row, 2)
                density_text = density.text().strip() if density is not None else ""
                if name_text:
                    bad, _ = self._numeric_issue(density_text)
                    self._set_cell_invalid(self._chem_table, row, 2, bad)
                    self._set_cell_invalid(self._chem_table, row, 0, False)
                else:
                    has_data = self._chem_row_has_data(row)
                    self._set_cell_invalid(self._chem_table, row, 0, has_data)
                    self._set_cell_invalid(self._chem_table, row, 2, False)

            # Component table: quantity (col 2); row-header tint for unknown chemical.
            known = set(self._current_chem_names())
            for row in range(self._comp_table.rowCount()):
                combo = self._comp_table.cellWidget(row, 0)
                if isinstance(combo, QComboBox):
                    chem = (combo.currentData() or combo.currentText()).strip()
                else:
                    item = self._comp_table.item(row, 0)
                    chem = item.text().strip() if item is not None else ""
                qty_item = self._comp_table.item(row, 2)
                qty_text = qty_item.text().strip() if qty_item is not None else ""
                bad, _ = self._numeric_issue(qty_text)
                self._set_cell_invalid(self._comp_table, row, 2, bool(chem) and bad)
                header = self._comp_table.verticalHeaderItem(row)
                if header is None:
                    header = QTableWidgetItem(str(row + 1))
                    self._comp_table.setVerticalHeaderItem(row, header)
                unknown = bool(chem) and chem not in known
                header.setBackground(INVALID_BG if unknown else QBrush())
        finally:
            self._chem_table.blockSignals(False)
            self._comp_table.blockSignals(False)

    # -- Build objects from tables ---------------------------------------------

    def _build_chem_catalog(self) -> ChemicalCatalog:
        return fio.build_chem_catalog(self._chem_table)

    def _build_current_solution(self) -> Solution | None:
        name = self._current_solution_name()
        if not name:
            return None
        sol = fio.build_solution(name, self._comp_table)
        self._sol_catalog.add(sol)
        return sol

    # -- Save / Load -----------------------------------------------------------

    def _on_save_canonical(self) -> None:
        """Persist both catalogs to the canonical ``data_root`` and signal change."""
        self._build_current_solution()  # flush on-screen component edits
        issues = self._validate_entries()
        self._highlight_invalid_cells()
        if issues and not self._confirm_save_with_issues(issues):
            return  # Cancel → NO write, NO catalogs_changed
        try:
            root = loader.data_root()
            root.mkdir(parents=True, exist_ok=True)
            self._build_chem_catalog().save_csv(root / "chemicals.csv")
            self._sol_catalog.save_csv(root / "solutions.csv")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))
            return
        self.catalogs_changed.emit()

    def _on_save_as(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not directory:
            return
        d = Path(directory)
        self._build_current_solution()
        issues = self._validate_entries()
        self._highlight_invalid_cells()
        if issues and not self._confirm_save_with_issues(issues):
            return  # Cancel → NO write, NO catalogs_changed
        try:
            self._build_chem_catalog().save_csv(d / "chemicals.csv")
            self._sol_catalog.save_csv(d / "solutions.csv")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))
            return
        self.catalogs_changed.emit()

    def _on_load(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Load Directory")
        if not directory:
            return
        d = Path(directory)
        self._chem_catalog = ChemicalCatalog.load_csv(d / "chemicals.csv")
        self._sol_catalog = SolutionCatalog.load_csv(d / "solutions.csv")
        self._refresh_tables_from_catalogs()
