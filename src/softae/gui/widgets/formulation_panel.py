from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from softae.config.loader import syringe_parallel_counts
from softae.core.formulation import (
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    TotalDepositTarget,
    compute_elution_volumes,
    deposited_component_names,
    map_to_pump_volumes,
    solve_formulation,
    species_concentration,
)
from softae.gui.widgets.catalog_editor_base import INVALID_BG, CatalogEditorMixin
from softae.gui.widgets.targets_editor import TargetsEditor

__all__ = ["INVALID_BG", "FormulationPanel"]


def _configured_pump_ids() -> list[int]:
    """Pump indices available for assignment, from config (falls back to 2)."""
    try:
        return sorted(syringe_parallel_counts().keys())
    except Exception:
        return [0, 1]


class FormulationPanel(CatalogEditorMixin, QDialog):
    volumes_calculated = Signal(list)
    catalogs_changed = Signal()  # emitted after any successful save (canonical or Save As)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        chem_catalog: ChemicalCatalog | None = None,
        sol_catalog: SolutionCatalog | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Formulation Manager")
        self.setMinimumWidth(760)
        self.setMinimumHeight(600)

        # Resolve catalogs: injected (caller owns them) or auto-load from the
        # canonical data_root; either path degrades to empty without a dialog.
        self._resolve_catalogs(chem_catalog, sol_catalog)

        self._last_result: list[float] | None = None
        self._pump_combos: dict[str, QComboBox] = {}
        self._frac_spins: dict[str, QDoubleSpinBox] = {}

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self._build_chem_group())      # mixin
        main_layout.addWidget(self._build_solution_group())  # mixin
        main_layout.addWidget(self._build_calculator_group())
        main_layout.addLayout(self._build_bottom_row())

        # Populate all tables/combos from the resolved catalogs (→ hook → pumps).
        self._refresh_tables_from_catalogs()

        # Live validation/highlight + component-combo sync on cell edits.
        self._wire_table_signals()

    # -- Hook override: the base notifies us when the solution set changes ------

    def _on_solution_set_changed(self) -> None:
        self._refresh_pump_assignments()
        self._refresh_target_choices()

    def _refresh_target_choices(self) -> None:
        """Feed the targets editor the species / deposited components of the
        checked stocks (so A/B are dropdowns of valid names)."""
        if not hasattr(self, "_targets_editor"):
            return
        catalog = self._build_chem_catalog()
        species: set[str] = set()
        components: set[str] = set()
        for name in self._checked_solution_names():
            if name in self._sol_catalog._solutions:
                sol = self._sol_catalog.get(name)
                try:  # best-effort: a stock with an unknown chemical ref is skipped
                    species.update(species_concentration(sol, catalog).keys())
                    components.update(deposited_component_names(sol))
                except (KeyError, ValueError):
                    continue
        self._targets_editor.set_available(
            sorted(species), sorted(components), len(self._checked_solution_names())
        )

    # -- Calculator / pump UI (FormulationPanel-only) --------------------------

    def _build_calculator_group(self) -> QGroupBox:
        """Build the Volume Calculator group (target spin + pump rows + result)."""
        calc_grp = QGroupBox("Volume Calculator")
        calc_lay = QVBoxLayout(calc_grp)

        # Mode: per-stock dep fractions (classic) vs composition targets solved by
        # solve_formulation (the same solver the twin and autonomous loop use).
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._combo_form_mode = QComboBox()
        self._combo_form_mode.addItems(["Manual fractions", "Composition targets"])
        self._combo_form_mode.currentIndexChanged.connect(self._on_form_mode_changed)
        mode_row.addWidget(self._combo_form_mode)
        mode_row.addStretch()
        calc_lay.addLayout(mode_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target deposition (µL):"))
        self._spin_target = QDoubleSpinBox()
        self._spin_target.setRange(0.1, 1000.0)
        self._spin_target.setValue(20.0)
        self._spin_target.setDecimals(2)
        target_row.addWidget(self._spin_target)
        target_row.addStretch()
        calc_lay.addLayout(target_row)

        # Composition-targets editor (shown only in that mode); the target-µL above
        # is the TotalDepositTarget added at solve time.
        self._targets_editor = TargetsEditor()
        self._targets_editor.setVisible(False)
        calc_lay.addWidget(self._targets_editor)

        self._pump_layout = QVBoxLayout()
        calc_lay.addLayout(self._pump_layout)

        self._btn_calculate = QPushButton("Calculate")
        self._btn_calculate.clicked.connect(self._on_calculate)
        calc_lay.addWidget(self._btn_calculate)

        self._lbl_result = QLabel("")
        self._lbl_result.setWordWrap(True)
        calc_lay.addWidget(self._lbl_result)
        return calc_grp

    def _build_bottom_row(self) -> QHBoxLayout:
        """Build the bottom button row (Apply / Save / Save As… / Load From… / Close)."""
        bottom_row = QHBoxLayout()

        self._btn_apply = QPushButton("Apply to All Channels")
        self._btn_apply.clicked.connect(self._on_apply)
        bottom_row.addWidget(self._btn_apply)

        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save_canonical)
        bottom_row.addWidget(self._btn_save)

        self._btn_save_as = QPushButton("Save As…")
        self._btn_save_as.clicked.connect(self._on_save_as)
        bottom_row.addWidget(self._btn_save_as)

        self._btn_load = QPushButton("Load From…")
        self._btn_load.clicked.connect(self._on_load)
        bottom_row.addWidget(self._btn_load)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        bottom_row.addWidget(self._btn_close)
        return bottom_row

    # -- Pump assignment helpers -----------------------------------------------

    def _refresh_pump_assignments(self) -> None:
        # Clear existing pump assignment widgets
        while self._pump_layout.count():
            item = self._pump_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        self._pump_combos.clear()
        self._frac_spins.clear()
        included = self._checked_solution_names()
        if not included:
            hint = QLabel("Check one or more solutions above to include them here.")
            hint.setEnabled(False)
            self._pump_layout.addWidget(hint)
            return
        for name in included:
            row_lay = QHBoxLayout()
            row_lay.addWidget(QLabel(name))
            combo = QComboBox()
            combo.addItems([f"Pump {pid}" for pid in _configured_pump_ids()])
            row_lay.addWidget(combo)

            row_lay.addWidget(QLabel("Dep. fraction:"))
            frac = QDoubleSpinBox()
            frac.setRange(0.0, 1.0)
            frac.setSingleStep(0.05)
            frac.setDecimals(3)
            frac.setValue(0.0)
            frac.setToolTip(
                "Share of the target deposition drawn from this stock.\n"
                "Leave at 0 to auto-balance the remainder equally across stocks."
            )
            row_lay.addWidget(frac)
            row_lay.addStretch()
            self._pump_layout.addLayout(row_lay)
            self._pump_combos[name] = combo
            self._frac_spins[name] = frac
        self._apply_form_mode_to_pumps()

    # -- Composition-targets mode ----------------------------------------------

    def _form_mode(self) -> str:
        """'manual' (per-stock fractions) or 'targets' (solve_formulation)."""
        return "targets" if self._combo_form_mode.currentIndex() == 1 else "manual"

    def _on_form_mode_changed(self, _index: int = 0) -> None:
        self._targets_editor.setVisible(self._form_mode() == "targets")
        self._apply_form_mode_to_pumps()

    def _apply_form_mode_to_pumps(self) -> None:
        """Dep-fraction spins are solved in targets mode → disable (not delete) them."""
        targets_mode = self._form_mode() == "targets"
        for spin in self._frac_spins.values():
            spin.setEnabled(not targets_mode)

    # -- Compute-time validation confirmation ----------------------------------

    def _confirm_compute_with_issues(self, issues: list[str]) -> bool:
        body = self._issue_dialog_body(
            issues,
            "Calculate anyway with the noted defaults?\n"
            "(A mass-based component that references an unknown chemical will still fail.)",
        )
        reply = QMessageBox.question(
            self,
            "Formulation Validation",
            body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Yes

    # -- Calculate -------------------------------------------------------------

    def _on_calculate(self) -> None:
        self._build_current_solution()  # flush on-screen component edits
        issues = self._validate_entries()
        self._highlight_invalid_cells()  # tint offending cells, as Save does
        if issues and not self._confirm_compute_with_issues(issues):
            return  # Cancel → NO compute, NO result, NO emit

        catalog = self._build_chem_catalog()

        # Only the solutions ticked in the menu list are included.
        included = self._checked_solution_names()
        solutions: dict[str, Solution] = {}
        for name in included:
            if name in self._sol_catalog._solutions:
                solutions[name] = self._sol_catalog.get(name)

        if not solutions:
            QMessageBox.warning(
                self, "Error", "No solutions selected — check at least one in the list."
            )
            return

        target = self._spin_target.value()
        pump_assignment: dict[str, int] = {}
        for name, combo in self._pump_combos.items():
            pump_assignment[name] = combo.currentIndex()

        # Per-solution deposition fractions: a positive value pins that stock's
        # share; 0 means "auto" (the core splits the remainder equally).
        fractions: dict[str, float] = {
            name: spin.value()
            for name, spin in self._frac_spins.items()
            if spin.value() > 0
        }

        notes: list[str] = []
        try:
            if self._form_mode() == "targets":
                # Composition targets → per-stock volumes → the same [pumps…, total]
                # vector the manual path emits (map_to_pump_volumes' shape).
                targets = [*self._targets_editor.targets(), TotalDepositTarget(target)]
                plan = solve_formulation(
                    solutions, catalog, targets, pump_assignment=pump_assignment
                )
                result = [*plan.per_pump_uL, plan.grand_total_uL]
                notes = plan.notes
            else:
                elution = compute_elution_volumes(
                    solutions, catalog, target, fractions or None
                )
                result = map_to_pump_volumes(elution, pump_assignment)
        except Exception as exc:
            QMessageBox.warning(self, "Calculation Error", str(exc))
            return

        self._last_result = result
        pump_parts = [f"Pump {i}: {vol:.2f} µL" for i, vol in enumerate(result[:-1])]
        text = " | ".join(pump_parts) + f" | Total: {result[-1]:.2f} µL"
        if notes:
            text += "\n" + " ; ".join(notes)
        self._lbl_result.setText(text)

    def _on_apply(self) -> None:
        if self._last_result is None:
            QMessageBox.warning(self, "Error", "Run calculation first.")
            return
        self.volumes_calculated.emit(self._last_result)
