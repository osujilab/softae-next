"""Deposition digital-twin panel: stocks + well plan in, live film prediction out."""

from __future__ import annotations

import csv

from PySide6.QtCore import QLineF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.core.deposition import (
    DepositionSummary,
    WellGeometry,
    carrier_component_keys,
    simulate_plate_deposition,
)
from softae.core.formulation import (
    ChemicalCatalog,
    ElutionResult,
    FormulationPlan,
    Solution,
    SolutionCatalog,
    TotalDepositTarget,
    compute_elution_volumes,
    deposited_component_names,
    elution_from_stock_volumes,
    solve_formulation,
    species_concentration,
)
from softae.gui.widgets.deposition_fractions import (
    FractionRow,
    normalize_to_one,
    resolve_sum_state,
)
from softae.gui.widgets.targets_editor import TargetsEditor

_ERROR_STYLE = "color: #c62828; font-weight: bold;"
_OVERFLOW_STYLE = _ERROR_STYLE + " background: #ffebee; padding: 4px;"
_SUM_OK_STYLE = "color: #2e7d32; font-weight: bold;"  # green
_SUM_WARN_STYLE = "color: #f9a825; font-weight: bold;"  # amber
_SUM_ERROR_STYLE = _ERROR_STYLE  # red (#c62828, reused)
_AUTO_SPIN_STYLE = "color: #9e9e9e;"  # muted grey for disabled Auto-on spins

# Stock table column indices.
_COL_USE = 0
_COL_AUTO = 1
_COL_NAME = 2
_COL_FRACTION = 3
_COL_ELUTED = 4
_COL_DEP = 5
_COL_CARRIER = 6


def _fmt(v: float) -> str:
    """Full-precision float for a data export (un-truncated for downstream tools)."""
    return f"{v:.6f}"


def build_deposition_csv_rows(
    elution: ElutionResult,
    summary: DepositionSummary,
    config: dict,
) -> list[list[str]]:
    """Flatten a computed deposition into sectioned CSV rows (no Qt, no I/O)."""
    rows: list[list[str]] = []

    # Section 1 — CONFIG (key,value).
    rows.append(["# CONFIG"])
    rows.append(["key", "value"])
    rows.append(["target_deposition_uL", _fmt(config["target_uL"])])
    rows.append(["well_diameter_mm", _fmt(config["diameter_mm"])])
    rows.append(["well_depth_mm", _fmt(config["depth_mm"])])
    rows.append(["well_capacity_uL", _fmt(config["capacity_uL"])])
    rows.append(["n_wells", str(config["n_wells"])])
    rows.append(["dispense_mode", str(config["dispense_mode"])])
    per_well = config["per_well_uL"]
    rows.append(["per_well_uL", _fmt(per_well) if per_well != "" else ""])
    rows.append(["evaporation_pct", _fmt(config["evaporation_pct"])])
    rows.append([])

    # Section 2 — STOCKS (per-stock elution + input config).
    rows.append(["# STOCKS"])
    rows.append([
        "solution", "used", "auto", "explicit_fraction",
        "resolved_fraction", "eluted_uL", "dep_uL", "carrier_uL",
    ])
    for stock in config["stocks"]:
        name = stock["name"]
        used = bool(stock["used"])
        if used and name in elution.per_solution:
            resolved = _fmt(elution.solution_fractions.get(name, 0.0))
            eluted = _fmt(elution.per_solution[name])
            dep = _fmt(elution.dep_vol_uL.get(name, 0.0))
            carrier = _fmt(elution.carrier_vol_uL.get(name, 0.0))
        else:
            resolved = eluted = dep = carrier = ""
        rows.append([
            name,
            "True" if used else "False",
            "True" if stock["auto"] else "False",
            _fmt(stock["explicit_fraction"]),
            resolved, eluted, dep, carrier,
        ])
    rows.append([])

    # Section 3 — MASS_BALANCE (key,value).
    rows.append(["# MASS_BALANCE"])
    rows.append(["key", "value"])
    rows.append(["total_eluted_uL", _fmt(summary.total_eluted_uL)])
    rows.append(["total_dispensed_uL", _fmt(summary.total_dispensed_uL)])
    rows.append(["undeposited_uL", _fmt(summary.undeposited_uL)])
    rows.append(["total_evaporated_uL", _fmt(summary.total_evaporated_uL)])
    rows.append(["total_final_uL", _fmt(summary.total_final_uL)])
    rows.append(["target_deposition_uL", _fmt(elution.target_deposition_uL)])
    rows.append(["n_wells", str(summary.n_wells)])
    rows.append(["any_overflow", "True" if summary.any_overflow else "False"])
    rows.append([])

    # Section 4 — WELLS (per-well).
    rows.append(["# WELLS"])
    rows.append([
        "well_index", "dispensed_uL", "dep_uL", "carrier_uL", "evaporated_uL",
        "residual_carrier_uL", "final_volume_uL", "wet_thickness_um",
        "final_thickness_um", "wet_fill_fraction", "final_fill_fraction",
        "overflows",
    ])
    for i, w in enumerate(summary.wells):
        rows.append([
            str(i),
            _fmt(w.dispensed_uL),
            _fmt(w.dep_uL),
            _fmt(w.carrier_uL),
            _fmt(w.evaporated_uL),
            _fmt(w.residual_carrier_uL),
            _fmt(w.final_volume_uL),
            _fmt(w.wet_thickness_um),
            _fmt(w.final_thickness_um),
            _fmt(w.wet_fill_fraction),
            _fmt(w.final_fill_fraction),
            "True" if w.overflows else "False",
        ])
    return rows


class WellSketch(QWidget):
    """Illustrative well cross-section: outline to scale, wet level, final film level.

    Purely visual — no interaction, no math beyond scaling three numbers to pixels.
    """

    _OVERFILL_CAP = 1.15  # levels above the rim are capped here so they stay visible

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._well: WellGeometry | None = None
        self._wet_fill = 0.0
        self._final_fill = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(160, 200)

    def set_state(self, well: WellGeometry | None, wet_fill: float, final_fill: float) -> None:
        self._well = well
        self._wet_fill = wet_fill
        self._final_fill = final_fill
        self.update()

    def paintEvent(self, event) -> None:  # Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin = 12.0
        avail_w = max(self.width() - 2 * margin, 10.0)
        avail_h = max((self.height() - 2 * margin) / self._OVERFILL_CAP, 10.0)
        aspect = 1.0
        if self._well is not None:
            aspect = self._well.diameter_mm / self._well.depth_mm
        aspect = min(max(aspect, 0.25), 4.0)  # keep degenerate ratios visible
        well_w = avail_w
        well_h = well_w / aspect
        if well_h > avail_h:
            well_h = avail_h
            well_w = well_h * aspect
        left = (self.width() - well_w) / 2.0
        bottom = self.height() - margin
        rim = bottom - well_h
        if self._well is not None:
            wet_h = min(self._wet_fill, self._OVERFILL_CAP) * well_h
            painter.fillRect(QRectF(left, bottom - wet_h, well_w, wet_h), QColor("#bbdefb"))
            final_h = min(self._final_fill, self._OVERFILL_CAP) * well_h
            painter.fillRect(QRectF(left, bottom - final_h, well_w, final_h), QColor("#37474f"))
        painter.setPen(QPen(QColor("#212121"), 2.0))
        painter.drawLine(QLineF(left, rim, left, bottom))
        painter.drawLine(QLineF(left, bottom, left + well_w, bottom))
        painter.drawLine(QLineF(left + well_w, bottom, left + well_w, rim))
        if self._well is not None and self._wet_fill > 1.0:
            painter.setPen(QPen(QColor("#c62828"), 2.0))
            painter.drawLine(QLineF(left, rim, left + well_w, rim))
        painter.end()


class DepositionPanel(QWidget):
    """Deposition digital-twin panel: stocks + well plan in, live film prediction out.

    Pure GUI shell: all math is delegated to softae.core.formulation
    (compute_elution_volumes) and softae.core.deposition (simulate_plate_deposition).
    Catalogs are injected read-only; editing them is FormulationPanel's job.
    """

    deposition_computed = Signal(object)  # emits DepositionSummary on every valid recompute
    manage_catalogs_requested = Signal()  # "Manage Catalogs…" clicked (launcher wires it)
    reload_catalogs_requested = Signal()  # "Reload catalogs" clicked (launcher wires it)

    def __init__(
        self,
        chem_catalog: ChemicalCatalog,
        sol_catalog: SolutionCatalog,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._chem_catalog = chem_catalog
        self._sol_catalog = sol_catalog
        self._spin_fractions: list[QDoubleSpinBox] = []
        self._last_elution: ElutionResult | None = None
        self._last_summary: DepositionSummary | None = None
        self._last_plan: FormulationPlan | None = None  # targets-mode solve result
        self._build_ui()
        self._populate_stocks()
        self._connect_signals()
        self._recompute()

    # -- UI construction --

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Two columns: the two table-bearing panels (Stock Solutions + Prediction,
        # both scrollable) stack on the LEFT; the config panels (Formulation Target,
        # Wells & Dispense, Evaporation) stack on the RIGHT, so the Formulation
        # Target panel gets the full column height for its expanded targets table.
        top_row = QHBoxLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        # === Stock Solutions ===
        stocks_grp = QGroupBox("Stock Solutions")
        stocks_lay = QVBoxLayout(stocks_grp)
        self._table_stocks = QTableWidget(0, 7)
        self._table_stocks.setHorizontalHeaderLabels(
            ["Use", "Auto", "Solution", "Fraction", "Eluted µL", "Dep µL", "Carrier µL"]
        )
        self._table_stocks.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        stocks_lay.addWidget(self._table_stocks)

        self._lbl_fraction_sum = QLabel("")
        stocks_lay.addWidget(self._lbl_fraction_sum)

        stocks_btn_row = QHBoxLayout()
        self._btn_toggle_all_stocks = QPushButton("Deselect All")
        self._btn_toggle_all_stocks.setToolTip(
            "Check or uncheck every stock's Use box at once."
        )
        self._btn_toggle_all_stocks.clicked.connect(self._on_toggle_all_stocks)
        stocks_btn_row.addWidget(self._btn_toggle_all_stocks)
        self._btn_auto_balance = QPushButton("Auto-balance all")
        self._btn_auto_balance.clicked.connect(self._on_auto_balance)
        stocks_btn_row.addWidget(self._btn_auto_balance)
        self._btn_normalize = QPushButton("Normalize")
        self._btn_normalize.setToolTip(
            "Rescale the explicit (Auto-off) fractions of the checked, dep-bearing "
            "stocks so they sum to 1.00. Auto rows are left untouched."
        )
        self._btn_normalize.setEnabled(False)  # _update_sum_indicator sets live state
        self._btn_normalize.clicked.connect(self._on_normalize)
        stocks_btn_row.addWidget(self._btn_normalize)
        self._btn_manage_catalogs = QPushButton("Manage Catalogs…")
        self._btn_manage_catalogs.clicked.connect(self.manage_catalogs_requested)
        stocks_btn_row.addWidget(self._btn_manage_catalogs)
        self._btn_reload_catalogs = QPushButton("Reload catalogs")
        self._btn_reload_catalogs.clicked.connect(self.reload_catalogs_requested)
        stocks_btn_row.addWidget(self._btn_reload_catalogs)
        self._chk_show_components = QCheckBox("Show component breakdown")
        self._chk_show_components.toggled.connect(self._on_toggle_components)
        stocks_btn_row.addWidget(self._chk_show_components)
        stocks_btn_row.addStretch()
        stocks_lay.addLayout(stocks_btn_row)

        self._table_components = QTableWidget(0, 4)
        self._table_components.setHorizontalHeaderLabels(
            ["Solution", "Component", "Role", "Eluted µL"]
        )
        self._table_components.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table_components.setVisible(False)
        stocks_lay.addWidget(self._table_components)

        left_col.addWidget(stocks_grp, 3)  # left column, top

        # === Formulation Target ===
        target_grp = QGroupBox("Formulation Target")
        target_lay = QVBoxLayout(target_grp)

        # Mode: manual per-stock fractions (the classic path) vs composition targets
        # solved by softae.core.formulation.solve_formulation (the same solver the
        # autonomous loop uses).
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._combo_form_mode = QComboBox()
        self._combo_form_mode.addItems(["Manual fractions", "Composition targets"])
        mode_row.addWidget(self._combo_form_mode)
        mode_row.addStretch()
        target_lay.addLayout(mode_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target deposition (µL):"))
        self._spin_target = QDoubleSpinBox()
        self._spin_target.setRange(0.1, 1000.0)
        self._spin_target.setValue(20.0)
        self._spin_target.setDecimals(2)
        target_row.addWidget(self._spin_target)
        target_row.addStretch()
        target_lay.addLayout(target_row)

        # Composition-targets editor (shown only in that mode). Each row is a
        # solve_formulation target; the target-µL above is the TotalDepositTarget.
        self._targets_editor = TargetsEditor()
        self._targets_editor.setVisible(False)
        target_lay.addWidget(self._targets_editor)

        self._lbl_elution = QLabel("")
        self._lbl_elution.setWordWrap(True)
        target_lay.addWidget(self._lbl_elution)
        right_col.addWidget(target_grp)

        # === Wells & Dispense ===
        wells_grp = QGroupBox("Wells & Dispense")
        wells_form = QFormLayout(wells_grp)
        self._spin_diameter = QDoubleSpinBox()
        self._spin_diameter.setRange(0.1, 100.0)
        self._spin_diameter.setValue(5.0)
        self._spin_diameter.setDecimals(3)
        wells_form.addRow("Well diameter (mm):", self._spin_diameter)
        self._spin_depth = QDoubleSpinBox()
        self._spin_depth.setRange(0.1, 100.0)
        self._spin_depth.setValue(2.0)
        self._spin_depth.setDecimals(3)
        wells_form.addRow("Well depth (mm):", self._spin_depth)
        self._spin_n_wells = QSpinBox()
        self._spin_n_wells.setRange(1, 384)
        self._spin_n_wells.setValue(2)
        wells_form.addRow("Number of wells:", self._spin_n_wells)
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Equal split", "Fixed µL per well"])
        wells_form.addRow("Dispense mode:", self._combo_mode)
        self._spin_per_well = QDoubleSpinBox()
        self._spin_per_well.setRange(0.0, 1000.0)
        self._spin_per_well.setValue(40.0)
        self._spin_per_well.setDecimals(2)
        self._spin_per_well.setEnabled(False)  # equal split is the default mode
        wells_form.addRow("Per-well volume (µL):", self._spin_per_well)
        self._lbl_capacity = QLabel("")
        wells_form.addRow(self._lbl_capacity)
        right_col.addWidget(wells_grp)

        # === Evaporation ===
        evap_grp = QGroupBox("Evaporation")
        evap_row = QHBoxLayout(evap_grp)
        self._slider_evap = QSlider(Qt.Orientation.Horizontal)
        self._slider_evap.setRange(0, 200)  # integer ticks = 0.5 % steps
        self._slider_evap.setValue(190)
        evap_row.addWidget(self._slider_evap)
        self._spin_evap = QDoubleSpinBox()
        self._spin_evap.setRange(0.0, 100.0)
        self._spin_evap.setSingleStep(0.5)
        self._spin_evap.setDecimals(1)
        self._spin_evap.setValue(95.0)
        self._spin_evap.setSuffix(" %")
        evap_row.addWidget(self._spin_evap)
        right_col.addWidget(evap_grp)
        right_col.addStretch()      # absorbs slack in manual mode (see _apply_right_stretch)
        self._right_col = right_col

        # === Prediction ===
        pred_grp = QGroupBox("Prediction")
        pred_lay = QVBoxLayout(pred_grp)
        pred_row = QHBoxLayout()
        self._table_wells = QTableWidget(0, 6)
        self._table_wells.setHorizontalHeaderLabels(
            ["Well", "Wet (µL)", "Final (µL)", "Final thickness (µm)", "Wet fill (%)", "Overflow"]
        )
        self._table_wells.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        pred_row.addWidget(self._table_wells, stretch=1)
        self._sketch = WellSketch()
        pred_row.addWidget(self._sketch)
        pred_lay.addLayout(pred_row)
        self._lbl_balance = QLabel("—")
        pred_lay.addWidget(self._lbl_balance)
        pred_btn_row = QHBoxLayout()
        self._btn_export_csv = QPushButton("Export CSV…")
        self._btn_export_csv.setEnabled(False)  # only when a valid result exists
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        pred_btn_row.addWidget(self._btn_export_csv)
        self._btn_sweep = QPushButton("Sweep ranges…")
        self._btn_sweep.setToolTip(
            "Flag well overflow across ranges of deposition volume and stock "
            "fractions — the whole parameter space at once."
        )
        self._btn_sweep.clicked.connect(self._on_sweep_overflow)
        pred_btn_row.addWidget(self._btn_sweep)
        pred_btn_row.addStretch()
        pred_lay.addLayout(pred_btn_row)
        left_col.addWidget(pred_grp, 2)  # left column, below Stock Solutions

        # Assemble: [Stock Solutions + Prediction | (Target, Wells, Evaporation)].
        top_row.addLayout(left_col, 3)
        top_row.addLayout(right_col, 2)
        main_layout.addLayout(top_row)
        self._apply_right_stretch()

        # === Status row ===
        self._lbl_overflow = QLabel("⚠ Wet volume exceeds well capacity")
        self._lbl_overflow.setStyleSheet(_OVERFLOW_STYLE)
        self._lbl_overflow.setVisible(False)
        main_layout.addWidget(self._lbl_overflow)
        self._lbl_status = QLabel("")
        main_layout.addWidget(self._lbl_status)

    def _populate_stocks(self) -> None:
        # Idempotent: clear old rows/cell-widgets and rebuild the row-aligned
        # fraction list so it stays indexable by row (see _selected_solutions).
        self._table_stocks.blockSignals(True)
        self._table_stocks.setRowCount(0)
        self._spin_fractions.clear()
        for name in self._sol_catalog.list_names():
            row = self._table_stocks.rowCount()
            self._table_stocks.insertRow(row)
            use_item = QTableWidgetItem()
            use_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            use_item.setCheckState(Qt.CheckState.Checked)
            self._table_stocks.setItem(row, _COL_USE, use_item)
            auto_item = QTableWidgetItem()
            auto_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            # OVERRIDE (explicit-first default): rows open Auto OFF, spin editable at 0.00.
            auto_item.setCheckState(Qt.CheckState.Unchecked)
            self._table_stocks.setItem(row, _COL_AUTO, auto_item)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table_stocks.setItem(row, _COL_NAME, name_item)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)  # specialValueText removed: 0.00 means literal zero
            spin.setValue(0.0)
            self._table_stocks.setCellWidget(row, _COL_FRACTION, spin)
            self._spin_fractions.append(spin)
            for col in (_COL_ELUTED, _COL_DEP, _COL_CARRIER):
                out_item = QTableWidgetItem("—")
                out_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                self._table_stocks.setItem(row, col, out_item)
        self._table_stocks.blockSignals(False)
        self._sync_fraction_enabled_states()

    def _connect_signals(self) -> None:
        self._table_stocks.itemChanged.connect(self._on_stock_item_changed)
        self._connect_fraction_spins()
        self._combo_form_mode.currentIndexChanged.connect(self._on_form_mode_changed)
        self._targets_editor.changed.connect(self._on_input_changed)
        for dspin in (self._spin_target, self._spin_diameter, self._spin_depth,
                      self._spin_per_well):
            dspin.valueChanged.connect(self._on_input_changed)
        self._spin_n_wells.valueChanged.connect(self._on_input_changed)
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._slider_evap.valueChanged.connect(self._on_slider_changed)
        self._spin_evap.valueChanged.connect(self._on_evap_spin_changed)

    def _connect_fraction_spins(self) -> None:
        """(Re)connect the per-row fraction spinboxes — they are new objects each
        time the stock table is rebuilt, so this is called after every populate."""
        for spin in self._spin_fractions:
            spin.valueChanged.connect(self._on_input_changed)

    def _sync_fraction_enabled_states(self) -> None:
        """Enable + un-grey the Fraction spin for Auto-off rows; disable + grey it
        for Auto-on rows.  In composition-targets mode the fractions are solved, not
        entered, so every spin is disabled.  Pure UI state — MUST NOT recompute."""
        targets_mode = self._form_mode() == "targets"
        for row in range(self._table_stocks.rowCount()):
            if row >= len(self._spin_fractions):
                break
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            auto_on = (
                auto_item is not None
                and auto_item.checkState() == Qt.CheckState.Checked
            )
            spin = self._spin_fractions[row]
            spin.setEnabled(not auto_on and not targets_mode)
            spin.setReadOnly(auto_on or targets_mode)
            spin.setStyleSheet(_AUTO_SPIN_STYLE if (auto_on or targets_mode) else "")

    # -- Composition-targets mode --

    def _form_mode(self) -> str:
        """'manual' (per-stock fractions) or 'targets' (solve_formulation)."""
        return "targets" if self._combo_form_mode.currentIndex() == 1 else "manual"

    def _refresh_target_choices(self, solutions: dict[str, Solution]) -> None:
        """Feed the targets editor the species / deposited components of the
        checked stocks, so A/B are dropdowns of valid names (not free text)."""
        species: set[str] = set()
        components: set[str] = set()
        for sol in solutions.values():
            try:  # best-effort: a stock with an unknown chemical ref is skipped
                species.update(species_concentration(sol, self._chem_catalog).keys())
                components.update(deposited_component_names(sol))
            except (KeyError, ValueError):
                continue
        self._targets_editor.set_available(
            sorted(species), sorted(components), len(solutions)
        )

    def _on_form_mode_changed(self, _index: int = 0) -> None:
        targets_mode = self._form_mode() == "targets"
        self._targets_editor.setVisible(targets_mode)
        # The manual fraction controls have no meaning while targets are solved.
        self._btn_auto_balance.setEnabled(not targets_mode)
        if targets_mode:
            self._btn_normalize.setEnabled(False)
        self._sync_fraction_enabled_states()
        self._apply_right_stretch()
        self._recompute()

    def _apply_right_stretch(self) -> None:
        """In targets mode the Formulation Target group takes the freed column
        height (its table expands); in manual mode the trailing stretch absorbs it."""
        targets = self._form_mode() == "targets"
        self._right_col.setStretch(0, 1 if targets else 0)  # Formulation Target group
        self._right_col.setStretch(  # trailing stretch (last item)
            self._right_col.count() - 1, 0 if targets else 1
        )

    # -- Public API --

    def show_status(self, message: str) -> None:
        """Show a neutral (non-error) message in the status label."""
        self._lbl_status.setStyleSheet("")
        self._lbl_status.setText(message)

    def set_catalogs(
        self, chem_catalog: ChemicalCatalog, sol_catalog: SolutionCatalog
    ) -> None:
        """Swap in new catalogs, repopulate the stock table, and recompute.

        Best-effort preserves per-row check state, Auto flag, and fraction for
        solutions whose names still exist in the new catalog; new solutions appear
        checked, Auto OFF, at 0.00; removed solutions drop out.  Triggers a
        recompute (or an error/empty state if no solutions remain) without raising.
        """
        # 1. Snapshot current selections keyed by solution name.
        snapshot: dict[str, tuple[bool, bool, float]] = {}
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            name_item = self._table_stocks.item(row, _COL_NAME)
            if use_item is None or name_item is None:
                continue
            checked = use_item.checkState() == Qt.CheckState.Checked
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            auto = auto_item is not None and auto_item.checkState() == Qt.CheckState.Checked
            snapshot[name_item.text()] = (checked, auto, self._spin_fractions[row].value())

        # 2. Swap catalogs and 3. rebuild the stock rows (idempotent).
        self._chem_catalog = chem_catalog
        self._sol_catalog = sol_catalog
        self._populate_stocks()

        # 4. Re-apply the snapshot for surviving names.
        self._table_stocks.blockSignals(True)
        for row in range(self._table_stocks.rowCount()):
            name = self._table_stocks.item(row, _COL_NAME).text()
            if name in snapshot:
                checked, auto, fraction = snapshot[name]
                self._table_stocks.item(row, _COL_USE).setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                self._table_stocks.item(row, _COL_AUTO).setCheckState(
                    Qt.CheckState.Checked if auto else Qt.CheckState.Unchecked
                )
                spin = self._spin_fractions[row]
                spin.blockSignals(True)
                spin.setValue(fraction)
                spin.blockSignals(False)
        self._table_stocks.blockSignals(False)

        # 5. Sync spin enabled states, reconnect the fresh spins, then 6. recompute.
        self._sync_fraction_enabled_states()
        self._connect_fraction_spins()
        self._update_toggle_all_stocks_label()
        self._recompute()

    # -- Slots --

    def _on_input_changed(self, *args: object) -> None:
        self._recompute()

    def _on_stock_item_changed(self, item: QTableWidgetItem) -> None:
        # An Auto-column toggle must re-sync the row's Fraction spin before recompute.
        if item is not None and item.column() == _COL_AUTO:
            self._sync_fraction_enabled_states()
        self._update_toggle_all_stocks_label()  # a Use toggle may flip the button
        self._recompute()

    def _stocks_all_checked(self) -> bool:
        n = self._table_stocks.rowCount()
        return n > 0 and all(
            (it := self._table_stocks.item(r, _COL_USE)) is not None
            and it.checkState() == Qt.CheckState.Checked
            for r in range(n)
        )

    def _update_toggle_all_stocks_label(self) -> None:
        self._btn_toggle_all_stocks.setText(
            "Deselect All" if self._stocks_all_checked() else "Select All"
        )

    def _on_toggle_all_stocks(self) -> None:
        """One button: uncheck every stock's Use box if all are checked, else check all."""
        if self._table_stocks.rowCount() == 0:
            return
        new_state = (
            Qt.CheckState.Unchecked if self._stocks_all_checked() else Qt.CheckState.Checked
        )
        self._table_stocks.blockSignals(True)
        for r in range(self._table_stocks.rowCount()):
            item = self._table_stocks.item(r, _COL_USE)
            if item is not None:
                item.setCheckState(new_state)
        self._table_stocks.blockSignals(False)
        self._update_toggle_all_stocks_label()
        self._recompute()  # one recompute after the batch

    def _on_auto_balance(self) -> None:
        self._table_stocks.blockSignals(True)
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            if use_item is not None and use_item.checkState() == Qt.CheckState.Checked:
                self._table_stocks.item(row, _COL_AUTO).setCheckState(Qt.CheckState.Checked)
        self._table_stocks.blockSignals(False)
        self._sync_fraction_enabled_states()  # grey/disable spins for the now-auto rows
        self._recompute()  # one recompute, not one per row

    def _on_normalize(self) -> None:
        # Gather normalizable rows: checked, Auto-off, dep-bearing (dep_fraction>0).
        rows: list[int] = []
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            if use_item is None or use_item.checkState() != Qt.CheckState.Checked:
                continue
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            if auto_item is not None and auto_item.checkState() == Qt.CheckState.Checked:
                continue
            name = self._table_stocks.item(row, _COL_NAME).text()
            try:
                dep = self._sol_catalog.get(name).dep_fraction(self._chem_catalog)
            except (KeyError, ValueError):
                dep = 0.0
            if dep > 0.0:
                rows.append(row)
        values = [self._spin_fractions[r].value() for r in rows]
        if sum(values) <= 1e-6:
            return  # defensive: button should already be disabled
        scaled = normalize_to_one(values, decimals=self._spin_fractions[rows[0]].decimals())
        for row, val in zip(rows, scaled):
            spin = self._spin_fractions[row]
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._recompute()  # one recompute after the batch, not one per spin

    def _on_toggle_components(self, checked: bool) -> None:
        self._table_components.setVisible(checked)
        if checked:
            self._recompute()  # repopulates the component table while visible

    def _on_mode_changed(self, index: int) -> None:
        self._spin_per_well.setEnabled(index == 1)
        self._recompute()

    def _on_slider_changed(self, value: int) -> None:
        self._spin_evap.blockSignals(True)
        self._spin_evap.setValue(value / 2.0)
        self._spin_evap.blockSignals(False)
        self._recompute()

    def _on_evap_spin_changed(self, value: float) -> None:
        self._slider_evap.blockSignals(True)
        self._slider_evap.setValue(round(value * 2))
        self._slider_evap.blockSignals(False)
        self._recompute()

    def _on_export_csv(self) -> None:
        if self._last_summary is None or self._last_elution is None:
            return  # button should be disabled, but guard anyway
        default = f"deposition_{self._last_summary.n_wells}wells.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export deposition CSV", default, "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return  # user cancelled
        config = self._export_config()
        rows = build_deposition_csv_rows(self._last_elution, self._last_summary, config)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows)
        except OSError as exc:
            self._lbl_status.setStyleSheet(_ERROR_STYLE)
            self._lbl_status.setText(f"CSV export failed: {exc}")
            return
        self.show_status(f"Exported deposition to {path}")

    def _export_config(self) -> dict:
        """Harvest the reproducible input config from the spins + stock table.

        Read-only: touches no signals and mutates nothing.
        """
        well = WellGeometry(self._spin_diameter.value(), self._spin_depth.value())
        fixed = self._combo_mode.currentIndex() == 1
        stocks: list[dict] = []
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            name = self._table_stocks.item(row, _COL_NAME).text()
            stocks.append({
                "name": name,
                "used": use_item is not None
                and use_item.checkState() == Qt.CheckState.Checked,
                "auto": auto_item is not None
                and auto_item.checkState() == Qt.CheckState.Checked,
                "explicit_fraction": self._spin_fractions[row].value(),
            })
        return {
            "target_uL": self._spin_target.value(),
            "diameter_mm": self._spin_diameter.value(),
            "depth_mm": self._spin_depth.value(),
            "capacity_uL": well.capacity_uL,
            "n_wells": self._spin_n_wells.value(),
            "dispense_mode": "fixed_per_well" if fixed else "equal_split",
            "per_well_uL": self._spin_per_well.value() if fixed else "",
            "evaporation_pct": self._spin_evap.value(),
            "stocks": stocks,
        }

    # -- Input harvesting --

    def _selected_solutions(self) -> dict[str, Solution]:
        out: dict[str, Solution] = {}
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            if use_item is not None and use_item.checkState() == Qt.CheckState.Checked:
                name = self._table_stocks.item(row, _COL_NAME).text()
                out[name] = self._sol_catalog.get(name)
        return out

    def _explicit_fractions(self) -> dict[str, float] | None:
        out: dict[str, float] = {}
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            if use_item is None or use_item.checkState() != Qt.CheckState.Checked:
                continue  # unchecked -> not in the solution set
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            if auto_item is not None and auto_item.checkState() == Qt.CheckState.Checked:
                continue  # AUTO ON -> OMIT (core absorbs the remainder)
            name = self._table_stocks.item(row, _COL_NAME).text()
            out[name] = self._spin_fractions[row].value()  # AUTO OFF -> exact, incl. 0.0
        return out or None  # None only when NO explicit (all-auto) rows

    def _build_fraction_rows(self) -> list[FractionRow]:
        """Snapshot the table into Qt-free rows for the Σ-indicator state machine."""
        rows: list[FractionRow] = []
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            name_item = self._table_stocks.item(row, _COL_NAME)
            if use_item is None or name_item is None:
                continue
            checked = use_item.checkState() == Qt.CheckState.Checked
            auto_item = self._table_stocks.item(row, _COL_AUTO)
            auto = auto_item is not None and auto_item.checkState() == Qt.CheckState.Checked
            name = name_item.text()
            try:
                dep_frac = self._sol_catalog.get(name).dep_fraction(self._chem_catalog)
            except (KeyError, ValueError):
                dep_frac = 0.0
            rows.append(
                FractionRow(name, checked, auto, self._spin_fractions[row].value(), dep_frac)
            )
        return rows

    def _update_sum_indicator(self) -> None:
        # In composition-targets mode the Σ indicator reports solver feasibility
        # instead of the (irrelevant) manual fraction sum.
        if self._form_mode() == "targets":
            self._btn_normalize.setEnabled(False)
            plan = self._last_plan
            if plan is None:
                self._lbl_fraction_sum.setStyleSheet("")
                self._lbl_fraction_sum.setText("")
                return
            if not plan.feasible:
                style, msg = _SUM_ERROR_STYLE, (plan.notes[0] if plan.notes else "infeasible")
            elif plan.notes:
                style, msg = _SUM_WARN_STYLE, plan.notes[0]
            else:
                headroom = plan.headroom_uL
                extra = f" · headroom {headroom:.1f} µL" if headroom != float("inf") else ""
                style, msg = _SUM_OK_STYLE, f"✓ targets met{extra}"
            self._lbl_fraction_sum.setStyleSheet(style)
            self._lbl_fraction_sum.setText(msg)
            return
        state = resolve_sum_state(self._build_fraction_rows(), self._spin_target.value())
        style = {
            "ok": _SUM_OK_STYLE,
            "warn": _SUM_WARN_STYLE,
            "error": _SUM_ERROR_STYLE,
        }[state.severity]
        self._lbl_fraction_sum.setStyleSheet(style)
        self._lbl_fraction_sum.setText(state.message)
        tol = 1e-6
        self._btn_normalize.setEnabled(
            state.explicit_sum > tol and state.severity != "ok"
        )

    # -- Recompute pipeline (the only "logic" in the panel) --

    def _recompute(self) -> None:
        try:
            well = WellGeometry(self._spin_diameter.value(), self._spin_depth.value())
            self._lbl_capacity.setText(f"Well capacity: {well.capacity_uL:.2f} µL")
            solutions = self._selected_solutions()
            if not solutions:
                self._show_error("Select at least one solution.")
                return
            if self._form_mode() == "targets":
                # Composition targets → per-stock volumes via the shared solver; the
                # well capacity is the elution budget, so overflow is caught upfront.
                self._refresh_target_choices(solutions)
                targets = [*self._targets_editor.targets(),
                           TotalDepositTarget(self._spin_target.value())]
                plan = solve_formulation(
                    solutions, self._chem_catalog, targets, budget_uL=well.capacity_uL
                )
                self._last_plan = plan
                elution = elution_from_stock_volumes(
                    plan.per_stock_uL, solutions, self._chem_catalog, self._spin_target.value()
                )
            else:
                self._last_plan = None
                fractions = self._explicit_fractions()
                elution = compute_elution_volumes(
                    solutions, self._chem_catalog, self._spin_target.value(), fractions
                )
            dispense = (
                None if self._combo_mode.currentIndex() == 0  # equal split
                else self._spin_per_well.value()  # fixed µL per well
            )
            summary = simulate_plate_deposition(
                elution, well, self._spin_evap.value(), self._spin_n_wells.value(), dispense,
                carrier_keys=carrier_component_keys(solutions),
            )
        except (ValueError, KeyError) as exc:
            self._show_error(str(exc))
            return
        self._show_results(elution, summary)
        self.deposition_computed.emit(summary)

    # -- Overflow sweep (flag overflow across a whole parameter space) --

    def sweepable_axes(self) -> dict[str, tuple[float, float]]:
        """Axes this panel can sweep, each with a sensible default ``(low, high)``.

        Always the target deposition volume; in manual-fractions mode, one axis
        per used stock (``x_<stock>``, a fraction in ``[0, 1]``).  Composition
        targets are heterogeneous target objects, so targets mode sweeps only the
        deposition volume (the composition stays solved to its fixed targets).
        """
        axes: dict[str, tuple[float, float]] = {}
        dep = float(self._spin_target.value())
        axes["deposition_uL"] = (max(0.1, 0.5 * dep), 1.5 * dep)
        if self._form_mode() != "targets":
            for name in self._selected_solutions():
                axes[f"x_{name}"] = (0.0, 1.0)
        return axes

    def _sweep_total_uL(self, point: dict) -> float:
        """Total elution volume (µL) for one swept point — the panel's solve path.

        Honours ``deposition_uL`` and any ``x_<stock>`` fractions in *point*,
        falling back to the current UI values for axes not being swept.
        """
        solutions = self._selected_solutions()
        if not solutions:
            return 0.0
        dep = float(point.get("deposition_uL", self._spin_target.value()))
        frac_keys = {k[2:]: float(v) for k, v in point.items() if k.startswith("x_")}
        if self._form_mode() == "targets" and not frac_keys:
            well = WellGeometry(self._spin_diameter.value(), self._spin_depth.value())
            targets = [*self._targets_editor.targets(), TotalDepositTarget(dep)]
            plan = solve_formulation(
                solutions, self._chem_catalog, targets, budget_uL=well.capacity_uL
            )
            return float(plan.grand_total_uL)
        fractions = None
        if frac_keys:
            from softae.core.formulation import simplex_fractions

            fractions = simplex_fractions(list(solutions.keys()), frac_keys)
        elution = compute_elution_volumes(solutions, self._chem_catalog, dep, fractions)
        return float(elution.grand_total_uL)

    def overflow_sweep(self, axes: dict[str, tuple[float, float]], *, steps: int = 5):
        """Flag well overflow across ``axes`` (``{name: (low, high)}``), ``steps`` each.

        Grid-samples the space and, per point, compares the total elution volume
        (via :meth:`_sweep_total_uL`) against the current well capacity — the same
        capacity/solve the single-point prediction uses.  Returns a
        :class:`softae.core.overflow.OverflowSweepResult`.
        """
        from softae.core.overflow import enumerate_space, sweep_overflow

        well = WellGeometry(self._spin_diameter.value(), self._spin_depth.value())
        bounds = {name: {"low": lo, "high": hi} for name, (lo, hi) in axes.items()}
        points = enumerate_space(bounds, steps=steps)
        return sweep_overflow(points, self._sweep_total_uL, well.capacity_uL)

    def _on_sweep_overflow(self) -> None:
        if not self._selected_solutions():
            self._show_error("Select at least one solution.")
            return
        dlg = _OverflowSweepDialog(self.sweepable_axes(), self.overflow_sweep, self)
        dlg.exec()

    # -- Output rendering --

    def _show_error(self, message: str) -> None:
        self._last_elution = None
        self._last_summary = None
        self._last_plan = None
        self._btn_export_csv.setEnabled(False)
        self._lbl_status.setStyleSheet(_ERROR_STYLE)
        self._lbl_status.setText(message)
        self._table_wells.setRowCount(0)
        self._lbl_balance.setText("—")
        self._lbl_overflow.setVisible(False)
        self._sketch.set_state(None, 0.0, 0.0)
        # Clear per-stock output cells and refresh the (display-only) Σ indicator.
        self._table_stocks.blockSignals(True)
        for row in range(self._table_stocks.rowCount()):
            for col in (_COL_ELUTED, _COL_DEP, _COL_CARRIER):
                self._set_output_cell(row, col, "—")
        self._table_stocks.blockSignals(False)
        self._update_sum_indicator()
        self._table_components.setRowCount(0)

    def _set_output_cell(self, row: int, col: int, text: str) -> None:
        item = self._table_stocks.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table_stocks.setItem(row, col, item)
        item.setText(text)

    def _populate_components(self, elution: ElutionResult) -> None:
        solutions = self._selected_solutions()
        carrier_keys = carrier_component_keys(solutions)
        keys = sorted(
            (k for k in elution.component_vol_uL if k[0] in solutions),
            key=lambda k: (k[0], k[1]),
        )
        self._table_components.blockSignals(True)
        self._table_components.setRowCount(len(keys))
        for i, key in enumerate(keys):
            sol_name, chem_name = key
            role = "carrier" if key in carrier_keys else "dep"
            cells = [sol_name, chem_name, role, f"{elution.component_vol_uL[key]:.2f}"]
            for col, text in enumerate(cells):
                self._table_components.setItem(i, col, QTableWidgetItem(text))
        self._table_components.blockSignals(False)

    def _show_results(self, elution: ElutionResult, summary: DepositionSummary) -> None:
        elution_text = (
            f"Eluted: {elution.grand_total_uL:.2f} µL  "
            f"(dep {elution.total_dep_uL:.2f} / carrier {elution.total_carrier_uL:.2f})"
        )
        if self._last_plan is not None:
            achieved = [
                f"{k}={v:.4g}"
                for k, v in self._last_plan.achieved.items()
                if k != "total_deposit_uL"
            ]
            if achieved:
                elution_text += "\nachieved: " + "  ·  ".join(achieved)
        self._lbl_elution.setText(elution_text)
        # Per-stock output cells + Auto-row resolved-share display, storm-guarded.
        self._table_stocks.blockSignals(True)
        for row in range(self._table_stocks.rowCount()):
            use_item = self._table_stocks.item(row, _COL_USE)
            name = self._table_stocks.item(row, _COL_NAME).text()
            checked = use_item is not None and use_item.checkState() == Qt.CheckState.Checked
            if checked and name in elution.per_solution:
                self._set_output_cell(row, _COL_ELUTED, f"{elution.per_solution[name]:.2f}")
                self._set_output_cell(row, _COL_DEP, f"{elution.dep_vol_uL[name]:.2f}")
                self._set_output_cell(row, _COL_CARRIER, f"{elution.carrier_vol_uL[name]:.2f}")
                auto_item = self._table_stocks.item(row, _COL_AUTO)
                if auto_item is not None and auto_item.checkState() == Qt.CheckState.Checked:
                    spin = self._spin_fractions[row]
                    spin.blockSignals(True)
                    spin.setValue(elution.solution_fractions.get(name, 0.0))
                    spin.blockSignals(False)
            else:
                for col in (_COL_ELUTED, _COL_DEP, _COL_CARRIER):
                    self._set_output_cell(row, col, "—")
        self._table_stocks.blockSignals(False)
        self._update_sum_indicator()
        if self._chk_show_components.isChecked():
            self._populate_components(elution)
        self._table_wells.setRowCount(len(summary.wells))
        for i, w in enumerate(summary.wells):
            cells = [
                str(i),
                f"{w.dispensed_uL:.2f}",
                f"{w.final_volume_uL:.2f}",
                f"{w.final_thickness_um:.1f}",
                f"{w.wet_fill_fraction * 100:.1f}",
                "YES" if w.overflows else "",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 5 and w.overflows:
                    item.setForeground(QColor("#c62828"))
                self._table_wells.setItem(i, col, item)
        self._lbl_balance.setText(
            f"Eluted {summary.total_eluted_uL:.2f} µL | "
            f"Dispensed {summary.total_dispensed_uL:.2f} µL | "
            f"Undeposited {summary.undeposited_uL:.2f} µL | "
            f"Evaporated {summary.total_evaporated_uL:.2f} µL | "
            f"Final {summary.total_final_uL:.2f} µL"
        )
        self._lbl_overflow.setVisible(summary.any_overflow)
        first = summary.wells[0]
        self._sketch.set_state(first.well, first.wet_fill_fraction, first.final_fill_fraction)
        if elution.grand_total_uL <= 0:
            self.show_status("No dep-bearing solution selected — eluted 0 µL.")
        else:
            self.show_status("")
        self._last_elution = elution
        self._last_summary = summary
        self._btn_export_csv.setEnabled(True)


class _OverflowSweepDialog(QDialog):
    """Range-sweep overflow map for the deposition digital twin.

    Presents the panel's sweepable axes with editable ``(min, max)`` ranges; on
    *Run sweep* it grid-samples the space via the panel's ``overflow_sweep``
    callback and renders the per-point overflow map plus a summary banner (which
    fraction of the space overflows and the worst case).
    """

    def __init__(self, axes: dict[str, tuple[float, float]], run_sweep, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overflow sweep")
        self.setMinimumWidth(560)
        self._run_sweep = run_sweep
        self._axis_widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox]] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select axes to sweep and their ranges, then Run. Points whose total "
            "cast volume exceeds the well capacity are flagged."
        ))

        form = QFormLayout()
        for name, (lo, hi) in axes.items():
            row = QHBoxLayout()
            chk = QCheckBox()
            chk.setChecked(True)
            lo_spin = QDoubleSpinBox()
            lo_spin.setRange(0.0, 100_000.0)
            lo_spin.setDecimals(3)
            lo_spin.setValue(float(lo))
            hi_spin = QDoubleSpinBox()
            hi_spin.setRange(0.0, 100_000.0)
            hi_spin.setDecimals(3)
            hi_spin.setValue(float(hi))
            row.addWidget(chk)
            row.addWidget(QLabel("min"))
            row.addWidget(lo_spin)
            row.addWidget(QLabel("max"))
            row.addWidget(hi_spin)
            holder = QWidget()
            holder.setLayout(row)
            form.addRow(name, holder)
            self._axis_widgets[name] = (chk, lo_spin, hi_spin)
        layout.addLayout(form)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps per axis:"))
        self._spin_steps = QSpinBox()
        self._spin_steps.setRange(2, 21)
        self._spin_steps.setValue(5)
        steps_row.addWidget(self._spin_steps)
        steps_row.addStretch()
        self._btn_run = QPushButton("Run sweep")
        self._btn_run.clicked.connect(self._run)
        steps_row.addWidget(self._btn_run)
        layout.addLayout(steps_row)

        self._lbl_summary = QLabel("")
        self._lbl_summary.setWordWrap(True)
        layout.addWidget(self._lbl_summary)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def _selected_axes(self) -> dict[str, tuple[float, float]]:
        return {
            name: (lo.value(), hi.value())
            for name, (chk, lo, hi) in self._axis_widgets.items()
            if chk.isChecked()
        }

    def _run(self) -> None:
        axes = self._selected_axes()
        if not axes:
            self._lbl_summary.setStyleSheet(_ERROR_STYLE)
            self._lbl_summary.setText("Select at least one axis to sweep.")
            return
        result = self._run_sweep(axes, steps=self._spin_steps.value())
        self._show(result, list(axes.keys()))

    def _show(self, result, axis_names: list[str]) -> None:
        cols = axis_names + ["Total (µL)", "Overflow"]
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(result.verdicts))
        for r, (point, verdict) in enumerate(result.verdicts):
            for c, ax in enumerate(axis_names):
                val = point.get(ax, "")
                text = f"{val:.3g}" if isinstance(val, (int, float)) else str(val)
                self._table.setItem(r, c, QTableWidgetItem(text))
            self._table.setItem(
                r, len(axis_names), QTableWidgetItem(f"{verdict.total_uL:.2f}")
            )
            of_item = QTableWidgetItem("YES" if verdict.overflows else "")
            if verdict.overflows:
                of_item.setForeground(QColor("#c62828"))
            self._table.setItem(r, len(axis_names) + 1, of_item)

        if result.any_overflow:
            _worst_point, worst = result.worst
            pct = 100.0 * result.overflow_fraction
            self._lbl_summary.setStyleSheet(_OVERFLOW_STYLE)
            self._lbl_summary.setText(
                f"⚠ {result.n_overflow}/{result.n_points} points ({pct:.0f}%) overflow "
                f"the {result.capacity_uL:.2f} µL well. Worst total {worst.total_uL:.2f} µL "
                f"({-worst.headroom_uL:.2f} µL over)."
            )
        else:
            self._lbl_summary.setStyleSheet(_SUM_OK_STYLE)
            self._lbl_summary.setText(
                f"✓ No overflow across {result.n_points} points "
                f"(peak {result.max_total_uL:.2f} µL ≤ {result.capacity_uL:.2f} µL well)."
            )
