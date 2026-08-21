"""Tab 4: High-Throughput Experiment Builder.

Formulation matrix editor, EIS preset selector, PCB config, workflow
preview, run controls (Start / Pause / Abort), live results table, and
CSV / PDF export.

All experiment execution flows through :class:`WorkflowExecutor` — the
tab never talks directly to instrument drivers.
"""

from __future__ import annotations

import asyncio
import csv
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from softae.config import loader
from softae.config.loader import (
    dropcast_config,
    default_pcb_name,
    eis_presets,
    liquid_handling_config,
    liquid_line_for_pump,
    pcb_configs,
    piezo_config,
    pico_for_channel,
)
from softae.analysis.eis_data import EISResult
from softae.core.liquid_handling import DeadVolumeCorrection, LiquidHandlingCorrector
from softae.core.rig_activity import workflow_instruments
from softae.core.task_catalog import TaskCatalog
from softae.gui.daemon_runner import DaemonRunnerMixin
from softae.gui.rig_claim import rig_run
from softae.gui.widgets.copyable_table import PasteableTableWidget
from softae.workflows.experiment_logger import ExperimentLogger
from softae.workflows.workflow_executor import (
    DEFAULT_MAX_CONSECUTIVE_CHANNEL_FAILURES,
    ExecutorState,
    WorkflowExecutor,
)
from softae.workflows.workflow_model import Workflow, WorkflowStep

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


class ExperimentBuilderTab(DaemonRunnerMixin, QWidget):
    """High-throughput experiment design and execution interface.

    The tab builds a :class:`Workflow` from the formulation matrix and
    configuration selections, then runs it via :class:`WorkflowExecutor`
    in a background thread.  Progress and results are fed back to the UI
    through Qt signals (thread-safe).
    """

    #: Connected dispensing pumps, in formulation-matrix column order. The HT
    #: formulation matrix has one editable column per pump (plus a Total column).
    PUMP_IDS: tuple[int, ...] = (0, 1, 2)

    #: Deposition settling-wait multiplier for the two-phase cast (Phase 3 will
    #: source this from a ``[dropcast]`` config section).
    SETTLE_FACTOR_DEFAULT: float = 2.0

    #: Equivalent circuit "Fit All EIS" starts on. Named, never positional —
    #: ``list(CIRCUIT_MODELS)[0]`` would let a dict reordering silently change
    #: what the HT tab fits with.
    DEFAULT_FIT_MODEL: str = "simpleSalt"

    # Thread-safe signals for executor callbacks → GUI updates
    _sig_step_start = Signal(str, int, int)            # step_name, index, total
    _sig_step_complete = Signal(str, int, int, object, float)  # step_name, idx, total, result, elapsed_s
    _sig_step_error = Signal(str, int, int, str)        # step_name, idx, total, error_msg
    _sig_step_skipped = Signal(str, int, int, str)      # step_name, idx, total, reason
    _sig_state_change = Signal(str, str)                # old_state, new_state
    _sig_workflow_done = Signal(int)                    # exit_code (0=ok, 1=error)
    _sig_channel_hold = Signal(str, int, float)         # channel, consecutive, timeout_s
    # Public signal: sidebar / external widgets listen to this.
    workflow_status_changed = Signal(str)               # human-readable status text
    catalogs_changed = Signal()                         # re-emitted when the editor saves catalogs

    def __init__(
        self,
        manager: "InstrumentManager",
        parent: QWidget | None = None,
        *,
        data_store=None,
    ):
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._ds_run_id: str | None = None  # active DataStore run id
        #: channel -> run_id for every spectrum captured this session (P.14).
        #: ``_ds_run_id`` is the *active* run marker and is cleared when the
        #: workflow finishes, but "Fit All EIS" is a post-run review button, so
        #: the pairing has to be retained where both are still in scope. Same
        #: naming and shape as :attr:`AnalysisTab._run_id_by_channel`.
        self._run_id_by_channel: dict[int, str] = {}
        # Board these casts record single-use occupancy under; set by the
        # pre-run occupancy gate (advances on a confirmed board replacement).
        self._active_board_id: int = 0
        self._executor: WorkflowExecutor | None = None
        self._exp_logger: ExperimentLogger | None = None
        self._run_thread: threading.Thread | None = None
        # The live consecutive-failure prompt, if one is open (see
        # `_ui_channel_hold`) — held only so Qt does not collect it.
        self._hold_box: QMessageBox | None = None
        self._results: list[dict[str, Any]] = []
        self._eis_results: list[EISResult] = []  # captured EIS data
        # Task catalog drives step generation (deposition recipes + EIS task
        # params are resolved from it directly — no role→task indirection).
        self._task_catalog = self._load_task_catalog()
        self._build_ui()
        self._connect_signals()

    @staticmethod
    def _load_task_catalog() -> TaskCatalog:
        """Load the task catalog from the canonical path; empty on any failure."""
        try:
            return TaskCatalog.load_toml(loader.tasks_toml_path())
        except Exception:
            logger.warning("task_catalog_load_failed", exc_info=True)
            return TaskCatalog()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # === Top row: config selectors ===
        config_row = QHBoxLayout()

        # Workflow mode selector
        mode_grp = QGroupBox("Workflow Mode")
        mode_lay = QHBoxLayout(mode_grp)
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Full Protocol", "Measure Only", "Formulate Only"])
        self._combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_lay.addWidget(self._combo_mode)
        config_row.addWidget(mode_grp)

        # Deposition selector — which recipe the engine builds and the method its
        # deposit phase uses.  Both are authored/matured in Process Studio and
        # resolved straight from the task catalog (no role→task indirection).
        proc_grp = QGroupBox("Deposition")
        proc_lay = QVBoxLayout(proc_grp)

        # The deposition recipe the engine runs + its deposit method.
        proc_row2 = QHBoxLayout()
        proc_row2.addWidget(QLabel("Recipe:"))
        self._combo_deposit_recipe = QComboBox()
        self._combo_deposit_recipe.setToolTip(
            "Deposition recipe the engine builds per channel. Authored and "
            "matured in Process Studio.")
        self._populate_deposit_recipe_combo()
        self._combo_deposit_recipe.currentIndexChanged.connect(self._on_recipe_changed)
        proc_row2.addWidget(self._combo_deposit_recipe)
        # Deposit-phase method override — sourced from the Process Studio method
        # library, so a method you author/mature there flows into the HT run.
        proc_row2.addWidget(QLabel("Method:"))
        self._combo_deposit_method = QComboBox()
        self._combo_deposit_method.setToolTip(
            "Method the recipe's deposit phase uses per channel. Authored and "
            "matured in Process Studio (Library/Builder).")
        self._combo_deposit_method.currentIndexChanged.connect(
            self._refresh_deposit_maturity_label)
        proc_row2.addWidget(self._combo_deposit_method, 1)
        self._lbl_deposit_maturity = QLabel("")
        proc_row2.addWidget(self._lbl_deposit_maturity)
        proc_lay.addLayout(proc_row2)

        self._populate_deposit_method_combo()
        self._on_recipe_changed()
        config_row.addWidget(proc_grp)

        # Run sequence — insertable anneal + pointwise/batch measurement ordering.
        seq_grp = QGroupBox("Run Sequence")
        seq_lay = QVBoxLayout(seq_grp)
        seq_row = QHBoxLayout()
        self._chk_anneal = QCheckBox("Insert anneal")
        self._chk_anneal.setToolTip(
            "Add a cure/anneal phase (temperature controller) between deposit and "
            "measure. Runs the catalog anneal task with the temperature/hold below.")
        self._chk_anneal.toggled.connect(self._update_sequence_preview)
        seq_row.addWidget(self._chk_anneal)
        seq_row.addWidget(QLabel("T (°C):"))
        self._spin_anneal_temp = QDoubleSpinBox()
        self._spin_anneal_temp.setRange(20.0, 300.0)
        self._spin_anneal_temp.setValue(150.0)
        seq_row.addWidget(self._spin_anneal_temp)
        seq_row.addWidget(QLabel("Hold (min):"))
        self._spin_anneal_hold = QDoubleSpinBox()
        self._spin_anneal_hold.setRange(0.0, 600.0)
        self._spin_anneal_hold.setValue(5.0)
        seq_row.addWidget(self._spin_anneal_hold)
        seq_lay.addLayout(seq_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Measurement:"))
        self._combo_measure_scope = QComboBox()
        self._combo_measure_scope.addItems(
            ["Per sample (interleaved)", "Batch (all samples, then measure)"])
        self._combo_measure_scope.setToolTip(
            "Per sample: deposit → (anneal) → measure each channel in turn.\n"
            "Batch: deposit all channels, then anneal the plate, then measure all.")
        self._combo_measure_scope.currentIndexChanged.connect(self._update_sequence_preview)
        scope_row.addWidget(self._combo_measure_scope, 1)
        seq_lay.addLayout(scope_row)

        self._lbl_sequence = QLabel("")
        self._lbl_sequence.setWordWrap(True)
        self._lbl_sequence.setStyleSheet("color: #555; font-style: italic;")
        seq_lay.addWidget(self._lbl_sequence)
        config_row.addWidget(seq_grp)
        self._update_sequence_preview()

        # PCB selector
        pcb_grp = QGroupBox("PCB Layout")
        pcb_lay = QHBoxLayout(pcb_grp)
        pcb_lay.addWidget(QLabel("Board:"))
        self._combo_pcb = QComboBox()
        pcbs = pcb_configs()
        self._pcb_data = pcbs
        for name in sorted(pcbs.keys()):
            ch = pcbs[name].get("channels", "?")
            self._combo_pcb.addItem(f"{name}  ({ch} ch)", userData=name)
        if self._combo_pcb.count() == 0:
            self._combo_pcb.addItem("(no PCBs in config)", userData=None)
        _default_pcb = default_pcb_name()
        if _default_pcb:
            _idx = self._combo_pcb.findData(_default_pcb)
            if _idx >= 0:
                self._combo_pcb.setCurrentIndex(_idx)
        self._combo_pcb.currentIndexChanged.connect(self._on_pcb_changed)
        pcb_lay.addWidget(self._combo_pcb)

        self._lbl_pcb_info = QLabel("")
        pcb_lay.addWidget(self._lbl_pcb_info)
        config_row.addWidget(pcb_grp)

        # EIS preset selector with editable parameters
        eis_grp = QGroupBox("EIS Preset")
        eis_vlay = QVBoxLayout(eis_grp)
        eis_top_row = QHBoxLayout()
        eis_top_row.addWidget(QLabel("Preset:"))
        self._combo_eis = QComboBox()
        presets = eis_presets()
        for name in sorted(presets.keys()):
            self._combo_eis.addItem(name)
        if self._combo_eis.count() == 0:
            self._combo_eis.addItem("Standard")
        eis_top_row.addWidget(self._combo_eis)
        eis_top_row.addStretch()
        eis_vlay.addLayout(eis_top_row)

        # EIS parameters stacked vertically (a compact, legible column) — the panel
        # lives in the right column of the second row (below), not this top row.
        eis_params_form = QFormLayout()
        self._spin_eis_f_hi = QSpinBox()
        self._spin_eis_f_hi.setRange(1, 200_000)
        self._spin_eis_f_hi.setSingleStep(10_000)
        eis_params_form.addRow("f_hi (Hz):", self._spin_eis_f_hi)
        self._spin_eis_f_lo = QSpinBox()
        self._spin_eis_f_lo.setRange(1, 200_000)
        eis_params_form.addRow("f_lo (mHz):", self._spin_eis_f_lo)
        self._spin_eis_npts = QSpinBox()
        self._spin_eis_npts.setRange(5, 100)
        eis_params_form.addRow("npts:", self._spin_eis_npts)
        self._spin_eis_mv_ac = QSpinBox()
        self._spin_eis_mv_ac.setRange(1, 200)
        eis_params_form.addRow("mVac:", self._spin_eis_mv_ac)
        self._spin_eis_mv_dc = QSpinBox()
        self._spin_eis_mv_dc.setRange(-500, 500)
        eis_params_form.addRow("mVdc:", self._spin_eis_mv_dc)
        self._combo_eis.currentTextChanged.connect(self._on_eis_preset_changed)
        self._on_eis_preset_changed(self._combo_eis.currentText())
        eis_vlay.addLayout(eis_params_form)
        eis_vlay.addStretch()  # let the params sit at the top; group grows below

        # Dispense config — defaults from the [dropcast] config section.
        dc = dropcast_config()
        self._disp_grp = QGroupBox("Dispense")
        disp_lay = QFormLayout(self._disp_grp)
        # Total deposition flow rate (split per-pump proportionally to volume in
        # the two-phase cast; applied directly in the legacy path).
        self._spin_rate = QDoubleSpinBox()
        self._spin_rate.setRange(0.1, 2120.0)
        self._spin_rate.setValue(float(dc["dispense_rate_uL_min"]))
        self._spin_rate.setSuffix(" µL/min")
        disp_lay.addRow("Dispense rate:", self._spin_rate)

        # Total flush flow rate — drives the start flush and, split per-pump, the
        # per-channel precondition flush (two-phase cast).
        self._spin_flush_rate = QDoubleSpinBox()
        self._spin_flush_rate.setRange(0.1, 2120.0)
        self._spin_flush_rate.setValue(float(dc["line_flush_rate_uL_min"]))
        self._spin_flush_rate.setSuffix(" µL/min")
        disp_lay.addRow("Line flush rate:", self._spin_flush_rate)

        # Per-pump start-flush volumes (step 0 general prime), e.g. "80, 80, 80".
        start_default = [float(v) for v in dc["start_flush_uL"]][: len(self.PUMP_IDS)]
        self._edit_start_flush = QLineEdit(
            ", ".join(f"{v:g}" for v in start_default))
        self._edit_start_flush.setPlaceholderText(
            ", ".join(["vol"] * len(self.PUMP_IDS)) + "  (one per pump)")
        self._edit_start_flush.setToolTip(
            "Per-pump start-flush volumes (µL) for the campaign-start general "
            "flush, comma-separated — one value per pump.")
        disp_lay.addRow("Start flush vol:", self._edit_start_flush)

        # Precondition preload multiplier: preload volume = deposition volume × factor.
        self._spin_flush_factor = QDoubleSpinBox()
        self._spin_flush_factor.setRange(0.0, 100.0)
        self._spin_flush_factor.setSingleStep(0.5)
        self._spin_flush_factor.setValue(float(dc["flush_factor"]))
        self._spin_flush_factor.setSuffix(" ×")
        disp_lay.addRow("Precondition factor:", self._spin_flush_factor)

        self._lbl_liquid_correction = QLabel("Disabled")
        disp_lay.addRow("Liquid correction:", self._lbl_liquid_correction)
        # Dispense group added to second row below (not the top config row)

        layout.addLayout(config_row)

        # === Formulation matrix ===
        form_grp = QGroupBox("Formulation Matrix (channel × component volumes in µL)")
        form_layout = QVBoxLayout(form_grp)

        self._form_table = PasteableTableWidget()
        self._form_table.setColumnCount(6)
        self._form_table.setHorizontalHeaderLabels(
            ["", "Channel", "Pump 0 (µL)", "Pump 1 (µL)", "Pump 2 (µL)", "Total (µL)"]
        )
        # Rectangular multi-cell selection so a region can be copy/pasted.
        self._form_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._form_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self._form_table.setToolTip(
            "Copy/paste value cells with Ctrl+C / Ctrl+V — between regions or "
            "to/from a spreadsheet (Excel).")
        self._form_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        # Checkbox column should be narrow
        self._form_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._populate_formulation_table(16)
        # Compress the matrix slightly (scrolls past ~8 rows) so the config panels
        # above and the EIS/Dispense column beside it have breathing room.
        self._form_table.setMaximumHeight(320)
        form_layout.addWidget(self._form_table)

        # Channel selection helpers
        ch_sel_row = QHBoxLayout()
        self._btn_select_all = QPushButton("Select All")
        self._btn_select_all.clicked.connect(self._on_select_all)
        ch_sel_row.addWidget(self._btn_select_all)

        self._btn_deselect_all = QPushButton("Deselect All")
        self._btn_deselect_all.clicked.connect(self._on_deselect_all)
        ch_sel_row.addWidget(self._btn_deselect_all)

        ch_sel_row.addWidget(QLabel("Channels:"))
        self._edit_channels = QLineEdit()
        self._edit_channels.setPlaceholderText("e.g. 1,3,5-8,12")
        self._edit_channels.setToolTip(
            "Enter channels to select (comma-separated, ranges with dash). "
            "Press Enter to apply."
        )
        self._edit_channels.returnPressed.connect(self._on_channel_entry)
        ch_sel_row.addWidget(self._edit_channels)

        self._btn_apply_channels = QPushButton("Apply")
        self._btn_apply_channels.clicked.connect(self._on_channel_entry)
        ch_sel_row.addWidget(self._btn_apply_channels)

        self._btn_formulation = QPushButton("Formulation Manager...")
        self._btn_formulation.clicked.connect(self._on_open_formulation)
        ch_sel_row.addWidget(self._btn_formulation)

        ch_sel_row.addStretch()
        form_layout.addLayout(ch_sel_row)

        # === Second row: Formulation Matrix (left) + [EIS over Dispense] (right) ===
        # EIS and the Dispense (liquid-handling) panel share a right column; EIS is
        # given the stretch so it absorbs the slack the short Dispense form leaves.
        right_col = QWidget()
        right_col_lay = QVBoxLayout(right_col)
        right_col_lay.setContentsMargins(0, 0, 0, 0)
        right_col_lay.addWidget(eis_grp, 1)
        right_col_lay.addWidget(self._disp_grp, 0)

        second_row = QSplitter(Qt.Orientation.Horizontal)
        second_row.setHandleWidth(5)
        second_row.setChildrenCollapsible(False)
        second_row.addWidget(form_grp)
        second_row.addWidget(right_col)
        second_row.setStretchFactor(0, 3)
        second_row.setStretchFactor(1, 1)
        layout.addWidget(second_row)

        # === Workflow preview ===
        preview_grp = QGroupBox("Workflow Preview")
        preview_lay = QVBoxLayout(preview_grp)
        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setMaximumHeight(120)
        self._txt_preview.setStyleSheet("font-family: monospace; font-size: 11px;")
        preview_lay.addWidget(self._txt_preview)

        btn_preview_row = QHBoxLayout()
        self._btn_generate = QPushButton("Generate Workflow")
        self._btn_generate.clicked.connect(self._on_generate_workflow)
        btn_preview_row.addWidget(self._btn_generate)
        btn_preview_row.addStretch()
        preview_lay.addLayout(btn_preview_row)

        # === Campaign annotation ===
        notes_grp = QGroupBox("Campaign Notes (stored with run)")
        notes_lay = QVBoxLayout(notes_grp)
        self._te_annotation = QTextEdit()
        self._te_annotation.setPlaceholderText(
            "Brief description of this experiment campaign "
            "(PCB ID, formulation, purpose, etc.)\u2026"
        )
        self._te_annotation.setFixedHeight(56)
        notes_lay.addWidget(self._te_annotation)

        # === Third row: Workflow Preview (left) + Campaign Notes (right) ===
        third_row = QSplitter(Qt.Orientation.Horizontal)
        third_row.setHandleWidth(5)
        third_row.setChildrenCollapsible(False)
        third_row.addWidget(preview_grp)
        third_row.addWidget(notes_grp)
        third_row.setStretchFactor(0, 3)
        third_row.setStretchFactor(1, 2)
        layout.addWidget(third_row)

        # === Run controls ===
        run_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start")
        self._btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; padding: 8px;"
        )
        self._btn_start.clicked.connect(self._on_start)
        run_row.addWidget(self._btn_start)

        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)
        run_row.addWidget(self._btn_pause)

        self._btn_abort = QPushButton("⏹  Abort")
        self._btn_abort.setStyleSheet("background-color: #f44336; color: white;")
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._on_abort)
        run_row.addWidget(self._btn_abort)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        run_row.addWidget(self._progress)

        self._lbl_status = QLabel("Idle")
        run_row.addWidget(self._lbl_status)
        run_row.addStretch()
        layout.addLayout(run_row)

        # === Results table ===
        res_grp = QGroupBox("Results")
        res_layout = QVBoxLayout(res_grp)

        self._results_table = QTableWidget()
        self._results_table.setColumnCount(5)
        self._results_table.setHorizontalHeaderLabels(
            ["Channel", "Step", "Status", "Duration (s)", "Detail"]
        )
        self._results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        res_layout.addWidget(self._results_table)

        export_row = QHBoxLayout()
        self._btn_csv = QPushButton("Save CSV")
        self._btn_csv.clicked.connect(self._on_save_csv)
        export_row.addWidget(self._btn_csv)

        self._btn_save_eis = QPushButton("Save EIS Data")
        self._btn_save_eis.setToolTip(
            "Save each EIS measurement as a separate file\n"
            "in Z'/Z''/phase format (compatible with Analysis tab)"
        )
        self._btn_save_eis.clicked.connect(self._on_save_eis_data)
        export_row.addWidget(self._btn_save_eis)

        # Which equivalent circuit "Fit All EIS" fits with. Defaulting by *name*
        # rather than by dict position: the order of CIRCUIT_MODELS is not a
        # design decision, and reordering it must not move the operator's model.
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        export_row.addWidget(QLabel("Model:"))
        self._combo_fit_model = QComboBox()
        for name, info in CIRCUIT_MODELS.items():
            self._combo_fit_model.addItem(f"{name} — {info['description']}", userData=name)
        default_row = self._combo_fit_model.findData(self.DEFAULT_FIT_MODEL)
        if default_row >= 0:
            self._combo_fit_model.setCurrentIndex(default_row)
        export_row.addWidget(self._combo_fit_model)

        self._btn_fit_all = QPushButton("Fit All EIS")
        self._btn_fit_all.setToolTip("Run circuit fitting on all captured EIS data")
        self._btn_fit_all.clicked.connect(self._on_fit_all_eis)
        export_row.addWidget(self._btn_fit_all)

        self._btn_pdf = QPushButton("Save PDF Report")
        self._btn_pdf.clicked.connect(self._on_save_pdf)
        export_row.addWidget(self._btn_pdf)
        export_row.addStretch()
        res_layout.addLayout(export_row)

        # Fit output gets its own pane. It used to be dumped into the Workflow
        # Preview box, so fitting destroyed the preview the operator was reading.
        self._txt_fit_output = QPlainTextEdit()
        self._txt_fit_output.setReadOnly(True)
        self._txt_fit_output.setMaximumHeight(120)
        self._txt_fit_output.setPlaceholderText("Circuit fit results appear here…")
        self._txt_fit_output.setStyleSheet("font-family: monospace; font-size: 11px;")
        res_layout.addWidget(self._txt_fit_output)

        layout.addWidget(res_grp)

        # Trigger initial PCB info display
        if self._combo_pcb.count() > 0:
            self._on_pcb_changed(0)
        self._set_liquid_correction_status(bool(liquid_handling_config().get("enabled", False)))

    # ── Signal wiring ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect thread-safe signals to UI update slots."""
        self._sig_step_start.connect(self._ui_step_start)
        self._sig_step_complete.connect(self._ui_step_complete)
        self._sig_step_error.connect(self._ui_step_error)
        self._sig_step_skipped.connect(self._ui_step_skipped)
        self._sig_state_change.connect(self._ui_state_change)
        self._sig_workflow_done.connect(self._ui_workflow_done)
        self._sig_channel_hold.connect(self._ui_channel_hold)

    # ── Formulation table ────────────────────────────────────────────────

    def _populate_formulation_table(self, n_channels: int) -> None:
        """Fill the formulation table with *n_channels* rows (checkbox + channel + 3 pump cols + total)."""
        self._form_table.setRowCount(n_channels)
        self._chk_anchor_row = None  # rows were rebuilt — drop any stale anchor
        for r in range(n_channels):
            # Col 0: checkbox for channel selection
            chk = QCheckBox()
            chk.setChecked(True)
            # Shift-click fills the range from the last plainly-clicked checkbox.
            chk.clicked.connect(lambda _checked, row=r: self._on_channel_check_clicked(row))
            self._form_table.setCellWidget(r, 0, chk)

            # Col 1: channel number (read-only)
            ch_item = QTableWidgetItem(str(r + 1))
            ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._form_table.setItem(r, 1, ch_item)

            # Cols 2–5: Pump 0 / Pump 1 / Pump 2 / Total volume entries
            for c in range(2, 6):
                item = QTableWidgetItem("0.0")
                self._form_table.setItem(r, c, item)

    def _on_pcb_changed(self, index: int) -> None:
        """Update channel count and info label when PCB selection changes."""
        pcb_name = self._combo_pcb.itemData(index)
        if pcb_name is None:
            return
        pcb = self._pcb_data.get(pcb_name, {})
        channels = pcb.get("channels", 16)
        grid = pcb.get("grid", [])
        self._lbl_pcb_info.setText(
            f"Channels: {channels}  Grid: {grid}  "
            f"Spacing: {pcb.get('spacing_mm', '?')} mm"
        )
        self._populate_formulation_table(channels)

    def _on_mode_changed(self, index: int) -> None:
        """Toggle Dispense controls visibility based on workflow mode."""
        # 0 = Full Protocol, 1 = Measure Only, 2 = Formulate Only.
        # Dispense controls are relevant whenever we dispense — i.e. every
        # mode except Measure Only.
        self._disp_grp.setVisible(index != 1)

    def _selected_channels(self) -> list[int]:
        """Return 0-based indices of checked channels in the formulation table."""
        selected: list[int] = []
        for r in range(self._form_table.rowCount()):
            chk = self._form_table.cellWidget(r, 0)
            if isinstance(chk, QCheckBox) and chk.isChecked():
                selected.append(r)
        return selected

    def _on_select_all(self) -> None:
        """Check all channel checkboxes."""
        for r in range(self._form_table.rowCount()):
            chk = self._form_table.cellWidget(r, 0)
            if isinstance(chk, QCheckBox):
                chk.setChecked(True)

    def _on_deselect_all(self) -> None:
        """Uncheck all channel checkboxes."""
        for r in range(self._form_table.rowCount()):
            chk = self._form_table.cellWidget(r, 0)
            if isinstance(chk, QCheckBox):
                chk.setChecked(False)

    def _on_channel_check_clicked(self, row: int) -> None:
        """Shift-click fills the range from the anchor with its state; else set anchor."""
        from PySide6.QtWidgets import QApplication

        shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        anchor = getattr(self, "_chk_anchor_row", None)
        if shift and anchor is not None and anchor != row:
            anchor_chk = self._form_table.cellWidget(anchor, 0)
            if isinstance(anchor_chk, QCheckBox):
                state = anchor_chk.isChecked()
                lo, hi = sorted((anchor, row))
                for r in range(lo, hi + 1):
                    chk = self._form_table.cellWidget(r, 0)
                    if isinstance(chk, QCheckBox):
                        chk.setChecked(state)
            return  # keep the anchor so the range can be re-extended
        self._chk_anchor_row = row

    @staticmethod
    def _parse_channel_spec(spec: str, max_ch: int) -> list[int]:
        """Parse a comma-separated channel specification into 1-based channel numbers.

        Supports individual channels and ranges::

            "1,3,5-8,12"  →  [1, 3, 5, 6, 7, 8, 12]

        Delegates to the one shared parser.  This field is deliberately
        *silent-drop* — bad tokens and out-of-bounds singles are ignored,
        ranges are clamped to ``[1, max_ch]`` — because the result drives a
        checkbox selection, where "nothing got selected" is visible on screen.
        """
        from softae.core.channel_spec import parse_channel_spec

        return parse_channel_spec(spec, max_ch=max_ch, on_invalid="drop")

    def _on_channel_entry(self) -> None:
        """Apply the comma-delimited channel field to the checkboxes."""
        spec = self._edit_channels.text().strip()
        if not spec:
            return
        n_rows = self._form_table.rowCount()
        selected_set = set(self._parse_channel_spec(spec, n_rows))
        for r in range(n_rows):
            chk = self._form_table.cellWidget(r, 0)
            if isinstance(chk, QCheckBox):
                chk.setChecked((r + 1) in selected_set)  # r is 0-based, spec is 1-based

    def open_formulation_manager(self) -> None:
        """Open the FormulationPanel; wire volume-calc fill AND catalog-change re-emit.

        Single shared open path: the tab button and any external trigger (e.g. the
        main-window Catalogs menu) both route here.
        """
        from softae.gui.widgets.formulation_panel import FormulationPanel

        dlg = FormulationPanel(self)
        dlg.volumes_calculated.connect(self._on_formulation_calculated)
        dlg.catalogs_changed.connect(self.catalogs_changed)  # re-emit upward
        dlg.exec()

    def _on_open_formulation(self) -> None:
        """Existing button slot → delegate to the shared opener."""
        self.open_formulation_manager()

    def _on_formulation_calculated(self, volumes: list[float]) -> None:
        """Fill the formulation table with calculated volumes."""
        for row in range(self._form_table.rowCount()):
            for col_offset, val in enumerate(volumes):
                item = QTableWidgetItem(f"{val:.2f}")
                self._form_table.setItem(row, col_offset + 2, item)

    # ── Workflow generation ──────────────────────────────────────────────

    def _read_formulation_matrix(self) -> list[list[float]]:
        """Read the volume matrix from the table (columns 2–5: p0, p1, p2, total)."""
        rows = self._form_table.rowCount()
        matrix: list[list[float]] = []
        for r in range(rows):
            row_vals: list[float] = []
            for c in range(2, 6):
                item = self._form_table.item(r, c)
                try:
                    row_vals.append(float(item.text()) if item else 0.0)
                except ValueError:
                    row_vals.append(0.0)
            matrix.append(row_vals)
        return matrix

    def _on_eis_preset_changed(self, name: str) -> None:
        """Populate EIS parameter spinboxes from the selected preset."""
        p = eis_presets().get(name, {})
        if not p:
            return
        self._spin_eis_f_hi.setValue(p.get("f_hi", 200_000))
        self._spin_eis_f_lo.setValue(p.get("f_lo_mHz", 4_000))
        self._spin_eis_npts.setValue(p.get("npts", 35))
        self._spin_eis_mv_ac.setValue(p.get("mv_ac", 10))
        self._spin_eis_mv_dc.setValue(p.get("mv_dc", 0))

    def _build_dispense_plan(
        self,
        selected: list[int],
        vol_master: list[list[float]],
    ) -> tuple[list[dict[str, Any]], "DeadVolumeCorrection", dict[int, float]]:
        """Return per-channel dispense plan from matrix (+ optional correction).

        The ``commanded_uL`` figures here are for display and metadata only —
        the volumes the hardware actually receives are corrected once, inside
        the marshaller (P2.2). Both use the same
        :class:`DeadVolumeCorrection`, so the two cannot disagree.
        """
        cfg = liquid_handling_config()
        correction = DeadVolumeCorrection.from_config(self.PUMP_IDS, cfg)
        correction_enabled = correction.enabled
        sys_cfg = correction.sys_cfg
        line_cfg_by_pump = correction.line_cfg_by_pump
        corrector = LiquidHandlingCorrector()
        prime_by_pump = {
            pump_id: corrector.prime_volume(line_cfg, sys_cfg)
            for pump_id, line_cfg in line_cfg_by_pump.items()
        }

        n_pumps = len(self.PUMP_IDS)
        plan: list[dict[str, Any]] = []
        for run_index, row_idx in enumerate(selected, start=1):
            row = vol_master[row_idx] if row_idx < len(vol_master) else []
            # Per-pump targets (Pump 0..N-1), then the trailing Total column.
            targets = [
                max(0.0, float(row[p] if len(row) > p else 0.0))
                for p in range(n_pumps)
            ]
            total_target = max(
                0.0, float(row[n_pumps] if len(row) > n_pumps else 0.0)
            )
            if total_target <= 0.0 and any(t > 0.0 for t in targets):
                total_target = sum(targets)

            commanded = correction.commanded(targets, run_index=run_index)
            total_cmd = sum(commanded)

            entry: dict[str, Any] = {
                "channel": row_idx + 1,
                "run_index": run_index,
                "total_target_uL": total_target,
                "total_commanded_uL": total_cmd,
                # General N-pump lists (source of truth for the deposit paths).
                "targets_uL": list(targets),
                "commanded_uL": list(commanded),
            }
            # Flat per-pump keys (pump0_*, pump1_*, …) for back-compat consumers.
            for p in range(n_pumps):
                entry[f"pump{p}_target_uL"] = targets[p]
                entry[f"pump{p}_commanded_uL"] = commanded[p]
            plan.append(entry)
        return plan, correction, prime_by_pump

    def _set_liquid_correction_status(self, enabled: bool) -> None:
        if enabled:
            self._lbl_liquid_correction.setText("Enabled")
            self._lbl_liquid_correction.setStyleSheet("color: #2e7d32; font-weight: bold;")
        else:
            self._lbl_liquid_correction.setText("Disabled")
            self._lbl_liquid_correction.setStyleSheet("color: #616161; font-weight: bold;")

    # ── Recipe-engine deposit-method selector (Process Studio → HT) ──────────

    def _deposit_method_names(self) -> list[str]:
        """Catalog ``liquid_handler`` methods that accept electrode injection.

        A method qualifies if its params carry the engine's electrode + volume
        slot keys (``x``/``y``/``vols``) — i.e. it's a per-channel deposit method
        the recipe engine can drive.  These are exactly the deposit methods you
        author/port/mature in Process Studio.
        """
        from softae.core.deposition_recipe import DepositionSlots

        slots = DepositionSlots()
        keys = {slots.electrode_x, slots.electrode_y, slots.volumes}
        return [
            n for n in self._task_catalog.list_names()
            if (t := self._task_catalog.get(n)).instrument == "liquid_handler"
            and keys.issubset(t.params.keys())
        ]

    def _populate_deposit_method_combo(self) -> None:
        combo = self._combo_deposit_method
        combo.blockSignals(True)
        combo.clear()
        for n in self._deposit_method_names():
            combo.addItem(n, userData=n)
        idx = combo.findData("single_drop_simul")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)
        self._refresh_deposit_maturity_label()

    def _selected_deposit_method(self) -> str:
        return self._combo_deposit_method.currentData() or "single_drop_simul"

    # ── Deposition-recipe selector (Legacy | engine recipes) ─────────────────

    def _populate_deposit_recipe_combo(self) -> None:
        """Fill the recipe selector with the catalogued deposition recipes.

        The initial selection comes from ``[dropcast].default_recipe`` (falling
        back to the first available recipe if the configured name is unknown —
        e.g. a stale ``legacy`` value after the legacy path was retired).
        """
        from softae.core.deposition_recipe import (
            deposition_recipe_names, get_deposition_recipe,
        )

        names = deposition_recipe_names()
        combo = self._combo_deposit_recipe
        combo.blockSignals(True)
        combo.clear()
        for n in names:
            combo.addItem(get_deposition_recipe(n).label, userData=n)
        default = dropcast_config().get("default_recipe")
        if default not in names:
            default = "two_phase" if "two_phase" in names else (names[0] if names else None)
        idx = combo.findData(default)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _selected_recipe(self) -> str:
        return self._combo_deposit_recipe.currentData() or "single_drop"

    def _on_recipe_changed(self, *_: Any) -> None:
        """Refresh the deposit-method maturity readout for the selected recipe."""
        self._refresh_deposit_maturity_label()

    def _refresh_deposit_maturity_label(self, *_: Any) -> None:
        lbl = getattr(self, "_lbl_deposit_maturity", None)
        if lbl is None:
            return
        name = self._combo_deposit_method.currentData()
        if not name or name not in self._task_catalog:
            lbl.setText("")
            return
        from softae.core.lifecycle import Maturity, method_maturity

        m = method_maturity(name, self._task_catalog)
        if m >= Maturity.VALIDATED:
            lbl.setText(f"● {m.label}")
            lbl.setStyleSheet("color: #16a34a;")
        else:
            lbl.setText(f"⚠ {m.label}")
            lbl.setStyleSheet("color: #d97706;")

    def _engine_recipe_maturity(self, recipe_name: str):
        """Effective maturity of an engine recipe: min over its methods.

        A deposition recipe is only as mature as its least-mature method (its
        startup/precondition/deposit/final methods, with the deposit phase using
        the selected deposit-method override).  Returns a ``Maturity`` or ``None``
        if none of the methods are catalogued.
        """
        from softae.core.deposition_recipe import get_deposition_recipe
        from softae.core.lifecycle import method_maturity

        cat = self._task_catalog
        try:
            recipe = get_deposition_recipe(recipe_name)
        except KeyError:
            return None
        names = set(recipe.method_deps())
        # The deposit phase's method may be overridden by the selector.
        dep = recipe.deposit_phase()
        if dep is not None:
            names.discard(dep.method)
            names.add(self._selected_deposit_method())
        maturities = [method_maturity(n, cat) for n in names if n in cat]
        return min(maturities) if maturities else None

    def _active_pcb_config(self) -> dict[str, Any]:
        """The currently selected PCB layout dict (grid/spacing/…)."""
        name = self._combo_pcb.currentData()
        return self._pcb_data.get(name, {}) if name else {}

    def _piezo_plan(self):
        """Build a :class:`PiezoPlan` from the ``[piezo]`` config.

        Disabled unless piezo is on, liquid-events are on, and channel A is
        selected — matching the legacy enablement rule.  A ``liquid_event_profile``
        settings source adds the one-shot event step in setup.
        """
        from softae.core.deposition_recipe import PiezoPlan

        cfg = piezo_config()
        events = cfg.get("liquid_events", {})
        if not isinstance(events, dict):
            events = {}
        enabled = (
            bool(cfg.get("enabled", False))
            and bool(events.get("enabled", False))
            and bool(events.get("channel_a", True))
        )
        source = str(events.get("settings_source", "manual_profile"))
        event_task = (
            "piezo_liquid_event" if (enabled and source == "liquid_event_profile") else None
        )
        event_params = None
        if event_task:
            event_params = {
                "frequency_hz": int(events.get("frequency_hz", 500)),
                "on_s": float(events.get("sweep_on_s", 2.0)),
                "rest_s": float(events.get("sweep_rest_s", 3.0)),
            }
        # Actuate the piezo around every elution event (startup flush, precondition,
        # deposit, final flush) by default; ``[piezo.liquid_events] all_elution=false``
        # reverts to bracketing the deposit phase only.
        elution_scope = "all_elution" if events.get("all_elution", True) else "deposit"
        return PiezoPlan(enabled=enabled, event_task=event_task, event_params=event_params,
                         elution_scope=elution_scope)

    def _build_run_plan(self, formulate_only: bool):
        """Assemble a :class:`~softae.core.run_plan.RunPlan` from the UI controls.

        Measurement is present unless in formulate-only mode; the "Insert anneal"
        toggle adds an anneal phase (at the chosen temperature/hold); the
        measurement-scope combo selects pointwise (interleaved per channel) vs
        batch (formulate-all → anneal-all → measure-all) ordering.
        """
        from softae.core.run_plan import RunPlan

        anneal = self._chk_anneal.isChecked()
        anneal_params = None
        if anneal:
            anneal_params = {
                "target_temp_C": float(self._spin_anneal_temp.value()),
                "hold_time_s": float(self._spin_anneal_hold.value()) * 60.0,
            }
        factory = RunPlan.batch if self._combo_measure_scope.currentIndex() == 1 else RunPlan.pointwise
        return factory(measure=not formulate_only, anneal=anneal, anneal_params=anneal_params)

    def _update_sequence_preview(self, *_args) -> None:
        """Refresh the read-only sequence preview so the operator can see the phases."""
        try:
            plan = self._build_run_plan(formulate_only=False)
            self._lbl_sequence.setText("Sequence:  " + plan.describe())
        except Exception:
            self._lbl_sequence.setText("")

    def _build_engine_workflow(
        self,
        recipe_name: str,
        dispense_plan: list[dict[str, Any]],
        eis_steps: list[WorkflowStep],
        *,
        formulate_only: bool,
        eis_preset: str,
        selected: list[int],
        correction: "DeadVolumeCorrection",
        prime_by_pump: dict[int, float] | None = None,
    ) -> Workflow:
        """Build a deposition workflow by running the selected recipe through the engine.

        The single deposition path: the :class:`DepositionRecipe` declares the
        per-channel phase sequence and the engine injects the electrode, per-channel
        (dead-volume-corrected) volumes, proportional-rate splits, derived settling
        wait, and — when ``[piezo]`` liquid-events are enabled — the per-channel
        piezo actuation around each deposit.  EIS steps (with mscr routing) are
        reused from the standard path.
        """
        from softae.core.deposition_recipe import (
            DepositionSettings, build_deposition_workflow, get_deposition_recipe,
        )

        cat = self._task_catalog
        recipe = get_deposition_recipe(recipe_name)
        dep_method = self._selected_deposit_method()
        # Validate the methods the engine will actually run (with the deposit-phase
        # override) are catalogued.
        needed = set(recipe.method_deps())
        dep_phase = recipe.deposit_phase()
        if dep_phase is not None:
            needed.discard(dep_phase.method)
            needed.add(dep_method)
        missing = [m for m in sorted(needed) if m not in cat]
        if missing:
            raise ValueError(
                f"Deposition recipe '{recipe.name}': missing method(s) "
                f"{missing} in the catalog.")

        dispense_rate = float(self._spin_rate.value())
        flush_rate = float(self._spin_flush_rate.value())
        flush_factor = float(self._spin_flush_factor.value())
        dc = dropcast_config()
        settle_factor = float(dc.get("settle_factor", self.SETTLE_FACTOR_DEFAULT))
        settle_base_s = float(dc.get("settle_base_s", 0.0))

        # Pass *delivered* targets, not the plan's commanded volumes: correction
        # now happens once, inside the marshaller (P2.2). Feeding the corrected
        # figures here would apply dead volume twice.
        formulation_by_channel = {
            int(p["channel"]): [float(v) for v in p["targets_uL"]]
            for p in dispense_plan
        }
        eis_by_channel = (
            None if formulate_only
            else {int(s.params.get("chan", 0)): s for s in eis_steps}
        )

        piezo = self._piezo_plan()

        # The shared marshalling contract (P2.1) — the same type the campaign
        # path builds from its spec, so the two surfaces cannot drift again.
        settings = DepositionSettings(
            pump_ids=tuple(self.PUMP_IDS),
            dispense_rate=dispense_rate,
            flush_rate=flush_rate,
            flush_factor=flush_factor,
            settle_factor=settle_factor,
            settle_base_s=settle_base_s,
            start_flush_uL=tuple(self._start_flush_volumes() or ()),
            deposit_method=dep_method,
            piezo=piezo,
            pcb=self._active_pcb_config(),
            run_plan=self._build_run_plan(formulate_only),
            # The *same* instance the dispense plan was built from — reading the
            # config twice is how the two could disagree.
            correction=correction,
        )

        wf = build_deposition_workflow(
            recipe,
            [int(p["channel"]) for p in dispense_plan],
            formulation_by_channel,
            settings=settings,
            catalog=cat,
            eis_step_by_channel=eis_by_channel,
            name=f"ht_{recipe.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        wf.variables = {"eis_preset": eis_preset, "selected_channels": selected}
        wf.metadata = {
            **wf.metadata,
            "source": "deposition_engine",
            "recipe": recipe.name,
            "mode": "formulate_only" if formulate_only else "full",
            "pcb": self._combo_pcb.currentData(),
            "liquid_handling_enabled": correction.enabled,
            "dispense_plan": dispense_plan,
            "prime_by_pump": prime_by_pump or {},
            "dispense_rate_uL_min": dispense_rate,
            "line_flush_rate_uL_min": flush_rate,
            "flush_factor": flush_factor,
            "settle_factor": settle_factor,
            "piezo": "applied" if piezo.enabled else "not_applied",
        }
        return wf

    # ── Start-flush helper (used by the deposition engine) ───────────────────

    def _start_flush_volumes(self) -> list[float]:
        """Per-pump start-flush volumes parsed from the UI (length == PUMP_IDS).

        Accepts a comma/space separated list; a single value broadcasts to every
        pump; a short list is zero-padded and a long one truncated; non-numeric
        tokens become ``0.0``.
        """
        n = len(self.PUMP_IDS)
        text = self._edit_start_flush.text().strip()
        if not text:
            return [0.0] * n
        vals: list[float] = []
        for tok in text.replace(",", " ").split():
            try:
                vals.append(max(0.0, float(tok)))
            except ValueError:
                vals.append(0.0)
        if len(vals) == 1:
            return vals * n
        if len(vals) < n:
            vals += [0.0] * (n - len(vals))
        return vals[:n]

    def _eis_timeout_retry(self) -> tuple[float, int]:
        """(timeout_s, retry) for EIS steps, from the ``measure_eis`` task or defaults."""
        if "measure_eis" in self._task_catalog:
            t = self._task_catalog.get("measure_eis")
            return (t.timeout_s if t.timeout_s is not None else 600, t.retry or 1)
        return (600, 1)

    def _generate_workflow(self) -> Workflow:
        """Build a :class:`Workflow` from the current UI configuration.

        Supports three modes:
        - **Full Protocol**: setup flush → per-channel (precondition +
          deposit + EIS measure) → teardown flush.
        - **Measure Only**: EIS measure on selected channels only (no
          syringe operations).
        - **Formulate Only**: setup flush → per-channel deposit → teardown
          flush, with NO EIS measurement (formulation/dispense only).

        Only channels whose checkbox is ticked are included.
        """
        selected = self._selected_channels()
        if not selected:
            raise ValueError("No channels selected — tick at least one channel.")

        vol_master = self._read_formulation_matrix()
        eis_preset = self._combo_eis.currentText()
        mode_index = self._combo_mode.currentIndex()
        measure_only = mode_index == 1
        formulate_only = mode_index == 2

        dispense_plan, correction, prime_by_pump = self._build_dispense_plan(selected, vol_master)
        self._set_liquid_correction_status(correction.enabled)
        # Stash for the run-start overflow guard (see _overflow_check).
        self._last_dispense_plan = dispense_plan

        # == Build per-channel EIS steps with automatic pico routing ==
        # Each selected channel maps to the correct pico via config.
        # Channel numbers are 1-based in the table.
        channel_numbers = [s + 1 for s in selected]  # 0-based index → 1-based ch

        # Formulate-only mode dispenses without measuring, so no EIS steps.
        eis_timeout, eis_retry = self._eis_timeout_retry()
        eis_steps: list[WorkflowStep] = []
        for ch_1based in channel_numbers if not formulate_only else []:
            pico_name = pico_for_channel(ch_1based)
            # Each channel gets its own .mscr file with the GPIO mux address
            # baked in.  eis_run_mscrbuild (called in _on_start before the
            # thread launches) handles pico2 remapping internally via
            # mod_channel_restart, exactly as the Manual tab does.
            mscr_path = os.path.join(
                tempfile.gettempdir(), f"softae_ch{ch_1based}.mscr"
            )
            eis_steps.append(
                WorkflowStep(
                    name=f"measure_eis_ch{ch_1based}",
                    instrument=pico_name,
                    method="sendscript_getdata",
                    params={
                        "mscrpath": mscr_path,
                        "outdir": os.path.join(
                            tempfile.gettempdir(), "softae_eis_output"
                        ),
                        # Full 1-based channel (1–32) used for logging,
                        # matching the Manual tab's sendscript_getdata call.
                        "chan": ch_1based,
                    },
                    timeout_s=eis_timeout,
                    retry=eis_retry,
                )
            )

        # Measure-only: EIS on the selected channels, no deposition.
        if measure_only:
            return self._build_measure_only_workflow(
                eis_steps, selected=selected, eis_preset=eis_preset)

        # All deposition runs go through the recipe engine (single_drop / two_phase).
        return self._build_engine_workflow(
            self._selected_recipe(), dispense_plan, eis_steps,
            formulate_only=formulate_only, eis_preset=eis_preset,
            selected=selected, correction=correction,
            prime_by_pump=prime_by_pump)

    def _build_measure_only_workflow(
        self, eis_steps: list[WorkflowStep], *, selected: list[int], eis_preset: str,
    ) -> Workflow:
        """Measure-only workflow: per-channel EIS flattened into setup, no syringe ops.

        Each channel has its own step with a unique mscr/chan, so they are
        flattened (the loop mechanism templates a single step, which would give
        the wrong channel for iterations > 1).
        """
        channel_numbers = [s + 1 for s in selected]
        setup_steps = [
            step.with_tags(channel=str(step.params.get("chan", "")))
            for step in eis_steps
        ]
        return Workflow(
            name=f"ht_measure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Measure-only {eis_preset} EIS on {len(selected)} channels",
            variables={
                "eis_preset": eis_preset,
                "selected_channels": selected,
                "channel_numbers": channel_numbers,
            },
            setup=setup_steps,
            iterations=1,
            metadata={
                "pcb": self._combo_pcb.currentData(),
                "mode": "measure_only",
                "channel_routing": {
                    ch: pico_for_channel(ch) for ch in channel_numbers
                },
            },
        )

    def _on_generate_workflow(self) -> None:
        """Preview the workflow in the text area (read-only YAML-like view)."""
        try:
            wf = self._generate_workflow()
            correction_enabled = bool(wf.metadata.get("liquid_handling_enabled", False))
            correction_text = "enabled" if correction_enabled else "disabled"
            lines = [
                f"name: {wf.name}",
                f"description: {wf.description}",
                f"iterations: {wf.iterations}",
                f"total_steps: {wf.total_steps}",
                f"liquid_correction: {correction_text}",
                "",
                "Resolved steps:",
            ]
            for i, step in enumerate(wf.resolve_steps()):
                lines.append(
                    f"  {i + 1}. {step.name} -> {step.instrument}.{step.method}()"
                )

            dispense_plan = wf.metadata.get("dispense_plan", [])
            prime_by_pump = wf.metadata.get("prime_by_pump", {})
            if isinstance(dispense_plan, list) and dispense_plan:
                lines.append("")
                lines.append("Dispense plan sample:")
                for row in dispense_plan[:3]:
                    ch = int(row.get("channel", 0))
                    targets = row.get("targets_uL") or []
                    commanded = row.get("commanded_uL") or []
                    for pump_id in self.PUMP_IDS:
                        t = float(targets[pump_id]) if pump_id < len(targets) else 0.0
                        c = float(commanded[pump_id]) if pump_id < len(commanded) else 0.0
                        lines.append(
                            f"  ch{ch} p{pump_id}: target={t:.2f} uL, commanded={c:.2f} uL"
                        )
                if prime_by_pump:
                    lines.append("  prime estimates:")
                    for pump_id in sorted(prime_by_pump):
                        line_id = liquid_line_for_pump(pump_id)
                        lines.append(
                            f"    line {line_id} (pump {pump_id}): {float(prime_by_pump[pump_id]):.2f} uL"
                        )
                if not correction_enabled:
                    lines.append("  correction disabled: commanded == target")
            if wf.metadata.get("source") == "deposition_engine":
                recipe = str(wf.metadata.get("recipe", ""))
                lines.append("")
                lines.append(
                    f"Deposition engine · recipe '{recipe}': "
                    f"dispense={wf.metadata.get('dispense_rate_uL_min')} µL/min, "
                    f"line flush={wf.metadata.get('line_flush_rate_uL_min')} µL/min, "
                    f"precondition ×{wf.metadata.get('flush_factor')}"
                )
                from softae.core.lifecycle import Maturity
                m = self._engine_recipe_maturity(recipe)
                if m is not None:
                    flag = "" if m >= Maturity.VALIDATED else "  (not validated)"
                    lines.append(f"  recipe maturity: {m.label}{flag}")
            self._txt_preview.setPlainText("\n".join(lines))
        except Exception as exc:
            self._txt_preview.setPlainText(f"Error: {exc}")

    # ── Run controls ─────────────────────────────────────────────────────

    def _overflow_check(self, dispense_plan: list[dict[str, Any]]) -> bool:
        """Warn if any channel's commanded cast volume exceeds the well capacity.

        The HT case of the shared overflow guard (:mod:`softae.core.overflow`):
        the simple "sum of component volumes vs the board's per-well capacity"
        test.  Returns ``True`` to proceed — no overflow, the board declares no
        capacity, or the user chose to continue — and ``False`` to cancel.
        """
        from softae.core.geometry import well_capacity_uL
        from softae.core.overflow import well_overflow

        capacity = well_capacity_uL(self._active_pcb_config())
        if not capacity:  # board declares no cap → nothing to enforce
            return True
        over = []
        for entry in dispense_plan or []:
            verdict = well_overflow(float(entry.get("total_commanded_uL", 0.0)), capacity)
            if verdict.overflows:
                over.append((int(entry.get("channel", 0)), verdict))
        if not over:
            return True
        lines = "\n".join(
            f"  • channel {ch}: {v.total_uL:.1f} µL ({-v.headroom_uL:.1f} µL over)"
            for ch, v in over
        )
        resp = QMessageBox.warning(
            self, "Overflow warning",
            f"{len(over)} channel(s) exceed the {capacity:.1f} µL well capacity:\n\n"
            f"{lines}\n\n"
            f"Reduce those channels' component volumes. Proceed anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return resp == QMessageBox.StandardButton.Yes

    # ── Predicted dry thickness (P.12) ───────────────────────────────────

    def _cast_stocks(self):
        """``(stocks, pump_assignment, chem_catalog)`` for what is on the pumps.

        ``None`` when nothing is declared. The loadout is project-scoped, so a
        fresh project starts empty — which is precisely the "nobody has said what
        is in the syringes" state, and no thickness can be attributed to it.

        Two loads, not one: :func:`stocks_from_loadout` returns
        :class:`Solution` objects only, and the elution split needs the
        *chemical* catalog to convert component quantities into volumes.
        """
        from softae.core import stock_assignment
        from softae.core.composition_axes import stocks_from_loadout

        loadout = stock_assignment.load_loadout(self._data_store)
        if loadout.is_empty():
            logger.info("ht_thickness_no_loadout",
                        msg="declare the pump stocks to get a predicted thickness")
            return None
        chem_catalog, sol_catalog = stock_assignment.catalogs_from_data_root()
        stocks, pump_assignment = stocks_from_loadout(loadout, sol_catalog)
        if not stocks:
            return None
        self._uncounted_particulates(stocks, chem_catalog)
        return stocks, pump_assignment, chem_catalog

    def _uncounted_particulates(self, stocks, chem_catalog) -> list[tuple[str, str]]:
        """``(stock, chemical)`` for particulates that will *not* count as deposit.

        ``is_particulate`` and the deposit roles are **separate axes**:
        ``is_particulate`` governs molar-mass availability and anti-clog purge
        routing, while what remains in the dry film is decided by ``role`` /
        ``counts_as_deposit``. So a catalogue row that marks a solid via
        ``is_particulate`` alone, leaving ``role`` blank, contributes nothing to
        the predicted thickness — too *thin*, which is the safe direction, but
        still wrong.

        Reported, never silently absorbed: a thickness that is quietly short by
        the whole silica loading is the hardest kind of wrong to find later.
        """
        from softae.core.formulation import deposited_component_names

        flagged: list[tuple[str, str]] = []
        for stock_name, sol in stocks.items():
            counted = set(deposited_component_names(sol))
            for comp in getattr(sol, "components", []) or []:
                if comp.chemical_name in counted:
                    continue
                try:
                    chem = chem_catalog.get(comp.chemical_name)
                except Exception:
                    continue
                if chem is not None and getattr(chem, "is_particulate", False):
                    flagged.append((stock_name, comp.chemical_name))
        if flagged:
            logger.warning(
                "ht_particulate_not_counted_as_deposit", components=flagged,
                msg="marked particulate but carries no deposit role — the "
                    "predicted thickness omits it; set its role or "
                    "counts_as_deposit in the catalogue",
            )
        return flagged

    def _cast_deposit_area_mm2(self) -> float | None:
        """The area this board's cast covers, or ``None`` if it declares none.

        Recorded beside the thickness it divided, whether or not the twin ran —
        a thickness is a quotient and the row must carry its denominator (P.7).
        """
        from softae.core.deposition_steps import resolve_pcb
        from softae.core.geometry import deposit_area_mm2

        try:
            area = deposit_area_mm2(resolve_pcb(self._combo_pcb.currentData())[1])
        except Exception:
            logger.warning("ht_deposit_area_unresolved", exc_info=True)
            return None
        return float(area) if area else None

    def _twin_thickness_um(self, channel_plan: dict[str, Any], context) -> float | None:
        """The twin's dry-film thickness (µm) for one channel, or ``None``.

        The **forward** direction of the same engine the BO loop runs in reverse:
        there a composition target is solved into volumes, here the volumes are
        already typed into the matrix. Both end in
        :func:`softae.core.autonomous_wiring.simulate_cast`, so HT and campaign
        casts cannot disagree about area, capacity, drying, or the elution split.
        """
        from softae.core.autonomous_wiring import simulate_cast

        if context is None:
            return None
        stocks, pump_assignment, chem_catalog = context

        per_pump_uL = {
            p: float(channel_plan.get(f"pump{p}_commanded_uL", 0.0) or 0.0)
            for p in self.PUMP_IDS
        }
        # `stocks_from_loadout` *skips* a pump whose stock the catalog does not
        # contain. For a composition search that is a warning; for a thickness it
        # silently deletes part of the film and reports a confident, too-thin
        # number — so refuse rather than inherit the skip. An undeclared pump that
        # is nonetheless casting lands in the same bucket for the same reason.
        unattributed = sorted(p for p, v in per_pump_uL.items()
                              if v > 0 and p not in set(pump_assignment.values()))
        if unattributed:
            logger.warning(
                "ht_thickness_unattributable", pumps=unattributed,
                msg="pump casts volume but its stock is undeclared or absent "
                    "from the catalog — refusing to predict a partial film",
            )
            return None

        twin = simulate_cast(
            {name: per_pump_uL.get(pump, 0.0)
             for name, pump in pump_assignment.items()},
            stocks, chem_catalog,
            pcb_name=self._combo_pcb.currentData(),
        )
        if twin is None or twin.final_thickness_um <= 0:
            # A pure-carrier stock dries to nothing, and σ divides by t. Zero is
            # not a thin film, it is the absence of one.
            return None
        return float(twin.final_thickness_um)

    def _predicted_cast(
        self, channel_plan: dict[str, Any], context
    ) -> tuple[float | None, float | None, str]:
        """``(thickness_um, area_mm2, method)`` to record for one channel.

        ``'unavailable'`` rather than NULL on a decline: the twin was asked and
        had nothing to say, which is itself a recorded fact. Never a guessed
        value — P.11 refuses an unattributable thickness downstream anyway.

        Best-effort throughout: this is bookkeeping for a cast that is about to
        happen regardless, so a catalog that will not load must not stop it.
        """
        area_mm2 = self._cast_deposit_area_mm2()
        try:
            um = self._twin_thickness_um(channel_plan, context)
        except Exception:
            logger.warning("ht_predicted_thickness_failed", exc_info=True)
            um = None
        return um, area_mm2, ("predicted" if um else "unavailable")

    def _verify_head_position(self) -> bool:
        """Confirm the dispenser-head position before an HT run (start-gate).

        Returns ``False`` to abort the start (operator dismissed the prompt, or
        a safety retract failed).  See
        :func:`softae.gui.widgets.head_check_dialog.verify_head_before_run`.
        """
        from softae.gui.widgets.head_check_dialog import verify_head_before_run

        return verify_head_before_run(
            self, self._manager, context="starting the experiment"
        )

    def _occupancy_gate(self, wf: Workflow) -> bool:
        """Warn before re-casting into recorded-occupied wells (single-use).

        Sets :attr:`_active_board_id` (the board these casts record under) and
        returns ``False`` to abort the run.  Deposition modes only — a
        measure-only run casts nothing, so it never conflicts.  On a confirmed
        board replacement the active board advances to a fresh, empty id; a
        deliberate "cast anyway" keeps the same board id.
        """
        self._active_board_id = 0
        if self._data_store is None or wf.metadata.get("mode") == "measure_only":
            return True
        try:
            board_id = int(self._data_store.current_board_id())
            channels = [s + 1 for s in self._selected_channels()]
        except Exception:
            return True
        self._active_board_id = board_id

        from softae.gui.widgets.occupancy_guard import (
            BoardReplacedDecision,
            occupied_conflicts,
            prompt_board_replaced,
        )

        conflicts = occupied_conflicts(self._data_store, board_id, channels)
        if not conflicts:
            return True
        decision = prompt_board_replaced(self, board_id, conflicts)
        if decision is BoardReplacedDecision.CANCEL:
            return False
        if decision is BoardReplacedDecision.FRESH:
            self._active_board_id = board_id + 1  # fresh board → empty occupancy
            # Persist the pointer so the swap survives even if the run is
            # aborted before any deposit records a cast on the new plate.
            try:
                self._data_store.set_active_board(self._active_board_id)
            except Exception:
                logger.exception("active_board_persist_error")
        return True

    def _on_start(self) -> None:
        """Generate workflow and start execution in a background thread."""
        # Head-position start-gate: the deposition workflow issues conditional
        # head commands, so the software belief must match reality first.
        if not self._verify_head_position():
            return

        try:
            wf = self._generate_workflow()
        except Exception as exc:
            QMessageBox.warning(self, "Workflow Error", str(exc))
            return

        # Overflow guard: deposition modes only (measure-only casts nothing).
        if wf.metadata.get("mode") != "measure_only" and not self._overflow_check(
            getattr(self, "_last_dispense_plan", [])
        ):
            return

        # Occupancy guard: warn before re-casting into recorded-occupied wells.
        if not self._occupancy_gate(wf):
            return

        # Build the per-channel .mscr files now so they exist when the
        # executor calls sendscript_getdata.  _generate_workflow uses
        # per-channel paths (softae_ch{N}.mscr), so we reproduce the same
        # channel list and preset params here.  Formulate-only mode runs no
        # EIS steps, so there is nothing to build.
        try:
            from softae.drivers.mscr_library import eis_run_mscrbuild

            selected = self._selected_channels()
            channel_numbers = (
                [] if wf.metadata.get("mode") == "formulate_only"
                else [s + 1 for s in selected]
            )
            p = {
                "f_hi": self._spin_eis_f_hi.value(),
                "f_lo_mHz": self._spin_eis_f_lo.value(),
                "npts": self._spin_eis_npts.value(),
                "mv_ac": self._spin_eis_mv_ac.value(),
                "mv_dc": self._spin_eis_mv_dc.value(),
            }

            for ch_1based in channel_numbers:
                mscr_path = os.path.join(
                    tempfile.gettempdir(), f"softae_ch{ch_1based}.mscr"
                )
                eis_run_mscrbuild(
                    mscr_path,
                    mux_ch=ch_1based,
                    mVac=p.get("mv_ac", 10),
                    f_hi=p.get("f_hi", 200_000),
                    f_lo=p.get("f_lo_mHz", 100),
                    npts=p.get("npts", 20),
                    mVdc=p.get("mv_dc", 0),
                )
        except Exception as exc:
            QMessageBox.warning(self, "Script Build Error", str(exc))
            return

        # Reset results
        self._results.clear()
        self._results_table.setRowCount(0)

        # Register run in DataStore FIRST so the run_id is ready before the
        # executor is constructed (it is captured in the executor's _run_id).
        self._ds_run_id = None
        if self._data_store is not None:
            try:
                self._ds_run_id = self._data_store.start_run(
                    wf.name,
                    mode=self._combo_mode.currentText(),
                    annotation=self._te_annotation.toPlainText().strip(),
                )
            except Exception:
                logger.exception("datastore_start_run_error")

        # Persist planned formulations before execution starts.
        if self._data_store is not None and self._ds_run_id is not None:
            if wf.metadata.get("mode") != "measure_only":
                notes = "corrected" if wf.metadata.get("liquid_handling_enabled") else "uncorrected"
                # Resolved once per run: the loadout and catalogs are the same
                # for every channel, only the volumes differ.
                try:
                    cast_context = self._cast_stocks()
                except Exception:
                    logger.warning("ht_cast_stocks_unresolved", exc_info=True)
                    cast_context = None
                for channel_plan in wf.metadata.get("dispense_plan", []):
                    try:
                        thickness_um, area_mm2, method = self._predicted_cast(
                            channel_plan, cast_context)
                        self._data_store.record_formulation(
                            self._ds_run_id,
                            int(channel_plan.get("channel", 0)),
                            pump0_uL=float(channel_plan.get("pump0_commanded_uL", 0.0)),
                            pump1_uL=float(channel_plan.get("pump1_commanded_uL", 0.0)),
                            pump2_uL=float(channel_plan.get("pump2_commanded_uL", 0.0)),
                            total_uL=float(channel_plan.get("total_commanded_uL", 0.0)),
                            dispense_rate_uL_min=float(self._spin_rate.value()),
                            predicted_thickness_um=thickness_um,
                            deposit_area_mm2=area_mm2,
                            thickness_method=method,
                            notes=notes,
                        )
                    except Exception:
                        logger.exception("datastore_record_formulation_error")

        # Set up executor with structured logging
        log_dir = Path(tempfile.gettempdir()) / "softae_logs"
        self._exp_logger = ExperimentLogger(log_dir, wf.name)
        # Graceful stage-timeout recovery: a wedged-stage timeout on one channel
        # replays that channel from its precondition (when no elution was
        # committed) or skips it, rather than aborting the whole campaign.
        try:
            stage_cfg = loader.instruments().get("stage", {})
        except Exception:
            stage_cfg = {}
        max_retries = int(stage_cfg.get("max_channel_retries", 1))
        # Per-channel retries are bounded; the number of channels allowed to fail
        # was not. Read from the same section as the retry count because the two
        # multiply: a wedged stage costs (1 + max_retries) attempts per channel.
        max_consecutive = int(stage_cfg.get(
            "max_consecutive_channel_failures",
            DEFAULT_MAX_CONSECUTIVE_CHANNEL_FAILURES,
        ))
        self._executor = WorkflowExecutor(
            self._manager,
            experiment_logger=self._exp_logger,
            data_store=self._data_store,
            run_id=self._ds_run_id,
            continue_on_error=True,
            max_channel_retries=max_retries,
            max_consecutive_channel_failures=max_consecutive,
        )

        # Waste accrual (P5.4) and in-run purge windows (P8), taken from the
        # window so the HT tab and a campaign book against the same ledger.
        window = self.window()
        self._executor.waste_ledger = getattr(window, "_waste_ledger", None)
        runner = getattr(window, "_purge_runner", None)
        if runner is not None:
            self._executor.on_purge_window = lambda step: runner.maybe_purge(
                context=f"step:{getattr(step, 'name', '?')}",
                owns_rig=True, allow_positioning=True, end_at_idle_rest=False,
            )

        # Wire callbacks (these run on the executor thread → emit signals)
        self._executor.on_step_start = self._cb_step_start
        self._executor.on_step_complete = self._cb_step_complete
        self._executor.on_step_error = self._cb_step_error
        self._executor.on_step_recover = self._cb_step_recover
        self._executor.on_step_skipped = self._cb_step_skipped
        self._executor.on_channel_failure_hold = self._cb_channel_hold
        self._executor.on_state_change = self._cb_state_change

        # UI state
        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_abort.setEnabled(True)
        self._progress.setRange(0, wf.total_steps)
        self._progress.setValue(0)
        self._lbl_status.setText("Running…")

        # Preview
        self._on_generate_workflow()

        # Run in background thread (with its own event loop)
        self._run_thread = threading.Thread(
            target=self._run_workflow_thread,
            args=(wf,),
            daemon=True,
        )
        self._run_thread.start()

    def _on_pause(self) -> None:
        """Toggle pause / resume on the running executor."""
        if self._executor is None:
            return
        if self._executor.state is ExecutorState.RUNNING:
            self._executor.pause()
            self._btn_pause.setText("▶  Resume")
        elif self._executor.state is ExecutorState.PAUSED:
            self._executor.resume()
            self._btn_pause.setText("⏸  Pause")

    def _on_abort(self) -> None:
        """Request the executor to abort the workflow."""
        if self._executor is not None:
            self._executor.abort()

    # ── Daemon shutdown seam (signal-first abort + bounded join) ─────────
    def _abort_run_impl(self) -> None:
        if self._executor is not None:
            self._executor.abort()          # ExecutorState -> ABORTED

    def _runner_thread(self):
        return self._run_thread

    # ── Background execution thread ──────────────────────────────────────

    def _run_workflow_thread(self, wf: Workflow) -> None:
        """Runs the async executor in a dedicated event loop (daemon thread)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Re-create per-instrument asyncio.Lock objects so they bind to THIS
        # event loop rather than the stale loop from GUI startup / connect_all
        # / a previous run. Without this, the executor's first
        # ``async with inst._lock`` in _dispatch() blocks forever on a lock
        # bound to a dead loop — the workflow appears to "hang" at step 1.
        # (Mirrors the Arrhenius sweep tab, which already does this.)
        try:
            self._manager.reset_locks()
        except Exception:
            logger.warning("experiment_reset_locks_failed", exc_info=True)
        try:
            # Claim the rig for the run and return it to idle rest afterwards.
            # HT workflows end with the head retracted wherever they finished;
            # the resting convention is head DOWN in the flush basin, so without
            # this the tip sits in air between runs — the exact failure the
            # anti-clog work exists to prevent.
            with rig_run(self, f"ht:{getattr(wf, 'name', 'workflow')}",
                         instruments=workflow_instruments(wf)):
                loop.run_until_complete(self._executor.run(wf))
            self._sig_workflow_done.emit(0)
        except Exception as exc:
            logger.error("experiment_tab_workflow_error", error=str(exc))
            self._sig_workflow_done.emit(1)
        finally:
            if self._exp_logger is not None:
                self._exp_logger.close()
            loop.close()

    # ── Executor callbacks (run on background thread) ────────────────────

    def _cb_step_start(self, step: WorkflowStep, index: int, total: int) -> None:
        self._sig_step_start.emit(step.name, index, total)

    def _cb_step_complete(
        self, step: WorkflowStep, index: int, total: int, result: Any, elapsed: float = 0.0
    ) -> None:
        self._sig_step_complete.emit(step.name, index, total, result, elapsed)

    def _cb_step_error(
        self, step: WorkflowStep, index: int, total: int, error: Exception
    ) -> None:
        self._sig_step_error.emit(step.name, index, total, str(error))

    def _cb_step_recover(
        self, step: WorkflowStep, error: Exception, attempt: int
    ) -> None:
        """Before a channel replay: reset the wedged stage's VISA session.

        This is the driver-level equivalent of a GUI session close (which the
        operator has observed recovers the stage ~95% of the time), scoped to
        just the stage so no other instrument connection is disturbed.
        """
        logger.warning(
            "channel_recover", step=step.name, attempt=attempt, error=str(error)
        )
        try:
            stage = self._manager.get("stage")
            reset = getattr(stage, "reset_session", None)
            if callable(reset):
                reset()
                logger.info("stage_session_reset_via_recover", step=step.name)
        except Exception:
            logger.warning("stage_reset_on_recover_failed", exc_info=True)

    def _cb_step_skipped(
        self, step: WorkflowStep, index: int, total: int, reason: str
    ) -> None:
        self._sig_step_skipped.emit(step.name, index, total, reason)

    def _cb_channel_hold(
        self, channel: str, consecutive: int, timeout_s: float
    ) -> None:
        self._sig_channel_hold.emit(str(channel), int(consecutive), float(timeout_s))

    def _cb_state_change(self, old: ExecutorState, new: ExecutorState) -> None:
        self._sig_state_change.emit(old.name, new.name)

    # ── UI update slots (run on the main / GUI thread) ───────────────────

    def _ui_step_start(self, step_name: str, index: int, total: int) -> None:
        self._progress.setValue(index)
        self._lbl_status.setText(f"[{index + 1}/{total}] {step_name}")

    @staticmethod
    def _parse_step_name(step_name: str) -> tuple[str, str]:
        """Return ``(channel, display_name)`` parsed from workflow step names."""
        channel = ""
        if "__iter" in step_name:
            channel = step_name.split("__iter")[-1]
        elif "_ch" in step_name:
            suffix = step_name.rsplit("_ch", 1)[-1]
            if suffix.isdigit():
                channel = suffix

        display_name = step_name.split("__iter")[0] if "__iter" in step_name else step_name
        return channel, display_name

    def _ui_step_complete(
        self, step_name: str, index: int, _total: int, result: object, elapsed: float = 0.0
    ) -> None:
        self._progress.setValue(index + 1)
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        ch, display_name = self._parse_step_name(step_name)
        self._results_table.setItem(row, 0, QTableWidgetItem(ch))
        self._results_table.setItem(row, 1, QTableWidgetItem(display_name))
        self._results_table.setItem(row, 2, QTableWidgetItem("✓"))
        self._results_table.setItem(row, 3, QTableWidgetItem(f"{elapsed:.1f}" if elapsed else ""))

        # ── Capture EIS data when an EIS step completes ──
        if display_name.startswith("measure_eis_ch") and result is not None:
            try:
                eis = self._raw_to_eis_result(result, ch)
                self._eis_results.append(eis)
                detail = f"{eis.npts} pts, f=[{eis.frequency.min():.0f}-{eis.frequency.max():.0f}] Hz"
                # Pair the spectrum with the run that produced it, here where
                # both are in scope. ``_ds_run_id`` is cleared at workflow end;
                # a post-run fit still needs the run to resolve thickness.
                if self._ds_run_id is not None:
                    self._run_id_by_channel[int(eis.channel)] = str(self._ds_run_id)
                # Persist to DataStore
                if self._data_store is not None and self._ds_run_id is not None:
                    try:
                        ch_int = int(ch) if ch.isdigit() else 0
                        measurement_id = self._data_store.record_measurement(
                            self._ds_run_id, eis, channel=ch_int,
                        )
                        from softae.core.conditions_capture import read_environment

                        env = read_environment(self._manager)
                        if any(v is not None for v in env.values()):
                            self._data_store.record_conditions(
                                measurement_id, "measurement", **env
                            )
                    except Exception:
                        logger.exception("datastore_record_measurement_error")
            except Exception as exc:
                detail = f"EIS parse error: {exc}"
                logger.warning("eis_capture_error", channel=ch, error=str(exc))
        else:
            detail = str(result)[:100] if result is not None else ""

        self._results_table.setItem(row, 4, QTableWidgetItem(detail))

        # Record single-use well occupancy on a completed deposit (a real cast
        # into that electrode).  Board id is fixed by the pre-run occupancy gate;
        # only ``deposit_ch{N}`` steps qualify (flushes/EIS never mark a well).
        if (
            display_name.startswith("deposit_ch")
            and ch.isdigit()
            and self._data_store is not None
        ):
            try:
                self._data_store.record_electrode_cast(
                    int(self._active_board_id), int(ch),
                    run_id=self._ds_run_id, iteration=None,
                )
            except Exception:
                logger.exception("datastore_record_occupancy_error")

        self._results.append(
            {"channel": ch, "step": display_name, "status": "ok", "result": result}
        )

    def _ui_step_error(
        self, step_name: str, _index: int, _total: int, error_msg: str
    ) -> None:
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        ch, display_name = self._parse_step_name(step_name)

        self._results_table.setItem(row, 0, QTableWidgetItem(ch))
        self._results_table.setItem(row, 1, QTableWidgetItem(display_name))
        err_item = QTableWidgetItem("✗")
        err_item.setForeground(Qt.GlobalColor.red)
        self._results_table.setItem(row, 2, err_item)
        self._results_table.setItem(row, 3, QTableWidgetItem(""))
        self._results_table.setItem(row, 4, QTableWidgetItem(error_msg[:100]))

        self._results.append(
            {"channel": ch, "step": display_name, "status": "error", "error": error_msg}
        )

    def _ui_step_skipped(
        self, step_name: str, _index: int, _total: int, reason: str
    ) -> None:
        """Surface a channel step skipped by the graceful-recovery policy."""
        row = self._results_table.rowCount()
        self._results_table.insertRow(row)

        ch, display_name = self._parse_step_name(step_name)

        self._results_table.setItem(row, 0, QTableWidgetItem(ch))
        self._results_table.setItem(row, 1, QTableWidgetItem(display_name))
        skip_item = QTableWidgetItem("⤼ skipped")
        skip_item.setForeground(Qt.GlobalColor.darkYellow)
        self._results_table.setItem(row, 2, skip_item)
        self._results_table.setItem(row, 3, QTableWidgetItem(""))
        self._results_table.setItem(row, 4, QTableWidgetItem(reason))

        self._results.append(
            {"channel": ch, "step": display_name, "status": "skipped", "reason": reason}
        )

    def _ui_channel_hold(
        self, channel: str, consecutive: int, timeout_s: float
    ) -> None:
        """The executor hit its consecutive-failure ceiling and paused. Ask.

        **Deliberately non-modal.** A modal dialog would take the tab's own Pause
        and Abort buttons away at the exact moment the operator most needs them,
        which is the "a pause must not be a lockout" rule inverted. This window
        offers the two answers; every other control stays live, including doing
        nothing and going to look at the rig.
        """
        QApplication.beep()
        self._lbl_status.setText(
            f"Held — {consecutive} channels failed in a row (last: {channel})"
        )
        self.workflow_status_changed.emit("Held — needs attention")
        self._btn_pause.setText("▶  Resume")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Run held — consecutive channel failures")
        box.setText(
            f"{consecutive} channels in a row were abandoned "
            f"(most recently channel {channel}).\n\n"
            "The run is paused. Nothing is moving."
        )
        box.setInformativeText(
            "Repeated comms/timeout failures across consecutive channels usually "
            "mean the stage is obstructed rather than the wells being bad — each "
            "of these channels already had its stage session reset and failed "
            "again.\n\nLook at the rig before continuing.\n\n"
            f"If nobody answers within {timeout_s / 60:.0f} min the run parks "
            "itself and stops."
        )
        continue_btn = box.addButton("Continue plate",
                                     QMessageBox.ButtonRole.AcceptRole)
        abort_btn = box.addButton("Abort run", QMessageBox.ButtonRole.RejectRole)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.finished.connect(
            lambda _code: self._on_channel_hold_answer(box, continue_btn, abort_btn)
        )
        # Held so Qt does not collect the dialog the moment this slot returns.
        self._hold_box = box
        box.show()

    def _on_channel_hold_answer(
        self, box: QMessageBox, continue_btn, abort_btn
    ) -> None:
        """Apply the operator's answer — including "neither".

        Closing the window without choosing is a third, legitimate answer: *I am
        going to go and look*. It must not be read as an abort, so it leaves the
        run exactly as the hold left it — paused, with the tab's own Pause/Abort
        buttons live. The bounded hold still runs underneath, which is what stops
        "going to look" from silently becoming "went home".
        """
        self._hold_box = None
        if self._executor is None:
            return
        clicked = box.clickedButton()
        if clicked is continue_btn:
            self._executor.resume()
            self._btn_pause.setText("⏸  Pause")
        elif clicked is abort_btn:
            self._executor.abort()

    def _ui_state_change(self, old_state: str, new_state: str) -> None:
        if new_state == "PAUSED":
            self._lbl_status.setText("Paused")
            self.workflow_status_changed.emit("Paused")
            # The executor can pause itself (the consecutive-failure hold), and a
            # button still reading "Pause" is then a button that says the opposite
            # of what pressing it does.
            self._btn_pause.setText("▶  Resume")
        elif new_state == "RUNNING":
            self._lbl_status.setText("Running…")
            self.workflow_status_changed.emit("Running")
            self._btn_pause.setText("⏸  Pause")
        else:
            self._lbl_status.setText(f"State: {new_state}")

    def _ui_workflow_done(self, exit_code: int) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setText("⏸  Pause")
        self._btn_abort.setEnabled(False)

        # `None` (no executor) and `[]` (an executor that skipped nothing) are
        # different claims and stay different all the way to the column: the
        # first is "nobody counted", the second is "counted, none skipped".
        skipped = list(self._executor.skipped_channels) if self._executor else None

        if exit_code == 0:
            # A plate that abandoned channels did not "complete" — it completed
            # around the wells it gave up on, and the operator is entitled to know
            # how many before they walk away from the screen.
            self._lbl_status.setText(
                f"Completed with {len(skipped)} skipped" if skipped
                else "Completed ✓"
            )
            self._progress.setValue(self._progress.maximum())
            self.workflow_status_changed.emit("Idle")
        else:
            state = self._executor.state.name if self._executor else "ERROR"
            if state == "ABORTED":
                self._lbl_status.setText("Aborted")
                self.workflow_status_changed.emit("Idle")
            else:
                self._lbl_status.setText(f"Failed ({state})")
                self.workflow_status_changed.emit("Idle")

        # Close DataStore run
        if self._data_store is not None and self._ds_run_id is not None:
            try:
                state = self._executor.state.name if self._executor else "ERROR"
                if exit_code == 0:
                    # The record is the only surface that survives this window.
                    # `done` on a plate with 31 skipped channels is the one lie
                    # that accumulates: a later reader cannot tell it from a good
                    # row, and no later fix recovers the rows already written.
                    status = "partial" if skipped else "done"
                elif state == "ABORTED":
                    status = "aborted"
                else:
                    status = "error"
                self._data_store.finish_run(
                    self._ds_run_id, status=status, skipped_channels=skipped,
                )
            except Exception:
                logger.exception("datastore_finish_run_error")
            self._ds_run_id = None

    # ── EIS data helpers ───────────────────────────────────────────────────

    def _raw_to_eis_result(self, raw_result: object, ch_str: str) -> EISResult:
        """Convert raw sendscript_getdata result to an EISResult.

        ``eis_extractdata`` returns ``[f, |Z|, phase, Z', -Z'']`` (five 1-D
        arrays) for both the mock and real drivers.
        """
        # Try to get the pico for extraction
        pico_name = "pico1"
        electrode_ch = int(ch_str) if ch_str.isdigit() else 0
        if electrode_ch > 0:
            pico_name = pico_for_channel(electrode_ch)

        pico = self._manager.get(pico_name)
        data = pico.eis_extractdata(raw_result)

        f = np.asarray(data[0])
        z_mag = np.asarray(data[1])
        phase = np.asarray(data[2])
        z_real = np.asarray(data[3])
        z_imag_neg = np.asarray(data[4])

        return EISResult(
            channel=electrode_ch,
            frequency=f,
            z_magnitude=z_mag,
            phase=phase,
            z_real=z_real,
            z_imag_neg=z_imag_neg,
            eis_params={"preset": self._combo_eis.currentText()},
        )

    def _on_save_eis_data(self) -> None:
        """Save each captured EIS result as a separate file."""
        if not self._eis_results:
            QMessageBox.information(
                self, "No EIS Data",
                "No EIS measurements captured yet.\n"
                "Run a workflow with EIS steps first.",
            )
            return

        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for EIS Data"
        )
        if not out_dir:
            return

        from datetime import datetime as _dt

        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for eis in self._eis_results:
            fname = f"E{eis.channel}_{ts}_eisdata.txt"
            try:
                eis.save(Path(out_dir) / fname, study_name=f"ht_experiment_{ts}")
                saved += 1
            except Exception as exc:
                logger.warning("eis_save_error", channel=eis.channel, error=str(exc))

        QMessageBox.information(
            self, "Saved",
            f"Saved {saved}/{len(self._eis_results)} EIS files to {out_dir}",
        )

    def _recorded_thickness_cm(self, channel: int) -> float | None:
        """The twin's recorded dry-film thickness for *channel*, in **cm**.

        ``None`` means "no usable thickness", which the caller must render as
        ``—`` rather than substitute a placeholder for. A record whose
        ``area_mm2`` is ``None`` also reads as nothing (P.11): the thickness is a
        quotient whose denominator was never recorded, so it has no known basis
        and dividing σ by it would manufacture a number. The Analysis tab's
        ``_recorded_thickness_cm`` guards the same way for the same reason.
        """
        if self._data_store is None:
            return None
        run_id = self._run_id_by_channel.get(int(channel))
        if not run_id:
            return None
        getter = getattr(self._data_store, "predicted_thickness_record", None)
        try:
            if callable(getter):
                record = getter(run_id, int(channel))
                if record is None:
                    return None
                if record.area_mm2 is None:
                    logger.warning("thickness_withheld_area_never_recorded",
                                   channel=int(channel), run_id=run_id)
                    return None
                um = record.um
            else:
                um = self._data_store.predicted_thickness_um(run_id, int(channel))
        except Exception:
            logger.debug("thickness_lookup_failed", channel=channel, exc_info=True)
            return None
        return (um * 1e-4) if um else None      # µm → cm

    def _fit_cell(self, channel: int) -> Any:
        """The cell constant for *channel*, or ``None`` if any term is unknown.

        The board supplies L and w; only t must be looked up per channel, and until
        a deposition twin records one there is no honest σ to print.
        """
        from softae.gui.eis_sigma import gui_cell

        pcb = self._active_pcb_config()
        return gui_cell(pcb.get("electrode_L_cm") or 0.0,
                        self._recorded_thickness_cm(channel),
                        pcb.get("electrode_w_cm") or 0.0)

    def _on_fit_all_eis(self) -> None:
        """Run circuit fitting on all captured EIS data.

        R0 and R1 are measured, so they print unconditionally. σ needs geometry
        the board and the deposition twin must supply, so it prints ``—`` when
        either is missing rather than a number derived from a placeholder.
        """
        if not self._eis_results:
            QMessageBox.information(
                self, "No EIS Data",
                "No EIS measurements captured yet.\n"
                "Run a workflow with EIS steps first.",
            )
            return

        import math

        from softae.analysis.eis.engine import analyze_spectrum
        from softae.gui.eis_sigma import report_sigma

        model = self._combo_fit_model.currentData() or self.DEFAULT_FIT_MODEL

        lines = [f"Circuit Fitting — {model}", "=" * 40]
        for eis in self._eis_results:
            # ``engine`` unset: ``[eis] engine`` chooses, here as everywhere else.
            # Both the fit and the σ come off one report, so the R this tab prints
            # is standard-suite-derived rather than a bare number from a raw fitter.
            report = analyze_spectrum(eis, cell=self._fit_cell(eis.channel),
                                      model_name=model)
            fr = report.fit
            if fr is None or not fr.success:
                err = getattr(fr, "error_msg", "") or "rejected before fitting"
                lines.append(f"E{eis.channel}: {err[:60]}  ✗")
                continue
            sigma = report_sigma(report)
            sigma_txt = f"{sigma:.3e} S/cm" if math.isfinite(sigma) else "—"
            lines.append(
                f"E{eis.channel}: R0={fr.R0:.1f}Ω  R1={fr.R1:.1f}Ω  "
                f"σ={sigma_txt}  ✓"
            )

        self._txt_fit_output.setPlainText("\n".join(lines))

    # ── Export ────────────────────────────────────────────────────────────

    def _on_save_csv(self) -> None:
        """Export the results table to CSV."""
        if not self._results:
            QMessageBox.information(self, "No Data", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "results.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["channel", "step", "status", "result", "error"]
                )
                writer.writeheader()
                for r in self._results:
                    writer.writerow(
                        {
                            "channel": r.get("channel", ""),
                            "step": r.get("step", ""),
                            "status": r.get("status", ""),
                            "result": str(r.get("result", ""))[:200],
                            "error": r.get("error", ""),
                        }
                    )
            QMessageBox.information(self, "Saved", f"Results saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _on_save_pdf(self) -> None:
        """Export the results as a simple PDF report (matplotlib figure)."""
        if not self._results:
            QMessageBox.information(self, "No Data", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "report.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title("Experiment Results Summary")
                ax.axis("off")

                col_labels = ["Ch", "Step", "Status"]
                cell_text = []
                for r in self._results[:50]:  # limit rows for readability
                    cell_text.append(
                        [r.get("channel", ""), r.get("step", ""), r.get("status", "")]
                    )
                if cell_text:
                    ax.table(
                        cellText=cell_text,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                    )
                pdf.savefig(fig)
                plt.close(fig)
            QMessageBox.information(self, "Saved", f"PDF report saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "PDF Error", str(exc))
