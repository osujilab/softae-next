"""Tab 5: Arrhenius Sweep.

Provides a GUI for running temperature-stepped EIS sweeps via
:class:`~softae.workflows.temp_eis_sweep.ArrheniusSweep`.  All channels in
the sweep are measured at *every* temperature before the setpoint advances —
guaranteed by the DAG topology in ``ArrheniusSweep.build_workflow()``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

import structlog
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from softae.core.channel_spec import parse_channel_spec
from softae.gui.daemon_runner import DaemonRunnerMixin
from softae.gui.rig_claim import rig_run

if TYPE_CHECKING:
    from pathlib import Path

    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


def _widget_alive(w: "QWidget") -> bool:
    """Return True only if the C++ QWidget backing *w* has not been deleted."""
    try:
        return w.isVisible()
    except RuntimeError:
        return False


def _restore_combo(combo: "QComboBox", wanted: str, field: str) -> None:
    """Select *wanted* in *combo*, and say so in the log when it is not offered.

    A saved config naming a model that no longer exists (a retired circuit, a
    renamed thermal law) used to miss in silence: ``findText`` returned -1, the
    combo kept whatever was already selected, and the sweep then fitted a
    *different* model than the config on disk records. Falling back is still the
    right behaviour — a restore path must not crash on a stale config — but the
    substitution has to be visible, and it has to name what will actually run so
    the operator can tell which model the data came from.
    """
    idx = combo.findText(wanted)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return
    logger.warning(
        "arrhenius_config_model_not_offered",
        field=field,
        requested=wanted,
        using=combo.currentText(),
        available=[combo.itemText(i) for i in range(combo.count())],
    )


class ArrheniusTab(DaemonRunnerMixin, QWidget):
    """Temperature-stepped EIS sweep control panel.

    Parameters
    ----------
    manager : InstrumentManager
    data_store : DataStore or None
    """

    # Class-level palette constants (avoid per-call allocation in _on_eis_point)
    _PALETTE    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    _LINESTYLES = ["-", "--", ":", "-."]
    _MARKERS    = ["o", "s", "^", "D", "v", "p", "h", "*"]

    # Emitted on the GUI thread when the background sweep finishes.
    _sig_sweep_done  = Signal(bool, str)             # (success, message)
    _sig_log_line    = Signal(str)                   # single log line to append
    _sig_progress    = Signal(int)                   # progress bar step index
    _sig_eis_point   = Signal(int, float, float, float, float, float)  # ch, T_C, sigma, R0, R1, rh_sp
    # Public signal: monitor tab listens to this.  (status_text, step, total)
    sweep_status_changed = Signal(str, int, int)

    def __init__(
        self,
        manager: "InstrumentManager",
        *,
        data_store: "DataStore | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._sweep_thread: threading.Thread | None = None
        self._plot_data: dict[int, tuple[list, list]] = {}
        self._plot_lines: dict[int, Any] = {}          # Line2D artists for incremental updates
        self._all_channels: list[int] = []             # sorted list of channels seen this sweep
        self._all_rhs: list[float] = []               # sorted list of RH setpoints seen
        self._last_rh_results: dict | None = None      # stored after RH sweep for on-demand 3D

        # Max channel number allowed — updated when Init tab changes PCB selection.
        self._max_channel: int = self._init_max_channel()

        self._sig_sweep_done.connect(self._on_sweep_done)
        self._sig_log_line.connect(self._log_line)
        self._sig_progress.connect(self._progress_set)
        self._sig_eis_point.connect(self._on_eis_point)

        self._build_ui()

    @staticmethod
    def _init_max_channel() -> int:
        """Read the largest channel count across all configured PCBs from the TOML."""
        try:
            from softae.config.loader import pcb_configs as _pcb_configs
            cfgs = _pcb_configs()
            if not cfgs:
                return 32  # safe fallback
            return max(
                info.get("channels", 0) or
                info.get("grid", [0, 0])[0] * info.get("grid", [0, 0])[1]
                for info in cfgs.values()
            ) or 32
        except Exception:
            return 32

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Outer horizontal splitter: left (params) | right (plot + log) ───
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(6)
        h_split.setChildrenCollapsible(False)

        # ════════════════════════════════════════════════════════════════════
        # LEFT SIDE — parameter panels in a vertical splitter
        # ════════════════════════════════════════════════════════════════════
        left_split = QSplitter(Qt.Orientation.Vertical)
        left_split.setHandleWidth(5)
        left_split.setChildrenCollapsible(False)

        # Temperature profile
        temp_grp = QGroupBox("Temperature Profile")
        temp_form = QFormLayout(temp_grp)
        self._spin_t_start = QDoubleSpinBox()
        self._spin_t_start.setRange(-50.0, 200.0)
        self._spin_t_start.setValue(25.0)
        self._spin_t_start.setSuffix(" °C")
        temp_form.addRow("T start:", self._spin_t_start)

        self._spin_t_stop = QDoubleSpinBox()
        self._spin_t_stop.setRange(-50.0, 200.0)
        self._spin_t_stop.setValue(75.0)
        self._spin_t_stop.setSuffix(" °C")
        temp_form.addRow("T stop:", self._spin_t_stop)

        self._spin_t_step = QDoubleSpinBox()
        self._spin_t_step.setRange(0.5, 50.0)
        self._spin_t_step.setValue(10.0)
        self._spin_t_step.setSuffix(" °C")
        temp_form.addRow("T step:", self._spin_t_step)

        self._spin_dwell = QDoubleSpinBox()
        self._spin_dwell.setRange(0.0, 7200.0)
        self._spin_dwell.setValue(60.0)
        self._spin_dwell.setSuffix(" s")
        temp_form.addRow("Dwell after equil.:", self._spin_dwell)

        self._spin_tolerance = QDoubleSpinBox()
        self._spin_tolerance.setRange(0.1, 10.0)
        self._spin_tolerance.setSingleStep(0.1)
        self._spin_tolerance.setValue(0.5)
        self._spin_tolerance.setSuffix(" °C")
        temp_form.addRow("Tolerance:", self._spin_tolerance)

        self._spin_rank_T = QSpinBox()
        self._spin_rank_T.setRange(1, 3)
        self._spin_rank_T.setValue(2)
        self._spin_rank_T.setToolTip(
            "Sweep rank for Temperature (1=outermost / slowest, 3=innermost / fastest)"
        )
        temp_form.addRow("Sweep rank:", self._spin_rank_T)

        # Channels + instruments
        chan_grp = QGroupBox("Channels && Instruments")
        chan_form = QFormLayout(chan_grp)

        self._le_channels = QLineEdit("1, 2")
        self._le_channels.setToolTip(
            f"Channel range (1\u2013{self._max_channel}). "
            "Use commas and dashes, e.g. \"1, 3-6, 9\""
        )
        chan_form.addRow("Channels:", self._le_channels)

        self._le_eis_inst = QLineEdit("pico1")
        chan_form.addRow("EIS instrument:", self._le_eis_inst)

        self._le_temp_inst = QLineEdit("temp_controller")
        chan_form.addRow("Temp instrument:", self._le_temp_inst)

        self._spin_timeout = QSpinBox()
        self._spin_timeout.setRange(60, 7200)
        self._spin_timeout.setValue(1800)
        self._spin_timeout.setSuffix(" s")
        chan_form.addRow("Wait timeout:", self._spin_timeout)

        self._spin_rank_ch = QSpinBox()
        self._spin_rank_ch.setRange(1, 3)
        self._spin_rank_ch.setValue(3)
        self._spin_rank_ch.setToolTip(
            "Sweep rank for Channels (1=outermost / slowest, 3=innermost / fastest)"
        )
        chan_form.addRow("Sweep rank:", self._spin_rank_ch)

        # Electrode geometry
        geom_grp = QGroupBox("Electrode Geometry")
        geom_form = QFormLayout(geom_grp)
        geom_form.addRow(
            QLabel(
                "Required to compute σ from R₁.\n"
                "Leave L_cm = 0 to skip σ calculation."
            )
        )

        self._spin_L = QDoubleSpinBox()
        self._spin_L.setRange(0.0, 10.0)
        self._spin_L.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_L.setSingleStep(0.01)
        self._spin_L.setSuffix(" cm")
        geom_form.addRow("L (electrode spacing):", self._spin_L)

        self._spin_t = QDoubleSpinBox()
        self._spin_t.setRange(0.0, 10.0)
        self._spin_t.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_t.setSingleStep(0.01)
        self._spin_t.setSuffix(" cm")
        geom_form.addRow("t (thickness):", self._spin_t)

        self._spin_w = QDoubleSpinBox()
        self._spin_w.setRange(0.0, 10.0)
        self._spin_w.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_w.setSingleStep(0.01)
        self._spin_w.setSuffix(" cm")
        geom_form.addRow("w (width):", self._spin_w)

        # EIS parameters
        eis_grp = QGroupBox("EIS Parameters")
        eis_vlay = QVBoxLayout(eis_grp)
        eis_top_row = QHBoxLayout()
        eis_top_row.addWidget(QLabel("Preset:"))
        self._combo_eis_preset = QComboBox()
        from softae.config.loader import eis_presets as _eis_presets
        for name in sorted(_eis_presets().keys()):
            self._combo_eis_preset.addItem(name)
        if self._combo_eis_preset.count() == 0:
            self._combo_eis_preset.addItem("Standard")
        eis_top_row.addStretch()
        eis_top_row.addWidget(self._combo_eis_preset)
        eis_vlay.addLayout(eis_top_row)

        eis_params_row = QHBoxLayout()
        eis_params_row.addWidget(QLabel("f_hi (Hz):"))
        self._spin_eis_f_hi = QSpinBox()
        self._spin_eis_f_hi.setRange(1, 200_000)
        self._spin_eis_f_hi.setSingleStep(10_000)
        eis_params_row.addWidget(self._spin_eis_f_hi)
        eis_params_row.addWidget(QLabel("f_lo (mHz):"))
        self._spin_eis_f_lo = QSpinBox()
        self._spin_eis_f_lo.setRange(1, 200_000)
        eis_params_row.addWidget(self._spin_eis_f_lo)
        eis_params_row.addWidget(QLabel("npts:"))
        self._spin_eis_npts = QSpinBox()
        self._spin_eis_npts.setRange(5, 100)
        eis_params_row.addWidget(self._spin_eis_npts)
        eis_params_row.addWidget(QLabel("mVac:"))
        self._spin_eis_mv_ac = QSpinBox()
        self._spin_eis_mv_ac.setRange(1, 200)
        eis_params_row.addWidget(self._spin_eis_mv_ac)
        eis_params_row.addWidget(QLabel("mVdc:"))
        self._spin_eis_mv_dc = QSpinBox()
        self._spin_eis_mv_dc.setRange(-500, 500)
        eis_params_row.addWidget(self._spin_eis_mv_dc)
        eis_params_row.addStretch()
        self._combo_eis_preset.currentTextChanged.connect(self._on_eis_preset_changed)
        self._on_eis_preset_changed(self._combo_eis_preset.currentText())
        eis_vlay.addLayout(eis_params_row)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Circuit model:"))
        self._combo_fit_model = QComboBox()
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS
        for m in CIRCUIT_MODELS:
            self._combo_fit_model.addItem(m)
        fit_row.addWidget(self._combo_fit_model)
        fit_row.addSpacing(12)
        fit_row.addWidget(QLabel("σ(T) model:"))
        self._combo_thermal_model = QComboBox()
        self._combo_thermal_model.addItems(["arrhenius", "vft"])
        self._combo_thermal_model.setToolTip(
            "Temperature-dependence model fitted to σ(T):\n"
            "• arrhenius — σ = A·exp(−Eₐ/k_BT) (linear in 1/T)\n"
            "• vft — σ = A·exp(−B/(T−T₀)) (curved; needs ≥ 3 temperatures)"
        )
        fit_row.addWidget(self._combo_thermal_model)
        fit_row.addStretch()
        eis_vlay.addLayout(fit_row)

        # RH sweep (optional)
        rh_grp = QGroupBox("RH Sweep (optional)")
        rh_grp.setCheckable(True)
        rh_grp.setChecked(False)
        self._rh_grp = rh_grp
        rh_form = QFormLayout(rh_grp)

        self._spin_rh_start = QDoubleSpinBox()
        self._spin_rh_start.setRange(0.0, 100.0)
        self._spin_rh_start.setValue(20.0)
        self._spin_rh_start.setSuffix(" %RH")
        rh_form.addRow("RH start:", self._spin_rh_start)

        self._spin_rh_stop = QDoubleSpinBox()
        self._spin_rh_stop.setRange(0.0, 100.0)
        self._spin_rh_stop.setValue(80.0)
        self._spin_rh_stop.setSuffix(" %RH")
        rh_form.addRow("RH stop:", self._spin_rh_stop)

        self._spin_rh_step = QDoubleSpinBox()
        self._spin_rh_step.setRange(1.0, 50.0)
        self._spin_rh_step.setValue(20.0)
        self._spin_rh_step.setSuffix(" %RH")
        rh_form.addRow("RH step:", self._spin_rh_step)

        self._spin_rh_dwell = QDoubleSpinBox()
        self._spin_rh_dwell.setRange(5.0, 3600.0)
        self._spin_rh_dwell.setValue(30.0)
        self._spin_rh_dwell.setSuffix(" s")
        rh_form.addRow("Dwell after RH set:", self._spin_rh_dwell)

        self._le_rh_inst = QLineEdit("rh_controller")
        rh_form.addRow("RH instrument:", self._le_rh_inst)

        self._spin_rank_rh = QSpinBox()
        self._spin_rank_rh.setRange(1, 3)
        self._spin_rank_rh.setValue(1)
        self._spin_rank_rh.setToolTip(
            "Sweep rank for RH (1=outermost / slowest, 3=innermost / fastest)"
        )
        rh_form.addRow("Sweep rank:", self._spin_rank_rh)
        # Grey-out rank spinbox when RH sweep is disabled
        self._rh_grp.toggled.connect(self._spin_rank_rh.setEnabled)

        # Campaign annotation
        notes_grp = QGroupBox("Campaign Notes (stored with run)")
        notes_lay = QVBoxLayout(notes_grp)
        self._te_annotation = QTextEdit()
        self._te_annotation.setPlaceholderText(
            "Brief description of this experiment campaign "
            "(material, concentration, purpose, etc.)…"
        )
        self._te_annotation.setFixedHeight(56)
        notes_lay.addWidget(self._te_annotation)

        # Progress / controls at the bottom of the left column
        ctrl_widget = QWidget()
        ctrl_vlay = QVBoxLayout(ctrl_widget)
        ctrl_vlay.setContentsMargins(0, 0, 0, 0)

        ctrl = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start Sweep")
        self._btn_start.setFixedHeight(36)
        self._btn_start.clicked.connect(self._on_start)
        ctrl.addWidget(self._btn_start)

        self._btn_abort = QPushButton("■  Abort")
        self._btn_abort.setFixedHeight(36)
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._on_abort)
        ctrl.addWidget(self._btn_abort)

        self._lbl_status = QLabel("Idle")
        self._lbl_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ctrl.addWidget(self._lbl_status)
        ctrl_vlay.addLayout(ctrl)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        ctrl_vlay.addWidget(self._progress)

        cfg_row = QHBoxLayout()
        self._btn_save_cfg = QPushButton("Save Config…")
        self._btn_save_cfg.setFixedHeight(24)
        self._btn_save_cfg.clicked.connect(self._on_save_config)
        cfg_row.addWidget(self._btn_save_cfg)
        self._btn_load_cfg = QPushButton("Load Config…")
        self._btn_load_cfg.setFixedHeight(24)
        self._btn_load_cfg.clicked.connect(self._on_load_config)
        cfg_row.addWidget(self._btn_load_cfg)
        cfg_row.addStretch()
        ctrl_vlay.addLayout(cfg_row)

        # ── Group panels into paired rows ────────────────────────────────────
        # Row 1: Temperature Profile (left) | RH Sweep (right)
        row1 = QSplitter(Qt.Orientation.Horizontal)
        row1.setHandleWidth(5)
        row1.setChildrenCollapsible(False)
        row1.addWidget(temp_grp)
        row1.addWidget(rh_grp)

        # Row 2: Channels & Instruments (left) | Electrode Geometry (right)
        row2 = QSplitter(Qt.Orientation.Horizontal)
        row2.setHandleWidth(5)
        row2.setChildrenCollapsible(False)
        row2.addWidget(chan_grp)
        row2.addWidget(geom_grp)

        left_split.addWidget(row1)
        left_split.addWidget(row2)
        left_split.addWidget(eis_grp)
        left_split.addWidget(notes_grp)
        left_split.addWidget(ctrl_widget)

        h_split.addWidget(left_split)

        # ════════════════════════════════════════════════════════════════════
        # RIGHT SIDE — vertical splitter: plot (top) | log (bottom)
        # ════════════════════════════════════════════════════════════════════
        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.setHandleWidth(5)
        right_split.setChildrenCollapsible(False)

        plot_grp = QGroupBox("Arrhenius Plot (σ vs 1000/T)")
        plot_layout = QVBoxLayout(plot_grp)
        self._fig = Figure(tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("1000/T  (K⁻¹)")
        self._ax.set_ylabel("σ (S/cm)")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas.setMinimumSize(0, 0)
        plot_layout.addWidget(self._canvas)
        plot_btn_row = QHBoxLayout()
        self._btn_export_plot = QPushButton("Export Plot…")
        self._btn_export_plot.clicked.connect(self._on_export_plot)
        plot_btn_row.addWidget(self._btn_export_plot)
        self._btn_show_3d = QPushButton("Show 3D Plot…")
        self._btn_show_3d.setToolTip("Re-open the RH-Arrhenius 3D pop-out from the last sweep")
        self._btn_show_3d.setEnabled(False)
        self._btn_show_3d.clicked.connect(self._on_show_3d)
        plot_btn_row.addWidget(self._btn_show_3d)
        plot_layout.addLayout(plot_btn_row)
        right_split.addWidget(plot_grp)

        log_grp = QGroupBox("Sweep Log")
        log_layout = QVBoxLayout(log_grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("Courier New")
        log_layout.addWidget(self._log)
        right_split.addWidget(log_grp)

        # Equal vertical split between plot and log
        right_split.setStretchFactor(0, 1)
        right_split.setStretchFactor(1, 1)

        h_split.addWidget(right_split)

        # Left ~40 %, right ~60 %
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 3)

        root.addWidget(h_split)

        # Internal abort flag read by the sweep thread
        self._abort_requested = False

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _on_eis_preset_changed(self, name: str) -> None:
        from softae.config.loader import eis_presets
        p = eis_presets().get(name, {})
        if not p:
            return
        self._spin_eis_f_hi.setValue(p.get("f_hi", 200_000))
        self._spin_eis_f_lo.setValue(p.get("f_lo_mHz", 4_000))
        self._spin_eis_npts.setValue(p.get("npts", 35))
        self._spin_eis_mv_ac.setValue(p.get("mv_ac", 10))
        self._spin_eis_mv_dc.setValue(p.get("mv_dc", 0))

    def _parse_channels(self) -> list[int]:
        """Parse the channels field supporting range notation, e.g. "1, 3-6, 9".

        Entry order is preserved because for a sweep it *is* measurement order.
        Raises ``ChannelSpecError`` (a ``ValueError``) on anything malformed,
        empty, or outside the board's ``[1, self._max_channel]``.
        """
        return parse_channel_spec(
            self._le_channels.text(),
            max_ch=self._max_channel,
            order="as-written",
        )

    def set_pcb_channel_count(self, n: int) -> None:
        """Slot: called when the Init tab selects a different PCB."""
        self._max_channel = max(1, n)
        self._le_channels.setToolTip(
            f"Channel range (1\u2013{self._max_channel}). "
            "Use commas and dashes, e.g. \"1, 3-6, 9\""
        )

    def _build_config(self):
        from softae.analysis.arrhenius import ArrheniusSweepConfig

        L = self._spin_L.value()
        t = self._spin_t.value()
        w = self._spin_w.value()
        geom = {"L_cm": L, "t_cm": t, "w_cm": w} if L > 0 else None

        # RH sweep setpoints (only when group box is checked)
        rh_setpoints: list[float] | None = None
        rh_dwell_s = 30.0
        rh_instrument = "rh_controller"
        if self._rh_grp.isChecked():
            rh_start = self._spin_rh_start.value()
            rh_stop = self._spin_rh_stop.value()
            rh_step = self._spin_rh_step.value()
            if rh_start == rh_stop:
                rh_setpoints = [round(rh_start, 2)]
            elif rh_step > 0 and rh_stop >= rh_start:
                n_rh = int(round((rh_stop - rh_start) / rh_step)) + 1
                rh_setpoints = [round(rh_start + i * rh_step, 2) for i in range(n_rh)]
            rh_dwell_s = self._spin_rh_dwell.value()
            rh_instrument = self._le_rh_inst.text().strip() or "rh_controller"

        return ArrheniusSweepConfig(
            channels=self._parse_channels(),
            T_start=self._spin_t_start.value(),
            T_stop=self._spin_t_stop.value(),
            T_step=self._spin_t_step.value(),
            dwell_s=self._spin_dwell.value(),
            tolerance_C=self._spin_tolerance.value(),
            wait_timeout_s=float(self._spin_timeout.value()),
            electrode_geometry=geom,
            eis_model=self._combo_fit_model.currentText(),
            thermal_model=self._combo_thermal_model.currentText(),
            eis_params={
                "f_hi": self._spin_eis_f_hi.value(),
                "f_lo_mHz": self._spin_eis_f_lo.value(),
                "npts": self._spin_eis_npts.value(),
                "mv_ac": self._spin_eis_mv_ac.value(),
                "mv_dc": self._spin_eis_mv_dc.value(),
            },
            rh_setpoints=rh_setpoints,
            rh_dwell_s=rh_dwell_s,
            rh_instrument=rh_instrument,
            sweep_order={
                "T": self._spin_rank_T.value(),
                "channels": self._spin_rank_ch.value(),
                "RH": self._spin_rank_rh.value(),
            },
        )

    # ── Config save / load ───────────────────────────────────────────────────

    def _on_save_config(self) -> None:
        """Serialise the current UI settings to a JSON file chosen by the user."""
        from PySide6.QtWidgets import QFileDialog
        try:
            config = self._build_config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Arrhenius Config", "",
            "Arrhenius Config (*.json);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(config.to_json())
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_load_config(self) -> None:
        """Load UI settings from a previously saved JSON config file."""
        from PySide6.QtWidgets import QFileDialog
        from softae.analysis.arrhenius import ArrheniusSweepConfig
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Arrhenius Config", "",
            "Arrhenius Config (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                config = ArrheniusSweepConfig.from_json(fh.read())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load config:\n{exc}")
            return
        self._populate_from_config(config)

    def _populate_from_config(self, config) -> None:
        """Fill all UI widgets from an :class:`~softae.analysis.arrhenius.ArrheniusSweepConfig`."""
        # Temperature profile
        self._spin_t_start.setValue(config.T_start)
        self._spin_t_stop.setValue(config.T_stop)
        self._spin_t_step.setValue(config.T_step)
        self._spin_dwell.setValue(config.dwell_s)
        self._spin_tolerance.setValue(config.tolerance_C)
        self._spin_rank_T.setValue(config.sweep_order.get("T", 2))

        # Channels & instruments
        self._le_channels.setText(", ".join(str(c) for c in config.channels))
        self._spin_timeout.setValue(int(config.wait_timeout_s))
        self._spin_rank_ch.setValue(config.sweep_order.get("channels", 3))

        # Electrode geometry
        geom = config.electrode_geometry or {}
        self._spin_L.setValue(geom.get("L_cm", 0.0))
        self._spin_t.setValue(geom.get("t_cm", 0.0))
        self._spin_w.setValue(geom.get("w_cm", 0.0))

        # EIS parameters
        if config.eis_params:
            self._spin_eis_f_hi.setValue(int(config.eis_params.get("f_hi", 100_000)))
            self._spin_eis_f_lo.setValue(int(config.eis_params.get("f_lo_mHz", 100)))
            self._spin_eis_npts.setValue(int(config.eis_params.get("npts", 50)))
            self._spin_eis_mv_ac.setValue(int(config.eis_params.get("mv_ac", 10)))
            self._spin_eis_mv_dc.setValue(int(config.eis_params.get("mv_dc", 0)))
        _restore_combo(self._combo_fit_model, config.eis_model, "eis_model")
        _restore_combo(
            self._combo_thermal_model,
            getattr(config, "thermal_model", "arrhenius"),
            "thermal_model",
        )

        # RH sweep
        if config.rh_setpoints:
            self._rh_grp.setChecked(True)
            sps = config.rh_setpoints
            self._spin_rh_start.setValue(sps[0])
            self._spin_rh_stop.setValue(sps[-1])
            if len(sps) > 1:
                self._spin_rh_step.setValue(round(sps[1] - sps[0], 4))
            self._spin_rh_dwell.setValue(config.rh_dwell_s)
            self._le_rh_inst.setText(config.rh_instrument)
            self._spin_rank_rh.setValue(config.sweep_order.get("RH", 1))
        else:
            self._rh_grp.setChecked(False)

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    def _progress_set(self, value: int) -> None:
        self._progress.setValue(value)

    def _on_eis_point(self, ch: int, T_C: float, sigma: float, R0: float, R1: float, rh_sp: float = float("nan")) -> None:
        """Incrementally update the live Arrhenius plot (GUI thread).

        Keyed by ``(ch, rh_sp)`` so that data from different RH setpoints
        appear as separate series rather than being merged into one line.
        """
        import bisect
        import math
        if math.isnan(T_C) or math.isnan(sigma) or sigma <= 0:
            return
        x_1000_T = 1000.0 / (T_C + 273.15)
        key = (ch, rh_sp)  # rh_sp is nan for non-RH sweeps
        if key not in self._plot_data:
            self._plot_data[key] = ([], [])
        self._plot_data[key][0].append(x_1000_T)
        self._plot_data[key][1].append(sigma)

        # Incremental sorted-list tracking — O(log n) per new channel/RH,
        # O(1) for subsequent points (avoids O(K) set comprehension per call).
        if ch not in self._all_channels:
            bisect.insort(self._all_channels, ch)
        if not math.isnan(rh_sp) and rh_sp not in self._all_rhs:
            bisect.insort(self._all_rhs, rh_sp)

        ch_idx = self._all_channels.index(ch)
        color  = self._PALETTE[ch_idx % len(self._PALETTE)]
        marker = self._MARKERS[ch_idx % len(self._MARKERS)]
        if math.isnan(rh_sp):
            ls = "-"
            label = f"ch{ch}"
        else:
            rh_idx = self._all_rhs.index(rh_sp)
            ls = self._LINESTYLES[rh_idx % len(self._LINESTYLES)]
            label = f"ch{ch} RH={rh_sp:.0f}%"

        if key not in self._plot_lines:
            # First point for this (channel, RH) series — create a new line artist
            line, = self._ax.plot(
                self._plot_data[key][0], self._plot_data[key][1],
                linestyle=ls, marker=marker,
                color=color, markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.5, markersize=6, label=label,
            )
            self._plot_lines[key] = line
            # Rebuild legend only when a new series appears
            self._ax.legend(loc="best", fontsize=8)
        else:
            xs, ys = self._plot_data[key]
            self._plot_lines[key].set_data(xs, ys)

        # Schedule a single deferred redraw; rapid-fire points collapse to one repaint.
        if not self._eis_plot_dirty:
            self._eis_plot_dirty = True
            QTimer.singleShot(0, self._flush_eis_plot)

    def _flush_eis_plot(self) -> None:
        """Deferred axis rescale + canvas repaint (called at most once per event loop tick)."""
        if not self._eis_plot_dirty:
            return
        self._eis_plot_dirty = False
        self._ax.relim()
        self._ax.autoscale_view()
        self._canvas.draw_idle()

    def _store_root(self) -> "Path":
        """The DataStore's root directory, as an absolute path.

        The thing this exists to *not* be is ``Path("softae_data")``, which is
        relative and therefore resolves against the process working directory
        rather than against the store. Launched from anywhere but the repo root,
        an export with no run id landed wherever the GUI happened to be started;
        a stray ``softae_data/`` tree in the repo root is how that was found.

        ``DataStore.project_dir`` is already ``expanduser().resolve()``-ed and is
        the authority when a store exists. With no store, the configured
        ``[data] project_dir`` is expanded **the same way the store expands it**,
        so the fallback lands where the store would have put it rather than
        somewhere merely absolute.
        """
        from pathlib import Path

        project_dir = getattr(self._data_store, "project_dir", None)
        if project_dir is not None:
            return Path(project_dir)
        try:
            from softae.config.loader import data_project_dir

            raw = data_project_dir()
        except Exception:              # no config file reachable at all
            raw = "~/softae_data"
        return Path(raw).expanduser().resolve()

    def _images_dir(self, run_id: str | None) -> "Path":
        """Where this run's exported figures belong. Never CWD-relative.

        ``DataStore.run_dir`` owns the ``runs/<run_id>/`` layout, so it is asked
        whenever there is a store and a run id to ask it about. The no-run-id
        case cannot ask, so it reproduces the same shape under the same root —
        one ``unknown/`` folder inside the store, not one per launch directory.
        """
        from pathlib import Path

        store = self._data_store
        if store is not None and run_id and hasattr(store, "run_dir"):
            return Path(store.run_dir(run_id)) / "images"
        return self._store_root() / "runs" / (run_id or "unknown") / "images"

    def _on_export_plot(self) -> None:
        """Save the current Arrhenius figure to the run's images/ folder.

        The renderer pass (savefig → BytesIO) runs here on the GUI thread
        because it requires a live Qt canvas, but the disk write is offloaded
        to a daemon thread to avoid blocking the event loop.
        """
        import io

        run_id = getattr(self, "_run_id", None)
        images_dir = self._images_dir(run_id)
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir / f"arrhenius_{run_id or 'plot'}.png"
            buf = io.BytesIO()
            self._fig.savefig(buf, format="png", dpi=150)
            buf.seek(0)
            data = buf.getvalue()
            sig = self._sig_log_line

            def _write() -> None:
                try:
                    out_path.write_bytes(data)
                    sig.emit(f"  📁 Plot saved → {out_path}")
                except Exception as exc:
                    sig.emit(f"  ⚠ Export write failed: {exc}")

            threading.Thread(target=_write, daemon=True).start()
        except Exception as exc:
            self._sig_log_line.emit(f"  ⚠ Export failed: {exc}")



    def _on_start(self) -> None:
        from softae.workflows.temp_eis_sweep import ArrheniusSweep

        # Validate sweep rank uniqueness
        r_T = self._spin_rank_T.value()
        r_ch = self._spin_rank_ch.value()
        r_rh = self._spin_rank_rh.value() if self._rh_grp.isChecked() else (
            next(r for r in (1, 2, 3) if r not in {r_T, r_ch})
        )
        if len({r_T, r_ch, r_rh}) != 3:
            QMessageBox.warning(
                self, "Sweep Rank Error",
                "Sweep ranks must be three distinct values (1, 2, and 3 each).\n"
                f"Current: T={r_T}, Channels={r_ch}, RH={r_rh}"
            )
            return

        # Validate config
        try:
            config = self._build_config()
            config.validate()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Configuration Error", str(exc))
            return

        temps = config.resolved_temperatures()
        n_steps = len(temps) * (len(config.channels) + 2)  # set + wait + n_eis per T

        # Register run in DataStore
        run_id: str | None = None
        if self._data_store is not None:
            try:
                run_id = self._data_store.start_run(
                    "arrhenius_sweep",
                    campaign="arrhenius",
                    quality="full",
                    annotation=self._te_annotation.toPlainText().strip(),
                )
            except Exception:
                logger.exception("arrhenius_datastore_start_run_error")

        sweep = ArrheniusSweep(
            config=config,
            manager=self._manager,
            data_store=self._data_store,
            run_id=run_id,
            eis_instrument=self._le_eis_inst.text().strip() or "pico1",
            temp_instrument=self._le_temp_inst.text().strip() or "temp_controller",
        )

        # UI: busy state
        self._btn_start.setEnabled(False)
        self._btn_abort.setEnabled(True)
        self._progress.setRange(0, n_steps)
        self._progress.setValue(0)
        self._lbl_status.setText(
            f"Running — {len(temps)} temps × {len(config.channels)} channels"
        )
        self._log.clear()
        self._abort_requested = False
        self._run_id = run_id
        self._n_steps = n_steps

        self.sweep_status_changed.emit(
            f"Running — {len(temps)} temps × {len(config.channels)} channels",
            0, n_steps,
        )

        # Reset live plot for the new sweep
        self._plot_data = {}
        self._plot_lines = {}
        self._all_channels = []
        self._all_rhs = []
        self._eis_plot_dirty: bool = False
        self._ax.cla()
        self._ax.set_xlabel("1000/T  (K⁻¹)")
        self._ax.set_ylabel("σ (S/cm)")
        self._ax.set_yscale("log")
        self._canvas.draw_idle()

        self._sig_log_line.emit(
            f"Starting sweep: T={temps[0]}–{temps[-1]} °C "
            f"({len(temps)} steps), channels={config.channels}"
        )
        if config.electrode_geometry is None:
            self._sig_log_line.emit(
                "⚠ electrode_geometry not set — σ will be NaN and Arrhenius fit will fail."
            )

        # Wire step callbacks so the log and progress bar update live
        def _on_step_done(step: Any, idx: int, total: int, result: Any, elapsed: float = 0.0) -> None:
            self._sig_log_line.emit(f"  ✓ [{idx + 1}/{total}] {step.name} ({elapsed:.1f}s)")
            self._sig_progress.emit(idx + 1)
            self.sweep_status_changed.emit(f"Step {idx + 1}/{total}: {step.name}", idx + 1, total)

        def _on_step_err(step: Any, idx: int, total: int, error: Exception) -> None:
            self._sig_log_line.emit(f"  ✗ [{idx + 1}/{total}] {step.name}: {error}")
            self.sweep_status_changed.emit(f"Error at step {idx + 1}: {step.name}", idx, total)

        # Wire step callbacks onto the sweep's public hook attributes so the
        # executor inside sweep.run() picks them up.
        sweep.on_step_complete = _on_step_done
        sweep.on_step_error = _on_step_err

        import math as _math

        def _on_eis_point_cb(ch: int, T_C: float, sigma: float, R0: float, R1: float, rh_sp: float = float("nan")) -> None:
            self._sig_eis_point.emit(ch, T_C, sigma, R0, R1, rh_sp)
            r0_s = f"{R0:.1f}" if not _math.isnan(R0) else "—"
            r1_s = f"{R1:.1f}" if not _math.isnan(R1) else "—"
            sig_s = f"{sigma:.3e}" if (not _math.isnan(sigma) and sigma > 0) else "—"
            self._sig_log_line.emit(
                f"     ↳ ch{ch}  R₀={r0_s} Ω  R₁={r1_s} Ω  σ={sig_s} S/cm"
            )

        sweep.on_eis_point = _on_eis_point_cb

        self._sweep = sweep
        self._sweep_thread = threading.Thread(
            target=self._run_sweep_thread,
            args=(sweep,),
            daemon=True,
            name="arrhenius-sweep",
        )
        self._sweep_thread.start()

    def _sweep_run_lock(self, sweep: Any):
        """Hold the cross-process rig lock for the **whole** sweep.

        ``WorkflowExecutor.run`` already takes the lock, so a sweep is not
        unlocked outright — but ``ArrheniusSweep`` runs *one executor per phase*
        and an RH sweep runs several, each acquiring and releasing in turn. The
        gaps between them are not idle: ``_run_rh_sweep`` starts the RH
        controller, writes its setpoint and polls it to stabilise **outside any
        executor**, and its ``finally`` writes the temperature back to ambient
        the same way. A headless tool starting in one of those windows would find
        the lock file free and take the rig out from under a board sitting at
        setpoint. Held here, those windows close.

        Nesting is safe by construction: ``acquire_run_lock`` is re-entrant per
        process, and the executor's ``mine_already`` discipline means it will not
        release a lock it did not create — so the release stays with this block.

        Simulated rigs are exempt, using the executor's own predicate rather than
        a second notion of "real": a mock run holding the lock would turn a dry
        run into an outage for a real one.
        """
        from contextlib import nullcontext

        from softae.core.run_lock import held_run_lock, rig_is_simulated

        if rig_is_simulated(self._manager):
            return nullcontext()
        run_id = getattr(sweep, "run_id", None)
        return held_run_lock(what=f"Arrhenius sweep ({run_id or 'no run id'})")

    def _run_sweep_thread(
        self,
        sweep: Any,
    ) -> None:
        """Run the sweep in a fresh event loop (daemon thread)."""
        from softae.core.run_lock import RunLockHeld, busy_rig_message

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Re-create per-instrument asyncio.Lock objects so they bind to THIS
        # event loop, not the stale loop from a previous sweep run.
        try:
            self._manager.reset_locks()
        except Exception:
            pass
        # Clear any stale abort/stop flags on instruments from a previous sweep.
        try:
            for inst_name in [
                sweep.temp_instrument,
                sweep.config.rh_instrument,
            ]:
                inst = self._manager.get(inst_name)
                for flag in ("_stop_wait", "_wait_abort"):
                    ev = getattr(inst, flag, None)
                    if ev is not None:
                        ev.clear()
        except Exception:
            pass
        try:
            # Claim the rig for the sweep. **Whole-rig, and not the three
            # instruments the sweep commands.** There is no ``Workflow`` object
            # spanning a sweep to derive a scope from — ``ArrheniusSweep`` builds
            # a fresh one per phase as it goes — so the alternative is the guess
            # ``{temp, rh, eis}``, which excludes the syringe and the stage and
            # would therefore leave the anti-clog purge free to travel the stage
            # to the flush basin and dispense while a board sits at setpoint
            # mid-dwell. Commanded is not the same as occupied; widening on
            # doubt is the direction ``RigActivity.conflicts`` asks for.
            #
            # ``manage_rest=False``: the sweep drives no fluidics, so the tip is
            # better left resting in flush for the hours it lasts than retracted
            # into air, and travelling the stage home afterwards would be motion
            # this run never asked for.
            #
            # The rig lock is the *other* axis, and it goes outermost: it says
            # no second **process** may drive this rig, where the claim says no
            # second activity in *this* process may. Outermost so a refusal
            # happens before any claim is taken and before the event loop is
            # handed a sweep it cannot run.
            with self._sweep_run_lock(sweep), \
                    rig_run(self,
                            f"arrhenius:{getattr(sweep, 'run_id', None) or 'sweep'}",
                            instruments=None, manage_rest=False):
                results = loop.run_until_complete(sweep.run())
            n_ok = sum(1 for r in results if r.fit_success)
            # Model-aware per-channel summary in the log.
            for r in results:
                if not r.fit_success:
                    continue
                if getattr(r, "model", "arrhenius") == "vft":
                    self._sig_log_line.emit(
                        f"     ↳ ch{r.channel} VFT: Eₐ={r.Ea_eV:.3f} eV  "
                        f"T₀={r.T0_C:.1f} °C  R²={r.R_squared:.4f}"
                    )
                else:
                    self._sig_log_line.emit(
                        f"     ↳ ch{r.channel} Arrhenius: Eₐ={r.Ea_eV:.3f} eV  "
                        f"R²={r.R_squared:.4f}"
                    )
            model_tag = sweep.config.thermal_model.upper()
            msg = (
                f"Complete — {n_ok}/{len(results)} channels fitted [{model_tag}] "
                f"(run_id={self._run_id or '—'})"
            )
            self._sig_sweep_done.emit(True, msg)
        except RunLockHeld as exc:
            # Never a bare "busy": the operator's only recourse against an
            # anonymous refusal is to start deleting lock files, so the holder is
            # named and every exit is spelled out. The full message goes to the
            # log, where multiple lines render; the status label gets one line.
            logger.warning("arrhenius_sweep_rig_held", holder=exc.lock.describe())
            self._sig_log_line.emit(
                "  ⚠ " + busy_rig_message(exc.lock, action="This Arrhenius sweep")
            )
            self._sig_sweep_done.emit(
                False,
                f"Rig busy — PID {exc.lock.pid} is running "
                f"{exc.lock.what or 'an unnamed run'}",
            )
        except Exception as exc:
            logger.exception("arrhenius_sweep_thread_error", error=str(exc))
            self._sig_sweep_done.emit(False, str(exc))
        finally:
            # Cancel any tasks that were left pending by an abort or exception so
            # they don't continue running and emitting signals after the loop closes.
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for _task in pending:
                        _task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            loop.close()


    def _on_abort(self) -> None:
        self._abort_requested = True
        self._lbl_status.setText("Aborting…")
        self._btn_abort.setEnabled(False)
        self._log.append("  \u26a0 Abort requested \u2014 stopping sweep and cleaning up hardware\u2026")
        sweep = getattr(self, "_sweep", None)
        if sweep is not None:
            # abort() sets the abort flag (wakes any RH dwell sleep early)
            # and cancels the workflow executor \u2014 both are thread-safe.
            try:
                sweep.abort()
            except Exception:
                pass

    # \u2500\u2500 Daemon shutdown seam (hardware safety: abort before any join) \u2500\u2500\u2500\u2500
    def _abort_run_impl(self) -> None:
        self._abort_requested = True
        sweep = getattr(self, "_sweep", None)
        if sweep is not None:
            sweep.abort()   # sets abort flag, wakes temp/RH waits, cancels executor

    def _runner_thread(self):
        return getattr(self, "_sweep_thread", None)

    def _on_sweep_done(self, success: bool, message: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_abort.setEnabled(False)
        self._lbl_status.setText(message)
        n = getattr(self, "_n_steps", 0)
        self.sweep_status_changed.emit(message, n if success else 0, n)
        if success:
            self._log.append(f"\n✓ {message}")
            self._on_export_plot()
            # If RH sweep was run, draw 3D surface when >1 RH point
            sweep = getattr(self, "_sweep", None)
            if sweep is not None and hasattr(sweep, "rh_results") and sweep.rh_results:
                self._draw_rh_arrhenius(sweep.rh_results)
                if len(sweep.rh_results) > 1:
                    self._btn_show_3d.setEnabled(True)
            if self._data_store is not None and self._run_id:
                try:
                    self._data_store.finish_run(self._run_id)
                except Exception:
                    pass
        else:
            self._log.append(f"\n✗ Error: {message}")
            if self._data_store is not None and self._run_id:
                try:
                    self._data_store.finish_run(self._run_id, status="error")
                except Exception:
                    pass

    def _draw_rh_arrhenius(self, rh_results: dict) -> None:
        """Draw RH-parameterised Arrhenius: 2-D overlay on the live plot, plus
        a per-channel 3-D pop-out window when more than one RH point was swept.

        The 3-D window shows one channel at a time; a horizontal slider at the
        bottom cycles through all channels.  The canvas auto-fills the window.
        """
        import numpy as np

        # ── store for on-demand re-open ──────────────────────────────────────
        self._last_rh_results = rh_results

        # ── 2-D overlay ──────────────────────────────────────────────────────
        self._ax.cla()
        self._ax.set_xlabel("1000 / T  (K⁻¹)")
        self._ax.set_ylabel("σ (S/cm)")
        self._ax.set_yscale("log")
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        line_idx = 0
        for rh_sp, results in sorted(rh_results.items()):
            for res in results:
                temps_K = np.array(res.temperatures_C) + 273.15
                sigmas = np.array(res.conductivities)
                valid = np.isfinite(sigmas) & (sigmas > 0)
                if valid.any():
                    color = colors[line_idx % len(colors)]
                    self._ax.plot(
                        1000.0 / temps_K[valid], sigmas[valid], "o-",
                        color=color,
                        label=f"Ch{res.channel} RH={rh_sp:.0f}%",
                        linewidth=1.5, markersize=5,
                    )
                    line_idx += 1
        self._ax.legend(fontsize=7, loc="best")
        self._canvas.draw_idle()

        # ── 3-D pop-out + static image saves (multi-RH only) ─────────────────
        if len(rh_results) > 1:
            self._open_3d_popout(rh_results)
            # Offload per-channel 3-D figure creation and savefig to a daemon
            # thread — these are headless matplotlib Figures (no Qt canvas),
            # so they are safe to create and save off the GUI thread.
            threading.Thread(
                target=self._save_3d_plots_to_images,
                args=(rh_results,),
                daemon=True,
                name="arrhenius-3d-export",
            ).start()

    def _save_3d_plots_to_images(self, rh_results: dict) -> None:
        """Render a per-channel 3-D RH-Arrhenius figure and save it as a PNG
        in the run's images/ folder.  Uses a headless matplotlib Figure so the
        GUI thread is not blocked by any display infrastructure.
        """
        try:
            import numpy as np
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            run_id = getattr(self, "_run_id", None)
            images_dir = self._images_dir(run_id)
            images_dir.mkdir(parents=True, exist_ok=True)

            rh_vals = sorted(rh_results.keys())
            ch_ids = sorted({r.channel for rlist in rh_results.values() for r in rlist})

            for ch_id in ch_ids:
                fig = Figure(tight_layout=True)
                ax3 = fig.add_subplot(1, 1, 1, projection="3d")
                ax3.set_title(f"Channel {ch_id} — RH-Arrhenius", pad=10)
                ax3.set_xlabel("1000/T  (K\u207b\u00b9)", labelpad=8)
                ax3.set_ylabel("RH (%)", labelpad=8)
                ax3.set_zlabel("log\u2081\u2080(\u03c3)", labelpad=8)
                for rh_sp in rh_vals:
                    for res in rh_results[rh_sp]:
                        if res.channel != ch_id:
                            continue
                        temps_K = np.array(res.temperatures_C) + 273.15
                        sigmas = np.array(res.conductivities)
                        valid = np.isfinite(sigmas) & (sigmas > 0)
                        if valid.any():
                            ax3.plot(
                                1000.0 / temps_K[valid],
                                np.full(valid.sum(), rh_sp),
                                np.log10(sigmas[valid]),
                                "o-", linewidth=1.5, markersize=4,
                                label=f"RH={rh_sp:.0f}%",
                            )
                ax3.legend(fontsize=7, loc="best")
                out_path = images_dir / f"rh_3d_ch{ch_id}_{run_id or 'plot'}.png"
                fig.savefig(str(out_path), dpi=150)
                self._sig_log_line.emit(f"  \U0001f4c1 3D plot saved \u2192 {out_path}")
        except Exception as exc:
            self._sig_log_line.emit(f"  \u26a0 3D plot save failed: {exc}")

    def _open_3d_popout(self, rh_results: dict) -> None:
        """Open (or re-open) the per-channel 3-D Arrhenius pop-out window."""
        import numpy as np
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from PySide6.QtWidgets import QHBoxLayout, QScrollBar, QSizePolicy

            rh_vals = sorted(rh_results.keys())
            ch_ids  = sorted({r.channel for rlist in rh_results.values() for r in rlist})

            win = QWidget(None, Qt.WindowType.Window)
            win.setWindowTitle("RH-Arrhenius 3D")
            vlayout = QVBoxLayout(win)
            vlayout.setContentsMargins(6, 6, 6, 6)
            vlayout.setSpacing(4)

            fig3d   = Figure(tight_layout=True)
            canvas3d = FigureCanvasQTAgg(fig3d)
            canvas3d.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            vlayout.addWidget(canvas3d, stretch=1)

            # ── navigation row ───────────────────────────────────────────────
            nav_row = QHBoxLayout()
            lbl_ch  = QLabel()
            lbl_ch.setFixedWidth(140)
            nav_row.addWidget(lbl_ch)

            slider = QScrollBar(Qt.Orientation.Horizontal)
            slider.setRange(0, len(ch_ids) - 1)
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.setPageStep(1)
            nav_row.addWidget(slider, stretch=1)
            vlayout.addLayout(nav_row)

            # Pre-compute per-channel plotting arrays once, outside the render closure.
            _ch_plot_data: dict[int, list[tuple]] = {}
            for ch_id in ch_ids:
                series: list[tuple] = []
                for rh_sp in rh_vals:
                    for res in rh_results[rh_sp]:
                        if res.channel != ch_id:
                            continue
                        temps_K = np.array(res.temperatures_C) + 273.15
                        sigmas  = np.array(res.conductivities)
                        valid   = np.isfinite(sigmas) & (sigmas > 0)
                        if valid.any():
                            series.append((
                                1000.0 / temps_K[valid],
                                np.full(valid.sum(), rh_sp),
                                np.log10(sigmas[valid]),
                                f"RH={rh_sp:.0f}%",
                            ))
                _ch_plot_data[ch_id] = series

            def _render_channel(idx: int) -> None:
                ch = ch_ids[idx]
                lbl_ch.setText(f"  Channel {ch}  ({idx + 1}/{len(ch_ids)})")
                fig3d.clear()
                ax3 = fig3d.add_subplot(1, 1, 1, projection="3d")
                ax3.set_title(f"Channel {ch} — RH-Arrhenius", pad=10)
                ax3.set_xlabel("1000/T  (K⁻¹)", labelpad=8)
                ax3.set_ylabel("RH (%)", labelpad=8)
                ax3.set_zlabel("log₁₀(σ)", labelpad=8)
                for xs, ys, zs, lbl in _ch_plot_data[ch]:
                    ax3.plot(xs, ys, zs, "o-", linewidth=1.5, markersize=4, label=lbl)
                if _ch_plot_data[ch]:
                    ax3.legend(fontsize=7, loc="best")
                canvas3d.draw_idle()

            # Debounce slider: buffer rapid drag events; render only after 150 ms idle.
            _debounce_timer = QTimer()
            _debounce_timer.setSingleShot(True)
            _debounce_timer.setInterval(150)
            _pending_idx: list[int] = [0]

            def _on_slider_changed(idx: int) -> None:
                _pending_idx[0] = idx
                _debounce_timer.start()

            _debounce_timer.timeout.connect(lambda: _render_channel(_pending_idx[0]))
            slider.valueChanged.connect(_on_slider_changed)
            _render_channel(0)

            win.resize(720, 580)
            win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            win.show()
            # Keep a reference so Python doesn't GC the window, but prune closed
            # windows first so the list doesn't grow unboundedly.
            if not hasattr(self, "_rh_3d_windows"):
                self._rh_3d_windows: list = []
            self._rh_3d_windows = [w for w in self._rh_3d_windows if _widget_alive(w)]
            self._rh_3d_windows.append(win)
        except Exception as exc:
            logger.warning("rh_3d_plot_failed", error=str(exc))

    def _on_show_3d(self) -> None:
        """Re-open the 3D pop-out for the most recent RH sweep."""
        if self._last_rh_results and len(self._last_rh_results) > 1:
            self._open_3d_popout(self._last_rh_results)

    # Public API for the main window to inject a data_store after construction
    def set_data_store(self, data_store: "DataStore") -> None:
        self._data_store = data_store
