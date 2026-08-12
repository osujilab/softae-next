"""Tab 3: Monitoring.

Live rolling plots for temperature (stage PV/SP overlay + chamber PV trace),
humidity (with setpoint overlay), stage position, instrument log, and
workflow progress.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from softae.gui.widgets.live_plot import LivePlotWidget
from softae.gui.widgets.sweep_status_widget import SweepStatusWidget
from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class _PollingWorker(StoppableWorker):
    """Background thread that polls instrument drivers every 2 s.

    Emits ``poll_done`` with a plain dict so the GUI thread performs
    zero blocking I/O.
    """

    poll_done = Signal(dict)

    _default_stop_timeout_ms = 5000

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._stop = False

    def run(self) -> None:
        import math
        self._stop = False
        while not self._stop:
            data = self._collect()
            self.poll_done.emit(data)
            self.msleep(2000)

    def _collect(self) -> dict:
        import math
        out: dict = {}
        try:
            tc = self._manager.get("temp_controller")
            out["temp_pv"] = tc.get_pv()
            out["temp_sp"] = tc.get_sp()
        except Exception:
            pass
        try:
            rh_ctrl = self._manager.get("rh_controller")
            # get_TH reads %RH and chamber T from one sensor transaction.
            if callable(getattr(rh_ctrl, "get_TH", None)):
                t, h = rh_ctrl.get_TH()
                if t is not None and not math.isnan(t):
                    out["chamber_temp"] = t
            else:
                h = rh_ctrl.get_H()
            sp_rh = getattr(rh_ctrl, "_setpoint", float("nan"))
            if callable(getattr(rh_ctrl, "status", None)):
                st = rh_ctrl.status()
                sp_rh = st.get("setpoint", sp_rh)
            out["rh"] = h
            out["rh_sp"] = sp_rh
        except Exception:
            pass
        try:
            stage = self._manager.get("stage")
            pos = stage.live_position()
            out["pos_x"] = pos[0]
            out["pos_y"] = pos[1]
        except Exception:
            pass
        return out

    def _request_stop(self) -> None:
        self._stop = True


class MonitoringTab(QWidget):
    """Real-time instrument monitoring dashboard."""

    HISTORY_LEN = 300  # ~10 min at 2 s polling

    def __init__(self, manager: InstrumentManager, *, poller=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._poller = poller  # shared InstrumentPoller (or None → local worker)
        self._poll_worker: _PollingWorker | None = None
        self._temp_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._temp_sp_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._chamber_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._rh_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._rh_sp_history: deque[float] = deque(maxlen=self.HISTORY_LEN)
        self._build_ui()
        self._start_polling()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter()

        # --- Left column: Live Readouts | Stage Position | Workflow Progress ---
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Row 1: Live Readouts (large)
        plots_grp = QGroupBox("Live Readouts")
        plots_layout = QVBoxLayout(plots_grp)

        self._temp_plot = LivePlotWidget(
            title="Temperature (°C)",
            color="red",
            y_label="°C",
            pv_label="Stage PV",
            sp_label="Stage SP",
            second_color="darkorange",
            second_label="Chamber PV",
        )
        plots_layout.addWidget(self._temp_plot)

        self._rh_plot = LivePlotWidget(
            title="Relative Humidity (%)", color="blue", y_label="%RH"
        )
        plots_layout.addWidget(self._rh_plot)

        self._lbl_temp_pv = QLabel("Stage PV: -- °C")
        self._lbl_temp_sp = QLabel("Stage SP: -- °C")
        self._lbl_chamber_pv = QLabel("Chamber PV: -- °C")
        self._lbl_rh_pv = QLabel("RH: -- %")
        self._lbl_rh_sp = QLabel("RH SP: -- %")
        for lbl in (
            self._lbl_temp_pv,
            self._lbl_temp_sp,
            self._lbl_chamber_pv,
            self._lbl_rh_pv,
            self._lbl_rh_sp,
        ):
            plots_layout.addWidget(lbl)

        left_splitter.addWidget(plots_grp)

        # Row 2: Stage Position (slim)
        pos_grp = QGroupBox("Stage Position")
        pos_layout = QVBoxLayout(pos_grp)
        self._lbl_pos = QLabel("X: --  Y: --")
        pos_layout.addWidget(self._lbl_pos)
        left_splitter.addWidget(pos_grp)

        # Row 3: Workflow Progress
        wf_grp = QGroupBox("Workflow Progress")
        wf_layout = QVBoxLayout(wf_grp)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        wf_layout.addWidget(self._progress_bar)
        self._lbl_workflow = QLabel("No workflow running")
        wf_layout.addWidget(self._lbl_workflow)
        left_splitter.addWidget(wf_grp)

        # Row 4: Arrhenius Sweep Status
        self._sweep_status = SweepStatusWidget()
        left_splitter.addWidget(self._sweep_status)

        left_splitter.setStretchFactor(0, 4)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(2, 2)
        left_splitter.setStretchFactor(3, 2)

        splitter.addWidget(left_splitter)

        # --- Right column: Instrument Log (full height, expanded) ---
        log_grp = QGroupBox("Instrument Log")
        log_layout = QVBoxLayout(log_grp)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        log_layout.addWidget(self._log_view)
        splitter.addWidget(log_grp)

        # Left column gets slightly less space than the log
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

    def _start_polling(self) -> None:
        if self._poller is not None:
            self._poller.monitor_ready.connect(self._on_poll_done)
            self._poll_worker = None  # shared poller handles all I/O
        else:
            self._poll_worker = _PollingWorker(self._manager, parent=self)
            self._poll_worker.poll_done.connect(self._on_poll_done)
            self._poll_worker.start()

    def _on_poll_done(self, data: dict) -> None:
        import math
        # Temperature — Stage PV/SP (temp_controller) + Chamber PV (RH sensor)
        pv = data.get("temp_pv")
        sp = data.get("temp_sp")
        ct = data.get("chamber_temp")
        if pv is not None:
            self._temp_history.append(pv)
            self._temp_sp_history.append(sp if sp is not None else pv)
            # Keep the chamber trace length-aligned with the stage trace;
            # a missing chamber reading is a NaN gap, not a dropped sample.
            self._chamber_history.append(
                ct if (ct is not None and not math.isnan(ct)) else float("nan")
            )
            self._temp_plot.update_data(
                list(self._temp_history),
                setpoint_values=list(self._temp_sp_history),
                second_values=list(self._chamber_history),
            )
            self._lbl_temp_pv.setText(f"Stage PV: {pv:.1f} °C")
        if sp is not None:
            self._lbl_temp_sp.setText(f"Stage SP: {sp:.1f} °C")
        if ct is not None and not math.isnan(ct):
            self._lbl_chamber_pv.setText(f"Chamber PV: {ct:.1f} °C")

        # Humidity
        h = data.get("rh")
        sp_rh = data.get("rh_sp", float("nan"))
        if h is not None and not math.isnan(h):
            self._rh_history.append(h)
            self._rh_sp_history.append(sp_rh if not math.isnan(sp_rh) else h)
            self._rh_plot.update_data(
                list(self._rh_history),
                setpoint_values=list(self._rh_sp_history),
            )
            self._lbl_rh_pv.setText(f"RH: {h:.1f} %")
        if sp_rh is not None and not math.isnan(sp_rh):
            self._lbl_rh_sp.setText(f"RH SP: {sp_rh:.1f} %")

        # Stage position
        pos_x = data.get("pos_x")
        pos_y = data.get("pos_y")
        if pos_x is not None and pos_y is not None:
            self._lbl_pos.setText(f"X: {pos_x:.2f}  Y: {pos_y:.2f}")

    # --- Public helpers for workflow integration --

    def set_workflow_progress(self, current: int, total: int, label: str = "") -> None:
        """Update the workflow progress bar from outside the tab."""
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
            self._lbl_workflow.setText(label or f"Step {current}/{total}")
        else:
            self._progress_bar.setValue(0)
            self._lbl_workflow.setText("No workflow running")

    def update_sweep_status(self, text: str, current: int, total: int) -> None:
        """Forward Arrhenius sweep progress to the SweepStatusWidget.

        Called by :class:`~softae.gui.tabs.tab_arrhenius.ArrheniusTab` via a
        ``sweep_status_changed`` signal wired in *MainWindow*.
        """
        if total > 0:
            self._sweep_status._progress.setRange(0, total)
            self._sweep_status._progress.setValue(current)
            self._sweep_status._progress.setFormat(f"%v / {total} steps")
        self._sweep_status._lbl_status.setText(text)

    def append_log(self, message: str) -> None:
        """Add a line to the instrument log viewer."""
        self._log_view.appendPlainText(message)

    def cleanup(self) -> None:
        """Stop this tab's worker thread (idempotent; no-op under a shared poller)."""
        if self._poll_worker is not None and self._poll_worker.isRunning():
            self._poll_worker.stop_worker()

    def hideEvent(self, event) -> None:
        if self._poll_worker is not None and self._poll_worker.isRunning():
            self._poll_worker.stop_worker()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        if self._poll_worker is not None and not self._poll_worker.isRunning():
            self._poll_worker.start()
        super().showEvent(event)


