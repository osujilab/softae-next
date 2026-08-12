"""Compact always-visible monitoring sidebar.

Displays camera feeds, live stage position, dispenser head status,
temperature/RH SP & PV readings, and per-tab workflow status — all
within a narrow column (≤ 15 % of the main window width).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.gui.widgets.camera_worker import CameraWorker
    from softae.gui.widgets.webcam_worker import WebcamWorker
    from softae.server.manager import InstrumentManager


class _SidebarPollWorker(StoppableWorker):
    """Background thread that polls instruments every 2 s without blocking the GUI.

    All serial I/O is performed here; results are emitted via ``poll_done``
    as a plain dict so the GUI thread only needs to update labels.
    """

    poll_done = Signal(dict)

    def __init__(self, manager: "InstrumentManager", parent=None) -> None:
        super().__init__(parent)
        self._manager = manager

    def run(self) -> None:  # noqa: C901
        while not self.isInterruptionRequested():
            out: dict = {}
            try:
                stage = self._manager.get("stage")
                pos = stage.live_position()
                out["stage_pos"] = (float(pos[0]), float(pos[1]))
            except Exception:
                pass
            try:
                syringe = self._manager.get("syringe")
                is_up = getattr(syringe, "_is_up", None)
                if is_up is None and callable(getattr(syringe, "status", None)):
                    is_up = syringe.status().get("is_up")
                out["syringe_is_up"] = is_up
            except Exception:
                pass
            try:
                tc = self._manager.get("temp_controller")
                out["temp_sp"] = tc.get_sp()
                out["temp_pv"] = tc.get_pv()
            except Exception:
                pass
            try:
                rh_ctrl = self._manager.get("rh_controller")
                st = rh_ctrl.status()
                out["rh_sp"] = st.get("setpoint", float("nan"))
                out["rh_pv"] = st.get("current_rh", float("nan"))
                # Chamber T comes from the same RH sensor (get_TH shares the
                # read with %RH); fall back to the cached status value.
                ct = st.get("chamber_temp", float("nan"))
                if callable(getattr(rh_ctrl, "get_T", None)):
                    try:
                        ct = rh_ctrl.get_T()
                    except Exception:
                        pass
                out["chamber_temp"] = ct
            except Exception:
                pass
            self.poll_done.emit(out)
            self.msleep(2000)


class _AspectLabel(QLabel):
    """QLabel that maintains a 16:9 aspect ratio as its width changes.

    Overrides ``hasHeightForWidth`` / ``heightForWidth`` so Qt's layout
    engine automatically adjusts the label's height whenever the sidebar
    is resized.  The ``QScrollArea`` in the sidebar shows a vertical
    scrollbar if the taller feeds push the status section out of view.
    """

    _RATIO: float = 9.0 / 16.0

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return max(60, round(width * self._RATIO))


class MonitorSidebar(QWidget):
    """Compact always-visible monitoring sidebar (~15 % window width).

    Sections (top → bottom):
    - ThorCam thumbnail feed
    - Webcam thumbnail feed
    - Stage X / Y live position
    - Dispenser head status (Retracted ↑ / Descended ↓)
    - Temperature: Stage SP / PV + Chamber (RH-sensor onboard T)
    - Humidity SP / PV
    - Workflow status: Arrhenius · HT Exp · Autonomous
    """

    def __init__(self, manager: InstrumentManager, *, poller=None, parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._poller = poller  # shared InstrumentPoller (or None → local worker)

        # Minimum width prevents it from collapsing to nothing; no maximum
        # so the QSplitter sash can expand the sidebar freely.
        self.setMinimumWidth(130)

        self._build_ui()
        self._start_polling()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(3)
        scroll.setWidget(container)

        # ── ThorCam ──────────────────────────────────────────────────────────
        cam_grp = QGroupBox("ThorCam")
        cam_grp.setFlat(True)
        cam_lay = QVBoxLayout(cam_grp)
        cam_lay.setContentsMargins(2, 2, 2, 2)
        self._lbl_thor = self._cam_label()
        cam_lay.addWidget(self._lbl_thor)
        layout.addWidget(cam_grp)

        # ── Webcam ───────────────────────────────────────────────────────────
        webcam_grp = QGroupBox("Webcam")
        webcam_grp.setFlat(True)
        wc_lay = QVBoxLayout(webcam_grp)
        wc_lay.setContentsMargins(2, 2, 2, 2)
        wc_lay.setSpacing(2)

        # Exposure slider row
        exp_row = QHBoxLayout()
        self._lbl_wc_exp_title = QLabel("Exp:")
        self._lbl_wc_exp_title.setStyleSheet("font-size: 8pt;")
        exp_row.addWidget(self._lbl_wc_exp_title)
        self._slider_wc_exp = QSlider(Qt.Orientation.Horizontal)
        self._slider_wc_exp.setRange(-9, -1)
        self._slider_wc_exp.setValue(-7)
        self._slider_wc_exp.valueChanged.connect(self._on_wc_exposure_changed)
        exp_row.addWidget(self._slider_wc_exp)
        self._lbl_wc_exp_val = QLabel("-7")
        self._lbl_wc_exp_val.setStyleSheet("font-size: 8pt;")
        exp_row.addWidget(self._lbl_wc_exp_val)
        # Manual restart for a webcam that failed to open / dropped its feed.
        self._btn_wc_retry = QPushButton("↻")
        self._btn_wc_retry.setToolTip("Restart the webcam feed")
        self._btn_wc_retry.setStyleSheet("font-size: 8pt;")
        self._btn_wc_retry.setFixedWidth(24)
        self._btn_wc_retry.clicked.connect(self._on_wc_retry)
        exp_row.addWidget(self._btn_wc_retry)
        wc_lay.addLayout(exp_row)

        self._lbl_webcam = self._cam_label()
        wc_lay.addWidget(self._lbl_webcam)
        layout.addWidget(webcam_grp)

        layout.addWidget(self._separator())

        # ── Stage position ────────────────────────────────────────────────────
        stage_grp = QGroupBox("Stage")
        stage_grp.setFlat(True)
        stage_lay = QVBoxLayout(stage_grp)
        stage_lay.setContentsMargins(4, 2, 4, 2)
        self._lbl_stage = self._data_label("X: --\nY: --")
        stage_lay.addWidget(self._lbl_stage)
        layout.addWidget(stage_grp)

        # ── Dispenser head ────────────────────────────────────────────────────
        head_grp = QGroupBox("Dispenser Head")
        head_grp.setFlat(True)
        head_lay = QVBoxLayout(head_grp)
        head_lay.setContentsMargins(4, 2, 4, 2)
        self._lbl_head = self._data_label("--")
        head_lay.addWidget(self._lbl_head)
        layout.addWidget(head_grp)

        layout.addWidget(self._separator())

        # ── Temperature ───────────────────────────────────────────────────────
        # Stage SP/PV come from the temp_controller; Chamber is the RH sensor's
        # onboard temperature reading.
        temp_grp = QGroupBox("Temperature")
        temp_grp.setFlat(True)
        temp_lay = QVBoxLayout(temp_grp)
        temp_lay.setContentsMargins(4, 2, 4, 2)
        self._lbl_t_sp = self._data_label("Stage SP: --")
        self._lbl_t_pv = self._data_label("Stage PV: --")
        self._lbl_chamber = self._data_label("Chamber: --")
        temp_lay.addWidget(self._lbl_t_sp)
        temp_lay.addWidget(self._lbl_t_pv)
        temp_lay.addWidget(self._lbl_chamber)
        layout.addWidget(temp_grp)

        # ── Humidity ──────────────────────────────────────────────────────────
        rh_grp = QGroupBox("Humidity")
        rh_grp.setFlat(True)
        rh_lay = QVBoxLayout(rh_grp)
        rh_lay.setContentsMargins(4, 2, 4, 2)
        self._lbl_rh_sp = self._data_label("SP: --")
        self._lbl_rh_pv = self._data_label("PV: --")
        rh_lay.addWidget(self._lbl_rh_sp)
        rh_lay.addWidget(self._lbl_rh_pv)
        layout.addWidget(rh_grp)

        layout.addWidget(self._separator())

        # ── Workflow status ───────────────────────────────────────────────────
        wf_grp = QGroupBox("Workflow Status")
        wf_grp.setFlat(True)
        wf_lay = QVBoxLayout(wf_grp)
        wf_lay.setContentsMargins(4, 2, 4, 2)
        self._lbl_wf_arrhenius = self._status_label("Arrhenius: Idle")
        self._lbl_wf_ht = self._status_label("HT Exp: Idle")
        self._lbl_wf_auto = self._status_label("Autonomous: Idle")
        wf_lay.addWidget(self._lbl_wf_arrhenius)
        wf_lay.addWidget(self._lbl_wf_ht)
        wf_lay.addWidget(self._lbl_wf_auto)
        layout.addWidget(wf_grp)

        layout.addStretch()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _cam_label(self) -> _AspectLabel:
        lbl = _AspectLabel("No feed")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ignored horizontally (layout drives width, not pixmap sizeHint).
        # Preferred vertically with height-for-width: Qt asks heightForWidth(w)
        # so the feed scales to 16:9 as the sidebar widens.
        sp = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        lbl.setSizePolicy(sp)
        lbl.setStyleSheet("background:#111; color:#555; border:1px solid #333; font-size:8pt;")
        return lbl

    def _data_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        font = QFont()
        font.setPointSize(8)
        lbl.setFont(font)
        lbl.setWordWrap(True)
        return lbl

    def _status_label(self, text: str) -> QLabel:
        lbl = self._data_label(text)
        lbl.setStyleSheet("color: #666; font-size: 8pt;")
        return lbl

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # ── Camera feed connections ───────────────────────────────────────────────

    def set_camera_worker(self, worker: CameraWorker) -> None:
        """Connect ThorCam frame_ready signal to the sidebar thumbnail."""
        worker.frame_ready.connect(self._on_thor_frame)

    def set_webcam_worker(self, worker: WebcamWorker) -> None:
        """Connect webcam frame_ready signal to the sidebar thumbnail."""
        self._webcam_worker = worker
        self._latest_wc_frame: np.ndarray | None = None
        # Store each arriving frame (cheap); render only on timer tick.
        # This coalesces any burst of frame_ready signals that queue up
        # behind a GUI-thread stall so at most one render happens per tick.
        worker.frame_ready.connect(self._store_wc_frame)
        # Surface failures (open/read) so the operator knows to hit Retry.
        worker.error_occurred.connect(self._on_wc_error)
        self._wc_frame_timer = QTimer(self)
        self._wc_frame_timer.timeout.connect(self._wc_tick)
        self._wc_frame_timer.start(85)  # ~12 FPS

    def _on_wc_retry(self) -> None:
        """Restart the webcam acquisition thread (reopens the camera)."""
        worker = getattr(self, "_webcam_worker", None)
        if worker is None:
            return
        worker.stop_worker()  # joins if running; no-op if already stopped
        if worker.isRunning():
            # Couldn't join (e.g. a wedged capture) — don't double-start().
            self._lbl_webcam.setText("Webcam busy —\nretry shortly")
            return
        self._latest_wc_frame = None
        self._lbl_webcam.setText("Restarting…")
        worker.start()  # run() resets its abort flag and reopens the camera

    def _on_wc_error(self, message: str) -> None:
        """Show a webcam open/read failure in the feed label."""
        self._lbl_webcam.setText(f"Webcam error:\n{message}")

    def _store_wc_frame(self, arr: object) -> None:
        """Store the latest frame; the 85 ms timer will render it."""
        self._latest_wc_frame = arr  # type: ignore[assignment]

    def _wc_tick(self) -> None:
        """Request a new frame, then render the last one that arrived."""
        worker = getattr(self, "_webcam_worker", None)
        if worker is not None:
            worker.request_frame()
        frame = self._latest_wc_frame
        if frame is not None:
            self._latest_wc_frame = None
            self._update_cam_label(self._lbl_webcam, frame)

    def _on_wc_exposure_changed(self, value: int) -> None:
        """Forward exposure slider change to the webcam worker."""
        self._lbl_wc_exp_val.setText(str(value))
        worker = getattr(self, "_webcam_worker", None)
        if worker is not None:
            worker.set_exposure(float(value))

    def _on_thor_frame(self, arr: object) -> None:
        self._update_cam_label(self._lbl_thor, arr)

    def _on_webcam_frame(self, arr: object) -> None:
        """Kept for backward-compat; normally replaced by _store_wc_frame."""
        self._update_cam_label(self._lbl_webcam, arr)

    def _update_cam_label(self, label: QLabel, arr: object) -> None:
        try:
            a = np.ascontiguousarray(arr)
            if a.ndim != 3 or a.shape[2] != 3:
                return
            h, w, _ = a.shape
            qimg = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            px = QPixmap.fromImage(qimg)
            label.setPixmap(
                px.scaled(
                    label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        except Exception:
            pass

    # ── Instrument polling ────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        if self._poller is not None:
            self._poller.sidebar_ready.connect(self._on_poll_done)
            self._poll_worker = None  # shared poller handles all I/O
        else:
            self._poll_worker = _SidebarPollWorker(self._manager, parent=self)
            self._poll_worker.poll_done.connect(self._on_poll_done)
            self._poll_worker.start()

    def _on_poll_done(self, data: dict) -> None:  # noqa: C901
        """Receive instrument readings from background thread and update labels."""
        # Stage position
        if "stage_pos" in data:
            x, y = data["stage_pos"]
            self._lbl_stage.setText(f"X: {x:.2f}\nY: {y:.2f}")
        else:
            self._lbl_stage.setText("X: --\nY: --")

        # Dispenser head
        if "syringe_is_up" in data:
            is_up = data["syringe_is_up"]
            if is_up is True:
                self._lbl_head.setText("Retracted ↑")
                self._lbl_head.setStyleSheet("color: #2a8a5e; font-size: 8pt;")
            elif is_up is False:
                self._lbl_head.setText("Descended ↓")
                self._lbl_head.setStyleSheet("color: #c07830; font-size: 8pt;")
            else:
                self._lbl_head.setText("Unknown")
                self._lbl_head.setStyleSheet("color: #888; font-size: 8pt;")
        else:
            self._lbl_head.setText("--")
            self._lbl_head.setStyleSheet("color: #888; font-size: 8pt;")

        # Temperature — Stage (temp_controller) SP/PV
        if "temp_sp" in data and "temp_pv" in data:
            self._lbl_t_sp.setText(f"Stage SP: {data['temp_sp']:.1f} °C")
            self._lbl_t_pv.setText(f"Stage PV: {data['temp_pv']:.1f} °C")
        else:
            self._lbl_t_sp.setText("Stage SP: --")
            self._lbl_t_pv.setText("Stage PV: --")

        # Chamber temperature — RH sensor onboard T
        ct = data.get("chamber_temp")
        if ct is not None and not math.isnan(ct):
            self._lbl_chamber.setText(f"Chamber: {ct:.1f} °C")
        else:
            self._lbl_chamber.setText("Chamber: --")

        # Humidity
        if "rh_sp" in data and "rh_pv" in data:
            sp_rh = data["rh_sp"]
            pv_rh = data["rh_pv"]
            sp_str = f"{sp_rh:.1f} %" if not math.isnan(sp_rh) else "--"
            pv_str = f"{pv_rh:.1f} %" if not math.isnan(pv_rh) else "--"
            self._lbl_rh_sp.setText(f"SP: {sp_str}")
            self._lbl_rh_pv.setText(f"PV: {pv_str}")
        else:
            self._lbl_rh_sp.setText("SP: --")
            self._lbl_rh_pv.setText("PV: --")

    # ── Workflow status update slots ──────────────────────────────────────────

    def update_arrhenius_status(self, text: str, current: int, total: int) -> None:
        """Slot for ArrheniusTab.sweep_status_changed(str, int, int)."""
        if total > 0:
            self._lbl_wf_arrhenius.setText(f"Arrhenius:\n{text}\n({current}/{total})")
            self._lbl_wf_arrhenius.setStyleSheet("color: #2277bb; font-size: 8pt;")
        else:
            idle = text.lower() in ("idle", "done", "complete", "ready", "")
            self._lbl_wf_arrhenius.setText(f"Arrhenius: {text}")
            self._lbl_wf_arrhenius.setStyleSheet(
                f"color: {'#666' if idle else '#2277bb'}; font-size: 8pt;"
            )

    def update_ht_status(self, text: str) -> None:
        """Slot for ExperimentBuilderTab.workflow_status_changed(str)."""
        idle = text.lower() in ("idle", "done", "complete", "ready", "")
        self._lbl_wf_ht.setText(f"HT Exp: {text}")
        self._lbl_wf_ht.setStyleSheet(
            f"color: {'#666' if idle else '#2277bb'}; font-size: 8pt;"
        )

    def update_auto_status(self, text: str) -> None:
        """Slot for AutonomousTab.workflow_status_changed(str)."""
        idle = text.lower() in ("idle", "done", "complete", "ready", "")
        self._lbl_wf_auto.setText(f"Autonomous: {text}")
        self._lbl_wf_auto.setStyleSheet(
            f"color: {'#666' if idle else '#2277bb'}; font-size: 8pt;"
        )
