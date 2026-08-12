"""Webcam feed display with exposure control, timestamp overlay, and zoom."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from softae.gui.widgets.webcam_worker import WebcamWorker


class WebcamFeedWidget(QWidget):
    """Webcam display with exposure slider, timestamp overlay, and click-drag zoom."""

    def __init__(self, worker: WebcamWorker, parent: QWidget | None = None):
        super().__init__(parent)
        self._worker = worker
        self._current_frame: np.ndarray | None = None
        self._zoom_active: bool = False
        self._zoom_rect: tuple[int, int, int, int] | None = None
        self._zoom_start: tuple[int, int] | None = None
        self._t0: float | None = None

        self._build_ui()

        # Connect worker signals
        worker.frame_ready.connect(self._on_frame_ready)
        worker.error_occurred.connect(self._on_error)

        # Frame request timer (~7 FPS)
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._request_next_frame)
        self._frame_timer.start(150)

    # --- UI construction ------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Exposure slider row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Exposure:"))
        self._slider_exposure = QSlider(Qt.Orientation.Horizontal)
        self._slider_exposure.setRange(-9, -1)
        self._slider_exposure.setValue(-7)
        self._slider_exposure.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider_exposure.setTickInterval(1)
        self._slider_exposure.valueChanged.connect(self._on_exposure_changed)
        ctrl_row.addWidget(self._slider_exposure)
        self._lbl_exposure_val = QLabel("-7")
        ctrl_row.addWidget(self._lbl_exposure_val)
        layout.addLayout(ctrl_row)

        # Frame display
        self._lbl_frame = QLabel("Waiting for webcam...")
        self._lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_frame.setMinimumSize(400, 300)
        # Ignored policy: label size is driven by layout, not by the
        # pixmap's sizeHint — prevents the "pan and zoom" growth effect.
        self._lbl_frame.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self._lbl_frame.setStyleSheet(
            "border: 2px solid #0066cc; background: #1a1a1a; color: #666;"
        )
        self._lbl_frame.setMouseTracking(True)
        layout.addWidget(self._lbl_frame)

        # Status row: timestamp label + reset zoom button
        status_row = QHBoxLayout()
        self._lbl_status = QLabel("Click-drag to zoom")
        self._lbl_status.setStyleSheet("color: #999; font-size: 10px;")
        status_row.addWidget(self._lbl_status)
        status_row.addStretch()
        self._btn_reset_zoom = QPushButton("Reset Zoom")
        self._btn_reset_zoom.setEnabled(False)
        self._btn_reset_zoom.clicked.connect(self._reset_zoom)
        status_row.addWidget(self._btn_reset_zoom)
        layout.addLayout(status_row)

    # --- Frame handling -------------------------------------------------------

    def _request_next_frame(self) -> None:
        self._worker.request_frame()

    def _on_frame_ready(self, arr: np.ndarray) -> None:
        if self._t0 is None:
            self._t0 = time.time()
        self._current_frame = arr
        self._update_display()

    def _on_error(self, msg: str) -> None:
        self._lbl_frame.setText(f"Webcam error: {msg}")

    # --- Exposure slider ------------------------------------------------------

    def _on_exposure_changed(self, value: int) -> None:
        self._lbl_exposure_val.setText(str(value))
        self._worker.set_exposure(float(value))

    # --- Display rendering ----------------------------------------------------

    def _update_display(self, draw_zoom_rect: bool = False) -> None:
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()

        # Apply zoom crop if active
        if self._zoom_active and self._zoom_rect:
            frame = self._apply_zoom(frame)

        # Timestamp overlay (white bar + status label text)
        frame = self._draw_timestamp(frame)

        # Draw zoom rectangle if dragging (green outline)
        if draw_zoom_rect and self._zoom_rect and not self._zoom_active:
            lw, lh = self._lbl_frame.width(), self._lbl_frame.height()
            fh, fw = frame.shape[:2]
            if lw > 0 and lh > 0:
                sx, sy = fw / lw, fh / lh
                fx1 = int(self._zoom_rect[0] * sx)
                fy1 = int(self._zoom_rect[1] * sy)
                fx2 = int(self._zoom_rect[2] * sx)
                fy2 = int(self._zoom_rect[3] * sy)
                self._draw_rect(frame, fx1, fy1, fx2, fy2, (0, 255, 0), 2)

        # Convert to QPixmap
        h, w = frame.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qimg = qimg.copy()  # detach from numpy buffer
        pixmap = QPixmap.fromImage(qimg)

        label_size = self._lbl_frame.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled = pixmap.scaled(
                label_size,
                aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
                mode=Qt.TransformationMode.SmoothTransformation,
            )
            self._lbl_frame.setPixmap(scaled)
        else:
            self._lbl_frame.setPixmap(pixmap)

    def _apply_zoom(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the zoom rectangle (mapped from label coords)."""
        if self._zoom_rect is None:
            return frame
        fh, fw = frame.shape[:2]
        lw, lh = self._lbl_frame.width(), self._lbl_frame.height()
        if lw <= 0 or lh <= 0:
            return frame
        sx, sy = fw / lw, fh / lh
        x1 = max(0, int(self._zoom_rect[0] * sx))
        y1 = max(0, int(self._zoom_rect[1] * sy))
        x2 = min(fw, int(self._zoom_rect[2] * sx))
        y2 = min(fh, int(self._zoom_rect[3] * sy))
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def _draw_rect(
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw a rectangle on a numpy RGB frame (no cv2 needed)."""
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        # Top and bottom edges
        frame[y1 : y1 + thickness, x1:x2] = color
        frame[max(y1, y2 - thickness) : y2, x1:x2] = color
        # Left and right edges
        frame[y1:y2, x1 : x1 + thickness] = color
        frame[y1:y2, max(x1, x2 - thickness) : x2] = color

    def _draw_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """Draw timestamp info: white bar on frame + text in status label."""
        elapsed = time.time() - self._t0 if self._t0 else 0.0
        ts = f"{elapsed:.2f}s  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        zoom_hint = "Zoomed" if self._zoom_active else "Click-drag to zoom"
        self._lbl_status.setText(f"{ts}  |  {zoom_hint}")

        # White bar at top of frame for visual timestamp indicator
        bar_h = min(24, frame.shape[0])
        frame[:bar_h, :] = 255
        return frame

    # --- Mouse events for zoom ------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._lbl_frame.mapFrom(self, event.position().toPoint())
            self._zoom_start = (pos.x(), pos.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._zoom_start:
            pos = self._lbl_frame.mapFrom(self, event.position().toPoint())
            x1, y1 = self._zoom_start
            x2, y2 = pos.x(), pos.y()
            self._zoom_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            self._update_display(draw_zoom_rect=True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._zoom_rect:
                dx = abs(self._zoom_rect[2] - self._zoom_rect[0])
                dy = abs(self._zoom_rect[3] - self._zoom_rect[1])
                if dx > 20 and dy > 20:
                    self._zoom_active = True
                    self._btn_reset_zoom.setEnabled(True)
                else:
                    self._zoom_rect = None
            self._zoom_start = None
            self._update_display()

    def _reset_zoom(self) -> None:
        """Reset zoom to full-frame view."""
        self._zoom_active = False
        self._zoom_rect = None
        self._btn_reset_zoom.setEnabled(False)
        self._update_display()
