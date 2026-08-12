"""Interactive electrode position map widget with live stage tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager


# Marker fill colors for the electrode map.  "Available" keeps the historical
# light-blue fill; occupied (already-cast, single-use) wells get a light maroon
# fill — deliberately distinct from the bright-red ring used for the current
# dispenser/target position so the two never read as the same state.
AVAILABLE_COLOR = "lightblue"
OCCUPIED_COLOR = "#c46b7a"


class _MoveWorker(QThread):
    """Run a single stage move on a background thread.

    Emits ``completed`` with the driver's return value on success, or
    ``failed(str)`` with the exception message on error.
    """

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _PositionWorker(StoppableWorker):
    """Background thread that queries stage position at ~500 ms intervals.

    Emits ``position_updated(x, y, z)`` on the main thread via
    Qt's queued-connection mechanism.  Stops cleanly when
    ``requestInterruption()`` is called.
    """

    position_updated = Signal(float, float, float)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                stage = self._manager.get("stage")
                pos = stage.live_position()
                x = float(pos[0])
                y = float(pos[1])
                z = float(pos[2]) if len(pos) > 2 else 0.0
                self.position_updated.emit(x, y, z)
            except Exception:
                pass
            self.msleep(500)


# Canonical grid geometry lives in softae.core.geometry (GUI-free, shared with
# the HT tab's per-channel position injection). Re-exported here under the
# historical name so existing widget/test imports keep working.
from softae.core.geometry import electrode_positions as calculate_electrode_positions


class PositionMapWidget(QWidget):
    """Interactive scatter-plot map of electrode positions with live stage marker."""

    def __init__(
        self,
        manager: InstrumentManager,
        pcb_config: dict[str, Any],
        home_pos: tuple[float, float] = (0.0, 0.0),
        dep1_pos: tuple[float, float] = (0.0, 0.0),
        parent: QWidget | None = None,
        data_store: "DataStore | None" = None,
        board_id: int | None = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._pcb_config = pcb_config
        self._home_x, self._home_y = home_pos
        self._dep1_x, self._dep1_y = dep1_pos
        self._electrode_positions: tuple[np.ndarray, np.ndarray] | None = None
        self._selected_electrode_idx: int | None = None
        # Persistent single-use well occupancy (board-relative electrode numbers,
        # 1-based to match the E1…EN labels).  When a DataStore is supplied the
        # occupied set is read from it on each refresh; ``board_id=None`` tracks
        # the current board.  Without a store, occupancy can still be injected
        # directly via ``set_occupied``.
        self._data_store = data_store
        self._board_id = board_id
        self._occupied: set[int] = set()

        self._build_ui()
        self._setup_matplotlib()
        self._start_position_polling()

    # --- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(5, 3), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas)

        status_bar = QHBoxLayout()
        self._lbl_status = QLabel("Ready")
        status_bar.addWidget(self._lbl_status)
        status_bar.addStretch()
        self._btn_clear = QPushButton("Clear Selection")
        self._btn_clear.clicked.connect(self._clear_selection)
        status_bar.addWidget(self._btn_clear)
        layout.addLayout(status_bar)

    def _setup_matplotlib(self) -> None:
        xs, ys = calculate_electrode_positions(
            self._pcb_config, origin_x=self._dep1_x, origin_y=self._dep1_y
        )
        self._electrode_positions = (xs, ys)

        # Refresh occupancy from the store (best-effort) before the first draw so
        # already-cast wells are colored correctly on open, not just after a poll.
        self._reload_occupied()

        self._scatter = self._ax.scatter(
            xs, ys,
            s=100, c=self._face_colors(len(xs)), edgecolors="black", linewidth=1,
            zorder=2, picker=True,
        )

        # Electrode labels (E1, E2, …, EN) next to each marker
        offset_x = max(abs(self._pcb_config.get("spacing_mm", [4, 4])[0]) * 0.15, 1.0)
        offset_y = max(abs(self._pcb_config.get("spacing_mm", [4, 4])[1]) * 0.15, 1.0)
        self._electrode_labels: list = []
        for idx, (ex, ey) in enumerate(zip(xs, ys)):
            txt = self._ax.text(
                ex + offset_x, ey + offset_y,
                f"E{idx + 1}",
                fontsize=6, ha="left", va="bottom",
                color="#444", zorder=5,
            )
            self._electrode_labels.append(txt)
        self._selected_patch = self._ax.plot(
            [], [], "o",
            markersize=12, markerfacecolor="none",
            markeredgecolor="red", markeredgewidth=2, zorder=3,
        )[0]
        self._position_marker = self._ax.plot(
            [], [], "s",
            markersize=10, color="green", alpha=0.7, zorder=4,
        )[0]

        self._add_legend()

        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        rows, cols = self._pcb_config.get("grid", [4, 4])
        self._ax.set_title(f"Electrode Grid ({rows}×{cols})")
        self._ax.grid(True)

        margin = 5
        if len(xs) > 0:
            self._ax.set_xlim(xs.min() - margin, xs.max() + margin)
            self._ax.set_ylim(ys.min() - margin, ys.max() + margin)

        self._fig.tight_layout()
        self._canvas.mpl_connect("pick_event", self._on_pick)

    # --- Occupancy -------------------------------------------------------------

    def _reload_occupied(self) -> None:
        """Refresh the occupied-well set from the DataStore (best-effort)."""
        store = self._data_store
        if store is None:
            return
        try:
            board_id = (
                self._board_id
                if self._board_id is not None
                else store.current_board_id()
            )
            self._occupied = set(store.occupied_electrodes(board_id))
        except Exception:
            # A missing/closed store must never break the map's rendering.
            pass

    def _face_colors(self, n: int) -> list[str]:
        """Per-electrode fill colors: maroon if occupied (1-based), else blue."""
        return [
            OCCUPIED_COLOR if (idx + 1) in self._occupied else AVAILABLE_COLOR
            for idx in range(n)
        ]

    def _add_legend(self) -> None:
        """Draw a small legend distinguishing available/occupied/position markers."""
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=AVAILABLE_COLOR, markeredgecolor="black",
                   label="Available"),
            Line2D([], [], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=OCCUPIED_COLOR, markeredgecolor="black",
                   label="Occupied"),
            Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor="none", markeredgecolor="red", markeredgewidth=2,
                   label="Target"),
            Line2D([], [], marker="s", linestyle="none", markersize=8,
                   markerfacecolor="green", alpha=0.7, label="Stage"),
        ]
        self._ax.legend(
            handles=handles, loc="upper right", fontsize=6,
            framealpha=0.85, handletextpad=0.3, borderpad=0.3,
        )

    def refresh_occupancy(self, board_id: int | None = None) -> None:
        """Re-read occupancy from the store and recolor the markers.

        Pass ``board_id`` to pin a specific board, or leave it ``None`` to track
        the current board.  Safe to call repeatedly (e.g. on tab show).
        """
        if board_id is not None:
            self._board_id = board_id
        self._reload_occupied()
        self._apply_occupancy_colors()

    def set_occupied(self, occupied: set[int]) -> None:
        """Directly set the occupied-well set (1-based) and recolor.

        Useful when occupancy comes from somewhere other than the bound store.
        """
        self._occupied = set(occupied)
        self._apply_occupancy_colors()

    def _apply_occupancy_colors(self) -> None:
        scatter = getattr(self, "_scatter", None)
        if scatter is None:
            return
        n = len(scatter.get_offsets())
        scatter.set_facecolors(self._face_colors(n))
        self._canvas.draw_idle()

    # --- Interaction -----------------------------------------------------------

    def _on_pick(self, event: Any) -> None:
        if event.artist is not self._scatter:
            return
        idx = event.ind[0]
        self._selected_electrode_idx = idx
        xs, ys = self._electrode_positions
        target_x, target_y = float(xs[idx]), float(ys[idx])
        self._selected_patch.set_data([target_x], [target_y])
        self._canvas.draw_idle()

        try:
            stage = self._manager.get("stage")
        except KeyError:
            self._lbl_status.setText(
                f"Electrode {idx + 1} @ ({target_x:.2f}, {target_y:.2f}) mm — stage not connected"
            )
            return

        self._lbl_status.setText(
            f"Electrode {idx + 1} → moving to ({target_x:.2f}, {target_y:.2f}) mm…"
        )

        def _do():
            stage.move_to(target_x, target_y)

        w = _MoveWorker(_do, parent=self)
        w.completed.connect(
            lambda _: self._lbl_status.setText(
                f"Electrode {idx + 1} at ({target_x:.2f}, {target_y:.2f}) mm"
            )
        )
        w.failed.connect(
            lambda e: self._lbl_status.setText(
                f"Electrode {idx + 1}: move failed — {e}"
            )
        )
        w.finished.connect(w.deleteLater)
        w.start()

    def _clear_selection(self) -> None:
        self._selected_electrode_idx = None
        self._selected_patch.set_data([], [])
        self._lbl_status.setText("Selection cleared")
        self._canvas.draw_idle()

    # --- Position polling ------------------------------------------------------

    def _start_position_polling(self) -> None:
        self._pos_worker = _PositionWorker(self._manager, parent=self)
        self._pos_worker.position_updated.connect(self._on_position_updated)
        self._pos_worker.start()

    def _on_position_updated(self, x: float, y: float, z: float) -> None:
        self._position_marker.set_data([x], [y])
        self._canvas.draw_idle()

    def _stop_polling(self) -> None:
        """Request the polling worker to stop and wait for it to exit."""
        worker = getattr(self, "_pos_worker", None)
        if worker is not None:
            worker.stop_worker()

    def cleanup(self) -> None:
        """Stop the position-polling worker (idempotent)."""
        self._stop_polling()

    def closeEvent(self, event) -> None:
        self._stop_polling()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        # hideEvent fires for child widgets when their parent hides/closes.
        # Stop the worker so it is not running when the widget is invisible.
        self._stop_polling()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        # Restart polling when the widget becomes visible again.
        worker = getattr(self, "_pos_worker", None)
        if worker is not None and not worker.isRunning():
            # Reset interruption flag by creating a fresh worker
            self._pos_worker = _PositionWorker(self._manager, parent=self)
            self._pos_worker.position_updated.connect(self._on_position_updated)
            self._pos_worker.start()
        # Pick up any wells cast (e.g. by a campaign) while the tab was hidden.
        self.refresh_occupancy()
        super().showEvent(event)

    # --- Public API ------------------------------------------------------------

    def set_pcb_config(
        self,
        pcb_config: dict[str, Any],
        home_pos: tuple[float, float] | None = None,
        dep1_pos: tuple[float, float] | None = None,
    ) -> None:
        """Replace the PCB layout and/or calibration positions, then redraw."""
        self._pcb_config = pcb_config
        if home_pos is not None:
            self._home_x, self._home_y = home_pos
        if dep1_pos is not None:
            self._dep1_x, self._dep1_y = dep1_pos
        self._selected_electrode_idx = None
        self._ax.clear()
        self._setup_matplotlib()
        self._canvas.draw_idle()
