"""Tests for PositionMapWidget (Tab 1 interactive electrode map)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.position_map import (
    AVAILABLE_COLOR,
    OCCUPIED_COLOR,
    PositionMapWidget,
    _PositionWorker,
    calculate_electrode_positions,
)
from matplotlib.colors import to_rgba


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager():
    return create_mock_manager(config={})


@pytest.fixture
def pcb_config_4x4():
    return {
        "channels": 16,
        "grid": [4, 4],
        "spacing_mm": [-45, -45],
        "electrode_L_cm": 0.2,
        "electrode_w_cm": 0.2,
    }


@pytest.fixture
def position_map_widget(qapp, manager, pcb_config_4x4):
    widget = PositionMapWidget(manager, pcb_config_4x4, home_pos=(0.0, 0.0))
    yield widget
    worker = getattr(widget, "_pos_worker", None)
    if worker is not None and worker.isRunning():
        worker.requestInterruption()
        worker.wait(2000)
    widget.close()


class TestPositionCalculation:
    def test_electrode_count_4x4(self, pcb_config_4x4):
        xs, ys = calculate_electrode_positions(pcb_config_4x4, origin_x=0, origin_y=0)
        assert len(xs) == 16
        assert len(ys) == 16

    def test_grid_bounds_match_spacing(self, pcb_config_4x4):
        xs, ys = calculate_electrode_positions(pcb_config_4x4, origin_x=0, origin_y=0)
        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()
        # spacing_mm is the inter-electrode pitch (45 mm); for a 4×4 grid
        # the span = (n_cols - 1) * pitch = 3 * 45 = 135 mm.
        assert abs(x_span - 135) < 1
        assert abs(y_span - 135) < 1

    def test_electrode_count_8x4(self):
        config_8x4 = {
            "channels": 32,
            "grid": [8, 4],
            "spacing_mm": [-10, -10],
        }
        xs, ys = calculate_electrode_positions(config_8x4, origin_x=0, origin_y=0)
        assert len(xs) == 32
        assert len(ys) == 32

    def test_origin_offset(self, pcb_config_4x4):
        xs, ys = calculate_electrode_positions(pcb_config_4x4, origin_x=10, origin_y=20)
        # With upper-left origin semantics, electrode 1 is placed at (origin_x, origin_y)
        # and subsequent electrodes have smaller X and Y values.
        assert abs(xs.max() - 10) < 0.1
        assert abs(ys.max() - 20) < 0.1


class TestWidgetConstruction:
    def test_widget_creates_without_error(self, position_map_widget):
        assert position_map_widget is not None

    def test_canvas_present(self, position_map_widget):
        assert position_map_widget._canvas is not None

    def test_status_label_initial_text(self, position_map_widget):
        assert "Ready" in position_map_widget._lbl_status.text()

    def test_scatter_has_correct_point_count(self, position_map_widget):
        offsets = position_map_widget._scatter.get_offsets()
        assert len(offsets) == 16


class TestInteraction:
    def test_clear_selection_resets_state(self, position_map_widget):
        position_map_widget._selected_electrode_idx = 3
        position_map_widget._clear_selection()
        assert position_map_widget._selected_electrode_idx is None
        assert position_map_widget._lbl_status.text() == "Selection cleared"

    def test_set_pcb_config_updates_widget(self, position_map_widget):
        new_config = {
            "channels": 32,
            "grid": [8, 4],
            "spacing_mm": [-10, -10],
        }
        position_map_widget.set_pcb_config(new_config, home_pos=(5.0, 5.0))
        assert position_map_widget._pcb_config == new_config
        assert position_map_widget._home_x == 5.0
        assert position_map_widget._home_y == 5.0
        offsets = position_map_widget._scatter.get_offsets()
        assert len(offsets) == 32


class TestOccupancyColoring:
    """Occupied wells render maroon; available wells stay light blue."""

    def test_all_available_by_default(self, position_map_widget):
        colors = position_map_widget._scatter.get_facecolors()
        avail = to_rgba(AVAILABLE_COLOR)
        for c in colors:
            assert tuple(round(v, 3) for v in c) == tuple(round(v, 3) for v in avail)

    def test_set_occupied_colors_those_wells_maroon(self, position_map_widget):
        # Electrodes 2 and 5 are 1-based; indices 1 and 4 in the scatter.
        position_map_widget.set_occupied({2, 5})
        colors = position_map_widget._scatter.get_facecolors()
        occ = tuple(round(v, 3) for v in to_rgba(OCCUPIED_COLOR))
        avail = tuple(round(v, 3) for v in to_rgba(AVAILABLE_COLOR))
        for idx, c in enumerate(colors):
            got = tuple(round(v, 3) for v in c)
            assert got == (occ if (idx + 1) in {2, 5} else avail)

    def test_occupancy_read_from_data_store_on_construction(self, qapp, manager,
                                                            pcb_config_4x4, tmp_path):
        from softae.core.data_store import DataStore
        store = DataStore(tmp_path / "proj")
        store.record_electrode_cast(0, 3)
        store.record_electrode_cast(0, 8)
        widget = PositionMapWidget(
            manager, pcb_config_4x4, home_pos=(0.0, 0.0), data_store=store,
        )
        try:
            assert widget._occupied == {3, 8}
            colors = widget._scatter.get_facecolors()
            occ = tuple(round(v, 3) for v in to_rgba(OCCUPIED_COLOR))
            assert tuple(round(v, 3) for v in colors[2]) == occ   # E3
            assert tuple(round(v, 3) for v in colors[7]) == occ   # E8
        finally:
            worker = getattr(widget, "_pos_worker", None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)
            widget.close()
            store.close()

    def test_refresh_occupancy_picks_up_new_casts(self, qapp, manager,
                                                  pcb_config_4x4, tmp_path):
        from softae.core.data_store import DataStore
        store = DataStore(tmp_path / "proj")
        widget = PositionMapWidget(
            manager, pcb_config_4x4, home_pos=(0.0, 0.0), data_store=store,
        )
        try:
            assert widget._occupied == set()
            store.record_electrode_cast(0, 4)
            widget.refresh_occupancy()
            assert widget._occupied == {4}
        finally:
            worker = getattr(widget, "_pos_worker", None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)
            widget.close()
            store.close()

    def test_legend_present(self, position_map_widget):
        legend = position_map_widget._ax.get_legend()
        assert legend is not None
        labels = {t.get_text() for t in legend.get_texts()}
        assert {"Available", "Occupied"} <= labels


class TestPositionPolling:
    def test_position_worker_started_on_construction(self, position_map_widget):
        """PositionMapWidget must start a _PositionWorker thread, not a QTimer."""
        assert hasattr(position_map_widget, "_pos_worker"), \
            "Widget should have _pos_worker attribute"
        assert position_map_widget._pos_worker.isRunning(), \
            "_pos_worker should be running after widget construction"

    def test_position_worker_no_qtimer_used(self, position_map_widget):
        """Old _poll_timer approach must not be used."""
        assert not hasattr(position_map_widget, "_poll_timer"), \
            "_poll_timer should be removed — polling moved to _PositionWorker"


class TestPositionWorkerUnit:
    """Unit tests for _PositionWorker independent of the full widget."""

    def test_worker_emits_position_updated_signal(self, qapp):
        """_PositionWorker must emit position_updated(x, y, z) within one poll cycle."""
        received = []
        mock_stage = MagicMock()
        mock_stage.live_position.return_value = (10.5, 20.25)
        mock_mgr = MagicMock()
        mock_mgr.get.return_value = mock_stage

        worker = _PositionWorker(mock_mgr)
        worker.position_updated.connect(lambda x, y, z: received.append((x, y, z)))
        worker.start()

        timeout = 3.0
        t0 = time.monotonic()
        while not received and (time.monotonic() - t0) < timeout:
            time.sleep(0.1)
            QApplication.processEvents()

        worker.requestInterruption()
        worker.wait(2000)

        assert received, "_PositionWorker did not emit position_updated within 3 s"
        x, y, z = received[0]
        assert abs(x - 10.5) < 0.01
        assert abs(y - 20.25) < 0.01
        assert z == 0.0  # 2-D stage, z defaults to 0.0

    def test_worker_update_slot_sets_marker(self, position_map_widget):
        """_on_position_updated must update the position marker data."""
        position_map_widget._on_position_updated(5.0, 7.5, 0.0)
        xdata, ydata = position_map_widget._position_marker.get_data()
        assert list(xdata) == [5.0]
        assert list(ydata) == [7.5]

    def test_worker_stops_cleanly_on_interruption(self, qapp):
        """_PositionWorker must exit cleanly after requestInterruption()."""
        mock_stage = MagicMock()
        mock_stage.live_position.return_value = (0.0, 0.0)
        mock_mgr = MagicMock()
        mock_mgr.get.return_value = mock_stage

        worker = _PositionWorker(mock_mgr)
        worker.start()
        time.sleep(0.1)
        worker.requestInterruption()
        stopped = worker.wait(3000)
        assert stopped, "_PositionWorker did not stop within 3 s after requestInterruption()"
        assert not worker.isRunning()

    def test_worker_handles_stage_exception_safely(self, qapp):
        """_PositionWorker must not crash when stage.live_position() raises."""
        mock_mgr = MagicMock()
        mock_mgr.get.side_effect = RuntimeError("stage offline")

        worker = _PositionWorker(mock_mgr)
        worker.start()
        time.sleep(0.1)
        worker.requestInterruption()
        assert worker.wait(3000), "Worker should stop cleanly even after exceptions"


class TestPositionMapCleanup:
    def test_position_map_cleanup_stops_pos_worker(self, position_map_widget):
        worker = position_map_widget._pos_worker
        assert worker.isRunning()
        position_map_widget.cleanup()
        assert not worker.isRunning()

    def test_position_worker_stop_worker_joins(self, qapp, manager):
        worker = _PositionWorker(manager)
        worker.start()
        try:
            assert worker.isRunning()
            worker.stop_worker()
            assert not worker.isRunning()
        finally:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(2000)
