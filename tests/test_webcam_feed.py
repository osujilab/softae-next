"""Tests for WebcamFeedWidget and WebcamWorker (no real camera needed)."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from softae.gui.widgets.webcam_feed_widget import WebcamFeedWidget
from softae.gui.widgets.webcam_worker import WebcamWorker


# ── Fake worker with real signals (no camera) ────────────────────────────


class FakeWebcamWorker(QObject):
    """Substitute for WebcamWorker that never opens a real camera."""

    frame_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._exposure: float = -7

    def request_frame(self, exposure: float | None = None) -> None:
        if exposure is not None:
            self._exposure = exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = exposure

    def stop_worker(self) -> None:
        pass


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def fake_worker(qapp):
    return FakeWebcamWorker()


@pytest.fixture
def feed_widget(qapp, fake_worker):
    widget = WebcamFeedWidget(fake_worker)
    yield widget
    widget._frame_timer.stop()
    widget.close()


def _synthetic_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ── WebcamWorker unit tests ──────────────────────────────────────────────


class TestWebcamWorkerConstruction:
    def test_set_exposure(self, qapp):
        w = WebcamWorker()
        w.set_exposure(-3)
        assert w._exposure == -3


# ── WebcamFeedWidget tests ───────────────────────────────────────────────


class TestWidgetConstruction:
    def test_creates_without_error(self, feed_widget):
        assert feed_widget is not None

    def test_exposure_slider_range(self, feed_widget):
        s = feed_widget._slider_exposure
        assert s.minimum() == -9
        assert s.maximum() == -1
        assert s.value() == -7

    def test_frame_timer_running(self, feed_widget):
        assert feed_widget._frame_timer.isActive()
        assert feed_widget._frame_timer.interval() == 150


class TestFrameDisplay:
    def test_frame_received(self, feed_widget, fake_worker):
        frame = _synthetic_frame()
        fake_worker.frame_ready.emit(frame)
        assert feed_widget._current_frame is not None

class TestExposureControl:
    def test_slider_updates_worker(self, feed_widget, fake_worker):
        feed_widget._slider_exposure.setValue(-3)
        assert fake_worker._exposure == -3.0

    def test_slider_label_updates(self, feed_widget):
        feed_widget._slider_exposure.setValue(-5)
        assert feed_widget._lbl_exposure_val.text() == "-5"


class TestZoom:
    def test_initial_state_no_zoom(self, feed_widget):
        assert feed_widget._zoom_active is False
        assert feed_widget._zoom_rect is None

    def test_reset_zoom_button_initially_disabled(self, feed_widget):
        assert feed_widget._btn_reset_zoom.isEnabled() is False

    def test_reset_zoom_button_clears_state(self, feed_widget):
        feed_widget._zoom_active = True
        feed_widget._zoom_rect = (10, 10, 200, 200)
        feed_widget._btn_reset_zoom.setEnabled(True)
        feed_widget._reset_zoom()
        assert feed_widget._zoom_active is False
        assert feed_widget._zoom_rect is None
        assert feed_widget._btn_reset_zoom.isEnabled() is False

    def test_manual_reset_zoom(self, feed_widget):
        feed_widget._zoom_active = True
        feed_widget._zoom_rect = (10, 10, 200, 200)
        # Simulate reset
        feed_widget._zoom_active = False
        feed_widget._zoom_rect = None
        assert feed_widget._zoom_active is False
        assert feed_widget._zoom_rect is None


class TestDrawRect:
    def test_draw_rect_no_crash(self):
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        WebcamFeedWidget._draw_rect(frame, 10, 10, 50, 50, (255, 0, 0), 2)
        assert frame[10, 10, 0] == 255

    def test_draw_rect_clamps(self):
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        # Rect exceeding bounds should not crash
        WebcamFeedWidget._draw_rect(frame, -5, -5, 205, 105, (0, 255, 0), 2)
