"""Tests for the MonitorSidebar webcam retry/restart button (no real camera)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.monitor_sidebar import MonitorSidebar


class _FakePoller(QObject):
    """Stand-in shared poller so the sidebar skips its local poll thread."""

    sidebar_ready = Signal(dict)


class _FakeWebcamWorker(QObject):
    """Records lifecycle calls; never opens a real camera."""

    frame_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, *, running_after_stop: bool = False):
        super().__init__()
        self.calls: list[str] = []
        self._running = False
        self._running_after_stop = running_after_stop

    def request_frame(self, exposure=None) -> None:  # called by the frame timer
        pass

    def set_exposure(self, exposure: float) -> None:
        pass

    def stop_worker(self) -> None:
        self.calls.append("stop")
        self._running = self._running_after_stop

    def start(self) -> None:
        self.calls.append("start")
        self._running = True

    def isRunning(self) -> bool:  # noqa: N802 — mirrors QThread API
        return self._running


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def sidebar(qapp):
    sb = MonitorSidebar(create_mock_manager(config={}), poller=_FakePoller())
    yield sb
    timer = getattr(sb, "_wc_frame_timer", None)
    if timer is not None:
        timer.stop()
    sb.close()


def test_retry_button_exists_and_labelled(sidebar):
    assert sidebar._btn_wc_retry.toolTip() == "Restart the webcam feed"


def test_retry_stops_then_restarts_worker(sidebar):
    fake = _FakeWebcamWorker(running_after_stop=False)  # joins cleanly
    sidebar.set_webcam_worker(fake)
    sidebar._on_wc_retry()
    assert fake.calls == ["stop", "start"]


def test_retry_button_click_triggers_restart(sidebar):
    fake = _FakeWebcamWorker()
    sidebar.set_webcam_worker(fake)
    sidebar._btn_wc_retry.click()
    assert "start" in fake.calls


def test_retry_does_not_double_start_when_still_running(sidebar):
    fake = _FakeWebcamWorker(running_after_stop=True)  # wedged capture
    sidebar.set_webcam_worker(fake)
    sidebar._on_wc_retry()
    assert fake.calls == ["stop"]  # no start() on a still-running thread
    assert "busy" in sidebar._lbl_webcam.text().lower()


def test_retry_with_no_worker_is_safe(sidebar):
    sidebar._on_wc_retry()  # never set a worker → must not raise


def test_worker_error_is_shown_in_feed_label(sidebar):
    fake = _FakeWebcamWorker()
    sidebar.set_webcam_worker(fake)
    fake.error_occurred.emit("Failed to open webcam at index 0")
    assert "Failed to open webcam" in sidebar._lbl_webcam.text()
