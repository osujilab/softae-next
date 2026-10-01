"""WebcamWorker failure recovery against a fake cv2 — never opens a real camera.

Synchronous tests call ``run()`` on the test thread: each fake ``read()`` queues
the next request until its script is spent, then requests stop, so the loop is
fully deterministic with no real sleeps (the backoff base is patched to 0).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from softae.gui.widgets import webcam_worker as ww
from softae.gui.widgets.webcam_worker import WebcamWorker

GOOD = "good"
BAD_READ = "ret_false"


class FakeCvError(Exception):
    """Stands in for cv2.error (a subclass of Exception in real OpenCV)."""


class FakeCv2:
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_EXPOSURE = 15
    COLOR_BGR2RGB = 4
    error = FakeCvError

    def __init__(self, script, opens=None, on_read=None):
        self.script = list(script)  # per read: GOOD, BAD_READ, or an exception
        self.opens = list(opens or [])  # per VideoCapture(): isOpened result (default True)
        self.on_read = on_read
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, index, backend):  # noqa: N802 — mirrors cv2 API
        opened = self.opens.pop(0) if self.opens else True
        cap = FakeCapture(self, opened)
        self.captures.append(cap)
        return cap

    @staticmethod
    def cvtColor(frame, code):  # noqa: N802
        return frame[..., ::-1]

    def exposure_sets(self):
        return [(i, v) for i, c in enumerate(self.captures)
                for prop, v in c.sets if prop == self.CAP_PROP_EXPOSURE]


class FakeCapture:
    def __init__(self, cv, opened):
        self.cv, self.opened = cv, opened
        self.sets, self.released = [], False

    def isOpened(self):  # noqa: N802
        return self.opened

    def set(self, prop, value):
        self.sets.append((prop, value))
        return True

    def release(self):
        self.released = True

    def read(self):
        item = self.cv.script.pop(0) if self.cv.script else GOOD
        if self.cv.on_read:
            self.cv.on_read()
        if isinstance(item, Exception):
            raise item
        if item == BAD_READ:
            return False, None
        return True, np.zeros((4, 6, 3), dtype=np.uint8)


def drive(worker, fake, n_reads, monkeypatch, before_read=None):
    """Run the worker synchronously for exactly *n_reads* frame requests."""
    remaining = [n_reads]

    def on_read():
        if before_read:
            before_read(n_reads - remaining[0])
        remaining[0] -= 1
        if remaining[0] > 0:
            worker.request_frame()
        else:
            worker._request_stop()

    fake.on_read = on_read
    monkeypatch.setattr(ww, "cv2", fake, raising=False)
    monkeypatch.setattr(ww, "_HAS_CV2", True)
    monkeypatch.setattr(ww, "BACKOFF_BASE_S", 0.0)
    frames, errors = [], []
    worker.frame_ready.connect(frames.append)
    worker.error_occurred.connect(errors.append)
    worker.request_frame()
    worker.run()
    return frames, errors


@pytest.fixture
def worker(qapp):
    return WebcamWorker(default_exposure=-7)


def test_worker_unchanged_exposure_set_once_per_open(worker, monkeypatch):
    fake = FakeCv2([GOOD] * 5)
    frames, errors = drive(worker, fake, 5, monkeypatch)
    assert len(frames) == 5 and errors == []
    assert fake.exposure_sets() == [(0, -7)]


def test_worker_set_exposure_change_reapplied_once(worker, monkeypatch):
    fake = FakeCv2([GOOD] * 5)
    drive(worker, fake, 5, monkeypatch,
          before_read=lambda i: worker.set_exposure(-4) if i == 1 else None)
    # read 1 already fetched -7 when the change lands, so read 2 applies -4
    assert fake.exposure_sets() == [(0, -7), (0, -4)]


def test_worker_three_raising_reads_reopens_and_resumes(worker, monkeypatch):
    boom = FakeCvError("Unknown C++ exception from OpenCV code")
    fake = FakeCv2([boom, boom, boom, GOOD, GOOD])
    frames, errors = drive(worker, fake, 5, monkeypatch)
    assert len(fake.captures) == 2
    assert fake.captures[0].released
    assert len(frames) == 2
    # exposure re-applied on the fresh device even though the value is unchanged
    assert fake.exposure_sets() == [(0, -7), (1, -7)]
    assert worker._reopen_attempts == 0


def test_worker_error_emitted_once_per_episode(worker, monkeypatch):
    boom = FakeCvError("Unknown C++ exception from OpenCV code")
    fake = FakeCv2([boom, BAD_READ, boom, GOOD, BAD_READ, BAD_READ, GOOD])
    frames, errors = drive(worker, fake, 7, monkeypatch)
    assert errors == ["Frame error: Unknown C++ exception from OpenCV code",
                      "Frame error: Failed to read frame"]
    assert len(frames) == 2


def test_worker_failed_reopen_reports_attempt_number(worker, monkeypatch):
    boom = FakeCvError("dead")
    # initial open ok; reopen attempt 1 fails to open; attempt 2 opens
    fake = FakeCv2([boom, boom, boom, GOOD], opens=[True, False, True])
    frames, errors = drive(worker, fake, 4, monkeypatch)
    assert errors == ["Frame error: dead",
                      "Frame error (reopening, attempt 1): could not reopen webcam at index 0"]
    assert len(fake.captures) == 3 and len(frames) == 1


def test_worker_reopen_that_keeps_failing_reports_next_attempt(worker, monkeypatch):
    boom = FakeCvError("dead")
    fake = FakeCv2([boom] * 6 + [GOOD])
    frames, errors = drive(worker, fake, 7, monkeypatch)
    assert errors == ["Frame error: dead", "Frame error (reopening, attempt 2): dead"]
    assert len(fake.captures) == 3 and len(frames) == 1


def test_worker_single_transient_failure_does_not_reopen(worker, monkeypatch):
    fake = FakeCv2([GOOD, FakeCvError("blip"), GOOD, GOOD])
    frames, errors = drive(worker, fake, 4, monkeypatch)
    assert len(fake.captures) == 1  # never reopened (released only at shutdown)
    assert errors == ["Frame error: blip"]
    assert len(frames) == 3


def test_worker_stop_during_backoff_returns_promptly(worker, monkeypatch):
    fake = FakeCv2([FakeCvError("dead")] * 100)
    monkeypatch.setattr(ww, "cv2", fake, raising=False)
    monkeypatch.setattr(ww, "_HAS_CV2", True)
    monkeypatch.setattr(ww, "BACKOFF_BASE_S", 30.0)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not fake.captures[0:1] or not fake.captures[0].released:
            assert time.monotonic() < deadline, "worker never reached the reopen backoff"
            worker.request_frame()
            time.sleep(0.01)
        time.sleep(0.05)  # let the thread enter the 30 s wait
        started = time.monotonic()
        worker.stop_worker(timeout_ms=3000)
        assert time.monotonic() - started < 1.0
        assert not worker.isRunning()
        assert len(fake.captures) == 1  # stopped before reopening
    finally:
        worker.stop_worker(timeout_ms=3000)
