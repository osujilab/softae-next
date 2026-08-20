"""The sidebar's rig-owner line — three states, one vocabulary.

The line has to distinguish *three* situations and the third is the one a
two-state line gets wrong: something holds the rig that is **not** a campaign, so
there is no stream to follow and no Pause or Abort that would reach it. Rendering
that as a campaign offers the operator stops that command nothing.

The composition is asserted on
:func:`~softae.gui.widgets.rig_owner.owner_status_line` rather than on the label,
because that is the whole reason the function lives in ``rig_owner``: the Init
tab, the Manual Control banner and this line must not drift into three
vocabularies for one fact.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.monitor_sidebar import MonitorSidebar
from softae.gui.widgets.rig_owner import OCCUPIED, owner_line, owner_status_line


class _FakePoller(QObject):
    """Stand-in shared poller so the sidebar skips its local poll thread."""

    sidebar_ready = Signal(dict)


class _Lock:
    """The fields :mod:`softae.gui.widgets.rig_owner` reads off a ``RunLock``."""

    def __init__(self, what: str, *, pid: int = 4242, log_path: str = "/runs/r1"):
        self.what = what
        self.pid = pid
        self.started_at = "2026-08-19T14:02:00"
        self.log_path = log_path


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


# ── The three states of the line ─────────────────────────────────────────────

def test_owner_status_line_no_foreign_lock_is_empty():
    assert owner_status_line(None) == ""


def test_owner_status_line_campaign_lock_names_phase_and_age():
    line = owner_status_line(
        _Lock("campaign:shadow-run:run-42"), phase="anneal", phase_age_s=12.4
    )
    assert line == "shadow-run · anneal · 12s — See Monitoring tab"


def test_owner_status_line_campaign_before_first_beat_says_so_not_zero():
    """"0s" and "we have not heard anything yet" are opposite facts."""
    line = owner_status_line(_Lock("campaign:shadow-run:run-42"))
    assert "waiting for the first heartbeat" in line
    assert "0s" not in line


def test_owner_status_line_non_campaign_lock_is_occupied_with_no_run_offered():
    lock = _Lock("workflow 'ht_sequence'", log_path="")
    line = owner_status_line(lock)
    assert line == f"{OCCUPIED} — {owner_line(lock)}"
    # Nothing to attach to, so nothing that points at a run.
    assert "Monitoring tab" not in line


def test_owner_status_line_non_campaign_lock_ignores_a_stale_phase():
    """A phase can only belong to a campaign; a bench lock must not borrow one."""
    line = owner_status_line(_Lock("bench sequence"), phase="anneal", phase_age_s=9)
    assert line.startswith(OCCUPIED)
    assert "anneal" not in line


# ── The slot that renders it ─────────────────────────────────────────────────

def test_update_rig_owner_shows_the_line_when_there_is_one(sidebar):
    sidebar.update_rig_owner("shadow-run · anneal · 12s — See Monitoring tab")
    # isHidden(), not isVisible(): the sidebar itself is never shown in a test,
    # and every child of an unshown parent reports isVisible() False.
    assert not sidebar._lbl_rig_owner.isHidden()
    assert sidebar._lbl_rig_owner.text().startswith("shadow-run")


def test_update_rig_owner_hides_the_label_when_the_rig_is_free(sidebar):
    sidebar.update_rig_owner("shadow-run · anneal · 12s — See Monitoring tab")
    sidebar.update_rig_owner("")
    assert sidebar._lbl_rig_owner.text() == ""
    assert sidebar._lbl_rig_owner.isHidden()


def test_update_rig_owner_starts_hidden(sidebar):
    """A free rig gets no permanent "nobody else" label to learn to ignore."""
    assert sidebar._lbl_rig_owner.isHidden()
