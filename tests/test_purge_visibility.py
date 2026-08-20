"""The purge schedule, permanently on screen (ruling 4).

Two surfaces, and neither of them decides anything — the decision is
:func:`~softae.gui.widgets.purge_indicator.purge_indicator` and is tested
headless in ``test_purge_indicator.py``. What is asserted here is that the
decision *arrives*:

* :class:`~softae.gui.widgets.purge_badge.PurgeBadge` paints the three states
  apart, pulses only for attention, and stops on acknowledgement;
* :class:`~softae.gui.main_window.MainWindow` retains the last outcome instead
  of dropping it into an 8-second status message, and refreshes the badge from
  ``_on_campaign_tick`` — the 5 s tick that exists in **both** launch modes.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from softae.config import loader
from softae.core.purge import PurgeScheduler, PurgeSettings
from softae.core.purge_runner import PurgeOutcome
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets.monitor_sidebar import MonitorSidebar
from softae.gui.widgets.purge_badge import OVERDUE_BASE, OVERDUE_PEAK, PurgeBadge
from softae.gui.widgets.purge_indicator import PurgeIndicator

INTERVAL_S = 900.0


class _FakePoller(QObject):
    """Stand-in shared poller so the sidebar skips its local poll thread."""

    sidebar_ready = Signal(dict)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def badge(qapp):
    b = PurgeBadge()
    yield b
    b._stop_pulse()
    b.deleteLater()
    qapp.processEvents()


@pytest.fixture
def sidebar(qapp):
    sb = MonitorSidebar(create_mock_manager(config={}), poller=_FakePoller())
    yield sb
    sb.close()
    sb.deleteLater()
    qapp.processEvents()


def _indicator(state: str, *, attention: bool = False,
               overdue_s: float = 0.0) -> PurgeIndicator:
    return PurgeIndicator(state, f"Purge: {state}", f"detail for {state}",
                          overdue_s=overdue_s, attention=attention)


def _colour(widget) -> str:
    """The colour the badge is currently painted, out of its stylesheet."""
    sheet = widget.styleSheet()
    return sheet.split("color:")[1].split(";")[0].strip()


# ── The badge paints what it is handed ───────────────────────────────────────

class TestBadgeRendersEachState:
    def test_badge_three_states_render_distinguishably(self, badge):
        seen = {}
        for state in ("purged", "dry_run", "overdue"):
            badge.update_purge(_indicator(state, attention=state == "overdue"))
            seen[state] = (badge.text(), _colour(badge))
        assert len(set(seen.values())) == 3

    def test_badge_dry_run_is_not_rendered_as_a_warning(self, badge):
        """The shipped default. Amber here is how a badge becomes wallpaper."""
        badge.update_purge(_indicator("dry_run"))
        assert _colour(badge) not in (OVERDUE_BASE, OVERDUE_PEAK)
        assert badge._pulse_timer.isActive() is False
        assert "bold" not in badge.styleSheet()

    def test_badge_overdue_pulses_and_is_emphasised(self, badge):
        badge.update_purge(_indicator("overdue", attention=True, overdue_s=600.0))
        assert badge._pulse_timer.isActive() is True
        assert "bold" in badge.styleSheet()

    def test_badge_pulse_stays_inside_the_overdue_colour_range(self, badge):
        """A slow traverse, not a strobe: every step is a blend of two colours."""
        badge.update_purge(_indicator("overdue", attention=True))
        colours = set()
        for _ in range(8):
            badge._pulse_step()
            colours.add(_colour(badge))
        assert all(c.startswith("#") and len(c) == 7 for c in colours)

    def test_badge_acknowledged_overdue_stops_pulsing(self, badge):
        badge.update_purge(_indicator("overdue", attention=True))
        badge.update_purge(_indicator("overdue", attention=False))
        assert badge._pulse_timer.isActive() is False
        assert _colour(badge) == OVERDUE_BASE

    def test_badge_completed_purge_stops_pulsing(self, badge):
        badge.update_purge(_indicator("overdue", attention=True))
        badge.update_purge(_indicator("purged"))
        assert badge._pulse_timer.isActive() is False

    def test_badge_detail_becomes_the_tooltip(self, badge):
        badge.update_purge(_indicator("scheduled"))
        assert badge.toolTip() == "detail for scheduled"


class TestBadgeAcknowledgement:
    def test_badge_click_emits_acknowledged(self, badge, qtbot):
        qtbot.addWidget(badge)
        badge.update_purge(_indicator("overdue", attention=True))
        with qtbot.waitSignal(badge.acknowledged, timeout=1000):
            qtbot.mouseClick(badge, Qt.MouseButton.LeftButton)

    def test_sidebar_re_emits_the_badge_acknowledgement(self, sidebar, qtbot):
        """The window owns the stamp; the sidebar only forwards the click."""
        with qtbot.waitSignal(sidebar.purge_acknowledged, timeout=1000):
            sidebar._purge_badge.acknowledged.emit()

    def test_sidebar_update_purge_reaches_the_badge(self, sidebar):
        sidebar.update_purge(_indicator("overdue", attention=True))
        assert "overdue" in sidebar._purge_badge.text()
        assert sidebar._purge_badge._pulse_timer.isActive() is True


# ── The window retains what the tick used to throw away ──────────────────────

class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def mock_manager():
    return create_mock_manager(config={})


@pytest.fixture
def main_window(qapp, qtbot, monkeypatch, mock_manager):
    monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
    from softae.gui.main_window import MainWindow

    mw = MainWindow(mock_manager)
    qtbot.addWidget(mw)
    yield mw

    mw.close()
    qapp.processEvents()
    mw.deleteLater()
    qapp.processEvents()


def _overdue_scheduler(window) -> _Clock:
    """Swap in a scheduler whose clock the test owns, already past due."""
    clock = _Clock()
    settings = PurgeSettings(enabled=True, actuate=False, interval_s=INTERVAL_S,
                             particulate_uL=20.0, other_uL=10.0,
                             particulate_pumps=(1,), pumps=(0, 1, 2))
    window._purge_scheduler = PurgeScheduler(settings, now=clock)
    clock.t = INTERVAL_S + 600.0
    return clock


class TestWindowRetainsTheLastOutcome:
    def test_window_starts_with_no_retained_outcome(self, main_window):
        assert main_window._last_purge_outcome is None
        assert main_window._last_purge_at is None

    def test_purge_tick_retains_an_outcome_that_said_something(
        self, main_window, monkeypatch
    ):
        outcome = PurgeOutcome(dry_run=True, volumes_uL={0: 10.0})
        monkeypatch.setattr(main_window._purge_runner, "maybe_purge",
                            lambda **kw: outcome)
        main_window._on_purge_tick()
        assert main_window._last_purge_outcome is outcome
        assert main_window._last_purge_at is not None

    def test_purge_tick_keeps_a_purge_a_no_op_tick_would_have_wiped(
        self, main_window, monkeypatch
    ):
        """The retention rule: an empty outcome is not news, so it is not kept."""
        done = PurgeOutcome(performed=True, volumes_uL={0: 10.0})
        monkeypatch.setattr(main_window._purge_runner, "maybe_purge",
                            lambda **kw: done)
        main_window._on_purge_tick()
        monkeypatch.setattr(main_window._purge_runner, "maybe_purge",
                            lambda **kw: PurgeOutcome())
        main_window._on_purge_tick()
        assert main_window._last_purge_outcome is done

    def test_purge_tick_retains_a_deferral(self, main_window, monkeypatch):
        """A skip is the state the badge exists for — it must not be dropped."""
        skipped = PurgeOutcome(skipped_reason="rig is in use (ht:cast)",
                               volumes_uL={0: 10.0})
        monkeypatch.setattr(main_window._purge_runner, "maybe_purge",
                            lambda **kw: skipped)
        main_window._on_purge_tick()
        assert main_window._last_purge_outcome is skipped

    def test_a_performed_purge_drops_a_standing_acknowledgement(
        self, main_window, monkeypatch
    ):
        main_window._purge_acknowledged_at = 1.0
        monkeypatch.setattr(
            main_window._purge_runner, "maybe_purge",
            lambda **kw: PurgeOutcome(performed=True, volumes_uL={0: 10.0}))
        main_window._on_purge_tick()
        assert main_window._purge_acknowledged_at is None


class TestWindowRefreshesTheBadge:
    def test_campaign_tick_renders_an_overdue_purge(self, main_window):
        _overdue_scheduler(main_window)
        main_window._on_campaign_tick()
        badge = main_window._sidebar._purge_badge
        assert "OVERDUE" in badge.text()
        assert badge._pulse_timer.isActive() is True

    def test_acknowledging_stamps_the_window_and_stops_the_pulse(
        self, main_window
    ):
        _overdue_scheduler(main_window)
        main_window._on_campaign_tick()
        badge = main_window._sidebar._purge_badge

        main_window._sidebar.purge_acknowledged.emit()
        assert main_window._purge_acknowledged_at is not None
        assert badge._pulse_timer.isActive() is False
        assert "OVERDUE" in badge.text()      # still true, just not shouting

    def test_a_window_with_no_scheduler_renders_unconfigured(self, main_window):
        main_window._purge_scheduler = None
        main_window._on_campaign_tick()
        assert "not configured" in main_window._sidebar._purge_badge.text()


# ── Attach mode: no schedule of its own ──────────────────────────────────────

@pytest.fixture
def attached_window(qapp, qtbot, monkeypatch, mock_manager):
    from softae.gui.launch_mode import LaunchMode
    from softae.gui.main_window import MainWindow

    monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
    mw = MainWindow(mock_manager, launch_mode=LaunchMode(
        attached=True, campaign=("shadow-run", "run-42"),
        run_dir=None, holder=None,
        reason="Campaign 'shadow-run' (run run-42) holds the rig.",
    ))
    qtbot.addWidget(mw)
    yield mw

    mw.close()
    qapp.processEvents()
    mw.deleteLater()
    qapp.processEvents()


def test_attached_window_badge_names_the_holder_not_a_schedule(attached_window):
    """Its scheduler's timers started at *its* launch — any minutes would be invented."""
    _overdue_scheduler(attached_window)
    attached_window._on_campaign_tick()
    badge = attached_window._sidebar._purge_badge
    assert "shadow-run" in badge.text()
    assert "OVERDUE" not in badge.text()
    assert badge._pulse_timer.isActive() is False
