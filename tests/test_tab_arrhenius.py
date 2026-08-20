"""Daemon-shutdown tests for the Arrhenius sweep tab (tab_arrhenius.py).

Hardware safety: cleanup()/abort_run() must signal the sweep's abort (which
stops issuing temp-controller / potentiostat commands) before any join.
"""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_arrhenius import ArrheniusTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


@pytest.fixture
def tab(qapp, manager):
    widget = ArrheniusTab(manager)
    yield widget
    widget.close()


class _StubSweep:
    """Sweep stub whose abort() sets a threading.Event (the run's abort signal)."""

    def __init__(self) -> None:
        self.ev = threading.Event()

    def abort(self) -> None:
        self.ev.set()


def _spin_on_event(ev: threading.Event) -> threading.Thread:
    def run() -> None:
        while not ev.is_set():
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestDaemonShutdown:
    def test_arrhenius_cleanup_aborts_running_thread(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        assert tab._sweep_thread.is_alive()
        tab.cleanup()
        assert sw.ev.is_set()
        assert tab._abort_requested is True
        assert not tab._sweep_thread.is_alive()

    def test_arrhenius_cleanup_is_noop_when_idle(self, tab: ArrheniusTab):
        assert getattr(tab, "_sweep", None) is None
        assert tab._sweep_thread is None
        tab.cleanup()  # must not raise / block

    def test_arrhenius_cleanup_is_idempotent(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        tab.cleanup()
        tab.cleanup()
        assert not tab._sweep_thread.is_alive()

    def test_arrhenius_abort_run_signals_without_joining(self, tab: ArrheniusTab):
        sw = _StubSweep()
        tab._sweep = sw
        tab._sweep_thread = _spin_on_event(sw.ev)
        tab.abort_run()
        assert sw.ev.is_set()
        assert tab._abort_requested is True
        assert tab._sweep_thread.is_alive()  # signal-only: not joined
        tab.cleanup()  # teardown join


class TestChannelParsing:
    """The channels field feeds ``ArrheniusSweepConfig(channels=...)`` directly.

    Order matters here in a way it does not elsewhere: for a sweep, channel
    order *is* measurement order, so these pin order preservation as hard as
    they pin validation.  ``_parse_channels`` delegates to the shared parser,
    which raises ``ChannelSpecError`` — a ``ValueError`` subclass, which is
    what the ``except (ValueError, TypeError)`` at the save-config call site
    catches.
    """

    def test_arrhenius_parse_channels_preserves_entry_order(self, tab: ArrheniusTab):
        tab.set_pcb_channel_count(32)
        tab._le_channels.setText("10, 2, 3-5")
        assert tab._parse_channels() == [10, 2, 3, 4, 5]

    def test_arrhenius_parse_channels_dedups_first_wins(self, tab: ArrheniusTab):
        tab.set_pcb_channel_count(32)
        tab._le_channels.setText("3, 1-3")
        assert tab._parse_channels() == [3, 1, 2]

    def test_arrhenius_parse_channels_bad_token_raises_a_value_error(
        self, tab: ArrheniusTab
    ):
        from softae.core.channel_spec import ChannelSpecError

        tab._le_channels.setText("1, x")
        with pytest.raises(ValueError):          # the call site's except clause
            tab._parse_channels()
        tab._le_channels.setText("1, x")
        with pytest.raises(ChannelSpecError):    # ...and it is the specific one
            tab._parse_channels()

    def test_arrhenius_parse_channels_empty_raises(self, tab: ArrheniusTab):
        tab._le_channels.setText("   ")
        with pytest.raises(ValueError):
            tab._parse_channels()

    def test_arrhenius_parse_channels_enforces_the_board_channel_bound(
        self, tab: ArrheniusTab
    ):
        tab.set_pcb_channel_count(8)
        tab._le_channels.setText("1, 9")
        with pytest.raises(ValueError):
            tab._parse_channels()
        tab.set_pcb_channel_count(16)
        assert tab._parse_channels() == [1, 9]   # same text, wider board
