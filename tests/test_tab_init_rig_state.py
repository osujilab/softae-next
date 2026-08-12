"""The instrument table says whether the rig is idle, working, or someone else's.

Three states over two independent facts, which is why they are composed rather than
enumerated: the per-instrument ``asyncio.Lock`` (``IDLE``/``ACTIVE``, advisory) and the
cross-process rig lock (``OCCUPIED``, authoritative). The tests below pin the boundary
between them -- particularly that nothing gates on the advisory one.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from softae.core.run_lock import RunLock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_init import (
    InitCalibrationTab,
    _annotate_busy,
    _compose_state,
    _owner_line,
)


def _foreign_lock(**kw) -> RunLock:
    """A lock owned by some other live process on this host."""
    defaults = dict(pid=os.getpid() + 1, what="commissioning blank_short",
                    started_at="2026-08-07T14:02:00+00:00", host="", log_path="")
    defaults.update(kw)
    return RunLock(**defaults)


def _own_lock(**kw) -> RunLock:
    defaults = dict(pid=os.getpid(), what="geometry series", started_at="now",
                    host="", log_path="")
    defaults.update(kw)
    return RunLock(**defaults)


# ── Composition ──────────────────────────────────────────────────────────────


class TestComposeState:
    def test_connected_and_unlocked_reads_idle_rather_than_bare_connected(self):
        """`CONNECTED` alone cannot distinguish "ready" from "mid-sweep"."""
        assert _compose_state("CONNECTED") == "CONNECTED · IDLE"

    def test_connected_and_locked_reads_active(self):
        assert _compose_state("CONNECTED", busy=True) == "CONNECTED · ACTIVE"

    def test_a_foreign_rig_lock_dominates_every_row_because_it_is_rig_wide(self):
        """The constraint is the hardware, not this manager's view of it.

        A headless child owns the ports; what this process believes about its own
        connections is no longer a description of what can be driven.
        """
        lock = _foreign_lock()
        for state in ("CONNECTED", "DISCONNECTED", "ERROR", "CONNECTING"):
            assert _compose_state(state, rig_lock=lock) == "OCCUPIED"

    def test_a_foreign_lock_outranks_a_busy_instrument(self):
        assert _compose_state(
            "CONNECTED", busy=True, rig_lock=_foreign_lock()) == "OCCUPIED"

    def test_this_processs_own_lock_leaves_the_per_instrument_view_intact(self):
        """A GUI-run workflow takes the same lock. Its own instrument locks are then
        the more informative signal, so ownership must not blank them out."""
        lock = _own_lock()
        assert _compose_state("CONNECTED", busy=True, rig_lock=lock) == "CONNECTED · ACTIVE"
        assert _compose_state("CONNECTED", rig_lock=lock) == "CONNECTED · IDLE"

    def test_states_other_than_connected_pass_through_unchanged(self):
        for state in ("DISCONNECTED", "ERROR", "CONNECTING", "UNKNOWN"):
            assert _compose_state(state, busy=True) == state


class TestOwnerLine:
    def test_the_owner_line_names_the_run_and_its_start_not_merely_busy(self):
        """PID reuse means a live-looking lock may be stale; only a human can tell,
        and only if the row says what and when."""
        text = _owner_line(_foreign_lock())
        assert "commissioning blank_short" in text
        assert "14:02" in text
        assert "\n" not in text, "must fit one table cell"

    def test_an_unnamed_run_still_renders(self):
        assert "unnamed run" in _owner_line(_foreign_lock(what="", started_at=""))


# ── Busy annotation ──────────────────────────────────────────────────────────


class TestAnnotateBusy:
    def test_busy_reflects_the_instruments_own_lock(self):
        manager = create_mock_manager(config={})
        statuses = manager.list_instruments()
        _annotate_busy(manager, statuses)
        assert all(s["busy"] is False for s in statuses)

        name = statuses[0]["name"]
        # Acquire without an event loop: `locked()` only reads the flag.
        manager.get(name)._lock._locked = True
        _annotate_busy(manager, statuses)
        assert next(s for s in statuses if s["name"] == name)["busy"] is True

    def test_an_unreadable_instrument_reports_not_busy_rather_than_raising(self):
        """This decorates a table. It must never be why the table stops refreshing."""
        class _Boom:
            def get(self, name):
                raise RuntimeError("driver layer is down")

        statuses = [{"name": "stage", "state": "CONNECTED"}]
        _annotate_busy(_Boom(), statuses)
        assert statuses[0]["busy"] is False


# ── Rendering ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def tab(qapp):
    widget = InitCalibrationTab(create_mock_manager(config={}))
    yield widget
    widget.cleanup()
    widget.close()


def _render(tab, statuses, rig_lock=None):
    tab._refresh_table(statuses, {}, {"pico1": "", "pico2": ""}, rig_lock)


class TestTableRendering:
    def test_the_state_column_shows_the_composed_text(self, tab):
        _render(tab, [
            {"name": "stage", "state": "CONNECTED", "busy": False},
            {"name": "syringe", "state": "CONNECTED", "busy": True},
            {"name": "pico1", "state": "DISCONNECTED", "busy": False},
        ])
        assert [tab._table.item(r, 2).text() for r in range(3)] == [
            "CONNECTED · IDLE", "CONNECTED · ACTIVE", "DISCONNECTED"]

    def test_busy_is_shown_in_the_state_column_not_leaked_into_details(self, tab):
        """`busy` is annotation, not instrument telemetry -- Details is for the latter."""
        _render(tab, [{"name": "stage", "state": "CONNECTED", "busy": True,
                       "position": (1.0, 2.0)}])
        details = tab._table.item(0, 3).text()
        assert "busy" not in details
        assert "position" in details

    def test_an_occupied_rig_replaces_details_with_the_owner_on_every_row(self, tab):
        """Those readings came from a manager that is not driving the hardware."""
        _render(tab, [
            {"name": "stage", "state": "CONNECTED", "busy": False, "position": (0, 0)},
            {"name": "pico1", "state": "DISCONNECTED", "busy": False},
        ], _foreign_lock())
        for row in range(2):
            assert tab._table.item(row, 2).text() == "OCCUPIED"
            assert "commissioning blank_short" in tab._table.item(row, 3).text()
            assert "PID" in tab._table.item(row, 2).toolTip()

    def test_the_last_polled_lock_is_retained_for_other_widgets(self, tab):
        lock = _foreign_lock()
        _render(tab, [{"name": "stage", "state": "CONNECTED"}], lock)
        assert tab._rig_lock is lock


# ── The guard that actually gates ────────────────────────────────────────────


def _sink():
    """Stand-in for ``_schedule_async`` that closes what it is handed.

    An unscheduled coroutine emits "never awaited" at collection time, in whichever
    unrelated test happens to trigger the GC.
    """
    return lambda coro: coro.close()


class TestConnectGuard:
    def test_connect_all_refuses_while_another_process_holds_the_rig(self, tab):
        with patch("softae.gui.tabs.tab_init._read_rig_lock",
                   return_value=_foreign_lock()), \
             patch("softae.gui.tabs.tab_init.QMessageBox.warning") as warn, \
             patch.object(tab, "_schedule_async") as sched:
            tab._on_connect_all()
        assert sched.call_count == 0, "must not open ports the child is driving"
        assert warn.call_count == 1

    def test_connect_selected_is_guarded_too(self, tab):
        with patch("softae.gui.tabs.tab_init._read_rig_lock",
                   return_value=_foreign_lock()), \
             patch("softae.gui.tabs.tab_init.QMessageBox.warning"), \
             patch.object(tab, "_selected_instrument", return_value="stage"), \
             patch.object(tab, "_schedule_async") as sched:
            tab._on_connect_selected()
        assert sched.call_count == 0

    def test_disconnect_all_is_not_guarded_because_letting_go_never_collides(self, tab):
        with patch("softae.gui.tabs.tab_init._read_rig_lock",
                   return_value=_foreign_lock()), \
             patch.object(tab, "_schedule_async", side_effect=_sink()) as sched:
            tab._on_disconnect_all()
        assert sched.call_count == 1

    def test_this_processs_own_lock_does_not_block_connecting(self, tab):
        """Re-entrancy: the GUI holding its own lock must not lock itself out."""
        with patch("softae.gui.tabs.tab_init._read_rig_lock",
                   return_value=_own_lock()), \
             patch.object(tab, "_schedule_async", side_effect=_sink()) as sched:
            tab._on_connect_all()
        assert sched.call_count == 1

    def test_the_guard_reads_fresh_rather_than_trusting_the_2s_poll(self, tab):
        """A run that started inside the poll window is the case worth catching."""
        tab._rig_lock = None
        with patch("softae.gui.tabs.tab_init._read_rig_lock",
                   return_value=_foreign_lock()) as read, \
             patch("softae.gui.tabs.tab_init.QMessageBox.warning"), \
             patch.object(tab, "_schedule_async") as sched:
            tab._on_connect_all()
        assert read.call_count == 1
        assert sched.call_count == 0


# ── Launcher entry point ─────────────────────────────────────────────────────


class TestBenchSequencesButton:
    def test_the_button_exists_on_the_configuration_tab(self, tab):
        assert tab._btn_bench.isEnabled()

    def test_it_opens_the_launcher_against_the_stores_project_directory(self, tab):
        tab._data_store = type("_S", (), {"project_dir": "/tmp/proj"})()
        with patch("softae.gui.widgets.calibration_launcher."
                   "CalibrationLauncherDialog") as dlg:
            tab._on_bench_sequences()
        assert dlg.call_args.args[1] == "/tmp/proj"

    def test_without_a_store_it_falls_back_to_the_configured_project_dir(self, tab):
        tab._data_store = None
        with patch("softae.config.loader.data_project_dir",
                   return_value="~/softae_data"):
            assert tab._project_dir() == "~/softae_data"
