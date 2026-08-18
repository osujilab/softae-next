"""Manual Control tells the operator who owns the rig — and never stops them.

The hazard being closed is real: `tab_manual.py` reaches the drivers directly on a
bare `QThread`, takes no per-instrument lock, and until now had no idea whether a
campaign in another process was mid-cast. Whether a collision raised or silently
interleaved was decided by the transport (PyVISA opens unlocked; the MCP2221 bus
lock is process-local), and the silent-interleave case is the dangerous one.

**The fix is awareness, not enforcement, and that is the whole design.** An
operator standing at the rig frequently reaches for manual control *because*
something has gone wrong; refusing them at that moment is the failure, not the
protection. So every test here that drives an actuation asserts the driver was
**still called**. Stopping a campaign is a designated, scoped control — E-Stop for
the rig, Abort and Pause for a campaign, each in the container that surfaces the
run — and none of them is an interlock on a jog button.
"""

from __future__ import annotations

import os
import socket

import pytest
import structlog

from softae.core.run_lock import RunLock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs import tab_manual as tm
from softae.gui.tabs.tab_manual import ManualControlTab
from softae.gui.widgets import rig_owner
from softae.gui.widgets.rig_owner import (
    OCCUPIED,
    campaign_identity,
    foreign_rig_lock,
    owner_line,
)


def _foreign_lock(**over) -> RunLock:
    base = dict(pid=4242, what="campaign:phase_map:20260817T090000Z_phase_map",
                started_at="2026-08-17T09:00:00+00:00",
                host="another-host", log_path=r"C:\proj\runs\20260817T090000Z_phase_map")
    base.update(over)
    return RunLock(**base)


def _my_lock(**over) -> RunLock:
    base = dict(pid=os.getpid(), what="workflow 'ht_sequence'",
                started_at="2026-08-17T09:00:00+00:00",
                host=socket.gethostname())
    base.update(over)
    return RunLock(**base)


@pytest.fixture
def tab(qapp, monkeypatch):
    """A Manual tab whose view of the rig lock is ours to set.

    Patched at the source so nothing in this module reads — or writes — the
    machine's real `~/.softae/rig.lock`. The banner refreshes on a 2 s timer in
    the running GUI; every test here calls `refresh_rig_owner()` itself, so the
    timer is never what a test waits on.
    """
    monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: None)
    widget = ManualControlTab(create_mock_manager(config={}))
    yield widget
    widget.cleanup()
    widget.close()


def _hold_rig(monkeypatch, lock: RunLock | None) -> None:
    monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: lock)


# ── One vocabulary for ownership ─────────────────────────────────────────────


class TestOwnerVocabulary:
    def test_the_owner_line_names_pid_run_and_start_not_merely_busy(self):
        text = owner_line(_foreign_lock())
        assert "4242" in text
        assert "campaign:phase_map" in text
        assert "2026-08-17T09:00:00+00:00" in text

    def test_the_init_tab_and_the_manual_banner_render_one_lock_one_way(self):
        """Two spellings of one lock file is how an operator learns to distrust both."""
        from softae.gui.tabs.tab_init import _owner_line as init_owner_line

        lock = _foreign_lock()
        assert init_owner_line(lock) == owner_line(lock)

    def test_a_campaign_lock_yields_its_campaign_and_run(self):
        assert campaign_identity(_foreign_lock()) == (
            "phase_map", "20260817T090000Z_phase_map")

    def test_a_workflow_lock_has_no_campaign_identity(self):
        """An HT sequence is not a campaign and must not be labelled as one."""
        assert campaign_identity(_my_lock()) is None

    def test_our_own_lock_is_not_a_foreign_owner(self, monkeypatch):
        """A GUI running its own sequence is not a second owner of anything."""
        monkeypatch.setattr(
            "softae.core.run_lock.read_run_lock", lambda *a, **k: _my_lock())
        assert foreign_rig_lock() is None

    def test_an_unreadable_lock_does_not_take_the_tab_down_with_it(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("lock file is on a dead network share")

        monkeypatch.setattr("softae.core.run_lock.read_run_lock", boom)
        assert foreign_rig_lock() is None


# ── The banner ───────────────────────────────────────────────────────────────


class TestOwnerBanner:
    def test_a_free_rig_shows_no_banner_at_all(self, tab):
        # `isHidden` rather than `isVisible`: a widget whose window was never
        # shown is invisible regardless, so `isVisible` would pass this test even
        # if the banner were never hidden at all.
        assert tab.refresh_rig_owner() is None
        assert tab._lbl_rig_owner.isHidden()
        assert tab._lbl_rig_owner.text() == ""

    def test_a_foreign_campaign_is_named_in_full(self, tab, monkeypatch):
        _hold_rig(monkeypatch, _foreign_lock())
        tab.refresh_rig_owner()

        text = tab._lbl_rig_owner.text()
        assert OCCUPIED in text
        assert "phase_map" in text                       # which campaign
        assert "20260817T090000Z_phase_map" in text      # which run
        assert "4242" in text                            # which process
        assert "2026-08-17T09:00:00+00:00" in text       # since when
        assert tab._lbl_rig_owner.toolTip()

    def test_the_banner_says_manual_control_still_works(self, tab, monkeypatch):
        """It must not read as a lockout notice — nothing here locks anything out."""
        _hold_rig(monkeypatch, _foreign_lock())
        tab.refresh_rig_owner()

        text = tab._lbl_rig_owner.text().lower()
        assert "not" in text and "block" in text

    def test_the_banner_points_at_the_owning_process_for_the_stop_controls(
            self, tab, monkeypatch):
        """Pause and Abort belong to the container that surfaces the run.

        The campaign is in another process; duplicating its controls here would
        mean shipping buttons with nothing behind them. The banner routes instead.
        """
        _hold_rig(monkeypatch, _foreign_lock())
        tab.refresh_rig_owner()

        text = tab._lbl_rig_owner.text().lower()
        assert "pause" in text or "stop" in text
        assert "owns it" in text or "owning" in text

    def test_the_banner_clears_when_the_campaign_finishes(self, tab, monkeypatch):
        _hold_rig(monkeypatch, _foreign_lock())
        tab.refresh_rig_owner()
        assert not tab._lbl_rig_owner.isHidden()
        assert tab._lbl_rig_owner.text()

        _hold_rig(monkeypatch, None)
        tab.refresh_rig_owner()
        assert tab._lbl_rig_owner.isHidden()
        assert tab._lbl_rig_owner.text() == ""

    def test_a_workflow_lock_is_reported_without_inventing_a_campaign(
            self, tab, monkeypatch):
        _hold_rig(monkeypatch, _foreign_lock(what="workflow 'blank_short'", pid=77))
        tab.refresh_rig_owner()

        text = tab._lbl_rig_owner.text()
        assert "Another process" in text
        assert "blank_short" in text


# ── The central claim: actuation is never refused ────────────────────────────


class TestActuationIsNeverRefused:
    """One test per actuation family. Every one asserts the driver was called.

    Each drives the handler, joins its command thread (`settle_qt`), and then
    asserts. The control coming back enabled is asserted rather than waited on:
    a button that stays dead is a refusal by another name, and the old
    poll-until-enabled loop turned that failure into a three-second timeout that
    the test then passed anyway.
    """

    def test_stage_go_to_still_moves_the_stage(self, tab, monkeypatch, settle_qt):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        stage = tab._manager.get("stage")
        monkeypatch.setattr(stage, "move_to", lambda x, y: calls.append((x, y)))

        tab._spin_x.setValue(3.0)
        tab._spin_y.setValue(4.0)
        tab._on_goto()
        settle_qt(tab)

        assert calls == [(3.0, 4.0)]
        assert tab._btn_goto.isEnabled()

    def test_stage_jog_still_moves_the_stage(self, tab, monkeypatch, settle_qt):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        stage = tab._manager.get("stage")
        monkeypatch.setattr(stage, "move_by", lambda dx, dy: calls.append((dx, dy)))

        tab._spin_jog_step.setValue(1.0)
        tab._on_jog(1, 0)
        settle_qt(tab)

        assert calls == [(1.0, 0.0)]
        assert tab._jog_buttons[0].isEnabled()

    def test_a_pump_still_dispenses(self, tab, monkeypatch, settle_qt):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        syr = tab._manager.get("syringe")
        monkeypatch.setattr(syr, "single_pump", lambda **kw: calls.append(kw))

        tab._chk_apply_correction.setChecked(False)
        tab._pump_widgets[0]["vol"].setValue(12.0)
        tab._on_infuse(0)
        settle_qt(tab)

        assert calls and calls[0]["dispense_vol"] == pytest.approx(12.0)
        assert tab._pump_widgets[0]["btn"].isEnabled()

    def test_the_head_still_descends(self, tab, monkeypatch, settle_qt):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        syr = tab._manager.get("syringe")
        monkeypatch.setattr(syr, "head_descend", lambda: calls.append("down"))

        tab._on_head_descend()
        settle_qt(tab)

        assert calls == ["down"]

    def test_the_head_still_retracts(self, tab, monkeypatch, settle_qt):
        """The safe direction is not a special case — nothing is refused, so
        nothing needs an exemption."""
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        syr = tab._manager.get("syringe")
        monkeypatch.setattr(syr, "head_retract", lambda: calls.append("up"))

        tab._on_head_retract()
        settle_qt(tab)

        assert calls == ["up"]

    def test_the_temperature_setpoint_still_changes(self, tab, monkeypatch, settle_qt):
        """Named explicitly by the operator as a thing they would do mid-pause."""
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        tc = tab._manager.get("temp_controller")
        monkeypatch.setattr(tc, "write_sp", lambda **kw: calls.append(kw))

        tab._spin_temp.setValue(65.0)
        tab._on_set_temp()
        settle_qt(tab)

        assert calls and calls[0]["T_SP"] == pytest.approx(65.0)
        assert tab._btn_set_temp.isEnabled()

    def test_the_humidity_setpoint_still_changes(self, tab, monkeypatch, settle_qt):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        rh = tab._manager.get("rh_controller")
        monkeypatch.setattr(rh, "set_setpoint", lambda sp: calls.append(sp))

        tab._spin_rh.setValue(35.0)
        tab._on_set_rh()
        settle_qt(tab)

        assert calls == [pytest.approx(35.0)]
        assert tab._btn_set_rh.isEnabled()

    def test_the_lamp_still_switches(self, tab, monkeypatch):
        _hold_rig(monkeypatch, _foreign_lock())
        calls = []
        lamp = tab._manager.get("lamp")
        monkeypatch.setattr(lamp, "on", lambda: calls.append("on"))

        tab._on_lamp_on()

        assert calls == ["on"]


# ── The overlap is recorded, since it is allowed ─────────────────────────────


class TestOverlapIsRecorded:
    def test_manual_use_over_a_live_campaign_leaves_a_log_line(self, tab, monkeypatch):
        """Allowed is not the same as unremarkable.

        If a collision does happen, it should be in the log with a timestamp and
        an owner beside it, not reconstructed afterwards from a ruined board.
        """
        _hold_rig(monkeypatch, _foreign_lock())
        with structlog.testing.capture_logs() as logs:
            tab._note_manual_actuation("stage go-to")

        entry = next(e for e in logs
                     if e["event"] == "manual_actuation_during_foreign_run")
        assert entry["action"] == "stage go-to"
        assert entry["owner_pid"] == 4242
        assert entry["owner_what"] == "campaign:phase_map:20260817T090000Z_phase_map"

    def test_the_status_line_says_the_command_overlapped_a_run(self, tab, monkeypatch):
        _hold_rig(monkeypatch, _foreign_lock())
        tab._note_manual_actuation("pump 0 dispense")

        text = tab._lbl_last_command.text()
        assert "pump 0 dispense" in text
        assert "4242" in text

    def test_a_free_rig_logs_nothing_and_says_nothing(self, tab):
        before = tab._lbl_last_command.text()
        with structlog.testing.capture_logs() as logs:
            assert tab._note_manual_actuation("stage go-to") is None

        assert not [e for e in logs
                    if e["event"] == "manual_actuation_during_foreign_run"]
        assert tab._lbl_last_command.text() == before

    def test_every_actuating_slot_reports_before_it_acts(self):
        """A new control that forgets this is the way the coverage rots.

        Asserted on the source rather than by driving fourteen widgets, because
        the property being defended is 'no actuating handler is missing the call',
        which is a statement about the set of handlers.
        """
        import inspect

        source = inspect.getsource(tm.ManualControlTab)
        actuating = [
            "_on_goto", "_on_jog", "_on_set_temp", "_on_ramp", "_on_set_rh",
            "_on_rh_start", "_on_rh_stop", "_on_head_retract", "_on_head_descend",
            "_on_infuse", "_on_piezo_a_on", "_on_piezo_a_off",
            "_on_piezo_apply_settings", "_on_eis_run", "_on_lamp_on",
            "_on_lamp_off",
        ]
        missing = []
        for name in actuating:
            body = source.split(f"def {name}(", 1)[1].split("\n    def ", 1)[0]
            if "_note_manual_actuation" not in body:
                missing.append(name)
        assert missing == [], f"actuating handlers with no ownership note: {missing}"
