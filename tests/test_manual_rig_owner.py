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
from softae.gui.launch_mode import decide_launch_mode
from softae.gui.tabs import tab_manual as tm
from softae.gui.tabs.tab_manual import REFUSED_WHILE_ATTACHED, ManualControlTab
from softae.gui.widgets import rig_owner
from softae.gui.widgets.rig_owner import (
    ATTACHED,
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


def _attached_mode(lock: RunLock | None = None):
    """The launch decision an attached window is built with.

    Produced by the real :func:`decide_launch_mode` rather than assembled field
    by field, so these tests exercise the same object the launcher passes in —
    including its ``reason``, which is what the banner's tooltip shows.
    """
    return decide_launch_mode(lock_reader=lambda: lock or _foreign_lock())


@pytest.fixture
def attached_tab(qapp, monkeypatch):
    """A Manual tab launched *attached* — no session of its own, anywhere.

    The rig lock reads as **free** for its whole life, deliberately: the refusal
    proved here must come from the launch decision and from nothing else, and a
    lock left in place would let a lock-derived implementation pass.
    """
    monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: None)
    widget = ManualControlTab(create_mock_manager(config={}), launch_mode=_attached_mode())
    yield widget
    widget.cleanup()
    widget.close()


#: One entry per actuating family whose ownership note is the slot's first act,
#: mirroring `TestActuationIsNeverRefused` case for case.
_ACTUATIONS = [
    ("stage go-to", "stage", "move_to", lambda t: t._on_goto()),
    ("stage jog", "stage", "move_by", lambda t: t._on_jog(1, 0)),
    ("temperature setpoint", "temp_controller", "write_sp", lambda t: t._on_set_temp()),
    ("humidity setpoint", "rh_controller", "set_setpoint", lambda t: t._on_set_rh()),
    ("dispenser head descend", "syringe", "head_descend", lambda t: t._on_head_descend()),
    ("dispenser head retract", "syringe", "head_retract", lambda t: t._on_head_retract()),
    ("lamp on", "lamp", "on", lambda t: t._on_lamp_on()),
]


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


# ── Attach mode: nothing to command, said out loud ───────────────────────────


class TestAttachedWindowRefuses:
    """A window launched attached opened no session, so its controls reach nothing.

    This is not the refusal the operator forbade. That one is *"a foreign
    campaign holds the lock"*, which is a live rig's normal state while manual
    control is legitimately in use — every case in
    `TestActuationIsNeverRefused` above still actuates through exactly that
    condition, unmodified, and their passing is the proof this predicate is a
    different one. What is refused here is a command with no session to travel
    down; what replaces it is a sentence naming the run that has them.
    """

    def test_manual_control_in_attach_mode_names_the_campaign_and_does_not_actuate(
            self, attached_tab, monkeypatch, settle_qt):
        calls = []
        stage = attached_tab._manager.get("stage")
        monkeypatch.setattr(stage, "move_to", lambda x, y: calls.append((x, y)))

        attached_tab._spin_x.setValue(3.0)
        attached_tab._on_goto()
        settle_qt(attached_tab)

        assert calls == []
        for surface in (attached_tab._lbl_rig_owner.text(),
                        attached_tab._lbl_last_command.text()):
            assert "phase_map" in surface                    # which campaign
            assert "20260817T090000Z_phase_map" in surface   # which run

    @pytest.mark.parametrize(
        "action,instrument,method,press", _ACTUATIONS, ids=[a[0] for a in _ACTUATIONS])
    def test_attached_actuating_slot_refuses_and_calls_no_driver(
            self, attached_tab, monkeypatch, settle_qt, action, instrument, method, press):
        calls = []
        driver = attached_tab._manager.get(instrument)
        monkeypatch.setattr(driver, method, lambda *a, **k: calls.append((a, k)))

        press(attached_tab)
        settle_qt(attached_tab)

        assert calls == []
        assert action in attached_tab._lbl_last_command.text()
        assert "Refused" in attached_tab._lbl_last_command.text()

    def test_attached_pump_refuses_before_any_fluid_is_commanded(
            self, attached_tab, monkeypatch, settle_qt):
        """The one that would cost a board — driven exactly as its owner-mode twin."""
        calls = []
        syr = attached_tab._manager.get("syringe")
        monkeypatch.setattr(syr, "single_pump", lambda **kw: calls.append(kw))

        attached_tab._chk_apply_correction.setChecked(False)
        attached_tab._pump_widgets[0]["vol"].setValue(12.0)
        attached_tab._on_infuse(0)
        settle_qt(attached_tab)

        assert calls == []

    def test_attached_note_returns_the_refusal_sentinel_not_a_lock(self, attached_tab):
        """The answer the 16 call sites branch on, pinned at the seam itself."""
        assert attached_tab._note_manual_actuation("stage go-to") is REFUSED_WHILE_ATTACHED

    def test_attached_refusal_holds_with_no_rig_lock_at_all(self, attached_tab, monkeypatch):
        """The predicate is the launch decision, never a lock read at press time.

        A lock-derived implementation passes every other test in this class and
        fails this one, which is the whole point of it.
        """
        _hold_rig(monkeypatch, None)
        assert attached_tab._note_manual_actuation("stage jog") is REFUSED_WHILE_ATTACHED

    def test_attached_banner_survives_the_campaign_lock_disappearing(
            self, attached_tab, monkeypatch):
        """The campaign ending does not hand this window the sessions.

        A banner that cleared with the lock would leave controls refusing with
        nothing on screen to say why.
        """
        _hold_rig(monkeypatch, None)
        attached_tab.refresh_rig_owner()

        assert not attached_tab._lbl_rig_owner.isHidden()
        assert ATTACHED in attached_tab._lbl_rig_owner.text()

    def test_attached_refusal_leaves_a_log_line_naming_the_run(self, attached_tab):
        with structlog.testing.capture_logs() as logs:
            attached_tab._note_manual_actuation("pump 0 dispense")

        entry = next(e for e in logs
                     if e["event"] == "manual_actuation_refused_while_attached")
        assert entry["action"] == "pump 0 dispense"
        assert entry["campaign"] == "phase_map"
        assert entry["run_id"] == "20260817T090000Z_phase_map"

    def test_attached_refusal_opens_no_modal_dialog(self, attached_tab, monkeypatch):
        """A modal here blocks the event loop of a window whose only job is to
        keep rendering a live campaign — and a queued one wedges the test run."""
        def boom(*a, **k):
            raise AssertionError("the refusal must not open a dialog")

        monkeypatch.setattr(tm.QMessageBox, "warning", boom)
        monkeypatch.setattr(tm.QMessageBox, "information", boom)
        attached_tab._on_head_descend()

    def test_attached_to_a_non_campaign_holder_refuses_without_inventing_a_campaign(
            self, qapp, monkeypatch):
        """A bench sequence holds the rig: still attached, still nothing to command."""
        monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: None)
        mode = _attached_mode(_foreign_lock(what="workflow 'blank_short'", pid=77))
        tab = ManualControlTab(create_mock_manager(config={}), launch_mode=mode)
        try:
            assert tab._note_manual_actuation("lamp on") is REFUSED_WHILE_ATTACHED
            assert "another process" in tab._lbl_rig_owner.text()
            assert "campaign '" not in tab._lbl_rig_owner.text()
        finally:
            tab.cleanup()
            tab.close()


class TestOwnerModeIsUnchangedByTheAttachPredicate:
    """The two predicates that would have been easier, and why they are illegal."""

    def test_an_explicit_owner_launch_mode_still_actuates_over_a_foreign_lock(
            self, qapp, monkeypatch, settle_qt):
        """Passing a mode is not what refuses — being attached is."""
        monkeypatch.setattr(rig_owner, "foreign_rig_lock", lambda: _foreign_lock())
        owner = decide_launch_mode(lock_reader=lambda: None)
        assert owner.owner
        tab = ManualControlTab(create_mock_manager(config={}), launch_mode=owner)
        try:
            calls = []
            monkeypatch.setattr(tab._manager.get("lamp"), "on", lambda: calls.append("on"))
            tab._on_lamp_on()
            assert calls == ["on"]
        finally:
            tab.cleanup()
            tab.close()

    def test_the_mock_rig_is_disconnected_so_connection_cannot_be_the_predicate(self, tab):
        """Why "this process holds no connected instruments" was rejected.

        Nothing calls `connect_all()` on the fixture's manager, so every
        instrument reports disconnected — and every case in
        `TestActuationIsNeverRefused` deliberately actuates one anyway. A
        connection-state refusal would refuse all eight.
        """
        assert not any(status.get("connected")
                       for status in tab._manager.status_all().values())


class TestAttachedReadOnlySurfaces:
    """Renders stay; instrument *reads* do not.

    An attached window may read, narrate and request — but a driver read is a
    serial transaction on a bus the owning process is mid-anneal on, which is why
    `MainWindow._instrument_source` already declines to make one. This tab has
    its own polling worker, so it has to decline separately.
    """

    def test_attached_tab_starts_no_instrument_polling_worker(self, attached_tab):
        assert attached_tab._pv_worker is None

    def test_owner_tab_still_starts_its_instrument_polling_worker(self, tab):
        assert tab._pv_worker is not None

    def test_attached_head_label_reports_the_owner_not_a_registered_belief(
            self, attached_tab):
        """`is_head_up()` is state *this* process registered, and it registered none."""
        text = attached_tab._lbl_head_status.text()
        assert "Retracted" not in text and "Descended" not in text
        assert "phase_map" in text

    def test_attached_readouts_keep_their_placeholders_rather_than_guessing(
            self, attached_tab):
        assert "--" in attached_tab._lbl_temp_pv.text()
        assert "?" in attached_tab._lbl_pos.text()
