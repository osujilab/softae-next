"""The E-Stop escalation ladder: what a press reaches, and what it never reaches.

Three properties carry this step's risk, and each has its own class below.

**No timer performs a rung.** T1 and T2 bound how long the ladder waits before
*offering* the next rung; auto-advancing would turn a slow park — a long EIS
sweep, a contended serial bus — into a killed process mid-dispense, which is the
un-parked death the whole control channel exists to avoid.

**The kill is by process id, and by one process id.** Never by image name: on
this machine the operator's GUI, the extension hosts and pytest share one
interpreter and one virtualenv. No test here terminates anything — the
terminator is injected everywhere, and the two cases that exercise the real
:func:`terminate_pid` exercise only its refusals.

**Rung 4 connects before it parks.** A park against a manager with nothing
connected commands nothing and raises nothing, so parking first would produce a
reassuring result about a rig this process had not yet spoken to.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket

import pytest

from softae.core.run_lock import RunLock
from softae.core.safe_park import (
    HEADLINE_COMMANDED,
    HEADLINE_NOTHING,
    SafeParkResult,
)
from softae.gui.estop_ladder import (
    STATE_ACKED_ONLY,
    STATE_AWAITING_ACK,
    STATE_AWAITING_PARK,
    STATE_EXHAUSTED,
    STATE_OFFERED_TAKEOVER,
    STATE_OFFERED_WAIT,
    STATE_PARKED,
    T1_ACK_S,
    T2_PARK_S,
    EstopLadder,
    reachable_rungs,
    terminate_pid,
)
from softae.gui.launch_mode import LaunchMode

THIS_HOST = socket.gethostname()


# ── Doubles ──────────────────────────────────────────────────────────────────

class _Clock:
    """A clock the test moves by hand, so a 120 s budget costs no seconds."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class _Manager:
    """Enough manager for rung 4: it knows whether its ports are open."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0

    async def connect_all(self) -> None:
        self.connect_calls += 1
        self.connected = True


class _Recorder:
    """A callable that records its calls and never does anything."""

    def __init__(self, result=None) -> None:
        self.calls: list[tuple] = []
        self._result = result

    def __call__(self, *args):
        self.calls.append(args)
        return self._result


def _append(run_dir, **record) -> None:
    record.setdefault("ts", "2026-08-19T12:00:00+00:00")
    path = run_dir / "events.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _campaign_lock(*, pid: int = 4242, host: str | None = None, log_path: str = "") -> RunLock:
    return RunLock(
        pid=pid, what="campaign:phase_map:run-7",
        started_at="2026-08-19T09:14:00+00:00",
        host=THIS_HOST if host is None else host,
        log_path=log_path,
    )


def _ladder(run_dir, *, clock, cross_host=False, lock=None, manager=None, **kwargs):
    """A ladder whose every consequential collaborator is a recorder by default."""
    kwargs.setdefault("breaker", _Recorder(result=lock))
    kwargs.setdefault("terminator", _Recorder(result=True))
    kwargs.setdefault("claimer", _Recorder(result=object()))
    return EstopLadder(
        str(run_dir) if run_dir is not None else None,
        lock=lock if lock is not None else _campaign_lock(),
        cross_host=cross_host,
        campaign=("phase_map", "run-7"),
        manager=manager,
        clock=clock,
        **kwargs,
    )


# ── 8.1 — the mode is declared before the press ──────────────────────────────

pytest.importorskip("PySide6.QtWidgets")

from softae.gui.widgets import emergency_stop as estop      # noqa: E402


class _StubDialog:
    """Stands in for the ladder dialog. Never opens, never waits."""

    class _Sig:
        def __init__(self):
            self.slots = []

        def connect(self, slot):
            self.slots.append(slot)

    def __init__(self, ladder, parent=None):
        self.ladder = ladder
        self.takeover_started = self._Sig()
        self.takeover_done = self._Sig()
        self.began = False

    def begin(self) -> bool:
        self.began = True
        return False        # never enters exec() in a test


class TestTheButtonDeclaresItsModeBeforeThePress:
    """Every assertion here is made on the **constructed** widget. A label that
    only became honest after a click would be no better than none."""

    def _button(self, qtbot, mode):
        btn = estop.EmergencyStopButton(
            object(), launch_mode=mode,
            ladder_factory=lambda: None, dialog_factory=_StubDialog)
        qtbot.addWidget(btn)
        return btn

    def test_owner_mode_button_keeps_its_label_and_names_no_campaign(self, qtbot):
        btn = self._button(qtbot, None)

        assert btn.text() == estop.LABEL_STOP
        assert btn.reachable_rungs == ()
        assert btn.toolTip() == estop.TOOLTIP_OWNER

    def test_attached_same_host_button_names_the_campaign_and_reaches_every_rung(
        self, qtbot
    ):
        lock = _campaign_lock(log_path="C:/runs/run-7")
        mode = LaunchMode(attached=True, campaign=("phase_map", "run-7"),
                          run_dir="C:/runs/run-7", holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.text() == estop.LABEL_STOP
        assert btn.reachable_rungs == (1, 2, 3, 4)
        assert "phase_map" in btn.toolTip()
        assert "never does so on a timer" in btn.toolTip()

    def test_attached_other_host_button_says_request_only_and_stops_at_rung_two(
        self, qtbot
    ):
        lock = _campaign_lock(host="OTHER-RIG-PC", log_path="C:/runs/run-7")
        mode = LaunchMode(attached=True, campaign=("phase_map", "run-7"),
                          run_dir="C:/runs/run-7", holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.text() == estop.LABEL_REQUEST_ONLY
        assert btn.reachable_rungs == (1, 2)
        assert 3 not in btn.reachable_rungs and 4 not in btn.reachable_rungs
        assert "ANOTHER MACHINE" in btn.toolTip()

    def test_attached_to_a_non_campaign_holder_offers_only_the_takeover(self, qtbot):
        lock = RunLock(pid=99, what="workflow 'ht_sweep'", host=THIS_HOST)
        mode = LaunchMode(attached=True, campaign=None, run_dir=None,
                          holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.reachable_rungs == (4,)
        assert "nothing to request" in btn.toolTip()

    def test_pressing_in_attached_mode_opens_the_ladder_instead_of_parking(
        self, qtbot, monkeypatch
    ):
        lock = _campaign_lock(log_path="C:/runs/run-7")
        mode = LaunchMode(attached=True, campaign=("phase_map", "run-7"),
                          run_dir="C:/runs/run-7", holder=lock, reason="")
        opened: list = []
        btn = estop.EmergencyStopButton(
            object(), launch_mode=mode, ladder_factory=lambda: "the-ladder",
            dialog_factory=lambda ladder, parent=None: opened.append(ladder)
            or _StubDialog(ladder, parent))
        qtbot.addWidget(btn)
        monkeypatch.setattr(
            estop, "_EStopWorker",
            lambda *a, **k: pytest.fail("an attached window must not park"))

        btn._on_stop()

        assert opened == ["the-ladder"]


# ── 8.2 rung 1 ───────────────────────────────────────────────────────────────

class TestRungOneWritesTheAbort:

    def test_start_writes_an_abort_control_request_naming_the_estop(self, tmp_path):
        clock = _Clock()
        ladder = _ladder(tmp_path, clock=clock)

        request = ladder.start()

        written = json.loads((tmp_path / "control.json").read_text(encoding="utf-8"))
        assert written["action"] == "abort"
        assert written["reason"] == "operator E-Stop"
        assert written["seq"] == request.seq
        assert ladder.state == STATE_AWAITING_ACK

    def test_start_with_no_run_directory_refuses_rather_than_writing_nowhere(
        self, tmp_path
    ):
        ladder = _ladder(None, clock=_Clock())

        with pytest.raises(RuntimeError):
            ladder.start()

    def test_a_park_already_in_the_history_does_not_satisfy_the_new_wait(
        self, tmp_path
    ):
        """A campaign that parked, recovered and carried on must not answer a
        press with a record written hours before it."""
        _append(tmp_path, type="park", reason="an earlier fault")
        _append(tmp_path, type="safe_park", ok=True)
        clock = _Clock()
        ladder = _ladder(tmp_path, clock=clock)
        request = ladder.start()
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")

        assert ladder.poll() == STATE_AWAITING_PARK
        clock.advance(T2_PARK_S - 1)
        assert ladder.poll() == STATE_AWAITING_PARK


# ── 8.2 rungs 2 and 3 — the budgets offer, they never perform ────────────────

class TestTheBudgetsOfferTheNextRung:

    def _started(self, tmp_path, clock, **kwargs):
        ladder = _ladder(tmp_path, clock=clock, **kwargs)
        request = ladder.start()
        return ladder, request

    def test_an_ack_advances_the_ladder_to_the_park_wait(self, tmp_path):
        clock = _Clock()
        ladder, request = self._started(tmp_path, clock)
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")

        assert ladder.poll() == STATE_AWAITING_PARK
        assert ladder.ack["outcome"] == "applied"

    def test_t1_expiry_offers_rung_three_rather_than_entering_it(self, tmp_path):
        clock = _Clock()
        ladder, _ = self._started(tmp_path, clock)

        clock.advance(T1_ACK_S)
        assert ladder.poll() == STATE_OFFERED_WAIT
        assert ladder.may_advance is True

        # And it stays offered, however long the timer runs.
        for _ in range(20):
            clock.advance(30.0)
            assert ladder.poll() == STATE_OFFERED_WAIT

        assert ladder.advance() == STATE_AWAITING_PARK

    def test_t2_expiry_offers_rung_four_rather_than_performing_it(self, tmp_path):
        clock = _Clock()
        ladder, request = self._started(tmp_path, clock)
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")
        ladder.poll()

        clock.advance(T2_PARK_S)
        assert ladder.poll() == STATE_OFFERED_TAKEOVER
        assert ladder.may_take_over is True

    def test_both_park_records_are_required_before_the_ladder_reports_parked(
        self, tmp_path
    ):
        clock = _Clock()
        ladder, request = self._started(tmp_path, clock)
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")
        ladder.poll()

        _append(tmp_path, type="park", reason="operator E-Stop")
        assert ladder.poll() == STATE_AWAITING_PARK      # the park is not the proof

        _append(tmp_path, type="safe_park", ok=True, commanded=["lamp off"])
        assert ladder.poll() == STATE_PARKED
        assert ladder.park_record["ok"] is True

    def test_a_late_park_withdraws_the_takeover_offer(self, tmp_path):
        """The only automatic transition that *removes* a rung: once the campaign
        has parked, an offer to kill it is an offer to kill a process that has
        already done what it was asked."""
        clock = _Clock()
        ladder, request = self._started(tmp_path, clock)
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")
        ladder.poll()
        clock.advance(T2_PARK_S)
        assert ladder.poll() == STATE_OFFERED_TAKEOVER

        _append(tmp_path, type="park", reason="operator E-Stop")
        _append(tmp_path, type="safe_park", ok=True, commanded=["lamp off"])

        assert ladder.poll() == STATE_PARKED
        assert ladder.may_take_over is False

    def test_a_park_arriving_without_an_ack_still_ends_the_ladder(self, tmp_path):
        clock = _Clock()
        ladder, _ = self._started(tmp_path, clock)
        clock.advance(T1_ACK_S)
        assert ladder.poll() == STATE_OFFERED_WAIT

        _append(tmp_path, type="park", reason="operator E-Stop")
        _append(tmp_path, type="safe_park", ok=True, commanded=["lamp off"])

        assert ladder.poll() == STATE_PARKED

    def test_a_late_ack_withdraws_the_offer_it_answers(self, tmp_path):
        clock = _Clock()
        ladder, request = self._started(tmp_path, clock)
        clock.advance(T1_ACK_S)
        assert ladder.poll() == STATE_OFFERED_WAIT

        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")
        assert ladder.poll() == STATE_AWAITING_PARK

    def test_the_wait_shows_elapsed_time_and_the_newest_event_line(self, tmp_path):
        clock = _Clock()
        ladder, _ = self._started(tmp_path, clock)
        _append(tmp_path, type="heartbeat", phase="anneal hold", phase_age_s=41)
        ladder.poll()
        clock.advance(7.0)

        assert ladder.elapsed_s == pytest.approx(7.0)
        assert ladder.budget_s == T1_ACK_S
        assert "heartbeat" in ladder.last_event_line
        assert "anneal hold" in ladder.last_event_line


# ── The property the whole step turns on ─────────────────────────────────────

class TestNoTimerPathReachesTheKill:

    def test_polling_past_both_budgets_breaks_no_lock_and_kills_nothing(
        self, tmp_path
    ):
        clock = _Clock()
        breaker, terminator, claimer = _Recorder(), _Recorder(), _Recorder()
        manager = _Manager()
        ladder = _ladder(tmp_path, clock=clock, manager=manager,
                         breaker=breaker, terminator=terminator, claimer=claimer)
        request = ladder.start()
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")

        # Ten minutes of polling — five times T2 — with nothing but the clock
        # moving.
        for _ in range(60):
            clock.advance(10.0)
            ladder.poll()

        assert ladder.state == STATE_OFFERED_TAKEOVER
        assert breaker.calls == []
        assert terminator.calls == []
        assert claimer.calls == []
        assert manager.connect_calls == 0

    def test_a_takeover_that_was_not_offered_is_refused(self, tmp_path):
        clock = _Clock()
        terminator = _Recorder()
        ladder = _ladder(tmp_path, clock=clock, terminator=terminator,
                         manager=_Manager())
        ladder.start()

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.performed is False
        assert "not offering" in result.refused
        assert terminator.calls == []

    def test_an_unconfirmed_takeover_is_refused_even_when_offered(self, tmp_path):
        clock = _Clock()
        terminator = _Recorder()
        ladder = _ladder(tmp_path, clock=clock, terminator=terminator,
                         manager=_Manager())
        request = ladder.start()
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")
        ladder.poll()
        clock.advance(T2_PARK_S)
        assert ladder.poll() == STATE_OFFERED_TAKEOVER

        result = asyncio.run(ladder.take_over(confirmed=False))

        assert result.performed is False
        assert terminator.calls == []


# ── Rung 4 — one PID, and the connect before the park ────────────────────────

def _offered(tmp_path, clock, **kwargs):
    """A ladder sitting on rung 4's offer, having got there the long way."""
    ladder = _ladder(tmp_path, clock=clock, **kwargs)
    request = ladder.start()
    _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
            outcome="applied")
    ladder.poll()
    clock.advance(T2_PARK_S)
    assert ladder.poll() == STATE_OFFERED_TAKEOVER
    return ladder


class TestRungFour:

    def test_the_kill_targets_the_locks_recorded_pid_and_nothing_else(self, tmp_path):
        clock = _Clock()
        terminator = _Recorder(result=True)
        lock = _campaign_lock(pid=31337, log_path=str(tmp_path))
        ladder = _offered(tmp_path, clock, lock=lock, terminator=terminator,
                          manager=_Manager(),
                          parker=_park_stub(SafeParkResult(commanded=["lamp off"])))

        asyncio.run(ladder.take_over(confirmed=True))

        assert terminator.calls == [(31337,)]

    def test_rung_four_connects_before_it_parks(self, tmp_path):
        clock = _Clock()
        manager = _Manager()
        seen: dict = {}

        async def parker(mgr, **kwargs):
            # What the park sees is the whole question: a park that ran first
            # would find nothing open and report success about it.
            seen["connected_when_parked"] = mgr.connected
            return SafeParkResult(commanded=["pumps halted", "lamp off"])

        ladder = _offered(tmp_path, clock, manager=manager, parker=parker,
                          lock=_campaign_lock(log_path=str(tmp_path)))

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert seen["connected_when_parked"] is True
        assert manager.connect_calls == 1
        assert result.steps.index("connect") < result.steps.index("park")
        assert result.steps.index("break_lock") < result.steps.index("terminate")

    def test_rung_four_reports_through_the_shared_park_headline(self, tmp_path):
        clock = _Clock()
        ladder = _offered(
            tmp_path, clock, manager=_Manager(),
            lock=_campaign_lock(log_path=str(tmp_path)),
            parker=_park_stub(SafeParkResult(commanded=["pumps halted"])))

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.headline() == (HEADLINE_COMMANDED, False)
        assert "pumps halted" in result.describe()

    def test_a_takeover_whose_connect_failed_is_not_reported_as_a_stop(self, tmp_path):
        """The B5/B6 defect, in its takeover form: a park across a manager that
        never opened commands nothing, and must not be headed as though it did."""
        clock = _Clock()

        class _DeadManager(_Manager):
            async def connect_all(self):
                raise OSError("COM3 is held by something else")

        ladder = _offered(
            tmp_path, clock, manager=_DeadManager(),
            lock=_campaign_lock(log_path=str(tmp_path)),
            parker=_park_stub(SafeParkResult(skipped=["lamp: not connected"])))

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.connected is False
        assert result.headline() == (HEADLINE_NOTHING, True)

    def test_a_holder_with_no_control_channel_offers_rung_four_from_the_outset(
        self, tmp_path
    ):
        lock = RunLock(pid=555, what="workflow 'ht_sweep'", host=THIS_HOST)
        ladder = _ladder(None, clock=_Clock(), lock=lock, manager=_Manager())

        assert ladder.reachable_rungs == (4,)
        assert ladder.may_take_over is True
        assert "nothing to request" in ladder.note


def _park_stub(result):
    async def parker(mgr, **kwargs):
        return result
    return parker


# ── The cross-host limit, enforced and not merely labelled ───────────────────

class TestTheCrossHostLadderCannotReachRungsThreeOrFour:

    def test_reachable_rungs_stops_at_two_for_another_host(self):
        assert reachable_rungs(run_dir="C:/runs/x", cross_host=True) == (1, 2)
        assert reachable_rungs(run_dir=None, cross_host=True) == ()

    def test_an_acknowledged_cross_host_abort_ends_the_ladder(self, tmp_path):
        clock = _Clock()
        ladder = _ladder(tmp_path, clock=clock, cross_host=True,
                         lock=_campaign_lock(host="OTHER-RIG-PC"))
        request = ladder.start()
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")

        assert ladder.poll() == STATE_ACKED_ONLY
        assert ladder.may_take_over is False

    def test_an_unacknowledged_cross_host_abort_offers_nothing(self, tmp_path):
        clock = _Clock()
        terminator = _Recorder()
        ladder = _ladder(tmp_path, clock=clock, cross_host=True,
                         terminator=terminator,
                         lock=_campaign_lock(host="OTHER-RIG-PC"))
        ladder.start()

        clock.advance(T1_ACK_S)
        assert ladder.poll() == STATE_EXHAUSTED
        assert ladder.may_advance is False
        assert ladder.may_take_over is False

        result = asyncio.run(ladder.take_over(confirmed=True))
        assert result.performed is False
        assert terminator.calls == []


# ── The dialog: it renders the ladder and owns the confirmation, nothing else ─

from softae.gui.widgets import estop_ladder_dialog as dlg    # noqa: E402


class TestTheDialogNeverActsOnItsOwnTimer:
    """The ladder enforces "no timer reaches the kill"; these assert the dialog
    is wired so it could not defeat that even by accident."""

    def _dialog(self, qtbot, ladder, *, confirm=None, schedule=None):
        widget = dlg.EstopLadderDialog(
            ladder, confirm=confirm or (lambda pid, evidence: False),
            schedule=schedule or (lambda coro: coro.close()))
        qtbot.addWidget(widget)
        return widget

    def test_ticking_past_both_budgets_only_offers_the_takeover(self, qtbot, tmp_path):
        clock = _Clock()
        breaker, terminator = _Recorder(), _Recorder()
        ladder = _ladder(tmp_path, clock=clock, manager=_Manager(),
                         breaker=breaker, terminator=terminator)
        widget = self._dialog(qtbot, ladder)
        widget.begin()
        request = ladder.request
        _append(tmp_path, type="control_ack", seq=request.seq, action="abort",
                outcome="applied")

        for _ in range(40):
            clock.advance(10.0)
            widget._tick()

        assert widget._btn_act.text() == dlg.ACT_TAKE_OVER
        assert breaker.calls == [] and terminator.calls == []

    def test_the_takeover_button_does_nothing_without_the_confirmation(
        self, qtbot, tmp_path
    ):
        clock = _Clock()
        terminator = _Recorder()
        ladder = _offered(tmp_path, clock, manager=_Manager(),
                          terminator=terminator)
        widget = self._dialog(qtbot, ladder, confirm=lambda pid, evidence: False)

        widget._on_act()

        assert terminator.calls == []
        assert ladder.state == STATE_OFFERED_TAKEOVER

    def test_the_confirmation_is_shown_the_locks_own_description(
        self, qtbot, tmp_path
    ):
        clock = _Clock()
        lock = _campaign_lock(pid=8821, log_path=str(tmp_path))
        ladder = _offered(tmp_path, clock, lock=lock, manager=_Manager())
        seen: dict = {}
        widget = self._dialog(
            qtbot, ladder,
            confirm=lambda pid, evidence: seen.update(pid=pid, evidence=evidence))

        widget._on_act()

        assert seen["pid"] == 8821
        assert "PID 8821" in seen["evidence"]
        assert "campaign:phase_map:run-7" in seen["evidence"]      # what
        assert "2026-08-19T09:14:00+00:00" in seen["evidence"]     # started_at
        assert "reuses process ids" in seen["evidence"]

    def test_the_offered_wait_button_only_advances_the_ladder(self, qtbot, tmp_path):
        clock = _Clock()
        terminator = _Recorder()
        ladder = _ladder(tmp_path, clock=clock, terminator=terminator,
                         manager=_Manager())
        widget = self._dialog(qtbot, ladder)
        widget.begin()
        clock.advance(T1_ACK_S)
        widget._tick()
        assert widget._btn_act.text() == dlg.ACT_WAIT

        widget._on_act()

        assert ladder.state == STATE_AWAITING_PARK
        assert terminator.calls == []

    def test_the_wait_display_carries_the_clock_and_the_newest_event(
        self, qtbot, tmp_path
    ):
        clock = _Clock()
        ladder = _ladder(tmp_path, clock=clock, manager=_Manager())
        widget = self._dialog(qtbot, ladder)
        widget.begin()
        _append(tmp_path, type="heartbeat", phase="anneal hold", phase_age_s=41)
        clock.advance(6.0)
        widget._tick()

        shown = widget._lbl_watch.text()
        assert "waiting 6s of 15s" in shown
        assert "anneal hold" in shown


# ── terminate_pid itself — only its refusals are exercised ───────────────────

class TestTerminatePidRefusesEverythingButOneNumber:
    """The real function, driven only into the branches that signal nothing.
    Nothing in this suite terminates a process."""

    @pytest.mark.parametrize("pid", [0, -1, -12345])
    def test_a_non_positive_pid_is_never_signalled(self, pid):
        assert terminate_pid(pid) is False

    def test_this_process_is_never_its_own_target(self):
        assert terminate_pid(os.getpid()) is False

    def test_it_takes_a_number_so_no_caller_can_pass_an_image_name(self):
        with pytest.raises((TypeError, ValueError)):
            terminate_pid("python.exe")      # type: ignore[arg-type]
