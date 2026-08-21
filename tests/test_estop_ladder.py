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

**Rung 4 is offered against a campaign and nothing else.** A ``gui:`` holder is
another softae window: it parks on close, so killing it is strictly worse than
closing it, and it publishes no event stream, so the wedged-versus-working
judgement the offer delegates to the operator has nothing behind it.

The doubles here produce the shapes **production** produces. ``connect_all``
returns ``{name: bool}`` and never raises, so a double that raises tests a branch
the real manager cannot enter; :class:`_Manager` returns the dict, and the
all-``False`` and partial cases have tests of their own.
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
    KIND_CAMPAIGN,
    KIND_GUI,
    NOTE_HOLDER_IS_A_WINDOW,
    STATE_ACKED_ONLY,
    STATE_AWAITING_ACK,
    STATE_AWAITING_PARK,
    STATE_EXHAUSTED,
    STATE_IDLE,
    STATE_OFFERED_TAKEOVER,
    STATE_OFFERED_WAIT,
    STATE_PARKED,
    T1_ACK_S,
    T2_PARK_S,
    EstopLadder,
    holder_kind,
    reachable_rungs,
    terminate_pid,
)
from softae.gui.launch_mode import LaunchMode
from softae.gui.widgets.rig_owner import campaign_identity

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


#: What ``InstrumentManager.connect_all`` returns on a rig where everything
#: opened. The *shape* is the point: a dict, never an exception.
ALL_OPENED = {"stage": True, "syringe": True, "temp": True, "potentiostat": True}


class _Manager:
    """Enough manager for rung 4, answering the way the real one answers.

    ``InstrumentManager.connect_all`` is ``async`` and returns ``{name: bool}``;
    it catches every per-instrument failure, so it **never raises**. A double
    that returned ``None`` — as this one used to — let ``connected`` be derived
    from "the coroutine finished", which is the defect these tests now pin.
    """

    def __init__(self, report: dict[str, bool] | None = None) -> None:
        self.connected = False
        self.connect_calls = 0
        self._report = ALL_OPENED if report is None else report

    async def connect_all(self) -> dict[str, bool]:
        self.connect_calls += 1
        self.connected = any(self._report.values())
        return dict(self._report)


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
    """A ladder whose every consequential collaborator is a recorder by default.

    ``campaign`` follows the lock, the way ``decide_launch_mode`` makes it
    follow: a non-campaign holder has no campaign identity, and a helper that
    handed one over would build a ladder production cannot build.
    """
    lock = lock if lock is not None else _campaign_lock()
    kwargs.setdefault("breaker", _Recorder(result=lock))
    kwargs.setdefault("terminator", _Recorder(result=True))
    kwargs.setdefault("claimer", _Recorder(result=object()))
    return EstopLadder(
        str(run_dir) if run_dir is not None else None,
        lock=lock,
        cross_host=cross_host,
        campaign=campaign_identity(lock),
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

    def test_attached_to_a_campaign_with_no_run_directory_offers_only_the_takeover(
        self, qtbot
    ):
        """The one holder that legitimately opens at rung 4: it *is* a campaign,
        so it may be unable to park itself, and nothing else can reach it."""
        lock = _campaign_lock(log_path="")
        mode = LaunchMode(attached=True, campaign=None, run_dir=None,
                          holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.holder_kind == KIND_CAMPAIGN
        assert btn.reachable_rungs == (4,)
        assert "nothing to request" in btn.toolTip()

    def test_attached_to_a_gui_holder_routes_to_that_window_and_reaches_no_rung(
        self, qtbot
    ):
        """A second GUI opened against the operator's live one. Every fact here
        is on the constructed widget: the operator must not have to press to
        learn that this button will not terminate their own window."""
        lock = RunLock(pid=23584, what="gui:desktop", host=THIS_HOST)
        mode = LaunchMode(attached=True, campaign=None, run_dir=None,
                          holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.holder_kind == KIND_GUI
        assert btn.reachable_rungs == ()
        assert btn.text() == estop.LABEL_OTHER_WINDOW
        assert btn.toolTip() == estop.TOOLTIP_GUI_HOLDER
        # It says what to do, not merely what it refuses.
        assert "press its E-Stop, or close it" in btn.toolTip()
        assert "will not terminate" in btn.toolTip()

    def test_attached_to_a_workflow_holder_reaches_no_rung_and_says_where_to_stop_it(
        self, qtbot
    ):
        """The executor's ``workflow '<name>'`` predates the kind grammar and has
        no kind at all. It publishes no event stream either, so the judgement
        rung 4 asks of the operator has nothing behind it."""
        lock = RunLock(pid=99, what="workflow 'ht_sweep'", host=THIS_HOST)
        mode = LaunchMode(attached=True, campaign=None, run_dir=None,
                          holder=lock, reason="")
        btn = self._button(qtbot, mode)

        assert btn.holder_kind == ""
        assert btn.reachable_rungs == ()
        assert btn.text() == estop.LABEL_UNREACHABLE
        assert "Stop it where it runs" in btn.toolTip()

    def test_attached_to_an_unreadable_lock_reaches_no_rung(self, qtbot):
        """``decide_launch_mode`` attaches with ``holder=None`` when the lock
        cannot be read. Unknown holder, unknown kind, no kill."""
        mode = LaunchMode(attached=True, campaign=None, run_dir=None,
                          holder=None, reason="")
        btn = self._button(qtbot, mode)

        assert btn.holder_kind == ""
        assert btn.reachable_rungs == ()
        assert btn.text() == estop.LABEL_UNREACHABLE

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

    def test_a_takeover_whose_connect_raised_is_not_reported_as_a_stop(self, tmp_path):
        """The B5/B6 defect, in its takeover form: a park across a manager that
        never opened commands nothing, and must not be headed as though it did.

        Kept for the shape a *connector* injection can produce; the production
        manager cannot raise, which is what the two tests below cover."""
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
        assert result.sessions is None
        assert "NOT REPORTED" in result.describe()
        assert result.headline() == (HEADLINE_NOTHING, True)

    def test_a_takeover_whose_connect_opened_nothing_is_not_reported_as_a_stop(
        self, tmp_path
    ):
        """**The shape production actually produces.** ``connect_all`` never
        raises; on a rig where Windows has not yet released the handles the
        terminated campaign held it returns all-``False`` and returns
        *successfully*. ``connected`` used to be derived from that success."""
        clock = _Clock()
        manager = _Manager({"stage": False, "syringe": False, "temp": False})

        ladder = _offered(
            tmp_path, clock, manager=manager,
            lock=_campaign_lock(log_path=str(tmp_path)),
            parker=_park_stub(SafeParkResult(skipped=["lamp: not connected"])))

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert manager.connect_calls == 1
        assert result.connected is False
        assert result.sessions == {"stage": False, "syringe": False, "temp": False}
        assert "instrument sessions opened: NONE — 0 of 3 opened" in result.describe()
        assert "instrument sessions opened: yes" not in result.describe()
        assert result.headline() == (HEADLINE_NOTHING, True)

    def test_a_takeover_whose_connect_partly_opened_reports_which_ports_failed(
        self, tmp_path
    ):
        """The interesting middle: some handles came back after the kill and some
        did not, so the park reaches only half the rig. The park's own headline
        cannot say this — it reports what it commanded, not what never opened."""
        clock = _Clock()
        manager = _Manager({"stage": True, "syringe": True,
                            "piezo": False, "potentiostat": False})

        ladder = _offered(
            tmp_path, clock, manager=manager,
            lock=_campaign_lock(log_path=str(tmp_path)),
            parker=_park_stub(SafeParkResult(commanded=["pumps halted"])))

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.connected is True
        assert result.sessions["piezo"] is False
        described = result.describe()
        assert "instrument sessions opened: 2 of 4" in described
        assert "FAILED: piezo, potentiostat" in described

    def test_a_campaign_with_no_run_directory_offers_rung_four_from_the_outset(
        self, tmp_path
    ):
        lock = _campaign_lock(pid=555, log_path="")
        ladder = _ladder(None, clock=_Clock(), lock=lock, manager=_Manager())

        assert ladder.holder_kind == KIND_CAMPAIGN
        assert ladder.reachable_rungs == (4,)
        assert ladder.may_take_over is True
        assert "nothing to request" in ladder.note


def _park_stub(result):
    async def parker(mgr, **kwargs):
        return result
    return parker


# ── The kill is offered against a campaign, and against nothing else ─────────

class TestOnlyACampaignCanBeTerminated:
    """Rung 4's two justifications are both about campaigns: a campaign may be
    unable to park *itself*, and its ``events.jsonl`` is what the operator is
    asked to judge wedged-from-working by. Neither holds for a window, which
    parks on close and publishes nothing — so no path may reach the kill."""

    def _gui_lock(self, pid: int = 23584) -> RunLock:
        return RunLock(pid=pid, what="gui:desktop",
                       started_at="2026-08-20T08:00:00+00:00", host=THIS_HOST)

    def test_holder_kind_reads_the_shipped_what_grammar(self):
        assert holder_kind(_campaign_lock()) == KIND_CAMPAIGN
        assert holder_kind(self._gui_lock()) == KIND_GUI
        assert holder_kind(RunLock(pid=1, what="tool:env-hold:run-3")) == "tool"

    def test_holder_kind_of_a_what_without_the_grammar_is_unknown(self):
        """The executor's lock predates ``<kind>:<name>:<run_id>``, and a kind
        invented after this module will be unknown too. Unknown is not a kind
        that may be killed — it is the one about which nothing is known."""
        assert holder_kind(RunLock(pid=1, what="workflow 'ht_sweep'")) == ""
        assert holder_kind(RunLock(pid=1, what="")) == ""
        assert holder_kind(None) == ""

    @pytest.mark.parametrize("kind", [KIND_GUI, "tool", "", "some_future_thing"])
    @pytest.mark.parametrize("run_dir", [None, "C:/runs/x"])
    @pytest.mark.parametrize("cross_host", [False, True])
    def test_reachable_rungs_offers_nothing_to_a_non_campaign_holder(
        self, kind, run_dir, cross_host
    ):
        assert reachable_rungs(run_dir=run_dir, cross_host=cross_host,
                               kind=kind) == ()

    def test_reachable_rungs_requires_the_kind_so_no_caller_can_omit_it(self):
        """A default in either direction is a rule a caller can forget, and
        forgetting it one way offers a kill against an unclassified holder."""
        with pytest.raises(TypeError):
            reachable_rungs(run_dir="C:/runs/x", cross_host=False)  # type: ignore[call-arg]

    def test_a_gui_holder_cannot_reach_the_takeover_by_any_path(self, tmp_path):
        """Driven past both budgets, polled sixty times, and asked directly.
        ``poll`` returns early from IDLE, so the clock is not even the risk —
        the risk is a future edit that removes that early return, which is why
        the terminator is asserted untouched rather than only the state."""
        clock = _Clock()
        breaker, terminator = _Recorder(), _Recorder()
        ladder = _ladder(None, clock=clock, lock=self._gui_lock(),
                         manager=_Manager(), breaker=breaker,
                         terminator=terminator)

        assert ladder.holder_kind == KIND_GUI
        assert ladder.reachable_rungs == ()
        assert ladder.may_take_over is False
        assert ladder.may_advance is False

        for _ in range(60):
            clock.advance(10.0)
            assert ladder.poll() == STATE_IDLE

        assert ladder.may_take_over is False
        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.performed is False
        assert terminator.calls == [] and breaker.calls == []

    def test_a_gui_holder_with_a_run_directory_still_cannot_reach_the_takeover(
        self, tmp_path
    ):
        """The kind gate is not a restatement of "no run directory". A window
        that somehow published one is still a window."""
        clock = _Clock()
        terminator = _Recorder()
        ladder = _ladder(tmp_path, clock=clock, lock=self._gui_lock(),
                         manager=_Manager(), terminator=terminator)

        assert ladder.reachable_rungs == ()
        assert ladder.may_take_over is False
        assert asyncio.run(ladder.take_over(confirmed=True)).performed is False
        assert terminator.calls == []

    def test_a_gui_holders_note_names_the_act_that_works_before_any_press(
        self, tmp_path
    ):
        ladder = _ladder(None, clock=_Clock(), lock=self._gui_lock(),
                         manager=_Manager())

        assert ladder.state == STATE_IDLE
        assert ladder.note == NOTE_HOLDER_IS_A_WINDOW
        assert "press its E-Stop, or close it" in ladder.note
        assert "no closeEvent runs" in ladder.note

    def test_a_gui_holders_refusal_routes_to_that_window(self, tmp_path):
        ladder = _ladder(None, clock=_Clock(), lock=self._gui_lock(),
                         manager=_Manager())

        refused = asyncio.run(ladder.take_over(confirmed=True)).refused

        assert "another softae window" in refused
        assert "press its E-Stop, or close it" in refused
        assert "Nothing was touched" in refused

    def test_an_unknown_kind_is_refused_and_told_where_to_stop_it(self, tmp_path):
        """The conservative direction: this module has never been taught how a
        future holder parks, so it does not offer to kill one."""
        lock = RunLock(pid=777, what="tool:env-hold:run-3", host=THIS_HOST)
        ladder = _ladder(None, clock=_Clock(), lock=lock, manager=_Manager())

        assert ladder.reachable_rungs == ()
        assert ladder.may_take_over is False
        refused = asyncio.run(ladder.take_over(confirmed=True)).refused
        assert "not a campaign" in refused
        assert "Stop it where it runs" in refused

    def test_a_campaign_holder_still_reaches_the_takeover(self, tmp_path):
        """The gate must not have closed the door it exists to leave open."""
        clock = _Clock()
        terminator = _Recorder(result=True)
        ladder = _offered(tmp_path, clock, terminator=terminator,
                          manager=_Manager(),
                          lock=_campaign_lock(pid=31337, log_path=str(tmp_path)),
                          parker=_park_stub(SafeParkResult(commanded=["lamp off"])))

        assert ladder.holder_kind == KIND_CAMPAIGN
        assert ladder.reachable_rungs == (1, 2, 3, 4)

        result = asyncio.run(ladder.take_over(confirmed=True))

        assert result.performed is True
        assert terminator.calls == [(31337,)]


# ── The cross-host limit, enforced and not merely labelled ───────────────────

class TestTheCrossHostLadderCannotReachRungsThreeOrFour:

    def test_reachable_rungs_stops_at_two_for_another_host(self):
        assert reachable_rungs(run_dir="C:/runs/x", cross_host=True,
                               kind=KIND_CAMPAIGN) == (1, 2)
        assert reachable_rungs(run_dir=None, cross_host=True,
                               kind=KIND_CAMPAIGN) == ()

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

    def test_a_gui_holder_gets_no_act_button_and_the_routing_note(
        self, qtbot, tmp_path
    ):
        """The dialog is the surface the operator meets. There is no red button
        to reach, and the note tells them where the working E-Stop is."""
        lock = RunLock(pid=23584, what="gui:desktop", host=THIS_HOST)
        ladder = _ladder(None, clock=_Clock(), lock=lock, manager=_Manager())
        widget = self._dialog(qtbot, ladder)

        assert widget.begin() is True          # nothing to write, and it says so
        assert widget._btn_act.isHidden() is True
        assert widget._btn_act.text() != dlg.ACT_TAKE_OVER
        assert widget._lbl_note.text() == NOTE_HOLDER_IS_A_WINDOW
        assert "Another softae window" in widget._lbl_head.text()
        assert "none" in widget._lbl_head.text()

    def test_a_gui_holders_act_button_terminates_nothing_if_it_is_pressed_anyway(
        self, qtbot, tmp_path
    ):
        """The button is hidden, not merely unstyled — but a hidden widget can
        still be clicked programmatically, and the ladder is what refuses."""
        clock = _Clock()
        terminator = _Recorder()
        lock = RunLock(pid=23584, what="gui:desktop", host=THIS_HOST)
        ladder = _ladder(None, clock=clock, lock=lock, manager=_Manager(),
                         terminator=terminator)
        widget = self._dialog(qtbot, ladder,
                              confirm=lambda pid, evidence: True)
        widget.begin()

        widget._on_act()
        for _ in range(20):
            clock.advance(30.0)
            widget._tick()
        widget._on_act()

        assert terminator.calls == []
        assert ladder.state == STATE_IDLE

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
