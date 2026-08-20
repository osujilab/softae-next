"""Purge actuation and the idle-rest state (P8).

This is the only mechanism in the system that moves hardware with nobody asking
it to, so the tests concentrate on the cases where it must **decline**: a parked
rig, a raised head, a disabled actuate flag, a failed pump.
"""

from __future__ import annotations

import pytest

from softae.core.purge import PurgeScheduler, PurgeSettings
from softae.core.purge_runner import (
    PURGE_OWNER,
    IdleRestState,
    PurgeRunner,
    enter_idle_rest,
    leave_idle_rest,
)
from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Syringe:
    def __init__(self, *, head_up: bool = False, fail_pumps=()) -> None:
        self._head_up = head_up
        self.calls: list[tuple[int, float]] = []
        self._fail = set(fail_pumps)

    def is_head_up(self) -> bool:
        return self._head_up

    def head_descend(self) -> None:
        self._head_up = False

    def head_retract(self) -> None:
        self._head_up = True

    def single_pump(self, *, res_vol, ID, rate, dispense_vol) -> None:
        if ID in self._fail:
            raise RuntimeError("pump jammed")
        self.calls.append((int(ID), float(dispense_vol)))


#: The calibrated flush basin in this repo's config. Wick sits 75 mm away in Y
#: and electrode 1 is 93.5 mm away in X, so the 2 mm tolerance cannot confuse them.
FLUSH = (-50.0, 50.0)
WELL = (43.5, 50.0)         # electrode 1 — head-down here means mid-cast


class _Stage:
    def __init__(self, pos=FLUSH) -> None:
        self.moves: list[tuple[float, float]] = []
        self._pos = (float(pos[0]), float(pos[1]))

    def live_position(self):
        return self._pos

    def move_to(self, x, y, *, head_may_be_down: bool = False) -> None:
        self.moves.append((float(x), float(y)))
        self._pos = (float(x), float(y))


class _Manager:
    def __init__(self, syringe=None, stage=None) -> None:
        self._items = {"syringe": syringe or _Syringe(), "stage": stage or _Stage()}

    def get(self, name):
        if name not in self._items:
            raise KeyError(name)
        return self._items[name]


class _Waste:
    def __init__(self) -> None:
        self.total = 0.0

    def add(self, volume_uL):
        self.total += float(volume_uL)


def _settings(**kw) -> PurgeSettings:
    base = dict(enabled=True, actuate=True, interval_s=900.0,
                particulate_uL=20.0, other_uL=10.0,
                particulate_pumps=(1,), pumps=(0, 1, 2))   # pump 1 = particulate
    base.update(kw)
    return PurgeSettings(**base)


def _due_runner(manager, *, settings=None, waste=None, park=None, at_rest=True,
                activity=None):
    clock = _Clock()
    sched = PurgeScheduler(settings or _settings(), now=clock)
    clock.t = 1000.0                      # past the interval → due
    runner = PurgeRunner(manager, sched, waste_ledger=waste, park_reason=park,
                         idle_rest=IdleRestState(at_rest), activity=activity,
                         flush_xy=FLUSH)
    return runner, sched, clock


# ── Nothing due ──────────────────────────────────────────────────────────────

def test_nothing_happens_when_no_purge_is_due():
    syringe = _Syringe()
    clock = _Clock()
    sched = PurgeScheduler(_settings(), now=clock)
    outcome = PurgeRunner(_Manager(syringe), sched).maybe_purge()

    assert not outcome.performed
    assert syringe.calls == []


# ── The default posture: scheduled but inert ─────────────────────────────────

class TestNotActuating:
    def test_a_due_purge_is_logged_but_not_dispensed(self):
        """Shipped posture — the schedule runs, nothing moves."""
        syringe = _Syringe()
        runner, _, _ = _due_runner(_Manager(syringe),
                                   settings=_settings(actuate=False))

        outcome = runner.maybe_purge()

        assert outcome.dry_run
        assert not outcome.performed
        assert syringe.calls == []
        assert "not actuating" in outcome.summary()

    def test_the_dry_run_reports_what_it_would_have_dispensed(self):
        runner, _, _ = _due_runner(_Manager(), settings=_settings(actuate=False))
        assert runner.maybe_purge().volumes_uL == {0: 10.0, 1: 20.0, 2: 10.0}

    def test_the_dry_run_resets_the_timer_so_the_log_shows_a_cadence(self):
        runner, sched, _ = _due_runner(_Manager(),
                                       settings=_settings(actuate=False))
        runner.maybe_purge()
        assert sched.due() is None


# ── Refusals ─────────────────────────────────────────────────────────────────

class TestRefusals:
    def test_a_parked_rig_is_never_purged(self):
        """A parked rig was deliberately made safe; do not actuate into it."""
        syringe = _Syringe()
        runner, _, _ = _due_runner(
            _Manager(syringe), park=lambda: "reservoir depleted")

        outcome = runner.maybe_purge()

        assert outcome.skipped_reason and "parked" in outcome.skipped_reason
        assert syringe.calls == []

    def test_head_down_over_a_well_is_never_purged(self):
        """THE hazard, and the reason head position alone is not enough.

        The head is down at the flush basin AND down mid-cast. Dispensing here
        would put purge volume straight into the sample being cast.
        """
        syringe = _Syringe(head_up=False)
        runner, _, _ = _due_runner(_Manager(syringe, _Stage(pos=WELL)))

        outcome = runner.maybe_purge()

        assert outcome.skipped_reason
        assert "away from the flush basin" in outcome.skipped_reason
        assert syringe.calls == []

    def test_head_down_over_a_well_does_not_move_the_stage_either(self):
        """Travelling out from under a lowered head would drag the tip."""
        stage = _Stage(pos=WELL)
        runner, _, _ = _due_runner(_Manager(_Syringe(head_up=False), stage))
        runner.maybe_purge()
        assert stage.moves == []

    def test_a_skip_does_not_reset_the_timer(self):
        """The line is still stagnating; hiding that behind a reset is worse."""
        runner, sched, _ = _due_runner(
            _Manager(_Syringe(head_up=False), _Stage(pos=WELL)))
        runner.maybe_purge()
        assert sched.due() is not None

    def test_an_unreadable_pose_is_refused(self):
        """"I could not tell" and "it is unsafe" must lead to the same action."""
        class _Mute:
            def is_head_up(self):
                raise RuntimeError("comms down")

        runner, _, _ = _due_runner(_Manager(_Mute()))
        outcome = runner.maybe_purge()
        assert outcome.skipped_reason and "could not be read" in outcome.skipped_reason

    def test_a_busy_rig_is_not_purged(self):
        """Ownership, not head position, is what says 'something else is running'."""
        from softae.core.rig_activity import RigActivity

        activity = RigActivity()
        activity.acquire("ht-dropcast")
        syringe = _Syringe()
        runner, _, _ = _due_runner(_Manager(syringe), activity=activity)

        outcome = runner.maybe_purge()

        assert outcome.skipped_reason and "in use" in outcome.skipped_reason
        assert "ht-dropcast" in outcome.skipped_reason
        assert syringe.calls == []

    def test_releasing_the_claim_lets_the_purge_through(self):
        from softae.core.rig_activity import RigActivity

        activity = RigActivity()
        activity.acquire("ht-dropcast")
        activity.release("ht-dropcast")
        runner, _, _ = _due_runner(_Manager(), activity=activity)

        assert runner.maybe_purge().performed

    def test_a_missing_syringe_is_reported_not_raised(self):
        class _Empty:
            def get(self, name):
                raise KeyError(name)

        runner, _, _ = _due_runner(_Empty())
        assert runner.maybe_purge().skipped_reason is not None


# ── Positioning: "retracted, wherever" ───────────────────────────────────────

class TestPositioning:
    """A raised head is no longer a refusal — it is a short trip to the basin."""

    def test_a_raised_head_travels_to_the_basin_and_purges(self):
        syringe = _Syringe(head_up=True)
        stage = _Stage(pos=WELL)
        runner, _, _ = _due_runner(_Manager(syringe, stage))

        outcome = runner.maybe_purge()

        assert outcome.performed
        assert stage.moves == [FLUSH]
        assert syringe.is_head_up() is False       # lowered into the basin
        assert dict(syringe.calls) == {0: 10.0, 1: 20.0, 2: 10.0}

    def test_already_at_the_basin_costs_no_motion(self):
        """Idle rest, a precondition flush, or an anneal parked there."""
        stage = _Stage(pos=FLUSH)
        runner, _, _ = _due_runner(_Manager(_Syringe(head_up=False), stage))

        assert runner.maybe_purge().performed
        assert stage.moves == []

    def test_a_caller_that_forbids_motion_gets_a_skip_not_a_trip(self):
        syringe = _Syringe(head_up=True)
        stage = _Stage(pos=WELL)
        runner, _, _ = _due_runner(_Manager(syringe, stage))

        outcome = runner.maybe_purge(allow_positioning=False)

        assert outcome.skipped_reason and "repositioning" in outcome.skipped_reason
        assert stage.moves == []
        assert syringe.calls == []

    def test_a_failed_trip_does_not_reset_the_timer(self):
        """The line never got purged, so it must come due again."""
        class _BadStage(_Stage):
            def move_to(self, x, y, *, head_may_be_down=False):
                raise RuntimeError("stage timeout")

        runner, sched, _ = _due_runner(
            _Manager(_Syringe(head_up=True), _BadStage(pos=WELL)))

        outcome = runner.maybe_purge()

        assert outcome.skipped_reason and "flush basin" in outcome.skipped_reason
        assert sched.due() is not None


# ── Actuating ────────────────────────────────────────────────────────────────

class TestActuating:
    def test_all_lines_are_purged_with_their_own_volumes(self):
        syringe = _Syringe()
        runner, _, _ = _due_runner(_Manager(syringe))

        outcome = runner.maybe_purge()

        assert outcome.performed
        assert dict(syringe.calls) == {0: 10.0, 1: 20.0, 2: 10.0}
        assert outcome.total_uL == 40.0

    def test_purged_volume_is_recorded_as_waste(self):
        waste = _Waste()
        runner, _, _ = _due_runner(_Manager(), waste=waste)
        runner.maybe_purge()
        assert waste.total == 40.0

    def test_a_successful_purge_resets_the_timer(self):
        runner, sched, _ = _due_runner(_Manager())
        runner.maybe_purge()
        assert sched.due() is None

    def test_a_failed_pump_does_not_stop_the_others(self):
        """Pump 1 is the particulate line — the worst one to lose, and the
        others must still be purged rather than the whole cycle abandoned."""
        syringe = _Syringe(fail_pumps=(1,))
        runner, _, _ = _due_runner(_Manager(syringe))

        outcome = runner.maybe_purge()

        assert dict(syringe.calls) == {0: 10.0, 2: 10.0}
        assert outcome.errors and "pump 1" in outcome.errors[0]

    def test_a_failed_line_comes_due_again_immediately(self):
        """It is still stagnating — resetting its timer would hide that."""
        runner, sched, clock = _due_runner(
            _Manager(_Syringe(fail_pumps=(1,))))
        runner.maybe_purge()
        assert sched.seconds_since(1) > sched.settings.interval_s

    def test_partial_failure_still_records_the_fluid_that_moved(self):
        waste = _Waste()
        runner, _, _ = _due_runner(
            _Manager(_Syringe(fail_pumps=(1,))), waste=waste)
        runner.maybe_purge()
        assert waste.total == 20.0        # 10 + 10; the 20 µL line jammed


# ── Idle rest ────────────────────────────────────────────────────────────────

class TestIdleRest:
    def test_entering_moves_to_flush_and_lowers_the_head(self):
        syringe, stage = _Syringe(head_up=True), _Stage()

        result = enter_idle_rest(_Manager(syringe, stage), flush_xy=(-50.0, 50.0))

        assert result.entered
        assert stage.moves == [(-50.0, 50.0)]
        assert syringe.is_head_up() is False

    def test_a_parked_rig_is_never_put_into_idle_rest(self):
        """Lowering the head would erase the visible sign something went wrong."""
        syringe, stage = _Syringe(head_up=True), _Stage()

        result = enter_idle_rest(
            _Manager(syringe, stage), park_reason=lambda: "hard fault",
            flush_xy=(-50.0, 50.0))

        assert not result.entered
        assert "parked" in result.reason
        assert stage.moves == []
        assert syringe.is_head_up() is True     # left exactly as park left it

    def test_a_failed_move_does_not_lower_the_head(self):
        """Head down anywhere but the flush station casts into whatever is there."""
        class _BadStage:
            def move_to(self, x, y):
                raise RuntimeError("stage timeout")

        syringe = _Syringe(head_up=True)
        result = enter_idle_rest(
            _Manager(syringe, _BadStage()), flush_xy=(-50.0, 50.0))

        assert not result.entered
        assert syringe.is_head_up() is True

    def test_leaving_retracts_the_head(self):
        syringe = _Syringe(head_up=False)
        assert leave_idle_rest(_Manager(syringe)) is True
        assert syringe.is_head_up() is True

    def test_the_flag_tracks_the_round_trip(self):
        state = IdleRestState()
        manager = _Manager(_Syringe(head_up=True), _Stage())

        enter_idle_rest(manager, state=state, flush_xy=(-50.0, 50.0))
        assert state.at_rest is True

        leave_idle_rest(manager, state=state)
        assert state.at_rest is False

    def test_a_failed_entry_does_not_mark_the_rig_at_rest(self):
        """Otherwise a timer would purge at a position the stage never reached."""
        class _BadStage:
            def move_to(self, x, y):
                raise RuntimeError("stage timeout")

        state = IdleRestState()
        enter_idle_rest(_Manager(_Syringe(head_up=True), _BadStage()),
                        state=state, flush_xy=(-50.0, 50.0))
        assert state.at_rest is False

    def test_a_failed_retract_still_clears_the_flag(self):
        """A stale 'at rest' belief is the dangerous direction to fail in."""
        class _Stuck:
            def head_retract(self):
                raise RuntimeError("stuck")

        class _M:
            def get(self, name):
                return _Stuck()

        state = IdleRestState(True)
        assert leave_idle_rest(_M(), state=state) is False
        assert state.at_rest is False

    def test_a_parked_rig_refusal_leaves_the_flag_alone(self):
        state = IdleRestState()
        enter_idle_rest(_Manager(_Syringe(head_up=True), _Stage()),
                        park_reason=lambda: "hard fault", state=state,
                        flush_xy=(-50.0, 50.0))
        assert state.at_rest is False

    def test_leaving_reports_failure_rather_than_raising(self):
        class _Stuck:
            def head_retract(self):
                raise RuntimeError("stuck")

        class _M:
            def get(self, name):
                return _Stuck()

        assert leave_idle_rest(_M()) is False


# ── Settings split ───────────────────────────────────────────────────────────

class TestActuateFlag:
    def test_actuate_defaults_off(self):
        assert PurgeSettings().actuate is False

    def test_consumption_is_billed_even_when_not_actuating(self):
        """Otherwise the runway would be flattered by nothing dispensing yet."""
        assert _settings(actuate=False).total_uL_per_day() == pytest.approx(3840.0)

    def test_describe_marks_the_inert_state(self):
        assert "not actuating" in _settings(actuate=False).describe()
        assert "not actuating" not in _settings(actuate=True).describe()


# ── Choke-point observation ──────────────────────────────────────────────────

class TestDispenseObservation:
    """Every dispense must reset that line's timer.

    Without this the harness purges at the full idle rate *during* an active
    campaign, paying the whole consumption bill for lines that had just moved.
    """

    def _pump(self):
        from softae.drivers.contracts import ParallelSyringeMixin

        class _P(ParallelSyringeMixin):
            name = "syringe"
            _max_rate = 10_000.0
            _min_rate = 0.0

        return _P()

    def test_a_dispense_resets_that_pumps_timer(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        pump = self._pump()
        pump.purge_scheduler = sched

        clock.t = 1000.0
        assert sched.due() is not None
        pump._validate_single_pump(res_vol=1000, rate=200.0,
                                   dispense_vol=5.0, pump_id=0)
        assert sched.seconds_since(0) == 0.0

    def test_only_the_dispensing_pump_is_credited(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        pump = self._pump()
        pump.purge_scheduler = sched

        clock.t = 1000.0
        pump._validate_single_pump(res_vol=1000, rate=200.0,
                                   dispense_vol=5.0, pump_id=0)
        assert sched.seconds_since(0) == 0.0
        assert sched.seconds_since(1) == 1000.0     # still stagnating

    def test_a_refused_dispense_does_not_reset_the_timer(self):
        """No fluid moved, so the line is still stagnating."""
        from softae.errors import SafetyError

        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        pump = self._pump()
        pump.purge_scheduler = sched
        clock.t = 1000.0

        with pytest.raises(SafetyError):
            pump._validate_single_pump(res_vol=1000, rate=99_999.0,
                                       dispense_vol=5.0, pump_id=0)
        assert sched.seconds_since(0) == 1000.0

    def test_no_scheduler_attached_is_not_an_error(self):
        self._pump()._validate_single_pump(res_vol=1000, rate=200.0,
                                           dispense_vol=5.0, pump_id=0)

    def test_a_zero_volume_pump_is_not_credited(self):
        """The case that matters most, end-to-end through the real driver.

        A BO run that zeroes the particulate component never moves that line —
        which is exactly when it most needs purging. The no-op short-circuits
        before validation on both drivers, so the timer must keep running.
        """
        from softae.drivers.mock_syringe import MockSyringe

        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        syringe = MockSyringe(name="syringe")
        syringe.purge_scheduler = sched
        clock.t = 1000.0

        syringe.single_pump(res_vol=1000, ID=0, rate=200.0, dispense_vol=0.0)
        syringe.single_pump(res_vol=1000, ID=1, rate=200.0, dispense_vol=5.0)

        assert sched.seconds_since(0) == 1000.0    # zeroed → still stagnating
        assert sched.seconds_since(1) == 0.0       # actually moved


# ── Deferral, not cancellation ───────────────────────────────────────────────

class TestDeferralNotCancellation:
    """Every refusal defers; nothing is ever dropped.

    Structural, not merely intended: `due()` is derived from the per-pump
    timers, so there is no queued event that could be lost.
    """

    def test_the_same_purge_is_still_owed_on_the_next_tick(self):
        stage = _Stage(pos=WELL)
        runner, sched, _ = _due_runner(_Manager(_Syringe(head_up=False), stage))

        first = runner.maybe_purge()
        assert first.skipped_reason
        assert sched.due() is not None                 # still owed

        # The obstruction clears; the very next attempt goes through.
        stage._pos = FLUSH
        assert runner.maybe_purge().performed

    def test_deferral_makes_the_purge_more_overdue_not_less(self):
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        runner = PurgeRunner(_Manager(_Syringe(head_up=False), _Stage(pos=WELL)),
                             sched, idle_rest=IdleRestState(True), flush_xy=FLUSH)

        clock.t = 1000.0
        first = sched.due().overdue_s
        runner.maybe_purge()
        clock.t = 2000.0
        assert sched.due().overdue_s > first

    def test_a_busy_rig_defers_every_tick_then_purges_when_free(self):
        from softae.core.rig_activity import RigActivity

        activity = RigActivity()
        activity.acquire("campaign:long")
        syringe = _Syringe()
        runner, sched, _ = _due_runner(_Manager(syringe), activity=activity)

        for _ in range(5):
            assert runner.maybe_purge().skipped_reason
        assert syringe.calls == []

        activity.release("campaign:long")
        assert runner.maybe_purge().performed

    def test_a_long_deferral_raises_an_alert(self):
        """Silent deferral is the one failure mode that looks like success."""
        alerts: list = []
        clock = _Clock()
        sched = PurgeScheduler(_settings(), now=clock)
        runner = PurgeRunner(_Manager(_Syringe(head_up=False), _Stage(pos=WELL)),
                             sched, idle_rest=IdleRestState(True), flush_xy=FLUSH,
                             defer_alert_after_s=3600.0)
        clock.t = 900.0 + 3700.0        # interval + past the escalation threshold

        import softae.core.alerts as alerts_mod
        orig = alerts_mod.raise_alert
        alerts_mod.raise_alert = lambda alert, **kw: alerts.append(alert)
        try:
            runner.maybe_purge()
            runner.maybe_purge()        # second tick must not re-alert
        finally:
            alerts_mod.raise_alert = orig

        assert len(alerts) == 1
        assert "deferred" in alerts[0].message
        assert alerts[0].kind == "purge"

    def test_a_short_deferral_does_not_alert(self):
        alerts: list = []
        runner, _, _ = _due_runner(
            _Manager(_Syringe(head_up=False), _Stage(pos=WELL)))

        import softae.core.alerts as alerts_mod
        orig = alerts_mod.raise_alert
        alerts_mod.raise_alert = lambda alert, **kw: alerts.append(alert)
        try:
            runner.maybe_purge()
        finally:
            alerts_mod.raise_alert = orig

        assert alerts == []


# ── The in-run (anneal) purge ────────────────────────────────────────────────

class TestInRunPurge:
    """The only path that can purge during a run.

    The background timer defers whenever the rig is claimed, and a campaign
    holds its claim for the entire run — so without this, purging protects the
    idle rig only, and a multi-hour anneal is exactly when the particulate line
    stagnates.
    """

    def test_the_hook_purges_despite_the_run_holding_the_claim(self):
        from softae.core.rig_activity import RigActivity

        activity = RigActivity()
        activity.acquire("campaign:demo")          # the run owns the rig
        syringe = _Syringe(head_up=False)          # bracketed: down at the basin
        runner, _, _ = _due_runner(_Manager(syringe, _Stage(pos=FLUSH)),
                                   activity=activity)

        outcome = runner.maybe_purge(context="anneal", allow_positioning=False,
                                     owns_rig=True)

        assert outcome.performed
        assert dict(syringe.calls) == {0: 10.0, 1: 20.0, 2: 10.0}

    def test_the_hook_costs_no_stage_motion(self):
        """The anneal bracket already parked the tip where the purge happens."""
        stage = _Stage(pos=FLUSH)
        runner, _, _ = _due_runner(_Manager(_Syringe(head_up=False), stage))
        runner.maybe_purge(context="anneal", allow_positioning=False,
                           owns_rig=True)
        assert stage.moves == []

    def test_the_hook_never_moves_the_rig_mid_anneal(self):
        """If the rig is somehow not at the basin, skip — do not disturb it."""
        stage = _Stage(pos=WELL)
        syringe = _Syringe(head_up=True)
        runner, _, _ = _due_runner(_Manager(syringe, stage))

        outcome = runner.maybe_purge(context="anneal", allow_positioning=False,
                                     owns_rig=True)

        assert outcome.skipped_reason
        assert stage.moves == []
        assert syringe.calls == []

    def test_owning_the_rig_does_not_override_a_park(self):
        """A park outranks everything; being the claim holder is no exemption."""
        syringe = _Syringe(head_up=False)
        runner, _, _ = _due_runner(_Manager(syringe),
                                   park=lambda: "thermal fault")

        outcome = runner.maybe_purge(context="anneal", owns_rig=True,
                                     allow_positioning=False)

        assert outcome.skipped_reason and "parked" in outcome.skipped_reason
        assert syringe.calls == []

    def test_the_anneal_is_just_a_tagged_step_now(self):
        """No driver cooperation: the anneal is one purge window among many.

        It was briefly wired through ``monitored_hold``'s watchdog poll, which
        only worked because a poll loop happened to exist there for thermal
        reasons. Executor-driven windows cover every shape of dead time, so the
        temperature driver no longer knows purging exists.
        """
        from softae.drivers import contracts

        assert not hasattr(contracts, "attach_anneal_purge_hook")
        import inspect

        assert "on_poll" not in inspect.signature(contracts.monitored_hold).parameters

    def test_an_in_run_purge_restores_the_pose_it_found(self):
        """Idle rest leaves the head DOWN; the next step would trip the guard.

        Both precondition_flush and single_drop_simul open with a bare move_to
        and no retract, so a purge that lowered the head mid-run would fail the
        very next channel.
        """
        syringe = _Syringe(head_up=True)
        stage = _Stage(pos=WELL)
        runner, _, _ = _due_runner(_Manager(syringe, stage))

        outcome = runner.maybe_purge(context="step:measure", owns_rig=True,
                                     end_at_idle_rest=False)

        assert outcome.performed
        assert syringe.is_head_up() is True      # restored, not left in the basin

    def test_an_in_run_purge_that_moved_nothing_leaves_the_pose_alone(self):
        """Already at the basin (anneal bracket) — nothing to restore."""
        syringe = _Syringe(head_up=False)
        runner, _, _ = _due_runner(_Manager(syringe, _Stage(pos=FLUSH)))

        runner.maybe_purge(context="step:anneal", owns_rig=True,
                           end_at_idle_rest=False)

        assert syringe.is_head_up() is False     # left as the anneal needs it

    def test_the_idle_timer_still_ends_at_idle_rest(self):
        """Between runs the tip belongs in the basin, not in air."""
        syringe = _Syringe(head_up=True)
        runner, _, _ = _due_runner(_Manager(syringe, _Stage(pos=WELL)))

        runner.maybe_purge(context="idle")       # end_at_idle_rest defaults True

        assert syringe.is_head_up() is False


# ── Instrument-scoped claims ─────────────────────────────────────────────────

class TestScopedClaims:
    """A step occupying only the potentiostat must not block the syringe."""

    def _activity(self, owner, instruments):
        from softae.core.rig_activity import RigActivity

        a = RigActivity()
        a.acquire(owner, instruments)
        return a

    def test_a_disjoint_claim_does_not_block_a_purge(self):
        activity = self._activity("eis", {"espico", "dac_switch"})
        runner, _, _ = _due_runner(_Manager(), activity=activity)
        assert runner.maybe_purge().performed

    def test_an_overlapping_claim_blocks_a_purge(self):
        """A step using the syringe must not race the purge for it."""
        activity = self._activity("dispense", {"syringe"})
        syringe = _Syringe()
        runner, _, _ = _due_runner(_Manager(syringe), activity=activity)

        outcome = runner.maybe_purge()

        assert outcome.skipped_reason and "in use" in outcome.skipped_reason
        assert syringe.calls == []

    def test_a_stage_claim_blocks_it_too(self):
        """The purge may need the stage to reach the basin."""
        activity = self._activity("cast", {"stage"})
        runner, _, _ = _due_runner(_Manager(), activity=activity)
        assert runner.maybe_purge().skipped_reason

    def test_a_whole_rig_claim_blocks_everything(self):
        """The conservative default: a campaign claims the rig outright."""
        activity = self._activity("campaign", None)
        runner, _, _ = _due_runner(_Manager(), activity=activity)
        assert runner.maybe_purge().skipped_reason

    def test_conflicts_reports_the_blocking_owner(self):
        from softae.core.rig_activity import PURGE_INSTRUMENTS

        activity = self._activity("cast", {"syringe"})
        assert activity.conflicts(PURGE_INSTRUMENTS) == "cast"
        assert activity.conflicts({"espico"}) is None


# ── The purge's own claim ────────────────────────────────────────────────────

class _RecordingActivity(RigActivity):
    """A real ``RigActivity`` that also records the order of calls made to it.

    A real one rather than a fake, because the property under test is an
    *ordering* between two of its methods, and a fake that answered
    ``conflicts`` from a hand-set flag could not tell whether the purge's own
    claim was visible to its own precondition check.

    ``claimed`` is not overridden: the base implementation calls ``self.acquire``
    and ``self.release``, so it routes through the overrides below for free.
    """

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str]] = []

    def conflicts(self, instruments):
        blocker = super().conflicts(instruments)
        self.events.append(("conflicts", blocker or ""))
        return blocker

    def acquire(self, owner, instruments=None):
        self.events.append(("acquire", owner))
        super().acquire(owner, instruments)

    def release(self, owner):
        self.events.append(("release", owner))
        super().release(owner)


class _ObservingSyringe(_Syringe):
    """A syringe that asks who owns the rig at the moment it is commanded.

    This is the manual-control question — *may I drive this?* — asked from the
    one instant that matters: mid-dispense.
    """

    def __init__(self, activity, **kw) -> None:
        super().__init__(**kw)
        self._activity = activity
        self.owner_seen: list[str | None] = []
        self.stage_owner_seen: list[str | None] = []

    def single_pump(self, **kw) -> None:
        self.owner_seen.append(self._activity.conflicts({"syringe"}))
        self.stage_owner_seen.append(self._activity.conflicts({"stage"}))
        super().single_pump(**kw)


class TestThePurgeClaimsWhileItPurges:
    """X6: the one mechanism that moves hardware unasked now says so.

    Before this, ``PurgeRunner`` read the arbitration table every tick and never
    wrote to it — so a manual jog issued during a purge was permitted straight
    into a stage that was already moving.
    """

    def test_the_purge_holds_a_claim_while_it_dispenses(self):
        activity = _RecordingActivity()
        syringe = _ObservingSyringe(activity)
        runner, _, _ = _due_runner(_Manager(syringe), activity=activity)

        assert runner.maybe_purge().performed
        assert syringe.owner_seen == [PURGE_OWNER] * 3
        # Scoped to PURGE_INSTRUMENTS, so the stage it may travel with is
        # covered too — a jog is refused, not just a dispense.
        assert syringe.stage_owner_seen == [PURGE_OWNER] * 3

    def test_the_claim_is_released_once_the_purge_finishes(self):
        activity = _RecordingActivity()
        runner, _, _ = _due_runner(_Manager(), activity=activity)

        runner.maybe_purge()

        assert activity.busy is False
        assert activity.conflicts(PURGE_INSTRUMENTS) is None

    def test_a_failing_purge_still_releases_the_claim(self, monkeypatch):
        """A leaked claim disables purging for the session, silently."""
        activity = _RecordingActivity()
        runner, _, _ = _due_runner(_Manager(), activity=activity)

        def _boom(*_a, **_kw):
            raise RuntimeError("driver exploded")

        monkeypatch.setattr(runner, "_dispense", _boom)

        with pytest.raises(RuntimeError):
            runner.maybe_purge()

        assert activity.busy is False

    def test_the_claim_is_taken_after_the_conflict_check(self):
        """The trap: a claim held during ``conflicts`` self-blocks forever."""
        activity = _RecordingActivity()
        runner, _, _ = _due_runner(_Manager(), activity=activity)

        runner.maybe_purge()

        kinds = [kind for kind, _ in activity.events]
        assert kinds.index("conflicts") < kinds.index("acquire")
        # And nothing asks again from inside the claim, which is the only other
        # way the purge could meet its own owner.
        assert "conflicts" not in kinds[kinds.index("acquire"):]

    def test_a_repeated_purge_never_defers_against_itself(self):
        """The end state the trap produces: purging silently off, for good."""
        activity = _RecordingActivity()
        runner, _, clock = _due_runner(_Manager(), activity=activity)

        for _ in range(5):
            assert runner.maybe_purge().performed
            clock.t += 1000.0

    def test_a_dry_run_claims_nothing(self):
        """The boundary: the claim guards actuation, and this actuates nothing."""
        activity = _RecordingActivity()
        runner, _, _ = _due_runner(_Manager(), activity=activity,
                                   settings=_settings(actuate=False))

        assert runner.maybe_purge().dry_run
        assert ("acquire", PURGE_OWNER) not in activity.events

    def test_a_deferred_purge_claims_nothing(self):
        """The other boundary — refused at the pose check, so never claimed."""
        activity = _RecordingActivity()
        runner, _, _ = _due_runner(
            _Manager(_Syringe(head_up=False), _Stage(pos=WELL)),
            activity=activity)

        assert runner.maybe_purge().skipped_reason
        assert ("acquire", PURGE_OWNER) not in activity.events

    def test_an_in_run_purge_claims_alongside_the_run_that_owns_the_rig(self):
        """``owns_rig`` skips the conflict check; it must not skip the claim.

        The executor's purge window drives the syringe while the run holds the
        rig, and Manual Control asks ``conflicts`` about *instruments*, not
        about runs — so without its own claim the purge would be invisible here
        too, behind an owner that is merely annealing.
        """
        activity = _RecordingActivity()
        activity.acquire("campaign:demo")
        syringe = _ObservingSyringe(activity, head_up=False)
        runner, _, _ = _due_runner(_Manager(syringe, _Stage(pos=FLUSH)),
                                   activity=activity)

        outcome = runner.maybe_purge(context="anneal", allow_positioning=False,
                                     owns_rig=True)

        assert outcome.performed
        assert ("acquire", PURGE_OWNER) in activity.events
        assert activity.owners() == ("campaign:demo",)   # purge claim released

    def test_a_runner_without_an_activity_registry_still_purges(self):
        """Nothing to claim against is not a reason to refuse to dispense."""
        syringe = _Syringe()
        runner, _, _ = _due_runner(_Manager(syringe), activity=None)

        assert runner.maybe_purge().performed
        assert len(syringe.calls) == 3
