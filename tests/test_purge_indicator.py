"""The purge indicator's decision — pure, headless, against a virtual clock.

No Qt anywhere in this file, and that is the point: the state worth having an
indicator *for* — a purge owed and prevented for hours — is reached by time
passing, which no widget test can arrange. Rendering is asserted separately in
``test_purge_visibility.py``.

Two rulings are pinned here rather than left to the renderer:

* **A dry run is not a warning.** ``[purge] actuate`` ships ``false``, so a dry
  run is the *normal* case; colouring it would make the badge permanent amber
  and teach the operator to stop seeing it.
* **Overdue dominates regardless of ``actuate``.** A stagnating line is a fact
  about the fluid, not about whether the harness is armed.
"""

from __future__ import annotations

from pathlib import Path

from softae.core.purge import PurgeScheduler, PurgeSettings
from softae.core.purge_runner import IdleRestState, PurgeOutcome, PurgeRunner
from softae.gui.widgets.purge_indicator import PurgeIndicator, purge_indicator

INTERVAL_S = 900.0
VOLUMES = {0: 10.0, 1: 20.0, 2: 10.0}


class _Clock:
    """A clock the test moves by hand — shared by the scheduler and the caller."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def _settings(**kw) -> PurgeSettings:
    base = dict(enabled=True, actuate=False, interval_s=INTERVAL_S,
                particulate_uL=20.0, other_uL=10.0,
                particulate_pumps=(1,), pumps=(0, 1, 2))
    base.update(kw)
    return PurgeSettings(**base)


def _scheduler(**kw) -> tuple[PurgeScheduler, _Clock]:
    clock = _Clock()
    return PurgeScheduler(_settings(**kw), now=clock), clock


# ── The quiet states ─────────────────────────────────────────────────────────

def test_indicator_fresh_scheduler_reports_the_next_purge():
    sched, clock = _scheduler()
    ind = purge_indicator(sched, now=clock())
    assert ind.state == "scheduled"
    assert "15 min" in ind.headline
    assert ind.attention is False
    assert ind.overdue_s == 0.0


def test_indicator_approaching_due_reports_near_without_attention():
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S - 60.0)
    ind = purge_indicator(sched, now=clock())
    assert ind.state == "near"
    assert "1 min" in ind.headline
    assert ind.attention is False


def test_indicator_absent_scheduler_reports_unconfigured():
    ind = purge_indicator(None, now=0.0)
    assert ind.state == "unconfigured"
    assert ind.attention is False


def test_indicator_disabled_purging_reports_off():
    sched, clock = _scheduler(enabled=False)
    ind = purge_indicator(sched, now=clock())
    assert ind.state == "unconfigured"
    assert "off" in ind.headline


def test_indicator_unreadable_scheduler_reports_unavailable():
    """A view must survive a scheduler it cannot read — it must not blank."""

    class _Broken:
        @property
        def settings(self):
            raise RuntimeError("gone")

    ind = purge_indicator(_Broken(), now=0.0)
    assert ind.state == "unconfigured"
    assert ind.attention is False


# ── Three states, distinguishable ────────────────────────────────────────────

def _after_a(outcome_kind: str) -> PurgeIndicator:
    """An indicator 60 s after a purge of *outcome_kind* reset the timers."""
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S)
    sched.note_purged()                       # what both branches actually do
    at = clock()
    clock.advance(60.0)
    outcome = PurgeOutcome(volumes_uL=dict(VOLUMES),
                           **{outcome_kind: True})
    return purge_indicator(sched, last_outcome=outcome, last_at=at, now=clock())


def test_indicator_dry_run_outcome_renders_neutral_not_a_warning():
    ind = _after_a("dry_run")
    assert ind.state == "dry_run"
    assert ind.attention is False
    assert "actuate is off" in ind.detail
    assert "40 µL" in ind.headline


def test_indicator_performed_outcome_reports_volume_and_age():
    ind = _after_a("performed")
    assert ind.state == "purged"
    assert ind.attention is False
    assert "40 µL" in ind.headline
    assert "1 min ago" in ind.headline


def test_indicator_distinguishes_dry_run_performed_and_overdue():
    """The three states the operator's ruling names, each rendered apart."""
    dry, done = _after_a("dry_run"), _after_a("performed")
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 300.0)          # nothing reset the timers
    late = purge_indicator(sched, now=clock())

    assert {dry.state, done.state, late.state} == {"dry_run", "purged", "overdue"}
    assert len({dry.headline, done.headline, late.headline}) == 3
    assert (dry.attention, done.attention, late.attention) == (False, False, True)


# ── Overdue ──────────────────────────────────────────────────────────────────

def test_indicator_blocked_purge_reports_the_overdue_magnitude():
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 300.0)
    blocked = PurgeOutcome(skipped_reason="rig is in use (ht:cast_series)",
                           volumes_uL=dict(VOLUMES))
    ind = purge_indicator(sched, last_outcome=blocked, now=clock())
    assert ind.state == "overdue"
    assert ind.overdue_s == 300.0
    assert "5 min" in ind.detail and "5 min" in ind.headline
    assert "ht:cast_series" in ind.detail


def test_indicator_overdue_magnitude_grows_across_repeated_deferrals():
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 300.0)
    first = purge_indicator(sched, now=clock())
    clock.advance(1200.0)
    second = purge_indicator(sched, now=clock())

    assert second.overdue_s > first.overdue_s
    assert second.detail != first.detail
    assert "25 min" in second.detail


def test_indicator_overdue_dominates_even_when_actuate_is_off():
    """The shipped default. The line stagnates whether or not the pump is armed."""
    sched, clock = _scheduler(actuate=False)
    clock.advance(INTERVAL_S + 600.0)
    ind = purge_indicator(sched, now=clock())
    assert ind.state == "overdue"
    assert ind.attention is True
    assert "actuate is off" in ind.detail        # stated, not coloured


def test_indicator_overdue_dominates_when_actuate_is_on():
    sched, clock = _scheduler(actuate=True)
    clock.advance(INTERVAL_S + 600.0)
    ind = purge_indicator(sched, now=clock())
    assert ind.state == "overdue"
    assert ind.attention is True


# ── Attention has to end ─────────────────────────────────────────────────────

def test_indicator_attention_ends_when_the_purge_runs():
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 600.0)
    assert purge_indicator(sched, now=clock()).attention is True

    sched.note_purged()                        # the purge happened
    done = PurgeOutcome(performed=True, volumes_uL=dict(VOLUMES))
    ind = purge_indicator(sched, last_outcome=done, last_at=clock(), now=clock())
    assert ind.state == "purged"
    assert ind.attention is False


def test_indicator_attention_ends_on_acknowledgement():
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 600.0)
    acked_at = clock()
    clock.advance(120.0)
    ind = purge_indicator(sched, now=clock(), acknowledged_at=acked_at)
    assert ind.state == "overdue"               # still true, just not shouting
    assert ind.attention is False


def test_indicator_acknowledgement_is_superseded_by_a_further_interval():
    """One click must not silence an unbounded problem."""
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 600.0)
    acked_at = clock()

    clock.advance(INTERVAL_S - 1.0)
    assert purge_indicator(sched, now=clock(),
                           acknowledged_at=acked_at).attention is False
    clock.advance(2.0)
    assert purge_indicator(sched, now=clock(),
                           acknowledged_at=acked_at).attention is True


def test_indicator_acknowledgement_from_an_earlier_episode_does_not_silence():
    """A click, then a purge, then a fresh block: the badge must light again."""
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 600.0)
    acked_at = clock()

    sched.note_purged()                        # episode over
    clock.advance(INTERVAL_S + 60.0)           # a new one begins, blocked again
    ind = purge_indicator(sched, now=clock(), acknowledged_at=acked_at)
    assert ind.state == "overdue"
    assert ind.attention is True


# ── Attached: no timing at all ───────────────────────────────────────────────

def test_indicator_attached_window_reports_the_holder_not_a_schedule():
    """Its scheduler's timers started at *its* launch, so any number is invented."""
    sched, clock = _scheduler()
    clock.advance(INTERVAL_S + 600.0)
    ind = purge_indicator(sched, now=clock(),
                          attached_holder="campaign 'shadow-run'")
    assert ind.state == "not_ours"
    assert "shadow-run" in ind.headline
    assert ind.overdue_s == 0.0
    assert ind.attention is False
    assert "min" not in ind.headline and "µL" not in ind.headline


# ── The asymmetry the whole badge rests on ───────────────────────────────────

class _Syringe:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def is_head_up(self) -> bool:
        return False

    def head_descend(self) -> None:
        pass

    def head_retract(self) -> None:
        pass

    def single_pump(self, *, res_vol, ID, rate, dispense_vol) -> None:
        self.calls.append((int(ID), float(dispense_vol)))


FLUSH = (-50.0, 50.0)


class _Stage:
    def live_position(self):
        return FLUSH

    def move_to(self, x, y, *, head_may_be_down: bool = False) -> None:
        pass


class _Manager:
    def __init__(self) -> None:
        self._items = {"syringe": _Syringe(), "stage": _Stage()}

    def get(self, name):
        return self._items[name]


class _Busy:
    """A whole-rig claim, without importing the real ``RigActivity``."""

    def conflicts(self, instruments):
        return "ht:cast_series"


def _runner(clock, *, activity=None) -> tuple[PurgeRunner, PurgeScheduler]:
    sched = PurgeScheduler(_settings(), now=clock)
    return PurgeRunner(_Manager(), sched, idle_rest=IdleRestState(True),
                       activity=activity, flush_xy=FLUSH), sched


def test_maybe_purge_dry_run_resets_the_timer_so_overdue_never_accumulates():
    """The trap. ``maybe_purge`` calls ``note_purged()`` in its dry-run branch.

    So on a free rig under the shipped default the badge never reaches
    ``overdue`` — and an implementer who assumed overdue accrues from the
    interval would ship one that cannot light.
    """
    clock = _Clock()
    runner, sched = _runner(clock)
    clock.advance(INTERVAL_S + 300.0)

    outcome = runner.maybe_purge()
    assert outcome.dry_run is True
    assert sched.due() is None                                    # timers reset
    assert purge_indicator(sched, last_outcome=outcome,
                           last_at=clock(), now=clock()).state == "dry_run"


def test_maybe_purge_blocked_does_not_reset_so_overdue_measures_time_blocked():
    """The other half: a skip returns *before* ``note_purged``, deliberately."""
    clock = _Clock()
    runner, sched = _runner(clock, activity=_Busy())
    clock.advance(INTERVAL_S + 300.0)

    first = runner.maybe_purge()
    assert first.skipped_reason and "ht:cast_series" in first.skipped_reason
    assert purge_indicator(sched, last_outcome=first,
                           now=clock()).overdue_s == 300.0

    clock.advance(600.0)
    runner.maybe_purge()
    assert purge_indicator(sched, now=clock()).overdue_s == 900.0


def test_purge_indicator_module_imports_no_gui_toolkit():
    """The decision is headless, and stays that way — the renderer is elsewhere."""
    source = Path(purge_indicator.__globals__["__file__"]).read_text(
        encoding="utf-8")
    assert "PySide6" not in source
