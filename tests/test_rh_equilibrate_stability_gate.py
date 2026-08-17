"""RH in the equilibrate phase: a **stability** gate, not a tracking gate.

An equilibration is a claim that the sample has stopped changing, and that claim
is void if the room did not stop too. So the settle window is judged twice: once
on σ (``settle_check``, unchanged) and once on the humidity the sample sat in.

The whole design turns on one distinction, and every test here is about some face
of it:

============================  ===========================================
question                      who asks it
============================  ===========================================
"is the controller obeying?"  ``classify_rh_hold`` — reads the setpoint
"was the room still?"         ``rh_window_spread`` — reads only the series
============================  ===========================================

The second reads **no setpoint**, which is what lets it sit beside a σ criterion
whose standing prohibition forbids ever making σ wait on a PV *reaching* one.

Everything below drives fabricated ``(t, %RH)`` series and fabricated
``RoundFit`` lists — no rig, no store — which is exactly what ``SettleTracker``
was built to allow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from softae.analysis.equilibration import (
    DEFAULT_RH_STABILITY_PCT,
    EXCLUDED_RH_UNREADABLE,
    RH_MOVED,
    SETTLE_CEILING,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    RoundFit,
    SettleTracker,
    rh_window_spread,
    round_rh_median,
)
from softae.core import autonomous_wiring as wiring
from softae.core.autonomous_wiring import (
    CampaignSpec,
    RHCeilingEscalation,
    SettleOutcome,
    drive_settle_phase,
)
from softae.core.autonomous_loop import LoopState
from softae.core.data_store import DataStore
from softae.core.run_plan import SettlePlan
from softae.drivers.mock_factory import create_mock_manager

RAILED_R1_OHMS = 100.0
GOOD_R1_OHMS = 4000.0

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


# ── Fabricated evidence ──────────────────────────────────────────────────────

def _flat(channels=(1, 2, 3, 4), sigma=1e-4, r1_ohms=GOOD_R1_OHMS):
    """σ that never moves: the window settles on evidence alone."""
    return [RoundFit(channel=ch, sigma=sigma, r1_ohms=r1_ohms) for ch in channels]


def _climbing(index: int, channels=(1, 2, 3, 4)):
    """σ climbing by a constant ratio — a film that never finishes.

    Geometric rather than linear: a linear ramp's *relative* spread shrinks as it
    climbs, so it eventually creeps inside any tolerance.
    """
    return [RoundFit(channel=ch, sigma=1e-4 * (1.5 ** index), r1_ohms=GOOD_R1_OHMS)
            for ch in channels]


def _feed(tracker: SettleTracker, rh_series, fits_for=lambda i: _flat()):
    """Drive *tracker* one round per entry of *rh_series*; return every verdict."""
    checks = []
    for index, rh in enumerate(rh_series):
        checks.append(tracker.observe(fits_for(index), rh_median_pct=rh))
    return checks


def _tracker(**over) -> SettleTracker:
    base = dict(tol_rel=0.10, n_rounds=3, min_channels=3,
                r1_bound_ohms=RAILED_R1_OHMS,
                rh_stability_pct=DEFAULT_RH_STABILITY_PCT)
    base.update(over)
    return SettleTracker(**base)


class FakeClock:
    """A clock that only moves when somebody waits on it."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def _plan(**over) -> SettlePlan:
    base = dict(round_period_s=50.0, min_hold_s=0.0, max_hold_s=1000.0,
                settle_n_rounds=3, settle_min_channels=3,
                rh_stability_pct=DEFAULT_RH_STABILITY_PCT)
    base.update(over)
    return SettlePlan(**base)


# ── The two pure statistics ──────────────────────────────────────────────────

def test_round_rh_median_empty_returns_none():
    """``None``, and specifically **not** ``0.0`` — which reads as a humidity."""
    assert round_rh_median([]) is None
    assert round_rh_median(None) is None
    assert round_rh_median([None, None]) is None
    assert round_rh_median([float("nan"), float("inf")]) is None
    assert round_rh_median([0.0]) == 0.0          # a real zero still reads as one


def test_round_rh_median_takes_the_median_not_the_mean():
    """The ±1.4 %RH loop ripple is not environmental change and must not count."""
    assert round_rh_median([15.0, 15.1, 15.2, 40.0]) == pytest.approx(15.15)


def test_rh_window_spread_any_none_returns_none():
    """A window with a hole in it has not been *observed* to be stable."""
    assert rh_window_spread([15.0, None, 15.1]) is None
    assert rh_window_spread([]) is None
    assert rh_window_spread([15.0, 15.4, 14.9]) == pytest.approx(0.5)


# ── The gate's central claims ────────────────────────────────────────────────

def test_settle_tracker_floor_limited_steady_rh_still_settles():
    """**The test that proves the prohibition is not violated.**

    The chamber sits pinned at 20 %RH while nothing on earth commanded 20. A
    *tracking* gate never fires here — that is the pathology the standing
    prohibition above ``SETTLE_SETTLED`` was written against, and why it forbids
    making σ wait on the PV reaching a setpoint. A *stability* gate fires on the
    first full window, because it is given no setpoint to read and compares the
    series only to itself. On the exact case the prohibition names, the two gates
    behave oppositely: the failure mode cannot be reached through this door.
    """
    tracker = _tracker()
    _feed(tracker, [20.0, 20.3, 19.9])
    assert tracker.settled is True
    assert tracker.outcome(stopped_early=True) == SETTLE_SETTLED
    assert tracker.rh_blocked_settle is False


def test_settle_tracker_oscillating_rh_does_not_settle():
    """σ flat, room swinging 12 ↔ 22 about a perfectly correct mean.

    This is the row of the table that makes the gate necessary rather than
    merely nice: ``classify_rh_hold`` grades this same series ``converging``,
    because its ``sustained_above`` / ``sustained_below`` clauses never trip when
    nothing is *sustained*. It is right for its own question and wrong for this
    one.
    """
    tracker = _tracker()
    _feed(tracker, [12.0, 22.0, 12.0])
    assert tracker.settled is False
    assert RH_MOVED in tracker.last.reason
    assert tracker.rh_blocked_settle is True
    assert tracker.last.evaluable is True     # evaluable, and it says no

    from softae.drivers.contracts import RH_CONVERGING, classify_rh_hold

    graded = classify_rh_hold(
        [(0.0, 12.0), (60.0, 22.0), (120.0, 12.0)], setpoint_pct=17.0)
    assert graded.state == RH_CONVERGING       # ...which is why this exists


def test_settle_tracker_monotone_drift_is_caught_by_the_window_test():
    """The per-round test's blind spot, and the reason the window test exists.

    Each round here is internally quiet — a per-round *range* would see nothing —
    but the room walks 22 → 15 %RH across the window, which is precisely the
    drift the operator forbade. A round-local test rejects oscillation and
    ignores drift; that is half a gate.
    """
    tracker = _tracker()
    _feed(tracker, [22.0, 18.5, 15.0])
    assert tracker.settled is False
    assert RH_MOVED in tracker.last.reason
    assert tracker.rh_spread_pct == pytest.approx(7.0)


def test_settle_tracker_monotone_drift_within_tolerance_settles():
    """A tolerance, not a demand for a flat line: 0.9 %RH of ramp is still still."""
    tracker = _tracker()
    _feed(tracker, [15.9, 15.4, 15.0])
    assert tracker.settled is True
    assert tracker.rh_spread_pct == pytest.approx(0.9)


def test_settle_tracker_unreadable_rh_is_not_evaluable():
    """Absence of evidence is not evidence — and above all is not a pass."""
    tracker = _tracker()
    _feed(tracker, [15.0, None, 15.1])
    assert tracker.last.evaluable is False
    assert tracker.last.settled is False       # asserted explicitly, on purpose
    assert tracker.settled is False
    assert EXCLUDED_RH_UNREADABLE in tracker.last.reason
    assert tracker.rh_unreadable is True
    # Never evaluable for the whole phase → the accurate finding is "nothing here
    # could tell us whether the environment was still", not "σ was still moving".
    assert tracker.outcome(stopped_early=False) == SETTLE_NOT_EVALUABLE


def test_settle_tracker_rh_gate_off_reproduces_prior_behaviour():
    """The regression guard for every existing caller: RH wild, verdicts identical."""
    wild = [5.0, 95.0, 40.0, 60.0, 12.0]
    off = _tracker(rh_stability_pct=None)
    reference = _tracker(rh_stability_pct=None)
    _feed(off, wild)
    for _ in wild:
        reference.observe(_flat())             # today's call, no keyword at all
    assert off.settled is reference.settled is True
    assert off.last.reason == reference.last.reason
    assert off.rh_spread_pct is None and off.rh_blocked_settle is False


def test_settle_tracker_rh_blocked_settle_is_not_set_by_a_window_already_failing_sigma():
    """Only a **binding** RH clause counts.

    Without this, the discriminator degrades into "RH was imperfect at some
    point" — true of nearly every phase — and the escalation counts noise.
    """
    tracker = _tracker()
    _feed(tracker, [22.0, 18.0, 15.0], fits_for=_climbing)
    assert tracker.last.settled is False       # σ said no first
    assert tracker.rh_blocked_settle is False  # so RH did not decide this window


def test_settle_tracker_railed_fits_and_moving_rh_is_not_evaluable():
    """Both exclusions at once: the σ shortfall wins the reason, checked first.

    Guards against a refactor in which one clause masks the other.
    """
    railed = [RoundFit(channel=ch, sigma=0.5, r1_ohms=RAILED_R1_OHMS)
              for ch in (1, 2, 3, 4)]
    tracker = _tracker()
    _feed(tracker, [22.0, 18.0, 15.0], fits_for=lambda i: railed)
    assert tracker.last.evaluable is False
    assert tracker.settled is False
    assert "participating channel(s)" in tracker.last.reason
    assert RH_MOVED not in tracker.last.reason
    assert tracker.rh_blocked_settle is False and tracker.rh_unreadable is False


# ── The plan-level tolerance ─────────────────────────────────────────────────

def test_settle_plan_rejects_non_positive_rh_stability_pct():
    with pytest.raises(ValueError, match="rh_stability_pct must be positive"):
        SettlePlan(round_period_s=1.0, min_hold_s=0.0, max_hold_s=1.0,
                   rh_stability_pct=0.0)
    # None is the off switch and must stay legal.
    assert SettlePlan(round_period_s=1.0, min_hold_s=0.0, max_hold_s=1.0,
                      rh_stability_pct=None).rh_stability_pct is None


def test_campaign_spec_threads_rh_stability_pct_into_the_settle_plan():
    """The seventh flat field, and the only one of the seven with a safe default."""
    def _spec(**over):
        base = dict(name="c", channels=(21, 22, 23, 24),
                    pcb_name="SoftAE_EIS_4Stripe", parameter_space=SPACE,
                    vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1),
                    deadvols=(10.0, 30.0), budget=1, seed=1,
                    equilibration_method="settle", round_period_s=1.0,
                    min_hold_s=0.0, max_hold_s=1.0)
        base.update(over)
        return CampaignSpec(**base)

    assert _spec().settle_plan().rh_stability_pct == pytest.approx(1.5)
    assert _spec(rh_stability_pct=0.8).settle_plan().rh_stability_pct == 0.8
    assert _spec(rh_stability_pct=None).settle_plan().rh_stability_pct is None


# ── The driver ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drive_settle_phase_rh_tolerance_set_without_supplier_raises():
    """The one real trap: a defaulted keyword meaning "unreadable", and a missing
    wire that also means "unreadable", are indistinguishable at the point of harm.
    """
    clock = FakeClock()
    with pytest.raises(ValueError, match="no rh_for_round supplier"):
        await drive_settle_phase(
            _plan(), channels=[1, 2, 3, 4],
            measure_round=lambda i: _round(i), fits_from=lambda raws: _flat(),
            r1_bound_ohms=RAILED_R1_OHMS, sleep=clock.sleep, now=clock.now)


async def _round(index: int):
    return {ch: index for ch in (1, 2, 3, 4)}


@pytest.mark.asyncio
async def test_drive_settle_phase_moving_rh_reaches_ceiling_without_parking():
    """**The phase never parks — under any circumstances.**

    Still exactly right after the escalation, and it now says something sharper:
    every park in this design is the *campaign's* decision, made K trials later
    on the record these phases leave behind. A phase that cannot certify holds to
    its ceiling, hands back its last round, and returns normally.
    """
    clock = FakeClock()
    drifting = iter([22.0, 21.0, 20.0, 19.0, 18.0, 17.0, 16.0, 15.0] + [15.0] * 50)
    outcome, last_raws = await drive_settle_phase(
        _plan(round_period_s=100.0, max_hold_s=500.0),
        channels=[1, 2, 3, 4],
        measure_round=_round,
        fits_from=lambda raws: _flat(),
        r1_bound_ohms=RAILED_R1_OHMS,
        rh_for_round=lambda i: next(drifting),
        sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome == SETTLE_CEILING and not outcome.settled
    assert last_raws == {ch: 5 for ch in (1, 2, 3, 4)}
    assert outcome.rh_limited is True and outcome.rh_unreadable is False


@pytest.mark.asyncio
async def test_settle_outcome_records_rh_spread():
    """``rh_spread_pct`` in the sidecar is what lets the provisional default
    re-derive itself from real campaigns at their own q."""
    clock = FakeClock()
    outcome, _ = await drive_settle_phase(
        _plan(), channels=[1, 2, 3, 4], measure_round=_round,
        fits_from=lambda raws: _flat(), r1_bound_ohms=RAILED_R1_OHMS,
        rh_for_round=lambda i: 15.0 + 0.2 * i,
        sleep=clock.sleep, now=clock.now)
    record = outcome.as_dict()
    assert record["settle_outcome"] == SETTLE_SETTLED
    assert record["rh_spread_pct"] == pytest.approx(0.4)
    assert record["rh_stability_pct"] == pytest.approx(1.5)
    assert record["rh_limited"] is False and record["rh_unreadable"] is False
    assert "RH spread 0.40%RH" in outcome.describe()


@pytest.mark.asyncio
async def test_settle_outcome_rh_blocked_ceiling_is_marked_rh_limited():
    """σ would have certified; the room moved. The phase must say which."""
    clock = FakeClock()
    outcome, _ = await drive_settle_phase(
        _plan(round_period_s=100.0, max_hold_s=400.0),
        channels=[1, 2, 3, 4], measure_round=_round,
        fits_from=lambda raws: _flat(), r1_bound_ohms=RAILED_R1_OHMS,
        rh_for_round=lambda i: 22.0 - 1.0 * i,
        sleep=clock.sleep, now=clock.now)
    assert outcome.outcome == SETTLE_CEILING
    assert outcome.rh_limited is True
    assert "the room moved, not the film" in outcome.describe()


@pytest.mark.asyncio
async def test_settle_outcome_slow_film_ceiling_is_not_rh_limited():
    """**The test the escalation stands on.**

    Without it every slow film in the campaign counts toward a park, and K is
    reached by the most ordinary outcome this system produces.
    """
    clock = FakeClock()
    outcome, _ = await drive_settle_phase(
        _plan(round_period_s=100.0, max_hold_s=400.0),
        channels=[1, 2, 3, 4], measure_round=_round,
        fits_from=lambda raws: _climbing(next(iter(raws.values()))),
        r1_bound_ohms=RAILED_R1_OHMS,
        rh_for_round=lambda i: 15.0,
        sleep=clock.sleep, now=clock.now)
    assert outcome.outcome == SETTLE_CEILING
    assert outcome.rh_limited is False and outcome.rh_unreadable is False


@pytest.mark.asyncio
async def test_settle_outcome_unreadable_rh_throughout_is_marked_unreadable():
    clock = FakeClock()
    outcome, _ = await drive_settle_phase(
        _plan(round_period_s=100.0, max_hold_s=400.0),
        channels=[1, 2, 3, 4], measure_round=_round,
        fits_from=lambda raws: _flat(), r1_bound_ohms=RAILED_R1_OHMS,
        rh_for_round=lambda i: None,
        sleep=clock.sleep, now=clock.now)
    assert outcome.outcome == SETTLE_NOT_EVALUABLE
    assert outcome.rh_unreadable is True and outcome.rh_limited is False
    assert "RH channel could not be judged" in outcome.describe()


# ── The escalation, as a rule ────────────────────────────────────────────────

def _outcome(**over) -> SettleOutcome:
    base = dict(outcome=SETTLE_CEILING, n_rounds=3, held_s=1.0,
                rh_stability_pct=1.5)
    base.update(over)
    return SettleOutcome(**base)


def _escalation(limit=3, **over) -> tuple[RHCeilingEscalation, list[str]]:
    parked: list[str] = []
    return RHCeilingEscalation(limit=limit, park=parked.append, **over), parked


def test_rh_decided_ceilings_below_the_limit_do_not_park():
    esc, parked = _escalation()
    assert [esc.note(_outcome(rh_limited=True)) for _ in range(2)] == [False, False]
    assert parked == [] and esc.streak == 2


def test_rh_decided_ceilings_reaching_the_limit_park_once():
    esc, parked = _escalation()
    for _ in range(3):
        esc.note(_outcome(rh_limited=True))
    assert len(parked) == 1
    assert "RH channel" in parked[0] and "humidity moved" in parked[0]
    # The loop is already stopping; a further RH-decided phase must not re-park.
    esc.note(_outcome(rh_limited=True))
    assert len(parked) == 1


def test_a_settled_trial_resets_the_rh_ceiling_streak():
    """Consecutive, not cumulative."""
    esc, parked = _escalation()
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(outcome=SETTLE_SETTLED))
    assert esc.streak == 0
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(rh_limited=True))
    assert parked == []


def test_a_slow_film_ceiling_resets_the_rh_ceiling_streak():
    """The ordinary outcome, and it must clear the count like any other."""
    esc, parked = _escalation()
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(outcome=SETTLE_CEILING))     # both booleans False
    assert esc.streak == 0 and parked == []


def test_a_disabled_rh_gate_neither_increments_nor_resets_the_streak():
    """A non-observation is evidence of neither health nor fault."""
    esc, parked = _escalation()
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(rh_stability_pct=None))      # gate off: no opinion
    assert esc.streak == 1 and parked == []
    esc.note(_outcome(rh_limited=True))
    esc.note(_outcome(rh_limited=True))
    assert len(parked) == 1                        # the off phase did not reset


def test_an_unreadable_rh_phase_counts_toward_the_same_limit():
    """Pins the widening, so a later reader does not "fix" the counter to
    ``ceiling``-only and silently exempt a dead sensor."""
    esc, parked = _escalation()
    for _ in range(3):
        esc.note(_outcome(outcome=SETTLE_NOT_EVALUABLE, rh_unreadable=True))
    assert len(parked) == 1 and "could not be read" in parked[0]


def test_a_zero_limit_disables_the_escalation_and_leaves_the_per_trial_rule():
    esc, parked = _escalation(limit=0)
    for _ in range(10):
        esc.note(_outcome(rh_limited=True))
    assert parked == [] and esc.streak == 10


def test_the_escalation_limit_comes_from_config_with_a_documented_default():
    """``[safety] rh_ceiling_park_after_trials``; 3 matches the loop's own streak."""
    assert wiring.DEFAULT_RH_CEILING_PARK_AFTER_TRIALS == 3
    assert wiring.rh_ceiling_park_after_trials() >= 0


# ── The escalation, on the real campaign path ────────────────────────────────

@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


def _campaign_spec(**over) -> CampaignSpec:
    base = dict(
        name="rh_gate_campaign", channels=(21, 22, 23, 24),
        pcb_name="SoftAE_EIS_4Stripe", parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1), deadvols=(10.0, 30.0),
        time_scale=0.0, budget=4, seed=7, equilibration_method="settle",
        round_period_s=0.01, min_hold_s=0.0, max_hold_s=0.05,
        settle_min_channels=3,
    )
    base.update(over)
    return CampaignSpec(**base)


def _script_rh_limited(monkeypatch):
    """Every equilibrate phase comes back RH-limited, with no rig time spent."""
    async def _fake(plan, **kw):
        return SettleOutcome(outcome=SETTLE_CEILING, n_rounds=3, held_s=0.0,
                             rh_spread_pct=4.2,
                             rh_stability_pct=plan.rh_stability_pct,
                             rh_limited=True), {}

    monkeypatch.setattr(wiring, "drive_settle_phase", _fake)


@pytest.mark.asyncio
async def test_the_escalation_parks_rather_than_raising(
    connected, tmp_path: Path, monkeypatch
):
    """No ``SafetyError`` escapes ``_equilibrate``, and the run really stops.

    A raise would be classified a hard fault and abort the trial mid-batch, which
    is the stop this whole design exists to avoid. And "asks to stop" is not
    "stops": the hook runs inside ``_post_measure``, whose caller immediately
    sets ANALYZING over the parked state — so the assertion that matters is that
    trial K is the **last** one, not merely that a park event was emitted.
    """
    _script_rh_limited(monkeypatch)
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    result = await wiring.run_autonomous_campaign(
        _campaign_spec(), manager=connected, data_store=store,
        on_event=events.append)

    parks = [e for e in events if e["type"] == "park"]
    assert len(parks) == 1
    assert "RH channel" in parks[0]["reason"]
    assert result.n_trials == 3                 # K, out of a budget of 4
    assert result.final_state == LoopState.STOPPED.name
    # Parked, so the resume point is deliberately kept.
    assert store.campaign_checkpoint("rh_gate_campaign") is not None
    store.close()


@pytest.mark.asyncio
async def test_the_park_happens_after_the_settle_record_is_written(
    connected, tmp_path: Path, monkeypatch
):
    """Pins the trial-boundary property the ``on_trial_measured`` siting buys.

    It is otherwise invisible, and would be lost by any refactor that moved the
    check earlier in ``_equilibrate``. Films already cast keep aging while the
    rig waits for a human, so a park that arrives before the batch is recorded
    loses the very evidence it was raised about.
    """
    _script_rh_limited(monkeypatch)
    store = DataStore(tmp_path / "proj")
    seen: dict[str, object] = {}
    run_id: list[str] = []

    def watch(event):
        if event["type"] == "run_started":
            run_id.append(event["run_id"])
        if event["type"] == "park" and run_id:
            sidecar = Path(store.run_dir(run_id[0])) / "settle.json"
            seen["exists_at_park"] = sidecar.exists()
            seen["records_at_park"] = len(
                json.loads(sidecar.read_text(encoding="utf-8")))

    await wiring.run_autonomous_campaign(
        _campaign_spec(), manager=connected, data_store=store, on_event=watch)

    assert seen["exists_at_park"] is True
    assert seen["records_at_park"] == 3          # trial K's verdict included
    store.close()


@pytest.mark.asyncio
async def test_the_rh_ceiling_streak_survives_a_checkpoint_round_trip(
    connected, tmp_path: Path, monkeypatch
):
    """**The test that fails if any of the persistence sites is missed** — the
    column, the migration, the write side or the restore side.

    It runs against a store whose ``campaign_checkpoints`` table was created
    *without* the column, because the DDL is a bare ``CREATE TABLE IF NOT
    EXISTS`` and every real store is an existing one: adding the column to the
    DDL alone is correct on a fresh store and silently broken everywhere else.

    Why it must survive at all: an RH streak parks nothing until it reaches K, so
    a restart can land mid-streak for entirely unrelated reasons, and a restart
    is among the likeliest things to happen while a chronic fault is developing.
    """
    project = tmp_path / "proj"
    legacy = DataStore(project)
    legacy._conn.execute("DROP TABLE campaign_checkpoints")
    legacy._conn.execute(
        "CREATE TABLE campaign_checkpoints ("
        " campaign TEXT PRIMARY KEY, run_id TEXT, iteration INTEGER NOT NULL,"
        " loop_state TEXT, board_id INTEGER, spec_json TEXT,"
        " optimizer_json TEXT, updated_at TEXT NOT NULL)")
    legacy._conn.commit()
    legacy.close()

    # Reopening runs the migration, which is the only thing that can add the
    # column to a table that already exists.
    store = DataStore(project)
    assert "rh_ceiling_streak" in {
        row[1] for row in
        store._conn.execute("PRAGMA table_info(campaign_checkpoints)").fetchall()}

    # A limit of 2 so the first run parks with the streak at 2 — and a parked run
    # is precisely the one that *keeps* its checkpoint. (A campaign that spends
    # its whole budget ends on purpose and clears it, so there is no streak left
    # to carry and nothing to resume.)
    monkeypatch.setattr(wiring, "rh_ceiling_park_after_trials", lambda: 2)
    _script_rh_limited(monkeypatch)
    spec = _campaign_spec(budget=6)
    first = await wiring.run_autonomous_campaign(
        spec, manager=connected, data_store=store)
    assert first.n_trials == 2
    saved = store.campaign_checkpoint(spec.name)
    assert saved is not None and saved["rh_ceiling_streak"] == 2

    # Resume with the real K: the restored streak is already 2, so **one** more
    # RH-decided trial reaches the limit. Were the streak zeroed on resume — as
    # `_consecutive_failures` is — this run would take three more trials instead,
    # and a chronic fault would get a fresh allowance after every restart.
    monkeypatch.setattr(wiring, "rh_ceiling_park_after_trials", lambda: 3)
    resumed_events: list[dict] = []
    resumed = await wiring.run_autonomous_campaign(
        _campaign_spec(budget=6), manager=connected, data_store=store,
        resume=True, on_event=resumed_events.append)
    assert len([e for e in resumed_events if e["type"] == "park"]) == 1
    assert resumed.n_trials == 3                # 2 restored + exactly one more
    store.close()
