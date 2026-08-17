"""The EQUILIBRATE phase: stop holding when the measurement stops moving.

A campaign trial used to cast, hold for a fixed time, and measure **once**. These
tests pin the phase that ends that: the plan-level ordering, the opt-in switch,
the driver loop, and the three-state outcome it records.

The criterion itself lives in :mod:`softae.analysis.equilibration` and is
exercised by ``test_equilibration_analysis.py``; nothing here re-tests it. What
is tested here is the *caller* — floor, ceiling, and the one thing only the
caller can get wrong: **threading the R₁ bound through so a railed fit is not
counted as evidence.** 325 of 1440 fits in
``20260811T023757Z_equilibration_characterization`` railed while reporting
``success = 1``; a rail reports a constant, and a constant is exactly what a
settle criterion mistakes for settled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from softae.analysis.equilibration import (
    EXCLUDED_RAILED,
    SETTLE_CEILING,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    RoundFit,
)
from softae.core import autonomous_wiring as wiring
from softae.core.autonomous_wiring import (
    CampaignSpec,
    build_equilibration_workflow,
    build_settle_round_workflow,
    drive_settle_phase,
    settle_r1_bound_ohms,
    settle_step_name,
)
from softae.core.data_store import DataStore
from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)
from softae.drivers.mock_factory import create_mock_manager

#: The bound the reference run's railed fits sat on.
RAILED_R1_OHMS = 100.0

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _plan(**over) -> SettlePlan:
    base = dict(round_period_s=50.0, min_hold_s=100.0, max_hold_s=10_000.0,
                settle_n_rounds=3, settle_min_channels=3)
    base.update(over)
    return SettlePlan(**base)


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="settle_campaign",
        channels=(21, 22, 23, 24),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


class FakeClock:
    """A clock that only moves when somebody waits on it.

    Which is the whole reason the driver takes ``sleep`` and ``now`` as
    parameters: an eight-hour cure is not a thing a test may spend.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self.measured_at: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def _rounds(clock: FakeClock, script):
    """A ``measure_round`` that records *when* each round was taken."""

    async def measure_round(index: int):
        clock.measured_at.append(clock.t)
        return {ch: index for ch in script}

    return measure_round


def _flat_after(n_moving: int, channels=(1, 2, 3, 4), *, r1_ohms=4000.0):
    """Fits that drift for *n_moving* rounds and are then flat to within noise."""

    def fits_from(raws):
        index = next(iter(raws.values()), 0)
        sigma = 1e-4 * (2.0 ** max(0, n_moving - index)) if index < n_moving else 1e-4
        return [RoundFit(channel=ch, sigma=sigma, r1_ohms=r1_ohms) for ch in channels]

    return fits_from


def _never_settles(channels=(1, 2, 3, 4)):
    """σ climbing by a constant *ratio* — a film that never finishes.

    Geometric rather than linear on purpose: a linear ramp's relative spread
    shrinks as it climbs, so it eventually creeps inside any tolerance and the
    test would pass for the wrong reason.
    """

    def fits_from(raws):
        index = next(iter(raws.values()), 0)
        sigma = 1e-4 * (1.5 ** index)
        return [RoundFit(channel=ch, sigma=sigma, r1_ohms=4000.0) for ch in channels]

    return fits_from


def _railed(channels=(1, 2, 3, 4), *, r1_ohms=RAILED_R1_OHMS):
    """A fit resting on the model's R₁ floor: the same σ every single round.

    This is the shape that makes the bound load-bearing. It passes a stability
    test perfectly, and it measured nothing.
    """

    def fits_from(raws):
        return [RoundFit(channel=ch, sigma=0.5, r1_ohms=r1_ohms) for ch in channels]

    return fits_from


# ── Change 1: the plan-level phase ───────────────────────────────────────────

def test_run_plan_orders_formulate_anneal_equilibrate_measure():
    plan = RunPlan.pointwise(anneal=True, settle=_plan())
    assert [p.kind for p in plan.phases] == [
        PhaseKind.FORMULATE, PhaseKind.ANNEAL,
        PhaseKind.EQUILIBRATE, PhaseKind.MEASURE,
    ]
    assert plan.has_equilibrate and plan.has_anneal


def test_batch_scope_puts_equilibrate_per_batch():
    """The q-channel batch round is the shape ``settle_check`` was built for."""
    plan = RunPlan.batch(anneal=True, settle=_plan())
    scopes = {p.kind: p.scope for p in plan.phases}
    assert scopes[PhaseKind.EQUILIBRATE] is PhaseScope.PER_BATCH
    assert scopes[PhaseKind.FORMULATE] is PhaseScope.PER_SAMPLE


def test_a_plan_without_a_settle_phase_is_untouched():
    assert not RunPlan.pointwise(anneal=True).has_equilibrate
    assert RunPlan.pointwise().phases == RunPlan.pointwise(settle=None).phases


def test_equilibrate_phase_requires_a_settle_plan():
    """``min_hold_s`` is the cure time — there is no safe default to invent."""
    with pytest.raises(ValueError, match="no safe default"):
        RunPlan((RunPhase(PhaseKind.FORMULATE),
                 RunPhase(PhaseKind.EQUILIBRATE)))


def test_only_equilibrate_may_carry_a_settle_plan():
    with pytest.raises(ValueError, match="terminates on evidence"):
        RunPlan((RunPhase(PhaseKind.FORMULATE),
                 RunPhase(PhaseKind.ANNEAL, settle=_plan())))


def test_settle_plan_refuses_a_ceiling_below_its_floor():
    with pytest.raises(ValueError, match="ceiling would fire before the floor"):
        SettlePlan(round_period_s=10.0, min_hold_s=600.0, max_hold_s=60.0)


def test_settle_defaults_come_from_the_measured_run_not_from_here():
    """0.10 clears the 5.98 % noise floor that run measured; 0.02 cannot."""
    plan = SettlePlan(round_period_s=1.0, min_hold_s=0.0, max_hold_s=1.0)
    assert plan.settle_tol_rel == pytest.approx(0.10)
    assert plan.settle_n_rounds == 3
    assert plan.settle_min_channels == 3


def test_the_phase_names_its_own_durations_in_the_plan_summary():
    text = RunPlan.batch(settle=_plan(min_hold_s=600.0, max_hold_s=3600.0)).describe()
    assert "Equilibrate" in text and "[per batch]" in text
    assert text.index("Equilibrate") < text.index("Measure")


# ── Change 2: the opt-in switch on CampaignSpec ──────────────────────────────

def test_wait_is_the_default_and_asks_for_no_settle_phase():
    assert _spec().equilibration_method == "wait"
    assert _spec().settle_plan() is None


def test_settle_resolves_the_six_parameters_into_a_plan():
    spec = _spec(equilibration_method="settle", round_period_s=120.0,
                 min_hold_s=600.0, max_hold_s=7200.0, settle_min_channels=2)
    plan = spec.settle_plan()
    assert plan == SettlePlan(round_period_s=120.0, min_hold_s=600.0,
                              max_hold_s=7200.0, settle_min_channels=2)


def test_settle_without_a_cure_time_is_refused_rather_than_invented():
    with pytest.raises(ValueError, match="min_hold_s"):
        _spec(equilibration_method="settle").settle_plan()


def test_settle_said_twice_is_refused_rather_than_reconciled():
    spec = _spec(run_plan=RunPlan.batch(settle=_plan()), min_hold_s=1.0)
    with pytest.raises(ValueError, match="say it once"):
        spec.settle_plan()


def test_settle_parameters_without_the_method_are_refused():
    with pytest.raises(ValueError, match="unless the method"):
        _spec(min_hold_s=600.0).settle_plan()


def test_a_run_plan_equilibrate_phase_supplies_the_campaign_plan():
    spec = _spec(run_plan=RunPlan.batch(settle=_plan(min_hold_s=42.0)))
    assert spec.settle_plan().min_hold_s == 42.0


def test_the_chamber_step_stays_on_wait_when_the_sample_settles():
    """``settle`` is a sample criterion; no instrument exposes it as a method."""
    wf = build_equilibration_workflow(
        _spec(equilibration_method="settle", round_period_s=1.0,
              min_hold_s=0.0, max_hold_s=1.0))
    assert wf.setup[0].method == "wait"
    assert build_equilibration_workflow(_spec()).setup[0].method == "wait"


def test_a_settle_round_reuses_the_measurement_step_builder():
    """Same sweep as a MEASURE phase, or the criterion judges another quantity."""
    spec = _spec()
    wf = build_settle_round_workflow(spec, [21, 22], round_index=3)
    steps = wf.setup
    assert [s.name for s in steps] == [settle_step_name(21, 3), settle_step_name(22, 3)]
    reference = wiring.measure_step_name(21)
    from softae.core.modality_registry import get_modality

    primary = get_modality(spec.measurement.modality).build_measure_step(
        21, spec.measurement)
    assert primary.name == reference
    assert steps[0].instrument == primary.instrument
    assert steps[0].method == primary.method
    assert steps[0].params == primary.params


def test_a_settle_round_can_never_enter_the_objective_by_itself():
    wf = build_settle_round_workflow(_spec(), [21], round_index=0)
    assert not wiring.is_primary_measurement(wf.setup[0].tags)


# ── Change 2/3: the driver loop ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flat_rounds_stop_at_settled_and_never_before_the_floor():
    clock = FakeClock()
    plan = _plan(min_hold_s=100.0, round_period_s=50.0)
    outcome, _last = await drive_settle_phase(
        plan, channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_flat_after(2),
        r1_bound_ohms=RAILED_R1_OHMS,
        sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome == SETTLE_SETTLED and outcome.settled
    # The floor is held BEFORE the first round, not averaged into the hold.
    assert min(clock.measured_at) >= plan.min_hold_s
    assert outcome.held_s >= plan.min_hold_s
    # Three consecutive flat rounds after two moving ones — no earlier.
    assert outcome.n_rounds == 5
    assert outcome.participating == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_rounds_that_never_settle_stop_at_the_ceiling():
    clock = FakeClock()
    plan = _plan(min_hold_s=100.0, round_period_s=100.0, max_hold_s=1000.0)
    outcome, _last = await drive_settle_phase(
        plan, channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_never_settles(),
        r1_bound_ohms=RAILED_R1_OHMS,
        sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome == SETTLE_CEILING
    assert not outcome.settled
    # The ceiling is unconditional: the phase never runs past it.
    assert outcome.held_s <= plan.max_hold_s
    # "Evaluable and said no" — not the same finding as "could not be judged".
    assert outcome.participating == [1, 2, 3, 4]
    assert outcome.max_deviation_rel > plan.settle_tol_rel


@pytest.mark.asyncio
async def test_railed_fits_are_excluded_so_a_constant_sigma_never_settles():
    """**The single most important correctness detail in this change.**

    A fit railed on the model's R₁ bound reports the same σ every round, and a
    constant passes a stability test perfectly. Four dead channels would
    otherwise declare equilibrium on round three and under-condition the whole
    campaign.
    """
    clock = FakeClock()
    plan = _plan(min_hold_s=0.0, round_period_s=100.0, max_hold_s=1000.0)
    outcome, _last = await drive_settle_phase(
        plan, channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_railed(),
        r1_bound_ohms=RAILED_R1_OHMS,
        sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome != SETTLE_SETTLED
    assert not outcome.settled
    # And it is NOT_EVALUABLE, not CEILING: nothing here could tell us whether σ
    # was moving, which is a different finding from "σ was still moving".
    assert outcome.outcome == SETTLE_NOT_EVALUABLE
    assert outcome.excluded == {ch: EXCLUDED_RAILED for ch in (1, 2, 3, 4)}
    assert outcome.participating == []


@pytest.mark.asyncio
async def test_the_r1_bound_is_what_excludes_them_and_the_driver_threads_it():
    """The same rounds, twice: the bound is the only difference, and it decides.

    Without ``r1_bound_ohms`` reaching ``SettleTracker`` the identical constant
    series settles on round three — which is the defect, demonstrated.
    """
    async def run(bound):
        clock = FakeClock()
        return await drive_settle_phase(
            _plan(min_hold_s=0.0, round_period_s=100.0, max_hold_s=1000.0),
            channels=[1, 2, 3, 4],
            measure_round=_rounds(clock, (1, 2, 3, 4)),
            fits_from=_railed(),
            r1_bound_ohms=bound, sleep=clock.sleep, now=clock.now,
        )

    with_bound, _ = await run(RAILED_R1_OHMS)
    without_bound, _ = await run(None)
    assert with_bound.outcome == SETTLE_NOT_EVALUABLE
    assert without_bound.outcome == SETTLE_SETTLED


@pytest.mark.asyncio
async def test_a_fit_just_above_the_bound_still_counts_as_evidence():
    """Near, not at — but a genuine 4 kΩ fit is not swept up by the tolerance."""
    clock = FakeClock()
    outcome, _last = await drive_settle_phase(
        _plan(min_hold_s=0.0), channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_railed(r1_ohms=RAILED_R1_OHMS * 40),
        r1_bound_ohms=RAILED_R1_OHMS, sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome == SETTLE_SETTLED
    assert outcome.excluded == {}


def test_the_bound_is_read_off_the_circuit_registry_not_written_down():
    assert settle_r1_bound_ohms() == pytest.approx(RAILED_R1_OHMS)


@pytest.mark.asyncio
async def test_the_verdict_carries_the_round_count_and_the_noise_floor():
    clock = FakeClock()
    outcome, last = await drive_settle_phase(
        _plan(min_hold_s=0.0), channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_flat_after(0),
        r1_bound_ohms=RAILED_R1_OHMS, sleep=clock.sleep, now=clock.now,
    )
    record = outcome.as_dict()
    assert record["settle_outcome"] == SETTLE_SETTLED
    assert record["n_rounds"] == outcome.n_rounds >= 3
    assert record["noise_floor_rel"] is not None
    assert record["tolerance_achievable"] is True
    # The last round's raws come back: that is the reading closest to equilibrium
    # and therefore the one worth recording.
    assert set(last) == {1, 2, 3, 4}


@pytest.mark.asyncio
async def test_a_phase_that_measures_nothing_runs_to_its_ceiling():
    """No evidence is never read as evidence of settling."""
    clock = FakeClock()
    outcome, last = await drive_settle_phase(
        _plan(min_hold_s=0.0, round_period_s=100.0, max_hold_s=500.0),
        channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=lambda raws: [RoundFit(channel=ch) for ch in (1, 2, 3, 4)],
        r1_bound_ohms=RAILED_R1_OHMS, sleep=clock.sleep, now=clock.now,
    )
    assert outcome.outcome == SETTLE_NOT_EVALUABLE
    assert last  # the rounds still happened; they just carried no fits


# ── Change 3: the outcome on the real campaign path ──────────────────────────

@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


def _fast_settle_spec(**over) -> CampaignSpec:
    """A campaign that settles, with the clock wound down to test speed."""
    base = dict(equilibration_method="settle", round_period_s=0.01,
                min_hold_s=0.0, max_hold_s=0.2, settle_min_channels=3)
    base.update(over)
    return _spec(**base)


@pytest.mark.asyncio
async def test_a_ceiling_does_not_park_the_campaign(
    connected, tmp_path: Path, monkeypatch
):
    """A slowly-drifting film is an ordinary film, not a reason to stop overnight.

    Parking an unattended run at 3 a.m. because one sample equilibrated slowly is
    the failure mode P0–P1 exists to prevent, so ``ceiling`` must be an outcome
    the campaign proceeds on.
    """
    rounds = {"n": 0}

    def scripted(raws, channels, **_kw):
        """A σ that keeps climbing, whatever the rig would have said."""
        rounds["n"] += 1
        return [RoundFit(channel=int(ch), sigma=1e-4 * (1.5 ** rounds["n"]),
                         r1_ohms=4000.0) for ch in channels]

    # No rig time: the point of this test is the campaign's reaction to the
    # verdict, and the verdict is scripted.
    monkeypatch.setattr(wiring, "build_settle_round_workflow", lambda *a, **k: None)
    monkeypatch.setattr(wiring, "settle_round_fits", scripted)

    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(), manager=connected, data_store=store,
        on_event=events.append,
    )

    verdicts = [e for e in events if e["type"] == "settle_verdict"]
    assert verdicts, "the phase ran but recorded nothing"
    assert all(v["settle_outcome"] == SETTLE_CEILING for v in verdicts)
    # The campaign proceeded: full budget spent, nothing parked.
    assert result.n_trials == 2
    assert not [e for e in events if e["type"] == "park"]
    store.close()


@pytest.mark.asyncio
async def test_a_board_narrower_than_the_criterion_is_announced_not_discovered(
    connected, tmp_path: Path
):
    """It still runs — to the ceiling — but not silently."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(channels=(21,), budget=1),
        manager=connected, data_store=store, on_event=events.append)

    warned = [e for e in events if e["type"] == "settle_unevaluable_board"]
    assert warned and warned[0]["settle_min_channels"] == 3
    assert result.n_trials == 1          # announced, not fatal
    verdicts = [e for e in events if e["type"] == "settle_verdict"]
    assert all(v["settle_outcome"] == SETTLE_NOT_EVALUABLE for v in verdicts)
    store.close()


@pytest.mark.asyncio
async def test_the_verdict_round_count_and_noise_floor_reach_the_trial_record(
    connected, tmp_path: Path
):
    """The event stream dies with the process; the sidecar does not."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(), manager=connected, data_store=store,
        on_event=events.append,
    )

    sidecar = Path(store.run_dir(result.run_id)) / "settle.json"
    assert sidecar.exists()
    records = json.loads(sidecar.read_text(encoding="utf-8"))
    assert records
    for record in records:
        assert record["settle_outcome"] in {
            SETTLE_SETTLED, SETTLE_CEILING, SETTLE_NOT_EVALUABLE}
        assert record["n_rounds"] >= 1
        assert "noise_floor_rel" in record
        assert record["channels"]
    store.close()


@pytest.mark.asyncio
async def test_settle_rounds_re_read_the_films_the_trial_just_cast(
    connected, tmp_path: Path
):
    """A round is a re-read, not a re-cast: same channels, no new well."""
    store = DataStore(tmp_path / "proj")
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(), manager=connected, data_store=store)

    paths = [str(r.get("eis_file_path") or "")
             for r in store.query_measurements(run_id=result.run_id)]
    assert any(wiring.SETTLE_STEP in p for p in paths)
    store.close()


@pytest.mark.asyncio
async def test_a_spec_without_the_new_keys_behaves_exactly_as_before(
    connected, tmp_path: Path
):
    """``wait`` is the default and must remain byte-for-byte today's campaign."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    result = await wiring.run_autonomous_campaign(
        _spec(), manager=connected, data_store=store, on_event=events.append)

    assert result.n_trials == 2
    assert not [e for e in events if e["type"] == "settle_verdict"]
    assert not (Path(store.run_dir(result.run_id)) / "settle.json").exists()
    paths = [str(r.get("eis_file_path") or "")
             for r in store.query_measurements(run_id=result.run_id)]
    assert paths and not any(wiring.SETTLE_STEP in p for p in paths)
    store.close()
