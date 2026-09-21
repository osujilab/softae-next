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
    SETTLE_CRITERION_DEVIATION,
    SETTLE_CRITERION_RATE,
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
    # `rh_stability_pct=None` — the RH stability gate is OFF for every test in
    # this file. These tests are about σ, the floor, the ceiling and the R₁
    # bound; the gate that also watches the room is `SettlePlan`'s shipped
    # default (pinned below) and is exercised end to end in
    # `test_rh_equilibrate_stability_gate.py`.
    base = dict(round_period_s=50.0, min_hold_s=100.0, max_hold_s=10_000.0,
                settle_n_rounds=3, settle_min_channels=3,
                rh_stability_pct=None)
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
    # And the RH stability gate ships ON: its failure mode is "held longer,
    # recorded ceiling", while OFF's is "measured under moving humidity".
    assert plan.rh_stability_pct == pytest.approx(1.5)


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
    # RH gate off: with no workflow there are no `conditions` rows either, so the
    # gate would (correctly) report `not_evaluable` and this test would stop
    # being about the σ ceiling it was written for.
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(rh_stability_pct=None), manager=connected,
        data_store=store, on_event=events.append,
    )

    verdicts = [e for e in events if e["type"] == "settle_verdict"]
    assert verdicts, "the phase ran but recorded nothing"
    assert all(v["settle_outcome"] == SETTLE_CEILING for v in verdicts)
    # The campaign proceeded: full budget spent, nothing parked.
    assert result.n_trials == 2
    assert not [e for e in events if e["type"] == "park"]
    store.close()


# T11.52: this test was `..._is_announced_not_discovered` and watched the retired
# `settle_unevaluable_board` emitter. **Re-pointing it at the replacement would
# have been vacuous.** `settle_min_channels_ignored` fires on
# `spec.settle_min_channels is not SPEC_UNSET` — a property of the *spec*, which
# `_fast_settle_spec` sets for every test in this file — so it has nothing to say
# about board width: run verbatim with the default four channels, the old body
# emitted the same event, returned the same `n_trials`, and recorded the same
# `not_evaluable`. Every assertion passed without a narrow board, which is
# `SUBAGENT_RULES` §3.1(e). The ignored-key claim is covered directly, both arms,
# by `test_autonomous_wiring.py::test_campaign_start_emits_settle_min_channels_
# ignored_when_the_spec_sets_it` and its `..._when_unset` twin, so it is not
# re-asserted here.
#
# What is kept is the claim T11.33 actually made and nothing pinned: a board
# narrower than the criterion's channel count is *judged*, well by well, rather
# than refused board-wide. That is what `board_minimum=None` buys, and the
# assertion below could not pass under the behaviour it replaced — the old path
# returned `not_evaluable` with `participating == []` for exactly this board.
@pytest.mark.asyncio
async def test_a_board_narrower_than_the_criterion_is_judged_well_by_well(
    connected, tmp_path: Path
):
    """One channel is a board like any other: judged, not refused.

    ``settle_min_channels=3`` against a single-channel board used to be a
    board-level refusal. After T11.33 the campaign path passes
    ``board_minimum=None``, so the one well present is tracked on its own merits
    and reaches an ordinary verdict.
    """
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    # `max_hold_s` must clear `settle_n_rounds` rounds or the hold ends before
    # the criterion has any history and *every* board — narrow or wide — reports
    # `not_evaluable` for want of rounds, which would make the assertion below
    # about the ceiling rather than about board width. RH gate off for the same
    # reason the σ-ceiling test above turns it off: the subject here is the
    # channel count, not the room.
    result = await wiring.run_autonomous_campaign(
        _fast_settle_spec(channels=(21,), budget=1, max_hold_s=120.0,
                          rh_stability_pct=None),
        manager=connected, data_store=store, on_event=events.append)

    assert result.n_trials == 1          # judged, not fatal
    verdicts = [e for e in events if e["type"] == "settle_verdict"]
    assert verdicts, "the phase ran but recorded nothing"
    for verdict in verdicts:
        # The load-bearing assertion: the sole well participated. Board-level
        # refusal produced `participating == []` here, so this line is the one
        # that could not have passed before.
        assert verdict["participating"] == [21]
        # Not merely "something other than `not_evaluable`": the well is judged
        # on its own merits and the mock's rounds are identical, so the deviation
        # criterion has one deterministic answer.
        assert verdict["settle_outcome"] == SETTLE_SETTLED
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


# ── T11.7: the criterion and the time axis reach the tracker ─────────────────
#
# `SettlePlan` has carried `criterion`/`rate_tol_dec_per_h` since T11.2 and
# `drive_settle_phase` read neither, so a plan asking for the rate criterion
# built a tracker on the DEVIATION default. That is a *silent substitution*
# rather than a conservative failure: the phase can return `settled` under a
# criterion it never ran. These tests pin both kwargs, and each is written so it
# cannot pass while its kwarg is missing.

#: σ creeping 3 % per round, at a 600 s period. The ratio is chosen so the two
#: criteria **disagree on this exact series**, which is what makes the pin below
#: a pin: over a 3-round deviation window the worst deviation is 2.97 %, well
#: inside the 10 % tolerance, while 3 %/round for 600 s is 0.077 dec/h — half
#: again the 0.05 dec/h band. A series both criteria refused, or both accepted,
#: would pass with `criterion` still dropped on the floor.
CREEP_RATIO = 1.03
CREEP_PERIOD_S = 600.0
CREEP_BAND_DEC_PER_H = 0.05


def _creeping(channels=(1, 2, 3, 4)):
    """Geometric creep — a slope with no scatter, so only the gate decides."""

    def fits_from(raws):
        index = next(iter(raws.values()), 0)
        return [RoundFit(channel=ch, sigma=1e-4 * (CREEP_RATIO ** index),
                         r1_ohms=4000.0) for ch in channels]

    return fits_from


async def _drive_creep(*, on_round=None, **plan_over):
    clock = FakeClock()
    outcome, _last = await drive_settle_phase(
        _plan(min_hold_s=0.0, round_period_s=CREEP_PERIOD_S,
              max_hold_s=CREEP_PERIOD_S * 12, **plan_over),
        channels=[1, 2, 3, 4],
        measure_round=_rounds(clock, (1, 2, 3, 4)),
        fits_from=_creeping(),
        r1_bound_ohms=RAILED_R1_OHMS, sleep=clock.sleep, now=clock.now,
        on_round=on_round,
    )
    return outcome


@pytest.mark.asyncio
async def test_a_rate_plan_routes_on_rate_where_deviation_would_have_settled():
    """The same rounds, twice: the criterion is the only difference, and it decides.

    Its own positive control, on the ``r1_bound_ohms`` precedent two sections
    up. With ``criterion`` unwired the rate plan builds a deviation tracker and
    both runs return ``settled`` — so this test fails on the defect rather than
    agreeing with it, which an outcome-only assertion on a steeply moving σ
    would not have done.
    """
    deviation = await _drive_creep()
    rate = await _drive_creep(criterion=SETTLE_CRITERION_RATE,
                              rate_tol_dec_per_h=CREEP_BAND_DEC_PER_H)

    assert deviation.outcome == SETTLE_SETTLED, (
        "the deviation criterion should certify a 2.97 % window at 10 % — if it "
        "does not, the series no longer discriminates and the pin below is void")
    assert rate.outcome == SETTLE_CEILING
    assert not rate.settled
    # Evaluable and said no, which is the finding that separates a criterion
    # that ran from one that could not be judged at all.
    assert rate.participating == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_the_rate_window_is_paced_by_the_clock_not_by_an_assumed_spacing():
    """``t_s`` is load-bearing under the rate criterion, and fatal by its absence.

    ``rate_check`` refuses a window carrying any unknown round time rather than
    guessing at the spacing, so an unwired ``t_s`` does not merely degrade the
    verdict — every round reports "0 of 7 round times are known" and the phase
    ends ``not_evaluable``, wearing the same word a board with too few channels
    produces. Asserted on the reason text because that is the surface
    ``on_round`` actually receives.
    """
    reasons: list[str] = []
    outcome = await _drive_creep(
        criterion=SETTLE_CRITERION_RATE,
        rate_tol_dec_per_h=CREEP_BAND_DEC_PER_H,
        on_round=lambda _i, check: reasons.append(
            "" if check is None else (check.reason or "")))

    spoken = " ".join(reasons)
    assert "needs a time axis" not in spoken, (
        f"the rate criterion never got a clock: {spoken!r}")
    # A rate verdict quotes a slope in ln/h; the deviation criterion never does.
    assert "ln/h" in spoken
    assert outcome.outcome == SETTLE_CEILING
    # And no deviation was computed, because no deviation window was judged.
    assert outcome.max_deviation_rel is None


# ── T11.7: the campaign says which criterion it is running ───────────────────

async def _settle_mode_event(spec, connected, tmp_path, monkeypatch) -> dict:
    """Run a campaign far enough to hear ``settle_mode``, without rig time."""
    monkeypatch.setattr(wiring, "build_settle_round_workflow", lambda *a, **k: None)
    monkeypatch.setattr(
        wiring, "settle_round_fits",
        lambda raws, channels, **_kw: [
            RoundFit(channel=int(ch), sigma=1e-4, r1_ohms=4000.0)
            for ch in channels])

    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    try:
        await wiring.run_autonomous_campaign(
            spec, manager=connected, data_store=store, on_event=events.append)
    finally:
        store.close()

    modes = [e for e in events if e["type"] == "settle_mode"]
    assert len(modes) == 1, f"expected one settle_mode event, got {len(modes)}"
    return modes[0]


@pytest.mark.asyncio
async def test_settle_mode_announces_the_deviation_default_and_its_absent_band(
    connected, tmp_path: Path, monkeypatch
):
    event = await _settle_mode_event(
        _fast_settle_spec(budget=1, rh_stability_pct=None),
        connected, tmp_path, monkeypatch)

    assert event["criterion"] == SETTLE_CRITERION_DEVIATION
    # Present and `None`, not absent: a default-criterion plan carries no band —
    # `SettlePlan` refuses to let it — and "no band" must be readable as such
    # rather than inferred from a missing key.
    assert "rate_tol_dec_per_h" in event
    assert event["rate_tol_dec_per_h"] is None
    # The seven flat fields still travel, unchanged by the two new kwargs.
    assert event["settle_min_channels"] == 3
    assert event["min_hold_s"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_settle_mode_announces_a_rate_plans_criterion_and_band(
    connected, tmp_path: Path, monkeypatch
):
    """An operator watching a rate run can see that it is running the rate gate.

    Reached through the structural ``run_plan`` spelling because that is the
    *only* spelling that carries a criterion: ``CampaignSpec`` has no flat
    ``criterion``/``rate_tol_dec_per_h`` pair, which is precisely why these two
    keys are named as explicit kwargs on the emit rather than added to
    ``_SETTLE_FIELDS``.
    """
    event = await _settle_mode_event(
        _spec(budget=1, run_plan=RunPlan.pointwise(settle=_plan(
            min_hold_s=0.0, round_period_s=0.01, max_hold_s=0.2,
            criterion=SETTLE_CRITERION_RATE,
            rate_tol_dec_per_h=CREEP_BAND_DEC_PER_H))),
        connected, tmp_path, monkeypatch)

    assert event["criterion"] == SETTLE_CRITERION_RATE
    assert event["rate_tol_dec_per_h"] == pytest.approx(CREEP_BAND_DEC_PER_H)
