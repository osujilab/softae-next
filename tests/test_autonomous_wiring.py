"""Tests for the agentic execution hook (autonomous_wiring).

Covers the per-trial builder (concrete per-channel deposition + EIS built by the
shared deposition engine, electrode positions resolved from geometry), concrete
volumes actually reaching steps, budget enforcement, and a full headless campaign
on channels 21-24.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.config import loader
from softae.config.loader import pico_for_channel
from softae.core.autonomous_wiring import (
    CampaignSpec,
    build_batch_trial_workflow,
    build_optimizer,
    build_trial_workflow,
    composition_target_objective,
    deposit_step_name,
    eis_impedance_objective_for_channel,
    measure_step_name,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.core.task_catalog import TaskCatalog
from softae.drivers.mock_factory import create_mock_manager
from softae.optimizers import BayesianOptimizer, GridSearchOptimizer

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}

#: A concrete suggestion the engine turns into a trial workflow.
PARAMS = {"vol_p0": 22.0, "vol_p1": 12.0}


def test_a_volume_mode_campaign_resolves_to_impedance_and_minimises_it():
    """Volume mode is a first-class mode, not a campaign missing a composition.

    Every spec in this module is ``vol_params``-only. Without stock identity there is
    no elution and hence no dry thickness, so conductivity is *impossible* rather than
    absent — and mean |Z| is the honest objective, minimised. ``auto`` derives both,
    which is why none of the campaign tests below need to say anything about it.
    """
    from softae.core.autonomous_wiring import resolve_direction, resolve_objective

    spec = _spec()
    kind, reason = resolve_objective(spec)
    assert kind == "mean_abs_z"
    assert "volume mode" in reason
    assert resolve_direction(spec)[0] == "minimize"
    assert build_optimizer(spec)._objective == "minimize"


@pytest.fixture
def catalog() -> TaskCatalog:
    """The real task catalog the shared engine resolves recipe methods from."""
    return TaskCatalog.load_toml(loader.tasks_toml_path())


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="test_campaign",
        channels=(21, 22, 23, 24),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=6,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


# ── Spec normalisation ───────────────────────────────────────────────────────

def test_channels_scalar_normalised_to_tuple():
    assert CampaignSpec(name="x", channels=5).channels == (5,)


def test_empty_channels_rejected():
    with pytest.raises(ValueError):
        CampaignSpec(name="x", channels=())


# ── Per-trial builder (shared deposition engine) ─────────────────────────────

def test_trial_has_deposit_and_measure_per_channel(catalog):
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    names = [s.name for s in wf.setup]
    assert names[0] == "startup_flush"
    for ch in (21, 22, 23, 24):
        assert deposit_step_name(ch) in names
        assert measure_step_name(ch) in names
    assert wf.teardown[0].name == "final_flush"
    # Built by the same engine the HT tab runs — the unity signal.
    assert wf.metadata["source"] == "deposition_engine"


def test_deposit_carries_electrode_position_and_concrete_vols(catalog):
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    dep = next(s for s in wf.setup if s.name == deposit_step_name(21))
    # Electrode position injected as concrete numbers.
    assert isinstance(dep.params["x"], (int, float))
    assert isinstance(dep.params["y"], (int, float))
    # Concrete volumes with per-pump dead volume folded in (deadvols=(10,30));
    # the engine zeroes deadvols (dispense = vol+deadvol is identical).
    assert dep.params["vols"] == [22.0 + 10.0, 12.0 + 30.0]
    assert dep.params["deadvols"] == [0.0, 0.0]
    assert dep.tags.get("channel") == "21"


def test_measure_routes_to_correct_pico(catalog):
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    for ch in (21, 22, 23, 24):
        m = next(s for s in wf.setup if s.name == measure_step_name(ch))
        assert m.instrument == pico_for_channel(ch)  # 21-24 -> pico2
        assert m.params["chan"] == ch


def test_distinct_channels_get_distinct_positions(catalog):
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    p21 = next(s for s in wf.setup if s.name == deposit_step_name(21)).params
    p22 = next(s for s in wf.setup if s.name == deposit_step_name(22)).params
    assert (p21["x"], p21["y"]) != (p22["x"], p22["y"])


def test_time_scale_threaded_into_liquid_handler_steps(catalog):
    # spec.time_scale=0.0 must reach the deposit step so mock dwells are instant.
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    dep = next(s for s in wf.setup if s.name == deposit_step_name(21))
    assert dep.params["time_scale"] == 0.0


# ── Two-phase cast trial ─────────────────────────────────────────────────────

def test_two_phase_inserts_precondition_before_deposit(catalog):
    wf = build_trial_workflow(_spec(two_phase=True), PARAMS, catalog=catalog)
    names = [s.name for s in wf.setup]
    assert names[0] == "startup_flush"
    for ch in (21, 22, 23, 24):
        assert names.index(f"precondition_ch{ch}") < names.index(deposit_step_name(ch))
    assert wf.metadata.get("two_phase") is True


def test_two_phase_startup_uses_line_rate_and_start_vector(catalog):
    wf = build_trial_workflow(
        _spec(two_phase=True, line_flush_rate=400.0, start_flush_uL=(10.0, 20.0)),
        PARAMS, catalog=catalog)
    start = wf.setup[0]
    assert start.params["disp_vols"] == [10.0, 20.0]
    assert start.params["disp_rate"] == 400.0


def test_two_phase_deposit_splits_rate_and_derives_wait(catalog):
    wf = build_trial_workflow(_spec(two_phase=True), PARAMS, catalog=catalog)
    dep = next(s for s in wf.setup if s.name == deposit_step_name(21))
    # Per-pump split rates + derived settle computed at BUILD time (concrete
    # volumes), not deferred to the driver. Two-phase does not fold deadvols.
    assert "disp_rates" in dep.params and len(dep.params["disp_rates"]) == 2
    assert "elution_wait_s" in dep.params
    assert dep.params["vols"] == [22.0, 12.0]
    assert dep.params["deadvols"] == [0.0, 0.0]


def test_two_phase_precondition_carries_split_flush_and_concrete_vols(catalog):
    wf = build_trial_workflow(
        _spec(two_phase=True, flush_factor=2.5), PARAMS, catalog=catalog)
    pre = next(s for s in wf.setup if s.name == "precondition_ch21")
    assert pre.method == "precondition_flush"
    # Per-pump flush rates split from the total line rate (not a single total).
    assert "rate_list" in pre.params and len(pre.params["rate_list"]) == 2
    assert pre.params["flush_factor"] == 2.5
    assert pre.params["vol_list"] == [22.0, 12.0]


# ── Optimizer construction ───────────────────────────────────────────────────

# ── q-batch builder / objective ──────────────────────────────────────────────

def test_build_batch_trial_workflow_casts_distinct_formulations(catalog):
    spec = _spec(batch=True)  # 4 channels
    batch = [
        {"vol_p0": 10.0, "vol_p1": 10.0},
        {"vol_p0": 20.0, "vol_p1": 10.0},
        {"vol_p0": 10.0, "vol_p1": 20.0},
        {"vol_p0": 25.0, "vol_p1": 25.0},
    ]
    wf = build_batch_trial_workflow(spec, batch, catalog=catalog)
    vols_by_ch = {
        int(s.name.rsplit("ch", 1)[1]): tuple(s.params["vols"])
        for s in wf.setup
        if s.name.startswith("deposit_ch")
    }
    assert set(vols_by_ch) == {21, 22, 23, 24}          # one deposit per channel
    assert len(set(vols_by_ch.values())) == 4           # each formulation distinct
    assert wf.metadata["batch"] is True


def test_build_batch_trial_workflow_length_mismatch_raises(catalog):
    spec = _spec(batch=True)  # 4 channels
    with pytest.raises(ValueError, match="must match channel count"):
        build_batch_trial_workflow(spec, [PARAMS, PARAMS], catalog=catalog)


# NOTE: the campaign tests below deliberately say *nothing* about the objective.
# Every spec here is volume-only, which resolves to mean |Z| minimised — see
# ``test_a_volume_mode_campaign_resolves_to_impedance_and_minimises_it`` above. Pinning
# it in each test (as an earlier revision did with a module-global patch) would mask the
# very resolution that test asserts, and would keep passing if `auto` broke entirely.
# Where a *unit* test needs a specific metric it passes ``kind=`` explicitly.


def test_eis_objective_for_channel_reads_only_that_channel():
    import numpy as np
    arr = np.array([[1.0, 2.0, 0.5, 3.0, 4.0]])  # one row, |Z| from last two cols
    results = {measure_step_name(22): [arr], measure_step_name(23): None}
    # Which channel is read, not which metric it is read in — so the cheap metric is
    # named explicitly rather than constructing a castable film.
    assert eis_impedance_objective_for_channel(results, 22, kind="mean_abs_z") > 0.0
    # Unusable / absent must be None, NOT 0.0 — a fabricated 0.0 would be told
    # to the optimizer as a real observation and corrupt the surrogate.
    for ch in (23, 99):   # unusable, absent
        assert eis_impedance_objective_for_channel(results, ch, kind="mean_abs_z") is None


def test_eis_aggregate_objective_returns_none_when_nothing_usable():
    from softae.core.autonomous_wiring import eis_impedance_objective
    assert eis_impedance_objective({}, PARAMS) is None
    assert eis_impedance_objective({measure_step_name(1): None}, PARAMS) is None


# ── Tag-based loop closure (T1.5) ────────────────────────────────────────────
# With a step-tag index, selection is decided by TAGS, never by step names:
# "channel" present AND role (default "sample") == "sample" AND measurement
# (default "primary") == "primary" — SESSION_MAIL #2 point 3 / #3.


def _trace(z_re: float, z_im: float):
    """A minimal usable EIS raw result whose mean |Z| is hypot(z_re, z_im)."""
    import numpy as np
    return [np.array([[1.0, 2.0, 0.5, z_re, z_im]])]


def test_drift_repeat_is_never_selected_as_an_objective_input():
    """The landmine: `geom_drift_repeat_ch3` matches `ch(\\d+)` and carries a
    channel tag, but role="drift_repeat" marks it commissioning data, not a
    trial. Scoring it would hand the optimizer a fabricated observation."""
    from softae.core.autonomous_wiring import eis_impedance_objective

    results = {
        measure_step_name(3): _trace(3.0, 4.0),      # |Z| = 5 — the real sample
        "geom_drift_repeat_ch3": _trace(6.0, 8.0),   # |Z| = 10 — the poison
    }
    step_tags = {
        measure_step_name(3): {"channel": "3", "measurement": "primary"},
        # geometry_series spreads the EIS step's tags then adds its role, so the
        # realistic drift tags carry measurement="primary" too — the ROLE is the
        # load-bearing discriminator, exactly as agreed in MAIL #2 point 3.
        "geom_drift_repeat_ch3": {"channel": "3", "role": "drift_repeat",
                                  "measurement": "primary"},
    }
    agg = eis_impedance_objective(results, {}, kind="mean_abs_z",
                                  step_tags=step_tags)
    assert agg == pytest.approx(5.0)   # 7.5 would mean the repeat contaminated it
    per = eis_impedance_objective_for_channel(results, 3, kind="mean_abs_z",
                                              step_tags=step_tags)
    assert per == pytest.approx(5.0)   # never 10.0: only the primary step's result


def test_bare_channel_tag_defaults_to_selected():
    """{"channel": "5"} with no role/measurement keys → defaults apply → IN.
    The name deliberately shares nothing with `measure_eis_ch*`: with a tag
    index, selection must not read the name at all."""
    from softae.core.autonomous_wiring import eis_impedance_objective

    results = {"probe_alpha": _trace(3.0, 4.0)}
    step_tags = {"probe_alpha": {"channel": "5"}}
    assert eis_impedance_objective(
        results, {}, kind="mean_abs_z", step_tags=step_tags
    ) == pytest.approx(5.0)
    assert eis_impedance_objective_for_channel(
        results, 5, kind="mean_abs_z", step_tags=step_tags
    ) == pytest.approx(5.0)


def test_secondary_measurement_is_never_scored():
    """measurement="secondary" (T2.6 pre-wiring) is recorded, not scored."""
    from softae.core.autonomous_wiring import eis_impedance_objective

    results = {"probe_beta": _trace(3.0, 4.0)}
    step_tags = {"probe_beta": {"channel": "5", "measurement": "secondary"}}
    assert eis_impedance_objective(
        results, {}, kind="mean_abs_z", step_tags=step_tags) is None
    assert eis_impedance_objective_for_channel(
        results, 5, kind="mean_abs_z", step_tags=step_tags) is None


def test_primary_measurement_predicate_vocabulary():
    """The selection predicate, pinned in its one home."""
    from softae.core.autonomous_wiring import is_primary_measurement

    assert is_primary_measurement({"channel": "5"}) is True
    assert is_primary_measurement(
        {"channel": "5", "role": "sample", "measurement": "primary"}) is True
    assert is_primary_measurement(None) is False
    assert is_primary_measurement({}) is False
    assert is_primary_measurement({"role": "sample"}) is False        # no channel
    assert is_primary_measurement({"channel": "5", "role": "blank_short"}) is False
    assert is_primary_measurement(
        {"channel": "5", "measurement": "secondary"}) is False


def test_eis_measure_step_tags_itself_as_a_primary_measurement():
    """The step self-describes as an objective input — explicit, not defaulted."""
    from softae.core.deposition_steps import eis_measure_step

    step = eis_measure_step(7)
    assert step.tags["channel"] == "7"
    assert step.tags["measurement"] == "primary"


def test_built_trial_workflow_carries_the_measurement_tag_index(catalog):
    """The workflow carries {measure step name: tags} for the wiring to close
    the extractors over. Scoped to measurement steps: deposit steps carry a
    channel tag too, so the INDEX (not the predicate) is what keeps them out."""
    wf = build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    idx = wf.metadata["measurement_step_tags"]
    assert set(idx) == {measure_step_name(ch) for ch in (21, 22, 23, 24)}
    for ch in (21, 22, 23, 24):
        tags = idx[measure_step_name(ch)]
        assert tags["channel"] == str(ch)
        assert tags["measurement"] == "primary"
    assert deposit_step_name(21) not in idx


@pytest.mark.parametrize("raw", [
    None,                 # np.asarray(None, float) -> nan WITHOUT raising
    [],                   # empty -> np.mean warns and yields nan
    float("nan"),
    float("inf"),
    [[float("nan"), float("nan")]],
])
def test_scalar_from_eis_raw_rejects_non_finite(raw):
    """No non-finite value may escape as an objective — NaN poisons a GP fit."""
    from softae.core.autonomous_wiring import _scalar_from_eis_raw
    assert _scalar_from_eis_raw(raw) is None


def test_scalar_from_eis_raw_accepts_valid_trace():
    import numpy as np
    from softae.core.autonomous_wiring import _scalar_from_eis_raw
    arr = np.array([[1.0, 2.0, 0.5, 3.0, 4.0]])  # |Z| = hypot(3, 4) = 5
    assert _scalar_from_eis_raw([arr], kind="mean_abs_z") == pytest.approx(5.0)




def test_unmeasured_trial_is_never_told_to_the_optimizer():
    """The P0.1 regression guard: no fabricated observation, DOE objective stays NULL."""
    from unittest.mock import MagicMock
    from softae.core.autonomous_loop import AutonomousLoop

    loop = AutonomousLoop.__new__(AutonomousLoop)   # no hardware needed
    loop._iteration = 3
    assert loop._is_unmeasured(None, PARAMS) is True
    assert loop._is_unmeasured(0.0, PARAMS) is False   # a real 0.0 is a real datum
    assert loop._is_unmeasured(1.5, PARAMS) is False


def test_build_optimizer_kinds():
    assert isinstance(build_optimizer(_spec(optimizer="bayesian")), BayesianOptimizer)
    assert isinstance(build_optimizer(_spec(optimizer="grid")), GridSearchOptimizer)
    with pytest.raises(ValueError):
        build_optimizer(_spec(optimizer="nope"))


def test_build_optimizer_threads_prior_mean_to_bayesian():
    prior = lambda p: 0.0  # noqa: E731
    opt = build_optimizer(_spec(optimizer="bayesian", prior_mean=prior))
    assert opt._prior_mean is prior


def test_build_optimizer_threads_acquisition_and_kappa():
    opt = build_optimizer(_spec(optimizer="bayesian", acquisition="ei", kappa=4.5))
    assert opt._acquisition == "ei"
    assert opt._kappa == 4.5


def test_build_optimizer_threads_batch_strategy():
    from softae.optimizers.batch import KrigingBelieverStrategy
    opt = build_optimizer(_spec(optimizer="bayesian", batch_strategy="kriging_believer"))
    assert isinstance(opt._batch_strategy, KrigingBelieverStrategy)


def test_prior_mean_ignored_warned_for_non_bayesian():
    import structlog
    with structlog.testing.capture_logs() as logs:
        opt = build_optimizer(_spec(optimizer="grid", prior_mean=lambda p: 0.0))
    assert isinstance(opt, GridSearchOptimizer)
    assert any(e.get("event") == "prior_mean_ignored" for e in logs)


# ── End-to-end campaign ──────────────────────────────────────────────────────

@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


@pytest.mark.asyncio
async def test_campaign_runs_budget_and_records_doe(connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    spec = _spec(optimizer="bayesian", budget=5)
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=obj, on_event=events.append,
    )

    # Budget enforced even though Bayesian.suggest() never returns None.
    assert result.n_trials == 5
    assert result.best_params is not None
    # DOE parameters were persisted with objectives.
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 5
    assert all(r["objective_value"] is not None for r in rows)
    # Event stream carried suggestions + results.
    assert sum(e["type"] == "suggestion" for e in events) == 5
    assert sum(e["type"] == "result" for e in events) == 5
    store.close()


@pytest.mark.asyncio
async def test_batch_campaign_tells_per_channel(connected, tmp_path: Path):
    """q-batch mode: each round casts q=4 distinct suggestions (one per channel)
    and tells q objectives; the budget counts individual evaluations."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    spec = _spec(batch=True, budget=8)  # 4 channels → 2 rounds of 4

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store, on_event=events.append,
    )

    assert result.n_trials == 8  # 2 rounds × 4 evaluations
    assert any(e["type"] == "batch_mode" and e["q"] == 4 for e in events)
    # A DOE row per evaluation, tagged with the electrode it was cast on.
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 8
    assert {r["channel"] for r in rows} == {21, 22, 23, 24}
    assert all(r["objective_value"] is not None for r in rows)
    assert sum(e["type"] == "result" for e in events) == 8
    store.close()


@pytest.mark.asyncio
async def test_the_final_round_narrows_to_the_budget_rather_than_overrunning_it(
    connected, tmp_path: Path
):
    """A budget is a bound, not a hint.

    This used to round ``max_iterations`` up to the next multiple of q on the
    grounds that a round is atomic — spending up to q-1 extra electrodes, and the
    hours of anneal that go with them, on a campaign the operator had already
    bounded. Narrowing the last round spends exactly the budget; nothing about
    q-BO requires every round to be the same width.
    """
    store = DataStore(tmp_path / "proj")
    spec = _spec(batch=True, budget=5)  # 4 channels → round(4) then round(1)
    result = await run_autonomous_campaign(spec, manager=connected, data_store=store)
    assert result.n_trials == 5
    assert len(store.query_doe_parameters(run_id=result.run_id)) == 5
    store.close()


@pytest.mark.asyncio
async def test_a_round_narrows_to_the_board_instead_of_straddling_an_exchange(
    connected, tmp_path: Path
):
    """q shrinks to what the current plate can hold; the swap falls *between* rounds.

    The previous design suggested a full q, cast what fit, and then prompted for the
    exchange with half the batch already on the plate — wet films held through an
    operator prompt of unbounded duration, and a constant-liar batch whose members
    were told either side of an arbitrary gap in time, humidity and plate identity.
    Narrowing the round keeps each batch cast, measured and told on one plate.
    """
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    exchanges: list[int] = []
    # 4 channels → q=4; board holds 2 → two rounds of 2, one swap between them.
    spec = _spec(batch=True, budget=4, electrode_capacity=2, equilibration_s=0.0)

    def on_exchange(board: int) -> BoardDecision:
        exchanges.append(board)
        return BoardDecision.PROCEED

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_event=events.append, on_board_exchange=on_exchange,
    )

    assert result.n_trials == 4                      # all 4 measured across boards
    assert exchanges == [1]                          # exactly one board swap
    assert any(e["type"] == "electrode_mode" for e in events)
    assert any(e["type"] == "board_exchange" and e["board"] == 1 for e in events)
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 4
    # Each plate carries exactly one whole round — the evidence that no round was
    # split across the swap.
    assert store.occupied_electrodes(0) == {1, 2}
    assert store.occupied_electrodes(1) == {1, 2}
    store.close()


@pytest.mark.asyncio
async def test_board_exchange_cancel_stops_run_but_keeps_measured(connected, tmp_path: Path):
    """The operator can cancel at the exchange (unintended overflow); samples
    already cast+measured this round are still recorded, then the run stops."""
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    spec = _spec(batch=True, budget=4, electrode_capacity=2, equilibration_s=0.0)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda board: BoardDecision.CANCEL,
    )

    assert result.final_state == "STOPPED"
    assert result.n_trials == 2                       # the 2 that fit were kept
    assert len(store.query_doe_parameters(run_id=result.run_id)) == 2
    store.close()


@pytest.mark.asyncio
async def test_board_exchange_without_handler_stops_instead_of_proceeding(
    connected, tmp_path: Path
):
    """P0.2: no exchange handler must NOT be read as "a fresh plate is in place".

    Swapping a plate is physical; assuming it happened would cast onto a board
    that is still full and destroy occupied single-use wells.
    """
    store = DataStore(tmp_path / "proj")
    spec = _spec(batch=True, budget=4, electrode_capacity=2, equilibration_s=0.0)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,   # no on_board_exchange
    )

    assert result.final_state == "STOPPED"
    assert result.n_trials == 2                  # only the wells that genuinely fit
    # The board pointer must NOT have advanced — no plate was actually installed.
    assert store.current_board_id() == 0
    assert store.occupied_electrodes(1) == set()
    store.close()


@pytest.mark.asyncio
async def test_board_pointer_does_not_advance_when_exchange_cancelled(
    connected, tmp_path: Path
):
    """Cancelling an exchange must leave the pointer on the plate still mounted.

    Advancing it would make the next session believe a fresh, empty board is in
    the machine and cast into the old board's occupied wells.
    """
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    spec = _spec(batch=True, budget=4, electrode_capacity=2, equilibration_s=0.0)

    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda board: BoardDecision.CANCEL,
    )

    assert store.current_board_id() == 0        # still the original plate
    assert store.occupied_electrodes(0) == {1, 2}
    store.close()


@pytest.mark.asyncio
async def test_park_drives_the_rig_safe_and_emits(connected, tmp_path: Path, monkeypatch):
    """P1.2/1.3: a parked campaign must leave the hardware safe, not just stop.

    The whole point of parking unattended is that the head is not left down and
    the heater is not left at setpoint for however long until someone returns.
    """
    import softae.core.autonomous_wiring as aw

    parked: list[dict] = []
    store = DataStore(tmp_path / "proj")

    # Nothing ever measures → consecutive failures → park.
    monkeypatch.setattr(aw, "eis_impedance_objective", lambda r, p: None)
    monkeypatch.setattr(aw, "eis_impedance_objective_for_channel", lambda r, c: None)

    spec = _spec(budget=8)
    events: list[dict] = []
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store, on_event=events.append,
    )

    kinds = [e["type"] for e in events]
    assert "park" in kinds
    assert "safe_park" in kinds
    park_ev = next(e for e in events if e["type"] == "safe_park")
    assert park_ev["ok"] is True                     # mock rig goes safe cleanly
    # Parked well before exhausting the budget.
    assert sum(1 for k in kinds if k == "suggestion") < 8
    store.close()


@pytest.mark.asyncio
async def test_park_writes_a_durable_alert(connected, tmp_path: Path, monkeypatch):
    """P1.5: the reason a run stopped must outlive the process that ran it."""
    import softae.core.autonomous_wiring as aw

    store = DataStore(tmp_path / "proj")
    monkeypatch.setattr(aw, "eis_impedance_objective", lambda r, p: None)
    monkeypatch.setattr(aw, "eis_impedance_objective_for_channel", lambda r, c: None)

    result = await run_autonomous_campaign(
        _spec(budget=8), manager=connected, data_store=store,
    )
    store.close()

    # Reopen: this is the morning-after query.
    with DataStore(tmp_path / "proj") as ds2:
        alerts = ds2.query_alerts(run_id=result.run_id)
        assert len(alerts) == 1
        assert alerts[0]["kind"] == "park"
        assert alerts[0]["severity"] == "critical"
        assert "parked" in alerts[0]["message"]
        assert alerts[0]["details"]["safe_park_ok"] is True


@pytest.mark.asyncio
async def test_campaign_finalizes_run_row(connected, tmp_path: Path):
    """P0.3: a finished campaign must not leave experiments.status = 'running'."""
    store = DataStore(tmp_path / "proj")
    spec = _spec(budget=2)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
    )

    row = store._conn.execute(
        "SELECT status, finished_at FROM experiments WHERE run_id = ?",
        (result.run_id,),
    ).fetchone()
    assert row["status"] != "running"
    assert row["finished_at"] is not None
    store.close()


@pytest.mark.asyncio
async def test_campaign_finalizes_run_row_on_crash(connected, tmp_path: Path, monkeypatch):
    """A crashed campaign is recorded as 'error', not left looking still-running."""
    import softae.core.autonomous_wiring as aw

    store = DataStore(tmp_path / "proj")
    spec = _spec(budget=2)

    def boom(*a, **k):
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(aw, "build_optimizer", boom)
    with pytest.raises(RuntimeError, match="optimizer exploded"):
        await run_autonomous_campaign(spec, manager=connected, data_store=store)

    row = store._conn.execute(
        "SELECT run_id, status, finished_at FROM experiments ORDER BY started_at DESC"
    ).fetchone()
    assert row["status"] == "error"
    assert row["finished_at"] is not None
    store.close()


@pytest.mark.asyncio
async def test_single_point_consumes_electrodes_and_swaps(connected, tmp_path: Path):
    """Board management applies to single-point campaigns too: one fresh
    electrode per trial, a board swap every ``capacity`` samples."""
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    swaps: list[int] = []
    spec = _spec(budget=5, electrode_capacity=2, equilibration_s=0.0)  # single-point

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda b: (swaps.append(b) or BoardDecision.PROCEED),
    )

    assert result.n_trials == 5
    assert swaps == [1, 2]                            # boards fill every 2 electrodes
    # Electrodes are board-relative (reset each board): 1,2 | 1,2 | 1
    chans = [r["channel"] for r in store.query_doe_parameters(run_id=result.run_id)]
    assert sorted(chans) == [1, 1, 1, 2, 2]
    store.close()


@pytest.mark.asyncio
async def test_occupancy_recorded_during_board_campaign(connected, tmp_path: Path):
    """A board-mode campaign persists single-use well occupancy (board 0)."""
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    spec = _spec(budget=3, electrode_capacity=8, equilibration_s=0.0)  # single-point
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda b: BoardDecision.PROCEED,
    )
    assert store.occupied_electrodes(0) == {1, 2, 3}  # electrodes cast in order
    store.close()


@pytest.mark.asyncio
async def test_resume_fresh_board_starts_clean(connected, tmp_path: Path):
    """On resume the operator says the plate is FRESH → new board id, clean wells."""
    from softae.core.autonomous_loop import BoardCheck, BoardDecision
    store = DataStore(tmp_path / "proj")
    store.record_electrode_cast(0, 1)  # a prior session used board 0, wells 1,2
    store.record_electrode_cast(0, 2)
    checks: list[tuple[int, set[int]]] = []

    spec = _spec(budget=2, electrode_capacity=8, equilibration_s=0.0)
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_check=lambda bid, occ: (checks.append((bid, occ)) or BoardCheck.FRESH),
        on_board_exchange=lambda b: BoardDecision.PROCEED,
    )
    assert checks == [(0, {1, 2})]
    assert store.occupied_electrodes(1) == {1, 2}   # cast on a fresh board id
    assert store.occupied_electrodes(0) == {1, 2}   # old board untouched
    assert store.current_board_id() == 1            # pointer advanced durably


@pytest.mark.asyncio
async def test_fresh_board_pointer_persists_without_casts(tmp_path: Path):
    """A FRESH decision is durable even when nothing is cast on the new plate.

    Exercises ``_prepare_electrode_allocator`` directly: it is the moment the
    swap is decided, and the regression is precisely "swap, then shut down
    before any cast lands".
    """
    from softae.core.autonomous_loop import BoardCheck
    from softae.core.autonomous_wiring import _prepare_electrode_allocator

    store = DataStore(tmp_path / "proj")
    store.record_electrode_cast(0, 1)

    spec = _spec(budget=2, electrode_capacity=8, equilibration_s=0.0)
    alloc = await _prepare_electrode_allocator(
        spec, store,
        lambda bid, occ: BoardCheck.FRESH,
        lambda *a, **k: None,
    )
    assert alloc is not None and alloc.board_index == 1
    store.close()

    # Reopen: the swap must not be forgotten (the bug this guards).
    with DataStore(tmp_path / "proj") as ds2:
        assert ds2.current_board_id() == 1
        assert ds2.occupied_electrodes(1) == set()


@pytest.mark.asyncio
async def test_full_board_resume_persists_advanced_pointer(tmp_path: Path):
    """RESUME onto a physically full board advances *and* persists the pointer."""
    from softae.core.autonomous_loop import BoardCheck
    from softae.core.autonomous_wiring import _prepare_electrode_allocator

    store = DataStore(tmp_path / "proj")
    for e in range(1, 5):                      # fill a 4-electrode board
        store.record_electrode_cast(0, e)

    spec = _spec(budget=2, electrode_capacity=4, equilibration_s=0.0)
    alloc = await _prepare_electrode_allocator(
        spec, store,
        lambda bid, occ: BoardCheck.RESUME,
        lambda *a, **k: None,
    )
    assert alloc is not None and alloc.board_index == 1  # rolled to a fresh board
    assert store.current_board_id() == 1
    store.close()


@pytest.mark.asyncio
async def test_resume_same_board_continues_past_used(connected, tmp_path: Path):
    """RESUME → keep the same board id, cast into the next unused wells."""
    from softae.core.autonomous_loop import BoardCheck, BoardDecision
    store = DataStore(tmp_path / "proj")
    for e in (1, 2, 3):
        store.record_electrode_cast(0, e)

    spec = _spec(budget=2, electrode_capacity=8, equilibration_s=0.0)
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_check=lambda bid, occ: BoardCheck.RESUME,
        on_board_exchange=lambda b: BoardDecision.PROCEED,
    )
    assert store.occupied_electrodes(0) == {1, 2, 3, 4, 5}  # continued at 4


@pytest.mark.asyncio
async def test_resume_cancel_aborts_before_casting(connected, tmp_path: Path):
    """CANCEL at the board-freshness check stops the run before any cast."""
    from softae.core.autonomous_loop import BoardCheck
    store = DataStore(tmp_path / "proj")
    store.record_electrode_cast(0, 1)

    spec = _spec(budget=2, electrode_capacity=8, equilibration_s=0.0)
    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_check=lambda bid, occ: BoardCheck.CANCEL,
    )
    assert result.final_state == "STOPPED"
    assert result.n_trials == 0
    assert store.occupied_electrodes(0) == {1}  # nothing new cast


@pytest.mark.asyncio
async def test_resume_headless_defaults_to_resume(connected, tmp_path: Path):
    """With no board-check handler (headless), the safe default is resume — the
    campaign never silently re-casts into an occupied well."""
    from softae.core.autonomous_loop import BoardDecision
    store = DataStore(tmp_path / "proj")
    store.record_electrode_cast(0, 1)
    store.record_electrode_cast(0, 2)

    spec = _spec(budget=1, electrode_capacity=8, equilibration_s=0.0)
    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        on_board_exchange=lambda b: BoardDecision.PROCEED,
    )
    assert store.occupied_electrodes(0) == {1, 2, 3}  # resumed at electrode 3


@pytest.mark.asyncio
async def test_seed_observations_warm_start_the_optimizer(connected, tmp_path: Path):
    """Seed observations are told to the optimizer before the loop and appear in
    the recorded history (physically/prior-informed warm-start)."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    seeds = (({"vol_p0": 22.0, "vol_p1": 12.0}, 0.99),)
    spec = _spec(optimizer="bayesian", budget=3, seed_observations=seeds)
    obj = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=obj, on_event=events.append,
    )

    assert any(e["type"] == "warm_start" and e["n_seed"] == 1 for e in events)
    # The seed observation is present in the optimizer history alongside the run.
    assert (seeds[0][0], seeds[0][1]) in result.history
    assert result.n_trials == 3  # budget counts loop trials, not seeds
    store.close()


@pytest.mark.asyncio
async def test_emits_maturity_warning_for_untested_methods(connected, tmp_path: Path):
    """Composite deposit methods are 'tested', not 'validated' -> warn, proceed."""
    store = DataStore(tmp_path / "proj_mat")
    events: list[dict] = []
    spec = _spec(optimizer="grid", budget=1, expected_maturity="validated")

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
        on_event=events.append,
    )
    warnings = [e for e in events if e["type"] == "maturity_warning"]
    warned = {w["method"] for w in warnings}
    # The composite deposition methods are catalogued as 'tested'.
    assert "single_drop_simul" in warned
    assert all(w["expected"] == "validated" for w in warnings)
    # Warn-and-proceed: the run still completed.
    assert result.n_trials == 1
    store.close()


@pytest.mark.asyncio
async def test_suggestion_reaches_hardware(connected, tmp_path: Path):
    """A suggested volume must actually be dispensed as a concrete amount."""
    store = DataStore(tmp_path / "proj2")
    spec = _spec(optimizer="grid", budget=2)
    before = connected.get("syringe")._dispensed.get(0, 0.0)

    await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
    )
    # Pump 0 dispensed a real (numeric) volume across the trials.
    assert connected.get("syringe")._dispensed.get(0, 0.0) > before
    store.close()


@pytest.mark.asyncio
async def test_two_phase_campaign_runs_end_to_end(connected, tmp_path: Path):
    """A two-phase campaign runs the precondition+deposit trial and records DOE.

    Exercises the engine path: each trial's concrete per-pump volumes drive the
    build-time rate split + derived settle, same as HT.
    """
    store = DataStore(tmp_path / "proj_tp")
    events: list[dict] = []
    spec = _spec(optimizer="grid", budget=2, two_phase=True)
    before = connected.get("syringe")._dispensed.get(0, 0.0)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
        on_event=events.append,
    )
    assert result.n_trials == 2
    assert connected.get("syringe")._dispensed.get(0, 0.0) > before
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 2
    store.close()


@pytest.mark.asyncio
async def test_approval_gate_can_stop(connected, tmp_path: Path):
    """A rejecting approval_fn halts the campaign at the first trial."""
    store = DataStore(tmp_path / "proj3")
    spec = _spec(optimizer="grid", budget=5, auto_approve=False)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
        approval_fn=lambda i, p: False,  # veto everything
    )
    assert result.n_trials == 0
    assert result.final_state == "STOPPED"
    store.close()


# ── Null-object default for the optional purge harness (T1.6) ────────────────
#
# Purging is optional: a rig with no `[purge]` schedule is configured, not
# broken. Absence therefore resolves to a NullPurgeRunner rather than None.
# The null absorbs a *side effect* only — it must never invent an answer about
# the rig, and it must not change what a campaign without purging does.


class _FakeScheduler:
    """A purge scheduler with nothing ever owed. Records that it was asked."""

    def __init__(self) -> None:
        self.asked = 0

    def due(self):
        self.asked += 1
        return None  # nothing owed → the real runner returns before any actuation


def test_resolve_purge_runner_without_a_scheduler_returns_a_null_not_none():
    from softae.core.autonomous_wiring import _resolve_purge_runner
    from softae.core.purge_runner import NullPurgeRunner

    runner = _resolve_purge_runner(create_mock_manager(config={}))

    assert isinstance(runner, NullPurgeRunner)
    assert runner.performs_purges is False


def test_resolve_purge_runner_never_caches_the_null_onto_the_syringe():
    """A cached null would outlive the absence it stands for.

    ``_resolve_purge_runner`` returns any runner already published on the
    syringe, so a null left there would be found forever after — a host that
    attaches a real scheduler later in the same process could never take effect.
    """
    from softae.core.autonomous_wiring import _resolve_purge_runner
    from softae.core.purge_runner import PurgeRunner

    mgr = create_mock_manager(config={})
    syringe = mgr.get("syringe")

    assert _resolve_purge_runner(mgr).performs_purges is False
    assert getattr(syringe, "purge_runner", None) is None

    syringe.purge_scheduler = _FakeScheduler()
    assert isinstance(_resolve_purge_runner(mgr), PurgeRunner)


def test_resolve_purge_runner_returns_a_null_when_the_syringe_is_unreadable():
    from softae.core.autonomous_wiring import _resolve_purge_runner
    from softae.core.purge_runner import NullPurgeRunner

    class _NoSyringe:
        def get(self, name):
            raise KeyError(name)

    runner = _resolve_purge_runner(_NoSyringe())
    assert isinstance(runner, NullPurgeRunner)
    assert runner.performs_purges is False


def test_null_purge_runner_maybe_purge_is_a_silent_no_op():
    """Called with the campaign's exact in-run flags: no raise, nothing claimed."""
    from softae.core.purge_runner import NullPurgeRunner

    outcome = NullPurgeRunner().maybe_purge(
        context="step:eis_ch21", owns_rig=True,
        allow_positioning=True, end_at_idle_rest=False,
    )

    assert outcome.performed is False
    assert outcome.dry_run is False       # a caller showing outcomes stays silent
    assert outcome.volumes_uL == {}
    assert "not configured" in outcome.skipped_reason


def test_null_purge_runner_logs_its_absence_once_at_construction():
    import structlog

    from softae.core.purge_runner import NullPurgeRunner

    with structlog.testing.capture_logs() as logs:
        runner = NullPurgeRunner()
        for _ in range(5):
            runner.maybe_purge(context="step:x")

    assert [e for e in logs if e.get("event") == "purge_runner_absent"] != []
    assert len(logs) == 1  # once at construction, not once per window


def test_null_purge_runner_refuses_to_answer_questions_about_the_rig():
    """It may absorb a purge; it may not invent rig state.

    Pose, park state and idle rest are facts about hardware. A null that
    returned a plausible default for them would make "nobody asked" look
    exactly like "measured, and the answer was no".
    """
    from softae.core.purge_runner import NullPurgeRunner

    runner = NullPurgeRunner()
    for attribute in ("idle_rest", "_pose", "_scheduler", "_blocking_reason"):
        with pytest.raises(NotImplementedError):
            getattr(runner, attribute)

    # Dunders still resolve normally, so ordinary duck typing is unaffected.
    assert hasattr(runner, "__await__") is False
    assert repr(runner)


@pytest.mark.asyncio
async def test_campaign_without_a_purge_scheduler_opens_no_purge_window(
    connected, tmp_path: Path, monkeypatch
):
    """Behavioural no-change: the null absorbs calls it is never even given.

    With no scheduler the hook stays unset, so the executor opens no concurrent
    window — no thread, no task, and the null's ``maybe_purge`` is never
    reached. The campaign itself is unchanged.
    """
    from softae.core import purge_runner as purge_mod

    calls: list[dict] = []
    monkeypatch.setattr(
        purge_mod.NullPurgeRunner, "maybe_purge",
        lambda self, **kw: calls.append(kw) or purge_mod.PurgeOutcome(),
    )

    store = DataStore(tmp_path / "proj_nullpurge")
    result = await run_autonomous_campaign(
        _spec(optimizer="grid", budget=1), manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
    )

    assert calls == []
    assert result.n_trials == 1
    assert len(store.query_doe_parameters(run_id=result.run_id)) == 1
    store.close()


@pytest.mark.asyncio
async def test_campaign_with_a_purge_scheduler_still_opens_purge_windows(
    connected, tmp_path: Path, monkeypatch
):
    """The positive control for the test above — the gate is not stuck off."""
    from softae.core import purge_runner as purge_mod

    scheduler = _FakeScheduler()
    connected.get("syringe").purge_scheduler = scheduler

    calls: list[dict] = []
    real = purge_mod.PurgeRunner.maybe_purge
    monkeypatch.setattr(
        purge_mod.PurgeRunner, "maybe_purge",
        lambda self, **kw: (calls.append(kw), real(self, **kw))[1],
    )

    store = DataStore(tmp_path / "proj_realpurge")
    result = await run_autonomous_campaign(
        _spec(optimizer="grid", budget=1), manager=connected, data_store=store,
        objective_extractor=composition_target_objective({"vol_p0": 20.0, "vol_p1": 10.0}),
    )

    assert result.n_trials == 1
    assert calls, "a co-runnable step should have offered a purge window"
    assert all(c["owns_rig"] and not c["end_at_idle_rest"] for c in calls)
    assert scheduler.asked >= 1        # the real runner asked; nothing was owed
    store.close()
