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
    campaign_spec_fingerprint,
    composition_target_objective,
    deposit_step_name,
    eis_impedance_objective_for_channel,
    measure_step_name,
    optimizer_tuning_identity,
    resolve_optimizer_tuning,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.core.phase_setpoints import (
    BASELINE_ABSENT_EVENT,
    BASELINE_EVENT,
    PhaseSetpoints,
)
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


def test_campaign_spec_conditions_defaults_to_none_and_round_trips():
    """The campaign-level ``[conditions]`` baseline is a field, not a phase's.

    Its absence was the whole T11.28 gap: ``campaign_spec_fields`` already
    registers a ``"conditions"`` codec and ``spec_from_dict`` derives its
    accepted-key list from ``dataclasses.fields(CampaignSpec)``, so until the
    field existed a ``[conditions]`` block in a campaign TOML was rejected as an
    unknown field by the very loader written to decode it. ``None`` is a real
    answer here — "nothing was commanded" — not a missing one.
    """
    from dataclasses import fields as dataclass_fields

    assert "conditions" in {f.name for f in dataclass_fields(CampaignSpec)}
    assert _spec().conditions is None

    baseline = PhaseSetpoints("floor", temp_setpoint_C=25.0, rh_setpoint_pct=30.0)
    assert _spec(conditions=baseline).conditions is baseline


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


def test_build_trial_workflow_threads_spec_conditions_to_the_engine_as_baseline(
    catalog, monkeypatch
):
    """The campaign baseline reaches the one marshaller, or it reaches nothing.

    ``deposition_recipe.build_deposition_workflow`` has accepted and threaded
    ``baseline=`` since T11.28's engine half landed, and this call site was the
    missing link — so a ``[conditions]`` block that parsed cleanly onto the spec
    still commanded the chamber nothing. Asserted at the kwarg rather than at
    the emitted steps on purpose: what the engine *does* with a baseline is
    ``test_deposition_recipe.py``'s subject, and reproving it here would couple
    this test to step names it does not own.
    """
    from softae.core import autonomous_wiring as _wiring

    seen: dict = {}
    real = _wiring.build_deposition_workflow

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(_wiring, "build_deposition_workflow", spy)

    baseline = PhaseSetpoints("floor", temp_setpoint_C=25.0, rh_setpoint_pct=30.0)
    build_trial_workflow(_spec(conditions=baseline), PARAMS, catalog=catalog)
    assert seen["baseline"] is baseline

    seen.clear()
    build_trial_workflow(_spec(), PARAMS, catalog=catalog)
    assert seen["baseline"] is None, (
        "a campaign with no [conditions] must reach the engine as an explicit "
        "None — the kwarg is always passed, so 'no baseline' is stated rather "
        "than left to the engine's default")


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


# ── T11.58 — n_initial as the eighth tuning knob ─────────────────────────────
#
# The warm-up length used to be the SOLE literal in `build_optimizer`'s bayesian
# call, `min(5, budget)`. These pin the three things that made that a problem:
# it is now stated rather than derived, it reaches the resume fingerprint, and a
# stated value is no longer clamped into silence by the same `min()` that spells
# an init-phase dry run and a mis-set campaign identically.

@pytest.fixture
def no_site_defaults(monkeypatch):
    """Silence the TOML layer so a test measures the spec field alone.

    The shipped ``[optimizer]`` section carries no ``n_initial`` key today, so
    these tests would pass without this — which is exactly why it is here: a site
    default landing later must not quietly retune what these assert.
    """
    monkeypatch.setattr(loader, "optimizer_tuning", lambda: {})
    monkeypatch.setattr(loader, "feasibility_config", lambda: {})


WARMUP_EVENT = "optimizer_warmup_spans_entire_budget"


def _warmup_events(spec) -> list[dict]:
    """Build *spec*'s optimizer, returning only the warm-up-span log records."""
    import structlog
    with structlog.testing.capture_logs() as logs:
        build_optimizer(spec)
    return [e for e in logs if e.get("event") == WARMUP_EVENT]


@pytest.mark.parametrize("budget,expected", [(6, 5), (5, 5), (3, 3), (1, 1)])
def test_resolve_optimizer_tuning_n_initial_unset_is_todays_min_of_five_and_budget(
        budget, expected, no_site_defaults):
    """Unset keeps the pre-T11.58 literal exactly — no new magic number."""
    assert resolve_optimizer_tuning(_spec(budget=budget))["n_initial"] == expected


@pytest.mark.parametrize("budget,expected", [(6, 5), (3, 3)])
def test_build_optimizer_n_initial_unset_matches_the_pre_t11_58_literal(
        budget, expected, no_site_defaults):
    opt = build_optimizer(_spec(optimizer="bayesian", budget=budget))
    assert opt._n_initial == expected == min(5, budget)


def test_build_optimizer_threads_an_explicit_n_initial_verbatim(no_site_defaults):
    opt = build_optimizer(_spec(optimizer="bayesian", budget=6, n_initial=2))
    assert opt._n_initial == 2


def test_an_explicit_n_initial_above_budget_reaches_the_optimizer_unclamped(
        no_site_defaults):
    """The ruling's substance: `min()` must not repair a stated value.

    Clamping to 6 here would make a deliberate all-warm-up run indistinguishable
    from a campaign that set the warm-up too long.
    """
    opt = build_optimizer(_spec(optimizer="bayesian", budget=6, n_initial=9))
    assert opt._n_initial == 9


def test_a_site_default_supplies_n_initial_when_the_spec_is_silent(monkeypatch):
    monkeypatch.setattr(loader, "optimizer_tuning", lambda: {"n_initial": 3})
    monkeypatch.setattr(loader, "feasibility_config", lambda: {})
    assert build_optimizer(_spec(optimizer="bayesian", budget=6))._n_initial == 3


def test_the_spec_field_beats_the_site_default(monkeypatch):
    monkeypatch.setattr(loader, "optimizer_tuning", lambda: {"n_initial": 3})
    monkeypatch.setattr(loader, "feasibility_config", lambda: {})
    opt = build_optimizer(_spec(optimizer="bayesian", budget=6, n_initial=4))
    assert opt._n_initial == 4


def test_a_warmup_spanning_the_whole_budget_is_announced(no_site_defaults):
    events = _warmup_events(_spec(optimizer="bayesian", budget=6, n_initial=9))
    assert len(events) == 1
    assert events[0]["n_initial"] == 9
    assert events[0]["budget"] == 6
    assert events[0]["campaign"] == "test_campaign"
    assert "no surrogate model" in events[0]["detail"]


def test_a_warmup_exactly_equal_to_budget_is_announced(no_site_defaults):
    """`>=`, not `>`: at equality the last ask is still index budget-1 < budget."""
    assert len(_warmup_events(_spec(optimizer="bayesian", budget=6, n_initial=6))) == 1


def test_an_all_warmup_dry_run_is_announced_even_with_n_initial_unset(
        no_site_defaults):
    """`examples/bo_init_bench.toml`'s shape: budget 4, nothing stated.

    The default resolves to `min(5, 4) = 4`, so the whole run is warm-up and the
    file's premise holds — and is now said out loud rather than inferred from
    arithmetic in a comment.
    """
    assert len(_warmup_events(_spec(optimizer="bayesian", budget=4))) == 1


def test_a_warmup_shorter_than_budget_is_not_announced(no_site_defaults):
    """The check can fail in both directions — a real campaign stays quiet."""
    assert _warmup_events(_spec(optimizer="bayesian", budget=6)) == []
    assert _warmup_events(_spec(optimizer="bayesian", budget=20, n_initial=5)) == []


def test_n_initial_contributes_to_the_resume_fingerprint_when_set():
    """A run that warms up for 8 trials is not the run that warmed up for 5."""
    assert optimizer_tuning_identity(_spec(n_initial=8)) == {"n_initial": 8}
    assert campaign_spec_fingerprint(_spec(n_initial=8)) != \
        campaign_spec_fingerprint(_spec())


def test_n_initial_left_unset_contributes_no_key_at_all():
    """Omission is the mechanism: an eighth defaulted key would rehash every
    in-flight checkpoint and blame a warm-up nobody changed."""
    assert optimizer_tuning_identity(_spec()) is None
    # And it is genuinely OMITTED rather than defaulted in: a spec that sets a
    # *different* knob contributes that one alone. (A `min(5, budget)` resolved
    # into the payload would show up here as a second key.)
    assert optimizer_tuning_identity(_spec(decision_rtol=0.25)) == \
        {"decision_rtol": 0.25}


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
async def test_campaign_run_start_narrates_the_baseline_present_and_absent(
    connected, tmp_path: Path
):
    """Two events, never zero, and both sited immediately after ``run_started``.

    Silence is not available as a spelling for "nothing was commanded": a
    campaign starting after a park is genuinely unconditioned until the first
    phase that speaks, and a transcript that merely *omits* the line cannot be
    told apart from one written before the field existed. So the absent case
    gets its own event rather than no event — ``SUBAGENT_RULES.md`` §3.1(a),
    "unknown must not be spelled with the same token as checked and clean".

    ``run_started`` stays first because that is where a watcher replaying from
    byte 0 learns the run's identity.
    """
    store = DataStore(tmp_path / "proj_baseline")
    try:
        absent: list[dict] = []
        await run_autonomous_campaign(
            _spec(optimizer="grid", budget=1), manager=connected,
            data_store=store, on_event=absent.append)

        baseline = PhaseSetpoints("floor", temp_setpoint_C=25.0,
                                  rh_setpoint_pct=30.0)
        present: list[dict] = []
        await run_autonomous_campaign(
            _spec(optimizer="grid", budget=1, conditions=baseline),
            manager=connected, data_store=store, on_event=present.append)
    finally:
        store.close()

    assert [e["type"] for e in absent][:2] == ["run_started", BASELINE_ABSENT_EVENT]
    assert BASELINE_EVENT not in {e["type"] for e in absent}

    assert [e["type"] for e in present][:2] == ["run_started", BASELINE_EVENT]
    assert present[1]["name"] == "floor"
    assert present[1]["temp_setpoint_C"] == 25.0
    assert present[1]["rh_setpoint_pct"] == 30.0
    # A setpoint is not an arrival: `command_steps` emits no wait, so nothing in
    # the run has checked the chamber got there. Stated, not implied.
    assert present[1]["waited"] is False


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


# -- The production read, and what the campaign actually scores ---------------
#
# Until this landed, a settling campaign scored the **last settle round** -- a
# step deliberately tagged `measurement="settle"` "so a round cannot enter the
# objective by itself", chosen by name anyway. `_equilibrate` now takes a real
# post-settle primary sweep and scores that. Three things are pinned here: the
# production read wins when it completes; a channel it missed falls back to that
# channel's own last settle round and **says so**; and the hold narrates every
# round instead of only its verdict.

from softae.core import autonomous_wiring as wiring  # noqa: E402
from softae.core.measurement_spec import MeasurementSpec  # noqa: E402
from softae.core.production_read import PRODUCTION_STEP_PREFIX  # noqa: E402
from softae.core.run_plan import (  # noqa: E402
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
)


def _settle_spec(**over) -> CampaignSpec:
    """A settling campaign with the clock wound down to test speed.

    ``rh_stability_pct=None``: the RH stability gate is exercised end to end in
    ``test_rh_equilibrate_stability_gate.py``; leaving it on here would decide
    these tests on the room rather than on the wiring they are about.
    """
    base = dict(channels=(21, 22), budget=1, optimizer="grid",
                equilibration_method="settle", round_period_s=0.01,
                min_hold_s=0.0, max_hold_s=0.05, settle_min_channels=2,
                rh_stability_pct=None)
    base.update(over)
    return _spec(**base)


def _spy_on_the_driver(monkeypatch) -> dict:
    """Record what ``drive_settle_phase`` handed back, without changing it.

    The last settle round is the thing the fallback is *supposed* to reach for,
    so a test asserting on the fallback has to be able to name it.
    """
    real = wiring.drive_settle_phase
    seen: dict = {}

    async def spy(*args, **kwargs):
        outcome, last_raws = await real(*args, **kwargs)
        seen["last_raws"] = dict(last_raws)
        seen["on_round"] = kwargs.get("on_round")
        return outcome, last_raws

    monkeypatch.setattr(wiring, "drive_settle_phase", spy)
    return seen


def _capture_scored(store_into: list):
    """An objective extractor that keeps the ``step_results`` it was scored on."""

    def extractor(step_results, params):
        store_into.append(dict(step_results))
        return 1.0

    return extractor


@pytest.mark.asyncio
async def test_equilibrate_scores_the_production_read_not_the_last_settle_round(
    connected, tmp_path: Path, monkeypatch
):
    """The whole point of the change: the recorded reading is the fresh sweep."""
    seen = _spy_on_the_driver(monkeypatch)

    async def fake_production_read(spec, channels, **kw):
        return {int(ch): f"PRODUCTION::{int(ch)}" for ch in channels}

    monkeypatch.setattr(wiring, "take_production_read", fake_production_read)

    store = DataStore(tmp_path / "proj_production")
    scored: list[dict] = []
    await run_autonomous_campaign(
        _settle_spec(), manager=connected, data_store=store,
        objective_extractor=_capture_scored(scored))

    assert scored, "no trial was scored"
    for channel in (21, 22):
        recorded = scored[-1][measure_step_name(channel)]
        assert recorded == f"PRODUCTION::{channel}"
        # And it is genuinely a *different* reading from the round it replaced --
        # otherwise this test would pass on a no-op wiring.
        assert recorded is not seen["last_raws"].get(channel)
    store.close()


@pytest.mark.asyncio
async def test_a_channel_the_production_read_missed_falls_back_and_says_so(
    connected, tmp_path: Path, monkeypatch
):
    """Per channel, not per trial -- and never silently.

    Leaving ``step_results`` untouched would keep the trial's *pre-settle* sweep,
    which is already primary-tagged and has already fired ``on_trial_measured``:
    a reading off a still-drying film masquerading as the production read with
    nothing on the record to say so.
    """
    seen = _spy_on_the_driver(monkeypatch)

    async def half_failed(spec, channels, **kw):
        return {int(ch): (f"PRODUCTION::{int(ch)}" if int(ch) == 21 else None)
                for ch in channels}

    monkeypatch.setattr(wiring, "take_production_read", half_failed)

    store = DataStore(tmp_path / "proj_partial")
    events: list[dict] = []
    scored: list[dict] = []
    await run_autonomous_campaign(
        _settle_spec(), manager=connected, data_store=store,
        objective_extractor=_capture_scored(scored), on_event=events.append)

    assert scored
    assert scored[-1][measure_step_name(21)] == "PRODUCTION::21"
    fallback = seen["last_raws"].get(22)
    assert fallback is not None, (
        "the settle round produced nothing, so this test never reached the "
        "fallback it exists to check")
    assert scored[-1][measure_step_name(22)] is fallback

    incomplete = [e for e in events if e["type"] == "production_read_incomplete"]
    assert [e["channel"] for e in incomplete] == [22]
    assert incomplete[0]["fell_back_to"] == "last_settle_round"
    store.close()


@pytest.mark.asyncio
async def test_a_production_read_that_raises_still_records_the_settle_verdict(
    connected, tmp_path: Path, monkeypatch
):
    """A read that raises is a failed read, and gets the failed read's answer.

    Letting it propagate is worse than it looks: ``_equilibrate`` runs under
    ``AutonomousLoop._post_measure``, which catches everything and returns the
    results untouched -- so the trial would silently keep its *pre-settle* sweep,
    no verdict would reach ``settle.json``, and the RH escalation would never see
    the phase at all. One ``trial_measured_hook_failed`` line would be the only
    trace.
    """
    seen = _spy_on_the_driver(monkeypatch)

    async def boom(spec, channels, **kw):
        raise RuntimeError("pico went away mid-sweep")

    monkeypatch.setattr(wiring, "take_production_read", boom)

    store = DataStore(tmp_path / "proj_raises")
    events: list[dict] = []
    scored: list[dict] = []
    result = await run_autonomous_campaign(
        _settle_spec(), manager=connected, data_store=store,
        objective_extractor=_capture_scored(scored), on_event=events.append)

    failed = [e for e in events if e["type"] == "production_read_failed"]
    assert failed and "RuntimeError" in failed[0]["error"]
    # The tail of `_equilibrate` still ran: a verdict was recorded, per channel
    # the last settle round was used, and every fallback said so.
    assert [e["type"] for e in events].count("settle_verdict") == 1
    assert (Path(store.run_dir(result.run_id)) / "settle.json").exists()
    for channel in (21, 22):
        assert scored[-1][measure_step_name(channel)] is seen["last_raws"][channel]
    incomplete = [e for e in events if e["type"] == "production_read_incomplete"]
    assert sorted(e["channel"] for e in incomplete) == [21, 22]
    store.close()


@pytest.mark.asyncio
async def test_a_production_read_that_completes_emits_no_fallback_event(
    connected, tmp_path: Path
):
    """The positive control for the test above -- and for the real read.

    Nothing is faked here: the campaign takes an actual post-settle sweep on the
    mock rig, so the fallback event's *absence* means the read worked rather than
    that the branch is unreachable.
    """
    store = DataStore(tmp_path / "proj_real_production")
    events: list[dict] = []
    result = await run_autonomous_campaign(
        _settle_spec(), manager=connected, data_store=store,
        on_event=events.append)

    assert not [e for e in events if e["type"] == "production_read_incomplete"]
    paths = [str(r.get("eis_file_path") or "")
             for r in store.query_measurements(run_id=result.run_id)]
    assert any(PRODUCTION_STEP_PREFIX in p for p in paths), (
        "no production sweep reached the store; the read never ran")
    store.close()


@pytest.mark.asyncio
async def test_the_hold_narrates_every_round_not_only_its_verdict(
    connected, tmp_path: Path, monkeypatch
):
    """A multi-hour hold that reports once, at the end, is unobservable.

    No rig time: the rounds are driven against a scripted sigma that never
    settles, so the phase runs to its ceiling over several rounds and the
    per-round narration is what is being counted.
    """
    from softae.analysis.equilibration import RoundFit

    rounds = {"n": 0}

    def scripted(raws, channels, **_kw):
        rounds["n"] += 1
        return [RoundFit(channel=int(ch), sigma=1e-4 * (1.5 ** rounds["n"]),
                         r1_ohms=4000.0) for ch in channels]

    monkeypatch.setattr(wiring, "build_settle_round_workflow", lambda *a, **k: None)
    monkeypatch.setattr(wiring, "settle_round_fits", scripted)

    store = DataStore(tmp_path / "proj_rounds")
    events: list[dict] = []
    await run_autonomous_campaign(
        _settle_spec(max_hold_s=0.2), manager=connected, data_store=store,
        on_event=events.append)

    narrated = [e for e in events if e["type"] == "settle_round"]
    verdicts = [e for e in events if e["type"] == "settle_verdict"]
    assert verdicts and verdicts[0]["n_rounds"] > 1, (
        "a single-round hold cannot show that narration is per round")
    assert [e["round"] for e in narrated] == list(range(verdicts[0]["n_rounds"]))
    assert narrated[0]["channels"] == [21, 22]

    # `SettleTracker.observe` has no window to judge until the trailing one
    # fills, so early rounds carry no verdict. They are still narrated, and
    # "not judged yet" is not spelled the same way as "judged and not settled".
    judged = [e for e in narrated if e["judged"]]
    assert judged, "no round was ever judged; the hold was too short to show this"
    # The tracker's own verdict fields ride along, so the live view and the final
    # one speak one vocabulary.
    assert {"evaluable", "settled", "participating", "reason"} <= set(judged[-1])
    unjudged = [e for e in narrated if not e["judged"]]
    assert all("evaluable" not in e for e in unjudged)
    store.close()


# -- Which sweep the production read takes -----------------------------------

def _plan_with_measure(measurement, *, scope=PhaseScope.PER_SAMPLE) -> RunPlan:
    return RunPlan((RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
                    RunPhase(PhaseKind.MEASURE, scope, measurement=measurement)))


def test_production_measurement_is_none_without_a_run_plan():
    """No plan -> the campaign's own `[measurement]` block, which is today's read."""
    assert wiring.production_measurement_override(_spec()) is None


def test_production_measurement_is_none_when_the_measure_phase_names_nothing():
    spec = _spec(run_plan=_plan_with_measure(None))
    assert wiring.production_measurement_override(spec) is None


def test_production_measurement_comes_from_the_measure_phase():
    """The only phase a preset is legal on is the only place this looks."""
    dense = MeasurementSpec(preset="Quick", overrides={"npts": 91})
    spec = _spec(run_plan=_plan_with_measure(dense))
    assert wiring.production_measurement_override(spec) == dense


def test_only_a_measure_phase_may_carry_a_measurement_block():
    """The premise the lookup rests on, asserted rather than assumed."""
    with pytest.raises(ValueError, match="only MEASURE"):
        RunPhase(PhaseKind.FORMULATE, measurement=MeasurementSpec(preset="Quick"))


def test_two_measure_phases_naming_different_sweeps_are_refused():
    """One production read, so one sweep.

    Picking silently is how a run records a sweep nobody wrote down.
    """
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_SAMPLE,
                 measurement=MeasurementSpec(preset="Quick")),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                 measurement=MeasurementSpec(preset="Quick",
                                             overrides={"npts": 91})),
    ))
    with pytest.raises(ValueError, match="different measurement blocks"):
        wiring.production_measurement_override(_spec(run_plan=plan))


def test_two_measure_phases_naming_the_same_sweep_are_not_a_conflict():
    """Equal blocks say the same thing twice; only disagreement is a refusal."""
    same = MeasurementSpec(preset="Quick", overrides={"npts": 91})
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_SAMPLE, measurement=same),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                 measurement=MeasurementSpec(preset="Quick",
                                             overrides={"npts": 91})),
    ))
    assert wiring.production_measurement_override(_spec(run_plan=plan)) == same


@pytest.mark.asyncio
async def test_the_measure_phases_override_reaches_the_production_read(
    connected, tmp_path: Path, monkeypatch
):
    """End to end: a plan that names a denser sweep gets that sweep read back."""
    dense = MeasurementSpec(preset="Quick", overrides={"npts": 91})
    asked: list = []

    async def record(spec, channels, **kw):
        asked.append(kw.get("measurement"))
        return {int(ch): f"PRODUCTION::{int(ch)}" for ch in channels}

    monkeypatch.setattr(wiring, "take_production_read", record)

    store = DataStore(tmp_path / "proj_override")
    await run_autonomous_campaign(
        _settle_spec(run_plan=_plan_with_measure(dense)),
        manager=connected, data_store=store)

    assert asked and all(m == dense for m in asked)
    store.close()


# ── The consensus feeder: arc_state / n_points_dropped ───────────────────────
#
# `RoundFit` has carried `arc_state` and `n_points_dropped` since eis-acq's
# landing — they are the two fields Proposal J's consensus-exclusion rule reads —
# but `settle_round_fits`, the *campaign's* feeder into `RoundFit`, built only
# channel/sigma/r1_ohms. Both therefore fell through to their "never asked"
# defaults on every campaign-driven settle round, which pinned J's rule in its
# `consensus_unavailable` guard state on the autonomous path no matter what the
# physical fits showed.
#
# These run the *real* analysis rather than a stubbed report, because the thing
# worth pinning is that the branch is reachable by data this rig produces
# (`SUBAGENT_RULES.md` §3.2) — `annotate_arc_closure` has to have actually run and
# attached the closure to the same fit R1 came off.

def _eis_raw(f, Z):
    """The 5-column array an EIS step returns: ``[f, |Z|, phase, Z', -Z'']``."""
    import numpy as np
    return [np.column_stack([f, np.abs(Z), np.degrees(np.angle(Z)),
                             Z.real, -Z.imag])]


def _closed_arc_raw():
    """A spectrum whose −Z″ peak is interior, so the arc closes in band."""
    from tests.eis_synthetic import reference_spectrum
    return _eis_raw(*reference_spectrum(Q=0.0, R_series=500.0,
                                        R_bulk=50000.0, C_par=1e-9))


def _open_arc_raw():
    """The default reference spectrum: still capacitive at the sweep floor."""
    from tests.eis_synthetic import reference_spectrum
    return _eis_raw(*reference_spectrum())


def test_settle_round_fits_closed_arc_reports_closed_state_and_drop_count():
    fit, = wiring.settle_round_fits({7: _closed_arc_raw()}, [7])

    assert fit.arc_state == "closed"
    # The fitter's own count, and an int — not the "never asked" sentinel. A fit
    # that dropped nothing is a *measured* zero.
    assert fit.n_points_dropped == 0
    assert isinstance(fit.n_points_dropped, int)
    # Read off the same fit R1 came from: no second analyze_spectrum call.
    assert fit.r1_ohms is not None


def test_settle_round_fits_open_arc_reports_open_state():
    fit, = wiring.settle_round_fits({7: _open_arc_raw()}, [7])

    assert fit.arc_state == "open"


def test_settle_round_fits_absent_channel_spells_never_asked_not_zero():
    """Positive control: a channel with no raw keeps the all-absent shape.

    `arc_state == ""` and `n_points_dropped is None` are the *absence of the
    question*, and this asserts they did not quietly become "closed"/0 when the
    two fields were wired up.
    """
    fit, = wiring.settle_round_fits({}, [7])

    assert fit.channel == 7
    assert fit.sigma is None and fit.r1_ohms is None
    assert fit.arc_state == ""
    assert fit.n_points_dropped is None


def test_settle_round_fits_absent_channel_spells_never_asked_on_the_new_fields():
    """The same positive control, extended to T11.33's three additions.

    `basis` says WHICH of the three absences this is — a sweep that never
    happened, spelled apart from a fit that failed, because they send an operator
    to different places. `sigma_mode` and `quality_verdict` must read "never
    asked" and not "asked, and clean": a feeder with nothing to report must not
    be indistinguishable from a clean board (`SUBAGENT_RULES.md` §3.1(a)).
    """
    from softae.analysis.equilibration import BASIS_ABSENT

    fit, = wiring.settle_round_fits({}, [7])

    assert fit.basis == BASIS_ABSENT
    assert fit.sigma_mode == ""
    assert fit.quality_verdict == ""
    # The helpers themselves, on the `None` report the absent branch hands them.
    assert wiring._report_sigma_mode(None) == ""
    assert wiring._report_quality_verdict(None) == ""


def test_report_sigma_refuses_a_fit_that_did_not_converge():
    """The new admission rule's ONLY `None` path for a report that exists.

    Design A admits on `fit.success` and a finite positive R₁ alone, which makes
    this the whole of what it still refuses — so without this the rule has no
    test that it can say no at all, and a `_report_sigma` that returned a number
    unconditionally would pass every other test in this file.

    `FitResult.success` defaults to `True`, so the absent-fit case is checked
    through `getattr(..., False)` rather than through the dataclass default.
    """
    class _Fit:
        success = False
        R1 = 50000.0

    class _Report:
        fit = _Fit()

    assert wiring._report_sigma(_Report()) is None
    assert wiring._report_sigma(None) is None

    # A converged fit whose R₁ is unusable is refused on the same rule.
    for bad_r1 in (0.0, -1.0, float("nan"), float("inf")):
        class _BadFit:
            success = True
            R1 = bad_r1

        class _BadReport:
            fit = _BadFit()

        assert wiring._report_sigma(_BadReport()) is None, bad_r1


def test_settle_round_fits_distinguishes_dropped_zero_from_never_asked():
    """`0` and `None` must not be spelled with the same token (§3.1(a))."""
    measured, absent = wiring.settle_round_fits({7: _closed_arc_raw()}, [7, 8])

    assert measured.n_points_dropped == 0
    assert absent.n_points_dropped is None
    assert measured.n_points_dropped != absent.n_points_dropped


# The helpers themselves, on shapes the real path can hand them.

def test_report_arc_state_returns_empty_for_a_report_that_does_not_exist():
    assert wiring._report_arc_state(None) == ""
    assert wiring._report_n_points_dropped(None) is None


def test_report_arc_state_returns_empty_for_a_fit_never_annotated():
    """A fit object with no `arc_closure` attached is "never asked", not a state."""
    class _Fit:
        n_points_dropped = 3

    class _Report:
        fit = _Fit()

    assert wiring._report_arc_state(_Report()) == ""
    assert wiring._report_n_points_dropped(_Report()) == 3


# ── T11.15: the settle gate fits through the direct (legacy) route ───────────
#
# USER RULING [a265] item 2, narrowing [a23] to exempt the geometry-free gate.
# The defect it fixes ([a280]): the gated engine's σ has a third state — a BOUND —
# which `_report_sigma` correctly refuses as a number, so ONE phase reading
# saturated at the instrument's rail drops that whole channel out of the settle
# rate window. On rung 3a's real board that starved the criterion below
# `min_channels` for seven consecutive rounds. `_legacy_report` builds only
# "unavailable"/"value" and never calls `decide_report_mode`, so the state does
# not exist on that route.
#
# The fixture below is a *positive control*, not a happy path: the gated half of
# the first test asserts the spectrum really does bound, so a green legacy half
# means the route changed the outcome rather than that the input was benign
# (`SUBAGENT_RULES.md` §3.1(e)).

#: Thickness the campaign would resolve per channel. Without one the cell cannot
#: report conductivity at all and EVERY mode is "unavailable" — which would make
#: the test below vacuous in exactly the way §3.1(a) describes.
_SETTLE_THICKNESS_UM = 50.0


#: Where the rail is planted, in Hz — NOT a free choice, and the reason is the
#: defect this fixture silently carried (2026-09-20). A railed point's Re Z is
#: ``|Z|·cos θ``, and the engine subtracts the fixture's own ``R_short`` (6.46 Ω on
#: mux16) before anything else looks at the spectrum. The previous placement was
#: ``Z[0:3]`` — the three HIGHEST-frequency points, where this circuit's |Z| is only
#: ~0.95 kΩ — so a −89.999° rail left Re Z ≈ 0.02 Ω, the correction drove it
#: negative, and `decide_report_mode` dropped all three as non-positive tan δ
#: *before* the windowed median ever saw them. Measured consequence: the corrupted
#: and uncorrupted spectra produced byte-identical headroom (24.457), i.e. the
#: corruption did nothing at all. Here |Z| ≈ 48–50 kΩ, so Re Z ≈ 41 Ω clears the
#: correction ~6×, while staying off the low-frequency plateau that carries R_bulk —
#: railing THAT costs both engines ~20 % of R₁ and breaks the `rel=0.1` agreement.
_RAIL_BAND_HZ = (600.0, 1010.0)

#: The rail's loss tangent as a FRACTION OF THE COMMITTED FLOOR, never an absolute
#: angle. `decide_report_mode` bounds when ``tan δ / tan ε < tand_headroom_mult``
#: (3.0), so 0.1 plants the rail 30× inside the bound region and re-derives itself
#: the next time the asset does. An absolute angle is what let this fixture decay:
#: it was tuned against a floor that has since been re-measured twice, and nothing
#: connected the two ([a338]'s standing-trap note).
_RAIL_HEADROOM_TARGET = 0.1


def _committed_phase_floor_tand(z_ohm: float) -> float:
    """``tan ε`` from the CURRENTLY COMMITTED mux16 asset, at this spectrum's z_med.

    Resolved the way `engine.analyze_spectrum` resolves it, so the fixture is pinned
    to the same number the decision uses rather than to a copy of it that ages.

    Raises rather than falling back. A fixture that cannot see the floor cannot know
    whether it is demonstrating a bound, and a quiet fallback here would be exactly
    the "could not check, and said fine" shape `SUBAGENT_RULES.md` §3.1(a) names.
    """
    import numpy as np

    from softae.analysis.eis.engine import _resolved_envelope
    from softae.analysis.eis.settings import eis_settings

    at = _resolved_envelope(eis_settings().fixture, role="film").floor_at(float(z_ohm))
    assert at.in_band and at.eps_deg == at.eps_deg, (
        f"no committed phase floor at z = {z_ohm:.4g} Ω (eps_deg={at.eps_deg}, "
        f"in_band={at.in_band}) — calibration/eis/mux16.toml did not resolve, so this "
        f"fixture cannot demonstrate a phase-floor bound")
    return float(np.tan(np.radians(at.eps_deg)))


def _phase_saturated_raw():
    """A closed arc with THREE adjacent readings pinned at the phase rail.

    [a280]'s root cause, and the rail angle is now **derived from the committed
    floor** instead of written down. `decide_report_mode` (T11.31, [a285]) takes the
    minimum of a 5-point running median of tan δ, so a single saturated point cannot
    pull the numerator under the floor — three adjacent ones can, because the median
    of a 5-wide window holding three of them IS one of them. The other points are
    untouched, and both engines still recover R₁ within ~2.4 % of the 50 kΩ truth:
    the spectrum is measurable, and only the gated engine declines to say so.

    Two things are pinned relative to the instrument rather than to a constant, each
    bought by a way this fixture has already failed:

    * **The angle** comes from `_RAIL_HEADROOM_TARGET` × the committed floor at this
      spectrum's own z_med (≈ 42.8 kΩ, bracketed by the 9.92 kΩ and 100.1 kΩ resistor
      rows). It lands near −89.95°, but it lands there *because the floor says so*.
    * **The placement** comes from `_RAIL_BAND_HZ`, where |Z| is large enough that the
      fixture's `R_short` correction cannot annihilate the corruption.

    The test asserts `numerator_phase_saturated`, which is the outcome-level check
    that the rail actually decided the verdict — the assertion the 2026-09-20 landing
    showed was missing, because without it a fully neutralised corruption reads
    exactly like a working one.
    """
    import numpy as np
    from tests.eis_synthetic import reference_spectrum
    from softae.analysis.eis.report import PHASE_SATURATION_DEG

    f, Z = reference_spectrum(Q=0.0, R_series=500.0, R_bulk=50000.0, C_par=1e-9)
    Z = Z.copy()

    # |Z| is untouched by a phase rotation, so the z_med the engine will score this
    # spectrum at is the uncorrupted one — taken here rather than guessed.
    tand_rail = _RAIL_HEADROOM_TARGET * _committed_phase_floor_tand(
        float(np.median(np.abs(Z))))
    phase_deg = -float(np.degrees(np.arctan2(1.0, tand_rail)))

    rail = (f >= _RAIL_BAND_HZ[0]) & (f <= _RAIL_BAND_HZ[1])
    assert int(rail.sum()) == 3, (
        f"expected 3 rail points in {_RAIL_BAND_HZ} Hz, got {int(rail.sum())} — the "
        f"reference sweep grid moved, and a 5-point window needs three to shift")
    assert abs(phase_deg) >= PHASE_SATURATION_DEG, (
        f"derived rail phase {phase_deg:.4f}° does not reach the saturation threshold "
        f"{PHASE_SATURATION_DEG}° — the floor has risen far enough that this fixture "
        f"no longer demonstrates a RAIL, whatever else it demonstrates")

    Z[rail] = np.abs(Z[rail]) * np.exp(1j * np.radians(phase_deg))
    return _eis_raw(f, Z)


def test_settle_round_fits_admits_a_converged_fit_with_a_bound_sigma_label():
    """T11.33 Design A: the bound label survives as provenance and stops gating.

    This REPLACES `test_spectrum_report_legacy_engine_never_bounds_a_phase_
    saturated_spectrum`, whose premise Design A retires. That test asserted
    `_report_sigma(gated) is None` — the bound starving the channel — as the
    defect T11.15's engine swap routed around. Under Design A the admission rule
    is `fit.success` and a finite positive R₁ alone, so BOTH engines now
    contribute `1/R₁` and the difference between them is recorded rather than
    acted on. The T11.15 ruling stands; it is simply no longer load-bearing.

    Both halves are asserted, because only the pair is informative: that the
    spectrum genuinely still bounds under `gated` (so this is not a benign input
    quietly passing, `SUBAGENT_RULES.md` §3.1(e)), and that the bound no longer
    withholds the number.
    """
    raw = _phase_saturated_raw()

    # The control, unchanged from the retired test: this spectrum really does
    # reach the bound branch. If it ever stops doing so the rest is vacuous.
    gated = wiring._spectrum_report_from_raw(
        raw, channel=7, thickness_um=_SETTLE_THICKNESS_UM, engine="gated")
    assert gated is not None
    assert gated.sigma.is_bound and not gated.sigma.is_value

    # The control ON THE CONTROL, added 2026-09-20 because its absence hid a dead
    # fixture for three days. `is_bound` above says only THAT it bound; these two say
    # the RAIL is why, which is the whole claim the fixture makes:
    #
    #   (a) the window that decided the headroom is the saturated one — when the rail
    #       points were being dropped as non-positive tan δ this read False while
    #       everything else still looked reasonable; and
    #   (b) the same circuit WITHOUT the corruption reports a value, so the bound is
    #       the pathology's doing and not the spectrum's (`SUBAGENT_RULES.md` §3.1(e)).
    assert gated.sigma.numerator_phase_saturated
    uncorrupted = wiring._spectrum_report_from_raw(
        _closed_arc_raw(), channel=7, thickness_um=_SETTLE_THICKNESS_UM, engine="gated")
    assert uncorrupted is not None
    assert uncorrupted.sigma.is_value and not uncorrupted.sigma.is_bound

    legacy = wiring._spectrum_report_from_raw(
        raw, channel=7, thickness_um=_SETTLE_THICKNESS_UM, engine="legacy")
    assert legacy is not None
    assert legacy.engine == "legacy"
    assert not legacy.sigma.is_bound
    assert legacy.sigma.mode == "value"

    # The change: a bound no longer withholds the number, and the number is the
    # conductance proxy — NOT the geometry-resolved σ the report also carries.
    for report in (gated, legacy):
        assert report.fit.success
        admitted = wiring._report_sigma(report)
        assert admitted is not None
        assert admitted == pytest.approx(1.0 / report.fit.R1)

    # ...and it is a different number from the σ that used to be admitted, so a
    # regression to `report.sigma.value` cannot pass this test.
    assert wiring._report_sigma(legacy) != pytest.approx(legacy.sigma.value)

    # The label rides beside it, verbatim, for the one engine that can produce it.
    # T11.39 (2026-09-17): "bound" narrowed to "bound_unqualified" once the mux16
    # calibration genuinely applied (it was silently stale since `744e034`, `[a307]`)
    # — the asset then carried ONE characterisation point, at 10.1 MΩ, and this
    # fixture's 42.8 kΩ z_med sat 2.4 decades outside its `valid_decades` window, so
    # the bound was real but extrapolated rather than locally measured. That comment
    # predicted T11.41 would move it back toward "bound", and the 2026-09-20 mux16
    # re-derive did exactly that: z_med is now BRACKETED by two commissioned resistor
    # rows (9.92 kΩ → 0.211°, 100.1 kΩ → 0.482°, the larger winning), so the floor is
    # locally measured and the bound is qualified.
    #
    # What the same re-derive also exposed, and the prediction did not anticipate: the
    # single 6.12° anchor had been inflating the floor ~12.7× at this impedance, which
    # is what had been bounding this spectrum. The rail was contributing nothing — see
    # `_phase_saturated_raw`. Both halves moved; only the fixture was wrong.
    assert wiring._report_sigma_mode(gated) == "bound"
    assert wiring._report_sigma_mode(legacy) == "value"

    # Both engines fit the same film; only the reporting decision differed.
    assert legacy.fit.R1 == pytest.approx(gated.fit.R1, rel=0.1)


def _reject_graded_raw():
    """A converged fit the grader REJECTS — `[p143]`'s ch22, reproduced.

    20 % multiplicative noise on the closed-arc reference. The fit still
    converges and still recovers R₁ within ~0.4 % of the 50 kΩ truth, but its RMS
    residual lands at ~18.9 % against `[quality] max_residual_pct = 15.0`, so
    `grade_fit` returns REJECT and `SpectrumReport.ok` (which is `quality.ok`) is
    False. Seeded, so the residual is deterministic rather than a coin flip near
    the ceiling.

    That combination — measurable R₁, refusing grade — is the exact shape that
    dropped ch22's rounds out of the rate window under the old admission rule.
    """
    from tests.eis_synthetic import reference_spectrum
    return _eis_raw(*reference_spectrum(Q=0.0, R_series=500.0, R_bulk=50000.0,
                                        C_par=1e-9, noise_pct=20.0, seed=0))


def test_settle_round_fits_admits_a_converged_fit_the_grader_rejected():
    """`[p143]`'s ch22 pinned permanently: a REJECT grade records, never gates.

    The old rule dropped this round via `report.ok`, which on the legacy route is
    `quality.ok` — and `_legacy_report` enforces `grade_fit`'s verdict
    unconditionally, unlike the gated engine which promotes it only when
    `[eis.gates] enabled`. So the route T11.15 installed is precisely the one on
    which a residual grade could still starve a channel. Design A ends that: the
    verdict reaches the operator on the row instead of the round vanishing.
    """
    from softae.analysis.quality import Verdict

    report = wiring._spectrum_report_from_raw(
        _reject_graded_raw(), channel=22, thickness_um=_SETTLE_THICKNESS_UM,
        engine="legacy")

    # The premise, asserted rather than assumed — a fit that converged, whose R₁
    # is good, and which the grader nonetheless refuses to speak for.
    assert report is not None
    assert report.fit.success
    assert report.quality.verdict is Verdict.REJECT
    assert not report.ok
    assert report.fit.quality["residual_rms_pct"] > 15.0
    assert report.fit.R1 == pytest.approx(50000.0, rel=0.05)

    # The change: admitted anyway, and graded on the record.
    admitted = wiring._report_sigma(report)
    assert admitted is not None
    assert admitted == pytest.approx(1.0 / report.fit.R1)

    # `.value`, not `str(verdict)` — `Verdict` is a str-mixin enum whose default
    # `str()` renders "Verdict.REJECT". This asserts the clean word specifically,
    # because the difference is invisible to every downstream comparison and
    # would only ever surface in front of an operator.
    assert wiring._report_quality_verdict(report) == "reject"
    assert wiring._report_quality_verdict(report) != str(report.quality.verdict)

    # End to end, through the feeder the campaign actually calls.
    fit, = wiring.settle_round_fits({22: _reject_graded_raw()}, [22],
                                    thickness_for=lambda ch: _SETTLE_THICKNESS_UM)
    assert fit.sigma is not None
    assert fit.quality_verdict == "reject"
    assert fit.basis == "fitted"


def test_settle_round_fits_needs_no_thickness_to_produce_a_rate():
    """T11.33 §2 consequence 3, made concrete — a real capability change.

    Without a thickness the cell constant is unresolvable, so `_legacy_report`
    builds `SigmaReport(mode="unavailable")` and the OLD rule returned `None` for
    every round of that channel forever. Design A's quantity is `1/R₁`, which a
    fit produces without any geometry at all.

    This is also the pin behind the rehearsal's `not_evaluable` scenario losing
    its mechanism (T11.33 §7): withholding a thickness no longer withholds a
    participant, so that scenario has to be re-forced through genuine absence.
    """
    from softae.analysis.equilibration import BASIS_FITTED

    for thickness_for in (None, lambda ch: None):
        fit, = wiring.settle_round_fits({7: _closed_arc_raw()}, [7],
                                        thickness_for=thickness_for)

        assert fit.sigma is not None
        assert fit.sigma == pytest.approx(1.0 / fit.r1_ohms)
        assert fit.basis == BASIS_FITTED
        # The control: the report really did decline to state a conductivity, so
        # the σ above cannot have come from the geometry path.
        assert fit.sigma_mode == "unavailable"


def test_sigma_from_eis_raw_still_resolves_the_configured_engine(monkeypatch):
    """The objective is untouched by Design A — the one thing that must not move.

    `[a23]`'s guard restated structurally rather than by engine keyword (that
    half is `test_sigma_objective_names_no_engine_so_it_still_follows_config`,
    which already exists and is kept). This asserts the *stronger* property the
    T11.33 edit put at risk: the objective path must not call the settle feeder's
    admission rule or either provenance helper, so that changing what the
    criterion regresses can never change what BO scores.

    Spying on the three helpers rather than reading the source, because an
    indirect call through a future refactor would still be caught.
    """
    from softae.analysis.eis import engine as eis_engine

    called: list[str] = []
    for name in ("_report_sigma", "_report_sigma_mode", "_report_quality_verdict"):
        real = getattr(wiring, name)

        def spy(report, *, _name=name, _real=real):
            called.append(_name)
            return _real(report)

        monkeypatch.setattr(wiring, name, spy)

    seen: list[dict] = []
    real_analyze = eis_engine.analyze_spectrum

    def analyze_spy(*args, **kwargs):
        seen.append(kwargs)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(eis_engine, "analyze_spectrum", analyze_spy)

    sigma = wiring._sigma_from_eis_raw(_closed_arc_raw(), channel=7,
                                       thickness_um=_SETTLE_THICKNESS_UM)

    assert called == []
    assert len(seen) == 1
    assert seen[0].get("engine") is None
    # T11.39 (2026-09-17) asserted `sigma is None` here: the mux16 calibration had
    # just started genuinely applying (`[a307]`), and under it this closed arc read as
    # resolution-limited, so the objective declined it. That comment named itself an
    # interim state and expected T11.41's per-row floor to move it. It did — and the
    # 2026-09-20 mux16 re-derive shows the decline was never honest:
    #
    #   the interim asset carried ONE point, 6.12° measured on the 10.1 MΩ reference
    #   resistor, applied 2.4 decades down at this fixture's 42.8 kΩ. That inflated
    #   the floor ~12.7×, and this perfectly healthy arc came out at headroom 1.93 —
    #   just under the 3.0 bound threshold. Scored against the rows that actually
    #   bracket it (9.92 kΩ and 100.1 kΩ), the same spectrum clears the floor 24.5×.
    #
    # So the honest verdict is a value, and the assertion moves to what this test has
    # always been for. The guard is that the objective must not return the settle
    # feeder's `1/R₁` proxy: the two differ by exactly the cell constant (K = 200 cm⁻¹
    # at 50 µm), so a leak reads ~2.0e-5 where the geometry-resolved σ reads ~4.0e-3 —
    # two orders apart, and no tolerance can confuse them. `called == []` above is the
    # structural half; this is the numeric half, and it is now a stronger guard than
    # `is None` was, because `None` was also what a fully broken analysis would return.
    assert sigma is not None
    assert sigma == pytest.approx(200.0 / 50_000.0, rel=0.05)
    assert sigma != pytest.approx(1.0 / 50_000.0, rel=0.5)


def test_settle_round_fits_requests_the_legacy_engine():
    """The call site names the engine, once per channel."""
    seen: list[dict] = []

    def spy(raw, **kwargs):
        seen.append(kwargs)
        return None

    original = wiring._spectrum_report_from_raw
    wiring._spectrum_report_from_raw = spy
    try:
        wiring.settle_round_fits({7: _closed_arc_raw(), 8: _open_arc_raw()}, [7, 8])
    finally:
        wiring._spectrum_report_from_raw = original

    assert len(seen) == 2
    assert [kw.get("engine") for kw in seen] == ["legacy", "legacy"]


def test_sigma_objective_names_no_engine_so_it_still_follows_config(monkeypatch):
    """Regression guard for [a23] on the path [a265] did NOT touch.

    The campaign's scored objective must keep resolving `[eis] engine`, so the
    number BO optimises against and the number the GUI shows cannot diverge.
    Spying on `analyze_spectrum` rather than on `_spectrum_report_from_raw`
    catches the stronger failure too: a default of "legacy" on the new keyword
    would re-couple the objective to the settle gate's choice invisibly.
    """
    from softae.analysis.eis import engine as eis_engine

    seen: list[dict] = []
    real = eis_engine.analyze_spectrum

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(eis_engine, "analyze_spectrum", spy)
    wiring._sigma_from_eis_raw(_closed_arc_raw(), channel=7,
                               thickness_um=_SETTLE_THICKNESS_UM)

    assert len(seen) == 1
    assert seen[0].get("engine") is None


def test_settle_round_fits_legacy_engine_still_populates_arc_state_and_drops(monkeypatch):
    """T11.15 does not undo the J feeder: both fields still come off the real fit.

    The same spy shape as the objective guard above, so a "legacy" here and a
    `None` there are read by one instrument — the check can distinguish them.
    """
    from softae.analysis.eis import engine as eis_engine

    seen: list[str | None] = []
    real = eis_engine.analyze_spectrum

    def spy(*args, **kwargs):
        seen.append(kwargs.get("engine"))
        return real(*args, **kwargs)

    monkeypatch.setattr(eis_engine, "analyze_spectrum", spy)
    fit, = wiring.settle_round_fits({7: _closed_arc_raw()}, [7])

    assert seen == ["legacy"]
    assert fit.arc_state == "closed"
    assert fit.n_points_dropped == 0
    assert isinstance(fit.n_points_dropped, int)
    assert fit.r1_ohms is not None


# ── The observation seam failing is itself an event ──────────────────────────

@pytest.mark.asyncio
async def test_a_failing_trial_measured_hook_is_wired_to_a_campaign_event(
    connected, tmp_path: Path, monkeypatch
):
    """`on_trial_measured_hook_failed` reaches `emit`, not just the log.

    On this path `on_trial_measured` is the whole settle → production-read → gate
    sequence, so a raise there keeps a stale pre-settle reading. Before this the
    only trace was a `logger.warning` nobody watches live.
    """
    from softae.core.autonomous_loop import AutonomousLoop

    built: list = []

    class _Capturing(AutonomousLoop):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    monkeypatch.setattr(wiring, "AutonomousLoop", _Capturing)

    events: list[dict] = []
    store = DataStore(tmp_path / "proj_hook_alert")
    await run_autonomous_campaign(
        _settle_spec(), manager=connected, data_store=store,
        on_event=events.append)

    assert built, "the campaign built no loop — the capture never happened"
    loop = built[0]
    assert loop.on_trial_measured_hook_failed is not None

    before = len(events)
    loop.on_trial_measured_hook_failed("RuntimeError: settle exploded")

    emitted = [e for e in events[before:]
               if e.get("type") == "trial_measured_hook_failed"]
    assert len(emitted) == 1
    assert emitted[0]["error"] == "RuntimeError: settle exploded"
    assert "iteration" in emitted[0]
    store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# _spectrum_report_from_raw — BOTH driver shapes (T11.29)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The campaign's single raw → physics hop has to read two legitimate shapes:
#
#   MockESPico.sendscript_getdata  → [ndarray(npts, 5)]  (one-element list)
#   AsyncESPico.sendscript_getdata → palmsens.mscript.parse_result_lines(...),
#                                    returned UNWRAPPED and opaque
#
# It used to hand-roll `np.asarray(..., dtype=float)`, which reads only the first.
# The real shape raised TypeError into the function's own `except Exception`, so
# every real spectrum became a silent `None` — no crash, total data loss on
# hardware. Every prior campaign test ran `--mock`, so nothing caught it.
#
# The two fixtures below are the SAME Nyquist shape scaled by 100x in impedance
# and 5x in frequency, so neither test can pass by accidentally taking the other
# branch: the mock fixture sits at R0~4.8e4 / R1~1.8e6, the real one at R0~4.8e2 /
# R1~1.8e4, two decades apart with no overlap in any asserted band.

import types  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import numpy as np  # noqa: E402

from softae.core.autonomous_wiring import _spectrum_report_from_raw  # noqa: E402

#: CPE exponent shared by both fixtures (matches the mock's own circuit).
_CPE_ALPHA = 0.7

#: Thickness so σ resolves to a value rather than "unavailable" — both tests
#: assert on σ, and σ needs a cell constant.
_THICKNESS_UM = 40.0


def _mock_shaped_raw():
    """Exactly what ``MockESPico.sendscript_getdata`` returns (mock_espico.py:118).

    A one-element list wrapping an ``(npts, 5)`` array whose columns are
    ``[f, |Z|, phase_deg, Z', -Z'']``.
    """
    from softae.drivers.mock_espico import _synthetic_eis
    return [_synthetic_eis(R0=4.81e4, R1=1.84e6, seed=7)]


def _real_shaped_columns():
    """``{column_index: values}`` for the real driver's ``get_values_by_column``.

    Column order is the one ``EISResult.from_raw`` reads on the palmsens branch:
    0 = frequency, 1 = Z', 2 = Z'' (note: Z'', not -Z'').
    """
    freq = np.geomspace(1.0e4, 0.2, 29)
    omega = 2.0 * np.pi * freq
    R0, R1, C, Q = 481.0, 1.84e4, 5.5e-8, 3.08517e-5
    Z = R0 + 1.0 / (Q * (1j * omega) ** _CPE_ALPHA) + R1 / (1.0 + 1j * omega * R1 * C)
    return {0: freq, 1: Z.real, 2: Z.imag}


def test_spectrum_report_from_raw_mock_shape_still_builds_report():
    """REGRESSION PIN: the mock's shape must keep working after the real-shape fix.

    Both callers (``settle_round_fits``, ``_sigma_from_eis_raw``) are exercised
    through this shape by every other campaign test, so a regression here would be
    a campaign-wide outage in simulation.
    """
    report = _spectrum_report_from_raw(_mock_shaped_raw(), channel=21,
                                       thickness_um=_THICKNESS_UM)

    assert report is not None, (
        "the mock's shape stopped parsing: _spectrum_report_from_raw swallowed an "
        "exception and returned None")
    assert report.fit is not None and report.fit.success

    # The mock fixture's own decade, two decades above the real-shape fixture.
    assert 2.0e4 < report.fit.R0 < 1.0e5, f"R0={report.fit.R0:g} is not the mock's"
    assert 1.0e6 < report.fit.R1 < 3.0e6, f"R1={report.fit.R1:g} is not the mock's"
    assert report.sigma.is_value
    assert 5.0e-5 < report.sigma.value < 5.0e-4


def test_spectrum_report_from_raw_real_palmsens_shape_builds_report():
    """The real driver's shape now parses — this is the bench defect (T11.29).

    ``AsyncESPico.sendscript_getdata`` returns ``parse_result_lines(...)`` unwrapped:
    an opaque PalmSens object, not float-castable. We fake the SDK boundary exactly
    as ``tests/test_real_drivers.py`` does rather than hand-building ``MScriptVar``
    rows (those need correctly hex-encoded ``data`` strings).

    Before the fix this asserted-on report was ``None`` — ``np.asarray(raw,
    dtype=float)`` raised ``TypeError`` into the function's own ``except``, and
    ``get_values_by_column`` was never called at all.
    """
    columns = _real_shaped_columns()

    ps_mscript = types.ModuleType("palmsens.mscript")
    ps_mscript.get_values_by_column = MagicMock(
        side_effect=lambda data, col: columns[col])
    ps_root = types.ModuleType("palmsens")
    ps_root.mscript = ps_mscript
    mods = {"palmsens": ps_root, "palmsens.mscript": ps_mscript}

    class _OpaqueRawData:
        """Stands in for ``parse_result_lines``'s return: not float-castable."""

    with patch.dict("sys.modules", mods):
        report = _spectrum_report_from_raw(_OpaqueRawData(), channel=21,
                                           thickness_um=_THICKNESS_UM)

    assert report is not None, (
        "the real driver's shape still does not parse: every bench spectrum "
        "silently becomes None")
    assert report.fit is not None and report.fit.success

    # POSITIVE CONTROL that the palmsens branch is the one that ran: the 2D-array
    # and 5-array branches never touch this function, so a non-zero call count is
    # proof the `else` branch was entered rather than coincidentally matched.
    assert ps_mscript.get_values_by_column.call_count == 3
    assert [c.args[1] for c in ps_mscript.get_values_by_column.call_args_list] == [0, 1, 2]

    # The real fixture's own decade — disjoint from the mock fixture's bands above,
    # so this cannot pass on data that took the wrong branch.
    assert 3.0e2 < report.fit.R0 < 1.0e3, f"R0={report.fit.R0:g} is not the fed-in one"
    assert 1.0e4 < report.fit.R1 < 3.0e4, f"R1={report.fit.R1:g} is not the fed-in one"
    assert report.sigma.is_value
    assert 5.0e-3 < report.sigma.value < 5.0e-2


# ── T11.33: every well is followed through ───────────────────────────────────
#
# The settle phase's per-well record, and the board minimum coming off the
# campaign path. Spec: `docs/SubAgent docs/t11_33_every_well_followed_through.md`.
# These drive `drive_settle_phase` directly against fabricated rounds — no rig,
# no store, no clock — because what is under test is the criterion's wiring, and
# `test_campaign_settle_phase.py` already owns the end-to-end shape.


class _FakeClock:
    """A clock the injected ``sleep`` advances, so rounds have a real span.

    The rate criterion regresses ``ln sigma`` against elapsed seconds, so a
    stopped clock makes every window ``rate_span_too_short`` and the test would
    pass on a refusal rather than on a verdict.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def _settle_plan(**over):
    """A settle plan wound down to test speed, with the RH gate off.

    ``rh_stability_pct=None`` because these rounds carry no humidity: with the
    gate on, every window would (correctly) report the room unreadable and the
    tests would stop being about the criterion.
    """
    from softae.core.run_plan import SettlePlan

    base = dict(round_period_s=600.0, min_hold_s=0.0, max_hold_s=6000.0,
                rh_stability_pct=None)
    base.update(over)
    return SettlePlan(**base)


async def _drive(plan, channels, sigma_for, *, r1_bound_ohms=None):
    """Run ``drive_settle_phase`` over fabricated rounds; return the outcome.

    ``sigma_for(channel, round_index)`` returns that cell's sigma for that round,
    or ``None`` to make the round absent for it.
    """
    from softae.analysis.equilibration import RoundFit
    from softae.core.autonomous_wiring import drive_settle_phase

    clock = _FakeClock()

    async def measure_round(index: int):
        return {int(ch): index for ch in channels}

    def fits_from(raws):
        fits = []
        for ch, index in raws.items():
            sigma = sigma_for(int(ch), int(index))
            if sigma is None:
                continue
            fits.append(RoundFit(channel=int(ch), sigma=sigma, r1_ohms=4000.0))
        return fits

    outcome, _ = await drive_settle_phase(
        plan, channels=list(channels), measure_round=measure_round,
        fits_from=fits_from, r1_bound_ohms=r1_bound_ohms,
        sleep=clock.sleep, now=clock)
    return outcome


@pytest.mark.asyncio
async def test_settle_outcome_by_channel_is_empty_under_the_default_deviation_criterion():
    """The shipped default computes no ``RateCheck``, so there is nothing to relay.

    ``{}`` here is an *absence of the question*, not a board on which every well
    was unjudgeable — the tracker's ``last_rate`` is only ever populated under
    ``criterion="rate"`` or ``"both"``. Spelling the two alike is how a reader
    concludes a board was judged when it never was.
    """
    outcome = await _drive(_settle_plan(), (21, 22, 23, 24),
                           lambda ch, i: 1.0e-4)

    assert outcome.settled, outcome.describe()
    assert outcome.by_channel == {}
    # And the projection survives the round trip into `settle.json`.
    assert outcome.as_dict()["by_channel"] == {}


@pytest.mark.asyncio
async def test_settle_outcome_by_channel_carries_the_last_windows_well_verdicts_under_rate_criterion():
    """Every channel in the last window leaves the phase with a word."""
    from softae.analysis.equilibration import (
        SETTLE_CRITERION_RATE,
        SETTLE_SETTLED,
        WellVerdict,
    )

    plan = _settle_plan(criterion=SETTLE_CRITERION_RATE,
                        rate_tol_dec_per_h=0.20, max_hold_s=12000.0)
    outcome = await _drive(plan, (21, 22, 23, 24), lambda ch, i: 1.0e-4)

    assert outcome.by_channel, "the rate criterion ran but relayed no verdict"
    assert set(outcome.by_channel) == {21, 22, 23, 24}
    for channel, verdict in outcome.by_channel.items():
        assert isinstance(verdict, WellVerdict)
        assert verdict.channel == channel
        # A cell whose sigma never moves is quiet, and a quiet cell certifies.
        assert verdict.word == SETTLE_SETTLED, verdict.reason

    # The projection carries the word and T11.17's pair, keyed by string for
    # `settle.json`; `forgiven_round` is deliberately not in it.
    projected = outcome.as_dict()["by_channel"]
    assert set(projected) == {"21", "22", "23", "24"}
    assert projected["21"]["word"] == SETTLE_SETTLED
    assert set(projected["21"]) == {"word", "rate_per_hour", "upper_bound_per_hour"}


@pytest.mark.asyncio
async def test_settle_outcome_by_channel_gives_a_partly_absent_well_a_word_too():
    """A well that missed some rounds is still followed through.

    The T11.33 ruling's other half: no well's state is dropped for being
    unjudgeable, so the optimizer's record says what happened to each of its
    samples rather than listing the survivors.
    """
    from softae.analysis.equilibration import (
        EXCLUDED_ABSENT,
        SETTLE_CRITERION_RATE,
        SETTLE_SETTLED,
    )

    plan = _settle_plan(criterion=SETTLE_CRITERION_RATE,
                        rate_tol_dec_per_h=0.20, max_hold_s=12000.0)
    # ch24 fits on the first round only; the other three are quiet throughout.
    outcome = await _drive(
        plan, (21, 22, 23, 24),
        lambda ch, i: 1.0e-4 if (ch != 24 or i == 0) else None)

    assert 24 in outcome.by_channel, "the absent well left the phase unrecorded"
    assert outcome.by_channel[24].word == EXCLUDED_ABSENT
    assert outcome.by_channel[24].rate_per_hour is None
    assert outcome.by_channel[21].word == SETTLE_SETTLED


@pytest.mark.asyncio
async def test_settle_outcome_by_channel_omits_a_well_that_never_fit_at_all():
    """The documented edge of "every well": the criterion has no board roster.

    ``RateCheck.by_well`` covers every channel **in the window**, and the window
    is built from the fits — so a well that produced no ``RoundFit`` in any round
    is not a well the criterion ever learned exists, and it gets no word. A well
    that fit *once* and then stopped does (the test above).

    Pinned rather than asserted away, because it is the gap between the spec's
    "every well on the board carries a word" and what ``equilibration.py`` can
    actually deliver. **What closes it on the write path is
    ``_equilibrate``**: the board word is stamped from ``round_channels`` — the
    roster the driver does have — so such a well still records its
    ``certification`` and leaves only ``well_verdict`` NULL, which is the honest
    record of a criterion that had nothing to say about it.
    """
    from softae.analysis.equilibration import SETTLE_CRITERION_RATE

    plan = _settle_plan(criterion=SETTLE_CRITERION_RATE,
                        rate_tol_dec_per_h=0.20, max_hold_s=12000.0)
    outcome = await _drive(plan, (21, 22, 23, 24),
                           lambda ch, i: None if ch == 24 else 1.0e-4)

    assert set(outcome.by_channel) == {21, 22, 23}
    assert 24 not in outcome.by_channel


@pytest.mark.asyncio
async def test_drive_settle_phase_never_refuses_a_narrow_board_on_count():
    """T11.33: the campaign tracker passes ``board_minimum=None``.

    Two quiet wells on a plan asking for five used to be ``not_evaluable`` — a
    statement about the board's *width*, not about the sample. Now each well is
    judged on its own evidence, so a narrow board settles on the same evidence a
    wide one would.

    This is the test that would go green on the safe answer if the change had not
    landed, so it asserts the *positive*: settled, not merely "not refused".
    """
    from softae.analysis.equilibration import SETTLE_NOT_EVALUABLE

    outcome = await _drive(_settle_plan(settle_min_channels=5), (21, 22),
                           lambda ch, i: 1.0e-4)

    assert outcome.outcome != SETTLE_NOT_EVALUABLE, outcome.describe()
    assert outcome.settled, outcome.describe()
    assert sorted(outcome.participating) == [21, 22]


@pytest.mark.asyncio
async def test_drive_settle_phase_still_reports_not_evaluable_when_no_well_is_judgeable():
    """The board word keeps its meaning: nothing judgeable, nothing to judge.

    Without this, the test above could be passing because the phase now certifies
    everything — which would make ``not_evaluable`` unreachable and the word a
    lie rather than a policy change.
    """
    from softae.analysis.equilibration import SETTLE_NOT_EVALUABLE

    outcome = await _drive(_settle_plan(), (21, 22), lambda ch, i: None)

    assert outcome.outcome == SETTLE_NOT_EVALUABLE, outcome.describe()


def _settle_campaign_spec(**over):
    """A settling campaign wound down to test speed (one trial, tiny holds)."""
    base = dict(equilibration_method="settle", round_period_s=0.01,
                min_hold_s=0.0, max_hold_s=0.2, budget=1, channels=(21, 22))
    base.update(over)
    return _spec(**base)


@pytest.mark.asyncio
async def test_campaign_start_emits_settle_min_channels_ignored_when_the_spec_sets_it(
    connected, tmp_path: Path
):
    """An ignored key is said once, loudly, rather than silently dropped.

    ``settle_min_channels`` is a per-tracker policy after T11.33 and the campaign
    path passes ``board_minimum=None``, so a spec that sets it is stating
    something this path will not do. Ignored rather than refused: every committed
    campaign spec sets the key, and it still governs the two tool paths.
    """
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    await run_autonomous_campaign(
        _settle_campaign_spec(settle_min_channels=3), manager=connected,
        data_store=store, on_event=events.append)

    ignored = [e for e in events if e["type"] == "settle_min_channels_ignored"]
    assert ignored, "the spec set a key this path ignores and said nothing"
    assert ignored[0]["settle_min_channels"] == 3
    # A `not of_type("settle_unevaluable_board")` line stood here and was retired
    # (T11.51, `SUBAGENT_RULES` §3.1(e)): T11.33 step 3 removed the last emitter
    # from `src/` — zero hits, verified by grep — so the negative could no longer
    # fail for any reason. The two assertions above are what this test needs: the
    # replacement event fires, with the right payload, and it fires *instead*.
    store.close()


@pytest.mark.asyncio
async def test_campaign_start_does_not_emit_settle_min_channels_ignored_when_unset(
    connected, tmp_path: Path
):
    """No key, no complaint — otherwise every campaign carries a warning."""
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    await run_autonomous_campaign(
        _settle_campaign_spec(), manager=connected,
        data_store=store, on_event=events.append)

    # The phase really ran, so the absence below is about the key and not about
    # a settle block that never happened.
    assert [e for e in events if e["type"] == "settle_mode"]
    assert not [e for e in events if e["type"] == "settle_min_channels_ignored"]
    store.close()
