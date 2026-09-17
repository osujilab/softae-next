"""Campaign preflight: stock feasibility and projected duration (P5.2).

Duration is reported as a **rate with bounds**, never a single ETA — a Bayesian
campaign stops on a convergence criterion, not a known iteration count, so a
confident finish time would be a fabrication.
"""

from __future__ import annotations

import pytest

from softae.analysis.rh_floor import TemperatureBin
from softae.config import loader
from softae.core.autonomous_wiring import CampaignSpec
from softae.core.eis_scripts import EISParams
from softae.core.measurement_spec import MeasurementSpec
from softae.core.phase_setpoints import CONDITIONS_PHASE_TAG, PhaseSetpoints
from softae.core.preflight import (
    CONDITIONS_PHASE,
    CampaignProjection,
    approach_ceiling_s,
    commanded_conditions,
    estimate_eis_duration,
    estimate_step_duration,
    estimate_workflow_duration,
    per_iteration_draw,
    project_campaign,
    rh_floor_advisories,
    unconditioned_start_warnings,
)
from softae.core.reservoir import ReservoirLedger
from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)
from softae.core.task_catalog import TaskCatalog
from softae.workflows.workflow_model import WorkflowStep


@pytest.fixture(scope="module")
def catalog() -> TaskCatalog:
    return TaskCatalog.load_toml(loader.tasks_toml_path())


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="proj", channels=(21, 22), pcb_name="SoftAE_EIS_4Stripe",
        parameter_space={
            "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
            "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
        },
        vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1),
        two_phase=True, budget=20,
    )
    base.update(over)
    return CampaignSpec(**base)


def _step(method, **params) -> WorkflowStep:
    return WorkflowStep(name=method, instrument="x", method=method, params=params)


# ── Step-level estimates ─────────────────────────────────────────────────────

class TestStepDuration:
    def test_a_pump_step_is_volume_over_rate(self):
        # 100 µL at 200 µL/min = 30 s
        assert estimate_step_duration(
            _step("single_pump", rate=200, dispense_vol=100)) == pytest.approx(30.0)

    def test_proportional_extrusion_uses_the_slowest_pump(self):
        """Rates are split so all components extrude together; max, not sum."""
        s = _step("single_drop_simul", vols=[20.0, 10.0], disp_rates=[50.0, 25.0])
        # both are 24 s; the pair finishes together, not in 48 s
        assert estimate_step_duration(s) == pytest.approx(24.0)

    def test_dwells_are_included(self):
        s = _step("single_drop_simul", vols=[20.0], disp_rates=[50.0],
                  elution_wait_s=48.0, wick_dwell_s=5.0)
        assert estimate_step_duration(s) == pytest.approx(24.0 + 53.0)

    def test_time_scale_scales_dwells(self):
        """Mock/demo runs set time_scale=0, and the projection must follow."""
        s = _step("single_drop_simul", vols=[20.0], disp_rates=[50.0],
                  elution_wait_s=48.0, time_scale=0.0)
        assert estimate_step_duration(s) == pytest.approx(24.0)

    def test_precondition_accounts_for_the_flush_factor(self):
        s = _step("precondition_flush", vol_list=[10.0], rate_list=[100.0],
                  flush_factor=3.0)
        # 30 µL preload at 100 µL/min = 18 s
        assert estimate_step_duration(s) == pytest.approx(18.0)

    def test_an_anneal_hold_dominates(self):
        assert estimate_step_duration(
            _step("anneal", hold_time_s=14400.0)) == pytest.approx(14400.0)

    def test_an_unmodelled_step_is_unknown_not_free(self):
        """Counting it as zero would understate the projection as if precise."""
        assert estimate_step_duration(_step("some_novel_method")) is None

    def test_a_zero_rate_does_not_divide_by_zero(self):
        assert estimate_step_duration(
            _step("single_pump", rate=0, dispense_vol=100)) == 0.0


# ── EIS sweep model ──────────────────────────────────────────────────────────

class TestEISDuration:
    def test_lower_frequencies_cost_more(self):
        fast = estimate_eis_duration(EISParams(f_hi=200_000, f_lo_mHz=4_000, npts=35))
        slow = estimate_eis_duration(EISParams(f_hi=200_000, f_lo_mHz=100, npts=35))
        assert slow > fast * 5

    def test_more_points_cost_more(self):
        few = estimate_eis_duration(EISParams(npts=10))
        many = estimate_eis_duration(EISParams(npts=40))
        assert many > few

    def test_degenerate_parameters_do_not_raise(self):
        assert estimate_eis_duration(EISParams(npts=0)) == 0.0


#: What one channel of each preset **actually** cost on this rig, from timestamp
#: deltas in production runs and reproducible to ±0.1 s over four months. These
#: are the three points the sweep model's two constants were fitted to;
#: ``Longest`` has never been timed and deliberately has no entry.
#:
#: The ``Quick`` row is the **20 Hz** sweep, spelled out as explicit ``EISParams``
#: rather than resolved from the preset for exactly this reason: the preset moved
#: to 7 Hz on 2026-08-14 and its entry in ``EIS_MEASURED_S_PER_CHANNEL`` was
#: retired with it. The fit was still made against these three, so this stays a
#: valid statement about the model even though only two presets are now "timed".
#: The 2026-08-17 bench session (channel 1, all four presets interleaved). These
#: are the *current* grids; the previous table here held the pre-retune ones.
MEASURED_S_PER_CHANNEL = {
    "Quick":    (EISParams(f_hi=200_000, f_lo_mHz=6_475, npts=27), 17.50),
    "Standard": (EISParams(f_hi=200_000, f_lo_mHz=3_912, npts=34), 37.19),
    "Extended": (EISParams(f_hi=200_000, f_lo_mHz=1_351, npts=53), 120.42),
    "Longest":  (EISParams(f_hi=200_000, f_lo_mHz=228, npts=39), 516.44),
}

#: Two constants cannot fit four points. ~9 % is what the best pair achieves, and
#: a third parameter buys only ~1 %, so this tolerance is the functional form's
#: limit and not slack left for a future edit to hide in.
DURATION_TOL_REL = 0.10


class TestEISDurationAgainstTheBench:
    """The point of this class: turn "the model is roughly right" into "the model
    is unchanged". It ran ~10x low for four months and nothing caught it, because
    nothing compared it to a stopwatch."""

    @pytest.mark.parametrize("preset", sorted(MEASURED_S_PER_CHANNEL))
    def test_model_eis_duration_matches_the_bench(self, preset):
        from softae.core.preflight import model_eis_duration

        params, measured = MEASURED_S_PER_CHANNEL[preset]
        assert model_eis_duration(params) == pytest.approx(
            measured, rel=DURATION_TOL_REL)

    @pytest.mark.parametrize("preset", sorted(MEASURED_S_PER_CHANNEL))
    def test_model_eis_duration_old_constants_fail_the_bench(self, preset, monkeypatch):
        # Proves the pin above has teeth. The pre-correction constants were 3.0
        # cycles and a 0.05 s floor; if this test can pass with those, it is not
        # measuring anything.
        import softae.core.preflight as preflight

        monkeypatch.setattr(preflight, "EIS_CYCLES_PER_POINT", 3.0)
        monkeypatch.setattr(preflight, "EIS_MIN_POINT_S", 0.05)
        params, measured = MEASURED_S_PER_CHANNEL[preset]
        assert preflight.model_eis_duration(params) != pytest.approx(
            measured, rel=DURATION_TOL_REL)

    @pytest.mark.parametrize("preset", sorted(MEASURED_S_PER_CHANNEL))
    def test_estimate_returns_the_stopwatch_exactly_not_the_model(self, preset):
        """The 2026-08-17 change: a timed grid must return its measurement, not a
        model of it. Modelling ``Quick`` cost every projection 22.8 % while the
        real number sat unused in the same module."""
        params, measured = MEASURED_S_PER_CHANNEL[preset]
        assert estimate_eis_duration(params) == pytest.approx(measured, abs=0.01)

    def test_eis_duration_basis_is_measured_for_every_shipped_preset(self):
        from softae.core.preflight import eis_duration_basis

        for preset in MEASURED_S_PER_CHANNEL:
            assert eis_duration_basis(preset) == "measured"
        assert eis_duration_basis(None) == "extrapolated"
        assert eis_duration_basis("NoSuchPreset") == "extrapolated"

    def test_a_sweep_off_every_timed_grid_is_extrapolated_and_modelled(self):
        """The fallback still exists and is still labelled. A custom ``f_lo``
        matches no anchor, so it must model rather than borrow a neighbour's
        stopwatch."""
        from softae.core.preflight import measured_duration_s, model_eis_duration

        custom = EISParams(f_hi=200_000, f_lo_mHz=500, npts=31)
        assert measured_duration_s(custom) is None
        assert estimate_eis_duration(custom) == pytest.approx(
            model_eis_duration(custom))


# ── Whole-workflow roll-up ───────────────────────────────────────────────────

class TestWorkflowEstimate:
    def test_a_real_trial_is_fully_timed(self, catalog):
        from softae.core.autonomous_wiring import build_trial_workflow

        wf = build_trial_workflow(
            _spec(), {"vol_p0": 20.0, "vol_p1": 10.0}, catalog=catalog)
        est = estimate_workflow_duration(wf)

        assert est.n_steps > 0
        assert est.total_s > 0
        assert est.is_complete, f"{est.n_unknown} step(s) untimed"

    def test_unknown_steps_are_counted_not_hidden(self):
        class _WF:
            def resolve_steps(self):
                return [_step("single_pump", rate=100, dispense_vol=100),
                        _step("mystery")]

        est = estimate_workflow_duration(_WF())
        assert est.n_unknown == 1
        assert not est.is_complete


class TestDraw:
    def test_per_pump_draw_comes_from_the_built_workflow(self, catalog):
        """Reflects what the hardware is commanded, correction included."""
        from softae.core.autonomous_wiring import build_trial_workflow

        wf = build_trial_workflow(
            _spec(), {"vol_p0": 20.0, "vol_p1": 10.0}, catalog=catalog)
        draw = per_iteration_draw(wf)

        assert set(draw) >= {0, 1}
        assert all(v > 0 for v in draw.values())

    def test_a_zeroed_component_draws_nothing(self):
        class _WF:
            def resolve_steps(self):
                return [_step("single_drop_simul", ids=[0, 1], vols=[20.0, 0.0])]

        assert per_iteration_draw(_WF()) == {0: 20.0}


# ── Projection and its verdict ───────────────────────────────────────────────

class TestProjection:
    def test_projects_time_and_draw(self, catalog):
        p = project_campaign(_spec(), catalog=catalog)
        assert p.per_iteration_s > 0
        assert sum(p.per_iteration_draw_uL.values()) > 0

    def test_time_to_budget_is_an_upper_bound(self, catalog):
        p = project_campaign(_spec(budget=20), catalog=catalog)
        assert p.time_to_budget_s == pytest.approx(p.per_iteration_s * 20)
        assert "sooner" in p.describe()      # framed as a bound, not an ETA

    def test_undeclared_stock_is_unknown_not_insufficient(self, catalog):
        p = project_campaign(_spec(), catalog=catalog)
        assert p.iterations_supported() is None
        assert p.stock_sufficient is None
        assert "unknown" in p.describe()

    def test_sufficient_stock_passes(self, catalog):
        led = ReservoirLedger()
        led.refill(0, 500_000.0)
        led.refill(1, 500_000.0)

        p = project_campaign(_spec(budget=5), catalog=catalog, ledger=led)

        assert p.stock_sufficient is True
        assert not any("hard-stop" in w for w in p.warnings)

    def test_a_shortfall_is_reported_before_the_run(self, catalog):
        """The whole point: not discovered as a park at iteration 40."""
        led = ReservoirLedger()
        led.refill(0, 1000.0)
        led.refill(1, 1000.0)

        p = project_campaign(_spec(budget=50), catalog=catalog, ledger=led)

        assert p.stock_sufficient is False
        assert any("hard-stop" in w for w in p.warnings)
        assert "NOT enough" in p.describe()

    def test_the_scarcest_pump_sets_the_runway(self, catalog):
        """An average would flatter it; the first stock to run out stops the run."""
        led = ReservoirLedger()
        led.refill(0, 1_000_000.0)
        led.refill(1, 1000.0)

        p = project_campaign(_spec(budget=100), catalog=catalog, ledger=led)

        assert p.stock_sufficient is False

    def test_purge_consumption_shortens_the_runway(self, catalog):
        """Purging accrues with elapsed time, not with iterations (P8)."""
        led = ReservoirLedger()
        led.refill(0, 20_000.0)
        led.refill(1, 20_000.0)

        without = project_campaign(_spec(budget=100), catalog=catalog, ledger=led)
        with_purge = project_campaign(
            _spec(budget=100), catalog=catalog, ledger=led,
            purge_uL_per_day={0: 1_000_000.0, 1: 1_000_000.0})

        assert with_purge.iterations_supported() < without.iterations_supported()

    def test_a_broken_spec_reports_rather_than_raising(self, catalog):
        """Preflight must never be the reason a campaign cannot start."""
        p = project_campaign(_spec(pcb_name="no-such-pcb"), catalog=catalog)
        assert p.warnings
        assert not p.duration_complete


class TestSummary:
    def test_describe_uses_human_units(self):
        p = CampaignProjection(
            per_iteration_s=3600.0, per_iteration_draw_uL={0: 100.0}, budget=48)
        text = p.describe()
        assert "1.0 h" in text          # per iteration
        assert "2.0 days" in text       # to budget

    def test_describe_never_promises_a_finish_time(self):
        p = CampaignProjection(
            per_iteration_s=60.0, per_iteration_draw_uL={0: 10.0}, budget=10)
        text = p.describe().lower()
        assert "at most" in text and "sooner" in text


# ── Run-plan costs the built workflow cannot price (B4) ──────────────────────
#
# `project_campaign` used to time a cast. A plan carrying an 8 h cure spends its
# time in three places the old projection either missed entirely or billed at
# zero, and the three are bounded from different directions — which is why the
# summary now says which is which.

ANNEAL_HOLD_S = 28800.0        # data/tasks.toml: anneal_85C_8h
SETTLE_CEILING_S = 14400.0
SETTLE_FLOOR_S = 1500.0
RH_APPROACH_S = 6000.0         # the ~5000 s descent, plus margin, per phase


def _conditions(name, temp, rh, **over) -> PhaseSetpoints:
    return PhaseSetpoints(name=name, temp_setpoint_C=temp,
                          rh_setpoint_pct=rh, **over)


def _bench_plan() -> RunPlan:
    """The four-phase bench instance: cast, cure, equilibrate, measure."""
    return RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
                 conditions=_conditions("casting", 25.0, 40.0)),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 anneal_task="anneal_85C_8h",
                 conditions=_conditions("anneal", 85.0, 20.0,
                                        rh_approach_timeout_s=RH_APPROACH_S)),
        RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                 settle=SettlePlan(round_period_s=240.0,
                                   min_hold_s=SETTLE_FLOOR_S,
                                   max_hold_s=SETTLE_CEILING_S),
                 conditions=_conditions("equilibrate", 25.0, 50.0)),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                 conditions=_conditions("equilibrate", 25.0, 50.0)),
    ))


class TestAWaitIsEitherADwellOrACeiling:
    """Two different things share the method name ``wait``, and they bound the
    duration from opposite sides.

    The ceiling case used to return **0.0** — not ``None``, so it was not even
    counted unknown. ``workflows/equilibration.py`` routes its own projection
    around this function specifically because of that, and says so in a comment.
    """

    def test_a_commanded_dwell_bills_its_duration(self):
        assert estimate_step_duration(
            _step("wait", duration_s=90.0)) == pytest.approx(90.0)

    def test_an_approach_bills_its_timeout_as_a_ceiling(self):
        assert estimate_step_duration(
            _step("wait", within=2.0, timeout=1800.0)) == pytest.approx(1800.0)

    def test_an_approach_is_not_billed_as_free(self):
        """The defect, stated directly: an 8 h cure's approach entered the
        projection as zero seconds and was not flagged unknown either."""
        assert estimate_step_duration(_step("wait", within=2.0, timeout=1800.0)) != 0.0

    def test_a_dwell_wins_over_a_timeout_when_both_are_present(self):
        """A dwell is what the step *will* take; a timeout is only its ceiling."""
        assert estimate_step_duration(
            _step("wait", duration_s=60.0, timeout=1800.0)) == pytest.approx(60.0)

    def test_a_wait_with_neither_is_still_free_rather_than_unknown(self):
        assert estimate_step_duration(_step("wait")) == 0.0

    def test_the_phase_tag_matches_the_one_the_emitter_stamps(self):
        """``preflight`` restates the tag rather than importing the emitter; the
        two must not fork, so they are pinned equal here."""
        assert CONDITIONS_PHASE == CONDITIONS_PHASE_TAG


class TestARunPlanIsProjectedHonestly:
    def _spec_with_plan(self, **over):
        return _spec(run_plan=_bench_plan(), batch=True, **over)

    def test_the_anneal_hold_is_billed(self, catalog):
        p = project_campaign(self._spec_with_plan(), catalog=catalog)
        assert p.workflow_s > ANNEAL_HOLD_S

    def test_the_condition_approaches_are_billed_at_their_ceilings(self, catalog):
        """Three distinct conditions, two axes each; the anneal's RH approach
        carries the phase's own 6000 s rather than the driver's 120 s default."""
        p = project_campaign(self._spec_with_plan(), catalog=catalog)
        assert p.approach_ceiling_s == pytest.approx(
            1800.0 * 5 + RH_APPROACH_S)          # 5 default waits + the anneal's

    def test_the_settle_window_is_added_from_the_spec_not_the_workflow(self, catalog):
        """The equilibrate phase emits no steps -- it terminates on evidence --
        so its ceiling cannot be read off the built workflow."""
        p = project_campaign(self._spec_with_plan(), catalog=catalog)
        assert p.settle_ceiling_s == pytest.approx(SETTLE_CEILING_S)
        assert p.settle_floor_s == pytest.approx(SETTLE_FLOOR_S)
        assert p.per_iteration_s == pytest.approx(p.workflow_s + SETTLE_CEILING_S)

    def test_the_projection_exceeds_the_cure_plus_the_settle_ceiling(self, catalog):
        """The headline: ``check`` must stop reporting a cast when a plan carries
        an 8 h cure and a 4 h equilibrate ceiling."""
        p = project_campaign(self._spec_with_plan(), catalog=catalog)
        assert p.per_iteration_s > ANNEAL_HOLD_S + SETTLE_CEILING_S

    def test_the_summary_names_both_ceilings_as_ceilings(self, catalog):
        text = project_campaign(self._spec_with_plan(), catalog=catalog).describe()
        assert "approaching commanded conditions" in text
        assert "a ceiling, not a dwell" in text
        assert "equilibrate window of at most" in text
        assert "may end sooner" in text

    def test_approach_ceiling_reads_the_tag_not_the_step_name(self):
        """Condition step names carry an operator-chosen label; the tag is the
        contract. A wait tagged for another phase must not be counted."""
        class _WF:
            def resolve_steps(self):
                return [
                    _step("wait", timeout=300.0).with_tags(phase=CONDITIONS_PHASE),
                    _step("wait", timeout=900.0).with_tags(phase="anneal"),
                ]

        assert approach_ceiling_s(_WF()) == pytest.approx(300.0)


class TestACampaignWithNoRunPlanIsUnchanged:
    """The byte-identity pin, and **its exact boundary**.

    Everything B4 added is additive and zero-valued for a campaign that neither
    commands conditions nor settles, so nothing such an operator already reads
    may move. Verified module-against-module against ``HEAD``'s own
    ``preflight``: 11 spec shapes — budgets, recipes, presets, overrides, a
    broken PCB, stock, a shortfall, purge billing — identical on every field,
    ``describe()`` included.

    The twelfth shape is **not** identical and must not be, which is why the
    boundary is stated as *settling* rather than as *carrying a run plan*: see
    :class:`TestASettleSpelledInTheFlatFieldsIsBilledToo`.
    """

    def test_the_three_new_totals_are_zero(self, catalog):
        p = project_campaign(_spec(), catalog=catalog)
        assert (p.approach_ceiling_s, p.settle_ceiling_s, p.settle_floor_s) == (
            0.0, 0.0, 0.0)

    def test_the_iteration_time_is_still_exactly_the_workflow_time(self, catalog):
        p = project_campaign(_spec(), catalog=catalog)
        assert p.per_iteration_s == p.workflow_s

    def test_the_summary_grows_no_new_lines(self, catalog):
        text = project_campaign(_spec(), catalog=catalog).describe()
        for phrase in ("approaching commanded conditions",
                       "equilibrate window", "RH-floor", "commands"):
            assert phrase not in text

    def test_a_default_projection_carries_no_rh_advisory(self, catalog):
        """No run plan means no commanded humidity, so there is nothing to
        advise about -- and, critically, no "not consulted" note either."""
        p = project_campaign(_spec(), catalog=catalog)
        assert not any("humidity" in w for w in p.warnings)


class TestASettleSpelledInTheFlatFieldsIsBilledToo:
    """**Operator-visible, and deliberate.** A campaign can ask to settle in two
    ways — an EQUILIBRATE phase in a ``run_plan``, or the flat
    ``equilibration_method = "settle"`` fields — and ``settle_plan()`` is the one
    authority over both. The window is the same rig time either way, so the
    projection bills it either way.

    Every ``equilibration_method = "settle"`` campaign that exists today
    therefore projects **longer** than it did before, by exactly the
    ``max_hold_s`` per trial it was already spending and never reporting. That
    is the under-report being fixed, not a new cost.
    """

    def _spec(self, **over):
        return _spec(equilibration_method="settle", round_period_s=240.0,
                     min_hold_s=600.0, max_hold_s=3600.0, **over)

    def test_the_flat_spelling_gets_the_same_ceiling(self, catalog):
        p = project_campaign(self._spec(), catalog=catalog)
        assert p.settle_ceiling_s == pytest.approx(3600.0)
        assert p.settle_floor_s == pytest.approx(600.0)
        assert p.per_iteration_s == pytest.approx(p.workflow_s + 3600.0)

    def test_a_campaign_that_does_not_settle_keeps_its_old_number(self, catalog):
        """The boundary, stated as the pair: settling is what adds the window,
        not the presence of a run plan."""
        assert project_campaign(_spec(), catalog=catalog).settle_ceiling_s == 0.0

    def test_no_conditions_are_commanded_so_no_approach_is_billed(self, catalog):
        """The flat spelling carries setpoints for nothing, so the *other* new
        window stays zero — the two are independent."""
        p = project_campaign(self._spec(), catalog=catalog)
        assert p.approach_ceiling_s == 0.0
        assert not any("RH-floor" in w for w in p.warnings)


class TestTheRHFloorAdvisory:
    """Advisory, never a refusal -- ``analysis/rh_floor.py``'s own warning
    forbids fitting a threshold to these numbers, so the wording is quoted from
    it."""

    @staticmethod
    def _bin(temp, floor, asked, n=12) -> TemperatureBin:
        return TemperatureBin(temperature_C=temp, rh_floor_pct=floor,
                              rh_setpoint_min_pct=asked, n_rows=n)

    def _plan(self, *, temp=85.0, rh=20.0) -> RunPlan:
        return RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     conditions=_conditions("anneal", temp, rh)),
        ))

    def test_a_setpoint_below_a_probed_floor_is_advised(self):
        # asked 15, delivered 22.0 -> saturated: the floor bounds observation.
        lines = rh_floor_advisories(self._plan(rh=20.0),
                                    [self._bin(85.0, 22.0, 15.0)])
        assert len(lines) == 1
        assert "do not command below this" in lines[0]
        assert "Advisory only" in lines[0]

    def test_the_advisory_quotes_the_modules_own_numbers_not_a_literal(self):
        """No constant may be baked in here: what the chamber reaches is read
        from its own history, per bin, at projection time. The 19.5-23.2 %RH
        figure that circulated as "the floor at 85 C" is an *approach window* --
        the same block later held at 14.9 -- so a literal would have been wrong
        as well as unmaintainable."""
        lines = rh_floor_advisories(self._plan(rh=10.0),
                                    [self._bin(85.0, 14.9, 12.0)])
        assert "14.9 %RH" in lines[0]
        assert "19.5" not in lines[0] and "23.2" not in lines[0]

    def test_an_unprobed_bin_says_so_in_the_modules_words(self):
        """Delivered 22 when asked for 22: the setpoint was met, so the real
        minimum is unknown and the line must not read as a limit."""
        lines = rh_floor_advisories(self._plan(rh=20.0),
                                    [self._bin(85.0, 22.0, 22.0)])
        assert len(lines) == 1
        assert "floor not probed" in lines[0]

    def test_a_setpoint_at_or_above_the_observed_floor_is_silent(self):
        assert rh_floor_advisories(self._plan(rh=30.0),
                                   [self._bin(85.0, 22.0, 15.0)]) == []

    def test_a_temperature_in_no_observed_bin_says_the_chamber_was_never_watched(self):
        """A floor seen at 25 C says nothing about 85 C — so it is not quoted at
        85 C, *and* the fact that nothing was observed there is stated. Silence
        would be the same sentence as "checked, and fine"."""
        lines = rh_floor_advisories(self._plan(temp=85.0),
                                    [self._bin(25.0, 22.0, 15.0)])
        assert len(lines) == 1
        assert "never been observed" in lines[0]
        assert "22.0" not in lines[0], "the 25 C bin must not be quoted at 85 C"

    def test_the_three_outcomes_are_three_distinguishable_strings(self):
        """Below a probed floor / below an unprobed minimum / never watched —
        the whole point is that a reader can tell which one they have."""
        below_probed = rh_floor_advisories(
            self._plan(rh=20.0), [self._bin(85.0, 22.0, 15.0)])[0]
        below_unprobed = rh_floor_advisories(
            self._plan(rh=20.0), [self._bin(85.0, 22.0, 22.0)])[0]
        unwatched = rh_floor_advisories(self._plan(rh=20.0), [])[0]
        assert len({below_probed, below_unprobed, unwatched}) == 3

    def test_a_phase_driving_humidity_but_not_temperature_gets_nothing(self):
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     conditions=PhaseSetpoints("anneal", rh_setpoint_pct=20.0)),
        ))
        assert rh_floor_advisories(plan, [self._bin(85.0, 22.0, 15.0)]) == []

    def test_no_observed_bins_at_all_is_still_reported_per_condition(self):
        """An empty history is "never watched" for every commanded condition,
        not a clean bill for any of them."""
        lines = rh_floor_advisories(self._plan(), [])
        assert len(lines) == 1
        assert "never been observed on this project" in lines[0]


    def test_a_repeated_condition_is_advised_once(self):
        """The engine establishes a repeated setpoint once, so advising twice
        would misrepresent what the run does."""
        conditions = _conditions("anneal", 85.0, 20.0)
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, conditions=conditions),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH, conditions=conditions),
        ))
        assert len(commanded_conditions(plan)) == 1
        assert len(rh_floor_advisories(plan, [self._bin(85.0, 22.0, 15.0)])) == 1


class TestNotConsultedIsNotACleanBill:
    """The failure shape this guards: a plan that commands humidity and a
    projection that read no history produce the same silence as a plan checked
    and found clean. "Unknown" must not be spelled the way "clean" is."""

    def test_a_humidity_commanding_plan_with_no_project_dir_says_so(self, catalog):
        p = project_campaign(_spec(run_plan=_bench_plan(), batch=True),
                             catalog=catalog)
        assert any("no RH-floor history was consulted" in w for w in p.warnings)

    def test_an_advisory_reaches_the_operator_through_warnings_and_describe(
            self, catalog, monkeypatch):
        """The whole delivery seam, end to end, without a DataStore.

        ``describe()`` renders every warning as a ``Note:`` line, and both
        surfaces that project a campaign already print ``describe()`` — so the
        advisory needs no new display path anywhere. That is the property being
        pinned: computing the advisory and then dropping it on the floor would
        otherwise look exactly like finding nothing to say.
        """
        import softae.core.preflight as preflight

        monkeypatch.setattr(
            preflight, "_rh_floor_bins",
            lambda _dir: [TemperatureBin(temperature_C=85.0, rh_floor_pct=22.0,
                                         rh_setpoint_min_pct=15.0, n_rows=9)])

        p = project_campaign(_spec(run_plan=_bench_plan(), batch=True),
                             catalog=catalog, project_dir="anywhere")

        assert any("do not command below this" in w for w in p.warnings)
        assert "Note: condition 'anneal' commands 20 %RH at 85 °C" in p.describe()

    def test_a_history_that_was_read_and_holds_nothing_says_that_instead(
            self, catalog, monkeypatch):
        """``None`` and ``[]`` are different answers and must not share a line:
        nobody looked, versus looked and the chamber has never been watched."""
        import softae.core.preflight as preflight

        monkeypatch.setattr(preflight, "_rh_floor_bins", lambda _dir: [])
        p = project_campaign(_spec(run_plan=_bench_plan(), batch=True),
                             catalog=catalog, project_dir="anywhere")

        assert any("never been observed on this project" in w for w in p.warnings)
        assert not any("was consulted" in w for w in p.warnings)

    def test_the_advisory_changes_no_verdict_and_no_duration(
            self, catalog, monkeypatch):
        """It is a note, never a gate. The same list carries the stock
        shortfall, which *is* a gate (``_project`` prompts on
        ``stock_sufficient is False``), so the two must be shown not to be
        coupled: the worst possible advisory and none at all must produce the
        same numbers and the same verdict."""
        import softae.core.preflight as preflight

        led = ReservoirLedger()
        led.refill(0, 1000.0)
        led.refill(1, 1000.0)
        spec = _spec(run_plan=_bench_plan(), batch=True, budget=50)

        quiet = project_campaign(spec, catalog=catalog, ledger=led)
        monkeypatch.setattr(
            preflight, "_rh_floor_bins",
            lambda _dir: [TemperatureBin(temperature_C=85.0, rh_floor_pct=99.0,
                                         rh_setpoint_min_pct=1.0, n_rows=9)])
        loud = project_campaign(spec, catalog=catalog, ledger=led,
                                project_dir="anywhere")

        assert any("do not command below this" in w for w in loud.warnings)
        assert (loud.per_iteration_s, loud.settle_ceiling_s,
                loud.approach_ceiling_s) == (
            quiet.per_iteration_s, quiet.settle_ceiling_s,
            quiet.approach_ceiling_s)
        assert loud.stock_sufficient == quiet.stock_sufficient is False
        assert loud.iterations_supported() == quiet.iterations_supported()

    def test_a_plan_that_drives_no_humidity_is_silent(self, catalog):
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
                     conditions=PhaseSetpoints("cast", temp_setpoint_C=25.0)),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
        ))
        p = project_campaign(_spec(run_plan=plan, batch=True), catalog=catalog)
        assert not any("RH-floor" in w for w in p.warnings)


class TestAdvisoriesNeverRefuse:
    def test_a_settle_spelled_twice_warns_rather_than_raising(self, catalog):
        """``settle_plan()`` refuses a spec that names settle twice. That refusal
        belongs to the launch path; a projection that cannot run is not a
        preflight, it is an outage."""
        spec = _spec(run_plan=_bench_plan(), batch=True,
                     equilibration_method="settle")
        p = project_campaign(spec, catalog=catalog)
        assert any("Equilibrate window not projected" in w for w in p.warnings)
        assert p.settle_ceiling_s == 0.0
        assert p.per_iteration_s > 0

    def test_an_unreadable_project_dir_is_a_missing_advisory_not_a_failure(
            self, catalog, tmp_path):
        p = project_campaign(_spec(run_plan=_bench_plan(), batch=True),
                             catalog=catalog, project_dir=tmp_path / "nothing-here")
        assert p.per_iteration_s > 0
        assert any("no RH-floor history was consulted" in w for w in p.warnings)


class TestMeasurementBlockOnTheMeasurePhase:
    def test_a_denser_production_preset_does_not_change_the_projection_shape(
            self, catalog):
        """A per-phase preset changes the *sweep*, not which windows exist.
        Pinned because the projection reads ``spec.eis_preset``, and a reader
        could reasonably expect the phase override to be picked up here too --
        it is not, and that is a known gap rather than an accident."""
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                     measurement=MeasurementSpec(preset="Extended")),
        ))
        p = project_campaign(_spec(run_plan=plan, batch=True), catalog=catalog)
        assert p.settle_ceiling_s == 0.0
        assert p.per_iteration_s == p.workflow_s


# ─────────────────────── the campaign-level [conditions] baseline (T11.28) ───────────────────────

BASELINE = PhaseSetpoints(name="baseline", temp_setpoint_C=25.0,
                          rh_setpoint_pct=40.0)


def _with_baseline(spec, baseline=BASELINE):
    """Attach a campaign-level baseline to *spec*.

    Set as an attribute rather than passed to the constructor: ``CampaignSpec``
    lives in another session's file and does not declare the field yet, and
    ``preflight`` reads it through ``getattr(spec, "conditions", None)``
    throughout. This works identically once the dataclass declares it.
    """
    spec.conditions = baseline
    return spec


def _silent_first_phase_plan() -> RunPlan:
    """The rung-3a shape: a FORMULATE phase that deliberately drives nothing.

    Nothing is cast in that run, so a casting humidity would only push four
    settled films off 22 %RH -- which is exactly how a campaign comes to run its
    startup flush with no axis driven at all.
    """
    return RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 anneal_task="anneal_85C_8h",
                 conditions=_conditions("anneal", 85.0, 20.0)),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
    ))


class TestTheUnconditionedStartAdvisory:
    """A5: an advisory, never a refusal -- an advisory commands no humidifier."""

    def test_preflight_advisory_when_first_phase_drives_no_axis(self, catalog):
        p = project_campaign(_spec(run_plan=_silent_first_phase_plan(),
                                   batch=True), catalog=catalog)
        assert any("unconditioned until the first phase" in w
                   for w in p.warnings)
        assert any("startup flush" in w for w in p.warnings)

    def test_preflight_no_advisory_when_baseline_declared(self, catalog):
        """The check can pass as well as fail -- the declaration silences it."""
        spec = _with_baseline(_spec(run_plan=_silent_first_phase_plan(),
                                    batch=True))
        p = project_campaign(spec, catalog=catalog)
        assert not any("unconditioned until the first phase" in w
                       for w in p.warnings)

    def test_preflight_no_advisory_when_the_first_phase_drives_an_axis(self):
        """One axis is enough: the advisory is about silence, not completeness."""
        assert unconditioned_start_warnings(
            _spec(run_plan=_bench_plan())) == []
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
                     conditions=PhaseSetpoints("cast", temp_setpoint_C=25.0)),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
        ))
        assert unconditioned_start_warnings(_spec(run_plan=plan)) == []

    def test_preflight_no_advisory_for_a_campaign_with_no_run_plan(self, catalog):
        """A legacy pointwise campaign never commanded the chamber at all.

        That is an older and different situation from a written plan whose first
        phase stays silent, and saying this sentence about it would put a note on
        every campaign that projects today.
        """
        assert unconditioned_start_warnings(_spec()) == []
        text = project_campaign(_spec(), catalog=catalog).describe()
        assert "unconditioned until the first phase" not in text


class TestTheBaselineIsCommandedAndCostsNoGatedTime:

    def test_preflight_approach_ceiling_unchanged_by_baseline(self):
        """``approach_ceiling_s`` sums condition steps whose method is ``wait``.

        The baseline emits none, so its number must not move -- which is correct
        rather than a gap: a commanded setpoint nothing waits for costs no gated
        time. Built through the engine rather than asserted about it, so a
        baseline that acquired a wait would be caught here.
        """
        from softae.core.deposition_recipe import (
            build_recipe_deposition_workflow,
            get_deposition_recipe,
        )

        cat = TaskCatalog.load_toml(loader.tasks_toml_path())
        kw = dict(catalog=cat, pump_ids=[0, 1], dispense_rate=100.0,
                  flush_rate=500.0, flush_factor=3.0, settle_factor=2.0,
                  pcb={"grid": [8, 4], "spacing_mm": [10, 10]},
                  origin_xy=(43.5, 50.0), run_plan=_bench_plan())
        recipe = get_deposition_recipe("single_drop")
        bare = build_recipe_deposition_workflow(recipe, [21], {21: [1.0, 1.0]}, **kw)
        with_baseline = build_recipe_deposition_workflow(
            recipe, [21], {21: [1.0, 1.0]}, baseline=BASELINE, **kw)

        assert approach_ceiling_s(with_baseline) == approach_ceiling_s(bare)
        assert any(s.tags.get("baseline") == "true"
                   for s in with_baseline.setup), "the baseline was not emitted"

    def test_preflight_commanded_conditions_puts_the_baseline_at_the_head(self):
        """Where the engine commands it: ahead of every phase."""
        plan = _bench_plan()
        assert commanded_conditions(plan, BASELINE)[0] is BASELINE
        assert commanded_conditions(plan, BASELINE)[1:] == commanded_conditions(plan)
        assert commanded_conditions(plan) == commanded_conditions(plan, None)

    def test_preflight_commanded_conditions_dedups_a_first_phase_equal_to_it(self):
        """The engine emits nothing for a matching first phase; nor does this.

        Advising twice about one approach would misrepresent what the run does.
        """
        first = _conditions("casting", 25.0, 40.0)
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE, conditions=first),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
        ))
        assert commanded_conditions(plan, first) == [first]

    def test_preflight_rh_floor_advisory_sees_the_baselines_humidity(self):
        """Without this it is the one commanded humidity the advisory cannot see."""
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH),
        ))
        bins = [TemperatureBin(temperature_C=25.0, rh_floor_pct=30.0,
                               rh_setpoint_min_pct=25.0, n_rows=12)]
        dry = PhaseSetpoints("baseline", temp_setpoint_C=25.0, rh_setpoint_pct=10.0)

        assert rh_floor_advisories(plan, bins) == []
        lines = rh_floor_advisories(plan, bins, baseline=dry)
        assert len(lines) == 1
        assert "condition 'baseline' commands 10 %RH" in lines[0]

    def test_rh_floor_warnings_baseline_without_a_run_plan_is_advised(
        self, catalog, tmp_path
    ):
        """A baseline is a commanded humidity even with no plan at all.

        ``_rh_floor_warnings`` used to short-circuit on ``run_plan is None``; the
        baseline is commanded before any phase exists, so the guard is now the
        commanded set itself. Untested, that behaviour would be indistinguishable
        from the advisory never having been reached.
        """
        spec = _with_baseline(_spec(),
                              PhaseSetpoints("baseline", temp_setpoint_C=25.0,
                                             rh_setpoint_pct=10.0))
        p = project_campaign(spec, catalog=catalog,
                             project_dir=tmp_path / "nothing-here")
        assert spec.run_plan is None
        assert any("no RH-floor history was consulted" in w for w in p.warnings)

    def test_rh_floor_warnings_no_baseline_and_no_run_plan_stays_silent(
        self, catalog, tmp_path
    ):
        """The companion: nothing commanded, nothing said -- and no "not
        consulted" note either, which is the half that would hide a check that
        had simply stopped running."""
        p = project_campaign(_spec(), catalog=catalog,
                             project_dir=tmp_path / "nothing-here")
        assert not any("RH-floor" in w for w in p.warnings)
