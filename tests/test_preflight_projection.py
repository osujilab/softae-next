"""Campaign preflight: stock feasibility and projected duration (P5.2).

Duration is reported as a **rate with bounds**, never a single ETA — a Bayesian
campaign stops on a convergence criterion, not a known iteration count, so a
confident finish time would be a fabrication.
"""

from __future__ import annotations

import pytest

from softae.config import loader
from softae.core.autonomous_wiring import CampaignSpec
from softae.core.eis_scripts import EISParams
from softae.core.preflight import (
    CampaignProjection,
    estimate_eis_duration,
    estimate_step_duration,
    estimate_workflow_duration,
    per_iteration_draw,
    project_campaign,
)
from softae.core.reservoir import ReservoirLedger
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
