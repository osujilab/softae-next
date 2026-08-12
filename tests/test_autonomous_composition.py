"""AE composition path: {eo_li_ratio, silica_vol_frac} → dropcast, end-to-end.

Exercises the seam wired in this change — the autonomous loop turns a *composition*
suggestion into per-pump volumes via ``plan_formulation`` (the same call the GUI
uses), with the per-electrode budget riding in from the board's ``well_capacity_uL``
and an infeasible cast blocked before it reaches hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.config import loader
from softae.core.autonomous_wiring import (
    CampaignSpec,
    GeneralFormulation,
    build_trial_workflow,
    deposit_step_name,
    plan_trial,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.core.formulation import (
    Basis,
    Chemical,
    ChemicalCatalog,
    DriedFractionTarget,
    FormulationContext,
    FormulationInfeasibleError,
    MolarRatioTarget,
    Solution,
    SolutionComponent,
)
from softae.core.task_catalog import TaskCatalog
from softae.drivers.mock_factory import create_mock_manager

# Composition axes the optimizer searches (not raw volumes).
SPACE = {
    "eo_li_ratio": {"type": "float", "low": 5.0, "high": 40.0},
    "silica_vol_frac": {"type": "float", "low": 0.0, "high": 0.2},
}
POINT = {"eo_li_ratio": 10.0, "silica_vol_frac": 0.1}


def _context(*, budget_uL=None, target=6.0) -> FormulationContext:
    """Physically-correct PEO(25 wt%)/LiCl(10 M)/silica stocks at a 300 µL basis."""
    cat = ChemicalCatalog()
    cat.add(Chemical("PEO", density_g_per_mL=1.21, molar_mass_g_per_mol=44.0))
    cat.add(Chemical("PEO_solvent", density_g_per_mL=1.0))
    cat.add(Chemical("LiCl", density_g_per_mL=2.07, molar_mass_g_per_mol=42.39))
    cat.add(Chemical("salt_water", density_g_per_mL=1.0))
    cat.add(Chemical("SiO2", density_g_per_mL=2.2, is_particulate=True))
    cat.add(Chemical("silica_solvent", density_g_per_mL=1.0))
    peo = Solution("PEO", [
        SolutionComponent("PEO", "dep", 5.0, "g"),          # → 5.94 M (25 wt%)
        SolutionComponent("PEO_solvent", "carrier", 15.0, "mL"),
    ])
    licl = Solution("LiCl", [
        # Solute (kept in the EO:Li ratio) but excluded from the dried film — the
        # salt's bulk volume is neglected, expressed on the stock, not in code.
        SolutionComponent("LiCl", "solute", 21.2, "g", counts_as_deposit=False),  # → 9.98 M
        SolutionComponent("salt_water", "carrier", 39.858, "mL"),
    ])
    silica = Solution("Silica", [
        SolutionComponent("SiO2", "dep", 1.0, "g"),
        SolutionComponent("silica_solvent", "carrier", 9.0, "mL"),
    ])
    return FormulationContext(
        peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=cat,
        pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
        target_deposition_uL=target,
        peo_dried_frac=0.222, silica_dried_frac=0.04,
        peo_basis_uL=300.0, budget_uL=budget_uL,
    )


def _board_capacity_uL() -> float:
    """The 4-stripe board's per-electrode budget, read from the board itself.

    Derived from the declared well rather than written down here, so a corrected
    dimension moves the expectation with it instead of failing this test.
    """
    from softae.config import loader
    from softae.core.geometry import well_capacity_uL

    return well_capacity_uL(loader.pcb_configs()["SoftAE_EIS_4Stripe"])


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="comp_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",   # 32-ch board; capacity DERIVED from its well
        parameter_space=SPACE,
        formulation=_context(),
        pump_ids=(0, 1, 2),              # PEO / LiCl / Silica
        optimizer="random",
        time_scale=0.0,
        budget=4,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
def catalog() -> TaskCatalog:
    return TaskCatalog.load_toml(loader.tasks_toml_path())


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


# ── Composition → per-pump volumes ───────────────────────────────────────────


def test_plan_trial_returns_three_pump_volumes():
    plan = plan_trial(_spec(), POINT)
    assert len(plan.per_pump_uL) == 3          # PEO, LiCl, Silica
    assert plan.per_pump_uL[0] > 0             # PEO always present
    assert plan.feasible is True
    assert plan.achieved["eo_li_ratio"] == pytest.approx(10.0, rel=1e-9)


def test_build_trial_injects_composition_volumes(catalog):
    """The deposit step carries the 3 per-pump volumes from plan_formulation."""
    spec = _spec()
    wf = build_trial_workflow(spec, POINT, catalog=catalog)
    deposit = next(s for s in wf.setup if s.name == deposit_step_name(21))
    vols = deposit.params["vols"]
    assert len(vols) == 3
    assert vols == pytest.approx(plan_trial(spec, POINT).per_pump_uL)
    assert deposit.params["ids"] == [0, 1, 2]


def test_infeasible_composition_blocked_before_hardware(catalog):
    """A cast over the per-electrode budget raises rather than overflowing."""
    spec = _spec(formulation=_context(budget_uL=10.0))   # 10 µL cap, casts ~30–55
    with pytest.raises(FormulationInfeasibleError):
        build_trial_workflow(spec, POINT, catalog=catalog)


# ── End-to-end campaign in mock ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composition_campaign_runs_end_to_end(connected, tmp_path: Path):
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    spec = _spec(budget=4)
    assert spec.formulation.budget_uL is None    # unset → should ride in from board

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store, on_event=events.append,
    )

    # The full composition → formulation → dropcast → EIS → tell loop ran.
    assert result.n_trials == 4
    assert result.best_params is not None
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 4
    # The searched params are composition axes, not volumes.
    assert set(spec.parameter_space) == {"eo_li_ratio", "silica_vol_frac"}
    # Per-electrode budget rode in from the 4-stripe board's well capacity.
    # Asserted against the board rather than a literal: the capacity is now derived
    # from the declared well (4.88 mm across, 6.35 mm deep), and pinning the number
    # here is what made this test fail when the hand-typed 120 uL was corrected to
    # the true 118.769 uL brim volume.
    expected = _board_capacity_uL()
    budget_events = [e for e in events if e["type"] == "budget_from_board"]
    assert budget_events
    assert budget_events[0]["well_capacity_uL"] == pytest.approx(expected)
    assert spec.formulation.budget_uL == pytest.approx(expected)
    store.close()


# ── General composition mode (arbitrary solve_formulation targets) ───────────


def _general(*, budget_uL=None, target=6.0) -> GeneralFormulation:
    """A general-target campaign over the PEO/LiCl/silica stocks."""
    ctx = _context(target=target)
    licl = ctx.licl_stock
    licl.components[0].counts_as_deposit = False  # salt excluded from the dried film

    def build_targets(p):
        return [MolarRatioTarget("PEO", "LiCl", p["eo_li_ratio"]),
                DriedFractionTarget("SiO2", p["silica_vol_frac"], Basis.VOLUME)]

    return GeneralFormulation(
        stocks={"PEO": ctx.peo_stock, "LiCl": licl, "Silica": ctx.silica_stock},
        catalog=ctx.catalog, pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
        target_deposition_uL=target, build_targets=build_targets, budget_uL=budget_uL,
    )


def test_general_trial_injects_three_pump_volumes(catalog):
    spec = _spec(formulation=None, general_formulation=_general())
    wf = build_trial_workflow(spec, POINT, catalog=catalog)
    deposit = next(s for s in wf.setup if s.name == deposit_step_name(21))
    assert len(deposit.params["vols"]) == 3
    assert deposit.params["ids"] == [0, 1, 2]


def test_general_infeasible_blocked_before_hardware(catalog):
    spec = _spec(formulation=None, general_formulation=_general(budget_uL=10.0))
    with pytest.raises(FormulationInfeasibleError):
        build_trial_workflow(spec, POINT, catalog=catalog)


@pytest.mark.asyncio
async def test_general_campaign_runs_end_to_end(connected, tmp_path):
    store = DataStore(tmp_path / "proj")
    events: list[dict] = []
    gf = _general()  # budget None → should ride in from the board
    spec = _spec(formulation=None, general_formulation=gf, budget=4)

    result = await run_autonomous_campaign(
        spec, manager=connected, data_store=store, on_event=events.append,
    )

    assert result.n_trials == 4
    assert result.best_params is not None
    expected = _board_capacity_uL()          # 4-stripe well capacity rode in
    assert gf.budget_uL == pytest.approx(expected)
    assert any(e["type"] == "budget_from_board"
               and e["well_capacity_uL"] == pytest.approx(expected)
               for e in events)
    store.close()


# ── Composition axes → a live campaign (the Live BO tab's composition mode) ──


class TestCampaignFromCompositionAxes:
    """The seam the Live BO tab's composition mode rides on.

    The tab turns its axes table into a ``GeneralFormulation``; everything past that
    is the path already exercised above. What is worth pinning here is that a spec
    built *from axes* behaves like one written by hand — including the consequence
    the operator actually cares about, which is that composition mode makes
    conductivity the objective where volume mode cannot.
    """

    def _axes(self):
        from softae.core.composition_axes import CompositionAxis

        return [
            CompositionAxis("molar_ratio", "PEO", "LiCl", low=5.0, high=40.0),
            CompositionAxis("dried_fraction", "SiO2", low=0.0, high=0.2),
        ]

    def _spec_from_axes(self, **over):
        from softae.core.composition_axes import (
            axes_parameter_space,
            build_targets_from_axes,
        )

        axes = self._axes()
        ctx = _context()
        base = dict(
            name="axes_campaign", channels=(21, 22),
            pcb_name="SoftAE_EIS_4Stripe",
            parameter_space=axes_parameter_space(axes),
            general_formulation=GeneralFormulation(
                stocks={"PEO": ctx.peo_stock, "LiCl": ctx.licl_stock,
                        "Silica": ctx.silica_stock},
                catalog=ctx.catalog,
                pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
                target_deposition_uL=6.0,
                build_targets=build_targets_from_axes(axes),
            ),
            pump_ids=(0, 1, 2), optimizer="random", time_scale=0.0, budget=2, seed=7,
        )
        base.update(over)
        return CampaignSpec(**base)

    def test_the_axes_become_the_searched_parameter_space(self):
        spec = self._spec_from_axes()
        assert set(spec.parameter_space) == {"ratio_PEO_LiCl", "driedfrac_SiO2"}
        assert spec.vol_params == ()      # volumes are solved, never searched

    def test_a_suggestion_becomes_three_concrete_pump_volumes(self, catalog):
        spec = self._spec_from_axes()
        point = {"ratio_PEO_LiCl": 10.0, "driedfrac_SiO2": 0.1}
        wf = build_trial_workflow(spec, point, catalog=catalog)
        deposit = next(s for s in wf.setup if s.name == deposit_step_name(21))
        assert len(deposit.params["vols"]) == 3
        assert deposit.params["ids"] == [0, 1, 2]
        assert all(v >= 0 for v in deposit.params["vols"])

    def test_the_twin_predicts_a_thickness_which_is_the_whole_point(self):
        # Volume mode cannot do this — no stock identity, so no elution, so no dry
        # thickness. It is what makes σ available as the objective.
        from softae.core.autonomous_wiring import simulate_trial

        twin = simulate_trial(self._spec_from_axes(),
                              {"ratio_PEO_LiCl": 10.0, "driedfrac_SiO2": 0.1})
        assert twin is not None and twin.final_thickness_um > 0

    def test_composition_mode_resolves_the_objective_to_maximising_conductivity(self):
        from softae.core.autonomous_wiring import resolve_direction, resolve_objective

        spec = self._spec_from_axes()
        kind, _reason = resolve_objective(spec)
        assert kind == "sigma"
        assert resolve_direction(spec) == ("maximize", "sigma")

    def test_a_pinned_axis_is_held_constant_across_every_suggestion(self, catalog):
        from softae.core.composition_axes import (
            CompositionAxis,
            axes_parameter_space,
            build_targets_from_axes,
        )

        axes = [CompositionAxis("molar_ratio", "PEO", "LiCl", low=5.0, high=40.0),
                CompositionAxis("dried_fraction", "SiO2", low=0.1, high=0.1)]
        assert set(axes_parameter_space(axes)) == {"ratio_PEO_LiCl"}
        build = build_targets_from_axes(axes)
        for ratio in (6.0, 30.0):
            frac = next(t for t in build({"ratio_PEO_LiCl": ratio})
                        if isinstance(t, DriedFractionTarget))
            assert frac.value == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_a_composition_axes_campaign_runs_end_to_end(connected, tmp_path: Path):
    """The whole path: axes → solver → volumes → cast → measure → tell."""
    store = DataStore(tmp_path / "proj")
    spec = TestCampaignFromCompositionAxes()._spec_from_axes(budget=2)

    result = await run_autonomous_campaign(spec, manager=connected, data_store=store)

    assert result.n_trials == 2
    rows = store.query_doe_parameters(run_id=result.run_id)
    assert len(rows) == 2
    # Every trial recorded the formulation it cast, so a thickness exists per well.
    assert store.predicted_thickness_um(result.run_id, 21) is not None
    store.close()
