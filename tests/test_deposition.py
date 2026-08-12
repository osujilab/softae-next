from __future__ import annotations

import dataclasses

import pytest

from softae.core.deposition import (
    WellGeometry,
    carrier_component_keys,
    simulate_plate_deposition,
    simulate_well_deposition,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
    compute_elution_volumes,
)


@pytest.fixture
def catalog():
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    cat.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    return cat


@pytest.fixture
def dep25_elution(catalog):
    """Aggregate dep fraction 0.25, target 20 uL -> 80 eluted (dep 20, carrier 60)."""
    sol = Solution("s25", [
        SolutionComponent("Ethanol", "dep", 1.0, "mL"),
        SolutionComponent("Water", "carrier", 3.0, "mL"),
    ])
    return compute_elution_volumes({"s25": sol}, catalog, 20.0)


@pytest.fixture
def solutions_25():
    return {
        "s25": Solution("s25", [
            SolutionComponent("Ethanol", "dep", 1.0, "mL"),
            SolutionComponent("Water", "carrier", 3.0, "mL"),
        ])
    }


@pytest.fixture
def zero_elution(catalog):
    """Carrier-only, no explicit fraction -> grand_total_uL == 0."""
    sol = Solution("water", [SolutionComponent("Water", "carrier", 10.0, "mL")])
    return compute_elution_volumes({"water": sol}, catalog, 20.0)


@pytest.fixture
def well_5x2():
    return WellGeometry(5.0, 2.0)


# ── WellGeometry ──


class TestWellGeometry:
    def test_geometry_area_matches_pi_r_squared(self):
        assert WellGeometry(5.0, 2.0).area_mm2 == pytest.approx(19.6350, abs=1e-3)

    def test_geometry_capacity_is_area_times_depth(self):
        assert WellGeometry(5.0, 2.0).capacity_uL == pytest.approx(39.2699, abs=1e-3)

    def test_geometry_zero_diameter_raises_valueerror(self):
        with pytest.raises(ValueError):
            WellGeometry(0.0, 2.0)

    def test_geometry_negative_depth_raises_valueerror(self):
        with pytest.raises(ValueError):
            WellGeometry(5.0, -1.0)

    def test_geometry_frozen_rejects_mutation(self, well_5x2):
        with pytest.raises(dataclasses.FrozenInstanceError):
            well_5x2.diameter_mm = 3.0


# ── simulate_well_deposition ──


class TestSimulateWellDeposition:
    def test_full_dispense_none_uses_grand_total(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, None)
        assert r.dispensed_uL == pytest.approx(80.0)

    def test_partial_dispense_scales_dep_and_carrier_linearly(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        assert r.dep_uL == pytest.approx(10.0)
        assert r.carrier_uL == pytest.approx(30.0)

    def test_zero_evaporation_final_equals_wet(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 0.0, 40.0)
        assert r.final_volume_uL == pytest.approx(r.dispensed_uL)
        assert r.evaporated_uL == pytest.approx(0.0)

    def test_full_evaporation_final_equals_dep_only(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 100.0, 40.0)
        assert r.final_volume_uL == pytest.approx(10.0)
        assert r.residual_carrier_uL == pytest.approx(0.0)

    def test_intermediate_evaporation_matches_hand_calc(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        assert r.evaporated_uL == pytest.approx(28.5)
        assert r.residual_carrier_uL == pytest.approx(1.5)
        assert r.final_volume_uL == pytest.approx(11.5)
        assert r.final_thickness_um == pytest.approx(585.7, abs=0.1)
        assert r.final_fill_fraction == pytest.approx(0.2928, abs=1e-3)

    def test_wet_thickness_matches_volume_over_area(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        assert r.wet_thickness_um == pytest.approx(2037.2, abs=0.1)

    def test_overfill_sets_overflow_flag_without_raising(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        assert r.overflows is True
        assert r.wet_fill_fraction == pytest.approx(40.0 / well_5x2.capacity_uL)

    @pytest.mark.parametrize("pct", [-1.0, 100.1])
    def test_evaporation_pct_out_of_range_raises_valueerror(self, dep25_elution, well_5x2, pct):
        with pytest.raises(ValueError):
            simulate_well_deposition(dep25_elution, well_5x2, pct, 40.0)

    def test_dispense_exceeding_eluted_raises_valueerror(self, dep25_elution, well_5x2):
        with pytest.raises(ValueError):
            simulate_well_deposition(dep25_elution, well_5x2, 95.0, 100.0)

    def test_zero_grand_total_returns_all_zero_result(self, zero_elution, well_5x2):
        assert zero_elution.grand_total_uL == pytest.approx(0.0)
        r = simulate_well_deposition(zero_elution, well_5x2, 95.0, None)
        assert r.dispensed_uL == pytest.approx(0.0)
        assert r.final_volume_uL == pytest.approx(0.0)
        assert r.wet_thickness_um == pytest.approx(0.0)
        assert r.overflows is False

    def test_negative_dispense_raises_valueerror(self, dep25_elution, well_5x2):
        with pytest.raises(ValueError):
            simulate_well_deposition(dep25_elution, well_5x2, 95.0, -1.0)


# ── component breakdown ──


class TestComponentBreakdown:
    def test_carrier_component_keys_classifies_roles(self):
        sols = {
            "s": Solution("s", [
                SolutionComponent("A", "dep", 1.0, "mL"),
                SolutionComponent("B", "solute", 1.0, "mL"),
                SolutionComponent("C", "active", 1.0, "mL"),
                SolutionComponent("D", "carrier", 1.0, "mL"),
            ])
        }
        keys = carrier_component_keys(sols)
        assert keys == {("s", "D")}

    def test_component_finals_sum_to_final_volume(self, dep25_elution, solutions_25, well_5x2):
        ck = carrier_component_keys(solutions_25)
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0, carrier_keys=ck)
        assert sum(r.component_final_uL.values()) == pytest.approx(r.final_volume_uL)

    def test_carrier_components_retain_one_minus_f(self, dep25_elution, solutions_25, well_5x2):
        ck = carrier_component_keys(solutions_25)
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0, carrier_keys=ck)
        # Water carrier: wet = 60 * 0.5 = 30; retained = 30 * (1 - 0.95) = 1.5
        assert r.component_final_uL[("s25", "Water")] == pytest.approx(1.5)
        # Ethanol dep: fully retained = 20 * 0.5 = 10
        assert r.component_final_uL[("s25", "Ethanol")] == pytest.approx(10.0)

    def test_without_carrier_keys_component_breakdown_empty(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        assert r.component_final_uL == {}


# ── simulate_plate_deposition ──


class TestSimulatePlateDeposition:
    def test_equal_split_default_divides_grand_total(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, None)
        assert [w.dispensed_uL for w in s.wells] == [pytest.approx(40.0), pytest.approx(40.0)]

    def test_scalar_dispense_applied_to_every_well(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, 30.0)
        assert all(w.dispensed_uL == pytest.approx(30.0) for w in s.wells)

    def test_heterogeneous_list_gives_distinct_well_results(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, [50.0, 20.0])
        assert s.wells[0].wet_thickness_um != pytest.approx(s.wells[1].wet_thickness_um)
        assert s.wells[0].dispensed_uL == pytest.approx(50.0)
        assert s.wells[1].dispensed_uL == pytest.approx(20.0)

    def test_list_length_mismatch_raises_valueerror(self, dep25_elution, well_5x2):
        with pytest.raises(ValueError):
            simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, [40.0])

    def test_dispense_sum_exceeding_eluted_raises_valueerror(self, dep25_elution, well_5x2):
        with pytest.raises(ValueError):
            simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, 45.0)

    def test_undeposited_remainder_is_eluted_minus_dispensed(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, 35.0)
        assert s.total_dispensed_uL == pytest.approx(70.0)
        assert s.undeposited_uL == pytest.approx(10.0)

    def test_mass_balance_dispensed_equals_evaporated_plus_final(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, 40.0)
        assert s.total_eluted_uL == pytest.approx(s.total_dispensed_uL + s.undeposited_uL)
        assert s.total_dispensed_uL == pytest.approx(s.total_evaporated_uL + s.total_final_uL)
        assert s.total_evaporated_uL == pytest.approx(57.0)
        assert s.total_final_uL == pytest.approx(23.0)


# ── reporting ──


class TestReporting:
    def test_well_result_as_dict_flattens_component_keys(
        self, dep25_elution, solutions_25, well_5x2
    ):
        ck = carrier_component_keys(solutions_25)
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0, carrier_keys=ck)
        d = r.as_dict()
        assert "s25 / Water" in d["component_final_uL"]
        assert "s25 / Ethanol" in d["component_final_uL"]

    def test_well_result_summary_lines_flags_overflow(self, dep25_elution, well_5x2):
        r = simulate_well_deposition(dep25_elution, well_5x2, 95.0, 40.0)
        text = "\n".join(r.summary_lines())
        assert "wet volume exceeds well capacity" in text

    def test_summary_as_dict_contains_mass_balance_fields(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, 40.0)
        d = s.as_dict()
        for key in (
            "total_eluted_uL",
            "total_dispensed_uL",
            "undeposited_uL",
            "total_evaporated_uL",
            "total_final_uL",
        ):
            assert key in d

    def test_summary_any_overflow_true_when_any_well_overflows(self, dep25_elution, well_5x2):
        s = simulate_plate_deposition(dep25_elution, well_5x2, 95.0, 2, [40.0, 20.0])
        assert s.wells[0].overflows is True
        assert s.wells[1].overflows is False
        assert s.any_overflow is True
