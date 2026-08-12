"""Authoritative deposit area and the thickness inverse solve (P7.2 / P7.3).

Thickness math already existed, but its *area* came from GUI spin boxes
defaulting to a 5 mm disc — so a headless campaign had no area at all and a GUI
one silently used a default no board declared. A thickness target is meaningless
without a board-tied area, which is why 7.2 comes first.
"""

from __future__ import annotations

import pytest

from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
    ThicknessTarget,
    TotalDepositTarget,
    solve_formulation,
)
from softae.core.geometry import (
    deposit_area_mm2,
    thickness_um,
    volume_for_thickness_uL,
)


class TestDepositArea:
    def test_the_electrode_footprint_is_the_default(self):
        """0.2 × 0.2 cm = 0.04 cm² = 4 mm²."""
        pcb = {"electrode_L_cm": 0.2, "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) == pytest.approx(4.0)

    def test_an_explicit_area_wins(self):
        """The escape hatch for a wetted area that is not the electrode."""
        pcb = {"electrode_L_cm": 0.2, "electrode_w_cm": 0.2,
               "deposit_area_mm2": 12.5}
        assert deposit_area_mm2(pcb) == 12.5

    def test_a_board_declaring_neither_reports_none(self):
        """Callers must treat thickness as unavailable, never substitute a guess."""
        assert deposit_area_mm2({}) is None

    def test_a_zero_footprint_reports_none(self):
        assert deposit_area_mm2({"electrode_L_cm": 0.0, "electrode_w_cm": 0.2}) is None

    def test_a_junk_explicit_area_falls_back_to_the_footprint(self):
        pcb = {"electrode_L_cm": 0.2, "electrode_w_cm": 0.2,
               "deposit_area_mm2": "wide"}
        assert deposit_area_mm2(pcb) == pytest.approx(4.0)

    def test_a_walled_board_resolves_and_a_sessile_one_declines(self):
        """This replaces an assertion that *every* shipped board resolves an area.

        That premise turned out to be wrong rather than merely unmet: the IDE board
        has no wells, so its cast is a free droplet whose wetted area is set by
        volume and contact angle. Nothing on the board predicts it, and the value
        the old assertion was passing on (the electrode rectangle) described the gap
        between two stripes. Declining is the correct answer there.
        """
        from softae.config import loader

        pcbs = (loader.load().get("pcb") or {})
        assert pcbs
        assert deposit_area_mm2(pcbs["SoftAE_EIS_4Stripe"]) == pytest.approx(
            18.7038, abs=1e-3)
        assert deposit_area_mm2(pcbs["SoftAE_IDE_EIS"]) is None


class TestThicknessMath:
    def test_one_microlitre_over_one_square_mm_is_a_millimetre(self):
        assert thickness_um(1.0, 1.0) == pytest.approx(1000.0)

    def test_the_inverse_round_trips(self):
        vol = volume_for_thickness_uL(0.3, 4.0)
        assert thickness_um(vol, 4.0) == pytest.approx(0.3)

    def test_a_non_positive_area_is_refused_both_ways(self):
        with pytest.raises(ValueError):
            thickness_um(1.0, 0.0)
        with pytest.raises(ValueError):
            volume_for_thickness_uL(1.0, 0.0)


class TestThicknessTargetValidation:
    def test_an_unknown_basis_is_refused(self):
        with pytest.raises(ValueError, match="dry.*wet"):
            ThicknessTarget(0.3, area_mm2=4.0, basis="damp")

    def test_a_missing_area_is_refused_loudly(self):
        """A guessed area silently corrupts every thickness derived from it."""
        with pytest.raises(ValueError, match="positive deposit area"):
            ThicknessTarget(0.3, area_mm2=0.0)

    def test_the_target_reports_its_volume(self):
        assert ThicknessTarget(0.5, area_mm2=4.0).volume_uL() == pytest.approx(0.002)


@pytest.fixture
def stocks():
    """Two stocks: one 10 % solids, one pure solvent."""
    chem = ChemicalCatalog()
    chem.add(Chemical("PEO", "C2H4O", density_g_per_mL=1.2,
                      molar_mass_g_per_mol=44.0))
    chem.add(Chemical("Water", "O", density_g_per_mL=1.0,
                      molar_mass_g_per_mol=18.0))

    solids = Solution("PEO 10%", [
        SolutionComponent("PEO", "dep", 1.0, "mL"),
        SolutionComponent("Water", "carrier", 9.0, "mL"),
    ])
    solvent = Solution("Water", [SolutionComponent("Water", "carrier", 10.0, "mL")])
    return chem, {"PEO 10%": solids, "Water": solvent}


class TestSolve:
    def test_a_dry_thickness_matches_the_equivalent_deposit_target(self, stocks):
        """The dry basis reduces exactly to TotalDepositTarget under full
        solvent loss — which is why it needs no new solver machinery."""
        chem, sols = stocks
        area = 4.0
        target_um = 0.5
        equivalent_uL = volume_for_thickness_uL(target_um, area)

        by_thickness = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="dry")])
        by_volume = solve_formulation(
            sols, chem, [TotalDepositTarget(equivalent_uL)])

        for name in sols:
            assert by_thickness.per_stock_uL[name] == pytest.approx(
                by_volume.per_stock_uL[name])

    def test_a_wet_thickness_fixes_the_dispensed_volume(self, stocks):
        """Independent of any evaporation assumption — total dispensed is the
        quantity being held."""
        chem, sols = stocks
        area, target_um = 4.0, 2.5
        plan = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="wet")])

        total = plan.grand_total_uL
        assert total == pytest.approx(volume_for_thickness_uL(target_um, area))

    def test_wet_and_dry_differ_when_solvent_is_lost(self, stocks):
        """If they agreed, the basis distinction would be decorative."""
        chem, sols = stocks
        area, target_um = 4.0, 2.5
        dry = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="dry")])
        wet = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="wet")])

        assert dry.grand_total_uL != pytest.approx(wet.grand_total_uL)

    def test_a_wet_target_needs_more_volume_than_a_dry_one(self, stocks):
        """Only part of what is dispensed stays behind, so hitting the same
        number as a *dried* thickness takes more liquid."""
        chem, sols = stocks
        area, target_um = 4.0, 2.5
        dry = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="dry")])
        wet = solve_formulation(
            sols, chem, [ThicknessTarget(target_um, area_mm2=area, basis="wet")])

        assert dry.grand_total_uL > wet.grand_total_uL
