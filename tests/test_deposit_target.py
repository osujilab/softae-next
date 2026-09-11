"""The one resolver that turns a declared scale into a solver row (W1).

``docs/SubAgent docs/thickness_target_all_paths.md``. Three things are pinned
here, and each was a way the system could have been silently wrong:

* **D1** — a dry thickness is coupled to ``[deposition] evaporation_pct``.
  Before this, a dry target held the *deposited* volume while the twin reported
  *deposit + residual carrier*; the two agree only at 100 % loss, and at 80 %
  with a 5 % stock a 50 µm target cast a ~250 µm film with nothing red. The
  partial-evaporation test below is the positive control for the fix: against
  the unmodified row it fails by ~5×.
* **D4** — a thickness on the legacy-identity path is refused, not approximated.
* **no invented area** — a sessile board declines rather than guessing.
"""

from __future__ import annotations

import pytest

from softae.core.deposit_target import (
    DepositTargetSpec,
    ThicknessUnresolvable,
    resolve_area_mm2,
    scale_for_thickness,
    scale_target,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
    ThicknessTarget,
    TotalDepositTarget,
    solve_formulation,
)

#: A walled board that declares a real deposit area, and one that cannot.
WALLED = "SoftAE_EIS_4Stripe"
SESSILE = "SoftAE_IDE_EIS"


@pytest.fixture
def stocks():
    """One 5 %-solids stock and one pure carrier."""
    chem = ChemicalCatalog()
    chem.add(Chemical("PEO", "C2H4O", density_g_per_mL=1.2,
                      molar_mass_g_per_mol=44.0))
    chem.add(Chemical("Water", "O", density_g_per_mL=1.0,
                      molar_mass_g_per_mol=18.0))
    solids = Solution("PEO 5%", [
        SolutionComponent("PEO", "dep", 0.5, "mL"),
        SolutionComponent("Water", "carrier", 9.5, "mL"),
    ])
    carrier = Solution("Water", [SolutionComponent("Water", "carrier", 10.0, "mL")])
    return chem, {"PEO 5%": solids}, {"Water": carrier}


class TestDepositTargetSpec:
    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="kind"):
            DepositTargetSpec("depth", 4.0)

    def test_an_unknown_basis_is_refused(self):
        with pytest.raises(ValueError, match="basis"):
            DepositTargetSpec("thickness", 60.0, basis="damp")

    def test_a_non_positive_value_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            DepositTargetSpec("volume", 0.0)


class TestAreaAuthority:
    def test_a_walled_board_resolves_its_well_area(self):
        assert resolve_area_mm2(WALLED) == pytest.approx(18.7038, abs=1e-3)

    def test_a_sessile_board_declines(self):
        assert resolve_area_mm2(SESSILE) is None


class TestScaleTarget:
    def test_no_deposit_target_reproduces_the_total_deposit_row(self):
        """The default path must be unchanged by the whole feature."""
        target = scale_target(None, fallback_uL=4.5, pcb_name=WALLED)

        assert target == TotalDepositTarget(4.5)

    def test_no_deposit_target_and_no_fallback_is_refused(self):
        with pytest.raises(ValueError, match="no scale row"):
            scale_target(None, fallback_uL=None, pcb_name=WALLED)

    def test_a_volume_kind_ignores_the_board_entirely(self):
        """A declared volume needs no area, so a sessile board is fine."""
        target = scale_target(DepositTargetSpec("volume", 4.5),
                              fallback_uL=None, pcb_name=SESSILE)

        assert target == TotalDepositTarget(4.5)

    def test_scale_target_refuses_a_sessile_board_rather_than_guessing_an_area(self):
        """An invented area silently corrupts every thickness derived from it."""
        with pytest.raises(ThicknessUnresolvable, match="no deposit area"):
            scale_target(DepositTargetSpec("thickness", 60.0),
                         fallback_uL=None, pcb_name=SESSILE)

    def test_scale_target_thickness_dry_matches_total_deposit_target_when_evaporation_is_100(
        self, stocks
    ):
        """D1's r = 0 limit: full carrier loss IS the deposited volume."""
        chem, solids, carrier = stocks
        sols = {**solids, **carrier}
        area = resolve_area_mm2(WALLED)
        target_um = 50.0

        by_thickness = solve_formulation(sols, chem, [scale_target(
            DepositTargetSpec("thickness", target_um, "dry"),
            fallback_uL=None, pcb_name=WALLED, evaporation_pct=100.0)])
        by_volume = solve_formulation(
            sols, chem, [TotalDepositTarget(target_um * area / 1000.0)])

        for name in sols:
            assert by_thickness.per_stock_uL[name] == pytest.approx(
                by_volume.per_stock_uL[name])

    def test_scale_target_dry_basis_agrees_with_the_twin_at_partial_evaporation(
        self, stocks
    ):
        """THE positive control for D1. Against the old row this fails ~5×.

        Solve a 50 µm dry target at 80 % carrier loss with a 5 % stock, then ask
        the deposition twin — the authority on what actually stays in the well —
        what film those volumes leave.
        """
        from softae.core.deposition import (
            WellGeometry,
            simulate_well_deposition,
        )
        from softae.core.formulation import elution_from_stock_volumes

        chem, solids, _carrier = stocks
        area = resolve_area_mm2(WALLED)
        target_um, evaporation = 50.0, 80.0

        plan = solve_formulation(solids, chem, [scale_target(
            DepositTargetSpec("thickness", target_um, "dry"),
            fallback_uL=None, pcb_name=WALLED, evaporation_pct=evaporation)])

        elution = elution_from_stock_volumes(plan.per_stock_uL, solids, chem)
        twin = simulate_well_deposition(
            elution, WellGeometry.from_board(area, 200.0), evaporation)

        assert twin.final_thickness_um == pytest.approx(target_um)

        # The positive control, in-tree rather than by mutation: the row this
        # replaced is exactly the r = 0 row, so solving at 100 % and casting at
        # 80 % reproduces the old behaviour. It misses by ~5×, which is what
        # makes the assertion above a discriminating one rather than a tautology.
        old = solve_formulation(solids, chem, [scale_target(
            DepositTargetSpec("thickness", target_um, "dry"),
            fallback_uL=None, pcb_name=WALLED, evaporation_pct=100.0)])
        old_twin = simulate_well_deposition(
            elution_from_stock_volumes(old.per_stock_uL, solids, chem),
            WellGeometry.from_board(area, 1000.0), evaporation)
        assert old_twin.final_thickness_um > 4.0 * target_um

    def test_a_wet_basis_is_independent_of_the_evaporation_setting(self, stocks):
        """Wet holds the as-dispensed volume, so no drying model enters it."""
        chem, solids, _carrier = stocks
        plans = [
            solve_formulation(solids, chem, [scale_target(
                DepositTargetSpec("thickness", 50.0, "wet"),
                fallback_uL=None, pcb_name=WALLED, evaporation_pct=e)])
            for e in (0.0, 55.0, 100.0)
        ]

        totals = [p.grand_total_uL for p in plans]
        assert totals[1] == pytest.approx(totals[0])
        assert totals[2] == pytest.approx(totals[0])

    def test_scale_target_and_the_twin_read_the_same_evaporation_setting(
        self, monkeypatch, stocks
    ):
        """Agreement is by construction — one parse point, moved once."""
        from softae.core import deposition
        from softae.core.autonomous_wiring import simulate_cast

        chem, solids, _carrier = stocks
        monkeypatch.setattr(deposition, "evaporation_pct", lambda *a, **k: 60.0)

        target = scale_target(DepositTargetSpec("thickness", 50.0, "dry"),
                              fallback_uL=None, pcb_name=WALLED)
        assert isinstance(target, ThicknessTarget)
        assert target.evaporation_pct == pytest.approx(60.0)

        plan = solve_formulation(solids, chem, [target])
        twin = simulate_cast(plan.per_stock_uL, solids, chem, pcb_name=WALLED,
                             capacity_uL=200.0)

        assert twin.evaporation_pct == pytest.approx(60.0)
        assert twin.final_thickness_um == pytest.approx(50.0)

    def test_a_thickness_target_on_a_legacy_identity_spec_is_refused(self):
        """D4. The legacy path searches per-pump volumes: no stock identity, no
        ``dep_fraction``, so every coefficient the dry row needs is undefined —
        and the wet row (1.0) would 'work' while ignoring the searched params."""
        with pytest.raises(ThicknessUnresolvable, match="no composition solve"):
            scale_target(DepositTargetSpec("thickness", 60.0, "dry"),
                         fallback_uL=None, pcb_name=WALLED, dep_fractions=())

        with pytest.raises(ThicknessUnresolvable, match="no composition solve"):
            scale_target(DepositTargetSpec("thickness", 60.0, "wet"),
                         fallback_uL=None, pcb_name=WALLED, dep_fractions=())

    def test_an_all_carrier_stock_set_is_refused_on_the_dry_basis(self):
        """Nothing dries, so no volume reaches any dry thickness."""
        with pytest.raises(ThicknessUnresolvable, match="pure carrier"):
            scale_target(DepositTargetSpec("thickness", 60.0, "dry"),
                         fallback_uL=None, pcb_name=WALLED,
                         evaporation_pct=100.0, dep_fractions=(0.0, 0.0))

    def test_an_all_carrier_stock_set_is_fine_when_nothing_evaporates(self):
        """The control: at 0 % loss the whole cast stays, so a dry thickness is
        reachable and refusing would be the wrong answer."""
        target = scale_target(DepositTargetSpec("thickness", 60.0, "dry"),
                              fallback_uL=None, pcb_name=WALLED,
                              evaporation_pct=0.0, dep_fractions=(0.0, 0.0))

        assert isinstance(target, ThicknessTarget)


class TestScaleForThickness:
    def test_scale_for_thickness_hits_the_target_exactly_for_a_known_stock(
        self, stocks
    ):
        """HT's inverse is closed form: thickness(k·v) == k·thickness(v)."""
        from softae.core.deposition import (
            WellGeometry,
            simulate_well_deposition,
        )
        from softae.core.formulation import elution_from_stock_volumes

        chem, solids, carrier = stocks
        sols = {**solids, **carrier}
        area = resolve_area_mm2(WALLED)
        typed = {"PEO 5%": 8.0, "Water": 2.0}

        scaled = scale_for_thickness(typed, sols, chem, target_um=30.0,
                                     basis="dry", pcb_name=WALLED,
                                     evaporation_pct=90.0)

        # Composition is untouched — only the scale moved.
        assert (scaled["PEO 5%"] / scaled["Water"]) == pytest.approx(
            typed["PEO 5%"] / typed["Water"])
        twin = simulate_well_deposition(
            elution_from_stock_volumes(scaled, sols, chem),
            WellGeometry.from_board(area, 500.0), 90.0)
        assert twin.final_thickness_um == pytest.approx(30.0)

    def test_the_wet_basis_targets_the_as_dispensed_volume(self, stocks):
        chem, solids, carrier = stocks
        sols = {**solids, **carrier}
        area = resolve_area_mm2(WALLED)

        scaled = scale_for_thickness({"PEO 5%": 8.0, "Water": 2.0}, sols, chem,
                                     target_um=30.0, basis="wet",
                                     pcb_name=WALLED, evaporation_pct=90.0)

        assert sum(scaled.values()) == pytest.approx(30.0 * area / 1000.0)

    def test_scale_for_thickness_refuses_an_all_carrier_stock_rather_than_scaling_to_infinity(
        self, stocks
    ):
        """k = target / 0 is not a large number, it is an absent answer."""
        chem, _solids, carrier = stocks

        with pytest.raises(ThicknessUnresolvable, match="zero thickness"):
            scale_for_thickness({"Water": 10.0}, carrier, chem, target_um=30.0,
                                basis="dry", pcb_name=WALLED,
                                evaporation_pct=100.0)

    def test_scale_for_thickness_refuses_a_sessile_board(self, stocks):
        chem, solids, _carrier = stocks

        with pytest.raises(ThicknessUnresolvable, match="no deposit area"):
            scale_for_thickness({"PEO 5%": 8.0}, solids, chem, target_um=30.0,
                                pcb_name=SESSILE)

    def test_a_non_positive_target_is_refused(self, stocks):
        chem, solids, _carrier = stocks

        with pytest.raises(ValueError, match="target_um"):
            scale_for_thickness({"PEO 5%": 8.0}, solids, chem, target_um=0.0,
                                pcb_name=WALLED)
