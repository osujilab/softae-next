"""General declarative formulation solver — arbitrary stocks + composition targets.

Proves the linear solver (a) reproduces the ternary EO:Li/silica preset exactly and
(b) generalizes to any stock set and any mix of target types (molar ratio,
dried-fraction on volume/mass/mole bases, absolute concentration, total deposit),
including multi-cation salts via the ``provides`` species map.
"""

from __future__ import annotations

import math

import pytest

from softae.core.formulation import (
    Basis,
    Chemical,
    ChemicalCatalog,
    ConcentrationTarget,
    DriedFractionTarget,
    FormulationContext,
    MolarRatioTarget,
    Solution,
    SolutionCatalog,
    SolutionComponent,
    TotalDepositTarget,
    compute_elution_volumes,
    elution_from_stock_volumes,
    plan_formulation,
    solve_formulation,
    species_concentration,
)


def _ternary():
    cat = ChemicalCatalog()
    cat.add(Chemical("PEO", density_g_per_mL=1.21, molar_mass_g_per_mol=44.0))
    cat.add(Chemical("PEO_solvent", density_g_per_mL=1.0))
    cat.add(Chemical("LiCl", density_g_per_mL=2.07, molar_mass_g_per_mol=42.39))
    cat.add(Chemical("salt_water", density_g_per_mL=1.0))
    cat.add(Chemical("SiO2", density_g_per_mL=2.2, is_particulate=True))
    cat.add(Chemical("silica_solvent", density_g_per_mL=1.0))
    peo = Solution("PEO", [
        SolutionComponent("PEO", "dep", 5.0, "g"),
        SolutionComponent("PEO_solvent", "carrier", 15.0, "mL"),
    ])
    # LiCl excluded from the dried film (its salt volume is neglected) — the
    # counts_as_deposit split, which the ternary preset hardcodes.
    licl = Solution("LiCl", [
        SolutionComponent("LiCl", "dep", 21.2, "g", counts_as_deposit=False),
        SolutionComponent("salt_water", "carrier", 39.858, "mL"),
    ])
    silica = Solution("Silica", [
        SolutionComponent("SiO2", "dep", 1.0, "g"),
        SolutionComponent("silica_solvent", "carrier", 9.0, "mL"),
    ])
    return cat, {"PEO": peo, "LiCl": licl, "Silica": silica}


# ── Equivalence: the ternary preset is one instance of the general solver ─────


class TestTernaryEquivalence:
    @pytest.mark.parametrize("eo,sil", [(40.0, 0.0), (10.0, 0.1), (5.0, 0.2)])
    def test_matches_plan_formulation(self, eo, sil):
        cat, stocks = _ternary()
        ctx = FormulationContext(
            peo_stock=stocks["PEO"], licl_stock=stocks["LiCl"], silica_stock=stocks["Silica"],
            catalog=cat, pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
            target_deposition_uL=13.5, peo_dried_frac=0.222, silica_dried_frac=0.04,
            peo_basis_uL=300.0,
        )
        preset = plan_formulation({"eo_li_ratio": eo, "silica_vol_frac": sil}, ctx)
        general = solve_formulation(
            stocks, cat,
            [MolarRatioTarget("PEO", "LiCl", eo),
             DriedFractionTarget("SiO2", sil, Basis.VOLUME),
             TotalDepositTarget(13.5)],
            pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
            dried_frac={"PEO": 0.222, "Silica": 0.04},
        )
        assert general.per_pump_uL == pytest.approx(preset.per_pump_uL, rel=1e-9)
        assert general.grand_total_uL == pytest.approx(preset.grand_total_uL, rel=1e-9)
        assert general.achieved[f"ratio[PEO/LiCl]"] == pytest.approx(eo, rel=1e-9)

    def test_licl_deposit_governed_by_its_flag_not_code(self):
        """The preset special-cases no species: LiCl's dried contribution is decided
        by its own ``counts_as_deposit`` flag, uniformly with every other component.

        Excluded (flag False) → the film is PEO + silica only.  Counted (a plain dep
        component) → LiCl joins the fixed 13.5 µL film, so PEO shrinks by exactly the
        LiCl dried volume and the total cast is a hair smaller.  Either way the EO:Li
        ratio (intensive) and the silica fraction (pinned to the total) are identical.
        """
        cat, stocks = _ternary()
        base = dict(
            peo_stock=stocks["PEO"], silica_stock=stocks["Silica"], catalog=cat,
            pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
            target_deposition_uL=13.5, peo_dried_frac=0.222, silica_dried_frac=0.04,
            peo_basis_uL=300.0,
        )
        point = {"eo_li_ratio": 5.0, "silica_vol_frac": 0.1}   # salt-rich → biggest Δ

        excl_licl = stocks["LiCl"]                             # counts_as_deposit=False
        incl_licl = Solution("LiCl", [                         # plain dep → counted
            SolutionComponent("LiCl", "dep", 21.2, "g"),
            SolutionComponent("salt_water", "carrier", 39.858, "mL"),
        ])
        excl = plan_formulation(point, FormulationContext(licl_stock=excl_licl, **base))
        incl = plan_formulation(point, FormulationContext(licl_stock=incl_licl, **base))

        # Excluded: only PEO + silica make up the film.
        dried_excl = excl.per_stock_uL["PEO"] * 0.222 + excl.per_stock_uL["Silica"] * 0.04
        assert dried_excl == pytest.approx(13.5, rel=1e-9)

        # Counted: PEO + silica no longer sum to the target — LiCl took a real slice.
        dried_incl_matrix = (
            incl.per_stock_uL["PEO"] * 0.222 + incl.per_stock_uL["Silica"] * 0.04
        )
        assert dried_incl_matrix < 13.5
        assert incl.grand_total_uL < excl.grand_total_uL          # smaller cast
        assert incl.per_stock_uL["Silica"] == pytest.approx(      # silica untouched
            excl.per_stock_uL["Silica"], rel=1e-9)
        # Composition handles identical regardless of the accounting choice.
        for plan in (excl, incl):
            assert plan.achieved["eo_li_ratio"] == pytest.approx(5.0, rel=1e-9)
            assert plan.achieved["silica_vol_frac"] == pytest.approx(0.1, rel=1e-6)


# ── Arbitrary component sets ─────────────────────────────────────────────────


class TestGenerality:
    def _sys(self):
        """A 4-stock system: polymer P, salt S (dried-excluded), filler F, co-polymer Q."""
        cat = ChemicalCatalog()
        cat.add(Chemical("P", density_g_per_mL=1.2, molar_mass_g_per_mol=100.0))
        cat.add(Chemical("Q", density_g_per_mL=1.5, molar_mass_g_per_mol=120.0))
        cat.add(Chemical("S", density_g_per_mL=2.0, molar_mass_g_per_mol=50.0))
        cat.add(Chemical("F", density_g_per_mL=2.5, is_particulate=True))
        cat.add(Chemical("solvent", density_g_per_mL=1.0))
        stocks = {
            "polyP": Solution("polyP", [
                SolutionComponent("P", "dep", 2.0, "mL"),
                SolutionComponent("solvent", "carrier", 8.0, "mL")]),
            "salt": Solution("salt", [
                SolutionComponent("S", "solute", 1.0, "mL", counts_as_deposit=False),
                SolutionComponent("solvent", "carrier", 9.0, "mL")]),
            "filler": Solution("filler", [
                SolutionComponent("F", "dep", 1.0, "mL"),
                SolutionComponent("solvent", "carrier", 9.0, "mL")]),
            "polyQ": Solution("polyQ", [
                SolutionComponent("Q", "dep", 3.0, "mL"),
                SolutionComponent("solvent", "carrier", 7.0, "mL")]),
        }
        return cat, stocks

    def test_quaternary_hits_all_targets(self):
        cat, stocks = self._sys()
        plan = solve_formulation(
            stocks, cat,
            [MolarRatioTarget("P", "S", 8.0),
             DriedFractionTarget("F", 0.10, Basis.VOLUME),
             DriedFractionTarget("Q", 0.25, Basis.VOLUME),
             TotalDepositTarget(10.0)],
        )
        assert plan.feasible is True
        assert all(v >= 0 for v in plan.per_stock_uL.values())
        assert plan.achieved["ratio[P/S]"] == pytest.approx(8.0, rel=1e-6)
        assert plan.achieved["dried_frac[F]"] == pytest.approx(0.10, rel=1e-6)
        assert plan.achieved["dried_frac[Q]"] == pytest.approx(0.25, rel=1e-6)
        assert plan.achieved["total_deposit_uL"] == pytest.approx(10.0, rel=1e-6)
        # 4 stocks → one pump each.
        assert len(plan.per_pump_uL) == 4

    def test_mass_and_mole_bases_differ_from_volume(self):
        """Same filler fraction on different bases → different stock volumes."""
        cat, stocks = self._sys()
        common = [MolarRatioTarget("P", "S", 8.0),
                  DriedFractionTarget("Q", 0.25, Basis.VOLUME),
                  TotalDepositTarget(10.0)]
        vol = solve_formulation(stocks, cat, [DriedFractionTarget("F", 0.10, Basis.VOLUME), *common])
        mass = solve_formulation(stocks, cat, [DriedFractionTarget("F", 0.10, Basis.MASS), *common])
        assert vol.achieved["dried_frac[F]"] == pytest.approx(0.10, rel=1e-6)
        # F is dense (2.5) → the same 10% by mass needs less F volume than by volume.
        assert mass.per_stock_uL["filler"] < vol.per_stock_uL["filler"]

    def test_mole_basis_particulate_raises(self):
        cat, stocks = self._sys()
        with pytest.raises(ValueError, match="mole-basis"):
            solve_formulation(
                stocks, cat,
                [DriedFractionTarget("F", 0.1, Basis.MOLE),  # F is particulate
                 MolarRatioTarget("P", "S", 8.0),
                 DriedFractionTarget("Q", 0.25, Basis.VOLUME),
                 TotalDepositTarget(10.0)],
            )

    def test_absolute_concentration_target(self):
        """Fix an absolute species molarity in the final mix (independent of ratio)."""
        cat, stocks = self._sys()
        plan = solve_formulation(
            stocks, cat,
            [ConcentrationTarget("S", 0.05),           # 0.05 M salt in the cast
             DriedFractionTarget("F", 0.10, Basis.VOLUME),
             DriedFractionTarget("Q", 0.25, Basis.VOLUME),
             TotalDepositTarget(10.0)],
        )
        assert plan.achieved["conc[S]_M"] == pytest.approx(0.05, rel=1e-6)

    def test_multi_cation_salt_counts_species_twice(self):
        """A 2:1 salt (provides Li:2) contributes twice its molarity to Li."""
        cat = ChemicalCatalog()
        cat.add(Chemical("Li2SO4", density_g_per_mL=2.2, molar_mass_g_per_mol=110.0,
                         provides={"Li": 2.0}))
        cat.add(Chemical("water", density_g_per_mL=1.0))
        stock = Solution("s", [
            SolutionComponent("Li2SO4", "solute", 1.0, "mL"),
            SolutionComponent("water", "carrier", 9.0, "mL"),
        ])
        conc = species_concentration(stock, cat)
        # molarity of Li2SO4 solute × 2 = [Li]
        from softae.core.formulation import molarity
        m = molarity(stock, cat)["Li2SO4"]
        assert conc["Li"] == pytest.approx(2.0 * m, rel=1e-9)


# ── Determinacy diagnostics + CSV ────────────────────────────────────────────


class TestElutionAdapter:
    """elution_from_stock_volumes reconstructs the ElutionResult a manual solve made."""

    def test_roundtrip_matches_compute_elution_volumes(self):
        cat, stocks = _ternary()
        forward = compute_elution_volumes(stocks, cat, 13.5)
        rebuilt = elution_from_stock_volumes(forward.per_solution, stocks, cat)
        assert rebuilt.per_solution == pytest.approx(forward.per_solution)
        assert rebuilt.dep_vol_uL == pytest.approx(forward.dep_vol_uL)
        assert rebuilt.carrier_vol_uL == pytest.approx(forward.carrier_vol_uL)
        assert rebuilt.grand_total_uL == pytest.approx(forward.grand_total_uL)
        for key, vol in forward.component_vol_uL.items():
            assert rebuilt.component_vol_uL[key] == pytest.approx(vol)

    def test_target_defaults_to_total_dep(self):
        """Twin usage: emergent dried fractions (no override) → adapter total == target."""
        cat, stocks = _ternary()
        plan = solve_formulation(   # emergent dep_fraction, as the twin calls it
            stocks, cat,
            [MolarRatioTarget("PEO", "LiCl", 10.0),
             DriedFractionTarget("SiO2", 0.1, Basis.VOLUME),
             TotalDepositTarget(13.5)],
        )
        el = elution_from_stock_volumes(plan.per_stock_uL, stocks, cat)
        assert el.total_dep_uL == pytest.approx(13.5, rel=1e-6)
        assert el.target_deposition_uL == pytest.approx(13.5, rel=1e-6)


class TestDiagnosticsAndPersistence:
    def test_underdetermined_is_flagged(self):
        cat, stocks = TestGenerality()._sys()
        plan = solve_formulation(   # 4 stocks, only 2 targets
            stocks, cat,
            [MolarRatioTarget("P", "S", 8.0), TotalDepositTarget(10.0)],
        )
        assert any("under-determined" in n for n in plan.notes)

    def test_infeasible_over_budget_flagged(self):
        cat, stocks = _ternary()
        plan = solve_formulation(
            stocks, cat,
            [MolarRatioTarget("PEO", "LiCl", 5.0),
             DriedFractionTarget("SiO2", 0.2, Basis.VOLUME),
             TotalDepositTarget(13.5)],
            dried_frac={"PEO": 0.222, "Silica": 0.04}, budget_uL=10.0,
        )
        assert plan.feasible is False
        assert any("exceeds budget" in n for n in plan.notes)

    def test_provides_csv_roundtrip(self, tmp_path):
        cat = ChemicalCatalog()
        cat.add(Chemical("Li2SO4", molar_mass_g_per_mol=110.0, provides={"Li": 2.0}))
        cat.add(Chemical("LiCl", molar_mass_g_per_mol=42.39))  # empty → self-species
        p = tmp_path / "chems.csv"
        cat.save_csv(p)
        loaded = ChemicalCatalog.load_csv(p)
        assert loaded.get("Li2SO4").provides == {"Li": 2.0}
        assert loaded.get("Li2SO4").species_map() == {"Li": 2.0}
        assert loaded.get("LiCl").provides == {}
        assert loaded.get("LiCl").species_map() == {"LiCl": 1.0}
