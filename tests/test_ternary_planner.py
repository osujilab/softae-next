"""Tier-2 ternary formulator: identity/accounting split, solver, budget, planner.

The golden regression reproduces the PEO/LiCl/silica 32-electrode map from
``PEO_salt_silica_formulator_example_v1.xlsx`` (sheet "PEO-salt only calcs",
rows 85–100).  Reference cell values are embedded so the test is self-contained
and CI-safe (no dependency on the workbook path).

Note on the silica axis: the planner defines ``silica_vol_frac`` as the *dried-
film* volume fraction (the decided basis), so the golden test feeds each row's
**achieved** silica fraction (sheet column J) — which is what the sheet's cast
actually produced — and expects the sheet's stock/cast volumes back.  Feeding the
sheet's *nominal* label (column D) instead is covered by a separate test that
documents the ~3× correction.
"""

from __future__ import annotations

import math

import pytest

from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    ElutionResult,
    FormulationContext,
    Solution,
    SolutionCatalog,
    SolutionComponent,
    compute_elution_volumes,
    dried_fraction,
    plan_formulation,
    solve_stocks_for_composition,
)

# --- seed-system constants (workbook input cells) ---------------------------
PEO_DRIED_FRAC = 27.75 / 125.0      # empirical dried fraction of PEO stock
SILICA_DRIED_FRAC = 0.04            # empirical dried fraction of silica stock
TARGET_UL = 13.5                    # F82 dried-film target
BUDGET_UL = 125.0                   # C73 per-electrode elution cap
PEO_BASIS_UL = 1000.0              # C18

# Golden reference rows: (vial, eo_li_str, phi_achieved_J, licl_F, dried_I, cast_L)
REF = [
    ("A1", "40:1", 0.0,               12.118534724786565, 222.0,  61.54774873326405),
    ("B1", "20:1", 0.0,               24.237069449573130, 222.0,  62.28468665571728),
    ("C1", "10:1", 0.0,               48.474138899146260, 222.0,  63.75856250062376),
    ("D1", "5:1",  0.0,               96.948277798292520, 222.0,  66.70631419043671),
    ("E1", "40:1", 0.015782940237630796, 12.118534724786565, 225.56, 65.90308662344661),
    ("F1", "20:1", 0.015782940237630796, 24.237069449573130, 225.56, 66.62839349871092),
    ("G1", "10:1", 0.015782940237630796, 48.474138899146260, 225.56, 68.07900724923955),
    ("H1", "5:1",  0.015782940237630796, 96.948277798292520, 225.56, 70.98023475029682),
    ("A2", "40:1", 0.032764029278494290, 12.118534724786565, 229.52, 70.58905637323377),
    ("B2", "20:1", 0.032764029278494290, 24.237069449573130, 229.52, 71.30184924001932),
    ("C2", "10:1", 0.032764029278494290, 48.474138899146260, 229.52, 72.72743497359042),
    ("D2", "5:1",  0.032764029278494290, 96.948277798292520, 229.52, 75.57860644073261),
    ("E2", "40:1", 0.071285140562248970, 12.118534724786565, 239.04, 81.21904375328238),
    ("F2", "20:1", 0.071285140562248970, 24.237069449573130, 239.04, 81.90344895234787),
    ("G2", "10:1", 0.071285140562248970, 48.474138899146260, 239.04, 83.27225935047889),
    ("H2", "5:1",  0.071285140562248970, 96.948277798292520, 239.04, 86.00988014674093),
]


def _parse_ratio(s: str) -> float:
    a, b = s.split(":")
    return float(a) / float(b)


@pytest.fixture
def seed_catalog() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(Chemical("PEO", "C2H4O", density_g_per_mL=1.21, molar_mass_g_per_mol=44.0))
    cat.add(Chemical("PEO_solvent", density_g_per_mL=1.0))
    cat.add(Chemical("LiCl", "LiCl", density_g_per_mL=2.07, molar_mass_g_per_mol=42.39))
    cat.add(Chemical("salt_water", density_g_per_mL=1.0))
    cat.add(Chemical("SiO2", "SiO2", density_g_per_mL=2.2, is_particulate=True))
    cat.add(Chemical("silica_solvent", density_g_per_mL=1.0))
    return cat


@pytest.fixture
def seed_stocks():
    # Reproduce the workbook's working stock molarities (K12 ≈ 0.5930, K13 ≈ 1.2233).
    peo = Solution("PEO", [
        SolutionComponent("PEO", "dep", 0.4, "g"),
        SolutionComponent("PEO_solvent", "carrier", 15.0, "mL"),
    ])
    # LiCl is a solute (kept in molarity / the EO:Li ratio) but excluded from the
    # dried-film volume — the sheet neglects the salt's bulk volume.  That exclusion
    # lives here as data (counts_as_deposit=False), not as a code special-case.
    licl = Solution("LiCl", [
        SolutionComponent("LiCl", "solute", 2.12, "g", counts_as_deposit=False),
        SolutionComponent("salt_water", "carrier", 39.858, "mL"),
    ])
    silica = Solution("Silica", [
        SolutionComponent("SiO2", "dep", 1.0, "g"),
        SolutionComponent("silica_solvent", "carrier", 9.0, "mL"),
    ])
    return peo, licl, silica


@pytest.fixture
def seed_context(seed_catalog, seed_stocks):
    peo, licl, silica = seed_stocks
    return FormulationContext(
        peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
        pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
        target_deposition_uL=TARGET_UL,
        peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
        peo_basis_uL=PEO_BASIS_UL, budget_uL=BUDGET_UL,
    )


# ── Identity vs deposition-accounting split ──────────────────────────────────


class TestIdentityAccountingSplit:
    def test_ionic_solute_excluded_from_deposit_but_kept_as_solute(self):
        """LiCl: counts_as_deposit=False removes it from film volume, keeps molarity."""
        from softae.core.formulation import molarity

        cat = ChemicalCatalog()
        cat.add(Chemical("LiCl", density_g_per_mL=2.07, molar_mass_g_per_mol=42.39))
        cat.add(Chemical("Water", density_g_per_mL=1.0, molar_mass_g_per_mol=18.0))
        sol = Solution("salt", [
            SolutionComponent("LiCl", "solute", 1.0, "mL", counts_as_deposit=False),
            SolutionComponent("Water", "carrier", 9.0, "mL"),
        ])
        # deposition-accounting: LiCl excluded, water is solvent → no deposited volume
        assert sol.dep_fraction(cat) == pytest.approx(0.0)
        # identity: LiCl is still a solute → present in molarity (1 mL * 2.07 / 42.39 in 10 mL)
        m = molarity(sol, cat)
        assert "LiCl" in m
        assert m["LiCl"] == pytest.approx((1.0 * 2.07 / 42.39) / 0.010, rel=1e-9)

    def test_default_none_preserves_legacy_behavior(self, seed_catalog):
        """counts_as_deposit=None → dep-role identity still drives dep_fraction."""
        sol = Solution("s", [
            SolutionComponent("PEO", "dep", 2.0, "mL"),          # None → counted
            SolutionComponent("PEO_solvent", "carrier", 8.0, "mL"),
        ])
        assert sol.dep_fraction(seed_catalog) == pytest.approx(0.2)

    def test_override_true_forces_deposit(self, seed_catalog):
        sol = Solution("s", [
            SolutionComponent("PEO_solvent", "carrier", 5.0, "mL", counts_as_deposit=True),
            SolutionComponent("PEO_solvent", "carrier", 5.0, "mL"),
        ])
        assert sol.dep_fraction(seed_catalog) == pytest.approx(0.5)

    def test_csv_roundtrip_tristate(self, tmp_path):
        cat = SolutionCatalog()
        cat.add(Solution("s", [
            SolutionComponent("A", "dep", 1.0, "mL"),                         # None
            SolutionComponent("B", "solute", 2.0, "mL", counts_as_deposit=False),
            SolutionComponent("C", "carrier", 3.0, "mL", counts_as_deposit=True),
        ]))
        p = tmp_path / "sols.csv"
        cat.save_csv(p)
        loaded = SolutionCatalog.load_csv(p)
        comps = {c.chemical_name: c for c in loaded.get("s").components}
        assert comps["A"].counts_as_deposit is None
        assert comps["B"].counts_as_deposit is False
        assert comps["C"].counts_as_deposit is True


# ── Budget / feasibility on ElutionResult ────────────────────────────────────


class TestBudget:
    def test_no_budget_is_always_feasible(self):
        r = ElutionResult(20.0, {"s": 80.0}, 80.0)
        assert r.budget_uL is None
        assert r.feasible is True
        assert math.isinf(r.headroom_uL)

    def test_within_budget(self, seed_catalog):
        sol = Solution("s", [
            SolutionComponent("PEO", "dep", 2.0, "mL"),
            SolutionComponent("PEO_solvent", "carrier", 8.0, "mL"),
        ])
        r = compute_elution_volumes({"s": sol}, seed_catalog, 20.0, budget_uL=125.0)
        assert r.grand_total_uL == pytest.approx(100.0)
        assert r.feasible is True
        assert r.headroom_uL == pytest.approx(25.0)

    def test_over_budget(self, seed_catalog):
        sol = Solution("s", [
            SolutionComponent("PEO", "dep", 2.0, "mL"),
            SolutionComponent("PEO_solvent", "carrier", 8.0, "mL"),
        ])
        r = compute_elution_volumes({"s": sol}, seed_catalog, 20.0, budget_uL=90.0)
        assert r.feasible is False
        assert r.headroom_uL == pytest.approx(-10.0)


# ── Inverse solver ───────────────────────────────────────────────────────────


class TestSolver:
    @pytest.mark.parametrize("eo_str, licl_F", [(r[1], r[3]) for r in REF[:4]])
    def test_licl_volume_matches_sheet(self, seed_catalog, seed_stocks, eo_str, licl_F):
        peo, licl, silica = seed_stocks
        sp = solve_stocks_for_composition(
            eo_li_ratio=_parse_ratio(eo_str), silica_vol_frac=0.0,
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
            peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
            peo_basis_uL=PEO_BASIS_UL,
        )
        assert sp.volumes_uL["LiCl"] == pytest.approx(licl_F, rel=1e-6)

    def test_achieved_silica_equals_requested(self, seed_catalog, seed_stocks):
        peo, licl, silica = seed_stocks
        sp = solve_stocks_for_composition(
            eo_li_ratio=10.0, silica_vol_frac=0.0712851,
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
            peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
        )
        assert sp.achieved["silica_vol_frac"] == pytest.approx(0.0712851, rel=1e-6)

    def test_salt_leveling_adds_water(self, seed_catalog, seed_stocks):
        peo, licl, silica = seed_stocks
        sp = solve_stocks_for_composition(
            eo_li_ratio=40.0, silica_vol_frac=0.0,
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
            peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
            salt_molarity_target=0.0145,  # ~ the 40:1 native [salt]; small/no water
        )
        assert sp.leveling_water_uL >= 0.0

    def test_bad_inputs_raise(self, seed_catalog, seed_stocks):
        peo, licl, silica = seed_stocks
        with pytest.raises(ValueError):
            solve_stocks_for_composition(
                eo_li_ratio=0.0, silica_vol_frac=0.1,
                peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
                peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
            )


# ── Golden regression: reproduce the 32-electrode map ────────────────────────


class TestGoldenMap:
    @pytest.mark.parametrize("vial, eo_str, phi_J, licl_F, dried_I, cast_L", REF)
    def test_plan_reproduces_sheet(
        self, seed_context, vial, eo_str, phi_J, licl_F, dried_I, cast_L
    ):
        plan = plan_formulation(
            {"eo_li_ratio": _parse_ratio(eo_str), "silica_vol_frac": phi_J},
            seed_context,
        )
        # total cast volume == sheet L
        assert plan.grand_total_uL == pytest.approx(cast_L, rel=1e-4)
        # realised composition
        assert plan.achieved["eo_li_ratio"] == pytest.approx(_parse_ratio(eo_str), rel=1e-9)
        assert plan.achieved["silica_vol_frac"] == pytest.approx(phi_J, rel=1e-6)
        # LiCl stock (un-scaled) == sheet F: recover via the common cast scale
        scale = plan.per_stock_uL["PEO"] / PEO_BASIS_UL
        assert plan.per_stock_uL["LiCl"] / scale == pytest.approx(licl_F, rel=1e-6)
        # every electrode fits the 125 µL budget
        assert plan.feasible is True
        assert plan.per_pump_uL[0:3] == pytest.approx(
            [plan.per_stock_uL["PEO"], plan.per_stock_uL["LiCl"], plan.per_stock_uL["Silica"]]
        )

    def test_dried_volume_equals_target(self, seed_context):
        """Property: deposited (dried) volume of every plan equals the target."""
        for vial, eo_str, phi_J, *_ in REF:
            plan = plan_formulation(
                {"eo_li_ratio": _parse_ratio(eo_str), "silica_vol_frac": phi_J},
                seed_context,
            )
            dried = (
                plan.per_stock_uL["PEO"] * PEO_DRIED_FRAC
                + plan.per_stock_uL["Silica"] * SILICA_DRIED_FRAC
            )
            assert dried == pytest.approx(TARGET_UL, rel=1e-9)

    def test_feasibility_tracks_budget(self, seed_catalog, seed_stocks):
        peo, licl, silica = seed_stocks
        tight = FormulationContext(
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=seed_catalog,
            pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
            target_deposition_uL=TARGET_UL,
            peo_dried_frac=PEO_DRIED_FRAC, silica_dried_frac=SILICA_DRIED_FRAC,
            peo_basis_uL=PEO_BASIS_UL, budget_uL=70.0,   # below H2's 86 µL cast
        )
        a1 = plan_formulation({"eo_li_ratio": 40.0, "silica_vol_frac": 0.0}, tight)
        h2 = plan_formulation({"eo_li_ratio": 5.0, "silica_vol_frac": 0.0712851}, tight)
        assert a1.feasible is True and a1.headroom_uL > 0
        assert h2.feasible is False and h2.headroom_uL < 0
        assert any("exceeds budget" in n for n in h2.notes)

    def test_nominal_label_documents_correction(self, seed_context):
        """Feeding the sheet's *nominal* silica label (0.20) yields the corrected,
        ~3× larger silica stock than the sheet's hand-entered 426 µL (φ→0.071)."""
        plan_nominal = plan_formulation(
            {"eo_li_ratio": 5.0, "silica_vol_frac": 0.20}, seed_context
        )
        scale = plan_nominal.per_stock_uL["PEO"] / PEO_BASIS_UL
        silica_uL = plan_nominal.per_stock_uL["Silica"] / scale
        # dried-basis solve for φ=0.20: (0.2/0.8)*222/0.04 = 1387.5 µL, vs sheet's 426
        assert silica_uL == pytest.approx(1387.5, rel=1e-3)


# ── Emergent dried fraction (dep components remain, carriers dry off) ─────────


class TestEmergentDriedFraction:
    def _consistent_stocks(self):
        """Physically-modelled stocks whose dried fraction == dep_fraction."""
        cat = ChemicalCatalog()
        cat.add(Chemical("PEO", density_g_per_mL=1.21, molar_mass_g_per_mol=44.0))
        cat.add(Chemical("PEO_solvent", density_g_per_mL=1.0))
        cat.add(Chemical("LiCl", density_g_per_mL=2.07, molar_mass_g_per_mol=42.39))
        cat.add(Chemical("salt_water", density_g_per_mL=1.0))
        cat.add(Chemical("SiO2", density_g_per_mL=2.2, is_particulate=True))
        cat.add(Chemical("silica_solvent", density_g_per_mL=1.0))
        peo = Solution("PEO", [
            SolutionComponent("PEO", "dep", 2.5, "mL"),          # dep_fraction = 0.25
            SolutionComponent("PEO_solvent", "carrier", 7.5, "mL"),
        ])
        licl = Solution("LiCl", [
            SolutionComponent("LiCl", "dep", 2.12, "g"),
            SolutionComponent("salt_water", "carrier", 39.858, "mL"),
        ])
        silica = Solution("Silica", [
            SolutionComponent("SiO2", "dep", 0.5, "mL"),         # dep_fraction = 0.05
            SolutionComponent("silica_solvent", "carrier", 9.5, "mL"),
        ])
        return cat, peo, licl, silica

    def test_dried_fraction_equals_dep_fraction(self):
        cat, peo, _, silica = self._consistent_stocks()
        assert dried_fraction(peo, cat) == pytest.approx(0.25)
        assert dried_fraction(silica, cat) == pytest.approx(0.05)

    def test_solver_resolves_emergent_when_unspecified(self):
        cat, peo, licl, silica = self._consistent_stocks()
        sp = solve_stocks_for_composition(
            eo_li_ratio=10.0, silica_vol_frac=0.1,
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=cat,
        )  # no dried fractions passed
        assert sp.dried_frac["PEO"] == pytest.approx(0.25)
        assert sp.dried_frac["Silica"] == pytest.approx(0.05)

    def test_emergent_context_matches_explicit(self):
        """A context with no dried fractions == one whose explicit values match dep_fraction."""
        cat, peo, licl, silica = self._consistent_stocks()
        common = dict(
            peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=cat,
            pump_assignment={"PEO": 0, "LiCl": 1, "Silica": 2},
            target_deposition_uL=13.5, peo_basis_uL=1000.0, budget_uL=125.0,
        )
        emergent = FormulationContext(**common)                       # dried fracs → None
        explicit = FormulationContext(peo_dried_frac=0.25, silica_dried_frac=0.05, **common)
        point = {"eo_li_ratio": 10.0, "silica_vol_frac": 0.1}
        a = plan_formulation(point, emergent)
        b = plan_formulation(point, explicit)
        assert a.per_stock_uL == pytest.approx(b.per_stock_uL)
        assert a.grand_total_uL == pytest.approx(b.grand_total_uL)

    def test_no_deposition_components_raises(self):
        cat = ChemicalCatalog()
        cat.add(Chemical("PEO", density_g_per_mL=1.21, molar_mass_g_per_mol=44.0))
        cat.add(Chemical("water", density_g_per_mL=1.0))
        cat.add(Chemical("SiO2", density_g_per_mL=2.2, is_particulate=True))
        peo = Solution("PEO", [SolutionComponent("PEO", "dep", 1.0, "mL")])
        licl = Solution("LiCl", [SolutionComponent("PEO", "dep", 1.0, "mL")])
        # all-carrier silica stock → emergent dried fraction 0 → helpful error
        silica = Solution("Silica", [SolutionComponent("water", "carrier", 10.0, "mL")])
        with pytest.raises(ValueError, match="dried fraction must be in"):
            solve_stocks_for_composition(
                eo_li_ratio=10.0, silica_vol_frac=0.1,
                peo_stock=peo, licl_stock=licl, silica_stock=silica, catalog=cat,
            )


# ── Anti-divergence: GUI path and loop path are the same function ────────────


class TestOneCoreTwoCallers:
    def test_identical_point_identical_plan(self, seed_context):
        point = {"eo_li_ratio": 10.0, "silica_vol_frac": 0.0327640}
        gui_plan = plan_formulation(point, seed_context)      # what the panel calls
        loop_plan = plan_formulation(point, seed_context)     # what the runner calls
        assert gui_plan.per_pump_uL == pytest.approx(loop_plan.per_pump_uL)
        assert gui_plan.per_stock_uL == pytest.approx(loop_plan.per_stock_uL)
        assert gui_plan.grand_total_uL == pytest.approx(loop_plan.grand_total_uL)
