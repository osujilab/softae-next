from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from softae.core.formulation import (
    Basis,
    Chemical,
    ChemicalCatalog,
    ElutionResult,
    Solution,
    SolutionCatalog,
    SolutionComponent,
    build_dispense_commands,
    composition_fractions,
    compute_elution_volumes,
    map_to_pump_volumes,
    molality,
    molarity,
    predicted_mixed_density,
    recommended_basis,
    simplex_fractions,
)
from softae.gui.widgets.formulation_panel import FormulationPanel


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def catalog():
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    cat.add(Chemical("NaCl", "NaCl", density_g_per_mL=2.16, molar_mass_g_per_mol=58.44))
    cat.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    return cat


# ── Chemical ──


class TestChemical:
    def test_chemical_fields(self):
        c = Chemical("Water", "H2O", 1.0, 18.015, "solvent")
        assert c.name == "Water"
        assert c.formula == "H2O"
        assert c.density_g_per_mL == 1.0
        assert c.molar_mass_g_per_mol == 18.015
        assert c.notes == "solvent"


# ── ChemicalCatalog ──


class TestChemicalCatalog:
    def test_add_and_get(self, catalog):
        c = catalog.get("Water")
        assert c.formula == "H2O"

    def test_remove(self):
        cat = ChemicalCatalog()
        cat.add(Chemical("X"))
        cat.remove("X")
        assert len(cat) == 0

    def test_list_names(self, catalog):
        names = catalog.list_names()
        assert names == ["Ethanol", "NaCl", "Water"]

    def test_csv_roundtrip(self, catalog, tmp_path):
        p = tmp_path / "chems.csv"
        catalog.save_csv(p)
        loaded = ChemicalCatalog.load_csv(p)
        assert len(loaded) == len(catalog)
        for name in catalog.list_names():
            orig = catalog.get(name)
            copy = loaded.get(name)
            assert orig.name == copy.name
            assert orig.formula == copy.formula
            assert orig.density_g_per_mL == pytest.approx(copy.density_g_per_mL)
            assert orig.molar_mass_g_per_mol == pytest.approx(copy.molar_mass_g_per_mol)


# ── Solution ──


class TestSolution:
    def test_total_volume_all_mL(self, catalog):
        sol = Solution("s1", [
            SolutionComponent("Water", "carrier", 8.0, "mL"),
            SolutionComponent("Ethanol", "dep", 2.0, "mL"),
        ])
        assert sol.total_volume_mL(catalog) == pytest.approx(10.0)

    def test_total_volume_with_grams(self, catalog):
        sol = Solution("s1", [
            SolutionComponent("Water", "carrier", 2.0, "g"),  # 2g / 1.0 = 2 mL
            SolutionComponent("NaCl", "dep", 2.16, "g"),       # 2.16g / 2.16 = 1 mL
        ])
        assert sol.total_volume_mL(catalog) == pytest.approx(3.0)

    def test_dep_fraction(self, catalog):
        sol = Solution("s1", [
            SolutionComponent("Water", "carrier", 8.0, "mL"),
            SolutionComponent("Ethanol", "dep", 2.0, "mL"),
        ])
        assert sol.dep_fraction(catalog) == pytest.approx(0.2)


# ── SolutionCatalog ──


class TestSolutionCatalog:
    def test_add_and_get(self):
        cat = SolutionCatalog()
        sol = Solution("sol1", [SolutionComponent("Water", "carrier", 10.0, "mL")])
        cat.add(sol)
        assert cat.get("sol1").name == "sol1"

    def test_csv_roundtrip(self, tmp_path):
        cat = SolutionCatalog()
        sol = Solution("sol1", [
            SolutionComponent("Water", "carrier", 10.0, "mL"),
            SolutionComponent("NaCl", "dep", 1.0, "g"),
        ])
        cat.add(sol)
        p = tmp_path / "sols.csv"
        cat.save_csv(p)
        loaded = SolutionCatalog.load_csv(p)
        assert len(loaded) == 1
        s = loaded.get("sol1")
        assert len(s.components) == 2
        names = {c.chemical_name for c in s.components}
        assert names == {"Water", "NaCl"}



# ── compute_elution_volumes ──


class TestComputeElutionVolumes:
    def test_single_solution_100pct(self, catalog):
        sol = Solution("s1", [
            SolutionComponent("Water", "carrier", 8.0, "mL"),
            SolutionComponent("Ethanol", "dep", 2.0, "mL"),
        ])
        result = compute_elution_volumes({"s1": sol}, catalog, 20.0)
        # dep_fraction = 0.2, so total = 20 / 0.2 = 100
        assert result.per_solution["s1"] == pytest.approx(100.0)
        assert result.grand_total_uL == pytest.approx(100.0)

    def test_two_solutions_equal(self, catalog):
        sol1 = Solution("s1", [
            SolutionComponent("Water", "carrier", 5.0, "mL"),
            SolutionComponent("Ethanol", "dep", 5.0, "mL"),
        ])
        sol2 = Solution("s2", [
            SolutionComponent("Water", "carrier", 8.0, "mL"),
            SolutionComponent("Ethanol", "dep", 2.0, "mL"),
        ])
        result = compute_elution_volumes({"s1": sol1, "s2": sol2}, catalog, 20.0)
        # each gets 10 µL deposition
        # s1: dep_frac=0.5, total=10/0.5=20
        # s2: dep_frac=0.2, total=10/0.2=50
        assert result.per_solution["s1"] == pytest.approx(20.0)
        assert result.per_solution["s2"] == pytest.approx(50.0)

    def test_custom_fractions(self, catalog):
        sol1 = Solution("s1", [
            SolutionComponent("Ethanol", "dep", 10.0, "mL"),
        ])
        sol2 = Solution("s2", [
            SolutionComponent("Ethanol", "dep", 5.0, "mL"),
            SolutionComponent("Water", "carrier", 5.0, "mL"),
        ])
        fractions = {"s1": 0.8, "s2": 0.2}
        result = compute_elution_volumes({"s1": sol1, "s2": sol2}, catalog, 100.0, fractions)
        # s1: dep_frac=1.0, target=80, total=80
        # s2: dep_frac=0.5, target=20, total=40
        assert result.per_solution["s1"] == pytest.approx(80.0)
        assert result.per_solution["s2"] == pytest.approx(40.0)

    def test_empty_solutions_raises(self, catalog):
        with pytest.raises(ValueError):
            compute_elution_volumes({}, catalog, 20.0)


# ── map_to_pump_volumes ──


class TestMapToPumpVolumes:
    def test_two_solutions_different_pumps(self):
        elution = ElutionResult(20.0, {"s1": 30.0, "s2": 50.0}, 80.0)
        result = map_to_pump_volumes(elution, {"s1": 0, "s2": 1})
        assert result == [pytest.approx(30.0), pytest.approx(50.0), pytest.approx(80.0)]

    def test_two_solutions_same_pump(self):
        elution = ElutionResult(20.0, {"s1": 30.0, "s2": 50.0}, 80.0)
        result = map_to_pump_volumes(elution, {"s1": 0, "s2": 0})
        assert result == [pytest.approx(80.0), pytest.approx(0.0), pytest.approx(80.0)]


# ── FormulationPanel (GUI) ──


class TestFormulationPanel:
    def test_panel_creates(self, qapp):
        panel = FormulationPanel()
        assert panel is not None
        panel.close()

    def test_add_chemical_row(self, qapp):
        # Inject an empty catalog so construction does not auto-load data_root.
        panel = FormulationPanel(chem_catalog=ChemicalCatalog())
        assert panel._chem_table.rowCount() == 0
        panel._btn_add_chem.click()
        assert panel._chem_table.rowCount() == 1
        panel.close()

    def test_volumes_calculated_signal_emitted(self, qapp):
        panel = FormulationPanel()
        received: list[list[float]] = []
        panel.volumes_calculated.connect(lambda v: received.append(v))
        panel._last_result = [10.0, 20.0, 30.0]
        panel._btn_apply.click()
        assert len(received) == 1
        assert received[0] == [10.0, 20.0, 30.0]
        panel.close()

    def test_close_button(self, qapp):
        panel = FormulationPanel()
        panel.show()
        panel._btn_close.click()
        assert not panel.isVisible()


# ── Composition-basis framing (WS1b) ──


@pytest.fixture
def basis_catalog():
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "H2O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015))
    cat.add(Chemical("Ethanol", "C2H5OH", density_g_per_mL=0.789, molar_mass_g_per_mol=46.07))
    cat.add(Chemical("Silica", "SiO2", density_g_per_mL=2.65, is_particulate=True))
    return cat


class TestCompositionBasis:
    def _ethanol_water(self):
        return Solution("s", [
            SolutionComponent("Water", "carrier", 8.0, "mL"),
            SolutionComponent("Ethanol", "dep", 2.0, "mL"),
        ])

    def test_volume_fraction(self, basis_catalog):
        fr = composition_fractions(self._ethanol_water(), basis_catalog, Basis.VOLUME)
        assert fr["Water"] == pytest.approx(0.8)
        assert fr["Ethanol"] == pytest.approx(0.2)

    def test_mass_fraction(self, basis_catalog):
        # water 8g, ethanol 2mL*0.789=1.578g, total 9.578g
        fr = composition_fractions(self._ethanol_water(), basis_catalog, Basis.MASS)
        assert fr["Water"] == pytest.approx(8.0 / 9.578, rel=1e-6)
        assert fr["Ethanol"] == pytest.approx(1.578 / 9.578, rel=1e-6)

    def test_mole_fraction(self, basis_catalog):
        # water 8/18.015 mol, ethanol 1.578/46.07 mol
        n_w = 8.0 / 18.015
        n_e = 1.578 / 46.07
        fr = composition_fractions(self._ethanol_water(), basis_catalog, Basis.MOLE)
        assert fr["Water"] == pytest.approx(n_w / (n_w + n_e), rel=1e-6)
        assert fr["Ethanol"] == pytest.approx(n_e / (n_w + n_e), rel=1e-6)

    def test_mole_fraction_particulate_raises(self, basis_catalog):
        sol = Solution("silica", [
            SolutionComponent("Silica", "dep", 1.0, "g"),
            SolutionComponent("Water", "carrier", 9.0, "mL"),
        ])
        with pytest.raises(ValueError):
            composition_fractions(sol, basis_catalog, Basis.MOLE)

    def test_molarity(self, basis_catalog):
        # ethanol dep: 1.578/46.07 mol in 10 mL = 0.010 L
        n_e = 1.578 / 46.07
        m = molarity(self._ethanol_water(), basis_catalog)
        assert m["Ethanol"] == pytest.approx(n_e / 0.010, rel=1e-6)
        assert "Water" not in m  # carrier, not a solute

    def test_molality(self, basis_catalog):
        # solvent = water carrier 8 g = 0.008 kg
        n_e = 1.578 / 46.07
        m = molality(self._ethanol_water(), basis_catalog)
        assert m["Ethanol"] == pytest.approx(n_e / 0.008, rel=1e-6)

    def test_recommended_basis(self, basis_catalog):
        assert recommended_basis(self._ethanol_water(), basis_catalog) is Basis.MOLE
        silica = Solution("sil", [SolutionComponent("Silica", "dep", 1.0, "g")])
        assert recommended_basis(silica, basis_catalog) is Basis.MASS

    def test_predicted_mixed_density(self, basis_catalog):
        # total mass 9.578 g over 10 mL
        d = predicted_mixed_density(self._ethanol_water(), basis_catalog)
        assert d == pytest.approx(9.578 / 10.0, rel=1e-6)

    def test_particulate_survives_csv_roundtrip(self, basis_catalog, tmp_path):
        p = tmp_path / "chems.csv"
        basis_catalog.save_csv(p)
        loaded = ChemicalCatalog.load_csv(p)
        assert loaded.get("Silica").is_particulate is True
        assert loaded.get("Water").is_particulate is False


# ── Simplex / N-stock (ternary-ready) ──


class TestSimplex:
    def test_simplex_from_list(self):
        fr = simplex_fractions(["A", "B", "C"], [0.5, 0.3])
        assert fr == {"A": pytest.approx(0.5), "B": pytest.approx(0.3), "C": pytest.approx(0.2)}

    def test_simplex_from_dict(self):
        fr = simplex_fractions(["A", "B", "C"], {"A": 0.6, "B": 0.1})
        assert fr["C"] == pytest.approx(0.3)

    def test_ternary_pump_mapping(self):
        elution = ElutionResult(30.0, {"s1": 10.0, "s2": 20.0, "s3": 30.0}, 60.0)
        result = map_to_pump_volumes(elution, {"s1": 0, "s2": 1, "s3": 2})
        assert result == [pytest.approx(10.0), pytest.approx(20.0),
                          pytest.approx(30.0), pytest.approx(60.0)]

    def test_binary_shape_preserved(self):
        elution = ElutionResult(20.0, {"s1": 30.0, "s2": 50.0}, 80.0)
        assert len(map_to_pump_volumes(elution, {"s1": 0, "s2": 1})) == 3


# ── Bug fixes (WS1f) ──


class TestBugFixes:
    def test_unknown_unit_raises(self, catalog):
        sol = Solution("s", [SolutionComponent("Water", "carrier", 1.0, "kg")])
        with pytest.raises(ValueError):
            sol.total_volume_mL(catalog)

    def test_carrier_only_explicit_dispenses_bulk(self, catalog):
        water = Solution("water", [SolutionComponent("Water", "carrier", 10.0, "mL")])
        active = Solution("active", [SolutionComponent("Ethanol", "dep", 10.0, "mL")])
        result = compute_elution_volumes(
            {"water": water, "active": active}, catalog, 100.0,
            {"water": 0.3, "active": 0.7},
        )
        assert result.per_solution["active"] == pytest.approx(70.0)  # dep-scaled
        assert result.per_solution["water"] == pytest.approx(30.0)  # bulk, not 0
        assert result.carrier_vol_uL["water"] == pytest.approx(30.0)
        assert result.dep_vol_uL["water"] == pytest.approx(0.0)


# ── Dispense commands: volume + rate per stock (WS1d) ──


class TestDispenseCommands:
    def test_scalar_rate(self):
        elution = ElutionResult(20.0, {"s1": 20.0, "s2": 50.0}, 70.0)
        cmds = build_dispense_commands(elution, {"s1": 0, "s2": 1}, 100.0)
        by_name = {c.solution: c for c in cmds}
        assert by_name["s1"].pump_id == 0
        assert by_name["s1"].volume_uL == pytest.approx(20.0)
        assert by_name["s1"].rate_uL_min == pytest.approx(100.0)
        assert by_name["s2"].rate_uL_min == pytest.approx(100.0)

    def test_per_stock_rate(self):
        elution = ElutionResult(20.0, {"s1": 20.0, "s2": 50.0}, 70.0)
        cmds = build_dispense_commands(elution, {"s1": 0, "s2": 1}, {"s1": 50.0, "s2": 80.0})
        by_name = {c.solution: c for c in cmds}
        assert by_name["s1"].rate_uL_min == pytest.approx(50.0)
        assert by_name["s2"].rate_uL_min == pytest.approx(80.0)
