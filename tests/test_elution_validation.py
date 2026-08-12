from __future__ import annotations

import pytest

from softae.core.elution_validation import (
    build_validation_rows,
    format_table,
    stock_viscosity,
    write_calibration_csv,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
)


@pytest.fixture
def catalog():
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", density_g_per_mL=1.0, viscosity_mPa_s=1.0))
    cat.add(Chemical("Glycerol", density_g_per_mL=1.261, viscosity_mPa_s=1412.0))
    return cat


@pytest.fixture
def solutions():
    water = Solution("water", [SolutionComponent("Water", "dep", 10.0, "mL")])
    glyc = Solution("glyc", [
        SolutionComponent("Glycerol", "dep", 5.0, "mL"),
        SolutionComponent("Water", "carrier", 5.0, "mL"),
    ])
    return {"water": water, "glyc": glyc}


class TestStockViscosity:
    def test_pure_water(self, catalog):
        sol = Solution("s", [SolutionComponent("Water", "dep", 10.0, "mL")])
        assert stock_viscosity(sol, catalog) == pytest.approx(1.0)

    def test_weighted_mean(self, catalog):
        sol = Solution("s", [
            SolutionComponent("Glycerol", "dep", 5.0, "mL"),
            SolutionComponent("Water", "carrier", 5.0, "mL"),
        ])
        # (5*1412 + 5*1.0) / 10
        assert stock_viscosity(sol, catalog) == pytest.approx((5 * 1412.0 + 5 * 1.0) / 10)

    def test_missing_viscosity_defaults_to_one(self):
        cat = ChemicalCatalog()
        cat.add(Chemical("X", density_g_per_mL=1.0))  # no viscosity
        sol = Solution("s", [SolutionComponent("X", "dep", 1.0, "mL")])
        assert stock_viscosity(sol, cat) == pytest.approx(1.0)


class TestValidationRows:
    def test_correction_adds_dead_volume(self, catalog, solutions):
        rows = build_validation_rows(
            solutions, catalog, 20.0,
            pump_assignment={"water": 0, "glyc": 1},
            rates=100.0,
            fractions={"water": 0.5, "glyc": 0.5},
        )
        by = {r.solution: r for r in rows}
        # corrected >= ideal (dead volume is additive and non-negative)
        for r in rows:
            assert r.corrected_uL >= r.uncorrected_uL
            assert r.dead_uL >= 0.0
        # viscous glycerol stock takes a larger dead volume than water
        assert by["glyc"].dead_uL > by["water"].dead_uL

    def test_rate_threaded(self, catalog, solutions):
        rows = build_validation_rows(
            solutions, catalog, 20.0,
            pump_assignment={"water": 0, "glyc": 1},
            rates={"water": 200.0, "glyc": 60.0},
            fractions={"water": 0.5, "glyc": 0.5},
        )
        by = {r.solution: r for r in rows}
        assert by["water"].rate_uL_min == pytest.approx(200.0)
        assert by["glyc"].rate_uL_min == pytest.approx(60.0)

    def test_format_and_csv(self, catalog, solutions, tmp_path):
        rows = build_validation_rows(
            solutions, catalog, 20.0,
            pump_assignment={"water": 0, "glyc": 1},
            rates=100.0,
            fractions={"water": 0.5, "glyc": 0.5},
        )
        assert "stock" in format_table(rows)
        p = write_calibration_csv(tmp_path / "cal.csv", rows, measured_uL={"water": 12.3})
        text = p.read_text(encoding="utf-8")
        assert "measured_uL" in text
        assert "12.3" in text
