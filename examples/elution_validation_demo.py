"""Demo: elution-volume validation harness (WS1g).

Given a small chemical/solution catalog and a target deposition, prints the
per-stock ideal (uncorrected) elution volume alongside the liquid-handling
corrected command and rate.  Weigh/measure the actual eluted volume on the
bench, then pass the measurements to ``write_calibration_csv`` to build a
calibration record.

Usage
-----
    python examples/elution_validation_demo.py

Prints a table and writes ``elution_calibration.csv`` in the CWD.
"""

from __future__ import annotations

from pathlib import Path

from softae.core.elution_validation import (
    build_validation_rows,
    format_table,
    write_calibration_csv,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionComponent,
)


def _demo_catalog() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(Chemical("Water", "O", density_g_per_mL=1.0, molar_mass_g_per_mol=18.015,
                     viscosity_mPa_s=1.0))
    cat.add(Chemical("Glycerol", "OCC(O)CO", density_g_per_mL=1.261,
                     molar_mass_g_per_mol=92.094, viscosity_mPa_s=1412.0))
    return cat


def main() -> int:
    catalog = _demo_catalog()

    # A binary system: pure-water stock vs a 50:50 glycerol/water stock.
    water = Solution("Water stock", [SolutionComponent("Water", "dep", 10.0, "mL")])
    glyc = Solution("50-50 glycerol/water", [
        SolutionComponent("Glycerol", "dep", 5.0, "mL"),
        SolutionComponent("Water", "carrier", 5.0, "mL"),
    ])
    solutions = {"Water stock": water, "50-50 glycerol/water": glyc}
    pump_assignment = {"Water stock": 0, "50-50 glycerol/water": 1}
    rates = {"Water stock": 200.0, "50-50 glycerol/water": 60.0}  # µL/min

    rows = build_validation_rows(
        solutions,
        catalog,
        target_deposition_uL=20.0,
        pump_assignment=pump_assignment,
        rates=rates,
        fractions={"Water stock": 0.4, "50-50 glycerol/water": 0.6},
        run_index=1,
    )

    print("Elution validation - commanded volumes (corrected vs ideal):\n")
    print(format_table(rows))
    print(
        "\nNote the larger corr% on the viscous glycerol stock - the correction "
        "now scales with per-stock viscosity (restored to the catalog in WS1a)."
    )

    out = write_calibration_csv(Path("elution_calibration.csv"), rows, measured_uL=None)
    print(f"\nWrote calibration template to {out} - fill the measured_uL column "
          "after weighing, then re-run to fit the dead-volume constants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
