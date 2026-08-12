"""Elution-volume validation harness (WS1g).

Given a chemical/solution catalog and a target composition, this computes the
per-stock **commanded volume and rate** — both the ideal (uncorrected) elution
volume and the liquid-handling-corrected command — so the user can weigh or
measure the *actual* eluted volume on the bench and compare.  The measured
values feed a calibration record so the ``liquid_handling`` physics constants
stop being placeholders.

The per-stock viscosity now flows through the correction: this is what makes
corrected-vs-uncorrected meaningful for viscous stocks, and it is only possible
because ``viscosity_mPa_s`` was restored to the catalog in WS1a.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from softae.core.formulation import (
    ChemicalCatalog,
    Solution,
    compute_elution_volumes,
)
from softae.core.liquid_handling import (
    CorrectionInput,
    LinePhysicsConfig,
    LiquidHandlingCorrector,
    SystemPhysicsConfig,
)

# Defaults mirror the config loader's liquid-handling defaults so the harness is
# usable without a live config.  Override via ``correction_cfg``.
_DEFAULT_CFG = {
    "cracking_kpa_per_valve": 8.0,
    "compliance_uL_per_kpa": 0.55,
    "alpha_base": 0.20,
    "valves_in_series": 2,
    "beta": 0.30,
    "eta_ref_mpas": 1.0,
    "alpha_growth_per_run": 0.0,
}


@dataclass
class ValidationRow:
    solution: str
    pump_id: int
    viscosity_mpas: float
    rate_uL_min: float
    uncorrected_uL: float
    dead_uL: float
    corrected_uL: float

    @property
    def correction_pct(self) -> float:
        if self.uncorrected_uL <= 0:
            return 0.0
        return 100.0 * (self.corrected_uL - self.uncorrected_uL) / self.uncorrected_uL


def stock_viscosity(solution: Solution, catalog: ChemicalCatalog) -> float:
    """Volume-weighted mean viscosity (mPa·s) over components that report one.

    Falls back to 1.0 (water-like) when no component has a viscosity.
    """
    num = 0.0
    denom = 0.0
    for comp in solution.components:
        chem = catalog.get(comp.chemical_name)
        if chem.viscosity_mPa_s is None:
            continue
        # weight by nominal recipe quantity (mL or g both fine as a proxy weight)
        w = max(0.0, comp.quantity)
        num += w * chem.viscosity_mPa_s
        denom += w
    return num / denom if denom > 0 else 1.0


def build_validation_rows(
    solutions: dict[str, Solution],
    catalog: ChemicalCatalog,
    target_deposition_uL: float,
    *,
    pump_assignment: dict[str, int],
    rates: dict[str, float] | float,
    fractions: dict[str, float] | None = None,
    run_index: int = 1,
    correction_cfg: dict | None = None,
) -> list[ValidationRow]:
    """Per-stock uncorrected vs corrected commanded volume (+ rate)."""
    cfg = {**_DEFAULT_CFG, **(correction_cfg or {})}
    sys_cfg = SystemPhysicsConfig(
        valves_in_series=int(cfg["valves_in_series"]),
        beta=float(cfg["beta"]),
        eta_ref_mpas=float(cfg["eta_ref_mpas"]),
        alpha_growth_per_run=float(cfg["alpha_growth_per_run"]),
    )
    corrector = LiquidHandlingCorrector()
    elution = compute_elution_volumes(solutions, catalog, target_deposition_uL, fractions)

    rows: list[ValidationRow] = []
    for name in solutions:
        target = elution.per_solution.get(name, 0.0)
        visc = stock_viscosity(solutions[name], catalog)
        pump_id = pump_assignment.get(name, 0)
        line_cfg = LinePhysicsConfig(
            line_id=pump_id,
            cracking_kpa_per_valve=float(cfg["cracking_kpa_per_valve"]),
            compliance_uL_per_kpa=float(cfg["compliance_uL_per_kpa"]),
            alpha_base=float(cfg["alpha_base"]),
            viscosity_mpas=visc,
        )
        result = corrector.corrected_command(
            CorrectionInput(target_uL=target, line_id=pump_id, run_index=run_index),
            line_cfg,
            sys_cfg,
        )
        rate = rates.get(name, 0.0) if isinstance(rates, dict) else float(rates)
        rows.append(
            ValidationRow(
                solution=name,
                pump_id=pump_id,
                viscosity_mpas=visc,
                rate_uL_min=rate,
                uncorrected_uL=result.target_uL,
                dead_uL=result.dead_uL,
                corrected_uL=result.commanded_uL,
            )
        )
    return rows


def format_table(rows: list[ValidationRow]) -> str:
    header = (
        f"{'stock':<28} {'pump':>4} {'visc':>7} {'rate':>9} "
        f"{'ideal_uL':>9} {'dead_uL':>8} {'cmd_uL':>9} {'corr%':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.solution[:28]:<28} {r.pump_id:>4} {r.viscosity_mpas:>7.2f} "
            f"{r.rate_uL_min:>9.2f} {r.uncorrected_uL:>9.3f} {r.dead_uL:>8.3f} "
            f"{r.corrected_uL:>9.3f} {r.correction_pct:>6.1f}"
        )
    return "\n".join(lines)


def write_calibration_csv(
    path: Path,
    rows: list[ValidationRow],
    measured_uL: dict[str, float] | None = None,
) -> Path:
    """Append a calibration record: commanded vs (optional) measured actual µL.

    ``measured_uL`` maps solution name → the volume the user actually weighed /
    measured, so a later fit can back out the true dead-volume constants.
    """
    measured = measured_uL or {}
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "solution",
                    "pump_id",
                    "viscosity_mpas",
                    "rate_uL_min",
                    "ideal_uL",
                    "dead_uL",
                    "commanded_uL",
                    "measured_uL",
                ]
            )
        for r in rows:
            writer.writerow(
                [
                    r.solution,
                    r.pump_id,
                    f"{r.viscosity_mpas:.4f}",
                    f"{r.rate_uL_min:.4f}",
                    f"{r.uncorrected_uL:.4f}",
                    f"{r.dead_uL:.4f}",
                    f"{r.corrected_uL:.4f}",
                    "" if r.solution not in measured else f"{measured[r.solution]:.4f}",
                ]
            )
    return path
