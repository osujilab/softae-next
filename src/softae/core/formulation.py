"""Chemical formulation & elution-volume calculator.

This module is the composition→dispense core of softae-next.  It carries the
bench-validated math ported forward from the standalone ``experiment_manager``
prototype (``elution_calculator.py``), expressed in softae-next's idiom.

Terminology
-----------
- **dep** (deposition) component: the material that actually ends up deposited;
  its volume *counts* toward the target deposition volume.
- **carrier** component: the diluent/transport fluid; its volume is *not*
  counted in the deposition volume but is still eluted alongside the dep
  fraction to achieve the correct mixture ratio.

Given a set of stock solutions and a ``target_deposition_uL``, the calculator
computes the elution volume (µL) of each stock such that::

    sum_over_solutions( elution_vol_i * dep_fraction_i ) == target_deposition_uL

where ``dep_fraction_i`` is the volume fraction of stock *i* that is dep-role.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence, Union

import numpy as np

# Recognised per-component quantity units (case-insensitive).
_VOLUME_UNITS = {"ml", "mp", "microliter", "µl", "ul"}
_MASS_UNITS = {"g", "gram", "grams"}


@dataclass
class Chemical:
    name: str
    formula: str = ""
    density_g_per_mL: float = 1.0
    molar_mass_g_per_mol: float = 0.0
    notes: str = ""
    # Restored from the prototype catalog: consumed by the liquid-handling
    # dead-volume correction (viscosity ratio) and future basis conversions.
    viscosity_mPa_s: float | None = None
    # Particulate/suspension components (e.g. fumed silica) have no meaningful
    # molar mass; ``density_g_per_mL`` is then an *effective* suspension density
    # and mole-basis conversions must degrade gracefully rather than mislead.
    is_particulate: bool = False
    # Molar-species stoichiometry for ratio / concentration targets: species name
    # → count per mole of this chemical.  E.g. LiCl → {"Li": 1}, Li₂SO₄ → {"Li": 2},
    # PEO (the EO repeat unit) → {"EO": 1}.  Empty (the default) means "provides one
    # unit of a species named after the chemical itself" — so a 1:1 salt/polymer
    # needs no annotation and ``eo_li_ratio`` == a ``PEO``:``LiCl`` molar ratio.
    provides: dict[str, float] = field(default_factory=dict)

    @property
    def has_molar_mass(self) -> bool:
        return not self.is_particulate and bool(self.molar_mass_g_per_mol)

    def species_map(self) -> dict[str, float]:
        """Species → count per mole, defaulting to one unit of a self-named species."""
        return dict(self.provides) if self.provides else {self.name: 1.0}


@dataclass
class SolutionComponent:
    chemical_name: str
    # ``role`` is the *identity* axis: solute vs solvent.  "dep"/"solute"/"active"
    # are solute aliases; "carrier"/"solvent" are solvent aliases.  Identity drives
    # molarity / molality / composition framing and the evaporation model.
    role: str
    quantity: float
    unit: str  # "mL" or "g"
    # Per-component basis flag ported from the prototype ("Volume-based" |
    # "Mass-based").  Metadata today; consumed by WS1b basis conversions.
    calc_mode: str = "Volume-based"
    # Deposition-accounting axis, orthogonal to identity: does this component's
    # retained volume count toward the deposited-film-volume target?  ``None``
    # (the default, and what legacy CSVs load as) means "same as solute identity"
    # — so nothing changes unless a component overrides it.  An ionic solute whose
    # volume is negligible in film geometry (e.g. LiCl) sets this ``False`` while
    # keeping ``role="solute"`` so molarity/evaporation stay correct.
    counts_as_deposit: bool | None = None


@dataclass
class Solution:
    name: str
    components: list[SolutionComponent] = field(default_factory=list)

    def total_volume_mL(self, catalog: ChemicalCatalog) -> float:
        return sum(_component_vol_mL(c, catalog) for c in self.components)

    def dep_volume_mL(self, catalog: ChemicalCatalog) -> float:
        return sum(
            _component_vol_mL(c, catalog) for c in self.components if _is_dep(c)
        )

    def dep_fraction(self, catalog: ChemicalCatalog) -> float:
        total = self.total_volume_mL(catalog)
        if total <= 0:
            return 0.0
        return self.dep_volume_mL(catalog) / total


@dataclass
class ElutionResult:
    target_deposition_uL: float
    per_solution: dict[str, float]  # {name: total elution µL} — primary output
    grand_total_uL: float
    # --- rich breakdown (ported from the prototype; defaulted for back-compat) ---
    solution_fractions: dict[str, float] = field(default_factory=dict)
    dep_vol_uL: dict[str, float] = field(default_factory=dict)
    carrier_vol_uL: dict[str, float] = field(default_factory=dict)
    component_vol_uL: dict[tuple[str, str], float] = field(default_factory=dict)
    # Optional per-electrode elution budget (µL).  ``None`` → no cap declared, so
    # the run is always ``feasible`` with infinite ``headroom_uL``.
    budget_uL: float | None = None

    @property
    def feasible(self) -> bool:
        """True when the total eluted volume fits the declared budget (or none)."""
        if self.budget_uL is None:
            return True
        return self.grand_total_uL <= self.budget_uL + 1e-9

    @property
    def headroom_uL(self) -> float:
        """Budget minus total eluted (µL); ``inf`` when no budget is declared."""
        if self.budget_uL is None:
            return float("inf")
        return self.budget_uL - self.grand_total_uL

    @property
    def total_dep_uL(self) -> float:
        return sum(self.dep_vol_uL.values())

    @property
    def total_carrier_uL(self) -> float:
        return sum(self.carrier_vol_uL.values())

    def summary_lines(self) -> list[str]:
        lines = [
            f"Target deposition : {self.target_deposition_uL:.2f} uL",
            f"Actual dep total  : {self.total_dep_uL:.4f} uL",
            f"Total carrier     : {self.total_carrier_uL:.4f} uL",
            f"Grand total eluted: {self.grand_total_uL:.4f} uL",
            "",
        ]
        for sol, vol in self.per_solution.items():
            frac = self.solution_fractions.get(sol, float("nan"))
            dep = self.dep_vol_uL.get(sol, 0.0)
            car = self.carrier_vol_uL.get(sol, 0.0)
            lines.append(
                f"  [{sol}]  fraction={frac:.4f}  dep={dep:.4f} uL  "
                f"carrier={car:.4f} uL  -> elute {vol:.4f} uL total"
            )
        if self.component_vol_uL:
            lines.append("")
            lines.append("  Per-component breakdown:")
            for (sol, comp), vol in self.component_vol_uL.items():
                lines.append(f"    {sol} / {comp} : {vol:.4f} uL")
        return lines

    def as_dict(self) -> dict:
        return {
            "target_deposition_uL": self.target_deposition_uL,
            "per_solution": self.per_solution,
            "grand_total_uL": self.grand_total_uL,
            "budget_uL": self.budget_uL,
            "feasible": self.feasible,
            "headroom_uL": self.headroom_uL,
            "solution_fractions": self.solution_fractions,
            "dep_vol_uL": self.dep_vol_uL,
            "carrier_vol_uL": self.carrier_vol_uL,
            "component_vol_uL": {
                f"{s} / {c}": v for (s, c), v in self.component_vol_uL.items()
            },
        }


class ChemicalCatalog:
    def __init__(self) -> None:
        self._chemicals: dict[str, Chemical] = {}

    def add(self, chem: Chemical) -> None:
        self._chemicals[chem.name] = chem

    def remove(self, name: str) -> None:
        del self._chemicals[name]

    def get(self, name: str) -> Chemical:
        return self._chemicals[name]

    def list_names(self) -> list[str]:
        return sorted(self._chemicals.keys())

    def __len__(self) -> int:
        return len(self._chemicals)

    def save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "name",
                    "formula",
                    "density_g_per_mL",
                    "molar_mass_g_per_mol",
                    "viscosity_mPa_s",
                    "is_particulate",
                    "provides",
                    "notes",
                ]
            )
            for name in self.list_names():
                c = self._chemicals[name]
                writer.writerow(
                    [
                        c.name,
                        c.formula,
                        c.density_g_per_mL,
                        c.molar_mass_g_per_mol,
                        "" if c.viscosity_mPa_s is None else c.viscosity_mPa_s,
                        int(c.is_particulate),
                        _format_provides(c.provides),
                        c.notes,
                    ]
                )

    @classmethod
    def load_csv(cls, path: Path) -> ChemicalCatalog:
        cat = cls()
        if not path.exists():
            return cat
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat.add(
                    Chemical(
                        name=row["name"],
                        formula=row.get("formula", ""),
                        density_g_per_mL=float(row.get("density_g_per_mL") or 1.0),
                        molar_mass_g_per_mol=float(
                            row.get("molar_mass_g_per_mol") or 0.0
                        ),
                        notes=row.get("notes", ""),
                        viscosity_mPa_s=_opt_float(row.get("viscosity_mPa_s")),
                        is_particulate=_as_bool(row.get("is_particulate")),
                        provides=_parse_provides(row.get("provides")),
                    )
                )
        return cat


class SolutionCatalog:
    def __init__(self) -> None:
        self._solutions: dict[str, Solution] = {}

    def add(self, sol: Solution) -> None:
        self._solutions[sol.name] = sol

    def remove(self, name: str) -> None:
        del self._solutions[name]

    def get(self, name: str) -> Solution:
        return self._solutions[name]

    def list_names(self) -> list[str]:
        return sorted(self._solutions.keys())

    def __len__(self) -> int:
        return len(self._solutions)

    def save_csv(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "solution_name",
                    "component_name",
                    "role",
                    "quantity",
                    "unit",
                    "calc_mode",
                    "counts_as_deposit",
                ]
            )
            for name in self.list_names():
                sol = self._solutions[name]
                for comp in sol.components:
                    writer.writerow(
                        [
                            sol.name,
                            comp.chemical_name,
                            comp.role,
                            comp.quantity,
                            comp.unit,
                            comp.calc_mode,
                            "" if comp.counts_as_deposit is None else int(comp.counts_as_deposit),
                        ]
                    )

    @classmethod
    def load_csv(cls, path: Path) -> SolutionCatalog:
        cat = cls()
        if not path.exists():
            return cat
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sol_name = row["solution_name"]
                if sol_name not in cat._solutions:
                    cat._solutions[sol_name] = Solution(name=sol_name)
                cat._solutions[sol_name].components.append(
                    SolutionComponent(
                        chemical_name=row["component_name"],
                        role=row["role"],
                        quantity=float(row["quantity"]),
                        unit=row["unit"],
                        calc_mode=row.get("calc_mode", "Volume-based") or "Volume-based",
                        counts_as_deposit=_opt_bool(row.get("counts_as_deposit")),
                    )
                )
        return cat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opt_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(value)


def _opt_bool(value: object) -> bool | None:
    """Tristate parse: blank/None → ``None`` (inherit); otherwise a real bool.

    Distinct from :func:`_as_bool` (which maps blank → ``False``) because
    ``counts_as_deposit`` uses ``None`` to mean "default to solute identity".
    """
    if value is None or str(value).strip() == "":
        return None
    return _as_bool(value)


def _format_provides(provides: dict[str, float]) -> str:
    """Encode a species map as ``"Li:1|EO:1"`` for CSV round-trip (empty → "")."""
    return "|".join(f"{sp}:{mult}" for sp, mult in provides.items())


def _parse_provides(value: object) -> dict[str, float]:
    """Parse ``"Li:1|EO:1"`` → ``{"Li": 1.0, "EO": 1.0}``; blank/bad → ``{}``."""
    if value is None or str(value).strip() == "":
        return {}
    out: dict[str, float] = {}
    for token in str(value).split("|"):
        token = token.strip()
        if not token:
            continue
        sp, _, mult = token.partition(":")
        sp = sp.strip()
        if not sp:
            continue
        try:
            out[sp] = float(mult) if mult.strip() else 1.0
        except (TypeError, ValueError):
            out[sp] = 1.0
    return out


def _is_dep(comp: SolutionComponent) -> bool:
    """True if the component's volume counts toward the deposition target.

    Reads the deposition-accounting axis (``counts_as_deposit``) when set, else
    falls back to the solute *identity* (a solute deposits by default).  This is
    the sole place the two axes are distinguished: identity questions
    (molarity/molality/evaporation) keep calling :func:`_is_dep_role` on ``role``.
    """
    if comp.counts_as_deposit is not None:
        return comp.counts_as_deposit
    return _is_dep_role(comp.role)


def _component_vol_mL(comp: SolutionComponent, catalog: ChemicalCatalog) -> float:
    """Convert a component's quantity to mL.

    Raises
    ------
    ValueError
        If ``comp.unit`` is not a recognised volume or mass unit.  Previously an
        unknown unit silently contributed 0 volume (a data-integrity hazard).
    """
    unit = comp.unit.strip().lower()
    if unit in _VOLUME_UNITS:
        return comp.quantity
    if unit in _MASS_UNITS:
        chem = catalog.get(comp.chemical_name)
        density = chem.density_g_per_mL if chem.density_g_per_mL and chem.density_g_per_mL > 0 else 1.0
        return comp.quantity / density
    raise ValueError(
        f"Unrecognised unit {comp.unit!r} for component {comp.chemical_name!r} "
        f"(expected one of mL/g)"
    )


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------

def simplex_fractions(
    names: list[str], x_values: list[float] | dict[str, float]
) -> dict[str, float]:
    """Build a sum-to-one composition over ``names`` from N-1 free fractions.

    Ternary-ready ``feasible``/simplex primitive ported from the prototype's
    ``compute_elution_for_run_row``: the last name takes the complement so the
    fractions always sum to 1.0.

    ``x_values`` may be an ordered list of the first ``len(names)-1`` fractions,
    or a ``{name: fraction}`` mapping (any missing names default to 0 before the
    complement is assigned to the final name).
    """
    names = list(names)
    n = len(names)
    if n == 0:
        return {}

    fractions: dict[str, float] = {}
    running = 0.0
    for i in range(n - 1):
        if isinstance(x_values, dict):
            val = float(x_values.get(names[i], 0.0) or 0.0)
        else:
            val = float(x_values[i]) if i < len(x_values) else 0.0
        fractions[names[i]] = val
        running += val
    fractions[names[-1]] = max(0.0, round(1.0 - running, 12))
    return fractions


# ---------------------------------------------------------------------------
# Composition-basis framing (WS1b)
#
# Frames an existing stock recipe in volume / weight / mole percent, and
# reports molarity / molality.  Mass↔volume uses density; mass↔mole uses molar
# mass.  Particulate components have no molar mass, so mole-basis framing is
# undefined and raises an explicit error rather than silently mis-computing.
# ---------------------------------------------------------------------------

class Basis(str, Enum):
    VOLUME = "volume"
    MASS = "mass"
    MOLE = "mole"


@dataclass
class ComponentAmounts:
    name: str
    role: str
    volume_mL: float
    mass_g: float
    moles: float | None  # None when the chemical is particulate / lacks molar mass


def _component_mass_g(comp: SolutionComponent, catalog: ChemicalCatalog) -> float:
    unit = comp.unit.strip().lower()
    if unit in _MASS_UNITS:
        return comp.quantity
    if unit in _VOLUME_UNITS:
        chem = catalog.get(comp.chemical_name)
        density = chem.density_g_per_mL if chem.density_g_per_mL and chem.density_g_per_mL > 0 else 1.0
        return comp.quantity * density
    raise ValueError(
        f"Unrecognised unit {comp.unit!r} for component {comp.chemical_name!r} "
        f"(expected one of mL/g)"
    )


def component_amounts(
    solution: Solution, catalog: ChemicalCatalog
) -> dict[str, ComponentAmounts]:
    """Per-component volume (mL), mass (g), and moles for a stock recipe.

    ``moles`` is ``None`` for particulate / molar-mass-less components.
    """
    out: dict[str, ComponentAmounts] = {}
    for comp in solution.components:
        chem = catalog.get(comp.chemical_name)
        vol = _component_vol_mL(comp, catalog)
        mass = _component_mass_g(comp, catalog)
        moles = mass / chem.molar_mass_g_per_mol if chem.has_molar_mass else None
        out[comp.chemical_name] = ComponentAmounts(
            name=comp.chemical_name,
            role=comp.role,
            volume_mL=vol,
            mass_g=mass,
            moles=moles,
        )
    return out


def composition_fractions(
    solution: Solution, catalog: ChemicalCatalog, basis: Basis = Basis.VOLUME
) -> dict[str, float]:
    """Frame a stock's composition as ``{component: fraction}`` in *basis*.

    Fractions sum to 1.0.  ``Basis.MOLE`` raises ``ValueError`` if any component
    lacks a molar mass (e.g. a particulate), because a mole fraction cannot be
    defined for it.
    """
    amounts = component_amounts(solution, catalog)
    if basis is Basis.VOLUME:
        weights = {n: a.volume_mL for n, a in amounts.items()}
    elif basis is Basis.MASS:
        weights = {n: a.mass_g for n, a in amounts.items()}
    elif basis is Basis.MOLE:
        missing = [n for n, a in amounts.items() if a.moles is None]
        if missing:
            raise ValueError(
                "Mole-fraction basis is undefined for particulate / molar-mass-less "
                f"components: {missing}. Use volume or mass basis for this solution."
            )
        weights = {n: (a.moles or 0.0) for n, a in amounts.items()}
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"Unknown basis {basis!r}")

    total = sum(weights.values())
    if total <= 0:
        return {n: 0.0 for n in weights}
    return {n: w / total for n, w in weights.items()}


def recommended_basis(solution: Solution, catalog: ChemicalCatalog) -> Basis:
    """Auto-select the sensible framing basis for a stock.

    Particulate-containing stocks are weighed (mass); fully molar-defined stocks
    default to mole; otherwise volume.
    """
    chems = [catalog.get(c.chemical_name) for c in solution.components]
    if any(c.is_particulate for c in chems):
        return Basis.MASS
    if chems and all(c.has_molar_mass for c in chems):
        return Basis.MOLE
    return Basis.VOLUME


def molarity(
    solution: Solution, catalog: ChemicalCatalog
) -> dict[str, float]:
    """Molar concentration (mol/L) of each dep (solute) component in the stock.

    Uses the total stock volume as the solution volume.  Components without a
    molar mass are omitted.
    """
    amounts = component_amounts(solution, catalog)
    total_vol_L = sum(a.volume_mL for a in amounts.values()) / 1000.0
    if total_vol_L <= 0:
        return {}
    return {
        n: a.moles / total_vol_L
        for n, a in amounts.items()
        if a.moles is not None and _is_dep_role(a.role)
    }


def molality(
    solution: Solution, catalog: ChemicalCatalog
) -> dict[str, float]:
    """Molal concentration (mol/kg solvent) of each dep (solute) component.

    Solvent mass is the total mass of carrier-role components.
    """
    amounts = component_amounts(solution, catalog)
    solvent_kg = sum(
        a.mass_g for a in amounts.values() if not _is_dep_role(a.role)
    ) / 1000.0
    if solvent_kg <= 0:
        return {}
    return {
        n: a.moles / solvent_kg
        for n, a in amounts.items()
        if a.moles is not None and _is_dep_role(a.role)
    }


def _is_dep_role(role: str) -> bool:
    return role.strip().lower() in ("dep", "solute", "active")


# Public alias so deposition.py (and carrier-key builders) can classify roles
# without importing a private helper.
is_dep_role = _is_dep_role


def deposited_component_names(solution: Solution) -> list[str]:
    """Chemical names of a stock's components that remain in the dried film.

    The deposit-accounting view (``counts_as_deposit`` / dep-role), used by the
    GUI to offer valid ``DriedFractionTarget`` components.  Order-preserving,
    de-duplicated.
    """
    seen: dict[str, None] = {}
    for c in solution.components:
        if _is_dep(c):
            seen.setdefault(c.chemical_name, None)
    return list(seen)


def predicted_mixed_density(solution: Solution, catalog: ChemicalCatalog) -> float | None:
    """Volume-weighted density (g/mL) of a mixed stock; ``None`` if no volume.

    Feeds the liquid-handling dead-volume correction, which scales cracking
    pressure by a viscosity/density ratio.  Ported from the prototype's
    ``PROPERTY_REGISTRY`` predicted-density idea, as a plain function.
    """
    amounts = component_amounts(solution, catalog)
    total_vol = sum(a.volume_mL for a in amounts.values())
    if total_vol <= 0:
        return None
    total_mass = sum(a.mass_g for a in amounts.values())
    return total_mass / total_vol


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------

def compute_elution_volumes(
    solutions: dict[str, Solution],
    catalog: ChemicalCatalog,
    target_deposition_uL: float,
    fractions: dict[str, float] | None = None,
    budget_uL: float | None = None,
) -> ElutionResult:
    """Compute per-solution elution volumes to reach ``target_deposition_uL``.

    ``fractions`` maps ``{solution_name: fraction}`` — the share of the target
    deposition volume drawn from each stock's dep components.  When omitted, the
    remaining fraction is split equally across stocks that have dep content
    (remainder-aware resolution ported from the prototype).

    A **carrier-only** stock (``dep_fraction == 0``) cannot contribute to the
    deposition target via dep scaling; when it is *explicitly* assigned a
    fraction, that fraction is instead interpreted as a direct bulk-volume share
    of ``target_deposition_uL`` so the stock is actually dispensed rather than
    silently dropped to 0.

    ``budget_uL`` (optional) records a per-electrode elution cap on the result so
    ``ElutionResult.feasible`` / ``headroom_uL`` are computed; it does not alter
    the volumes.
    """
    if not solutions:
        raise ValueError("solutions must not be empty")

    names = list(solutions.keys())
    explicit = fractions or {}

    # --- remainder-aware fraction resolution -------------------------------
    dep_solutions = [nm for nm in names if solutions[nm].dep_fraction(catalog) > 0]
    unassigned = [nm for nm in dep_solutions if nm not in explicit]
    assigned_sum = sum(explicit.get(nm, 0.0) for nm in dep_solutions if nm in explicit)
    remainder = max(0.0, 1.0 - assigned_sum)

    resolved: dict[str, float] = {}
    for nm in names:
        if nm in explicit:
            resolved[nm] = explicit[nm]
        elif nm in unassigned:
            resolved[nm] = remainder / len(unassigned) if unassigned else 0.0
        else:
            resolved[nm] = 0.0

    per_solution: dict[str, float] = {}
    dep_vol: dict[str, float] = {}
    carrier_vol: dict[str, float] = {}
    comp_vol: dict[tuple[str, str], float] = {}

    for nm in names:
        sol = solutions[nm]
        frac = resolved.get(nm, 0.0)
        dep_fraction = sol.dep_fraction(catalog)
        target_dep_here_uL = frac * target_deposition_uL

        if dep_fraction > 0:
            total_elution_uL = target_dep_here_uL / dep_fraction
        elif nm in explicit and frac > 0:
            # Carrier-only stock explicitly requested: dispense the bulk volume.
            total_elution_uL = target_dep_here_uL
        else:
            total_elution_uL = 0.0

        per_solution[nm] = total_elution_uL
        dep_vol[nm] = total_elution_uL * dep_fraction
        carrier_vol[nm] = total_elution_uL * (1.0 - dep_fraction)

        # per-sub-component breakdown, proportional to each component's stock vol
        stock_total_mL = sol.total_volume_mL(catalog)
        for c in sol.components:
            c_frac = (
                _component_vol_mL(c, catalog) / stock_total_mL
                if stock_total_mL > 0
                else 0.0
            )
            comp_vol[(nm, c.chemical_name)] = c_frac * total_elution_uL

    return ElutionResult(
        target_deposition_uL=target_deposition_uL,
        per_solution=per_solution,
        grand_total_uL=sum(per_solution.values()),
        solution_fractions=resolved,
        dep_vol_uL=dep_vol,
        carrier_vol_uL=carrier_vol,
        component_vol_uL=comp_vol,
        budget_uL=budget_uL,
    )


def elution_from_stock_volumes(
    per_stock_uL: dict[str, float],
    solutions: dict[str, Solution],
    catalog: ChemicalCatalog,
    target_deposition_uL: float | None = None,
) -> ElutionResult:
    """Build an :class:`ElutionResult` from already-solved per-stock volumes.

    The inverse framing of :func:`compute_elution_volumes`: given the cast volume
    of each stock (e.g. from :func:`solve_formulation` or :func:`plan_formulation`),
    reconstruct the same rich breakdown — per-solution / dep / carrier / component
    volumes and dep-share fractions — using the *identical* split logic, so any
    downstream consumer (the deposition twin's ``simulate_plate_deposition``, CSV
    export) treats a target-based plan exactly like a manual-fraction one.

    ``target_deposition_uL`` defaults to the total deposited (dep) volume.
    """
    per_solution: dict[str, float] = {}
    dep_vol: dict[str, float] = {}
    carrier_vol: dict[str, float] = {}
    comp_vol: dict[tuple[str, str], float] = {}
    for nm, vol in per_stock_uL.items():
        sol = solutions[nm]
        vol = float(vol)
        dep_fraction = sol.dep_fraction(catalog)
        per_solution[nm] = vol
        dep_vol[nm] = vol * dep_fraction
        carrier_vol[nm] = vol * (1.0 - dep_fraction)
        stock_total_mL = sol.total_volume_mL(catalog)
        for c in sol.components:
            c_frac = (
                _component_vol_mL(c, catalog) / stock_total_mL if stock_total_mL > 0 else 0.0
            )
            comp_vol[(nm, c.chemical_name)] = c_frac * vol

    total_dep = sum(dep_vol.values())
    solution_fractions = {
        nm: (dep_vol[nm] / total_dep if total_dep > 0 else 0.0) for nm in per_stock_uL
    }
    return ElutionResult(
        target_deposition_uL=(total_dep if target_deposition_uL is None else target_deposition_uL),
        per_solution=per_solution,
        grand_total_uL=sum(per_solution.values()),
        solution_fractions=solution_fractions,
        dep_vol_uL=dep_vol,
        carrier_vol_uL=carrier_vol,
        component_vol_uL=comp_vol,
    )


def map_to_pump_volumes(
    elution: ElutionResult,
    pump_assignment: dict[str, int],
) -> list[float]:
    """Map per-solution elution volumes onto N physical pump channels.

    Returns ``[pump_0, pump_1, ..., pump_{N-1}, total]``.  ``N`` is at least 2
    (so the binary case keeps its historical ``[pump0, pump1, total]`` shape)
    and grows to cover the highest assigned pump index — ternary and beyond need
    no change here, only a wider ``pump_assignment``.
    """
    max_idx = max(pump_assignment.values(), default=-1)
    n_pumps = max(2, max_idx + 1)
    pumps = [0.0] * n_pumps
    for name, vol in elution.per_solution.items():
        pumps[pump_assignment.get(name, 0)] += vol
    return [*pumps, sum(pumps)]


@dataclass
class StockDispense:
    """A single stock's physical dispense command: which pump, how much, how fast."""
    solution: str
    pump_id: int
    volume_uL: float
    rate_uL_min: float


def build_dispense_commands(
    elution: ElutionResult,
    pump_assignment: dict[str, int],
    rates: dict[str, float] | float,
) -> list[StockDispense]:
    """Turn per-solution elution volumes into per-stock ``StockDispense`` commands.

    This is the formulator's hardware-facing output: **rate + volume per stock**
    (the two realizations of ``dep_fraction`` — a proportional flow rate or a
    sequential volume).  ``rates`` is either a single rate applied to every stock
    or a ``{solution_name: rate_uL_min}`` mapping.  N-stock by construction —
    binary today, ternary needs only a wider ``pump_assignment``/``rates``.
    """
    commands: list[StockDispense] = []
    for name, volume in elution.per_solution.items():
        rate = rates.get(name, 0.0) if isinstance(rates, dict) else float(rates)
        commands.append(
            StockDispense(
                solution=name,
                pump_id=pump_assignment.get(name, 0),
                volume_uL=volume,
                rate_uL_min=rate,
            )
        )
    return commands


# ---------------------------------------------------------------------------
# Inverse composition solver + planner facade (Tier-2 ternary formulator).
#
# The forward calculator above answers "given stocks and dep shares, how much to
# elute?".  A campaign speaks *composition* — an EO:Li ratio and a silica volume
# fraction — so the solver inverts targets → stock volumes, and the facade ties
# solve → elute → co-scale → pump-map into a single call that a bench user (GUI)
# and an optimizer (autonomous loop) invoke identically.  See
# docs/TERNARY_FORMULATOR_PLANNER.md.
#
# Design note: the empirical *dried* (non-volatile) fraction of a stock is a
# measured property passed explicitly — it is NOT derived from the stock recipe's
# ``dep_fraction``, because a stock's solute concentration (used for the ratio
# math) and its dried-volume shrinkage are independent facts (in the seed
# spreadsheet the PEO stock is 2 wt% for the salt ratio yet dries to ~22 % of its
# volume — irreconcilable in one recipe).  Identity (molarity) and
# deposition-accounting (dried volume) therefore never touch the same object.
# ---------------------------------------------------------------------------


def _sole_solute_molarity(
    solution: Solution, catalog: ChemicalCatalog, label: str
) -> float:
    """Molarity of a stock's single molar solute (for the ratio math).

    Raises if the stock has zero or several molar solutes — the solver needs an
    unambiguous concentration for ``label`` (PEO's EO units, or the salt).
    """
    m = molarity(solution, catalog)
    if len(m) != 1:
        raise ValueError(
            f"{label} stock must contain exactly one molar solute for the "
            f"composition solver; found {sorted(m)}"
        )
    return next(iter(m.values()))


def _sole_solute_species(
    solution: Solution, catalog: ChemicalCatalog, label: str
) -> str:
    """Species name of a stock's single molar solute (for a molar-ratio target).

    The ternary preset's ratio handle is 1:1 in a species — the sole solute's
    :meth:`Chemical.species_map` must name exactly one species (empty ``provides``
    → the chemical's own name, so ``eo_li_ratio`` is a plain ``PEO``:``LiCl``
    ratio).  A multi-species solute is ambiguous for the preset; use
    :func:`solve_formulation` directly with an explicit :class:`MolarRatioTarget`.
    """
    m = molarity(solution, catalog)
    if len(m) != 1:
        raise ValueError(
            f"{label} stock must contain exactly one molar solute for the "
            f"composition solver; found {sorted(m)}"
        )
    smap = catalog.get(next(iter(m))).species_map()
    if len(smap) != 1:
        raise ValueError(
            f"{label} solute {next(iter(m))!r} provides multiple species "
            f"{sorted(smap)}; use solve_formulation with an explicit ratio target"
        )
    return next(iter(smap))


@dataclass
class StockPlan:
    """Solved per-stock volumes (µL) at the PEO basis, plus realised composition."""

    volumes_uL: dict[str, float]        # keyed by the passed stock names
    achieved: dict[str, float]          # realised eo_li_ratio, silica_vol_frac, salt_molarity
    leveling_water_uL: float = 0.0      # water to hold total [salt] (0 unless requested)
    dried_frac: dict[str, float] = field(default_factory=dict)  # resolved dried fraction per stock


def dried_fraction(stock: Solution, catalog: ChemicalCatalog) -> float:
    """Emergent dried (non-volatile) volume fraction of a stock.

    Carriers evaporate entirely; the deposition (dep-role / ``counts_as_deposit``)
    components remain — so the dried fraction is exactly ``dep_fraction``.  This is
    the physical default; an explicit override exists only as a fallback for
    stocks whose modelled composition does not carry their measured shrinkage.
    """
    return stock.dep_fraction(catalog)


def solve_stocks_for_composition(
    *,
    eo_li_ratio: float,
    silica_vol_frac: float,             # dried-film volume-fraction basis
    peo_stock: Solution,
    licl_stock: Solution,
    silica_stock: Solution,
    catalog: ChemicalCatalog,
    peo_dried_frac: float | None = None,     # None → emergent from PEO stock dep_fraction
    silica_dried_frac: float | None = None,  # None → emergent from silica stock dep_fraction
    peo_basis_uL: float = 1000.0,
    salt_molarity_target: float | None = None,
) -> StockPlan:
    """Invert an EO:Li ratio + dried silica fraction into stock volumes (µL).

    Reproduces the seed spreadsheet's EO:cation block: PEO mmol at the basis,
    ``Li_mmol = PEO_mmol / eo_li_ratio``, ``LiCl_uL = Li_mmol / M_LiCl``; then
    inverts the *dried-film* silica fraction ``φ`` via
    ``V_dried(SiO₂) = φ/(1-φ)·V_dried(PEO)`` and ``Silica_uL = V_dried(SiO₂) /
    silica_dried_frac``.  ``salt_molarity_target`` (optional) adds water to hold a
    common absolute salt concentration across an EO:Li sweep (sheet rows 23–33);
    an infeasible (negative) leveling volume clamps to 0.

    The dried fractions **emerge** from each stock's dep/carrier composition
    (:func:`dried_fraction`) unless explicitly overridden — the override is a
    fallback crutch, not the norm.  The resolved fractions are echoed on the
    returned :class:`StockPlan`.
    """
    if eo_li_ratio <= 0:
        raise ValueError(f"eo_li_ratio must be > 0, got {eo_li_ratio}")
    if not (0.0 <= silica_vol_frac < 1.0):
        raise ValueError(f"silica_vol_frac must be in [0, 1), got {silica_vol_frac}")

    # Emergent-by-default: dried fraction == dep-component volume fraction.
    peo_dried_frac = (
        peo_dried_frac if peo_dried_frac is not None else dried_fraction(peo_stock, catalog)
    )
    silica_dried_frac = (
        silica_dried_frac if silica_dried_frac is not None
        else dried_fraction(silica_stock, catalog)
    )
    for lbl, stock, dfrac in (
        ("PEO", peo_stock, peo_dried_frac), ("silica", silica_stock, silica_dried_frac)
    ):
        if not (0.0 < dfrac <= 1.0):
            raise ValueError(
                f"{lbl} dried fraction must be in (0, 1]; got {dfrac} for stock "
                f"{stock.name!r}. Model the stock with deposition (non-volatile) "
                f"components so it emerges from dep_fraction, or pass an explicit "
                f"{lbl.lower()}_dried_frac override."
            )

    peo_M = _sole_solute_molarity(peo_stock, catalog, "PEO")       # mol EO / L
    licl_M = _sole_solute_molarity(licl_stock, catalog, "LiCl")    # mol / L

    peo_mmol = (peo_basis_uL / 1000.0) * peo_M                     # M == mmol/mL
    li_mmol = peo_mmol / eo_li_ratio
    licl_uL = (li_mmol / licl_M) * 1000.0

    peo_dried = peo_dried_frac * peo_basis_uL
    if silica_vol_frac <= 0.0:
        silica_dried = 0.0
        silica_uL = 0.0
    else:
        silica_dried = silica_vol_frac / (1.0 - silica_vol_frac) * peo_dried
        silica_uL = silica_dried / silica_dried_frac

    leveling_water_uL = 0.0
    if salt_molarity_target is not None and salt_molarity_target > 0:
        total_for_target_mL = li_mmol / salt_molarity_target
        leveling_water_uL = max(
            0.0, (total_for_target_mL - licl_uL / 1000.0 - peo_basis_uL / 1000.0) * 1000.0
        )

    soln_vol_mL = (peo_basis_uL + licl_uL + leveling_water_uL) / 1000.0
    achieved = {
        "eo_li_ratio": (peo_mmol / li_mmol) if li_mmol > 0 else float("inf"),
        "silica_vol_frac": (
            silica_dried / (silica_dried + peo_dried) if (silica_dried + peo_dried) > 0 else 0.0
        ),
        "salt_molarity": (li_mmol / soln_vol_mL) if soln_vol_mL > 0 else 0.0,
    }
    return StockPlan(
        volumes_uL={
            peo_stock.name: peo_basis_uL,
            licl_stock.name: licl_uL,
            silica_stock.name: silica_uL,
        },
        achieved=achieved,
        leveling_water_uL=leveling_water_uL,
        dried_frac={peo_stock.name: peo_dried_frac, silica_stock.name: silica_dried_frac},
    )


@dataclass
class FormulationContext:
    """Run-level invariants a campaign fixes once; only ``point`` varies per sample."""

    peo_stock: Solution
    licl_stock: Solution
    silica_stock: Solution
    catalog: ChemicalCatalog
    pump_assignment: dict[str, int]     # stock name → pump index
    target_deposition_uL: float         # dried-film volume target (sheet F82)
    # Dried fractions default to ``None`` → emergent from each stock's dep/carrier
    # composition (:func:`dried_fraction`).  Set them only to override that
    # physical default (a fallback for stocks whose model omits their shrinkage).
    peo_dried_frac: float | None = None
    silica_dried_frac: float | None = None
    peo_basis_uL: float = 1000.0
    budget_uL: float | None = None      # per-electrode elution cap (sheet C73)
    salt_molarity_target: float | None = None
    water_stock_name: str | None = None  # required only when leveling adds water

    @property
    def stocks(self) -> dict[str, Solution]:
        """``name -> Solution``, keyed exactly as ``FormulationPlan.per_stock_uL``.

        The three-stock context names its solutions individually while
        :class:`GeneralFormulation` carries a dict; consumers that take either
        (the deposition twin, elution) want the dict shape. Providing it here
        rather than at each call site keeps the two contexts interchangeable —
        the twin's fixed-context branch previously assumed this attribute existed
        and raised ``AttributeError`` the first time a composition campaign asked
        it to predict a thickness.
        """
        return {s.name: s for s in
                (self.peo_stock, self.licl_stock, self.silica_stock)}


class FormulationInfeasibleError(ValueError):
    """A composition's cast volume exceeds the per-electrode elution budget.

    Raised at the formulation→hardware boundary as a fail-safe: dispensing a plan
    whose ``grand_total_uL`` exceeds the well capacity would overflow the
    electrode.  Carries the offending :class:`FormulationPlan` for diagnostics and
    for a caller that wants to soft-penalise rather than abort.
    """

    def __init__(self, plan: "FormulationPlan", budget_uL: float | None = None):
        self.plan = plan
        self.budget_uL = budget_uL
        over = (plan.grand_total_uL - budget_uL) if budget_uL is not None else 0.0
        super().__init__(
            f"formulation cast {plan.grand_total_uL:.2f} µL exceeds the "
            f"per-electrode budget {budget_uL} µL (over by {over:.2f} µL)"
        )


@dataclass
class FormulationPlan:
    """The facade's output: per-pump cast volumes + realised composition + verdict."""

    per_pump_uL: list[float]            # → deposition_recipe formulation_by_channel[ch]
    per_stock_uL: dict[str, float]      # cast µL per stock name
    achieved: dict[str, float]          # realised composition (for logging / QC)
    grand_total_uL: float
    feasible: bool
    headroom_uL: float
    notes: list[str] = field(default_factory=list)


def plan_formulation(
    point: dict[str, float], context: FormulationContext
) -> FormulationPlan:
    """Composition design point → per-pump cast volumes, one call for both surfaces.

    ``point`` carries ``eo_li_ratio`` and ``silica_vol_frac`` — the same dict a user
    types and an optimizer proposes.  This is the ternary EO:Li/silica *preset*: it
    restates the two handles as declarative targets — a PEO:Li molar ratio and a
    dried-film silica volume fraction — plus the total-deposit scale, and delegates
    the actual solve to :func:`solve_formulation`, the single N-stock authority.
    Every stock's components count toward the dried film exactly as their own
    ``counts_as_deposit`` flag declares — no species is special-cased here; a salt
    whose bulk volume is non-additive is modelled with ``counts_as_deposit=False``
    on that stock (see the seed LiCl).  Optional salt-leveling water is layered on
    afterwards and feasibility is judged against the budget on the full cast.  The
    ``TestTernaryEquivalence`` regression pins this delegation.
    """
    ctx = context
    peo_name, licl_name, sil_name = (
        ctx.peo_stock.name, ctx.licl_stock.name, ctx.silica_stock.name
    )

    # Ratio species: each stock's sole molar solute (the 1:1 salt/polymer preset).
    eo_species = _sole_solute_species(ctx.peo_stock, ctx.catalog, "PEO")
    li_species = _sole_solute_species(ctx.licl_stock, ctx.catalog, "LiCl")
    licl_M = _sole_solute_molarity(ctx.licl_stock, ctx.catalog, "LiCl")

    # Silica handle: the silica stock's sole deposited component.
    sil_components = deposited_component_names(ctx.silica_stock)
    if len(sil_components) != 1:
        raise ValueError(
            f"silica stock {sil_name!r} must have exactly one deposited component "
            f"for the ternary preset; found {sil_components}"
        )
    sil_component = sil_components[0]

    targets: list[FormulationTarget] = [
        MolarRatioTarget(eo_species, li_species, float(point["eo_li_ratio"])),
        DriedFractionTarget(sil_component, float(point["silica_vol_frac"]), Basis.VOLUME),
        TotalDepositTarget(ctx.target_deposition_uL),
    ]
    overrides: dict[str, float] = {}
    if ctx.peo_dried_frac is not None:
        overrides[peo_name] = ctx.peo_dried_frac
    if ctx.silica_dried_frac is not None:
        overrides[sil_name] = ctx.silica_dried_frac

    # Stocks pass through as modelled — each component's own counts_as_deposit flag
    # decides its dried-film contribution (LiCl opts out on the stock, not in code).
    stocks = {
        peo_name: ctx.peo_stock,
        licl_name: ctx.licl_stock,
        sil_name: ctx.silica_stock,
    }
    # Solve the exactly-determined 3-target/3-stock core without a budget; the
    # optional leveling water is added below and feasibility judged on the whole cast.
    core = solve_formulation(
        stocks, ctx.catalog, targets,
        pump_assignment=ctx.pump_assignment,
        dried_frac=overrides or None,
    )

    per_stock: dict[str, float] = dict(core.per_stock_uL)  # PEO / LiCl / Silica cast µL
    peo_cast = per_stock[peo_name]
    scale = peo_cast / ctx.peo_basis_uL if ctx.peo_basis_uL > 0 else 0.0
    notes = ["silica axis: dried-film volume fraction (achieved == requested)"]

    # Salt-leveling water (optional): dilute the salt solution to a common absolute
    # [salt] across an EO:Li sweep.  Held over PEO + LiCl + water (silica excluded),
    # matching the seed sheet; an infeasible (negative) volume clamps to zero.
    li_mmol_cast = (per_stock[licl_name] / 1000.0) * licl_M
    if ctx.salt_molarity_target is not None and ctx.salt_molarity_target > 0:
        total_for_target_uL = (li_mmol_cast / ctx.salt_molarity_target) * 1000.0
        water_cast = max(0.0, total_for_target_uL - per_stock[licl_name] - peo_cast)
        if water_cast > 0:
            if not ctx.water_stock_name:
                raise ValueError(
                    "salt leveling produced water but context has no water_stock_name"
                )
            per_stock[ctx.water_stock_name] = water_cast
            water_basis = water_cast / scale if scale > 0 else 0.0
            notes.append(f"salt-leveling water: {water_basis:.1f} µL at basis")

    grand_total = sum(per_stock.values())

    # Ternary-flavoured realised composition (preserves the preset's achieved keys).
    salt_soln_mL = (
        peo_cast + per_stock[licl_name]
        + per_stock.get(ctx.water_stock_name or "", 0.0)
    ) / 1000.0
    achieved = {
        "eo_li_ratio": core.achieved.get(
            f"ratio[{eo_species}/{li_species}]", float("nan")
        ),
        "silica_vol_frac": core.achieved.get(f"dried_frac[{sil_component}]", 0.0),
        "salt_molarity": (li_mmol_cast / salt_soln_mL) if salt_soln_mL > 0 else 0.0,
    }

    if ctx.budget_uL is None:
        feasible, headroom = True, float("inf")
    else:
        feasible = grand_total <= ctx.budget_uL + 1e-9
        headroom = ctx.budget_uL - grand_total
        if not feasible:
            notes.append(
                f"cast {grand_total:.2f} µL exceeds budget {ctx.budget_uL:.2f} µL"
            )

    n_pumps = max(ctx.pump_assignment.values(), default=-1) + 1
    per_pump = [0.0] * n_pumps
    for name, vol in per_stock.items():
        if name not in ctx.pump_assignment:
            raise ValueError(f"no pump assigned for stock {name!r}")
        per_pump[ctx.pump_assignment[name]] += vol

    return FormulationPlan(
        per_pump_uL=per_pump,
        per_stock_uL=per_stock,
        achieved=achieved,
        grand_total_uL=grand_total,
        feasible=feasible,
        headroom_uL=headroom,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# General declarative formulation solver (N-stock, N-target, linear).
#
# The ternary EO:Li/silica solver above is one instantiation of a general model:
# a set of stocks and a set of *composition targets* — molar ratios between
# species, dried-film fractions of a component (volume/mass/mole basis), absolute
# species concentrations, and the total deposited volume.  Every such target is
# **linear in the per-stock volumes** (moles, dried volume, and mix volume are all
# linear; ratios/fractions/concentrations cross-multiply to homogeneous rows), so
# the whole system is ``A v = b`` — solved once for the per-stock cast volumes.
#
# Adding a component is adding a stock; adding a handle is adding a target.  N
# stocks need N independent targets (one of which is the total-deposit scale).
# See docs/TERNARY_FORMULATOR_PLANNER.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolarRatioTarget:
    """Hold ``moles(numerator) / moles(denominator) == value`` across the mix.

    ``numerator``/``denominator`` are *species* names (see
    :meth:`Chemical.species_map`).  EO:Li on a 1:1 salt is
    ``MolarRatioTarget("PEO", "LiCl", ratio)`` with no annotation; a 2:1 salt sets
    ``provides={"Li": 2}`` and targets ``"Li"``.
    """

    numerator: str
    denominator: str
    value: float


@dataclass(frozen=True)
class DriedFractionTarget:
    """Hold a component's share of the dried film at ``value`` on a chosen basis.

    ``component`` is a chemical name; ``basis`` picks volume / mass / mole framing
    of "share of the dried (non-volatile) film".  Mole basis is undefined for a
    particulate component (no molar mass) and raises.
    """

    component: str
    value: float
    basis: Basis = Basis.VOLUME


@dataclass(frozen=True)
class ConcentrationTarget:
    """Hold the absolute molar concentration of a species in the final mix (mol/L)."""

    species: str
    value_M: float


@dataclass(frozen=True)
class TotalDepositTarget:
    """Hold the total dried (deposited) film volume at ``value_uL`` — the scale."""

    value_uL: float


@dataclass(frozen=True)
class ThicknessTarget:
    """Hold the film thickness at ``value_um`` over a declared deposit area (P7.3).

    The inverse solve: *thickness → required elution volume*. Like
    :class:`TotalDepositTarget` it fixes the **scale** of the formulation; the
    other targets fix relative volumes.

    ``basis`` selects what the thickness refers to, and the two answer different
    questions:

    * ``"dry"`` — thickness of the **dried film**, the physically meaningful one
      (it is the ``t`` in σ = L/(R·w·t)). With the default assumption of full
      solvent loss, dry thickness × area *is* the dried deposit volume, so this
      basis reduces exactly to :class:`TotalDepositTarget` once an authoritative
      area is available. That is deliberate — it needs no new solver machinery
      and no solvent model, which is what makes dry thickness deterministic.
    * ``"wet"`` — thickness of the **as-dispensed** liquid. Directly
      controllable and independent of any evaporation assumption, but it is not
      the quantity conductivity depends on.

    ⚠️ **A computed thickness is a nominal, dense-film geometric estimate.**
    Volumes are treated as additive and no dry-film density or porosity is
    modelled, so a porous real film is thicker than this number. Sound as a
    control target and as a consistent relative measure across a campaign; not a
    metrology claim.
    """

    value_um: float
    area_mm2: float
    basis: str = "dry"

    def __post_init__(self) -> None:
        if self.basis not in ("dry", "wet"):
            raise ValueError(
                f"ThicknessTarget basis must be 'dry' or 'wet', got {self.basis!r}"
            )
        if self.area_mm2 <= 0:
            raise ValueError(
                "ThicknessTarget needs a positive deposit area — see "
                "geometry.deposit_area_mm2; a guessed area silently corrupts "
                "every thickness derived from it"
            )

    def volume_uL(self) -> float:
        """The volume this thickness corresponds to over :attr:`area_mm2`."""
        from softae.core.geometry import volume_for_thickness_uL

        return volume_for_thickness_uL(self.value_um, self.area_mm2)


FormulationTarget = Union[
    MolarRatioTarget, DriedFractionTarget, ConcentrationTarget, TotalDepositTarget,
    ThicknessTarget,
]


def species_concentration(
    solution: Solution, catalog: ChemicalCatalog
) -> dict[str, float]:
    """Molar concentration (mol/L) of each *species* in a stock.

    Sums each dep solute's molarity times its chemical's :meth:`species_map`, so a
    salt that provides 2 cations per mole contributes twice its molarity to that
    species.  Feeds molar-ratio and concentration targets.
    """
    out: dict[str, float] = {}
    for solute_name, molar in molarity(solution, catalog).items():
        for sp, mult in catalog.get(solute_name).species_map().items():
            out[sp] = out.get(sp, 0.0) + molar * mult
    return out


def _dried_profile(
    solution: Solution, catalog: ChemicalCatalog, override: float | None
) -> tuple[dict[str, tuple[float, float, float | None]], float]:
    """Per-µL dried amounts of each dep component and the stock's dried vol fraction.

    Returns ``({chemical: (dried_vol_uL, dried_mass_g, dried_mol|None)}, dep_frac)``,
    all per 1 µL of dispensed stock.  Dep components (``counts_as_deposit`` /
    dep-role) remain; carriers evaporate.  ``override`` (when given) rescales the
    dep components so their volume fraction sums to it — the shrinkage crutch for
    stocks whose modelled composition doesn't carry their measured dried fraction.
    """
    total_vol = solution.total_volume_mL(catalog)
    comps: list[tuple[str, float]] = []  # (chemical, dried volume fraction)
    dep_sum = 0.0
    for c in solution.components:
        if not _is_dep(c):
            continue
        vf = (_component_vol_mL(c, catalog) / total_vol) if total_vol > 0 else 0.0
        comps.append((c.chemical_name, vf))
        dep_sum += vf
    if override is not None:
        scale = (override / dep_sum) if dep_sum > 0 else 0.0
        comps = [(nm, vf * scale) for nm, vf in comps]
        dep_sum = override if dep_sum > 0 else 0.0

    prof: dict[str, tuple[float, float, float | None]] = {}
    for name, vf in comps:
        chem = catalog.get(name)
        dvol = vf                                   # µL dried per µL stock
        dmass = vf * 1e-3 * chem.density_g_per_mL   # g per µL stock
        dmol = None if not chem.has_molar_mass else dmass / chem.molar_mass_g_per_mol
        prev = prof.get(name)
        if prev is None:
            prof[name] = (dvol, dmass, dmol)
        else:                                       # same chemical twice → accumulate
            pv, pm, pmol = prev
            prof[name] = (pv + dvol, pm + dmass,
                          None if (pmol is None or dmol is None) else pmol + dmol)
    return prof, dep_sum


def solve_formulation(
    stocks: dict[str, Solution],
    catalog: ChemicalCatalog,
    targets: Sequence[FormulationTarget],
    *,
    pump_assignment: dict[str, int] | None = None,
    budget_uL: float | None = None,
    dried_frac: dict[str, float] | None = None,
    tol: float = 1e-6,
) -> FormulationPlan:
    """Solve per-stock cast volumes for arbitrary stocks + composition targets.

    Assembles the linear system the targets imply (one row each) and solves for the
    per-stock volumes (µL); ``TotalDepositTarget`` fixes the scale, the rest fix the
    relative volumes.  Returns a :class:`FormulationPlan` with ``achieved`` holding
    each target's realised value and ``notes`` flagging under/over-determination,
    unmet targets, negative volumes, or a budget breach.  Non-negativity is checked
    (not enforced by projection — an infeasible target set is reported, not silently
    repaired).
    """
    if not stocks:
        raise ValueError("solve_formulation requires at least one stock")
    names = list(stocks)
    n = len(names)
    dried_frac = dried_frac or {}

    conc = [species_concentration(stocks[nm], catalog) for nm in names]
    profiles: list[dict[str, tuple[float, float, float | None]]] = []
    depf: list[float] = []
    for nm in names:
        prof, df = _dried_profile(stocks[nm], catalog, dried_frac.get(nm))
        profiles.append(prof)
        depf.append(df)

    def _dvol(i: int, comp: str) -> float:
        return profiles[i].get(comp, (0.0, 0.0, None))[0]

    def _dmass(i: int, comp: str) -> float:
        return profiles[i].get(comp, (0.0, 0.0, None))[1]

    def _dmol(i: int, comp: str) -> float | None:
        return profiles[i].get(comp, (0.0, 0.0, None))[2]

    rows: list[list[float]] = []
    rhs: list[float] = []
    for t in targets:
        if isinstance(t, TotalDepositTarget):
            rows.append([depf[i] for i in range(n)])
            rhs.append(float(t.value_uL))
        elif isinstance(t, ThicknessTarget):
            # Same shape as TotalDepositTarget — both fix the scale — differing
            # only in which volume the thickness refers to.
            if t.basis == "dry":
                # Dried volume, weighted by each stock's deposited fraction.
                # With full solvent loss this IS TotalDepositTarget, which is
                # why the dry basis needs no new solver machinery.
                rows.append([depf[i] for i in range(n)])
            else:  # "wet"
                # As-dispensed volume: every stock contributes its full volume,
                # so the coefficients are 1 regardless of what dries out.
                rows.append([1.0 for _ in range(n)])
            rhs.append(t.volume_uL())
        elif isinstance(t, MolarRatioTarget):
            rows.append([
                conc[i].get(t.numerator, 0.0) - t.value * conc[i].get(t.denominator, 0.0)
                for i in range(n)
            ])
            rhs.append(0.0)
        elif isinstance(t, ConcentrationTarget):
            rows.append([conc[i].get(t.species, 0.0) - t.value_M for i in range(n)])
            rhs.append(0.0)
        elif isinstance(t, DriedFractionTarget):
            if t.basis is Basis.VOLUME:
                num = [_dvol(i, t.component) for i in range(n)]
                den = [depf[i] for i in range(n)]
            elif t.basis is Basis.MASS:
                num = [_dmass(i, t.component) for i in range(n)]
                den = [sum(v[1] for v in profiles[i].values()) for i in range(n)]
            elif t.basis is Basis.MOLE:
                if any(_dmol(i, t.component) is None and _dvol(i, t.component) > 0
                       for i in range(n)):
                    raise ValueError(
                        f"mole-basis dried fraction undefined for particulate "
                        f"component {t.component!r}"
                    )
                num = [(_dmol(i, t.component) or 0.0) for i in range(n)]
                den = [sum((v[2] or 0.0) for v in profiles[i].values()) for i in range(n)]
            else:  # pragma: no cover
                raise ValueError(f"unknown basis {t.basis!r}")
            rows.append([num[i] - t.value * den[i] for i in range(n)])
            rhs.append(0.0)
        else:
            raise TypeError(f"unknown formulation target {type(t).__name__}")

    a = np.array(rows, dtype=float)
    b = np.array(rhs, dtype=float)
    v, _res, rank, _sv = np.linalg.lstsq(a, b, rcond=None)
    v = np.asarray(v, dtype=float)

    notes: list[str] = []
    if rank < n:
        notes.append(
            f"under-determined: {len(targets)} target(s), {n} stock(s), rank {rank} "
            f"— minimum-norm solution (add {n - rank} more target(s) to pin it)"
        )
    scale_b = 1.0 + (float(np.max(np.abs(b))) if len(b) else 0.0)
    target_residual = float(np.max(np.abs(a @ v - b))) if len(rows) else 0.0
    if target_residual > 1e-6 * scale_b:
        notes.append(f"targets not exactly met (max residual {target_residual:.3g})")

    per_stock = {nm: float(v[i]) for i, nm in enumerate(names)}
    neg = [nm for nm, val in per_stock.items() if val < -tol]
    if neg:
        notes.append(f"negative (physically infeasible) volume for stock(s): {neg}")
    per_stock = {nm: max(0.0, val) for nm, val in per_stock.items()}
    grand_total = sum(per_stock.values())

    if budget_uL is None:
        feasible, headroom = (not neg), float("inf")
    else:
        feasible = (not neg) and grand_total <= budget_uL + 1e-9
        headroom = budget_uL - grand_total
        if grand_total > budget_uL + 1e-9:
            notes.append(f"cast {grand_total:.2f} µL exceeds budget {budget_uL:.2f} µL")

    # Realised target values (for QC / logging).
    def _mix_species(sp: str) -> float:
        return sum(v[i] * conc[i].get(sp, 0.0) for i in range(n))

    total_dried = sum(v[i] * depf[i] for i in range(n))
    achieved: dict[str, float] = {"total_deposit_uL": float(total_dried)}
    for t in targets:
        if isinstance(t, MolarRatioTarget):
            den_moles = _mix_species(t.denominator)
            achieved[f"ratio[{t.numerator}/{t.denominator}]"] = (
                float(_mix_species(t.numerator) / den_moles)
                if abs(den_moles) > 1e-30 else float("inf")
            )
        elif isinstance(t, ConcentrationTarget):
            achieved[f"conc[{t.species}]_M"] = (
                float(_mix_species(t.species) / grand_total) if grand_total > 0 else 0.0
            )
        elif isinstance(t, DriedFractionTarget):
            num_v = sum(v[i] * _dvol(i, t.component) for i in range(n))
            achieved[f"dried_frac[{t.component}]"] = (
                float(num_v / total_dried) if total_dried > 0 else 0.0
            )

    assign = pump_assignment or {nm: i for i, nm in enumerate(names)}
    n_pumps = max(assign.values(), default=-1) + 1
    per_pump = [0.0] * n_pumps
    for nm, val in per_stock.items():
        if nm not in assign:
            raise ValueError(f"no pump assigned for stock {nm!r}")
        per_pump[assign[nm]] += val

    return FormulationPlan(
        per_pump_uL=per_pump,
        per_stock_uL=per_stock,
        achieved=achieved,
        grand_total_uL=grand_total,
        feasible=feasible,
        headroom_uL=headroom,
        notes=notes,
    )
