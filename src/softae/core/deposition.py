"""Drop-cast deposition digital twin.

This module is the deposition half of a formulation->deposition digital twin.
The formulation core (``compute_elution_volumes`` / :class:`ElutionResult`)
answers *"what do I elute?"* and stops at the mixed liquid.  This module answers
*"what ends up in the well?"*: the eluted mixture is drop-cast into cylindrical
wells, the carrier (solvent) evaporates, and a solid film remains.

Physical model
--------------
- **Wells are cylinders.**  :class:`WellGeometry` turns user-input diameter and
  depth into a flat disc area (mm^2) and a capacity (uL); with the convention
  ``1 mm^3 == 1 uL`` capacity falls straight out of area * depth.
- **Carrier-only evaporation.**  A single ``evaporation_pct`` in ``[0, 100]`` is
  applied to the carrier (solvent) volume only.  Dep (solute) material is
  non-volatile and fully retained at every percentage.  At 100 % the film is the
  pure dep volume; at 0 % the "film" is the wet dispense.
- **Flat-disc films.**  The deposit is treated as a cylinder of the well's area,
  so thickness is simply ``volume / area``.  No coffee-ring, meniscus, or
  contact-line effects; volumes are additive and conserved except by evaporation.
- **Homogeneous mixture.**  Because the eluted batch is well mixed, dispensing a
  partial volume ``V`` scales every share (dep, carrier, per-component) linearly
  by ``s = V / grand_total_uL``.

Like ``compute_elution_volumes`` this is a pure-math twin: deterministic,
stdlib-only, no hardware, no Qt.  Volumes are book-kept so the user can always
see the solvent content of the cast mixture, the total volume eluted, and the
total deposited volume that survives evaporation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from softae.core.formulation import ElutionResult, Solution, is_dep_role

# Absolute float tolerance for "dispense must not exceed eluted" comparisons.
_VOL_TOL_uL = 1e-9


@dataclass(frozen=True)
class WellGeometry:
    """A cylindrical well.  1 uL == 1 mm^3, so capacity falls out in uL directly."""

    diameter_mm: float
    depth_mm: float

    def __post_init__(self) -> None:
        # frozen dataclass: validate only, never mutate
        if self.diameter_mm <= 0:
            raise ValueError(f"diameter_mm must be > 0, got {self.diameter_mm}")
        if self.depth_mm <= 0:
            raise ValueError(f"depth_mm must be > 0, got {self.depth_mm}")

    @property
    def area_mm2(self) -> float:
        return math.pi * (self.diameter_mm / 2.0) ** 2

    @property
    def capacity_uL(self) -> float:
        # 1 mm^3 == 1 uL exactly
        return self.area_mm2 * self.depth_mm

    @classmethod
    def from_board(cls, area_mm2: float, capacity_uL: float) -> "WellGeometry":
        """Equivalent geometry for a board that declares an area and a capacity.

        Boards declare ``deposit_area_mm2`` (or an electrode rectangle) and
        ``well_capacity_uL`` — not a cylinder. The twin, however, reads **only**
        :attr:`area_mm2` and :attr:`capacity_uL`, so back-solving an equivalent
        cylinder (diameter from the area, depth from capacity ÷ area) reproduces
        both exactly and every simulated quantity is unchanged.

        ⚠️ The resulting ``diameter_mm``/``depth_mm`` are an *equivalent*
        cylinder, not a measurement — a rectangular electrode has no diameter.
        They are correct to compute with and misleading to quote, so
        :meth:`WellDepositionResult.summary_lines` output should not be read as
        a physical description of a board well.
        """
        if area_mm2 <= 0:
            raise ValueError(f"area_mm2 must be > 0, got {area_mm2}")
        if capacity_uL <= 0:
            raise ValueError(f"capacity_uL must be > 0, got {capacity_uL}")
        diameter = 2.0 * math.sqrt(area_mm2 / math.pi)
        return cls(diameter_mm=diameter, depth_mm=capacity_uL / area_mm2)


def carrier_component_keys(solutions: dict[str, Solution]) -> set[tuple[str, str]]:
    """(solution_name, chemical_name) keys for every carrier-role component.

    Uses the same role convention as formulation.py: "dep"/"solute"/"active"
    count as dep; everything else (canonically "carrier") is solvent.  Pass the
    result as ``carrier_keys`` to the entry points to enable the optional
    per-component final-volume breakdown.
    """
    keys: set[tuple[str, str]] = set()
    for name, sol in solutions.items():
        for comp in sol.components:
            if not is_dep_role(comp.role):
                keys.add((name, comp.chemical_name))
    return keys


@dataclass
class WellDepositionResult:
    """Simulated deposition into a single cylindrical well."""

    well: WellGeometry
    evaporation_pct: float

    # wet (as-dispensed) state
    dispensed_uL: float  # wet volume placed in this well
    dep_uL: float  # non-volatile solute share of dispensed_uL
    carrier_uL: float  # solvent share of dispensed_uL

    # dry (post-evaporation) state
    evaporated_uL: float  # carrier lost = carrier_uL * evaporation_pct/100
    residual_carrier_uL: float  # carrier retained
    final_volume_uL: float  # dep_uL + residual_carrier_uL

    # film / fill metrics (flat-disc assumption)
    wet_thickness_um: float  # dispensed_uL / area_mm2 * 1000
    final_thickness_um: float  # final_volume_uL / area_mm2 * 1000
    wet_fill_fraction: float  # dispensed_uL / capacity_uL
    final_fill_fraction: float  # final_volume_uL / capacity_uL
    overflows: bool  # wet_fill_fraction > 1 (flag, not an error)

    # optional per-component final volumes, keyed like component_vol_uL
    component_final_uL: dict[tuple[str, str], float] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Well              : d={self.well.diameter_mm:.3f} mm  "
            f"depth={self.well.depth_mm:.3f} mm  "
            f"capacity={self.well.capacity_uL:.4f} uL",
            f"Evaporation       : {self.evaporation_pct:.2f} %",
            f"Dispensed (wet)   : {self.dispensed_uL:.4f} uL",
            f"  dep             : {self.dep_uL:.4f} uL",
            f"  carrier         : {self.carrier_uL:.4f} uL",
            f"Evaporated        : {self.evaporated_uL:.4f} uL",
            f"Residual carrier  : {self.residual_carrier_uL:.4f} uL",
            f"Final volume      : {self.final_volume_uL:.4f} uL",
            f"Wet thickness     : {self.wet_thickness_um:.2f} um",
            f"Final thickness   : {self.final_thickness_um:.2f} um",
            f"Wet fill fraction : {self.wet_fill_fraction:.4f}",
            f"Final fill fract. : {self.final_fill_fraction:.4f}",
        ]
        if self.overflows:
            lines.append("  !! wet volume exceeds well capacity")
        if self.component_final_uL:
            lines.append("")
            lines.append("  Per-component final volumes:")
            for (sol, comp), vol in self.component_final_uL.items():
                lines.append(f"    {sol} / {comp} : {vol:.4f} uL")
        return lines

    def as_dict(self) -> dict:
        return {
            "diameter_mm": self.well.diameter_mm,
            "depth_mm": self.well.depth_mm,
            "capacity_uL": self.well.capacity_uL,
            "evaporation_pct": self.evaporation_pct,
            "dispensed_uL": self.dispensed_uL,
            "dep_uL": self.dep_uL,
            "carrier_uL": self.carrier_uL,
            "evaporated_uL": self.evaporated_uL,
            "residual_carrier_uL": self.residual_carrier_uL,
            "final_volume_uL": self.final_volume_uL,
            "wet_thickness_um": self.wet_thickness_um,
            "final_thickness_um": self.final_thickness_um,
            "wet_fill_fraction": self.wet_fill_fraction,
            "final_fill_fraction": self.final_fill_fraction,
            "overflows": self.overflows,
            "component_final_uL": {
                f"{s} / {c}": v for (s, c), v in self.component_final_uL.items()
            },
        }


@dataclass
class DepositionSummary:
    """Mass-balance layer: one eluted batch cast across N wells."""

    total_eluted_uL: float  # = ElutionResult.grand_total_uL
    total_dispensed_uL: float  # sum of per-well dispenses
    undeposited_uL: float  # total_eluted_uL - total_dispensed_uL (>= 0)
    wells: list[WellDepositionResult]  # one entry per well, in dispense order

    @property
    def n_wells(self) -> int:
        return len(self.wells)

    @property
    def total_evaporated_uL(self) -> float:
        return sum(w.evaporated_uL for w in self.wells)

    @property
    def total_final_uL(self) -> float:
        return sum(w.final_volume_uL for w in self.wells)

    @property
    def any_overflow(self) -> bool:
        return any(w.overflows for w in self.wells)

    def summary_lines(self) -> list[str]:
        lines = [
            f"Total eluted      : {self.total_eluted_uL:.4f} uL",
            f"Total dispensed   : {self.total_dispensed_uL:.4f} uL",
            f"Undeposited       : {self.undeposited_uL:.4f} uL",
            f"Total evaporated  : {self.total_evaporated_uL:.4f} uL",
            f"Total final       : {self.total_final_uL:.4f} uL",
            f"Wells             : {self.n_wells}",
            f"Any overflow      : {self.any_overflow}",
        ]
        for i, well in enumerate(self.wells):
            lines.append("")
            lines.append(f"  [well {i}]")
            lines.extend(f"    {line}" for line in well.summary_lines())
        return lines

    def as_dict(self) -> dict:
        return {
            "total_eluted_uL": self.total_eluted_uL,
            "total_dispensed_uL": self.total_dispensed_uL,
            "undeposited_uL": self.undeposited_uL,
            "total_evaporated_uL": self.total_evaporated_uL,
            "total_final_uL": self.total_final_uL,
            "n_wells": self.n_wells,
            "any_overflow": self.any_overflow,
            "wells": [w.as_dict() for w in self.wells],
        }


#: Fraction of carrier assumed lost when nothing more specific is declared.
#: **100 % is the deliberate default (P7.4)**: it makes dry thickness
#: deterministic without a solvent model, which is what lets a dry-basis
#: :class:`~softae.core.formulation.ThicknessTarget` reduce exactly to a
#: deposited-volume target. Anything less needs per-solvent volatility to be
#: meaningful, and a half-modelled evaporation is worse than a stated assumption.
DEFAULT_EVAPORATION_PCT = 100.0


def evaporation_pct(config: dict | None = None) -> float:
    """Configured carrier loss (%), or :data:`DEFAULT_EVAPORATION_PCT`.

    Read from ``[deposition] evaporation_pct``. A single parse point so the GUI
    spin box, the campaign twin, and any headless path start from the same
    number rather than three independent defaults.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("deposition", {}) or {}
        except Exception:
            config = {}
    try:
        value = float(config.get("evaporation_pct", DEFAULT_EVAPORATION_PCT))
    except (TypeError, ValueError):
        return DEFAULT_EVAPORATION_PCT
    return value if 0.0 <= value <= 100.0 else DEFAULT_EVAPORATION_PCT


def _validate_evaporation_pct(evaporation_pct: float) -> None:
    if not (0.0 <= evaporation_pct <= 100.0):
        raise ValueError(
            f"evaporation_pct must be in [0, 100], got {evaporation_pct}"
        )


def simulate_well_deposition(
    elution: ElutionResult,
    well: WellGeometry,
    evaporation_pct: float,
    dispense_uL: float | None = None,
    *,
    carrier_keys: set[tuple[str, str]] | None = None,
) -> WellDepositionResult:
    """Simulate casting (part of) an eluted batch into one cylindrical well.

    ``dispense_uL=None`` dispenses the full eluted volume (``grand_total_uL``)
    into the well; otherwise dep/carrier/component volumes are scaled by
    ``dispense_uL / grand_total_uL`` (the mixture is homogeneous, so every share
    scales linearly).  ``carrier_keys`` (from :func:`carrier_component_keys`)
    enables the per-component final-volume breakdown; without it
    ``component_final_uL`` is ``{}``.
    """
    _validate_evaporation_pct(evaporation_pct)

    grand_total_uL = elution.grand_total_uL
    dispensed_uL = grand_total_uL if dispense_uL is None else dispense_uL
    if dispensed_uL < 0:
        raise ValueError(f"dispense_uL must be >= 0, got {dispensed_uL}")
    if dispensed_uL > grand_total_uL + _VOL_TOL_uL:
        raise ValueError(
            f"dispense_uL {dispensed_uL} exceeds eluted volume {grand_total_uL}"
        )

    frac = evaporation_pct / 100.0
    scale = dispensed_uL / grand_total_uL if grand_total_uL > 0 else 0.0

    dep_uL = elution.total_dep_uL * scale
    carrier_uL = elution.total_carrier_uL * scale
    evaporated_uL = carrier_uL * frac
    residual_carrier_uL = carrier_uL - evaporated_uL
    final_volume_uL = dep_uL + residual_carrier_uL

    area_mm2 = well.area_mm2
    capacity_uL = well.capacity_uL
    wet_fill_fraction = dispensed_uL / capacity_uL
    component_final_uL: dict[tuple[str, str], float] = {}
    if carrier_keys is not None:
        for (sol, comp), vol in elution.component_vol_uL.items():
            wet_comp_uL = vol * scale
            if (sol, comp) in carrier_keys:
                component_final_uL[(sol, comp)] = wet_comp_uL * (1.0 - frac)
            else:
                component_final_uL[(sol, comp)] = wet_comp_uL

    return WellDepositionResult(
        well=well,
        evaporation_pct=evaporation_pct,
        dispensed_uL=dispensed_uL,
        dep_uL=dep_uL,
        carrier_uL=carrier_uL,
        evaporated_uL=evaporated_uL,
        residual_carrier_uL=residual_carrier_uL,
        final_volume_uL=final_volume_uL,
        wet_thickness_um=dispensed_uL / area_mm2 * 1000.0,
        final_thickness_um=final_volume_uL / area_mm2 * 1000.0,
        wet_fill_fraction=wet_fill_fraction,
        final_fill_fraction=final_volume_uL / capacity_uL,
        overflows=wet_fill_fraction > 1.0,
        component_final_uL=component_final_uL,
    )


def _resolve_dispenses(
    dispense_uL: float | list[float] | None,
    n_wells: int,
    grand_total_uL: float,
) -> list[float]:
    """Resolve the per-well dispense volumes and validate the mass balance."""
    if isinstance(dispense_uL, list):
        if len(dispense_uL) != n_wells:
            raise ValueError(
                f"dispense list length {len(dispense_uL)} != n_wells {n_wells}"
            )
        volumes = [float(v) for v in dispense_uL]
    elif dispense_uL is None:
        volumes = [grand_total_uL / n_wells] * n_wells
    else:
        volumes = [float(dispense_uL)] * n_wells

    for v in volumes:
        if v < 0:
            raise ValueError(f"dispense volume must be >= 0, got {v}")
    if sum(volumes) > grand_total_uL + _VOL_TOL_uL:
        raise ValueError(
            f"total dispense {sum(volumes)} exceeds eluted volume {grand_total_uL}"
        )
    return volumes


def simulate_plate_deposition(
    elution: ElutionResult,
    well: WellGeometry,
    evaporation_pct: float,
    n_wells: int,
    dispense_uL: float | list[float] | None = None,
    *,
    carrier_keys: set[tuple[str, str]] | None = None,
) -> DepositionSummary:
    """Cast one eluted batch across ``n_wells`` identical wells.

    ``dispense_uL=None`` -> equal split: ``grand_total_uL / n_wells`` per well.
    ``dispense_uL=float`` -> that volume in every well (``n_wells * v`` must not
    exceed ``grand_total_uL``).  ``dispense_uL=list`` -> heterogeneous per-well
    volumes; ``len`` must equal ``n_wells`` and the sum must not exceed
    ``grand_total_uL``.
    """
    _validate_evaporation_pct(evaporation_pct)
    if n_wells < 1:
        raise ValueError(f"n_wells must be >= 1, got {n_wells}")

    grand_total_uL = elution.grand_total_uL
    volumes = _resolve_dispenses(dispense_uL, n_wells, grand_total_uL)

    wells = [
        simulate_well_deposition(
            elution, well, evaporation_pct, v, carrier_keys=carrier_keys
        )
        for v in volumes
    ]
    total_dispensed_uL = sum(volumes)
    return DepositionSummary(
        total_eluted_uL=grand_total_uL,
        total_dispensed_uL=total_dispensed_uL,
        undeposited_uL=max(0.0, grand_total_uL - total_dispensed_uL),
        wells=wells,
    )
