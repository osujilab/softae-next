"""Two-phase dropcast planning — the proportional-rate split.

Pure, UI-free helpers shared by the HT Experiment tab and the Autonomous tab.
Given a formulation's **per-pump deposition volumes** plus a few campaign scalars,
this derives the per-pump *rates* for a two-phase cast:

* a **precondition flush** — preload the lines with the next composition;
* a **deposition** cast — extrude the drop at the electrode.

The key idea (see ``docs/AE_DROPCAST_RECIPE_SPEC.md``): per-pump rates are split
*proportionally to per-pump volumes*, so every pump extrudes for the **same
duration** and the components mix in the correct proportions as they move through
the lines.  A single total flow rate is specified per phase; this module turns it
into the per-pump vector.

Scope: rate/volume/dwell derivation only.  Positions, plug settings, dead volume,
and electrode geometry stay with the catalog methods / the calling tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PhaseParams",
    "DropcastPlan",
    "split_rate",
    "build_dropcast_plan",
]


@dataclass(frozen=True)
class PhaseParams:
    """Per-pump volumes and rates for one phase of a cast.

    ``volumes_uL[i]`` and ``rates_uL_min[i]`` describe pump ``i``; the two lists
    are always the same length.  ``rates_uL_min`` sums to the phase's total flow
    rate (within rounding), and — because it is split proportionally to
    ``volumes_uL`` — every pump shares the same extrusion duration.
    """

    volumes_uL: list[float]
    rates_uL_min: list[float]

    def duration_min(self) -> float:
        """Extrusion duration (min), identical across pumps; 0 if nothing moves."""
        total_vol = sum(self.volumes_uL)
        total_rate = sum(self.rates_uL_min)
        if total_rate <= 0.0:
            return 0.0
        return total_vol / total_rate


@dataclass(frozen=True)
class DropcastPlan:
    """A single channel's two-phase cast, derived from per-pump volumes + scalars.

    * :attr:`deposition` — volumes (the formulation) and per-pump deposition rates.
    * :attr:`flush_rates_uL_min` — per-pump rates for the precondition flush; the
      flush *volumes* are ``volumes_uL[i] * flush_factor`` and are applied inside
      the ``precondition_flush`` driver method (which takes ``flush_factor``), so
      the plan carries the factor rather than the pre-multiplied volumes.
    * :attr:`settle_wait_s` — the deposition elution/settling dwell, derived from
      the deposition duration and a settle factor.
    """

    deposition: PhaseParams
    flush_rates_uL_min: list[float]
    flush_factor: float
    settle_wait_s: float

    @property
    def volumes_uL(self) -> list[float]:
        """The per-pump deposition volumes (the formulation)."""
        return self.deposition.volumes_uL

    def preload_volumes_uL(self) -> list[float]:
        """Per-pump precondition preload volumes (``volume * flush_factor``)."""
        return [v * self.flush_factor for v in self.deposition.volumes_uL]


def split_rate(total_rate: float, volumes: Sequence[float]) -> list[float]:
    """Split ``total_rate`` across pumps in proportion to ``volumes``.

    ``rate[i] = total_rate * volumes[i] / sum(volumes)`` — so pump ``i`` finishes
    its ``volumes[i]`` in the same time as every other pump (equal duration →
    proportional mixing).  If the volumes sum to zero (nothing to dispense), every
    rate is ``0.0`` — no pump moves and there is no divide-by-zero.
    """
    vols = [max(0.0, float(v)) for v in volumes]
    total_vol = sum(vols)
    if total_vol <= 0.0:
        return [0.0] * len(vols)
    r = max(0.0, float(total_rate))
    return [r * v / total_vol for v in vols]


def build_dropcast_plan(
    volumes_uL: Sequence[float],
    *,
    dispense_rate_total: float,
    flush_rate_total: float,
    flush_factor: float,
    settle_factor: float,
    settle_base_s: float = 0.0,
) -> DropcastPlan:
    """Derive a :class:`DropcastPlan` from per-pump volumes and campaign scalars.

    Parameters
    ----------
    volumes_uL :
        Per-pump deposition volumes (the formulation; already dead-volume
        corrected by the caller if correction is enabled).
    dispense_rate_total :
        Total deposition flow rate (µL/min); split proportionally per pump.
    flush_rate_total :
        Total precondition-flush flow rate (µL/min); split proportionally per pump.
    flush_factor :
        Precondition preload multiplier (preload volume = ``volume * flush_factor``,
        applied inside ``precondition_flush``).
    settle_factor :
        Deposition dwell multiplier: ``settle_wait_s = duration_s * settle_factor``.
    settle_base_s :
        A fixed floor added to the derived settling wait (default 0).
    """
    vols = [max(0.0, float(v)) for v in volumes_uL]
    dep_rates = split_rate(dispense_rate_total, vols)
    flush_rates = split_rate(flush_rate_total, vols)

    deposition = PhaseParams(volumes_uL=vols, rates_uL_min=dep_rates)
    settle_wait_s = deposition.duration_min() * 60.0 * float(settle_factor) + float(settle_base_s)

    return DropcastPlan(
        deposition=deposition,
        flush_rates_uL_min=flush_rates,
        flush_factor=float(flush_factor),
        settle_wait_s=settle_wait_s,
    )
