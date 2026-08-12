"""Overflow guard — is a formulation's cast volume within the well capacity?

A pure, UI-free primitive shared by every formulation-bearing AE surface: the
deposition digital-twin panel, the HT Experiment tab, and the Live BO campaign
pre-flight.  "Overflow" is the same wet-volume-vs-capacity question the deposition
simulator flags per well (:mod:`softae.core.deposition`, ``wet_fill > 1``) and the
same budget the formulation solver fails-safe on
(:mod:`softae.core.formulation`, ``grand_total <= budget``) — lifted into one
place so each surface asks it identically, whether **pointwise** (one formulation)
or **swept** across an entire composition parameter space.

The volume→"is it too much" test is trivial; the value here is a single shared
vocabulary (:class:`OverflowVerdict` / :class:`OverflowSweepResult`) and a single
space-enumeration helper, so the three surfaces cannot drift on the boundary
convention or the "which region overflows" summary.

Callers supply how a composition point becomes a total cast volume (the
``total_for_point`` mapper) by reusing whatever they already use to reach the
hardware — ``plan_formulation(...).grand_total_uL`` /
``solve_formulation(...).grand_total_uL`` in composition mode, or a plain
``sum(volumes)`` in the HT fixed-volume case — so this module stays free of any
dependency on the formulation solver or the DOE layer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "OverflowVerdict",
    "OverflowSweepResult",
    "well_overflow",
    "sweep_overflow",
    "enumerate_space",
]

#: Composition point: parameter/axis name → value.
Point = dict[str, float]

#: Slack (µL) below which a total is treated as *at* capacity, not over it — the
#: same tolerance the formulation solver uses for its budget feasibility check
#: (:mod:`softae.core.formulation`), so the boundary agrees across surfaces.
_TOL_UL = 1e-9


@dataclass(frozen=True)
class OverflowVerdict:
    """Whether one formulation's total cast volume fits the well.

    ``headroom_uL`` is ``capacity - total`` (negative when overflowing), so a
    caller can report both "does it overflow" and "by how much / how much room
    is left" without recomputing.

    ``void_uL`` is optional and splits the *fitting* case in two. On a board with
    non-wetting walls, ``capacity_uL`` already includes a permitted bead standing
    proud of the brim (see :func:`softae.core.geometry.elution_capacity_uL`), so a
    total above the void is still castable — the well is merely full. Passing the
    void lets a caller say that, instead of reporting "fits" for a volume that
    visibly domes over the rim.

    ``overflows`` keeps its exact original meaning: past the capacity, hard stop.
    """

    total_uL: float
    capacity_uL: float
    overflows: bool
    headroom_uL: float
    void_uL: float | None = None

    @property
    def above_brim(self) -> bool:
        """Fits, but only as a bead above a full well. A warning, never a stop."""
        if self.void_uL is None or self.overflows:
            return False
        return self.total_uL > self.void_uL + _TOL_UL

    @property
    def bead_uL(self) -> float:
        """Volume standing above the brim (0.0 when at or below it, or unknown)."""
        if self.void_uL is None:
            return 0.0
        return max(0.0, self.total_uL - self.void_uL)


def well_overflow(
    total_uL: float, capacity_uL: float, *, void_uL: float | None = None,
) -> OverflowVerdict:
    """Verdict for a single formulation: does ``total_uL`` exceed the well?

    ``total_uL`` is the *wet* volume actually placed in the well (the sum of the
    per-component cast volumes).  A total exactly at capacity does not overflow.

    Pass ``void_uL`` — the brim-full volume of the walls alone — to distinguish
    "fits inside the well" from "fits, as a permitted bead above it". Omitted, the
    verdict is exactly what it always was.
    """
    total = float(total_uL)
    cap = float(capacity_uL)
    return OverflowVerdict(
        total_uL=total,
        capacity_uL=cap,
        overflows=total > cap + _TOL_UL,
        headroom_uL=cap - total,
        void_uL=None if void_uL is None else float(void_uL),
    )


@dataclass(frozen=True)
class OverflowSweepResult:
    """Overflow verdicts across a swept composition parameter space."""

    #: One ``(point, verdict)`` per enumerated composition point, in sweep order.
    verdicts: list[tuple[Point, OverflowVerdict]]
    capacity_uL: float

    @property
    def n_points(self) -> int:
        return len(self.verdicts)

    #: Brim-full volume of the walls alone, when the board declared a well.
    void_uL: float | None = None

    @property
    def n_overflow(self) -> int:
        return sum(1 for _, v in self.verdicts if v.overflows)

    @property
    def n_above_brim(self) -> int:
        """Points that cast as a bead above a full well — warn, not stop."""
        return sum(1 for _, v in self.verdicts if v.above_brim)

    @property
    def any_above_brim(self) -> bool:
        return any(v.above_brim for _, v in self.verdicts)

    @property
    def any_overflow(self) -> bool:
        return any(v.overflows for _, v in self.verdicts)

    @property
    def all_overflow(self) -> bool:
        return bool(self.verdicts) and all(v.overflows for _, v in self.verdicts)

    @property
    def overflow_fraction(self) -> float:
        """Fraction of enumerated points that overflow (0.0 when empty)."""
        return self.n_overflow / self.n_points if self.verdicts else 0.0

    @property
    def worst(self) -> tuple[Point, OverflowVerdict] | None:
        """The point with the least headroom (the worst overflow), or ``None``."""
        if not self.verdicts:
            return None
        return min(self.verdicts, key=lambda pv: pv[1].headroom_uL)

    @property
    def max_total_uL(self) -> float:
        """Largest total cast volume seen — the peak the well must accommodate."""
        return max((v.total_uL for _, v in self.verdicts), default=0.0)

    def overflowing_points(self) -> list[tuple[Point, OverflowVerdict]]:
        return [(p, v) for p, v in self.verdicts if v.overflows]


def sweep_overflow(
    points: Sequence[Mapping[str, Any]],
    total_for_point: Callable[[Mapping[str, Any]], float],
    capacity_uL: float,
    *,
    void_uL: float | None = None,
) -> OverflowSweepResult:
    """Flag overflow at every point of a composition parameter space.

    Parameters
    ----------
    points :
        The enumerated composition points (e.g. from :func:`enumerate_space` or
        an :class:`~softae.campaigns.doe.ExperimentDesign` candidate pool).
    total_for_point :
        Maps one point to its total cast volume (µL).  Reuse the caller's own
        composition→volume path so the sweep and the eventual run agree — e.g.
        ``lambda p: plan_formulation(p, ctx).grand_total_uL``.
    capacity_uL :
        The per-well budget — the hard stop
        (``geometry.elution_capacity_uL``: void plus any permitted bead).
    void_uL :
        Optional brim-full volume of the walls alone, so the sweep can report
        which points cast above the brim as well as which overflow outright.
    """
    cap = float(capacity_uL)
    void = None if void_uL is None else float(void_uL)
    verdicts: list[tuple[Point, OverflowVerdict]] = []
    for pt in points:
        total = float(total_for_point(pt))
        verdicts.append((dict(pt), well_overflow(total, cap, void_uL=void)))
    return OverflowSweepResult(verdicts=verdicts, capacity_uL=cap, void_uL=void)


def _axis_values(spec: Any, *, default_steps: int) -> list[Any]:
    """Enumerate one axis of a parameter space into concrete sample values.

    Recognised specs:

    * an explicit sequence of values → used verbatim;
    * a bounds mapping ``{"low": lo, "high": hi, "steps"?: k, "type"?: ...}`` (the
      BO ``parameter_space`` shape) → a ``k``-point linear grid (``int`` axes are
      rounded; ``categorical`` axes enumerate ``choices``/``values``);
    * an object exposing ``.values()`` (e.g. a DOE ``ParamScale``) → its values.
    """
    # Mapping first: a bounds/categorical spec is itself a dict (and a dict has a
    # ``.values()`` method, so the duck-type check below must not see it).
    if isinstance(spec, Mapping):
        kind = str(spec.get("type", "float")).lower()
        if kind == "categorical" or "choices" in spec or "values" in spec:
            return list(spec.get("choices", spec.get("values", [])))
        if "low" in spec and "high" in spec:
            lo, hi = float(spec["low"]), float(spec["high"])
            n = int(spec.get("steps", default_steps))
            vals = _linspace(lo, hi, n)
            if kind == "int":
                vals = sorted({int(round(v)) for v in vals})
            return list(vals)
        raise ValueError(f"unrecognised axis spec: {spec!r}")
    # A DOE ParamScale (or anything exposing an axis enumerator).
    if hasattr(spec, "values") and callable(getattr(spec, "values")):
        return list(spec.values())
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return list(spec)
    raise ValueError(f"unrecognised axis spec: {spec!r}")


def enumerate_space(
    space: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    steps: int = 5,
) -> list[Point]:
    """Enumerate a composition parameter space into a list of point dicts.

    Accepts either an already-enumerated **pool** (a sequence of point dicts —
    e.g. :meth:`ExperimentDesign.candidate_pool`, returned unchanged) or a
    **bounds/axes mapping** ``{name: spec}`` (the BO ``parameter_space`` shape or
    DOE ``ParamScale`` axes), which is grid-sampled per axis (``steps`` points by
    default, or the axis's own ``steps``) and Cartesian-producted.

    ``steps`` trades coverage for cost; a caller wanting only the extreme corners
    can pass ``steps=2``.
    """
    # Already an explicit pool of points.
    if isinstance(space, Sequence) and not isinstance(space, (str, bytes, Mapping)):
        return [dict(p) for p in space]

    if not isinstance(space, Mapping):
        raise TypeError(f"space must be a pool or an axes mapping, got {type(space)!r}")

    names = list(space.keys())
    value_lists = [_axis_values(space[n], default_steps=steps) for n in names]
    pool: list[Point] = []
    for combo in itertools.product(*value_lists) if value_lists else [()]:
        pool.append(dict(zip(names, combo)))
    return pool


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """``n`` evenly spaced points on ``[lo, hi]`` inclusive (``n==1`` → ``[lo]``)."""
    if n <= 1:
        return [float(lo)]
    step = (float(hi) - float(lo)) / (n - 1)
    return [float(lo) + i * step for i in range(n)]
