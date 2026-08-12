"""Composition *axes*: a formulation target whose value the optimizer searches.

The deposition twin fixes a composition by naming targets — "EO:Li = 20",
"SiO2 is 10 % of the dried volume" — and solving for the stock volumes that meet
them. A campaign wants the same vocabulary with a **range** instead of a value:
search EO:Li over 5–40 while holding the silica fraction at 0.1, and let the solver
turn each suggestion into volumes.

That is all a :class:`CompositionAxis` is — one :class:`~softae.core.formulation`
target with ``low``/``high`` in place of ``value``. Fixed targets are expressed as
an axis with ``low == high``, so a campaign carries **one** list rather than two
that must be kept consistent.

**Why this is worth the indirection.** Raw pump volumes are the easier thing to
search — feasibility is native, a volume limit is just a bound — but the twin cannot
say what a volume *is* without stock identity, so there is no dry thickness and
therefore no conductivity (see ``[eis] objective``). Searching composition instead
gives every trial a predicted thickness, which is what makes σ the objective rather
than mean |Z|. Same rig, same solver, different question asked of it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from softae.core.formulation import (
    Basis,
    ConcentrationTarget,
    DriedFractionTarget,
    FormulationTarget,
    MolarRatioTarget,
)

logger = structlog.get_logger(__name__)

#: Target kinds an axis can carry, in the vocabulary the twin's editor already uses.
AXIS_KINDS = ("molar_ratio", "dried_fraction", "concentration")

#: Human labels, shared by the GUI so the two cannot drift apart.
AXIS_LABELS = {
    "molar_ratio": "Molar ratio",
    "dried_fraction": "Dried fraction",
    "concentration": "Concentration",
}


def _slug(text: str) -> str:
    """A parameter-name-safe fragment of a chemical name."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_") or "x"


@dataclass(frozen=True)
class CompositionAxis:
    """One searchable composition target.

    ``low == high`` pins the target instead of searching it — the optimizer is never
    handed a degenerate dimension, because :func:`axes_parameter_space` omits it and
    :func:`build_targets_from_axes` substitutes the constant.
    """

    kind: str
    a: str
    b: str = ""
    low: float = 0.0
    high: float = 1.0
    basis: str = Basis.VOLUME.value

    def __post_init__(self) -> None:
        if self.kind not in AXIS_KINDS:
            raise ValueError(f"unknown axis kind {self.kind!r}; expected {AXIS_KINDS}")
        if not str(self.a).strip():
            raise ValueError(f"{AXIS_LABELS[self.kind]} axis needs a species/component")
        if self.kind == "molar_ratio" and not str(self.b).strip():
            raise ValueError("a molar-ratio axis needs both A and B")
        if self.high < self.low:
            raise ValueError(
                f"axis '{self.name}' has high ({self.high}) below low ({self.low})"
            )

    @property
    def name(self) -> str:
        """The parameter name the optimizer searches, and the DOE column it lands in."""
        if self.kind == "molar_ratio":
            return f"ratio_{_slug(self.a)}_{_slug(self.b)}"
        if self.kind == "dried_fraction":
            return f"driedfrac_{_slug(self.a)}"
        return f"conc_{_slug(self.a)}"

    @property
    def is_fixed(self) -> bool:
        """A pinned target — held constant, not searched."""
        return self.high <= self.low

    def describe(self) -> str:
        label = AXIS_LABELS[self.kind]
        subject = f"{self.a}/{self.b}" if self.kind == "molar_ratio" else self.a
        if self.kind == "dried_fraction":
            subject = f"{subject} ({self.basis})"
        span = (f"= {self.low:g}" if self.is_fixed
                else f"∈ [{self.low:g}, {self.high:g}]")
        return f"{label} {subject} {span}"

    def target(self, value: float) -> FormulationTarget:
        """This axis at *value* as a solver target."""
        if self.kind == "molar_ratio":
            return MolarRatioTarget(self.a, self.b, float(value))
        if self.kind == "dried_fraction":
            return DriedFractionTarget(self.a, float(value), Basis(self.basis))
        return ConcentrationTarget(self.a, float(value))


def axes_parameter_space(axes: Sequence[CompositionAxis]) -> dict[str, dict[str, Any]]:
    """``parameter_space`` for the searchable axes — pinned ones are excluded.

    Handing the optimizer a dimension whose bounds are equal costs a GP dimension
    and a scaling division by zero to learn nothing, so a pinned target stays a
    constant of the campaign rather than becoming a degenerate axis.
    """
    space: dict[str, dict[str, Any]] = {}
    for axis in axes:
        if axis.is_fixed:
            continue
        space[axis.name] = {"type": "float", "low": float(axis.low),
                            "high": float(axis.high)}
    return space


def build_targets_from_axes(
    axes: Sequence[CompositionAxis],
) -> Callable[[dict[str, Any]], list[FormulationTarget]]:
    """``params -> targets``, the callable :class:`GeneralFormulation` expects.

    Pinned axes contribute their fixed value; searched axes read the suggestion.
    A searched axis missing from *params* falls back to its ``low`` rather than
    raising — a solver that runs on a stale bound is recoverable, a campaign that
    dies mid-round on a ``KeyError`` is not.
    """
    frozen = tuple(axes)

    def _build(params: dict[str, Any]) -> list[FormulationTarget]:
        out: list[FormulationTarget] = []
        for axis in frozen:
            if axis.is_fixed:
                value = axis.low
            elif axis.name in params:
                value = params[axis.name]
            else:
                logger.warning("composition_axis_missing", axis=axis.name,
                               msg="suggestion lacks this axis; using its lower bound")
                value = axis.low
            out.append(axis.target(value))
        return out

    return _build


def stocks_from_loadout(
    loadout: Any, sol_catalog: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    """``(stocks, pump_assignment)`` for the solutions currently on the pumps.

    The loadout is the authority on *what is physically loaded where* — the same
    record the consumables ledger and the particulate-pump resolver read — so a
    composition campaign is built from what the rig actually holds rather than from
    a list typed into the campaign form. A pump naming a solution the catalog does
    not contain is skipped with a warning: an unknown stock cannot be solved for,
    and silently dropping it would change the composition without saying so.
    """
    stocks: dict[str, Any] = {}
    pump_assignment: dict[str, int] = {}
    for pump_id in getattr(loadout, "declared_pumps", tuple)():
        name = loadout.solution_for(pump_id)
        if not name:
            continue
        try:
            stocks[name] = sol_catalog.get(name)
        except (KeyError, AttributeError):
            logger.warning("loadout_stock_not_in_catalog", pump=pump_id, stock=name)
            continue
        pump_assignment[name] = int(pump_id)
    return stocks, pump_assignment


def describe_axes(axes: Sequence[CompositionAxis]) -> str:
    """One line for the log and the campaign header."""
    if not axes:
        return "no composition targets"
    searched = sum(1 for a in axes if not a.is_fixed)
    return (f"{len(axes)} target(s), {searched} searched: "
            + "; ".join(a.describe() for a in axes))


def validate_axes(axes: Sequence[CompositionAxis], *, n_stocks: int) -> list[str]:
    """Problems an operator should fix before launching. Empty means launchable.

    The determinacy rule is the solver's, not this module's: ``solve_formulation``
    needs one target per stock, and ``TotalDepositTarget`` (the deposition-µL box)
    supplies one of them. Under-constraining does not fail loudly — it yields *a*
    solution that satisfies the targets given while leaving the rest to the solver's
    own preference, which is how a campaign ends up exploring an axis nobody chose.
    """
    issues: list[str] = []
    if not axes:
        return ["Add at least one composition target."]

    names = [a.name for a in axes]
    for name in sorted(set(names)):
        if names.count(name) > 1:
            issues.append(f"'{name}' is declared {names.count(name)} times.")

    if not any(not a.is_fixed for a in axes):
        issues.append(
            "Every target is pinned (low == high), so there is nothing to search.")

    needed = max(0, int(n_stocks) - 1)   # the deposition-µL target covers one stock
    if n_stocks and len(axes) < needed:
        issues.append(
            f"{len(axes)} target(s) for {n_stocks} stocks — the solver needs "
            f"{needed} beside the deposition volume, or the composition is "
            f"under-determined."
        )
    return issues
