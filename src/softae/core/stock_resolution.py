"""One answer to "which stock is on which pump", for everything that asks.

Two records say what is in the syringes and they never met. **Intent** is the
spec's ``[general_formulation]`` — ``stocks`` plus ``pump_assignment``, validated
against the solution catalog by
:func:`~softae.core.campaign_spec_fields.decode_general_formulation`.
**Reality** is :class:`~softae.core.stock_assignment.PumpLoadout`, declared at the
bench and persisted as one ``rig_state`` row. Nothing compared them at launch, and
no run artifact recorded a stock name against a pump at all, so a spec naming
last month's stocks loaded clean and cast the wrong chemistry.

This module is the comparison, and nothing else: pure, no I/O, no catalogs read
from disk, no GUI. The caller supplies the spec, the loadout it already read and
the catalog it already holds; the result is a frozen record both the Final-Check
digest and ``provenance.json`` are rendered from, so the page an operator reads
and the file a reader reproduces from cannot come to say different things.

**The rule the result exists to serve:** everywhere a run records a volume, it
records both the pump index and the resolved stock name. The pump index is the
physical fact — the direction ``stock_assignment`` insists on — and the name is
its interpretation, so a wrong name stays checkable later against the pump the
volume actually left.

**Three absences stay three distinct answers**, because they call for three
different actions: a pump the spec assigns and the bench does not declare
(:attr:`~ResolvedStocks.undeclared`) means *go and declare it*; a pump the bench
declares and the spec does not use (:attr:`~ResolvedStocks.unused_declared`) is
ordinary; a pump both name differently (:attr:`~ResolvedStocks.disagreements`) is
the launch that casts the wrong chemistry. Collapsing any two of them into one
"missing" would lose exactly the distinction that decides what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from softae.core.catalog_digest import canonical

__all__ = [
    "SPEC", "DECLARED", "NONE", "UNRESOLVED_KEY",
    "ResolvedStocks", "resolve_stocks", "spec_stock_by_pump",
    "stocks_resolved_event",
]

#: The spec's ``pump_assignment`` was authoritative.
SPEC = "spec"
#: The bench's declared loadout was authoritative (the spec named no pumps).
DECLARED = "declared"
#: Neither record names a stock against a pump. Distinct from "both agree that
#: nothing is loaded", which no surface here can say.
NONE = "none"

#: Key of the tagged marker standing in for a catalog row that could not be read.
#: A marker rather than ``None`` because "this name is not in the catalog" and
#: "there was no catalog to look it up in" are different facts, and a reader of
#: ``provenance.json`` months later cannot ask which one happened.
UNRESOLVED_KEY = "__unresolved__"

_NO_CATALOG = {UNRESOLVED_KEY: "no solution catalog was supplied"}
_NOT_CATALOGUED = {UNRESOLVED_KEY: "not in the solution catalog"}

#: Joins two stock names assigned to one pump. ``pump_assignment`` is
#: name → pump, so nothing stops two stocks naming the same line; the inverted
#: view has to say so rather than silently keep whichever came last.
_BOTH = " + "


def spec_stock_by_pump(spec: Any) -> dict[int, str]:
    """``pump index -> stock name`` the spec assigns, inverted from its own map.

    Empty when the spec declares no ``[general_formulation]``, which is a
    campaign that names no stocks at all rather than one that names none loaded.
    """
    formulation = getattr(spec, "general_formulation", None)
    assignment = getattr(formulation, "pump_assignment", None) or {}
    out: dict[int, str] = {}
    for stock, pump in assignment.items():
        pid = int(pump)
        out[pid] = f"{out[pid]}{_BOTH}{stock}" if pid in out else str(stock)
    return out


@dataclass(frozen=True)
class ResolvedStocks:
    """What is on which pump, which record said so, and where the two differ."""

    #: :data:`SPEC`, :data:`DECLARED` or :data:`NONE` — never inferred by a
    #: reader from whether ``by_pump`` happens to be empty.
    source: str
    #: The answer, pump-first. Empty under :data:`NONE`.
    by_pump: dict[int, str] = field(default_factory=dict)
    #: The same answer name-first, for a caller that has a stock and wants its
    #: line. Not the strict inverse of :attr:`by_pump`: two stocks on one pump
    #: keep their own entries here while ``by_pump`` shows the joined pair.
    by_name: dict[str, int] = field(default_factory=dict)
    #: ``stock name -> catalog row`` for every resolved name, snapshotted so a
    #: later reader is not left trusting that the catalog has not moved.
    catalog_rows: dict[str, Any] = field(default_factory=dict)
    #: ``pump -> {"spec": …, "declared": …}`` where both records name a stock and
    #: the names differ. Reported, never merged: neither side is evidence about
    #: the other, and picking one silently is the failure this module exists for.
    disagreements: dict[int, dict[str, str]] = field(default_factory=dict)
    #: Pumps the spec assigns that the bench declares nothing on.
    undeclared: tuple[int, ...] = ()
    #: Pumps the bench declares that the spec assigns nothing to.
    unused_declared: tuple[int, ...] = ()
    #: The two halves as read, kept so a caller rendering a per-pump comparison
    #: does not have to invert the spec a second time and risk inverting it
    #: differently. Deliberately not in :meth:`to_record`.
    spec_by_pump: dict[int, str] = field(default_factory=dict)
    declared_by_pump: dict[int, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether nothing names a stock against a pump."""
        return self.source == NONE

    def to_record(self) -> dict[str, Any]:
        """The JSON-safe form written into ``provenance.json``.

        Both directions are written, per the rule in the module docstring: the
        pump indices are the physical fact and the names are the interpretation,
        and a record carrying only one of them cannot be checked against the
        other afterwards.
        """
        return {
            "source": self.source,
            "by_pump": {str(pid): name for pid, name in sorted(self.by_pump.items())},
            "by_name": {name: int(pid) for name, pid in sorted(self.by_name.items())},
            "catalog_rows": {name: self.catalog_rows[name]
                             for name in sorted(self.catalog_rows)},
            "disagreements": {str(pid): dict(sides)
                              for pid, sides in sorted(self.disagreements.items())},
            "undeclared": [int(p) for p in self.undeclared],
            "unused_declared": [int(p) for p in self.unused_declared],
        }


def _catalog_row(name: str, sol_catalog: Any) -> Any:
    """One solution's catalog row as JSON-safe primitives, or a tagged marker."""
    if sol_catalog is None:
        return dict(_NO_CATALOG)
    try:
        solution = sol_catalog.get(name)
    except Exception:
        return dict(_NOT_CATALOGUED)
    if solution is None:
        return dict(_NOT_CATALOGUED)
    return canonical(solution)


def resolve_stocks(spec: Any, loadout: Any = None,
                   sol_catalog: Any = None) -> ResolvedStocks:
    """Compare the spec's pump assignment against the bench's declared loadout.

    *loadout* may be ``None`` (no store was open, so the bench said nothing) and
    *sol_catalog* may be ``None`` (the rows are then marked unresolved rather
    than omitted). Neither absence is reported as agreement.

    **Which record wins.** The spec does, whenever it names any pump: a campaign
    is launched from a file, and the file is the intent under test. The bench
    wins only when the spec names none — the ``stocks = "declared"`` direction —
    and when neither names one the answer is :data:`NONE`, not an empty ``spec``.
    """
    from_spec = spec_stock_by_pump(spec)
    declared = {int(pid): str(name)
                for pid, name in (getattr(loadout, "by_pump", None) or {}).items()
                if name}

    if from_spec:
        source, by_pump = SPEC, dict(from_spec)
    elif declared and getattr(spec, "general_formulation", None) is not None:
        source, by_pump = DECLARED, dict(declared)
    else:
        source, by_pump = NONE, {}

    by_name: dict[str, int] = {}
    if source == SPEC:
        assignment = getattr(spec.general_formulation, "pump_assignment", None) or {}
        by_name = {str(stock): int(pump) for stock, pump in assignment.items()}
    elif source == DECLARED:
        by_name = {name: pid for pid, name in sorted(declared.items())}

    disagreements = {
        pid: {"spec": from_spec[pid], "declared": declared[pid]}
        for pid in sorted(set(from_spec) & set(declared))
        if from_spec[pid] != declared[pid]
    }
    return ResolvedStocks(
        source=source,
        by_pump=by_pump,
        by_name=by_name,
        catalog_rows={name: _catalog_row(name, sol_catalog) for name in by_name},
        disagreements=disagreements,
        undeclared=tuple(sorted(set(from_spec) - set(declared))),
        unused_declared=tuple(sorted(set(declared) - set(from_spec))),
        spec_by_pump=dict(from_spec),
        declared_by_pump=dict(declared),
    )


def stocks_resolved_event(resolved: ResolvedStocks) -> dict[str, Any]:
    """Payload for one ``stocks_resolved`` record on the campaign event stream.

    :attr:`~ResolvedStocks.catalog_rows` is left out: the rows are bulky, they do
    not change during a run, and ``provenance.json`` already holds them beside
    this event in the same run directory. Everything a reader of ``events.jsonl``
    needs to say what was cast — both directions, and both disagreements and
    absences — is here.
    """
    record = resolved.to_record()
    record.pop("catalog_rows", None)
    return record
