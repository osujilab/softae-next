"""Method-maturity lifecycle — the logic layer for the development pipeline.

Pure logic over a :class:`~softae.core.task_catalog.TaskCatalog`: read a
capability's maturity, check whether a promotion's gate is met, and perform the
``promote`` / ``sign_off`` transitions that advance it.  Persistence stays the
caller's responsibility (``catalog.save_toml``) so this module has no I/O of its
own beyond an optional DataStore lookup to verify a sign-off run.

See ``docs/METHOD_MATURITY_PIPELINE.md`` for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import IntEnum
from typing import Any

import structlog

from softae.core.task_catalog import Task, TaskCatalog

logger = structlog.get_logger(__name__)


class Maturity(IntEnum):
    """Ordered method-maturity stages (draft < prototype < tested < validated)."""

    DRAFT = 0
    PROTOTYPE = 1
    TESTED = 2
    VALIDATED = 3

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: "Maturity | str | int") -> "Maturity":
        """Coerce a label / int / Maturity into a :class:`Maturity`."""
        if isinstance(value, Maturity):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            valid = ", ".join(m.label for m in cls)
            raise ValueError(
                f"unknown maturity '{value}'; expected one of: {valid}"
            ) from exc


class GateError(RuntimeError):
    """Raised when a promotion is refused because its gate is unmet."""

    def __init__(self, name: str, to: Maturity, reasons: list[str]):
        self.method = name
        self.to = to
        self.reasons = reasons
        joined = "\n  - ".join(reasons)
        super().__init__(
            f"cannot promote '{name}' to '{to.label}':\n  - {joined}"
        )


@dataclass
class LifecycleStatus:
    """A snapshot of a capability's lifecycle state."""

    name: str
    maturity: Maturity
    version: int
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [f"{self.name}  [{self.maturity.label}]  v{self.version}"]
        if self.provenance:
            src = self.provenance.get("source", "")
            by = self.provenance.get("ported_by", "")
            on = self.provenance.get("ported_on", "")
            lines.append(f"  provenance: {src}  (ported by {by}, {on})")
        tests = self.evidence.get("tests") or []
        if tests:
            lines.append(f"  tests: {len(tests)} linked")
        if self.evidence.get("validated_run_id"):
            lines.append(
                f"  validated: run {self.evidence['validated_run_id']} "
                f"by {self.evidence.get('validated_by', '?')} "
                f"on {self.evidence.get('validated_on', '?')}"
            )
        return "\n".join(lines)


# ── Reading ──────────────────────────────────────────────────────────────────

def method_maturity(name: str, catalog: TaskCatalog) -> Maturity:
    """Maturity of a single catalog method."""
    return Maturity.parse(catalog.get(name).maturity)


def status(name: str, catalog: TaskCatalog) -> LifecycleStatus:
    """A :class:`LifecycleStatus` snapshot for one method."""
    t = catalog.get(name)
    return LifecycleStatus(
        name=name,
        maturity=Maturity.parse(t.maturity),
        version=t.version,
        provenance=dict(t.provenance),
        evidence=dict(t.evidence),
    )


def effective_maturity(
    name: str,
    catalog: TaskCatalog,
    recipes: Any = None,
) -> Maturity:
    """Effective maturity of a method or recipe.

    A method's is its own recorded stage.  A recipe's is capped by its least
    mature dependency: ``min(own, *[maturity(m) for m in recipe.methods])`` —
    so a recipe never presents as more validated than the methods it runs.
    """
    if recipes is not None and name in recipes:
        recipe = recipes.get(name)
        own = Maturity.parse(recipe.maturity)
        deps = [method_maturity(m, catalog) for m in recipe.methods if m in catalog]
        return min([own, *deps]) if deps else own
    return method_maturity(name, catalog)


# ── Workflow maturity scan (warn-and-proceed input) ──────────────────────────

def _unique_catalog_method(catalog: TaskCatalog, instrument: str, method: str) -> str | None:
    """The catalog task uniquely matching (instrument, method), else None.

    Ambiguous matches (e.g. several ``syringe.single_pump`` tasks) and no-match
    steps return None — they are simply untracked for maturity purposes, so the
    scan focuses on the distinct composite capabilities we actually gate.
    """
    matches = [
        n for n in catalog.list_names()
        if (t := catalog.get(n)).instrument == instrument and t.method == method
    ]
    return matches[0] if len(matches) == 1 else None


def workflow_method_maturities(workflow: Any, catalog: TaskCatalog) -> dict[str, Maturity]:
    """Map a workflow's steps to their catalog methods → maturities.

    Steps whose (instrument, method) resolves to a unique catalog task are
    tracked; others are skipped (see :func:`_unique_catalog_method`).
    """
    out: dict[str, Maturity] = {}
    for step in workflow.resolve_steps():
        name = _unique_catalog_method(catalog, step.instrument, step.method)
        if name is not None:
            out[name] = method_maturity(name, catalog)
    return out


def maturity_warnings(
    workflow: Any,
    catalog: TaskCatalog,
    *,
    expected: "Maturity | str | int" = Maturity.VALIDATED,
) -> list[dict[str, str]]:
    """Tracked methods in *workflow* below the *expected* maturity.

    Each item: ``{"method", "maturity", "expected"}``.  Empty means every
    tracked method meets the bar.  This is the input to the autonomous
    warn-and-proceed guard — it never blocks.
    """
    exp = Maturity.parse(expected)
    return [
        {"method": n, "maturity": m.label, "expected": exp.label}
        for n, m in sorted(workflow_method_maturities(workflow, catalog).items())
        if m < exp
    ]


# ── DataStore helpers (sign-off verification) ────────────────────────────────

def _run_row(data_store: Any, run_id: str) -> dict[str, Any] | None:
    try:
        for row in data_store.query_runs():
            if row.get("run_id") == run_id:
                return row
    except Exception:
        return None
    return None


# ── Gates ────────────────────────────────────────────────────────────────────

def _gate_reasons(task: Task, stage: Maturity, *, data_store: Any = None) -> list[str]:
    """Reasons the *stage*'s own entry gate is unmet for *task* (empty = met)."""
    reasons: list[str] = []
    if stage == Maturity.PROTOTYPE:
        # Methods (Tasks) must be runnable; protocols have no instrument/method.
        if hasattr(task, "instrument") and not (task.instrument and task.method):
            reasons.append("method has no instrument/method — not runnable")
    elif stage == Maturity.TESTED:
        if not (task.evidence.get("tests") or []):
            reasons.append(
                "no linked tests (evidence.tests) — required for 'tested'"
            )
    elif stage == Maturity.VALIDATED:
        ev = task.evidence
        run_id = ev.get("validated_run_id")
        if not run_id:
            reasons.append(
                "no validated_run_id — a recorded real-hardware run is required"
            )
        if not ev.get("validated_by"):
            reasons.append("no validated_by — an operator sign-off is required")
        if run_id and data_store is not None and _run_row(data_store, run_id) is None:
            reasons.append(f"validated_run_id '{run_id}' not found in the DataStore")
    return reasons


def can_promote(
    name: str,
    to: "Maturity | str | int",
    catalog: TaskCatalog,
    *,
    data_store: Any = None,
) -> tuple[bool, list[str]]:
    """Whether *name* can advance to *to*; returns ``(ok, reasons)``.

    A forward promotion must satisfy the entry gate of **every** stage it
    crosses (so promoting straight to ``validated`` still requires linked tests
    from the ``tested`` gate).
    """
    to = Maturity.parse(to)
    task = catalog.get(name)
    current = Maturity.parse(task.maturity)
    if to <= current:
        return False, [f"already at '{current.label}' (>= '{to.label}')"]

    reasons: list[str] = []
    for stage in Maturity:
        if current < stage <= to:
            reasons.extend(_gate_reasons(task, stage, data_store=data_store))
    return (not reasons, reasons)


# ── Transitions ──────────────────────────────────────────────────────────────

def link_tests(name: str, tests: list[str], catalog: TaskCatalog) -> Task:
    """Record the pytest node IDs that prove *name* (evidence.tests)."""
    task = catalog.get(name)
    task.evidence["tests"] = list(tests)
    catalog.add(task)
    return task


def promote(
    name: str,
    to: "Maturity | str | int",
    catalog: TaskCatalog,
    *,
    data_store: Any = None,
) -> Task:
    """Advance *name* to *to*, or raise :class:`GateError` if the gate is unmet."""
    to = Maturity.parse(to)
    ok, reasons = can_promote(name, to, catalog, data_store=data_store)
    if not ok:
        raise GateError(name, to, reasons)
    task = catalog.get(name)
    task.maturity = to.label
    catalog.add(task)
    logger.info("method_promoted", method=name, maturity=to.label)
    return task


def supersede(
    name: str,
    new_task: Task,
    catalog: TaskCatalog,
    *,
    reset_to: "Maturity | str | int" = Maturity.DRAFT,
) -> tuple[str, Task]:
    """Replace method *name* with *new_task*, archiving the old version.

    A behavior-changing change never overwrites in place (the rollback
    guarantee): the prior entry is retained under ``name@v<old>`` with
    ``provenance.superseded_by`` set, while *new_task* takes the canonical
    ``name`` at ``version = old+1``, records ``provenance.supersedes``, and has
    its maturity reset (a new version is unproven).  Any role still bound to the
    archived key continues to resolve, so reverting is harmless.

    Returns ``(archived_key, new_task)``.
    """
    old = catalog.get(name)
    old_version = int(old.version or 1)
    archived_key = f"{name}@v{old_version}"

    old.name = archived_key
    old.provenance = {**old.provenance, "superseded_by": name}
    catalog.remove(name)
    catalog.add(old)

    new_task.name = name
    new_task.version = old_version + 1
    new_task.provenance = {**new_task.provenance, "supersedes": archived_key}
    new_task.maturity = Maturity.parse(reset_to).label
    catalog.add(new_task)
    logger.info("method_superseded", method=name, archived=archived_key,
                version=new_task.version)
    return archived_key, new_task


def version_chain(name: str, catalog: TaskCatalog) -> list[str]:
    """All catalog keys for *name* — the canonical name plus any ``name@vN``."""
    return sorted(
        n for n in catalog.list_names()
        if n == name or n.startswith(f"{name}@v")
    )


def sign_off(
    name: str,
    *,
    run_id: str,
    by: str,
    catalog: TaskCatalog,
    config_hash: str | None = None,
    notes: str = "",
    data_store: Any = None,
    on: str | None = None,
) -> Task:
    """Record a hardware sign-off and promote *name* to ``validated``.

    ``config_hash`` is pulled from the DataStore run when not supplied.  The
    ``validated`` gate is still enforced (linked tests + a resolvable run), so a
    sign-off cannot skip the ladder.
    """
    task = catalog.get(name)
    if config_hash is None and data_store is not None:
        row = _run_row(data_store, run_id)
        if row is not None:
            config_hash = row.get("config_hash", "")

    task.evidence.update(
        {
            "validated_run_id": run_id,
            "validated_by": by,
            "validated_on": on or date.today().isoformat(),
            "config_hash": config_hash or "",
            "notes": notes,
        }
    )
    catalog.add(task)
    # Enforce the full gate (tests linked + run resolvable) as it sets validated.
    return promote(name, Maturity.VALIDATED, catalog, data_store=data_store)
