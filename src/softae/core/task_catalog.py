"""Task catalog — named, reusable atomic step instances.

A :class:`Task` is one *atomic* automated step (a single instrument method call
with fixed parameters), catalogued under a human-readable name so it can be
inserted into a process recipe from the Process Configuration tab.  A Task maps
1:1 to a :class:`~softae.workflows.workflow_model.WorkflowStep`.

The catalog is TOML-backed (``data_root()/tasks.toml``) and rewritten whole on
every save — task params are heterogeneous and nested, and the file is
app-owned (no user comments to preserve), so a whole-file
``tomllib``/``tomli_w`` round-trip is simpler and safer than the in-place
line-editing used for ``softae_config.toml``.

The public surface (``add``/``remove``/``get``/``list_names``/``__len__`` +
``save_toml``/``load_toml``) mirrors :class:`ChemicalCatalog` /
:class:`SolutionCatalog` in :mod:`softae.core.formulation`, including the
"missing file → empty catalog, never raise" load contract.

Loading also *validates* (:func:`validate_task`): a task whose declared ceiling
cannot outlast the hold it asks for is rejected rather than catalogued, since
that inconsistency only shows up hours into an unattended run.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import tomli_w

from softae.errors import ValidationError_
from softae.workflows.workflow_model import WorkflowStep

logger = structlog.get_logger(__name__)

#: Method name whose ``hold_time_s`` param dictates how long the step must live.
ANNEAL_METHOD = "anneal"


class TaskValidationError(ValidationError_):
    """A catalogued task is internally inconsistent and must not be run."""


def _positive_float(value: Any) -> float | None:
    """Coerce *value* to a positive float, or ``None`` if it isn't one."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def validate_task(task: "Task") -> list[str]:
    """Return the reasons *task* must not be run — empty list when it is sound.

    Today there is exactly one rule, and it exists because the failure it
    prevents is silent and expensive.  An ``anneal`` step's duration is set by
    its ``hold_time_s`` **param**, while its execution ceiling is the separate
    ``timeout_s`` **field**; a run-plan override
    (:attr:`~softae.core.run_plan.RunPhase.anneal_params`) rewrites the former
    and not the latter.  A task shipping an 8 h hold under a 600 s ceiling is
    killed mid-hold with the stage hot — overnight, unattended, no operator.

    :func:`~softae.core.deposition_recipe.anneal_timeout_s` already raises the
    ceiling at *build* time for the deposition engine, so this is defence in
    depth rather than the only guard — but it is the one that covers the paths
    that hand a catalogued task straight to a step (Process Studio, the sandbox,
    a hand-built workflow) and the one that catches the mistake in the file
    rather than on the rig.
    """
    if task.method != ANNEAL_METHOD:
        return []
    hold = _positive_float(task.params.get("hold_time_s"))
    if hold is None:  # no declared hold → nothing to outlast
        return []
    if task.timeout_s is None:
        return [
            f"anneal task declares hold_time_s={hold:g} s but no timeout_s; "
            f"the executor's default ceiling will kill the hold partway"
        ]
    if float(task.timeout_s) <= hold:
        return [
            f"anneal task timeout_s={float(task.timeout_s):g} s does not exceed "
            f"hold_time_s={hold:g} s; the hold would be aborted with the stage hot "
            f"(allow the hold plus ramp, settle and setpoint restore)"
        ]
    return []


@dataclass
class Task:
    """One catalogued atomic step (a single instrument+method+params call).

    Fields mirror :class:`WorkflowStep` (minus ``depends_on``/``tags``, which are
    recipe-composition concerns) plus light catalog metadata.
    """

    name: str
    instrument: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    timeout_s: float | None = None
    retry: int = 0
    description: str = ""
    category: str = ""  # palette grouping, e.g. liquid_handling|thermal|eis|piezo|stage

    # ── Method-maturity lifecycle (see docs/METHOD_MATURITY_PIPELINE.md) ──
    # All default so pre-lifecycle catalogs load unchanged.
    maturity: str = "draft"  # draft | prototype | tested | validated
    version: int = 1
    provenance: dict[str, Any] = field(default_factory=dict)  # source/ported_by/ported_on/…
    evidence: dict[str, Any] = field(default_factory=dict)  # tests/validated_run_id/…

    def validate(self) -> list[str]:
        """Reasons this task must not be run (see :func:`validate_task`)."""
        return validate_task(self)

    def to_step(self, step_name: str | None = None) -> WorkflowStep:
        """Build a :class:`WorkflowStep` from this task.

        ``step_name`` overrides the step's name (used when inserting a task more
        than once into the same phase); ``depends_on``/``tags`` are left empty.
        """
        return WorkflowStep(
            name=step_name or self.name,
            instrument=self.instrument,
            method=self.method,
            params=dict(self.params),
            timeout_s=self.timeout_s,
            retry=self.retry,
        )

    @classmethod
    def from_step(
        cls,
        step: WorkflowStep,
        *,
        name: str | None = None,
        description: str = "",
        category: str = "",
    ) -> Task:
        """Create a catalog Task from a :class:`WorkflowStep`."""
        return cls(
            name=name or step.name,
            instrument=step.instrument,
            method=step.method,
            params=dict(step.params),
            timeout_s=step.timeout_s,
            retry=step.retry,
            description=description,
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a TOML-friendly table (omits empty/default fields)."""
        d: dict[str, Any] = {"instrument": self.instrument, "method": self.method}
        if self.timeout_s is not None:
            d["timeout_s"] = self.timeout_s
        if self.retry:
            d["retry"] = self.retry
        if self.description:
            d["description"] = self.description
        if self.category:
            d["category"] = self.category
        # Lifecycle metadata (omit defaults so pre-lifecycle files stay clean).
        if self.maturity and self.maturity != "draft":
            d["maturity"] = self.maturity
        if self.version and self.version != 1:
            d["version"] = self.version
        if self.provenance:
            d["provenance"] = dict(self.provenance)
        if self.evidence:
            d["evidence"] = dict(self.evidence)
        # params last: written as a nested [tasks.<name>.params] sub-table.
        if self.params:
            d["params"] = dict(self.params)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> Task:
        """Rebuild a Task from a parsed TOML table.  Tolerant of missing keys."""
        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        timeout = data.get("timeout_s")
        provenance = data.get("provenance") or {}
        evidence = data.get("evidence") or {}
        return cls(
            name=name,
            instrument=str(data.get("instrument", "")),
            method=str(data.get("method", "")),
            params=dict(params),
            timeout_s=None if timeout is None else float(timeout),
            retry=int(data.get("retry", 0) or 0),
            description=str(data.get("description", "")),
            category=str(data.get("category", "")),
            maturity=str(data.get("maturity", "draft")),
            version=int(data.get("version", 1) or 1),
            provenance=dict(provenance) if isinstance(provenance, dict) else {},
            evidence=dict(evidence) if isinstance(evidence, dict) else {},
        )


class TaskCatalog:
    """In-memory dict of named :class:`Task` instances with TOML persistence."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task, *, strict: bool = False) -> None:
        """Insert or replace a task (keyed by name, so add doubles as update).

        An unsound task (:func:`validate_task`) is always logged.
        ``strict=True`` additionally refuses it — used by :meth:`load_toml`, so
        a malformed long hold cannot reach the rig from the catalog file, while
        interactive callers (the sandbox's "save step as task") still get their
        edit stored with a loud complaint rather than an exception mid-dialog.
        """
        problems = validate_task(task)
        if problems:
            logger.warning("task_invalid", task=task.name, problems=problems)
            if strict:
                raise TaskValidationError(f"task '{task.name}': {'; '.join(problems)}")
        self._tasks[task.name] = task

    def validate(self) -> dict[str, list[str]]:
        """Return ``{task_name: [problem, ...]}`` for every unsound task."""
        return {
            name: problems
            for name in self.list_names()
            if (problems := validate_task(self._tasks[name]))
        }

    def remove(self, name: str) -> None:
        del self._tasks[name]

    def get(self, name: str) -> Task:
        return self._tasks[name]

    def list_names(self) -> list[str]:
        return sorted(self._tasks.keys())

    def list_by_category(self) -> dict[str, list[str]]:
        """Return ``{category: [task_name, ...]}``, each list sorted by name.

        Uncategorised tasks are grouped under the empty-string key.
        """
        out: dict[str, list[str]] = {}
        for name in self.list_names():
            out.setdefault(self._tasks[name].category, []).append(name)
        return out

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, name: str) -> bool:
        return name in self._tasks

    def to_dict(self) -> dict[str, Any]:
        """Whole-catalog mapping: ``{"tasks": {name: {...}}}``."""
        return {"tasks": {name: self._tasks[name].to_dict() for name in self.list_names()}}

    def save_toml(self, path: Path) -> None:
        """Rewrite the whole catalog to ``path`` (binary mode, per ``tomli_w``)."""
        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def load_toml(cls, path: Path) -> TaskCatalog:
        """Load a catalog from ``path``; a missing file yields an EMPTY catalog.

        Mirrors :meth:`ChemicalCatalog.load_csv` — never raises on a missing
        file so a fresh install (no ``tasks.toml`` yet) degrades gracefully.

        Tasks failing :func:`validate_task` are **rejected**: logged at *error*
        and left out of the catalog, so a name that would abort mid-hold fails
        loudly at resolution rather than quietly on the rig.  One bad table does
        not cost the rest of the file.
        """
        cat = cls()
        if not path.exists():
            return cat
        with open(path, "rb") as f:
            data = tomllib.load(f)
        tasks = data.get("tasks", {})
        if isinstance(tasks, dict):
            for name, table in tasks.items():
                if not isinstance(table, dict):
                    continue
                try:
                    cat.add(Task.from_dict(name, table), strict=True)
                except TaskValidationError:
                    logger.error("task_rejected_from_catalog", task=name, path=str(path))
        return cat
