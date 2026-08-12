"""Workflow data model — dataclasses that describe an experiment as data.

A :class:`Workflow` is a sequence of :class:`WorkflowStep` objects
organised into three phases:

* **setup** — one-time preparation (flush lines, ramp temperature, …)
* **loop** — repeated per-channel or per-iteration steps
* **teardown** — cleanup regardless of success or failure

The :class:`WorkflowParser` produces these objects from YAML / JSON;
the :class:`WorkflowExecutor` consumes them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    """Single atomic operation in a workflow.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"preconditionFlush"``).
    instrument : str
        Key in :class:`~softae.server.manager.InstrumentManager`.
    method : str
        Method name on the instrument driver.
    params : dict
        Keyword arguments forwarded to the method.
    depends_on : list[str]
        Step names that **must** complete before this one starts.
    timeout_s : float | None
        Maximum wall-clock seconds for this step. ``None`` = no limit.
    retry : int
        Number of retry attempts on transient failure (0 = no retry).
    tags : dict[str, str]
        Arbitrary metadata (e.g. ``{"channel": "5"}``).
    """

    name: str
    instrument: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    timeout_s: float | None = None
    retry: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    def with_params(self, **overrides: Any) -> WorkflowStep:
        """Return a copy with updated *params* (immutable-friendly)."""
        merged = {**self.params, **overrides}
        return WorkflowStep(
            name=self.name,
            instrument=self.instrument,
            method=self.method,
            params=merged,
            depends_on=list(self.depends_on),
            timeout_s=self.timeout_s,
            retry=self.retry,
            tags=dict(self.tags),
        )

    def with_timeout(self, timeout_s: float | None) -> WorkflowStep:
        """Return a copy with a different execution ceiling (immutable-friendly).

        Needed where a step's *duration* is driven by its own parameters (a long
        anneal hold) rather than by a constant declared on the catalogued task —
        see ``deposition_recipe.anneal_timeout_s``.
        """
        return WorkflowStep(
            name=self.name,
            instrument=self.instrument,
            method=self.method,
            params=dict(self.params),
            depends_on=list(self.depends_on),
            timeout_s=timeout_s,
            retry=self.retry,
            tags=dict(self.tags),
        )

    def with_tags(self, **extra: str) -> WorkflowStep:
        """Return a copy with additional *tags*."""
        merged = {**self.tags, **extra}
        return WorkflowStep(
            name=self.name,
            instrument=self.instrument,
            method=self.method,
            params=dict(self.params),
            depends_on=list(self.depends_on),
            timeout_s=self.timeout_s,
            retry=self.retry,
            tags=merged,
        )


@dataclass
class Workflow:
    """Complete experiment definition.

    Attributes
    ----------
    name : str
        Short identifier (used as log prefix and file stem).
    description : str
        Human-readable summary.
    variables : dict
        Global variables available for ``$var`` interpolation in step params
        (expanded by :class:`WorkflowParser`).
    setup : list[WorkflowStep]
        Steps executed once at the start.
    loop_steps : list[WorkflowStep]
        Template steps replicated for each channel / iteration.
    teardown : list[WorkflowStep]
        Steps executed once at the end (even after abort — best effort).
    iterate_over : str | None
        What the loop iterates over (e.g. ``"channels"``).
    iterations : int
        Number of loop iterations (inferred from *variables* by the parser).
    metadata : dict
        Arbitrary key/value pairs stored in experiment logs.
    """

    name: str
    description: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    setup: list[WorkflowStep] = field(default_factory=list)
    loop_steps: list[WorkflowStep] = field(default_factory=list)
    teardown: list[WorkflowStep] = field(default_factory=list)
    iterate_over: str | None = None
    iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_steps(self) -> list[WorkflowStep]:
        """Flatten into an ordered list of steps ready for execution.

        * Setup steps come first.
        * Loop steps are replicated ``self.iterations`` times, with each
          copy tagged ``{"iteration": "<n>"}``.
        * Teardown steps come last.

        Returns
        -------
        list[WorkflowStep]
        """
        resolved: list[WorkflowStep] = list(self.setup)

        for i in range(self.iterations):
            for step in self.loop_steps:
                expanded = step.with_tags(iteration=str(i))
                # Give each iteration a unique step name to avoid dependency clashes
                expanded = WorkflowStep(
                    name=f"{step.name}__iter{i}",
                    instrument=expanded.instrument,
                    method=expanded.method,
                    params=dict(expanded.params),
                    depends_on=list(expanded.depends_on),
                    timeout_s=expanded.timeout_s,
                    retry=expanded.retry,
                    tags=dict(expanded.tags),
                )
                resolved.append(expanded)

        resolved.extend(self.teardown)
        return resolved

    @property
    def total_steps(self) -> int:
        """Total number of steps after loop expansion."""
        return len(self.setup) + len(self.loop_steps) * self.iterations + len(self.teardown)
