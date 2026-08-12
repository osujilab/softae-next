"""Run plan — the ordered phase sequence a deposition run executes.

A **run plan** lifts the implicit "flush → per-channel deposit + EIS → flush"
ordering of the deposition engine into an explicit, inspectable object so a run
can insert an **anneal** (cure) phase and choose *when* measurement happens —
and, critically, so the same description serves both **pointwise** and **batch**
runs:

* **pointwise** (single sample, or HT sequential): every phase is
  :attr:`PhaseScope.PER_SAMPLE`, so each channel is formulated, annealed, and
  measured before moving to the next — today's interleaved behaviour, plus an
  optional anneal;
* **batch** (q-BO across a board): formulation is per-sample but anneal and
  measurement are :attr:`PhaseScope.PER_BATCH`, so the engine casts **all**
  samples, then anneals the whole plate once, then measures **all** samples —
  "formulate-all → anneal-all → measure-all".

The plan is a pure, declarative object.  :func:`~softae.core.deposition_recipe.build_recipe_deposition_workflow`
consumes it (a ``None`` plan defaults to the legacy pointwise ordering, so
existing callers are unchanged); the engine decides how each phase's steps are
emitted from the phase's :class:`PhaseKind` and :class:`PhaseScope`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "PhaseKind",
    "PhaseScope",
    "RunPhase",
    "RunPlan",
    "DEFAULT_ANNEAL_TASK",
]

#: Catalog task used for an anneal phase when none is named.
DEFAULT_ANNEAL_TASK = "anneal_150C_5min"


class PhaseKind(Enum):
    """What a phase does.

    ``ARRHENIUS`` is reserved for a future temperature-sweep measurement phase
    and is intentionally not yet handled by the engine.
    """

    FORMULATE = "formulate"   # precondition + drop-cast (the recipe's phases)
    ANNEAL = "anneal"         # hold the plate/sample at temperature (cure)
    MEASURE = "measure"       # EIS measurement
    ARRHENIUS = "arrhenius"   # reserved (temperature-sweep EIS) — not implemented


class PhaseScope(Enum):
    """Whether a phase runs once per sample or once across the whole batch."""

    PER_SAMPLE = "per_sample"  # interleaved: done for each channel in turn
    PER_BATCH = "per_batch"    # a boundary: done for all channels together


@dataclass(frozen=True)
class RunPhase:
    """One phase in a run plan.

    ``anneal_task`` / ``anneal_params`` apply only to :attr:`PhaseKind.ANNEAL`:
    the catalog task to run (default :data:`DEFAULT_ANNEAL_TASK`) and optional
    per-run overrides (e.g. ``{"target_temp_C": 120, "hold_time_s": 600}``).
    """

    kind: PhaseKind
    scope: PhaseScope = PhaseScope.PER_SAMPLE
    anneal_task: str = DEFAULT_ANNEAL_TASK
    anneal_params: Mapping[str, Any] | None = None

    def label(self) -> str:
        """Short human-readable phase label (for :meth:`RunPlan.describe`)."""
        if self.kind is PhaseKind.ANNEAL:
            over = dict(self.anneal_params or {})
            temp = over.get("target_temp_C")
            hold = over.get("hold_time_s")
            if temp is not None or hold is not None:
                bits = []
                if temp is not None:
                    bits.append(f"{temp}°C")
                if hold is not None:
                    bits.append(f"{float(hold) / 60:g}min")
                detail = "/".join(bits)
            else:
                detail = self.anneal_task
            name = f"Anneal ({detail})"
        elif self.kind is PhaseKind.FORMULATE:
            name = "Formulate"
        elif self.kind is PhaseKind.MEASURE:
            name = "Measure EIS"
        else:
            name = self.kind.value.capitalize()
        scope = "per sample" if self.scope is PhaseScope.PER_SAMPLE else "per batch"
        return f"{name} [{scope}]"


@dataclass(frozen=True)
class RunPlan:
    """An ordered sequence of :class:`RunPhase` a deposition run executes."""

    phases: tuple[RunPhase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        formulate = [p for p in self.phases if p.kind is PhaseKind.FORMULATE]
        if not formulate:
            raise ValueError("a run plan must contain a FORMULATE phase")
        if any(p.scope is not PhaseScope.PER_SAMPLE for p in formulate):
            raise ValueError("FORMULATE must be per-sample (each sample is cast individually)")
        if any(p.kind is PhaseKind.ARRHENIUS for p in self.phases):
            raise ValueError("ARRHENIUS phase is reserved and not yet supported")

    # ── queries ───────────────────────────────────────────────────────────

    def has_kind(self, kind: PhaseKind) -> bool:
        return any(p.kind is kind for p in self.phases)

    @property
    def has_measure(self) -> bool:
        return self.has_kind(PhaseKind.MEASURE)

    @property
    def has_anneal(self) -> bool:
        return self.has_kind(PhaseKind.ANNEAL)

    @property
    def defers_measurement(self) -> bool:
        """True if any MEASURE phase runs per-batch (measurement after all casts)."""
        return any(
            p.kind is PhaseKind.MEASURE and p.scope is PhaseScope.PER_BATCH
            for p in self.phases
        )

    def segments(self) -> list[tuple[PhaseScope, list[RunPhase]]]:
        """Group phases into consecutive runs of the same scope, preserving order.

        The engine emits each ``PER_SAMPLE`` segment by looping channels (so a
        channel's phases stay adjacent) and each ``PER_BATCH`` segment as a
        boundary block across all channels.
        """
        out: list[tuple[PhaseScope, list[RunPhase]]] = []
        for phase in self.phases:
            if out and out[-1][0] is phase.scope:
                out[-1][1].append(phase)
            else:
                out.append((phase.scope, [phase]))
        return out

    def describe(self) -> str:
        """One-line ordered summary for display (GUI sequence preview)."""
        return "  →  ".join(p.label() for p in self.phases)

    # ── factories ─────────────────────────────────────────────────────────

    @classmethod
    def pointwise(
        cls,
        *,
        measure: bool = True,
        anneal: bool = False,
        anneal_task: str = DEFAULT_ANNEAL_TASK,
        anneal_params: Mapping[str, Any] | None = None,
    ) -> "RunPlan":
        """Everything per-sample: formulate → (anneal) → (measure), interleaved.

        With ``anneal=False`` and ``measure=True`` this is exactly the legacy
        deposition ordering (deposit then EIS, per channel).
        """
        phases: list[RunPhase] = [RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE)]
        if anneal:
            phases.append(RunPhase(
                PhaseKind.ANNEAL, PhaseScope.PER_SAMPLE,
                anneal_task=anneal_task, anneal_params=anneal_params,
            ))
        if measure:
            phases.append(RunPhase(PhaseKind.MEASURE, PhaseScope.PER_SAMPLE))
        return cls(tuple(phases))

    @classmethod
    def batch(
        cls,
        *,
        measure: bool = True,
        anneal: bool = False,
        anneal_task: str = DEFAULT_ANNEAL_TASK,
        anneal_params: Mapping[str, Any] | None = None,
    ) -> "RunPlan":
        """Formulate-all → anneal-all → measure-all: cast per-sample, the rest per-batch."""
        phases: list[RunPhase] = [RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE)]
        if anneal:
            phases.append(RunPhase(
                PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                anneal_task=anneal_task, anneal_params=anneal_params,
            ))
        if measure:
            phases.append(RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH))
        return cls(tuple(phases))
