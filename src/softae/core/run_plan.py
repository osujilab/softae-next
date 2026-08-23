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

:attr:`PhaseKind.EQUILIBRATE` is the exception, and deliberately so. It cannot
be a list of steps: it terminates on **evidence** — measure, judge, decide —
which is a loop, not a sequence, so the deposition engine emits nothing for it
and :func:`softae.core.autonomous_wiring.drive_settle_phase` drives it on the
campaign path instead.

.. warning::
   That means an EQUILIBRATE phase in a plan handed straight to the deposition
   engine (the HT tab's path) is currently a **no-op**: the plan describes a
   hold nothing performs. Only the campaign path drives it today. Teaching the
   engine to call the driver is a separate change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from softae.analysis.equilibration import (
    DEFAULT_RH_STABILITY_PCT,
    DEFAULT_SETTLE_MIN_CHANNELS,
    DEFAULT_SETTLE_N_ROUNDS,
    DEFAULT_SETTLE_TOL_REL,
    settle_tol_rel_refusal,
)

__all__ = [
    "PhaseKind",
    "PhaseScope",
    "RunPhase",
    "RunPlan",
    "SettlePlan",
    "DEFAULT_ANNEAL_TASK",
]

#: Catalog task used for an anneal phase when none is named.
DEFAULT_ANNEAL_TASK = "anneal_150C_5min"


class PhaseKind(Enum):
    """What a phase does.

    ``ANNEAL`` and ``EQUILIBRATE`` are **not** the same step wearing two names,
    and a run may legitimately carry both — cure, then equilibrate:

    ==================  ==========================  ===========================
    ..                  ANNEAL                      EQUILIBRATE
    ==================  ==========================  ===========================
    purpose             hold at temperature to      hold until the *measurement*
                        **cure**                    stops moving
    terminates on       elapsed time (a catalog     evidence, or a ceiling
                        task)
    measures during     no                          yes, every ``round_period_s``
    outcome             done / failed               settled / ceiling /
                                                    not-evaluable
    ==================  ==========================  ===========================

    ``ARRHENIUS`` is reserved for a future temperature-sweep measurement phase
    and is intentionally not yet handled by the engine.
    """

    FORMULATE = "formulate"   # precondition + drop-cast (the recipe's phases)
    ANNEAL = "anneal"         # hold the plate/sample at temperature (cure)
    EQUILIBRATE = "equilibrate"  # hold until the measurement stops moving
    MEASURE = "measure"       # EIS measurement
    ARRHENIUS = "arrhenius"   # reserved (temperature-sweep EIS) — not implemented


class PhaseScope(Enum):
    """Whether a phase runs once per sample or once across the whole batch."""

    PER_SAMPLE = "per_sample"  # interleaved: done for each channel in turn
    PER_BATCH = "per_batch"    # a boundary: done for all channels together


@dataclass(frozen=True)
class SettlePlan:
    """When an :attr:`PhaseKind.EQUILIBRATE` phase may stop, and when it must.

    The **time** half of the settle decision. Its evidence half lives in
    :class:`~softae.analysis.equilibration.SettleTracker`, whose docstring says
    why the split exists: *"Deliberately holds no clock and no store: the caller
    owns the floor and the ceiling, because those are time and this is
    evidence."* This is that caller's half, written down.

    The three durations are **required** and have no defaults, because none of
    them can be invented safely:

    * ``min_hold_s`` is the cure and belongs to the recipe — a wrong one measures
      a wet film;
    * ``max_hold_s`` is the ceiling that guarantees the phase terminates at all;
    * ``round_period_s`` is instrument time spent per sample.

    The four settle parameters default to the values measured on
    ``20260811T023757Z_equilibration_characterization`` and are imported from
    :mod:`softae.analysis.equilibration` rather than restated, so a criterion
    retuned there moves the phase defaults with it. Notably
    ``settle_tol_rel = 0.10``: the measured noise floor on that run was 5.98 %,
    so a 2 % band is unsatisfiable by any hold length.

    The fourth, ``rh_stability_pct``, is the only one that judges the *room*
    rather than the sample. It belongs here and not in ``[safety]`` because it is
    a spread over **this window** — a tolerance coupled to ``settle_n_rounds``
    and ``round_period_s``, which a key in another file would be retuned
    independently of — and because it parks nothing.
    """

    round_period_s: float
    min_hold_s: float
    max_hold_s: float
    settle_tol_rel: float = DEFAULT_SETTLE_TOL_REL
    settle_n_rounds: int = DEFAULT_SETTLE_N_ROUNDS
    settle_min_channels: int = DEFAULT_SETTLE_MIN_CHANNELS
    #: How far the chamber's %RH may move across the judged window and still let
    #: the phase certify ``settled``. A **stability** tolerance, not a tracking
    #: one: it is compared against the spread of the PV about itself and no
    #: setpoint is read, which is why it lives here beside the window it
    #: describes rather than in ``[safety]`` beside the ``rh_deviation_*``
    #: tracking bands. It parks nothing — the streak limit that does
    #: (``rh_ceiling_park_after_trials``) is a different quantity in a different
    #: file. ``None`` switches the gate off.
    #:
    #: **On by default.** The failure mode of ON is *"held longer, recorded
    #: ceiling"*; the failure mode of OFF is *"measured under moving humidity"*.
    #: The gate can only ever make settling harder, never earlier, so it cannot
    #: produce the early-measurement hazard that made settle itself opt-in.
    rh_stability_pct: float | None = DEFAULT_RH_STABILITY_PCT

    def __post_init__(self) -> None:
        if self.round_period_s < 0 or self.min_hold_s < 0:
            raise ValueError("round_period_s and min_hold_s must be non-negative")
        if self.max_hold_s <= 0:
            raise ValueError("max_hold_s must be positive — it is the ceiling that "
                             "guarantees the phase terminates")
        if self.max_hold_s < self.min_hold_s:
            raise ValueError(
                f"max_hold_s ({self.max_hold_s:g}s) is below min_hold_s "
                f"({self.min_hold_s:g}s); the ceiling would fire before the floor"
            )
        # The same rule, and now literally the same code, as the one
        # `eis_validate_hold.validate_plan` applies to its own `--settle-tol-rel`.
        # Behaviour and message are unchanged; only the restatement is gone.
        if (refusal := settle_tol_rel_refusal(self.settle_tol_rel)) is not None:
            raise ValueError(refusal)
        if self.rh_stability_pct is not None and self.rh_stability_pct <= 0:
            raise ValueError("rh_stability_pct must be positive; a zero band can "
                             "never be satisfied — use None to switch the RH "
                             "stability gate off")

    def label(self) -> str:
        """``'≤2h, ≥30min, every 2min'`` — the three durations, in one glance."""
        return (f"≤{_minutes(self.max_hold_s)}, ≥{_minutes(self.min_hold_s)}, "
                f"every {_minutes(self.round_period_s)}")


def _minutes(seconds: float) -> str:
    """Duration in the largest unit that stays readable."""
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}min"
    return f"{seconds / 3600:g}h"


@dataclass(frozen=True)
class RunPhase:
    """One phase in a run plan.

    ``anneal_task`` / ``anneal_params`` apply only to :attr:`PhaseKind.ANNEAL`:
    the catalog task to run (default :data:`DEFAULT_ANNEAL_TASK`) and optional
    per-run overrides (e.g. ``{"target_temp_C": 120, "hold_time_s": 600}``).

    ``settle`` applies only to :attr:`PhaseKind.EQUILIBRATE`, where it is
    **required** — an equilibrate phase with no floor and no ceiling is a hold
    with no stopping rule at either end.
    """

    kind: PhaseKind
    scope: PhaseScope = PhaseScope.PER_SAMPLE
    anneal_task: str = DEFAULT_ANNEAL_TASK
    anneal_params: Mapping[str, Any] | None = None
    settle: SettlePlan | None = None

    def label(self) -> str:
        """Short human-readable phase label (for :meth:`RunPlan.describe`)."""
        if self.kind is PhaseKind.EQUILIBRATE:
            name = f"Equilibrate ({self.settle.label()})" if self.settle else "Equilibrate"
        elif self.kind is PhaseKind.ANNEAL:
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
        for phase in self.phases:
            if phase.kind is PhaseKind.EQUILIBRATE and phase.settle is None:
                raise ValueError(
                    "EQUILIBRATE requires a SettlePlan — min_hold_s is the cure "
                    "time and belongs to the recipe, so there is no safe default"
                )
            if phase.kind is not PhaseKind.EQUILIBRATE and phase.settle is not None:
                raise ValueError(
                    f"{phase.kind.name} carries a SettlePlan; only EQUILIBRATE "
                    f"terminates on evidence"
                )

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
    def has_equilibrate(self) -> bool:
        return self.has_kind(PhaseKind.EQUILIBRATE)

    def equilibrate_phases(self) -> list[RunPhase]:
        """Every EQUILIBRATE phase, in plan order (each carries its own plan)."""
        return [p for p in self.phases if p.kind is PhaseKind.EQUILIBRATE]

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
        settle: SettlePlan | None = None,
    ) -> "RunPlan":
        """Everything per-sample: formulate → (anneal) → (equilibrate) → (measure).

        With ``anneal=False``, ``settle=None`` and ``measure=True`` this is
        exactly the legacy deposition ordering (deposit then EIS, per channel).
        """
        return cls._assemble(PhaseScope.PER_SAMPLE, measure=measure, anneal=anneal,
                             anneal_task=anneal_task, anneal_params=anneal_params,
                             settle=settle)

    @classmethod
    def batch(
        cls,
        *,
        measure: bool = True,
        anneal: bool = False,
        anneal_task: str = DEFAULT_ANNEAL_TASK,
        anneal_params: Mapping[str, Any] | None = None,
        settle: SettlePlan | None = None,
    ) -> "RunPlan":
        """Formulate-all → anneal-all → measure-all: cast per-sample, the rest per-batch.

        An equilibrate phase lands per-batch too, and that is the case it was
        built for: :func:`~softae.analysis.equilibration.settle_check` judges a
        round **across channels** and refuses to settle on fewer than
        ``settle_min_channels`` of them, which is exactly the shape a q-channel
        batch round already produces.
        """
        return cls._assemble(PhaseScope.PER_BATCH, measure=measure, anneal=anneal,
                             anneal_task=anneal_task, anneal_params=anneal_params,
                             settle=settle)

    @classmethod
    def _assemble(
        cls,
        scope: PhaseScope,
        *,
        measure: bool,
        anneal: bool,
        anneal_task: str,
        anneal_params: Mapping[str, Any] | None,
        settle: SettlePlan | None,
    ) -> "RunPlan":
        """Cast per-sample, then the optional tail at *scope* — the shared spine.

        Order is cure → equilibrate → measure: the anneal carries the bulk of the
        hold, the equilibrate phase decides when the *tail* of it has stopped
        moving, and only then is the reading worth recording.
        """
        phases: list[RunPhase] = [RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE)]
        if anneal:
            phases.append(RunPhase(PhaseKind.ANNEAL, scope,
                                   anneal_task=anneal_task,
                                   anneal_params=anneal_params))
        if settle is not None:
            phases.append(RunPhase(PhaseKind.EQUILIBRATE, scope, settle=settle))
        if measure:
            phases.append(RunPhase(PhaseKind.MEASURE, scope))
        return cls(tuple(phases))
