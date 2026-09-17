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
from typing import TYPE_CHECKING, Any, Mapping

from softae.analysis.equilibration import (
    DEFAULT_RH_STABILITY_PCT,
    DEFAULT_SETTLE_MIN_CHANNELS,
    DEFAULT_SETTLE_N_ROUNDS,
    DEFAULT_SETTLE_TOL_REL,
    SETTLE_CRITERIA,
    SETTLE_CRITERION_DEVIATION,
    settle_tol_rel_refusal,
)

if TYPE_CHECKING:  # annotation-only: keeps the deposition engine's import path
    # free of the measurement and workflow-model layers.
    from softae.core.measurement_spec import MeasurementSpec
    from softae.core.phase_setpoints import PhaseSetpoints
    from softae.core.task_catalog import TaskCatalog

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

    ``criterion`` and ``rate_tol_dec_per_h`` are the fifth and sixth, and they
    travel together: which of the sibling gates in
    :class:`~softae.analysis.equilibration.SettleTracker` **routes**, and the band
    the rate one is measured against. They exist only in the structural
    ``[run_plan.phases.settle]`` spelling — the flat settle fields on
    :class:`~softae.core.autonomous_wiring.CampaignSpec` are the legacy operator
    spelling and do not grow, so a new capability is stated in one place rather
    than two that can disagree.

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
    #: Which of the sibling settle gates **routes** —
    #: :data:`~softae.analysis.equilibration.SETTLE_CRITERION_DEVIATION` (the
    #: default, and every verdict this rig has ever taken),
    #: ``SETTLE_CRITERION_RATE``, or ``SETTLE_CRITERION_BOTH``, which routes on
    #: deviation and reports the rate beside it. Imported rather than restated so
    #: the word this plan carries is the word the tracker accepts.
    criterion: str = SETTLE_CRITERION_DEVIATION
    #: The rate band, in **decades per hour** — the operator's unit, the one
    #: ``--settle-rate-tol-dec-per-h`` and ``H3_MAX_HOLD_DRIFT_DEC`` are written
    #: in. The tracker's own arithmetic is in ln-units and the conversion belongs
    #: at *its* boundary (:func:`~softae.analysis.equilibration.rate_tol_ln_per_hour`),
    #: not here: a plan holding ln-units would put the gate's unit in the
    #: operator's file. ``None`` means no band is configured, which is the honest
    #: state of a deviation-only plan.
    rate_tol_dec_per_h: float | None = None

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
        if (refusal := _criterion_refusal(
                self.criterion, self.rate_tol_dec_per_h)) is not None:
            raise ValueError(refusal)

    def label(self) -> str:
        """``'≤2h, ≥30min, every 2min'`` — the three durations, in one glance.

        The criterion joins them only when it is not the default, so every label
        an operator has already read back is byte-identical. The band is never
        rendered alone: on a deviation plan a bare ``≤0.05 dec/h`` would read as
        the gate that is running.
        """
        label = (f"≤{_minutes(self.max_hold_s)}, ≥{_minutes(self.min_hold_s)}, "
                 f"every {_minutes(self.round_period_s)}")
        if self.criterion == SETTLE_CRITERION_DEVIATION:
            return label
        return f"{label}, {self.criterion} ≤{self.rate_tol_dec_per_h:g} dec/h"


def _criterion_refusal(criterion: str, rate_tol_dec_per_h: float | None) -> str | None:
    """Why this criterion/band pair cannot run, or ``None``.

    A function beside :func:`settle_tol_rel_refusal` rather than three more
    branches inside ``__post_init__``, and separate from the tracker's own
    construction check because the two refuse at different moments: the tracker
    refuses when a run is already under way, this refuses when the file is read.

    **The middle refusal is the reason this pair is validated at all.** A rate
    criterion with no band computes no verdict whatsoever —
    ``SettleTracker._rate_verdict`` returns ``None`` rather than inventing a
    tolerance — so under ``rate`` the phase can never certify and burns to
    ``max_hold_s``, and under ``both`` the shadow half silently never runs while
    deviation routes exactly as it always did. Both failures wear a working run's
    clothes.
    """
    if str(criterion) not in SETTLE_CRITERIA:
        return f"criterion {criterion!r} is not one of {SETTLE_CRITERIA}"
    if criterion != SETTLE_CRITERION_DEVIATION and rate_tol_dec_per_h is None:
        return (f"criterion={criterion!r} needs a rate_tol_dec_per_h band; with "
                f"none, no rate verdict is ever computed — the phase would run "
                f"to max_hold_s under 'rate', and under 'both' the shadow "
                f"comparison would silently never happen. 0.05 dec/h is what "
                f"softae-eis-validate defaults to; 'deviation' is how a plan "
                f"says it wants no rate gate")
    if rate_tol_dec_per_h is not None and rate_tol_dec_per_h <= 0:
        return ("rate_tol_dec_per_h must be positive; a zero or negative band "
                "can never be satisfied — use None to mean no rate band")
    return None


def _minutes(seconds: float) -> str:
    """Duration in the largest unit that stays readable."""
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}min"
    return f"{seconds / 3600:g}h"


def _degrees(value: Any) -> str:
    """``85``, ``85.0`` and ``'85'`` all render ``'85°C'``; anything else as-is.

    ``anneal_params`` is a free ``Mapping[str, Any]`` and a catalogued task's
    ``params`` is no narrower, so the two sources of a cure temperature reach
    this with different types for the same number. Normalising here is what
    keeps ``85.0`` from the catalog and ``85`` from an override rendering as two
    different anneals; the non-numeric fallback prints rather than raises,
    because a label is not the place a malformed param is discovered.
    """
    try:
        return f"{float(value):g}°C"
    except (TypeError, ValueError):
        return f"{value}°C"


def _task_cure_temp_C(catalog: "TaskCatalog | None", task_name: str) -> Any | None:
    """The catalogued task's ``target_temp_C``, or ``None`` when nobody said.

    ``None`` covers three genuinely different situations — no catalog was
    supplied, the catalog does not hold this task, the task states no
    temperature — and they are deliberately spelled the same because the label
    says the same thing about all three: *this phase does not know the cure
    temperature, and here is the task name that does*. What it must never do is
    invent one, so there is no default anywhere on this path.

    Duck-typed (``in`` then ``get``) rather than imported: the catalog stays an
    annotation, so ``run_plan`` keeps its runtime import surface.
    """
    if catalog is None or task_name not in catalog:
        return None
    return getattr(catalog.get(task_name), "params", {}).get("target_temp_C")


@dataclass(frozen=True)
class RunPhase:
    """One phase in a run plan.

    ``anneal_task`` / ``anneal_params`` apply only to :attr:`PhaseKind.ANNEAL`:
    the catalog task to run (default :data:`DEFAULT_ANNEAL_TASK`) and optional
    per-run overrides (e.g. ``{"target_temp_C": 120, "hold_time_s": 600}``).

    ``hold_s`` is the **typed** per-run spelling of that task's ``hold_time_s``
    and is legal only on an ANNEAL phase. The catalog task remains the sole
    authority for the hardware command; this is the one duration a run plan may
    override, and saying it here *and* in ``anneal_params["hold_time_s"]`` is
    refused rather than resolved, because picking one silently is how a campaign
    holds for a duration nobody wrote down. The emitter writes it into the step
    **before** the ceiling is derived, so
    :func:`~softae.core.deposition_recipe.anneal_timeout_s` covers it.

    ``settle`` applies only to :attr:`PhaseKind.EQUILIBRATE`, where it is
    **required** — an equilibrate phase with no floor and no ceiling is a hold
    with no stopping rule at either end.

    ``conditions`` is the environment the phase is **commanded** to hold — see
    :class:`~softae.core.phase_setpoints.PhaseSetpoints` on why the type is named
    for setpoints while this field keeps the operator's word. It is legal on
    **any** kind: a cast has a casting environment, a cure has a cure
    environment, and an equilibrate hold has its own. ``None`` means the phase
    inherits whatever the previous one established.

    **On an ANNEAL phase it is the temperature the chamber is RESTORED to when
    the hold ends — not a second spelling of the cure.** ``anneal()`` reads the
    standing setpoint *after* the conditions block has written it and writes it
    back in a ``finally``, so the task wins the hold and ``conditions`` wins the
    resting state after it. Equal values are therefore the trap rather than the
    agreement: the chamber rests at the cure temperature, and a following phase
    with no conditions of its own reads a hot film. The deposition engine warns
    on exactly that case.

    ``measurement`` overrides the campaign's own :class:`MeasurementSpec` for
    this phase only, and is legal **solely on** :attr:`PhaseKind.MEASURE`. A
    denser preset on a FORMULATE phase would be silently ignored, so it is
    refused instead.

    Both are trailing and defaulted, so every existing positional construction
    of a ``RunPhase`` is unchanged.
    """

    kind: PhaseKind
    scope: PhaseScope = PhaseScope.PER_SAMPLE
    anneal_task: str = DEFAULT_ANNEAL_TASK
    anneal_params: Mapping[str, Any] | None = None
    settle: SettlePlan | None = None
    conditions: "PhaseSetpoints | None" = None
    measurement: "MeasurementSpec | None" = None
    hold_s: float | None = None

    def __post_init__(self) -> None:
        if self.measurement is not None and self.kind is not PhaseKind.MEASURE:
            raise ValueError(
                f"{self.kind.name} carries a measurement block; only MEASURE "
                f"acquires data, so a preset named on any other phase would be "
                f"accepted and never reach the instrument"
            )
        if self.hold_s is None:
            return
        if self.kind is not PhaseKind.ANNEAL:
            raise ValueError(
                f"{self.kind.name} carries hold_s; only ANNEAL holds at "
                f"temperature for a stated duration, so a hold named on any "
                f"other phase would be accepted and never reach the chamber"
            )
        if "hold_time_s" in (self.anneal_params or {}):
            raise ValueError(
                f"anneal hold is specified twice: hold_s={self.hold_s:g} and "
                f"anneal_params['hold_time_s']="
                f"{self.anneal_params['hold_time_s']} — say it once so the "
                f"hold has one authority"
            )
        if self.hold_s <= 0:
            raise ValueError(
                f"hold_s must be positive (got {self.hold_s:g}); a hold of no "
                f"time is not a cure, and 'no hold stated' is spelled None"
            )

    def label(self, catalog: "TaskCatalog | None" = None) -> str:
        """Short human-readable phase label (for :meth:`RunPlan.describe`).

        *catalog* is optional and used only by an ANNEAL phase, to name the cure
        temperature its task carries (see :meth:`_anneal_label`). It is passed
        in and never loaded here: a frozen dataclass that opened
        ``data/tasks.toml`` itself would make every label — and, through
        ``_run_plan_digest``, every resume fingerprint — depend on a gitignored,
        machine-local file.
        """
        if self.kind is PhaseKind.EQUILIBRATE:
            name = f"Equilibrate ({self.settle.label()})" if self.settle else "Equilibrate"
        elif self.kind is PhaseKind.ANNEAL:
            name = self._anneal_label(catalog)
        elif self.kind is PhaseKind.FORMULATE:
            name = "Formulate"
        elif self.kind is PhaseKind.MEASURE:
            name = "Measure EIS"
        else:
            name = self.kind.value.capitalize()
        if self.measurement is not None:
            name = f"{name} ({self.measurement.preset})"
        if self.conditions is not None:
            name = f"{name} @ {self.conditions.label()}"
        scope = "per sample" if self.scope is PhaseScope.PER_SAMPLE else "per batch"
        return f"{name} [{scope}]"

    def _anneal_label(self, catalog: "TaskCatalog | None" = None) -> str:
        """``'Anneal (anneal_85C_8h: 85°C/8h) → rests at 25 °C'`` — who, what, then after.

        **The rule: the task name is always present, whatever else is.** It used
        to drop out the moment any typed field appeared, so the commonest real
        phase of all — a bare ``hold_s`` on a catalogued cure — rendered
        ``'Anneal (8h) → rests at 25 °C'``, in which the only temperature shown
        is the one the chamber returns to *after* the cure and the cure's own
        temperature appears nowhere. A label that names a rest state and not the
        hold invites reading the rest state as the hold.

        So the shape is ``Anneal (<task>: <temp>/<hold>)``, degrading left to
        right and never inventing a number:

        =========================== ==============================================
        What is known               Renders
        =========================== ==============================================
        task only                   ``Anneal (anneal_85C_8h)``
        task + hold                 ``Anneal (anneal_85C_8h: 8h)``
        task + cure temp + hold     ``Anneal (anneal_85C_8h: 85°C/8h)``
        =========================== ==============================================

        The cure temperature comes from ``anneal_params["target_temp_C"]`` when
        the run overrides it, otherwise from *catalog* when one is supplied and
        holds the task — the same precedence the emitter applies to build the
        step, so the label cannot name a temperature the rig will not use. With
        no catalog and no override the row above degrades to the task name
        alone, which is honest: the task is where the temperature is written
        down. See :func:`_task_cure_temp_C` on why an absent catalog, an absent
        task and a silent task all render alike.

        The ``→ rests at`` suffix is unchanged and keeps its own role, because
        ``conditions`` on an ANNEAL phase is the RESTORE target rather than a
        second spelling of the cure — the distinction the driver already makes
        and the reason both numbers are worth one line.
        """
        over = dict(self.anneal_params or {})
        temp = over.get("target_temp_C")
        if temp is None:
            temp = _task_cure_temp_C(catalog, self.anneal_task)
        hold = self.hold_s if self.hold_s is not None else over.get("hold_time_s")
        bits = []
        if temp is not None:
            bits.append(_degrees(temp))
        if hold is not None:
            bits.append(_minutes(float(hold)))
        name = f"Anneal ({self.anneal_task}"
        name = f"{name}: {'/'.join(bits)})" if bits else f"{name})"
        # `conditions` is the RESTORE target, not the cure, so the label says
        # which one it is — visible at `check` time rather than in a docstring.
        rest = self.conditions.temp_setpoint_C if self.conditions else None
        return name if rest is None else f"{name} → rests at {rest:g} °C"


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

    def describe(self, catalog: "TaskCatalog | None" = None) -> str:
        """One-line ordered summary for display (GUI sequence preview).

        *catalog* is forwarded to :meth:`RunPhase.label` so an ANNEAL phase can
        name its task's cure temperature; omitted, every phase describes itself
        from what it carries. Callers that already hold a catalog — anything
        past ``TaskCatalog.load_toml`` — get a strictly more informative line by
        passing it.
        """
        return "  →  ".join(p.label(catalog) for p in self.phases)

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
        conditions: "Mapping[PhaseKind, PhaseSetpoints] | None" = None,
        measurement: "MeasurementSpec | None" = None,
        hold_s: float | None = None,
    ) -> "RunPlan":
        """Everything per-sample: formulate → (anneal) → (equilibrate) → (measure).

        With ``anneal=False``, ``settle=None`` and ``measure=True`` this is
        exactly the legacy deposition ordering (deposit then EIS, per channel).
        """
        return cls._assemble(PhaseScope.PER_SAMPLE, measure=measure, anneal=anneal,
                             anneal_task=anneal_task, anneal_params=anneal_params,
                             settle=settle, conditions=conditions,
                             measurement=measurement, hold_s=hold_s)

    @classmethod
    def batch(
        cls,
        *,
        measure: bool = True,
        anneal: bool = False,
        anneal_task: str = DEFAULT_ANNEAL_TASK,
        anneal_params: Mapping[str, Any] | None = None,
        settle: SettlePlan | None = None,
        conditions: "Mapping[PhaseKind, PhaseSetpoints] | None" = None,
        measurement: "MeasurementSpec | None" = None,
        hold_s: float | None = None,
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
                             settle=settle, conditions=conditions,
                             measurement=measurement, hold_s=hold_s)

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
        conditions: "Mapping[PhaseKind, PhaseSetpoints] | None" = None,
        measurement: "MeasurementSpec | None" = None,
        hold_s: float | None = None,
    ) -> "RunPlan":
        """Cast per-sample, then the optional tail at *scope* — the shared spine.

        Order is cure → equilibrate → measure: the anneal carries the bulk of the
        hold, the equilibrate phase decides when the *tail* of it has stopped
        moving, and only then is the reading worth recording.

        *conditions* is keyed **by phase kind** rather than spread over four
        ``casting_conditions=`` / ``anneal_conditions=`` keywords: the factories
        build at most one phase of each kind, so the mapping is total, and a key
        for a phase the arguments did not create is a silent no-op rather than a
        combinatorial signature. A plan needing two phases of one kind is built
        from ``RunPhase`` objects directly, which is what the TOML codec does.
        """
        if measurement is not None and not measure:
            raise ValueError(
                "measurement= names a preset for a MEASURE phase, but measure=False "
                "builds no MEASURE phase; the override would be silently dropped"
            )
        by_kind: Mapping[PhaseKind, "PhaseSetpoints"] = conditions or {}
        phases: list[RunPhase] = [RunPhase(
            PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
            conditions=by_kind.get(PhaseKind.FORMULATE))]
        if anneal:
            phases.append(RunPhase(PhaseKind.ANNEAL, scope,
                                   anneal_task=anneal_task,
                                   anneal_params=anneal_params,
                                   conditions=by_kind.get(PhaseKind.ANNEAL),
                                   hold_s=hold_s))
        elif hold_s is not None:
            raise ValueError(
                "hold_s states an anneal hold, but anneal=False builds no "
                "ANNEAL phase; the hold would be silently dropped")
        if settle is not None:
            phases.append(RunPhase(PhaseKind.EQUILIBRATE, scope, settle=settle,
                                   conditions=by_kind.get(PhaseKind.EQUILIBRATE)))
        if measure:
            phases.append(RunPhase(PhaseKind.MEASURE, scope,
                                   conditions=by_kind.get(PhaseKind.MEASURE),
                                   measurement=measurement))
        return cls(tuple(phases))
