"""Recipe-driven deposition — a slot-annotated engine for per-channel casting.

The foundation of the recipe-driven HT deposition path (replacing the role-map
mechanism over time).  A **deposition recipe** is a catalogued deposit *method*
(e.g. ``single_drop_simul``) plus a declaration of its **typed injection slots**
— the param keys that receive per-channel runtime values:

* ``electrode`` → the deposit step's ``x``/``y`` (from PCB geometry per channel);
* ``volumes``   → the per-pump dispense volumes (per-channel formulation);
* ``eis_channel`` → the EIS step's channel (pico routing per channel).

:func:`build_slotted_deposition_workflow` iterates the selected channels and
injects each slot's value, producing the per-channel Workflow — the same shape
the HT tab's ``_generate_workflow`` builds by hand, but driven by a recipe's
declared slots rather than a fixed role vocabulary.

Scope: electrode + per-channel volumes + optional EIS.  Piezo events and
liquid-handling correction remain in the legacy path until this engine is
hardware-validated and the HT tab cuts over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import structlog

from softae.config.loader import pico_for_channel
from softae.core.deposition_steps import (
    DEFAULT_DISP_RATE_UL_MIN,
    DEFAULT_FLUSH_RATE_UL_MIN,
    DEFAULT_FLUSH_VOL_UL,
    deposition_positions,
)
from softae.core.dropcast_plan import build_dropcast_plan
from softae.core.geometry import electrode_xy_for_channel
from softae.core.liquid_handling import DeadVolumeCorrection
from softae.core.run_plan import PhaseKind, PhaseScope, RunPhase, RunPlan
from softae.core.task_catalog import Task, TaskCatalog
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)


#: Slack added on top of an anneal's hold time to cover the ramp to target, the
#: ``wait(within=tolerance)`` settle, and the setpoint restore on the way out.
ANNEAL_RAMP_ALLOWANCE_S = 900.0
#: Proportional margin on the hold itself (slow ramps stretch long holds).
ANNEAL_TIMEOUT_MARGIN = 1.25


def anneal_timeout_s(params: dict[str, Any], declared: float | None) -> float | None:
    """Ceiling for an anneal step, derived from the hold it was actually asked for.

    A catalogued anneal task carries a hand-set ``timeout_s`` (600 s for the
    built-in 5-minute anneal), but a campaign may override ``hold_time_s`` via
    :attr:`RunPlan.anneal_params` **without touching that constant**. Soft-material
    anneals run for hours, so the declared ceiling silently becomes far shorter
    than the hold — the executor would abort the step partway through.

    That failure is especially insidious now that campaigns run with graceful
    recovery enabled: the aborted anneal is absorbed as a channel replay/skip and
    the campaign continues, so the samples are *wrongly annealed* rather than
    obviously missing. Deriving the ceiling from the parameters removes the
    chance for the two to disagree.

    Returns the larger of the declared ceiling and the derived floor, so an
    explicitly generous task timeout is never reduced.
    """
    try:
        hold = float(params.get("hold_time_s") or 0.0)
    except (TypeError, ValueError):
        return declared
    if hold <= 0:
        return declared
    floor = hold * ANNEAL_TIMEOUT_MARGIN + ANNEAL_RAMP_ALLOWANCE_S
    return max(float(declared or 0.0), floor)


@dataclass
class DepositionSlots:
    """Names the deposit method's param keys that receive per-channel injection."""

    electrode_x: str = "x"
    electrode_y: str = "y"
    volumes: str = "vols"       # a list param (per-pump)
    eis_channel: str = "chan"   # on the EIS step


def build_slotted_deposition_workflow(
    deposit_task: Task,
    channels: Sequence[int],
    formulation_by_channel: dict[int, Sequence[float]],
    *,
    pcb: dict,
    origin_xy: tuple[float, float] | None = None,
    slots: DepositionSlots | None = None,
    measure_eis: bool = True,
    startup_task: Task | None = None,
    final_flush_task: Task | None = None,
    name: str = "deposition",
) -> Workflow:
    """Build a per-channel deposition Workflow by injecting slot values.

    Parameters
    ----------
    deposit_task : Task
        The catalogued deposit **method** (its fixed params supply everything the
        slots don't override).
    channels : sequence of int
        1-based electrodes to deposit on, in order (one sample per channel).
    formulation_by_channel : dict
        ``{channel: [per-pump volumes]}`` — the injected ``volumes`` slot value.
    pcb, origin_xy :
        PCB layout and the electrode-1 origin (defaults to the calibrated
        ``dep1`` position) used to resolve each channel's electrode ``(x, y)``.
    slots :
        Which param keys receive injection (see :class:`DepositionSlots`).
    measure_eis :
        Append a per-channel EIS step (routed to the correct pico).
    startup_task, final_flush_task :
        Optional catalogued setup/teardown methods; literals used when omitted.
    """
    slots = slots or DepositionSlots()
    channels = list(channels)
    if origin_xy is None:
        origin_xy = deposition_positions().origin

    setup: list[WorkflowStep] = []
    if startup_task is not None:
        setup.append(startup_task.to_step("startup_flush"))

    electrode_xy: dict[str, list[float]] = {}
    for ch in channels:
        ex, ey = electrode_xy_for_channel(pcb, ch, origin_x=origin_xy[0], origin_y=origin_xy[1])
        electrode_xy[str(ch)] = [ex, ey]
        vols = list(formulation_by_channel.get(ch, []))
        step = deposit_task.to_step(f"deposit_ch{ch}").with_params(**{
            slots.electrode_x: ex,
            slots.electrode_y: ey,
            slots.volumes: vols,
        }).with_tags(slot="electrode+volumes", channel=str(ch))
        setup.append(step)

        if measure_eis:
            setup.append(
                WorkflowStep(
                    name=f"measure_eis_ch{ch}",
                    instrument=pico_for_channel(ch),
                    method="sendscript_getdata",
                    params={slots.eis_channel: ch},
                    timeout_s=600,
                    retry=1,
                    tags={"slot": "eis_channel", "channel": str(ch)},
                )
            )

    teardown: list[WorkflowStep] = []
    if final_flush_task is not None:
        teardown.append(final_flush_task.to_step("final_flush"))

    logger.info("slotted_deposition_built", channels=channels, measure_eis=measure_eis)
    return Workflow(
        name=name,
        description=f"Recipe-driven deposition on channels {','.join(map(str, channels))}",
        setup=setup,
        teardown=teardown,
        iterations=1,
        metadata={"source": "deposition_recipe", "channels": list(channels),
                  "electrode_xy": electrode_xy},
    )


# ── Deposition recipes — one engine, many per-channel phase sequences ─────────
#
# A *deposition recipe* is the unit the HT/AE deposition engine runs: an ordered
# list of per-channel **phases** (each a catalogued method + how it receives the
# per-channel runtime values), plus the campaign-level startup/teardown flushes.
# ``single_drop`` is one phase; ``two_phase`` adds a precondition-flush phase.
# The engine (:func:`build_recipe_deposition_workflow`) is recipe-agnostic — the
# recipe declares the structure, the engine injects the values.
#
# Code-defined for now (hybrid: TOML authoring is a planned follow-up); see
# ``docs/TWO_PHASE_CUTOVER.md``.


@dataclass(frozen=True)
class DepositionPhase:
    """One per-channel step in a deposition recipe.

    ``rate`` selects how the pump rate is set for this phase:
    ``"flat_dispense"`` (the scalar dispense rate on every pump),
    ``"split_dispense"`` / ``"split_flush"`` (per-pump rates split proportionally
    to volume from the total dispense / flush rate — see
    :func:`softae.core.dropcast_plan.split_rate`), or ``"none"``.
    """

    key: str                                 # semantic: "deposit" | "precondition"
    method: str                              # catalogued task / driver method name
    name_prefix: str                         # step name → f"{prefix}_ch{ch}"
    inject_electrode: bool = False           # inject x/y from PCB geometry
    inject_volumes_as: str | None = None     # param that receives per-pump volumes
    rate: str = "none"                       # none|flat_dispense|split_dispense|split_flush
    rate_param: str | None = None            # param that receives the rate(s)
    pass_flush_factor: bool = False          # inject flush_factor
    derive_settle: bool = False              # set elution_wait_s from the plan
    zero_deadvols: bool = False              # deadvols = [0]*n (correction already folded in)
    is_deposit: bool = False                 # the electrode-deposit phase (EIS follows it)


@dataclass(frozen=True)
class DepositionRecipe:
    """An ordered per-channel phase sequence plus startup/teardown flushes."""

    name: str
    label: str
    phases: tuple[DepositionPhase, ...]
    startup_method: str = "startup_flush_full"
    final_method: str = "final_flush"
    description: str = ""

    def method_deps(self) -> list[str]:
        """Catalogued methods this recipe runs (for maturity roll-up), sorted."""
        deps = {self.startup_method, self.final_method}
        deps.update(p.method for p in self.phases)
        return sorted(deps)

    def deposit_phase(self) -> DepositionPhase | None:
        return next((p for p in self.phases if p.is_deposit), None)


@dataclass(frozen=True)
class PiezoPlan:
    """Optional piezo actuation around elution events.

    When :attr:`enabled`, the engine applies an optional setup event-profile step
    and returns the piezo to standby in teardown.  :attr:`elution_scope` selects
    *which* events the piezo channel is enabled around:

    * ``"deposit"`` (default): only each channel's deposit phase — the piezo is
      enabled before the deposit and disabled after it (and after that channel's
      EIS).  Step names ``piezo_on_ch{n}`` / ``piezo_off_ch{n}``.
    * ``"all_elution"``: **every** elution event — the startup flush, each
      channel's precondition and deposit phases, and the final flush — is
      individually bracketed with the piezo on/off (off before the non-elution
      EIS/measurement).  Step names ``piezo_on_{step}`` / ``piezo_off_{step}``.

    The step methods come from the named catalog tasks; a missing task is silently
    skipped.  A cross-cutting option (both ``single_drop`` and ``two_phase``
    support it), driven by ``[piezo]`` config in the HT tab.
    """

    enabled: bool = False
    on_task: str = "piezo_channel_a_on"
    off_task: str = "piezo_channel_a_off"
    standby_task: str = "piezo_standby"
    event_task: str | None = None          # setup event-profile step (None = skip)
    event_params: dict[str, Any] | None = None
    elution_scope: str = "deposit"         # "deposit" | "all_elution"


_SINGLE_DROP = DepositionRecipe(
    name="single_drop",
    label="Single-drop",
    description="One simultaneous drop-cast per channel at a flat dispense rate.",
    phases=(
        DepositionPhase(
            key="deposit", method="single_drop_simul", name_prefix="deposit",
            inject_electrode=True, inject_volumes_as="vols",
            rate="flat_dispense", rate_param="disp_rate",
            zero_deadvols=True, is_deposit=True,
        ),
    ),
)

_TWO_PHASE = DepositionRecipe(
    name="two_phase",
    label="Two-phase (precondition + cast)",
    description=(
        "Per channel: precondition flush (preload the next formulation) then a "
        "drop-cast, with per-pump rates split so all components extrude for the "
        "same duration."
    ),
    phases=(
        DepositionPhase(
            key="precondition", method="precondition_flush", name_prefix="precondition",
            inject_volumes_as="vol_list", rate="split_flush", rate_param="rate_list",
            pass_flush_factor=True,
        ),
        DepositionPhase(
            key="deposit", method="single_drop_simul", name_prefix="deposit",
            inject_electrode=True, inject_volumes_as="vols",
            rate="split_dispense", rate_param="disp_rates",
            derive_settle=True, zero_deadvols=True, is_deposit=True,
        ),
    ),
)

#: Built-in deposition recipes, keyed by name.
BUILTIN_DEPOSITION_RECIPES: dict[str, DepositionRecipe] = {
    r.name: r for r in (_SINGLE_DROP, _TWO_PHASE)
}


def deposition_recipe_names() -> list[str]:
    """Names of the available deposition recipes (built-ins for now)."""
    return list(BUILTIN_DEPOSITION_RECIPES.keys())


def get_deposition_recipe(name: str) -> DepositionRecipe:
    """Look up a deposition recipe by name (raises ``KeyError`` if unknown)."""
    return BUILTIN_DEPOSITION_RECIPES[name]


# ── The single marshalling contract (P2.1) ───────────────────────────────────

@dataclass(frozen=True)
class DepositionSettings:
    """Everything the engine needs that is **not** per-run data.

    One engine has always sat under the HT tab, the autonomous campaign, and the
    Process Studio preview — but each marshalled its kwargs independently, and
    the three had already drifted: the campaign path silently dropped
    ``deposit_method`` and ``piezo`` (so an unattended run ignored the
    deposit-method selection and never actuated the piezo), HT could not scale
    dwells, and the two read ``settle_factor`` from different sources. This type
    is the shared contract that makes such a divergence impossible to express:
    every surface builds one of these, and
    :func:`build_deposition_workflow` is the only caller of
    :func:`build_recipe_deposition_workflow`.

    Per-run data — channels, formulation, catalog, EIS steps, workflow name —
    stays an explicit argument, because it changes every trial while these
    settings hold for a whole run.
    """

    pump_ids: tuple[int, ...] = (0, 1, 2)
    dispense_rate: float = DEFAULT_DISP_RATE_UL_MIN
    flush_rate: float = DEFAULT_FLUSH_RATE_UL_MIN
    flush_factor: float = 3.0
    settle_factor: float = 2.0
    settle_base_s: float = 0.0
    #: Per-pump start flush; empty → the engine's own default.
    start_flush_uL: tuple[float, ...] = ()
    #: Overrides the deposit phase's method (the HT deposit-method selector).
    deposit_method: str | None = None
    piezo: PiezoPlan | None = None
    pcb: dict[str, Any] = field(default_factory=dict)
    origin_xy: tuple[float, float] | None = None
    slots: DepositionSlots | None = None
    #: Scales driver dwells (``0.0`` for instant mock runs). ``None`` → real time.
    time_scale: float | None = None
    run_plan: RunPlan | None = None
    #: Dead-volume correction applied at the hardware boundary (P2.2). ``None``
    #: or a disabled instance → volumes pass through untouched, which is the
    #: live default. Formulations reaching the marshaller are always *desired
    #: delivered* volumes; this is the one place they become *commanded* ones.
    correction: "DeadVolumeCorrection | None" = None

    @classmethod
    def from_config(
        cls,
        pcb: dict[str, Any] | None = None,
        *,
        pump_ids: Sequence[int] = (0, 1, 2),
        **overrides: Any,
    ) -> "DepositionSettings":
        """Build settings from the ``[dropcast]`` config section.

        The documented default source, shared by the Process Studio preview and
        any surface that wants "what the rig is configured to do" without
        threading widgets through. *overrides* wins over config, so a caller can
        take the configured baseline and adjust a field or two.
        """
        from softae.config.loader import dropcast_config

        dc = dropcast_config()
        base: dict[str, Any] = {
            "pump_ids": tuple(int(p) for p in pump_ids),
            "dispense_rate": float(dc["dispense_rate_uL_min"]),
            "flush_rate": float(dc["line_flush_rate_uL_min"]),
            "flush_factor": float(dc["flush_factor"]),
            "settle_factor": float(dc["settle_factor"]),
            "settle_base_s": float(dc.get("settle_base_s", 0.0)),
            "start_flush_uL": tuple(float(v) for v in dc["start_flush_uL"]),
            "pcb": pcb or {},
        }
        base.update(overrides)
        return cls(**base)


def build_deposition_workflow(
    recipe: DepositionRecipe,
    channels: Sequence[int],
    formulation_by_channel: dict[int, Sequence[float]],
    *,
    settings: DepositionSettings,
    catalog: TaskCatalog,
    eis_step_by_channel: dict[int, WorkflowStep] | None = None,
    name: str = "deposition",
) -> Workflow:
    """Build a deposition Workflow from a :class:`DepositionSettings`.

    The **sole** marshaller onto :func:`build_recipe_deposition_workflow`. Kept
    as a thin wrapper rather than folded into the engine so that the engine's
    explicit-kwarg signature — which the recipe tests exercise directly — stays
    untouched; this adds a contract without disturbing the machinery under it.

    ``formulation_by_channel`` arrives as *desired delivered* volumes. This is
    the hardware boundary, so it is also the one place dead-volume correction is
    applied (P2.2) — see :class:`DeadVolumeCorrection` for why the conversion
    belongs this late rather than inside the solver.
    """
    if settings.correction is not None and settings.correction.enabled:
        formulation_by_channel = dict(
            settings.correction.apply_by_channel(formulation_by_channel, channels)
        )
    return build_recipe_deposition_workflow(
        recipe,
        channels,
        formulation_by_channel,
        catalog=catalog,
        pump_ids=list(settings.pump_ids),
        dispense_rate=settings.dispense_rate,
        flush_rate=settings.flush_rate,
        flush_factor=settings.flush_factor,
        settle_factor=settings.settle_factor,
        settle_base_s=settings.settle_base_s,
        start_flush_uL=list(settings.start_flush_uL) or None,
        deposit_method=settings.deposit_method,
        eis_step_by_channel=eis_step_by_channel,
        piezo=settings.piezo,
        pcb=settings.pcb,
        origin_xy=settings.origin_xy,
        slots=settings.slots,
        time_scale=settings.time_scale,
        run_plan=settings.run_plan,
        name=name,
    )


def build_recipe_deposition_workflow(
    recipe: DepositionRecipe,
    channels: Sequence[int],
    formulation_by_channel: dict[int, Sequence[float]],
    *,
    catalog: TaskCatalog,
    pump_ids: Sequence[int],
    dispense_rate: float,
    flush_rate: float,
    flush_factor: float,
    settle_factor: float,
    settle_base_s: float = 0.0,
    start_flush_uL: Sequence[float] | None = None,
    deposit_method: str | None = None,
    eis_step_by_channel: dict[int, WorkflowStep] | None = None,
    piezo: PiezoPlan | None = None,
    pcb: dict,
    origin_xy: tuple[float, float] | None = None,
    slots: DepositionSlots | None = None,
    time_scale: float | None = None,
    run_plan: RunPlan | None = None,
    name: str = "deposition",
) -> Workflow:
    """Build a per-channel deposition Workflow from a :class:`DepositionRecipe`.

    The single engine behind both the single-drop and two-phase HT paths: it
    emits the campaign startup flush, then the ordered phases of a
    :class:`~softae.core.run_plan.RunPlan`, then the teardown flush.  Per-pump
    rate splits and the settle wait come from
    :func:`softae.core.dropcast_plan.build_dropcast_plan` (the single split impl).

    ``run_plan`` controls the phase ordering.  When omitted it defaults to
    :meth:`RunPlan.pointwise` with measurement enabled iff ``eis_step_by_channel``
    is given — reproducing the legacy layout (startup flush → per channel
    [precondition + deposit + EIS] → teardown).  A **per-batch** plan
    (:meth:`RunPlan.batch`) instead casts every channel, then anneals the whole
    plate once, then measures every channel — "formulate-all → anneal-all →
    measure-all".  An inserted ANNEAL phase runs the plan's catalogued anneal task
    on the temperature controller.

    ``deposit_method`` overrides the deposit phase's method (the HT deposit-method
    selector).  ``eis_step_by_channel`` omitted (and no MEASURE phase) → no EIS
    (formulate-only).  ``time_scale`` (when given) is injected into every
    ``liquid_handler`` step so the driver scales its dwells — the autonomous path
    passes ``0.0`` for instant mock runs; the HT path leaves it ``None``.
    """
    slots = slots or DepositionSlots()
    channels = list(channels)
    ids = [int(p) for p in pump_ids]
    if run_plan is None:
        run_plan = RunPlan.pointwise(measure=eis_step_by_channel is not None)
    if origin_xy is None:
        origin_xy = deposition_positions().origin

    setup: list[WorkflowStep] = []

    piezo_on = piezo is not None and piezo.enabled
    piezo_all = piezo is not None and piezo.enabled and piezo.elution_scope == "all_elution"

    def _wrap_elution(step: WorkflowStep, ch: int | None) -> list[WorkflowStep]:
        """Bracket an elution step with piezo on/off in ``all_elution`` scope.

        In any other scope this is a passthrough (``[step]``), so the deposit-only
        behaviour is unchanged. The on/off steps borrow the elution step's name so
        every wrapped event gets a unique, traceable pair.
        """
        if not piezo_all:
            return [step]
        assert piezo is not None  # piezo_all implies an enabled plan
        tags = {"phase": "piezo"}
        if ch is not None:
            tags["channel"] = str(ch)
        out: list[WorkflowStep] = []
        if piezo.on_task in catalog:
            out.append(catalog.get(piezo.on_task).to_step(
                f"piezo_on_{step.name}").with_tags(**tags))
        out.append(step)
        if piezo.off_task in catalog:
            out.append(catalog.get(piezo.off_task).to_step(
                f"piezo_off_{step.name}").with_tags(**tags))
        return out

    # Optional one-shot event-profile step FIRST, so the sweep profile is set
    # before any piezo actuation (including the startup flush in all-elution mode).
    if (piezo is not None and piezo.enabled and piezo.event_task
            and piezo.event_task in catalog):
        ev = catalog.get(piezo.event_task).to_step("piezo_event")
        if piezo.event_params:
            ev = ev.with_params(**piezo.event_params)
        setup.append(ev)

    # Campaign-start general flush: per-pump start volumes at the (broadcast) line rate.
    if recipe.startup_method in catalog:
        start_vols = (
            list(start_flush_uL) if start_flush_uL is not None
            else [DEFAULT_FLUSH_VOL_UL] * len(ids)
        )
        start_vols = (start_vols + [0.0] * len(ids))[: len(ids)]
        startup_params: dict = {
            "ids": list(ids), "disp_vols": start_vols, "disp_rate": float(flush_rate),
        }
        if time_scale is not None and catalog.get(recipe.startup_method).instrument == "liquid_handler":
            startup_params["time_scale"] = float(time_scale)
        startup_step = catalog.get(recipe.startup_method).to_step("startup_flush").with_params(
            **startup_params
        )
        setup.extend(_wrap_elution(startup_step, None))

    # Per-channel context (volumes, rate plan, electrode xy) computed once.
    electrode_xy: dict[str, list[float]] = {}
    ch_ctx: dict[int, tuple[list[float], Any, float, float]] = {}
    for ch in channels:
        volumes = [float(v) for v in formulation_by_channel.get(ch, [])]
        plan = build_dropcast_plan(
            volumes, dispense_rate_total=dispense_rate, flush_rate_total=flush_rate,
            flush_factor=flush_factor, settle_factor=settle_factor,
            settle_base_s=settle_base_s,
        )
        ex, ey = electrode_xy_for_channel(pcb, ch, origin_x=origin_xy[0], origin_y=origin_xy[1])
        electrode_xy[str(ch)] = [ex, ey]
        ch_ctx[ch] = (volumes, plan, ex, ey)

    def _formulate_steps(ch: int) -> list[WorkflowStep]:
        """Recipe precondition/deposit steps for one channel (piezo-on before deposit)."""
        volumes, plan, ex, ey = ch_ctx[ch]
        steps: list[WorkflowStep] = []
        for phase in recipe.phases:
            # Deposit-scope piezo: enable just before the deposit phase.  All-elution
            # scope instead brackets every phase individually (via _wrap_elution below).
            if (phase.is_deposit and not piezo_all and piezo is not None
                    and piezo.enabled and piezo.on_task in catalog):
                steps.append(catalog.get(piezo.on_task).to_step(
                    f"piezo_on_ch{ch}").with_tags(channel=str(ch), phase="piezo"))
            method = deposit_method if (phase.is_deposit and deposit_method) else phase.method
            params: dict = {}
            if phase.inject_electrode:
                params[slots.electrode_x] = ex
                params[slots.electrode_y] = ey
            if phase.inject_volumes_as is not None:
                params[phase.inject_volumes_as] = list(volumes)
                params["ids"] = list(ids)
            if phase.rate == "flat_dispense":
                params[phase.rate_param] = float(dispense_rate)
            elif phase.rate == "split_dispense":
                params[phase.rate_param] = list(plan.deposition.rates_uL_min)
            elif phase.rate == "split_flush":
                params[phase.rate_param] = list(plan.flush_rates_uL_min)
            if phase.pass_flush_factor:
                params["flush_factor"] = float(flush_factor)
            if phase.derive_settle:
                params["elution_wait_s"] = float(plan.settle_wait_s)
            if phase.zero_deadvols:
                params["deadvols"] = [0.0] * len(ids)
            task = catalog.get(method)
            if time_scale is not None and task.instrument == "liquid_handler":
                params["time_scale"] = float(time_scale)
            phase_step = task.to_step(f"{phase.name_prefix}_ch{ch}").with_params(
                **params).with_tags(channel=str(ch), phase=phase.key)
            # In all-elution scope each phase (precondition, deposit) is bracketed
            # with the piezo; otherwise this is a plain append.
            steps.extend(_wrap_elution(phase_step, ch))
        return steps

    def _measure_steps(ch: int) -> list[WorkflowStep]:
        """The per-channel EIS step, if one was supplied for this channel.

        Tagged as a **purge window**: an EIS sweep is a single opaque blocking
        read with no interior yield point, so unlike an anneal it cannot offer
        repeated opportunities through a poll loop — the only way to use its dead
        time is to run the purge concurrently on disjoint instruments. Operator
        confirmed 2026-08-03 that dispensing does not perturb the measurement.
        """
        if eis_step_by_channel and ch in eis_step_by_channel:
            return [eis_step_by_channel[ch].with_tags(
                channel=str(ch), purge_window="concurrent")]
        return []

    def _anneal_steps(rp: RunPhase, ch: int | None) -> list[WorkflowStep]:
        """Anneal step for one channel (``ch``) or the whole plate (``ch is None``).

        Bracketed so the hold happens with the tip resting in the flush basin.
        Two things follow from that, and both matter:

        * the dispenser tip is protected for the whole hold, which for a
          soft-material anneal can be hours — the longest stretch of a run where
          it would otherwise sit in open air;
        * an anti-clog purge during the hold costs **no motion and no extra
          time**, because the rig is already where it purges. That is the only
          point in a run where that is true, and it is also the only stretch
          where no pump moves — so it is exactly where the particulate line is
          most at risk. See ``monitored_hold``'s ``on_poll`` hook.

        The trailing retract is not optional: the head guard refuses stage motion
        while lowered, so leaving it down would block the next phase.
        """
        if rp.anneal_task not in catalog:
            logger.warning("anneal_task_missing", task=rp.anneal_task,
                           channel=ch if ch is not None else "all")
            return []
        step = catalog.get(rp.anneal_task).to_step(
            f"anneal_ch{ch}" if ch is not None else "anneal_all")
        if rp.anneal_params:
            step = step.with_params(**dict(rp.anneal_params))
        # Derive the ceiling from the hold actually requested, so a long anneal
        # cannot be aborted partway by a task's short hand-set timeout.
        step = step.with_timeout(anneal_timeout_s(step.params, step.timeout_s))
        tags = {"phase": "anneal"}
        if ch is not None:
            tags["channel"] = str(ch)
        suffix = f"ch{ch}" if ch is not None else "all"
        return [
            *_rest_at_flush_steps(suffix, tags),
            # A purge window like any other. The executor re-offers on a cadence
            # for as long as the step runs, so a five-hour hold gets repeated
            # opportunities without the temperature driver knowing purging
            # exists. Bracketed above, so each one costs no motion.
            step.with_tags(**tags, purge_window="concurrent"),
            WorkflowStep(name=f"anneal_leave_rest_{suffix}", instrument="syringe",
                         method="head_retract", params={}).with_tags(**tags),
        ]

    def _rest_at_flush_steps(suffix: str, tags: dict) -> list[WorkflowStep]:
        """Travel to the flush basin and lower the head, in that order."""
        flush = deposition_positions().flush
        return [
            WorkflowStep(
                name=f"anneal_to_flush_{suffix}", instrument="stage",
                method="move_to", params={"x": flush[0], "y": flush[1]},
            ).with_tags(**tags),
            WorkflowStep(
                name=f"anneal_rest_{suffix}", instrument="syringe",
                method="head_descend", params={},
            ).with_tags(**tags),
        ]

    def _emit_per_sample(rp: RunPhase, ch: int) -> list[WorkflowStep]:
        if rp.kind is PhaseKind.FORMULATE:
            return _formulate_steps(ch)
        if rp.kind is PhaseKind.ANNEAL:
            return _anneal_steps(rp, ch)
        if rp.kind is PhaseKind.MEASURE:
            return _measure_steps(ch)
        return []

    # Emit the run plan.  A per-sample segment loops channels so each channel's
    # phases stay contiguous (the executor's dispense-recovery relies on this);
    # a per-batch segment is a boundary block across all channels (cast-all →
    # anneal-all → measure-all).
    for scope, seg_phases in run_plan.segments():
        if scope is PhaseScope.PER_SAMPLE:
            seg_has_formulate = any(p.kind is PhaseKind.FORMULATE for p in seg_phases)
            for ch in channels:
                for rp in seg_phases:
                    setup.extend(_emit_per_sample(rp, ch))
                # Deposit-scope: disable the piezo after this channel's per-sample
                # phases (deposit + any adjacent per-sample EIS), mirroring the
                # legacy order.  All-elution scope disables per event instead.
                if (seg_has_formulate and not piezo_all and piezo is not None
                        and piezo.enabled and piezo.off_task in catalog):
                    setup.append(catalog.get(piezo.off_task).to_step(
                        f"piezo_off_ch{ch}").with_tags(channel=str(ch), phase="piezo"))
        else:  # PER_BATCH — whole-plate boundary
            for rp in seg_phases:
                if rp.kind is PhaseKind.ANNEAL:
                    setup.extend(_anneal_steps(rp, None))
                elif rp.kind is PhaseKind.MEASURE:
                    for ch in channels:
                        setup.extend(_measure_steps(ch))

    teardown: list[WorkflowStep] = []
    if recipe.final_method in catalog:
        final_step = catalog.get(recipe.final_method).to_step("final_flush")
        teardown.extend(_wrap_elution(final_step, None))
    # Return the piezo to standby last.
    if piezo is not None and piezo.enabled and piezo.standby_task in catalog:
        teardown.append(catalog.get(piezo.standby_task).to_step("piezo_standby"))

    logger.info("recipe_deposition_built", recipe=recipe.name, channels=channels,
                piezo=piezo_on)
    return Workflow(
        name=name,
        description=f"{recipe.label} deposition on channels {','.join(map(str, channels))}",
        setup=setup,
        teardown=teardown,
        iterations=1,
        metadata={
            "source": "deposition_engine",
            "recipe": recipe.name,
            "channels": list(channels),
            "electrode_xy": electrode_xy,
            "piezo": "applied" if piezo_on else "not_applied",
            "run_plan": run_plan.describe(),
            "deferred_measurement": run_plan.defers_measurement,
        },
    )
