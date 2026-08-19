"""Agentic execution hook — declarative wiring around :class:`AutonomousLoop`.

This module is the production caller and workflow-template builder that
:mod:`softae.core.autonomous_loop` was written to expect but never had.  It
turns a small declarative :class:`CampaignSpec` into a fully-wired closed loop
(``suggest → [approve] → execute → analyze → tell``) running headlessly over the
instrument manager — no GUI, no button clicks.  That headless surface is what an
autonomous agent (an LLM policy, a scheduler, or a human via CLI) drives:

    result = await run_autonomous_campaign(spec)

Each trial is built by the **shared deposition engine**
(:func:`softae.core.deposition_recipe.build_recipe_deposition_workflow`) — the same
modular recipe path the HT Experiment tab runs — so HT and autonomous are one
workflow story, not two.  Because the optimizer's suggested per-pump volumes are
concrete per trial, the engine builds a fully concrete workflow (no ``$var``
templating, no driver-deferred rate split): the split and the derived settle come
from :mod:`softae.core.dropcast_plan` at build time, exactly as in HT.  Electrode
positions are resolved by the engine from PCB geometry (never encoded in a recipe).

An ``approval_fn`` gate and an ``on_event`` stream make the loop *steerable*:
the caller can require sign-off per suggestion and observe every state change,
which is the seam a higher-level agent plugs into for supervised autonomy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Sequence

import structlog

if TYPE_CHECKING:  # annotation-only; the settle criterion is imported on use
    from softae.analysis.equilibration import RoundFit

from softae.config import loader
from softae.core.autonomous_loop import (
    AutonomousLoop,
    BoardCheck,
    BoardDecision,
    LoopState,
)
from softae.core.campaign_events import (
    CampaignNarrator,
    ConditionsPublisher,
    ControlWatcher,
    open_conditions_publisher,
    open_control_watcher,
    open_narrator,
)
from softae.core.data_store import DataStore
from softae.core.electrode_allocator import ElectrodeAllocator
from softae.core.deposition_recipe import (
    DepositionSettings,
    PiezoPlan,
    build_deposition_workflow,
    get_deposition_recipe,
)
from softae.core.deposition_steps import (
    DEFAULT_DISP_RATE_UL_MIN,
    DEFAULT_ELUTION_WAIT_S,
    DEFAULT_FLUSH_RATE_UL_MIN,
    resolve_pcb,
)
from softae.core.formulation import (
    ChemicalCatalog,
    FormulationContext,
    FormulationInfeasibleError,
    FormulationPlan,
    FormulationTarget,
    Solution,
    TotalDepositTarget,
    plan_formulation,
    solve_formulation,
)
from softae.core.geometry import well_capacity_uL
from softae.core.measurement_spec import (
    LEGACY_UNSET,
    MeasurementSpec,
    canonicalize_measurement,
    measurement_identity,
)
# Safe at module scope in this direction only: the registry imports *this*
# module exclusively inside functions, so its module body never reaches back
# here. Keep it that way — see `modality_registry`'s "Import discipline".
from softae.core.modality_registry import ObjectiveSpec, get_modality
from softae.core.run_lock import held_run_lock, rig_is_simulated
from softae.core.run_plan import RunPlan, SettlePlan
from softae.core.alerts import CRITICAL, Alert, raise_alert
from softae.core.safe_park import safe_park
from softae.core.shutdown import park_on_shutdown
from softae.core.task_catalog import TaskCatalog
from softae.optimizers import (
    BaseOptimizer,
    BayesianOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)
from softae.optimizers.bayesian import PriorMean
from softae.optimizers.failure_labels import (
    CONFIRM_MEASUREMENT,
    MAX_CONFIRMATION_SWEEPS,
    FailureLabelEngine,
    reject_signature,
)
from softae.optimizers.feasibility import ABSOLUTE_LABEL_FLOOR, FeasibilityConfig
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)

# (step_results, suggested_params) -> scalar objective.
# Returns None to mean "no usable measurement"; the loop skips rather than
# telling the optimizer a fabricated value. See eis_impedance_objective.
ObjectiveExtractor = Callable[[dict[str, Any], dict[str, Any]], "float | None"]

# Event stream callback: receives a plain dict an agent/UI can react to.
EventCallback = Callable[[dict[str, Any]], Any]

# Approval gate: given (iteration, params) return True to proceed. May be async.
ApprovalFn = Callable[[int, dict[str, Any]], "bool | Awaitable[bool]"]

# Board-exchange gate: given the new board index, return a BoardDecision (or a
# bool: True=proceed, False=cancel the run). May be async.
BoardExchangeFn = Callable[[int], "Any"]

# Board-freshness check on resume: given (board_id, occupied_electrodes), return
# a BoardCheck (FRESH / RESUME / CANCEL). May be async.
BoardCheckFn = Callable[[int, "set[int]"], "Any"]

_DEPOSIT_STEP = "deposit"  # engine names deposit steps f"deposit_ch{ch}"
_MEASURE_STEP = "measure_eis"
#: ``CampaignSpec.equilibration_method`` value that turns on the EQUILIBRATE phase.
SETTLE_METHOD = "settle"
#: Step-name prefix and ``measurement`` tag for a settle round's sweep. Tagged
#: apart from ``primary`` so a round can never enter the objective on its own —
#: the *last* one does, by being injected under the trial's own measure-step
#: name, and that injection is a decision the driver makes rather than a
#: side-effect of a tag (the same mechanism ``confirmation_measure_step`` uses).
SETTLE_STEP = "settle_eis"
SETTLE_MEASUREMENT = "settle"
#: The circuit model the campaign's σ path fits with. The campaign measure step
#: declares no ``circuit_model``, so ``analyze_spectrum`` fits with its own
#: default — named here **only** so ``r1_lower_bound_ohms`` can be asked for the
#: matching R₁ bound. A fit railed on that bound reports a CONSTANT σ, and a
#: constant is exactly what a settle criterion mistakes for settled: 325 of 1440
#: fits in the reference run railed while reporting ``success = 1``.
SETTLE_CIRCUIT_MODEL = "simpleSalt"

#: The six flat settle fields on :class:`CampaignSpec`, and the three of them
#: that cannot be defaulted. Named once so :meth:`CampaignSpec.settle_plan` and
#: its refusals cannot drift from the dataclass.
_SETTLE_FIELDS = (
    "round_period_s", "min_hold_s", "max_hold_s",
    "settle_tol_rel", "settle_n_rounds", "settle_min_channels",
    "rh_stability_pct",
)
_SETTLE_REQUIRED = ("round_period_s", "min_hold_s", "max_hold_s")

#: Consecutive RH-decided equilibrate phases that park the campaign. Three, to
#: match the streak :class:`~softae.core.autonomous_loop.AutonomousLoop` already
#: parks on (``park_after_failed_trials = 3``) — a second streak limit at a
#: different K in the same loop would be an unexplained inconsistency. ``0``
#: disables the escalation and leaves the per-trial behaviour (refuse to certify,
#: run to the ceiling, continue) exactly as it is. Overridable as
#: ``[safety] rh_ceiling_park_after_trials``.
DEFAULT_RH_CEILING_PARK_AFTER_TRIALS = 3

#: Workflow-metadata key carrying ``{step_name: tags}`` for the steps built to
#: *measure*. Loop closure is tag-based (T1.5): the objective extractors receive
#: only ``{step_name: result}``, so the set of measurement steps — and their
#: tags — must travel with the workflow the wiring built. Scoped to measurement
#: steps deliberately: deposit steps also carry a ``channel`` tag, so "has a
#: channel tag" alone cannot mean "objective input".
_MEASUREMENT_TAGS_KEY = "measurement_step_tags"

#: Terminal loop state → ``experiments.status`` recorded by ``finish_run``.
#: A budget-exhausted run breaks out without setting a terminal state, so the
#: default ("done") covers it.
_RUN_STATUS_BY_STATE = {
    LoopState.CONVERGED: "converged",
    LoopState.STOPPED: "stopped",
    LoopState.ERROR: "error",
}

#: Run statuses that mean **the previous run stopped by a path that told a
#: human, and unwound far enough to say so**. This is the read side of
#: ``_RUN_STATUS_BY_STATE`` and of ``finish_run``'s documented vocabulary; see
#: ``_previous_exit_was_acknowledged`` for what it is used to decide.
#:
#: ``stopped``    a park (``AutonomousLoop._park`` sets STOPPED and raises a
#:                CRITICAL ``kind="park"`` alert) *or* an operator's own
#:                ``loop.stop()``. The DB cannot separate the two — there is no
#:                ``'parked'`` status — but it does not need to: both are an
#:                operator being told, or an operator acting.
#: ``converged``  the campaign met its own convergence criterion.
#: ``done``       normal completion, and the budget-exhausted default.
#: ``partial``    finished every step it attempted, having abandoned channels.
#: ``aborted``    an operator's deliberate abort — their own hand on the switch.
#:
#: Everything else is unacknowledged **by omission**, which is deliberate: a new
#: status added later defaults to preserving the streak, and the failure mode of
#: that default is a park that comes early rather than a chronic fault that never
#: escalates.
_ACKNOWLEDGED_EXIT_STATUSES = frozenset(
    {"stopped", "converged", "done", "partial", "aborted"}
)


def _previous_exit_was_acknowledged(
    data_store: "DataStore", run_id: str | None
) -> tuple[bool, str]:
    """Did the run this resume continues stop in a way a human was told about?

    Returns ``(acknowledged, why)``; *why* is a short phrase for the log, because
    the whole point of this decision is that it must not be silent.

    ``run_id`` is the **previous** run's id, carried on the checkpoint row. Three
    ways it can fail to answer, all of which resolve to *not* acknowledged:
    the checkpoint predates the column and holds ``None``; the row was deleted;
    the status is one nothing here has heard of.
    """
    if not run_id:
        return False, "the checkpoint records no previous run id"
    try:
        outcome = data_store.run_outcome(run_id)
    except Exception:  # noqa: BLE001 - a resume must not fail on a status read
        logger.warning("run_outcome_failed", run_id=run_id, exc_info=True)
        return False, "the previous run's status could not be read"
    if outcome is None:
        return False, f"no run row for {run_id}"
    status = outcome["status"]
    if not outcome["finished"]:
        # `finished_at` NULL beats the status string: a hard kill leaves the row
        # at its 'running' default, and the recovery sweep that rewrites it to
        # 'interrupted' runs at the *next* launch, which may not have happened.
        return False, f"the previous run was never closed (status {status!r})"
    return status in _ACKNOWLEDGED_EXIT_STATUSES, f"the previous run ended {status!r}"


# ── Campaign specification ───────────────────────────────────────────────────

@dataclass
class GeneralFormulation:
    """Composition campaign over *arbitrary* stocks + ``solve_formulation`` targets.

    The general analogue of the ternary :class:`FormulationContext`: instead of the
    fixed ``eo_li_ratio`` / ``silica_vol_frac`` axes, the optimizer searches values
    that ``build_targets`` maps into any mix of composition targets (molar ratio /
    dried fraction / concentration).  ``TotalDepositTarget(target_deposition_uL)`` is
    appended automatically, and the per-electrode ``budget_uL`` rides in from the
    board (filled in :func:`run_autonomous_campaign` when left ``None``).

    ``build_targets`` maps a suggestion's params → the composition targets, e.g.::

        lambda p: [MolarRatioTarget("PEO", "LiCl", p["eo_li"]),
                   DriedFractionTarget("SiO2", p["silica"], Basis.VOLUME)]
    """

    stocks: dict[str, Solution]
    catalog: ChemicalCatalog
    pump_assignment: dict[str, int]        # stock name → pump index
    target_deposition_uL: float
    build_targets: Callable[[dict[str, Any]], Sequence[FormulationTarget]]
    budget_uL: float | None = None
    dried_frac: dict[str, float] | None = None  # per-stock dried-fraction override


class _SpecUnset:
    """Sentinel for a spec field the campaign never mentioned.

    Distinct from *any* value the field can take — including ``None`` and the
    shipped default — so a resolver can tell "the campaign did not speak" from
    "the campaign chose the default". Without it, ``exclusion_radius = None``
    (the shipped behaviour) and "unset" would be the same object and a TOML site
    default could never be honoured.
    """

    _instance: "_SpecUnset | None" = None

    def __new__(cls) -> "_SpecUnset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<unset>"

    def __bool__(self) -> bool:
        return False


SPEC_UNSET = _SpecUnset()


@dataclass
class CampaignSpec:
    """Declarative description of an autonomous deposition campaign.

    The optimizer searches ``parameter_space``; each parameter name becomes a
    ``$var`` the trial workflow can reference.  ``vol_params`` names the subset
    (in pump order) that drive the per-pump dispense volumes of the composite
    drop-cast — that is how a suggested composition reaches the hardware.
    """

    name: str
    channels: tuple[int, ...] = (1,)  # 1-based electrodes to deposit on
    parameter_space: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Composition mode (the standard path): a :class:`FormulationContext` naming
    #: the stocks, pump assignment, dried mode, and dep-volume target.  When set,
    #: ``parameter_space`` holds **composition axes** (e.g. ``eo_li_ratio``,
    #: ``silica_vol_frac``) and each suggestion is turned into per-pump volumes by
    #: :func:`softae.core.formulation.plan_formulation` — the same call the GUI
    #: uses.  The per-electrode budget rides in from the board's ``well_capacity_uL``
    #: (filled at run time from the PCB config unless the context sets its own).
    #: When ``None``, the campaign falls back to the *legacy* identity map where
    #: the searched params ARE raw per-pump volumes (deprecated; see ``vol_params``).
    formulation: FormulationContext | None = None
    #: General composition mode: arbitrary stocks + ``solve_formulation`` targets
    #: (see :class:`GeneralFormulation`).  Takes precedence over ``formulation`` /
    #: the legacy path when set — the non-ternary analogue of ``formulation``.
    general_formulation: GeneralFormulation | None = None
    vol_params: tuple[str, ...] = ()  # legacy: ordered param names → deposit vols
    pump_ids: tuple[int, ...] = (0, 1)
    #: Optimisation direction. ``"auto"`` (default) derives it from the
    #: resolved objective — maximise conductivity, minimise impedance — since
    #: the direction is fixed by the metric rather than chosen. An explicit
    #: value is honoured only if it agrees; see :func:`resolve_direction`.
    objective: str = "auto"  # "auto" | "maximize" | "minimize"
    optimizer: str = "bayesian"  # "random" | "grid" | "bayesian"
    acquisition: str = "ucb"  # "ucb" | "ei" (bayesian optimizer only)
    kappa: float = 2.0  # UCB exploration weight (bayesian optimizer only)
    #: q-batch BO: each round proposes q = len(channels) *distinct* suggestions
    #: and casts one per electrode in a single physical run, scoring each against
    #: its own channel's EIS. Off → one suggestion replicated across all channels.
    #: Bayesian/random optimizers only.
    batch: bool = False
    #: Batch diversification method (bayesian only): "constant_liar" (default),
    #: "kriging_believer", or "botorch_mc" (planned). See optimizers.batch.
    batch_strategy: str = "constant_liar"
    #: Electrode/board management. Drop-cast wells are single-use, so each sample
    #: consumes a fresh electrode. When ``electrode_capacity`` is set (e.g. 32),
    #: electrodes are allocated sequentially ``electrode_start..capacity``; when a
    #: board fills mid-run the loop prompts a board exchange (with a cancel-run
    #: option) and runs an equilibration routine on the fresh board before
    #: continuing. ``None`` (default) → today's fixed-``channels`` behavior.
    electrode_capacity: int | None = None
    electrode_start: int = 1
    #: Hardware equilibration routine run after a board swap (one step).
    equilibration_instrument: str = "rh_controller"
    #: ``"wait"`` (the default) or ``"settle"``.
    #:
    #: ``"wait"`` is the behaviour that has always shipped: a fixed
    #: ``equilibration_s`` stabilization on the RH controller after a board
    #: exchange, and **nothing that waits for the sample**. Against the τ ≈ 500 s
    #: measured at the first setpoint of
    #: ``20260811T023757Z_equilibration_characterization`` the shipped 60 s is
    #: ~0.12 τ, so every σ such a campaign feeds its optimiser is taken off a film
    #: that is still drying.
    #:
    #: ``"settle"`` adds an EQUILIBRATE phase per trial: the run measures every
    #: ``round_period_s`` and stops when σ stops moving, or at ``max_hold_s``.
    #: **Opt-in on purpose.** An unattended campaign that starts measuring on a
    #: criterion nobody has exercised is a worse failure than one that waits too
    #: little, so the default is unchanged until a real batch has run it.
    equilibration_method: str = "wait"
    equilibration_s: float = 60.0
    #: ── EQUILIBRATE phase parameters (``equilibration_method = "settle"``) ──
    #:
    #: Sentinel-defaulted, like the T1.3 optimizer knobs: "unset" and "set to the
    #: shipped default" must stay distinct, because a spec that also carries an
    #: EQUILIBRATE phase in its ``run_plan`` is speaking twice and
    #: :meth:`settle_plan` refuses rather than picking one. Names mirror
    #: :class:`~softae.core.run_plan.SettlePlan` and the equilibration CLI so an
    #: operator meets one vocabulary. See :meth:`settle_plan`.
    round_period_s: float = SPEC_UNSET       # type: ignore[assignment]
    min_hold_s: float = SPEC_UNSET           # type: ignore[assignment]
    max_hold_s: float = SPEC_UNSET           # type: ignore[assignment]
    settle_tol_rel: float = SPEC_UNSET       # type: ignore[assignment]
    settle_n_rounds: int = SPEC_UNSET        # type: ignore[assignment]
    settle_min_channels: int = SPEC_UNSET    # type: ignore[assignment]
    #: How far the room may move across a judged settle window (%RH). Unset →
    #: :data:`~softae.analysis.equilibration.DEFAULT_RH_STABILITY_PCT`, i.e. the
    #: gate is ON; pass ``None`` explicitly to switch it off. Unlike the three
    #: durations this one has a defensible default, so it stays out of
    #: :data:`_SETTLE_REQUIRED`.
    rh_stability_pct: float | None = SPEC_UNSET  # type: ignore[assignment]
    #: Optional explicit run plan (phase ordering). ``None`` → the engine's legacy
    #: pointwise layout (per-channel deposit + EIS). Set :meth:`RunPlan.batch` to
    #: defer anneal/measurement into whole-plate blocks (formulate-all →
    #: anneal-all → measure-all), or insert an ANNEAL phase to cure between
    #: deposit and measure. See :mod:`softae.core.run_plan`.
    run_plan: "RunPlan | None" = None
    budget: int = 12
    #: What this campaign measures, and how (T2.4). One block naming a
    #: **modality** alongside its preset/overrides, so a second modality needs no
    #: new spec fields. Defaults to exactly today's behaviour: EIS, ``Quick``, no
    #: overrides, enabled. This is the authority at run time — the three
    #: ``eis_*`` fields below are canonicalized into it by ``__post_init__``.
    measurement: "MeasurementSpec | None" = None
    #: DEPRECATED (T2.4) — the EIS-shaped spelling of the block above.
    #:
    #: Transitional shim per the restructuring spec's compatibility policy:
    #: still accepted by the constructor, the TOML loader and the GUI, still
    #: *readable* (``__post_init__`` mirrors the canonical block back onto them),
    #: and removed **after one full campaign runs from a measurement-block
    #: spec**. Supplying both spellings with different values raises rather than
    #: picking one — see :func:`~softae.core.measurement_spec.canonicalize_measurement`.
    #: ``eis_overrides`` keys: ``f_hi``, ``f_lo_mHz``, ``npts``, ``mv_ac``,
    #: ``mv_dc``. See :class:`softae.core.eis_scripts.EISParams`.
    eis_preset: str = LEGACY_UNSET          # type: ignore[assignment]
    eis_overrides: dict[str, Any] = LEGACY_UNSET   # type: ignore[assignment]
    measure_eis: bool = LEGACY_UNSET        # type: ignore[assignment]
    auto_approve: bool = True
    seed: int | None = 42
    pcb_name: str | None = None  # None → first PCB in config
    disp_rate: float = DEFAULT_DISP_RATE_UL_MIN
    deadvols: tuple[float, ...] = ()  # per-pump dead volume; default zeros
    elution_wait_s: float = DEFAULT_ELUTION_WAIT_S
    time_scale: float = 1.0  # scale all routine dwells (set <1 for fast demos)
    #: Two-phase cast (precondition flush → deposition) with per-pump rates split
    #: proportionally to the suggested per-pump volumes. When on, ``disp_rate`` is
    #: the *total* deposition rate and ``line_flush_rate`` the *total* flush rate;
    #: the driver splits both across pumps at run time (see dropcast_plan).
    two_phase: bool = False
    #: Deposition recipe **by name** (P2.3). The HT tab has always selected a
    #: recipe from the registry, but the spec could only encode ``two_phase`` —
    #: a bool that spans exactly two recipes and needs a new flag for every
    #: future one. ``None`` falls back to that bool, so existing specs are
    #: unaffected; see :meth:`resolved_recipe_name`.
    recipe_name: str | None = None
    line_flush_rate: float = 500.0
    flush_factor: float = 3.0
    settle_factor: float = 2.0
    settle_base_s: float = 0.0
    start_flush_uL: tuple[float, ...] = ()  # per-pump start flush; default 80 each
    #: Deposit-phase method override — the campaign-side equivalent of the HT
    #: deposit-method selector. Before P2.1 the campaign path could not express
    #: this at all, so an autonomous run silently ran the recipe's built-in
    #: method however the operator had set HT.
    deposit_method: str | None = None
    #: Piezo actuation around elution events. Likewise unexpressible before P2.1,
    #: so campaigns never actuated the piezo even with ``[piezo]`` configured.
    piezo: "PiezoPlan | None" = None
    #: Advisory maturity bar for the methods this campaign runs. Below it, the
    #: loop warns (per the pipeline's warn-and-proceed policy) but proceeds.
    expected_maturity: str = "validated"
    #: Physically/prior-informed BO hooks (bayesian optimizer only):
    #: ``prior_mean`` is a physics model ``m(params) -> objective`` the GP models
    #: the residual from; ``seed_observations`` are prior ``(params, value)``
    #: points fed to the optimizer via ``tell`` before the loop (warm-start).
    prior_mean: "PriorMean | None" = None
    seed_observations: tuple[tuple[dict[str, Any], float], ...] = ()

    #: ── Optimizer tuning (T1.3 knobs, reachable since T3.1; + T3.1's own) ──
    #:
    #: All seven are **sentinel-defaulted**, not value-defaulted, so "unset" and
    #: "explicitly set to the shipped default" stay distinct — the T2.4 shim
    #: mechanism. That distinction is what lets `build_optimizer` honour a TOML
    #: site default only when the campaign did not speak, and what keeps a spec
    #: that never mentions these contributing nothing to the resume fingerprint.
    #:
    #: `build_optimizer` is the SOLE place any of them is read. Reading the TOML
    #: anywhere else and passing the answer in would agree today and drift the
    #: moment the rule changes in one place — the T2.6b lesson.
    #:
    #: T1.3 (spec fields + TOML close the reachability gap; user decision (iii)):
    decision_rtol: float = SPEC_UNSET        # type: ignore[assignment]
    exclusion_radius: float | None = SPEC_UNSET  # type: ignore[assignment]
    #: T3.1 learned feasibility. Master switch off: no classifier constructed,
    #: no label read, no behaviour change. It never turns itself on.
    learned_feasibility: bool = SPEC_UNSET   # type: ignore[assignment]
    feasibility_strategy: str = SPEC_UNSET   # type: ignore[assignment]
    feasibility_min_filter: bool = SPEC_UNSET     # type: ignore[assignment]
    feasibility_min_infeasible: int = SPEC_UNSET  # type: ignore[assignment]
    feasibility_min_feasible: int = SPEC_UNSET    # type: ignore[assignment]

    def resolved_vol_params(self) -> tuple[str, ...]:
        """Volume-driving param names, defaulting to the whole space in order."""
        return self.vol_params or tuple(self.parameter_space.keys())

    def resolved_recipe_name(self) -> str:
        """The deposition recipe to run.

        :attr:`recipe_name` wins; otherwise the legacy :attr:`two_phase` bool is
        honoured, so specs written before the field existed behave identically.
        """
        if self.recipe_name:
            return self.recipe_name
        return "two_phase" if self.two_phase else "single_drop"

    def settle_plan(self) -> SettlePlan | None:
        """This campaign's EQUILIBRATE settings, or ``None`` if it does not settle.

        **One authority, two spellings, and it refuses to guess between them.**
        A :class:`~softae.core.run_plan.SettlePlan` may arrive either on an
        EQUILIBRATE phase inside :attr:`run_plan` (the structural spelling, which
        also fixes the phase's *scope*) or on the six flat fields above (the
        operator spelling, reachable from a config file). Supplying both raises,
        exactly as :func:`~softae.core.measurement_spec.canonicalize_measurement`
        refuses a spec that names its measurement twice — picking one silently is
        how a campaign ends up holding for a duration nobody wrote down.

        The three durations have no defaults (see :class:`SettlePlan`), so
        ``equilibration_method = "settle"`` without them is an error rather than
        an invented hold.
        """
        phases = self.run_plan.equilibrate_phases() if self.run_plan else []
        supplied = {f: getattr(self, f) for f in _SETTLE_FIELDS
                    if getattr(self, f) is not SPEC_UNSET}
        wants_settle = str(self.equilibration_method).strip().lower() == SETTLE_METHOD

        if phases:
            if supplied or wants_settle:
                raise ValueError(
                    "settle is specified twice: run_plan carries an EQUILIBRATE "
                    f"phase and the spec also sets {sorted(supplied) or ['equilibration_method']}"
                    " — say it once so the hold has one authority"
                )
            if len({p.settle for p in phases}) > 1:
                raise ValueError("run_plan carries EQUILIBRATE phases with "
                                 "different SettlePlans; the campaign path drives one")
            return phases[0].settle

        if not wants_settle:
            if supplied:
                raise ValueError(
                    f"{sorted(supplied)} set but equilibration_method is "
                    f"'{self.equilibration_method}'; settle parameters do nothing "
                    f"unless the method is '{SETTLE_METHOD}'"
                )
            return None

        missing = [f for f in _SETTLE_REQUIRED if f not in supplied]
        if missing:
            raise ValueError(
                f"equilibration_method='{SETTLE_METHOD}' requires {missing}; "
                f"min_hold_s is the cure time and belongs to the recipe, and "
                f"max_hold_s is the ceiling that makes the phase terminate — "
                f"neither has a safe default"
            )
        return SettlePlan(**supplied)

    def deposition_settings(
        self, *, pcb: dict[str, Any], n_pumps: int | None = None
    ) -> DepositionSettings:
        """Project the deposition-execution subset onto the shared contract.

        The spec's BO fields (parameter space, optimizer, budget, priors) stay
        here; only what the engine consumes crosses over. *pcb* is resolved by
        the caller because :attr:`pcb_name` needs a config lookup, and *n_pumps*
        trims :attr:`pump_ids` to the width of the actual volume vector.
        """
        from softae.core.liquid_handling import DeadVolumeCorrection

        ids = tuple(self.pump_ids[:n_pumps]) if n_pumps else tuple(self.pump_ids)
        # Two-phase broadcasts the total line rate (the engine splits it per
        # pump); single-drop uses the plain prime rate for its startup flush.
        # Keyed off the *resolved* recipe, not the legacy bool, so a spec that
        # names "two_phase" via recipe_name still gets the two-phase rate.
        two_phase = self.resolved_recipe_name() == "two_phase"
        flush_rate = self.line_flush_rate if two_phase else DEFAULT_FLUSH_RATE_UL_MIN
        return DepositionSettings(
            pump_ids=ids or (0,),
            dispense_rate=self.disp_rate,
            flush_rate=flush_rate,
            flush_factor=self.flush_factor,
            settle_factor=self.settle_factor,
            settle_base_s=self.settle_base_s,
            start_flush_uL=tuple(self.start_flush_uL),
            deposit_method=self.deposit_method,
            piezo=self.piezo,
            pcb=pcb,
            time_scale=self.time_scale,
            run_plan=self.run_plan,
            # Campaigns previously skipped correction entirely, so with it
            # enabled they would have under-delivered relative to HT.
            correction=DeadVolumeCorrection.from_config(ids or (0,)),
        )

    def with_measurement(self, measurement: MeasurementSpec) -> "CampaignSpec":
        """Copy of this spec carrying *measurement*, dropping the legacy mirrors.

        ``dataclasses.replace(spec, measurement=...)`` cannot do this: it carries
        the mirrored ``eis_*`` values forward, so the new block would be checked
        against the old spelling and rejected as a conflict. Clearing them here
        is what makes the block the only thing supplied.
        """
        return replace(
            self, measurement=measurement,
            eis_preset=LEGACY_UNSET, eis_overrides=LEGACY_UNSET,
            measure_eis=LEGACY_UNSET,
        )

    def __post_init__(self) -> None:
        # Accept a bare int or a list for `channels` and normalise to a tuple.
        if isinstance(self.channels, int):
            self.channels = (self.channels,)
        else:
            self.channels = tuple(self.channels)
        if not self.channels:
            raise ValueError("CampaignSpec.channels must name at least one channel")

        # One authority for what gets measured (T2.4). Runs on every
        # construction, including `dataclasses.replace`, so a spec is never
        # observed in the half-canonical state.
        self.measurement = canonicalize_measurement(
            measurement=self.measurement,
            eis_preset=self.eis_preset,
            eis_overrides=self.eis_overrides,
            measure_eis=self.measure_eis,
            owner="CampaignSpec",
        )
        # Mirror the canonical block back onto the deprecated names. This is the
        # *read* half of the shim — `preflight.py`, the GUI and existing tests
        # still do `spec.eis_preset` — and it also makes `replace()` idempotent:
        # the values it carries forward now always agree with the block.
        self.eis_preset = self.measurement.preset
        self.eis_overrides = dict(self.measurement.overrides)
        self.measure_eis = self.measurement.enabled


@dataclass
class CampaignResult:
    """Outcome of a completed (or stopped) campaign."""

    run_id: str
    best_params: dict[str, Any] | None
    best_objective: float | None
    n_trials: int
    final_state: str
    converged: bool
    history: list[tuple[dict[str, Any], float]]
    #: Why the loop parked, or ``None`` if it did not. The loop has always known
    #: this (``AutonomousLoop.park_reason``); until it was carried out here the
    #: only surface that could see it was the event stream, which dies with the
    #: process. A parked run reports ``final_state == "STOPPED"`` exactly like a
    #: clean stop, so without this field a cron wrapper cannot tell a campaign
    #: that finished from one that gave up at 3 a.m. — and the CLI returned 0 for
    #: both. Defaulted so every existing constructor keeps working.
    park_reason: str | None = None


# ── Checkpoint serialization (P3.2) ──────────────────────────────────────────

#: Spec fields that define *what is being searched*. A resume whose spec differs
#: on any of these is not a continuation of the same experiment, so they form the
#: identity fingerprint rather than the whole spec.
#:
#: **``budget`` is deliberately excluded.** It is a stopping rule, not part of
#: the search definition, and extending it is the normal way to continue a run
#: that stopped at its limit — the resume path explicitly advises doing so. Rates
#: and timings are excluded for the same reason: they may legitimately be
#: re-tuned between sessions without invalidating prior observations. ``seed``
#: *is* included, because it seeds the surrogate's own fitting and changing it
#: would refit the GP differently against the same history.
_SPEC_IDENTITY_FIELDS = (
    "name", "parameter_space", "vol_params", "objective", "optimizer",
    "acquisition", "kappa", "batch", "batch_strategy", "seed",
    "pump_ids", "pcb_name",
)


def campaign_spec_fingerprint(spec: "CampaignSpec") -> str:
    """Stable hash of the search-defining fields of *spec*.

    Resuming a checkpoint under a spec with a different parameter space or
    objective would silently graft one experiment's observations onto another's
    search. Comparing fingerprints makes that detectable instead.

    **T2.4 canonicalization.** The measurement block joins the payload through
    :func:`~softae.core.measurement_spec.measurement_identity`, which contributes
    a key *only* for a non-default modality. So a legacy ``eis_*`` spec and its
    new-form equivalent hash identically, and every spec expressible before T2.4
    hashes to exactly the value it hashed to before — checkpoints written by the
    old code verify against specs loaded by the new code, unchanged. The
    omission is load-bearing, not an oversight: a defaulted
    ``{"modality": "eis"}`` key would have changed the hashed JSON of every
    campaign in existence and made every in-flight checkpoint unresumable.
    """
    payload = {
        f: _jsonable(getattr(spec, f, None)) for f in _SPEC_IDENTITY_FIELDS
    }
    identity = measurement_identity(getattr(spec, "measurement", None))
    if identity is not None:
        payload["measurement"] = _jsonable(identity)
    tuning = optimizer_tuning_identity(spec)
    if tuning is not None:
        payload["optimizer_tuning"] = _jsonable(tuning)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: The seven optimizer-tuning fields T1.3 and T3.1 added. Order is irrelevant
#: (the payload is hashed sorted); membership is what matters.
_OPTIMIZER_TUNING_FIELDS = (
    "decision_rtol", "exclusion_radius",
    "learned_feasibility", "feasibility_strategy", "feasibility_min_filter",
    "feasibility_min_infeasible", "feasibility_min_feasible",
)


def optimizer_tuning_identity(spec: "CampaignSpec") -> dict[str, Any] | None:
    """The tuning fields' contribution to the resume fingerprint, or ``None``.

    **Returns ``None`` — contributing no key at all — for any spec that does not
    mention them**, which is every spec expressible before T3.1. That omission is
    the mechanism, not an oversight: seven defaulted keys in the hashed payload
    would have rehashed every campaign in existence and made every in-flight
    checkpoint unresumable, blaming a parameter space nobody touched. Same rule
    :func:`~softae.core.measurement_spec.measurement_identity` follows for a
    defaulted modality (T2.4), applied to a second batch of fields.

    A campaign that *does* set one is genuinely searching differently — a
    tolerance-widened argmax and a strict one propose different points from the
    same surrogate — so it contributes, and only what it actually set.
    """
    supplied = {
        f: getattr(spec, f)
        for f in _OPTIMIZER_TUNING_FIELDS
        if getattr(spec, f, SPEC_UNSET) is not SPEC_UNSET
    }
    return supplied or None


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def serialize_campaign_spec(spec: "CampaignSpec") -> str:
    """JSON snapshot of a spec for the resume checkpoint.

    Deliberately **not** a full round-trip. Several fields are live Python
    objects — ``prior_mean`` is an arbitrary callable, ``formulation`` /
    ``run_plan`` / ``piezo`` are rich objects — and a lossy reconstruction that
    *looked* complete would be worse than none: a campaign could resume with a
    silently different formulation context.

    So the resume path re-supplies the spec from its own config (the GUI already
    saves one, and P6's CLI loads one), and this snapshot exists to **verify**
    that it is the same experiment: identity fingerprint, the primitive fields
    for inspection, and explicit flags for the parts that must be re-attached.
    """
    # Identity fields plus a few informational ones. `budget` is shown because an
    # operator deciding whether to resume wants to see it, but it is *not* part
    # of the fingerprint — see _SPEC_IDENTITY_FIELDS.
    snapshot_fields = _SPEC_IDENTITY_FIELDS + (
        "budget", "channels", "electrode_capacity", "electrode_start",
        "recipe_name", "two_phase", "eis_preset",
    )
    fields = {f: _jsonable(getattr(spec, f, None)) for f in snapshot_fields}
    # The canonical block, spelled out. `eis_preset` above is the mirror of
    # `measurement.preset`, kept so a snapshot written after T2.4 still reads the
    # way one written before it did.
    fields["measurement"] = spec.measurement.as_dict()
    payload: dict[str, Any] = {
        "fingerprint": campaign_spec_fingerprint(spec),
        "fields": fields,
        # Not reconstructable — the resuming caller must provide these itself.
        "requires": {
            "prior_mean": spec.prior_mean is not None,
            "formulation": spec.formulation is not None,
            "general_formulation": spec.general_formulation is not None,
            "run_plan": spec.run_plan is not None,
            "piezo": spec.piezo is not None,
            "seed_observations": len(spec.seed_observations),
        },
    }
    return json.dumps(payload, sort_keys=True)


# ── Optimizer construction ───────────────────────────────────────────────────

def _site_default(section: str) -> dict[str, Any]:
    """One TOML section, or ``{}`` if it or the config is unavailable.

    Degrading to empty is correct here: an absent section means "no site
    default", which the resolver below turns into the constructor default —
    exactly today's behaviour. A missing config file must not stop a campaign
    over two optional knobs.
    """
    try:
        return dict(getattr(loader, section)())
    except Exception:
        logger.warning("optimizer_site_default_unavailable", section=section,
                       exc_info=True)
        return {}


def resolve_optimizer_tuning(spec: CampaignSpec) -> dict[str, Any]:
    """Settle the seven tuning knobs. **The sole resolution point** (§8).

    Precedence: an explicitly-set ``CampaignSpec`` field beats the TOML site
    default, which beats the constructor default. "Explicitly set" is identity
    against :data:`SPEC_UNSET`, not equality against a value, so a campaign that
    deliberately pins ``decision_rtol = 0.0`` keeps it even under a site default
    that turned it on.

    The coherence condition the restructuring spec asks for is **not** "one
    source" but "one resolution point" — two sources are fine, and are what
    decision (iii) chose; a second *reader* is what drifts.
    """
    toml_opt = _site_default("optimizer_tuning")
    toml_feas = _site_default("feasibility_config")

    def pick(field: str, key: str, section: dict[str, Any], default: Any) -> Any:
        value = getattr(spec, field, SPEC_UNSET)
        if value is not SPEC_UNSET:
            return value
        return section.get(key, default)

    return {
        "decision_rtol": float(
            pick("decision_rtol", "decision_rtol", toml_opt, 0.0) or 0.0),
        "exclusion_radius": pick(
            "exclusion_radius", "exclusion_radius", toml_opt, None),
        "feasibility": FeasibilityConfig(
            enabled=bool(pick("learned_feasibility", "enabled", toml_feas, False)),
            strategy=str(pick("feasibility_strategy", "strategy", toml_feas, "fwa")),
            min_filter=bool(
                pick("feasibility_min_filter", "min_filter", toml_feas, True)),
            min_infeasible=int(pick(
                "feasibility_min_infeasible", "min_infeasible", toml_feas,
                ABSOLUTE_LABEL_FLOOR)),
            min_feasible=int(pick(
                "feasibility_min_feasible", "min_feasible", toml_feas,
                ABSOLUTE_LABEL_FLOOR)),
        ),
    }


def build_optimizer(spec: CampaignSpec) -> BaseOptimizer:
    """Instantiate the optimizer named by ``spec.optimizer``."""
    if not spec.parameter_space:
        raise ValueError("CampaignSpec.parameter_space must be non-empty")
    # Optimizers take a concrete direction, so "auto" is resolved here rather than
    # leaking a sentinel into the surrogate.
    direction, _objective_kind = resolve_direction(spec)
    spec = replace(spec, objective=direction)
    kind = spec.optimizer.lower()
    # Prior-mean informing is a GP-surrogate feature; the enumerative random/grid
    # optimizers ignore it. Surface that rather than silently dropping it.
    if spec.prior_mean is not None and kind != "bayesian":
        logger.warning("prior_mean_ignored", optimizer=kind)
    if kind == "random":
        return RandomSearchOptimizer(
            spec.parameter_space, spec.objective, spec.seed, n_trials=spec.budget
        )
    if kind == "grid":
        return GridSearchOptimizer(
            spec.parameter_space, spec.objective, spec.seed, n_points=spec.budget
        )
    if kind == "bayesian":
        tuning = resolve_optimizer_tuning(spec)
        return BayesianOptimizer(
            spec.parameter_space, spec.objective, spec.seed,
            n_initial=min(5, spec.budget),
            acquisition=spec.acquisition,
            kappa=spec.kappa,
            prior_mean=spec.prior_mean,
            batch_strategy=spec.batch_strategy,
            decision_rtol=tuning["decision_rtol"],
            exclusion_radius=tuning["exclusion_radius"],
            feasibility=tuning["feasibility"],
        )
    raise ValueError(f"unknown optimizer '{spec.optimizer}'")


# ── Per-trial workflow builder (shared deposition engine) ────────────────────

def deposit_step_name(channel: int) -> str:
    """Deposit step name for a channel (stable, so results can be keyed by it)."""
    return f"{_DEPOSIT_STEP}_ch{channel}"


def measure_step_name(channel: int) -> str:
    """EIS measure step name for a channel."""
    return f"{_MEASURE_STEP}_ch{channel}"


def confirmation_step_name(channel: int, attempt: int) -> str:
    """Name of a §3.2 confirmation repeat on *channel* (1-based *attempt*)."""
    return f"confirm_eis_ch{channel}_r{attempt}"


def confirmation_measure_step(
    channel: int, attempt: int, measurement: MeasurementSpec | None = None
) -> WorkflowStep | None:
    """A repeat sweep on the same channel — **a re-read, not a self-test** (§3.2).

    On a gate REJECT of a primary measurement, one or two of these ask whether
    the same reading comes back twice. The rig's asymmetry is the whole argument:
    a repeat costs one short sweep against a well plus an anneal for a fresh
    cast, so confirmation is close to free relative to the label it buys.

    It changes no setting, drives no extra hardware and asserts nothing about the
    instrument — it re-measures the *same well* with the *same* parameters. Under
    decision (vi) that matters: this is label hygiene, never rig diagnosis.

    **The repeat can never enter the objective path**, and gets that for free:
    the step is tagged ``measurement="confirm"``, and
    :func:`is_primary_measurement` requires ``"primary"`` — so it is excluded by
    the T1.5 predicate **with no edit to that predicate**, the same mechanism
    T2.7's ``image`` modality used. A dedicated role rather than reusing
    ``"secondary"``, so a genuine secondary probe and a confirmation retry stay
    distinguishable in the record. ``tags["channel"]`` is still set, so T2.6's
    ``_stamp_sample_uuids`` gives the repeat the same ``sample_uuid`` as the film
    it re-measures: one sample, several measurements.
    """
    from softae.core.measurement_spec import MeasurementSpec as _MSpec

    spec = measurement or _MSpec()
    step = get_modality(spec.modality).build_measure_step(int(channel), spec)
    if step is None:
        return None
    return type(step)(
        name=confirmation_step_name(channel, attempt),
        instrument=step.instrument,
        method=step.method,
        params=dict(step.params),
        timeout_s=step.timeout_s,
        retry=step.retry,
        tags={
            **dict(step.tags),
            "measurement": CONFIRM_MEASUREMENT,
            "confirm_attempt": str(int(attempt)),
        },
    )


def plan_trial(spec: CampaignSpec, params: dict[str, Any]) -> FormulationPlan:
    """Turn a composition suggestion into a :class:`FormulationPlan`.

    The composition→volumes step shared with the GUI: reads the campaign's
    :class:`FormulationContext` and calls :func:`plan_formulation` on the
    suggestion.  Requires ``spec.formulation`` to be set (composition mode).
    """
    if spec.formulation is None:
        raise ValueError(
            "plan_trial requires composition mode (spec.formulation is set)"
        )
    return plan_formulation(params, spec.formulation)


def _trial_volumes(spec: CampaignSpec, params: dict[str, Any]) -> list[float]:
    """Per-pump dispense volumes for a suggestion.

    **Composition mode** (``spec.formulation`` set — the standard path): the
    suggestion's composition axes are turned into per-pump cast volumes by
    :func:`plan_formulation`.  The plan's feasibility is a fail-safe against
    overflowing the well — an infeasible cast (over the per-electrode budget)
    raises :class:`FormulationInfeasibleError` rather than dispensing.

    **General composition mode** (``spec.general_formulation`` set): the suggestion's
    params are mapped to arbitrary ``solve_formulation`` targets and solved over any
    stock set — same feasibility fail-safe.

    **Legacy mode** (all formulation contexts ``None`` — deprecated): the searched
    params ARE the per-pump volumes (identity map).  The shared engine zeroes
    ``deadvols``; since the driver dispenses ``vol + deadvol`` per pump, the
    single-drop path folds each pump's dead volume into the suggested volume here.
    The two-phase path does not (its precondition flush primes the lines).
    """
    if spec.general_formulation is not None:
        gf = spec.general_formulation
        targets = [*gf.build_targets(params),
                   TotalDepositTarget(gf.target_deposition_uL)]
        plan = solve_formulation(
            gf.stocks, gf.catalog, targets,
            pump_assignment=gf.pump_assignment, budget_uL=gf.budget_uL,
            dried_frac=gf.dried_frac,
        )
        if not plan.feasible:
            raise FormulationInfeasibleError(plan, gf.budget_uL)
        return list(plan.per_pump_uL)

    if spec.formulation is not None:
        plan = plan_formulation(params, spec.formulation)
        if not plan.feasible:
            raise FormulationInfeasibleError(plan, spec.formulation.budget_uL)
        return list(plan.per_pump_uL)

    vol_params = spec.resolved_vol_params()
    raw = [float(params.get(p, 0.0)) for p in vol_params]
    if spec.resolved_recipe_name() == "two_phase":
        return raw
    deadvols = list(spec.deadvols) if spec.deadvols else [0.0] * len(vol_params)
    return [v + (deadvols[i] if i < len(deadvols) else 0.0) for i, v in enumerate(raw)]


# ── Overflow pre-flight (scan the whole parameter space before a run) ─────────

def campaign_well_capacity_uL(spec: CampaignSpec) -> float:
    """Per-electrode well capacity (µL) this campaign casts into.

    Prefers an explicit formulation ``budget_uL`` (whatever the feasibility gate
    actually enforces); otherwise reads the board's ``well_capacity_uL`` from the
    PCB config — the same value :func:`run_autonomous_campaign` fills the budget
    from at run time.
    """
    if spec.general_formulation is not None and spec.general_formulation.budget_uL is not None:
        return float(spec.general_formulation.budget_uL)
    if spec.formulation is not None and spec.formulation.budget_uL is not None:
        return float(spec.formulation.budget_uL)
    _pcb_name, pcb = resolve_pcb(spec.pcb_name)
    return float(well_capacity_uL(pcb))


def _trial_total_uL(spec: CampaignSpec, params: dict[str, Any]) -> float:
    """Total cast volume (µL) a suggestion would place in the well.

    Uses the *same* composition→volume path as :func:`_trial_volumes` but reads
    the plan's ``grand_total_uL`` instead of raising on infeasibility — so a
    sweep can *flag* an overflowing region rather than abort at the first point.
    """
    if spec.general_formulation is not None:
        gf = spec.general_formulation
        targets = [*gf.build_targets(params),
                   TotalDepositTarget(gf.target_deposition_uL)]
        plan = solve_formulation(
            gf.stocks, gf.catalog, targets,
            pump_assignment=gf.pump_assignment, budget_uL=gf.budget_uL,
            dried_frac=gf.dried_frac,
        )
        return float(plan.grand_total_uL)
    if spec.formulation is not None:
        return float(plan_formulation(params, spec.formulation).grand_total_uL)
    return float(sum(max(0.0, v) for v in _trial_volumes(spec, params)))


def _trial_stock_volumes(
    spec: CampaignSpec, params: dict[str, Any]
) -> "tuple[dict[str, float], dict[str, Any], Any] | None":
    """Per-stock cast volumes for a suggestion, plus the stocks and catalog.

    ``None`` when the spec casts by raw per-pump volumes rather than by
    composition — there is no stock identity to simulate against, so the twin
    has nothing to say beyond the total the overflow guard already checks.
    """
    if spec.general_formulation is not None:
        gf = spec.general_formulation
        targets = [*gf.build_targets(params),
                   TotalDepositTarget(gf.target_deposition_uL)]
        plan = solve_formulation(
            gf.stocks, gf.catalog, targets,
            pump_assignment=gf.pump_assignment, budget_uL=gf.budget_uL,
            dried_frac=gf.dried_frac,
        )
        return plan.per_stock_uL, gf.stocks, gf.catalog
    if spec.formulation is not None:
        ctx = spec.formulation
        plan = plan_formulation(params, ctx)
        return plan.per_stock_uL, ctx.stocks, ctx.catalog
    return None


def simulate_cast(
    per_stock_uL: dict[str, float],
    stocks: dict[str, Any],
    catalog: Any,
    *,
    pcb_name: str | None = None,
    capacity_uL: float | None = None,
):
    """Run the deposition twin on already-resolved per-stock volumes (P.12).

    Returns a :class:`~softae.core.deposition.WellDepositionResult` — fill
    fractions, wet and dry thickness, overflow — or ``None`` when the board
    declares neither a deposit area nor a capacity.

    **The one engine, entered from either side.** A campaign arrives here from
    :func:`simulate_trial`, which solves a composition suggestion into stock
    volumes first; the HT tab arrives here directly, because its volumes are
    already typed into the matrix. Everything downstream of "which stock, how
    much" — the area, the capacity, the elution split, the drying assumption —
    lives here once, so the two surfaces cannot disagree about what the film is.
    A second implementation would be free to drift on any of those four, the way
    three marshallers drifted before P2.2 collapsed them.

    ``capacity_uL`` overrides the board's own ``well_capacity_uL`` — a campaign
    enforces whatever budget its feasibility gate enforces, which need not be the
    board's brim.
    """
    from softae.core.deposition import (
        WellGeometry,
        carrier_component_keys,
        evaporation_pct,
        simulate_well_deposition,
    )
    from softae.core.formulation import elution_from_stock_volumes
    from softae.core.geometry import deposit_area_mm2

    _name, pcb = resolve_pcb(pcb_name)
    area = deposit_area_mm2(pcb)
    capacity = well_capacity_uL(pcb) if capacity_uL is None else capacity_uL
    if not area or not capacity:
        # A guessed area silently corrupts every thickness derived from it, so
        # report "unavailable" rather than substituting one.
        logger.info("twin_unavailable", area_mm2=area, capacity_uL=capacity)
        return None

    elution = elution_from_stock_volumes(per_stock_uL, stocks, catalog)
    return simulate_well_deposition(
        elution,
        WellGeometry.from_board(area, capacity),
        evaporation_pct(),
        carrier_keys=carrier_component_keys(stocks),
    )


def simulate_trial(spec: CampaignSpec, params: dict[str, Any]):
    """Run the deposition twin for one suggestion (P7.5).

    Returns a :class:`~softae.core.deposition.WellDepositionResult` — fill
    fractions, wet and dry thickness, overflow — or ``None`` when the twin
    cannot speak to this trial (no composition, or a board that declares neither
    a capacity nor a deposit area).

    **The twin was previously reachable only from the Deposition panel**, so a
    campaign could neither predict the film it was about to cast nor record the
    thickness it produced. This is the shared entry point for both.

    The suggestion→volumes half is genuinely campaign-specific (axes, solver,
    feasibility); the volumes→film half is :func:`simulate_cast`, which the HT
    tab enters directly.
    """
    resolved = _trial_stock_volumes(spec, params)
    if resolved is None:
        return None
    per_stock_uL, stocks, catalog = resolved

    return simulate_cast(
        per_stock_uL, stocks, catalog,
        pcb_name=getattr(spec, "pcb_name", None),
        capacity_uL=campaign_well_capacity_uL(spec),
    )


#: Prefix for minted sample identifiers, so a bare value is self-describing
#: wherever it surfaces — an ``attrs`` key, a netCDF file, a database column, a
#: log line pasted into a notebook. T1.7 declined to prefix *campaign* and *run*
#: identifiers because structlog's key already names their kind; this one is
#: different in that it travels **outside** structlog, into payload attrs and
#: SQLite columns where nothing else states what it is.
SAMPLE_UUID_PREFIX = "SAM-"


def mint_sample_uuid() -> str:
    """A fresh identity for one physical sample: ``SAM-<uuid4>``.

    Minted at the **well-consumption event** — one call per (trial, channel),
    because one well holds one sample. Not per trial: a replicate round casts the
    same formulation into four wells, and those are four samples that will dry
    differently, be measured separately and can be discarded independently.
    Sharing an identity across them would make the four indistinguishable in
    exactly the analysis the spine exists to serve.
    """
    return f"{SAMPLE_UUID_PREFIX}{uuid.uuid4()}"


def mint_sample_uuids(channels: Sequence[int]) -> dict[int, str]:
    """``{channel: sample_uuid}`` — one distinct identity per well consumed."""
    return {int(ch): mint_sample_uuid() for ch in channels}


def _record_trial_formulations(
    spec: CampaignSpec,
    batch: list[dict[str, Any]],
    channels: list[int],
    *,
    data_store: Any,
    run_id: str | None,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> None:
    """Persist each trial's cast volumes, predicted thickness and its area (P7.6/P.7).

    Campaigns previously recorded only the DOE row, so a campaign-cast channel
    had no ``formulations`` entry at all — and the analysis tab, which joins on
    ``(run_id, channel)``, therefore had nothing to read a thickness from and
    fell back to a hand-typed ``t``.

    ``sample_uuid_by_channel`` carries the identity minted for each well this
    trial consumes (T2.6). Absent, or missing an entry, writes ``NULL`` — the
    same posture as an unrecorded area: a row with no identity says so rather
    than inventing one that nothing else shares.

    Best-effort: a bookkeeping failure must never stop a cast that is about to
    happen anyway.
    """
    if data_store is None or run_id is None:
        return

    from softae.core.geometry import deposit_area_mm2

    # Re-derived from config rather than read back off ``twin.well.area_mm2``.
    # ``WellGeometry.from_board`` does round-trip the area exactly, so on success
    # either route agrees — but ``simulate_trial`` collapses "no area" and "no
    # capacity" into the same ``None``, so on a decline the area is unrecoverable
    # from the twin even when it was perfectly well known and only the capacity was
    # missing. This records the denominator the row was written under, whether or
    # not the twin ran. Guarded because the whole function is bookkeeping.
    try:
        area_mm2 = deposit_area_mm2(resolve_pcb(spec.pcb_name)[1])
    except Exception:
        logger.warning("trial_deposit_area_unresolved", exc_info=True)
        area_mm2 = None

    for params, channel in zip(batch, channels):
        try:
            volumes = list(_trial_volumes(spec, params))
            twin = simulate_trial(spec, params)
            data_store.record_formulation(
                run_id, int(channel),
                pump0_uL=volumes[0] if len(volumes) > 0 else 0.0,
                pump1_uL=volumes[1] if len(volumes) > 1 else 0.0,
                pump2_uL=volumes[2] if len(volumes) > 2 else 0.0,
                total_uL=float(sum(volumes)),
                predicted_thickness_um=(
                    twin.final_thickness_um if twin is not None else None),
                deposit_area_mm2=float(area_mm2) if area_mm2 else None,
                # 'unavailable' rather than NULL: the twin was asked and had nothing
                # to say, which is a recorded fact. NULL is reserved for rows nobody
                # ever asked on behalf of.
                thickness_method="predicted" if twin is not None else "unavailable",
                sample_uuid=(sample_uuid_by_channel or {}).get(int(channel)),
                notes=f"campaign:{spec.name}",
            )
        except Exception:
            logger.warning("trial_formulation_record_failed",
                           channel=channel, exc_info=True)


def twin_feasibility_fn(spec: CampaignSpec):
    """``(params) -> bool``: does this suggestion fit in a well? (P7.1)

    The **enforcement** half of the guardrail. Its counterpart,
    :func:`preflight_overflow`, stays advisory: it answers "what fraction of the
    declared space is infeasible?", which is how an operator fixes bad bounds.
    Two mechanisms because they serve two purposes — one shapes the search, the
    other explains it.

    Returns ``None`` when the board declares no capacity, so an unconstrained
    board is left unconstrained rather than being handed a filter that admits
    everything and costs a solve per candidate.
    """
    capacity = campaign_well_capacity_uL(spec)
    if not capacity or capacity <= 0:
        return None

    def _feasible(params: dict[str, Any]) -> bool:
        try:
            return _trial_total_uL(spec, params) <= capacity
        except Exception:
            # An unsolvable point is not necessarily an overflowing one. Let it
            # through and fail loudly downstream rather than silently carving a
            # hole in the search space for a reason nobody can see.
            return True

    return _feasible


def preflight_overflow(spec: CampaignSpec, *, steps: int = 5) -> "OverflowSweepResult":
    """Flag well overflow across the campaign's whole parameter space.

    Enumerates ``spec.parameter_space`` (grid-sampled, ``steps`` points per axis)
    and, for each composition point, compares the total cast volume against the
    per-electrode well capacity.  Intended as a *pre-run guard*: it surfaces the
    overflowing sub-region up-front so the user can lower the deposition volume,
    rather than discovering it as a mid-run :class:`FormulationInfeasibleError`.
    Advisory only — it does not mutate the spec or block the run.
    """
    from softae.core.overflow import enumerate_space, sweep_overflow

    capacity = campaign_well_capacity_uL(spec)
    points = enumerate_space(spec.parameter_space, steps=steps)
    return sweep_overflow(points, lambda p: _trial_total_uL(spec, p), capacity)


def _build_deposition_workflow(
    spec: CampaignSpec,
    formulation_by_channel: dict[int, list[float]],
    *,
    catalog: TaskCatalog,
    channels: list[int] | None = None,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> Workflow:
    """Build a per-channel deposition Workflow via the shared engine.

    The single code path behind the replicate (single-suggestion), batched
    (one distinct formulation per channel), and board-placement (explicit
    ``channels``) trials — only the ``formulation_by_channel`` mapping and the
    target ``channels`` differ.  Goes through
    :func:`~softae.core.deposition_recipe.build_deposition_workflow` — the one
    marshaller — with the recipe from :meth:`CampaignSpec.resolved_recipe_name`,
    so this runs the *same* engine and recipes the HT tab runs.  Per-channel electrode
    ``(x, y)`` and pico routing are resolved by the engine; the rate split +
    derived settle come from ``dropcast_plan`` at build time (volumes concrete).
    ``spec.time_scale`` scales the driver dwells for fast mock/demo runs.

    ``sample_uuid_by_channel`` stamps each channel's minted identity onto every
    step that names that channel — deposit *and* measure alike (T2.6). Applied
    here, after the engine has built the workflow, rather than inside the recipe:
    the engine's business is what a cast physically does, and it neither mints
    nor needs identities. See :func:`_stamp_sample_uuids`.
    """
    chans = list(channels) if channels is not None else list(spec.channels)
    # Pump count comes from the actual per-pump volume vector (composition mode:
    # one entry per stock; legacy mode: one per vol_param), never from the
    # parameter-space width — those differ once params are composition axes.
    n_pumps = len(next(iter(formulation_by_channel.values()), []))
    if n_pumps <= 0:
        n_pumps = len(spec.resolved_vol_params())
    ids = list(spec.pump_ids[:n_pumps]) or [0]

    recipe = get_deposition_recipe(spec.resolved_recipe_name())
    # Modality-dispatched step building (T2.5). The registry decides what a
    # measurement step *is* for whatever `spec.measurement` names; this function
    # no longer knows the word "EIS". Reads the measurement block, not the
    # retired `measure_eis` mirror (T2.4).
    #
    # A modality that does not measure per electrode returns None per channel
    # (an analysis-only or whole-board stream, T2.7), which collapses to no
    # measurement steps — deliberately the same shape as `enabled=False`, so the
    # engine and the tag index below need no third case.
    measure_by_channel: dict[int, WorkflowStep] | None = None
    if spec.measurement.enabled:
        _modality = get_modality(spec.measurement.modality)
        _built = {ch: _modality.build_measure_step(ch, spec.measurement)
                  for ch in chans}
        measure_by_channel = {ch: s for ch, s in _built.items() if s is not None} or None

    pcb_name, pcb = resolve_pcb(spec.pcb_name)
    wf = build_deposition_workflow(
        recipe,
        chans,
        formulation_by_channel,
        settings=spec.deposition_settings(pcb=pcb, n_pumps=len(ids)),
        catalog=catalog,
        eis_step_by_channel=measure_by_channel,
        name=f"{spec.name}_trial",
    )
    _stamp_sample_uuids(wf, sample_uuid_by_channel)
    wf.metadata = {
        **wf.metadata,
        "campaign": spec.name,
        "pcb": pcb_name,
        "recipe": recipe.name,
        # Kept for existing consumers; derived from the recipe actually run so it
        # cannot disagree with it.
        "two_phase": recipe.name == "two_phase",
        # Tag-based loop closure (T1.5, SESSION_MAIL #2/#3): the builder is the
        # one place that knows which steps were created to measure, so the
        # {name: tags} index rides on the workflow for the objective extractors
        # to close over. Names stay human-readable labels; selection reads tags.
        _MEASUREMENT_TAGS_KEY: _measurement_step_tags(wf, measure_by_channel),
    }
    return wf


def _stamp_sample_uuids(
    wf: Workflow, sample_uuid_by_channel: Mapping[int, str] | None
) -> None:
    """Tag every channel-bearing step with the identity of the sample it touches.

    Keyed on ``tags["channel"]``, which the deposition engine already writes on
    every per-channel step, so this needs no vocabulary of its own and picks up
    any step the engine adds later. Steps with no channel — the startup flush,
    the head retract, anything board-wide — are left alone: they belong to no
    single sample, and a tag there would be a claim, not a label.

    Present-only, like the router's geometry: a channel with no minted uuid gets
    no tag, so ``tags.get("sample_uuid")`` downstream distinguishes *unidentified*
    from *identified as nothing*.

    Mutates ``wf`` in place because it replaces whole step lists rather than
    individual steps. :class:`WorkflowStep` is copy-on-write
    (:meth:`~softae.workflows.workflow_model.WorkflowStep.with_tags` returns a new
    object), and ``Workflow.resolve_steps`` hands out the *setup* list's own
    objects — so mutating tags through it would edit steps that other references
    also see. Rebuilding the lists keeps that impossible.
    """
    if not sample_uuid_by_channel:
        return

    by_channel = {str(int(ch)): u for ch, u in sample_uuid_by_channel.items() if u}

    def _stamp(step: WorkflowStep) -> WorkflowStep:
        sample_uuid = by_channel.get(str(step.tags.get("channel", "")))
        return step.with_tags(sample_uuid=sample_uuid) if sample_uuid else step

    wf.setup = [_stamp(s) for s in wf.setup]
    wf.loop_steps = [_stamp(s) for s in wf.loop_steps]
    wf.teardown = [_stamp(s) for s in wf.teardown]


def _measurement_step_tags(
    wf: Workflow, measure_by_channel: dict[int, WorkflowStep] | None
) -> dict[str, dict[str, str]]:
    """``{step_name: tags}`` for the measurement steps, as they will actually run.

    Read back from the *built* workflow rather than from the pre-insertion step
    objects: the engine adds tags of its own (e.g. ``purge_window``), and an
    index that disagreed with the executed steps would be a second source of
    truth. Keyed to the names of the steps handed to the engine to measure with
    — which is what scopes the index to measurements (see
    ``_MEASUREMENT_TAGS_KEY``).
    """
    if not measure_by_channel:
        return {}
    names = {s.name for s in measure_by_channel.values()}
    return {s.name: dict(s.tags) for s in wf.resolve_steps() if s.name in names}


def build_trial_workflow(
    spec: CampaignSpec,
    params: dict[str, Any],
    *,
    catalog: TaskCatalog,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> Workflow:
    """Build one concrete trial workflow from a single suggestion.

    Maps a suggestion → per-pump volumes → a per-channel deposition Workflow;
    the suggested formulation is applied to every target channel as replicates.
    Each replicate is a **separate physical sample**, so each carries its own
    ``sample_uuid`` when one is supplied.
    """
    vols = _trial_volumes(spec, params)
    return _build_deposition_workflow(
        spec, {ch: vols for ch in spec.channels}, catalog=catalog,
        sample_uuid_by_channel=sample_uuid_by_channel,
    )


def build_placement_workflow(
    spec: CampaignSpec,
    params_list: list[dict[str, Any]],
    channels: list[int],
    *,
    catalog: TaskCatalog,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> Workflow:
    """Cast q **distinct** suggestions onto q explicit electrodes, measured per channel.

    The general placement builder: ``params_list[i]`` → ``channels[i]``.  Used by
    the batched path (channels = ``spec.channels``) and by board-aware placement
    (channels supplied by the electrode allocator).  Requires
    ``len(params_list) == len(channels)``.
    """
    channels = list(channels)
    if len(params_list) != len(channels):
        raise ValueError(
            f"params ({len(params_list)}) must match channel count ({len(channels)})"
        )
    formulation_by_channel = {
        ch: _trial_volumes(spec, p) for ch, p in zip(channels, params_list)
    }
    wf = _build_deposition_workflow(
        spec, formulation_by_channel, catalog=catalog, channels=channels,
        sample_uuid_by_channel=sample_uuid_by_channel,
    )
    wf.metadata = {
        **wf.metadata,
        "batch": len(channels) > 1,
        "channels": list(channels),
        "batch_params": [dict(p) for p in params_list],
    }
    return wf


def build_batch_trial_workflow(
    spec: CampaignSpec,
    params_list: list[dict[str, Any]],
    *,
    catalog: TaskCatalog,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> Workflow:
    """Batched (q-BO) trial on the fixed ``spec.channels`` (delegates to placement)."""
    return build_placement_workflow(
        spec, params_list, list(spec.channels), catalog=catalog,
        sample_uuid_by_channel=sample_uuid_by_channel,
    )


def build_equilibration_workflow(spec: CampaignSpec) -> Workflow:
    """A one-step hardware equilibration routine, run after a board exchange.

    Defaults to the RH controller's stabilization wait (``equilibration_s``); the
    instrument/method are configurable for a fuller re-equilibration routine.

    ``equilibration_method = "settle"`` names a **sample** criterion, not an
    instrument method — no driver exposes it — so the chamber step falls back to
    ``"wait"``. The two are different equilibrations: this one is the chamber
    recovering from an open lid, and the settle phase is the film drying. A
    campaign that settles still wants the chamber wait after a board exchange.
    """
    method = spec.equilibration_method
    if str(method).strip().lower() == SETTLE_METHOD:
        method = "wait"
    return Workflow(
        name=f"{spec.name}_equilibrate",
        description="Board-exchange equilibration",
        setup=[
            WorkflowStep(
                name="equilibrate",
                instrument=spec.equilibration_instrument,
                method=method,
                params={"timeout": float(spec.equilibration_s)},
                timeout_s=max(120.0, spec.equilibration_s * 2),
            )
        ],
        iterations=1,
        metadata={"source": "equilibration", "campaign": spec.name},
    )


def build_confirmation_workflow(
    spec: CampaignSpec, channel: int, attempt: int
) -> Workflow | None:
    """A one-step workflow re-reading *channel* — §3.2's confirmation sweep.

    One attempt per workflow rather than both in one: a repeat that already
    disagreed has answered the question, and the second sweep is rig time spent
    on a label that can no longer be issued. "Up to 2" is a ceiling, not a quota.
    """
    step = confirmation_measure_step(channel, attempt, spec.measurement)
    if step is None:
        return None
    return Workflow(
        name=f"{spec.name}_confirm_ch{channel}_r{attempt}",
        description="Confirmation re-read (T3.1 §3.2) — no re-cast, no new well",
        setup=[step],
        iterations=1,
        metadata={
            "source": "confirmation_sweep",
            "campaign": spec.name,
            _MEASUREMENT_TAGS_KEY: {step.name: dict(step.tags)},
        },
    )


# ── The EQUILIBRATE phase: stop holding when the measurement stops moving ────
#
# A campaign trial has always cast, held for a fixed time, and measured ONCE.
# Nothing waited for the *sample*. This is the driver that does — the caller
# `SettleTracker` was written for and never had:
#
#     "Deliberately holds no clock and no store: the caller owns the floor and
#      the ceiling, because those are time and this is evidence."
#
# So the split is: `analysis/equilibration.py` decides whether σ has stopped
# moving, and everything below decides when to ask it and when to stop asking.
# No settle criterion is reimplemented here, and none should be.

def settle_step_name(channel: int, round_index: int) -> str:
    """Name of one settle round's sweep on *channel* (0-based *round_index*)."""
    return f"{SETTLE_STEP}_ch{channel}_r{round_index}"


def settle_measure_step(
    channel: int, round_index: int, measurement: MeasurementSpec | None = None
) -> WorkflowStep | None:
    """One settle round's sweep — **the trial's own measurement, taken early**.

    Built from the modality's own ``build_measure_step``, not from a second
    route: a settle round and a measure round must be the *same sweep*, or the
    criterion is judging a different quantity than the campaign records.

    Tagged ``measurement="settle"`` so a round cannot enter the objective by
    itself — the same mechanism ``confirmation_measure_step`` uses, and for the
    same reason. The round the campaign finally scores is chosen explicitly by
    the driver, not by a tag that happens to match.
    """
    from softae.core.measurement_spec import MeasurementSpec as _MSpec

    spec = measurement or _MSpec()
    step = get_modality(spec.modality).build_measure_step(int(channel), spec)
    if step is None:
        return None
    return type(step)(
        name=settle_step_name(channel, round_index),
        instrument=step.instrument,
        method=step.method,
        params=dict(step.params),
        timeout_s=step.timeout_s,
        retry=step.retry,
        tags={**dict(step.tags), "measurement": SETTLE_MEASUREMENT,
              "settle_round": str(int(round_index))},
    )


def build_settle_round_workflow(
    spec: CampaignSpec, channels: Sequence[int], round_index: int
) -> Workflow | None:
    """One round: every channel in scope measured once, no cast and no new well.

    ``None`` when the modality builds no per-electrode measurement step — an
    analysis-only stream has nothing for a settle criterion to watch, and a
    phase that cannot observe must not pretend to have waited.
    """
    steps = [s for s in (settle_measure_step(ch, round_index, spec.measurement)
                         for ch in channels) if s is not None]
    if not steps:
        return None
    return Workflow(
        name=f"{spec.name}_settle_r{round_index}",
        description="Equilibrate round — re-read every channel, no re-cast",
        setup=steps,
        iterations=1,
        metadata={"source": "settle_round", "campaign": spec.name,
                  "settle_round": int(round_index),
                  _MEASUREMENT_TAGS_KEY: {s.name: dict(s.tags) for s in steps}},
    )


def settle_round_fits(
    raws: Mapping[int, Any],
    channels: Sequence[int],
    *,
    thickness_for: Callable[[int], Any] | None = None,
) -> list["RoundFit"]:
    """σ and R₁ per channel for one round — **one entry per channel, always**.

    A channel whose step did not complete comes back as an all-``None``
    :class:`~softae.analysis.equilibration.RoundFit` rather than being dropped,
    exactly as :func:`~softae.analysis.equilibration.load_round_fits` does it: a
    shorter list would read to ``settle_check`` as a smaller board rather than as
    missing evidence.

    **R₁ rides beside σ and is never discarded**, because it is the only thing
    that distinguishes a film that stopped changing from a fit that came to rest
    on the model's R₁ floor. The second reports a constant, and a constant is
    what a settle criterion mistakes for settled.
    """
    from softae.analysis.equilibration import RoundFit

    out: list[RoundFit] = []
    for channel in channels:
        raw = raws.get(int(channel))
        thickness = thickness_for(int(channel)) if thickness_for is not None else None
        report = (None if raw is None else
                  _spectrum_report_from_raw(raw, channel=int(channel),
                                            thickness_um=thickness))
        out.append(RoundFit(channel=int(channel),
                            sigma=_report_sigma(report),
                            r1_ohms=_report_r1(report)))
    return out


def _report_sigma(report: Any) -> float | None:
    """σ from a spectrum report when it is a *value*, else ``None``.

    An upper bound is a legitimate scientific result and not a number: a window
    of bounds is as constant as a railed fit, and would settle just as falsely.
    """
    try:
        if report is None or not report.ok or not report.sigma.is_value:
            return None
        value = float(report.sigma.value)
    except Exception:
        return None
    return value if math.isfinite(value) and value > 0 else None


def _report_r1(report: Any) -> float | None:
    """The fitted R₁, whatever the gates thought of the spectrum around it."""
    fit = getattr(report, "fit", None)
    try:
        r1 = float(getattr(fit, "R1"))
    except (AttributeError, TypeError, ValueError):
        return None
    return r1 if math.isfinite(r1) else None


class RHCeilingEscalation:
    """Counts *consecutive* RH-decided equilibrate phases; parks the run at K.

    Per trial nothing happens — a phase that cannot certify holds to
    ``max_hold_s``, records ``ceiling`` (or ``not_evaluable``) and the campaign
    continues. What is new is across trials: K in a row and the room, not the
    film, is running this campaign, and somebody should be told before another
    twenty electrodes are spent proving it.

    Three transitions, each deliberate:

    ============================================  ==========================
    phase                                         effect
    ============================================  ==========================
    ``rh_limited`` **or** ``rh_unreadable``       increment
    settled, or a ceiling with both false         **reset to zero**
    gate off (``rh_stability_pct is None``)       neither
    ============================================  ==========================

    *Both booleans feed one counter, deliberately.* ``rh_unreadable`` is a dead
    sensor rather than moving humidity, but it is the same shape of problem —
    *the RH channel, not the film, decided this phase* — and it burns a campaign
    faster, since with the gate on it makes every phase ``not_evaluable``. They
    stay separate in the record so the park message can say which occurred.

    *A slow-film ceiling resets.* That is the whole load-bearing distinction: it
    is the most ordinary outcome this system produces, and a counter that
    included it would reach K on healthy campaigns.

    *A non-observation neither increments nor resets.* Zeroing on a phase whose
    gate was off would be the same error as counting it — the room was not
    judged, and that is no more evidence of health than of fault.

    Deliberately holds no loop and no store: *park* is injected, so the whole
    rule is testable against fabricated :class:`SettleOutcome` objects.
    """

    def __init__(
        self,
        *,
        limit: int,
        park: Callable[[str], None],
        on_streak: Callable[[int, "SettleOutcome"], None] | None = None,
        streak: int = 0,
    ) -> None:
        self.limit = max(0, int(limit))
        self.streak = max(0, int(streak))
        self.parked = False
        self._park = park
        self._on_streak = on_streak

    def note(self, outcome: "SettleOutcome") -> bool:
        """Record one phase's verdict; ``True`` when it parked the campaign."""
        if self.parked or outcome.rh_stability_pct is None:
            # A campaign parks once. The loop is already winding down, and a
            # second park would fire a second safe_park and a second CRITICAL
            # alert for one fault.
            return False
        if not (outcome.rh_limited or outcome.rh_unreadable):
            self.streak = 0
            return False
        self.streak += 1
        if self._on_streak is not None:
            self._on_streak(self.streak, outcome)
        if self.limit <= 0 or self.streak < self.limit:
            return False
        cause = ("the humidity moved" if outcome.rh_limited
                 else "the RH channel could not be read")
        self.parked = True
        self._park(
            f"{self.streak} consecutive equilibrate phases decided by the RH "
            f"channel and not by the film ({cause}); every σ since is a claim "
            f"about a room this campaign cannot certify"
        )
        return True


def rh_ceiling_park_after_trials() -> int:
    """K — consecutive RH-decided equilibrate phases that park a campaign.

    Resolved from ``[safety] rh_ceiling_park_after_trials`` **here** rather than
    through :func:`~softae.drivers.contracts.rh_watchdog_config`: that resolver's
    output is splatted straight into ``classify_rh_hold``, which would then need a
    second ``del``-style absorber to swallow a key it has no use for — and
    ``RHHoldWatch`` has no business knowing a campaign-level streak limit.

    Missing or unreadable → :data:`DEFAULT_RH_CEILING_PARK_AFTER_TRIALS`.
    """
    try:
        from softae.config.loader import safety

        value = safety().get("rh_ceiling_park_after_trials")
        return (DEFAULT_RH_CEILING_PARK_AFTER_TRIALS if value is None
                else max(0, int(value)))
    except Exception:
        return DEFAULT_RH_CEILING_PARK_AFTER_TRIALS


def settle_r1_bound_ohms() -> float | None:
    """The R₁ lower bound the campaign's own fits rest against, or ``None``.

    Read off the circuit registry through the one existing spelling of it rather
    than written down again — a bound restated in a second place is a bound that
    will disagree with the fitter after the first edit.
    """
    from softae.analysis.equilibration import r1_lower_bound_ohms

    return r1_lower_bound_ohms(SETTLE_CIRCUIT_MODEL)


@dataclass(frozen=True)
class SettleOutcome:
    """Why an equilibrate phase stopped, and what evidence it stopped on.

    Three states, never two. *"We stopped because it settled"* and *"we stopped
    because time ran out"* are different claims about the sample, and a σ taken
    at ``ceiling`` is a weaker claim than one taken at ``settled`` — downstream
    must be able to see which it has, the same posture ``arc_state`` takes for an
    extrapolated R₁.

    ``ceiling`` is an **ordinary outcome, not a failure**. A film that drifts
    slowly is a normal film; parking an unattended run at 3 a.m. because one
    equilibrated slowly is the failure mode P0–P1 exists to prevent.
    """

    outcome: str
    n_rounds: int
    held_s: float
    participating: list[int] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)
    max_deviation_rel: float | None = None
    noise_floor_rel: float | None = None
    tolerance_achievable: bool | None = None
    endorsement: str = ""
    #: The RH spread this phase's last judged window achieved, and the tolerance
    #: it was judged against. Recorded on **every** phase so the provisional
    #: default self-calibrates from real campaigns at their own q — and so a park
    #: raised by a mis-tuned tolerance is diagnosable at all.
    rh_spread_pct: float | None = None
    rh_stability_pct: float | None = None
    #: The two booleans the campaign-level escalation counts, and **the only
    #: route to it**. They cannot be derived after the fact: this class carries no
    #: reason field, ``SettleTracker.outcome()`` discards the cause, and
    #: :attr:`rh_spread_pct` describes only the last window while the binding one
    #: may have been earlier. Drop them as "nice-to-have" and the escalation
    #: silently stops working while still appearing to be wired.
    rh_limited: bool = False       # would have certified but for moving humidity
    rh_unreadable: bool = False    # the RH channel could not be judged at all

    @property
    def settled(self) -> bool:
        from softae.analysis.equilibration import SETTLE_SETTLED

        return self.outcome == SETTLE_SETTLED

    def as_dict(self) -> dict[str, Any]:
        return {
            "settle_outcome": self.outcome,
            "n_rounds": self.n_rounds,
            "held_s": round(float(self.held_s), 3),
            "participating": list(self.participating),
            "excluded": {str(k): v for k, v in self.excluded.items()},
            "max_deviation_rel": self.max_deviation_rel,
            "noise_floor_rel": self.noise_floor_rel,
            "tolerance_achievable": self.tolerance_achievable,
            "endorsement": self.endorsement,
            "rh_spread_pct": self.rh_spread_pct,
            "rh_stability_pct": self.rh_stability_pct,
            "rh_limited": self.rh_limited,
            "rh_unreadable": self.rh_unreadable,
        }

    def describe(self) -> str:
        """One line an operator can read at the bench.

        The RH cause is named here rather than left in the sidecar: an operator
        reading ``ceiling`` in a log should not have to open ``settle.json`` to
        learn it was the room.
        """
        line = (f"{self.outcome}: {self.n_rounds} round(s) over "
                f"{self.held_s:.0f}s, {len(self.participating)} channel(s) "
                f"participating, {len(self.excluded)} excluded")
        if self.rh_stability_pct is not None:
            spread = ("unreadable" if self.rh_spread_pct is None
                      else f"{self.rh_spread_pct:.2f}%RH")
            line += f", RH spread {spread} vs {self.rh_stability_pct:.2f}%RH"
        if self.rh_limited:
            line += " — the room moved, not the film"
        elif self.rh_unreadable:
            line += " — the RH channel could not be judged"
        return line


async def drive_settle_phase(
    plan: SettlePlan,
    *,
    channels: Sequence[int],
    measure_round: Callable[[int], Awaitable[Mapping[int, Any]]],
    fits_from: Callable[[Mapping[int, Any]], Sequence["RoundFit"]],
    r1_bound_ohms: float | None,
    rh_for_round: Callable[[int], float | None] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    now: Callable[[], float] | None = None,
    on_round: Callable[[int, Any], None] | None = None,
) -> tuple[SettleOutcome, dict[int, Any]]:
    """Hold, measure, ask, stop — and return the round the campaign should score.

    The loop the spec describes, and nothing else:

    1. hold ``min_hold_s`` **before** the first round (the bulk of a cure);
    2. measure every channel in scope every ``round_period_s``;
    3. stop as soon as :class:`SettleTracker` says the trailing window settled;
    4. stop unconditionally at ``max_hold_s``, whatever the tracker says.

    Everything time-shaped is injected (``measure_round`` / ``sleep`` / ``now``)
    so the phase is testable end-to-end against fabricated rounds — no rig, no
    store, no eight-hour wait. Consecutiveness, the two-round floor and the
    evaluable/settled split are **not** re-implemented here: ``SettleTracker``
    already owns all three, and a second copy would be a second thing to get
    wrong.

    Returns the outcome and the **last round's raw results**, which is the
    reading taken closest to equilibrium and therefore the one worth recording.

    ``rh_for_round(index)`` supplies that round's median %RH — the *stability*
    half of the criterion. It is required whenever ``plan.rh_stability_pct`` is
    set, and refused loudly rather than defaulted: ``observe``'s ``None`` means
    "unreadable", so a missing wire would burn every phase to the ceiling
    reporting a dead sensor that is not dead.
    """
    import asyncio
    import time

    from softae.analysis.equilibration import SETTLE_NOT_EVALUABLE, SettleTracker

    if plan.rh_stability_pct is not None and rh_for_round is None:
        raise ValueError(
            f"rh_stability_pct={plan.rh_stability_pct:g} is configured but no "
            f"rh_for_round supplier was wired; every window would report "
            f"'rh_unreadable' and the phase would burn to max_hold_s for a "
            f"reason that is not true. Wire the supplier or set "
            f"rh_stability_pct=None."
        )

    sleep = sleep or asyncio.sleep
    now = now or time.monotonic

    tracker = SettleTracker(
        tol_rel=plan.settle_tol_rel,
        n_rounds=plan.settle_n_rounds,
        min_channels=plan.settle_min_channels,
        # THE correctness detail of this whole change. Without it a board whose
        # fits railed on the R₁ floor reports the same σ every round and settles
        # on round three, under-conditioning the entire campaign.
        r1_bound_ohms=r1_bound_ohms,
        rh_stability_pct=plan.rh_stability_pct,
    )

    start = now()
    deadline = start + float(plan.max_hold_s)
    await sleep(float(plan.min_hold_s))

    last_raws: dict[int, Any] = {}
    n_rounds = 0
    settled_early = False
    while True:
        raws = dict(await measure_round(n_rounds))
        n_rounds += 1
        if raws:
            last_raws = raws
        check = tracker.observe(
            fits_from(raws),
            rh_median_pct=(None if rh_for_round is None
                           else rh_for_round(n_rounds - 1)),
        )
        if on_round is not None:
            on_round(n_rounds - 1, check)
        if tracker.settled:
            settled_early = True
            break
        remaining = deadline - now()
        if remaining <= 0:
            break
        await sleep(min(float(plan.round_period_s), remaining))

    endorsed, endorsement, noise_floor_rel = tracker.endorsement()
    verdict = tracker.outcome(stopped_early=settled_early)
    outcome = SettleOutcome(
        outcome=verdict,
        n_rounds=n_rounds,
        held_s=now() - start,
        participating=tracker.participating,
        excluded=dict(tracker.last.excluded) if tracker.last else {},
        max_deviation_rel=(tracker.last.max_deviation_rel if tracker.last else None),
        noise_floor_rel=noise_floor_rel,
        tolerance_achievable=endorsed,
        endorsement=endorsement,
        rh_spread_pct=tracker.rh_spread_pct,
        rh_stability_pct=tracker.rh_stability_pct,
        # Set explicitly at the moment each was the *binding* constraint, never
        # inferred here from the spread — the binding window may have been an
        # earlier one than the spread describes. `rh_unreadable` is additionally
        # conditioned on the phase actually ending not_evaluable, so a phase that
        # lost the RH channel early and recovered reports the ceiling it earned.
        rh_limited=tracker.rh_blocked_settle,
        rh_unreadable=bool(tracker.rh_unreadable
                           and verdict == SETTLE_NOT_EVALUABLE),
    )
    logger.info("campaign_settle_verdict", channels=list(channels),
                **outcome.as_dict())
    return outcome, last_raws


class DataStoreOutcomeSink:
    """Backs T3.1's :class:`OutcomeSink` with the real ``doe_parameters`` columns.

    This is the T3.1b half of the interim contract: the label engine has always
    written *through* the protocol, so swapping the in-memory stand-in for this
    changed no caller. What it buys is durability — before the columns existed,
    the reason a trial failed lived only in the log stream, which dies with the
    process, so a resumed campaign could not tell a stage timeout from a film
    that never formed.
    """

    def __init__(self, data_store: Any) -> None:
        self._data_store = data_store

    def record_outcome(
        self,
        *,
        run_id: str | None,
        channel: int | None,
        params: Mapping[str, Any],
        outcome: str,
        failure_reason: str | None,
    ) -> None:
        if self._data_store is None:
            return
        self._data_store.update_doe_outcome(
            run_id=run_id, channel=channel, outcome=outcome,
            failure_reason=failure_reason,
        )


# ── Objective extractors ─────────────────────────────────────────────────────

#: Objective direction each extractor requires. Minimising |Z| and maximising σ are
#: the *same* physical goal, so swapping extractors without swapping this would steer
#: a campaign toward the worst conductor on the board.
OBJECTIVE_DIRECTION = {
    "mean_abs_z": "minimize",
    "sigma": "maximize",
}


def objective_kind(settings: Any = None) -> str:
    """What ``[eis] objective`` says — ``"auto"`` (default), ``"sigma"`` or ``"mean_abs_z"``.

    This is the *configured* value, not the resolved one. ``"auto"`` is not an
    objective a campaign can be steered by; :func:`resolve_objective` turns it into
    one using the campaign's mode. Callers wanting a metric want that function.

    Read deliberately *not* from ``[eis] engine``. The σ extractor forces the gated
    physics internally because that is what produces a σ at all, while ``engine``
    governs the analysis tab and the auto-route. Two consumers, two switches.

    Falls back to ``"mean_abs_z"`` only if the settings cannot be read — never a
    silent σ, which would need a thickness the caller may not have.
    """
    try:
        from softae.analysis.eis.settings import eis_settings

        cfg = settings if settings is not None else eis_settings()
        return str(cfg.objective)
    except Exception:
        return "mean_abs_z"


def _probe_params(spec: CampaignSpec) -> dict[str, Any]:
    """A representative mid-range point, for asking the twin a question before a run."""
    probe: dict[str, Any] = {}
    for name, bounds in (spec.parameter_space or {}).items():
        if isinstance(bounds, dict) and "low" in bounds and "high" in bounds:
            probe[name] = (float(bounds["low"]) + float(bounds["high"])) / 2.0
        elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
            probe[name] = (float(bounds[0]) + float(bounds[1])) / 2.0
    return probe


def resolve_objective(spec: CampaignSpec, *, settings: Any = None) -> tuple[str, str]:
    """Pick this campaign's objective. Returns ``(kind, reason)``.

    The two campaign modes want different objectives, and neither is a degraded
    version of the other:

    *Composition mode* — the spec carries a formulation, so the twin knows what was
    cast, can predict a thickness, and **conductivity** is both available and the
    quantity actually wanted.

    *Volume mode* — the spec explores raw pump volumes. Exploration is easier and
    feasibility is native (a volume limit is just a bound), with composition resolved
    post hoc. Without stock identity there is no elution and hence no dry thickness,
    so σ is *impossible* rather than merely absent, and **mean |Z|** is the honest
    objective.

    ``auto`` follows the mode. A pinned value is honoured, or refused if the campaign
    cannot deliver it — never silently swapped, because the two are optimised in
    **opposite directions** and switching one for the other would invert the search.
    """
    from softae.errors import CampaignError

    configured = objective_kind(settings)
    can_sigma = simulate_trial(spec, _probe_params(spec)) is not None

    if configured == "mean_abs_z":
        return "mean_abs_z", "pinned by [eis] objective"

    has_composition = (spec.formulation is not None
                       or spec.general_formulation is not None)
    blocker = (
        "the board declares neither a deposit area nor a well capacity"
        if has_composition else
        "volume mode — the spec carries pump volumes but no composition, so the "
        "twin cannot know what was cast and no dry thickness exists"
    )

    if configured == "sigma":
        if can_sigma:
            return "sigma", "pinned by [eis] objective"
        raise CampaignError(
            f"[eis] objective = 'sigma' needs a per-sample thickness, but {blocker}. "
            f"Give the campaign a formulation context, or set "
            f"[eis] objective = 'auto' (impedance in volume mode) or 'mean_abs_z'."
        )

    if can_sigma:
        return "sigma", "composition mode — the twin can predict a thickness"
    return "mean_abs_z", blocker


@dataclass(frozen=True)
class ThicknessReading:
    """A thickness and **where it came from**, which σ's provenance depends on.

    The value alone is not enough. ``resolve_thickness_cm`` ranks four sources and
    records which was used, and a measured film labelled "predicted" would report a
    confident provenance for a number that never came from the twin — or, worse, the
    reverse.
    """

    um: float
    method: str = "predicted"      # profilometry | target | predicted | dispensed

    def __float__(self) -> float:  # so a legacy float consumer still works
        return float(self.um)


def _thickness_parts(value: Any) -> tuple[float | None, str]:
    """Normalise a lookup result to ``(µm, method)``.

    Accepts a bare float for callers predating :class:`ThicknessReading`; those are
    the twin's prediction, which is what the parameter always meant.
    """
    if value is None:
        return None, "unavailable"
    if isinstance(value, ThicknessReading):
        return float(value.um), value.method
    try:
        return float(value), "predicted"
    except (TypeError, ValueError):
        return None, "unavailable"


def make_thickness_lookup(
    data_store: Any = None,
    run_id: str | None = None,
    *,
    plan_id: str | None = None,
) -> Callable[[int], ThicknessReading | None]:
    """Build ``channel -> (thickness µm, source)`` for the σ objective.

    Two sources, in the precedence :func:`~softae.analysis.eis.geometry.resolve_thickness_cm`
    already declares:

    ``profilometry``
        A real measurement of the film, from the ``measured_thickness`` table. Framework
        §7.3 makes an independent measurement the dominant uncertainty term for the
        geometry route, so it wins whenever one exists.
    ``predicted``
        The deposition twin's ``formulations.predicted_thickness_um`` (P7.6) — computed
        from well geometry and solved stock volumes assuming complete drying.

    Until the thickness harness existed the profilometry tier was **unreachable**: it
    was ranked first and nothing could supply it. Consulting the store here is what
    makes the ranking mean something.

    Returns a lookup yielding ``None`` when nothing was recorded. **That is a result,
    not a failure** — σ divides by thickness, and an invented one silently corrupts
    every objective the campaign is steered by.

    A predicted thickness whose ``deposit_area_mm2`` was never recorded counts as
    "nothing recorded" (P.11). It is a quotient with an unknown denominator: rows
    written either side of the 2026-08-07 area correction differ by 4.676× in the
    same column, and the row does not say which board it came from. Returning it
    would be the very corruption the paragraph above exists to prevent, so it is
    withheld — with a warning, because an operator watching σ go quiet is owed the
    reason.
    """
    if data_store is None or not run_id:
        return lambda _channel: None

    def _lookup(channel: int) -> ThicknessReading | None:
        ch = int(channel)
        measured = None
        getter = getattr(data_store, "thickness_for", None)
        if callable(getter):
            try:
                measured = getter(ch, plan_id=plan_id, run_id=run_id)
            except Exception:
                logger.warning("measured_thickness_lookup_failed", channel=ch,
                               exc_info=True)
        if measured is not None:
            return ThicknessReading(float(measured), "profilometry")

        record_getter = getattr(data_store, "predicted_thickness_record", None)
        if callable(record_getter):
            try:
                record = record_getter(run_id, ch)
            except Exception:
                logger.warning("thickness_lookup_failed", channel=ch, exc_info=True)
                return None
            if record is None:
                return None
            if record.area_mm2 is None:
                # Unknown basis, so no basis. The row's thickness could have been
                # divided by 4.0 mm² or by 18.704 mm² and nothing in it says which;
                # rescaling would invent the answer. Withheld, not corrected.
                logger.warning("thickness_withheld_area_never_recorded",
                               channel=ch, run_id=run_id,
                               reason="deposit_area_mm2 was never recorded for this "
                                      "cast, so the thickness has no known basis")
                return None
            return ThicknessReading(float(record.um), "predicted")

        # A store predating P.11 — typically a test stub with no area concept at
        # all. It keeps today's behaviour rather than being forced to grow one.
        try:
            predicted = data_store.predicted_thickness_um(run_id, ch)
        except Exception:
            logger.warning("thickness_lookup_failed", channel=ch, exc_info=True)
            return None
        if predicted is None:
            return None
        return ThicknessReading(float(predicted), "predicted")

    return _lookup


def resolve_direction(spec: CampaignSpec, *, settings: Any = None) -> tuple[str, str]:
    """The optimisation direction this campaign must use. Returns ``(direction, kind)``.

    The direction is **not a preference** — it is fixed by the metric. Maximise
    conductivity, minimise impedance; they are the same goal expressed two ways.
    ``CampaignSpec.objective`` is therefore ``"auto"`` by default and derived here.

    An explicit ``"maximize"``/``"minimize"`` is honoured only when it agrees with the
    resolved metric. It cannot be allowed to disagree: a campaign optimising the wrong
    sign spends its entire budget hunting the **worst** material on the board while
    every step reports success.
    """
    from softae.errors import CampaignError

    kind, reason = resolve_objective(spec, settings=settings)
    # Modality-dispatched (T2.5): the direction comes from the registered
    # modality's own objective table rather than from this module's map, so a
    # future modality steers by its own metrics. For EIS the two are the same
    # object — the registry derives its directions from OBJECTIVE_DIRECTION
    # rather than re-spelling them — so this changes no campaign's sign today.
    required = get_modality(spec.measurement.modality).objective(kind).direction
    stated = str(getattr(spec, "objective", "auto")).strip().lower()

    if stated in ("", "auto"):
        logger.info("campaign_objective_resolved", objective=kind,
                    direction=required, reason=reason)
        return required, kind

    if stated != required:
        raise CampaignError(
            f"CampaignSpec.objective='{stated}' contradicts the '{kind}' objective "
            f"({reason}), which must be {required}d — lower impedance and higher "
            f"conductivity are the same goal. Use objective='auto' to derive it."
        )
    return required, kind


def _trial_objective_kind(kind: str | None, *, has_thickness: bool) -> str:
    """Settle an extractor's metric for one trial. Never returns ``"auto"``.

    Campaign paths resolve the metric once from the spec (:func:`resolve_direction`)
    and thread it down, so *kind* arrives already settled and this is a pass-through.
    The fallback exists for direct calls — demos, tests, the examples — where there is
    no spec to consult; there, thickness availability *is* the mode signal, because a
    dry thickness exists exactly when the twin knew what was cast.

    Resolved **once per trial and applied to every channel in it**. Deciding per
    measurement would let one replicate score σ and the next mean |Z| inside a single
    average — two metrics with opposite optimisation directions, silently mixed.
    """
    settled = kind if kind is not None else objective_kind()
    if settled in ("", "auto", None):
        return "sigma" if has_thickness else "mean_abs_z"
    return str(settled)


def _scalar_from_eis_raw(raw: Any, *, channel: int = 0,
                         thickness_um: float | None = None,
                         kind: str | None = None) -> float | None:
    """Mean impedance magnitude from one EIS step result, or None if unusable.

    Rejects non-finite results explicitly.  ``np.asarray(None, dtype=float)``
    yields ``array(nan)`` *without raising*, so a missing step result used to
    slip past the ``except`` and return NaN — which, told to the optimizer,
    poisons the whole GP fit rather than just one point.
    """
    if raw is None:
        return None

    # P4.3 quality gate. A rejected measurement returns None and therefore takes
    # the *existing* unmeasured path (P0.1): the well is still recorded as cast,
    # and nothing is told to the optimizer. Reusing that route rather than adding
    # a second rejection mechanism keeps one definition of "unmeasured".
    try:
        from softae.analysis.quality import gate_raw_measurement

        report = gate_raw_measurement(raw)
        if not report.ok:
            logger.warning("objective_rejected_by_quality_gate",
                           reason=report.summary())
            return None
    except Exception:
        # The gate is a safeguard, not a dependency — if it cannot run, fall
        # through to the extractor's own finite-check rather than discarding a
        # measurement on the strength of a broken checker.
        logger.warning("quality_gate_unavailable", exc_info=True)

    try:
        import numpy as np

        arr = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            value = float(np.mean(np.hypot(arr[:, -2], arr[:, -1])))
        else:
            value = float(np.mean(arr))
        legacy = value if np.isfinite(value) else None
    except Exception:
        return None

    # E1.5 — the objective becomes conductivity when the gated engine is selected.
    #
    # mean|Z| is not conductivity and never was: it averages across decades, so it is
    # dominated by the low-frequency end and moves with C_par and electrode blocking
    # as much as with the sample. R1 forbids reading a fixed frequency; averaging the
    # whole sweep is worse.
    #
    # While mean |Z| is the campaign's metric the σ path still runs and logs what it
    # *would* have returned, so the two can be compared over a real campaign before
    # anything is flipped — the same observe-before-acting posture as [purge] actuate.
    sigma = _sigma_from_eis_raw(raw, channel=channel, thickness_um=thickness_um)
    if _trial_objective_kind(kind, has_thickness=thickness_um is not None) == "sigma":
        # No thickness under a σ campaign means *unmeasured*, never mean |Z|: handing
        # a maximiser an impedance for one channel would score the worst conductor in
        # the trial as its best result.
        return sigma
    if sigma is not None or legacy is not None:
        logger.info("eis_objective_shadow", mean_abs_z=legacy, sigma=sigma,
                    msg="σ objective observed; mean|Z| in use")
    return legacy


def _sigma_from_eis_raw(raw: Any, *, channel: int = 0,
                        thickness_um: float | None = None) -> float | None:
    """Conductivity for one EIS step result, or ``None`` if it cannot be claimed.

    Returns ``None`` — the existing *unmeasured* path — when the gates reject the
    spectrum, when the fit is inadmissible, or when σ is only an **upper bound**. A
    bound is a legitimate scientific result but it is not a value, and a campaign that
    consumed one as a number would chase the instrument's own resolution limit.

    Never raises: a broken analysis path must not discard a measurement.
    """
    report = _spectrum_report_from_raw(raw, channel=channel, thickness_um=thickness_um)
    if report is None:
        return None
    try:
        if not report.ok:
            logger.warning("objective_rejected_by_gates",
                           reason=report.quality.summary())
            return None
        if not report.sigma.is_value:
            logger.info("objective_declined_bound", mode=report.sigma.mode,
                        upper_bound=report.sigma.upper_bound,
                        msg="σ is an upper bound, not a value — reported as unmeasured")
            return None
        value = float(report.sigma.value)
    except Exception:
        logger.warning("sigma_objective_unavailable", exc_info=True)
        return None
    return value if math.isfinite(value) and value > 0 else None


def _spectrum_report_from_raw(raw: Any, *, channel: int = 0,
                              thickness_um: float | None = None) -> Any | None:
    """One EIS step result → a ``SpectrumReport``, or ``None`` if it cannot be built.

    **The single raw → physics hop on the campaign path.** σ and R₁ come out of
    the *same* analysis of the *same* spectrum, which is what the settle
    criterion needs: it asks whether σ stopped moving and whether the fit that
    produced it railed on the model's R₁ floor, and those two questions are only
    comparable when one fit answers both.

    Never raises: a broken analysis path must not discard a measurement.
    """
    try:
        import numpy as np

        from softae.analysis.eis.engine import analyze_spectrum
        from softae.analysis.eis.geometry import cell_constant_for_sample
        from softae.analysis.eis_data import EISResult

        arr = np.asarray(raw[0] if isinstance(raw, (list, tuple)) else raw, dtype=float)
        if arr.ndim < 2 or arr.shape[1] < 2:
            return None

        freq = arr[:, 0] if arr.shape[1] >= 3 else np.arange(arr.shape[0], dtype=float)
        eis = EISResult.from_arrays(channel=channel, f=freq,
                                    z_real=arr[:, -2], z_imag_neg=arr[:, -1])

        # Dispatch by source so the cell constant records the right provenance:
        # resolve_thickness_cm takes one keyword per tier and reports which it used.
        um, method = _thickness_parts(thickness_um)
        cell = cell_constant_for_sample(
            **({f"{method}_um": um} if method != "unavailable" else {}),
            re_connection="bridged_by_sample")

        # The rig cast this film across the coplanar gap itself, so material spanning
        # the stripes is a fact of the workflow rather than a guess — and the RE stripe
        # sits between CE and WE, so the control loop is closed by that material (F13).
        # Leaving this at the default made the quadrant gate warn "RE integrity
        # UNVERIFIED, suspect an open control loop" on every autonomous spectrum, which
        # is the one path where the answer is known. A warning that always fires is a
        # warning nobody reads, and it would have masked a genuine open loop.
        #
        # This does NOT assert `re_contact_verified`: cast material spanning the gap is
        # not a confirmation that it wets the reference stripe. R26's precondition stays
        # unmet here, so K_config_factor stays 1.0 — which is also its shipped value.
        # T2.6b — USER RULING, 2026-08-09 (TASKS.md T2.6b; mail [a23]): "the GUI and
        # objective should report the same conductivity; nothing changes about the
        # casting nor measurement between them." `engine` is therefore LEFT UNSET, which
        # is the GUI's own mechanism — `analyze_spectrum` resolves `[eis] engine` when
        # the keyword is omitted, and every migrated GUI fit site omits it too. There is
        # deliberately no second resolver here: reading `eis_settings().engine` and
        # passing it back in would be a copy of the rule that could drift from it.
        #
        # This retires the old `engine="gated"` hardcode. The argument for it was that a
        # BO objective consumes σ as a number to optimise against, so it should always
        # enforce the gates — but with the shipped `[eis] engine = "legacy"` that made
        # the campaign and the screen report different σ for one spectrum, with nothing
        # saying so. The user ruled that one config key governs σ everywhere; flipping
        # `[eis] engine` now moves the objective and the display together.
        return analyze_spectrum(eis, cell=cell, re_connection="bridged_by_sample")
    except Exception:
        logger.warning("sigma_objective_unavailable", exc_info=True)
        return None


def is_primary_measurement(tags: Mapping[str, Any] | None) -> bool:
    """Do these step tags mark a *primary campaign measurement* (objective input)?

    THE selection predicate for tag-based loop closure — every objective
    extractor routes through here so the vocabulary lives in exactly one place
    (agreed with the parallel session: SESSION_MAIL #2 point 3 / #3):

    * ``channel`` must be present, and the channel number is read from it —
      never from the step name. Names are human-readable labels (and file
      naming that uses them is untouched); parsing ``ch(\\d+)`` out of a name
      would also read a channel out of ``geom_drift_repeat_ch3``.
    * ``role`` defaults to ``"sample"`` and must be ``"sample"``. Commissioning
      reuses the ordinary EIS step but tags its role (the geometry series'
      drift repeats carry ``role="drift_repeat"``); scoring one as a trial
      would hand the optimizer a fabricated observation. Keying on the role
      keeps that true even if the parallel session renames the step.
    * ``measurement`` defaults to ``"primary"`` and must be ``"primary"`` —
      pre-wiring for Tier-2 secondary metrology (T2.6): a secondary probe of
      the same sample will tag ``measurement="secondary"`` and be recorded but
      never scored.

    The defaults mean a channel-tagged measurement step with no extra tags is
    IN: an absent tag records "nothing special", not "excluded".
    """
    if not tags or "channel" not in tags:
        return False
    return (str(tags.get("role", "sample")) == "sample"
            and str(tags.get("measurement", "primary")) == "primary")


def _legacy_measure_channel(name: str) -> int:
    """Channel from a ``measure_eis_ch<N>`` label — the retired name protocol.

    Kept ONLY for direct callers that pass no tag index (the campaign always
    supplies one, so the runtime objective path never parses a name). Strict
    suffix parse rather than the old ``ch(\\d+)`` regex; behaviour is identical
    for every name the old protocol actually produced.
    """
    _stem, sep, tail = name.partition("_ch")
    digits = tail[: len(tail) - len(tail.lstrip("0123456789"))]
    return int(digits) if sep and digits else 0


def eis_impedance_objective(
    step_results: dict[str, Any],
    params: dict[str, Any],
    *,
    thickness_for: Callable[[int], float | None] | None = None,
    kind: str | None = None,
    step_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> float | None:
    """Default real objective, aggregated over the trial's EIS measurements.

    Averages the per-channel metric across every primary measurement step
    (replicate electrodes), so multi-channel campaigns get one objective. Which
    metric that is comes from *kind*, resolved once per campaign by
    :func:`resolve_objective`: **conductivity** in composition mode, **mean |Z|**
    in volume mode.

    *step_tags* is the trial's ``{step_name: tags}`` index (T1.5): with it,
    selection is :func:`is_primary_measurement` and the channel is read from the
    tags — the step name is never parsed. Without it (direct callers pinning the
    historical contract) the legacy ``measure_eis_ch*`` name protocol applies
    unchanged; the campaign path always supplies the index.

    *thickness_for* maps a channel to its cast thickness in µm — see
    :func:`make_thickness_lookup`. Without it the σ path has no geometry and reports
    every trial unmeasured, so a campaign wanting σ must supply one.

    Returns ``None`` when nothing usable is present. **``None`` means "not measured"
    and must never be coerced to a number** — this function once returned ``0.0``,
    which the loop told the optimizer as a legitimate observation, making the
    surrogate confident about a point that was never measured. The loop skips a
    ``None`` instead of telling it.
    """
    measured: list[tuple[str, int, Any]] = []  # (step name, channel, raw result)
    if step_tags is not None:
        for name, raw in step_results.items():
            tags = step_tags.get(name)
            if not is_primary_measurement(tags):
                continue
            try:
                channel = int(str(tags["channel"]))
            except (TypeError, ValueError):
                # A malformed channel tag is a wiring bug; dropping the step
                # loudly beats crashing an unattended campaign over it — the
                # loop's unmeasured path (P0.1) absorbs the loss.
                logger.warning("bad_channel_tag", step=name, tags=dict(tags))
                continue
            measured.append((name, channel, raw))
    else:
        measured = [(name, _legacy_measure_channel(name), raw)
                    for name, raw in step_results.items()
                    if name.startswith(_MEASURE_STEP)]
    thickness = {
        name: (thickness_for(channel) if thickness_for is not None else None)
        for name, channel, _raw in measured
    }
    # Settled across the whole trial, so replicates of one suggestion are always
    # averaged in a single metric — see _trial_objective_kind.
    settled = _trial_objective_kind(
        kind, has_thickness=any(t is not None for t in thickness.values()))

    scalars: list[float] = []
    for name, channel, raw in measured:
        value = _scalar_from_eis_raw(
            raw, channel=channel, thickness_um=thickness[name], kind=settled)
        if value is not None:
            scalars.append(value)
    if not scalars:
        return None
    return float(sum(scalars) / len(scalars))


def eis_impedance_objective_for_channel(
    step_results: dict[str, Any],
    channel: int,
    *,
    thickness_for: Callable[[int], float | None] | None = None,
    kind: str | None = None,
    step_tags: Mapping[str, Mapping[str, Any]] | None = None,
) -> float | None:
    """Per-channel EIS objective — the batched (q-BO) analog of the aggregate.

    Reads only *this channel's* primary measurement so each of the q distinct
    suggestions in a batched round is scored against *its own* electrode's
    measurement. Returns ``None`` (never ``0.0``) when that channel produced
    nothing usable — see :func:`eis_impedance_objective` for why that
    distinction matters.

    *step_tags* selects the step exactly as in the aggregate (T1.5): the step
    whose tags pass :func:`is_primary_measurement` AND name this channel,
    whatever the step is called. Without an index the legacy
    ``measure_eis_ch{channel}`` name key applies unchanged.

    *kind* comes from the campaign, exactly as for the aggregate: the q members of a
    batched round are compared against each other by the optimizer, so they must all
    be scored in the same metric.
    """
    if step_tags is None:
        raw = step_results.get(measure_step_name(channel))
    else:
        raw = None
        want = str(channel)
        for name, candidate in step_results.items():
            tags = step_tags.get(name)
            if is_primary_measurement(tags) and str(tags.get("channel")) == want:
                raw = candidate
                break
    if raw is None:
        return None
    thickness = thickness_for(channel) if thickness_for is not None else None
    scalar = _scalar_from_eis_raw(raw, channel=channel, thickness_um=thickness,
                                  kind=kind)
    return float(scalar) if scalar is not None else None


def composition_target_objective(
    target: dict[str, float], *, sharpness: float = 1.0
) -> ObjectiveExtractor:
    """A synthetic objective peaking when suggested params hit ``target``.

    Useful for demos and CI: the loop executes the *real* composite workflow
    each trial, but the objective is a smooth function of the suggested params
    with a known optimum, so convergence is observable and deterministic.
    Higher is better (pair with ``objective="maximize"``).
    """
    def _obj(_step_results: dict[str, Any], params: dict[str, Any]) -> float:
        sq = sum((float(params.get(k, 0.0)) - v) ** 2 for k, v in target.items())
        return float(1.0 / (1.0 + sharpness * sq))

    return _obj


# ── Electrode-board resume (persistent single-use occupancy) ─────────────────

def _resolve_purge_runner(manager: Any) -> Any:
    """Find the host's purge runner, or build a transient one for headless runs.

    The GUI attaches a long-lived runner (it also drives the idle timer); a
    headless campaign has no such host, so one is constructed here around the
    scheduler already attached to the syringe. Either way the *scheduler* is
    shared, so both surfaces bill the same timers — a second scheduler would
    silently double the purge rate.

    **Never returns ``None``.** A rig with no purge schedule is a legitimate
    configuration, not a misconfiguration, so absence resolves to a
    :class:`~softae.core.purge_runner.NullPurgeRunner` and callers may invoke a
    purge unconditionally. Callers that must also decide whether to stand up
    the *machinery* around one ask ``performs_purges``.
    """
    from softae.core.purge_runner import NullPurgeRunner, PurgeRunner

    try:
        syringe = manager.get("syringe")
    except Exception:
        return NullPurgeRunner(reason="the syringe could not be read")

    existing = getattr(syringe, "purge_runner", None)
    if existing is not None:
        return existing

    scheduler = getattr(syringe, "purge_scheduler", None)
    if scheduler is None:
        # Deliberately NOT published onto the syringe: a null cached there would
        # be found by `existing` forever after, so a host that attaches a real
        # scheduler later in the same process could never take effect.
        return NullPurgeRunner()

    runner = PurgeRunner(manager, scheduler)
    syringe.purge_runner = runner
    logger.info("purge_runner_created_for_campaign")
    return runner


async def _prepare_electrode_allocator(
    spec: CampaignSpec,
    data_store: DataStore,
    on_board_check: "BoardCheckFn | None",
    emit: Callable[..., None],
) -> "ElectrodeAllocator | None":
    """Build the allocator, honoring persisted single-use well occupancy.

    Reads the current board's recorded casts (surviving prior sessions).  If the
    board already has casts, asks ``on_board_check`` whether the plate is fresh
    (start a clean new board id), the same one (resume past the used wells), or
    to cancel.  Without a handler the safe headless default is **resume** — never
    silently re-cast.  Returns ``None`` when the operator cancels.
    """
    capacity = int(spec.electrode_capacity)
    start = int(spec.electrode_start)
    board_id = data_store.current_board_id()
    occupied = data_store.occupied_electrodes(board_id)

    if occupied:
        decision = BoardCheck.RESUME
        if on_board_check is not None:
            result = on_board_check(board_id, set(occupied))
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, BoardCheck):
                decision = result
        emit("board_check", board_id=board_id,
             occupied=sorted(occupied), decision=decision.name)

        if decision is BoardCheck.CANCEL:
            return None
        if decision is BoardCheck.FRESH:
            board_id += 1                       # a clean, empty board id
            data_store.set_active_board(board_id)
            occupied = set()
        else:  # RESUME — reuse the same plate, skipping the wells already cast
            free = [e for e in range(start, capacity + 1) if e not in occupied]
            if not free:                        # the board is physically full
                board_id += 1                   # → treat as a fresh board
                data_store.set_active_board(board_id)
                occupied = set()
            else:
                # Hand the occupancy set to the allocator rather than jumping to
                # max(occupied)+1: a channel skipped by error recovery leaves a
                # *free* well behind, and skipping past it wastes a single-use
                # site plus the anneal time already invested in that board.
                emit("resume_free_wells", board_id=board_id, n_free=len(free),
                     first=free[0], last=free[-1])

    return ElectrodeAllocator(
        capacity=capacity, start=start, board_index=board_id,
        occupied=frozenset(occupied),
    )


# ── The hook ─────────────────────────────────────────────────────────────────

async def run_autonomous_campaign(
    spec: CampaignSpec,
    *,
    manager: InstrumentManager | None = None,
    data_store: DataStore | None = None,
    project_dir: str | None = None,
    objective_extractor: ObjectiveExtractor | None = None,
    on_event: EventCallback | None = None,
    approval_fn: ApprovalFn | None = None,
    on_board_exchange: "BoardExchangeFn | None" = None,
    on_board_check: "BoardCheckFn | None" = None,
    resume: bool = False,
    heartbeat_s: float | None = None,
    conditions_poll_s: float | None = None,
) -> CampaignResult:
    """Wire and drive a full autonomous campaign, headlessly.

    Any of ``manager`` / ``data_store`` may be supplied by the caller (e.g. to
    share a live session or inspect the DB afterwards); when omitted, a mock
    manager and a temp-dir DataStore are created and torn down here.

    ``resume=True`` continues ``spec.name``'s saved checkpoint (P3.3) instead of
    starting a fresh search: the optimizer is rebuilt with its history and RNG,
    and the remaining budget is what is left of ``spec.budget``. It raises
    :class:`~softae.core.campaign_resume.ResumeMismatchError` if the checkpoint
    belongs to a different search, and falls back to a normal start when there is
    no checkpoint at all. **Off by default** — silently resuming would make a
    re-run of the same spec continue an old experiment instead of repeating it.

    Returns a :class:`CampaignResult`.  Emits an event dict to ``on_event`` for
    every suggestion, result, state change, and convergence — the observation
    stream an overseeing agent consumes.

    ``heartbeat_s`` and ``conditions_poll_s`` override the ``[campaign]`` config
    cadences for the two run-directory sidecars this campaign publishes; ``None``
    means "whatever the config says", and ``0`` disables that sidecar. They are
    separate knobs on separate clocks deliberately — see
    :class:`~softae.core.campaign_events.ConditionsPublisher`.
    """
    # Resolve the measurement modality FIRST (T2.5) — before a manager connects
    # or a run row is written. The registry supplies this campaign's pre-run
    # hook, objectives and directions; a modality nothing has registered raises
    # UnknownModalityError right here rather than letting a spec asking for
    # `image` silently receive EIS steps and record them as what it asked for.
    # (T2.4 hand-wrote that refusal; `UnknownModalityError` subclasses
    # NotImplementedError, so the contract it established is unchanged.)
    modality = get_modality(spec.measurement.modality)
    # Resolved here for the same reason: a spec that names its settle parameters
    # twice, or asks for `"settle"` without a cure time, is a caller bug and must
    # fail before a run row is written rather than eight hours into a hold.
    settle_plan = spec.settle_plan()

    owns_manager = manager is None
    owns_store = data_store is None

    if manager is None:
        from softae.drivers.factory import create_manager

        manager = create_manager(mock=True)
        await manager.connect_all()
    if data_store is None:
        base = project_dir or os.path.join(
            tempfile.mkdtemp(prefix="softae_autonomous_"), "project"
        )
        data_store = DataStore(base)

    # The durable half of `emit`. Assigned once the run has an identity (a run
    # directory needs a run id); until then narration has nowhere to go, and the
    # only thing before that point is a database insert.
    narrator: CampaignNarrator | None = None
    # The inbound half of the same pair, built once the loop exists. Declared
    # here so the teardown below can close it whatever path the run exits by.
    control: ControlWatcher | None = None
    # The third sidecar: what this process can see of the rig, for a watcher
    # that holds no sessions and therefore cannot look for itself.
    conditions: ConditionsPublisher | None = None

    # Cadences resolved once, here. An explicit kwarg wins; otherwise the
    # `[campaign]` section answers; and if reading the config raises, the value
    # stays `None` and the sidecar's own `open_*` helper applies the shipped
    # default. A monitoring knob does not get to refuse to start a campaign, and
    # the number itself still lives in exactly one place.
    def _cadence(explicit: float | None, accessor: Any) -> float | None:
        if explicit is not None:
            return float(explicit)
        try:
            return float(accessor())
        except Exception:
            logger.warning("campaign_cadence_unreadable", exc_info=True)
            return None

    heartbeat_s = _cadence(heartbeat_s, loader.campaign_heartbeat_s)
    conditions_poll_s = _cadence(conditions_poll_s,
                                 loader.campaign_conditions_poll_s)

    def emit(event_type: str, **payload: Any) -> None:
        # Persisted *before* dispatch, deliberately. `on_event` is arbitrary
        # caller code — a GUI slot, a `print` to a pipe that may be broken — and
        # if it raises, the record of what the campaign was doing when it did
        # must already be on disk. The narrator's own contract is that it never
        # raises, so this cannot reverse the failure direction.
        if narrator is not None:
            narrator.record(event_type, payload)
        if on_event:
            on_event({"type": event_type, **payload})

    run_id: str | None = None
    run_finalized = False
    # Set by `_on_park`: a campaign parks once, so a crash *after* the loop has
    # already parked must not fire a second safe_park and a second CRITICAL
    # alert for one fault.
    loop_parked = False
    # The rig claim is entered once the run has an identity and closed last of
    # all, so it spans everything between — including the gaps between trials
    # where the per-workflow lock used to disappear.
    rig_claim = ExitStack()

    def _finalize_run(status: str) -> None:
        """Close the run row exactly once (idempotent, never raises).

        Without this a campaign leaves ``experiments.status`` at ``'running'``
        forever — including after a crash — so nothing downstream (least of all a
        resumed campaign) can tell whether the run actually completed.
        """
        nonlocal run_finalized
        if run_id is None or run_finalized:
            return
        run_finalized = True
        try:
            data_store.finish_run(run_id, status)
        except Exception:
            logger.warning("finish_run_failed", run_id=run_id, status=status,
                           exc_info=True)

    try:
        config_hash = ""
        try:
            config_hash = loader.config_hash()
        except Exception:
            pass

        run_id = data_store.start_run(
            spec.name,
            mode="autonomous",
            pcb_name=resolve_pcb(spec.pcb_name)[0],
            eis_preset=spec.measurement.preset,
            campaign=spec.name,
            config_hash=config_hash,
        )
        # ── The rig is claimed for the whole campaign, not for each trial ────
        # `WorkflowExecutor.run` takes the lock per workflow and drops it in its
        # `finally`, and one trial is one `executor.run`. So between trials —
        # during the BO fit and suggest, the checkpoint write, the settle sidecar
        # — the lock file did not exist, and anything that asked `read_run_lock()`
        # was told the rig was free while a campaign was mid-round. A lock that is
        # absent for part of every round is worse than no lock, because it is
        # believed.
        #
        # Claiming here makes the answer true for the run's whole length and says
        # *which* campaign holds it: `what` carries `campaign:<name>:<run_id>` and
        # `log_path` the run directory, so a second process can name the owner
        # rather than guess at it. This is for **telling the truth about
        # ownership**, not for refusing anyone — the operator at the bench keeps
        # manual control regardless of what this file says.
        #
        # Sited after `start_run` because the run id is half of that identity, and
        # nothing has actuated yet: the only thing between is a database insert.
        # Sited *before* `run_started` because that event is where a watcher begins
        # watching, and it must not be able to see the rig unclaimed on the very
        # first thing it is told. Simulated rigs stay exempt for the same reason
        # the executor exempts them — a mock run holding the lock turns a dry run
        # into an outage for a real one.
        if not rig_is_simulated(manager):
            rig_claim.enter_context(held_run_lock(
                what=f"campaign:{spec.name}:{run_id}",
                log_path=str(data_store.run_dir(run_id)),
            ))

        # ── The narration stream (stage 3) ──────────────────────────────────
        # `emit` is in-memory dispatch and dies with the process, which is why
        # `settle.json`, the checkpoint and the alert rows exist. Those carry the
        # scientific record; what still died was the *narration* — which mode was
        # resolved, which step was recovered, why it parked — and, worse, any
        # sign of life at all inside a step long enough to matter.
        #
        # Sited immediately before `run_started` so that event is the file's
        # first record: a reader replaying from byte 0 learns the run's identity
        # from the stream itself. Sited after the rig claim for the reason the
        # claim gives above — a watcher must not be able to see the rig
        # unclaimed on the very first thing it is told.
        narrator = open_narrator(
            data_store.run_dir(run_id),
            **({} if heartbeat_s is None else {"heartbeat_s": heartbeat_s}))

        emit("run_started", run_id=run_id, spec=spec.name)

        # Beats on the event loop, not between steps. Sync instrument methods
        # are dispatched through `run_in_executor` (`server/base_instrument.py`),
        # so the loop stays free for the whole of an 8-hour anneal — which is
        # exactly the case a watcher cares about and the one a step-boundary
        # heartbeat cannot serve.
        if narrator is not None:
            narrator.start_heartbeat()

        # ── What the rig looks like, for a watcher that cannot look (stage 5) ─
        # An attached GUI opens no instrument sessions, so it cannot read a
        # temperature this process owns — a read is a serial transaction on a bus
        # this process is using. So the campaign publishes `conditions.json`
        # beside the stream and the GUI renders what it finds.
        #
        # Its own task and its own clock, beside the heartbeat rather than folded
        # into it: the beat cadence is load-bearing for the three-beat staleness
        # verdict, and a shared clock would let a monitoring-comfort knob
        # silently redefine what "wedged" means. The read itself goes to a worker
        # thread, so a contended serial bus can delay a temperature reading and
        # can never delay this loop's `control.json` poll — i.e. never an Abort.
        conditions = open_conditions_publisher(
            data_store.run_dir(run_id), manager=manager,
            **({} if conditions_poll_s is None
               else {"poll_s": conditions_poll_s}))
        if conditions is not None:
            conditions.start()

        # Let the modality prepare whatever its measurement steps will read,
        # before any of them runs (T2.5). For EIS that is the `.mscr` scripts:
        # a measurement step carries only a path, so without this the run
        # measured with whatever an earlier HT/manual session left in the temp
        # directory while recording its own preset as provenance.
        #
        # Which channels is a *campaign* fact, not a modality one, so it is
        # computed here: the union of the active channels and the whole board
        # the electrode allocator may walk onto mid-run. A channel the allocator
        # reaches with no script prepared would measure with stale parameters,
        # and nothing fails until the board advances.
        measure_channels = sorted(set(spec.channels) | set(
            range(spec.electrode_start, (spec.electrode_capacity or 0) + 1)
        ))
        modality.prepare_run(spec.measurement, measure_channels, emit=emit)

        # Per-electrode elution budget rides in from the board: fill the
        # formulation context's budget from the PCB's ``well_capacity_uL`` unless
        # the caller set an explicit one.  This is the seam where the board's
        # physical well capacity reaches the formulation feasibility gate.
        for _fctx in (spec.formulation, spec.general_formulation):
            if _fctx is not None and _fctx.budget_uL is None:
                _pcb_name, _pcb = resolve_pcb(spec.pcb_name)
                cap = well_capacity_uL(_pcb)
                if cap is not None:
                    _fctx.budget_uL = cap
                    emit("budget_from_board", pcb=_pcb_name, well_capacity_uL=cap)
                    logger.info("budget_from_board", pcb=_pcb_name, well_capacity_uL=cap)

        # Resume (P3.3): rebuild the optimizer from the checkpoint rather than
        # starting a fresh search. Seed observations are NOT re-applied — they
        # are already inside the restored history, and re-telling them would
        # double-count prior knowledge on every resume.
        resume_plan = None
        if resume:
            from softae.core.campaign_resume import load_resume_plan

            resume_plan = load_resume_plan(data_store, spec)
            if resume_plan is None:
                emit("resume_no_checkpoint", campaign=spec.name)
                logger.info("resume_no_checkpoint", campaign=spec.name)

        if resume_plan is not None:
            optimizer = resume_plan.optimizer
            emit("resumed", campaign=spec.name, iteration=resume_plan.iteration,
                 n_observations=optimizer.n_trials,
                 remaining_budget=resume_plan.remaining_budget,
                 warnings=list(resume_plan.warnings))
            for _w in resume_plan.warnings:
                logger.warning("resume_warning", campaign=spec.name, detail=_w)
        else:
            optimizer = build_optimizer(spec)

            # Warm-start: feed prior observations to the optimizer before the loop
            # so the surrogate opens with existing knowledge (physically/prior-
            # informed BO). These count toward warm-up, not the run budget.
            for seed_params, seed_value in spec.seed_observations:
                optimizer.tell(dict(seed_params), float(seed_value))
            if spec.seed_observations:
                emit("warm_start", n_seed=len(spec.seed_observations))
                logger.info("optimizer_warm_started",
                            n_seed=len(spec.seed_observations))

        # Twin guardrail (P7.1) — set for resumed runs too, since it is a live
        # callable rebuilt from the spec rather than restored from a checkpoint.
        _feasible = twin_feasibility_fn(spec)
        if _feasible is not None:
            optimizer.feasibility_fn = _feasible
            emit("feasibility_filter_enabled",
                 well_capacity_uL=campaign_well_capacity_uL(spec))

        # T3.1 §9 — the operator must be able to see the learned model steering,
        # or it is an invisible hand on the campaign. Announced beside the HARD
        # filter above deliberately: the two are easy to confuse, and only one of
        # them can refuse a candidate. Nothing is emitted when the layer is off.
        _learned = getattr(optimizer, "feasibility", None)
        if _learned is not None and _learned.config.enabled:
            emit("learned_feasibility_enabled",
                 strategy=_learned.config.strategy,
                 min_feasible=_learned.config.min_feasible,
                 min_infeasible=_learned.config.min_infeasible,
                 min_filter=_learned.config.min_filter,
                 clamp=_learned.config.clamp if _learned.config.min_filter else None)
            logger.info("learned_feasibility_enabled",
                        campaign=spec.name,
                        strategy=_learned.config.strategy)

        # The label engine runs whether or not the LEARNED layer is enabled: its
        # outcome/reason rows are an operator-facing record in their own right,
        # and accruing them while the feature is observed is what makes it
        # possible to turn the feature on with evidence rather than on faith.
        # `model` is None when the layer is off, so nothing is trained.
        label_engine = FailureLabelEngine(
            model=_learned if (_learned is not None and _learned.config.enabled)
            else None,
            sink=DataStoreOutcomeSink(data_store),
            emit=emit,
            data_store=data_store,
            run_id=run_id,
        )

        # The shared deposition engine needs the task catalog (it resolves the
        # recipe's methods from it); build each trial's concrete workflow from
        # the suggestion via the same engine the HT tab runs.
        catalog = TaskCatalog.load_toml(loader.tasks_toml_path())

        # q-batch BO: q = channel count; each round casts q distinct suggestions,
        # one per electrode, scored against that electrode's own EIS.
        channels = list(spec.channels)
        batch_size = len(channels) if spec.batch else 1

        # Tag-index bridge (T1.5): the extractors receive only {step_name: result},
        # but tag-based closure needs each step's tags. The builders are the one
        # place that sees the built workflow, so each trial refreshes this mutable
        # index from the workflow's measurement-step tags — the same bridge
        # pattern as `last_params` below. Refreshed, not accumulated: a stale
        # entry could otherwise resurrect a step that no longer exists.
        trial_step_tags: dict[str, dict[str, str]] = {}

        def _reindex_step_tags(wf: Workflow) -> Workflow:
            trial_step_tags.clear()
            trial_step_tags.update(wf.metadata.get(_MEASUREMENT_TAGS_KEY) or {})
            return wf

        # Sample-identity spine (T2.6). Minted **here**, in the wiring, and not
        # in `AutonomousLoop._advance_iteration`, for two reasons:
        #
        #  1. This is the layer where a well is consumed — the builders are the
        #     only place that knows which channels a trial is about to be cast
        #     onto, which is what "one well, one sample" needs. The loop is
        #     deliberately measurement-agnostic and must not learn the word
        #     "sample" to keep T2.7's analysis-only modalities possible.
        #  2. `_advance_iteration` fires once per *iteration*, and a failed
        #     board-aware round advances it once per allocated channel after the
        #     wells were already spent (T1.1). Minting there would issue
        #     identities on a path where nothing was cast, and none where a
        #     narrowed round cast fewer wells than it advanced.
        #
        # Refreshed per builder call, like `trial_step_tags` above: this maps a
        # channel to the identity of the sample *currently* in it, so a re-cast
        # well must overwrite rather than accumulate.
        trial_sample_uuids: dict[int, str] = {}

        def _mint_for(place_channels: Sequence[int]) -> dict[int, str]:
            trial_sample_uuids.clear()
            trial_sample_uuids.update(mint_sample_uuids(place_channels))
            return dict(trial_sample_uuids)

        def sample_uuid_for(channel: int) -> str | None:
            """The identity minted for *channel* in the round now being cast.

            Handed to the loop so the occupancy row it writes carries the same
            uuid as the formulation row written here — the loop owns that write
            (it is the only thing that knows the allocator's board index), and
            without this it would have no way to learn the identity.
            """
            return trial_sample_uuids.get(int(channel))

        # Formulations are recorded at *build* time, before the workflow runs, so the
        # thickness is on record by the time the objective is extracted. P7.6 wired
        # this into the placement builder only; the single-point and batch paths cast
        # without recording, which left them with no thickness and therefore no σ.
        def workflow_builder(params: dict[str, Any]) -> Workflow:
            # The single-point path casts one suggestion across every channel as
            # replicates, and _record_trial_formulations zips batch against channels
            # — so the params are repeated to give each replicate its own row.
            minted = _mint_for(channels)
            _record_trial_formulations(
                spec, [params] * len(channels), list(channels),
                data_store=data_store, run_id=run_id,
                sample_uuid_by_channel=minted)
            return _reindex_step_tags(
                build_trial_workflow(spec, params, catalog=catalog,
                                     sample_uuid_by_channel=minted))

        def batch_builder(batch: list[dict[str, Any]]) -> Workflow:
            # A round may be narrower than q — the final one shrinks to the budget
            # rather than overrunning it — so cast onto the first len(batch)
            # channels rather than assuming the full set.
            used = list(channels)[: len(batch)]
            minted = _mint_for(used)
            _record_trial_formulations(
                spec, batch, used, data_store=data_store, run_id=run_id,
                sample_uuid_by_channel=minted)
            return _reindex_step_tags(
                build_placement_workflow(spec, batch, used, catalog=catalog,
                                         sample_uuid_by_channel=minted))

        # The metric, its direction and the thickness source are campaign-level facts.
        # Resolve them **once**, here, and thread them into all three extractors — the
        # single-point, batched and placement paths must score identically or the
        # optimizer is comparing trials measured in different units. (Each of these
        # was previously resolved, or omitted, per extractor: the placement path
        # passed no thickness at all and so could never produce a σ.)
        _thickness_for = make_thickness_lookup(data_store, run_id)
        _objective_kind: str | None = None
        _objective: ObjectiveSpec | None = None
        # Only a *registered* modality's objective has a physically required
        # direction. A caller-supplied extractor owns its own sign convention
        # (the synthetic composition-target objective, for instance, is
        # legitimately maximised), so the registry is not consulted for one.
        if objective_extractor is None:
            _direction, _objective_kind = resolve_direction(spec)
            # The extractors below come from the modality (T2.5) rather than
            # being named here, so all three scoring paths use whatever this
            # modality measures with — and, as before, the *same* one, since a
            # mixed trial would average two metrics with opposite directions.
            _objective = modality.objective(_objective_kind)
            emit("objective_resolved", objective=_objective_kind,
                 direction=_direction)

        def batch_extractor(
            step_results: dict[str, Any], index: int, params: dict[str, Any]
        ) -> float:
            # A custom (results, params) objective is params-based, so it applies
            # per batch member directly; the default scores each member against
            # its own channel's EIS measurement.
            if objective_extractor is not None:
                return objective_extractor(step_results, params)
            return _objective.channel_extractor(
                step_results, channels[index],
                thickness_for=_thickness_for, kind=_objective_kind,
                step_tags=trial_step_tags)

        if spec.batch and batch_size > 1:
            emit("batch_mode", q=batch_size, channels=channels)

        if settle_plan is not None:
            emit("settle_mode", **{f: getattr(settle_plan, f)
                                   for f in _SETTLE_FIELDS})
            # Said once, at the start, rather than discovered eight hours in: a
            # board narrower than `settle_min_channels` can never make the
            # criterion evaluable, so every trial would run to its ceiling. That
            # is the SAFE direction and the run proceeds — but silently spending
            # `max_hold_s` per trial for a reason nobody can see is not.
            if len(channels) < settle_plan.settle_min_channels:
                emit("settle_unevaluable_board", channels=len(channels),
                     settle_min_channels=settle_plan.settle_min_channels)
                logger.warning(
                    "settle_min_channels_exceeds_board", campaign=spec.name,
                    channels=len(channels),
                    settle_min_channels=settle_plan.settle_min_channels,
                    detail="fewer channels than the criterion needs; every trial "
                           "will run to max_hold_s and record 'not_evaluable'")

        # Electrode/board management: when a capacity is set, electrodes are
        # single-use and allocated sequentially; a full board triggers a prompted
        # exchange + equilibration. The placement builder casts onto the allocator's
        # explicit channels (superseding spec.channels' fixed values).
        allocator = None
        if spec.electrode_capacity is not None:
            allocator = await _prepare_electrode_allocator(
                spec, data_store, on_board_check, emit
            )
            if allocator is None:  # operator cancelled at the board-freshness check
                emit("run_finished", run_id=run_id, best=None, n_trials=0)
                return CampaignResult(
                    run_id=run_id, best_params=None, best_objective=None,
                    n_trials=0, final_state="STOPPED", converged=False, history=[],
                )
            emit("electrode_mode", capacity=spec.electrode_capacity,
                 start=allocator.start, board_id=allocator.board_index,
                 batch_size=batch_size)

        def placement_builder(
            batch: list[dict[str, Any]], place_channels: list[int]
        ) -> Workflow:
            minted = _mint_for(place_channels)
            _record_trial_formulations(
                spec, batch, place_channels, data_store=data_store, run_id=run_id,
                sample_uuid_by_channel=minted)
            return _reindex_step_tags(build_placement_workflow(
                spec, batch, place_channels, catalog=catalog,
                sample_uuid_by_channel=minted))

        def placement_extractor(
            step_results: dict[str, Any], channel: int, params: dict[str, Any]
        ) -> float:
            if objective_extractor is not None:
                return objective_extractor(step_results, params)
            return _objective.channel_extractor(
                step_results, channel,
                thickness_for=_thickness_for, kind=_objective_kind,
                step_tags=trial_step_tags)

        def equilibration_builder() -> Workflow:
            return build_equilibration_workflow(spec)

        # Warn-and-proceed maturity guard: surface (but don't block) any method
        # this campaign will run that hasn't reached the expected maturity.  Scan
        # a representative concrete trial (nominal unit volumes).
        try:
            from softae.core import lifecycle as _lc

            nominal = {p: 1.0 for p in spec.resolved_vol_params()}
            for w in _lc.maturity_warnings(
                workflow_builder(nominal), catalog, expected=spec.expected_maturity
            ):
                logger.warning("method_below_maturity", **w)
                emit("maturity_warning", **w)
        except Exception:
            logger.warning("maturity_check_skipped", exc_info=True)

        # Adapt this module's (results, params) extractors to the loop's
        # single-arg signature by capturing the latest suggestion. The metric was
        # resolved above, alongside the batched and placement extractors.
        if objective_extractor is not None:
            obj = objective_extractor
        else:
            def obj(step_results: dict[str, Any], params: dict[str, Any]):
                return _objective.extractor(
                    step_results, params,
                    thickness_for=_thickness_for, kind=_objective_kind,
                    step_tags=trial_step_tags)
        last_params: dict[str, Any] = {}

        def loop_extractor(step_results: dict[str, Any]) -> float:
            return obj(step_results, last_params)

        async def _run_confirmation(channel: int, attempt: int) -> Any:
            """Execute one repeat sweep and return its raw result, or ``None``.

            A nested executor is sound here: the run lock is re-entrant per
            process (a workflow that nests another is unaffected), and the outer
            executor has already returned by the time this runs.
            """
            wf = build_confirmation_workflow(spec, channel, attempt)
            if wf is None:
                return None
            from softae.workflows.workflow_executor import WorkflowExecutor

            captured: dict[str, Any] = {}
            executor = WorkflowExecutor(manager, data_store=data_store,
                                        run_id=run_id)
            executor.on_step_complete = (
                lambda step, idx, total, result, *_: captured.update(
                    {step.name: result}))
            await executor.run(wf)
            # Keep the tag index in step with what ran, so anything reading tags
            # sees the confirm marking rather than an unknown step name.
            trial_step_tags.update(wf.metadata.get(_MEASUREMENT_TAGS_KEY) or {})
            return captured.get(confirmation_step_name(channel, attempt))

        async def _confirm_and_label(
            step_results: dict[str, Any]
        ) -> dict[str, Any]:
            """Gate each primary measurement; confirm rejects; derive labels.

            Runs BEFORE the objective is extracted, which is the only point where
            the raw results, the channel and an awaitable context coexist.

            **The gate's REPORT is what is read, never its post-gate verdict.**
            With the shipped ``[quality] enabled = false`` a would-be REJECT comes
            back as SUSPECT plus "gate disabled" and the measurement is still
            used — so keying on the verdict would make the richest label source
            inert exactly while the gate is being observed. Nothing here flips
            that switch or grants the gate any authority it does not have: the
            objective path is untouched, and a rejected-but-enabled measurement
            takes the same unmeasured route it takes today.
            """
            from softae.analysis.quality import gate_raw_measurement

            board_id = None
            try:
                board_id = data_store.current_board_id()
            except Exception:
                logger.debug("board_id_unavailable_for_labels", exc_info=True)

            # Snapshot: the confirmation runs mutate `trial_step_tags`, and a
            # dict cannot be iterated while it grows.
            primaries = [
                (name, raw) for name, raw in list(step_results.items())
                if is_primary_measurement(trial_step_tags.get(name))
            ]

            # TWO PASSES, and the split is load-bearing. Condition 3 asks whether
            # anything else on this board measured *during this trial* — a
            # property of the whole round, not of dict order. Gathering accepts
            # and judging rejects in one pass would make the label depend on
            # which channel numpy happened to enumerate first: ch21 judged before
            # ch22's ACCEPT was known would be withheld, and the same evidence in
            # the other order would label. Same trial, two answers.
            gated: list[tuple[int, Any]] = []
            for name, raw in primaries:
                tags = trial_step_tags.get(name) or {}
                try:
                    channel = int(str(tags["channel"]))
                except (KeyError, TypeError, ValueError):
                    continue
                report = gate_raw_measurement(raw)
                gated.append((channel, report))
                if reject_signature(report) is None and report.ok:
                    # Corroborates the board (condition 3). The *feasible* label
                    # is still only issued on the tell path — a spectrum that
                    # gates cleanly has not yet earned one.
                    label_engine.note_accept(channel=channel, board_id=board_id)

            for channel, report in gated:
                if reject_signature(report) is None:
                    continue

                # Up to 2 repeats, stopping at the first disagreement — the
                # question is already answered by then.
                confirmations: list[Any] = []
                for attempt in range(1, MAX_CONFIRMATION_SWEEPS + 1):
                    repeat_raw = await _run_confirmation(channel, attempt)
                    step_results[confirmation_step_name(channel, attempt)] = (
                        repeat_raw)
                    repeat_report = gate_raw_measurement(repeat_raw)
                    confirmations.append(repeat_report)
                    if reject_signature(repeat_report) != reject_signature(report):
                        break

                label_engine.record_gate_reject(
                    params=dict(last_params), channel=channel,
                    board_id=board_id, primary_report=report,
                    confirmations=confirmations,
                )

            return step_results

        # ── The EQUILIBRATE phase (opt-in: equilibration_method = "settle") ──
        #
        # Sited on `on_trial_measured` because that is the one hook where the
        # trial's raw results, the channels they came from and an awaitable
        # context coexist. The trial's own sweep has already run by then and is
        # kept on record as the pre-equilibration reading; the round the campaign
        # SCORES is the last settle round, injected below under the trial's own
        # measure-step name. That injection is the whole point: an optimiser fed
        # a σ off a still-drying film cannot tell a better formulation from one
        # that was measured later.
        settle_verdicts: list[dict[str, Any]] = []

        def _primary_channels(step_results: dict[str, Any]) -> dict[int, str]:
            """``{channel: step name}`` for this trial's objective-bearing sweeps.

            Read off the tag index rather than from ``spec.channels``: a narrowed
            final round, and every board-aware round, casts onto fewer or other
            electrodes than the spec names, and settling channels that were never
            cast would hold on wells that hold nothing.
            """
            found: dict[int, str] = {}
            for name in step_results:
                tags = trial_step_tags.get(name)
                if not is_primary_measurement(tags):
                    continue
                try:
                    found[int(str(tags["channel"]))] = name
                except (KeyError, TypeError, ValueError):
                    continue
            return found

        #: Wall-clock bounds of each settle round, so the round's own RH can be
        #: read back off the ``conditions`` rows it wrote. Keyed by round index.
        settle_round_window: dict[int, tuple[str, str]] = {}

        async def _settle_round(
            round_channels: list[int], round_index: int
        ) -> dict[int, Any]:
            """One round: re-read every channel in scope. No cast, no new well."""
            wf = build_settle_round_workflow(spec, round_channels, round_index)
            if wf is None:
                return {}
            from softae.core.data_store import _now_iso
            from softae.workflows.workflow_executor import WorkflowExecutor

            # One sample, several measurements: the rounds re-read the films the
            # trial just cast, so they carry those films' identities.
            _stamp_sample_uuids(wf, trial_sample_uuids)
            captured: dict[str, Any] = {}
            executor = WorkflowExecutor(manager, data_store=data_store,
                                        run_id=run_id)
            executor.on_step_complete = (
                lambda step, idx, total, result, *_: captured.update(
                    {step.name: result}))
            started = _now_iso()
            try:
                await executor.run(wf)
            finally:
                # Bounded even on failure: a round that died halfway still wrote
                # whatever conditions rows it got to, and those are the honest
                # evidence of what the room did while it ran.
                settle_round_window[int(round_index)] = (started, _now_iso())
            return {ch: captured.get(settle_step_name(ch, round_index))
                    for ch in round_channels}

        #: The RH the room actually held during one settle round. Rows only —
        #: **no new instrument read**: the EIS router already snapshots the
        #: environment at measurement time, and it routes on ``step.method``,
        #: which a settle round copies verbatim from the modality's own measure
        #: step. The ``measurement="settle"`` tag governs objective eligibility,
        #: never routing, so a settle round records a ``conditions`` row exactly
        #: as a measure round does.
        _SETTLE_RH_SQL = (
            "SELECT rh_pv_pct FROM conditions "
            "WHERE run_id = ? AND timestamp BETWEEN ? AND ? "
            "AND rh_pv_pct IS NOT NULL"
        )

        def _settle_round_rh(round_index: int) -> float | None:
            """This round's median %RH, or ``None`` if the room was not observed.

            Both absences collapse to ``None``, and must: no row at all (capture
            found nothing to write) and a row with a NULL ``rh_pv_pct`` (the RH
            controller unreadable while the stage thermometer answered) are
            equally *the room was not observed*. **A round with no RH reading is
            not a round with stable RH** — the window it falls in becomes
            non-evaluable rather than silently passing.
            """
            from softae.analysis.equilibration import round_rh_median

            bounds = settle_round_window.get(int(round_index))
            if bounds is None:
                return None
            try:
                rows = data_store._conn.execute(
                    _SETTLE_RH_SQL, (run_id, bounds[0], bounds[1])).fetchall()
            except Exception:
                logger.warning("settle_rh_readback_failed", run_id=run_id,
                               settle_round=round_index, exc_info=True)
                return None
            return round_rh_median([row[0] for row in rows])

        def _record_settle(outcome: SettleOutcome, round_channels: list[int]) -> None:
            """Say which of the three outcomes this was — and keep saying it.

            The event stream dies with the process, so the verdict also goes to a
            sidecar beside the run: a σ taken at ``ceiling`` is a weaker claim
            than one taken at ``settled``, and a reader months later must be able
            to tell which one the campaign recorded.
            """
            payload = {"iteration": loop.iteration,
                       "channels": list(round_channels), **outcome.as_dict()}
            settle_verdicts.append(payload)
            emit("settle_verdict", **payload)
            try:
                path = data_store.run_dir(run_id) / "settle.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(settle_verdicts, indent=2),
                                encoding="utf-8")
            except Exception:
                logger.warning("settle_sidecar_failed", run_id=run_id, exc_info=True)

        def _park_on_rh_streak(reason: str) -> None:
            """Stop the campaign, and mean it.

            **Park, never raise.** A ``SafetyError`` here would be classified a
            hard fault by ``AutonomousLoop._is_hard_fault`` and abort the trial
            immediately — the mid-batch stop this whole design exists to avoid.
            ``_park`` records the reason, fires ``on_park`` (safe_park plus a
            durable CRITICAL alert) and asks for ``LoopState.STOPPED``.
            """
            loop._park(reason)
            # Asking is not enough, and this second line is not belt-and-braces.
            # This runs inside `AutonomousLoop._post_measure`, and the very next
            # statement on every round path is `_set_state(ANALYZING)`, which
            # overwrites STOPPED — the loop would otherwise run on. Every
            # *existing* `_park` call site is followed by an immediate `break`;
            # this one has no such lever, so it closes the budget instead and the
            # loop's own top-of-round check ends the run at this trial boundary,
            # with the current batch cast, measured and recorded. (The tidier fix
            # belongs in AutonomousLoop, whose `_set_state` should refuse to leave
            # a terminal state; that file is not this change's to edit.)
            loop._max_iterations = loop.iteration

        rh_escalation = RHCeilingEscalation(
            limit=rh_ceiling_park_after_trials(),
            park=_park_on_rh_streak,
            on_streak=lambda streak, outcome: emit(
                "rh_ceiling_streak", streak=streak, limit=rh_escalation.limit,
                rh_limited=outcome.rh_limited,
                rh_unreadable=outcome.rh_unreadable,
                rh_spread_pct=outcome.rh_spread_pct),
        )

        async def _equilibrate(step_results: dict[str, Any]) -> dict[str, Any]:
            """Hold until σ stops moving, then hand back the settled reading.

            **A ``ceiling`` returns normally.** It is an ordinary outcome for a
            slowly-drifting film, not a fault, and nothing here raises, parks or
            withholds the measurement on one — parking an unattended run at 3 a.m.
            because a film equilibrated slowly is the failure mode P0–P1 exists
            to prevent. What the ceiling changes is the *claim*, and the claim is
            recorded.
            """
            targets = _primary_channels(step_results)
            if not targets:
                return step_results
            round_channels = sorted(targets)
            outcome, last_raws = await drive_settle_phase(
                settle_plan,
                channels=round_channels,
                measure_round=lambda i: _settle_round(round_channels, i),
                fits_from=lambda raws: settle_round_fits(
                    raws, round_channels, thickness_for=_thickness_for),
                r1_bound_ohms=settle_r1_bound_ohms(),
                rh_for_round=_settle_round_rh,
            )
            for channel, name in targets.items():
                raw = last_raws.get(channel)
                if raw is not None:
                    step_results[name] = raw
            _record_settle(outcome, round_channels)
            # Last, and it matters: the park this may raise must land *after* the
            # batch is measured and its verdict written, which is what siting the
            # check here — on `on_trial_measured`, after `_record_settle` — buys
            # for free. Nothing in flight is lost.
            rh_escalation.note(outcome)
            return step_results

        async def _equilibrate_then_label(
            step_results: dict[str, Any]
        ) -> dict[str, Any]:
            """Settle first, gate second — the gate must judge the recorded sweep."""
            return await _confirm_and_label(await _equilibrate(step_results))

        loop = AutonomousLoop(
            optimizer=optimizer,
            workflow_template=None,
            manager=manager,
            data_store=data_store,
            run_id=run_id,
            objective_extractor=loop_extractor,
            workflow_builder=workflow_builder,
            auto_approve=spec.auto_approve and approval_fn is None,
            max_iterations=spec.budget,
            batch_size=batch_size,
            batch_workflow_builder=batch_builder if batch_size > 1 else None,
            batch_objective_extractor=batch_extractor if batch_size > 1 else None,
            batch_channels=channels if batch_size > 1 else None,
            electrode_allocator=allocator,
            placement_workflow_builder=placement_builder if allocator else None,
            placement_objective_extractor=placement_extractor if allocator else None,
            on_board_exchange=on_board_exchange if allocator else None,
            equilibration_builder=equilibration_builder if allocator else None,
            track_occupancy=allocator is not None,
            sample_uuid_for=sample_uuid_for,
        )

        if resume_plan is not None:
            # Continue the iteration count rather than restarting it. With
            # max_iterations kept absolute (spec.budget), the loop's own budget
            # check then stops at the right total, and checkpoint iteration
            # numbers stay monotonic across restarts instead of rewinding.
            loop._iteration = resume_plan.iteration
            # ── The two escalation streaks on resume ─────────────────────────
            # Both are per-campaign and both are persisted. Neither parks
            # anything until it reaches its limit, so a restart can land
            # mid-streak, and how a resume treats the count decides whether a
            # chronic fault can ever escalate.
            #
            # This has now been argued three ways, and the first two were each
            # half right:
            #
            #   1. *Always zero* — "a resume follows a park an operator has just
            #      been to the rig to address, so the slate is genuinely clean."
            #      True of a park. False of a crash or an overnight shutdown,
            #      which acknowledge nothing; and in a crash-restart loop the
            #      counter could never reach its limit at all.
            #   2. *Always persist* — fixes the crash case and breaks the park
            #      case, handing the operator who *did* fix the rig a run that
            #      parks again on its first unlucky trial.
            #   3. *Conditional on how the previous run stopped* — this. A stop
            #      that unwound far enough to write a terminal status is a stop
            #      something reported: a park raises a CRITICAL alert, an
            #      operator stop is the operator's own hand. A row left `error`
            #      or `interrupted`, or never closed at all, is nobody telling
            #      anybody anything.
            #
            # The discriminator is the previous run's row, reached through the
            # run id on the checkpoint (`_previous_exit_was_acknowledged`).
            #
            # **Known limitation, stated rather than papered over.** "The run
            # exited cleanly" is a proxy for "the operator acknowledged the
            # park", and it is not the same claim: an operator can resume a
            # parked campaign from the CLI without having gone near the rig, and
            # this will clear their streak. The truer signal would be an
            # acknowledged flag on the park alert — but `alerts` has no such
            # column and no API that writes one (id / raised_at / run_id / kind /
            # severity / message / details, and that is all), so the honest
            # options were this proxy or inventing an ack system as a side effect
            # of a counter fix. Should an ack state ever exist, this is the call
            # site that should consume it.
            #
            # Being wrong here is bounded in the direction that matters: a
            # wrongly-cleared streak costs at most `park_after_failed_trials`
            # further wells before the fault parks the run again, and a genuinely
            # fixed rig clears the streak on its first measured trial anyway via
            # `_note_trial_success`. Unknown provenance therefore resolves to
            # *preserve*, not clear.
            #
            # **The RH ceiling streak deliberately does NOT take this branch.**
            # `AutonomousLoop._note_trial_failure`'s comment says the three
            # counters should share their treatment on resume; that is no longer
            # true, and this is the exception with a reason. The operator's
            # ruling was about the trial-failure streak specifically, and
            # `test_the_rh_ceiling_streak_survives_a_checkpoint_round_trip` pins
            # the RH streak surviving a park on its own argument — an RH ceiling
            # is a slowly developing environmental condition, and a park for some
            # *other* fault is not evidence anybody addressed it. Restoring it
            # unconditionally stays correct until someone rules otherwise.
            _saved = data_store.campaign_checkpoint(spec.name) or {}
            rh_escalation.streak = int(_saved.get("rh_ceiling_streak") or 0)
            _streak = int(_saved.get("consecutive_failures") or 0)
            _acknowledged, _why = _previous_exit_was_acknowledged(
                data_store, resume_plan.run_id or _saved.get("run_id"))
            loop._consecutive_failures = 0 if _acknowledged else _streak
            # Say which way it went, and why. A counter that silently changes
            # value across a restart is precisely what made this hard to see:
            # both of the earlier behaviours were invisible at runtime, so the
            # only way to know which one a given resume had applied was to read
            # the source it happened to be running.
            logger.info(
                "resume_failure_streak",
                campaign=spec.name,
                action="cleared" if _acknowledged else "restored",
                saved=_streak,
                consecutive_failures=loop._consecutive_failures,
                rh_ceiling_streak=rh_escalation.streak,
                why=_why,
            )
            emit("resume_failure_streak",
                 action="cleared" if _acknowledged else "restored",
                 saved=_streak, consecutive_failures=loop._consecutive_failures,
                 why=_why)

        def on_suggestion(iteration: int, params: dict[str, Any]) -> None:
            last_params.clear()
            last_params.update(params)
            # `steered` is the money field (§9): "did the learned model change the
            # answer?" is the only question an operator actually has. Both keys
            # are added ONLY when the layer actually weighted this proposal, so a
            # default-off campaign's event payload is unchanged.
            extra: dict[str, Any] = {}
            p_feas = getattr(optimizer, "last_p_feas", None)
            if p_feas is not None:
                extra["p_feas"] = p_feas
                extra["steered"] = bool(getattr(optimizer, "last_steered", False))
            emit("suggestion", iteration=iteration, params=dict(params), **extra)

        loop.on_suggestion = on_suggestion

        def _on_result(i: int, p: dict[str, Any], o: float) -> None:
            emit("result", iteration=i, params=dict(p), objective=o)
            # THE feasible label, recorded here and nowhere else, because "the
            # composition mixed, cast, dried and measured" means *reached
            # optimizer.tell* — which is exactly this callback and no earlier
            # point. A spectrum that gates cleanly but whose σ is only an upper
            # bound never gets here, and must not be labelled feasible: that
            # would fabricate the same kind of fact the NULL discipline exists
            # to refuse.
            #
            # `channel=None` on purpose: this objective may aggregate several
            # replicate electrodes, so there is no single channel it belongs to.
            # §3.3's retraction is unaffected — it drops a flagged channel's
            # *rejects*, and a channel that ACCEPTed is not what the pattern is
            # about.
            label_engine.record_measured(params=p, channel=None,
                                         objective_value=o)

        loop.on_result = _on_result
        loop.on_trial_measured = (
            _equilibrate_then_label if settle_plan is not None else _confirm_and_label)
        loop.on_state_change = lambda old, new: emit("state", old=old.name, new=new.name)
        loop.on_converged = lambda i, best: emit("converged", iteration=i, best=best)
        loop.on_board_exchange_requested = lambda board, remaining: emit(
            "board_exchange", board=board, remaining=remaining
        )
        # Recovery visibility: without these, a campaign that silently lost half
        # its channels to retries looks identical to a healthy one.
        loop.on_step_recovered = lambda step, error, attempt: emit(
            "step_recovered", step=step, error=error, attempt=attempt
        )
        loop.on_step_skipped = lambda step, reason: emit(
            "step_skipped", step=step, reason=reason
        )

        def _on_park(reason: str) -> None:
            """Make the rig physically safe, then report why we stopped.

            This is the whole point of parking: an unattended run that gives up
            must not leave the head down, the heater at setpoint and the lamp on
            for however many hours pass before someone walks in.
            """
            nonlocal loop_parked
            loop_parked = True
            emit("park", reason=reason)
            try:
                result = safe_park(manager, reason=reason)
                emit("safe_park", ok=result.ok, actions=result.actions,
                     errors=result.errors, skipped=result.skipped)
            except Exception as exc:  # safe_park does not raise, but never trust that here
                logger.error("safe_park_failed", error=str(exc))
                result = None
                emit("safe_park", ok=False, errors=[str(exc)])
            # Durable record: the event stream dies with this process, but the
            # operator still has to learn *why* the rig stopped overnight.
            raise_alert(
                Alert(
                    kind="park",
                    message=f"Campaign '{spec.name}' parked: {reason}",
                    severity=CRITICAL,
                    run_id=run_id,
                    details={
                        "iteration": loop.iteration,
                        "safe_park_ok": bool(result.ok) if result else False,
                        "safe_park_errors": list(result.errors) if result else ["safe_park raised"],
                    },
                ),
                data_store=data_store,
            )

        loop.on_park = _on_park

        def _on_checkpoint(iteration: int) -> None:
            """Persist the resume point after every completed iteration (P3.2).

            Written *after* the observation reached the optimizer, so a crash in
            between costs one data point rather than claiming an observation for
            a well that was never cast. Best-effort by contract — the loop
            swallows failures here, because losing resumability must not end an
            otherwise healthy multi-day campaign.
            """
            data_store.save_campaign_checkpoint(
                spec.name,
                iteration=iteration,
                run_id=run_id,
                loop_state=loop.state.name,
                board_id=data_store.current_board_id(),
                spec_json=serialize_campaign_spec(spec),
                optimizer_json=json.dumps(optimizer.to_dict()),
                # Deliberately columns and not keys inside `spec_json`: that blob
                # is fingerprint-verified *spec identity*, and counters that
                # change every iteration do not belong inside the thing whose
                # stability proves the resume is legitimate.
                rh_ceiling_streak=rh_escalation.streak,
                # Already incremented for a failing trial: `_note_trial_failure`
                # counts before it advances the iteration that lands here.
                consecutive_failures=loop.consecutive_failures,
            )

        loop.on_checkpoint = _on_checkpoint

        # Waste accrual (P5.4) — flushes book themselves as they execute.
        if data_store is not None:
            try:
                from softae.core.consumables import attach_consumables

                loop.waste_ledger, _boards = attach_consumables(data_store)
            except Exception:
                logger.warning("waste_ledger_attach_failed", exc_info=True)

        # Anti-clog purge alongside co-runnable steps (P8). Sited here so the
        # GUI and the headless CLI get it from the same place — the interlock
        # must not be live on one surface and inert on the other.
        purge_runner = _resolve_purge_runner(manager)

        def _purge_window(step: Any) -> None:
            # owns_rig=True: this runs inside the campaign's own claim.
            # allow_positioning=True: a measurement leaves the head wherever
            #   the cast ended, so the purge may travel to the basin itself.
            #   (During an anneal the bracket already parked it there, so
            #   this costs nothing — the pose is already AT_FLUSH.)
            # end_at_idle_rest=False: MUST restore the pose it found. Idle
            #   rest leaves the head DOWN, and both precondition_flush and
            #   single_drop_simul open with a bare move_to and no retract —
            #   so leaving it down would make the next step trip the stage
            #   head guard and fail the channel.
            purge_runner.maybe_purge(
                context=f"step:{getattr(step, 'name', '?')}",
                owns_rig=True, allow_positioning=True,
                end_at_idle_rest=False,
            )

        # The runner is never None (a null one absorbs the call), but the
        # *window* still has to be declined when there is nothing to purge:
        # the executor opens a thread and an asyncio task per co-runnable step
        # for as long as this hook is set, and paying that to call a no-op
        # would be a behaviour change, not an absence of one.
        loop.on_purge_window = _purge_window if purge_runner.performs_purges else None

        # ── Reaching a campaign that is running in another process (stage 4) ─
        # `emit` goes out; this is the one thing that comes in. The GUI (or
        # `softae-campaign control`) writes `runs/<run_id>/control.json` and the
        # watcher below picks it up within a poll.
        #
        # The dispatch table is *here*, not in the watcher, because this is the
        # layer that knows what a campaign-scoped stop means. The watcher
        # delivers and records; `AutonomousLoop` decides. Deliberately three
        # entries and not one `halt`: Pause must not park and Abort must, and a
        # single entry point would have to infer that from an argument.
        #
        # Every request is acknowledged on the narration stream — the same file,
        # in the same order, as what the campaign was doing when it arrived. A
        # control an operator pressed and heard nothing back from is worse than
        # no control at all.
        loop.on_pause_change = lambda phase, detail: emit(
            "campaign_pause", phase=phase, detail=detail,
            iteration=loop.iteration,
        )
        control = open_control_watcher(
            data_store.run_dir(run_id),
            handlers={
                "pause": lambda req: loop.pause(req.reason),
                "resume": lambda req: loop.resume(),
                "abort": lambda req: loop.abort(
                    req.reason or "operator abort (control.json)"),
            },
            on_ack=lambda ack: emit("control_ack", **ack),
        )
        if control is not None:
            control.start()

        # Optional external approval gate (human or agent). When present, the
        # loop runs with auto_approve off and we release each trial here.
        if approval_fn is not None:
            import asyncio

            async def _approver() -> None:
                while loop.state not in (
                    LoopState.CONVERGED, LoopState.STOPPED, LoopState.ERROR,
                ):
                    if loop.state is LoopState.AWAITING_APPROVAL:
                        decision = approval_fn(loop.iteration, dict(last_params))
                        if hasattr(decision, "__await__"):
                            decision = await decision
                        if decision:
                            loop.approve()
                        else:
                            loop.stop()
                            loop.approve()  # release the wait so run() sees STOPPED
                    await asyncio.sleep(0.02)

            approver_task = asyncio.create_task(_approver())
            try:
                best = await loop.run()
            finally:
                approver_task.cancel()
        else:
            best = await loop.run()

        result = CampaignResult(
            run_id=run_id,
            best_params=best[0] if best else None,
            best_objective=best[1] if best else None,
            n_trials=loop.iteration,
            final_state=loop.state.name,
            converged=loop.state is LoopState.CONVERGED,
            history=list(optimizer.history),
            park_reason=loop.park_reason,
        )
        status = _RUN_STATUS_BY_STATE.get(loop.state, "done")
        _finalize_run(status)
        # Drop the resume point only when the campaign ended *on purpose*.
        # A parked run keeps its checkpoint — being able to resume after a park
        # is exactly why the checkpoint exists — and a crash never reaches here.
        if loop.state in (LoopState.CONVERGED, LoopState.STOPPED) and not loop.park_reason:
            try:
                data_store.clear_campaign_checkpoint(spec.name)
            except Exception:
                logger.warning("clear_checkpoint_failed", campaign=spec.name)
        emit("run_finished", run_id=run_id, best=best, n_trials=loop.iteration,
             status=status)
        return result

    except BaseException as exc:
        # Includes cancellation and KeyboardInterrupt: a run that dies must not
        # be left looking like it is still executing — *and must not be left
        # driving the rig*. Until this parked, the only exit that made the
        # hardware safe was the loop's own decision to park; anything the loop
        # did not foresee (a crash, a Ctrl-C, an operator abort, a step raising
        # through) unwound straight into `disconnect_all`, after which
        # `safe_park` skips every instrument as not connected and a park becomes
        # structurally impossible. The heater stays at setpoint and the lamp
        # stays on until someone walks in.
        #
        # Sited before `_finalize_run` deliberately: that is a database write and
        # this is a heater. Both run — the park is in a `try`/`finally` — but the
        # physical one goes first.
        try:
            if not loop_parked:
                reason = (f"campaign '{spec.name}' aborted: "
                          f"{type(exc).__name__}: {exc}"[:300])
                park_result = park_on_shutdown(manager, reason)
                if park_result is not None:
                    try:
                        emit("park", reason=reason)
                        emit("safe_park", ok=park_result.ok,
                             actions=park_result.actions,
                             errors=park_result.errors,
                             skipped=park_result.skipped)
                    except Exception:
                        # A consumer's callback must not replace the exception
                        # that is actually unwinding with one about reporting it.
                        logger.warning("abort_park_emit_failed", exc_info=True)
                    # Durable, because the event stream dies with this process
                    # and an overnight crash has no other reader.
                    try:
                        raise_alert(
                            Alert(
                                kind="park",
                                message=(f"Campaign '{spec.name}' aborted and was "
                                         f"parked: {type(exc).__name__}"),
                                severity=CRITICAL,
                                run_id=run_id,
                                details={
                                    "error": str(exc)[:500],
                                    "safe_park_ok": bool(park_result.ok),
                                    "safe_park_errors": list(park_result.errors),
                                },
                            ),
                            data_store=data_store,
                        )
                    except Exception:
                        logger.warning("abort_park_alert_failed", exc_info=True)
        finally:
            _finalize_run("error")
        raise

    finally:
        # Stop listening before anything else. A control request accepted during
        # teardown would reach a loop that has already ended — `abort()` refuses
        # that case on its own, but a watcher that outlived the run it controls
        # is a promise the process can no longer keep.
        if control is not None:
            await control.aclose()
        # Then stop reading the rig — before `disconnect_all` below, because a
        # conditions read racing a disconnect is a read of a session being taken
        # away from under it. An in-flight read is let go of rather than waited
        # on: a thread blocked in a driver retry cannot be cancelled, and a
        # campaign teardown must not queue behind a monitoring read.
        if conditions is not None:
            await conditions.aclose()
        # Then: stop beating. Every event worth narrating — `run_finished`, or
        # the `park`/`safe_park` pair from the catch-all above — has already been
        # emitted by now, and a heartbeat that outlived the campaign would tell a
        # watcher the run is alive while it is being torn down.
        if narrator is not None:
            await narrator.aclose()
        try:
            # Must run before the store closes.  A no-op if already finalized
            # above; this catches any exit path that bypassed both branches.
            _finalize_run("error")
            if owns_store:
                data_store.close()
            if owns_manager:
                await manager.disconnect_all()
        finally:
            # Last, and in its own `finally` so a failing teardown cannot strand
            # the claim: disconnecting drives hardware, and handing the rig to
            # someone else before that finishes is the window the lock exists to
            # close.
            rig_claim.close()
