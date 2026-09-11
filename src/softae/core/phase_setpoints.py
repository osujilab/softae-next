"""Commanded temperature/humidity for one phase of a run — **setpoints, not readings**.

``conditions`` in this codebase has always meant the **recorded** environment:
the DataStore ``conditions`` table, :class:`softae.core.conditions_capture.Environment`
and its ``rh_pv_pct`` / ``stage_temp_pv_C`` process values, ``record_conditions``,
``ConditionsPublisher``. Every one of those is *what the chamber was doing*, read
back after the fact. This type is the other half — *what the chamber was told to
do*, before it does it — which is why it is named ``PhaseSetpoints`` and not
``Conditions``: SP, never PV, mirroring the rig's own ``rh_sp_pct``/``rh_pv_pct``
and ``stage_temp_sp_C``/``stage_temp_pv_C`` pairing. The *field* on
:class:`~softae.core.run_plan.RunPhase` and the TOML table an operator writes
both stay spelled ``conditions``, because that is the operator's word for it; the
type name is what keeps the two meanings apart in code.

``None`` on an axis means **do not drive it**, and is a different statement from
any value — the same distinction ``campaign_spec_io``'s ``explicit_none`` array
exists to preserve across a TOML round trip for ``rh_stability_pct``. A phase
with ``rh_setpoint_pct = None`` emits no humidity steps at all; a phase with
``rh_setpoint_pct = 0.0`` commands a dry purge. A codec that dropped the
difference would silently start (or stop) driving an axis.

Field names deliberately match :class:`softae.workflows.equilibration.EquilibrationConfig`'s
(``rh_setpoint_pct``, ``tolerance_C``, ``rh_tolerance_pct``, ``approach_timeout_s``,
``rh_approach_timeout_s``), and ``ValidationPlan`` in ``tools/eis_validate_hold.py``
inlines the same six. Collapsing those two copies onto this type is deliberately
**not** done here — both files are held by another session — but the naming is
chosen so that the collapse is mechanical rather than a redesign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from softae.workflows.workflow_model import WorkflowStep

__all__ = [
    "PhaseSetpoints",
    "DEFAULT_TOLERANCE_C",
    "DEFAULT_RH_TOLERANCE_PCT",
    "DEFAULT_APPROACH_TIMEOUT_S",
    "DEFAULT_RH_APPROACH_TIMEOUT_S",
    "APPROACH_STEP_MARGIN",
    "APPROACH_STEP_ALLOWANCE_S",
    "TEMP_INSTRUMENT",
    "RH_INSTRUMENT",
    "CONDITIONS_PHASE_TAG",
]

#: Instrument keys, matching ``data/tasks.toml``'s ``rh_*`` tasks and
#: ``EquilibrationConfig.temp_instrument`` / ``.rh_instrument``.
TEMP_INSTRUMENT = "temp_controller"
RH_INSTRUMENT = "rh_controller"

#: ``tags["phase"]`` every emitted step carries, so a condition step is
#: distinguishable from the cast, cure and measure steps around it.
CONDITIONS_PHASE_TAG = "conditions"

# The four tolerance/timeout defaults are the values
# ``softae.workflows.equilibration`` measured and documents at length
# (``DEFAULT_TOLERANCE_C`` = 2.0 °C because 0.5 graded a 0.6 °C dip as "hold not
# met"; ``DEFAULT_APPROACH_TIMEOUT_S`` = 1800 s is the *ascending* allowance).
# They are restated here rather than imported because that module is a 2 800-line
# async workflow pulling in the driver contracts, and this type sits under
# ``core`` on the deposition engine's import path. ``test_phase_setpoints.py``
# pins them equal to the originals, so a retune there fails loudly here instead
# of forking silently.
DEFAULT_TOLERANCE_C = 2.0
DEFAULT_RH_TOLERANCE_PCT = 2.0
DEFAULT_APPROACH_TIMEOUT_S = 1800.0
DEFAULT_RH_APPROACH_TIMEOUT_S = 1800.0

#: Slack between the *driver's* wait timeout and the *step's* execution ceiling.
#: ``data/tasks.toml``'s ``rh_wait`` states the rule in words — *"Task timeout_s
#: must exceed the 'timeout' param"* — and carries 1500 s against a 1200 s wait.
#: Derived here for the same reason ``deposition_recipe.anneal_timeout_s``
#: derives an anneal's: a hand-set constant and a parameterised duration are two
#: numbers that can disagree, and when they do the executor aborts a step that
#: was still working.
APPROACH_STEP_MARGIN = 1.25
APPROACH_STEP_ALLOWANCE_S = 60.0


def _step_ceiling(driver_timeout_s: float) -> float:
    """Execution ceiling for a wait step whose driver timeout is *driver_timeout_s*."""
    return driver_timeout_s * APPROACH_STEP_MARGIN + APPROACH_STEP_ALLOWANCE_S


@dataclass(frozen=True)
class PhaseSetpoints:
    """The environment one phase is **commanded** to hold. See the module docstring.

    ``approach_timeout_s`` and ``rh_approach_timeout_s`` belong to the *phase*,
    not to the driver, and that is the whole reason they are fields. The RH
    driver's own defaults are ``tol = 2.0, timeout = 120.0`` — appropriate for a
    monitoring poll and hopeless as a gate: the observed descent to ~20 %RH at
    85 °C takes on the order of 5 000 s, so a phase that inherited the driver
    default would fail its approach every time while the chamber was working
    normally. A phase that needs longer says so in its own file.

    The attainable RH floor **rises with chamber temperature** (the flush basin
    humidifies the enclosure as it warms): commanded 15 %RH returned a PV of
    19.5–23.2 at 85 °C. A setpoint below the floor saturates the PID with nothing
    broken, and no refusal here can tell that case from a genuine fault — see
    ``analysis/rh_floor.py``, whose own warning forbids fitting an absolute
    threshold to it. The advisory belongs at ``check`` time, not in this
    constructor.
    """

    name: str
    temp_setpoint_C: float | None = None
    rh_setpoint_pct: float | None = None
    tolerance_C: float = DEFAULT_TOLERANCE_C
    rh_tolerance_pct: float = DEFAULT_RH_TOLERANCE_PCT
    approach_timeout_s: float = DEFAULT_APPROACH_TIMEOUT_S
    rh_approach_timeout_s: float = DEFAULT_RH_APPROACH_TIMEOUT_S

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PhaseSetpoints.name must be a non-empty string — it "
                             "is what the operator reads back in the phase order")
        for axis in ("temp_setpoint_C", "rh_setpoint_pct"):
            value = getattr(self, axis)
            if value is None:
                continue
            # NaN is the one value that passes every comparison below and then
            # commands a setpoint nothing can reach; `None` is how "do not drive
            # this axis" is spelled, and it is not this.
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"PhaseSetpoints.{axis} must be a finite number or None "
                    f"(got {value!r}); None means 'do not drive this axis'"
                )
        for field_name in ("tolerance_C", "rh_tolerance_pct",
                           "approach_timeout_s", "rh_approach_timeout_s"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"PhaseSetpoints.{field_name} must be positive (got {value!r}); "
                    f"a zero band can never be satisfied and a zero timeout fails "
                    f"the approach before the chamber has moved"
                )

    # ── queries ──────────────────────────────────────────────────────────

    @property
    def drives_temperature(self) -> bool:
        return self.temp_setpoint_C is not None

    @property
    def drives_humidity(self) -> bool:
        return self.rh_setpoint_pct is not None

    def label(self) -> str:
        """``'anneal (85 °C, 20 %RH)'`` — the name and the axes actually driven."""
        axes = []
        if self.temp_setpoint_C is not None:
            axes.append(f"{float(self.temp_setpoint_C):g} °C")
        if self.rh_setpoint_pct is not None:
            axes.append(f"{float(self.rh_setpoint_pct):g} %RH")
        return f"{self.name} ({', '.join(axes) if axes else 'not driven'})"

    # ── emission ─────────────────────────────────────────────────────────

    def establish_steps(self, suffix: str) -> list[WorkflowStep]:
        """The steps that put the chamber into these setpoints, in order.

        Five steps at most, and the order is not cosmetic::

            temp write_sp → temp wait → rh set_setpoint → rh start → rh wait

        ``rh_start`` sits between the setpoint and the wait because *the setpoint
        alone actuates nothing* — ``data/tasks.toml``'s ``rh_start`` says so, and
        a wait before it would poll a stationary reading until it timed out.
        Temperature comes first because the attainable RH floor depends on it.

        An axis whose setpoint is ``None`` contributes **no steps**, so a phase
        that only sets temperature is two steps and one that drives neither is
        empty.

        The RH wait carries ``raise_on_timeout=True``, and that is the refusal
        this whole type exists to make possible: a phase that cannot reach its
        humidity **stops**, rather than proceeding into an 8 h cure at an unknown
        RH. It is the ruling ``data/tasks.toml``'s ``rh_wait`` already ships and
        ``anneal_recipe_long_hold.md`` §4 already argues; this only carries it
        onto the campaign path. Temperature has no equivalent flag because the
        driver's ``wait`` does not offer one — it logs ``temp_wait_timeout`` and
        returns — which is a real asymmetry and not an oversight here.

        Steps are built **directly rather than looked up in the catalog**: the
        values belong to the phase, not to a named catalog entry, which is the
        same argument ``anneal_recipe_long_hold.md`` §2 gave for parameterising
        ``rh_set_low``'s ``val``. ``rh_set_low`` / ``rh_start`` / ``rh_wait`` /
        ``rh_stop`` remain the hand-authored equivalents for Process Studio and
        sandbox workflows, and the steps emitted here are structurally
        indistinguishable from them — same instrument, same method, same param
        names — so the executor cannot tell the two apart.

        *suffix* disambiguates the step names (``"all"`` for a per-batch phase,
        ``"ch3"`` for a per-sample one, mirroring ``_anneal_steps``). Step names
        must be unique within a workflow, and the caller owns that: two phases
        emitting the same condition name under the same suffix would collide.
        """
        tags = {"phase": CONDITIONS_PHASE_TAG, "condition": self.name}
        steps: list[WorkflowStep] = []

        if self.temp_setpoint_C is not None:
            temp_tags = {**tags, "axis": "temperature"}
            steps.append(WorkflowStep(
                name=f"conditions_{self.name}_temp_sp_{suffix}",
                instrument=TEMP_INSTRUMENT,
                method="write_sp",
                # `T_SP`/`print_flag` are the driver's own parameter names
                # (`async_temp_controller.write_sp`), not `data/tasks.toml`'s
                # `set_temperature_25C`, which spells the first one `val` and
                # would raise TypeError through `BaseInstrument.execute`.
                params={"T_SP": float(self.temp_setpoint_C), "print_flag": 0},
                timeout_s=30.0,
                tags=temp_tags,
            ))
            steps.append(WorkflowStep(
                name=f"conditions_{self.name}_temp_wait_{suffix}",
                instrument=TEMP_INSTRUMENT,
                method="wait",
                params={"within": float(self.tolerance_C),
                        "timeout": float(self.approach_timeout_s)},
                timeout_s=_step_ceiling(float(self.approach_timeout_s)),
                tags=temp_tags,
            ))

        if self.rh_setpoint_pct is not None:
            rh_tags = {**tags, "axis": "humidity"}
            steps.append(WorkflowStep(
                name=f"conditions_{self.name}_rh_sp_{suffix}",
                instrument=RH_INSTRUMENT,
                method="set_setpoint",
                params={"val": float(self.rh_setpoint_pct)},
                timeout_s=30.0,
                tags=rh_tags,
            ))
            steps.append(WorkflowStep(
                name=f"conditions_{self.name}_rh_start_{suffix}",
                instrument=RH_INSTRUMENT,
                method="start",
                params={},
                timeout_s=30.0,
                tags=rh_tags,
            ))
            steps.append(WorkflowStep(
                name=f"conditions_{self.name}_rh_wait_{suffix}",
                instrument=RH_INSTRUMENT,
                method="wait",
                params={"target": float(self.rh_setpoint_pct),
                        "tol": float(self.rh_tolerance_pct),
                        "timeout": float(self.rh_approach_timeout_s),
                        "raise_on_timeout": True},
                timeout_s=_step_ceiling(float(self.rh_approach_timeout_s)),
                tags=rh_tags,
            ))

        return steps
