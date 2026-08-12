"""Legacy-derived drop-cast sweep — sequential dropcasting over a set of wells.

A standalone, optimizer-free loop that exercises just the liquid-handling
routines we ported from ``SoftAE_classPkg/liquid_handling.py``.  The model is
**one electrode per iteration**: for each channel in ``channels`` the stage
moves to that well and dispenses a fixed formulation via the ``liquid_handler``
composite ``single_drop_simul`` (port of the legacy ``singleDrop_simul``),
optionally measuring EIS afterward.  Because a well can only be cast once, the
natural budget is simply the number of wells.

This is the deterministic substrate the autonomous loop will later wrap
(swapping the fixed formulation for an optimizer suggestion per well); kept
separate so the drop-cast mechanics can be driven and watched on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from softae.config import loader
from softae.core.deposition_steps import (
    DEFAULT_DISP_RATE_UL_MIN,
    DEFAULT_ELUTION_WAIT_S,
    DEFAULT_FLUSH_RATE_UL_MIN,
    DEFAULT_FLUSH_VOL_UL,
    deposition_positions,
    eis_measure_step,
    final_flush_step,
    resolve_pcb,
    single_drop_step,
    startup_flush_step,
)
from softae.core.geometry import electrode_positions, electrode_xy_for_channel
from softae.core.hardware_safety import (
    HardwareNotArmedError,
    assert_hardware_armed,
)
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

logger = structlog.get_logger(__name__)

EventCallback = Callable[[dict[str, Any]], Any]

# Instruments the drop-cast sweep needs registered.
_REQUIRED_INSTRUMENTS = ("liquid_handler", "stage", "syringe")

# Below this per-syringe stroke, a real pump likely can't dispense reliably.
_MIN_RELIABLE_PER_SYRINGE_UL = 0.1


class DropcastPreflightError(RuntimeError):
    """Raised when a real sweep is asked to execute but preflight found errors."""


@dataclass
class DropcastFormulation:
    """The fixed per-well formulation dispensed on every electrode of a sweep."""

    ids: tuple[int, ...] = (0, 1)
    vols: tuple[float, ...] = (21.0, 21.0)  # µL per pump, ordered like ids
    deadvols: tuple[float, ...] = ()  # per-pump dead volume; default zeros
    disp_rate: float = DEFAULT_DISP_RATE_UL_MIN
    elution_wait_s: float = DEFAULT_ELUTION_WAIT_S
    wick_dwell_s: float = 5.0
    time_scale: float = 1.0  # scale routine dwells (0 → collapse for fast demos)
    # Startup prime (run once before the sweep)
    flush_vol: float = DEFAULT_FLUSH_VOL_UL
    flush_rate: float = DEFAULT_FLUSH_RATE_UL_MIN

    def __post_init__(self) -> None:
        self.ids = tuple(self.ids)
        self.vols = tuple(float(v) for v in self.vols)
        if len(self.ids) != len(self.vols):
            raise ValueError(
                f"ids ({len(self.ids)}) and vols ({len(self.vols)}) must be the same length"
            )
        if self.deadvols:
            self.deadvols = tuple(float(v) for v in self.deadvols)
            if len(self.deadvols) != len(self.ids):
                raise ValueError("deadvols must match the number of pumps")
        else:
            self.deadvols = tuple(0.0 for _ in self.ids)


@dataclass
class DropcastResult:
    """Outcome of a sweep (or a dry run, when ``steps_run == 0``)."""

    workflow_name: str
    channels: tuple[int, ...]
    electrode_xy: dict[int, tuple[float, float]]
    steps_run: int
    dispensed_uL_by_pump: dict[int, float] = field(default_factory=dict)
    preflight: "PreflightReport | None" = None
    executed: bool = False


@dataclass
class PreflightReport:
    """Structured safety check for a drop-cast sweep.

    ``ok`` is False iff there are blocking ``errors``.  ``warnings`` do not
    block but must be acknowledged (they surface things like sub-resolution
    per-syringe strokes).  ``info`` is advisory context (mock vs real, etc.).
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    mock_instruments: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"Preflight: {'PASS' if self.ok else 'FAIL'}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        for i in self.info:
            lines.append(f"  [info]  {i}")
        return "\n".join(lines)


def _parallel_count(syr_cfg: dict[str, Any], pump_id: int) -> int:
    """Parallel-syringe count for a pump, per ``[instruments.syringe]`` config."""
    base = int(syr_cfg.get("parallel_syringes", 1) or 1)
    return max(1, int(syr_cfg.get(f"parallel_syringes_pump{pump_id}", base) or base))


def preflight_dropcast(
    channels: tuple[int, ...],
    formulation: DropcastFormulation,
    *,
    manager: InstrumentManager,
    pcb_name: str | None = None,
    measure_eis: bool = False,
) -> PreflightReport:
    """Validate a sweep against instrument limits and config before any motion.

    Checks instrument presence (and flags mock fallbacks), pump rate bounds,
    per-syringe stroke resolution (accounting for parallel syringes), stage
    travel bounds for every electrode/flush/wick position, and channel validity
    against the PCB grid.  Returns a :class:`PreflightReport`; callers decide
    whether to proceed on warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    # -- Instruments present, and which are mock --
    registered = set(manager.names)
    mock_insts: list[str] = []
    for name in _REQUIRED_INSTRUMENTS:
        if name not in registered:
            errors.append(f"required instrument '{name}' is not registered")
            continue
        if type(manager.get(name)).__name__.startswith("Mock"):
            mock_insts.append(name)

    # -- Config-sourced limits --
    try:
        safety = loader.safety()
    except Exception:
        safety = {}
    try:
        syr_cfg = loader.instruments().get("syringe", {})
    except Exception:
        syr_cfg = {}

    rate_min = float(safety.get("pump_rate_min", syr_cfg.get("min_rate", 0.05)))
    rate_max = float(safety.get("pump_rate_max", syr_cfg.get("max_rate", 2120.0)))
    x_min = float(safety.get("stage_x_min_mm", -100.0))
    x_max = float(safety.get("stage_x_max_mm", 100.0))
    y_min = float(safety.get("stage_y_min_mm", -50.0))
    y_max = float(safety.get("stage_y_max_mm", 50.0))

    # -- Dispense rate --
    if not (rate_min <= formulation.disp_rate <= rate_max):
        errors.append(
            f"disp_rate {formulation.disp_rate} µL/min outside pump limits "
            f"[{rate_min}, {rate_max}]"
        )

    # -- Per-pump volume / parallel-syringe resolution --
    for pump_id, vol in zip(formulation.ids, formulation.vols):
        if vol <= 0:
            errors.append(f"pump {pump_id}: dispense volume must be > 0 (got {vol})")
            continue
        n_par = _parallel_count(syr_cfg, pump_id)
        per_syringe = vol / n_par
        if n_par > 1:
            info.append(
                f"pump {pump_id}: {n_par} parallel syringes → commanded {vol} µL "
                f"strokes {per_syringe:.4f} µL/syringe"
            )
        if per_syringe < _MIN_RELIABLE_PER_SYRINGE_UL:
            warnings.append(
                f"pump {pump_id}: {per_syringe:.4f} µL/syringe is below the "
                f"{_MIN_RELIABLE_PER_SYRINGE_UL} µL reliable-dispense threshold "
                f"(commanded {vol} µL ÷ {n_par} parallel) — verify the pump can "
                f"deliver this before a real run"
            )

    # -- Stage bounds for every position the sweep visits --
    positions = deposition_positions()
    fixed = {"flush": positions.flush, "wick": positions.wick}
    pcb_display, pcb = resolve_pcb(pcb_name)
    n_electrodes = len(electrode_positions(pcb)[0])
    for ch in channels:
        if ch < 1 or ch > n_electrodes:
            errors.append(
                f"channel {ch} outside PCB '{pcb_display}' grid (1..{n_electrodes})"
            )
            continue
        ex, ey = electrode_xy_for_channel(
            pcb, ch, origin_x=positions.origin[0], origin_y=positions.origin[1]
        )
        fixed[f"well {ch}"] = (ex, ey)
    for label, (px, py) in fixed.items():
        if not (x_min <= px <= x_max and y_min <= py <= y_max):
            errors.append(
                f"{label} position ({px:.2f}, {py:.2f}) outside stage travel "
                f"[{x_min},{x_max}]×[{y_min},{y_max}]"
            )

    if mock_insts:
        info.append(f"mock (simulated) instruments: {', '.join(mock_insts)}")
    if measure_eis:
        info.append("EIS measurement enabled (routes per channel via pico mapping)")

    return PreflightReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        info=info,
        mock_instruments=tuple(mock_insts),
    )


def parse_int_spec(spec: Any) -> tuple[int, ...]:
    """Parse '21-24' / '21,22,23' / [21,22] into an int tuple (ranges + commas)."""
    if isinstance(spec, (list, tuple)):
        return tuple(int(x) for x in spec)
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def build_dropcast_from_params(
    *,
    wells: Any = "21-24",
    pumps: Any = "0,1,2",
    vol: float = 0.1,
    rate: float = 1000.0,
    measure_eis: bool = False,
    time_scale: float = 1.0,
    pcb_name: str = "SoftAE_EIS_4Stripe",
    elution_wait_s: float = 240.0,
    name: str = "dropcast",
) -> Workflow:
    """Flat-parameter adapter over :func:`build_dropcast_sweep_workflow`.

    The GUI-friendly entry point for the ``dropcast_sweep`` recipe: the same
    knobs the CLI takes (wells/pumps/vol/rate/EIS/timing) as a flat kwargs dict,
    so a parameter panel can drive it.  A single ``vol`` is applied to every
    pump (matching the CLI's ``--vol``).
    """
    channels = parse_int_spec(wells)
    ids = parse_int_spec(pumps)
    formulation = DropcastFormulation(
        ids=ids,
        vols=tuple(float(vol) for _ in ids),
        deadvols=tuple(0.0 for _ in ids),
        disp_rate=float(rate),
        elution_wait_s=float(elution_wait_s),
        time_scale=float(time_scale),
    )
    return build_dropcast_sweep_workflow(
        channels, formulation, pcb_name=pcb_name,
        measure_eis=bool(measure_eis), name=name,
    )


def build_dropcast_sweep_workflow(
    channels: tuple[int, ...],
    formulation: DropcastFormulation,
    *,
    pcb_name: str | None = None,
    measure_eis: bool = False,
    name: str = "dropcast_sweep",
) -> Workflow:
    """Build the sequential-sweep workflow: prime → per-well cast → final flush.

    Each well's electrode ``(x, y)`` is resolved here from PCB geometry + the
    calibrated ``dep1`` origin (never encoded in a recipe); EIS, when enabled,
    routes to the correct pico per channel.
    """
    channels = tuple(channels)
    if not channels:
        raise ValueError("channels must name at least one well")

    positions = deposition_positions()
    pcb_display, pcb = resolve_pcb(pcb_name)

    ids = list(formulation.ids)
    setup: list[WorkflowStep] = [
        startup_flush_step(
            positions, ids,
            disp_rate=formulation.flush_rate, disp_vol=formulation.flush_vol,
            time_scale=formulation.time_scale,
        ),
    ]

    electrode_xy: dict[int, tuple[float, float]] = {}
    for ch in channels:
        ex, ey = electrode_xy_for_channel(
            pcb, ch, origin_x=positions.origin[0], origin_y=positions.origin[1]
        )
        electrode_xy[ch] = (ex, ey)
        setup.append(
            single_drop_step(
                ch, (ex, ey), positions,
                ids=ids, vols=list(formulation.vols),
                disp_rate=formulation.disp_rate,
                deadvols=list(formulation.deadvols),
                elution_wait_s=formulation.elution_wait_s,
                time_scale=formulation.time_scale,
                name=f"dropcast_ch{ch}",
                extra_params={"wick_dwell_s": formulation.wick_dwell_s},
            )
        )
        if measure_eis:
            setup.append(eis_measure_step(ch))

    teardown = [final_flush_step(ids[0])]

    return Workflow(
        name=name,
        description=(
            f"Sequential drop-cast on wells {','.join(map(str, channels))} ({pcb_display})"
        ),
        setup=setup,
        teardown=teardown,
        iterations=1,
        metadata={
            "source": "dropcast_sweep",
            "channels": list(channels),
            "pcb": pcb_display,
            "electrode_xy": {str(k): list(v) for k, v in electrode_xy.items()},
        },
    )


async def run_dropcast_sweep(
    channels: tuple[int, ...],
    formulation: DropcastFormulation,
    *,
    manager: InstrumentManager | None = None,
    pcb_name: str | None = None,
    measure_eis: bool = False,
    name: str = "dropcast_sweep",
    on_event: EventCallback | None = None,
    preflight: bool = True,
    dry_run: bool = False,
    confirm_fn: Callable[[PreflightReport], bool] | None = None,
) -> DropcastResult:
    """Drive the sweep headlessly through the real :class:`WorkflowExecutor`.

    If ``manager`` is omitted, a connected mock manager is created and torn down
    here.  Emits ``preflight`` / ``sweep_started`` / ``step_start`` /
    ``step_done`` / ``sweep_finished`` (and ``dry_run`` / ``aborted``) events to
    ``on_event`` — the live trace a UI or an agent can watch.

    Safety controls:
    - ``preflight`` (default True): run :func:`preflight_dropcast` first.  On a
      real execution, blocking errors raise :class:`DropcastPreflightError`.
    - ``dry_run``: run preflight and emit the plan, but execute **no** motion.
    - ``confirm_fn``: called with the preflight report just before execution;
      return False to abort (the human/agent go/no-go gate for a real cast).

    Returns a :class:`DropcastResult` (``steps_run == 0`` and ``executed ==
    False`` for a dry run or an aborted/failed-preflight sweep).
    """
    owns_manager = manager is None
    if manager is None:
        from softae.drivers.factory import create_manager

        manager = create_manager(mock=True)
        await manager.connect_all()

    def emit(kind: str, **payload: Any) -> None:
        if on_event:
            on_event({"type": kind, **payload})

    try:
        wf = build_dropcast_sweep_workflow(
            channels, formulation, pcb_name=pcb_name, measure_eis=measure_eis, name=name,
        )
        electrode_xy = {
            int(k): tuple(v) for k, v in wf.metadata["electrode_xy"].items()
        }

        report: PreflightReport | None = None
        if preflight:
            report = preflight_dropcast(
                channels, formulation, manager=manager,
                pcb_name=pcb_name, measure_eis=measure_eis,
            )
            emit("preflight", ok=report.ok, errors=list(report.errors),
                 warnings=list(report.warnings), info=list(report.info))
            if not report.ok and not dry_run:
                emit("aborted", reason="preflight_failed")
                raise DropcastPreflightError(
                    "drop-cast preflight failed:\n" + report.render()
                )

        def _result(steps: int, dispensed: dict[int, float], executed: bool) -> DropcastResult:
            return DropcastResult(
                workflow_name=wf.name, channels=channels, electrode_xy=electrode_xy,
                steps_run=steps, dispensed_uL_by_pump=dispensed,
                preflight=report, executed=executed,
            )

        plan = {
            "channels": list(channels), "wells": len(channels), "pcb": wf.metadata["pcb"],
            "electrode_xy": {k: list(v) for k, v in electrode_xy.items()},
            "formulation": {
                "ids": list(formulation.ids), "vols": list(formulation.vols),
                "disp_rate": formulation.disp_rate,
            },
            "total_steps": wf.total_steps,
        }

        if dry_run:
            emit("dry_run", **plan)
            return _result(0, {}, executed=False)

        # Hard interlock: never drive real motion unless deliberately armed.
        # A no-op for mock managers; raises for real, un-armed hardware.
        try:
            assert_hardware_armed(manager, action="run drop-cast motion")
        except HardwareNotArmedError as exc:
            emit("aborted", reason="hardware_not_armed", detail=str(exc))
            raise

        if confirm_fn is not None and not confirm_fn(report or PreflightReport(ok=True)):
            emit("aborted", reason="declined")
            return _result(0, {}, executed=False)

        emit("sweep_started", **plan)

        executor = WorkflowExecutor(manager)
        steps_run = {"n": 0}
        executor.on_step_start = lambda step, idx, total: emit(
            "step_start", index=idx, total=total, step=step.name,
            instrument=step.instrument, method=step.method,
            channel=step.tags.get("channel"),
        )

        def _done(step, idx, total, result, *_):
            steps_run["n"] += 1
            emit("step_done", index=idx, total=total, step=step.name, result=result)

        executor.on_step_complete = _done

        await executor.run(wf)

        # Read mock syringe dispense counters when available (verification).
        dispensed: dict[int, float] = {}
        try:
            dispensed = dict(manager.get("syringe")._dispensed)  # type: ignore[attr-defined]
        except Exception:
            pass

        emit("sweep_finished", steps_run=steps_run["n"], dispensed=dispensed)
        return _result(steps_run["n"], dispensed, executed=True)
    finally:
        if owns_manager:
            await manager.disconnect_all()
