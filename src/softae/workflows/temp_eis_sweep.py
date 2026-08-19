"""Temperature-stepped EIS sweep with Arrhenius fitting.

The :class:`ArrheniusSweep` class builds and executes a workflow that:

1. Steps the temperature stage through a user-configured profile.
2. Measures EIS on each selected channel after each temperature equilibrates.
3. Fits the ionic conductivity at every point.
4. Performs a linearised Arrhenius fit per channel to extract Eₐ.
5. Persists results to the DataStore (``arrhenius_results`` table) and
   optionally exports a JSON sidecar.

Typical usage::

    from softae.analysis.arrhenius import ArrheniusSweepConfig
    from softae.workflows.temp_eis_sweep import ArrheniusSweep

    config = ArrheniusSweepConfig(
        channels=[1, 2],
        T_start=25.0, T_stop=75.0, T_step=10.0,
        dwell_s=60.0,
        eis_model="simpleSalt",
        electrode_geometry={"L_cm": 0.2, "t_cm": 0.175, "w_cm": 0.2},
    )
    sweep = ArrheniusSweep(
        config=config,
        manager=manager,
        data_store=data_store,
        run_id=run_id,
        eis_instrument="pico1",
        temp_instrument="temp_controller",
    )
    results = await sweep.run()

Alternatively, instantiate from a YAML workflow definition::

    sweep = ArrheniusSweep.from_yaml("workflows/temp_eis_sweep.yaml", manager)
    results = await sweep.run()
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from softae.analysis.arrhenius import ArrheniusResult, ArrheniusSweepConfig
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant, cell_from_legacy_terms
from softae.analysis.eis_data import EISResult
from softae.workflows.workflow_model import Workflow, WorkflowStep

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

# Regex to parse step names produced by build_workflow() — new format first,
# legacy format (T-index) kept for backward compatibility.
_EIS_STEP_RE = re.compile(
    r"^eis_ch(?P<channel>\d+)_T(?P<temp_int>\d+)_RH(?P<rh_int>\d+)$"
)
_EIS_STEP_RE_LEGACY = re.compile(r"^eis_ch(?P<channel>\d+)_T(?P<tidx>\d+)$")


def _parse_eis_result(
    step: WorkflowStep,
    raw: Any,
    *,
    eis_params: dict[str, Any] | None = None,
    elapsed: float = 0.0,
) -> EISResult | None:
    """Convert a raw ``sendscript_getdata`` return value to :class:`EISResult`.

    Returns ``None`` if the result cannot be parsed.
    """
    try:
        channel = int(step.params.get("chan", step.params.get("channel", 0)))
        return EISResult.from_raw(
            raw,
            channel=channel,
            eis_params=eis_params or {},
            measurement_time_s=elapsed,
        )
    except Exception as exc:
        logger.warning("eis_parse_failed", step=step.name, error=str(exc))
        return None


class ArrheniusSweep:
    """Orchestrator for a temperature-stepped EIS sweep with Arrhenius analysis.

    Parameters
    ----------
    config : ArrheniusSweepConfig
        Sweep parameters.
    manager : InstrumentManager
        Instrument registry used by :class:`~softae.workflows.workflow_executor.WorkflowExecutor`.
    data_store : DataStore or None
        If supplied, Arrhenius results are written to the ``arrhenius_results``
        table after the sweep completes.
    run_id : str or None
        Experiment run identifier.  Required when *data_store* is provided.
    eis_instrument : str
        Key for the EIS device in *manager*.
    temp_instrument : str
        Key for the temperature controller in *manager*.
    """

    def __init__(
        self,
        config: ArrheniusSweepConfig,
        manager: "InstrumentManager",
        *,
        data_store: "DataStore | None" = None,
        run_id: str | None = None,
        eis_instrument: str = "pico1",
        temp_instrument: str = "temp_controller",
    ) -> None:
        config.validate()
        self.config = config
        self.manager = manager
        self.data_store = data_store
        self.run_id = run_id
        self.eis_instrument = eis_instrument
        self.temp_instrument = temp_instrument

        # Populated during run()
        self._eis_results: dict[tuple[int, int], EISResult] = {}

        # Acquisition scout, OBSERVE-ONLY on this path — see
        # _build_channel_scripts for why it cannot acquire here. Constructed
        # lazily so a sweep that never measures never reads the config.
        self._scout_planner: Any | None = None

        # Thermal fitter selected by config (Arrhenius or VFT); both share the
        # same fit() signature and return a result with .model / .R_squared / etc.
        from softae.analysis.thermal import make_fitter

        self._fitter = make_fitter(config.thermal_model)

        # Optional external callbacks — set by the caller before run().
        # on_step_complete(step, index, total, result, elapsed)
        # on_step_error(step, index, total, error)
        # on_eis_point(channel, T_C, sigma, R0, R1, rh_sp) — fired live per EIS measurement
        self.on_step_complete: Any | None = None
        self.on_step_error: Any | None = None
        self.on_eis_point: Any | None = None

        # Thread-safe abort flag.  Set via abort(); checked by _abortable_sleep().
        self._abort_flag: threading.Event = threading.Event()
        # True if _run_rh_sweep auto-started the RH controller (so it must
        # stop it on completion/abort).
        self._rh_started_by_sweep: bool = False

    # ── Alternative constructor ─────────────────────────────────────────

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        manager: "InstrumentManager",
        *,
        data_store: "DataStore | None" = None,
        run_id: str | None = None,
    ) -> "ArrheniusSweep":
        """Create an :class:`ArrheniusSweep` from a YAML workflow file.

        The YAML must have ``metadata.experiment_type == "arrhenius_sweep"``
        and the following variables::

            variables:
              channels: [1, 2]
              T_start: 25.0
              T_stop:  75.0
              T_step:  10.0
              dwell_s: 60.0
              eis_model: simpleSalt
              electrode_L_cm: 0.2     # optional
              electrode_t_cm: 0.175   # optional
              electrode_w_cm: 0.2     # optional

        Parameters
        ----------
        path : str or Path
        manager : InstrumentManager
        data_store : DataStore or None
        run_id : str or None
        """
        from softae.workflows.workflow_parser import parse_file

        wf = parse_file(path)
        meta = wf.metadata
        if meta.get("experiment_type") != "arrhenius_sweep":
            raise ValueError(
                f"Workflow '{wf.name}' is not an arrhenius_sweep "
                f"(metadata.experiment_type = {meta.get('experiment_type')!r})"
            )

        v = wf.variables

        geom: dict[str, float] | None = None
        if all(k in v for k in ("electrode_L_cm", "electrode_t_cm", "electrode_w_cm")):
            geom = {
                "L_cm": float(v["electrode_L_cm"]),
                "t_cm": float(v["electrode_t_cm"]),
                "w_cm": float(v["electrode_w_cm"]),
            }

        temperatures = v.get("temperatures", None)
        config = ArrheniusSweepConfig(
            channels=[int(c) for c in v["channels"]],
            T_start=float(v.get("T_start", 25.0)),
            T_stop=float(v.get("T_stop", 75.0)),
            T_step=float(v.get("T_step", 10.0)),
            temperatures=temperatures,
            dwell_s=float(v.get("dwell_s", 60.0)),
            tolerance_C=float(v.get("temp_tolerance", 0.5)),
            wait_timeout_s=float(v.get("wait_timeout_s", 1800.0)),
            eis_model=str(v.get("eis_model", "simpleSalt")),
            thermal_model=str(v.get("thermal_model", "arrhenius")),
            electrode_geometry=geom,
        )

        eis_instrument = str(v.get("eis_instrument", "pico1"))
        temp_instrument = str(v.get("temp_instrument", "temp_controller"))

        return cls(
            config=config,
            manager=manager,
            data_store=data_store,
            run_id=run_id,
            eis_instrument=eis_instrument,
            temp_instrument=temp_instrument,
        )

    # ── Workflow construction ───────────────────────────────────────────

    def build_workflow(self, rh_sp: float | None = None) -> Workflow:
        """Build the measurement workflow, respecting ``config.sweep_order``.

        When the default ordering (T outer, channels inner) is in effect this
        produces the original parallel-fan DAG.  When channels rank < T rank the
        call is forwarded to :meth:`_build_workflow_ch_outer` which generates a
        sequential channel-first chain instead.
        """
        order = self.config.sweep_order
        ch_rank = order.get("channels", 3)
        t_rank = order.get("T", 2)
        if ch_rank < t_rank:
            return self._build_workflow_ch_outer(rh_sp=rh_sp)
        return self._build_workflow_t_outer(rh_sp=rh_sp)

    def _build_workflow_t_outer(self, rh_sp: float | None = None) -> Workflow:
        """Build a :class:`Workflow` with explicit ``depends_on`` chains.

        DAG topology for temperatures ``[T0, T1, …]`` and channels ``[c0, c1]``::

            set_T0  →  wait_T0  →  eis_c0_T0  ─┐
                                  eis_c1_T0  ─┘─→  set_T1  →  wait_T1  →  …

        All steps are placed in ``setup``; the teardown restores ambient.

        Parameters
        ----------
        rh_sp : float or None
            Current RH setpoint (% RH).  Embedded as ``_RH<int>`` in each EIS
            step name so that the saved filename carries the RH value.
            Defaults to ``0`` when ``None`` (no RH sweep active).
        """
        temps = self.config.resolved_temperatures()
        channels = self.config.channels
        steps: list[WorkflowStep] = []
        _rh_int = round(rh_sp) if rh_sp is not None else 0

        geom = self.config.electrode_geometry or {}

        prev_eis_names: list[str] = []  # EIS step names for previous temperature

        for t_idx, temp in enumerate(temps):
            set_name = f"set_temp_T{t_idx}"
            wait_name = f"wait_temp_T{t_idx}"

            # Depends_on: all EIS steps from previous temperature (or nothing)
            set_depends = list(prev_eis_names)

            steps.append(
                WorkflowStep(
                    name=set_name,
                    instrument=self.temp_instrument,
                    method="write_sp",
                    params={"T_SP": temp, "print_flag": 0},
                    depends_on=set_depends,
                    tags={"temperature": str(temp), "t_idx": str(t_idx)},
                )
            )
            steps.append(
                WorkflowStep(
                    name=wait_name,
                    instrument=self.temp_instrument,
                    method="wait",
                    params={
                        "within": self.config.tolerance_C,
                        "equilibration_time": self.config.dwell_s,
                        "timeout": self.config.wait_timeout_s,
                    },
                    depends_on=[set_name],
                    timeout_s=self.config.wait_timeout_s,
                    tags={"temperature": str(temp), "t_idx": str(t_idx)},
                )
            )

            eis_names: list[str] = []
            for ch in channels:
                eis_name = f"eis_ch{ch}_T{round(temp)}_RH{_rh_int}"
                mscr_path = str(Path(tempfile.gettempdir()) / f"softae_ch{ch}.mscr")
                eis_params: dict[str, Any] = {
                    "mscrpath": mscr_path,
                    "outdir": str(Path(tempfile.gettempdir()) / "softae_eis_output"),
                    "chan": ch,
                    "circuit_model": self.config.eis_model,
                }
                if geom:
                    eis_params["electrode_L_cm"] = geom["L_cm"]
                    eis_params["electrode_t_cm"] = geom["t_cm"]
                    eis_params["electrode_w_cm"] = geom["w_cm"]

                steps.append(
                    WorkflowStep(
                        name=eis_name,
                        instrument=self.eis_instrument,
                        method="sendscript_getdata",
                        params=eis_params,
                        depends_on=[wait_name],
                        timeout_s=600.0,
                        retry=1,
                        tags={
                            "channel": str(ch),
                            "temperature": str(temp),
                            "t_idx": str(t_idx),
                        },
                    )
                )
                eis_names.append(eis_name)

            prev_eis_names = eis_names

        # Teardown — restore ambient temperature
        teardown_steps = [
            WorkflowStep(
                name="restore_ambient",
                instrument=self.temp_instrument,
                method="write_sp",
                params={"T_SP": temps[0], "print_flag": 0},
                tags={"phase": "teardown"},
            )
        ]

        return Workflow(
            name="temp_eis_sweep",
            description=(
                f"Temperature-stepped EIS sweep, "
                f"T={self.config.T_start}–{self.config.T_stop} °C, "
                f"channels={self.config.channels}"
            ),
            variables={
                "channels": self.config.channels,
                "temperatures": temps,
                "eis_model": self.config.eis_model,
            },
            setup=steps,
            loop_steps=[],
            teardown=teardown_steps,
            metadata={"experiment_type": "arrhenius_sweep"},
        )

    def _build_workflow_ch_outer(self, rh_sp: float | None = None) -> Workflow:
        """Sequential workflow: channels outermost, temperature innermost.

        For each channel a full temperature sweep is performed before moving
        to the next channel.  Set/wait T steps are re-issued whenever the
        temperature changes (including the reset from T_max of the previous
        channel back to T_start for the next channel).
        """
        temps = self.config.resolved_temperatures()
        channels = self.config.channels
        _rh_int = round(rh_sp) if rh_sp is not None else 0
        geom = self.config.electrode_geometry or {}

        steps: list[WorkflowStep] = []
        si = 0           # unique counter for set/wait step names
        prev_T: Any = object()   # sentinel — forces set on first iteration
        last_step: str | None = None

        for ch in channels:
            prev_T = object()   # each channel restarts the T sequence
            for temp in temps:
                cur_deps: list[str] = [last_step] if last_step else []

                if temp != prev_T:
                    set_name = f"set_temp_{si}"
                    wait_name = f"wait_temp_{si}"
                    steps.append(WorkflowStep(
                        name=set_name,
                        instrument=self.temp_instrument,
                        method="write_sp",
                        params={"T_SP": temp, "print_flag": 0},
                        depends_on=cur_deps,
                        tags={"temperature": str(temp)},
                    ))
                    steps.append(WorkflowStep(
                        name=wait_name,
                        instrument=self.temp_instrument,
                        method="wait",
                        params={
                            "within": self.config.tolerance_C,
                            "equilibration_time": self.config.dwell_s,
                            "timeout": self.config.wait_timeout_s,
                        },
                        depends_on=[set_name],
                        timeout_s=self.config.wait_timeout_s,
                    ))
                    si += 1
                    cur_deps = [wait_name]
                    prev_T = temp

                eis_name = f"eis_ch{ch}_T{round(temp)}_RH{_rh_int}"
                mscr_path = str(Path(tempfile.gettempdir()) / f"softae_ch{ch}.mscr")
                eis_params: dict[str, Any] = {
                    "mscrpath": mscr_path,
                    "outdir": str(Path(tempfile.gettempdir()) / "softae_eis_output"),
                    "chan": ch,
                    "circuit_model": self.config.eis_model,
                }
                if geom:
                    eis_params["electrode_L_cm"] = geom["L_cm"]
                    eis_params["electrode_t_cm"] = geom["t_cm"]
                    eis_params["electrode_w_cm"] = geom["w_cm"]

                steps.append(WorkflowStep(
                    name=eis_name,
                    instrument=self.eis_instrument,
                    method="sendscript_getdata",
                    params=eis_params,
                    depends_on=cur_deps,
                    timeout_s=600.0,
                    retry=1,
                    tags={"channel": str(ch), "temperature": str(temp)},
                ))
                last_step = eis_name

        teardown_steps = [WorkflowStep(
            name="restore_ambient",
            instrument=self.temp_instrument,
            method="write_sp",
            params={"T_SP": temps[0], "print_flag": 0},
            tags={"phase": "teardown"},
        )]

        return Workflow(
            name="temp_eis_sweep",
            description=(
                f"Channel-outer EIS sweep, "
                f"T={self.config.T_start}–{self.config.T_stop} °C, "
                f"channels={self.config.channels}"
            ),
            variables={
                "channels": self.config.channels,
                "temperatures": temps,
                "eis_model": self.config.eis_model,
            },
            setup=steps,
            loop_steps=[],
            teardown=teardown_steps,
            metadata={"experiment_type": "arrhenius_sweep"},
        )

    def _build_full_ordered_workflow(self) -> Workflow:
        """Build a fully-ordered workflow covering all three axes (RH, T, channels).

        Used when RH rank != 1 (RH is not the outermost loop), requiring RH
        setpoint steps to be interleaved directly into the workflow DAG rather
        than handled by an outer Python loop.  Generates a strictly sequential
        chain — each step depends on the one before it.
        """
        from itertools import product as _cart

        order = self.config.sweep_order
        _default: dict[str, int] = {"RH": 1, "T": 2, "channels": 3}
        temps = self.config.resolved_temperatures()
        channels = self.config.channels
        rh_sps: list[Any] = self.config.rh_setpoints if self.config.rh_setpoints else [None]
        geom = self.config.electrode_geometry or {}

        axis_values: dict[str, list] = {
            "T": temps,
            "channels": channels,
            "RH": rh_sps,
        }
        sorted_axes = sorted(axis_values.keys(), key=lambda a: order.get(a, _default[a]))

        combos = list(_cart(
            axis_values[sorted_axes[0]],
            axis_values[sorted_axes[1]],
            axis_values[sorted_axes[2]],
        ))

        steps: list[WorkflowStep] = []
        si = 0
        _sentinel = object()
        prev_T: Any = _sentinel
        prev_RH: Any = _sentinel
        last_step: str | None = None

        for combo in combos:
            vals = dict(zip(sorted_axes, combo))
            T_val = vals["T"]
            ch_val = vals["channels"]
            RH_val = vals["RH"]

            cur_deps: list[str] = [last_step] if last_step else []

            # Insert set/wait steps for axes that changed, outermost-first order
            for axis in sorted_axes:
                if axis == "RH" and RH_val is not None and RH_val != prev_RH:
                    set_rh = f"set_rh_{si}"
                    wait_rh = f"wait_rh_{si}"
                    steps.append(WorkflowStep(
                        name=set_rh,
                        instrument=self.config.rh_instrument,
                        method="set_setpoint",
                        params={"val": RH_val},
                        depends_on=list(cur_deps),
                    ))
                    steps.append(WorkflowStep(
                        name=wait_rh,
                        instrument=self.config.rh_instrument,
                        method="wait",
                        params={
                            "target": RH_val,
                            "tol": self.config.rh_tolerance,
                            "timeout": self.config.rh_wait_timeout_s,
                            "equilibration_s": self.config.rh_dwell_s,
                        },
                        depends_on=[set_rh],
                        timeout_s=self.config.rh_wait_timeout_s + self.config.rh_dwell_s + 30.0,
                    ))
                    si += 1
                    cur_deps = [wait_rh]
                    prev_RH = RH_val

                elif axis == "T" and T_val != prev_T:
                    set_T = f"set_temp_{si}"
                    wait_T = f"wait_temp_{si}"
                    steps.append(WorkflowStep(
                        name=set_T,
                        instrument=self.temp_instrument,
                        method="write_sp",
                        params={"T_SP": T_val, "print_flag": 0},
                        depends_on=list(cur_deps),
                        tags={"temperature": str(T_val)},
                    ))
                    steps.append(WorkflowStep(
                        name=wait_T,
                        instrument=self.temp_instrument,
                        method="wait",
                        params={
                            "within": self.config.tolerance_C,
                            "equilibration_time": self.config.dwell_s,
                            "timeout": self.config.wait_timeout_s,
                        },
                        depends_on=[set_T],
                        timeout_s=self.config.wait_timeout_s,
                    ))
                    si += 1
                    cur_deps = [wait_T]
                    prev_T = T_val

            rh_int_name = round(RH_val) if RH_val is not None else 0
            eis_name = f"eis_ch{ch_val}_T{round(T_val)}_RH{rh_int_name}"
            mscr_path = str(Path(tempfile.gettempdir()) / f"softae_ch{ch_val}.mscr")
            eis_params: dict[str, Any] = {
                "mscrpath": mscr_path,
                "outdir": str(Path(tempfile.gettempdir()) / "softae_eis_output"),
                "chan": ch_val,
                "circuit_model": self.config.eis_model,
            }
            if geom:
                eis_params["electrode_L_cm"] = geom["L_cm"]
                eis_params["electrode_t_cm"] = geom["t_cm"]
                eis_params["electrode_w_cm"] = geom["w_cm"]

            steps.append(WorkflowStep(
                name=eis_name,
                instrument=self.eis_instrument,
                method="sendscript_getdata",
                params=eis_params,
                depends_on=list(cur_deps),
                timeout_s=600.0,
                retry=1,
                tags={"channel": str(ch_val), "temperature": str(T_val)},
            ))
            last_step = eis_name

        teardown_steps = [WorkflowStep(
            name="restore_ambient",
            instrument=self.temp_instrument,
            method="write_sp",
            params={"T_SP": temps[0], "print_flag": 0},
            tags={"phase": "teardown"},
        )]

        return Workflow(
            name="temp_eis_sweep",
            description="Ordered EIS sweep",
            variables={
                "channels": self.config.channels,
                "temperatures": temps,
                "eis_model": self.config.eis_model,
            },
            setup=steps,
            loop_steps=[],
            teardown=teardown_steps,
            metadata={"experiment_type": "arrhenius_sweep"},
        )

    # ── Abort / async helpers ────────────────────────────────────────────

    def abort(self) -> None:
        """Request an immediate abort of the running sweep.

        Thread-safe: may be called from the GUI thread while the sweep runs
        on a worker thread.  Sets the internal abort flag (which causes any
        active RH dwell sleep to exit early) and cancels the
        :class:`~softae.workflows.workflow_executor.WorkflowExecutor` if one
        is currently active.  Also sets ``_wait_abort`` on the RH controller
        so any in-progress synchronous ``wait()`` call returns immediately.
        """
        self._abort_flag.set()
        # Unblock any synchronous rh_controller.wait() running inside the
        # WorkflowExecutor's thread pool.
        rh_inst = getattr(self, "_rh_inst_ref", None)
        if rh_inst is not None and hasattr(rh_inst, "_wait_abort"):
            rh_inst._wait_abort.set()
        # Unblock any synchronous temp_controller.wait() / _with_retry() running
        # in the thread pool.
        try:
            temp_inst = self.manager.get(self.temp_instrument)
            if hasattr(temp_inst, "_stop_wait"):
                temp_inst._stop_wait.set()
        except Exception:
            pass
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.abort()
            except Exception:
                pass

    async def _abortable_sleep(self, seconds: float, poll: float = 0.25) -> None:
        """Sleep for *seconds*, exiting early if the abort flag is set."""
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while True:
            if self._abort_flag.is_set():
                raise asyncio.CancelledError("abort requested during RH dwell")
            remaining = end - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(poll, remaining))

    async def _abortable_wait_rh(
        self,
        rh_inst: Any,
        target: float,
        poll_s: float = 2.0,
    ) -> None:
        """Poll *rh_inst* until RH is within ``config.rh_tolerance`` of *target*.

        Checks the abort flag every *poll_s* seconds; raises
        :class:`asyncio.CancelledError` immediately if abort is requested.
        After reaching the target, holds for the configured
        ``rh_dwell_s`` (also abortable).

        Falls back to a plain timed sleep when *rh_inst* is ``None``
        (no hardware available).
        """
        timeout_s = self.config.rh_wait_timeout_s
        tol = self.config.rh_tolerance
        dwell_s = self.config.rh_dwell_s

        if rh_inst is None:
            # No hardware: just wait the dwell time
            await self._abortable_sleep(dwell_s)
            return

        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            if self._abort_flag.is_set():
                raise asyncio.CancelledError("abort requested during RH wait")
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "rh_wait_timeout",
                    target=target,
                    timeout=timeout_s,
                )
                break
            try:
                rh_now = rh_inst.get_H()
                if not math.isnan(rh_now) and abs(rh_now - target) <= tol:
                    logger.info("rh_target_reached", target=target, rh=rh_now)
                    break
            except Exception as exc:
                logger.debug("rh_read_failed", error=str(exc))
            await asyncio.sleep(min(poll_s, remaining))

        # Post-stabilisation hold (abortable)
        await self._abortable_sleep(dwell_s)

    # ── Acquisition scripts ─────────────────────────────────────────────

    @property
    def _scout(self):
        """The acquisition planner, built on first use."""
        if self._scout_planner is None:
            from softae.core.eis_scout_scripts import ScoutPlanner

            self._scout_planner = ScoutPlanner(site="arrhenius_sweep")
        return self._scout_planner

    def _build_channel_scripts(self) -> None:
        """Write every channel's ``.mscr`` for the sweep about to run.

        **The scout observes here; it does not acquire.** Every channel's script
        is written once, before ``executor.run``, and the workflow handed to the
        executor then holds the whole temperature axis (and, on the ordered path,
        the RH axis too) — so between two measurements on a channel there is no
        moment at which a different script could be written. The only granularity
        this path could ever offer is *per sweep*, i.e. one sweep's spectra
        choosing the next sweep's grids; and that is exactly the cross-boundary
        carry the always-replan rule forbids, because the sweeps either side of
        that boundary are at different RH setpoints and the apex was measured
        moving ~100x across an RH change.

        So on this path ``[eis.scout] actuate`` has no effect by construction, and
        ``enabled`` buys the verdict log and the per-row verdict stamp. Making it
        acquire would mean moving this build into the step-completion path, which
        is a change to the workflow's shape rather than to its wiring.
        """
        try:
            from softae.drivers.mscr_library import eis_run_mscrbuild
        except ImportError:
            return  # mscr_library not available in test/CI environments

        p = self.config.eis_params or {}
        for ch in self.config.channels:
            mscr_path = str(Path(tempfile.gettempdir()) / f"softae_ch{ch}.mscr")
            eis_run_mscrbuild(
                mscr_path,
                mux_ch=ch,
                mVac=p.get("mv_ac", 10),
                f_hi=p.get("f_hi", 200_000),
                f_lo=p.get("f_lo_mHz", 100),
                npts=p.get("npts", 20),
                mVdc=p.get("mv_dc", 0),
            )

    def _eis_params_for(self, channel: int) -> dict[str, Any]:
        """The grid that reached the instrument for *channel*, as a row records it.

        A fresh dict per measurement while the scout is observing, because it
        stamps its verdict onto the row's params and ``config.eis_params`` is one
        dict shared by every measurement in the sweep. Stamping the shared one
        would leave every earlier row claiming the last spectrum's verdict — and
        would edit the caller's configuration on the way past.
        """
        base = self.config.eis_params or {}
        return dict(base) if self._scout.observing else base

    # ── Execution ───────────────────────────────────────────────────────

    async def run(self) -> list[ArrheniusResult]:
        """Execute the sweep and return per-channel :class:`ArrheniusResult` objects.

        If :attr:`config.rh_setpoints` is set, repeats the full T sweep at each
        RH setpoint (outer loop = RH, inner loop = T).

        Returns
        -------
        list[ArrheniusResult]
            One entry per channel in :attr:`config`.  When an RH sweep is active,
            the returned list covers only the *last* RH point; full multi-RH
            data is available via :attr:`rh_results`.
        """
        rh_sps = self.config.rh_setpoints
        rh_rank = self.config.sweep_order.get("RH", 1)
        if rh_sps and rh_rank != 1:
            # RH is not the outermost axis — use unified ordered execution path
            return await self._run_ordered()
        if rh_sps:
            return await self._run_rh_sweep(rh_sps)
        return await self._run_single(rh_idx=None, rh_sp=None)

    async def _run_ordered(self) -> list[ArrheniusResult]:
        """Execute sweep with all three axes interleaved in a single workflow.

        Used when ``config.sweep_order["RH"] != 1`` (RH is not the outermost
        loop), requiring RH setpoint steps to be embedded inside the DAG
        rather than handled by a Python outer loop.
        """
        from collections import defaultdict
        from softae.workflows.workflow_executor import WorkflowExecutor

        self._eis_results_full: dict[tuple, Any] = {}  # (ch, t_int, rh_int) → EISResult
        temps = self.config.resolved_temperatures()

        self._build_channel_scripts()

        workflow = self._build_full_ordered_workflow()
        executor = WorkflowExecutor(
            manager=self.manager, data_store=self.data_store, run_id=self.run_id
        )
        self._executor = executor
        prev_on = self.on_step_complete

        # ── RH controller lifecycle (mirrors _run_rh_sweep) ──────────────
        rh_inst_name = self.config.rh_instrument
        rh_inst_ordered: Any = None
        self._rh_started_by_sweep = False
        self._rh_inst_ref: Any = None          # stored so abort() can set _wait_abort
        if self.config.rh_setpoints:
            try:
                rh_inst_ordered = self.manager.get(rh_inst_name)
            except Exception:
                logger.warning("rh_instrument_not_found", name=rh_inst_name)
            if rh_inst_ordered is not None:
                self._rh_inst_ref = rh_inst_ordered
                # Clear any stale abort event from a previous sweep
                if hasattr(rh_inst_ordered, "_wait_abort"):
                    rh_inst_ordered._wait_abort.clear()
                try:
                    already_running = rh_inst_ordered.status().get("running", False)
                except Exception:
                    already_running = False
                if not already_running:
                    try:
                        rh_inst_ordered.start()
                        self._rh_started_by_sweep = True
                        logger.info("rh_control_auto_started")
                    except Exception as exc:
                        logger.warning("rh_auto_start_failed", error=str(exc))

        def _capture_ordered(step, index, total, raw, elapsed=0.0):
            m = _EIS_STEP_RE.match(step.name) or _EIS_STEP_RE_LEGACY.match(step.name)
            if m:
                ch = int(m.group("channel"))
                t_int = (
                    int(m.group("temp_int"))
                    if "temp_int" in m.groupdict() and m.group("temp_int") is not None
                    else 0
                )
                rh_int = (
                    int(m.group("rh_int"))
                    if "rh_int" in m.groupdict() and m.group("rh_int") is not None
                    else 0
                )
                eis = _parse_eis_result(
                    step, raw, eis_params=self._eis_params_for(ch), elapsed=elapsed
                )
                if eis is not None:
                    self._eis_results_full[(ch, t_int, rh_int)] = eis
                    eis.T_sp = float(t_int)
                    # Observe-only here: the verdict belongs on the row it was
                    # drawn from, and nothing on this path can act on it.
                    self._scout.observe(ch, eis)
                    if self.on_eis_point is not None:
                        r0, r1, sigma = self._live_point(eis)
                        _rh_sp_float = next(
                            (sp for sp in (self.config.rh_setpoints or []) if round(sp) == rh_int),
                            float(rh_int),
                        )
                        self.on_eis_point(ch, float(t_int), sigma, r0, r1, _rh_sp_float)
            if prev_on:
                prev_on(step, index, total, raw, elapsed)

        executor.on_step_complete = _capture_ordered
        if self.on_step_error:
            executor.on_step_error = self.on_step_error

        try:
            await executor.run(workflow)
        finally:
            # ── Stop RH controller if we auto-started it ──────────────────
            if self._rh_started_by_sweep and rh_inst_ordered is not None:
                try:
                    rh_inst_ordered.stop()
                    logger.info("rh_control_auto_stopped")
                except Exception as exc:
                    logger.warning("rh_auto_stop_failed", error=str(exc))

        # ── Aggregate results per (ch, rh_int) ──────────────────────────
        groups: dict[tuple, list[tuple[float, Any]]] = defaultdict(list)
        for (ch, t_int, rh_int), eis in self._eis_results_full.items():
            groups[(ch, rh_int)].append((float(t_int), eis))
        for key in groups:
            groups[key].sort(key=lambda x: x[0])

        rh_ints = sorted({rhi for (_, rhi) in groups})
        self.rh_results: dict[float, list[ArrheniusResult]] = {}
        last_results: list[ArrheniusResult] = []

        for rhi in rh_ints:
            results_for_rh: list[ArrheniusResult] = []
            for ch in self.config.channels:
                entries = groups.get((ch, rhi), [])
                t_vals = [t for t, _ in entries]
                sigmas = [self._sigma_from_eis(eis) for _, eis in entries]
                result = self._fitter.fit(
                    t_vals, sigmas, channel=ch, run_id=self.run_id or ""
                )
                results_for_rh.append(result)
            self.rh_results[float(rhi)] = results_for_rh
            last_results = results_for_rh

        await self._persist(last_results)
        return last_results

    async def _run_rh_sweep(self, rh_sps: list[float]) -> list[ArrheniusResult]:
        """Outer RH loop: set RH, dwell, then run a full T sweep per setpoint.

        RH controller lifecycle
        -----------------------
        If the RH controller is not already running at sweep start, this
        method calls ``start()`` on it and tracks that *we* started it so it
        can be stopped again when the sweep finishes or is aborted.  If the
        controller was already running (e.g. started from the Manual Control
        tab) it is left running after the sweep completes.
        """
        self.rh_results: dict[float, list[ArrheniusResult]] = {}
        all_results: list[ArrheniusResult] = []
        self._rh_started_by_sweep = False
        self._rh_inst_ref: Any = None          # stored so abort() can set _wait_abort

        rh_inst_name = self.config.rh_instrument
        try:
            rh_inst = self.manager.get(rh_inst_name)
        except Exception:
            rh_inst = None
            logger.warning("rh_instrument_not_found", name=rh_inst_name)

        # ── Auto-start RH controller if PID loop is not running ──────────
        if rh_inst is not None:
            self._rh_inst_ref = rh_inst
            # Clear any stale abort event from a previous sweep
            if hasattr(rh_inst, "_wait_abort"):
                rh_inst._wait_abort.clear()
            try:
                already_running = rh_inst.status().get("running", False)
            except Exception:
                already_running = False
            if not already_running:
                try:
                    rh_inst.start()
                    self._rh_started_by_sweep = True
                    logger.info("rh_control_auto_started")
                except Exception as exc:
                    logger.warning("rh_auto_start_failed", error=str(exc))

        try:
            for rh_idx, rh_sp in enumerate(rh_sps):
                logger.info("rh_setpoint", rh_idx=rh_idx, rh_sp=rh_sp)
                # Set RH setpoint if instrument is available
                if rh_inst is not None:
                    try:
                        if asyncio.iscoroutinefunction(rh_inst.set_setpoint):
                            await rh_inst.set_setpoint(rh_sp)
                        else:
                            rh_inst.set_setpoint(rh_sp)
                    except Exception as exc:
                        logger.warning("rh_set_failed", rh_sp=rh_sp, error=str(exc))
                # Wait for humidity to stabilise then hold for rh_dwell_s (abortable)
                await self._abortable_wait_rh(rh_inst, rh_sp)
                # Run inner T sweep
                results = await self._run_single(rh_idx=rh_idx, rh_sp=rh_sp)
                self.rh_results[rh_sp] = results
                all_results = results  # last RH is the canonical return value

                # Notify GUI about completed RH slice via on_eis_point if wired
                # (the callback is already fired per-point inside _run_single)
        finally:
            # ── Stop RH controller if we auto-started it ─────────────────
            if self._rh_started_by_sweep and rh_inst is not None:
                try:
                    rh_inst.stop()
                    logger.info("rh_control_auto_stopped")
                except Exception as exc:
                    logger.warning("rh_auto_stop_failed", error=str(exc))
            # ── Restore temperature stage to ambient (first setpoint) ────
            temps = self.config.resolved_temperatures()
            if temps:
                try:
                    temp_inst = self.manager.get(self.temp_instrument)
                    if hasattr(temp_inst, "write_sp"):
                        temp_inst.write_sp(T_SP=temps[0], print_flag=0)
                except Exception as exc:
                    logger.warning("temp_restore_failed", error=str(exc))

        return all_results

    async def _run_single(
        self,
        *,
        rh_idx: int | None,
        rh_sp: float | None,
    ) -> list[ArrheniusResult]:
        """Run one full T sweep and return per-channel ArrheniusResult."""
        from softae.workflows.workflow_executor import WorkflowExecutor

        self._eis_results.clear()
        workflow = self.build_workflow(rh_sp=rh_sp)
        temps = self.config.resolved_temperatures()

        # Build per-channel .mscr files before execution, mirroring HT tab
        # behaviour.  eis_run_mscrbuild handles pico2 channel remapping internally.
        self._build_channel_scripts()

        executor = WorkflowExecutor(
            manager=self.manager,
            data_store=self.data_store,
            run_id=self.run_id,
        )
        self._executor = executor  # expose for external abort()

        # Hook into step completions to capture EIS data; also forward to any
        # external callback wired by the GUI layer.
        prev_on_step_complete = self.on_step_complete

        def _capture(step: WorkflowStep, index: int, total: int, raw: Any, elapsed: float = 0.0) -> None:
            m = _EIS_STEP_RE.match(step.name) or _EIS_STEP_RE_LEGACY.match(step.name)
            if m:
                ch = int(m.group("channel"))
                # New format embeds temp as integer in name; legacy uses t_idx.
                if "temp_int" in m.groupdict() and m.group("temp_int") is not None:
                    t_idx = next(
                        (i for i, t in enumerate(temps) if round(t) == int(m.group("temp_int"))),
                        0,
                    )
                else:
                    t_idx = int(m.group("tidx"))
                eis = _parse_eis_result(
                    step,
                    raw,
                    eis_params=self._eis_params_for(ch),
                    elapsed=elapsed,
                )
                if eis is not None:
                    self._eis_results[(ch, t_idx)] = eis
                    # Observe-only here: the verdict belongs on the row it was
                    # drawn from, and nothing on this path can act on it.
                    self._scout.observe(ch, eis)
                    # Stamp commanded T_sp and live T_pv / RH readings
                    eis.T_sp = float(temps[t_idx]) if t_idx < len(temps) else float("nan")
                    try:
                        _tc = self.manager.get(self.temp_instrument)
                        eis.T_pv = float(_tc.get_pv())
                    except Exception:
                        pass
                    try:
                        _rh = self.manager.get(self.config.rh_instrument or "rh_controller")
                        _st = _rh.status()
                        eis.rh_sp = float(_st.get("setpoint", float("nan")))
                        eis.rh_pv = float(_st.get("current_rh", float("nan")))
                    except Exception:
                        pass
                    logger.info(
                        "eis_captured",
                        channel=ch,
                        t_idx=t_idx,
                        temperature=temps[t_idx] if t_idx < len(temps) else "?",
                        npts=len(eis.frequency),
                    )
                    if self.on_eis_point is not None:
                        r0, r1, sigma = self._live_point(eis)
                        t_c = float(temps[t_idx]) if t_idx < len(temps) else float("nan")
                        self.on_eis_point(ch, t_c, sigma, r0, r1, rh_sp if rh_sp is not None else float("nan"))
            if prev_on_step_complete:
                prev_on_step_complete(step, index, total, raw, elapsed)

        executor.on_step_complete = _capture
        if self.on_step_error:
            executor.on_step_error = self.on_step_error

        logger.info(
            "arrhenius_sweep_start",
            run_id=self.run_id,
            rh_sp=rh_sp,
            channels=self.config.channels,
            temperatures=temps,
        )
        await executor.run(workflow)
        logger.info("arrhenius_sweep_complete", run_id=self.run_id, rh_sp=rh_sp)

        results = self._compute_results(temps)
        await self._persist(results)
        return results

    # ── Internal helpers ────────────────────────────────────────────────

    def _cell(self) -> CellConstant | None:
        """The sweep's cell constant, or ``None`` when the geometry is unusable.

        One builder for all three σ sites in this file, so the live plot and the
        stored Arrhenius σ cannot be computed from different constants. The guard on
        *whether a cell exists at all* is
        :func:`~softae.analysis.eis.geometry.cell_from_legacy_terms`, shared with the
        GUI, the web adapter and the result router.
        """
        geom = self.config.electrode_geometry
        if not geom:
            return None
        try:
            return cell_from_legacy_terms(geom["L_cm"], geom["t_cm"], geom["w_cm"])
        except (KeyError, TypeError):
            return None

    def _live_point(self, eis: EISResult) -> tuple[float, float, float]:
        """``(R0, R1, σ)`` for the live-plot callback — ``NaN`` for anything unknown.

        The two capture callbacks used to spell this out twice, identically, which
        is two places for the analysis route to drift apart while both plots kept
        drawing. Never raises: a live plot must not abort a running sweep.
        """
        try:
            # ``engine`` unset — the live trace follows ``[eis] engine``.
            report = analyze_spectrum(eis, cell=self._cell(),
                                      model_name=self.config.eis_model)
            fit = report.fit
            if fit is None or not fit.success:
                return float("nan"), float("nan"), float("nan")
            r0, r1 = float(fit.R0), float(fit.R1)
            sigma = (float(report.sigma.value)
                     if report.sigma.mode == "value" and not math.isnan(r1) and r1 > 0
                     else float("nan"))
            return r0, r1, sigma
        except Exception:
            return float("nan"), float("nan"), float("nan")

    def _sigma_from_eis(self, eis: EISResult) -> float:
        """Fit EIS spectrum and compute σ.  Returns ``NaN`` on failure."""
        if self.config.electrode_geometry is None:
            logger.warning(
                "sigma_skipped_no_geometry",
                channel=eis.channel,
                detail="electrode_geometry not set in ArrheniusSweepConfig — "
                        "all σ values will be NaN and Arrhenius fit will fail. "
                        "Set L_cm, t_cm, w_cm in the config.",
            )
            return float("nan")
        try:
            # ``engine`` unset: ``[eis] engine`` governs the sweep as well, and this
            # σ is the one the Arrhenius fit consumes and the DataStore persists.
            report = analyze_spectrum(eis, cell=self._cell(),
                                      model_name=self.config.eis_model)
            fit = report.fit
            if fit is None or not fit.success or math.isnan(fit.R1) or fit.R1 <= 0:
                return float("nan")
            if report.sigma.mode != "value":
                return float("nan")
            return float(report.sigma.value)
        except Exception as exc:
            logger.warning("sigma_fit_failed", channel=eis.channel, error=str(exc))
            return float("nan")

    def _compute_results(self, temps: list[float]) -> list[ArrheniusResult]:
        """For each channel, aggregate σ vs T and call :class:`ArrheniusFitter`."""
        results: list[ArrheniusResult] = []
        for ch in self.config.channels:
            t_vals: list[float] = []
            sigma_vals: list[float] = []
            for t_idx, temp in enumerate(temps):
                eis = self._eis_results.get((ch, t_idx))
                if eis is not None:
                    sigma = self._sigma_from_eis(eis)
                else:
                    sigma = float("nan")
                t_vals.append(temp)
                sigma_vals.append(sigma)

            result = self._fitter.fit(
                t_vals,
                sigma_vals,
                channel=ch,
                run_id=self.run_id or "",
            )
            results.append(result)
            ok = result.fit_success
            # Both models report Eₐ (eV); VFT additionally has a Vogel T₀.
            if getattr(result, "model", "arrhenius") == "vft":
                headline = {
                    "Ea_eV": round(result.Ea_eV, 4) if ok else None,
                    "T0_C": round(result.T0_C, 2) if ok else None,
                }
            else:
                headline = {"Ea_eV": round(result.Ea_eV, 4) if ok else None}
            logger.info(
                "thermal_fit",
                channel=ch,
                model=getattr(result, "model", "arrhenius"),
                R_squared=round(result.R_squared, 4) if ok else None,
                n_points=result.n_points,
                success=ok,
                **headline,
            )
        return results

    async def _persist(self, results: list[ArrheniusResult]) -> None:
        """Write results to DataStore and JSON sidecar."""
        run_id = self.run_id

        if self.data_store is not None and run_id:
            for result in results:
                self.data_store.record_arrhenius(run_id, result)
            logger.info("arrhenius_stored", run_id=run_id, n_channels=len(results))

        if run_id:
            await self._write_json_sidecar(results, run_id)

    async def _write_json_sidecar(
        self, results: list[ArrheniusResult], run_id: str
    ) -> None:
        """Write JSON export to ``db/runs/{run_id}/arrhenius_results.json``."""
        out_dir: Path | None = None

        # Try to locate the project_dir via data_store
        if self.data_store is not None and hasattr(self.data_store, "project_dir"):
            out_dir = Path(self.data_store.project_dir) / "runs" / run_id
        else:
            out_dir = Path("db") / "runs" / run_id

        def _num(v: Any) -> Any:
            return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

        def _fit_to_dict(r: Any) -> dict[str, Any]:
            model = getattr(r, "model", "arrhenius")
            d: dict[str, Any] = {
                "model": model,
                "channel": r.channel,
                "run_id": r.run_id,
                "temperatures_C": r.temperatures_C,
                "conductivities": r.conductivities,
                "ln_A": _num(getattr(r, "ln_A", None)),
                "R_squared": _num(r.R_squared),
                "T_min_C": _num(getattr(r, "T_min_C", None)),
                "T_max_C": _num(getattr(r, "T_max_C", None)),
                "n_points": r.n_points,
                "fit_success": r.fit_success,
                "error_msg": r.error_msg,
            }
            if model == "vft":
                d.update(
                    Ea_eV=_num(getattr(r, "Ea_eV", None)),
                    Ea_kJ_per_mol=_num(getattr(r, "Ea_kJ_per_mol", None)),
                    A=_num(getattr(r, "A", None)),
                    B=_num(getattr(r, "B", None)),
                    T0_K=_num(getattr(r, "T0_K", None)),
                    T0_C=_num(getattr(r, "T0_C", None)),
                )
            else:
                d.update(
                    Ea_eV=_num(getattr(r, "Ea_eV", None)),
                    Ea_kJ_per_mol=_num(getattr(r, "Ea_kJ_per_mol", None)),
                )
            return d

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "arrhenius_results.json"
            payload = [_fit_to_dict(r) for r in results]
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("arrhenius_json_written", path=str(out_path))
        except Exception as exc:
            logger.warning("arrhenius_json_write_failed", error=str(exc))
