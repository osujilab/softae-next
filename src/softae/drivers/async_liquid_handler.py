"""Liquid-handling coordinator — composite stage+syringe deposition routines.

Faithful ports of the legacy ``SoftAE_classPkg/liquid_handling.py`` routines,
which coordinate the linear **stage** and the **syringe** pump as a single
higher-level operation (move → descend → dispense → wait → retract → wick).
Because the :class:`~softae.workflows.workflow_executor.WorkflowExecutor`
dispatches one ``instrument.method`` per step, these multi-instrument
sequences cannot live on the stage or syringe driver alone; they are hosted
here on a coordinator instrument that reaches the stage and syringe through the
shared :class:`~softae.server.manager.InstrumentManager`.

Design notes
------------
* **Positions are split by concern.** ``flush``/``wick`` are apparatus-fixed
  (same every run) and are passed as bound task params.  The *electrode* target
  (``x``/``y`` in :meth:`single_drop_simul`) is per-channel and is injected at
  run time by the HT Experiment tab — Process Config never encodes a channel.
* **Timing is faithful but scalable.** Each routine derives its dwell/elution
  waits from rate/volume exactly as the legacy code did, but multiplies every
  wait by ``time_scale`` (default ``1.0``) so tests and mock runs can pass a
  small factor and complete quickly.
* Sub-instrument calls go through ``inst.execute(...)`` so each acquires its own
  per-instrument lock; the coordinator holds only its own lock for the duration
  of a routine, which is a different lock and therefore never deadlocks.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Sequence

import structlog

from softae.errors import InstrumentError
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)

#: Attribute name used to tag an in-flight deposition error with whether elution
#: was already commanded (the double-dispense point of no return).
DISPENSE_COMMITTED_ATTR = "dispense_committed"


def _annotate_dispense_committed(exc: BaseException, committed: bool) -> None:
    """Tag *exc* with the deposition point-of-no-return flag, best-effort.

    Read by :class:`~softae.workflows.workflow_executor.WorkflowExecutor` to
    decide, on a recoverable deposit failure, between replay-from-precondition
    (``committed=False`` — nothing dispensed) and skip-channel (``committed=True``
    — a retry would double-dispense). A deeper composite call may have already
    set the flag, so it is never overwritten; some built-in exceptions reject
    attribute assignment, hence the guard.
    """
    try:
        if not hasattr(exc, DISPENSE_COMMITTED_ATTR):
            setattr(exc, DISPENSE_COMMITTED_ATTR, committed)
    except Exception:
        pass


def _star_coordinates(
    n_points: int, cx: float, cy: float, outer_radius: float, inner_radius: float
) -> tuple[list[float], list[float]]:
    """Vertices of an ``n_points`` star about ``(cx, cy)`` — port of ``star_coordinates``.

    Alternates outer/inner radius vertices (a negative ``inner_radius`` inverts
    the inner points, giving the classic star path used for in-drop mixing).
    """
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_points + 1):
        a_out = math.pi / 2 + 2 * math.pi * i / n_points
        xs.append(round(cx + outer_radius * math.cos(a_out), 3))
        ys.append(round(cy + outer_radius * math.sin(a_out), 3))
        a_in = a_out + math.pi / n_points
        xs.append(round(cx + inner_radius * math.cos(a_in), 3))
        ys.append(round(cy + inner_radius * math.sin(a_in), 3))
    return xs, ys


class AsyncLiquidHandler(BaseInstrument):
    """Coordinator instrument for composite stage+syringe deposition routines.

    Holds no hardware of its own.  It drives the ``stage`` and ``syringe``
    instruments registered alongside it in the :class:`InstrumentManager`,
    which must be assigned to :attr:`manager` after construction (the factories
    do this at registration time).
    """

    def __init__(self, name: str = "liquid_handler", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self.manager: Any = None
        self._stage_name: str = self.config.get("stage_name", "stage")
        self._syringe_name: str = self.config.get("syringe_name", "syringe")

    # --- BaseInstrument interface ---------------------------------------------

    async def connect(self) -> None:
        # Purely a coordinator — nothing to open. Ready as soon as its stage and
        # syringe exist; those are connected independently by the manager.
        logger.info("liquid_handler_connect")
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED
        logger.info("liquid_handler_disconnect")

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(stage=self._stage_name, syringe=self._syringe_name)
        return s

    # --- Internal helpers -----------------------------------------------------

    def _instr(self, name: str) -> BaseInstrument:
        """Resolve a sub-instrument from the manager (raises if unavailable)."""
        if self.manager is None:
            raise InstrumentError(
                "no manager wired — cannot reach sub-instruments", instrument=self.name
            )
        return self.manager.get(name)

    async def _stage(self, method: str, **params: Any) -> Any:
        return await self._instr(self._stage_name).execute(method, **params)

    async def _syringe(self, method: str, **params: Any) -> Any:
        return await self._instr(self._syringe_name).execute(method, **params)

    @staticmethod
    async def _dwell(seconds: float, time_scale: float) -> None:
        """Sleep ``seconds * time_scale`` (never negative), yielding the loop."""
        await asyncio.sleep(max(0.0, float(seconds) * float(time_scale)))

    # --- Composite routines (ports of liquid_handling.py) ---------------------

    async def startup_flush(
        self,
        flush_x: float,
        flush_y: float,
        wick_x: float,
        wick_y: float,
        disp_rate: float,
        disp_vol: float,
        ids: Sequence[int] = (0, 1),
        disp_vols: Sequence[float] | None = None,
        disp_rates: Sequence[float] | None = None,
        res_vol: float = 1000.0,
        settle_factor: float = 1.5,
        post_flush_dwell_s: float = 60.0,
        wick_dwell_s: float = 5.0,
        time_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Prime the syringe lines at startup — port of ``startUpFlush``.

        A daily line-clearing prime.  Retracts the head, moves to the flush
        basin, dispenses from each pump in ``ids``, waits for the flush to
        complete, then wicks residual liquid.  Apparatus-fixed positions only.

        Per-pump volumes/rates: pass ``disp_vols`` / ``disp_rates`` (lists
        matching ``ids``) to use different amounts per pump; otherwise the scalar
        ``disp_vol`` / ``disp_rate`` is applied to every pump.
        """
        ids = [int(p) for p in ids]
        vols = list(disp_vols) if disp_vols is not None else [disp_vol] * len(ids)
        rates = list(disp_rates) if disp_rates is not None else [disp_rate] * len(ids)
        if not (len(ids) == len(vols) == len(rates)):
            raise InstrumentError(
                f"ids/disp_vols/disp_rates length mismatch: "
                f"{len(ids)}/{len(vols)}/{len(rates)}", instrument=self.name)

        await self._syringe("head_retract")
        await self._stage("move_to", x=flush_x, y=flush_y)
        await self._syringe("head_descend")
        for idx, pump_id in enumerate(ids):
            await self._syringe(
                "single_pump", res_vol=res_vol, ID=pump_id,
                rate=rates[idx], dispense_vol=vols[idx],
            )
        # Legacy: sleep(vol/rate * settle_factor * 60 + post_flush_dwell); the
        # longest single-pump dispense governs the settle wait.
        longest = max((v / max(r, 1e-9) for v, r in zip(vols, rates)), default=0.0)
        settle = longest * settle_factor * 60.0 + post_flush_dwell_s
        await self._dwell(settle, time_scale)

        await self._syringe("head_retract")
        await self._stage("move_to", x=wick_x, y=wick_y)
        await self._syringe("head_descend")
        await self._dwell(wick_dwell_s, time_scale)
        await self._syringe("head_retract")

        logger.info("startup_flush_done", ids=ids, disp_vols=vols)
        return {"flushed_ids": ids, "dispensed_uL": vols}

    async def single_drop_simul(
        self,
        x: float,
        y: float,
        wick_x: float,
        wick_y: float,
        ids: Sequence[int],
        disp_rate: float,
        vols: Sequence[float],
        deadvols: Sequence[float] | None = None,
        disp_rates: Sequence[float] | None = None,
        disp_rate_total: float | None = None,
        settle_factor: float = 2.0,
        settle_base_s: float = 30.0,
        res_vol: float = 10000.0,
        elution_wait_s: float = 240.0,
        wick_dwell_s: float = 5.0,
        dispense: bool = True,
        time_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Dispense one drop at the electrode ``(x, y)`` — port of ``singleDrop_simul``.

        All components are dispensed simultaneously (varied by volume), then a
        single ``elution_wait_s`` allows the pumps to finish and the drop to mix,
        before the head retracts and wicks.  ``(x, y)`` is the per-channel
        electrode target injected by the HT tab at run time.

        Rate precedence (each pump extrudes at its own rate for equal duration):

        * ``disp_rates`` — an explicit per-pump list (the HT tab precomputes this);
        * else ``disp_rate_total`` — a single total rate **split** here across the
          pumps in proportion to ``vols`` (the autonomous path, whose per-pump
          volumes are only known at run time). When given with ``settle_factor``,
          the settling wait is derived: ``(Σvol / disp_rate_total) × 60 ×
          settle_factor + settle_base_s`` (overriding ``elution_wait_s``);
        * else the scalar ``disp_rate`` applied to every pump (legacy behavior).

        Both split paths reuse :func:`softae.core.dropcast_plan.split_rate`.
        """
        deadvols = list(deadvols) if deadvols is not None else [0.0] * len(list(ids))
        ids = list(ids)
        vols = list(vols)
        if disp_rates is not None:
            rates = list(disp_rates)
        elif disp_rate_total is not None:
            from softae.core.dropcast_plan import split_rate

            rates = split_rate(disp_rate_total, vols)
        else:
            rates = [disp_rate] * len(ids)
        if not (len(ids) == len(vols) == len(deadvols) == len(rates)):
            raise InstrumentError(
                f"ids/vols/deadvols/disp_rates length mismatch: "
                f"{len(ids)}/{len(vols)}/{len(deadvols)}/{len(rates)}",
                instrument=self.name,
            )
        # Derive the settling wait from the total rate when requested (the
        # autonomous path, where the drop volume is only known at run time).
        if disp_rate_total is not None and settle_factor is not None:
            total_v = sum(float(v) for v in vols)
            duration_s = (total_v / max(float(disp_rate_total), 1e-9)) * 60.0
            elution_wait_s = duration_s * float(settle_factor) + float(settle_base_s)

        # Track the point of no return: once any pump has been commanded to
        # elute, this drop cannot be safely re-cast (a retry would double-
        # dispense). We tag any raised error with `dispense_committed` so the
        # executor can choose replay-from-precondition (not yet committed) vs
        # skip-channel (already committed). See WorkflowExecutor recovery.
        dispense_committed = False
        try:
            await self._stage("move_to", x=x, y=y)
            await self._syringe("head_descend")
            if dispense:
                for idx, pump_id in enumerate(ids):
                    # A zeroed formulation component means "leave this pump
                    # alone" — skip it (and its dead volume) entirely, and do not
                    # mark the drop committed, so a later real-pump failure can
                    # still safely replay rather than skip the channel.
                    if float(vols[idx]) <= 0.0:
                        continue
                    dispense_committed = True  # about to command elution
                    await self._syringe(
                        "single_pump",
                        res_vol=res_vol,
                        ID=int(pump_id),
                        rate=float(rates[idx]),
                        dispense_vol=float(vols[idx]) + float(deadvols[idx]),
                    )
            await self._dwell(elution_wait_s, time_scale)

            await self._syringe("head_retract")
            await self._stage("move_to", x=wick_x, y=wick_y)
            await self._syringe("head_descend")
            await self._dwell(wick_dwell_s, time_scale)
            await self._syringe("head_retract")
        except Exception as exc:
            _annotate_dispense_committed(exc, dispense_committed)
            raise

        logger.info("single_drop_simul_done", x=x, y=y, ids=ids, vols=vols)
        return {"electrode_xy": [x, y], "dispensed_ids": ids}

    async def precondition_flush(
        self,
        flush_x: float,
        flush_y: float,
        wick_x: float,
        wick_y: float,
        ids: Sequence[int],
        rate_list: Sequence[float] | None = None,
        vol_list: Sequence[float] | None = None,
        rate_total: float | None = None,
        flush_factor: float = 3.0,
        plug_ids: Sequence[int] = (0, 2),
        plug_rate: float = 1500.0,
        plug_vol: float = 30.0,
        plug_res_vol: float = 1000.0,
        preload_res_vol: float = 1000.0,
        plug_settle_s: float = 80.0,
        plug_dwell_s: float = 20.0,
        wick_dwell_s: float = 5.0,
        dispense: bool = True,
        time_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Precondition the lines with the next formulation — port of ``preconditionFlush``.

        Unlike :meth:`startup_flush` (a fixed daily prime), this varies by the
        formulation fed to it: it wicks, pushes a water/rinsing-agent **plug**
        (``plug_ids``), then **pre-loads** the next composition at scaled volumes
        (``vol_list[i] * flush_factor``) and per-pump rates.

        Rates: pass an explicit per-pump ``rate_list`` (the HT tab precomputes it),
        or a single ``rate_total`` that is **split** here across the pumps in
        proportion to ``vol_list`` (the autonomous path) via
        :func:`softae.core.dropcast_plan.split_rate`.

        (The legacy plug loop referenced an undefined ``idx``; here the plug uses
        the explicit ``plug_rate`` / ``plug_vol``.)
        """
        ids = [int(p) for p in ids]
        vol_list = list(vol_list) if vol_list is not None else []
        if not vol_list:
            raise InstrumentError(
                "precondition_flush requires vol_list", instrument=self.name)
        if rate_list is not None:
            rate_list = list(rate_list)
        elif rate_total is not None:
            from softae.core.dropcast_plan import split_rate

            rate_list = split_rate(rate_total, vol_list)
        else:
            raise InstrumentError(
                "precondition_flush needs rate_list or rate_total",
                instrument=self.name)
        if not (len(ids) == len(rate_list) == len(vol_list)):
            raise InstrumentError(
                f"ids/rate_list/vol_list length mismatch: "
                f"{len(ids)}/{len(rate_list)}/{len(vol_list)}", instrument=self.name)

        # Wick first.
        await self._stage("move_to", x=wick_x, y=wick_y)
        await self._syringe("head_descend")
        await self._dwell(wick_dwell_s, time_scale)
        await self._syringe("head_retract")

        # Push a rinsing-agent plug at the flush basin.
        await self._stage("move_to", x=flush_x, y=flush_y)
        await self._syringe("head_descend")
        for plug_id in plug_ids:
            await self._syringe(
                "single_pump", res_vol=plug_res_vol, ID=int(plug_id),
                rate=plug_rate, dispense_vol=plug_vol,
            )
            await self._dwell(plug_dwell_s, time_scale)
        await self._dwell(plug_settle_s, time_scale)

        # Pre-load the next composition at scaled volumes.
        if dispense:
            for idx, pump_id in enumerate(ids):
                await self._syringe(
                    "single_pump", res_vol=preload_res_vol, ID=pump_id,
                    rate=rate_list[idx], dispense_vol=vol_list[idx] * flush_factor,
                )
        total_v = sum(vol_list) * flush_factor
        total_r = max(sum(rate_list), 1e-9)
        await self._dwell(60.0 + total_v / total_r * 2 * 60.0, time_scale)

        # Wick again.
        await self._syringe("head_retract")
        await self._stage("move_to", x=wick_x, y=wick_y)
        await self._syringe("head_descend")
        await self._dwell(wick_dwell_s, time_scale)
        await self._syringe("head_retract")

        logger.info("precondition_flush_done", ids=ids, vol_list=vol_list,
                    flush_factor=flush_factor)
        return {"preconditioned_ids": ids, "preload_uL": [v * flush_factor for v in vol_list]}

    async def star_mix(
        self,
        x: float,
        y: float,
        r_extent: float = 1.5,
        n_points: int = 6,
        dwell_s: float = 0.3,
        descend: bool = True,
        time_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Star-pattern in-drop mixing at ``(x, y)`` — port of ``mix`` / ``move_and_mix``.

        Moves to the drop centre, (optionally) lowers the head, traces an
        ``n_points`` star of radius ``r_extent`` dwelling ``dwell_s`` at each
        vertex, returns to centre, then retracts.  Set ``descend=False`` to trace
        the pattern with the head already positioned (the bare legacy ``mix``).
        """
        await self._stage("move_to", x=x, y=y)
        if descend:
            await self._syringe("head_descend")
        await self._dwell(dwell_s, time_scale)

        # The ONLY sequence that legitimately translates the stage with the head
        # lowered — tracing the pattern *is* the point, so it opts out of the
        # head guard explicitly rather than being silently exempt.
        # Unconditional: with descend=True this method just lowered the head, and
        # with descend=False the caller lowered it before handing over (the bare
        # legacy ``mix``). Either way the tip is in the drop for the whole trace.
        xs, ys = _star_coordinates(n_points, x, y, r_extent, -r_extent)
        for vx, vy in zip(xs, ys):
            await self._stage("move_to", x=vx, y=vy, head_may_be_down=True)
            await self._dwell(dwell_s, time_scale)
        await self._stage("move_to", x=x, y=y, head_may_be_down=True)

        if descend:
            await self._syringe("head_retract")

        logger.info("star_mix_done", x=x, y=y, n_points=n_points, r_extent=r_extent)
        return {"center": [x, y], "vertices": len(xs)}
