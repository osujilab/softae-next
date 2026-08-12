"""Shared deposition building blocks — the pieces every deposition builder needs.

Single source for the stage-calibration fallback positions, PCB resolution,
the shared dispense defaults, and the literal :class:`WorkflowStep` shapes
(startup flush, single-drop cast, per-channel EIS, final flush) that the
sweep builder (:mod:`softae.core.dropcast`), the autonomous trial template
(:mod:`softae.core.autonomous_wiring`) and the recipe engine
(:mod:`softae.core.deposition_recipe`) previously each declared by hand —
three copies that had already drifted.

Positions come from ``stage_calibration()`` with the fallbacks defined here
(and only here); electrode ``(x, y)`` stays a build-time property resolved
from PCB geometry + the calibrated ``dep1`` origin, never a recipe value.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Sequence

from softae.config.loader import (
    default_pcb_name,
    pcb_configs,
    pico_for_channel,
    stage_calibration,
)
from softae.workflows.workflow_model import WorkflowStep

# ── Shared dispense defaults (previously duplicated per-module) ───────────────

#: Default total dispense rate for a drop-cast, µL/min.
DEFAULT_DISP_RATE_UL_MIN = 75.0
#: Default post-deposit settling wait, seconds.
DEFAULT_ELUTION_WAIT_S = 240.0
#: Default line-flush rate, µL/min (startup prime / final flush).
DEFAULT_FLUSH_RATE_UL_MIN = 200.0
#: Default flush volume, µL (startup prime / final flush, per pump).
DEFAULT_FLUSH_VOL_UL = 80.0

# Stage-calibration fallbacks (used only when the config carries no value).
_FALLBACK_ORIGIN = (43.5, 50.0)   # dep1 = electrode-1 origin
_FALLBACK_FLUSH = (-50.0, 50.0)
_FALLBACK_WICK = (-50.0, -25.0)


@dataclass(frozen=True)
class DepositionPositions:
    """The three calibrated stage positions every deposition builder uses."""

    origin: tuple[float, float]  # dep1: electrode-1 origin for PCB geometry
    flush: tuple[float, float]
    wick: tuple[float, float]


def deposition_positions() -> DepositionPositions:
    """Resolve dep1/flush/wick from stage calibration (single fallback source)."""
    cal = stage_calibration()
    return DepositionPositions(
        origin=(cal.get("dep1_x", _FALLBACK_ORIGIN[0]),
                cal.get("dep1_y", _FALLBACK_ORIGIN[1])),
        flush=(cal.get("flush_x", _FALLBACK_FLUSH[0]),
               cal.get("flush_y", _FALLBACK_FLUSH[1])),
        wick=(cal.get("wick_x", _FALLBACK_WICK[0]),
              cal.get("wick_y", _FALLBACK_WICK[1])),
    )


def resolve_pcb(pcb_name: str | None) -> tuple[str, dict[str, Any]]:
    """Return ``(name, pcb_dict)``; default to the configured default PCB.

    Degenerate fallback (no PCBs configured) keeps demos working without a
    config file.
    """
    pcbs = pcb_configs()
    if pcb_name and pcb_name in pcbs:
        return pcb_name, pcbs[pcb_name]
    if not pcbs:
        return "default_4x4", {"grid": [4, 4], "spacing_mm": [15, 15]}
    name = pcb_name or default_pcb_name() or sorted(pcbs)[0]
    return name, pcbs[name]


# ── Literal step shapes ───────────────────────────────────────────────────────

def startup_flush_step(
    positions: DepositionPositions,
    ids: Sequence[int],
    *,
    disp_rate: float,
    disp_vol: float,
    disp_vols: Sequence[float] | None = None,
    time_scale: float = 1.0,
) -> WorkflowStep:
    """Campaign-start line prime at the flush/wick stations.

    ``disp_vols`` (per-pump volumes) supersedes the scalar ``disp_vol`` when
    given — the driver requires ``disp_vol`` regardless, so it is always sent.
    """
    params: dict[str, Any] = {
        "flush_x": positions.flush[0], "flush_y": positions.flush[1],
        "wick_x": positions.wick[0], "wick_y": positions.wick[1],
        "disp_rate": disp_rate, "disp_vol": disp_vol,
        "ids": list(ids), "time_scale": time_scale,
    }
    if disp_vols is not None:
        params["disp_vols"] = list(disp_vols)
    return WorkflowStep(
        name="startup_flush",
        instrument="liquid_handler",
        method="startup_flush",
        params=params,
        timeout_s=900,
    )


def single_drop_step(
    channel: int,
    xy: tuple[float, float],
    positions: DepositionPositions,
    *,
    ids: Sequence[int],
    vols: Sequence[Any],  # floats, or "$var" strings in an optimizer template
    disp_rate: float,
    deadvols: Sequence[float],
    elution_wait_s: float | None = None,
    time_scale: float = 1.0,
    name: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> WorkflowStep:
    """One ``single_drop_simul`` cast at a channel's electrode.

    ``extra_params`` carries builder-specific additions (e.g. ``wick_dwell_s``
    for the sweep, or the driver-deferred two-phase split knobs
    ``disp_rate_total``/``settle_factor``/``settle_base_s`` for the autonomous
    template, where volumes are still symbolic at build time).
    """
    params: dict[str, Any] = {
        "x": xy[0], "y": xy[1],
        "wick_x": positions.wick[0], "wick_y": positions.wick[1],
        "ids": list(ids), "disp_rate": disp_rate,
        "vols": list(vols), "deadvols": list(deadvols),
        "time_scale": time_scale,
    }
    if elution_wait_s is not None:
        params["elution_wait_s"] = elution_wait_s
    if extra_params:
        params.update(extra_params)
    return WorkflowStep(
        name=name or f"deposit_ch{channel}",
        instrument="liquid_handler",
        method="single_drop_simul",
        params=params,
        timeout_s=600,
        tags={"position": "electrode", "channel": str(channel)},
    )


def eis_measure_step(channel: int, *, name: str | None = None) -> WorkflowStep:
    """Per-channel EIS measurement, routed to the channel's pico."""
    return WorkflowStep(
        name=name or f"measure_eis_ch{channel}",
        instrument=pico_for_channel(channel),
        method="sendscript_getdata",
        params={
            "mscrpath": os.path.join(tempfile.gettempdir(), f"softae_ch{channel}.mscr"),
            "outdir": os.path.join(tempfile.gettempdir(), "softae_eis_output"),
            "chan": channel,
        },
        timeout_s=600,
        retry=1,
        # "measurement": "primary" is the loop-closure predicate's default and
        # could be omitted; emitted explicitly (T1.5) so the step self-describes
        # as a campaign objective input — see
        # ``autonomous_wiring.is_primary_measurement``. Consumers that repurpose
        # this step for calibration (commissioning, geometry series) spread
        # these tags and add a non-"sample" ``role``, which excludes them there.
        tags={"channel": str(channel), "measurement": "primary"},
    )


def final_flush_step(pump_id: int) -> WorkflowStep:
    """Teardown line flush on one pump."""
    return WorkflowStep(
        name="final_flush",
        instrument="syringe",
        method="single_pump",
        params={
            "res_vol": 1000, "ID": pump_id,
            "rate": DEFAULT_FLUSH_RATE_UL_MIN, "dispense_vol": DEFAULT_FLUSH_VOL_UL,
        },
    )
