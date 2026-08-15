"""EIS sweep parameters and the MethodSCRIPT files they produce.

An ``eis_measure_step`` carries only a **path** to a ``.mscr`` file — the sweep
parameters themselves (frequency bounds, point count, amplitude) live inside that
file, which some earlier caller must have written. That indirection hid a real
defect: **nothing on the campaign path ever wrote one.** An autonomous run's
measurement step pointed at ``%TEMP%/softae_ch{N}.mscr``, so it either

* failed outright, if no HT or manual run had ever built that channel's file, or
* silently measured with **whatever parameters some previous session happened to
  leave there** — possibly days old, from a different preset.

The second case is the dangerous one: the run records ``eis_preset`` in its
metadata, so the stored provenance asserts a preset that never actually reached
the instrument. Data that looks trustworthy and is not.

This module makes the parameters explicit and writes the scripts, so a campaign
measures with the settings it claims to.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Defaults mirroring ``drivers.mscr_library.eis_run_mscrbuild``.
DEFAULT_F_HI_HZ = 100_000
DEFAULT_F_LO_MHZ = 100
DEFAULT_NPTS = 20
DEFAULT_MV_AC = 10
DEFAULT_MV_DC = 0

#: Effective cell capacitance of the 4-stripe board, from 1152 spectra of
#: 20260811T023757Z_equilibration_characterization: median 0.09 nF, IQR 0.07-0.16.
#: It is a property of the ELECTRODES, not the sample -- R varied 109x across that
#: set while this moved ~2x -- which is what lets `sigma_floor_S_per_cm` predict a
#: reach for a material nobody has measured yet. It is board-specific: a different
#: electrode geometry needs its own number, and this one should move to the board
#: config when a second board is characterised.
CELL_CAPACITANCE_F = 0.09e-9


def sigma_floor_S_per_cm(
    f_lo_hz: float | None,
    L_cm: float | None,
    t_cm: float | None,
    w_cm: float | None,
    *,
    C_F: float = CELL_CAPACITANCE_F,
) -> float | None:
    """Lowest conductivity whose arc still closes inside a sweep ending at *f_lo_hz*.

    The semicircle's apex sits at ``f_peak = 1/(2*pi*R*C)`` and σ = ``L/(R*t*w)``,
    so eliminating R turns a frequency floor into a conductivity floor::

        sigma_min = 2*pi*f_lo*C_cell*L/(t*w)

    Below it the −Z″ peak falls off the bottom of the sweep and R₁ comes from
    extrapolating the high-frequency limb, which measures a **60.9 % median
    overestimate** at only 1.5× past the apex (and 175 % with the full CPE
    fitter). That is a systematic bias, not a widened error bar, so the honest
    move is to state the reach rather than to extrapolate past it.

    Returns ``None`` — never a number computed from a default geometry — when any
    term is absent or non-positive. A σ reach quoted against a geometry nobody
    supplied is exactly the silently-wrong number this refusal exists to prevent;
    it is the posture :func:`~softae.analysis.eis.geometry.resolve_thickness_cm`
    already takes for thickness.
    """
    try:
        terms = [float(x) for x in (f_lo_hz, L_cm, t_cm, w_cm, C_F)]
    except (TypeError, ValueError):
        return None
    if any(x <= 0 for x in terms):
        return None
    f_lo, L, t, w, C = terms
    return 2.0 * math.pi * f_lo * C * L / (t * w)


@dataclass(frozen=True)
class EISParams:
    """A concrete EIS sweep — no preset indirection left to resolve."""

    f_hi: int = DEFAULT_F_HI_HZ
    f_lo_mHz: int = DEFAULT_F_LO_MHZ
    npts: int = DEFAULT_NPTS
    mv_ac: int = DEFAULT_MV_AC
    mv_dc: int = DEFAULT_MV_DC

    @classmethod
    def from_preset(cls, preset: str | None, **overrides: Any) -> "EISParams":
        """Resolve a ``[eis_presets.<name>]`` section, then apply *overrides*.

        An unknown or missing preset falls back to the defaults rather than
        raising — a measurement with known-default settings is recoverable, a
        campaign that refuses to start at 3 a.m. is not. The resolved values are
        what get recorded, so the fallback is visible after the fact.
        """
        values: dict[str, Any] = {}
        if preset:
            try:
                from softae.config.loader import eis_presets

                section = eis_presets().get(preset) or {}
            except Exception:
                section = {}
            if not section:
                logger.warning("eis_preset_unknown", preset=preset)
            for key in ("f_hi", "f_lo_mHz", "npts", "mv_ac", "mv_dc"):
                if key in section:
                    values[key] = section[key]
        values.update({k: v for k, v in overrides.items() if v is not None})

        def _int(key: str, default: int) -> int:
            try:
                return int(values.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            f_hi=_int("f_hi", DEFAULT_F_HI_HZ),
            f_lo_mHz=_int("f_lo_mHz", DEFAULT_F_LO_MHZ),
            npts=_int("npts", DEFAULT_NPTS),
            mv_ac=_int("mv_ac", DEFAULT_MV_AC),
            mv_dc=_int("mv_dc", DEFAULT_MV_DC),
        )

    def as_metadata(self) -> dict[str, int]:
        """The values actually applied, for run provenance."""
        return {
            "eis_f_hi": self.f_hi,
            "eis_f_lo_mHz": self.f_lo_mHz,
            "eis_npts": self.npts,
            "eis_mv_ac": self.mv_ac,
            "eis_mv_dc": self.mv_dc,
        }


def mscr_path_for_channel(channel: int) -> str:
    """Where :func:`~softae.core.deposition_steps.eis_measure_step` looks."""
    return os.path.join(tempfile.gettempdir(), f"softae_ch{int(channel)}.mscr")


def build_eis_scripts(channels, params: EISParams) -> list[str]:
    """Write a ``.mscr`` per channel; return the paths written.

    Always overwrites, so a stale file from an earlier session with different
    parameters cannot survive into this run. Best-effort per channel: a channel
    that cannot be written is logged and skipped rather than aborting the whole
    run, since the executor will surface the missing script as a step failure
    with far better context than a build-time crash.
    """
    from softae.drivers.mscr_library import eis_run_mscrbuild

    written: list[str] = []
    for ch in channels:
        path = mscr_path_for_channel(ch)
        try:
            eis_run_mscrbuild(
                path, mux_ch=int(ch),
                mVac=params.mv_ac, f_hi=params.f_hi, f_lo=params.f_lo_mHz,
                npts=params.npts, mVdc=params.mv_dc,
            )
            written.append(path)
        except Exception:
            logger.warning("eis_script_build_failed", channel=int(ch), path=path,
                           exc_info=True)
    logger.info("eis_scripts_built", n=len(written), **params.as_metadata())
    return written
