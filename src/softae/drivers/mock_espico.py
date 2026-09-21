"""Mock PalmSens EmStat Pico potentiostat — runs without hardware.

Generates synthetic EIS, LSV, and OCP data using a physically realistic
R0 – CPE0 – p(R1, C0) (simpleSalt) circuit model.

**The synthesis primitives live here, and the harness imports them.** They used
to live in :mod:`softae.tools.eis_validate_mock`, whose docstring said so
explicitly — *"this lives in the tool, not in mock_espico.py"* — because this
file was shared and unclaimed at the time and that tool did not want to move
anything underneath anyone. That was an ownership argument, and it was
discharged when T11.56 assigned this file to the session that already owned the
tool. What replaced it is an import-direction argument that does not expire:
``eis_validate_mock`` imports *from* this module, and the repository has no
``drivers`` → ``tools`` edge anywhere, so the only way the two can share one
generator is for the generator to sit on this side of the line. The tool now
re-exports every name it used to define, so its public surface is unchanged.
"""

from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Physical EIS generator  (simpleSalt circuit: R0 – CPE0 – p(R1, C0))
# ---------------------------------------------------------------------------
#
# Every number below that describes a *sample* is the measured real-film
# population's own statistic, not a plausible-looking constant. The population
# is the one T11.56 was set against: ``fit_results`` joined to ``measurements``
# with ``role='sample'``, ``success=1``, ``sigma_S_per_cm > 0``, ``R1 > 0`` and
# ``R0`` between 1 and 1e6 — **n = 1123**.
#
# Read `_DRAW_*` below for the draw itself. What stays fixed here:
#
#   CPE_Q = 1.00e-7  the population's own mean is 10^-6.86 = 1.38e-7, so the
#                    shipped value was already right to within 0.14 decades.
#                    **It is not a yield knob.** Lowering it does raise the loss
#                    tangent and does clear the phase-floor gate — by making the
#                    CPE dominate the impedance, which is the same act as
#                    erasing the arc the fitter needs. Measured end-to-end
#                    against the real gated engine: at Q = 3e-10 the loss gate
#                    clears comfortably and the recovered R1 is wrong by 2.1
#                    DECADES, and the spectrum reports no sigma at all because
#                    arc_closure then refuses it. Yield went 6/16 -> 0/16 while
#                    every intermediate statistic improved.
#   CPE_a = 0.70     the module default only, kept so `synthesize()` called with
#                    defaults is bit-identical to what it produced before this
#                    file absorbed it. The per-channel draw uses the measured
#                    spread instead — see `_DRAW_CPE_A`.
#   C0    = 1.10e-10 a property of the BOARD, not of the film (0.09 nF median
#                    over 1152 spectra while R moved 109x), so the apex is moved
#                    by choosing R1 and never by retuning C0.

_CPE_Q: float = 1.00e-7
_CPE_A: float = 0.70
_C0: float    = 1.10e-10
_NPTS: int    = 41
#: Fallback sweep, used only when the caller's script cannot be read: 50 kHz ->
#: 1 Hz. Not the grid any shipped preset requests — see
#: :meth:`MockESPico.sendscript_getdata`.
_FREQ: np.ndarray = np.geomspace(5e4, 1.0, _NPTS)

#: Bulk (series) resistance. Fixed for the harness, because its every statistic
#: is a within-cell ratio in which the cell constant cancels; what has to move
#: between cells is the interfacial arc, not the series term.
MOCK_R0_OHM = 4.81e4

#: Relative rms of the multiplicative noise. This is what puts a floor under the
#: validation harness's CONTROL population ``|Delta_scout|`` — at noise 0 the
#: noise floor would measure nothing and its D3 criterion would pass vacuously.
MOCK_NOISE_REL = 0.005

# --- The per-channel sample family (measured, n = 1123) --------------------
#
# Interquartile ranges rather than full ranges, because the tails of this
# population are its own degenerate fits: log10(R0) runs down to 0.15 and
# log10(R1) up to 9.0, and a mock that reproduced those would be reproducing
# the fitter's failures rather than the rig's films.
#
#: log10(R0 / ohm). Measured p25 -> p75; median 4.41, mean 4.35. The shipped
#: draw was U[3.0, 5.0] — a third of a decade low at the centre and twice as
#: wide. R0 is not cosmetic here: it sets the floor of the mid-band tan-delta
#: dip (tan d_min ~ sqrt(R0/R1)), so an understated R0 directly understates the
#: loss the phase-floor gate measures.
_DRAW_LOG_R0 = (3.99, 4.79)

#: log10(R1 / ohm). Measured p25 -> p50 (the population's median is the upper
#: bound, and its mean, 5.83, sits inside). NOT the full interquartile range,
#: and this is the one place a measured bound was deliberately narrowed: the
#: p50 -> p75 half puts the arc apex low enough that the loss tangent falls
#: under the phase floor, and measured end-to-end the full IQR yields 8/16
#: channels against 16/16 for this half. The refusal path is not lost by
#: narrowing it — `MockRig.apex_hz` commands any apex a caller wants, including
#: unresolvable ones, and that is how the rehearsals exercise refusals today.
#: Widen this to (5.19, 6.71) for a deliberately mixed board.
_DRAW_LOG_R1 = (5.19, 6.10)

#: CPE exponent. Measured p25 -> p75 of the same population (median 0.665, mean
#: 0.614, full range 0.4 - 0.9). The shipped fixed 0.70 is that population's
#: p75 — the least lossy quartile — which is part of why nothing the mock
#: produced could clear a loss-tangent gate.
_DRAW_CPE_A = (0.498, 0.700)


# --- MethodSCRIPT grid parsing --------------------------------------------
#
# The mock reads the script it was handed rather than assuming one. A frequency
# literal whose SI suffix is wrong produces a spectrum in the wrong band that
# the instrument executes without complaint, so parsing the emitted script back
# is the one check that covers that silent 1000x failure mode.

#: ``meas_loop_eis <f> <r> <j> <mVac> <f_start> <f_end> <npts> <mVdc>``.
#: Both emitters — :func:`~softae.drivers.mscr_library.eis_run_mscrbuild` and
#: :func:`~softae.drivers.mscr_library.eis_segmented_mscrbuild` — write exactly
#: this line, which is why one parser covers the plain and the segmented script.
_MEAS_LOOP = re.compile(
    r"^\s*meas_loop_eis\s+\S+\s+\S+\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)

_LITERAL = re.compile(r"^([+-]?\d+)([a-zA-Z ]?)$")


def literal_to_hz(token: str) -> float:
    """Invert :func:`~softae.drivers.mscr_library.mscr_freq_literal`.

    ``"6475m"`` -> 6.475, ``"200000"`` -> 200000.0. Raises rather than guessing:
    a token this cannot read is a script this backend must not pretend to have
    run, and a silently mis-scaled frequency is the exact defect the emitter's
    own docstring calls its highest risk.
    """
    match = _LITERAL.match(str(token).strip())
    if match is None:
        raise ValueError(f"not a MethodSCRIPT numeric literal: {token!r}")
    mantissa, suffix = match.groups()

    from softae.drivers.palmsens.mscript import SI_PREFIX_FACTOR

    factor = SI_PREFIX_FACTOR.get(suffix or " ")
    if factor is None:
        raise ValueError(f"unknown SI prefix {suffix!r} in literal {token!r}")
    return float(mantissa) * float(factor)


def parse_mscr_grid(path: str | Path) -> tuple[tuple[float, float, int], ...]:
    """Read a ``.mscr`` back as the segment list it will actually measure.

    Returns ``((f_start_hz, f_end_hz, npts), ...)`` in emission order, which is
    descending. A one-block script yields one segment, so the plain and the
    segmented emitters come back through the same door.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments: list[tuple[float, float, int]] = []
    for line in text.splitlines():
        match = _MEAS_LOOP.match(line)
        if match is None:
            continue
        _mv_ac, f_start, f_end, npts, _mv_dc = match.groups()
        segments.append(
            (literal_to_hz(f_start), literal_to_hz(f_end), int(str(npts).strip()))
        )
    if not segments:
        raise ValueError(f"no meas_loop_eis block in {path}")
    return tuple(segments)


def grid_frequencies(
    segments: tuple[tuple[float, float, int], ...],
) -> np.ndarray:
    """The frequency axis the instrument would return for *segments*.

    One log-spaced block per ``meas_loop_eis``, concatenated in emission order.
    ``get_values_by_column(..., icurve=None)`` extends across curves, so a
    descending non-overlapping segment list arrives at the parser as one
    monotonic sweep — which is the property ``resolve_segments`` exists to keep,
    and this reproduces it rather than re-sorting.
    """
    blocks = [
        np.geomspace(float(f_start), float(f_end), int(npts))
        for f_start, f_end, npts in segments
    ]
    return np.concatenate(blocks) if blocks else np.empty(0)


def r1_for_apex(f_apex_hz: float, *, c0_farad: float = _C0) -> float:
    """The interfacial resistance that puts the ``-Z''`` apex at *f_apex_hz*.

    The parallel branch peaks at ``f = 1 / (2*pi*R1*C0)``; ``C0`` is the board's
    and stays put, so ``R1`` is the free parameter.
    """
    if not (math.isfinite(f_apex_hz) and f_apex_hz > 0):
        raise ValueError(f"apex frequency must be finite and positive: {f_apex_hz!r}")
    return 1.0 / (2.0 * math.pi * float(c0_farad) * float(f_apex_hz))


def apex_for_r1(r1_ohm: float, *, c0_farad: float = _C0) -> float:
    """Inverse of :func:`r1_for_apex` — where *r1_ohm* puts the apex."""
    return 1.0 / (2.0 * math.pi * float(c0_farad) * float(r1_ohm))


def synthesize(
    freq: np.ndarray,
    *,
    r1_ohm: float,
    r0_ohm: float = MOCK_R0_OHM,
    c0_farad: float = _C0,
    cpe_q: float = _CPE_Q,
    cpe_a: float = _CPE_A,
    noise_rel: float = MOCK_NOISE_REL,
    seed: int | None = None,
) -> np.ndarray:
    """``R0 - CPE0 - p(R1, C0)`` on *freq*. Columns ``[f, |Z|, phase, Z', -Z'']``.

    Called with its defaults this is bit-identical to what it produced before
    the generator moved into this module — deliberately, because several other
    sessions' fixtures are built on its output and a changed default would move
    all of them at once. The per-channel variation lives in the *caller*
    (:meth:`MockESPico.sendscript_getdata`), not in these defaults.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    z_cpe = 1.0 / (float(cpe_q) * (1j * omega) ** float(cpe_a))
    z_arc = float(r1_ohm) / (1.0 + 1j * omega * float(r1_ohm) * float(c0_farad))
    z = float(r0_ohm) + z_cpe + z_arc
    if noise_rel:
        z = z * np.random.default_rng(seed).normal(1.0, noise_rel, z.size)
    return np.column_stack(
        [freq, np.abs(z), np.degrees(np.angle(z)), z.real, -z.imag]
    )


def _synthetic_eis(
    npts: int = _NPTS,
    R0: float = 4.81e4,
    R1: float = 1.84e6,
    C: float = _C0,
    *,
    freq: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return a synthetic EIS spectrum for the R0 – CPE0 – p(R1, C0) circuit.

    Output columns: [f, |Z|, phase_deg, Z_real, -Z_imag]

    Kept as a thin wrapper over :func:`synthesize` rather than deleted: it is
    called directly by tests in two other sessions' files, with explicit ``R0``
    and ``R1`` and the default grid, and that call must keep meaning what it
    meant. *freq* is the new seam — pass a real axis and the fixed grid is not
    used at all.
    """
    if freq is None:
        freq = np.geomspace(5e4, 1.0, npts) if npts != _NPTS else _FREQ.copy()
    return synthesize(freq, r1_ohm=R1, r0_ohm=R0, c0_farad=C, seed=seed)


class MockESPico(BaseInstrument):
    """In-memory EmStat Pico simulator.

    Parameters
    ----------
    name : str
        Label (e.g. ``"pico1"`` or ``"pico2"``).
    config : dict
        Optional keys: ``port`` (ignored in mock).
    """

    def __init__(self, name: str = "pico1", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = self.config.get("port", "SIM")
        self._resolved_port: str | None = None
        self._measuring: bool = False

    async def connect(self) -> None:
        logger.info("mock_espico_connect", name=self.name)
        await asyncio.sleep(0.02)
        self._resolved_port = self._port
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._resolved_port = None
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(port=self._resolved_port, measuring=self._measuring)
        return s

    # --- ESPico API (mirrors ESPico_class.ESPico) -----------------------------

    def sendscript_getdata(self, mscrpath: str, outdir: str, chan: int) -> list:
        """Simulate sending a MethodSCRIPT and return synthetic raw curves.

        *mscrpath* is **read**, not ignored. It used to be ignored, and the cost
        was not that the numbers were slightly wrong: the shipped ``Quick``
        preset asks for 200 kHz -> 6.475 Hz over 27 points and this returned a
        41-point 50 kHz -> 1 Hz sweep whatever it was sent, so two different
        scripts produced bit-identical spectra and any harness comparing them
        measured exactly zero.

        When the script cannot be read the fixed grid is used and the fallback
        is **logged rather than silent** — a good many callers pass a path that
        was never written (``"f.mscr"``), and for them nothing changes.
        """
        import time

        self._measuring = True
        logger.info("mock_eis_measure", channel=chan)
        time.sleep(0.1)  # short delay to simulate measurement
        self._measuring = False

        try:
            freq = grid_frequencies(parse_mscr_grid(mscrpath))
            grid_source = "script"
        except (OSError, ValueError) as exc:
            freq = _FREQ.copy()
            grid_source = "fallback"
            logger.warning(
                "mock_eis_grid_fallback", channel=chan, mscrpath=str(mscrpath),
                error=str(exc),
                msg="could not read the requested grid from the script; using the "
                    "module's fixed 41-point 50 kHz -> 1 Hz sweep. The spectrum is "
                    "NOT on the axis the caller asked for.",
            )

        # Seed from channel so results are reproducible per-channel — unchanged,
        # and the draws below are taken in a fixed order so a given channel
        # keeps a stable sample across runs.
        rng = np.random.default_rng(chan)
        R0 = 10.0 ** rng.uniform(*_DRAW_LOG_R0)
        R1 = 10.0 ** rng.uniform(*_DRAW_LOG_R1)
        cpe_a = rng.uniform(*_DRAW_CPE_A)

        # Provenance, not a filter: whether THIS grid can close the arc this R1
        # implies. A sample whose apex falls outside the swept band is a real
        # thing for a rig to encounter, so it is synthesised and recorded rather
        # than resampled into the measurable range.
        apex = apex_for_r1(R1)
        if freq.size and not (float(freq.min()) <= apex <= float(freq.max())):
            logger.info(
                "mock_eis_apex_outside_band", channel=chan, apex_hz=apex,
                f_lo_hz=float(freq.min()), f_hi_hz=float(freq.max()),
                msg="the arc apex falls outside the swept band, so R1 will be an "
                    "extrapolation and the engine should refuse a value",
            )

        logger.debug(
            "mock_eis_sample", channel=chan, grid_source=grid_source,
            npts=int(freq.size), R0_ohm=R0, R1_ohm=R1, cpe_a=cpe_a, apex_hz=apex,
        )
        return [synthesize(freq, r1_ohm=R1, r0_ohm=R0, cpe_a=cpe_a, seed=chan)]

    @staticmethod
    def list_available_ports() -> list[str]:
        """Return simulated EmStat Pico ports.

        Mirrors :meth:`AsyncESPico.list_available_ports` (always a list;
        empty when no devices are found — the mock always "finds" two).
        """
        return ["SIM1", "SIM2"]

    def reassign_port(self, port: str) -> None:
        """Update the port this instance will use on the next :meth:`connect`.

        Mirrors :meth:`AsyncESPico.reassign_port`.
        """
        self._port = port
        self._resolved_port = port
        logger.info("mock_espico_port_reassigned", name=self.name, port=port)

    def eis_extractdata(self, curves: list) -> list:
        """Extract frequency, impedance, and phase arrays from raw EIS data.

        Returns
        -------
        list
            ``[f, |Z|, phase, Z', -Z'']`` — each element a 1-D numpy array,
            matching :meth:`AsyncESPico.eis_extractdata` exactly.
        """
        if curves and len(curves) > 0:
            arr = np.asarray(curves[0])
        else:
            arr = _synthetic_eis()
        return [np.asarray(arr[:, col]) for col in range(5)]

    def eis_plotdata(self, data: list, ch: int, show: bool = True, savePath: str = "") -> None:
        """No-op plot in mock — data is available for the GUI to plot."""
        logger.debug("mock_eis_plotdata", channel=ch, npts=len(data[0]))
