"""A mock EmStat Pico that reads the ``.mscr`` it was handed.

**The shipped mock cannot exercise this tool.**
:meth:`softae.drivers.mock_espico.MockESPico.sendscript_getdata` ignores
*mscrpath* entirely and returns a fixed 41-point 50 kHz -> 1 Hz sweep seeded
from the channel number. Under it, the validation harness's reference arm and
its adaptive arm return **bit-identical spectra regardless of which script
ran**, so:

- every ``Delta`` is exactly 0;
- verdicts do not respond to the grid, so ``extend_low`` never fires and no
  follow-up is ever built;
- ``Delta_hold`` is identically 0, so the hold-integrity criterion H3 always
  passes;
- **a mock run prints a perfect null indistinguishable from a real result.**

That is worse than no mock. It is the failure mode
:func:`softae.tools.eis_timing._print_report` guards against in prose -- *"a
mock run measures how fast this host can write a .mscr file and nothing
whatever about the rig"* -- except that here it would be silently fabricating
the science rather than merely the timings.

So this module parses the emitted script **back**, inverting
``mscr_freq_literal`` via :data:`palmsens.mscript.SI_PREFIX_FACTOR`, and
evaluates the circuit on *those* bounds and *those* point counts. Reading the
planned grid off the planner instead would test everything except the one unit
with a **silent 1000x failure mode**: a frequency literal whose suffix is wrong
produces a spectrum in the wrong band that the instrument executes without
complaint.

**This lives in the tool, not in ``mock_espico.py``.** The whole tree shares
that double; nothing there moves underneath anyone because of this file.

**Where the apex goes, and why R moves rather than C.** The board's cell
capacitance is a property of the *board* -- 0.09 nF median over 1152 spectra
while R moved 109x -- so an apex at a commanded frequency is produced by
solving ``R1 = 1 / (2*pi*C0*f_apex)`` and leaving ``C0`` alone. That is the
same story the ``[eis_presets.Quick]`` config comment tells, and it means the
three populations are reachable by asking for an apex:

===================  ==================  ============================
``--mock-apex-hz``   population          why
===================  ==================  ============================
> 64.75 Hz           CONTROL             the baseline sweep closes it
13.51 - 64.75 Hz     TREATMENT           only the reference closes it
< 13.51 Hz           UNRESOLVED          neither closes it
===================  ==================  ============================

(the two thresholds are ``f_lo`` of the baseline and reference presets times
one decade of required band, and the harness derives them from the resolved
presets rather than hard-coding these numbers.)
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from softae.drivers.mock_espico import _C0, _CPE_A, _CPE_Q, MockESPico
from softae.drivers.mock_rh_controller import MockRHController
from softae.drivers.mock_temp_controller import MockTempController

logger = structlog.get_logger(__name__)

#: Bulk (series) resistance. Fixed, because the harness's every statistic is a
#: within-cell ratio in which the cell constant cancels; what has to move
#: between cells is the interfacial arc, not the series term.
MOCK_R0_OHM = 4.81e4

#: Relative rms of the multiplicative noise, matching the shipped mock. This is
#: what puts a floor under the CONTROL population's ``|Delta_scout|`` -- with
#: noise at 0 the noise floor would measure nothing and D3 would pass vacuously.
MOCK_NOISE_REL = 0.005

#: ``meas_loop_eis <f> <r> <j> <mVac> <f_start> <f_end> <npts> <mVdc>``.
#: Both emitters -- :func:`~softae.drivers.mscr_library.eis_run_mscrbuild` and
#: :func:`~softae.drivers.mscr_library.eis_segmented_mscrbuild` -- write exactly
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
    monotonic sweep -- which is the property ``resolve_segments`` exists to
    keep, and this reproduces it rather than re-sorting.
    """
    blocks = [
        np.geomspace(float(f_start), float(f_end), int(npts))
        for f_start, f_end, npts in segments
    ]
    return np.concatenate(blocks) if blocks else np.empty(0)


def r1_for_apex(f_apex_hz: float, *, c0_farad: float = _C0) -> float:
    """The interfacial resistance that puts the ``-Z''`` apex at *f_apex_hz*.

    The parallel branch peaks at ``f = 1 / (2*pi*R1*C0)``; ``C0`` is the board's
    and stays put, so ``R1`` is the free parameter. See the module docstring.
    """
    if not (math.isfinite(f_apex_hz) and f_apex_hz > 0):
        raise ValueError(f"apex frequency must be finite and positive: {f_apex_hz!r}")
    return 1.0 / (2.0 * math.pi * float(c0_farad) * float(f_apex_hz))


def synthesize(
    freq: np.ndarray,
    *,
    r1_ohm: float,
    r0_ohm: float = MOCK_R0_OHM,
    c0_farad: float = _C0,
    noise_rel: float = MOCK_NOISE_REL,
    seed: int | None = None,
) -> np.ndarray:
    """``R0 - CPE0 - p(R1, C0)`` on *freq*. Columns ``[f, |Z|, phase, Z', -Z'']``.

    The same circuit and the same CPE constants the shipped mock uses -- what
    changes here is only that the frequency axis is the script's rather than a
    module-level constant.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    z_cpe = 1.0 / (_CPE_Q * (1j * omega) ** _CPE_A)
    z_arc = float(r1_ohm) / (1.0 + 1j * omega * float(r1_ohm) * float(c0_farad))
    z = float(r0_ohm) + z_cpe + z_arc
    if noise_rel:
        z = z * np.random.default_rng(seed).normal(1.0, noise_rel, z.size)
    return np.column_stack(
        [freq, np.abs(z), np.degrees(np.angle(z)), z.real, -z.imag]
    )


@dataclass
class MockRig:
    """The synthetic sample every mock pico in a run shares.

    One object, not one per instrument, so channels 1-16 and 17-32 drift on the
    same clock: a run split across two picos must not observe two different
    samples, and the drift check would report the split rather than the hold.
    """

    #: Apex frequency per channel; :attr:`default_apex_hz` covers the rest.
    apex_hz: dict[int, float] = field(default_factory=dict)
    default_apex_hz: float = 30.0
    #: Decades of ``log10(sigma)`` per hour. Positive means sigma **rises**, so
    #: ``R1`` falls -- the sign an operator reads as "the film is still drying
    #: out of the wet state" rather than an arbitrary convention.
    drift_decades_per_hour: float = 0.0
    #: Injectable clock. ``time.monotonic`` in production; tests advance a list.
    now: Any = time.monotonic
    #: A **virtual** clock: when positive, elapsed time is counted in sweeps
    #: rather than seconds. A mock run takes milliseconds where the rig takes an
    #: hour, so a drift expressed per hour would be unobservable on the wall
    #: clock and ``Delta_hold`` would come back identically zero -- which is the
    #: exact failure this whole module exists to prevent.
    virtual_s_per_sweep: float = 0.0
    noise_rel: float = MOCK_NOISE_REL
    _t0: float | None = field(default=None, repr=False)
    _sweeps: int = field(default=0, repr=False)

    def elapsed_hours(self) -> float:
        if self.virtual_s_per_sweep > 0:
            return self._sweeps * self.virtual_s_per_sweep / 3600.0
        t = float(self.now())
        if self._t0 is None:
            self._t0 = t
        return max(0.0, (t - self._t0) / 3600.0)

    def r1_now(self, channel: int) -> float:
        """``R1`` for *channel* at the current moment, drift included."""
        apex = float(self.apex_hz.get(int(channel), self.default_apex_hz))
        r1 = r1_for_apex(apex)
        if self.drift_decades_per_hour:
            r1 *= 10.0 ** (-self.drift_decades_per_hour * self.elapsed_hours())
        return r1

    def measure(self, channel: int, script_path: str | Path) -> np.ndarray:
        """Parse *script_path*, evaluate the circuit on its grid, return columns."""
        segments = parse_mscr_grid(script_path)
        freq = grid_frequencies(segments)
        self._sweeps += 1
        spectrum = synthesize(
            freq,
            r1_ohm=self.r1_now(channel),
            noise_rel=self.noise_rel,
            # Seeded from the sweep index as well as the channel, so two sweeps
            # on one cell are not the same numbers. Without that, a repeat
            # measurement would carry zero replicate scatter and the CONTROL
            # noise floor would be an artifact of the mock.
            seed=int(channel) * 10_000 + self._sweeps,
        )
        logger.debug(
            "eis_validate_mock_sweep",
            channel=int(channel),
            n_segments=len(segments),
            npts=int(freq.size),
            f_hi_hz=float(freq[0]) if freq.size else float("nan"),
            f_lo_hz=float(freq[-1]) if freq.size else float("nan"),
        )
        return spectrum


class GridAwareMockPico(MockESPico):
    """A :class:`MockESPico` whose spectrum depends on the script it was sent.

    Subclassed rather than patched: everything except ``sendscript_getdata`` --
    connect, disconnect, status, the manager's view of it -- is the shipped
    behaviour, and stays that way.
    """

    def __init__(
        self,
        name: str = "pico1",
        config: dict[str, Any] | None = None,
        *,
        rig: MockRig | None = None,
    ) -> None:
        super().__init__(name, config)
        self.rig = rig if rig is not None else MockRig()
        self._output_dir = str((config or {}).get("output_dir", ""))

    def sendscript_getdata(self, mscrpath: str, outdir: str, chan: int) -> list:
        """Measure what *mscrpath* actually asks for.

        No ``time.sleep``: the shipped mock's 0.1 s stands in for a measurement,
        and this backend is driven by a harness that runs hundreds of sweeps in
        a test. Wall-clock realism belongs to the projection, which is modelled
        from :mod:`softae.core.preflight`'s anchors and not from the mock.
        """
        self._measuring = True
        try:
            return [self.rig.measure(chan, mscrpath)]
        finally:
            self._measuring = False


class FastMockTempController(MockTempController):
    """The shipped mock's physics, on the caller's clock instead of the wall's.

    ``MockTempController._update_sim`` drives a first-order response against
    ``time.time()`` -- right for a GUI demo, and unusable here: an approach that
    is *correct* to take twenty real minutes means the harness's approach,
    settle and refusal paths are never exercised by any test or smoke run. This
    subclass advances the same first-order response **per call** instead, so the
    number of polls, not the wall clock, is what converges it.

    Only ``_update_sim`` is overridden. Every public method, every guard and
    every validation stays the shipped one.
    """

    #: Fraction of the remaining error closed per read. 0.25 puts a 15 C step
    #: inside a 2 C band in ~7 polls, which is enough polls that a timeout path
    #: is still reachable by shortening the timeout.
    approach_fraction = 0.25

    def _update_sim(self) -> None:
        self._pv += (self._sp - self._pv) * self.approach_fraction
        self._pv_surf = self._pv - 0.5
        self._last_update = time.time()


class FastMockRHController(MockRHController):
    """As :class:`FastMockTempController`, for the humidity axis."""

    approach_fraction = 0.25

    def _update_sim(self) -> None:
        if self._running:
            self._rh += (self._setpoint - self._rh) * self.approach_fraction
        self._temp += (23.0 - self._temp) * self.approach_fraction
        self._last_update = time.time()


def install_mock_picos(manager: Any, rig: MockRig) -> MockRig:
    """Replace every registered pico with a grid-aware one sharing *rig*.

    Returns *rig* so a caller can keep the handle it needs to advance the clock.
    """
    for name in ("pico1", "pico2"):
        try:
            existing = manager.get(name)
        except Exception:
            continue
        config = dict(getattr(existing, "config", {}) or {})
        manager._instruments[name] = GridAwareMockPico(name, config, rig=rig)
    return rig


def install_fast_conditions(manager: Any) -> None:
    """Swap the temp/RH mocks for per-call ones. ``--mock`` only."""
    for name, cls in (("temp_controller", FastMockTempController),
                      ("rh_controller", FastMockRHController)):
        try:
            existing = manager.get(name)
        except Exception:
            continue
        manager._instruments[name] = cls(name, dict(
            getattr(existing, "config", {}) or {}))


def parse_apex_spec(spec: str | None, *, default: float = 30.0) -> tuple[dict[int, float], float]:
    """Read ``--mock-apex-hz``: a bare number, or ``"18:30,19:200"`` per channel.

    Returns ``(per_channel, default_hz)``. A bare number sets the default and
    leaves the map empty, which is the common case; the per-channel form is what
    puts one cell in each population in a single run.
    """
    if spec is None or not str(spec).strip():
        return {}, float(default)
    text = str(spec).strip()
    if ":" not in text:
        return {}, float(text)
    per_channel: dict[int, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        ch_str, _, hz_str = token.partition(":")
        per_channel[int(ch_str)] = float(hz_str)
    return per_channel, float(default)


__all__ = [
    "MOCK_NOISE_REL",
    "MOCK_R0_OHM",
    "FastMockRHController",
    "FastMockTempController",
    "GridAwareMockPico",
    "MockRig",
    "grid_frequencies",
    "install_fast_conditions",
    "install_mock_picos",
    "literal_to_hz",
    "parse_apex_spec",
    "parse_mscr_grid",
    "r1_for_apex",
    "synthesize",
]
