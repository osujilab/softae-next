"""A mock EmStat Pico that reads the ``.mscr`` it was handed.

**The grid-reading generator now lives in the driver, and this module
re-exports it.** ``literal_to_hz``, ``parse_mscr_grid``, ``grid_frequencies``,
``r1_for_apex`` and ``synthesize`` are imported from
:mod:`softae.drivers.mock_espico` and re-exported here unchanged, so every
existing importer of this module's names keeps working. What moved is the
definition, not the surface.

**Why it moved (T11.56).** This module's earlier docstring said the opposite --
*"this lives in the tool, not in mock_espico.py; the whole tree shares that
double"* -- and that was an **ownership** argument: the driver was unclaimed and
shared, and a tool had no business moving anything underneath anyone. The
operator assigned ``mock_espico.py`` to this module's own session, which
discharged it. The argument that replaced it does not expire: this module
imports *from* ``drivers`` and the repository has **no** ``drivers`` -> ``tools``
edge at all, so a shared generator can only sit on the driver's side of that
line. The alternative was to keep two generators, which is what T11.56 was filed
about -- the tree carried a measurable one and an unmeasurable one, and nothing
made them agree.

**The shipped mock used to be unable to exercise this tool.**
:meth:`softae.drivers.mock_espico.MockESPico.sendscript_getdata` ignored
*mscrpath* entirely and returned a fixed 41-point 50 kHz -> 1 Hz sweep seeded
from the channel number. Under it, the validation harness's reference arm and
its adaptive arm returned **bit-identical spectra regardless of which script
ran**, so:

- every ``Delta`` was exactly 0;
- verdicts did not respond to the grid, so ``extend_low`` never fired and no
  follow-up was ever built;
- ``Delta_hold`` was identically 0, so the hold-integrity criterion H3 always
  passed;
- **a mock run printed a perfect null indistinguishable from a real result.**

That was worse than no mock. It is the failure mode
:func:`softae.tools.eis_timing._print_report` guards against in prose -- *"a
mock run measures how fast this host can write a .mscr file and nothing
whatever about the rig"* -- except that here it was silently fabricating the
science rather than merely the timings. The driver now parses the emitted script
**back**, inverting ``mscr_freq_literal`` via
:data:`palmsens.mscript.SI_PREFIX_FACTOR`, and evaluates the circuit on *those*
bounds and *those* point counts. Reading the planned grid off the planner
instead would test everything except the one unit with a **silent 1000x failure
mode**: a frequency literal whose suffix is wrong produces a spectrum in the
wrong band that the instrument executes without complaint.

:class:`GridAwareMockPico` is therefore **no longer the only grid-aware
backend**, and it is kept rather than retired because it is still the seam that
binds a sweep to a :class:`MockRig` -- the shared sample, its drift clock and
its ``_designed_r1`` register. The base class reads the grid; this subclass is
what makes two picos observe *one* sample.

**The** ``MockRig._designed_r1`` **register is a test-only seam.** Every sweep
:meth:`MockRig.measure` returns is recorded against its own spectrum bytes,
paired with the ``R1`` it was synthesised from (drift included), and
:meth:`MockRig.designed_r1` reads it back; :meth:`MockRig.clear_designed_r1`
empties it. **Nothing in ``src/`` reads it and nothing should** -- it exists so
``tests/test_eis_validate.py`` can skip a circuit fit whose answer the mock
already knows, on spectra (``R0 + CPE + R1||C0`` plus noise) that no closed form
recovers. It is a register rather than an ``eis_params`` stamp because
``measure`` returns raw columns: the ``EISResult`` is assembled downstream in
production code, so a stamp would have to be written by the rig's own path.

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

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import structlog

from softae.drivers.mock_espico import (
    MOCK_NOISE_REL,
    MOCK_R0_OHM,
    MockESPico,
    grid_frequencies,
    literal_to_hz,
    parse_mscr_grid,
    r1_for_apex,
    synthesize,
)
from softae.drivers.mock_rh_controller import MockRHController
from softae.drivers.mock_temp_controller import MockTempController

logger = structlog.get_logger(__name__)

# `MOCK_R0_OHM` and `MOCK_NOISE_REL` are imported above rather than defined
# here. They were duplicated between this module and the driver as a matter of
# comment -- "matching the shipped mock" -- with nothing enforcing it, which is
# precisely the coupling T11.56 found broken elsewhere in the same pair.

#: The name under which :class:`MockRig` records the ``R1`` it designed into a
#: spectrum, drift included. **A register key, not an** ``eis_params`` **stamp**,
#: and the difference is forced by the plumbing rather than chosen:
#: :meth:`MockRig.measure` returns raw columns, and the ``EISResult`` is built
#: downstream in production code (``eis_validate.acquire``) from
#: caller-supplied ``eis_params`` -- so a stamp there would write a mock-only key
#: into every persisted ``eis_params_json`` row on the real rig's code path. The
#: register carries the same fact and touches no production module.
#:
#: What reads it: the canned-fit fixture in ``tests/test_eis_validate.py``, which
#: needs the R1 a mock spectrum was built from because no closed form recovers it
#: (the mock synthesises ``R0 + CPE + R1||C0`` plus noise). Nothing in ``src/``
#: reads it, and nothing should.
MOCK_DESIGNED_R1_KEY = "eis_mock_designed_r1_ohms"


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

    #: Spectrum digest -> the ``R1`` designed into it. Class-level and shared
    #: across instances because the reader holds no handle on the rig:
    #: :func:`install_mock_picos` builds one per run, deep inside ``cmd_run``.
    #: Keyed on the spectrum's own bytes, so a lookup cannot attribute one
    #: channel's R1 to another's sweep even when both were taken in the same
    #: round. Bounded by the digest, not by the arrays.
    _designed_r1: ClassVar[dict[bytes, float]] = {}

    @staticmethod
    def _spectrum_key(freq: Any, z_real: Any) -> bytes:
        """A digest of the two columns that survive ``EISResult.from_raw`` intact.

        ``frequency`` and ``Z'`` are copied through unchanged (``arr[:, 0]`` and
        ``arr[:, 3]``); ``-Z''`` is negated twice on the way, so it is left out
        rather than relied upon.
        """
        digest = hashlib.blake2b(digest_size=16)
        for column in (freq, z_real):
            digest.update(
                np.ascontiguousarray(np.asarray(column, dtype=float)).tobytes())
        return digest.digest()

    @classmethod
    def clear_designed_r1(cls) -> None:
        """Empty the register. Called from the test fixture's teardown.

        The register is class-level, so without this it grows for the life of the
        interpreter and one test's spectra remain answerable in the next. Neither
        is a correctness problem -- the key is the spectrum's own bytes -- but an
        unbounded cache in a 246-test file is a leak, and a stale hit is a
        provenance claim about a rig that no longer exists.
        """
        cls._designed_r1.clear()

    @classmethod
    def designed_r1(cls, eis: Any) -> float | None:
        """The ``R1`` *eis* was synthesised from, or ``None`` if it was not.

        ``None`` is the honest answer for every spectrum this rig did not build
        -- a real stored sweep, a synthetic arc written by a test -- and callers
        must treat it as "ask the fitter", never as a value.
        """
        try:
            key = cls._spectrum_key(getattr(eis, "frequency", ()),
                                    getattr(eis, "z_real", ()))
        except Exception:
            return None
        return cls._designed_r1.get(key)

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
        designed_r1 = self.r1_now(channel)
        spectrum = synthesize(
            freq,
            r1_ohm=designed_r1,
            noise_rel=self.noise_rel,
            # Seeded from the sweep index as well as the channel, so two sweeps
            # on one cell are not the same numbers. Without that, a repeat
            # measurement would carry zero replicate scatter and the CONTROL
            # noise floor would be an artifact of the mock.
            seed=int(channel) * 10_000 + self._sweeps,
        )
        # What this sweep was BUILT from, recorded against the sweep itself. See
        # `MOCK_DESIGNED_R1_KEY` for why it is a register rather than a stamp.
        MockRig._designed_r1[
            MockRig._spectrum_key(spectrum[:, 0], spectrum[:, 3])] = float(
                designed_r1)
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
    "MOCK_DESIGNED_R1_KEY",
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
