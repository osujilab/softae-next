"""Mock humidity PID controller — runs without hardware.

Simulates the RH control loop with a Trinket PWM output.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time as _time
from typing import Any

from softae.drivers.contracts import validate_rh_setpoint
from softae.errors import InstrumentError
from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockRHController(BaseInstrument):
    """In-memory humidity control simulator with PID behaviour."""

    def __init__(self, name: str = "rh_controller", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._max_rh: float = float(self.config.get("max_rh", 95.0))
        # Same key, same default, same spelling as the real driver — because
        # :meth:`safe_dry` parks at it and a mock that read a different number
        # would grade every ``--mock`` park against a duty the rig never uses.
        self._out_min: float = float(self.config.get("out_min", 0.01))
        self._setpoint: float = 50.0
        self._rh: float = 45.0
        self._temp: float = 23.0  # simulated chamber temperature (°C)
        self._duty: float = 0.0
        self._running: bool = False
        self._last_update: float = _time.time()
        self._t0: float = _time.time()
        self._data: list[tuple[float, float, float]] = []
        self._wait_abort: threading.Event = threading.Event()

    async def connect(self) -> None:
        self._state = InstrumentState.CONNECTED
        logger.info("mock_rh_connect")

    async def disconnect(self) -> None:
        self._running = False
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        self._update_sim()
        s = self._base_status()
        s.update(
            setpoint=self._setpoint,
            rh=round(self._rh, 1),
            chamber_temp=round(self._temp, 1),
            duty_cycle=round(self._duty, 3),
            running=self._running,
        )
        return s

    # --- RH Control API -------------------------------------------------------

    def set_setpoint(self, val: float) -> None:
        """Set the target RH (%).

        Raises ``SafetyError`` if *val* exceeds the configured maximum —
        mirrors :meth:`AsyncRHController.set_setpoint`.
        """
        validate_rh_setpoint(val, self._max_rh, self.name)
        self._setpoint = val

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._duty = 0.0

    #: Parity with :attr:`AsyncRHController.last_safe_off_error`. Always ``""``
    #: here — a simulated humidifier has no transport to fail — but present, and
    #: present on the class for the same reason, so ``core.safe_park`` reads the
    #: same attribute on either driver rather than branching on which it got.
    last_safe_off_error: str = ""

    def safe_off(self) -> None:
        """Stop and zero — the mock's statement of the same safe state.

        A superset of :meth:`stop`, which leaves ``_setpoint`` where it was.
        ``create_manager`` falls back to this class for ``rh_controller`` and
        ``safe_park`` cannot tell a mock from a real driver, so every mock-backed
        park, every ``--mock`` tool run and the whole simulated campaign path
        goes through here: a ``safe_off`` that accepted the call and did nothing
        would make all of them pass while proving nothing.
        """
        self.last_safe_off_error = ""
        self._running = False
        self._duty = 0.0
        self._setpoint = 0.0

    #: Parity with the real driver's pair of :meth:`safe_dry` report attributes,
    #: present for the same reason :attr:`last_safe_off_error` is: ``safe_park``
    #: reads the same names on either driver rather than branching on which it got.
    last_safe_dry_error: str = ""
    last_safe_dry_duty: float = 0.0

    def safe_dry(self) -> None:
        """Stop, and hold ``out_min`` — the mock's statement of the dry purge.

        Implemented rather than inherited-as-a-no-op for exactly the reason
        :meth:`safe_off` is: ``create_manager`` falls back to this class for
        ``rh_controller`` and ``safe_park`` cannot tell a mock from a real
        driver, so every mock-backed park, every ``--mock`` tool run and the
        whole simulated campaign path goes through here. A ``safe_dry`` that
        accepted the call and did nothing would make all of them pass while
        proving nothing — and it would prove nothing about precisely the
        distinction this method exists to make, since a no-op mock is
        indistinguishable from ``safe_off``.

        The degenerate-``out_min`` fallback mirrors the real driver's, message
        and all. See :meth:`AsyncRHController.safe_dry` for why the fallback goes
        to ``safe_off`` and why it is reported rather than silent.
        """
        self.last_safe_dry_error = ""
        self.last_safe_dry_duty = 0.0

        if not (self._out_min > 0.0):
            logger.error("mock_rh_safe_dry_degenerate_out_min",
                         instrument=self.name, out_min=self._out_min)
            self.safe_off()
            self.last_safe_dry_error = (
                f"config [instruments.rh_controller] out_min = "
                f"{self._out_min:g} is not positive, so there is no dry-purge "
                f"duty to command: duty 0 shuts both valves. The humidifier was "
                f"zeroed instead — safe, but the chamber will collapse to room RH.")
            return

        self._running = False
        self._duty = float(self._out_min)
        self._setpoint = 0.0
        self.last_safe_dry_duty = float(self._out_min)

    def wait(
        self,
        target: float | None = None,
        tol: float = 2.0,
        timeout: float = 120.0,
        equilibration_s: float = 0.0,
        raise_on_timeout: bool = False,
    ) -> None:
        """Snap RH to target in mock (instant stabilisation).

        Respects ``_wait_abort`` so abort tests work correctly.

        ``raise_on_timeout`` mirrors :meth:`AsyncRHController.wait` in both
        signature *and* behaviour.  The signature must match because
        ``BaseInstrument.execute`` forwards task params verbatim — a catalogued
        ``rh_wait`` carrying the flag would otherwise ``TypeError`` on the
        simulated path, which is exactly where an unattended recipe is exercised
        before it touches hardware.  The behaviour must match because a mock
        that accepts the flag and then silently never honours it is worse than
        one that rejects it: the aborted case genuinely does not reach target,
        and a gated step has to hear about that in simulation too.
        """
        if target is None:
            target = self._setpoint
        if not self._wait_abort.is_set():
            # Clamped into the band: the mock is claiming to have settled, so it
            # must actually satisfy the same check the real driver applies.
            self._rh = target + max(-tol, min(tol, random.gauss(0, tol * 0.3)))
        if raise_on_timeout and abs(self._rh - target) > tol:
            raise InstrumentError(
                f"RH did not reach {target:g} ±{tol:g} %RH within {timeout:g} s "
                f"(last reading {self._rh:g} %RH)",
                instrument=self.name,
            )

    def get_H(self) -> float:
        self._update_sim()
        return round(self._rh, 1)

    def get_T(self) -> float:
        """Return the simulated chamber temperature (°C)."""
        self._update_sim()
        return round(self._temp, 1)

    def get_TH(self) -> tuple[float, float]:
        """Return ``(chamber_temp_C, %RH)`` — mirrors the real driver."""
        self._update_sim()
        return round(self._temp, 1), round(self._rh, 1)

    def get_data(self) -> list[tuple[float, float, float]]:
        """Return buffered ``(time, rh, setpoint)`` samples — mirrors the real driver."""
        return list(self._data)

    # --- Simulation -----------------------------------------------------------

    def _update_sim(self) -> None:
        now = _time.time()
        dt = now - self._last_update
        self._last_update = now
        # Chamber temperature drifts gently around ambient regardless of the
        # control loop (it is a passive sensor reading, not an actuated value).
        self._temp += random.gauss(0, 0.05)
        if self._running:
            tau = 20.0
            alpha = min(dt / tau, 1.0)
            self._rh += alpha * (self._setpoint - self._rh) + random.gauss(0, 0.2)
            self._duty = max(0.01, min(1.0, (self._setpoint - self._rh) * 0.02 + 0.3))
            self._data.append((now - self._t0, self._rh, self._setpoint))
            if len(self._data) > 500:  # mirror the real driver's deque(maxlen=500)
                del self._data[0]
