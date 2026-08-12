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
from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockRHController(BaseInstrument):
    """In-memory humidity control simulator with PID behaviour."""

    def __init__(self, name: str = "rh_controller", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._max_rh: float = float(self.config.get("max_rh", 95.0))
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

    def wait(
        self,
        target: float | None = None,
        tol: float = 2.0,
        timeout: float = 120.0,
        equilibration_s: float = 0.0,
    ) -> None:
        """Snap RH to target in mock (instant stabilisation).\n\n        Respects ``_wait_abort`` so abort tests work correctly.\n        """
        if target is None:
            target = self._setpoint
        if not self._wait_abort.is_set():
            self._rh = target + random.gauss(0, tol * 0.3)

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
