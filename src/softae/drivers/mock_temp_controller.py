"""Mock Novus N1040 temperature controller + thermocouple — runs without hardware.

Simulates setpoint/PV behaviour with a first-order thermal lag.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time as _time
from typing import Any

from softae.drivers.contracts import validate_temp_setpoint
from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockTempController(BaseInstrument):
    """In-memory temperature controller simulator.

    Simulates a first‑order thermal response: PV drifts toward the setpoint
    at a configurable rate each time :meth:`get_pv` is called.
    """

    def __init__(self, name: str = "temp_controller", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._sp: float = 25.0   # setpoint °C
        self._pv: float = 22.0   # process variable °C
        self._pv_surf: float = 21.5
        self._last_update: float = _time.time()
        self._stop_wait: threading.Event = threading.Event()

    async def connect(self) -> None:
        logger.info("mock_temp_connect", port=self.config.get("port", "SIM"))
        await asyncio.sleep(0.02)
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        self._update_sim()
        s = self._base_status()
        s.update(setpoint=self._sp, pv=self._pv, pv_surface=self._pv_surf)
        return s

    # --- tempControl API (mirrors tempControl_class.tempControl) ---------------

    def get_sp(self) -> float:
        return self._sp

    def get_pv(self, n_avg: int = 1) -> float:
        self._update_sim()
        return round(self._pv, 1)

    def get_pv_surf(self, n_avg: int = 1) -> float:
        self._update_sim()
        return round(self._pv_surf, 1)

    def write_sp(self, T_SP: float, print_flag: int = 1) -> None:
        validate_temp_setpoint(T_SP, self.config, self.name)
        old = self._sp
        self._sp = T_SP
        if print_flag:
            logger.info("mock_temp_setpoint", old=old, new=T_SP)

    def wait(self, within: float, equilibration_time: float = 0, timeout: float = 900) -> None:
        """Block until PV is within *within* °C of setpoint (instant in mock)."""
        if self._stop_wait.is_set():
            logger.info("mock_temp_wait_aborted")
            return
        # In mock, just snap PV to setpoint
        self._pv = self._sp + random.gauss(0, within * 0.3)
        self._pv_surf = self._pv - 0.5
        logger.debug("mock_temp_wait_done", pv=self._pv)

    def ramp_linear(
        self,
        T_start: float,
        T_end: float,
        t_span: float,
        up_int: float,
        print_flag: int = 1,
    ) -> None:
        """Simulate a linear temperature ramp (instant in mock).

        Mirrors :meth:`AsyncTempController.ramp_linear`:

        Parameters
        ----------
        T_start : float
            Starting temperature (°C).
        T_end : float
            Final temperature (°C).
        t_span : float
            Total ramp duration (seconds) — ignored by the simulation.
        up_int : float
            Setpoint update interval (seconds) — ignored by the simulation.

        Both endpoints are validated against the safety limits via
        :meth:`write_sp`, matching the real driver (which raises on the
        first out-of-range setpoint step).
        """
        self.write_sp(round(float(T_start), 1), print_flag=0)
        self.write_sp(round(float(T_end), 1), print_flag=0)
        self._pv = self._sp + random.gauss(0, 0.2)
        self._pv_surf = self._pv - 0.5
        if print_flag:
            logger.info(
                "mock_ramp_done",
                start=T_start, end=T_end, t_span=t_span, up_int=up_int,
            )

    def anneal(
        self,
        target_temp_C: float,
        hold_time_s: float,
        ramp_rate: float | None = None,
        tolerance: float = 1.0,
    ) -> None:
        """Ramp to *target_temp_C*, hold for *hold_time_s*, then restore the original setpoint.

        Parameters
        ----------
        target_temp_C:
            Target anneal temperature (°C).
        hold_time_s:
            Duration to hold at the target temperature (seconds).
        ramp_rate:
            Rate of temperature change (°C/s).  If *None* the setpoint is
            written directly without a controlled ramp.
        tolerance:
            Acceptable deviation from target before the hold begins (°C).
        """
        original_sp = self.get_sp()
        logger.info("anneal_start", target=target_temp_C, hold_time=hold_time_s, original_sp=original_sp)
        try:
            if ramp_rate is not None and ramp_rate > 0:
                t_span = abs(target_temp_C - original_sp) / ramp_rate
                up_int = max(t_span / 100, 1.0)
                self.ramp_linear(original_sp, target_temp_C, t_span, up_int, print_flag=0)
            else:
                self.write_sp(target_temp_C, print_flag=0)
            self.wait(within=tolerance)
            logger.info("anneal_hold_start", target=target_temp_C, duration_s=hold_time_s)
            # The mock hold is instant *by design* — a faithful one would make
            # every campaign test wait out real anneal times. This is the one
            # place the mock deliberately does not mirror the real driver, which
            # runs a watched hold (contracts.run_anneal_hold) that can abort on a
            # sustained thermal excursion. The watchdog's own behaviour is
            # covered directly against contracts.monitored_hold with injected
            # clocks, so the skip here does not leave it untested.
            logger.debug("mock_anneal_hold_skipped", duration_s=hold_time_s)
        finally:
            logger.info("anneal_return", original_sp=original_sp)
            self.write_sp(original_sp, print_flag=0)

    # --- Internal simulation --------------------------------------------------

    def _update_sim(self) -> None:
        """First-order drift of PV toward SP."""
        now = _time.time()
        dt = now - self._last_update
        self._last_update = now
        tau = 30.0  # thermal time constant (seconds)
        alpha = min(dt / tau, 1.0)
        self._pv += alpha * (self._sp - self._pv) + random.gauss(0, 0.05)
        self._pv_surf = self._pv - 0.5 + random.gauss(0, 0.1)
