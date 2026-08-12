"""Real RH control driver — Trinket M0 PWM + PID loop.

Unifies three components behind the :class:`BaseInstrument` ABC:

1. **TrinketPWM** — serial sender (writes duty-cycle to Adafruit Trinket M0)
2. **SHT31-D** — humidity sensor (via ``AsyncHTSensor`` or injected callable)
3. **PID loop** — ``simple-pid`` controller in a daemon thread

Hardware Requirements
---------------------
- Adafruit Trinket M0 on a serial port (default ``COM11``)
- Adafruit SHT31-D sensor (or any callable returning %RH)
- Packages: ``pyserial``, ``simple-pid``

Configuration (``softae_config.toml``)::

    [instruments.rh_controller]
    port        = "COM11"
    baud        = 115200
    kp          = 0.008
    ki          = 0.0015
    kd          = 0.05
    out_min     = 0.01
    out_max     = 1.0
    poll_period = 2.0       # PID loop period (seconds)
    max_rh      = 95.0      # safety cap (%)
    max_consecutive_failures = 5   # sensor resets after this many consecutive errors
    max_stale_s = 30.0             # seconds before a stale last-good reading becomes NaN
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from typing import Any, Callable

import structlog

from softae.drivers.contracts import validate_rh_setpoint
from softae.drivers.mcp2221_bus import I2C_BUS_LOCK
from softae.errors import (
    CommunicationError,
    ConnectionError_,
)
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class AsyncRHController(BaseInstrument):
    """Async-wrapped RH controller with PID loop.

    The PID loop runs in a daemon thread (like the original
    ``HumidityControlMethod``), but the public API is designed
    for ``execute()`` dispatch from the async event loop.

    Parameters
    ----------
    name : str
        Instrument registry name (default ``"rh_controller"``).
    config : dict
        TOML section for this instrument.
    rh_reader : callable, optional
        Zero-arg function returning current %RH.  If *None*, the class
        will attempt to import and use ``AsyncHTSensor``.  For tests
        you can inject a lambda.
    th_reader : callable, optional
        Zero-arg function returning ``(temperature_C, %RH)`` from a single
        sensor transaction.  When available it is used in preference to
        *rh_reader* so the chamber temperature is captured from the very
        same read that drives the PID loop — no extra I²C traffic.  If
        *None* (and no internal ``AsyncHTSensor`` is created), the chamber
        temperature is reported as ``NaN``.
    """

    def __init__(
        self,
        name: str = "rh_controller",
        config: dict[str, Any] | None = None,
        rh_reader: Callable[[], float] | None = None,
        th_reader: Callable[[], tuple[float, float]] | None = None,
    ):
        super().__init__(name, config)

        # Serial config
        self._port: str = self.config.get("port", "COM11")
        self._baud: int = int(self.config.get("baud", 115200))

        # PID config
        self._kp: float = float(self.config.get("kp", 0.008))
        self._ki: float = float(self.config.get("ki", 0.0015))
        self._kd: float = float(self.config.get("kd", 0.05))
        self._out_min: float = float(self.config.get("out_min", 0.01))
        self._out_max: float = float(self.config.get("out_max", 1.0))
        self._poll_period: float = float(self.config.get("poll_period", 2.0))
        self._max_rh: float = float(self.config.get("max_rh", 95.0))

        # Runtime state
        self._serial = None  # serial.Serial once connected
        self._pid = None  # simple_pid.PID instance
        self._rh_reader = rh_reader
        self._th_reader = th_reader  # returns (temp_C, %RH) in one read
        self._ht_sensor = None  # AsyncHTSensor if we create one locally

        self._max_consecutive_failures: int = int(self.config.get("max_consecutive_failures", 5))
        self._max_stale_s: float = float(self.config.get("max_stale_s", 30.0))

        self._setpoint: float = 0.0
        self._current_rh: float = float("nan")
        self._current_temp: float = float("nan")  # chamber T (RH sensor onboard)
        self._running: bool = False
        self._stop_event = threading.Event()
        self._wait_abort = threading.Event()   # set by ArrheniusSweep.abort() to unblock wait()
        self._thread_lock = threading.Lock()  # guards _data / _current_rh / _setpoint
        self._data: deque[tuple[float, float, float]] = deque(maxlen=500)
        self._thread: threading.Thread | None = None

        # Fault tracking for the PID loop
        self._consecutive_failures: int = 0
        self._last_good_rh: float = float("nan")
        self._last_good_ts: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open serial port to Trinket and prepare PID controller."""
        try:
            import serial
            from simple_pid import PID

            ser = serial.Serial(self._port, self._baud, timeout=1)
            time.sleep(0.5)  # let Trinket settle after port open

            pid = PID(
                Kp=self._kp,
                Ki=self._ki,
                Kd=self._kd,
                setpoint=self._setpoint,
                output_limits=(self._out_min, self._out_max),
            )

            self._serial = ser
            self._pid = pid

            # If no reader was injected, create one from AsyncHTSensor.
            # An injected rh_reader or th_reader is left untouched (tests /
            # custom wiring): the combined th_reader is preferred everywhere,
            # and get_H falls back to it when no rh_reader is present.
            if self._rh_reader is None and self._th_reader is None:
                try:
                    from softae.drivers.async_ht_sensor import AsyncHTSensor
                    ht = AsyncHTSensor(name="ht_sensor_rh_internal", config=self.config)
                    await ht.connect()
                    self._ht_sensor = ht
                    self._rh_reader = ht.get_H
                    # Prefer the combined reader so the PID loop captures the
                    # chamber temperature from the same I²C transaction as %RH.
                    self._th_reader = ht.get_TH
                    logger.info("rh_controller_ht_sensor_connected")
                except Exception as exc:
                    logger.warning(
                        "rh_controller_ht_sensor_unavailable",
                        error=str(exc),
                    )
                    # Provide a NaN reader so PID loop won't crash
                    self._rh_reader = lambda: float("nan")

            self._state = InstrumentState.CONNECTED
            logger.info("rh_controller_connected", port=self._port)

        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect RH controller: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Stop PID loop and close serial port."""
        self._stop_pid_loop()

        if self._ht_sensor is not None:
            try:
                await self._ht_sensor.disconnect()
            except Exception:
                pass
            self._ht_sensor = None

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        self._pid = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("rh_controller_disconnected")

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        with self._thread_lock:
            s["setpoint"] = self._setpoint
            s["current_rh"] = self._current_rh
            s["chamber_temp"] = self._current_temp
            s["running"] = self._running
            s["data_points"] = len(self._data)
        return s

    # ── Serial helpers ───────────────────────────────────────────────────

    def _send_duty(self, duty: float) -> None:
        """Send duty-cycle value to Trinket via serial.

        Opens/closes port per send, matching original pattern to avoid
        the Trinket freezing when the port stays open.
        """
        if self._serial is None:
            raise CommunicationError("Serial not connected", instrument=self.name)

        try:
            # Close + reopen for per-send pattern (matches original)
            if not self._serial.is_open:
                self._serial.open()

            self._serial.write(f"{duty:.4f}\n".encode())
            self._serial.flush()
            logger.debug("rh_duty_sent", duty=duty)

        except Exception as exc:
            raise CommunicationError(
                f"Failed to send duty cycle: {exc}",
                instrument=self.name,
            ) from exc

    # ── PID loop (daemon thread) ─────────────────────────────────────────

    def _reset_ht_sensor(self) -> None:
        """Attempt a soft-reset on the internal HT sensor (best-effort)."""
        if self._ht_sensor is None:
            return
        try:
            if self._ht_sensor._sensor is not None:
                with I2C_BUS_LOCK:
                    self._ht_sensor._sensor._reset()
                time.sleep(0.015)
                logger.info("rh_ht_sensor_soft_reset")
        except Exception as exc:
            logger.warning("rh_ht_sensor_reset_failed", error=str(exc))

    def _read_th(self) -> tuple[float, float]:
        """Read ``(temperature_C, %RH)`` from the sensor.

        Uses the combined ``_th_reader`` when available (one I²C transaction
        for both values); otherwise falls back to the %RH-only ``_rh_reader``
        and reports temperature as ``NaN``.  May raise — callers handle the
        exception the same way a bare RH read would.
        """
        if self._th_reader is not None:
            return self._th_reader()
        return float("nan"), self._rh_reader()

    def _pid_loop(self) -> None:
        """Run PID controller in a daemon thread."""
        logger.info("rh_pid_loop_started")
        t0 = time.time()

        while not self._stop_event.is_set():
            try:
                t, rh = self._read_th()
                # Successful read — reset fault counters
                self._consecutive_failures = 0
                self._last_good_rh = rh
                self._last_good_ts = time.time()
                with self._thread_lock:
                    self._current_rh = rh
                    self._current_temp = t

                if math.isnan(rh):
                    self._stop_event.wait(timeout=self._poll_period)
                    continue

                self._pid.setpoint = self._setpoint
                output = self._pid(rh)

                self._send_duty(output)

                elapsed = time.time() - t0
                with self._thread_lock:
                    self._data.append((elapsed, rh, self._setpoint))

            except Exception as exc:
                self._consecutive_failures += 1
                logger.warning(
                    "rh_pid_loop_error",
                    error=str(exc),
                    consecutive_failures=self._consecutive_failures,
                )

                # Issue sensor soft-reset when repeated failures occur
                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.warning(
                        "rh_sensor_threshold_reached",
                        consecutive_failures=self._consecutive_failures,
                    )
                    self._reset_ht_sensor()
                    self._consecutive_failures = 0

                # Use last-good reading if it is fresh enough so the PID
                # continues to actuate rather than stalling on NaN.
                stale = (time.time() - self._last_good_ts) > self._max_stale_s
                held_rh = self._last_good_rh if (not stale and not math.isnan(self._last_good_rh)) else float("nan")
                with self._thread_lock:
                    self._current_rh = held_rh

                if not math.isnan(held_rh):
                    try:
                        self._pid.setpoint = self._setpoint
                        output = self._pid(held_rh)
                        self._send_duty(output)
                        elapsed = time.time() - t0
                        with self._thread_lock:
                            self._data.append((elapsed, held_rh, self._setpoint))
                    except Exception:
                        pass

            self._stop_event.wait(timeout=self._poll_period)

        # Turn off humidifier when loop stops
        try:
            self._send_duty(0.0)
        except Exception:
            pass

        logger.info("rh_pid_loop_stopped")

    def _start_pid_loop(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._pid_loop, daemon=True)
        self._thread.start()
        self._running = True

    def _stop_pid_loop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._running = False

    # ── Public API (mirrors MockRHController) ────────────────────────────

    def set_setpoint(self, val: float) -> None:
        """Set the target RH (%).

        Raises ``SafetyError`` if *val* exceeds the configured maximum.
        """
        validate_rh_setpoint(val, self._max_rh, self.name)
        with self._thread_lock:
            self._setpoint = val
        if self._pid is not None:
            self._pid.setpoint = val
        logger.info("rh_setpoint_changed", val=val)

    def start(self) -> None:
        """Start the PID control loop."""
        self._start_pid_loop()
        logger.info("rh_control_started", setpoint=self._setpoint)

    def stop(self) -> None:
        """Stop the PID control loop and turn off the humidifier."""
        self._stop_pid_loop()
        logger.info("rh_control_stopped")

    def wait(
        self,
        target: float | None = None,
        tol: float = 2.0,
        timeout: float = 120.0,
        equilibration_s: float = 0.0,
    ) -> None:
        """Block until RH reaches *target* ± *tol* or *timeout* expires.

        If *target* is ``None``, uses the current setpoint.  When
        *equilibration_s* > 0 the method holds for that many additional seconds
        after the setpoint is reached, matching the temperature controller
        ``wait(equilibration_time=…)`` pattern.
        """
        if target is None:
            target = self._setpoint
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._thread_lock:
                rh = self._current_rh
            if not math.isnan(rh) and abs(rh - target) <= tol:
                logger.info("rh_target_reached", target=target, rh=rh)
                if equilibration_s > 0:
                    time.sleep(equilibration_s)
                return
            time.sleep(self._poll_period)

        logger.warning("rh_wait_timeout", target=target, timeout=timeout)

    def get_TH(self) -> tuple[float, float]:
        """Return ``(chamber_temp_C, %RH)`` — both from one sensor read.

        When the PID control loop is **not** running, reads directly from the
        sensor so the values are always fresh (e.g. for monitoring without
        active control); a single I²C transaction yields both.  When the loop
        IS running, the cached values from the most recent PID tick are used
        to avoid redundant sensor reads.  Temperature is ``NaN`` when no
        combined reader is available (e.g. an injected %RH-only reader).
        """
        if not self._running:
            if self._th_reader is not None:
                try:
                    t, h = self._th_reader()
                    with self._thread_lock:
                        self._current_temp = t
                        self._current_rh = h
                    return t, h
                except Exception as exc:
                    logger.debug("rh_direct_th_read_failed", error=str(exc))
            elif self._rh_reader is not None:
                try:
                    h = self._rh_reader()
                    with self._thread_lock:
                        self._current_rh = h
                except Exception as exc:
                    logger.debug("rh_direct_read_failed", error=str(exc))
        with self._thread_lock:
            return self._current_temp, self._current_rh

    def get_H(self) -> float:
        """Return the current humidity reading (%).

        When the PID control loop is **not** running, reads directly from the
        sensor so that the value is always fresh (e.g. for monitoring without
        active control).  When the loop IS running the cached value from the
        most recent PID tick is used to avoid redundant sensor reads.
        """
        return self.get_TH()[1]

    def get_T(self) -> float:
        """Return the chamber temperature (°C) — the RH sensor's onboard T.

        Follows the same live-read-when-idle / cached-when-running policy as
        :meth:`get_H`, sharing a single sensor transaction with it.
        """
        return self.get_TH()[0]

    def get_data(self) -> list[tuple[float, float, float]]:
        """Return the buffered data as a list of (time, rh, setpoint)."""
        with self._thread_lock:
            return list(self._data)
