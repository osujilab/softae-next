"""Real Novus N1040 temperature controller driver (Modbus RTU + NI-DAQ).

Wraps the blocking Modbus/NI-DAQ calls from the original
``tempControl_class.py`` behind the :class:`BaseInstrument` ABC so
the :class:`InstrumentManager` can manage connections, locks, and
status polling uniformly.

Hardware Requirements
---------------------
- Novus N1040 PID controller on a serial/Modbus RTU port
- (Optional) NI cDAQ with Type-K thermocouple for surface temperature
- ``minimalmodbus`` and (optionally) ``nidaqmx`` Python packages

Configuration (``softae_config.toml``)::

    [instruments.temp_controller]
    port   = "com6"
    baud   = 115200
    addr   = 1
    reg_sp = 0
    reg_pv = 1
    daq_channel = "cDAQ1Mod1/ai0"   # optional — surface thermocouple
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import structlog

from softae.drivers.contracts import validate_temp_setpoint
from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class AsyncTempController(BaseInstrument):
    """Async-wrapped Novus N1040 temperature controller.

    All blocking Modbus I/O is dispatched to the shared
    :data:`~softae.server.base_instrument._io_pool` so the event loop
    is never blocked.
    """

    def __init__(self, name: str = "temp_controller", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = self.config.get("port", "com6")
        self._baud: int = int(self.config.get("baud", 115200))
        self._addr: int = int(self.config.get("addr", 1))
        self._reg_sp: int = int(self.config.get("reg_sp", 0))
        self._reg_pv: int = int(self.config.get("reg_pv", 1))
        self._daq_channel: str | None = self.config.get("daq_channel")
        self._instrument = None  # minimalmodbus.Instrument
        self._serial_lock = threading.Lock()  # serialise access from multiple polling threads
        # Set by ArrheniusSweep.abort() to interrupt an in-progress wait() /
        # _with_retry() immediately without waiting for retries to exhaust.
        self._stop_wait: threading.Event = threading.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Modbus RTU serial connection."""
        try:
            import minimalmodbus

            inst = minimalmodbus.Instrument(self._port, self._addr, minimalmodbus.MODE_RTU)
            inst.serial.baudrate = self._baud
            inst.close_port_after_each_call = True
            self._instrument = inst
            self._state = InstrumentState.CONNECTED
            logger.info(
                "temp_controller_connected",
                port=self._port,
                baud=self._baud,
                addr=self._addr,
            )
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect to temp controller on {self._port}: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Close the serial connection."""
        if self._instrument is not None:
            try:
                self._instrument.serial.close()
            except Exception:
                pass
            self._instrument = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("temp_controller_disconnected", port=self._port)

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        if self.is_connected:
            try:
                s["setpoint"] = self.get_sp()
                s["pv"] = self.get_pv()
            except Exception as exc:
                s["setpoint"] = None
                s["pv"] = None
                s["error"] = str(exc)
        return s

    # ── Public API (mirrors original tempControl_class) ──────────────────

    def get_sp(self) -> float:
        """Read the current temperature setpoint (°C)."""
        return self._with_retry(
            self._instrument.read_register, self._reg_sp,
            timeout=10.0, max_retries=3,
        ) / 10

    def get_pv(self, n_avg: int = 1) -> float:
        """Read the process (block) temperature (°C)."""
        return self._with_retry(
            self._instrument.read_register, self._reg_pv,
            timeout=10.0, max_retries=3,
        ) / 10

    def get_pv_surf(self, n_avg: int = 1) -> float:
        """Read surface temperature (°C) from NI-DAQ thermocouple.

        Returns NaN if no DAQ channel is configured or nidaqmx is unavailable.
        """
        if not self._daq_channel:
            return float("nan")
        try:
            import nidaqmx
            import nidaqmx.constants

            def _read() -> float:
                with nidaqmx.Task() as task:
                    task.ai_channels.add_ai_thrmcpl_chan(
                        self._daq_channel,
                        min_val=0.0,
                        max_val=100.0,
                        cjc_source=nidaqmx.constants.CJCSource(10200),
                    )
                    return task.read()

            return self._with_retry(_read)
        except ImportError:
            logger.warning("nidaqmx_not_available")
            return float("nan")

    def write_sp(self, T_SP: float, print_flag: int = 1) -> None:
        """Write a new temperature setpoint (°C).

        Enforces min/max safety limits from config.
        """
        validate_temp_setpoint(T_SP, self.config, self.name)
        cur_sp = self.get_sp()
        self._with_retry(self._instrument.write_register, self._reg_sp, int(T_SP * 10))
        if print_flag:
            logger.info("temp_setpoint_changed", old=cur_sp, new=T_SP)

    def wait(self, within: float, equilibration_time: float = 0, timeout: float = 900) -> None:
        """Block until PV is within *within* °C of the setpoint.

        Parameters
        ----------
        within : float
            Acceptable deviation in °C.
        equilibration_time : float
            Extra hold time (seconds) after reaching band.
        timeout : float
            Maximum wait (seconds). ``None`` → infinite.
        """
        t0 = time.time()
        deadline = None if timeout is None else t0 + timeout

        while abs(self.get_sp() - self.get_pv()) > within:
            if self._stop_wait.is_set():
                logger.info("temp_wait_aborted", sp=self.get_sp())
                return
            if deadline is not None and time.time() >= deadline:
                logger.warning(
                    "temp_wait_timeout",
                    timeout=timeout,
                    sp=self.get_sp(),
                    pv=self.get_pv(),
                )
                return
            time.sleep(5)

        if equilibration_time > 0:
            logger.info("temp_equilibrating", duration=equilibration_time)
            time.sleep(equilibration_time)

        logger.info("temp_ready", pv=self.get_pv(), sp=self.get_sp())

    def ramp_linear(
        self,
        T_start: float,
        T_end: float,
        t_span: float,
        up_int: float,
        print_flag: int = 1,
    ) -> None:
        """Execute a blocking linear temperature ramp.

        Parameters
        ----------
        T_start : float
            Starting temperature (°C).
        T_end : float
            Final temperature (°C).
        t_span : float
            Total ramp duration (seconds).
        up_int : float
            Setpoint update interval (seconds).
        """
        n_steps = max(int(t_span / up_int), 1)
        t_vals = np.linspace(0, t_span, n_steps)
        T_vals = np.round(np.linspace(T_start, T_end, n_steps), 1)

        t0 = time.time()
        for i, t_pt in enumerate(t_vals):
            while (time.time() - t0) < t_pt:
                time.sleep(0.5)
            self.write_sp(T_vals[i], print_flag=0)
            if print_flag:
                logger.info(
                    "ramp_step",
                    pv=self.get_pv(),
                    sp=T_vals[i],
                    step=i + 1,
                    total=n_steps,
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

        Raises
        ------
        SafetyError
            If the process value stays outside the fault band — or cannot be
            read at all — for longer than the configured grace period. The hold
            is *watched*, not slept through; see
            :func:`softae.drivers.contracts.monitored_hold`.
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
            from softae.drivers.contracts import run_anneal_hold

            report = run_anneal_hold(self, hold_time_s, target_temp_C)
            # ``rh`` is always None on this path today: the call above passes no
            # ``rh_reader``, so the humidity watchdog is production-dead until that
            # wiring lands as its own task.  Logged anyway -- this line is what
            # makes the verdict observable the day it does.
            logger.info(
                "anneal_hold_done", held_s=round(report.held_s, 1),
                n_samples=report.n_samples, excursion_C=report.excursion_C,
                n_warn=report.n_warn, aborted=report.aborted,
                rh=report.rh.state if report.rh else None,
            )
        finally:
            logger.info("anneal_return", original_sp=original_sp)
            self.write_sp(original_sp, print_flag=0)

    # ── Internal ─────────────────────────────────────────────────────────

    def _with_retry(
        self,
        fn,
        *args,
        timeout: float = 60,
        backoff_base: float = 1.0,
        backoff_max: float = 15.0,
        max_retries: int | None = None,
        **kwargs,
    ):
        """Call *fn* with exponential back-off, re-initialising on NoResponseError.

        Acquires ``_serial_lock`` for the full duration so concurrent polling
        threads never share the serial handle simultaneously.
        """
        import minimalmodbus

        with self._serial_lock:
            deadline = time.time() + timeout
            wait = backoff_base
            last_exc = None
            attempt = 0

            while time.time() < deadline:
                if self._stop_wait.is_set():
                    raise CommunicationError(
                        "Temp controller communication aborted by sweep abort",
                        instrument=self.name,
                    )
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    attempt += 1
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    if max_retries is not None and attempt >= max_retries:
                        break
                    sleep_time = min(wait, remaining)
                    logger.warning(
                        "temp_comm_retry",
                        attempt=attempt,
                        error=str(exc),
                        sleep=round(sleep_time, 1),
                    )
                    time.sleep(sleep_time)
                    wait = min(wait * 2, backoff_max)

                    is_bad_handle = isinstance(exc, OSError) and (
                        getattr(exc, "winerror", None) == 6  # Windows ERROR_INVALID_HANDLE
                        or exc.errno == 9                    # EBADF cross-platform fallback
                    )
                    if isinstance(exc, minimalmodbus.NoResponseError) or is_bad_handle:
                        self._reinit_port()

            raise CommunicationError(
                f"Temp controller communication failed after {timeout} s "
                f"({attempt} retries). Last error: {last_exc}",
                instrument=self.name,
            )

    def _reinit_port(self) -> None:
        """Re-instantiate serial handle (recovery from USB re-enumeration)."""
        import minimalmodbus

        try:
            self._instrument.serial.close()
        except Exception:
            pass
        self._instrument = minimalmodbus.Instrument(self._port, self._addr, minimalmodbus.MODE_RTU)
        self._instrument.serial.baudrate = self._baud
        self._instrument.close_port_after_each_call = True
        logger.info("temp_port_reinit", port=self._port)
