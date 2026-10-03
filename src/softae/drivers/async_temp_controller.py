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
from typing import Any, Callable

import structlog

from softae.drivers.contracts import validate_temp_setpoint
from softae.drivers.temp_ramp import (
    RampReport,
    RampStep,
    ramp_span_s,
    refuse_unknown_anneal_kwargs,
    run_ramp,
)
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
        # Held across write_sp's read+write pair, and by a ramp across its
        # cancel re-check and its write, so a park's setpoint always lands after
        # any ramp step it races. Re-entrant because a ramp step takes it and
        # then calls write_sp. Distinct from _serial_lock, which is per
        # transaction and must stay a plain Lock.
        self._sp_write_lock = threading.RLock()
        # Set by ArrheniusSweep.abort() to interrupt an in-progress wait() /
        # _with_retry() immediately without waiting for retries to exhaust.
        self._stop_wait: threading.Event = threading.Event()
        #: The instrument registry, set at registration by
        #: :func:`softae.drivers.factory.create_manager` — the same
        #: back-reference ``AsyncLiquidHandler`` takes there, and for the same
        #: reason: a driver that must reach a *sibling* instrument gets the
        #: registry rather than opening a second connection of its own. An
        #: anneal hold watches humidity as well as temperature, and the %RH it
        #: watches belongs to ``rh_controller``. ``None`` — a controller
        #: constructed directly, as every test does — means the hold is exactly
        #: the thermal-only hold it has always been.
        #:
        #: ``create_manager(mock=True)`` short-circuits to ``create_mock_manager``
        #: and does not set this, which costs nothing: ``MockTempController``'s
        #: hold is instant by design and never calls ``run_anneal_hold`` at all,
        #: so there is no watch for a mock rig to open.
        self.manager: Any = None
        #: Optional destination for the hold's humidity alert row and the run it
        #: belongs to. The verdict is announced through
        #: :func:`softae.core.alerts.raise_alert` either way — it logs at
        #: WARNING/CRITICAL with no store — but it is only *persisted* with one.
        #: Set by whichever host owns the run, as ``autonomous_wiring`` already
        #: sets ``syringe.purge_runner``.
        self.data_store: Any = None
        self.run_id: str | None = None
        #: ``(event_type, payload)`` — the host's durable event stream, set by
        #: ``autonomous_wiring`` for the campaign's lifetime (D8, rung 3c: the
        #: cure and cool-down left no record a run directory could answer "did
        #: it reach temperature?" from). Called from the I/O worker thread.
        #: ``None`` leaves every method exactly as it was.
        self.on_event: Callable[[str, dict[str, Any]], Any] | None = None

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

        Enforces min/max safety limits from config. The SP read and the SP
        write happen under ``_sp_write_lock`` together, so no other setpoint
        writer — a park in particular — can land between them.
        """
        validate_temp_setpoint(T_SP, self.config, self.name)
        with self._sp_write_lock:
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

        A timeout still **returns** (no raise; that is a separate operator
        ruling) — but it is now reported through :attr:`on_event` as
        ``temp_wait_timeout``, beside ``temp_wait_reached`` and
        ``temp_wait_aborted``, so the outcome outlives the process log.
        """
        t0 = time.time()
        deadline = None if timeout is None else t0 + timeout

        def _outcome(event: str, sp: float, pv: float | None) -> None:
            self._note(event, sp_C=sp, pv_C=pv, within_C=within,
                       timeout_s=timeout, waited_s=round(time.time() - t0, 1))

        while abs(self.get_sp() - self.get_pv()) > within:
            if self._stop_wait.is_set():
                sp = self.get_sp()
                logger.info("temp_wait_aborted", sp=sp)
                _outcome("temp_wait_aborted", sp, None)  # PV not read: aborting
                return
            if deadline is not None and time.time() >= deadline:
                sp, pv = self.get_sp(), self.get_pv()
                logger.warning("temp_wait_timeout", timeout=timeout, sp=sp, pv=pv)
                _outcome("temp_wait_timeout", sp, pv)
                return
            time.sleep(5)

        if equilibration_time > 0:
            logger.info("temp_equilibrating", duration=equilibration_time)
            time.sleep(equilibration_time)

        sp, pv = self.get_sp(), self.get_pv()
        logger.info("temp_ready", pv=pv, sp=sp)
        _outcome("temp_wait_reached", sp, pv)

    def _note(self, event_type: str, **payload: Any) -> None:
        """Report one event to :attr:`on_event`, if set. Never raises.

        A reporting failure must not become a thermal one: this sits inside the
        anneal and the approach wait, and the hold must go on regardless.
        """
        hook = self.on_event
        if hook is None:
            return
        try:
            hook(event_type, payload)
        except Exception:
            logger.warning("temp_event_hook_failed", event_type=event_type,
                           exc_info=True)

    def ramp_linear(
        self,
        T_start: float,
        T_end: float,
        t_span: float,
        up_int: float | None = None,
        print_flag: int = 1,
        *,
        on_step: Callable[[RampStep], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> RampReport:
        """Execute a blocking linear temperature ramp; see :func:`~softae.drivers.temp_ramp.run_ramp`.

        Parameters
        ----------
        T_start, T_end : float
            Start and final setpoints (°C).
        t_span : float
            Total ramp duration (seconds).
        up_int : float or None
            Setpoint update interval (seconds). ``None`` derives it from the
            rate (:func:`~softae.drivers.temp_ramp.update_interval_s`); an
            explicit value is honoured but capped at 60 s.
        print_flag : int
            Non-zero reads PV and logs an INFO ``ramp_step`` line per write.
        on_step : callable, optional
            Called with a :class:`RampStep` after every write, on this thread.
        cancel : threading.Event, optional
            A dedicated abort — deliberately **not** ``_stop_wait``, which makes
            every later serial call raise. Once set, nothing further is written
            and the last-written setpoint holds. It is re-checked under
            ``_sp_write_lock``, so a writer that sets it and then calls
            :meth:`write_sp` is never overwritten by a step.
        """
        return run_ramp(
            lambda sp: self.write_sp(sp, print_flag=0),
            T_start, T_end, t_span, up_int,
            on_step=on_step, cancel=cancel,
            read_pv=self.get_pv if print_flag else None,
            write_lock=self._sp_write_lock,
        )

    def anneal(
        self,
        target_temp_C: float,
        hold_time_s: float,
        ramp_rate_C_per_min: float | None = None,
        tolerance: float = 1.0,
        **legacy: Any,
    ) -> None:
        """Ramp to *target_temp_C*, hold for *hold_time_s*, then restore the original setpoint.

        Parameters
        ----------
        target_temp_C:
            Target anneal temperature (°C).
        hold_time_s:
            Duration to hold at the target temperature (seconds).
        ramp_rate_C_per_min:
            Rate of temperature change (**°C/min**).  If *None* the setpoint is
            written directly without a controlled ramp.  The retired
            ``ramp_rate`` key (°C/s) is refused before anything is read or
            written; see :func:`~softae.drivers.temp_ramp.legacy_ramp_rate_problem`.
        tolerance:
            Acceptable deviation from target before the hold begins (°C).

        Raises
        ------
        ValidationError_
            If the retired ``ramp_rate`` key is passed.
        SafetyError
            If the process value stays outside the fault band — or cannot be
            read at all — for longer than the configured grace period. The hold
            is *watched*, not slept through; see
            :func:`softae.drivers.contracts.monitored_hold`.
        """
        refuse_unknown_anneal_kwargs(legacy)
        original_sp = self.get_sp()
        logger.info("anneal_start", target=target_temp_C, hold_time=hold_time_s, original_sp=original_sp)
        self._note("anneal_started", target_C=target_temp_C, hold_s=hold_time_s,
                   ramp_C_per_min=ramp_rate_C_per_min, original_sp_C=original_sp)
        try:
            if ramp_rate_C_per_min is not None and ramp_rate_C_per_min > 0:
                t_span = ramp_span_s(original_sp, target_temp_C, ramp_rate_C_per_min)
                self.ramp_linear(original_sp, target_temp_C, t_span, None, print_flag=0)
            else:
                self.write_sp(target_temp_C, print_flag=0)
            self.wait(within=tolerance)
            logger.info("anneal_hold_start", target=target_temp_C, duration_s=hold_time_s)
            self._note_hold_start(target_temp_C, hold_time_s, tolerance)
            from softae.drivers.contracts import run_anneal_hold

            rh_reader, rh_setpoint_pct = self._rh_watch_source()
            try:
                report = run_anneal_hold(
                    self, hold_time_s, target_temp_C,
                    rh_reader=rh_reader, rh_setpoint_pct=rh_setpoint_pct,
                    data_store=self.data_store, run_id=self.run_id,
                )
            except Exception as exc:
                self._note("anneal_hold_failed", target_C=target_temp_C,
                           error=f"{type(exc).__name__}: {exc}"[:300])
                raise
            # ``rh`` is None when no reader could be resolved -- a controller with
            # no registry attached, or a rig with no RH controller on it -- and in
            # that case this is exactly the thermal-only hold it has always been.
            logger.info(
                "anneal_hold_done", held_s=round(report.held_s, 1),
                n_samples=report.n_samples, excursion_C=report.excursion_C,
                n_warn=report.n_warn, aborted=report.aborted,
                rh=report.rh.state if report.rh else None,
            )
            self._note("anneal_hold_ended", target_C=target_temp_C,
                       held_s=round(report.held_s, 1), n_samples=report.n_samples,
                       excursion_C=report.excursion_C, n_warn=report.n_warn,
                       aborted=report.aborted,
                       rh=report.rh.state if report.rh else None)
        finally:
            logger.info("anneal_return", original_sp=original_sp)
            self.write_sp(original_sp, print_flag=0)
            self._note("anneal_restored", restore_sp_C=original_sp)

    def _note_hold_start(self, target_C: float, hold_s: float, tolerance: float) -> None:
        """Report the hold starting, with whether PV had actually reached the band.

        ``wait`` returns on timeout, so a hold can start out of band; this is the
        record that says so. Only reached when someone is listening, so a bare
        controller makes no extra serial reads.
        """
        if self.on_event is None:
            return
        try:
            pv = self.get_pv()
        except Exception:
            pv = None
        self._note("anneal_hold_started", target_C=target_C, hold_s=hold_s,
                   pv_C=pv, tolerance_C=tolerance,
                   in_band=None if pv is None else abs(pv - target_C) <= tolerance)

    def _rh_watch_source(self) -> tuple[Any, float | None]:
        """``(reader, commanded %RH)`` for the hold's humidity watch, or two ``None``.

        Both halves are required by :func:`~softae.drivers.contracts.run_anneal_hold`
        and neither can be invented: a reader with no setpoint has nothing to
        classify against, and a setpoint with no reader has nothing to classify.
        Either missing yields ``(None, None)`` and the hold is thermal-only —
        which is what a bare controller, or a rig with no humidity control, gets.

        **The setpoint is read through**
        :func:`softae.core.conditions_capture.read_environment`, not off the RH
        driver directly, and that is deliberate: ``rh_sp_pct`` has no public
        getter (it lives inside ``status()["setpoint"]``), and that function is
        already the single place that knows so. Reading it anywhere else would
        put a second spelling of the setpoint beside the one every ``conditions``
        row is written from, and the whole point of watching humidity through a
        cure is that the verdict and the record agree about what was asked for.

        Never raises and never actuates: every read is best-effort, and a hold
        that cannot resolve a humidity watch proceeds as a thermal hold rather
        than failing. Losing the *secondary* variable's watchdog must not cost a
        board.
        """
        manager = self.manager
        if manager is None:
            return None, None
        try:
            from softae.core.conditions_capture import RH_CONTROLLER, read_environment

            if RH_CONTROLLER not in manager.names:
                return None, None
            rh = manager.get(RH_CONTROLLER)
            # `get_TH` returns (chamber_air_C, %RH) in one transaction, and the
            # chamber air is the right thermometer for a claim about enclosure
            # humidity — `contracts._read_rh` says so and unpacks that shape.
            reader = getattr(rh, "get_TH", None) or getattr(rh, "get_H", None)
            setpoint = read_environment(manager).get("rh_sp_pct")
        except Exception:
            logger.warning("anneal_rh_watch_unavailable", exc_info=True)
            return None, None
        if reader is None or setpoint is None:
            logger.info("anneal_rh_watch_absent",
                        has_reader=reader is not None,
                        has_setpoint=setpoint is not None)
            return None, None
        logger.info("anneal_rh_watch_attached", rh_setpoint_pct=setpoint)
        return reader, float(setpoint)

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
