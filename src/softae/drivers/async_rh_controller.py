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
    out_min     = 0.01      # also the duty :meth:`AsyncRHController.safe_dry`
                            # parks at — see that method before lowering it
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
    InstrumentError,
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

        # Five keys are dual-spelled, long spelling PREFERRED, short name as
        # fallback, code default last.  `softae_config.toml` documents
        # `trinket_port` / `trinket_baud` / `pid_kp` / `pid_ki` / `pid_kd` under
        # `[instruments.rh_controller]`, but this driver only ever read the short
        # names and `factory.py` passes the section raw — so none of those five
        # keys reached anything, and the rig ran the code-default kp = 0.008
        # instead of the operator's tuned pid_kp = 0.007.  Reading them changes
        # the physical loop response, which is the point.  See SESSION_MAIL
        # [a69] and `rh_safe_state_and_hold_spec.md` §9.2.

        # Serial config
        self._port: str = self.config.get("trinket_port", self.config.get("port", "COM11"))
        self._baud: int = int(self.config.get("trinket_baud", self.config.get("baud", 115200)))

        # PID config
        self._kp: float = float(self.config.get("pid_kp", self.config.get("kp", 0.008)))
        self._ki: float = float(self.config.get("pid_ki", self.config.get("ki", 0.0015)))
        self._kd: float = float(self.config.get("pid_kd", self.config.get("kd", 0.05)))
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
        #: Duty the PID thread writes on its way out. ``0.0`` is the historical
        #: (and still default) value; :meth:`safe_dry` raises it so the exiting
        #: thread cannot slam a zero in ahead of the dry-purge duty. Set only via
        #: :meth:`_stop_pid_loop`, and only *before* ``_stop_event`` is set — see
        #: that method for why that ordering is the whole of the thread-safety
        #: argument.
        self._exit_duty: float = 0.0
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

        # The exit write. Historically an unconditional zero; now whatever the
        # stopper asked for, because "stop the loop" and "leave the device at
        # duty 0" are two decisions and only the first belongs to every caller.
        # A `safe_dry` that let this write 0.0 first would shut both PSVs for one
        # frame before reopening them to dry air — a valve slam the firmware
        # would see, since `ctrl == 0` is its own special case there.
        exit_duty = self._exit_duty
        try:
            self._send_duty(exit_duty)
        except Exception:
            pass

        logger.info("rh_pid_loop_stopped", exit_duty=exit_duty)

    def _start_pid_loop(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._pid_loop, daemon=True)
        self._thread.start()
        self._running = True

    def _stop_pid_loop(self, exit_duty: float = 0.0) -> None:
        """Stop the PID thread, telling it what to leave on the wire.

        *exit_duty* defaults to ``0.0``, so every existing caller — ``stop``,
        ``disconnect``, ``safe_off`` — is unchanged.

        **Why writing the attribute here is enough.** ``_exit_duty`` is stored
        *before* ``_stop_event.set()``, and the loop reads it only after it has
        observed the event. ``Event.set()`` releases the event's internal lock
        and the loop's ``wait()``/``is_set()`` acquires it, so the store
        happens-before the load by the same release/acquire pairing that makes
        ``_stop_event`` itself work. No extra lock can strengthen that, and a
        lock the loop would have to acquire on its exit path could deadlock
        against a wedged tick.

        Set on **every** stop rather than only on the dry path: a value left over
        from a previous ``safe_dry`` must not become the exit duty of the next
        plain ``stop()``.
        """
        if not self._running:
            return
        self._exit_duty = float(exit_duty)
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

    #: Why the last :meth:`safe_off` could **not** write duty 0 — ``""`` when it
    #: did. Declared on the class rather than assigned in ``__init__`` so this
    #: entry point stays strictly additive: every instance reads the class
    #: default until its first ``safe_off``, which is exactly right — no call,
    #: no failure.
    #:
    #: It exists because :meth:`safe_off` must not raise (the park's contract is
    #: never-raise) while a park that silently swallowed a dead port would report
    #: *"humidifier off"* about a write that never landed —  the exact class of
    #: falsehood ``SafeParkResult`` exists to remove. ``core.safe_park`` reads
    #: this after the call and files the step under ``errors`` instead.
    last_safe_off_error: str = ""

    def safe_off(self) -> None:
        """Stop the loop **and** write duty 0. The humidifier's safe state.

        Not an alias for :meth:`stop`, and the park must never substitute one for
        the other. :meth:`_stop_pid_loop` returns immediately when ``_running`` is
        ``False``, so a process that connected, set a setpoint and never called
        :meth:`start` writes **nothing** — leaving the Trinket at whatever duty a
        previous session left it at. And when the loop *is* running, the
        ``join(timeout=5.0)`` clears ``_thread``/``_running`` whether or not the
        thread exited, so a loop wedged in an I²C read or a port reopen lets
        ``stop()`` return cleanly having sent nothing. This method closes both by
        writing the zero itself.

        A duplicate zero is accepted deliberately: in the ordinary case the
        exiting thread also writes ``0.0``, the value is idempotent, and a second
        ``"0.0000\\n"`` costs one frame on a 115200-baud link. Suppressing it would
        mean reading an outcome ``_stop_pid_loop`` does not report.

        The stored setpoint is zeroed too, so a later bare :meth:`start` cannot
        resume the pre-park target — ``_pid_loop`` re-reads ``_setpoint`` every
        tick, so leaving it at 45 % would reactuate immediately.

        Never raises, and never disconnects: the port stays open and the state
        stays ``CONNECTED``, because a park runs while sessions are open and the
        caller owns teardown. A failed write is reported through
        :attr:`last_safe_off_error`.
        """
        self.last_safe_off_error = ""

        # Stop first, then write: the reverse races a still-running loop, which
        # writes a fresh PID output every poll period and would overwrite the zero.
        self._stop_pid_loop()

        with self._thread_lock:
            self._setpoint = 0.0
        if self._pid is not None:
            self._pid.setpoint = 0.0

        if self._serial is None:
            # Never connected, or already disconnected. Nothing to write to —
            # said out loud rather than reported as a successful park.
            self.last_safe_off_error = (
                "no serial transport — the humidifier could not be zeroed from "
                "this process")
            logger.warning("rh_safe_off_no_transport", instrument=self.name)
            return

        try:
            self._send_duty(0.0)
        except Exception as exc:
            self.last_safe_off_error = str(exc)
            logger.warning("rh_safe_off_send_failed", instrument=self.name,
                           error=str(exc))
            return

        logger.info("rh_safe_off", instrument=self.name)

    #: Why the last :meth:`safe_dry` could **not** command ``out_min`` — ``""``
    #: when it did. Read by ``core.safe_park`` exactly as
    #: :attr:`last_safe_off_error` is, and on the class for the same reason.
    last_safe_dry_error: str = ""

    #: The duty :meth:`safe_dry` actually put on the wire, published so a report
    #: can *name* it. A park that says "dry purge" without the number is asking
    #: the operator to trust a config file they cannot see from the dialog.
    last_safe_dry_duty: float = 0.0

    def safe_dry(self) -> None:
        """Stop the loop and leave the chamber **purging dry air** at ``out_min``.

        The sibling of :meth:`safe_off`, and named to sit beside it: both are
        ``safe_<end state>``, both stop the loop, both zero the stored setpoint,
        both never raise. The one word that differs names the one thing that
        differs — whether gas keeps flowing. ``dry_purge()`` would have dropped
        the ``safe_`` prefix that marks a park entry point; ``safe_dry_purge()``
        adds length without adding information.

        **Why this exists.** ``ctrl`` near 0 is *dry* air and ``ctrl = 1`` is
        fully humid — bench-verified 2026-08-21. (``scripts/trinket_firmware/
        README.md`` asserts the inverse; that sentence is wrong and is being
        corrected separately. The firmware itself is unambiguous: ``V0_range``,
        the humidity signal, rises with ``ctrl`` while ``V1_range``, the dry-air
        signal, falls with it.) ``ctrl == 0`` *exactly* is a firmware special
        case that shuts **both** Aalborg PSVs, so :meth:`safe_off` leaves no flow
        at all and room air wins: after a long low-RH hold the chamber goes
        10 %RH to ~50 %RH in tens of seconds, and every GUI restart or
        GUI-to-headless switchover costs a full re-equilibration and re-drying.

        Commanding ``out_min`` instead keeps dry air flowing while the host is
        away, and the Trinket's own deadman — ``ctrl_timeout = 20`` iterations of
        a ~1.25 s loop — forces ``ctrl = 0`` about 25 s after this last command,
        shutting the valves without the host having to still be alive. It is
        self-recovering: any new float resumes control instantly. So the plateau
        the operator wants is already firmware behaviour, and all this method
        does is stop pre-empting it.

        **``self._out_min``, not a literal 0.01 and not the last PID output.**
        The literal would silently disagree with a retuned config. The last PID
        output would strand a *humidifying* duty on the wire whenever the exit
        happens mid-approach to a wet setpoint — the exact failure the park
        exists to prevent, arrived at by a different road.

        The duplicate write is deliberate, for the reason :meth:`safe_off` gives:
        ``_stop_pid_loop`` reports nothing about whether the thread reached its
        own exit write, and a loop wedged past its ``join(timeout=5.0)`` never
        will. Unlike ``safe_off`` the duplicate is not merely tolerated here — it
        is what a never-started or wedged loop depends on entirely.
        """
        self.last_safe_dry_error = ""
        self.last_safe_dry_duty = 0.0

        # A degenerate `out_min` makes "dry purge" a lie: 0 is the firmware's
        # valve-shutoff sentinel and a negative duty is not a duty at all. Three
        # responses were possible and only one is defensible. Refusing outright
        # would leave the humidifier under whatever duty it already had — worse
        # than today on a park path. Clamping to an invented positive floor would
        # push gas at a rate nobody configured. So it falls back to `safe_off`,
        # which is a genuinely safe state, and reports that it did: the operator
        # gets the safe direction *and* is told the dry purge they asked for did
        # not happen, because a fallback nobody hears about is how a config typo
        # becomes six months of unexplained RH collapses.
        if not (self._out_min > 0.0):
            logger.error("rh_safe_dry_degenerate_out_min",
                         instrument=self.name, out_min=self._out_min)
            self.safe_off()
            outcome = (f" The fallback to safe_off also failed: "
                       f"{self.last_safe_off_error}"
                       if self.last_safe_off_error else
                       " The humidifier was zeroed instead — safe, but the "
                       "chamber will collapse to room RH.")
            self.last_safe_dry_error = (
                f"config [instruments.rh_controller] out_min = "
                f"{self._out_min:g} is not positive, so there is no dry-purge "
                f"duty to command: duty 0 shuts both valves.{outcome}")
            return

        duty = float(self._out_min)

        # Stop first, then write — same race as `safe_off`: a still-running loop
        # writes a fresh PID output every poll period and would overwrite this.
        # The exit duty goes with the stop so the thread's own parting write is
        # `duty` rather than the zero it used to send.
        self._stop_pid_loop(exit_duty=duty)

        with self._thread_lock:
            self._setpoint = 0.0
        if self._pid is not None:
            self._pid.setpoint = 0.0

        if self._serial is None:
            self.last_safe_dry_error = (
                "no serial transport — the dry purge could not be commanded "
                "from this process")
            logger.warning("rh_safe_dry_no_transport", instrument=self.name)
            return

        try:
            self._send_duty(duty)
        except Exception as exc:
            self.last_safe_dry_error = str(exc)
            logger.warning("rh_safe_dry_send_failed", instrument=self.name,
                           error=str(exc))
            return

        self.last_safe_dry_duty = duty
        logger.info("rh_safe_dry", instrument=self.name, duty=duty)

    def wait(
        self,
        target: float | None = None,
        tol: float = 2.0,
        timeout: float = 120.0,
        equilibration_s: float = 0.0,
        raise_on_timeout: bool = False,
    ) -> None:
        """Block until RH reaches *target* ± *tol* or *timeout* expires.

        If *target* is ``None``, uses the current setpoint.  When
        *equilibration_s* > 0 the method holds for that many additional seconds
        after the setpoint is reached, matching the temperature controller
        ``wait(equilibration_time=…)`` pattern.

        ``raise_on_timeout`` decides what an unmet target *means*.  Default
        ``False`` keeps the historical behaviour — log and return — so existing
        callers (monitoring, best-effort settles) are unchanged.  A workflow step
        that **gates** on humidity sets it ``True``: a step which cannot fail is
        not a check, and proceeding into a multi-hour cure at an unknown RH is
        worse than stopping.  The failure is an :class:`InstrumentError` rather
        than a comms/timeout error precisely so the executor does *not* treat it
        as a recoverable glitch and replay the channel — the humidity did not
        arrive, and retrying the wait does not change that.
        """
        if target is None:
            target = self._setpoint
        deadline = time.time() + timeout

        rh = float("nan")
        while time.time() < deadline:
            with self._thread_lock:
                rh = self._current_rh
            if not math.isnan(rh) and abs(rh - target) <= tol:
                logger.info("rh_target_reached", target=target, rh=rh)
                if equilibration_s > 0:
                    time.sleep(equilibration_s)
                return
            time.sleep(self._poll_period)

        logger.warning("rh_wait_timeout", target=target, timeout=timeout, rh=rh)
        if raise_on_timeout:
            # Deliberately NOT CommunicationError/StepTimeoutError: those are the
            # classes WorkflowExecutor._recoverable_cause replays the channel on,
            # and retrying a wait does not make the humidity arrive.
            raise InstrumentError(
                f"RH did not reach {target:g} ±{tol:g} %RH within {timeout:g} s "
                f"(last reading {rh:g} %RH)",
                instrument=self.name,
            )

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
