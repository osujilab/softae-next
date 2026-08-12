"""Real Newport ESP301 linear-stage driver (PyVISA serial).

Wraps the blocking VISA commands from the original ``stage_class.py``
behind the :class:`BaseInstrument` ABC so the :class:`InstrumentManager`
can manage connections, locks, and status polling uniformly.

Hardware Requirements
---------------------
- Newport ESP301 motion controller on a VISA serial (ASRL) port
- ``pyvisa`` and ``pyvisa-py`` (or NI-VISA backend)

Configuration (``softae_config.toml``)::

    [instruments.stage]
    port              = "ASRL7::INSTR"
    baud              = 921_600
    velocity          = 10.0
    visa_timeout_ms   = 8000        # PyVISA read timeout (default ~2000 if unset)
    write_termination = "\r"        # ESP301 commands end with CR (manual §3.1)
    read_termination  = "\r"        # robust for CR- or CRLF-terminated replies
    flow_control      = "none"      # "rts_cts" for real RS-232C (manual §3.2.1)
    query_delay       = 0.0         # seconds between write and read on a query

Timeout note
------------
``VI_ERROR_TMO`` from this driver is a *host-side* PyVISA read timeout — the
controller returned nothing in time — not the stage taking too long to move.
Per the ESP301 manual (Table 3.2) a status query answers in ~25 ms over
RS-232C, so the timeout, termination, and flow-control settings above (all left
at library defaults previously) are the real levers. Motion-completion time is
handled separately by :meth:`_wait_until_idle`, which polls and never raises.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog

from softae.drivers.contracts import check_head_clear_to_move, check_stage_bounds
from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)

# Accepted values for the ``flow_control`` config key → PyVISA ControlFlow name.
# Kept as strings so we only import ``pyvisa.constants`` when a non-default flow
# control is actually requested (mock/test paths never touch it).
_FLOW_CONTROL_NAMES = {"none", "rts_cts", "xon_xoff", "dtr_dsr"}


class AsyncStage(BaseInstrument):
    """Async-wrapped Newport ESP301 two-axis linear stage.

    All blocking VISA I/O is dispatched to the shared
    :data:`~softae.server.base_instrument._io_pool` via :meth:`execute`.
    """

    def __init__(self, name: str = "stage", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = self.config.get("port", "ASRL7::INSTR")
        self._baud: int = int(self.config.get("baud", 921_600))
        self._velocity: float = float(self.config.get("velocity", 10.0))
        self._x_min: float = float(self.config.get("x_min", -100.0))
        self._x_max: float = float(self.config.get("x_max", 100.0))
        self._y_min: float = float(self.config.get("y_min", -50.0))
        self._y_max: float = float(self.config.get("y_max", 50.0))
        # --- Comms settings (previously left at PyVISA defaults) ---
        self._visa_timeout_ms: int = int(self.config.get("visa_timeout_ms", 8000))
        self._write_termination: str = self.config.get("write_termination", "\r")
        self._read_termination: str = self.config.get("read_termination", "\r")
        self._flow_control: str = str(self.config.get("flow_control", "none")).lower()
        self._query_delay: float = float(self.config.get("query_delay", 0.0))
        # --- Self-heal cascade on VISA timeout (see _visa_op) ---
        self._tmo_soft_retries: int = int(self.config.get("tmo_soft_retries", 2))
        self._tmo_session_resets: int = int(self.config.get("tmo_session_resets", 1))
        self._tmo_backoff_s: float = float(self.config.get("tmo_backoff_s", 0.5))
        self._tmo_backoff_max: float = float(self.config.get("tmo_backoff_max", 5.0))
        self._visa_inst = None  # pyvisa instrument handle
        self._rm = None         # pyvisa ResourceManager

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the VISA serial connection to the ESP301."""
        try:
            import pyvisa

            self._rm = pyvisa.ResourceManager()
            self._visa_inst = self._rm.open_resource(self._port)
            self._apply_visa_settings(self._visa_inst)
            self._state = InstrumentState.CONNECTED
            logger.info(
                "stage_connected",
                port=self._port,
                baud=self._baud,
                timeout_ms=self._visa_timeout_ms,
                flow_control=self._flow_control,
            )
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect to stage on {self._port}: {exc}",
                instrument=self.name,
            ) from exc

    def _apply_visa_settings(self, inst: Any) -> None:
        """Configure baud, read timeout, termination, and flow control.

        These match the ESP301's documented protocol (manual §3.1–3.2): CR
        command terminator, a generous read timeout, and — on true RS-232C —
        the controller's CTS/RTS hardware handshake. Leaving them at PyVISA
        defaults is the most likely source of intermittent ``VI_ERROR_TMO``.
        Shared by :meth:`connect` and the session-reset recovery path so both
        open the port identically.
        """
        inst.baud_rate = self._baud
        inst.timeout = self._visa_timeout_ms
        inst.write_termination = self._write_termination
        inst.read_termination = self._read_termination
        if self._query_delay:
            inst.query_delay = self._query_delay
        # Flow control is only touched when explicitly requested — the default
        # ("none") leaves USB-CDC virtual COM ports (which don't wire modem
        # lines) untouched, while "rts_cts" enables the ESP301's RS-232C
        # hardware handshake. Non-fatal if the backend rejects it.
        if self._flow_control and self._flow_control != "none":
            try:
                import pyvisa

                flow = getattr(pyvisa.constants.ControlFlow, self._flow_control)
                inst.flow_control = flow
            except Exception as exc:
                logger.warning(
                    "stage_flow_control_unsupported",
                    flow_control=self._flow_control,
                    error=str(exc),
                )

    async def disconnect(self) -> None:
        """Close the VISA session."""
        if self._visa_inst is not None:
            try:
                self._visa_inst.close()
            except Exception:
                pass
            self._visa_inst = None
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("stage_disconnected", port=self._port)

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        if self.is_connected:
            try:
                x, y = self.live_position()
                s["x"] = x
                s["y"] = y
            except Exception as exc:
                s["x"] = None
                s["y"] = None
                s["error"] = str(exc)
        return s

    # ── Public API (mirrors stage_class.stage) ───────────────────────────

    def stage_init(self) -> None:
        """Enable motors and confirm communication."""
        self._visa_inst.baud_rate = self._baud
        logger.info("stage_init", port=self._port)

    def live_position(self, print_flag: int = 0) -> tuple[float, float]:
        """Query and return the current ``(x, y)`` stage position.

        Self-heals on a VISA timeout (position reads are idempotent).

        Returns
        -------
        tuple[float, float]
            Position in mm for each axis.
        """
        return self._visa_op(
            "live_position", lambda: self._live_position_raw(print_flag)
        )

    def _live_position_raw(self, print_flag: int = 0) -> tuple[float, float]:
        """Raw (unwrapped) position query — see :meth:`live_position`."""
        self._visa_inst.write("1MO")
        self._visa_inst.write("2MO")
        x_vals = self._visa_inst.query_ascii_values("1TP")
        x = float(x_vals[0]) if x_vals else 0.0
        y_vals = self._visa_inst.query_ascii_values("2TP")
        y = float(y_vals[0]) if y_vals else 0.0
        if print_flag:
            logger.info("stage_position", x=x, y=y)
        return (x, y)

    def _check_bounds(self, x: float, y: float) -> None:
        """Raise :class:`SafetyError` if ``(x, y)`` is outside stage bounds."""
        check_stage_bounds(
            x, y,
            x_min=self._x_min, x_max=self._x_max,
            y_min=self._y_min, y_max=self._y_max,
            instrument=self.name,
        )

    def _check_head_clear(self, head_may_be_down: bool) -> None:
        """Refuse to translate the stage while the dispenser head is lowered."""
        if head_may_be_down:
            return
        check_head_clear_to_move(
            getattr(self, "head_source", None), instrument=self.name
        )

    def move_to(self, x: float, y: float, *, head_may_be_down: bool = False) -> None:
        """Absolute move to ``(x, y)`` and block until both axes are idle.

        Self-heals on a VISA timeout: an absolute ``PA`` move is idempotent, so
        re-issuing the whole move after a session reset lands the stage at the
        same target — the recovery never re-triggers any surrounding dispense.

        Parameters
        ----------
        x, y : float
            Target coordinates in mm.
        head_may_be_down : bool
            Permit the move with the head lowered. Only in-drop patterning
            (``star_mix``) legitimately needs this; everything else must retract
            first. See :func:`~softae.drivers.contracts.check_head_clear_to_move`.
        """
        self._check_bounds(x, y)  # bounds errors fail fast — never retried
        self._check_head_clear(head_may_be_down)
        self._visa_op("move_to", lambda: self._move_to_raw(x, y))

    def _move_to_raw(self, x: float, y: float) -> None:
        """Raw (unwrapped) absolute move — see :meth:`move_to`."""
        self._visa_inst.write(f"1PA{x}")
        time.sleep(0.5)
        self._visa_inst.write(f"2PA{y}")
        time.sleep(0.5)
        self._wait_until_idle()
        pos = self._live_position_raw()
        logger.info("stage_move_to", target_x=x, target_y=y, pos_x=pos[0], pos_y=pos[1])

    def move_by(self, dx: float, dy: float, *,
                head_may_be_down: bool = False) -> None:
        """Relative move by ``(dx, dy)`` and block until both axes are idle.

        The relative displacement is resolved to an **absolute** target once
        (from a single position read), then issued via the idempotent absolute
        path — so a VISA-timeout retry can never double-apply the displacement.

        Parameters
        ----------
        dx, dy : float
            Relative displacement in mm.

        Note
        ----
        The original ``stage_class.move_by`` referenced an uninitialised
        ``timeout`` variable.  This implementation fixes that bug.
        """
        cur = self.live_position()
        new_x = float(cur[0]) + dx
        new_y = float(cur[1]) + dy
        self._check_bounds(new_x, new_y)
        self._check_head_clear(head_may_be_down)
        self._visa_op(
            "move_by", lambda: self._move_to_raw(new_x, new_y)
        )
        logger.info("stage_move_by", dx=dx, dy=dy, target_x=new_x, target_y=new_y)

    def home_stage(self, velocity: float | None = None) -> None:
        """Execute the homing routine for both axes.

        Self-heals on a VISA timeout (homing to the reference is idempotent).

        Parameters
        ----------
        velocity : float, optional
            Velocity override (currently unused by ESP301 OR command,
            kept for API compatibility with MockStage).
        """
        self._visa_op("home_stage", self._home_stage_raw)

    def _home_stage_raw(self) -> None:
        """Raw (unwrapped) homing routine — see :meth:`home_stage`."""
        self._visa_inst.write("2MO")
        self._visa_inst.write("1MO")
        self._visa_inst.write("1OR")
        self._visa_inst.write("2OR")
        self._wait_until_idle()
        logger.info("stage_homed")

    def stage_end(self) -> None:
        """Close the VISA resource (alias for API compatibility)."""
        if self._visa_inst is not None:
            self._visa_inst.close()
            self._visa_inst = None

    # ── Internal ─────────────────────────────────────────────────────────

    def _axis_motion_done(self, axis: int) -> bool:
        """Return ``True`` when ``axis`` reports motion complete via ``xxMD?``.

        The ESP301 ``MD`` command (manual §3-94) is the documented per-axis
        motion-done query and monitors homing as well as absolute/relative
        moves. It returns ``1`` (done) or ``0`` (moving).
        """
        resp = self._visa_inst.query(f"{axis}MD?").strip()
        return resp.endswith("1")

    def _wait_until_idle(self, timeout: float = 30.0, poll_interval: float = 0.25) -> None:
        """Poll ``1MD?``/``2MD?`` until both axes finish moving or *timeout* expires.

        Uses the ESP301's per-axis motion-done query rather than the global
        ``TS`` status byte — ``MD`` is the command Newport documents for move
        completion (manual §3-94). On timeout this only warns and returns; it is
        the *motion* budget, deliberately decoupled from the VISA read timeout.
        """
        elapsed = 0.0
        while elapsed < timeout:
            if self._axis_motion_done(1) and self._axis_motion_done(2):
                return
            logger.debug("stage_moving", elapsed=round(elapsed, 1))
            time.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning("stage_wait_timeout", timeout=timeout)

    # ── Self-heal cascade ────────────────────────────────────────────────

    def _is_recoverable_visa_error(self, exc: BaseException) -> bool:
        """True for a VISA/comms timeout that a session reset might clear.

        Deliberately excludes :class:`SafetyError` (an out-of-bounds command is
        a program error, never a transient one) so bounds violations fail fast.
        """
        from softae.errors import SafetyError

        if isinstance(exc, SafetyError):
            return False
        if isinstance(exc, CommunicationError):
            return True
        try:
            import pyvisa

            if isinstance(exc, pyvisa.errors.VisaIOError):
                return True
        except Exception:
            pass
        msg = str(exc)
        return (
            "VI_ERROR_TMO" in msg
            or "Timeout expired" in msg
            or "-1073807339" in msg
        )

    def _try_visa_clear(self) -> None:
        """Best-effort VISA buffer clear — the cheapest unwedge before a retry."""
        try:
            if self._visa_inst is not None:
                self._visa_inst.clear()
        except Exception:
            pass

    def _reset_session(self) -> None:
        """Close and reopen the VISA session — the driver-level equivalent of a
        GUI session close, which empirically recovers a wedged stage ~95% of the
        time. Synchronous (runs in the I/O pool thread) and reuses
        :meth:`_apply_visa_settings` so the reopened port is configured identically.
        """
        try:
            if self._visa_inst is not None:
                self._visa_inst.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass
        import pyvisa

        self._rm = pyvisa.ResourceManager()
        self._visa_inst = self._rm.open_resource(self._port)
        self._apply_visa_settings(self._visa_inst)
        self._state = InstrumentState.CONNECTED
        logger.info("stage_session_reset_done", port=self._port)

    def reset_session(self) -> None:
        """Public session-reset seam for external recovery (e.g. the executor's
        ``on_step_recover`` hook). No-op if the driver was never connected."""
        if self._rm is None and self._visa_inst is None:
            return
        self._reset_session()

    def _visa_op(self, label: str, fn: Callable[[], Any]) -> Any:
        """Run *fn* (an idempotent VISA op) under the timeout self-heal cascade.

        Escalation, driven by config keys on ``[instruments.stage]``:

        1. **soft** — on a recoverable VISA timeout, ``clear()`` the buffer,
           back off, and re-issue (``tmo_soft_retries`` times);
        2. **session reset** — if soft retries are exhausted, close/reopen the
           VISA session (``tmo_session_resets`` times) and retry the soft cycle.

        Only recoverable VISA/comms timeouts are caught; anything else (bounds
        violations, programming errors) propagates immediately. When the whole
        cascade is exhausted a :class:`CommunicationError` is raised for the
        executor's higher-level recovery to act on.
        """
        last_exc: BaseException | None = None
        attempts = 0
        for reset_i in range(self._tmo_session_resets + 1):
            if reset_i > 0:
                logger.warning("stage_session_reset", op=label, reset=reset_i)
                try:
                    self._reset_session()
                except Exception as exc:  # reset itself failed — keep escalating
                    last_exc = exc
                    logger.error("stage_session_reset_failed", op=label, error=str(exc))
                    continue
            for soft_i in range(self._tmo_soft_retries + 1):
                try:
                    return fn()
                except Exception as exc:
                    if not self._is_recoverable_visa_error(exc):
                        raise
                    last_exc = exc
                    attempts += 1
                    self._try_visa_clear()
                    logger.warning(
                        "stage_visa_retry",
                        op=label,
                        attempt=attempts,
                        reset_cycle=reset_i,
                        error=str(exc),
                    )
                    time.sleep(
                        min(self._tmo_backoff_s * (2 ** soft_i), self._tmo_backoff_max)
                    )
        raise CommunicationError(
            f"Stage op '{label}' failed after {attempts} attempt(s) across "
            f"{self._tmo_session_resets} session reset(s). Last error: {last_exc}",
            instrument=self.name,
        )
