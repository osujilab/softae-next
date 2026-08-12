"""Mock Newport ESP301 linear stage — runs without hardware.

Simulates absolute/relative moves, position queries, and homing with
realistic timing.  Coordinates are stored in memory.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from softae.drivers.contracts import check_head_clear_to_move, check_stage_bounds
from softae.server.base_instrument import BaseInstrument, InstrumentState, _io_pool

import structlog

logger = structlog.get_logger(__name__)


class MockStage(BaseInstrument):
    """In-memory stage simulator.

    Parameters
    ----------
    name : str
        Label (default ``"stage"``).
    config : dict
        Expected keys: ``port``, ``baud``, ``velocity``.
    """

    def __init__(self, name: str = "stage", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._pos: list[float] = [0.0, 0.0]
        self._velocity: float = self.config.get("velocity", 10.0)  # mm/s
        self._x_min: float = float(self.config.get("x_min", -100.0))
        self._x_max: float = float(self.config.get("x_max", 100.0))
        self._y_min: float = float(self.config.get("y_min", -50.0))
        self._y_max: float = float(self.config.get("y_max", 50.0))
        self._homed: bool = False
        # --- Fault injection (tests only) ---
        # Number of upcoming move_to calls to fail with a simulated VISA timeout,
        # and the error text to raise. Exercises the graceful-recovery cascade
        # (stage self-heal, channel replay) without real hardware.
        self._fail_next_n: int = int(self.config.get("fail_next_n", 0))
        # Targeted variant: fail move_to only when commanded to this (x, y),
        # so a two-phase test can fail the deposit move while the precondition's
        # wick/flush moves succeed. ``fail_at_xy = [x, y]``.
        _fx = self.config.get("fail_at_xy")
        self._fail_at_xy: tuple[float, float] | None = (
            (float(_fx[0]), float(_fx[1])) if _fx else None
        )
        self._fail_at_xy_times: int = int(self.config.get("fail_at_xy_times", 1))
        self._fail_error_msg: str = self.config.get(
            "fail_error_msg",
            "VI_ERROR_TMO (-1073807339): Timeout expired before operation completed.",
        )

    # --- BaseInstrument interface ---------------------------------------------

    async def connect(self) -> None:
        logger.info("mock_stage_connect", port=self.config.get("port", "SIM"))
        await asyncio.sleep(0.05)  # simulate handshake
        self._state = InstrumentState.CONNECTED
        self._homed = False

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED
        logger.info("mock_stage_disconnect")

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            x=self._pos[0],
            y=self._pos[1],
            homed=self._homed,
            velocity=self._velocity,
        )
        return s

    # --- Stage API (mirrors stage_class.stage) --------------------------------

    def stage_init(self) -> None:
        """Enable motors (no-op in mock)."""
        logger.info("mock_stage_init")

    def live_position(self, print_flag: int = 0) -> tuple[float, float]:
        """Return the current ``(x, y)`` position in mm — mirrors AsyncStage."""
        x = round(self._pos[0], 4)
        y = round(self._pos[1], 4)
        if print_flag:
            logger.info("mock_stage_position", x=x, y=y)
        return (x, y)

    def _check_bounds(self, x: float, y: float) -> None:
        """Raise :class:`SafetyError` if ``(x, y)`` is outside stage bounds."""
        check_stage_bounds(
            x, y,
            x_min=self._x_min, x_max=self._x_max,
            y_min=self._y_min, y_max=self._y_max,
            instrument=self.name,
        )

    def reset_session(self) -> None:
        """Parity seam with :class:`AsyncStage.reset_session` — the mock's
        'session reset' clears any injected fault (models a real reset
        recovering a wedged stage)."""
        self._fail_next_n = 0
        self._fail_at_xy_times = 0
        logger.info("mock_stage_session_reset")

    def _maybe_fail(self, x: float, y: float) -> None:
        """Raise a simulated VISA timeout if fault injection is armed."""
        from softae.errors import CommunicationError

        if self._fail_next_n > 0:
            self._fail_next_n -= 1
            raise CommunicationError(self._fail_error_msg, instrument=self.name)
        if (
            self._fail_at_xy is not None
            and self._fail_at_xy_times > 0
            and abs(x - self._fail_at_xy[0]) < 1e-6
            and abs(y - self._fail_at_xy[1]) < 1e-6
        ):
            self._fail_at_xy_times -= 1
            raise CommunicationError(self._fail_error_msg, instrument=self.name)

    def _check_head_clear(self, head_may_be_down: bool) -> None:
        """Refuse to translate the stage while the dispenser head is lowered."""
        if head_may_be_down:
            return
        check_head_clear_to_move(
            getattr(self, "head_source", None), instrument=self.name
        )

    def move_to(self, x: float, y: float, *, head_may_be_down: bool = False) -> None:
        """Simulate absolute move with realistic delay.

        ``head_may_be_down`` mirrors :meth:`AsyncStage.move_to` — the guard is
        shared via :func:`~softae.drivers.contracts.check_head_clear_to_move` so
        mock and real cannot drift on which moves are permitted.
        """
        self._check_bounds(x, y)
        self._check_head_clear(head_may_be_down)
        self._maybe_fail(x, y)
        dx = abs(x - self._pos[0])
        dy = abs(y - self._pos[1])
        travel = max(dx, dy)
        delay = travel / self._velocity if self._velocity else 0
        import time

        time.sleep(min(delay, 0.2))  # capped for mock speed
        # Add tiny noise to mimic real stage jitter
        self._pos = [
            x + random.gauss(0, 0.001),
            y + random.gauss(0, 0.001),
        ]
        logger.debug("mock_stage_move_to", x=x, y=y, pos=self._pos)

    def move_by(self, dx: float, dy: float, *,
                head_may_be_down: bool = False) -> None:
        """Simulate relative move."""
        self.move_to(self._pos[0] + dx, self._pos[1] + dy,
                     head_may_be_down=head_may_be_down)

    def home_stage(self, velocity: float | None = None) -> None:
        """Simulate homing sequence."""
        self._pos = [0.0, 0.0]
        self._homed = True
        logger.info("mock_stage_homed")

    def stage_end(self) -> None:
        """Close (no-op in mock)."""
        pass
