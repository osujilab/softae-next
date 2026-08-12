"""Mock Harvard Apparatus syringe pump + pneumatic head — runs without hardware.

Simulates dispensing and head flip/retract/descend, and accumulates dispensed
volume per pump.  Note that ``single_pump``'s ``res_vol`` is the syringe volume
declared to the pump firmware, not stock on hand — remaining stock lives in
:class:`~softae.core.reservoir.ReservoirLedger`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from softae.drivers.contracts import ParallelSyringeMixin
from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockSyringe(ParallelSyringeMixin, BaseInstrument):
    """In-memory syringe pump simulator.

    Parameters
    ----------
    name : str
        Label (default ``"syringe"``).
    config : dict
        Expected keys: ``port``, ``baud``, ``diameter``.
    """

    def __init__(self, name: str = "syringe", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._diameter: float = self.config.get("diameter", 14.4)
        self._max_rate: float = float(self.config.get("max_rate", 2120.0))
        self._min_rate: float = float(self.config.get("min_rate", 0.001))
        self._init_parallel_syringes(self.config)
        self._is_up: bool = True  # head retracted
        self._dispensed: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}  # uL per pump
        # --- Fault injection (tests only) ---
        # Fail the next N single_pump calls with a simulated timeout, to exercise
        # the *committed*-dispense skip path (a pump failure means elution was
        # commanded, so the channel must be skipped, never replayed).
        self._fail_next_n: int = int(self.config.get("fail_next_n", 0))
        self._fail_error_msg: str = self.config.get(
            "fail_error_msg",
            "VI_ERROR_TMO (-1073807339): Timeout expired before operation completed.",
        )

    async def connect(self) -> None:
        logger.info("mock_syringe_connect", port=self.config.get("port", "SIM"))
        await asyncio.sleep(0.02)
        self._state = InstrumentState.CONNECTED
        # Mirrors AsyncSyringe: connecting must NOT assert a head position, or it
        # overwrites the operator's launch-time answer with a guess.

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            diameter=self._diameter,
            head_up=self._is_up,
            dispensed_uL=dict(self._dispensed),
            parallel_syringes=self._parallel_syringes,
            parallel_syringes_by_pump=dict(self._parallel_syringes_by_pump),
        )
        return s

    # --- Syringe API (mirrors syringe_class.syringe) --------------------------
    #
    # set_parallel_syringes() and effective_per_syringe_volume() are provided
    # by ParallelSyringeMixin (shared with AsyncSyringe).

    def single_pump(self, res_vol: float, ID: int, rate: float, dispense_vol: float) -> None:
        """Simulate dispensing *dispense_vol* µL from pump *ID*.

        A commanded ``dispense_vol`` of ``0`` (or less) is a no-op ("leave this
        pump alone") — returned before fault injection, since a pump that is
        never commanded cannot time out.
        """
        if self._is_noop_pump_command(dispense_vol):
            logger.debug("mock_pump_skip", ID=ID, reason="zero_volume")
            return
        if self._fail_next_n > 0:
            self._fail_next_n -= 1
            from softae.errors import CommunicationError

            raise CommunicationError(self._fail_error_msg, instrument=self.name)
        self._validate_single_pump(res_vol, rate, dispense_vol, ID)
        rate = max(rate, 0.001)
        dispense_vol = max(dispense_vol, 0.001)

        hw_vol = self.effective_per_syringe_volume(dispense_vol, ID)

        import time

        delay = (hw_vol / rate) * 60  # seconds
        time.sleep(min(delay, 0.1))  # capped
        self._dispensed[ID] = self._dispensed.get(ID, 0.0) + hw_vol
        logger.debug("mock_pump", ID=ID, rate=rate, vol=hw_vol, requested_vol=dispense_vol)

    def head_flip(self) -> None:
        """Toggle head position."""
        self._is_up = not self._is_up
        logger.debug("mock_head_flip", is_up=self._is_up)

    def head_check(self, confirm_fn=None) -> None:
        """Confirm head position via callback (no ``input()`` in mock)."""
        if confirm_fn is None:
            # In mock mode, assume retracted
            self._is_up = True
            return
        if not confirm_fn("Is the head retracted? (y/n) "):
            self.head_flip()

    def is_head_up(self) -> bool:
        """Registered head position (``True`` = retracted/raised, safe)."""
        return self._is_up

    def set_head_state(self, is_up: bool) -> None:
        """Register the head position without motion (operator verification)."""
        self._is_up = bool(is_up)
        logger.debug("mock_head_state_registered", is_up=self._is_up)

    def head_retract(self) -> None:
        """Ensure head is retracted."""
        if not self._is_up:
            self.head_flip()

    def head_descend(self) -> None:
        """Ensure head is lowered."""
        if self._is_up:
            self.head_flip()

    def syr_end(self) -> None:
        pass
