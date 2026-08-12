"""Mock Keithley DAQ6510 multimeter — runs without hardware.

Returns synthetic 4-wire resistance readings.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockKeithley(BaseInstrument):
    """In-memory multimeter simulator."""

    def __init__(self, name: str = "keithley", config: dict[str, Any] | None = None):
        super().__init__(name, config)

    async def connect(self) -> None:
        logger.info("mock_keithley_connect")
        await asyncio.sleep(0.02)
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        return self._base_status()

    # --- Keithley API ---------------------------------------------------------

    def singleCh_measure(self, ch: int = 101, nplc: float = 1.0) -> float:
        """Return a synthetic resistance reading (Ohms)."""
        base = 150 + ch * 10
        return base + random.gauss(0, 5)

    def multi_measure(self, ch_start: int = 101, ch_end: int = 110, nplc: float = 1.0) -> list[float]:
        """Return synthetic readings for a range of channels."""
        return [self.singleCh_measure(ch, nplc) for ch in range(ch_start, ch_end + 1)]
