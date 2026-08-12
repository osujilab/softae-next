"""Mock SHT31-D humidity/temperature sensor — runs without hardware.

Returns synthetic environmental readings with drift.
"""

from __future__ import annotations

import asyncio
import random
import time as _time
from typing import Any

from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockHTSensor(BaseInstrument):
    """In-memory humidity/temperature sensor simulator."""

    def __init__(self, name: str = "ht_sensor", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._temp: float = 23.0
        self._humidity: float = 45.0

    async def connect(self) -> None:
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(temperature=self.get_T(), humidity=self.get_H())
        return s

    def get_T(self) -> float:
        """Return simulated temperature (°C)."""
        self._temp += random.gauss(0, 0.1)
        return round(self._temp, 1)

    def get_H(self) -> float:
        """Return simulated relative humidity (%)."""
        self._humidity += random.gauss(0, 0.3)
        self._humidity = max(10, min(95, self._humidity))
        return round(self._humidity, 1)

    def get_TH(self) -> tuple[float, float]:
        """Return ``(temperature_C, relative_humidity_pct)`` in one read.

        Mirrors :meth:`AsyncHTSensor.get_TH` — the preferred combined read.
        """
        return self.get_T(), self.get_H()
