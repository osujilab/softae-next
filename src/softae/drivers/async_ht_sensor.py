"""Real SHT31-D humidity/temperature sensor driver (I²C via MCP2221).

Wraps the existing blocking reads from ``HTsense_class.py`` behind the
:class:`BaseInstrument` ABC.

Hardware Requirements
---------------------
- Adafruit SHT31-D sensor
- MCP2221 USB-to-I²C adapter
- Environment variable ``BLINKA_MCP2221=1`` must be set **before** import
- Packages: ``hidapi``, ``adafruit-blinka``, ``adafruit-circuitpython-sht31d``

Configuration (``softae_config.toml``)::

    [instruments.ht_sensor]
    vendor_id         = 0x04D8
    product_id        = 0x00DD
    read_retries      = 3      # per-call retry attempts on transient I2C errors
    read_retry_delay_s = 0.05  # seconds between retries
    reset_on_unrecoverable = true  # issue soft-reset on "Unrecoverable" errors
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Any

import structlog

from softae.drivers.mcp2221_bus import I2C_BUS_LOCK
from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)

# These substrings in an exception message identify errors that are worth
# retrying (transient bus noise / CRC corruption).
_RETRYABLE_MSGS = ("crc", "i2c status", "i2c read", "unrecoverable", "oserror", "ioerror")


class AsyncHTSensor(BaseInstrument):
    """Async-wrapped SHT31-D temperature/humidity sensor.

    The HID device is opened once on :meth:`connect` and held open.
    Reads are dispatched via ``run_in_executor`` to avoid blocking
    the event loop.

    Transient I²C failures (CRC mismatch, bus-status errors, unrecoverable
    read failures) are handled with an automatic retry loop inside
    :meth:`_read_with_retry`.  If the sensor reports an "Unrecoverable"
    failure, a soft reset is issued before retrying.
    """

    def __init__(self, name: str = "ht_sensor", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._vendor_id: int = int(self.config.get("vendor_id", 0x04D8))
        self._product_id: int = int(self.config.get("product_id", 0x00DD))
        self._read_retries: int = int(self.config.get("read_retries", 3))
        self._read_retry_delay_s: float = float(self.config.get("read_retry_delay_s", 0.05))
        self._reset_on_unrecoverable: bool = bool(
            self.config.get("reset_on_unrecoverable", True)
        )
        self._device = None
        self._i2c = None
        self._sensor = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the MCP2221 HID device and initialise the SHT31-D."""
        # Blinka requires this env var BEFORE importing board/busio
        os.environ["BLINKA_MCP2221"] = "1"

        try:
            import hid
            import busio
            import board
            import adafruit_sht31d

            device = hid.device()
            device.open(self._vendor_id, self._product_id)

            i2c = busio.I2C(board.SCL, board.SDA)
            sensor = adafruit_sht31d.SHT31D(i2c)

            self._device = device
            self._i2c = i2c
            self._sensor = sensor
            self._state = InstrumentState.CONNECTED
            logger.info("ht_sensor_connected")

        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect HT sensor: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Close the HID device."""
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
            self._i2c = None
            self._sensor = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("ht_sensor_disconnected")

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        if self.is_connected and self._sensor is not None:
            try:
                t, h = self._read_with_retry()
                s["temperature"] = round(t, 1)
                s["humidity"] = round(h, 1)
            except Exception as exc:
                s["error"] = str(exc)
        return s

    # ── Internal I²C read with retry ─────────────────────────────────────

    def _read_with_retry(self) -> tuple[float, float]:
        """Read temperature and humidity, retrying on transient I²C errors.

        The SHT31-D returns both values in a single I²C transaction, so we
        read them together and cache both.  Three error classes are handled:

        * **CRC mismatch** — one corrupted byte; retry immediately.
        * **Couldn't get I²C status** — MCP2221 bridge busy; wait briefly
          and retry.
        * **Unrecoverable I²C read failure** — issue a sensor soft-reset, then
          retry.

        Returns
        -------
        tuple[float, float]
            ``(temperature_C, relative_humidity_pct)``

        Raises
        ------
        CommunicationError
            If all retry attempts are exhausted.
        """
        if self._sensor is None:
            raise CommunicationError("Sensor not connected", instrument=self.name)

        last_exc: Exception | None = None
        for attempt in range(self._read_retries):
            try:
                # adafruit_sht31d reads both T and RH in one transaction.
                # Access temperature first; the library caches both together.
                # Hold the shared MCP2221 lock only around the actual I²C
                # transaction (not the retry sleep below) so DAC-switch writes
                # such as the lamp can't collide with this read.
                with I2C_BUS_LOCK:
                    t = self._sensor.temperature
                    h = self._sensor.relative_humidity
                if attempt > 0:
                    logger.debug(
                        "ht_sensor_read_recovered",
                        attempt=attempt,
                        temperature=round(t, 2),
                        humidity=round(h, 2),
                    )
                return t, h

            except Exception as exc:
                last_exc = exc
                msg_lower = str(exc).lower()
                is_retryable = any(k in msg_lower for k in _RETRYABLE_MSGS)

                if not is_retryable:
                    # Non-transient error — don't retry
                    break

                logger.debug(
                    "ht_sensor_read_retry",
                    attempt=attempt + 1,
                    max_attempts=self._read_retries,
                    error=str(exc),
                )

                # Soft-reset the sensor before retrying unrecoverable errors
                if self._reset_on_unrecoverable and "unrecoverable" in msg_lower:
                    try:
                        with I2C_BUS_LOCK:
                            self._sensor._reset()
                        logger.info("ht_sensor_soft_reset")
                        # Extra settle time on top of what _reset() already waits
                        time.sleep(0.015)
                    except Exception as reset_exc:
                        logger.warning("ht_sensor_reset_failed", error=str(reset_exc))
                        time.sleep(self._read_retry_delay_s)
                else:
                    time.sleep(self._read_retry_delay_s)

        raise CommunicationError(
            f"I²C read failed after {self._read_retries} attempt(s): {last_exc}",
            instrument=self.name,
        ) from last_exc

    # ── Public API (mirrors HTsense_class) ───────────────────────────────

    def get_T(self) -> float:
        """Read temperature (°C) from the SHT31-D (with retry)."""
        if self._sensor is None:
            raise CommunicationError("Sensor not connected", instrument=self.name)
        t, _ = self._read_with_retry()
        return t

    def get_H(self) -> float:
        """Read relative humidity (%) from the SHT31-D (with retry)."""
        if self._sensor is None:
            raise CommunicationError("Sensor not connected", instrument=self.name)
        _, h = self._read_with_retry()
        return h

    def get_TH(self) -> tuple[float, float]:
        """Read ``(temperature_C, relative_humidity_pct)`` in one I²C transaction.

        The SHT31-D returns both values in a single transaction, so callers
        that need temperature *and* humidity should prefer this over separate
        :meth:`get_T` / :meth:`get_H` calls to halve the bus traffic.
        """
        if self._sensor is None:
            raise CommunicationError("Sensor not connected", instrument=self.name)
        return self._read_with_retry()
