"""Generic on/off switch backed by a single Adafruit MCP4728 DAC channel.

Use one :class:`AsyncDACSwitch` per load (lamp, power supply, laser, valve,
…).  All instances that share a physical MCP4728 chip connect to it
independently through the MCP2221 USB-to-I²C adapter.

Hardware Requirements
---------------------
- Adafruit MCP4728 quad-channel DAC
- MCP2221 USB-to-I²C adapter
- ``BLINKA_MCP2221=1`` set before ``board``/``busio`` import
- Packages: ``hidapi``, ``adafruit-blinka``,
  ``adafruit-circuitpython-mcp4728``

Configuration (``softae_config.toml``)::

    [instruments.lamp]
    driver   = "dac_switch"
    channel  = "A"        # MCP4728 channel A–D
    address  = "0x60"     # MCP4728 I²C address
    v_psu    = 5.0        # supply / full-scale voltage
    on_volt  = 0.0        # active-low relay: 0 V = ON
    off_volt = 5.0        # relay passive at full scale

    # Future instruments on the same chip just add more entries:
    [instruments.some_b_load]
    driver   = "dac_switch"
    channel  = "B"
    address  = "0x60"
    v_psu    = 5.0
    on_volt  = 0.0
    off_volt = 5.0
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

from softae.drivers.mcp2221_bus import I2C_BUS_LOCK
from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class AsyncDACSwitch(BaseInstrument):
    """Async-wrapped MCP4728 single-channel on/off switch.

    ``connect()`` opens the I²C bus and initialises the MCP4728 DAC.
    ``on()`` / ``off()`` write the configured voltages to the assigned channel.

    If the DAC libraries or hardware are unavailable, ``on()`` / ``off()``
    degrade to a logged no-op so the rest of the system stays usable on
    a dev machine without the physical device.

    Attributes
    ----------
    _channel : str
        MCP4728 channel label — ``"A"``, ``"B"``, ``"C"``, or ``"D"``.
    _address : str
        I²C address as a hex string (e.g. ``"0x60"``).
    _v_psu : float
        Supply voltage used as the 16-bit full-scale reference.
    _on_volt : float
        Voltage written by :meth:`on`.
    _off_volt : float
        Voltage written by :meth:`off`.
    """

    def __init__(self, name: str = "dac_switch", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._channel: str = str(self.config.get("channel", "A"))
        self._address: str = str(self.config.get("address", "0x60"))
        self._v_psu: float = float(self.config.get("v_psu", 5.0))
        self._on_volt: float = float(self.config.get("on_volt", 0.0))
        self._off_volt: float = float(self.config.get("off_volt", 5.0))
        self._is_on: bool = False
        self._i2c = None
        self._dac = None
        self._channel_obj = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the I²C bus and initialise the MCP4728 DAC."""
        os.environ["BLINKA_MCP2221"] = "1"

        try:
            import board
            import busio
            import adafruit_mcp4728

            i2c = busio.I2C(board.SCL, board.SDA)
            dac = adafruit_mcp4728.MCP4728(i2c, address=int(self._address, 16))

            self._i2c = i2c
            self._dac = dac
            self._channel_obj = {
                "A": dac.channel_a,
                "B": dac.channel_b,
                "C": dac.channel_c,
                "D": dac.channel_d,
            }[self._channel]

            self._state = InstrumentState.CONNECTED
            logger.info(
                "dac_switch_connected",
                name=self.name,
                channel=self._channel,
                address=self._address,
            )
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect DAC switch '{self.name}': {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Release the DAC / I²C handles."""
        self._dac = None
        self._i2c = None
        self._channel_obj = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("dac_switch_disconnected", name=self.name, channel=self._channel)

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(is_on=self._is_on, channel=self._channel)
        return s

    # ── Switch API ───────────────────────────────────────────────────────

    def _set_voltage(self, voltage: float) -> None:
        if voltage < 0 or voltage > self._v_psu:
            raise CommunicationError(
                f"Voltage {voltage} V out of range 0–{self._v_psu} V",
                instrument=self.name,
            )
        # Serialise against the SHT31-D reads that share this MCP2221 I²C bus,
        # so a lamp write from the GUI thread (e.g. Emergency Stop) can't collide
        # with the background humidity poll. See softae.drivers.mcp2221_bus.
        with I2C_BUS_LOCK:
            self._channel_obj.value = int((voltage / self._v_psu) * 65535)

    def on(self) -> None:
        """Drive the channel to ``on_volt`` (relay active / load on)."""
        if self._dac is None:
            logger.warning(
                "dac_switch_not_available",
                name=self.name,
                msg="on() requires a connected MCP4728 — no-op",
            )
            self._is_on = True
            return
        self._set_voltage(self._on_volt)
        self._is_on = True
        logger.info("dac_switch_on", name=self.name, channel=self._channel, voltage=self._on_volt)

    def off(self) -> None:
        """Drive the channel to ``off_volt`` (relay passive / load off)."""
        if self._dac is None:
            logger.warning(
                "dac_switch_not_available",
                name=self.name,
                msg="off() requires a connected MCP4728 — no-op",
            )
            self._is_on = False
            return
        self._set_voltage(self._off_volt)
        self._is_on = False
        logger.info("dac_switch_off", name=self.name, channel=self._channel, voltage=self._off_volt)

    def set_eeprom_defaults(self) -> None:
        """Set all four DAC channels to V_psu (5 V) and write to EEPROM.

        Ensures every relay powers up passive / off after any power cycle.
        The MCP4728 saves all four channels in one atomic write, so this
        sets the whole chip — not just this switch's channel.

        Call **once during hardware setup**, not in routine loops (EEPROM
        write cycles are limited, ~100 k).

        Raises
        ------
        CommunicationError
            If :meth:`connect` has not been called yet.
        """
        if self._dac is None:
            raise CommunicationError(
                f"DAC switch '{self.name}' not connected — call connect() first",
                instrument=self.name,
            )
        with I2C_BUS_LOCK:
            for ch in (
                self._dac.channel_a,
                self._dac.channel_b,
                self._dac.channel_c,
                self._dac.channel_d,
            ):
                ch.value = 65535
            time.sleep(1)
            self._dac.save_settings()
        logger.info(
            "dac_switch_eeprom_defaults_saved",
            name=self.name,
            all_channels_V=self._v_psu,
        )
