"""Async-managed serial driver for Trinket piezo control."""

from __future__ import annotations

import time
from typing import Any

import structlog

from softae.core import piezo_protocol
from softae.drivers.contracts import apply_piezo_profile
from softae.errors import CommunicationError, ConnectionError_, InstrumentError
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class AsyncPiezoController(BaseInstrument):
    """Serial piezo controller for legacy + CFG-capable Trinket firmware."""

    def __init__(self, name: str = "piezo", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = str(self.config.get("port", "COM16"))
        self._baud: int = int(self.config.get("baud", 115200))
        self._timeout: float = float(self.config.get("timeout", 0.5))
        self._enabled: bool = bool(self.config.get("enabled", False))

        self._serial = None
        self._supports_l2: bool = False
        self._supports_cfg: bool = False
        self._config_supported: bool = False
        self._caps_checked: bool = False
        self._caps_last_probe_s: float = 0.0

        self._frequency_hz: int = int(self.config.get("frequency_hz", 500))
        self._sweep_on_s: float = float(self.config.get("sweep_on_s", 2.0))
        self._sweep_rest_s: float = float(self.config.get("sweep_rest_s", 3.0))
        self._channel_state: dict[str, int] = {"A": 0, "B": 0}

    async def connect(self) -> None:
        if not self._enabled:
            self._state = InstrumentState.CONNECTED
            logger.info("piezo_disabled", instrument=self.name)
            return

        try:
            import serial

            self._serial = serial.Serial(
                self._port,
                self._baud,
                timeout=self._timeout,
                write_timeout=self._timeout,
            )
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()
            self._probe_capabilities()
            self._state = InstrumentState.CONNECTED
            logger.info(
                "piezo_connected",
                instrument=self.name,
                port=self._port,
                supports_l2=self._supports_l2,
                supports_cfg=self._supports_cfg,
            )
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect piezo on {self._port}: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        # Startup sequencing can make the first probe race the Trinket boot.
        # Re-probe periodically until capability support is discovered.
        if (
            self._enabled
            and self._serial is not None
            and self._serial.is_open
            and not self._config_supported
        ):
            now = time.monotonic()
            if now - self._caps_last_probe_s >= 1.0:
                self._caps_checked = False
                self._probe_capabilities()

        s = self._base_status()
        s.update(
            enabled=self._enabled,
            port=self._port,
            baud=self._baud,
            supports_l2=self._supports_l2,
            supports_cfg=self._supports_cfg,
            config_supported=self._config_supported,
            channel_state=dict(self._channel_state),
            frequency_hz=self._frequency_hz,
            sweep_on_s=self._sweep_on_s,
            sweep_rest_s=self._sweep_rest_s,
        )
        return s

    def _ensure_ready(self) -> None:
        if not self._enabled:
            raise InstrumentError("piezo instrument is disabled by config", instrument=self.name)
        if self._serial is None or not self._serial.is_open:
            raise CommunicationError("piezo serial port is not open", instrument=self.name)

    def _send_line(self, line: str) -> None:
        self._ensure_ready()
        self._serial.write((line + "\n").encode("ascii"))
        self._serial.flush()

    def _readline(self, timeout_override: float | None = None) -> str:
        self._ensure_ready()
        if timeout_override is None:
            raw = self._serial.readline()
        else:
            old_timeout = self._serial.timeout
            self._serial.timeout = timeout_override
            try:
                raw = self._serial.readline()
            finally:
                self._serial.timeout = old_timeout
        if not raw:
            return ""
        return raw.decode("utf-8", errors="ignore").strip()

    def _probe_capabilities(self) -> bool:
        if self._caps_checked:
            return self._config_supported
        self._caps_last_probe_s = time.monotonic()
        try:
            self._send_line(piezo_protocol.format_l2_caps_query())
            parsed = piezo_protocol.parse_capability_response(self._readline(timeout_override=0.2))
            self._supports_l2 = bool(parsed.get("supports_l2", False))

            if not self._supports_l2:
                self._send_line(piezo_protocol.format_caps_query())
                parsed = piezo_protocol.parse_capability_response(self._readline(timeout_override=0.2))
                self._supports_cfg = bool(parsed.get("supports_cfg", False))
            else:
                self._supports_cfg = False
        except Exception:
            self._supports_l2 = False
            self._supports_cfg = False
        self._config_supported = self._supports_l2 or self._supports_cfg
        self._caps_checked = True
        return self._config_supported

    def _send_config(
        self,
        cfg_line: str,
        l2_line: str | None = None,
        *,
        allow_legacy_noop: bool = True,
    ) -> str:
        self._probe_capabilities()
        if self._supports_l2:
            self._send_line(l2_line or cfg_line)
            response = self._readline(timeout_override=0.2)
            if not response:
                return "OK"
            if response.lower().startswith("e"):
                raise RuntimeError(response)
            kind, payload = piezo_protocol.parse_response(response)
            if kind == "ERR":
                raise RuntimeError(payload or "ERR")
            return payload or "OK"
        if self._supports_cfg:
            self._send_line(cfg_line)
            kind, payload = piezo_protocol.parse_response(self._readline(timeout_override=0.5))
            if kind == "OK":
                return "OK"
            if kind == "ERR":
                raise RuntimeError(payload or "ERR")
            return payload or "OK"
        if allow_legacy_noop:
            return "LEGACY_NOOP"
        raise RuntimeError("Firmware does not support config commands")

    def set_channel(self, channel: str, enabled: bool) -> str:
        self._probe_capabilities()
        if self._supports_l2:
            cmd = piezo_protocol.format_l2_legacy_command(channel, enabled)
        else:
            cmd = piezo_protocol.format_legacy_command(channel, enabled)
        self._send_line(cmd)
        ch = piezo_protocol.normalize_channel(channel)
        self._channel_state[ch] = 1 if enabled else 0
        return cmd

    def standby(self) -> None:
        self.set_channel("A", False)
        self.set_channel("B", False)

    def set_frequency(self, hz: int, *, allow_legacy_noop: bool = True) -> str:
        cfg_line = piezo_protocol.format_cfg_freq(hz)
        l2_line = piezo_protocol.format_l2_freq(hz)
        response = self._send_config(cfg_line, l2_line, allow_legacy_noop=allow_legacy_noop)
        if response == "LEGACY_NOOP":
            return response
        self._frequency_hz = piezo_protocol.validate_frequency_hz(hz)
        return response

    def set_sweep(self, on_s: float, rest_s: float, *, allow_legacy_noop: bool = True) -> str:
        cfg_line = piezo_protocol.format_cfg_sweep(on_s, rest_s)
        on_ms = int(round(piezo_protocol.validate_sweep_seconds(on_s, "on_s") * 1000.0))
        rest_ms = int(round(piezo_protocol.validate_sweep_seconds(rest_s, "rest_s") * 1000.0))
        l2_line = piezo_protocol.format_l2_sweep_ms(on_ms, rest_ms)
        response = self._send_config(cfg_line, l2_line, allow_legacy_noop=allow_legacy_noop)
        if response == "LEGACY_NOOP":
            return response
        self._sweep_on_s = piezo_protocol.validate_sweep_seconds(on_s, "on_s")
        self._sweep_rest_s = piezo_protocol.validate_sweep_seconds(rest_s, "rest_s")
        return response

    def reset_config(self, *, allow_legacy_noop: bool = True) -> str:
        response = self._send_config(
            piezo_protocol.format_cfg_reset(),
            piezo_protocol.format_l2_reset(),
            allow_legacy_noop=allow_legacy_noop,
        )
        if response == "LEGACY_NOOP":
            return response
        self._frequency_hz = 500
        self._sweep_on_s = 2.0
        self._sweep_rest_s = 3.0
        return response

    def apply_profile(
        self,
        frequency_hz: int,
        on_s: float,
        rest_s: float,
        *,
        allow_legacy_noop: bool = True,
    ) -> str:
        return apply_piezo_profile(
            self, frequency_hz, on_s, rest_s, allow_legacy_noop=allow_legacy_noop
        )
