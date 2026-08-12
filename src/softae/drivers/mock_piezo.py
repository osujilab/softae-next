"""Mock piezo driver for GUI and workflow tests."""

from __future__ import annotations

from typing import Any

from softae.core import piezo_protocol
from softae.drivers.contracts import apply_piezo_profile
from softae.errors import InstrumentError
from softae.server.base_instrument import BaseInstrument, InstrumentState


class MockPiezoController(BaseInstrument):
    """In-memory piezo controller with optional CFG capability support."""

    def __init__(self, name: str = "piezo", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._enabled: bool = bool(self.config.get("enabled", False))
        self._supports_l2: bool = bool(self.config.get("supports_l2", False))
        self._supports_cfg: bool = bool(self.config.get("supports_cfg", True))
        self._config_supported: bool = self._supports_l2 or self._supports_cfg
        self._frequency_hz: int = int(self.config.get("frequency_hz", 500))
        self._sweep_on_s: float = float(self.config.get("sweep_on_s", 2.0))
        self._sweep_rest_s: float = float(self.config.get("sweep_rest_s", 3.0))
        self._channel_state: dict[str, int] = {"A": 0, "B": 0}

    async def connect(self) -> None:
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            enabled=self._enabled,
            supports_l2=self._supports_l2,
            supports_cfg=self._supports_cfg,
            config_supported=self._config_supported,
            frequency_hz=self._frequency_hz,
            sweep_on_s=self._sweep_on_s,
            sweep_rest_s=self._sweep_rest_s,
            channel_state=dict(self._channel_state),
        )
        return s

    def _ensure_enabled(self) -> None:
        if not self._enabled:
            raise InstrumentError("piezo instrument is disabled by config", instrument=self.name)

    def set_channel(self, channel: str, enabled: bool) -> str:
        self._ensure_enabled()
        ch = piezo_protocol.normalize_channel(channel)
        self._channel_state[ch] = 1 if enabled else 0
        return piezo_protocol.format_legacy_command(ch, enabled)

    def standby(self) -> None:
        self._ensure_enabled()
        self._channel_state["A"] = 0
        self._channel_state["B"] = 0

    def set_frequency(self, hz: int, *, allow_legacy_noop: bool = True) -> str:
        self._ensure_enabled()
        if not self._config_supported:
            if allow_legacy_noop:
                return "LEGACY_NOOP"
            raise RuntimeError("Firmware does not support config commands")
        self._frequency_hz = piezo_protocol.validate_frequency_hz(hz)
        return "OK"

    def set_sweep(self, on_s: float, rest_s: float, *, allow_legacy_noop: bool = True) -> str:
        self._ensure_enabled()
        if not self._config_supported:
            if allow_legacy_noop:
                return "LEGACY_NOOP"
            raise RuntimeError("Firmware does not support config commands")
        self._sweep_on_s = piezo_protocol.validate_sweep_seconds(on_s, "on_s")
        self._sweep_rest_s = piezo_protocol.validate_sweep_seconds(rest_s, "rest_s")
        return "OK"

    def reset_config(self, *, allow_legacy_noop: bool = True) -> str:
        self._ensure_enabled()
        if not self._config_supported:
            if allow_legacy_noop:
                return "LEGACY_NOOP"
            raise RuntimeError("Firmware does not support config commands")
        self._frequency_hz = 500
        self._sweep_on_s = 2.0
        self._sweep_rest_s = 3.0
        return "OK"

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
