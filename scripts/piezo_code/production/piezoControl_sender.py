"""Letter-pair serial command sender script for a Trinket M0.
Intended to adjust extrusion head piezo state, for when head is preparing to dispense.

This module provides a small, reusable PC-side handle for a serial command
scheme where each command is a single channel letter plus a binary state:

* ``A0`` / ``A1`` target the channel on Trinket ``board.A3``
* ``B0`` / ``B1`` target the channel on Trinket ``board.A4``

The class is intentionally GUI-free so it can be imported by scripts,
automation tools, or future controllers that only need to push commands to
the Trinket over ``COM16``.
"""

from __future__ import annotations

import serial

try:
    from softae.core import piezo_protocol
except Exception:  # pragma: no cover - standalone sender fallback
    piezo_protocol = None


MIN_FREQUENCY_HZ = 10
MAX_FREQUENCY_HZ = 5000
MIN_SWEEP_SECONDS = 0.01
MAX_SWEEP_SECONDS = 120.0


class TrinketLetterPairInstrument:
    """Persistent serial handle for a two-channel Trinket command set.

    The handle keeps the serial port open between writes so it can be reused
    by higher-level control code without paying the open/close cost each time.

    Args:
        port: Serial port name. Defaults to ``COM16``.
        baud: Serial baud rate.
        timeout: Serial read/write timeout in seconds.
        auto_open: Open the serial port during construction when ``True``.
    """

    valid_channels = {"A", "B"}
    valid_states = {0, 1}

    def __init__(self, port: str = "COM16", baud: int = 115200, timeout: float = 1.0, auto_open: bool = True):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None
        self._caps_checked = False
        self._supports_l2 = False
        self._supports_cfg = False
        self._config_supported = False

        if auto_open:
            self.open()

    def open(self) -> None:
        """Open the serial connection and clear junk data."""
        if self._ser is not None and self._ser.is_open:
            return

        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout, write_timeout=self.timeout)
        
        # Clear out any residual garbage from previous runs
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        self._probe_capabilities()

    def close(self) -> None:
        """Close the serial connection if it is open."""

        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def _ensure_open(self) -> None:
        if self._ser is None or not self._ser.is_open:
            self.open()

    @staticmethod
    def _is_finite_number(value: float) -> bool:
        return value == value and value not in (float("inf"), float("-inf"))

    @staticmethod
    def _validate_frequency(hz: int) -> int:
        if piezo_protocol is not None:
            return piezo_protocol.validate_frequency_hz(hz)
        value = int(hz)
        if value < MIN_FREQUENCY_HZ or value > MAX_FREQUENCY_HZ:
            raise ValueError(
                f"Frequency {value} out of range [{MIN_FREQUENCY_HZ}, {MAX_FREQUENCY_HZ}]"
            )
        return value

    @staticmethod
    def _validate_sweep_seconds(value: float, field_name: str) -> float:
        if piezo_protocol is not None:
            return piezo_protocol.validate_sweep_seconds(value, field_name)
        out = float(value)
        if not TrinketLetterPairInstrument._is_finite_number(out):
            raise ValueError(f"{field_name} must be finite")
        if out < MIN_SWEEP_SECONDS or out > MAX_SWEEP_SECONDS:
            raise ValueError(
                f"{field_name} {out} out of range [{MIN_SWEEP_SECONDS}, {MAX_SWEEP_SECONDS}]"
            )
        return out

    @staticmethod
    def _normalize_command(channel: str, state: int) -> str:
        if piezo_protocol is not None:
            return piezo_protocol.format_legacy_command(channel, state)
        channel = str(channel).strip().upper()
        state = int(state)

        if channel not in TrinketLetterPairInstrument.valid_channels:
            raise ValueError(f"Invalid channel {channel!r}; expected 'A' or 'B'.")
        if state not in TrinketLetterPairInstrument.valid_states:
            raise ValueError(f"Invalid state {state!r}; expected 0 or 1.")

        return f"{channel}{state}"

    def send_pair(self, channel: str, state: int) -> str:
        """Send one channel/state command, for example ``A1`` or ``B0``.

        A newline is appended so the Trinket-side parser can read complete
        commands with ``readline()``.

        Returns:
            The normalized command string that was transmitted.
        """

        self._ensure_open()
        self._probe_capabilities()
        if piezo_protocol is not None and self._supports_l2:
            command = piezo_protocol.format_l2_legacy_command(channel, state)
        else:
            command = self._normalize_command(channel, state)
        self._ser.write(f"{command}\n".encode("ascii"))
        self._ser.flush()
        return command

    def _readline(self, timeout_override: float | None = None) -> str:
        self._ensure_open()
        if timeout_override is None:
            raw = self._ser.readline()
        else:
            old_timeout = self._ser.timeout
            self._ser.timeout = timeout_override
            try:
                raw = self._ser.readline()
            finally:
                self._ser.timeout = old_timeout
        if not raw:
            return ""
        return raw.decode("utf-8", errors="ignore").strip()

    def _probe_capabilities(self) -> bool:
        """Detect optional l2/CFG protocol support with short timeout."""
        if self._caps_checked:
            return self._config_supported

        self._ensure_open()
        self._supports_l2 = False
        self._supports_cfg = False
        self._config_supported = False
        try:
            self._ser.reset_input_buffer()
            caps_query = piezo_protocol.format_l2_caps_query() if piezo_protocol is not None else "?"
            self._ser.write((caps_query + "\n").encode("ascii"))
            self._ser.flush()
            line = self._readline(timeout_override=0.2)
            if piezo_protocol is not None:
                parsed = piezo_protocol.parse_capability_response(line)
                self._supports_l2 = bool(parsed.get("supports_l2", False))
            else:
                self._supports_l2 = line.strip().lower() == "l2"

            if not self._supports_l2:
                self._ser.reset_input_buffer()
                legacy_caps_query = piezo_protocol.format_caps_query() if piezo_protocol is not None else "CAPS?"
                self._ser.write((legacy_caps_query + "\n").encode("ascii"))
                self._ser.flush()
                line = self._readline(timeout_override=0.2)
                if piezo_protocol is not None:
                    parsed = piezo_protocol.parse_capability_response(line)
                    self._supports_cfg = bool(parsed.get("supports_cfg", False))
                else:
                    self._supports_cfg = line.upper().startswith("CAPS PIEZO_CFG_V1")
        except Exception:
            self._supports_l2 = False
            self._supports_cfg = False
        self._config_supported = self._supports_l2 or self._supports_cfg
        self._caps_checked = True
        return self._config_supported

    @property
    def supports_l2(self) -> bool:
        self._probe_capabilities()
        return self._supports_l2

    @property
    def supports_cfg(self) -> bool:
        self._probe_capabilities()
        return self._supports_cfg

    @property
    def config_supported(self) -> bool:
        return self._probe_capabilities()

    def _send_config_line(self, line: str, *, allow_legacy_noop: bool = True, strict: bool = False) -> str:
        """Send l2/CFG command and parse responses for both modes."""
        self._ensure_open()
        self._probe_capabilities()

        if self._supports_l2:
            self._ser.write((line + "\n").encode("ascii"))
            self._ser.flush()
            response = self._readline(timeout_override=0.2)
            if not response:
                return "OK"
            low = response.lower()
            if low.startswith("e"):
                raise RuntimeError(response)
            if response.startswith("ERR"):
                raise RuntimeError(response)
            return response

        if self._supports_cfg:
            self._ser.write((line + "\n").encode("ascii"))
            self._ser.flush()
            response = self._readline(timeout_override=0.5)
            if response == "OK":
                return response
            if response.startswith("ERR"):
                raise RuntimeError(response)
            return response or "OK"

        if allow_legacy_noop and not strict:
            return "LEGACY_NOOP"
        raise RuntimeError("Firmware does not support config commands")

    def send_command(self, command: str) -> str:
        """Send a preformatted command string such as ``A1`` or ``B0``."""

        command = str(command).strip().upper()
        if len(command) != 2:
            raise ValueError("Command must be exactly two characters long.")

        return self.send_pair(command[0], int(command[1]))

    def set_channel(self, channel: str, enabled: bool) -> str:
        """Convenience wrapper that maps truthy values to ``1`` and falsy to ``0``."""
        state = 1 if enabled else 0
        if piezo_protocol is not None:
            self._probe_capabilities()
            if self._supports_l2:
                command = piezo_protocol.format_l2_legacy_command(channel, state)
                self._ensure_open()
                self._ser.write(f"{command}\n".encode("ascii"))
                self._ser.flush()
                return command
        return self.send_pair(channel, state)

    def set_frequency(self, hz: int, *, allow_legacy_noop: bool = True, strict: bool = False) -> str:
        """Set shared PWM frequency via CFG protocol when supported."""
        self._probe_capabilities()
        if piezo_protocol is not None and self._supports_l2:
            line = piezo_protocol.format_l2_freq(hz)
        elif piezo_protocol is not None:
            line = piezo_protocol.format_cfg_freq(hz)
        else:
            freq = self._validate_frequency(hz)
            line = f"CFG FREQ {freq}"
        return self._send_config_line(line, allow_legacy_noop=allow_legacy_noop, strict=strict)

    def set_sweep(self, on_s: float, rest_s: float, *, allow_legacy_noop: bool = True, strict: bool = False) -> str:
        """Set shared sweep on/rest timing via CFG protocol when supported."""
        self._probe_capabilities()
        if piezo_protocol is not None and self._supports_l2:
            on_ms = int(round(self._validate_sweep_seconds(on_s, "on_s") * 1000.0))
            rest_ms = int(round(self._validate_sweep_seconds(rest_s, "rest_s") * 1000.0))
            line = piezo_protocol.format_l2_sweep_ms(on_ms, rest_ms)
        elif piezo_protocol is not None:
            line = piezo_protocol.format_cfg_sweep(on_s, rest_s)
        else:
            on_seconds = self._validate_sweep_seconds(on_s, "on_s")
            rest_seconds = self._validate_sweep_seconds(rest_s, "rest_s")
            line = f"CFG SWEEP {on_seconds:.3f} {rest_seconds:.3f}"
        return self._send_config_line(line, allow_legacy_noop=allow_legacy_noop, strict=strict)

    def reset_config(self, *, allow_legacy_noop: bool = True, strict: bool = False) -> str:
        """Reset frequency and sweep settings to firmware defaults."""
        self._probe_capabilities()
        if piezo_protocol is not None and self._supports_l2:
            line = piezo_protocol.format_l2_reset()
        else:
            line = piezo_protocol.format_cfg_reset() if piezo_protocol is not None else "CFG RESET"
        return self._send_config_line(line, allow_legacy_noop=allow_legacy_noop, strict=strict)

    def apply_profile(
        self,
        frequency_hz: int,
        on_s: float,
        rest_s: float,
        *,
        allow_legacy_noop: bool = True,
        strict: bool = False,
    ) -> None:
        """Apply frequency + sweep settings as one profile action."""
        self.set_frequency(frequency_hz, allow_legacy_noop=allow_legacy_noop, strict=strict)
        self.set_sweep(on_s, rest_s, allow_legacy_noop=allow_legacy_noop, strict=strict)

    def query_caps(self) -> str:
        """Return capability token string, probing if needed."""
        self._probe_capabilities()
        if self._supports_l2:
            return "L2"
        if self._supports_cfg:
            return "CAPS PIEZO_CFG_V1"
        return "LEGACY"

    def standby(self) -> None:
        """Turn both Trinket PWM channels off with a tiny safety delay."""
        import time
        self.send_pair("A", 0)
        time.sleep(0.05)  # Give the Trinket 50ms to process 'A0'
        self.send_pair("B", 0)

__all__ = ["TrinketLetterPairInstrument"]