"""Protocol helpers for the piezo Trinket command channel.

This module centralizes validation and command formatting for both
low-level sender code and src-level drivers.
"""

from __future__ import annotations

from typing import Literal

MIN_FREQUENCY_HZ = 10
MAX_FREQUENCY_HZ = 5000
MIN_SWEEP_SECONDS = 0.01
MAX_SWEEP_SECONDS = 120.0
MIN_SWEEP_MS = int(MIN_SWEEP_SECONDS * 1000)
MAX_SWEEP_MS = int(MAX_SWEEP_SECONDS * 1000)

ChannelName = Literal["A", "B"]


def _is_finite_number(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def normalize_channel(channel: str) -> ChannelName:
    normalized = str(channel).strip().upper()
    if normalized not in {"A", "B"}:
        raise ValueError(f"Invalid channel {channel!r}; expected 'A' or 'B'.")
    return normalized  # type: ignore[return-value]


def normalize_state(enabled: bool | int) -> int:
    value = 1 if bool(enabled) else 0
    return value


def validate_frequency_hz(hz: int) -> int:
    value = int(hz)
    if value < MIN_FREQUENCY_HZ or value > MAX_FREQUENCY_HZ:
        raise ValueError(
            f"Frequency {value} out of range [{MIN_FREQUENCY_HZ}, {MAX_FREQUENCY_HZ}]"
        )
    return value


def validate_sweep_seconds(value: float, field_name: str) -> float:
    out = float(value)
    if not _is_finite_number(out):
        raise ValueError(f"{field_name} must be finite")
    if out < MIN_SWEEP_SECONDS or out > MAX_SWEEP_SECONDS:
        raise ValueError(
            f"{field_name} {out} out of range [{MIN_SWEEP_SECONDS}, {MAX_SWEEP_SECONDS}]"
        )
    return out


def format_legacy_command(channel: str, enabled: bool | int) -> str:
    return f"{normalize_channel(channel)}{normalize_state(enabled)}"


def format_l2_legacy_command(channel: str, enabled: bool | int) -> str:
    return f"{normalize_channel(channel).lower()}{normalize_state(enabled)}"


def validate_sweep_ms(ms: int, field_name: str) -> int:
    value = int(ms)
    if value < MIN_SWEEP_MS or value > MAX_SWEEP_MS:
        raise ValueError(
            f"{field_name} {value} out of range [{MIN_SWEEP_MS}, {MAX_SWEEP_MS}]"
        )
    return value


def format_l2_freq(hz: int) -> str:
    return f"f{validate_frequency_hz(hz)}"


def format_l2_sweep_ms(on_ms: int, rest_ms: int) -> str:
    on_value = validate_sweep_ms(on_ms, "on_ms")
    rest_value = validate_sweep_ms(rest_ms, "rest_ms")
    return f"w{on_value},{rest_value}"


def format_l2_reset() -> str:
    return "r"


def format_l2_caps_query() -> str:
    return "?"


def format_cfg_freq(hz: int) -> str:
    return f"CFG FREQ {validate_frequency_hz(hz)}"


def format_cfg_sweep(on_s: float, rest_s: float) -> str:
    on_seconds = validate_sweep_seconds(on_s, "on_s")
    rest_seconds = validate_sweep_seconds(rest_s, "rest_s")
    return f"CFG SWEEP {on_seconds:.3f} {rest_seconds:.3f}"


def format_cfg_reset() -> str:
    return "CFG RESET"


def format_caps_query() -> str:
    return "CAPS?"


def parse_response(line: str | bytes | None) -> tuple[str, str]:
    """Return ``(kind, payload)`` for listener responses.

    kinds:
    - ``OK``
    - ``ERR`` (payload is reason)
    - ``CAPS`` (payload is capability token)
    - ``UNKNOWN``
    """
    if line is None:
        return ("UNKNOWN", "")

    if isinstance(line, bytes):
        text = line.decode("utf-8", errors="ignore")
    else:
        text = str(line)

    text = text.strip()
    upper = text.upper()

    if upper == "OK":
        return ("OK", "")
    if upper in {"L2", "L1"}:
        return ("CAPS", upper)
    if upper.startswith("ERR"):
        parts = text.split(" ", 1)
        reason = parts[1].strip() if len(parts) == 2 else ""
        return ("ERR", reason)
    if upper.startswith("CAPS"):
        parts = text.split(" ", 1)
        caps = parts[1].strip() if len(parts) == 2 else ""
        return ("CAPS", caps)
    return ("UNKNOWN", text)


def caps_supports_cfg(caps_payload: str) -> bool:
    tokens = str(caps_payload).upper().split()
    return "PIEZO_CFG_V1" in tokens


def caps_supports_l2(caps_payload: str) -> bool:
    tokens = str(caps_payload).upper().split()
    return "L2" in tokens


def parse_capability_response(line: str | bytes | None) -> dict[str, bool]:
    kind, payload = parse_response(line)
    supports_l2 = False
    supports_cfg = False
    if kind == "CAPS":
        supports_l2 = caps_supports_l2(payload)
        supports_cfg = caps_supports_cfg(payload)
    return {
        "supports_l2": supports_l2,
        "supports_cfg": supports_cfg,
        "config_supported": supports_l2 or supports_cfg,
    }
