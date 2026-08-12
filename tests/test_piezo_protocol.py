from __future__ import annotations

import pytest

from softae.core import piezo_protocol


def test_format_legacy_command():
    assert piezo_protocol.format_legacy_command("a", True) == "A1"
    assert piezo_protocol.format_legacy_command("B", 0) == "B0"


def test_format_cfg_commands():
    assert piezo_protocol.format_cfg_freq(500) == "CFG FREQ 500"
    assert piezo_protocol.format_cfg_sweep(2.0, 3.0) == "CFG SWEEP 2.000 3.000"
    assert piezo_protocol.format_cfg_reset() == "CFG RESET"
    assert piezo_protocol.format_caps_query() == "CAPS?"


def test_format_l2_commands():
    assert piezo_protocol.format_l2_legacy_command("a", True) == "a1"
    assert piezo_protocol.format_l2_freq(500) == "f500"
    assert piezo_protocol.format_l2_sweep_ms(2000, 3000) == "w2000,3000"
    assert piezo_protocol.format_l2_reset() == "r"
    assert piezo_protocol.format_l2_caps_query() == "?"


@pytest.mark.parametrize("hz", [9, 5001])
def test_validate_frequency_rejects_out_of_range(hz):
    with pytest.raises(ValueError):
        piezo_protocol.validate_frequency_hz(hz)


@pytest.mark.parametrize("value", [0.0, 121.0, float("inf"), float("nan")])
def test_validate_sweep_rejects_invalid(value):
    with pytest.raises(ValueError):
        piezo_protocol.validate_sweep_seconds(value, "on_s")


def test_parse_response_ok_err_caps_unknown():
    assert piezo_protocol.parse_response("OK") == ("OK", "")
    assert piezo_protocol.parse_response("ERR BAD_SWEEP") == ("ERR", "BAD_SWEEP")
    assert piezo_protocol.parse_response("CAPS PIEZO_CFG_V1") == ("CAPS", "PIEZO_CFG_V1")
    assert piezo_protocol.parse_response("l2") == ("CAPS", "L2")
    assert piezo_protocol.parse_response("HELLO") == ("UNKNOWN", "HELLO")


def test_caps_supports_cfg():
    assert piezo_protocol.caps_supports_cfg("PIEZO_CFG_V1") is True
    assert piezo_protocol.caps_supports_cfg("LEGACY_ONLY") is False


def test_parse_capability_response_prefers_l2_and_cfg():
    caps = piezo_protocol.parse_capability_response("l2")
    assert caps["supports_l2"] is True
    assert caps["supports_cfg"] is False
    assert caps["config_supported"] is True

    caps = piezo_protocol.parse_capability_response("CAPS PIEZO_CFG_V1")
    assert caps["supports_l2"] is False
    assert caps["supports_cfg"] is True
    assert caps["config_supported"] is True

    caps = piezo_protocol.parse_capability_response("")
    assert caps["supports_l2"] is False
    assert caps["supports_cfg"] is False
    assert caps["config_supported"] is False
