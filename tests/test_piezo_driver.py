from __future__ import annotations

import asyncio

import pytest

from softae.drivers.mock_piezo import MockPiezoController


def _connect(inst: MockPiezoController) -> None:
    asyncio.run(inst.connect())


def _disconnect(inst: MockPiezoController) -> None:
    asyncio.run(inst.disconnect())


def test_mock_piezo_channel_and_standby():
    inst = MockPiezoController(config={"enabled": True})
    _connect(inst)
    try:
        assert inst.set_channel("A", True) == "A1"
        assert inst.status()["channel_state"]["A"] == 1
        inst.standby()
        status = inst.status()["channel_state"]
        assert status["A"] == 0
        assert status["B"] == 0
    finally:
        _disconnect(inst)


def test_mock_piezo_apply_profile_updates_status():
    inst = MockPiezoController(config={"enabled": True, "supports_l2": True, "supports_cfg": False})
    _connect(inst)
    try:
        assert inst.apply_profile(750, 1.2, 2.4) == "OK"
        s = inst.status()
        assert s["supports_l2"] is True
        assert s["supports_cfg"] is False
        assert s["config_supported"] is True
        assert s["frequency_hz"] == 750
        assert s["sweep_on_s"] == pytest.approx(1.2)
        assert s["sweep_rest_s"] == pytest.approx(2.4)
    finally:
        _disconnect(inst)


def test_mock_piezo_cfg_fallback_when_l2_unavailable():
    inst = MockPiezoController(config={"enabled": True, "supports_l2": False, "supports_cfg": True})
    _connect(inst)
    try:
        assert inst.set_frequency(600) == "OK"
        assert inst.set_sweep(1.5, 2.5) == "OK"
        s = inst.status()
        assert s["supports_l2"] is False
        assert s["supports_cfg"] is True
        assert s["config_supported"] is True
    finally:
        _disconnect(inst)


def test_mock_piezo_legacy_cfg_default_noop():
    inst = MockPiezoController(config={"enabled": True, "supports_cfg": False})
    _connect(inst)
    try:
        assert inst.status()["config_supported"] is False
        assert inst.set_frequency(700) == "LEGACY_NOOP"
        assert inst.set_sweep(1.0, 2.0) == "LEGACY_NOOP"
        assert inst.reset_config() == "LEGACY_NOOP"
    finally:
        _disconnect(inst)


def test_mock_piezo_legacy_cfg_strict_raises():
    inst = MockPiezoController(config={"enabled": True, "supports_cfg": False})
    _connect(inst)
    try:
        with pytest.raises(RuntimeError):
            inst.set_frequency(700, allow_legacy_noop=False)
    finally:
        _disconnect(inst)


def test_mock_piezo_disabled_raises_for_commands():
    inst = MockPiezoController(config={"enabled": False})
    _connect(inst)
    try:
        with pytest.raises(Exception):
            inst.set_channel("A", True)
    finally:
        _disconnect(inst)
