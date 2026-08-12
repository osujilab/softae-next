"""Tests for reading temp/humidity SP+PVs off the instrument manager."""

from __future__ import annotations

import math

import pytest

from softae.core.conditions_capture import ENV_KEYS, read_environment


class _FakeTemp:
    """Temperature controller: get_sp = stage SP, get_pv = stage PV (Modbus)."""

    def __init__(self, sp=25.0, pv=22.0):
        self._sp, self._pv = sp, pv

    def get_sp(self):
        return self._sp

    def get_pv(self, n_avg=1):
        return self._pv


class _FakeRH:
    """Humidity controller: get_T = chamber PV, get_H = RH PV, setpoint = RH SP."""

    def __init__(self, sp=50.0, rh=45.0, chamber_T=23.0):
        self._sp, self._rh, self._T = sp, rh, chamber_T

    def get_H(self):
        return self._rh

    def get_T(self):
        return self._T

    def status(self):
        return {"setpoint": self._sp, "current_rh": self._rh}


class _FakeManager:
    def __init__(self, instruments):
        self._instruments = instruments

    @property
    def names(self):
        return list(self._instruments)

    def get(self, name):
        return self._instruments[name]


def test_reads_all_five_values():
    mgr = _FakeManager({"temp_controller": _FakeTemp(), "rh_controller": _FakeRH()})
    env = read_environment(mgr)
    assert set(env) == set(ENV_KEYS)
    assert env["stage_temp_sp_C"] == pytest.approx(25.0)          # stage SP  (temp.get_sp)
    assert env["stage_temp_pv_C"] == pytest.approx(22.0)    # stage PV  (temp.get_pv, Modbus)
    assert env["chamber_air_C"] == pytest.approx(23.0)          # chamber PV (rh.get_T)
    assert env["rh_sp_pct"] == pytest.approx(50.0)          # RH SP
    assert env["rh_pv_pct"] == pytest.approx(45.0)          # RH PV


def test_missing_controllers_yield_none():
    env = read_environment(_FakeManager({}))
    assert env == {k: None for k in ENV_KEYS}


def test_nan_reading_becomes_none():
    """A NaN reading (e.g. RH sensor with a %RH-only reader) maps to None."""
    mgr = _FakeManager(
        {"temp_controller": _FakeTemp(), "rh_controller": _FakeRH(chamber_T=float("nan"))}
    )
    env = read_environment(mgr)
    assert env["chamber_air_C"] is None                        # chamber PV unavailable
    assert env["stage_temp_pv_C"] == pytest.approx(22.0)   # stage PV unaffected


def test_driver_error_is_swallowed():
    class _Boom(_FakeTemp):
        def get_pv(self, n_avg=1):
            raise RuntimeError("comms timeout")

    mgr = _FakeManager({"temp_controller": _Boom(), "rh_controller": _FakeRH()})
    env = read_environment(mgr)
    assert env["stage_temp_pv_C"] is None            # failed stage-PV read
    assert env["stage_temp_sp_C"] == pytest.approx(25.0)   # sibling reads still work


def test_none_manager_is_safe():
    env = read_environment(None)
    assert env == {k: None for k in ENV_KEYS}


def test_with_real_mock_manager():
    """End-to-end against the actual mock drivers registered in the manager."""
    from softae.drivers.mock_factory import create_mock_manager

    env = read_environment(create_mock_manager())
    # Mock seeds: temp SP=25, PV≈22, surf≈21.5; RH SP=50, RH≈45.
    for key in ("stage_temp_sp_C", "chamber_air_C", "stage_temp_pv_C", "rh_sp_pct", "rh_pv_pct"):
        assert env[key] is not None and math.isfinite(env[key])
