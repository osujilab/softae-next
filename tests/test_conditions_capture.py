"""Tests for reading temp/humidity SP+PVs off the instrument manager."""

from __future__ import annotations

import math

import pytest

from softae.core.conditions_capture import ENV_KEYS, read_environment, stamp_environment


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


# ── stamp_environment: the snapshot onto the spectrum-file header fields ─────


class _Header:
    """The four header fields of an EISResult, NaN until something stamps them."""

    def __init__(self, **preset):
        self.T_sp = self.T_pv = self.rh_sp = self.rh_pv = math.nan
        for name, value in preset.items():
            setattr(self, name, value)


_FULL_ENV = {"stage_temp_sp_C": 10.0, "chamber_air_C": 24.1,
             "stage_temp_pv_C": 26.4, "rh_sp_pct": 22.0, "rh_pv_pct": 20.9}


def test_stamp_environment_nan_fields_filled_from_env():
    result = _Header()
    stamp_environment(result, _FULL_ENV)
    assert (result.T_sp, result.T_pv, result.rh_sp, result.rh_pv) == (
        10.0, 26.4, 22.0, 20.9)


def test_stamp_environment_preset_finite_value_not_overwritten():
    """A value the caller already set wins over the snapshot.

    A caller that stamps its intended temperature knows something the live read
    does not; overwriting it would silently replace a declaration with a reading.
    """
    result = _Header(T_sp=60.0)
    stamp_environment(result, _FULL_ENV)
    assert result.T_sp == 60.0
    assert result.T_pv == 26.4


def test_stamp_environment_none_env_value_leaves_nan():
    result = _Header()
    stamp_environment(result, {**_FULL_ENV, "rh_pv_pct": None, "stage_temp_sp_C": None})
    assert math.isnan(result.rh_pv) and math.isnan(result.T_sp)
    assert result.T_pv == 26.4 and result.rh_sp == 22.0


def test_stamp_environment_chamber_air_never_reaches_header():
    """Chamber air has no header slot: with only it readable, nothing is stamped."""
    result = _Header()
    stamp_environment(result, {k: None for k in ENV_KEYS} | {"chamber_air_C": 24.1})
    assert all(math.isnan(v) for v in (result.T_sp, result.T_pv, result.rh_sp, result.rh_pv))
    assert not hasattr(result, "chamber_air_C")


def test_stamp_environment_real_eis_result_round_trips_through_file(tmp_path):
    """Stamped values survive save/load on a real EISResult, not just a stand-in."""
    import numpy as np

    from softae.analysis.eis_data import EISResult

    f = np.logspace(5, 0, 6)
    z = np.full(6, 1000.0) + 1j * np.linspace(-10.0, -100.0, 6)
    raw = np.column_stack([f, np.abs(z), np.angle(z, deg=True), z.real, -z.imag])
    result = EISResult.from_raw(raw, channel=1)
    stamp_environment(result, _FULL_ENV)
    loaded = EISResult.load(result.save(tmp_path / "s.txt"))
    assert (loaded.T_sp, loaded.T_pv, loaded.rh_sp, loaded.rh_pv) == pytest.approx(
        (10.0, 26.4, 22.0, 20.9))
