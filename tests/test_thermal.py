"""Tests for VFT/Arrhenius model selection, unified storage, and config validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softae.analysis.arrhenius import ArrheniusFitter, ArrheniusResult, ArrheniusSweepConfig
from softae.analysis.thermal import THERMAL_MODELS, fit_thermal, make_fitter
from softae.analysis.vft import VftFitter, VftResult
from softae.core.data_store import DataStore
from softae.errors import AnalysisError


def _arrhenius_series(Ea=0.3, temps=(25, 45, 65, 85)):
    kB = 8.617333e-5
    sig = [math.exp(-Ea / (kB * (t + 273.15))) for t in temps]
    return list(map(float, temps)), sig


# ── dispatch layer ─────────────────────────────────────────────────────────

def test_make_fitter_selects_class():
    assert isinstance(make_fitter("arrhenius"), ArrheniusFitter)
    assert isinstance(make_fitter("vft"), VftFitter)
    assert set(THERMAL_MODELS) == {"arrhenius", "vft"}


def test_make_fitter_unknown_raises():
    with pytest.raises(AnalysisError, match="unknown thermal model"):
        make_fitter("nope")


def test_fit_thermal_returns_tagged_result():
    temps, sig = _arrhenius_series()
    ar = fit_thermal("arrhenius", temps, sig)
    vf = fit_thermal("vft", temps, sig)
    assert isinstance(ar, ArrheniusResult) and ar.model == "arrhenius" and ar.fit_success
    assert isinstance(vf, VftResult) and vf.model == "vft"


def test_vft_result_has_tmin_tmax_on_success():
    A, B, T0 = 1.0, 600.0, 180.0
    temps = [20.0, 40.0, 60.0, 80.0, 100.0]
    sig = [A * math.exp(-B / (t + 273.15 - T0)) for t in temps]
    res = VftFitter().fit(temps, sig)
    assert res.fit_success
    assert res.T_min_C == pytest.approx(20.0)
    assert res.T_max_C == pytest.approx(100.0)


def test_vft_reports_activation_energy():
    # Activation-energy formalism: Eₐ = B · k_B (B fitted in Kelvin).
    kB = 8.617333e-5
    A, B, T0 = 1.0, 600.0, 180.0
    temps = [20.0, 40.0, 60.0, 80.0, 100.0]
    sig = [A * math.exp(-B / (t + 273.15 - T0)) for t in temps]
    res = VftFitter().fit(temps, sig)
    assert res.fit_success
    assert res.B == pytest.approx(600.0, rel=0.05)
    assert res.Ea_eV == pytest.approx(res.B * kB, rel=1e-9)
    assert res.Ea_eV == pytest.approx(600.0 * kB, rel=0.05)
    assert res.A == pytest.approx(1.0, rel=0.1)            # σ_∞ prefactor


def test_arrhenius_fit_uses_natural_log():
    # The fit must be in natural log: intercept = ln(σ_∞), slope = −Eₐ/k_B.
    # Build σ = A·exp(−Eₐ/(k_B·T)) with a non-trivial prefactor and recover both.
    kB = 8.617333e-5
    A_true, Ea_true = 12.5, 0.42        # A_true ≠ 1 so log-base matters
    temps = [25.0, 45.0, 65.0, 85.0, 105.0]
    sig = [A_true * math.exp(-Ea_true / (kB * (t + 273.15))) for t in temps]
    res = ArrheniusFitter().fit(temps, sig)
    assert res.fit_success
    assert res.Ea_eV == pytest.approx(Ea_true, rel=1e-6)
    # exp(ln_A) recovers A_true → intercept is a NATURAL log, not log10.
    assert math.exp(res.ln_A) == pytest.approx(A_true, rel=1e-6)
    # (A log10 fit would give 10**intercept ≈ A_true instead, i.e. ln_A ≈ ln(A).)


# ── unified storage (one table, both models) ───────────────────────────────

@pytest.fixture()
def store(tmp_path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def test_thermal_table_has_vft_columns(store):
    cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(arrhenius_results)").fetchall()
    }
    assert {"model", "A", "B", "T0_K", "T0_C"} <= cols


def test_record_query_vft_roundtrip(store):
    run_id = store.start_run("t")
    temps = [20.0, 40.0, 60.0, 80.0, 100.0]
    sig = [1.0 * math.exp(-600.0 / (t + 273.15 - 180.0)) for t in temps]
    res = VftFitter().fit(temps, sig, channel=3)
    store.record_thermal_fit(run_id, res)
    rows = store.query_thermal_fits(run_id=run_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "vft"
    assert row["channel"] == 3
    assert row["B"] == pytest.approx(res.B, rel=1e-6)
    assert row["T0_K"] == pytest.approx(res.T0_K, rel=1e-6)
    # VFT now reports Eₐ via the activation-energy formalism (Eₐ = B·k_B).
    assert row["Ea_eV"] == pytest.approx(res.Ea_eV, rel=1e-6)
    assert np.allclose(row["temperatures_C"], temps)


def test_record_arrhenius_alias_sets_model(store):
    run_id = store.start_run("t")
    temps, sig = _arrhenius_series()
    res = ArrheniusFitter().fit(temps, sig, channel=1)
    store.record_arrhenius(run_id, res)          # back-compat alias
    row = store.query_arrhenius(run_id=run_id)[0]
    assert row["model"] == "arrhenius"
    assert row["Ea_eV"] is not None
    assert row["B"] is None                      # not applicable to Arrhenius


# ── config validation ──────────────────────────────────────────────────────

def test_config_rejects_unknown_thermal_model():
    cfg = ArrheniusSweepConfig(channels=[1], T_start=25, T_stop=55, T_step=10,
                               thermal_model="nope")
    with pytest.raises(ValueError, match="thermal_model"):
        cfg.validate()


def test_config_vft_requires_three_temperatures():
    cfg = ArrheniusSweepConfig(channels=[1], T_start=25, T_stop=45, T_step=20,
                               thermal_model="vft")  # only 2 temps → invalid
    with pytest.raises(ValueError, match="at least 3 temperatures"):
        cfg.validate()


def test_config_vft_ok_with_three_temperatures():
    cfg = ArrheniusSweepConfig(channels=[1], T_start=25, T_stop=65, T_step=20,
                               thermal_model="vft")  # 25,45,65 → 3 temps
    cfg.validate()  # should not raise
    assert cfg.thermal_model == "vft"


def test_config_roundtrip_preserves_thermal_model():
    cfg = ArrheniusSweepConfig(channels=[1], thermal_model="vft",
                               T_start=25, T_stop=65, T_step=20)
    cfg2 = ArrheniusSweepConfig.from_json(cfg.to_json())
    assert cfg2.thermal_model == "vft"
