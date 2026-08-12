"""Tests for EIS data pipeline and circuit-fitting modules.

Covers:
  * EISResult construction, save/load round-trip, to_dict
  * _parse_channel_spec edge cases
  * extract_features basic sanity
  * z_to_sigma formula
  * FitResult.sigma property
  * CIRCUIT_MODELS registry completeness
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis_data import EISResult
from softae.analysis.circuit_fitting import (
    CIRCUIT_MODELS,
    FitResult,
    extract_features,
    predict_fit_curve,
    z_to_sigma,
)


# ── EISResult construction ───────────────────────────────────────────────


class TestEISResult:
    """EISResult dataclass lifecycle."""

    @pytest.fixture
    def sample_result(self) -> EISResult:
        """A minimal synthetic EIS result."""
        f = np.logspace(1, 5, 50)
        z_real = 100 + 50 / (1 + (2 * np.pi * f * 1e-6) ** 2)
        z_imag_neg = -50 * 2 * np.pi * f * 1e-6 / (1 + (2 * np.pi * f * 1e-6) ** 2)
        z_mag = np.sqrt(z_real**2 + z_imag_neg**2)
        phase = np.degrees(np.arctan2(-z_imag_neg, z_real))
        return EISResult(
            channel=5,
            frequency=f,
            z_magnitude=z_mag,
            phase=phase,
            z_real=z_real,
            z_imag_neg=np.abs(z_imag_neg),  # convention: positive
        )

    def test_npts(self, sample_result: EISResult):
        assert sample_result.npts == 50

    def test_z_complex_shape(self, sample_result: EISResult):
        z = sample_result.z_complex
        assert z.shape == (50,)
        assert np.iscomplexobj(z)

    def test_to_dict_keys(self, sample_result: EISResult):
        d = sample_result.to_dict()
        assert "channel" in d
        assert "npts" in d
        assert "f_min_Hz" in d
        assert "f_max_Hz" in d

    def test_save_load_roundtrip(self, sample_result: EISResult, tmp_path: Path):
        out = tmp_path / "test_eis.txt"
        sample_result.save(out, study_name="unit_test")
        loaded = EISResult.load(out)
        assert loaded.channel == sample_result.channel
        assert loaded.npts == sample_result.npts
        np.testing.assert_allclose(loaded.frequency, sample_result.frequency, rtol=1e-6)
        np.testing.assert_allclose(loaded.z_real, sample_result.z_real, rtol=1e-6)

    def test_save_creates_file(self, sample_result: EISResult, tmp_path: Path):
        out = tmp_path / "created.txt"
        sample_result.save(out, study_name="test")
        assert out.exists()

    def test_save_load_roundtrip_with_residual_columns(self, sample_result: EISResult, tmp_path: Path):
        out = tmp_path / "test_eis_with_residuals.txt"
        sample_result.residual_real_pct = np.linspace(-2.0, 2.0, sample_result.npts)
        sample_result.residual_imag_pct = np.linspace(1.5, -1.5, sample_result.npts)

        sample_result.save(out, study_name="unit_test_residuals")
        loaded = EISResult.load(out)

        assert loaded.residual_real_pct is not None
        assert loaded.residual_imag_pct is not None
        np.testing.assert_allclose(loaded.residual_real_pct, sample_result.residual_real_pct, rtol=1e-6)
        np.testing.assert_allclose(loaded.residual_imag_pct, sample_result.residual_imag_pct, rtol=1e-6)

    def test_from_arrays(self):
        f = np.array([100, 1000, 10000], dtype=float)
        z_real = np.array([100, 90, 80], dtype=float)
        z_imag_neg = np.array([10, 20, 5], dtype=float)
        r = EISResult.from_arrays(3, f, z_real, z_imag_neg)
        assert r.npts == 3
        assert r.channel == 3
        np.testing.assert_array_equal(r.frequency, f)


# ── Circuit-fitting helpers ──────────────────────────────────────────────


class TestCircuitFitting:
    """Circuit-fitting helper coverage."""

    def test_circuit_models_registry(self):
        assert "simpleSalt" in CIRCUIT_MODELS
        assert "flexSalt" in CIRCUIT_MODELS
        assert "simpleSaltMembrane" in CIRCUIT_MODELS

    def test_circuit_model_has_required_keys(self):
        for name, model in CIRCUIT_MODELS.items():
            assert "circuit" in model, f"{name} missing 'circuit'"
            assert "bounds" in model, f"{name} missing 'bounds'"

    def test_z_to_sigma_basic(self):
        """σ = L / (t × w × R1) — and the deprecation is expected, not suppressed.

        ``z_to_sigma`` warns since P.20 and has zero production callers; it survives
        only as the parity oracle, so its own tests state the warning explicitly.
        """
        with pytest.warns(DeprecationWarning, match="z_to_sigma is deprecated"):
            sigma = z_to_sigma(L=0.2, t=0.175, w=0.2, R1=1000.0)
        expected = 0.2 / (0.175 * 0.2 * 1000.0)
        assert abs(sigma - expected) < 1e-10

    def test_z_to_sigma_zero_r1_returns_inf(self):
        """R1=0 → division by zero yields inf (numpy semantics)."""
        with pytest.warns(DeprecationWarning, match="z_to_sigma is deprecated"):
            sigma = z_to_sigma(0.2, 0.175, 0.2, 0.0)
        assert np.isinf(sigma)

    def test_extract_features_returns_dict(self):
        f = np.logspace(1, 5, 100)
        z_real = 50 + 100 / (1 + (2 * np.pi * f * 1e-5) ** 2)
        z_imag_neg = 100 * 2 * np.pi * f * 1e-5 / (1 + (2 * np.pi * f * 1e-5) ** 2)
        feats = extract_features(f, z_real, z_imag_neg)
        assert isinstance(feats, dict)
        assert "r0_guess" in feats
        assert "r1_guess" in feats

    def test_fit_result_sigma(self):
        fr = FitResult(
            model_name="simpleSalt",
            parameters={"R0": 10.0, "R1": 500.0},
            R0=10.0,
            R1=500.0,
            R0_guess=10.0,
            R1_guess=500.0,
            z_indices=[],
            success=True,
        )
        with pytest.warns(DeprecationWarning, match="FitResult.sigma is deprecated"):
            sigma = fr.sigma(L=0.2, t=0.175, w=0.2)
        expected = 0.2 / (0.175 * 0.2 * 500.0)
        assert abs(sigma - expected) < 1e-10

    def test_fit_result_failed(self):
        fr = FitResult(
            model_name="test",
            parameters={},
            R0=0.0,
            R1=0.0,
            R0_guess=0.0,
            R1_guess=0.0,
            z_indices=[],
            success=False,
            error_msg="convergence failure",
        )
        assert not fr.success
        assert "convergence" in fr.error_msg

    def test_predict_fit_curve_unknown_model_returns_none(self):
        """Unknown model name → None (no impedance backend needed)."""
        fr = FitResult(
            model_name="does_not_exist",
            parameters=np.array([1.0, 2.0]),
            R0=1.0, R1=2.0, R0_guess=1.0, R1_guess=2.0,
            z_indices=[0, 1], success=True,
        )
        assert predict_fit_curve(fr, np.logspace(0, 4, 10)) is None

    def test_predict_fit_curve_reconstructs_known_curve(self):
        """Helper rebuilds the same complex curve as a hand-built CustomCircuit."""
        pytest.importorskip("impedance")
        from impedance.models.circuits import CustomCircuit

        freq = np.logspace(0, 5, 25)
        params = [50.0, 1e-7, 0.8, 1000.0, 1e-10]
        cfg = CIRCUIT_MODELS["simpleSalt"]
        expected = CustomCircuit(cfg["circuit"], initial_guess=params).predict(freq)

        fr = FitResult(
            model_name="simpleSalt",
            parameters=np.array(params),
            R0=50.0, R1=1000.0, R0_guess=50.0, R1_guess=1000.0,
            z_indices=[0, 3], success=True,
        )
        got = predict_fit_curve(fr, freq)
        assert got is not None
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-9)
