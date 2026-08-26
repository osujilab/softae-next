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
    fit_circuit,
    predict_fit_curve,
    usable_points,
    z_to_sigma,
)
from tests.eis_synthetic import as_eis_result, reference_spectrum


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


# ── Non-finite points must not fail the whole spectrum ───────────────────


def _poke_nonfinite(result: EISResult, idx, value=np.nan) -> EISResult:
    """Return a copy of *result* with ``Z`` non-finite at *idx*.

    Both components go together because the instrument writes them together: a
    range-limited point on this rig produces a NaN pair, not half a number.
    """
    z_real = np.array(result.z_real, dtype=float)
    z_imag_neg = np.array(result.z_imag_neg, dtype=float)
    z_real[idx] = value
    z_imag_neg[idx] = value
    return EISResult.from_arrays(
        channel=result.channel, f=np.array(result.frequency, dtype=float),
        z_real=z_real, z_imag_neg=z_imag_neg,
    )


class TestFitCircuitFiniteness:
    """A spectrum is fitted on its usable points, not failed whole.

    The measured defect: ``curve_fit`` calls ``asarray_chkfinite`` on its ordinate, so
    one non-finite point failed the entire spectrum. On the ``probe-3ch-v3`` set
    ``ch22_003`` failed with **52 good points of 53**, identically to its sibling with
    37 of 53 — and the pair were read as an unclosed reference electrode for two days,
    because a spectrum that cannot be fitted looks like a spectrum that is wrong.
    """

    @pytest.fixture
    def clean(self) -> EISResult:
        pytest.importorskip("impedance")
        f, Z = reference_spectrum()
        return as_eis_result(f, Z, channel=22)

    def test_one_nonfinite_point_among_many_good_ones_still_fits(self, clean):
        """The ch22_003 shape: 1 bad point of many must not cost the other 52.

        Pinned against the all-finite fit of the same spectrum rather than against a
        literal, so this asserts *the same measurement was recovered* and not merely
        that something converged.
        """
        baseline = fit_circuit(clean)
        assert baseline.success, "the clean synthetic must fit, or this proves nothing"

        # Index 0 is the top of the band — where the real non-finite run sits, being a
        # high-frequency range/phase-resolution limit rather than scattered noise.
        holed = _poke_nonfinite(clean, 0)
        got = fit_circuit(holed)

        assert got.success, f"one NaN point failed the spectrum: {got.error_msg}"
        assert got.R1 == pytest.approx(baseline.R1, rel=0.05)

    def test_dropped_points_are_counted_on_the_result(self, clean):
        """R17: nothing is dropped silently — the count travels with the fit."""
        got = fit_circuit(_poke_nonfinite(clean, [0, 1, 2]))
        assert got.success
        assert got.n_points_dropped == 3
        assert got.n_points_used == clean.npts - 3

    def test_infinities_are_masked_as_well_as_nans(self, clean):
        """``asarray_chkfinite`` rejects both, so both must be masked."""
        got = fit_circuit(_poke_nonfinite(clean, 0, value=np.inf))
        assert got.success, got.error_msg
        assert got.n_points_dropped == 1

    def test_initial_guesses_are_finite_when_a_point_is_nan(self, clean):
        """The mask has to reach ``extract_features``, not just the optimiser.

        ``extract_features`` takes ``np.argmin``, which returns the index *of* the NaN,
        so an unmasked guess extraction yields ``r0_guess = nan`` and the fit fails on
        its starting point even after the data is clean. Measured on ch22_003.
        """
        got = fit_circuit(_poke_nonfinite(clean, 0))
        assert np.isfinite(got.R0_guess)
        assert np.isfinite(got.R1_guess)

    def test_nonpositive_frequency_is_dropped(self, clean):
        """``curve_fit`` checks the ordinate only, so a bad abscissa needs us."""
        f = np.array(clean.frequency, dtype=float)
        f[0] = 0.0
        holed = EISResult.from_arrays(
            channel=clean.channel, f=f,
            z_real=np.array(clean.z_real), z_imag_neg=np.array(clean.z_imag_neg),
        )
        got = fit_circuit(holed)
        assert got.n_points_dropped == 1
        assert got.n_points_used == clean.npts - 1

    def test_a_clean_spectrum_is_unchanged_and_reports_no_drops(self, clean):
        """The fix must be inert on data that already fitted."""
        got = fit_circuit(clean)
        assert got.success
        assert got.n_points_dropped == 0
        assert got.n_points_used == clean.npts

    def test_z_fit_stays_aligned_with_the_full_spectrum(self, clean):
        """``z_fit`` is predicted on the full grid even when points were dropped.

        ``compute_fit_quality`` indexes ``z_fit`` against ``eis_result`` and
        ``tab_analysis`` overlays it on the measured trace; a ``z_fit`` shorter than the
        spectrum would shift every point after the drop rather than fail.
        """
        got = fit_circuit(_poke_nonfinite(clean, 5))
        assert got.success
        assert got.z_fit is not None
        assert got.z_fit.size == clean.npts

    def test_quality_metrics_survive_a_dropped_point(self, clean):
        """A masked fit is still graded — a fit nobody graded is the F11 failure."""
        got = fit_circuit(_poke_nonfinite(clean, 0))
        assert got.quality.get("r_squared") is not None
        assert np.isfinite(got.quality["r_squared"])

    def test_refuses_rather_than_fitting_a_remnant(self, clean):
        """Below the floor the refusal is explicit, not a bad number.

        The counter-failure to the one being fixed: a 5-parameter circuit fitted to 4
        surviving points reports an ``R1`` that ``σ = K/R`` consumes as happily as a
        real one.
        """
        got = fit_circuit(_poke_nonfinite(clean, slice(4, None)), min_points=8)

        assert not got.success
        assert got.n_points_used == 4
        assert "usable" in got.error_msg and "need 8" in got.error_msg
        assert "infs or NaNs" not in got.error_msg, (
            "the refusal must name the real reason, not leak curve_fit's message"
        )
        assert np.isnan(got.R1)

    def test_the_floor_does_not_revoke_short_all_finite_sweeps(self):
        """A short *clean* sweep is an operator's choice of preset, not a remnant.

        The floor is checked only when the mask actually removed something, so this
        change cannot quietly withdraw a capability the legacy path already had.
        """
        pytest.importorskip("impedance")
        f, Z = reference_spectrum(np.logspace(5, 2, 6)[::-1])
        got = fit_circuit(as_eis_result(f, Z), min_points=8)
        assert got.n_points_dropped == 0
        assert "usable" not in got.error_msg


class TestUsablePointsMatchesTheGate:
    """One definition of "usable point", not two.

    Two different answers to that question in one codebase is the defect one layer up
    from the one this fixes, so the legacy mask is pinned against
    :func:`~softae.analysis.eis.gates.gate_finiteness` directly.
    """

    def test_agrees_with_gate_finiteness_point_for_point(self):
        from softae.analysis.eis.gates import gate_finiteness

        f, Z = reference_spectrum()
        f, Z = np.array(f), np.array(Z)
        Z[0] = np.nan
        Z[3] = complex(np.inf, 0.0)
        Z[7] = complex(1.0, np.nan)      # half-finite: not half a point
        f[11] = -1.0
        f[12] = np.nan

        theirs = np.asarray(gate_finiteness(f, Z, {}).mask, dtype=bool)
        ours = usable_points(f, Z.real, -Z.imag)
        np.testing.assert_array_equal(ours, theirs)

    def test_differs_only_by_the_duplicate_frequency_rule(self):
        """The one deliberate divergence, asserted rather than left to drift.

        ``gate_finiteness`` also drops repeated frequencies — for the Kramers–Kronig
        basis and the topology triad's ``polyfit``, neither of which runs on the legacy
        path. Dropping them here would remove points from spectra that fit correctly
        today, for a reason that does not apply to them.
        """
        from softae.analysis.eis.gates import gate_finiteness

        f, Z = reference_spectrum()
        f, Z = np.array(f), np.array(Z)
        f[5] = f[4]                       # a duplicate, everything else finite

        theirs = np.asarray(gate_finiteness(f, Z, {}).mask, dtype=bool)
        ours = usable_points(f, Z.real, -Z.imag)

        assert not theirs[5], "gate_finiteness is expected to drop the duplicate"
        assert ours.all(), "the legacy fit mask deliberately keeps it"
        np.testing.assert_array_equal(np.delete(ours, 5), np.delete(theirs, 5))
