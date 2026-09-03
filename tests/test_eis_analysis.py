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

    def test_the_retired_membrane_model_stays_out_of_the_registry(self):
        """Retired 2026-09-02: one fit-history row ever, ``success=0``, ``R1`` NULL.

        A registry entry is cheap to re-add and expensive to notice, so the absence is
        asserted rather than merely left. ``fit_circuit`` refuses an unknown model by
        name, which is the behaviour a re-add would silently remove.
        """
        assert "simpleSaltMembrane" not in CIRCUIT_MODELS

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


# ── Optimiser budget ─────────────────────────────────────────────────────

#: The real pathological spectrum, not a fixture of one.  Measurement 3840 of
#: ``20260825T154521Z_arrhenius_sweep`` (ch20, 60.1 C) is the one spectrum in 54 that
#: spends ~100,000 function evaluations and 348 s at 100 % CPU before raising anyway,
#: and it is the reason this budget exists.  Its ``usable_points`` mask is all-true
#: (34 of 34), so it isolates the optimiser from the finiteness path cleanly.
PATHOLOGICAL_SPECTRUM = Path(
    r"C:\Users\Osuji\softae_data\runs\20260825T154521Z_arrhenius_sweep"
    r"\eis\eis_ch20_T60_RH0.txt"
)


def _count_circuit_evaluations(monkeypatch) -> dict[str, int]:
    """Count every circuit evaluation the optimiser makes; return a live counter.

    ``impedance.models.circuits.fitting.wrapCircuit`` builds the residual function
    ``curve_fit`` is handed, so it is the one point every evaluation passes through and
    the honest place to measure the optimiser's spend. The count is not ``nfev``: it
    includes the finite-difference jacobian's calls, and runs ~5.2x the budget.
    """
    import impedance.models.circuits.fitting as impedance_fitting

    box = {"n": 0}
    original = impedance_fitting.wrapCircuit

    def counting(circuit, constants):
        inner = original(circuit, constants)

        def wrapped(*args, **kwargs):
            box["n"] += 1
            return inner(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(impedance_fitting, "wrapCircuit", counting)
    return box


def _config_overlay(monkeypatch, **eis_keys):
    """Replace keys in ``[eis]`` at the loader, leaving every other section alone.

    ``softae.config.loader.load`` is the one parse point ``_legacy_max_nfev`` reads,
    the same one ``eis_settings`` reads ``engine`` from, so patching here exercises the
    real resolution end to end.  Patching ``_legacy_max_nfev`` itself would test the
    test.  Every other section is carried through from the real file because the
    quality grader and the engine read their own.
    """
    from softae.config import loader

    overlaid = dict(loader.load())
    overlaid["eis"] = {**(overlaid.get("eis") or {}), **eis_keys}
    monkeypatch.setattr(loader, "load", lambda *a, **k: overlaid)


class TestFitCircuitOptimiserBudget:
    """The optimiser gets a bounded number of evaluations, and says when it runs out.

    The measured defect: ``fit_circuit`` passed no iteration budget, so it inherited
    impedance.py's ``maxfev = 1e5`` at ``ftol = 1e-13``.  Over the 54-spectrum
    ``20260825T154521Z_arrhenius_sweep`` corpus, 53 spectra fit in under 0.22 s and one
    — measurement 3840 — burned **348.56 s at 100 % CPU and then raised**, which is
    what the GUI's "hung" Arrhenius fit actually was.

    A cap is safe here for one specific reason, asserted below rather than assumed:
    ``curve_fit`` **raises** when the budget is exhausted and never returns a degraded
    best-so-far, so a cap either does not fire (bit-identical ``R1``) or fires and the
    fit fails as it was already going to.
    """

    @pytest.fixture
    def clean(self) -> EISResult:
        pytest.importorskip("impedance")
        f, Z = reference_spectrum()
        return as_eis_result(f, Z, channel=22)

    # -- the cap reaches the optimiser, on both call sites -----------------

    def test_the_cap_reaches_the_bounded_call_site(self, clean):
        """``simpleSalt`` carries ``bounds``, so it takes the ``bounds=`` branch.

        Pinned against the *same* spectrum fitting fine at a generous cap, so this
        asserts the budget did the failing and not the data.
        """
        assert CIRCUIT_MODELS["simpleSalt"]["bounds"] is not None, (
            "premise of this test: simpleSalt must take the bounded branch"
        )
        starved = fit_circuit(clean, "simpleSalt", max_nfev=2)
        roomy = fit_circuit(clean, "simpleSalt", max_nfev=5000)

        assert roomy.success, "the clean synthetic must fit, or this proves nothing"
        assert not starved.success
        assert starved.failure_kind == "budget_exhausted"

    def test_the_cap_reaches_the_unbounded_call_site(self, clean):
        """``flexSalt`` has ``bounds = None``, so it takes the other branch.

        Both branches are exercised because the correct kwarg is not obvious: impedance
        fills in default bounds when given none, so *both* reach ``curve_fit`` bounded
        and run ``trf``.  A ``max_nfev=`` spelling works on neither.
        """
        assert CIRCUIT_MODELS["flexSalt"]["bounds"] is None, (
            "premise of this test: flexSalt must take the unbounded branch"
        )
        starved = fit_circuit(clean, "flexSalt", max_nfev=2)
        roomy = fit_circuit(clean, "flexSalt", max_nfev=5000)

        assert roomy.success, "the clean synthetic must fit, or this proves nothing"
        assert not starved.success
        assert starved.failure_kind == "budget_exhausted"

    def test_a_cap_that_does_not_fire_changes_nothing(self, clean):
        """The no-op property the safety argument rests on, asserted not assumed.

        If a cap could return a *degraded* fit rather than raising, every number on this
        path would move when the default landed.  Bit-identical is the claim, so
        bit-identical is the assertion — not ``approx``.
        """
        for name in ("simpleSalt", "flexSalt"):
            capped = fit_circuit(clean, name, max_nfev=5000)
            uncapped = fit_circuit(clean, name, max_nfev=0)
            assert capped.success and uncapped.success
            assert capped.R1 == uncapped.R1, name
            assert capped.R0 == uncapped.R0, name

    # -- where the number comes from --------------------------------------

    def test_the_default_budget_comes_from_config(self, clean, monkeypatch):
        """``[eis] legacy_max_nfev`` governs a call that passes no ``max_nfev``.

        Driven through the loader rather than through the argument, because the argument
        path is already covered above and the thing at risk is the *wiring* — a fitter
        that silently ignored the key would pass every argument-level test.
        """
        _config_overlay(monkeypatch, legacy_max_nfev=2)
        starved = fit_circuit(clean, "simpleSalt")

        assert not starved.success
        assert starved.failure_kind == "budget_exhausted"
        assert "budget of 2 function evaluations" in starved.error_msg

    def test_the_shipped_default_is_2000(self):
        """The shipped file, read as the operator's rig reads it.

        2000 is not arbitrary: it is ~23x the worst *converging* fit measured on the
        corpus (65 nfev on simpleSalt, 87 on flexSalt) and 2 % of what the pathological
        spectrum wants, and it is the same number as ``[eis.pregate] max_nfev`` so the
        codebase carries one answer rather than two.
        """
        from softae.analysis.circuit_fitting import (
            DEFAULT_LEGACY_MAX_NFEV,
            _legacy_max_nfev,
        )
        from softae.config import loader

        assert DEFAULT_LEGACY_MAX_NFEV == 2000
        assert _legacy_max_nfev() == 2000
        eis_cfg = loader.load().get("eis", {}) or {}
        assert eis_cfg.get("legacy_max_nfev") == 2000, (
            "the shipped softae_config.toml must carry the key, not rely on the fallback"
        )
        assert eis_cfg.get("pregate", {}).get("max_nfev") == DEFAULT_LEGACY_MAX_NFEV, (
            "one number for 'how long may a fit run', not two"
        )

    def test_a_caller_argument_beats_config(self, clean, monkeypatch):
        """Precedence, in the direction :func:`_min_fit_points` already set."""
        _config_overlay(monkeypatch, legacy_max_nfev=2)

        assert not fit_circuit(clean, "simpleSalt").success, (
            "premise: config alone must starve this fit"
        )
        assert fit_circuit(clean, "simpleSalt", max_nfev=5000).success

    @pytest.mark.parametrize("off", [0, -1, -2000])
    def test_zero_or_negative_means_no_cap(self, clean, monkeypatch, off):
        """"Off" omits the kwarg entirely rather than passing a large number.

        The distinction matters: passing a large ``maxfev`` is still *our* number, and
        the documented meaning of 0 is impedance.py's own ``1e5`` — the exact behaviour
        that shipped before this key existed.
        """
        from softae.analysis.circuit_fitting import _legacy_max_nfev

        assert _legacy_max_nfev(off) is None
        _config_overlay(monkeypatch, legacy_max_nfev=off)
        assert _legacy_max_nfev() is None
        assert fit_circuit(clean, "simpleSalt").success

    # -- the three refusals must not read alike ----------------------------

    def test_budget_and_dropped_point_refusals_are_distinct(self, clean):
        """The operator read one of these as "the EIS gates are enabled".

        Both surface as red text in the GUI's Error column, so they are pinned apart
        both structurally (``failure_kind``) and in wording — and the budget message is
        required to say in words that no gate was involved, because that is the
        inference that was actually drawn.
        """
        budget = fit_circuit(clean, "simpleSalt", max_nfev=2)
        remnant = fit_circuit(
            _poke_nonfinite(clean, slice(4, None)), "simpleSalt", min_points=8
        )

        assert budget.failure_kind == "budget_exhausted"
        assert remnant.failure_kind == "too_few_points"
        assert budget.failure_kind != remnant.failure_kind

        assert "budget" in budget.error_msg and "NOT a gate" in budget.error_msg
        assert "no gate rejected this spectrum" in budget.error_msg
        assert "usable" not in budget.error_msg, (
            "the budget refusal must not borrow the remnant refusal's vocabulary"
        )
        assert "budget" not in remnant.error_msg
        assert "usable" in remnant.error_msg and "need 8" in remnant.error_msg

    def test_a_budget_refusal_reports_no_r1_rather_than_a_bad_one(self, clean):
        """σ = K/R consumes a NaN loudly and a wrong float silently."""
        got = fit_circuit(clean, "simpleSalt", max_nfev=2)
        assert not got.success
        assert np.isnan(got.R1) and np.isnan(got.R0)

    def test_a_successful_fit_carries_no_failure_kind(self, clean):
        """"Did not fail" must not be spelled with the same token as "failed somehow"."""
        got = fit_circuit(clean, "simpleSalt")
        assert got.success
        assert got.failure_kind == ""

    def test_an_unrelated_fit_error_is_not_labelled_a_budget_failure(self):
        """The third refusal, and the one the other two must not absorb.

        This test was briefly retired with ``simpleSaltMembrane`` on 2026-09-02, on the
        belief that no surviving model produces a natural fit error and that re-vehicling
        it would mean patching ``CustomCircuit`` to raise.  That belief was wrong, and
        the vehicle below is why: ``simpleSalt`` carries its own ``bounds``, whose ``R1``
        floor is 100 Ω, while ``extract_features`` takes ``R1``'s initial guess *from the
        spectrum*.  A cell conductive enough to put the bulk arc under 100 Ω therefore
        hands ``curve_fit`` an ``x0`` outside its own bounds, and scipy raises
        ``ValueError("Initial guess is outside of provided bounds")`` from
        ``_lsq/least_squares.py`` before the optimiser spends a single evaluation.
        Nothing is patched, monkeypatched or injected here — the model's shipped bounds
        meet a spectrum, which is a reachable contract and not a fabricated input.

        What is under test is the ``else`` of the classifier in ``fit_circuit``'s
        ``except``: a non-budget exception must be labelled ``fit_error``.  Both
        alternatives are excluded by construction rather than by assumption — the budget
        is generous (so ``budget_exhausted`` cannot be right) and the spectrum is
        all-finite (so ``too_few_points`` cannot be either) — and both are asserted, so
        a classifier that stopped discriminating cannot pass this quietly.
        """
        lower_bound_R1 = CIRCUIT_MODELS["simpleSalt"]["bounds"][0][3]
        assert lower_bound_R1 == 100.0, (
            "premise: simpleSalt's own R1 floor is what this spectrum falls under"
        )

        # The reference topology with every impedance 1000x lower — the same cell, 1000x
        # more conductive. Physical, and not a scaling trick applied to Z after the fact.
        f, Z = reference_spectrum(
            R_series=0.05, R_bulk=50.0, C_par=3.5e-7, Q=1.0e-4
        )
        conductive = as_eis_result(f, Z, channel=22)

        assert usable_points(
            conductive.frequency, conductive.z_real, conductive.z_imag_neg
        ).all(), "premise: all-finite, so the too_few_points refusal cannot be the path"
        guess = extract_features(
            conductive.frequency, conductive.z_real, conductive.z_imag_neg
        )["r1_guess"]
        assert guess < lower_bound_R1, (
            f"premise: the extracted R1 guess ({guess:.4g}) must fall under the model's "
            "own floor, or scipy has nothing to object to"
        )

        got = fit_circuit(conductive, "simpleSalt", max_nfev=5000)

        assert not got.success
        assert got.failure_kind == "fit_error"
        assert got.failure_kind != "budget_exhausted", (
            "an exception that is not budget exhaustion must not be filed as one"
        )
        assert got.failure_kind != "too_few_points"
        assert got.n_points_dropped == 0
        assert np.isnan(got.R1)

        # The refusals are pinned apart in wording as well as in token, because the GUI
        # shows only the wording. A fit error must not borrow the budget refusal's prose.
        assert "Initial guess is outside of provided bounds" in got.error_msg
        assert "budget" not in got.error_msg
        assert "NOT a gate" not in got.error_msg

        # And the same fitter on the ordinary spectrum succeeds, so this is the data
        # meeting the bounds and not the machinery being broken.
        assert fit_circuit(
            as_eis_result(*reference_spectrum(), channel=22),
            "simpleSalt",
            max_nfev=5000,
        ).success

    # -- the real manifold, not a fixture of it ----------------------------

    def test_the_real_pathological_spectrum_is_capped(self, monkeypatch):
        """Measurement 3840 itself, because a synthetic would not prove the plumbing.

        Asserted under a *small* cap: the point is that the budget fires on this real
        spectrum through the real ``EISResult.load`` path, not to re-run the 7 s or
        348 s cases in the suite.  The spectrum's own mask is all-finite, so this
        isolates the optimiser from :func:`usable_points` entirely.

        **The work done is counted, not inferred from the refusal.**  An earlier draft
        of this test asserted only the message, and a mutation that deleted the kwarg
        from the ``model.fit`` call left it green — because the fit still raised the
        same wording, 379 s later.  That is the 3.1(d) shape exactly: an assertion on
        the report rather than on the thing being reported.  ``wrapCircuit`` is the
        single point every circuit evaluation passes through, so counting there
        measures the optimiser's spend directly.

        Calibration: this spectrum costs ~480 evaluations at a cap of 100 and ~10,300
        at 2000 (~5.2 per unit of budget, the jacobian's finite differences), against
        ~5e5 uncapped.  5000 sits a decade below the uncapped cost and an order above
        the capped one, so neither noise nor a scipy step-size change can move it.
        """
        pytest.importorskip("impedance")
        if not PATHOLOGICAL_SPECTRUM.exists():
            pytest.skip(
                f"real corpus absent: {PATHOLOGICAL_SPECTRUM}. This test is the only "
                "one here that exercises the production manifold; a green suite "
                "without it has NOT checked that the cap fires on real data."
            )

        real = EISResult.load(PATHOLOGICAL_SPECTRUM)
        assert usable_points(
            real.frequency, real.z_real, real.z_imag_neg
        ).all(), "premise: 3840 is all-finite, so only the budget can refuse it"

        n_eval = _count_circuit_evaluations(monkeypatch)
        got = fit_circuit(real, "simpleSalt", max_nfev=100)

        assert not got.success
        assert got.failure_kind == "budget_exhausted"
        assert got.n_points_dropped == 0
        assert "budget of 100 function evaluations" in got.error_msg
        assert np.isnan(got.R1)
        assert 0 < n_eval["n"] < 5000, (
            f"the cap did not bound the optimiser: {n_eval['n']} circuit evaluations "
            "for a budget of 100. Uncapped this spectrum costs ~5e5 and 348 s."
        )
