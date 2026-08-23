"""E4 — Kramers–Kronig stationarity truncation (§3.6) and plateau-in-band (§3.7).

The K–K test is the one gate in the framework that assumes **no circuit model** — only
causality, linearity and stationarity. That makes its separation from the topology triad
worth asserting explicitly: a pure series RC and a dispersive dielectric are both
perfectly causal, so K–K must *pass* them and let the triad reject them. A K–K test that
also caught those would be catching them for the wrong reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis.envelope import instrument_envelope
from softae.analysis.eis.gates import gate_kk_truncation, gate_plateau_in_band
from softae.analysis.eis.kk import (
    DEFAULT_KK_MAX_TRUNCATE_FRAC,
    lin_kk,
    low_frequency_run,
)
from softae.analysis.eis.policy import build_context
from softae.analysis.eis.settings import GateSettings

from .eis_synthetic import (
    dispersive_dielectric,
    log_frequencies,
    pure_series_rc,
    reference_spectrum,
)


def ctx(**gate_kw):
    return build_context(envelope=instrument_envelope(),
                         gates=GateSettings(**gate_kw), cell=None)


def drifted(n_points: int, factor_hi: float = 1.6, factor_lo: float = 1.15):
    """A spectrum whose *lowest* frequencies have drifted during acquisition."""
    f, Z = reference_spectrum()
    Zd = np.array(Z, dtype=complex)
    lo = np.argsort(f)[:n_points]
    Zd[lo] *= np.linspace(factor_hi, factor_lo, n_points)
    return f, Zd


# ── The ladder ───────────────────────────────────────────────────────────────

class TestLinKK:
    def test_a_clean_spectrum_is_kk_compliant(self):
        f, Z = reference_spectrum()
        r = lin_kk(f, Z, blocking=True)
        assert r.ok, r.error
        assert r.max_resid_pct < 1.0
        assert r.M > 0

    def test_the_numpy2_namespace_patch_is_what_makes_this_run_at_all(self):
        """``impedance`` 1.7.1 formats numpy scalars into a string it then ``eval``s.

        NumPy 2 renders those as ``np.float64(...)``, and that namespace has no ``np``,
        so every call raised ``NameError``. Unpatched, the gate would degrade to a flag
        on every spectrum and the pipeline would appear to run a stationarity test it
        had never once executed — which is worse than crashing.
        """
        from impedance import validation

        assert "np" in validation.circuit_elements

    def test_residuals_index_the_callers_own_points_not_the_sorted_ones(self):
        """The rig sweeps high→low; linKK needs ascending. The un-sort must be exact."""
        f, Z = drifted(3)
        r = lin_kk(f, Z, blocking=True)
        assert r.ok
        assert f[0] > f[-1], "fixture should be a descending sweep"
        # The drift was injected at the lowest frequencies = the tail of the array.
        lo = np.argsort(f)[:3]
        assert r.resid_pct[lo].mean() > r.resid_pct[np.argsort(f)[-10:]].mean()

    def test_too_few_points_is_reported_not_raised(self):
        r = lin_kk(np.array([1.0, 2.0]), np.array([1 + 0j, 2 + 0j]))
        assert not r.ok
        assert "too few" in r.error

    def test_it_never_raises_so_one_bad_spectrum_cannot_end_a_batch(self):
        f = log_frequencies()
        r = lin_kk(f, np.full(f.size, np.nan, dtype=complex))
        assert isinstance(r.ok, bool)   # verdict either way, but no exception


class TestLowFrequencyRun:
    def test_it_finds_the_run_at_the_low_frequency_end_of_a_descending_sweep(self):
        """A helper assuming ascending order would truncate the *high*-frequency end,
        deleting the arc that carries ``R_bulk`` while reporting it removed drift."""
        f = log_frequencies()            # descending
        bad = np.zeros(f.size, bool)
        bad[-3:] = True                  # tail of array = lowest frequencies
        assert np.array_equal(np.where(low_frequency_run(f, bad))[0],
                              np.array([f.size - 3, f.size - 2, f.size - 1]))

    def test_it_works_on_an_ascending_sweep_too(self):
        f = log_frequencies(descending=False)
        bad = np.zeros(f.size, bool)
        bad[:3] = True                   # head of array = lowest frequencies
        assert low_frequency_run(f, bad).sum() == 3

    def test_an_isolated_mid_band_failure_is_never_removed(self):
        f = log_frequencies()
        bad = np.zeros(f.size, bool)
        bad[5] = True
        assert low_frequency_run(f, bad).sum() == 0

    def test_the_run_stops_at_the_first_passing_point(self):
        f = log_frequencies()
        order = np.argsort(f)
        bad = np.zeros(f.size, bool)
        bad[order[0]] = bad[order[1]] = True
        bad[order[3]] = True             # detached — must not be swept up
        assert low_frequency_run(f, bad).sum() == 2

    def test_nothing_failing_removes_nothing(self):
        f = log_frequencies()
        assert low_frequency_run(f, np.zeros(f.size, bool)).sum() == 0


# ── The gate ─────────────────────────────────────────────────────────────────

class TestKKTruncationGate:
    def test_a_clean_spectrum_keeps_every_point(self):
        f, Z = reference_spectrum()
        r = gate_kk_truncation(f, Z, ctx())
        assert r.passed and r.n_dropped == 0

    def test_a_drifting_tail_is_truncated_from_the_low_frequency_end(self):
        f, Z = drifted(3, 1.25, 1.08)
        r = gate_kk_truncation(f, Z, ctx())
        assert r.n_dropped > 0
        dropped = np.where(~np.asarray(r.mask, dtype=bool))[0]
        # every dropped point must be below every surviving one in frequency
        kept = np.where(np.asarray(r.mask, dtype=bool))[0]
        assert f[dropped].max() <= f[kept].min()

    def test_truncating_past_the_bound_rejects_instead_of_cutting(self):
        """The ladder fit is global, so a small drift makes almost every point fail.

        Unbounded truncation would delete the spectrum while reporting it had tidied a
        tail. §3.6's licence to cut rests on the cut staying clear of the arc.
        """
        f, Z = reference_spectrum()
        r = gate_kk_truncation(f, Z, ctx(kk_max_truncate_frac=0.0, kk_resid_pct=0.0))
        assert r.severity == "block_spectrum"
        assert not r.passed
        assert "non-stationary across the band" in r.detail

    def test_isolated_failures_are_flagged_and_kept(self):
        f, Z = drifted(3, 1.25, 1.08)
        r = gate_kk_truncation(f, Z, ctx())
        if r.metrics.get("kk_isolated", 0):
            assert "not removed" in r.detail

    def test_a_ladder_that_will_not_fit_passes_with_a_note(self):
        """A test that could not run is an absence of evidence, not a failure —
        otherwise a numerical problem would start rejecting good spectra."""
        f = log_frequencies()
        r = gate_kk_truncation(f[:2], np.array([1 + 0j, 2 + 0j]), ctx())
        assert r.passed
        assert r.severity == "flag"
        assert "did not run" in r.detail

    def test_the_bound_default_is_half_the_band(self):
        assert DEFAULT_KK_MAX_TRUNCATE_FRAC == 0.5

    @pytest.mark.parametrize("blocking", [False, True])
    def test_kk_truncation_cell_blocking_reaches_the_ladder(self, monkeypatch,
                                                            blocking):
        """``build_context`` stores the flag at ``ctx["cell"]["blocking"]``.

        A gate reading a *top-level* ``ctx["blocking"]`` finds nothing and silently
        takes its default — so a non-blocking cell would be K–K tested as blocking
        with no error anywhere. The ``ctx()`` helper above never passes ``blocking``,
        which is why that read went unnoticed. Delegating to the real ladder rather
        than stubbing it keeps the gate's behaviour intact, so the only thing this
        test can be measuring is the kwarg.
        """
        import softae.analysis.eis.kk as kk_module

        seen: list[bool] = []
        real = kk_module.lin_kk

        def spy(f, Z, **kw):
            seen.append(kw["blocking"])
            return real(f, Z, **kw)

        monkeypatch.setattr(kk_module, "lin_kk", spy)

        f, Z = reference_spectrum()
        gate_kk_truncation(f, Z, build_context(envelope=instrument_envelope(),
                                               gates=GateSettings(), cell=None,
                                               blocking=blocking))
        assert seen == [blocking]


class TestKKIsModelFree:
    """K–K must pass anything causal, and leave topology to the topology gates.

    A pure series RC and a dispersive dielectric are both perfectly causal linear
    systems. They are pathologies of the *model*, not of stationarity, and §8.3 requires
    each pathology to be caught by its intended gate **and no other**.
    """

    @pytest.mark.parametrize("name,gen", [("series_rc", pure_series_rc),
                                          ("dispersive", dispersive_dielectric)])
    def test_a_causal_pathology_passes_the_stationarity_test(self, name, gen):
        f, Z = gen()
        r = gate_kk_truncation(f, Z, ctx())
        assert r.passed, f"{name} is causal — K–K must not be what rejects it"


class TestPlateauInBand:
    def test_a_clean_spectrum_has_a_plateau(self):
        f, Z = reference_spectrum()
        r = gate_plateau_in_band(f, Z, ctx())
        assert r.passed
        assert r.metrics["plateau_decades"] >= 0.5

    def test_the_plateau_median_beats_the_model_free_estimate_where_it_degrades(self):
        """Why the gate is self-referential rather than keyed to ``model_free_r_bulk``.

        That estimator is documented as ~41 % low at ``R_bulk ≈ 50 kΩ`` because the
        plateau is being squeezed. A gate comparing points against it reported "no
        plateau" on a spectrum with nearly a decade of one.
        """
        from softae.analysis.eis.admittance import model_free_r_bulk

        f, Z = reference_spectrum()          # R_series + R_bulk = 50 + 5e4
        truth = 5.005e4
        r = gate_plateau_in_band(f, Z, ctx())

        plateau_err = abs(r.metrics["plateau_R_ohm"] - truth) / truth
        model_free_err = abs(model_free_r_bulk(Z) - truth) / truth
        assert plateau_err < model_free_err

    def test_a_narrow_plateau_is_flagged_not_rejected(self):
        f, Z = reference_spectrum()
        r = gate_plateau_in_band(f, Z, ctx(plateau_min_decades=5.0))
        assert not r.passed
        assert r.severity == "flag"
        assert "extrapolated" in r.detail

    def test_a_monotonic_tail_with_no_flat_region_blocks_the_spectrum(self):
        """Zero overlap escalates: any R_bulk would be pure extrapolation.

        The realistic shape of "no plateau" — the bulk resistance pushed so far above
        the band that the blocking tail dominates every point, so ``Re Z`` falls
        monotonically and never flattens.
        """
        f, Z = reference_spectrum(R_bulk=1.0e9)
        r = gate_plateau_in_band(f, Z, ctx())
        assert r.severity == "block_spectrum"
        assert not r.passed

    def test_a_purely_reactive_response_is_evidence_of_absence_not_missing_data(self):
        """Re Z ≤ 0 everywhere is not "unmeasurable" — it is a measured lack of any
        resistive component, and inventing an R_bulk from it is the failure."""
        f = log_frequencies()
        Z = 1.0 / (1j * 2 * np.pi * f * 1e-9)      # pure capacitor: Re Z == 0
        r = gate_plateau_in_band(f, Z, ctx())
        assert r.severity == "block_spectrum"
        assert "purely reactive" in r.detail

    def test_genuinely_unmeasurable_data_only_flags(self):
        f = log_frequencies()
        r = gate_plateau_in_band(f, np.full(f.size, np.nan, dtype=complex), ctx())
        assert r.severity == "flag"
        assert r.passed

    def test_it_removes_no_points(self):
        """§3.7 is a verdict on the spectrum, never a reason to drop data."""
        f, Z = reference_spectrum()
        assert gate_plateau_in_band(f, Z, ctx()).n_dropped == 0


class TestExecutionOrder:
    def test_both_gates_run_after_the_fixture_correction(self):
        """§6: K–K is step 6 and plateau is step 9, both downstream of correction at 4."""
        from softae.analysis.eis.gates import (
            FRONT1_POST_CORRECTION,
            FRONT1_PRE_CORRECTION,
        )

        assert gate_kk_truncation in FRONT1_POST_CORRECTION
        assert gate_plateau_in_band in FRONT1_POST_CORRECTION
        assert gate_kk_truncation not in FRONT1_PRE_CORRECTION

    def test_kk_runs_before_the_topology_triad(self):
        """Step 6 before step 8: the triad is specified to see truncated data."""
        from softae.analysis.eis.gates import FRONT1_POST_CORRECTION, TOPOLOGY_TRIAD

        seq = list(FRONT1_POST_CORRECTION)
        assert seq.index(gate_kk_truncation) < min(seq.index(g) for g in TOPOLOGY_TRIAD)

    def test_plateau_runs_after_the_triad(self):
        from softae.analysis.eis.gates import FRONT1_POST_CORRECTION, TOPOLOGY_TRIAD

        seq = list(FRONT1_POST_CORRECTION)
        assert seq.index(gate_plateau_in_band) > max(seq.index(g)
                                                     for g in TOPOLOGY_TRIAD)
