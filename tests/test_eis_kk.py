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
    DEFAULT_KK_C,
    DEFAULT_KK_MAX_M,
    DEFAULT_KK_MAX_TRUNCATE_FRAC,
    DEFAULT_KK_MU_FLOOR,
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

#: The sweep this rig actually runs: 53 log points, 1.351 Hz – 200 kHz, descending.
#:
#: ``eis_synthetic``'s defaults (``R_bulk = 5e4``, ``f_lo = 20``, 41 points) describe a
#: cell one to four decades away, and every K–K test in this file used to run there.
#: That is not a harmless convenience: measured, one decade of ``R_bulk`` is the whole
#: difference between "passes, drops nothing" and ``block_spectrum``, so a fixture
#: parked at 5e4 asserts the gate's behaviour in the one regime the rig never visits.
RIG_FREQ = np.logspace(np.log10(2.0e5), np.log10(1.351), 53)

#: The four ``R_bulk`` decades the rig and its fixtures span, low end to high.
RIG_R_BULK = (5.0e4, 5.0e6, 5.0e7, 1.0e8)


def ctx(**gate_kw):
    return build_context(envelope=instrument_envelope(),
                         gates=GateSettings(**gate_kw), cell=None)


def drifted(n_points: int, factor_hi: float = 1.6, factor_lo: float = 1.15,
            *, freq: np.ndarray | None = None, R_bulk: float = 5.0e4):
    """A spectrum whose *lowest* frequencies have drifted during acquisition."""
    f, Z = reference_spectrum(freq, R_bulk=R_bulk)
    Zd = np.array(Z, dtype=complex)
    lo = np.argsort(f)[:n_points]
    Zd[lo] *= np.linspace(factor_hi, factor_lo, n_points)
    return f, Zd


# ── The ladder ───────────────────────────────────────────────────────────────

def mock_spectrum(freq: np.ndarray, *, r1_ohm: float = 4.8e7,
                  noise_rel: float = 0.005, seed: int = 0):
    """``R0-CPE0-p(R1,C0)`` with known injected noise — the validation harness's own.

    Deliberately the harness generator rather than ``reference_spectrum``: it is the
    population the shadow-run statistics are computed on, and its noise amplitude is a
    *declared* number, which is what makes "did the ladder recover the noise or report
    its own order?" an answerable question rather than a judgement call.
    """
    from softae.tools.eis_validate_mock import synthesize

    cols = synthesize(freq, r1_ohm=r1_ohm, noise_rel=noise_rel, seed=seed)
    return freq, cols[:, 3] - 1j * cols[:, 4]


class TestLinKK:
    @pytest.mark.parametrize("R_bulk", RIG_R_BULK)
    def test_a_clean_spectrum_is_kk_compliant(self, R_bulk):
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=R_bulk)
        r = lin_kk(f, Z, blocking=True)
        assert r.ok, r.error
        assert r.max_resid_pct < 1.0
        assert r.M > 0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_ladder_recovers_the_injected_noise_not_its_own_order(self, seed):
        """T3. The residual on a mock spectrum must measure the *sample*.

        0.5 % rms is injected on purpose, so a well-chosen ladder order should land
        near it and nowhere near 1 %. Under first-crossing order selection at
        ``kk_c = 0.85`` this same spectrum returns a **2.48 % median at M = 5** — five
        times its own noise, and systematic rather than random: a low-frequency lobe
        rising from 0.3 % at 200 kHz to ~6 % at 1.35 Hz. That shape is precisely what
        :func:`gate_kk_truncation` is empowered to cut, so the number below is not
        cosmetic. Measured after the fix: 0.24 – 0.31 % across seeds.
        """
        f, Z = mock_spectrum(RIG_FREQ, seed=seed)
        r = lin_kk(f, Z, blocking=True)
        assert r.ok, r.error
        finite = r.resid_pct[np.isfinite(r.resid_pct)]
        assert float(np.median(finite)) < 1.0

    def test_the_walk_does_not_stop_at_the_first_mu_crossing(self):
        """The durable half of the fix, isolated from the constant that hides it.

        ``kk_c`` is pinned at Schönleber's 0.85 here on purpose: at *this* rig's
        ``kk_c = 0.30`` the first crossing and the chosen order often coincide, so a
        test at the shipped value would pass whether or not the walk continues. μ is
        not monotone in M — measured on a noise-free control it runs 0.81, 0.31, 0.39,
        0.71 over M = 5…8 — so first-crossing stops on a cliff edge, and the
        comparison below is against the library's own implementation of that rule
        rather than against a remembered number.
        """
        import contextlib
        import io

        from impedance.validation import linKK

        f, Z = reference_spectrum(RIG_FREQ, R_bulk=5.0e7)
        asc = np.argsort(f)
        with contextlib.redirect_stdout(io.StringIO()):
            _M, _mu, _fit, res_re, res_im = linKK(
                f[asc], Z[asc], c=0.85, max_M=DEFAULT_KK_MAX_M,
                fit_type="complex", add_cap=True)
        first_crossing = float(np.median(np.hypot(res_re, res_im) * 100.0))

        chosen = lin_kk(f, Z, blocking=True, c=0.85)
        assert chosen.ok, chosen.error
        # Measured: 0.87 % at the first crossing (M = 8) against 0.0002 % at M = 31.
        assert float(np.median(chosen.resid_pct)) < first_crossing / 10.0

    def test_the_conditioning_floor_stops_the_ladder_interpolating_the_data(self):
        """Minimising the residual, unbounded, buys an order the data cannot resolve.

        On a 27-point scout sweep the lowest median residual of the entire walk sits at
        **M = 33** — six more time constants than there are frequencies — reached only
        because μ has collapsed to 0.005, i.e. the ladder's positive and negative
        resistor mass have grown equal and opposite. ``mu_floor`` is what makes that
        unreachable; without it "best fit" and "meaningless fit" are the same answer.
        """
        f, Z = mock_spectrum(np.logspace(np.log10(2.0e5), np.log10(6.475), 27))

        r = lin_kk(f, Z, blocking=True)
        assert r.ok, r.error
        assert r.M <= f.size
        assert r.mu >= DEFAULT_KK_MU_FLOOR

        unbounded = lin_kk(f, Z, blocking=True, mu_floor=0.0)
        assert unbounded.M > f.size, (
            "the floor must be the thing excluding that order, not luck — if the "
            "unbounded walk no longer overshoots, this test has stopped measuring")

    def test_a_single_dip_below_the_conditioning_floor_does_not_end_the_walk(self):
        """μ is not monotone below the floor either, so the walk cannot stop on one dip.

        On this exact spectrum μ reads 0.050 at M = 30 and 0.038 at M = 31 — both under
        the floor — and returns to 0.055 at M = 32, which is the order that actually
        fits best. Ending the walk at the first excursion selects M = 28 instead.
        The cost of being wrong here is small (0.250 % against 0.240 %) and that is the
        point: it is small enough that no threshold notices, which is exactly the kind
        of quiet dependence on an unstated monotonicity assumption worth removing.
        """
        f, Z = mock_spectrum(RIG_FREQ, seed=1)
        r = lin_kk(f, Z, blocking=True)
        assert r.ok, r.error
        assert r.M >= 32

    def test_the_max_M_ceiling_is_not_where_the_walk_is_allowed_to_stop(self):
        """``max_M = 50`` was documented as a benign runtime bound. It is a cliff.

        Forced to M = 49 the noise-free control returns a **434 % median / 14 109 %
        max** residual — the normal-equations inverse has lost its conditioning. So a
        selection rule that could ever return the ceiling would report catastrophe as
        measurement. It must return an interior order with a residual near zero.
        """
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=5.0e7)
        r = lin_kk(f, Z, blocking=True, max_M=DEFAULT_KK_MAX_M)
        assert r.ok, r.error
        assert r.M < DEFAULT_KK_MAX_M
        assert r.max_resid_pct < 1.0

    def test_an_unreachable_mu_target_still_produces_a_verdict(self):
        """No conditioned order can satisfy ``μ ≤ 0`` while also clearing the floor.

        A K–K test that returned nothing would be worse than one that returned a
        slightly under-flexible order, because :func:`gate_kk_truncation` treats "did
        not run" as a pass — so an empty feasible set would silently disarm the gate.
        The fallback exists for that, and this is the case that reaches it.
        """
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=5.0e7)
        r = lin_kk(f, Z, blocking=True, c=0.0)
        assert r.ok, r.error
        assert r.mu >= DEFAULT_KK_MU_FLOOR
        assert r.max_resid_pct < 1.0

    def test_both_modules_ship_the_same_kk_defaults(self):
        """``kk`` and ``settings`` each declare ``DEFAULT_KK_C``; a split is a silent
        behaviour change, since the gate reads one and the config parser the other."""
        from softae.analysis.eis import settings as eis_settings

        assert DEFAULT_KK_C == eis_settings.DEFAULT_KK_C == 0.30
        assert DEFAULT_KK_MAX_M == eis_settings.DEFAULT_KK_MAX_M
        assert eis_settings.DEFAULT_KK_RESID_PCT == 3.0

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
    @pytest.mark.parametrize("R_bulk", RIG_R_BULK)
    def test_a_clean_spectrum_keeps_every_point_at_every_regime(self, R_bulk):
        """T1. The anti-vacuity test, and the one that fails hardest today.

        This used to run only at ``R_bulk = 5e4`` on the 41-point 20 Hz fixture, where
        the shipped gate happens to pass with 0 dropped and a 0.69 % max residual — so
        it would stay green for *any* value of ``kk_c`` or ``kk_resid_pct``, in either
        direction, and could not distinguish "the gate works" from "the gate is not
        running". Measured at the rig's own sweep under the shipped settings: 5e6 →
        ``block_spectrum``, 5e7 → **25 of 53 points dropped**.

        The spectra come from the suite's own noise-free generator and are exactly
        causal by construction, so a failure here is unambiguously the gate's.
        """
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=R_bulk)
        r = gate_kk_truncation(f, Z, ctx())
        assert r.passed, r.detail
        assert r.n_dropped == 0, r.detail

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_threshold_sits_above_this_rigs_noise_rather_than_below_it(self, seed):
        """The other half of the fix, and the half no noise-free fixture can reach.

        After the ladder is chosen properly a synthetic clean spectrum sits three
        decades under *either* threshold, so nothing in this file would notice
        ``kk_resid_pct`` moving. What notices is scatter at the rig's own amplitude:
        this rig's per-point noise floor is ~3 % (1.25 – 5.31 % over the ten reference
        spectra, by a local-scatter estimate that uses no K–K basis at all). Given a
        spectrum with that much **stationary** scatter and no drift whatsoever, the
        specification's 1 % ceiling truncates 6 – 12 low-frequency points — moving
        ``R1``, and therefore σ, on data whose only defect is having been measured.

        The second assertion is the anti-vacuity half: if 1 % ever stops firing here,
        the first has stopped meaning anything.
        """
        f, Z = mock_spectrum(RIG_FREQ, noise_rel=0.03, seed=seed)
        assert gate_kk_truncation(f, Z, ctx()).n_dropped == 0
        assert gate_kk_truncation(f, Z, ctx(kk_resid_pct=1.0)).n_dropped > 0

    def test_the_gate_still_separates_a_drifted_spectrum_from_a_clean_one(self):
        """T2. T1 alone would license relaxing the threshold to infinity.

        Both halves are needed and neither is the interesting one alone. The clean
        half fails today — the shipped gate drops **25 points from a spectrum with no
        drift at all** — and it is the sharper statement of the defect: a gate that
        cannot tell 0 % from 60 % is not measuring drift, it is measuring its own
        ladder order.
        """
        f, Z = reference_spectrum(RIG_FREQ, R_bulk=5.0e7)
        clean = gate_kk_truncation(f, Z, ctx())
        assert clean.n_dropped == 0
        # The median is what order selection minimises, so it is the number that says
        # whether a residual is the sample's or the ladder's. Logged for that reason.
        assert clean.metrics["kk_median_resid_pct"] < GateSettings().kk_resid_pct

        f, Zd = drifted(5, freq=RIG_FREQ, R_bulk=5.0e7)
        drift_result = gate_kk_truncation(f, Zd, ctx())
        assert drift_result.n_dropped >= 3, drift_result.detail
        assert not drift_result.passed

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
