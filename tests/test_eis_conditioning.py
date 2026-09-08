"""Stage-1 conditioning — the repair chain, not the gates.

Every spectrum here is **synthesised from plain NumPy or from the shared reference
topology**, so these tests hold on a machine with no ``AMP_v1`` beside it, no DataStore
and no commissioning artifact. Ground truth is then exactly known, which is the only way
to assert that a *conditioner* removed the points it was supposed to and no others.

Two properties get more attention than the rest, because both are ways this module could
pass a whole suite while doing nothing:

**The wrappers must actually reach the shipped gates.** Steps 3 and 4 delegate to
``gates.gate_magnitude`` and ``gates.gate_hf_inductive``, which read their thresholds out
of ``ctx`` through ``_ctx_get`` — a helper that **falls back to a default on a missing
section** rather than raising. A wrapper that filed the window at the top level instead
of under ``"envelope"`` would therefore run with an unbounded window, drop nothing, and
report a clean spectrum. So there are spy tests that the call happens *and* shape tests
that would fail on a wrong key, rather than only positive controls that pass either way.

**The chain must compose.** Each stage runs on the previous stage's survivors, so the
per-stage counts have to partition the input exactly — no point dropped twice, no stage
tripped by a gap an earlier stage left.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.calibration import CalibrationSet
from softae.analysis.eis.conditioning import (
    DEFAULT_KK_TRIM_PCT,
    MIN_CONDITIONED_POINTS,
    MIN_KK_POINTS,
    STAGE_HF_INDUCTIVE,
    STAGE_KK,
    STAGE_PHYSICAL,
    STAGE_SHORT,
    STAGE_WINDOW,
    ConditioningResult,
    condition,
    hf_inductive_mask,
    kk_trim_mask,
    magnitude_window_mask,
    physical_mask,
    short_constants,
    short_correct,
)

from .eis_synthetic import (
    hf_phase_artifact,
    log_frequencies,
    negative_real_part,
    reference_spectrum,
)

#: A window wide enough to admit the whole reference spectrum, so a test that is not
#: about the window can still switch it on and prove it removed nothing.
WIDE_WINDOW = {"z_min_ohm": 1.0, "z_max_ohm": 1.0e12}


def calibration(channel: int = 7, *, R: float = 12.0, L: float = 4.18e-6,
                **kw) -> CalibrationSet:
    """A minimal commissioned set carrying only what step 2 reads.

    ``L = 4.18 µH`` is the short blank's measured lead inductance on this fixture — the
    number the overhaul contrasts against the 400–500 µH a fit invents when the HF
    inductive run is left in.
    """
    return CalibrationSet(
        fixture_id="test", channels_measured=(channel,),
        R_short_ohm={channel: R}, L_lead_H={channel: L}, **kw,
    )


# ── Step 1: the physical domain ──────────────────────────────────────────────

class TestPhysicalFilter:
    def test_physical_mask_drops_a_negative_real_point_and_keeps_the_rest(self):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[17] = -8.1e6 + 1j * Z[17].imag      # the prototype's own 10 MΩ pathology
        mask = physical_mask(Z)
        assert not mask[17]
        assert mask.sum() == Z.size - 1
        assert np.flatnonzero(~mask).tolist() == [17]

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_physical_mask_drops_a_non_finite_point(self, bad):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[5] = complex(bad, 0.0)
        assert np.flatnonzero(~physical_mask(Z)).tolist() == [5]

    def test_physical_mask_drops_a_point_with_a_non_finite_imaginary_part(self):
        # `Re Z > 0` alone would admit it; the finiteness half of the predicate is what
        # stops a NaN reaching the K-K basis at step 5.
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[5] = complex(Z[5].real, np.nan)
        assert np.flatnonzero(~physical_mask(Z)).tolist() == [5]

    def test_physical_mask_keeps_every_point_of_a_clean_spectrum(self):
        _, Z = reference_spectrum()
        assert physical_mask(Z).all()

    def test_physical_stage_runs_first_so_one_absurd_point_never_reaches_the_ladder(self):
        # The ordering claim in the module docstring, asserted rather than asserted-about:
        # the bad point is gone by the time any later stage sees the spectrum.
        f, Z = negative_real_part(n_points=3)
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.dropped(STAGE_PHYSICAL) == 3
        assert r.stages[0].name == STAGE_PHYSICAL
        assert (r.Z.real > 0).all() and np.isfinite(r.Z).all()


# ── Step 2: short / fixture series correction ────────────────────────────────

class TestShortCorrection:
    def test_short_correct_removes_exactly_the_series_term(self):
        f, Z_true = reference_spectrum()
        R, L = 12.0, 4.18e-6
        measured = Z_true + (R + 1j * 2.0 * np.pi * f * L)
        recovered = short_correct(f, measured, R_short_ohm=R, L_lead_H=L)
        assert np.allclose(recovered, Z_true, rtol=0, atol=1e-9 * np.abs(Z_true).max())

    def test_short_correct_is_frequency_dependent_through_the_inductance(self):
        # A test that passed with `L` ignored would be satisfied by subtracting `R`
        # alone, which is the whole failure the pinned-L row of the 2.1 table describes.
        f = log_frequencies()
        Z = np.full(f.size, 1000.0 + 0j)
        out = short_correct(f, Z, R_short_ohm=0.0, L_lead_H=1e-5)
        assert np.allclose(out.imag, -2.0 * np.pi * f * 1e-5)
        assert out.imag.min() < out.imag.max()

    def test_short_constants_are_read_from_the_calibration_set(self):
        R, L, why = short_constants(calibration(7, R=12.0, L=4.18e-6), 7)
        assert (R, L) == (12.0, 4.18e-6)
        assert why == ""

    def test_short_constants_for_an_uncalibrated_channel_skip_rather_than_return_nan(self):
        # NaN constants would not degrade the spectrum, they would erase it: every point
        # becomes NaN and step 1's verdict is retroactively meaningless.
        R, L, why = short_constants(calibration(7), 99)
        assert (R, L) == (0.0, 0.0)
        assert "99" in why and "R_short_ohm" in why

    @pytest.mark.parametrize("cal,ch", [(None, 7), (calibration(7), None)])
    def test_short_constants_without_a_calibration_or_channel_are_skipped(self, cal, ch):
        R, L, why = short_constants(cal, ch)
        assert (R, L) == (0.0, 0.0) and "skipped" in why

    def test_the_chain_applies_the_calibrations_constants_to_the_returned_spectrum(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, calibration=calibration(7, R=500.0, L=1e-5),
                      channel=7, **WIDE_WINDOW)
        assert (r.R_short_ohm, r.L_lead_H) == (500.0, 1e-5)
        expected = short_correct(f, Z, R_short_ohm=500.0, L_lead_H=1e-5)[r.mask]
        assert np.allclose(r.Z, expected)
        assert not np.allclose(r.Z, Z[r.mask]), "the correction must actually be applied"

    def test_an_absent_calibration_is_declined_and_leaves_the_spectrum_uncorrected(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.stage(STAGE_SHORT).declined
        assert (r.R_short_ohm, r.L_lead_H) == (0.0, 0.0)
        assert np.allclose(r.Z, Z[r.mask])

    def test_the_short_stage_removes_no_points(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, calibration=calibration(7), channel=7, **WIDE_WINDOW)
        assert r.dropped(STAGE_SHORT) == 0


# ── Step 3: the |Z| magnitude window ─────────────────────────────────────────

class TestMagnitudeWindow:
    def test_magnitude_window_drops_the_points_outside_the_band(self):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[3] *= 1e6                                  # far over range
        Z[20] *= 1e-9                                # far under range
        mask = magnitude_window_mask(f, Z, z_min_ohm=1.0, z_max_ohm=1.0e9)
        assert np.flatnonzero(~mask).tolist() == [3, 20]

    def test_magnitude_window_keeps_every_point_of_an_in_band_spectrum(self):
        f, Z = reference_spectrum()
        assert magnitude_window_mask(f, Z, **WIDE_WINDOW).all()

    def test_an_omitted_bound_is_unbounded_on_that_side(self):
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[3] *= 1e6
        assert magnitude_window_mask(f, Z, z_min_ohm=1.0).all()
        assert np.flatnonzero(
            ~magnitude_window_mask(f, Z, z_max_ohm=1.0e9)).tolist() == [3]

    def test_the_window_reaches_the_gate_under_the_key_it_reads(self):
        """NON-VACUITY. ``_ctx_get`` falls back to ``gate_magnitude``'s own defaults —
        ``0.0`` and ``inf`` — on a missing ``"envelope"`` section, so a wrapper that filed
        the bounds anywhere else would drop **nothing** and report a clean spectrum. This
        asserts a drop that is impossible under the fallback."""
        f, Z = reference_spectrum()
        mag = np.abs(Z)
        cut = float(np.median(mag))
        mask = magnitude_window_mask(f, Z, z_max_ohm=cut)
        assert 0 < int((~mask).sum()) < mag.size
        assert np.array_equal(mask, mag <= cut)

    def test_the_wrapper_calls_the_shipped_gate_rather_than_reimplementing_it(self,
                                                                             monkeypatch):
        """NON-VACUITY, the other half: the ctx handed over carries the bounds where the
        gate reads them, checked on the object actually passed."""
        from softae.analysis.eis import conditioning as module

        seen: list[dict] = []
        real = module.gate_magnitude

        def spy(f, Z, ctx):
            seen.append(ctx)
            return real(f, Z, ctx)

        monkeypatch.setattr(module, "gate_magnitude", spy)
        f, Z = reference_spectrum()
        magnitude_window_mask(f, Z, z_min_ohm=1.0, z_max_ohm=2.0e9)
        assert len(seen) == 1
        assert seen[0]["envelope"] == {"z_min_ohm": 1.0, "z_max_ohm": 2.0e9}

    def test_the_chain_takes_its_window_from_the_calibration_set(self):
        f, Z = reference_spectrum()
        cal = calibration(7, z_min_ohm=1.0, z_max_ohm=float(np.median(np.abs(Z))))
        r = condition(f, Z, calibration=cal, channel=7)
        assert r.dropped(STAGE_WINDOW) > 0

    def test_an_explicit_window_overrides_the_calibrations(self):
        f, Z = reference_spectrum()
        cal = calibration(7, z_min_ohm=1.0, z_max_ohm=float(np.median(np.abs(Z))))
        r = condition(f, Z, calibration=cal, channel=7, **WIDE_WINDOW)
        assert r.dropped(STAGE_WINDOW) == 0
        assert not r.stage(STAGE_WINDOW).declined

    def test_an_uncommissioned_window_is_declined_not_guessed(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, calibration=calibration(7), channel=7)
        stage = r.stage(STAGE_WINDOW)
        assert stage.declined and stage.n_dropped == 0


# ── Step 4: HF-inductive truncation ──────────────────────────────────────────

class TestHFInductiveTruncation:
    def test_a_contiguous_inductive_run_at_the_top_of_the_band_is_truncated(self):
        f, Z = hf_phase_artifact(n_points=4)
        mask = hf_inductive_mask(f, Z)
        dropped = np.flatnonzero(~mask)
        assert dropped.size == 4
        # `log_frequencies` descends, so the top of the band is the head of the array.
        assert set(dropped.tolist()) == set(np.argsort(f)[::-1][:4].tolist())

    def test_the_truncation_stops_at_the_first_capacitive_point(self):
        # An inductive point *below* a capacitive one is not part of the run at the top
        # of the band and must survive - otherwise this is a sign filter, not a
        # truncation, and it would delete isolated noise wherever it occurred.
        f, Z = hf_phase_artifact(n_points=2)
        Z = Z.copy()
        Z[10] = Z[10].real + 1j * abs(Z[10].imag)
        assert np.flatnonzero(~hf_inductive_mask(f, Z)).tolist() == [0, 1]

    def test_a_clean_blocking_spectrum_is_truncated_nowhere(self):
        f, Z = reference_spectrum()
        assert hf_inductive_mask(f, Z).all()

    def test_a_non_blocking_cell_keeps_its_inductive_run(self):
        """NON-VACUITY on the ctx shape. ``blocking`` is read from ``ctx["cell"]``; a
        wrapper that put it at the top level would leave the gate on its ``True``
        default and truncate a spectrum whose inductance may be real."""
        f, Z = hf_phase_artifact(n_points=4)
        assert hf_inductive_mask(f, Z, blocking=False).all()
        assert not hf_inductive_mask(f, Z, blocking=True).all()

    def test_the_wrapper_calls_the_shipped_gate_with_the_cell_section_populated(
            self, monkeypatch):
        from softae.analysis.eis import conditioning as module

        seen: list[dict] = []
        real = module.gate_hf_inductive

        def spy(f, Z, ctx):
            seen.append(ctx)
            return real(f, Z, ctx)

        monkeypatch.setattr(module, "gate_hf_inductive", spy)
        f, Z = hf_phase_artifact(n_points=4)
        hf_inductive_mask(f, Z, blocking=False)
        assert len(seen) == 1
        assert seen[0]["cell"]["blocking"] is False
        assert "blocking" not in seen[0], "the gate reads ctx['cell'], not the top level"

    def test_the_chain_truncates_the_run_and_counts_it_against_this_stage(self):
        f, Z = hf_phase_artifact(n_points=4)
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.dropped(STAGE_HF_INDUCTIVE) == 4
        assert r.dropped(STAGE_PHYSICAL) == 0
        assert (r.Z.imag <= 0).all()

    def test_the_chain_honours_a_non_blocking_cell(self):
        f, Z = hf_phase_artifact(n_points=4)
        r = condition(f, Z, blocking=False, **WIDE_WINDOW)
        assert r.dropped(STAGE_HF_INDUCTIVE) == 0


# ── Step 5: linear-K–K low-frequency trim ────────────────────────────────────

def drifted(n_points: int = 5, factor_hi: float = 1.6, factor_lo: float = 1.15):
    """A spectrum whose *lowest* frequencies drifted during acquisition.

    The pathology §3.6 licenses truncating: acquisition at the bottom of the band is slow
    enough for the sample itself to change mid-sweep, and ``R_bulk`` does not live there.
    """
    f, Z = reference_spectrum()
    Zd = np.array(Z, dtype=complex)
    lo = np.argsort(f)[:n_points]
    Zd[lo] *= np.linspace(factor_hi, factor_lo, n_points)
    return f, Zd


class TestKKTrim:
    def test_a_drifting_low_frequency_tail_is_trimmed(self):
        """The trim removes a contiguous run from the low-frequency end that covers the
        drift — and, measured here, somewhat more than the drift.

        Perturbing 5 points removes 7. That is **not** slack in the assertion: the ladder
        is a *global* weighted fit, so drift at the tail shifts the whole basis and
        neighbouring good points fail the residual threshold too. ``kk.py`` records the
        extreme of the same effect (5 perturbed points driving 40 of 41 past 1 %), which
        is why :data:`~softae.analysis.eis.kk.DEFAULT_KK_MAX_TRUNCATE_FRAC` exists as a
        correctness bound rather than as conservatism. Asserting the drifted points are a
        *subset* of what goes is the honest shape of the property; asserting equality
        would be asserting something the method does not provide.
        """
        f, Z = drifted(5)
        mask, kk = kk_trim_mask(f, Z)
        assert kk.ok
        ascending = np.argsort(f)
        dropped = set(np.flatnonzero(~mask).tolist())
        assert set(ascending[:5].tolist()) <= dropped
        # A contiguous prefix in ascending-frequency order, and nothing else.
        assert dropped == set(ascending[:len(dropped)].tolist())
        # The high-frequency arc that carries R_bulk is untouched.
        assert mask[ascending[-1]] and len(dropped) < f.size // 2

    def test_a_clean_spectrum_survives_the_ladder_intact(self):
        f, Z = reference_spectrum()
        mask, kk = kk_trim_mask(f, Z)
        assert kk.ok and mask.all()

    def test_an_isolated_mid_band_failure_is_never_trimmed(self):
        # The directionality policy: only drift at the low-frequency end is removable.
        # An interior outlier fails the same threshold and must survive, because the
        # criterion licensing removal does not apply to it.
        f, Z = reference_spectrum()
        Z = Z.copy()
        Z[len(f) // 2] *= 1.5
        mask, kk = kk_trim_mask(f, Z)
        assert kk.ok
        assert mask[len(f) // 2]

    def test_the_trim_consumes_the_ladders_own_resid_pct(self, monkeypatch):
        """NON-VACUITY: the threshold must be applied to ``LinKKResult.resid_pct``, not
        recomputed. Feeding a stub ladder that declares which points fail proves the
        trim reads the field rather than judging the spectrum some other way."""
        from softae.analysis.eis import conditioning as module
        from softae.analysis.eis.kk import LinKKResult

        f, Z = reference_spectrum()
        resid = np.zeros(f.size)
        resid[np.argsort(f)[:3]] = 99.0        # the three lowest frequencies "fail"
        monkeypatch.setattr(
            module, "lin_kk",
            lambda f, Z, **kw: LinKKResult(ok=True, M=9, mu=0.5, resid_pct=resid))
        mask, _ = kk_trim_mask(f, Z, kk_pct=1.0)
        assert np.flatnonzero(~mask).tolist() == sorted(np.argsort(f)[:3].tolist())

    def test_a_ladder_that_will_not_fit_trims_nothing(self):
        f = log_frequencies()
        mask, kk = kk_trim_mask(f, np.full(f.size, np.nan, dtype=complex))
        assert not kk.ok and mask.all()

    def test_the_chain_declines_the_trim_below_the_ladder_minimum(self):
        f, Z = reference_spectrum(log_frequencies(npts=MIN_KK_POINTS - 1))
        r = condition(f, Z, **WIDE_WINDOW)
        stage = r.stage(STAGE_KK)
        assert stage.declined and stage.n_dropped == 0
        assert r.kk is None

    def test_the_chain_records_the_ladder_it_used(self):
        f, Z = drifted(5)
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.kk is not None and r.kk.ok and r.kk.M > 0
        assert r.dropped(STAGE_KK) > 0


# ── The chain as a whole ─────────────────────────────────────────────────────

class TestConditionComposes:
    def test_a_clean_spectrum_passes_every_stage_untouched(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, calibration=calibration(7, R=0.0, L=0.0), channel=7,
                      **WIDE_WINDOW)
        assert r.n_in == r.n_out == f.size
        assert r.declined == ()
        assert [s.n_dropped for s in r.stages] == [0, 0, 0, 0, 0]
        assert np.allclose(r.Z, Z) and np.allclose(r.f, f)

    def test_stages_run_in_the_specified_order(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, **WIDE_WINDOW)
        assert [s.name for s in r.stages] == [
            STAGE_PHYSICAL, STAGE_SHORT, STAGE_WINDOW, STAGE_HF_INDUCTIVE, STAGE_KK]

    def test_the_per_stage_counts_partition_the_input_exactly(self):
        # No point dropped twice, none dropped without an owner.
        f, Z = hf_phase_artifact(n_points=4)
        Z = Z.copy()
        Z[20] = -abs(Z[20].real) + 1j * Z[20].imag
        Z[25] *= 1e6
        r = condition(f, Z, z_min_ohm=1.0, z_max_ohm=1.0e9)
        assert sum(s.n_dropped for s in r.stages) == r.n_in - r.n_out == r.n_dropped
        assert int(r.mask.sum()) == r.n_out == r.f.size == r.Z.size

    def test_each_pathology_is_charged_to_its_own_stage(self):
        f, Z = hf_phase_artifact(n_points=4)
        Z = Z.copy()
        Z[20] = -abs(Z[20].real) + 1j * Z[20].imag     # step 1
        Z[25] *= 1e6                                    # step 3
        r = condition(f, Z, z_min_ohm=1.0, z_max_ohm=1.0e9)
        assert r.dropped(STAGE_PHYSICAL) == 1
        assert r.dropped(STAGE_WINDOW) == 1
        assert r.dropped(STAGE_HF_INDUCTIVE) == 4
        assert not r.mask[20] and not r.mask[25]

    def test_an_earlier_stages_removal_does_not_break_the_hf_walk(self):
        """Order matters, and this is the shape that would break it. The HF walk stops at
        the first capacitive point counting down from the top; if it ran on the full array
        rather than on the survivors, the window-dropped point sitting inside the
        inductive run would either halt the walk early or let it continue through a gap."""
        f, Z = hf_phase_artifact(n_points=4)
        Z = Z.copy()
        hi = np.argsort(f)[::-1]
        Z[hi[2]] *= 1e6                                 # over range, inside the run
        r = condition(f, Z, z_min_ohm=1.0, z_max_ohm=1.0e9)
        assert r.dropped(STAGE_WINDOW) == 1
        assert r.dropped(STAGE_HF_INDUCTIVE) == 3       # the other three of the four
        for i in hi[:4]:
            assert not r.mask[i]

    def test_the_returned_mask_indexes_the_callers_own_points(self):
        f, Z = hf_phase_artifact(n_points=4)
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.mask.shape == (f.size,)
        assert np.array_equal(r.f, f[r.mask])

    def test_a_stage_declines_rather_than_taking_the_spectrum_below_the_floor(self):
        f, Z = reference_spectrum(log_frequencies(npts=MIN_CONDITIONED_POINTS + 1))
        # A window admitting only two points: applying it would leave 2 < 6.
        mag = np.abs(Z)
        lo, hi = float(np.sort(mag)[0]), float(np.sort(mag)[1])
        r = condition(f, Z, z_min_ohm=lo, z_max_ohm=hi)
        stage = r.stage(STAGE_WINDOW)
        assert stage.declined and stage.n_dropped == 0
        assert r.n_out == r.n_in, "a conditioner that deletes a spectrum has not repaired it"

    def test_the_floor_declines_one_stage_without_undoing_an_earlier_one(self):
        # The deliberate divergence from the prototype, which reverted to the window-only
        # mask on shortfall and so discarded the physical filter's verdict as collateral.
        f, Z = reference_spectrum(log_frequencies(npts=MIN_CONDITIONED_POINTS + 2))
        Z = Z.copy()
        Z[1] = -abs(Z[1].real) + 1j * Z[1].imag
        mag = np.abs(Z)
        finite = np.sort(mag[physical_mask(Z)])
        r = condition(f, Z, z_min_ohm=float(finite[0]), z_max_ohm=float(finite[1]))
        assert r.dropped(STAGE_PHYSICAL) == 1
        assert r.stage(STAGE_WINDOW).declined
        assert not r.mask[1]

    @pytest.mark.parametrize("npts", [0, 1, 5])
    def test_a_spectrum_too_short_to_condition_is_returned_not_raised(self, npts):
        f = np.logspace(5, 1, npts) if npts else np.empty(0)
        Z = np.full(npts, 1000.0 - 1000.0j, dtype=complex)
        r = condition(f, Z, **WIDE_WINDOW)
        assert isinstance(r, ConditioningResult)
        assert r.n_in == npts and r.n_out == npts

    def test_mismatched_input_lengths_are_truncated_to_the_shorter(self):
        f, Z = reference_spectrum()
        r = condition(f, Z[:30], **WIDE_WINDOW)
        assert r.n_in == 30

    def test_describe_is_ascii_so_logging_it_cannot_raise_on_a_cp1252_console(self):
        f, Z = hf_phase_artifact(n_points=4)
        text = condition(f, Z, **WIDE_WINDOW).describe()
        text.encode("cp1252")                            # raises if it is not
        assert STAGE_HF_INDUCTIVE in text

    def test_dropped_returns_zero_for_a_stage_that_is_not_in_the_chain(self):
        f, Z = reference_spectrum()
        r = condition(f, Z, **WIDE_WINDOW)
        assert r.stage("not_a_stage") is None
        assert r.dropped("not_a_stage") == 0

    def test_the_default_kk_threshold_is_the_prototypes(self):
        assert DEFAULT_KK_TRIM_PCT == 1.0


# ── Inertness ────────────────────────────────────────────────────────────────

def test_module_is_not_wired_into_any_live_path() -> None:
    """It ships inert. This test is what makes that a contract rather than a comment.

    Matched on an **import-shaped** pattern, where the equivalent test for
    ``constrained_fit`` can use a plain substring. The difference is not stylistic: the
    bare word "conditioning" already appears as prose in seven ``src/softae`` modules —
    ``kk.py`` alone uses it four times, for the μ-floor's conditioning bound — so a
    substring test here would report six false importers and would have to be silenced,
    which is how a real wiring later goes unnoticed.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "softae"
    pattern = re.compile(r"^\s*(?:from|import)\s+[^\n]*\bconditioning\b", re.MULTILINE)
    importers = [str(p) for p in src.rglob("*.py")
                 if p.name != "conditioning.py"
                 and pattern.search(p.read_text(encoding="utf-8"))]
    assert importers == [], f"conditioning is imported by {importers}"


def test_the_inertness_pattern_would_catch_a_real_import(tmp_path) -> None:
    """The check above is only worth having if it fires. Every import spelling that
    would actually wire this module in, matched against the same pattern."""
    pattern = re.compile(r"^\s*(?:from|import)\s+[^\n]*\bconditioning\b", re.MULTILINE)
    for line in ("from softae.analysis.eis.conditioning import condition",
                 "from softae.analysis.eis import conditioning",
                 "import softae.analysis.eis.conditioning",
                 "    from softae.analysis.eis import conditioning  # deferred"):
        assert pattern.search(line), line
    assert not pattern.search("#: the conditioning half of order selection")
    assert not pattern.search("from softae.analysis.eis.kk import lin_kk")
