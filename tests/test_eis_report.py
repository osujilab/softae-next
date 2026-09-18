"""Tests for `SpectrumReport.gate_summary` (report.py) — the Analysis-tab-independent
surface of the Gate column.

`_gate_item` (tab_analysis.py) was fixed first ([p136] section 3) to name a failed
`flag`-severity gate instead of letting it fall through to a bare "pass". That mail
post's own "found, not fixed" note flagged the identical defect here, six lines away in
`gate_summary`, which the docstring says is adopted by `_gate_item` "token-for-token so
the two surfaces cannot disagree" — a claim that was false for exactly this branch until
now.
"""
from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis.admittance import par_capacitance_estimate
from softae.analysis.eis.engine import (
    CEILING_BELOW_FIT,
    _note_sigma_ceiling,
    _sigma_from_R,
)
from softae.analysis.eis.envelope import InstrumentEnvelope
from softae.analysis.eis.gates import FLAG, GateResult
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.policy import reduce_gates
from softae.analysis.eis.report import (
    SigmaCeiling,
    SigmaReport,
    SpectrumReport,
    sigma_loss_ceiling,
)

_MASK_OK = np.ones(5, dtype=bool)

#: Same geometry ``tests/test_eis_engine.py`` uses, so the two files' numbers compare.
_CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                     thickness_method="predicted")


def _flag_entry(passed: bool, name: str = "arc_closure") -> dict:
    """A `flag`-severity gate log entry in the real `GateResult.as_log_entry()` shape.

    `passed=False` is `gate_arc_closure` on an OPEN arc: it ran (`checked=True`),
    found the arc did not close, and refuses nothing — no `n_dropped`, no rejection.
    That is the shape with no counter of its own, so it has to be named or it is
    invisible.
    """
    detail = "within tolerance" if passed else "apex not bracketed"
    return GateResult(name, FLAG, passed, detail, _MASK_OK).as_log_entry()


def test_gate_summary_failed_flag_renders_flagged_passed_flag_renders_pass():
    failed = SpectrumReport(engine="gated", gate_log=(_flag_entry(passed=False),))
    passed = SpectrumReport(engine="gated", gate_log=(_flag_entry(passed=True),))

    assert failed.gate_summary() == "arc_closure flagged"
    assert passed.gate_summary() == "pass"


class TestSigmaLossCeiling:
    """T11.34 - the ceiling is evaluated where the comparison was MADE.

    ``sigma_upper_bound`` computed ``K*eps*omega*C`` at ``omega = 2*pi*min(f)`` with
    ``C`` the top-decade median: a verbatim second copy of
    ``InstrumentEnvelope.sigma_min``, the detection FLOOR, taken at the frequency that
    makes a floor smallest and then rendered with a ``≲``. On the eleven bound
    spectra of ``rung3a_fake_cast`` it read ~3.9e-10 S/cm against a fit-implied
    3.1e-06...4.9e-05 - a ceiling three to five decades *under* the point estimate it
    claimed to cap.

    The bound branch fires because ``tan d = G/(omega*C)`` fell under ``eps`` at ONE
    frequency, so the statement it licenses is ``G(omega*) <= eps*omega**C(omega*)`` at
    that omega - the headroom numerator's. Taking the most favourable omega is right
    for a floor and inverted for a ceiling (``SUBAGENT_RULES`` 3.3).
    """

    @staticmethod
    def _pure_capacitor(c_farad: float = 1.5e-9, n: int = 41):
        """``C_app = 1/(omega*|Z''|) = C`` at every frequency.

        A flat ``C_app`` is what makes the scaling test an exact-ratio assertion: the
        ceiling's only frequency dependence left is the explicit omega.
        """
        f = np.logspace(0, 5, n)
        Z = -1j / (2.0 * np.pi * f * c_farad)
        return f, Z

    @staticmethod
    def _envelope(**kw):
        return InstrumentEnvelope(**kw)

    def test_sigma_loss_ceiling_at_numerator_frequency_scales_with_that_frequency(self):
        f, Z = self._pure_capacitor()
        env = self._envelope()

        lo = sigma_loss_ceiling(f, Z, envelope=env, cell=_CELL, at_f_hz=100.0)
        hi = sigma_loss_ceiling(f, Z, envelope=env, cell=_CELL, at_f_hz=1000.0)

        assert lo.basis == hi.basis == "loss_at_numerator"
        assert hi.value == pytest.approx(lo.value * 10.0, rel=1e-12)
        assert (lo.f_hz, hi.f_hz) == (100.0, 1000.0)

    def test_sigma_loss_ceiling_missing_numerator_frequency_returns_unavailable(self):
        """The positive control: it must NOT quietly fall back to ``min(f)``.

        The old expression is computed here and asserted finite, so a green result
        proves the branch *could* have produced a number and declined to - rather than
        proving only that some unrelated guard returned NaN first
        (``SUBAGENT_RULES`` 3.1(a)).
        """
        f, Z = self._pure_capacitor()
        env = self._envelope()

        old = (env.eps_rad * 2.0 * np.pi * float(np.min(f))
               * par_capacitance_estimate(f, Z) * _CELL.K_per_cm)
        assert np.isfinite(old) and old > 0

        ceiling = sigma_loss_ceiling(f, Z, envelope=env, cell=_CELL,
                                     at_f_hz=float("nan"))

        assert ceiling.basis == "unavailable"
        assert np.isnan(ceiling.value)
        assert np.isnan(ceiling.f_hz)

    def test_sigma_loss_ceiling_unmeasured_epsilon_returns_magnitude_ceiling(self):
        f, Z = self._pure_capacitor()
        blind = self._envelope(phase_noise_measured=False,
                               phase_noise_deg=float("nan"))

        assert np.isnan(blind.eps_rad)  # the ONLY condition that may reach this basis

        ceiling = sigma_loss_ceiling(f, Z, envelope=blind, cell=_CELL, at_f_hz=1000.0)

        assert ceiling.basis == "magnitude_ceiling"
        assert ceiling.value == pytest.approx(_CELL.K_per_cm / blind.z_max_ohm)
        assert np.isnan(ceiling.eps_rad)

    def test_sigma_loss_ceiling_measured_epsilon_without_capacitance_returns_unavailable(self):
        """A missing capacitance is an unknown, NOT the magnitude ceiling.

        The first revision of this function let this case fall out of the ``eps``
        block and into the ``K/Z_max`` return, so a spectrum that simply had no usable
        ``C_app`` came back labelled ``magnitude_ceiling`` — the one basis that means
        *checked, and this is a real ceiling on sigma*. That is ``SUBAGENT_RULES``
        3.1(a) exactly: an unknown spelled with the checked answer's token.

        The positive control is the same shape as the missing-frequency test: the
        ``K/Z_max`` number is computed here and asserted finite, so a green proves the
        function had that answer available and declined to give it.
        """
        f = np.logspace(0, 5, 41)
        purely_real = np.full(f.shape, 1.0e7 + 0j)  # Im Z == 0 => C_app undefined
        env = self._envelope()

        assert np.isfinite(env.eps_rad)
        available = _CELL.K_per_cm / env.z_max_ohm
        assert np.isfinite(available) and available > 0

        ceiling = sigma_loss_ceiling(f, purely_real, envelope=env, cell=_CELL,
                                     at_f_hz=1000.0)

        assert ceiling.basis == "unavailable"
        assert np.isnan(ceiling.value)
        assert ceiling.reason == "no capacitance at the numerator frequency"
        assert ceiling.eps_rad == env.eps_rad

    def test_sigma_loss_ceiling_uses_the_capacitance_at_that_frequency(self):
        """``C`` is the one inside the ``tan d`` that decided the label, not a median.

        This spectrum's ``C_app`` is 10x larger below 100 Hz than in the top decade, so
        the top-decade median ``par_capacitance_estimate`` takes and the local value
        differ by exactly that factor and the two cannot be confused.
        """
        f = np.logspace(0, 5, 51)
        c_local = np.where(f < 100.0, 1.5e-8, 1.5e-9)
        Z = -1j / (2.0 * np.pi * f * c_local)
        env = self._envelope()

        assert par_capacitance_estimate(f, Z) == pytest.approx(1.5e-9)

        at = float(f[f < 100.0].max())
        ceiling = sigma_loss_ceiling(f, Z, envelope=env, cell=_CELL, at_f_hz=at)

        assert ceiling.c_farad == pytest.approx(1.5e-8)
        assert ceiling.value == pytest.approx(
            env.eps_rad * 2.0 * np.pi * at * 1.5e-8 * _CELL.K_per_cm)


class TestSigmaReportCeilingRendering:
    """The ceiling and the fit it disagrees with travel together, or neither is much use."""

    def test_sigma_report_bound_carries_the_fit_implied_sigma(self):
        ceiling = SigmaCeiling(value=3.2e-7, f_hz=1033.0, c_farad=1.5e-9,
                               eps_rad=2.6e-3, basis="loss_at_numerator")

        sigma = _sigma_from_R(1.0e7, _CELL, mode="bound", provisional=False,
                              ceiling=ceiling, phase_headroom=1.2,
                              model_free_R=1.0e7)

        assert sigma.is_bound
        assert sigma.upper_bound == 3.2e-7
        assert sigma.upper_bound_f_hz == 1033.0
        assert sigma.upper_bound_basis == "loss_at_numerator"
        assert sigma.fit_implied_sigma == _CELL.sigma(1.0e7)

    def test_sigma_report_bound_text_names_the_frequency_and_the_fit(self):
        text = SigmaReport(mode="bound", upper_bound=3.2e-7, upper_bound_f_hz=1033.0,
                           fit_implied_sigma=4.2e-6).as_text()

        assert "≲ 3.2e-07 S/cm" in text
        assert "@1.03 kHz" in text
        assert "fit implies 4.2e-06" in text

    def test_sigma_report_ceiling_below_fit_says_so_in_the_text(self):
        text = SigmaReport(mode="bound_unqualified", upper_bound=3.2e-7,
                           upper_bound_f_hz=1033.0, fit_implied_sigma=4.2e-6,
                           provisional=True).as_text()

        assert "loss ceiling below the fit" in text
        assert text.endswith("(provisional)")

    def test_sigma_report_ceiling_above_fit_omits_the_disagreement_clause(self):
        text = SigmaReport(mode="bound", upper_bound=4.2e-6, upper_bound_f_hz=1033.0,
                           fit_implied_sigma=3.2e-7).as_text()

        assert "fit implies 3.2e-07" in text
        assert "below the fit" not in text

    def test_spectrum_report_ceiling_below_fit_raises_a_flag_not_a_refusal(self):
        """A flag, never a gate - the ruling ``arc_closure`` got on 2026-09-10.

        The two numbers are different estimators of different quantities (the loss
        tangent's parallel conductance at one frequency against the fit's DC
        resistance), so their disagreement is information, not evidence the spectrum is
        bad. Refusing on it would refuse real films.
        """
        baseline = reduce_gates([], n_surviving=20, min_fit_pts=8, report_mode="bound")
        quality = reduce_gates([], n_surviving=20, min_fit_pts=8, report_mode="bound")
        ceiling = SigmaCeiling(value=3.2e-7, f_hz=1033.0, c_farad=1.5e-9,
                               eps_rad=2.6e-3, basis="loss_at_numerator")
        sigma = _sigma_from_R(1.0e7, _CELL, mode="bound", provisional=False,
                              ceiling=ceiling, phase_headroom=1.2,
                              model_free_R=1.0e7)
        assert sigma.upper_bound < sigma.fit_implied_sigma

        _note_sigma_ceiling(sigma, ceiling, quality)

        assert any(i.startswith(CEILING_BELOW_FIT) for i in quality.issues)
        assert quality.ok
        assert quality.verdict is baseline.verdict
