"""Railing is asked of both resistances, not only of the measurand.

Until 2026-08-27 :func:`~softae.analysis.eis.models.railed_measurand` put one
question — *is ``R_bulk`` on its bound?* — and discarded the answer to every other
one, including on the gated path where ``FitCovariance.pegged()`` had already
computed them all. A fit whose ``R0`` had collapsed to 4.6e-62 Ω beside a perfectly
healthy ``R1`` therefore reported ``success = True`` with no reason, on both paths,
and stored a conductivity.

The population is not hypothetical: 449 of 3 618 stored ``simpleSalt`` fits carry an
``R0`` below 1e-30 Ω, and 222 of those sit beside an ``R1`` nowhere near its 100 Ω
floor — so they are invisible to the R₁ test specifically.

The watch is ``R_series`` and ``R_bulk`` and stops there. Extending it to *every*
fitted parameter was implemented first and reverted: the blocking-electrode CPE rails
on ordinary spectra, so it demoted whole boards and turned 14 settle-phase tests in
``test_eis_validate.py`` red. ``TestNuisanceParametersDoNotDemote`` is that finding
kept as a test.

This file lives apart from ``test_eis_engine.py`` deliberately. That file holds the
original railed-fit suite and is in flight under a different claim; a new module
importing the same surface costs it zero edits (``SUBAGENT_RULES`` §7).
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.circuit_fitting import CIRCUIT_MODELS, FitResult
from softae.analysis.eis.engine_support import _demote_if_railed
from softae.analysis.eis.fitter import FitCovariance
from softae.analysis.eis.models import COLLAPSED_AT_ZERO, railed_measurand

#: ``simpleSalt``'s fitted vector, in the order ``impedance`` reports it. The bounds
#: are read from the registry rather than restated, for the reason
#: ``r1_lower_bound_ohms`` gives: a bound written down twice is a bound that will
#: disagree with the fitter after the first edit.
NAMES = ("R0", "CPE0_0", "CPE0_1", "R1", "C0")
LOWER, UPPER = CIRCUIT_MODELS["simpleSalt"]["bounds"]

#: Comfortably inside every bound: R0 two decades above its floor of 0, R1 ~500x its
#: 100 Ω floor, the CPE exponent mid-range, C0 a decade off its 1e-11 floor.
HEALTHY = [50.0, 1e-7, 0.7, 5.0e4, 1e-10]


def _cov(values, *, bounds=(LOWER, UPPER)) -> FitCovariance:
    lo, hi = bounds
    return FitCovariance(
        names=NAMES,
        values=np.asarray(values, dtype=float),
        pcov=np.eye(len(values)) * 1e-4,
        n_points=40,
        bounds=(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)),
    )


def _fit(values, *, covariance=None, model_name="simpleSalt") -> FitResult:
    values = list(values)
    return FitResult(
        model_name=model_name,
        parameters=np.asarray(values, dtype=float),
        R0=float(values[0]),
        R1=float(values[3]),
        R0_guess=100.0,
        R1_guess=1.0e4,
        z_indices=[0, 3],
        success=True,
        covariance=covariance,
    )


def _legacy(values) -> FitResult:
    """The path that actually runs today — ``[eis] engine`` ships ``legacy``, and a
    legacy ``FitResult`` carries ``covariance = None`` because ``impedance``'s
    ``CustomCircuit`` discards ``pcov``."""
    return _fit(values)


def _gated(values) -> FitResult:
    return _fit(values, covariance=_cov(values))


# ── The defect: a collapsed R0 beside a healthy R1 ───────────────────────────

class TestSeriesResistanceCollapse:
    """``R0 = 4.6e-62`` Ω is not a series resistance, and an ``R1`` sitting beside it
    is not a bulk resistance either: the two are in series and the optimiser trades
    between them freely, so a floored ``R0`` means ``R1`` has absorbed it."""

    COLLAPSED = [4.6e-62, 1e-7, 0.7, 5.0e4, 1e-10]

    @pytest.mark.parametrize("build", [_legacy, _gated], ids=["legacy", "gated"])
    def test_railed_measurand_names_a_collapsed_series_resistance(self, build):
        reason = railed_measurand(build(self.COLLAPSED))
        assert reason, "an R0 of 4.6e-62 ohm was reported as a measurement"
        assert "R0" in reason
        # The value is carried so a stored row says *how* it railed, not merely that
        # it did — an R0 at 4.6e-62 and a C0 on its 1e-11 floor are different faults.
        assert "4.6e-62" in reason

    @pytest.mark.parametrize("build", [_legacy, _gated], ids=["legacy", "gated"])
    def test_demotion_removes_the_conductivity_not_only_the_success_flag(self, build):
        fit = build(self.COLLAPSED)
        assert _demote_if_railed(fit)
        assert fit.success is False
        assert "railed" in fit.error_msg
        # `DataStore.record_fit` derives sigma from `R1 and R1 > 0` and never consults
        # `success`, so clearing `success` alone would still store a conductivity.
        # NaN is what actually suppresses it.
        assert fit.R1 != fit.R1

    def test_the_railed_value_survives_as_a_diagnostic(self):
        fit = _legacy(self.COLLAPSED)
        _demote_if_railed(fit)
        # Demoted, not erased: `parameters_json` still records where it railed.
        assert fit.parameters[0] == pytest.approx(4.6e-62)


# ── The R1 behaviour that must not regress ───────────────────────────────────

class TestR1MessagesAreUnchanged:
    """Downstream readers tell a railed row from a non-converged one by this string,
    so widening *what* is detected must not move *what R₁ says*."""

    ON_FLOOR = [50.0, 1e-7, 0.7, 100.0, 1e-10]
    ALL_FIVE = [0.0, 1e-8, 0.4, 1e2, 1e-11]

    def test_legacy_r1_rail_still_names_the_model_and_the_bound_value(self):
        assert (railed_measurand(_legacy(self.ON_FLOOR))
                == "R1 rests on the 'simpleSalt' lower bound of 100 ohm")

    def test_gated_r1_rail_still_reports_the_original_phrase(self):
        assert railed_measurand(_gated(self.ON_FLOOR)) == "R1 rests on a fitted bound"

    @pytest.mark.parametrize("build", [_legacy, _gated], ids=["legacy", "gated"])
    def test_all_five_on_their_bounds_still_answers_for_r1_first(self, build):
        """The ``ch22_001`` shape. This case was *already* demoted before this change,
        because R₁ is one of the five and sits on its 100 Ω floor — so it is a
        regression pin, not a new catch. R₁ answers first on purpose: a reader
        grepping for the bound value must still find it."""
        reason = railed_measurand(build(self.ALL_FIVE))
        assert reason.startswith("R1 rests on")
        if build is _legacy:
            assert "100" in reason

    def test_a_model_declaring_no_bounds_can_still_never_rail(self):
        """``flexSalt`` has ``bounds = None``. Reporting a rail there would invent a
        constraint the optimiser never had — and the widened sweep must not do so
        by reading some *other* model's bounds."""
        assert not railed_measurand(_fit(self.ALL_FIVE, model_name="flexSalt"))


# ── The negative control ─────────────────────────────────────────────────────

class TestHealthyFitsAreUntouched:

    @pytest.mark.parametrize("build", [_legacy, _gated], ids=["legacy", "gated"])
    def test_a_healthy_fit_is_not_demoted(self, build):
        fit = build(HEALTHY)
        assert railed_measurand(fit) == ""
        assert _demote_if_railed(fit) == ""
        assert fit.success is True
        assert fit.error_msg == ""
        assert fit.R1 == pytest.approx(5.0e4)

    def test_a_resistance_a_decade_off_its_floor_is_set_by_the_data(self):
        """R1 at 5e4 against a floor of 100 is set by the data, not the constraint."""
        assert railed_measurand(_legacy(HEALTHY)) == ""


# ── What "at a bound" means when the bound is zero ───────────────────────────

class TestZeroBoundIsAbsoluteNotRelative:
    """A relative tolerance against a bound of exactly 0 reads ``|v| <= tol*|v|``,
    which is false for every value carrying a scale of its own. What survived that
    was not a threshold but an artefact: ``pegged`` clamped its scale at 1e-30, so
    its effective cut was ``tol * 1e-30 = 1e-33`` — three decades below the legacy
    floor and, since it multiplied ``tol``, a function of ``[gates] bound_tol``. The
    rule for a zero bound has to be absolute, and since 2026-08-31 ``pegged`` is
    where it is written."""

    @pytest.mark.parametrize("value", [0.0, 5e-324, 1e-62, 1e-31, COLLAPSED_AT_ZERO])
    def test_values_at_or_below_the_floor_are_collapsed(self, value):
        assert "R0" in railed_measurand(_legacy([value, 1e-7, 0.7, 5.0e4, 1e-10]))

    @pytest.mark.parametrize("value", [1e-29, 1e-12, 1.0, 50.0])
    def test_values_above_the_floor_are_left_alone(self, value):
        """Deliberately *not* caught. Catching these would mean choosing a physical
        floor for a series resistance, which is a new gate rather than a reading of
        the bound the optimiser was given — and the stored corpus offers no gap to
        put one in (89 rows in (1e-30, 1e-20], 918 in (1e-12, 1e-6])."""
        assert railed_measurand(_legacy([value, 1e-7, 0.7, 5.0e4, 1e-10])) == ""

    @pytest.mark.parametrize("value", [0.0, 5e-324, 1e-62, 1e-31, COLLAPSED_AT_ZERO])
    def test_both_paths_agree_at_the_floor(self, value):
        """The gated path used to be handed the floor as a *supplement* because
        ``pegged`` could not express it. The supplement is retired, so this is now the
        assertion that ``pegged`` carries the whole of it — if it regressed to the
        relative rule, the 1e-31 and 1e-30 cases would go red here first."""
        borderline = [value, 1e-7, 0.7, 5.0e4, 1e-10]
        assert "R0" in railed_measurand(_gated(borderline))
        assert "R0" in _cov(borderline).pegged()

    @pytest.mark.parametrize("value", [1e-29, 1e-12, 50.0])
    def test_both_paths_agree_above_the_floor(self, value):
        assert railed_measurand(_gated([value, 1e-7, 0.7, 5.0e4, 1e-10])) == ""

    def test_the_gated_path_agrees_against_the_bounds_it_actually_fits(self):
        """``_cov`` above hangs the *legacy registry's* bounds on a ``FitCovariance``,
        which the gated path never does — a fixture from a code state that never
        existed (``SUBAGENT_RULES`` §3.2). What the gated fitter really carries is
        ``set_default_bounds``: all-zero below, ``+inf`` above except the CPE
        exponent. Both resistances therefore have *only* a zero lower bound, so on
        the real bounds the zero-bound rule is the only rule there is."""
        from impedance.models.circuits.fitting import set_default_bounds

        production = set_default_bounds(CIRCUIT_MODELS["simpleSalt"]["circuit"])
        assert list(production[0]) == [0.0] * 5

        collapsed = _fit(TestSeriesResistanceCollapse.COLLAPSED,
                         covariance=_cov(TestSeriesResistanceCollapse.COLLAPSED,
                                         bounds=production))
        assert "R0" in railed_measurand(collapsed)
        healthy = _fit(HEALTHY, covariance=_cov(HEALTHY, bounds=production))
        assert railed_measurand(healthy) == ""


# ── The widened policy, stated as a test because it costs sigma ──────────────

class TestNuisanceParametersDoNotDemote:
    """The watch is ``R_series`` and ``R_bulk``, and stops there.

    Demoting on *any* pegged parameter was implemented first and is wrong. σ is
    ``K/R`` and ``R`` is built from those two alone; the CPE and ``C_par`` terms shape
    the electrode response and never enter it. ``FitCovariance.rel_se`` already draws
    the same line — "a nuisance parameter may legitimately be poorly determined, but
    the measurand may not".

    The evidence is not merely tidiness. The blocking-electrode CPE runs to its
    constraint on **ordinary** spectra, so the wider rule demoted every channel of the
    settle-phase fixtures: 14 tests in ``test_eis_validate.py`` went red, σ went null
    board-wide, the survivor set fell under ``DEFAULT_SETTLE_MIN_CHANNELS`` and the
    verdict moved from ``ceiling`` to ``not_evaluable``. On the stored corpus the wide
    rule demotes 539 of 3 618 fits (14.9 %) where the defect is 222. A gate that fires
    on the normal condition certifies nothing."""

    def test_a_capacitance_on_its_floor_does_not_demote_the_fit(self):
        fit = _legacy([50.0, 1e-7, 0.7, 5.0e4, 1e-11])
        assert railed_measurand(fit) == ""
        assert _demote_if_railed(fit) == ""
        assert fit.R1 == pytest.approx(5.0e4)

    def test_a_cpe_at_both_of_its_constraints_does_not_demote_the_fit(self):
        """The exact shape that turned the settle suite red: ``CPE0_0`` on its 9e-6
        ceiling and ``CPE0_1`` on its 0.9 ceiling, with both resistances healthy."""
        assert railed_measurand(_legacy([50.0, 9e-6, 0.9, 5.0e4, 1e-10])) == ""

    def test_a_series_collapse_is_still_caught_beside_railed_nuisance_terms(self):
        """Narrowing the watch must not lose the defect it was widened for."""
        reason = railed_measurand(_legacy([0.0, 9e-6, 0.9, 5.0e4, 1e-10]))
        assert "R0" in reason
        # ... and it says nothing about the CPE, which is not being judged.
        assert "CPE" not in reason


# ── End to end, through the surface the campaign reads ───────────────────────

def test_record_fit_stores_no_conductivity_for_a_collapsed_series_resistance(tmp_path):
    """``record_fit`` knows nothing about railing, so this is the assertion that the
    demotion actually reaches the database rather than stopping at the report."""
    from softae.core.data_store import DataStore
    from tests.eis_synthetic import as_eis_result, reference_spectrum

    store = DataStore(tmp_path / "project")
    try:
        run_id = store.start_run("r0-collapse")
        eis = as_eis_result(*reference_spectrum())
        measurement_id = store.record_measurement(run_id, eis)

        fit = _legacy([4.6e-62, 1e-7, 0.7, 5.0e4, 1e-10])
        _demote_if_railed(fit)
        fit_id = store.record_fit(measurement_id, fit, L_cm=0.2, t_cm=0.015, w_cm=0.2)
        row = dict(store._conn.execute(
            "SELECT success, sigma_S_per_cm, R1, error_msg FROM fit_results "
            "WHERE fit_id = ?", (fit_id,)).fetchone())
    finally:
        store.close()

    assert row["success"] == 0
    assert row["sigma_S_per_cm"] is None
    assert row["R1"] is None
    assert "R0" in row["error_msg"]
