"""The fitter pre-gate — reading ``arc_closure`` *before* the fit instead of after.

Two routes behind two independent flags, and the distinction between them is the whole
subject of this file:

``budget_cap`` (A)
    Bounds how long the **same** estimator may run. It cannot move a reported number,
    and that is provable rather than hopeful: a bounded run is a strict *prefix* of the
    unbounded one's trajectory, so a fit that was going to exhaust 20 000 evaluations
    exhausts 2 000 instead, declines identically, and falls back identically.
``two_point_open`` (B)
    Changes **which** estimator produces ``R1`` on the open population, and therefore
    changes the number. Operator-authorized, epoch-grade (schema epoch 5), labelled on
    the row, and contained to open arcs — the containment being the property
    :class:`TestClosedArcsAreUntouched` exists to pin.

**Neither is a skip-on-OPEN, and no test here may be written as though it were.**
``arc.py`` says outright that nothing there demotes a fit, because refusing the open
third *"would throw away most of the cold end of every temperature sweep"* — 8 % open at
the hot, wet end against 73 % at the cold end, which is exactly where an Arrhenius slope
has its leverage. Every spectrum still produces an ``R1``.
"""

from __future__ import annotations

import ast
import inspect
import tomllib
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis import engine as engine_mod
from softae.analysis.eis.arc import CLOSED, OPEN, UNKNOWN, ArcClosure, arc_closure
from softae.analysis.eis.engine import TWO_POINT, _two_point_fit, analyze_spectrum
from softae.analysis.eis.engine_support import (
    PregateSettings,
    blocking_open,
    pregate_settings,
)
from softae.analysis.eis.geometry import CellConstant
from softae.analysis.eis.settings import EISSettings, GateSettings
from tests.eis_synthetic import as_eis_result, reference_spectrum

REPO = Path(__file__).resolve().parents[1]

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")


def _gated(enabled: bool = False) -> EISSettings:
    """The engine the pre-gate lives on. Gates observing, as the rig ships them."""
    return EISSettings(engine="gated", gates=GateSettings(enabled=enabled))


# ── Spectra ──────────────────────────────────────────────────────────────────

def debye(R_series: float = 50.0, R_bulk: float = 1.0e5, tau: float = 1.0e-4,
          f_lo: float = 20.0, f_hi: float = 2.0e5, npts: int = 25):
    """An **exact** ideal Debye arc — the one shape the two-point read must nail.

    Not :func:`tests.eis_synthetic.reference_spectrum`, which carries a CPE and is
    therefore *depressed*: a circle centred on the real axis is then the wrong model by
    construction, and a test built on it could only ever assert a tolerance nobody could
    justify. Here the estimator's assumption is exactly true, so the assertion is
    equality and any failure is arithmetic rather than modelling.
    """
    f = np.logspace(np.log10(f_hi), np.log10(f_lo), npts)
    Z = R_series + R_bulk / (1.0 + 1j * 2.0 * np.pi * f * tau)
    return f, Z


def blocking_tail(R_series: float = 500.0, C: float = 1e-9,
                  f_lo: float = 20.0, f_hi: float = 2.0e5, npts: int = 25):
    """A series RC: no arc at all, ``−Z″ ∝ f⁻¹``, phase pinned near −90° at the floor.

    The population (A) diverts and (B) must **decline** on — there is no circle here to
    read, and the two lowest points sit on a near-vertical tail.
    """
    f = np.logspace(np.log10(f_hi), np.log10(f_lo), npts)
    return f, R_series + 1.0 / (1j * 2.0 * np.pi * f * C)


def open_debye(R_series: float = 50.0, R_bulk: float = 1.0e5, tau: float = 1.0e-1,
               f_lo: float = 20.0, f_hi: float = 2.0e5, npts: int = 25):
    """An ideal arc whose apex (≈1.6 Hz) sits **below** the sweep floor.

    Open, steeply capacitive at the floor (≈ −85°), and still a genuine circle — so it
    is the one fixture on which the two-point read actually *fires* end to end. The
    blocking tail declines, which is right but makes it useless for testing what a
    successful divert does.
    """
    return debye(R_series, R_bulk, tau, f_lo, f_hi, npts)


def _arc_of(f, Z) -> ArcClosure:
    eis = as_eis_result(f, Z)
    return arc_closure(eis.frequency, eis.z_imag_neg, eis.phase)


# ── The defaults ship off; the config arms exactly one route ─────────────────

class TestThePregateShipsInert:
    def test_pregate_settings_default_to_both_routes_off(self):
        """The dataclass default is the unreadable-config fallback, and stays off.

        Unchanged by the 2026-08-19 arming of ``budget_cap`` in the file: a config a
        campaign cannot parse must leave the engine on the route it takes today.
        """
        cfg = PregateSettings()
        assert cfg.budget_cap is False
        assert cfg.two_point_open is False
        assert cfg.engaged is False

    def test_the_shipped_config_arms_the_budget_cap_and_nothing_else(self):
        """The file on disk, not the dataclass default — they are different claims.

        ``budget_cap`` was armed **2026-08-19** on the evidence of 80 stored spectra:
        ``R1`` bit-identical on **80/80**, worst relative difference **0.000e+00**, for
        ≈3.4× off the analysis path. It cannot move a number because on the population
        it targets the covariance fit exhausts its ``nfev`` ceiling and is discarded
        100 % of the time regardless — the capped run returns the same ``None`` and the
        same ``fit_circuit`` fallback.

        ``two_point_open`` stays false: it is the flag that deliberately *does* move
        ``R1`` (schema epoch 5), and the one-at-a-time rule is the point of this test.
        """
        config = tomllib.loads(
            (REPO / "softae_config.toml").read_text(encoding="utf-8"))
        table = config["eis"]["pregate"]
        assert table["budget_cap"] is True
        assert table["two_point_open"] is False
        # …and the severity threshold is the one the measurement chose, not OPEN alone.
        assert table["phase_low_max_deg"] == -60.0

    def test_the_two_routes_are_separately_flippable(self):
        """[a57] §3's argument, applied here: two causes need two switches.

        Armed together, a moved ``R1`` could be the cheaper fit route or the changed
        estimator and the stored spectrum cannot say which. The type must permit either
        one alone.
        """
        assert PregateSettings(budget_cap=True).two_point_open is False
        assert PregateSettings(two_point_open=True).budget_cap is False
        assert PregateSettings(budget_cap=True).engaged
        assert PregateSettings(two_point_open=True).engaged

    def test_an_unparseable_threshold_falls_back_instead_of_raising(self):
        """Mirrors ``eis_settings``: a typo must not stop a campaign."""
        cfg = pregate_settings({"phase_low_max_deg": "very negative",
                                "max_nfev": None})
        assert cfg.phase_low_max_deg == PregateSettings().phase_low_max_deg
        assert cfg.max_nfev == PregateSettings().max_nfev

    def test_an_empty_table_is_the_shipped_off_state(self):
        assert pregate_settings({}).engaged is False

    def test_a_disabled_pregate_does_not_even_read_the_arc(self):
        """Off is *free*, not merely cheap.

        The pre-fit ``arc_closure`` call sits behind ``engaged`` so that a rig with the
        flags down pays nothing at all — which is what makes "default off" a statement
        about cost as well as about behaviour. ``annotate_arc_closure`` still runs after
        the fit exactly as it always has; it reaches ``arc_closure`` through ``arc.py``'s
        own namespace, so patching the name *here* isolates the pre-gate's call.
        """
        calls: list[int] = []
        real = engine_mod.arc_closure

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        eis = as_eis_result(*debye())
        engine_mod.arc_closure = counting
        try:
            analyze_spectrum(eis, cell=CELL, settings=_gated(),
                             pregate=PregateSettings())
            assert calls == []
            analyze_spectrum(eis, cell=CELL, settings=_gated(),
                             pregate=PregateSettings(budget_cap=True))
            assert len(calls) == 1
        finally:
            engine_mod.arc_closure = real


# ── The predicate ────────────────────────────────────────────────────────────

class TestBlockingOpenIsSeverityNotJustVerdict:
    """``state == OPEN`` alone is the wrong discriminator, and this is why.

    Measured over ``20260811T023757Z_equilibration_characterization``: of 40 open arcs,
    the 28 with ``phase_low_deg ≤ −60°`` cost a median 49.1 s and only 2 of them
    converge, while the 12 above −60° converge 9 times in 12 at a median of 0.48 s.
    Diverting on the verdict alone would spend the whole change on that second group
    for no saving — and, under (B), would move their ``R1`` for nothing.
    """

    @pytest.mark.parametrize("state,phase,expected", [
        (OPEN, -92.0, True),
        (OPEN, -81.7, True),
        (OPEN, -60.0, True),               # the boundary is inclusive
        (OPEN, -31.1, False),              # measured: converges in 0.05 s
        (OPEN, -5.0, False),
        (CLOSED, -92.0, False),            # closed is never diverted, however steep
        (UNKNOWN, -92.0, False),
    ])
    def test_both_conditions_are_required(self, state, phase, expected):
        arc = ArcClosure(state, 20.0, 20.0, phase)
        assert blocking_open(arc, PregateSettings()) is expected

    def test_a_spectrum_with_no_phase_takes_the_route_it_takes_today(self):
        """Severity unknown ⇒ do not divert. Conservative, and chosen rather than
        defaulted into: ``phase_low_deg`` is NaN whenever the caller supplied no phase,
        and condition 2 is precisely the one that cannot be evaluated then."""
        assert blocking_open(ArcClosure(OPEN, 20.0, 20.0, float("nan")),
                             PregateSettings()) is False

    def test_the_threshold_is_configurable_and_moves_the_cut(self):
        arc = ArcClosure(OPEN, 20.0, 20.0, -45.0)
        assert blocking_open(arc, PregateSettings()) is False
        assert blocking_open(arc, PregateSettings(phase_low_max_deg=-40.0)) is True

    def test_the_predicate_reads_no_acquisition_field(self):
        """The scout's interior-apex fields answer a different question and are barred.

        ``f_apex_interior_hz`` / ``apex_prominence_rel`` / ``band_below_apex_decades``
        exist to plan *where the next sweep should put its points*. What this predicate
        asks is what the optimiser will do with the sweep already in hand. Reaching for
        them here would couple two changes that are deliberately flagged apart, so the
        guard is on the source rather than on the behaviour.
        """
        tree = ast.parse(inspect.getsource(blocking_open))
        # Both spellings: `arc.state` reads as an Attribute, `getattr(arc, "state")`
        # as a string constant, and a guard that saw only the first would be blind to
        # exactly the form this function uses.
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {a.value for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr"
                  for a in n.args[1:2] if isinstance(a, ast.Constant)
                  and isinstance(a.value, str)}
        assert {"state", "phase_low_deg"} <= names
        assert not names & {"f_apex_interior_hz", "apex_prominence_rel",
                            "band_below_apex_decades", "f_peak_hz"}

    def test_the_predicate_is_pure_and_calls_no_gate_machinery(self):
        """It runs per spectrum on the fit path; it may not reach the K–K ladder.

        ``run_gates`` at ``kk_max_M = 50`` and the O(n²) plateau search are the
        expensive Front-1 chain, and a pre-gate that called them would cost more than
        the fit it is trying to avoid.
        """
        src = inspect.getsource(blocking_open)
        for forbidden in ("run_gates", "kk_", "instrument_envelope", "fit_"):
            assert forbidden not in src


# ── (A) the budget cap ───────────────────────────────────────────────────────

class TestBudgetCapBoundsEffortWithoutMovingANumber:
    def test_fit_spectrum_without_a_budget_makes_the_call_it_always_made(self):
        """The additive-only proof for ``fitter.py``.

        ``max_nfev=None`` must not become ``max_nfev=DEFAULT_MAX_NFEV`` at the call
        site: restating the default would be a second copy of it, and the claim that
        this keyword changes nothing rests on the argument simply not being passed.
        """
        from softae.analysis.eis import fitter as fitter_mod

        seen: list[dict] = []
        real = fitter_mod.fit_with_covariance

        def spy(*args, **kwargs):
            seen.append(dict(kwargs))
            return real(*args, **kwargs)

        fitter_mod.fit_with_covariance = spy
        try:
            fitter_mod.fit_spectrum(as_eis_result(*debye()), "simpleSalt")
            assert "max_nfev" not in seen[0]
            seen.clear()
            fitter_mod.fit_spectrum(as_eis_result(*debye()), "simpleSalt",
                                    max_nfev=250)
            assert seen[0]["max_nfev"] == 250
        finally:
            fitter_mod.fit_with_covariance = real

    def test_the_cap_reaches_the_fitter_only_on_the_blocking_open_population(self):
        import softae.analysis.eis.fitter as fitter_mod

        seen: list[object] = []
        real_fit = fitter_mod.fit_spectrum          # bound BEFORE the patch, or the
                                                    # spy re-imports itself and recurses

        def spy(eis_result, model_name, *, max_nfev=None):
            seen.append(max_nfev)
            return real_fit(eis_result, model_name, max_nfev=max_nfev)

        fitter_mod.fit_spectrum = spy
        try:
            armed = PregateSettings(budget_cap=True)
            analyze_spectrum(as_eis_result(*blocking_tail()), cell=CELL,
                             settings=_gated(), pregate=armed)
            assert seen == [armed.max_nfev]          # diverted: bounded
            seen.clear()
            analyze_spectrum(as_eis_result(*debye()), cell=CELL,
                             settings=_gated(), pregate=armed)
            assert seen == [None]                    # closed arc: untouched
        finally:
            fitter_mod.fit_spectrum = real_fit

    def test_a_capped_spectrum_reports_the_same_r1_as_an_uncapped_one(self):
        """(A)'s acceptance criterion, on the population it actually diverts.

        Replayed over 40 open and 40 closed stored spectra the agreement was **exact**
        — 0.000e+00 worst relative difference — because the capped fit declines exactly
        where the uncapped one did and both then take the same fallback. Here the same
        equality is asserted end to end rather than inside the fitter.
        """
        eis = as_eis_result(*blocking_tail())
        off = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                               pregate=PregateSettings())
        capped = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                                  pregate=PregateSettings(budget_cap=True))
        assert np.isnan(off.fit.R1) == np.isnan(capped.fit.R1)
        if not np.isnan(off.fit.R1):
            assert capped.fit.R1 == off.fit.R1
        assert capped.fit.success == off.fit.success
        # …and it is still the CPE route's answer, not a different estimator's.
        assert getattr(capped.fit, "estimator", None) is None

    def test_the_cap_never_makes_a_spectrum_unanalysable(self):
        """Not a skip: a diverted spectrum still comes back with a fit object."""
        report = analyze_spectrum(as_eis_result(*blocking_tail()), cell=CELL,
                                  settings=_gated(),
                                  pregate=PregateSettings(budget_cap=True))
        assert report.fit is not None
        assert report.quality is not None


# ── (B) the two-point read ───────────────────────────────────────────────────

class TestTwoPointReadIsAClosedFormCircle:
    def test_it_recovers_the_diameter_of_an_exact_debye_arc(self):
        f, Z = debye(R_series=50.0, R_bulk=1.0e5, tau=1.0e-4)
        fit = _two_point_fit(as_eis_result(f, Z), "simpleSalt")
        assert fit is not None
        assert fit.R1 == pytest.approx(1.0e5, rel=1e-6)
        assert fit.R0 == pytest.approx(50.0, rel=1e-3)

    def test_it_recovers_the_diameter_from_a_truncated_arc_too(self):
        """The case it exists for: the arc's low-frequency end was never measured."""
        f, Z = debye(R_bulk=1.0e5, tau=1.0e-4, f_lo=5.0e3, f_hi=2.0e5)
        fit = _two_point_fit(as_eis_result(f, Z), "simpleSalt")
        assert fit is not None
        assert fit.R1 == pytest.approx(1.0e5, rel=1e-6)

    def test_sweep_order_does_not_change_the_answer(self):
        """The instrument sweeps high→low; nothing may depend on that."""
        f, Z = debye()
        down = _two_point_fit(as_eis_result(f, Z), "simpleSalt")
        up = _two_point_fit(as_eis_result(f[::-1], Z[::-1]), "simpleSalt")
        assert up is not None and down is not None
        assert up.R1 == pytest.approx(down.R1, rel=1e-12)

    def test_it_declines_on_a_blocking_tail_rather_than_inventing_a_circle(self):
        """[p35]'s complaint about the CPE fitter was that *"none declined"*.

        On a series RC there is no arc, the two lowest points sit on a near-vertical
        tail, and the circle through them is enormous and meaningless. Returning a
        confident number there is the failure [a53] names — biased ``R1`` flowing
        through unlabelled — so the physical guards refuse instead.
        """
        assert _two_point_fit(as_eis_result(*blocking_tail()), "simpleSalt") is None

    def test_it_declines_when_the_series_resistance_would_be_negative(self):
        f, Z = debye(R_series=50.0, R_bulk=1.0e5)
        Z = Z - 60.0                       # shove the arc left of the origin
        assert _two_point_fit(as_eis_result(f, Z), "simpleSalt") is None

    def test_it_declines_on_an_inductive_or_flat_low_frequency_pair(self):
        f = np.array([2.0e5, 1.0e5, 5.0e4, 2.0e4, 1.0e4])
        Z = np.array([100 + 0j, 100 + 0j, 100 + 0j, 100 + 0j, 100 + 0j])
        assert _two_point_fit(as_eis_result(f, Z), "simpleSalt") is None

    def test_a_declining_read_leaves_the_ordinary_route_in_charge(self):
        """A decline may never cost a spectrum its fit — it falls through, silently."""
        eis = as_eis_result(*blocking_tail())
        off = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                               pregate=PregateSettings())
        armed = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                                 pregate=PregateSettings(two_point_open=True))
        assert getattr(armed.fit, "estimator", None) is None
        assert np.isnan(off.fit.R1) == np.isnan(armed.fit.R1)


class TestClosedArcsAreUntouched:
    """**The containment guarantee.** (B) may only ever act on the open population.

    This is the test that makes the epoch statement true: rows either side of the flip
    are comparable only among themselves *on the open population*, and the closed
    population is continuous across it. If this ever fails, the epoch note in
    ``SCHEMA_EPOCHS`` is a false statement about the database.
    """

    @pytest.mark.parametrize("spectrum", ["ideal", "depressed"])
    def test_a_closed_arc_reports_the_identical_r1_with_two_point_armed(self, spectrum):
        # Both an ideal arc and a *depressed* one (a CPE in the parallel branch, which
        # is what real films give), because "closed" is the containment boundary and it
        # must hold for arcs the two-point circle would model badly if it ever saw one.
        # The blocking CPE is switched off — with it the low-frequency tail owns the
        # global −Z″ maximum and the sweep reads OPEN, which is a different fixture.
        f, Z = (debye() if spectrum == "ideal"
                else reference_spectrum(Q=0.0, C_par_exponent=0.9))
        eis = as_eis_result(f, Z)
        assert _arc_of(f, Z).state == CLOSED, "fixture must be a closed arc"

        off = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                               pregate=PregateSettings())
        armed = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                                 pregate=PregateSettings(two_point_open=True,
                                                         budget_cap=True))
        assert armed.fit.R1 == off.fit.R1                 # bit-for-bit
        assert armed.fit.R0 == off.fit.R0
        assert armed.sigma.mode == off.sigma.mode
        assert armed.sigma.value == off.sigma.value
        assert getattr(armed.fit, "estimator", None) is None

    def test_the_route_is_never_taken_on_a_closed_arc_however_capacitive_the_floor(self):
        arc = ArcClosure(CLOSED, 20.0, 20.0, -89.9)
        assert blocking_open(arc, PregateSettings(two_point_open=True)) is False

    def test_the_pregate_and_the_stored_arc_state_judge_the_same_array(self):
        """**The containment guarantee stated precisely**, and it is not about the
        spectrum as measured — it is about the spectrum as *fitted*.

        Front-1 can drop low-frequency points even while merely observing, so a sweep
        whose raw arc closed can arrive at the fitter with the apex outside what
        survives. Replaying stored spectra, exactly one of 40 raw-closed sweeps did
        that and was diverted. That is **correct**: the fit sees ``surviving``, and so
        must the decision — judging the raw sweep would credit a fit with a point a
        gate had already removed, which is the same argument the comment above
        ``annotate_arc_closure`` already makes.

        What it means is that "closed arcs are untouched" has to be read against the
        column the database actually stores. Both reads take the same object, so a row
        can never claim ``arc_state = 'closed'`` and carry a two-point ``R1``.
        """
        src = inspect.getsource(engine_mod.analyze_spectrum)
        assert "arc_closure(surviving.frequency, surviving.z_imag_neg," in src
        assert "annotate_arc_closure(fit, surviving)" in src

    def test_a_diverted_row_always_stores_an_open_arc_state(self, tmp_path):
        """The same invariant, end to end and through the database.

        Not vacuous: the fixture is asserted to produce a labelled row, so a change
        that quietly stopped the pre-gate firing would fail here rather than pass by
        never testing anything.
        """
        from softae.core.data_store import DataStore

        eis = as_eis_result(*open_debye())
        report = analyze_spectrum(eis, cell=CELL, settings=_gated(),
                                  pregate=PregateSettings(two_point_open=True))
        assert report.fit.estimator == TWO_POINT, "fixture no longer diverts"

        store = DataStore(tmp_path / "project")
        try:
            run_id = store.start_run("pregate_containment")
            mid = store.record_measurement(run_id, eis)
            store.record_fit(mid, report.fit, report=report)
            row = store.query_fits(measurement_id=mid)[0]
            assert row["engine"] == "gated_two_point"
            assert row["arc_state"] == OPEN
        finally:
            store.close()

    def test_the_two_point_read_recovers_the_true_diameter_of_that_open_arc(self):
        """And the number it stores is right, on the one case where truth is known.

        The apex is below the sweep floor, so the CPE fitter has no in-band feature to
        anchor on — [p35]'s regime exactly. The circle through two measured points
        still has the arc's geometry and returns it.
        """
        fit = _two_point_fit(as_eis_result(*open_debye()), "simpleSalt")
        assert fit is not None
        assert fit.R1 == pytest.approx(1.0e5, rel=1e-6)


class TestTheEstimatorIsLabelledOnTheRow:
    """[a53] §1: the failure mode on this rig is a biased ``R1`` flowing **unlabelled**.

    A date cannot answer which estimator produced a row — the flag is
    per-configuration, so one database can hold both kinds written the same afternoon.
    The label has to travel with the row.
    """

    def test_a_two_point_fit_carries_the_estimator_name(self):
        fit = _two_point_fit(as_eis_result(*debye()), "simpleSalt")
        assert fit.estimator == TWO_POINT

    def test_an_ordinary_fit_carries_no_estimator_and_so_no_row_changes(self):
        from softae.analysis.eis.fitter import fit_spectrum

        assert getattr(fit_spectrum(as_eis_result(*debye()), "simpleSalt"),
                       "estimator", None) is None

    def test_record_fit_stamps_the_estimator_into_the_engine_column(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            eis = as_eis_result(*debye())
            run_id = store.start_run("pregate_label")
            mid = store.record_measurement(run_id, eis)

            plain = _two_point_fit(eis, "simpleSalt")
            store.record_fit(mid, plain)
            assert store.query_fits(measurement_id=mid)[0]["engine"] == "two_point"

            from softae.analysis.eis.report import SigmaReport, SpectrumReport

            mid2 = store.record_measurement(run_id, eis)
            store.record_fit(mid2, _two_point_fit(eis, "simpleSalt"),
                             report=SpectrumReport(engine="gated",
                                                   sigma=SigmaReport()))
            assert store.query_fits(
                measurement_id=mid2)[0]["engine"] == "gated_two_point"
        finally:
            store.close()

    def test_an_unlabelled_fit_stores_exactly_what_it_always_stored(self, tmp_path):
        """The no-regression half: every row written before this change is untouched."""
        from softae.analysis.circuit_fitting import FitResult
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            run_id = store.start_run("pregate_label_off")
            mid = store.record_measurement(run_id, as_eis_result(*debye()))
            store.record_fit(mid, FitResult(
                model_name="simpleSalt", parameters=np.array([50.0, 1e5]),
                R0=50.0, R1=1e5, R0_guess=50.0, R1_guess=1e5, z_indices=[0, 1]))
            assert store.query_fits(measurement_id=mid)[0]["engine"] == "legacy"
        finally:
            store.close()


class TestSchemaEpochFive:
    """T2.3's precedent, applied. ``R1`` stays a REAL in ohms and means something else.

    That is version 2's situation — ``deposit_area_mm2`` changed *derivation* while the
    column held still — and not version 3's, where the numbers held still and the names
    moved. The ``kind`` column is what tells those apart, so it has to be right.
    """

    def test_epoch_five_is_a_data_epoch_naming_population_and_authorisation(self):
        from softae.core.data_store import SCHEMA_EPOCHS

        version, kind, note = SCHEMA_EPOCHS[-1]
        assert version == 5
        assert kind == "data-epoch"
        assert "2026-08-18" in note
        assert "R1" in note
        for claim in ("OPEN-ARC", "two_point_open", "NO BACKFILL",
                      "comparable only among", "gated_two_point"):
            assert claim in note, f"the epoch note does not state: {claim}"

    def test_the_versions_are_unique_and_ordered(self):
        from softae.core.data_store import SCHEMA_EPOCHS

        versions = [v for v, _, _ in SCHEMA_EPOCHS]
        assert versions == sorted(versions) == list(range(1, len(versions) + 1))

    def test_a_fresh_store_seeds_the_epoch_and_reseeding_is_a_no_op(self, tmp_path):
        """``INSERT OR IGNORE`` is idempotent *and* append-only in one stroke —
        ``applied_at`` records when this database first learned of the epoch, which a
        rewriting seeder would destroy."""
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            row = store._conn.execute(
                "SELECT version, kind, applied_at FROM schema_version "
                "WHERE version = 5").fetchone()
            assert row is not None and row[1] == "data-epoch"
            first_seen = row[2]
        finally:
            store.close()

        again = DataStore(tmp_path / "project")
        try:
            assert again._conn.execute(
                "SELECT applied_at FROM schema_version WHERE version = 5"
            ).fetchone()[0] == first_seen
            assert again._conn.execute(
                "SELECT COUNT(*) FROM schema_version WHERE version = 5"
            ).fetchone()[0] == 1
        finally:
            again.close()

    def test_the_epoch_carries_no_migration_and_backfills_nothing(self):
        """No historical ``R1`` is recomputed, and there is no code that could.

        Inventing recomputed values would manufacture exactly the false comparability
        the ledger exists to prevent — the argument
        ``_migrate_experiment_skipped_channels`` already makes for leaving NULL alone.
        """
        source = (REPO / "src" / "softae" / "core"
                  / "data_store.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        migrations = [n.name for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name.startswith("_migrate")]
        assert not any("two_point" in name or "estimator" in name
                       for name in migrations)
        assert "UPDATE fit_results" not in source

    def test_the_version_ledger_is_still_seeded_last(self):
        """It records epochs the migrations above have just finished establishing,
        so it must not claim them before they hold."""
        tree = ast.parse((REPO / "src" / "softae" / "core"
                          / "data_store.py").read_text(encoding="utf-8"))
        init = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        migrations = [n.func.attr for n in ast.walk(init)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr.startswith("_migrate")]
        assert migrations[-1] == "_migrate_schema_version"


class TestTheEngineResolutionPointIsStillSingular:
    def test_the_pregate_does_not_become_a_second_engine_decision(self):
        """[a23] is a user ruling: one resolver, no per-surface opinions.

        ``pregate`` is an override in the same shape as ``gates`` and ``envelope`` — it
        selects a *route within* the gated engine and never which engine runs, so it
        must not read ``[eis] engine`` or accept an engine name.
        """
        src = inspect.getsource(engine_mod.analyze_spectrum)
        tree = ast.parse(inspect.getsource(pregate_settings))
        assert "engine" not in {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        # One resolution of the engine, still, and it is the existing line.
        assert src.count("cfg.engine") == 1

    def test_the_pregate_defaults_to_none_so_config_governs(self):
        assert inspect.signature(
            analyze_spectrum).parameters["pregate"].default is None

    def test_the_legacy_engine_never_reaches_the_pregate(self):
        """Legacy is bit-for-bit what the rig always did, and it stays that way."""
        report = analyze_spectrum(as_eis_result(*blocking_tail()), cell=CELL,
                                  engine="legacy",
                                  pregate=PregateSettings(two_point_open=True,
                                                          budget_cap=True))
        assert report.engine == "legacy"
        assert getattr(report.fit, "estimator", None) is None
