"""``sigma_observation``: one SpectrumReport → value / lower bound / upper bound (R3, R7).

Reports are built from the real ``SigmaReport`` / ``SpectrumReport`` constructors; the
last section runs the engine itself so the mapping is checked on a report the production
path builds, not only on hand-made ones (``SUBAGENT_RULES`` §3.2).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from softae.analysis.eis import observation as ob
from softae.analysis.eis.report import (
    REGIME_A_FIT_RB,
    REGIME_A_FOOT_LOWER,
    REGIME_A_PASSIVE,
    REGIME_B_ARC_LOWER,
    REGIME_B_FILM_ARC,
    REGIME_B_NO_ARC,
    SigmaReport,
    SpectrumReport,
)

NAN = float("nan")


def _unstamped_report(ok: bool = True, **sigma) -> SpectrumReport:
    quality = None if ok else SimpleNamespace(ok=False)
    return SpectrumReport(engine="gated", sigma=SigmaReport(**sigma), quality=quality)


def _report(ok: bool = True, **sigma) -> SpectrumReport:
    """A report as the engine stamps it with ``[eis] regime_aware`` on (the armed rules);
    the flag-off rollback is exercised through :func:`_unstamped_report`."""
    return _unstamped_report(ok=ok, **{"regime_aware": True, **sigma})


#: Route-A states as the engine builds them (``engine._regime_a_sigma``).
A_VALUE = dict(mode="value", value=2e-4, R_reported_ohm=1e5, R_basis=REGIME_A_FIT_RB,
               regime="A", regime_mode="value", regime_sigma=2e-4,
               regime_sigma_lower=1e-4, regime_R_foot_ohm=2e5, regime_active=True)
A_BOUND = dict(mode="bound", upper_bound=5e-4, upper_bound_basis=REGIME_A_PASSIVE,
               R_reported_ohm=4e4, R_basis=REGIME_A_PASSIVE, regime="A",
               regime_mode="bound", regime_sigma=5e-4, regime_sigma_lower=1e-4,
               regime_R_foot_ohm=2e5, regime_active=True)
A_LOWER = dict(mode="unavailable", R_reported_ohm=2e5, R_basis=REGIME_A_FOOT_LOWER,
               regime="A", regime_mode="unavailable", regime_sigma_lower=1e-4,
               regime_R_foot_ohm=2e5, regime_active=True)
LEGACY_BOUND = dict(mode="bound", upper_bound=3e-7, upper_bound_basis="loss_at_numerator",
                    R_reported_ohm=8e5, R_basis="split_bulk", regime="A",
                    regime_sigma_lower=1e-4, regime_R_foot_ohm=2e5)


# ── The mapping, row by row ─────────────────────────────────────────────────────


def test_sigma_observation_value_admitted_with_r_basis():
    o = ob.sigma_observation(_report(**A_VALUE))
    assert (o.kind, o.reported, o.lower) == (ob.VALUE, 2e-4, None)
    assert (o.basis, o.R_film_ohm, o.R_film_basis) == (REGIME_A_FIT_RB, 1e5, REGIME_A_FIT_RB)
    assert o.admitted and o.classified and o.regime == "A"


def test_sigma_observation_legacy_value_flag_off_admitted():
    o = ob.sigma_observation(_report(mode="value", value=1e-5, R_reported_ohm=3e6))
    assert o.kind == ob.VALUE and o.admitted and o.basis == "split_bulk"
    assert not o.classified and o.regime == ""


# ── Regime U, split by reason (F3 as narrowed by the operator, 2026-10-01) ─────────

U_VALUE = dict(mode="value", value=1.8e-4, R_reported_ohm=3e5, regime="U")
U_LEGACY_BOUND = {**LEGACY_BOUND, "regime": "U"}


@pytest.mark.parametrize("reason", [*ob.U_BAD_DATA_REASONS, "", "a_reason_nobody_filed"])
def test_sigma_observation_u_bad_data_value_recorded_not_admitted_not_measured(reason):
    """Bad-data U — and, fail safe, a missing or unknown reason: no tell, counts to a park."""
    o = ob.sigma_observation(_report(**U_VALUE, regime_reason=reason))
    assert (o.kind, o.reported, o.regime, o.regime_reason) == (ob.VALUE, 1.8e-4, "U", reason)
    assert not o.admitted and not o.classified
    assert not o.counts_as_measured


@pytest.mark.parametrize("reason", ob.U_SHAPE_REASONS)
def test_sigma_observation_u_shape_value_recorded_not_admitted_but_measured(reason):
    """Ruling 2026-10-02: an unrecognised-shape U's value is recorded, never told, and
    the spectrum still counts as measured (feasible, never parks)."""
    o = ob.sigma_observation(_report(**U_VALUE, regime_reason=reason))
    assert (o.kind, o.reported, o.regime_reason) == (ob.VALUE, 1.8e-4, reason)
    assert not o.admitted
    assert o.classified and o.counts_as_measured


@pytest.mark.parametrize("reason", ob.U_SHAPE_REASONS)
def test_sigma_observation_u_shape_bound_or_nothing_measured_not_admitted(reason):
    bound = ob.sigma_observation(_report(**U_LEGACY_BOUND, regime_reason=reason))
    assert bound.kind == ob.UPPER_BOUND and not bound.admitted
    assert bound.classified and bound.counts_as_measured
    nothing = ob.sigma_observation(_report(mode="unavailable", regime="U",
                                           regime_reason=reason))
    assert nothing.kind is None and nothing.classified and nothing.counts_as_measured


@pytest.mark.parametrize("reason", [*ob.U_SHAPE_REASONS, *ob.U_BAD_DATA_REASONS])
def test_sigma_observation_u_not_ok_value_not_admitted_classification_unchanged(reason):
    o = ob.sigma_observation(_report(ok=False, **U_VALUE, regime_reason=reason))
    assert o.kind == ob.VALUE and not o.admitted
    assert o.classified is (reason in ob.U_SHAPE_REASONS)


def test_sigma_observation_u_reason_families_cover_every_classifier_u_token():
    """The two families partition every U reason ``classify_regime`` can emit.

    The tokens are retyped (``regime.py`` has no constants for them), so this reads the
    classifier's own source: a new U reason fails here until someone files it — and until
    then the code treats it as bad data, the fail-safe side.
    """
    import re

    from softae.analysis.eis import regime as rg

    src = Path(rg.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'(?:RegimeVerdict|verdict)\(U,\s*"([a-z_]+)"', src))
    shape, bad = set(ob.U_SHAPE_REASONS), set(ob.U_BAD_DATA_REASONS)
    assert len(emitted) >= 8 and not shape & bad
    assert emitted == shape | bad, emitted ^ (shape | bad)


def test_unmeasured_u_without_reason_is_bad_data_with_shape_reason_classified():
    assert not ob.unmeasured("U", regime_aware=True).classified
    shaped = ob.unmeasured("U", "no_cpe_drop", regime_aware=True)
    assert shaped.classified and shaped.counts_as_measured and shaped.kind is None


@pytest.mark.parametrize("regime", ["", "A", "C"])
def test_sigma_observation_value_a_c_or_unclassified_admitted(regime):
    """A and C keep today's value; so does ``""`` (the classifier did not run)."""
    o = ob.sigma_observation(_report(**{**U_VALUE, "regime": regime}))
    assert o.kind == ob.VALUE and o.admitted and o.counts_as_measured
    assert o.classified is (regime != "")


def test_sigma_observation_b_value_recorded_not_admitted_but_measured():
    """Ruling 2026-10-02: a two-feature spectrum's value is the fit's choice of feature,
    not the film's σ — recorded with its regime, never told, and measured (no park)."""
    o = ob.sigma_observation(_report(**{**U_VALUE, "regime": "B",
                                        "regime_reason": "valley_then_rise"}))
    assert (o.kind, o.reported, o.regime) == (ob.VALUE, 1.8e-4, "B")
    assert not o.admitted
    assert o.classified and o.counts_as_measured


@pytest.mark.parametrize("regime, reason", [("B", "valley_then_rise"),
                                            *(("U", r) for r in ob.U_SHAPE_REASONS)])
def test_sigma_observation_withheld_value_flag_off_told_as_pre_5234d37(regime, reason):
    """Flag off is the clean rollback: the same B / shape-U value is told, unclassified."""
    report = _unstamped_report(**{**U_VALUE, "regime": regime, "regime_reason": reason})
    o = ob.sigma_observation(report)
    assert o.kind == ob.VALUE and o.admitted and not o.classified
    assert o.reported == _pre_5234d37_told(report) == 1.8e-4


def test_sigma_observation_legacy_upper_bound_recorded_not_admitted():
    o = ob.sigma_observation(_report(**LEGACY_BOUND))
    assert (o.kind, o.reported) == (ob.UPPER_BOUND, 3e-7)
    assert o.basis == "loss_at_numerator" and o.R_film_basis == "split_bulk"
    assert o.lower is None            # no interval without the active route
    assert not o.admitted and o.classified


@pytest.mark.parametrize("mode", ["bound", "bound_unqualified"])
def test_sigma_observation_route_a_passive_bound_admitted_with_lower(mode):
    o = ob.sigma_observation(_report(**{**A_BOUND, "mode": mode}))
    assert (o.kind, o.reported, o.lower) == (ob.UPPER_BOUND, 5e-4, 1e-4)
    assert (o.basis, o.R_film_ohm) == (REGIME_A_PASSIVE, 4e4)
    assert o.admitted


def test_sigma_observation_active_bound_non_route_basis_not_admitted():
    """Admission keys on the basis too: an active report whose ceiling is not route A's."""
    o = ob.sigma_observation(_report(**{**A_BOUND, "upper_bound_basis": "arc_open"}))
    assert o.kind == ob.UPPER_BOUND and not o.admitted


def test_sigma_observation_foot_lower_bound_admitted():
    o = ob.sigma_observation(_report(**A_LOWER))
    assert (o.kind, o.reported, o.lower) == (ob.LOWER_BOUND, 1e-4, None)
    assert (o.basis, o.R_film_ohm, o.R_film_basis) == (
        REGIME_A_FOOT_LOWER, 2e5, REGIME_A_FOOT_LOWER)
    assert o.admitted and o.counts_as_measured


def test_sigma_observation_flag_off_lower_bound_never_appears():
    """The flag-off shadow carries a finite regime_sigma_lower; it must never surface."""
    o = ob.sigma_observation(_report(**{**A_LOWER, "regime_active": False}))
    assert o.kind is None and o.reported is None and not o.admitted
    assert o.classified                   # still a measured spectrum for the park counter


def test_sigma_observation_unavailable_without_lower_is_unmeasured():
    o = ob.sigma_observation(_report(**{**A_LOWER, "regime_sigma_lower": NAN}))
    assert o == ob.unmeasured("A", regime_aware=True)


# ── Regime-B route outcomes (slice 2 §3.4), as ``engine._regime_b_sigma`` builds them ──

B_ROUTE_VALUE = dict(mode="value", value=4.9e-5, R_reported_ohm=1.26e6,
                     R_basis=REGIME_B_FILM_ARC, regime="B", regime_reason="valley_then_rise",
                     regime_mode="value", regime_sigma=4.9e-5, regime_active=True,
                     regime_b_mode="value", regime_b_sigma=4.9e-5)
B_ROUTE_LOWER = dict(mode="unavailable", R_reported_ohm=3.7e5, R_basis=REGIME_B_ARC_LOWER,
                     regime="U", regime_reason="no_cpe_drop", regime_mode="unavailable",
                     regime_sigma_lower=1.7e-4, regime_R_foot_ohm=3.7e5,
                     regime_active=True, regime_b_mode="lower_bound")
B_ROUTE_UPPER = dict(mode="bound", upper_bound=2.0e-7, upper_bound_basis=REGIME_B_NO_ARC,
                     R_reported_ohm=3.1e8, R_basis=REGIME_B_NO_ARC, provisional=True,
                     regime="B", regime_reason="valley_then_rise", regime_mode="bound",
                     regime_sigma=2.0e-7, regime_active=True, regime_b_mode="upper_bound")
#: Unavailable route: today's B value stands, flagged only by the shadow fields.
B_ROUTE_UNAVAILABLE = {**U_VALUE, "regime": "B", "regime_reason": "valley_then_rise",
                       "R_basis": "sum", "regime_b_mode": "unavailable",
                       "regime_b_detail": "electrode_not_negligible"}


@pytest.mark.parametrize("state, kind, basis, admitted", [
    (B_ROUTE_VALUE, ob.VALUE, REGIME_B_FILM_ARC, True),
    (B_ROUTE_LOWER, ob.LOWER_BOUND, REGIME_B_ARC_LOWER, True),
    (B_ROUTE_UPPER, ob.UPPER_BOUND, REGIME_B_NO_ARC, False),
    (B_ROUTE_UNAVAILABLE, ob.VALUE, "sum", False),
], ids=["value", "lower", "upper", "unavailable"])
def test_sigma_observation_b_route_admission_matrix(state, kind, basis, admitted):
    """Value told; lower bound admitted censored; no-arc ceiling recorded only (Q5); an
    unavailable route leaves today's B value withheld (f8b08b7). All measured (no park)."""
    o = ob.sigma_observation(_report(**state))
    assert (o.kind, o.basis, o.admitted) == (kind, basis, admitted)
    assert o.classified and o.counts_as_measured


def test_sigma_observation_b_lower_bound_basis_not_regime_a():
    """§3.3: a B lower bound must not reach the store spelled as route A's foot."""
    o = ob.sigma_observation(_report(**B_ROUTE_LOWER))
    assert (o.reported, o.R_film_ohm, o.lower) == (1.7e-4, 3.7e5, None)
    assert o.basis == o.R_film_basis == REGIME_B_ARC_LOWER
    assert REGIME_A_FOOT_LOWER not in (o.basis, o.R_film_basis)


def test_sigma_observation_lower_bound_unknown_basis_is_unmeasured_not_regime_a():
    """An active non-A lower bound with no lower-bound token reads ``""`` (§3.1(a)): an
    unknown never borrows route A's checked token. On A the foot is the only one there is."""
    unknown = ob.sigma_observation(_report(**{**B_ROUTE_LOWER, "R_basis": "split_bulk"}))
    assert unknown.kind == ob.LOWER_BOUND and unknown.basis == unknown.R_film_basis == ""
    a = ob.sigma_observation(_report(**{**A_LOWER, "R_basis": REGIME_A_FIT_RB}))
    assert a.basis == a.R_film_basis == REGIME_A_FOOT_LOWER


@pytest.mark.parametrize("change", [{"regime_active": False}, {"R_basis": "sum"}],
                         ids=["not_active", "other_basis"])
def test_sigma_observation_b_value_carve_out_needs_basis_and_active(change):
    """The carve-out keys on the film-arc basis **and** a taken route, never on the label."""
    o = ob.sigma_observation(_report(**{**B_ROUTE_VALUE, **change}))
    assert o.kind == ob.VALUE and not o.admitted and o.classified


def test_sigma_observation_b_route_upper_bound_never_admitted_even_as_route_a_basis_set():
    """The upper bound's refusal is the basis list, not the regime: positive control that
    the same report with route A's ceiling basis WOULD be admitted."""
    assert REGIME_B_NO_ARC not in ob.ROUTE_A_BOUND_BASES
    a_basis = {**B_ROUTE_UPPER, "upper_bound_basis": REGIME_A_PASSIVE}
    assert ob.sigma_observation(_report(**a_basis)).admitted is True
    assert ob.sigma_observation(_report(**B_ROUTE_UPPER)).admitted is False


@pytest.mark.parametrize("state", [B_ROUTE_VALUE, B_ROUTE_LOWER, B_ROUTE_UPPER,
                                   B_ROUTE_UNAVAILABLE],
                         ids=["value", "lower", "upper", "unavailable"])
def test_sigma_observation_b_route_flag_off_pre_regime_rule(state):
    """Flag off is the rollback: no bound admitted, nothing classified; a value told."""
    o = ob.sigma_observation(_unstamped_report(**state))
    assert not o.classified
    assert o.admitted is (o.kind == ob.VALUE)


def test_sigma_observation_not_ok_recorded_not_admitted():
    for state, kind in ((A_VALUE, ob.VALUE), (A_BOUND, ob.UPPER_BOUND),
                        (A_LOWER, ob.LOWER_BOUND)):
        o = ob.sigma_observation(_report(ok=False, **state))
        assert o.kind == kind and o.reported is not None, kind
        assert not o.admitted, kind
        assert o.classified, kind


@pytest.mark.parametrize("bad", [NAN, math.inf, 0.0, -1e-4])
def test_sigma_observation_non_finite_or_nonpositive_reported_is_none(bad):
    for state, field in ((A_VALUE, "value"), (A_BOUND, "upper_bound"),
                         (A_LOWER, "regime_sigma_lower")):
        o = ob.sigma_observation(_report(**{**state, field: bad}))
        assert o.kind is None and o.reported is None and not o.admitted, field
        assert o.regime == "A" and o.classified, field


@pytest.mark.parametrize("regime, classified",
                         [("A", True), ("B", True), ("C", True), ("U", False), ("", False)])
def test_sigma_observation_regime_label_sets_classified(regime, classified):
    o = ob.sigma_observation(_report(mode="unavailable", regime=regime))
    assert o.kind is None and o.regime == regime
    assert o.classified is classified and o.counts_as_measured is classified


def test_sigma_observation_malformed_report_unmeasured_never_raises():
    """Unreadable = unstamped = the pre-regime rule: nothing classified. A stamped report
    that fails mid-read keeps the armed classification."""
    assert ob.sigma_observation(None) == ob.unmeasured(regime_aware=False)
    broken = SimpleNamespace(ok=True, sigma=SimpleNamespace(regime="B"))
    assert ob.sigma_observation(broken) == ob.unmeasured("B", regime_aware=False)
    assert not ob.sigma_observation(broken).classified
    stamped = SimpleNamespace(ok=True, sigma=SimpleNamespace(regime="B", regime_aware=True))
    assert ob.sigma_observation(stamped) == dataclasses.replace(
        ob.unmeasured("B", regime_aware=True), classified=False)


class _SigmaThatRaises:
    """A stamped ``SigmaReport`` stand-in whose labels read cleanly but whose body raises
    inside ``_observe`` -- an analysis crash after the classifier has labelled it."""

    def __init__(self, regime: str, regime_aware: bool) -> None:
        self.regime, self.regime_reason, self.regime_aware = regime, "", regime_aware

    @property
    def regime_active(self):
        raise RuntimeError("analysis crashed")


@pytest.mark.parametrize("aware", [True, False])
@pytest.mark.parametrize("regime", ["A", "B", "C"])
def test_sigma_observation_exception_classified_regime_never_counts_as_measured(regime, aware):
    """Ruling 2026-10-01: a crash is never a measurement, whatever the flag or label; the
    labels are kept as a record."""
    o = ob.sigma_observation(SimpleNamespace(ok=True, sigma=_SigmaThatRaises(regime, aware)))
    assert o.classified is False and o.counts_as_measured is False and o.admitted is False
    assert o.kind is None
    assert (o.regime, o.regime_reason, o.regime_aware) == (regime, "", aware)


def test_unmeasured_is_nothing_stateable():
    o = ob.unmeasured(regime_aware=True)
    assert (o.kind, o.reported, o.lower, o.R_film_ohm) == (None, None, None, None)
    assert (o.regime, o.basis, o.admitted, o.classified) == ("", "", False, False)
    assert not o.counts_as_measured


@pytest.mark.parametrize("kind, admitted, classified, expected", [
    (ob.VALUE, True, False, True),
    (ob.UPPER_BOUND, False, False, False),
    (None, True, False, False),
    (None, False, True, True),
    (ob.UPPER_BOUND, False, True, True),
    (None, False, False, False),
])
def test_sigma_observation_counts_as_measured_truth_table(kind, admitted, classified,
                                                          expected):
    o = dataclasses.replace(ob.unmeasured(regime_aware=True), kind=kind, admitted=admitted,
                            classified=classified)
    assert o.counts_as_measured is expected


def test_sigma_observation_kinds_are_the_store_vocabulary():
    assert ob.KINDS == ("value", "lower_bound", "upper_bound")


# ── On a report the engine built ──────────────────────────────────────────────────


def test_sigma_observation_engine_compressed_a_lower_only_with_flag_on():
    """The production path: flag on → an admitted foot lower bound; flag off → not, and
    nothing classified (the rollback)."""
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from tests import eis_regime_synthetic as syn
    from tests.eis_regime_golden import golden_inputs

    eis = syn.as_eis(syn.RIG_F, syn.regime_a_rs(
        "compressed", 1.5e5, 0.6, 0.85, 1e5, 1e5, 20e-6, True, seed=503))
    on, off = (ob.sigma_observation(analyze_spectrum(
        eis, **golden_inputs(15), regime=RegimeSettings(enabled=flag)))
        for flag in (True, False))
    assert on.kind == ob.LOWER_BOUND and on.basis == REGIME_A_FOOT_LOWER
    assert on.R_film_ohm is not None and on.R_film_ohm > 0
    assert on.classified and on.counts_as_measured
    assert off.kind != ob.LOWER_BOUND and not off.classified


@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_sigma_observation_engine_jagged_u_value_not_admitted(flag):
    """§3.2 for F3: real data reaches it. Q5 (``142833Z`` ch25) classifies U, yet today's
    route still states a value (~1.8e-4 S/cm, arming review §5.2) — in both flag states.
    Refused flag on (F3); told flag off, exactly as before ``5234d37`` (rollback ruling)."""
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from softae.analysis.eis_data import EISResult
    from tests.eis_regime_golden import DATA, golden_inputs

    path = DATA / "jagged_20260820T142833Z_ch25.txt"
    if not path.exists():
        pytest.skip(f"real fixture absent: {path} (tests/data/ is gitignored)")
    report = analyze_spectrum(EISResult.load(path), **golden_inputs(25),
                              regime=RegimeSettings(enabled=flag))
    assert report.sigma.regime == "U" and report.sigma.mode == "value"
    assert report.sigma.regime_reason == "phase_incoherent"        # bad data: Q5 refused
    o = ob.sigma_observation(report)
    assert o.kind == ob.VALUE and o.reported is not None
    assert o.admitted is (not flag) and o.counts_as_measured is (not flag)
    assert not o.classified


@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_sigma_observation_mock_rig_floor_phase_ambiguous_value_told_only_flag_off(flag):
    """The mock rig's default spectrum is unrecognised-shape U. Through the campaign's
    own raw -> report hop its value is told flag off (rollback) and withheld flag on
    (ruling 2026-10-02) — but measured either way, so a mock-rig campaign never parks."""
    from softae.analysis.eis.regime_route import RegimeSettings
    from softae.core.autonomous_wiring import _spectrum_report_from_raw
    from softae.drivers.mock_espico import _synthetic_eis

    report = _spectrum_report_from_raw([_synthetic_eis(seed=0)], channel=5,
                                       thickness_um=150.0,
                                       regime=RegimeSettings(enabled=flag))
    s = report.sigma
    assert (s.regime, s.regime_reason, s.mode) == ("U", "floor_phase_ambiguous", "value")
    o = ob.sigma_observation(report)
    assert o.kind == ob.VALUE and o.reported is not None and o.counts_as_measured
    assert o.admitted is (not flag)
    assert o.classified is flag                 # flag off: nothing is classified


@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_sigma_observation_engine_b_value_withheld_flag_on_a_value_admitted(flag):
    """§3.2 for the 2026-10-02 ruling: engine-built reports reach both sides. A Step-1 B
    spectrum states a value in both flag states (the report is identical); flag on it is
    recorded untold but measured, flag off it is told by the pre-``5234d37`` rule. A
    Step-1 A spectrum's value is told in both."""
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from tests import eis_regime_synthetic as syn
    from tests.eis_regime_golden import golden_inputs

    def observe(gen):
        _, Z = next(gen())
        report = analyze_spectrum(syn.as_eis(syn.RIG_F, Z), **golden_inputs(15),
                                  regime=RegimeSettings(enabled=flag))
        return report, ob.sigma_observation(report)

    b_report, b = observe(syn.step1_regime_b)
    assert (b_report.sigma.regime, b_report.sigma.mode) == ("B", "value")
    assert b.kind == ob.VALUE and b.reported is not None and b.counts_as_measured
    assert b.admitted is (not flag)
    if not flag:
        assert b.reported == _pre_5234d37_told(b_report)

    a_report, a = observe(syn.step1_regime_a)
    assert (a_report.sigma.regime, a_report.sigma.mode) == ("A", "value")
    assert a.kind == ob.VALUE and a.admitted and a.counts_as_measured


#: Board-2 PAA well 14 (``eis_regime_arming_review.md`` §4): non_monotone U, old-route
#: ceiling. Read in place from the run directory, read-only; skips where it is absent.
PAA_NON_MONOTONE = Path(r"C:\Users\Osuji\softae_data\runs\20260929T150528Z_manual_eis"
                        r"\eis\ch14_manual.txt")


@pytest.mark.parametrize("flag", [False, True], ids=["flag_off", "flag_on"])
def test_sigma_observation_engine_paa_non_monotone_bound_counts_as_measured(flag):
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from softae.analysis.eis_data import EISResult
    from tests.eis_regime_golden import golden_inputs

    if not PAA_NON_MONOTONE.exists():
        pytest.skip(f"real spectrum absent: {PAA_NON_MONOTONE}")
    report = analyze_spectrum(EISResult.load(PAA_NON_MONOTONE), **golden_inputs(14),
                              regime=RegimeSettings(enabled=flag))
    assert (report.sigma.regime, report.sigma.regime_reason) == ("U", "non_monotone")
    o = ob.sigma_observation(report)
    assert o.kind == ob.UPPER_BOUND and not o.admitted
    # Flag off: an old-route bound is unmeasured and counts to a park, as before 5234d37.
    assert o.classified is flag and o.counts_as_measured is flag


# ── Flag off is a clean rollback to the pre-5234d37 decision (operator ruling 2026-10-01) ──


def _pre_5234d37_told(report) -> float | None:
    """The campaign decision before ``5234d37``, reproduced from git, not from memory.

    ``git show 5234d37^:src/softae/core/autonomous_wiring.py``, ``_sigma_from_eis_raw``
    (unchanged by ``9e586d8``, which touched neither wiring nor the loop)::

        if report is None: return None
        if not report.ok: return None                      # objective_rejected_by_gates
        if not report.sigma.is_value: return None          # objective_declined_bound
        value = float(report.sigma.value)                  # except -> None
        return value if math.isfinite(value) and value > 0 else None

    ``is_value`` is ``mode == "value"`` (``report.py``). The loop then told any non-None
    number and counted ``None`` as unmeasured, toward the park (``_is_unmeasured``).
    """
    if report is None or not report.ok or report.sigma.mode != "value":
        return None
    try:
        value = float(report.sigma.value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _flag_off_decision(report) -> tuple[float | None, bool]:
    """Today's campaign decision: ``(told number or None, counts toward the park)``.

    Read the way ``_sigma_from_eis_raw`` and ``AutonomousLoop._close_well`` read it: an
    admitted ``value`` is told; anything else is measured-untold if it
    ``counts_as_measured``, else unmeasured. The feasibility label and the DOE outcome
    follow from exactly these two facts.
    """
    o = ob.sigma_observation(report)
    told = o.reported if (o.kind == ob.VALUE and o.admitted) else None
    return told, (told is None and not o.counts_as_measured)


def _old_decision(report) -> tuple[float | None, bool]:
    told = _pre_5234d37_told(report)
    return told, told is None


#: Flag-off reports as the engine leaves them: classifier labels set (the shadow runs in
#: both flag states), ``regime_active`` False, and no ``regime_aware`` stamp.
FLAG_OFF_CASES = {
    **{f"value_{r or 'unclassified'}": dict(U_VALUE, regime=r) for r in ("", "A", "B", "C")},
    "value_u_incoherent": dict(U_VALUE, regime_reason="incoherent"),
    "value_u_phase_incoherent": dict(U_VALUE, regime_reason="phase_incoherent"),
    "value_u_shape_non_monotone": dict(U_VALUE, regime_reason="non_monotone"),
    "old_bound_b": {**LEGACY_BOUND, "regime": "B", "regime_reason": "valley_then_rise"},
    "old_bound_c": {**LEGACY_BOUND, "regime": "C", "regime_reason": "no_plateau"},
    "old_bound_a": LEGACY_BOUND,
    "old_bound_unqualified_b": {**LEGACY_BOUND, "mode": "bound_unqualified", "regime": "B"},
    "old_bound_u_shape": {**LEGACY_BOUND, "regime": "U", "regime_reason": "non_monotone"},
    "a_shadow_passive_bound": {**A_BOUND, "regime_active": False},
    "a_shadow_lower_unavailable": {**A_LOWER, "regime_active": False},
    # Route-A states with no stamp (hand-built, or older than the field): stored as
    # stated, never admitted.
    "a_route_bound_unstamped": A_BOUND,
    "a_route_lower_unstamped": A_LOWER,
    "a_route_value_unstamped": A_VALUE,
    "c_unavailable": dict(mode="unavailable", regime="C", regime_reason="no_plateau"),
    "u_unavailable_shape": dict(mode="unavailable", regime="U",
                                regime_reason="floor_phase_ambiguous"),
    **{f"value_b_{tag}": dict(U_VALUE, regime="B", value=v)
       for tag, v in (("nan", NAN), ("inf", math.inf), ("zero", 0.0), ("neg", -1e-4))},
}


@pytest.mark.parametrize("ok", [True, False], ids=["ok", "not_ok"])
@pytest.mark.parametrize("name", sorted(FLAG_OFF_CASES))
def test_sigma_observation_flag_off_constructed_matches_pre_5234d37_decision(name, ok):
    """No stamp = ``regime_aware`` False, the field's default: the old rule, every kind."""
    report = _unstamped_report(ok=ok, **FLAG_OFF_CASES[name])
    assert _flag_off_decision(report) == _old_decision(report)


def _flag_off_corpus():
    """The 34-spectrum probe set: six Step-1 A/B/C each, plus the real regime corpus."""
    import itertools

    from tests import eis_regime_synthetic as syn
    from tests.eis_regime_golden import real_corpus

    for family, gen in (("A", syn.step1_regime_a), ("B", syn.step1_regime_b),
                        ("C", syn.step1_regime_c)):
        for i, (_, Z) in enumerate(itertools.islice(gen(), 6)):
            yield f"step1_{family}{i}", syn.as_eis(syn.RIG_F, Z), True
    for name, eis in real_corpus():
        yield name, eis, False


def test_sigma_observation_flag_off_engine_corpus_matches_pre_5234d37_decision():
    """§3.2: the rollback holds on reports the production path builds, flag off.

    Also asserts the branches the defect lived on are actually visited by the synthetic
    half (always present; ``tests/data/`` is gitignored): an old-route bound on a
    classified regime (C), a classified unavailable (C) and a U — so a green here is not
    a green over a population that never reached them. Flag off, a Step-1 A spectrum
    reports today's route, so the A-route bound/unavailable never appear (by design).
    """
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.analysis.eis.regime_route import RegimeSettings
    from tests.eis_regime_golden import golden_inputs

    mismatches, visited = [], set()
    for name, eis, synthetic in _flag_off_corpus():
        report = analyze_spectrum(eis, **golden_inputs(15),
                                  regime=RegimeSettings(enabled=False))
        if synthetic:
            visited.add((report.sigma.regime, report.sigma.mode))
        now, old = _flag_off_decision(report), _old_decision(report)
        if now != old:
            mismatches.append((name, report.sigma.regime, report.sigma.regime_reason,
                               report.sigma.mode, now, old))
    assert {("A", "value"), ("B", "value"), ("C", "bound"), ("C", "unavailable"),
            ("U", "bound")} <= visited, visited
    assert mismatches == []
