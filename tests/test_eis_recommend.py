"""Threshold recommendations — the arithmetic, and the refusals that matter more.

Every distribution below is synthetic and stated inline, because the whole claim of
this module is *"given these numbers, the fence goes here"* and a test that fitted the
fence to whatever the code produced would assert nothing.

The refusal tests carry as much weight as the rule tests. A recommender that always
answers is worse than none: it dresses a percentile of four rejects as a calibrated
threshold, and the operator has no way to see the difference from the table.
"""

from __future__ import annotations

import math

import pytest

from softae.analysis.eis.recommend import (
    METRIC_KEYS,
    Recommendation,
    SpectrumRecord,
    deduplicate,
    joint_would_reject,
    recommend_all,
)
from softae.analysis.eis.recommend_report import as_toml_block, observed_only
from softae.analysis.eis.recommend_rules import (
    complement_fence,
    count_minimum,
    gap_split,
    lower_fence,
    physical_point_floor,
    upper_fence,
)

N = 40  # comfortably above the default evidence floor of 20


def records(metric: str, values, **fields) -> list[SpectrumRecord]:
    """One record per value, each with a distinct fingerprint."""
    return [SpectrumRecord(key=f"c01:{i:012x}", metrics={metric: float(v)},
                           n_surviving=41, **fields)
            for i, v in enumerate(values)]


def only(recs: "list[Recommendation]", key: str) -> Recommendation:
    return next(r for r in recs if r.key == key)


def spread(centre: float, n: int = N, step: float = 0.001) -> list[float]:
    """A tight but non-constant population around ``centre``."""
    return [centre + step * (i - n / 2) for i in range(n)]


# ── The four rule families ───────────────────────────────────────────────────

class TestRuleMath:
    def test_upper_fence_places_the_threshold_above_the_healthy_bulk(self):
        values = list(range(1, 41))            # 1..40, P75 = 30.25, IQR = 19.5
        fence = upper_fence(values)
        assert fence > max(values)
        assert fence == pytest.approx(max(30.25 + 1.5 * 19.5, 1.25 * 38.05))

    def test_upper_fence_is_floored_at_a_margin_above_p95_on_a_tight_distribution(self):
        # A very tight population makes the Tukey fence tight too, which would arm a
        # gate that rejects merely-mediocre spectra. The 1.25xP95 floor is the guard,
        # and on this distribution it is the binding half of the max().
        import numpy as np

        values = spread(1.0)
        p25, p75, p95 = (float(np.percentile(values, q)) for q in (25, 75, 95))
        assert p75 + 1.5 * (p75 - p25) < 1.25 * p95
        assert upper_fence(values) == pytest.approx(1.25 * p95)

    def test_lower_fence_mirrors_the_upper_one_on_a_reflected_distribution(self):
        values = [float(v) for v in range(1, 41)]
        assert lower_fence(values) == pytest.approx(-upper_fence([-v for v in values]))

    def test_count_rule_never_recommends_fewer_points_than_the_model_has_parameters(
            self):
        # Every spectrum kept 40 points, so P5 is 40 — and the rule still may not
        # propose fewer than the physical floor if the population were poorer.
        assert count_minimum([6.0] * 40, floor=8) == 8
        assert count_minimum([40.0] * 40, floor=8) == 40

    def test_the_point_floor_comes_from_the_model_registry_not_a_literal(self,
                                                                        monkeypatch):
        from softae.analysis import circuit_fitting

        assert physical_point_floor("simpleSalt") == 8      # 5 parameters + 3
        monkeypatch.setitem(circuit_fitting.CIRCUIT_MODELS, "wide",
                            {"initial_guess": [0] * 9})
        assert physical_point_floor("wide") == 12
        assert physical_point_floor("no-such-model") == 8

    def test_r_squared_is_recommended_through_its_complement_and_stays_below_one(self):
        # A clamp would silently return 1.0 and arm a gate demanding a perfect fit.
        proposed = complement_fence([0.999 - 0.0001 * i for i in range(N)])
        assert 0.0 < proposed < 1.0

    def test_bimodal_tand_slope_is_split_at_the_gap_between_the_populations(self):
        parallel = [-0.95 + 0.01 * i for i in range(10)]     # -0.95 .. -0.86
        parasitic = [0.30 + 0.01 * i for i in range(10)]     # +0.30 .. +0.39
        split = gap_split(parallel + parasitic, min_gap=0.25, span=(-1.2, 0.5))
        assert split == pytest.approx((-0.86 + 0.30) / 2)

    def test_unimodal_tand_slope_holds_the_theory_anchored_default(self):
        assert gap_split(spread(-0.9, step=0.01), min_gap=0.25,
                         span=(-1.2, 0.5)) is None

    def test_a_gap_with_fewer_than_three_points_on_one_side_is_not_two_populations(
            self):
        # One outlier and a bulk is not bimodality, however wide the gap.
        assert gap_split([-0.9] * 20 + [0.4, 0.41], min_gap=0.25,
                         span=(-1.2, 0.5)) is None

    def test_min_abs_z_uses_a_decade_margin_not_a_percentage(self):
        # The film population spans 10^6-10^8 ohm; a 25 % fence around it would reject
        # an ordinary wet film. One decade of clearance is what "implausible" means.
        # The shorts and opens are what make the two keys fire at all.
        import numpy as np

        values = ([1e6 * 1.06 ** i for i in range(N)] + [1e-6] * 4 + [1e13] * 4)
        recs = recommend_all(records("z_median", values))
        low, high = only(recs, "min_abs_z"), only(recs, "max_abs_z")
        assert low.status == high.status == "recommended"
        assert low.value == pytest.approx(float(np.percentile(values, 5)) / 10.0)
        assert high.value == pytest.approx(float(np.percentile(values, 95)) * 10.0)
        # A percentage fence would have landed within a few percent of the percentile.
        assert high.value / float(np.percentile(values, 95)) == pytest.approx(10.0)


# ── Refusals and holds ───────────────────────────────────────────────────────

class TestRefusals:
    def test_a_metric_never_observed_refuses_with_that_reason_by_name(self):
        rec = only(recommend_all(records("cap_slope", spread(0.05))), "max_rel_se")
        assert rec.status == "refused" and rec.value is None
        assert "never observed" in rec.reason and "rel_se_measurand" in rec.reason

    def test_a_metric_below_the_evidence_floor_refuses_rather_than_extrapolating(self):
        rec = only(recommend_all(records("cap_slope", spread(0.05, n=19))),
                   "cap_flatness_max")
        assert rec.status == "refused" and rec.n == 19
        assert "need 20" in rec.reason and "single observation" in rec.reason

    def test_lowering_the_evidence_floor_is_the_operator_overriding_on_the_record(self):
        values = spread(0.05, n=19)
        assert only(recommend_all(records("cap_slope", values), min_evidence=12),
                    "cap_flatness_max").status != "refused"

    def test_a_constant_distribution_refuses_rather_than_recommending_its_own_value(
            self):
        rec = only(recommend_all(records("cap_slope", [0.05] * N)),
                   "cap_flatness_max")
        assert rec.status == "refused"
        assert "a constant is not a distribution" in rec.reason

    def test_nan_metric_values_are_excluded_from_n_rather_than_counted(self):
        good = records("cap_slope", spread(0.05, n=25))
        blank = [SpectrumRecord(key=f"c02:{i:012x}", metrics={}) for i in range(50)]
        rec = only(recommend_all(good + blank), "cap_flatness_max")
        assert rec.n == 25

    def test_a_gate_that_fired_on_nothing_is_held_and_labelled_unexercised(self):
        # Every spectrum is far inside the default 0.15, so nothing challenged it.
        rec = only(recommend_all(records("cap_slope", spread(0.01, step=1e-4))),
                   "cap_flatness_max")
        assert rec.status == "hold" and rec.value is None
        assert rec.exercise == "unexercised"
        assert "Untested is not validated" in rec.reason

    def test_a_gate_that_fired_on_ninety_percent_is_labelled_measures_the_rig(self):
        rec = only(recommend_all(records("cap_slope", spread(2.0, step=0.01))),
                   "cap_flatness_max")
        assert rec.exercise == "measures-the-rig"
        assert "measuring the rig, not the sample" in rec.reason

    def test_a_negative_r_squared_population_refuses_instead_of_fencing_below_zero(self):
        # The real case: the 2026-08-14 rehearsal proposed min_r_squared = -0.17. A
        # negative R2 floor is not a loose gate, it is a nonsensical one — it admits
        # fits that explain the data WORSE than the sample mean. The complement rule is
        # arithmetically correct here; what it is telling us is that the population is
        # unfittable, and that has to be said rather than pasted into a config.
        rec = only(recommend_all(records("r_squared", [-0.02 - 0.004 * i
                                                       for i in range(N)])),
                   "min_r_squared")
        assert rec.status == "refused" and rec.value is None
        assert "sanity floor" in rec.reason and "-0.46" in rec.reason
        assert "worse than predicting the mean" in rec.reason.lower()

    def test_the_sanity_floor_leaves_an_ordinary_bad_tail_alone(self):
        # The floor must stay a rare refusal, not a second default: a population with a
        # genuinely poor tail still gets its number.
        values = [0.999 - 0.0005 * i for i in range(30)] + [0.5 + 0.01 * i
                                                            for i in range(10)]
        rec = only(recommend_all(records("r_squared", values)), "min_r_squared")
        assert rec.status == "recommended" and rec.value > 0.0
        assert "sanity floor" not in rec.reason

    def test_an_empty_population_refuses_every_key_with_the_pre_t71_reason(self):
        recs = recommend_all([])
        assert len(recs) == len(METRIC_KEYS)
        assert all(r.status == "refused" and "pre-T7.1 log" in r.reason for r in recs)


# ── Counting, direction and the behavioural marker ───────────────────────────

class TestCounting:
    def test_the_would_reject_count_at_the_proposed_value_is_reported_per_key(self):
        # 30 healthy spectra plus 10 dispersive ones: the default catches all ten, the
        # proposed fence sits above the bulk and catches fewer.
        values = spread(0.05, n=30) + [1.0 + 0.1 * i for i in range(10)]
        rec = only(recommend_all(records("cap_slope", values)), "cap_flatness_max")
        assert rec.status == "recommended"
        assert rec.fired_at_default == 10
        assert rec.would_reject_at_value < rec.fired_at_default

    def test_the_joint_would_reject_count_is_not_the_sum_of_the_per_key_counts(self):
        # One spectrum fails two gates at once. Adding the columns would count it twice
        # and overstate what applying the block actually costs.
        bad = SpectrumRecord(key="c01:ffffffffffff",
                             metrics={"cap_slope": 9.0, "residual_rms_pct": 900.0})
        healthy = [SpectrumRecord(
            key=f"c01:{i:012x}",
            metrics={"cap_slope": 0.05 + 1e-4 * i, "residual_rms_pct": 2.0 + 0.1 * i})
            for i in range(N)]
        population = healthy + [bad]
        recs = recommend_all(population)
        per_key = sum(r.would_reject_at_value for r in recs if r.applied)
        assert joint_would_reject(recs, population) == 1
        assert per_key > 1

    def test_a_recommendation_that_relaxes_a_gate_is_marked_loosens(self):
        values = spread(0.05, n=30) + [1.0 + 0.1 * i for i in range(10)]
        assert only(recommend_all(records("cap_slope", values)),
                    "cap_flatness_max").direction == "loosens"

    def test_a_tightening_recommendation_is_marked_and_only_reachable_once_fired(self):
        # R^2 clustered just under a stricter-than-default bulk: the complement rule
        # proposes below the 0.95 default (loosens). A population that is uniformly
        # excellent never fires and is held instead, so no key tightens on silence.
        recs = recommend_all(records("r_squared", spread(0.999, step=1e-5)))
        held = only(recs, "min_r_squared")
        assert held.status == "hold" and held.direction == "-"
        assert all(r.fired_at_default >= 1 for r in recs if r.direction == "tightens")

    def test_the_kk_and_rho_keys_are_marked_behavioural(self):
        # These change which points reach the fit and which resistance sigma comes
        # from. Presenting them as verdict-only tuning would be misleading.
        behavioural = {k.key for k in METRIC_KEYS if k.behavioural}
        assert behavioural == {"kk_resid_pct", "kk_max_truncate_frac", "rho_degenerate"}

    def test_the_kk_truncate_fraction_is_zero_for_a_compliant_spectrum_not_missing(
            self):
        # The gate stores only the numerator, and only when a point failed. Treating a
        # compliant spectrum as "no observation" would leave the key with four samples.
        compliant = SpectrumRecord.from_event({
            "event": "eis_spectrum_metrics", "n_surviving": 40, "n_dropped": 0,
            "metrics": {"kk_max_resid_pct": 0.2}})
        truncated = SpectrumRecord.from_event({
            "event": "eis_spectrum_metrics", "n_surviving": 36, "n_dropped": 4,
            "metrics": {"kk_max_resid_pct": 3.0, "kk_truncated": 4.0}})
        assert compliant.metrics["kk_truncate_frac"] == 0.0
        assert truncated.metrics["kk_truncate_frac"] == pytest.approx(0.1)

    def test_a_spectrum_whose_kk_ladder_never_ran_carries_no_truncation_fraction(self):
        record = SpectrumRecord.from_event({
            "event": "eis_spectrum_metrics", "n_surviving": 40, "n_dropped": 0,
            "metrics": {"cap_slope": 0.1}})
        assert "kk_truncate_frac" not in record.metrics


# ── Deduplication ────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_two_analyze_calls_on_one_spectrum_deduplicate_to_one_record(self):
        thin = SpectrumRecord(key="c07:abc", metrics={"cap_slope": 0.1})
        rich = SpectrumRecord(key="c07:abc",
                              metrics={"cap_slope": 0.1, "r_squared": 0.99})
        assert deduplicate([thin, rich]) == [rich]
        assert deduplicate([rich, thin]) == [rich]

    def test_two_different_spectra_on_one_channel_do_not_deduplicate(self):
        assert len(deduplicate([SpectrumRecord(key="c07:aaa"),
                                SpectrumRecord(key="c07:bbb")])) == 2

    def test_the_evidence_floor_bites_at_the_true_sample_size_not_the_doubled_one(self):
        # Without dedup a 16-well run would look like 32 spectra and clear the floor of
        # 20 it was set to fail.
        doubled = records("cap_slope", spread(0.05, n=16)) * 2
        assert only(recommend_all(deduplicate(doubled)),
                    "cap_flatness_max").status == "refused"


class TestEnforcingRegimes:
    def test_front2_metrics_from_an_enforcing_run_are_excluded_from_a_mixed_log(self):
        # R18 skips the fit for a rejected spectrum when gates enforce, so its Front-2
        # values are a strict sample of the admitted population and must not be pooled.
        observing = records("r_squared", spread(0.98, step=1e-4), enforced=False)
        enforcing = records("r_squared", [0.999] * 10, enforced=True)
        for i, record in enumerate(enforcing):
            object.__setattr__(record, "key", f"c02:{i:012x}")
        rec = only(recommend_all(observing + enforcing), "min_r_squared")
        assert rec.n == len(observing)

    def test_a_single_regime_log_pools_everything_it_has(self):
        rec = only(recommend_all(records("r_squared", spread(0.98, step=1e-4),
                                         enforced=True)), "min_r_squared")
        assert rec.n == N

    def test_split_identifiable_is_excluded_from_a_mixed_log_like_any_front2_metric(
            self):
        # `gate_degeneracy` reads `split_identifiable` off the covariance, so it cannot
        # exist without a fit — Front-2 by the set's own definition.
        #
        # LATENT, NOT LIVE: no METRIC_KEYS entry targets it today, so nothing pools it
        # and no recommendation is currently wrong. The membership is what makes the
        # §6.2 exclusion already true on the day a fence IS placed against it, so this
        # test places one rather than asserting set contents and calling that a reason.
        from softae.analysis.eis.recommend import FRONT2_METRICS, _Key, _observations

        assert not [k for k in METRIC_KEYS if k.metric == "split_identifiable"], (
            "a real key now targets split_identifiable — this test's premise is stale")
        assert "split_identifiable" in FRONT2_METRICS

        spec = _Key("eis.gates", "hypothetical_fence", "split_identifiable", "gap",
                    "lower_eq")
        observing = records("split_identifiable", [1.0] * 10, enforced=False)
        enforcing = records("split_identifiable", [0.0] * 10, enforced=True)
        assert _observations(spec, observing + enforcing, True) == [1.0] * 10
        assert len(_observations(spec, observing + enforcing, False)) == 20


# ── The emitted TOML block ───────────────────────────────────────────────────

class TestTomlBlock:
    @pytest.fixture()
    def block(self):
        values = spread(0.05, n=30) + [1.0 + 0.1 * i for i in range(10)]
        recs = recommend_all(records("cap_slope", values))
        return as_toml_block(recs, source="run.log", n_spectra=40, when="2026-08-13")

    def test_the_toml_block_never_sets_enabled_true(self, block):
        live = [ln for ln in block.splitlines() if not ln.lstrip().startswith("#")]
        assert [ln for ln in live if ln.startswith("enabled")]
        assert not any("true" in ln.lower() for ln in live)
        assert "ARMING IS YOURS" in block

    def test_a_refused_key_is_emitted_commented_out_with_its_reason(self, block):
        line = next(ln for ln in block.splitlines() if "max_rel_se" in ln)
        assert line.startswith("#") and "REFUSED" in line and "never observed" in line

    def test_a_recommended_key_is_emitted_live_with_its_evidence(self, block):
        line = next(ln for ln in block.splitlines()
                    if ln.startswith("cap_flatness_max"))
        assert "=" in line and "n=40" in line and "upper-fence" in line

    def test_a_behavioural_key_is_marked_in_the_block(self, block):
        assert any("(!)" in ln and "kk_resid_pct" in ln for ln in block.splitlines())
        assert "change stored NUMBERS" in block

    def test_the_block_parses_as_toml_with_every_recommendation_applied(self, block):
        import tomllib

        # `[eis.gates]` is a dotted table, so it nests under `eis` exactly as it does
        # in softae_config.toml — the block must paste in without re-heading.
        parsed = tomllib.loads(block)
        assert parsed["eis"]["gates"]["enabled"] is False
        assert parsed["quality"]["enabled"] is False
        assert parsed["eis"]["gates"]["cap_flatness_max"] > 0.15


# ── Constants with no config line ────────────────────────────────────────────

class TestObservedOnly:
    def test_a_hardcoded_constant_is_reported_with_its_firing_rate_not_recommended(
            self):
        population = records("cross_check_pct", [5.0] * 30 + [80.0] * 10)
        row = next(o for o in observed_only(population)
                   if o["metric"] == "cross_check_pct")
        assert row["threshold"] == 25.0 and row["fired"] == 10 and row["n"] == 40
        assert not any(k.metric == "cross_check_pct" for k in METRIC_KEYS)

    def test_the_runs_test_constant_is_judged_on_the_magnitude_of_z(self):
        row = next(o for o in observed_only(records("runs_z", [-9.0] * 5 + [0.5] * 5))
                   if o["metric"] == "runs_z")
        assert row["fired"] == 5

    def test_a_constant_no_spectrum_measured_is_omitted_rather_than_reported_as_zero(
            self):
        assert observed_only(records("cap_slope", [0.1] * 5)) == []


def test_every_recommendation_carries_a_reason_whatever_its_status():
    # The reason is the product. A blank one turns a refusal into a silent absence.
    for population in ([], records("cap_slope", [0.05] * 3),
                       records("cap_slope", spread(0.05))):
        for rec in recommend_all(population):
            assert rec.reason.strip()
            assert rec.value is None or math.isfinite(rec.value)
