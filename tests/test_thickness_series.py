"""Planning a thickness series that can actually answer a thickness question.

The design defect this prevents (overhaul F12) is not recoverable after the fact, so the
tests are about the *plan*, not the analysis. The reference case throughout is the real
one: CH27/28 = 200 µm, CH29/30 = 150, CH31/32 = 100 — thickness and channel index moving
together, which scores **r = −0.956** and makes a channel artifact and a thickness effect
mathematically indistinguishable.
"""

from __future__ import annotations

import pytest

from softae.core.thickness_series import (
    DEFAULT_MAX_CORRELATION,
    MIN_REPLICATES,
    ConfoundReport,
    ThicknessPlan,
    ThicknessPlanError,
    correlation,
    detect_confounding,
    plan_series,
)

F12_CHANNELS = [27, 28, 29, 30, 31, 32]
F12_LEVELS = [200.0, 200.0, 150.0, 150.0, 100.0, 100.0]


class TestCorrelation:
    def test_the_real_f12_series_scores_near_minus_one(self):
        assert correlation(F12_CHANNELS, F12_LEVELS) == pytest.approx(-0.956, abs=0.01)

    def test_a_decorrelated_assignment_scores_near_zero(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=42)
        assert abs(plan.achieved_correlation) <= DEFAULT_MAX_CORRELATION

    def test_a_single_level_has_no_defined_correlation(self):
        r = correlation([1, 2, 3], [100.0, 100.0, 100.0])
        assert r != r      # NaN — zero variance in level

    def test_too_few_points_is_nan_not_an_exception(self):
        assert correlation([1], [100.0]) != correlation([1], [100.0])


class TestPlanning:
    def test_a_plan_is_balanced_across_levels(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=1)
        counts = sorted(plan.replicates.values())
        assert max(counts) - min(counts) <= 1
        assert sum(counts) == 32

    def test_every_channel_gets_exactly_one_level(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=1)
        assert plan.channels == tuple(range(1, 33))
        assert len(plan.assignment) == 32

    def test_it_is_reproducible_from_the_seed(self):
        a = plan_series([100, 150, 200, 250], range(1, 33), seed=7)
        b = plan_series([100, 150, 200, 250], range(1, 33), seed=7)
        assert a.assignment == b.assignment

    def test_different_seeds_give_different_assignments(self):
        a = plan_series([100, 150, 200, 250], range(1, 33), seed=1)
        b = plan_series([100, 150, 200, 250], range(1, 33), seed=2)
        assert a.assignment != b.assignment

    def test_the_achieved_correlation_is_recorded_not_assumed(self):
        """Randomisation does not *guarantee* decorrelation — a uniform shuffle can
        land on an ordered arrangement. The plan carries what it actually achieved."""
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=3)
        assert plan.achieved_correlation == pytest.approx(
            correlation(plan.channels, [plan.assignment[c] for c in plan.channels]))
        assert plan.draws >= 1

    def test_an_impossible_request_raises_rather_than_returning_a_confounded_plan(self):
        """Two channels and two levels: every arrangement is perfectly correlated.

        Returning a plan that failed its own constraint would be worse than none — the
        point of the plan is to be able to say afterwards that the design was sound.
        """
        with pytest.raises(ThicknessPlanError, match="no assignment"):
            plan_series([100, 200], [1, 2], seed=0, max_draws=50)

    def test_one_level_is_refused(self):
        with pytest.raises(ThicknessPlanError, match="two distinct"):
            plan_series([100], range(1, 9))

    def test_more_levels_than_channels_is_refused(self):
        with pytest.raises(ThicknessPlanError, match="cannot carry"):
            plan_series([100, 150, 200, 250], [1, 2])

    def test_duplicate_channels_are_refused(self):
        with pytest.raises(ThicknessPlanError, match="duplicate"):
            plan_series([100, 200], [1, 1, 2, 2])

    def test_the_error_says_what_to_do_about_it(self):
        with pytest.raises(ThicknessPlanError) as exc:
            plan_series([100, 200], [1, 2], seed=0, max_draws=20)
        assert "add channels" in str(exc.value)


class TestAdequacyForE5:
    """Unconfounded is necessary but not sufficient — §5.6 wants more than that."""

    def test_a_clean_four_level_plan_is_adequate(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=5)
        ok, why = plan.is_adequate_for_geometry_series()
        assert ok, why

    def test_two_levels_is_unconfoundable_but_still_inadequate(self):
        """A slope through two levels has no residual, so nothing can show the linear
        model is wrong."""
        plan = plan_series([100, 250], range(1, 33), seed=5)
        assert abs(plan.achieved_correlation) <= DEFAULT_MAX_CORRELATION
        ok, why = plan.is_adequate_for_geometry_series()
        assert not ok
        assert "levels" in why

    def test_too_narrow_a_span_is_reported(self):
        plan = plan_series([100, 110, 120, 130], range(1, 33), seed=5)
        ok, why = plan.is_adequate_for_geometry_series()
        assert not ok
        assert "span" in why

    def test_a_level_with_one_replicate_is_reported(self):
        plan = ThicknessPlan(
            levels_um=(100, 150, 200, 250),
            assignment={1: 100.0, 2: 100.0, 3: 150.0, 4: 150.0,
                        5: 200.0, 6: 200.0, 7: 250.0},
        )
        ok, why = plan.is_adequate_for_geometry_series()
        assert not ok
        assert f"< {MIN_REPLICATES} replicates" in why


class TestDetectConfounding:
    def test_the_real_f12_series_is_reported_as_confounded(self):
        report = detect_confounding(F12_CHANNELS, F12_LEVELS)
        assert report.confounded
        assert "indistinguishable" in report.describe()

    def test_a_planned_series_is_not(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=9)
        rows = plan.as_rows()
        report = detect_confounding([c for c, _ in rows], [v for _, v in rows])
        assert not report.confounded

    def test_casting_something_other_than_the_plan_is_caught(self):
        """A sound plan followed inattentively produces exactly the dataset the plan
        existed to prevent."""
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=9)
        rows = plan.as_rows()
        levels = [v for _, v in rows]

        # Swap two channels that genuinely differ — swapping equal levels is a no-op,
        # which is correct behaviour and would test nothing.
        j = next(k for k in range(1, len(levels)) if levels[k] != levels[0])
        levels[0], levels[j] = levels[j], levels[0]

        report = detect_confounding([c for c, _ in rows], levels, plan=plan)
        assert report.matches_plan is False
        assert len(report.deviations) == 2
        assert "planned" in report.deviations[0]

    def test_a_faithful_cast_matches_the_plan(self):
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=9)
        rows = plan.as_rows()
        report = detect_confounding([c for c, _ in rows], [v for _, v in rows],
                                    plan=plan)
        assert report.matches_plan is True
        assert report.deviations == ()

    def test_a_partial_cast_of_a_sound_plan_can_still_read_as_confounded(self):
        """Honest, and worth pinning: the verdict is on the data *as it stands*.

        Dropping one channel from an 8-channel plan that scored r = −0.098 leaves a
        7-channel subset at −0.55. If casting stopped there the dataset really would be
        confounded, so the verdict is correct — what the operator needs alongside it is
        the pending count, which is why the two are reported together.
        """
        plan = plan_series([100, 150, 200, 250], range(1, 9), seed=9)
        rows = plan.as_rows()[:-1]
        report = detect_confounding([c for c, _ in rows], [v for _, v in rows],
                                    plan=plan)
        assert report.pending == (8,)
        assert report.deviations == ()
        assert report.matches_plan is True

    def test_a_channel_cast_outside_the_plan_is_a_deviation(self):
        plan = plan_series([100, 150, 200, 250], range(1, 9), seed=9)
        rows = plan.as_rows()
        report = detect_confounding(
            [c for c, _ in rows] + [99], [v for _, v in rows] + [175.0], plan=plan)
        assert report.matches_plan is False
        assert any("not in the plan" in d for d in report.deviations)

    def test_a_single_level_series_certifies_nothing(self):
        """Absent evidence of decorrelation is not evidence of it — but a series with
        one level is *not yet judgeable* rather than a proven confound."""
        report = detect_confounding([1, 2, 3, 4], [100.0, 100.0, 100.0, 100.0])
        assert not report.certified
        assert report.verdict == "indeterminate"

    def test_a_partial_cast_is_pending_not_deviant(self):
        """A series is measured one channel at a time; "not yet" must not read as a
        fault or the report cries wolf through the middle of every campaign."""
        plan = plan_series([100, 150, 200, 250], range(1, 33), seed=9)
        rows = plan.as_rows()[:6]
        report = detect_confounding([c for c, _ in rows], [v for _, v in rows],
                                    plan=plan)
        assert report.deviations == ()
        assert len(report.pending) == 26
        assert report.matches_plan is True


class TestSerialisation:
    def test_a_plan_round_trips(self):
        plan = plan_series([100, 150, 200, 250], range(1, 17), seed=11,
                           plan_id="p1", created_at="2026-08-06", notes="bench")
        back = ThicknessPlan.from_json(plan.to_json())
        assert back.assignment == plan.assignment
        assert back.plan_id == "p1"
        assert back.notes == "bench"
        assert back.achieved_correlation == pytest.approx(plan.achieved_correlation)

    def test_channel_keys_survive_json_string_keys(self):
        plan = plan_series([100, 200, 300, 400], range(1, 9), seed=2)
        back = ThicknessPlan.from_json(plan.to_json())
        assert all(isinstance(k, int) for k in back.assignment)


class TestReportShape:
    def test_an_empty_report_certifies_nothing(self):
        """The safe default: nothing measured cannot be certified unconfounded —
        but it is *indeterminate*, not a design failure."""
        r = ConfoundReport()
        assert r.verdict == "indeterminate"
        assert not r.certified
        assert not r.confounded
