"""Casting a planned thickness series — E5 link 2, Option C.

The plan is authoritative here, which is the whole design decision. `ElectrodeAllocator`
exists to hand out the next free well; a thickness plan says *channel 7 gets 150 µm*.
Both claim the right to choose channels, and if the allocator wins it silently skips an
occupied well, the plan loses a member, and the level balance breaks — F12 again, this
time disguised by the plan's existence.
"""

from __future__ import annotations

import pytest

from softae.core.thickness_series import ThicknessPlan, correlation
from softae.workflows.geometry_series import (
    GeometrySeriesError,
    choose_drift_channel,
    plan_geometry_series_run,
    round_robin_by_level,
    verify_channels_free,
)

# A crossed 16-channel, 4-level assignment of the shape the planner emits.
ASSIGN = {1: 150.0, 2: 250.0, 3: 200.0, 4: 100.0, 5: 250.0, 6: 200.0, 7: 200.0,
          8: 100.0, 9: 250.0, 10: 150.0, 11: 150.0, 12: 100.0, 13: 200.0,
          14: 100.0, 15: 150.0, 16: 250.0}


def _plan(assignment=None, plan_id="geo-1"):
    a = dict(ASSIGN if assignment is None else assignment)
    return ThicknessPlan(
        plan_id=plan_id, levels_um=tuple(sorted(set(a.values()))),
        assignment=a, achieved_correlation=correlation(sorted(a),
                                                       [a[c] for c in sorted(a)]))


class TestCastOrderKeepsEveryPrefixBalanced:
    """The reason the order is not channel order. Measured, not asserted."""

    def test_each_block_covers_every_level_once(self):
        order = round_robin_by_level(ASSIGN)
        n_levels = len(set(ASSIGN.values()))
        for i in range(0, len(order) - n_levels + 1, n_levels):
            block = order[i:i + n_levels]
            assert len(set(ASSIGN[c] for c in block)) == n_levels

    def test_a_half_finished_run_stays_usable_where_channel_order_would_not(self):
        """Measured over 30 planned series, worst case across all prefixes >= n.

        The threshold is 8 of 16 and not lower, because it is not true lower. Balance
        (equal counts per level) and decorrelation (channel not tracking level) are
        different properties, and a plain round-robin delivers only the first — the
        opening block of a real plan measured |r| = 0.61 that way, worse than the 0.2
        the planner achieved for the series as a whole.
        """
        from softae.core.thickness_series import plan_series

        def worst_from(order, assign, start):
            return max(abs(correlation(order[:n], [assign[c] for c in order[:n]]))
                       for n in range(start, len(order) + 1))

        worst_chan = worst_rr = 0.0
        for seed in range(30):
            plan = plan_series([100, 150, 200, 250], list(range(1, 17)), seed=seed)
            a = {int(k): float(v) for k, v in plan.assignment.items()}
            worst_chan = max(worst_chan, worst_from(sorted(a), a, 8))
            worst_rr = max(worst_rr, worst_from(round_robin_by_level(a), a, 8))

        assert worst_rr < 0.40, f"greedy prefix reached |r| = {worst_rr:.3f}"
        assert worst_chan > 0.70                      # channel order is genuinely bad
        assert worst_rr < worst_chan / 1.5

    def test_it_is_not_claimed_to_work_for_very_short_prefixes(self):
        # A quarter-finished series is not a small version of a good design. With 4-6
        # points there are too few arrangements for any ordering to decorrelate them,
        # and asserting the limit keeps a future change from quietly claiming more.
        from softae.core.thickness_series import plan_series

        worst = 0.0
        for seed in range(30):
            plan = plan_series([100, 150, 200, 250], list(range(1, 17)), seed=seed)
            a = {int(k): float(v) for k, v in plan.assignment.items()}
            order = round_robin_by_level(a)
            worst = max(worst, abs(correlation(order[:4],
                                               [a[c] for c in order[:4]])))
        assert worst > 0.5, "if this now passes, the docstring's table is stale"

    def test_the_order_is_stable_so_a_re_run_casts_the_same_way(self):
        assert round_robin_by_level(ASSIGN) == round_robin_by_level(dict(ASSIGN))

    def test_every_planned_channel_is_cast_exactly_once(self):
        order = round_robin_by_level(ASSIGN)
        assert sorted(order) == sorted(ASSIGN)
        assert len(order) == len(set(order))

    def test_it_handles_an_uneven_series(self):
        # Replicates need not divide evenly; the walk must not drop the remainder.
        uneven = {1: 100.0, 2: 200.0, 3: 100.0, 4: 200.0, 5: 100.0}
        assert sorted(round_robin_by_level(uneven)) == [1, 2, 3, 4, 5]


class TestThePlanIsAuthoritative:
    def test_an_occupied_channel_is_refused_not_substituted(self):
        with pytest.raises(GeometrySeriesError, match="already cast"):
            verify_channels_free([1, 2, 3], occupied={2})

    def test_the_refusal_names_the_channels_and_the_fix(self):
        with pytest.raises(GeometrySeriesError) as exc:
            verify_channels_free([1, 2, 3, 4], occupied={2, 4})
        msg = str(exc.value)
        assert "2, 4" in msg
        assert "softae-thickness plan" in msg

    def test_a_free_board_passes(self):
        verify_channels_free([1, 2, 3], occupied=set())

    def test_occupancy_elsewhere_on_the_board_is_irrelevant(self):
        # A series may use a SUBSET of a board -- 16 of 32 -- so wells outside the plan
        # being occupied must not block it.
        verify_channels_free([1, 2, 3], occupied={20, 21, 22})


class TestDriftControl:
    def test_it_picks_a_mid_level_member_not_an_extreme(self):
        ch = choose_drift_channel(ASSIGN)
        levels = sorted(set(ASSIGN.values()))
        assert ASSIGN[ch] not in (levels[0], levels[-1])

    def test_a_single_level_series_gets_no_drift_control(self):
        # Nothing to control for: with one level there is no slope to corrupt.
        assert choose_drift_channel({1: 100.0, 2: 100.0}) is None

    def test_it_is_stable_so_re_running_names_the_same_channel(self):
        assert choose_drift_channel(ASSIGN) == choose_drift_channel(dict(ASSIGN))

    def test_the_repeat_role_is_not_a_commissioning_role(self):
        # It is a re-measured film, not a fixture artifact. Enrolling it would put it
        # in softae-commission's prompts and capability ladder, where it means nothing.
        from softae.analysis.eis.calibration import (
            COMMISSIONING_ROLES,
            DRIFT_REPEAT_ROLE,
            MEASUREMENT_ROLES,
        )

        assert DRIFT_REPEAT_ROLE in MEASUREMENT_ROLES
        assert DRIFT_REPEAT_ROLE not in COMMISSIONING_ROLES

    def test_the_repeat_is_a_distinct_role_so_the_fit_excludes_it(self):
        # The concrete reason it is not a flag on `sample`: the fit takes the most
        # recent sample spectrum per channel, so a repeat tagged `sample` would REPLACE
        # the in-sequence measurement and move that member to the wrong point in time.
        from softae.analysis.eis.calibration import DRIFT_REPEAT_ROLE

        assert DRIFT_REPEAT_ROLE != "sample"


class TestResolvingTheRun:
    def test_it_resolves_a_plan_into_an_order_and_volumes(self):
        run = plan_geometry_series_run(
            _plan(), volumes_by_level={100.0: [1.0], 150.0: [1.5],
                                       200.0: [2.0], 250.0: [2.5]})
        assert run.n_channels == 16
        assert sorted(run.cast_order) == sorted(ASSIGN)
        assert run.volumes_by_channel[4] == [1.0]      # ch4 is a 100 um member
        assert run.volumes_by_channel[2] == [2.5]      # ch2 is a 250 um member

    def test_an_unsolved_level_is_refused_before_anything_is_cast(self):
        # Finding an infeasible target halfway through a board wastes the board.
        with pytest.raises(GeometrySeriesError, match="no solved volume"):
            plan_geometry_series_run(
                _plan(), volumes_by_level={100.0: [1.0], 150.0: [1.5]})

    def test_an_occupied_channel_stops_the_resolution(self):
        with pytest.raises(GeometrySeriesError, match="already cast"):
            plan_geometry_series_run(
                _plan(), volumes_by_level={100.0: [1.0], 150.0: [1.5],
                                           200.0: [2.0], 250.0: [2.5]},
                occupied={7})

    def test_an_empty_plan_is_refused(self):
        with pytest.raises(GeometrySeriesError, match="assigns no channels"):
            plan_geometry_series_run(_plan({}), volumes_by_level={})

    def test_drift_control_can_be_declined(self):
        run = plan_geometry_series_run(
            _plan(), volumes_by_level={100.0: [1.0], 150.0: [1.5],
                                       200.0: [2.0], 250.0: [2.5]},
            drift_control=False)
        assert run.drift_channel is None
        assert "NO drift control" in run.describe()

    def test_a_sixteen_channel_subset_of_a_thirty_two_well_board_is_normal(self):
        # The operator's constraint: this phase never guarantees full occupancy, so a
        # series must be self-contained on part of a board.
        run = plan_geometry_series_run(
            _plan(), volumes_by_level={100.0: [1.0], 150.0: [1.5],
                                       200.0: [2.0], 250.0: [2.5]},
            occupied=set(range(17, 33)))
        assert run.n_channels == 16
