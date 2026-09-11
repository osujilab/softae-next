"""Twin-evaluated feasibility filter at suggestion time (P7.1).

Enforcement, as distinct from the advisory preflight sweep: an infeasible point
is never *proposed*, rather than proposed and then refused.

The constraint is known, deterministic and cheap — the twin is arithmetic, not
an experiment — so restricting the acquisition maximiser's domain is exact and
needs no constraint GP. It also avoids bounds-clipping, which is unsound here:
overflow is a diagonal half-space (Σvᵢ ≤ capacity), so axis-aligned clipping to
capacity/n would discard feasible points.
"""

from __future__ import annotations

import pytest

from softae.optimizers.base import Feasibility
from softae.optimizers.bayesian import BayesianOptimizer
from softae.optimizers.grid import GridSearchOptimizer
from softae.optimizers.random import RandomSearchOptimizer

SPACE = {
    "v0": {"type": "float", "low": 0.0, "high": 100.0},
    "v1": {"type": "float", "low": 0.0, "high": 100.0},
}


def _opt(**kw):
    return BayesianOptimizer(SPACE, objective="maximize", seed=7, **kw)


def _total_under(cap):
    return lambda p: (p["v0"] + p["v1"]) <= cap


class TestWarmUp:
    def test_warm_up_points_respect_the_filter(self):
        """The warm-up is random exactly where the guard matters most."""
        opt = _opt()
        opt.feasibility_fn = _total_under(40.0)
        for _ in range(8):
            p = opt.suggest()
            assert p["v0"] + p["v1"] <= 40.0
            opt.tell(p, 1.0)

    def test_no_filter_means_no_constraint(self):
        assert _opt().suggest() is not None

    def test_an_impossible_filter_exhausts_rather_than_proposing_a_known_bad_point(self):
        """CONTRACT CHANGE (W2). This used to return a point the filter had just
        refused, on the grounds that ``None`` reads as 'exhausted' and ends the
        run. It does end the run — correctly: spending three wells' identities to
        prove arithmetic is worse than saying so. ``n_rejected`` is what tells the
        loop this was *search_space_infeasible*, not convergence."""
        opt = _opt()
        opt.feasibility_fn = lambda p: False
        assert opt.suggest() is None
        assert opt._n_rejected == opt._feasibility_max_tries


class TestAcquisition:
    def _fit(self, opt, n=6):
        for _ in range(n):
            p = opt.suggest()
            opt.tell(p, p["v0"] + p["v1"])      # reward large totals
        return opt

    def test_the_argmax_never_proposes_an_infeasible_point(self):
        """The objective actively pulls toward the infeasible corner, so an
        unfiltered optimizer would head straight there."""
        opt = self._fit(_opt())
        opt.feasibility_fn = _total_under(30.0)
        for _ in range(5):
            p = opt.suggest()
            assert p["v0"] + p["v1"] <= 30.0
            opt.tell(p, p["v0"] + p["v1"])

    def test_the_diagonal_constraint_admits_a_lopsided_point(self):
        """Bounds-clipping to capacity/n would exclude these; filtering does not."""
        opt = _opt()
        opt.feasibility_fn = _total_under(100.0)
        assert any(
            opt._is_feasible(p) and (p["v0"] > 60.0 or p["v1"] > 60.0)
            for p in (opt._random_point() for _ in range(200))
        )


class TestRobustness:
    def test_a_raising_filter_is_still_admitted(self):
        """Refusing every point on a bug would silently stall the campaign.

        Admitted, but no longer *as feasible* — see
        ``test_a_raising_feasibility_fn_is_unchecked_and_not_feasible``."""
        opt = _opt()
        opt.feasibility_fn = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
        assert opt.suggest() is not None

    def test_every_optimizer_carries_the_attribute(self):
        assert RandomSearchOptimizer(SPACE, n_trials=3).feasibility_fn is None
        assert GridSearchOptimizer(SPACE).feasibility_fn is None
        assert RandomSearchOptimizer(SPACE, n_trials=3).check_feasible_fn is None
        assert GridSearchOptimizer(SPACE).check_feasible_fn is None

    def test_the_hook_is_not_serialized(self):
        """A live callable belonging to the host's twin; a resumed run rebuilds
        it from the spec rather than restoring a stale closure."""
        opt = _opt()
        opt.feasibility_fn = _total_under(50.0)
        assert "feasibility_fn" not in opt.to_dict()


class TestBatch:
    def test_a_q_batch_filters_every_point(self):
        """The fantasy loop calls suggest() per point, so it inherits the filter."""
        opt = _opt()
        for _ in range(6):
            p = opt.suggest()
            opt.tell(p, p["v0"] + p["v1"])
        opt.feasibility_fn = _total_under(25.0)

        batch = opt.suggest_batch(4)

        assert len(batch) == 4
        for p in batch:
            assert p["v0"] + p["v1"] <= 25.0


# ── W2 — tri-state feasibility for EVERY optimizer ───────────────────────────
#
# The filter used to live inside ``BayesianOptimizer.suggest`` alone, so a random
# or grid campaign carried the attribute and ignored it. ``BaseOptimizer.suggest``
# is now a template around a subclass ``_propose``, and the verdict is tri-state:
# "could not check" is no longer spelled with the same token as "checked, fine".


class TestTriStateContract:
    def test_a_raising_feasibility_fn_is_unchecked_and_not_feasible(self):
        """Could-not-check must not read as feasible (SUBAGENT_RULES 3.1a).

        It is still *admitted* — carving an invisible hole in the search space
        on a bug is worse — but it is counted and reportable, where before it
        was indistinguishable from a point the twin had approved.
        """
        opt = _opt()
        opt.feasibility_fn = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))

        assert opt.check_feasible({"v0": 1.0, "v1": 1.0}) is Feasibility.UNCHECKED
        assert opt.suggest() is not None
        assert opt._n_unchecked >= 1
        assert opt._n_rejected == 0

    def test_an_uncapped_board_reports_unchecked_rather_than_no_filter(self):
        """The second spelling of "unknown" as "fine".

        A board declaring no capacity hands down a filter that cannot answer,
        not the absence of a filter. Both admit the point; only one of them can
        be counted, and an optimizer with no declared constraint at all must not
        report every draw as unchecked.
        """
        opt = _opt()
        opt.check_feasible_fn = lambda p: Feasibility.UNCHECKED
        assert opt.check_feasible({"v0": 1.0, "v1": 1.0}) is Feasibility.UNCHECKED
        opt.suggest()
        assert opt._n_unchecked == 1

        legacy = _opt()
        legacy.feasibility_fn = lambda p: None          # the bool-ish spelling
        assert legacy.check_feasible({"v0": 1.0}) is Feasibility.UNCHECKED

        unconstrained = _opt()
        assert unconstrained.check_feasible({"v0": 1.0}) is Feasibility.FEASIBLE
        unconstrained.suggest()
        assert unconstrained._n_unchecked == 0

    def test_a_tri_state_callable_is_consumed_directly(self):
        opt = _opt()
        opt.check_feasible_fn = lambda p: (
            Feasibility.FEASIBLE if p["v0"] <= 10.0 else Feasibility.INFEASIBLE)
        for _ in range(5):
            assert opt.suggest()["v0"] <= 10.0
        assert opt._n_rejected > 0


class TestRandomSearch:
    def _random(self, **kw):
        return RandomSearchOptimizer(SPACE, objective="maximize", seed=3, **kw)

    def test_random_search_refuses_an_infeasible_draw_and_redraws(self):
        """POSITIVE CONTROL: random search never read the filter at all."""
        opt = self._random(n_trials=10)
        opt.feasibility_fn = _total_under(40.0)     # ~8 % of the box
        for _ in range(10):
            p = opt.suggest()
            assert p is not None
            assert p["v0"] + p["v1"] <= 40.0
        assert opt._n_rejected > 0                  # it really did redraw

    def test_rejected_draws_do_not_consume_the_random_search_budget(self):
        """The budget counts trials the campaign will run, not dice rolls."""
        opt = self._random(n_trials=3)
        opt.feasibility_fn = _total_under(40.0)
        assert all(opt.suggest() is not None for _ in range(3))
        assert opt.suggest() is None                # exactly 3, not fewer
        assert opt._n_suggested == 3
        assert opt._n_rejected > 0

    def test_batch_proposal_inherits_the_feasibility_filter_from_suggest(self):
        """suggest_batch is q suggest() calls, so the template covers it."""
        opt = self._random(n_trials=20)
        opt.feasibility_fn = _total_under(30.0)
        batch = opt.suggest_batch(5)
        assert len(batch) == 5
        assert all(p["v0"] + p["v1"] <= 30.0 for p in batch)


class TestGridSearch:
    def test_grid_search_skips_infeasible_points_and_reports_rejection_not_convergence(self):
        """POSITIVE CONTROL: grid never read the filter, and a grid exhausted by
        rejection must not print the same word as one exhausted by search."""
        opt = GridSearchOptimizer(SPACE, n_points=4)      # 16 points, only (0,0) fits
        opt.feasibility_fn = _total_under(10.0)

        assert opt.suggest() == {"v0": 0.0, "v1": 0.0}
        assert opt.suggest() is None                      # nothing else qualifies
        assert opt._n_rejected == 15
        assert opt._n_rejected == opt.n_grid_points - 1   # exhausted by rejection

    def test_a_grid_with_no_filter_still_walks_every_point(self):
        opt = GridSearchOptimizer(SPACE, n_points=3)
        assert sum(1 for _ in iter(opt.suggest, None)) == 9
        assert opt._n_rejected == 0                       # genuine convergence


class TestPooledPool:
    def test_the_pool_optimizer_filters_its_candidate_pool(self):
        from softae.optimizers.pooled_bayesian import PooledBayesianOptimizer

        pool = [{"v0": float(i), "v1": 0.0} for i in range(10)]
        opt = PooledBayesianOptimizer(SPACE, seed=1, pool=pool, n_initial=2)
        opt.feasibility_fn = lambda p: p["v0"] <= 4.0
        for _ in range(5):                    # exactly the admissible points
            p = opt.suggest()
            assert p is not None
            assert p["v0"] <= 4.0
            opt.tell(p, p["v0"])
        assert opt.suggest() is None          # the admissible pool is spent
        # ...and that None is rejection, not convergence: 5 refused points left.
        assert opt._n_rejected == 5


class TestWiring:
    def test_no_declared_capacity_means_no_filter(self):
        """Rather than one that admits everything and costs a solve per candidate."""
        import softae.core.autonomous_wiring as aw

        orig = aw.campaign_well_capacity_uL
        aw.campaign_well_capacity_uL = lambda spec: None
        try:
            assert aw.twin_feasibility_fn(object()) is None
        finally:
            aw.campaign_well_capacity_uL = orig

    def test_an_unsolvable_point_is_admitted_not_carved_out(self):
        """An unsolvable point is not necessarily an overflowing one — fail
        loudly downstream rather than silently shrinking the search space."""
        import softae.core.autonomous_wiring as aw

        orig_cap, orig_total = aw.campaign_well_capacity_uL, aw._trial_total_uL
        aw.campaign_well_capacity_uL = lambda spec: 50.0
        aw._trial_total_uL = lambda spec, p: (_ for _ in ()).throw(
            RuntimeError("infeasible target set"))
        try:
            assert aw.twin_feasibility_fn(object())({"v0": 1.0}) is True
        finally:
            aw.campaign_well_capacity_uL, aw._trial_total_uL = orig_cap, orig_total

    def test_the_filter_enforces_the_declared_capacity(self):
        import softae.core.autonomous_wiring as aw

        orig_cap, orig_total = aw.campaign_well_capacity_uL, aw._trial_total_uL
        aw.campaign_well_capacity_uL = lambda spec: 50.0
        aw._trial_total_uL = lambda spec, p: p["total"]
        try:
            fn = aw.twin_feasibility_fn(object())
            assert fn({"total": 49.0}) is True
            assert fn({"total": 50.0}) is True        # at capacity is allowed
            assert fn({"total": 51.0}) is False
        finally:
            aw.campaign_well_capacity_uL, aw._trial_total_uL = orig_cap, orig_total
