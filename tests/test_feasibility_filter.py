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

from softae.optimizers.bayesian import BayesianOptimizer

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

    def test_an_impossible_filter_still_returns_a_point(self):
        """Returning None would read as 'exhausted' and end the run; better to
        propose and let the overflow guard refuse the cast."""
        opt = _opt()
        opt.feasibility_fn = lambda p: False
        assert opt.suggest() is not None


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
    def test_a_raising_filter_is_treated_as_feasible(self):
        """Refusing every point on a bug would silently stall the campaign."""
        opt = _opt()
        opt.feasibility_fn = lambda p: (_ for _ in ()).throw(RuntimeError("boom"))
        assert opt.suggest() is not None

    def test_every_optimizer_carries_the_attribute(self):
        from softae.optimizers.grid import GridSearchOptimizer
        from softae.optimizers.random import RandomSearchOptimizer

        assert RandomSearchOptimizer(SPACE, n_trials=3).feasibility_fn is None
        assert GridSearchOptimizer(SPACE).feasibility_fn is None

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
