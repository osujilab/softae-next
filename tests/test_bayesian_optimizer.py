"""Tests for Bayesian optimizer (C3)."""

import pytest
import numpy as np

from softae.errors import OptimizerError
from softae.optimizers import BayesianOptimizer

SIMPLE_SPACE = {
    "x": {"type": "float", "low": 0.0, "high": 10.0},
    "y": {"type": "float", "low": -5.0, "high": 5.0},
}

MIXED_SPACE = {
    "temp": {"type": "float", "low": 25.0, "high": 80.0},
    "rh":   {"type": "int",   "low": 30,   "high": 90},
    "solvent": {"type": "categorical", "choices": ["water", "DMSO", "DMF"]},
}


class TestBayesianInit:
    def test_invalid_acquisition_rejected(self):
        with pytest.raises(OptimizerError, match="acquisition"):
            BayesianOptimizer(SIMPLE_SPACE, acquisition="bad")

    def test_n_initial_zero_rejected(self):
        with pytest.raises(OptimizerError, match="n_initial"):
            BayesianOptimizer(SIMPLE_SPACE, n_initial=0)

    def test_default_construction(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42)
        assert opt.n_trials == 0
        assert opt.best() is None


class TestBayesianWarmUp:
    """During n_initial warm-up, suggestions should be random."""

    def test_warm_up_suggestions_within_bounds(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=3)
        for _ in range(3):
            params = opt.suggest()
            assert params is not None
            assert 0.0 <= params["x"] <= 10.0
            assert -5.0 <= params["y"] <= 5.0
            opt.tell(params, params["x"] * 2 - params["y"])

    def test_reproducible_with_seed(self):
        def run(seed):
            opt = BayesianOptimizer(SIMPLE_SPACE, seed=seed, n_initial=3)
            return [opt.suggest() for _ in range(3)]

        a = run(99)
        b = run(99)
        for p1, p2 in zip(a, b):
            assert p1 == p2


class TestBayesianSuggestTell:
    """After warm-up, GP-guided suggestions should be valid."""

    def test_gp_guided_suggestion_within_bounds(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=3, n_candidates=500)
        # Warm up
        for _ in range(3):
            p = opt.suggest()
            opt.tell(p, p["x"] - abs(p["y"]))
        # GP-guided
        p = opt.suggest()
        assert p is not None
        assert 0.0 <= p["x"] <= 10.0
        assert -5.0 <= p["y"] <= 5.0

    def test_best_tracks_maximum(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=3, objective="maximize")
        values = []
        for _ in range(5):
            p = opt.suggest()
            v = p["x"] - abs(p["y"])
            opt.tell(p, v)
            values.append(v)
        best_params, best_val = opt.best()
        assert best_val == max(values)

    def test_best_tracks_minimum(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=3, objective="minimize")
        values = []
        for _ in range(5):
            p = opt.suggest()
            v = p["x"] + abs(p["y"])
            opt.tell(p, v)
            values.append(v)
        best_params, best_val = opt.best()
        assert best_val == min(values)


class TestBayesianMixedSpace:
    def test_mixed_space_suggest_valid(self):
        opt = BayesianOptimizer(MIXED_SPACE, seed=42, n_initial=3, n_candidates=200)
        for _ in range(5):
            p = opt.suggest()
            assert 25.0 <= p["temp"] <= 80.0
            assert 30 <= p["rh"] <= 90
            assert isinstance(p["rh"], int)
            assert p["solvent"] in ["water", "DMSO", "DMF"]
            opt.tell(p, p["temp"] * 0.1)


class TestBayesianAcquisition:
    def test_ucb_acquisition(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=2, acquisition="ucb")
        for _ in range(3):
            p = opt.suggest()
            opt.tell(p, p["x"])
        assert opt.n_trials == 3

    def test_ei_acquisition(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=2, acquisition="ei")
        for _ in range(3):
            p = opt.suggest()
            opt.tell(p, p["x"])
        assert opt.n_trials == 3

    def test_history_length(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=42, n_initial=2)
        for i in range(4):
            p = opt.suggest()
            opt.tell(p, float(i))
        assert len(opt.history) == 4


class TestBayesianConvergence:
    """The GP-guided phase should drive suggestions toward a known optimum.

    Exact suggested coordinates are not asserted (they depend on GP internals);
    instead we assert the *property* that the optimizer converges near the
    optimum of a smooth synthetic objective and improves on warm-up alone.
    """

    OPTIMUM = {"x": 7.0, "y": 2.0}

    @staticmethod
    def _objective(p):
        # Maximized at (7, 2); smooth concave bowl.
        return -((p["x"] - 7.0) ** 2 + (p["y"] - 2.0) ** 2)

    @pytest.mark.parametrize("acquisition", ["ucb", "ei"])
    def test_converges_toward_optimum(self, acquisition):
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=1, n_initial=5, acquisition=acquisition
        )
        # Warm-up phase, then record the best achievable from random alone.
        for _ in range(5):
            p = opt.suggest()
            opt.tell(p, self._objective(p))
        warmup_best = opt.best()[1]

        # GP-guided phase.
        for _ in range(25):
            p = opt.suggest()
            assert 0.0 <= p["x"] <= 10.0  # suggestions stay in bounds
            assert -5.0 <= p["y"] <= 5.0
            opt.tell(p, self._objective(p))

        best_params, best_val = opt.best()
        dist = ((best_params["x"] - 7.0) ** 2 + (best_params["y"] - 2.0) ** 2) ** 0.5
        assert dist < 0.5  # converged near the true optimum
        assert best_val >= warmup_best  # guided search improves on warm-up


class TestBayesianPriorMean:
    """Physically/prior-informed BO: the GP models the residual from a prior."""

    OPTIMUM = {"x": 7.0, "y": 2.0}

    @staticmethod
    def _objective(p):
        return -((p["x"] - 7.0) ** 2 + (p["y"] - 2.0) ** 2)

    def test_prior_mean_none_is_unchanged(self):
        """prior_mean=None and prior_mean=lambda: 0 produce identical search."""
        def run(prior):
            opt = BayesianOptimizer(
                SIMPLE_SPACE, seed=7, n_initial=3, n_candidates=300,
                prior_mean=prior,
            )
            out = []
            for _ in range(6):
                p = opt.suggest()
                opt.tell(p, self._objective(p))
                out.append((round(p["x"], 6), round(p["y"], 6)))
            return out

        assert run(None) == run(lambda _p: 0.0)

    def test_exact_prior_accelerates_convergence(self):
        """With the objective itself as the prior mean, the residual GP is ~flat,
        so the acquisition rides the prior straight to the optimum in just a few
        guided steps — far fewer than the ~25 the no-prior path needs."""
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=3, n_initial=3, n_candidates=800,
            acquisition="ucb", prior_mean=self._objective,
        )
        for _ in range(3):  # warm-up
            p = opt.suggest()
            opt.tell(p, self._objective(p))
        for _ in range(5):  # only 5 guided steps
            p = opt.suggest()
            assert 0.0 <= p["x"] <= 10.0 and -5.0 <= p["y"] <= 5.0
            opt.tell(p, self._objective(p))

        bp, _ = opt.best()
        dist = ((bp["x"] - 7.0) ** 2 + (bp["y"] - 2.0) ** 2) ** 0.5
        assert dist < 1.0  # prior guided it near-optimum quickly


class TestBayesianBatch:
    """Constant-liar batched suggestion (q>1 diverse points per round)."""

    def test_batch_returns_q_points_and_leaves_history_clean(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1, n_initial=3, n_candidates=400)
        # Get past warm-up with real observations so the GP drives diversity.
        for _ in range(4):
            p = opt.suggest()
            opt.tell(p, p["x"] - abs(p["y"]))
        n_before = opt.n_trials
        batch = opt.suggest_batch(3)
        assert len(batch) == 3
        # Liars must not linger in history.
        assert opt.n_trials == n_before
        for p in batch:
            assert 0.0 <= p["x"] <= 10.0 and -5.0 <= p["y"] <= 5.0

    def test_batch_points_are_distinct(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=2, n_initial=3, n_candidates=800)
        for _ in range(5):
            p = opt.suggest()
            opt.tell(p, -((p["x"] - 5) ** 2 + p["y"] ** 2))  # peak near (5,0)
        batch = opt.suggest_batch(4)
        # Constant-liar should spread the batch: no two identical encoded points.
        keys = {opt._encoder.key(p) for p in batch}
        assert len(keys) == len(batch)

    def test_batch_q1_matches_single_suggest(self):
        a = BayesianOptimizer(SIMPLE_SPACE, seed=5, n_initial=2)
        b = BayesianOptimizer(SIMPLE_SPACE, seed=5, n_initial=2)
        assert a.suggest_batch(1) == [b.suggest()]

    def test_batch_invalid_q_raises(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1)
        with pytest.raises(OptimizerError):
            opt.suggest_batch(0)


class TestBatchStrategies:
    """Pluggable batch-proposal strategies (constant-liar, Kriging-believer, hook)."""

    def _warmed(self, strategy):
        from softae.optimizers import BayesianOptimizer
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=3, n_initial=3, n_candidates=500,
            batch_strategy=strategy,
        )
        for _ in range(5):
            p = opt.suggest()
            opt.tell(p, -((p["x"] - 5) ** 2 + p["y"] ** 2))
        return opt

    def test_registry_and_resolution(self):
        from softae.optimizers.batch import (
            BATCH_STRATEGIES, ConstantLiarStrategy, KrigingBelieverStrategy,
            make_batch_strategy,
        )
        assert set(BATCH_STRATEGIES) >= {"constant_liar", "kriging_believer", "botorch_mc"}
        assert isinstance(make_batch_strategy("kriging_believer"), KrigingBelieverStrategy)
        inst = ConstantLiarStrategy()
        assert make_batch_strategy(inst) is inst  # instance passthrough
        with pytest.raises(OptimizerError):
            make_batch_strategy("nope")

    def test_default_strategy_is_constant_liar(self):
        from softae.optimizers.batch import ConstantLiarStrategy
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1)
        assert isinstance(opt._batch_strategy, ConstantLiarStrategy)

    @pytest.mark.parametrize("strategy", ["constant_liar", "kriging_believer"])
    def test_strategy_returns_q_distinct_clean(self, strategy):
        opt = self._warmed(strategy)
        n_before = opt.n_trials
        batch = opt.suggest_batch(4)
        assert len(batch) == 4
        assert opt.n_trials == n_before  # no fantasies linger
        keys = {opt._encoder.key(p) for p in batch}
        assert len(keys) == 4  # diversified

    def test_kriging_believer_posterior_mean_fallback(self):
        # Before any fit (empty history), _posterior_mean must not raise.
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1, batch_strategy="kriging_believer")
        assert opt._posterior_mean({"x": 1.0, "y": 0.0}) == 0.0

    def test_botorch_mc_is_a_guarded_stub(self):
        import importlib.util
        from softae.optimizers.batch import BoTorchMonteCarloStrategy
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1, batch_strategy="botorch_mc")
        # Raises either the missing-dependency error or the not-yet-implemented
        # seam — never silently returns wrong results.
        expected = (
            NotImplementedError if importlib.util.find_spec("botorch") else OptimizerError
        )
        with pytest.raises(expected):
            opt.suggest_batch(3)


class TestDecisionRtol:
    """ATLAS-style tolerance-set argmax (decision_rtol; default 0 = strict)."""

    @staticmethod
    def _objective(p):
        return -((p["x"] - 7.0) ** 2 + (p["y"] - 2.0) ** 2)

    def test_invalid_rtol_rejected(self):
        with pytest.raises(OptimizerError, match="decision_rtol"):
            BayesianOptimizer(SIMPLE_SPACE, decision_rtol=-0.1)

    def test_rtol_zero_preserves_argmax_bit_for_bit(self):
        """Defaults and an explicit decision_rtol=0.0 trace identical searches.

        The off path must not touch the RNG, so the whole interleaved
        suggest/tell sequence — warm-up and GP-guided — is bit-identical."""
        def run(**kw):
            opt = BayesianOptimizer(
                SIMPLE_SPACE, seed=11, n_initial=3, n_candidates=300, **kw
            )
            out = []
            for _ in range(7):
                p = opt.suggest()
                opt.tell(p, self._objective(p))
                out.append((p["x"], p["y"]))
            return out

        assert run() == run(decision_rtol=0.0)

    def test_rtol_selects_only_within_tolerance_set(self):
        """Every draw lands in the tolerance set, and ties are actually mixed."""
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=0, n_initial=2, decision_rtol=0.05
        )
        cand_X = np.zeros((4, 2))  # positions irrelevant without exclusion
        scores = np.array([1.0, 0.99, 0.5, 0.97])
        # Threshold = 1.0 - 0.05*|1.0| = 0.95 → tolerance set is {0, 1, 3}.
        picks = {opt._select_index(cand_X, scores) for _ in range(60)}
        assert picks <= {0, 1, 3}
        assert len(picks) > 1  # uniform draw, not a disguised argmax

    def test_rtol_zero_select_index_is_strict_argmax(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=0)
        scores = np.array([0.2, 0.9, 0.9, 0.1])  # exact tie → first index wins
        for _ in range(5):
            assert opt._select_index(np.zeros((4, 2)), scores) == 1

    def test_rtol_guards_flat_acquisition_surface(self):
        """On a perfectly flat surface the whole pool is the tolerance set."""
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=1, decision_rtol=0.01)
        scores = np.zeros(50)
        picks = {opt._select_index(np.zeros((50, 2)), scores) for _ in range(40)}
        assert len(picks) > 1  # not stuck on index 0

    def test_rtol_draw_uses_checkpointed_rng(self):
        """The tie-break draw rides the optimizer's own (serialized) RNG."""
        import json
        from softae.optimizers.base import BaseOptimizer

        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=5, n_initial=3, n_candidates=200,
            decision_rtol=0.2,
        )
        for _ in range(4):
            p = opt.suggest()
            opt.tell(p, self._objective(p))
        blob = json.dumps(opt.to_dict())
        expected = opt.suggest()
        restored = BaseOptimizer.from_dict(json.loads(blob))
        assert restored.suggest() == expected


class TestExclusionRadius:
    """ATLAS-style measured-point exclusion (exclusion_radius; default None)."""

    @staticmethod
    def _norm(p):
        # SIMPLE_SPACE normalized to the unit square.
        return np.array([p["x"] / 10.0, (p["y"] + 5.0) / 10.0])

    def test_invalid_radius_rejected(self):
        with pytest.raises(OptimizerError, match="exclusion_radius"):
            BayesianOptimizer(SIMPLE_SPACE, exclusion_radius=0.0)

    def test_excluded_candidate_is_not_selected(self):
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=0, exclusion_radius=0.3)
        opt.tell({"x": 5.0, "y": 0.0}, 1.0)
        cand = [{"x": 5.0, "y": 0.0}, {"x": 9.0, "y": 4.0}]
        cand_X = np.array([opt._encoder.encode(p) for p in cand])
        scores = np.array([10.0, 1.0])  # best score sits on the measured point
        assert opt._select_index(cand_X, scores) == 1

    def test_saturated_exclusion_falls_back_to_unexcluded_argmax(self):
        # Radius 10 in the unit square (diameter √2) excludes everything.
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=0, exclusion_radius=10.0)
        opt.tell({"x": 5.0, "y": 0.0}, 1.0)
        cand = [{"x": 5.0, "y": 0.0}, {"x": 9.0, "y": 4.0}]
        cand_X = np.array([opt._encoder.encode(p) for p in cand])
        scores = np.array([10.0, 1.0])
        assert opt._select_index(cand_X, scores) == 0  # plain argmax

    def test_saturated_exclusion_suggest_does_not_fail(self):
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=2, n_initial=3, n_candidates=200,
            exclusion_radius=10.0,
        )
        for _ in range(4):
            p = opt.suggest()
            assert p is not None
            assert 0.0 <= p["x"] <= 10.0 and -5.0 <= p["y"] <= 5.0
            opt.tell(p, p["x"])

    def test_guided_suggestions_avoid_measured_neighborhoods(self):
        radius = 0.15
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=4, n_initial=3, n_candidates=500,
            exclusion_radius=radius,
        )
        for _ in range(3):  # warm-up (exclusion applies to the argmax only)
            p = opt.suggest()
            opt.tell(p, -((p["x"] - 7.0) ** 2 + (p["y"] - 2.0) ** 2))
        for _ in range(5):  # guided phase
            p = opt.suggest()
            dists = [
                np.linalg.norm(self._norm(p) - self._norm(q))
                for q, _ in opt.history
            ]
            assert min(dists) >= radius
            opt.tell(p, -((p["x"] - 7.0) ** 2 + (p["y"] - 2.0) ** 2))

    def test_exclusion_off_matches_legacy_search(self):
        """exclusion_radius=None (default) changes nothing, draw for draw."""
        def run(**kw):
            opt = BayesianOptimizer(
                SIMPLE_SPACE, seed=9, n_initial=3, n_candidates=300, **kw
            )
            out = []
            for _ in range(6):
                p = opt.suggest()
                opt.tell(p, p["x"] - abs(p["y"]))
                out.append((p["x"], p["y"]))
            return out

        assert run() == run(exclusion_radius=None)


class TestAtlasSerialization:
    """decision_rtol / exclusion_radius survive to_dict/from_dict."""

    def _round_trip(self, opt):
        import json
        from softae.optimizers.base import BaseOptimizer

        return BaseOptimizer.from_dict(json.loads(json.dumps(opt.to_dict())))

    def test_fields_round_trip(self):
        opt = BayesianOptimizer(
            SIMPLE_SPACE, seed=1, decision_rtol=0.02, exclusion_radius=0.1
        )
        state = opt.to_dict()
        assert state["extra"]["decision_rtol"] == 0.02
        assert state["extra"]["exclusion_radius"] == 0.1

        restored = self._round_trip(opt)
        assert restored._decision_rtol == 0.02
        assert restored._exclusion_radius == 0.1
        assert restored.to_dict()["extra"]["decision_rtol"] == 0.02
        assert restored.to_dict()["extra"]["exclusion_radius"] == 0.1

    def test_defaults_round_trip_as_off(self):
        restored = self._round_trip(BayesianOptimizer(SIMPLE_SPACE, seed=1))
        assert restored._decision_rtol == 0.0
        assert restored._exclusion_radius is None

    def test_legacy_checkpoint_without_fields_restores_off(self):
        """A pre-enhancement checkpoint (no keys in extra) resumes with both off."""
        import json
        from softae.optimizers.base import BaseOptimizer

        state = json.loads(json.dumps(BayesianOptimizer(SIMPLE_SPACE, seed=1).to_dict()))
        state["extra"].pop("decision_rtol")
        state["extra"].pop("exclusion_radius")
        restored = BaseOptimizer.from_dict(state)
        assert restored._decision_rtol == 0.0
        assert restored._exclusion_radius is None


class TestBayesianWarmStart:
    """Seed observations fed via tell() before the first suggest()."""

    def test_seeds_count_toward_warmup(self):
        # n_initial=3; seed 3 prior points → the next suggest is GP-guided.
        opt = BayesianOptimizer(SIMPLE_SPACE, seed=0, n_initial=3, n_candidates=300)
        seeds = [
            ({"x": 7.0, "y": 2.0}, 10.0),
            ({"x": 1.0, "y": -4.0}, 1.0),
            ({"x": 5.0, "y": 0.0}, 4.0),
        ]
        for params, value in seeds:
            opt.tell(params, value)
        assert opt.n_trials == 3
        # History is now at n_initial, so this suggestion is surrogate-driven and
        # must remain in-bounds (exercises the GP fit on the seeded data).
        p = opt.suggest()
        assert 0.0 <= p["x"] <= 10.0 and -5.0 <= p["y"] <= 5.0
        assert opt.best()[0] == {"x": 7.0, "y": 2.0}  # best seed retained
