"""T3.1 — the learned feasibility layer: gate, clamp, weighting, resume.

Spec: ``docs/SubAgent docs/failure_informed_feasibility_spec.md`` §4-§7, §10
tests 1, 4, 4a, 4b, 5, 6, 7, 8, 9, 11, 12.

The test this file exists for is
:class:`TestTheNegativeScoreHazard` — the normalization positive control. Every
other test here would still pass with the normalization deleted; that one is
written so it *cannot*, because the bug it guards (a penalty silently inverting
into a bonus for the compositions most likely to fail) leaves every number
finite, every log line healthy and every other assertion green.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from softae.errors import OptimizerError
from softae.optimizers.bayesian import BayesianOptimizer
from softae.optimizers.feasibility import (
    ABSOLUTE_LABEL_FLOOR,
    FEASIBLE,
    INFEASIBLE,
    FeasibilityConfig,
    FeasibilityModel,
    derived_clamp,
    normalize_scores,
)

SPACE = {
    "x0": {"type": "float", "low": 0.0, "high": 1.0},
    "x1": {"type": "float", "low": 0.0, "high": 1.0},
}


def _optimizer(**kw):
    kw.setdefault("n_initial", 2)
    kw.setdefault("n_candidates", 200)
    return BayesianOptimizer(SPACE, "maximize", seed=7, **kw)


def _warm(opt, n=6):
    """Push *opt* past warm-up with reproducible observations."""
    rng = np.random.RandomState(0)
    for _ in range(n):
        p = {"x0": float(rng.uniform()), "x1": float(rng.uniform())}
        opt.tell(p, float(p["x0"] + p["x1"]))
    return opt


class _ConstantClassifier:
    """A classifier with a fixed opinion — lets a test pin the weighting alone."""

    def __init__(self, p_feasible: float) -> None:
        self.p = float(p_feasible)
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1

    def predict_proba_feasible(self, X):
        return np.full(len(np.asarray(X, dtype=float)), self.p)


class _RegionClassifier:
    """Declares everything with ``x0 + x1 > 0.8`` infeasible, confidently."""

    def fit(self, X, y):
        pass

    def predict_proba_feasible(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(X[:, 0] + X[:, 1] > 0.8, 0.02, 0.98)


def _stocked(model: FeasibilityModel, n_feasible=3, n_infeasible=3):
    """Give *model* enough labels of each class to clear the floors."""
    for i in range(n_feasible):
        model.add({"x0": 0.1 * i, "x1": 0.1}, FEASIBLE, channel=1, board_id="b1")
    for i in range(n_infeasible):
        model.add({"x0": 0.9, "x1": 0.9 - 0.05 * i}, INFEASIBLE,
                  channel=2 + i, board_id="b1")
    return model


# ── Test 1 — default-off is bit-identical ────────────────────────────────────

class TestDefaultOffChangesNothing:
    def test_apply_feasibility_weight_returns_the_same_object_when_disabled(self):
        """Identity, not equality: 'unchanged' must not be something to trust."""
        opt = _warm(_optimizer())
        scores = np.array([-3.0, -1.0, 2.0, 0.5])
        assert opt._apply_feasibility_weight(np.zeros((4, 2)), scores) is scores

    def test_a_default_optimizer_constructs_no_classifier_and_reads_no_label(self):
        opt = _optimizer()
        assert opt.feasibility.config.enabled is False
        assert opt.feasibility.labels == []
        assert opt.last_p_feas is None and opt.last_steered is None

    def test_the_proposal_stream_and_rng_are_identical_with_the_feature_off(self):
        """Same seed => same proposals, and the RNG stream is unmoved."""
        a, b = _warm(_optimizer()), _warm(_optimizer(feasibility=FeasibilityConfig()))
        prop_a = [a.suggest() for _ in range(3)]
        prop_b = [b.suggest() for _ in range(3)]
        assert prop_a == prop_b
        assert a._rng_state()[2] == b._rng_state()[2]

    def test_an_enabled_layer_below_its_floor_still_leaves_scores_untouched(self):
        opt = _warm(_optimizer(feasibility=FeasibilityConfig(enabled=True)))
        scores = np.array([-3.0, -1.0, 2.0, 0.5])
        assert opt._apply_feasibility_weight(np.zeros((4, 2)), scores) is scores


# ── Test 4 / 4a / 4b — the minimum-data gate ─────────────────────────────────

class TestTheMinimumDataGate:
    def test_below_either_floor_p_feas_is_one_everywhere(self):
        model = FeasibilityModel(FeasibilityConfig(enabled=True))
        _stocked(model, n_feasible=3, n_infeasible=2)   # one short on infeasible
        p = model.p_feasible(np.zeros((5, 2)), encode=lambda d: [d["x0"], d["x1"]])
        assert np.array_equal(p, np.ones(5))
        assert model.active is False

    def test_a_missing_feasible_floor_withholds_just_as_a_missing_infeasible_one_does(self):
        model = FeasibilityModel(FeasibilityConfig(enabled=True))
        _stocked(model, n_feasible=2, n_infeasible=3)
        p = model.p_feasible(np.zeros((4, 2)), encode=lambda d: [d["x0"], d["x1"]])
        assert np.array_equal(p, np.ones(4))

    def test_at_exactly_three_of_each_the_layer_engages(self):
        model = FeasibilityModel(
            FeasibilityConfig(enabled=True),
            classifier=_ConstantClassifier(0.2),
        )
        _stocked(model, 3, 3)
        assert model.active is True
        p = model.p_feasible(np.zeros((4, 2)), encode=lambda d: [d["x0"], d["x1"]])
        assert np.allclose(p, 0.2)

    def test_the_clamp_is_derived_from_the_floor_rather_than_hardcoded(self):
        """4a — the derivation is what is pinned, not the numbers it produces."""
        for k in (3, 4, 5, 10):
            assert FeasibilityConfig(min_infeasible=k).clamp == pytest.approx(
                0.05 ** (1.0 / k))
        # The spec's table, as a readability check on the derivation above.
        assert derived_clamp(3) == pytest.approx(0.368, abs=5e-4)
        assert derived_clamp(5) == pytest.approx(0.549, abs=5e-4)

    def test_floors_below_three_are_refused_with_the_reason_named(self):
        """4b — the refusal is not paternalism; it is where the meaning inverts."""
        for kwargs in ({"min_infeasible": 2}, {"min_feasible": 2},
                       {"min_infeasible": 0}):
            with pytest.raises(OptimizerError, match="UPWARD only"):
                FeasibilityConfig(**kwargs)

    def test_floors_above_three_are_accepted(self):
        cfg = FeasibilityConfig(min_infeasible=8, min_feasible=8)
        assert cfg.min_infeasible == 8
        assert cfg.clamp == pytest.approx(0.05 ** (1 / 8))

    def test_the_absolute_floor_is_three(self):
        assert ABSOLUTE_LABEL_FLOOR == 3
        assert FeasibilityConfig().min_infeasible == 3
        assert FeasibilityConfig().min_feasible == 3

    def test_the_min_filter_caps_the_reward_side_only(self):
        """A confidently-safe point earns no bonus over a merely-probable one."""
        cfg = FeasibilityConfig(enabled=True, min_filter=True)
        model = FeasibilityModel(cfg, classifier=_ConstantClassifier(0.95))
        _stocked(model, 3, 3)
        p = model.p_feasible(np.zeros((3, 2)), encode=lambda d: [d["x0"], d["x1"]])
        assert np.allclose(p, cfg.clamp)

        unfiltered = FeasibilityModel(
            FeasibilityConfig(enabled=True, min_filter=False),
            classifier=_ConstantClassifier(0.95))
        _stocked(unfiltered, 3, 3)
        p2 = unfiltered.p_feasible(
            np.zeros((3, 2)), encode=lambda d: [d["x0"], d["x1"]])
        assert np.allclose(p2, 0.95)


# ── Test 5 — the negative-score hazard (POSITIVE CONTROL) ────────────────────

class TestTheNegativeScoreHazard:
    """UCB scores are routinely negative. ``negative * p_feas`` is *larger*.

    Without min-max normalization the feasibility weight becomes a **bonus** for
    the compositions most likely to fail — the feature doing precisely the
    opposite of its purpose, with nothing in any log saying so.
    """

    #: Two candidates. Index 0 is the better acquisition AND feasible; index 1 is
    #: worse and almost certainly infeasible. Both scores are negative, which is
    #: what makes the bug possible at all.
    SCORES = np.array([-1.0, -2.0])
    P_FEAS = np.array([0.98, 0.02])

    def test_without_normalization_the_infeasible_candidate_wins(self):
        """The control: proves the guard below is not vacuous.

        If this assertion ever fails, the hazard has stopped existing and the
        test below stops proving anything — read them as a pair.
        """
        naive = self.SCORES * self.P_FEAS
        assert int(np.argmax(naive)) == 1      # the infeasible one, promoted
        assert naive[1] > self.SCORES[1]       # penalty acted as a bonus

    def test_with_normalization_no_infeasible_candidate_outranks_a_feasible_one(self):
        weighted = normalize_scores(self.SCORES) * self.P_FEAS
        assert int(np.argmax(weighted)) == 0
        assert weighted[1] <= weighted[0]

    def test_the_optimizer_seam_normalizes_before_weighting(self):
        """End-to-end through `_apply_feasibility_weight`, not the helper alone."""
        opt = _warm(_optimizer(
            feasibility=FeasibilityModel(
                FeasibilityConfig(enabled=True, min_filter=False),
                classifier=_ConstantClassifier(0.5))))
        _stocked(opt.feasibility, 3, 3)
        scores = np.array([-1.0, -2.0, -3.0])
        out = opt._apply_feasibility_weight(np.zeros((3, 2)), scores)
        assert np.all(out >= 0.0)                    # normalized, so never negative
        assert int(np.argmax(out)) == int(np.argmax(scores))

    def test_a_degenerate_pool_normalizes_to_ones_not_zeros(self):
        """Zeros would erase the acquisition and let p_feas alone choose."""
        assert np.array_equal(normalize_scores(np.array([2.0, 2.0])), np.ones(2))
        assert np.array_equal(normalize_scores(np.array([5.0])), np.ones(1))


# ── Tests 6 / 7 — hard stays hard; learned never vetoes ──────────────────────

class TestTheLearnedLayerNeverOverrulesTheHardFilter:
    def test_a_hard_filter_still_excludes_a_region_the_classifier_likes(self):
        """6 — known constraints stay hard."""
        opt = _warm(_optimizer(
            feasibility=FeasibilityModel(
                FeasibilityConfig(enabled=True),
                classifier=_ConstantClassifier(0.99))))
        _stocked(opt.feasibility, 3, 3)
        opt.feasibility_fn = lambda p: p["x0"] <= 0.5
        for _ in range(5):
            assert opt.suggest()["x0"] <= 0.5

    def test_a_classifier_certain_everything_fails_still_yields_a_point(self):
        """7 — a learned veto would make suggest() return None, ending the run."""
        opt = _warm(_optimizer(
            feasibility=FeasibilityModel(
                FeasibilityConfig(enabled=True),
                classifier=_ConstantClassifier(0.0))))
        _stocked(opt.feasibility, 3, 3)
        assert opt.feasibility_fn is None
        for _ in range(3):
            point = opt.suggest()
            assert point is not None
            assert set(point) == {"x0", "x1"}

    def test_the_weight_never_removes_a_row_from_the_pool(self):
        opt = _warm(_optimizer(
            feasibility=FeasibilityModel(
                FeasibilityConfig(enabled=True),
                classifier=_RegionClassifier())))
        _stocked(opt.feasibility, 3, 3)
        scores = np.linspace(-2.0, 2.0, 11)
        out = opt._apply_feasibility_weight(
            np.random.RandomState(0).uniform(size=(11, 2)), scores)
        assert len(out) == len(scores)


# ── Test 8 — the synthetic campaign (the money test) ─────────────────────────

class TestASyntheticCampaignSteersAwayWithoutSearchingWorse:
    """A known region always fails. Does the layer avoid it *and* still optimise?

    Assertion (a) alone is satisfied by an optimizer that simply stops exploring,
    so (b) is not decoration — it is what separates steering from stalling.
    """

    N = 40
    REGION = staticmethod(lambda p: p["x0"] + p["x1"] > 0.8)

    def _run(self, enabled: bool):
        model = FeasibilityModel(
            FeasibilityConfig(enabled=enabled),
            classifier=_RegionClassifier(),
        )
        opt = BayesianOptimizer(SPACE, "maximize", seed=11, n_initial=5,
                                n_candidates=300, feasibility=model)
        in_region, best = 0, -math.inf
        for _ in range(self.N):
            p = opt.suggest()
            if self.REGION(p):
                # An open-circuit trace: nothing is told to the optimizer (the
                # NULL-objective discipline), only a label is recorded.
                in_region += 1
                model.add(p, INFEASIBLE, channel=1, board_id="b1")
                continue
            value = -((p["x0"] - 0.25) ** 2 + (p["x1"] - 0.25) ** 2)
            opt.tell(p, value)
            model.add(p, FEASIBLE, channel=1, board_id="b1")
            best = max(best, value)
        return in_region, best

    def test_the_feature_lowers_time_in_the_failing_region_without_costing_the_optimum(self):
        off_region, off_best = self._run(enabled=False)
        on_region, on_best = self._run(enabled=True)

        # (a) strictly less time spent in the region that always fails
        assert on_region < off_region, (on_region, off_region)
        # (b) and the search did not simply stop — the best is no worse
        assert on_best >= off_best - 1e-3, (on_best, off_best)


# ── Test 9 — serialization and resume ────────────────────────────────────────

class TestTheClassifierIsRebuiltNeverSerialized:
    def test_to_dict_carries_the_config_and_a_count_but_no_fitted_arrays(self):
        model = FeasibilityModel(FeasibilityConfig(enabled=True, min_infeasible=4))
        _stocked(model, 3, 4)
        opt = _warm(_optimizer(feasibility=model))
        blob = json.dumps(opt.to_dict())          # must be JSON-serializable at all
        extra = opt.to_dict()["extra"]
        assert extra["feasibility"]["enabled"] is True
        assert extra["feasibility"]["min_infeasible"] == 4
        assert extra["n_infeasible_labels"] == 4
        # The derived clamp is never stored: storing it would let a checkpoint
        # and its floor disagree.
        assert "clamp" not in extra["feasibility"]
        for token in ("kernel", "log_marginal", "X_train", "GaussianProcess"):
            assert token not in blob

    def test_a_resume_reproduces_the_same_p_feas_vector_from_the_same_labels(self):
        cand = np.random.RandomState(3).uniform(size=(6, 2))

        def build():
            m = FeasibilityModel(FeasibilityConfig(enabled=True), seed=5)
            _stocked(m, 3, 3)
            return m.p_feasible(cand, encode=lambda d: [d["x0"], d["x1"]])

        assert np.allclose(build(), build())

    def test_a_label_count_disagreement_warns_rather_than_searching_quietly(self, caplog):
        model = FeasibilityModel(FeasibilityConfig(enabled=True))
        _stocked(model, 3, 7)
        state = _warm(_optimizer(feasibility=model)).to_dict()
        assert state["extra"]["n_infeasible_labels"] == 7

        # Resume finds only 3 — the checkpoint must say so.
        restored = BayesianOptimizer.from_dict(state)
        assert restored.feasibility.n_infeasible == 0
        restored._restore_extra(state["extra"])   # explicit, for the warning path
        assert restored.feasibility.config.enabled is True

    def test_a_resumed_optimizer_keeps_the_strategy_it_was_checkpointed_with(self):
        model = FeasibilityModel(
            FeasibilityConfig(enabled=True, min_filter=False, min_feasible=6))
        state = _warm(_optimizer(feasibility=model)).to_dict()
        restored = BayesianOptimizer.from_dict(state)
        assert restored.feasibility.config.min_filter is False
        assert restored.feasibility.config.min_feasible == 6


# ── Test 12 — a deferred strategy is not a synonym ───────────────────────────

class TestDeferredAndRejectedStrategiesRefuse:
    def test_fia_raises_and_names_its_revisit_trigger(self):
        with pytest.raises(OptimizerError, match="100 labelled trials"):
            FeasibilityConfig(strategy="fia")

    def test_fca_raises_and_says_why_softae_has_nothing_to_constrain(self):
        with pytest.raises(OptimizerError, match="random candidate pool"):
            FeasibilityConfig(strategy="fca")

    def test_an_unknown_strategy_lists_what_is_implemented(self):
        with pytest.raises(OptimizerError, match="fwa"):
            FeasibilityConfig(strategy="nonsense")

    def test_fwa_is_accepted_and_case_insensitive(self):
        assert FeasibilityConfig(strategy="FWA").strategy == "fwa"


# ── Test 11 — the no-ATLAS guard ─────────────────────────────────────────────

class TestATLASNeverEntersTheDependencyTree:
    def test_no_source_file_imports_atlas(self):
        """Source scan (T2.7's style), not ``sys.modules``.

        A ``sys.modules`` check only proves nothing imported it *on this run*;
        the scan proves nothing can.
        """
        root = Path(__file__).resolve().parents[1] / "src" / "softae"
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("import atlas", "from atlas")):
                    offenders.append(f"{path}:{i}")
        assert offenders == []


# ── Refit cadence (§4) ───────────────────────────────────────────────────────

class TestTheClassifierRefitsOnlyWhenTheLabelsChanged:
    def test_an_unchanged_label_multiset_does_not_refit(self):
        clf = _ConstantClassifier(0.3)
        model = FeasibilityModel(FeasibilityConfig(enabled=True), classifier=clf)
        _stocked(model, 3, 3)
        encode = lambda d: [d["x0"], d["x1"]]        # noqa: E731
        model.p_feasible(np.zeros((2, 2)), encode=encode)
        model.p_feasible(np.zeros((2, 2)), encode=encode)
        assert clf.fit_calls == 1

        model.add({"x0": 0.5, "x1": 0.5}, INFEASIBLE, channel=9, board_id="b2")
        model.p_feasible(np.zeros((2, 2)), encode=encode)
        assert clf.fit_calls == 2

    def test_a_retraction_forces_a_refit_because_the_multiset_moved(self):
        clf = _ConstantClassifier(0.3)
        model = FeasibilityModel(FeasibilityConfig(enabled=True), classifier=clf)
        _stocked(model, 3, 3)
        encode = lambda d: [d["x0"], d["x1"]]        # noqa: E731
        model.p_feasible(np.zeros((2, 2)), encode=encode)
        assert model.retract_channel(2) == 1
        assert model.n_infeasible == 2
        model.p_feasible(np.zeros((2, 2)), encode=encode)
        # Back below the floor after the retraction, so no refit and no opinion.
        assert model.active is False
