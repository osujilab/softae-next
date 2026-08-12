"""Bayesian optimizer over a continuous box, backed by the shared surrogate stack.

Unlike :class:`~softae.optimizers.pooled_bayesian.PooledBayesianOptimizer`, which
selects from a finite candidate pool, this optimizer proposes points over the
**continuous parameter box** by scoring a large batch of random candidates — the
formulation the live hardware autonomous loop drives.

It fits through a :class:`~softae.optimizers.surrogates.SurrogateBackend`
(default :class:`~softae.optimizers.surrogates.SklearnGPBackend`: inputs
standardized per axis, an ARD Matérn kernel, and a learned homoscedastic noise
level) and scores candidates with an
:class:`~softae.optimizers.acquisitions.AcquisitionStrategy` (``"ucb"`` or
``"ei"``).  Categorical parameters are one-hot encoded via the shared
:class:`~softae.optimizers.encoding.OneHotEncoder`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import structlog

from softae.errors import OptimizerError
from softae.optimizers.acquisitions import make_acquisition
from softae.optimizers.base import BaseOptimizer
from softae.optimizers.batch import BatchStrategy, make_batch_strategy
from softae.optimizers.encoding import OneHotEncoder
from softae.optimizers.feasibility import (
    FeasibilityConfig,
    FeasibilityModel,
    normalize_scores,
)
from softae.optimizers.surrogates import make_backend

logger = structlog.get_logger(__name__)

#: A prior-mean / physics model: maps a parameter dict to an expected objective
#: value.  When supplied, the GP models the **residual** from this prior rather
#: than the raw objective (physically/prior-informed BO).
PriorMean = Callable[[dict[str, Any]], float]


class BayesianOptimizer(BaseOptimizer):
    """GP-based Bayesian optimization over a continuous box.

    Parameters
    ----------
    parameter_space
        Search space definition (see :class:`BaseOptimizer`).
    objective
        ``"maximize"`` or ``"minimize"``.
    seed
        RNG seed for reproducibility.
    n_initial
        Number of random warm-up evaluations before the surrogate kicks in.
    acquisition
        ``"ucb"`` or ``"ei"``.
    kappa
        UCB exploration weight (only used when *acquisition* = ``"ucb"``).
    n_candidates
        Number of random candidate points evaluated per acquisition step.
    prior_mean
        Optional prior-mean / physics model ``m(params) -> value``.  When given,
        the GP is fit to the **residual** ``y - m(x)`` and the posterior mean is
        reconstructed as ``m(x) + gp(x)`` before scoring — so a physical model
        supplies the trend and the GP only learns the correction.  ``None``
        (default) is an ordinary zero-mean-residual GP (unchanged behavior).
    decision_rtol
        Relative tolerance for the acquisition argmax (ATLAS-style). ``0.0``
        (default) keeps the strict argmax bit-for-bit. When > 0, the proposal
        is drawn **uniformly at random (from the optimizer's own RNG, so it
        stays resume-safe)** among all candidates whose acquisition value is
        within ``rtol`` of the best — guarding flat/degenerate acquisition
        surfaces where strict argmax just returns the first index.
    exclusion_radius
        Normalized-space exclusion radius around already-measured points
        (ATLAS-style). ``None`` (default) is off. When set, candidates within
        this Euclidean distance — computed in the encoded space with float/int
        axes rescaled to [0, 1] by their bounds; one-hot axes as-is — of any
        measured point are excluded from the acquisition argmax, preventing
        near-duplicate re-requests. If **every** candidate is excluded, the
        unexcluded argmax is used and a warning logged rather than failing.
    feasibility
        Learned-feasibility configuration or a prebuilt
        :class:`~softae.optimizers.feasibility.FeasibilityModel` (T3.1). ``None``
        (default) constructs a model whose config is ``enabled=False``, which is
        arithmetically invisible: no classifier is fit, no label is read, and
        :meth:`_apply_feasibility_weight` returns the score vector it was given.
        When enabled, a classification surrogate fit on pass/fail labels
        *softens* the acquisition — it **never vetoes**, never removes a
        candidate from the pool and never returns ``None`` from :meth:`suggest`.

    Notes
    -----
    A **seed / warm-start** is not a constructor argument: call :meth:`tell`
    with prior ``(params, value)`` observations before the first :meth:`suggest`
    to pre-populate the GP (they count toward ``n_initial``).
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
        *,
        n_initial: int = 5,
        acquisition: str = "ucb",
        kappa: float = 2.0,
        n_candidates: int = 5000,
        prior_mean: PriorMean | None = None,
        batch_strategy: str | BatchStrategy = "constant_liar",
        decision_rtol: float = 0.0,
        exclusion_radius: float | None = None,
        feasibility: "FeasibilityConfig | FeasibilityModel | None" = None,
    ) -> None:
        super().__init__(parameter_space, objective, seed)
        if acquisition not in ("ucb", "ei"):
            raise OptimizerError(f"acquisition must be 'ucb' or 'ei', got '{acquisition}'")
        if n_initial < 1:
            raise OptimizerError("n_initial must be >= 1")
        if decision_rtol < 0:
            raise OptimizerError("decision_rtol must be >= 0")
        if exclusion_radius is not None and exclusion_radius <= 0:
            raise OptimizerError("exclusion_radius must be > 0 (or None to disable)")

        self._n_initial = n_initial
        self._acquisition = acquisition
        self._kappa = kappa
        self._n_candidates = n_candidates
        self._prior_mean = prior_mean
        self._decision_rtol = float(decision_rtol)
        self._exclusion_radius = (
            float(exclusion_radius) if exclusion_radius is not None else None
        )
        self._rng = np.random.RandomState(seed)

        #: The learned-feasibility layer (T3.1). Always present so callers need
        #: no ``is None`` dance, but inert unless its config says otherwise.
        if isinstance(feasibility, FeasibilityModel):
            self.feasibility = feasibility
        else:
            self.feasibility = FeasibilityModel(
                feasibility or FeasibilityConfig(), seed=seed)

        self._encoder = OneHotEncoder(self._parameter_space)
        self.backend = make_backend("sklearn", seed=seed)
        self._strategy = make_acquisition(acquisition)
        self._batch_strategy = make_batch_strategy(batch_strategy)

        #: This is a continuous, homoscedastic-noise path (no per-candidate
        #: aleatoric variance), so the duck-typed acquisition sees ``None`` here.
        self.candidate_variance: np.ndarray | None = None

        #: Visibility for the last proposal (T3.1 §9). ``None`` means the learned
        #: layer had no opinion on it — feature off, warm-up, or below the floor.
        self.last_p_feas: float | None = None
        self.last_steered: bool | None = None
        self._last_p_feas: np.ndarray | None = None

    # ── Serialization (P3.1) ────────────────────────────────────────────

    @classmethod
    def _construct_from(cls, state: dict[str, Any]) -> "BayesianOptimizer":
        """Rebuild from a checkpoint.

        ``prior_mean`` is **not** serializable — it is an arbitrary Python
        callable — so a resumed optimizer starts without it and the caller must
        re-attach the same model. Silently resuming with no prior would change
        the surrogate the campaign is fitting, which is why
        :meth:`_state_extra` records whether one was in use.
        """
        extra = state.get("extra") or {}
        exclusion_radius = extra.get("exclusion_radius")
        return cls(
            state["parameter_space"],
            state.get("objective", "maximize"),
            state.get("seed"),
            n_initial=int(extra.get("n_initial", 5)),
            acquisition=extra.get("acquisition", "ucb"),
            kappa=float(extra.get("kappa", 2.0)),
            n_candidates=int(extra.get("n_candidates", 5000)),
            batch_strategy=extra.get("batch_strategy", "constant_liar"),
            decision_rtol=float(extra.get("decision_rtol", 0.0)),
            exclusion_radius=(
                float(exclusion_radius) if exclusion_radius is not None else None
            ),
            # Config rides in the checkpoint (§7) so a resumed run cannot
            # silently change strategy; the *fitted* classifier does not, because
            # it is a pure function of (labels, config) and is refit from the
            # labels the resuming caller supplies.
            feasibility=FeasibilityConfig.from_dict(extra.get("feasibility")),
        )

    def _rng_state(self):
        # np.random.RandomState.get_state() carries a uint32 key array; convert
        # for JSON and rebuild on restore.
        kind, keys, pos, has_gauss, cached = self._rng.get_state()
        return [kind, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)]

    def _restore_rng(self, state) -> None:
        import numpy as _np

        kind, keys, pos, has_gauss, cached = state
        self._rng.set_state(
            (kind, _np.array(keys, dtype=_np.uint32), int(pos), int(has_gauss),
             float(cached))
        )

    def _state_extra(self) -> dict[str, Any]:
        return {
            "n_initial": self._n_initial,
            "acquisition": self._acquisition,
            "kappa": self._kappa,
            "n_candidates": self._n_candidates,
            # The registry key (``.name``), not the class name — those differ
            # ("constant_liar" vs "ConstantLiarStrategy") and make_batch_strategy
            # only accepts the former.
            "batch_strategy": getattr(
                self._batch_strategy, "name", "constant_liar"),
            "decision_rtol": self._decision_rtol,
            "exclusion_radius": self._exclusion_radius,
            # Recorded so a resume can detect that a prior model must be
            # re-attached rather than quietly fitting a different surrogate.
            "had_prior_mean": self._prior_mean is not None,
            # T3.1 §7: the CONFIG, never the fitted classifier. Plus one integer
            # so a resume can *notice* a label-count disagreement — a checkpoint
            # that saw 7 infeasible labels and finds 3 must say so rather than
            # quietly searching differently.
            "feasibility": self.feasibility.config.as_dict(),
            "n_infeasible_labels": self.feasibility.n_infeasible,
        }

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        if extra.get("had_prior_mean") and self._prior_mean is None:
            logger.warning(
                "resumed_without_prior_mean",
                msg="checkpoint used a prior mean; re-attach it before resuming "
                    "or the surrogate will differ from the original run",
            )
        checkpointed = extra.get("n_infeasible_labels")
        if checkpointed is not None and int(checkpointed) != self.feasibility.n_infeasible:
            logger.warning(
                "resumed_feasibility_label_mismatch",
                checkpoint_n_infeasible=int(checkpointed),
                restored_n_infeasible=self.feasibility.n_infeasible,
                msg="the checkpoint's infeasible-label count differs from what "
                    "was restored; the learned layer will steer differently than "
                    "it did before the interruption",
            )

    # ── Sampling / prior ────────────────────────────────────────────────

    def _prior(self, params: dict[str, Any]) -> float:
        """Prior-mean value at *params* (0.0 when no prior model is set)."""
        return float(self._prior_mean(params)) if self._prior_mean is not None else 0.0

    def _random_point(self) -> dict[str, Any]:
        """Sample a random point from the parameter space."""
        params: dict[str, Any] = {}
        for name, spec in self._parameter_space.items():
            ptype = spec["type"]
            if ptype == "float":
                params[name] = self._rng.uniform(spec["low"], spec["high"])
            elif ptype == "int":
                params[name] = self._rng.randint(spec["low"], spec["high"] + 1)
            else:
                params[name] = self._rng.choice(self._encoder.cat_maps[name])
        return params

    def _feasible_random_point(self, max_tries: int = 200) -> dict[str, Any]:
        """A random point the twin admits, or the last try if none was found.

        Returning an infeasible point after exhausting the budget is deliberate:
        the campaign should proceed and let the overflow guard refuse the cast,
        rather than the optimizer hanging or returning ``None`` (which the loop
        reads as "exhausted" and would end the run).
        """
        point = self._random_point()
        if getattr(self, "feasibility_fn", None) is None:
            return point
        for attempt in range(max_tries):
            if self._is_feasible(point):
                return point
            point = self._random_point()
        logger.warning(
            "no_feasible_warmup_point", tries=max_tries,
            msg="proposing an infeasible point — the overflow guard will refuse "
                "it; the declared bounds are probably wrong",
        )
        return point

    def _is_feasible(self, params: dict[str, Any]) -> bool:
        """Whether the twin admits *params*. No hook set → everything is feasible.

        A raising feasibility function is treated as **feasible**: refusing every
        point on a bug would silently stall the campaign, which is far worse than
        proposing one the overflow guard will catch downstream anyway.
        """
        fn = getattr(self, "feasibility_fn", None)
        if fn is None:
            return True
        try:
            return bool(fn(params))
        except Exception:
            logger.warning("feasibility_fn_failed", exc_info=True)
            return True

    def _random_candidates(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Sample candidate points, returning both the dicts and their encoding.

        Keeping the dicts avoids a decode round-trip and lets the prior-mean
        model (which is defined in parameter space) be evaluated directly.

        **Infeasible candidates are removed before the acquisition argmax** (P7.1),
        so an infeasible point is never proposed. Restricting the maximiser's
        domain is exact here because the constraint is known, deterministic and
        cheap — the twin is arithmetic, not an experiment — so no constraint GP
        or probability-of-feasibility weighting is warranted. It also avoids the
        unsoundness of bounds-clipping: overflow is ``Σvᵢ ≤ capacity``, a
        diagonal half-space, and clipping each axis to ``capacity/n`` would
        discard feasible points including a legitimately optimal (high v₀, low
        v₁) corner.
        """
        points = [self._random_point() for _ in range(self._n_candidates)]
        feasible = [p for p in points if self._is_feasible(p)]

        if feasible:
            if len(feasible) < len(points):
                # Visible, not silently slow: a tiny feasible fraction starves
                # the acquisition, and the fix is the operator's bounds.
                logger.info("candidates_filtered", feasible=len(feasible),
                            sampled=len(points))
            points = feasible
        elif getattr(self, "feasibility_fn", None) is not None:
            logger.warning(
                "no_feasible_candidates", sampled=len(points),
                msg="proposing unfiltered — the declared space may be entirely "
                    "infeasible for the twin",
            )

        X = np.array([self._encoder.encode(p) for p in points], dtype=float)
        return points, X

    # ── BaseOptimizer interface ─────────────────────────────────────────

    def suggest(self) -> dict[str, Any] | None:
        # Warm-up: random exploration. Rejection-sampled against the twin, or
        # the first n_initial trials would be exactly the ones that overflow —
        # the warm-up is random precisely where the guard matters most.
        if len(self._history) < self._n_initial:
            return self._feasible_random_point()

        # Fit the surrogate on the residual from the prior (homoscedastic — no
        # per-point noise). With no prior model the residual is just ``y``.
        X = np.array([self._encoder.encode(p) for p, _ in self._history], dtype=float)
        y = np.array([v - self._prior(p) for p, v in self._history], dtype=float)
        self.backend.fit(X, y, alpha=None)

        # Predict the residual posterior, add the prior mean back, then score.
        cand_points, cand_X = self._random_candidates()
        mu_resid, sigma = self.backend.predict(cand_X)
        if self._prior_mean is not None:
            mu = mu_resid + np.array([self._prior(p) for p in cand_points])
        else:
            mu = mu_resid
        scores = np.asarray(
            self._strategy.score(self, cand_X, mu, sigma, self._history), dtype=float)

        # T3.1 — the learned layer sits BETWEEN scoring and selection, so
        # `_select_index` (and therefore decision_rtol / exclusion_radius, and
        # the RNG stream) is untouched and composes with this unchanged.
        weighted = self._apply_feasibility_weight(cand_X, scores)
        index = self._select_index(cand_X, weighted)

        if weighted is not scores:
            # "Did the learned model change the answer?" is the only question an
            # operator actually has, and it costs one extra argmax over an array
            # already in memory. RNG-free on purpose: calling `_select_index` a
            # second time would consume the checkpointed random stream.
            unweighted = int(np.argmax(scores))
            self.last_p_feas = float(self._last_p_feas[index])
            self.last_steered = bool(unweighted != index)
            logger.info(
                "feasibility_weighted_suggestion",
                p_feas=self.last_p_feas, steered=self.last_steered,
            )

        return cand_points[index]

    def _apply_feasibility_weight(
        self, cand_X: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """``fwa``: normalized acquisition x ``p_feas`` (spec §5–§6).

        Returns the **same object** it was given whenever the layer has no
        opinion — feature off, or below the minimum-data floor — so "unchanged"
        is identity rather than an equality the caller has to trust, and the
        default-off path is bit-identical to pre-T3.1 behaviour.

        Three invariants, each with a test:

        * **Normalize before weighting.** UCB scores are routinely negative and
          ``negative x p_feas`` is *larger*, so without
          :func:`~softae.optimizers.feasibility.normalize_scores` the penalty
          inverts into a bonus for the compositions most likely to fail.
        * **Reweight, never remove.** Every row survives, so the learned model
          can never empty the pool, never make ``suggest()`` return ``None``
          (which the loop reads as "exhausted" and ends the run on), and never
          starve the argmax.
        * **The hard filter stays hard.** ``_random_candidates`` has already
          removed twin-infeasible candidates; this only reorders what is left.
        """
        model = getattr(self, "feasibility", None)
        if model is None or not model.config.enabled:
            return scores

        # Candidates are mapped to the SAME unit cube the labels are encoded in.
        # Predicting on raw encoded coordinates while training on normalized ones
        # would compare two different geometries and quietly return noise.
        lows, spans = self._normalization()
        cand_norm = (np.asarray(cand_X, dtype=float) - lows) / np.where(
            spans == 0, 1.0, spans)

        p_feas = model.p_feasible(cand_norm, encode=self._encode_unit_cube)
        if np.all(p_feas >= 1.0):
            # The gate withheld: p_feas is identically 1.0 and the score vector
            # must come back elementwise unchanged, not merely order-equivalent.
            return scores

        self._last_p_feas = p_feas
        return normalize_scores(scores) * p_feas

    def _encode_unit_cube(self, params: Any) -> np.ndarray:
        """Encode to the same unit-cube coordinates the exclusion radius uses.

        Both surrogates then see the same geometry: the classifier's Matérn
        length scale is comparable across axes with different physical units,
        which it would not be on raw volumes in µL beside a mole ratio.
        """
        lows, spans = self._normalization()
        x = np.array(self._encoder.encode(params), dtype=float)
        return (x - lows) / np.where(spans == 0, 1.0, spans)

    # ── Proposal selection (ATLAS-inspired, both off by default) ────────

    def _normalization(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-encoded-axis ``(low, span)`` mapping the box onto the unit cube.

        Float/int axes rescale by their declared bounds; one-hot axes are
        already 0/1 and pass through (low 0, span 1). Order matches
        :meth:`OneHotEncoder.encode`.
        """
        lows: list[float] = []
        spans: list[float] = []
        for name, spec in self._parameter_space.items():
            if spec["type"] in ("float", "int"):
                low, high = float(spec["low"]), float(spec["high"])
                lows.append(low)
                spans.append(high - low)
            else:  # categorical → one axis per choice
                for _ in self._encoder.cat_maps[name]:
                    lows.append(0.0)
                    spans.append(1.0)
        return np.array(lows, dtype=float), np.array(spans, dtype=float)

    def _select_index(self, cand_X: np.ndarray, scores: np.ndarray) -> int:
        """Pick the proposal index from scored candidates.

        With defaults (``decision_rtol=0``, ``exclusion_radius=None``) this is
        exactly ``argmax(scores)`` and draws **nothing** from the RNG, so the
        legacy random stream — and therefore resume behavior — is unchanged.

        * ``exclusion_radius`` restricts the argmax to candidates farther than
          the radius (normalized space) from every measured point; if that
          excludes everything, it falls back to the unexcluded set with a
          warning rather than failing the campaign.
        * ``decision_rtol`` replaces the strict argmax with a uniform draw
          (from ``self._rng``, which is checkpointed) among all candidates
          whose score is within ``rtol`` of the best.
        """
        eligible = np.arange(len(scores))

        if self._exclusion_radius is not None and self._history:
            lows, spans = self._normalization()
            meas = np.array(
                [self._encoder.encode(p) for p, _ in self._history], dtype=float
            )
            cand_n = (cand_X - lows) / spans
            meas_n = (meas - lows) / spans
            # Min distance from each candidate to any measured point.
            d = np.linalg.norm(cand_n[:, None, :] - meas_n[None, :, :], axis=2)
            keep = d.min(axis=1) >= self._exclusion_radius
            if keep.any():
                eligible = eligible[keep]
            else:
                logger.warning(
                    "all_candidates_excluded",
                    exclusion_radius=self._exclusion_radius,
                    n_candidates=len(scores),
                    msg="every candidate is within the exclusion radius of a "
                        "measured point — falling back to the unexcluded argmax",
                )

        sub = scores[eligible]
        if self._decision_rtol > 0.0:
            best = float(np.max(sub))
            ties = eligible[sub >= best - self._decision_rtol * abs(best)]
            return int(self._rng.choice(ties))
        return int(eligible[int(np.argmax(sub))])

    def _posterior_mean(self, params: dict[str, Any]) -> float:
        """GP posterior mean at *params* (for the Kriging-believer fantasy).

        Falls back to the mean of observed objectives when the surrogate is not
        yet fit (e.g. during warm-up), so a batch can still be proposed.
        """
        try:
            x = np.array([self._encoder.encode(params)], dtype=float)
            mu, _ = self.backend.predict(x)
            return float(mu[0])
        except Exception:
            ys = [v for _, v in self._history]
            return float(sum(ys) / len(ys)) if ys else 0.0

    def suggest_batch(self, q: int) -> list[dict[str, Any]]:
        """Propose *q* diverse points via the configured :class:`BatchStrategy`.

        Diversification (constant-liar, Kriging-believer, or a future Monte-Carlo
        acquisition) is delegated to ``batch_strategy`` — the optimizer only
        exposes ``suggest``/``_history``/``_posterior_mean`` for the strategy to
        drive.  All temporary fantasies are removed before returning, so the real
        objectives (told later, one per parallel evaluation) replace them.
        """
        return self._batch_strategy.propose(self, q)

    def tell(self, params: dict[str, Any], result: float) -> None:
        self._history.append((params, result))

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._find_best()
