"""Failure-informed feasibility: a second, *classification* surrogate (T3.1).

softae records failures well and tells the optimizer nothing about them. A trial
cast but not measured leaves ``doe_parameters.objective_value`` NULL and is
deliberately withheld from ``optimizer.tell`` — correct discipline, but its side
effect is that a well, an anneal and hours of rig time buy **zero** search signal.

This module is the other half: a classifier fit on pass/fail labels that *softens*
the acquisition, while the regression GP continues to see only real measurements.
Nothing about the NULL discipline changes, and nothing here can veto a candidate.

Three properties are load-bearing and each has a test that fails without it:

* **The learned layer never vetoes.** It reweights a score vector; it never
  removes a row from the candidate pool. A learned model that could empty the
  pool would make ``suggest()`` return ``None``, which the loop reads as
  "exhausted" and ends the run on.
* **Acquisition scores are min-max normalized to [0, 1] before weighting.** UCB
  scores are routinely negative, and multiplying a negative score by
  ``p_feas < 1`` *raises* it — the penalty inverts into a bonus for exactly the
  compositions most likely to fail. See :func:`normalize_scores`.
* **The minimum-data floors are 3/3 and tunable upward only.** Below 3 the
  derived clamp exceeds 0.63 and the weight starts penalising points the
  classifier considers *more likely feasible than not*, so the refusal is not
  paternalism about sample size — it is the point at which the mechanism inverts
  its own meaning.

Spec: ``docs/SubAgent docs/failure_informed_feasibility_spec.md`` §4–§6.
ATLAS supplied the concepts and three formulas; **it never enters the dependency
tree** — there is no ``atlas`` import here or anywhere in ``src/softae``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import structlog

from softae.errors import OptimizerError

logger = structlog.get_logger(__name__)

__all__ = [
    "ABSOLUTE_LABEL_FLOOR",
    "FEASIBLE",
    "INFEASIBLE",
    "FeasibilityConfig",
    "FeasibilityLabel",
    "FeasibilityModel",
    "FeasibilityClassifier",
    "GPCFeasibilityClassifier",
    "derived_clamp",
    "normalize_scores",
]

#: Label values. 0 = the composition mixed, cast, dried and measured; 1 = it did
#: not make a sample. Deliberately *not* a bool: the classifier's positive class
#: is "infeasible", and ``p_feas`` below is the probability of class 0.
FEASIBLE = 0
INFEASIBLE = 1

#: The hard floor on both class minimums (user decision (v)). Operators may raise
#: a floor to demand more evidence in a known-hostile region; nothing may lower
#: one. See :func:`derived_clamp` for why 3 is where the mechanism stops meaning
#: what it says.
ABSOLUTE_LABEL_FLOOR = 3

#: The one-sided confidence level the derived clamp is computed at.
_CLAMP_CONFIDENCE = 0.05

#: Strategies this module implements. ``fia`` is DEFERRED behind a named trigger
#: (≥ 100 labelled trials under one parameter space) and ``fca`` is REJECTED —
#: softae scores a random candidate pool and takes an argmax, so there is no
#: constrained acquisition optimizer to hand a nonlinear constraint to.
_IMPLEMENTED_STRATEGIES = ("fwa",)
_DEFERRED_STRATEGIES = {
    "fia": (
        "'fia' is DEFERRED, not available: its blend weight r is a single global "
        "scalar estimated from the same scarce labels, so a scarce-label error "
        "replaces the objective with p_feas everywhere at once rather than "
        "perturbing a ranking. Revisit trigger: >= 100 labelled trials under one "
        "parameter space (r is a binomial proportion; its 95% half-width is "
        "<= 0.1 only once n >= 96). Use 'fwa'."
    ),
    "fca": (
        "'fca' is REJECTED for softae: it constrains a gradient/SLSQP acquisition "
        "optimizer, and suggest() scores a random candidate pool and takes an "
        "argmax — there is nothing to hand a constraint to. Degrading it to "
        "'drop candidates below a p_feas cutoff' would be a hard veto by a "
        "learned model, which this design forbids. Use 'fwa'."
    ),
}


def derived_clamp(min_infeasible: int) -> float:
    """The min-filter clamp implied by a floor of *min_infeasible* labels.

    ``0.05 ** (1/k)`` is the exact one-sided 95 % lower bound on a region's
    failure probability after ``k`` consecutive failures, so using it as the
    clamp means the down-weighting **never claims more confidence than the floor
    supports**. The dependency runs floor → clamp, not the other way round: ATLAS
    hardcodes 0.5, which this reproduces as a near-special-case at ``k = 5``.

    ======================  ======  ==========================================
    ``min_infeasible`` (k)  clamp   down-weighting begins at
    ======================  ======  ==========================================
    3 (default)             0.368   ``p_feas < 0.368``
    4                       0.473   ``p_feas < 0.473``
    5                       0.549   ``p_feas < 0.549`` (ATLAS's constant 0.5)
    10                      0.741   ``p_feas < 0.741``
    ======================  ======  ==========================================

    Read it as: *the weaker the evidence base, the more confident the classifier
    must be before its opinion is allowed to move anything.* The confidence is
    not lowered; the threshold for acting on it is raised.
    """
    k = int(min_infeasible)
    if k < 1:
        raise OptimizerError("derived_clamp requires min_infeasible >= 1")
    return float(_CLAMP_CONFIDENCE ** (1.0 / k))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max the candidate pool's acquisition scores onto [0, 1].

    **This is load-bearing, not cosmetic.** UCB scores are routinely negative,
    and ``negative * p_feas`` where ``p_feas < 1`` is *larger* than the original —
    so without this the feasibility weight becomes a bonus for the compositions
    most likely to fail, and the feature does the exact opposite of its purpose.
    Pinned by a positive control that must fail when this is removed.

    Normalizing across the pool whose scores are already computed is also cheaper
    than ATLAS's approach, which draws 3,000 extra samples to estimate the range
    only because its acquisition optimizer is gradient-based and has no pool.

    A degenerate pool (every score equal, including a single candidate) maps to
    all-ones rather than all-zeros: zeros would erase the acquisition entirely and
    let ``p_feas`` alone choose, which is ``fia``'s failure mode, not ``fwa``'s.
    """
    arr = np.asarray(scores, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.ones_like(arr)
    lo, hi = float(finite.min()), float(finite.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(arr)
    out = (arr - lo) / (hi - lo)
    # A non-finite score cannot win an argmax it never had a claim on.
    return np.where(np.isfinite(arr), out, 0.0)


@dataclass(frozen=True)
class FeasibilityConfig:
    """Resolved knobs for the learned feasibility layer. Default: entirely off.

    Validated at construction so a bad floor is refused where it is *written*
    rather than at the first ``suggest()`` several hours into a campaign.
    """

    enabled: bool = False
    strategy: str = "fwa"
    min_filter: bool = True
    min_infeasible: int = ABSOLUTE_LABEL_FLOOR
    min_feasible: int = ABSOLUTE_LABEL_FLOOR

    def __post_init__(self) -> None:
        strategy = str(self.strategy).lower()
        if strategy in _DEFERRED_STRATEGIES:
            raise OptimizerError(_DEFERRED_STRATEGIES[strategy])
        if strategy not in _IMPLEMENTED_STRATEGIES:
            raise OptimizerError(
                f"unknown feasibility_strategy '{self.strategy}'; "
                f"implemented: {list(_IMPLEMENTED_STRATEGIES)}"
            )
        object.__setattr__(self, "strategy", strategy)

        for name in ("min_infeasible", "min_feasible"):
            value = int(getattr(self, name))
            if value < ABSOLUTE_LABEL_FLOOR:
                raise OptimizerError(
                    f"{name} must be >= {ABSOLUTE_LABEL_FLOOR}, got {value}. The "
                    f"floors are tunable UPWARD only: below {ABSOLUTE_LABEL_FLOOR} "
                    f"the derived clamp exceeds 0.63, so the weight would begin "
                    f"penalising points the classifier considers more likely "
                    f"feasible than not — the mechanism would invert its own "
                    f"meaning. Raise a floor to demand more evidence; never lower one."
                )
            object.__setattr__(self, name, value)

    @property
    def clamp(self) -> float:
        """The min-filter clamp, derived from :attr:`min_infeasible` (never set)."""
        return derived_clamp(self.min_infeasible)

    def as_dict(self) -> dict[str, Any]:
        """Checkpoint form. The *clamp* is omitted because it is derived."""
        return {
            "enabled": bool(self.enabled),
            "strategy": self.strategy,
            "min_filter": bool(self.min_filter),
            "min_infeasible": int(self.min_infeasible),
            "min_feasible": int(self.min_feasible),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FeasibilityConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            strategy=str(data.get("strategy", "fwa")),
            min_filter=bool(data.get("min_filter", True)),
            min_infeasible=int(data.get("min_infeasible", ABSOLUTE_LABEL_FLOOR)),
            min_feasible=int(data.get("min_feasible", ABSOLUTE_LABEL_FLOOR)),
        )

    def is_default(self) -> bool:
        """Whether this is exactly the shipped default (contributes no hash key)."""
        return self == FeasibilityConfig()


@dataclass(frozen=True)
class FeasibilityLabel:
    """One label, with the provenance a §3.3 retraction needs.

    ``channel`` and ``board_id`` are carried not for the classifier — which sees
    only the composition — but so a label issued from a channel later found to
    reject across boards can be **dropped**, including retroactively. A label
    with no provenance could only ever be forgotten wholesale.
    """

    params: dict[str, Any]
    label: int
    channel: int | None = None
    board_id: Any = None

    def __post_init__(self) -> None:
        if self.label not in (FEASIBLE, INFEASIBLE):
            raise OptimizerError(
                f"label must be {FEASIBLE} (feasible) or {INFEASIBLE} "
                f"(infeasible), got {self.label!r}"
            )


@runtime_checkable
class FeasibilityClassifier(Protocol):
    """The swappable classifier seam, mirroring ``SurrogateBackend``.

    Kept a Protocol so the GPC is replaceable without touching the optimizer —
    the same posture ``optimizers/surrogates.py`` takes for the regression GP.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on encoded points *X* and labels *y* (0 feasible / 1 infeasible)."""

    def predict_proba_feasible(self, X: np.ndarray) -> np.ndarray:
        """Probability of class 0 (feasible) for each row of *X*, in [0, 1]."""


class GPCFeasibilityClassifier:
    """``sklearn`` GP classifier, Matérn kernel, Laplace-approximated Bernoulli.

    The direct native analogue of ATLAS's ``ClassificationGPMatern``, chosen over
    a tree ensemble because a *weighting* term needs a smooth calibrated
    ``predict_proba`` and a forest cannot give one at n ≈ 20. scikit-learn is
    already declared and is the default regression surrogate backend.

    **Class imbalance is deliberately NOT compensated** — a divergence from
    ATLAS, which tiles the minority class. Duplicating points sharpens a Laplace
    GPC's posterior on evidence that was never independent; ATLAS survives it
    only because its variational ELBO is early-stopped on fold-averaged epochs.
    The two-sided label floor is what the tiling was compensating for.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed
        self._model: Any = None
        #: Set when every training label is one class. A GPC cannot be fit on a
        #: single class, and the floors make it unreachable in production — but a
        #: direct caller must get a defined answer rather than an exception.
        self._degenerate: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        if classes.size < 2:
            # p_feas is then the observed constant, which is the honest answer.
            self._degenerate = 1.0 if int(classes[0]) == FEASIBLE else 0.0
            self._model = None
            return

        self._degenerate = None
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self._model = GaussianProcessClassifier(
            kernel=kernel, random_state=self._seed, n_restarts_optimizer=0
        )
        self._model.fit(np.asarray(X, dtype=float), y)

    def predict_proba_feasible(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._degenerate is not None:
            return np.full(len(X), float(self._degenerate))
        if self._model is None:
            return np.ones(len(X), dtype=float)
        proba = self._model.predict_proba(X)
        # Column order follows `classes_`; never assume index 0 is the feasible
        # class — with a reordered label set that silently inverts the weight.
        classes = list(self._model.classes_)
        col = classes.index(FEASIBLE) if FEASIBLE in classes else 0
        return np.clip(np.asarray(proba, dtype=float)[:, col], 0.0, 1.0)


class FeasibilityModel:
    """Labels in, ``p_feas`` out — with the minimum-data gate in between.

    The model is **rebuilt, never serialized** (spec §7): a fitted classifier is a
    pure function of (labels, config), exactly as the regression GP is a pure
    function of ``history``, and ``to_dict`` already refuses to persist fitted
    sklearn internals for that reason. Only the *config* and one integer,
    ``n_infeasible``, ride in the checkpoint — the integer purely so a resume can
    **notice** disagreement and warn, the same honesty as ``had_prior_mean``.
    """

    def __init__(
        self,
        config: FeasibilityConfig | None = None,
        *,
        seed: int | None = None,
        classifier: FeasibilityClassifier | None = None,
        emit: Callable[..., None] | None = None,
    ) -> None:
        self.config = config or FeasibilityConfig()
        self._seed = seed
        self._classifier = classifier or GPCFeasibilityClassifier(seed=seed)
        self._emit = emit
        self._labels: list[FeasibilityLabel] = []
        self._fitted_signature: str | None = None
        self._withheld_reason: str | None = None

    # ── Labels ──────────────────────────────────────────────────────────

    @property
    def labels(self) -> list[FeasibilityLabel]:
        return list(self._labels)

    @property
    def n_feasible(self) -> int:
        return sum(1 for lb in self._labels if lb.label == FEASIBLE)

    @property
    def n_infeasible(self) -> int:
        return sum(1 for lb in self._labels if lb.label == INFEASIBLE)

    def add(
        self,
        params: Mapping[str, Any],
        label: int,
        *,
        channel: int | None = None,
        board_id: Any = None,
    ) -> FeasibilityLabel:
        """Record one label. Idempotence is the caller's — see the §3.1 engine."""
        entry = FeasibilityLabel(dict(params), int(label), channel, board_id)
        self._labels.append(entry)
        return entry

    def retract_channel(self, channel: int) -> int:
        """Drop every label issued from *channel*; returns how many (§3.3).

        Retroactive by design: when a reject pattern tracks a channel rather than
        the chemistry, a label already issued from it was never evidence about any
        composition, and leaving it in would teach the classifier chemistry from a
        connector.
        """
        want = int(channel)
        before = len(self._labels)
        self._labels = [
            lb for lb in self._labels
            if lb.channel is None or int(lb.channel) != want
        ]
        dropped = before - len(self._labels)
        if dropped:
            # Force a refit: the label multiset changed underneath the fit.
            self._fitted_signature = None
        return dropped

    def label_signature(self) -> str:
        """Count + hash of the label multiset — the refit trigger (spec §4)."""
        blob = "|".join(
            f"{lb.label}:{sorted(lb.params.items(), key=lambda kv: kv[0])}"
            for lb in self._labels
        )
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return f"{len(self._labels)}:{digest}"

    # ── The minimum-data gate ───────────────────────────────────────────

    def floors_met(self) -> bool:
        return (
            self.n_feasible >= self.config.min_feasible
            and self.n_infeasible >= self.config.min_infeasible
        )

    def _withheld_floor(self) -> str | None:
        """Which floor is unmet, or ``None`` when both are satisfied."""
        if self.n_feasible < self.config.min_feasible:
            if self.n_infeasible < self.config.min_infeasible:
                return "both"
            return "min_feasible"
        if self.n_infeasible < self.config.min_infeasible:
            return "min_infeasible"
        return None

    @property
    def active(self) -> bool:
        """Whether the layer is both enabled and past its floors."""
        return bool(self.config.enabled) and self.floors_met()

    # ── p_feas ──────────────────────────────────────────────────────────

    def p_feasible(
        self,
        cand_X: np.ndarray,
        *,
        encode: Callable[[Mapping[str, Any]], Sequence[float]],
    ) -> np.ndarray:
        """``p_feas`` per candidate — **all ones** whenever the layer is inactive.

        All-ones is the explicit "no opinion" value: multiplying by it is the
        identity, so an inactive layer is not merely harmless but arithmetically
        invisible, and the caller can detect it without asking a second question.
        """
        n = len(np.asarray(cand_X, dtype=float))
        if not self.config.enabled:
            return np.ones(n, dtype=float)

        floor = self._withheld_floor()
        if floor is not None:
            # One event per *change* of reason: a per-suggest() line would bury
            # the log of a long campaign under an unchanging fact.
            if self._withheld_reason != floor:
                self._withheld_reason = floor
                self._event(
                    "learned_feasibility_withheld",
                    n_feasible=self.n_feasible,
                    n_infeasible=self.n_infeasible,
                    floor_unmet=floor,
                    min_feasible=self.config.min_feasible,
                    min_infeasible=self.config.min_infeasible,
                    msg="below the minimum-data floor — p_feas is 1.0 everywhere "
                        "and scores are untouched",
                )
            return np.ones(n, dtype=float)
        self._withheld_reason = None

        self._fit_if_stale(encode)
        p = np.asarray(
            self._classifier.predict_proba_feasible(np.asarray(cand_X, dtype=float)),
            dtype=float,
        )
        p = np.clip(np.nan_to_num(p, nan=1.0), 0.0, 1.0)

        if self.config.min_filter:
            # Cap the *reward* side: a confidently-safe point earns no bonus over
            # a merely-probably-safe one, so the weight acts purely as a penalty.
            p = np.minimum(p, self.config.clamp)
        return p

    def _fit_if_stale(
        self, encode: Callable[[Mapping[str, Any]], Sequence[float]]
    ) -> None:
        """Refit at most once per ``suggest()``, only if the labels changed."""
        signature = self.label_signature()
        if signature == self._fitted_signature:
            return
        X = np.array([encode(lb.params) for lb in self._labels], dtype=float)
        y = np.array([lb.label for lb in self._labels], dtype=int)
        self._classifier.fit(X, y)
        self._fitted_signature = signature

        accuracy: float | None = None
        try:
            p = self._classifier.predict_proba_feasible(X)
            predicted = np.where(p >= 0.5, FEASIBLE, INFEASIBLE)
            accuracy = float(np.mean(predicted == y))
        except Exception:  # a metric must never cost the fit it describes
            logger.debug("feasibility_train_accuracy_unavailable", exc_info=True)

        self._event(
            "learned_feasibility_fit",
            n_feasible=self.n_feasible,
            n_infeasible=self.n_infeasible,
            train_accuracy=accuracy,
            clamp=self.config.clamp if self.config.min_filter else None,
        )

    def _event(self, name: str, **payload: Any) -> None:
        logger.info(name, **payload)
        if self._emit is not None:
            try:
                self._emit(name, **payload)
            except Exception:
                logger.debug("feasibility_emit_failed", event=name, exc_info=True)
