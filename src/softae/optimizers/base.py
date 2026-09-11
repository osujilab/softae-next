"""Base optimizer ABC for autonomous experimentation."""

from __future__ import annotations

import abc
import enum
from typing import Any

import structlog

from softae.errors import OptimizerError

logger = structlog.get_logger(__name__)

_VALID_TYPES = {"float", "int", "categorical"}

#: Redraws allowed per :meth:`BaseOptimizer.suggest` before the space is
#: declared infeasible rather than searched.
DEFAULT_FEASIBILITY_MAX_TRIES = 200


class Feasibility(enum.Enum):
    """Verdict of a deposition-feasibility check on one candidate point.

    Tri-state on purpose. The predecessor contract was ``(params) -> bool`` with
    two separate spellings of "unknown" collapsed onto ``True``: a raising check,
    and a board that declares no capacity. Both then read, to every consumer, as
    *checked and clean* — SUBAGENT_RULES 3.1(a).

    :attr:`UNCHECKED` is admitted by the optimizer (an unsolvable point is not a
    proven-bad one, and carving an invisible hole in the search space is worse)
    but it is **counted**, so the pre-cast gate and the run report can act on
    something the optimizer could not vouch for.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNCHECKED = "unchecked"


class BaseOptimizer(abc.ABC):
    """Abstract base class for all SoftAE optimizers.

    Subclasses implement :meth:`_propose`, :meth:`tell`, and :meth:`best`.
    :meth:`suggest` is a **template method** owned by this class: it wraps
    ``_propose`` in the feasibility loop so every backend — random, grid,
    Bayesian, pooled — is filtered identically, rather than only the one that
    happened to read the hook.
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
    ) -> None:
        if not isinstance(parameter_space, dict) or len(parameter_space) == 0:
            raise OptimizerError("parameter_space must be a non-empty dict")

        for name, spec in parameter_space.items():
            if "type" not in spec:
                raise OptimizerError(f"Parameter '{name}' missing 'type' key")
            ptype = spec["type"]
            if ptype not in _VALID_TYPES:
                raise OptimizerError(
                    f"Parameter '{name}' has unknown type '{ptype}'; "
                    f"expected one of {_VALID_TYPES}"
                )
            if ptype in ("float", "int"):
                if "low" not in spec or "high" not in spec:
                    raise OptimizerError(
                        f"Parameter '{name}' (type={ptype}) requires 'low' and 'high'"
                    )
                if spec["low"] >= spec["high"]:
                    raise OptimizerError(
                        f"Parameter '{name}': low ({spec['low']}) must be < high ({spec['high']})"
                    )
            elif ptype == "categorical":
                choices = spec.get("choices", [])
                if not choices:
                    raise OptimizerError(
                        f"Parameter '{name}' (categorical) requires a non-empty 'choices' list"
                    )

        if objective not in ("maximize", "minimize"):
            raise OptimizerError(
                f"objective must be 'maximize' or 'minimize', got '{objective}'"
            )

        self._parameter_space = parameter_space
        self._objective = objective
        self._seed = seed
        #: ``(params) -> Feasibility`` — the tri-state constraint every optimizer
        #: is filtered by (W2). Set externally by the campaign host; ``None``
        #: means no constraint was declared, which is *not* the same as one that
        #: could not answer.
        #:
        #: Deliberately NOT serialized by ``to_dict``: it is a live callable
        #: belonging to the host's twin, and a resumed run rebuilds it from the
        #: spec rather than restoring a stale closure.
        self.check_feasible_fn = None
        #: Legacy ``(params) -> bool | None`` hook (P7.1), still honoured because
        #: the campaign wiring sets it. Adapted in :meth:`check_feasible`:
        #: ``None`` — the "board declares no capacity" spelling — and a raising
        #: call both become :attr:`Feasibility.UNCHECKED` rather than ``True``.
        self.feasibility_fn = None
        #: Redraws per :meth:`suggest`. Instance-level so a subclass or a test
        #: can shorten it without touching the module default.
        self._feasibility_max_tries = DEFAULT_FEASIBILITY_MAX_TRIES
        #: Proposals refused by the filter, and proposals it could not judge.
        #: Both are checkpointed: a resumed run must still be able to report
        #: that the space was mostly infeasible.
        self._n_rejected = 0
        self._n_unchecked = 0
        self._history: list[tuple[dict[str, Any], float]] = []

    # ── Feasibility (W2) ────────────────────────────────────────────

    def check_feasible(self, params: dict[str, Any]) -> "Feasibility":
        """Tri-state verdict on *params*.

        No hook at all → :attr:`Feasibility.FEASIBLE`: nothing was declared, so
        there is no constraint to violate. A hook that *exists* and cannot answer
        → :attr:`Feasibility.UNCHECKED`. Those two must not share a token, which
        is the whole point of the enum.
        """
        fn = self.check_feasible_fn
        if fn is not None:
            try:
                verdict = fn(params)
            except Exception:
                logger.warning("check_feasible_fn_failed", exc_info=True)
                return Feasibility.UNCHECKED
            if isinstance(verdict, Feasibility):
                return verdict
            logger.warning(
                "check_feasible_fn_returned_non_verdict", got=repr(verdict),
                msg="expected a Feasibility member; treating as unchecked",
            )
            return Feasibility.UNCHECKED

        fn = self.feasibility_fn
        if fn is None:
            return Feasibility.FEASIBLE
        try:
            verdict = fn(params)
        except Exception:
            logger.warning("feasibility_fn_failed", exc_info=True)
            return Feasibility.UNCHECKED
        if verdict is None:
            return Feasibility.UNCHECKED
        return Feasibility.FEASIBLE if verdict else Feasibility.INFEASIBLE

    @property
    def n_rejected(self) -> int:
        """Proposals the filter refused. Non-zero alongside a ``None`` from
        :meth:`suggest` means *exhausted by rejection*, not converged."""
        return self._n_rejected

    @property
    def n_unchecked(self) -> int:
        """Proposals admitted that the filter could not judge."""
        return self._n_unchecked

    # ── Template method ─────────────────────────────────────────────

    def suggest(self) -> dict[str, Any] | None:
        """Next parameter set to evaluate, or ``None`` if exhausted.

        Two ``None``s, and the caller can tell them apart via :attr:`n_rejected`:
        ``_propose`` returning ``None`` is *genuine* exhaustion (budget spent,
        grid walked, pool empty); running out of tries is a space the filter
        refused. A grid exhausted by rejection and a grid exhausted by search
        must not print the same word.
        """
        for _ in range(self._feasibility_max_tries):
            params = self._propose()
            if params is None:
                return None
            verdict = self.check_feasible(params)
            if verdict is Feasibility.INFEASIBLE:
                self._n_rejected += 1
                continue
            if verdict is Feasibility.UNCHECKED:
                self._n_unchecked += 1
            self._on_accept(params)
            return params
        logger.warning(
            "search_space_infeasible", tries=self._feasibility_max_tries,
            n_rejected=self._n_rejected,
            msg="every proposal was refused by the feasibility filter; the "
                "declared bounds are probably wrong",
        )
        return None

    def _on_accept(self, params: dict[str, Any]) -> None:
        """Hook fired once per *accepted* proposal.

        Anything that counts campaign trials belongs here rather than in
        ``_propose``: a rejected draw must not burn budget.
        """

    # ── Abstract methods ────────────────────────────────────────────

    @abc.abstractmethod
    def _propose(self) -> dict[str, Any] | None:
        """One candidate point, unfiltered, or ``None`` if genuinely exhausted."""

    @abc.abstractmethod
    def tell(self, params: dict[str, Any], result: float) -> None:
        """Record an observation (parameter set → objective value)."""

    @abc.abstractmethod
    def best(self) -> tuple[dict[str, Any], float] | None:
        """Return ``(best_params, best_objective)`` or ``None`` if empty."""

    # ── Batch proposal ──────────────────────────────────────────────

    def suggest_batch(self, q: int) -> list[dict[str, Any]]:
        """Propose up to *q* points for parallel (batched) evaluation.

        Default implementation: *q* independent :meth:`suggest` draws — correct
        for stochastic optimizers (random search) where draws are already
        diverse.  GP-based optimizers override this to *diversify within the
        batch* (e.g. constant-liar) so the q points are not near-duplicates.
        Stops early (returns a shorter list) if the optimizer exhausts.
        """
        if q < 1:
            raise OptimizerError("batch size q must be >= 1")
        out: list[dict[str, Any]] = []
        for _ in range(q):
            s = self.suggest()
            if s is None:
                break
            out.append(s)
        return out

    # ── Concrete properties ─────────────────────────────────────────

    @property
    def history(self) -> list[tuple[dict[str, Any], float]]:
        """Chronological list of ``(params, result)`` observations."""
        return list(self._history)

    @property
    def n_trials(self) -> int:
        """Number of observations recorded so far."""
        return len(self._history)

    # ── Shared helpers ──────────────────────────────────────────────

    def _find_best(self) -> tuple[dict[str, Any], float] | None:
        """Scan ``_history`` for the best observation."""
        if not self._history:
            return None
        if self._objective == "maximize":
            return max(self._history, key=lambda x: x[1])
        return min(self._history, key=lambda x: x[1])

    # ── Serialization (P3.1) ────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot sufficient to reconstruct this optimizer.

        **What is deliberately NOT stored: fitted surrogate hyperparameters.**
        The GP is constructed with a fixed ``random_state`` and refit from the
        full history, so rebuilding from ``history`` reproduces the same fitted
        model exactly — verified by
        ``test_gp_refit_from_history_is_deterministic``. Persisting sklearn
        internals instead would be fragile across library versions and could
        silently restore a model that no longer matches the data.

        **What IS stored, because history alone is not enough: RNG state.**
        Replaying observations with :meth:`tell` does not advance the RNG the way
        the original interleaved suggest/tell did, so a naive replay resumes on a
        *different* random stream than a true continuation — it re-draws
        candidate pools the run already used. See
        ``test_replaying_history_alone_diverges_from_a_true_continuation``.
        """
        return {
            "optimizer": type(self).__name__,
            "parameter_space": self._parameter_space,
            "objective": self._objective,
            "seed": self._seed,
            "history": [[dict(p), float(v)] for p, v in self._history],
            "rng_state": self._rng_state(),
            "extra": self._state_extra(),
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "BaseOptimizer":
        """Rebuild an optimizer from :meth:`to_dict`.

        Dispatches on the recorded class name, so a checkpoint written by one
        optimizer cannot be silently resumed as another — an unknown or
        mismatched name raises rather than degrading to a different search
        strategy mid-campaign.
        """
        name = state.get("optimizer")
        target = cls if cls.__name__ == name else _OPTIMIZER_REGISTRY.get(name)
        if target is None:
            raise OptimizerError(
                f"Unknown optimizer '{name}' in checkpoint; cannot resume safely."
            )

        obj = target._construct_from(state)
        obj._history = [
            (dict(p), float(v)) for p, v in state.get("history", [])
        ]
        obj._restore_extra(state.get("extra") or {})
        rng_state = state.get("rng_state")
        if rng_state is not None:
            obj._restore_rng(rng_state)
        return obj

    @classmethod
    def _construct_from(cls, state: dict[str, Any]) -> "BaseOptimizer":
        """Build a bare instance from a checkpoint (subclasses add their kwargs)."""
        return cls(
            state["parameter_space"],
            state.get("objective", "maximize"),
            state.get("seed"),
        )

    # Subclass hooks — the base has no RNG or strategy config of its own.

    def _rng_state(self) -> Any:
        """JSON-safe RNG state, or ``None`` when the optimizer has no RNG."""
        return None

    def _restore_rng(self, state: Any) -> None:
        """Restore what :meth:`_rng_state` produced."""

    def _state_extra(self) -> dict[str, Any]:
        """Subclass configuration/counters worth checkpointing.

        Subclasses extend this (``{**super()._state_extra(), ...}``) rather than
        replacing it, so the feasibility counters survive every backend's resume
        — the hook itself is rebuilt from the spec, but "how much of this space
        was refused" is history and cannot be recomputed.
        """
        return {"n_rejected": self._n_rejected, "n_unchecked": self._n_unchecked}

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        """Apply what :meth:`_state_extra` produced."""
        self._n_rejected = int(extra.get("n_rejected", 0))
        self._n_unchecked = int(extra.get("n_unchecked", 0))


#: Name → class, for :meth:`BaseOptimizer.from_dict` dispatch. Populated by
#: ``__init_subclass__`` so a new optimizer is resumable without extra wiring.
_OPTIMIZER_REGISTRY: dict[str, type[BaseOptimizer]] = {}


def _register(cls: type[BaseOptimizer]) -> None:
    _OPTIMIZER_REGISTRY[cls.__name__] = cls


_BaseInitSubclass = BaseOptimizer.__init_subclass__


def _init_subclass(cls, **kwargs):  # type: ignore[no-untyped-def]
    _BaseInitSubclass(**kwargs)
    _register(cls)


BaseOptimizer.__init_subclass__ = classmethod(_init_subclass)  # type: ignore[assignment]
