"""Base optimizer ABC for autonomous experimentation."""

from __future__ import annotations

import abc
from typing import Any

from softae.errors import OptimizerError

_VALID_TYPES = {"float", "int", "categorical"}


class BaseOptimizer(abc.ABC):
    """Abstract base class for all SoftAE optimizers.

    Subclasses must implement :meth:`suggest`, :meth:`tell`, and :meth:`best`.
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
        #: ``(params) -> bool`` — a known, deterministic, cheap constraint the
        #: optimizer must not propose outside of (P7.1). Declared here so every
        #: optimizer carries the attribute even if only some act on it, and so
        #: the caller sets it the same way regardless of backend.
        #:
        #: Deliberately NOT serialized by ``to_dict``: it is a live callable
        #: belonging to the host's twin, and a resumed run rebuilds it from the
        #: spec rather than restoring a stale closure.
        self.feasibility_fn = None
        self._history: list[tuple[dict[str, Any], float]] = []

    # ── Abstract methods ────────────────────────────────────────────

    @abc.abstractmethod
    def suggest(self) -> dict[str, Any] | None:
        """Return the next parameter set to evaluate, or ``None`` if exhausted."""

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
        """Subclass configuration/counters worth checkpointing."""
        return {}

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        """Apply what :meth:`_state_extra` produced."""


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
