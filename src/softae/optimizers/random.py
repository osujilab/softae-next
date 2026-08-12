"""Budget-limited random search optimizer."""

from __future__ import annotations

import random as _random
from typing import Any

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """Uniform random sampling over a bounded parameter space.

    Parameters
    ----------
    parameter_space
        Search space definition (see :class:`BaseOptimizer`).
    objective
        ``"maximize"`` or ``"minimize"``.
    seed
        RNG seed for reproducibility.
    n_trials
        Maximum number of suggestions before exhaustion.
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
        *,
        n_trials: int = 20,
    ) -> None:
        super().__init__(parameter_space, objective, seed)
        if n_trials < 1:
            raise OptimizerError("n_trials must be >= 1")
        self._budget = n_trials
        self._rng = _random.Random(seed)
        self._n_suggested = 0

    def suggest(self) -> dict[str, Any] | None:
        if self._n_suggested >= self._budget:
            return None
        params: dict[str, Any] = {}
        for name, spec in self._parameter_space.items():
            ptype = spec["type"]
            if ptype == "float":
                params[name] = self._rng.uniform(spec["low"], spec["high"])
            elif ptype == "int":
                params[name] = self._rng.randint(spec["low"], spec["high"])
            else:  # categorical
                params[name] = self._rng.choice(spec["choices"])
        self._n_suggested += 1
        return params

    def tell(self, params: dict[str, Any], result: float) -> None:
        self._history.append((params, result))

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._find_best()

    # ── Serialization (P3.1) ────────────────────────────────────────

    @classmethod
    def _construct_from(cls, state: dict[str, Any]) -> "RandomSearchOptimizer":
        extra = state.get("extra") or {}
        return cls(
            state["parameter_space"],
            state.get("objective", "maximize"),
            state.get("seed"),
            n_trials=int(extra.get("budget", 20)),
        )

    def _rng_state(self):
        # random.Random.getstate() nests a tuple; JSON round-trips it as lists,
        # so setstate() gets tuples back in _restore_rng.
        version, internal, gauss_next = self._rng.getstate()
        return [version, list(internal), gauss_next]

    def _restore_rng(self, state) -> None:
        version, internal, gauss_next = state
        self._rng.setstate((version, tuple(internal), gauss_next))

    def _state_extra(self) -> dict[str, Any]:
        # n_suggested is the exhaustion counter: without it a resumed run gets a
        # fresh budget and would overrun the campaign's trial count.
        return {"budget": self._budget, "n_suggested": self._n_suggested}

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        self._budget = int(extra.get("budget", self._budget))
        self._n_suggested = int(extra.get("n_suggested", 0))
