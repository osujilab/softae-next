"""Exhaustive grid search optimizer."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer


class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive grid search over a bounded parameter space.

    Parameters
    ----------
    parameter_space
        Search space definition (see :class:`BaseOptimizer`).
    objective
        ``"maximize"`` or ``"minimize"``.
    seed
        Unused (grid is deterministic); accepted for API uniformity.
    n_points
        Number of points per continuous/integer dimension.
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
        *,
        n_points: int = 5,
    ) -> None:
        super().__init__(parameter_space, objective, seed)
        if n_points < 1:
            raise OptimizerError("n_points must be >= 1")
        self._n_points = n_points
        self._grid = self._build_grid()
        self._grid_index = 0

    def _build_grid(self) -> list[dict[str, Any]]:
        names = list(self._parameter_space.keys())
        axes: list[list[Any]] = []
        for name in names:
            spec = self._parameter_space[name]
            ptype = spec["type"]
            if ptype == "float":
                axes.append(
                    np.linspace(spec["low"], spec["high"], self._n_points).tolist()
                )
            elif ptype == "int":
                count = min(self._n_points, spec["high"] - spec["low"] + 1)
                raw = np.linspace(spec["low"], spec["high"], count)
                vals = sorted(set(int(round(v)) for v in raw))
                axes.append(vals)
            else:  # categorical
                axes.append(list(spec["choices"]))
        return [dict(zip(names, combo)) for combo in itertools.product(*axes)]

    def _propose(self) -> dict[str, Any] | None:
        """Walk the cursor forward one point.

        No rejection loop of its own: the template retries, and because the
        cursor always advances a refused point is never revisited. ``None`` here
        means the grid is walked out — whether that is convergence or a space the
        filter refused is read off :attr:`n_rejected`.
        """
        if self._grid_index >= len(self._grid):
            return None
        point = self._grid[self._grid_index]
        self._grid_index += 1
        return point

    @property
    def n_grid_points(self) -> int:
        """Total points in the grid, so a caller can compare against
        :attr:`n_rejected` and tell rejection from convergence."""
        return len(self._grid)

    def tell(self, params: dict[str, Any], result: float) -> None:
        self._history.append((params, result))

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._find_best()

    # ── Serialization (P3.1) ────────────────────────────────────────

    @classmethod
    def _construct_from(cls, state: dict[str, Any]) -> "GridSearchOptimizer":
        """Rebuild from a checkpoint.

        Without this the base default rebuilt the grid at ``n_points=5`` whatever
        the campaign declared, and with a cursor at 0 — so a resumed grid run
        silently re-walked, on a differently shaped grid, and nothing went red.
        """
        extra = state.get("extra") or {}
        return cls(
            state["parameter_space"],
            state.get("objective", "maximize"),
            state.get("seed"),
            n_points=int(extra.get("n_points", 5)),
        )

    def _state_extra(self) -> dict[str, Any]:
        return {
            **super()._state_extra(),
            "n_points": self._n_points,
            "grid_index": self._grid_index,
        }

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        super()._restore_extra(extra)
        self._grid_index = int(extra.get("grid_index", 0))
