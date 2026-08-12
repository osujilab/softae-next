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

    def suggest(self) -> dict[str, Any] | None:
        if self._grid_index >= len(self._grid):
            return None
        point = self._grid[self._grid_index]
        self._grid_index += 1
        return point

    def tell(self, params: dict[str, Any], result: float) -> None:
        self._history.append((params, result))

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._find_best()
