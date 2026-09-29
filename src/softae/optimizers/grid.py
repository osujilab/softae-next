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
        Seeds the *shuffle* only. With ``shuffle=False`` (the default) the grid
        is deterministic and the seed is unused, which is what it always was.
    n_points
        Number of points per continuous/integer dimension.
    shuffle
        Walk the grid in a random order instead of ``itertools.product`` order.
        The grid's *contents* are untouched — only the cursor's route through
        it — so an exhaustive walk still visits every point exactly once. The
        permutation is drawn once, at construction, from
        ``np.random.default_rng(seed)``: a given ``seed`` reproduces a given
        order, and **``seed=None`` seeds from OS entropy, so the order is not
        reproducible across constructions.** That is acceptable because the
        order is checkpointed rather than redrawn (see :meth:`_draw_order`), so
        a resumed run walks the same route either way.
    """

    def __init__(
        self,
        parameter_space: dict[str, dict[str, Any]],
        objective: str = "maximize",
        seed: int | None = None,
        *,
        n_points: int = 5,
        shuffle: bool = False,
    ) -> None:
        super().__init__(parameter_space, objective, seed)
        if n_points < 1:
            raise OptimizerError("n_points must be >= 1")
        self._n_points = n_points
        self._shuffle = bool(shuffle)
        self._grid = self._build_grid()
        self._order = self._draw_order()
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

    def _draw_order(self) -> list[int]:
        """Indices into :attr:`_grid`, in the order the cursor will walk them.

        Drawn **once**, here, and thereafter carried in the checkpoint rather
        than redrawn on restore: ``seed=None`` seeds from OS entropy, so a
        resume that redrew would walk a different route and re-cast
        compositions already measured — the exact fault ``_construct_from``
        exists to prevent for ``n_points``.
        """
        if not self._shuffle:
            return list(range(len(self._grid)))
        return [int(i) for i in np.random.default_rng(self._seed).permutation(
            len(self._grid))]

    def _propose(self) -> dict[str, Any] | None:
        """Walk the cursor forward one point.

        No rejection loop of its own: the template retries, and because the
        cursor always advances a refused point is never revisited. ``None`` here
        means the grid is walked out — whether that is convergence or a space the
        filter refused is read off :attr:`n_rejected`.
        """
        if self._grid_index >= len(self._order):
            return None
        point = self._grid[self._order[self._grid_index]]
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
            shuffle=bool(extra.get("shuffle", False)),
        )

    def _state_extra(self) -> dict[str, Any]:
        state = {
            **super()._state_extra(),
            "n_points": self._n_points,
            "grid_index": self._grid_index,
            "shuffle": self._shuffle,
        }
        # Only the shuffled walk needs its route recorded; the unshuffled one is
        # the identity and recomputing it cannot disagree.
        if self._shuffle:
            state["grid_order"] = list(self._order)
        return state

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        super()._restore_extra(extra)
        self._grid_index = int(extra.get("grid_index", 0))
        if not self._shuffle:
            return
        order = extra.get("grid_order")
        if not isinstance(order, list) or len(order) != len(self._grid):
            # Refuse rather than fall back on the constructor's fresh draw: that
            # is the silent re-cast, wearing a successful resume's clothes.
            raise OptimizerError(
                "Shuffled GridSearchOptimizer checkpoint carries no usable "
                "'grid_order'; resuming would redraw the permutation and "
                "re-cast compositions already measured."
            )
        self._order = [int(i) for i in order]
