"""One suggestion, cast into *k* wells.

The campaign loop already works per batch member, and a batch member is already
a well: it casts, scores and ``tell``s each member on its own channel. So
replication is a batch-shape transform and nothing downstream has to change —
``suggest_batch(q)`` asks the inner optimizer for ``ceil(q / k)`` distinct
compositions and repeats each one ``k`` times — adjacently, so a pair lands on
neighbouring channels, or interleaved, so it does not.

``budget`` therefore counts *distinct compositions*, on the inner optimizer;
the loop ceiling of ``budget x k`` wells is the wiring's business, not this
class's. The inner is told once per WELL — k observations at the same x, which
is what replicates are for and must never be de-duplicated.
"""

from __future__ import annotations

from typing import Any

import structlog

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer

logger = structlog.get_logger(__name__)


class ReplicatingOptimizer(BaseOptimizer):
    """Emits every inner proposal *replicates* times, ordered by *layout*.

    A **view** onto the inner optimizer rather than a second optimizer: history,
    RNG, budget, feasibility hook and counters all live in the inner and are
    forwarded, so the base's own feasibility loop never runs a second time out
    here and a checkpoint carries exactly one search state.
    """

    #: ``adjacent`` puts a pair on neighbouring channels; ``interleaved`` emits
    #: the block k times, so a pair sits ``ceil(q/k)`` apart instead.
    LAYOUTS = ("adjacent", "interleaved")

    def __init__(self, inner: BaseOptimizer, replicates: int,
                 layout: str = "adjacent") -> None:
        if not isinstance(inner, BaseOptimizer):
            raise OptimizerError("inner must be a BaseOptimizer instance")
        if not isinstance(replicates, int) or replicates < 1:
            raise OptimizerError(f"replicates must be an int >= 1, got {replicates!r}")
        if layout not in self.LAYOUTS:
            raise OptimizerError(
                f"layout must be one of {', '.join(self.LAYOUTS)}; got {layout!r}"
            )
        # Bound BEFORE super().__init__(): the forwarding properties below read
        # self._inner, and the base constructor assigns through three of them.
        self._inner = inner
        self._k = replicates
        self._layout = layout
        carried = (inner._history, inner.check_feasible_fn, inner.feasibility_fn)
        super().__init__(inner._parameter_space, inner._objective, inner._seed)
        # Put back what the base constructor blanked through those properties:
        # wrapping an optimizer must not reset the one being wrapped.
        inner._history, inner.check_feasible_fn, inner.feasibility_fn = carried

    # ── Forwarding: the inner holds the state ───────────────────────

    @property
    def inner(self) -> BaseOptimizer:
        """The wrapped optimizer."""
        return self._inner

    @property
    def replicates(self) -> int:
        """Wells cast per distinct composition."""
        return self._k

    @property
    def layout(self) -> str:
        """How a round's replicates are ordered — see :attr:`LAYOUTS`."""
        return self._layout

    @property
    def _history(self) -> list[tuple[dict[str, Any], float]]:
        # Forwarded rather than mirrored, so the base's `history`, `n_trials`
        # and `from_dict` history restore all land on the inner's single copy.
        return self._inner._history

    @_history.setter
    def _history(self, value: list[tuple[dict[str, Any], float]]) -> None:
        self._inner._history = value

    @property
    def check_feasible_fn(self):
        """Tri-state hook. The campaign wiring sets this on the *outer* object."""
        return self._inner.check_feasible_fn

    @check_feasible_fn.setter
    def check_feasible_fn(self, fn) -> None:
        self._inner.check_feasible_fn = fn

    @property
    def feasibility_fn(self):
        """Legacy bool hook. The campaign wiring sets this on the *outer* object."""
        return self._inner.feasibility_fn

    @feasibility_fn.setter
    def feasibility_fn(self, fn) -> None:
        self._inner.feasibility_fn = fn

    @property
    def n_rejected(self) -> int:
        return self._inner.n_rejected

    @property
    def n_unchecked(self) -> int:
        return self._inner.n_unchecked

    # ── Proposal ────────────────────────────────────────────────────

    def suggest(self) -> dict[str, Any] | None:
        """The inner's single-point path, unchanged.

        The non-batch loop already casts one suggestion across every channel, so
        replicating here would multiply what is replicated already.
        """
        return self._inner.suggest()

    def suggest_batch(self, q: int) -> list[dict[str, Any]]:
        """``ceil(q / k)`` distinct compositions, each repeated k times, cut to q."""
        proposals = self._inner.suggest_batch(-(-q // self._k))
        if q % self._k:
            logger.warning(
                "replicate_round_not_whole", q=q, replicates=self._k,
                layout=self._layout,
                msg="round size is not a whole number of replicate groups; the "
                    "last composition will be cast fewer than k times",
            )
        if self._layout == "interleaved":
            expanded = [p for _ in range(self._k) for p in proposals]
        else:
            expanded = [p for p in proposals for _ in range(self._k)]
        return expanded[:q]

    def _propose(self) -> dict[str, Any] | None:
        """Unreachable through this class — `suggest` and `suggest_batch`
        delegate whole — but the base declares it abstract."""
        return self._inner._propose()

    def tell(self, params: dict[str, Any], result: float) -> None:
        """One call per WELL: the inner deliberately receives k observations at
        the same x, which is the scatter replication exists to measure."""
        self._inner.tell(params, result)

    def best(self) -> tuple[dict[str, Any], float] | None:
        return self._inner.best()

    # ── Serialization (P3.1) ────────────────────────────────────────

    def _rng_state(self) -> Any:
        # The wrapper has no RNG of its own; the inner's rides in extra["inner"].
        return None

    def _state_extra(self) -> dict[str, Any]:
        # The inner's FULL checkpoint, not a summary: its budget, counters and
        # RNG are the search state, and `from_dict` must rebuild it by name.
        return {"replicates": self._k, "layout": self._layout,
                "inner": self._inner.to_dict()}

    def _restore_extra(self, extra: dict[str, Any]) -> None:
        """No-op: :meth:`_construct_from` already rebuilt the inner from its own
        checkpoint. Applying the base restore here would zero the feasibility
        counters the inner just recovered."""

    @classmethod
    def _construct_from(cls, state: dict[str, Any]) -> "ReplicatingOptimizer":
        extra = state.get("extra") or {}
        inner_state = extra.get("inner")
        if not isinstance(inner_state, dict):
            raise OptimizerError(
                "ReplicatingOptimizer checkpoint carries no inner optimizer state; "
                "cannot resume safely."
            )
        return cls(
            BaseOptimizer.from_dict(inner_state),
            int(extra.get("replicates", 1)),
            # No key = a checkpoint predating the option, which ran adjacent.
            layout=extra.get("layout") or "adjacent",
        )
