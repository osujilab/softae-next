"""Batch-proposal strategies for parallel (q>1) Bayesian optimization.

A :class:`BatchStrategy` turns a sequential optimizer into a *batch* proposer:
given a request for ``q`` points to evaluate in parallel, it returns q **diverse**
suggestions so they are not near-duplicates.  The strategy is pluggable so the
diversification method can be swapped without touching the optimizer or the loop.

Built-ins (both fantasy-based — pick, temporarily "tell" a fantasy value so the
surrogate treats the point as evaluated, refit, repeat; all fantasies are removed
before returning):

* :class:`ConstantLiarStrategy` — every fantasy is the same pessimistic constant
  (the worst observed objective).  Robust, cheap, the default.
* :class:`KrigingBelieverStrategy` — each fantasy is the GP posterior **mean** at
  the just-picked point (the model's own belief).  Sharper than constant-liar
  when the surrogate is trustworthy.

Extension seam:

* :class:`BoTorchMonteCarloStrategy` — a lazy-imported hook for Monte-Carlo batch
  acquisition (qEI / qNEI / qLogEI) from Ax/BoTorch.  Stubbed: it raises a clear
  install/enable message so the integration point is explicit and discoverable.

Resolve a name or instance with :func:`make_batch_strategy`.
"""

from __future__ import annotations

import abc
import importlib.util
from typing import Any

from softae.errors import OptimizerError


class BatchStrategy(abc.ABC):
    """Propose ``q`` diverse points from a sequential optimizer."""

    #: Registry key.
    name: str = "batch"
    #: pip distributions this strategy needs (checked by :meth:`ensure_available`).
    requires: tuple[str, ...] = ()

    @abc.abstractmethod
    def propose(self, optimizer: Any, q: int) -> list[dict[str, Any]]:
        """Return up to ``q`` diverse parameter dicts (shorter if exhausted)."""

    @classmethod
    def ensure_available(cls) -> None:
        """Raise a clear :class:`OptimizerError` if a required package is missing."""
        for pkg in cls.requires:
            if importlib.util.find_spec(pkg) is None:
                raise OptimizerError(
                    f"batch strategy '{cls.name}' needs '{pkg}'. "
                    f"Install it with: pip install softae[bo-{cls.name}]"
                )


class _SequentialFantasyStrategy(BatchStrategy):
    """Shared engine for fantasy-based batch proposal.

    Iteratively: pick a point via ``optimizer.suggest()``, temporarily append a
    *fantasy* ``(params, value)`` to the optimizer's history so the next pick
    treats it as already-evaluated, and refit on the next ``suggest``.  Every
    fantasy is removed before returning, so the real per-evaluation objectives
    (told later) replace them.  Subclasses supply the fantasy value.
    """

    def propose(self, optimizer: Any, q: int) -> list[dict[str, Any]]:
        if q < 1:
            raise OptimizerError("batch size q must be >= 1")
        real_n = len(optimizer._history)
        self._prepare(optimizer)
        batch: list[dict[str, Any]] = []
        try:
            for i in range(q):
                p = optimizer.suggest()
                if p is None:
                    break
                batch.append(p)
                optimizer._history.append((p, self._fantasy(optimizer, p, i)))
        finally:
            del optimizer._history[real_n:]  # drop every fantasy
        return batch

    def _prepare(self, optimizer: Any) -> None:
        """Compute any once-per-batch state (from the real history only)."""

    @abc.abstractmethod
    def _fantasy(self, optimizer: Any, params: dict[str, Any], index: int) -> float:
        """The fantasy objective to temporarily assign to *params*."""


class ConstantLiarStrategy(_SequentialFantasyStrategy):
    """Constant-liar: a single pessimistic fantasy (worst observed) for all picks."""

    name = "constant_liar"

    def _prepare(self, optimizer: Any) -> None:
        ys = [v for _, v in optimizer._history]
        if ys:
            self._liar = min(ys) if optimizer._objective == "maximize" else max(ys)
        else:
            self._liar = 0.0

    def _fantasy(self, optimizer: Any, params: dict[str, Any], index: int) -> float:
        return self._liar


class KrigingBelieverStrategy(_SequentialFantasyStrategy):
    """Kriging believer: each fantasy is the GP posterior mean at the picked point."""

    name = "kriging_believer"

    def _fantasy(self, optimizer: Any, params: dict[str, Any], index: int) -> float:
        return optimizer._posterior_mean(params)


class BoTorchMonteCarloStrategy(BatchStrategy):
    """Hook for Monte-Carlo batch acquisition (qEI/qNEI/qLogEI) via Ax/BoTorch.

    Not yet wired — this is the extension seam.  ``propose`` first checks the
    dependency (clear install hint if absent), then raises ``NotImplementedError``
    so the integration point is explicit rather than silently falling back.
    """

    name = "botorch_mc"
    requires = ("botorch",)

    def propose(self, optimizer: Any, q: int) -> list[dict[str, Any]]:
        self.ensure_available()
        raise NotImplementedError(
            "BoTorch Monte-Carlo batch acquisition (qEI/qNEI/qLogEI) is a planned "
            "integration; wire it here, proposing q points jointly from the "
            "optimizer's parameter space and observed history."
        )


#: Registry mapping config strings → strategy classes.
BATCH_STRATEGIES: dict[str, type[BatchStrategy]] = {
    ConstantLiarStrategy.name: ConstantLiarStrategy,
    KrigingBelieverStrategy.name: KrigingBelieverStrategy,
    BoTorchMonteCarloStrategy.name: BoTorchMonteCarloStrategy,
}


def make_batch_strategy(strategy: str | BatchStrategy) -> BatchStrategy:
    """Resolve a batch-strategy name or instance to a :class:`BatchStrategy`."""
    if isinstance(strategy, BatchStrategy):
        return strategy
    if strategy not in BATCH_STRATEGIES:
        raise OptimizerError(
            f"unknown batch strategy '{strategy}'; available: {sorted(BATCH_STRATEGIES)}"
        )
    return BATCH_STRATEGIES[strategy]()
