"""Rebuild an interrupted campaign from its checkpoint (P3.3).

P3.2 writes a resume point after every completed iteration. This turns one back
into a runnable state: the optimizer with its history and RNG restored, the
iteration count, and the board the run was on.

**The spec is supplied by the caller, not reconstructed.** A campaign spec holds
live Python objects (``prior_mean`` is an arbitrary callable; ``formulation`` and
``run_plan`` are rich objects), so a lossy rebuild that *looked* complete could
silently resume a different experiment. The caller passes the spec it already
has — the GUI saves one, P6's CLI loads one — and this module's job is to
**verify** it is the same experiment and refuse when it is not.

Reconciliation with the board is deliberately *not* done here: durable electrode
occupancy is already the authority (`_prepare_electrode_allocator` reads it at
start-up and the allocator skips cast wells), so a resumed run inherits the same
protection as a fresh one without a second, divergent code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from softae.errors import OptimizerError
from softae.optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from softae.core.autonomous_wiring import CampaignSpec
    from softae.core.data_store import DataStore

logger = structlog.get_logger(__name__)


class ResumeMismatchError(Exception):
    """The checkpoint does not belong to the spec being resumed.

    Raised rather than warned: continuing would append one experiment's
    observations to another's search, which corrupts both and is not visible in
    the resulting data.
    """


@dataclass
class ResumePlan:
    """Everything needed to continue an interrupted campaign."""

    campaign: str
    optimizer: BaseOptimizer
    iteration: int
    board_id: int | None
    run_id: str | None
    remaining_budget: int
    #: Non-fatal notes the caller should surface (e.g. a prior to re-attach).
    warnings: list[str] = field(default_factory=list)

    @property
    def is_exhausted(self) -> bool:
        """True when the original budget was already met."""
        return self.remaining_budget <= 0


def load_resume_plan(
    data_store: "DataStore",
    spec: "CampaignSpec",
    *,
    strict: bool = True,
) -> ResumePlan | None:
    """Rebuild the optimizer for ``spec.name``, or ``None`` if no checkpoint.

    Parameters
    ----------
    strict
        When ``True`` (the default) a fingerprint mismatch raises
        :class:`ResumeMismatchError`. Pass ``False`` only for inspection — never
        to force a resume onto a changed spec.
    """
    from softae.core.autonomous_wiring import campaign_spec_fingerprint

    cp = data_store.campaign_checkpoint(spec.name)
    if cp is None:
        return None

    warnings: list[str] = []

    # 1. Is this the same experiment?
    #
    # T2.4 note — a checkpoint written before the measurement block existed still
    # verifies here without special-casing. `campaign_spec_fingerprint` omits the
    # block from its payload whenever the modality is the default, so the hash of
    # every spec that could exist before T2.4 is byte-identical to what it was,
    # and a legacy `eis_*` spec hashes the same as its new-form equivalent. No
    # dual-accept fallback is needed; if one ever is, it belongs here.
    stored = json.loads(cp.get("spec_json") or "{}")
    stored_fp = stored.get("fingerprint")
    current_fp = campaign_spec_fingerprint(spec)
    if stored_fp and stored_fp != current_fp:
        msg = (
            f"Checkpoint for '{spec.name}' was written for a different search "
            f"(fingerprint {stored_fp} != {current_fp}). The parameter space, "
            f"objective, or optimizer settings have changed since it was saved."
        )
        if strict:
            raise ResumeMismatchError(msg)
        warnings.append(msg)

    # 2. Rebuild the optimizer (history + RNG; see optimizers/base.to_dict).
    opt_json = cp.get("optimizer_json")
    if not opt_json:
        raise ResumeMismatchError(
            f"Checkpoint for '{spec.name}' has no optimizer state; it cannot be "
            f"resumed without silently restarting the search."
        )
    try:
        optimizer = BaseOptimizer.from_dict(json.loads(opt_json))
    except (OptimizerError, ValueError, KeyError) as exc:
        raise ResumeMismatchError(
            f"Could not rebuild the optimizer for '{spec.name}': {exc}"
        ) from exc

    # 3. Flag anything the caller must re-attach itself.
    requires = stored.get("requires") or {}
    if requires.get("prior_mean") and spec.prior_mean is None:
        warnings.append(
            "The interrupted run used a prior-mean model; this spec has none, so "
            "the surrogate will differ from the original run."
        )
    if requires.get("formulation") and spec.formulation is None:
        warnings.append(
            "The interrupted run used a formulation context; this spec has none."
        )

    iteration = int(cp.get("iteration") or 0)
    remaining = max(0, int(spec.budget) - iteration)
    if remaining == 0:
        warnings.append(
            f"The checkpoint is already at the budget ({iteration}/{spec.budget}); "
            f"raise the budget to continue."
        )

    board_id = cp.get("board_id")
    logger.info(
        "resume_plan_loaded", campaign=spec.name, iteration=iteration,
        remaining_budget=remaining, board_id=board_id,
        n_observations=optimizer.n_trials, warnings=len(warnings),
    )
    return ResumePlan(
        campaign=spec.name,
        optimizer=optimizer,
        iteration=iteration,
        board_id=None if board_id is None else int(board_id),
        run_id=cp.get("run_id"),
        remaining_budget=remaining,
        warnings=warnings,
    )


def describe_resume(plan: ResumePlan) -> str:
    """One-paragraph summary for an operator deciding whether to resume."""
    lines = [
        f"Campaign '{plan.campaign}' stopped after {plan.iteration} iteration(s) "
        f"with {plan.optimizer.n_trials} recorded observation(s).",
    ]
    best = plan.optimizer.best()
    if best is not None:
        params, value = best
        lines.append(f"Best so far: {value:.4g} at {params}.")
    if plan.board_id is not None:
        lines.append(f"It was running on board {plan.board_id}.")
    lines.append(
        f"{plan.remaining_budget} of the budget remain."
        if not plan.is_exhausted else "The budget is already exhausted."
    )
    lines.extend(f"Note: {w}" for w in plan.warnings)
    return "\n".join(lines)
