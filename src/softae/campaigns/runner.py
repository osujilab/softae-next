"""Offline campaign runner — the hardware-free analogue of ``AutonomousLoop``.

Drives ``suggest → oracle.reveal → tell`` over a finite pool until a stopping
rule fires or the pool/budget is exhausted, recording per-step metrics.  No Qt,
no ``InstrumentManager`` — the oracle replaces instrument execution, so the whole
loop is synchronous and scriptable.

:func:`build_campaign` assembles every component from a
:class:`~softae.campaigns.config.BOCampaignConfig`; :class:`CampaignRunner` runs
a pre-assembled set of components (handy for tests and the benchmark harness).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from softae.campaigns.acquisitions import make_acquisition
from softae.campaigns.adapters import AggregatedTxtAdapter, DatasetAdapter, DataStoreAdapter
from softae.campaigns.config import BOCampaignConfig
from softae.campaigns.datasets import GroundTruthDataset, Oracle
from softae.campaigns.metrics import (
    ModelAccuracyStoppingRule,
    OptimizationStoppingRule,
    StepMetrics,
    StoppingRule,
    best_so_far,
    surrogate_pool_metrics,
)
from softae.campaigns.noise import build_noise_model
from softae.campaigns.objectives import get_transform
from softae.errors import CampaignError
from softae.optimizers.pooled_bayesian import PooledBayesianOptimizer

_ADAPTERS = {
    "aggregated_txt": AggregatedTxtAdapter,
    "datastore": DataStoreAdapter,
}


@dataclass
class CampaignResult:
    """Outcome of one campaign run (JSON-serialisable)."""

    converged: bool
    steps_to_tolerance: int | None
    n_steps: int
    best_params: dict[str, Any]
    best_value: float
    true_optimum_params: dict[str, Any]
    true_optimum_value: float
    steps: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CampaignResult":
        d = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def regret_curve(self) -> list[float]:
        return [s["simple_regret"] for s in self.steps]


class CampaignRunner:
    """Runs one simulated campaign over pre-assembled components."""

    def __init__(
        self,
        optimizer: PooledBayesianOptimizer,
        oracle: Oracle,
        dataset: GroundTruthDataset,
        stopping_rule: StoppingRule,
        *,
        maximize: bool = True,
        max_steps: int | None = None,
        seed: int = 0,
        on_step: Callable[[int, dict, StepMetrics], None] | None = None,
        should_abort: Callable[[], bool] | None = None,
        config: BOCampaignConfig | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.oracle = oracle
        self.dataset = dataset
        self.stopping_rule = stopping_rule
        self.maximize = maximize
        self.pool_size = dataset.size
        self.max_steps = min(max_steps, self.pool_size) if max_steps else self.pool_size
        self.on_step = on_step
        #: Cooperative abort hook checked at the top of each step (e.g. a GUI
        #: Abort button).  Returns True to stop early.
        self.should_abort = should_abort
        self.config = config
        self._rng = np.random.RandomState(seed)

    def run(self) -> CampaignResult:
        opt_params, opt_value = self.dataset.true_optimum(maximize=self.maximize)
        y_true_pool = self.dataset.y_true()

        metrics: list[StepMetrics] = []
        true_values: list[float] = []      # true means of sampled points
        cumulative = 0.0
        converged = False
        steps_to_tol: int | None = None
        step_records: list[dict[str, Any]] = []

        for it in range(self.max_steps):
            if self.should_abort is not None and self.should_abort():
                break  # cooperative abort (e.g. GUI Abort button)
            params = self.optimizer.suggest()
            if params is None:
                break  # pool exhausted
            obs = self.oracle.reveal(params, self._rng)
            self.optimizer.tell(params, obs.value)

            # Regret on the *true* mean of the chosen point (noise-free).
            true_val = self.dataset.true_value(obs.point_id)
            true_values.append(true_val)
            bsf = best_so_far(true_values, self.maximize)

            sign = 1.0 if self.maximize else -1.0
            simple_regret = max(sign * (opt_value - bsf), 0.0)
            cumulative += max(sign * (opt_value - true_val), 0.0)

            rmse, coverage = self._surrogate_metrics(y_true_pool)

            m = StepMetrics(
                iteration=it,
                sampled_value=obs.value,
                best_so_far=bsf,
                simple_regret=simple_regret,
                cumulative_regret=cumulative,
                pool_fraction=(it + 1) / self.pool_size,
                surrogate_rmse=rmse,
                coverage=coverage,
            )
            metrics.append(m)
            step_records.append(
                {
                    "iteration": it,
                    "params": params,
                    "point_id": obs.point_id,
                    "sampled_value": obs.value,
                    "true_value": true_val,
                    "is_rail": obs.is_rail,
                    "best_so_far": bsf,
                    "simple_regret": simple_regret,
                    "cumulative_regret": cumulative,
                    "pool_fraction": m.pool_fraction,
                    "surrogate_rmse": rmse,
                    "coverage": coverage,
                }
            )

            if self.on_step is not None:
                self.on_step(it, params, m)

            if not converged and self.stopping_rule.should_stop(metrics):
                converged = True
                steps_to_tol = it + 1
                break

        best = self.optimizer.best()
        best_params, best_value = best if best is not None else ({}, float("nan"))
        return CampaignResult(
            converged=converged,
            steps_to_tolerance=steps_to_tol,
            n_steps=len(step_records),
            best_params=best_params,
            best_value=best_value,
            true_optimum_params=opt_params,
            true_optimum_value=opt_value,
            steps=step_records,
            config=dataclasses.asdict(self.config) if self.config else None,
        )

    def _surrogate_metrics(self, y_true_pool: np.ndarray) -> tuple[float, float]:
        """Predict over the full pool with the optimizer's fitted backend."""
        try:
            pool_X = self.dataset.encoded_pool(self.optimizer._encode)
            mu, sigma = self.optimizer.backend.predict(pool_X)
        except Exception:
            return float("nan"), float("nan")
        return surrogate_pool_metrics(mu, sigma, y_true_pool)


# ---------------------------------------------------------------------------
# Config-driven assembly
# ---------------------------------------------------------------------------

def build_dataset(
    config: BOCampaignConfig,
    *,
    feasible: Callable[[dict[str, Any]], bool] | None = None,
    adapter: DatasetAdapter | None = None,
) -> GroundTruthDataset:
    """Build the ground-truth dataset described by *config*.

    ``feasible`` optionally pre-filters candidates by their parameter dict — e.g.
    a composition simplex / sum-to-one constraint for a 3-component system.
    ``adapter`` overrides adapter construction — required for the ``datastore``
    adapter, which needs a live :class:`~softae.core.data_store.DataStore` object
    rather than a file path.
    """
    if adapter is None:
        if config.dataset_adapter not in _ADAPTERS:
            raise CampaignError(
                f"unknown dataset_adapter '{config.dataset_adapter}'; "
                f"available: {sorted(_ADAPTERS)}"
            )
        adapter_cls = _ADAPTERS[config.dataset_adapter]
        if adapter_cls is DataStoreAdapter:
            raise CampaignError(
                "the 'datastore' adapter needs a live DataStore; construct "
                "DataStoreAdapter(store, run_id) and pass it via adapter= "
                "(or build the dataset yourself and pass dataset=)."
            )
        adapter = adapter_cls(config.dataset_path, **config.dataset_kwargs)

    # Temperature-derived objective: fold σ(T) into an Arrhenius/VFT parameter.
    if config.temperature_objective != "none":
        from softae.campaigns.derived import build_derived_objective

        derived = build_derived_objective(
            config.temperature_objective,
            target_temp_C=config.target_temp_C,
            var_floor=config.var_floor,
        )
        return GroundTruthDataset.from_tidy_derived(
            adapter.to_tidy(),
            derived_objective=derived,
            rail_sigma_ceiling=config.rail_sigma_ceiling,
            rail_fitted_Z_max=config.rail_fitted_Z_max,
            feasible=feasible,
        )

    transform = get_transform(config.transform)
    noise_model = build_noise_model(
        config.noise_model,
        sources=config.noise_sources,
        combine=config.noise_combine,
        target_is_mean=config.target_is_mean,
        var_floor=config.var_floor,
        k_fit=config.k_fit,
        replicate_shrinkage=config.replicate_shrinkage,
    )
    return GroundTruthDataset.from_adapter(
        adapter,
        transform=transform,
        noise_model=noise_model,
        rail_sigma_ceiling=config.rail_sigma_ceiling,
        rail_fitted_Z_max=config.rail_fitted_Z_max,
        exclude_rails_from_optimum=config.exclude_rails_from_optimum,
        rail_variance=config.rail_variance,
        feasible=feasible,
    )


def build_campaign(
    config: BOCampaignConfig,
    *,
    dataset: GroundTruthDataset | None = None,
    on_step: Callable[[int, dict, StepMetrics], None] | None = None,
    should_abort: Callable[[], bool] | None = None,
    feasible: Callable[[dict[str, Any]], bool] | None = None,
    adapter: DatasetAdapter | None = None,
) -> CampaignRunner:
    """Assemble a :class:`CampaignRunner` from a validated config.

    A pre-built *dataset* may be passed (e.g. by the benchmark harness, which
    loads the immutable dataset once and shares it across many runs).  ``feasible``
    pre-filters candidates (e.g. a composition constraint) when *dataset* is built
    here.  ``adapter`` overrides adapter construction (required for ``datastore``).
    """
    if dataset is None:
        dataset = build_dataset(config, feasible=feasible, adapter=adapter)
    config.validate(pool_size=dataset.size)

    # Per-candidate variances keyed by point_id; re-key onto the optimizer's
    # encoded-vector space so observed/candidate lookups are exact.
    pool_var_by_id = dataset.pool_variance()

    optimizer = PooledBayesianOptimizer(
        dataset.parameter_space(),
        objective=config.objective_direction,
        seed=config.seed,
        pool=dataset.pool_points(),
        backend=config.backend,
        acquisition=make_acquisition(config.acquisition),
        n_initial=config.n_initial,
        kappa=config.kappa,
        # "acquisition_weight" withholds variances from the GP alpha but still
        # exposes them to the acquisition via candidate_variance.
        use_alpha=config.noise_channel in ("alpha", "both"),
    )
    optimizer._pool_variance = {
        optimizer._key(c.params): pool_var_by_id[c.point_id] for c in dataset.cells
    }

    oracle = Oracle(dataset, noiseless=config.noiseless_oracle)

    if config.stopping_mode == "optimization":
        _, opt_value = dataset.true_optimum(maximize=config.maximize)
        rule: StoppingRule = OptimizationStoppingRule(
            opt_value, rel_tol=config.rel_tol, abs_tol=config.abs_tol,
            patience=config.patience,
        )
    else:
        rule = ModelAccuracyStoppingRule(
            rmse_tol=config.rmse_tol, coverage_tol=config.coverage_tol
        )

    return CampaignRunner(
        optimizer,
        oracle,
        dataset,
        rule,
        maximize=config.maximize,
        max_steps=config.max_steps,
        seed=config.seed,
        on_step=on_step,
        should_abort=should_abort,
        config=config,
    )


def run_campaign(config: BOCampaignConfig) -> CampaignResult:
    """One-shot: build and run a campaign from a config."""
    return build_campaign(config).run()
