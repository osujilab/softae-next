"""Configuration dataclass for a simulated BO campaign.

Mirrors :class:`softae.analysis.arrhenius.ArrheniusSweepConfig`: ``validate()``,
``to_json()`` / ``from_json()`` (the latter ignores unknown keys so configs from
a newer version load in an older one).  Every knob the campaign engine reads is
a plain primitive here, so a config fully describes a reproducible run.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from softae.campaigns.acquisitions import ACQUISITIONS
from softae.campaigns.derived import DERIVED_OBJECTIVES
from softae.campaigns.noise import NOISE_MODELS
from softae.campaigns.objectives import OBJECTIVES
from softae.errors import CampaignError
from softae.optimizers.surrogates import BACKENDS

_STOPPING_MODES = ("optimization", "model_accuracy")


@dataclass
class BOCampaignConfig:
    """Full specification of one simulated campaign.

    Dataset
    -------
    dataset_adapter, dataset_path, dataset_kwargs
        Which adapter parses the source and its options.

    Optimizer / objective
    ---------------------
    backend, acquisition, objective_direction, transform, n_initial, seed, kappa.

    Noise model
    -----------
    noise_model, noise_sources, noise_combine, noise_channel, target_is_mean,
    replicate_shrinkage, k_fit, var_floor.

    Stopping
    --------
    stopping_mode, rel_tol, abs_tol, patience, rmse_tol, coverage_tol, max_steps.

    Rails
    -----
    rail_sigma_ceiling, rail_fitted_Z_max, exclude_rails_from_optimum, rail_variance.
    """

    # ── dataset ──
    dataset_adapter: str = "aggregated_txt"
    dataset_path: str = ""
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── optimizer / objective ──
    backend: str = "sklearn"
    acquisition: str = "ucb"
    objective_direction: str = "maximize"
    transform: str = "log10_sigma"
    n_initial: int = 5
    seed: int = 0
    kappa: float = 2.0

    # ── temperature-derived objective (P3) ──
    # "none" uses the per-σ transform above; otherwise temperature is folded into
    # an Arrhenius/VFT fit and the named parameter becomes the objective.
    temperature_objective: str = "none"
    target_temp_C: float = 25.0

    # ── noise model ──
    noise_model: str = "composite"
    noise_sources: list[str] = field(default_factory=lambda: ["replicate", "fit_quality"])
    noise_combine: str = "sum"
    noise_channel: str = "alpha"          # "alpha" | "acquisition_weight" | "both"
    target_is_mean: bool = True
    replicate_shrinkage: str = "pooled"   # "pooled" | "none"
    k_fit: float = 1.0
    var_floor: float = 1e-6

    # ── stopping ──
    stopping_mode: str = "optimization"
    rel_tol: float = 1e-2
    abs_tol: float = 0.0
    patience: int = 5
    rmse_tol: float = 0.25
    coverage_tol: float = 0.9
    max_steps: int | None = None

    # ── rails ──
    rail_sigma_ceiling: float | None = 0.05
    rail_fitted_Z_max: float = 150.0
    exclude_rails_from_optimum: bool = True
    rail_variance: float = 100.0

    # ── observability ──
    noiseless_oracle: bool = False
    annotation: str = ""

    # ── validation ────────────────────────────────────────────────────────

    def validate(self, *, pool_size: int | None = None) -> None:
        """Raise :class:`CampaignError` for an invalid configuration.

        ``pool_size`` (when known) lets us check ``n_initial < pool_size``.
        """
        if self.backend not in BACKENDS:
            raise CampaignError(
                f"backend '{self.backend}' unknown; available: {sorted(BACKENDS)}"
            )
        if self.acquisition not in ACQUISITIONS:
            raise CampaignError(
                f"acquisition '{self.acquisition}' unknown; available: {sorted(ACQUISITIONS)}"
            )
        if self.transform not in OBJECTIVES:
            raise CampaignError(
                f"transform '{self.transform}' unknown; available: {sorted(OBJECTIVES)}"
            )
        if self.temperature_objective != "none" and (
            self.temperature_objective not in DERIVED_OBJECTIVES
        ):
            raise CampaignError(
                f"temperature_objective '{self.temperature_objective}' unknown; "
                f"available: 'none' or {sorted(DERIVED_OBJECTIVES)}"
            )
        if self.objective_direction not in ("maximize", "minimize"):
            raise CampaignError("objective_direction must be 'maximize' or 'minimize'")
        if self.noise_model not in NOISE_MODELS:
            raise CampaignError(
                f"noise_model '{self.noise_model}' unknown; available: {sorted(NOISE_MODELS)}"
            )
        if self.noise_combine not in ("sum", "max"):
            raise CampaignError("noise_combine must be 'sum' or 'max'")
        if self.noise_channel not in ("alpha", "acquisition_weight", "both"):
            raise CampaignError(
                "noise_channel must be 'alpha', 'acquisition_weight', or 'both'"
            )
        if self.replicate_shrinkage not in ("pooled", "none"):
            raise CampaignError("replicate_shrinkage must be 'pooled' or 'none'")
        if self.stopping_mode not in _STOPPING_MODES:
            raise CampaignError(f"stopping_mode must be one of {_STOPPING_MODES}")
        if self.n_initial < 1:
            raise CampaignError("n_initial must be >= 1")
        if pool_size is not None and self.n_initial >= pool_size:
            raise CampaignError(
                f"n_initial ({self.n_initial}) must be < pool_size ({pool_size})"
            )
        for tol_name in ("rel_tol", "abs_tol", "rmse_tol"):
            if getattr(self, tol_name) < 0:
                raise CampaignError(f"{tol_name} must be >= 0")
        if not 0.0 <= self.coverage_tol <= 1.0:
            raise CampaignError("coverage_tol must be in [0, 1]")
        if self.patience < 1:
            raise CampaignError("patience must be >= 1")
        if self.var_floor <= 0:
            raise CampaignError("var_floor must be > 0")
        if self.k_fit < 0:
            raise CampaignError("k_fit must be >= 0")
        if self.max_steps is not None and self.max_steps < 1:
            raise CampaignError("max_steps must be >= 1 or None")

    # ── serialisation ───────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BOCampaignConfig":
        d = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def maximize(self) -> bool:
        return self.objective_direction == "maximize"
