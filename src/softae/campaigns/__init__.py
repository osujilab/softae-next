"""Simulated Bayesian-optimization campaign suite.

Treats a finite materials dataset as an *imperfect ground-truth oracle* and
simulates pool-based BO campaigns: at each step the optimizer "uncovers" one of
the provided candidate points, and the suite measures how many steps it takes to
converge — toward the objective optimum or toward a low-uncertainty surrogate.

Headless-first (no GUI/hardware dependency); layered on
:mod:`softae.optimizers` and the project's analysis conventions.
"""

from softae.campaigns.adapters import (
    AggregatedTxtAdapter,
    DatasetAdapter,
    DataStoreAdapter,
)
from softae.campaigns.benchmark import BenchmarkResult, run_grid
from softae.campaigns.config import BOCampaignConfig
from softae.campaigns.datasets import GroundTruthDataset, Observation, Oracle, detect_rails
from softae.campaigns.derived import (
    DERIVED_OBJECTIVES,
    DerivedObjective,
    build_derived_objective,
)
from softae.campaigns.metrics import (
    ModelAccuracyStoppingRule,
    OptimizationStoppingRule,
    StepMetrics,
    StoppingRule,
)
from softae.campaigns.noise import (
    CellStats,
    CompositeNoise,
    FitQualityNoise,
    HomoscedasticNoise,
    NoiseModel,
    ReplicateNoise,
    build_noise_model,
)
from softae.campaigns.objectives import (
    Log10Sigma,
    ObjectiveTransform,
    RawSigma,
    get_transform,
)
from softae.campaigns.pareto import pareto_indices, pareto_mask
from softae.campaigns.persistence import record_campaign
from softae.campaigns.runner import (
    CampaignResult,
    CampaignRunner,
    build_campaign,
    build_dataset,
    run_campaign,
)

__all__ = [
    # adapters / schema
    "DatasetAdapter",
    "AggregatedTxtAdapter",
    "DataStoreAdapter",
    # dataset / oracle
    "GroundTruthDataset",
    "Oracle",
    "Observation",
    "detect_rails",
    # objectives
    "ObjectiveTransform",
    "Log10Sigma",
    "RawSigma",
    "get_transform",
    # noise
    "NoiseModel",
    "CellStats",
    "ReplicateNoise",
    "FitQualityNoise",
    "CompositeNoise",
    "HomoscedasticNoise",
    "build_noise_model",
    # metrics / stopping
    "StepMetrics",
    "StoppingRule",
    "OptimizationStoppingRule",
    "ModelAccuracyStoppingRule",
    # config / runner
    "BOCampaignConfig",
    "CampaignRunner",
    "CampaignResult",
    "build_campaign",
    "build_dataset",
    "run_campaign",
    # benchmark / persistence
    "BenchmarkResult",
    "run_grid",
    "record_campaign",
    # derived objectives (temperature-folded)
    "DerivedObjective",
    "DERIVED_OBJECTIVES",
    "build_derived_objective",
    # pareto / multi-objective utilities
    "pareto_mask",
    "pareto_indices",
]
