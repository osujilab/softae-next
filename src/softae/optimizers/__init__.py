"""Optimizer subsystem for autonomous experimentation.

Importing every backend here is what populates ``BaseOptimizer``'s name →
class registry, so a checkpoint resumes in a process that imported only
``softae.optimizers.base`` (which is all ``core/campaign_resume.py`` does).
Dropping one of these imports does not fail here — it fails at resume.
"""

from softae.optimizers.base import BaseOptimizer
from softae.optimizers.bayesian import BayesianOptimizer
from softae.optimizers.grid import GridSearchOptimizer
from softae.optimizers.pooled_bayesian import PooledBayesianOptimizer
from softae.optimizers.random import RandomSearchOptimizer
from softae.optimizers.replicated import ReplicatingOptimizer

__all__ = [
    "BaseOptimizer",
    "BayesianOptimizer",
    "GridSearchOptimizer",
    "PooledBayesianOptimizer",
    "RandomSearchOptimizer",
    "ReplicatingOptimizer",
]
