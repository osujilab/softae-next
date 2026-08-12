"""Tests for the pluggable observation-noise models."""

from __future__ import annotations

import math

import pytest

from softae.campaigns.noise import (
    CellStats,
    CompositeNoise,
    FitQualityNoise,
    HomoscedasticNoise,
    ReplicateNoise,
    build_noise_model,
)
from softae.campaigns.objectives import Log10Sigma
from softae.errors import CampaignError

_LN10 = math.log(10.0)
T = Log10Sigma()


def cell(sigma_mean, sigma_std, n_rep, *, fitted_Z=float("nan"), adjusted_Z=float("nan"), pid="p"):
    return CellStats(
        point_id=pid,
        params={"x": 1.0},
        sigma_values=[sigma_mean] * max(n_rep, 1),
        sigma_mean=sigma_mean,
        sigma_std=sigma_std,
        n_rep=n_rep,
        fitted_Z=fitted_Z,
        adjusted_Z=adjusted_Z,
        y=T.apply_scalar(sigma_mean),
    )


def test_replicate_noise_log_space_delta_method():
    c = cell(1e-4, 1e-5, 2)
    nm = ReplicateNoise(shrinkage="none", target_is_mean=False, var_floor=1e-12)
    nm.prepare([c], T)
    expected = (1e-5 / (1e-4 * _LN10)) ** 2 + 1e-12
    assert nm.variance(c, T) == pytest.approx(expected, rel=1e-6)


def test_sem_vs_single_draw_factor_of_n():
    c = cell(1e-4, 1e-5, 4)
    single = ReplicateNoise(shrinkage="none", target_is_mean=False, var_floor=1e-15)
    mean = ReplicateNoise(shrinkage="none", target_is_mean=True, var_floor=1e-15)
    single.prepare([c], T)
    mean.prepare([c], T)
    # SEM² is the single-draw variance divided by n_rep.
    assert mean.variance(c, T) == pytest.approx(single.variance(c, T) / 4, rel=1e-6)


def test_replicate_shrinkage_pulls_toward_pooled():
    quiet = cell(1e-4, 1e-6, 2, pid="quiet")
    loud = cell(1e-4, 1e-3, 2, pid="loud")
    none = ReplicateNoise(shrinkage="none", lam=0.5, var_floor=1e-15)
    pooled = ReplicateNoise(shrinkage="pooled", lam=0.5, var_floor=1e-15)
    none.prepare([quiet, loud], T)
    pooled.prepare([quiet, loud], T)
    # Shrinkage moves the loud cell's huge variance down toward the pool mean.
    assert pooled.variance(loud, T) < none.variance(loud, T)
    # ...and raises the quiet cell's tiny variance.
    assert pooled.variance(quiet, T) > none.variance(quiet, T)


def test_single_replicate_falls_back_to_floor_or_pooled():
    c = cell(1e-4, float("nan"), 1)  # std undefined
    none = ReplicateNoise(shrinkage="none", var_floor=1e-6)
    none.prepare([c], T)
    assert none.variance(c, T) == pytest.approx(1e-6)


def test_fit_quality_noise_rises_with_discrepancy():
    good = cell(1e-4, 1e-6, 2, fitted_Z=1000.0, adjusted_Z=1000.0)
    bad = cell(1e-4, 1e-6, 2, fitted_Z=1500.0, adjusted_Z=1000.0)
    nm = FitQualityNoise(var_floor=1e-15)
    nm.prepare([good, bad], T)
    assert nm.variance(bad, T) > nm.variance(good, T)


def test_composite_sum_vs_max():
    c = cell(1e-4, 1e-5, 2, fitted_Z=1500.0, adjusted_Z=1000.0)
    sources = [ReplicateNoise(var_floor=1e-15), FitQualityNoise(var_floor=1e-15)]
    csum = CompositeNoise(sources, combine="sum", var_floor=1e-15)
    cmax = CompositeNoise(sources, combine="max", var_floor=1e-15)
    csum.prepare([c], T)
    cmax.prepare([c], T)
    assert csum.variance(c, T) >= cmax.variance(c, T)


def test_homoscedastic_none_sets_learn_noise():
    assert HomoscedasticNoise(level=None).learn_noise is True
    assert HomoscedasticNoise(level=0.01).learn_noise is False


def test_composite_rejects_learn_noise_source():
    with pytest.raises(CampaignError, match="learn-noise"):
        CompositeNoise([HomoscedasticNoise(level=None)])


def test_build_noise_model_registry():
    nm = build_noise_model("composite", sources=["replicate", "fit_quality"])
    assert isinstance(nm, CompositeNoise)
    with pytest.raises(CampaignError, match="unknown noise"):
        build_noise_model("does_not_exist")
