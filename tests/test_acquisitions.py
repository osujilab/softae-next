"""Tests for acquisition strategies (optimization + active-learning families)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from softae.campaigns.acquisitions import (
    ACQUISITIONS,
    EiAcquisition,
    IntegratedVarianceAcquisition,
    MaxVarianceAcquisition,
    UcbAcquisition,
    UncertaintyWeightedAcquisition,
    make_acquisition,
)
from softae.errors import CampaignError


def fake_opt(objective="maximize", kappa=2.0, candidate_variance=None, backend=None):
    return types.SimpleNamespace(
        _objective=objective,
        _kappa=kappa,
        candidate_variance=candidate_variance,
        backend=backend,
    )


def test_ucb_respects_direction():
    mu = np.array([0.0, 1.0, 2.0])
    sigma = np.zeros(3)
    cand = mu.reshape(-1, 1)
    s_max = UcbAcquisition().score(fake_opt("maximize"), cand, mu, sigma, [])
    s_min = UcbAcquisition().score(fake_opt("minimize"), cand, mu, sigma, [])
    assert int(np.argmax(s_max)) == 2  # highest mu
    assert int(np.argmax(s_min)) == 0  # lowest mu


def test_max_variance_selects_highest_sigma():
    mu = np.zeros(3)
    sigma = np.array([0.1, 0.9, 0.3])
    cand = np.arange(3).reshape(-1, 1)
    s = MaxVarianceAcquisition().score(fake_opt(), cand, mu, sigma, [])
    assert int(np.argmax(s)) == 1


def test_max_variance_uses_epistemic_sigma():
    # Equal total sigma, but candidate 0 has more aleatoric (known) noise, so it
    # has less *reducible* uncertainty and should not be preferred.
    mu = np.zeros(2)
    sigma = np.array([1.0, 1.0])
    cand = np.arange(2).reshape(-1, 1)
    opt = fake_opt(candidate_variance=np.array([0.9, 0.1]))
    s = MaxVarianceAcquisition().score(opt, cand, mu, sigma, [])
    assert s[1] > s[0]


def test_ei_zero_without_history_uses_sigma():
    mu = np.zeros(3)
    sigma = np.array([0.1, 0.5, 0.2])
    cand = np.arange(3).reshape(-1, 1)
    s = EiAcquisition().score(fake_opt(), cand, mu, sigma, [])
    assert int(np.argmax(s)) == 1  # falls back to exploration


def test_integrated_variance_falls_back_without_kernel():
    mu = np.zeros(3)
    sigma = np.array([0.1, 0.9, 0.3])
    cand = np.arange(3).reshape(-1, 1)
    # backend=None → no cross_cov → should behave like max_variance.
    s = IntegratedVarianceAcquisition().score(fake_opt(backend=None), cand, mu, sigma, [])
    assert int(np.argmax(s)) == 1


def test_uncertainty_weighting_downweights_noisy_points():
    mu = np.zeros(2)
    sigma = np.array([1.0, 1.0])
    cand = np.arange(2).reshape(-1, 1)
    opt = fake_opt(candidate_variance=np.array([0.01, 1.0]))
    s = UncertaintyWeightedAcquisition().score(opt, cand, mu, sigma, [])
    assert s[0] > s[1]  # the low-noise point is preferred


def test_registry_maps_strings():
    assert isinstance(make_acquisition("ucb"), UcbAcquisition)
    assert isinstance(make_acquisition("max_variance"), MaxVarianceAcquisition)
    assert set(ACQUISITIONS) >= {
        "ucb", "ei", "max_variance", "integrated_variance", "uncertainty_weighted"
    }
    with pytest.raises(CampaignError, match="unknown acquisition"):
        make_acquisition("nope")
