"""Tests for the pluggable surrogate backends."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from softae.errors import CampaignError
from softae.optimizers.surrogates import (
    BACKENDS,
    SklearnGPBackend,
    make_backend,
)


def test_sklearn_backend_fit_predict_shapes():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(8, 2))
    y = X[:, 0] + 0.5 * X[:, 1]
    be = SklearnGPBackend(seed=0)
    be.fit(X, y, alpha=1e-6)
    mu, sigma = be.predict(X)
    assert mu.shape == (8,)
    assert sigma.shape == (8,)
    assert np.all(sigma >= 0)


def test_heteroscedastic_alpha_consumed():
    # Two coincident-y points with very different known noise: the noisy one
    # should retain larger posterior uncertainty.
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 0.0])
    be = SklearnGPBackend(seed=0)
    be.fit(X, y, alpha=np.array([1e-8, 1.0]))
    _, sigma = be.predict(X)
    assert sigma[1] > sigma[0]


def test_alpha_length_mismatch_raises():
    be = SklearnGPBackend()
    with pytest.raises(CampaignError, match="alpha length"):
        be.fit(np.zeros((3, 1)), np.zeros(3), alpha=np.ones(2))


def test_predict_before_fit_raises():
    be = SklearnGPBackend()
    with pytest.raises(CampaignError, match="before fit"):
        be.predict(np.zeros((1, 1)))


def test_make_backend_unknown_raises():
    with pytest.raises(CampaignError, match="unknown surrogate backend"):
        make_backend("does_not_exist")


def test_sklearn_in_registry_and_resolves():
    assert "sklearn" in BACKENDS
    assert isinstance(make_backend("sklearn"), SklearnGPBackend)


@pytest.mark.skipif(
    importlib.util.find_spec("botorch") is not None, reason="botorch installed"
)
def test_missing_optional_backend_gives_install_hint():
    with pytest.raises(CampaignError, match="pip install softae"):
        make_backend("botorch")


def test_ard_with_disparate_axis_ranges_is_not_flat():
    # Two axes with a ~175x range mismatch (like EO:Li vs silica vol-frac) and a
    # smooth target.  Standardization + ARD must keep per-axis structure rather
    # than collapsing the length scale and predicting a flat interior.
    a = np.linspace(5.0, 40.0, 8)       # wide-range axis
    b = np.linspace(0.0, 0.20, 8)       # narrow-range axis
    A, B = np.meshgrid(a, b)
    X = np.column_stack([A.ravel(), B.ravel()])
    y = np.sin(A.ravel() / 10.0) + 3.0 * B.ravel()  # varies along BOTH axes
    be = SklearnGPBackend(seed=0)
    be.fit(X, y, alpha=1e-6)
    # ARD → one length scale per (active) axis.
    assert np.ndim(be._gp.kernel_.k1.length_scale if hasattr(be._gp.kernel_, "k1")
                   else be._gp.kernel_.length_scale) == 1
    # Interior prediction along the narrow axis is not flat.
    base = np.array([[22.0, 0.10]])
    grid = np.tile(base, (15, 1))
    grid[:, 1] = np.linspace(0.0, 0.20, 15)
    mu, _ = be.predict(grid)
    assert float(mu.max() - mu.min()) > 0.1


def test_constant_axis_is_dropped():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])  # 2nd col constant
    y = np.array([1.0, 2.0, 3.0, 4.0])
    be = SklearnGPBackend(seed=0)
    be.fit(X, y, alpha=1e-6)
    assert be._active.tolist() == [True, False]   # constant axis masked out
    mu, sigma = be.predict(X)
    assert mu.shape == (4,) and sigma.shape == (4,)
