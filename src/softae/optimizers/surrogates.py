"""Pluggable surrogate-model backends for pool-based Bayesian optimization.

Decouples "fit a GP, give me posterior mean/std (and optionally cross-covariance)"
from any specific library, so :class:`~softae.optimizers.pooled_bayesian.PooledBayesianOptimizer`
and the acquisition layer depend only on ``(mu, sigma)`` — never on sklearn,
BoTorch, GPyTorch, or GPCAM directly.

* :class:`SklearnGPBackend` — default, no new dependencies.  Uses
  ``GaussianProcessRegressor``.  A per-point ``alpha`` array makes it a
  fixed-noise / heteroscedastic GP (``Matern`` kernel, noise in ``alpha``);
  ``alpha=None`` falls back to ``Matern + WhiteKernel`` (homoscedastic, learned),
  matching the existing :class:`~softae.optimizers.bayesian.BayesianOptimizer`.
* :class:`BoTorchBackend`, :class:`GPyTorchBackend`, :class:`GPCAMBackend` —
  lazy-imported optional extras; raise a clear install hint if the package is
  absent.  Stubbed in P0/P1, signatures fixed now.

Use :func:`make_backend` to resolve a string or instance to a backend.
"""

from __future__ import annotations

import abc
import importlib.util

import numpy as np

from softae.errors import CampaignError


class SurrogateBackend(abc.ABC):
    """Abstract GP surrogate: ``fit`` then ``predict`` posterior mean/std."""

    #: pip distribution names this backend needs (checked by :meth:`ensure_available`).
    requires: tuple[str, ...] = ()
    #: True if :meth:`cross_cov` returns a real cross-covariance (for ALC acquisition).
    supports_kernel: bool = False
    #: Registry key.
    name: str = "surrogate"

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray | float | None) -> None:
        """Fit to observations ``X`` (n×d), ``y`` (n,).

        ``alpha`` is a per-point observation-noise **variance** array (n,), a
        scalar, or ``None`` (let the backend learn a single homoscedastic level).
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior ``(mu, sigma)`` at ``X`` (sigma = std-dev, ≥ 0)."""

    def cross_cov(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray | None:
        """Posterior/prior cross-covariance ``k(Xa, Xb)`` or ``None`` if unsupported."""
        return None

    @classmethod
    def ensure_available(cls) -> None:
        """Raise a clear ``CampaignError`` if a required package is missing."""
        for pkg in cls.requires:
            if importlib.util.find_spec(pkg) is None:
                extra = f"bo-{cls.name}"
                raise CampaignError(
                    f"surrogate backend '{cls.name}' needs '{pkg}'. "
                    f"Install it with: pip install softae[{extra}]"
                )


class SklearnGPBackend(SurrogateBackend):
    """scikit-learn ``GaussianProcessRegressor`` surrogate (default).

    Inputs are **standardized per axis** before fitting, and the Matérn kernel is
    **ARD** (one length scale per dimension) with a bounded length scale.  This
    matters for materials spaces where axes have very different ranges (e.g.
    EO:Li ~ 5–40 vs silica vol-frac ~ 0–0.2): without scaling, a single isotropic
    length scale is dominated by the widest axis and — with near-zero noise — the
    marginal-likelihood optimum collapses the length scale to its lower bound,
    interpolating each training point as a spike and reverting to the mean
    everywhere else (a dead-flat interior).  Standardization + ARD + a sane lower
    bound prevent that.
    """

    requires = ()
    supports_kernel = True
    name = "sklearn"

    def __init__(
        self,
        *,
        nu: float = 2.5,
        n_restarts_optimizer: int = 10,
        normalize_y: bool = True,
        seed: int | None = None,
        white_noise_level: float = 1e-2,
        ard: bool = True,
        length_scale_bounds: tuple[float, float] = (1e-1, 1e2),
    ) -> None:
        self.nu = nu
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.seed = seed
        self.white_noise_level = white_noise_level
        self.ard = ard
        self.length_scale_bounds = length_scale_bounds
        self._gp = None
        self._kernel = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._active: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        Xa = np.asarray(X, dtype=float)[:, self._active]
        return (Xa - self._x_mean) / self._x_std

    def _build(self, alpha: np.ndarray | float | None, d: int):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel

        length_scale = [1.0] * d if self.ard else 1.0
        matern = Matern(
            length_scale=length_scale,
            length_scale_bounds=self.length_scale_bounds,
            nu=self.nu,
        )
        if alpha is None:
            # Homoscedastic: let a WhiteKernel learn the noise level.
            kernel = matern + WhiteKernel(
                noise_level=self.white_noise_level, noise_level_bounds=(1e-6, 1.0)
            )
            gp_alpha = 1e-10  # tiny numerical nugget
        else:
            # Heteroscedastic / fixed-noise: noise lives entirely in alpha.
            kernel = matern
            gp_alpha = alpha
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=gp_alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray | float | None) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Drop zero-variance axes (constant features carry no signal and would
        # give the GP an unidentifiable length scale), then standardize the rest.
        active = X.std(axis=0) > 1e-12
        if not active.any():
            active = np.ones(X.shape[1], dtype=bool)  # degenerate: keep all
        self._active = active
        Xa = X[:, active]
        self._x_mean = Xa.mean(axis=0)
        std = Xa.std(axis=0)
        std[std < 1e-12] = 1.0  # guard constant columns (e.g. single observation)
        self._x_std = std
        Xs = self._standardize(X)

        if alpha is not None and not np.isscalar(alpha):
            alpha = np.asarray(alpha, dtype=float)
            if alpha.shape[0] != X.shape[0]:
                raise CampaignError(
                    f"alpha length {alpha.shape[0]} != number of observations {X.shape[0]}"
                )
        self._gp = self._build(alpha, Xs.shape[1])
        self._gp.fit(Xs, y)
        self._kernel = self._gp.kernel_

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._gp is None:
            raise CampaignError("backend.predict called before fit")
        mu, sigma = self._gp.predict(self._standardize(X), return_std=True)
        return mu, sigma

    def cross_cov(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray | None:
        if self._kernel is None:
            return None
        return self._kernel(self._standardize(Xa), self._standardize(Xb))


class BoTorchBackend(SurrogateBackend):
    """BoTorch ``SingleTaskGP`` surrogate (optional; ``pip install softae[bo-botorch]``).

    Heteroscedastic noise is supplied via ``train_Yvar`` (a fixed-noise GP);
    ``alpha=None`` infers a homoscedastic level.  Inputs are normalised to the
    unit cube and outcomes standardised, per BoTorch convention.  Predictions
    return the **latent** posterior std (no observation noise), matching
    :class:`SklearnGPBackend`.

    Untested in this environment (package absent); exercised by ``skipif`` tests
    when BoTorch is installed.
    """

    requires = ("botorch", "torch")
    supports_kernel = False
    name = "botorch"

    def __init__(self, *, fit_restarts: int = 1) -> None:
        self.ensure_available()
        self.fit_restarts = fit_restarts
        self._model = None

    def fit(self, X, y, alpha):  # pragma: no cover - requires botorch
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        Xt = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
        Yt = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.double).unsqueeze(-1)
        d = Xt.shape[-1]
        kw = dict(input_transform=Normalize(d=d), outcome_transform=Standardize(m=1))
        if alpha is None:
            model = SingleTaskGP(Xt, Yt, **kw)
        else:
            if np.isscalar(alpha):
                Yvar = torch.full_like(Yt, float(alpha))
            else:
                Yvar = torch.as_tensor(
                    np.asarray(alpha, dtype=float), dtype=torch.double
                ).unsqueeze(-1)
            model = SingleTaskGP(Xt, Yt, train_Yvar=Yvar, **kw)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self._model = model

    def predict(self, X):  # pragma: no cover - requires botorch
        import torch

        if self._model is None:
            raise CampaignError("backend.predict called before fit")
        self._model.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
        with torch.no_grad():
            post = self._model.posterior(Xt)  # latent posterior (no obs noise)
            mu = post.mean.squeeze(-1).cpu().numpy()
            var = post.variance.squeeze(-1).clamp_min(0.0).cpu().numpy()
        return mu, np.sqrt(var)


class GPyTorchBackend(SurrogateBackend):
    """Minimal GPyTorch ExactGP surrogate (optional; ``pip install softae[bo-gpytorch]``).

    RBF + ScaleKernel with a constant mean, standardised targets, trained by Adam.
    Heteroscedastic noise uses ``FixedNoiseGaussianLikelihood``; ``alpha=None`` uses
    a learned ``GaussianLikelihood``.  Returns latent posterior std.

    Untested in this environment; exercised by ``skipif`` tests when installed.
    """

    requires = ("gpytorch", "torch")
    supports_kernel = False
    name = "gpytorch"

    def __init__(self, *, train_iters: int = 100, lr: float = 0.1) -> None:
        self.ensure_available()
        self.train_iters = train_iters
        self.lr = lr
        self._model = None

    def fit(self, X, y, alpha):  # pragma: no cover - requires gpytorch
        import gpytorch
        import torch

        Xt = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
        yt = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.double)
        self._y_mean = yt.mean()
        self._y_std = yt.std().clamp_min(1e-9)
        yt_n = (yt - self._y_mean) / self._y_std

        if alpha is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        else:
            noise = (
                torch.full_like(yt, float(alpha))
                if np.isscalar(alpha)
                else torch.as_tensor(np.asarray(alpha, dtype=float), dtype=torch.double)
            )
            noise_n = (noise / (self._y_std ** 2)).clamp_min(1e-9)
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noise_n, learn_additional_noise=False
            )

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, lik):
                super().__init__(tx, ty, lik)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        model = _ExactGP(Xt, yt_n, likelihood).double()
        likelihood.double()
        model.train()
        likelihood.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(self.train_iters):
            opt.zero_grad()
            loss = -mll(model(Xt), yt_n)
            loss.backward()
            opt.step()
        self._model = model
        self._likelihood = likelihood

    def predict(self, X):  # pragma: no cover - requires gpytorch
        import gpytorch
        import torch

        if self._model is None:
            raise CampaignError("backend.predict called before fit")
        self._model.eval()
        self._likelihood.eval()
        Xt = torch.as_tensor(np.asarray(X, dtype=float), dtype=torch.double)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self._model(Xt)  # latent function posterior
            mu = (post.mean * self._y_std + self._y_mean).cpu().numpy()
            sigma = (post.variance.clamp_min(0.0).sqrt() * self._y_std).cpu().numpy()
        return mu, sigma


class GPCAMBackend(SurrogateBackend):
    """gpCAM / fvGP surrogate (optional; ``pip install softae[bo-gpcam]``).

    Best-effort adapter against the ``fvgp.GP`` API (gpCAM ≥ 8 builds on fvGP),
    which accepts per-point ``noise_variances`` — a natural fit for the
    heteroscedastic ``alpha`` channel.  fvGP's API varies across versions; if it
    differs from this adapter, a clear :class:`CampaignError` is raised so the
    adapter can be updated.  Untested in this environment.
    """

    requires = ("fvgp",)
    supports_kernel = False
    name = "gpcam"

    def __init__(self) -> None:
        self.ensure_available()
        self._gp = None

    def fit(self, X, y, alpha):  # pragma: no cover - requires gpcam/fvgp
        try:
            from fvgp import GP
        except Exception as exc:
            raise CampaignError(f"failed to import fvgp.GP: {exc}") from exc

        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        noise = None
        if alpha is not None:
            noise = (
                np.full_like(ya, float(alpha))
                if np.isscalar(alpha)
                else np.asarray(alpha, dtype=float)
            )
        try:
            self._gp = GP(Xa, ya, noise_variances=noise)
            self._gp.train()
        except Exception as exc:
            raise CampaignError(
                f"gpcam/fvgp adapter failed ({exc}); the fvGP API may differ from "
                f"this version — update GPCAMBackend in optimizers/surrogates.py."
            ) from exc

    def predict(self, X):  # pragma: no cover - requires gpcam/fvgp
        if self._gp is None:
            raise CampaignError("backend.predict called before fit")
        Xa = np.asarray(X, dtype=float)
        try:
            mu = np.asarray(self._gp.posterior_mean(Xa)["f(x)"], dtype=float)
            var = np.asarray(self._gp.posterior_covariance(Xa)["v(x)"], dtype=float)
        except Exception as exc:
            raise CampaignError(
                f"gpcam/fvgp prediction failed ({exc}); update GPCAMBackend."
            ) from exc
        return mu, np.sqrt(np.clip(var, 0.0, None))


#: Registry mapping config strings → backend classes.
BACKENDS: dict[str, type[SurrogateBackend]] = {
    SklearnGPBackend.name: SklearnGPBackend,
    BoTorchBackend.name: BoTorchBackend,
    GPyTorchBackend.name: GPyTorchBackend,
    GPCAMBackend.name: GPCAMBackend,
}


def make_backend(
    backend: str | SurrogateBackend, *, seed: int | None = None
) -> SurrogateBackend:
    """Resolve a backend name or instance to a :class:`SurrogateBackend`."""
    if isinstance(backend, SurrogateBackend):
        return backend
    if backend not in BACKENDS:
        raise CampaignError(
            f"unknown surrogate backend '{backend}'; available: {sorted(BACKENDS)}"
        )
    cls = BACKENDS[backend]
    cls.ensure_available()
    if cls is SklearnGPBackend:
        return SklearnGPBackend(seed=seed)
    return cls()  # lazy backends raise their own clear error
