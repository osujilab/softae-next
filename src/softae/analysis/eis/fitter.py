"""Fitting that keeps the covariance ``impedance.py`` throws away.

``impedance.models.circuits.fitting.circuit_fit`` computes ``pcov`` from
``scipy.optimize.curve_fit`` and then returns only ``sqrt(diag(pcov))`` — every
off-diagonal term is discarded. That is fatal for R2, because the off-diagonal
``ρ(R_series, R_bulk)`` is *precisely* the quantity deciding whether the split may be
reported at all (framework §1.6).

The topology encodes the two resistances at two places in frequency: the low-frequency
resistive plateau is ``R_series + R_bulk``, while the high-frequency foot of the bulk
arc is ``R_series`` alone. The foot is only visible above ``f_c = 1/(2π·R_bulk·C_par)``,
and since ``R_bulk = K/σ`` that corner moves **up** with conductivity. Once it leaves
the band the split is unidentifiable and the optimiser trades resistance between the
two terms at near-zero cost — with ``ρ → −1``, large individual variances, and

    Var(R_series + R_bulk) ≪ Var(R_series) + Var(R_bulk)

**The sum is the robust observable and the correct thing to report.** Overhaul §3.1
records what reporting ``R_bulk`` alone did instead: it silently dropped a
σ-dependent fraction of the true resistance, which is the origin of the apparent
"non-constant cell constant" in the KCl campaign.

Rather than reimplement the circuit-string DSL, this module fits *through*
``impedance.py``'s own ``wrapCircuit`` and simply keeps the matrix. Same topologies,
same parameter ordering, same bounds machinery — one extra return value.

.. warning::
   **``x_scale="jac"`` is what makes this fit converge at all**, and its absence is the
   most consequential defect this module fixes. A circuit's parameters span some
   fourteen orders of magnitude — ``C_par`` near 3e-10 F beside ``R_bulk`` near 5e4 Ω —
   and ``curve_fit``'s default ``x_scale=1.0`` makes a unit step simultaneously
   absurd for one and invisible to the other. The optimiser then terminates at
   *iteration zero* and returns the initial guess unchanged.

   Measured on a synthetic spectrum whose true parameters are
   ``R_series=50, R_bulk=50000, n=0.8, C_par=3.5e-10``: unscaled, the fit returns the
   initial guess verbatim (``194``, ``51259``, ``0.8``, ``3.0e-10``) no matter how
   tight ``ftol`` is set. With ``x_scale="jac"`` it recovers every parameter exactly.

   ``impedance.py`` passes ``ftol=1e-13`` but never ``x_scale``, so the legacy path has
   been returning near-initial-guess parameters and calling them a fit. Because
   ``extract_features`` produces a decent guess, the result can still score a high R²
   — which is precisely what made this invisible.

.. note::
   ``weight_by_modulus`` defaults to **True** here and to ``False`` in
   ``impedance.py``. The legacy path never overrode it, so fits have been unweighted
   while :func:`softae.analysis.quality.compute_fit_quality` graded them *with*
   modulus weighting. Unweighted least squares over decades of ``|Z|`` is dominated by
   the low-frequency end and cannot see a small series term against a large tail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Iteration ceiling. Generous — a failure to converge should be reported as one,
#: not manufactured by a budget too small for a five-parameter fit.
DEFAULT_MAX_NFEV = 20_000

#: Convergence tolerance for ``ftol``/``xtol``/``gtol``.
#:
#: **1e-10, and loosening it was measured and rejected.** Recording the numbers so the
#: question is not reopened from first principles: it looks like an easy saving and is
#: not. Median over three seeds, per-parameter error against known truth
#: (``R_series = 50 Ω``, ``R_bulk = 50 kΩ``):
#:
#: ===== ======== ================ ==============
#: tol    t (0 %)  R_series err %   R_bulk err %
#: ===== ======== ================ ==============
#: 1e-6    0.53 s          13.321 %       0.0052 %
#: 1e-8    1.48 s           1.522 %       0.0024 %
#: 1e-9    2.43 s           0.406 %       0.0007 %
#: 1e-10   3.23 s           0.156 %       0.0003 %
#: 1e-11   3.12 s           0.156 %       0.0003 %
#: ===== ======== ================ ==============
#:
#: **The tolerance is load-bearing for the small parameter specifically.** Judged on
#: ``R_series + R_bulk`` the whole table looks flat — a 1.8 Ω error against 50 kΩ is
#: 0.004 % and invisible. Judged on ``R_series`` alone, 1e-8 is *ten times* worse than
#: 1e-10. Since recovering the series term the legacy path absorbs is the reason this
#: module exists (§3.1), that is the parameter the tolerance has to be chosen for.
#:
#: 1e-10 is the knee: 1e-11 costs the same and buys nothing.
#:
#: **On real spectra none of this applies, and that is the point of stating it.** At
#: 1 % noise every tolerance from 1e-6 to 1e-11 gives an identical fit in an identical
#: 0.33 s — the noise floor stops the optimiser long before the tolerance does, and
#: ``R_series`` is not individually recoverable at all (119 % error: the noise on a
#: 50 kΩ measurement is 500 Ω, ten times the term being sought). Which is precisely why
#: the ρ-driven sum-vs-split resolution exists.
#:
#: So the several-second fit belongs to *noise-free synthetic* spectra, which no
#: instrument produces. The cost falls on the test suite, and it buys the parameter
#: recovery that verifies the fitter — framework §8.2's acceptance criterion. It is not
#: an operational cost: a real 32-channel round pays about 0.3 s a spectrum.
DEFAULT_FIT_TOL = 1e-10


@dataclass(frozen=True)
class FitCovariance:
    """Fitted parameters with their full covariance matrix."""

    names: tuple[str, ...]
    values: np.ndarray
    pcov: np.ndarray
    singular: bool = False
    n_points: int = 0
    weight_by_modulus: bool = True
    bounds: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    # ── Parameter access by role-resolved name ───────────────────────────────

    def index(self, name: str) -> int | None:
        """Position of *name* in the parameter vector, or ``None``."""
        try:
            return self.names.index(name)
        except ValueError:
            return None

    def value(self, name: str) -> float:
        i = self.index(name)
        return float(self.values[i]) if i is not None else float("nan")

    def se(self, name: str) -> float:
        """Standard error of one parameter, NaN when the fit was singular."""
        i = self.index(name)
        if i is None or self.singular:
            return float("nan")
        var = float(self.pcov[i, i])
        return float(np.sqrt(var)) if var == var and var >= 0 else float("nan")

    def rel_se(self, name: str) -> float:
        """``SE(p)/|p|`` — a nuisance parameter may legitimately be poorly determined,
        but the measurand may not."""
        v, s = self.value(name), self.se(name)
        if not (v == v and s == s) or v == 0:
            return float("nan")
        return abs(s / v)

    # ── Identifiability ──────────────────────────────────────────────────────

    def rho(self, a: str, b: str) -> float:
        """``Σab / sqrt(Σaa·Σbb)`` — the correlation the framework gates on."""
        ia, ib = self.index(a), self.index(b)
        if ia is None or ib is None or self.singular:
            return float("nan")
        saa, sbb, sab = (float(self.pcov[ia, ia]), float(self.pcov[ib, ib]),
                         float(self.pcov[ia, ib]))
        denom = np.sqrt(saa * sbb)
        if not (denom == denom) or denom <= 0:
            return float("nan")
        return float(sab / denom)

    def sum_value(self, a: str, b: str) -> float:
        return self.value(a) + self.value(b)

    def sum_se(self, a: str, b: str) -> float:
        """``sqrt(Σaa + Σbb + 2Σab)``.

        The large negative cross-term collapses this precisely when the individual
        variances are largest — the sum is well determined exactly where the split
        is meaningless.
        """
        ia, ib = self.index(a), self.index(b)
        if ia is None or ib is None or self.singular:
            return float("nan")
        var = (float(self.pcov[ia, ia]) + float(self.pcov[ib, ib])
               + 2.0 * float(self.pcov[ia, ib]))
        return float(np.sqrt(var)) if var == var and var >= 0 else float("nan")

    def pegged(self, tol: float = 1e-3) -> tuple[str, ...]:
        """Parameters resting on a box constraint.

        A pegged parameter is unidentified: the data pushed it as far as the optimiser
        allowed, so its value is a property of the bound and its reported uncertainty
        is meaningless.
        """
        if self.bounds is None:
            return ()
        lo, hi = self.bounds
        out: list[str] = []
        for i, name in enumerate(self.names):
            if i >= len(lo) or i >= len(hi):
                break
            v, low, high = float(self.values[i]), float(lo[i]), float(hi[i])
            scale = max(abs(v), 1e-30)
            if np.isfinite(low) and abs(v - low) <= tol * scale:
                out.append(name)
            elif np.isfinite(high) and abs(v - high) <= tol * scale:
                out.append(name)
        return tuple(out)

    def describe(self) -> str:
        if self.singular:
            return f"{len(self.names)} parameters, covariance singular (unidentifiable)"
        parts = [f"{n}={self.value(n):.4g}±{self.se(n):.2g}" for n in self.names[:4]]
        return f"{len(self.names)} parameters over {self.n_points} pts: " + ", ".join(parts)


def fit_with_covariance(
    freq: np.ndarray,
    Z: np.ndarray,
    circuit: str,
    initial_guess: Sequence[float],
    *,
    constants: Mapping[str, float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    weight_by_modulus: bool = True,
    max_nfev: int = DEFAULT_MAX_NFEV,
    tol: float = DEFAULT_FIT_TOL,
) -> FitCovariance | None:
    """Fit *circuit* to ``(freq, Z)`` and keep ``pcov``.

    *Z* must be in the physics convention (``Im Z < 0`` for a capacitive response),
    matching what ``wrapCircuit`` produces.

    Returns ``None`` when the backend is unavailable or the optimiser genuinely failed
    — never raises. A singular Jacobian is *not* a failure: it returns a
    :class:`FitCovariance` with ``singular=True`` and NaN covariance, so degeneracy is
    reported as **unidentifiable** rather than as a fabricated ``ρ``, and one bad
    spectrum cannot end a 32-channel batch.
    """
    try:
        from impedance.models.circuits.fitting import (  # type: ignore
            set_default_bounds,
            wrapCircuit,
        )
        from scipy.optimize import curve_fit  # type: ignore
    except Exception:
        logger.warning("eis_fitter_backend_unavailable", exc_info=True)
        return None

    from softae.analysis.eis.models import parameter_names

    held = dict(constants or {})
    f = np.asarray(freq, dtype=float)
    Zc = np.asarray(Z, dtype=complex)

    good = np.isfinite(f) & np.isfinite(Zc) & (f > 0)
    f, Zc = f[good], Zc[good]
    if f.size < 2:
        return None

    if bounds is None:
        try:
            bounds = set_default_bounds(circuit, constants=held)
        except Exception:
            bounds = None
    lo = np.asarray(bounds[0], dtype=float) if bounds else None
    hi = np.asarray(bounds[1], dtype=float) if bounds else None

    ydata = np.hstack([Zc.real, Zc.imag])
    kwargs: dict[str, Any] = {}
    if bounds is not None:
        # The bounded (trf) path is preferred not only for the bounds but because it
        # is the only one that accepts ``x_scale``.  See below — that argument is
        # what makes this fit converge at all.
        # ``tol`` defaults to 1e-8; see DEFAULT_FIT_TOL for the measurement. Iteration
        # count dominates the cost here because ``wrapCircuit`` evaluates a built
        # circuit *string* on every residual call — 0.76 ms each, measured.
        kwargs.update(
            bounds=(lo, hi),
            x_scale="jac",
            max_nfev=int(max_nfev),
            ftol=float(tol), xtol=float(tol), gtol=float(tol),
        )
    else:
        kwargs["maxfev"] = int(max_nfev)

    if weight_by_modulus:
        mag = np.abs(Zc)
        # absolute_sigma=False scales pcov by the reduced chi-square, which is right
        # because the instrument's absolute noise is not known — only its shape.
        kwargs["sigma"] = np.hstack([mag, mag])
        kwargs["absolute_sigma"] = False

    try:
        popt, pcov = curve_fit(
            wrapCircuit(circuit, held), f, ydata,
            p0=np.asarray(initial_guess, dtype=float), **kwargs,
        )
    except Exception as exc:
        logger.info("eis_fit_failed", circuit=circuit, error=str(exc))
        return None

    pcov = np.asarray(pcov, dtype=float)
    singular = not np.all(np.isfinite(pcov))
    if singular:
        pcov = np.full_like(pcov, np.nan)
        logger.info(
            "eis_fit_singular", circuit=circuit,
            msg="Jacobian singular — parameters unidentifiable, reporting no covariance",
        )

    names = parameter_names(circuit, held)
    if len(names) != len(popt):
        names = tuple(f"p{i}" for i in range(len(popt)))

    return FitCovariance(
        names=names,
        values=np.asarray(popt, dtype=float),
        pcov=pcov,
        singular=singular,
        n_points=int(f.size),
        weight_by_modulus=bool(weight_by_modulus),
        bounds=(lo, hi) if bounds is not None else None,
    )


def fit_spectrum(eis_result: Any, model_name: str = "blocking_coplanar"):
    """Fit one spectrum and return a legacy-shaped ``FitResult`` carrying covariance.

    Produces the *same* :class:`~softae.analysis.circuit_fitting.FitResult` type the
    legacy path returns, so every existing consumer — the analysis tab, the browser,
    the web layer, ``record_fit`` — keeps working unchanged. What differs is that
    ``covariance`` is populated, ``R0``/``R1`` are resolved **by element name** rather
    than by the fragile positional ``z_indices``, and the fit is scaled and weighted.

    Never raises: a failure returns ``success=False`` with NaN resistances, matching
    :func:`softae.analysis.circuit_fitting.fit_circuit`'s contract.
    """
    from softae.analysis.circuit_fitting import FitResult, extract_features
    from softae.analysis.eis.models import EIS_CIRCUITS, roles_for

    model = EIS_CIRCUITS.get(model_name)
    if model is None:
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        legacy = CIRCUIT_MODELS.get(model_name)
        if legacy is None:
            raise ValueError(
                f"Unknown circuit model '{model_name}'. "
                f"Available: {sorted(set(EIS_CIRCUITS) | set(CIRCUIT_MODELS))}"
            )
        circuit, constants = legacy["circuit"], dict(legacy["constants"] or {})
        roles = roles_for(model_name)
    else:
        circuit, constants = model.circuit, dict(model.constants)
        roles = dict(model.roles)

    freq = np.asarray(eis_result.frequency, dtype=float)
    Z = np.asarray(eis_result.z_real, dtype=float) - 1j * np.asarray(
        eis_result.z_imag_neg, dtype=float)

    features = extract_features(freq, eis_result.z_real, eis_result.z_imag_neg)
    r0_guess, r1_guess = features["r0_guess"], features["r1_guess"]

    from softae.analysis.eis.models import parameter_names

    names = parameter_names(circuit, constants)
    seed = {"R0": max(r0_guess, 1e-6), "R1": max(r1_guess, 1e-6),
            "CPE0_0": 1e-7, "CPE0_1": 0.8, "C0": 3e-10, "L0": 1e-6}
    guess = [seed.get(n, 1.0) for n in names] or [r0_guess, 1e-7, 0.8, r1_guess, 3e-10]

    cov = fit_with_covariance(freq, Z, circuit, guess, constants=constants)

    if cov is None:
        return FitResult(
            model_name=model_name, parameters=np.full(len(guess), np.nan),
            R0=np.nan, R1=np.nan, R0_guess=r0_guess, R1_guess=r1_guess,
            z_indices=[0, 0], success=False,
            error_msg="covariance fit did not converge",
        )

    r_series = roles.get("R_series", "R0")
    r_bulk = roles.get("R_bulk", "R1")

    z_fit = None
    try:
        from impedance.models.circuits.fitting import wrapCircuit  # type: ignore

        stacked = wrapCircuit(circuit, constants)(freq, *cov.values)
        half = stacked.size // 2
        z_fit = stacked[:half] + 1j * stacked[half:]
    except Exception:
        z_fit = None

    quality: dict[str, float] = {}
    try:
        from softae.analysis.quality import compute_fit_quality

        quality = compute_fit_quality(eis_result, z_fit, n_params=len(cov.values))
    except Exception:
        quality = {}

    return FitResult(
        model_name=model_name,
        parameters=cov.values,
        R0=cov.value(r_series),
        R1=cov.value(r_bulk),
        R0_guess=r0_guess,
        R1_guess=r1_guess,
        z_indices=[cov.index(r_series) or 0, cov.index(r_bulk) or 0],
        z_fit=z_fit,
        quality=quality,
        covariance=cov,
    )
