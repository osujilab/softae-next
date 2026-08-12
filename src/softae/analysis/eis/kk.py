"""Kramers–Kronig stationarity: which points no causal linear system could produce.

Framework §3.6 / §1.5. Fit a K–K-compliant basis — a log-spaced Voigt (RC) ladder plus
series ``R``, ``L`` and optionally ``C`` — by weighted least squares, and read the
residuals. A point the *linear, causal, stationary* basis cannot represent is not
describable by the physical model either, whatever that model turns out to be. That is
what makes this test model-free: it never assumes the circuit, only causality.

**Directionality is the whole policy, and it is asymmetric on purpose.** Only the
contiguous failing run at the **low-frequency end** is truncated. Two facts make that
free: acquisition there is slow enough for the sample itself to drift during the sweep
(§1.9), and ``R_bulk`` does not live there (§1.8). An isolated mid-band failure means a
noisy point or an outlier — not drift — and removing it would be quietly discarding data
on a criterion that does not apply to it. Those are flagged and kept.

.. note::
   **``add_cap`` follows the cell, and it matters more here than anywhere else.**
   A blocking electrode has a capacitive low-frequency tail. A pure Voigt ladder cannot
   represent one — its residuals blow up exactly where the tail dominates, which is
   exactly the low-frequency end this module is empowered to truncate. Running without
   the series capacitance on a blocking cell would therefore manufacture the failure it
   then acts on, and would do so most aggressively on the best-behaved blocking spectra.
   :data:`softae.analysis.eis.policy` carries ``blocking`` from ``[eis.cell]``; this
   module consumes it rather than guessing.

The ladder itself comes from ``impedance.validation`` (``linKK``, ``eval_linKK``,
``residuals_linKK``, ``calc_mu``), so there is no in-house Voigt implementation to keep
correct — including the μ-criterion order selection §3.6 asks for.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def _patch_impedance_eval_namespace() -> bool:
    """Make ``impedance`` 1.7.1's ``eval_linKK`` work under NumPy 2. Idempotent.

    ``eval_linKK`` builds a circuit string by formatting the fitted element values
    into it, then ``eval``s that string against ``circuit_elements`` as globals.
    NumPy 2 changed scalar ``repr`` from ``5.0`` to ``np.float64(5.0)``, so the
    generated source now references a name that namespace does not contain, and
    **every** call raises ``NameError: name 'np' is not defined``.

    Left unpatched this is worse than a crash, because :func:`lin_kk` is required not
    to raise: every spectrum would report "K–K ladder did not converge", the gate
    would degrade to a flag, and the pipeline would look like it was running a
    stationarity test it had in fact never once executed.

    Adding ``np`` to that namespace is the whole fix — it changes no arithmetic, only
    what the evaluated literal can resolve.
    """
    try:
        from impedance import validation

        ns = getattr(validation, "circuit_elements", None)
        if isinstance(ns, dict) and "np" not in ns:
            ns["np"] = np
            logger.debug("impedance_linkk_namespace_patched",
                         numpy=np.__version__,
                         msg="added np for NumPy-2 scalar reprs in eval_linKK")
        return isinstance(ns, dict)
    except Exception:                                          # pragma: no cover
        return False

#: μ-criterion target for ladder order selection. ``impedance.py``'s default, and the
#: value the Schönleber criterion was published with: below it the ladder starts
#: fitting noise, above it it is under-flexible.
DEFAULT_KK_C = 0.85

#: Ceiling on ladder order. Not a tuning knob so much as a runtime bound — the μ search
#: walks orders upward and stops on the criterion, so this only bites on pathological
#: spectra where it never triggers.
DEFAULT_KK_MAX_M = 50

#: Largest fraction of a spectrum the low-frequency truncation may remove before it
#: stops being a truncation.
#:
#: **This bound is not conservatism, it is a correctness requirement**, and it exists
#: because the ladder fit is *global*. Perturbing the five lowest-frequency points of an
#: otherwise clean synthetic spectrum drives 40 of 41 points past a 1 % residual — the
#: whole fit shifts to accommodate the drift, so almost every point "fails". The
#: contiguous low-frequency run is then nearly the entire sweep, and unbounded
#: truncation would delete the spectrum while reporting that it had tidied a tail.
#:
#: §3.6 licenses truncation on the grounds that ``R_bulk`` does not live at low
#: frequency. Once the failing run reaches that far up the band, the premise is gone:
#: this is not drift at the tail, it is a spectrum that is non-stationary throughout,
#: and the honest verdict is rejection rather than surgery.
DEFAULT_KK_MAX_TRUNCATE_FRAC = 0.5


@dataclass
class LinKKResult:
    """Per-point K–K residuals and what the ladder needed to achieve them."""

    ok: bool = False
    M: int = 0
    mu: float = float("nan")
    #: Per-point residual magnitude as a percentage of ``|Z|``.
    resid_pct: np.ndarray = field(default_factory=lambda: np.empty(0))
    res_real: np.ndarray = field(default_factory=lambda: np.empty(0))
    res_imag: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: Why the ladder could not be fitted, when ``ok`` is False.
    error: str = ""

    @property
    def max_resid_pct(self) -> float:
        finite = self.resid_pct[np.isfinite(self.resid_pct)]
        return float(np.max(finite)) if finite.size else float("nan")


def lin_kk(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    blocking: bool = True,
    c: float = DEFAULT_KK_C,
    max_M: int = DEFAULT_KK_MAX_M,
) -> LinKKResult:
    """Fit the K–K basis and return per-point residuals.

    Never raises. A ladder that will not fit is reported as ``ok=False`` with a reason:
    one unfittable spectrum must not end a 32-channel batch, and a K–K test that cannot
    run is an absence of evidence, not evidence of failure.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    n = int(min(freq.size, Zc.size))
    if n < 4:
        return LinKKResult(error=f"only {n} points — too few for a K–K ladder")

    if not _patch_impedance_eval_namespace():                  # pragma: no cover
        return LinKKResult(error="impedance.validation unavailable")

    try:
        from impedance.validation import linKK
    except Exception as exc:                                  # pragma: no cover
        return LinKKResult(error=f"impedance.validation unavailable: {exc}")

    # linKK expects ascending frequency; the rig sweeps high→low.
    order = np.argsort(freq)
    try:
        # linKK prints its ladder order and RMSE to stdout every tenth iteration.
        # Harmless for one spectrum at a REPL, noise across 32 channels a round.
        with contextlib.redirect_stdout(io.StringIO()):
            M, mu, _Z_fit, res_re, res_im = linKK(
                freq[order], Zc[order], c=float(c), max_M=int(max_M),
                fit_type="complex", add_cap=bool(blocking),
            )
    except Exception as exc:
        logger.info("eis_linkk_failed", error=str(exc), n_points=n)
        return LinKKResult(error=f"K–K ladder did not converge: {exc}")

    res_re = np.asarray(res_re, dtype=float)
    res_im = np.asarray(res_im, dtype=float)
    # residuals_linKK returns them already normalised by |Z|; combine the two
    # components into one per-point magnitude so a single threshold governs both.
    resid = np.hypot(res_re, res_im) * 100.0

    # Undo the sort so every array indexes the caller's own points.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return LinKKResult(
        ok=True, M=int(M), mu=float(mu),
        resid_pct=resid[inverse],
        res_real=res_re[inverse], res_imag=res_im[inverse],
    )


def low_frequency_run(f: np.ndarray, failing: np.ndarray) -> np.ndarray:
    """The contiguous failing run at the low-frequency end, as a boolean mask.

    Returns which points §3.6 licenses *removing*. Everything else that failed is an
    isolated failure and must be flagged rather than dropped.

    Frequency order is not assumed. The rig sweeps high→low, so "the low-frequency end"
    is the tail of the array rather than its head — and a helper that quietly assumed
    ascending order would truncate the *high*-frequency end, silently deleting the arc
    that carries ``R_bulk`` while reporting that it had removed drift.
    """
    freq = np.asarray(f, dtype=float)
    bad = np.asarray(failing, dtype=bool)
    out = np.zeros(bad.shape, dtype=bool)
    if not bad.any():
        return out

    ascending = np.argsort(freq)          # lowest frequency first
    for idx in ascending:
        if not bad[idx]:
            break                          # the run has ended
        out[idx] = True
    return out
