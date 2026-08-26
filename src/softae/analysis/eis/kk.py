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
correct. **Order selection is ours**, and it has to be: see :func:`lin_kk`.
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

#: μ ceiling for ladder order selection: an order flexible enough to have this much
#: negative-resistor mass is not under-fitting.
#:
#: Schönleber published 0.85 and ``impedance.py`` defaults to it. **This rig ships
#: 0.30**, because 0.85 admits M ≈ 5 on a 53-point sweep and a five-element ladder
#: leaves a ~2.5 % *systematic* residual with a low-frequency lobe on a spectrum that
#: is noise-free and exactly causal by construction — a run
#: :func:`~softae.analysis.eis.gates.gate_kk_truncation` then truncates, having
#: manufactured it. Measured, noise-free control: at M = 5, 33 of 53 points exceed
#: 1 %; at the order this module now selects, the median residual is 0.0002 %.
DEFAULT_KK_C = 0.30

#: Floor on μ, and the conditioning half of order selection.
#:
#: μ = 1 − Σ|R_k<0| / Σ|R_k≥0|, so μ → 0 means the ladder's positive and negative
#: resistor mass have grown equal and opposite: the fit is a cancellation, not a
#: description. ``fit_type="complex"`` inverts the normal matrix directly rather than
#: through a pseudo-inverse, so that regime is where it loses its conditioning.
#:
#: **This is the guard that stops "minimise the residual" from becoming "interpolate
#: the data".** Measured on a 27-point scout spectrum: orders with μ < 0.05 begin at
#: M = 28 — one more time constant than there are frequencies — and unconstrained the
#: lowest median residual of the whole walk sits at **M = 33**, an order the data
#: cannot resolve. With the floor the walk selects M = 23. On the 53-point noise-free
#: control μ first drops under 0.05 at M = 41, which is exactly where the median
#: residual leaves its 0.001–0.3 % plateau and climbs to 434 % by M = 49.
DEFAULT_KK_MU_FLOOR = 0.05

#: Ceiling on ladder order. A runtime bound, and only that: conditioning is enforced by
#: :data:`DEFAULT_KK_MU_FLOOR`, not by this. It was never a safe stopping point on its
#: own — the noise-free control fitted at a forced M = 49 returns a **434 % median /
#: 14 109 % max** residual, so a policy that walked up to it would report catastrophe
#: as measurement.
DEFAULT_KK_MAX_M = 50

#: Consecutive orders below :data:`DEFAULT_KK_MU_FLOOR` that end the walk.
#:
#: Not 1, because μ is not monotone: on a 0.5 %-noise mock it dips to 0.038 at M = 31
#: and returns to 0.055 at M = 32. Measured, it dips once and never twice, so three in
#: a row is the ladder having entered the cancelling regime for good — at which point
#: every remaining order is both unusable and expensive, since those are the near-
#: singular inversions.
_MU_FLOOR_PATIENCE = 3

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


@dataclass(frozen=True)
class _Order:
    """One rung of the ladder walk, kept so the walk can compare rungs."""

    M: int
    mu: float
    median_resid_pct: float
    resid_pct: np.ndarray
    res_real: np.ndarray
    res_imag: np.ndarray


def _fit_at_order(f_asc: np.ndarray, Z_asc: np.ndarray, M: int, *,
                  add_cap: bool) -> _Order | None:
    """Fit exactly *M* Voigt elements. ``None`` when the fit yields no finite residual.

    ``c=None`` is ``linKK``'s documented manual mode: it skips the μ walk entirely and
    solves at ``max_M``, which is what lets the order be chosen here instead of there.
    """
    from impedance.validation import linKK

    _M, mu, _Z_fit, res_re, res_im = linKK(
        f_asc, Z_asc, c=None, max_M=int(M), fit_type="complex", add_cap=add_cap,
    )
    res_re = np.asarray(res_re, dtype=float)
    res_im = np.asarray(res_im, dtype=float)
    # residuals_linKK returns them already normalised by |Z|; combine the two
    # components into one per-point magnitude so a single threshold governs both.
    resid = np.hypot(res_re, res_im) * 100.0
    finite = resid[np.isfinite(resid)]
    if finite.size == 0:
        return None
    return _Order(int(M), float(mu), float(np.median(finite)), resid, res_re, res_im)


def _walk_orders(f_asc: np.ndarray, Z_asc: np.ndarray, *, add_cap: bool,
                 c: float, max_M: int, mu_floor: float) -> tuple[_Order | None,
                                                                 _Order | None]:
    """Walk M upward and return ``(best_under_c, best_conditioned)``.

    Both are the minimum-median-residual rung of their set, and both are ``None`` when
    no order fitted at all. The second exists so a spectrum whose μ never reaches *c*
    still gets a K–K test rather than no verdict.
    """
    best_under_c: _Order | None = None
    best_conditioned: _Order | None = None
    consecutive_below_floor = 0

    for M in range(1, int(max_M) + 1):
        try:
            rung = _fit_at_order(f_asc, Z_asc, M, add_cap=add_cap)
        except Exception:
            continue
        if rung is None:
            continue
        # ``calc_mu`` divides by the positive-resistor mass, so μ comes back non-finite
        # when a ladder puts none there — nan for 0/0, −inf when the mass is all
        # negative. Neither comparison below is written defensively around that, and
        # the asymmetry is deliberate: −inf *is* the degenerate all-negative fit and
        # should fail the floor, while nan is an absence of evidence and should not.
        # A nan rung can never satisfy ``μ ≤ c`` either, so it survives only as the
        # fallback — which is the right standing for an order we know nothing about.
        if rung.mu < mu_floor:
            consecutive_below_floor += 1
            if consecutive_below_floor >= _MU_FLOOR_PATIENCE:
                break
            continue
        consecutive_below_floor = 0
        if (best_conditioned is None
                or rung.median_resid_pct < best_conditioned.median_resid_pct):
            best_conditioned = rung
        if rung.mu <= c and (best_under_c is None
                             or rung.median_resid_pct < best_under_c.median_resid_pct):
            best_under_c = rung
    return best_under_c, best_conditioned


def lin_kk(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    blocking: bool = True,
    c: float = DEFAULT_KK_C,
    max_M: int = DEFAULT_KK_MAX_M,
    mu_floor: float = DEFAULT_KK_MU_FLOOR,
) -> LinKKResult:
    """Fit the K–K basis at the best-conditioned order and return per-point residuals.

    **Order selection is this function's job, not ``linKK``'s, and that is the whole
    difference.** ``impedance.validation.linKK`` implements Schönleber's published
    rule — walk M upward, stop at the *first* order whose μ falls below ``c`` — which
    is only sound if μ decreases monotonically in M. On this rig's spectra it does not:
    measured on a noise-free control, μ runs 0.814 at M = 5, 0.308 at M = 6, 0.389 at
    M = 7, 0.710 at M = 8, and oscillates between 0.3 and 0.93 until M ≈ 37. "First
    crossing" therefore stops on a cliff edge — and the order it stopped at, M = 5,
    left 33 of 53 points above a 1 % residual on a spectrum that is *exactly* K–K
    compliant. The gate downstream is empowered to truncate a contiguous low-frequency
    failing run, so under-fitting here does not merely mis-measure: it manufactures the
    tail that then gets cut, and moves the fitted ``R1`` — and so σ = K/R1 — by up to
    272× on real spectra.

    So the walk continues to ``max_M`` and selects the order that **minimises the
    median residual**, subject to two conditions:

    ``μ ≤ c``
        Schönleber's criterion, kept, but as a *filter* rather than a stopping rule:
        the ladder must be flexible enough not to be under-fitting.
    ``μ ≥ mu_floor``
        The conditioning bound. Without it "minimise the residual" degenerates into
        "interpolate the data" — see :data:`DEFAULT_KK_MU_FLOOR` for the 27-point
        spectrum on which the unconstrained minimum sits at M = 33.

    A spectrum whose μ never reaches ``c`` falls back to the best conditioned order and
    says so in the log, because a K–K test that produced no verdict is worse than one
    that produced a slightly under-flexible one.

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
        # Imported for availability, not for use — the walk imports it per rung. A
        # missing library must produce this one clear reason rather than a walk that
        # skips every order and reports "no order fitted".
        from impedance.validation import linKK  # noqa: F401
    except Exception as exc:                                  # pragma: no cover
        return LinKKResult(error=f"impedance.validation unavailable: {exc}")

    # linKK expects ascending frequency; the rig sweeps high→low.
    order = np.argsort(freq)
    try:
        # linKK prints its ladder order and RMSE to stdout every tenth iteration.
        # Harmless for one spectrum at a REPL, noise across 32 channels a round.
        with contextlib.redirect_stdout(io.StringIO()):
            under_c, conditioned = _walk_orders(
                freq[order], Zc[order], add_cap=bool(blocking),
                c=float(c), max_M=int(max_M), mu_floor=float(mu_floor),
            )
    except Exception as exc:                                  # pragma: no cover
        logger.info("eis_linkk_failed", error=str(exc), n_points=n)
        return LinKKResult(error=f"K–K ladder did not converge: {exc}")

    chosen = under_c or conditioned
    if chosen is None:
        return LinKKResult(error="K–K ladder did not converge: no order fitted")
    if under_c is None:
        logger.info("eis_linkk_mu_target_unreachable", kk_c=float(c), M=chosen.M,
                    mu=chosen.mu, n_points=n,
                    msg="no conditioned order reached the μ target; using the "
                        "best-conditioned one")

    # Undo the sort so every array indexes the caller's own points.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return LinKKResult(
        ok=True, M=chosen.M, mu=chosen.mu,
        resid_pct=chosen.resid_pct[inverse],
        res_real=chosen.res_real[inverse], res_imag=chosen.res_imag[inverse],
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
