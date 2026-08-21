"""σ(t) equilibration statistics — τ, t_tol, and the measured noise floor (P.22).

The equilibration characterization run records EIS repeatedly at a held (T, RH,
leg) and asks one question per channel: **how long does this sample take to stop
changing?** Two statistics answer it, and they are both kept because they fail
differently:

============  ==================================================  ====================
statistic     definition                                          fails when
============  ==================================================  ====================
``τ``         fit of σ(t) = σ_∞ + (σ₀ − σ_∞)·exp(−t/τ)            window < τ; two
                                                                  relaxations; noise
                                                                  ≫ Δσ
``t_tol``     first *t* after which every later σ is within        the series never
              ``tol_rel`` of σ_∞                                   settles → ``None``
============  ==================================================  ====================

τ is the physics and is comparable across samples and temperatures; **t_tol is
the number the campaign will actually configure**, and it is model-free. A τ from
a fit that converged on a bogus σ_∞ can be small with a small residual; t_tol
cannot lie in that direction.

Shape mirrors :mod:`softae.analysis.thermal` — a registry, a factory, and a
model-agnostic entry point — so a caller stays model-agnostic and a new
relaxation model is taught in one place.

**It refuses rather than emitting a bogus τ.** Every refusal sets
``fit_success=False`` and populates ``refusal``; none of them raise. That is the
posture P.11/P.12/P.20 established: an absent number beats a plausible wrong one,
because a plausible wrong τ becomes a conditioning hold duration and then a
campaign's worth of off-equilibrium spectra.

``s_noise`` is measured **in the same run** (:func:`noise_floor`), so the noise
refusal threshold is not a constant somebody typed.

τ from R₁ — a free check on the whole cell-constant path
--------------------------------------------------------
τ is a relaxation *time*, and ``σ = K/R₁`` with ``K`` constant during a hold. So a
τ fitted to the R₁ channel **must equal** the τ fitted to σ: the cell constant
cancels exactly. Computing both therefore costs one extra fit and buys a check on
every link between the circuit fit and the reported conductivity — a per-point
``K`` that is not actually constant (geometry resolved differently on different
rounds), a σ derived from the wrong resistance, a thickness that moved mid-series.
**A material disagreement means something is wrong in the cell-constant path**,
and that is the entire reason to keep it.

*One implementation note, because the identity is not free-form:* the diagnostic
fits the **conductance** ``1/R₁``, not ``R₁``. ``1/R₁ = σ/K`` is σ times a
constant, and an exponential's τ is invariant under a scaling of amplitude — so
τ(1/R₁) ≡ τ(σ) exactly, which is the property the check depends on. Fitting
``R₁`` itself would not have it: if σ(t) is a single exponential then
``R₁(t) = K/σ(t)`` is *not*, and the two τ would differ by an amount that grows
with the relaxation amplitude, turning the diagnostic into a source of false
alarms. See :func:`r1_conductance`.

σ stays the primary observable throughout. The diagnostic never sets
``fit_success``, never fills ``tau_s``, and refuses under exactly the same
discipline — an R₁ that is NULL because a circuit fit failed must not silently
stand in for a missing σ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import structlog

from softae.analysis.conditions import (
    TEMPERATURE_UNAVAILABLE,
    combine_temperature_sources,
    resolve_temperature_C,
)
from softae.errors import AnalysisError

logger = structlog.get_logger(__name__)

#: Default relative tolerance for "settled" — 2 % of σ_∞.
DEFAULT_TOL_REL = 0.02
#: Rounds at the tail of a series that define the settled block.
DEFAULT_N_SETTLE = 5
#: Three free parameters (σ₀, σ_∞, τ); four points fit any τ.
#:
#: **The acquisition side imports this**, and must: ``settle_floor_rounds`` in
#: :mod:`softae.workflows.equilibration` folds it into the fewest rounds a
#: setpoint may run, and ``EquilibrationConfig.validate`` refuses a
#: ``rounds_per_setpoint`` ceiling below it. That coupling is what stops the run
#: emitting a series this module will then refuse — the failure it closed was a
#: 660 s round period, where every hold floor bought fewer than five rounds. So
#: this is one authority, not a number two modules happen to agree on: raising it
#: lengthens the run, and lowering it shortens it, automatically.
#:
#: It is folded in **only where a τ is wanted**, which is not everywhere. The
#: films dry once, at the start: the 2026-08-11 run measured a per-setpoint σ
#: swing of 1600–2800 % at S0 and 57–1370 % at S1, then 0.5–8.5 % and 0.8–3.1 %
#: at S2/S3 against a 5.98 % noise floor. Past the first setpoint or two there is
#: no relaxation left to fit, so ``EquilibrationConfig.tau_setpoints`` bounds how
#: far into the run this floor applies. Beyond it the acquisition side can stop
#: at the settle/time floor alone — and the series it emits there is one this
#: module will refuse for τ, **on purpose**, because no τ was being bought.
MIN_POINTS_FOR_TAU = 5
#: |σ₀ − σ_∞| must exceed this many noise sigmas for a relaxation to exist.
NOISE_REFUSAL_FACTOR = 3.0
#: A τ longer than half the observation window is an extrapolation, not a fit.
MIN_WINDOWS_PER_TAU = 2.0
#: Symmetric relative difference above which τ(σ) and τ(R₁) *disagree*. They should
#: agree to fit noise, since the cell constant cancels exactly; 5 % is loose enough
#: that ordinary scatter in two fits of the same data does not trip it, and tight
#: enough that a K which is not constant across the series does.
R1_AGREEMENT_TOL_REL = 0.05

REFUSAL_TOO_FEW_POINTS = "too_few_points"
REFUSAL_WINDOW_SHORTER_THAN_TAU = "window_shorter_than_tau"
REFUSAL_NOISE_DOMINATED = "noise_dominated"
REFUSAL_NON_MONOTONIC = "non_monotonic"
REFUSAL_UNCONVERGED = "unconverged"
REFUSAL_SIGMA_UNAVAILABLE = "sigma_unavailable"
#: Not a failure: the ``"none"`` model is asked for the model-free numbers only.
REFUSAL_NO_MODEL = "no_model_requested"
#: The R₁ **diagnostic** has nothing to fit. Spelled separately from
#: ``sigma_unavailable`` so a reader can never mistake "the diagnostic was
#: unavailable" for "the observable was unavailable" — R₁ is NULL whenever a
#: circuit fit failed, and it must not stand in for σ.
REFUSAL_R1_UNAVAILABLE = "r1_unavailable"


@dataclass
class EquilibrationResult:
    """One channel's equilibration statistics at one (leg, setpoint).

    ``tau_s`` is ``NaN`` unless ``fit_success``. ``t_tol_s`` and
    ``noise_floor_rel`` are independent of the fit and are populated whenever the
    data supports them — a refused τ does not cost the campaign its hold time.
    """

    model: str = "exponential"
    channel: int = 0
    run_id: str = ""
    leg: str = ""
    setpoint_index: int = -1
    times_s: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    tau_s: float = float("nan")
    tau_stderr_s: float = float("nan")
    sigma_0: float = float("nan")
    sigma_inf: float = float("nan")
    sigma_settled: float = float("nan")
    t_tol_s: float | None = None
    tol_rel: float = DEFAULT_TOL_REL
    n_settle: int = DEFAULT_N_SETTLE
    noise_floor_rel: float | None = None
    #: Measurement noise **plus** residual short-term drift. The two cannot be
    #: separated without a repeat taken with zero time between them, which is not
    #: physically available — so this is an UPPER BOUND on pure measurement noise.
    noise_floor_is_upper_bound: bool = True
    r_squared: float = float("nan")
    n_points: int = 0
    #: The temperature this series was held at, and — inseparably — which
    #: thermometer read it. They travel together because the rig has two, and a
    #: temperature reported without its source is the defect this pair closed:
    #: the air probe read up to 42 °C below the stage at the same setpoint. See
    #: :mod:`softae.analysis.conditions`.
    temperature_C: float = float("nan")
    temperature_source: str = TEMPERATURE_UNAVAILABLE
    fit_success: bool = False
    refusal: str = ""
    error_msg: str = ""
    #: τ from the R₁ channel — a **diagnostic beside** ``tau_s``, never a
    #: replacement for it. See the module docstring: the cell constant cancels, so
    #: these two must agree, and their disagreement is the finding.
    tau_r1_s: float = float("nan")
    tau_r1_stderr_s: float = float("nan")
    r1_fit_success: bool = False
    r1_refusal: str = ""
    #: ``|τ_σ − τ_R₁| / mean(τ_σ, τ_R₁)``; ``None`` unless *both* fits succeeded.
    tau_agreement_rel: float | None = None
    #: ``True`` when the two τ agree inside :data:`R1_AGREEMENT_TOL_REL`, ``False``
    #: when they do not, ``None`` when the comparison could not be made. The third
    #: state is load-bearing: "not checked" is not "checked and fine".
    r1_diagnostic_ok: bool | None = None

    def describe(self) -> str:
        """One line an operator can read at the bench."""
        head = (f"ch{self.channel} {self.leg or '?'}/S{self.setpoint_index}"
                f" {self.describe_temperature()}")
        tau = (f"tau={self.tau_s:.0f}s" if self.fit_success
               else f"tau=REFUSED({self.refusal or 'unknown'})")
        ttol = "t_tol=never" if self.t_tol_s is None else f"t_tol={self.t_tol_s:.0f}s"
        nf = ("noise=?" if self.noise_floor_rel is None
              else f"noise<={self.noise_floor_rel * 100:.2f}%")
        return (f"{head}: {tau}  {ttol}  {nf}  n={self.n_points}"
                f"  {self.describe_r1()}")

    def describe_temperature(self) -> str:
        """The hold temperature **with** its thermometer — never the number alone.

        An operator reading ``T=43C`` cannot tell a stage at 43 °C from a stage at
        85 °C whose air probe read 43. The source label is what makes the number
        actionable, so it is not optional formatting.
        """
        if not np.isfinite(self.temperature_C):
            return f"T=?[{self.temperature_source}]"
        return f"T={self.temperature_C:.1f}C[{self.temperature_source}]"

    def describe_r1(self) -> str:
        """The R₁ cross-check, said in the one place it can be acted on."""
        if not self.r1_fit_success:
            return f"tau(R1)=REFUSED({self.r1_refusal or 'not attempted'})"
        agree = self.tau_agreement_rel
        if agree is None:
            return f"tau(R1)={self.tau_r1_s:.0f}s (no sigma tau to compare)"
        flag = "" if self.r1_diagnostic_ok else "  ** CELL-CONSTANT PATH SUSPECT **"
        return f"tau(R1)={self.tau_r1_s:.0f}s (d={agree * 100:.2f}%){flag}"


# ── Model-free primitives ────────────────────────────────────────────────────

def settled_mean(sigmas: Sequence[float], *, n_settle: int = DEFAULT_N_SETTLE) -> float:
    """Mean of the last *n_settle* finite σ values (σ_∞, model-free)."""
    finite = [float(s) for s in sigmas if np.isfinite(s)]
    if not finite:
        return float("nan")
    tail = finite[-max(1, int(n_settle)):]
    return float(np.mean(tail))


def noise_floor(
    sigmas: Sequence[float], *, n_settle: int = DEFAULT_N_SETTLE
) -> float | None:
    """Relative scatter of the settled tail — an **upper bound** on σ noise.

    A conditioning tolerance below this can never be satisfied, which is why the
    run reports it as a first-class output rather than leaving it implicit. It is
    measurement noise *plus* whatever residual drift is left at the tail; the
    caller must label it as a bound, not as the noise itself.
    """
    finite = [float(s) for s in sigmas if np.isfinite(s)]
    tail = finite[-max(2, int(n_settle)):]
    if len(tail) < 2:
        return None
    mean = float(np.mean(tail))
    if not np.isfinite(mean) or mean == 0.0:
        return None
    return float(np.std(tail, ddof=1) / abs(mean))


def settling_time(
    times_s: Sequence[float],
    sigmas: Sequence[float],
    *,
    tol_rel: float = DEFAULT_TOL_REL,
    n_settle: int = DEFAULT_N_SETTLE,
) -> float | None:
    """First *t* after which **every** later σ is within *tol_rel* of σ_∞.

    ``None`` when the series never settles inside the observation window — an
    honest answer, and the one the campaign needs, because it says the hold was
    not long enough rather than inventing a time that was never demonstrated.

    The settled block must contain at least *n_settle* points: a "settled" run of
    two points at the end of a still-drifting series is scatter, not evidence.
    """
    t = np.asarray(times_s, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if t.size != s.size or t.size == 0:
        return None
    ok = np.isfinite(t) & np.isfinite(s)
    t, s = t[ok], s[ok]
    if t.size < 2:
        return None

    target = settled_mean(s, n_settle=n_settle)
    if not np.isfinite(target) or target == 0.0:
        return None
    band = abs(target) * float(tol_rel)
    need = max(2, int(n_settle))

    within = np.abs(s - target) <= band
    for i in range(t.size - need + 1):
        if bool(np.all(within[i:])):
            return float(t[i])
    return None


def endorse_tolerance(
    tol_rel: float, noise_floor_rel: float | None
) -> tuple[bool, str]:
    """Can a proposed conditioning tolerance be met at all? (``report``'s refusal)

    A tolerance below the measured noise floor can never be satisfied, so a hold
    time derived from it would be a number with no achievable meaning. Refused
    rather than printed with a caveat.
    """
    if noise_floor_rel is None:
        return False, ("no noise floor measured for this channel — cannot say "
                       "whether the tolerance is achievable")
    if float(tol_rel) <= 0:
        return False, "a tolerance of zero or less can never be satisfied"
    if float(tol_rel) < float(noise_floor_rel):
        return False, (
            f"tol_rel {tol_rel * 100:.2f}% is BELOW the measured noise floor "
            f"{noise_floor_rel * 100:.2f}% — no hold length can satisfy it"
        )
    return True, (f"tol_rel {tol_rel * 100:.2f}% is above the {noise_floor_rel * 100:.2f}% "
                  f"noise floor and is achievable")


def _sign_changes(diffs: np.ndarray, *, threshold: float) -> int:
    """Sign changes in a smoothed first difference, ignoring sub-noise steps."""
    if diffs.size < 2:
        return 0
    kernel = np.ones(3) / 3.0
    smooth = np.convolve(diffs, kernel, mode="valid") if diffs.size >= 3 else diffs
    signs = np.sign(smooth[np.abs(smooth) > threshold])
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(signs) != 0))


# ── Fitters ──────────────────────────────────────────────────────────────────

class _BaseEquilibrationFitter:
    """Shared plumbing: the model-free numbers every model reports."""

    model = "base"

    def _seed(
        self,
        times_s: Sequence[float],
        sigmas: Sequence[float],
        *,
        channel: int,
        run_id: str,
        leg: str,
        setpoint_index: int,
        tol_rel: float,
        n_settle: int,
    ) -> EquilibrationResult:
        t_tol = settling_time(times_s, sigmas, tol_rel=tol_rel, n_settle=n_settle)
        return EquilibrationResult(
            model=self.model,
            channel=channel,
            run_id=run_id,
            leg=leg,
            setpoint_index=setpoint_index,
            times_s=[float(v) for v in times_s],
            sigmas=[float(v) for v in sigmas],
            sigma_settled=settled_mean(sigmas, n_settle=n_settle),
            t_tol_s=t_tol,
            tol_rel=float(tol_rel),
            n_settle=int(n_settle),
            noise_floor_rel=noise_floor(sigmas, n_settle=n_settle),
            n_points=int(len(list(sigmas))),
        )


class ToleranceOnlyFitter(_BaseEquilibrationFitter):
    """``"none"`` — t_tol and the settled mean, with **no fit attempted**.

    A deliberate registry member rather than a placeholder: a run that wants the
    model-free number should be able to ask for it without a fitter pretending to
    have found a τ. ``fit_success`` is False and ``refusal`` says why, so the
    absence of τ is recorded as a choice rather than read as a failure.
    """

    model = "none"

    def fit(
        self,
        times_s: Sequence[float],
        sigmas: Sequence[float],
        *,
        channel: int = 0,
        run_id: str = "",
        leg: str = "",
        setpoint_index: int = -1,
        tol_rel: float = DEFAULT_TOL_REL,
        n_settle: int = DEFAULT_N_SETTLE,
    ) -> EquilibrationResult:
        result = self._seed(times_s, sigmas, channel=channel, run_id=run_id, leg=leg,
                            setpoint_index=setpoint_index, tol_rel=tol_rel,
                            n_settle=n_settle)
        result.refusal = REFUSAL_NO_MODEL
        result.error_msg = "model 'none': t_tol only, no relaxation fitted"
        return result


class ExponentialRelaxationFitter(_BaseEquilibrationFitter):
    """``"exponential"`` — σ(t) = σ_∞ + (σ₀ − σ_∞)·exp(−t/τ).

    **The scaling matters.** ``curve_fit``'s default step is unit-scaled and τ
    lives in the 10²–10⁴ s range, so an unscaled fit wanders and then reports
    convergence — the same class of defect as the E0/E1 ``x_scale`` bug. Time is
    therefore rescaled by a seed taken from the model-free ``t_tol`` and the fit
    is done in units of that seed, then converted back.
    """

    model = "exponential"

    def fit(
        self,
        times_s: Sequence[float],
        sigmas: Sequence[float],
        *,
        channel: int = 0,
        run_id: str = "",
        leg: str = "",
        setpoint_index: int = -1,
        tol_rel: float = DEFAULT_TOL_REL,
        n_settle: int = DEFAULT_N_SETTLE,
    ) -> EquilibrationResult:
        result = self._seed(times_s, sigmas, channel=channel, run_id=run_id, leg=leg,
                            setpoint_index=setpoint_index, tol_rel=tol_rel,
                            n_settle=n_settle)
        t = np.asarray(times_s, dtype=float)
        s = np.asarray(sigmas, dtype=float)

        # Order is deliberate: input validity, then "is there a relaxation at
        # all", then the fit. Each refusal below fires only when the ones above
        # it have already passed, so the reported reason is the first true one.
        if t.size != s.size or t.size == 0 or not np.all(np.isfinite(s)) \
                or not np.all(np.isfinite(t)):
            return _refuse(result, REFUSAL_SIGMA_UNAVAILABLE,
                           "a sigma is NULL or non-finite; P.20 stores NULL when the "
                           "geometry is absent, and a NaN must not become a point")
        if t.size < MIN_POINTS_FOR_TAU:
            return _refuse(result, REFUSAL_TOO_FEW_POINTS,
                           f"{t.size} points; three free parameters need "
                           f"≥ {MIN_POINTS_FOR_TAU}")

        sigma_inf_hat = result.sigma_settled
        sigma_0_hat = float(s[0])
        result.sigma_0, result.sigma_inf = sigma_0_hat, sigma_inf_hat
        tail = s[-max(2, int(n_settle)):]
        s_noise = float(np.std(tail, ddof=1)) if tail.size >= 2 else 0.0
        amplitude = abs(sigma_0_hat - sigma_inf_hat)

        if amplitude < NOISE_REFUSAL_FACTOR * s_noise:
            return _refuse(result, REFUSAL_NOISE_DOMINATED,
                           f"|sigma_0 - sigma_inf| = {amplitude:.3g} is below "
                           f"{NOISE_REFUSAL_FACTOR:g}x the {s_noise:.3g} tail noise; "
                           f"the series is already settled, tau is not zero")
        if _sign_changes(np.diff(s), threshold=s_noise) > 1:
            return _refuse(result, REFUSAL_NON_MONOTONIC,
                           "more than one sign change in the smoothed first "
                           "difference; a one-relaxation model cannot describe two")

        t_span = float(t[-1] - t[0])
        # 2 % of an exponential is reached at ~3.9 tau, so t_tol/3.9 is the right
        # order for the seed; the span/4 fallback covers a series that never met
        # the tolerance at all.
        seed = (result.t_tol_s / 3.9) if result.t_tol_s else (t_span / 4.0)
        seed = float(seed) if np.isfinite(seed) and seed > 0 else max(t_span, 1.0) / 4.0

        def _model(x: np.ndarray, s_inf: float, s_zero: float, tau_scaled: float):
            return s_inf + (s_zero - s_inf) * np.exp(-x / tau_scaled)

        try:
            from scipy.optimize import curve_fit

            popt, pcov = curve_fit(
                _model, (t - t[0]) / seed, s,
                p0=(sigma_inf_hat, sigma_0_hat, 1.0),
                bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
        except Exception as exc:
            return _refuse(result, REFUSAL_UNCONVERGED, f"curve_fit failed: {exc}")

        if not np.all(np.isfinite(pcov)):
            return _refuse(result, REFUSAL_UNCONVERGED,
                           "non-finite covariance; the parameters are not determined")

        tau = float(popt[2]) * seed
        result.sigma_inf = float(popt[0])
        result.sigma_0 = float(popt[1])
        result.tau_s = tau
        result.tau_stderr_s = float(np.sqrt(pcov[2][2])) * seed

        if not np.isfinite(tau) or tau <= 0:
            return _refuse(result, REFUSAL_UNCONVERGED, f"non-physical tau {tau:.3g} s")
        if t_span < MIN_WINDOWS_PER_TAU * tau:
            return _refuse(result, REFUSAL_WINDOW_SHORTER_THAN_TAU,
                           f"window {t_span:.0f}s < {MIN_WINDOWS_PER_TAU:g}x the fitted "
                           f"tau {tau:.0f}s; that is an extrapolation, not a fit")

        pred = _model((t - t[0]) / seed, *popt)
        ss_res = float(np.sum((s - pred) ** 2))
        ss_tot = float(np.sum((s - float(np.mean(s))) ** 2))
        result.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        result.fit_success = True
        return result


def _refuse(result: EquilibrationResult, refusal: str, why: str) -> EquilibrationResult:
    """Record a refusal on *result* — never raise, never emit a number."""
    result.fit_success = False
    result.tau_s = float("nan")
    result.tau_stderr_s = float("nan")
    result.refusal = refusal
    result.error_msg = why
    logger.info("equilibration_fit_refused", channel=result.channel, leg=result.leg,
                setpoint_index=result.setpoint_index, refusal=refusal, detail=why)
    return result


#: Registry mapping model name → fitter class (mirrors ``THERMAL_MODELS``).
EQUILIBRATION_MODELS: dict[str, type] = {
    "exponential": ExponentialRelaxationFitter,
    "none": ToleranceOnlyFitter,
}


def make_equilibration_fitter(model: str):
    """Return a fitter instance for *model* (``"exponential"`` or ``"none"``)."""
    try:
        return EQUILIBRATION_MODELS[model]()
    except KeyError:
        raise AnalysisError(
            f"unknown equilibration model '{model}'; "
            f"available: {sorted(EQUILIBRATION_MODELS)}"
        ) from None


def r1_conductance(r1_ohms: Sequence[Any]) -> list[float]:
    """``1/R₁`` — the quantity σ is proportional to, with the same τ.

    ``σ = K/R₁``, so ``1/R₁`` is σ divided by the cell constant. Scaling an
    exponential's amplitude leaves its τ untouched, which is what makes
    τ(1/R₁) ≡ τ(σ) an *exact* identity rather than an approximation, and therefore
    what makes a disagreement diagnostic rather than expected.

    A NULL, non-finite or non-positive R₁ becomes ``NaN`` on purpose: the fitter's
    own refusal is the correct outcome, and dropping the point would shorten the
    series behind the caller's back.
    """
    out: list[float] = []
    for value in r1_ohms:
        try:
            r = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out.append(float("nan"))
            continue
        out.append(1.0 / r if np.isfinite(r) and r > 0 else float("nan"))
    return out


def _matched_to(values: Sequence[float], reference: Sequence[float]) -> list[float]:
    """Rescale *values* to the magnitude of *reference*. τ is unaffected; noise is.

    ``curve_fit``'s convergence criteria are partly absolute, so a series living
    at 10⁻⁶ and the same series at 10⁻⁴ stop at slightly different optima — the two
    τ then differ by ~10⁻⁴ *relative* for purely numerical reasons. That is small,
    but this diagnostic exists to say whether a difference is real, and a floor of
    numerical disagreement is exactly what it must not have.

    A single multiplicative constant cannot change an exponential's τ, so this
    **cannot** hide a genuine disagreement: a cell constant that drifts across the
    series still changes the shape, and shape is all τ sees.
    """
    scale = _matching_scale(values, reference)
    return [float(v) * scale for v in values]


def _matching_scale(values: Sequence[float], reference: Sequence[float]) -> float:
    ref = np.abs(np.asarray(list(reference), dtype=float))
    own = np.abs(np.asarray(list(values), dtype=float))
    if ref.size != own.size or ref.size == 0:
        return 1.0
    ref_mean, own_mean = float(np.mean(ref)), float(np.mean(own))
    if not (np.isfinite(ref_mean) and np.isfinite(own_mean)) or own_mean == 0.0 \
            or ref_mean == 0.0:
        return 1.0
    return ref_mean / own_mean


def add_r1_diagnostic(
    result: EquilibrationResult,
    times_s: Sequence[float],
    r1_ohms: Sequence[Any],
    *,
    tol_rel: float = DEFAULT_TOL_REL,
    n_settle: int = DEFAULT_N_SETTLE,
) -> EquilibrationResult:
    """Fit τ on the R₁ channel and record it **beside** the σ-based τ.

    Mutates and returns *result*. It touches only the ``r1_*`` fields: σ remains
    the primary observable and a refused σ fit is not rescued by a successful R₁
    one. The reverse also holds — an R₁ that is NULL for even one round refuses
    with :data:`REFUSAL_R1_UNAVAILABLE` rather than fitting a shortened series,
    because R₁ goes NULL precisely when a circuit fit failed and that point's σ is
    suspect too.
    """
    conductance = r1_conductance(r1_ohms)
    if not conductance or not all(np.isfinite(g) for g in conductance):
        result.r1_refusal = REFUSAL_R1_UNAVAILABLE
        result.tau_r1_s = float("nan")
        result.r1_fit_success = False
        result.r1_diagnostic_ok = None
        return result
    conductance = _matched_to(conductance, result.sigmas)

    fitted = ExponentialRelaxationFitter().fit(
        times_s, conductance, channel=result.channel, run_id=result.run_id,
        leg=result.leg, setpoint_index=result.setpoint_index, tol_rel=tol_rel,
        n_settle=n_settle)
    result.r1_fit_success = fitted.fit_success
    result.r1_refusal = fitted.refusal
    result.tau_r1_s = fitted.tau_s
    result.tau_r1_stderr_s = fitted.tau_stderr_s
    result.tau_agreement_rel = tau_agreement(result.tau_s, fitted.tau_s) \
        if (result.fit_success and fitted.fit_success) else None
    result.r1_diagnostic_ok = (
        None if result.tau_agreement_rel is None
        else bool(result.tau_agreement_rel <= R1_AGREEMENT_TOL_REL))
    if result.r1_diagnostic_ok is False:
        logger.warning(
            "equilibration_r1_tau_disagrees", channel=result.channel,
            leg=result.leg, setpoint_index=result.setpoint_index,
            tau_sigma_s=result.tau_s, tau_r1_s=result.tau_r1_s,
            relative_difference=result.tau_agreement_rel,
            detail="sigma = K/R1 with K constant, so these must agree; a material "
                   "difference means the cell-constant path is wrong",
        )
    return result


def tau_agreement(tau_sigma_s: float, tau_r1_s: float) -> float | None:
    """Symmetric relative difference between two τ, or ``None`` if either is unusable.

    Symmetric rather than "relative to σ" so the number does not depend on which
    of two equally-good fits is called the reference.
    """
    try:
        a, b = float(tau_sigma_s), float(tau_r1_s)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    mean = (abs(a) + abs(b)) / 2.0
    return None if mean == 0.0 else float(abs(a - b) / mean)


def fit_equilibration(
    model: str,
    times_s: Sequence[float],
    sigmas: Sequence[float],
    *,
    channel: int = 0,
    run_id: str = "",
    leg: str = "",
    setpoint_index: int = -1,
    tol_rel: float = DEFAULT_TOL_REL,
    n_settle: int = DEFAULT_N_SETTLE,
) -> EquilibrationResult:
    """Fit one σ(t) series with the selected relaxation *model*.

    Never raises for data reasons: an unfittable series comes back with
    ``fit_success=False`` and a populated ``refusal``. Only an unknown *model*
    raises, because that is a caller bug rather than a measurement outcome.
    """
    return make_equilibration_fitter(model).fit(
        times_s, sigmas, channel=channel, run_id=run_id, leg=leg,
        setpoint_index=setpoint_index, tol_rel=tol_rel, n_settle=n_settle,
    )


# ── Reading a recorded run back ──────────────────────────────────────────────

#: Columns the σ(t) reconstruction needs. No schema change: every one of these is
#: already written by ``analysis.eis.router`` on an ordinary EIS shot.
#:
#: ``f.R1`` rides along for the τ cross-check. It is already in the row the join
#: reads, so carrying it costs one column and buys the cell-constant diagnostic.
#:
#: ``c.stage_temp_pv_C`` is here because ``c.chamber_air_C`` is the humidity
#: sensor's chamber-air reading, not the sample's. This query selected the air
#: probe alone until 2026-08-11, when the columns were still named ``temp_pv_C`` /
#: ``temp_sp_C`` and read like an SP/PV pair off one instrument; see
#: :mod:`softae.analysis.conditions` for what that cost. Both are read and
#: :func:`~softae.analysis.conditions.resolve_temperature_C`
#: decides between them, so the choice is made in one place for the whole system.
_SERIES_SQL = """
SELECT m.measurement_id, m.channel, m.timestamp, m.eis_file_path,
       f.sigma_S_per_cm, f.R1, c.stage_temp_sp_C, c.stage_temp_pv_C, c.chamber_air_C,
       c.rh_sp_pct, c.rh_pv_pct
FROM measurements m
LEFT JOIN fit_results f ON f.measurement_id = m.measurement_id
LEFT JOIN conditions  c ON c.measurement_id = m.measurement_id
WHERE m.run_id = ?
ORDER BY m.timestamp, m.measurement_id
"""


def load_sigma_series(store: Any, run_id: str, sidecar: dict[str, Any]) -> dict[
        tuple[int, str, int, str], list[dict[str, Any]]]:
    """Reconstruct σ(t) per coordinate by joining measurements × fits × conditions.

    The **coordinate** is the one thing the database cannot supply — ``router``
    reads a fixed list of tags and drops the rest — so it comes from the sidecar
    this run wrote, keyed on the step name. The link is
    ``measurements.eis_file_path``, whose stem the router derives from the step
    name (``<step>_ch<N>.txt`` for any name that is not the Arrhenius sweep's).

    Returns ``{(channel, leg, setpoint_index, kind): [point, …]}`` ordered by
    ``t_since_hold_s``. Points whose σ is NULL are **kept with ``sigma=None``**,
    so the fitter can refuse them rather than silently seeing a shorter series.

    Each point carries ``temperature_C`` **and** ``temperature_source``, never one
    without the other. The two raw reads survive beside them under names that say
    which instrument they came from — ``stage_temp_pv_C`` and ``chamber_air_C`` —
    because the RH physics still needs the humidity sensor's own air temperature
    even though it is the wrong number for the sample.
    """
    by_stem: dict[str, dict[str, Any]] = {
        f"{p['step_name']}_ch{int(p['channel'])}": p
        for p in (sidecar.get("points") or [])
    }
    out: dict[tuple[int, str, int, str], list[dict[str, Any]]] = {}
    for row in store._conn.execute(_SERIES_SQL, (run_id,)).fetchall():
        (mid, channel, timestamp, path, sigma, r1,
         t_sp, stage_pv, air_pv, rh_sp, rh_pv) = row
        stem = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
        if stem.endswith(".txt"):
            stem = stem[:-4]
        point = by_stem.get(stem)
        if point is None:
            continue
        key = (int(channel), str(point["leg"]), int(point["setpoint_index"]),
               str(point.get("kind", "series")))
        temperature_C, temperature_source = resolve_temperature_C(
            stage_pv_C=stage_pv, stage_sp_C=t_sp, chamber_air_C=air_pv)
        out.setdefault(key, []).append({
            "measurement_id": int(mid),
            "t_since_hold_s": float(point["t_since_hold_s"]),
            "round_index": int(point["round_index"]),
            "sigma": None if sigma is None else float(sigma),
            "R1": None if r1 is None else float(r1),
            "timestamp": timestamp,
            "temperature_C": temperature_C,
            "temperature_source": temperature_source,
            "temp_sp_C": t_sp,
            "stage_temp_pv_C": stage_pv,
            # Named for the instrument, matching the column since the 2026-08-11 rename.
            # The RH physics needs the humidity sensor's own air temperature; it is
            # simply not the sample's.
            "chamber_air_C": air_pv,
            "rh_sp_pct": rh_sp, "rh_pv_pct": rh_pv,
        })
    for points in out.values():
        points.sort(key=lambda p: p["t_since_hold_s"])
    return out


#: Every stored spectrum of a run, keyed the way :func:`load_sigma_series` keys it.
_ARC_SQL = """
SELECT m.channel, m.eis_file_path, m.payload_path
FROM measurements m
WHERE m.run_id = ?
ORDER BY m.measurement_id
"""


def _payload_arc(project_dir: Any, payload_path: str) -> Any:
    """One payload's arc-closure state, or ``UNKNOWN`` when it cannot be read."""
    from pathlib import Path

    import xarray as xr

    from softae.analysis.eis.arc import UNKNOWN, ArcClosure, arc_closure
    from softae.analysis.eis_data import FREQ_DIM

    path = Path(payload_path)
    if not path.is_absolute():
        path = Path(project_dir) / path
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return arc_closure(ds[FREQ_DIM].values, ds["z_imag_neg"].values,
                               ds["phase"].values if "phase" in ds else None)
    except (OSError, KeyError, ValueError):
        return ArcClosure(UNKNOWN, reason="payload unreadable")


def arc_closure_rates(store: Any, run_id: str,
                      sidecar: dict[str, Any]) -> dict[str, Any]:
    """How often the semicircle closed in band, per setpoint block and per channel.

    Read from the stored payloads rather than from ``fit_results``: the check
    post-dates every row already on disk, and the spectrum is the evidence in
    either case. Coordinates come from the sidecar exactly as
    :func:`load_sigma_series` takes them, so these blocks are the blocks the rest
    of the report names.

    Returns ``{"n", "n_open", "n_unknown", "by_block", "by_channel"}``; the two
    maps hold ``[open, total]``.
    """
    from softae.analysis.eis.arc import OPEN, UNKNOWN

    by_stem = {f"{p['step_name']}_ch{int(p['channel'])}": p
               for p in (sidecar.get("points") or [])}
    blocks: dict[tuple[str, int], list[int]] = {}
    channels: dict[int, list[int]] = {}
    total = n_open = n_unknown = 0
    for channel, path, payload in store._conn.execute(_ARC_SQL, (run_id,)).fetchall():
        stem = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
        point = by_stem.get(stem[:-4] if stem.endswith(".txt") else stem)
        if point is None or not payload:
            continue
        arc = _payload_arc(store.project_dir, payload)
        total += 1
        n_open += arc.state == OPEN
        n_unknown += arc.state == UNKNOWN
        key = (str(point["leg"]), int(point["setpoint_index"]))
        for tally in (blocks.setdefault(key, [0, 0]),
                      channels.setdefault(int(channel), [0, 0])):
            tally[0] += arc.state == OPEN
            tally[1] += 1
    return {"n": total, "n_open": n_open, "n_unknown": n_unknown,
            "by_block": blocks, "by_channel": channels}


def series_temperature(points: Sequence[dict[str, Any]]) -> tuple[float, str]:
    """The temperature a held series sat at, and which thermometer says so.

    Median rather than mean: a setpoint's first round is taken while the stage is
    still arriving, and one approach transient must not drag the number that labels
    the whole hold. Points that resolved to no thermometer are skipped — they carry
    no temperature to average — and a series whose rounds resolved to *different*
    instruments comes back labelled ``mixed`` rather than adopting the first one's
    label. See :func:`~softae.analysis.conditions.combine_temperature_sources`.
    """
    usable = [p for p in points
              if str(p.get("temperature_source", TEMPERATURE_UNAVAILABLE))
              != TEMPERATURE_UNAVAILABLE
              and np.isfinite(float(p.get("temperature_C", float("nan"))))]
    if not usable:
        return float("nan"), TEMPERATURE_UNAVAILABLE
    source = combine_temperature_sources(p["temperature_source"] for p in usable)
    return float(np.median([float(p["temperature_C"]) for p in usable])), source


def fit_run(
    series: dict[tuple[int, str, int, str], list[dict[str, Any]]],
    *,
    model: str = "exponential",
    run_id: str = "",
    tol_rel: float = DEFAULT_TOL_REL,
    n_settle: int = DEFAULT_N_SETTLE,
    kind: str = "series",
) -> list[EquilibrationResult]:
    """Fit every ``kind`` coordinate in a loaded run, sorted for stable output.

    A NULL σ is passed through as ``NaN`` on purpose: the fitter's
    ``sigma_unavailable`` refusal is the correct outcome, and dropping the point
    here would hide a geometry gap behind a shorter but apparently clean series.

    Every series also gets the R₁ cross-check (:func:`add_r1_diagnostic`), which
    is why it is not opt-in: a consistency check nobody remembers to ask for is
    one that never runs.

    Each result also carries the hold temperature **and its source**
    (:func:`series_temperature`), so no consumer downstream has to guess which of
    the rig's two thermometers produced the number it is about to put in a 1/T.
    """
    results: list[EquilibrationResult] = []
    for key in sorted(series, key=lambda k: (k[1], k[2], k[0], k[3])):
        channel, leg, sp_idx, point_kind = key
        if kind and point_kind != kind:
            continue
        points = series[key]
        times = [p["t_since_hold_s"] for p in points]
        sigmas = [float("nan") if p["sigma"] is None else p["sigma"] for p in points]
        result = fit_equilibration(
            model, times, sigmas, channel=channel, run_id=run_id, leg=leg,
            setpoint_index=sp_idx, tol_rel=tol_rel, n_settle=n_settle,
        )
        result.temperature_C, result.temperature_source = series_temperature(points)
        results.append(add_r1_diagnostic(
            result, times, [p.get("R1") for p in points],
            tol_rel=tol_rel, n_settle=n_settle))
    return results


# ── Session drift, at zero instrument cost ───────────────────────────────────

def session_drift(
    results: Sequence[EquilibrationResult], *, tol_rel: float = DEFAULT_TOL_REL,
) -> list[dict[str, Any]]:
    """First settled block of the up leg vs last settled block of the down leg.

    Both are at the same nominal condition — the up leg starts where the down leg
    ends — so their disagreement is **session drift**: the sample, the cell or the
    instrument moving over the nine hours between them. It is the same reasoning
    that put ``DRIFT_REPEAT_ROLE`` in the commissioning path, and it doubles as
    question 3's retrace evidence at the reference point.

    It costs **no instrument time**. The spec originally bought this with two
    ``Longest`` rounds per setpoint, retired on the argument that they acquired
    points below a ~9 Hz phase-reliable floor. That floor rested on the ``Z_φ``
    ceiling ``analysis/eis/envelope.py`` has since **withdrawn**, so the retirement
    now stands on cost rather than on a floor — 0.2 Hz for reach nothing here needs.
    Either way the blocks compared here are the ordinary settled tails that the
    σ(t) series already produces.

    The yardstick is the run's **own measured noise floor**, not a typed constant:
    a drift smaller than the scatter of the settled block is not evidence of
    anything. ``significant`` is ``None`` when no noise floor could be measured —
    "not checked" is not "checked and fine".
    """
    by_key = {(r.leg, r.setpoint_index, r.channel): r for r in results}
    down_indices = [sp for (leg, sp, _ch) in by_key if leg == "down"]
    if not down_indices:
        return []
    last_down = max(down_indices)

    rows: list[dict[str, Any]] = []
    for channel in sorted({ch for (_leg, _sp, ch) in by_key}):
        start = by_key.get(("up", 0, channel))
        end = by_key.get(("down", last_down, channel))
        if start is None or end is None:
            continue
        s0, s1 = start.sigma_settled, end.sigma_settled
        mean = (abs(s0) + abs(s1)) / 2.0 if np.isfinite(s0) and np.isfinite(s1) else 0.0
        drift = float(abs(s0 - s1) / mean) if mean > 0 else None
        floors = [f for f in (start.noise_floor_rel, end.noise_floor_rel)
                  if f is not None]
        floor = max(floors) if floors else None
        rows.append({
            "channel": channel,
            "start": {"leg": "up", "setpoint_index": 0,
                      "sigma_settled": _finite_or_none(s0)},
            "end": {"leg": "down", "setpoint_index": last_down,
                    "sigma_settled": _finite_or_none(s1)},
            "drift_rel": drift,
            "noise_floor_rel": floor,
            "tol_rel": float(tol_rel),
            "significant": (None if drift is None or floor is None
                            else bool(drift > max(floor, float(tol_rel)))),
        })
    return rows


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


# ── The adaptive settle criterion — the run's own stopping rule ──────────────
#
# The 2026-08-11 production run held every setpoint for a fixed 15 rounds. The
# evidence says that is right once and wrong seven times: the σ swing at the
# first setpoint was 1600–2800 %, at the second 57–1370 %, and at the third and
# fourth 0.5–8.5 % and 0.8–3.1 % — flat to within the noise. Seven of the eight
# setpoints were over-held, at 45 minutes each.
#
# So `rounds_per_setpoint` becomes a CEILING and the series stops when σ stops
# moving. Everything below is pure: it takes a window of fits and returns a
# verdict, with no clock, no store and no chamber, because the one decision that
# must never be got wrong here is *which channels count as evidence* and that is
# not a decision worth needing a rig to test.
#
# **The σ criterion never reads a setpoint, and no gate here may make σ wait on
# the PV *reaching* one.** In the 2026-08-11 run the chamber took ~5000 s to
# bring 85 °C down to its commanded 15 %RH — it arrives (min PV 13.63 there), but
# far outside the 1800 s `rh_approach_timeout_s`, so 4 of 16 approaches timed
# out. A rule that waited for RH-in-band would therefore fire late or never at
# exactly the setpoints where hold length matters, and the run would degenerate
# to the ceiling. σ is what the campaign configures a hold time from; σ is what
# is graded.
#
# A **stability** test is a different thing and is permitted: `rh_window_spread`
# compares the PV series *to itself* over the judged window and is given no
# setpoint to read. Under a chamber pinned above its command but steady, a
# tracking gate never fires and this one fires immediately. What it withholds is
# `settled` while the environment is genuinely still moving, which is the
# intended behaviour. **If a later edit hands this function a setpoint, the
# prohibition above is back in force.**

#: The setpoint stopped because σ stopped moving.
SETTLE_SETTLED = "settled"
#: The setpoint ran out of rounds. The criterion *was* evaluable and said no.
SETTLE_CEILING = "ceiling"
#: The setpoint ran out of rounds and the criterion was **never evaluable** —
#: too few channels ever carried usable evidence. Spelled apart from
#: :data:`SETTLE_CEILING` because "σ was still moving" and "nothing here could
#: tell us whether σ was moving" are different findings and only one of them is
#: about the sample.
SETTLE_NOT_EVALUABLE = "not_evaluable"
#: The criterion was switched off; the setpoint ran exactly to the ceiling.
SETTLE_DISABLED = "disabled"

#: Must exceed the run's measured noise floor (median 5.98 % over 96 series) or
#: the criterion can never be met. 10 % clears it with room for the 22 of 96
#: series that scattered above 20 % to still fail honestly rather than silently.
DEFAULT_SETTLE_TOL_REL = 0.10
#: Consecutive rounds that must all sit inside the tolerance. Two rounds is a
#: coincidence at this noise level; three is a claim.
DEFAULT_SETTLE_N_ROUNDS = 3
#: Below this many participating channels the window is not evidence, and the
#: setpoint runs to its ceiling instead of "settling" on one channel's opinion.
DEFAULT_SETTLE_MIN_CHANNELS = 3
#: ~3 τ at the session's first setpoint (τ = 425–575 s measured, films drying
#: from ambient to 15 %RH). The first setpoint carries essentially the whole
#: transient, so it gets its own floor.
DEFAULT_MIN_HOLD_FIRST_S = 1500.0
#: Every later setpoint: the films are already dry and the swing is single-digit
#: percent by 65 °C, but the chamber still has to re-establish RH.
DEFAULT_MIN_HOLD_S = 600.0

#: How close to the circuit model's own R₁ lower bound counts as *railed*. The
#: bounded least-squares path lands on the bound to within its own step size
#: rather than exactly on it, so an equality test would miss most railed fits.
RAILED_R1_TOL_REL = 1e-3

#: Range of the per-round RH medians a settle window may span and still count as
#: a *still* room. Measured on ``20260811T023757Z_equilibration_characterization``
#: over 3-round windows with 12 samples per round: steady windows cluster below
#: 1.0 %RH (93 % of 106) and genuine drift windows sit above 1.5, with ~1 % of
#: windows in the band between — so the threshold goes in the gap.
#:
#: **Provisional for small q.** At a campaign's q = 4 the per-round median is
#: taken over 4 samples rather than 12, so the window spread runs somewhat
#: larger. ``SettleOutcome.rh_spread_pct`` records the achieved spread on every
#: phase precisely so this number re-derives itself from real campaigns instead
#: of staying an assertion — the same posture :func:`window_noise_floor` and
#: :func:`endorse_tolerance` already take toward ``settle_tol_rel``.
DEFAULT_RH_STABILITY_PCT = 1.5

#: Why a channel was left out of a settle window.
EXCLUDED_ABSENT = "absent"
EXCLUDED_SIGMA_NULL = "sigma_null"
EXCLUDED_RAILED = "railed_R1"
EXCLUDED_ZERO_MEAN = "zero_mean"

#: Why an RH window could not certify. These two must never collapse into one:
#: :data:`RH_MOVED` is a **verdict** — the room was observed and it moved — while
#: :data:`EXCLUDED_RH_UNREADABLE` is an **absence**, and absence of evidence is
#: not evidence. The first leaves the window evaluable and saying "no"; the
#: second makes it non-evaluable, mirroring the ``min_channels`` shortfall.
RH_MOVED = "rh_moved"
EXCLUDED_RH_UNREADABLE = "rh_unreadable"

# ── The rate criterion's vocabulary ──────────────────────────────────────────
#
# The deviation criterion above measures `max|σ − mean| / |mean|` over a window,
# and for a 3-round window that statistic **is** the window's noise floor to
# within a factor in [0.866, 1.000] — a scatter estimate being asked a question
# about motion. Run `20260820T183625Z_eis_validate` is the proof: one channel
# scattering ~90 % about a *stable* mean held a fifteen-channel board at the
# ceiling for an hour, and the verdict it produced was the same word a channel
# genuinely still drying produced. Those two want opposite actions from an
# operator — stop waiting versus wait longer — so they need different names.
#
# The separation is a two-parameter OLS of `ln σ` on `t`: a **slope** (motion)
# and a **residual** (scatter), gated on a confidence bound of the slope. It is
# not a relaxation fit — no σ_∞, no τ, no monotonicity requirement, no model —
# and it therefore survives τ's retirement. See :func:`log_rate`.

#: `U > tol` and the bound is driven by |ĝ|: the slope is itself significant, so
#: the cell is **still moving**. The one refusal here that is evidence about the
#: sample, and so the one that blocks.
RATE_MOVING = "rate_moving"
#: `U > tol` and the bound is driven by SE(ĝ): the cell is too noisy to be judged
#: at this tolerance. Spelled apart from :data:`RATE_MOVING` for the reason
#: :data:`SETTLE_NOT_EVALUABLE` is spelled apart from :data:`SETTLE_CEILING` —
#: "σ is moving" and "nothing here can tell us whether σ is moving" are different
#: findings and only one of them is about the sample.
RATE_UNDETECTABLE = "rate_undetectable"
#: Fewer than ``min_fit_points`` usable rounds in the window.
RATE_TOO_FEW_POINTS = "rate_too_few_points"
#: Even a **perfectly flat** channel could not have certified over this span at
#: this noise: `t(0.975, k−2)·SE(ĝ)` alone already exceeds the tolerance. This is
#: where :data:`MIN_WINDOWS_PER_TAU`'s discipline survives without its constant —
#: "a window shorter than the dynamics is an extrapolation, not a fit" becomes
#: span-vs-noise rather than span-vs-τ. A statement about the *observation* and
#: not about the sample, so it is not evaluable.
RATE_SPAN_TOO_SHORT = "rate_span_too_short"
#: This channel's **own** noise floor exceeds the relative tolerance, so no hold
#: length can ever certify it. :func:`window_noise_floor` takes the MEDIAN across
#: participants — deliberately — while the criterion aggregates with MAX, so the
#: two point in opposite directions by design and a single unsettleable cell can
#: hold a board whose tolerance was endorsed as achievable.
EXCLUDED_UNSETTLEABLE = "unsettleable"

#: df = 2, `t(0.975, 2) = 4.303` — the fewest points at which a confidence
#: interval exists at all, and the hard floor :func:`rate_check` never fits
#: below whatever it is asked for. k = 3 gives df = 1 and t = **12.706**, 6.5×
#: the z ≈ 1.96 a reader assumes, which would make :data:`RATE_UNDETECTABLE` the
#: universal verdict rather than a diagnosis.
SETTLE_MIN_FIT_POINTS = 4
#: df = 4, t = 2.776 — the fewest at which an interval is worth *quoting*, and
#: the default. `SE(ĝ) = s_resid/√Σ(tᵢ−t̄)² ≈ s_resid·√12/(T·√k)`: span enters
#: linearly and count only as √k, so more points is the weaker of the two levers
#: and this is a floor rather than a target.
DEFAULT_SETTLE_MIN_FIT_POINTS = 6


@dataclass(frozen=True)
class RoundFit:
    """One channel's fit from one round — the two numbers the criterion reads.

    ``sigma`` is ``None`` whenever the circuit fit produced no conductivity, and
    ``r1_ohms`` is carried beside it because a *successful* fit that railed at the
    model's R₁ bound reports a σ that is a bound artefact, not a measurement.
    """

    channel: int
    sigma: float | None = None
    r1_ohms: float | None = None


@dataclass
class SettleCheck:
    """The verdict on one window of rounds.

    ``evaluable`` and ``settled`` are separate on purpose. A window that could not
    be judged is not a window that said "no": the first must run to the ceiling
    and say why, the second is an ordinary not-yet.
    """

    evaluable: bool
    settled: bool
    participating: list[int] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)
    max_deviation_rel: float | None = None
    n_rounds: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ChannelRate:
    """One cell's rate verdict — motion and scatter, reported separately.

    The whole point of the pair ``rate_per_hour`` / ``resid_rel`` is that the
    deviation criterion collapses them into one number and then cannot say which
    one it saw. Three shapes, and the third is the one that has no name today:

    ==========  =========  ===========================  ==================
    ĝ           s_resid    U = |ĝ| + t·SE               verdict
    ==========  =========  ===========================  ==================
    ≈ 0         small      small                        settled
    large       any        large, driven by ĝ           ``rate_moving``
    ≈ 0         **large**  large, driven by SE          ``rate_undetectable``
    ==========  =========  ===========================  ==================

    Rates are in **ln-units per hour** — a fractional rate, so a constant
    multiplicative factor on σ (the cell constant) cancels exactly.
    """

    channel: int
    evaluable: bool = False
    settled: bool = False
    #: ĝ, the OLS slope of ``ln σ`` on ``t``. Negative for a drying film, whose
    #: σ falls; the sign is load-bearing and is asserted, never assumed.
    rate_per_hour: float | None = None
    stderr_per_hour: float | None = None
    #: `U = |ĝ| + t(0.975, k−2)·SE(ĝ)` — a one-sided 95 % upper bound on |rate|.
    #: The channel is certified quiet only if even the top of its interval
    #: cannot be moving faster than the tolerance.
    upper_bound_per_hour: float | None = None
    #: Residual RMS in ln units — the per-channel noise floor, measured on the
    #: same window rather than assumed.
    resid_rel: float | None = None
    #: The channel's relative scatter about its own window mean, from
    #: :func:`channel_noise_floors`. What :func:`endorse_tolerance` judges.
    noise_floor_rel: float | None = None
    n_points: int = 0
    span_s: float = 0.0
    #: `t(0.975, k−2)`, carried so a reader can see it is not 1.96.
    t_multiplier: float | None = None
    refusal: str = ""
    reason: str = ""


@dataclass
class RateCheck:
    """The rate verdict on one window — per cell, and then aggregated.

    ``evaluable`` / ``settled`` carry the same meaning they carry on
    :class:`SettleCheck`, and the per-cell lists below are why the pair is worth
    having: a board that will not clear is now attributable. ``moving`` is the
    only list that blocks. ``undetectable`` and ``unsettleable`` are cells the
    observation cannot speak for, recorded so they can be dropped under an
    explicit budget rather than silently.
    """

    evaluable: bool
    settled: bool
    participating: list[int] = field(default_factory=list)
    excluded: dict[int, str] = field(default_factory=dict)
    by_channel: dict[int, ChannelRate] = field(default_factory=dict)
    quiet: list[int] = field(default_factory=list)
    moving: list[int] = field(default_factory=list)
    undetectable: list[int] = field(default_factory=list)
    unsettleable: list[int] = field(default_factory=list)
    #: The population rate — mean ĝ over the cells that produced a fit. Reported,
    #: **never routed on**: pooling divides SE by √N (3.87× at N = 15) but it
    #: certifies the population rather than the cell, and probe-3ch-v3 is direct
    #: evidence that a population certificate does not carry to a per-cell
    #: endpoint. The ``_status``-vs-``_ok`` discipline, one level down.
    pooled_rate_per_hour: float | None = None
    #: The worst per-cell upper bound — the quantity the gate compares to `tol`.
    max_upper_bound_per_hour: float | None = None
    span_s: float = 0.0
    n_rounds: int = 0
    reason: str = ""


def r1_lower_bound_ohms(circuit_model: str) -> float | None:
    """The R₁ lower bound *this* circuit model fits against, or ``None``.

    Read off :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS` rather than
    written down here, because a bound restated in a second place is a bound that
    will disagree with the fitter after the first edit. ``z_indices`` already
    names which parameter is R₁ (``[R0_index, R1_index]``), so the bound is
    ``bounds[0][z_indices[1]]`` and nothing about the parameter ordering is
    assumed twice.

    ``None`` for a model that declares no bounds — then no fit can be *railed*
    and the participation rule falls back to σ alone, which it says so in the
    exclusion reasons rather than pretending it checked.
    """
    try:
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        spec = CIRCUIT_MODELS[str(circuit_model)]
        lower, _upper = spec["bounds"]
        bound = float(lower[spec["z_indices"][1]])
    except (ImportError, KeyError, TypeError, IndexError, ValueError):
        return None
    return bound if np.isfinite(bound) and bound > 0 else None


def is_railed(r1_ohms: Any, bound_ohms: float | None) -> bool:
    """Did this fit come to rest on the model's R₁ floor rather than on the data?

    325 of 1440 fits in the production run (23 %) sat at R₁ = 100 Ω, the
    ``simpleSalt`` lower bound, and every one of them reported ``success = 1``
    with σ = 0.5 S/cm. A railed channel therefore returns *the same number every
    round*, and a constant is trivially "settled" — which is exactly how a board
    with four dead channels would declare equilibrium on round three and
    under-condition the whole run.
    """
    if bound_ohms is None or r1_ohms is None:
        return False
    try:
        r1 = float(r1_ohms)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(r1):
        return False
    return r1 <= float(bound_ohms) * (1.0 + RAILED_R1_TOL_REL)


def _window_series(window: Sequence[Sequence[RoundFit]]) -> dict[int, list[RoundFit]]:
    """Channel → its fit in each round, with a placeholder for the rounds it missed."""
    channels = sorted({int(fit.channel) for rounds in window for fit in rounds})
    out: dict[int, list[RoundFit]] = {ch: [] for ch in channels}
    for rounds in window:
        by_channel = {int(fit.channel): fit for fit in rounds}
        for channel in channels:
            out[channel].append(by_channel.get(channel, RoundFit(channel=channel)))
    return out


def _exclusion(fits: Sequence[RoundFit], bound_ohms: float | None) -> str:
    """Why this channel cannot carry the window, or ``""`` if it can.

    Order matters: a channel absent from a round is reported as absent rather
    than as a NULL σ, because those send an operator to different places (a step
    that never completed vs. a fit that failed).
    """
    for fit in fits:
        if fit.sigma is None and fit.r1_ohms is None:
            return EXCLUDED_ABSENT
        if fit.sigma is None or not np.isfinite(float(fit.sigma)):
            return EXCLUDED_SIGMA_NULL
        if is_railed(fit.r1_ohms, bound_ohms):
            return EXCLUDED_RAILED
    mean = float(np.mean([float(fit.sigma) for fit in fits]))  # type: ignore[arg-type]
    if not np.isfinite(mean) or mean == 0.0:
        return EXCLUDED_ZERO_MEAN
    return ""


def settle_check(
    window: Sequence[Sequence[RoundFit]],
    *,
    tol_rel: float = DEFAULT_SETTLE_TOL_REL,
    min_channels: int = DEFAULT_SETTLE_MIN_CHANNELS,
    r1_bound_ohms: float | None = None,
) -> SettleCheck:
    """Has σ stopped moving across *window*? Pure, and it never guesses.

    Settled means: for every **participating** channel, every σ in the window
    lies within *tol_rel* of that channel's mean over the window.

    Participation excludes any channel that is absent from a round, has a NULL or
    non-finite σ in a round, or whose R₁ sits on the model's bound in a round.
    Those three are all the same failure wearing different clothes — a series
    that is constant because nothing measured it — and the whole reason this
    function exists is that a constant passes a stability test perfectly.

    Fewer than *min_channels* participants is **not** a "no": it is
    ``evaluable=False``, and the caller must run to its ceiling rather than treat
    an absence of evidence as evidence of settling.
    """
    rounds = [list(r) for r in window]
    if not rounds:
        return SettleCheck(evaluable=False, settled=False, reason="no rounds yet")

    by_channel = _window_series(rounds)
    excluded = {ch: why for ch, fits in by_channel.items()
                if (why := _exclusion(fits, r1_bound_ohms))}
    participating = sorted(ch for ch in by_channel if ch not in excluded)
    needed = max(1, int(min_channels))
    if len(participating) < needed:
        return SettleCheck(
            evaluable=False, settled=False, participating=participating,
            excluded=excluded, n_rounds=len(rounds),
            reason=(f"{len(participating)} participating channel(s) < {needed} "
                    f"required; the criterion cannot be evaluated"))

    worst = 0.0
    for channel in participating:
        sigmas = [float(fit.sigma) for fit in by_channel[channel]]  # type: ignore[arg-type]
        mean = float(np.mean(sigmas))
        worst = max(worst, max(abs(s - mean) for s in sigmas) / abs(mean))
    settled = bool(worst <= float(tol_rel))
    verb = "within" if settled else "outside"
    return SettleCheck(
        evaluable=True, settled=settled, participating=participating,
        excluded=excluded, max_deviation_rel=worst, n_rounds=len(rounds),
        reason=(f"{len(rounds)} rounds, {len(participating)} channel(s): worst "
                f"deviation {worst * 100:.2f}% is {verb} {float(tol_rel) * 100:.2f}%"))


def channel_noise_floors(
    window: Sequence[Sequence[RoundFit]], participating: Sequence[int],
) -> dict[int, float | None]:
    """Per-channel relative scatter over *window* — one entry per participant.

    Exactly what :func:`window_noise_floor` takes the median of, exposed because
    the median answers a different question from the one an operator asks when a
    gate will not clear. *"Is this setpoint's tolerance achievable?"* is a
    property of the board and wants the median; *"which cell can never satisfy
    it?"* is a property of one cell and the median actively hides it — a single
    channel at 90 % scatter moves a fifteen-channel median by nothing at all,
    while :func:`settle_check` aggregates with ``max`` and lets that one channel
    hold the whole board to its ceiling.

    ``None`` for a channel whose window carries fewer than two usable σ: an
    unmeasured floor is not a floor of zero.
    """
    by_channel = _window_series([list(r) for r in window])
    floors: dict[int, float | None] = {}
    for channel in participating:
        fits = by_channel.get(int(channel)) or []
        sigmas = [fit.sigma for fit in fits if fit.sigma is not None]
        floors[int(channel)] = noise_floor(sigmas, n_settle=len(sigmas))
    return floors


def window_noise_floor(
    window: Sequence[Sequence[RoundFit]], participating: Sequence[int],
) -> float | None:
    """Median relative scatter of the participating channels over *window*.

    The same quantity :func:`noise_floor` measures, taken over the window the
    criterion is actually judging rather than over a settled tail chosen later.
    Median across channels so one noisy channel does not decide whether the whole
    setpoint's tolerance was achievable.

    One line over :func:`channel_noise_floors`, so the per-channel and pooled
    views can never disagree about what a floor is.
    """
    floors = [floor for floor in channel_noise_floors(window, participating).values()
              if floor is not None]
    return float(np.median(floors)) if floors else None


def log_rate(
    times_s: Sequence[float], sigmas: Sequence[float]
) -> tuple[float, float, float] | None:
    """OLS of ``ln σ`` on ``t`` → ``(ĝ, SE(ĝ), s_resid)``, all per **second**.

    A two-parameter slope estimate, closed-form, and deliberately **not** a
    relaxation fit: it assumes no mechanism, has no σ_∞ and no τ, requires no
    monotonicity, and carries none of ``curve_fit``'s scaling hazards — which is
    why it survives τ's retirement while :class:`ExponentialRelaxationFitter`
    does not. A later reader must not take "no τ" to mean "back to thresholds":
    a threshold on a magnitude is precisely what failed.

    ``ln σ`` rather than σ so that ĝ is a *fractional* rate. With ``σ = K/R₁``
    and K constant across the window, ``d ln σ/dt = −d ln R₁/dt``: the cell
    constant becomes an additive offset on ``ln σ``, absorbed by the intercept,
    and cancels from the slope exactly. Non-uniform round spacing is handled by
    construction, because ``t`` is a regressor and not an index.

    ``None`` when the estimate is underdetermined — fewer than three finite,
    strictly positive pairs (df ≥ 1 is needed for a residual at all) or a window
    with no time span. Refused rather than returned with an infinite standard
    error, on this module's standing posture: an absent number beats a plausible
    wrong one.
    """
    t = np.asarray(times_s, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    if t.size != s.size or t.size < 3:
        return None
    usable = np.isfinite(t) & np.isfinite(s) & (s > 0.0)
    t, y = t[usable], np.log(s[usable])
    if t.size < 3:
        return None
    centred = t - t.mean()
    sxx = float(np.sum(centred ** 2))
    if not np.isfinite(sxx) or sxx <= 0.0:
        return None
    slope = float(np.sum(centred * (y - y.mean())) / sxx)
    residual = y - (y.mean() + slope * centred)
    s_resid = float(np.sqrt(float(np.sum(residual ** 2)) / (t.size - 2)))
    return slope, s_resid / float(np.sqrt(sxx)), s_resid


def _t_multiplier(df: int) -> float:
    """``t(0.975, df)`` — computed, never recalled as z ≈ 1.96.

    At the window lengths a settle gate can afford the two are not
    interchangeable: df = 2 (k = 4) wants 4.303, which is 2.20× the normal
    multiplier, and df = 1 (k = 3) wants 12.706, which is 6.48×. An interval
    quoted at z would be a third of its true width exactly where the decision is
    hardest, so the multiplier is taken from the distribution the residual
    actually has.
    """
    from scipy import stats

    return float(stats.t.ppf(0.975, max(1, int(df))))


def _reference_half_width(
    times_s: Sequence[float],
    fits: Sequence[tuple[float, float, float] | None] | Any,
) -> float | None:
    """`t(0.975, k−2)·SE` at the window's **median** residual, in ln/h.

    What a *typical* cell on this board could have certified over this
    observation, and therefore the quantity that separates "the window was too
    short" from "this one cell is too noisy". Median across channels for exactly
    the reason :func:`window_noise_floor` takes one — a single scattering cell
    must not be allowed to declare the whole observation inadequate, which is
    the mirror image of the failure that motivated this criterion.

    ``None`` when no channel produced a fit: without a reference the two
    diagnoses cannot be told apart, and the caller says so rather than guessing.
    """
    residuals = [fit[2] for fit in fits if fit is not None]
    t = np.asarray(list(times_s), dtype=float)
    sxx = float(np.sum((t - t.mean()) ** 2)) if t.size else 0.0
    if not residuals or sxx <= 0.0:
        return None
    return float(_t_multiplier(t.size - 2) * np.median(residuals)
                 / np.sqrt(sxx) * 3600.0)


def _channel_rate(
    channel: int,
    times_s: Sequence[float],
    sigmas: Sequence[float],
    fit: tuple[float, float, float] | None,
    *,
    tol_per_hour: float,
    tol_rel: float | None,
    min_fit_points: int,
    reference_half_width: float | None,
) -> ChannelRate:
    """One cell's verdict. The numbers are filled in even under a refusal, so a
    dropped cell can be audited rather than merely named.

    **The order the tests run in is load-bearing, and it is not the order a
    naive reading gives.** Two of them are worth stating outright, because both
    were got wrong first and both fail toward "do not block":

    1. *Moving is decided before the span guard.* A noisy cell that is
       nonetheless **provably** moving — ĝ significant against its own SE — is
       moving, whatever the span. Checking span first excuses it as
       non-evaluable, which turns a channel that must block into one that may be
       dropped. A drying film has both a large residual and a large slope, so
       this is the common case and not a corner.
    2. *Unsettleable is decided on the* **residual**, *not on the raw scatter.*
       ``noise_floor`` measures `std/|mean|`, which during a transient is
       dominated by the drift rather than by the noise — the exact conflation
       this criterion exists to remove. A drying cell's raw floor is five times
       its residual, so endorsing against the raw floor would condemn every
       moving cell as unachievable. ``s_resid`` is the scatter *with the trend
       taken out*, which is what "this cell's own noise floor" has to mean once
       a slope estimate exists.
    """
    span_s = float(max(times_s) - min(times_s)) if len(times_s) else 0.0
    fields: dict[str, Any] = {
        "channel": int(channel), "n_points": len(sigmas), "span_s": span_s,
        "noise_floor_rel": noise_floor(sigmas, n_settle=len(sigmas)),
    }
    needed = max(SETTLE_MIN_FIT_POINTS, int(min_fit_points))
    if fit is None or len(sigmas) < needed:
        return ChannelRate(
            evaluable=False, settled=False, refusal=RATE_TOO_FEW_POINTS,
            reason=(f"{len(sigmas)} point(s) over {span_s:.0f} s; {needed} are "
                    f"needed for an interval worth quoting"),
            **fields)

    slope, stderr, s_resid = fit
    multiplier = _t_multiplier(len(sigmas) - 2)
    half_width = multiplier * stderr * 3600.0
    rate = slope * 3600.0
    bound = abs(rate) + half_width
    fields.update(rate_per_hour=rate, stderr_per_hour=stderr * 3600.0,
                  resid_rel=s_resid, t_multiplier=multiplier,
                  upper_bound_per_hour=bound)

    tol = float(tol_per_hour)
    if tol_rel is not None:
        achievable, why = endorse_tolerance(float(tol_rel), s_resid)
        if not achievable:
            return ChannelRate(evaluable=False, settled=False,
                               refusal=EXCLUDED_UNSETTLEABLE, reason=why, **fields)
    if bound <= tol:
        return ChannelRate(
            evaluable=True, settled=True,
            reason=(f"|rate| is at most {bound:.4f} ln/h at 95 % over "
                    f"{span_s:.0f} s, within {tol:.4f} ln/h"),
            **fields)
    if abs(rate) >= half_width:
        return ChannelRate(
            evaluable=True, settled=False, refusal=RATE_MOVING,
            reason=(f"rate {rate:+.4f} ln/h is significant against its own "
                    f"t*SE {half_width:.4f}, and the 95 % bound {bound:.4f} "
                    f"exceeds {tol:.4f} ln/h"),
            **fields)
    if reference_half_width is not None and reference_half_width > tol:
        return ChannelRate(
            evaluable=False, settled=False, refusal=RATE_SPAN_TOO_SHORT,
            reason=(f"a cell at this window's median residual could itself have "
                    f"certified no better than {reference_half_width:.4f} ln/h "
                    f"over {span_s:.0f} s, above the {tol:.4f} tolerance — the "
                    f"observation was too short, which is not a finding about "
                    f"this cell"),
            **fields)
    return ChannelRate(
        evaluable=False, settled=False, refusal=RATE_UNDETECTABLE,
        reason=(f"the 95 % bound {bound:.4f} ln/h exceeds {tol:.4f}, but it is "
                f"driven by scatter (t*SE {half_width:.4f}) rather than by the "
                f"rate ({rate:+.4f}) — this cell is too noisy to be judged here, "
                f"which is not the same as moving"),
        **fields)


def rate_check(
    window: Sequence[Sequence[RoundFit]],
    times_s: Sequence[float | None],
    *,
    tol_per_hour: float,
    tol_rel: float | None = None,
    min_fit_points: int = DEFAULT_SETTLE_MIN_FIT_POINTS,
    min_channels: int = DEFAULT_SETTLE_MIN_CHANNELS,
    r1_bound_ohms: float | None = None,
) -> RateCheck:
    """Is σ still *moving*, as distinct from merely noisy? Pure, and per cell.

    Sibling of :func:`settle_check`, never a mode of it. Both read the same
    window and the same participation rule; they differ in what they estimate.
    :func:`settle_check` measures a magnitude and compares it to a drift
    tolerance, which conflates the two components it is made of. This one
    regresses ``ln σ`` on ``t`` for every participating channel and gates on a
    one-sided 95 % **upper bound** of the slope, so a cell is certified quiet
    only if even the top of its interval cannot be moving faster than
    *tol_per_hour*.

    Aggregation is **per cell**, matching the validator's endpoints, which are
    per cell too. A channel proven moving blocks the window; a channel that is
    merely too noisy to judge is identified and recorded rather than allowed to
    block, which is the whole difference between this and today's ``max``.

    *times_s* is elapsed seconds, one per round, index-aligned with *window*. A
    window with any round time missing is refused rather than assumed evenly
    spaced — the spacing is the regressor, and inventing it would invent the
    answer. **No setpoint is read here**, and none may ever be passed: like
    :func:`rh_window_spread` this compares a series to itself, and *tol_per_hour*
    is a tolerance rather than a target.

    *tol_rel* is optional and buys the second finding: given the *relative*
    tolerance the deviation criterion is configured with, a cell whose residual
    exceeds it is named :data:`EXCLUDED_UNSETTLEABLE` — no hold length can
    certify it, so waiting is the wrong instruction. Omit it and the window is
    judged on rate alone.

    **Nothing calls this yet, by design.** It ships as a pure unit so that the
    criterion can be measured against real windows before it is given any
    routing power; the selector that would call it is a later stage.
    """
    rounds = [list(r) for r in window]
    if not rounds:
        return RateCheck(evaluable=False, settled=False, reason="no rounds yet")

    times = list(times_s)
    known = [t for t in times if t is not None and np.isfinite(float(t))]
    if len(times) != len(rounds) or len(known) != len(rounds):
        return RateCheck(
            evaluable=False, settled=False, n_rounds=len(rounds),
            reason=(f"{len(known)} of {len(rounds)} round times are known; the "
                    "rate criterion needs a time axis and will not assume even "
                    "spacing"))
    axis = [float(t) for t in known]

    by_channel = _window_series(rounds)
    excluded = {ch: why for ch, fits in by_channel.items()
                if (why := _exclusion(fits, r1_bound_ohms))}
    # A conductance that is not strictly positive has no logarithm, and it is not
    # a measurement of a conducting cell either. Spelled with the existing name
    # because the cause is the existing one: the fit produced no usable σ.
    for channel, fits in by_channel.items():
        if channel not in excluded and any(
                fit.sigma is None or float(fit.sigma) <= 0.0 for fit in fits):
            excluded[channel] = EXCLUDED_SIGMA_NULL
    participating = sorted(ch for ch in by_channel if ch not in excluded)
    span_s = max(axis) - min(axis)
    needed = max(1, int(min_channels))
    if len(participating) < needed:
        return RateCheck(
            evaluable=False, settled=False, participating=participating,
            excluded=excluded, n_rounds=len(rounds), span_s=span_s,
            reason=(f"{len(participating)} participating channel(s) < {needed} "
                    f"required; the criterion cannot be evaluated"))

    series = {channel: [float(fit.sigma) for fit in by_channel[channel]]  # type: ignore[arg-type]
              for channel in participating}
    fitted = {channel: log_rate(axis, sigmas)
              for channel, sigmas in series.items()}
    reference = _reference_half_width(axis, fitted.values())
    by_rate = {
        channel: _channel_rate(
            channel, axis, sigmas, fitted[channel],
            tol_per_hour=tol_per_hour, tol_rel=tol_rel,
            min_fit_points=min_fit_points, reference_half_width=reference)
        for channel, sigmas in series.items()
    }
    grouped = {name: sorted(ch for ch, rate in by_rate.items()
                            if rate.refusal == name)
               for name in (RATE_MOVING, RATE_UNDETECTABLE,
                            EXCLUDED_UNSETTLEABLE, RATE_SPAN_TOO_SHORT,
                            RATE_TOO_FEW_POINTS)}
    quiet = sorted(ch for ch, rate in by_rate.items() if rate.settled)
    bounds = [rate.upper_bound_per_hour for rate in by_rate.values()
              if rate.upper_bound_per_hour is not None]
    rates = [rate.rate_per_hour for rate in by_rate.values()
             if rate.rate_per_hour is not None]
    common = {
        "participating": participating, "excluded": excluded,
        "by_channel": by_rate, "quiet": quiet,
        "moving": grouped[RATE_MOVING],
        "undetectable": grouped[RATE_UNDETECTABLE],
        "unsettleable": grouped[EXCLUDED_UNSETTLEABLE],
        "pooled_rate_per_hour": float(np.mean(rates)) if rates else None,
        "max_upper_bound_per_hour": max(bounds) if bounds else None,
        "span_s": span_s, "n_rounds": len(rounds),
    }
    tol = float(tol_per_hour)
    if grouped[RATE_MOVING]:
        return RateCheck(
            evaluable=True, settled=False,
            reason=(f"{len(grouped[RATE_MOVING])} channel(s) still moving above "
                    f"{tol:.4f} ln/h: "
                    + " ".join(f"ch{ch}" for ch in grouped[RATE_MOVING])),
            **common)
    if len(quiet) >= needed:
        return RateCheck(
            evaluable=True, settled=True,
            reason=(f"{len(rounds)} rounds over {span_s:.0f} s, {len(quiet)} "
                    f"quiet channel(s): worst 95 % bound "
                    f"{max(bounds) if bounds else float('nan'):.4f} ln/h is "
                    f"within {tol:.4f} ln/h"),
            **common)
    tally = ", ".join(
        f"{len(grouped[name])} {name}" for name in
        (RATE_UNDETECTABLE, EXCLUDED_UNSETTLEABLE, RATE_SPAN_TOO_SHORT,
         RATE_TOO_FEW_POINTS) if grouped[name])
    return RateCheck(
        evaluable=False, settled=False,
        reason=(f"{len(quiet)} channel(s) certified quiet < {needed} required "
                f"({tally or 'no channel could be judged'}); an absence of "
                f"evidence, not a verdict"),
        **common)


def round_rh_median(samples: Sequence[Any] | None) -> float | None:
    """The %RH this round stood at — median over its samples, or ``None``.

    The **median**, not the mean, because the PID loop and the sensor together
    put ~1.4 %RH of ripple across a round's channels on the reference run, and
    ripple is not environmental change.

    ``None`` — and never ``0.0`` — for an empty or all-non-finite sample list: a
    plausible-looking humidity is worse than an admitted absence. Non-finite
    entries are dropped, but the *count* is deliberately not judged here. "Two
    channels reported RH" and "twelve did" are different confidence claims and
    only the caller can weigh them.
    """
    values: list[float] = []
    for sample in samples or []:
        try:
            value = float(sample)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else None


def rh_window_spread(round_medians: Sequence[float | None]) -> float | None:
    """Range of the per-round medians across a window — ``None`` if any is missing.

    Step 2 of the stability test, and the step a per-round range test cannot do.
    Within one round the ~5000 s approach measured on the reference run
    contributes ~0.2 %RH, buried under the ripple, so a round-local test rejects
    oscillation and is **blind to drift**; across the window that same drift is
    the whole signal. A draining basin walking the PV down a few tenths per round
    is exactly this shape.

    **Given no setpoint, and it must never be given one.** It compares the series
    to itself, which is the entire reason it is permitted beside the σ criterion
    — see the note above :data:`SETTLE_SETTLED`.

    ``None`` when any round in the window lacks a median: a window with a hole in
    it has not been *observed* to be stable.
    """
    medians = list(round_medians)
    if not medians or any(m is None for m in medians):
        return None
    values = [float(m) for m in medians]  # type: ignore[arg-type]
    return float(max(values) - min(values))


class SettleTracker:
    """Accumulates a setpoint's rounds and answers "may this setpoint stop?".

    Deliberately holds no clock and no store: the caller owns the floor and the
    ceiling, because those are time and this is evidence. Testable end-to-end by
    feeding it lists of :class:`RoundFit`.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tol_rel: float = DEFAULT_SETTLE_TOL_REL,
        n_rounds: int = DEFAULT_SETTLE_N_ROUNDS,
        min_channels: int = DEFAULT_SETTLE_MIN_CHANNELS,
        r1_bound_ohms: float | None = None,
        rh_stability_pct: float | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tol_rel = float(tol_rel)
        self.n_rounds = max(2, int(n_rounds))
        self.min_channels = max(1, int(min_channels))
        self.r1_bound_ohms = r1_bound_ohms
        #: How far the room may move across the judged window and still count as
        #: still. ``None`` — the default — is the gate off, so every existing
        #: caller keeps today's verdicts exactly.
        self.rh_stability_pct = (None if rh_stability_pct is None
                                 else float(rh_stability_pct))
        self.rounds: list[list[RoundFit]] = []
        #: One per round, aligned with :attr:`rounds`. ``None`` = this round's RH
        #: was never observed, which is not the same as "was steady".
        self.rh_medians: list[float | None] = []
        #: Elapsed seconds at each round, aligned with :attr:`rounds` on exactly
        #: the :attr:`rh_medians` precedent. ``None`` = this round carried no
        #: clock, and a rate estimate over a window containing one refuses
        #: rather than assuming even spacing. A **duration**, never a target:
        #: the setpoint prohibition above applies to this axis unchanged.
        self.times_s: list[float | None] = []
        self.last: SettleCheck | None = None
        #: Achieved spread of the **last** judged window, for the record. It is
        #: deliberately not the discriminator below: the binding window may have
        #: been an earlier one.
        self.rh_spread_pct: float | None = None
        #: Was the criterion ever evaluable at this setpoint? The difference
        #: between CEILING and NOT_EVALUABLE, and it must survive a later window
        #: that happened to lose a channel.
        self.ever_evaluable = False
        #: *"This phase would have certified but for the humidity."* Set at the
        #: one moment the RH clause is the **binding** constraint — σ said yes and
        #: RH overrode it — and never inferred from :attr:`rh_spread_pct`, which
        #: records the last window rather than the binding one. A window already
        #: failing on σ must leave it alone, or the discriminator degrades into
        #: "RH was imperfect at some point", which is true of nearly every phase.
        self.rh_blocked_settle = False
        #: The room could not be judged at all in a window that was otherwise
        #: evaluable. Same binding-constraint discipline as
        #: :attr:`rh_blocked_settle`: a window already short on channels says
        #: nothing about the RH channel's health.
        self.rh_unreadable = False

    def observe(
        self, fits: Sequence[RoundFit], *, rh_median_pct: float | None = None,
        t_s: float | None = None,
    ) -> SettleCheck | None:
        """Record one round; return the verdict on the trailing window, if any.

        *rh_median_pct* is this round's :func:`round_rh_median` and defaults to
        ``None`` so every pre-existing caller compiles unchanged. **The default
        is also the value that means "unreadable"**, which is why
        :func:`~softae.core.autonomous_wiring.drive_settle_phase` refuses at entry
        to run a configured tolerance with no supplier wired: a missing wire and a
        dead sensor are otherwise indistinguishable at the point of harm.

        *t_s* is elapsed seconds since the phase began — a duration and not a
        target — and follows the same convention: ``None`` records the absence,
        and :func:`rate_check` refuses a window containing one rather than
        guessing at the spacing. Nothing reads it under the deviation criterion,
        so every existing verdict is unchanged.
        """
        self.rounds.append(list(fits))
        self.rh_medians.append(
            None if rh_median_pct is None else float(rh_median_pct))
        self.times_s.append(None if t_s is None else float(t_s))
        self.last = None
        if not self.enabled or len(self.rounds) < self.n_rounds:
            return None
        self.last = settle_check(
            self.rounds[-self.n_rounds:], tol_rel=self.tol_rel,
            min_channels=self.min_channels, r1_bound_ohms=self.r1_bound_ohms)
        self._apply_rh_clause(self.last)
        # After the clause, never before: a window the room made non-evaluable
        # must not be counted as evidence that the criterion was ever evaluable.
        self.ever_evaluable = self.ever_evaluable or self.last.evaluable
        return self.last

    def _apply_rh_clause(self, check: SettleCheck) -> None:
        """Was the room still across this window? Mutates *check* in place.

        A **stability** test, never a tracking test: no setpoint is read here or
        anywhere below it. Three outcomes, and the third is the one that is easy
        to get wrong —

        =========================  ==========================================
        window RH state            result
        =========================  ==========================================
        spread ≤ tolerance         unchanged; the environment held still
        spread > tolerance         ``settled=False``, reason gains ``rh_moved``
        any round's median missing ``evaluable=False``, ``settled=False``
        =========================  ==========================================

        A window already non-evaluable on σ is left entirely alone: the σ
        shortfall was checked first and owns the reason, and neither
        :attr:`rh_blocked_settle` nor :attr:`rh_unreadable` may be set off it.
        """
        if self.rh_stability_pct is None:
            return
        spread = rh_window_spread(self.rh_medians[-self.n_rounds:])
        self.rh_spread_pct = spread
        if not check.evaluable:
            return
        if spread is None:
            self.rh_unreadable = True
            check.evaluable = False
            check.settled = False
            check.reason = (f"{EXCLUDED_RH_UNREADABLE}: no RH reading for at "
                            f"least one of the last {self.n_rounds} round(s); "
                            f"the room was not observed, so it cannot be called "
                            f"still")
            return
        if spread > self.rh_stability_pct:
            if check.settled:
                self.rh_blocked_settle = True
            check.settled = False
            check.reason = (f"{check.reason}; {RH_MOVED}: RH spread "
                            f"{spread:.2f}%RH over the window exceeds "
                            f"{self.rh_stability_pct:.2f}%RH — σ flat under a "
                            f"moving room is not evidence")

    @property
    def settled(self) -> bool:
        return bool(self.enabled and self.last is not None and self.last.settled)

    @property
    def participating(self) -> list[int]:
        return list(self.last.participating) if self.last is not None else []

    def outcome(self, *, stopped_early: bool) -> str:
        """:data:`SETTLE_SETTLED` / ``CEILING`` / ``NOT_EVALUABLE`` / ``DISABLED``."""
        if not self.enabled:
            return SETTLE_DISABLED
        if stopped_early:
            return SETTLE_SETTLED
        return SETTLE_CEILING if self.ever_evaluable else SETTLE_NOT_EVALUABLE

    def endorsement(self) -> tuple[bool | None, str, float | None]:
        """Could the configured tolerance be met **at all**, on this run's own noise?

        One rule, :func:`endorse_tolerance`, applied to the noise floor this
        setpoint measured. A tolerance below the floor cannot be satisfied by any
        number of rounds, so the honest thing is to say it once and let the
        ceiling do its job rather than to widen the tolerance behind the operator.

        ``(None, …)`` when there is nothing to judge — never ``True``.
        """
        if not self.enabled or self.last is None or not self.last.participating:
            return None, "not evaluated: no participating channels", None
        floor = window_noise_floor(self.rounds[-self.n_rounds:],
                                   self.last.participating)
        ok, why = endorse_tolerance(self.tol_rel, floor)
        return ok, why, floor

    def per_channel_endorsement(
        self,
    ) -> dict[int, tuple[bool | None, str, float | None]]:
        """The same question, asked of each cell instead of of the board.

        :meth:`endorsement` aggregates with the **median** and the criterion
        aggregates with the **max**; they point in opposite directions by design,
        so a board can be told its tolerance is achievable while one cell it can
        never satisfy holds it at the ceiling. That is not hypothetical — it is
        run ``20260820T183625Z_eis_validate``, where a channel at ~90 % scatter
        about a stable mean cost an hour and was never named.

        Returned per channel as :meth:`endorsement` returns it for the board,
        with the same discipline: ``None`` — never ``True`` — where the floor
        could not be measured, because "not checked" is not "checked and fine".
        """
        if not self.enabled or self.last is None or not self.last.participating:
            return {}
        floors = channel_noise_floors(self.rounds[-self.n_rounds:],
                                      self.last.participating)
        judged: dict[int, tuple[bool | None, str, float | None]] = {}
        for channel, floor in floors.items():
            ok, why = endorse_tolerance(self.tol_rel, floor)
            judged[channel] = (None if floor is None else ok, why, floor)
        return judged


#: This round's fits, keyed the way :func:`load_sigma_series` keys the whole run —
#: on the stem of ``measurements.eis_file_path``, which the router derives from
#: the step name. Restricted to the run, because a step name repeats across runs.
_ROUND_FITS_SQL = """
SELECT m.channel, m.eis_file_path, f.sigma_S_per_cm, f.R1
FROM measurements m
LEFT JOIN fit_results f ON f.measurement_id = m.measurement_id
WHERE m.run_id = ?
ORDER BY m.measurement_id
"""


def load_round_fits(
    store: Any, run_id: str, step_names: dict[int, str],
) -> list[RoundFit]:
    """σ and R₁ for one round, read back mid-run so the series can decide to stop.

    Returns **one entry per requested channel**, always. A channel whose step did
    not complete, or whose fit produced no row, comes back as an all-``None``
    :class:`RoundFit` rather than being dropped — a shorter list would read to
    :func:`settle_check` as a smaller board rather than as missing evidence.
    """
    wanted = {f"{name}_ch{int(channel)}": int(channel)
              for channel, name in step_names.items()}
    found: dict[int, RoundFit] = {}
    for channel, path, sigma, r1 in store._conn.execute(
            _ROUND_FITS_SQL, (str(run_id),)).fetchall():
        stem = str(path or "").replace("\\", "/").rsplit("/", 1)[-1]
        if stem.endswith(".txt"):
            stem = stem[:-4]
        if stem not in wanted:
            continue
        found[int(channel)] = RoundFit(
            channel=int(channel),
            sigma=None if sigma is None else float(sigma),
            r1_ohms=None if r1 is None else float(r1))
    return [found.get(int(ch), RoundFit(channel=int(ch)))
            for ch in sorted(step_names)]
