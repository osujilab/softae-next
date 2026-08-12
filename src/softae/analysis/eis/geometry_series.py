"""Geometry series as a calibration route in its own right (E5, framework §5.6, R13).

Where no conductivity standard exists — and none does for these solid films — a series
of samples differing only in thickness **substitutes for the standard and validates
itself at the same time**. The whole argument is that in admittance, at a fixed
frequency, the measurand is linear in the varied dimension and every nuisance term is
an additive constant::

    Re(Y_meas)(t) = σ · (L_stripe / L_gap) · (t − h)  +  G_fixture
    Im(Y_meas)(t) / ω = C_sample(t)                   +  C_stray

Three consequences, in the order they matter:

1. **σ comes from the slope, so no blank subtraction is required.** The fixture enters
   as an intercept and drops out of the derivative. That makes this route immune to
   §5.2's failure mode — an unreproducible blank corrupting every corrected spectrum —
   which is exactly the failure F6 recorded on this fixture (OSL, mean error 32%).
2. **The intercept is a free measurement** of ``G_fixture`` and ``C_stray``, available
   to cross-check a blank when one is eventually trustworthy.
3. **The internal validation is the frequency-independence of the slope.** A genuine
   DC-like ionic conductance is frequency-flat (§1.4); residual dielectric loss scales
   ≈ ω. Fit at several frequencies: if the slope drifts, *the extracted number is not
   σ*, and no amount of goodness-of-fit at any single frequency says otherwise.

Point 3 is why this module refuses rather than warns. A per-frequency R² near 1 is
perfectly compatible with fitting a dielectric loss and calling it conductivity — the
line is straight either way. The frequency axis is the only thing that separates them.

Dead height
-----------
``h`` is **not identifiable from a thickness series alone**, at any number of levels.
The line has one intercept and it carries two unknowns::

    b = G_fixture − m·h

so ``h`` follows only once ``G_fixture`` is known independently — from an open blank,
which on this three-electrode fixture needs the RE→CE jumper before it measures the
fixture rather than inter-stripe geometry. :meth:`GeometrySeriesFit.dead_height_cm`
therefore *requires* that external value as an argument and returns NaN without it.
This is a stronger statement than the overhaul's own ±20 µm caveat: more levels do not
help, because the deficiency is rank, not noise.

Confounding
-----------
F12 is checked here as well as at cast time. The planner in
:mod:`softae.core.thickness_series` prevents channel index from tracking thickness
*before* casting; this module re-checks the data *as it actually arrived*, because a
sound plan followed inattentively produces precisely the dataset the plan existed to
prevent. Two checks at the two moments it can go wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Minimum thickness levels for a geometry series to answer anything (§5.6, overhaul
#: §3.4). Three levels cannot separate an offset from a power law, and two have no
#: residual at all — a line through two points fits perfectly and says nothing.
MIN_LEVELS = 4

#: Minimum span, as a ratio of the largest level to the smallest. A series that barely
#: moves the dimension it varies extracts a slope from noise.
MIN_SPAN_RATIO = 2.0

#: Minimum samples per level. Without replication, one bad film is indistinguishable
#: from a real level effect.
MIN_REPLICATES = 2

#: Largest fractional spread in σ across the analysis frequencies for the slope to
#: count as frequency-independent.
#:
#: **Uncalibrated**, like every other threshold in this suite: chosen as an engineering
#: default with no reference to this rig's spectra, because no unconfounded series
#: exists to calibrate it against. It is reported alongside the number it judged so the
#: verdict can be overridden on inspection.
DEFAULT_SIGMA_SPREAD_TOL = 0.25

#: Largest ``|d ln m / d ln f|`` for the slope to count as flat. The sharper of the two
#: tests: residual dielectric loss contributes a term scaling ≈ ω, so a contaminated
#: slope shows a power-law exponent trending toward 1, while a genuine DC-like ionic
#: conductance sits at 0. Scatter inflates the spread test but not this one.
DEFAULT_SLOPE_EXPONENT_TOL = 0.15


@dataclass(frozen=True)
class SeriesMember:
    """One sample in a geometry series: a spectrum and the thickness it was cast at."""

    thickness_cm: float
    frequency: np.ndarray
    Z: np.ndarray
    channel: int = -1
    label: str = ""

    @property
    def thickness_um(self) -> float:
        return self.thickness_cm * 1e4


@dataclass(frozen=True)
class FrequencySlope:
    """The regression at one frequency. The unit the frequency-independence test acts on."""

    frequency_hz: float
    slope_S_per_cm: float
    slope_se: float
    intercept_S: float
    intercept_se: float
    #: ``slope · L_gap / L_stripe``. Equals σ **only if** the slope is frequency-flat.
    sigma_S_per_cm: float
    r_squared: float
    n_points: int
    #: ``Im(Y)/ω`` intercept — ``C_stray``, a free measurement per §5.6 point 2.
    C_intercept_F: float
    C_slope_F_per_cm: float


@dataclass(frozen=True)
class GeometrySeriesFit:
    """The whole series, its verdict, and what it does *not* license.

    ``sigma_S_per_cm`` is NaN whenever :attr:`slope_frequency_independent` is False.
    That is deliberate and is the module's main opinion: a drifting slope means the
    quantity being extracted is not a conductivity, so emitting one with a caveat
    attached would put a number into a report that nothing downstream re-reads the
    caveat for.
    """

    slopes: tuple[FrequencySlope, ...] = ()
    L_gap_cm: float = float("nan")
    L_stripe_cm: float = float("nan")
    sigma_S_per_cm: float = float("nan")
    sigma_median_raw: float = float("nan")
    sigma_spread: float = float("nan")
    slope_exponent: float = float("nan")
    slope_frequency_independent: bool = False
    #: Median intercept. This is ``G_fixture`` **only at h = 0**; in general it is
    #: ``G_fixture − m·h``, which is why it is not named ``G_fixture``.
    intercept_S: float = float("nan")
    C_stray_F: float = float("nan")
    n_levels: int = 0
    n_samples: int = 0
    span_ratio: float = float("nan")
    min_replicates: int = 0
    confound_verdict: str = "indeterminate"
    confound_correlation: float = float("nan")
    issues: tuple[str, ...] = ()

    @property
    def adequate(self) -> bool:
        """Whether the *design* can answer a geometry-series question at all.

        Separate from whether the fit succeeded. A perfectly clean regression over
        three confounded levels is still unable to support the claim, and reporting
        only the fit quality would hide that.
        """
        return (
            self.n_levels >= MIN_LEVELS
            and self.span_ratio >= MIN_SPAN_RATIO
            and self.min_replicates >= MIN_REPLICATES
            and self.confound_verdict == "ok"
        )

    @property
    def usable(self) -> bool:
        """Adequate design *and* a frequency-flat slope. Both, or no σ."""
        return self.adequate and self.slope_frequency_independent

    def _resolve_G(self, G_fixture_S: Any, freq_hz: float) -> float:
        """``G_fixture`` at one frequency, from a scalar, a mapping or a sequence."""
        if isinstance(G_fixture_S, Mapping):
            keys = [float(k) for k in G_fixture_S
                    if float(k) == float(k) and float(k) > 0]
            if not keys or not (freq_hz > 0):
                return float("nan")
            best = min(keys, key=lambda k: abs(np.log10(k) - np.log10(freq_hz)))
            try:
                return float(G_fixture_S[best])  # type: ignore[index]
            except Exception:
                return float(G_fixture_S[type(next(iter(G_fixture_S)))(best)])
        if isinstance(G_fixture_S, (list, tuple, np.ndarray)):
            arr = np.asarray(G_fixture_S, dtype=float)
            idx = [i for i, s in enumerate(self.slopes)
                   if s.frequency_hz == freq_hz]
            if not idx or idx[0] >= arr.size:
                return float("nan")
            return float(arr[idx[0]])
        try:
            return float(G_fixture_S)
        except (TypeError, ValueError):
            return float("nan")

    def dead_height_profile(
        self, G_fixture_S: Any
    ) -> tuple[tuple[float, float], ...]:
        """``(frequency, h)`` per frequency, so the answer's consistency is visible.

        A real dead height is a **geometric** quantity and cannot depend on frequency.
        If ``h`` drifts across the band, either ``G_fixture`` or the slope is wrong,
        and the median would hide it.
        """
        out: list[tuple[float, float]] = []
        for s in self.slopes:
            G = self._resolve_G(G_fixture_S, s.frequency_hz)
            m = s.slope_S_per_cm
            if not (m == m and m > 0) or not (G == G):
                continue
            out.append((s.frequency_hz, float((G - s.intercept_S) / m)))
        return tuple(out)

    def dead_height_cm(self, G_fixture_S: Any) -> float:
        """``h = (G_fixture − b) / m`` — requires an independently measured conductance.

        Returns NaN without one, and that is the honest answer rather than a fallback.
        A thickness series produces one intercept carrying two unknowns
        (``b = G_fixture − m·h``), so ``h`` is not identifiable here at any number of
        levels. The rank deficiency does not improve with data.

        ``G_fixture_S`` may be a scalar, a ``{frequency: G}`` mapping, or a sequence
        aligned with :attr:`slopes`. **On this rig a scalar is wrong**, and measurably
        so: seven tied open blanks give ``d ln G / d ln f`` between +0.87 and +1.04,
        i.e. the fixture's real part is **dielectric loss (G ∝ ω), not an ohmic leak**.
        A single number therefore describes it at exactly one frequency. The scalar
        form is kept because it is right for a fixture that does leak ohmically, and
        because refusing it would only push callers to pass the value at some
        unstated frequency anyway.

        **Never auto-applied**, whatever is supplied: ``h`` inherits every error in
        ``G_fixture``, and the open blank it comes from is itself only as good as the
        RE→CE jumper that makes it measurable.
        """
        profile = self.dead_height_profile(G_fixture_S)
        if not profile:
            return float("nan")
        return float(np.median([h for _, h in profile]))

    def _median_slope(self) -> float:
        vals = [s.slope_S_per_cm for s in self.slopes
                if s.slope_S_per_cm == s.slope_S_per_cm]
        return float(np.median(vals)) if vals else float("nan")

    def describe(self) -> str:
        if not self.slopes:
            return "geometry series: no fit — " + ("; ".join(self.issues) or "no data")
        head = (
            f"geometry series: {self.n_samples} samples over {self.n_levels} levels "
            f"({self.span_ratio:.1f}× span, min {self.min_replicates} replicates), "
            f"{len(self.slopes)} frequencies"
        )
        if self.slope_frequency_independent:
            body = (
                f" → σ = {self.sigma_S_per_cm:.3g} S/cm "
                f"(spread {self.sigma_spread * 100:.0f}%, "
                f"d ln m/d ln f = {self.slope_exponent:+.2f})"
            )
        else:
            body = (
                f" → σ REFUSED: slope is not frequency-independent "
                f"(spread {self.sigma_spread * 100:.0f}%, "
                f"d ln m/d ln f = {self.slope_exponent:+.2f}) — the extracted "
                f"quantity is not a conductivity. Raw median {self.sigma_median_raw:.3g}"
            )
        tail = f". C_stray = {self.C_stray_F * 1e12:.1f} pF" if self.C_stray_F == self.C_stray_F else ""
        if not self.adequate:
            tail += f". DESIGN INADEQUATE: {'; '.join(self.issues) or 'see issues'}"
        return head + body + tail


def _weighted_lstsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Ordinary least squares ``y = m·x + b``. Returns ``(m, b, m_se, b_se, r2)``."""
    n = x.size
    if n < 2:
        return (float("nan"),) * 5  # type: ignore[return-value]
    A = np.column_stack([x, np.ones_like(x)])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return (float("nan"),) * 5  # type: ignore[return-value]
    m, b = float(coef[0]), float(coef[1])
    resid = y - (m * x + b)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if n > 2:
        s2 = ss_res / (n - 2)
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        m_se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
        b_se = float(np.sqrt(s2 * (1.0 / n + np.mean(x) ** 2 / sxx))) if sxx > 0 else float("nan")
    else:
        # Two points fit a line exactly. Reporting SE 0 would read as certainty.
        m_se = b_se = float("nan")
    return m, b, m_se, b_se, r2


def _common_frequencies(members: Sequence[SeriesMember], per_decade: int = 2) -> np.ndarray:
    """Frequencies present in every member, thinned to *per_decade* representatives.

    Interpolating onto a synthetic grid would let one member's extrapolated tail set a
    slope, so only frequencies every member actually measured are used.
    """
    sets = [set(np.round(np.log10(np.asarray(m.frequency, dtype=float)), 6))
            for m in members if np.asarray(m.frequency).size]
    if not sets:
        return np.asarray([], dtype=float)
    common = sorted(set.intersection(*sets))
    if not common:
        return np.asarray([], dtype=float)
    chosen: list[float] = []
    seen: set[tuple[int, int]] = set()
    for lg in common:
        decade = int(np.floor(lg))
        slot = int((lg - decade) * per_decade)
        key = (decade, slot)
        if key not in seen:
            seen.add(key)
            chosen.append(lg)
    return np.asarray([10.0 ** lg for lg in sorted(chosen)], dtype=float)


def fit_geometry_series(
    members: Sequence[SeriesMember],
    *,
    L_gap_cm: float,
    L_stripe_cm: float,
    frequencies: Sequence[float] | None = None,
    sigma_spread_tol: float | None = None,
    slope_exponent_tol: float | None = None,
    max_correlation: float | None = None,
) -> GeometrySeriesFit:
    """Regress ``Re(Y)`` against thickness at several frequencies (§5.6).

    Returns a :class:`GeometrySeriesFit` whose ``sigma_S_per_cm`` is NaN unless the
    slope is frequency-independent *and* the design is adequate. Never raises: a
    geometry series is analysed long after the samples are gone, so a failure has to
    come back as a report that names what went wrong, not as a traceback.

    The two tolerances default to ``[eis.gates]``, falling back to this module's
    constants if the config cannot be read — an unreadable config must not silently
    tighten or loosen a threshold.
    """
    if sigma_spread_tol is None or slope_exponent_tol is None:
        try:
            from softae.analysis.eis.settings import eis_settings

            g = eis_settings().gates
            if sigma_spread_tol is None:
                sigma_spread_tol = float(g.geom_sigma_spread_tol)
            if slope_exponent_tol is None:
                slope_exponent_tol = float(g.geom_slope_exponent_tol)
        except Exception:
            logger.warning("geometry_series_settings_unreadable", exc_info=True)
    if sigma_spread_tol is None:
        sigma_spread_tol = DEFAULT_SIGMA_SPREAD_TOL
    if slope_exponent_tol is None:
        slope_exponent_tol = DEFAULT_SLOPE_EXPONENT_TOL

    issues: list[str] = []
    members = [m for m in members
               if m.thickness_cm == m.thickness_cm and m.thickness_cm > 0]
    if len(members) < 2:
        return GeometrySeriesFit(
            L_gap_cm=L_gap_cm, L_stripe_cm=L_stripe_cm, n_samples=len(members),
            issues=("fewer than two usable samples",))

    thick = np.asarray([m.thickness_cm for m in members], dtype=float)
    levels = sorted({round(float(t), 12) for t in thick})
    n_levels = len(levels)
    span = float(max(thick) / min(thick)) if min(thick) > 0 else float("nan")
    reps: dict[float, int] = {}
    for t in thick:
        key = round(float(t), 12)
        reps[key] = reps.get(key, 0) + 1
    min_reps = min(reps.values()) if reps else 0

    if n_levels < MIN_LEVELS:
        issues.append(
            f"{n_levels} thickness levels, need {MIN_LEVELS} — three cannot separate "
            f"an offset from a power law and two have no residual")
    if not (span >= MIN_SPAN_RATIO):
        issues.append(f"{span:.2f}× span, need {MIN_SPAN_RATIO:g}×")
    if min_reps < MIN_REPLICATES:
        issues.append(
            f"{min_reps} replicate(s) at the thinnest-covered level, need "
            f"{MIN_REPLICATES} — without replication one bad film reads as a level effect")

    # F12, re-checked on the data as it arrived rather than as it was planned.
    confound_verdict, confound_r = "indeterminate", float("nan")
    channels = [m.channel for m in members]
    if all(c >= 0 for c in channels):
        try:
            from softae.core.thickness_series import detect_confounding

            kw: dict[str, Any] = {}
            if max_correlation is not None:
                kw["max_correlation"] = float(max_correlation)
            report = detect_confounding(channels, [t * 1e4 for t in thick], **kw)
            confound_verdict = report.verdict
            confound_r = report.correlation
            if confound_verdict == "confounded":
                issues.append(
                    f"channel index tracks thickness (r = {confound_r:+.3f}) — F12: no "
                    f"analysis afterwards can separate a fixture difference from a "
                    f"thickness effect")
        except Exception:
            logger.warning("geometry_series_confound_check_failed", exc_info=True)
    else:
        issues.append("channels not supplied — the F12 confound check did not run")

    freqs = (np.asarray(list(frequencies), dtype=float)
             if frequencies is not None else _common_frequencies(members))
    if freqs.size == 0:
        issues.append("no frequency is present in every member")
        return GeometrySeriesFit(
            L_gap_cm=L_gap_cm, L_stripe_cm=L_stripe_cm, n_levels=n_levels,
            n_samples=len(members), span_ratio=span, min_replicates=min_reps,
            confound_verdict=confound_verdict, confound_correlation=confound_r,
            issues=tuple(issues))

    from softae.analysis.eis.admittance import to_admittance

    slopes: list[FrequencySlope] = []
    for f0 in freqs:
        G_vals, C_vals, t_vals = [], [], []
        for m in members:
            fm = np.asarray(m.frequency, dtype=float)
            if fm.size == 0:
                continue
            idx = int(np.argmin(np.abs(np.log10(np.maximum(fm, 1e-30))
                                       - np.log10(max(f0, 1e-30)))))
            Y = to_admittance(np.asarray(m.Z, dtype=complex)[idx:idx + 1])[0]
            if not np.isfinite(Y):
                continue
            omega = 2.0 * np.pi * fm[idx]
            G_vals.append(float(np.real(Y)))
            C_vals.append(float(np.imag(Y) / omega) if omega > 0 else float("nan"))
            t_vals.append(float(m.thickness_cm))
        if len(t_vals) < 2:
            continue
        x = np.asarray(t_vals)
        mG, bG, mG_se, bG_se, r2 = _weighted_lstsq(x, np.asarray(G_vals))
        Cy = np.asarray(C_vals)
        ok = np.isfinite(Cy)
        mC, bC = (_weighted_lstsq(x[ok], Cy[ok])[:2] if ok.sum() >= 2
                  else (float("nan"), float("nan")))
        sigma = (mG * L_gap_cm / L_stripe_cm
                 if (mG == mG and L_stripe_cm > 0) else float("nan"))
        slopes.append(FrequencySlope(
            frequency_hz=float(f0), slope_S_per_cm=mG, slope_se=mG_se,
            intercept_S=bG, intercept_se=bG_se, sigma_S_per_cm=sigma,
            r_squared=r2, n_points=len(t_vals),
            C_intercept_F=bC, C_slope_F_per_cm=mC))

    if not slopes:
        issues.append("no frequency yielded two usable points")
        return GeometrySeriesFit(
            L_gap_cm=L_gap_cm, L_stripe_cm=L_stripe_cm, n_levels=n_levels,
            n_samples=len(members), span_ratio=span, min_replicates=min_reps,
            confound_verdict=confound_verdict, confound_correlation=confound_r,
            issues=tuple(issues))

    sig = np.asarray([s.sigma_S_per_cm for s in slopes], dtype=float)
    good = np.isfinite(sig) & (sig > 0)
    sigma_med = float(np.median(sig[good])) if good.any() else float("nan")
    spread = (float((np.max(sig[good]) - np.min(sig[good])) / sigma_med)
              if good.sum() >= 2 and sigma_med > 0 else float("nan"))

    # The sharper test. Residual dielectric loss scales ~ω, so a contaminated slope
    # shows a power-law exponent heading toward 1; a DC-like ionic conductance sits at
    # 0. Scatter inflates `spread` but leaves this near zero, so the two disagree in a
    # way that is itself informative.
    exponent = float("nan")
    f_arr = np.asarray([s.frequency_hz for s in slopes], dtype=float)
    m_arr = np.asarray([s.slope_S_per_cm for s in slopes], dtype=float)
    fit_ok = np.isfinite(f_arr) & np.isfinite(m_arr) & (f_arr > 0) & (m_arr > 0)
    if fit_ok.sum() >= 3:
        exponent = _weighted_lstsq(np.log10(f_arr[fit_ok]),
                                   np.log10(m_arr[fit_ok]))[0]

    flat = bool(
        spread == spread and spread <= sigma_spread_tol
        and (exponent != exponent or abs(exponent) <= slope_exponent_tol)
    )
    if not flat:
        issues.append(
            f"slope is not frequency-independent (spread {spread:.2f}, "
            f"d ln m/d ln f = {exponent:+.3f}) — §5.6: the extracted number is not σ")
        logger.warning(
            "geometry_series_slope_drifts", spread=spread, exponent=exponent,
            msg="σ refused: a frequency-dependent slope is dielectric loss, not "
                "conductivity, and a per-frequency R² near 1 does not distinguish them",
        )

    intercepts = np.asarray([s.intercept_S for s in slopes], dtype=float)
    C_ints = np.asarray([s.C_intercept_F for s in slopes], dtype=float)

    fit = GeometrySeriesFit(
        slopes=tuple(slopes),
        L_gap_cm=float(L_gap_cm), L_stripe_cm=float(L_stripe_cm),
        sigma_S_per_cm=sigma_med if flat else float("nan"),
        sigma_median_raw=sigma_med,
        sigma_spread=spread, slope_exponent=exponent,
        slope_frequency_independent=flat,
        intercept_S=float(np.median(intercepts[np.isfinite(intercepts)]))
        if np.isfinite(intercepts).any() else float("nan"),
        C_stray_F=float(np.median(C_ints[np.isfinite(C_ints)]))
        if np.isfinite(C_ints).any() else float("nan"),
        n_levels=n_levels, n_samples=len(members), span_ratio=span,
        min_replicates=min_reps,
        confound_verdict=confound_verdict, confound_correlation=confound_r,
        issues=tuple(issues),
    )
    logger.info("geometry_series_fit", summary=fit.describe(), usable=fit.usable)
    return fit
