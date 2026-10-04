"""The regime-B route: the film arc's resistance, identified by the cell's capacitance.

Slice 2 of the regime-aware σ work (``docs/SubAgent docs/regime_b_slice2_spec.md`` §1–§3;
operator rulings Q1–Q10 of 2026-10-02). A regime-B spectrum is a series element
``R_x ∥ Q_x`` (C_x ≤ 3e-11 F — never the film), the film arc ``R_f ∥ Q_f`` whose
capacitance is the measured cell capacitance C_cell, and a blocking double layer ``Q_e``.

The fit is slice 1's ``small_fit`` with a shunt on its series resistor —
``jωL + R_x∥Q_x + R_f∥Q_f + Q_e``, 9 parameters, 12 starts, the film arc seeded **at
C_cell** — and its partner is :mod:`~softae.analysis.eis.feature_split`'s model-free
``R_mf = R_foot − R_plateau``. :func:`regime_b_kind` turns the two into one of four
outcomes (spec §3.2, first match wins):

==============  =============================================  ============================
outcome         when                                           resistance behind it
==============  =============================================  ============================
value           film arc identified, every gate passes         ``R_fit``  (σ = K/R_fit)
lower_bound     a closed in-band arc fails a value gate        ``max(R_fit, R_mf)`` (σ ≥ K/R)
upper_bound     no film arc in band, electrode negligible      ``Re Z_lo − R_x`` (σ ≤ K/R)
unavailable     anything else; ``detail`` says which           —
==============  =============================================  ============================

**Nothing is ever derived from R_x**, and **no C_cell means no value** — never a default
band (``[eis.instrument] c_cell_F`` is 3–6× the measured value). This module returns
resistances, an outcome and a detail token; σ arithmetic stays in ``engine.py``
(``tests/test_eis_universal_fit_route.py``). It is not wired into the engine yet (wave 2).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from softae.analysis.eis import feature_split as fs
from softae.analysis.eis.regime_route import NO_CELL_CONSTANT, PAIR_RATIO

# ── Outcomes and detail tokens (spec §3.2) ─────────────────────────────────────────
VALUE, LOWER_BOUND, UPPER_BOUND, UNAVAILABLE = (
    "value", "lower_bound", "upper_bound", "unavailable")

C_CELL_UNMEASURED = "c_cell_unmeasured"
FIT_FAILED = "fit_failed"
ARC_UNRESOLVED = "arc_unresolved"
SERIES_ELEMENT_UNIDENTIFIED = "series_element_unidentified"
NO_FILM_ARC = "no_film_arc"
ELECTRODE_NOT_NEGLIGIBLE = "electrode_not_negligible"
ARC_BELOW_CELL_BAND = "arc_below_cell_band"
RESOLVED_IN_BAND = "resolved_in_band"
INTERPHASE_BAND = "interphase_band"
ARC_NON_IDEAL = "arc_non_ideal"
PAIR_DISAGREES = "pair_disagrees"
R_F_SE_WIDE = "r_f_se_wide"
FIT_MULTIMODAL = "fit_multimodal"

#: ``RegimeBFit.cx_source``: C_x measured by the fit, or unresolved because Q_x pinned at
#: its lower bound (the shunt's apex lies above the band). Operator ruling 2026-10-03: an
#: unresolved C_x is NaN — never computed from ``1/(2π R_x f_top)`` — and rule 4 refuses
#: only on a *measured* C_x. The token is how a stored outcome says which it was.
CX_MEASURED, CX_UNRESOLVED = "measured", "unresolved"

# ── Gates ──────────────────────────────────────────────────────────────────────────
#: The film arc's ideality floor (3b/3c values fit n = 1.0; ch19's first read 0.75).
N_F_MIN = 0.8
#: Decades the film apex must sit inside the screened band at both ends.
APEX_MARGIN_DEC = 0.5
#: Precision gate on log10 R_f (1σ, Jacobian covariance).
SE_MAX_DEC = 0.1
#: Starts within this fraction of the best cost must agree on log10 R_f within ...
MULTIMODAL_COST_FRAC = 0.10
#: ... this many decades, or the fit is multimodal.
MULTIMODAL_SPREAD_DEC = 0.1
#: Fewer points than this between the valley and ``f_lo`` and the arc is unresolved.
MIN_ARC_POINTS = 5
#: The upper bound needs the fitted electrode at ``f_lo`` ≤ this fraction of Re Z_lo.
ELECTRODE_NEGLIGIBLE_FRAC = 0.1

# ── Fit (log10 except the n's; spec §1.2) ──────────────────────────────────────────
#: Parameter order: L, R_x, Q_x, n_x, R_f, Q_f, n_f, Q_e, n_e.
_LO = np.array([-12, 0, -16, 0.7, 2, -14, 0.5, -12, 0.3], dtype=float)
_HI = np.array([-2, 7, -8, 1.0, 11, -6, 1.0, -3, 1.0], dtype=float)
_IQX = 2
#: Q_x this close (dec) to its lower bound means the apex ran off the top of the band.
_PIN_TOL_DEC = 1e-3
_SEED_CX_APEX_HZ = (3e5, 3e6)
_SEED_RF_MULT = (1.0, 0.3, 3.0)
_SEED_NE = (0.6, 0.8)
_MAX_NFEV = 400

__all__ = [
    "VALUE", "LOWER_BOUND", "UPPER_BOUND", "UNAVAILABLE", "NO_CELL_CONSTANT",
    "C_CELL_UNMEASURED", "FIT_FAILED", "ARC_UNRESOLVED", "SERIES_ELEMENT_UNIDENTIFIED",
    "NO_FILM_ARC", "ELECTRODE_NOT_NEGLIGIBLE", "ARC_BELOW_CELL_BAND", "RESOLVED_IN_BAND",
    "INTERPHASE_BAND", "ARC_NON_IDEAL", "PAIR_DISAGREES", "R_F_SE_WIDE", "FIT_MULTIMODAL",
    "CX_MEASURED", "CX_UNRESOLVED", "RegimeBFit", "RegimeBEstimates", "RegimeBOutcome",
    "two_feature_model", "fit_two_feature", "regime_b_estimates", "regime_b_kind",
    "regime_b_route", "film_arc_resistance",
]


def two_feature_model(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``jωL + R_x∥Q_x + R_f∥Q_f + 1/(Q_e (jω)^n_e)`` on log10 parameters (spec §1.2)."""
    return _model_and_jac(p, w, jac=False)[0]


_LN10 = math.log(10.0)


def _model_and_jac(p: np.ndarray, w: np.ndarray, *, jac: bool = True):
    """The model and, if asked, its analytic ∂Z/∂p (columns in parameter order).

    Analytic rather than finite-difference: 9 parameters × 12 starts made the
    2-point Jacobian the dominant cost (``reduce work, not spread it``).
    """
    lL, lRx, lQx, nx, lRf, lQf, nf, lQe, ne = p
    s = 1j * w
    log_s = np.log(w) + 0.5j * math.pi
    Rx, Qx, Rf, Qf, Qe = 10 ** lRx, 10 ** lQx, 10 ** lRf, 10 ** lQf, 10 ** lQe
    Ysx, Ysf = Qx * s ** nx, Qf * s ** nf
    Zx, Zf = 1.0 / (1.0 / Rx + Ysx), 1.0 / (1.0 / Rf + Ysf)
    Ze = 1.0 / (Qe * s ** ne)
    Z = s * 10 ** lL + Zx + Zf + Ze
    if not jac:
        return Z, None
    Zx2, Zf2 = Zx * Zx, Zf * Zf
    J = np.stack([
        s * 10 ** lL * _LN10,
        Zx2 * _LN10 / Rx, -Zx2 * Ysx * _LN10, -Zx2 * Ysx * log_s,
        Zf2 * _LN10 / Rf, -Zf2 * Ysf * _LN10, -Zf2 * Ysf * log_s,
        -Ze * _LN10, -Ze * log_s,
    ], axis=1)
    return Z, J


@dataclass(frozen=True)
class RegimeBFit:
    """The canonicalised fit: ``x`` is the element with the higher apex frequency."""

    R_x: float
    Q_x: float
    n_x: float
    C_x: float
    cx_source: str               # CX_MEASURED, or CX_UNRESOLVED (then C_x is NaN)
    R_f: float
    Q_f: float
    n_f: float
    C_f: float
    f_apex_f: float
    Q_e: float
    n_e: float
    L: float
    se_log_R_f: float
    rms: float
    starts_spread_dec: float      # log10 R_f spread over starts within 10 % of best cost
    n_starts: int                 # starts that converged

    def z_electrode(self, f_hz: float) -> float:
        """``|Z_e(f)|`` of the fitted double layer."""
        return float(1.0 / (self.Q_e * (2 * math.pi * f_hz) ** self.n_e))


def _seeds(split: fs.FeatureSplit, c_cell: float, f_lo: float, z_lo: float):
    R_x0 = split.R_plateau if split.R_plateau == split.R_plateau and split.R_plateau > 1 \
        else max(split.re_z_lo, 1e3)
    R_mf = split.R_mf
    # No partner: the chord Re Z_lo − R_x, read here off R_x0 because R_plateau may not
    # exist (a valley at the top point); never below R_x0 itself.
    R_f_base = R_mf if R_mf == R_mf else max(split.re_z_lo - R_x0, R_x0)
    w_lo = 2 * math.pi * f_lo
    for f_cx, mult, ne in itertools.product(_SEED_CX_APEX_HZ, _SEED_RF_MULT, _SEED_NE):
        p0 = np.array([math.log10(2e-6), math.log10(R_x0),
                       math.log10(1.0 / (2 * math.pi * f_cx * R_x0)), 1.0,
                       math.log10(max(R_f_base * mult, 1e2)), math.log10(c_cell), 0.95,
                       math.log10(1.0 / (z_lo * w_lo ** ne)), ne])
        yield np.clip(p0, _LO + 1e-6, _HI - 1e-6)


def _canonical(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """``x`` with the higher-apex RQ element first; and whether it was swapped."""
    fa1 = fs.apex_frequency(10 ** x[1], 10 ** x[2], x[3])
    fa2 = fs.apex_frequency(10 ** x[4], 10 ** x[5], x[6])
    if fa1 >= fa2:
        return x, False
    y = x.copy()
    y[1:4], y[4:7] = x[4:7], x[1:4]
    return y, True


def fit_two_feature(f: np.ndarray, Z: np.ndarray, *, c_cell: float,
                    split: fs.FeatureSplit | None = None) -> RegimeBFit | None:
    """The 9-parameter fit from 12 starts, residual weighted 1/|Z|; ``None`` if none ran.

    Seeds (spec §1.2): R_x at the model-free plateau, C_x from an apex at 3e5 or 3e6 Hz,
    R_f at {1, 0.3, 3}× R_mf (else ``Re Z_lo − R_x``), **Q_f = C_cell** with n_f 0.95, and
    Q_e from ``|Z_lo|`` with n_e ∈ {0.6, 0.8}.
    """
    from scipy.optimize import least_squares

    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    split = split or fs.split_features(f, Z)
    w = 2 * np.pi * f
    wt = 1.0 / np.abs(Z)

    def residual(p: np.ndarray) -> np.ndarray:
        d = (_model_and_jac(p, w, jac=False)[0] - Z) * wt
        return np.r_[d.real, d.imag]

    def jacobian(p: np.ndarray) -> np.ndarray:
        J = _model_and_jac(p, w)[1] * wt[:, None]
        return np.r_[J.real, J.imag]

    runs = []
    for p0 in _seeds(split, c_cell, float(f[0]), float(abs(Z[0]))):
        try:
            r = least_squares(residual, p0, jac=jacobian, bounds=(_LO, _HI),
                              x_scale="jac", max_nfev=_MAX_NFEV)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(r.x)) and math.isfinite(r.cost):
            runs.append(r)
    if not runs:
        return None
    best = min(runs, key=lambda r: r.cost)
    x, swapped = _canonical(best.x)
    near = [_canonical(r.x)[0][4] for r in runs if r.cost <= best.cost * (1 + MULTIMODAL_COST_FRAC)]
    m, k = 2 * f.size, x.size
    rss = 2 * best.cost
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * rss / max(m - k, 1)
        se = float(math.sqrt(max(cov[1 if swapped else 4, 1 if swapped else 4], 0.0)))
    except (np.linalg.LinAlgError, ValueError):
        se = float("nan")
    R_x, Q_x, n_x = 10 ** x[1], 10 ** x[2], float(x[3])
    R_f, Q_f, n_f = 10 ** x[4], 10 ** x[5], float(x[6])
    pinned = x[_IQX] - _LO[_IQX] <= _PIN_TOL_DEC
    C_x = float("nan") if pinned else fs.brug_capacitance(R_x, Q_x, n_x)
    return RegimeBFit(
        R_x=float(R_x), Q_x=float(Q_x), n_x=n_x, C_x=float(C_x),
        cx_source=CX_UNRESOLVED if pinned else CX_MEASURED,
        R_f=float(R_f), Q_f=float(Q_f), n_f=n_f, C_f=fs.brug_capacitance(R_f, Q_f, n_f),
        f_apex_f=fs.apex_frequency(R_f, Q_f, n_f), Q_e=float(10 ** x[7]), n_e=float(x[8]),
        L=float(10 ** x[0]), se_log_R_f=se, rms=float(math.sqrt(rss / m)),
        starts_spread_dec=float(max(near) - min(near)), n_starts=len(runs),
    )


@dataclass(frozen=True)
class RegimeBEstimates:
    """Everything :func:`regime_b_kind` decides on, and what the shadow log records."""

    c_cell: float
    split: fs.FeatureSplit | None
    fit: RegimeBFit | None = None

    @property
    def R_fit(self) -> float:
        return self.fit.R_f if self.fit is not None else float("nan")

    @property
    def R_mf(self) -> float:
        return self.split.R_mf if self.split is not None else float("nan")

    @property
    def pair_ratio(self) -> float:
        return fs.pair_ratio(self.R_fit, self.R_mf)

    @property
    def c_ratio(self) -> float:
        """``C_f / C_cell``; NaN without a fit."""
        return self.fit.C_f / self.c_cell if self.fit is not None else float("nan")

    @property
    def R_lo(self) -> float:
        """``Re Z_lo − R_x``: the upper bound's resistance (spec §3.2 rule 5)."""
        if self.fit is None or self.split is None:
            return float("nan")
        return self.split.re_z_lo - self.fit.R_x

    @property
    def R_lower(self) -> float:
        """``max(R_fit, R_mf)``: the lower bound's resistance — the conservative side."""
        vals = [r for r in (self.R_fit, self.R_mf) if r == r and r > 0]
        return max(vals) if vals else float("nan")

    def shadow_fields(self) -> dict[str, float | str]:
        """The flat fields the wave-2 ``sigma_regime_shadow`` line adds for B (spec §3.2)."""
        ft, nan = self.fit, float("nan")
        return dict(
            R_x=ft.R_x if ft else nan, C_x=ft.C_x if ft else nan,
            cx_source=ft.cx_source if ft else "", R_fit=self.R_fit,
            C_f=ft.C_f if ft else nan, n_f=ft.n_f if ft else nan,
            f_apex=ft.f_apex_f if ft else nan, R_mf=self.R_mf,
            se_log_R_f=ft.se_log_R_f if ft else nan, c_cell=self.c_cell,
            c_ratio=self.c_ratio, R_lo=self.R_lo,
            starts_spread_dec=ft.starts_spread_dec if ft else nan,
            tand_foot=self.split.tand_foot if self.split else nan,
        )


def _valid_c_cell(c_cell: float | None) -> bool:
    return c_cell is not None and math.isfinite(c_cell) and c_cell > 0


def regime_b_estimates(f: np.ndarray, Z: np.ndarray, c_cell: float | None) -> RegimeBEstimates:
    """Split and fit the screened, fixture-corrected points.

    Without a measured C_cell nothing is fitted (there is nothing to seed or judge the
    film arc against) and the estimates carry ``c_cell = NaN``.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    split = fs.split_features(f, Z) if f.size >= 3 else None
    if not _valid_c_cell(c_cell) or split is None:
        return RegimeBEstimates(c_cell=float(c_cell) if _valid_c_cell(c_cell) else float("nan"),
                                split=split)
    fit = fit_two_feature(f, Z, c_cell=float(c_cell), split=split)
    return RegimeBEstimates(c_cell=float(c_cell), split=split, fit=fit)


def _apex_in_band(fit: RegimeBFit, split: fs.FeatureSplit) -> bool:
    lo = split.f_lo * 10 ** APEX_MARGIN_DEC
    hi = split.f_top * 10 ** -APEX_MARGIN_DEC
    return lo <= fit.f_apex_f <= hi


def regime_b_kind(est: RegimeBEstimates, *, has_cell: bool) -> tuple[str, str]:
    """``(outcome, detail)`` by spec §3.2's decision order, first match wins.

    ``has_cell`` says whether a cell constant K exists; the route divides by it later
    (``engine.py``), so without one there is nothing to state.
    """
    if not has_cell:
        return UNAVAILABLE, NO_CELL_CONSTANT
    if not _valid_c_cell(est.c_cell):
        return UNAVAILABLE, C_CELL_UNMEASURED
    fit, split = est.fit, est.split
    if fit is None or split is None:
        return UNAVAILABLE, FIT_FAILED
    if split.closed and split.n_below_valley < MIN_ARC_POINTS:
        return UNAVAILABLE, ARC_UNRESOLVED
    if fit.cx_source != CX_UNRESOLVED and not fs.is_series_element(fit.C_x):
        return UNAVAILABLE, SERIES_ELEMENT_UNIDENTIFIED
    if not (split.closed and _apex_in_band(fit, split)):
        electrode_ok = fit.z_electrode(split.f_lo) <= ELECTRODE_NEGLIGIBLE_FRAC * split.re_z_lo
        if est.R_lo > 0 and electrode_ok:
            return UPPER_BOUND, NO_FILM_ARC
        return UNAVAILABLE, NO_FILM_ARC if not est.R_lo > 0 else ELECTRODE_NOT_NEGLIGIBLE
    band = fs.cell_band_position(fit.C_f, est.c_cell)
    if band in (fs.BAND_BELOW, fs.BAND_UNKNOWN):
        return UNAVAILABLE, ARC_BELOW_CELL_BAND
    ratio = est.pair_ratio
    gates = (
        (band == fs.BAND_IN, INTERPHASE_BAND),
        (fit.n_f >= N_F_MIN, ARC_NON_IDEAL),
        (ratio == ratio and ratio <= PAIR_RATIO, PAIR_DISAGREES),
        (fit.se_log_R_f <= SE_MAX_DEC, R_F_SE_WIDE),
        (fit.starts_spread_dec <= MULTIMODAL_SPREAD_DEC, FIT_MULTIMODAL),
    )
    failed = [why for ok, why in gates if not ok]
    if not failed:
        return VALUE, RESOLVED_IN_BAND
    return LOWER_BOUND, failed[0]


@dataclass(frozen=True)
class RegimeBOutcome:
    """The route's answer: an outcome, why, and the one resistance it states.

    ``R_ohm`` is ``R_fit`` for a value, ``max(R_fit, R_mf)`` for a lower bound,
    ``Re Z_lo − R_x`` for an upper bound and NaN when unavailable. The engine turns it
    into σ = K/R_ohm with the matching direction.
    """

    kind: str
    detail: str
    R_ohm: float
    estimates: RegimeBEstimates


def regime_b_route(f: np.ndarray, Z: np.ndarray, *, c_cell: float | None,
                   has_cell: bool = True) -> RegimeBOutcome:
    """:func:`regime_b_estimates` then :func:`regime_b_kind`, with the stated resistance."""
    est = regime_b_estimates(f, Z, c_cell)
    kind, detail = regime_b_kind(est, has_cell=has_cell)
    R = {VALUE: est.R_fit, LOWER_BOUND: est.R_lower, UPPER_BOUND: est.R_lo}.get(kind, float("nan"))
    return RegimeBOutcome(kind=kind, detail=detail, R_ohm=float(R), estimates=est)


def film_arc_resistance(f: np.ndarray, Z: np.ndarray, c_cell: float | None) -> float | None:
    """R_f of an identified, closed, in-band film arc — the settle observable (spec §4).

    Rules 2–6 of :func:`regime_b_kind` only (identification, not the value gates of rule
    7, as in T11.33 Design A): an arc at or above the cell band still returns its R_f.
    ``None`` when the arc is unidentified — never a fallback to the series element.
    """
    est = regime_b_estimates(f, Z, c_cell)
    kind, _ = regime_b_kind(est, has_cell=True)
    return est.R_fit if kind in (VALUE, LOWER_BOUND) else None
