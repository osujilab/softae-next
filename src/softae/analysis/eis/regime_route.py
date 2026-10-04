"""The regime-A route: the film's own resistance, with the series chain excluded.

Slice 1 of ``docs/SubAgent docs/eis_regime_slice1_spec.md`` (rev 2, operator rulings of
2026-09-30). A regime-A spectrum is a blocking-electrode film — a bulk arc, then the
electrode's CPE spike — and **σ means K/R_b**: the arc's resistance only, never the series
elements (fixture, trace, contact) in front of it. Three estimates, of which the first is
reported:

``R_fit``
    A small fit of ``jωL + R_s + R_b∥CPE_geo + CPE_el`` on the screened points, reporting
    R_b. Validated with R_s present, unlike ``K·max(Re Y)`` (withdrawn: it reads the HF
    side of the arc and lands up to 1.5 dec off once ``R_s ≳ R_b``).
``R_mf = R_foot − R_HF``
    The model-free partner: Re Z at the foot of the arc, less the HF real-axis intercept of
    a circle through the arc above the foot.
``R_b_min``
    The passive-chain bound ``R_foot − min Re Z − |Im Z(foot)|·cot(α·90°)``: rigorous, so a
    fit below it has violated physics.

**A value only when the arc is resolved in band**: the pair agrees within ×1.5 (the ruled
50 %), the foot sits ≥ 1.5 dec below the top screened point, ``R_fit ≥ R_b_min``, and the
plateau's own headroom decision is ``value``. Otherwise a bound ``σ ≤ K/R_b_min`` where one
exists, otherwise *unavailable* — with ``σ ≥ K/R_foot`` logged, because a compressed
plateau cannot separate R_s from R_b and no estimator here pretends it can.

**Nothing here computes a conductivity.** ``tests/test_eis_universal_fit_route.py`` allows
σ arithmetic only in ``engine.py``; this module returns resistances and the engine divides.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.regime import B, RegimeVerdict, U
from softae.analysis.eis.report import (
    REGIME_A_FIT_RB,
    REGIME_A_FOOT_LOWER,
    REGIME_A_PASSIVE,
    decide_report_mode,
)

logger = structlog.get_logger(__name__)

#: The ruled agreement band between ``R_fit`` and ``R_mf`` (Q1: 50 %).
PAIR_RATIO = 1.5
#: Decades from the foot to the top screened point below which the arc is not resolved.
#: 1.5 leaves 10 false values on 2508 synthetic A, all compressed + HF artefact.
SPAN_MIN_DEC = 1.5
#: The :class:`~softae.analysis.eis.report.SigmaCeiling` reason when the passive bound
#: is not positive, so no ceiling separates R_b from the series chain. A ceiling reason,
#: never a ``regime_reason`` (that stays the classifier's) and never a basis.
SERIES_NOT_SEPARABLE = "series_not_separable"
#: ``regime_route_detail`` when the route ran but had no K to divide by.
NO_CELL_CONSTANT = "no_cell_constant"
#: ``regime_route_detail`` when the pair could not be compared (``R_mf`` or ``R_fit`` absent).
PAIR_UNAVAILABLE = "pair_unavailable"
#: ``regime_b_detail`` when the B route raised. The shadow never costs a spectrum, and an
#: exception is spelled apart from every outcome the route can decide (§3.1(a)).
B_ROUTE_ERROR = "route_error"
#: U reasons routed to regime B (slice 2 spec §1.1): the shapes the two-feature analysis
#: validated. ``floor_phase_ambiguous`` is unanalysed (and the mock rig's shape, Q7).
B_ROUTED_U_REASONS = ("no_cpe_drop", "non_monotone")

VALUE, BOUND, UNAVAILABLE = "value", "bound", "unavailable"

__all__ = ["REGIME_A_FIT_RB", "REGIME_A_FOOT_LOWER", "REGIME_A_PASSIVE",
           "SERIES_NOT_SEPARABLE", "NO_CELL_CONSTANT", "PAIR_UNAVAILABLE", "B_ROUTE_ERROR",
           "B_ROUTED_U_REASONS", "RegimeSettings", "RegimeAEstimates", "b_routed",
           "plateau_decision", "regime_a_estimates", "regime_a_kind", "regime_settings",
           "log_shadow", "route_basis"]


@dataclass(frozen=True)
class RegimeSettings:
    """``[eis] regime_aware`` and ``[eis] regime_b``. Each off unless its key is ``true``.

    The regime-B route is live only when **both** are on (:attr:`b_route`): ``regime_b``
    is slice 2's own rollback, so arming it cannot happen by arming slice 1.
    """

    enabled: bool = False
    regime_b: bool = False

    @property
    def b_route(self) -> bool:
        return self.enabled is True and self.regime_b is True


def _raw_flag(config: dict[str, Any], key: str) -> bool:
    raw = config.get(key)
    if raw is not None and not isinstance(raw, bool):
        logger.warning(f"eis_{key}_unparseable", value=repr(raw),
                       msg=f"[eis] {key} must be a boolean; the route stays off")
    return raw is True


def regime_settings(config: dict[str, Any] | None = None) -> RegimeSettings:
    """Read ``[eis] regime_aware`` and ``[eis] regime_b`` raw. Anything but ``True`` is off.

    Read raw like ``pregate_settings`` — no loader or toml edit, so ``config_hash`` does not
    move. **Arms only on** ``value is True``: the string ``"false"`` is truthy, and a typo
    must only ever leave the engine on today's route.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("eis", {}) or {}
        except Exception:      # noqa: BLE001 - an unreadable config must not stop a fit
            config = {}
    return RegimeSettings(enabled=_raw_flag(config, "regime_aware"),
                          regime_b=_raw_flag(config, "regime_b"))


def b_routed(verdict: RegimeVerdict | None) -> bool:
    """Whether the regime-B route applies to this verdict: B, or a routed-shape U."""
    if verdict is None:
        return False
    return verdict.label == B or (verdict.label == U and verdict.reason in B_ROUTED_U_REASONS)


@dataclass(frozen=True)
class RegimeAEstimates:
    """The three resistances, and what the decision needs alongside them."""

    R_fit: float = float("nan")
    R_fit_s: float = float("nan")      # the small fit's R_s — logged, never reported
    R_mf: float = float("nan")         # R_foot − R_HF
    R_hf: float = float("nan")
    R_b_min: float = float("nan")
    R_foot: float = float("nan")
    R_maxReY: float = float("nan")
    span: float = 0.0
    fit_rms: float = float("nan")

    @property
    def pair_ratio(self) -> float:
        """``max/min`` of the pair; NaN when either is missing or non-positive."""
        a, b = self.R_fit, self.R_mf
        if not (a == a and b == b and a > 0 and b > 0):
            return float("nan")
        return max(a, b) / min(a, b)


def _model(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    lL, lRs, lRb, lQg, ng, lQe, ne = p
    Zg = 1.0 / (1.0 / 10 ** lRb + 10 ** lQg * (1j * w) ** ng)
    return 1j * w * 10 ** lL + 10 ** lRs + Zg + 1.0 / (10 ** lQe * (1j * w) ** ne)


_LO = np.array([-12, -3, 0, -16, 0.5, -14, 0.3], dtype=float)
_HI = np.array([-2, 8, 11, -4, 1.0, -2, 1.0], dtype=float)


def small_fit(f: np.ndarray, Z: np.ndarray, R_guess: float) -> tuple[float, float, float]:
    """``(R_b, R_s, rms)`` from 6 starts of a 7-log-parameter fit, residual weighted 1/|Z|."""
    from scipy.optimize import least_squares

    w = 2 * np.pi * f
    wt = 1.0 / np.abs(Z)

    def residual(p: np.ndarray) -> np.ndarray:
        d = (_model(p, w) - Z) * wt
        return np.r_[d.real, d.imag]

    R0 = max(float(R_guess), 1.0)
    best = None
    for fc0, rs_frac in itertools.product((1e3, 3e4, 1e6), (1e-3, 0.3)):
        p0 = np.array([np.log10(2e-6), np.log10(rs_frac * R0),
                       np.log10(R0 * (1 - rs_frac) + 1),
                       np.log10(1 / (R0 * (2 * np.pi * fc0) ** 0.95)), 0.95,
                       np.log10(1 / (R0 * (2 * np.pi * 100.0) ** 0.7)), 0.7])
        p0 = np.clip(p0, _LO + 1e-6, _HI - 1e-6)
        try:
            r = least_squares(residual, p0, bounds=(_LO, _HI), x_scale="jac", max_nfev=400)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return float("nan"), float("nan"), float("nan")
    return (float(10 ** best.x[2]), float(10 ** best.x[1]),
            float(np.sqrt(2 * best.cost / f.size)))


def circle_hf_intercept(Z_arc: np.ndarray) -> float:
    """Kasa circle through the arc points; its HF real-axis intercept, or NaN."""
    x, y = Z_arc.real, -Z_arc.imag
    if x.size < 4:
        return float("nan")
    A = np.c_[x, y, np.ones_like(x)]
    try:
        D, E, Fc = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    xc, yc = -D / 2, -E / 2
    r2 = xc ** 2 + yc ** 2 - Fc
    if not (r2 > yc ** 2):
        return float("nan")
    return float(xc - math.sqrt(r2 - yc ** 2))


def passive_bound(Z: np.ndarray, i_foot: int) -> float:
    """``R_foot − min Re Z − |Im Z(foot)|·cot(α·90°)``; α from the floor's median phase."""
    alpha = -float(np.median(np.degrees(np.angle(Z[:3])))) / 90.0
    alpha = min(max(alpha, 0.05), 0.99)
    re_el_max = abs(Z.imag[i_foot]) / math.tan(math.radians(alpha * 90.0))
    return float(Z.real[i_foot] - Z.real.min() - re_el_max)


def regime_a_estimates(verdict: RegimeVerdict) -> RegimeAEstimates:
    """All three resistances off the verdict's screened points. A only."""
    s, i = verdict.screen, verdict.foot_index
    f, Z = s.f, s.Z
    R_foot = float(Z.real[i])
    R_hf = circle_hf_intercept(Z[i:])
    R_fit, R_fit_s, rms = small_fit(f, Z, R_foot)
    return RegimeAEstimates(
        R_fit=R_fit, R_fit_s=R_fit_s, R_hf=R_hf,
        R_mf=R_foot - R_hf if R_hf == R_hf else float("nan"),
        R_b_min=passive_bound(Z, i), R_foot=R_foot,
        R_maxReY=float(1.0 / np.max((1.0 / Z).real)), span=verdict.span, fit_rms=rms,
    )


def plateau_decision(verdict: RegimeVerdict, *, envelope: Any, cell: Any,
                     tand_headroom_mult: float) -> Any:
    """``decide_report_mode`` on the points at and above the foot — off the electrode tail.

    The engine's own decision takes its windowed minimum over the whole sweep, which on
    regime A lands in the CPE spike: a blocking electrode's tan δ is small *because it
    blocks*, not because the film's conductance is unresolved.
    """
    s, i = verdict.screen, verdict.foot_index
    return decide_report_mode(s.f[i:], s.Z[i:], envelope=envelope, cell=cell,
                              tand_headroom_mult=tand_headroom_mult)


def regime_a_kind(est: RegimeAEstimates, plateau_mode: str, *,
                  span_min: float = SPAN_MIN_DEC) -> tuple[str, str]:
    """``(value|bound|unavailable, detail)`` — spec §4's three outcomes, first match wins.

    A pair with a missing or non-positive side (no circle above a compressed foot, a
    failed small fit) reads ``pair_unavailable``: *could not compare* is not spelled as
    *compared and disagreed* (``SUBAGENT_RULES`` §3.1(a)). Either way it is no value.
    """
    ratio = est.pair_ratio
    pair = ((ratio <= PAIR_RATIO, "pair_disagrees") if ratio == ratio
            else (False, PAIR_UNAVAILABLE))
    checks = (
        pair,
        (est.span >= span_min, "arc_not_resolved_in_band"),
        (est.R_b_min > 0 and est.R_fit >= est.R_b_min, "fit_below_passive_bound"),
        (plateau_mode == VALUE, f"plateau_headroom_{plateau_mode or 'none'}"),
    )
    failed = [why for ok, why in checks if not ok]
    if not failed:
        return VALUE, "resolved_in_band"
    if est.R_b_min > 0:
        return BOUND, failed[0]
    return UNAVAILABLE, failed[0]


def _sigma_of(report: Any) -> float:
    """The one σ a report states: its value, or the bound's ceiling."""
    return float(report.value if report.mode == VALUE else report.upper_bound)


def log_shadow(channel: Any, verdict: RegimeVerdict, chosen: Any, today: Any, *,
               today_arc_gate: bool, estimates: RegimeAEstimates | None = None,
               plateau_mode: str = "", route_detail: str = "",
               sigma_value: float = float("nan"), sigma_sum: float = float("nan"),
               sigma_lower: float = float("nan"),
               b_fields: dict[str, Any] | None = None) -> None:
    """One ``sigma_regime_shadow`` line per classified spectrum, in both flag states.

    ``today_*`` is the flag-off report; ``regime_*`` is what the regime-A route says
    (empty for B/C/U, where today's route is the route). ``sigma_sum = K/(R_s+R_b)`` rides
    beside ``sigma_value`` because the in-band R_s may be partly film (spec §3).
    ``b_fields`` is the regime-B route's shadow (slice 2 spec §3.2), logged under a ``b_``
    prefix so its R_fit / R_mf never collide with route A's; no ``sigma_sum`` for B, which
    is meaningless when R_x ~ R_f.
    """
    est = estimates or RegimeAEstimates()
    b_log = {f"b_{k}": v for k, v in (b_fields or {}).items()}
    # A taken B route states its basis on the report itself; route_basis is A's table.
    b_active = bool(chosen.regime_active) and not verdict.is_a
    logger.info(
        "sigma_regime_shadow", channel=channel, regime=verdict.label,
        reason=verdict.reason, route_active=bool(chosen.regime_active),
        **verdict.screen.counts(), **verdict.features,
        R_fit=est.R_fit, R_fit_s=est.R_fit_s, R_mf=est.R_mf, R_b_min=est.R_b_min,
        R_foot=est.R_foot, R_maxReY=est.R_maxReY, fit_rms=est.fit_rms,
        plateau_mode=plateau_mode, route_detail=route_detail,
        sigma_value=sigma_value, sigma_sum=sigma_sum, sigma_lower=sigma_lower,
        today_mode=today.mode, today_sigma=_sigma_of(today),
        today_basis=(today.upper_bound_basis if today.mode != VALUE else today.R_basis),
        today_arc_gate=bool(today_arc_gate),
        regime_mode=chosen.regime_mode, regime_sigma=chosen.regime_sigma,
        regime_basis=(_stated_basis(chosen) if b_active
                      else route_basis(chosen.regime_mode, chosen.regime_sigma_lower)),
        **b_log,
    )


def _stated_basis(report: Any) -> str:
    """The basis of the number a report states: the ceiling's for a bound, else ``R_basis``."""
    bound = report.mode in ("bound", "bound_unqualified")
    return report.upper_bound_basis if bound else report.R_basis


def route_basis(kind: str, sigma_lower: float = float("nan")) -> str:
    """The basis token the A route reports for each outcome; ``""`` when not A.

    *unavailable* states the lower bound ``K/R_foot`` when it has one, and nothing — ``""``
    — when it does not (no cell constant): an absent number has no basis.
    """
    if kind == UNAVAILABLE:
        return REGIME_A_FOOT_LOWER if sigma_lower == sigma_lower else ""
    return {VALUE: REGIME_A_FIT_RB, BOUND: REGIME_A_PASSIVE}.get(kind, "")
