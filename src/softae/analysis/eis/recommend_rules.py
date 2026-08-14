"""The distribution math behind a threshold recommendation.

Five rule families, and nothing else: these functions know about sequences of floats
and about the physics of the quantity, but nothing about config keys, gates, records or
rendering. That separation is what lets them be tested on stated distributions — a rule
tested only through the pipeline that feeds it proves the pipeline, not the rule.

**Every rule here is distribution-free.** No normality is assumed anywhere, because a
metric like ``residual_rms_pct`` is heavy-tailed by construction and a ``mean ± kσ``
fence on it lands *inside* the bulk it was meant to enclose. Percentiles and gaps
survive that; moments do not.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

#: Fewest spectra either side of a candidate gap before it may be called two
#: populations rather than one outlier and a bulk.
MIN_GAP_SIDE = 3


def pct(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of ``values``."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


def upper_fence(values: Sequence[float]) -> float:
    """Tukey's upper fence, floored at a margin above the worst normal spectrum.

    ``P75 + 1.5 × IQR`` answers "outside the bulk", and on its own that is the whole
    rule for a spread population. The ``1.25 × P95`` floor is the safety half: a very
    tight population makes the Tukey fence tight too, which would arm a gate that
    rejects merely-mediocre spectra. Keeping a quarter of headroom above the 95th
    spectrum leaves room for a sample the run happened not to see.
    """
    p25, p75, p95 = (pct(values, q) for q in (25, 75, 95))
    return max(p75 + 1.5 * (p75 - p25), 1.25 * p95)


def lower_fence(values: Sequence[float]) -> float:
    """The mirror of :func:`upper_fence` for a metric where smaller is worse."""
    p5, p25, p75 = (pct(values, q) for q in (5, 25, 75))
    return min(p25 - 1.5 * (p75 - p25), 0.8 * p5)


def complement_fence(values: Sequence[float]) -> float:
    """Fence a metric bounded above by 1 through ``1 − metric``.

    Keeps the proposal inside ``[0, 1)`` without a clamp — a clamp would silently
    return 1.0 and arm a gate demanding a perfect fit — and it is the natural reading
    of R² anyway: "unexplained variance no worse than x".
    """
    return 1.0 - upper_fence([1.0 - float(v) for v in values])


def decade_margin(values: Sequence[float], *, upper: bool) -> float:
    """One decade of clearance beyond the observed population.

    For ``[quality] min_abs_z`` / ``max_abs_z``, which answer "is this a dead short or
    an open circuit?" rather than "is this 25 % worse than usual". The population spans
    ``10⁶–10⁸ Ω`` and a percentage fence around it would reject an ordinary wet film;
    a decade on each side is the honest translation of *implausible*.
    """
    return pct(values, 95) * 10.0 if upper else pct(values, 5) / 10.0


def count_minimum(values: Sequence[float], floor: int) -> float:
    """``max(physical_floor, ⌊P5⌋)`` — a count rule that cannot go below the physics.

    **A recommendation may never fall below the floor however good the evidence looks.**
    No distribution licenses fitting five parameters to five points, so a run in which
    every spectrum survived gating cleanly still recommends the floor rather than the
    5th percentile of a healthy population.
    """
    return float(max(int(floor), int(math.floor(pct(values, 5)))))


def gap_split(values: Sequence[float], *, min_gap: float,
              span: tuple[float, float]) -> float | None:
    """Split a bimodal metric at its widest interior gap, or ``None`` if unimodal.

    For a signed metric whose *physics* separates two populations rather than grading
    one. ``tand_slope`` is −1 for ideal parallel conduction and +1 for an ideal series
    parasitic; ``rho`` is ≈ −1 once the relaxation corner leaves the band. A percentile
    of a bimodal sample is meaningless — it lands wherever the mixing ratio happens to
    put it — so the honest answer when no gap exists is to hold the theory-anchored
    default and say why.

    Both sides must carry at least :data:`MIN_GAP_SIDE` points, or a single outlier and
    the bulk would read as two populations.
    """
    lo, hi = span
    xs = sorted(float(v) for v in values if lo <= float(v) <= hi)
    widest, midpoint = 0.0, None
    for i in range(len(xs) - 1):
        if i + 1 < MIN_GAP_SIDE or len(xs) - (i + 1) < MIN_GAP_SIDE:
            continue
        gap = xs[i + 1] - xs[i]
        if gap > widest:
            widest, midpoint = gap, (xs[i] + xs[i + 1]) / 2.0
    return midpoint if widest >= float(min_gap) else None


def physical_point_floor(model_name: str = "simpleSalt") -> int:
    """Fitted-parameter count + 3 — the fewest points that can support this model.

    Read off :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS` rather than written
    down, for the same reason ``r1_lower_bound_ohms`` reads its bound there: a count
    restated in a second place disagrees with the fitter after the first edit. Falls
    back to the shipped default for a model the registry does not know.
    """
    try:
        from softae.analysis.circuit_fitting import CIRCUIT_MODELS

        return int(len(CIRCUIT_MODELS[str(model_name)]["initial_guess"])) + 3
    except (ImportError, KeyError, TypeError, ValueError):
        return 8
