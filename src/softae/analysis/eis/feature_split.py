"""Two-feature spectra, read model-free: the shared primitives of regime B.

A regime-B spectrum (``docs/SubAgent docs/regime_b_two_feature_analysis.md``) is a
*series element* ``R_x ∥ C_x`` (C_x ≤ 3e-11 F, apex above the band), then the *film arc*
``R_f ∥ C_f`` whose capacitance is the cell's own parasitic capacitance C_cell, then a
blocking double layer. Read in tan δ = Re Z / −Im Z, walking down in frequency:

``plateau``
    the tan δ maximum above the valley — the series element's resistive plateau;
    ``R_plateau`` (Re Z there) is the model-free R_x.
``valley``
    the capacitive valley between the two features (−84…−88° on rung 3b), where C_cell
    dominates.
``foot``
    the tan δ maximum below the valley, where the film arc closes onto the real axis;
    ``R_foot`` is Re Z there.

The model-free film resistance is ``R_mf = R_foot − R_plateau``, and it **exists only if
tan δ(foot) ≥ 1**: an arc that never reached resistive dominance is not an arc in band, so
the partner's existence test is also the no-arc test (spec §1.3).

The windows are read from the data, never from a fit. This module is a numpy leaf: no
fit, no I/O, no config read and no conductivity (σ arithmetic is ``engine.py``'s alone).
:mod:`softae.analysis.eis.regime_b` builds the slice-2 estimator on it; the bench
discriminator scripts (``bench_discriminator_scripts.md`` §5) are meant to import it too,
so the API is general: arrays in, numbers and a frozen dataclass out.

**Unknown is never spelled as checked** (``SUBAGENT_RULES`` §3.1(a)): a non-finite
capacitance is *not* a series element and sits in no band (:data:`BAND_UNKNOWN`); a split
without a foot is not closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from softae.analysis.eis.regime import _med3

#: tan δ at the foot at or above which the film arc has closed to resistive dominance.
TAND_CLOSED = 1.0
#: The series element's shunt C ceiling. 218/218 isolated fits ≤ 2.9e-11 F (95 % ≤ 8e-12).
SERIES_C_MAX_F = 3e-11
#: The cell band is ``[C_cell/3, 3·C_cell]`` (operator ruling Q3, 2026-10-02).
CELL_BAND_FACTOR = 3.0
#: A valley must sit at least this far (dec of tan δ) below *both* its plateau and its
#: foot to count as the two-feature valley — the same fall regime.py asks of a foot.
VALLEY_FALL_DEC = 0.1
#: Points averaged (median) for the low-frequency Re Z.
N_LO = 3

BAND_BELOW, BAND_IN, BAND_ABOVE, BAND_UNKNOWN = "below", "in", "above", "unknown"

__all__ = ["TAND_CLOSED", "SERIES_C_MAX_F", "CELL_BAND_FACTOR", "VALLEY_FALL_DEC",
           "BAND_BELOW", "BAND_IN", "BAND_ABOVE", "BAND_UNKNOWN", "FeatureSplit",
           "tan_delta", "split_features", "brug_capacitance", "apex_frequency",
           "cell_band_position", "is_series_element", "pair_ratio"]


@dataclass(frozen=True)
class FeatureSplit:
    """The three windows of :func:`split_features` and the resistances read at them.

    Indices are into the ascending-frequency arrays passed in; ``-1`` means the window
    does not exist (no foot below the valley, no plateau above it). ``valley_two_sided``
    says whether the valley was found with a ≥ :data:`VALLEY_FALL_DEC` rise on both sides
    or fell back to the global tan δ minimum.
    """

    i_valley: int
    i_plateau: int
    i_foot: int
    valley_two_sided: bool
    f_valley: float
    f_plateau: float
    f_foot: float
    tand_valley: float
    tand_plateau: float
    tand_foot: float
    R_plateau: float
    R_foot: float
    re_z_lo: float          # median Re Z of the N_LO lowest-frequency points
    f_lo: float
    f_top: float

    @property
    def n_below_valley(self) -> int:
        """Points strictly between the valley and ``f_lo`` (inclusive of ``f_lo``)."""
        return max(self.i_valley, 0)

    @property
    def closed(self) -> bool:
        """The film arc closed in band: a foot exists and tan δ(foot) ≥ 1."""
        return self.i_foot >= 0 and self.tand_foot >= TAND_CLOSED

    @property
    def R_mf(self) -> float:
        """``R_foot − R_plateau`` when the arc closed and the difference is positive; else NaN."""
        if not self.closed or not (self.R_plateau == self.R_plateau):
            return float("nan")
        r = self.R_foot - self.R_plateau
        return r if r > 0 else float("nan")

    @property
    def R_chord(self) -> float:
        """``Re Z_lo − R_plateau``: the film's DC resistance plus whatever electrode Re Z
        remains at ``f_lo``. NaN when no plateau was found."""
        return self.re_z_lo - self.R_plateau if self.R_plateau == self.R_plateau else float("nan")


def tan_delta(Z: np.ndarray) -> np.ndarray:
    """``Re Z / −Im Z`` (the loss tangent of the impedance; ≥ 0 on screened points)."""
    Z = np.asarray(Z, dtype=complex)
    return Z.real / -Z.imag


def _valley_index(lt: np.ndarray) -> tuple[int, bool]:
    """The lowest interior tan δ minimum with a ≥ VALLEY_FALL_DEC rise on both sides.

    The spec's "tan δ minimum over the screened points" is the global minimum on a B
    spectrum, but on a single-arc film (route-A shape, forced through B) the global minimum
    is the electrode floor at ``f_lo`` and the two-feature valley is a *local* one above
    the arc. Requiring both sides to rise selects the valley between the two features in
    both shapes. Without such a valley (an insulator: tan δ falls monotonically toward
    ``f_lo``) the global minimum is returned with ``two_sided=False``.
    """
    n = lt.size
    best = -1
    for i in range(1, n - 1):
        if (lt[i + 1:].max() - lt[i] >= VALLEY_FALL_DEC
                and lt[:i].max() - lt[i] >= VALLEY_FALL_DEC
                and (best < 0 or lt[i] < lt[best])):
            best = i
    if best >= 0:
        return best, True
    return int(np.argmin(lt)), False


def split_features(f: np.ndarray, Z: np.ndarray) -> FeatureSplit:
    """Plateau, valley and foot of a two-feature spectrum, read off tan δ.

    ``f``/``Z`` must be ascending, physical points (``regime.screen_points``' output,
    fixture-corrected). tan δ is median-of-3 smoothed in log before any window is chosen;
    resistances are read off the raw Re Z at the chosen indices.
    """
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    if f.size < 3:
        raise ValueError("split_features needs at least 3 points")
    lt = np.log10(_med3(tan_delta(Z)))
    v, two_sided = _valley_index(lt)
    i_plateau = v + 1 + int(np.argmax(lt[v + 1:])) if v < f.size - 1 else -1
    i_foot = int(np.argmax(lt[:v])) if v > 0 else -1
    nan = float("nan")

    def at(i: int, arr: np.ndarray) -> float:
        return float(arr[i]) if i >= 0 else nan

    td = 10 ** lt
    return FeatureSplit(
        i_valley=v, i_plateau=i_plateau, i_foot=i_foot, valley_two_sided=two_sided,
        f_valley=at(v, f), f_plateau=at(i_plateau, f), f_foot=at(i_foot, f),
        tand_valley=at(v, td), tand_plateau=at(i_plateau, td), tand_foot=at(i_foot, td),
        R_plateau=at(i_plateau, Z.real), R_foot=at(i_foot, Z.real),
        re_z_lo=float(np.median(Z.real[:N_LO])), f_lo=float(f[0]), f_top=float(f[-1]),
    )


def brug_capacitance(R: float, Q: float, n: float) -> float:
    """Effective C of ``R ∥ CPE(Q, n)``: ``(Q·R^(1−n))^(1/n)`` (Brug). Equals Q at n = 1."""
    return float((Q * R ** (1.0 - n)) ** (1.0 / n))


def apex_frequency(R: float, Q: float, n: float) -> float:
    """Apex (−Z″ maximum) frequency of ``R ∥ CPE(Q, n)``: ``1/(2π (R·Q)^(1/n))``."""
    return float(1.0 / (2.0 * math.pi * (R * Q) ** (1.0 / n)))


def cell_band_position(C: float, c_cell: float, factor: float = CELL_BAND_FACTOR) -> str:
    """Where ``C`` sits against ``[c_cell/factor, factor·c_cell]``.

    :data:`BAND_UNKNOWN` when either capacitance is non-finite or non-positive — a band
    test that could not run is not a pass (``SUBAGENT_RULES`` §3.1(a)).
    """
    if not (math.isfinite(C) and math.isfinite(c_cell) and C > 0 and c_cell > 0):
        return BAND_UNKNOWN
    if C < c_cell / factor:
        return BAND_BELOW
    if C > c_cell * factor:
        return BAND_ABOVE
    return BAND_IN


def is_series_element(C_x: float, c_max: float = SERIES_C_MAX_F) -> bool:
    """``C_x ≤ c_max``: a shunt too small to be the cell's — not between the stripes,
    never the film. False for a non-finite or non-positive ``C_x``."""
    return bool(math.isfinite(C_x) and 0 < C_x <= c_max)


def pair_ratio(a: float, b: float) -> float:
    """``max/min`` of two positive resistances; NaN when either is missing or ≤ 0."""
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        return float("nan")
    return max(a, b) / min(a, b)
