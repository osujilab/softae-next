"""Known-truth two-feature (regime-B) spectra for the slice-2 estimator tests.

``Z = jωL + R_x∥C_x + R_f∥CPE_f + CPE_e`` on the rig's 53-point grid, with 0.5 % complex
Gaussian noise and an optional HF artefact (the +40° phase rotation slice 1 uses). The
film arc is specified by its **Brug capacitance** ``C_f`` — the quantity the estimator
judges against C_cell — so ``Q_f = C_f^n · R_f^(n−1)``.

Every spectrum takes its own seed, so it is the same whichever test builds it (the
convention of :mod:`tests.eis_regime_synthetic`, whose grid this reuses). The grids are
deterministic subsamples of spec §5.2's grid, sized for tier 1: the full product is 6480
spectra at ~0.5 s each.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from tests.eis_regime_synthetic import NOISE, RIG_F, _noisy

#: The C_cell every synthetic spectrum is judged against (spec §5.2).
C_CELL = 2.5e-10
F_ASC = np.sort(RIG_F)


@dataclass(frozen=True)
class TwoFeature:
    """One spectrum's truth."""

    R_x: float = 1e5
    C_x: float = 3e-12
    R_f: float = 1e7
    c_ratio: float = 1.0          # C_f / C_CELL
    n_f: float = 1.0
    Q_e: float = 3e-8
    n_e: float = 0.8
    L: float = 0.0
    art: bool = False

    @property
    def C_f(self) -> float:
        return self.c_ratio * C_CELL

    @property
    def f_apex(self) -> float:
        Q = self.C_f ** self.n_f * self.R_f ** (self.n_f - 1)
        return 1.0 / (2 * math.pi * (self.R_f * Q) ** (1 / self.n_f))


def impedance(t: TwoFeature, freq: np.ndarray = F_ASC) -> np.ndarray:
    """The noiseless spectrum of ``t`` on ``freq``."""
    jw = 1j * 2 * np.pi * freq
    Q_f = t.C_f ** t.n_f * t.R_f ** (t.n_f - 1)
    Zx = 1.0 / (1.0 / t.R_x + t.C_x * jw)
    Zf = 1.0 / (1.0 / t.R_f + Q_f * jw ** t.n_f)
    Z = jw * t.L + Zx + Zf + 1.0 / (t.Q_e * jw ** t.n_e)
    if t.art:
        Z = Z * np.exp(1j * np.radians(40.0) * np.clip(np.log10(freq) - np.log10(2e4), 0, 1))
    return Z


def spectrum(t: TwoFeature, *, seed: int, noise: float = NOISE,
             freq: np.ndarray = F_ASC) -> tuple[np.ndarray, np.ndarray]:
    """``(f, Z)``, ascending, with noise."""
    return freq, _noisy(impedance(t, freq), seed, noise)


def screened(t: TwoFeature, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``spectrum`` through ``regime.screen_points`` — what the route receives."""
    from softae.analysis.eis.regime import screen_points

    s = screen_points(*spectrum(t, seed=seed))
    return s.f, s.Z


def _grid(axes: dict[str, tuple], seed0: int, every: int = 1, keep=lambda t: True):
    """Deterministic subsample (every ``every``-th point) of the product of ``axes``."""
    keys = list(axes)
    for i, vals in enumerate(itertools.product(*axes.values())):
        t = TwoFeature(**dict(zip(keys, vals)))
        if i % every == 0 and keep(t):
            yield t, seed0 + i


#: Truth apex at least this far (dec) inside the band — clear of the 0.5 dec gate.
_APEX_CLEAR_DEC = 0.7


def apex_clear_in_band(t: TwoFeature, freq: np.ndarray = F_ASC) -> bool:
    lo, hi = freq[0] * 10 ** _APEX_CLEAR_DEC, freq[-1] * 10 ** -_APEX_CLEAR_DEC
    return lo <= t.f_apex <= hi


#: R_f values whose arc apex (at C_f) sits ≥ 0.5 dec inside the band for every ratio.
_R_F_INBAND = (1e6, 1e7, 1e8)


def inband_grid():
    """C_f/C_cell ∈ {0.5, 1, 2}, n_f ≥ 0.9: arcs that should give values."""
    return _grid(dict(R_x=(3e4, 1e5, 3e5), C_x=(2e-12, 1e-11), R_f=_R_F_INBAND,
                      c_ratio=(0.5, 1.0, 2.0), n_f=(0.9, 1.0), Q_e=(1e-8, 1e-7),
                      n_e=(0.6, 0.8), art=(False, True)), 31000, every=7,
                 keep=apex_clear_in_band)


def interphase_grid():
    """C_f/C_cell ∈ {5, 20}: an arc above the cell band — never a value."""
    return _grid(dict(R_x=(3e4, 1e5, 3e5), C_x=(2e-12, 1e-11), R_f=(1e5, 1e6, 1e7),
                      c_ratio=(5.0, 20.0), n_f=(0.9, 1.0), Q_e=(1e-8, 1e-7),
                      n_e=(0.6, 0.8), art=(False, True)), 32000, every=7,
                 keep=apex_clear_in_band)


def non_ideal_grid():
    """n_f = 0.75 in band: below the ideality floor — never a value."""
    return _grid(dict(R_x=(3e4, 1e5, 3e5), C_x=(2e-12,), R_f=_R_F_INBAND,
                      c_ratio=(0.5, 1.0, 2.0), n_f=(0.75,), Q_e=(1e-8, 1e-7),
                      n_e=(0.6, 0.8)), 33000, every=3, keep=apex_clear_in_band)


def out_of_band_grid():
    """R_f 1e9…1e11 at the cell C: apex at or below ``f_lo`` — no film arc in band."""
    return _grid(dict(R_x=(3e4, 1e5, 3e5), C_x=(2e-12, 1e-11), R_f=(1e9, 1e10, 1e11),
                      c_ratio=(0.5, 1.0, 2.0), n_f=(0.9, 1.0), Q_e=(1e-8, 1e-7),
                      n_e=(0.6, 0.8)), 34000, every=9)


def series_film_grid():
    """C_x = 3e-10: the HF feature is itself film-like — never a value."""
    return _grid(dict(R_x=(3e4, 1e5, 3e5), C_x=(3e-10,), R_f=_R_F_INBAND,
                      c_ratio=(0.5, 1.0, 2.0), n_f=(1.0,), Q_e=(1e-8, 1e-7),
                      n_e=(0.6, 0.8)), 35000, every=3, keep=apex_clear_in_band)
