"""Synthetic regime-A/B/C spectra for the regime-aware σ tests (slice 1).

Ported from the slice-1 evidence scripts (``docs/SubAgent docs/evidence_regime_slice1/est.py``
and ``evidence_regime_characterisation/synth.py``). One deliberate change: every spectrum
takes its own ``seed``, so a spectrum is the same whichever test builds it and in whatever
order — the prototypes drew from one module-level generator, so their exact noise depends on
call order. The noise *level* (0.5 % complex Gaussian) and every element value are unchanged.

Used by ``tests/test_eis_regime.py`` and ``tests/test_eis_engine.py`` (two files, so it does
not go in ``conftest``).
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np

#: The rig's 53-point grid, 200 kHz → 1.351 Hz (descending, as acquired). Read off
#: ``tests/data/eis_regime/e148_rh47_20260930T013854Z_ch15.txt``.
RIG_F = np.array([
    200000.0, 159074.0, 126522.984, 100632.704, 80040.328, 63661.752, 50634.708,
    40273.376, 32032.272, 25477.538, 20264.092, 16117.47, 12819.367, 10196.152, 8109.723,
    6450.239, 5130.332, 4080.517, 3245.525, 2581.396, 2053.167, 1633.03, 1298.864,
    1033.079, 821.681, 653.541, 519.808, 413.44, 328.838, 261.548, 208.028, 165.459,
    131.601544, 104.67204, 83.253096, 66.217096, 52.667152, 41.889924, 33.318028,
    26.50019, 21.077482, 16.764418, 13.333931, 10.605421, 8.435244, 6.709148, 5.336261,
    4.244307, 3.375798, 2.685012, 2.13558, 1.698579, 1.351,
])

NOISE = 0.005


def _noisy(Z: np.ndarray, seed: int, noise: float = NOISE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return Z * (1 + noise * (rng.standard_normal(Z.size) + 1j * rng.standard_normal(Z.size)))


def _cpe(w: np.ndarray, Q: float, a: float) -> np.ndarray:
    return 1.0 / (Q * (1j * w) ** a)


def _par(Za, Zb):
    return Za * Zb / (Za + Zb)


def _rq(w: np.ndarray, R: float, f_apex: float, a: float) -> np.ndarray:
    """R ∥ CPE with its apex at ``f_apex``."""
    return _par(R, _cpe(w, 1.0 / (R * (2 * np.pi * f_apex) ** a), a))


def series_model(L: float, Rs: float, Rb: float, Qg: float, ng: float, Qe: float, ne: float,
                 w: np.ndarray) -> np.ndarray:
    """``jωL + R_s + R_b ∥ CPE_geo + CPE_el`` — the regime-A model, linear parameters."""
    Zg = 1.0 / (1.0 / Rb + Qg * (1j * w) ** ng)
    return 1j * w * L + Rs + Zg + 1.0 / (Qe * (1j * w) ** ne)


def regime_a_rs(shape: str, fc: float, a_el: float, n_g: float, Rs: float, Rb: float,
                L: float, art: bool, *, seed: int, freq: np.ndarray = RIG_F,
                noise: float = NOISE) -> np.ndarray:
    """One regime-A spectrum on the R_s grid (``est.py``'s ``synth``).

    ``shape`` is ``"inband"`` (spike from ``f_c/30``) or ``"compressed"`` (spike from
    300 Hz). ``art`` adds the +40° HF phase rotation ramping in log f from 20 to 200 kHz.
    """
    w = 2 * np.pi * freq
    fx = 300.0 if shape == "compressed" else fc / 30.0
    Qg = 1.0 / (Rb * (2 * np.pi * fc) ** n_g)
    Qe = 1.0 / (Rb * (2 * np.pi * fx) ** a_el)
    Z = series_model(L, Rs, Rb, Qg, n_g, Qe, a_el, w)
    if art:
        th = np.radians(40.0) * np.clip(np.log10(freq) - np.log10(2e4), 0, 1)
        Z = Z * np.exp(1j * th)
    return _noisy(Z, seed, noise)


#: The full R_s grid of spec §3 (2880 spectra): shape × f_c index × α_el × n_geo × R_s ×
#: R_b × L × artefact.
RS_GRID_AXES = (("compressed", "inband"), (0, 1), (0.6, 0.7, 0.8), (1.0, 0.85),
                (0.0, 1e3, 1e4, 3e4, 1e5), (1e3, 1e4, 1e5, 1e6), (0.0, 5e-6, 20e-6), (0, 1))
FC = {"compressed": (5e4, 1.5e5), "inband": (3e3, 1e4)}


def rs_grid():
    """Yield ``(params, Z)`` over the full R_s grid, each with its own seed."""
    for i, (shape, k, a, ng, Rs, Rb, L, art) in enumerate(itertools.product(*RS_GRID_AXES)):
        fc = FC[shape][k]
        p = dict(shape=shape, fc=fc, alpha=a, n_g=ng, Rs=Rs, Rb=Rb, L=L, art=art)
        yield p, regime_a_rs(shape, fc, a, ng, Rs, Rb, L, art, seed=20260930 + i)


def step1_regime_a(freq: np.ndarray = RIG_F):
    """Step-1 regime A: bulk corner fixed at 50 kHz, ideal C_geo, no R_s."""
    w = 2 * np.pi * freq
    grid = itertools.product((0.55, .6, .65, .7, .75, .8, .9), (0, 5e-6, 20e-6),
                             (1e3, 1e4, 1e5, 1e6), (100.0, 300.0, 1000.0))
    for i, (a, L, Rb, fx) in enumerate(grid):
        Cg = 1 / (2 * np.pi * Rb * 5e4)
        Q = 1 / (Rb * (2 * np.pi * fx) ** a)
        Z = 1j * w * L + _par(Rb, 1 / (1j * w * Cg)) + _cpe(w, Q, a)
        yield dict(alpha=a, L=L, Rb=Rb, fx=fx), _noisy(Z, 1000 + i)


def step1_regime_b(freq: np.ndarray = RIG_F):
    """Rung-3b mimic: HF feature (apex ~1 MHz) then an LF arc closing in band."""
    w = 2 * np.pi * freq
    grid = itertools.product((3e4, 1e5, 3e5), (1e7, 3e7), (20.0, 50.0, 150.0), (0.85, 0.95))
    for i, (R1, R2, f2, a2) in enumerate(grid):
        Z = _rq(w, R1, 1e6, 0.9) + _rq(w, R2, f2, a2)
        yield dict(R1=R1, R2=R2, f2=f2, alpha2=a2), _noisy(Z, 2000 + i)


def step1_regime_c(freq: np.ndarray = RIG_F):
    """A genuinely open arc: apex 0.3–2 decades below the sweep floor."""
    w = 2 * np.pi * freq
    grid = itertools.product((1e6, 1e7, 3e7), (0.3, 0.6, 1.0, 1.5, 2.0), (0.8, 0.9, 1.0))
    for i, (R, dec, a) in enumerate(grid):
        fap = float(np.min(freq)) * 10 ** (-dec)
        yield dict(R=R, dec=dec, alpha=a), _noisy(_rq(w, R, fap, a), 3000 + i)


def as_eis(freq: np.ndarray, Z: np.ndarray, channel: int = 15) -> SimpleNamespace:
    """The minimal ``EISResult`` shape ``analyze_spectrum`` reads."""
    Z = np.asarray(Z, complex)
    return SimpleNamespace(frequency=np.asarray(freq, float), z_real=Z.real,
                           z_imag_neg=-Z.imag, phase=np.degrees(np.angle(Z)),
                           z_magnitude=np.abs(Z), channel=channel)
