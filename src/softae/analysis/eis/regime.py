"""Which physical regime a spectrum is in — before anyone decides what σ it supports.

Slice 1 of the regime-aware σ work (``docs/SubAgent docs/eis_regime_slice1_spec.md``).
A numpy leaf: no fit, no conductivity, no ``softae`` import. Two steps:

:func:`screen_points`
    Drop what is not a physical impedance (non-finite, the HF artefact band, ``Re Z ≤ 0``
    or ``Im Z ≥ 0``), and measure how coherent the sweep is. It runs on *all* points,
    fixture-corrected, **before** the gate mask: the gates drop non-physical points
    silently, and in observe mode an empty well still reaches the fit.
:func:`classify_regime`
    The v5 rule table. **A** is a blocking-electrode film: a low-frequency *foot* (the
    tan δ maximum that ends the bulk arc), then a monotone CPE decline to a floor in
    −75…−40°. **B** is a valley then a rise (rung 3b), **C** a genuinely open arc, and
    **U** anything this table cannot vouch for.

**U is not a regime, it is a refusal to name one.** A spectrum the rules cannot place is
never relabelled into the nearest class — "could not classify" must not be spelled with
the token of a checked answer (``SUBAGENT_RULES`` §3.1(a)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Thresholds (spec §2 "Margins"; each names what it costs when crossed) ──────────
#: Median |Δ² log₁₀|Z|| over all finite points. Real A max 0.039, empty well 0.139.
ROUGH_MAX = 0.08
#: Refuse when more than half the sweep is non-physical.
FRAC_BAD_MAX = 0.5
#: Fewer survivors than this and no shape can be read.
N_MIN = 15
#: tan δ at the foot below 1 ⇒ no resistive plateau at all (C).
T_P = 1.0
#: Largest rise of log tan δ walking foot → floor: ≥ 0.60 is a valley then a rise (B) …
EXC_B = 0.60
#: … and above 0.15 the decline is not monotone (U). Rev 1 was 0.20 (margin 0.006).
EXC_A = 0.15
#: Floor phase at or below −80° is still capacitive: an open arc (C).
PHI_C = -80.0
#: **Load-bearing.** A's floor (and its tan δ minimum) must sit at or above −75°; below
#: it an open arc reads as A. Margin 2.9° synthetic, 6.8° to real C.
PHI_A_LO = -75.0
PHI_A_HI = -40.0
#: The CPE decline below the foot must fall at least this many decades in tan δ.
DROP_A = 0.40
#: A tolerates at most 10 % non-physical points.
A_FRAC_BAD_MAX = 0.10
#: Median |Δ² phase| (degrees) over the screened points. The U guard for jagged phase
#: (operator ruling 2026-09-30, ``20260820T142833Z`` ch25); see :func:`classify_regime`.
PHASE_ROUGH_MAX = 1.5

#: The foot must have a ≥ 0.1 dec fall on both sides to count as interior.
_FOOT_FALL_DEC = 0.1
#: Interior feet are searched at f ≥ 30 Hz; the compressed fallback at f ≥ 300 Hz.
_FOOT_F_MIN_HZ = 30.0
_PLATEAU_F_MIN_HZ = 300.0
#: An interior foot above this is a plateau at the band top, not an arc in band.
_ARC_FOOT_MAX_HZ = 3e4

A, B, C, U = "A", "B", "C", "U"


@dataclass(frozen=True, eq=False)
class Screen:
    """What :func:`screen_points` kept, and what it dropped and why."""

    f: np.ndarray                 # ascending, physical points only
    Z: np.ndarray
    n_total: int = 0
    n_nan: int = 0
    n_hf_band: int = 0
    n_nonphys: int = 0
    rough: float = float("nan")
    phase_rough: float = float("nan")

    @property
    def n_ok(self) -> int:
        return int(self.f.size)

    @property
    def frac_bad(self) -> float:
        return (self.n_nan + self.n_nonphys) / max(self.n_total, 1)

    def counts(self) -> dict[str, Any]:
        return dict(n_total=self.n_total, n_nan=self.n_nan, n_hf_band=self.n_hf_band,
                    n_nonphys=self.n_nonphys, n_ok=self.n_ok, frac_bad=self.frac_bad,
                    rough=self.rough, phase_rough=self.phase_rough)


@dataclass(frozen=True, eq=False)
class RegimeVerdict:
    """The label, why, and the features that decided it.

    ``foot_index`` indexes ``screen.f``; ``-1`` means no foot was located.
    """

    label: str
    reason: str
    screen: Screen
    features: dict[str, float] = field(default_factory=dict)
    foot_index: int = -1
    foot_interior: bool = False

    @property
    def is_a(self) -> bool:
        return self.label == A

    @property
    def span(self) -> float:
        """Decades from the foot to the top screened point; 0 for a non-interior foot."""
        return float(self.features.get("span", 0.0))


def _med3(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    if x.size >= 3:
        y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


def _rise(seq: np.ndarray) -> float:
    """Largest rise met walking ``seq`` in order (value above its running minimum)."""
    if seq.size < 2:
        return 0.0
    return float(np.max(seq - np.minimum.accumulate(seq)))


def screen_points(freq: Any, Z: Any) -> Screen:
    """Spec §1, steps S1–S3, plus the two roughness measures. Never raises on bad input."""
    f = np.asarray(freq, dtype=float).ravel()
    Z = np.asarray(Z, dtype=complex).ravel()
    n_total = int(min(f.size, Z.size))
    f, Z = f[:n_total], Z[:n_total]

    finite = np.isfinite(f) & (f > 0) & np.isfinite(Z)               # S1
    f, Z = f[finite], Z[finite]
    order = np.argsort(f, kind="stable")
    f, Z = f[order], Z[order]
    n_nan = n_total - int(f.size)

    rough = (float(np.median(np.abs(np.diff(np.log10(np.abs(Z)), 2))))
             if f.size > 4 and np.all(np.abs(Z) > 0) else float("nan"))

    keep = np.ones(f.size, dtype=bool)
    if f.size:                                                        # S2
        hf_inductive = (f >= f.max() / 10.0) & (Z.imag >= 0)
        if hf_inductive.any():
            keep &= f < f[hf_inductive].min()
    n_hf_band = int((~keep).sum())
    bad = keep & ((Z.real <= 0) | (Z.imag >= 0))                       # S3
    ok = keep & ~bad
    fs, Zs = f[ok], Z[ok]
    phase_rough = (float(np.median(np.abs(np.diff(np.degrees(np.angle(Zs)), 2))))
                   if fs.size > 4 else float("nan"))
    return Screen(f=fs, Z=Zs, n_total=n_total, n_nan=n_nan, n_hf_band=n_hf_band,
                  n_nonphys=int(bad.sum()), rough=rough, phase_rough=phase_rough)


def locate_foot(f: np.ndarray, Z: np.ndarray) -> tuple[int, bool]:
    """Index of the LF-side tan δ maximum, and whether it is interior.

    Interior: the first local maximum going *up* from the floor (f ≥ 30 Hz) with a
    ≥ 0.1 dec fall on both sides. Otherwise the maximum over f ≥ 300 Hz — the compressed
    plateau, whose arc corner sits at or above the band top.
    """
    lt = np.log10(_med3(Z.real / -Z.imag))
    for i in range(1, lt.size - 1):
        if not (lt[i] >= lt[i - 1] and lt[i] >= lt[i + 1] and f[i] >= _FOOT_F_MIN_HZ):
            continue
        if (lt[i] - lt[i:].min() >= _FOOT_FALL_DEC and lt[i:].argmin() > 0
                and lt[i] - lt[:i].min() >= _FOOT_FALL_DEC):
            return i, True
    hi = f >= _PLATEAU_F_MIN_HZ
    if not hi.any():
        return int(np.argmax(lt)), False
    return int(np.flatnonzero(hi)[np.argmax(lt[hi])]), False


def _features(s: Screen) -> tuple[dict[str, float], int, bool]:
    lt = np.log10(_med3(s.Z.real / -s.Z.imag))
    ph = np.degrees(np.angle(s.Z))
    i, interior = locate_foot(s.f, s.Z)
    below = lt[: i + 1]
    j = int(np.argmin(below))
    feats = dict(
        foot_tand=float(10 ** lt[i]), foot_f_hz=float(s.f[i]),
        exc=_rise(below[::-1]), drop=float(lt[i] - below.min()),
        phase_floor=float(np.median(ph[:3])), phase_at_min=float(ph[j]),
        span=float(np.log10(s.f[-1] / s.f[i])) if interior else 0.0,
    )
    return feats, i, interior


def classify_regime(freq: Any, Z: Any) -> RegimeVerdict:
    """Spec §2's rule table, first match wins, plus the jagged-phase U guard.

    **The U guard (operator ruling 2026-09-30).** ``20260820T142833Z`` ch25 passes every
    v5 rule with a jagged phase trace; ``rough`` (on log|Z|) cannot see it because a
    jagged *phase* barely moves |Z|. ``phase_rough`` — median |Δ² phase| on the screened
    points — is the same statistic on the quantity that is actually jagged, applied only
    on the way to A (the guard can demote an A to U and nothing else).
    """
    s = screen_points(freq, Z)
    if not (s.rough <= ROUGH_MAX):
        return RegimeVerdict(U, "incoherent", s)
    if s.frac_bad > FRAC_BAD_MAX:
        return RegimeVerdict(U, "too_many_nonphysical", s)
    if s.n_ok < N_MIN:
        return RegimeVerdict(U, "too_few_points", s)

    feats, i, interior = _features(s)

    def verdict(label: str, reason: str) -> RegimeVerdict:
        return RegimeVerdict(label, reason, s, feats, i, interior)

    if feats["foot_tand"] < T_P:
        return verdict(C, "no_plateau")
    if feats["exc"] >= EXC_B:
        return verdict(B, "valley_then_rise")
    if feats["exc"] > EXC_A:
        return verdict(U, "non_monotone")
    if feats["phase_floor"] <= PHI_C:
        return verdict(C, "plateau_then_open_arc")
    if not (PHI_A_LO <= feats["phase_floor"] <= PHI_A_HI
            and feats["phase_at_min"] >= PHI_A_LO):
        return verdict(U, "floor_phase_ambiguous")
    if feats["drop"] < DROP_A:
        return verdict(U, "no_cpe_drop")
    if s.frac_bad > A_FRAC_BAD_MAX:
        return verdict(U, "a_too_many_nonphysical")
    if not (s.phase_rough <= PHASE_ROUGH_MAX):
        return verdict(U, "phase_incoherent")
    in_band_arc = interior and feats["foot_f_hz"] < _ARC_FOOT_MAX_HZ
    return verdict(A, "arc_then_cpe" if in_band_arc else "plateau_then_cpe")
