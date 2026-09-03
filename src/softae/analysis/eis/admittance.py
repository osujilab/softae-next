"""Analyse the parallel core in admittance, where the measurand adds linearly.

Series impedances add in impedance; parallel admittances add in admittance. The
cell's parallel core — bulk conductance, geometric capacitance, fixture stray,
substrate leakage — is therefore *additive* in ``Y``, which turns nuisance removal
into subtraction of a constant rather than a nonlinear deconvolution (framework §1.2)::

    Y = G + jωC ,   G = 1/R_bulk

Two identities do most of the diagnostic work:

``tan δ = Re(Y)/Im(Y) = G/(ωC)``
    Falling with frequency ⇒ a conductance sits in parallel with the capacitance and
    the measurand exists. **Rising** ⇒ the dissipation is a series parasitic and the
    spectrum contains no parallel-conduction information at any frequency.

``C_app(f) = 1/(ω|Z''|)``
    Flat ⇒ an ideal geometric or fixture capacitance. Falling ⇒ a dispersive lossy
    dielectric, which carries its own parallel conductance in the same quadrature
    component as the sample's ionic conduction — so a scalar stray subtraction will
    not be enough.

.. note::
   ``tan δ = Z'/|Z''|`` and ``tan δ = Re(Y)/Im(Y)`` are the *same quantity* for a
   two-terminal impedance: ``Y = conj(Z)/|Z|²``, so ``Re Y = Z'/|Z|²`` and
   ``Im Y = −Z''/|Z|²`` and the ratio is ``Z'/(−Z'')``. The framework writes it both
   ways in §1.4 and §3.5.1. This is not a discrepancy — do not "fix" one of them.

All functions here take **physics-convention** complex impedance (``Im Z < 0`` for a
capacitive response). :meth:`softae.analysis.eis_data.EISResult.z_complex` already
returns that; a raw array does not, which is why the loader is where the sign is fixed.
"""

from __future__ import annotations

import numpy as np

#: Guard against dividing by an exactly-zero reactance or frequency.
_EPS = 1e-30

#: Shortest ``tan δ`` falling segment still fitted on its own terms rather than
#: discarded for the full band — see :func:`parallel_branch_window`, which uses it at
#: **both** its guards, and :func:`~softae.analysis.eis.gates.gate_tand_slope`, which
#: passes it to :func:`log_slope` so the fit and the window agree on what is enough.
#:
#: Four, because four is what this rig produces. Over the 1440-spectrum
#: ``20260811T023757Z_equilibration_characterization`` corpus (349 spectra physically
#: valid by ``min Re Z >= 0``) the loss-tangent gate rejected 203 — and **196 of those
#: rejections were false**. In every one of the 196 the falling segment held exactly
#: four points, one short of the threshold of five this constant replaces, so the
#: full-band fallback fired and substituted a fit over the whole sweep — CPE limb
#: included — for the segment the window exists to isolate. Measured on that
#: population: the segment reads a median ``−0.73``, a comfortable pass against
#: ``tand_slope_max = −0.3``; the full-band substitute reads a median ``+0.28``, a
#: reject. Nothing that passes today changes.
#:
#: **Not lower.** The false-reject population's own p10 is also 4 (median 4, p90 4),
#: so the corpus says nothing at all about three-point segments. Going below what was
#: measured would be an untargeted loosening riding on a targeted fix.
MIN_FALLING_SEGMENT_POINTS = 4


def to_admittance(Z: np.ndarray) -> np.ndarray:
    """``Y = 1/Z``, with non-finite results where ``Z`` vanishes."""
    Z = np.asarray(Z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(Z) > _EPS, 1.0 / np.where(np.abs(Z) > _EPS, Z, 1.0),
                        np.nan + 0j)


def conductance(Z: np.ndarray) -> np.ndarray:
    """``Re(Y)`` — the measurand, as a plain additive term."""
    return np.real(to_admittance(Z))


def susceptance(Z: np.ndarray) -> np.ndarray:
    """``Im(Y) = ωC`` — the quadrature component the conductance is projected out of."""
    return np.imag(to_admittance(Z))


def loss_tangent(Z: np.ndarray) -> np.ndarray:
    """``tan δ = Re(Y)/Im(Y)``, i.e. ``Z'/|Z''|`` (see the module note)."""
    Y = to_admittance(Z)
    im = np.imag(Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(im) > _EPS, np.real(Y) / np.where(np.abs(im) > _EPS, im, 1.0),
                        np.nan)


def apparent_capacitance(freq: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``C_app(f) = 1/(ω|Z''|)`` in farads."""
    f = np.asarray(freq, dtype=float)
    zim = np.abs(np.imag(np.asarray(Z, dtype=complex)))
    denom = 2.0 * np.pi * f * zim
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > _EPS, 1.0 / np.where(denom > _EPS, denom, 1.0), np.nan)


def log_slope(x: np.ndarray, y: np.ndarray, *, min_points: int = 5) -> float:
    """``d log y / d log x`` by least squares, or NaN when it cannot be taken.

    Requires strictly positive, finite pairs — a single non-positive value would
    make ``log10`` produce a NaN that silently poisons the whole polyfit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(min(x.size, y.size))
    if n == 0:
        return float("nan")
    x, y = x[:n], y[:n]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(ok.sum()) < int(min_points):
        return float("nan")
    try:
        return float(np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0])
    except Exception:
        return float("nan")


def parallel_branch_window(
    freq: np.ndarray, Z: np.ndarray, *, min_points: int = MIN_FALLING_SEGMENT_POINTS
) -> np.ndarray:
    """The falling segment of ``tan δ`` — the only band where parallel conduction shows.

    **This window is a deliberate deviation from framework §3.5.1**, which fits the
    loss-tangent slope globally across the whole sweep. That prescription is only
    valid for a bare parallel RC. The framework's *own* reference topology (§1.1) puts
    ``Z_CPE`` in series with the parallel core, and a blocking electrode makes ``tan δ``
    **non-monotonic**: it rises through the CPE-dominated low-frequency region, peaks,
    then falls toward the ``f⁻¹`` signature the gate is looking for.

    Measured on a well-formed synthetic blocking-cell spectrum, the global slope is
    −0.24 — which the specification's own ``tand_slope_max = −0.3`` **rejects**, while
    the same spectrum above its ``tan δ`` peak gives −0.83 and passes comfortably. So
    fitting globally would discard exactly the data this rig produces.

    One rule covers every shape: **the window runs from the global maximum to the first
    minimum above it**, which is the falling segment adjacent to the peak.

    ==========================  ==================  ===========================
    shape                       ``tan δ`` maximum   window
    ==========================  ==================  ===========================
    unimodal, peak in band      interior            peak → top of band
    U-shaped, peak below band   low endpoint        low end → the ``tan δ`` min
    monotone rising             top endpoint        empty ⇒ **full band**
    ==========================  ==================  ===========================

    Rows one and three are the same expression: when ``tan δ`` falls all the way to the
    top of the sweep the trough *is* the last point and the window is simply "at or
    above the peak", and when it rises monotonically the trough is the peak itself and
    the one-point window falls through to the full band.

    Row two is the rig's **normal** condition, not an edge case. A larger ``R_bulk``
    pushes the relaxation corner down in frequency, and past roughly ``R_bulk = 5×10⁷``
    the true peak has migrated *below* the sweep's 1.351 Hz floor. What remains in band
    is the falling limb plus the high-frequency rise where ``Z → R_series`` and
    ``Im Z → 0``, so ``tan δ → ∞`` — real physics, not noise. Anchoring on ``argmax``
    alone put the anchor on the low-frequency endpoint, ``f >= f_peak`` selected the
    whole band, and — this is the part that made it invisible — that mask has more than
    ``min_points`` entries, so it was returned by the *primary* path looking exactly
    like a legitimate above-peak selection while meaning the opposite. Measured at rig
    ``R_bulk``, including the rising tail dragged the slope from −0.85 to −0.49
    (5×10⁷) and −0.42 (10⁸) against a −0.3 threshold: still passing, but with a third
    of the margin, and degrading with ``R_bulk``.

    Falling back to the full band when the falling segment is too short is not a safety
    valve, it is part of the discriminator: a series parasitic has ``tan δ`` rising
    monotonically, so its maximum *is* the top of the band, and the full-band fit then
    correctly returns ``+1``. ``min_points`` sets *how* short is too short, at both
    guards below — enough usable points to locate a peak at all, and enough segment
    left afterwards to fit — and :data:`MIN_FALLING_SEGMENT_POINTS` records why it is
    four rather than five, which is the difference between this fallback firing on a
    series parasitic and firing on most of the rig's real spectra.

    Order-agnostic: the rig sweeps high→low, so the anchors are found on a sorted copy
    and applied as scalar frequency comparisons against the original array.
    """
    f = np.asarray(freq, dtype=float)
    tand = loss_tangent(Z)
    full = np.ones(f.size, dtype=bool)

    ok = np.isfinite(f) & np.isfinite(tand) & (f > 0) & (tand > 0)
    if int(ok.sum()) < int(min_points):
        return full

    ascending = np.argsort(f[ok])
    f_asc, tand_asc = f[ok][ascending], tand[ok][ascending]

    i_peak = int(np.argmax(tand_asc))
    i_trough = i_peak + int(np.argmin(tand_asc[i_peak:]))

    window = f >= float(f_asc[i_peak])
    if i_trough < tand_asc.size - 1:      # a rising tail sits above the trough
        window &= f <= float(f_asc[i_trough])
    return window if int(window.sum()) >= int(min_points) else full


def top_decade_window(freq: np.ndarray, *, min_points: int = 5) -> np.ndarray:
    """The highest decade of the sweep — where ``C_par`` dominates the reactance.

    ``C_app = 1/(ω|Z''|)`` only equals the geometric/fixture capacitance well *above*
    the relaxation corner, where the parallel branch has gone fully capacitive. Below
    it the reactance is dominated by the bulk arc and ``C_app`` is not a capacitance
    at all — so measuring flatness across the whole band asks the question in a region
    where the answer means nothing.
    """
    f = np.asarray(freq, dtype=float)
    full = np.ones(f.size, dtype=bool)
    ok = np.isfinite(f) & (f > 0)
    if not ok.any():
        return full
    window = ok & (f >= float(f[ok].max()) / 10.0)
    return window if int(window.sum()) >= int(min_points) else full


def model_free_r_bulk(Z: np.ndarray) -> float:
    """``R_bulk ≈ 1/max(Re Y)`` — an estimate that uses no circuit model.

    Framework §4.4 carries this alongside every fit. Agreement with the fitted
    ``R_series + R_bulk`` within a few percent says the model is not mis-specified at
    that operating point; divergence localises mis-specification to a particular
    sample or frequency regime. It is also the bootstrap the plateau-in-band check
    needs before any optimiser has run.

    .. note::
       It degrades as the plateau leaves the band, and **that degradation is the
       signal rather than a defect in the estimator**. Measured against synthetic
       spectra of known ``R_series + R_bulk``: within 9 % when the plateau sits
       squarely in band (``R_bulk`` ≈ 2 kΩ), 17 % low at 20 kΩ, 41 % low at 50 kΩ and
       75 % low at 200 kΩ — because by then the resistive plateau has been squeezed
       between the blocking onset and the relaxation corner and no model-free
       estimate can recover it. A large cross-check therefore means "the plateau is
       marginal here", which is exactly what §3.7 asks to be flagged.
    """
    G = conductance(Z)
    G = G[np.isfinite(G)]
    if G.size == 0:
        return float("nan")
    g_max = float(np.max(G))
    return 1.0 / g_max if g_max > 0 else float("nan")


def par_capacitance_estimate(freq: np.ndarray, Z: np.ndarray) -> float:
    """``C_par`` from the high-frequency end of ``C_app``.

    Uses the median of the top decade rather than a single point, because the top of
    the band is exactly where phase error lives (overhaul §2) and one bad point there
    would move the estimate a long way.

    .. warning::
       Do **not** predict the relaxation corner from the sample's permittivity.
       Framework §5.8 records that ``f = σ/(2πε₀ε_r)`` was wrong by ~50× here,
       because the real ``C_par`` is fixture-dominated (nF) rather than the sample
       dielectric (tens of pF). Use this fitted/measured value.
    """
    f = np.asarray(freq, dtype=float)
    C = apparent_capacitance(f, Z)
    ok = np.isfinite(f) & np.isfinite(C) & (f > 0) & (C > 0)
    if not ok.any():
        return float("nan")
    f_ok, C_ok = f[ok], C[ok]
    top = f_ok >= (f_ok.max() / 10.0)
    return float(np.median(C_ok[top])) if top.any() else float(np.median(C_ok))
