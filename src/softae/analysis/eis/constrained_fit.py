"""Constrained global fit: three parameters shared across a set of spectra.

**SHIPS INERT.** Nothing in ``src/`` calls this module, and that is deliberate. Wiring
it into a live path is a separate, separately-reviewable step.

Why it is a module of its own rather than a branch inside
:func:`~softae.analysis.eis.engine.analyze_spectrum`: a global fit is a **batch**
operation. ``analyze_spectrum(one_spectrum) -> SpectrumReport`` is per-spectrum by
construction and cannot express a parameter shared across N spectra. The batch logic
therefore lives here, and inference against a frozen artifact — which *is* per-spectrum
— is :func:`fit_frozen`.

The measured case for constraining at all (``constrained_fit_chain.md`` §1, and the
offline harness at ``docs/SubAgent docs/constrained_fit_harness/``): scoring ``R_sol →
K → σ`` against the four NIST KCl standards, an unconstrained per-spectrum fit of the
same circuit, with the same cleaning and the same optimiser, gives mean |σ error| in
the **thousands of percent**, while sharing three parameters across the set gives
~1.4 % in-sample. **The accuracy is bought by the constraint structure, not by the
circuit.**

The split, and why each line is where it is::

    PINNED    L                 CalibrationSet.L_lead_H -- per-channel, commissioned,
                                hardware_hash-keyed. Never fitted: overhaul F5 recorded
                                fitted inductances of 400-500 uH against a short blank's
                                true 4.18 uH.
    KNOWN     G_fixture(f)      CalibrationSet.G_fixture, consumed POINT BY POINT.
                                Never collapsed to a scalar, and never fitted alongside
                                a free leak -- see `fixture_admittance`.
    SHARED    Qg, ng, nd        fitted once across the set
    PER WELL  R0, R1, Qd        fitted per spectrum
    REPORTED  R_sol = R0 + R1   unconditionally

**All three shared parameters are load-bearing, and this was measured.** Sharing only
``Qg`` and ``nd`` — pinning the geometric element to an ideal capacitor, ``ng = 1`` —
does not degrade gracefully: it gives ``+4.3e8 %`` on the most conductive standard.
Sharing two of the three gives 4.93 % / 7.38 % against 1.55 %.

Topology. This is :data:`~softae.analysis.eis.models.EIS_CIRCUITS`'s
``blocking_coplanar_L`` (``L0-R0-CPE0-p(R1,C0)``) with **two deliberate differences**,
and it is defined in closed form here rather than reused from that registry:

1. ``C0`` becomes a CPE with a *fitted, shared* exponent. The registry's pure capacitor
   is the ``ng = 1`` case the paragraph above measures as catastrophic.
2. A **known** shunt branch ``Y_shunt(f)`` sits across the path. The registry's circuit
   strings are fitted through ``impedance.py``'s ``wrapCircuit``, which has no way to
   express either a frequency-resolved constant or a parameter shared across spectra.

So the registry model is not reusable *as a registry model* here. Its docstring
(*"SHIPS UNUSED — L must be pinned from a short blank"*) names exactly this design, and
the pinning it asks for is what :attr:`ConstrainedSpectrum.pinned_l_H` carries.

**One preparation step is a precondition, and it is not optional.** Spectra must reach
this module with the contiguous ``Im Z > 0`` run at the top of the band already removed
— :func:`~softae.analysis.eis.gates.gate_hf_inductive`'s job, and it already ships.
Measured on the four NIST standards, leaving those points in takes the in-sample
``R_sol`` error from **2.9 % to 6295 %**, while the short-blank correction, the |Z|
window and the linear-KK truncation are together worth about 1.3 percentage points. The
asymmetry has a cause: a blocking cell has no inductance of its own, so an inductive
point is the fixture, and the optimiser parks it in ``Qg`` — a **shared** parameter, so
a handful of bad points at one end of one spectrum corrupts the artifact for the entire
set. Sharing parameters shares contamination too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from scipy.optimize import least_squares

logger = structlog.get_logger(__name__)

TWO_PI = 2.0 * math.pi

#: Fitted parameters, in vector order. ``L`` is *not* here: it is pinned, and a pinned
#: quantity carried in the parameter vector is one bounds change away from being fitted.
PARAM_NAMES: tuple[str, ...] = ("R0", "R1", "Qg", "ng", "Qd", "nd")
SHARED_PARAMS: tuple[str, ...] = ("Qg", "ng", "nd")
PER_SPECTRUM_PARAMS: tuple[str, ...] = ("R0", "R1", "Qd")

#: ``name -> (lower, upper)``. Physical, not tuned: ``ng``/``nd`` are CPE exponents in
#: [0, 1] narrowed to the range a geometric or a blocking element can occupy.
BOUNDS: dict[str, tuple[float, float]] = {
    "R0": (1e-3, 1e11),
    "R1": (1e-3, 1e11),
    "Qg": (1e-15, 1e-3),
    "ng": (0.50, 1.00),
    "Qd": (1e-14, 1e-1),
    "nd": (0.30, 1.00),
}

#: ``xtol``/``ftol``/``gtol`` for every fit in this module.
#:
#: **Not a preference, and scipy's default is a wrong answer that looks converged.**
#: Measured on the stacked four-standard fit, everything else held fixed:
#:
#: ======  ========  ========  ======  ================================
#: tol     cost      status    nfev    mean |R_sol error| vs AMP
#: ======  ========  ========  ======  ================================
#: 1e-8    3.60      3         63      **8014 %**
#: 1e-10   0.337     3         39      1.55 %
#: 1e-12   0.332     3         44      1.56 %
#: 1e-14   0.332     3         47      1.56 %
#: ======  ========  ========  ======  ================================
#:
#: ``status = 3`` is *"ftol termination"* — the optimiser's own report of a converged
#: fit — in **every** row, the 8014 % one included, and no parameter is railed there
#: either. That is ``SUBAGENT_RULES`` §3.1 in its purest form: an instrument returning
#: the shape of a pass for a different question. The knee is between 1e-8 and 1e-10 and
#: the extra decades are free (47 ``nfev`` against 63), so the constant sits at the far
#: side of it rather than on it.
#:
#: The tolerance is therefore a module constant rather than a keyword with a permissive
#: default, and :func:`fit_shared` logs a warning when a caller loosens it.
#:
#: Distinct from :data:`~softae.analysis.eis.fitter.DEFAULT_FIT_TOL` (1e-10) because the
#: problems are different: that one fits 5 parameters against one spectrum, this one
#: fits ``3 + 3N`` against N stacked spectra, where the shared parameters are informed
#: only by the *differences* between spectra and the gradient in them is correspondingly
#: shallow.
CONSTRAINED_FIT_TOL = 1e-14

#: Iteration ceilings. Generous for the same reason ``fitter.DEFAULT_MAX_NFEV`` is:
#: a failure to converge should be reported, not manufactured by too small a budget.
MAX_NFEV_SINGLE = 40_000
MAX_NFEV_GLOBAL = 120_000

#: Multipliers on the conductance estimate of ``R_sol`` used to seed :func:`fit_frozen`.
#:
#: **Multi-start is required, and its necessity is diagnostic.** Once the artifact is
#: good the frozen fit is seed-insensitive and every multiplier lands in the same
#: basin; when the artifact is *not* good the seed decides the answer outright. So a
#: wide spread across multipliers is itself evidence about the artifact, which is why
#: :attr:`WellFit.seed_spread_pct` is reported rather than discarded.
R_SOL_SEED_MULTIPLIERS: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: The only parameters re-seeded from the spectrum itself when :func:`fit_shared` walks
#: the set building its starting point. Everything else is **carried forward from the
#: previous spectrum's fit** — a continuation, not N independent fits.
#:
#: **Measured, and it is the difference between 1.4 % and 28 000 %.** Re-seeding ``Qd``
#: per spectrum as well (the obvious reading of "seed each spectrum from itself") drops
#: the stacked fit into a different basin: on the four NIST standards it converges at
#: cost 3.53 after 12 451 ``nfev`` with ``Qg`` three decades high and ``nd`` near its
#: lower bound, against cost 0.33 after 47 ``nfev`` for the continuation. Both runs
#: report a completed optimisation. ``R0``/``R1`` must be re-seeded because they span
#: the conductivity ladder the set exists to cover; the CPE parameters must not, because
#: their continuity across the set is the information the shared fit is built from.
CONTINUATION_RESET: tuple[str, ...] = ("R0", "R1")

#: How many spectra in the fitting set must show a **resolved** geometric arc before
#: the shared parameters are identifiable. See :func:`check_admissible`.
MIN_ARC_RESOLVED = 2


# ── The known fixture shunt ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ShuntTable:
    """``Y_shunt(f)`` evaluated on one spectrum's frequencies, with its own accounting.

    ``n_held`` is not decoration. :meth:`FixtureConductance.at
    <softae.analysis.eis.calibration.FixtureConductance.at>` returns NaN outside the
    band it was measured in, correctly and on purpose, and this module has to *decide*
    what to do with that rather than inherit whichever answer happens to fall out.
    Holding the endpoint is a choice with a cost, so the count of points it was applied
    to travels with the values.
    """

    y: np.ndarray
    n_points: int
    n_held: int = 0
    coverage_hz: tuple[float, float] = (float("nan"), float("nan"))
    beyond_coverage: str = "hold"

    @property
    def held_fraction(self) -> float:
        return self.n_held / self.n_points if self.n_points else float("nan")


def fixture_admittance(
    freq_hz: np.ndarray,
    conductance: Any,
    c_stray_F: float = 0.0,
    *,
    beyond_coverage: str = "hold",
) -> ShuntTable:
    """``Y_shunt(f) = G_fixture(f) + jωC_stray``, evaluated per frequency point.

    **Consumed as a table, never collapsed to a scalar.** On this fixture the shunt's
    real part is dielectric loss — seven tied open blanks give ``d ln G/d ln f`` between
    +0.87 and +1.04 — so a single number describes the fixture at exactly one frequency
    and is wrong everywhere else.

    **And never fitted while also being supplied.** Handing the optimiser a free leak
    resistance *and* the measured shunt is degenerate: measured on the 10 kΩ rung the
    combination gives ``+1.6e6 %`` with the fitted ``R_leak`` running to 2.6e11 Ω. That
    is why there is no ``Rleak`` in :data:`PARAM_NAMES`. AMP_v1 fits a scalar leak
    because it has no open blank; we have one, and a measurement beats a free parameter.

    *beyond_coverage* decides the points the conductance table does not cover — a real
    case, not a corner one: ``G_fixture`` on ch25 spans 14.3 Hz–200 kHz while the
    reference-resistor sweeps start near 1.2 Hz, so 10–11 points of each fall below it.

    ``"hold"``
        Clamp to the nearest measured endpoint. **The default, and measured against
        ``"drop"`` on the three reference resistors** — mean |error| 1.03 % (worst
        1.70 %) holding, against 2.15 % (worst 5.24 %) dropping, and leave-one-out
        0.54 % against 4.88 %. The dropped points are the *bottom* decade of the sweep,
        which is where a blocking cell's information about ``R_sol`` lives, so dropping
        them costs more than the extrapolation does. It is also defensible in direction:
        below the table ``G ∝ ω`` is still falling, so holding the lowest measured ``G``
        *overstates* the shunt there. The count is reported via
        :attr:`ShuntTable.n_held` and logged, so it can never read as coverage that was
        actually measured.
    ``"drop"``
        Report the uncovered points via ``n_held`` and emit NaN for them, leaving the
        caller to drop them. The honest option when the excursion is large — and the
        one to reach for rather than widening ``"hold"``'s reach silently.
    ``"zero"``
        Treat the uncovered points as having no fixture shunt. **Only** legitimate when
        the open blank was judged unusable, which is itself positive evidence that the
        shunt is negligible.
    """
    f = np.asarray(freq_hz, dtype=float)
    y = np.zeros(f.shape, dtype=complex)
    n_held = 0
    lo = hi = float("nan")

    gf = np.asarray(getattr(conductance, "freq_hz", ()) or (), dtype=float)
    gs = np.asarray(getattr(conductance, "G_S", ()) or (), dtype=float)
    if gf.size and gf.size == gs.size:
        order = np.argsort(gf)
        gf, gs = gf[order], gs[order]
        lo, hi = float(gf[0]), float(gf[-1])
        outside = (f < lo) | (f > hi)
        n_held = int(np.count_nonzero(outside))
        g = np.interp(np.log10(f), np.log10(gf), gs, left=gs[0], right=gs[-1])
        if beyond_coverage == "drop":
            g = np.where(outside, np.nan, g)
        elif beyond_coverage == "zero":
            g = np.where(outside, 0.0, g)
        elif beyond_coverage != "hold":
            raise ValueError(f"unknown beyond_coverage policy {beyond_coverage!r}")
        y = y + g
    elif gf.size:
        raise ValueError("G_fixture freq_hz and G_S differ in length")

    stray = float(c_stray_F)
    if stray == stray and stray:
        y = y + 1j * TWO_PI * f * stray

    if n_held:
        logger.info(
            "eis_constrained_fit_shunt_beyond_coverage",
            n_held=n_held, n_points=int(f.size), policy=beyond_coverage,
            coverage_hz=(lo, hi),
            msg=("G_fixture has no coverage for these points; the policy named here "
                 "is what was applied to them — not a measurement"),
        )
    return ShuntTable(y=y, n_points=int(f.size), n_held=n_held,
                      coverage_hz=(lo, hi), beyond_coverage=beyond_coverage)


# ── The model ────────────────────────────────────────────────────────────────

def _z_cpe(omega: np.ndarray, q: float, n: float) -> np.ndarray:
    return 1.0 / (q * (1j * omega) ** n)


def model_impedance(
    freq_hz: np.ndarray,
    params: dict[str, float],
    y_shunt: np.ndarray,
    pinned_l_H: float,
) -> np.ndarray:
    """``Z(f) = jωL + [ (R0 + (R1 ∥ CPE_g) + CPE_d) ∥ Y_shunt ]``.

    ``L`` and ``y_shunt`` are inputs, not unknowns — the whole point of the design.
    """
    omega = TWO_PI * np.asarray(freq_hz, dtype=float)
    z_geo = _z_cpe(omega, params["Qg"], params["ng"])
    z_path = (params["R0"]
              + 1.0 / (1.0 / params["R1"] + 1.0 / z_geo)
              + _z_cpe(omega, params["Qd"], params["nd"]))
    y_total = 1.0 / z_path + np.asarray(y_shunt, dtype=complex)
    return 1j * omega * float(pinned_l_H) + 1.0 / y_total


def conductance_r_sol(freq_hz: np.ndarray, z: np.ndarray) -> float:
    """``1 / max Re(Y)`` — the seed estimate, and a model-free sanity value."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1.0 / np.nanmax((1.0 / np.asarray(z, dtype=complex)).real))


# ── One spectrum, ready to fit ───────────────────────────────────────────────

@dataclass(frozen=True)
class ConstrainedSpectrum:
    """One cleaned spectrum with everything the fit treats as known already attached.

    The spectrum is expected in the **corrected** domain — whatever
    :func:`~softae.analysis.eis.fixture.apply_series_correction` was going to do has
    been done. ``pinned_l_H`` is then whatever inductance remains to be modelled, which
    is ``0.0`` when the series correction already consumed ``L_lead_H`` and
    ``L_lead_H`` itself when it did not. Both are "L is pinned from commissioning";
    only the arithmetic differs, and ``l_source`` records which.
    """

    label: str
    freq_hz: np.ndarray
    z: np.ndarray
    y_shunt: np.ndarray
    pinned_l_H: float = 0.0
    channel: int | None = None
    l_source: str = ""
    #: Known answer, when there is one (a standard, a reference resistor). Never used
    #: by the fit — only by :func:`holdout_report`.
    reference_ohm: float = float("nan")
    shunt: ShuntTable | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.freq_hz).size)
        for name in ("z", "y_shunt"):
            if int(np.asarray(getattr(self, name)).size) != n:
                raise ValueError(
                    f"{self.label}: {name} has "
                    f"{np.asarray(getattr(self, name)).size} points against "
                    f"{n} frequencies")


def arc_is_resolved(freq_hz: np.ndarray, z: np.ndarray) -> bool:
    """Whether the geometric arc's **low-frequency flank** is inside the measured band.

    The observable is an interior local minimum in ``-Im Z`` — the valley where the
    falling blocking tail crosses the rising bulk arc. Below that crossover the sweep
    sees only the blocking electrode and nothing constrains ``Qg`` or ``ng``. The
    crossover moves **up** in frequency with conductivity, so on a conductivity ladder
    the *dilute* members are the ones that show it: on the four NIST standards
    ``kcl_45uS`` and ``kcl_84uS`` do (valleys at 124 kHz, near the top of their sweeps)
    and ``kcl_1413uS``/``kcl_4500uS`` do not.

    Model-free on purpose. An admissibility test that had to run the fit first would be
    answering the question it exists to gate.

    **What this is not.** It is a *necessary* condition validated against one corpus,
    not a proof of identifiability. The stricter reading — require the arc's **apex** in
    band — was tried and rejected on measurement: the apex is out of band for all four
    NIST standards, including the two whose presence makes the artifact work, so that
    criterion refuses the very fit that scores 1.43 %. Conversely a synthetic spectrum
    can show the crossover with its apex still a decade above the sweep. The validation
    that this reading is the useful one is in :func:`check_admissible`.
    """
    f = np.asarray(freq_hz, dtype=float)
    im = -np.asarray(z, dtype=complex).imag
    ok = np.isfinite(im) & np.isfinite(f)
    if int(np.count_nonzero(ok)) < 3:
        return False
    y = im[ok][np.argsort(f[ok])]
    return bool(np.any((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])))


# ── Admissibility ────────────────────────────────────────────────────────────

class InadmissibleFitSet(ValueError):
    """The fitting set cannot identify the shared parameters. A refusal, not a failure."""


@dataclass(frozen=True)
class Admissibility:
    """Whether a set of spectra can identify ``Qg, ng, nd`` — and if not, why not."""

    admissible: bool
    reason: str
    n_spectra: int = 0
    n_arc_resolved: int = 0
    arc_resolved: tuple[str, ...] = ()
    n_free: int = 0
    n_residuals: int = 0

    def describe(self) -> str:
        return (f"{'admissible' if self.admissible else 'INADMISSIBLE'}: {self.reason} "
                f"({self.n_spectra} spectra, {self.n_arc_resolved} arc-resolved, "
                f"{self.n_free} free parameters against {self.n_residuals} residuals)")


def check_admissible(
    spectra: list[ConstrainedSpectrum], *, min_arc_resolved: int = MIN_ARC_RESOLVED
) -> Admissibility:
    """Can this set identify the shared three? **Measured, and it refuses.**

    Leave-one-out over the four NIST standards, with this gate deliberately bypassed so
    the bad artifacts could be looked at rather than assumed:

    ===============  ==========  =============  ===========
    fold (held out)  fitted Qg   R_sol error    this gate
    ===============  ==========  =============  ===========
    ``kcl_45uS``     9.04e-08    **−84.06 %**   REFUSE
    ``kcl_84uS``     8.79e-08    **−74.38 %**   REFUSE
    ``kcl_1413uS``   9.28e-10    +2.45 %        pass
    ``kcl_4500uS``   8.93e-10    +1.81 %        pass
    ===============  ==========  =============  ===========

    The two refused folds are exactly the two that drop an arc-resolved standard, and
    their ``Qg`` is **97× high**. Nothing about those artifacts looks wrong from the
    inside: the optimiser converges, the residuals are small, no parameter is railed,
    and the numbers are simply not the ones being asked for. The gate fires on both bad
    folds and on neither good one — so it discriminates on the real corpus, which is
    the check ``SUBAGENT_RULES`` §3.2 asks for and which no amount of unit testing
    would have supplied.

    The difference between a usable artifact and a confidently wrong one is therefore a
    property of the **fitting set**, knowable before any fitting happens, and a refusal
    is the only honest output for the bad case.
    """
    n = len(spectra)
    resolved = tuple(s.label for s in spectra if arc_is_resolved(s.freq_hz, s.z))
    n_free = len(SHARED_PARAMS) + n * len(PER_SPECTRUM_PARAMS)
    n_resid = 2 * sum(int(np.asarray(s.freq_hz).size) for s in spectra)
    base = dict(n_spectra=n, n_arc_resolved=len(resolved), arc_resolved=resolved,
                n_free=n_free, n_residuals=n_resid)

    if n < 2:
        return Admissibility(False, "a shared parameter needs at least two spectra to "
                                    "be shared across", **base)
    if len(resolved) < min_arc_resolved:
        return Admissibility(
            False,
            f"only {len(resolved)} of {n} spectra resolve the geometric arc in band "
            f"(need {min_arc_resolved}); Qg and ng have nothing to be identified from "
            f"and the artifact will be confidently wrong",
            **base)
    if n_resid <= n_free:
        return Admissibility(False, f"{n_free} free parameters against {n_resid} "
                                    f"residuals — underdetermined", **base)
    return Admissibility(True, "shared parameters are identifiable from this set", **base)


# ── Packing ──────────────────────────────────────────────────────────────────

def _pack_bounds(n: int) -> tuple[np.ndarray, np.ndarray]:
    lo = [BOUNDS[p][0] for p in SHARED_PARAMS] + [BOUNDS[p][0] for p in PER_SPECTRUM_PARAMS] * n
    hi = [BOUNDS[p][1] for p in SHARED_PARAMS] + [BOUNDS[p][1] for p in PER_SPECTRUM_PARAMS] * n
    return np.array(lo), np.array(hi)


def _unpack(x: np.ndarray, i: int) -> dict[str, float]:
    p = {name: float(x[j]) for j, name in enumerate(SHARED_PARAMS)}
    base = len(SHARED_PARAMS) + i * len(PER_SPECTRUM_PARAMS)
    for j, name in enumerate(PER_SPECTRUM_PARAMS):
        p[name] = float(x[base + j])
    return p


def _residual(spec: ConstrainedSpectrum, params: dict[str, float]) -> np.ndarray:
    """Modulus-weighted real/imaginary residual, stacked.

    Weighting matches :mod:`~softae.analysis.eis.fitter`'s ``weight_by_modulus=True``:
    unweighted least squares over decades of ``|Z|`` is dominated by the low-frequency
    end and cannot see a small series term against a large tail.
    """
    z_model = model_impedance(spec.freq_hz, params, spec.y_shunt, spec.pinned_l_H)
    w = 1.0 / np.abs(spec.z)
    return np.concatenate([(z_model.real - spec.z.real) * w,
                           (z_model.imag - spec.z.imag) * w])


def _seed(spec: ConstrainedSpectrum, multiplier: float = 1.0) -> dict[str, float]:
    r = conductance_r_sol(spec.freq_hz, spec.z) * multiplier
    return {"R0": 0.15 * r, "R1": 0.85 * r,
            "Qg": 4e-11, "ng": 0.95, "Qd": 2e-6, "nd": 0.85}


def _fit_free(
    spec: ConstrainedSpectrum, start: dict[str, float], free: tuple[str, ...], tol: float
) -> tuple[dict[str, float], Any]:
    lo = [BOUNDS[p][0] for p in free]
    hi = [BOUNDS[p][1] for p in free]

    def residual(q: np.ndarray) -> np.ndarray:
        p = dict(start)
        p.update({n: float(v) for n, v in zip(free, q)})
        return _residual(spec, p)

    q0 = np.clip([start[p] for p in free], lo, hi)
    res = least_squares(residual, q0, bounds=(lo, hi), method="trf",
                        max_nfev=MAX_NFEV_SINGLE, x_scale="jac",
                        xtol=tol, ftol=tol, gtol=tol)
    out = dict(start)
    out.update({n: float(v) for n, v in zip(free, res.x)})
    return out, res


# ── The two fits ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SharedArtifact:
    """The frozen shared three, and everything needed to judge them later."""

    Qg: float
    ng: float
    nd: float
    labels: tuple[str, ...] = ()
    cost: float = float("nan")
    status: int = 0
    nfev: int = 0
    n_free: int = 0
    n_residuals: int = 0
    tol: float = CONSTRAINED_FIT_TOL
    #: Points across the set where the shunt table had no coverage. Carried on the
    #: artifact because it is a property of the artifact, not of one spectrum.
    n_shunt_held: int = 0
    admissibility: Admissibility | None = None

    @property
    def shared(self) -> dict[str, float]:
        return {"Qg": self.Qg, "ng": self.ng, "nd": self.nd}

    def railed(self) -> tuple[str, ...]:
        """Shared parameters resting on a bound — the artifact's own alarm.

        A railed ``nd`` is the specific signature of the inadmissible folds, and it
        survives even when :func:`check_admissible` passed, so it is reported rather
        than assumed away.
        """
        out = []
        for name, v in self.shared.items():
            lo, hi = BOUNDS[name]
            if abs(v - lo) <= 1e-6 * max(abs(v), abs(lo)) or \
                    abs(v - hi) <= 1e-6 * max(abs(v), abs(hi)):
                out.append(name)
        return tuple(out)

    def describe(self) -> str:
        rail = f", RAILED: {','.join(self.railed())}" if self.railed() else ""
        return (f"artifact from {len(self.labels)} spectra: Qg={self.Qg:.4g} "
                f"ng={self.ng:.4f} nd={self.nd:.4f} (cost {self.cost:.4g}, "
                f"status {self.status}, {self.nfev} nfev{rail})")


@dataclass(frozen=True)
class WellFit:
    """One spectrum against a frozen artifact. ``R_sol`` is the reported observable."""

    label: str
    R0: float
    R1: float
    Qd: float
    cost: float = float("nan")
    ok: bool = False
    seed_multiplier: float = float("nan")
    seed_spread_pct: float = float("nan")

    @property
    def R_sol(self) -> float:
        """``R0 + R1``, **unconditionally**.

        Not split-vs-sum by ρ. The topology puts the two resistances at two places in
        frequency and the corner between them leaves the band as conductivity rises;
        once it has, the optimiser trades between them at near-zero cost and only the
        sum is observable. Reporting ``R1`` alone silently drops a σ-dependent fraction
        of the true resistance — the origin of the apparent non-constant cell constant
        in the KCl campaign.
        """
        return self.R0 + self.R1


def fit_shared(
    spectra: list[ConstrainedSpectrum],
    *,
    tol: float = CONSTRAINED_FIT_TOL,
    min_arc_resolved: int = MIN_ARC_RESOLVED,
) -> SharedArtifact:
    """Fit ``Qg, ng, nd`` once across *spectra*, with ``R0, R1, Qd`` free per spectrum.

    Raises :class:`InadmissibleFitSet` when :func:`check_admissible` refuses — the
    refusal is the deliverable for that case, not a degraded artifact.
    """
    admissible = check_admissible(spectra, min_arc_resolved=min_arc_resolved)
    if not admissible.admissible:
        raise InadmissibleFitSet(admissible.describe())
    if tol > CONSTRAINED_FIT_TOL:
        logger.warning(
            "eis_constrained_fit_loose_tolerance", tol=tol,
            recommended=CONSTRAINED_FIT_TOL,
            msg=("a stacked fit at scipy's default tolerance terminates early and "
                 "returns a converged-looking wrong answer — measured at 17.5% high "
                 "on R_sol with status=3"),
        )

    n = len(spectra)
    seeds: list[dict[str, float]] = []
    previous: dict[str, float] | None = None
    for spec in spectra:
        start = _seed(spec) if previous is None else dict(previous)
        start.update({p: _seed(spec)[p] for p in CONTINUATION_RESET})
        fitted, _ = _fit_free(spec, start, PARAM_NAMES, tol)
        seeds.append(fitted)
        previous = fitted

    x0 = [float(np.median([s[p] for s in seeds])) for p in SHARED_PARAMS]
    for s in seeds:
        x0 += [s[p] for p in PER_SPECTRUM_PARAMS]
    lo, hi = _pack_bounds(n)
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)

    def stacked(x: np.ndarray) -> np.ndarray:
        return np.concatenate([_residual(spectra[i], _unpack(x, i)) for i in range(n)])

    res = least_squares(stacked, x0, bounds=(lo, hi), method="trf",
                        max_nfev=MAX_NFEV_GLOBAL, x_scale="jac",
                        xtol=tol, ftol=tol, gtol=tol)

    held = sum(s.shunt.n_held for s in spectra if s.shunt is not None)
    return SharedArtifact(
        Qg=float(res.x[0]), ng=float(res.x[1]), nd=float(res.x[2]),
        labels=tuple(s.label for s in spectra),
        cost=float(res.cost), status=int(res.status), nfev=int(res.nfev),
        n_free=int(res.x.size), n_residuals=int(res.fun.size), tol=float(tol),
        n_shunt_held=held, admissibility=admissible,
    )


def fit_frozen(
    spectrum: ConstrainedSpectrum,
    artifact: SharedArtifact,
    *,
    tol: float = CONSTRAINED_FIT_TOL,
    multipliers: tuple[float, ...] = R_SOL_SEED_MULTIPLIERS,
) -> WellFit:
    """Fit ``R0, R1, Qd`` with the artifact's shared three **frozen**. Multi-start.

    This is the per-spectrum half, and the one an engine could eventually call: it
    takes one spectrum and one frozen artifact and returns a deterministic answer,
    which is the invariant ``constrained_fit_chain.md`` §5 requires of σ.
    """
    best: tuple[dict[str, float], Any, float] | None = None
    sums: list[float] = []
    for m in multipliers:
        start = _seed(spectrum, m)
        start.update(artifact.shared)
        fitted, res = _fit_free(spectrum, start, PER_SPECTRUM_PARAMS, tol)
        sums.append(fitted["R0"] + fitted["R1"])
        if best is None or res.cost < best[1].cost:
            best = (fitted, res, m)
    assert best is not None

    fitted, res, m = best
    finite = [v for v in sums if v == v and v > 0]
    spread = (100.0 * (max(finite) - min(finite)) / min(finite)
              if len(finite) > 1 else float("nan"))
    return WellFit(label=spectrum.label, R0=fitted["R0"], R1=fitted["R1"],
                   Qd=fitted["Qd"], cost=float(res.cost), ok=bool(res.status > 0),
                   seed_multiplier=float(m), seed_spread_pct=spread)


# ── Validation: hold one out, report the error ───────────────────────────────

@dataclass(frozen=True)
class HoldoutResult:
    """One known answer, what was measured for it, and the deviation.

    The shape ``derive_reference_r``'s ``error_pct`` and ``blank_load``'s
    ``load_error_pct`` are both already instances of, named once. Run something whose
    answer is known, report the deviation. (Those two are not migrated onto this class
    here — they live in files this task does not own.)
    """

    label: str
    reference: float
    measured: float

    @property
    def error_pct(self) -> float:
        if not (self.reference == self.reference and self.reference):
            return float("nan")
        return 100.0 * (self.measured - self.reference) / self.reference


@dataclass(frozen=True)
class HoldoutReport:
    """``kind`` is load-bearing: an in-sample error is not a generalisation claim."""

    kind: str
    results: tuple[HoldoutResult, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[SharedArtifact, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> list[float]:
        return [abs(r.error_pct) for r in self.results if r.error_pct == r.error_pct]

    @property
    def mean_abs_error_pct(self) -> float:
        e = self.errors
        return float(np.mean(e)) if e else float("nan")

    @property
    def worst_abs_error_pct(self) -> float:
        e = self.errors
        return float(max(e)) if e else float("nan")

    def describe(self) -> str:
        refused = f", {len(self.refused)} fold(s) refused" if self.refused else ""
        return (f"{self.kind}: mean |err| {self.mean_abs_error_pct:.2f}%, worst "
                f"{self.worst_abs_error_pct:.2f}% over {len(self.results)} case(s)"
                f"{refused}")


def holdout_report(
    spectra: list[ConstrainedSpectrum],
    *,
    kind: str = "leave_one_out",
    tol: float = CONSTRAINED_FIT_TOL,
    min_arc_resolved: int = MIN_ARC_RESOLVED,
) -> HoldoutReport:
    """Fit, then score against :attr:`ConstrainedSpectrum.reference_ohm`.

    ``kind="in_sample"`` fits one artifact on everything and scores everything against
    it. ``kind="leave_one_out"`` fits N artifacts, each missing one spectrum, and scores
    only the held-out one — the number that is actually a generalisation claim.

    **A refused fold is recorded, not skipped.** With four standards, two of the four
    LOO folds are inadmissible by construction (each drops one of the two arc-resolved
    spectra), so a report that silently averaged over "the folds that worked" would
    hide the module's own headline limitation.
    """
    if kind == "in_sample":
        artifact = fit_shared(spectra, tol=tol, min_arc_resolved=min_arc_resolved)
        results = tuple(
            HoldoutResult(s.label, s.reference_ohm,
                          fit_frozen(s, artifact, tol=tol).R_sol)
            for s in spectra if s.reference_ohm == s.reference_ohm)
        return HoldoutReport("in_sample", results, (), (artifact,))

    if kind != "leave_one_out":
        raise ValueError(f"unknown holdout kind {kind!r}")

    results: list[HoldoutResult] = []
    refused: list[tuple[str, str]] = []
    artifacts: list[SharedArtifact] = []
    for held in spectra:
        rest = [s for s in spectra if s is not held]
        try:
            artifact = fit_shared(rest, tol=tol, min_arc_resolved=min_arc_resolved)
        except InadmissibleFitSet as exc:
            refused.append((held.label, str(exc)))
            continue
        artifacts.append(artifact)
        if held.reference_ohm == held.reference_ohm:
            results.append(HoldoutResult(held.label, held.reference_ohm,
                                         fit_frozen(held, artifact, tol=tol).R_sol))
    return HoldoutReport("leave_one_out", tuple(results), tuple(refused),
                         tuple(artifacts))
