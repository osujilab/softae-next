"""Is there resolvable parallel conduction in this spectrum at all?

Three scalars — S1, S2, S3 — asked of every spectrum *before* a circuit fit, so that
"the fitter is about to model the instrument" is a thing the pipeline can say rather
than a thing a reviewer notices afterwards. Spec: ``docs/SubAgent docs/measurability_scalars.md``.

**These are metrics. They are not gates.** Nothing here returns a
:class:`~softae.analysis.eis.gates.GateResult`, carries a severity, reads configuration,
writes ``ctx``, or is readable by :func:`~softae.analysis.eis.policy.reduce_gates`.
Arming is Stage 3 and is deferred on evidence (spec §8); ``GateSettings`` itself asks for
a campaign's worth of distribution before any threshold is chosen, and choosing one from
the rejected tail is choosing a percentile of one's own prior.

Three things about these functions are load-bearing and are here rather than in the spec
because a spec is not read at the moment someone edits the code.

**1. S1 is computed on ``Im(Y)/ω``, never on the apparent capacitance ``C_app``, and the
reason is an identity rather than a preference.** From ``Y = conj(Z)/|Z|²``::

    Im(Y)/ω = −Z'' / (ω|Z|²)      C_app = 1 / (ω|Z''|)

    C_app / (Im(Y)/ω) = |Z|²/Z''² = 1 + (Z'/Z'')² = 1 + tan²δ

**Exactly** — not to leading order and not empirically. Checked against four real
commissioning spectra, the largest relative deviation is 2×10⁻¹⁶ – 4×10⁻¹⁶, i.e. machine
epsilon. So ``C_app`` and ``Im(Y)/ω`` are **one measurement in two coordinate systems, not
two metrics**, and nothing in this module or its tests may present them as a second opinion
on each other.

**``tan δ`` is S2.** So S1-computed-on-``C_app`` does not merely correlate with S2 — it
*contains* S2, as a multiplicative factor. Reporting both would double-count **by
construction**, not by accident. Choosing ``Im(Y)/ω`` therefore **de-confounds S1 from S2**,
which is the one thing §3's non-independence most needs, and it is bought by a change of
variable rather than by a caveat.

The measured consequence follows from the algebra rather than standing on its own: across
four commissioning states the mid-band excursion-depth **ordering reverses end to end**. On
``C_app`` the healthiest spectrum is the *most* excursive (0.711, the deepest of the four)
and S1 is anti-discriminating; on ``Im(Y)/ω`` it is the *least* (0.341, the shallowest) and
S1 discriminates correctly. The leak is visible in the factor itself — the good 75 °C trace
is inflated **5.94×** where the other three are inflated ~1.2×, and ``(1 + tan²δ)`` peaks
exactly in the below-plateau dip S1 measures. That spectrum's apparent deep excursion *was*
S2 leaking into S1.

**A second and independent reason for the same choice: ``C_app = 1/(ω|Z''|)`` is singular as
``Z'' → 0``; ``Im(Y)/ω`` is not.** On the 75 °C spectrum ``C_app`` reaches 4.6×10⁴ pF where
the true capacitance stays under 664 pF — a 69× spike that is the resistive limit, not the
film. A *self-referential* plateau statistic, which is what :func:`capacitance_plateau` is,
would anchor itself to that artefact.

**Evidence and its limits: the Stage 0 data half — four spectra, one channel, one film. It
settles which QUANTITY S1 uses. It does not settle where any threshold sits**, and a later
reader must not read it as if it did. Commissioning's ``derive_open_constants`` already uses
``Im(Y)/ω``. :func:`~softae.analysis.eis.gates.gate_cap_flatness` keeps ``C_app`` and is
untouched.

**2. S1's statistic is the low-frequency LIFT above the plateau, not the depth below it.**
The spec's Correction 1 rejected the depth statistic as *anti-discriminating* — the good
75 °C trace dips deepest, so a depth criterion ranks the states backwards. That observation
was made in ``C_app`` coordinates, so it was measuring the confounding of §1 rather than the
statistic; in ``Im(Y)/ω`` the inversion is gone. The lift is kept regardless, because it is
what the figure separates by three decades — ~123× on the conducting trace against ≤1× on
the two non-conducting ones — where the depths span under a factor of three either way. The
below-plateau depth is *reported*, by :attr:`ConductionLift.below_plateau_depth`, and is
**explicitly not a criterion**; de-confounded, it is a candidate again rather than a refuted
one, which is a Stage 2a question and not this module's to answer.

**3. S1, S2 and S3 are not three independent tests (spec §7).** Panel B's lift, panel C's
tan δ valley and panel D's conductance headroom occur at the *same frequency on the same
trace*, because they are three projections of one statement: *parallel conduction is
present and above the resolution floor.* This codebase already says the same thing twice
— ``admittance.py``'s note that ``Z'/|Z''|`` and ``Re(Y)/Im(Y)`` are the same quantity,
and ``gate_series_rc``'s "redundant by design […] agreement between two independent
formulations is the confirmation". **So do not combine S1/S2/S3 as if their false-positive
rates were independent.** A 3-of-3 ``AND`` is far stricter than it looks; a 1-of-3 ``OR``
far looser. Their value is *confirmation and diagnosis* — which coordinate the failure is
legible in — not statistical independence. S3 is the partial exception: a negative
``Re Z`` is a statement about the **network**, not about the measurand's size, which is
why it is the one scalar that could ever carry a hard threshold — and even then only
conditioned on ``re_state``, since ``open_by_geometry`` floats the reference electrode by
construction and a healthy RE/CE-tied open blank on this rig measures 17–23 % violation as
its normal condition.

All functions take **physics-convention** complex impedance (``Im Z < 0`` for a capacitive
response), as :attr:`~softae.analysis.eis_data.EISResult.z_complex` returns.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from softae.analysis.eis.admittance import loss_tangent, susceptance
from softae.analysis.eis.gates import MIN_PLATEAU_POINTS
from softae.analysis.eis.policy import RE_STATES

__all__ = [
    "PLATEAU_TOL_PCT",
    "PLATEAU_MIN_DECADES",
    "OUTCOMES",
    "UNJUDGEABLE_OUTCOMES",
    "Plateau",
    "ConductionLift",
    "TandMargin",
    "NegativeConductance",
    "parallel_capacitance",
    "capacitance_plateau",
    "conduction_lift",
    "tand_margin",
    "negative_conductance_count",
    "eps_is_clamped",
]

#: Flatness tolerance for the plateau search, as a percent of the window's own median.
#: **Not re-derived** — this is ``gate_plateau_in_band``'s ``plateau_tol_pct`` default,
#: carried across so that "flat" means the same thing when said of ``Re Z`` and of
#: ``Im(Y)/ω``. It is a plain constant rather than a config read because nothing in this
#: module has config authority (spec §8).
PLATEAU_TOL_PCT = 10.0

#: Width below which a plateau is too narrow to serve as a reference. Likewise
#: ``gate_plateau_in_band``'s ``plateau_min_decades`` default, and likewise not re-derived.
#: The commissioning figure's blanks hold their plateau over ~2.3–2.6 decades, so 0.5 is
#: permissive here rather than marginal.
PLATEAU_MIN_DECADES = 0.5

#: S1's outcome vocabulary. **Three of these are the spec's §11 risk-2 requirement that
#: "no plateau" be distinct from both "flat" and "excursion"; the fourth is the same
#: distinction one level down** — see :func:`conduction_lift`.
OUTCOMES = ("no_plateau", "no_low_band", "flat", "excursion")

#: Outcomes in which S1 **was not measured**, as opposed to measured-and-absent. A caller
#: that treats these as either a pass or a fail is claiming an observation it does not
#: have. ``lift`` is NaN in both, and :attr:`ConductionLift.judgeable` is False.
UNJUDGEABLE_OUTCOMES = frozenset({"no_plateau", "no_low_band"})


class Plateau(NamedTuple):
    """The mid-band flat run of ``Im(Y)/ω``, or its absence.

    ``decades == 0`` and a NaN ``C_plateau`` mean **no plateau was found**, which is a
    third thing beside "wide" and "narrow": there is no denominator for S1 at all.
    """

    C_plateau: float
    lo_hz: float
    hi_hz: float
    decades: float

    @property
    def found(self) -> bool:
        return self.decades > 0.0 and self.C_plateau == self.C_plateau

    @property
    def wide_enough(self) -> bool:
        """Whether the run clears :data:`PLATEAU_MIN_DECADES`. Reported, not enforced."""
        return self.decades >= PLATEAU_MIN_DECADES


class ConductionLift(NamedTuple):
    """S1 — how far ``Im(Y)/ω`` rises above its own plateau below the plateau band.

    ``outcome`` is first, and deliberately so: it is the answer, and ``lift`` is only
    meaningful once ``outcome`` has been read. Unpacking this positionally into a float
    raises :class:`TypeError` on the first arithmetic rather than quietly producing a
    verdict from a string (spec §11, risk 2 — *"make it structurally impossible to
    misread"*).
    """

    outcome: str
    lift: float
    below_plateau_depth: float
    plateau: Plateau

    @property
    def judgeable(self) -> bool:
        """False when S1 was not measured — no plateau, or nothing below it."""
        return self.outcome not in UNJUDGEABLE_OUTCOMES

    def describe(self) -> str:
        if not self.judgeable:
            return f"S1 not measured ({self.outcome})"
        return (f"S1 {self.outcome}: lift {self.lift:.3g}x above a "
                f"{self.plateau.decades:.2f}-decade plateau of "
                f"{self.plateau.C_plateau:.3e} F "
                f"({self.plateau.lo_hz:.3g}-{self.plateau.hi_hz:.3g} Hz)")


class TandMargin(NamedTuple):
    """S2 — ``min(tan δ)`` against the phase floor **at that point's own |Z|**.

    A NaN ``eps_deg`` means the table refused to answer at this impedance, and the margin
    is NaN with it: **provisional, never a pass**. A *finite* ``eps_deg`` is not by itself
    proof that the impedance was characterised — see :func:`eps_is_clamped`.
    """

    margin: float
    f_at_min: float
    z_at_min: float
    eps_deg: float

    @property
    def characterised(self) -> bool:
        """Whether the table returned an ``ε`` at all at :attr:`z_at_min`."""
        return self.eps_deg == self.eps_deg

    @property
    def provisional(self) -> bool:
        """True whenever no margin could be formed. Never collapses to a pass."""
        return not (self.margin == self.margin)


class NegativeConductance(NamedTuple):
    """S3 — points outside the passive quadrant, and the RE state that contextualises them.

    ``re_state`` is echoed rather than inferred: ``Z`` alone cannot say how the reference
    electrode was wired, and guessing it is what would turn this into a gate that rejects
    bare boards for being bare boards.
    """

    n: int
    frac: float
    re_state: str

    @property
    def expected_by_construction(self) -> bool:
        """``open_by_geometry`` floats the RE by design; violations there are structural."""
        return self.re_state == "open_by_geometry"


# ── S1 ───────────────────────────────────────────────────────────────────────

def _widest_flat_run(
    x: np.ndarray, y: np.ndarray, tol: float
) -> tuple[int, int, float]:
    """Widest contiguous run of *y* staying within ``±tol`` of **its own median**.

    The :func:`~softae.analysis.eis.gates.gate_plateau_in_band` pattern, applied to a
    different ordinate. Self-referential on purpose: measuring flatness against the data
    itself needs no constant and has none of the bias of comparing against an external
    estimator. :data:`~softae.analysis.eis.gates.MIN_PLATEAU_POINTS` is imported rather
    than restated — below three points "flat" is a statement about sweep density, and that
    minimum was bought by measurement (a two-point window certified a plateau on a
    synthetic that had none).

    *x* must be sorted ascending and positive; returns ``(i, j, decades)``.
    """
    best = (0, 0, 0.0)
    for i in range(x.size):
        for j in range(x.size - 1, i + MIN_PLATEAU_POINTS - 2, -1):
            seg = y[i:j + 1]
            med = float(np.median(seg))
            if med <= 0:
                continue
            if float(np.max(np.abs(seg - med))) <= tol * med:
                span = float(np.log10(x[j] / x[i]))
                if span > best[2]:
                    best = (i, j, span)
                break                    # longer j for this i cannot be narrower
    return best


def parallel_capacitance(freq: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """``Im(Y)/ω`` in farads — commissioning's capacitance, not ``C_app`` (module §1)."""
    f = np.asarray(freq, dtype=float)
    omega = 2.0 * math.pi * f
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(omega > 0.0, susceptance(Z) / np.where(omega > 0.0, omega, 1.0),
                        np.nan)


def capacitance_plateau(
    freq: np.ndarray, Z: np.ndarray, *, tol_pct: float = PLATEAU_TOL_PCT
) -> Plateau:
    """The mid-band flat run of ``Im(Y)/ω``, found per-spectrum.

    The reference stays **per-spectrum and is never commissioned** (spec §4.4).
    ``C_stray_F`` in the calibration set is 10–25 pF of MUX and cabling; the cell plateau
    is an order of magnitude larger and a different quantity. Commissioning supplies the
    *method* here — mid-band window, median, self-reference — never the *value*.

    The high-frequency roll-off is excluded by construction rather than by a hard-coded
    window: it is not flat, so the search does not select it.
    """
    f = np.asarray(freq, dtype=float)
    C = parallel_capacitance(f, Z)
    none = Plateau(float("nan"), float("nan"), float("nan"), 0.0)

    good = np.isfinite(f) & (f > 0) & np.isfinite(C) & (C > 0)
    if int(good.sum()) < MIN_PLATEAU_POINTS:
        return none

    order = np.argsort(f[good])
    fs, Cs = f[good][order], C[good][order]
    i, j, decades = _widest_flat_run(fs, Cs, float(tol_pct) / 100.0)
    if decades <= 0.0:
        return none
    return Plateau(float(np.median(Cs[i:j + 1])), float(fs[i]), float(fs[j]), decades)


def conduction_lift(
    freq: np.ndarray, Z: np.ndarray, *, tol_pct: float = PLATEAU_TOL_PCT
) -> ConductionLift:
    """S1 — ``max(Im(Y)/ω) / C_plateau`` over ``f < lo_hz``.

    Real parallel conduction adds to the apparent capacitance at low frequency and lifts
    it monotonically above the plateau; its **absence** is the finding. The failure
    statement is *"``Im(Y)/ω`` never rises above its own mid-band plateau anywhere below
    it"*.

    Four outcomes, and the split between the measured and the unmeasured pair is the whole
    point of returning a structure rather than a float:

    ``no_plateau``
        No flat mid-band run, so no denominator. **Not measured.**
    ``no_low_band``
        A plateau, but it reaches the bottom of the sweep, so there is no below-plateau
        region to look in. **Not measured** — reporting this as ``flat`` would state an
        absence that was never looked for, which is precisely the value-vs-bound confusion
        this package exists to prevent.
    ``flat``
        Measured, and the lift is within the plateau's own tolerance of unity — no
        resolvable parallel conduction.
    ``excursion``
        Measured, and ``Im(Y)/ω`` rises above the plateau below it — conduction present.

    The threshold between the last two is the plateau's own ``tol_pct`` rather than an
    invented constant: a point 5 % above a plateau defined as flat to ±10 % has not left
    it. :attr:`ConductionLift.below_plateau_depth` is reported alongside and is
    **explicitly not a criterion** (module §2).

    .. note::
       Measured on the two real RE/CE-tied open blanks this rig produced on 2026-08-06 —
       spectra with no conduction in them at all — the outcome is ``excursion`` at 1.4×
       and 2.1×, against ~123× for the commissioning figure's conducting trace. **So the
       outcome label alone does not discriminate on real data.** Only the *magnitude*,
       against a distribution, can; that is Stage 2a's job and Stage 3's decision.
    """
    f = np.asarray(freq, dtype=float)
    C = parallel_capacitance(f, Z)
    tol = float(tol_pct) / 100.0
    plateau = capacitance_plateau(f, Z, tol_pct=tol_pct)
    if not plateau.found:
        return ConductionLift("no_plateau", float("nan"), float("nan"), plateau)

    finite = np.isfinite(f) & (f > 0) & np.isfinite(C)
    depth = (1.0 - float(np.min(C[finite])) / plateau.C_plateau
             if finite.any() else float("nan"))

    below = finite & (f < plateau.lo_hz)
    if not below.any():
        return ConductionLift("no_low_band", float("nan"), depth, plateau)

    lift = float(np.max(C[below])) / plateau.C_plateau
    outcome = "excursion" if lift > 1.0 + tol else "flat"
    return ConductionLift(outcome, lift, depth, plateau)


# ── S2 ───────────────────────────────────────────────────────────────────────

def eps_is_clamped(table: Any, z_ohm: float) -> bool:
    """Whether ``epsilon_deg(z_ohm)`` is an endpoint value rather than an interpolation.

    :meth:`~softae.analysis.eis.calibration.PhaseAccuracyTable.epsilon_deg` refuses to
    extrapolate and returns NaN outside ``valid_decades`` of every tabulated point — but
    *inside* that radius and past the last point, ``np.interp`` **clamps to the endpoint**.
    So a finite ``ε`` is not proof the impedance was characterised, and the two cases have
    to be told apart by hand. This is a separate function rather than a fifth field on
    :class:`TandMargin` so that the spec's four-value return stays exactly four.
    """
    zs = [z for z in getattr(table, "z_ohm", ()) or () if z > 0]
    if not zs or not (z_ohm > 0):
        return False
    if not table.covers(z_ohm):
        return False
    return z_ohm < min(zs) or z_ohm > max(zs)


def tand_margin(freq: np.ndarray, Z: np.ndarray, table: Any) -> TandMargin:
    """S2 — ``min(tan δ) / tan(ε(|Z|))``, both estimators chosen against each other.

    **Conservatism runs in opposite directions on the two sides of this ratio, and that is
    the single most important thing about it** (spec §6). ``derive_phase_table`` carries an
    explicit, evidenced refusal to use the minimum for the *instrument* — a minimum across
    a sweep is the single luckiest point, and a phase floor that small qualifies almost any
    spectrum as a value. That refusal is correct **and applying it uniformly is the bug**:

    ===============  ==================  ===================================================
    side             estimator           because
    ===============  ==================  ===================================================
    denominator ε    per-decade MEDIAN   the floor must not be understated; a lucky minimum
                                         makes the instrument look better than it is and
                                         **over-qualifies everything**
    numerator tan δ  MINIMUM over band   understating the sample's own margin errs toward
                                         reporting a *bound*; a median **over-qualifies the
                                         sample** and throws the safe direction away
    ===============  ==================  ===================================================

    The median fails here for a concrete reason visible in the figure: every state converges
    on ``tan δ ≈ 5`` at 10⁵ Hz, so a band median is dominated by the part of the band where
    all spectra look alike — the statistic meant to detect *"there is no measurement here"*
    computed where nothing distinguishes anything.

    **The denominator is per-point.** ``ε`` varies 4.5× across the commissioned table while
    the spectrum traverses ~4 decades of ``|Z|``. ``CalibrationSet.envelope()`` collapses
    the table to its **lowest-|Z|** entry, which on the committed set is near the table's
    *maximum* — matching the sample's floor would be a coincidence, not a measurement. So
    the table is queried at the minimum-``tan δ`` point's own ``|Z|``.

    **The minimum is taken over the whole band, including the points ``gate_quadrant``
    would drop**; masking them first is half of the shipped defect. Note the two sets are
    not the same set, and the difference is not academic: ``gate_quadrant`` drops
    ``Re Z < 0``, while a ratio needs ``tan δ > 0``, and ``tan δ = Z'/(−Z'')`` is positive
    whenever ``Z'`` and ``−Z''`` agree in sign. So a point at ``Re Z < 0, Im Z > 0`` — the
    doubly unphysical corner — has a **positive** ``tan δ`` and **is** admitted here while
    the gate would have removed it, which is the spec's intent. Conversely a passive but
    inductive point (``Re Z > 0, Im Z > 0``) has ``tan δ < 0``, cannot enter the ratio, and
    is *not* one of the points :func:`negative_conductance_count` counts — it is excluded
    without an S3 row against it, and that is a known blind spot rather than a claim of
    coverage.

    NaN from the table ⇒ NaN margin ⇒ **provisional, never a pass**. ``eps_deg`` is returned
    as used so a reviewer can see which value qualified the answer, and
    :func:`eps_is_clamped` says whether it was interpolated or clamped.

    .. note::
       The point is chosen by ``argmin(tan δ)`` and ``ε`` is then read at *that* point, per
       spec §8 — not by minimising the ratio itself. Where ``ε`` rises steeply with ``|Z|``
       the two can differ, and minimising the ratio would be the stricter statistic. Left
       as specified; recorded here because it is a real choice, not an oversight.
    """
    f = np.asarray(freq, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    tand = loss_tangent(Zc)
    nothing = TandMargin(float("nan"), float("nan"), float("nan"), float("nan"))

    usable = np.isfinite(f) & (f > 0) & np.isfinite(tand) & (tand > 0)
    if not usable.any():
        return nothing

    k = int(np.flatnonzero(usable)[int(np.argmin(tand[usable]))])
    f_min, z_min, tand_min = float(f[k]), float(abs(Zc[k])), float(tand[k])

    eps_deg = float(getattr(table, "epsilon_deg", lambda _z: float("nan"))(z_min))
    if not (eps_deg == eps_deg) or eps_deg <= 0.0:
        return TandMargin(float("nan"), f_min, z_min, eps_deg)

    return TandMargin(tand_min / math.tan(math.radians(eps_deg)), f_min, z_min, eps_deg)


# ── S3 ───────────────────────────────────────────────────────────────────────

def negative_conductance_count(
    Z: np.ndarray, *, re_state: str = "unverified"
) -> NegativeConductance:
    """S3 — how many points sit outside the passive quadrant, and in what RE state.

    ``Re(Y) = Z'/|Z|²``, so this is the same count as ``Re Z < 0``; it is expressed in
    admittance because that is the coordinate the measurand lives in. ``gate_quadrant``
    already computes this count exactly — the gap S3 names is **severity, not
    computation**: the gate drops the offending points, ``reduce_gates`` demotes the
    spectrum only as far as SUSPECT, the remnant clears ``min_fit_pts`` and is fitted. This
    function computes nothing new; it makes the number available to a caller that is not a
    gate.

    ``re_state`` is a caller-supplied fact from :data:`~softae.analysis.eis.policy.RE_STATES`
    and is echoed unchanged, because a violation means opposite things in different states.
    On ``open_by_geometry`` the reference electrode floats **by construction** and the
    violation is structural. And even on a closed loop the healthy count is not zero: two
    real RE/CE-tied open blanks measured on this rig 65 s apart carry 6/35 (17 %) and 8/35
    (23 %), and the first of those is the blank the committed ``mux16`` calibration was
    derived from. An unconditional ``n == 0`` would reject the calibration's own source.
    """
    if re_state not in RE_STATES:
        raise ValueError(f"re_state {re_state!r} is not one of {RE_STATES}")
    Zc = np.asarray(Z, dtype=complex)
    finite = np.isfinite(Zc.real) & np.isfinite(Zc.imag) & (np.abs(Zc) > 0)
    n_finite = int(finite.sum())
    n_bad = int(np.count_nonzero(Zc.real[finite] < 0))
    frac = n_bad / n_finite if n_finite else float("nan")
    return NegativeConductance(n_bad, frac, re_state)
