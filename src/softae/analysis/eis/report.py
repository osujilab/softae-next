"""What may be said about this spectrum — a value, a bound, or nothing.

Overhaul §3.3 is the failure this module exists to prevent. A dry, salt-dilute film
produced a spectrum whose conduction was absent at every frequency, and the defensible
output was ``σ ≲ 4×10⁻⁷ S/cm`` — an upper bound, not a fitted value. A pipeline that
cannot express the difference will report the bound as a measurement, and a campaign
optimising against it will chase instrument noise.

So conductivity is reported through a small vocabulary rather than as a bare float:

``value``
    Measured, with an uncertainty.
``bound``
    Resolution-limited, with a *qualified* ceiling — phase accuracy was measured.
``bound_unqualified``
    Resolution-limited, ceiling *provisional* — phase accuracy is still an estimate.
``unavailable``
    No per-sample thickness, so no cell constant, so no conductivity. Not an error.

.. warning::
   The switch **degrades toward caution**, but the reason has changed. An earlier
   version fell back to a magnitude proxy against ``Z_φ ≈ 5×10⁷ Ω``; that ceiling is
   **withdrawn** (it was a floating-reference-electrode artefact), and nothing here
   uses it.

   What replaces it is narrower and defensible. Phase noise *has* now been measured, so
   a genuine headroom comparison is possible. Inside the calibrated band a bound is
   **qualified**; outside it the same bound is **provisional**, because extrapolating
   an instrument constant across three decades without saying so is exactly how the
   withdrawn ceiling came to be believed in the first place.

.. note::
   **Which floor this function divides by changed on 2026-09-04**, and the numbers in
   older prose are the old one. Until then the envelope was always
   ``[eis.instrument]``'s configured estimate — 0.149° on a 9.9 kΩ **resistive** load,
   ``tan δ`` floor 0.0026 — because ``CalibrationSet.envelope()`` had no call site in
   ``src/``. It has one now, so on the gated engine the floor is the commissioned one:
   on ``mux16`` that is 6.12° at 10.1 MΩ **capacitive**, a ``tan δ`` floor of 0.1072 —
   **41× larger**. Every headroom is 41× smaller, and on the stored corpus 90 % of
   spectra reporting a value falls to 19 %.

   The shipped configuration is unaffected: ``[eis] engine = "legacy"`` and
   ``_legacy_report`` never calls this function. **Shadow-rehearsal output from before
   and after that wiring is not comparable and must not be pooled.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

#: Fraction of surviving points above ``Z_φ`` beyond which an unqualified envelope
#: forces a bound. Half is deliberate: it is the point past which the *typical* point
#: in the spectrum is untrustworthy, not merely the tail.
DEFAULT_ABOVE_CEILING_FRAC = 0.5


@dataclass(frozen=True)
class SigmaReport:
    """Conductivity, the resistance it came from, and what may be claimed about it."""

    mode: str = "unavailable"
    value: float = float("nan")
    upper_bound: float = float("nan")
    rel_uncertainty: float = float("nan")
    provisional: bool = False

    R_reported_ohm: float = float("nan")
    R_reported_se_ohm: float = float("nan")
    R_basis: str = "split_bulk"       # "sum" | "split_bulk"
    rho: float = float("nan")

    K_per_cm: float = float("nan")
    K_route: str = "geometric"
    thickness_method: str = "unavailable"
    #: Electrode configuration behind ``K_config_factor`` (framework §1.1, R20).
    #: A wiring fact, recorded independently of whether the factor was confirmed.
    electrode_config: str = "unverified"
    k_config_factor: float = 1.0
    #: False ⇒ the *absolute scale* of σ is unqualified: F16 is a clean ~2× error with
    #: no other symptom. **Relative trends stay valid regardless**, since the factor is
    #: constant across a series — so a campaign ranking formulations is unaffected.
    config_factor_verified: bool = False
    #: Whether this sample had a verified ionic path to the reference stripe (R26).
    #: Reported separately from :attr:`config_factor_verified` because it is the one
    #: precondition that varies sample to sample on an otherwise verified board.
    re_contact_verified: bool = False

    model_free_R_ohm: float = float("nan")
    cross_check_pct: float = float("nan")
    phase_headroom: float = float("nan")

    @property
    def is_bound(self) -> bool:
        return self.mode in ("bound", "bound_unqualified")

    @property
    def is_value(self) -> bool:
        return self.mode == "value"

    def as_text(self) -> str:
        """Operator-facing rendering — ``1.2e-04 ±8%`` or ``≲ 4.0e-07 (provisional)``."""
        if self.mode == "unavailable":
            return "σ unavailable"
        tag = " (provisional)" if self.provisional else ""
        if self.is_bound:
            return f"σ ≲ {self.upper_bound:.2g} S/cm{tag}"
        unc = (
            f" ±{self.rel_uncertainty * 100:.0f}%"
            if self.rel_uncertainty == self.rel_uncertainty else ""
        )
        return f"{self.value:.3g} S/cm{unc}{tag}"

    def describe(self) -> str:
        basis = "R_series+R_bulk" if self.R_basis == "sum" else "R_bulk"
        if self.config_factor_verified:
            cfg = f", {self.electrode_config} ÷{self.k_config_factor:g}"
        else:
            why = ("no verified RE contact"
                   if self.electrode_config == "3-electrode"
                   and not self.re_contact_verified else "unverified")
            cfg = (f", {self.electrode_config}, K_config_factor {why} — "
                   f"absolute scale unqualified (relative trends unaffected)")
        return (
            f"{self.as_text()} from {basis} = {self.R_reported_ohm:.4g} Ω, "
            f"K = {self.K_per_cm:.1f} /cm [{self.K_route}], "
            f"t from {self.thickness_method}{cfg}"
        )


@dataclass(frozen=True)
class SpectrumReport:
    """One spectrum's complete analysis — the single return shape of both engines.

    Both engines return this so that DataStore, the analysis tab and the browser each
    learn exactly one new type and never branch on which engine produced it. Flipping
    ``[eis] engine`` is then the whole cutover, and it is reversible per run.
    """

    engine: str
    fit: Any = None                                  # circuit_fitting.FitResult
    sigma: SigmaReport = field(default_factory=SigmaReport)
    quality: Any = None                              # quality.QualityReport
    gate_log: tuple[dict[str, Any], ...] = ()
    mask: np.ndarray | None = None
    cell: Any = None                                 # geometry.CellConstant
    envelope: Any = None
    #: What was subtracted as fixture, and what that did (E3). ``None`` on the legacy
    #: engine, which corrects nothing — and a ``mode = "none"`` correction on the gated
    #: one, which is a different statement: *considered, and deliberately not applied*.
    correction: Any = None           # fixture.FixtureCorrection
    correction_outcome: Any = None   # fixture.CorrectionOutcome
    #: **Which fitter actually produced** :attr:`fit`. ``""`` where the question does not
    #: arise — the legacy engine has exactly one fitter, so ``engine`` already answers it.
    #:
    #: On the gated engine it does *not*: four routes can produce ``fit`` and three of
    #: them are not the gated fitter. ``engine="gated"`` therefore names the **cascade**
    #: that ran, not the estimator that returned the number, and a report carrying only
    #: ``engine`` describes a check it may not have performed. That gap is measured, not
    #: hypothetical: 22 of 54 rows on ``20260825T154521Z_arrhenius_sweep`` — 41 % — fell
    #: back to :func:`~softae.analysis.circuit_fitting.fit_circuit` while reporting
    #: themselves as gated output.
    #:
    #: ==========================  ================================================
    #: value                       what returned ``fit``
    #: ==========================  ================================================
    #: ``"gated"``                 ``fitter.fit_spectrum`` — the E1 fitter
    #: ``"two_point"``             the pre-gate open-arc route; **changes R₁**
    #: ``"legacy_fit_failed"``     ``fit_circuit``, because the gated fit did not
    #:                             converge — **this is the 41 %**
    #: ``"gated_no_fallback"``     the gated fit did not converge and the model has NO
    #:                             legacy equivalent, so nothing was fallen back *to*;
    #:                             the failed gated fit stands, ``success=False``
    #: ``""``                      not applicable (legacy engine), or a report built
    #:                             before this field existed
    #: ==========================  ================================================
    #:
    #: ``"legacy_fit_failed"`` and ``"gated_no_fallback"`` are kept apart deliberately:
    #: one says a legacy number was substituted, the other says none could be and the
    #: row therefore carries no resistance at all. Collapsing them would hide which,
    #: and only the second leaves the measurand missing.
    #:
    #: .. note::
    #:    A fourth value, ``"legacy_unknown_model"``, was documented here until
    #:    2026-09-03 and **was never reachable**. It sat behind an
    #:    ``except ValueError`` around ``fit_spectrum``, which raises only for a model
    #:    in *neither* registry — whereupon the handler called ``fit_circuit``, which
    #:    raises the identical error one line before the label could be assigned. The
    #:    dead branch is gone; the value never appeared in a stored row, so nothing
    #:    persisted needs migrating.
    fitter: str = ""

    @property
    def corrected(self) -> bool:
        """Whether fixture impedance was actually removed from *this* spectrum.

        Reads the **outcome**, not the correction. ``correction.applies`` says only
        that the constants exist and are usable — a spectrum rejected at admission
        carries exactly such a correction and was never corrected by it, because §6
        places the subtraction downstream of the gates that rejected it.
        """
        return bool(getattr(self.correction_outcome, "applied", False))

    @property
    def ok(self) -> bool:
        """True when the measurement may be used at all (mirrors ``QualityReport.ok``)."""
        return bool(getattr(self.quality, "ok", True))

    @property
    def n_dropped(self) -> int:
        return int(sum(e.get("n_dropped", 0) for e in self.gate_log))

    def gate_summary(self) -> str:
        """The one-cell rendering for the analysis tab's ``Gate`` column.

        **"pass" means every gate ran and none refused — not merely that none refused.**
        A gate that cannot evaluate its criterion fails *open*: it returns
        ``passed=True`` as a placeholder and marks it ``checked=False``
        (:meth:`~softae.analysis.eis.gates.GateResult.unchecked`, whose own docstring
        says ``checked=False`` "is what stops that posture from being reported as a clean
        result"). Reading ``passed`` alone reported it as a clean result anyway, which is
        the conflation the field was added to remove.

        ``checked`` is read with **no default**, because absent is a third answer:

        =============  ==============================================  ================
        ``checked``    what the entry is saying                        rendered here
        =============  ==============================================  ================
        ``True``       the gate ran and returned a verdict             by ``passed``
        ``False``      it could not run; ``passed`` is a placeholder   **"unchecked"**
        absent         the row predates the field                      by ``passed``
        =============  ==============================================  ================

        **The third row is a judgement, and it is deliberately the permissive one** —
        ``is not False`` rather than ``is True``. Every ``gate_log_json`` in the DataStore
        today predates ``checked``, so treating absent as unchecked would mark the entire
        stored corpus "unchecked" and make the distinction useless on the only data there
        is. This is the same ruling
        :meth:`softae.tools.eis_validate_records.FitRecord.passed_gates` makes, adopted
        here so the two surfaces cannot disagree by one default.

        Drops and unchecked gates are reported *together* rather than one shadowing the
        other: they are independent facts about the sweep, and a cell that showed only
        the first would hide the second exactly when both are true.
        """
        if self.engine != "gated":
            return "—"
        for entry in self.gate_log:
            if not entry.get("passed", True) and entry.get("severity") in (
                "block_spectrum", "block_session"
            ):
                return f"REJECTED: {entry.get('gate', '?')}"
        # A quality rejection NEVER reaches `gate_log`, so rescanning the log alone
        # cannot see it. `grade_fit` writes its reasons — R², RMS residual, failed
        # convergence — into `quality.issues`, and the raw-trace checks write theirs
        # the same way; neither is a gate and neither leaves an entry here.
        #
        # Without this branch a spectrum whose gates all passed but whose fit is
        # unusable renders "pass" while `report.ok` is False and the point is silently
        # withheld from the campaign objective — the cell says the measurement is fine
        # at exactly the moment the optimiser is refusing it. The true verdict reached
        # only the DataStore's `gate_verdict` column, read offline and never live.
        if not self.ok:
            issues = list(getattr(self.quality, "issues", ()) or ())
            return f"REJECTED: {issues[0]}" if issues else "REJECTED: quality"
        parts = []
        if self.n_dropped:
            parts.append(f"{self.n_dropped} dropped")
        unchecked = sum(1 for e in self.gate_log if e.get("checked") is False)
        if unchecked:
            parts.append(f"{unchecked} unchecked")
        return ", ".join(parts) if parts else "pass"

    def describe(self) -> str:
        return f"[{self.engine}] {self.gate_summary()} — {self.sigma.describe()}"


def decide_report_mode(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
    tand_headroom_mult: float = 3.0,
) -> tuple[str, bool, float]:
    """Decide value vs bound. Returns ``(mode, provisional, phase_headroom)``.

    The comparison is always the same one (framework §4.8) —
    ``headroom = tan δ_measured / tan(ε)``, and below ``tand_headroom_mult`` the
    extracted conductance sits at or under the instrument's resolution. What varies is
    whether ``ε`` may be *trusted at this impedance*:

    * ``ε`` measured **and** the spectrum inside its calibrated band → ``value`` or a
      qualified ``bound``.
    * ``ε`` measured but the spectrum decades away from where it was characterised →
      the same decision, marked **provisional**. Films sit at 10⁶–10⁸ Ω against a
      10⁴ Ω resistive characterisation, and an instrument constant carried three
      decades without comment is how the withdrawn ``Z_φ`` ceiling was born.
    * ``ε`` unmeasured → provisional bound. Never a value.

    **The numerator is the MINIMUM ``tan δ``, not the median, and the asymmetry is the
    whole point.** ``derive_phase_table`` carries an evidenced refusal to use a minimum —
    across a sweep it is the single luckiest point, and a phase floor that small qualifies
    almost any spectrum as a value. That refusal is correct for the **instrument**, which
    is the *denominator*, and applying it to the *numerator* as well was the defect this
    function shipped with. Conservatism runs in opposite directions on the two sides of a
    ratio: understating the **sample's** own margin errs toward reporting a *bound*, which
    is the safe direction; a median over-qualifies the sample and throws that direction
    away. Concretely, every state in the commissioning figure converges on ``tan δ ≈ 5`` at
    10⁵ Hz, so a band median is dominated by the region where every spectrum looks alike —
    the statistic meant to detect *"there is no measurement here"* was being computed where
    nothing distinguishes anything.

    Non-positive ``tan δ`` is still excluded, because a ratio needs a positive numerator
    and a negative ``tan δ`` is a statement that the passive-quadrant assumption failed at
    that point, **not** evidence that the sample's loss is under the floor. Admitting it
    would drive the minimum negative and force a bound for a reason that is not resolution.
    The exclusion is **logged rather than silent**, because masking these without saying so
    is half of what made the median look defensible. Most of them are the points
    :func:`~softae.analysis.eis.measurability.negative_conductance_count` counts as S3, but
    not all: a passive *inductive* point has ``tan δ < 0`` with ``Re Z > 0``, so the log
    line is the only place it is recorded.

    .. note::
       A second masking site remains and is **not** fixed here: ``engine.py`` calls this on
       the *survivors* of ``gate_quadrant``, so the excluded count seen below is already
       net of that drop. That site is coupled to the envelope wiring and is scoped
       separately.
    """
    if cell is None:
        return "unavailable", False, float("nan")

    from softae.analysis.eis.admittance import loss_tangent

    Z = np.asarray(Z, dtype=complex)
    mag = np.abs(Z)
    finite = np.isfinite(mag)
    z_med = float(np.median(mag[finite])) if finite.any() else float("nan")

    measured = bool(getattr(envelope, "phase_noise_measured", False))
    # Both fallbacks below are CONSERVATIVE on purpose, and the choice is load-bearing:
    # an envelope that cannot answer "was the phase floor measured, and does it apply
    # here?" makes this result **provisional**, never qualified.
    #
    # `gates.py:385` takes the opposite default on the same predicate — `lambda _z: True`
    # — so a missing capability makes the gate assume the floor applies and pass. The
    # two are not interchangeable and this asymmetry is currently undocumented at the
    # other site ([p96] §3, parallel's to resolve; `gates.py` is theirs).
    #
    # If the two are ever reconciled, reconcile TOWARD THIS ONE. `SUBAGENT_RULES` §3.1(a)
    # already condemns the permissive direction on this exact subject: the envelope's
    # `phase_noise_measured` once defaulted True, "so the guard that exists to force a
    # provisional result when the floor was never measured can never fire". A default
    # that answers the safe-sounding question when it has no information is the failure
    # this whole module is written against.
    in_band = bool(
        getattr(envelope, "phase_noise_valid_at", lambda _z: False)(z_med))
    floor = getattr(envelope, "tand_floor", float("nan"))

    if not measured or not (floor == floor) or floor <= 0:
        logger.info("eis_reported_as_bound",
                    reason="phase noise unmeasured", provisional=True)
        return "bound_unqualified", True, float("nan")

    tand = loss_tangent(Z)
    finite = np.isfinite(tand)
    n_excluded = int(np.count_nonzero(finite & (tand <= 0)))
    tand = tand[finite & (tand > 0)]
    if n_excluded:
        logger.info(
            "eis_tand_points_excluded", n_excluded=n_excluded, n_used=int(tand.size),
            msg="non-positive loss tangent outside the passive quadrant, excluded from "
                "the headroom numerator (see measurability.negative_conductance_count)",
        )
    if tand.size == 0:
        return "bound_unqualified", True, float("nan")

    headroom = float(np.min(tand)) / floor
    resolution_limited = headroom < float(tand_headroom_mult)

    if resolution_limited:
        logger.info(
            "eis_reported_as_bound", reason="loss tangent below the phase floor",
            phase_headroom=headroom, z_median_ohm=z_med, provisional=not in_band,
        )
        return ("bound" if in_band else "bound_unqualified"), not in_band, headroom

    if not in_band:
        logger.info(
            "eis_phase_floor_extrapolated", z_median_ohm=z_med,
            phase_headroom=headroom,
            msg="value reported, but the phase floor it cleared is extrapolated",
        )
    return "value", not in_band, headroom


def sigma_upper_bound(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
) -> float:
    """The defensible ceiling ``σ ≲ ε·ω·C·K`` when the loss is below the phase floor.

    A spectrum whose loss sits under the instrument's resolution still yields a
    rigorous statement, and refusing to make it would throw away a real result.

    Evaluated at the **lowest** usable frequency in the spectrum, not the median: the
    bound scales with ω, so the tightest defensible ceiling is the one at the bottom of
    the sweep. That is the same reason the updated envelope favours low-frequency
    sweeps — ``σ_min`` runs from ~1e-9 S/cm at 1 Hz to ~1e-7 S/cm at 100 Hz.

    Falls back to what the magnitude ceiling licenses (``K/Z_max``) only when ε is
    unavailable — never to the withdrawn ``K/Z_φ``.
    """
    if cell is None:
        return float("nan")
    K = getattr(cell, "K_per_cm", float("nan"))
    if not (K == K):
        return float("nan")

    eps = getattr(envelope, "eps_rad", float("nan"))
    if eps == eps:
        C = float("nan")
        try:
            from softae.analysis.eis.admittance import par_capacitance_estimate

            C = par_capacitance_estimate(freq, Z)
        except Exception:
            C = float("nan")
        f = np.asarray(freq, dtype=float)
        f = f[np.isfinite(f) & (f > 0)]
        if C == C and f.size:
            omega = 2.0 * math.pi * float(np.min(f))
            return float(eps * omega * C * K)

    z_max = float(getattr(envelope, "z_max_ohm", float("nan")))
    return K / z_max if z_max == z_max and z_max > 0 else float("nan")
