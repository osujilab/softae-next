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

from softae.analysis.eis.engine_support import BASIS_SUM_UNQUALIFIED

logger = structlog.get_logger(__name__)

#: Fraction of surviving points above ``Z_φ`` beyond which an unqualified envelope
#: forces a bound. Half is deliberate: it is the point past which the *typical* point
#: in the spectrum is untrustworthy, not merely the tail.
DEFAULT_ABOVE_CEILING_FRAC = 0.5

#: How :meth:`SigmaReport.describe` names each reporting basis, keyed by the token
#: :func:`~softae.analysis.eis.engine_support._resolve_reported_resistance` returns.
#:
#: **A mapping rather than a two-branch conditional, and the reason is a defect this
#: shape makes unrepeatable.** Until 2026-09-08 the line read
#: ``"R_series+R_bulk" if R_basis == "sum" else "R_bulk"``, so every basis the test did
#: not name fell to the ``else`` — and on 2026-09-04 a third basis arrived. A row
#: reported on :data:`~softae.analysis.eis.engine_support.BASIS_SUM_UNQUALIFIED` is a
#: **sum**, and the operator was told ``R_bulk``: the string named the *split* for a
#: measurement that is the *chain*, with no hint that the two had been confused. Any
#: consumer reading ``describe()`` — the analysis tab, a browse dialog, a pasted line in
#: a lab notebook — would have read a partition that was never determined.
#:
#: **The default for an unrecognised key is deliberately not a basis name.** An
#: unrecognised token means the renderer has fallen behind its producer, which is
#: exactly what happened here, and ``SUBAGENT_RULES`` §3.1(a) is that "unknown" must not
#: be spelled with the token for a specific checked claim.
#:
#: ``"(unqualified)"`` borrows :attr:`SigmaReport.mode`'s own idiom — the same word
#: ``"bound_unqualified"`` uses, for the same reason: the claim stands, the evidence
#: that would qualify it does not. Here that evidence is a covariance, so there is no
#: ρ with which to judge the split and no standard error to propagate; the sum is still
#: the DC resistance of the chain by arithmetic, which is what licenses reporting it.
#:
#: **Blast radius is zero on the shipped configuration and this is still not cosmetic.**
#: ``[eis] engine = "legacy"`` and ``_legacy_report`` hardcodes ``"split_bulk"``, so no
#: stored row has ever carried the third basis. It goes live with the E6 flip.
BASIS_TEXT = {
    "split_bulk": "R_bulk",
    "sum": "R_series+R_bulk",
    BASIS_SUM_UNQUALIFIED: "R_series+R_bulk (unqualified)",
}


def _format_hz(f_hz: float) -> str:
    """``1033.0`` → ``"1.03 kHz"``. Operator-facing only."""
    f = float(f_hz)
    return f"{f / 1000.0:.3g} kHz" if abs(f) >= 1000.0 else f"{f:.3g} Hz"


@dataclass(frozen=True)
class SigmaReport:
    """Conductivity, the resistance it came from, and what may be claimed about it."""

    mode: str = "unavailable"
    value: float = float("nan")
    #: The loss ceiling from :func:`sigma_loss_ceiling` — **name deliberately
    #: unchanged**, because ``core/autonomous_wiring.py`` reads
    #: ``report.sigma.upper_bound`` for its ``objective_declined_bound`` log line and
    #: that file belongs to another session. What changed is the *number*: it is now
    #: ``K·ε·ω*·C(ω*)`` at the headroom numerator's frequency, not at ``min(f)``.
    upper_bound: float = float("nan")
    #: Where the ceiling was evaluated, and on what basis — the keys of
    #: :class:`SigmaCeiling`. NaN / ``"unavailable"`` means no ceiling could be
    #: stated, which is **not** the same claim as a small one (``SUBAGENT_RULES``
    #: §3.1(a)); in particular it must never be re-spelled as the ``min(f)`` number.
    upper_bound_f_hz: float = float("nan")
    upper_bound_basis: str = "unavailable"
    #: ``cell.sigma(R_reported_ohm)`` — carried in the **bound** branch too, where the
    #: pairing used to be thrown away. A ceiling and the fit that contradicts it are
    #: only comparable if both travel; see :meth:`as_text`.
    fit_implied_sigma: float = float("nan")
    rel_uncertainty: float = float("nan")
    provisional: bool = False

    R_reported_ohm: float = float("nan")
    R_reported_se_ohm: float = float("nan")
    #: Which resistance :attr:`R_reported_ohm` is — the keys of :data:`BASIS_TEXT`,
    #: which is also what renders it. ``_resolve_reported_resistance`` is the only
    #: production producer of anything but the default.
    R_basis: str = "split_bulk"       # "split_bulk" | "sum" | "sum_unqualified"
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
    #: Where :attr:`phase_headroom`'s numerator came from (T11.31). The margin is
    #: now the minimum over running medians of a :data:`DEFAULT_HEADROOM_WINDOW`-wide
    #: frequency window, so it names a real measured point — and at |phase| → 90°,
    #: where ``Re Z → 0`` drives ``tan δ`` to zero whatever the sample is doing, that
    #: point's phase is the difference between a margin and an artefact.
    #:
    #: ``headroom_window == 1`` means the minimum was taken on a single point — too
    #: few survivors to window. **Not persisted**: ``data_store`` stores
    #: ``phase_headroom`` alone, and 0 of 3918 stored rows carry one, so no stored
    #: number changes meaning.
    numerator_f_hz: float = float("nan")
    numerator_phase_deg: float = float("nan")
    numerator_phase_saturated: bool = False
    headroom_window: int = 1

    @property
    def is_bound(self) -> bool:
        return self.mode in ("bound", "bound_unqualified")

    @property
    def is_value(self) -> bool:
        return self.mode == "value"

    def as_text(self) -> str:
        """Operator-facing rendering — ``1.2e-04 ±8%``, or for a bound::

            σ ≲ 4.2e-06 S/cm @1.03 kHz; fit implies 3.2e-07 (provisional)
            σ ≲ 3.2e-07 S/cm @1.03 kHz; fit implies 4.2e-06 — loss ceiling below the fit

        (``(provisional)`` is appended last in both cases; it is shown on one line only
        to keep the other inside the line length.)

        The frequency is named because the ceiling is proportional to ω and therefore
        means nothing without it, and the fit-implied σ is named because the two are
        different estimators of different quantities. When the ceiling sits *below*
        the fit it does not cover it, and the text says so rather than choosing — the
        disagreement is a flag (``ceiling_below_fit``), never a refusal.
        """
        if self.mode == "unavailable":
            return "σ unavailable"
        tag = " (provisional)" if self.provisional else ""
        if self.is_bound:
            f_hz = self.upper_bound_f_hz
            at = f" @{_format_hz(f_hz)}" if f_hz == f_hz else ""
            fit = self.fit_implied_sigma
            clause = ""
            if fit == fit:
                clause = f"; fit implies {fit:.2g}"
                if self.upper_bound == self.upper_bound and self.upper_bound < fit:
                    clause += " — loss ceiling below the fit"
            return f"σ ≲ {self.upper_bound:.2g} S/cm{at}{clause}{tag}"
        unc = (
            f" ±{self.rel_uncertainty * 100:.0f}%"
            if self.rel_uncertainty == self.rel_uncertainty else ""
        )
        return f"{self.value:.3g} S/cm{unc}{tag}"

    def describe(self) -> str:
        basis = BASIS_TEXT.get(self.R_basis, f"unrecognised basis {self.R_basis!r}")
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
    #: ``"legacy_fit_railed"``     ``fit_circuit``, because the gated fit *converged*
    #:                             and then rested on a box constraint, so
    #:                             ``_demote_if_railed`` removed its measurement claim
    #: ``"gated_no_fallback"``     the gated fit did not converge (or converged and
    #:                             railed) and the model has NO legacy equivalent, so
    #:                             nothing was fallen back *to*; the failed gated fit
    #:                             stands, ``success=False``
    #: ``""``                      not applicable (legacy engine), or a report built
    #:                             before this field existed
    #: ==========================  ================================================
    #:
    #: ``"legacy_fit_failed"`` and ``"gated_no_fallback"`` are kept apart deliberately:
    #: one says a legacy number was substituted, the other says none could be and the
    #: row therefore carries no resistance at all. Collapsing them would hide which,
    #: and only the second leaves the measurand missing.
    #:
    #: ``"legacy_fit_railed"`` is split from ``"legacy_fit_failed"`` on the same
    #: principle and was added 2026-09-10 with the ``fit_max_nfev`` raise. Both
    #: substitute a legacy number, but the reasons are not the same finding: one says
    #: the optimiser never arrived, the other that it arrived at a wall. Reporting the
    #: second as the first would make this table's own definition of
    #: ``"legacy_fit_failed"`` — *"because the gated fit did not converge"* — false for
    #: part of its population, and would silently inflate the 41 % statistic with fits
    #: that converged.
    #:
    #: **Rare, and measured rather than assumed.** A converged-then-railed fit occurred
    #: **0 times in 600 sampled spectra at the 20 000 ceiling** and 1 time in the 17
    #: that ceiling was truncating (measurement 1400, which converges at nfev 42 924).
    #: That is *not observed*, not *impossible*: nothing stops a fit converging quickly
    #: and railing, so the route is structurally reachable at any ceiling and simply was
    #: not taken by the sample. No stored row carries this value, so nothing persisted
    #: needs migrating — but a reader should not treat its absence as a guarantee.
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

        **A failed ``flag`` refuses nothing, drops nothing and leaves no counter, so it
        has to be named or it is invisible.** ``gate_arc_closure`` on an ``OPEN`` arc is
        exactly that shape — ``passed=False``, ``severity="flag"``, ``n_dropped=0``,
        ``checked=True`` — and before this branch existed it fell through to the bare
        ``"pass"`` that means *every gate ran and none refused*, the same defect
        :meth:`~softae.gui.tabs.tab_analysis._gate_item` carried and was fixed first. The
        **name** is rendered for a lone flag because the name is the information; two or
        more fall back to a count. Only ``flag`` severity is counted here: a
        ``block_point`` failure is already visible as ``N dropped`` and would otherwise
        be reported twice.
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
        # Exactly the entries `_gate_item` marks FAIL and no counter below records:
        # `checked is not False` first, so the two surfaces order the states alike.
        flagged = [
            e for e in self.gate_log
            if e.get("checked") is not False
            and not e.get("passed", True)
            and e.get("severity") == "flag"
        ]
        if len(flagged) == 1:
            parts.append(f"{flagged[0].get('gate', '?')} flagged")
        elif flagged:
            parts.append(f"{len(flagged)} flagged")
        if self.n_dropped:
            parts.append(f"{self.n_dropped} dropped")
        unchecked = sum(1 for e in self.gate_log if e.get("checked") is False)
        if unchecked:
            parts.append(f"{unchecked} unchecked")
        return ", ".join(parts) if parts else "pass"

    def describe(self) -> str:
        return f"[{self.engine}] {self.gate_summary()} — {self.sigma.describe()}"


#: Width, in points, of the frequency window whose running **median** the headroom
#: numerator minimises over. Odd by construction, so the winning median is a real
#: measured point rather than an interpolation between two.
#:
#: **Five, because two is the widest rail-adjacent cluster the corpus produces.** On
#: run ``20260915T172522Z_rung3a_fake_cast`` (84 spectra) the single-point minimum
#: landed on isolated points at |phase| → 90°, where ``Z' → 0`` drives ``tan δ`` to
#: zero whatever the sample is doing: ch13 r10 is a 300× notch one point wide, and
#: ch11 r9/r12 are *two* consecutive such points. A 3-wide median still lands on one
#: of a pair, which is why ``k = 3`` leaves 3 of the 11 false bounds standing and
#: ``k = 5`` leaves none.
#:
#: **Not a config key, deliberately.** The width is the definition of a statistic, not
#: a site knob; ``tand_headroom_mult`` is already the configurable strictness, and a
#: tunable width invites turning a diagnostic into a filter.
DEFAULT_HEADROOM_WINDOW = 5

#: |phase| at or above which the numerator point is **flagged** as rail-adjacent.
#:
#: **A flag, never a filter.** Numerator phases on rung 3a run *continuously* from
#: −78.25° to −89.997°; only 1 of 84 is within 0.005° of the rail, and cutting at 89.9°
#: would still leave 6 of the 11 false bounds standing, because ch11 r2/r11/r13 and its
#: production read are decided at −89.65…−89.83°. A hard cut is a cliff in the middle
#: of a continuum, and it does not fix the defect. What the threshold buys is the
#: operator sentence — *"the margin was decided by a phase-saturated point at 1.03 kHz"*
#: — which nobody could say about rung 3a without re-reading 84 netCDFs.
PHASE_SATURATION_DEG = 89.9


@dataclass(frozen=True)
class HeadroomDecision:
    """What :func:`decide_report_mode` decided, and what decided it.

    A 3-tuple carried the verdict and nothing about its provenance, so the fact that a
    bound had been decided by one reading at the instrument's phase rail was
    unrecoverable downstream. The flag has to travel *with* the decision or it is a log
    line nobody reads back.

    ``window == 1`` means the minimum was taken on a **single point** — the fallback
    when fewer than ``headroom_window`` points survived (an odd width is *narrowed* to
    fit a short spectrum; an even one is refused outright, since nothing passes it).
    It is recorded rather than implied, because an unknown must not be spelled like
    a checked answer
    (``SUBAGENT_RULES`` §3.1(a)).
    """

    mode: str
    provisional: bool
    headroom: float
    #: The frequency and phase of the point whose ``tan δ`` *is* the winning median —
    #: a real measured point, not an interpolation.
    numerator_f_hz: float = float("nan")
    numerator_phase_deg: float = float("nan")
    #: Whether **any** point in the winning window sits at or past
    #: :data:`PHASE_SATURATION_DEG`. The window decided the margin, so the window is
    #: what the flag describes.
    numerator_phase_saturated: bool = False
    n_saturated_in_window: int = 0
    window: int = 1
    window_f_lo_hz: float = float("nan")
    window_f_hi_hz: float = float("nan")
    n_tand_used: int = 0
    #: The floor this decision divided by, and where it came from. ``floor_rows_used``
    #: is a **count**: 2 for a bracketed pair, 1 for a one-decade extension, 0 for the
    #: legacy single anchor — which is also how a reader tells the three apart.
    floor_tand: float = float("nan")
    floor_eps_deg: float = float("nan")
    floor_rows_used: int = 0
    floor_z_anchor_ohm: float = float("nan")
    #: The local *capacitive* rows disagree with the resistive floor by more than the
    #: reference part's own rated loss. **Orthogonal to** :attr:`provisional`, which
    #: means "out of band"; this one means "in band, and the two load classes do not
    #: agree here". Neither changes :attr:`mode` — it is recorded, not acted on.
    floor_class_provisional: bool = False
    #: ``cap_eps − res_eps`` in degrees. NaN means *no comparison was possible*, which
    #: is not the same fact as a gap of zero (``SUBAGENT_RULES`` §3.1(a)).
    floor_class_gap_deg: float = float("nan")


def decide_report_mode(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
    tand_headroom_mult: float = 3.0,
    headroom_window: int = DEFAULT_HEADROOM_WINDOW,
) -> HeadroomDecision:
    """Decide value vs bound. Returns a :class:`HeadroomDecision`.

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

    **The floor is per-impedance, and it comes from the resistor ladder** (T11.41).
    :meth:`~softae.analysis.eis.envelope.InstrumentEnvelope.floor_at` brackets ``z_med``
    between the two nearest characterised *resistive* rows and takes the larger of the
    pair; one decade past the outermost row that row stands alone, and beyond it there
    is no floor and the result is ``bound_unqualified``. An envelope with no rows
    returns its single anchor from the same call, so nothing changes for one. The local
    *capacitive* rows never set the floor — they only flag
    :attr:`~HeadroomDecision.floor_class_provisional` when they disagree with it by more
    than the reference part's rated loss.

    **The numerator is a WINDOWED minimum — the most pessimistic five-point region of
    the spectrum, not its single smallest reading.** Two rules are at work here and they
    are orthogonal, which is what the earlier prose on this line conflated.

    *Direction.* ``derive_phase_table`` refuses to use a minimum for the **instrument**,
    and is right: understating the floor qualifies almost any spectrum as a value. On the
    **sample** — the numerator — the direction reverses, because understating the
    sample's own margin errs toward reporting a *bound*, which is the safe direction. A
    band median over-qualifies the sample: every state in the commissioning figure
    converges on ``tan δ ≈ 5`` at 10⁵ Hz, so it is computed exactly where nothing
    distinguishes anything. **Median for the instrument, minimum for the sample**, and
    that has not changed.

    *Robustness.* What changed is that a **single-point** minimum is not an
    understatement of the sample's margin — it is a noise statistic. At |phase| → 90°,
    ``Z' → 0``, so ``tan δ = Z'/|Z''| → 0`` whatever the sample is doing, and one such
    reading decides the whole spectrum. On rung 3a (84 spectra, 21 rounds over which R₁
    moved ~1 %) the single minimum produced 11 bounds, ten of them on one channel, which
    it flipped between value and bound fourteen times while the film did not change; the
    five-wide windowed minimum produces none, and leaves the negative-control channel's
    spread at 1.5× against 1.6× today. A minimum is defenceless against one unlucky
    point; a windowed minimum is not. Robustness is what makes the direction mean
    anything.

    Concretely: the surviving ``(f, tan δ, phase)`` triples are sorted into **ascending
    frequency** — explicitly, because the engine merely *happens* to hand them over in
    sweep order and a statistic must not depend on that — the running median is taken
    over every full ``headroom_window``-wide window, and the smallest of those medians is
    the numerator. Being odd-width, that median *is* one of the measured points, so the
    decision carries a real frequency and a real phase, and
    :data:`PHASE_SATURATION_DEG` flags when the window that decided it sat on the rail.

    **The flag is not a filter, and this function never drops a point for its phase.**
    See :data:`PHASE_SATURATION_DEG` for why a hard cut cannot do this job.

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

    .. note::
       ``measurability.tand_margin`` takes the identical single-point minimum for S2 and
       is **not** changed here — it feeds only the offline ``tools/measurability_sweep.py``
       and carries its own ledger item.
    """
    # Refused, not silently repaired. An even width has no median *point*, so there
    # is no reading to hang a frequency and a phase on, and both plausible repairs —
    # round down to the next odd, or up to the largest odd that fits — are guesses
    # about a value no caller passes. This module's posture everywhere else is to
    # refuse rather than substitute.
    if (int(headroom_window) != headroom_window
            or headroom_window < 1 or headroom_window % 2 == 0):
        raise ValueError(
            f"headroom_window must be a positive odd integer, got {headroom_window!r}")

    if cell is None:
        return HeadroomDecision("unavailable", False, float("nan"))

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
    # `gates.py`'s `_gate_phase_noise_extrapolated` now takes the same default on the
    # same predicate (`lambda _z: False`, with its own note recording that the two sites
    # were reconciled toward this one). The earlier note here said it took
    # `lambda _z: True`; that has not been so since the reconciliation, and a stale
    # cross-reference to another session's file is worse than none.
    #
    # If they ever diverge again, reconcile TOWARD THIS ONE. `SUBAGENT_RULES` §3.1(a)
    # already condemns the permissive direction on this exact subject: the envelope's
    # `phase_noise_measured` once defaulted True, "so the guard that exists to force a
    # provisional result when the floor was never measured can never fire". A default
    # that answers the safe-sounding question when it has no information is the failure
    # this whole module is written against.
    #
    # `floor_at` is the per-impedance floor (T11.41) and subsumes both scalars: with no
    # characterised rows it returns the single-anchor answer, so ONE path serves the
    # commissioned and the uncommissioned envelope alike and the fallback is reachable
    # from a test. The `getattr` chain below stays for hand-built envelope stand-ins
    # that predate the method.
    floor_at = getattr(envelope, "floor_at", None)
    if callable(floor_at):
        at = floor_at(z_med)
        in_band = bool(at.in_band)
        floor_eps_deg = float(at.eps_deg)
        floor_rows_used = len(at.rows_used)
        floor_z_anchor_ohm = float(at.z_anchor_ohm)
        floor = (math.tan(math.radians(floor_eps_deg))
                 if floor_eps_deg == floor_eps_deg else float("nan"))
    else:
        in_band = bool(
            getattr(envelope, "phase_noise_valid_at", lambda _z: False)(z_med))
        floor = getattr(envelope, "tand_floor", float("nan"))
        floor_eps_deg = float(getattr(envelope, "phase_noise_deg", float("nan")))
        floor_rows_used = 0
        floor_z_anchor_ohm = float(
            getattr(envelope, "phase_noise_at_ohm", float("nan")))

    # A flag, never a floor and never a refusal: where the local capacitor rows exceed
    # the resistive floor by more than the reference part's rated loss, the verdict
    # says so and keeps the resistive number (T11.41 §3.3).
    consistency = getattr(envelope, "class_consistency", None)
    if callable(consistency):
        class_provisional, class_gap_deg, cap_eps_deg, cap_rows = consistency(z_med)
    else:
        class_provisional, class_gap_deg = False, float("nan")
        cap_eps_deg, cap_rows = float("nan"), ()

    if not measured or not (floor == floor) or floor <= 0:
        logger.info("eis_reported_as_bound",
                    reason="phase noise unmeasured", provisional=True)
        return HeadroomDecision("bound_unqualified", True, float("nan"))

    if class_provisional:
        from softae.analysis.eis.envelope import PHASE_CLASS_CONSISTENCY_BUDGET_DEG

        logger.info(
            "eis_phase_floor_class_disagreement", z_median_ohm=z_med,
            resistive_eps_deg=floor_eps_deg, capacitive_eps_deg=cap_eps_deg,
            gap_deg=class_gap_deg, budget_deg=PHASE_CLASS_CONSISTENCY_BUDGET_DEG,
            capacitive_rows=list(cap_rows),
            msg="the local capacitor rows exceed the resistive floor by more than the "
                "reference part's rated loss — recorded as provisional provenance, "
                "not applied: the resistor ladder remains the floor",
        )

    tand_all = loss_tangent(Z)
    finite = np.isfinite(tand_all)
    n_excluded = int(np.count_nonzero(finite & (tand_all <= 0)))
    keep = finite & (tand_all > 0)
    tand = tand_all[keep]
    if n_excluded:
        logger.info(
            "eis_tand_points_excluded", n_excluded=n_excluded, n_used=int(tand.size),
            msg="non-positive loss tangent outside the passive quadrant, excluded from "
                "the headroom numerator (see measurability.negative_conductance_count)",
        )
    if tand.size == 0:
        return HeadroomDecision("bound_unqualified", True, float("nan"))

    f_kept = np.asarray(freq, dtype=float).ravel()[keep]
    phase_kept = np.angle(Z, deg=True)[keep]

    order = np.argsort(f_kept, kind="stable")          # NaN frequencies sort last
    tand_s, f_s, phase_s = tand[order], f_kept[order], phase_kept[order]

    n = int(tand_s.size)
    k = int(headroom_window)
    if k > n:
        k = n - 1 if n % 2 == 0 else n                 # largest odd width that fits
    k = max(k, 1)

    if k <= 1:
        i_num = int(np.argmin(tand_s))
        lo = hi = i_num
    else:
        medians = np.median(
            np.lib.stride_tricks.sliding_window_view(tand_s, k), axis=1)
        lo = int(np.argmin(medians))
        hi = lo + k - 1
        # The median of an odd-width window IS one of its points, and ``argsort``'s
        # middle index names which — so the numerator is a measured reading with a
        # frequency and a phase, not a summary statistic floating free of the sweep.
        i_num = lo + int(np.argsort(tand_s[lo:hi + 1], kind="stable")[k // 2])

    headroom = float(tand_s[i_num]) / floor
    win_phase = np.abs(phase_s[lo:hi + 1])
    n_saturated = int(np.count_nonzero(
        np.isfinite(win_phase) & (win_phase >= PHASE_SATURATION_DEG)))

    provenance = dict(
        headroom_window=k,
        numerator_f_hz=float(f_s[i_num]),
        numerator_phase_deg=float(phase_s[i_num]),
        numerator_phase_saturated=bool(n_saturated),
        n_saturated_in_window=n_saturated,
        n_tand_used=n,
        floor_tand=float(floor),
        floor_eps_deg=floor_eps_deg,
        floor_rows_used=floor_rows_used,
        floor_z_anchor_ohm=floor_z_anchor_ohm,
        floor_class_provisional=bool(class_provisional),
        floor_class_gap_deg=float(class_gap_deg),
    )

    def _decide(mode: str, provisional: bool) -> HeadroomDecision:
        return HeadroomDecision(
            mode, provisional, headroom,
            numerator_f_hz=provenance["numerator_f_hz"],
            numerator_phase_deg=provenance["numerator_phase_deg"],
            numerator_phase_saturated=provenance["numerator_phase_saturated"],
            n_saturated_in_window=n_saturated,
            window=k,
            window_f_lo_hz=float(f_s[lo]),
            window_f_hi_hz=float(f_s[hi]),
            n_tand_used=n,
            floor_tand=float(floor),
            floor_eps_deg=floor_eps_deg,
            floor_rows_used=floor_rows_used,
            floor_z_anchor_ohm=floor_z_anchor_ohm,
            floor_class_provisional=bool(class_provisional),
            floor_class_gap_deg=float(class_gap_deg),
        )

    # Emitted whatever the mode, because a *value* decided at the rail is exactly as
    # worth seeing as a bound — and only one of those two would ever be looked for.
    if n_saturated:
        logger.info(
            "eis_headroom_numerator_phase_saturated",
            phase_headroom=headroom, z_median_ohm=z_med,
            window_f_lo_hz=float(f_s[lo]), window_f_hi_hz=float(f_s[hi]),
            msg="the headroom numerator's window touches the instrument's phase rail, "
                "where Re Z -> 0 drives tan delta to zero whatever the sample is doing",
            **provenance,
        )

    resolution_limited = headroom < float(tand_headroom_mult)

    if resolution_limited:
        logger.info(
            "eis_reported_as_bound", reason="loss tangent below the phase floor",
            phase_headroom=headroom, z_median_ohm=z_med, provisional=not in_band,
            **provenance,
        )
        return _decide("bound" if in_band else "bound_unqualified", not in_band)

    if not in_band:
        logger.info(
            "eis_phase_floor_extrapolated", z_median_ohm=z_med,
            phase_headroom=headroom,
            msg="value reported, but the phase floor it cleared is extrapolated",
            **provenance,
        )
    return _decide("value", not in_band)


#: Basis token for the fit-free ceiling refusal (a) reports on an unclosed arc:
#: ``σ ≤ K·max(Re Y)``, i.e. ``K / model_free_r_bulk(Z)`` (T11.46).
#:
#: **Beside** ``magnitude_ceiling`` because it copies that branch's shape exactly — a
#: constant over a measured extremum of the spectrum, no fit anywhere in it — and
#: **apart from** it because the two cap different quantities from different evidence:
#: ``magnitude_ceiling`` is ``K/Z_max`` off the *instrument's* magnitude window and is
#: reachable only when ε is unmeasured, while this one is ``K·max(Re Y)`` off *this
#: spectrum's* admittance and is reachable only when the arc did not close. Spelling
#: them with one token would make a ceiling's provenance unreadable, which is the
#: failure :class:`SigmaCeiling` exists to end (``SUBAGENT_RULES`` §3.1(a)).
#:
#: **Why the admittance and not the fit.** ``Re Y(0) = G_DC`` and ``Re Y`` is
#: non-decreasing in ω for every registry model, so ``max(Re Y) ≥ G_DC`` and
#: ``1/max(Re Y)`` is a LOWER bound on R — hence ``K·max(Re Y)`` is an UPPER bound on
#: σ, and ``CellConstant.sigma`` is monotone decreasing so the direction survives the
#: conversion. The fit cannot supply this: on an unclosed arc an extrapolated ``R₁`` is
#: on this rig a median **2.752× OVER**-estimate, so a ceiling built from it sits
#: *below* the truth — the one direction a ceiling must never take.
#:
#: **It carries no frequency**: a maximum over the band is not a reading at one
#: frequency, so :attr:`SigmaCeiling.f_hz` stays NaN and :meth:`SigmaReport.as_text`
#: drops its ``@f`` clause.
ARC_OPEN_CEILING = "admittance_ceiling"


@dataclass(frozen=True)
class SigmaCeiling:
    """What :func:`sigma_loss_ceiling` concluded, and what it concluded it from.

    A bare float could not distinguish *"the ceiling is 3.9e-10"* from *"no ceiling
    could be stated"*, and the second was being spelled as the first
    (``SUBAGENT_RULES`` §3.1(a)). The basis is the discriminator, and it is also the
    label: only ``magnitude_ceiling`` is a ceiling on σ itself.
    """

    #: ``K·ε·ω*·C(ω*)`` for the loss basis, ``K/Z_max`` for the magnitude basis.
    value: float = float("nan")
    #: The frequency it was evaluated at — a real measured point, not an interpolation.
    f_hz: float = float("nan")
    #: The apparent capacitance **at that frequency**, not the top-decade median.
    c_farad: float = float("nan")
    #: The ε that went in, recorded even when no ceiling came out.
    eps_rad: float = float("nan")
    #: ``"loss_at_numerator"`` | ``"magnitude_ceiling"`` | :data:`ARC_OPEN_CEILING`
    #: | ``"unavailable"``. ``magnitude_ceiling`` is reachable **only** when ε is
    #: unmeasured and :data:`ARC_OPEN_CEILING` **only** from ``engine.py``'s refusal
    #: (a) — this function never produces the latter; every other way of failing to
    #: produce a number is ``unavailable`` with a :attr:`reason`.
    basis: str = "unavailable"
    #: Why, when :attr:`basis` is ``"unavailable"``. Empty otherwise. An unknown has to
    #: say which unknown it is, or it reads as a checked answer (``SUBAGENT_RULES``
    #: §3.1(a)) — the exact failure this class was introduced to end.
    reason: str = ""


def _capacitance_at(freq: np.ndarray, Z: np.ndarray, f_hz: float) -> float:
    """``C_app`` at the measured point nearest ``f_hz`` in log-frequency, or NaN.

    Nearest rather than exact because nothing guarantees the caller's frequency is a
    member of the array it passes; on the production path it always is, since
    ``numerator_f_hz`` is read out of this same sweep.
    """
    from softae.analysis.eis.admittance import apparent_capacitance

    f = np.asarray(freq, dtype=float)
    C = apparent_capacitance(f, Z)
    ok = np.isfinite(f) & np.isfinite(C) & (f > 0) & (C > 0)
    idx = np.flatnonzero(ok)
    if not idx.size:
        return float("nan")
    j = idx[int(np.argmin(np.abs(np.log10(f[idx]) - np.log10(float(f_hz)))))]
    return float(C[j])


def sigma_loss_ceiling(
    freq: np.ndarray,
    Z: np.ndarray,
    *,
    envelope: Any,
    cell: Any,
    at_f_hz: float,
) -> SigmaCeiling:
    """A ceiling on the dielectric-loss conductance, ``K·ε·ω*·C(ω*)`` (T11.34).

    **This is not a bound on σ_DC, and it is not evaluated at ``min(f)``.** Its
    predecessor, ``sigma_upper_bound``, was a second copy of
    :meth:`InstrumentEnvelope.sigma_min` — the *detection floor* — taken at the
    frequency that makes a floor smallest and rendered with a ``≲``. On the eleven
    bound spectra of ``20260915T172522Z_rung3a_fake_cast`` it read 3.9e-10 S/cm
    against a fit-implied 3.1e-06…4.9e-05: three to five decades under the point
    estimate it was supposed to cap.

    The physics fixes ω. The bound branch fires because ``tan δ = G/(ωC)`` fell below
    the instrument's resolvable floor ε **at one frequency**, so what that licenses is
    ``G(ω*) ≤ ε·ω*·C(ω*)`` at *that* ω — the headroom numerator's. Asserting it at
    ``min(f)`` asserts a comparison that was never made, where the threshold is
    hundreds to thousands of times smaller. Taking the most favourable ω is right for
    a floor and inverted for a ceiling (``SUBAGENT_RULES`` §3.3).

    ``at_f_hz`` is therefore **required and has no fallback**: missing ⇒
    ``basis="unavailable"``, value NaN. Falling back to ``min(f)`` would spell an
    unknown with the same token as a checked answer, and it is the specific wrong
    answer this function exists to remove.

    ``K/Z_max`` survives as the ε-unmeasured branch under ``basis="magnitude_ceiling"``
    — a genuine ceiling on σ, and the only branch entitled to the word. Never the
    withdrawn ``K/Z_φ``, and **never reached for any reason other than an unmeasured
    ε**: an earlier revision let a missing capacitance fall through to it, which
    labelled an unknown with the one basis that means *checked*.
    """
    if cell is None:
        return SigmaCeiling(reason="no cell constant")
    K = getattr(cell, "K_per_cm", float("nan"))
    if not (K == K):
        return SigmaCeiling(reason="no cell constant")

    eps = float(getattr(envelope, "eps_rad", float("nan")))

    # ε unmeasured is the ONLY route to the magnitude ceiling. Every other way of
    # failing to produce a loss ceiling is ``unavailable``: falling through to
    # ``K/Z_max`` because some *other* input was missing would answer a question
    # nobody asked and label it a checked result.
    if eps != eps:
        z_max = float(getattr(envelope, "z_max_ohm", float("nan")))
        if z_max == z_max and z_max > 0:
            return SigmaCeiling(value=float(K) / z_max, eps_rad=eps,
                                basis="magnitude_ceiling")
        return SigmaCeiling(reason="epsilon unmeasured and no magnitude window")

    if not (at_f_hz == at_f_hz and at_f_hz > 0):
        return SigmaCeiling(eps_rad=eps, reason="no numerator frequency")

    C = _capacitance_at(freq, Z, at_f_hz)
    if not (C == C and C > 0):
        return SigmaCeiling(f_hz=float(at_f_hz), eps_rad=eps,
                            reason="no capacitance at the numerator frequency")

    omega = 2.0 * math.pi * float(at_f_hz)
    return SigmaCeiling(value=float(eps * omega * C * K), f_hz=float(at_f_hz),
                        c_farad=C, eps_rad=eps, basis="loss_at_numerator")
