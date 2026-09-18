"""What this instrument can actually measure, as opposed to what it claims.

Every threshold in the gate framework derives from a handful of numbers. This module
holds them, records **whether each was measured or assumed**, and derives the limits
that follow.

.. warning::
   **The "phase-reliable ceiling" ``Z_φ ≈ 5×10⁷ Ω`` is withdrawn.** Earlier drafts of
   ``docs/EIS_SUITE_OVERHAUL_new.md`` asserted a ceiling roughly twenty times below the
   magnitude ceiling, plus a derived conductance offset near 1×10⁻⁴ S. Both were
   inferred from measurements taken with the **reference electrode floating**
   (``_new`` §3.7). With RE correctly connected, a 9.9 kΩ standard returned 45/45
   valid points from 1.2 Hz to 200 kHz with **zero** negative ``Re Z`` and a maximum
   phase magnitude of 79.9°.

   There is currently **no evidence** for a phase-reliable ceiling below the magnitude
   ceiling, so this module no longer carries one and nothing gates on one. A quadrant
   violation is attributed to the state of the control loop first (F13/R19).

.. note::
   **Existing film spectra are not compromised, and no re-measurement is needed.** On
   this board the reference stripe lies between CE and WE with no direct electrical
   contact — it reaches the cell only through material spanning the coplanar gap. A
   cast film therefore *closes the loop itself*, which is why sample measurements were
   valid all along while two-terminal reference components across CE/WE (touching no
   RE stripe) produced the pathology.

   Bare-board and open blanks are a different matter, and the difference is structural
   rather than a fault: with only air between the stripes the loop cannot close. Such a
   spectrum is a real measurement of the inter-stripe geometry — useful as a probe of
   the board's own ``C_cell`` — but it is **not** a fixture open and cannot serve as an
   OSL term. Getting a genuine open on this geometry requires RE tied to CE at the
   connector.

The measured envelope (``_new`` §2), all with RE correctly connected:

==========================  ==================  =========================
Quantity                    Measured            Conditions
==========================  ==================  =========================
Magnitude accuracy          **+0.32 %**         9.9 kΩ, flat over 5 decades
Phase noise ``ε``           **0.149°**          9.9 kΩ, *resistive*
``tan δ`` floor             **≈0.0026**         derived from ε
Instrument stray C          **5.85 ± 0.41 pF**  MUX + cabling only
Low-frequency limit         **1.2 Hz**          no degradation at 10⁴ Ω
==========================  ==================  =========================

Two consequences that reverse earlier guidance:

**Low frequency is now strongly favoured.** ``σ_min ≈ K·ε·ω·C_cell`` scales with ω,
so the smallest detectable conductivity *improves* as frequency falls — roughly
1×10⁻⁹ S/cm at 1 Hz against 1×10⁻⁷ S/cm at 100 Hz for a 1.5 nF cell. The previous
formula ``σ_min ≈ K·ε/Z_φ`` was frequency-independent and far more pessimistic.

**The nF-scale stray capacitance is the board, not the wiring.** Only ~6 pF lives in
the MUX and cabling, so guarding the cable harness cannot move the 0.35–1.5 nF cell
term. That is a hardware-priority conclusion, not an analysis one.

.. note::
   ``ε`` was measured on a **resistive** load at 10⁴ Ω. Films sit at 10⁶–10⁸ Ω and are
   capacitive — both conditions differ. :meth:`InstrumentEnvelope.phase_noise_valid_at`
   is what stops that number being extrapolated silently; until a low-loss capacitor
   and high-value resistors are run, a bound taken far from 10⁴ Ω is *provisional*.
   The capacitor tried so far (marked "102") measured ~150 nF with ``tan δ ≥ 0.179``,
   some 70× above the instrument floor — unusable as a phase reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import structlog

logger = structlog.get_logger(__name__)


#: Lowest settable frequency (Hz).
DEFAULT_F_MIN_HZ = 0.016
#: Highest settable frequency (Hz).
DEFAULT_F_MAX_HZ = 200_000.0
#: Lowest frequency verified free of degradation (Hz), at 10⁴ Ω.
DEFAULT_F_VERIFIED_LO_HZ = 1.2
#: Lower bound of reproducible |Z| (Ω).
DEFAULT_Z_MIN_OHM = 10.0
#: Upper bound of reproducible |Z| (Ω). Upper decades are **not yet verified**.
DEFAULT_Z_MAX_OHM = 1e9
#: Phase noise (degrees), measured on a 9.9 kΩ resistive load.
DEFAULT_PHASE_NOISE_DEG = 0.149
#: Impedance (Ω) at which the phase noise above was measured.
DEFAULT_PHASE_NOISE_AT_OHM = 9.9e3
#: How far from that impedance the measured phase noise may be trusted, in decades.
#: One decade is a judgement call, not a measurement — the specification says only
#: that 10⁶–10⁸ Ω "must be re-measured before being trusted".
DEFAULT_PHASE_NOISE_VALID_DECADES = 1.0
#: Magnitude accuracy (%), measured on a 9.9 kΩ standard across five decades.
DEFAULT_MAGNITUDE_ACCURACY_PCT = 0.32
#: Instrument-path stray capacitance (F) — MUX and cabling only.
#:
#: **A board median, not a channel.** Seven tied open blanks (ch17–23, 2026-08-06) span
#: 10.2–24.7 pF, a 2.4× spread across nominally identical stripes, while repeating to
#: 1% on any one channel — so the spread is real per-channel variation, not scatter.
#: A per-channel :attr:`~softae.analysis.eis.calibration.CalibrationSet.C_stray_F`
#: supersedes this wherever one exists; this is only the no-calibration fallback, and
#: a fallback must not be the most optimistic channel (a low ``C_stray`` overstates
#: ``Z_φ`` and so over-reports how many σ are values rather than bounds).
#:
#: The prior 5.85 pF was a three-electrode figure, i.e. a floating-divider artifact
#: (F17) rather than a measurement of anything.
DEFAULT_STRAY_C_INSTRUMENT_F = 18.5e-12
#: Cell/board shunt capacitance (F). Belongs to the PCB, **not** the instrument path,
#: and is dispersive rather than a clean capacitance.
DEFAULT_C_CELL_F = 1.0e-9
#: Largest excitation amplitude (mV) shown not to saturate the current range.
#:
#: Overhaul §3.8 measured this directly: at 100 mV two channels returned **bitwise
#: identical** |Z| at two frequencies, 13 low-frequency points collapsed onto 8–9
#: discrete values, and the |Z| ratio deviated up to 2.3× across 6–30 kHz — peaking at
#: the impedance *minimum*, where the current is largest. Deviation peaking at peak
#: current is current-range saturation, not interfacial nonlinearity.
#:
#: **The remedy is a higher current range, not a lower voltage.** A sample-side divider
#: analysis said the film could tolerate ~40 mV, and up to volts in the high-impedance
#: regime; that analysis was right about the sample and irrelevant to the real limit,
#: which is the instrument front end.
#:
#: The deployed MethodSCRIPT runs **15 mV**, comfortably below this, so the shipped
#: acquisition is not at risk — this value exists to catch a future change to it.
DEFAULT_MAX_AMPLITUDE_MV = 25.0

#: How far a *capacitive* row may exceed the local *resistive* floor before the verdict
#: is flagged provisional (degrees). See :meth:`InstrumentEnvelope.class_consistency`.
#:
#: **A rated loss, not a chosen threshold.** A C0G/NP0 reference is specified at
#: ``tan δ < 1e-3``, i.e. its own contribution to the measured angle is 0.06°; anything
#: past the resistor floor plus that is by construction not the part. The neighbouring
#: ``commissioning.PHASE_REFERENCE_MAX_EPS_DEG`` (15.0°) answers a *different* question
#: — "is this part a low-loss capacitor at all" — and reused here it would be a check
#: that cannot fail, since the largest capacitive ε ever recorded on this rig is 4.536°
#: (``SUBAGENT_RULES`` §3.1(e)).
#:
#: It lives here rather than beside its sibling because :mod:`softae.workflows` imports
#: :mod:`softae.analysis`, never the reverse, and this module is the one that reads it.
PHASE_CLASS_CONSISTENCY_BUDGET_DEG = 0.06

#: Measurement roles judged against the commissioned **sample** ``|Z|`` window.
#:
#: An allow-list rather than a list of exemptions, because an undeclared or unrecognised
#: role must not be spelled the same way as "this is a sample" (``SUBAGENT_RULES`` §3.1(a)).
#: The DataStore already carries a role outside :data:`MEASUREMENT_ROLES` —
#: ``reference_r_misplaced_lead``, four rows — and an exemption list would have narrowed
#: those reference resistors against a window derived from their own siblings.
SAMPLE_ROLES = frozenset({"sample", "drift_repeat"})


def magnitude_window_applies(role: str | None) -> bool:
    """Whether the commissioned sample ``|Z|`` window may judge a *role*'s spectrum.

    **A blank is not a sample.** The commissioned window on this fixture is
    795.6 Ω–1.4456×10⁸ Ω, derived from **reference resistors**; a short blank sits near
    7.7 Ω, four decades below its floor, so the window genuinely does not certify it —
    and judging a commissioning artifact by a window its own siblings produced is
    circular whatever the numbers say.

    Measured: ``measurement_id = 3490`` is ``mux16.toml``'s own ``sources.blank_short``,
    and under the sample window it keeps **2 of 10** surviving points against a
    ``min_fit_pts`` of 8. It is not only the ``blank_short`` — all 25 commissioning rows
    in the corpus are judged this way by anything that replays stored spectra through
    ``analyze_spectrum``, ``shadow_rehearse`` included.

    Commissioning **re-derivation** is not among them, and the spec that proposed this
    carve-out was wrong about that: ``workflows/commissioning.py`` runs no gate stack at
    all, calling :mod:`softae.analysis.eis.calibration_derive` directly on raw
    ``(f, Z)``. So the carve-out protects the replay paths, not the derivation — and a
    blank is judged instead by ``derive_short``'s own plausibility check, which is the
    check that actually knows what a short looks like.
    """
    return str(role or "").strip().lower() in SAMPLE_ROLES


@dataclass(frozen=True)
class PhaseFloorRow:
    """One characterised ``(|Z|, ε)`` point, with the class and part behind it.

    Defined here rather than imported from :mod:`softae.analysis.eis.calibration` so
    this module keeps importing nothing from ``calibration*``: the dependency runs one
    way, lazily, from ``calibration_set.envelope()`` into this module.
    """

    z_ohm: float
    eps_deg: float
    #: ``"resistive"`` or ``"capacitive"`` — which family of part measured this row.
    load_kind: str = "resistive"
    #: Acquisition this row was derived from; ``-1`` where none was recorded. Provenance
    #: only, so it gates nothing and a ``-1`` never suppresses a floor.
    source_measurement_id: int = -1


class PhaseFloorAt(NamedTuple):
    """What :meth:`InstrumentEnvelope.floor_at` concluded, and what concluded it.

    :attr:`z_anchor_ohm` is carried beyond the bare verdict because it is the first
    thing an operator asks of a floor — *measured where?* — and it is unrecoverable
    downstream once the rows have been reduced to one number.
    """

    eps_deg: float = float("nan")
    in_band: bool = False
    #: ``source_measurement_id`` of every row that decided this floor, low ``|Z|``
    #: first. Empty on the legacy single-anchor path, where there are no rows to name.
    rows_used: tuple[int, ...] = ()
    z_anchor_ohm: float = float("nan")


@dataclass(frozen=True)
class InstrumentEnvelope:
    """The measured envelope, with provenance flags that gate how it may be used.

    The ``*_measured`` flags are load-bearing rather than documentary: they select
    between qualified and provisional reporting in :mod:`softae.analysis.eis.report`.
    """

    f_min_hz: float = DEFAULT_F_MIN_HZ
    f_max_hz: float = DEFAULT_F_MAX_HZ
    f_verified_lo_hz: float = DEFAULT_F_VERIFIED_LO_HZ
    z_min_ohm: float = DEFAULT_Z_MIN_OHM
    z_max_ohm: float = DEFAULT_Z_MAX_OHM
    phase_noise_deg: float = DEFAULT_PHASE_NOISE_DEG
    phase_noise_at_ohm: float = DEFAULT_PHASE_NOISE_AT_OHM
    phase_noise_valid_decades: float = DEFAULT_PHASE_NOISE_VALID_DECADES
    phase_noise_load: str = "resistive"
    magnitude_accuracy_pct: float = DEFAULT_MAGNITUDE_ACCURACY_PCT
    stray_C_instrument_F: float = DEFAULT_STRAY_C_INSTRUMENT_F
    c_cell_F: float = DEFAULT_C_CELL_F
    max_amplitude_mV: float = DEFAULT_MAX_AMPLITUDE_MV
    phase_noise_measured: bool = True
    magnitude_window_measured: bool = False
    measured_at: str = ""
    #: Every characterised row, with its own load class — the per-impedance floor.
    #:
    #: **Appended last, and defaulted empty on purpose.** Empty means "this envelope
    #: predates the column", which :meth:`floor_at` answers with the legacy
    #: single-anchor result rather than with a refusal; it must never read as "no row
    #: qualifies here" (``SUBAGENT_RULES`` §3.1(a)).
    phase_rows: tuple[PhaseFloorRow, ...] = ()

    def amplitude_is_safe(self, amplitude_mV: float | None) -> bool | None:
        """Whether an excitation amplitude sits below the measured saturation limit.

        ``None`` when the amplitude was not recorded — undeclared is not the same as
        safe, and F14 is invisible in a single spectrum, so an unrecorded amplitude
        must read as *unknown* rather than pass.
        """
        try:
            amp = float(amplitude_mV)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not (amp > 0):
            return None
        return amp <= self.max_amplitude_mV

    # ── Derived limits ───────────────────────────────────────────────────────

    @property
    def eps_rad(self) -> float:
        """Phase noise in radians — NaN when unmeasured."""
        d = self.phase_noise_deg
        return math.radians(d) if d == d else float("nan")

    @property
    def tand_floor(self) -> float:
        """Smallest resolvable loss tangent, ``tan δ_min ≈ ε`` (framework §1.7).

        0.0026 for the measured 0.149°.
        """
        eps = self.eps_rad
        return math.tan(eps) if eps == eps else float("nan")

    def phase_noise_valid_at(self, z_ohm: float) -> bool:
        """Can the measured ``ε`` be trusted at this impedance?

        ``ε`` was characterised on a resistive load at 10⁴ Ω. Films sit three to four
        decades higher and are capacitive. Extrapolating across that gap without
        saying so is exactly how the withdrawn ``Z_φ`` came to be believed, so a
        result outside the calibrated band is reported as *provisional*.
        """
        if not self.phase_noise_measured:
            return False
        try:
            z = abs(float(z_ohm))
        except (TypeError, ValueError):
            return False
        if z <= 0 or self.phase_noise_at_ohm <= 0:
            return False
        return abs(math.log10(z / self.phase_noise_at_ohm)) <= float(
            self.phase_noise_valid_decades)

    # ── The floor, by load class and locality ────────────────────────────────

    def floor_at(self, z_ohm: float, *,
                 load_kind: str = "resistive") -> PhaseFloorAt:
        """The phase floor at this ``|Z|``, from the rows of one load class.

        **Resistive by default**, because the resistor ladder *is* the instrument
        characterisation: a resistor's true phase is exactly zero, so what the
        instrument reports on one is its own error. The capacitor family is measured
        with its own loss unknown at the 0.06° level and off the mux16 board, so it
        checks the answer (:meth:`class_consistency`) and never sets it.

        Three regimes, in order:

        1. **No rows** — the legacy single-anchor answer, so one path serves both the
           commissioned and the uncommissioned envelope and the fallback is reachable
           from a test rather than merely believed.
        2. **Bracketed** — the nearest row below and the nearest above, and the floor
           is the ``max`` of that *pair*. **No interpolant**: two load classes and a
           fixture make a smooth curve between adjacent rows a fiction, and the live
           table's interpolant crosses 4.35° → 0.18° between neighbours.
        3. **Outside by at most** ``phase_noise_valid_decades`` — the outermost row of
           the class stands alone. Beyond that, ``NaN`` and ``in_band = False``, which
           is what puts :func:`~softae.analysis.eis.report.decide_report_mode` onto
           ``bound_unqualified``.
        """
        rows = tuple(self.phase_rows)
        if not rows:
            return PhaseFloorAt(self.phase_noise_deg,
                                self.phase_noise_valid_at(z_ohm),
                                (), self.phase_noise_at_ohm)
        try:
            z = abs(float(z_ohm))
        except (TypeError, ValueError):
            return PhaseFloorAt()
        if not (z > 0):
            return PhaseFloorAt()

        want = str(load_kind).strip().lower()
        candidates = sorted(
            (r for r in rows
             if str(r.load_kind).strip().lower() == want
             and r.z_ohm > 0 and r.eps_deg == r.eps_deg),
            key=lambda r: r.z_ohm,
        )
        if not candidates:
            return PhaseFloorAt()

        below = [r for r in candidates if r.z_ohm <= z]
        above = [r for r in candidates if r.z_ohm >= z]
        if below and above:
            lo, hi = below[-1], above[0]
            pair = (lo,) if lo is hi else (lo, hi)
            win = max(pair, key=lambda r: r.eps_deg)
            return PhaseFloorAt(
                float(win.eps_deg), True,
                tuple(int(r.source_measurement_id) for r in pair),
                float(win.z_ohm))

        edge = candidates[0] if not below else candidates[-1]
        if abs(math.log10(z / edge.z_ohm)) <= float(self.phase_noise_valid_decades):
            return PhaseFloorAt(float(edge.eps_deg), True,
                                (int(edge.source_measurement_id),), float(edge.z_ohm))
        return PhaseFloorAt()

    def tand_floor_at(self, z_ohm: float, *,
                      load_kind: str = "resistive") -> float:
        """:meth:`floor_at`'s ε as a loss tangent — ``NaN`` where there is no floor."""
        eps = self.floor_at(z_ohm, load_kind=load_kind).eps_deg
        return math.tan(math.radians(eps)) if eps == eps else float("nan")

    def class_consistency(
        self, z_ohm: float, *,
        budget_deg: float = PHASE_CLASS_CONSISTENCY_BUDGET_DEG,
    ) -> tuple[bool, float, float, tuple[int, ...]]:
        """``(provisional, gap_deg, cap_eps_deg, cap_rows)`` — a flag, never a floor.

        The local capacitor rows are bracketed by the same rule and compared against
        the local resistor floor. Past that floor plus the part's own *rated* loss
        (:data:`PHASE_CLASS_CONSISTENCY_BUDGET_DEG`) the excess is not the part; which
        of instrument or auxiliary fixture it is has been ruled unclear, so it is
        **recorded rather than acted on**. This never returns a floor and never refuses.

        A ``NaN`` gap means *no comparison was possible* — no rows at all, or no
        capacitive coverage here — which must not be spelled like a gap of zero.
        """
        if not self.phase_rows:
            return False, float("nan"), float("nan"), ()
        res = self.floor_at(z_ohm, load_kind="resistive")
        cap = self.floor_at(z_ohm, load_kind="capacitive")
        if not (res.eps_deg == res.eps_deg and cap.eps_deg == cap.eps_deg):
            return False, float("nan"), float(cap.eps_deg), cap.rows_used
        gap = float(cap.eps_deg) - float(res.eps_deg)
        return bool(gap > float(budget_deg)), gap, float(cap.eps_deg), cap.rows_used

    def sigma_min(self, K_per_cm: float, freq_hz: float,
                  c_cell_F: float | None = None) -> float:
        """Smallest detectable conductivity, ``σ_min ≈ K·ε·ω·C_cell`` (S/cm).

        **Frequency-dependent, and low frequency is strongly favoured** — the reverse
        of the withdrawn ``K·ε/Z_φ`` form, which was frequency-independent and about
        two decades more pessimistic. At K = 50 /cm, ε = 2.6e-3 rad and a 1.5 nF cell
        this runs from ~1e-9 S/cm at 1 Hz to ~1e-7 S/cm at 100 Hz, which is why even
        weakly conducting films may be reachable if the sweep goes low enough.

        **This is a detection FLOOR at the most favourable ω, and a ceiling must never
        be re-derived from it** — :func:`softae.analysis.eis.report.sigma_loss_ceiling`
        owns that, evaluated at the frequency where the comparison was actually made.
        Taking the smallest ω is correct here and inverted there (T11.34).
        """
        eps = self.eps_rad
        C = float(c_cell_F if c_cell_F is not None else self.c_cell_F)
        if eps != eps or C <= 0:
            return float("nan")
        omega = 2.0 * math.pi * float(freq_hz)
        return float(K_per_cm) * eps * omega * C

    def sigma_floor(self, K_per_cm: float) -> float:
        """Conductivity corresponding to the magnitude ceiling, ``K/Z_max``."""
        if self.z_max_ohm <= 0:
            return float("nan")
        return float(K_per_cm) / self.z_max_ohm

    def describe(self) -> str:
        """One line stating what is measured and what is still assumed."""
        parts = [
            f"{self.f_min_hz:g}–{self.f_max_hz:g} Hz "
            f"(verified to {self.f_verified_lo_hz:g} Hz)",
            f"|Z| {self.z_min_ohm:g}–{self.z_max_ohm:g} Ω",
        ]
        if self.phase_noise_measured:
            parts.append(
                f"phase noise {self.phase_noise_deg:.3f}° "
                f"(tan δ floor {self.tand_floor:.4f}) at "
                f"{self.phase_noise_at_ohm:.3g} Ω {self.phase_noise_load}"
            )
        else:
            parts.append("phase noise UNMEASURED")
        base = ", ".join(parts)
        if not self.magnitude_window_measured:
            base += ". Upper |Z| decades not yet verified."
        return f"Instrument envelope: {base}"


def instrument_envelope(config: dict[str, Any] | None = None) -> InstrumentEnvelope:
    """Read ``[eis.instrument]`` — the single parse point for the envelope.

    This is the *fallback* source. Once a calibration set exists,
    :meth:`softae.analysis.eis.calibration.CalibrationSet.envelope` supersedes it with
    values derived from measured reference components.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("eis", {}).get("instrument", {}) or {}
        except Exception:
            config = {}

    def _f(key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return default

    return InstrumentEnvelope(
        f_min_hz=_f("f_min_hz", DEFAULT_F_MIN_HZ),
        f_max_hz=_f("f_max_hz", DEFAULT_F_MAX_HZ),
        f_verified_lo_hz=_f("f_verified_lo_hz", DEFAULT_F_VERIFIED_LO_HZ),
        z_min_ohm=_f("z_min_ohm", DEFAULT_Z_MIN_OHM),
        z_max_ohm=_f("z_max_ohm", DEFAULT_Z_MAX_OHM),
        phase_noise_deg=_f("phase_noise_deg", DEFAULT_PHASE_NOISE_DEG),
        phase_noise_at_ohm=_f("phase_noise_at_ohm", DEFAULT_PHASE_NOISE_AT_OHM),
        phase_noise_valid_decades=_f(
            "phase_noise_valid_decades", DEFAULT_PHASE_NOISE_VALID_DECADES),
        phase_noise_load=str(config.get("phase_noise_load", "resistive")),
        magnitude_accuracy_pct=_f(
            "magnitude_accuracy_pct", DEFAULT_MAGNITUDE_ACCURACY_PCT),
        stray_C_instrument_F=_f(
            "stray_C_instrument_F", DEFAULT_STRAY_C_INSTRUMENT_F),
        c_cell_F=_f("c_cell_F", DEFAULT_C_CELL_F),
        max_amplitude_mV=_f("max_amplitude_mV", DEFAULT_MAX_AMPLITUDE_MV),
        phase_noise_measured=bool(config.get("phase_noise_measured", True)),
        magnitude_window_measured=bool(
            config.get("magnitude_window_measured", False)),
        measured_at=str(config.get("measured_at", "")),
    )


def recommend_preset(
    presets: dict[str, dict[str, Any]] | None = None,
    envelope: InstrumentEnvelope | None = None,
) -> tuple[str | None, list[str]]:
    """Rank sweep presets by how low they reach. Returns ``(best, warnings)``.

    **This reverses earlier guidance.** A previous version of this module warned that
    presets reaching below ~9 Hz were unusable, derived from the now-withdrawn ``Z_φ``.
    The opposite is true: ``σ_min ∝ ω`` (see :meth:`InstrumentEnvelope.sigma_min`), the
    instrument is verified to 1.2 Hz with no degradation, and so **the preset that
    reaches lowest resolves the smallest conductivity**.

    Warnings are emitted for presets that stop *above* the verified low-frequency
    limit while leaving detection headroom unused, and for any that reach below the
    instrument's own floor.
    """
    env = envelope or instrument_envelope()
    if presets is None:
        try:
            from softae.config import loader

            presets = loader.eis_presets()
        except Exception:
            presets = {}

    warnings: list[str] = []
    best: str | None = None
    best_lo = float("inf")

    for name, cfg in (presets or {}).items():
        try:
            f_lo_hz = float(cfg.get("f_lo_mHz", 0.0)) / 1000.0
        except (TypeError, ValueError, AttributeError):
            continue
        if f_lo_hz <= 0:
            continue
        if f_lo_hz < env.f_min_hz:
            warnings.append(f"{name}: {f_lo_hz:g} Hz is below the instrument floor")
            continue
        if f_lo_hz < best_lo:
            best_lo, best = f_lo_hz, name
        if f_lo_hz > 10.0 * env.f_verified_lo_hz:
            warnings.append(
                f"{name}: stops at {f_lo_hz:g} Hz, well above the {env.f_verified_lo_hz:g} Hz "
                f"verified limit — σ_min scales with frequency, so this leaves "
                f"detection headroom unused on low-conductivity films"
            )

    if warnings:
        logger.info("eis_preset_review", best=best, best_f_lo_hz=best_lo,
                    warnings=warnings)
    return best, warnings
