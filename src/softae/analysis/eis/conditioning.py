"""Repair the artefacts the fixture put into a spectrum, before anything judges it.

Conditioning is **not** gating, and the estimator-pipeline document (§2.1) records the
muddle between the two as the reason this chain never reached ``src/``. A gate asks
*may this spectrum be used?* and answers by refusing. A conditioner asks *which of these
points are the instrument rather than the sample?* and answers by removing them, so that
what survives is a measurement. **Nothing here refuses.** A spectrum that arrives
unusable leaves unusable, with a report saying what was taken off it; the refusal is
:mod:`softae.analysis.eis.gates`' job and happens afterwards.

The repair is worth making. Measured against the reference-resistor answer key (§2.1)
with the same thin preparation on both sides and the HF-inductive truncation the only
thing varied::

    preparation      R_sol MAE    sigma MAE
    truncated          3.544 %      0.98 %
    not truncated      3.504 %      2.41 %

Roughly 2.5× on σ, and nothing measurable on ``R_sol`` — a genuine refinement of the
quantity this project actually reports, and not a precondition for anything downstream.

An earlier version of this docstring claimed far more: a five-step ladder running
6254 % → 6295 % → 6295 % → 2.88 % → 1.56 %, three orders of magnitude at the truncation
row, described as the largest single effect measured anywhere in this project's EIS
work. **It was not, and the ladder is not restated here because it does not reproduce.**
That run went through the constrained-fit harness while its ``fit_shared`` seeded each
spectrum from the previous one's fit — an order dependence, since fixed — so what the
ladder ranked was largely the seeding, and the huge early rows were that bug rather than
the raw spectra. The two-row table above is the same question asked of the fixed module.
Nothing in this file rests on the retracted numbers, and no reader should be left
believing one step carries the chain.

The chain, in order, and where each step's implementation actually lives:

======  =============================  ================================================
step    stage                          implementation
======  =============================  ================================================
1       physical-domain filter         :func:`physical_mask` (here — it is three lines)
2       short / fixture series         :func:`short_correct` (here), constants from
                                       :class:`~softae.analysis.eis.calibration.CalibrationSet`
3       ``|Z|`` magnitude window       :func:`~softae.analysis.eis.gates.gate_magnitude`
4       HF-inductive truncation        :func:`~softae.analysis.eis.gates.gate_hf_inductive`
5       linear-K–K low-frequency trim  :func:`~softae.analysis.eis.kk.lin_kk` +
                                       :func:`~softae.analysis.eis.kk.low_frequency_run`
======  =============================  ================================================

**Steps 3–5 are borrowed, not reimplemented, and that is the point of the module.** The
prototype — ``cdata.py``, in the constrained-fit harness directory under
``docs/SubAgent docs/`` — carries its own version of each, and every one of them is the
weaker one: its ``lin_kk`` picks a ladder
order from a fixed ``3·log₁₀(f_max/f_min)`` formula, where :mod:`softae.analysis.eis.kk`
walks the order and selects on minimum median residual — a difference measured at up to
272× in the fitted ``R1``, and so in ``σ = K/R1``. Porting the prototype's arithmetic
alongside the shipped arithmetic would have created two answers to the same question and
no way to tell which one produced a given number. So the wrappers here exist to *adapt
interfaces*, not to compute: each builds the ``ctx`` those functions read, hands them the
points still surviving, and consumes ``GateResult.mask``.

**One step of the prototype's chain is deliberately absent.** ``cdata.next_Yshunt``
subtracts a fixture shunt admittance ``G_fixture + jωC_stray``. That is a *circuit-model*
term belonging to the constrained-fit module, not a conditioning step: it changes what
the remaining points mean rather than deciding which points are real, and §2's Stage-1
row does not list it. It is not ported here and must not be added.

Nothing is ever dropped silently — :class:`ConditioningResult` carries ``n_in``,
``n_out`` and a per-stage :class:`StageReport`, following ``gates.GateResult`` and
``kk.LinKKResult``. A stage that *declined* to act is reported as such rather than as a
stage that found nothing, because those two are the failure pair this codebase keeps
being bitten by: unknown wearing the same token as clean.

.. note::
   **This module ships inert.** Nothing in ``src/softae`` imports it, and
   ``tests/test_eis_conditioning.py`` asserts that as a contract. Wiring it into the
   analysis path is a separate decision with its own evidence bar — §2.1 measures the
   chain against the reference-resistor answer key, not against films — and belongs to
   later work.

.. note::
   **Do not "tidy" the harness path above into one token, and do not name the
   constrained-fit module in this file in its importable spelling.** That module's test
   file enforces *its* inertness with a plain substring search for its own name over all
   of ``src/softae``, so the literal — even inside a docstring, even in a path, even in a
   comment explaining this — fails another module's test from within this one. It did,
   twice: once in the path, and once in the note added to explain the first. The
   hyphenated prose spelling used throughout this file is the fix.

   This module's own inertness test cannot use that mechanism in reverse, for the
   mirror-image reason recorded there: "conditioning" is ordinary English and already
   appears as prose in seven ``src/softae`` modules, so it matches an import-shaped
   regex instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from softae.analysis.eis.gates import gate_hf_inductive, gate_magnitude
from softae.analysis.eis.kk import LinKKResult, lin_kk, low_frequency_run
from softae.analysis.eis.policy import build_context

if TYPE_CHECKING:  # pragma: no cover - typing only
    from softae.analysis.eis.calibration import CalibrationSet

logger = structlog.get_logger(__name__)

#: Stage names. Constants rather than literals because they are the keys
#: :meth:`ConditioningResult.dropped` is looked up by, and a typo in a lookup would
#: return zero — "this stage removed nothing" — for a stage that removed half the sweep.
STAGE_PHYSICAL = "physical"
STAGE_SHORT = "short_correction"
STAGE_WINDOW = "magnitude_window"
STAGE_HF_INDUCTIVE = "hf_inductive"
STAGE_KK = "kk_trim"

#: The order the stages run in, and the order they appear in the report.
STAGE_ORDER = (STAGE_PHYSICAL, STAGE_SHORT, STAGE_WINDOW, STAGE_HF_INDUCTIVE, STAGE_KK)

#: Surviving-point floor below which a stage stops removing points.
#:
#: Adapted from the prototype's ``if mask.sum() < 6``. It is **not** a fitness bar —
#: :func:`~softae.analysis.eis.gates.gate_min_points` owns that question and wants 8 —
#: it is the point at which a *repair* has stopped being a repair. A conditioner that
#: deletes a spectrum has not conditioned it, and the honest output is the spectrum plus
#: a report saying the stage declined, which lets the gate downstream refuse it for the
#: real reason rather than for "only 2 points survived".
#:
#: The prototype's fallback differs from this one and the divergence is deliberate: on
#: shortfall it reverted to the *window-only* mask, discarding the physical filter's
#: verdict as collateral. Here each stage is declined individually, so an earlier stage's
#: correct removal is never undone by a later stage's over-reach.
MIN_CONDITIONED_POINTS = 6

#: Points needed before the K–K ladder is worth fitting at all.
#:
#: The prototype's threshold, kept. :func:`~softae.analysis.eis.kk.lin_kk` will fit from
#: 4, but the trim it feeds is a *global* fit used to judge individual points, and on a
#: handful of points the ladder describes the noise rather than the causality.
MIN_KK_POINTS = 8

#: Default per-point K–K residual, in percent of ``|Z|``, above which a low-frequency
#: point is treated as drift rather than data. The prototype's ``kk_pct=1.0``.
DEFAULT_KK_TRIM_PCT = 1.0


@dataclass(frozen=True)
class StageReport:
    """What one conditioning stage did, including when it chose to do nothing.

    ``n_dropped == 0`` is ambiguous on its own — it is both "the spectrum was clean" and
    "the stage could not run" — so :attr:`declined` separates them. Every consumer that
    wants to know whether the chain actually ran can read one boolean per stage instead
    of parsing :attr:`detail`.
    """

    name: str
    n_dropped: int
    #: True when the stage was skipped or its result was refused, never when it ran and
    #: found nothing to remove. See :data:`MIN_CONDITIONED_POINTS`.
    declined: bool = False
    detail: str = ""


@dataclass(frozen=True)
class ConditioningResult:
    """The conditioned spectrum and a full account of what was taken off it.

    Mirrors the prototype's ``(f, Z, report)`` return — ``n_in``, ``n_out`` and a
    per-stage dropped count — as a frozen dataclass rather than a dict, so the field
    names are checked at the call site instead of at the moment somebody misspells a key.
    """

    #: Surviving frequencies, in the caller's own order.
    f: np.ndarray
    #: Surviving impedances, **short-corrected**. Not the caller's input values.
    Z: np.ndarray
    #: Boolean mask into the caller's original arrays: ``True`` where the point survived.
    #: Kept so a caller can carry its own parallel arrays (uncertainties, timestamps)
    #: through the chain without re-deriving which points went.
    mask: np.ndarray
    n_in: int
    n_out: int
    stages: tuple[StageReport, ...] = ()
    #: Series constants actually applied by step 2. ``0.0`` means *no correction was
    #: applied*, which the ``short_correction`` stage also reports as ``declined``.
    R_short_ohm: float = 0.0
    L_lead_H: float = 0.0
    #: The K–K ladder that judged step 5, or ``None`` when the stage never ran. Carried
    #: whole rather than reduced to a count because ``M``, ``mu`` and ``resid_pct`` are
    #: what say whether the trim was a measurement or an under-fitted ladder inventing
    #: the tail it then cut.
    kk: LinKKResult | None = None

    @property
    def n_dropped(self) -> int:
        return int(self.n_in - self.n_out)

    def stage(self, name: str) -> StageReport | None:
        """The report for one stage, or ``None`` if that stage is not in this chain."""
        for s in self.stages:
            if s.name == name:
                return s
        return None

    def dropped(self, name: str) -> int:
        """How many points *name* removed. ``0`` for a stage that did not run."""
        s = self.stage(name)
        return 0 if s is None else int(s.n_dropped)

    @property
    def declined(self) -> tuple[str, ...]:
        """Stages that declined to act. Empty is the ordinary case."""
        return tuple(s.name for s in self.stages if s.declined)

    def describe(self) -> str:
        """One line, for a log or a console.

        **ASCII deliberately**, where every ``detail`` string in this module is free to
        use ``Ω`` and ``μ`` as ``gates.py`` does. This is the string that gets logged, and
        this rig's console is cp1252: a ``→`` here raises ``UnicodeEncodeError`` inside
        the logging call and takes down the caller that was only trying to record what it
        had already successfully computed.
        """
        parts = [f"{s.name} -{s.n_dropped}" + (" (declined)" if s.declined else "")
                 for s in self.stages]
        return f"conditioned {self.n_in} -> {self.n_out} pts: " + ", ".join(parts)


# ── Step 1: the physical domain ──────────────────────────────────────────────

def physical_mask(Z: np.ndarray) -> np.ndarray:
    """Points a passive two-terminal network could actually have produced.

    ``Re Z > 0`` and finite. Everything else is not a measurement of an impedance,
    whatever else it may be a measurement of.

    **This runs first, and the ordering is load-bearing rather than tidy.** The K–K
    ladder at step 5 is a *global* weighted least-squares fit, so a single absurd point
    does not merely fail itself — it drags the whole basis. The prototype records the
    case: one point on the 10 MΩ rung at 165 Hz reading ``Re Z = −8.1×10⁶`` was enough to
    wreck the fit and take 41 good points down with it.

    Overlaps :func:`~softae.analysis.eis.gates.gate_quadrant` deliberately and is not
    replaced by it. That gate carries the reference-electrode attribution logic — is this
    the instrument, or a control loop left open by the board's coplanar geometry? — which
    is a *diagnosis* and belongs in the gate log. Here the same points are removed
    without any claim about cause, because the fit at step 5 cannot wait for the
    diagnosis, and a conditioner has no business making one.
    """
    Zc = np.asarray(Z, dtype=complex)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(Zc) & (Zc.real > 0.0)
    return np.asarray(ok, dtype=bool)


# ── Step 2: short / fixture series correction ────────────────────────────────

def short_correct(
    f: np.ndarray, Z: np.ndarray, *, R_short_ohm: float, L_lead_H: float
) -> np.ndarray:
    """``Z − (R_short + jωL_lead)`` — the fixture's series term, removed.

    Takes the two constants **already resolved from a**
    :class:`~softae.analysis.eis.calibration.CalibrationSet`, never a file. The prototype
    reads them out of ``calibration/eis/mux16.toml`` at each call, which is the one thing
    that must not be ported: :func:`~softae.analysis.eis.calibration.resolve_calibration`
    exists precisely because a raw TOML read applies a *stale* short blank — another
    board's constants, silently, with every resulting number looking plausible. Use
    :func:`short_constants` to get here from a calibration set, and let it drop the
    constants when the hardware hash has moved.

    A non-finite constant is treated as ``0.0`` by :func:`short_constants` rather than
    propagated: ``NaN`` here would not degrade the spectrum, it would erase it, and an
    absent calibration is a reason to skip the correction, not to destroy the data.
    """
    freq = np.asarray(f, dtype=float)
    Zc = np.asarray(Z, dtype=complex)
    return Zc - (float(R_short_ohm) + 1j * 2.0 * np.pi * freq * float(L_lead_H))


def short_constants(
    calibration: "CalibrationSet | None", channel: int | None
) -> tuple[float, float, str]:
    """``(R_short_ohm, L_lead_H, reason)`` for one channel, or zeros and why not.

    ``reason`` is empty when both constants came from the calibration set; otherwise it
    names what was missing. Returning the reason rather than logging it keeps the
    decision inspectable from :class:`ConditioningResult` — a warning in a log is not
    something a downstream consumer of ``R_short_ohm = 0.0`` can read.

    Delegates to :meth:`~softae.analysis.eis.calibration.CalibrationSet.for_channel`, so
    a channel inheriting a representative channel's constants still emits that method's
    warning. That warning is not bookkeeping: the measured channel-to-channel spread on
    this fixture is 2.4×, roughly an order of magnitude above the per-channel repeat
    error it would otherwise be mistaken for.
    """
    if calibration is None:
        return 0.0, 0.0, "no calibration set — series correction skipped"
    if channel is None:
        return 0.0, 0.0, "no channel given — series correction skipped"

    constants = calibration.for_channel(int(channel))
    R = float(constants.get("R_short_ohm", float("nan")))
    L = float(constants.get("L_lead_H", float("nan")))
    missing = [n for n, v in (("R_short_ohm", R), ("L_lead_H", L)) if v != v]
    if missing:
        return 0.0, 0.0, (f"channel {int(channel)} has no {' or '.join(missing)} in "
                          f"calibration '{calibration.fixture_id}' — skipped")
    return R, L, ""


# ── Steps 3 and 4: thin adapters onto the shipped gates ──────────────────────

def _gate_ctx(*, z_min_ohm: float | None = None, z_max_ohm: float | None = None,
              blocking: bool = True) -> dict[str, Any]:
    """The ``ctx`` mapping the gates read, built by production's own builder.

    Built through :func:`~softae.analysis.eis.policy.build_context` rather than as a
    literal, and that is not ceremony. Gates reach into ``ctx`` by ``(section, key)``
    through ``_ctx_get``, which **falls back to a default on a missing section** — the
    right behaviour for a gate and a terrible property for a hand-built dict, because a
    wrapper that wrote ``{"z_min_ohm": ...}`` at the top level instead of under
    ``"envelope"`` would run :func:`~softae.analysis.eis.gates.gate_magnitude` with an
    unbounded window and report that the window found nothing to remove. One builder,
    one shape, and the gate's defaults reached only on purpose.

    The envelope goes in as a plain mapping because ``_ctx_get`` accepts either a mapping
    or an object, and the window this chain applies is a caller's argument rather than
    the whole instrument envelope. ``None`` is left out entirely so the gate's own
    defaults (``0.0`` / ``inf``) apply — an unbounded window, which is what "no window
    was measured" honestly means.
    """
    envelope: dict[str, float] = {}
    if z_min_ohm is not None:
        envelope["z_min_ohm"] = float(z_min_ohm)
    if z_max_ohm is not None:
        envelope["z_max_ohm"] = float(z_max_ohm)
    return build_context(envelope=envelope, blocking=bool(blocking))


def magnitude_window_mask(
    f: np.ndarray, Z: np.ndarray, *,
    z_min_ohm: float | None = None, z_max_ohm: float | None = None,
) -> np.ndarray:
    """Step 3, as a mask. Delegates to :func:`gates.gate_magnitude`.

    Outside the instrument's reproducible range the accuracy specification does not
    apply, so a residual taken there is not a measurement — yet least squares weights it
    like one. The gate is pointwise, which is the correct granularity: a spectrum is
    routinely usable across most of its band while running over range at one end.
    """
    return np.asarray(
        gate_magnitude(f, Z, _gate_ctx(z_min_ohm=z_min_ohm, z_max_ohm=z_max_ohm)).mask,
        dtype=bool,
    )


def hf_inductive_mask(
    f: np.ndarray, Z: np.ndarray, *, blocking: bool = True
) -> np.ndarray:
    """Step 4, as a mask. Delegates to :func:`gates.gate_hf_inductive`.

    The step §2.1 measures at three orders of magnitude. A blocking cell with no faradaic
    process is capacitive at every frequency, so a contiguous ``Im Z > 0`` run at the top
    of the band is lead inductance or instrument phase error. Left in, it forces the fit
    to allocate an ``L`` of 400–500 µH against a short blank's measured 4.18 µH, and that
    absurdity lands exactly where ``R_series`` is determined.

    ``blocking=False`` passes the spectrum through untouched, because on a non-blocking
    cell the inductance may be real. The flag reaches the gate under ``ctx["cell"]``,
    where it is spelled — a top-level ``{"blocking": ...}`` would satisfy nothing and
    silently truncate a non-blocking spectrum.
    """
    return np.asarray(
        gate_hf_inductive(f, Z, _gate_ctx(blocking=blocking)).mask, dtype=bool
    )


# ── Step 5: linear-K–K low-frequency trim ────────────────────────────────────

def kk_trim_mask(
    f: np.ndarray, Z: np.ndarray, *,
    kk_pct: float = DEFAULT_KK_TRIM_PCT, blocking: bool = True,
) -> tuple[np.ndarray, LinKKResult]:
    """Step 5, as ``(mask, ladder)``. Drops the contiguous failing low-frequency run.

    Two pieces of the shipped module do the work, and neither is the prototype's:

    :func:`~softae.analysis.eis.kk.lin_kk`
        Fits the Voigt ladder and returns per-point ``resid_pct``. The prototype fixes
        the order at ``3·log₁₀(f_max/f_min)``; this walks M and selects on minimum median
        residual under a conditioning floor. An under-fitted ladder does not merely
        mis-measure here — it *manufactures* a low-frequency residual lobe that this
        function then cuts, which is the worst available failure for a repair step.
    :func:`~softae.analysis.eis.kk.low_frequency_run`
        The trim itself: the contiguous failing run at the low-frequency end, and nothing
        else. Identical in effect to the prototype's "walk in ascending frequency,
        dropping while bad, stop at the first good point", but it does not assume the
        sweep ascends — this rig sweeps high→low, so the low-frequency end is the *tail*
        of the array, and a helper that guessed wrong would truncate the high-frequency
        end and delete the arc carrying ``R_bulk`` while reporting that it removed drift.

    Only the low-frequency run is removed, and the asymmetry is the policy rather than an
    optimisation: acquisition down there is slow enough for the sample to drift during the
    sweep, and ``R_bulk`` does not live there. An isolated mid-band failure is a noisy
    point, not drift, and the criterion licensing removal does not apply to it — so it
    stays, and stays visible in :attr:`ConditioningResult.kk`.

    ``resid_pct`` is ``hypot(res_real, res_imag)`` in percent, where the prototype
    thresholds each component separately. One per-point magnitude is what the shipped
    module chose so a single threshold governs both, and it is the stricter reading of
    the same ``kk_pct``.
    """
    result = lin_kk(f, Z, blocking=bool(blocking))
    n = int(min(np.asarray(f).size, np.asarray(Z).size))
    if not result.ok or result.resid_pct.size != n:
        return np.ones(n, dtype=bool), result
    with np.errstate(invalid="ignore"):
        failing = np.asarray(result.resid_pct, dtype=float) > float(kk_pct)
    return ~low_frequency_run(f, failing), result


# ── The chain ────────────────────────────────────────────────────────────────

def _apply_stage(
    keep: np.ndarray, sub_mask: np.ndarray, name: str, detail: str
) -> tuple[np.ndarray, StageReport]:
    """Scatter a survivors-space mask back into caller space, honouring the floor.

    Every stage runs on ``f[keep], Z[keep]`` rather than on the full arrays, which is not
    an efficiency: :func:`~softae.analysis.eis.gates.gate_hf_inductive` walks down from
    the highest frequency and **stops at the first capacitive point**, so a point an
    earlier stage already removed would either halt that walk early or extend it through
    a gap. Running each stage on the survivors is what makes the chain order-dependent in
    the way it is supposed to be, and it is also what stops a point being counted as
    dropped twice.
    """
    idx = np.flatnonzero(keep)
    candidate = keep.copy()
    candidate[idx[~np.asarray(sub_mask, dtype=bool)]] = False

    n_drop = int(keep.sum() - candidate.sum())
    if n_drop and int(candidate.sum()) < MIN_CONDITIONED_POINTS:
        logger.info(
            "eis_conditioning_stage_declined", stage=name, would_drop=n_drop,
            would_leave=int(candidate.sum()), floor=MIN_CONDITIONED_POINTS,
            msg="stage would take the spectrum below the conditioning floor — "
                "declined; the gates refuse this spectrum, a conditioner does not",
        )
        return keep, StageReport(
            name, 0, True,
            f"would have dropped {n_drop} pts leaving {int(candidate.sum())}, below "
            f"the {MIN_CONDITIONED_POINTS}-pt floor — declined, nothing removed",
        )
    return candidate, StageReport(name, n_drop, False, detail)


def _resolve_window(
    calibration: "CalibrationSet | None",
    z_min_ohm: float | None,
    z_max_ohm: float | None,
) -> tuple[float | None, float | None]:
    """Explicit argument wins; else the calibration's measured window; else unbounded.

    ``None`` out means *no bound on that side*, which is the truthful reading of an
    uncommissioned rig — :meth:`CalibrationSet.capabilities` spells the same state as
    "run the reference resistors". Substituting a configured estimate here would put a
    guess and a measurement behind one argument name.
    """
    def _pick(explicit: float | None, measured: float) -> float | None:
        if explicit is not None:
            return float(explicit)
        return float(measured) if measured == measured else None

    if calibration is None:
        return (None if z_min_ohm is None else float(z_min_ohm),
                None if z_max_ohm is None else float(z_max_ohm))
    return (_pick(z_min_ohm, calibration.z_min_ohm),
            _pick(z_max_ohm, calibration.z_max_ohm))


def condition(
    f: np.ndarray,
    Z: np.ndarray,
    *,
    calibration: "CalibrationSet | None" = None,
    channel: int | None = None,
    z_min_ohm: float | None = None,
    z_max_ohm: float | None = None,
    kk_pct: float = DEFAULT_KK_TRIM_PCT,
    blocking: bool = True,
) -> ConditioningResult:
    """Run the whole chain, in order, and report every point it removed.

    Steps 1–5 as tabled in the module docstring. Each runs on what the previous one left,
    so the counts partition the input exactly: ``n_in − n_out`` equals the sum of the
    per-stage drops, always, whatever any stage did.

    ``calibration`` supplies the step-2 constants and, unless overridden, the step-3
    window. Absent, both are skipped rather than guessed and both say so in the report —
    the chain still runs, because steps 1, 4 and 5 need no commissioning and step 4 is
    the one carrying the effect.

    Refuses nothing, raises on nothing, and returns an empty spectrum only if it was
    handed one. A caller wanting a verdict runs
    :func:`~softae.analysis.eis.gates.run_gates` on the result.
    """
    freq = np.asarray(f, dtype=float)
    raw = np.asarray(Z, dtype=complex)
    n_in = int(min(freq.size, raw.size))
    freq, raw = freq[:n_in], raw[:n_in]

    keep = np.ones(n_in, dtype=bool)
    stages: list[StageReport] = []

    # 1 — physical domain, on the raw spectrum and before anything else.
    keep, report = _apply_stage(
        keep, physical_mask(raw), STAGE_PHYSICAL,
        "Re Z ≤ 0 or non-finite — not a passive two-terminal measurement")
    stages.append(report)

    # 2 — fixture series correction. Removes no points; changes every one.
    #
    # Applied to the whole array, not the survivors, so `Z_corr` stays index-aligned with
    # the caller's input and `mask` remains a mask into it.
    R_short, L_lead, why_not = short_constants(calibration, channel)
    Z_corr = short_correct(freq, raw, R_short_ohm=R_short, L_lead_H=L_lead)
    stages.append(StageReport(
        STAGE_SHORT, 0, bool(why_not),
        why_not or f"Z − ({R_short:.4g} Ω + jω·{L_lead:.4g} H)"))
    # Deliberately NOT re-running step 1 on `Z_corr`. Subtracting `R_short` can carry a
    # small-|Z| point across `Re Z = 0`, and it is tempting to filter again — but that
    # would be the conditioner diagnosing its own correction, and the honest reading of
    # such a point is "the correction is wrong for this channel", which is
    # `gate_quadrant`'s question and needs the RE-state context this module does not have.

    # 3 — magnitude window.
    z_lo, z_hi = _resolve_window(calibration, z_min_ohm, z_max_ohm)
    if z_lo is None and z_hi is None:
        stages.append(StageReport(
            STAGE_WINDOW, 0, True,
            "no |Z| window measured or given — window not applied"))
    else:
        keep, report = _apply_stage(
            keep,
            magnitude_window_mask(freq[keep], Z_corr[keep],
                                  z_min_ohm=z_lo, z_max_ohm=z_hi),
            STAGE_WINDOW,
            f"outside [{z_lo if z_lo is not None else 0.0:.3g}, "
            f"{z_hi if z_hi is not None else float('inf'):.3g}] Ω")
        stages.append(report)

    # 4 — HF-inductive truncation. The step §2.1 measures at three orders of magnitude.
    keep, report = _apply_stage(
        keep, hf_inductive_mask(freq[keep], Z_corr[keep], blocking=blocking),
        STAGE_HF_INDUCTIVE,
        "contiguous Im Z > 0 run at the top of the band" if blocking
        else "skipped — non-blocking cell, inductance may be real")
    stages.append(report)

    # 5 — linear-K–K low-frequency trim.
    kk: LinKKResult | None = None
    if int(keep.sum()) < MIN_KK_POINTS:
        stages.append(StageReport(
            STAGE_KK, 0, True,
            f"{int(keep.sum())} surviving pts, below the {MIN_KK_POINTS}-pt ladder "
            f"minimum — K–K trim not attempted"))
    else:
        sub_mask, kk = kk_trim_mask(freq[keep], Z_corr[keep],
                                    kk_pct=kk_pct, blocking=blocking)
        if not kk.ok:
            stages.append(StageReport(
                STAGE_KK, 0, True, f"K–K ladder unavailable: {kk.error}"))
        else:
            keep, report = _apply_stage(
                keep, sub_mask, STAGE_KK,
                f"low-frequency run above {kk_pct:.3g}% K–K residual "
                f"(ladder M={kk.M}, μ={kk.mu:.3f})")
            stages.append(report)

    result = ConditioningResult(
        f=freq[keep], Z=Z_corr[keep], mask=keep,
        n_in=n_in, n_out=int(keep.sum()), stages=tuple(stages),
        R_short_ohm=float(R_short), L_lead_H=float(L_lead), kk=kk,
    )
    logger.info("eis_conditioned", n_in=result.n_in, n_out=result.n_out,
                channel=channel, declined=result.declined,
                detail=result.describe())
    return result
