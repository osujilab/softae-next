"""The pre-registered decision rule of spec 6, and nothing else.

Thresholds, the statistics they are applied to, and the outcome routing. This
module reads no database and prints nothing: it takes the
:class:`~softae.tools.eis_validate_records.Cell` objects the reading half
assembled and returns a :class:`Verdict`. That separation is the point -- a
validation whose success criterion is chosen after the data arrives proves
nothing, so the arithmetic that will judge the rig sits in one file, is pinned by
its own tests against hand-built records, and cannot be reached through a
rendering change.

Four rules govern everything below:

1. **Offset and scatter are never combined into an RMS.** The failure mode under
   test is *biased, not scattered*. A median near zero with wide scatter means
   something categorically different from a small consistent offset, and an RMS
   destroys exactly that distinction. :class:`Spread` therefore carries median,
   MAD, IQR, min, max and the sign split, side by side, for every population.
2. **A mock run never emits a GO.** A simulated verdict is not a verdict. The
   arithmetic that would produce one is still exercised in full -- that is what
   the grid-aware backend in :mod:`softae.tools.eis_validate_mock` is for -- but
   :func:`evaluate` routes the outcome to WITHHELD.
3. **A criterion that cannot be evaluated says so, and does not move the
   outcome.** What a criterion *displays* and what it *contributes to the
   routing* are two different things, and this file keeps them apart on purpose::

       <name>_ok   a bool, pre-registered, the ONLY thing the routing reads
       status      PASS / FAIL / INSUFFICIENT, what a human reads

   The routing booleans below are exactly what they were when the rule was
   registered -- an absent T1 ratio still satisfies ``t1_ok``, an absent D1
   median still fails ``d1_ok`` -- because changing which runs are declared GO is
   a scientific decision, not a reporting one. The *status* is separate, computed
   by :func:`_status`, and reports INSUFFICIENT wherever the input was simply not
   there. Rounding "no evidence" to PASS is what let the first hardware run print
   ``[PASS] T1 ... observed n/a`` off two zero sums; rounding it to FAIL, as D1,
   D2 and D4 did, is the same error pointed the other way. Neither is a verdict,
   so neither is printed as one.
4. **The hold certification marks rows and selects none of them.** D1-D4 pick
   their populations by *band*, never by
   :attr:`~softae.tools.eis_validate_records.Cell.stillness_certified`, and a
   cell the settle gate could not speak for contributes its numbers exactly as
   it always did. That is the operator's decision and it has a reason: those
   numbers are the metrology the production limits will later be calibrated
   from, and a row dropped here is a row that has to be re-earned on the rig.
   The certification therefore reaches this file **only as a note**
   (:func:`_retained_uncertified`), appended to D1-D4 when the count is non-zero
   so that "median over 11 cells, 3 of them uncertified" cannot be read as
   "median over 11 certified cells". A criterion whose note grew is still the
   same criterion: no ``*_ok`` boolean, no population and no threshold reads the
   stamp. H1 already withholds the verdict on any certification but ``settled``,
   which is what makes keeping the rows safe.

Rendered by :mod:`softae.tools.eis_validate_report`, which is also the single
import surface these names are re-exported through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from softae.tools.eis_validate_records import CONTROL, EXCLUDED, TREATMENT, UNRESOLVED, Cell

#: D1 -- the regime being escaped costs 0.204 decades (a x1.598 median R1
#: overestimate is a /1.598 sigma underestimate). Recovering at least half of it
#: is the minimum that justifies paying for a second sweep.
D1_MIN_MEDIAN_IMPROVEMENT_DEC = 0.10
#: D2 -- a median driven by two cells is not an offset.
D2_MIN_POSITIVE_FRACTION = 2.0 / 3.0
#: D3 -- the noise floor, on cells where the scout says nothing needs fixing.
#: Also ~37x the instrument's own ``magnitude_accuracy_pct`` of 0.32, so it is
#: achievable rather than aspirational.
D3_MAX_CONTROL_DEVIATION_DEC = 0.05
#: H3 -- the hold has to hold to the same precision the noise floor claims.
H3_MAX_HOLD_DRIFT_DEC = 0.05
#: T1 -- arm B is at most two sweeps. From ``Quick`` one rung is ``Standard``,
#: so the ratio is ``1 + f * (37.19 / 17.50) = 1 + 2.125 f``; 1.5 accommodates a
#: quarter of cells extending.
T1_MAX_TIME_RATIO = 1.5

PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT"

OUTCOME_GO = "GO"
OUTCOME_CONDITIONAL_GO = "CONDITIONAL GO"
OUTCOME_MECHANISM_LIMITED = "MECHANISM-LIMITED"
OUTCOME_NO_GO = "NO-GO"
OUTCOME_INSUFFICIENT = "INSUFFICIENT"
OUTCOME_WITHHELD = "WITHHELD"


# ── Statistics: offset and scatter, never combined ───────────────────────────

@dataclass
class Spread:
    """Offset **and** scatter, side by side. There is deliberately no RMS here."""

    n: int = 0
    median: float | None = None
    mad: float | None = None
    q1: float | None = None
    q3: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    n_positive: int = 0
    n_negative: int = 0
    n_zero: int = 0

    @property
    def iqr(self) -> float | None:
        if self.q1 is None or self.q3 is None:
            return None
        return self.q3 - self.q1

    @property
    def positive_fraction(self) -> float | None:
        return None if self.n == 0 else self.n_positive / self.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "median": self.median, "mad": self.mad,
            "q1": self.q1, "q3": self.q3, "iqr": self.iqr,
            "min": self.minimum, "max": self.maximum,
            "n_positive": self.n_positive, "n_negative": self.n_negative,
            "n_zero": self.n_zero, "positive_fraction": self.positive_fraction,
        }


def describe(values: Sequence[float | None]) -> Spread:
    """Median, MAD, IQR, extremes and the sign split. Nothing is collapsed."""
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    if not finite:
        return Spread()
    med = _median(finite)
    return Spread(
        n=len(finite),
        median=med,
        mad=_median([abs(v - med) for v in finite]),
        q1=_quantile(finite, 0.25),
        q3=_quantile(finite, 0.75),
        minimum=finite[0],
        maximum=finite[-1],
        n_positive=sum(1 for v in finite if v > 0),
        n_negative=sum(1 for v in finite if v < 0),
        n_zero=sum(1 for v in finite if v == 0),
    )


# ── The pre-registered rule ──────────────────────────────────────────────────

@dataclass
class Criterion:
    name: str
    threshold: str
    observed: str
    status: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.name, "threshold": self.threshold,
            "observed": self.observed, "status": self.status, "note": self.note,
        }


def _status(ok: bool, evaluable: bool) -> str:
    """The status a criterion *displays* -- never what the routing reads.

    ``ok`` is the pre-registered boolean, passed through untouched; ``evaluable``
    says whether there was anything to apply it to. When there was not, the
    criterion reports INSUFFICIENT rather than borrowing whichever verdict its
    own ``or None`` / ``and not None`` phrasing happened to fall to. The caller
    keeps using ``ok`` for the outcome, so this cannot move a verdict.
    """
    if not evaluable:
        return INSUFFICIENT
    return PASS if ok else FAIL


def _retained_uncertified(contributors: Sequence[Cell]) -> str:
    """How many cells *inside one statistic* were retained uncertified, or ``""``.

    A note, never a filter. *contributors* is the exact set of cells whose
    numbers entered the median being reported, so the sentence describes the
    statistic that is printed beside it rather than the run as a whole -- a
    median over eleven cells of which three were uncertified is a different
    claim from one over eleven certified cells, and only the count makes the two
    distinguishable.

    Empty when every contributing cell was certified, which is what keeps a
    fully certified run's report exactly what it was.
    """
    uncertified = [c for c in contributors if not c.stillness_certified]
    if not uncertified:
        return ""
    words = ", ".join(sorted({c.certification for c in uncertified}))
    return (
        f"RETAINED UNCERTIFIED: {len(uncertified)} of {len(contributors)} "
        f"cell(s) in this statistic were not certified still ({words}). Their "
        "numbers are counted here ON PURPOSE and this is not a failure -- "
        "metrology on more channels is what production limits get calibrated "
        "from, and a row dropped here would have to be re-earned on the rig. "
        "H1 withholds the verdict regardless, so nothing is licensed by them."
    )


def _note(*parts: str) -> str:
    """Join a criterion's standing note to a conditional one, as sentences.

    Every part but the last is terminated so the two cannot run together in the
    wrapped block. The last is passed through **byte for byte**, so a criterion
    with nothing to add renders exactly the characters it always did -- which is
    what keeps a fully certified run's report unchanged.
    """
    kept = [part.strip() for part in parts if part and part.strip()]
    if not kept:
        return ""
    lead = [p if p.endswith(".") else f"{p}." for p in kept[:-1]]
    return " ".join(lead + kept[-1:])


@dataclass
class Verdict:
    outcome: str
    criteria: list[Criterion] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def evaluate_vetoes(cells: Sequence[Cell]) -> list[str]:
    """V1-V3. Any one, anywhere, forces NO-GO regardless of the primary.

    Fit quality is a **veto, never evidence**: a bad model fits a bad spectrum
    well, so these cannot establish improvement -- but they can establish
    regression, and a regression blocks a GO whatever the deviation says.
    """
    vetoes: list[str] = []
    for cell in cells:
        scout, follow_up = cell.scout, cell.follow_up
        if scout is None or follow_up is None:
            continue
        if scout.gate_verdict == "accept" and follow_up.gate_verdict == "reject":
            vetoes.append(
                f"V1 ch{cell.channel}: scout accepted, follow-up rejected"
            )
        if follow_up.segmented:
            regressed = sorted(scout.passed_gates() & follow_up.failed_gates())
            if regressed:
                vetoes.append(
                    f"V2 ch{cell.channel}: segmented follow-up failed "
                    f"{', '.join(regressed)}, which the scout sweep passed"
                )
        f_lo_scout, f_lo_follow = scout.f_lo_hz, follow_up.f_lo_hz
        if f_lo_scout > 0 and f_lo_follow > 0 and f_lo_follow > f_lo_scout:
            vetoes.append(
                f"V3 ch{cell.channel}: follow-up floor {f_lo_follow:g} Hz is "
                f"ABOVE the sweep it followed ({f_lo_scout:g} Hz) -- narrower"
            )
    return vetoes


def evaluate(
    cells: Sequence[Cell],
    *,
    min_treatment: int = 6,
    mock: bool = False,
) -> Verdict:
    """Every criterion of the spec's decision rule, mechanically, no editorialising."""
    # H2: a row taken inside a warn-grade excursion window is excluded from
    # every accuracy table and the exclusion is COUNTED, never silent. The cell
    # still appears in the population sizes -- it was measured, and hiding it
    # would understate how much of the strip the excursion cost.
    usable = [c for c in cells if not c.excursion]
    n_excluded = len(cells) - len(usable)
    by_pop = {
        pop: [c for c in usable if c.population == pop]
        for pop in (CONTROL, TREATMENT, UNRESOLVED, EXCLUDED)
    }
    treatment, control = by_pop[TREATMENT], by_pop[CONTROL]

    improvement = describe([c.improvement() for c in treatment])
    d_scout_treat = describe([c.delta_scout() for c in treatment])
    d_scout_control_abs = describe(
        [abs(v) for v in (c.delta_scout() for c in control) if v is not None]
    )
    hold = describe(
        [abs(v) for v in (c.delta_hold() for c in usable) if v is not None]
    )

    # Exactly the cells whose numbers entered each median above -- not the
    # populations, which are larger wherever a cell yielded no deviation. Used
    # for the RETAINED UNCERTIFIED note and for nothing else; no threshold, no
    # population and no `*_ok` boolean below reads these lists.
    improving = [c for c in treatment if c.improvement() is not None]
    deviating = [c for c in treatment if c.delta_scout() is not None]
    control_deviating = [c for c in control if c.delta_scout() is not None]

    criteria: list[Criterion] = []

    # H1 first: if the hold did not hold, nothing below means anything.
    certifications = {
        row.hold_certified
        for cell in cells
        for row in (cell.reference, cell.scout, cell.follow_up)
        if row is not None and row.hold_certified
    }
    del by_pop[EXCLUDED]  # not a population: cells adaptive declines to act on
    h1_ok = bool(certifications) and certifications <= {"settled"}
    criteria.append(Criterion(
        "H1 settle certified before the first reference sweep",
        "settled", ", ".join(sorted(certifications)) or "no rows carry one",
        _status(h1_ok, bool(certifications)),
        "" if h1_ok else "ceiling / not_evaluable / disabled withholds the outcome",
    ))

    criteria.append(Criterion(
        "H2 no fault-grade excursion", "0 fault-grade",
        f"{n_excluded} cell(s) EXCLUDED from the accuracy tables "
        f"(warn-grade window), {len(usable)} usable" if cells
        else "no cells recorded",
        _status(bool(usable), bool(cells)),
        "a fault-grade excursion parks the run, so its absence is implied by "
        "there being rows at all; warn-grade cells are excluded above and "
        "counted here, which is why the count and not just the grade is printed",
    ))

    h3_ok = hold.median is None or hold.median <= H3_MAX_HOLD_DRIFT_DEC
    h3_status = _status(h3_ok, bool(hold.n))
    criteria.append(Criterion(
        "H3 median |Delta_hold| on the drift-check subset",
        f"<= {H3_MAX_HOLD_DRIFT_DEC:.2f} dec",
        _fmt(hold.median, "dec") + f" (n={hold.n})" if hold.n
        else "no drift-check rows (n=0)",
        h3_status,
        "" if hold.n else
        "no reference_end rows: --drift-check 0 makes H3 UNEVALUABLE. The "
        "spec routes H3 *failure* to INSUFFICIENT and does not say where "
        "unevaluable goes; it is routed there too, deliberately and "
        "conservatively -- an unverified hold cannot license a GO, and "
        "'undeclared is unknown, never empty' is the house rule for exactly "
        "this shape. Run --drift-check >= 1 to reach a verdict.",
    ))

    d3_ok = (
        d_scout_control_abs.median is not None
        and d_scout_control_abs.median <= D3_MAX_CONTROL_DEVIATION_DEC
    )
    criteria.append(Criterion(
        "D3 median |Delta_scout| on CONTROL (the noise floor)",
        f"<= {D3_MAX_CONTROL_DEVIATION_DEC:.2f} dec",
        _fmt(d_scout_control_abs.median, "dec") + f" (n={d_scout_control_abs.n})"
        if d_scout_control_abs.n else "no CONTROL cells (n=0)",
        _status(d3_ok, bool(d_scout_control_abs.n)),
        _note("no improvement below this floor can be believed",
              _retained_uncertified(control_deviating)),
    ))

    d1_ok = (
        improvement.median is not None
        and improvement.median >= D1_MIN_MEDIAN_IMPROVEMENT_DEC
    )
    criteria.append(Criterion(
        "D1 median improvement on TREATMENT",
        f">= +{D1_MIN_MEDIAN_IMPROVEMENT_DEC:.2f} dec",
        _fmt(improvement.median, "dec") + f" (n={improvement.n})"
        if improvement.n else "no TREATMENT cell yielded an improvement (n=0)",
        _status(d1_ok, bool(improvement.n)),
        _retained_uncertified(improving),
    ))

    frac = improvement.positive_fraction
    d2_ok = frac is not None and frac >= D2_MIN_POSITIVE_FRACTION
    criteria.append(Criterion(
        "D2 fraction of TREATMENT cells with improvement > 0",
        f">= {D2_MIN_POSITIVE_FRACTION:.3f}",
        f"{frac:.3f} ({improvement.n_positive}/{improvement.n})"
        if frac is not None else "no TREATMENT cell yielded an improvement (0/0)",
        _status(d2_ok, frac is not None),
        _note("separates a real shift from a tail",
              _retained_uncertified(improving)),
    ))

    d4_negative = d_scout_treat.median is not None and d_scout_treat.median < 0
    criteria.append(Criterion(
        "D4 median Delta_scout on TREATMENT is negative",
        "< 0 (direction only)",
        _fmt(d_scout_treat.median, "dec") if d_scout_treat.n
        else "no TREATMENT deviation (n=0)",
        _status(d4_negative, bool(d_scout_treat.n)),
        _note(
            "PRE-REGISTERED SIGN: R1 extrapolated past the apex is OVERestimated "
            "(x1.598), and sigma = K/R, so the plain preset must read sigma LOW "
            "against the reference (-0.204 dec). Reported as a FLAG, not a hard "
            "fail -- but a GO with this inverted must be argued in prose.",
            _retained_uncertified(deviating)),
    ))

    ratio_cells = [c for c in usable if c.population in (CONTROL, TREATMENT)]
    sum_adaptive = sum(c.t_adaptive() for c in ratio_cells)
    sum_control = sum(c.t_control() for c in ratio_cells)
    ratio = (sum_adaptive / sum_control) if sum_control > 0 else None
    # `t1_ok` keeps its pre-registered phrasing -- an absent ratio does not veto
    # a GO -- while the status below refuses to call that absence a pass.
    t1_ok = ratio is None or ratio <= T1_MAX_TIME_RATIO
    criteria.append(Criterion(
        "T1 sum(t_adaptive) / sum(t_control)",
        f"<= {T1_MAX_TIME_RATIO:.1f}x",
        f"{ratio:.3f}x" if ratio is not None
        else ("no CONTROL or TREATMENT cells" if not ratio_cells
              else "sum_control = 0 s"),
        _status(t1_ok, ratio is not None),
        "counts the scout sweep on EVERY cell, including accepted ones",
    ))

    vetoes = evaluate_vetoes(cells)
    reasons: list[str] = []

    # -- outcome routing ------------------------------------------------------
    if mock:
        outcome = OUTCOME_WITHHELD
        reasons.append(
            "MOCK RUN. Every number above was computed by the real rule on "
            "synthetic spectra. A simulated verdict is not a verdict."
        )
    elif not h1_ok:
        outcome = (
            OUTCOME_WITHHELD if "disabled" in certifications
            else OUTCOME_INSUFFICIENT
        )
        reasons.append(
            "The hold was not certified `settled`, so the comparison is not "
            "interpretable: " + (", ".join(sorted(certifications)) or "no rows")
        )
    elif vetoes:
        outcome = OUTCOME_NO_GO
        reasons.append("A fit-quality regression vetoes the primary criteria.")
    elif len(treatment) < min_treatment:
        outcome = OUTCOME_INSUFFICIENT
        reasons.append(
            f"TREATMENT n={len(treatment)} < --min-treatment {min_treatment}. "
            "Change the setpoint or widen the reference; do not draw a "
            "conclusion from this many cells."
        )
    elif not d3_ok:
        outcome = OUTCOME_INSUFFICIENT
        reasons.append(
            "D3 failed: the reference and the plain preset disagree by more "
            "than the noise floor on cells the scout says are fine, so the "
            "reference is not resolving anything."
        )
    elif h3_status != PASS:
        outcome = OUTCOME_INSUFFICIENT
        reasons.append(
            "H3 failed or was unevaluable: a moving sample means the paired "
            "differences are not what they claim to be."
        )
    elif not d1_ok:
        required, delivered = _rescue_medians(treatment)
        if required is not None and delivered is not None and delivered < required:
            outcome = OUTCOME_MECHANISM_LIMITED
            reasons.append(
                f"D1 failed AND the follow-up delivered {delivered:.3f} dec of "
                f"reach against {required:.3f} required (median). That is a "
                "defect in the one-rung ladder, not in the scout -- send it to "
                "iterated widening, not to abandoning adaptive."
            )
        else:
            outcome = OUTCOME_NO_GO
            reasons.append(
                "D1 failed with adequate reach delivered: the mechanism "
                "reached far enough and the number did not move."
            )
    elif not d2_ok:
        outcome = OUTCOME_NO_GO
        reasons.append(
            "D2 failed while D1 passed -- a median not supported by the sign "
            "split. NOTE: the spec's outcome table does not route this case "
            "explicitly; it is treated as NO-GO because a GO requires D1-D3."
        )
    elif not t1_ok or not d4_negative:
        outcome = OUTCOME_CONDITIONAL_GO
        if not t1_ok:
            reasons.append("D1-D3 pass but T1 fails: gate `actuate` per modality "
                           "pending the cost question.")
        if not d4_negative:
            reasons.append("D4 is INVERTED: the offset runs the other way from "
                           "the pre-registered direction. The phenomenon being "
                           "corrected is not the one that was measured, and "
                           "that needs explaining before anything is acted on.")
    else:
        outcome = OUTCOME_GO
        reasons.append(
            "Flip [eis.scout] enabled = true, actuate = true for HT and AE. "
            "Manual stays optional with actuate_manual = false -- that tab's "
            "checkbox remains authoritative."
        )

    return Verdict(outcome=outcome, criteria=criteria, vetoes=vetoes, reasons=reasons)


def _rescue_medians(
    treatment: Sequence[Cell],
) -> tuple[float | None, float | None]:
    depths = [d for d in (c.rescue_depth() for c in treatment) if d is not None]
    if not depths:
        return None, None
    return (
        _median(sorted(d[0] for d in depths)),
        _median(sorted(d[1] for d in depths)),
    )


# ── Small helpers ────────────────────────────────────────────────────────────

def _median(ordered: Sequence[float]) -> float:
    values = sorted(ordered)
    n = len(values)
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def _quantile(ordered: Sequence[float], q: float) -> float:
    values = sorted(ordered)
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _fmt(value: float | None, unit: str = "") -> str:
    """One column width for every number a criterion or a report section quotes.

    Lives here rather than with the renderer because a ``Criterion.observed``
    string is part of the rule's own output -- it is what the JSON payload
    carries -- and both halves must format a decade identically or the printed
    report and the machine-readable one would disagree in the last digit.
    """
    if value is None or not math.isfinite(value):
        return "n/a".rjust(9)
    return f"{value:+9.4f}" + (f" {unit}" if unit else "")


__all__ = [
    "D1_MIN_MEDIAN_IMPROVEMENT_DEC", "D2_MIN_POSITIVE_FRACTION",
    "D3_MAX_CONTROL_DEVIATION_DEC", "FAIL", "H3_MAX_HOLD_DRIFT_DEC",
    "INSUFFICIENT", "OUTCOME_CONDITIONAL_GO", "OUTCOME_GO",
    "OUTCOME_INSUFFICIENT", "OUTCOME_MECHANISM_LIMITED", "OUTCOME_NO_GO",
    "OUTCOME_WITHHELD", "PASS", "T1_MAX_TIME_RATIO",
    "Criterion", "Spread", "Verdict", "describe", "evaluate", "evaluate_vetoes",
]
