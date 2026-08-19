"""Read back an EIS adaptive-acquisition validation and evaluate the rule.

Read-only, and regenerable at any moment -- including while the runner is still
measuring. That is the point of the runner/reporter split it inherits from
:mod:`softae.tools.shadow_rehearse` / :mod:`~softae.tools.shadow_rehearse_report`:
the runner persists after every single sweep and holds nothing in memory, so a
report is a pure function of what is on disk and a crash costs one spectrum.

**No :class:`~softae.core.data_store.DataStore` is ever constructed here.** The
connection is opened ``mode=ro``, so SQLite itself refuses every write and there
is no code path to audit.

Three things this module will not do, each because the corresponding mistake is
the one that would matter:

1. **It never quotes a deviation against an open-arc reference as accuracy.**
   ``Extended`` reaches 1.351 Hz, so its own arc closes only for an apex above
   about 13.51 Hz; below that its ``R1`` came from extrapolating the
   high-frequency limb, measured on this rig at a **+60.9 % median** overestimate
   (+175.2 % with the full CPE fitter, p16 = 0.031). A "deviation" there is a
   deviation between two extrapolations. Those cells are UNRESOLVED: counted,
   listed, and kept out of every accuracy table.

2. **It never combines offset and scatter into an RMS.** The failure mode under
   test is *biased, not scattered*. A median near zero with wide scatter means
   something categorically different from a small consistent offset, and an RMS
   destroys exactly that distinction. Median, MAD, IQR, min, max and the sign
   split are reported separately, for every population.

3. **It never emits a GO from a mock run.** A simulated verdict is not a
   verdict. The arithmetic that would produce one is still exercised in full --
   that is what the grid-aware backend in :mod:`softae.tools.eis_validate_mock`
   is for -- but the outcome line says WITHHELD.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: JSON payload schema, mirroring ``"softae.eis_timing/1"``.
REPORT_SCHEMA = "softae.eis_validate/1"

ARM_REFERENCE = "reference"
ARM_SCOUT = "adaptive_scout"
ARM_FOLLOW_UP = "adaptive_follow_up"
ARM_REFERENCE_END = "reference_end"

CONTROL = "CONTROL"
TREATMENT = "TREATMENT"
UNRESOLVED = "UNRESOLVED"
EXCLUDED = "EXCLUDED"

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


# ── Reading ──────────────────────────────────────────────────────────────────

def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """The one place a connection is made, and SQLite itself refuses the write.

    ``mode=ro`` makes every ``INSERT`` on this handle raise ``OperationalError``
    and leaves the WAL file untouched -- no code path to audit. Lifted verbatim
    from :func:`softae.tools.shadow_rehearse._connect_ro` for exactly that
    property.
    """
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


#: A **textual** prefilter, then an exact match in Python -- deliberately not
#: ``json_extract(eis_params_json, '$.eis_validation_name') = ?``.
#:
#: ``DataStore`` serialises ``eis_params`` with ``json.dumps``, which emits the
#: bare token ``NaN`` for a non-finite float. That is valid to Python's loader
#: and **invalid JSON**, so SQLite's JSON1 rejects the whole document -- and
#: because the predicate is evaluated per row, *one* such row anywhere in
#: ``measurements`` makes the entire query raise ``malformed JSON`` and this
#: validation unreadable. This harness never writes one (see
#: ``eis_validate._finite_or_none``), but it does not own the table.
_SELECT = """
SELECT m.measurement_id, m.run_id, m.channel, m.timestamp, m.measurement_time_s,
       m.eis_params_json,
       f.sigma_S_per_cm, f.sigma_is_bound, f.R1, f.gate_verdict, f.gate_log_json,
       f.arc_state
  FROM measurements m
  LEFT JOIN fit_results f ON f.measurement_id = m.measurement_id
 WHERE m.eis_params_json LIKE '%"eis_validation_name"%'
 ORDER BY m.measurement_id
"""


@dataclass
class SweepRecord:
    """One persisted sweep, with everything the rule needs and nothing else."""

    measurement_id: int
    run_id: str
    channel: int
    timestamp: str
    seconds: float
    params: dict[str, Any]
    sigma: float | None
    sigma_is_bound: bool
    r1_ohm: float | None
    gate_verdict: str | None
    gate_log: list[dict[str, Any]]
    fit_arc_state: str | None

    @property
    def arm(self) -> str:
        return str(self.params.get("eis_validation_arm", ""))

    @property
    def cell(self) -> str:
        return str(self.params.get("eis_validation_cell", ""))

    @property
    def scout_verdict(self) -> str:
        return str(self.params.get("eis_scout_verdict", ""))

    @property
    def arc_closed(self) -> bool:
        """Closure **of the spectrum this row is**, stamped at acquisition.

        Read from the harness's own key rather than ``fit_results.arc_state``:
        the fit column is annotated on whatever survived the admission gates,
        which is not always the sweep that was taken, and the population
        partition is a statement about the *sweep*.
        """
        return str(self.params.get("eis_validation_arc_state", "")) == "closed"

    @property
    def reference_valid(self) -> bool:
        """Is a reference taken on **this** grid actually a reference?

        Not the same question as :attr:`arc_closed`, and conflating the two is a
        mistake that quietly promotes cells into the accuracy tables that the
        whole design exists to keep out. ``arc_closure.state == "closed"`` means
        only that the apex fell *inside* the swept window -- an apex one point
        above the floor closes. Being a reference means the sweep reached a
        **full ``band_below_apex_min_decades``** past the apex, which is the cut
        that separates "measured" from "extrapolated", and it is the cut the
        spec's own definition of the resolving window is computed from:
        ``apex >= f_lo * 10 ** band_min``.

        Seen on the very first mock run: an apex at 5.34 Hz on the reference's
        1.351 Hz floor is 0.60 decades of band -- ``state == "closed"`` and
        **not** a reference. Judged on closure alone that cell would have been
        called TREATMENT and its 1.3-decade deviation quoted as accuracy.
        """
        if not self.arc_closed:
            return False
        band = _as_float(self.params.get("eis_validation_band_below_apex_decades"))
        minimum = _as_float(self.params.get("eis_validation_band_min_decades"))
        if not math.isfinite(minimum):
            minimum = 1.0
        return math.isfinite(band) and band >= minimum

    @property
    def apex_hz(self) -> float:
        return _as_float(self.params.get("eis_validation_apex_hz"))

    @property
    def f_lo_hz(self) -> float:
        """The grid's actual floor. ``f_lo_mHz`` is millihertz; this is not."""
        return _as_float(self.params.get("eis_validation_f_lo_hz"))

    @property
    def hold_epoch(self) -> int:
        try:
            return int(self.params.get("eis_validation_hold_epoch", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def hold_certified(self) -> str:
        return str(self.params.get("eis_validation_hold_certified", ""))

    @property
    def hold_excursion(self) -> bool:
        return bool(self.params.get("eis_validation_hold_excursion", False))

    @property
    def is_mock(self) -> bool:
        return bool(self.params.get("eis_validation_mock", False))

    @property
    def segmented(self) -> bool:
        return str(self.params.get("eis_sweep", "")) == "segmented"

    def passed_gates(self) -> set[str]:
        return {
            str(entry.get("gate", ""))
            for entry in self.gate_log
            if entry.get("passed")
        }

    def failed_gates(self) -> set[str]:
        return {
            str(entry.get("gate", ""))
            for entry in self.gate_log
            if entry.get("passed") is False
        }


def load_records(db_path: Path, validation_name: str) -> list[SweepRecord]:
    """Every sweep of *validation_name*, oldest first. Read-only."""
    conn = _connect_ro(db_path)
    try:
        rows = conn.execute(_SELECT).fetchall()
    finally:
        conn.close()

    records: list[SweepRecord] = []
    for row in rows:
        (mid, run_id, channel, ts, seconds, params_json, sigma, is_bound, r1,
         gate_verdict, gate_log_json, arc_state) = row
        params = _loads(params_json, {})
        if params.get("eis_validation_name") != validation_name:
            continue
        records.append(
            SweepRecord(
                measurement_id=int(mid),
                run_id=str(run_id),
                channel=int(channel),
                timestamp=str(ts),
                seconds=float(seconds or 0.0),
                params=params,
                sigma=None if sigma is None else float(sigma),
                sigma_is_bound=bool(is_bound),
                r1_ohm=None if r1 is None else float(r1),
                gate_verdict=gate_verdict,
                gate_log=_loads(gate_log_json, []),
                fit_arc_state=arc_state,
            )
        )
    return records


def load_checkpoint(db_path: Path, validation_name: str) -> dict[str, Any]:
    """The run's own record of *what was asked*, as the runner checkpointed it."""
    conn = _connect_ro(db_path)
    try:
        row = conn.execute(
            "SELECT spec_json FROM campaign_checkpoints WHERE campaign = ?",
            (checkpoint_campaign(validation_name),),
        ).fetchone()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    return _loads(row[0], {}) if row and row[0] else {}


def checkpoint_campaign(validation_name: str) -> str:
    return f"eis_validate:{validation_name}"


# ── The unit of analysis ─────────────────────────────────────────────────────

@dataclass
class Cell:
    """One channel, at one held condition, in one visit -- with both arms.

    Every statistic computed from a cell is a **within-cell ratio**, so the cell
    constant ``K`` in ``sigma = K/R`` cancels exactly and an entire class of
    confound -- electrode geometry, dead height, thickness method -- is removed
    for free. That is why nothing here is ever compared across cells.
    """

    key: str
    channel: int
    reference: SweepRecord | None = None
    scout: SweepRecord | None = None
    follow_up: SweepRecord | None = None
    reference_end: SweepRecord | None = None

    @property
    def adaptive(self) -> SweepRecord | None:
        """What adaptive acquisition actually delivered for this cell."""
        return self.follow_up or self.scout

    @property
    def complete(self) -> bool:
        """Both arms present. A half pair yields no deviation and is discarded."""
        return self.reference is not None and self.scout is not None

    @property
    def excursion(self) -> bool:
        return any(
            row.hold_excursion
            for row in (self.reference, self.scout, self.follow_up)
            if row is not None
        )

    @property
    def population(self) -> str:
        """CONTROL / TREATMENT / UNRESOLVED, decided by two frequencies.

        Both the scout's verdict and the reference's validity are functions of
        where the cell's apex sits, so the partition is exact and known per
        cell rather than estimated.

        The CONTROL population is automatically reference-valid and that falls
        out for free: ``ok`` requires a full decade of band below the apex on
        the baseline sweep, so a CONTROL cell's apex is far above the frequency
        at which the reference itself stops closing. The noise floor therefore
        never has to be estimated on cells where the reference is suspect.
        """
        if not self.complete:
            return EXCLUDED
        verdict = self.scout.scout_verdict
        if verdict == "ok":
            return CONTROL
        if verdict != "extend_low":
            # `no_arc`, `no_data`, `extend_high` -- adaptive declines to act on
            # these by design, so there is no treatment to measure.
            return EXCLUDED
        return TREATMENT if self.reference.reference_valid else UNRESOLVED

    # -- deviations, all in decades of log10 ----------------------------------

    def delta_scout(self) -> float | None:
        return _delta(self.scout, self.reference)

    def delta_adaptive(self) -> float | None:
        return _delta(self.adaptive, self.reference)

    def improvement(self) -> float | None:
        """``|Delta_scout| - |Delta_adaptive|``: **> 0 means adaptive moved toward
        the reference.**

        On a CONTROL cell this is identically 0 by construction, not by
        arithmetic luck: ``build_follow_up`` returned ``None``, nothing was
        written over the script, and :attr:`adaptive` *is* :attr:`scout` -- the
        same row, the same numbers.
        """
        d_scout, d_adapt = self.delta_scout(), self.delta_adaptive()
        if d_scout is None or d_adapt is None:
            return None
        return abs(d_scout) - abs(d_adapt)

    def delta_hold(self) -> float | None:
        """``log10(sigma_ref_end / sigma_ref_start)`` -- did the hold hold?"""
        return _delta(self.reference_end, self.reference)

    # -- time budget ----------------------------------------------------------

    def t_adaptive(self) -> float:
        """**Always both terms.** A comparison that hides the scout's cost is
        not a comparison; on a CONTROL cell the two are the same number and the
        marginal cost of adaptive is exactly zero, correctly."""
        return (self.scout.seconds if self.scout else 0.0) + (
            self.follow_up.seconds if self.follow_up else 0.0
        )

    def t_control(self) -> float:
        """The same sweep. The control **is** the scout."""
        return self.scout.seconds if self.scout else 0.0

    def t_reference(self) -> float:
        return sum(
            row.seconds
            for row in (self.reference, self.reference_end)
            if row is not None
        )

    # -- mechanism ------------------------------------------------------------

    def rescue_depth(self) -> tuple[float, float] | None:
        """``(required_reach, delivered_reach)`` in decades, or ``None``.

        Required reach comes from the **reference's** apex, which exists
        precisely because TREATMENT is defined by the reference having closed.
        Delivered reach is how far the follow-up's floor actually moved. The
        gap between them is what separates a mechanism that was too short from
        a mechanism that reached far enough and bought nothing -- and so what
        routes a null to MECHANISM-LIMITED rather than NO-GO.
        """
        if self.scout is None or self.reference is None:
            return None
        f_lo_baseline = self.scout.f_lo_hz
        apex = self.reference.apex_hz
        if not (f_lo_baseline > 0 and apex > 0):
            return None
        required = math.log10(f_lo_baseline / (apex / 10.0))
        f_lo_follow = self.follow_up.f_lo_hz if self.follow_up else f_lo_baseline
        if not f_lo_follow > 0:
            return None
        delivered = math.log10(f_lo_baseline / f_lo_follow)
        return required, delivered


def assemble_cells(records: Sequence[SweepRecord]) -> list[Cell]:
    """Group sweeps into cells by ``eis_validation_cell``, in first-seen order."""
    cells: dict[str, Cell] = {}
    for row in records:
        key = row.cell
        if not key:
            continue
        cell = cells.get(key)
        if cell is None:
            cell = cells[key] = Cell(key=key, channel=row.channel)
        if row.arm == ARM_REFERENCE:
            cell.reference = row
        elif row.arm == ARM_SCOUT:
            cell.scout = row
        elif row.arm == ARM_FOLLOW_UP:
            cell.follow_up = row
        elif row.arm == ARM_REFERENCE_END:
            cell.reference_end = row
    return list(cells.values())


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
        "settled", ", ".join(sorted(certifications)) or "(no rows)",
        PASS if h1_ok else FAIL,
        "" if h1_ok else "ceiling / not_evaluable / disabled withholds the outcome",
    ))

    criteria.append(Criterion(
        "H2 no fault-grade excursion", "0 fault-grade",
        f"{n_excluded} cell(s) EXCLUDED from the accuracy tables "
        f"(warn-grade window), {len(usable)} usable",
        PASS if usable else FAIL,
        "a fault-grade excursion parks the run, so its absence is implied by "
        "there being rows at all; warn-grade cells are excluded above and "
        "counted here, which is why the count and not just the grade is printed",
    ))

    h3_ok = hold.median is None or hold.median <= H3_MAX_HOLD_DRIFT_DEC
    criteria.append(Criterion(
        "H3 median |Delta_hold| on the drift-check subset",
        f"<= {H3_MAX_HOLD_DRIFT_DEC:.2f} dec",
        _fmt(hold.median, "dec") + f" (n={hold.n})",
        PASS if hold.n and h3_ok else (INSUFFICIENT if not hold.n else FAIL),
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
        _fmt(d_scout_control_abs.median, "dec") + f" (n={d_scout_control_abs.n})",
        PASS if d3_ok else (INSUFFICIENT if not d_scout_control_abs.n else FAIL),
        "no improvement below this floor can be believed",
    ))

    d1_ok = (
        improvement.median is not None
        and improvement.median >= D1_MIN_MEDIAN_IMPROVEMENT_DEC
    )
    criteria.append(Criterion(
        "D1 median improvement on TREATMENT",
        f">= +{D1_MIN_MEDIAN_IMPROVEMENT_DEC:.2f} dec",
        _fmt(improvement.median, "dec") + f" (n={improvement.n})",
        PASS if d1_ok else FAIL,
    ))

    frac = improvement.positive_fraction
    d2_ok = frac is not None and frac >= D2_MIN_POSITIVE_FRACTION
    criteria.append(Criterion(
        "D2 fraction of TREATMENT cells with improvement > 0",
        f">= {D2_MIN_POSITIVE_FRACTION:.3f}",
        (f"{frac:.3f}" if frac is not None else "n/a")
        + f" ({improvement.n_positive}/{improvement.n})",
        PASS if d2_ok else FAIL,
        "separates a real shift from a tail",
    ))

    d4_negative = d_scout_treat.median is not None and d_scout_treat.median < 0
    criteria.append(Criterion(
        "D4 median Delta_scout on TREATMENT is negative",
        "< 0 (direction only)",
        _fmt(d_scout_treat.median, "dec"),
        PASS if d4_negative else FAIL,
        "PRE-REGISTERED SIGN: R1 extrapolated past the apex is OVERestimated "
        "(x1.598), and sigma = K/R, so the plain preset must read sigma LOW "
        "against the reference (-0.204 dec). Reported as a FLAG, not a hard "
        "fail -- but a GO with this inverted must be argued in prose.",
    ))

    ratio_cells = [c for c in usable if c.population in (CONTROL, TREATMENT)]
    sum_adaptive = sum(c.t_adaptive() for c in ratio_cells)
    sum_control = sum(c.t_control() for c in ratio_cells)
    ratio = (sum_adaptive / sum_control) if sum_control > 0 else None
    t1_ok = ratio is None or ratio <= T1_MAX_TIME_RATIO
    criteria.append(Criterion(
        "T1 sum(t_adaptive) / sum(t_control)",
        f"<= {T1_MAX_TIME_RATIO:.1f}x",
        (f"{ratio:.3f}x" if ratio is not None else "n/a"),
        PASS if t1_ok else FAIL,
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
    elif criteria[2].status != PASS:                     # H3
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


# ── Rendering ────────────────────────────────────────────────────────────────

def build_payload(
    records: Sequence[SweepRecord],
    cells: Sequence[Cell],
    spec: dict[str, Any],
    verdict: Verdict,
) -> dict[str, Any]:
    """The machine-readable report, carrying enough to re-analyse without re-running."""
    usable = [c for c in cells if not c.excursion]
    by_pop: dict[str, list[Cell]] = {}
    for cell in usable:
        by_pop.setdefault(cell.population, []).append(cell)

    treatment = by_pop.get(TREATMENT, [])
    control = by_pop.get(CONTROL, [])
    ratio_cells = control + treatment

    return {
        "schema": REPORT_SCHEMA,
        "validation_name": spec.get("validation_name", ""),
        "mock": any(r.is_mock for r in records),
        "spec": spec,
        "completeness": {
            "n_sweeps": len(records),
            "n_cells": len(cells),
            "n_complete_cells": sum(1 for c in cells if c.complete),
            "hold_epochs": sorted({r.hold_epoch for r in records}),
            "run_ids": sorted({r.run_id for r in records}),
            "n_excluded_excursion_cells": len(cells) - len(usable),
        },
        "populations": {
            pop: [c.key for c in group] for pop, group in sorted(by_pop.items())
        },
        "apex_histogram": _apex_histogram(cells),
        "noise_floor": describe(
            [abs(v) for v in (c.delta_scout() for c in control) if v is not None]
        ).as_dict(),
        "deviation": {
            "delta_scout": describe([c.delta_scout() for c in treatment]).as_dict(),
            "delta_adaptive": describe(
                [c.delta_adaptive() for c in treatment]).as_dict(),
            "improvement": describe([c.improvement() for c in treatment]).as_dict(),
        },
        "hold": {
            "delta_hold": describe([c.delta_hold() for c in usable]).as_dict(),
            "per_channel": {
                str(c.channel): c.delta_hold()
                for c in usable if c.delta_hold() is not None
            },
        },
        "mechanism": {
            "closure_discordance": _closure_discordance(treatment),
            "rescue_depth": [
                {"cell": c.key, "required_dec": d[0], "delivered_dec": d[1]}
                for c, d in ((c, c.rescue_depth()) for c in treatment)
                if d is not None
            ],
            "unresolved_verdicts": _verdict_counts(by_pop.get(UNRESOLVED, [])),
        },
        "time_budget": {
            "sum_t_adaptive_s": sum(c.t_adaptive() for c in ratio_cells),
            "sum_t_scout_s": sum(c.t_control() for c in ratio_cells),
            "sum_t_follow_up_s": sum(
                c.follow_up.seconds for c in ratio_cells if c.follow_up),
            "sum_t_control_s": sum(c.t_control() for c in ratio_cells),
            "sum_t_reference_s": sum(c.t_reference() for c in usable),
            "unresolved_seconds": sum(
                c.t_adaptive() for c in by_pop.get(UNRESOLVED, [])),
        },
        "sigma_bound_rows": [
            r.measurement_id for r in records if r.sigma_is_bound
        ],
        "decision_rule": [c.as_dict() for c in verdict.criteria],
        "vetoes": verdict.vetoes,
        "outcome": verdict.outcome,
        "outcome_reasons": verdict.reasons,
        "cells": [
            {
                "cell": c.key, "channel": c.channel, "population": c.population,
                "complete": c.complete, "excursion": c.excursion,
                "scout_verdict": c.scout.scout_verdict if c.scout else "",
                "reference_arc_closed": bool(
                    c.reference.arc_closed) if c.reference else None,
                "reference_apex_hz": c.reference.apex_hz if c.reference else None,
                "delta_scout": c.delta_scout(),
                "delta_adaptive": c.delta_adaptive(),
                "improvement": c.improvement(),
                "delta_hold": c.delta_hold(),
                "t_adaptive_s": c.t_adaptive(),
                "t_control_s": c.t_control(),
                "t_reference_s": c.t_reference(),
            }
            for c in cells
        ],
        "caveats": CAVEATS,
    }


CAVEATS: tuple[str, ...] = (
    "There is NO ground truth for sigma on this rig. `Extended` is a proxy and "
    "a closed arc is still a fit, not a certificate.",
    "The reference is valid ONLY where its own arc closed. UNRESOLVED cells "
    "carry no accuracy number, by construction, not by omission.",
    "Segmented grids carry eis_duration_basis = 'extrapolated' for a structural "
    "reason: the grid is generated per sample, so no timing anchor can match it.",
    "Offset and scatter are reported separately. The failure mode under test is "
    "biased, not scattered, and an RMS would destroy that distinction.",
)


def render(payload: dict[str, Any]) -> str:
    """ASCII only. A report whose warnings arrive as mojibake gets skipped."""
    spec = payload.get("spec", {})
    out: list[str] = []
    add = out.append

    add("")
    add("=" * 74)
    add(f"EIS ADAPTIVE ACQUISITION -- VALIDATION REPORT"
        f"{'  (MOCK)' if payload['mock'] else ''}")
    add("=" * 74)
    add(f"  validation      {payload['validation_name']}")
    add(f"  condition       RH {spec.get('rh_setpoint_pct', '?')} %  "
        f"T {spec.get('temp_setpoint_c', '?')} C")
    add(f"  baseline        {spec.get('baseline_preset', '?')}  "
        f"(resolved from {spec.get('baseline_source', 'unknown')})")
    add(f"  reference       {spec.get('reference_preset', '?')}")
    comp = payload["completeness"]
    add(f"  completeness    {comp['n_complete_cells']}/{comp['n_cells']} cells "
        f"complete, {comp['n_sweeps']} sweeps, hold epochs {comp['hold_epochs']}")

    add("")
    add("-- 2. HOLD " + "-" * 63)
    hold = payload["hold"]["delta_hold"]
    add(f"  median |Delta_hold| {_fmt(hold['median'], 'dec')}  "
        f"MAD {_fmt(hold['mad'], 'dec')}  n={hold['n']}")
    for ch, value in sorted(payload["hold"]["per_channel"].items()):
        add(f"    ch{ch:<4} Delta_hold = {value:+.4f} dec")

    add("")
    add("-- 3. POPULATIONS " + "-" * 56)
    pops = payload["populations"]
    for pop in (CONTROL, TREATMENT, UNRESOLVED, EXCLUDED):
        add(f"  {pop:<11} {len(pops.get(pop, [])):>3}")
    hist = payload["apex_histogram"]
    add(f"  apex histogram: below {hist['ref_close_hz']:g} Hz: "
        f"{hist['below']} | {hist['ref_close_hz']:g}-{hist['baseline_ok_hz']:g} Hz: "
        f"{hist['window']} | above: {hist['above']}")
    add("  THE REFERENCE IS A PROXY, valid only where its OWN arc closed with a")
    add("  full decade of band below the apex: "
        f"{hist['reference_valid']}/{hist['reference_total']} cells.")
    excluded = payload["completeness"].get("n_excluded_excursion_cells", 0)
    if excluded:
        add(f"  {excluded} cell(s) EXCLUDED above: taken inside a warn-grade "
            "hold excursion.")

    add("")
    add("-- 4. NOISE FLOOR (CONTROL) " + "-" * 46)
    _add_spread(add, payload["noise_floor"], "|Delta_scout|")

    add("")
    add("-- 5. DEVIATION (TREATMENT) " + "-" * 46)
    for label, key in (("Delta_scout", "delta_scout"),
                       ("Delta_adaptive", "delta_adaptive"),
                       ("improvement", "improvement")):
        _add_spread(add, payload["deviation"][key], label)

    add("")
    add("-- 6. MECHANISM " + "-" * 58)
    disc = payload["mechanism"]["closure_discordance"]
    add("  closure: scout CLOSED? vs adaptive CLOSED?")
    add(f"    open  -> open  {disc['open_open']:>3}     "
        f"open  -> closed {disc['open_closed']:>3}")
    add(f"    closed-> open  {disc['closed_open']:>3}     "
        f"closed-> closed {disc['closed_closed']:>3}")
    depths = payload["mechanism"]["rescue_depth"]
    if depths:
        add("  rescue depth (required vs delivered, decades):")
        for entry in depths:
            add(f"    {entry['cell']:<28} required {entry['required_dec']:+.3f}  "
                f"delivered {entry['delivered_dec']:+.3f}")
    add(f"  UNRESOLVED verdicts: {payload['mechanism']['unresolved_verdicts'] or '{}'}")

    add("")
    add("-- 7. TIME BUDGET " + "-" * 56)
    budget = payload["time_budget"]
    add(f"  sum t_adaptive  {budget['sum_t_adaptive_s']:9.1f} s  "
        f"(scout {budget['sum_t_scout_s']:.1f} + follow-up "
        f"{budget['sum_t_follow_up_s']:.1f})")
    add(f"  sum t_control   {budget['sum_t_control_s']:9.1f} s")
    add(f"  reference cost  {budget['sum_t_reference_s']:9.1f} s   "
        "REPORTED, NEVER IN THE RATIO -- nobody proposes running the reference "
        "per AE trial")
    add(f"  UNRESOLVED      {budget['unresolved_seconds']:9.1f} s   "
        "excluded from the ratio: no ratio exists against a failed reference")

    add("")
    add("-- 8. THE DECISION RULE " + "-" * 50)
    for entry in payload["decision_rule"]:
        add(f"  [{entry['status']:<12}] {entry['criterion']}")
        add(f"                 threshold {entry['threshold']}   "
            f"observed {entry['observed']}")
        if entry["note"]:
            for line in _wrap(entry["note"], 66):
                add(f"                 {line}")
    for veto in payload["vetoes"]:
        add(f"  [VETO        ] {veto}")

    add("")
    add("  " + "=" * 70)
    add(f"  OUTCOME: {payload['outcome']}")
    for reason in payload["outcome_reasons"]:
        for line in _wrap(reason, 68):
            add(f"    {line}")
    add("  " + "=" * 70)

    add("")
    add("-- 9. CAVEATS " + "-" * 60)
    for caveat in payload["caveats"]:
        for i, line in enumerate(_wrap(caveat, 68)):
            add(("  * " if i == 0 else "    ") + line)
    add("")
    return "\n".join(out)


def _add_spread(add: Any, spread: dict[str, Any], label: str) -> None:
    add(f"  {label}")
    add(f"    n {spread['n']:<4} median {_fmt(spread['median'], 'dec')}  "
        f"MAD {_fmt(spread['mad'], 'dec')}  IQR {_fmt(spread['iqr'], 'dec')}")
    add(f"    min {_fmt(spread['min'], 'dec')}  max {_fmt(spread['max'], 'dec')}  "
        f"sign +{spread['n_positive']} / -{spread['n_negative']} "
        f"/ 0:{spread['n_zero']}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def generate(
    db_path: Path, validation_name: str, *, min_treatment: int | None = None
) -> dict[str, Any]:
    """Load, classify, evaluate. The whole report as a dict."""
    records = load_records(db_path, validation_name)
    cells = assemble_cells(records)
    spec = load_checkpoint(db_path, validation_name)
    spec.setdefault("validation_name", validation_name)
    threshold = (
        min_treatment if min_treatment is not None
        else int(spec.get("min_treatment", 6))
    )
    verdict = evaluate(
        cells, min_treatment=threshold, mock=any(r.is_mock for r in records)
    )
    return build_payload(records, cells, spec, verdict)


def resolve_project(explicit: str | None) -> Path:
    """``--project``, else the loader's ``[data] project_dir``. Never a literal.

    The repo root holds a 0-row database stub that a hardcoded relative path
    would silently find and report as an empty validation.
    """
    if explicit:
        return Path(explicit).expanduser()
    from softae.config import loader

    return Path(loader.data_project_dir()).expanduser()


def resolve_db(project: Path) -> Path:
    try:
        from softae.config import loader

        name = loader.data_db_filename()
    except Exception:
        name = "softae.db"
    return Path(project) / "db" / name


def cmd_report(args: Any) -> int:
    project = resolve_project(getattr(args, "project", None))
    db_path = resolve_db(project)
    if not db_path.exists():
        print(f"No database at {db_path}")
        return 1

    payload = generate(
        db_path, args.validation_name,
        min_treatment=getattr(args, "min_treatment", None),
    )
    if not payload["completeness"]["n_sweeps"]:
        print(f"No sweeps recorded for validation '{args.validation_name}'.")
        return 1

    print(render(payload))
    out = getattr(args, "out", None)
    if out:
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {out}")
    return 0


# ── Small helpers ────────────────────────────────────────────────────────────

def _delta(numerator: SweepRecord | None, denominator: SweepRecord | None) -> float | None:
    """``log10(sigma_num / sigma_den)``, falling back to ``log10(R_den / R_num)``.

    The fallback is exact rather than approximate: ``sigma = K/R`` with the same
    ``K`` on both sides of a within-cell ratio, so ``sigma_a/sigma_b`` IS
    ``R_b/R_a``. Where no cell geometry resolved, the decision is unaffected and
    sigma is simply reported absent.

    A **bounded** sigma is not missing -- it is informative about closure and
    uninformative about magnitude -- so it is excluded from the deviation rather
    than silently treated as a value or as an absence.
    """
    if numerator is None or denominator is None:
        return None
    if numerator.sigma_is_bound or denominator.sigma_is_bound:
        return None
    a, b = numerator.sigma, denominator.sigma
    if a is not None and b is not None and a > 0 and b > 0:
        return math.log10(a / b)
    ra, rb = numerator.r1_ohm, denominator.r1_ohm
    if ra is not None and rb is not None and ra > 0 and rb > 0:
        return math.log10(rb / ra)
    return None


def _apex_histogram(cells: Sequence[Cell]) -> dict[str, Any]:
    ref_close = _first_float(cells, "eis_validation_ref_close_hz", 13.51)
    baseline_ok = _first_float(cells, "eis_validation_baseline_ok_hz", 64.75)
    apexes = [
        c.reference.apex_hz for c in cells
        if c.reference is not None and c.reference.apex_hz > 0
    ]
    return {
        "ref_close_hz": ref_close,
        "baseline_ok_hz": baseline_ok,
        "below": sum(1 for a in apexes if a < ref_close),
        "window": sum(1 for a in apexes if ref_close <= a < baseline_ok),
        "above": sum(1 for a in apexes if a >= baseline_ok),
        "reference_valid": sum(
            1 for c in cells
            if c.reference is not None and c.reference.reference_valid),
        "reference_total": sum(1 for c in cells if c.reference is not None),
    }


def _first_float(cells: Sequence[Cell], key: str, default: float) -> float:
    for cell in cells:
        for row in (cell.reference, cell.scout):
            if row is not None and key in row.params:
                value = _as_float(row.params[key])
                if math.isfinite(value):
                    return value
    return default


def _closure_discordance(treatment: Sequence[Cell]) -> dict[str, int]:
    table = {"open_open": 0, "open_closed": 0, "closed_open": 0, "closed_closed": 0}
    for cell in treatment:
        if cell.scout is None or cell.adaptive is None:
            continue
        key = ("closed" if cell.scout.arc_closed else "open") + "_" + (
            "closed" if cell.adaptive.arc_closed else "open")
        table[key] += 1
    return table


def _verdict_counts(cells: Sequence[Cell]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        if cell.scout is None:
            continue
        counts[cell.scout.scout_verdict] = counts.get(cell.scout.scout_verdict, 0) + 1
    return counts


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


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _loads(text: Any, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "n/a".rjust(9)
    return f"{value:+9.4f}" + (f" {unit}" if unit else "")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


__all__ = [
    "ARM_FOLLOW_UP", "ARM_REFERENCE", "ARM_REFERENCE_END", "ARM_SCOUT",
    "CONTROL", "EXCLUDED", "REPORT_SCHEMA", "TREATMENT", "UNRESOLVED",
    "Cell", "Criterion", "Spread", "SweepRecord", "Verdict",
    "assemble_cells", "build_payload", "checkpoint_campaign", "cmd_report",
    "describe", "evaluate", "evaluate_vetoes", "generate", "load_checkpoint",
    "load_records", "render", "resolve_db", "resolve_project",
]
