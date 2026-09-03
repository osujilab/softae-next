"""What was persisted, read back: sweeps, cells, and the deviations between them.

The reading half of the reporter. :mod:`softae.tools.eis_validate` writes rows;
this module is the only place that reads them, and it reads them **read-only** --
the connection is opened ``mode=ro``, so SQLite itself refuses every write and
there is no code path to audit. **No :class:`~softae.core.data_store.DataStore`
is ever constructed here.**

Nothing in this module knows a threshold. It answers three questions and stops:

1. *what rows exist* -- :class:`SweepRecord`, :func:`load_records`;
2. *which rows belong together* -- :class:`Cell`, :func:`assemble_cells`;
3. *what a within-cell comparison is worth* -- the ``delta_*`` methods, all in
   decades of log10, all within-cell so the cell constant ``K`` in ``sigma = K/R``
   cancels exactly (spec 3.4).

The judgements that do live here are statements about *the sweeps that were
taken* rather than about a criterion, which is why they can be computed without
the rule module ever being imported:

* :attr:`Cell.population` -- whether a cell is CONTROL, TREATMENT or UNRESOLVED
  is decided by two frequencies, the scout's verdict and the reference's own
  validity, both properties of the rows. That partition is what keeps a deviation
  between two extrapolations out of every accuracy table
  (:attr:`SweepRecord.reference_valid`).
* :attr:`Cell.stillness_certified` -- whether the hold gate could speak for this
  cell, read off the per-measurement ``hold_certified`` stamp. It **marks** rows
  and selects nothing: the populations above are unchanged by it, deliberately.

The rule that judges these numbers is :mod:`softae.tools.eis_validate_rule`; the
report that renders them is :mod:`softae.tools.eis_validate_report`, which is
also the single import surface both halves are re-exported through.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ARM_REFERENCE = "reference"
ARM_SCOUT = "adaptive_scout"
ARM_FOLLOW_UP = "adaptive_follow_up"
ARM_REFERENCE_END = "reference_end"

CONTROL = "CONTROL"
TREATMENT = "TREATMENT"
UNRESOLVED = "UNRESOLVED"
EXCLUDED = "EXCLUDED"

#: The hold certifications under which a **cell's** stillness was established.
#:
#: Two words, not one. ``settled`` is the whole board certified quiet;
#: ``survivors`` is a *partitioned* settle phase in which these cells were
#: certified quiet and others were dropped
#: (:data:`softae.analysis.equilibration.SETTLE_SURVIVORS`, whose own docstring
#: says it: "these cells were certified quiet, those cells could not be judged
#: and were dropped"). A survivor is a cell the gate **could** speak for, so
#: folding it in with the dropped cells would mark every row of a survivor run
#: and erase the one distinction this vocabulary exists to draw.
#:
#: This is a **cell**-level predicate and deliberately not the board-level
#: :attr:`softae.tools.eis_validate_hold.SettleOutcome.certified`, which is
#: ``verdict == "settled"`` and stays ``False`` for a survivor run. H1 keeps the
#: board-level rule and withholds the verdict on anything but ``settled``;
#: nothing here weakens it.
#:
#: Spelled as literals rather than imported from
#: :mod:`softae.analysis.equilibration`, for the same reason
#: :mod:`softae.tools.eis_validate_rule` spells ``"settled"`` as one: the reading
#: half imports nothing from ``softae``, and pulling numpy in through a shared
#: analysis module to read two strings is not a trade worth making.
CERTIFIED_STILL: frozenset[str] = frozenset({"settled", "survivors"})

#: What a cell whose rows carry no certification at all is called in a table.
#: Named rather than left blank because "undeclared is unknown, never empty" is
#: the house rule, and an empty column reads as an oversight.
UNSTAMPED = "unstamped"


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
        """Gates that **ran and passed** -- never one that could not check.

        ``GateResult.checked`` exists because a gate unable to evaluate its
        criterion must fail *open*: it returns ``passed=True`` as a placeholder
        and marks the placeholder with ``checked=False``
        (:meth:`softae.analysis.eis.gates.GateResult.unchecked`). Reading
        ``passed`` alone therefore counts "could not check" as "checked and
        clean", which is the conflation the field was added to remove.

        Three states, read via :func:`_gate_checked` with **no default**,
        because ``None`` is a third answer and not a synonym for ``True``:

        =============  ===========================================  ===========
        ``checked``    what the entry is saying                     in this set?
        =============  ===========================================  ===========
        ``True``       the gate ran and returned a verdict          if ``passed``
        ``False``      it could not run; ``passed`` is a            **never**
                       fail-open placeholder, not a verdict
        absent         the row predates the field                   if ``passed``
        =============  ===========================================  ===========

        **The third row is a judgement, not a default.** For a row written
        before the field existed, ``passed=True`` means exactly what it has
        always meant here -- "this gate did not refuse the spectrum" -- and no
        better information is recoverable from it, so this method reports it
        unchanged. Excluding it instead would empty this set for the *entire*
        stored corpus (every ``gate_log_json`` in the DataStore today predates
        ``checked``), silently retiring
        :func:`softae.tools.eis_validate_rule.evaluate_vetoes`'s V2 rather than
        repairing it. The absence is not swept under the pass either: it is
        readable at :attr:`gate_log_distinguishes_unchecked`, so a caller that
        needs the stronger claim can ask whether this row can support it.
        """
        return {
            str(entry.get("gate", ""))
            for entry in self.gate_log
            if entry.get("passed") and _gate_checked(entry) is not False
        }

    def unchecked_gates(self) -> set[str]:
        """Gates that could not evaluate their criterion on this sweep.

        Disjoint from :meth:`passed_gates` by construction and from
        :meth:`failed_gates` by ``GateResult.__post_init__``'s invariant, so
        the three sets partition a ``checked``-era log. A non-empty result
        means this spectrum's record has holes in it -- a different statement
        from either a pass or a refusal, and the one that used to be
        unavailable.
        """
        return {
            str(entry.get("gate", ""))
            for entry in self.gate_log
            if _gate_checked(entry) is False
        }

    @property
    def gate_log_distinguishes_unchecked(self) -> bool:
        """Can this row tell "checked and clean" from "could not check"?

        ``False`` for a log written before ``GateResult.checked`` existed and
        ``False`` for an empty log. In both cases :meth:`unchecked_gates` is
        empty because nothing was *recorded*, never because nothing was
        unchecked -- and the two must not read alike.
        """
        return bool(self.gate_log) and all(
            _gate_checked(entry) is not None for entry in self.gate_log
        )

    def failed_gates(self) -> set[str]:
        """Gates that ran and refused. Deliberately unchanged by ``checked``.

        ``GateResult.__post_init__`` forbids ``checked=False, passed=False``,
        so ``passed is False`` already implies the gate ran; an unchecked entry
        cannot reach this set. The identity test -- not truthiness -- is also
        what keeps an entry with no ``passed`` key out of it.
        """
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
    def rows(self) -> tuple[SweepRecord, ...]:
        """Every sweep this cell actually holds, in arm order."""
        return tuple(
            row for row in
            (self.reference, self.scout, self.follow_up, self.reference_end)
            if row is not None
        )

    @property
    def certifications(self) -> tuple[str, ...]:
        """Every distinct ``hold_certified`` this cell's rows carry, first seen first.

        Derived from the rows rather than stored, and that is the whole of the
        plumbing: :attr:`SweepRecord.hold_certified` has always read the stamp
        out of ``eis_params_json``, so no second query, no new column and no
        field that can go stale against the rows it summarises. A hand-built
        :class:`Cell` -- there are some, in tests -- reports its rows' real
        certification instead of a constructor default nobody passed.

        **All four arms, including ``reference_end``.** The stamp is written per
        measurement, so it is a fact about the row; a cell that quoted only three
        of its four rows would under-report, and ``delta_hold`` -- the one
        statistic ``reference_end`` feeds -- is a number a reader sees.
        """
        seen: list[str] = []
        for row in self.rows:
            if row.hold_certified and row.hold_certified not in seen:
                seen.append(row.hold_certified)
        return tuple(seen)

    @property
    def certification(self) -> str:
        """One word for a table cell; ``a + b`` when the cell's rows disagree."""
        return " + ".join(self.certifications) or UNSTAMPED

    @property
    def stillness_certified(self) -> bool:
        """Was this cell's stillness established? **Marks the row; keeps the number.**

        A cell this returns ``False`` for is *retained evidence*, never an error
        and never a dropped row. Its numbers stay in D1-D4 exactly as they were,
        because the operator's reason for keeping them is that they are the data
        the production limits will be calibrated from: *"having metrology on more
        channels before having valid criteria avoids downstream confusion and
        allows refinement on criteria for production limits."* A dropped row is
        data that would otherwise have to be re-earned on the rig, and keeping it
        costs nothing as long as its provenance is visible -- which is what this
        flag is for, and the whole of what it is for.

        So the obvious-looking "fix" -- filtering these cells out of the accuracy
        tables -- is wrong twice over: it discards the calibration evidence, and
        it edits a pre-registered criterion after the data arrived, which
        :mod:`softae.tools.eis_validate_rule` exists to forbid. Nothing routes on
        this property. H1 already withholds the verdict on any certification
        other than ``settled``, so no false ``GO`` can be emitted either way.

        **A cell whose rows disagree is not certified.** The stamp is per
        measurement, so disagreement inside one cell is a real possibility, and
        every statistic a cell produces is a within-cell ratio *across* those
        rows -- one uncertified row is enough to make the ratio a comparison
        against a hold that was not certified. The weaker claim therefore wins,
        and :attr:`certification` keeps both words so the report can say which
        two disagreed.
        """
        return bool(self.certifications) and set(self.certifications) <= CERTIFIED_STILL

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


def _gate_checked(entry: dict[str, Any]) -> bool | None:
    """``entry["checked"]`` as a tri-state: ``True``, ``False``, or *unrecorded*.

    ``.get("checked")`` with **no default**, deliberately. A default of ``True``
    would spell "this writer had no such concept" with the same token as "the
    gate ran and found nothing wrong", and a default of ``False`` would call
    every historical row unchecked; both are claims the record does not make.
    ``None`` is returned for an explicit JSON ``null`` too -- nothing writes one
    (``GateResult.as_log_entry`` emits ``bool(self.checked)``), and if something
    ever did, "unrecorded" is the honest reading of it.
    """
    value = entry.get("checked")
    return None if value is None else bool(value)


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


__all__ = [
    "ARM_FOLLOW_UP", "ARM_REFERENCE", "ARM_REFERENCE_END", "ARM_SCOUT",
    "CERTIFIED_STILL", "CONTROL", "EXCLUDED", "TREATMENT", "UNRESOLVED",
    "UNSTAMPED",
    "Cell", "SweepRecord", "assemble_cells", "checkpoint_campaign",
    "load_checkpoint", "load_records",
]
