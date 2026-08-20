"""Read back an EIS adaptive-acquisition validation and evaluate the rule.

Read-only, and regenerable at any moment -- including while the runner is still
measuring. That is the point of the runner/reporter split it inherits from
:mod:`softae.tools.shadow_rehearse` / :mod:`~softae.tools.shadow_rehearse_report`:
the runner persists after every single sweep and holds nothing in memory, so a
report is a pure function of what is on disk and a crash costs one spectrum.

The reporter is three modules, cut along the seam the spec itself draws --
reading (spec 8.7's rows), the rule (spec 6), and the rendering of both::

    eis_validate_records.py   rows -> SweepRecord -> Cell, and the deviations
    eis_validate_rule.py      the pre-registered thresholds and the outcome
    eis_validate_report.py    this -- the JSON payload, the text report, the CLI

**This module is the import surface.** Every name either half defines is
re-exported here, so `softae.tools.eis_validate_report` remains the one path any
caller imports and the split stays an internal fact. There is exactly one
spelling of each name; nothing is renamed on the way through.

**No :class:`~softae.core.data_store.DataStore` is ever constructed.** The
connection is opened ``mode=ro`` in the reading half, so SQLite itself refuses
every write and there is no code path to audit.

Three things the reporter will not do, each because the corresponding mistake is
the one that would matter:

1. **It never quotes a deviation against an open-arc reference as accuracy.**
   ``Extended`` reaches 1.351 Hz, so its own arc closes only for an apex above
   about 13.51 Hz; below that its ``R1`` came from extrapolating the
   high-frequency limb, measured on this rig at a **+60.9 % median** overestimate
   (+175.2 % with the full CPE fitter, p16 = 0.031). A "deviation" there is a
   deviation between two extrapolations. Those cells are UNRESOLVED: counted,
   listed, and kept out of every accuracy table
   (:attr:`~softae.tools.eis_validate_records.Cell.population`).

2. **It never combines offset and scatter into an RMS.** The failure mode under
   test is *biased, not scattered*. A median near zero with wide scatter means
   something categorically different from a small consistent offset, and an RMS
   destroys exactly that distinction. Median, MAD, IQR, min, max and the sign
   split are reported separately, for every population
   (:class:`~softae.tools.eis_validate_rule.Spread`).

3. **It never emits a GO from a mock run.** A simulated verdict is not a
   verdict. The arithmetic that would produce one is still exercised in full --
   that is what the grid-aware backend in :mod:`softae.tools.eis_validate_mock`
   is for -- but the outcome line says WITHHELD.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from softae.tools.eis_validate_records import (
    ARM_FOLLOW_UP,
    ARM_REFERENCE,
    ARM_REFERENCE_END,
    ARM_SCOUT,
    CONTROL,
    EXCLUDED,
    TREATMENT,
    UNRESOLVED,
    Cell,
    SweepRecord,
    _as_float,
    _connect_ro,  # noqa: F401  -- re-export; see the note below
    assemble_cells,
    checkpoint_campaign,
    load_checkpoint,
    load_records,
)
from softae.tools.eis_validate_rule import (
    D1_MIN_MEDIAN_IMPROVEMENT_DEC,
    D2_MIN_POSITIVE_FRACTION,
    D3_MAX_CONTROL_DEVIATION_DEC,
    FAIL,
    H3_MAX_HOLD_DRIFT_DEC,
    INSUFFICIENT,
    OUTCOME_CONDITIONAL_GO,
    OUTCOME_GO,
    OUTCOME_INSUFFICIENT,
    OUTCOME_MECHANISM_LIMITED,
    OUTCOME_NO_GO,
    OUTCOME_WITHHELD,
    PASS,
    T1_MAX_TIME_RATIO,
    Criterion,
    Spread,
    Verdict,
    _fmt,
    _median,  # noqa: F401  -- re-export; see the note below
    _quantile,  # noqa: F401  -- re-export; see the note below
    describe,
    evaluate,
    evaluate_vetoes,
)

# On the three re-exports the renderer does not itself call: the connection
# factory and the two order statistics are pinned directly by their tests,
# because "is this handle really read-only" and "is this really the median" are
# properties worth asserting at the definition rather than through a report.
# They travel under the same spelling, from one definition, without an alias.

#: JSON payload schema, mirroring ``"softae.eis_timing/1"``.
REPORT_SCHEMA = "softae.eis_validate/1"

#: Width of section 8's status column, derived from the statuses themselves so
#: that ``INSUFFICIENT`` -- a criterion that could not be evaluated, which is
#: neither a pass nor a failure -- lines up with ``PASS`` and ``FAIL`` instead of
#: shoving its own row two characters right. A literal here would misalign
#: silently the next time the rule grows a status.
STATUS_WIDTH = max(len(s) for s in (PASS, FAIL, INSUFFICIENT, "VETO"))
#: Column the criterion's own detail lines hang from: ``"  ["`` + status + ``"] "``.
_DETAIL_INDENT = " " * (STATUS_WIDTH + 5)


# ── The machine-readable report ──────────────────────────────────────────────

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


# ── Rendering ────────────────────────────────────────────────────────────────

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
    # How long the SAMPLE sat at that condition before the first spectrum. The
    # line above states what was commanded; without this one a reader cannot
    # tell a film measured on the drying transient from an equilibrated one, and
    # the two produce the same-looking report. Printed even at zero, because
    # "no soak" is the finding in that case.
    add(f"  soak            {_soak_line(spec)}")
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
        add(f"  [{entry['status']:<{STATUS_WIDTH}}] {entry['criterion']}")
        for line in _criterion_detail(entry):
            add(f"{_DETAIL_INDENT}{line}")
    for veto in payload["vetoes"]:
        add(f"  [{'VETO':<{STATUS_WIDTH}}] {veto}")

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


def _criterion_detail(entry: dict[str, Any]) -> list[str]:
    """The threshold/observed line and the note, wrapped to the block width.

    ``observed`` is a sentence, not a number, whenever a criterion could not be
    evaluated -- ``no CONTROL cells (n=0)`` rather than ``n/a``, because the
    reader's next question after INSUFFICIENT is always *why*. Those sentences
    are long enough to overflow the block and get re-wrapped by the terminal at
    an arbitrary column, which is exactly what a fixed-width report is for
    avoiding, so an overlong pair is split here instead.
    """
    threshold, observed = entry["threshold"], entry["observed"]
    joined = f"threshold {threshold}   observed {observed}"
    if len(joined) <= 66:
        lines = [joined]
    else:
        lines = [f"threshold {threshold}"] + _wrap(f"observed {observed}", 66)
    if entry["note"]:
        lines += _wrap(entry["note"], 66)
    return lines


def _soak_line(spec: dict[str, Any]) -> str:
    """How long the sample was held at condition before the first spectrum.

    ``soak_s`` reaches the reporter for free: it is a ``ValidationPlan`` field,
    so ``as_dict`` carries it into the campaign checkpoint's ``spec_json`` and
    ``load_checkpoint`` reads it back. A validation recorded before the soak
    existed has no such key -- reported as *not stated*, never as zero, because
    those are different claims and only one of them was made.
    """
    if "soak_s" not in spec:
        return "not stated (recorded before --soak-h existed)"
    try:
        seconds = float(spec.get("soak_s") or 0.0)
    except (TypeError, ValueError):
        return "not stated"
    if seconds <= 0:
        return "none -- the first spectrum followed the settle gate directly"
    return (f"{seconds / 3600:.2f} h held at condition before the first "
            "spectrum")


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
    "CONTROL", "D1_MIN_MEDIAN_IMPROVEMENT_DEC", "D2_MIN_POSITIVE_FRACTION",
    "D3_MAX_CONTROL_DEVIATION_DEC", "EXCLUDED", "FAIL",
    "H3_MAX_HOLD_DRIFT_DEC", "INSUFFICIENT", "OUTCOME_CONDITIONAL_GO",
    "OUTCOME_GO", "OUTCOME_INSUFFICIENT", "OUTCOME_MECHANISM_LIMITED",
    "OUTCOME_NO_GO", "OUTCOME_WITHHELD", "PASS", "REPORT_SCHEMA",
    "STATUS_WIDTH", "T1_MAX_TIME_RATIO", "TREATMENT", "UNRESOLVED",
    "Cell", "Criterion", "Spread", "SweepRecord", "Verdict",
    "assemble_cells", "build_payload", "checkpoint_campaign", "cmd_report",
    "describe", "evaluate", "evaluate_vetoes", "generate", "load_checkpoint",
    "load_records", "render", "resolve_db", "resolve_project",
]
