"""``softae-thickness`` — plan an unconfounded thickness series, then record it.

A CLI for the same reason ``softae-commission`` is one: this is bench work. The operator
is standing at the rig with a profilometer, reading a number off a screen for one channel
at a time.

Four subcommands, in the order they are actually used::

    softae-thickness plan --levels 100,150,200,250 --channels 1-32
    softae-thickness record --channel 7 --um 148.2 --uncertainty 3.0
    softae-thickness check --plan geo-2026-08-06
    softae-thickness list --plan geo-2026-08-06

**``plan`` comes first, and that ordering is the whole point.** Overhaul F12's confounded
series — CH27/28 = 200 µm, CH29/30 = 150, CH31/32 = 100 — was not a recording mistake. The
thickness levels were *assigned* in channel order at cast time, so a channel artifact and a
thickness effect became mathematically indistinguishable, and no later analysis can
separate them. A tool that only recorded measurements would have recorded that series
faithfully and said nothing.

``check`` closes the loop: it compares what was cast against what was planned, because a
sound plan followed inattentively produces exactly the dataset the plan existed to prevent.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from softae.core.thickness_series import (
    DEFAULT_MAX_CORRELATION,
    ThicknessPlanError,
    detect_confounding,
    plan_series,
)
from softae.tools import use_utf8_console

logger = structlog.get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFOUNDED = 3


def _parse_channels(text: str) -> list[int]:
    """``"1, 3-6"`` → ``[1, 3, 4, 5, 6]`` — the same syntax every other tool accepts."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _parse_levels(text: str) -> list[float]:
    return [float(p) for p in str(text).split(",") if p.strip()]


def _open_store(args):
    """The project store everything else already uses."""
    from softae.config import loader
    from softae.core.data_store import DataStore

    project = args.project or loader.data_project_dir()
    return DataStore(project, db_filename=loader.data_db_filename())


# ── plan ─────────────────────────────────────────────────────────────────────

def _cmd_plan(args) -> int:
    levels = _parse_levels(args.levels)
    channels = _parse_channels(args.channels)
    plan_id = args.id or f"geo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    try:
        plan = plan_series(
            levels, channels, seed=args.seed,
            max_correlation=args.max_correlation,
            created_at=datetime.now().isoformat(timespec="seconds"),
            plan_id=plan_id, notes=args.notes or "",
        )
    except ThicknessPlanError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_FAILED

    print(plan.describe())
    print()
    ok, why = plan.is_adequate_for_geometry_series()
    if not ok:
        print(f"  ! This plan is unconfounded but NOT sufficient for a geometry "
              f"series: {why}")
        print("    It is still a valid series — it just cannot answer E5.")
        print()

    print("Cast list — channel: target thickness (um)")
    row = []
    for ch, lv in plan.as_rows():
        row.append(f"{ch:>3}:{lv:>6.0f}")
        if len(row) == 8:
            print("  " + "  ".join(row))
            row = []
    if row:
        print("  " + "  ".join(row))

    store = _open_store(args)
    try:
        store.record_thickness_plan(plan)
    finally:
        store.close()

    print()
    print(f"Saved as plan '{plan_id}'.")
    # The list above is channel -> level, which is the assignment, NOT the order to
    # cast in. Casting down it would leave any interrupted prefix confounded, so the
    # next step is `cast`, which round-robins the levels.
    print("  The list above is the ASSIGNMENT, not the cast order. Next:")
    print(f"    softae-thickness cast --plan {plan_id}      # resolves the order")
    print(f"    softae-thickness record --plan {plan_id} --channel N --um X")
    return EXIT_OK


# ── record ───────────────────────────────────────────────────────────────────

def _cmd_record(args) -> int:
    store = _open_store(args)
    try:
        level = args.level
        plan = store.thickness_plan(args.plan) if args.plan else None
        if plan is not None and level is None:
            level = plan.assignment.get(int(args.channel))
            if level is None:
                print(f"  ! ch{args.channel} is not in plan '{args.plan}'.",
                      file=sys.stderr)

        rid = store.record_thickness(
            args.channel, args.um, plan_id=args.plan, run_id=args.run,
            level_um=level, uncertainty_um=args.uncertainty,
            instrument=args.instrument, operator=args.operator, notes=args.notes,
        )
        target = f" (planned {level:g} um)" if level is not None else ""
        print(f"Recorded #{rid}: ch{args.channel} = {args.um:g} um{target}")

        if level is not None and level > 0:
            dev = (args.um - level) / level * 100.0
            if abs(dev) >= 20.0:
                print(f"  ! {dev:+.0f}% from the planned level. Worth a second look "
                      f"before it enters a geometry fit.")
    finally:
        store.close()
    return EXIT_OK


# ── fit ──────────────────────────────────────────────────────────────────────

def _spectrum_for_channel(store, channel: int, run_id: str | None) -> Any:
    """The most recent **sample** spectrum for a channel, loaded from disk.

    Deliberately restricted to ``role = 'sample'``: the same channel may carry
    commissioning blanks, and regressing a blank against a film thickness would fit
    the fixture and call it conductivity.
    """
    from softae.analysis.eis_data import EISResult

    sql = ("SELECT eis_file_path FROM measurements "
           "WHERE channel = ? AND role = 'sample' AND eis_file_path IS NOT NULL")
    args: list[Any] = [int(channel)]
    if run_id:
        sql += " AND run_id = ?"
        args.append(run_id)
    sql += " ORDER BY measurement_id DESC LIMIT 1"
    row = store._conn.execute(sql, args).fetchone()
    if not row or not row[0]:
        return None
    path = Path(row[0])
    if not path.is_absolute():
        path = Path(store.project_dir) / path
    if not path.exists():
        return None
    try:
        return EISResult.load(path)
    except Exception:
        logger.warning("thickness_fit_spectrum_unreadable", channel=int(channel),
                       path=str(path), exc_info=True)
        return None


def _session_drift(store, run_id: str | None) -> tuple[int, float] | None:
    """``(channel, fractional change)`` from the §5.6 drift control, if one was run.

    One member is measured again at the end of the session and tagged
    ``role = 'drift_repeat'``; differencing it against its own first measurement
    isolates **session** drift — the films continuing to equilibrate over the ~11
    minutes a 16-channel sweep takes. That is the one error the geometry route is not
    otherwise immune to, because if measurement order correlates with thickness it
    becomes a false slope rather than scatter.

    Compared at the **median frequency of the overlap**, not at an end: the extremes of
    the band are where the phase floor and any HF artifact live, and a drift metric
    should not be dominated by either.
    """
    import numpy as np

    rows = store._conn.execute(
        "SELECT channel, eis_file_path FROM measurements "
        "WHERE role = 'drift_repeat'"
        + (" AND run_id = ?" if run_id else "")
        + " ORDER BY measurement_id DESC LIMIT 1",
        ([run_id] if run_id else []),
    ).fetchone()
    if not rows or not rows[1]:
        return None

    channel = int(rows[0])
    first = _spectrum_for_channel(store, channel, run_id)
    if first is None:
        return None

    from softae.analysis.eis_data import EISResult

    path = Path(rows[1])
    if not path.is_absolute():
        path = Path(store.project_dir) / path
    try:
        repeat = EISResult.load(path)
    except Exception:
        return None

    f0, f1 = np.asarray(first.frequency, float), np.asarray(repeat.frequency, float)
    common = sorted(set(np.round(np.log10(f0), 6)) & set(np.round(np.log10(f1), 6)))
    if not common:
        return None
    lf = common[len(common) // 2]
    i0 = int(np.argmin(np.abs(np.log10(f0) - lf)))
    i1 = int(np.argmin(np.abs(np.log10(f1) - lf)))
    with np.errstate(all="ignore"):
        g0 = float(np.real(1.0 / first.z_complex[i0]))
        g1 = float(np.real(1.0 / repeat.z_complex[i1]))
    if not (g0 == g0 and g0 != 0):
        return None
    return channel, (g1 - g0) / abs(g0)


def _cmd_fit(args) -> int:
    """Run the geometry-series fit over a recorded series (E5, framework §5.6).

    This is the entry point ``fit_geometry_series`` did not have. The module was
    written, tested against synthetic spectra, exported — and callable from nothing,
    which is the defect shape that reads as complete from inside the module.
    """
    import numpy as np

    from softae.analysis.eis.calibration import resolve_calibration
    from softae.analysis.eis.geometry import cell_config
    from softae.analysis.eis.geometry_series import SeriesMember, fit_geometry_series

    store = _open_store(args)
    try:
        rows = store.measured_thickness(plan_id=args.plan, run_id=args.run)
        if not rows:
            print("No thickness measurements recorded"
                  + (f" for plan '{args.plan}'." if args.plan else "."),
                  file=sys.stderr)
            return EXIT_FAILED

        cfg = cell_config()
        L_gap, L_stripe = cfg["L_gap_cm"], cfg["L_stripe_cm"]

        members: list[SeriesMember] = []
        missing: list[int] = []
        for r in rows:
            ch = int(r["channel"])
            eis = _spectrum_for_channel(store, ch, args.run)
            if eis is None:
                missing.append(ch)
                continue
            members.append(SeriesMember(
                thickness_cm=float(r["thickness_um"]) * 1e-4,
                frequency=np.asarray(eis.frequency, dtype=float),
                Z=eis.z_complex, channel=ch,
                label=str(r.get("instrument") or ""),
            ))

        if missing:
            # Named, not silently dropped: a channel measured with a profilometer but
            # never swept is a gap in the series, and a fit over what remains may be
            # unbalanced in exactly the way the planner worked to avoid.
            print(f"  ! no EIS spectrum for channel(s): "
                  f"{', '.join(str(c) for c in missing)}", file=sys.stderr)

        if len(members) < 2:
            print("Fewer than two channels have both a thickness and a spectrum.",
                  file=sys.stderr)
            return EXIT_FAILED

        fit = fit_geometry_series(members, L_gap_cm=L_gap, L_stripe_cm=L_stripe)
        print(fit.describe())
        print()
        print(f"  cell: L_gap {L_gap:g} cm, L_stripe {L_stripe:g} cm")
        print(f"  confounding: {fit.confound_verdict} (r = {fit.confound_correlation:+.3f})")
        if fit.issues:
            print()
            print("  Issues:")
            for i in fit.issues:
                print(f"    - {i}")

        # Session drift (§5.6's protocol control), if one was run.
        print()
        drift = _session_drift(store, args.run)
        if drift is None:
            print("  session drift: NOT MEASURED — no drift-control repeat in this run.")
            print("    Without it, drift that tracked measurement order is "
                  "indistinguishable")
            print("    from a real slope. Cast with `softae-thickness cast` to include "
                  "one.")
        else:
            ch, frac = drift
            print(f"  session drift: {frac * 100:+.1f}% on ch{ch} "
                  f"(first vs end-of-session repeat)")
            if abs(frac) > 0.05:
                print("    ! the sample changed materially during the sweep. The "
                      "interleaved")
                print("      cast order turns this into scatter rather than a slope, "
                      "but a drift")
                print("      of this size widens every uncertainty here.")

        # Dead height, if the fixture's own conductance was ever measured.
        cal = resolve_calibration(args.fixture)
        chans = [m.channel for m in members]
        G_tables = [cal.G_fixture[c] for c in chans
                    if cal is not None and c in getattr(cal, "G_fixture", {})] \
            if cal is not None else []
        print()
        if not G_tables:
            print("  dead height: UNAVAILABLE — no measured G_fixture for these "
                  "channels.")
            print("    h is not identifiable from a thickness series alone: the line "
                  "has one")
            print("    intercept carrying two unknowns (b = G_fixture - m*h). More "
                  "levels do")
            print("    not help. Run an open blank with RE tied to CE, then derive.")
        else:
            freqs = sorted({f for g in G_tables for f in g.freq_hz})
            median_G = {}
            for f0 in freqs:
                vals = [g.at(f0) for g in G_tables]
                vals = [v for v in vals if v == v]
                if vals:
                    median_G[f0] = float(np.median(vals))
            h = fit.dead_height_cm(median_G)
            profile = fit.dead_height_profile(median_G)
            hs = [v for _, v in profile]
            print(f"  dead height: {h * 1e4:.1f} um  (ADVISORY, never auto-applied)")
            if hs:
                print(f"    across {len(hs)} frequencies: "
                      f"{min(hs) * 1e4:.1f} to {max(hs) * 1e4:.1f} um")
                if max(hs) - min(hs) > abs(h) * 0.25:
                    # h is geometric and cannot depend on frequency. Drift means either
                    # G_fixture or the slope is wrong -- the median would hide it.
                    print("    ! h DRIFTS with frequency, so it is not a dead height. "
                          "Either G_fixture")
                    print("      or the slope is wrong; do not use this number.")
            spreads = [g.at(1e3) for g in G_tables]
            spreads = [s for s in spreads if s == s and s > 0]
            if len(spreads) >= 2 and max(spreads) / min(spreads) > 1.5:
                print(f"    ! G_fixture varies {max(spreads) / min(spreads):.1f}x "
                      f"across these channels, so the")
                print("      series' 'common intercept' is smeared. The measured "
                      "per-channel spread")
                print("      on this fixture is 2.4x, which enters the fit as scatter.")

        return EXIT_OK if fit.usable else EXIT_CONFOUNDED
    finally:
        store.close()


# ── cast ─────────────────────────────────────────────────────────────────────

def _cmd_cast(args) -> int:
    """Resolve a plan into a cast order and check it against the board (E5, link 2).

    **Prints by default and casts only with ``--execute``.** This is the one command
    here that spends a board and hours of anneal, and the order it resolves is not
    obvious — it round-robins the levels rather than walking channels, so what it
    prints is worth reading before it runs.
    """
    from softae.workflows.geometry_series import (
        GeometrySeriesError,
        choose_drift_channel,
        round_robin_by_level,
        verify_channels_free,
    )

    store = _open_store(args)
    try:
        plan = store.thickness_plan(args.plan)
        if plan is None:
            print(f"No plan '{args.plan}'.", file=sys.stderr)
            return EXIT_FAILED

        assignment = {int(k): float(v) for k, v in plan.assignment.items()}
        channels = sorted(assignment)
        occupied = store.occupied_electrodes(args.board)

        print(f"Plan '{args.plan}': {len(channels)} channels, "
              f"{len(set(assignment.values()))} levels, board {args.board}")
        print(f"  planned |r| = {plan.achieved_correlation:+.3f}")
        print()

        try:
            verify_channels_free(channels, occupied)
        except GeometrySeriesError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_FAILED

        order = round_robin_by_level(assignment)
        drift = choose_drift_channel(assignment) if not args.no_drift_control else None

        print("Cast/measure order (levels round-robin, NOT channel order):")
        for i in range(0, len(order), 8):
            chunk = order[i:i + 8]
            print("  " + "  ".join(f"ch{c}:{assignment[c]:g}um" for c in chunk))
        print()
        print("  Why this order: levels round-robin so an interrupted run stays")
        print("  balanced, and §5.6 wants drift to show as scatter, not as a slope.")
        half = max(4, len(order) // 2)
        print(f"  Measured: stopping any time after ~{half} channels keeps |r| below")
        print("  ~0.4, against ~0.75 for channel order. STOPPING EARLIER THAN THAT")
        print("  leaves a confounded subset whatever the order — too few points.")
        print()
        if drift is None:
            print("  drift control: NONE — session drift will be unmeasurable.")
        else:
            print(f"  drift control: ch{drift} measured again at the end of the "
                  f"session")
            print(f"    (a mid-level member: the thinnest and thickest films are the "
                  f"two")
            print(f"     most likely to behave atypically)")

        if not args.execute:
            print()
            print("Dry run. Re-run with --execute to cast.")
            return EXIT_OK

        print()
        print("Executing a geometry series requires the campaign spec that names the")
        print("stocks, pumps and recipe. Build it with:")
        print("    from softae.workflows.geometry_series import (")
        print("        volumes_for_levels, plan_geometry_series_run,")
        print("        build_geometry_series_workflow)")
        print("and run the returned Workflow through WorkflowExecutor, exactly as the")
        print("HT tab does. Wiring a spec into this CLI is deliberately not guessed at.")
        return EXIT_OK
    finally:
        store.close()


# ── check ────────────────────────────────────────────────────────────────────

def _cmd_check(args) -> int:
    """The gate. Exits non-zero on a confounded series so a script cannot ignore it."""
    store = _open_store(args)
    try:
        rows = store.measured_thickness(plan_id=args.plan, run_id=args.run)
        if not rows:
            print("No thickness measurements recorded"
                  + (f" for plan '{args.plan}'." if args.plan else "."),
                  file=sys.stderr)
            return EXIT_FAILED

        plan = store.thickness_plan(args.plan) if args.plan else None
        channels = [int(r["channel"]) for r in rows]
        # Check the level the cast *aimed at* where known, else what was measured.
        levels = [float(r["level_um"]) if r["level_um"] is not None
                  else float(r["thickness_um"]) for r in rows]

        report = detect_confounding(channels, levels,
                                    max_correlation=args.max_correlation, plan=plan)
        print(report.describe())
        print()
        print(f"  levels and replicates: "
              f"{ {k: v for k, v in sorted(report.replicates.items())} }")

        if report.deviations:
            print()
            print("  Deviations from the plan:")
            for d in report.deviations:
                print(f"    - {d}")

        if report.pending:
            chans = ", ".join(str(c) for c in report.pending[:12])
            more = f" (+{len(report.pending) - 12} more)" if len(
                report.pending) > 12 else ""
            print()
            print(f"  Not yet cast/measured ({len(report.pending)}): {chans}{more}")

        if report.verdict == "indeterminate":
            print()
            print("  Not enough of the series exists yet to certify it either way.")
            print("  Re-run `check` once more channels are recorded.")
            return EXIT_OK

        if report.confounded:
            print()
            print("  This series cannot separate a thickness effect from a")
            print("  channel-to-channel fixture difference. See overhaul F12.")
            if report.pending:
                # A sound plan part-way through casting can read as correlated: the
                # subset measured so far is its own design. Say so, rather than
                # sending the operator to re-plan something that is already correct.
                print(f"  However {len(report.pending)} planned channel(s) are still "
                      f"uncast — finish the cast and re-check before re-planning; the "
                      f"full plan scored |r| = "
                      f"{abs(plan.achieved_correlation):.3f}."
                      if plan is not None else
                      f"  However {len(report.pending)} planned channel(s) are still "
                      f"uncast — finish the cast and re-check.")
            else:
                print("  Re-plan with `softae-thickness plan` and cast to that order.")
            return EXIT_CONFOUNDED

        if plan is not None:
            ok, why = plan.is_adequate_for_geometry_series()
            print()
            print(f"  Geometry-series adequacy: {why}")
    finally:
        store.close()
    return EXIT_OK


# ── list ─────────────────────────────────────────────────────────────────────

def _cmd_list(args) -> int:
    store = _open_store(args)
    try:
        if args.plans:
            plans = store.thickness_plans()
            if not plans:
                print("No plans recorded.")
                return EXIT_OK
            for p in plans:
                print(f"{p['plan_id']:<28} {p['created_at'] or '':<20} "
                      f"{p['notes'] or ''}")
            return EXIT_OK

        rows = store.measured_thickness(plan_id=args.plan, run_id=args.run)
        if not rows:
            print("No thickness measurements recorded.")
            return EXIT_OK
        print(f"{'ch':>3} {'measured':>10} {'+/-':>7} {'planned':>8} "
              f"{'instrument':<14} {'when':<20}")
        for r in rows:
            unc = f"{r['uncertainty_um']:.2f}" if r["uncertainty_um"] else ""
            lvl = f"{r['level_um']:.0f}" if r["level_um"] is not None else ""
            print(f"{r['channel']:>3} {r['thickness_um']:>10.2f} {unc:>7} {lvl:>8} "
                  f"{(r['instrument'] or ''):<14} {(r['measured_at'] or ''):<20}")
    finally:
        store.close()
    return EXIT_OK


# ── Entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softae-thickness",
        description="Plan an unconfounded thickness series, then record what was cast.",
        epilog="Plan BEFORE casting: confounding cannot be undone afterwards.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--project", help="project directory (default: [data] project_dir)")
        return sp

    pl = common(sub.add_parser("plan", help="assign levels to channels, unconfounded"))
    pl.add_argument("--levels", required=True, help='e.g. "100,150,200,250" (um)')
    pl.add_argument("--channels", required=True, help='e.g. "1-32"')
    pl.add_argument("--id", help="plan id (default: geo-<timestamp>)")
    pl.add_argument("--seed", type=int, default=0, help="reproducibility seed")
    pl.add_argument("--max-correlation", type=float, default=DEFAULT_MAX_CORRELATION,
                    dest="max_correlation",
                    help=f"|r| ceiling (default {DEFAULT_MAX_CORRELATION})")
    pl.add_argument("--notes")
    pl.set_defaults(func=_cmd_plan)

    rec = common(sub.add_parser("record", help="record one measured thickness"))
    rec.add_argument("--channel", type=int, required=True)
    rec.add_argument("--um", type=float, required=True, help="measured thickness, um")
    rec.add_argument("--uncertainty", type=float, help="1-sigma, um")
    rec.add_argument("--plan", help="plan id this channel belongs to")
    rec.add_argument("--run", help="run id, if part of a run")
    rec.add_argument("--level", type=float,
                     help="planned level (um); taken from the plan when omitted")
    rec.add_argument("--instrument", help="e.g. 'Dektak XT'")
    rec.add_argument("--operator")
    rec.add_argument("--notes")
    rec.set_defaults(func=_cmd_record)

    ck = common(sub.add_parser("check", help="is this series confounded?"))
    ck.add_argument("--plan")
    ck.add_argument("--run")
    ck.add_argument("--max-correlation", type=float, default=DEFAULT_MAX_CORRELATION,
                    dest="max_correlation")
    ck.set_defaults(func=_cmd_check)

    ca = common(sub.add_parser(
        "cast", help="resolve a plan into a cast order and check the board"))
    ca.add_argument("--plan", required=True)
    ca.add_argument("--board", type=int, default=0, help="board index (default 0)")
    ca.add_argument("--no-drift-control", action="store_true",
                    dest="no_drift_control",
                    help="omit the §5.6 end-of-session repeat")
    ca.add_argument("--execute", action="store_true",
                    help="actually cast (default is a dry run)")
    ca.set_defaults(func=_cmd_cast)

    ft = common(sub.add_parser(
        "fit", help="geometry-series fit: sigma from the slope, h if G_fixture exists"))
    ft.add_argument("--plan")
    ft.add_argument("--run")
    ft.add_argument("--fixture", default="mux16",
                    help="fixture whose calibration supplies G_fixture (default mux16)")
    ft.set_defaults(func=_cmd_fit)

    ls = common(sub.add_parser("list", help="show measurements or plans"))
    ls.add_argument("--plan")
    ls.add_argument("--run")
    ls.add_argument("--plans", action="store_true", help="list plans instead")
    ls.set_defaults(func=_cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        logger.warning("thickness_cli_failed", exc_info=True)
        return EXIT_FAILED


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
