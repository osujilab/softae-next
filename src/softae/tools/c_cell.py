"""``python -m softae.tools.c_cell`` — measure C_cell per channel group, preview it, record it.

Slice 2, lane L2 (``docs/SubAgent docs/regime_b_slice2_spec.md`` §2.2–2.3; operator rulings
Q1–Q2 of 2026-10-02)::

    python -m softae.tools.c_cell --board 2 --run 20260923T183923Z_rung3b_bench \\
        --run 20260929T150528Z_manual_eis --run 20261002T053100Z_rung3c_paa_bench
    python -m softae.tools.c_cell --board 2 --run ... --write --accepted-by CO

**Preview is the default and reads only.** The store is opened ``mode=ro``, so SQLite itself
refuses every write; spectra are read from their files, fixture-corrected as the engine
does, and judged by :mod:`softae.analysis.eis.cell_capacitance`. Every spectrum is listed
with its value and every rule it failed, then each group's C_cell with its sources and
spread. A group with no qualifying spectrum shows as *unavailable*, never a default.

**Writing needs ``--write``, ``--accepted-by`` and a typed ``yes``**, and goes only through
``DataStore.record_cell_capacitance`` — eis-acq's table (lane L3). Until that method exists
``--write`` refuses and writes nothing anywhere: there is no fallback file.

**Selection.** ``--run`` takes each channel's *latest* read in the named runs (the production
read) unless ``--all-reads``; ``--measurement`` adds exact ids, never thinned. Only
``role='sample'`` rows: a commissioning capacitor would pass every insulator rule.

**The board is the operator's statement** (``--board``): measurements do not record one.
Occupancy then checks it per read: the board's ``electrode_occupancy`` row must show the
well cast by that run or before that read, else the read is refused (empty or unrecorded).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import structlog

from softae.analysis.eis import cell_capacitance as cc
from softae.tools import use_utf8_console

logger = structlog.get_logger(__name__)

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
API_MISSING = ("the cell_capacitance table has not landed yet: DataStore has no "
               "record_cell_capacitance (eis-acq lane L3). Nothing was written.")


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """The only connection this tool's reading half makes. A missing store is an error."""
    if not Path(db_path).is_file():
        raise FileNotFoundError(f"DataStore not found: {db_path}")
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def select_measurements(conn: sqlite3.Connection, *, runs: Sequence[str] = (),
                        measurement_ids: Sequence[int] = (),
                        channels: Sequence[int] | None = None,
                        all_reads: bool = False) -> list[dict[str, Any]]:
    """Sample rows with a spectrum file: latest per channel from *runs*, plus exact ids."""
    base = ("SELECT measurement_id, run_id, channel, timestamp, eis_file_path "
            "FROM measurements WHERE eis_file_path IS NOT NULL AND eis_file_path != '' "
            "AND COALESCE(role, 'sample') = 'sample' AND {} ORDER BY timestamp, measurement_id")
    by_run: list[dict[str, Any]] = []
    if runs:
        q = base.format(f"run_id IN ({','.join('?' * len(runs))})")
        by_run = [dict(r) for r in conn.execute(q, list(runs))]
    if channels is not None:
        by_run = [r for r in by_run if int(r["channel"]) in set(channels)]
    if not all_reads:
        by_run = list({int(r["channel"]): r for r in by_run}.values())   # last wins
    explicit: list[dict[str, Any]] = []
    if measurement_ids:
        q = base.format(f"measurement_id IN ({','.join('?' * len(measurement_ids))})")
        explicit = [dict(r) for r in conn.execute(q, [int(i) for i in measurement_ids])]
    seen: set[int] = set()
    return [r for r in by_run + explicit
            if not (int(r["measurement_id"]) in seen or seen.add(int(r["measurement_id"])))]


def _aware(text: Any, *, naive_is_utc: bool) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(text).replace(" ", "T"))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc) if naive_is_utc else t.astimezone()
    return t


def occupancy_state(row: dict[str, Any] | None, *, run_id: str, timestamp: str) -> str:
    """Was the well cast when this read was taken? Unknown is never *occupied*.

    The run's own cast counts whatever ``cast_at`` says: rungs 3b and 3c stamped their
    occupancy after the run (fixed in ``86c856f``). Otherwise the cast must precede the
    read. ``cast_at`` naive is SQLite's UTC default; a naive measurement timestamp is the
    rig's local clock (``EISResult.timestamp.isoformat()``).
    """
    if row is None:
        return cc.UNKNOWN
    if row.get("run_id") and row["run_id"] == run_id:
        return cc.OCCUPIED
    cast = _aware(row.get("cast_at"), naive_is_utc=True)
    read = _aware(timestamp, naive_is_utc=False)
    if cast is None or read is None:
        return cc.UNKNOWN
    return cc.OCCUPIED if cast <= read else cc.EMPTY


def _load_corrected(path: Path, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """``(f, Z)`` fixture-corrected exactly as the gated engine does."""
    from softae.analysis.eis.engine_support import apply_correction_arrays
    from softae.analysis.eis.fixture import resolve_correction
    from softae.analysis.eis_data import EISResult

    e = EISResult.load(path)
    f = np.asarray(e.frequency, dtype=float)
    Z = np.asarray(e.z_real, dtype=float) - 1j * np.asarray(e.z_imag_neg, dtype=float)
    Zc, _ = apply_correction_arrays(f, Z, resolve_correction(int(channel)))
    return f, Zc


def assess(rows: Sequence[dict[str, Any]], *, project: Path,
           occupancy: dict[int, dict[str, Any]]
           ) -> tuple[list[cc.InsulatorCheck], list[str]]:
    """One check per readable row; unreadable rows are named, never silently dropped."""
    checks, skipped = [], []
    for r in rows:
        mid, ch = int(r["measurement_id"]), int(r["channel"])
        path = Path(str(r["eis_file_path"]).replace("\\", "/"))
        path = path if path.is_absolute() else Path(project) / path
        try:
            f, Z = _load_corrected(path, ch)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the preview
            skipped.append(f"measurement {mid} ch{ch}: unreadable ({path}): {exc}")
            continue
        occ = occupancy_state(occupancy.get(ch), run_id=str(r["run_id"]),
                              timestamp=str(r["timestamp"]))
        checks.append(cc.assess_spectrum(f, Z, channel=ch, occupancy=occ, measurement_id=mid))
    return checks, skipped


def render(checks: Sequence[cc.InsulatorCheck], groups: dict, *, board_id: int) -> str:
    lines = [f"C_cell preview, board {board_id}, method {cc.METHOD}", "",
             f"{'meas':>6} {'ch':>3} {'regime':<24} {'C (F)':>10} {'n':>2} {'phase':>6} "
             f"{'flat':>5} {'f_lo':>7} {'foot':>6}  verdict"]
    for c in sorted(checks, key=lambda c: (c.channel, c.measurement_id or 0)):
        cval = f"{c.c_cell_F:.3e}" if c.c_cell_F is not None else "-"
        verdict = "SOURCE" if c.qualifies else "refused: " + ", ".join(c.reasons)
        lines.append(f"{c.measurement_id or '-':>6} {c.channel:>3} "
                     f"{(c.regime or '?') + '/' + c.regime_reason:<24} {cval:>10} "
                     f"{c.n_band:>2} {c.phase_med_deg:6.1f} {c.flat_ratio:5.2f} "
                     f"{c.f_lo_hz:7.2f} {c.foot_tand:6.2f}  {verdict}")
    lines += ["", "Per group (median over channels; spread = log10 max/min):"]
    for g in cc.CHANNEL_GROUPS:
        rec = groups.get(g)
        if rec is None:
            lines.append(f"  {g[0]:>2}-{g[1]:<2}  unavailable (no qualifying insulating film)")
            continue
        chans = ", ".join(f"ch{ch} {v:.3e}" for ch, v in rec.channel_values)
        lines.append(f"  {g[0]:>2}-{g[1]:<2}  C_cell = {rec.c_cell_F:.3e} F  spread "
                     f"{rec.spread_dec:.3f} dec  [{chans}]  sources "
                     f"{list(rec.source_measurement_ids)}")
    return "\n".join(lines)


def write_groups(store: Any, groups: dict, *, accepted_by: str) -> list[Any]:
    """Append one row per group through eis-acq's contract. Nothing else is written."""
    return [store.record_cell_capacitance(
        board_id=rec.board_id, group=rec.group, c_cell_F=rec.c_cell_F,
        spread=rec.spread_dec, method=rec.method,
        source_measurement_ids=list(rec.source_measurement_ids), accepted_by=accepted_by)
        for _, rec in sorted(groups.items())]


def _default_store_cls() -> type:
    from softae.core.data_store import DataStore

    return DataStore


def _parse_channels(text: str) -> list[int]:
    out: list[int] = []
    for part in filter(None, (p.strip() for p in str(text).split(","))):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m softae.tools.c_cell", description=(
        "Measure C_cell per channel group from insulating films; preview by default."))
    p.add_argument("--board", type=int, required=True, help="board the reads were taken on")
    p.add_argument("--run", action="append", default=[], help="run id (repeatable)")
    p.add_argument("--measurement", type=int, action="append", default=[],
                   help="exact measurement id (repeatable)")
    p.add_argument("--channels", type=_parse_channels, default=None, help='e.g. "1-8,20"')
    p.add_argument("--all-reads", action="store_true",
                   help="every read per channel, not only the latest")
    p.add_argument("--project", default=None, help="data project dir (default: config)")
    p.add_argument("--write", action="store_true", help="append the groups (typed yes)")
    p.add_argument("--accepted-by", default="", help="operator initials (with --write)")
    return p


def main(argv: Sequence[str] | None = None, *, input_fn: Callable[[str], str] = input,
         store_cls: type | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    if not (args.run or args.measurement):
        print("Nothing selected: give --run and/or --measurement.", file=sys.stderr)
        return EXIT_FAILED
    if args.write and not args.accepted_by.strip():
        print("--write needs --accepted-by <initials>.", file=sys.stderr)
        return EXIT_REFUSED
    from softae.config import loader

    project = Path(args.project or loader.data_project_dir()).expanduser()
    db_filename = loader.data_db_filename()
    conn = connect_ro(project / "db" / db_filename)
    try:
        rows = select_measurements(conn, runs=args.run, measurement_ids=args.measurement,
                                   channels=args.channels, all_reads=args.all_reads)
        occ = {int(r["electrode"]): dict(r) for r in conn.execute(
            "SELECT electrode, run_id, cast_at FROM electrode_occupancy WHERE board_id = ?",
            (args.board,))}
    finally:
        conn.close()
    checks, skipped = assess(rows, project=project, occupancy=occ)
    groups = cc.aggregate_groups(checks, board_id=args.board)
    print(render(checks, groups, board_id=args.board))
    for s in skipped:
        print(f"  skipped {s}")
    if not args.write:
        print("\nPreview only: nothing was written.")
        return EXIT_OK
    if not groups:
        print("\nNo group has a qualifying source: nothing to write.")
        return EXIT_FAILED
    store_cls = store_cls or _default_store_cls()
    if not callable(getattr(store_cls, "record_cell_capacitance", None)):
        print("\n" + API_MISSING)
        return EXIT_REFUSED
    try:
        answer = input_fn(f"\nAppend {len(groups)} row(s) to cell_capacitance for board "
                          f"{args.board}, accepted by {args.accepted_by}? Type yes: ")
    except EOFError:
        answer = ""
    if answer.strip() != "yes":
        print("Not confirmed: nothing was written.")
        return EXIT_REFUSED
    store = store_cls(project, db_filename=db_filename)
    try:
        ids = write_groups(store, groups, accepted_by=args.accepted_by.strip())
    finally:
        getattr(store, "close", lambda: None)()
    logger.info("c_cell_recorded", board_id=args.board, rows=ids)
    print(f"Recorded {len(ids)} row(s): {ids}")
    return EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
