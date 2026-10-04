"""``python -m softae.tools.regime_b_replay`` — the regime-B shadow review, before and after.

Slice 2, wave 2 (``docs/SubAgent docs/regime_b_slice2_spec.md`` §5.3)::

    python -m softae.tools.regime_b_replay --run 20261002T053100Z_rung3c_paa_bench
    python -m softae.tools.regime_b_replay --run 20260923T183923Z_rung3b_bench \\
        --board 2 --c-cell 1-8=2.66e-10 --c-cell 17-24=3.08e-10 --thickness-um 160.4

Every selected spectrum runs through ``analyze_spectrum`` twice — the engine left to
``[eis] engine``, as every fit site leaves it ([a23]) — ``[eis] regime_aware`` on both
times: **before** with ``regime_b`` off (today's armed configuration) and **after** with it
on. One row per spectrum shows what each states — mode, σ, basis — and the B route's
outcome, detail, C_x source and film resistance; ``*`` marks a row whose stated claim moves.

**Reads only, and writes nothing anywhere.** The DataStore is opened ``mode=ro`` (SQLite
itself refuses every write) and never through ``DataStore``, whose open runs migrations.
Spectra are read from their files. Nothing is saved: the table goes to stdout.

**C_cell** per channel: a ``--c-cell`` override (``LO-HI=F`` for a group, ``CH=F`` for one
channel; a channel beats a group), else the store's current ``cell_capacitance`` row for
the board by ``preferred_record``, else *unmeasured* — never a default. The board is
``--board``, or the one board ``electrode_occupancy`` names for the selected runs.

**K** comes from ``--thickness-um`` (geometry from config), else the newest stored
``fit_results`` geometry for the measurement, else none (σ is then not stated).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import structlog

from softae.analysis.eis.cell_capacitance import preferred_record
from softae.analysis.eis.regime_route import RegimeSettings
from softae.tools import use_utf8_console
from softae.tools.c_cell import _parse_channels, connect_ro, select_measurements

EXIT_OK, EXIT_FAILED = 0, 1
BEFORE = RegimeSettings(enabled=True, regime_b=False)
AFTER = RegimeSettings(enabled=True, regime_b=True)


# ── C_cell, board and K (all reads) ─────────────────────────────────────────────────


def parse_c_cell(text: str) -> tuple[tuple[int, int], float]:
    """``"1-8=2.66e-10"`` → ``((1, 8), 2.66e-10)``; ``"19=3e-10"`` → ``((19, 19), 3e-10)``."""
    span, sep, value = str(text).partition("=")
    lo, _, hi = span.strip().partition("-")
    try:
        c = float(value)
        group = (int(lo), int(hi or lo))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--c-cell wants LO-HI=F or CH=F, got {text!r}")
    if not (sep and math.isfinite(c) and c > 0 and group[0] <= group[1]):
        raise argparse.ArgumentTypeError(f"--c-cell wants a positive farad value: {text!r}")
    return group, c


@dataclass(frozen=True)
class CCell:
    value: float | None
    source: str          # "override", "store #<id>", or "unmeasured"


def override_c_cell(channel: int, overrides: Sequence[tuple[tuple[int, int], float]]
                    ) -> float | None:
    """The narrowest override covering *channel*; the last given wins a tie."""
    covering = [(hi - lo, -i, c) for i, ((lo, hi), c) in enumerate(overrides)
                if lo <= channel <= hi]
    return min(covering)[2] if covering else None


def store_c_cell_rows(conn: sqlite3.Connection, board_id: int | None) -> tuple[list, str]:
    """The board's current ``cell_capacitance`` rows, and a note when there are none."""
    if board_id is None:
        return [], "no board: store C_cell not consulted (give --board)"
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM cell_capacitance WHERE board_id = ? AND superseded_at IS NULL "
            "ORDER BY id", (int(board_id),))]
    except sqlite3.OperationalError as exc:      # the table has not been created here
        return [], f"store has no cell_capacitance table ({exc})"
    return rows, "" if rows else f"store has no current C_cell row for board {board_id}"


def resolve_c_cell(channel: int, overrides: Sequence, rows: Sequence[dict]) -> CCell:
    c = override_c_cell(channel, overrides)
    if c is not None:
        return CCell(c, "override")
    rec = preferred_record(channel, rows)
    if rec is not None:
        return CCell(float(rec["c_cell_F"]), f"store #{rec['id']}")
    return CCell(None, "unmeasured")


def board_for_runs(conn: sqlite3.Connection, runs: Sequence[str]) -> int | None:
    """The single board ``electrode_occupancy`` names for *runs*; ``None`` if not one."""
    if not runs:
        return None
    q = (f"SELECT DISTINCT board_id FROM electrode_occupancy WHERE run_id IN "
         f"({','.join('?' * len(runs))})")
    boards = [int(r[0]) for r in conn.execute(q, list(runs))]
    return boards[0] if len(boards) == 1 else None


def stored_cell(conn: sqlite3.Connection, measurement_id: int) -> Any:
    """``CellConstant`` from the newest stored fit geometry for the measurement, or None."""
    from softae.analysis.eis.geometry import CellConstant

    row = conn.execute(
        "SELECT electrode_L_cm, electrode_w_cm, electrode_t_cm, thickness_method "
        "FROM fit_results WHERE measurement_id = ? AND electrode_t_cm > 0 "
        "ORDER BY fit_id DESC LIMIT 1", (int(measurement_id),)).fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None
    return CellConstant(L_gap_cm=float(row[0]), L_stripe_cm=float(row[1]),
                        thickness_cm=float(row[2]),
                        thickness_method=str(row[3] or "stored"))


def thickness_cell(thickness_um: float) -> Any:
    from softae.analysis.eis.geometry import cell_constant_for_sample

    return cell_constant_for_sample(predicted_um=float(thickness_um))


# ── The replay ──────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _quiet(verbose: bool) -> Iterator[None]:
    """The engine logs per spectrum; keep the table readable, and restore after."""
    if verbose:
        yield
        return
    saved = structlog.get_config()
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))
    try:
        yield
    finally:
        structlog.configure(**saved)


def _stated(s: Any) -> tuple[str, float, str]:
    """``(mode, σ, basis)`` — the number a report states, in its direction."""
    if s.mode in ("bound", "bound_unqualified"):
        return s.mode, s.upper_bound, s.upper_bound_basis
    if s.mode == "value":
        return s.mode, s.value, s.R_basis
    if s.regime_active and s.regime_sigma_lower == s.regime_sigma_lower:
        return "lower", s.regime_sigma_lower, s.R_basis
    return s.mode, float("nan"), ""


@dataclass(frozen=True)
class ReplayRow:
    engine: str
    measurement_id: int
    channel: int
    regime: str
    c_cell: CCell
    before: tuple[str, float, str]
    after: tuple[str, float, str]
    b_mode: str
    b_detail: str
    cx_source: str
    film_R_ohm: float
    film_basis: str

    @property
    def moved(self) -> bool:
        (m0, s0, b0), (m1, s1, b1) = self.before, self.after
        same_sigma = s0 == s1 or (s0 != s0 and s1 != s1)
        return not (m0 == m1 and b0 == b1 and same_sigma)


def replay_one(eis: Any, *, measurement_id: int, cell: Any, c_cell: CCell) -> ReplayRow:
    from softae.analysis.eis.engine import analyze_spectrum

    reports = [analyze_spectrum(eis, cell=cell, regime=flags, cell_capacitance=c_cell.value)
               for flags in (BEFORE, AFTER)]
    b, a = (r.sigma for r in reports)
    return ReplayRow(
        engine=reports[1].engine, measurement_id=measurement_id,
        channel=int(getattr(eis, "channel", -1) or -1),
        regime=f"{a.regime or '-'}/{a.regime_reason}", c_cell=c_cell,
        before=_stated(b), after=_stated(a), b_mode=a.regime_b_mode,
        b_detail=a.regime_b_detail, cx_source=a.regime_cx_source,
        film_R_ohm=a.regime_film_R_ohm, film_basis=a.regime_film_basis)


def _g(x: float) -> str:
    return f"{x:.3g}" if x == x else "-"


def render(rows: Sequence[ReplayRow], notes: Sequence[str]) -> str:
    head = (f"  {'meas':>6} {'ch':>3} {'regime':<20} {'C_cell (F)':>10} {'source':<11} "
            f"| {'before: mode':<16} {'σ':>9} {'basis':<20} "
            f"| {'after: mode':<16} {'σ':>9} {'basis':<20} "
            f"| {'B outcome':<11} {'detail':<24} {'cx':<10} {'film R':>9} film basis")
    lines = ["Regime-B shadow replay — before: regime_b off; after: regime_b on "
             "(regime_aware on in both). * = the stated claim moves.", "", head]
    for r in sorted(rows, key=lambda r: (r.channel, r.measurement_id)):
        cc = _g(r.c_cell.value) if r.c_cell.value is not None else "-"
        lines.append(
            f"{'*' if r.moved else ' '} {r.measurement_id:>6} {r.channel:>3} {r.regime:<20} "
            f"{cc:>10} {r.c_cell.source:<11} "
            f"| {r.before[0]:<16} {_g(r.before[1]):>9} {r.before[2]:<20} "
            f"| {r.after[0]:<16} {_g(r.after[1]):>9} {r.after[2]:<20} "
            f"| {r.b_mode or '-':<11} {r.b_detail or '-':<24} {r.cx_source or '-':<10} "
            f"{_g(r.film_R_ohm):>9} {r.film_basis or '-'}")
    moved = sum(r.moved for r in rows)
    if any(r.engine != "gated" for r in rows):
        notes = [*notes, "[eis] engine is not 'gated': the regime routes run on the gated "
                         "engine only, so nothing here can move"]
    lines += ["", f"{len(rows)} spectra, {moved} moved. Nothing was written."]
    lines += [f"  note: {n}" for n in notes]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m softae.tools.regime_b_replay", description=(
        "Replay stored spectra with [eis] regime_b off and on; read-only."))
    p.add_argument("--run", action="append", default=[], help="run id (repeatable)")
    p.add_argument("--measurement", type=int, action="append", default=[],
                   help="exact measurement id (repeatable)")
    p.add_argument("--channels", type=_parse_channels, default=None, help='e.g. "1-8,20"')
    p.add_argument("--all-reads", action="store_true",
                   help="every read per channel, not only the latest")
    p.add_argument("--board", type=int, default=None,
                   help="board for the store's C_cell (default: from electrode_occupancy)")
    p.add_argument("--c-cell", type=parse_c_cell, action="append", default=[],
                   help="override, LO-HI=F or CH=F (repeatable; beats the store)")
    p.add_argument("--thickness-um", type=float, default=None,
                   help="film thickness for K (default: stored fit geometry)")
    p.add_argument("--project", default=None, help="data project dir (default: config)")
    p.add_argument("-v", "--verbose", action="store_true", help="keep the engine's logs")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    if not (args.run or args.measurement):
        print("Nothing selected: give --run and/or --measurement.", file=sys.stderr)
        return EXIT_FAILED
    from softae.analysis.eis_data import EISResult
    from softae.config import loader

    project = Path(args.project or loader.data_project_dir()).expanduser()
    conn = connect_ro(project / "db" / loader.data_db_filename())
    notes: list[str] = []
    rows: list[ReplayRow] = []
    try:
        selected = select_measurements(conn, runs=args.run, measurement_ids=args.measurement,
                                       channels=args.channels, all_reads=args.all_reads)
        board = args.board if args.board is not None else board_for_runs(conn, args.run)
        store_rows, note = store_c_cell_rows(conn, board)
        notes += [note] if note else []
        fixed_cell = (thickness_cell(args.thickness_um) if args.thickness_um is not None
                      else None)
        with _quiet(args.verbose):
            for r in selected:
                mid, ch = int(r["measurement_id"]), int(r["channel"])
                path = Path(str(r["eis_file_path"]).replace("\\", "/"))
                path = path if path.is_absolute() else project / path
                try:
                    eis = EISResult.load(path)
                except Exception as exc:  # noqa: BLE001 - one bad file must not end it
                    notes.append(f"measurement {mid} ch{ch}: unreadable ({path}): {exc}")
                    continue
                cell = fixed_cell if fixed_cell is not None else stored_cell(conn, mid)
                if cell is None:
                    notes.append(f"measurement {mid} ch{ch}: no K (no thickness) — σ unstated")
                rows.append(replay_one(eis, measurement_id=mid, cell=cell,
                                       c_cell=resolve_c_cell(ch, args.c_cell, store_rows)))
    finally:
        conn.close()
    print(render(rows, notes))
    return EXIT_OK


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
