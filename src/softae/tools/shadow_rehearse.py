"""``softae-shadow rehearse`` — replay the gated engine over spectra already on disk.

Bench queue item 7 is a **single-shot** run, and two things about it are guesses: what
observe-only gated analysis costs per spectrum, and what the T7.1 recommender produces
on a real population rather than on synthetic distributions.  Both are answerable from
spectra the rig already measured, for CPU time.  So this selects a stratified sample of
an existing run and pushes each spectrum through
:func:`~softae.analysis.eis.engine.analyze_spectrum` with ``settings=`` naming the gated
engine.  The engine's own structlog stream lands in a file ``softae-shadow review`` reads
with no special case at all; :mod:`softae.tools.shadow_rehearse_report` renders the cost.

Three non-goals, structural rather than promised:

* **No database writes.**  :class:`~softae.core.data_store.DataStore` is never
  constructed — its ``__init__`` mkdirs, sets ``journal_mode=WAL``, runs the DDL and
  eight migrations, and commits.  Idempotent is not read-only, and the whole claim is
  that a replay cannot alter the record it replays.  See :func:`_connect_ro`.
* **No rig.**  Analysis modules only; nothing is opened and no stage moves.
* **No config edits.**  The engine arrives as a ``settings=`` *argument*, so the live
  ``[eis] engine`` is neither read nor changed.

**It owns its log handle** (:func:`open_log_stream`), because the first probe run died
with ``UnicodeEncodeError`` on ``\\u03b4`` — a ``tan δ`` in a gate detail, through a
cp1252 stdout — which kills a rehearsal on its first *interesting* spectrum.
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from softae.tools import use_utf8_console
from softae.tools.shadow_rehearse_report import (
    CSV_COLUMNS, EXTRAPOLATION_WELLS, Bucket, MetricsTap, RehearsalArtifacts,
    TimingRecord, TimingSummary, format_duration, open_log_stream,
    project_duration_seconds, render_summary, summarize_timing, timing_csv_path)

logger = structlog.get_logger(__name__)

EXIT_OK = 0
#: Nothing to replay — no run, no files, unreadable database.
EXIT_NOTHING = 1

#: ``eq_ch1_Lup_S0_R0_ch1.txt`` — the equilibration workflow's filename, which encodes
#: the whole grid.  Selection therefore needs no database join beyond ``eis_file_path``.
_GRID = re.compile(
    r"^eq_ch(?P<c1>\d+)_(?P<leg>Lup|Ldown)_S(?P<sp>\d+)_R(?P<rd>\d+)_ch(?P<c2>\d+)\.txt$",
    re.IGNORECASE,
)

DEFAULT_ROUNDS = 2


# ── Corpus ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CorpusRow:
    """One measurement, joined to its latest fit.  Everything a replay needs —
    ``role`` included, because it decides whether the gated engine judges the
    spectrum against the commissioned ``|Z|`` window at all.
    """

    measurement_id: int
    channel: int
    eis_file_path: str
    role: str = "sample"
    model_name: str = "simpleSalt"
    L_cm: float | None = None
    t_cm: float | None = None
    w_cm: float | None = None
    leg: str = "?"
    setpoint: int = -1
    round: int = 0

    @property
    def cell_key(self) -> tuple[str, int, int]:
        """``(leg, setpoint, channel)`` — the stratification unit.

        A filename the grid pattern does not match becomes **its own cell**, keyed on
        the measurement id, so it is never averaged into a block it does not belong to
        and never dropped for being unrecognised.
        """
        if self.leg == "?":
            return ("?", -1, self.measurement_id)
        return (self.leg, self.setpoint, self.channel)

    @property
    def block(self) -> str:
        return "unparsed" if self.leg == "?" else f"{self.leg}/S{self.setpoint}"

    @property
    def has_geometry(self) -> bool:
        return all(v is not None and v > 0 for v in (self.L_cm, self.t_cm, self.w_cm))


def _grid_fields(stored_path: str) -> tuple[str, int, int]:
    """``(leg, setpoint, round)`` from the stored path's basename, or the unparsed key."""
    name = str(stored_path or "").replace("\\", "/").rsplit("/", 1)[-1]
    m = _GRID.match(name)
    if m is None:
        return "?", -1, 0
    leg = m.group("leg")
    return leg[0].upper() + leg[1:].lower(), int(m.group("sp")), int(m.group("rd"))


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """The one place a connection is made, and SQLite itself refuses the write.

    ``mode=ro`` makes every ``INSERT`` on this handle raise ``OperationalError`` and
    leaves the WAL file untouched — no code path to audit.
    """
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def _latest_run(conn: sqlite3.Connection) -> str | None:
    """The most recent run that actually has spectra on disk to replay."""
    row = conn.execute(
        "SELECT e.run_id FROM experiments e "
        "JOIN measurements m ON m.run_id = e.run_id "
        "WHERE m.eis_file_path IS NOT NULL AND m.eis_file_path != '' "
        "GROUP BY e.run_id ORDER BY e.started_at DESC LIMIT 1"
    ).fetchone()
    return None if row is None else str(row[0])


def _read_corpus(db_path: Path, run_id: str | None = None
                 ) -> tuple[str | None, list[CorpusRow]]:
    """``(run_id, rows)`` for one run, read-only.  Two SELECTs, no DataStore.

    The fit join is a ``LEFT JOIN`` on purpose: a measurement with no fit row is
    replayable, it simply has no geometry, and the timing record says so
    (``cell_source="absent"``) rather than the spectrum vanishing from the population.

    ``m.role`` is selected rather than left to ``analyze_spectrum``'s own
    ``role="sample"`` default, because a commissioning run keeps its blanks and
    reference parts in the same table as its samples: the default judges every one of
    them as a sample, and the verdict that comes back looks entirely ordinary.
    """
    conn = _connect_ro(db_path)
    try:
        run = str(run_id) if run_id else _latest_run(conn)
        if not run:
            return None, []
        rows: list[CorpusRow] = []
        seen: set[int] = set()
        for (mid, channel, path, role, model, L, t, w) in conn.execute(
            "SELECT m.measurement_id, m.channel, m.eis_file_path, m.role, "
            "       f.model_name, f.electrode_L_cm, f.electrode_t_cm, f.electrode_w_cm "
            "FROM measurements m "
            "LEFT JOIN fit_results f ON f.measurement_id = m.measurement_id "
            "WHERE m.run_id = ? AND m.eis_file_path IS NOT NULL "
            "  AND m.eis_file_path != '' "
            "ORDER BY m.measurement_id, f.fit_id DESC", (run,)
        ):
            if int(mid) in seen:      # keep the latest fit per measurement
                continue
            seen.add(int(mid))
            leg, setpoint, rnd = _grid_fields(str(path))
            rows.append(CorpusRow(
                measurement_id=int(mid), channel=int(channel or -1),
                eis_file_path=str(path), role=str(role or "sample"),
                model_name=str(model or "simpleSalt"),
                L_cm=None if L is None else float(L),
                t_cm=None if t is None else float(t),
                w_cm=None if w is None else float(w),
                leg=leg, setpoint=setpoint, round=rnd,
            ))
        return run, rows
    finally:
        conn.close()


def resolve_path(project: Path | str, stored: str) -> Path:
    """``eis_file_path`` → an absolute path, on any platform.

    Stored paths carry **Windows separators**, and ``Path(project) / rel`` silently
    yields one bogus filename on POSIX, where the test suite runs.  Normalised first;
    an absolute stored path is used as-is.
    """
    raw = str(stored or "").strip().replace("\\", "/")
    p = Path(raw)
    return p if p.is_absolute() else Path(project) / p


# ── Selection ────────────────────────────────────────────────────────────────

@dataclass
class RehearsalPlan:
    """What will be replayed, decided before a single spectrum is analysed."""

    rows: list[CorpusRow] = field(default_factory=list)
    n_corpus: int = 0
    n_cells: int = 0
    n_cells_selected: int = 0
    rounds_requested: int = DEFAULT_ROUNDS
    round_labels: tuple[int, ...] = ()
    seed: int | None = None
    take_all: bool = False
    limit: int | None = None

    @property
    def n_selected(self) -> int:
        return len(self.rows)

    @property
    def dropped_cells(self) -> int:
        return max(0, self.n_cells - self.n_cells_selected)

    @property
    def projected_seconds(self) -> float:
        """Wall-clock the last rehearsal's rates imply.  An estimate, labelled as one."""
        return project_duration_seconds(self.n_selected)

    def describe(self) -> str:
        if self.take_all:
            spec = "every round"
        elif self.round_labels:
            spec = "rounds " + ", ".join(f"R{r}" for r in self.round_labels)
        else:
            spec = "rounds varied per cell"
        line = (f"{self.n_cells} cells x {self.rounds_requested} rounds = "
                f"{self.n_selected} spectra of {self.n_corpus}; {spec}")
        if self.dropped_cells:
            line += f"  [--limit {self.limit} dropped {self.dropped_cells} cells]"
        return line


def _picks(available: int, rounds: int, take_all: bool,
           rng: random.Random | None) -> list[int]:
    """Which positions inside one cell to take.

    Evenly spaced across the round axis — ``floor(i × available / N)`` — because rounds
    *are* the drift axis (R0 is pre-equilibration, R14 is settled), so spacing the picks
    samples that for free.  Clustering at zero would sample one end of the ramp twice.
    """
    if take_all or rounds >= available:
        return list(range(available))
    if rng is not None:
        return sorted(rng.sample(range(available), rounds))
    return sorted({(i * available) // rounds for i in range(rounds)})


def select_spectra(rows: "list[CorpusRow]", *, rounds: int = DEFAULT_ROUNDS,
                   take_all: bool = False, limit: int | None = None,
                   seed: int | None = None) -> RehearsalPlan:
    """A stratified, deterministic plan: *n* rounds from every ``(leg, setpoint, channel)``.

    Every block is represented, including the expensive tail.  Per-block open-arc rates
    span 8 % to 72 % and the open fraction is what sets total cost, so a convenience
    slice — "the first 200 rows" — is all one block and would report the *fast* mode as
    the whole distribution.  ``seed`` is not the default: without it two rehearsals of
    one corpus are comparable line by line.  ``limit`` truncates **after** stratification
    and the ordering is round-major, so a cut is a prefix of a balanced plan rather than
    of one block; cells it lost entirely are counted, not hidden.
    """
    cells: dict[tuple[str, int, int], list[CorpusRow]] = {}
    for row in rows:
        cells.setdefault(row.cell_key, []).append(row)

    rng = None if seed is None else random.Random(int(seed))
    rounds = max(1, int(rounds))
    passes: list[list[CorpusRow]] = []
    labels: set[int] = set()
    for key in sorted(cells, key=lambda k: (k[0], k[1], k[2])):
        members = sorted(cells[key], key=lambda r: (r.round, r.measurement_id))
        chosen = _picks(len(members), rounds, take_all, rng)
        for depth, idx in enumerate(chosen):
            while len(passes) <= depth:
                passes.append([])
            passes[depth].append(members[idx])
            labels.add(members[idx].round)

    ordered = [row for group in passes for row in group]
    if limit is not None and limit >= 0:
        ordered = ordered[:int(limit)]

    return RehearsalPlan(
        rows=ordered, n_corpus=len(rows), n_cells=len(cells),
        n_cells_selected=len({r.cell_key for r in ordered}),
        rounds_requested=rounds,
        round_labels=tuple(sorted(labels)) if seed is None and len(labels) <= 8 else (),
        seed=seed, take_all=take_all, limit=limit,
    )


# ── The replay ───────────────────────────────────────────────────────────────

@dataclass
class RehearsalResult:
    """Everything the run produced, so the caller renders rather than re-derives."""

    plan: RehearsalPlan
    records: list[TimingRecord] = field(default_factory=list)
    n_missing: int = 0
    n_unloadable: int = 0
    n_raised: int = 0
    model_name: str = "simpleSalt"
    enforced: bool = False
    n_cell_from_fit: int = 0
    #: Ctrl-C ended the loop early.  The records are real; the *population* is a prefix
    #: of the plan, which is why the summary labels itself partial rather than silently
    #: reporting a smaller n.
    interrupted: bool = False

    @property
    def mode(self) -> str:
        return "enforcing" if self.enforced else "observing"


#: Skip reason → the counter it increments.  One mapping so a new reason cannot be
#: logged without also being counted, which is how a skip goes silently missing.
_SKIP_COUNTER = {"missing": "n_missing", "unloadable": "n_unloadable",
                 "raised": "n_raised"}


def _skip(result: "RehearsalResult", reason: str, row: CorpusRow, path: Path,
          exc: Exception | None = None) -> None:
    """Count one lost spectrum and say why.  Never raises, never ends the batch."""
    field_name = _SKIP_COUNTER[reason]
    setattr(result, field_name, getattr(result, field_name) + 1)
    logger.warning("rehearsal_spectrum_failed", reason=reason,
                   measurement_id=row.measurement_id, path=str(path),
                   error="" if exc is None else str(exc))


def replay_settings(enforced: bool = False) -> Any:
    """The settings a replay runs under: **live thresholds, forced engine**.

    Gate *thresholds* come from the live ``[eis.gates]`` table, because the rehearsal's
    claim is that it judges what a bench run would: an operator who has armed calibrated
    values must get verdicts under those, not under shipped generic defaults.  Read
    through the explicit ``config=`` parameter, so the answer depends on what was loaded
    here rather than on whatever an earlier caller cached.

    Exactly two fields are forced and neither is reachable from the file: ``engine`` is
    always ``"gated"`` (a rig sitting on ``legacy`` is the normal case and must still be
    replayable), and ``gates.enabled`` follows *enforced*.  ``objective`` and
    ``[eis.fixture]`` pass through untouched.
    """
    from softae.analysis.eis.settings import EISSettings, eis_settings

    try:
        from softae.config import loader

        live = eis_settings(config=loader.load().get("eis", {}) or {})
    except Exception:      # noqa: BLE001 - an unreadable config must not stop a replay
        live = EISSettings()
    return replace(live, engine="gated",
                   gates=replace(live.gates, enabled=bool(enforced)))


def run_rehearsal(plan: RehearsalPlan, project: Path, *,
                  artifacts: "RehearsalArtifacts | None" = None,
                  enforced: bool = False, model_override: str | None = None
                  ) -> RehearsalResult:
    """Replay every selected spectrum.  One skip never costs the batch.

    A rehearsal that aborts at spectrum 140 of 192 has spent an hour and answered
    nothing, so a missing file, an unparseable one, or an exception out of
    ``analyze_spectrum`` logs ``rehearsal_spectrum_failed``, counts, and carries on.
    A **Ctrl-C** ends the loop rather than the process: the records already earned are
    returned, flagged :attr:`RehearsalResult.interrupted`, and every one of them is
    already on disk because ``artifacts`` flushed it per spectrum.

    ``settings=`` is how the engine is chosen — ``engine.py`` reads ``cfg = settings if
    settings is not None else eis_settings()`` — so :func:`replay_settings` decides it
    here and the ambient config never does.  Everything else self-resolves (envelope,
    fixture correction) because the bench run will not pass them either.
    """
    settings = replay_settings(enforced)
    result = RehearsalResult(plan=plan, enforced=bool(enforced))

    for row in plan.rows:
        try:
            record = _replay_one(row, project, settings, result,
                                 artifacts=artifacts, model_override=model_override,
                                 warmup=not result.records)
        except KeyboardInterrupt:
            result.interrupted = True
            logger.warning("rehearsal_interrupted", n_analysed=len(result.records),
                           n_planned=plan.n_selected,
                           msg="every completed row is already on disk")
            break
        if record is None:      # skipped; _replay_one has already counted and said why
            continue
        result.records.append(record)
        if artifacts is not None:
            artifacts.write_row(record)
        logger.info("rehearsal_spectrum_done", spectrum_key=record.spectrum_key,
                    measurement_id=record.measurement_id, block=record.block,
                    channel=record.channel, round=record.round,
                    seconds=round(record.seconds, 4), verdict=record.verdict,
                    n_gates_failed=record.n_gates_failed, arc_state=record.arc_state)
    return result


def _replay_one(row: CorpusRow, project: Path, settings: Any, result: RehearsalResult, *,
                artifacts: "RehearsalArtifacts | None", model_override: str | None,
                warmup: bool) -> "TimingRecord | None":
    """Load, analyse and time one spectrum, or ``None`` if it could not be replayed.

    The three ``None`` paths — missing, unloadable, raised — are the resilience contract:
    each counts itself through :func:`_skip` before returning, so a caller that sees
    ``None`` needs to know nothing about why.
    """
    from softae.analysis.eis import engine as eis_engine
    from softae.analysis.eis.geometry import CellConstant
    from softae.analysis.eis_data import EISResult

    path = resolve_path(project, row.eis_file_path)
    if not path.is_file():
        _skip(result, "missing", row, path)
        return None

    t_load = time.perf_counter()
    try:
        eis = EISResult.load(path)
    except Exception as exc:      # noqa: BLE001 - one bad file must not end the run
        _skip(result, "unloadable", row, path, exc)
        return None
    load_seconds = time.perf_counter() - t_load

    cell = None
    if row.has_geometry:
        cell = CellConstant(L_gap_cm=float(row.L_cm), L_stripe_cm=float(row.w_cm),
                            thickness_cm=float(row.t_cm), thickness_method="nominal")
        result.n_cell_from_fit += 1
    model = model_override or row.model_name
    result.model_name = model

    t0 = time.perf_counter()
    try:
        report = eis_engine.analyze_spectrum(
            eis, cell=cell, model_name=model, settings=settings, role=row.role)
    except Exception as exc:      # noqa: BLE001 - see run_rehearsal's docstring
        _skip(result, "raised", row, path, exc)
        return None
    seconds = time.perf_counter() - t0

    return TimingRecord.of(
        row, report, artifacts.take() if artifacts is not None else {},
        seconds=seconds, load_seconds=load_seconds,
        cell_source="fit_row" if cell is not None else "absent",
        warmup=warmup, mode=result.mode)


# ── Command ──────────────────────────────────────────────────────────────────

def _default_log_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("logs") / f"rehearsal_{stamp}.log"


def _resolve_project(explicit: str | None) -> Path:
    """``--project``, else the config loader's ``[data] project_dir``.  Never a literal.

    The repo root holds a 0-row ``softae_results.db`` stub that a hardcoded relative
    path would silently find and report as an empty corpus.
    """
    if explicit:
        return Path(explicit).expanduser()
    from softae.config import loader

    return Path(loader.data_project_dir()).expanduser()


def _resolve_db(project: Path) -> Path:
    try:
        from softae.config import loader

        name = loader.data_db_filename()
    except Exception:      # noqa: BLE001 - a missing config must not stop a fixture run
        name = "softae.db"
    return project / "db" / name


def cmd_rehearse(args) -> int:
    """Select, replay, summarise.  Exit 1 when there is nothing to replay."""
    use_utf8_console()
    project = _resolve_project(getattr(args, "project", None))
    db_path = _resolve_db(project)
    if not db_path.is_file():
        print(f"No database at {db_path}", file=sys.stderr)
        return EXIT_NOTHING

    try:
        run_id, rows = _read_corpus(db_path, getattr(args, "run_id", None))
    except sqlite3.Error as exc:
        print(f"Cannot read {db_path}: {exc}", file=sys.stderr)
        return EXIT_NOTHING
    if not run_id or not rows:
        print(f"Nothing to replay in {db_path}"
              + (f" for run {args.run_id}" if getattr(args, "run_id", None) else ""),
              file=sys.stderr)
        return EXIT_NOTHING

    plan = select_spectra(rows, rounds=getattr(args, "rounds", DEFAULT_ROUNDS),
                          take_all=bool(getattr(args, "all", False)),
                          limit=getattr(args, "limit", None),
                          seed=getattr(args, "seed", None))
    if not plan.rows:
        print("Selection is empty; nothing to replay.", file=sys.stderr)
        return EXIT_NOTHING

    mode = "enforcing" if getattr(args, "enforced", False) else "observing"
    header = (f"run {run_id}\n{plan.describe()}\n"
              f"engine=gated  gates={mode}  projected "
              f"{format_duration(plan.projected_seconds)} "
              f"(2026-08-14 rates on a 39%-open mix; this corpus may differ)")
    if getattr(args, "dry_run", False):
        print(header)
        return EXIT_OK

    log_path = Path(getattr(args, "out", None) or _default_log_path()).expanduser()
    if log_path.exists():
        print(f"Refusing to overwrite {log_path}. Name a new file.", file=sys.stderr)
        return EXIT_NOTHING
    print(header)

    # The CSV is opened and header-written here, alongside the log, and each row is
    # flushed as its spectrum completes — see RehearsalArtifacts. Nothing is written
    # after the loop, so an interrupted run keeps everything it earned.
    with open_log_stream(log_path, tee=bool(getattr(args, "tee", False))) as artifacts:
        logger.info("rehearsal_started", run_id=run_id, project=str(project),
                    n_selected=plan.n_selected, n_corpus=plan.n_corpus,
                    n_cells=plan.n_cells, rounds=plan.rounds_requested,
                    seed=plan.seed, engine="gated", gates=mode,
                    projected_seconds=round(plan.projected_seconds, 1))
        result = run_rehearsal(plan, project, artifacts=artifacts,
                               enforced=bool(getattr(args, "enforced", False)),
                               model_override=getattr(args, "model", None))
        summary = summarize_timing(result.records)
        logger.info("rehearsal_summary", n_analysed=len(result.records),
                    n_missing=result.n_missing, n_unloadable=result.n_unloadable,
                    n_raised=result.n_raised, gates=mode,
                    interrupted=result.interrupted,
                    median_seconds=round(summary.all.median, 4)
                    if summary.all.n else None,
                    open_fraction=round(summary.open_fraction, 4)
                    if summary.open_fraction == summary.open_fraction else None)

    print("\n" + render_summary(result, summary, log_path, project))
    # 1 on an interrupt, which is what an uncaught KeyboardInterrupt would have returned
    # through ``main``; swallowing the signal must not turn a half-run into a success.
    return EXIT_NOTHING if result.interrupted else EXIT_OK


def _at_least_one(text: str) -> int:
    """An ``int`` argparse refuses below 1.

    ``--limit 0`` and ``--limit -5`` were silent no-ops — ``rows[:0]`` selects nothing,
    and a negative slice trims from the *end* of a balanced plan, the one truncation
    stratification exists to prevent.
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def add_rehearse_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags, defined here so the subparser in ``shadow_review`` stays wiring only."""
    parser.add_argument("--project", help="project directory (default: [data] project_dir)")
    parser.add_argument("--run-id", dest="run_id",
                        help="run to replay (default: the most recent with spectra)")
    parser.add_argument("--rounds", type=_at_least_one, default=DEFAULT_ROUNDS,
                        metavar="N",
                        help=f"rounds per (leg, setpoint, channel) cell "
                             f"(default {DEFAULT_ROUNDS})")
    parser.add_argument("--all", action="store_true",
                        help="every spectrum in the run")
    parser.add_argument("--limit", type=_at_least_one, default=None, metavar="N",
                        help="hard cap applied after stratification")
    parser.add_argument("--seed", type=int, default=None, metavar="S",
                        help="randomise the round picks (default: deterministic)")
    parser.add_argument("--out", metavar="PATH",
                        help="log file (default logs/rehearsal_<UTC>.log); the timing "
                             "CSV sits beside it. Refuses to overwrite.")
    parser.add_argument("--tee", action="store_true",
                        help="mirror the log to stdout for a watched run")
    parser.add_argument("--model", help="override the fit row's model_name")
    parser.add_argument("--enforced", action="store_true",
                        help="replay with gates ENFORCING, to time what observing costs")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="print the plan and the projected duration; analyse nothing")
    return parser


#: The artifact half — log stream, tap, CSV, summary, rendering — moved to
#: :mod:`softae.tools.shadow_rehearse_report` when this module passed the house line
#: limit, the same treatment ``shadow_review`` gave ``db_summary`` and ``render``.
#: Those names are re-exported here because this module was their public surface
#: first, and a pure move must not break a caller.
__all__ = ["CSV_COLUMNS", "EXTRAPOLATION_WELLS", "Bucket", "CorpusRow", "MetricsTap",
           "RehearsalArtifacts", "RehearsalPlan", "RehearsalResult", "TimingRecord",
           "TimingSummary", "add_rehearse_arguments", "cmd_rehearse", "format_duration",
           "open_log_stream", "render_summary", "replay_settings", "resolve_path",
           "run_rehearsal", "select_spectra", "summarize_timing", "timing_csv_path"]
