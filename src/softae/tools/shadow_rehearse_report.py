"""What a rehearsal cost, and how that reads — the reporting half of T7.8.

:mod:`softae.tools.shadow_rehearse` decides *which* spectra to replay and replays them.
This module owns the **artifacts** that replay produces: the log stream it is written to
and the tap that reads the engine's own events back off it, the per-spectrum record, the
CSV sidecar, the aggregation that turns a pile of records into a distribution, and the
operator-facing rendering of that.

The split is one-way — ``shadow_rehearse`` imports from here and nothing here imports
back — and it exists because the two halves change for unrelated reasons.  The replay
changes when the engine's call signature or the corpus layout changes; the artifacts
change when someone asks a different question of the same numbers.  Keeping them in one
file put the module at 879 lines, well past the house limit.

**Why the arc-closure split rather than a single median.**  The cost distribution is
bimodal and the fast mode is ~240× faster than the slow one.  An arc that closed inside
the swept window fits in tenths of a second (**median 0.16 s**); an open one gives the
fitter no in-band feature to converge onto and takes the long way round (**median 38 s,
max 58 s**).  Measured over the 192 real spectra of the 2026-08-14 rehearsal, where the
mix was **39 % open**.

A single median would therefore land wherever the open-arc fraction happened to put it —
and that fraction is a property of *the material*, not of the analysis: it ran 8 % to
72 % across the blocks of one run.  So the report gives the two modes and the mix, and
the extrapolation is bracketed rather than totalled (:meth:`TimingSummary.bracket`).
"""

from __future__ import annotations

import csv
import logging
import statistics
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import structlog

if TYPE_CHECKING:  # pragma: no cover - the reverse import would be a cycle
    from softae.tools.shadow_rehearse import RehearsalResult

#: Well counts the extrapolation brackets, matching the shadow campaign's two board
#: layouts.
EXTRAPOLATION_WELLS = (16, 32)

#: The measured distribution, used **only** to project a duration before a run commits.
#:
#: These were the six-spectrum probe's estimates (0.25 / 45.0 / 0.30) until the
#: 2026-08-14 rehearsal replayed 192 real spectra and measured them; a measurement
#: retires an estimate, so the probe's numbers are gone rather than kept alongside.
#:
#: They live beside the prose above that quotes them so a later rehearsal moves one
#: place rather than two.  **A projection is still not a result** — a re-run reports
#: what it measured, never these.
PROBE_CLOSED_SECONDS = 0.16
PROBE_OPEN_SECONDS = 38.0
PROBE_OPEN_FRACTION = 0.39


def project_duration_seconds(n_spectra: int) -> float:
    """What *n_spectra* should cost at the last rehearsal's rates.

    An estimate, and labelled as one wherever it is printed: the open-arc fraction is a
    property of the material, so a corpus with a different mix will not match it.
    """
    per = (PROBE_OPEN_FRACTION * PROBE_OPEN_SECONDS
           + (1.0 - PROBE_OPEN_FRACTION) * PROBE_CLOSED_SECONDS)
    return per * max(0, int(n_spectra))

CSV_COLUMNS = ("spectrum_key", "measurement_id", "channel", "leg", "setpoint", "round",
               "seconds", "load_seconds", "verdict", "n_gates_failed", "arc_state",
               "sigma_mode", "cell_source", "warmup", "mode")


@dataclass(frozen=True)
class TimingRecord:
    """What one spectrum cost, and what it cost *that*."""

    spectrum_key: str
    measurement_id: int
    channel: int
    leg: str
    setpoint: int
    round: int
    seconds: float
    load_seconds: float
    verdict: str
    n_gates_failed: int
    arc_state: str
    sigma_mode: str
    cell_source: str
    warmup: bool
    mode: str = "observing"

    @property
    def block(self) -> str:
        return "unparsed" if self.leg == "?" else f"{self.leg}/S{self.setpoint}"

    def as_row(self) -> "list[Any]":
        return [getattr(self, name) for name in CSV_COLUMNS]

    @classmethod
    def of(cls, row: Any, report: Any, event: "dict[str, Any]", *, seconds: float,
           load_seconds: float, cell_source: str, warmup: bool,
           mode: str) -> "TimingRecord":
        """One record from the engine's answer and the corpus row that produced it.

        *event* is the tapped ``eis_spectrum_metrics`` payload and wins wherever it has
        an opinion, because it is the engine's own words; *report* is the fallback for a
        caller replaying without a tap.
        """
        return cls(
            spectrum_key=str(event.get("spectrum_key", "")),
            measurement_id=row.measurement_id, channel=row.channel, leg=row.leg,
            setpoint=row.setpoint, round=row.round, seconds=seconds,
            load_seconds=load_seconds,
            verdict=str(event.get("verdict", "")
                        or getattr(getattr(report, "quality", None), "verdict", "")),
            n_gates_failed=(len(event["gates_failed"])
                            if event.get("gates_failed") is not None
                            else _n_gates_failed(report)),
            arc_state=_arc_state(report),
            sigma_mode=str(getattr(getattr(report, "sigma", None), "mode", "")),
            cell_source=cell_source, warmup=warmup, mode=mode)


def _n_gates_failed(report: Any) -> int:
    """How many gates this spectrum failed, off the report's own gate log.

    The tap is the primary source; this is the fallback, because a record reporting
    zero failures because nobody was listening is worse than one that counts the log.
    """
    return sum(1 for entry in (getattr(report, "gate_log", ()) or ())
               if isinstance(entry, dict) and not entry.get("passed", True))


def _arc_state(report: Any) -> str:
    """The closure state the report already computed — never a second computation.

    Read off ``report.fit.arc_closure``, where
    :func:`~softae.analysis.eis.arc.annotate_arc_closure` attaches it. The ``gate_log``
    scan is a fallback for a report shape that carries it there instead; the gated
    engine does not, which is why the attribute is tried first.
    """
    arc = getattr(getattr(report, "fit", None), "arc_closure", None)
    state = getattr(arc, "state", None)
    if isinstance(state, str) and state:
        return state
    for entry in getattr(report, "gate_log", ()) or ():
        if isinstance(entry, dict) and entry.get("gate") == "arc_closure":
            return str(entry.get("state", "unknown"))
    return "unknown"


def timing_csv_path(log_path: Path) -> Path:
    """``logs/rehearsal_X.log`` → ``logs/rehearsal_X.timing.csv``, wherever ``--out`` points."""
    return log_path.with_suffix(".timing.csv")


# ── Log stream ───────────────────────────────────────────────────────────────

class MetricsTap:
    """A structlog processor that keeps the last ``eis_spectrum_metrics`` event.

    ``spectrum_key`` and the verdict are computed inside ``analyze_spectrum`` and
    published only through the log.  Recomputing them here would be a second
    implementation of the same fact, free to disagree; tapping the stream takes the
    engine's own answer and joins the CSV to the log by construction.
    """

    def __init__(self) -> None:
        self.last: dict[str, Any] | None = None

    def __call__(self, _logger: Any, _method: str, event_dict: dict[str, Any]):
        if event_dict.get("event") == "eis_spectrum_metrics":
            self.last = dict(event_dict)
        return event_dict

    def take(self) -> dict[str, Any]:
        out, self.last = self.last or {}, None
        return out


class _Tee:
    """Write to the log file and the console at once, for a watched run.

    A stream that refuses is skipped rather than raised through: ``--tee`` is a
    convenience, and losing the mirror must not cost the run it was mirroring.
    """

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def _each(self, method: str, *a: Any) -> None:
        for stream in self._streams:
            try:
                getattr(stream, method)(*a)
            except (ValueError, OSError, AttributeError):
                pass

    def write(self, text: str) -> int:
        self._each("write", text)
        return len(text)

    def flush(self) -> None:
        self._each("flush")


@dataclass
class RehearsalArtifacts:
    """The two files a running rehearsal writes, and the tap that reads one back.

    :meth:`write_row` appends **and flushes** one CSV row per completed spectrum rather
    than serialising the batch at the end.  A 192-spectrum pass runs for the better part
    of an hour and is the single most interruptible thing this tool does; a run killed at
    spectrum 140 must keep the 139 measurements it already paid for, and the module
    docstring's promise that a skip never costs the batch is worth nothing if a Ctrl-C
    costs all of it.  The flush is what makes the rows survive the signal, not merely the
    write.
    """

    tap: MetricsTap
    _writer: Any = None
    _handle: Any = None

    def take(self) -> "dict[str, Any]":
        """The engine's own last ``eis_spectrum_metrics`` event, consumed."""
        return self.tap.take()

    def write_row(self, record: TimingRecord) -> None:
        if self._writer is None:
            return
        self._writer.writerow(record.as_row())
        self._handle.flush()


@contextmanager
def open_log_stream(path: Path, *, tee: bool = False,
                    level: int = logging.INFO) -> "Iterator[RehearsalArtifacts]":
    """Open both artifacts — the log at *path* and the CSV beside it — and yield them.

    Owning the log handle rather than trusting a shell redirect buys two things: an
    operator who forgets the redirect loses the hour, and the encoding is *guaranteed*
    rather than inherited from a code page that cannot render ``tan δ``.  INFO rather
    than unconfigured, because unconfigured structlog renders every level and over 1440
    spectra the DEBUG traffic is the majority of the artifact.  The previous config is
    restored on exit so a rehearsal inside a longer process — a test session, most
    obviously — does not leave structlog writing to a closed file.

    The CSV header is written on entry, so an interrupted run still leaves a readable
    file rather than a zero-byte one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = structlog.get_config()
    tap = MetricsTap()
    handle = path.open("w", encoding="utf-8", errors="replace", newline="")
    csv_handle = timing_csv_path(path).open("w", encoding="utf-8", newline="")
    try:
        writer = csv.writer(csv_handle)
        writer.writerow(CSV_COLUMNS)
        csv_handle.flush()
        sink: Any = _Tee(handle, sys.stdout) if tee else handle
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
                tap,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(file=sink),
            cache_logger_on_first_use=False,
        )
        yield RehearsalArtifacts(tap, writer, csv_handle)
    finally:
        structlog.configure(**saved)
        handle.close()
        csv_handle.close()


# ── Aggregation ──────────────────────────────────────────────────────────────

def _percentile(values: "list[float]", frac: float) -> float:
    """Nearest-rank percentile; NaN on an empty bucket, never a raise or a zero."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1,
                       max(0, round(frac * (len(ordered) - 1))))]


def _median(values: "list[float]") -> float:
    return statistics.median(values) if values else float("nan")


@dataclass(frozen=True)
class Bucket:
    """One row of the cost table."""

    name: str
    n: int
    median: float
    p90: float
    maximum: float
    total: float

    @staticmethod
    def of(name: str, seconds: "list[float]") -> "Bucket":
        return Bucket(name, len(seconds), _median(seconds), _percentile(seconds, 0.90),
                      max(seconds) if seconds else float("nan"), float(sum(seconds)))


@dataclass
class TimingSummary:
    """The distribution, split the one way that explains it."""

    all: Bucket
    closed: Bucket
    open: Bucket
    unknown: Bucket
    blocks: "dict[str, tuple[int, float, float]]" = field(default_factory=dict)
    warmup_seconds: float = float("nan")
    n_records: int = 0

    @property
    def open_fraction(self) -> float:
        graded = self.closed.n + self.open.n
        return (self.open.n / graded) if graded else float("nan")

    def bracket(self, wells: int) -> tuple[float, float, float]:
        """``(floor, mix, ceiling)`` seconds of *analysis* for *wells* spectra.

        Three numbers rather than one because the mix-weighted middle is the weakest of
        them: it assumes the bench population's open-arc fraction matches this corpus's,
        and the corpus is PEO/LiCl under an RH ramp while the campaign casts something
        else.  All-closed and all-open are assumption-free and bracket whatever happens.
        """
        closed = self.closed.median if self.closed.n else self.all.median
        opened = self.open.median if self.open.n else self.all.median
        frac = self.open_fraction
        if frac != frac:                      # NaN — nothing graded
            frac = 0.0
        mix = frac * opened + (1.0 - frac) * closed
        return wells * closed, wells * mix, wells * opened


def summarize_timing(records: "list[TimingRecord]") -> TimingSummary:
    """Aggregate the records, **excluding the warm-up** from every statistic.

    The first analysis in a process pays ~2 s for the ``impedance`` import and the
    Lin-KK namespace patch.  Left in, that single record sets the maximum of an
    otherwise sub-second population and drags a small run's median with it; it is
    reported separately instead, because it is a real cost the operator pays once.
    """
    warm = next((r.seconds for r in records if r.warmup), float("nan"))
    body = [r for r in records if not r.warmup]

    def _sec(state: str) -> list[float]:
        return [r.seconds for r in body if r.arc_state == state]

    blocks: dict[str, tuple[int, float, float]] = {}
    for name in sorted({r.block for r in body}):
        members = [r for r in body if r.block == name]
        graded = [r for r in members if r.arc_state in ("closed", "open")]
        n_open = sum(1 for r in graded if r.arc_state == "open")
        blocks[name] = (
            len(members),
            (n_open / len(graded)) if graded else float("nan"),
            _median([r.seconds for r in members]),
        )

    return TimingSummary(
        all=Bucket.of("all", [r.seconds for r in body]),
        closed=Bucket.of("arc CLOSED", _sec("closed")),
        open=Bucket.of("arc OPEN", _sec("open")),
        unknown=Bucket.of("arc UNKNOWN", _sec("unknown")),
        blocks=blocks, warmup_seconds=warm, n_records=len(records),
    )


# ── Rendering ────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Seconds → ``0.31s`` / ``45m 12s`` / ``1h 07m``; NaN renders as a dash.

    Public because the replay half prints the *projected* duration from the same
    vocabulary before a run commits, and two spellings of an hour would read as two
    different quantities.
    """
    if seconds != seconds:
        return "  -  "
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


def _bucket_line(bucket: Bucket, *, total: bool = False) -> str:
    label = f"{bucket.name} (n={bucket.n})"
    cells = f"{format_duration(bucket.median):>9} {format_duration(bucket.p90):>9} {format_duration(bucket.maximum):>9}"
    return f"  {label:<26}{cells}" + (f" {format_duration(bucket.total):>9}" if total else "")


def render_summary(result: "RehearsalResult", summary: TimingSummary, log_path: Path,
                   project: Path) -> str:
    """The operator-facing report — the distribution, the mix, and what it does *not* claim.

    *result* is typed for readers only; nothing here imports it, which is what keeps the
    dependency one-way.  Every attribute read below is one the replay half already
    publishes.
    """
    plan = result.plan
    n_cells_txt = f"{result.n_cell_from_fit}/{len(result.records)}"
    # A partial run's numbers are real but its *population* is not the planned one — it
    # is a prefix of a round-major plan, so the block coverage is uneven. Say so in the
    # first line rather than letting the totals imply a completed pass.
    head = "REHEARSAL"
    if getattr(result, "interrupted", False):
        head = (f"REHEARSAL (PARTIAL - interrupted after {len(result.records)} of "
                f"{plan.n_selected}; block coverage is uneven)")
    out = [
        f"{head} - {plan.describe()}",
        f"  engine=gated  gates={result.mode}  model={result.model_name}  "
        f"cell=from fit row ({n_cells_txt})",
        "",
        f"  {'cost per spectrum':<26}{'median':>9} {'P90':>9} {'max':>9} {'total':>9}",
        _bucket_line(summary.all, total=True),
        _bucket_line(summary.closed),
        _bucket_line(summary.open),
        _bucket_line(summary.unknown),
    ]
    if summary.warmup_seconds == summary.warmup_seconds:
        out.append(f"  (first spectrum {format_duration(summary.warmup_seconds)} excluded as "
                   f"warm-up)")

    if summary.blocks:
        out += ["", "  by block"]
        for name, (n, frac, median) in summary.blocks.items():
            pct = "  -  " if frac != frac else f"{frac * 100:.0f}%"
            out.append(f"    {name:<14} n={n:<5} open-arc {pct:>5}   "
                       f"median {format_duration(median)}")

    out += ["",
            f"  skipped: {result.n_missing} missing, {result.n_unloadable} unloadable, "
            f"{result.n_raised} raised"]

    out += ["", "  EXTRAPOLATION - what a bench shadow run costs in analysis alone"]
    for wells in EXTRAPOLATION_WELLS:
        floor, mix, ceiling = summary.bracket(wells)
        out.append(f"    {wells:>2} wells:  all-closed {format_duration(floor):>9}   "
                   f"mix {format_duration(mix):>9}   all-open {format_duration(ceiling):>9}")
    frac = summary.open_fraction
    frac_txt = "unknown" if frac != frac else f"{frac * 100:.0f}%"
    out += [
        "    the mix column assumes the bench population's open-arc fraction matches",
        f"    this run's ({frac_txt}). It will not: these are PEO/LiCl films under an RH",
        "    ramp and the campaign casts something else. The floor and ceiling bracket",
        "    it; the mix estimate is the weakest of the three.",
        "",
        # Forward slashes on both: the line is meant to be copied into a shell, and a
        # Windows backslash inside it reads as an escape everywhere it might be pasted.
        f"  Next:  softae-shadow review {Path(log_path).as_posix()} "
        f"--project {Path(project).as_posix()}",
    ]
    return "\n".join(out)


__all__ = ["CSV_COLUMNS", "EXTRAPOLATION_WELLS", "Bucket", "MetricsTap",
           "RehearsalArtifacts", "TimingRecord", "TimingSummary", "format_duration",
           "open_log_stream", "render_summary", "summarize_timing", "timing_csv_path"]
