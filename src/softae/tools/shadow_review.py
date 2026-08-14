"""``softae-shadow`` — arm the rig for a shadow campaign, then review what it saw.

Bench queue item 7 (``docs/DEVELOPMENT_FRONTS.md``) is *"one shadow campaign with
``engine = "gated"``, gates observing"*, and it is the cheapest unblock in the project:
it arms both data-quality gates and the E6 cutover at once.  Two subcommands, one on
each side of the run::

    softae-shadow status                      # is the config armed for a shadow run?
    softae-shadow review shadow_run.log --project <dir>

``status`` is read-only and answers the question twice — before the run ("did the flip
take?") and after the revert ("is the rig back?").  The whole procedure is
``docs/SHADOW_CAMPAIGN.md``.

**Where the verdicts actually live, and what is lost.**  A shadow run's gate *verdicts*
are still not persisted.  ``analysis/eis/router.py`` does not pass the real
:class:`~softae.analysis.eis.report.SpectrumReport` to ``record_fit`` — that is P.18,
and it remains open — so every ``fit_results`` row a gated campaign writes still carries
``engine='legacy'``, a NULL ``gate_verdict`` and ``sigma_is_bound = 0``.  The only
record of a would-reject verdict is the **structlog stream**, and in the headless CLI
structlog is unconfigured — a ``PrintLogger`` to stdout and nowhere else.  So the
reviewable artifact of bench item 7 exists only if the operator redirects the console,
which is why the procedure makes ``| tee shadow_run.log`` a required step and why this
tool's primary input is a log file rather than the database.

``arc_state`` and its three companions are the **one exception**, and they are real
columns rather than a JSON payload.  ``record_fit`` writes them from
``fit_result.arc_closure`` — the annotation
:func:`~softae.analysis.eis.arc.annotate_arc_closure` attaches on every
``analyze_spectrum`` that produces a fit, on **both** engines — so a routed row carries
the arc verdict whether or not any report is passed.  ``gate_log_json`` is therefore
empty again on rows written since the ``arc_provenance`` shim was retired; the
T7.1-era rows that carry the record as a JSON entry are still read, which is why
:func:`softae.tools.shadow_db.arc_summary` distinguishes three eras of row.  It remains
honest evidence — the arc states are observations, not stamped defaults — and the rest
of P.18 is exactly where it was.

The DataStore is otherwise read for what it can honestly supply: which run, how many
measurements per channel, the stored σ, and which fits railed on the model's own R₁
bound (:func:`~softae.tools.shadow_db.railed_summary`).  It is asked nothing it would
have to invent — in particular ``fit_results.engine`` and ``sigma_is_bound`` are
reported as *stamped defaults*, never as evidence of which engine ran or of what it
concluded.

Two attribution limits, stated rather than smoothed:

* ``eis_gate_would_reject`` carries **no channel, run_id or sample_uuid** — only
  ``issues`` and ``metrics``.  Per-*gate* counting is exact, because the gate name is
  the head of each issue string and ``eis_gate_rejected`` carries ``gate=`` outright.
  Per-*channel* counting can only be positional.
* Positional attribution is sound for gate events emitted during the workflow's
  auto-fit, which immediately follow the router's ``eis_autorouted`` line, and
  **unsound** for events emitted during objective extraction, which happens after the
  round's measurements and so lands on whichever channel was routed last.  Rows carry
  which anchor they came from and the summary counts the unattributed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from softae.analysis.eis.recommend import (
    DEFAULT_MIN_EVIDENCE,
    Recommendation,
    SpectrumRecord,
    deduplicate,
    recommend_all,
)
from softae.analysis.eis.recommend_report import as_toml_block
from softae.tools import use_utf8_console
from softae.tools.shadow_db import arc_summary, db_summary, railed_summary
from softae.tools.shadow_render import render, render_status

#: ``db_summary`` moved to :mod:`softae.tools.shadow_db` and ``render`` to
#: :mod:`softae.tools.shadow_render` when this module passed the house line limit.
#: Both are re-exported here because this module was their public surface first, and a
#: pure move must not break a caller.
__all__ = ["arc_summary", "db_summary", "railed_summary", "main", "parse_line",
           "render", "render_status", "summarize", "build_parser", "ShadowReview"]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_A_SHADOW_RUN = 2

#: Events only the gated engine can emit.  Their presence is the proof the flip took;
#: their absence means the log is from a legacy run whatever the config says now.
GATED_ONLY_EVENTS = frozenset({
    "eis_gate_would_reject", "eis_gate_rejected", "eis_gate_points_dropped",
    "eis_gate_suspect", "eis_gate_raised", "eis_split_degenerate",
    "eis_correction_skipped", "eis_fit_not_admitted", "eis_spectrum_metrics",
    "objective_declined_bound", "objective_rejected_by_gates",
})

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
#: ``2026-08-10 14:44:43 [warning  ] eis_gate_would_reject   issues=[...] ...``
_CONSOLE = re.compile(
    r"^\s*(?:\S+(?:[ T]\S+)?\s+)?\[(?P<level>[a-z]+)\s*\]\s+(?P<event>\S+)\s*(?P<rest>.*)$"
)
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def _split_kv(rest: str) -> dict[str, Any]:
    """``"a=1 b=['x: y'] c=z"`` → ``{"a": 1, "b": ["x: y"], "c": "z"}``.

    Quote- and bracket-aware, because ``detail=`` and ``msg=`` values routinely contain
    both spaces and ``=``.  A naive ``split()`` on whitespace shreds them, and a naive
    regex for ``\\w+=`` finds keys inside quoted prose.
    """
    bounds: list[tuple[int, str]] = []
    depth = 0
    quote: str | None = None
    i = 0
    while i < len(rest):
        ch = rest[i]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth = max(0, depth - 1)
        elif depth == 0 and (i == 0 or rest[i - 1].isspace()):
            m = _KEY.match(rest, i)
            if m:
                bounds.append((i, m.group()[:-1]))
                i = m.end()
                continue
        i += 1

    out: dict[str, Any] = {}
    for n, (start, key) in enumerate(bounds):
        end = bounds[n + 1][0] if n + 1 < len(bounds) else len(rest)
        raw = rest[start + len(key) + 1:end].strip()
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def parse_line(line: str) -> dict[str, Any] | None:
    """One log line → ``{"event": …, **fields}``, or ``None`` if it is not one.

    Accepts both renderings structlog can produce: JSON (if the operator configured a
    ``JSONRenderer``) and the default console form.  Anything else — the campaign CLI's
    own ``print`` output, traceback bodies, blank lines — returns ``None`` rather than
    being coerced into a half-parsed event.
    """
    text = _ANSI.sub("", line).rstrip("\n")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and "event" in obj:
            return dict(obj)
    m = _CONSOLE.match(text)
    if not m:
        return None
    out: dict[str, Any] = {"event": m.group("event"), "level": m.group("level")}
    out.update(_split_kv(m.group("rest")))
    return out


#: ``policy.reduce_gates`` emits three issue shapes.  Two name a gate; the third is a
#: policy-level statement about the spectrum as a whole.
_DROPPED = re.compile(r"^(\S+) dropped \d+ points?$")


def _gate_names(fields: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``issues`` → ``(gate names, policy-level issues)``.

    ``"<gate>: <detail>"`` and ``"<gate> dropped N points"`` name a gate; anything else
    — ``"only 5 points survived gating (need 8)"``, ``"resolution-limited — σ reported
    as an upper bound"``, ``"gates observing only"``, and the fit-grade and
    fixture-correction issues appended after the reduction — does not, and putting it
    in the gate column would invent a gate that has no threshold to calibrate.
    """
    issues = fields.get("issues")
    if not isinstance(issues, (list, tuple)):
        return [], []
    names: list[str] = []
    other: list[str] = []
    for issue in issues:
        text = str(issue).strip()
        head = text.split(":", 1)[0].strip()
        dropped = _DROPPED.match(text)
        if ":" in text and head and " " not in head:
            names.append(head)
        elif dropped:
            names.append(dropped.group(1))
        else:
            other.append(text)
    return names, other


@dataclass
class ShadowReview:
    """Everything the log established, counted."""

    n_lines: int = 0
    n_events: int = 0
    events: Counter = field(default_factory=Counter)
    would_reject: int = 0          # post-fit verdicts = distinct spectra
    would_reject_verdicts: int = 0  # every logged line, pre-fit and post-fit
    quality_would_reject: int = 0
    gate_would_reject: Counter = field(default_factory=Counter)
    other_issues: Counter = field(default_factory=Counter)
    gate_blocking_fail: Counter = field(default_factory=Counter)
    gate_points_dropped: Counter = field(default_factory=Counter)
    gates_raised: Counter = field(default_factory=Counter)
    bound_modes: Counter = field(default_factory=Counter)
    channel_would_reject: Counter = field(default_factory=Counter)
    channel_bound: Counter = field(default_factory=Counter)
    unattributed: int = 0
    sigma_shadow: list[tuple[float | None, float | None]] = field(default_factory=list)
    n_routed: int = 0
    routed_channels: Counter = field(default_factory=Counter)
    #: Every ``eis_spectrum_metrics`` event, before deduplication.
    metric_events: list = field(default_factory=list)

    @property
    def n_gated_events(self) -> int:
        return sum(self.events[e] for e in GATED_ONLY_EVENTS)

    @property
    def is_shadow_run(self) -> bool:
        return self.n_gated_events > 0

    @property
    def spectra(self) -> "list[SpectrumRecord]":
        """The metric events as *spectra* — one record per physical measurement.

        ``n_events`` and ``len(spectra)`` are both reported so the 2× that repeat
        analysis produces is visible rather than silently collapsed.
        """
        return deduplicate(self.metric_events)

    @property
    def n_spectra_seen(self) -> int:
        """How many spectra this log is evidence about — router count, or the metrics.

        ``n_routed`` counts ``eis_autorouted`` lines, which a **campaign** emits and a
        **rehearsal** never does: a rehearsal replays spectra already on disk and routes
        nothing. Reading the run size from the router alone therefore reported ``0
        spectrum(s)`` for a rehearsal carrying hundreds of metrics events — the one
        number an operator sizes the review by, wrong in the direction that says "there
        is nothing here".

        The router count stays authoritative wherever it exists, because it anchors
        spectra to channels and the metrics events do not. The fallback is used only
        when there are no anchors at all, and the renderer says so where it is used.
        """
        return self.n_routed or len(self.spectra)


def summarize(lines: "list[str]") -> ShadowReview:
    """Aggregate a run's console log.  Pure — no file or database access.

    **One rejected spectrum logs ``eis_gate_would_reject`` twice.**
    ``analyze_spectrum`` reduces the gate log once before the fit (the admission
    verdict) and once after (with the Front-2 gates appended), and with
    ``enabled = false`` neither call short-circuits, so both fire.  Counting lines
    would double every number in this report.

    The two are paired by the mechanism that produces them: the gate results
    *accumulate*, so the post-fit line's gate set is a **superset** of the admission
    line's.  A verdict whose predecessor is not a subset — or one separated from it by
    a fresh ``eis_autorouted`` — is a spectrum of its own.  An unpaired verdict counts
    as one spectrum rather than being dropped.
    """
    rv = ShadowReview()
    channel: int | None = None
    pending: tuple[list[str], list[str], int | None] | None = None

    def _record(entry: tuple[list[str], list[str], int | None]) -> None:
        names, other, chan = entry
        rv.would_reject += 1
        for name in names:
            rv.gate_would_reject[name] += 1
        for issue in other:
            rv.other_issues[issue] += 1
        if chan is None:
            rv.unattributed += 1
        else:
            rv.channel_would_reject[chan] += 1

    def _flush() -> None:
        nonlocal pending
        if pending is not None:
            _record(pending)
            pending = None

    for line in lines:
        rv.n_lines += 1
        fields = parse_line(line)
        if fields is None:
            continue
        rv.n_events += 1
        event = str(fields.get("event", ""))
        rv.events[event] += 1

        # Positional anchor.  Any event carrying a channel re-anchors attribution; see
        # the module docstring for exactly how far that can be trusted.
        raw_channel = fields.get("channel")
        if isinstance(raw_channel, int):
            channel = raw_channel
        elif isinstance(raw_channel, str) and raw_channel.lstrip("-").isdigit():
            channel = int(raw_channel)

        if event == "eis_autorouted":
            _flush()
            rv.n_routed += 1
            if channel is not None:
                rv.routed_channels[channel] += 1
        elif event == "eis_gate_would_reject":
            rv.would_reject_verdicts += 1
            names, other = _gate_names(fields)
            if pending is not None and set(pending[0]) <= set(names):
                pending = None
                _record((names, other, channel))
            else:
                _flush()
                pending = (names, other, channel)
        elif event == "quality_gate_would_reject":
            rv.quality_would_reject += 1
        elif event == "eis_gate_rejected":
            rv.gate_blocking_fail[str(fields.get("gate", "?"))] += 1
        elif event == "eis_gate_points_dropped":
            try:
                n = int(fields.get("n", 0) or 0)
            except (TypeError, ValueError):
                n = 0
            rv.gate_points_dropped[str(fields.get("gate", "?"))] += n
        elif event == "eis_gate_raised":
            rv.gates_raised[str(fields.get("gate", "?"))] += 1
        elif event == "objective_declined_bound":
            rv.bound_modes[str(fields.get("mode", "?"))] += 1
            if channel is None:
                rv.unattributed += 1
            else:
                rv.channel_bound[channel] += 1
        elif event == "eis_objective_shadow":
            rv.sigma_shadow.append(
                (_num(fields.get("mean_abs_z")), _num(fields.get("sigma"))))
        elif event == "eis_spectrum_metrics":
            record = SpectrumRecord.from_event(fields)
            if record is not None:
                rv.metric_events.append(record)

    _flush()
    return rv


def _num(value: Any) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def recommendations(rv: ShadowReview,
                    min_evidence: int = DEFAULT_MIN_EVIDENCE
                    ) -> "list[Recommendation]":
    """The proposed thresholds this log supports.  Deduplicated first, always."""
    return recommend_all(rv.spectra, min_evidence=int(min_evidence))

# ── Commands ─────────────────────────────────────────────────────────────────

#: What each armed-state the renderer identifies means to a shell.  The exit code is a
#: CLI policy, so it lives here rather than travelling with the text that explains it.
_STATUS_EXIT = {"armed": EXIT_OK, "enforcing": EXIT_FAILED,
                "not_armed": EXIT_NOT_A_SHADOW_RUN}


def _cmd_status(args) -> int:
    """Report the flags the shadow procedure flips.  Reads; never writes."""
    from softae.analysis.eis.settings import eis_settings
    from softae.config import loader

    eis = eis_settings()
    quality = bool((loader.load().get("quality", {}) or {}).get("enabled", False))
    text, state = render_status(eis, quality, config_path=str(loader.config_path()))
    print(text)
    return _STATUS_EXIT[state]


def _cmd_review(args) -> int:
    path = Path(args.log)
    if str(args.log) == "-":
        lines = sys.stdin.read().splitlines()
        source = "<stdin>"
    elif not path.is_file():
        print(f"No such log file: {path}", file=sys.stderr)
        return EXIT_FAILED
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        source = str(path)

    review = summarize(lines)
    # All three DataStore reads are gated on --project and each degrades to an
    # ``{"error": …}`` of its own, so a project that answers one question and not
    # another reports exactly that rather than costing the whole database half.
    db = railed = arc = None
    if args.project:
        db = db_summary(args.project, args.run_id)
        railed = railed_summary(args.project, args.run_id)
        arc = arc_summary(args.project, args.run_id)
    recs = recommendations(review, getattr(args, "min_evidence", DEFAULT_MIN_EVIDENCE))
    print(render(review, db, source, recs, railed=railed, arc=arc))

    emit = getattr(args, "emit_toml", None)
    if emit and _write_toml(Path(emit), recs, review, source) != EXIT_OK:
        return EXIT_FAILED
    return EXIT_OK if review.is_shadow_run else EXIT_NOT_A_SHADOW_RUN


def _write_toml(path: Path, recs: "list[Recommendation]", rv: ShadowReview,
                source: str) -> int:
    """Write the paste-ready block, or refuse and say why.

    Two refusals, both absolute. This tool **never edits the live config** — arming is
    a decision taken by reading the would-reject table, not by running a command that
    happens to write a file — and it **never clobbers**, because the one file an
    operator would point it at twice is the one holding the previous run's proposal.
    """
    from datetime import datetime, timezone

    from softae.config import loader

    try:
        live = Path(loader.config_path()).resolve()
    except Exception:  # noqa: BLE001 - an unreadable config must not mask the refusal
        live = None
    target = path.expanduser()
    if live is not None and target.resolve(strict=False) == live:
        print(f"Refusing to write to the live config at {live}. This tool proposes "
              f"values; arming them is yours.", file=sys.stderr)
        return EXIT_FAILED
    if target.exists():
        print(f"Refusing to overwrite {target}. Name a new file.", file=sys.stderr)
        return EXIT_FAILED

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target.write_text(
        as_toml_block(recs, source=source, n_spectra=len(rv.spectra), when=when),
        encoding="utf-8")
    print(f"\nWrote {target} — paste-ready, nothing armed.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softae-shadow",
        description="Arm and review a shadow campaign (engine=gated, gates observing).",
        epilog="Procedure: docs/SHADOW_CAMPAIGN.md — bench queue item 7.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="is the config armed for a shadow run?")
    st.set_defaults(func=_cmd_status)

    rev = sub.add_parser("review", help="summarize a shadow run's console log")
    rev.add_argument("log", help="the redirected run log, or '-' for stdin")
    rev.add_argument("--project", help="project directory, to add the DataStore half")
    rev.add_argument("--run-id", dest="run_id",
                     help="run to read (default: the most recent in the project)")
    rev.add_argument("--emit-toml", dest="emit_toml", metavar="PATH",
                     help="write the paste-ready threshold block to a NEW file; "
                          "never the live config, never an existing path")
    rev.add_argument("--min-evidence", dest="min_evidence", type=int,
                     default=DEFAULT_MIN_EVIDENCE, metavar="N",
                     help=f"spectra required before a key may be recommended "
                          f"(default {DEFAULT_MIN_EVIDENCE})")
    rev.set_defaults(func=_cmd_review)

    reh = sub.add_parser(
        "rehearse",
        help="replay the gated engine over spectra already on disk (T7.8)",
        description="Answer what observe-only gating costs per spectrum, and what the "
                    "recommender says about a real population, before the single-shot "
                    "bench run spends the board. Reads a run's spectra through a "
                    "read-only sqlite connection; writes nothing but its own log.",
    )
    from softae.tools.shadow_rehearse import add_rehearse_arguments

    add_rehearse_arguments(reh)
    reh.set_defaults(func=_cmd_rehearse)
    return p


def _cmd_rehearse(args) -> int:
    """Dispatch to the replay tool.

    This import is **not** what keeps ``softae-shadow status`` fast, and an earlier
    version of this comment claimed it was. :func:`build_parser` imports
    ``add_rehearse_arguments`` unconditionally, so :mod:`softae.tools.shadow_rehearse`
    is loaded on every invocation whatever the subcommand. What actually costs nothing
    is that module's *import surface*: stdlib plus structlog, with numpy, ``impedance``
    and the fitter imported inside ``run_rehearsal`` at the moment a spectrum is
    analysed. The wrapper stays because it reads as the dispatch it is.
    """
    from softae.tools.shadow_rehearse import cmd_rehearse

    return cmd_rehearse(args)


def main(argv: "list[str] | None" = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
