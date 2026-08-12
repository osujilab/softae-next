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

**Where the verdicts actually live, and what is lost.**  A shadow run's gate verdicts
are *not* persisted.  ``analysis/eis/router.py`` deliberately does not pass ``report=``
to ``record_fit`` (that is P.18), so every ``fit_results`` row a gated campaign writes
still carries ``engine='legacy'``, ``gate_verdict=NULL`` and ``gate_log_json='[]'``.
The only record of a would-reject verdict is the **structlog stream**, and in the
headless CLI structlog is unconfigured — a ``PrintLogger`` to stdout and nowhere else.
So the reviewable artifact of bench item 7 exists only if the operator redirects the
console, which is why the procedure makes ``| tee shadow_run.log`` a required step and
why this tool's primary input is a log file rather than the database.

The DataStore is still read, for what it can honestly supply: which run, how many
measurements per channel, and the stored σ.  It is asked nothing it would have to
invent — in particular ``fit_results.engine`` is reported as a *stamped default*, never
as evidence of which engine ran.

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

from softae.tools import use_utf8_console

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_A_SHADOW_RUN = 2

#: Events only the gated engine can emit.  Their presence is the proof the flip took;
#: their absence means the log is from a legacy run whatever the config says now.
GATED_ONLY_EVENTS = frozenset({
    "eis_gate_would_reject", "eis_gate_rejected", "eis_gate_points_dropped",
    "eis_gate_suspect", "eis_gate_raised", "eis_split_degenerate",
    "eis_correction_skipped", "eis_fit_not_admitted",
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

    @property
    def n_gated_events(self) -> int:
        return sum(self.events[e] for e in GATED_ONLY_EVENTS)

    @property
    def is_shadow_run(self) -> bool:
        return self.n_gated_events > 0


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

    _flush()
    return rv


def _num(value: Any) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if out == out else None


# ── DataStore side ───────────────────────────────────────────────────────────

def db_summary(project: str, run_id: str | None) -> dict[str, Any]:
    """Per-channel measurement/σ rows for one run.  Read-only.

    Returns ``{"error": …}`` rather than raising: a missing or unreadable project must
    not cost the operator the log half of the review, which is the half that carries
    the verdicts.
    """
    from softae.core.data_store import DataStore

    store = None
    try:
        store = DataStore(project)
        if run_id is None:
            runs = store.query_runs()
            if not runs:
                return {"error": f"no runs recorded in {project}"}
            run_id = str(runs[0]["run_id"])
        fits = {int(f["measurement_id"]): f for f in store.query_fits(run_id=run_id)}
        rows = []
        engines = Counter()
        for m in store.query_measurements(run_id=run_id):
            fit = fits.get(int(m["measurement_id"]), {})
            engines[str(fit.get("engine") or "-")] += 1
            rows.append({
                "channel": int(m["channel"]),
                "measurement_id": int(m["measurement_id"]),
                "sigma": fit.get("sigma_S_per_cm"),
                "R1": fit.get("R1"),
                "engine": fit.get("engine"),
                "gate_verdict": fit.get("gate_verdict"),
            })
        return {"run_id": run_id, "rows": rows, "engines": engines}
    except Exception as exc:  # noqa: BLE001 - a CLI boundary over an optional input
        return {"error": str(exc)}
    finally:
        if store is not None:
            store.close()


# ── Rendering ────────────────────────────────────────────────────────────────

def _table(header: "tuple[str, ...]", rows: "list[tuple[Any, ...]]") -> str:
    if not rows:
        return "   (none)"
    cols = [max(len(str(header[i])), *(len(str(r[i])) for r in rows))
            for i in range(len(header))]
    out = ["   " + "  ".join(str(h).ljust(cols[i]) for i, h in enumerate(header))]
    out.append("   " + "  ".join("-" * c for c in cols))
    for r in rows:
        out.append("   " + "  ".join(str(v).ljust(cols[i]) for i, v in enumerate(r)))
    return "\n".join(out)


def render(rv: ShadowReview, db: dict[str, Any] | None, source: str) -> str:
    """The operator-readable report."""
    out: list[str] = [f"SHADOW CAMPAIGN REVIEW — {source}",
                      f"   {rv.n_lines} line(s) → {rv.n_events} structured event(s)",
                      ""]

    out.append("1. DID THE GATED ENGINE RUN?")
    if rv.is_shadow_run:
        out.append(f"   YES — {rv.n_gated_events} gated-engine event(s), "
                   f"{rv.n_routed} spectrum(s) routed to the store.")
    else:
        out.append("   NO — not one gated-engine event in this log. Either the config "
                   "flip did not take,")
        out.append("   or this is a legacy run. Check `softae-shadow status` and rerun; "
                   "reviewing this")
        out.append("   log would arm a gate against evidence it never produced.")
    out.append("")

    out.append("2. WOULD-REJECT VERDICTS  ([eis.gates] enabled = false)")
    out.append(f"   spectra that WOULD have been discarded : {rv.would_reject}"
               + (f"  of {rv.n_routed} routed" if rv.n_routed else ""))
    out.append(f"   verdict lines logged                   : "
               f"{rv.would_reject_verdicts} — the engine reduces the gate log twice "
               f"per spectrum")
    out.append("                                             (pre-fit admission, then "
               "post-fit). The two are")
    out.append("                                             paired; the count above "
               "is spectra, not lines.")
    out.append(f"   [quality] would-reject                 : {rv.quality_would_reject}")
    gates = sorted(set(rv.gate_would_reject) | set(rv.gate_blocking_fail)
                   | set(rv.gate_points_dropped))
    out.append("")
    out.append(_table(
        ("gate", "would-reject", "blocking-fail", "points-dropped"),
        [(g, rv.gate_would_reject[g], rv.gate_blocking_fail[g],
          rv.gate_points_dropped[g]) for g in gates]))
    out.append("   'would-reject' counts SPECTRA; 'blocking-fail' counts GATE failures "
               "(a spectrum can fail")
    out.append("   several); 'points-dropped' are removed from the fit EVEN NOW — "
               "block_point masks are not")
    out.append("   behind the enabled flag.")
    if rv.other_issues:
        out.append("")
        out.append("   policy-level issues (no gate, so no threshold to calibrate):")
        for issue, n in rv.other_issues.most_common():
            out.append(f"     {n:>4}  {issue}")
    if rv.gates_raised:
        out.append(f"   ⚠ gates that RAISED and were skipped: {dict(rv.gates_raised)} "
                   "— those checks did not run.")
    out.append("")

    out.append("3. VALUE-VS-BOUND DEMOTIONS")
    total_bound = sum(rv.bound_modes.values())
    out.append(f"   σ declined as an upper bound: {total_bound} "
               f"{dict(rv.bound_modes) if rv.bound_modes else ''}")
    out.append("   Not governed by [eis.gates] enabled. In a σ-objective campaign each "
               "of these is an")
    out.append("   UNMEASURED trial; in this volume-mode spec the objective is mean|Z| "
               "and none of them")
    out.append("   costs the search anything.")
    pairs = [p for p in rv.sigma_shadow if p[1] is not None]
    out.append(f"   eis_objective_shadow pairs (mean|Z| in use, σ observed): "
               f"{len(rv.sigma_shadow)}, of which {len(pairs)} produced a σ")
    out.append("")

    out.append("4. CHANNEL ATTRIBUTION  (POSITIONAL — INFERRED, NOT RECORDED)")
    channels = sorted(set(rv.channel_would_reject) | set(rv.channel_bound)
                      | set(rv.routed_channels))
    out.append(_table(
        ("channel", "routed", "would-reject", "bound"),
        [(c, rv.routed_channels[c], rv.channel_would_reject[c], rv.channel_bound[c])
         for c in channels]))
    out.append(f"   unattributed (no preceding channel= line): {rv.unattributed}")
    out.append("   The would-reject event carries no channel. These come from the "
               "nearest preceding")
    out.append("   channel= line: sound for auto-fit gate events, UNSOUND for "
               "objective-side ones, which")
    out.append("   run after the round and land on whichever channel was routed last.")
    out.append("")

    out.append("5. PER-CHANNEL σ  (DataStore)")
    if db is None:
        out.append("   (not read — pass --project to include it)")
    elif "error" in db:
        out.append(f"   (unavailable: {db['error']})")
    else:
        out.append(f"   run_id {db['run_id']}")
        out.append(_table(
            ("channel", "meas", "R1 (Ω)", "σ (S/cm)", "engine col", "gate_verdict"),
            [(r["channel"], r["measurement_id"], _g(r["R1"]), _g(r["sigma"]),
              r["engine"] or "-", r["gate_verdict"] or "-") for r in db["rows"]]))
        out.append(f"   engine column: {dict(db['engines'])} — a STAMPED DEFAULT, not "
                   "an observation.")
        out.append("   The router does not pass report= to record_fit (P.18 open), so "
                   "a gated run still")
        out.append("   writes engine='legacy' and a NULL gate_verdict. Section 1 is the "
                   "engine evidence.")
    out.append("")

    out.append("6. ARM / DON'T-ARM DECISION INPUTS")
    out.append("   DEVELOPMENT_FRONTS asks two things of this review, and neither is a "
               "number this tool")
    out.append("   can decide for you:")
    out.append(f"     [eis.gates] enabled — needs 'a reviewed would-reject log'. "
               f"You have {rv.would_reject} would-reject")
    out.append("       verdict(s) over " + f"{rv.n_routed} spectrum(s). Read section 2 "
               "gate by gate and ask of each: would")
    out.append("       discarding those samples have been right? Every threshold "
               "shipped is an engineering")
    out.append("       default chosen without reference to this rig.")
    out.append(f"     [quality] enabled — needs threshold calibration. "
               f"{rv.quality_would_reject} would-reject verdict(s) here.")
    out.append("     E6 cutover criterion 2 — 'a campaign's worth of reviewed "
               "eis_gate_would_reject'. Met by")
    out.append("       reviewing this log; criterion 1 (a version-controlled "
               "calibration set) is separate and")
    out.append("       still blocked on the RE→CE jumper.")
    return "\n".join(out)


def _g(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4g}"


# ── Commands ─────────────────────────────────────────────────────────────────

def _cmd_status(args) -> int:
    """Report the flags the shadow procedure flips.  Reads; never writes."""
    from softae.analysis.eis.settings import eis_settings
    from softae.config import loader

    eis = eis_settings()
    quality = bool((loader.load().get("quality", {}) or {}).get("enabled", False))

    print(f"config: {loader.config_path()}")
    print(f"   {eis.describe()}")
    print(f"   [eis] engine          = {eis.engine!r}")
    print(f"   [eis] objective       = {eis.objective!r}")
    print(f"   [eis.gates] enabled   = {str(eis.gates.enabled).lower()}")
    print(f"   [quality] enabled     = {str(quality).lower()}")
    print(f"   [eis.fixture] mode    = {eis.fixture.mode!r} on "
          f"{eis.fixture.fixture_id!r}")
    print()

    if eis.engine == "gated" and not eis.gates.enabled and not quality:
        print("ARMED FOR A SHADOW RUN — gated physics, every check observing, "
              "nothing removed at the")
        print("spectrum level. Revert `engine` to \"legacy\" when the run is reviewed.")
        return EXIT_OK
    if eis.engine == "gated":
        print("GATED AND ENFORCING — this is a cutover, not a shadow run. A shadow run "
              "needs")
        print("[eis.gates] enabled = false and [quality] enabled = false.")
        return EXIT_FAILED
    print("NOT ARMED — the shipped legacy engine. Set [eis] engine = \"gated\" in the "
          "config above")
    print("to run a shadow campaign (docs/SHADOW_CAMPAIGN.md).")
    return EXIT_NOT_A_SHADOW_RUN


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
    db = db_summary(args.project, args.run_id) if args.project else None
    print(render(review, db, source))
    return EXIT_OK if review.is_shadow_run else EXIT_NOT_A_SHADOW_RUN


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
    rev.set_defaults(func=_cmd_review)
    return p


def main(argv: "list[str] | None" = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
