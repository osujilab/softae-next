"""The Final-Check digest — what the campaign intends, beside what the rig holds.

A launch reads one file and one bench. The file says which stocks the solve
needs, which wells it will cast, what each phase commands; the bench holds
whatever syringes were last loaded and whatever board was last cast on. Nothing
puts those two side by side, so the commonest launch mistakes — a spec assigning
silica to the pump that carries LiCl, a channel list aimed at occupied wells, a
plan with no ``run_plan`` and therefore no anneal and no conditions at all — are
discovered at the bench rather than at the prompt.

This module renders both halves as one text page and names every place they
disagree. It is **pure read**: config, the shipped catalogs, the DataStore's
recorded loadout / occupancy / reservoir levels, and the spec. It opens no
instrument session, commands nothing, and writes nothing back.

**Three severities, and the middle one is the point.** ``ok`` is a fact checked
and agreed; ``warn`` is something to read before answering; ``block`` is the file
and the bench contradicting each other, which no answer at the prompt resolves
(see :func:`confirm_final_check`). What it must never do is spell *unknown* with
the same token as *checked and clean*: built without a project directory, it
reports the board as **unchecked**, not free.

Every line is assembled from a describer that already exists — ``RunPhase.label``,
``PhaseSetpoints``, ``project_campaign``, ``describe_or_absent`` — rather than
re-derived, so the digest cannot come to say something different from the surface
that owns it.

Run it directly to look at the layout::

    py -m softae.core.final_check examples/bench_instance.toml --project <dir>
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

# The projection's duration vocabulary, reused rather than re-invented: a digest
# printing "4.0 h" beside a projection printing "14400 s" would read as two
# different quantities.
from softae.core.preflight import _fmt_duration as _dur

__all__ = [
    "OK", "WARN", "BLOCK",
    "Finding", "Section", "FinalCheck",
    "build_final_check", "render_final_check", "confirm_final_check", "main",
]

OK = "ok"
WARN = "warn"
BLOCK = "block"

_RANK = {BLOCK: 0, WARN: 1, OK: 2}
_TAG = {BLOCK: "BLOCK", WARN: "warn ", OK: "ok   "}


@dataclass(frozen=True)
class Finding:
    """One thing the operator should read before answering the prompt."""

    severity: str
    text: str


@dataclass(frozen=True)
class Section:
    """A titled block of ``label: value`` rows, plus what they imply."""

    title: str
    rows: tuple[tuple[str, str], ...] = ()
    findings: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class FinalCheck:
    """The whole digest: sections in reading order, findings collected."""

    title: str
    sections: tuple[Section, ...]

    @property
    def findings(self) -> tuple[Finding, ...]:
        """Every finding, blocks first, then warnings, then the agreements.

        Byte-identical findings collapse to one: two sections legitimately reach
        the same advisory from different directions, and printing one sentence
        twice teaches the reader to skim the list.
        """
        unique: dict[tuple[str, str], Finding] = {}
        for section in self.sections:
            for finding in section.findings:
                unique.setdefault((finding.severity, finding.text), finding)
        return tuple(sorted(unique.values(),
                            key=lambda f: _RANK.get(f.severity, 9)))

    @property
    def n_block(self) -> int:
        return sum(1 for f in self.findings if f.severity == BLOCK)

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.severity == WARN)

    @property
    def has_block(self) -> bool:
        return self.n_block > 0


# ── Small shared helpers ─────────────────────────────────────────────────────

def _try(call: Callable[[], Any], default: Any = None) -> Any:
    """Run *call*, or answer *default*. A digest never fails a launch itself."""
    try:
        return call()
    except Exception:
        return default


def _conditions_line(conditions: Any, *, waits: bool = True) -> str:
    """``'casting: T 25 °C ±2 (≤30 min) · RH 22 % ±2 (≤4.0 h)'``.

    *waits* is false for the campaign-level baseline, whose bands and timeouts
    have no reader — :meth:`PhaseSetpoints.command_steps` emits no wait — so
    printing them there would show a gate that does not exist.
    """
    if conditions is None:
        return "inherits the standing setpoints"
    bits = []
    if conditions.temp_setpoint_C is None:
        bits.append("T not driven")
    elif waits:
        bits.append(f"T {float(conditions.temp_setpoint_C):g} °C "
                    f"±{float(conditions.tolerance_C):g} "
                    f"(≤{_dur(conditions.approach_timeout_s)})")
    else:
        bits.append(f"T {float(conditions.temp_setpoint_C):g} °C")
    if conditions.rh_setpoint_pct is None:
        bits.append("RH not driven")
    elif waits:
        bits.append(f"RH {float(conditions.rh_setpoint_pct):g} % "
                    f"±{float(conditions.rh_tolerance_pct):g} "
                    f"(≤{_dur(conditions.rh_approach_timeout_s)})")
    else:
        bits.append(f"RH {float(conditions.rh_setpoint_pct):g} %")
    return f"{conditions.name}: " + " · ".join(bits)


# ── 1. Campaign ──────────────────────────────────────────────────────────────

def _campaign_section(spec: Any, source: str | None) -> Section:
    import softae
    from softae.config import loader

    measurement = getattr(spec, "measurement", None)
    digest = _try(lambda: loader.config_hash())
    rows = [
        ("name", str(getattr(spec, "name", "?"))),
        ("spec file", source or "(built in memory)"),
        ("optimizer", f"{getattr(spec, 'optimizer', '?')}, budget "
                      f"{getattr(spec, 'budget', '?')}, "
                      f"{'batched' if getattr(spec, 'batch', False) else 'sequential'}"),
        ("objective", str(getattr(spec, "objective", "?"))),
        ("channels", ", ".join(str(c) for c in getattr(spec, "channels", ()) or ())),
        ("measurement", f"{getattr(measurement, 'modality', '?')} / "
                        f"{getattr(measurement, 'preset', '?')}"
                        + ("" if getattr(measurement, "enabled", True)
                           else "  (DISABLED — casts but does not measure)")),
        ("config", f"{digest[:12] if digest else 'unreadable'}  ·  softae "
                   f"{softae.__version__}"),
    ]
    findings = []
    if measurement is not None and not measurement.enabled:
        findings.append(Finding(
            WARN, "Measurement is disabled: this campaign casts and cures but "
                  "records no spectra, so the optimizer is fed nothing."))
    return Section("Campaign", tuple(rows), tuple(findings))


# ── 2. Stock on the pumps ────────────────────────────────────────────────────

def _stock_row(loaded: str | None, wanted: str | None,
               remaining: float | None, particulate: bool) -> str:
    left = loaded or "NOT DECLARED"
    if particulate:
        left = f"{left} [particulate]"
    level = "unknown" if remaining is None else f"{remaining:,.0f} µL"
    if wanted is None:
        return f"{left}  ·  {level}  ·  not used by this spec"
    if loaded is None:
        return f"{left}  ·  {level}  ·  spec wants '{wanted}'"
    if loaded != wanted:
        return f"{left}  ·  {level}  ·  SPEC WANTS '{wanted}'"
    return f"{left}  ·  {level}  ·  matches the spec"


def _stock_section(spec: Any, data_store: Any, ledger: Any,
                   chem_catalog: Any, sol_catalog: Any) -> Section:
    """The spec's intent beside the bench's declaration, per pump.

    The comparison itself is :func:`~softae.core.stock_resolution.resolve_stocks`,
    which ``provenance.json`` also renders from — so the page the operator reads
    and the file the run records cannot come to disagree about which stock the
    campaign thinks is on which line.
    """
    from softae.core.stock_assignment import load_loadout, solution_is_particulate
    from softae.core.stock_resolution import resolve_stocks

    loadout = _try(lambda: load_loadout(data_store))
    resolved = resolve_stocks(spec, loadout, sol_catalog)
    wanted_by_pump = resolved.spec_by_pump
    if not wanted_by_pump:
        return Section(
            "Stock on the pumps",
            (("formulation", "the spec declares no [general_formulation], so no "
                             "stock is assigned to any pump"),),
            (Finding(WARN, "No formulation block: nothing here can check that the "
                           "syringes on the bench are the ones the run needs."),))

    declared = resolved.declared_by_pump
    rows: list[tuple[str, str]] = []
    findings: list[Finding] = []

    for pid in sorted(set(wanted_by_pump) | set(declared)
                      | set(int(p) for p in getattr(spec, "pump_ids", ()) or ())):
        loaded, wanted = declared.get(pid), wanted_by_pump.get(pid)
        remaining = _try(lambda p=pid: ledger.remaining_uL(p)) if ledger else None
        solution = _try(lambda n=loaded: sol_catalog.get(n)) if loaded else None
        particulate = bool(_try(
            lambda s=solution: solution_is_particulate(s, chem_catalog), False))
        rows.append((f"pump {pid}",
                     _stock_row(loaded, wanted, remaining, particulate)))

        if wanted is not None and loaded is None:
            findings.append(Finding(
                WARN, f"Pump {pid} carries '{wanted}' in the spec, but nothing is "
                      f"declared loaded on it. Declare the loadout, or the run "
                      f"dispenses whatever is physically in that syringe."))
        elif wanted is not None and loaded != wanted:
            findings.append(Finding(
                BLOCK, f"Pump {pid}: intent and bench disagree — the spec assigns "
                       f"'{wanted}' and the bench record says '{loaded}' is loaded."))
        elif wanted is not None:
            findings.append(Finding(
                OK, f"Pump {pid} is loaded with '{loaded}', which is what the "
                    f"spec assigns to it."))
        elif loaded is not None:
            findings.append(Finding(
                OK, f"Pump {pid} carries '{loaded}', which this spec does not use."))
    return Section("Stock on the pumps", tuple(rows), tuple(findings))


# ── 3. Sequence ──────────────────────────────────────────────────────────────

#: Said in full because it is a real trap today: a spec with no ``run_plan``
#: silently runs the deposition engine's legacy pointwise layout.
_NO_RUN_PLAN = (
    "No [[run_plan.phases]]: this campaign runs the engine's legacy pointwise "
    "layout — deposit then measure, per channel, with NO anneal, NO equilibrate "
    "hold and NO commanded temperature or humidity at any point. Every "
    "measurement is taken on a film that is still drying, at whatever the "
    "chamber happens to be holding."
)


def _sequence_section(spec: Any, task_catalog: Any) -> Section:
    # The unconditioned-start advisory is preflight's, not ours, and is taken
    # from it verbatim: a second wording of the same condition is a second
    # opinion. Taken here as well as inside the projection so the digest still
    # says it when no task catalog was supplied and no projection ran.
    from softae.core.preflight import unconditioned_start_warnings

    run_plan = getattr(spec, "run_plan", None)
    baseline = getattr(spec, "conditions", None)
    rows: list[tuple[str, str]] = [
        ("baseline", _conditions_line(baseline, waits=False)
         if baseline is not None
         else "none — the chamber holds whatever it was last told to hold")]
    if run_plan is None:
        return Section("Sequence", tuple(rows), (Finding(WARN, _NO_RUN_PLAN),))

    for i, phase in enumerate(run_plan.phases, start=1):
        rows.append((f"{i}. {phase.kind.value}",
                     _try(lambda p=phase: p.label(task_catalog), "?")))
        if phase.conditions is not None:
            rows.append(("", _conditions_line(phase.conditions)))
    findings = tuple(Finding(WARN, w)
                     for w in _try(lambda: unconditioned_start_warnings(spec), []))
    return Section("Sequence", tuple(rows), findings)


# ── 4. Board ─────────────────────────────────────────────────────────────────

def _board_section(spec: Any, data_store: Any) -> Section:
    channels = sorted(int(c) for c in getattr(spec, "channels", ()) or ())
    capacity = getattr(spec, "electrode_capacity", None)
    # `_prepare_electrode_allocator` is only reached when `electrode_capacity` is
    # set; without one the run casts onto `spec.channels` verbatim and no
    # board-freshness question is ever asked. So the SAME overlap is a skipped
    # well under an allocator and a re-cast onto a used well without one.
    allocates = capacity is not None

    if data_store is None:
        return Section(
            "Board",
            (("occupancy", "NOT CHECKED — no project directory supplied"),
             ("spec targets", ", ".join(str(c) for c in channels) or "(none)")),
            (Finding(WARN, "Board occupancy was not read (no project directory), "
                           "so nothing here says whether the wells this campaign "
                           "targets have already been cast on."),))

    # The lambda is load-bearing: `_try(data_store.current_board_id)` resolves
    # the attribute *before* the guard runs, so a store surface lacking the
    # method takes the whole digest down instead of costing one row.
    board_id = _try(lambda: data_store.current_board_id())
    occupied = sorted(_try(lambda: data_store.occupied_electrodes(board_id), set()))
    overlap = sorted(set(occupied) & set(channels))
    rows = [
        ("board id", str(board_id)),
        ("occupied wells", ", ".join(str(e) for e in occupied) or "none recorded"),
        ("spec targets", ", ".join(str(c) for c in channels) or "(none)"),
        ("allocation", f"sequential {getattr(spec, 'electrode_start', 1)}..{capacity} "
                       f"(skips occupied wells)" if allocates
         else "fixed channel list (no allocator; occupied wells are NOT skipped)"),
    ]
    findings: list[Finding] = []
    if overlap and allocates:
        findings.append(Finding(
            WARN, f"Wells {overlap} are already cast, and the allocator resumes "
                  f"past them rather than re-casting — the run will land on other "
                  f"wells than the channel list suggests."))
    elif overlap:
        findings.append(Finding(
            BLOCK, f"Wells {overlap} on board {board_id} are already cast, and "
                   f"this spec has no electrode_capacity, so nothing skips them: "
                   f"the run would drop fresh formulations onto used wells."))
    elif occupied:
        findings.append(Finding(
            OK, f"Board {board_id} has {len(occupied)} used well(s), none of them "
                f"a channel this campaign targets."))
    return Section("Board", tuple(rows), tuple(findings))


# ── 5. Environment & calibration ─────────────────────────────────────────────

def _environment_section(spec: Any, data_store: Any,
                         config: Mapping[str, Any] | None) -> Section:
    from softae.analysis.eis.calibration import describe_or_absent, resolve_calibration
    from softae.analysis.eis.settings import eis_settings
    from softae.core.purge import load_purge_settings

    eis = _try(lambda: eis_settings(dict(config) if config else None))
    fixture = getattr(getattr(eis, "fixture", None), "fixture_id", None) or "default"
    calibration = _try(lambda: resolve_calibration(fixture))
    purge = _try(lambda: load_purge_settings(data_store))

    measured = set(getattr(calibration, "channels_measured", ()) or ())
    assumed = set(getattr(calibration, "channels_assumed", ()) or ())
    channels = [int(c) for c in getattr(spec, "channels", ()) or ()]
    uncalibrated = [c for c in channels if c not in measured and c not in assumed]
    engine = getattr(eis, "engine", "?")

    rows = [
        ("eis engine", str(engine)),
        ("fixture", f"{fixture}  ·  "
                    f"{_try(lambda: describe_or_absent(calibration), 'unavailable')}"),
        ("uncalibrated", ", ".join(str(c) for c in uncalibrated)
         or "none of the declared channels"),
        ("purge", getattr(purge, "describe", lambda: "unavailable")()),
    ]
    findings: list[Finding] = []
    if uncalibrated and engine != "legacy":
        findings.append(Finding(
            WARN, f"Channels {uncalibrated} have no measured calibration and the "
                  f"'{engine}' engine applies commissioning constants — those "
                  f"channels are corrected with configured estimates."))
    elif uncalibrated:
        findings.append(Finding(
            OK, f"Channels {uncalibrated} are uncommissioned, but engine="
                f"'{engine}' applies no commissioning constants."))
    if getattr(purge, "actuate", False):
        findings.append(Finding(
            WARN, "[purge] actuate is ON: the anti-clog harness will move fluid "
                  "during this run without being asked."))
    return Section("Environment & calibration", tuple(rows), tuple(findings))


# ── 6. Projection ────────────────────────────────────────────────────────────

def _projection_section(spec: Any, task_catalog: Any, ledger: Any,
                        project_dir: Any) -> Section:
    if task_catalog is None:
        return Section(
            "Projection",
            (("duration", "not projected (no task catalog supplied)"),),
            (Finding(WARN, "No duration or stock projection was run, so nothing "
                           "here says whether the declared stock covers the "
                           "budget."),))

    from softae.core.preflight import project_campaign

    try:
        projection = project_campaign(spec, catalog=task_catalog, ledger=ledger,
                                      project_dir=project_dir)
    except Exception as exc:
        return Section("Projection",
                       (("duration", f"unavailable: {exc}"),),
                       (Finding(WARN, f"The projection could not be built: {exc}"),))

    supported = projection.iterations_supported()
    draw = sum(projection.per_iteration_draw_uL.values())
    rows = [
        ("per iteration", _dur(projection.per_iteration_s)
         + ("" if projection.duration_complete else "  (lower bound)")),
        ("to budget", f"at most {_dur(projection.time_to_budget_s)} for "
                      f"{projection.budget} iteration(s)"),
        ("stock draw", f"{draw:,.0f} µL per iteration"),
        ("runway", "unknown — no reservoir levels declared" if supported is None
         else f"about {supported} iteration(s) on declared stock"),
    ]
    findings = [Finding(WARN, w) for w in projection.warnings]
    if projection.stock_sufficient is False:
        findings.insert(0, Finding(
            WARN, f"Declared stock covers only ~{supported} of "
                  f"{projection.budget} iterations; the run hard-stops early."))
    elif projection.stock_sufficient:
        findings.append(Finding(
            OK, f"Declared stock covers the full {projection.budget}-iteration "
                f"budget."))
    return Section("Projection", tuple(rows), tuple(findings))


# ── Assembly ─────────────────────────────────────────────────────────────────

def build_final_check(
    spec: Any,
    *,
    data_store: Any = None,
    task_catalog: Any = None,
    chem_catalog: Any = None,
    sol_catalog: Any = None,
    config: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> FinalCheck:
    """The digest for *spec*, read from whatever context is supplied.

    Every argument is optional and **every absence is reported as an absence**:
    without *data_store* the board and the loadout are unchecked rather than
    clean, without *task_catalog* there is no projection. *chem_catalog* and
    *sol_catalog* default to the shared chemistry seam
    (:func:`~softae.core.campaign_spec_fields.catalogs`), which a test replaces
    to avoid depending on this machine's ``data/``.
    """
    from softae.core.campaign_spec_fields import catalogs
    from softae.core.reservoir import ReservoirLedger

    if chem_catalog is None or sol_catalog is None:
        loaded_chem, loaded_sol = _try(catalogs, (None, None))
        chem_catalog = chem_catalog or loaded_chem
        sol_catalog = sol_catalog or loaded_sol

    ledger = ReservoirLedger(data_store) if data_store is not None else None
    project_dir = getattr(data_store, "project_dir", None)

    return FinalCheck(
        title=str(getattr(spec, "name", "campaign")),
        sections=(
            _campaign_section(spec, source),
            _stock_section(spec, data_store, ledger, chem_catalog, sol_catalog),
            _sequence_section(spec, task_catalog),
            _board_section(spec, data_store),
            _environment_section(spec, data_store, config),
            _projection_section(spec, task_catalog, ledger, project_dir),
        ),
    )


# ── Rendering ────────────────────────────────────────────────────────────────

_VERIFY_TITLE = "Verify before proceeding"


def _row_lines(label: str, value: str, pad: int, width: int) -> list[str]:
    head = f"   {label.ljust(pad)}  {value}" if label else f"   {' ' * pad}  {value}"
    return textwrap.wrap(head, width=width,
                         subsequent_indent=" " * (pad + 5)) or [head]


def render_final_check(fc: FinalCheck, width: int = 80) -> str:
    """The digest as plain monospace text — no colour, no cursor control.

    Deliberately not a table library and not a Rich renderable: this has to read
    the same in a terminal, a log file and a GUI text pane, and the surfaces that
    will show it are not all built yet.
    """
    lines = [f"FINAL CHECK — {fc.title}", "=" * width]
    for index, section in enumerate(fc.sections, start=1):
        lines.append("")
        lines.append(f"{index}. {section.title}")
        pad = max((len(label) for label, _ in section.rows), default=0)
        for label, value in section.rows:
            lines.extend(_row_lines(label, value, pad, width))

    lines.append("")
    lines.append(f"{len(fc.sections) + 1}. {_VERIFY_TITLE}")
    for finding in fc.findings:
        lines.extend(textwrap.wrap(
            f"   [{_TAG.get(finding.severity, '?')}] {finding.text}",
            width=width, subsequent_indent=" " * 11) or [finding.text])
    if not fc.findings:
        lines.append("   nothing to verify — no checks produced a finding")

    lines.append("")
    lines.append(f"{fc.n_block} blocking, {fc.n_warn} warning(s). "
                 + ("Resolve the blocking item(s) before launching."
                    if fc.has_block else "Read them, then answer."))
    return "\n".join(lines)


# ── The prompt ───────────────────────────────────────────────────────────────

def confirm_final_check(fc: FinalCheck, *, assume_yes: bool,
                        ask: Callable[[str], str] = input) -> bool:
    """Whether to launch. Prints nothing; the caller has already shown the page.

    **A ``block`` is not overridable at the prompt, and ``assume_yes`` is.** A
    block means the file and the bench contradict each other, and a ``y`` typed
    in the moment is not evidence the contradiction was resolved — only that
    someone wanted to continue. The fix is to change the spec or the bench and
    run the check again. ``--yes`` is different because it is a decision recorded
    in the launch command, the same standing ``softae-campaign run --yes``
    already has over a projected stock shortfall.
    """
    if assume_yes:
        return True
    try:
        answer = str(ask("Proceed? [y/N] ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        # The same reading `softae-campaign`'s own `_confirm` takes: a prompt
        # nobody could answer is a no, never a yes.
        return False
    return answer in ("y", "yes") and not fc.has_block


def main(argv: "Sequence[str] | None" = None) -> int:
    """``py -m softae.core.final_check <spec.toml> [--project DIR] [--yes]``.

    The prototype surface, so the layout can be iterated on before it is wired
    into ``softae-campaign run`` and the GUI. Exits 0 to proceed, 3 otherwise.
    """
    import argparse

    from softae.config import loader
    from softae.core.campaign_spec_io import load_campaign_spec
    from softae.core.task_catalog import TaskCatalog
    from softae.tools import use_utf8_console

    use_utf8_console()
    parser = argparse.ArgumentParser(prog="softae.core.final_check")
    parser.add_argument("spec")
    parser.add_argument("--project", default=None)
    parser.add_argument("--yes", "-y", action="store_true")
    parser.add_argument("--width", type=int, default=80)
    args = parser.parse_args(argv)

    spec = load_campaign_spec(args.spec)
    catalog = _try(lambda: TaskCatalog.load_toml(loader.tasks_toml_path()))

    store = None
    if args.project:
        from softae.core.data_store import DataStore

        store = DataStore(args.project)
    try:
        digest = build_final_check(spec, data_store=store, task_catalog=catalog,
                                   source=str(args.spec))
        print(render_final_check(digest, width=args.width))
        print()
        # `isatty()` is not sufficient on its own: a PowerShell child process
        # reports a terminal and then reads EOF on the first `input()`, so the
        # unanswerable case is detected by the read failing, not by the flag.
        unanswerable = not (sys.stdin and sys.stdin.isatty())

        def _ask(prompt: str) -> str:
            nonlocal unanswerable
            try:
                return input(prompt)
            except (EOFError, KeyboardInterrupt):
                unanswerable = True
                return ""

        proceed = confirm_final_check(
            digest, assume_yes=args.yes,
            ask=(lambda _p: "") if unanswerable else _ask)
        if unanswerable and not args.yes:
            print("(no terminal — pass --yes at launch)")
        return 0 if proceed else 3
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
