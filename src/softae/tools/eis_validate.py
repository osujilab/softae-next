"""On-rig validation of adaptive EIS acquisition: does it produce better science?

Adaptive acquisition shipped inert behind ``[eis.scout] actuate`` /
``actuate_manual``, both ``false``. It works **scout-then-measure**: the
operator's selected sweep runs first and *is* the measurement whenever the
verdict is ``ok``; only an inadequate spectrum earns a second sweep. The segment
format is bench-verified on hardware. What is **not** established is whether the
result is better on real samples -- and this tool is the third call site of
:class:`~softae.core.eis_scout_scripts.ScoutPlanner`, ``site="validation"``,
that measures it.

The experiment, in three steps, at **one definable RH and temperature setpoint,
equilibrated and held**:

1. measure the reference preset on a set of live channels;
2. run adaptive acquisition on the *same cells*, seconds apart;
3. report the deviation, with its direction.

**The one paragraph that governs the design.** ``Extended`` reaches 1.351 Hz, so
its own arc closes only for an apex above about 13.51 Hz; below that its ``R1``
came from extrapolating the high-frequency limb -- a measured **+60.9 % median
overestimate** -- and it is not a reference there at all. Meanwhile ``Quick``
(6.475 Hz) returns ``ok`` for any apex above 64.75 Hz, and on ``ok`` the two arms
are **byte-identical**. So this experiment can resolve anything only in the apex
window **13.5 - 65 Hz**, 0.68 decades wide. The three-way partition, the
arc-capture watch that checks the setpoint *before* committing, and the
INSUFFICIENT outcome all follow from that one fact.

Ordering
--------
1. resolve the plan; print the projection; **thermal confirmation**
2. approach temperature -> approach RH        (refuses on timeout)
3. settle phase -> arc-capture watch          (refuses on ceiling/not_evaluable)
4. per channel, **interleaved**: reference, then adaptive, then the next channel
5. drift check: re-run the reference on the first N channels
6. park (default) or hold; disconnect

Interleaved, not blocked, because running every cell's reference first would
leave ~30 minutes between the paired measurements the entire primary metric is
computed from. Interleaving puts them seconds apart, which removes essentially
all within-pair drift and is strictly better than modelling it.

Safety
------
``assert_hardware_armed`` is a **documented no-op for this axis set** --
``MOTION_INSTRUMENTS`` is ``("stage", "syringe", "piezo")``, so ``probe_motion``
returns empty for a temp/RH manager and the assert passes unconditionally. It is
called anyway, because it catches a stage in the manager, but **the gate that
bites is** :func:`~softae.tools.equilibration.confirm_thermal`, which requires
the literal word ``"yes"``. ``safe_park`` runs in a ``finally`` on every exit
path, always with ``retract_head=None``: absent an operator, the correct
response to an unknown head position is to add no motion to it.

A park drives the heater to 10 C and suspends anti-clog purging, so **a park
ends the condition** -- and ``--resume`` therefore re-runs the full approach and
settle gate before taking a single sweep. No flag skips it.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from softae.tools import use_utf8_console
from softae.tools.eis_validate_hold import (
    DEFAULT_DRIFT_CHECK,
    DEFAULT_MIN_TREATMENT,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    DEFAULT_SETTLE_MAX_HOLD_S,
    DEFAULT_TEMP_APPROACH_TIMEOUT_S,
    HoldWatch,
    RefuseToStart,
    ValidationPlan,
    VirtualClock,
    approach_condition,
    assert_settle_licensed,
    population_thresholds,
    project,
    render_arc_watch,
    render_projection,
    settle_phase,
    validate_plan,
)
from softae.tools.eis_validate_report import (
    ARM_FOLLOW_UP,
    ARM_REFERENCE,
    ARM_REFERENCE_END,
    ARM_SCOUT,
    TREATMENT,
    checkpoint_campaign,
    resolve_db,
    resolve_project,
)

logger = structlog.get_logger(__name__)

CONSOLE_SCRIPT = "softae-eis-validate"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DECLINED = 2
EXIT_INTERRUPTED = 130

#: The operator's channel set: 18-32, fifteen channels, all on pico2
#: (``pico2_range = [17, 32]``, remapped by ``mod_channel_restart`` to pico2's
#: 2-16). Channel 18 is the one bench-verified for segmented scripts.
EXAMPLE_CHANNELS = "18-32"


# ── Plan resolution ──────────────────────────────────────────────────────────

def resolve_baseline(explicit: str | None) -> tuple[str, str]:
    """``(preset, where it came from)`` -- and the report prints both.

    **The baseline is a lever, and the wrong choice makes the run empty.** The
    TREATMENT population is bounded above by the baseline's own closure
    threshold, so from the reference preset *as* baseline the window would be
    ``[13.51, 13.51)`` -- empty, because baseline and reference would be the
    same grid. The decision being made is "should adaptive replace *that*", so
    the default has to come from what the modality is really configured with.

    **A finding, recorded rather than papered over:** there is no global config
    key that states one. ``softae_config.toml`` declares ``[eis_presets.*]``
    grids and ``[eis] engine``, but the preset a *campaign* measures at rides in
    its own :class:`~softae.core.measurement_spec.MeasurementSpec`, per spec
    file, and the HT path carries its own. So the resolved default here is
    ``MeasurementSpec``'s default and the *source string says so*; an operator
    validating against a campaign that overrides it must pass ``--baseline``.
    Inventing a config key to close this gap would be a change to shared
    configuration in a tool that promised to change none.
    """
    if explicit:
        return explicit, "--baseline"
    from softae.core.measurement_spec import DEFAULT_PRESET

    return DEFAULT_PRESET, ("measurement_spec.DEFAULT_PRESET -- no global "
                            "config key states a modality preset; pass "
                            "--baseline to match a campaign that overrides it")


def build_plan(args: argparse.Namespace) -> ValidationPlan:
    from softae.analysis.eis.scout import scout_settings
    from softae.core.channel_spec import parse_channel_spec
    from softae.core.eis_scripts import EISParams

    baseline, source = resolve_baseline(args.baseline)
    band = scout_settings().band_below_apex_min_decades
    ref_close, baseline_ok = population_thresholds(
        EISParams.from_preset(baseline).f_lo_mHz / 1000.0,
        EISParams.from_preset(args.reference_preset).f_lo_mHz / 1000.0,
        band,
    )
    return ValidationPlan(
        validation_name=args.validation_name,
        channels=tuple(parse_channel_spec(args.channels)),
        rh_setpoint_pct=float(args.rh_setpoint_pct),
        temp_setpoint_c=float(args.temp_setpoint_c),
        baseline_preset=baseline,
        baseline_source=source,
        reference_preset=args.reference_preset,
        drift_check=int(args.drift_check),
        min_treatment=int(args.min_treatment),
        order=args.order,
        max_follow_ups=int(args.max_follow_ups),
        rh_tolerance_pct=float(args.rh_tolerance_pct),
        tolerance_c=float(args.tolerance_c),
        rh_approach_timeout_s=float(args.rh_approach_timeout_s),
        temp_approach_timeout_s=float(args.temp_approach_timeout_s),
        settle=(args.settle == "on"),
        settle_max_hold_s=float(args.settle_max_hold_s),
        end_state=args.end_state,
        retries=int(args.retries),
        max_consecutive_failures=int(args.max_consecutive_failures),
        mock=bool(args.mock),
        ref_close_hz=ref_close,
        baseline_ok_hz=baseline_ok,
        band_min_decades=float(band),
    )


# ── The run ──────────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    """Everything one invocation carries. Nothing accumulates beyond this."""

    plan: ValidationPlan
    manager: Any
    data_store: Any
    run_id: str
    run_dir: Path
    hold_epoch: int = 1
    hold_certified: str = "settled"
    watch: HoldWatch | None = None
    seq: dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    n_recorded: int = 0

    def next_seq(self, cell: str) -> int:
        self.seq[cell] = self.seq.get(cell, 0) + 1
        return self.seq[cell]

    def script_path(self, channel: int, arm: str) -> str:
        """Run-scoped, never the bare tempdir.

        ``tab_manual.py`` writes ``%TEMP%/softae_testing.mscr`` and
        ``temp_eis_sweep.py`` writes ``%TEMP%/softae_ch{ch}.mscr``. The
        operator's GUI is assumed live, so reusing either name lets a Run press
        overwrite this script between build and send -- producing a row that
        names a sweep which never ran, the exact defect ``core/eis_scripts.py``
        exists to prevent. Every emitted script also survives as a durable
        artifact of the validation.
        """
        scripts = self.run_dir / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        return str(scripts / f"ch{int(channel):02d}_{arm}.mscr")


def acquire(
    ctx: RunContext, channel: int, params: dict[str, Any], script_path: str
) -> Any:
    """Send whatever is at *script_path* and parse what comes back.

    *params* is what the row records, so it must describe the script that is
    about to run -- never the one that was selected if a different one was
    written over it.
    """
    from softae.analysis.eis_data import EISResult
    from softae.config.loader import pico_for_channel

    pico = ctx.manager.get(pico_for_channel(channel))
    outdir = getattr(pico, "_output_dir", None) or str(ctx.run_dir / "eis")
    started = time.monotonic()
    raw = pico.sendscript_getdata(script_path, outdir, channel)
    return EISResult.from_raw(
        raw, channel=channel,
        measurement_time_s=time.monotonic() - started, eis_params=params,
    )


def _finite_or_none(value: Any) -> float | None:
    """``None``, never ``NaN``, for anything bound for ``eis_params_json``.

    **A real hazard, found by running this tool.** ``DataStore`` serialises
    ``eis_params`` with ``json.dumps``, which emits the bare token ``NaN`` --
    valid for Python's own loader and **invalid JSON**. SQLite's JSON1 refuses
    the whole document, so ``json_extract(eis_params_json, '$.anything')``
    raises ``malformed JSON`` and the row becomes invisible to every
    JSON-extracting reader, not just the key that carried the NaN.

    ``arc_closure`` returns NaN for an apex it did not find, which is the common
    case on exactly the open-arc sweeps this harness exists to study -- so a
    naive ``float(...)`` here would have made the majority of rows unreadable.
    Anyone else stamping a float into ``eis_params`` has the same trap.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def persist(ctx: RunContext, eis: Any, arm: str) -> int:
    """Save, record, fit, capture conditions -- **after every single sweep**.

    Nothing accumulates in memory. A crash after the conditions write loses the
    sweep in flight and nothing else, and the report is regenerable from disk at
    any moment.
    """
    from softae.analysis.eis.arc import arc_closure
    from softae.analysis.eis.engine import analyze_spectrum
    from softae.core.conditions_capture import read_environment

    channel = int(eis.channel)
    cell = ctx.plan.cell_key(channel)
    closure = arc_closure(eis.frequency, eis.z_imag_neg, getattr(eis, "phase", None))
    eis.eis_params.update({
        "eis_validation_name": ctx.plan.validation_name,
        "eis_validation_arm": arm,
        "eis_validation_cell": cell,
        "eis_validation_rh_sp_pct": ctx.plan.rh_setpoint_pct,
        "eis_validation_temp_sp_C": ctx.plan.temp_setpoint_c,
        "eis_validation_hold_epoch": ctx.hold_epoch,
        "eis_validation_hold_certified": ctx.hold_certified,
        "eis_validation_hold_excursion": bool(
            ctx.watch.excursion if ctx.watch else False),
        "eis_validation_seq": ctx.next_seq(cell),
        "eis_validation_mock": bool(ctx.plan.mock),
        # Closure of THIS spectrum, stamped at acquisition. The reporter
        # partitions on it, and `fit_results.arc_state` is annotated on whatever
        # survived the admission gates -- which is not always the sweep taken.
        "eis_validation_arc_state": closure.state,
        "eis_validation_apex_hz": _finite_or_none(closure.f_apex_interior_hz),
        "eis_validation_band_below_apex_decades": _finite_or_none(
            closure.band_below_apex_decades),
        # The grid's actual floor, in Hz. `f_lo_mHz` is millihertz and a
        # segmented row's is an aggregate over bands; the rescue-depth statistic
        # and veto V3 both need the number the instrument actually reached.
        "eis_validation_f_lo_hz": _finite_or_none(
            min(eis.frequency) if len(eis.frequency) else float("nan")),
        "eis_validation_ref_close_hz": ctx.plan.ref_close_hz,
        "eis_validation_baseline_ok_hz": ctx.plan.baseline_ok_hz,
        # The cut that separates "measured" from "extrapolated", stamped on the
        # row rather than assumed by the reporter: it is a configured value
        # (`[eis.scout] band_below_apex_min_decades`), so a run analysed after
        # it moves must be judged by the cut that was in force when it ran.
        "eis_validation_band_min_decades": ctx.plan.band_min_decades,
    })

    seq = eis.eis_params["eis_validation_seq"]
    eis_dir = ctx.run_dir / "eis"
    eis_dir.mkdir(parents=True, exist_ok=True)
    eis.save(eis_dir / f"ch{channel:02d}_{seq:03d}_{arm}.txt")

    measurement_id = ctx.data_store.record_measurement(
        ctx.run_id, eis, role="sample")
    try:
        report = analyze_spectrum(eis, model_name="simpleSalt")
        ctx.data_store.record_fit(measurement_id, report.fit, report=report)
    except Exception as exc:
        logger.warning("eis_validate_fit_failed", channel=channel, arm=arm,
                       error=str(exc))
    try:
        ctx.data_store.record_conditions(
            measurement_id, "measurement", **read_environment(ctx.manager))
    except Exception as exc:                              # pragma: no cover
        logger.warning("eis_validate_conditions_failed", error=str(exc))

    ctx.n_recorded += 1
    return measurement_id


def measure_reference(ctx: RunContext, channel: int, arm: str) -> Any:
    """The reference arm: the wide preset, once, no planner."""
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    grid = EISParams.from_preset(ctx.plan.reference_preset)
    path = ctx.script_path(channel, arm)
    eis_run_mscrbuild(path, mux_ch=channel, mVac=grid.mv_ac, f_hi=grid.f_hi,
                      f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
    params = {
        "f_hi": grid.f_hi, "f_lo_mHz": grid.f_lo_mHz, "npts": grid.npts,
        "mv_ac": grid.mv_ac, "mv_dc": grid.mv_dc,
        "eis_preset": ctx.plan.reference_preset,
    }
    eis = acquire(ctx, channel, params, path)
    persist(ctx, eis, arm)
    return eis


def measure_adaptive(ctx: RunContext, planner: Any, channel: int) -> tuple[Any, Any]:
    """Scout-then-measure, **recording both sweeps**.

    A deliberate divergence from ``tab_manual.py``, which discards the
    superseded scout sweep and keeps only its cost in ``eis_scout_sweep_s``.
    Here the scout row **is** the control arm, so discarding it would discard
    the comparison; time accounting must not depend on one write-only JSON key
    surviving a round trip; and a discarded spectrum cannot be re-analysed when
    a threshold moves.
    """
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    grid = EISParams.from_preset(ctx.plan.baseline_preset)
    path = ctx.script_path(channel, "adaptive")
    eis_run_mscrbuild(path, mux_ch=channel, mVac=grid.mv_ac, f_hi=grid.f_hi,
                      f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
    base_params = {
        "f_hi": grid.f_hi, "f_lo_mHz": grid.f_lo_mHz, "npts": grid.npts,
        "mv_ac": grid.mv_ac, "mv_dc": grid.mv_dc,
        "eis_preset": ctx.plan.baseline_preset,
    }

    scout = acquire(ctx, channel, dict(base_params), path)
    decision = planner.observe(channel, scout)
    persist(ctx, scout, ARM_SCOUT)

    follow_up_eis = None
    current, params = decision, dict(base_params)
    for _ in range(max(0, int(ctx.plan.max_follow_ups))):
        follow_up = planner.build_follow_up(path, channel, params, current)
        if follow_up is None:
            break
        scout_sweep_s = (follow_up_eis or scout).measurement_time_s
        follow_up_eis = acquire(ctx, channel, follow_up, path)
        # Stamped here rather than inside `ScoutPlanner`: the superseded sweep is
        # persisted by this harness, but the *cost attribution* still has to say
        # which sweep this row superseded, and moving the stamp into the planner
        # is a separate, declinable change to the manual tab's territory.
        follow_up_eis.eis_params["eis_scout_sweep_s"] = float(scout_sweep_s)
        persist(ctx, follow_up_eis, ARM_FOLLOW_UP)
        current = planner.observe(channel, follow_up_eis)
        params = dict(follow_up)
    return scout, follow_up_eis


def run_cells(ctx: RunContext, planner: Any, channels: list[int]) -> None:
    """Interleaved per channel: R then B, before moving to the next."""
    total = len(channels)
    for index, channel in enumerate(channels, start=1):
        if ctx.watch is not None:
            ctx.watch.poll()
        arms = ["R", "B"]
        if ctx.plan.order == "alternate" and index % 2 == 0:
            arms.reverse()
        try:
            for arm in arms:
                if arm == "R":
                    reference = measure_reference(ctx, channel, ARM_REFERENCE)
                    print(f"[{index:02d}/{total:02d}] ch{channel}  R   "
                          f"{ctx.plan.reference_preset:<10} "
                          f"{reference.measurement_time_s:6.1f} s", flush=True)
                else:
                    scout, follow_up = measure_adaptive(ctx, planner, channel)
                    verdict = scout.eis_params.get("eis_scout_verdict", "-")
                    print(f"[{index:02d}/{total:02d}] ch{channel}  B0  "
                          f"{ctx.plan.baseline_preset:<10} "
                          f"{scout.measurement_time_s:6.1f} s  verdict={verdict}",
                          flush=True)
                    if follow_up is not None:
                        print(f"[{index:02d}/{total:02d}] ch{channel}  B1  "
                              f"{'follow-up':<10} "
                              f"{follow_up.measurement_time_s:6.1f} s", flush=True)
            ctx.consecutive_failures = 0
        except Exception as exc:
            ctx.consecutive_failures += 1
            logger.warning("eis_validate_cell_failed", channel=channel,
                           error=str(exc),
                           consecutive=ctx.consecutive_failures)
            print(f"  ! ch{channel} failed ({exc}); "
                  f"{ctx.consecutive_failures} consecutive", flush=True)
            if ctx.consecutive_failures >= ctx.plan.max_consecutive_failures:
                raise RuntimeError(
                    f"{ctx.consecutive_failures} consecutive cell failures"
                ) from exc


def drift_check(ctx: RunContext, channels: list[int]) -> None:
    """Re-run the reference on the first N channels measured.

    Suppressing drift with a hold is simpler and stronger than modelling it --
    but an unmeasured assumption is not a suppressed one, so the hold is then
    *checked*. A large drift does not fail the feature; it makes the run
    INSUFFICIENT, because a moving sample means the paired differences are not
    what they claim to be.
    """
    for channel in channels[: max(0, int(ctx.plan.drift_check))]:
        if ctx.watch is not None:
            ctx.watch.poll()
        try:
            eis = measure_reference(ctx, channel, ARM_REFERENCE_END)
            print(f"[drift ] ch{channel}  {ctx.plan.reference_preset:<10} "
                  f"{eis.measurement_time_s:6.1f} s", flush=True)
        except Exception as exc:
            logger.warning("eis_validate_drift_failed", channel=channel,
                           error=str(exc))


# ── Resume ───────────────────────────────────────────────────────────────────

class ResumeMismatch(RuntimeError):
    """The plan moved under a resume.

    Refused for the reason ``campaign_resume`` refuses the same thing:
    continuing would append one experiment's observations to another's search,
    which corrupts both and is not visible in the resulting data.
    """


def complete_cells(data_store: Any, plan: ValidationPlan) -> set[str]:
    """Cells whose every planned arm already has a recorded measurement.

    Derived from the **rows**, never from the checkpoint: the checkpoint records
    what was asked, the rows record what was done, and only the second can say
    whether a cell is finished. A partially complete cell is discarded and
    re-run in full -- half a pair gives no deviation, and one cell costs 138 s.
    """
    from softae.tools.eis_validate_report import (
        ARM_REFERENCE as REF,
    )
    from softae.tools.eis_validate_report import (
        ARM_SCOUT as SCOUT,
    )
    from softae.tools.eis_validate_report import (
        assemble_cells,
        load_records,
    )

    db_path = resolve_db(data_store.project_dir)
    try:
        records = load_records(db_path, plan.validation_name)
    except Exception:
        return set()
    done: set[str] = set()
    for cell in assemble_cells(records):
        arms = {r.arm for r in records if r.cell == cell.key}
        if REF in arms and SCOUT in arms:
            done.add(cell.key)
    return done


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        description="Validate adaptive EIS acquisition against a wide-preset "
                    "reference, at one equilibrated and held condition.",
        epilog=(
            f"Example (the operator's channel set, 15 channels on pico2):\n"
            f"  {CONSOLE_SCRIPT} run --channels {EXAMPLE_CHANNELS} "
            f"--rh-setpoint-pct 30 --temp-setpoint-c 25 "
            f"--validation-name adaptive-2026-08\n"
            f"  {CONSOLE_SCRIPT} report --validation-name adaptive-2026-08"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="drive the rig and persist every sweep")
    run.add_argument("--channels", required=True,
                     help='channel list/range, e.g. "18-32"')
    run.add_argument("--rh-setpoint-pct", required=True, type=float,
                     help="REQUIRED, no default: an unstated condition is not "
                          "a condition")
    run.add_argument("--temp-setpoint-c", required=True, type=float,
                     help="REQUIRED, no default")
    run.add_argument("--validation-name", required=True,
                     help="groups invocations; the resume key")
    run.add_argument("--baseline", default=None,
                     help="the scout's grid; defaults to the modality's "
                          "CONFIGURED preset, not a literal")
    run.add_argument("--reference-preset", default="Extended",
                     help="'longest' widens the resolving window 2.1x for 4.3x "
                          "the reference cost -- the documented escape hatch")
    run.add_argument("--drift-check", type=int, default=DEFAULT_DRIFT_CHECK)
    run.add_argument("--min-treatment", type=int, default=DEFAULT_MIN_TREATMENT)
    run.add_argument("--order", choices=("ref-first", "alternate"),
                     default="ref-first",
                     help="'alternate' is a diagnostic, not a design element")
    run.add_argument("--max-follow-ups", type=int, default=1,
                     help="1 validates what ships; >1 is exploratory and is "
                          "excluded from every criterion")
    run.add_argument("--rh-tolerance-pct", type=float, default=2.0)
    run.add_argument("--tolerance-c", type=float, default=2.0)
    run.add_argument("--rh-approach-timeout-s", type=float,
                     default=DEFAULT_RH_APPROACH_TIMEOUT_S,
                     help="NOT the shipped 1800: a measured RH descent took "
                          "~5000 s")
    run.add_argument("--temp-approach-timeout-s", type=float,
                     default=DEFAULT_TEMP_APPROACH_TIMEOUT_S)
    run.add_argument("--settle", choices=("on", "off"), default="on")
    run.add_argument("--settle-max-hold-s", type=float,
                     default=DEFAULT_SETTLE_MAX_HOLD_S)
    run.add_argument("--end-state", choices=("park", "hold"), default="park")
    run.add_argument("--retries", type=int, default=1)
    run.add_argument("--max-consecutive-failures", type=int, default=3)
    run.add_argument("--resume", action="store_true",
                     help="re-enter an existing validation; ALWAYS "
                          "re-equilibrates, because a park ended the condition")
    run.add_argument("--project", default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--mock", action="store_true",
                     help="grid-aware synthetic backend; never prompts, never "
                          "arms, and never emits a GO")
    run.add_argument("--mock-apex-hz", default=None,
                     help='apex placement: a number, or "18:30,19:200"')
    run.add_argument("--mock-drift-decades-per-hour", type=float, default=0.0)
    run.add_argument("--yes", "-y", action="store_true")
    run.add_argument("--dry-run", action="store_true",
                     help="print plan + projection, run nothing")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="read back and evaluate the rule")
    report.add_argument("--validation-name", required=True)
    report.add_argument("--project", default=None)
    report.add_argument("--out", default=None)
    report.add_argument("--min-treatment", type=int, default=None)
    from softae.tools.eis_validate_report import cmd_report

    report.set_defaults(func=cmd_report)
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from softae.core.data_store import DataStore
    from softae.core.eis_scout_scripts import ScoutPlanner
    from softae.core.hardware_safety import assert_hardware_armed
    from softae.core.safe_park import safe_park
    from softae.drivers.factory import create_manager

    plan = build_plan(args)
    projection = project(plan)
    print(render_projection(plan, projection))
    try:
        validate_plan(plan)
    except RefuseToStart as exc:
        print(f"\nREFUSING TO START: {exc}")
        return EXIT_FAILED
    if args.dry_run:
        print("\n--dry-run: nothing was heated and nothing was measured.")
        return EXIT_OK

    manager = create_manager(mock=bool(args.mock))
    if args.mock:
        from softae.tools.eis_validate_mock import (
            MockRig,
            install_fast_conditions,
            install_mock_picos,
            parse_apex_spec,
        )

        per_channel, default_hz = parse_apex_spec(args.mock_apex_hz)
        install_fast_conditions(manager)
        install_mock_picos(manager, MockRig(
            apex_hz=per_channel, default_apex_hz=default_hz,
            drift_decades_per_hour=float(args.mock_drift_decades_per_hour),
            # A mock run finishes in milliseconds, so drift per *hour* has to be
            # counted in sweeps or it would be unobservable and Delta_hold would
            # come back identically zero -- a fake null, which is precisely what
            # the grid-aware backend exists to stop.
            virtual_s_per_sweep=60.0,
        ))

    # Cheap, and it catches a stage in the manager -- but it is a documented
    # NO-OP for a temp/RH manager, because MOTION_INSTRUMENTS never matches one.
    # Do not delete this as dead code; a test pins it for exactly that reason.
    assert_hardware_armed(
        manager, action=f"run EIS validation on channels {args.channels}")
    if not args.mock and not _confirm(plan, projection, assume_yes=args.yes):
        return EXIT_DECLINED

    store = DataStore(resolve_project(args.project))
    try:
        import asyncio

        asyncio.run(manager.connect_all())
        ctx = _enter_run(store, manager, plan, args)
        remaining = _remaining_channels(store, plan, resume=bool(args.resume))
        if not remaining:
            print("Every planned cell is already complete; nothing to measure.")
            return EXIT_OK

        _establish_condition(ctx, plan, remaining)
        ctx.watch = HoldWatch(manager=manager, plan=plan)
        planner = ScoutPlanner(site="validation", actuate=True)
        run_cells(ctx, planner, remaining)
        drift_check(ctx, remaining)
        _write_report(store, plan, args)
        return EXIT_OK
    except RefuseToStart as exc:
        print(f"\nREFUSING TO START: {exc}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\nInterrupted. Recorded rows stand; nothing new was started.")
        return EXIT_INTERRUPTED
    except Exception as exc:
        logger.error("eis_validate_run_failed", error=str(exc))
        print(f"\nRUN FAILED: {exc}")
        return EXIT_FAILED
    finally:
        # Every exit path, success included. `retract_head=None` because absent
        # an operator the correct response to an unknown is to add no motion to
        # it -- `safe_park`'s default is reversed for exactly this caller class.
        if plan.end_state == "park":
            result = safe_park(manager, reason="eis validation complete",
                               retract_head=None)
            print(f"\n[park  ] {result.summary()}")
            print("         A PARK ENDS THE CONDITION: the heater is driven to "
                  "10 C and anti-clog purging is suspended. --resume "
                  "re-equilibrates from scratch.")
        else:
            print("\n[hold  ] --end-state hold: the heater and humidifier are "
                  "STILL DRIVEN. You are expected to be standing there.")
        try:
            import asyncio

            asyncio.run(manager.disconnect_all())
        except Exception:                                 # pragma: no cover
            pass


def _confirm(plan: ValidationPlan, projection: Any, *, assume_yes: bool) -> bool:
    """The gate that actually bites: a typed ``"yes"``, not a keypress.

    ``confirm_thermal`` is reused unmodified. Its own banner projects the
    *approach and settle* phases from an :class:`EquilibrationConfig`; the
    measurement block this harness adds on top is disclosed as an explicit line
    rather than folded in, because a number that silently disagreed with the
    projection printed a moment earlier would be worse than two labelled ones.
    """
    from softae.tools.equilibration import confirm_thermal
    from softae.workflows.equilibration import EquilibrationConfig

    config = EquilibrationConfig(
        channels=list(plan.channels),
        temperatures_C=[plan.temp_setpoint_c],
        legs=("up",),
        rh_setpoint_pct=plan.rh_setpoint_pct,
        eis_preset=plan.baseline_preset,
        tolerance_C=plan.tolerance_c,
        rh_tolerance_pct=plan.rh_tolerance_pct,
        rh_approach_timeout_s=plan.rh_approach_timeout_s,
        approach_timeout_s=plan.temp_approach_timeout_s,
    )
    disclosure = (
        "hours above cover APPROACH + SETTLE only; the measurement block adds",
        "0 min",
        f"{projection.measurement_low_s / 60:.0f} - "
        f"{projection.measurement_high_s / 60:.0f} min",
    )
    return confirm_thermal(config, assume_yes=assume_yes,
                           plan_overrides=[disclosure])


def _enter_run(store: Any, manager: Any, plan: ValidationPlan,
               args: argparse.Namespace) -> RunContext:
    """Start or re-enter the run, and checkpoint *what was asked*."""
    campaign = checkpoint_campaign(plan.validation_name)
    saved = store.campaign_checkpoint(campaign) or {}
    spec = json.loads(saved.get("spec_json") or "{}") if saved else {}

    if args.resume and spec:
        if spec.get("fingerprint") != plan.fingerprint():
            raise ResumeMismatch(
                "the plan changed since this validation started "
                f"({spec.get('fingerprint')} -> {plan.fingerprint()}). "
                "Continuing would append one experiment's observations to "
                "another's, which corrupts both and is not visible in the "
                "resulting data."
            )
        run_id = str(saved.get("run_id") or "")
        hold_epoch = int(spec.get("hold_epoch", 1)) + 1
        print(f"[resume] re-entering run {run_id}, hold epoch {hold_epoch}. "
              "A park ended the previous condition, so the full approach and "
              "settle gate run again before any sweep.")
    else:
        run_id = store.start_run(
            "eis_validate", mode="validation",
            eis_preset=plan.baseline_preset,
            campaign=campaign,
            annotation=f"EIS adaptive validation '{plan.validation_name}'",
        )
        hold_epoch = 1

    payload = plan.as_dict()
    payload["hold_epoch"] = hold_epoch
    store.save_campaign_checkpoint(
        campaign, iteration=hold_epoch, run_id=run_id,
        loop_state="running", spec_json=json.dumps(payload))

    return RunContext(
        plan=plan, manager=manager, data_store=store, run_id=run_id,
        run_dir=Path(store.project_dir) / "runs" / run_id,
        hold_epoch=hold_epoch,
    )


def _remaining_channels(store: Any, plan: ValidationPlan, *, resume: bool) -> list[int]:
    if not resume:
        return list(plan.channels)
    done = complete_cells(store, plan)
    remaining = [ch for ch in plan.channels if plan.cell_key(ch) not in done]
    print(f"[resume] {len(done)} cell(s) already complete; "
          f"{len(remaining)} remaining.")
    return remaining


def _establish_condition(ctx: RunContext, plan: ValidationPlan,
                         channels: list[int]) -> None:
    """Approach, settle, arc-capture watch -- then and only then, measure.

    Under ``--mock`` the *pacing* is collapsed and nothing else is: the same
    ``approach_setpoint``, the same ``SettleTracker``, the same verdicts and the
    same refusals run, but the waits between polls and between rounds go to
    zero. A 25-minute minimum hold is a statement about a real thermal mass, and
    there isn't one; keeping it would only mean the mock path never gets
    exercised.
    """
    clock = VirtualClock() if plan.mock else None
    pacing: dict[str, Any] = (
        {"sleep": clock.sleep, "now": clock} if clock else {}
    )
    for report in approach_condition(ctx.manager, plan, **pacing):
        print(f"[approach] {report.axis:<12} -> {report.target:g}  "
              f"PV {report.pv_final:g}  {report.elapsed_s / 60:.1f} min")

    settle_pacing: dict[str, Any] = (
        {"sleep": clock.sleep, "now": clock, "min_hold_first_s": 0.0}
        if clock else {}
    )
    outcome = settle_phase(ctx.manager, plan, lambda ch: _settle_sweep(ctx, ch),
                           **settle_pacing)
    assert_settle_licensed(outcome)
    ctx.hold_certified = outcome.verdict
    print(render_arc_watch(outcome, plan))

    projected = outcome.projected.get(TREATMENT, 0)
    if plan.settle and projected < plan.min_treatment:
        raise RefuseToStart(
            f"only {projected} channel(s) project into TREATMENT, below "
            f"--min-treatment {plan.min_treatment}. The resolving window is "
            f"[{plan.ref_close_hz:.2f}, {plan.baseline_ok_hz:.2f}) Hz and the "
            "SETPOINT is the lever, not the sample size: change the condition, "
            "or widen the reference with --reference-preset longest (2.1x the "
            "window for 4.3x the reference cost). Stopping now rather than "
            "spending the measurement block to report INSUFFICIENT."
        )


def _settle_sweep(ctx: RunContext, channel: int) -> Any:
    """One baseline-preset sweep for the settle gate. **Not persisted.**

    These spectra decide whether the material has stopped moving and supply the
    apex histogram; they are not measurements of the condition under test, and
    recording them would put pre-equilibration rows in the same validation the
    reporter reads.
    """
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    grid = EISParams.from_preset(ctx.plan.baseline_preset)
    path = ctx.script_path(channel, "settle")
    eis_run_mscrbuild(path, mux_ch=channel, mVac=grid.mv_ac, f_hi=grid.f_hi,
                      f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
    return acquire(ctx, channel, {"eis_preset": ctx.plan.baseline_preset}, path)


def _write_report(store: Any, plan: ValidationPlan,
                  args: argparse.Namespace) -> None:
    from softae.tools.eis_validate_report import generate, render

    db_path = resolve_db(store.project_dir)
    payload = generate(db_path, plan.validation_name,
                       min_treatment=plan.min_treatment)
    print(render(payload))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {args.out}")


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
