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
2. start (or re-enter) the run row, **open the stream**, **claim the rig**, connect
3. command BOTH setpoints; judge temperature, then RH   (refuses on timeout)
4. settle phase -> arc-capture watch          (refuses on ceiling/not_evaluable)
5. soak: hold the established condition       (``--soak-h``, default 0)
6. per channel, **interleaved**: reference, then adaptive, then the next channel
7. drift check: re-run the reference on the first N channels
8. park (default) or hold; disconnect; **release the claim**; close the stream

Step 5 exists because step 4 answers a different question than it appears to.
The settle gate certifies that **the rig** stopped moving; a film taking up water
at a new RH moves on a far longer timescale and can hold a locally flat trailing
window while it is still hours from equilibrium. ``--soak-h`` is a floor on
*continuous time at condition*, clocked from the moment the approach completes --
so the settle rounds, which run at condition, count against it.

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

The run also **claims the rig** for its whole connected life, as
``tool:eis-validate:<run_id>`` via
:func:`~softae.core.rig_session.held_rig_session` -- taken before
``connect_all`` and given back after ``disconnect_all``. Until it did, an
operator who opened the desktop GUI mid-sweep got a window whose own claim
*succeeded*, because this tool claimed nothing, and which then connected onto
the serial ports this tool was mid-sweep on. Closing the run row
(``owner_pid``) made the **record** of such a run safe; only the claim makes its
**ports** safe.

Watching one, mid-run
---------------------
The claim answered "may I take the rig"; it did not answer the operator's other
question at hour two, which is *"is it still at setpoint, and how far along is
it?"*. So the run now publishes the two sidecars a campaign publishes --
``events.jsonl`` and ``conditions.json``, beside the run, both best-effort --
through :mod:`softae.tools.eis_validate_narrate`, and the claim's ``log_path``
names that directory so a watcher can find it at all. The narration is opened
**before** the claim, so the directory the lock advertises already holds this
run's own stream rather than a promise of one.

Read that module before changing anything here: the beat is a thread rather than
an asyncio task because this runner is synchronous, and ``conditions.json`` is
published from the capture :func:`persist` already performs rather than from a
second reader competing for the serial lock a sweep is using.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from softae.tools import (
    add_verbosity_flag,
    configure_logging,
    run_finalizer,
    use_utf8_console,
)
from softae.tools.eis_validate_hold import (
    DEFAULT_DRIFT_CHECK,
    DEFAULT_MIN_TREATMENT,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    DEFAULT_SETTLE_CRITERION,
    DEFAULT_SETTLE_MAX_HOLD_S,
    DEFAULT_SETTLE_RATE_TOL_DEC_PER_H,
    DEFAULT_SETTLE_TOL_REL,
    DEFAULT_SOAK_S,
    DEFAULT_TEMP_APPROACH_TIMEOUT_S,
    SETTLE_RATE_TOL_DEC_PER_H_MAX,
    SETTLE_TOL_REL_MAX,
    SOAK_CEILING_FACTOR,
    SOAK_PRINT_EVERY_N_POLLS,
    HoldWatch,
    RefuseToStart,
    ValidationPlan,
    VirtualClock,
    approach_condition,
    assert_settle_licensed,
    band_by_channel,
    population_thresholds,
    project,
    render_arc_watch,
    render_projection,
    settle_phase,
    soak_phase,
    validate_plan,
)
from softae.tools.eis_validate_narrate import (
    PHASE_APPROACH,
    PHASE_CELLS,
    PHASE_DRIFT,
    PHASE_FINISHED,
    PHASE_PARK,
    PHASE_REPORT,
    PHASE_SETTLE,
    PHASE_SOAK,
    RunNarration,
    open_narration,
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

#: ``<kind>:<name>:<run_id>`` -- the grammar ``core.rig_session`` documents,
#: whose shipped siblings are ``campaign:<name>:<run_id>`` and
#: ``tool:env-hold:<run_id>``. The third field is **filled**, unlike the GUI's
#: ``gui:desktop`` which omits it because a window is not a run: this tool has a
#: real run id, and a bare ``tool:eis-validate:`` in a lock file would assert
#: "there is a run id and it is blank".
CLAIM_KIND = "tool:eis-validate"

#: Exit statuses whose park is an **orderly** one, as opposed to a fault class.
#:
#: **Currently unused, and left standing deliberately.** Its only consumer was
#: the end-of-run ``safe_park`` call, which used it to pick the humidifier's end
#: state: orderly exits got a dry purge, fault exits kept duty 0. That choice no
#: longer exists -- operator ruling 2026-08-24, recorded in ``core/safe_park.py``
#: with the rule it reversed: **every** park dry-purges now, because dry gas
#: carries very little volatile species and duty 0 is the firmware's
#: auto-shutoff, which collapses the chamber to room RH.
#:
#: It is not deleted here because that is a separate decision from the one the
#: ruling made, and the distinction it draws is still a real one this file
#: computes cheaply from ``outcome["status"]``:
#:
#: * ``done`` -- the run finished, or found nothing to measure.
#: * ``aborted`` -- a *decision*. A gate refused, or the rig was claimed. Nothing
#:   faulted.
#: * ``interrupted`` -- the operator pressed Ctrl-C and will very likely restart.
#:
#: ``error`` is absent: an unnamed exception is this harness's fault class.
#: If nothing has taken it up by the time ``safe_park``'s deprecated
#: ``rh_dry_purge`` parameter is removed, delete it then.
ORDERLY_EXIT_STATUSES = frozenset({"done", "aborted", "interrupted"})

#: What ``--end-state hold`` leaves behind, said accurately.
#:
#: It replaces *"the heater and humidifier are STILL DRIVEN"*, which was half
#: true and so worse than wholly wrong: temperature really does hold, and the
#: humidifier really was being shut off by ``disconnect_all`` one line later.
#: The three facts an operator needs are which axis holds, which does not and for
#: how long, and the one command that takes the axis over -- because **no exiting
#: process can hold RH**: the Trinket wants a continuous heartbeat and its
#: deadman is ~25 s. ``softae-env hold`` is a live process and is the only thing
#: on this rig that can. It is printed, never spawned: an automatic hand-off is a
#: separate item the operator has deferred.
HOLD_NOTICE = """\
         --end-state hold: TEMPERATURE HOLDS. The heater keeps its setpoint;
         nothing on this path writes to it, and no park was performed.
         HUMIDITY DOES NOT HOLD, and no exiting process can make it: the Trinket
         needs a continuous heartbeat and its deadman is ~{deadman:g} s. The PID
         loop is stopped and the humidifier is left commanded DRY, so both valves
         shut ~{deadman:g} s from now and the chamber then drifts toward room RH
         unless something takes the axis over inside that window.
         To actually keep this condition, start the holder NOW:
             python -m softae.tools.env_hold hold --rh {rh:g} --execute --yes
         (add --duration-h H to bound it; -y skips the typed confirmation the
         {deadman:g} s window has no room for. This run gives the rig claim back a
         moment after this line -- if the holder refuses as busy, retry it once.)"""


def _no_run_to_finalize(status: str) -> None:
    """The finalizer before a run row exists. Deliberately does nothing.

    ``cmd_run`` opens its ``DataStore`` before the ``try`` that owns the exit
    paths, so the ``except``/``finally`` arms are reachable with no ``run_id``
    yet -- most plausibly a ``--resume`` that raises ``ResumeMismatch`` inside
    ``_enter_run``. There is no row of this process's to close there, and the row
    the checkpoint names belongs to a *different* plan.
    """


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
        # `getattr`, on `softae.tools.equilibration`'s precedent: a namespace
        # built by a caller older than the flag keeps the shipped band rather
        # than raising.
        settle_tol_rel=float(getattr(args, "settle_tol_rel",
                                     DEFAULT_SETTLE_TOL_REL)),
        # Same `getattr` discipline, same reason: a namespace built by a caller
        # older than the flag keeps the shipped criterion rather than raising.
        settle_criterion=str(getattr(args, "settle_criterion",
                                     DEFAULT_SETTLE_CRITERION)),
        settle_rate_tol_dec_per_h=float(
            getattr(args, "settle_rate_tol_dec_per_h",
                    DEFAULT_SETTLE_RATE_TOL_DEC_PER_H)),
        survivors=(getattr(args, "survivors", "off") == "on"),
        # Hours in, seconds on the plan: the flag is in the operator's unit and
        # every field beside it is in the arithmetic's.
        soak_s=float(args.soak_h) * 3600.0,
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
    #: ``{channel: hold_certified}`` for the cells a survivor partition dropped.
    #: Empty on every run that did not partition, which is every run at the
    #: defaults. A dropped cell is still SWEPT -- §6.4 of the spec, and the
    #: cheaper half of the argument: one extra sweep set per phase buys the
    #: evidence to audit the drop and recalibrate the threshold, where omitting
    #: the cell loses it the way console scrollback lost the 2026-08-21 rounds.
    dropped: dict[int, str] = field(default_factory=dict)
    watch: HoldWatch | None = None
    seq: dict[str, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    n_recorded: int = 0
    #: Where this run says what it is doing. Defaulted to an **inert** narration
    #: rather than to ``None`` so every call site is a plain method call: a
    #: context built directly -- in a test, or by a future caller -- narrates
    #: nothing and works unchanged, and no null check can be forgotten inside a
    #: run block that drives a heater.
    narration: Any = None

    def __post_init__(self) -> None:
        if self.narration is None:
            self.narration = RunNarration(self.run_dir)

    def certification(self, channel: int) -> str:
        """What ``hold_certified`` this channel's rows carry.

        Per channel and not per run, because under a survivor partition the two
        differ and the difference is the whole point: the run proceeded, and this
        cell is not one the gate could speak for.
        :func:`~softae.tools.eis_validate_rule` already filters populations on
        this column, so a dropped cell's rows are excluded from every statistic
        by machinery that exists -- while its data survives on disk, which is
        what makes the drop auditable.
        """
        return self.dropped.get(int(channel), self.hold_certified)

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
        "eis_validation_hold_certified": ctx.certification(channel),
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
        # The plan's model, not a literal: the settle gate excludes a channel
        # whose R1 rests on *this* model's lower bound, so the gate and the
        # reported fit have to be the same model or the bound describes a number
        # nobody reads. Same value as the literal it replaces.
        report = analyze_spectrum(eis, model_name=ctx.plan.circuit_model)
        ctx.data_store.record_fit(measurement_id, report.fit, report=report)
    except Exception as exc:
        logger.warning("eis_validate_fit_failed", channel=channel, arm=arm,
                       error=str(exc))
    # ONE read, two consumers: the `conditions` row and the run's
    # `conditions.json` slot. `capture` performs the same
    # `read_environment(manager)` this line always did and publishes what it got,
    # so a watcher sees the rig without a single extra serial transaction on the
    # bus the next sweep is about to use.
    try:
        ctx.data_store.record_conditions(
            measurement_id, "measurement", **ctx.narration.capture(ctx.manager))
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
            # Counts, never results: which cell this is out of how many. What the
            # sweep FOUND is in the DataStore, which is the only thing that can
            # say what it means.
            ctx.narration.progress(PHASE_CELLS, index, total, channel=channel)
        except Exception as exc:
            ctx.consecutive_failures += 1
            logger.warning("eis_validate_cell_failed", channel=channel,
                           error=str(exc),
                           consecutive=ctx.consecutive_failures)
            print(f"  ! ch{channel} failed ({exc}); "
                  f"{ctx.consecutive_failures} consecutive", flush=True)
            ctx.narration.progress(PHASE_CELLS, index, total, channel=channel,
                                   failed=True,
                                   consecutive=ctx.consecutive_failures)
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
    rechecks = channels[: max(0, int(ctx.plan.drift_check))]
    for index, channel in enumerate(rechecks, start=1):
        if ctx.watch is not None:
            ctx.watch.poll()
        try:
            eis = measure_reference(ctx, channel, ARM_REFERENCE_END)
            print(f"[drift ] ch{channel}  {ctx.plan.reference_preset:<10} "
                  f"{eis.measurement_time_s:6.1f} s", flush=True)
            ctx.narration.progress(PHASE_DRIFT, index, len(rechecks),
                                   channel=channel)
        except Exception as exc:
            logger.warning("eis_validate_drift_failed", channel=channel,
                           error=str(exc))
            ctx.narration.progress(PHASE_DRIFT, index, len(rechecks),
                                   channel=channel, failed=True)


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
    add_verbosity_flag(parser)
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
    # Spelled exactly as `softae.tools.equilibration` spells it, because it is
    # the same quantity feeding the same `SettleTracker`; two tools spelling one
    # concept differently is how an operator learns that a flag means whatever
    # the tool feels like. The help carries the UNIT, loudly: a comment in
    # `settle_phase` records an operator reading `spread 0.130` as %RH and
    # nearly typing 1.0 here.
    run.add_argument("--settle-tol-rel", dest="settle_tol_rel", type=float,
                     default=DEFAULT_SETTLE_TOL_REL,
                     help=f"the settle band, as a RELATIVE DEVIATION OF SIGMA "
                          f"from its own window mean -- dimensionless, and NOT "
                          f"%%RH. 0.20 means 20%%: a cell whose conductivity "
                          f"swings +/-20%% across the judged window still "
                          f"certifies as settled, and 1.0 would accept a film "
                          f"whose conductivity doubled between rounds. Default "
                          f"{DEFAULT_SETTLE_TOL_REL:g}; the run REFUSES above "
                          f"{SETTLE_TOL_REL_MAX:g}. It must also exceed this "
                          f"board's own noise floor or no hold length can "
                          f"satisfy it -- the run says so, and names a workable "
                          f"value, at its first judged window")
    # Which of the two SIBLING criteria routes. `rate_check` is not a mode of
    # `settle_check` -- they estimate different things -- so this selects the one
    # that decides, and `both` runs the rate as a SHADOW beside the shipped
    # criterion. That middle word is the whole reason this flag exists: the rate
    # criterion has never been measured against a real board, and a criterion
    # given routing power on the strength of an argument is how the current one
    # got here.
    run.add_argument("--settle-criterion",
                     choices=("deviation", "rate", "both"),
                     default=DEFAULT_SETTLE_CRITERION,
                     help="which gate DECIDES. 'deviation' is shipped: "
                          "max|sigma - mean|/|mean| against --settle-tol-rel, "
                          "which for a 3-round window is a scatter estimate "
                          "compared against a drift tolerance. 'rate' regresses "
                          "ln sigma on t per cell and gates on a 95%% upper "
                          "bound of the slope, so a cell that is MOVING is told "
                          "apart from one that is merely too noisy to judge. "
                          "'both' routes on deviation and reports the rate -- "
                          "the shadow mode, and the only honest way to get the "
                          "comparison a cutover needs")
    # DECADES per hour, not ln-units per hour, and the name carries it. Three
    # reasons: H3 -- the criterion this gate must eventually GUARANTEE -- is
    # already in decades (`H3_MAX_HOLD_DRIFT_DEC = 0.05`); the spec derives this
    # number from it as `0.05 dec / T_meas`, so an operator who computes it
    # computes decades; and ln-units are the arithmetic's internal unit, which is
    # not a thing anyone should have to type. The conversion happens once, in
    # `rate_tol_ln_per_hour`. `--settle-tol-rel`'s help is the model for the
    # loudness: a comment in `settle_phase` records an operator reading
    # `spread 0.130` as %RH.
    run.add_argument("--settle-rate-tol-dec-per-h", dest="settle_rate_tol_dec_per_h",
                     type=float, default=DEFAULT_SETTLE_RATE_TOL_DEC_PER_H,
                     help="the rate band, in DECADES OF SIGMA PER HOUR -- not a "
                          "relative deviation, and not %%RH. 0.025 means a cell "
                          "moving 0.025 decades in an hour still certifies as "
                          "still. Derive it from H3: 0.05 dec / (measurement "
                          "block hours), so a 2 h block wants 0.025. REQUIRED "
                          "by --settle-criterion rate and both; the run REFUSES "
                          f"above {SETTLE_RATE_TOL_DEC_PER_H_MAX:g} dec/h, which "
                          "would certify a cell whose conductivity changes "
                          "threefold every hour")
    # Off by default, and it stays off unless it is typed: it changes what the
    # run is allowed to conclude, not merely how long it waits.
    run.add_argument("--survivors", choices=("off", "on"), default="off",
                     help="at the ceiling, PARTITION instead of refusing: "
                          "proceed on the cells the rate criterion certified "
                          "quiet, and record every dropped cell with its "
                          "reason. Cells PROVEN to be moving still refuse -- "
                          "only cells that could not be JUDGED are droppable. "
                          "Needs --settle-criterion rate or both. Every result "
                          "a survivor run produces is CONDITIONAL ON SETTLING, "
                          "which is fine for 'does the scout resolve the arc?' "
                          "and wrong for any hold-time or objective number")
    # `--soak-h`, not `--settle-min-hold-s`. Three reasons, in the order they
    # decided it. (1) The name is already taken: `settle_phase` has a
    # `min_hold_first_s` parameter, sourced from the shipped
    # `DEFAULT_MIN_HOLD_FIRST_S`, and it *is* the settle gate's minimum hold --
    # a floor below which `settled` will not be declared, with Quick rounds
    # sweeping throughout. A flag named `--settle-min-hold-s` would name that
    # parameter and mean something else. (2) The two are different quantities:
    # the settle gate's criterion is rig stability, and this is the sample's
    # equilibration, which no criterion on this rig can observe. (3) HOURS, not
    # seconds, against the file's own `_s` convention -- deliberately. This knob
    # is set in hours, and of the two 60x slips only one is caught: entering
    # 14400 for minutes shows up in the projection as 240 h and the operator
    # declines, while entering 2 for hours-as-seconds silently produces a run
    # with no soak that looks exactly like a correct one.
    run.add_argument("--soak-h", type=float, default=DEFAULT_SOAK_S / 3600.0,
                     help="HOURS to hold the established condition before the "
                          "first spectrum. Default 0. The settle gate proves "
                          "the RIG stopped moving; the soak is the SAMPLE's own "
                          "equilibration. Settle time counts against it")
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

    for subparser in (run, report):
        add_verbosity_flag(subparser)
    return parser


def _rig_claim(manager: Any, plan: ValidationPlan, run_id: str,
               log_path: str = "") -> Any:
    """Hold the rig while this process holds the ports -- or, under ``--mock``,
    hold nothing at all.

    ``log_path`` **now names this run's directory**, and the objection that kept
    it empty is what changed rather than the argument.
    :func:`~softae.core.rig_session.claim_rig_session` leaves the field empty by
    default because it advertises a run directory carrying an ``events.jsonl``
    stream, and offering one that holds some *other* run's stream -- a live lock
    plus a present file does not make the file the lock's -- is a lie. This
    harness published no stream at all, so the same objection applied here in its
    strongest form. It now publishes its own, into its own run directory, opened
    **before** the claim is taken; the field is what lets a watcher discover the
    run at all, so leaving it blank would now be withholding rather than
    caution. It falls back to empty when the stream could not be opened -- see
    :attr:`~softae.tools.eis_validate_narrate.RunNarration.log_path`.

    **Why the ``--mock`` gate is here and not left to ``held_rig_session``.**
    ``tools/env_hold.py``, the tool this follows, passes its manager in
    unconditionally on the stated grounds that ``held_rig_session`` "skips the
    claim entirely when every driver is a mock, so ``--mock`` claims nothing and
    cannot lock out a real run". That invariant is true there and **false here**,
    measured rather than assumed: ``session_is_simulated`` recognises a mock by
    the ``Mock`` prefix on its class name, and ``--mock`` in this tool swaps in
    :mod:`~softae.tools.eis_validate_mock`'s ``GridAwareMockPico``,
    ``FastMockTempController`` and ``FastMockRHController`` -- legitimately-named
    subclasses of the shipped mocks, none of which carries that prefix. An
    unconditional claim would therefore read a fully simulated manager as
    **real** and take ``~/.softae/rig.lock`` for a run that touches no hardware:
    precisely the "a mock run holding the rig turns a dry run into an outage for
    a real one" the exemption exists to prevent, and it would refuse the
    operator's GUI on the way.

    Repairing the predicate is the better fix and it belongs to
    ``core/rig_session.py``, which this file may import and must not edit. So the
    divergence is stated here and reported there, not patched around silently.
    """
    if plan.mock:
        return contextlib.nullcontext()
    from softae.core.rig_session import held_rig_session

    return held_rig_session(manager, what=f"{CLAIM_KIND}:{run_id}",
                            log_path=log_path)


def cmd_run(args: argparse.Namespace) -> int:
    from softae.core.data_store import DataStore
    from softae.core.eis_scout_scripts import ScoutPlanner
    from softae.core.hardware_safety import assert_hardware_armed
    from softae.core.run_lock import RunLockHeld, busy_rig_message, foreign_run_lock
    from softae.core.safe_park import (
        RH_DEADMAN_S,
        dry_purge_humidifier,
        safe_park,
    )
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

    # Ask who holds the rig **before the store is opened**, which is
    # `tools/env_hold.py`'s ordering and `tools/campaign.py`'s before it: a
    # refusal over hardware this run never touched must leave nothing behind.
    # That matters more here than there, because `_enter_run` does not only start
    # a row -- it writes the campaign checkpoint, and a non-`--resume` invocation
    # REPLACES it. A refusal taken one step later would destroy a validation's
    # existing resume point on its way to saying "the rig is busy". Asking before
    # `_confirm` for the same reason in the operator's currency: nobody should
    # read a nine-hour projection and type "yes" into a run that is already lost.
    #
    # The residual race is accepted, not closed: a holder arriving between this
    # peek and the claim below still raises `RunLockHeld`, and that path
    # finalizes its row. `acquire_run_lock`'s exclusive create is what makes the
    # claim safe; the peek only keeps the common refusal free of side effects.
    #
    # `foreign_run_lock`, never `read_run_lock` -- the latter reports this
    # process's own claim as a holder, which is how the Calibration Launcher came
    # to refuse itself once the GUI started claiming.
    if not plan.mock:
        holder = foreign_run_lock()
        if holder is not None:
            print(f"\nREFUSING TO START: "
                  f"{busy_rig_message(holder, action='This validation')}")
            return EXIT_FAILED

    if not args.mock and not _confirm(plan, projection, assume_yes=args.yes):
        return EXIT_DECLINED

    store = DataStore(resolve_project(args.project))
    # Rebound the moment `_enter_run` yields a run_id. Until then there is no row
    # of ours to close: a `--resume` whose fingerprint has moved raises before one
    # is adopted, and stamping the *previous* plan's row would be a lie about a
    # run this process never entered.
    finalize = _no_run_to_finalize
    # The claim is entered part-way through the `try` -- its `what` needs a run id
    # that does not exist at the head of the block -- and given back in the
    # `finally` **after** `disconnect_all`. `core.rig_session`'s rule is "acquire
    # when the ports open, release when they close", and a claim dropped before
    # the park would leave another process free to connect on top of a park still
    # in progress. An `ExitStack` is what lets one lexical block own a lifetime it
    # cannot open at its own head; `close()` on a stack that never entered
    # anything -- the `RunLockHeld` path -- is a no-op.
    claim = contextlib.ExitStack()
    # Inert until `_enter_run` yields a run directory to write into -- the same
    # null-object shape as `_no_run_to_finalize` above, and for the same reason:
    # the `except`/`finally` arms are reachable before there is anything to
    # narrate, and a `None` there is an AttributeError inside a harness that
    # drives a heater.
    narration: Any = RunNarration(Path(store.project_dir))
    # Bound before the `try` for the same reason: the `finally` reads it, and
    # `_enter_run` is the first statement that can raise.
    ctx: RunContext | None = None
    # How this run ended, as the stream will say it. Set by whichever arm below
    # wins and emitted **once**, in the `finally`, so `run_finished` is the last
    # record on every path -- including the ones no `except` names -- and lands
    # after the park rather than before it.
    outcome: dict[str, Any] = {"status": "error"}
    try:
        import asyncio

        # `_enter_run` **before** the claim, and so before `connect_all`. It needs
        # no hardware: it starts (or re-enters) the `experiments` row and writes
        # the resume checkpoint, both pure DataStore work, and it touches
        # `manager` only to hand it to the `RunContext`. Running it first is what
        # makes the claim's run id available before a single port is opened, and
        # it moves `--resume`'s `ResumeMismatch` *earlier*: it now raises before
        # anything is claimed or connected, which is stricter than before, never
        # looser.
        ctx = _enter_run(store, manager, plan, args)
        finalize = run_finalizer(store, ctx.run_id)
        # The stream opens **first**, and that ordering is the whole reason the
        # claim below may name a `log_path` at all: by the time the lock file is
        # written, the directory it advertises already holds THIS run's
        # `events.jsonl` with `run_started` in it. Entered on the same stack, and
        # first, so it is closed LAST -- after the park, after the disconnect and
        # after the claim is given back, which is what lets `run_finished` be a
        # true statement rather than an optimistic one.
        narration = claim.enter_context(open_narration(ctx.run_dir))
        ctx.narration = narration
        narration.record(
            "run_started", run_id=ctx.run_id, validation=plan.validation_name,
            channels=len(plan.channels), rh_setpoint_pct=plan.rh_setpoint_pct,
            temp_setpoint_C=plan.temp_setpoint_c, soak_s=plan.soak_s,
            hold_epoch=ctx.hold_epoch, resume=bool(args.resume),
            mock=bool(plan.mock))
        claim.enter_context(
            _rig_claim(manager, plan, ctx.run_id, narration.log_path))
        asyncio.run(manager.connect_all())
        remaining = _remaining_channels(store, plan, resume=bool(args.resume))
        if not remaining:
            print("Every planned cell is already complete; nothing to measure.")
            outcome = {"status": "done", "reason": "nothing to measure"}
            finalize("done")
            return EXIT_OK

        # `ctx.watch` is now built inside `_establish_condition`: the soak needs
        # it before the first sweep, and a watch created afterwards would have no
        # series across the hours it was meant to be watching.
        _establish_condition(ctx, plan, remaining)
        planner = ScoutPlanner(site="validation", actuate=True)
        narration.state(PHASE_CELLS, channels=len(remaining))
        run_cells(ctx, planner, remaining)
        narration.state(PHASE_DRIFT, channels=plan.drift_check)
        drift_check(ctx, remaining)
        narration.state(PHASE_REPORT)
        _write_report(store, plan, args)
        outcome = {"status": "done"}
        finalize("done")
        return EXIT_OK
    except RunLockHeld as held:
        # The peek above makes this the race rather than the routine case: a
        # holder that arrived in the moment between asking and claiming. Refused
        # with the harness's own vocabulary, and `aborted` for the same reason the
        # arm below is -- the harness declined; nothing was interrupted. The row
        # exists by this point, so closing it is not optional: an unfinished row
        # is byte-for-byte what a crash looks like.
        outcome = {"status": "aborted", "reason": "rig claimed by another process"}
        finalize("aborted")
        print(f"\nREFUSING TO START: "
              f"{busy_rig_message(held.lock, action='This validation')}")
        return EXIT_FAILED
    except RefuseToStart as exc:
        # A refusal is a decision, not an accident: the harness declined to spend
        # the measurement block. `aborted`, never `interrupted` -- nothing was
        # interrupted, and this row is the only place the distinction survives.
        outcome = {"status": "aborted", "reason": str(exc)}
        finalize("aborted")
        print(f"\nREFUSING TO START: {exc}")
        return EXIT_FAILED
    except KeyboardInterrupt:
        outcome = {"status": "interrupted"}
        finalize("interrupted")
        print("\nInterrupted. Recorded rows stand; nothing new was started.")
        return EXIT_INTERRUPTED
    except Exception as exc:
        outcome = {"status": "error", "reason": str(exc)}
        finalize("error")
        logger.error("eis_validate_run_failed", error=str(exc))
        print(f"\nRUN FAILED: {exc}")
        return EXIT_FAILED
    finally:
        # The catch-all, and it runs before the park: an exception no `except`
        # above names must not be the reason the row stays open, and the park
        # below is the longest thing left in the process. Idempotent, so it is a
        # no-op on every path above.
        finalize("error")
        # Every exit path, success included. `retract_head=None` because absent
        # an operator the correct response to an unknown is to add no motion to
        # it -- `safe_park`'s default is reversed for exactly this caller class.
        narration.state(PHASE_PARK, end_state=plan.end_state)
        if plan.end_state == "park":
            # The humidifier's end state is no longer this caller's to choose,
            # and no longer depends on how the run ended. `safe_park` dry-purges
            # unconditionally -- operator ruling 2026-08-24, recorded in
            # `core/safe_park.py` along with the opt-in rule it reversed: dry gas
            # carries very little volatile species, so the flow is not the hazard
            # the old rule took it for, while duty 0 *is* the firmware's
            # auto-shutoff and hands a chamber the run spent an hour drying back
            # to room RH in tens of seconds.
            #
            # So the `orderly`/fault split this call used to make is gone. What
            # changed here in behaviour is the FAULT path: a failed run used to
            # zero the humidifier and now dry-purges like every other exit.
            #
            # A park still does NOT preserve the condition -- the heater goes to
            # 10 C and `--resume` re-equilibrates from scratch, which is what the
            # line below has always said and still says.
            result = safe_park(manager, reason="eis validation complete",
                               retract_head=None)
            print(f"\n[park  ] {result.summary()}")
            print("         A PARK ENDS THE CONDITION: the heater is driven to "
                  "10 C and anti-clog purging is suspended. --resume "
                  "re-equilibrates from scratch.")
            # `park` is the campaign's own record for exactly this act, and the
            # park is the longest thing left in the process -- so a watcher that
            # sees it knows the measurement block is over and the rig is on its
            # way to safe, rather than inferring it from a silence.
            # No `rh_dry_purge` key. It used to report a *request* -- a decision
            # this branch made and a watcher could not otherwise see. There is no
            # request any more, and the field cannot be repurposed to report
            # success the way the `hold` branch's can: `result.commanded` here
            # spans pumps, heater and lamp as well, so "commanded and no errors"
            # would mean *the whole park* succeeded, which is what `ok` already
            # says. An always-true field would be worse than an absent one.
            narration.record("park", reason="eis validation complete",
                             ok=bool(getattr(result, "ok", True)))
        else:
            # The heater is deliberately untouched -- that is what the flag buys,
            # and temperature genuinely holds because the controller keeps its own
            # setpoint. The humidifier is NOT left alone, because it cannot be:
            # `disconnect_all()` below reaches `AsyncRHController.disconnect` ->
            # `_stop_pid_loop()`, whose thread-exit write commands the exit duty,
            # and that default is 0.0 -- the firmware's auto-shutoff, closing both
            # valves. "Still driven" was therefore never an available end state on
            # this path; the only choice was valves shut now versus dry air
            # flowing for the deadman window, and `dry_purge_humidifier` takes the
            # second. It stops the loop itself, so the `_stop_pid_loop()` inside
            # `disconnect()` finds `_running` False and writes nothing over it.
            result = dry_purge_humidifier(
                manager, reason="eis validation --end-state hold")
            print(f"\n[hold  ] {result.summary()}")
            # The graded lines themselves, not a paraphrase: `DRY_PURGE_COMMANDED`
            # already names the duty and what closes the valves, and a degenerate
            # `out_min` -- the one case where "commanded dry" would be a lie --
            # arrives here as an error rather than being silently absent.
            for line in (*result.commanded, *result.errors, *result.skipped):
                print(f"         {line}")
            print(HOLD_NOTICE.format(rh=plan.rh_setpoint_pct,
                                     deadman=RH_DEADMAN_S))
            # `rh_dry_purge` is KEPT here, and it survived the ruling because it
            # never reported a request: it reports whether the dry purge actually
            # LANDED, which is a different question and still a varying one -- a
            # degenerate `out_min`, a dead transport or a failed write each turn
            # it False. `dry_purge_humidifier` touches only the humidifier, so
            # "commanded and no errors" means exactly that here and nothing
            # wider. `ok=False` is about the *hold*, not about the purge, which
            # is why this cannot be folded into it.
            narration.record("park", reason="--end-state hold: not parked",
                             ok=False, held=True,
                             rh_dry_purge=bool(result.commanded
                                               and not result.errors))
        try:
            import asyncio

            asyncio.run(manager.disconnect_all())
        except Exception:                                 # pragma: no cover
            pass
        # After the disconnect, so it is the truth rather than a prediction, and
        # before `claim.close()` -- which unwinds the narration itself.
        narration.state(PHASE_FINISHED)
        narration.record("run_finished", status=outcome.get("status", "error"),
                         n_recorded=int(getattr(ctx, "n_recorded", 0)),
                         **{k: v for k, v in outcome.items() if k != "status"})
        # Last, and deliberately after the disconnect: the claim outlives the
        # ports it was taken for, so no other process can open them in the gap
        # between the park and the close. The narration was entered *before* the
        # claim, so this same call closes the stream last of all.
        claim.close()


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
    disclosures = [(
        "hours above cover APPROACH + SETTLE only; the measurement block adds",
        "0 min",
        f"{projection.measurement_low_s / 60:.0f} - "
        f"{projection.measurement_high_s / 60:.0f} min",
    )]
    # Disclosed here as well as in the projection, because `confirm_thermal`
    # builds its own hours from an `EquilibrationConfig`, which has no soak in
    # it -- so on the one screen where the operator commits, an undisclosed soak
    # would be hours the banner's own number silently omits.
    if plan.soak_s > 0:
        disclosures.append((
            "and a SOAK is held at condition before the first sweep, bounded at",
            f"{plan.soak_s / 3600:.2f} h",
            f"{plan.soak_s * SOAK_CEILING_FACTOR / 3600:.2f} h",
        ))
    return confirm_thermal(config, assume_yes=assume_yes,
                           plan_overrides=disclosures)


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
    # `loop_state` is deliberately left at "running" on every exit, and it is not
    # a second unclosed liveness claim like the run row was. This checkpoint is a
    # *resume point*: `--resume` re-enters it after a park, so "not finished" is
    # true of it for as long as it exists, and the row that answers "did this
    # process die" is `experiments` -- which `cmd_run` now closes. Nothing reads
    # this column for liveness (`unfinished_runs` does not see this table), and
    # settling it would mean a second whole-row INSERT OR REPLACE on the exit
    # path, risking the resume point of a nine-hour run to update a field no
    # reader consults. The schema's terminal move is `clear_campaign_checkpoint`,
    # and clearing is what destroys resumability.
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
    """Approach, settle, arc-capture watch, soak -- then and only then, measure.

    Under ``--mock`` the *pacing* is collapsed and nothing else is: the same
    ``approach_setpoint``, the same ``SettleTracker``, the same verdicts and the
    same refusals run, but the waits between polls and between rounds go to
    zero. A 25-minute minimum hold is a statement about a real thermal mass, and
    there isn't one; keeping it would only mean the mock path never gets
    exercised. The soak inherits the same clock, so ``--mock --soak-h 6``
    exercises every branch of :func:`~softae.tools.eis_validate_hold.soak_phase`
    in milliseconds; a test that really slept for a soak would be a defect.

    **The approach overlap is degenerate under ``--mock``, and is exercised
    anyway.** Collapsed pacing means the humidifier gets no real head start --
    ``FastMockRHController`` advances per *read*, and nothing reads RH during the
    temperature approach -- so a mock run cannot demonstrate the *saving*. What
    it can and does demonstrate is the *sequence*: the same early
    ``set_setpoint``/``start`` runs, on the same code path, and the virtual clock
    the mock hands down is what makes ``ApproachReport.lead_s`` non-zero there,
    so the ordering and the reported lead are both pinned by tests rather than
    taken on trust. Nothing about the overlap is branched on ``plan.mock``.

    The soak sits **after** the min-treatment refusal, not before it. Both are
    gates on the same first spectrum, and the free one goes first: refusing on a
    setpoint that projects too few TREATMENT cells costs nothing, while doing it
    on the far side of a six-hour soak spends the soak to learn something that
    was knowable before it started.
    """
    from softae.analysis.equilibration import LN_PER_DECADE
    from softae.tools.eis_validate_hold import survivor_row_stamp

    clock = VirtualClock() if plan.mock else None
    pacing: dict[str, Any] = (
        {"sleep": clock.sleep, "now": clock} if clock else {}
    )
    narration = ctx.narration
    narration.state(PHASE_APPROACH, rh_setpoint_pct=plan.rh_setpoint_pct,
                    temp_setpoint_C=plan.temp_setpoint_c)
    reports = approach_condition(ctx.manager, plan,
                                 on_command=_approach_command_observer(ctx),
                                 **pacing)
    for index, report in enumerate(reports, start=1):
        # `elapsed_s` is the JUDGED window -- the one the timeout bounds -- and
        # the lead is printed beside it rather than folded into it, because the
        # operator reads this line to decide whether the axis is slow. An RH row
        # reading "0.3 min" with no lead would look like a chamber that dries in
        # twenty seconds.
        lead = (f"  (+{report.lead_s / 60:.1f} min commanded during the heat)"
                if report.lead_s > 0 else "")
        print(f"[approach] {report.axis:<12} -> {report.target:g}  "
              f"PV {report.pv_final:g}  {report.elapsed_s / 60:.1f} min{lead}")
        # The axis and how long it took, not the PV it reached. A PV is a
        # reading, and readings belong in `conditions` rows and in
        # `conditions.json` -- both of which this run already writes.
        narration.progress(PHASE_APPROACH, index, len(reports),
                           axis=report.axis, elapsed_s=round(report.elapsed_s, 1),
                           lead_s=round(report.lead_s, 1),
                           attempts=report.attempts)

    # THE MOMENT THE CONDITION EXISTS, and so the moment the soak clock starts.
    # Everything before this line is an approach: the chamber was on its way to
    # a setpoint the sample had not yet seen. Everything after it -- the settle
    # rounds included -- is time the film is actually spending at the new RH.
    established_at = float(clock() if clock else time.monotonic())

    settle_pacing: dict[str, Any] = (
        {"sleep": clock.sleep, "now": clock, "min_hold_first_s": 0.0}
        if clock else {}
    )
    narration.state(PHASE_SETTLE, max_hold_s=plan.settle_max_hold_s)
    outcome = settle_phase(ctx.manager, plan, lambda ch: _settle_sweep(ctx, ch),
                           on_round=_settle_observer(ctx), **settle_pacing)
    # `settle_verdict` is `run_autonomous_campaign`'s own record for this, and
    # what rides on it is the gate's answer -- did the rig stop moving, over how
    # many rounds, in how long.
    narration.record("settle_verdict", verdict=outcome.verdict,
                     rounds=outcome.n_rounds,
                     elapsed_s=round(outcome.elapsed_s, 1))
    # The BAND each cell projects onto, and not the apex frequency that decided
    # it. The band is gate state -- it is the quantity `--min-treatment` refuses
    # on, and it is the answer to "why did this run stop", which is what a
    # stream is for. The apex in Hz is a reading taken off a PRE-EQUILIBRATION
    # sweep this harness deliberately does not persist, and putting it here
    # would invite exactly the analysis the settle gate exists to prevent.
    narration.record("settle_bands", counts=dict(outcome.projected),
                     by_channel={str(channel): band for channel, band
                                 in band_by_channel(outcome.apex_by_channel,
                                                    plan).items()})
    # BEFORE the refusal, not after it. The histogram is built from sweeps
    # already taken, at zero extra cost, and it is the single most useful thing
    # to know when settle fails -- it says whether the cells are even in the
    # resolving window. Printed only on success, it was missing from both of the
    # 2026-08-20 runs that died in this gate.
    # The partition, and the cells it could not speak for. Recorded whenever the
    # gate computed a rate at all -- under `--survivors off` too -- because the
    # denominator is what neither 2026-08-21 run could reconstruct afterwards,
    # and recording is not routing. `settle_survivors` is a separate record from
    # `settle_verdict` because it is a claim about WHICH cells, where the verdict
    # is a claim about the board.
    if outcome.survivors or outcome.dropped:
        narration.record(
            "settle_survivors", mode="on" if plan.survivors else "off",
            criterion=plan.settle_criterion,
            survivors=list(outcome.survivors),
            dropped={str(channel): why
                     for channel, why in sorted(outcome.dropped.items())},
            survivor_counts=dict(outcome.survivor_projected),
            pooled_rate_dec_per_h=(
                None if outcome.pooled_rate_per_hour is None
                else round(outcome.pooled_rate_per_hour / LN_PER_DECADE, 6)),
            floors_ok=not outcome.survivor_refusal,
            floor_refusal=outcome.survivor_refusal)
    print(render_arc_watch(outcome, plan))
    assert_settle_licensed(outcome)
    ctx.hold_certified = outcome.verdict
    # Keyed on the FLAG and not on the verdict. Under `--survivors on` the rate
    # criterion can return `settled` while cells it could not judge sit in the
    # board -- it certifies on the quiet ones -- and those cells' rows would then
    # carry `hold_certified="settled"`, which is false for them and is silent
    # survivorship. Whenever the operator asked for a partition, the partition is
    # what the rows record.
    if plan.survivors:
        ctx.dropped = {channel: survivor_row_stamp(why)
                       for channel, why in outcome.dropped.items()}

    # AFTER the drop, never before. `outcome.projected` counts the whole board,
    # and a board that clears --min-treatment says nothing about the survivor set
    # that will actually be certified -- so under a partition the denominator is
    # the survivors. This floor stays here, with the flag that owns it, rather
    # than moving into `survivor_floors`: one owner per floor.
    partitioned = bool(plan.survivors)
    census = outcome.survivor_projected if partitioned else outcome.projected
    projected = census.get(TREATMENT, 0)
    if plan.settle and projected < plan.min_treatment:
        scope = ("channel(s) SURVIVED into TREATMENT" if partitioned
                 else "channel(s) project into TREATMENT")
        raise RefuseToStart(
            f"only {projected} {scope}, below "
            f"--min-treatment {plan.min_treatment}. The resolving window is "
            f"[{plan.ref_close_hz:.2f}, {plan.baseline_ok_hz:.2f}) Hz and the "
            "SETPOINT is the lever, not the sample size: change the condition, "
            "or widen the reference with --reference-preset longest (2.1x the "
            "window for 4.3x the reference cost). Stopping now rather than "
            "spending the measurement block to report INSUFFICIENT."
        )

    # The watch is built here rather than in `cmd_run` so the soak has something
    # to watch, and so its series begins at the condition rather than at the
    # first sweep. Under `--mock` it shares the virtual clock: a series stamped
    # from the wall while the soak advances a virtual clock would collapse every
    # grace window to zero span and make a sustained excursion unobservable.
    ctx.watch = HoldWatch(manager=ctx.manager, plan=plan,
                          **({"now": clock} if clock else {}))
    narration.state(PHASE_SOAK, target_s=plan.soak_s,
                    credit_s=round(max(0.0, (clock() if clock else time.monotonic())
                                       - established_at), 1))
    soak_phase(plan, ctx.watch, established_at=established_at,
               on_poll=_soak_observer(ctx), on_restart=_soak_restart(ctx),
               **pacing)


def _approach_command_observer(ctx: RunContext) -> Any:
    """Publish each setpoint WRITE, distinctly from the arrival it precedes.

    The RH loop is commanded before the temperature approach rather than after
    it, and that is a fact about what the rig is *doing* that no other record
    carries: the ``progress`` record for the RH axis is emitted when RH
    **arrives**, which on a chamber that dries well is minutes after the heat and
    an hour or more after the humidifier started. Between the two there would
    otherwise be nothing in the stream saying the loop was live, and a watcher
    would read the gap as an idle humidifier.

    ``target`` and not a PV, matching the ``progress`` records' own rule: this is
    what was *commanded*, which is an instruction rather than a reading.
    """
    def _observer(axis: str, target: float) -> None:
        ctx.narration.record("approach_commanded", axis=str(axis),
                             target=float(target),
                             judged_after_temperature=(str(axis) != "temperature"))

    return _observer


def _settle_observer(ctx: RunContext) -> Any:
    """Put each settle round on disk, where console scrollback is not.

    Two runs (2026-08-20) spent 1 h 44 m and 64 min in this gate and acquired
    nothing, and the ``eis/`` directory was empty afterwards because settle
    sweeps are deliberately not persisted. *Which* channel was at 0.48 existed
    in exactly one place -- the terminal -- so the question could not be asked
    again. One record per round answers it.

    **This is gate state, not scientific record**, and the distinction is where
    the stream's own rule is drawn. ``campaign_events`` excludes measurements
    because "every scientific fact in the stream is already in a table or a
    sidecar"; for these rounds that premise is simply false, and the deviations
    recorded here are *ratios of a channel to itself* -- dimensionless,
    geometry-free, and not invertible to the sigma they came from. The sigma
    itself stays out: it is the observable, it has units, and admitting it would
    make this file a second, schema-less measurement record.

    ``progress`` rather than ``record`` is deliberately not used: a settle phase
    has no denominator. It runs until the material stops moving or until the
    ceiling, and ``done/total`` would be a completion claim the gate cannot make.
    """
    def observe(payload: dict[str, Any]) -> None:
        ctx.narration.record("settle_round", **payload)
    return observe


def _soak_observer(ctx: RunContext) -> Any:
    """Publish the rig, and narrate the wait, from inside the soak.

    The soak is this run's longest silence and exactly what an operator at hour
    two is asking about -- and it is also the only long phase with an **idle
    bus**, because no sweep runs during it. ``HoldWatch`` is already polling both
    controllers every 30 s here, so a :func:`~softae.core.conditions_capture.read_environment`
    on the same cadence adds five Modbus reads per 30 s to a bus with nothing
    else on it: the same order as what the watch already spends there. Nothing
    comparable is added anywhere a sweep is in flight -- ``run_cells`` and
    ``drift_check`` publish only the capture :func:`persist` was already taking.

    The ``progress`` record rides the **console's** cadence rather than the
    poll's, so the stream and the terminal say the same thing at the same
    moments, and a four-hour soak costs ~48 records instead of ~480.
    """
    def observe(polls: int, soaked_s: float, target_s: float,
                restarts: int) -> None:
        ctx.narration.capture(ctx.manager)
        if polls % SOAK_PRINT_EVERY_N_POLLS == 0:
            ctx.narration.progress(PHASE_SOAK, int(soaked_s), int(target_s),
                                   unit="s", restarts=restarts)
    return observe


def _soak_restart(ctx: RunContext) -> Any:
    """Narrate an excursion that reset the continuity clock.

    Its own record rather than one more ``progress`` line, because it is the one
    thing in the soak that is not monotone: a watcher reading progress alone
    would see the count go backwards with no account of why, and *a soak whose
    clock restarted* is precisely what somebody checking at hour two needs to be
    told rather than left to infer.
    """
    def restarted(restart: int, lost_s: float, target_s: float) -> None:
        ctx.narration.record("soak_restart", restart=int(restart),
                             lost_s=round(float(lost_s), 1),
                             target_s=round(float(target_s), 1))
        # `soak_phase` skips the per-poll observer on a restart, and this is the
        # one poll whose conditions are most worth having: republished here so an
        # excursion is visible in the rig reading, not only in the narration.
        ctx.narration.capture(ctx.manager)
    return restarted


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
    # Before dispatch, so every subcommand is covered and there is exactly one
    # place the level is decided.
    configure_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
