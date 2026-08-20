"""The P.22 equilibration characterization run.

A CLI rather than a GUI tab, for the same reason ``softae-commission`` is one:
this is a **bench task**, and the run is an overnight one the operator starts and
walks away from. The four subcommands are in operator order, read-only first::

    python -m softae.tools.equilibration plan --save plan.toml
    python -m softae.tools.equilibration run --from-plan plan.toml --execute
    python -m softae.tools.equilibration fit  --run <run_id>
    python -m softae.tools.equilibration report --run <run_id> --tol-rel 0.02

Spelled as ``python -m`` throughout, and not as the ``softae-equilibration``
console script :data:`CONSOLE_SCRIPT` declares. That entry point is in
``pyproject.toml``, but a console script exists only once someone has run
``pip install -e .`` **since it was added** — it was absent from the working venv
when this tool was written and was generated on 2026-08-11 — so **whether it
resolves is a fact about the install, not about the tool**. The module form
resolves in either state, which is why everything printed here uses it.

``plan`` and ``run`` are separate process invocations **sharing no state**, so
every design flag not repeated on ``run`` silently reverts to its default. That
is not hypothetical: on 2026-08-10 it cost ~40 minutes of rig time and the whole
scientific result — ``--preset`` fell back to ``Standard`` (40.7 s/channel
measured, against ``Quick``'s then-10.47) and the electrode geometry was dropped
whole, so ``sigma_S_per_cm`` was NULL for all 41 fits while every log line
reported success.

Two things answer that. ``--channels`` is **mandatory** on ``run``, because a
defaulted channel set would energise the channels a subset deliberately
excluded. And ``plan --save`` writes the **fully resolved** design — every value
the run will use, defaults included — which ``run --from-plan`` executes
verbatim. A flag typed alongside ``--from-plan`` still wins, but only as a
printed diff against the file; a silent override would be the original defect
wearing a hat.

**Hardware safety.** ``run --execute`` drives the stage heater to the peak
temperature and actuates the humidifier for ~9–15 h. Nothing in the shipped
configuration refuses it: 85 °C is far inside ``[safety] temp_max_C = 200.0``,
and ``validate_rh_setpoint`` is a cap with no floor. Worse,
``assert_hardware_armed`` is a **no-op on this run** — ``MOTION_INSTRUMENTS`` is
``("stage", "syringe", "piezo")``, so ``probe_motion`` returns empty on an
EIS-plus-environment workflow and the assert passes unconditionally.

So the interlock is called for consistency and **is not relied on**. The real
gate is :func:`confirm_thermal`, which states the peak temperature, the projected
duration and the channels that will be driven, and requires the word ``yes`` —
not ``y``, because a reflex keypress is exactly what a nine-hour unattended
thermal run must not accept.

**Progress.** The run emits events; :class:`ProgressRenderer` here owns every
character. It is TTY-aware for a reason that is not cosmetic: a 15 h run is
overwhelmingly likely to be started as ``> run.log``, and an in-place redraw at
poll cadence writes megabytes of carriage returns into that file. On a terminal
it redraws one status line; redirected, it emits periodic milestone lines and no
control characters at all.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import structlog

from softae.analysis.eis.geometry import THICKNESS_METHODS
from softae.analysis.equilibration import (
    DEFAULT_MIN_HOLD_FIRST_S,
    DEFAULT_MIN_HOLD_S,
    DEFAULT_N_SETTLE,
    DEFAULT_SETTLE_MIN_CHANNELS,
    DEFAULT_SETTLE_N_ROUNDS,
    DEFAULT_SETTLE_TOL_REL,
    DEFAULT_TOL_REL,
    EQUILIBRATION_MODELS,
    MIN_POINTS_FOR_TAU,
    R1_AGREEMENT_TOL_REL,
    arc_closure_rates,
    endorse_tolerance,
    fit_run,
    load_sigma_series,
    session_drift,
)
from softae.core.channel_spec import (
    ChannelSpecError,
    format_channel_spec,
    parse_channel_spec,
)
from softae.core.hardware_safety import ARM_ENV_VAR, HardwareNotArmedError
from softae.tools import run_finalizer, use_utf8_console
from softae.workflows.equilibration import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_DOWN_APPROACH_TIMEOUT_S,
    DEFAULT_EIS_PRESET,
    DEFAULT_FAULT_C,
    DEFAULT_GRACE_S,
    DEFAULT_N_CHANNELS,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    DEFAULT_RH_SETPOINT_PCT,
    DEFAULT_RH_TOLERANCE_PCT,
    DEFAULT_ROUND_PERIOD_S,
    DEFAULT_TAU_SETPOINTS,
    DEFAULT_TOLERANCE_C,
    DEFAULT_WARN_C,
    ENV_ABSENT,
    ENV_SKIPPED,
    EV_AMBIENT_RESTORED,
    EV_APPROACH_FINISHED,
    EV_COST_WARNING,
    EV_HOLD_VERDICT,
    EV_LEG_FINISHED,
    EV_LEG_STARTED,
    EV_RUN_FINISHED,
    EV_RUN_STARTED,
    EV_SETPOINT_FINISHED,
    EV_SETPOINT_STARTED,
    EV_SETTLE_VERDICT,
    REFERENCE_GEOMETRY_CM,
    ROUND_BUFFER_S,
    VERDICT_ABORTED,
    VERDICT_MET,
    VERDICT_UNMET,
    EquilibrationAbort,
    EquilibrationConfig,
    EquilibrationRun,
    inter_round_gap_s,
    load_sidecar,
    minimum_feasible_period_s,
    model_underestimate_frac,
    project_duration,
    round_cost_s,
    round_headroom_s_per_channel,
)
from softae.core.preflight import EIS_MEASURED_S_PER_CHANNEL

logger = structlog.get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DECLINED = 2


#: The typed word that starts a nine-hour thermal run. Not "y".
CONFIRM_WORD = "yes"

#: The console-script name ``pyproject.toml [project.scripts]`` declares. Kept as
#: the parser's ``prog`` because that is what the entry point *is*, and used
#: nowhere else: whether it resolves depends on when the venv was last installed
#: from, so every command this tool **prints** must be the module form below,
#: which resolves either way.
CONSOLE_SCRIPT = "softae-equilibration"
#: Written as a literal, not as ``__name__``: run as ``python -m`` this module is
#: imported as ``__main__``, and a saved plan attributing itself to ``__main__``
#: names nothing a reader could go and open.
MODULE = "softae.tools.equilibration"
#: How this tool refers to itself in anything an operator is meant to type.
CLI = f"python -m {MODULE}"

#: Status-line budget. Deliberately narrow: the rig console is whatever window
#: happens to be open, and a line that wraps defeats an in-place redraw entirely.
STATUS_WIDTH = 78
#: How often a **redirected** run gets a progress line, and how often either kind
#: of console gets a full controls-telemetry line. Five minutes: often enough to
#: see a controller drift, sparse enough that a 15 h log stays readable.
DEFAULT_MILESTONE_INTERVAL_S = 300.0

#: What one channel of a ``Standard`` round **actually** cost on this rig, measured
#: over 12 channels: 40.7 s, and essentially constant channel to channel (40.687 /
#: 40.719 / 40.718 / 40.703 s on four consecutive channels). Preflight's own
#: ``Standard`` anchor, taken separately, is 40.85 s — the two agree to 0.4 %.
#:
#: This number is why the sweep model is trustworthy today. It used to model
#: ~3.9 s/channel here, roughly **ten times low**, and a plan resting on it told
#: an operator 240 s for a round that took 2166 s. The measured-vs-modelled
#: machinery throughout this module dates from the era of that gap and is kept
#: because a model that was wrong once can be wrong again, not because it is
#: currently wrong.
#:
#: **Derived, not restated.** This was the literal ``40.7`` until 2026-08-17. That
#: reading described the *pre-retune* ``Standard`` grid, and when the mains-notch
#: retune moved every preset the literal stayed behind — still quoting operators a
#: stopwatch, in ``--help`` and in the modelled-basis advice, for a sweep that no
#: longer existed. Reading it out of the anchor table makes the desync
#: unrepresentable: there is one bench number for ``Standard`` in this codebase and
#: this is a view of it.
#:
#: Quoted in ``--help`` and in the modelled-basis note so an operator has a
#: stopwatch number beside the model's. Still **not** used as a default: it belongs
#: to one rig and one preset, and a silently applied constant would be wrong the
#: moment either changed.
MEASURED_PER_CHANNEL_S_STANDARD = EIS_MEASURED_S_PER_CHANNEL["Standard"]

# The modelled-basis caution below is sized by
# `model_underestimate_frac`, imported from the workflow rather than restated here.
# It replaced a flat `OVERHEAD_HEADROOM_WARN_S = 10.0` s/channel, which reserved
# room for a fixed per-channel overhead the model was believed not to carry, back
# when the model ran ~10x low. Since the 2026-08 recalibration the residual is a
# two-sided ~8 % fit error that *scales with the sweep*, and a flat 10 s/channel
# would demand 89 % of `Quick`'s entire per-channel cost as reserve against 8 % of
# `Extended`'s -- cautioning on rounds that comfortably fit and going quiet on ones
# that do not.

#: What an unreadable, stale or NaN value looks like. Never ``0.0``, never the
#: last good number: ``AsyncRHController`` deliberately turns a held reading into
#: ``NaN`` past ``max_stale_s``, and flattening that into a plausible figure would
#: let a dead sensor read as a working one for nine hours.
NA = "--"

#: Events that earn a line of their own, on a terminal and in a log file alike.
_ANNOUNCE = frozenset({
    EV_RUN_STARTED, EV_LEG_STARTED, EV_SETPOINT_STARTED, EV_APPROACH_FINISHED,
    EV_SETPOINT_FINISHED, EV_LEG_FINISHED, EV_COST_WARNING, EV_SETTLE_VERDICT,
    EV_RUN_FINISHED, EV_AMBIENT_RESTORED,
})


def _open_store(args):
    """The project store, defaulting to the one the GUI and campaigns already use.

    ``--mock`` writes to an isolated ``<project>/mock`` store: a simulated σ(t)
    landing in the production store would let a later ``fit`` derive a
    conditioning hold time from synthetic data with nothing in the record to say
    so.
    """
    from softae.config import loader
    from softae.core.data_store import DataStore

    project = getattr(args, "project", None)
    if not project:
        project = loader.data_project_dir()
        if getattr(args, "mock", False):
            project = str(Path(project).expanduser() / "mock")
    return DataStore(project, db_filename=loader.data_db_filename()), project


# ── Progress rendering ───────────────────────────────────────────────────────

def hms(seconds: float) -> str:
    """``H:MM:SS`` — ASCII, fixed width, and readable at 3 a.m."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def reconciled_eta_s(elapsed_s: float, fraction: float,
                     projected_total_s: float) -> float:
    """Seconds remaining, from the projection **reconciled against what happened**.

    The projection alone is a config-derived upper/typical bound computed before
    anything was touched; by hour six it is the weaker of the two estimates,
    because the run has by then measured its own approach times. But early on the
    opposite holds — a fraction of 0.01 turns any timing jitter into a wild
    extrapolation.

    So the two are blended by the fraction itself: at 0 % the answer is the
    projection, at 100 % it is pure observation, and in between it slides. No
    tuning constant, and the crossover is where it should be.
    """
    total = max(0.0, float(projected_total_s))
    f = min(1.0, max(0.0, float(fraction)))
    if f > 0.0:
        total = f * (float(elapsed_s) / f) + (1.0 - f) * total
    return max(0.0, total - float(elapsed_s))


class ProgressRenderer:
    """Turns :class:`ProgressEvent` into console output, and nothing else.

    Kept out of the workflow deliberately: the run must be able to feed a GUI tab
    that draws none of this, and a renderer must be replaceable without opening a
    module that drives a heater.

    **It cannot break the run.** Every call is wrapped — a formatting bug, a
    closed pipe, a console that cannot encode a character — and degrades to
    silence plus a :attr:`failures` count. The run wraps it a second time; two
    guards, because the cost of the second is nothing and the cost of a crash at
    hour seven is the whole night.
    """

    def __init__(self, config: EquilibrationConfig, *, stream: Any = None,
                 isatty: bool | None = None, quiet: bool = False,
                 milestone_interval_s: float = DEFAULT_MILESTONE_INTERVAL_S) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.config = config
        self.quiet = bool(quiet)
        self.milestone_interval_s = float(milestone_interval_s)
        #: The modelled projection, kept for the whole run even after a measured
        #: one exists. The gap between them is the finding.
        self.projected_total_s = project_duration(config).typical_s
        #: Re-projected from what rounds actually cost, once that is known.
        self.measured_total_s: float | None = None
        self.tty = self._detect_tty() if isatty is None else bool(isatty)
        self.failures = 0
        self._live = False
        self._last_periodic_s = 0.0
        self._last_env_s: float | None = None
        self._measured_series_s: float | None = None

    def _detect_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def __call__(self, event: Any) -> None:
        try:
            self._render(event)
        except Exception:
            self.failures += 1

    # ── dispatch ─────────────────────────────────────────────────────────

    def _render(self, event: Any) -> None:
        # The controls monitor first, and independently of --quiet: it is the
        # whole reason a headless run does not need the GUI opened on it, and
        # `--quiet > run.log` is the canonical unattended invocation.
        self._maybe_telemetry(event)
        self._absorb_cost(event)
        if event.kind in _ANNOUNCE:
            self._line(self._milestone(event))
            return
        if event.kind == EV_HOLD_VERDICT:
            # Only the unmet ones. A met window is the expected case and there are
            # ~240 of them in a shipped run; announcing each would bury the
            # setpoint verdicts, which are the primary result.
            if event.verdict == VERDICT_UNMET:
                self._line(f"[{hms(event.elapsed_s)}]   ! {event.detail}")
            return
        if self.quiet:
            return
        if self.tty:
            self._status(self._status_line(event))
        elif event.elapsed_s - self._last_periodic_s >= self.milestone_interval_s:
            # Redirected: periodic lines, no control characters. An in-place
            # redraw at poll cadence would be megabytes of carriage returns.
            self._last_periodic_s = float(event.elapsed_s)
            self._line(self._status_line(event))

    # ── what a round really costs ────────────────────────────────────────

    def _absorb_cost(self, event: Any) -> None:
        """Re-project the remaining time from measured round cost, once known.

        The modelled projection is kept alongside rather than replaced: an ETA
        that quietly slides from 9 h to 15 h teaches nobody anything, whereas
        ``ETA 14.8h (model 9.3h)`` says plainly that ``estimate_eis_duration``
        under-counts a round — which is the fact everything downstream needs.
        """
        duration = getattr(event, "round_duration_s", float("nan"))
        if duration != duration or float(duration) <= 0:   # NaN or absent
            return
        if not event.round_kind:
            return
        self._measured_series_s = float(duration)
        self.measured_total_s = project_duration(
            self.config, measured_series_round_s=self._measured_series_s).typical_s

    # ── the controls monitor ─────────────────────────────────────────────

    def _maybe_telemetry(self, event: Any) -> None:
        """One timestamped PV line per interval, on its own line and in scrollback.

        Deliberately *not* folded into the redrawn status line alone: a monitor
        that overwrites itself has no history, and "are the controls working?" is
        a question about a trend. The compact form still rides the live line for
        an at-a-glance answer.
        """
        if event.env_status == ENV_ABSENT:
            return
        elapsed = float(event.elapsed_s)
        if (self._last_env_s is not None
                and elapsed - self._last_env_s < self.milestone_interval_s):
            return
        self._last_env_s = elapsed
        self._line(f"[{hms(elapsed)}] {event.wall_clock}  {self._env_full(event)}")

    @staticmethod
    def _val(value: Any, fmt: str = "{:.1f}") -> str:
        """A number, or :data:`NA`. There is no third rendering."""
        try:
            f = float(value)
        except (TypeError, ValueError):
            return NA
        return NA if f != f else fmt.format(f)

    def _env_full(self, event: Any) -> str:
        if event.env_status == ENV_SKIPPED:
            return "env  (not read: rig in use -- telemetry never delays a spectrum)"
        env = event.env or {}
        return (f"env  T sp {self._val(env.get('stage_temp_sp_C'))} "
                f"pv {self._val(env.get('chamber_air_C'))} "
                f"stage {self._val(env.get('stage_temp_pv_C'))}C   "
                f"RH sp {self._val(env.get('rh_sp_pct'))} "
                f"pv {self._val(env.get('rh_pv_pct'))}%")

    def _env_compact(self, event: Any) -> str:
        """The at-a-glance form for the live line — 20 characters at most."""
        if event.env_status == ENV_ABSENT:
            return ""
        if event.env_status == ENV_SKIPPED:
            return "env busy"
        env = event.env or {}
        stage = self._val(env.get("stage_temp_pv_C"), "{:.0f}")
        temp_sp = self._val(env.get("stage_temp_sp_C"), "{:.0f}")
        rh_pv = self._val(env.get("rh_pv_pct"), "{:.0f}")
        rh_sp = self._val(env.get("rh_sp_pct"), "{:.0f}")
        return f"T{stage}/{temp_sp} RH{rh_pv}/{rh_sp}"

    # ── text ─────────────────────────────────────────────────────────────

    def _where(self, event: Any, *, terse: bool = False) -> str:
        """leg -> setpoint, the first two rungs of the hierarchy.

        ``terse`` drops the setpoints, because when live telemetry is on the same
        line it is already showing them next to their PVs — and the line has to
        fit a terminal nobody promised would be wide.
        """
        if not event.leg:
            return ""
        sp = f"S{event.setpoint_index + 1}/{event.n_setpoints}"
        if terse:
            return f"{event.leg} {sp}"
        temp = ("" if event.temperature_C != event.temperature_C
                else f" {event.temperature_C:g}C")
        rh = ("" if event.rh_setpoint_pct != event.rh_setpoint_pct
              else f"/{event.rh_setpoint_pct:g}%RH")
        return f"{event.leg} {sp}{temp}{rh}"

    def _phase(self, event: Any) -> str:
        """phase -> round -> channel, the last three rungs."""
        if event.phase == "approach":
            pv = "?" if event.pv != event.pv else f"{event.pv:.1f}"
            target = "?" if event.target != event.target else f"{event.target:g}"
            return f"approach {event.axis} {pv}->{target}"
        if event.phase == "hold":
            pv = "?" if event.pv != event.pv else f"{event.pv:.1f}"
            return f"hold {event.axis} {pv}"
        if event.round_index >= 0:
            where = f"{event.round_kind or 'round'} r{event.round_index + 1}/{event.n_rounds}"
            return f"{where} ch{event.channel}" if event.channel else where
        return event.phase or "…"

    def _status_line(self, event: Any) -> str:
        total = (self.measured_total_s if self.measured_total_s is not None
                 else self.projected_total_s)
        eta = reconciled_eta_s(event.elapsed_s, event.fraction, total)
        head = f"[{hms(event.elapsed_s)}]"
        tail = f"{event.fraction * 100:.0f}% ETA {eta / 3600:.1f}h"
        if (self.measured_total_s is not None
                and abs(self.measured_total_s - self.projected_total_s) > 60.0):
            tail += f" (model {self.projected_total_s / 3600:.1f}h)"
        env = self._env_compact(event)
        parts = [head, self._where(event, terse=bool(env)), "|", self._phase(event)]
        if env:
            parts += ["|", env]
        parts += ["|", tail]
        return " ".join(p for p in parts if p)

    def _milestone(self, event: Any) -> str:
        stamp = f"[{hms(event.elapsed_s)}]"
        if event.kind == EV_RUN_STARTED:
            return f"{stamp} START  {event.detail}"
        if event.kind == EV_LEG_STARTED:
            return f"{stamp} === leg '{event.leg}' ==="
        if event.kind == EV_LEG_FINISHED:
            return f"{stamp} === leg '{event.leg}' complete ==="
        if event.kind == EV_SETPOINT_STARTED:
            return f"{stamp} --- {self._where(event)} ---"
        if event.kind == EV_COST_WARNING:
            return f"{stamp}   !! ROUND OVERRUNS THE PERIOD: {event.detail}"
        if event.kind == EV_APPROACH_FINISHED:
            mark = "ok" if event.verdict == VERDICT_MET else "!!"
            return f"{stamp}   {mark} {event.detail}"
        if event.kind == EV_AMBIENT_RESTORED:
            # Unindented and marked, unlike the approach line: a failed restore is
            # the one message on this console that needs a human to walk to the rig.
            mark = "ok" if event.verdict == VERDICT_MET else "!!!!"
            return f"{stamp} {mark} {event.detail}"
        if event.kind == EV_SETTLE_VERDICT:
            # Marked, not merely printed: a setpoint that stopped short is the one
            # line an operator must be able to read as "sigma settled" rather than
            # "something gave up", and the two are otherwise the same shape.
            mark = "ok" if event.verdict == VERDICT_MET else ".."
            return f"{stamp}   {mark} SERIES {self._where(event)}: {event.detail}"
        if event.kind == EV_SETPOINT_FINISHED:
            mark = "HELD" if event.verdict == VERDICT_MET else "NOT HELD"
            return f"{stamp}   VERDICT {self._where(event)}: {mark} -- {event.detail}"
        if event.kind == EV_RUN_FINISHED:
            if event.verdict == VERDICT_ABORTED:
                return f"{stamp} ABORTED  {event.detail}"
            return f"{stamp} DONE  {event.detail}"
        return f"{stamp} {event.kind} {event.detail}".rstrip()

    # ── writing ──────────────────────────────────────────────────────────

    def _line(self, text: str) -> None:
        if self._live:
            self._write("\r" + " " * STATUS_WIDTH + "\r")
            self._live = False
        self._write(text[:STATUS_WIDTH * 3] + "\n")

    def _status(self, text: str) -> None:
        self._write("\r" + text[:STATUS_WIDTH].ljust(STATUS_WIDTH))
        self._live = True

    def _write(self, text: str) -> None:
        self.stream.write(text)
        flush = getattr(self.stream, "flush", None)
        if flush is not None:
            flush()

    def close(self) -> None:
        """Leave the cursor on a fresh line so the next print is not overwritten."""
        try:
            if self._live:
                self._write("\n")
                self._live = False
        except Exception:
            self.failures += 1


class ChannelsNotStated(ValueError):
    """``run`` was invoked with no ``--channels``.

    A ``ValueError`` subclass so a caller that only knows the general failure
    still catches it, but distinct so ``_cmd_run`` can print the one thing that
    matters here: *which* flag, and why it cannot be inherited from ``plan``.
    """


#: The three terms of σ = L/(R·t·w), each with the flag that supplies it and the
#: key ``EquilibrationConfig`` stores it under. One table, so the refusal below,
#: the warning ``plan`` prints and the flags the next-action line echoes cannot
#: disagree about which terms exist.
GEOMETRY_TERMS = (("L_cm", "electrode_l_cm", "--electrode-l-cm"),
                  ("t_cm", "electrode_t_cm", "--electrode-t-cm"),
                  ("w_cm", "electrode_w_cm", "--electrode-w-cm"))


def _resolve_geometry(args) -> dict[str, float] | None:
    """L/t/w, or nothing — and **never a silent partial**.

    What this replaces was ``if args.electrode_l_cm and args.electrode_t_cm and
    args.electrode_w_cm``, which had two defects in one line. Supplying one or
    two terms discarded the whole dict with no statement that anything had been
    dropped, so σ was NULL for the entire run. And it was a *truthiness* test, so
    ``--electrode-t-cm 0`` was indistinguishable from omitting the flag — a
    stated, wrong value read as an absent one, in the denominator of
    σ = L/(R·t·w).

    The all-three requirement itself stands: σ genuinely needs every term. It was
    the silence that was the bug, so a partial refuses and names both halves.
    """
    supplied: list[tuple[str, float]] = []
    missing: list[str] = []
    nonpositive: list[str] = []
    for key, dest, flag in GEOMETRY_TERMS:
        value = getattr(args, dest, None)
        if value is None:
            missing.append(flag)
        elif float(value) <= 0:
            nonpositive.append(f"{flag} {float(value):g}")
        else:
            supplied.append((key, float(value)))
    # Checked first, and separately from omission: a zero is a value the operator
    # typed, and telling them "it is missing" would send them to add a flag they
    # already added.
    if nonpositive:
        raise ValueError(
            f"non-positive electrode geometry: {', '.join(nonpositive)}. A zero or "
            f"negative term is a STATED value, not an absent one, and "
            f"sigma = L/(R*t*w) divides by t and w.")
    if not supplied:
        return None
    if missing:
        raise ValueError(
            f"partial electrode geometry: supplied "
            f"{', '.join(f'{k}={v:g}' for k, v in supplied)}, MISSING "
            f"{', '.join(missing)}. All three are needed for sigma = L/(R*t*w), so "
            f"this would otherwise be dropped whole and every sigma in the run "
            f"would be NULL. Give the missing term(s), or give none at all.")
    return dict(supplied)


def build_config(args) -> EquilibrationConfig:
    """One config builder for every subcommand, so ``plan`` cannot describe a run
    different from the one ``run`` executes."""
    # `run` declares --channels with no default (see `_add_design_args`). Reaching
    # `parse_channel_spec(None)` here would be an AttributeError at the top of a
    # command whose next move is to heat a chamber.
    if getattr(args, "channels", None) is None:
        raise ChannelsNotStated("no --channels was given")
    channels = parse_channel_spec(args.channels)
    temps = [float(t) for t in str(args.temperatures).split(",") if t.strip()]
    geometry = _resolve_geometry(args)
    legs = tuple(leg.strip() for leg in str(args.legs).split(",") if leg.strip())
    config = EquilibrationConfig(
        channels=channels, temperatures_C=temps, legs=legs,
        rh_setpoint_pct=args.rh, rounds_per_setpoint=args.rounds,
        round_period_s=args.round_period_s, eis_preset=args.preset,
        eis_f_lo_mHz=getattr(args, "f_lo_mHz", None),
        eis_model=args.model,
        electrode_geometry=geometry,
        thickness_method=getattr(args, "thickness_method", "target"),
        # `getattr` with the module defaults, not `args.x`: a hand-built namespace
        # (the GUI, a test, a caller predating these flags) must still produce the
        # shipped criterion rather than an AttributeError at the top of a command
        # whose next move is to heat a chamber.
        settle_enabled=str(getattr(args, "settle", "on")).lower() != "off",
        settle_tol_rel=getattr(args, "settle_tol_rel", DEFAULT_SETTLE_TOL_REL),
        settle_n_rounds=getattr(args, "settle_n_rounds", DEFAULT_SETTLE_N_ROUNDS),
        settle_min_channels=getattr(args, "settle_min_channels",
                                    DEFAULT_SETTLE_MIN_CHANNELS),
        min_hold_first_s=getattr(args, "min_hold_first_s", DEFAULT_MIN_HOLD_FIRST_S),
        min_hold_s=getattr(args, "min_hold_s", DEFAULT_MIN_HOLD_S),
        tau_setpoints=getattr(args, "tau_setpoints", DEFAULT_TAU_SETPOINTS),
        **_chamber_settings(args),
    )
    config.validate()
    return config


#: The chamber bands and allowances, by ``EquilibrationConfig`` field name — which
#: is also the ``argparse`` dest and the plan-file key, so the flag an operator
#: types, the value the plan records and the field the run reads are one name from
#: end to end. Paired with the module default so a hand-built namespace (the GUI,
#: a test, a caller predating these flags) still produces the shipped chamber
#: rather than an ``AttributeError`` at the top of a command that heats things.
CHAMBER_SETTINGS = (
    ("tolerance_C", DEFAULT_TOLERANCE_C),
    ("rh_tolerance_pct", DEFAULT_RH_TOLERANCE_PCT),
    ("warn_C", DEFAULT_WARN_C),
    ("fault_C", DEFAULT_FAULT_C),
    ("grace_s", DEFAULT_GRACE_S),
    ("approach_timeout_s", DEFAULT_APPROACH_TIMEOUT_S),
    ("down_approach_timeout_s", DEFAULT_DOWN_APPROACH_TIMEOUT_S),
    ("rh_approach_timeout_s", DEFAULT_RH_APPROACH_TIMEOUT_S),
)


def _chamber_settings(args) -> dict[str, float]:
    """The eight chamber values off the namespace, defaulted from one table."""
    return {name: float(getattr(args, name, default))
            for name, default in CHAMBER_SETTINGS}


# ── The plan as an executable artifact ───────────────────────────────────────
#
# `plan` and `run` share no state, so the design has to travel between them as a
# file or not at all. `plan --save` writes every value the run will use --
# including the defaults nobody typed -- and `run --from-plan` executes exactly
# that. TOML, because it is this codebase's run-spec format
# (`core/campaign_spec_io.py` reads campaign specs from it and `tomli_w` is a
# declared dependency), and because a plan is read by a human at 3 a.m. before a
# nine-hour heat.

#: Bumped when the [design] contract changes. `run --from-plan` refuses anything
#: else outright rather than reading what it recognises: a plan half-understood
#: is a run half-defaulted, which is the defect this file exists to close.
#:
#: ``/2`` added the six ``settle_*`` / ``min_hold_*`` keys. A ``/1`` plan is
#: refused rather than read with those defaulted, which is the whole rule: a plan
#: written when ``rounds`` meant "run exactly this many" would otherwise execute
#: under a criterion that can stop the setpoint at three, and the file would say
#: nothing about it.
#:
#: ``/3`` added ``tau_setpoints`` and the eight chamber bands and allowances
#: (:data:`CHAMBER_SETTINGS`). Same rule, and it bites harder: a ``/2`` plan was
#: written when ``tolerance_C`` was 0.5, ``rh`` was 15 and the descending leg had
#: the ascending leg's 1800 s allowance. Executing one here with those defaulted
#: would silently change what "held" means and how long the chamber is given to
#: get there, on a file that states neither. Re-save the design.
#:
#: ``/4`` added ``f_lo_mHz``. Same rule again, and it decides what the run can
#: *see*: a ``/3`` plan was written when ``Quick`` ended at 20 Hz, and executing
#: one here would take it to the preset's new 7 Hz — a different sweep, roughly
#: twice the cost, and a different set of samples whose arcs close — on a file
#: that names no floor at all. Re-save the design.
PLAN_SCHEMA = "equilibration-plan/4"

#: Every design flag :func:`build_config` reads, by its ``argparse`` dest. A key
#: absent from a saved plan is a value that reverts to its default on ``run``, so
#: all of them are written whether or not they were typed.
#:
#: ``--project`` / ``--mock`` are deliberately **not** design: they choose which
#: store and which drivers, not which experiment, and a plan file that pinned
#: ``mock = true`` would silently divert a real night's data into the mock store.
#: ``--fixture`` is a ``plan``-only calibration advisory and ``run`` has no such
#: flag to receive it.
PLAN_DESIGN_KEYS = (
    "channels", "temperatures", "legs", "rh", "rounds", "round_period_s",
    "settle", "settle_tol_rel", "settle_n_rounds", "settle_min_channels",
    "min_hold_first_s", "min_hold_s", "tau_setpoints",
    "preset", "f_lo_mHz", "model", "measured_per_channel_s",
    "electrode_l_cm", "electrode_t_cm", "electrode_w_cm", "thickness_method",
    # The chamber. In the plan for the same reason the settle criterion is: what
    # counts as "held", and how long the chamber is given to get there, is as much
    # a part of the experiment as which temperatures it visits — and a verdict
    # graded against a tolerance that reverted between `plan` and `run` is the
    # 2026-08-10 defect in a field where it would never be noticed.
    *(name for name, _default in CHAMBER_SETTINGS),
)

#: The design keys that legitimately have no value — the geometry, which may be
#: absent altogether; the measured cost, which most rigs do not have; and the
#: sweep floor, whose absence *is* a design statement ("take the preset's"),
#: exactly as ``preset`` records a name rather than the sweep behind it. Every
#: other key missing from a plan file is a corrupt plan, not an empty one.
PLAN_OPTIONAL_KEYS = frozenset({"measured_per_channel_s", "f_lo_mHz",
                                "electrode_l_cm", "electrode_t_cm",
                                "electrode_w_cm"})


class PlanFileError(ValueError):
    """The file does not describe a run this tool can execute.

    Always a refusal, never a fallback. Reading a plan partially — or defaulting
    the parts that would not parse — would reproduce exactly the failure the plan
    file was introduced to prevent, with the added insult of a file on disk
    saying otherwise.
    """


def _design_flag_spellings() -> dict[str, str]:
    """dest → the flag an operator types, read off the real parser.

    Derived rather than restated so a renamed flag cannot leave this module
    printing a command nobody can run.
    """
    probe = argparse.ArgumentParser(add_help=False)
    _add_design_args(probe, required_channels=True)
    return {action.dest: action.option_strings[0] for action in probe._actions
            if action.dest in PLAN_DESIGN_KEYS}


def _explicit_design_flags(argv: Sequence[str]) -> set[str]:
    """Which design flags were **typed**, which ``argparse`` does not record.

    A namespace value equal to its default is indistinguishable from one nobody
    supplied, and "the operator typed ``--preset Standard`` while the plan says
    ``Quick``" must not be read as "nobody chose a preset". So the same design
    surface is re-parsed with every default suppressed; what survives is what was
    on the command line.

    Safe to re-parse: the real parser has already accepted this argv, so nothing
    here can fail that would not have failed first.
    """
    probe = argparse.ArgumentParser(add_help=False)
    _add_design_args(probe, required_channels=True)
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    typed, _unknown = probe.parse_known_args(list(argv))
    return set(vars(typed))


def _plan_value(value: Any) -> str:
    """One rendering for a design value, in a file and in a diff alike."""
    if value is None:
        return "(unset)"
    return f"{value:g}" if isinstance(value, float) else str(value)


def design_flags(args) -> list[str]:
    """The resolved design as flags an operator could retype, defaults included.

    Omitting a flag here because it happens to hold its default is what produced
    a ``Standard`` run from a ``Quick`` plan, so nothing is omitted except the
    values that genuinely have none.
    """
    spellings = _design_flag_spellings()
    out: list[str] = []
    for key in PLAN_DESIGN_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        out += [spellings[key], _plan_value(value)]
    return out


def plan_payload(args) -> dict[str, Any]:
    """The fully resolved design, ready to serialize."""
    design = {key: getattr(args, key, None) for key in PLAN_DESIGN_KEYS}
    return {
        "schema": PLAN_SCHEMA,
        "written_by": MODULE,
        # The same wall clock `ProgressEvent.wall_clock` stamps every event of the
        # resulting run with, so a plan and its run are read off one clock in one
        # format rather than two spellings of "when".
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "design": {k: v for k, v in design.items() if v is not None},
    }


def _plan_header(path: Path) -> str:
    """The file says what it is and how to execute it, to whoever opens it."""
    return (
        "# The equilibration design, fully resolved -- EVERY value the run will\n"
        f"# use, including the defaults nobody typed. Written by `{CLI} plan --save`.\n"
        "#\n"
        "# `plan` and `run` are separate processes sharing no state, so a flag not\n"
        "# repeated on `run` reverts to its default. Execute this design verbatim:\n"
        "#\n"
        f"#   {CLI} run --from-plan {path} --execute\n"
        "#\n"
        "# A flag typed alongside --from-plan still wins, but is printed as a diff\n"
        "# against this file and repeated in the thermal confirmation.\n"
        "\n"
    )


def write_plan(args, path: Any) -> Path:
    """Write the resolved design to *path*, and return where it landed."""
    import tomli_w

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(_plan_header(target).encode("utf-8"))
        tomli_w.dump(plan_payload(args), handle)
    return target


def load_plan(path: Any) -> dict[str, Any]:
    """Read a saved plan, or refuse. The design mapping, keyed by argparse dest.

    Every failure mode is a refusal with a reason: a missing file, a file that is
    not TOML, a schema this build does not know, an unknown key (a typo would
    otherwise take its default silently — the same rule
    ``core/campaign_spec_io.py`` applies to campaign specs), and a required key
    that is absent.
    """
    import tomllib

    target = Path(path).expanduser()
    if not target.exists():
        raise PlanFileError(f"no such plan file: {target}")
    try:
        with target.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise PlanFileError(f"{target} could not be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlanFileError(f"{target} is not readable as TOML: {exc}") from exc

    schema = data.get("schema")
    if schema != PLAN_SCHEMA:
        raise PlanFileError(
            f"{target} declares schema {schema!r}, and this build executes only "
            f"{PLAN_SCHEMA!r}. Re-save the design rather than running a file whose "
            f"fields mean something else here.")
    design = data.get("design")
    if not isinstance(design, dict):
        raise PlanFileError(
            f"{target} has no [design] table, so it states no design at all.")
    unknown = sorted(set(design) - set(PLAN_DESIGN_KEYS))
    if unknown:
        raise PlanFileError(
            f"{target}: unknown design key(s) {unknown}. A misspelled key would "
            f"silently leave that value at its default, so it is refused rather "
            f"than ignored. Valid keys: {list(PLAN_DESIGN_KEYS)}")
    absent = [key for key in PLAN_DESIGN_KEYS
              if key not in design and key not in PLAN_OPTIONAL_KEYS]
    if absent:
        raise PlanFileError(
            f"{target} is missing required design key(s) {absent}. A plan that "
            f"omits a value is a plan that defaults it, which is what this file "
            f"exists to prevent.")
    return {key: design.get(key) for key in PLAN_DESIGN_KEYS}


def apply_plan(args, design: dict[str, Any],
               explicit: set[str]) -> list[tuple[str, Any, Any]]:
    """Seat *design* on the namespace; return what the command line overrode.

    An override is allowed — re-planning a whole night to change one preset would
    be worse — but it is never silent: the caller prints every entry of the
    returned diff and repeats it in the thermal confirmation.
    """
    overrides: list[tuple[str, Any, Any]] = []
    for key, planned in design.items():
        typed = getattr(args, key, None)
        if key in explicit and typed != planned:
            overrides.append((key, planned, typed))
            continue
        setattr(args, key, planned)
    return overrides


def _print_plan_source(path: Any, overrides: list[tuple[str, Any, Any]]) -> None:
    print(f"  plan: {path}")
    if not overrides:
        print("    executed exactly as saved; nothing was overridden.")
        return
    print("    ! OVERRIDDEN ON THE COMMAND LINE -- this is NOT the saved design:")
    for key, planned, typed in overrides:
        print(f"        {key}: {_plan_value(planned)} -> {_plan_value(typed)}")


# ── plan ─────────────────────────────────────────────────────────────────────

def _measured_per_channel(value: Any) -> float | None:
    """A usable measured per-channel cost, or ``None`` — meaning fall back to the model.

    Zero and negatives are treated as absent rather than refused: the flag is an
    input to a projection, and a plan that exits on a typo teaches less than one
    that falls back to the model and says loudly that it *is* a model.

    Applied at every entry point rather than only where the flag is parsed, so
    "which basis is this?" is decided by one predicate. Splitting it would let a
    projection be computed from the model while the page above it said MEASURED.
    """
    if value is None or float(value) <= 0:
        return None
    return float(value)


def _preset_f_lo_hz(preset: str) -> float:
    """A preset's own sweep floor, in Hz."""
    from softae.core.eis_scripts import EISParams

    return EISParams.from_preset(preset).f_lo_mHz / 1000.0


def _reference_sigma_floor() -> float:
    """The σ the default preset reaches on the 4-stripe board's nominal geometry.

    Computed from the live preset and :data:`REFERENCE_GEOMETRY_CM` rather than
    written down, so moving ``[eis_presets.Quick]`` again — or re-characterising
    ``CELL_CAPACITANCE_F`` — moves the number quoted in ``--help`` instead of
    leaving it asserting a reach the tool no longer has.
    """
    from softae.core.eis_scripts import sigma_floor_S_per_cm

    return sigma_floor_S_per_cm(_preset_f_lo_hz(DEFAULT_EIS_PRESET),
                                *REFERENCE_GEOMETRY_CM) or 0.0


def _reference_geometry_str() -> str:
    L, t, w = REFERENCE_GEOMETRY_CM
    return f"L={L:g} t={t:g} w={w:g} cm"


def _print_sigma_reach(config: EquilibrationConfig) -> None:
    """What the sweep floor can actually see — the guardrail, stated up front.

    The floor is a conductivity floor wearing a frequency's clothes: the -Z''
    peak sits at ``1/(2*pi*R*C_cell)``, so a sample below
    ``2*pi*f_lo*C_cell*L/(t*w)`` never brings its apex inside the sweep and its
    R₁ is read off the high-frequency limb instead — a *systematic* 61 % median
    overestimate (175 % with a CPE fit), not scatter that averages out. Run
    ``20260811T023757Z`` produced 476 such spectra and said so nowhere before the
    fit; this line is where it now says so, before the night is committed.

    Printed from the plan's **own** ``f_lo`` and geometry. With no geometry there
    is no number to print and none is invented: σ = L/(R·t·w) has no value here,
    and quoting a reach against a geometry nobody supplied would be the same class
    of silent wrongness in the opposite direction.
    """
    from softae.core.eis_scripts import CELL_CAPACITANCE_F, sigma_floor_S_per_cm

    f_lo_hz = config.eis_params().f_lo_mHz / 1000.0
    geometry = config.electrode_geometry or {}
    sigma = sigma_floor_S_per_cm(f_lo_hz, geometry.get("L_cm"),
                                 geometry.get("t_cm"), geometry.get("w_cm"))
    if sigma is None:
        print(f"  sigma reach:  unavailable, geometry not supplied -- the "
              f"{f_lo_hz:g} Hz floor is")
        print("                a conductivity floor, but naming it needs L, t and w.")
        return
    print(f"  sigma reach:  f_lo {f_lo_hz:g} Hz -> arcs close for sigma >~ "
          f"{sigma:.1e} S/cm")
    print(f"                at L={geometry['L_cm']:g} t={geometry['t_cm']:g} "
          f"w={geometry['w_cm']:g} cm (C_cell {CELL_CAPACITANCE_F * 1e9:g} nF); "
          f"below that")
    print("                R1 is EXTRAPOLATED off the high-frequency limb, which "
          "reads")
    print("                ~61% HIGH as a bias, not as noise.")


def _print_design(config: EquilibrationConfig, *,
                  measured_per_channel_s: float | None = None) -> None:
    """The design, budgeted on **one** cost basis throughout.

    ``measured_per_channel_s`` replaces the modelled cost everywhere below — the
    round cost, the inter-round gap, the headroom, the sampling interval and the
    whole-run duration — because a projection that mixes a measured round with a
    modelled gap is wrong in a way no single printed figure exposes.
    """
    measured_per_channel_s = _measured_per_channel(measured_per_channel_s)
    measured = measured_per_channel_s is not None
    cost = round_cost_s(config, measured_per_channel_s=measured_per_channel_s)
    projection = project_duration(
        config, measured_series_round_s=cost if measured else None)
    print(f"Equilibration characterization run: {len(config.channels)} channel(s), "
          f"{config.n_setpoints} setpoints")
    # NOT `channels[0]-channels[-1]`: that renders "1-3,8-16" as "1-16" and tells
    # the operator seven channels are in the run when they are not.
    print(f"  channels:     {format_channel_spec(config.channels)} "
          f"({len(config.channels)})")
    print(f"  temperatures: {', '.join(f'{t:g}' for t in config.temperatures_C)} C")
    print(f"  legs:         {' then '.join(config.legs)}  "
          f"(the down leg revisits every temperature)")
    print(f"  humidity:     {config.rh_setpoint_pct:g} %RH, re-established at EVERY "
          f"temperature")
    print(f"  rounds:       {'up to ' if config.settle_enabled else ''}"
          f"{config.rounds_per_setpoint} x "
          f"{config.round_period_s:g}s per setpoint ({config.eis_preset}), "
          f"one preset throughout")
    _print_sigma_reach(config)
    _print_settle(config, projection)
    print()
    _print_round_cost(config, cost, measured=measured)
    _print_period_caution(config, cost, measured=measured)
    print()
    _print_projection(projection)
    _print_basis(config, cost, measured=measured)
    # The interval the executor will really achieve, not the configured period:
    # when the round overruns, the gap floors at one poll interval and the true
    # sampling interval is the round plus that floor. Quoting the configured
    # period here is exactly the reassurance that let a 240 s plan precede an
    # 11.4 min cycle.
    interval = cost + inter_round_gap_s(config, cost)
    print()
    print(f"  Sampling interval {interval:.0f}s -> shortest resolvable tau ~ "
          f"{2 * interval / 60:.0f} min (2 x the interval).")


def _print_settle(config: EquilibrationConfig, projection: Any) -> None:
    """State that ``--rounds`` is a ceiling, and what the floors really buy.

    Every floor is printed, not their minimum. They differ — only one setpoint in
    the run gets ``min_hold_first_s``, and only the first ``tau_setpoints`` carry
    the fit minimum — and the number an operator needs when reading a short
    setpoint later is the one that applied to *that* setpoint.

    The effective figure is what the run enforces, which is not the time floor
    divided by the period: :data:`MIN_POINTS_FOR_TAU` is folded in wherever a τ is
    wanted, and at the shipped 660 s period it is the term that binds there.
    Printing only the time floors would understate the fast end of the budget by
    exactly the amount that makes the transient analysable.
    """
    if not config.settle_enabled:
        print("  settle:       OFF -- every setpoint runs exactly "
              f"{config.rounds_per_setpoint} rounds.")
        return
    print(f"  settle:       stop at {config.settle_tol_rel * 100:g}% over "
          f"{config.settle_n_rounds} rounds, >= {config.settle_min_channels} usable "
          f"channel(s)")
    print(f"                floors {config.min_hold_first_s:g}s at the FIRST setpoint "
          f"of the run, {config.min_hold_s:g}s after")
    _print_floor_regimes(config, projection)
    print(f"                so a setpoint runs {projection.min_rounds}-"
          f"{config.rounds_per_setpoint} rounds -- --rounds is a CEILING, not a count.")
    _print_tau_window(config)
    print("                A channel with NULL sigma, or an R1 railed on the circuit")
    print("                model's bound, does not count towards the criterion: a")
    print("                railed fit is constant, and a constant is always 'settled'.")


def _tau_window(config: EquilibrationConfig) -> int:
    """Setpoints of this run that carry the fit minimum, clamped to the run."""
    return max(0, min(int(config.tau_setpoints), int(config.n_setpoints)))


def _print_floor_regimes(config: EquilibrationConfig, projection: Any) -> None:
    """The effective minimum in **every** regime, and where the boundaries are.

    Two numbers were enough while :data:`MIN_POINTS_FOR_TAU` applied everywhere.
    It no longer does, so an operator reading a 3-round setpoint next to a
    5-round one has to be able to see that the difference is the τ window and not
    a configuration change.
    """
    n_tau = _tau_window(config)
    print(f"                effective minimum {projection.min_rounds_first} rounds at "
          f"setpoint 1 of the run,")
    # The middle regime — inside the τ window but past the first setpoint — is
    # printed only when it is a regime: it does not exist for a window of one, and
    # it is not a separate number when it agrees with what follows it.
    if n_tau <= 1 or projection.min_rounds_tau == projection.min_rounds_later:
        print(f"                {projection.min_rounds_later} after")
        return
    span = "setpoint 2" if n_tau == 2 else f"setpoints 2-{n_tau}"
    print(f"                {projection.min_rounds_tau} at {span}, "
          f"{projection.min_rounds_later} after that")


def _print_tau_window(config: EquilibrationConfig) -> None:
    """Why the first setpoints are floored harder than the rest — and where to move it."""
    n_tau = _tau_window(config)
    if n_tau <= 0:
        print(f"                --tau-setpoints 0: the {MIN_POINTS_FOR_TAU}-round fit "
              f"minimum applies NOWHERE, so")
        print("                no setpoint in this run is guaranteed to yield a tau.")
        return
    print(f"                The first {n_tau} setpoint(s) may not stop under "
          f"{MIN_POINTS_FOR_TAU} rounds: that is")
    print("                the offline fitter's own MIN_POINTS_FOR_TAU (sigma(t) has")
    print("                three free parameters), so a shorter series would be")
    print("                acquired and then REFUSED for tau. Past them the films are")
    print("                dry and there is no relaxation left to fit, so the floor is")
    print("                --settle-n-rounds and the hold floor alone (--tau-setpoints).")


def _print_projection(projection: Any) -> None:
    """The budget, as a **range**, because the length is now data-dependent.

    A single figure here would be a promise the run cannot keep in either
    direction: it stops early when σ settles, and it does not when σ does not.
    The floor is what the criterion and the hold floors guarantee; the ceiling is
    the old fixed-count number.
    """
    def _row(label: str, pairs: Sequence[tuple[float, float]], unit: str) -> None:
        scale, fmt = (60.0, ".1f") if unit == "min" else (3600.0, ".2f")
        cells = []
        for floor_s, ceiling_s in pairs:
            span = (f"{ceiling_s / scale:{fmt}}" if abs(ceiling_s - floor_s) < 1.0
                    else f"{floor_s / scale:{fmt}}-{ceiling_s / scale:{fmt}}")
            cells.append(f"{span:>14s} {unit:<3s}")
        print(f"    {label:<20s} " + " ".join(cells))

    # Only the σ series shortens: the approaches are the chamber's to spend and
    # the criterion has no opinion about them.
    series_floor = projection.min_rounds * projection.series_round_s
    print("  Projected duration          typical            worst case")
    for key in ("temperature_approach", "rh_approach", "sigma_series"):
        typical, worst = (projection.breakdown_typical[key],
                          projection.breakdown_worst[key])
        floor = series_floor if key == "sigma_series" else None
        _row(key, [(typical if floor is None else floor, typical),
                   (worst if floor is None else floor, worst)], "min")
    _row("per setpoint",
         [(projection.per_setpoint_typical_floor_s, projection.per_setpoint_typical_s),
          (projection.per_setpoint_worst_floor_s, projection.per_setpoint_worst_s)], "h")
    _row("WHOLE RUN",
         [(projection.typical_floor_s, projection.typical_s),
          (projection.worst_floor_s, projection.worst_case_s)], "h")
    _print_approach_allowances(projection)
    if projection.adaptive:
        print("    (a RANGE, not an estimate: the floor is every setpoint settling at")
        print(f"     its earliest -- {'/'.join(str(f) for f in projection.floor_rounds)}"
              f" rounds in run order, of {projection.rounds_per_setpoint} -- and the")
        print("     ceiling is none of them settling at all. Both ends are achievable.)")
    print("    (worst case uses the TIMEOUTS, not an assumed ramp rate, so it is an")
    print("     upper bound rather than a guess. Holds are computed from this config:")
    print("     estimate_workflow_duration projects every temperature wait as 0.0 s.)")


def _print_approach_allowances(projection: Any) -> None:
    """The two temperature allowances, whenever they are not the same number.

    The ``temperature_approach`` row above is the per-setpoint **mean** across the
    legs, which is the right term in a total and the wrong number to quote at an
    operator deciding whether to extend one. Cooling is passive: the descending
    allowance is the one that ran out on 2026-08-11.
    """
    up = projection.temp_approach_timeout_up_s
    down = projection.temp_approach_timeout_down_s
    if abs(up - down) < 1.0:
        return
    print("    (the temperature_approach row is the per-setpoint MEAN of two "
          "allowances:")
    print(f"     {up / 60:.0f} min going UP, {down / 60:.0f} min coming DOWN "
          f"(--down-approach-timeout-s).")
    print("     Cooling is passive and asymptotic; the ascending allowance timed out")
    print("     mid-descent on 2026-08-11 and 15 rounds spanned a 5 C ramp.)")


def _print_round_cost(config: EquilibrationConfig, cost: float, *,
                      measured: bool) -> None:
    """What one all-channel round costs, and **on whose authority**."""
    basis = "MEASURED, from --measured-per-channel-s" if measured else "modelled"
    print(f"  Round cost ({basis}, all channels):")
    print(f"    {config.eis_preset:<10s} {cost / 60:6.1f} min "
          f"({cost / max(1, len(config.channels)):.1f}s/channel)")
    if measured:
        return
    print("    NOTE: the model is FITTED to presets timed on this rig and")
    print(f"    reproduces them within ~{model_underestimate_frac():.0%}. That is a fit "
          f"residual, not a")
    print("    bound: it says nothing about a preset that was never timed, and it")
    print("    does not carry mux switching, script upload, data retrieval or the")
    print("    file write. This run measures its own round cost and records it in")
    print("    the sidecar, so the next plan need not rest on the model at all.")


def _print_basis(config: EquilibrationConfig, cost: float, *,
                 measured: bool) -> None:
    """Never let a modelled duration be read as a measurement.

    The two are close now — the model was refitted to the bench in 2026-08 and is
    ~8 % out on ``Standard`` where it was ~10x out before — so this note no longer
    exists to warn of a chasm. It exists because a night is being committed on the
    strength of a number, and the operator is entitled to know whether that number
    came from a stopwatch or from an extrapolation.
    """
    per_channel = cost / max(1, len(config.channels))
    if measured:
        print(f"    (basis: MEASURED {per_channel:.1f}s/channel, given on the command "
              f"line. The")
        print("     model is not used anywhere above.)")
        return

    # Since 2026-08-17 a preset still on its timed grid resolves through the
    # anchor table, so the cost above IS a stopwatch reading even though nobody
    # typed one. Printing "MODELLED" here would understate the number's standing
    # and push operators toward re-supplying a figure the system already has.
    from softae.core.preflight import measured_duration_s

    if measured_duration_s(config.eis_params()) is not None:
        print(f"    (basis: MEASURED {per_channel:.1f}s/channel, timed on this rig "
              f"for this exact")
        print("     preset grid. The model is not used anywhere above; edit the "
              "preset and")
        print("     this reverts to MODELLED on its own.)")
        return

    frac = model_underestimate_frac()
    print(f"    (basis: MODELLED {per_channel:.1f}s/channel, from a sweep model "
          f"fitted to the")
    print(f"     presets timed on this rig. It runs up to {frac:.0%} UNDER a real "
          f"round on those:")

    # Grid-checked rather than the module constant: if `Standard` were ever moved
    # off its timed grid, quoting its anchor here would repeat in the advice text
    # exactly the staleness the anchor interlock removes from the arithmetic.
    from softae.core.preflight import measured_s_for_preset

    reference = measured_s_for_preset("Standard")
    if reference is None:
        print(f"     Treat it as +/-{frac:.0%}, not as a prediction; "
              f"--measured-per-channel-s")
        print("     plans from a stopwatch instead.)")
    else:
        print(f"     'Standard' measured {reference:g}s/channel here. Treat it as")
        print(f"     +/-{frac:.0%}, not as a prediction; --measured-per-channel-s "
              f"{reference:g} plans")
        print("     from that stopwatch instead.)")
    _print_floor_extrapolation(config)


def _print_floor_extrapolation(config: EquilibrationConfig) -> None:
    """Say when the cost above rests on no stopwatch at all.

    Two ways to get here and they deserve one notice, because the operator's
    exposure is identical. Either the preset itself has no anchor —
    ``eis_duration_basis`` says so, and ``Quick`` has said so since its floor moved
    to 7 Hz and its 20 Hz reading was retired — or ``--f-lo-mHz`` has changed the
    sweep out from under an anchored preset, which is the same downgrade
    ``preflight.project_campaign`` applies on the campaign path. Said here because
    the equilibration plan does not pass through that function, and a night is
    committed off this screen.
    """
    from softae.core.eis_scripts import EISParams
    from softae.core.preflight import (
        EIS_MEASURED_S_PER_CHANNEL,
        eis_duration_basis,
    )

    preset_f_lo = EISParams.from_preset(config.eis_preset).f_lo_mHz
    f_lo = config.eis_params().f_lo_mHz
    overridden = f_lo != preset_f_lo
    if not overridden and eis_duration_basis(config.eis_preset) == "measured":
        return
    if overridden:
        print(f"     ! EXTRAPOLATED: --f-lo-mHz {f_lo:d} makes this NOT the "
              f"'{config.eis_preset}' the")
        print(f"     preset defines. That sweeps to {preset_f_lo / 1000:g} Hz; this "
              f"run sweeps to {f_lo / 1000:g} Hz.")
    else:
        print(f"     ! EXTRAPOLATED: '{config.eis_preset}' has never been timed on "
              f"this rig at its")
        print(f"     {f_lo / 1000:g} Hz floor, so nothing above rests on a stopwatch.")
    # Deliberately not the word "anchor" here: in this tool's output that names
    # the retired --anchor-preset round, and one noun for two ideas on the screen
    # an operator reads before a nine-hour heat is one too many.
    print(f"     The model is calibrated against "
          f"{', '.join(sorted(EIS_MEASURED_S_PER_CHANNEL))} only, and this is not")
    print("     one of them. Time one all-channel round at the rig and plan the "
          "next")
    print("     from it with --measured-per-channel-s.")


def _print_period_caution(config: EquilibrationConfig, cost: float, *,
                          measured: bool) -> None:
    """Say, before the night is committed, whether the period can contain a round.

    ``plan`` is the one thing the operator reads before starting a 15 h run, and
    the failure mode here is quiet: the executor does not refuse an overrunning
    round, it simply takes longer, and σ(t) ends up sampled at a period nobody
    configured.

    Three statements, and they answer different questions. The remainder line is
    the same arithmetic on either basis, but it means different things: against a
    measured cost what is left is simply slack, and against a modelled one it is
    slack that still has to absorb the model's own error. The CAUTION is
    modelled-only for that reason — a stopwatch has no fit residual to allow for.
    The feasibility block is unconditional: at this channel count and this
    per-channel cost, either the period contains a round or it does not.
    """
    headroom = round_headroom_s_per_channel(config, cost if measured else None)
    if measured:
        print(f"    Configured period {config.round_period_s:.0f}s vs MEASURED "
              f"{cost:.0f}s leaves {headroom:+.1f}s/channel of slack.")
    else:
        print(f"    Configured period {config.round_period_s:.0f}s vs modelled "
              f"{cost:.0f}s leaves {headroom:.1f}s/channel of margin.")
        _print_headroom_caution(config, cost, headroom)
    _print_feasibility(config, cost, measured=measured)


def _print_headroom_caution(config: EquilibrationConfig, modelled: float,
                            headroom: float) -> None:
    """The modelled-basis warning: the period fits the model but not the model's error.

    Sized as a *fraction* of the modelled cost rather than a flat per-channel
    reserve — see :func:`model_underestimate_frac` for why the flat one had to go.
    The question asked is exactly: if this preset behaves like the worst-fitted of
    the three timed ones, does the round still fit? A "no" here is not a prediction
    of overrun; it is a refusal to promise there will not be one.

    Silent once the round does not fit *at all*: :func:`_print_feasibility` then
    says so outright, and hedging that a round "may not fit" immediately above
    "UNACHIEVABLE" reads as two verdicts of different strength on one question.
    """
    frac = model_underestimate_frac()
    needed = modelled * (1.0 + frac)
    if modelled >= float(config.round_period_s):
        return
    if frac <= 0.0 or headroom * len(config.channels) >= needed - modelled:
        return
    print(f"    ! CAUTION: {headroom:.1f}s/channel of margin, and the sweep model "
          f"runs up to")
    print(f"      {frac:.0%} under a real round on the presets it was fitted to -- so "
          f"a round")
    print("      MAY NOT FIT, and the run will NOT shorten one or adjust the period:")
    print("      the period is the sampling interval of sigma(t).")
    print(f"      --round-period-s {math.ceil(needed / 10.0) * 10.0:.0f} covers that "
          f"error; fewer channels is the other lever.")


def _print_feasibility(config: EquilibrationConfig, cost: float, *,
                       measured: bool) -> None:
    """The plain statement that a configured period cannot be honoured.

    Framed as the operator needs it rather than as a ratio: the sampling interval
    *is* the experimental parameter, because it sets the shortest equilibration
    time constant this run can resolve (roughly 2x the interval). A τ faster than
    that is not measured slightly badly, it is not measured at all.
    """
    if cost < float(config.round_period_s):
        return
    per_channel = cost / max(1, len(config.channels))
    minimum = minimum_feasible_period_s(cost)
    interval = cost + inter_round_gap_s(config, cost)
    qualifier = ("MEASURED" if measured
                 else f"MODELLED, and up to {model_underestimate_frac():.0%} low")
    print(f"    ! UNACHIEVABLE: one round costs {cost:.0f}s at "
          f"{len(config.channels)} channel(s)")
    print(f"      ({per_channel:.1f}s/channel, {qualifier}) but --round-period-s is "
          f"{config.round_period_s:.0f}.")
    print("      A round is never shortened and sigma(t) is never resampled, so the")
    print(f"      series would be taken at {interval:.0f}s -- the round plus the "
          f"{config.poll_interval_s:.0f}s poll")
    print("      floor the watched gap never drops below.")
    print(f"      Minimum feasible --round-period-s {minimum:.0f}; that sampling "
          f"interval of")
    print(f"      {interval:.0f}s resolves tau no shorter than ~{2 * interval / 60:.0f} "
          f"min (2 x the interval).")
    print("      Fewer channels is the other lever: the cost is per channel.")


def _cmd_plan(args) -> int:
    try:
        config = build_config(args)
    except (ChannelSpecError, ValueError) as exc:
        print(f"Cannot plan this run: {exc}", file=sys.stderr)
        return EXIT_FAILED

    _print_design(config, measured_per_channel_s=getattr(
        args, "measured_per_channel_s", None))
    print()

    _print_geometry_gap(args)

    store, project = _open_store(args)
    try:
        print(f"  store: {project}")
        _print_thickness_note(config, store)
        uncalibrated = _uncalibrated(config.channels, args.fixture)
        if uncalibrated:
            print(f"  · ADVISORY: no commissioning calibration for channel(s) "
                  f"{', '.join(str(c) for c in uncalibrated)}. The run is still "
                  f"valid; the constants are simply not applied.")
    finally:
        store.close()

    saved = _save_plan(args)
    if saved is None:
        return EXIT_FAILED
    _print_next_action(args, config, saved)
    return EXIT_OK


def _print_geometry_gap(args) -> None:
    """Name the terms that are absent. "NO geometry" is false when one was given.

    Derived from what was typed rather than from ``electrode_geometry is None``,
    because those are not the same statement: a partial geometry is refused by
    :func:`_resolve_geometry` before this runs, and the difference between "none
    of the three" and "you gave me two" is the difference between a design
    decision and a typo.
    """
    missing = [flag for _key, dest, flag in GEOMETRY_TERMS
               if getattr(args, dest, None) is None]
    if not missing:
        return
    lead = ("NO electrode geometry" if len(missing) == len(GEOMETRY_TERMS)
            else "PARTIAL electrode geometry")
    print(f"  ! {lead}: missing {', '.join(missing)}.")
    print("    router.handle takes L/t/w from the step params; without them")
    print("    fit_results.sigma_S_per_cm is NULL and there is no sigma(t) at all.")


def _save_plan(args) -> Any:
    """Write ``--save`` if it was given: the saved path, ``""``, or ``None`` on failure.

    ``None`` rather than an exception because the design has already printed and
    the operator's remaining question is only whether the file exists — but it is
    still a failure exit: a plan believed saved and not saved is worse than one
    never asked for, because the next command reads from it.
    """
    target = getattr(args, "save", None)
    if not target:
        return ""
    try:
        return write_plan(args, target)
    except OSError as exc:
        print(f"\nCould not save the plan to {target}: {exc}", file=sys.stderr)
        print("  The design above was NOT written; do not run --from-plan against it.",
              file=sys.stderr)
        return None


def _print_next_action(args, config: EquilibrationConfig, saved: Any) -> None:
    """The command printed here must both **work** and reproduce **this** design.

    It did neither. It named ``softae-equilibration``, which was declared in
    ``pyproject.toml`` but generated by no install in the venv at the time, so
    the name did not resolve; and it printed
    ``run --channels ... --execute`` with no ``--preset``
    and no geometry, so an operator following the tool's own advice got
    ``Standard`` and a NULL σ for every fit — precisely the 2026-08-10 failure.

    Lengthening the line would only postpone that. The saved plan carries every
    resolved value, defaults included, so the suggestion points at a file instead
    of at a flag list that has to stay in sync with the design surface by hand.
    """
    print()
    print("Next most valuable action:")
    if config.electrode_geometry is None:
        print("  supply the electrode geometry, or every sigma in this run is NULL:")
        print(f"    {CLI} plan {' '.join(design_flags(args))} "
              f"--electrode-l-cm 0.2 --electrode-t-cm 0.0175 --electrode-w-cm 0.2 "
              f"--save equilibration_plan.toml")
        return
    if saved:
        print("  start the run at the rig, having read the duration above. The saved")
        print("  plan carries EVERY value printed here, so nothing reverts to a")
        print("  default across the two processes:")
        print(f"    {CLI} run --from-plan {saved} --execute")
        return
    print("  save this design and start the run from the file -- every flag not")
    print("  repeated on 'run' silently reverts to its default:")
    print(f"    {CLI} plan {' '.join(design_flags(args))} "
          f"--save equilibration_plan.toml")
    print(f"    {CLI} run --from-plan equilibration_plan.toml --execute")


def _print_thickness_note(config: EquilibrationConfig, store: Any) -> None:
    """What this run's thickness is — and, deliberately, what it is not.

    ``store.measured_thickness()`` answers "does a per-channel *recorded*
    thickness exist?", which is worth knowing and is why it is still queried. It
    does **not** decide whether σ exists here. ``DataStore.record_fit`` computes σ
    from the ``L_cm``/``t_cm``/``w_cm`` handed to it by ``router.handle`` — which
    reads them straight from the step params — and consults no table at all. P.11
    governs ``make_thickness_lookup`` and ``tab_analysis``; neither is in this
    run's path, so an absent row here refuses nothing.

    The message therefore turns on the geometry, not on the rows: with geometry
    supplied the honest statement is that σ *is* computed and the thickness is
    uniform and operator-attributed; without it, σ is NULL because ``record_fit``
    was handed no ``t_cm``.
    """
    recorded = {int(row["channel"]) for row in store.measured_thickness()}
    absent = [c for c in config.channels if c not in recorded]
    present = [c for c in config.channels if c in recorded]

    if config.electrode_geometry is None:
        if absent:
            print(f"  · no recorded thickness for channel(s) "
                  f"{format_channel_spec(absent)} — not the reason sigma is")
            print("    NULL above; record_fit is simply handed no t_cm.")
        return

    t_um = float(config.electrode_geometry["t_cm"]) * 1.0e4
    print(f"  · thickness {t_um:g} um is OPERATOR-SUPPLIED and UNIFORM across all "
          f"{len(config.channels)} channel(s):")
    print("    sigma IS computed for every one of them, but per-channel variation")
    print(f"    is not captured, and the provenance is --thickness-method "
          f"'{config.thickness_method}'.")
    if present:
        print(f"    ! it OVERRIDES the recorded thickness on file for channel(s) "
              f"{format_channel_spec(present)}.")
    elif absent:
        print("    No channel here has a recorded thickness to compare it against.")


def _uncalibrated(channels, fixture: str) -> list[int]:
    """Channels with no derived commissioning constants — advisory only (T5.4)."""
    try:
        from softae.analysis.eis.calibration import resolve_calibration

        cal = resolve_calibration(fixture)
    except Exception:
        return list(channels)
    if cal is None:
        return list(channels)
    known = set(getattr(cal, "R_short_ohm", {}) or {})
    return [c for c in channels if c not in known]


# ── run ──────────────────────────────────────────────────────────────────────

def confirm_no_geometry(config: EquilibrationConfig, *, assume_yes: bool = False,
                        reader: Any = None) -> bool:
    """Re-ask, because an absent geometry is far more often an omission than a choice.

    Not a refusal and not a new opt-out flag: a run that records only R₁ is a
    legitimate thing to want, and the operator is the one who knows. What is not
    legitimate is discovering after nine hours of rig time that
    ``sigma_S_per_cm`` is NULL for every measurement because three flags were not
    repeated on ``run`` — which is what happened on 2026-08-10.

    Asked **before** :func:`confirm_thermal` so the thermal gate stays the last
    thing a human reads before the chamber is driven, and skipped by ``--yes``
    exactly as that gate is, or no equilibration run could be scripted.
    """
    if config.electrode_geometry is not None:
        return True
    print()
    print("  ! This run has NO electrode geometry "
          "(--electrode-l-cm/-t-cm/-w-cm).")
    print("    sigma_S_per_cm will be NULL for EVERY measurement in the run: only")
    print("    R1 is recorded, and no sigma(t) can be reconstructed from it later.")
    if assume_yes:
        print("  --yes given; proceeding without geometry.")
        return True
    try:
        reply = (reader or input)(
            f"  Type '{CONFIRM_WORD}' to run without it: ").strip().lower()
    except EOFError:
        # A non-TTY declines, as it does at the thermal gate: an unattended
        # invocation that meant this had --yes to say so.
        reply = ""
    if reply != CONFIRM_WORD:
        print("Declined — nothing was heated and nothing was measured.")
        return False
    return True


def confirm_thermal(config: EquilibrationConfig, *, assume_yes: bool = False,
                    reader: Any = None,
                    measured_per_channel_s: float | None = None,
                    plan_overrides: Sequence[tuple[str, Any, Any]] = ()) -> bool:
    """The **only** real gate on this run, and it is this module's own.

    ``assert_hardware_armed`` covers ``("stage", "syringe", "piezo")`` and returns
    without objection for a thermal + EIS workflow, so it cannot be the barrier.
    This states the peak temperature and the projected duration and requires the
    whole word, so a reflex keypress cannot start a nine-hour unattended heat.

    It states the **channel selection** for the same reason: this is the last
    thing a human reads before the chamber is driven, and a wrong ``--channels``
    is otherwise invisible until the wrong samples have been measured.
    ``format_channel_spec``, never ``channels[0]-channels[-1]``, which renders
    ``1-3,8-16`` as ``1-16``.

    The hours quoted are on the **same basis** as the design printed a moment
    earlier: a modelled duration on the last screen before an overnight heat, when
    a measured one was available, is the gap that made a 240 s plan precede an
    11.4 min cycle.
    """
    measured_per_channel_s = _measured_per_channel(measured_per_channel_s)
    cost = round_cost_s(config, measured_per_channel_s=measured_per_channel_s)
    projection = project_duration(
        config,
        measured_series_round_s=cost if measured_per_channel_s is not None else None)
    print()
    print("  " + "=" * 68)
    print(f"  THIS DRIVES THE STAGE HEATER TO {config.peak_temperature_C:g} C")
    print(f"  and actuates the humidifier at {config.rh_setpoint_pct:g} %RH for")
    # Floor to ceiling, not typical to worst: the run stops setpoints early when
    # sigma settles, so the low end of what an operator is committing to is the
    # floor. Quoting `typical_s` as the low end would understate the commitment at
    # exactly the screen where it is being made.
    print(f"  {projection.typical_floor_s / 3600:.1f}-"
          f"{projection.worst_case_s / 3600:.1f} hours, unattended.")
    print(f"  CHANNELS DRIVEN: {format_channel_spec(config.channels)} "
          f"({len(config.channels)}) -- no other channel is measured.")
    # Repeated here rather than only where the plan was loaded: this banner is the
    # last screen before an overnight heat, and an override that scrolled past
    # twenty lines ago is an override nobody confirmed.
    for key, planned, typed in plan_overrides:
        print(f"  OVERRIDDEN vs the saved plan: {key} "
              f"{_plan_value(planned)} -> {_plan_value(typed)}")
    print("  " + "=" * 68)
    if assume_yes:
        print("  --yes given; proceeding without confirmation.")
        return True
    try:
        reply = (reader or input)(f"  Type '{CONFIRM_WORD}' to start: ").strip().lower()
    except EOFError:
        reply = ""
    if reply != CONFIRM_WORD:
        print("Declined — nothing was heated and nothing was measured.")
        return False
    return True


def _refuse_unstated_channels() -> int:
    """The refusal that prevents an excluded channel being energised anyway.

    ``plan`` and ``run`` are separate processes: nothing carries a planned subset
    forward, and a default here would silently restore the channels the operator
    removed — and then energise them. Printed before the dry run as well as before
    ``--execute`` — a dry run that models a different channel set than the real
    one is worse than no dry run, because it is read as confirmation.
    """
    print("Refusing to run: the channel selection was not stated.", file=sys.stderr)
    print("  'plan' and 'run' are separate invocations and share no state, so 'run'",
          file=sys.stderr)
    print("  ships NO default: an omitted --channels would drive all 16 after a",
          file=sys.stderr)
    print("  subset was planned, energising the ones you deliberately excluded.",
          file=sys.stderr)
    print("  State it explicitly, or run a saved plan, which carries it:",
          file=sys.stderr)
    print(f"    {CLI} run --channels 1-16 --execute", file=sys.stderr)
    print(f"    {CLI} run --from-plan equilibration_plan.toml --execute",
          file=sys.stderr)
    return EXIT_FAILED


def _refuse_bad_plan(exc: PlanFileError) -> int:
    """A plan file that cannot be executed stops the run. It never defaults.

    Falling back to the built-in defaults on an unreadable plan would be the
    original bug wearing a hat: the operator asked for *this* design, and the
    defaults are a ``Standard`` preset with no geometry.
    """
    print(f"Refusing to run: {exc}", file=sys.stderr)
    print("  --from-plan does NOT fall back to the built-in defaults -- those are",
          file=sys.stderr)
    print("  a 'Standard' preset and no electrode geometry, which is the run that",
          file=sys.stderr)
    print("  produced NULL sigma for every fit. Re-save the design first:",
          file=sys.stderr)
    print(f"    {CLI} plan ... --save <path>", file=sys.stderr)
    return EXIT_FAILED


def _seat_plan(args) -> list[tuple[str, Any, Any]]:
    """Resolve ``--from-plan`` onto the namespace, printing what it overrode.

    Runs before :func:`build_config`, which is what makes ``--from-plan`` and the
    mandatory ``--channels`` compose: the plan supplies the channel selection, so
    the refusal has nothing left to fire on.
    """
    source = getattr(args, "from_plan", None)
    if not source:
        return []
    design = load_plan(source)
    # `_RecordingParser` always supplies `raw_argv`; the fallback covers a
    # hand-built namespace, where "nothing was typed" makes the plan authoritative
    # — the safe direction, since the alternative is a design nobody saved.
    overrides = apply_plan(
        args, design, _explicit_design_flags(getattr(args, "raw_argv", ())))
    _print_plan_source(source, overrides)
    return overrides


def _cmd_run(args) -> int:
    try:
        overrides = _seat_plan(args)
    except PlanFileError as exc:
        return _refuse_bad_plan(exc)

    try:
        config = build_config(args)
    except ChannelsNotStated:
        return _refuse_unstated_channels()
    except (ChannelSpecError, ValueError) as exc:
        print(f"Cannot run: {exc}", file=sys.stderr)
        return EXIT_FAILED

    measured_per_channel_s = getattr(args, "measured_per_channel_s", None)
    _print_design(config, measured_per_channel_s=measured_per_channel_s)

    if not args.execute:
        print()
        print("Dry run — no instrument was opened and nothing was heated.")
        print("Re-run with --execute to drive the chamber.")
        return EXIT_OK

    # Before any driver is created: a declined run must leave the rig untouched,
    # and "are you sure you meant no geometry?" is a question about the design,
    # not about the hardware.
    if not confirm_no_geometry(config, assume_yes=args.yes):
        return EXIT_DECLINED

    from softae.drivers.factory import create_manager

    try:
        manager = create_manager(mock=True if args.mock else False)
    except Exception as exc:
        print(f"Could not open the instruments: {exc}", file=sys.stderr)
        print("  This run refuses to fall back to simulated drivers. Use --mock to",
              file=sys.stderr)
        print("  exercise the workflow without hardware.", file=sys.stderr)
        return EXIT_FAILED

    # Called before `_open_store` so a declined run leaves no orphan experiments
    # row. It is a no-op for a thermal run (see the module docstring) and is not
    # relied on — `confirm_thermal` below is the barrier.
    from softae.core.hardware_safety import assert_hardware_armed

    try:
        assert_hardware_armed(manager, action="run the equilibration characterization")
    except HardwareNotArmedError as exc:
        print(f"\nHardware is not armed: {exc}", file=sys.stderr)
        print(f"  Set {ARM_ENV_VAR}=1 in this shell and re-run.", file=sys.stderr)
        return EXIT_DECLINED

    if not confirm_thermal(config, assume_yes=args.yes,
                           measured_per_channel_s=measured_per_channel_s,
                           plan_overrides=overrides):
        return EXIT_DECLINED

    store, _project = _open_store(args)
    print(f"  recording to: {store.project_dir}")
    if args.mock:
        print("  ! MOCK — synthetic spectra. Do not derive a hold time from these.")

    run_id = store.start_run("equilibration_characterization", mode="characterization",
                             annotation=f"{config.rh_setpoint_pct:g}%RH, peak "
                                        f"{config.peak_temperature_C:g}C")
    finalize = run_finalizer(store, run_id)
    runner = EquilibrationRun(config, manager, data_store=store, run_id=run_id)
    renderer = ProgressRenderer(
        config, quiet=getattr(args, "quiet", False),
        milestone_interval_s=getattr(args, "telemetry_interval_s",
                                     DEFAULT_MILESTONE_INTERVAL_S))
    runner.on_progress = renderer

    async def _go():
        await manager.connect_all()
        try:
            return await runner.run()
        finally:
            await manager.disconnect_all()

    try:
        asyncio.run(_go())
    except EquilibrationAbort as exc:
        finalize("aborted")
        print(f"\nABORTED ({exc.kind}): {exc}", file=sys.stderr)
        _print_teardown(runner)
        return EXIT_FAILED
    except KeyboardInterrupt:
        finalize("interrupted")
        print("\nInterrupted.", file=sys.stderr)
        _print_teardown(runner)
        return EXIT_FAILED
    else:
        finalize("done")
    finally:
        # The catch-all, and it must run before `store.close()` — a closed
        # connection cannot record anything. Idempotent, so it is a no-op unless
        # an exception no `except` above names is on its way out.
        finalize("error")
        renderer.close()
        store.close()

    print()
    print(f"Recorded run {run_id}: {len(runner.points)} spectra, "
          f"{len(runner.holds)} watched hold windows.")
    _print_measured_cost(runner)
    if renderer.failures or runner.progress_failures:
        print(f"  (progress rendering dropped {renderer.failures} line(s); the run "
              f"was unaffected)")
    print(f"  {CLI} fit --run {run_id}")
    return EXIT_OK


def _print_teardown(runner: EquilibrationRun) -> None:
    """What the teardown **actually** did, read off the run rather than assumed.

    The two lines this replaces were reassurances, and one of them was simply
    false: ``"Interrupted — partial rounds are recorded"`` printed on a path where
    neither the ambient restore nor the sidecar write had run at all. Both facts
    now come from the run's own state.

    Everything goes to stderr, including the good news, so an operator running
    ``--quiet > run.log`` still sees on the terminal whether the chamber came
    down. The failure case names the setpoint the chamber may still be sitting at:
    that is the difference between an operator who walks away and one who does not.
    """
    if runner.restore_error:
        print(f"  !!!! AMBIENT WAS NOT RESTORED: {runner.restore_error}",
              file=sys.stderr)
        print(f"       The chamber may still be commanded to "
              f"{runner.last_commanded_description()}.", file=sys.stderr)
        print("       CHECK IT AT THE RIG -- nothing further will bring it down.",
              file=sys.stderr)
    else:
        print(f"  Ambient restored: {runner.config.temp_instrument} commanded to "
              f"{runner.config.ambient_C:g} C.", file=sys.stderr)
    if runner.sidecar_written:
        print(f"  Sidecar written ({len(runner.points)} spectra, "
              f"{len(runner.holds)} hold windows): {runner.sidecar_path()}",
              file=sys.stderr)
        return
    detail = runner.sidecar_error or "no run_id, so there was nowhere to write it"
    print(f"  ! THE SIDECAR WAS NOT WRITTEN: {detail}", file=sys.stderr)
    print("    The spectra are in the database, but the hold verdicts are not",
          file=sys.stderr)
    print("    reconstructable from it -- they exist nowhere else.", file=sys.stderr)


def _print_measured_cost(runner: EquilibrationRun) -> None:
    """What a round really cost, beside what the model said it would.

    This is the number ``estimate_eis_duration`` is answerable to — it is what the
    2026-08 recalibration was fitted against — so it is printed rather than left in
    the sidecar for someone to find. Every run adds an anchor, and a run whose
    ratio drifts from 1 is the earliest warning the model has gone stale again.
    """
    summary = runner.measured_cost_summary()
    rows = [(kind, summary[kind]) for kind in ("series",) if kind in summary]
    if not rows:
        return
    print("  Measured round cost (wall clock, includes the per-channel mux switch,")
    print("  upload, retrieval and file write the sweep model does not carry):")
    for kind, row in rows:
        ratio = row["ratio_measured_over_modelled"]
        print(f"    {kind:<7s} {row['measured_round_s']:7.1f}s "
              f"({row['measured_per_channel_s']:.1f}s/ch)  vs modelled "
              f"{row['modelled_round_s']:.1f}s"
              f"{'' if ratio is None else f'  = {ratio:.1f}x'}")
        print(f"            unmodelled overhead "
              f"{row['unmodelled_per_channel_s']:.1f}s/channel")


# ── fit ──────────────────────────────────────────────────────────────────────

def _load_run(args) -> tuple[Any, dict[str, Any] | None, str]:
    store, project = _open_store(args)
    return store, load_sidecar(store.project_dir, args.run), project


def _cmd_fit(args) -> int:
    store, sidecar, _project = _load_run(args)
    try:
        if sidecar is None:
            print(f"No equilibration sidecar for run '{args.run}'.", file=sys.stderr)
            print("  The coordinate (leg / setpoint / round / t) lives there — the",
                  file=sys.stderr)
            print("  database alone cannot say which point is which.", file=sys.stderr)
            return EXIT_FAILED

        series = load_sigma_series(store, args.run, sidecar)
        results = fit_run(series, model=args.model, run_id=args.run,
                          tol_rel=args.tol_rel, n_settle=args.n_settle)
        if not results:
            print("No sigma(t) series could be reconstructed. Check that the EIS "
                  "steps carried circuit_model and geometry.", file=sys.stderr)
            return EXIT_FAILED

        print(f"Run {args.run}: {len(results)} series, model '{args.model}'")
        print()
        for result in results:
            print("  " + result.describe())

        refused = [r for r in results if not r.fit_success]
        print()
        print(f"  {len(results) - len(refused)} fitted, {len(refused)} refused.")
        for refusal in sorted({r.refusal for r in refused if r.refusal}):
            n = sum(1 for r in refused if r.refusal == refusal)
            print(f"    {refusal}: {n}")

        sidecar["stats"] = [_stats_row(r) for r in results]
        sidecar["session_drift"] = session_drift(results, tol_rel=args.tol_rel)
        _rewrite_sidecar(store.project_dir, args.run, sidecar)
        print(f"  stats written back to the sidecar ({len(results)} rows).")
    finally:
        store.close()
    return EXIT_OK


def _stats_row(result) -> dict[str, Any]:
    return {"channel": result.channel, "leg": result.leg,
            "setpoint_index": result.setpoint_index, "model": result.model,
            "tau_s": _jsonable(result.tau_s),
            "tau_stderr_s": _jsonable(result.tau_stderr_s),
            "t_tol_s": result.t_tol_s, "tol_rel": result.tol_rel,
            "sigma_settled": _jsonable(result.sigma_settled),
            "noise_floor_rel": result.noise_floor_rel,
            "noise_floor_is_upper_bound": True,
            "r_squared": _jsonable(result.r_squared), "n_points": result.n_points,
            "fit_success": result.fit_success, "refusal": result.refusal,
            # The cell-constant cross-check, carried beside sigma and never
            # instead of it.
            "tau_r1_s": _jsonable(result.tau_r1_s),
            "r1_fit_success": result.r1_fit_success,
            "r1_refusal": result.r1_refusal,
            "tau_agreement_rel": result.tau_agreement_rel,
            "r1_diagnostic_ok": result.r1_diagnostic_ok}


def _jsonable(value) -> Any:
    import math

    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _rewrite_sidecar(project_dir, run_id: str, payload: dict[str, Any]) -> None:
    import json

    path = Path(project_dir) / "runs" / str(run_id) / "equilibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── report ───────────────────────────────────────────────────────────────────

def _cmd_report(args) -> int:
    store, sidecar, _project = _load_run(args)
    try:
        if sidecar is None:
            print(f"No equilibration sidecar for run '{args.run}'.", file=sys.stderr)
            return EXIT_FAILED

        stats = sidecar.get("stats")
        drift = sidecar.get("session_drift")
        if not stats:
            series = load_sigma_series(store, args.run, sidecar)
            results = fit_run(series, model=args.model, run_id=args.run,
                              tol_rel=args.tol_rel, n_settle=args.n_settle)
            stats = [_stats_row(r) for r in results]
            drift = session_drift(results, tol_rel=args.tol_rel)

        print(f"Run {args.run} — equilibration report")
        print()
        print("  Hold verdicts (an unmet setpoint is a RESULT here, not a failure):")
        for sp in sidecar.get("setpoints") or []:
            print(f"    {sp['leg']:>4s}/S{sp['setpoint_index']} "
                  f"{sp['temperature_C']:>6.1f}C  T held={_verdict(sp['hold_met'])}  "
                  f"RH held={_verdict(sp.get('rh_hold_met'))}  "
                  f"T pv {_span(sp.get('hold_pv_min'), sp.get('hold_pv_max'))}  "
                  f"RH pv {_span(sp.get('rh_hold_pv_min'), sp.get('rh_hold_pv_max'))}")
        if sidecar.get("aborted"):
            print(f"    ! RUN ABORTED: {sidecar.get('abort_reason', '')}")
        print()
        print("  Per series:")
        for row in stats:
            tau = ("REFUSED(%s)" % (row["refusal"] or "?") if not row["fit_success"]
                   else f"{row['tau_s']:.0f}s")
            t_tol = "never" if row["t_tol_s"] is None else f"{row['t_tol_s']:.0f}s"
            floor = ("?" if row["noise_floor_rel"] is None
                     else f"<={row['noise_floor_rel'] * 100:.2f}%")
            print(f"    ch{row['channel']:<3d} {row['leg']:>4s}/S"
                  f"{row['setpoint_index']}  tau={tau:<22s} t_tol={t_tol:<8s} "
                  f"noise={floor}")
        print()
        print("  The noise floor is measurement noise PLUS residual short-term drift:")
        print("  an UPPER BOUND on pure measurement noise. Separating them needs a")
        print("  repeat with zero time between, which is not physically available.")
        _print_thickness(sidecar)
        _print_arc_closure(arc_closure_rates(store, args.run, sidecar))
        _print_r1_diagnostic(stats)
        _print_session_drift(drift or [])

        return _endorse(stats, args.tol_rel)
    finally:
        store.close()


def _print_session_drift(rows: list[dict[str, Any]]) -> None:
    """Start of the up leg vs end of the down leg, at the same nominal condition.

    This is what the retired ``Longest`` anchor rounds were meant to provide; it
    is obtained here from data the σ(t) series already produced, for no instrument
    time at all.
    """
    print()
    if not rows:
        print("  Session drift: NOT AVAILABLE (needs both legs at the same "
              "reference condition).")
        return
    print("  Session drift -- first settled block (up/S0) vs last (down), same")
    print("  nominal condition, graded against this run's OWN measured noise floor:")
    for row in rows:
        drift = row["drift_rel"]
        floor = row["noise_floor_rel"]
        if drift is None:
            verdict, detail = "?", "no settled sigma at one end"
        elif row["significant"] is None:
            verdict, detail = "?", f"{drift * 100:.2f}% but no noise floor measured"
        else:
            verdict = "DRIFTED" if row["significant"] else "stable"
            detail = f"{drift * 100:.2f}% vs floor {floor * 100:.2f}%"
        print(f"    ch{row['channel']:<3d} {verdict:<8s} {detail}")
    print("  It doubles as the retrace evidence at the reference point: the up and")
    print("  down legs meet there, so a drift here is also a failure to retrace.")


def _print_thickness(sidecar: dict[str, Any]) -> None:
    """Say which tier the thickness — and therefore every σ here — rests on."""
    thickness = sidecar.get("thickness") or {}
    if not thickness:
        return
    print()
    print(f"  Thickness: {thickness.get('value_um')} um, method "
          f"'{thickness.get('thickness_method')}'.")
    if thickness.get("thickness_method") == "target":
        print("  A TARGET, not a measurement: every sigma above is divided by a")
        print("  hand-computed number. fit_results.thickness_method is NULL for this")
        print("  run, so this sidecar is the only record that says so.")


def _print_arc_closure(summary: dict[str, Any]) -> None:
    """How much of each block's σ rests on an extrapolated R₁.

    A spectrum whose ``−Z″`` has not peaked by the lowest swept frequency gives R₁
    by extrapolating off the high-frequency side. The number is still usable, so
    nothing above excludes it — but a block where most spectra are in that state
    has a σ, and a σ scatter, set partly by the circuit model, and the operator's
    two decisions here are whether to trust that block's noise floor and whether
    the cold end needs a slower preset.
    """
    n = summary["n"]
    print()
    if not n:
        print("  Arc closure: NOT AVAILABLE (no stored payloads for this run).")
        return
    n_open = summary["n_open"]
    print(f"  Arc closure: {n_open} of {n} spectra ({n_open / n * 100:.0f}%) did not "
          f"close inside the")
    print("  swept window; their R1 is extrapolated off the high-frequency side.")
    if summary["n_unknown"]:
        print(f"    {summary['n_unknown']} could not be judged (non-finite or too "
              f"short a sweep).")
    blocks = summary["by_block"]
    if blocks:
        print("    by block:  " + "  ".join(
            f"{leg}/S{sp} {o / t * 100:.0f}%"
            for (leg, sp), (o, t) in sorted(blocks.items(),
                                            key=lambda kv: (kv[0][0] != "up", kv[0][1]))))
    worst = sorted(summary["by_channel"].items(), key=lambda kv: -kv[1][0])[:4]
    if worst and worst[0][1][0]:
        print("    worst channels:  " + "  ".join(
            f"ch{ch} {o}/{t}" for ch, (o, t) in worst))


def _print_r1_diagnostic(stats: list[dict[str, Any]]) -> None:
    """The τ(σ) vs τ(R₁) cross-check — a free test of the cell-constant path.

    σ = K/R₁ with K constant during a hold, so the two τ must agree exactly. A
    material difference is not a fitting curiosity: it means K was not constant,
    or σ was not derived from this R₁, and every conductivity in the run inherits
    that.
    """
    checked = [r for r in stats if r.get("tau_agreement_rel") is not None]
    print()
    if not checked:
        refusals = sorted({r.get("r1_refusal") or "not attempted" for r in stats})
        print(f"  tau(R1) cross-check: NOT AVAILABLE ({', '.join(refusals)}).")
        print("  R1 is NULL whenever a circuit fit failed; it does not stand in for")
        print("  a missing sigma, and 'not checked' is not 'checked and fine'.")
        return
    bad = [r for r in checked if not r.get("r1_diagnostic_ok")]
    worst = max(r["tau_agreement_rel"] for r in checked)
    print(f"  tau(R1) cross-check: {len(checked) - len(bad)}/{len(checked)} agree "
          f"(worst {worst * 100:.2f}%, tolerance "
          f"{R1_AGREEMENT_TOL_REL * 100:.0f}%).")
    for row in bad[:8]:
        print(f"    ! ch{row['channel']} {row['leg']}/S{row['setpoint_index']}: "
              f"tau={row['tau_s']:.0f}s but tau(R1)={row['tau_r1_s']:.0f}s "
              f"({row['tau_agreement_rel'] * 100:.1f}%)")
    if bad:
        print("    sigma = K/R1 with K constant, so these must be identical. They are")
        print("    not: something in the cell-constant path is wrong.")


def _verdict(value) -> str:
    return "?" if value is None else ("yes" if value else "NO")


def _span(lo, hi) -> str:
    if lo is None or hi is None:
        return "?"
    return f"{lo:.1f}..{hi:.1f}"


def _endorse(stats: list[dict[str, Any]], tol_rel: float) -> int:
    """Refuse to endorse a conditioning tolerance below the measured noise floor.

    Printed as a refusal and exited non-zero rather than printed with a caveat: a
    hold time derived from an unachievable tolerance is a number with no meaning,
    and it would go straight into the campaign's conditioning config.
    """
    print()
    print(f"  Proposed conditioning tolerance: {tol_rel * 100:.2f}%")
    refused = []
    for row in stats:
        ok, why = endorse_tolerance(tol_rel, row["noise_floor_rel"])
        if not ok:
            refused.append((row, why))
    if not refused:
        print("    endorsed for every series measured here.")
        return EXIT_OK
    print(f"    REFUSED for {len(refused)} of {len(stats)} series:")
    for row, why in refused[:12]:
        print(f"      ch{row['channel']} {row['leg']}/S{row['setpoint_index']}: {why}")
    if len(refused) > 12:
        print(f"      (+{len(refused) - 12} more)")
    print("    No hold time is printed for these: a tolerance below the noise floor")
    print("    can never be satisfied, however long the hold.")
    return EXIT_FAILED


# ── Entry point ──────────────────────────────────────────────────────────────

def _add_design_args(parser: argparse.ArgumentParser, *,
                     required_channels: bool = False) -> None:
    """The design surface, identical on ``plan`` and ``run`` but for one default.

    ``required_channels`` is the whole difference, declared once: ``run`` gets no
    default channel set, so an operator who planned a subset and then typed
    ``run --execute`` is refused rather than silently given all sixteen. It is
    **not** ``argparse``'s ``required=True``, which exits 2 with a message that
    explains none of that — ``build_config`` raises and ``_cmd_run`` prints the
    reason and the flag.

    Split across three ``--help`` groups because the surface is large and the
    three answer different questions — *what is measured*, *when a setpoint may
    stop*, and *what the chamber must do* — and an operator reaching for a
    timeout should not have to read the geometry flags to find it. The grouping
    is cosmetic only: every flag below lands in the saved plan exactly as before,
    and :data:`PLAN_DESIGN_KEYS` is what decides that, not the group.
    """
    design = parser.add_argument_group(
        "the experiment", "what is measured, where, and at what cost")
    design.add_argument("--channels", default=None if required_channels else "1-16",
                        help='e.g. "1-16" or "2,4,5-10"'
                             + (" -- REQUIRED here; never inherited from 'plan'"
                                if required_channels else ""))
    design.add_argument("--temperatures", default="27.5,45,65,85",
                        help="setpoints for the UP leg; the down leg retraces them")
    design.add_argument("--legs", default="up,down")
    design.add_argument("--rh", type=float, default=DEFAULT_RH_SETPOINT_PCT,
                        help=f"%%RH setpoint, re-established at EVERY temperature "
                             f"(default {DEFAULT_RH_SETPOINT_PCT:g}). NOT 15: the "
                             f"flush basin holds water inside the heated enclosure, "
                             f"so warming the chamber humidifies it with surplus "
                             f"moisture. Commanded 15 on 2026-08-11 and measured a PV "
                             f"of 16.9-20.4 at 65 C and 19.5-23.2 at 85 C -- 15 is "
                             f"below what this enclosure can deliver hot, and asking "
                             f"for it grades every hot setpoint unmet for a reason "
                             f"that is plumbing, not control.")
    design.add_argument("--rounds", type=int, default=15,
                        help=f"the CEILING on sigma(t) rounds per setpoint -- NOT a "
                             f"fixed count. A setpoint stops as soon as sigma has "
                             f"settled (see --settle-tol-rel) and the hold floor has "
                             f"elapsed; it runs this many only when it has not. Inside "
                             f"the --tau-setpoints window it must also have run "
                             f"{MIN_POINTS_FOR_TAU} rounds -- the offline fitter's own "
                             f"MIN_POINTS_FOR_TAU, imported rather than retyped: "
                             f"sigma(t) has three free parameters, so a shorter series "
                             f"is REFUSED for tau, and the run must not be able to "
                             f"acquire a setpoint it cannot analyse. A ceiling below "
                             f"{MIN_POINTS_FOR_TAU} is refused outright -- no setpoint "
                             f"in such a run could ever yield a tau.")
    _add_settle_args(parser)
    design.add_argument("--round-period-s", dest="round_period_s", type=float,
                        default=DEFAULT_ROUND_PERIOD_S,
                        help=f"sigma(t) sampling interval. Default "
                             f"{DEFAULT_ROUND_PERIOD_S:g}s is two terms kept apart: "
                             f"{DEFAULT_N_CHANNELS} channels (the --channels default) "
                             f"x the per-channel cost of "
                             f"'{DEFAULT_EIS_PRESET}' (the --preset default), which is "
                             f"derived, plus a CHOSEN {ROUND_BUFFER_S:g}s per-round "
                             f"buffer for executor and mscr overhead, rounded up to a "
                             f"typable ten. That cost is currently MODELLED, not "
                             f"measured -- the 7 Hz floor has never been timed here. "
                             f"It does NOT bound the settle check, which asks only "
                             f"whether sigma has gone flat for --settle-n-rounds "
                             f"rounds and needs no tau at all. It bounds the tau "
                             f"DIAGNOSTIC: the shortest resolvable tau is ~2x this, so "
                             f"~{2 * DEFAULT_ROUND_PERIOD_S / 60:.0f} min at "
                             f"{DEFAULT_N_CHANNELS} channels -- which is the "
                             f"SATURATED-BOARD case, not a typical batch. A batch of "
                             f"4/6/8 channels derives 120/160/200s and resolves tau "
                             f"down to ~4.0/5.3/6.7 min, comfortably under the ~8.3 "
                             f"min tau once seen at a first setpoint (one observation, "
                             f"not a target; tau moves with sample, formulation and "
                             f"setpoint). FEWER CHANNELS is the lever, not a longer "
                             f"period: the cost is PER CHANNEL, and a period a round "
                             f"does not fit inside is not honoured, it is simply "
                             f"exceeded.")
    design.add_argument("--preset", default=DEFAULT_EIS_PRESET,
                        help=f"EIS preset for the series. Default "
                             f"'{DEFAULT_EIS_PRESET}': at 'Standard' an all-channel "
                             f"round costs 654s, forcing a sampling interval whose "
                             f"tau floor (~22 min) is coarser than the ~8.3 min tau "
                             f"once seen at a first setpoint -- a tau of that order "
                             f"could not be fitted at all.")
    # A frequency in the flag, a CONDUCTIVITY in the help: the floor is only
    # incidentally about Hz.
    #
    # No default, and that is the point: `[eis_presets.Quick]` already carries the
    # 7 Hz floor, so an ordinary run needs no override at all. Defaulting this to
    # 7000 would make every run look like an override and downgrade its own
    # duration to EXTRAPOLATED for a reason that is not true. This flag exists for
    # the run that needs a DIFFERENT floor -- a material two decades less
    # conductive -- where editing global config for one night is the wrong tool.
    design.add_argument("--f-lo-mHz", dest="f_lo_mHz", type=int, default=None,
                        help=f"lowest sweep frequency in mHz, overriding the "
                             f"preset's own (default: the preset's, "
                             f"{_preset_f_lo_hz(DEFAULT_EIS_PRESET):g} Hz on "
                             f"'{DEFAULT_EIS_PRESET}'). This is a CONDUCTIVITY "
                             f"floor wearing a frequency's clothes: the -Z\" peak "
                             f"sits at f = 1/(2*pi*R*C_cell), so a sample below "
                             f"sigma = 2*pi*f_lo*C_cell*L/(t*w) never closes its "
                             f"arc and its R1 is EXTRAPOLATED off the "
                             f"high-frequency limb -- measured at a 61%% median "
                             f"overestimate (175%% with a CPE fit), a systematic "
                             f"bias and not a widened error bar. The preset's "
                             f"{_preset_f_lo_hz(DEFAULT_EIS_PRESET):g} Hz reaches "
                             f"sigma ~ {_reference_sigma_floor():.1e} S/cm at "
                             f"{_reference_geometry_str()}, about a decade below "
                             f"anything this rig has measured. The relationship is "
                             f"LINEAR -- to reach a tenth the conductivity, take a "
                             f"tenth the floor -- and the plan prints the reach "
                             f"this run actually buys.")
    # No default. The measured number belongs to one rig and one preset, and a
    # constant applied silently would be wrong the moment either changed -- while
    # still reading, to the next operator, like a prediction.
    design.add_argument("--measured-per-channel-s", dest="measured_per_channel_s",
                        type=float, default=None,
                        help="plan from a MEASURED per-channel round cost instead of "
                             "the model. Rarely needed since 2026-08-17: every "
                             "shipped preset was stopwatched that day, so a plan on "
                             "a stock preset already runs off a real measurement "
                             f"({MEASURED_PER_CHANNEL_S_STANDARD:g} s/channel on "
                             f"'Standard', {EIS_MEASURED_S_PER_CHANNEL[DEFAULT_EIS_PRESET]:g} "
                             f"on '{DEFAULT_EIS_PRESET}'). Reach for this when the "
                             "sweep is NOT a stock preset -- an overridden --f-lo-mHz "
                             "leaves every timed grid behind and falls back to the "
                             "model, which is fitted to those presets and runs up to "
                             "~10%% under them. Overrides the modelled cost in the "
                             "round cost, the inter-round gap, the headroom, the "
                             "sampling interval and the total duration.")
    # `--model` meant the EIS CIRCUIT model here and the RELAXATION model on
    # `fit`/`report` -- one spelling, two vocabularies, both plausible-looking
    # strings, so `fit --model simpleSalt` failed confusingly rather than
    # obviously. Each name is now unambiguous and each help text names the other,
    # with `--model` kept as a working alias on both so no saved script breaks.
    # `dest` stays `model`, so plan files written before the rename still load.
    design.add_argument("--circuit-model", "--model", dest="model",
                        default="simpleSalt",
                        help="EIS CIRCUIT model fitted to each spectrum (e.g. "
                             "simpleSalt). Not the relaxation model: that is "
                             "--relaxation-model on 'fit'/'report'. '--model' is a "
                             "deprecated alias for this flag here.")
    design.add_argument("--electrode-l-cm", dest="electrode_l_cm", type=float)
    design.add_argument("--electrode-t-cm", dest="electrode_t_cm", type=float)
    design.add_argument("--electrode-w-cm", dest="electrode_w_cm", type=float)
    design.add_argument("--thickness-method", dest="thickness_method",
                        default="target", choices=list(THICKNESS_METHODS),
                        help="how --electrode-t-cm was obtained. Default 'target': "
                             "a hand-computed digital-twin number, NOT a measurement. "
                             "Recorded in the run sidecar because "
                             "fit_results.thickness_method stays NULL for this run.")
    _add_chamber_args(parser)
    parser.add_argument("--project", help="project directory (default: [data] project_dir)")
    parser.add_argument("--mock", action="store_true")


def _add_chamber_args(parser: argparse.ArgumentParser) -> None:
    """The bands and allowances — **what counts as held, and how long it may take**.

    Not settable without editing source until now, and the omission cost a real
    run: the operator hit ``approach_timeout_s`` at the last down-leg setpoint on
    2026-08-11 and had no flag to extend it, so fifteen rounds labelled 27.5 °C
    were taken while the stage fell from 34.1 to 29.0 °C. On the **shared** design
    surface for the same reason the settle criterion is: these decide the verdict
    that is this run's primary result, and one that reverted between ``plan`` and
    ``run`` would change what a recorded ``hold_met`` means with nothing on disk
    saying so.
    """
    group = parser.add_argument_group(
        "the chamber", "what counts as HELD, and how long the chamber is given "
                       "to get there. Every one of these lands in the saved plan")
    group.add_argument("--tolerance-c", dest="tolerance_C", type=float,
                       default=DEFAULT_TOLERANCE_C,
                       help=f"half-width of the temperature band that decides "
                            f"'held' (default {DEFAULT_TOLERANCE_C:g} C). NOT 0.5: at "
                            f"0.5 a 0.6 C dip (PV 64.4 against 65.0) graded a whole "
                            f"setpoint 'hold not met' on a chamber that wanders a few "
                            f"tenths, and an unmet verdict that is not a failure "
                            f"teaches an operator to ignore unmet verdicts. Sits "
                            f"inside --warn-c ({DEFAULT_WARN_C:g}), so an excursion is "
                            f"still warned before the band is anywhere near --fault-c.")
    group.add_argument("--rh-tolerance-pct", dest="rh_tolerance_pct", type=float,
                       default=DEFAULT_RH_TOLERANCE_PCT,
                       help=f"the same band on humidity (default "
                            f"{DEFAULT_RH_TOLERANCE_PCT:g} %%RH). Whether the rig can "
                            f"hold the setpoint hot is a PRIMARY RESULT of this run, "
                            f"so widening this to make verdicts look better answers "
                            f"the question it was asked to measure.")
    group.add_argument("--warn-c", dest="warn_C", type=float, default=DEFAULT_WARN_C,
                       help=f"deviation that logs an excursion (default "
                            f"{DEFAULT_WARN_C:g} C). Counted, never fatal, and it must "
                            f"stay above --tolerance-c or every out-of-band sample "
                            f"warns and the count says nothing.")
    group.add_argument("--fault-c", dest="fault_C", type=float, default=DEFAULT_FAULT_C,
                       help=f"the RUNAWAY guard (default {DEFAULT_FAULT_C:g} C). A "
                            f"sustained PV above target+this for --grace-s aborts the "
                            f"run and restores ambient. Not a tolerance: failing to "
                            f"REACH a setpoint is recorded and continues, overshooting "
                            f"one is a hazard and stops.")
    group.add_argument("--grace-s", dest="grace_s", type=float, default=DEFAULT_GRACE_S,
                       help=f"how long an overshoot must be sustained before it is a "
                            f"runaway rather than a transient (default "
                            f"{DEFAULT_GRACE_S:g}s).")
    group.add_argument("--approach-timeout-s", dest="approach_timeout_s", type=float,
                       default=DEFAULT_APPROACH_TIMEOUT_S,
                       help=f"how long temperature may take to reach band on the UP "
                            f"leg, where the heater is driving (default "
                            f"{DEFAULT_APPROACH_TIMEOUT_S:g}s). Timing out is not an "
                            f"abort: the run measures from wherever the PV got to and "
                            f"records the approach as not reached.")
    group.add_argument("--down-approach-timeout-s", dest="down_approach_timeout_s",
                       type=float, default=DEFAULT_DOWN_APPROACH_TIMEOUT_S,
                       help=f"the same allowance on the DOWN leg, where nothing is "
                            f"driving (default {DEFAULT_DOWN_APPROACH_TIMEOUT_S:g}s). "
                            f"Cooling is passive and asymptotic: measured down-leg "
                            f"approaches were 0.5 min at 85 C, 12.0 at 65, 22.5 at 45 "
                            f"and 30.0 at 27.5 -- where it hit the "
                            f"{DEFAULT_APPROACH_TIMEOUT_S:g}s timeout WITHOUT reaching "
                            f"tolerance, so 15 rounds labelled 27.5 C spanned a 5 C "
                            f"ramp (34.1 C at the first, 29.0 C at the last). The "
                            f"default is that {DEFAULT_APPROACH_TIMEOUT_S:g}s plus "
                            f"~60 min: at the measured end-of-series rate of ~5 C per "
                            f"44 min, 34.1 C -> 27.5 C needs ~58 min more. A separate "
                            f"timeout rather than a factor, so the number you are "
                            f"extending is visible.")
    group.add_argument("--rh-approach-timeout-s", dest="rh_approach_timeout_s",
                       type=float, default=DEFAULT_RH_APPROACH_TIMEOUT_S,
                       help=f"how long humidity may take to reach band, at EVERY "
                            f"temperature (default "
                            f"{DEFAULT_RH_APPROACH_TIMEOUT_S:g}s). Not the 120 s "
                            f"AsyncRHController.wait default: holding one %%RH from "
                            f"27.5 to 85 C moves the absolute water content ~9.6x.")


def _add_settle_args(parser: argparse.ArgumentParser) -> None:
    """The adaptive stopping rule, on the **shared** design surface.

    Declared here rather than on ``run`` alone so it lands in a saved plan like
    every other design value: how long a setpoint is held is as much a part of
    the experiment as which temperatures it visits, and a criterion that reverted
    to its default between ``plan`` and ``run`` would be the 2026-08-10 defect
    again in a new field.

    ``--settle on|off`` rather than ``--no-settle``, because a store_true cannot
    be written into a plan file and retyped from it: ``design_flags`` renders
    every resolved value as a flag an operator could paste back.
    """
    group = parser.add_argument_group(
        "the settle criterion", "when a setpoint is allowed to stop short of "
                                "--rounds, and the floors under that")
    group.add_argument("--settle", dest="settle", choices=("on", "off"),
                       default="on",
                       help="stop a setpoint once sigma has settled instead of "
                            "always running --rounds. 'off' restores the old "
                            "fixed-count behaviour exactly.")
    group.add_argument("--settle-tol-rel", dest="settle_tol_rel", type=float,
                       default=DEFAULT_SETTLE_TOL_REL,
                       help=f"relative half-width of the settle band (default "
                            f"{DEFAULT_SETTLE_TOL_REL:g}). MUST exceed the run's own "
                            f"noise floor -- 5.98%% median was measured over 96 "
                            f"series, with 22 of them above 20%% -- or no hold "
                            f"length can satisfy it and every setpoint runs to its "
                            f"ceiling. The run says so per setpoint when it happens.")
    group.add_argument("--settle-n-rounds", dest="settle_n_rounds", type=int,
                       default=DEFAULT_SETTLE_N_ROUNDS,
                       help=f"consecutive rounds that must all sit inside the band "
                            f"(default {DEFAULT_SETTLE_N_ROUNDS}). A DETECTION "
                            f"WINDOW, and a different question from how many points "
                            f"a tau needs: setting it below {MIN_POINTS_FOR_TAU} "
                            f"narrows the window exactly as asked and is NOT "
                            f"rewritten. Inside the --tau-setpoints window the "
                            f"setpoint still cannot stop before {MIN_POINTS_FOR_TAU} "
                            f"rounds, because the fitter refuses a shorter series.")
    group.add_argument("--settle-min-channels", dest="settle_min_channels", type=int,
                       default=DEFAULT_SETTLE_MIN_CHANNELS,
                       help=f"fewest channels that must carry usable evidence for "
                            f"the criterion to be evaluated at all (default "
                            f"{DEFAULT_SETTLE_MIN_CHANNELS}). A channel whose sigma "
                            f"is NULL, or whose R1 railed on the circuit model's "
                            f"lower bound, does NOT count -- a railed fit returns "
                            f"the same number every round and a constant is "
                            f"trivially 'settled'. Below this the setpoint runs to "
                            f"its ceiling and records 'not_evaluable'.")
    group.add_argument("--min-hold-first-s", dest="min_hold_first_s", type=float,
                       default=DEFAULT_MIN_HOLD_FIRST_S,
                       help=f"floor on the hold at the FIRST setpoint of the run "
                            f"(default {DEFAULT_MIN_HOLD_FIRST_S:g}s ~ 3 tau, with "
                            f"tau = 425-575s measured while the films dry from "
                            f"ambient down to the RH setpoint). The whole transient "
                            f"is here. A TIME floor: the rounds it buys is "
                            f"ceil(it / --round-period-s), and the effective minimum "
                            f"is that against {MIN_POINTS_FOR_TAU} rounds and "
                            f"--settle-n-rounds, whichever is largest.")
    group.add_argument("--min-hold-s", dest="min_hold_s", type=float,
                       default=DEFAULT_MIN_HOLD_S,
                       help=f"floor on every later setpoint (default "
                            f"{DEFAULT_MIN_HOLD_S:g}s). The films are already dry, "
                            f"but the chamber still has to re-establish RH. A TIME "
                            f"floor like the one above: at the default "
                            f"{DEFAULT_ROUND_PERIOD_S:g}s period it buys "
                            f"ceil({DEFAULT_MIN_HOLD_S:g}/{DEFAULT_ROUND_PERIOD_S:g}) "
                            f"rounds, so inside the --tau-setpoints window "
                            f"{MIN_POINTS_FOR_TAU} -- the fitter's minimum -- is what "
                            f"actually binds and outside it --settle-n-rounds does; "
                            f"'plan' prints the effective figure for every regime.")
    group.add_argument("--tau-setpoints", dest="tau_setpoints", type=int,
                       default=DEFAULT_TAU_SETPOINTS,
                       help=f"how many setpoints OF THE RUN (not of each leg) carry "
                            f"the {MIN_POINTS_FOR_TAU}-round floor that guarantees a "
                            f"fittable tau (default {DEFAULT_TAU_SETPOINTS}). The "
                            f"films dry ONCE, at the start, and stay dry: measured "
                            f"per-setpoint sigma swing on the up leg was 1600-2800%% "
                            f"at S0 and 57-1370%% at S1, then 0.5-8.5%% at S2 and "
                            f"0.8-3.1%% at S3, against a 5.98%% noise floor. Past S1 "
                            f"there is no relaxation left to fit, so forcing "
                            f"{MIN_POINTS_FOR_TAU} rounds there buys a tau nobody can "
                            f"use. Beyond the Nth setpoint the floor is "
                            f"max(--settle-n-rounds, ceil(hold floor / period)) "
                            f"alone. 0 removes it everywhere; a value at or above the "
                            f"setpoint count restores it everywhere.")


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="run_id of a recorded run")
    parser.add_argument("--relaxation-model", "--model", dest="model",
                        default="exponential",
                        choices=sorted(EQUILIBRATION_MODELS),
                        help="RELAXATION model fitted to sigma(t) ('none' = t_tol "
                             "only, no fit). Not the circuit model: that is "
                             "--circuit-model on 'plan'/'run'. '--model' is a "
                             "deprecated alias for this flag here.")
    parser.add_argument("--tol-rel", dest="tol_rel", type=float, default=DEFAULT_TOL_REL,
                        help="relative settling tolerance (default 0.02)")
    parser.add_argument("--n-settle", dest="n_settle", type=int,
                        default=DEFAULT_N_SETTLE)
    parser.add_argument("--project", help="project directory (default: [data] project_dir)")
    parser.add_argument("--mock", action="store_true")


def _add_verbosity(parser: argparse.ArgumentParser) -> None:
    """``-v`` on the top-level parser *and* on every subcommand.

    ``default=argparse.SUPPRESS`` is load-bearing, not tidiness: a subparser
    copies its own defaults over the outer namespace after it parses, so a plain
    ``default=False`` here would silently discard
    ``python -m softae.tools.equilibration -v run ...`` — the exact spelling an
    operator reaches for when a run is misbehaving.
    """
    parser.add_argument("-v", "--verbose", action="store_true",
                        default=argparse.SUPPRESS,
                        help="DEBUG logging, overriding [logging] level. Noisy: "
                             "the RH controller logs a duty cycle on every update.")


def configure_logging(verbose: bool = False) -> int:
    """Filter the log stream once, and return the level applied.

    Nothing else configures structlog on this path. The GUI does it at
    ``gui/app.py``; a headless entry point that skips it inherits structlog's
    default ``PrintLogger``, which emits **every** level — including
    ``rh_duty_sent``, logged on each RH control update. Over a six-hour
    unattended run that buries the run's own reporting in DEBUG.

    It does not touch that reporting. :class:`ProgressRenderer` writes the
    milestones, hold verdicts, telemetry lines and the live status line straight
    to stdout, and the workflow's milestone log calls are ``info``/``warning``.
    Filtering at INFO leaves every one of them visible, which is the whole point:
    the operator loses the spam and keeps the run.
    """
    from softae.config import loader

    level = (logging.DEBUG if verbose
             else getattr(logging, loader.log_level(), logging.INFO))
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(level))
    return level


class _RecordingParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that keeps the argv it was handed.

    ``argparse`` discards which flags were *typed* — a namespace value equal to
    its default is indistinguishable from one nobody supplied. ``--from-plan``
    needs that distinction to tell an override from a value the plan supplied,
    and getting it wrong in either direction is a silent design change. Recording
    the argv here means every caller has it, including tests that build a
    namespace directly rather than going through :func:`main`.
    """

    def parse_args(self, args=None, namespace=None):      # type: ignore[override]
        parsed = super().parse_args(args, namespace)
        parsed.raw_argv = list(sys.argv[1:] if args is None else args)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    p = _RecordingParser(
        prog=CONSOLE_SCRIPT,
        description="Record sigma(t) while the chamber is brought to condition, and "
                    "derive the conditioning hold time from it.",
        epilog=f"Read-only first: 'plan ... --save plan.toml', then "
               f"'run --from-plan plan.toml --execute' at the rig, then fit and "
               f"report. Invoke as '{CLI}': the '{CONSOLE_SCRIPT}' console script "
               f"is declared in pyproject.toml but is not installed in this venv.",
    )
    _add_verbosity(p)
    sub = p.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="the design, the budget, and what will refuse")
    _add_design_args(plan)
    plan.add_argument("--fixture", default="default", help="calibration fixture id")
    plan.add_argument("--save", metavar="PATH",
                      help="write the FULLY RESOLVED design (defaults included) to "
                           "a TOML file that 'run --from-plan' executes verbatim. "
                           "'plan' and 'run' share no state, so any flag not "
                           "repeated on 'run' reverts to its default; this is how "
                           "it stops doing that.")
    plan.set_defaults(func=_cmd_plan)

    run = sub.add_parser("run", help="execute the run (DRY RUN unless --execute)")
    _add_design_args(run, required_channels=True)
    run.add_argument("--from-plan", dest="from_plan", metavar="PATH",
                     help="execute a design saved by 'plan --save'. Supplies "
                          "--channels, so that flag is not required alongside it. "
                          "Flags typed as well still win, but every one is printed "
                          "as a diff against the file and repeated in the thermal "
                          "confirmation.")
    run.add_argument("--execute", action="store_true",
                     help="actually drive the chamber. Without it, nothing is opened.")
    run.add_argument("--yes", "-y", action="store_true",
                     help="skip the thermal confirmation prompt")
    run.add_argument("--quiet", action="store_true",
                     help="suppress the live status line. Milestones, hold "
                          "verdicts and the controls-telemetry lines still print, "
                          "and everything still reaches structlog.")
    run.add_argument("--telemetry-interval-s", dest="telemetry_interval_s",
                     type=float, default=DEFAULT_MILESTONE_INTERVAL_S,
                     help="seconds between controls-monitor lines (default 300)")
    run.set_defaults(func=_cmd_run)

    fit = sub.add_parser("fit", help="re-fit tau / t_tol offline from a recorded run")
    _add_analysis_args(fit)
    fit.set_defaults(func=_cmd_fit)

    rep = sub.add_parser("report", help="tau, t_tol, noise floor and hold verdicts")
    _add_analysis_args(rep)
    rep.set_defaults(func=_cmd_report)

    for parser in (plan, run, fit, rep):
        _add_verbosity(parser)
    return p


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    # Before dispatch, so every subcommand is covered and there is exactly one
    # place the level is decided.
    configure_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
