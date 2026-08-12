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
measured, against ``Quick``'s 10.47) and the electrode geometry was dropped
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
from softae.tools import use_utf8_console
from softae.workflows.equilibration import (
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
    VERDICT_ABORTED,
    VERDICT_MET,
    VERDICT_UNMET,
    EquilibrationAbort,
    EquilibrationConfig,
    EquilibrationRun,
    inter_round_gap_s,
    load_sidecar,
    minimum_feasible_period_s,
    project_duration,
    round_cost_s,
    round_headroom_s_per_channel,
)

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

#: Below this much per-channel headroom, ``plan`` cautions that the configured
#: round period is unlikely to contain a real round. It is a **warning threshold**,
#: not an overhead model: the mux switch, the script upload, the data retrieval and
#: the file write are per-channel costs ``estimate_eis_duration`` does not carry,
#: and bench observation puts them at ~9-10 s each (a measured ~10-12 s/channel
#: against a 1.4 s/channel model for ``Quick``). No rig's number is written into
#: the projection — the run measures its own and records it; this threshold only
#: decides whether a caution is printed.
OVERHEAD_HEADROOM_WARN_S = 10.0

#: What one channel of a ``Standard`` round **actually** cost on this rig, measured
#: over 12 channels: 40.7 s, and essentially constant channel to channel (40.687 /
#: 40.719 / 40.718 / 40.703 s on four consecutive channels). The model says ~3.9 s,
#: so it is roughly **ten times low** — it covers the frequency sweep and nothing
#: else, and the entire 40.7 s sits between the script send and the data return.
#: Post-acquisition work (routing, fitting, the payload write) all lands inside the
#: same log second and is not a factor.
#:
#: Quoted in ``--help`` and in the modelled-basis note so an operator has a real
#: number to plan from, rather than having to discover the gap the way this run
#: did — by finding round 4 starting at elapsed 2166 s against an intended 240 s
#: period. It is **not** used as a default: it belongs to one rig and one preset,
#: and a silently applied constant would be wrong the moment either changed.
MEASURED_PER_CHANNEL_S_STANDARD = 40.7

#: The default σ(t) sampling interval, **derived from the other defaults rather
#: than chosen**. ``--channels`` defaults to all 16 and ``--preset`` to
#: ``Standard``, which is 16 × :data:`MEASURED_PER_CHANNEL_S_STANDARD` = 651.2 s of
#: round; the previous 120 s default could not contain that on any preset (16
#: channels costs ~168 s even on ``Quick``, measured at ~10.5 s/channel), so an
#: operator accepting every default got a run that could not honour its own period
#: and a σ(t) the fitter reads as evenly spaced when it is not.
#:
#: Rounded up to a whole ten by the same rule :func:`minimum_feasible_period_s`
#: uses, so the number is one an operator can retype. It is deliberately a
#: *feasible* default rather than a *good* one: a 660 s interval resolves τ no
#: shorter than ~22 min, and the measured τ at the first setpoint is ~500 s — so
#: a run that cares about the transient must take fewer channels or a faster
#: preset, and this default makes that trade visible in ``plan`` instead of
#: burying it in an overrun.
DEFAULT_ROUND_PERIOD_S = math.ceil(16 * MEASURED_PER_CHANNEL_S_STANDARD / 10.0) * 10.0

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
    )
    config.validate()
    return config


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
PLAN_SCHEMA = "equilibration-plan/2"

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
    "min_hold_first_s", "min_hold_s",
    "preset", "model", "measured_per_channel_s",
    "electrode_l_cm", "electrode_t_cm", "electrode_w_cm", "thickness_method",
)

#: The design keys that legitimately have no value — the geometry, which may be
#: absent altogether, and the measured cost, which most rigs do not have. Every
#: other key missing from a plan file is a corrupt plan, not an empty one.
PLAN_OPTIONAL_KEYS = frozenset({"measured_per_channel_s", "electrode_l_cm",
                                "electrode_t_cm", "electrode_w_cm"})


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

    Both floors are printed, not their minimum. They differ — only one setpoint
    in the run gets ``min_hold_first_s`` — and the number an operator needs when
    reading a short setpoint later is the one that applied to *that* setpoint.

    The effective figure is what the run enforces, which is not the time floor
    divided by the period: :data:`MIN_POINTS_FOR_TAU` is folded in, and at the
    shipped 660 s period it is the term that binds at **every** setpoint. Printing
    only the time floors would understate the fast end of the budget by exactly
    the amount that makes the run analysable.
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
    print(f"                effective minimum {projection.min_rounds_first} rounds at "
          f"the first setpoint, {projection.min_rounds_later} after")
    print(f"                so a setpoint runs "
          f"{min(projection.min_rounds_first, projection.min_rounds_later)}-"
          f"{config.rounds_per_setpoint} rounds -- --rounds is a CEILING, not a count.")
    print(f"                No setpoint may stop under {MIN_POINTS_FOR_TAU} rounds: "
          f"that is the offline")
    print("                fitter's own MIN_POINTS_FOR_TAU (sigma(t) has three free")
    print("                parameters), so a shorter series would be acquired and then")
    print("                REFUSED for tau -- the one number this run exists to get.")
    print("                A channel with NULL sigma, or an R1 railed on the circuit")
    print("                model's bound, does not count towards the criterion: a")
    print("                railed fit is constant, and a constant is always 'settled'.")


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
    series_floor = projection.min_rounds_later * projection.series_round_s
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
    if projection.adaptive:
        print("    (a RANGE, not an estimate: the floor is every setpoint settling at")
        print(f"     its earliest -- {projection.min_rounds_first} rounds at the first, "
              f"{projection.min_rounds_later} after, of "
              f"{projection.rounds_per_setpoint} -- and the")
        print("     ceiling is none of them settling at all. Both ends are achievable.)")
    print("    (worst case uses the TIMEOUTS, not an assumed ramp rate, so it is an")
    print("     upper bound rather than a guess. Holds are computed from this config:")
    print("     estimate_workflow_duration projects every temperature wait as 0.0 s.)")


def _print_round_cost(config: EquilibrationConfig, cost: float, *,
                      measured: bool) -> None:
    """What one all-channel round costs, and **on whose authority**."""
    basis = "MEASURED, from --measured-per-channel-s" if measured else "modelled"
    print(f"  Round cost ({basis}, all channels):")
    print(f"    {config.eis_preset:<10s} {cost / 60:6.1f} min "
          f"({cost / max(1, len(config.channels)):.1f}s/channel)")
    if measured:
        return
    print("    NOTE: the model covers the FREQUENCY SWEEP ONLY. Mux switching,")
    print("    script upload, data retrieval and the file write are per-channel")
    print("    costs it does not carry, and they are roughly fixed rather than")
    print("    proportional to the sweep -- so a real round is LONGER, and a faster")
    print("    preset does not shrink the difference. This run measures its own")
    print("    round cost and records it in the sidecar.")


def _print_basis(config: EquilibrationConfig, cost: float, *,
                 measured: bool) -> None:
    """Never let a modelled duration be read as a prediction.

    The model is not merely approximate here, it is out by roughly an order of
    magnitude, and the whole point of this note is that the operator sees that
    before committing a night rather than after.
    """
    per_channel = cost / max(1, len(config.channels))
    if measured:
        print(f"    (basis: MEASURED {per_channel:.1f}s/channel, given on the command "
              f"line. The")
        print("     model is not used anywhere above.)")
        return
    print(f"    (basis: MODELLED {per_channel:.1f}s/channel -- a FLOOR, not a "
          f"prediction. Bench")
    print(f"     measurement puts a real round SEVERAL TIMES higher: "
          f"{MEASURED_PER_CHANNEL_S_STANDARD:g}s/channel was")
    print("     measured on 'Standard' over 12 channels, ~10x this. Re-plan with")
    print(f"     --measured-per-channel-s {MEASURED_PER_CHANNEL_S_STANDARD:g} for a "
          f"duration that reflects the bench.)")


def _print_period_caution(config: EquilibrationConfig, cost: float, *,
                          measured: bool) -> None:
    """Say, before the night is committed, whether the period can contain a round.

    ``plan`` is the one thing the operator reads before starting a 15 h run, and
    the failure mode here is quiet: the executor does not refuse an overrunning
    round, it simply takes longer, and σ(t) ends up sampled at a period nobody
    configured.

    Three statements, and they answer different questions. The remainder line is
    the same arithmetic on either basis, but it means different things: against a
    modelled cost it is *headroom for the overhead the model omits*, and against a
    measured one the overhead is already inside the number, so what is left is
    simply slack. The CAUTION is modelled-only for that reason — it warns about an
    omission that a measurement does not have. The feasibility block is
    unconditional: at this channel count and this per-channel cost, either the
    period contains a round or it does not.
    """
    headroom = round_headroom_s_per_channel(config, cost if measured else None)
    if measured:
        print(f"    Configured period {config.round_period_s:.0f}s vs MEASURED "
              f"{cost:.0f}s leaves {headroom:+.1f}s/channel of slack.")
    else:
        print(f"    Configured period {config.round_period_s:.0f}s vs modelled "
              f"{cost:.0f}s leaves {headroom:.1f}s/channel for that overhead.")
        _print_headroom_caution(config, cost, headroom)
    _print_feasibility(config, cost, measured=measured)


def _print_headroom_caution(config: EquilibrationConfig, modelled: float,
                            headroom: float) -> None:
    """The modelled-basis warning: too little room left for what the model omits."""
    if headroom >= OVERHEAD_HEADROOM_WARN_S:
        return
    needed = modelled + OVERHEAD_HEADROOM_WARN_S * len(config.channels)
    print(f"    ! CAUTION: under {OVERHEAD_HEADROOM_WARN_S:.0f}s/channel of headroom. "
          f"Rounds will very likely")
    print("      overrun --round-period-s, and the run will NOT shorten them or")
    print("      adjust the period -- the period is the sampling interval of")
    print(f"      sigma(t). Consider --round-period-s {needed:.0f} or fewer channels.")


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
                 else "MODELLED, so the real shortfall is several times worse")
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
        print(f"\nABORTED ({exc.kind}): {exc}", file=sys.stderr)
        _print_teardown(runner)
        return EXIT_FAILED
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        _print_teardown(runner)
        return EXIT_FAILED
    finally:
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

    This is the number that corrects ``estimate_eis_duration`` for everything
    downstream, so it is printed rather than left in the sidecar for someone to
    find.
    """
    summary = runner.measured_cost_summary()
    rows = [(kind, summary[kind]) for kind in ("series",) if kind in summary]
    if not rows:
        return
    print("  Measured round cost (wall clock, includes per-channel overhead the")
    print("  model does not carry):")
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
    """
    parser.add_argument("--channels", default=None if required_channels else "1-16",
                        help='e.g. "1-16" or "2,4,5-10"'
                             + (" -- REQUIRED here; never inherited from 'plan'"
                                if required_channels else ""))
    parser.add_argument("--temperatures", default="27.5,45,65,85",
                        help="setpoints for the UP leg; the down leg retraces them")
    parser.add_argument("--legs", default="up,down")
    parser.add_argument("--rh", type=float, default=15.0, help="%%RH setpoint")
    parser.add_argument("--rounds", type=int, default=15,
                        help=f"the CEILING on sigma(t) rounds per setpoint -- NOT a "
                             f"fixed count. A setpoint stops as soon as sigma has "
                             f"settled (see --settle-tol-rel), the hold floor has "
                             f"elapsed AND at least {MIN_POINTS_FOR_TAU} rounds have "
                             f"run; it runs this many only when it has not. That last "
                             f"floor is the offline fitter's own MIN_POINTS_FOR_TAU, "
                             f"imported rather than retyped: sigma(t) has three free "
                             f"parameters, so a shorter series is REFUSED for tau, and "
                             f"the run must not be able to acquire a setpoint it "
                             f"cannot analyse. A ceiling below "
                             f"{MIN_POINTS_FOR_TAU} is refused outright -- no setpoint "
                             f"in such a run could ever yield a tau.")
    _add_settle_args(parser)
    parser.add_argument("--round-period-s", dest="round_period_s", type=float,
                        default=DEFAULT_ROUND_PERIOD_S,
                        help=f"sigma(t) sampling interval. Default "
                             f"{DEFAULT_ROUND_PERIOD_S:g}s is DERIVED, not chosen: "
                             f"16 channels (the --channels default) x "
                             f"{MEASURED_PER_CHANNEL_S_STANDARD:g}s/channel measured "
                             f"on 'Standard' (the --preset default) = "
                             f"{16 * MEASURED_PER_CHANNEL_S_STANDARD:.0f}s, rounded up "
                             f"to a typable ten. Shorten it by taking fewer channels "
                             f"or a faster preset -- the cost is PER CHANNEL, and a "
                             f"period a round does not fit inside is not honoured, it "
                             f"is simply exceeded.")
    parser.add_argument("--preset", default="Standard", help="EIS preset for the series")
    # No default. The measured number belongs to one rig and one preset, and a
    # constant applied silently would be wrong the moment either changed -- while
    # still reading, to the next operator, like a prediction.
    parser.add_argument("--measured-per-channel-s", dest="measured_per_channel_s",
                        type=float, default=None,
                        help="plan from a MEASURED per-channel round cost instead of "
                             "the model, which covers the frequency sweep only. This "
                             f"rig measured {MEASURED_PER_CHANNEL_S_STANDARD:g} "
                             "s/channel on 'Standard' over 12 channels, against ~3.9 "
                             "s/channel modelled -- about 10x. Overrides the modelled "
                             "cost in the round cost, the inter-round gap, the "
                             "headroom, the sampling interval and the total duration.")
    # `--model` meant the EIS CIRCUIT model here and the RELAXATION model on
    # `fit`/`report` -- one spelling, two vocabularies, both plausible-looking
    # strings, so `fit --model simpleSalt` failed confusingly rather than
    # obviously. Each name is now unambiguous and each help text names the other,
    # with `--model` kept as a working alias on both so no saved script breaks.
    # `dest` stays `model`, so plan files written before the rename still load.
    parser.add_argument("--circuit-model", "--model", dest="model",
                        default="simpleSalt",
                        help="EIS CIRCUIT model fitted to each spectrum (e.g. "
                             "simpleSalt). Not the relaxation model: that is "
                             "--relaxation-model on 'fit'/'report'. '--model' is a "
                             "deprecated alias for this flag here.")
    parser.add_argument("--electrode-l-cm", dest="electrode_l_cm", type=float)
    parser.add_argument("--electrode-t-cm", dest="electrode_t_cm", type=float)
    parser.add_argument("--electrode-w-cm", dest="electrode_w_cm", type=float)
    parser.add_argument("--thickness-method", dest="thickness_method",
                        default="target", choices=list(THICKNESS_METHODS),
                        help="how --electrode-t-cm was obtained. Default 'target': "
                             "a hand-computed digital-twin number, NOT a measurement. "
                             "Recorded in the run sidecar because "
                             "fit_results.thickness_method stays NULL for this run.")
    parser.add_argument("--project", help="project directory (default: [data] project_dir)")
    parser.add_argument("--mock", action="store_true")


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
    parser.add_argument("--settle", dest="settle", choices=("on", "off"),
                        default="on",
                        help="stop a setpoint once sigma has settled instead of "
                             "always running --rounds. 'off' restores the old "
                             "fixed-count behaviour exactly.")
    parser.add_argument("--settle-tol-rel", dest="settle_tol_rel", type=float,
                        default=DEFAULT_SETTLE_TOL_REL,
                        help=f"relative half-width of the settle band (default "
                             f"{DEFAULT_SETTLE_TOL_REL:g}). MUST exceed the run's own "
                             f"noise floor -- 5.98%% median was measured over 96 "
                             f"series, with 22 of them above 20%% -- or no hold "
                             f"length can satisfy it and every setpoint runs to its "
                             f"ceiling. The run says so per setpoint when it happens.")
    parser.add_argument("--settle-n-rounds", dest="settle_n_rounds", type=int,
                        default=DEFAULT_SETTLE_N_ROUNDS,
                        help=f"consecutive rounds that must all sit inside the band "
                             f"(default {DEFAULT_SETTLE_N_ROUNDS}). A DETECTION "
                             f"WINDOW, and a different question from how many points "
                             f"a tau needs: setting it below {MIN_POINTS_FOR_TAU} "
                             f"narrows the window exactly as asked and is NOT "
                             f"rewritten, but the setpoint still cannot stop before "
                             f"{MIN_POINTS_FOR_TAU} rounds, because the fitter refuses "
                             f"a shorter series.")
    parser.add_argument("--settle-min-channels", dest="settle_min_channels", type=int,
                        default=DEFAULT_SETTLE_MIN_CHANNELS,
                        help=f"fewest channels that must carry usable evidence for "
                             f"the criterion to be evaluated at all (default "
                             f"{DEFAULT_SETTLE_MIN_CHANNELS}). A channel whose sigma "
                             f"is NULL, or whose R1 railed on the circuit model's "
                             f"lower bound, does NOT count -- a railed fit returns "
                             f"the same number every round and a constant is "
                             f"trivially 'settled'. Below this the setpoint runs to "
                             f"its ceiling and records 'not_evaluable'.")
    parser.add_argument("--min-hold-first-s", dest="min_hold_first_s", type=float,
                        default=DEFAULT_MIN_HOLD_FIRST_S,
                        help=f"floor on the hold at the FIRST setpoint of the run "
                             f"(default {DEFAULT_MIN_HOLD_FIRST_S:g}s ~ 3 tau, with "
                             f"tau = 425-575s measured while the films dry from "
                             f"ambient to 15 %%RH). The whole transient is here. A "
                             f"TIME floor: the rounds it buys is "
                             f"ceil(it / --round-period-s), and the effective minimum "
                             f"is that against {MIN_POINTS_FOR_TAU} rounds and "
                             f"--settle-n-rounds, whichever is largest.")
    parser.add_argument("--min-hold-s", dest="min_hold_s", type=float,
                        default=DEFAULT_MIN_HOLD_S,
                        help=f"floor on every later setpoint (default "
                             f"{DEFAULT_MIN_HOLD_S:g}s). The films are already dry, "
                             f"but the chamber still has to re-establish RH. At the "
                             f"default {DEFAULT_ROUND_PERIOD_S:g}s period this buys "
                             f"only one round, so {MIN_POINTS_FOR_TAU} -- the fitter's "
                             f"minimum -- is what actually binds; 'plan' prints the "
                             f"effective figure for both floors.")


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
