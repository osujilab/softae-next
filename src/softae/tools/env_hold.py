"""``softae-env`` — hold the chamber at a humidity, and measure nothing.

Every shipped path to the humidifier bundles it into a measurement protocol:
the only one is ``EquilibrationRun._start_rh``, inside a nine-to-fifteen-hour EIS
characterization run. An operator who simply wants a board conditioned at 45 %RH
for four hours had no surface at all, and an ad-hoc script would take no rig
claim, attach no watchdog and restore nothing on the way out.

This is that surface, and it is deliberately **one axis**. There is no ``--temp``
and no heater is driven: ``softae-equilibration run`` already owns the temperature
hold — approach timeout, tolerance band, watched hold, ambient restore — and a
second copy of that here would be a second path to the same hardware. Chamber
temperature is *reported* beside the humidity (``get_TH`` returns both from one
transaction) because a humidity number without the air temperature is not
actionable. Reporting is not driving.

Two subcommands, read-only first, and ``hold`` is a dry run unless ``--execute``::

    python -m softae.tools.env_hold plan --rh 45 --duration-h 4
    python -m softae.tools.env_hold hold --rh 45 --duration-h 4 --execute

Spelled as ``python -m`` in everything printed, for the reason
``tools/equilibration.py`` states: whether the :data:`CONSOLE_SCRIPT` entry point
resolves is a fact about when the venv was last installed from, not about the tool.

**How every exit ends.** Clean, signalled or faulted alike, the chamber is left
**purging dry** rather than zeroed: PID stopped, setpoint 0, duty held at
``out_min``, which on this rig is *dry air*. This is the park path's ruling
(``core.safe_park``, operator 2026-08-24) reaching the one tool that never routed
through it — and this is the tool it matters most in, because a four-hour hold at
10 %RH is exactly the state that ``ctrl == 0`` throws away: zero is the firmware's
auto-shutoff, both Aalborg PSVs close at once, and the room wins in tens of
seconds.

**What this does not close.** A ``SIGKILL``, a power cut or a blue screen runs no
Python, so nothing here writes anything and the Trinket is left at its last
commanded duty. The firmware's own deadman does close that — ``ctrl_timeout``
forces ``ctrl = 0`` roughly :data:`~softae.core.safe_park.RH_DEADMAN_S` s after
the last command it received — but it is the *firmware's* guarantee, arriving on
the firmware's schedule, and nothing on this side is timed by it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Callable

import structlog

# The one number, imported rather than restated. `RH_DEADMAN_S`'s own comment
# says why a second copy is the hazard: it is a *restatement of the firmware's*
# `ctrl_timeout`, so a literal here would agree today and quietly lie the moment
# the Trinket is retuned. At module scope, unlike every other `softae` import in
# this file, because `_report_restore` runs from a signal handler and from
# interpreter teardown — neither is a place to take the import lock. Nothing
# else of `core.safe_park` is used: this tool deliberately does not route
# through the park (see `install_handlers`).
from softae.core.safe_park import RH_DEADMAN_S
from softae.tools import run_finalizer, use_utf8_console

logger = structlog.get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DECLINED = 2
#: Nobody decided anything — the rig was simply occupied. ``tools/campaign.py``'s
#: value and its reasoning: a wrapper that retried an operator's "no" would be
#: overriding them, and one that gives up on a collision loses the night.
EXIT_BUSY = 4

#: The typed word that starts unattended actuation. Not "y".
CONFIRM_WORD = "yes"

CONSOLE_SCRIPT = "softae-env"
MODULE = "softae.tools.env_hold"
CLI = f"python -m {MODULE}"

#: What an unreadable, stale or NaN value looks like. Never ``0.0`` and never the
#: last good number — ``AsyncRHController`` deliberately NaNs a held reading past
#: ``max_stale_s``, and flattening that into a plausible figure would let a dead
#: sensor read as a working one for four hours.
NA = "--"

#: Seconds between heartbeat lines. Matches ``DEFAULT_MILESTONE_INTERVAL_S``.
DEFAULT_HEARTBEAT_S = 300.0

#: Ceiling on the loop's own period. ``RHHoldWatch.sample`` is internally
#: throttled to ``rh_poll_interval_s``, so calling it more often than that is
#: free — but calling it *less* often silently lengthens the effective cadence,
#: which is why the loop never naps longer than this.
SAMPLE_CEILING_S = 5.0

#: ``<kind>:<name>:<run_id>`` — the grammar ``core.rig_session`` documents, whose
#: shipped sibling is ``campaign:<name>:<run_id>``. The third field is filled
#: rather than left trailing: a bare ``tool:env-hold:`` in a lock file asserts
#: "there is a run id and it is blank".
CLAIM_KIND = "tool:env-hold"


# ── Small renderings ─────────────────────────────────────────────────────────

def hms(seconds: float) -> str:
    """``H:MM:SS`` — ASCII, fixed width, and readable at 3 a.m."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    return f"{total // 3600:d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _val(value: Any, fmt: str = "{:.1f}") -> str:
    """A number, or :data:`NA`. There is no third rendering."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return NA
    return NA if f != f else fmt.format(f)


def describe_duration(duration_s: float | None) -> str:
    """``4h 00m`` / ``12m 30s`` / ``until interrupted``.

    The scale changes because the number is read as a commitment: rendering a
    30 s dry-run check as ``0h 00m`` reads as "no duration was understood".
    """
    if duration_s is None:
        return "until interrupted"
    total = max(0, int(duration_s))
    if total >= 3600:
        return f"{total // 3600}h {total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total}s"


def duration_s(args) -> float | None:
    """The bounded duration, or ``None`` for a hold that ends when signalled.

    Until-signal is the honest default for an operator conditioning a board who
    will decide when to stop; a bounded hold is what a scripted invocation wants.
    """
    if getattr(args, "duration_h", None) is not None:
        return float(args.duration_h) * 3600.0
    if getattr(args, "duration_s", None) is not None:
        return float(args.duration_s)
    return None


def _open_store(args):
    """The project store, defaulting to the one the GUI and campaigns already use.

    ``--mock`` writes to an isolated ``<project>/mock`` store. Copied from
    ``tools/equilibration._open_store`` rather than imported: that module is on
    this claim's NOT-touched list, and the rule is four lines.
    """
    from softae.config import loader
    from softae.core.data_store import DataStore

    project = getattr(args, "project", None)
    if not project:
        project = loader.data_project_dir()
        if getattr(args, "mock", False):
            project = str(Path(project).expanduser() / "mock")
    return DataStore(project, db_filename=loader.data_db_filename()), project


def plan_text(args) -> str:
    """What a hold *would* do — the thing an operator pastes into a log."""
    from softae.drivers.contracts import rh_watchdog_config

    cfg = rh_watchdog_config()
    store_kind = ("MOCK (isolated <project>/mock store)" if getattr(args, "mock", False)
                  else "the project store")
    return "\n".join([
        "",
        "  " + "=" * 68,
        "  ENVIRONMENT HOLD -- humidity only. No heater is driven.",
        f"    setpoint      {float(args.rh):g} %RH",
        f"    duration      {describe_duration(duration_s(args))}",
        f"    watchdog      warn at {cfg['warn_pct']:g} %RH, fault at "
        f"{cfg['fault_pct']:g} %RH,",
        f"                  sustained over a {cfg['grace_s']:g} s grace window, "
        f"sampled every {cfg['poll_interval_s']:g} s",
        f"    heartbeat     one line every "
        f"{float(getattr(args, 'heartbeat_s', DEFAULT_HEARTBEAT_S)):g} s",
        f"    recording to  {store_kind}",
        "    on exit       the humidifier is left PURGING DRY, not zeroed",
        f"                  (duty = out_min; the Trinket's deadman shuts both "
        f"valves ~{RH_DEADMAN_S:g} s later)",
        "  " + "=" * 68,
        "  No verdict stops the hold. A humidity that is off command is announced",
        "  as an alert row and printed; it is not a reason for software to stop",
        "  actuating a chamber a human asked to be held.",
    ])


def confirm_hold(rh_pct: float, seconds: float | None, *,
                 assume_yes: bool = False,
                 reader: Callable[[str], str] | None = None) -> bool:
    """The barrier. Requires the literal word ``yes``, never ``y``.

    A non-TTY ``input()`` raises ``EOFError``, which is read as a **decline**:
    an unattended invocation that meant to run had ``--yes`` to say so.
    """
    print()
    print("  " + "!" * 68)
    print(f"  THIS ACTUATES THE HUMIDIFIER AT {float(rh_pct):g} %RH")
    print(f"  for {describe_duration(seconds)}, UNATTENDED.")
    print("  " + "!" * 68)
    if assume_yes:
        print("  --yes given; proceeding without confirmation.")
        return True
    try:
        reply = (reader or input)(f"  Type '{CONFIRM_WORD}' to start: ").strip().lower()
    except EOFError:
        reply = ""
    if reply != CONFIRM_WORD:
        print("Declined — nothing was humidified.")
        return False
    return True


# ── The hold itself ──────────────────────────────────────────────────────────

class HoldSession:
    """Everything the teardown has to undo, and the one place that undoes it.

    The exit is **one-shot and never raises**: a second signal arriving during
    teardown must be declined rather than start a second write sequence down the
    same serial line, which is ``ParkGuard``'s discipline for the same reason.
    """

    def __init__(self, rh: Any, manager: Any, store: Any, finalize: Callable[[str], None],
                 *, run_async: Callable[[Any], Any], setpoint_pct: float) -> None:
        self.rh = rh
        self.manager = manager
        self.store = store
        self.finalize = finalize
        self.setpoint_pct = float(setpoint_pct)
        self._run_async = run_async
        self._exit_started = False
        self.in_progress = False
        #: Why the chamber is **not** purging dry — ``""`` when it is. The
        #: teardown's whole verdict lives in these three, and they are three
        #: rather than one because there are three end states an operator has to
        #: tell apart: purging, zeroed instead, and unknown.
        self.purge_error = ""
        #: The duty actually put on the wire, so the report can name it.
        self.purge_duty = 0.0
        #: The fallback landed: no dry purge, but the humidifier really is off.
        #: Safe hardware, lost chamber — a warning, never the alarm.
        self.zeroed_instead = False

    def safe_exit(self, status: str) -> None:
        """Leave the chamber purging dry, say so, close the row, disconnect."""
        if self._exit_started:
            return
        self._exit_started = True
        self.in_progress = True
        try:
            self._dry_purge_humidifier()
            self._report_restore()
            self.finalize(status)
            self._disconnect()
            self._close_store()
        finally:
            self.in_progress = False

    # Each step swallows its own failure: a teardown that stops at the first
    # problem leaves the *later* steps undone, and the later steps here are the
    # run row and the rig claim.

    def _dry_purge_humidifier(self) -> None:
        """Leave the chamber purging dry — ``safe_park._park_humidifier``'s shape.

        ``safe_off`` survives as one thing only: the fallback for a driver that
        exposes no ``safe_dry`` at all. It is *recorded* and then taken anyway,
        which is that function's discipline and for its reason — zeroing is the
        strictly safer end state, so refusing it would leave a humidifier
        energised in order to make a point about a missing method.

        The driver's own degenerate-``out_min`` fallback is **not** re-implemented
        here. It reports itself through ``last_safe_dry_error``, and that message
        says in its own words that the humidifier was zeroed instead — which is
        what stops :meth:`_report_restore` misdescribing it.
        """
        dry = getattr(self.rh, "safe_dry", None)
        if not callable(dry):
            self.purge_error = "driver exposes no safe_dry()"
            self._zero_as_fallback()
            return
        try:
            dry()
        except Exception as exc:                       # noqa: BLE001 - never raise
            self.purge_error = str(exc)
            return
        # `safe_dry` never raises on a comms failure — the park's never-raise
        # contract read back into the driver — so without this the teardown would
        # announce a purge that never reached the Trinket. Non-`str` means no
        # report, so a driver predating the attribute is not accused of a failure
        # it never had.
        err = getattr(self.rh, "last_safe_dry_error", "")
        if isinstance(err, str) and err:
            self.purge_error = err
            return
        duty = getattr(self.rh, "last_safe_dry_duty", 0.0)
        self.purge_duty = float(duty) if isinstance(duty, (int, float)) else 0.0

    def _zero_as_fallback(self) -> None:
        """The pre-``safe_dry`` driver's end state. Appends to :attr:`purge_error`."""
        off = getattr(self.rh, "safe_off", None)
        if not callable(off):
            self.purge_error += (" and no safe_off() -- nothing in this process "
                                 "stopped the humidifier")
            return
        try:
            off()
        except Exception as exc:                       # noqa: BLE001 - never raise
            self.purge_error += f"; the fallback to safe_off() also failed: {exc}"
            return
        err = getattr(self.rh, "last_safe_off_error", "")
        if isinstance(err, str) and err:
            self.purge_error += f"; the fallback to safe_off() also failed: {err}"
            return
        self.zeroed_instead = True

    def _report_restore(self) -> None:
        """To **stderr**, including the good news.

        ``--quiet > hold.log`` must still show on the terminal how the chamber
        was left, which is the difference between an operator who walks away and
        one who does not. What that sentence *means* changed with the end state,
        and the three branches below are the whole of the change:

        * **purging** — success, and it leaves gas deliberately flowing. So the
          line has to say the flow is the point, or an operator reads a
          successful teardown as a humidifier nobody switched off. Same problem
          ``safe_park.DRY_PURGE_COMMANDED`` solved, same three moves: name the
          duty, call the standing command deliberate, say what shuts the valves.
        * **zeroed instead** — the hardware is off and safe, and the chamber is
          gone. Under the old teardown this *was* the success case, which is
          exactly why it now needs saying out loud rather than passing silently.
        * **unknown** — nothing confirmed either. The only alarm, and it no
          longer *asserts* the humidifier is still driving: the driver's
          degenerate-``out_min`` fallback lands here having genuinely zeroed the
          device, and its own message says so directly above.
        """
        def say(*lines: str) -> None:
            for line in lines:
                print(line, file=sys.stderr)

        def reason(indent: str) -> str:
            # Wrapped, because the message that most needs reading is the
            # longest: the driver's degenerate-`out_min` report runs past 200
            # characters, and the sentence that stops the headline above it
            # being a misdescription is at the *end* of it.
            return textwrap.fill(self.purge_error, width=78,
                                 initial_indent=indent, subsequent_indent=indent)

        if self.zeroed_instead:
            say("  !! NO DRY PURGE -- THE HUMIDIFIER WAS ZEROED INSTEAD:",
                reason("     "),
                "     The hardware is safe: PID stopped, duty 0, both valves shut.",
                "     But nothing is flowing, so the chamber collapses to room RH within",
                "     tens of seconds. This hold's dried state is lost, not merely ending.")
            return
        if self.purge_error:
            say("  !!!! NO DRY PURGE WAS CONFIRMED, AND THE HUMIDIFIER'S STATE IS UNKNOWN:",
                reason("       "),
                f"       It may still be driving {self.setpoint_pct:g} %RH, and the chamber"
                f" is not being purged.",
                "       Nothing else in this process will act on it. CHECK IT AT THE RIG.")
            return
        say(f"  Humidifier DRY-PURGED: PID stopped, setpoint 0, duty held at "
            f"{self.purge_duty:g} = dry air.",
            "     Gas is STILL FLOWING and that is DELIBERATE -- not a humidifier left on.",
            f"     The Trinket's deadman shuts both valves ~{RH_DEADMAN_S:g} s from now; the"
            f" chamber",
            "     keeps its dry state over the changeover instead of collapsing to room RH.")

    def _disconnect(self) -> None:
        # After the dry purge, never before: a disconnected driver has no serial
        # handle and nothing can be commanded through it at all.
        #
        # And the purge survives this call rather than being undone by it:
        # `safe_dry` stops the PID loop itself, so `disconnect`'s own
        # `_stop_pid_loop` finds `_running` already False and returns having
        # written nothing. The `out_min` duty is what stays on the wire.
        try:
            self._run_async(self.manager.disconnect_all())
        except Exception:                              # noqa: BLE001 - never raise
            logger.warning("env_hold_disconnect_failed", exc_info=True)

    def _close_store(self) -> None:
        try:
            self.store.close()
        except Exception:                              # noqa: BLE001 - never raise
            logger.warning("env_hold_store_close_failed", exc_info=True)


def install_handlers(session: HoldSession,
                     signals: tuple[int, ...] | None = None) -> Callable[[], None]:
    """Park **this tool's one actuator** on SIGINT/SIGTERM/SIGBREAK. Returns the restore.

    Deliberately not ``core.shutdown.install_signal_park``: that is built around
    ``ParkGuard`` parking a whole manager, which is right for a campaign and
    wrong for a tool whose only actuator is one humidifier. The three
    load-bearing properties of its handler are copied verbatim — **park,
    uninstall itself, raise KeyboardInterrupt**:

    * *park* — for SIGTERM there is no other Python that will ever run;
    * *uninstall* — a second Ctrl-C must reach the default handler, or an
      operator watching a wedged teardown has only the task manager;
    * *raise* — so the exit unwinds through the caller's own ``finally``.

    On Windows SIGTERM is not deliverable to a Python handler and
    ``os.kill(pid, SIGTERM)`` **is** ``TerminateProcess``; SIGINT and SIGBREAK
    are the reachable signals there.
    """
    from softae.core.shutdown import shutdown_signals

    targets = tuple(signals) if signals is not None else shutdown_signals()
    previous: dict[int, Any] = {}
    installed: list[int] = []
    restored = {"done": False}

    def _restore() -> None:
        if restored["done"]:
            return
        restored["done"] = True
        for sig in installed:
            try:
                signal.signal(sig, previous[sig])
            except Exception:                          # noqa: BLE001 - exit must not raise
                logger.debug("env_hold_signal_restore_failed", signal=int(sig))

    def _handler(signum: int, _frame) -> None:
        if session.in_progress:
            logger.warning("env_hold_signal_during_teardown", signal=int(signum))
            return
        logger.warning("env_hold_signal_stopping", signal=int(signum))
        session.safe_exit("interrupted")
        _restore()
        raise KeyboardInterrupt(f"signal {int(signum)} received")

    if threading.current_thread() is threading.main_thread():
        for sig in targets:
            try:
                previous[sig] = signal.signal(sig, _handler)
                installed.append(int(sig))
            except (ValueError, OSError, RuntimeError, AttributeError) as exc:
                logger.info("env_hold_signal_not_installed", signal=int(sig),
                            error=str(exc))
    else:
        logger.info("env_hold_signal_not_installed", why="not the main thread")
    return _restore


def heartbeat_line(elapsed_s: float, watch: Any, setpoint_pct: float,
                   remaining_s: float | None) -> str:
    """``[0:15:00] 2026-08-19 14:17:03  RH sp 45.0 pv 44.2%  air 22.8C  converging  (…)``

    One timestamped line per interval, on its own line and in scrollback: a
    monitor that overwrites itself has no history, and "is the loop working?" is
    a question about a trend.
    """
    verdict = watch.verdict
    left = ("until interrupted" if remaining_s is None
            else f"{describe_duration(remaining_s)} left")
    return (f"[{hms(elapsed_s)}] {time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"RH sp {float(setpoint_pct):.1f} pv {_val(verdict.pv_pct)}%  "
            f"air {_val(watch.temperature_C)}C  {verdict.state}  ({left})")


def hold_loop(watch: Any, setpoint_pct: float, *, seconds: float | None,
              heartbeat_s: float, quiet: bool,
              poll_interval_s: float = SAMPLE_CEILING_S,
              now: Callable[[], float] | None = None,
              sleep: Callable[[float], None] | None = None,
              emit: Callable[[str], None] | None = None) -> None:
    """Sample, announce, and wait. **No verdict stops the hold.**

    That is the policy ``run_anneal_hold`` documents and ``[safety]`` argues at
    length: a hold whose humidity is off command is a fact the operator needs, not
    a reason for software to stop actuating a chamber a human asked to be held.

    ``RHHoldWatch`` owns no clock — ``wrap_reader`` exists to ride a temperature
    poll this tool does not have — so sampling is driven from here, and the loop
    never naps longer than :data:`SAMPLE_CEILING_S` **or** the watchdog's own
    ``rh_poll_interval_s``, whichever is shorter. The ceiling alone is not enough:
    a rig configured to poll faster than 5 s would have that setting silently
    ignored, and the watchdog's grace window is counted in samples it actually got.

    A non-positive or unreadable interval falls back to the ceiling rather than
    spinning: ``sleep(0)`` in this loop is a hot loop on a serial line.
    """
    now = now or time.monotonic
    sleep = sleep or time.sleep
    emit = emit or print
    try:
        period = min(SAMPLE_CEILING_S, float(poll_interval_s))
    except (TypeError, ValueError):
        period = SAMPLE_CEILING_S
    if not period > 0:
        period = SAMPLE_CEILING_S
    started = now()
    deadline = None if seconds is None else started + float(seconds)
    next_line = started
    while True:
        watch.sample()
        t = now()
        if not quiet and t >= next_line:
            remaining = None if deadline is None else max(0.0, deadline - t)
            emit(heartbeat_line(t - started, watch, setpoint_pct, remaining))
            next_line = t + float(heartbeat_s)
        if deadline is not None and t >= deadline:
            return
        nap = period
        if deadline is not None:
            nap = min(nap, max(0.0, deadline - t))
        sleep(nap)


# ── Subcommands ──────────────────────────────────────────────────────────────

def _cmd_plan(args) -> int:
    print(plan_text(args))
    print()
    print(f"  {CLI} hold --rh {float(args.rh):g} --execute")
    return EXIT_OK


def _cmd_hold(args) -> int:
    seconds = duration_s(args)
    print(plan_text(args))

    if not args.execute:
        print()
        print("Dry run — no instrument was opened and nothing was humidified.")
        print("Re-run with --execute to actuate the humidifier.")
        return EXIT_OK

    if not confirm_hold(args.rh, seconds, assume_yes=args.yes):
        return EXIT_DECLINED

    from softae.core.rig_session import held_rig_session
    from softae.core.run_lock import RunLockHeld, busy_rig_message, foreign_run_lock
    from softae.drivers.contracts import RHHoldWatch, rh_watchdog_config
    from softae.drivers.factory import create_manager
    from softae.errors import CommunicationError, InstrumentError, SafetyError

    try:
        # `mock=False` forces real drivers and raises rather than silently
        # simulating: a hold that quietly actuated nothing would look identical
        # to one that worked.
        manager = create_manager(mock=True if args.mock else False)
    except Exception as exc:                           # noqa: BLE001 - operator message
        print(f"Could not open the instruments: {exc}", file=sys.stderr)
        print("  This hold refuses to fall back to simulated drivers. Use --mock to",
              file=sys.stderr)
        print("  exercise it without hardware.", file=sys.stderr)
        return EXIT_FAILED

    # Ask who holds the rig **before** the store exists, which is
    # `tools/campaign.py`'s ordering and for the same reason: a refusal over
    # hardware this hold never touched must not leave an `aborted` run row
    # behind, because that row is indistinguishable from a hold that started and
    # failed. Gated on `args.mock`, the same flag the claim below is gated on and
    # for the same reason: a mock hold that takes no lock must not be refused
    # over one either.
    #
    # The residual race is deliberately accepted: a holder that arrives between
    # this peek and the claim below still raises `RunLockHeld` after the row
    # exists, and that path keeps its `aborted` finalization. The peek does not
    # close the window — `acquire_run_lock`'s exclusive create is what makes the
    # claim itself safe — it only makes the stray `aborted` row rare instead of
    # the routine outcome of running this tool while the rig is busy.
    if not args.mock:
        holder = foreign_run_lock()
        if holder is not None:
            print(f"\n!! NOT STARTING THIS HOLD\n\n"
                  f"{busy_rig_message(holder, action='This hold')}", flush=True)
            return EXIT_BUSY

    store, _project = _open_store(args)
    print(f"  recording to: {store.project_dir}")
    run_id = store.start_run("env_hold", mode="hold",
                             annotation=f"{float(args.rh):g}%RH")
    finalize = run_finalizer(store, run_id)

    try:
        # `--mock` claims nothing, and the gate is **this tool's own flag** rather
        # than `held_rig_session`'s `session_is_simulated` exemption, which would
        # otherwise reach the same answer here today.
        #
        # The original reason is **retracted**: SESSION_MAIL [e6] §1 measured that
        # exemption recognising a mock by the `Mock` prefix on its class *name*,
        # so a legitimately-named mock **subclass** read as real. [p39] §3
        # repaired the predicate to an `isinstance` test against the shipped mock
        # bases, which survives subclassing. Two reasons outlive the repair:
        #
        #   * the `foreign_run_lock` peek above consults **no predicate at all**,
        #     so only this flag stops a hold that claims nothing from being
        #     *refused* over a lock it never wanted; and
        #   * `_mock_driver_classes` is a hand-maintained registry whose own
        #     docstring names its failure direction — a mock added to
        #     `softae.drivers` and forgotten there reads as real. The exemption
        #     can still return a wrong verdict; the mistake is now about
        #     *membership* rather than spelling.
        #
        # The flag is the invariant's own evidence, so it is what the invariant
        # is gated on. Same shape as `tools/eis_validate.py`'s `_rig_claim`.
        claim = (contextlib.nullcontext() if args.mock
                 else held_rig_session(manager, what=f"{CLAIM_KIND}:{run_id}"))
        with claim:
            return _run_hold(args, manager, store, run_id, finalize, seconds,
                             RHHoldWatch=RHHoldWatch,
                             thresholds=rh_watchdog_config(),
                             driver_errors=(InstrumentError, CommunicationError,
                                            SafetyError))
    except RunLockHeld as held:
        finalize("aborted")
        store.close()
        print(f"\n!! NOT STARTING THIS HOLD\n\n"
              f"{busy_rig_message(held.lock, action='This hold')}", flush=True)
        return EXIT_BUSY


def _run_hold(args, manager, store, run_id, finalize, seconds, *,
              RHHoldWatch, thresholds, driver_errors) -> int:
    """Connect, set, start, watch — and make the exit unconditional."""
    loop = asyncio.new_event_loop()

    def _run_async(coro):
        return loop.run_until_complete(coro)

    session: HoldSession | None = None
    restore = None
    try:
        _run_async(manager.connect_all())
        rh = manager.get("rh_controller")
        session = HoldSession(rh, manager, store, finalize, run_async=_run_async,
                              setpoint_pct=float(args.rh))
        restore = install_handlers(session)

        # The same two calls in the same order as `EquilibrationRun._start_rh`,
        # including its already-running check so a live loop is not restarted.
        #
        # Unlike that method, the `status()` call is deliberately **not** wrapped
        # in try/except. An equilibration run meets this mid-sequence, with a
        # board already cast and hours already spent, so degrading around a flaky
        # `status()` and carrying on is the right trade there. Here it is the
        # first thing that happens: a hold that cannot verify its control loop
        # started has nothing to preserve by continuing, and continuing means
        # walking away for four hours from a chamber that may not be being
        # driven at all. Letting this raise into the generic error path — which
        # finalizes the row and runs the dry purge — is the intended outcome.
        rh.set_setpoint(float(args.rh))
        if not rh.status().get("running", False):
            rh.start()

        watch = RHHoldWatch(rh.get_TH, float(args.rh), thresholds=thresholds,
                            data_store=store, run_id=run_id)
        hold_loop(watch, float(args.rh), seconds=seconds,
                  heartbeat_s=float(getattr(args, "heartbeat_s",
                                            DEFAULT_HEARTBEAT_S)),
                  quiet=bool(getattr(args, "quiet", False)),
                  # One source for the cadence: the same thresholds the watch
                  # throttles itself by, so the loop cannot sample slower than
                  # the watchdog was configured to be told about.
                  poll_interval_s=float(thresholds.get("poll_interval_s",
                                                       SAMPLE_CEILING_S)))
    except KeyboardInterrupt:
        if session is not None:
            session.safe_exit("interrupted")
        print("\nInterrupted.", file=sys.stderr)
        # An until-signal hold that stopped because the operator signalled it got
        # exactly what it asked for. The *status* still records that nothing but a
        # person decided the end; the exit code says the tool did its job.
        return EXIT_OK if seconds is None else EXIT_FAILED
    except driver_errors as exc:
        if session is not None:
            session.safe_exit("aborted")
        print(f"\nABORTED: {exc}", file=sys.stderr)
        return EXIT_FAILED
    else:
        session.safe_exit("done")
        print(f"\nHeld {float(args.rh):g} %RH for {describe_duration(seconds)}. "
              f"Recorded run {run_id}.")
        return EXIT_OK
    finally:
        if restore is not None:
            restore()
        # The catch-all for paths no `except` above names. `run_finalizer` is
        # idempotent and `safe_exit` is one-shot, so this is a no-op unless one
        # of them is genuinely on its way out.
        if session is not None:
            session.safe_exit("error")
        else:
            finalize("error")
            store.close()
        loop.close()


# ── Parser ───────────────────────────────────────────────────────────────────

def _add_hold_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rh", type=float, required=True, metavar="PCT",
                   help="the humidity setpoint (%%RH). Validated at set_setpoint "
                        "time against the driver's own cap; this tool does not "
                        "re-implement it.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--duration-s", dest="duration_s", type=float, metavar="S",
                       help="hold for this many seconds")
    group.add_argument("--duration-h", dest="duration_h", type=float, metavar="H",
                       help="hold for this many hours")
    p.add_argument("--project", metavar="PATH",
                   help="project directory (default: [data] project_dir)")
    p.add_argument("--mock", action="store_true",
                   help="simulated drivers, recording to an isolated "
                        "<project>/mock store. Claims no rig.")
    p.add_argument("--heartbeat-s", dest="heartbeat_s", type=float,
                   default=DEFAULT_HEARTBEAT_S, metavar="S",
                   help=f"seconds between heartbeat lines "
                        f"(default {DEFAULT_HEARTBEAT_S:g})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=CONSOLE_SCRIPT,
        description="Hold the chamber at a humidity and measure nothing. "
                    "Humidity only — no heater is driven.",
        epilog=f"Read-only first: '{CLI} plan --rh 45 --duration-h 4', then "
               f"'{CLI} hold --rh 45 --duration-h 4 --execute' at the rig.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="what a hold would do; opens nothing")
    _add_hold_args(plan)
    plan.set_defaults(func=_cmd_plan)

    hold = sub.add_parser("hold", help="hold the humidity (DRY RUN unless --execute)")
    _add_hold_args(hold)
    hold.add_argument("--execute", action="store_true",
                      help="actually actuate the humidifier. Without it, nothing "
                           "is opened.")
    hold.add_argument("--yes", "-y", action="store_true",
                      help="skip the actuation confirmation prompt")
    hold.add_argument("--quiet", action="store_true",
                      help="suppress the per-interval heartbeat line. Milestones, "
                           "the teardown report and every alert still print.")
    hold.set_defaults(func=_cmd_hold)
    return p


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
