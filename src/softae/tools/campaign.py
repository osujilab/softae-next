"""``softae-campaign`` — run an autonomous campaign headlessly (P6.2).

    softae-campaign check   <spec.toml>            # parse + project, run nothing
    softae-campaign run     <spec.toml> [--yes] [--resume] [--mock]
    softae-campaign resume  <spec.toml>            # alias for `run --resume`

Exit codes: 0 ok · 1 campaign parked or failed · 2 usage/spec error · 3 declined.

**This does not reimplement the campaign.** It calls the same
:func:`~softae.core.autonomous_wiring.run_autonomous_campaign` the GUI does, so
workflow generation, the interlocks, and the resume path are literally the same
code — the requirement that the two surfaces never diverge is met by there being
only one implementation, not by keeping two in step.

What *is* CLI-specific is how the human gates are answered, since there is no
window to raise a dialog in. Every one of them defaults to the **safe** answer:

* **Head position** must be stated explicitly (``--head-up`` / ``--head-down``)
  or confirmed on the terminal. It is never assumed — the loop drives the head
  with conditional commands, so a wrong belief costs one wrong flip.
* **Board exchange** cancels: swapping a plate is physical, and nobody is there.
* **Board freshness** resumes past used wells, never re-casting them.
* A **projected stock shortfall** stops the run unless ``--yes`` is given.

Stopping is a CLI concern for the same reason. There is no window to close, so
:mod:`softae.core.shutdown` installs SIGINT/SIGTERM/SIGBREAK handlers that drive
the rig safe **while the instruments are still connected** — the teardown that
disconnects them is what makes a later park impossible — and the next launch
picks up whatever a hard kill left behind, via the unfinished run row.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from softae.core.campaign_spec_io import SpecLoadError, load_campaign_spec
from softae.tools import use_utf8_console

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_DECLINED = 3


# ── Output ───────────────────────────────────────────────────────────────────

def _emit(event: dict[str, Any]) -> None:
    """Print the campaign event stream in a form a human or a log can read."""
    etype = event.get("type", "?")
    if etype == "suggestion":
        print(f"  [{event.get('iteration')}] suggest {_fmt(event.get('params'))}",
              flush=True)
    elif etype == "result":
        print(f"  [{event.get('iteration')}] -> {event.get('objective'):.6g}",
              flush=True)
    elif etype == "park":
        print(f"!! PARKED: {event.get('reason')}", flush=True)
    elif etype == "safe_park":
        state = "ok" if event.get("ok") else f"INCOMPLETE {event.get('errors')}"
        print(f"   safe-park: {state}", flush=True)
    elif etype == "resumed":
        print(f"   resumed at iteration {event.get('iteration')} with "
              f"{event.get('n_observations')} observation(s)", flush=True)
        for warning in event.get("warnings") or []:
            print(f"   note: {warning}", flush=True)
    elif etype in ("run_started", "converged", "run_finished", "board_check",
                   "step_skipped", "step_recovered"):
        print(f"   {etype}: "
              f"{ {k: v for k, v in event.items() if k != 'type'} }", flush=True)


def _report_park(reason: str, result) -> None:
    """Say what the shutdown park did, on the terminal and in the log.

    Printed rather than only logged because the operator who pressed Ctrl-C is
    looking at this terminal, and "which subsystems refused to go safe" is the
    one thing they need before walking away from the rig.

    ``result.describe()`` rather than a verdict: it separates what was
    *commanded* from what was *verified* (nothing, on this rig) and names the
    head as unverifiable. A shutdown park does not move the head, so "the rig is
    safe" is the one sentence this must not print.
    """
    print(f"\n!! PARKING ({reason})", flush=True)
    print(result.describe(), flush=True)


def _fmt(params: Any) -> str:
    if not isinstance(params, dict):
        return str(params)
    return ", ".join(f"{k}={float(v):g}" for k, v in params.items())


# ── Gates (headless equivalents of the GUI dialogs) ──────────────────────────

def _confirm(prompt: str, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin or not sys.stdin.isatty():
        # Non-interactive and not pre-approved: refuse rather than guess. A
        # cron-launched run that silently assumed "yes" would be exactly the
        # unattended failure this whole phase exists to prevent.
        print(f"{prompt} — no terminal to ask on; pass --yes to pre-approve.")
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _register_head_state(manager, args) -> bool:
    """Establish the dispenser-head belief before anything moves."""
    if args.head_up or args.head_down:
        is_up = bool(args.head_up)
    else:
        if not sys.stdin or not sys.stdin.isatty():
            print("Head position unknown: pass --head-up or --head-down "
                  "(the loop issues conditional head commands, so it must not "
                  "be guessed).")
            return False
        answer = input("Is the dispenser head raised? [y/N] ").strip().lower()
        is_up = answer in ("y", "yes")

    try:
        syringe = manager.get("syringe")
        if hasattr(syringe, "set_head_state"):
            syringe.set_head_state(is_up)
    except Exception as exc:
        print(f"Could not register head state: {exc}")
        return False
    print(f"   head registered as {'raised' if is_up else 'LOWERED'}")
    return True


# ── Anti-clog purge (P8) — the same wiring the GUI does ──────────────────────

def _attach_purge(manager, store=None):
    """Attach the purge scheduler on the headless path, as the GUI does.

    Without this, ``attach_purge_scheduler`` was reachable **only** from
    ``gui/main_window.py``, so a headless campaign resolved a
    :class:`~softae.core.purge_runner.NullPurgeRunner` and never purged — while
    :func:`_project` billed the purge consumption to the runway all the same. A
    projection that charges for something the run cannot do is the same class of
    untruth as a comment that outlived its behaviour.

    Attachment is orthogonal to actuation: ``[purge] actuate`` still governs
    whether fluid moves, and it still ships **off**, so this makes the headless
    path log its due purges as dry runs exactly like the GUI's does.

    ``[purge] enabled`` is honoured by declining to attach at all. A disabled
    schedule bills nothing (``uL_per_day()`` is empty), so attaching one would
    stand up the executor's concurrent purge window — a thread and a task per
    co-runnable step — around something that can never fire.
    """
    from softae.core.purge import attach_purge_scheduler, load_purge_settings

    settings = load_purge_settings(store)
    if not settings.enabled:
        return None
    return attach_purge_scheduler(manager, settings, data_store=store)


def _purge_is_attached(manager) -> bool:
    """Will a purge actually be scheduled on this rig?

    Answers the question :func:`softae.core.autonomous_wiring._resolve_purge_runner`
    answers at run time — a real runner iff the syringe carries a runner or a
    scheduler — without constructing anything, so asking it during a projection
    has no side effect on the rig.
    """
    try:
        syringe = manager.get("syringe")
    except Exception:
        return False
    return (getattr(syringe, "purge_runner", None) is not None
            or getattr(syringe, "purge_scheduler", None) is not None)


# ── Projection ───────────────────────────────────────────────────────────────

def _project(spec, manager, *, assume_yes: bool, store=None) -> bool:
    """Print the duration/stock projection; stop on a predicted shortfall."""
    from softae.config import loader
    from softae.core.preflight import project_campaign
    from softae.core.purge import load_purge_settings
    from softae.core.task_catalog import TaskCatalog

    purge = load_purge_settings(store)
    # Bill purge consumption only when a purge can actually happen. The rate and
    # the attachment are two independent facts and the projection must read the
    # second, not assume it from the first.
    billed = _purge_is_attached(manager)
    try:
        catalog = TaskCatalog.load_toml(loader.tasks_toml_path())
        ledger = getattr(manager.get("syringe"), "reservoir_ledger", None)
        projection = project_campaign(
            spec, catalog=catalog, ledger=ledger,
            purge_uL_per_day=purge.uL_per_day() if billed else {})
    except Exception as exc:
        print(f"   (projection unavailable: {exc})")
        return True

    print()
    if billed or not purge.enabled:
        print(purge.describe())
    else:
        print("Anti-clog purging is configured but no scheduler is attached "
              "here — nothing will purge, so nothing is billed.")
    print(projection.describe())
    print()

    if projection.stock_sufficient is False:
        return _confirm("Declared stock will not cover the budget. Start anyway?",
                        assume_yes=assume_yes)
    return True


# ── Calibration freshness (advisory) ─────────────────────────────────────────

def _calibration_advisory(spec) -> None:
    """Report EIS calibration state for the channels this campaign declares.

    **Advisory only — never a stop.** With the shipped ``[eis] engine =
    "legacy"`` the commissioning constants are not applied to anything the
    campaign computes, so refusing to start over a missing short blank would
    block a run for a reason that does not bear on its numbers. It is printed
    because it *will* bear on them the moment the engine is flipped, and an
    operator deciding to flip it should not have to run a second command to
    find out which channels are uncommissioned.

    Reads exactly what ``softae-commission status`` reads —
    :func:`~softae.analysis.eis.calibration.resolve_calibration` on
    ``[eis.fixture] fixture_id`` — so the two surfaces cannot disagree. Read
    only: nothing here writes, derives or invalidates a calibration.

    The channels reported are the ones the **spec declares**; a board-aware
    campaign may allocate elsewhere on the same board, and a channel absent
    from the calibration is uncommissioned wherever it is used.
    """
    try:
        from softae.analysis.eis.calibration import (
            describe_or_absent,
            resolve_calibration,
        )
        from softae.analysis.eis.settings import eis_settings

        modality = getattr(getattr(spec, "measurement", None), "modality", "eis")
        if modality != "eis":
            return

        eis = eis_settings()
        fixture = eis.fixture.fixture_id or "default"
        calibration = resolve_calibration(fixture)
    except Exception as exc:  # noqa: BLE001 - an advisory never breaks preflight
        print(f"   (calibration state unavailable: {exc})")
        return

    measured = set(getattr(calibration, "channels_measured", ()) or ())
    assumed = set(getattr(calibration, "channels_assumed", ()) or ())
    channels = [int(c) for c in (getattr(spec, "channels", ()) or ())]
    uncalibrated = [c for c in channels if c not in measured and c not in assumed]
    inherited = [c for c in channels if c in assumed and c not in measured]

    print(f"EIS calibration [advisory] — fixture '{fixture}':")
    print(f"   {describe_or_absent(calibration)}")
    if uncalibrated:
        print("   uncalibrated channels: "
              + ", ".join(str(c) for c in uncalibrated))
    if inherited:
        print("   channels inheriting another channel's constants: "
              + ", ".join(str(c) for c in inherited))
    if channels and not uncalibrated and not inherited:
        print("   every declared channel was measured on this fixture.")
    # Which engine is running is printed via the settings' own rendered line
    # rather than by reading `[eis] engine` here. `analyze_spectrum` is the ONE
    # place that key is resolved (T2.6b / [a23]), and a second read — even one
    # that only formats a sentence — is how a second opinion starts.
    print(f"   {eis.describe()}")
    print("   [advisory] Commissioning constants are applied by the gated engine "
          "only, and nothing on this line ever stops a campaign.")
    print()


# ── Commands ─────────────────────────────────────────────────────────────────

def _cmd_check(args) -> int:
    """Parse and project without touching hardware."""
    spec = load_campaign_spec(args.spec)
    print(f"Campaign '{spec.name}': {len(spec.parameter_space)} parameter(s), "
          f"budget {spec.budget}, channels {list(spec.channels)}")

    from softae.drivers.mock_factory import create_mock_manager

    manager = create_mock_manager(config={})
    # Attached here too, so `check` projects the run that `run` would perform
    # rather than a purge-free variant of it.
    _attach_purge(manager, None)
    _project(spec, manager, assume_yes=True, store=None)
    _calibration_advisory(spec)

    from softae.core.data_store import DataStore

    store = DataStore(args.project) if args.project else None
    if store is not None:
        try:
            from softae.core.campaign_resume import describe_resume, load_resume_plan

            plan = load_resume_plan(store, spec, strict=False)
            if plan is not None:
                print("A checkpoint exists for this campaign:")
                print(describe_resume(plan))
        finally:
            store.close()
    return EXIT_OK


def _cmd_run(args) -> int:
    spec = load_campaign_spec(args.spec)

    from softae.core.autonomous_wiring import run_autonomous_campaign
    from softae.core.data_store import DataStore
    from softae.core.reservoir import attach_reservoir_ledger
    from softae.core.run_lock import (
        busy_rig_message,
        foreign_run_lock,
        rig_is_simulated,
    )

    from softae.core.shutdown import (
        ParkGuard,
        detect_unfinished_runs,
        install_signal_park,
        park_on_shutdown,
        recover_from_unclean_shutdown,
    )

    # `InstrumentManager.from_config()` never existed — this raised AttributeError on
    # every non-mock invocation, so `softae-campaign run` has only ever worked with
    # --mock. `create_manager` is the factory the GUI uses. `mock=False` forces real
    # drivers and raises if they are unavailable, rather than the auto mode's silent
    # fall-back: a campaign that quietly ran on simulated instruments would record
    # fabricated spectra as real data.
    from softae.drivers.factory import create_manager

    manager = create_manager(mock=True if args.mock else False)

    # ── Liveness BEFORE recovery. This ordering is the fix — do not reorder. ──
    #
    # `detect_unfinished_runs` (below) is project-wide and cannot tell a *live*
    # run's row from a crashed one. Asked while another campaign is genuinely
    # running, it reads that campaign's own row, marks it `interrupted` and parks
    # the rig underneath it — killing the run it was meant to protect. Asking who
    # holds the rig first is what makes that unreachable: a live holder is
    # refused here and never reaches the recovery path at all.
    holder = foreign_run_lock()

    # Refused only when this run would itself claim the rig. `run_autonomous_campaign`
    # takes the lock under exactly this predicate (`autonomous_wiring.py`, "the rig is
    # claimed for the whole campaign"), so contention is defined in one place rather
    # than guessed at twice — and a `--mock` run, which claims nothing and moves
    # nothing, is not refused over hardware it will never touch.
    if holder is not None and not rig_is_simulated(manager):
        print(f"\n!! NOT STARTING '{spec.name}'\n\n"
              f"{busy_rig_message(holder, action='This campaign')}", flush=True)
        return EXIT_DECLINED

    # `--project` is required by the parser for `run`/`resume`, so the store is
    # never None here — the campaign always has somewhere to record what it did.
    store = DataStore(args.project)
    attach_reservoir_ledger(manager, store)

    # Same choke point as the stock ledger, for the same reason: every dispense
    # the campaign makes must reset that line's purge timer, or the harness pays
    # the full idle rate for lines the run had just used itself.
    _attach_purge(manager, store)

    guard = ParkGuard(manager, on_park=_report_park)

    try:
        if not _register_head_state(manager, args):
            return EXIT_DECLINED
        if not _project(spec, manager, assume_yes=args.yes, store=store):
            return EXIT_DECLINED
        _calibration_advisory(spec)

        # Queried before anything connects so the operator is told early, but
        # acted on after (see `_go`): a park needs live sessions.
        #
        # Skipped outright while another process holds the rig. Only a *real* run
        # is refused above, so this is reachable on the `--mock` path — where the
        # rig is not at stake but the DataStore still is: `record_unclean_shutdown`
        # would mark the live campaign's row `interrupted`, and a simulated run
        # corrupts a real run's record just as thoroughly as a real one would.
        #
        # Deferring loses nothing. The unfinished row is durable and never clears
        # itself, so a genuinely crashed run is still detected and parked at the
        # next launch that is not racing a live campaign.
        unfinished = None if holder is not None else detect_unfinished_runs(store)
        if holder is not None:
            print(f"\n   skipping unclean-shutdown recovery — {holder.describe()}\n"
                  "   Unfinished runs are durable; they will be handled at the "
                  "next launch.", flush=True)
        if unfinished:
            print(f"\n!! PREVIOUS SESSION DID NOT FINISH\n   "
                  f"{unfinished.describe()}", flush=True)

        print(f"Starting '{spec.name}'"
              f"{' (resuming)' if args.resume else ''}...", flush=True)

        async def _go():
            await manager.connect_all()
            try:
                if unfinished:
                    # Sited here, not before `connect_all`, because `safe_park`
                    # skips every instrument that is not `is_connected` — a
                    # recovery park issued earlier would record four skips and
                    # move nothing.
                    recover_from_unclean_shutdown(
                        manager, store, unfinished=unfinished, report=print)
                return await run_autonomous_campaign(
                    spec,
                    manager=manager,
                    data_store=store,
                    on_event=_emit,
                    # Headless gates default to the safe answer: never swap a
                    # plate nobody is there to change, never re-cast a used well.
                    on_board_exchange=None,
                    on_board_check=None,
                    resume=bool(args.resume),
                )
            except BaseException:
                # The campaign parks in its own catch-all, and the guard makes
                # this a no-op when it did. It is here for the exits that never
                # reached the campaign at all — a signal during `connect_all`,
                # the recovery park raising — because the very next statement
                # disconnects, and after that a park is impossible rather than
                # merely late.
                park_on_shutdown(manager, "campaign CLI unwinding")
                raise
            finally:
                # Last, always: `safe_park` needs these sessions open, so every
                # park decision above has to have been made by now.
                await manager.disconnect_all()

        with install_signal_park(guard):
            result = asyncio.run(_go())
    except SpecLoadError:
        raise
    except KeyboardInterrupt:
        print(f"\nInterrupted — {guard.describe()}.")
        print("The checkpoint is retained; resume with `softae-campaign resume`.")
        return EXIT_FAILED
    finally:
        store.close()

    print()
    print(f"{result.final_state}: {result.n_trials} trial(s)")
    if result.best_params is not None:
        print(f"best {result.best_objective:.6g} at {_fmt(result.best_params)}")
    if result.park_reason:
        print(f"PARKED: {result.park_reason}")
    # Zero means "ended on purpose", which a park is not. `_park` leaves the loop
    # in STOPPED — the same state a clean stop leaves — so state alone reported a
    # 3 a.m. hard-fault park as success and contradicted this CLI's own header.
    #
    # Character for character the predicate `run_autonomous_campaign` uses to
    # decide whether to drop the checkpoint, so the CLI and the checkpoint cannot
    # come to disagree about what "on purpose" means. That is why `park_reason`
    # disqualifies a CONVERGED run too: a campaign that converged *and* parked
    # keeps its checkpoint, and a run whose checkpoint was worth keeping did not
    # finish.
    ended_on_purpose = (
        result.final_state in ("CONVERGED", "STOPPED") and not result.park_reason
    )
    return EXIT_OK if ended_on_purpose else EXIT_FAILED


# ── Entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softae-campaign",
        description="Run an autonomous deposition campaign without a GUI.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp, *, project_required: bool):
        sp.add_argument("spec", help="campaign spec (.toml)")
        # Required to *run*, optional to `check`. A campaign without a store
        # records no measurements, writes no checkpoint, and so cannot be
        # resumed — the run happens and the evidence of it does not. `check`
        # runs nothing and so needs nowhere to put it.
        sp.add_argument("--project", required=project_required,
                        help="project directory for the data store"
                             + ("" if project_required else " (optional)"))
        return sp

    _common(sub.add_parser("check", help="parse and project; run nothing"),
            project_required=False)

    run = _common(sub.add_parser("run", help="run the campaign"),
                  project_required=True)
    run.add_argument("--yes", "-y", action="store_true",
                     help="pre-approve prompts (required when not on a terminal)")
    run.add_argument("--resume", action="store_true",
                     help="continue this campaign's saved checkpoint")
    run.add_argument("--mock", action="store_true",
                     help="run against mock instruments")
    head = run.add_mutually_exclusive_group()
    head.add_argument("--head-up", action="store_true",
                      help="dispenser head is raised")
    head.add_argument("--head-down", action="store_true",
                      help="dispenser head is lowered")

    res = _common(sub.add_parser("resume", help="alias for `run --resume`"),
                  project_required=True)
    res.add_argument("--yes", "-y", action="store_true")
    res.add_argument("--mock", action="store_true")
    rhead = res.add_mutually_exclusive_group()
    rhead.add_argument("--head-up", action="store_true")
    rhead.add_argument("--head-down", action="store_true")
    return p


def main(argv: "list[str] | None" = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    if args.command == "resume":
        args.resume = True
        args.command = "run"

    try:
        if args.command == "check":
            return _cmd_check(args)
        return _cmd_run(args)
    except SpecLoadError as exc:
        print(f"Spec error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"Campaign failed: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
