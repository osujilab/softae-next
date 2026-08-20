"""``softae-commission`` — run the EIS commissioning pass and derive the calibration.

A CLI rather than a GUI tab, for the same reason ``softae-campaign`` is one: this is a
**bench task**. The operator is standing at the rig with a jumpered board in one hand,
swapping physical parts between sweeps. A terminal that prints what to install next and
waits for Enter fits that better than a tab they have to walk back to.

The four subcommands map onto the four things an operator actually does::

    softae-commission status                     # what is calibrated, what is missing
    softae-commission run blank_short --channels 1-8
    softae-commission derive --fixture mux16     # spectra -> CalibrationSet -> TOML
    softae-commission history --fixture mux16    # successive sets = drift

``run`` acquires and tags; ``derive`` reads the tagged spectra back and writes the
canonical ``calibration/eis/<fixture_id>.toml``. They are separate because acquisition
happens over several sessions as parts arrive — the reference capacitor in particular is
usually the one still on order — and each ``derive`` produces the best calibration the
artifacts so far support.

**Order matters**, and ``status`` tells you the next artifact rather than listing every
absence: short blank → load blank → reference capacitor, by value per hour of bench time
(framework §7.4).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from softae.analysis.eis.calibration import (
    ARTIFACT_NOMINAL_UNITS,
    COMMISSIONING_ROLES,
    describe_or_absent,
    hardware_hash,
    load_calibration,
    resolve_calibration,
    save_calibration,
)
from softae.analysis.eis.policy import RE_STATES
from softae.core.hardware_safety import ARM_ENV_VAR, HardwareNotArmedError
from softae.tools import run_finalizer, use_utf8_console
from softae.workflows.commissioning import (
    ARTIFACT_SETUP,
    CommissioningError,
    build_commissioning_workflow,
    derive_calibration,
    next_artifact,
)

logger = structlog.get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_DECLINED = 2


def _parse_channels(text: str) -> list[int]:
    """``"1, 3-6"`` → ``[1, 3, 4, 5, 6]`` — the same syntax the GUI accepts."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen: set[int] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _open_store(args):
    """The project store, defaulting to the one the GUI and campaigns already use.

    ``--project`` was originally required, which is a bad demand to make of someone
    standing at the rig with a jumper in one hand: the answer is not a choice, it is
    *the same store as everything else*, and the operator has no reason to know its
    path. Defaulting to ``[data] project_dir`` also makes ``derive`` work across
    sessions without anyone remembering which directory the last one used — which is
    the whole point of acquiring artifacts weeks apart.
    """
    from softae.config import loader
    from softae.core.data_store import DataStore

    project = args.project
    if not project:
        project = loader.data_project_dir()
        # A --mock run writes synthetic spectra tagged exactly like real ones. Landing
        # those in the production store would let a later `derive` build a calibration
        # from simulated data with nothing in the record to say so — which happened
        # once during development and had to be deleted by hand. Dry runs get their
        # own store unless a project is named explicitly.
        if getattr(args, "mock", False):
            project = str(Path(project).expanduser() / "mock")
    return DataStore(project, db_filename=loader.data_db_filename()), project


# ── status ───────────────────────────────────────────────────────────────────

def _cmd_status(args) -> int:
    cal = resolve_calibration(args.fixture)
    print(describe_or_absent(cal))
    print()

    if cal is not None:
        raw = load_calibration(args.fixture)
        if raw is not None and raw.is_stale(current_hash=hardware_hash()):
            print("  ⚠ STALE — this calibration was taken on different hardware.")
            print(f"    recorded {raw.hardware_hash or '(none)'} vs "
                  f"current {hardware_hash()}")
            print("    Its constants are NOT being applied. Re-run commissioning.")
            print()

    nxt = next_artifact(cal)
    if nxt is None:
        print("Nothing left to run — every commissioning artifact is present.")
        return EXIT_OK

    print(f"Next: {nxt}")
    print(f"  needs: {ARTIFACT_SETUP.get(nxt, '?')}")
    print(f"  then:  softae-commission run {nxt} --channels 1-8"
          f"{' --nominal <value>' if nxt in ARTIFACT_NOMINAL_UNITS else ''}")
    return EXIT_OK


# ── run ──────────────────────────────────────────────────────────────────────

def _cmd_run(args) -> int:
    from softae.workflows.workflow_executor import WorkflowExecutor

    channels = _parse_channels(args.channels)
    if not channels:
        print("No channels parsed from --channels.", file=sys.stderr)
        return EXIT_FAILED

    unit = ARTIFACT_NOMINAL_UNITS.get(args.role)
    if unit and args.nominal is None:
        print(f"'{args.role}' needs --nominal (its marked value, in {unit}).",
              file=sys.stderr)
        print("  The marking and the measurement disagreeing is exactly the check",
              file=sys.stderr)
        print("  that catches an unusable reference part — it needs both numbers.",
              file=sys.stderr)
        return EXIT_FAILED

    print(f"Commissioning: {args.role} on channel(s) "
          f"{', '.join(str(c) for c in channels)}")
    print(f"  fixture: {args.fixture}")
    print(f"  INSTALL: {ARTIFACT_SETUP.get(args.role, '?')}")

    from softae.analysis.eis.calibration import TWO_TERMINAL_ROLES

    needs_two = args.role in TWO_TERMINAL_ROLES
    if needs_two:
        print()
        print("  JUMPER: tie RE to CE at the connector (two-electrode sensing).")
        print("    A two-terminal load gives the reference no ionic path, so in")
        print("    three-electrode mode RE floats onto a capacitive divider whose")
        print("    ratio depends on the load — measured 2.2x to 23x (overhaul F17).")
        print("    A value taken that way is uncalibratable, not merely uncalibrated,")
        print("    and will be REFUSED at derive time.")
    print()

    if not args.yes:
        try:
            reply = input("Is the hardware in place? [y/N] ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            print("Declined — nothing measured.")
            return EXIT_DECLINED

        if needs_two and args.electrode_mode is None:
            # Asked separately from "is the hardware in place", and deliberately so:
            # the jumper is the single most consequential thing the operator can get
            # wrong here, and folding it into a general yes/no invites a reflex press.
            try:
                jumper = input("Is RE tied to CE (two-electrode)? [y/N] ").strip().lower()
            except EOFError:
                jumper = ""
            if jumper not in ("y", "yes"):
                print("Declined — measure two-terminal references with RE tied to CE.",
                      file=sys.stderr)
                print("  Re-run once the jumper is fitted, or pass --electrode-mode "
                      "three\n  to record the spectrum anyway (it will not enter a "
                      "calibration).", file=sys.stderr)
                return EXIT_DECLINED
            args.electrode_mode = "two"

    if args.electrode_mode is None:
        args.electrode_mode = "two" if needs_two else "three"
    print(f"  electrode mode: {args.electrode_mode}-electrode")

    from softae.drivers.factory import create_manager

    # mock=False forces real drivers and *raises* if they are unavailable, rather
    # than the auto mode's silent fall-back to mocks. That strictness is the whole
    # point here: a commissioning sweep that quietly ran on simulated instruments
    # would write synthetic spectra tagged `blank_short` into the store, and a later
    # `derive` would build a calibration from them with nothing to say they were not
    # real. Failing loudly is the only safe behaviour for this command.
    try:
        manager = create_manager(mock=True if args.mock else False)
    except Exception as exc:
        print(f"Could not open the instruments: {exc}", file=sys.stderr)
        print("  Commissioning refuses to fall back to simulated drivers — a mock",
              file=sys.stderr)
        print("  spectrum tagged as a real blank would corrupt the calibration.",
              file=sys.stderr)
        print("  Use --mock to exercise the workflow without hardware.",
              file=sys.stderr)
        return EXIT_FAILED

    # Check the interlock *before* opening a store or starting a run. The executor
    # asserts it too — that is the real choke point — but reaching it first would
    # leave an empty run row behind on every declined attempt, and an operator
    # arming the rig on the second try would find the first one already recorded.
    from softae.core.hardware_safety import assert_hardware_armed

    try:
        assert_hardware_armed(manager, action=f"run the {args.role} sweep")
    except HardwareNotArmedError as exc:
        print(f"\nHardware is not armed: {exc}", file=sys.stderr)
        print(f"  Set {ARM_ENV_VAR}=1 in this shell and re-run. It is deliberately",
              file=sys.stderr)
        print("  a conscious, session-scoped act — a headless command must not arm",
              file=sys.stderr)
        print("  real motion hardware on its own.", file=sys.stderr)
        return EXIT_DECLINED

    store, _project = _open_store(args)
    # The *resolved* path, not the configured string: "~/softae_data" tells an
    # operator nothing about where their commissioning data actually went, and this
    # is the one line that lets them check.
    print(f"  recording to: {store.project_dir}")
    if args.mock:
        print("  ⚠ MOCK — synthetic spectra. Do not derive a calibration from these.")

    wf = build_commissioning_workflow(
        args.role, channels, fixture_id=args.fixture, nominal=args.nominal,
        electrode_mode=args.electrode_mode)

    run_id = store.start_run(wf.name, mode="commissioning",
                             annotation=f"{args.role} / {args.fixture}")
    finalize = run_finalizer(store, run_id)

    async def _go():
        await manager.connect_all()
        try:
            executor = WorkflowExecutor(manager, data_store=store, run_id=run_id)
            return await executor.run(wf)
        finally:
            await manager.disconnect_all()

    try:
        asyncio.run(_go())
    except HardwareNotArmedError as exc:
        # By design: headless paths cannot self-arm, so the operator arms the rig
        # deliberately and session-scoped. Say how, rather than surfacing the raw
        # exception at the bench.
        finalize("aborted")
        print(f"\nHardware is not armed: {exc}", file=sys.stderr)
        print(f"  Set {ARM_ENV_VAR}=1 in this shell and re-run. It is deliberately",
              file=sys.stderr)
        print("  a conscious, session-scoped act — a headless command must not arm",
              file=sys.stderr)
        print("  real motion hardware on its own.", file=sys.stderr)
        return EXIT_DECLINED
    except KeyboardInterrupt:
        finalize("interrupted")
        print("\nInterrupted — partial spectra are recorded and can be re-run.")
        return EXIT_FAILED
    else:
        finalize("done")
    finally:
        # The catch-all, and it must run before `store.close()` — a closed
        # connection cannot record anything. Idempotent, so it is a no-op unless
        # an exception no `except` above names is on its way out.
        finalize("error")
        store.close()

    print()
    print(f"Recorded run {run_id} with role='{args.role}'.")
    print("Derive the calibration when the artifacts you want are in:")
    print(f"  softae-commission derive --fixture {args.fixture}")
    return EXIT_OK


# ── derive ───────────────────────────────────────────────────────────────────

def _load_role_spectra(
    store: Any, fixture_id: str
) -> tuple[dict[str, list], dict[str, float], dict[str, str]]:
    """Read back every commissioning spectrum recorded for this fixture.

    Returns ``(artifacts, nominals, modes)``. Spectra come from the *database*, not from
    whatever this process just measured, so a calibration can be assembled from
    artifacts acquired weeks apart — which is how it actually happens when the
    reference capacitor is still on order.

    The **nominals come back with them**. Requiring them again on the ``derive``
    command was a silent trap: the derivation needs a marked value to compute a load
    error, so omitting the flag left ``load_error_pct`` NaN, left
    ``can_validate_correction`` false, and made ``next_artifact`` keep asking for a
    load blank that had already been measured.

    They come back **per acquisition**, on :class:`AcquiredSpectrum`. The database has
    always stored ``nominal_value`` and ``electrode_mode`` per measurement row; this
    function used to fold them into role-level dicts on the way out, last row winning,
    which is a lossy summary of data that was never ambiguous. On the mux16 record that
    meant three reference capacitors marked 1e-10, 1e-10 and 1e-9 F all derived against
    1e-9, and a pre-jumper ``unknown``-mode sweep inherited a later sibling's "two".

    The role-level dicts are still returned, but their meaning has narrowed: *nominals*
    is now only the **fallback** for an acquisition that recorded none of its own, and
    *modes* is a display summary (``"mixed"`` where a role's rows disagree).
    """
    import numpy as np

    from softae.analysis.eis_data import EISResult
    from softae.workflows.commissioning import AcquiredSpectrum

    out: dict[str, list] = {}
    nominals: dict[str, float] = {}
    seen_modes: dict[str, set[str]] = {}
    rows = store._conn.execute(
        "SELECT measurement_id, channel, role, eis_file_path, nominal_value, "
        "electrode_mode "
        "FROM measurements "
        "WHERE role != 'sample' AND (fixture_id = ? OR fixture_id IS NULL) "
        "ORDER BY measurement_id",
        (fixture_id,),
    ).fetchall()

    for mid, channel, role, rel_path, nominal, mode in rows:
        mode = str(mode or "unknown")
        if nominal is not None:
            # Last one wins — but only as the fallback for acquisitions that recorded
            # nothing. Each acquisition's own value governs its own derivation.
            nominals[role] = float(nominal)
        if not rel_path:
            continue
        path = Path(store.project_dir) / rel_path
        if not path.exists():
            logger.warning("commissioning_spectrum_missing", measurement_id=mid,
                           path=str(path))
            continue
        try:
            eis = EISResult.load(path)
        except Exception:
            logger.warning("commissioning_spectrum_unreadable",
                           measurement_id=mid, exc_info=True)
            continue
        Z = np.asarray(eis.z_real, dtype=float) - 1j * np.asarray(
            eis.z_imag_neg, dtype=float)
        seen_modes.setdefault(role, set()).add(mode)
        out.setdefault(role, []).append(AcquiredSpectrum(
            channel=int(channel),
            freq_hz=np.asarray(eis.frequency),
            Z=Z,
            nominal=float(nominal) if nominal is not None else None,
            electrode_mode=mode,
            measurement_id=int(mid),
        ))

    modes = {role: (next(iter(seen)) if len(seen) == 1 else "mixed")
             for role, seen in seen_modes.items()}
    return out, nominals, modes


def _declare_electrode_mode(
    store: Any, fixture_id: str, mode: str, artifacts: dict[str, list]
) -> None:
    """Record an operator's assertion about how existing spectra were sensed.

    The refusal exists to stop an *unknown* being silently trusted, not to stop the
    operator saying what they know. Someone who remembers fitting the jumper has
    genuine information the database does not, and re-measuring to supply it would be
    wasteful.

    Two constraints keep the affordance from becoming a laundering route:

    **Only ``unknown`` rows are touched.** A spectrum explicitly recorded as
    three-electrode stays refused, because that is not missing information — it is
    information saying the value is uncalibratable (F17). Overriding it would be
    asserting something known to be false; re-import the file if the record itself is
    wrong.

    **The declaration is persisted, not applied in passing.** It is written back to the
    measurement rows, so the next derive does not ask again and the provenance shows
    what was asserted rather than what was measured. The caller re-reads the spectra
    afterwards, so what the derivation sees is what the database now says.

    Roles are inspected **per acquisition**. A role summarised by its newest row would
    report "two" for a set that still contains an unknown-mode sweep, and this branch
    would then skip the very rows it exists to fix.
    """
    changed: list[str] = []
    for role, items in sorted(artifacts.items()):
        unknown = [a for a in items if (a.electrode_mode or "unknown") == "unknown"]
        if not unknown:
            explicit = sorted({a.electrode_mode for a in items if a.electrode_mode})
            if explicit and mode not in explicit:
                print(f"  ! {role} is recorded as {'/'.join(explicit)}-electrode — not "
                      f"overridden. Re-import the file if that record is wrong.")
            continue
        cur = store._conn.execute(
            "UPDATE measurements SET electrode_mode = ? "
            "WHERE role = ? AND electrode_mode = 'unknown' "
            "AND (fixture_id = ? OR fixture_id IS NULL)",
            (mode, role, fixture_id),
        )
        if cur.rowcount:
            changed.append(f"{role} ({cur.rowcount} row(s))")
    store._conn.commit()

    if changed:
        print(f"  Declared {mode}-electrode for: {', '.join(changed)}")
        print("    Recorded against the measurements — this is now their provenance.")
        logger.warning("commissioning_electrode_mode_declared", fixture=fixture_id,
                       mode=mode, roles=changed,
                       msg="operator assertion, not a measured fact")
    else:
        print(f"  (nothing to declare — no '{fixture_id}' spectra had an unknown mode)")


def _print_found(artifacts: dict[str, list], modes: dict[str, str]) -> None:
    """What the database holds, per acquisition rather than per role.

    A role line reading "reference_cap: 3 channel(s) [two-electrode]" hid exactly the
    facts that turned out to matter — that one of the three was recorded at a different
    marking, and one at an unknown mode. Listing them is how an operator can see the
    thing the derivation is about to act on.
    """
    print("Found:")
    for role, items in sorted(artifacts.items()):
        unit = ARTIFACT_NOMINAL_UNITS.get(role, "")
        print(f"  {role}: {len(items)} acquisition(s) "
              f"[{modes.get(role, 'unknown')}-electrode]")
        for acq in items:
            marked = f"{acq.nominal:g} {unit}" if acq.nominal is not None else \
                ("no marking" if unit else "—")
            print(f"      #{acq.measurement_id} ch{acq.channel}: {marked}, "
                  f"{acq.electrode_mode or 'unknown'}-electrode")


def _warn_missing_nominals(
    artifacts: dict[str, list], nominals: dict[str, float],
    overrides: dict[str, float],
) -> None:
    """Say which acquisitions have no marking of their own, and what will be used."""
    for role in sorted(artifacts):
        if role not in ARTIFACT_NOMINAL_UNITS or role in overrides:
            continue
        missing = [a for a in artifacts[role] if a.nominal is None]
        if not missing:
            continue
        where = ", ".join(f"ch{a.channel}" for a in missing)
        if role in nominals:
            print(f"  ⚠ {role}: {where} recorded no marked value — falling back to the "
                  f"role's latest, {nominals[role]:g} "
                  f"{ARTIFACT_NOMINAL_UNITS[role]}.")
        else:
            print(f"  ⚠ {role}: no marked value recorded — its check will be "
                  f"skipped. Re-run it with --nominal, or pass "
                  f"--nominal-{role.split('_')[-1]} here.")


def _sources(artifacts: dict[str, list]) -> dict[str, int]:
    """role → newest contributing ``measurement_id``, for R17 traceability.

    One id per role is lossy where a role has several acquisitions, and deliberately
    so: ``CalibrationSet.sources`` is a pointer back into the record, not a second copy
    of it. The newest is the one an operator asking "which spectrum is this?" means.
    """
    out: dict[str, int] = {}
    for role, items in artifacts.items():
        ids = [a.measurement_id for a in items if a.measurement_id is not None]
        if ids:
            out[role] = max(ids)
    return out


def _cmd_derive(args) -> int:
    store, _project = _open_store(args)
    try:
        artifacts, nominals, modes = _load_role_spectra(store, args.fixture)
        if not artifacts:
            print(f"No commissioning spectra found for fixture '{args.fixture}'.",
                  file=sys.stderr)
            print("Run one first:  softae-commission run blank_short --channels 1-8",
                  file=sys.stderr)
            return EXIT_FAILED

        _print_found(artifacts, modes)

        if args.declare_electrode_mode:
            _declare_electrode_mode(
                store, args.fixture, args.declare_electrode_mode, artifacts)
            # Re-read rather than patch in place: the declaration was written to the
            # database, and what derives must be what the record now says.
            artifacts, nominals, modes = _load_role_spectra(store, args.fixture)

        # Each acquisition's own recorded value governs. The flags stay as a role-wide
        # OVERRIDE for a part that was mis-entered at acquisition time — they correct
        # the whole role's records, so they are announced rather than applied quietly.
        overrides: dict[str, float] = {}
        for role, value in (("blank_load", args.nominal_load),
                            ("reference_cap", args.nominal_cap),
                            ("reference_r", args.nominal_r)):
            if value is not None:
                overrides[role] = value
                unit = ARTIFACT_NOMINAL_UNITS.get(role, "")
                print(f"  ! {role}: overriding every acquisition's recorded marking "
                      f"with {value:g} {unit}.")

        _warn_missing_nominals(artifacts, nominals, overrides)

        try:
            cal = derive_calibration(
                artifacts,
                fixture_id=args.fixture,
                created_at=datetime.now().isoformat(timespec="seconds"),
                hardware_hash_value=hardware_hash(),
                nominals=nominals,
                nominal_overrides=overrides,
                sources=_sources(artifacts),
                all_channels=_parse_channels(args.channels) if args.channels else None,
                representative_channel=args.representative,
                electrode_modes=modes,
            )
        except CommissioningError as exc:
            # A setup mistake, not a crash: the operator can fix it and re-run, so
            # they get the reason rather than a traceback.
            print()
            print(str(exc), file=sys.stderr)
            return EXIT_FAILED

        print()
        print(cal.describe())
        print()

        path = save_calibration(cal)
        store.record_calibration(cal)
        print(f"Wrote {path}")
        print("  ^ commit this: framework §8.5 requires commissioning data to be")
        print("    version-controlled with the code.")
        print("Appended to the calibration history.")

        nxt = next_artifact(cal)
        if nxt:
            print()
            print(f"Next most valuable artifact: {nxt} "
                  f"({ARTIFACT_SETUP.get(nxt, '?')})")
    finally:
        store.close()
    return EXIT_OK


# ── history ──────────────────────────────────────────────────────────────────

def _cmd_history(args) -> int:
    store, _project = _open_store(args)
    try:
        rows = store.calibration_history(args.fixture)
        if not rows:
            print(f"No calibration history for '{args.fixture}'.")
            return EXIT_OK

        print(f"{len(rows)} calibration(s) for '{args.fixture}', oldest first.")
        print("Successive sets on the same hardware ARE a drift measurement.")
        print()
        for row in rows:
            cal = row["calibration"]
            live = "" if row["superseded_at"] else "  ← current"
            shorts = cal.get("R_short_ohm") or {}
            summary = (", ".join(f"ch{k}={float(v):.4g}Ω"
                                 for k, v in sorted(shorts.items())[:4])
                       or "no short blank")
            print(f"  [{row['created_at']}] @{row['hardware_hash'][:8] or '?':8s} "
                  f"{summary}{live}")
    finally:
        store.close()
    return EXIT_OK



# ── import ───────────────────────────────────────────────────────────────────

def _cmd_import(args) -> int:
    """Register an existing spectrum file as a commissioning artifact.

    Re-commissioning does not always mean re-measuring. A spectrum already on disk —
    from an earlier session, another rig, or a bench instrument's own export — is just
    as valid an artifact provided its **electrode mode is declared honestly**, which is
    why ``--electrode-mode`` is required here rather than defaulted.

    This is also the only route back for a spectrum that was measured before the mode
    was recorded: if you know the jumper was fitted, re-import it saying so, and the
    derivation will accept it. If you do not know, it stays refused, which is correct.
    """
    from softae.analysis.eis.calibration import ARTIFACT_NOMINAL_UNITS as _UNITS
    from softae.analysis.eis.calibration import electrode_mode_ok
    from softae.analysis.eis_data import EISResult

    src = Path(args.file).expanduser()
    if not src.exists():
        print(f"No such file: {src}", file=sys.stderr)
        return EXIT_FAILED

    unit = _UNITS.get(args.role)
    if unit and args.nominal is None:
        print(f"'{args.role}' needs --nominal (its marked value, in {unit}).",
              file=sys.stderr)
        return EXIT_FAILED

    ok, why = electrode_mode_ok(args.role, args.electrode_mode)
    if not ok:
        # Refused here rather than at derive time: importing a spectrum that can never
        # be used, and only discovering it later, wastes the operator's attention twice.
        print(why, file=sys.stderr)
        return EXIT_FAILED

    # R19: an open-cell blank requires RE tied to CE, and the fact must be recorded.
    # Warned rather than refused, because the state is an operator assertion about the
    # past and refusing would only push it to be asserted untruthfully. The measured
    # difference is not subtle: floating, seven nominally identical channels gave
    # capacitances uncorrelated with their true values (Spearman +0.11) and α spanning
    # 6.1–22.9, while the same channel repeated across sessions moved 56%.
    if args.role == "blank_open" and args.re_connection != "tied_to_ce":
        print(f"  ! R19: an open blank floats the RE unless it is tied to CE. "
              f"Recorded re_connection = '{args.re_connection}'; if the jumper was "
              f"in fact fitted, re-import with --re-connection tied_to_ce.",
              file=sys.stderr)

    try:
        eis = EISResult.load(src)
    except Exception as exc:
        print(f"Could not read {src}: {exc}", file=sys.stderr)
        return EXIT_FAILED

    store, _project = _open_store(args)
    try:
        channel = args.channel if args.channel is not None else int(
            getattr(eis, "channel", 0) or 0)
        eis.channel = channel

        run_id = store.start_run(f"import_{args.role}", mode="commissioning",
                                 annotation=f"imported {src.name}")
        # The second of this module's two `start_run` sites. They are
        # alternatives, not nested — `import` opens its own store and never runs
        # a workflow — so it needs its own finalizer rather than sharing one.
        finalize = run_finalizer(store, run_id)
        try:
            dest = (Path(store.project_dir) / "eis"
                    / f"import_{args.role}_ch{channel}.csv")
            dest.parent.mkdir(parents=True, exist_ok=True)
            eis.save(dest)

            mid = store.record_measurement(
                run_id, eis, role=args.role, fixture_id=args.fixture,
                nominal_value=args.nominal, electrode_mode=args.electrode_mode,
                re_connection=args.re_connection,
            )
            finalize("done")
            print(f"Imported {src.name} as {args.role} on ch{channel} "
                  f"({args.electrode_mode}-electrode), measurement #{mid}")
            print(f"  stored: {dest}")
            print(f"  then:  softae-commission derive --fixture {args.fixture}")
        finally:
            # Idempotent; a no-op once "done" is recorded. A failed `eis.save` or
            # `record_measurement` would otherwise leave the row open forever.
            finalize("error")
    finally:
        store.close()
    return EXIT_OK


# ── Entry point ──────────────────────────────────────────────────────────────

def _default_fixture() -> str:
    """``[eis.fixture] fixture_id``, so the CLI and the engine cannot drift apart.

    Commissioning ``mux16`` while the analysis engine looks for ``default`` would
    produce a calibration nothing ever applies — visible nowhere, because both halves
    behave correctly in isolation. One source of truth removes the possibility.
    """
    try:
        from softae.analysis.eis.settings import eis_settings

        return eis_settings().fixture.fixture_id or "default"
    except Exception:
        return "default"


def build_parser() -> argparse.ArgumentParser:
    default_fixture = _default_fixture()
    p = argparse.ArgumentParser(
        prog="softae-commission",
        description="Acquire and derive the EIS commissioning calibration.",
        epilog="Order by value per hour of bench time: "
               "short blank -> load blank -> reference capacitor.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("status", help="what is calibrated and what to run next")
    st.add_argument("--fixture", default=default_fixture,
                     help=f"fixture id (default: {default_fixture})")
    st.set_defaults(func=_cmd_status)

    run = sub.add_parser("run", help="acquire one commissioning artifact")
    run.add_argument("role", choices=list(COMMISSIONING_ROLES))
    run.add_argument("--channels", default="1", help='e.g. "1, 3-6"')
    run.add_argument("--fixture", default=default_fixture,
                     help=f"fixture id (default: {default_fixture})")
    run.add_argument("--project",
                     help="project directory (default: [data] project_dir)")
    run.add_argument("--nominal", type=float,
                     help="the part's marked value (ohms / farads)")
    run.add_argument("--electrode-mode", choices=("two", "three"),
                     dest="electrode_mode", default=None,
                     help="how the cell is sensed; two-terminal references REQUIRE "
                          "'two' (RE tied to CE). Prompted when omitted.")
    run.add_argument("--yes", "-y", action="store_true",
                     help="skip the 'is the hardware in place?' prompt")
    run.add_argument("--mock", action="store_true")
    run.set_defaults(func=_cmd_run)

    imp = sub.add_parser(
        "import", help="register an existing spectrum file as an artifact")
    imp.add_argument("role", choices=list(COMMISSIONING_ROLES))
    imp.add_argument("--file", required=True, help="path to an EIS spectrum file")
    imp.add_argument("--electrode-mode", choices=("two", "three"),
                     dest="electrode_mode", required=True,
                     help="REQUIRED: declaring it is the point of importing")
    imp.add_argument("--channel", type=int,
                     help="channel (default: whatever the file records)")
    imp.add_argument("--fixture", default=default_fixture,
                     help=f"fixture id (default: {default_fixture})")
    imp.add_argument("--nominal", type=float,
                     help="the part's marked value (ohms / farads)")
    imp.add_argument("--re-connection", dest="re_connection",
                     choices=list(RE_STATES), default="unverified",
                     help="what closed the control loop (R19). 'tied_to_ce' for any "
                          "two-terminal reference and for open blanks")
    imp.add_argument("--project",
                     help="project directory (default: [data] project_dir)")
    imp.set_defaults(func=_cmd_import)

    der = sub.add_parser("derive", help="build the calibration from recorded spectra")
    der.add_argument("--fixture", default=default_fixture,
                     help=f"fixture id (default: {default_fixture})")
    der.add_argument("--project",
                     help="project directory (default: [data] project_dir)")
    der.add_argument("--channels", help="full channel set, to record which are assumed")
    der.add_argument("--representative", type=int,
                     help="channel whose constants unmeasured channels inherit")
    der.add_argument("--declare-electrode-mode", choices=("two", "three"),
                     dest="declare_electrode_mode", default=None,
                     help="assert how spectra with an UNKNOWN mode were sensed, and "
                          "record it. Never overrides an explicitly recorded mode.")
    # Overrides, not inputs: each acquisition's own recorded marking governs by
    # default. These exist for a part mis-entered at acquisition time, and they apply
    # to EVERY acquisition of that role.
    der.add_argument("--nominal-load", type=float,
                     help="OVERRIDE every load blank's recorded marking, ohms")
    der.add_argument("--nominal-cap", type=float,
                     help="OVERRIDE every reference capacitor's marking, farads")
    der.add_argument("--nominal-r", type=float,
                     help="OVERRIDE every reference resistor's marking, ohms")
    der.set_defaults(func=_cmd_derive)

    hist = sub.add_parser("history", help="successive calibrations = drift")
    hist.add_argument("--fixture", default=default_fixture,
                     help=f"fixture id (default: {default_fixture})")
    hist.add_argument("--project",
                      help="project directory (default: [data] project_dir)")
    hist.set_defaults(func=_cmd_history)

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
