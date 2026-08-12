"""Prepped runner for a real (or mock) cast-only drop-cast sweep.

SAFETY-FIRST DEFAULTS: with no flags this does a **mock dry run** — it builds
the plan, runs the preflight safety gate, and prints both, executing no motion.

To actually move hardware you must opt in explicitly:

    # inspect the plan + safety report (no hardware, no motion) — the default:
    python examples/dropcast_real_run.py

    # simulate the full sweep on mock drivers:
    python examples/dropcast_real_run.py --mock --execute

    # drive REAL hardware (requires the pumps/stage connected; prompts y/N):
    python examples/dropcast_real_run.py --real --execute

Flags: --wells 21-24 | --pumps 0,1,2 | --vol 0.1 | --rate 1000 | --eis |
--fast (collapse dwells) | --yes (skip the confirmation prompt).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from softae.core.dropcast import (
    DropcastFormulation,
    DropcastPreflightError,
    PreflightReport,
    run_dropcast_sweep,
)
from softae.core.hardware_safety import ARM_ENV_VAR, HardwareNotArmedError, hardware_is_armed


def _parse_wells(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return tuple(out)


def _live(evt: dict) -> None:
    kind = evt.get("type")
    if kind == "preflight":
        print(f"\nPreflight: {'PASS' if evt['ok'] else 'FAIL'}")
        for e in evt["errors"]:
            print(f"  [ERROR] {e}")
        for w in evt["warnings"]:
            print(f"  [WARN]  {w}")
        for i in evt["info"]:
            print(f"  [info]  {i}")
    elif kind in ("dry_run", "sweep_started"):
        f = evt["formulation"]
        header = "DRY RUN — plan only (no motion)" if kind == "dry_run" else "EXECUTING sweep"
        print(f"\n{header}: wells {evt['channels']} on {evt['pcb']}")
        print(f"  formulation: pumps {f['ids']} x {f['vols']} uL @ {f['disp_rate']} uL/min")
        print(f"  total steps: {evt['total_steps']}")
        for ch, xy in evt["electrode_xy"].items():
            print(f"    well {ch}: (x={xy[0]:.2f}, y={xy[1]:.2f})")
    elif kind == "step_start":
        ch = f" [well {evt['channel']}]" if evt.get("channel") else ""
        print(f"  -> [{evt['index'] + 1}/{evt['total']}] {evt['step']} "
              f"({evt['instrument']}.{evt['method']}){ch}")
    elif kind == "aborted":
        print(f"\nAborted: {evt['reason']}")
    elif kind == "sweep_finished":
        print(f"\nSweep complete: {evt['steps_run']} steps.")


def _confirm(report: PreflightReport) -> bool:
    if report.warnings:
        print("\n!! Preflight raised warnings (above). Review before proceeding.")
    ans = input("\nProceed to drive REAL hardware? [y/N] ").strip().lower()
    return ans in ("y", "yes")


async def _main(args: argparse.Namespace) -> int:
    from softae.drivers.factory import create_manager

    wells = _parse_wells(args.wells)
    pumps = tuple(int(p) for p in args.pumps.split(","))
    formulation = DropcastFormulation(
        ids=pumps,
        vols=tuple(args.vol for _ in pumps),
        deadvols=tuple(0.0 for _ in pumps),
        disp_rate=args.rate,
        time_scale=0.0 if args.fast else 1.0,
    )

    # Construct drivers WITHOUT connecting. Preflight + a dry run need no live
    # hardware, so a real dry run can preview the plan on any machine and never
    # opens a port or energizes a pump. We connect only to actually execute.
    manager = create_manager(mock=not args.real)

    # Hard gate BEFORE opening any port: a real execute requires deliberate
    # arming (env var) and an interactive terminal to confirm. This is what
    # stops a stray or agent-issued command from ever energizing hardware.
    if args.execute and args.real:
        if not hardware_is_armed():
            print(f"\nRefusing to drive REAL hardware: the safety interlock is not "
                  f"armed.\nSet {ARM_ENV_VAR}=1 in your shell to deliberately arm, e.g.:\n"
                  f"    {ARM_ENV_VAR}=1 python examples/dropcast_real_run.py --real --execute\n"
                  f"(No hardware was touched.)", file=sys.stderr)
            return 5
        if not sys.stdin.isatty() and not args.yes:
            print("\nRefusing: real execute needs an interactive terminal to confirm "
                  "(or an explicit --yes). No hardware was touched.", file=sys.stderr)
            return 5

    connected = False
    if args.execute:
        try:
            await manager.connect_all()
            connected = True
        except Exception as exc:
            print(f"\nCould not connect instruments ({'real' if args.real else 'mock'}): "
                  f"{exc}\nIs the hardware attached? Use the default (mock dry-run) to "
                  f"preview without hardware.", file=sys.stderr)
            return 4

    confirm = None
    if args.execute and args.real and not args.yes:
        confirm = _confirm

    try:
        result = await run_dropcast_sweep(
            wells, formulation,
            manager=manager,
            pcb_name=args.pcb,
            measure_eis=args.eis,
            name="dropcast_castonly",
            on_event=_live,
            dry_run=not args.execute,
            confirm_fn=confirm,
        )
    except DropcastPreflightError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
    except HardwareNotArmedError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 5
    finally:
        if connected:
            await manager.disconnect_all()

    mode = "executed" if result.executed else "dry run (no motion)"
    print(f"\nMode: {mode}  |  wells: {', '.join(map(str, result.channels))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cast-only drop-cast sweep runner.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="Use real hardware.")
    mode.add_argument("--mock", action="store_true",
                      help="Use mock drivers (the default; explicit for clarity).")
    p.add_argument("--execute", action="store_true",
                   help="Actually run the sweep (else dry-run: plan + preflight only).")
    p.add_argument("--yes", action="store_true", help="Skip the real-hardware confirmation prompt.")
    p.add_argument("--wells", default="21-24", help="Wells, e.g. '21-24' or '21,22,23,24'.")
    p.add_argument("--pumps", default="0,1,2", help="Pump IDs, e.g. '0,1,2'.")
    p.add_argument("--vol", type=float, default=0.1, help="µL commanded per pump per well.")
    p.add_argument("--rate", type=float, default=1000.0, help="Dispense rate µL/min.")
    p.add_argument("--pcb", default="SoftAE_EIS_4Stripe", help="PCB layout name.")
    p.add_argument("--eis", action="store_true", help="Measure EIS after each cast.")
    p.add_argument("--fast", action="store_true", help="Collapse routine dwells (demo speed).")
    args = p.parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
