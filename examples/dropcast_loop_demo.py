"""Live demo: legacy-derived drop-cast loop over wells 21-24.

Drives ONLY the liquid-handling loop (no optimizer, no objective): prime the
lines, then sequentially drop-cast a fixed formulation into wells 21, 22, 23,
24 on the 32-channel SoftAE_EIS_4Stripe board — one well per iteration — moving
the stage to each electrode via the ported ``single_drop_simul`` routine.

Formulation for this run: 3 pumps (0/1/2), 0.1 µL each, 1000 µL/min, no EIS,
timing collapsed so the whole sweep runs in seconds on mock hardware.

Usage
-----
    python examples/dropcast_loop_demo.py
"""

from __future__ import annotations

import asyncio

from softae.core.dropcast import DropcastFormulation, run_dropcast_sweep

CHANNELS = (21, 22, 23, 24)


def _live(evt: dict) -> None:
    kind = evt.get("type")
    if kind == "sweep_started":
        f = evt["formulation"]
        print(f"Drop-cast sweep on wells {evt['channels']}  ({evt['pcb']})")
        print(f"  formulation: pumps {f['ids']} x {f['vols']} uL @ {f['disp_rate']} uL/min")
        print("  electrode positions (mm):")
        for ch, xy in evt["electrode_xy"].items():
            print(f"    well {ch}: (x={xy[0]:.2f}, y={xy[1]:.2f})")
        print()
    elif kind == "step_start":
        ch = f" [well {evt['channel']}]" if evt.get("channel") else ""
        print(f"  -> [{evt['index'] + 1}/{evt['total']}] {evt['step']}"
              f"  ({evt['instrument']}.{evt['method']}){ch}")
    elif kind == "step_done":
        pass
    elif kind == "sweep_finished":
        print(f"\nSweep complete: {evt['steps_run']} steps executed.")
        disp = evt["dispensed"]
        if disp:
            pretty = ", ".join(f"pump {k}: {v:.3f} uL" for k, v in sorted(disp.items()))
            print(f"Total dispensed (mock hardware counters): {pretty}")


async def main() -> None:
    formulation = DropcastFormulation(
        ids=(0, 1, 2),
        vols=(0.1, 0.1, 0.1),
        deadvols=(0.0, 0.0, 0.0),
        disp_rate=1000.0,
        time_scale=0.0,  # collapse dwells for a fast, watchable demo
    )

    result = await run_dropcast_sweep(
        CHANNELS,
        formulation,
        pcb_name="SoftAE_EIS_4Stripe",
        measure_eis=False,
        name="dropcast_21to24",
        on_event=_live,
    )

    print(f"\nWells cast: {', '.join(map(str, result.channels))}")
    print(f"Workflow  : {result.workflow_name}")


if __name__ == "__main__":
    asyncio.run(main())
