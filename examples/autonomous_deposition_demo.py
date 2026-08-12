"""Demo: agentic autonomous deposition campaign on channels 21-24.

Drives the closed loop headlessly — no GUI — over mock hardware:

    suggest (optimizer) -> execute (composite deposition + EIS on ch 21-24)
      -> analyze (objective) -> tell -> repeat until budget/convergence

Each trial dispenses the suggested two-pump formulation onto electrodes
21, 22, 23, 24 (the 32-channel SoftAE_EIS_4Stripe board), moving the stage to
each electrode via the ``liquid_handler`` composite routine, then measures EIS.
The electrode positions are resolved from PCB geometry at wiring time — never
encoded in the recipe.

The objective here is synthetic (a smooth peak at a known target formulation)
so convergence is observable and deterministic, while the *execution* is the
real workflow through the real WorkflowExecutor.  Swap in
``eis_impedance_objective`` for the EIS-derived metric.

Usage
-----
    python examples/autonomous_deposition_demo.py
"""

from __future__ import annotations

import asyncio

from softae.core.autonomous_wiring import (
    CampaignSpec,
    composition_target_objective,
    run_autonomous_campaign,
)


def _print_event(evt: dict) -> None:
    kind = evt.get("type")
    if kind == "run_started":
        print(f"* run started: {evt['run_id']}")
    elif kind == "suggestion":
        p = evt["params"]
        vols = ", ".join(f"{k}={v:.2f}" for k, v in p.items())
        print(f"  [{evt['iteration']:>2}] suggest  {vols}")
    elif kind == "result":
        print(f"  [{evt['iteration']:>2}] objective = {evt['objective']:.4f}")
    elif kind == "converged":
        print(f"* converged after {evt['iteration']} trials")
    elif kind == "run_finished":
        print("* run finished")


async def main() -> None:
    spec = CampaignSpec(
        name="binary_dropcast_21to24",
        channels=(21, 22, 23, 24),
        pcb_name="SoftAE_EIS_4Stripe",  # 32-ch board so 21-24 are valid
        parameter_space={
            "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
            "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
        },
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        optimizer="bayesian",
        objective="maximize",
        budget=12,
        auto_approve=True,
        time_scale=0.0,  # collapse routine dwells so the demo runs fast
        seed=7,
    )

    # Known-optimum synthetic objective for an observable convergence trace.
    objective = composition_target_objective({"vol_p0": 22.0, "vol_p1": 12.0})

    print(f"Targets: channels {', '.join(map(str, spec.channels))}  "
          f"(pico routing per channel)\n")

    result = await run_autonomous_campaign(
        spec, objective_extractor=objective, on_event=_print_event
    )

    print("\n-- Result --")
    print(f"trials       : {result.n_trials}")
    print(f"final state  : {result.final_state}")
    if result.best_params:
        bp = ", ".join(f"{k}={v:.2f}" for k, v in result.best_params.items())
        print(f"best params  : {bp}")
        print(f"best objective: {result.best_objective:.4f}")
    print(f"run_id       : {result.run_id}")


if __name__ == "__main__":
    asyncio.run(main())
