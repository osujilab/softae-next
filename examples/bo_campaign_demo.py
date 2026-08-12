"""Demo: run a simulated BO campaign on the PEO/LiCl/silica conductivity dataset.

Treats the aggregated conductivity file as an imperfect ground-truth oracle, runs
a pool-based Bayesian-optimization campaign that "uncovers" one formulation at a
time, and plots the simple-regret convergence curve.

Usage
-----
    python examples/bo_campaign_demo.py [PATH_TO_DATASET]

The dataset is taken from the first argument, or from the ``SOFTAE_SEED_DATASET``
environment variable when no argument is given.

Writes ``bo_campaign_regret.png`` next to the dataset (or CWD) and prints a
summary including steps-to-convergence and the effect of the rail toggle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from softae.campaigns import BOCampaignConfig, build_campaign, build_dataset

DATASET_ENV_VAR = "SOFTAE_SEED_DATASET"


def main(argv: list[str]) -> int:
    dataset_path = argv[1] if len(argv) > 1 else os.environ.get(DATASET_ENV_VAR, "")
    if not dataset_path:
        print(
            "no dataset given: pass the path to an aggregated conductivity dataset "
            f"as the first argument, or set {DATASET_ENV_VAR} to point at one."
        )
        return 1
    if not Path(dataset_path).exists():
        print(f"dataset not found: {dataset_path}")
        return 1

    base = dict(
        dataset_adapter="aggregated_txt",
        dataset_path=dataset_path,
        objective_direction="maximize",
        transform="log10_sigma",
        n_initial=5,
        seed=1,
        noiseless_oracle=True,  # converge on the underlying truth, not the noise
    )

    # Compare two acquisition strategies on the same pool.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for acq in ("ucb", "ei"):
        cfg = BOCampaignConfig(acquisition=acq, **base)
        runner = build_campaign(cfg)
        result = runner.run()
        curve = result.regret_curve()
        ax.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, label=f"{acq.upper()}")
        print(
            f"[{acq.upper():3}] pool={runner.pool_size}  "
            f"converged={result.converged}  "
            f"steps_to_tol={result.steps_to_tolerance}  "
            f"best_log10_sigma={result.best_value:.3f}"
        )

    # Show the rail toggle's effect on the declared ground-truth optimum.
    ds_excl = build_dataset(BOCampaignConfig(exclude_rails_from_optimum=True, **base))
    ds_incl = build_dataset(BOCampaignConfig(exclude_rails_from_optimum=False, **base))
    _, y_excl = ds_excl.true_optimum(maximize=True)
    _, y_incl = ds_incl.true_optimum(maximize=True)
    print(f"\ntrue optimum log10(sigma): exclude_rails={y_excl:.3f}  "
          f"include_rails={y_incl:.3f}  (rails sit at ~ -1.0)")

    ax.set_xlabel("step (formulations uncovered)")
    ax.set_ylabel("simple regret  |  log10(sigma) gap to optimum")
    ax.set_title("Simulated BO campaign — convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(dataset_path).with_name("bo_campaign_regret.png")
    try:
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nwrote regret plot: {out}")
    except OSError:
        out = Path.cwd() / "bo_campaign_regret.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nwrote regret plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
