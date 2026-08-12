"""Benchmark harness: run many campaigns over a grid and aggregate.

Sweeps the Cartesian product of ``{seed × acquisition × backend × transform ×
direction}``, running one campaign per cell, and returns a tidy results frame
plus an aggregation grouped by configuration.  Headless-first: no GUI import.

The immutable :class:`~softae.campaigns.datasets.GroundTruthDataset` depends only
on the transform + noise + rail settings, so it is built **once per transform**
and shared across all seeds/acquisitions/backends — the expensive parse +
aggregation is not repeated per cell.

A cell whose backend is an uninstalled optional package (or that otherwise
errors) is recorded with ``error`` set rather than aborting the whole grid.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from softae.campaigns.config import BOCampaignConfig
from softae.campaigns.datasets import GroundTruthDataset
from softae.campaigns.runner import CampaignResult, build_campaign, build_dataset
from softae.errors import CampaignError


def _summary_row(config: BOCampaignConfig, result: CampaignResult) -> dict[str, Any]:
    curve = result.regret_curve()
    final_step = result.steps[-1] if result.steps else {}
    return {
        "seed": config.seed,
        "acquisition": config.acquisition,
        "backend": config.backend,
        "transform": config.transform,
        "direction": config.objective_direction,
        "noise_model": config.noise_model,
        "noise_channel": config.noise_channel,
        "converged": result.converged,
        "steps_to_tolerance": result.steps_to_tolerance,
        "n_steps": result.n_steps,
        "best_value": result.best_value,
        "true_optimum_value": result.true_optimum_value,
        "final_simple_regret": curve[-1] if curve else float("nan"),
        "final_cumulative_regret": final_step.get("cumulative_regret", float("nan")),
        "regret_auc": float(np.sum(curve)) if curve else float("nan"),
        "final_rmse": final_step.get("surrogate_rmse", float("nan")),
        "error": "",
    }


@dataclass
class BenchmarkResult:
    """Per-campaign summary rows for a benchmark grid."""

    rows: pd.DataFrame
    grid: dict[str, list[Any]] = field(default_factory=dict)

    GROUP_KEYS = ["acquisition", "backend", "transform", "direction"]

    def aggregate(self) -> pd.DataFrame:
        """Aggregate steps-to-convergence and regret per configuration.

        Steps-to-convergence stats are computed over *converged* campaigns only;
        ``frac_converged`` reports the rest.  Returns one row per
        ``(acquisition, backend, transform, direction)``.
        """
        ok = self.rows[self.rows["error"] == ""]
        if ok.empty:
            return pd.DataFrame(columns=self.GROUP_KEYS)

        records = []
        for keys, grp in ok.groupby(self.GROUP_KEYS, sort=False):
            conv = grp[grp["converged"]]
            stt = conv["steps_to_tolerance"].dropna()
            rec = dict(zip(self.GROUP_KEYS, keys if isinstance(keys, tuple) else (keys,)))
            rec.update(
                {
                    "n_campaigns": len(grp),
                    "frac_converged": float(grp["converged"].mean()),
                    "stt_mean": float(stt.mean()) if len(stt) else float("nan"),
                    "stt_median": float(stt.median()) if len(stt) else float("nan"),
                    "stt_q25": float(stt.quantile(0.25)) if len(stt) else float("nan"),
                    "stt_q75": float(stt.quantile(0.75)) if len(stt) else float("nan"),
                    "final_regret_mean": float(grp["final_simple_regret"].mean()),
                    "regret_auc_mean": float(grp["regret_auc"].mean()),
                }
            )
            records.append(rec)
        return pd.DataFrame(records)

    def pareto_configs(self) -> pd.DataFrame:
        """Non-dominated configs on the (speed, quality) trade-off.

        Treats each aggregated configuration as a point in
        (steps-to-convergence ↓, final-regret ↓) space and returns the
        Pareto-optimal subset — the configs you can't beat on speed without
        losing quality (or vice versa).
        """
        from softae.campaigns.pareto import pareto_mask

        agg = self.aggregate()
        if agg.empty:
            return agg
        usable = agg.dropna(subset=["stt_mean", "final_regret_mean"])
        if usable.empty:
            return usable
        values = usable[["stt_mean", "final_regret_mean"]].to_numpy()
        mask = pareto_mask(values, maximize=[False, False])
        return usable.loc[mask].reset_index(drop=True)


def run_grid(
    base_config: BOCampaignConfig,
    *,
    seeds: list[int],
    acquisitions: list[str] | None = None,
    backends: list[str] | None = None,
    transforms: list[str] | None = None,
    directions: list[str] | None = None,
    dataset: GroundTruthDataset | None = None,
    on_campaign: Callable[[BOCampaignConfig, CampaignResult], None] | None = None,
) -> BenchmarkResult:
    """Run the campaign grid and return a :class:`BenchmarkResult`.

    Any axis left ``None`` collapses to the corresponding value in *base_config*.
    A pre-built *dataset* may be supplied to skip parsing/aggregation; it pins the
    transform, so the ``transforms`` axis must then be ≤ 1 value.
    """
    acquisitions = acquisitions or [base_config.acquisition]
    backends = backends or [base_config.backend]
    transforms = transforms or [base_config.transform]
    directions = directions or [base_config.objective_direction]

    # Build (and cache) one dataset per transform — the only axis that changes
    # the dataset's structure here.  A supplied dataset is reused for all cells.
    if dataset is not None:
        if len(transforms) > 1:
            raise CampaignError(
                "a pre-built dataset pins the transform; pass at most one transform"
            )
        dataset_cache: dict[str, GroundTruthDataset] = {tf: dataset for tf in transforms}
    else:
        dataset_cache = {}
        for tf in transforms:
            cfg = dataclasses.replace(base_config, transform=tf)
            dataset_cache[tf] = build_dataset(cfg)

    rows: list[dict[str, Any]] = []
    for seed, acq, backend, tf, direction in itertools.product(
        seeds, acquisitions, backends, transforms, directions
    ):
        cfg = dataclasses.replace(
            base_config,
            seed=seed,
            acquisition=acq,
            backend=backend,
            transform=tf,
            objective_direction=direction,
        )
        try:
            runner = build_campaign(cfg, dataset=dataset_cache[tf])
            result = runner.run()
            row = _summary_row(cfg, result)
            if on_campaign is not None:
                on_campaign(cfg, result)
        except Exception as exc:  # uninstalled backend, bad combo, etc.
            row = {
                "seed": seed, "acquisition": acq, "backend": backend,
                "transform": tf, "direction": direction,
                "noise_model": cfg.noise_model, "noise_channel": cfg.noise_channel,
                "converged": False, "steps_to_tolerance": None, "n_steps": 0,
                "best_value": float("nan"), "true_optimum_value": float("nan"),
                "final_simple_regret": float("nan"),
                "final_cumulative_regret": float("nan"),
                "regret_auc": float("nan"), "final_rmse": float("nan"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)

    grid = {
        "seeds": seeds, "acquisitions": acquisitions, "backends": backends,
        "transforms": transforms, "directions": directions,
    }
    return BenchmarkResult(rows=pd.DataFrame(rows), grid=grid)
