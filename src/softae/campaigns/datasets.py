"""Ground-truth dataset + oracle for simulated BO campaigns.

A :class:`GroundTruthDataset` turns a tidy replicate-level frame into:

* a **candidate pool** — one parameter point per distinct condition (``point_id``),
* per-candidate **objective values** under the active transform,
* per-candidate **observation-noise variances** from the active noise model
  (one set for the GP ``alpha`` channel, one for simulating noisy reveals).

An :class:`Oracle` wraps the dataset and "reveals" a (optionally noisy) value
for any pool point — the dataset is the *imperfect ground truth*.

Rail handling (per design): the two σ ≈ 0.1 S/cm points pair with a fitted
impedance of ~100 Ω and a zeroed manually-adjusted impedance, i.e. they look
like a fit/sensor rail rather than a real measurement.  Detection is at the
**replicate (row) level** because a rail's duplicate can be a perfectly normal
measurement.  ``exclude_rails_from_optimum`` (default True) drops rail rows from
the statistics and excludes any rail-containing candidate from the true optimum;
set it False to treat the rails as genuine high-conductivity data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from softae.campaigns.adapters import DatasetAdapter
from softae.campaigns.noise import CellStats, NoiseModel, build_noise_model
from softae.campaigns.objectives import Log10Sigma, ObjectiveTransform
from softae.campaigns.schema import validate_tidy
from softae.errors import CampaignError

# Columns that are never candidate coordinates.
_NON_PARAM_COLUMNS = frozenset(
    {"replicate", "conductivity", "fitted_Z", "adjusted_Z", "fit_residual",
     "point_id", "source",
     # Which thermometer read `temp_C`. Constant within a point_id and so it
     # would otherwise be inferred as a coordinate — a string one, into a GP.
     "temp_source"}
)


@dataclass
class Observation:
    """One revealed measurement from the oracle."""

    point_id: str
    value: float          # objective-space value (possibly noisy)
    variance: float       # observation-noise variance used for the reveal
    is_rail: bool = False
    n_rep: int = 0


def detect_rails(
    df: pd.DataFrame,
    *,
    rail_sigma_ceiling: float | None = 0.05,
    rail_fitted_Z_max: float = 150.0,
) -> pd.Series:
    """Boolean row mask flagging rail-like replicate measurements.

    A row is a rail when its conductivity is at/above ``rail_sigma_ceiling``
    **and** it shows a rail impedance signature: a fitted impedance at/below
    ``rail_fitted_Z_max`` *or* a non-positive manually-adjusted impedance.
    Returns an all-False mask when ``rail_sigma_ceiling`` is None or impedance
    columns are absent.
    """
    mask = pd.Series(False, index=df.index)
    if rail_sigma_ceiling is None:
        return mask
    sigma_hi = df["conductivity"] >= rail_sigma_ceiling
    z_signature = pd.Series(False, index=df.index)
    if "fitted_Z" in df.columns:
        z_signature = z_signature | (df["fitted_Z"] <= rail_fitted_Z_max)
    if "adjusted_Z" in df.columns:
        z_signature = z_signature | (df["adjusted_Z"] <= 0)
    return sigma_hi & z_signature


class GroundTruthDataset:
    """Finite candidate pool + objective + noise, derived from a tidy frame."""

    def __init__(
        self,
        cells: list[CellStats],
        *,
        transform: ObjectiveTransform,
        pool_variance: dict[str, float],
        reveal_variance: dict[str, float],
        param_columns: list[str],
        excluded_optimum_ids: set[str],
    ) -> None:
        self._cells = cells
        self._by_id = {c.point_id: c for c in cells}
        self.transform = transform
        self._pool_variance = pool_variance
        self._reveal_variance = reveal_variance
        self.param_columns = list(param_columns)
        self._excluded_optimum_ids = excluded_optimum_ids

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_adapter(cls, adapter: DatasetAdapter, **kwargs) -> "GroundTruthDataset":
        """Build from any :class:`DatasetAdapter` (calls :meth:`from_tidy`)."""
        return cls.from_tidy(adapter.to_tidy(), **kwargs)

    @classmethod
    def from_tidy(
        cls,
        df: pd.DataFrame,
        *,
        transform: ObjectiveTransform | None = None,
        noise_model: NoiseModel | None = None,
        param_columns: list[str] | None = None,
        rail_sigma_ceiling: float | None = 0.05,
        rail_fitted_Z_max: float = 150.0,
        exclude_rails_from_optimum: bool = True,
        rail_variance: float = 100.0,
        feasible: "Callable[[dict[str, Any]], bool] | None" = None,
    ) -> "GroundTruthDataset":
        """Aggregate replicates → candidates, apply transform + noise model.

        Parameters mirror the campaign config rail/noise knobs.  ``param_columns``
        defaults to every column that is (a) not a reserved measurement column and
        (b) constant within each ``point_id`` — i.e. the candidate coordinates.
        ``feasible`` optionally pre-filters candidates by their parameter dict
        (e.g. a composition simplex / sum-to-1 constraint).
        """
        validate_tidy(df)
        transform = transform or Log10Sigma()
        noise_model = noise_model or build_noise_model()

        df = df.copy()
        rails = detect_rails(
            df, rail_sigma_ceiling=rail_sigma_ceiling, rail_fitted_Z_max=rail_fitted_Z_max
        )
        df["_is_rail"] = rails

        param_columns = param_columns or cls._infer_param_columns(df)

        cells: list[CellStats] = []
        excluded: set[str] = set()
        for pid, group in df.groupby("point_id", sort=False):
            cell, exclude_from_opt = cls._build_cell(
                pid, group, transform, param_columns, exclude_rails_from_optimum
            )
            if cell is None:
                continue  # no usable σ → not a viable candidate
            if feasible is not None and not feasible(cell.params):
                continue  # outside the feasible region (e.g. simplex constraint)
            cells.append(cell)
            if exclude_from_opt:
                excluded.add(pid)

        if not cells:
            raise CampaignError("no viable candidates after aggregation (all σ invalid)")

        # Fit noise model over all cells (pooled-shrinkage needs the full set).
        noise_model.prepare(cells, transform)

        pool_variance: dict[str, float] = {}
        reveal_variance: dict[str, float] = {}
        for c in cells:
            genuine = noise_model.variance(c, transform)
            reveal_variance[c.point_id] = genuine
            # GP-alpha channel distrusts rail-containing cells when excluding.
            if c.is_rail and exclude_rails_from_optimum:
                pool_variance[c.point_id] = max(genuine, rail_variance)
            else:
                pool_variance[c.point_id] = genuine

        return cls(
            cells,
            transform=transform,
            pool_variance=pool_variance,
            reveal_variance=reveal_variance,
            param_columns=param_columns,
            excluded_optimum_ids=excluded,
        )

    @classmethod
    def from_tidy_derived(
        cls,
        df: pd.DataFrame,
        *,
        derived_objective,
        param_columns: list[str] | None = None,
        rail_sigma_ceiling: float | None = 0.05,
        rail_fitted_Z_max: float = 150.0,
        feasible: "Callable[[dict[str, Any]], bool] | None" = None,
    ) -> "GroundTruthDataset":
        """Build candidates whose objective is derived from each σ(T) series.

        Temperature is the *fitting variable*, not a candidate coordinate: rows are
        grouped by composition + non-temperature environment, and each group's
        ``(T, σ)`` points are folded into a scalar by *derived_objective* (e.g. an
        Arrhenius ``Ea`` or σ extrapolated to a target T — see
        :mod:`softae.campaigns.derived`).  Rail rows are excluded before fitting.
        """
        validate_tidy(df)
        if "temp_C" not in df.columns:
            raise CampaignError(
                "temperature-derived objectives require a 'temp_C' column"
            )
        df = df.copy()
        rails = detect_rails(
            df, rail_sigma_ceiling=rail_sigma_ceiling, rail_fitted_Z_max=rail_fitted_Z_max
        )
        df = df[~rails]

        all_params = param_columns or cls._infer_param_columns(df)
        cand_cols = [c for c in all_params if c != "temp_C"]
        if not cand_cols:
            raise CampaignError("no candidate axes remain after removing 'temp_C'")

        cells: list[CellStats] = []
        pool_variance: dict[str, float] = {}
        reveal_variance: dict[str, float] = {}

        for _key, group in df.groupby(cand_cols, sort=False):
            sig = group["conductivity"].to_numpy(dtype=float)
            ok_rows = np.isfinite(sig) & (sig > 0)
            g = group[ok_rows]
            if g.empty:
                continue
            per_T = g.groupby("temp_C")["conductivity"].mean()
            temps = per_T.index.to_numpy(dtype=float)
            sigmas = per_T.to_numpy(dtype=float)

            value, variance, ok = derived_objective.compute(temps, sigmas)
            if not ok or not math.isfinite(value):
                continue
            params = {c: _scalar(group[c].iloc[0]) for c in cand_cols}
            if feasible is not None and not feasible(params):
                continue
            pid = "|".join(
                f"{c}={params[c]:g}" if isinstance(params[c], float) else f"{c}={params[c]}"
                for c in cand_cols
            )
            cells.append(
                CellStats(
                    point_id=pid,
                    params=params,
                    sigma_values=[float(s) for s in sigmas],
                    sigma_mean=float("nan"),
                    sigma_std=float("nan"),
                    n_rep=int(temps.size),
                    y=value,
                )
            )
            pool_variance[pid] = variance
            reveal_variance[pid] = variance

        if not cells:
            raise CampaignError(
                "no viable candidates after temperature-derived fitting "
                "(need enough distinct temperatures per composition)"
            )

        return cls(
            cells,
            transform=Log10Sigma(),  # placeholder; y is already in derived units
            pool_variance=pool_variance,
            reveal_variance=reveal_variance,
            param_columns=cand_cols,
            excluded_optimum_ids=set(),
        )

    @staticmethod
    def _infer_param_columns(df: pd.DataFrame) -> list[str]:
        candidates = [
            c for c in df.columns
            if c not in _NON_PARAM_COLUMNS and not c.startswith("_")
        ]
        # Keep only columns constant within each point_id (true coordinates).
        const_cols = []
        for c in candidates:
            nunique = df.groupby("point_id")[c].nunique(dropna=False)
            if (nunique <= 1).all():
                const_cols.append(c)
        if not const_cols:
            raise CampaignError(
                "could not infer candidate coordinate columns; pass param_columns explicitly"
            )
        return const_cols

    @staticmethod
    def _build_cell(
        pid: str,
        group: pd.DataFrame,
        transform: ObjectiveTransform,
        param_columns: list[str],
        exclude_rails: bool,
    ) -> tuple[CellStats | None, bool]:
        has_rail = bool(group["_is_rail"].any())
        # When excluding, rail rows do not contribute to σ statistics.
        stat_rows = group[~group["_is_rail"]] if exclude_rails else group

        sigma_all = group["conductivity"].to_numpy(dtype=float)
        sigma_stat = stat_rows["conductivity"].to_numpy(dtype=float)
        valid = np.isfinite(sigma_stat) & (sigma_stat > 0)
        sigma_valid = sigma_stat[valid]
        n_rep = int(sigma_valid.size)

        if n_rep == 0:
            return None, True  # nothing usable

        sigma_mean = float(np.mean(sigma_valid))
        sigma_std = float(np.std(sigma_valid, ddof=1)) if n_rep >= 2 else float("nan")

        params = {c: _scalar(group[c].iloc[0]) for c in param_columns}
        fitted_Z = _col_mean(group, "fitted_Z")
        adjusted_Z = _col_mean(group, "adjusted_Z")
        fit_residual = _col_mean(group, "fit_residual")

        cell = CellStats(
            point_id=pid,
            params=params,
            sigma_values=[float(s) for s in sigma_all],
            sigma_mean=sigma_mean,
            sigma_std=sigma_std,
            n_rep=n_rep,
            fitted_Z=fitted_Z,
            adjusted_Z=adjusted_Z,
            fit_residual=fit_residual,
            y=transform.apply_scalar(sigma_mean),
            is_rail=has_rail,
        )
        # Exclude from optimum if it contains a rail and we're excluding rails.
        exclude_from_opt = has_rail and exclude_rails
        if not math.isfinite(cell.y):
            return None, True
        return cell, exclude_from_opt

    # ── Pool accessors ───────────────────────────────────────────────────

    @property
    def cells(self) -> list[CellStats]:
        return list(self._cells)

    @property
    def size(self) -> int:
        return len(self._cells)

    def pool_points(self) -> list[dict[str, Any]]:
        """Candidate parameter dicts (one per ``point_id``), pool order."""
        return [dict(c.params) for c in self._cells]

    def point_ids(self) -> list[str]:
        return [c.point_id for c in self._cells]

    def pool_variance(self) -> dict[str, float]:
        """Per-candidate variance for the GP ``alpha`` channel (rail-penalised)."""
        return dict(self._pool_variance)

    def y_true(self) -> np.ndarray:
        """Objective-space true means aligned to :meth:`pool_points` order."""
        return np.array([c.y for c in self._cells], dtype=float)

    def parameter_space(self) -> dict[str, dict[str, Any]]:
        """A continuous ``BaseOptimizer`` parameter space spanning the pool.

        Bounds are the observed min/max per coordinate; this is what the pooled
        optimizer validates against (it only ever proposes pool points).
        """
        space: dict[str, dict[str, Any]] = {}
        for col in self.param_columns:
            vals = [c.params[col] for c in self._cells]
            lo, hi = min(vals), max(vals)
            if lo == hi:
                hi = lo + 1.0  # degenerate axis → widen so low < high
            space[col] = {"type": "float", "low": float(lo), "high": float(hi)}
        return space

    def true_value(self, point_id: str) -> float:
        """Objective-space true mean for a candidate (used for noise-free regret)."""
        cell = self._by_id.get(point_id)
        if cell is None:
            raise CampaignError(f"unknown point_id: {point_id}")
        return cell.y

    def encoded_pool(self, encode) -> np.ndarray:
        """Encode every pool point with *encode* (pool order) → matrix for prediction."""
        return np.array([encode(c.params) for c in self._cells], dtype=float)

    def point_id_for(self, params: dict[str, Any]) -> str | None:
        """Match a parameter dict to a candidate ``point_id`` (rounded keys)."""
        key = _param_key(params, self.param_columns)
        for c in self._cells:
            if _param_key(c.params, self.param_columns) == key:
                return c.point_id
        return None

    def true_optimum(self, maximize: bool = True) -> tuple[dict[str, Any], float]:
        """Best candidate (params, y) under the given direction, honouring rails."""
        eligible = [
            c for c in self._cells
            if c.point_id not in self._excluded_optimum_ids and math.isfinite(c.y)
        ]
        if not eligible:
            raise CampaignError("no eligible candidates for true_optimum")
        best = max(eligible, key=lambda c: c.y) if maximize else min(eligible, key=lambda c: c.y)
        return dict(best.params), best.y


class Oracle:
    """Reveals (optionally noisy) objective values for pool points."""

    def __init__(self, dataset: GroundTruthDataset, *, noiseless: bool = False) -> None:
        self.dataset = dataset
        self.noiseless = noiseless

    def reveal(self, params: dict[str, Any], rng: np.random.RandomState) -> Observation:
        pid = self.dataset.point_id_for(params)
        if pid is None:
            raise CampaignError(f"point not in dataset pool: {params}")
        cell = self.dataset._by_id[pid]
        var = self.dataset._reveal_variance[pid]
        value = cell.y
        if not self.noiseless and var > 0:
            value = value + rng.normal(0.0, math.sqrt(var))
        return Observation(
            point_id=pid, value=float(value), variance=float(var),
            is_rail=cell.is_rail, n_rep=cell.n_rep,
        )


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _scalar(v: Any) -> Any:
    """Coerce numpy scalars to native Python for clean param dicts."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _col_mean(group: pd.DataFrame, col: str) -> float:
    if col not in group.columns:
        return float("nan")
    arr = group[col].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _param_key(params: dict[str, Any], columns: list[str]) -> tuple:
    """Hashable, float-robust key for matching parameter dicts."""
    key = []
    for c in columns:
        v = params[c]
        if isinstance(v, float):
            key.append(round(v, 9))
        else:
            key.append(v)
    return tuple(key)
