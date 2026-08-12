"""Tidy long-form schema shared by every dataset adapter.

A *tidy* campaign table holds **one row per replicate measurement**.  Every
:class:`~softae.campaigns.adapters.DatasetAdapter` — whether it parses a
hand-aggregated text file or reads a live ``DataStore`` run — emits this same
schema, so the rest of the campaign engine (aggregation, oracle, optimizer)
never sees the original file format.

Composition axes are intentionally *not* fixed: ``COMPOSITION_COLUMNS`` lists
the columns the seed dataset uses, but extra composition columns (e.g. a third
component for a ternary system) are allowed.  :func:`validate_tidy` only
enforces the columns that the engine strictly requires.
"""

from __future__ import annotations

import pandas as pd

from softae.errors import CampaignError

# Columns the engine strictly requires from every adapter.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "point_id",        # candidate identity (one value per distinct condition)
    "replicate",       # 0-based replicate index within a candidate
    "conductivity",    # σ (S/cm); the raw measured objective
    "source",          # adapter tag, for provenance / multi-dataset pooling
)

# Environmental / control columns the seed dataset provides.  Optional in
# general (a dataset may omit RH or temperature), but recognised by name so
# adapters and objective transforms agree on units.
ENV_COLUMNS: tuple[str, ...] = ("rh_pct", "temp_C")

# Composition columns used by the seed PEO/LiCl/silica dataset.  Other datasets
# may use different (or more) composition columns; these are the canonical names
# for this one.
COMPOSITION_COLUMNS: tuple[str, ...] = ("eo_li_ratio", "silica_vol_frac")

# EIS-derived columns that feed the fit-quality noise source.  Optional: a
# dataset without impedance fits simply omits them and the fit-quality noise
# source contributes nothing.
EIS_COLUMNS: tuple[str, ...] = ("fitted_Z", "adjusted_Z", "fit_residual")

# Full canonical column order for the seed dataset, for readable output.
TIDY_COLUMNS: tuple[str, ...] = (
    *COMPOSITION_COLUMNS,
    *ENV_COLUMNS,
    "replicate",
    "conductivity",
    "fitted_Z",
    "adjusted_Z",
    "point_id",
    "source",
)


def validate_tidy(df: pd.DataFrame) -> None:
    """Validate that *df* satisfies the tidy schema contract.

    Checks only the strictly required columns and their basic dtypes; extra
    columns are always permitted so the schema is forward-compatible.

    Raises
    ------
    CampaignError
        If a required column is missing, the frame is empty, ``conductivity``
        is non-numeric, or ``replicate`` is not integer-like.
    """
    if not isinstance(df, pd.DataFrame):
        raise CampaignError(f"tidy data must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise CampaignError("tidy data is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CampaignError(
            f"tidy data missing required column(s): {missing}; "
            f"present columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["conductivity"]):
        raise CampaignError("'conductivity' column must be numeric")

    # replicate must be integer-like (allow float that holds whole numbers,
    # since CSV/round-trips often widen ints to float).
    rep = df["replicate"]
    if not pd.api.types.is_numeric_dtype(rep):
        raise CampaignError("'replicate' column must be numeric (integer-like)")
    if not (rep.dropna() % 1 == 0).all():
        raise CampaignError("'replicate' column must contain whole-number indices")
