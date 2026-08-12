"""Shared fixtures/helpers for campaign tests (synthetic tidy frames + files)."""

from __future__ import annotations

import os

import pandas as pd

# Path to the real seed dataset, supplied via the ``SOFTAE_SEED_DATASET``
# environment variable; tests that need it skip when unset or absent.
SEED_DATASET = os.environ.get("SOFTAE_SEED_DATASET", "")


def make_tidy(
    x_values,
    sigma_fn,
    *,
    n_rep: int = 2,
    rh: float = 30.0,
    source: str = "synthetic",
) -> pd.DataFrame:
    """Build a tidy frame over a single composition axis ``x``.

    ``sigma_fn(x, replicate) -> conductivity``.  No impedance columns, so the
    fit-quality noise source contributes nothing.
    """
    rows = []
    for x in x_values:
        for r in range(n_rep):
            rows.append(
                {
                    "x": float(x),
                    "rh_pct": float(rh),
                    "replicate": r,
                    "conductivity": float(sigma_fn(x, r)),
                    "point_id": f"x{x:g}",
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def write_small_aggregated(path) -> dict:
    """Write a tiny 3-block aggregated file and return its layout for assertions.

    Layout: 2 RH x 2 EO x 2 silica x 2 replicates = 16 values per block.
    Flat index 0..15; conductivity = index+1 (so values are distinguishable).
    """
    n = 16
    cond = [float(i + 1) * 1e-6 for i in range(n)]
    fitted = [float(i + 1) * 1e3 for i in range(n)]
    adjusted = [float(i + 1) * 1e3 for i in range(n)]

    def fmt(vals):
        return ", ".join(repr(v) for v in vals)

    text = (
        "Synthetic small aggregated file\n\n"
        f"Conductivities (S/cm): {fmt(cond)}\n\n"
        f"Fitted impedances (Ohm):\n{fmt(fitted)}\n\n"
        f"Manually adjusted impedances (Ohm):\n{fmt(adjusted)}\n"
    )
    path.write_text(text, encoding="utf-8")
    return {
        "rh_levels": [10.0, 30.0],
        "eo_li_levels": [40.0, 20.0],
        "silica_levels": [0.0, 0.10],
        "n_replicates": 2,
        "conductivity": cond,
    }
