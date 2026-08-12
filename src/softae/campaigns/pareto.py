"""Pareto / multi-objective utilities.

Provides non-dominated-set computation for trading off several objectives at
once.  Used today to surface the non-dominated *campaign configurations* from a
benchmark grid (e.g. fastest-to-converge vs lowest-regret), and as the building
block for future vector-valued campaign objectives (σ + low Ea + low cost).

Full multi-objective *campaigns* (a vector objective per candidate with an EHVI
acquisition) are a planned extension; this module supplies the Pareto maths they
will reuse.
"""

from __future__ import annotations

import numpy as np


def pareto_mask(values: np.ndarray, maximize: list[bool]) -> np.ndarray:
    """Boolean mask of Pareto-optimal (non-dominated) rows.

    Parameters
    ----------
    values
        ``(n, m)`` array of ``n`` points with ``m`` objectives.
    maximize
        Length-``m`` list; ``True`` to maximize that objective, ``False`` to
        minimize.  A point is dominated if another is at least as good in every
        objective and strictly better in at least one.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2-D array (n_points, n_objectives)")
    n, m = values.shape
    if len(maximize) != m:
        raise ValueError("maximize must have one entry per objective")

    # Convert everything to a minimization problem.
    signs = np.array([-1.0 if mx else 1.0 for mx in maximize])
    v = values * signs

    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # Rows that dominate i: all <= v[i] and any < v[i].
        le = np.all(v <= v[i], axis=1)
        lt = np.any(v < v[i], axis=1)
        dominators = le & lt
        dominators[i] = False
        if np.any(dominators):
            mask[i] = False
    return mask


def pareto_indices(values: np.ndarray, maximize: list[bool]) -> list[int]:
    """Indices of the Pareto-optimal rows (see :func:`pareto_mask`)."""
    return [int(i) for i in np.flatnonzero(pareto_mask(values, maximize))]
