"""One σ route for every GUI surface (P.16 Stage 1).

Six places in the GUI used to compute conductivity by hand as
``z_to_sigma(L, t, w, R1)``. Each therefore chose its own analysis, and flipping
``[eis] engine`` would have moved five of them and missed the sixth. This module is
the on-ramp: every surface now builds the *same* :class:`CellConstant` and reads σ
off a :class:`~softae.analysis.eis.report.SpectrumReport` produced by
:func:`~softae.analysis.eis.engine.analyze_spectrum`, whose engine is chosen by
``[eis] engine`` and by nothing else.

**Stage 1 moves no displayed number.** ``analyze_spectrum`` with the shipped
``engine = "legacy"`` calls :func:`~softae.analysis.circuit_fitting.fit_circuit`
verbatim, and :meth:`CellConstant.sigma` at ``dead_height_cm = 0`` and
``k_config_factor = 1.0`` is the same arithmetic ``z_to_sigma`` ran. (It is the same
arithmetic, not the same *association order* — ``K/R`` where ``K = L/(t·w)`` versus
``L/(R·t·w)`` can differ in the final bit. The gap is ~1e-16 relative and invisible
at every precision the GUI prints.)

.. warning::
   Never thread an ``electrode_configuration`` — or anything resembling one — into
   the cell built here. ``CONFIG_FACTORS["3-electrode"]`` is 2.0, and arming it would
   halve ``K`` and so halve every σ on screen while every self-consistent test still
   passed. The defaults are load-bearing: leave them alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from softae.analysis.eis.geometry import CellConstant, cell_from_legacy_terms


def gui_cell(L: Any, t: Any, w: Any) -> CellConstant | None:
    """The cell constant a GUI surface builds from its own ``(L, t, w)`` in cm.

    ``None`` when any term is missing or non-positive. That propagates to
    ``sigma.mode == "unavailable"`` and then to a blank cell — a missing thickness
    yields *no* conductivity, never one built on a nominal.

    The guard itself is :func:`~softae.analysis.eis.geometry.cell_from_legacy_terms`,
    shared with the web adapter, the temperature sweep and the result router. This
    wrapper stays because it is the name every GUI surface imports and therefore where
    a GUI author reads the module warning above: **pass no ``electrode_configuration``
    through here.** ``cell_from_legacy_terms`` forwards ``**kwargs`` to
    :meth:`CellConstant.from_legacy`, and this call deliberately supplies none —
    arming ``CONFIG_FACTORS["3-electrode"] = 2.0`` would halve every σ on screen while
    every self-consistent test still passed.
    """
    return cell_from_legacy_terms(L, t, w)


def report_sigma(report: Any) -> float:
    """σ in S/cm from a ``SpectrumReport``, or ``nan`` when none may be claimed.

    ``nan`` rather than ``None`` because that is what the fit workers have always
    put in the results tuple for a failed fit; callers that want ``None`` test
    :func:`numpy.isfinite` themselves.
    """
    sigma = getattr(report, "sigma", None)
    if sigma is None or getattr(sigma, "mode", "unavailable") != "value":
        return float("nan")
    try:
        return float(sigma.value)
    except (TypeError, ValueError):
        return float("nan")


def cell_sigma(cell: CellConstant | None, R1: Any) -> float | None:
    """σ from an already-fitted ``R1`` — for the surfaces that hold no spectrum.

    ``_row_sigma`` recomputes on every thickness keystroke, and
    ``_conductivity_from_fit`` is handed a ``FitResult`` whose spectrum is long out
    of scope. Neither may re-run a fit, so they take the σ arithmetic from
    the same :class:`CellConstant` the engine would have used and stop there.
    """
    if cell is None:
        return None
    try:
        R = float(R1)
    except (TypeError, ValueError):
        return None
    if not (R > 0):
        return None
    sigma = cell.sigma(R)
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else None
