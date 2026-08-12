"""Canonical :class:`EISEntry` data container.

A pure (Qt-free, matplotlib-free) dataclass bundling one EIS measurement with
its optional circuit fit and derived conductivity.  Lives in ``softae.analysis``
so both the Qt GUI (``softae.gui``) and the headless web layer (``softae.web``)
can share it without either depending on the other's presentation stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis_data import EISResult


@dataclass
class EISEntry:
    """One visualisable EIS measurement with optional fit and conductivity."""

    label: str
    eis: EISResult
    fit: FitResult | None
    sigma: float | None
    run_id: str | None = None
    # DataStore linkage — needed to persist a browser-initiated re-fit. ``None``
    # for entries not backed by a DataStore measurement (e.g. loose files).
    measurement_id: int | None = None
    # (L, t, w) in cm used for this entry's sigma. May differ per sample; ``None``
    # until a fit with a known geometry is attached.
    geometry: tuple[float, float, float] | None = None


__all__ = ["EISEntry"]
