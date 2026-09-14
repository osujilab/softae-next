"""Canonical :class:`EISEntry` data container.

A pure (Qt-free, matplotlib-free) dataclass bundling one EIS measurement with
its optional circuit fit and derived conductivity.  Lives in ``softae.analysis``
so both the Qt GUI (``softae.gui``) and the headless web layer (``softae.web``)
can share it without either depending on the other's presentation stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis_data import EISResult

if TYPE_CHECKING:  # annotation only — keeps this module free of the eis engine
    from softae.analysis.eis.report import SpectrumReport


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
    # The analysis that produced ``fit``, kept so a consumer that persists the fit
    # can persist its evidence too — ``DataStore.record_fit(report=...)`` reads the
    # engine label and the gate log from here and from nowhere else, so an entry
    # that drops it writes a verdict nobody can re-derive.
    #
    # ``None`` means *this fit did not come from an analysis run here*: an entry
    # rebuilt from a stored ``fit_results`` row has a fit and no report, and that is
    # an absence to record rather than to paper over. Written only alongside
    # ``fit``, by :func:`~softae.gui.widgets.eis_visualizer_widget.fit_entry`, so
    # the pair can never describe two different analyses.
    report: "SpectrumReport | None" = None


__all__ = ["EISEntry"]
