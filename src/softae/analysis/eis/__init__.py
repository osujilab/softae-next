"""Gated EIS analysis — the physics layer specified in ``docs/EIS_GATE_FRAMEWORK_new.md``.

This package sits *on top of* the stable flat modules (:mod:`softae.analysis.eis_data`,
:mod:`softae.analysis.circuit_fitting`, :mod:`softae.analysis.quality`) and never the
other way round. Those three are what the rig has always run and they stay unedited
except for additive, defaulted fields; everything new lives here.

The whole package is reachable through one function::

    from softae.analysis.eis import analyze_spectrum
    report = analyze_spectrum(eis_result, cell=cell)

which returns a :class:`~softae.analysis.eis.report.SpectrumReport` whatever engine
ran. ``[eis] engine`` selects between them and ships as ``legacy``.
"""

from softae.analysis.eis.admittance import (
    apparent_capacitance,
    conductance,
    log_slope,
    loss_tangent,
    model_free_r_bulk,
    to_admittance,
)
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.envelope import (
    InstrumentEnvelope,
    instrument_envelope,
    recommend_preset,
)
from softae.analysis.eis.gates import (
    BLOCK_POINT,
    BLOCK_SESSION,
    BLOCK_SPECTRUM,
    FLAG,
    FRONT1_GATES,
    TOPOLOGY_TRIAD,
    GateResult,
    run_gates,
)
from softae.analysis.eis.geometry import (
    CellConstant,
    cell_constant_for_sample,
    resolve_thickness_cm,
)
from softae.analysis.eis.geometry_series import (
    GeometrySeriesFit,
    SeriesMember,
    fit_geometry_series,
)
from softae.analysis.eis.models import EIS_CIRCUITS, CircuitModel, roles_for
from softae.analysis.eis.policy import (
    RE_CLOSED_LOOP,
    RE_IONIC_CONTACT,
    RE_STATES,
    build_context,
    reduce_gates,
)
from softae.analysis.eis.report import SigmaReport, SpectrumReport
from softae.analysis.eis.settings import EISSettings, GateSettings, eis_settings

__all__ = [
    "BLOCK_POINT",
    "BLOCK_SESSION",
    "BLOCK_SPECTRUM",
    "EIS_CIRCUITS",
    "EISSettings",
    "FLAG",
    "FRONT1_GATES",
    "RE_CLOSED_LOOP",
    "RE_IONIC_CONTACT",
    "RE_STATES",
    "TOPOLOGY_TRIAD",
    "CellConstant",
    "CircuitModel",
    "GateResult",
    "GateSettings",
    "GeometrySeriesFit",
    "InstrumentEnvelope",
    "SeriesMember",
    "SigmaReport",
    "SpectrumReport",
    "analyze_spectrum",
    "apparent_capacitance",
    "build_context",
    "cell_constant_for_sample",
    "recommend_preset",
    "conductance",
    "eis_settings",
    "fit_geometry_series",
    "instrument_envelope",
    "log_slope",
    "loss_tangent",
    "model_free_r_bulk",
    "reduce_gates",
    "resolve_thickness_cm",
    "roles_for",
    "run_gates",
    "to_admittance",
]
