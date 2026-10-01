"""The flag-off golden pin for regime slice 1: pinned inputs, corpus, and the record shape.

Shared by the capture script (``docs/SubAgent docs/evidence_regime_slice1/capture_goldens.py``,
run once against pre-slice source) and ``tests/test_eis_engine.py``'s golden test, so the
inputs that produced the golden and the inputs that check it are one definition.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from tests import eis_regime_synthetic as syn

DATA = Path(__file__).resolve().parent / "data" / "eis_regime"
OUT = DATA / "head_goldens.json"

#: The synthetic part of the corpus: (id, Z). Small on purpose — each gated analysis
#: costs a fit — but it spans both A sub-shapes, R_s, the HF artefact, B and C.
_RS_PICKS = (
    ("inband", 0, 0.7, 1.0, 0.0, 1e5, 0.0, 0),
    ("inband", 1, 0.8, 0.85, 1e4, 1e5, 5e-6, 1),
    ("compressed", 0, 0.7, 1.0, 1e3, 1e4, 0.0, 0),
    ("compressed", 1, 0.6, 0.85, 1e5, 1e5, 20e-6, 1),
)


def synthetic_corpus() -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for i, (shape, k, a, ng, Rs, Rb, L, art) in enumerate(_RS_PICKS):
        Z = syn.regime_a_rs(shape, syn.FC[shape][k], a, ng, Rs, Rb, L, art, seed=500 + i)
        out.append((f"synth_A_{shape}_{i}", syn.RIG_F, Z))
    b = list(syn.step1_regime_b())
    c = list(syn.step1_regime_c())
    out.append(("synth_B_0", syn.RIG_F, b[0][1]))
    out.append(("synth_C_0", syn.RIG_F, c[7][1]))
    return out


def real_corpus() -> list[tuple[str, Any]]:
    from softae.analysis.eis_data import EISResult

    return [(p.stem, EISResult.load(p)) for p in sorted(DATA.glob("*.txt"))]


def golden_inputs(channel: int) -> dict[str, Any]:
    """Every ``analyze_spectrum`` keyword the golden pins (config-independent)."""
    from softae.analysis.eis.engine_support import PregateSettings
    from softae.analysis.eis.envelope import InstrumentEnvelope
    from softae.analysis.eis.fixture import FixtureCorrection
    from softae.analysis.eis.geometry import CellConstant
    from softae.analysis.eis.settings import EISSettings, GateSettings

    return dict(
        cell=CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                          thickness_method="predicted"),
        engine="gated",
        envelope=InstrumentEnvelope(),
        settings=EISSettings(engine="gated", gates=GateSettings(enabled=False)),
        correction=FixtureCorrection(mode="series", channel=channel, fixture_id="golden",
                                     R_short_ohm=6.5, L_lead_H=2.5e-6),
        pregate=PregateSettings(budget_cap=True),
    )


def _enc(v: Any) -> Any:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v).hex()
    return repr(v)


def record(report: Any, sigma_fields: list[str] | None = None) -> dict[str, Any]:
    """The pinned shape of one report. ``sigma_fields`` restricts to the HEAD field set."""
    s = report.sigma
    names = sigma_fields or [f.name for f in dataclasses.fields(s)]
    fit = report.fit
    return dict(
        sigma={n: _enc(getattr(s, n)) for n in names},
        fit_R0=_enc(getattr(fit, "R0", None)), fit_R1=_enc(getattr(fit, "R1", None)),
        fit_success=_enc(getattr(fit, "success", None)),
        fitter=report.fitter,
        verdict=str(getattr(report.quality, "verdict", None)),
        issues=list(getattr(report.quality, "issues", []) or []),
        n_dropped=int(report.n_dropped),
        gate_summary=report.gate_summary(),
    )


def run_one(eis: Any, channel: int, **extra) -> Any:
    from softae.analysis.eis.engine import analyze_spectrum

    return analyze_spectrum(eis, **golden_inputs(channel), **extra)


def _pinned_config():
    """Replace ``loader.load`` with ``{}`` for the duration; returns a restore callable."""
    from softae.config import loader

    orig = loader.load
    loader.load = lambda *a, **k: {}
    return lambda: setattr(loader, "load", orig)


def corpus() -> list[tuple[str, Any, int]]:
    items = [(name, e, int(getattr(e, "channel", 15) or 15)) for name, e in real_corpus()]
    items += [(name, syn.as_eis(f, Z), 15) for name, f, Z in synthetic_corpus()]
    return items


def capture_all() -> dict[str, Any]:
    restore = _pinned_config()
    try:
        return {name: record(run_one(e, ch)) for name, e, ch in corpus()}
    finally:
        restore()
