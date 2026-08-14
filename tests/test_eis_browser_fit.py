"""Tests for per-spectrum fitting in the EIS Browser (Plan B).

Covers:
  * ``fit_entry`` mutating an entry (fit + geometry + geometry-derived sigma)
  * ``_fitresult_from_fit_row`` reconstruction incl. geometry
  * DataStoreSource populating ``measurement_id`` + ``geometry``
  * _InspectionPane fit controls: prefill + on_fit_saved callback
  * AnalysisTab persistence: a browser fit round-trips into the DataStore
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis_data import EISResult
from softae.gui.widgets.eis_visualizer_widget import (
    DEFAULT_GEOMETRY,
    DataStoreSource,
    EISEntry,
    _InspectionPane,
    _fitresult_from_fit_row,
    fit_entry,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def sample_eis() -> EISResult:
    f = np.logspace(1, 5, 50)
    tau = 2 * np.pi * f * 1e-6
    z_real = 100 + 50 / (1 + tau**2)
    z_imag_neg = np.abs(50 * tau / (1 + tau**2))
    return EISResult.from_arrays(channel=1, f=f, z_real=z_real, z_imag_neg=z_imag_neg)


@pytest.fixture
def sample_fit() -> FitResult:
    return FitResult(
        model_name="simpleSalt",
        parameters=np.array([100.0, 1e-7, 0.7, 500.0, 1e-10]),
        R0=100.0,
        R1=500.0,
        R0_guess=100.0,
        R1_guess=500.0,
        z_indices=[0, 3],
        success=True,
    )


@pytest.fixture
def entry(sample_eis) -> EISEntry:
    return EISEntry(label="Ch01 — run", eis=sample_eis, fit=None, sigma=None)


# ── fit_entry ────────────────────────────────────────────────────────────────


def test_fit_entry_sets_geometry_and_sigma(monkeypatch, entry, sample_fit):
    monkeypatch.setattr(
        "softae.analysis.circuit_fitting.fit_circuit",
        lambda eis, model_name: sample_fit,
    )
    out = fit_entry(entry, "simpleSalt", 0.3, 0.1, 0.25)
    assert out is entry
    assert entry.fit is sample_fit
    assert entry.geometry == (0.3, 0.1, 0.25)
    # sigma = L / (R1 · t · w) = 0.3 / (500 · 0.1 · 0.25)
    assert entry.sigma == pytest.approx(0.3 / (500.0 * 0.1 * 0.25))


def test_fit_entry_failed_fit_yields_none_sigma(monkeypatch, entry):
    failed = FitResult(
        model_name="simpleSalt",
        parameters=np.array([0.0]),
        R0=0.0, R1=0.0, R0_guess=0.0, R1_guess=0.0,
        z_indices=[0, 1], success=False, error_msg="did not converge",
    )
    monkeypatch.setattr(
        "softae.analysis.circuit_fitting.fit_circuit",
        lambda eis, model_name: failed,
    )
    fit_entry(entry, "simpleSalt", 0.2, 0.175, 0.2)
    assert entry.fit is failed
    assert entry.geometry == (0.2, 0.175, 0.2)
    assert entry.sigma is None


@pytest.mark.skipif(
    pytest.importorskip("impedance", reason="impedance backend required") is None,
    reason="impedance backend required",
)
def test_fit_entry_real_backend(entry):
    """End-to-end fit against the real impedance backend."""
    fit_entry(entry, "simpleSalt", 0.2, 0.175, 0.2)
    assert entry.fit is not None
    assert entry.geometry == (0.2, 0.175, 0.2)


# ── _fitresult_from_fit_row ──────────────────────────────────────────────────


def test_fitresult_from_fit_row_roundtrip():
    row = {
        "model_name": "flexSalt",
        "parameters_json": "[100.0, 1e-7, 0.83, 500.0, 2e-10]",
        "R0": 100.0,
        "R1": 500.0,
        "sigma_S_per_cm": 1.2e-4,
        "electrode_L_cm": 0.5,
        "electrode_t_cm": 0.05,
        "electrode_w_cm": 0.4,
        "success": 1,
        "error_msg": "",
    }
    fit, sigma, geom = _fitresult_from_fit_row(row)
    assert fit.model_name == "flexSalt"
    assert fit.R1 == 500.0
    assert sigma == pytest.approx(1.2e-4)
    assert geom == (0.5, 0.05, 0.4)


def test_fitresult_from_fit_row_missing_geometry():
    row = {
        "model_name": "simpleSalt", "parameters_json": "[]",
        "R0": 10.0, "R1": 20.0, "sigma_S_per_cm": None,
        "electrode_L_cm": None, "electrode_t_cm": None, "electrode_w_cm": None,
        "success": 1, "error_msg": "",
    }
    fit, sigma, geom = _fitresult_from_fit_row(row)
    assert geom is None
    assert sigma is None
    # Falls back to [R0, R1] when no parameters_json present.
    assert list(fit.parameters) == [10.0, 20.0]


# ── DataStoreSource linkage ──────────────────────────────────────────────────


def test_datastore_source_carries_measurement_id_and_geometry(tmp_path, sample_eis, sample_fit):
    from softae.core.data_store import DataStore

    eis_file = tmp_path / "sample_eis.txt"
    sample_eis.save(eis_file, study_name="test")
    sample_eis.raw_file_path = str(eis_file)

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)
        store.record_fit(mid, sample_fit, L_cm=0.5, t_cm=0.05, w_cm=0.4)
        entries = DataStoreSource(store).get_entries()

    assert len(entries) == 1
    assert entries[0].measurement_id == mid
    assert entries[0].geometry == (0.5, 0.05, 0.4)


# ── _InspectionPane fit controls ─────────────────────────────────────────────


def test_sync_fit_controls_prefills_from_entry(qapp, sample_eis, sample_fit):
    e = EISEntry(
        label="Ch01 — run", eis=sample_eis, fit=sample_fit, sigma=1e-4,
        geometry=(0.5, 0.05, 0.4),
    )
    e.fit.model_name = "flexSalt"
    pane = _InspectionPane()
    pane.refresh([e])
    pane._list.setCurrentRow(0)  # triggers _show_entry → _sync_fit_controls
    assert pane._combo_model.currentText() == "flexSalt"
    assert pane._spin_L.value() == pytest.approx(0.5)
    assert pane._spin_t.value() == pytest.approx(0.05)
    assert pane._spin_w.value() == pytest.approx(0.4)
    assert pane._btn_fit.isEnabled()


def test_sync_fit_controls_defaults_when_no_geometry(qapp, entry):
    pane = _InspectionPane(default_geometry=DEFAULT_GEOMETRY)
    pane.refresh([entry])
    pane._list.setCurrentRow(0)
    assert pane._spin_L.value() == pytest.approx(DEFAULT_GEOMETRY[0])


def test_fit_done_invokes_on_fit_saved(qapp, entry, sample_fit):
    saved: list = []
    pane = _InspectionPane(on_fit_saved=lambda e, L, t, w: saved.append((e, L, t, w)))
    pane.refresh([entry])
    pane._list.setCurrentRow(0)
    # Simulate a completed worker fit.
    entry.fit = sample_fit
    entry.geometry = (0.3, 0.1, 0.25)
    entry.sigma = 2.4e-2
    pane._on_fit_done(entry, "")
    assert saved == [(entry, 0.3, 0.1, 0.25)]
    assert pane._btn_fit.isEnabled()


def test_fit_done_failure_does_not_save(qapp, entry):
    saved: list = []
    pane = _InspectionPane(on_fit_saved=lambda *a: saved.append(a))
    pane.refresh([entry])
    pane._on_fit_done(None, "boom")
    assert saved == []
    assert "boom" in pane._fit_status.text()


# ── AnalysisTab persistence round-trip ───────────────────────────────────────


def test_browser_fit_persists_to_datastore(qapp, tmp_path, sample_eis, sample_fit):
    from softae.core.data_store import DataStore
    from softae.gui.tabs.tab_analysis import AnalysisTab

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)

        tab = AnalysisTab(MagicMock(), data_store=store)
        e = EISEntry(
            label="Ch01 — run", eis=sample_eis, fit=sample_fit, sigma=None,
            measurement_id=mid,
        )
        tab._on_browser_fit_saved(e, 0.3, 0.1, 0.25)

        fits = store.query_fits(measurement_id=mid)

    assert fits, "expected a persisted fit row"
    latest = fits[-1]
    assert latest["electrode_L_cm"] == pytest.approx(0.3)
    assert latest["electrode_t_cm"] == pytest.approx(0.1)
    assert latest["electrode_w_cm"] == pytest.approx(0.25)
    # sigma = 0.3 / (500 · 0.1 · 0.25)
    assert latest["sigma_S_per_cm"] == pytest.approx(0.3 / (500.0 * 0.1 * 0.25))


def test_browser_fit_persists_the_arc_state_column(qapp, tmp_path, sample_eis,
                                                   sample_fit):
    """A browser-saved row must be the same era of row as a router-written one.

    The entry's fit comes from `fit_entry` -> `analyze_spectrum` ->
    `annotate_arc_closure`, so it carries `.arc_closure`; the columns are read off
    the fit, and `report=arc_provenance(fit)` additionally lines the gate log up
    with what the router writes.
    """
    import json

    from softae.analysis.eis.arc import ArcClosure
    from softae.core.data_store import DataStore
    from softae.gui.tabs.tab_analysis import AnalysisTab

    sample_fit.arc_closure = ArcClosure("open", 20.0, 20.0, -41.5)

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)

        tab = AnalysisTab(MagicMock(), data_store=store)
        e = EISEntry(label="Ch01 — run", eis=sample_eis, fit=sample_fit, sigma=None,
                     measurement_id=mid)
        tab._on_browser_fit_saved(e, 0.3, 0.1, 0.25)

        latest = store.query_fits(measurement_id=mid)[-1]

    assert latest["arc_state"] == "open"
    assert latest["arc_f_low_hz"] == pytest.approx(20.0)
    assert latest["arc_phase_low_deg"] == pytest.approx(-41.5)
    # One era of rows, not two: the shim's JSON matches the router's.
    assert json.loads(latest["gate_log_json"])[0]["gate"] == "arc_closure"


def test_browser_fit_persists_a_fit_without_an_annotation_without_raising(
        qapp, tmp_path, sample_eis, sample_fit):
    # `arc_provenance` returns None for an unannotated object, so the call site is
    # safe on whatever the browser hands over — a hand-built FitResult included.
    from softae.core.data_store import DataStore
    from softae.gui.tabs.tab_analysis import AnalysisTab

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)

        tab = AnalysisTab(MagicMock(), data_store=store)
        e = EISEntry(label="Ch01 — run", eis=sample_eis, fit=sample_fit, sigma=None,
                     measurement_id=mid)
        tab._on_browser_fit_saved(e, 0.3, 0.1, 0.25)

        latest = store.query_fits(measurement_id=mid)[-1]

    assert latest["arc_state"] is None
    assert latest["gate_log_json"] == "[]"


def test_browser_fit_saved_noop_without_store(qapp):
    from softae.gui.tabs.tab_analysis import AnalysisTab

    tab = AnalysisTab(MagicMock(), data_store=None)
    # Must not raise even with a bare entry (no measurement_id).
    tab._on_browser_fit_saved(MagicMock(measurement_id=None, fit=None), 0.2, 0.1, 0.2)
