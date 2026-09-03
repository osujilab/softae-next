"""The two GUI ``record_fit`` call sites must land **auditable** rows.

`[e33]` §2 measured the defect these tests exist to prevent: in the live store the
verdict-bearing rows and the evidence-bearing rows are **disjoint sets** — 126 of 126
rows carrying a ``gate_verdict`` have an empty ``gate_log_json`` and no spectrum file,
and not one row in 3619 has both. A stored verdict that nobody can re-derive is worse
than no verdict.

**These tests assert the consequence, not the argument.** "``report=`` was passed" is
worth almost nothing — the row is what a future auditor reads. So each test writes
through the real ``DataStore`` and then reads the row back, asserting that

* ``engine`` names the engine that actually ran, rather than ``FIT_ENGINE_UNKNOWN``; and
* ``gate_log_json`` carries the gates' own records, where the engine produced any.

The second assertion is on **content**, deliberately. `[e33]` §3: ``gate_log_json`` is
declared ``TEXT NOT NULL DEFAULT '[]'``, so it is structurally incapable of reporting its
own absence and a null-check would report 100 % coverage over a 99.4 %-empty column.

The gated engine is stubbed rather than configured. ``[eis] engine`` decides which engine
runs, and the legacy engine emits ``gate_log=()`` **by design** — so a test that let the
configuration choose would assert "no gate log" on a legacy rig and prove nothing about
threading. Stubbing ``analyze_spectrum`` pins the contract *"whatever the engine reported,
the row records"* under either setting.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis.report import SigmaReport, SpectrumReport
from softae.analysis.eis_data import EISResult
from softae.analysis.quality import QualityReport, Verdict
from softae.core.data_store import FIT_ENGINE_UNKNOWN, DataStore

# One gate that ran, said something, and dropped a point — the shape `[e33]` found
# missing from every verdict-bearing row in the store.
GATE_LOG = (
    {"gate": "hf_inductive_tail", "passed": True, "n_dropped": 2,
     "severity": "trim", "detail": "2 points above 90 kHz trimmed"},
)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sample_eis() -> EISResult:
    f = np.logspace(1, 5, 50)
    tau = 2 * np.pi * f * 1e-6
    return EISResult.from_arrays(
        channel=1, f=f, z_real=100 + 50 / (1 + tau**2),
        z_imag_neg=np.abs(50 * tau / (1 + tau**2)),
    )


@pytest.fixture
def sample_fit() -> FitResult:
    return FitResult(
        model_name="simpleSalt",
        parameters=np.array([100.0, 1e-7, 0.7, 500.0, 1e-10]),
        R0=100.0, R1=500.0, R0_guess=100.0, R1_guess=500.0,
        z_indices=[0, 3], success=True,
    )


def gated_report(fit: FitResult) -> SpectrumReport:
    """A report from a gated run: an engine name, a verdict, and the log behind it."""
    return SpectrumReport(
        engine="gated", fit=fit,
        sigma=SigmaReport(mode="value", value=1.0e-4, R_reported_ohm=float(fit.R1)),
        quality=QualityReport(Verdict.ACCEPT),
        gate_log=GATE_LOG,
    )


#: The three tests below assert **post-P.18** behaviour against source that was
#: deliberately reverted, so they are correctly red and the reds are not a defect to
#: chase. ``strict=True`` on purpose: it is the only marking that cannot rot quietly,
#: because it turns red the moment P.18 lands and nobody remembers to unmark it. The
#: other two tests in this file pass today and must stay unmarked — under ``strict``
#: an XPASS is a failure.
HELD = pytest.mark.xfail(
    strict=True,
    reason="held pending T7.9 — P.18 GUI report threading paused, see [e34]",
)


def assert_auditable(row: dict, *, engine: str = "gated") -> None:
    """The row names its engine AND carries the evidence behind its verdict."""
    assert row["engine"] == engine, "the row does not name the engine that ran"
    assert row["gate_verdict"] == "accept"
    log = json.loads(row["gate_log_json"])
    assert log, "verdict stored with an empty gate log — `[e33]`'s exact defect"
    assert log[0]["gate"] == "hf_inductive_tail"
    assert log[0]["n_dropped"] == 2


# ── The manual tab ───────────────────────────────────────────────────────────


def _manual_worker(store, monkeypatch, sample_eis, report):
    """A ``_ManualEisWorker`` whose acquisition is stubbed and whose analysis is *report*."""
    from softae.gui.tabs.tab_manual import _ManualEisWorker

    monkeypatch.setattr("softae.analysis.eis.engine.analyze_spectrum",
                        lambda eis_result, **kw: report)
    monkeypatch.setattr("softae.drivers.mscr_library.eis_run_mscrbuild",
                        lambda *a, **k: None)
    monkeypatch.setattr(EISResult, "from_raw",
                        classmethod(lambda cls, raw, **kw: sample_eis))

    pico = SimpleNamespace(sendscript_getdata=lambda *a: object(), _output_dir=".")
    return _ManualEisWorker(
        SimpleNamespace(get=lambda name: pico), store,
        channels=[1], eis_params={}, auto_fit=True,
        fit_model="simpleSalt", auto_save=True,
    )


@HELD
def test_manual_measurement_lands_an_auditable_fit_row(
        tmp_path, monkeypatch, sample_eis, sample_fit):
    """A manual auto-fit must store the verdict and the evidence in the same row."""
    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("manual_eis", mode="manual",
                                 campaign="manual", quality="explore")
        worker = _manual_worker(store, monkeypatch, sample_eis, gated_report(sample_fit))
        payload = worker._measure_one(1, "pico1", run_id)
        row = store.query_fits(measurement_id=payload["measurement_id"])[-1]

    assert_auditable(row)


def test_manual_measurement_payload_still_carries_the_fit(
        tmp_path, monkeypatch, sample_eis, sample_fit):
    """Binding the report must not change what the tab renders.

    ``_measure_one`` used to ``.fit`` the report inline; the payload key the results
    table and the series plot read is ``fit_result``, and it is still the engine's own
    fit object rather than a copy.
    """
    report = gated_report(sample_fit)
    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("manual_eis", mode="manual",
                                 campaign="manual", quality="explore")
        worker = _manual_worker(store, monkeypatch, sample_eis, report)
        payload = worker._measure_one(1, "pico1", run_id)

    assert payload["fit_result"] is sample_fit
    assert payload["fit_error"] is None


# ── The EIS Browser (Analysis tab) ───────────────────────────────────────────


@HELD
def test_browser_fit_lands_an_auditable_fit_row(
        qapp, tmp_path, monkeypatch, sample_eis, sample_fit):
    """The browser's re-fit path, end to end: ``fit_entry`` → ``_on_browser_fit_saved``.

    ``fit_entry`` is called for real so the test proves the report survives the whole
    carry — analysis → ``EISEntry`` → the persist callback — rather than proving that a
    hand-placed attribute is readable.
    """
    from softae.gui.tabs.tab_analysis import AnalysisTab
    from softae.gui.widgets.eis_visualizer_widget import EISEntry, fit_entry

    monkeypatch.setattr("softae.analysis.eis.engine.analyze_spectrum",
                        lambda eis_result, **kw: gated_report(sample_fit))

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)

        entry = EISEntry(label="Ch01 — run", eis=sample_eis, fit=None, sigma=None,
                         measurement_id=mid)
        fit_entry(entry, "simpleSalt", 0.3, 0.1, 0.25)

        AnalysisTab(MagicMock(), data_store=store)._on_browser_fit_saved(
            entry, 0.3, 0.1, 0.25)
        row = store.query_fits(measurement_id=mid)[-1]

    assert_auditable(row)
    # Geometry still recorded — the report rides alongside the σ inputs, not instead.
    assert row["electrode_L_cm"] == pytest.approx(0.3)


@HELD
def test_fit_entry_keeps_the_report_that_produced_the_fit(
        monkeypatch, sample_eis, sample_fit):
    """The entry is the only object crossing the fit worker's thread boundary.

    So if ``fit_entry`` does not park the report on it, nothing downstream can reach
    it — which is precisely how the browser's rows lost their provenance.
    """
    from softae.gui.widgets.eis_visualizer_widget import EISEntry, fit_entry

    report = gated_report(sample_fit)
    monkeypatch.setattr("softae.analysis.eis.engine.analyze_spectrum",
                        lambda eis_result, **kw: report)

    entry = EISEntry(label="x", eis=sample_eis, fit=None, sigma=None)
    fit_entry(entry, "simpleSalt", 0.3, 0.1, 0.25)

    assert entry.report is report
    assert entry.fit is report.fit, "fit and report must describe one analysis"


def test_browser_fit_without_a_report_records_the_absence_not_a_claim(
        qapp, tmp_path, sample_eis, sample_fit):
    """An entry rebuilt from a stored row has a fit and no report — and says so.

    This is the honest-absence half of the change. ``report=None`` must keep meaning
    exactly what omitting the argument meant: ``engine`` unknown, no verdict, empty log.
    A fabricated engine label here would be a worse defect than the one being fixed,
    because it would be unfalsifiable.
    """
    from softae.gui.tabs.tab_analysis import AnalysisTab
    from softae.gui.widgets.eis_visualizer_widget import EISEntry

    with DataStore(tmp_path / "ds") as store:
        run_id = store.start_run("test", mode="manual", campaign="c", quality="explore")
        mid = store.record_measurement(run_id, sample_eis)

        entry = EISEntry(label="Ch01 — run", eis=sample_eis, fit=sample_fit,
                         sigma=None, measurement_id=mid)
        assert entry.report is None, "a history-loaded entry has no report"

        AnalysisTab(MagicMock(), data_store=store)._on_browser_fit_saved(
            entry, 0.3, 0.1, 0.25)
        row = store.query_fits(measurement_id=mid)[-1]

    assert row["engine"] == FIT_ENGINE_UNKNOWN
    assert row["gate_verdict"] is None
    assert json.loads(row["gate_log_json"]) == []
