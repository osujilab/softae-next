"""Whether the impedance semicircle closed inside the swept window.

The synthetic cases pin the rule; the four rig cases pin it against the
instrument. Both are needed: a rule that only ever meets constructed spectra
tests its own author, and archived spectra alone cannot say what *should* have
happened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.arc import (
    CLOSED,
    OPEN,
    UNKNOWN,
    annotate_arc_closure,
    arc_closure,
)
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant
from tests.eis_synthetic import as_eis_result, reference_spectrum

CELL = CellConstant(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=0.015,
                    thickness_method="predicted")

#: The characterisation run, read only. Skipped where it is not present, because
#: the rest of the suite must stay runnable on a machine that never saw the rig.
RUN_DIR = Path("C:/Users/Osuji/softae_data/runs/"
               "20260811T023757Z_equilibration_characterization/data/eis")


def semicircle(f_peak: float = 1.0e3, R: float = 1.0e5,
               f_lo: float = 20.0, f_hi: float = 2.0e5, npts: int = 25):
    """A bare Debye arc: ``−Z″`` peaks at *f_peak* and falls away either side."""
    f = np.logspace(np.log10(f_hi), np.log10(f_lo), npts)
    x = f / f_peak
    return f, R * x / (1.0 + x**2)


class TestArcClosureReadsTheShapeOfTheSweep:
    def test_arc_closure_full_semicircle_reads_closed(self):
        f, y = semicircle()
        arc = arc_closure(f, y)
        assert arc.state == CLOSED
        assert arc.f_peak_hz > arc.f_low_hz

    def test_arc_closure_truncated_semicircle_reads_open(self):
        # Same arc, sweep stopped an order of magnitude above the peak: −Z″ is
        # still climbing when the instrument runs out of frequency.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.f_peak_hz == pytest.approx(arc.f_low_hz)

    def test_arc_closure_open_arc_carries_phase_at_the_sweep_floor_as_severity(self):
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        phase = np.full(f.size, -80.0)
        assert arc_closure(f, y, phase).phase_low_deg == pytest.approx(-80.0)

    def test_arc_closure_without_phase_still_decides_the_state(self):
        # Severity is unavailable, the verdict is not: the peak decides it.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        arc = arc_closure(f, y)
        assert arc.state == OPEN
        assert arc.phase_low_deg != arc.phase_low_deg          # NaN, not 0°

    def test_arc_closure_sweep_order_does_not_change_the_verdict(self):
        f, y = semicircle()
        assert arc_closure(f[::-1], y[::-1]).state == arc_closure(f, y).state


class TestArcClosureIsRobustToOneNoisyPoint:
    """The sweep floor is the noisiest point in the spectrum, and it is also the
    one ``argmax`` is most likely to pick. A guard that did not exist here would
    make every low-frequency outlier read as a physical finding."""

    def test_arc_closure_single_spike_at_the_sweep_floor_stays_closed(self):
        f, y = semicircle()
        y[-1] = y.max() * 3.0            # f ascends last in this descending sweep
        arc = arc_closure(f, y)
        assert arc.state == CLOSED
        assert "excursion" in arc.reason
        # The reported peak is the arc's, not the spike's.
        assert arc.f_peak_hz > arc.f_low_hz

    def test_arc_closure_genuine_rise_at_the_sweep_floor_is_not_suppressed(self):
        # The guard must not swallow the real thing it is guarding against.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        assert arc_closure(f, y).state == OPEN

    def test_arc_closure_noise_level_wobble_below_the_peak_still_reads_open(self):
        # A 1 % wobble two points up-sweep is scatter, not a peak.
        f, y = semicircle(f_peak=2.0, f_lo=20.0)
        y[-2] *= 0.99
        assert arc_closure(f, y).state == OPEN


class TestArcClosureRefusesRatherThanGuesses:
    def test_arc_closure_too_few_points_reads_unknown(self):
        f, y = semicircle(npts=4)
        arc = arc_closure(f, y)
        assert arc.state == UNKNOWN
        assert arc.state != CLOSED
        assert "4 points" in arc.reason

    def test_arc_closure_non_finite_point_reads_unknown(self):
        f, y = semicircle()
        y[3] = np.nan
        assert arc_closure(f, y).state == UNKNOWN

    def test_arc_closure_flat_sweep_reads_unknown(self):
        f, _ = semicircle()
        assert arc_closure(f, np.ones(f.size)).state == UNKNOWN

    def test_arc_closure_mismatched_lengths_reads_unknown(self):
        f, y = semicircle()
        assert arc_closure(f, y[:-1]).state == UNKNOWN


class TestArcClosureAgainstTheRig:
    """Four spectra from run ``20260811T023757Z_equilibration_characterization``,
    round 7 of their block. ch1 is a low-conductivity channel whose arc never
    closes; ch14 is a high-conductivity one whose arc closes at both ends of the
    sweep. Read from the netCDF payloads, never written."""

    CASES = [
        ("eq_ch1_Lup_S0_R7_ch1", OPEN, 20.0, -45.1),
        ("eq_ch1_Ldown_S3_R7_ch1", OPEN, 20.0, -93.8),
        ("eq_ch14_Lup_S0_R7_ch14", CLOSED, 2000.0, -14.7),
        ("eq_ch14_Ldown_S3_R7_ch14", CLOSED, 43.09, -34.4),
    ]

    @pytest.mark.parametrize("stem,state,f_peak,phase", CASES)
    def test_arc_closure_stored_spectrum_matches_the_measured_verdict(
        self, stem, state, f_peak, phase
    ):
        xr = pytest.importorskip("xarray")
        path = RUN_DIR / f"{stem}.nc"
        if not path.exists():
            pytest.skip("characterisation run not present on this machine")
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            arc = arc_closure(ds["frequency_hz"].values, ds["z_imag_neg"].values,
                              ds["phase"].values)
        assert arc.state == state
        assert arc.f_peak_hz == pytest.approx(f_peak, rel=1e-3)
        assert arc.phase_low_deg == pytest.approx(phase, abs=0.05)
        assert arc.f_low_hz == pytest.approx(20.0, rel=1e-3)


class TestTheEngineAnnotatesWithoutDemoting:
    def test_engine_open_arc_does_not_fail_the_fit(self):
        # The regression pin against over-refusing. An extrapolated R₁ is a weaker
        # claim, not a failed measurement, and a third of every run is in that
        # state — demoting them would throw the cold end of the sweep away.
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert report.fit.arc_closure.state == OPEN
        assert report.fit.success is True
        assert report.fit.error_msg == ""
        assert report.fit.R1 == report.fit.R1                  # not NaN'd
        assert report.sigma.mode == "value"

    def test_engine_open_arc_is_named_in_the_quality_issues(self):
        report = analyze_spectrum(as_eis_result(*reference_spectrum()),
                                  cell=CELL, engine="legacy")
        assert any("did not close" in issue for issue in report.quality.issues)

    def test_engine_gated_path_annotates_the_fit_too(self):
        from softae.analysis.eis.settings import EISSettings, GateSettings

        report = analyze_spectrum(
            as_eis_result(*reference_spectrum()), cell=CELL,
            settings=EISSettings(engine="gated", gates=GateSettings(enabled=False)))
        assert report.fit.arc_closure.state in (OPEN, CLOSED)


class TestTheAnnotationIsPersisted:
    def test_record_fit_legacy_path_stores_the_arc_record(self, tmp_path):
        # The path that ships: `[eis] engine` is legacy, and `analysis/eis/router.py`
        # is what writes a campaign's fit rows. The record lands in the four columns,
        # read straight off the fit with no `report=` in the call.
        import json

        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "project")
        try:
            run_id = store.start_run("arc")
            eis = as_eis_result(*reference_spectrum())
            measurement_id = store.record_measurement(run_id, eis)
            fit = analyze_spectrum(eis, cell=CELL, engine="legacy").fit
            fit_id = store.record_fit(measurement_id, fit, L_cm=0.2, t_cm=0.015,
                                      w_cm=0.2)
            row = dict(store._conn.execute(
                "SELECT gate_log_json, parameters_json, success, gate_verdict, engine, "
                "arc_state, arc_f_peak_hz, arc_f_low_hz, arc_phase_low_deg "
                "FROM fit_results WHERE fit_id = ?", (fit_id,)).fetchone())
        finally:
            store.close()

        assert row["arc_state"] == OPEN
        assert row["arc_f_peak_hz"] == pytest.approx(row["arc_f_low_hz"])
        assert row["arc_f_low_hz"] == pytest.approx(20.0)
        assert row["arc_phase_low_deg"] < 0.0
        # The columns agree with the annotation the engine put on the fit — one
        # verdict, not a copy that can drift.
        assert row["arc_state"] == fit.arc_closure.state
        assert row["arc_f_peak_hz"] == pytest.approx(fit.arc_closure.f_peak_hz)
        # Everything else on the row is what it was before the annotation existed,
        # `gate_log_json` included: no report is passed, so it keeps its literal.
        assert row["gate_log_json"] == "[]"
        assert row["success"] == 1
        assert row["gate_verdict"] is None
        assert row["engine"] == "legacy"
        assert json.loads(row["parameters_json"]) == pytest.approx(
            fit.parameters.tolist())


class TestAnnotateArcClosureLeavesTheFitAlone:
    def test_annotate_arc_closure_touches_nothing_but_the_annotation(self):
        from softae.analysis.circuit_fitting import fit_circuit

        eis = as_eis_result(*reference_spectrum())
        fit = fit_circuit(eis, "simpleSalt")
        before = (fit.success, fit.R1, fit.error_msg, fit.parameters.copy())
        annotate_arc_closure(fit, eis)
        assert (fit.success, fit.R1, fit.error_msg) == before[:3]
        assert np.array_equal(fit.parameters, before[3])
