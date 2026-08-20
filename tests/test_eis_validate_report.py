"""The pre-registered decision rule, verified before any data exists.

Every test here runs against **hand-built records**, which is the point: a
validation whose success criterion is chosen after the data arrives proves
nothing, so the arithmetic that will judge the rig is pinned first.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from softae.tools import eis_validate_report as R

# ── Builders ─────────────────────────────────────────────────────────────────

def _record(
    *,
    measurement_id: int = 1,
    channel: int = 18,
    arm: str = R.ARM_REFERENCE,
    cell: str = "18:30:25:1",
    sigma: float | None = 1e-6,
    sigma_is_bound: bool = False,
    r1: float | None = None,
    verdict: str = "",
    arc_state: str = "closed",
    band: float | None = 1.5,
    band_min: float = 1.0,
    apex_hz: float | None = 30.0,
    f_lo_hz: float = 6.475,
    seconds: float = 17.5,
    gate_verdict: str | None = "accept",
    gate_log: list | None = None,
    hold_certified: str = "settled",
    excursion: bool = False,
    mock: bool = False,
    segmented: bool = False,
) -> R.SweepRecord:
    params = {
        "eis_validation_name": "v",
        "eis_validation_arm": arm,
        "eis_validation_cell": cell,
        "eis_validation_hold_epoch": 1,
        "eis_validation_hold_certified": hold_certified,
        "eis_validation_hold_excursion": excursion,
        "eis_validation_arc_state": arc_state,
        "eis_validation_band_below_apex_decades": band,
        "eis_validation_band_min_decades": band_min,
        "eis_validation_apex_hz": apex_hz,
        "eis_validation_f_lo_hz": f_lo_hz,
        "eis_validation_mock": mock,
        "eis_scout_verdict": verdict,
    }
    if segmented:
        params["eis_sweep"] = "segmented"
    return R.SweepRecord(
        measurement_id=measurement_id, run_id="run", channel=channel,
        timestamp="2026-08-19T00:00:00", seconds=seconds, params=params,
        sigma=sigma, sigma_is_bound=sigma_is_bound, r1_ohm=r1,
        gate_verdict=gate_verdict, gate_log=gate_log or [],
        fit_arc_state=arc_state,
    )


def _treatment_cell(
    index: int, *, d_scout: float, d_adaptive: float, **kwargs
) -> list[R.SweepRecord]:
    """One TREATMENT cell whose deviations are exactly as asked, in decades."""
    cell, channel = f"{index}:30:25:1", index
    sigma_ref = 1e-6
    return [
        _record(measurement_id=index * 10, channel=channel, cell=cell,
                arm=R.ARM_REFERENCE, sigma=sigma_ref, seconds=120.42,
                arc_state="closed", band=1.5, apex_hz=30.0, f_lo_hz=1.351),
        _record(measurement_id=index * 10 + 1, channel=channel, cell=cell,
                arm=R.ARM_SCOUT, sigma=sigma_ref * 10 ** d_scout,
                verdict="extend_low", seconds=17.5, arc_state="open",
                band=0.2, f_lo_hz=6.475, **kwargs),
        _record(measurement_id=index * 10 + 2, channel=channel, cell=cell,
                arm=R.ARM_FOLLOW_UP, sigma=sigma_ref * 10 ** d_adaptive,
                seconds=37.19, arc_state="closed", band=1.2, f_lo_hz=3.912),
    ]


def _control_cell(index: int, *, d_scout: float = 0.0) -> list[R.SweepRecord]:
    cell, channel = f"{index}:30:25:1", index
    return [
        _record(measurement_id=index * 10, channel=channel, cell=cell,
                arm=R.ARM_REFERENCE, sigma=1e-6, seconds=120.42, f_lo_hz=1.351),
        _record(measurement_id=index * 10 + 1, channel=channel, cell=cell,
                arm=R.ARM_SCOUT, sigma=1e-6 * 10 ** d_scout, verdict="ok",
                seconds=17.5, f_lo_hz=6.475),
    ]


def _controls(count: int) -> list[R.SweepRecord]:
    return [r for i in range(90, 90 + count)
            for r in _control_cell(i, d_scout=0.01)]


def _cells(records):
    return R.assemble_cells(records)


# ── D1-D2: what "improvement" means ──────────────────────────────────────────

def test_improvement_is_absolute_deviation_reduction():
    cell = _cells(_treatment_cell(1, d_scout=-0.30, d_adaptive=-0.05))[0]
    assert cell.delta_scout() == pytest.approx(-0.30, abs=1e-9)
    assert cell.delta_adaptive() == pytest.approx(-0.05, abs=1e-9)
    # Moving TOWARD the reference is positive, whichever side it started on.
    assert cell.improvement() == pytest.approx(0.25, abs=1e-9)


def test_improvement_is_negative_when_adaptive_moves_further_away():
    cell = _cells(_treatment_cell(1, d_scout=-0.05, d_adaptive=-0.30))[0]
    assert cell.improvement() == pytest.approx(-0.25, abs=1e-9)


def test_control_cells_have_identically_zero_improvement():
    """Not approximately: the adaptive row IS the scout row, the same object."""
    cell = _cells(_control_cell(2, d_scout=0.02))[0]
    assert cell.population == R.CONTROL
    assert cell.adaptive is cell.scout
    assert cell.improvement() == 0.0


# ── Populations ──────────────────────────────────────────────────────────────

def test_population_partition_matches_the_apex_windows():
    records = (
        _control_cell(3)                                        # apex high
        + _treatment_cell(4, d_scout=-0.2, d_adaptive=-0.05)     # ref closed
    )
    # An extend_low cell whose reference did NOT reach a decade past its apex.
    records += [
        _record(measurement_id=500, channel=5, cell="5:30:25:1",
                arm=R.ARM_REFERENCE, arc_state="closed", band=0.6,
                apex_hz=5.34, f_lo_hz=1.351),
        _record(measurement_id=501, channel=5, cell="5:30:25:1",
                arm=R.ARM_SCOUT, verdict="extend_low", arc_state="open",
                band=0.0, f_lo_hz=6.475),
    ]
    populations = {c.channel: c.population for c in _cells(records)}
    assert populations == {3: R.CONTROL, 4: R.TREATMENT, 5: R.UNRESOLVED}


def test_reference_valid_needs_band_not_merely_a_closed_state():
    """`state == closed` is an apex inside the window; it is not a decade of band."""
    closed_but_shallow = _record(arc_state="closed", band=0.6, band_min=1.0)
    assert closed_but_shallow.arc_closed is True
    assert closed_but_shallow.reference_valid is False
    assert _record(arc_state="closed", band=1.2).reference_valid is True
    assert _record(arc_state="open", band=1.2).reference_valid is False


def test_verdicts_adaptive_declines_to_act_on_are_excluded():
    for verdict in ("no_arc", "no_data", "extend_high"):
        records = [
            _record(measurement_id=1, arm=R.ARM_REFERENCE),
            _record(measurement_id=2, arm=R.ARM_SCOUT, verdict=verdict),
        ]
        assert _cells(records)[0].population == R.EXCLUDED


def test_open_arc_reference_never_yields_an_accuracy_number():
    """UNRESOLVED cells carry no deviation into any accuracy table."""
    unresolved = [
        _record(measurement_id=600, channel=6, cell="6:30:25:1",
                arm=R.ARM_REFERENCE, sigma=1e-6, arc_state="closed", band=0.4,
                apex_hz=4.0, f_lo_hz=1.351),
        _record(measurement_id=601, channel=6, cell="6:30:25:1",
                arm=R.ARM_SCOUT, sigma=1e-4, verdict="extend_low",
                arc_state="open", band=0.0, f_lo_hz=6.475),
    ]
    records = unresolved + _treatment_cell(7, d_scout=-0.2, d_adaptive=-0.1)
    payload = R.build_payload(records, _cells(records), {},
                              R.evaluate(_cells(records)))
    # The 2-decade UNRESOLVED deviation would dominate any median it entered.
    assert payload["deviation"]["delta_scout"]["n"] == 1
    assert payload["deviation"]["delta_scout"]["median"] == pytest.approx(-0.2)
    assert payload["populations"][R.UNRESOLVED] == ["6:30:25:1"]


# ── D3-D4 and the thresholds ─────────────────────────────────────────────────

def _nine_treatment(improvements: list[float]) -> list[R.SweepRecord]:
    records: list[R.SweepRecord] = []
    for i, gain in enumerate(improvements, start=10):
        records += _treatment_cell(i, d_scout=-0.40, d_adaptive=-(0.40 - gain))
    return records + _drift_row()


def _drift_row(delta: float = 0.0) -> list[R.SweepRecord]:
    """A held drift check. Without one H3 is UNEVALUABLE and every outcome is
    INSUFFICIENT -- which is the intended routing, pinned separately below."""
    return [_record(measurement_id=999, channel=10, cell="10:30:25:1",
                    arm=R.ARM_REFERENCE_END, sigma=1e-6 * 10 ** delta,
                    seconds=120.42)]


def test_d1_passes_at_the_threshold_and_fails_below_it():
    at = R.evaluate(_cells(_nine_treatment([0.10] * 6)), min_treatment=6)
    below = R.evaluate(_cells(_nine_treatment([0.09] * 6)), min_treatment=6)
    assert _status(at, "D1") == R.PASS
    assert _status(below, "D1") == R.FAIL


def test_d2_sign_consistency_rejects_a_two_cell_median():
    """A large median carried by 2 of 9 cells is a tail, not an offset."""
    gains = [0.60, 0.60] + [-0.01] * 7
    verdict = R.evaluate(_cells(_nine_treatment(gains)), min_treatment=6)
    assert _status(verdict, "D2") == R.FAIL


def test_d3_noise_floor_failure_yields_insufficient_not_no_go():
    records = _nine_treatment([0.30] * 6) + _control_cell(90, d_scout=0.20)
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert _status(verdict, "D3") == R.FAIL
    assert verdict.outcome == R.OUTCOME_INSUFFICIENT


def test_d4_reports_an_inverted_sign_without_auto_failing():
    """A positive median Delta_scout is a flag; D1-D3 still decide the outcome."""
    records = [
        r for i in range(10, 16)
        for r in _treatment_cell(i, d_scout=+0.40, d_adaptive=+0.10)
    ] + _controls(20) + _drift_row()
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert _status(verdict, "D4") == R.FAIL
    assert _status(verdict, "D1") == R.PASS
    # Not NO-GO: an inverted sign downgrades to CONDITIONAL GO and demands prose.
    assert verdict.outcome == R.OUTCOME_CONDITIONAL_GO
    assert any("INVERTED" in reason for reason in verdict.reasons)


def test_offset_and_scatter_are_reported_separately():
    """No RMS anywhere: a wide-scatter zero median must not look like a small offset."""
    spread = R.describe([-0.5, -0.1, 0.0, 0.1, 0.5])
    assert spread.median == pytest.approx(0.0)
    assert spread.mad == pytest.approx(0.1)
    assert spread.iqr == pytest.approx(0.2)
    assert (spread.minimum, spread.maximum) == (-0.5, 0.5)
    assert (spread.n_positive, spread.n_negative, spread.n_zero) == (2, 2, 1)
    assert not hasattr(spread, "rms")
    assert "rms" not in json.dumps(spread.as_dict()).lower()


# ── Vetoes ───────────────────────────────────────────────────────────────────

def test_gate_regression_vetoes_a_passing_primary():
    records = _nine_treatment([0.30] * 6) + _control_cell(90, d_scout=0.01)
    for row in records:
        if row.arm == R.ARM_FOLLOW_UP and row.channel == 10:
            row.gate_verdict = "reject"
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert _status(verdict, "D1") == R.PASS
    assert verdict.outcome == R.OUTCOME_NO_GO
    assert any(v.startswith("V1") for v in verdict.vetoes)


def test_segmented_follow_up_failing_a_gate_the_scout_passed_vetoes():
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    records[1].gate_log = [{"gate": "hf_inductive", "passed": True}]
    records[2].gate_log = [{"gate": "hf_inductive", "passed": False}]
    records[2].params["eis_sweep"] = "segmented"
    vetoes = R.evaluate_vetoes(_cells(records))
    assert any(v.startswith("V2") and "hf_inductive" in v for v in vetoes)


def test_a_narrower_follow_up_vetoes():
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    records[2].params["eis_validation_f_lo_hz"] = 20.0   # ABOVE the scout's 6.475
    vetoes = R.evaluate_vetoes(_cells(records))
    assert any(v.startswith("V3") for v in vetoes)


# ── Time budget ──────────────────────────────────────────────────────────────

def test_time_ratio_counts_the_scout_on_accepted_cells():
    """Omitting the accepted cells' scout time changes the ratio -- so it is counted."""
    records = (
        _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1) + _control_cell(2))
    cells = _cells(records)
    payload = R.build_payload(records, cells, {}, R.evaluate(cells))
    budget = payload["time_budget"]
    # scout 17.5 x 2 cells + one 37.19 follow-up, over scout 17.5 x 2.
    assert budget["sum_t_control_s"] == pytest.approx(35.0)
    assert budget["sum_t_adaptive_s"] == pytest.approx(72.19)
    honest = budget["sum_t_adaptive_s"] / budget["sum_t_control_s"]
    dishonest = 72.19 / 17.5          # control cell's scout dropped
    assert honest == pytest.approx(2.0626, abs=1e-3)
    assert dishonest != pytest.approx(honest)


def test_reference_cost_is_excluded_from_the_production_ratio():
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    cells = _cells(records)
    budget = R.build_payload(records, cells, {},
                             R.evaluate(cells))["time_budget"]
    assert budget["sum_t_reference_s"] == pytest.approx(120.42)
    assert budget["sum_t_reference_s"] not in (
        budget["sum_t_adaptive_s"], budget["sum_t_control_s"])
    assert budget["sum_t_adaptive_s"] == pytest.approx(17.5 + 37.19)


# ── Hold integrity ───────────────────────────────────────────────────────────

def test_h3_drift_failure_yields_insufficient():
    records = [r for r in _nine_treatment([0.30] * 6)
               if r.arm != R.ARM_REFERENCE_END]
    records += _control_cell(90, d_scout=0.01) + _drift_row(delta=0.4)
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert _status(verdict, "H3") == R.FAIL
    assert verdict.outcome == R.OUTCOME_INSUFFICIENT


def test_settle_disabled_withholds_the_outcome():
    records = _nine_treatment([0.30] * 6) + _control_cell(90, d_scout=0.01)
    for row in records:
        row.params["eis_validation_hold_certified"] = "disabled"
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert _status(verdict, "H1") == R.FAIL
    assert verdict.outcome == R.OUTCOME_WITHHELD


def test_settle_ceiling_yields_insufficient_not_withheld():
    records = _nine_treatment([0.30] * 6) + _control_cell(90, d_scout=0.01)
    for row in records:
        row.params["eis_validation_hold_certified"] = "ceiling"
    assert R.evaluate(_cells(records), min_treatment=6).outcome == (
        R.OUTCOME_INSUFFICIENT)


def test_excursion_cells_are_excluded_and_the_count_is_printed():
    records = _nine_treatment([0.30] * 6) + _control_cell(90, d_scout=0.01)
    for row in records:
        if row.channel == 10:
            row.params["eis_validation_hold_excursion"] = True
    cells = _cells(records)
    payload = R.build_payload(records, cells, {}, R.evaluate(cells))
    assert payload["completeness"]["n_excluded_excursion_cells"] == 1
    assert payload["deviation"]["improvement"]["n"] == 5
    assert "EXCLUDED" in R.render(payload)


# ── Mechanism ────────────────────────────────────────────────────────────────

def test_rescue_depth_separates_mechanism_limited_from_no_go():
    # The follow-up reaches barely anywhere against a deep requirement.
    short = _nine_treatment([0.0] * 6)
    for row in short:
        if row.arm == R.ARM_FOLLOW_UP:
            row.params["eis_validation_f_lo_hz"] = 6.0   # ~0.03 dec delivered
        if row.arm == R.ARM_REFERENCE:
            row.params["eis_validation_apex_hz"] = 14.0  # ~0.66 dec required
    limited = R.evaluate(_cells(short + _control_cell(90, d_scout=0.01)),
                         min_treatment=6)
    assert limited.outcome == R.OUTCOME_MECHANISM_LIMITED

    deep = _nine_treatment([0.0] * 6)
    for row in deep:
        if row.arm == R.ARM_FOLLOW_UP:
            row.params["eis_validation_f_lo_hz"] = 0.05  # far past what was asked
    no_go = R.evaluate(_cells(deep + _control_cell(90, d_scout=0.01)),
                       min_treatment=6)
    assert no_go.outcome == R.OUTCOME_NO_GO


def test_closure_discordance_is_reported_on_treatment():
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    cells = _cells(records)
    table = R.build_payload(records, cells, {},
                            R.evaluate(cells))["mechanism"]["closure_discordance"]
    assert table["open_closed"] == 1
    assert sum(table.values()) == 1


# ── sigma_is_bound, and the outcomes ─────────────────────────────────────────

def test_sigma_bound_rows_are_not_treated_as_missing():
    """A bound is informative about closure and uninformative about magnitude."""
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    records[2].sigma, records[2].sigma_is_bound = None, True
    records[2].r1_ohm = 1.0                       # would otherwise fall back to R1
    cell = _cells(records)[0]
    assert cell.delta_adaptive() is None          # excluded, not folded in
    assert cell.delta_scout() is not None
    payload = R.build_payload(records, _cells(records), {},
                              R.evaluate(_cells(records)))
    assert payload["sigma_bound_rows"] == [records[2].measurement_id]


def test_deviation_falls_back_to_r1_when_no_geometry_resolved():
    """sigma = K/R with the same K on both sides, so R1 gives the same decade."""
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    for row in records:
        row.sigma = None
    records[0].r1_ohm = 1.0e8                                  # reference
    records[1].r1_ohm = 1.0e8 * 10 ** 0.3                      # scout reads R high
    records[2].r1_ohm = 1.0e8 * 10 ** 0.1
    cell = _cells(records)[0]
    assert cell.delta_scout() == pytest.approx(-0.3, abs=1e-9)
    assert cell.improvement() == pytest.approx(0.2, abs=1e-9)


def test_a_passing_run_reaches_go():
    # 6 extending cells against 20 accepted ones: ratio = 1 + 2.125 * f, and
    # T1's 1.5x accommodates about a quarter of the strip extending.
    records = [
        r for i in range(10, 16)
        for r in _treatment_cell(i, d_scout=-0.40, d_adaptive=-0.10)
    ] + _controls(20) + _drift_row()
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert [c.status for c in verdict.criteria if c.name.startswith("T1")] == [R.PASS]
    assert verdict.outcome == R.OUTCOME_GO


def test_insufficient_treatment_stops_short_of_a_verdict():
    records = _nine_treatment([0.30] * 2) + _control_cell(90, d_scout=0.01)
    verdict = R.evaluate(_cells(records), min_treatment=6)
    assert verdict.outcome == R.OUTCOME_INSUFFICIENT
    assert any("min-treatment" in reason for reason in verdict.reasons)


def test_mock_report_refuses_a_go():
    records = [
        r for i in range(10, 16)
        for r in _treatment_cell(i, d_scout=-0.40, d_adaptive=-0.10)
    ] + _controls(20) + _drift_row()
    for row in records:
        row.params["eis_validation_mock"] = True
    verdict = R.evaluate(_cells(records), min_treatment=6, mock=True)
    assert verdict.outcome == R.OUTCOME_WITHHELD
    payload = R.build_payload(records, _cells(records), {}, verdict)
    assert payload["mock"] is True
    assert "(MOCK)" in R.render(payload)


# ── Reading the database ─────────────────────────────────────────────────────

def _seed_db(path, rows):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE measurements (
            measurement_id INTEGER PRIMARY KEY, run_id TEXT, channel INTEGER,
            timestamp TEXT, measurement_time_s REAL, eis_params_json TEXT);
        CREATE TABLE fit_results (
            fit_id INTEGER PRIMARY KEY, measurement_id INTEGER,
            sigma_S_per_cm REAL, sigma_is_bound INTEGER, R1 REAL,
            gate_verdict TEXT, gate_log_json TEXT, arc_state TEXT);
        CREATE TABLE campaign_checkpoints (campaign TEXT PRIMARY KEY, spec_json TEXT);
        """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO measurements VALUES (?,?,?,?,?,?)",
            (row.measurement_id, row.run_id, row.channel, row.timestamp,
             row.seconds, json.dumps(row.params)))
        conn.execute(
            "INSERT INTO fit_results (measurement_id, sigma_S_per_cm, "
            "sigma_is_bound, R1, gate_verdict, gate_log_json, arc_state) "
            "VALUES (?,?,?,?,?,?,?)",
            (row.measurement_id, row.sigma, int(row.sigma_is_bound), row.r1_ohm,
             row.gate_verdict, json.dumps(row.gate_log), row.fit_arc_state))
    conn.commit()
    conn.close()


def test_report_is_read_only(tmp_path):
    db = tmp_path / "softae.db"
    _seed_db(db, _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1))
    conn = R._connect_ro(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO measurements (measurement_id) VALUES (99)")
    conn.close()
    assert R.load_records(db, "v")


def test_provenance_keys_survive_the_json_round_trip(tmp_path):
    db = tmp_path / "softae.db"
    _seed_db(db, _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1))
    row = R.load_records(db, "v")[0]
    for key in ("eis_validation_name", "eis_validation_arm", "eis_validation_cell",
                "eis_validation_hold_epoch", "eis_validation_hold_certified",
                "eis_validation_hold_excursion", "eis_validation_arc_state",
                "eis_validation_apex_hz", "eis_validation_f_lo_hz",
                "eis_validation_band_min_decades"):
        assert key in row.params, key


def test_a_nan_bearing_row_elsewhere_does_not_break_the_query(tmp_path):
    """json.dumps writes bare `NaN`, which SQLite's JSON1 rejects for the WHOLE row.

    A predicate built on `json_extract` would raise `malformed JSON` for the
    entire scan because of one unrelated row, so the query is textual.
    """
    db = tmp_path / "softae.db"
    _seed_db(db, _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1))
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO measurements VALUES (?,?,?,?,?,?)",
        (777, "other", 1, "t", 1.0,
         '{"eis_validation_name": "somebody_else", "x": NaN}'))
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(db).execute(
            "SELECT json_extract(eis_params_json, '$.eis_validation_name') "
            "FROM measurements").fetchall()
    assert len(R.load_records(db, "v")) == 3


def test_report_regenerates_mid_run(tmp_path):
    """A half-complete validation reports its completeness and withholds a verdict."""
    db = tmp_path / "softae.db"
    partial = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)[:1]  # reference only
    _seed_db(db, partial)
    payload = R.generate(db, "v", min_treatment=6)
    assert payload["completeness"]["n_sweeps"] == 1
    assert payload["completeness"]["n_complete_cells"] == 0
    assert payload["outcome"] == R.OUTCOME_INSUFFICIENT
    assert R.render(payload)


def test_checkpoint_spec_is_read_back(tmp_path):
    db = tmp_path / "softae.db"
    _seed_db(db, _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1))
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO campaign_checkpoints VALUES (?,?)",
                 (R.checkpoint_campaign("v"), json.dumps({"fingerprint": "abc"})))
    conn.commit()
    conn.close()
    assert R.load_checkpoint(db, "v")["fingerprint"] == "abc"


def test_report_states_how_long_the_sample_soaked(tmp_path):
    """A reader of the verdict must be able to tell an equilibrated film from
    one measured on the drying transient. The two produce the same-looking
    report, so the soak is stated in the condition header rather than inferred.
    """
    records = _treatment_cell(1, d_scout=-0.3, d_adaptive=-0.1)
    payload = R.build_payload(
        records, _cells(records),
        {"rh_setpoint_pct": 30.0, "temp_setpoint_c": 25.0, "soak_s": 14400.0},
        R.evaluate(_cells(records)))
    assert "soak            4.00 h held at condition" in R.render(payload)


def test_report_soak_zero_is_stated_as_none_not_omitted():
    """No soak is itself the finding, so the row appears at zero as well."""
    assert "none" in R._soak_line({"soak_s": 0.0})
    assert "settle gate directly" in R._soak_line({"soak_s": 0})


def test_report_soak_absent_from_the_spec_is_not_stated_never_zero():
    """A validation recorded before `--soak-h` existed made no soak claim at
    all, which is a different assertion from having made one of zero."""
    assert "not stated" in R._soak_line({})
    assert "not stated" in R._soak_line({"soak_s": "nonsense"})


def test_quantiles_and_median_on_small_samples():
    assert R._median([1.0]) == 1.0
    assert R._median([1.0, 3.0]) == 2.0
    assert R._quantile([1.0], 0.25) == 1.0
    assert R._quantile([0.0, 1.0, 2.0, 3.0], 0.5) == pytest.approx(1.5)
    assert R.describe([]).n == 0
    assert R.describe([float("nan"), None]).median is None


def _status(verdict: R.Verdict, prefix: str) -> str:
    return next(c.status for c in verdict.criteria if c.name.startswith(prefix))
