"""The runner: hold gates, acquisition sequence, persistence, resume, mock.

Every path here goes through the mock. Nothing in this file actuates anything.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from softae.tools import eis_validate as V
from softae.tools import eis_validate_hold as H
from softae.tools import eis_validate_mock as M
from softae.tools import eis_validate_report as R

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _args(tmp_path, **overrides):
    argv = [
        "run", "--channels", overrides.pop("channels", "18,19,20"),
        "--rh-setpoint-pct", "30", "--temp-setpoint-c", "25",
        "--validation-name", overrides.pop("name", "t"),
        "--project", str(tmp_path), "--mock", "--min-treatment", "1",
        "--drift-check", str(overrides.pop("drift_check", 1)),
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return V.build_parser().parse_args(argv)


def _manager(apexes=None, drift=0.0):
    from softae.drivers.factory import create_manager

    manager = create_manager(mock=True)
    M.install_fast_conditions(manager)
    M.install_mock_picos(manager, M.MockRig(
        apex_hz=dict(apexes or {}), default_apex_hz=30.0,
        drift_decades_per_hour=drift, virtual_s_per_sweep=60.0))
    return manager


def _context(tmp_path, manager, plan):
    from softae.core.data_store import DataStore

    store = DataStore(tmp_path)
    run_id = store.start_run("eis_validate", mode="validation")
    return V.RunContext(
        plan=plan, manager=manager, data_store=store, run_id=run_id,
        run_dir=Path(store.project_dir) / "runs" / run_id)


def _plan(tmp_path, **overrides):
    return V.build_plan(_args(tmp_path, **overrides))


def _rows(tmp_path, name="t"):
    db = R.resolve_db(Path(tmp_path))
    return R.load_records(db, name) if db.exists() else []


# ── H-series: hold and safety ────────────────────────────────────────────────

def test_thermal_confirmation_is_required_and_typed(tmp_path, monkeypatch, capsys):
    """`confirm_thermal` is the gate that bites; `y` is not `yes`."""
    from softae.tools.equilibration import CONFIRM_WORD

    assert CONFIRM_WORD == "yes"
    plan = _plan(tmp_path)
    projection = H.project(plan)

    calls = []
    real = V._confirm

    import softae.tools.equilibration as EQ
    monkeypatch.setattr(EQ, "input", lambda *_a: calls.append("asked") or "y",
                        raising=False)
    monkeypatch.setattr(
        EQ, "confirm_thermal",
        lambda config, *, assume_yes=False, reader=None, **kw:
            True if assume_yes else (reader or (lambda _p: ""))("?") == "yes")
    assert real(plan, projection, assume_yes=True) is True

    monkeypatch.undo()
    from softae.workflows.equilibration import EquilibrationConfig

    config = EquilibrationConfig(channels=[18], temperatures_C=[25.0])
    assert EQ.confirm_thermal(config, reader=lambda _p: "y") is False
    assert EQ.confirm_thermal(config, reader=lambda _p: "yes") is True
    capsys.readouterr()


def test_arming_assert_is_called_even_though_it_is_a_noop_here(tmp_path, monkeypatch):
    """Pins the finding, so a later reader does not delete the call as dead code.

    ``MOTION_INSTRUMENTS`` never matches a temp/RH/EIS manager, so the assert
    returns an empty list and blocks nothing -- which is exactly why the typed
    thermal confirmation exists. Both facts are asserted together.
    """
    from softae.core.hardware_safety import MOTION_INSTRUMENTS

    assert MOTION_INSTRUMENTS == ("stage", "syringe", "piezo")
    seen = []
    import softae.core.hardware_safety as HS

    monkeypatch.setattr(
        HS, "assert_hardware_armed",
        lambda manager, action="": seen.append(action) or [])
    V.cmd_run(_args(tmp_path, dry_run=True))
    # dry-run exits before arming; the real path is covered by the run below.
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "arm",
                   "--project", str(tmp_path), "--mock", "--min-treatment", "1",
                   "--drift-check", "0"]) == V.EXIT_OK
    assert seen and "18" in seen[-1]


def test_approach_timeout_refuses_to_start(tmp_path, monkeypatch):
    """`reached=False` is a disqualification here, not a result. No sweep is taken."""
    import softae.workflows.equilibration as EQ
    from softae.workflows.equilibration import ApproachOutcome

    attempts = []

    def _never(read_pv, target, *, axis, **kw):
        attempts.append(axis)
        return ApproachOutcome(axis=axis, target=target, reached=False,
                               elapsed_s=1.0, pv_final=99.0, n_samples=1)

    monkeypatch.setattr(EQ, "approach_setpoint", _never)
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "to",
                   "--project", str(tmp_path), "--mock",
                   "--min-treatment", "1"]) == V.EXIT_FAILED
    assert attempts == ["temperature", "temperature"]     # exactly one retry
    assert _rows(tmp_path, "to") == []


@pytest.mark.parametrize("verdict", ["ceiling", "not_evaluable"])
def test_settle_ceiling_refuses_to_start(verdict):
    """The campaign path records these and continues; this harness refuses."""
    outcome = H.SettleOutcome(verdict, n_rounds=9, elapsed_s=5400.0)
    with pytest.raises(H.RefuseToStart) as excinfo:
        H.assert_settle_licensed(outcome)
    assert verdict in str(excinfo.value)


def test_settle_off_withholds_the_outcome(tmp_path, capsys):
    """Allowed, but never a silent proceed: rows carry `disabled`."""
    H.assert_settle_licensed(H.SettleOutcome("disabled", 0, 0.0))
    assert "WITHHELD" in capsys.readouterr().out

    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "off",
                   "--project", str(tmp_path), "--mock", "--settle", "off",
                   "--min-treatment", "1", "--drift-check", "0"]) == V.EXIT_OK
    rows = _rows(tmp_path, "off")
    assert rows and all(r.hold_certified == "disabled" for r in rows)
    payload = R.generate(R.resolve_db(Path(tmp_path)), "off", min_treatment=1)
    assert payload["outcome"] == R.OUTCOME_WITHHELD


def test_hold_fault_parks_and_exits_nonzero(tmp_path, monkeypatch):
    from softae.errors import SafetyError

    parks = []
    import softae.core.safe_park as SP

    monkeypatch.setattr(SP, "safe_park", lambda manager, **kw:
                        parks.append(kw) or _ParkResult())
    monkeypatch.setattr(H.HoldWatch, "poll",
                        lambda self: (_ for _ in ()).throw(
                            SafetyError("temperature ran away")))
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "fault",
                   "--project", str(tmp_path), "--mock",
                   "--min-treatment", "1"]) == V.EXIT_FAILED
    assert parks and parks[-1]["retract_head"] is None


def test_keyboard_interrupt_parks_with_retract_head_none(tmp_path, monkeypatch):
    """`retract_head=None`: absent an operator, add no motion to an unknown."""
    parks = []
    import softae.core.safe_park as SP

    monkeypatch.setattr(SP, "safe_park", lambda manager, **kw:
                        parks.append(kw) or _ParkResult())
    monkeypatch.setattr(H.HoldWatch, "poll",
                        lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "kbd",
                   "--project", str(tmp_path), "--mock",
                   "--min-treatment", "1"]) == V.EXIT_INTERRUPTED
    assert len(parks) == 1 and parks[0]["retract_head"] is None


def test_end_state_parks_by_default_and_hold_does_not(tmp_path, monkeypatch):
    parks = []
    import softae.core.safe_park as SP

    monkeypatch.setattr(SP, "safe_park", lambda manager, **kw:
                        parks.append(kw) or _ParkResult())
    base = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--project", str(tmp_path), "--mock",
            "--min-treatment", "1", "--drift-check", "0"]
    assert V.main(base + ["--validation-name", "p"]) == V.EXIT_OK
    assert len(parks) == 1
    assert V.main(base + ["--validation-name", "h", "--end-state", "hold"]) == V.EXIT_OK
    assert len(parks) == 1                                  # unchanged


def test_rh_approach_timeout_default_is_not_the_shipped_1800(tmp_path):
    """Pins the divergence AND its reason, so nobody 'fixes' it back."""
    from softae.workflows.equilibration import DEFAULT_RH_APPROACH_TIMEOUT_S

    assert DEFAULT_RH_APPROACH_TIMEOUT_S == 1800.0
    assert H.DEFAULT_RH_APPROACH_TIMEOUT_S == 5400.0
    plan = _plan(tmp_path)
    assert plan.rh_approach_timeout_s == 5400.0
    text = H.render_projection(plan, H.project(plan))
    assert "5000 s" in text and "1800 s" in text


def test_warn_excursion_stamps_only_the_rows_inside_the_window(tmp_path):
    manager = _manager()
    plan = _plan(tmp_path)
    ctx = _context(tmp_path, manager, plan)
    ctx.watch = H.HoldWatch(manager=manager, plan=plan)

    ctx.watch.excursion = False
    V.measure_reference(ctx, 18, R.ARM_REFERENCE)
    ctx.watch.excursion = True
    V.measure_reference(ctx, 19, R.ARM_REFERENCE)

    stamps = {r.channel: r.hold_excursion for r in _rows(tmp_path)}
    assert stamps == {18: False, 19: True}


def test_hold_watch_grades_rh_by_state_not_by_object_identity(tmp_path):
    """`classify_rh_hold` returns a verdict OBJECT; comparing it to the string
    constant is always unequal and would mark every poll an excursion."""
    manager = _manager()
    plan = _plan(tmp_path)
    watch = H.HoldWatch(manager=manager, plan=plan)
    for _ in range(4):
        watch.poll()
    assert watch.excursion is False


def test_too_few_channels_is_refused_before_anything_is_heated(tmp_path):
    """The settle criterion needs 3 participating channels; 2 can never certify.

    Caught at plan time rather than after the full --settle-max-hold-s, which
    would otherwise be 90 minutes at temperature ending in a refusal that names
    the wrong cause.
    """
    from softae.analysis.equilibration import DEFAULT_SETTLE_MIN_CHANNELS

    assert DEFAULT_SETTLE_MIN_CHANNELS == 3
    with pytest.raises(H.RefuseToStart):
        H.validate_plan(_plan(tmp_path, channels="18,19"))
    H.validate_plan(_plan(tmp_path, channels="18,19,20"))
    off = _plan(tmp_path, channels="18,19")
    off.settle = False
    H.validate_plan(off)                       # --settle off is the stated escape

    assert V.main(["run", "--channels", "18,19", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "few",
                   "--project", str(tmp_path), "--mock"]) == V.EXIT_FAILED
    assert _rows(tmp_path, "few") == []


# ── P-series: populations and the arc-capture watch ──────────────────────────

def test_ok_cells_are_always_reference_valid(tmp_path):
    """Falls out for free: `ok` needs a decade below the apex on the BASELINE."""
    plan = _plan(tmp_path)
    assert plan.baseline_ok_hz > plan.ref_close_hz
    assert plan.baseline_ok_hz == pytest.approx(64.75, rel=1e-6)
    assert plan.ref_close_hz == pytest.approx(13.51, rel=1e-6)


def test_population_partition_matches_the_apex_windows(tmp_path):
    plan = _plan(tmp_path)
    assert H.classify_apex(5.0, plan) == R.UNRESOLVED
    assert H.classify_apex(30.0, plan) == R.TREATMENT
    assert H.classify_apex(200.0, plan) == R.CONTROL
    assert H.classify_apex(float("nan"), plan) == R.UNRESOLVED


def test_arc_capture_watch_stops_below_min_treatment(tmp_path):
    """Stop and ask, BEFORE any reference sweep -- not 40 minutes to INSUFFICIENT."""
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "thin",
                   "--project", str(tmp_path), "--mock",
                   "--mock-apex-hz", "5",             # everything UNRESOLVED
                   "--min-treatment", "6"]) == V.EXIT_FAILED
    assert _rows(tmp_path, "thin") == []


def test_apex_histogram_comes_from_the_settle_rounds(tmp_path):
    """No extra sweeps: the histogram is built from spectra already being taken."""
    manager = _manager({18: 5.0, 19: 30.0, 20: 200.0})
    plan = _plan(tmp_path, channels="18,19,20")
    ctx = _context(tmp_path, manager, plan)

    sweeps = []

    def _measure(channel):
        sweeps.append(channel)
        return V._settle_sweep(ctx, channel)

    clock = H.VirtualClock()
    outcome = H.settle_phase(manager, plan, _measure, sleep=clock.sleep,
                             now=clock, min_hold_first_s=0.0)
    assert outcome.certified
    assert outcome.projected == {R.UNRESOLVED: 1, R.TREATMENT: 1, R.CONTROL: 1}
    # Every sweep taken was a settle round's; none was extra, and none persisted.
    assert len(sweeps) == 3 * outcome.n_rounds
    assert _rows(tmp_path) == []


# ── Q-series: the acquisition sequence ───────────────────────────────────────

def test_ok_verdict_makes_the_scout_row_the_measurement(tmp_path):
    """No follow-up written; the script bytes are untouched; one adaptive row."""
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 500.0})
    plan = _plan(tmp_path, channels="18,19,20")
    ctx = _context(tmp_path, manager, plan)
    planner = ScoutPlanner(site="validation", actuate=True)

    scout, follow_up = V.measure_adaptive(ctx, planner, 18)
    path = ctx.script_path(18, "adaptive")
    after = Path(path).read_bytes()

    assert scout.eis_params["eis_scout_verdict"] == "ok"
    assert follow_up is None
    assert b"6475m" in after                    # still the baseline grid
    arms = [r.arm for r in _rows(tmp_path)]
    assert arms == [R.ARM_SCOUT]


def test_scout_sweep_is_persisted_even_when_superseded(tmp_path):
    """The scout row IS the control arm; discarding it discards the comparison."""
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 30.0})
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    scout, follow_up = V.measure_adaptive(
        ctx, ScoutPlanner(site="validation", actuate=True), 18)

    assert scout.eis_params["eis_scout_verdict"] == "extend_low"
    assert follow_up is not None
    assert sorted(r.arm for r in _rows(tmp_path)) == [
        R.ARM_FOLLOW_UP, R.ARM_SCOUT]


def test_follow_up_records_scout_sweep_seconds(tmp_path):
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 30.0})
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    scout, follow_up = V.measure_adaptive(
        ctx, ScoutPlanner(site="validation", actuate=True), 18)
    assert follow_up.eis_params["eis_scout_sweep_s"] == pytest.approx(
        scout.measurement_time_s)
    row = next(r for r in _rows(tmp_path) if r.arm == R.ARM_FOLLOW_UP)
    assert "eis_scout_sweep_s" in row.params


def test_max_follow_ups_one_takes_exactly_one(tmp_path):
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 30.0})
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    planner = ScoutPlanner(site="validation", actuate=True)
    built = []
    real = planner.build_follow_up
    planner.build_follow_up = lambda *a, **k: built.append(1) or real(*a, **k)

    V.measure_adaptive(ctx, planner, 18)
    assert len(built) == 1


def test_channels_are_interleaved_not_blocked(tmp_path):
    """R and B for channel k both precede R for channel k+1."""
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 30.0, 19: 30.0})
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19"))
    V.run_cells(ctx, ScoutPlanner(site="validation", actuate=True), [18, 19])

    order = [(r.channel, r.arm) for r in _rows(tmp_path)]
    first_ch19 = next(i for i, (ch, _) in enumerate(order) if ch == 19)
    assert all(ch == 18 for ch, _ in order[:first_ch19])
    assert order[0] == (18, R.ARM_REFERENCE)
    assert order[first_ch19] == (19, R.ARM_REFERENCE)


def test_order_alternate_flips_within_the_cell(tmp_path):
    from softae.core.eis_scout_scripts import ScoutPlanner

    manager = _manager({18: 500.0, 19: 500.0})
    plan = _plan(tmp_path, channels="18,19", order="alternate")
    ctx = _context(tmp_path, manager, plan)
    V.run_cells(ctx, ScoutPlanner(site="validation", actuate=True), [18, 19])

    order = [(r.channel, r.arm) for r in _rows(tmp_path)]
    assert order[0] == (18, R.ARM_REFERENCE)      # odd cell: reference first
    assert order[2] == (19, R.ARM_SCOUT)          # even cell: adaptive first


def test_drift_check_reruns_the_first_n_channels(tmp_path):
    manager = _manager()
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20",
                                            drift_check=2))
    V.drift_check(ctx, [18, 19, 20])
    ends = [r.channel for r in _rows(tmp_path) if r.arm == R.ARM_REFERENCE_END]
    assert ends == [18, 19]


def test_scripts_are_written_run_scoped_not_to_bare_tempdir(tmp_path):
    """The operator's GUI is assumed live and owns `%TEMP%/softae_testing.mscr`."""
    import tempfile

    manager = _manager()
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    path = Path(ctx.script_path(18, R.ARM_REFERENCE))
    assert path.parent == ctx.run_dir / "scripts"
    assert Path(tempfile.gettempdir()).resolve() != path.parent.resolve()
    assert path.name == "ch18_reference.mscr"
    V.measure_reference(ctx, 18, R.ARM_REFERENCE)
    assert path.exists()                           # a durable run artifact


# ── R-series: persistence and resume ─────────────────────────────────────────

def test_each_sweep_is_recorded_before_the_next_begins(tmp_path):
    """Nothing is batched: the row count after sweep k is k."""
    manager = _manager()
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    for index, channel in enumerate([18, 19, 20], start=1):
        V.measure_reference(ctx, channel, R.ARM_REFERENCE)
        assert len(_rows(tmp_path)) == index


def test_no_nan_is_ever_written_into_eis_params_json(tmp_path):
    """A bare `NaN` is invalid JSON and makes the row unreadable to JSON1."""
    manager = _manager({18: 5.0})               # open arc -> apex is NaN
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    grid = EISParams.from_preset("Quick")
    path = ctx.script_path(18, "adaptive")
    eis_run_mscrbuild(path, mux_ch=18, mVac=grid.mv_ac, f_hi=grid.f_hi,
                      f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
    V.persist(ctx, V.acquire(ctx, 18, {}, path), R.ARM_SCOUT)

    db = R.resolve_db(Path(tmp_path))
    raw = sqlite3.connect(db).execute(
        "SELECT eis_params_json FROM measurements").fetchone()[0]
    assert "NaN" not in raw
    json.loads(raw)                                        # strict JSON parses
    assert sqlite3.connect(db).execute(
        "SELECT json_extract(eis_params_json, '$.eis_validation_arm') "
        "FROM measurements").fetchone()[0] == R.ARM_SCOUT


def test_provenance_keys_are_all_written(tmp_path):
    manager = _manager()
    ctx = _context(tmp_path, manager, _plan(tmp_path, channels="18,19,20"))
    V.measure_reference(ctx, 18, R.ARM_REFERENCE)
    params = _rows(tmp_path)[0].params
    for key in ("eis_validation_name", "eis_validation_arm",
                "eis_validation_cell", "eis_validation_rh_sp_pct",
                "eis_validation_temp_sp_C", "eis_validation_hold_epoch",
                "eis_validation_hold_certified", "eis_validation_hold_excursion",
                "eis_validation_seq", "eis_validation_arc_state",
                "eis_validation_f_lo_hz", "eis_validation_band_min_decades"):
        assert key in params, key
    assert params["eis_validation_cell"] == "18:30:25:1"


def test_resume_refuses_on_a_changed_plan(tmp_path):
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "rs",
                   "--project", str(tmp_path), "--mock", "--min-treatment", "1",
                   "--drift-check", "0"]) == V.EXIT_OK
    # A different channel set is a different experiment.
    assert V.main(["run", "--channels", "18,19,20,21", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "rs",
                   "--project", str(tmp_path), "--mock", "--min-treatment", "1",
                   "--drift-check", "0", "--resume"]) == V.EXIT_FAILED


def test_resume_reuses_the_run_id_skips_complete_cells_and_bumps_the_epoch(tmp_path):
    argv = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--validation-name", "rs2",
            "--project", str(tmp_path), "--mock", "--min-treatment", "1",
            "--drift-check", "0"]
    assert V.main(argv) == V.EXIT_OK
    first = _rows(tmp_path, "rs2")
    assert first
    run_ids = {r.run_id for r in first}

    assert V.main(argv + ["--resume"]) == V.EXIT_OK
    after = _rows(tmp_path, "rs2")
    assert len(after) == len(first)                 # nothing complete re-measured
    assert {r.run_id for r in after} == run_ids     # start_run not called again

    spec = R.load_checkpoint(R.resolve_db(Path(tmp_path)), "rs2")
    assert spec["hold_epoch"] == 2                  # a park ended the condition


def test_resume_rediscards_a_partial_cell(tmp_path):
    """Half a pair yields no deviation, so the cell is re-run in full."""
    manager = _manager()
    plan = _plan(tmp_path, channels="18,19")
    ctx = _context(tmp_path, manager, plan)
    V.measure_reference(ctx, 18, R.ARM_REFERENCE)          # reference only
    done = V.complete_cells(ctx.data_store, plan)
    assert done == set()

    from softae.core.eis_scout_scripts import ScoutPlanner

    V.measure_adaptive(ctx, ScoutPlanner(site="validation", actuate=True), 18)
    assert V.complete_cells(ctx.data_store, plan) == {"18:30:25:1"}


def test_no_new_data_store_columns(tmp_path):
    """Provenance rides in eis_params_json: no DDL, no migration, no epoch."""
    from softae.core.data_store import DataStore

    before = _schema(DataStore(tmp_path / "a"))
    manager = _manager()
    ctx = _context(tmp_path / "b", manager, _plan(tmp_path, channels="18,19,20"))
    V.measure_reference(ctx, 18, R.ARM_REFERENCE)
    assert _schema(ctx.data_store) == before


def _schema(store):
    rows = store._conn.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
    return [tuple(r) for r in rows]


# ── M-series: the mock backend ───────────────────────────────────────────────

def test_mock_backend_reads_the_grid_back_from_the_mscr(tmp_path):
    from softae.drivers.mscr_library import eis_segmented_mscrbuild

    path = tmp_path / "seg.mscr"
    eis_segmented_mscrbuild(str(path), mux_ch=18,
                            segments=[(200_000, 20_000, 10), (20_000, 2_000, 24)])
    from softae.drivers.mscr_library import resolve_segments

    segments = M.parse_mscr_grid(path)
    # `resolve_segments` nudges a TOUCHING boundary one log-step inward, and
    # the parser reports what was EMITTED -- so this pins the emitter's real
    # output, not the planner's request. Reading the plan would have missed it.
    assert segments == resolve_segments(
        [(200_000, 20_000, 10), (20_000, 2_000, 24)])
    assert segments[1][0] < 20_000.0

    freq = M.grid_frequencies(segments)
    assert freq.size == 34
    assert freq[0] == pytest.approx(200_000.0)
    assert freq[-1] == pytest.approx(2_000.0)

    spectrum = M.MockRig(default_apex_hz=30.0).measure(18, path)
    assert spectrum.shape == (34, 5)
    assert spectrum[0, 0] == pytest.approx(200_000.0)


def test_mock_backend_inverts_the_frequency_literal(tmp_path):
    """The 1000x trap: `6475m` is 6.475 Hz, not 6475."""
    from softae.drivers.mscr_library import mscr_freq_literal, quantize_hz

    assert M.literal_to_hz("6475m") == pytest.approx(6.475)
    assert M.literal_to_hz("200000") == pytest.approx(200_000.0)
    for hz in (0.016, 0.228, 1.351, 6.475, 3912.0, 200_000.0):
        snapped = quantize_hz(hz)
        assert M.literal_to_hz(mscr_freq_literal(snapped)) == pytest.approx(snapped)
    with pytest.raises(ValueError):
        M.literal_to_hz("1.5")


def test_mock_grid_actually_changes_the_spectrum(tmp_path):
    """The whole reason this backend exists: a different script, a different sweep."""
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    rig = M.MockRig(default_apex_hz=30.0)
    spectra = {}
    for preset in ("Quick", "Extended"):
        grid = EISParams.from_preset(preset)
        path = tmp_path / f"{preset}.mscr"
        eis_run_mscrbuild(str(path), mux_ch=18, mVac=grid.mv_ac, f_hi=grid.f_hi,
                          f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
        spectra[preset] = rig.measure(18, path)
    assert spectra["Quick"].shape[0] == 27
    assert spectra["Extended"].shape[0] == 53
    assert spectra["Quick"][-1, 0] == pytest.approx(6.475, rel=1e-6)
    assert spectra["Extended"][-1, 0] == pytest.approx(1.351, rel=1e-6)


def test_shipped_mock_espico_would_have_faked_a_null(tmp_path):
    """**Why deliverable (c) exists.** Under `MockESPico` every Delta is exactly 0.

    It ignores the `.mscr` and returns a fixed 41-point 50 kHz -> 1 Hz sweep
    seeded from the channel, so a reference sweep and a scout sweep on the same
    channel are bit-identical and a mock report is a perfect, meaningless null.
    """
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mock_espico import MockESPico
    from softae.drivers.mscr_library import eis_run_mscrbuild

    shipped = MockESPico("pico2", {})
    paths = []
    for preset in ("Quick", "Extended"):
        grid = EISParams.from_preset(preset)
        path = tmp_path / f"{preset}.mscr"
        eis_run_mscrbuild(str(path), mux_ch=18, mVac=grid.mv_ac, f_hi=grid.f_hi,
                          f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)
        paths.append(str(path))

    a = shipped.sendscript_getdata(paths[0], str(tmp_path), 18)[0]
    b = shipped.sendscript_getdata(paths[1], str(tmp_path), 18)[0]
    assert a.shape == b.shape == (41, 5)
    assert np.array_equal(a, b)                    # bit-identical: Delta == 0

    grid_aware = M.GridAwareMockPico("pico2", {}, rig=M.MockRig())
    c = grid_aware.sendscript_getdata(paths[0], str(tmp_path), 18)[0]
    d = grid_aware.sendscript_getdata(paths[1], str(tmp_path), 18)[0]
    assert c.shape != d.shape                      # the grid reached the sweep


def test_mock_apex_places_a_cell_in_each_population(tmp_path):
    from softae.analysis.eis.arc import arc_closure
    from softae.core.eis_scripts import EISParams
    from softae.drivers.mscr_library import eis_run_mscrbuild

    plan = _plan(tmp_path)
    grid = EISParams.from_preset("Extended")
    path = tmp_path / "ref.mscr"
    eis_run_mscrbuild(str(path), mux_ch=18, mVac=grid.mv_ac, f_hi=grid.f_hi,
                      f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=grid.mv_dc)

    observed = {}
    for commanded in (5.0, 30.0, 200.0):
        rig = M.MockRig(default_apex_hz=commanded)
        spectrum = rig.measure(18, path)
        closure = arc_closure(spectrum[:, 0], spectrum[:, 4])
        observed[commanded] = H.classify_apex(
            float(closure.f_apex_interior_hz), plan)
    assert observed == {5.0: R.UNRESOLVED, 30.0: R.TREATMENT, 200.0: R.CONTROL}


def test_mock_drift_moves_sigma_at_the_injected_size():
    """Without a virtual clock a mock run finishes in ms and Delta_hold is 0."""
    rig = M.MockRig(default_apex_hz=30.0, drift_decades_per_hour=0.5,
                    virtual_s_per_sweep=3600.0)
    r_start = rig.r1_now(18)
    rig._sweeps = 1                                # one virtual hour later
    r_end = rig.r1_now(18)
    # sigma = K/R, so a +0.5 dec sigma drift is a -0.5 dec R drift.
    assert math.log10(r_start / r_end) == pytest.approx(0.5, abs=1e-9)


def test_mock_drift_is_seen_by_the_settle_gate(tmp_path):
    """A moving sample must not certify as settled."""
    manager = _manager(drift=3.0)
    plan = _plan(tmp_path, channels="18,19,20")
    plan.settle_max_hold_s = 1200.0
    ctx = _context(tmp_path, manager, plan)
    clock = H.VirtualClock()
    outcome = H.settle_phase(manager, plan, lambda ch: V._settle_sweep(ctx, ch),
                             sleep=clock.sleep, now=clock, min_hold_first_s=0.0)
    assert not outcome.certified


def test_r1_for_apex_inverts_the_apex_relation():
    from softae.drivers.mock_espico import _C0

    r1 = M.r1_for_apex(30.0)
    assert 1.0 / (2.0 * math.pi * r1 * _C0) == pytest.approx(30.0)
    with pytest.raises(ValueError):
        M.r1_for_apex(float("nan"))


def test_parse_apex_spec_accepts_a_scalar_and_a_per_channel_map():
    assert M.parse_apex_spec(None) == ({}, 30.0)
    assert M.parse_apex_spec("42") == ({}, 42.0)
    per_channel, default = M.parse_apex_spec("18:5,19:200")
    assert per_channel == {18: 5.0, 19: 200.0}
    assert default == 30.0


# ── Plan, projection, CLI ────────────────────────────────────────────────────

def test_projection_is_printed_before_anything_is_driven(tmp_path, capsys):
    assert V.cmd_run(_args(tmp_path, dry_run=True)) == V.EXIT_OK
    out = capsys.readouterr().out
    assert "Projected run" in out and "RESOLVING WINDOW" in out
    assert "nothing was heated and nothing was measured" in out


def test_fifteen_channel_projection_matches_the_measured_anchors(tmp_path):
    """The operator's set is 18-32: fifteen channels, all on pico2."""
    from softae.config.loader import pico_for_channel

    plan = _plan(tmp_path, channels=V.EXAMPLE_CHANNELS, drift_check=3)
    assert len(plan.channels) == 15
    assert {pico_for_channel(c) for c in plan.channels} == {"pico2"}

    projection = H.project(plan)
    assert projection.reference_s == pytest.approx(120.42 * 15)
    assert projection.scout_s == pytest.approx(17.50 * 15)
    assert projection.follow_up_s == pytest.approx(37.19 * 15)
    assert projection.drift_s == pytest.approx(120.42 * 3)
    assert projection.measurement_low_s / 60 == pytest.approx(40.5, abs=0.1)
    assert projection.measurement_high_s / 60 == pytest.approx(49.8, abs=0.1)


def test_longest_reference_widens_the_window_and_costs_more(tmp_path):
    extended = _plan(tmp_path, channels=V.EXAMPLE_CHANNELS)
    longest = _plan(tmp_path, channels=V.EXAMPLE_CHANNELS,
                    reference_preset="Longest")
    width = math.log10(extended.baseline_ok_hz / extended.ref_close_hz)
    wider = math.log10(longest.baseline_ok_hz / longest.ref_close_hz)
    assert width == pytest.approx(0.68, abs=0.01)
    assert wider == pytest.approx(1.45, abs=0.01)
    assert wider / width == pytest.approx(2.1, abs=0.1)
    assert (H.project(longest).reference_s / H.project(extended).reference_s
            == pytest.approx(4.3, abs=0.1))


def test_baseline_defaults_from_configuration_and_says_from_where(tmp_path):
    preset, source = V.resolve_baseline(None)
    assert preset == "Quick"
    assert "measurement_spec" in source
    assert V.resolve_baseline("Standard") == ("Standard", "--baseline")


def test_baseline_equal_to_the_reference_empties_the_window(tmp_path):
    """Pins why --baseline must not default to the reference preset."""
    plan = _plan(tmp_path, baseline="Extended")
    assert plan.ref_close_hz == pytest.approx(plan.baseline_ok_hz)


def test_fingerprint_moves_with_the_experiment_not_with_the_timeouts(tmp_path):
    base = _plan(tmp_path)
    assert _plan(tmp_path).fingerprint() == base.fingerprint()
    assert _plan(tmp_path, rh_approach_timeout_s=999).fingerprint() == (
        base.fingerprint())
    assert _plan(tmp_path, channels="18,19,20,21").fingerprint() != (
        base.fingerprint())


def test_setpoints_and_name_are_required():
    for missing in ("--rh-setpoint-pct", "--temp-setpoint-c", "--validation-name"):
        argv = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                "--temp-setpoint-c", "25", "--validation-name", "n"]
        index = argv.index(missing)
        del argv[index:index + 2]
        with pytest.raises(SystemExit):
            V.build_parser().parse_args(argv)


def test_report_subcommand_is_wired(tmp_path, capsys):
    assert V.main(["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
                   "--temp-setpoint-c", "25", "--validation-name", "rep",
                   "--project", str(tmp_path), "--mock", "--min-treatment", "1",
                   "--drift-check", "1"]) == V.EXIT_OK
    out = tmp_path / "report.json"
    assert V.main(["report", "--validation-name", "rep",
                   "--project", str(tmp_path), "--out", str(out),
                   "--min-treatment", "1"]) == V.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == R.REPORT_SCHEMA
    assert payload["mock"] is True
    assert payload["outcome"] == R.OUTCOME_WITHHELD
    assert "(MOCK)" in capsys.readouterr().out


def test_console_script_is_registered():
    text = Path(__file__).resolve().parents[1].joinpath(
        "pyproject.toml").read_text(encoding="utf-8")
    assert 'softae-eis-validate = "softae.tools.eis_validate:main"' in text


class TestRunRowFinalization:
    """`_enter_run`'s `start_run` had no matching `finish_run` on any exit path.

    A row left with `finished_at` NULL is byte-for-byte what a *crashed* run
    looks like, so a validation that completed perfectly was read at the next GUI
    launch as an unclean shutdown, offered a recovery park of the rig, and had
    its row permanently relabelled `interrupted`. Every path below goes through
    `--mock`, which is also the answer to "does the mock path finalize" -- it
    starts a real row in a real store and closes it the same way.
    """

    ARGV = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--mock", "--min-treatment", "1",
            "--drift-check", "0"]

    def _run(self, tmp_path, name="fin", *extra):
        return V.main(self.ARGV + ["--validation-name", name,
                                   "--project", str(tmp_path), *extra])

    def _outcomes(self, tmp_path):
        """How each run in the project ended, oldest first."""
        from softae.core.data_store import DataStore

        with DataStore(Path(tmp_path)) as ds:
            run_ids = [r[0] for r in ds._conn.execute(
                "SELECT run_id FROM experiments ORDER BY started_at")]
            return [ds.run_outcome(run_id) for run_id in run_ids]

    def test_a_completed_validation_closes_its_row_done(self, tmp_path):
        """`done`, not `error`: the `finally` catch-all must not overwrite it."""
        assert self._run(tmp_path) == V.EXIT_OK
        assert self._outcomes(tmp_path) == [{"status": "done", "finished": True}]

    def test_a_completed_validation_is_not_reported_as_an_unclean_shutdown(
            self, tmp_path):
        """The defect as the operator met it, pinned at its own surface."""
        from softae.core.data_store import DataStore

        self._run(tmp_path)
        with DataStore(Path(tmp_path)) as ds:
            assert ds.unfinished_runs() == []

    def test_a_refusal_closes_its_row_aborted(self, tmp_path, monkeypatch):
        """A refusal is a decision, not an accident -- and not an interruption.

        The approach timeout refuses *after* the row exists, which is the only
        reason this path needs the finalizer at all.
        """
        import softae.workflows.equilibration as EQ
        from softae.workflows.equilibration import ApproachOutcome

        monkeypatch.setattr(EQ, "approach_setpoint",
                            lambda read_pv, target, *, axis, **kw: ApproachOutcome(
                                axis=axis, target=target, reached=False,
                                elapsed_s=1.0, pv_final=99.0, n_samples=1))
        assert self._run(tmp_path, "ref") == V.EXIT_FAILED
        assert self._outcomes(tmp_path) == [{"status": "aborted", "finished": True}]

    def test_a_ctrl_c_closes_its_row_interrupted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(H.HoldWatch, "poll",
                            lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
        assert self._run(tmp_path, "kb") == V.EXIT_INTERRUPTED
        assert self._outcomes(tmp_path) == [
            {"status": "interrupted", "finished": True}]

    def test_an_unnamed_failure_closes_its_row_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(H.HoldWatch, "poll",
                            lambda self: (_ for _ in ()).throw(
                                RuntimeError("the mux stopped replying")))
        assert self._run(tmp_path, "err") == V.EXIT_FAILED
        assert self._outcomes(tmp_path) == [{"status": "error", "finished": True}]

    def test_a_finalization_failure_does_not_fail_the_run(self, tmp_path,
                                                          monkeypatch):
        """Recording *how* a run ended must not decide *whether* it succeeded."""
        from softae.core.data_store import DataStore

        def _boom(self, *_a, **_kw):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(DataStore, "finish_run", _boom)
        assert self._run(tmp_path) == V.EXIT_OK

    def test_a_resumed_validation_closes_the_row_it_re_entered(self, tmp_path):
        """`--resume` adopts the existing run_id, so it closes that same row."""
        assert self._run(tmp_path, "rf") == V.EXIT_OK
        assert self._run(tmp_path, "rf", "--resume") == V.EXIT_OK
        # One row, not two: the resume re-entered rather than starting a run.
        assert self._outcomes(tmp_path) == [{"status": "done", "finished": True}]

    def test_a_refused_resume_leaves_the_earlier_row_and_the_next_resume_intact(
            self, tmp_path):
        """A `ResumeMismatch` names a *different* plan's row. It stamps nothing.

        And finalizing never blocks resuming: `_enter_run` resumes off the
        campaign checkpoint and `_remaining_channels` off the measurements, so
        neither consults the run row's status.
        """
        assert self._run(tmp_path, "rf2") == V.EXIT_OK
        mismatch = V.main(
            ["run", "--channels", "18,19,20,21", "--rh-setpoint-pct", "30",
             "--temp-setpoint-c", "25", "--mock", "--min-treatment", "1",
             "--drift-check", "0", "--validation-name", "rf2",
             "--project", str(tmp_path), "--resume"])
        assert mismatch == V.EXIT_FAILED
        assert self._outcomes(tmp_path) == [{"status": "done", "finished": True}]
        assert self._run(tmp_path, "rf2", "--resume") == V.EXIT_OK

    def test_the_checkpoint_loop_state_is_deliberately_left_running(self, tmp_path):
        """Not an oversight, and not a second unclosed liveness claim.

        The checkpoint is a *resume point*, which outlives the process on
        purpose: a park ends the condition and `--resume` re-enters. The row
        that answers "did this process die" is `experiments`, and it is now
        closed. Pinned so nobody 'fixes' the checkpoint to a terminal value and
        breaks resume in the process.
        """
        from softae.core.data_store import DataStore

        assert self._run(tmp_path, "ls") == V.EXIT_OK
        with DataStore(Path(tmp_path)) as ds:
            checkpoint = ds.campaign_checkpoint(V.checkpoint_campaign("ls"))
            assert checkpoint["loop_state"] == "running"
            assert ds.unfinished_runs() == []


class _ParkResult:
    def summary(self) -> str:
        return "parked (test double)"
