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
from softae.tools import eis_validate_narrate as N
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


# ── Rig claim ────────────────────────────────────────────────────────────────

def _boom(*_a, **_kw):
    raise AssertionError("this must not be reached")


async def _boom_async(*_a, **_kw):
    raise AssertionError("no port may be opened on this path")


class TestRigClaim:
    """The tool holds the rig for exactly as long as it holds the ports.

    `gui/app.py` claims the rig when it opens ports; this tool claimed nothing,
    so an operator who opened the GUI mid-sweep got a window whose own claim
    *succeeded* -- the rig read free -- and which then connected onto the serial
    ports the run was mid-sweep on. `owner_pid` made such a run's **record**
    safe. Everything below is about its **ports**.

    None of these runs passes `--mock`, deliberately: `--mock` is the one mode
    that claims nothing, so a claim test written under it would assert against
    the exemption instead of the claim. The drivers are mocks all the same --
    `create_manager` is patched -- and nothing here opens a real port.
    """

    ARGV = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--min-treatment", "1",
            "--drift-check", "0", "--yes"]

    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch):
        """A per-test lock scope, and a park that commands nothing.

        Narrower than `conftest`'s session-wide redirect on purpose: a lock left
        behind by one test here must not be readable by the next one.
        """
        import softae.core.run_lock as RL
        import softae.core.safe_park as SP

        monkeypatch.setattr(RL, "DEFAULT_SCOPE", tmp_path / "rig_scope")
        monkeypatch.setattr(SP, "safe_park", lambda manager, **kw: _ParkResult())

    @pytest.fixture()
    def quiet_block(self, monkeypatch):
        """Stub everything downstream of the claim.

        `--mock` is what collapses this harness's pacing, and these runs are not
        `--mock`: an unstubbed `_establish_condition` is real minutes of thermal
        approach against a mock that moves on the wall clock. None of it is what
        these tests pin.
        """
        monkeypatch.setattr(V, "_establish_condition", lambda *_a, **_kw: None)
        monkeypatch.setattr(V, "run_cells", lambda *_a, **_kw: None)
        monkeypatch.setattr(V, "drift_check", lambda *_a, **_kw: None)
        monkeypatch.setattr(V, "_write_report", lambda *_a, **_kw: None)

    def _run(self, tmp_path, monkeypatch, *, manager=None, name="claim", extra=()):
        # The manager is built *before* the patch: `_manager` calls the very
        # `create_manager` being replaced, so patching first is a recursion.
        rig = _manager() if manager is None else manager
        monkeypatch.setattr("softae.drivers.factory.create_manager",
                            lambda **_kw: rig)
        return V.main(self.ARGV + ["--validation-name", name,
                                   "--project", str(tmp_path), *extra])

    @staticmethod
    def _outcomes(tmp_path):
        from softae.core.data_store import DataStore

        with DataStore(Path(tmp_path)) as ds:
            run_ids = [r[0] for r in ds._conn.execute(
                "SELECT run_id FROM experiments ORDER BY started_at")]
            return [ds.run_outcome(run_id) for run_id in run_ids]

    @staticmethod
    def _holder():
        from softae.core.run_lock import RunLock

        return RunLock(pid=424242, what="gui:desktop",
                       started_at="2026-08-19T14:02:00+00:00", host="bench")

    # -- the claim itself ----------------------------------------------------

    def test_a_run_claims_the_rig_naming_its_run_id(self, tmp_path, monkeypatch,
                                                    quiet_block):
        """`tool:eis-validate:<run_id>` -- third field FILLED, per the grammar.

        `gui:desktop` omits it because a window is not a run. This is a run, so a
        trailing bare colon would assert "there is a run id and it is blank".
        """
        from softae.core.run_lock import read_run_lock

        seen = {}

        def _peek(ctx, _plan, _channels):
            lock = read_run_lock()
            seen.update(what=lock.what, log_path=lock.log_path, run_id=ctx.run_id)

        monkeypatch.setattr(V, "_establish_condition", _peek)
        assert self._run(tmp_path, monkeypatch) == V.EXIT_OK
        assert seen["what"] == f"{V.CLAIM_KIND}:{seen['run_id']}"
        assert V.CLAIM_KIND == "tool:eis-validate"

    def test_the_claim_names_this_runs_own_run_directory(self, tmp_path,
                                                          monkeypatch, quiet_block):
        """The field a watcher discovers the run through -- and it is OURS.

        It was empty while this harness published nothing, on
        `claim_rig_session`'s own argument: a directory holding some *other*
        run's stream, offered as the live holder's, is a lie. The stream is now
        opened into this run's directory BEFORE the claim is taken, so by the
        time the lock file exists the directory it names already holds this
        run's `events.jsonl` with `run_started` in it. Both halves are asserted,
        because the ordering is what makes the field honest.
        """
        from softae.core.campaign_events import events_path
        from softae.core.run_lock import read_run_lock

        seen = {}

        def _peek(ctx, _plan, _channels):
            lock = read_run_lock()
            seen.update(log_path=lock.log_path, run_dir=str(ctx.run_dir))

        monkeypatch.setattr(V, "_establish_condition", _peek)
        assert self._run(tmp_path, monkeypatch) == V.EXIT_OK
        assert seen["log_path"] == seen["run_dir"]
        assert events_path(seen["log_path"]).exists()

    def test_the_claim_names_nothing_when_the_stream_could_not_be_opened(
            self, tmp_path, monkeypatch, quiet_block):
        """A directory a watcher would find nothing in is worse than no field.

        The objection that kept `log_path` empty is answered by the stream
        existing, so it comes straight back when the stream does not -- rather
        than advertising a run directory that carries no narration at all.
        """
        import softae.tools.eis_validate_narrate as NA
        from softae.core.run_lock import read_run_lock

        monkeypatch.setattr(NA, "open_narrator", lambda *_a, **_kw: None)
        seen = {}
        monkeypatch.setattr(
            V, "_establish_condition",
            lambda *_a, **_kw: seen.update(log_path=read_run_lock().log_path))
        assert self._run(tmp_path, monkeypatch) == V.EXIT_OK
        assert seen["log_path"] == ""

    def test_the_claim_is_taken_before_any_port_is_opened(self, tmp_path,
                                                          monkeypatch, quiet_block):
        """Ordering is the whole point: a claim after `connect_all` holds nothing."""
        order = []
        rig = _manager()
        connect = rig.connect_all

        async def _watched_connect():
            from softae.core.run_lock import read_run_lock

            order.append(("connect", read_run_lock() is not None))
            return await connect()

        rig.connect_all = _watched_connect
        assert self._run(tmp_path, monkeypatch, manager=rig) == V.EXIT_OK
        assert order == [("connect", True)]

    # -- giving it back ------------------------------------------------------

    def test_a_completed_run_gives_the_rig_back(self, tmp_path, monkeypatch,
                                                quiet_block):
        from softae.core.run_lock import read_run_lock

        assert self._run(tmp_path, monkeypatch) == V.EXIT_OK
        assert read_run_lock() is None

    def test_an_unnamed_failure_gives_the_rig_back(self, tmp_path, monkeypatch,
                                                   quiet_block):
        from softae.core.run_lock import read_run_lock

        monkeypatch.setattr(V, "run_cells", lambda *_a, **_kw: (
            _ for _ in ()).throw(RuntimeError("the mux stopped replying")))
        assert self._run(tmp_path, monkeypatch) == V.EXIT_FAILED
        assert read_run_lock() is None

    def test_a_ctrl_c_gives_the_rig_back(self, tmp_path, monkeypatch, quiet_block):
        from softae.core.run_lock import read_run_lock

        monkeypatch.setattr(V, "run_cells", lambda *_a, **_kw: (
            _ for _ in ()).throw(KeyboardInterrupt()))
        assert self._run(tmp_path, monkeypatch) == V.EXIT_INTERRUPTED
        assert read_run_lock() is None

    def test_a_refusal_inside_the_block_gives_the_rig_back(self, tmp_path,
                                                           monkeypatch, quiet_block):
        monkeypatch.setattr(V, "_establish_condition", lambda *_a, **_kw: (
            _ for _ in ()).throw(H.RefuseToStart("not enough TREATMENT cells")))
        from softae.core.run_lock import read_run_lock

        assert self._run(tmp_path, monkeypatch) == V.EXIT_FAILED
        assert read_run_lock() is None

    def test_the_claim_outlives_the_park_and_the_disconnect(self, tmp_path,
                                                            monkeypatch, quiet_block):
        """`rig_session`'s rule read to its end: release when the ports CLOSE.

        A claim dropped at the end of the measurement block would leave another
        process free to connect on top of a park still in progress -- the same
        defect this claim exists to close, in miniature.
        """
        import softae.core.safe_park as SP
        from softae.core.run_lock import read_run_lock

        order = []
        rig = _manager()
        disconnect = rig.disconnect_all

        async def _watched_disconnect():
            order.append(("disconnect", read_run_lock() is not None))
            return await disconnect()

        rig.disconnect_all = _watched_disconnect
        monkeypatch.setattr(SP, "safe_park", lambda manager, **kw: (
            order.append(("park", read_run_lock() is not None)) or _ParkResult()))
        assert self._run(tmp_path, monkeypatch, manager=rig) == V.EXIT_OK
        assert order == [("park", True), ("disconnect", True)]
        assert read_run_lock() is None

    # -- refusal -------------------------------------------------------------

    def test_a_foreign_holder_refuses_before_any_port_is_opened(
            self, tmp_path, monkeypatch, quiet_block, capsys):
        """The whole point of the peek: refuse without opening anything."""
        rig = _manager()
        rig.connect_all = _boom_async
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock",
                            lambda *_a, **_kw: self._holder())
        assert self._run(tmp_path, monkeypatch, manager=rig) == V.EXIT_FAILED
        out = capsys.readouterr().out
        assert "REFUSING TO START" in out and "424242" in out

    def test_a_foreign_holder_refusal_leaves_no_run_row_behind(
            self, tmp_path, monkeypatch, quiet_block):
        """Asked before the store is opened, so there is nothing to finalize."""
        rig = _manager()
        rig.connect_all = _boom_async
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock",
                            lambda *_a, **_kw: self._holder())
        assert self._run(tmp_path, monkeypatch) == V.EXIT_FAILED
        assert self._outcomes(tmp_path) == []

    def test_a_foreign_holder_refusal_preserves_an_existing_resume_point(
            self, tmp_path, monkeypatch, quiet_block):
        """Why the peek precedes `_enter_run` rather than following it.

        `_enter_run` REPLACES the campaign checkpoint on a non-`--resume`
        invocation, so a refusal taken one step later would destroy a
        validation's resume point on its way to saying "the rig is busy".
        """
        from softae.core.data_store import DataStore

        def _checkpoint():
            with DataStore(Path(tmp_path)) as ds:
                return ds.campaign_checkpoint(V.checkpoint_campaign("keep"))

        assert self._run(tmp_path, monkeypatch, name="keep") == V.EXIT_OK
        before = _checkpoint()

        rig = _manager()
        rig.connect_all = _boom_async
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock",
                            lambda *_a, **_kw: self._holder())
        assert self._run(tmp_path, monkeypatch, manager=rig, name="keep") \
            == V.EXIT_FAILED
        assert _checkpoint() == before
        assert self._outcomes(tmp_path) == [{"status": "done", "finished": True}]

    def test_a_claim_lost_in_the_race_closes_its_row_aborted(
            self, tmp_path, monkeypatch, quiet_block, capsys):
        """A holder arriving between the peek and the claim. Still no ports.

        `aborted`, for the reason every other refusal here is: the harness
        declined, nothing was interrupted. And the row must close -- an
        unfinished row is byte-for-byte what a crash looks like, which is what
        the next GUI launch would offer to park the rig over.
        """
        from softae.core.run_lock import RunLockHeld

        def _held(*_a, **_kw):
            raise RunLockHeld(self._holder())

        rig = _manager()
        rig.connect_all = _boom_async
        monkeypatch.setattr("softae.core.rig_session.claim_rig_session", _held)
        assert self._run(tmp_path, monkeypatch, manager=rig) == V.EXIT_FAILED
        assert self._outcomes(tmp_path) == [{"status": "aborted", "finished": True}]
        assert "REFUSING TO START" in capsys.readouterr().out

    def test_a_refused_resume_claims_nothing_and_opens_nothing(
            self, tmp_path, monkeypatch, quiet_block):
        """`ResumeMismatch` still precedes both, now by construction.

        `_enter_run` moved ahead of the claim so the claim could name a run id.
        The refusal it raises therefore lands *earlier* than it used to, never
        later: before a lock is taken and before a port is opened.
        """
        assert self._run(tmp_path, monkeypatch, name="rm") == V.EXIT_OK

        rig = _manager()
        rig.connect_all = _boom_async
        monkeypatch.setattr("softae.core.rig_session.acquire_run_lock", _boom)
        monkeypatch.setattr("softae.drivers.factory.create_manager",
                            lambda **_kw: rig)
        moved = ["run", "--channels", "18,19,20,21", "--rh-setpoint-pct", "30",
                 "--temp-setpoint-c", "25", "--min-treatment", "1",
                 "--drift-check", "0", "--yes", "--resume",
                 "--validation-name", "rm", "--project", str(tmp_path)]
        assert V.main(moved) == V.EXIT_FAILED

    # -- the exemption -------------------------------------------------------

    def test_a_mock_run_neither_claims_the_rig_nor_asks_who_holds_it(
            self, tmp_path, monkeypatch):
        """`--mock` claims nothing, so it cannot lock out a real run.

        And it is not refused one either: a simulated run that took no lock must
        not be turned away over somebody else's.

        The gate lives in `_rig_claim` rather than in `held_rig_session`'s own
        exemption because that exemption cannot see it: `session_is_simulated`
        recognises a mock by its class name's `Mock` prefix, and this tool's
        `--mock` installs `GridAwareMockPico`, `FastMockTempController` and
        `FastMockRHController`, none of which carries one. Measured, not assumed
        -- the assertion below is that measurement.
        """
        from softae.core.rig_session import session_is_simulated

        monkeypatch.setattr("softae.core.run_lock.acquire_run_lock", _boom)
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", _boom)
        monkeypatch.setattr("softae.core.rig_session.acquire_run_lock", _boom)
        assert V.main(TestRunRowFinalization.ARGV + [
            "--validation-name", "mk", "--project", str(tmp_path)]) == V.EXIT_OK
        # The reason the gate cannot be delegated, pinned so a later reader does
        # not "simplify" `_rig_claim` into an unconditional `held_rig_session`.
        assert session_is_simulated(_manager()) is False

    # -- [p37]'s warning, checked rather than assumed -------------------------

    def test_the_lock_is_never_read_through_read_run_lock(self):
        """`read_run_lock` reports THIS process's own claim as a holder.

        That is how the Calibration Launcher came to permanently disable its own
        launch button once the GUI started claiming `gui:desktop`. This tool read
        the lock nowhere before the claim landed, so it could not carry the bug;
        the claim introduced exactly one read, and it is the right predicate.
        Asserted over the parse tree, not the text, because the comment beside
        that import names both functions.
        """
        import ast

        tree = ast.parse(Path(V.__file__).read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert "foreign_run_lock" in imported
        assert "read_run_lock" not in imported


# ── S-series: the soak ───────────────────────────────────────────────────────

class _CountingWatch:
    """A ``HoldWatch`` stand-in that warns on nominated polls.

    Standing in rather than driving the real one, because ``HoldWatch``'s RH
    warn is grace-windowed over a 600 s trailing run: producing one from a mock
    driver would mean staging a sustained %RH excursion just to reach the branch
    the soak owns. What the soak actually consumes is two attributes, and those
    are what is faked.
    """

    def __init__(self, warn_on=(), raises=None):
        self.polls = 0
        self.excursion = False
        self._warn_on = set(warn_on)
        self._raises = raises

    def poll(self):
        self.polls += 1
        if self._raises is not None:
            raise self._raises
        self.excursion = (self.polls in self._warn_on) or ("all" in self._warn_on)


def _soak_plan(tmp_path, hours):
    return _plan(tmp_path, soak_h=hours)


class TestSoakPhase:
    """`--soak-h`: a floor on CONTINUOUS time at condition before sweep one.

    The settle gate certifies that the RIG stopped moving. A film taking up
    water at a new RH moves on its own, much longer timescale and can hold a
    locally flat trailing window while still hours from equilibrium; measuring
    there confounds the scout-vs-reference comparison and would surface as trend
    in the drift re-checks rather than as noise.
    """

    ARGV = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--mock", "--min-treatment", "1",
            "--drift-check", "0"]

    # -- the default is nothing -----------------------------------------------

    def test_soak_default_is_zero_and_waits_nothing(self, tmp_path):
        """Every invocation written before the soak existed must be unchanged."""
        assert H.DEFAULT_SOAK_S == 0.0
        assert _plan(tmp_path).soak_s == 0.0

        clock = H.VirtualClock()
        slept = []
        outcome = H.soak_phase(
            _plan(tmp_path), _CountingWatch(warn_on=("all",)),
            established_at=clock(), sleep=lambda s: slept.append(s), now=clock)
        assert slept == []                      # never sleeps, never polls
        assert outcome.waited_s == 0.0
        assert outcome.restarts == 0

    def test_soak_zero_prints_no_soak_lines_at_all(self, tmp_path, capsys):
        assert V.main(self.ARGV + ["--validation-name", "sk0",
                                   "--project", str(tmp_path)]) == V.EXIT_OK
        assert "[soak  ]" not in capsys.readouterr().out

    # -- when the clock starts ------------------------------------------------

    def test_soak_clock_starts_at_the_condition_not_at_launch(self, tmp_path):
        """The approach buys nothing; the settle rounds, at condition, count.

        The rejected alternative is starting the clock where ``soak_phase`` sits
        in the sequence -- after the settle gate -- which charges the operator
        twice for time the sample has already spent at the new RH.
        """
        clock = H.VirtualClock()
        clock.t = 5000.0                # a long RH descent, NOT at condition
        established = clock()
        clock.t += 600.0                # 10 min of settle rounds, AT condition

        plan = _soak_plan(tmp_path, 0.5)                    # 1800 s
        outcome = H.soak_phase(plan, None, established_at=established,
                               sleep=clock.sleep, now=clock)
        assert outcome.settle_credit_s == pytest.approx(600.0)
        assert outcome.waited_s == pytest.approx(1200.0)    # only the remainder
        assert outcome.soaked_s == pytest.approx(1800.0)
        # The 5000 s spent approaching bought exactly none of the soak.
        assert clock() - 5000.0 == pytest.approx(1800.0)

    def test_soak_already_covered_by_a_long_settle_waits_nothing(self, tmp_path):
        """Time at condition is time at condition, whatever ran during it."""
        clock = H.VirtualClock()
        established = clock()
        clock.t += 7200.0                       # a settle that ran to two hours
        outcome = H.soak_phase(_soak_plan(tmp_path, 1.0), None,
                               established_at=established,
                               sleep=clock.sleep, now=clock)
        assert outcome.waited_s == 0.0
        assert outcome.soaked_s == pytest.approx(7200.0)

    def test_soak_completes_before_the_first_spectrum_is_acquired(
            self, tmp_path, capsys):
        """The whole point: no persisted sweep until the soak has elapsed."""
        assert V.main(self.ARGV + ["--validation-name", "sk1", "--soak-h", "1",
                                   "--project", str(tmp_path)]) == V.EXIT_OK
        out = capsys.readouterr().out
        assert out.index("[soak  ] complete") < out.index("] ch18  R ")
        # ...and after the settle gate, which is what licenses holding at all.
        assert out.index("[watch ]") < out.index("[soak  ] holding")

    # -- what happens during it -----------------------------------------------

    def test_soak_polls_the_hold_watch_rather_than_sitting_idle(self, tmp_path):
        """A soak that drifted out of tolerance and measured anyway would be
        worse than no soak: it attaches a CLAIM of equilibration to a sample
        that had been moved."""
        clock = H.VirtualClock()
        watch = _CountingWatch()
        H.soak_phase(_soak_plan(tmp_path, 0.5), watch, established_at=clock(),
                     sleep=clock.sleep, now=clock)
        assert watch.polls == 1800 / H.SOAK_POLL_INTERVAL_S

    def test_soak_warn_excursion_restarts_the_clock(self, tmp_path):
        """Continuity is the asserted quantity, so an excursion resets it."""
        clock = H.VirtualClock()
        watch = _CountingWatch(warn_on=(3,))            # a blip 90 s in
        outcome = H.soak_phase(_soak_plan(tmp_path, 300 / 3600), watch,
                               established_at=clock(), sleep=clock.sleep,
                               now=clock)
        assert outcome.restarts == 1
        assert outcome.waited_s == pytest.approx(390.0)  # the 90 s is discarded
        assert outcome.soaked_s == pytest.approx(300.0)

    def test_soak_sustained_excursions_past_the_ceiling_refuse(self, tmp_path):
        """A condition that cannot hold through the soak will not hold through
        the measurement block, so the harness refuses instead of measuring."""
        clock = H.VirtualClock()
        watch = _CountingWatch(warn_on=("all",))
        with pytest.raises(H.RefuseToStart) as excinfo:
            H.soak_phase(_soak_plan(tmp_path, 300 / 3600), watch,
                         established_at=clock(), sleep=clock.sleep, now=clock)
        assert "restart" in str(excinfo.value)
        # Bounded, not unbounded: it gave up rather than waiting for ever.
        assert clock() <= 300 * H.SOAK_CEILING_FACTOR + H.SOAK_POLL_INTERVAL_S

    def test_soak_fault_parks_before_any_spectrum_is_recorded(
            self, tmp_path, monkeypatch):
        """`fault` reuses the runner's existing park path -- no new exit."""
        from softae.errors import SafetyError

        parks = []
        import softae.core.safe_park as SP

        monkeypatch.setattr(SP, "safe_park", lambda manager, **kw:
                            parks.append(kw) or _ParkResult())
        monkeypatch.setattr(H.HoldWatch, "poll", lambda self:
                            (_ for _ in ()).throw(SafetyError("RH ran away")))
        assert V.main(self.ARGV + ["--validation-name", "skf", "--soak-h", "1",
                                   "--project", str(tmp_path)]) == V.EXIT_FAILED
        assert parks and parks[-1]["retract_head"] is None
        assert _rows(tmp_path, "skf") == []             # nothing was measured

    # -- --settle off ---------------------------------------------------------

    def test_soak_with_settle_off_waits_the_whole_duration(self, tmp_path, capsys):
        """Settling off removes the only evidence anything stopped moving, which
        makes the soak MORE meaningful, not less -- it is then the sole thing
        between the approach and the first sweep. It needs no special case: a
        disabled settle returns instantly, so the credit is ~0."""
        assert V.main(self.ARGV + ["--validation-name", "sko", "--soak-h", "1",
                                   "--settle", "off",
                                   "--project", str(tmp_path)]) == V.EXIT_OK
        out = capsys.readouterr().out
        assert "[soak  ] holding at condition for 60 min; 0.0 min" in out
        assert "[soak  ] complete: 60.0 min" in out

    # -- resume ---------------------------------------------------------------

    def test_soak_re_runs_on_resume_at_the_next_hold_epoch(
            self, tmp_path, capsys, monkeypatch):
        """A park ends the condition, so the sample re-equilibrates at ambient
        in between and the soak is owed again. No flag skips it.

        The first run is made to lose its last cell, because a resume with
        nothing left to measure returns before `_establish_condition` -- and a
        soak with no sweep to license would be an hour spent on nothing.
        """
        argv = self.ARGV + ["--validation-name", "skr", "--soak-h", "1",
                            "--project", str(tmp_path),
                            "--max-consecutive-failures", "1"]
        real = V.measure_adaptive
        monkeypatch.setattr(
            V, "measure_adaptive", lambda ctx, planner, channel:
            (_ for _ in ()).throw(RuntimeError("cell lost")) if channel == 20
            else real(ctx, planner, channel))
        assert V.main(argv) == V.EXIT_FAILED
        monkeypatch.undo()

        assert V.main(argv + ["--resume"]) == V.EXIT_OK
        out = capsys.readouterr().out
        assert "1 remaining" in out
        assert out.count("[soak  ] complete") == 2
        spec = R.load_checkpoint(R.resolve_db(Path(tmp_path)), "skr")
        assert spec["hold_epoch"] == 2

    def test_a_resume_that_forgets_the_soak_is_refused(self, tmp_path):
        """Otherwise an unsoaked sample is measured into a soaked dataset."""
        argv = self.ARGV + ["--validation-name", "skm", "--project", str(tmp_path)]
        assert V.main(argv + ["--soak-h", "1"]) == V.EXIT_OK
        assert V.main(argv + ["--resume"]) == V.EXIT_FAILED

    # -- the projection, the record, the fingerprint --------------------------

    def test_projection_carries_the_soak_as_its_own_row(self, tmp_path):
        """The operator types 'yes' against this table; a soak absent from it is
        an hour the projection lied about."""
        plan = _soak_plan(tmp_path, 4)
        text = H.render_projection(plan, H.project(plan))
        assert "soak       hold at condition (--soak-h)" in text
        assert "0 - 240 min" in text                     # what it adds
        assert "480 min" in text                         # the restart ceiling
        assert "SOAK 4.00 h" in text
        # The row is there at zero too: no soak is itself worth seeing.
        zero = H.render_projection(_plan(tmp_path), H.project(_plan(tmp_path)))
        assert "soak       hold at condition (--soak-h)" in zero
        assert "SOAK" not in zero

    def test_the_persisted_plan_carries_the_soak(self, tmp_path):
        """`as_dict` -> the campaign checkpoint's `spec_json` -> the reporter."""
        assert V.main(self.ARGV + ["--validation-name", "skp", "--soak-h", "2",
                                   "--project", str(tmp_path)]) == V.EXIT_OK
        spec = R.load_checkpoint(R.resolve_db(Path(tmp_path)), "skp")
        assert spec["soak_s"] == 7200.0

    def test_soak_moves_the_fingerprint_but_the_ceilings_still_do_not(
            self, tmp_path):
        """Ceiling vs floor, not seconds vs not: `settle_max_hold_s` bounds how
        long the harness waits for a criterion, while the soak sets the sample's
        state at measurement time. A film 30 min into an RH step and the same
        film 6 h in are different specimens."""
        base = _plan(tmp_path)
        assert _soak_plan(tmp_path, 4).fingerprint() != base.fingerprint()
        assert _soak_plan(tmp_path, 4).fingerprint() != (
            _soak_plan(tmp_path, 6).fingerprint())
        assert _plan(tmp_path, settle_max_hold_s=99).fingerprint() == (
            base.fingerprint())

    def test_a_negative_soak_is_refused_before_anything_is_heated(self, tmp_path):
        with pytest.raises(H.RefuseToStart) as excinfo:
            H.validate_plan(_soak_plan(tmp_path, -1))
        assert "negative" in str(excinfo.value)
        assert V.main(self.ARGV + ["--validation-name", "skn", "--soak-h", "-1",
                                   "--project", str(tmp_path)]) == V.EXIT_FAILED
        assert _rows(tmp_path, "skn") == []

    def test_the_confirmation_banner_discloses_the_soak(self, tmp_path, capsys):
        """`confirm_thermal` builds its hours from an `EquilibrationConfig`,
        which has no soak in it, so the one screen the operator commits on would
        otherwise omit them."""
        plan = _soak_plan(tmp_path, 3)
        assert V._confirm(plan, H.project(plan), assume_yes=True) is True
        out = capsys.readouterr().out
        assert "SOAK is held at condition before the first sweep" in out
        assert "3.00 h" in out and "6.00 h" in out

    def test_soak_observers_are_called_and_a_failing_one_cannot_refuse_the_soak(
            self, tmp_path):
        """Monitoring is a convenience; it must never be why a soak refuses.

        Both hooks are exercised on the same run: a blip at poll 3 fires
        `on_restart`, every poll fires `on_poll`, and both are made to raise.
        """
        clock = H.VirtualClock()
        seen = {"poll": 0, "restart": 0}

        def _boom_poll(*_a):
            seen["poll"] += 1
            raise RuntimeError("the watcher's disk is full")

        def _boom_restart(*_a):
            seen["restart"] += 1
            raise RuntimeError("still full")

        outcome = H.soak_phase(
            _soak_plan(tmp_path, 300 / 3600), _CountingWatch(warn_on=(3,)),
            established_at=clock(), sleep=clock.sleep, now=clock,
            on_poll=_boom_poll, on_restart=_boom_restart)
        assert outcome.restarts == 1                    # the soak completed
        assert seen["restart"] == 1
        assert seen["poll"] > 1

    def test_the_soak_flag_is_in_hours_not_seconds(self, tmp_path):
        """Of the two 60x slips only one is caught: 14400-for-minutes shows up in
        the projection as 240 h and is declined, while 2-for-hours-as-seconds
        would silently produce a run with no soak that looks entirely correct.

        `--settle-min-hold-s` was rejected for a second reason: `settle_phase`
        already takes a `min_hold_first_s`, which IS the settle gate's minimum
        hold, so the symmetric name would have named an existing parameter and
        meant something else.
        """
        from softae.analysis.equilibration import DEFAULT_MIN_HOLD_FIRST_S

        assert DEFAULT_MIN_HOLD_FIRST_S > 0     # the name that was already taken
        assert _soak_plan(tmp_path, 1).soak_s == 3600.0
        assert _soak_plan(tmp_path, 0.25).soak_s == 900.0
        with pytest.raises(SystemExit):
            V.build_parser().parse_args(
                ["run", "--channels", "18", "--rh-setpoint-pct", "30",
                 "--temp-setpoint-c", "25", "--validation-name", "n",
                 "--soak-s", "3600"])


# ── N-series: narration, liveness and the published conditions ───────────────

class _Beats:
    """A ``wait`` that lets the beat loop run a fixed number of times.

    Injected in place of the stop event's ``wait`` so the loop body is driven
    deterministically. A test that really slept for three heartbeats would spend
    90 s proving something a counter proves in microseconds.
    """

    def __init__(self, times: int) -> None:
        self.calls: list[float] = []
        self._times = times

    def __call__(self, seconds: float) -> bool:
        self.calls.append(seconds)
        return len(self.calls) > self._times


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _stream(run_dir):
    """Every record a real watcher would have, through the real reader."""
    from softae.core.campaign_events import read_events

    events, _cursor = read_events(run_dir)
    return events


def _only_run_dir(tmp_path):
    runs = sorted((Path(tmp_path) / "runs").iterdir())
    assert len(runs) == 1, runs
    return runs[0]


class TestRunNarration:
    """A validation holds the rig for hours and used to publish nothing at all.

    An operator who opened the GUI at hour two got a banner saying the rig was
    busy and no temperature, no RH and no progress. The run now writes the two
    sidecars a campaign writes -- ``events.jsonl`` and ``conditions.json`` --
    beside the run, and the rig claim's ``log_path`` names the directory so a
    watcher can find them.

    Everything below goes through ``--mock``. Nothing here actuates anything,
    and nothing here sleeps.
    """

    ARGV = ["run", "--channels", "18,19,20", "--rh-setpoint-pct", "30",
            "--temp-setpoint-c", "25", "--mock", "--min-treatment", "1",
            "--drift-check", "1"]

    def _run(self, tmp_path, name="nar", *extra):
        return V.main(self.ARGV + ["--validation-name", name,
                                   "--project", str(tmp_path), *extra])

    # -- the stream exists, and the real reader can follow it ------------------

    def test_a_run_publishes_a_stream_into_its_own_run_directory(self, tmp_path):
        from softae.core.campaign_events import events_path

        assert self._run(tmp_path) == V.EXIT_OK
        assert events_path(_only_run_dir(tmp_path)).exists()

    def test_a_watcher_using_the_real_reader_follows_the_phase_spine(self, tmp_path):
        """`read_events`, not a bespoke tailer -- and the transitions are a chain.

        `old` is filled from the narration's own last phase rather than by the
        caller, so the spine is continuous by construction: every `state`
        record's `old` is the previous one's `new`, on every exit path including
        the ones reached from inside a `finally`.
        """
        assert self._run(tmp_path) == V.EXIT_OK
        events = _stream(_only_run_dir(tmp_path))

        assert events[0]["type"] == "run_started"
        assert events[-1]["type"] == "run_finished"
        assert events[-1]["status"] == "done"

        states = [e for e in events if e["type"] == "state"]
        assert [s["new"] for s in states] == [
            "approach", "settle", "soak", "cells", "drift", "report",
            "park", "finished"]
        assert states[0]["old"] == "starting"
        for before, after in zip(states, states[1:]):
            assert after["old"] == before["new"]

    def test_a_watcher_polling_with_a_cursor_gets_each_record_once(self, tmp_path):
        """The reader's own contract, exercised against this tool's stream."""
        from softae.core.campaign_events import EventCursor, read_events

        run_dir = tmp_path / "runs" / "poll"
        narration = N.open_narration(run_dir)
        narration.record("run_started", run_id="poll")
        first, cursor = read_events(run_dir)
        assert [e["type"] for e in first] == ["run_started"]
        assert isinstance(cursor, EventCursor)

        narration.state(N.PHASE_APPROACH)
        second, cursor = read_events(run_dir, cursor=cursor)
        assert [e["type"] for e in second] == ["state"]

        third, _ = read_events(run_dir, cursor=cursor)
        assert third == []                        # nothing new, nothing repeated
        narration.close()

    def test_the_phase_records_alone_would_read_as_stale(self, tmp_path):
        """Why the beat exists at all, stated as an assertion.

        `liveness` counts ANY record, so the phase spine does feed a watcher --
        but one `Extended` reference sweep is ~120 s and a soak is hours of one
        line every 30 s, against a staleness rule of three beats (90 s). Without
        a beat a perfectly healthy run reads STALE.
        """
        from softae.core.campaign_events import (
            DEFAULT_HEARTBEAT_S,
            LIVENESS_STALE,
            liveness,
        )

        base = 1_700_000_000.0
        events = [{"ts": _iso(base), "type": "state", "old": "settle",
                   "new": "cells"}]
        assert liveness(events, now=base + 3 * DEFAULT_HEARTBEAT_S) == LIVENESS_STALE

    def test_liveness_stays_live_across_a_simulated_thirty_minute_sweep(
            self, tmp_path, monkeypatch):
        """The case the thread exists for: 30 min of blocking acquisition.

        The clock is virtual and the stamps are taken from it, so the whole
        half-hour costs microseconds. What is exercised is the real
        `CampaignNarrator.beat` and the real `liveness`.
        """
        from softae.core import campaign_events as CE

        clock = H.VirtualClock(start=1_700_000_000.0)
        monkeypatch.setattr(CE, "_stamp", lambda: _iso(clock()))

        run_dir = tmp_path / "runs" / "long"
        narration = N.RunNarration(
            run_dir, narrator=CE.CampaignNarrator(run_dir, heartbeat_s=0,
                                                  now=clock))
        narration.state(N.PHASE_CELLS)             # the last record before silence
        verdicts = set()
        for _ in range(int(1800 / CE.DEFAULT_HEARTBEAT_S)):
            clock.sleep(CE.DEFAULT_HEARTBEAT_S)
            narration.beat()
            verdicts.add(CE.liveness(_stream(narration.run_dir), now=clock()))
        narration.close()

        assert CE.LIVENESS_STALE not in verdicts   # never stale, for 30 minutes
        beats = [e for e in _stream(narration.run_dir) if e["type"] == "heartbeat"]
        assert len(beats) == 60
        # And the beat says what it is waiting on, not merely that it is there.
        assert beats[-1]["phase"] == "state"
        assert beats[-1]["phase_age_s"] == pytest.approx(1800.0, abs=1.0)

    # -- the beat is a thread, because this runner is synchronous --------------

    def test_the_narrator_is_opened_with_its_asyncio_heartbeat_disabled(
            self, tmp_path):
        """`start_heartbeat` schedules an `asyncio` task and `cmd_run` is sync.

        `asyncio.run(connect_all())` tears its loop down before the first sweep,
        so a task scheduled onto it would never run again -- exactly across the
        window that needs narrating. The loop-bound path is therefore disabled at
        the source (`heartbeat_s=0` is that class's own switch for it) rather
        than merely left uncalled, and the synchronous `beat()` is driven from a
        thread instead.
        """
        narration = N.open_narration(tmp_path / "runs" / "async")
        assert narration._narrator.heartbeat_s == 0
        narration._narrator.start_heartbeat()      # a no-op, not a scheduled task
        assert narration._narrator._task is None
        narration.close()

    def test_the_heartbeat_thread_beats_on_cadence_and_is_joined_on_close(
            self, tmp_path):
        from softae.core.campaign_events import DEFAULT_HEARTBEAT_S

        wait = _Beats(3)
        narration = N.open_narration(tmp_path / "runs" / "beat", wait=wait)
        narration.start()
        thread = narration._thread
        narration.close()

        assert wait.calls[:3] == [DEFAULT_HEARTBEAT_S] * 3
        assert thread is not None and not thread.is_alive()
        assert len([e for e in _stream(narration.run_dir)
                    if e["type"] == "heartbeat"]) == 3

    def test_a_beat_and_a_record_from_two_threads_do_not_tear_a_line(
            self, tmp_path):
        """`CampaignNarrator._append` takes a `threading.Lock` and says why.

        The thread is the caller that needed it. Every line must still parse,
        which is what the reader silently dropping unparseable lines would
        otherwise hide.
        """
        import threading

        narration = N.open_narration(tmp_path / "runs" / "race")
        done = threading.Event()

        def _narrate():
            for index in range(200):
                narration.progress(N.PHASE_CELLS, index, 200)
            done.set()

        writer = threading.Thread(target=_narrate)
        writer.start()
        while not done.is_set():
            narration.beat()
        writer.join()
        narration.close()

        text = narration.events_path.read_text(encoding="utf-8")
        lines = [line for line in text.split("\n") if line.strip()]
        for line in lines:
            json.loads(line)                        # every one, not most of them
        assert len(lines) >= 201

    # -- never fails the run --------------------------------------------------

    def test_a_narrator_that_fails_to_open_does_not_fail_the_run(
            self, tmp_path, monkeypatch):
        """`None` means run unnarrated, never 'do not run'."""
        import softae.tools.eis_validate_narrate as NA

        monkeypatch.setattr(NA, "open_narrator", lambda *_a, **_kw: None)
        assert self._run(tmp_path, "blind") == V.EXIT_OK
        assert _rows(tmp_path, "blind")                    # the science landed
        assert not (_only_run_dir(tmp_path) / "events.jsonl").exists()
        assert not (_only_run_dir(tmp_path) / "conditions.json").exists()

    def test_an_unwritable_stream_does_not_fail_the_run(self, tmp_path,
                                                        monkeypatch):
        """A disk that fills mid-run is the realistic version of the same thing."""
        from softae.core.campaign_events import CampaignNarrator

        def _boom(self, record):
            raise OSError("no space left on device")

        monkeypatch.setattr(CampaignNarrator, "_append", _boom)
        assert self._run(tmp_path, "full") == V.EXIT_OK
        assert _rows(tmp_path, "full")

    def test_an_inert_narration_leaves_every_call_site_a_plain_method_call(
            self, tmp_path):
        """A `RunContext` built directly narrates nothing and works unchanged.

        `None` would have meant a null check at a dozen sites inside a run block
        three sessions are writing, and the one that gets forgotten is an
        AttributeError raised out of a harness that drives a heater.
        """
        ctx = _context(tmp_path, _manager(), _plan(tmp_path))
        assert isinstance(ctx.narration, N.RunNarration)
        assert ctx.narration.live is False
        assert ctx.narration.log_path == ""
        ctx.narration.state(N.PHASE_SOAK)
        ctx.narration.progress(N.PHASE_CELLS, 1, 3)
        ctx.narration.beat()
        assert not ctx.narration.events_path.exists()
        assert not ctx.narration.conditions_path.exists()

    # -- narration, never scientific record -----------------------------------

    def test_the_stream_carries_no_scientific_value(self, tmp_path):
        """A record here is a claim about what the run was DOING, never about
        what it found.

        Sweep results, fits and sigma are in the DataStore, which is the only
        thing that can say what they mean. Asserted over the raw bytes rather
        than over parsed keys, so a value smuggled into a nested payload is
        caught too -- and paired with a positive check that the science really
        did land somewhere.
        """
        assert self._run(tmp_path, "sci") == V.EXIT_OK
        assert _rows(tmp_path, "sci")                     # it landed in the store

        text = (_only_run_dir(tmp_path) / "events.jsonl").read_text(
            encoding="utf-8").lower()
        for forbidden in ("r1", "sigma", "z_real", "z_imag", "frequency",
                          "apex", "eis_params", "fit", "ohms", "impedance",
                          "spectrum", "arc_state", "scout_verdict", "deviation"):
            assert forbidden not in text, forbidden

    def test_progress_records_carry_counts_and_never_readings(self, tmp_path):
        """An approach record says which axis and how long, not what PV it hit.

        A PV is a reading, and readings belong in `conditions` rows and in
        `conditions.json` -- both of which this run already writes.
        """
        assert self._run(tmp_path) == V.EXIT_OK
        approach = [e for e in _stream(_only_run_dir(tmp_path))
                    if e["type"] == "progress" and e["phase"] == "approach"]
        assert [e["axis"] for e in approach] == ["temperature", "rh"]
        assert all(set(e) == {"ts", "seq", "type", "phase", "done", "total",
                              "axis", "elapsed_s", "attempts"} for e in approach)

    # -- conditions.json ------------------------------------------------------

    def test_conditions_are_published_from_the_capture_the_run_already_takes(
            self, tmp_path, monkeypatch):
        """ONE read, two consumers -- the `conditions` row and the sidecar.

        Standing up `ConditionsPublisher` instead would have put a second reader
        on the serial lock the measurement is using, which is the cost its own
        docstring names. Pinned by counting reads against sweeps: the sidecar
        adds none.
        """
        import softae.core.conditions_capture as CC

        reads = []
        real = CC.read_environment
        monkeypatch.setattr(
            CC, "read_environment",
            lambda manager: reads.append(1) or real(manager))

        assert self._run(tmp_path, "cap") == V.EXIT_OK
        # No soak, so every read is a `persist` read: exactly one per sweep.
        assert len(reads) == len(_rows(tmp_path, "cap"))

        payload = json.loads(
            (_only_run_dir(tmp_path) / "conditions.json").read_text(
                encoding="utf-8"))
        assert set(payload) == {"started_at", "completed_at", "read_ms", "env",
                                "skipped_beats"}
        assert payload["env"]["stage_temp_sp_C"] == pytest.approx(25.0)
        assert payload["env"]["rh_sp_pct"] == pytest.approx(30.0)

    def test_the_published_slot_is_the_shape_the_shipped_reader_takes(
            self, tmp_path):
        """`ConditionsFileSource` reads a campaign's slot and must not learn a
        second shape, so this one is key-for-key `ConditionsPublisher`'s."""
        from softae.core.campaign_events import ConditionsPublisher

        narration = N.open_narration(tmp_path / "runs" / "shape")
        narration.capture(_manager())
        published = ConditionsPublisher(tmp_path / "runs" / "other").payload()
        assert set(narration.payload()) == set(published)
        assert set(narration.payload()["env"]) == set(published["env"])
        narration.close()

    def test_a_beat_republishes_the_slot_so_a_long_sweep_is_visibly_stale(
            self, tmp_path):
        """The publisher's own rule, kept: visibly stale rather than silently
        stale.

        A 30-minute sweep must not leave a frozen file, because a frozen file is
        ambiguous between 'the numbers are old' and 'the publisher died'. So the
        slot is rewritten on every beat with `skipped_beats` incremented and
        `completed_at` UNCHANGED -- mtime advances, the numbers date themselves,
        and `ConditionsFileSource` correctly renders `--` rather than showing a
        two-minute-old temperature as current.
        """
        narration = N.open_narration(tmp_path / "runs" / "stale")
        narration.capture(_manager())
        fresh = narration.payload()
        assert fresh["skipped_beats"] == 0

        for _ in range(4):
            narration.beat()
        after = json.loads(
            narration.conditions_path.read_text(encoding="utf-8"))
        narration.close()

        assert after["skipped_beats"] == 4
        assert after["completed_at"] == fresh["completed_at"]
        assert after["env"] == fresh["env"]

    def test_a_failed_read_publishes_nulls_rather_than_a_stale_value(
            self, tmp_path, monkeypatch):
        """An old number wearing a fresh stamp is the one lie this file must
        not tell."""
        import softae.core.conditions_capture as CC

        narration = N.open_narration(tmp_path / "runs" / "nulls")
        narration.capture(_manager())
        assert narration.payload()["env"]["rh_pv_pct"] is not None

        monkeypatch.setattr(CC, "read_environment", lambda _m: (
            _ for _ in ()).throw(RuntimeError("the bus is gone")))
        env = narration.capture(_manager())
        narration.close()
        # The five keys survive -- `record_conditions(**env)` still has a shape to
        # splat -- but every value is unknown rather than yesterday's.
        assert set(env) == set(narration.payload()["env"])
        assert all(value is None for value in env.values())
        assert all(value is None for value in narration.payload()["env"].values())

    # -- the soak, which is the longest silence --------------------------------

    def test_the_soak_publishes_the_rig_on_its_own_poll_cadence(self, tmp_path,
                                                               monkeypatch):
        """The one long phase with an IDLE bus, and the one an operator at hour
        two is actually asking about.

        `HoldWatch` already polls both controllers here every 30 s; a capture on
        the same cadence adds five Modbus reads per 30 s to a bus with no sweep
        in flight. Nothing comparable is added anywhere a sweep IS in flight.
        """
        import softae.core.conditions_capture as CC

        reads = []
        real_read = CC.read_environment
        monkeypatch.setattr(
            CC, "read_environment",
            lambda manager: reads.append(1) or real_read(manager))
        # Counted rather than computed: how many polls a soak spends depends on
        # the settle credit, and a test that restated that arithmetic would be
        # asserting its own copy of it.
        watch_polls = []
        real_poll = H.HoldWatch.poll
        monkeypatch.setattr(
            H.HoldWatch, "poll",
            lambda self: watch_polls.append(1) or real_poll(self))

        assert self._run(tmp_path, "sksee", "--soak-h", "1") == V.EXIT_OK
        sweeps = len(_rows(tmp_path, "sksee"))
        # `run_cells` polls once per channel and `drift_check` once per recheck;
        # everything else the watch saw was the soak's own cadence.
        soak_polls = len(watch_polls) - 3 - 1
        assert soak_polls > 100                            # a real hour of them
        assert len(reads) == sweeps + soak_polls

        progress = [e for e in _stream(_only_run_dir(tmp_path))
                    if e["type"] == "progress" and e["phase"] == "soak"]
        # One record per CONSOLE line, not one per poll: a four-hour soak costs
        # ~48 records instead of ~480.
        assert len(progress) == soak_polls // H.SOAK_PRINT_EVERY_N_POLLS
        assert progress[-1]["total"] == 3600

    def test_a_soak_whose_clock_restarted_says_so_in_the_stream(self, tmp_path,
                                                               monkeypatch):
        """Exactly what somebody checking at hour two needs to be told.

        Its own record rather than one more `progress` line, because it is the
        one thing in the soak that is not monotone: a watcher reading progress
        alone would see the count go backwards with no account of why.
        """
        polls = {"n": 0}

        def _blip(self):
            polls["n"] += 1
            self.excursion = polls["n"] in (3, 4)

        monkeypatch.setattr(H.HoldWatch, "poll", _blip)
        assert self._run(tmp_path, "skx", "--soak-h", "0.1") == V.EXIT_OK

        restarts = [e for e in _stream(_only_run_dir(tmp_path))
                    if e["type"] == "soak_restart"]
        assert [e["restart"] for e in restarts] == [1, 2]
        assert restarts[0]["target_s"] == pytest.approx(360.0)
        assert restarts[0]["lost_s"] > 0

    # -- how it ended, on every path ------------------------------------------

    @pytest.mark.parametrize(
        "patch, expected, exit_code",
        [
            (RuntimeError("the mux stopped replying"), "error", V.EXIT_FAILED),
            (KeyboardInterrupt(), "interrupted", V.EXIT_INTERRUPTED),
            (H.RefuseToStart("not enough TREATMENT cells"), "aborted",
             V.EXIT_FAILED),
        ],
    )
    def test_run_finished_is_the_last_record_on_every_exit_path(
            self, tmp_path, monkeypatch, patch, expected, exit_code):
        """One `run_finished`, emitted from the `finally` AFTER the park.

        Emitted there rather than on each arm so it is the last record even on
        the paths no `except` names, and so it is a statement about a rig that
        has already been parked rather than a prediction about one that has not.
        """
        monkeypatch.setattr(H.HoldWatch, "poll", lambda self: (
            _ for _ in ()).throw(patch))
        assert self._run(tmp_path, "end") == exit_code

        events = _stream(_only_run_dir(tmp_path))
        finished = [e for e in events if e["type"] == "run_finished"]
        assert len(finished) == 1
        assert finished[0]["status"] == expected
        assert events[-1] is finished[0]
        assert events[-2]["type"] == "state" and events[-2]["new"] == "finished"
        # The park is narrated, and it precedes the finish.
        types = [e["type"] for e in events]
        assert types.index("park") < len(events) - 2
