"""``softae-shadow rehearse`` — the replay harness (T7.8).

The rehearsal exists to spend CPU instead of a board: it replays spectra already on
disk through the gated engine so the single-shot bench run is entered with a measured
wall-clock budget and a real section 7 rather than two guesses.

Everything here runs against a ``tmp_path`` fixture project built from synthetic
spectra.  No test touches the live DataStore, and none is bound to one machine's data —
a smoke replay against the real corpus would catch drift and would also make the suite
unrunnable anywhere else.

The two claims worth pinning hardest are structural rather than behavioural: the
rehearsal **cannot** write to the database it reads (``mode=ro``, and no ``DataStore``
is ever constructed, so the eight migrations never run), and it **cannot** be killed by
its own log on a cp1252 console — the crash that ended the first probe run was a
``tan δ`` in a gate's own warning text.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from softae.tools import shadow_rehearse as R
from softae.tools.shadow_review import GATED_ONLY_EVENTS, build_parser, parse_line, summarize
from tests.eis_synthetic import as_eis_result, reference_spectrum

RUN_ID = "20260811T023757Z_equilibration_characterization"

_SCHEMA = """
CREATE TABLE experiments (
    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
    workflow_name TEXT NOT NULL DEFAULT 'equilibration');
CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    channel INTEGER NOT NULL, timestamp TEXT NOT NULL DEFAULT '',
    eis_file_path TEXT);
CREATE TABLE fit_results (
    fit_id INTEGER PRIMARY KEY AUTOINCREMENT, measurement_id INTEGER NOT NULL,
    run_id TEXT NOT NULL, model_name TEXT NOT NULL,
    electrode_L_cm REAL, electrode_t_cm REAL, electrode_w_cm REAL);
"""


# ── Fixture corpora ──────────────────────────────────────────────────────────

def rows(legs=("Lup", "Ldown"), setpoints=(0, 1, 2, 3), channels=(1, 2),
         rounds=15, model="simpleSalt", geometry=True):
    """A synthetic corpus with no database and no files — enough for selection."""
    out, mid = [], 0
    for leg in legs:
        for sp in setpoints:
            for ch in channels:
                for rnd in range(rounds):
                    mid += 1
                    out.append(R.CorpusRow(
                        measurement_id=mid, channel=ch,
                        eis_file_path=f"runs\\{RUN_ID}\\eis\\"
                                      f"eq_ch{ch}_{leg}_S{sp}_R{rnd}_ch{ch}.txt",
                        model_name=model,
                        L_cm=0.2 if geometry else None,
                        t_cm=0.02 if geometry else None,
                        w_cm=0.2 if geometry else None,
                        leg=leg, setpoint=sp, round=rnd))
    return out


def make_project(tmp_path: Path, *, legs=("Lup",), setpoints=(0,), channels=(1, 2),
                 rounds=2, run_id=RUN_ID, with_fits=True, write_files=True,
                 model="simpleSalt", geometry=True, extra_runs=()) -> Path:
    """A tmp_path project: ``db/softae.db`` plus spectra under ``runs/<id>/eis/``."""
    project = tmp_path / "project"
    (project / "db").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(project / "db" / "softae.db")
    conn.executescript(_SCHEMA)
    for extra, started in extra_runs:
        conn.execute("INSERT INTO experiments (run_id, started_at) VALUES (?, ?)",
                     (extra, started))
    conn.execute("INSERT INTO experiments (run_id, started_at) VALUES (?, ?)",
                 (run_id, "2026-08-11T02:37:57+00:00"))

    f, Z = reference_spectrum()
    for row in rows(legs, setpoints, channels, rounds, model=model, geometry=geometry):
        rel = row.eis_file_path.replace(f"runs\\{RUN_ID}\\", f"runs\\{run_id}\\")
        cur = conn.execute(
            "INSERT INTO measurements (run_id, channel, eis_file_path) VALUES (?,?,?)",
            (run_id, row.channel, rel))
        if with_fits:
            conn.execute(
                "INSERT INTO fit_results (measurement_id, run_id, model_name, "
                "electrode_L_cm, electrode_t_cm, electrode_w_cm) VALUES (?,?,?,?,?,?)",
                (cur.lastrowid, run_id, model, row.L_cm, row.t_cm, row.w_cm))
        if write_files:
            target = R.resolve_path(project, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            as_eis_result(f, Z, channel=row.channel).save(target)
    conn.commit()
    conn.close()
    return project


def args(*argv):
    """Parse through the real ``softae-shadow`` parser, so the wiring is exercised."""
    return build_parser().parse_args(["rehearse", *argv])


def fake_report(state="closed"):
    return SimpleNamespace(
        fit=SimpleNamespace(arc_closure=SimpleNamespace(state=state)),
        sigma=SimpleNamespace(mode="value"),
        quality=SimpleNamespace(verdict="accept"), gate_log=())


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace ``analyze_spectrum`` with a recorder — most tests are about the harness."""
    from softae.analysis.eis import engine

    calls: list[dict] = []

    def _fake(eis, **kwargs):
        calls.append(dict(kwargs))
        return fake_report()

    monkeypatch.setattr(engine, "analyze_spectrum", _fake)
    return calls


# ── 6.1 Selection ────────────────────────────────────────────────────────────

class TestSelection:
    def test_default_selection_takes_two_rounds_from_every_cell(self):
        plan = R.select_spectra(rows())
        per_cell = {}
        for row in plan.rows:
            per_cell[row.cell_key] = per_cell.get(row.cell_key, 0) + 1
        assert set(per_cell.values()) == {2}

    def test_default_selection_of_the_full_grid_is_exactly_192_spectra(self):
        plan = R.select_spectra(rows(channels=tuple(
            [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16])))
        assert plan.n_cells == 96
        assert plan.n_selected == 192

    def test_every_cell_including_the_slowest_block_is_represented(self):
        corpus = rows()
        plan = R.select_spectra(corpus)
        assert {r.cell_key for r in plan.rows} == {r.cell_key for r in corpus}
        assert any(r.leg == "Ldown" and r.setpoint == 3 for r in plan.rows)

    def test_selection_is_deterministic_across_two_calls_without_a_seed(self):
        corpus = rows()
        first = [r.measurement_id for r in R.select_spectra(corpus).rows]
        second = [r.measurement_id for r in R.select_spectra(corpus).rows]
        assert first == second

    def test_a_seed_changes_the_rounds_but_not_the_cell_coverage(self):
        corpus = rows()
        plain = R.select_spectra(corpus)
        seeded = R.select_spectra(corpus, seed=7)
        assert {r.cell_key for r in seeded.rows} == {r.cell_key for r in plain.rows}
        assert ([r.measurement_id for r in seeded.rows]
                != [r.measurement_id for r in plain.rows])

    def test_rounds_are_spaced_across_the_round_axis_not_clustered_at_zero(self):
        assert sorted({r.round for r in R.select_spectra(rows()).rows}) == [0, 7]
        assert sorted({r.round for r in R.select_spectra(rows(), rounds=3).rows}) \
            == [0, 5, 10]

    def test_all_selects_every_spectrum_in_the_run(self):
        corpus = rows()
        plan = R.select_spectra(corpus, take_all=True)
        assert plan.n_selected == len(corpus) == plan.n_corpus

    def test_rounds_above_the_available_count_clamps_to_all(self):
        corpus = rows(rounds=3)
        assert R.select_spectra(corpus, rounds=99).n_selected == len(corpus)

    def test_limit_truncates_a_balanced_plan_and_reports_the_dropped_cells(self):
        corpus = rows()
        plan = R.select_spectra(corpus, limit=10)
        assert plan.n_selected == 10
        # Round-major ordering: a cut takes one round from ten cells rather than
        # every round of five, so the prefix stays balanced across the grid.
        assert len({r.cell_key for r in plan.rows}) == 10
        assert plan.dropped_cells == plan.n_cells - 10
        assert f"dropped {plan.dropped_cells} cells" in plan.describe()

    @pytest.mark.parametrize("flag,value", [("--limit", "0"), ("--limit", "-5"),
                                            ("--rounds", "0")])
    def test_a_non_positive_count_is_refused_at_the_parser(self, flag, value):
        # Previously silent no-ops: rows[:0] selects nothing, and a negative slice trims
        # from the END of a balanced plan — the one truncation stratification prevents.
        with pytest.raises(SystemExit):
            args(flag, value)

    def test_a_filename_that_does_not_match_the_grid_pattern_is_kept_as_its_own_cell(self):
        corpus = rows(legs=("Lup",), setpoints=(0,), channels=(1,), rounds=2)
        odd = R.CorpusRow(measurement_id=999, channel=4,
                          eis_file_path="runs\\x\\eis\\hand_measured.txt")
        plan = R.select_spectra([*corpus, odd])
        assert odd.cell_key == ("?", -1, 999)
        assert 999 in {r.measurement_id for r in plan.rows}
        assert odd.block == "unparsed"


# ── 6.2 Path and corpus resolution ───────────────────────────────────────────

class TestCorpus:
    def test_a_windows_relative_path_resolves_under_the_project_on_any_platform(self):
        out = R.resolve_path(Path("/proj"), "runs\\r\\eis\\a.txt")
        # One filename, not four — the failure mode on POSIX is a single path
        # component literally named ``runs\r\eis\a.txt``.
        assert out.name == "a.txt"
        assert out.as_posix().endswith("/proj/runs/r/eis/a.txt")

    def test_an_absolute_stored_path_is_used_as_is(self, tmp_path):
        absolute = (tmp_path / "elsewhere" / "a.txt")
        assert R.resolve_path(tmp_path / "project", str(absolute)) == absolute

    def test_the_project_dir_comes_from_the_config_loader_when_no_flag_is_given(
            self, tmp_path, monkeypatch):
        from softae.config import loader

        monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "cfg"))
        assert R._resolve_project(None) == tmp_path / "cfg"
        assert R._resolve_project(str(tmp_path / "flag")) == tmp_path / "flag"

    def test_a_project_with_no_runs_exits_one_rather_than_replaying_nothing(
            self, tmp_path):
        project = tmp_path / "empty"
        (project / "db").mkdir(parents=True)
        conn = sqlite3.connect(project / "db" / "softae.db")
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()
        assert R.cmd_rehearse(args("--project", str(project))) == R.EXIT_NOTHING
        assert R.cmd_rehearse(args("--project", str(tmp_path / "nodb"))) \
            == R.EXIT_NOTHING

    def test_the_model_name_comes_from_the_fit_row_not_a_literal(self, tmp_path):
        project = make_project(tmp_path, model="randles")
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        assert {r.model_name for r in corpus} == {"randles"}

    def test_the_cell_is_reconstructed_from_the_fit_rows_electrode_columns(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        cell = stub_engine[0]["cell"]
        assert (cell.L_gap_cm, cell.L_stripe_cm, cell.thickness_cm) == (0.2, 0.2, 0.02)
        assert all(r.cell_source == "fit_row" for r in result.records)

    def test_a_measurement_with_no_fit_row_passes_cell_none_and_is_marked_absent(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path, with_fits=False)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        assert stub_engine[0]["cell"] is None
        assert all(r.cell_source == "absent" for r in result.records)


# ── 6.3 Invocation fidelity ──────────────────────────────────────────────────

class TestInvocation:
    def test_the_engine_is_gated_even_when_the_live_config_says_legacy(
            self, monkeypatch):
        # The rig sits on engine="legacy" as its normal state; a replay must still be
        # a replay. engine and gates.enabled are the only two fields the config cannot
        # reach.
        from softae.config import loader

        monkeypatch.setattr(loader, "load", lambda *a, **k: {
            "eis": {"engine": "legacy", "gates": {"enabled": True}}})
        assert R.replay_settings(enforced=False).engine == "gated"
        assert R.replay_settings(enforced=False).gates.enabled is False
        assert R.replay_settings(enforced=True).gates.enabled is True

    def test_gate_thresholds_mirror_the_live_config_rather_than_generic_defaults(
            self, monkeypatch):
        # An operator who has armed calibrated values must get verdicts under those
        # values; a replay judged by shipped defaults describes a population the rig
        # would never produce.
        from softae.analysis.eis.settings import DEFAULT_MAX_REL_SE
        from softae.config import loader

        monkeypatch.setattr(loader, "load", lambda *a, **k: {
            "eis": {"gates": {"max_rel_se": 0.42, "tand_slope_max": -0.75}}})
        gates = R.replay_settings().gates
        assert gates.max_rel_se == 0.42 != DEFAULT_MAX_REL_SE
        assert gates.tand_slope_max == -0.75

    def test_an_unreadable_config_still_yields_a_gated_replay(self, monkeypatch):
        from softae.config import loader

        def _boom(*a, **k):
            raise FileNotFoundError("no config anywhere")

        monkeypatch.setattr(loader, "load", _boom)
        settings = R.replay_settings()
        assert settings.engine == "gated" and settings.gates.enabled is False

    def test_the_rehearsal_runs_the_gated_engine_with_gates_observing(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path, channels=(1,), rounds=1)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        R.run_rehearsal(R.select_spectra(corpus), project)
        settings = stub_engine[0]["settings"]
        assert settings.engine == "gated" and settings.gates.enabled is False

    def test_a_rejected_spectrum_is_still_analysed_because_gates_only_observe(
            self, tmp_path):
        # A blocking spectrum takes the long way to failing: with the gates observing
        # there is no R18 early return, so the fitter still runs and the record exists.
        # ``dispersive_dielectric`` rather than ``pure_series_rc`` on purpose — both
        # fail admission, but the series-RC one also defeats the fitter and takes 85 s
        # to do it, which is the *cost* finding, not something to pay per test run.
        from tests.eis_synthetic import dispersive_dielectric

        project = make_project(tmp_path, channels=(1,), rounds=1)
        f, Z = dispersive_dielectric()
        for path in (project / "runs").rglob("*.txt"):
            as_eis_result(f, Z, channel=1).save(path)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        assert len(result.records) == 1
        assert result.records[0].n_gates_failed >= 1

    def test_the_config_file_is_not_written(self, tmp_path, stub_engine):
        from softae.config import loader

        config = Path(loader.config_path())
        before = config.stat().st_mtime_ns
        project = make_project(tmp_path)
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(tmp_path / "r.log"))) == R.EXIT_OK
        assert config.stat().st_mtime_ns == before


# ── 6.4 The no-write guarantee ───────────────────────────────────────────────

class TestNoWrite:
    def test_record_fit_is_never_called(self, tmp_path, stub_engine, monkeypatch):
        from softae.core.data_store import DataStore

        def _boom(*a, **k):
            raise AssertionError("a rehearsal must never write a fit row")

        monkeypatch.setattr(DataStore, "record_fit", _boom)
        project = make_project(tmp_path)
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(tmp_path / "r.log"))) == R.EXIT_OK

    def test_the_database_is_opened_read_only(self, tmp_path, monkeypatch):
        project = make_project(tmp_path)
        db = project / "db" / "softae.db"
        seen: list[str] = []
        real = sqlite3.connect

        def _spy(target, *a, **k):
            seen.append(str(target))
            return real(target, *a, **k)

        monkeypatch.setattr(sqlite3, "connect", _spy)
        conn = R._connect_ro(db)
        try:
            assert seen and "mode=ro" in seen[0]
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO experiments (run_id, started_at) "
                             "VALUES ('x', 'y')")
        finally:
            conn.close()

    def test_the_database_file_mtime_is_unchanged_by_a_rehearsal(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path)
        db = project / "db" / "softae.db"
        before = db.stat().st_mtime_ns
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(tmp_path / "r.log"))) == R.EXIT_OK
        assert db.stat().st_mtime_ns == before
        assert not (project / "db" / "softae.db-wal").exists()

    def test_no_datastore_instance_is_constructed(self, tmp_path, stub_engine,
                                                  monkeypatch):
        # ``DataStore.__init__`` mkdirs, sets WAL, runs the DDL and eight migrations.
        # Every one is idempotent; none of them is read-only.
        from softae.core.data_store import DataStore

        def _boom(*a, **k):
            raise AssertionError("DataStore must never be constructed by a rehearsal")

        monkeypatch.setattr(DataStore, "__init__", _boom)
        project = make_project(tmp_path)
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(tmp_path / "r.log"))) == R.EXIT_OK


# ── 6.5 Timing records and resilience ────────────────────────────────────────

class TestTimingAndResilience:
    def test_a_timing_record_carries_seconds_verdict_and_gate_failure_count(
            self, tmp_path):
        project = make_project(tmp_path, channels=(1,), rounds=1)
        log = tmp_path / "r.log"
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(log))) == R.EXIT_OK
        row = list(R.timing_csv_path(log).read_text(encoding="utf-8").splitlines())[1]
        record = dict(zip(R.CSV_COLUMNS, row.split(",")))
        assert float(record["seconds"]) > 0
        assert record["verdict"] and record["spectrum_key"].startswith("c01:")
        assert record["n_gates_failed"].isdigit()
        assert record["arc_state"] in ("closed", "open", "unknown")

    def test_the_first_spectrum_is_flagged_as_warmup_and_excluded_from_the_median(self):
        def rec(seconds, warm=False):
            return R.TimingRecord("k", 1, 1, "Lup", 0, 0, seconds, 0.01, "accept",
                                  0, "closed", "value", "fit_row", warm)

        summary = R.summarize_timing([rec(9.0, warm=True), rec(1.0), rec(3.0)])
        assert summary.warmup_seconds == 9.0
        assert summary.all.n == 2 and summary.all.median == 2.0
        assert summary.all.maximum == 3.0

    def test_the_timing_csv_has_one_row_per_analysed_spectrum_plus_a_header(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path, channels=(1, 2), rounds=2)
        log = tmp_path / "r.log"
        R.cmd_rehearse(args("--project", str(project), "--out", str(log)))
        lines = R.timing_csv_path(log).read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",") == list(R.CSV_COLUMNS)
        assert len(lines) == 1 + 4

    def test_the_timing_csv_sits_beside_the_log_wherever_out_points(self, tmp_path,
                                                                   stub_engine):
        out = tmp_path / "deep" / "nested" / "run.log"
        project = make_project(tmp_path)
        R.cmd_rehearse(args("--project", str(project), "--out", str(out)))
        assert (out.parent / "run.timing.csv").is_file()
        assert R.timing_csv_path(out) == out.parent / "run.timing.csv"

    def test_a_missing_file_is_skipped_and_counted_rather_than_aborting_the_batch(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path, channels=(1, 2), rounds=1)
        next(iter(sorted((project / "runs").rglob("*.txt")))).unlink()
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        assert result.n_missing == 1 and len(result.records) == 1

    def test_an_unloadable_file_is_skipped_and_counted(self, tmp_path, stub_engine):
        project = make_project(tmp_path, channels=(1, 2), rounds=1)
        target = sorted((project / "runs").rglob("*.txt"))[0]
        target.write_text("not a spectrum at all\n", encoding="utf-8")
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        assert result.n_unloadable == 1 and len(result.records) == 1

    def test_an_exception_from_analyze_spectrum_skips_one_spectrum_and_continues(
            self, tmp_path, monkeypatch):
        from softae.analysis.eis import engine

        seen = {"n": 0}

        def _flaky(eis, **kwargs):
            seen["n"] += 1
            if seen["n"] == 1:
                raise RuntimeError("the fitter fell over")
            return fake_report()

        monkeypatch.setattr(engine, "analyze_spectrum", _flaky)
        project = make_project(tmp_path, channels=(1, 2), rounds=1)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        result = R.run_rehearsal(R.select_spectra(corpus), project)
        assert result.n_raised == 1 and len(result.records) == 1

    def test_each_csv_row_is_flushed_as_its_spectrum_completes(self, tmp_path,
                                                               monkeypatch):
        # The batch is the most interruptible thing this tool does. If the CSV were
        # serialised at the end, a Ctrl-C at spectrum 140 of 192 would lose 139 real
        # measurements — the exact failure the resilience contract promises to prevent.
        from softae.analysis.eis import engine

        project = make_project(tmp_path, channels=(1, 2), rounds=2)
        log = tmp_path / "flush.log"
        seen: list[int] = []

        def _watch(eis, **kwargs):
            # Count the rows already durable on disk *before* this spectrum is analysed.
            csv_file = R.timing_csv_path(log)
            body = csv_file.read_text(encoding="utf-8").splitlines()[1:] \
                if csv_file.exists() else []
            seen.append(len(body))
            return fake_report()

        monkeypatch.setattr(engine, "analyze_spectrum", _watch)
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(log))) == R.EXIT_OK
        # Before spectrum 1: 0 rows durable. Before spectrum 4: 3 rows durable.
        assert seen == [0, 1, 2, 3]

    def test_an_interrupt_keeps_every_completed_row_and_labels_the_summary_partial(
            self, tmp_path, monkeypatch, capsys):
        from softae.analysis.eis import engine

        project = make_project(tmp_path, channels=(1, 2), rounds=2)
        log = tmp_path / "interrupted.log"
        calls = {"n": 0}

        def _ctrl_c_on_the_third(eis, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return fake_report()

        monkeypatch.setattr(engine, "analyze_spectrum", _ctrl_c_on_the_third)
        # 1, not 0: swallowing the signal must not turn a half-run into a success.
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(log))) == R.EXIT_NOTHING
        rows_on_disk = R.timing_csv_path(log).read_text(
            encoding="utf-8").splitlines()[1:]
        assert len(rows_on_disk) == 2
        out = capsys.readouterr().out
        assert "PARTIAL" in out and "interrupted after 2 of 4" in out

    def test_out_refuses_to_overwrite_an_existing_log(self, tmp_path, stub_engine):
        project = make_project(tmp_path)
        log = tmp_path / "already.log"
        log.write_text("previous run\n", encoding="utf-8")
        assert R.cmd_rehearse(args("--project", str(project),
                                   "--out", str(log))) == R.EXIT_NOTHING
        assert log.read_text(encoding="utf-8") == "previous run\n"

    def test_dry_run_writes_no_log_and_analyses_nothing(self, tmp_path, stub_engine,
                                                        capsys):
        project = make_project(tmp_path)
        log = tmp_path / "r.log"
        assert R.cmd_rehearse(args("--project", str(project), "--out", str(log),
                                   "--dry-run")) == R.EXIT_OK
        assert not log.exists() and not R.timing_csv_path(log).exists()
        assert stub_engine == []
        assert "projected" in capsys.readouterr().out


# ── 6.6 Round trip into the review ───────────────────────────────────────────

@pytest.fixture(scope="module")
def rehearsal_log(tmp_path_factory):
    """A real (unstubbed) rehearsal of two spectra, and its log.

    Module-scoped: this is the only fixture here that runs the actual fitter, and
    four tests read the same artifact.  Re-running it per test would pay the fit
    four times over to produce a byte-identical log.
    """
    tmp_path = tmp_path_factory.mktemp("round_trip")
    project = make_project(tmp_path, channels=(1, 2), rounds=1)
    log = tmp_path / "round_trip.log"
    assert R.cmd_rehearse(args("--project", str(project), "--out", str(log))) \
        == R.EXIT_OK
    return log


class TestRoundTrip:
    def test_a_rehearsal_log_parses_through_parse_line(self, rehearsal_log):
        text = rehearsal_log.read_text(encoding="utf-8")
        for line in text.splitlines():
            fields = parse_line(line)
            assert fields is None or "event" in fields

    def test_a_rehearsal_log_is_recognised_as_a_shadow_run_by_summarize(
            self, rehearsal_log):
        review = summarize(rehearsal_log.read_text(encoding="utf-8").splitlines())
        assert review.is_shadow_run
        assert review.events["eis_spectrum_metrics"] >= 2

    def test_a_rehearsal_log_yields_spectrum_records_for_the_recommender(
            self, rehearsal_log):
        review = summarize(rehearsal_log.read_text(encoding="utf-8").splitlines())
        assert len(review.spectra) == 2
        assert all(record.metrics for record in review.spectra)

    def test_the_rehearsals_own_events_do_not_pollute_the_gated_event_count(
            self, rehearsal_log):
        review = summarize(rehearsal_log.read_text(encoding="utf-8").splitlines())
        assert review.events["rehearsal_spectrum_done"] == 2
        assert not any(name.startswith("rehearsal_") for name in GATED_ONLY_EVENTS)
        gated = sum(review.events[e] for e in GATED_ONLY_EVENTS)
        assert review.n_gated_events == gated

    def test_a_gate_detail_containing_tan_delta_is_written_without_raising(
            self, tmp_path, monkeypatch):
        # The first probe run died here: ``tan δ`` through a cp1252 stdout, inside
        # ``run_gates``' own warning, on the first spectrum that failed tand_slope.
        import structlog

        log = tmp_path / "delta.log"
        monkeypatch.setattr(sys, "stdout",
                            open(tmp_path / "cp1252.out", "w", encoding="cp1252"))
        # tee=True is load-bearing: without it the patched cp1252 stream is never
        # written and the test proves nothing. With it, _Tee genuinely attempts the
        # console write that raised UnicodeEncodeError in the probe.
        with R.open_log_stream(log, tee=True):
            structlog.get_logger("t").warning(
                "eis_gate_rejected", gate="tand_slope",
                detail="tan δ slope +0.42 > -0.30 — σ ≲ 4e-07 S/cm")
        sys.stdout.close()
        text = log.read_text(encoding="utf-8")
        assert "tan δ" in text and "σ ≲" in text
        assert parse_line(text.splitlines()[0])["event"] == "eis_gate_rejected"


# ── Q2: the enforcing comparison pass ────────────────────────────────────────

class TestEnforcedMode:
    def test_enforced_passes_gates_enabled_true_so_the_two_costs_are_comparable(
            self, tmp_path, stub_engine):
        project = make_project(tmp_path, channels=(1,), rounds=1)
        _, corpus = R._read_corpus(project / "db" / "softae.db")
        R.run_rehearsal(R.select_spectra(corpus), project, enforced=True)
        assert stub_engine[0]["settings"].gates.enabled is True

    def test_the_timing_csv_records_which_mode_produced_the_row(self, tmp_path,
                                                                stub_engine):
        project = make_project(tmp_path, channels=(1,), rounds=1)
        log = tmp_path / "e.log"
        R.cmd_rehearse(args("--project", str(project), "--out", str(log), "--enforced"))
        body = R.timing_csv_path(log).read_text(encoding="utf-8").splitlines()[1]
        assert body.split(",")[R.CSV_COLUMNS.index("mode")] == "enforcing"


# ── Rendering ────────────────────────────────────────────────────────────────

class TestSummaryRendering:
    def test_the_summary_splits_by_arc_state_and_brackets_the_extrapolation(self):
        def rec(seconds, state):
            return R.TimingRecord("k", 1, 1, "Ldown", 3, 0, seconds, 0.01, "suspect",
                                  1, state, "value", "fit_row", False)

        records = [rec(0.2, "closed"), rec(0.3, "closed"), rec(45.0, "open")]
        summary = R.summarize_timing(records)
        assert summary.closed.n == 2 and summary.open.n == 1
        assert summary.open_fraction == pytest.approx(1 / 3)
        floor, mix, ceiling = summary.bracket(16)
        assert floor < mix < ceiling

        result = R.RehearsalResult(plan=R.select_spectra(rows(rounds=2)),
                                   records=records)
        text = R.render_summary(result, summary, Path("logs/r.log"), Path("~/x"))
        assert "arc CLOSED" in text and "arc OPEN" in text
        assert "EXTRAPOLATION" in text and "16 wells" in text
        assert "softae-shadow review logs/r.log --project ~/x" in text
        assert "Ldown/S3" in text
