"""``softae.tools.c_cell``: read-only preview, the typed-yes write, and the stored-data check.

The write path is exercised only against a **fake** store exposing eis-acq's proposed L3
contract (``record_cell_capacitance`` / ``cell_capacitance``) and a temp project; the real
DataStore is only ever opened ``mode=ro``, by the stored-data test, which skips when absent.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis import cell_capacitance as cc
from softae.tools import c_cell as tool
from tests import eis_regime_synthetic as syn

W = 2 * np.pi * syn.RIG_F
RUN = "20260101T000000Z_test"


def _rc(R, C):
    return R / (1 + 1j * W * R * C)


def _spectrum(kind: str, seed: int) -> np.ndarray:
    base = _rc(1e5, 3e-12)
    if kind == "film":
        return syn._noisy(base + _rc(1e8, 2.5e-10) + 1 / (1e-7 * (1j * W) ** 0.8), seed)
    C = {"ins_a": 2.0e-10, "ins_b": 3.0e-10}[kind]
    return syn._noisy(base + _rc(1e11, C), seed)


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temp project: the real schema (temp DataStore), three channels, one read each."""
    from softae.analysis.eis_data import EISResult
    from softae.config import loader
    from softae.core.data_store import DataStore

    monkeypatch.setattr(loader, "data_db_filename", lambda: "softae.db")
    DataStore(tmp_path).close()
    conn = sqlite3.connect(tmp_path / "db" / "softae.db")
    conn.execute("INSERT INTO experiments (run_id, started_at, workflow_name) "
                 "VALUES (?, '2026-01-01T00:00:00', 't')", (RUN,))
    for ch, kind, ts in ((9, "ins_a", "2026-01-01T01:00:00"), (10, "ins_b", "2026-01-01T01:01:00"),
                         (11, "film", "2026-01-01T01:02:00")):
        rel = f"runs/{RUN}/eis/ch{ch}.txt"
        Z = _spectrum(kind, ch)
        EISResult.from_arrays(ch, syn.RIG_F, Z.real, -Z.imag).save(tmp_path / rel)
        conn.execute("INSERT INTO measurements (run_id, channel, timestamp, eis_file_path) "
                     "VALUES (?, ?, ?, ?)", (RUN, ch, ts, rel))
        conn.execute("INSERT INTO electrode_occupancy (board_id, electrode, run_id, cast_at) "
                     "VALUES (5, ?, ?, '2026-01-01T00:30:00+00:00')", (ch, RUN))
    conn.commit()
    conn.close()
    return tmp_path


class FakeStore:
    """The proposed L3 contract, in memory. Class-level so a test can inspect it."""

    rows: list[dict] = []
    constructed = 0

    def __init__(self, project_dir, db_filename="softae.db"):
        type(self).constructed += 1

    def record_cell_capacitance(self, board_id, group, c_cell_F, spread, method,
                                source_measurement_ids, accepted_by):
        type(self).rows.append(dict(board_id=board_id, channel=None, group_lo=group[0],
                                    group_hi=group[1], c_cell_F=c_cell_F, spread_dec=spread,
                                    method=method, source_measurement_ids=source_measurement_ids,
                                    accepted_by=accepted_by))
        return len(type(self).rows)

    def cell_capacitance(self, board_id, channel):
        return cc.preferred_record(channel, [r for r in type(self).rows
                                             if r["board_id"] == board_id])


class NoApiStore:
    constructed = 0

    def __init__(self, *a, **k):
        type(self).constructed += 1


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeStore.rows, FakeStore.constructed, NoApiStore.constructed = [], 0, 0


def _db_digest(project: Path) -> str:
    return hashlib.sha256((project / "db" / "softae.db").read_bytes()).hexdigest()


def _run(project, *extra, answer="yes", store_cls=FakeStore):
    asked = []

    def input_fn(prompt):
        asked.append(prompt)
        if answer is EOFError:
            raise EOFError
        return answer

    code = tool.main(["--board", "5", "--run", RUN, "--project", str(project), *extra],
                     input_fn=input_fn, store_cls=store_cls)
    return code, asked


# ── Preview never writes ─────────────────────────────────────────────────────

def test_c_cell_preview_prints_group_and_never_writes(project, capsys):
    before = _db_digest(project)
    code, asked = _run(project)
    out = capsys.readouterr().out
    assert code == tool.EXIT_OK and not asked and FakeStore.constructed == 0
    value = float(re.search(r"9-16\s+C_cell = (\S+) F", out).group(1))
    assert value == pytest.approx(2.5e-10, rel=0.01)
    assert "film_arc_present" in out and "Preview only" in out
    assert _db_digest(project) == before


def test_c_cell_connect_ro_refuses_insert(project):
    conn = tool.connect_ro(project / "db" / "softae.db")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO experiments (run_id, started_at, workflow_name) "
                     "VALUES ('x', 'y', 'z')")
    conn.close()


def test_c_cell_missing_store_raises_not_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        tool.connect_ro(tmp_path / "db" / "softae.db")


# ── Write refusals ───────────────────────────────────────────────────────────

def test_c_cell_write_api_absent_refuses_before_prompt(project, capsys):
    code, asked = _run(project, "--write", "--accepted-by", "CO", store_cls=NoApiStore)
    assert code == tool.EXIT_REFUSED and not asked and NoApiStore.constructed == 0
    assert "has not landed yet" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["no", "YES", "y", "", EOFError])
def test_c_cell_write_without_typed_yes_writes_nothing(project, answer):
    code, asked = _run(project, "--write", "--accepted-by", "CO", answer=answer)
    assert code == tool.EXIT_REFUSED and asked
    assert FakeStore.rows == [] and FakeStore.constructed == 0


def test_c_cell_write_without_accepted_by_refused(project):
    code, asked = _run(project, "--write")
    assert code == tool.EXIT_REFUSED and not asked and FakeStore.rows == []


def test_c_cell_write_default_store_class_refuses_or_prompts(project, capsys):
    """The default wiring: today's DataStore without the L3 method refuses before the
    prompt; once L3 lands the prompt is reached (and "no" still writes nothing)."""
    from softae.core.data_store import DataStore

    landed = callable(getattr(DataStore, "record_cell_capacitance", None))
    code, asked = _run(project, "--write", "--accepted-by", "CO", answer="no", store_cls=None)
    assert code == tool.EXIT_REFUSED and bool(asked) == landed
    assert ("has not landed yet" in capsys.readouterr().out) != landed


# ── Write round trip (fake store) ────────────────────────────────────────────

def test_c_cell_write_typed_yes_round_trips_through_contract(project, capsys):
    code, _ = _run(project, "--write", "--accepted-by", "CO")
    assert code == tool.EXIT_OK and FakeStore.constructed == 1
    (row,) = FakeStore.rows
    assert (row["board_id"], row["group_lo"], row["group_hi"]) == (5, 9, 16)
    assert row["c_cell_F"] == pytest.approx(np.median([2.0e-10, 3.0e-10]), rel=0.01)
    assert row["method"] == cc.METHOD and row["accepted_by"] == "CO"
    assert len(row["source_measurement_ids"]) == 2           # the film (ch11) is not a source
    store = FakeStore(project)
    assert store.cell_capacitance(5, 12) is row
    assert store.cell_capacitance(5, 3) is None and store.cell_capacitance(4, 10) is None


# ── The store refuses a row ([e158] L4 note) ─────────────────────────────────

class RefusingStore(FakeStore):
    """The store's own validation contract: ``ValueError`` before anything is written."""

    def record_cell_capacitance(self, **kw):
        raise ValueError("source_measurement_ids must be a non-empty sequence of ints")


def test_c_cell_write_store_value_error_reported_cleanly_exit_failed(project, capsys):
    code, asked = _run(project, "--write", "--accepted-by", "CO", store_cls=RefusingStore)
    err = capsys.readouterr().err
    assert code == tool.EXIT_FAILED and asked
    assert "The store refused group 9-16: source_measurement_ids" in err
    assert "Recorded before the refusal: nothing." in err
    assert "Traceback" not in err and FakeStore.rows == []


def _group(group, ids, c=2.5e-10):
    return cc.CellCapacitance(board_id=5, group=group, c_cell_F=c, spread_dec=0.0,
                              source_measurement_ids=tuple(ids),
                              channel_values=((group[0], c),))


def test_c_cell_write_groups_real_store_refusal_names_group_and_prior_rows(tmp_path):
    """Against the REAL store: a group with no measurement ids is refused by the store's
    own validation, and the group written before it is reported, not lost."""
    from softae.core.data_store import DataStore

    store = DataStore(tmp_path)
    try:
        groups = {(1, 8): _group((1, 8), [11]), (9, 16): _group((9, 16), [])}
        with pytest.raises(tool.GroupRefused) as err:
            tool.write_groups(store, groups, accepted_by="CO")
        assert err.value.group == (9, 16) and len(err.value.written) == 1
        assert isinstance(err.value, ValueError) and "non-empty" in str(err.value)
        assert store.cell_capacitance(5, 3)["id"] == err.value.written[0]
        assert store.cell_capacitance(5, 12) is None
    finally:
        store.close()


# ── Selection and occupancy ──────────────────────────────────────────────────

def test_select_measurements_latest_per_channel_all_reads_and_role(project):
    db = project / "db" / "softae.db"
    w = sqlite3.connect(db)
    w.execute("INSERT INTO measurements (run_id, channel, timestamp, eis_file_path) "
              "VALUES (?, 9, '2026-01-01T02:00:00', 'later.txt')", (RUN,))
    w.execute("INSERT INTO measurements (run_id, channel, timestamp, eis_file_path, role) "
              "VALUES (?, 12, '2026-01-01T02:00:00', 'cap.txt', 'reference')", (RUN,))
    w.commit()
    w.close()
    conn = tool.connect_ro(db)
    latest = tool.select_measurements(conn, runs=[RUN])
    every = tool.select_measurements(conn, runs=[RUN], all_reads=True)
    only10 = tool.select_measurements(conn, runs=[RUN], channels=[10])
    exact = tool.select_measurements(conn, measurement_ids=[1])
    conn.close()
    assert sorted(r["channel"] for r in latest) == [9, 10, 11]
    assert next(r for r in latest if r["channel"] == 9)["eis_file_path"] == "later.txt"
    assert len(every) == 4 and [r["channel"] for r in only10] == [10]
    assert [r["measurement_id"] for r in exact] == [1]


@pytest.mark.parametrize("row,run,ts,state", [
    (None, RUN, "2026-01-01T01:00:00", cc.UNKNOWN),
    ({"run_id": RUN, "cast_at": "2099-01-01T00:00:00+00:00"}, RUN, "2026-01-01T01:00:00",
     cc.OCCUPIED),                                           # the run's own cast, stamped late
    ({"run_id": None, "cast_at": "2026-01-01T00:00:00+00:00"}, "other", "2026-01-02T00:00:00",
     cc.OCCUPIED),
    ({"run_id": "later_run", "cast_at": "2026-01-03T00:00:00+00:00"}, "other",
     "2026-01-02T00:00:00", cc.EMPTY),                       # read before the well was cast
    ({"run_id": None, "cast_at": "2026-01-01 00:00:00"}, "other", "2026-01-02T00:00:00",
     cc.OCCUPIED),                                           # SQLite's naive-UTC default
    ({"run_id": None, "cast_at": "garbage"}, "other", "2026-01-02T00:00:00", cc.UNKNOWN),
])
def test_occupancy_state_cases(row, run, ts, state):
    assert tool.occupancy_state(row, run_id=run, timestamp=ts) == state


def test_c_cell_other_board_reads_refused_as_unrecorded(project, capsys):
    code = tool.main(["--board", "7", "--run", RUN, "--project", str(project)],
                     input_fn=lambda p: "yes", store_cls=FakeStore)
    out = capsys.readouterr().out
    assert code == tool.EXIT_OK and "occupancy_unrecorded" in out and "C_cell =" not in out


# ── Stored data: board 2 (read-only; skips when the store is absent) ─────────

BOARD2_RUNS = ("20260923T183923Z_rung3b_bench", "20260929T150528Z_manual_eis",
               "20261002T053100Z_rung3c_paa_bench")


@pytest.fixture(scope="module")
def board2():
    from softae.config import loader

    project = Path(loader.data_project_dir()).expanduser()
    db = project / "db" / loader.data_db_filename()
    if not db.is_file():
        pytest.skip(f"stored DataStore absent: {db}")
    conn = tool.connect_ro(db)
    try:
        rows = tool.select_measurements(conn, runs=BOARD2_RUNS)
        occ = {int(r["electrode"]): dict(r) for r in conn.execute(
            "SELECT electrode, run_id, cast_at FROM electrode_occupancy WHERE board_id = 2")}
    finally:
        conn.close()
    if len(rows) != 24:
        pytest.skip(f"board-2 runs not all present ({len(rows)} channels)")
    checks, skipped = tool.assess(rows, project=project, occupancy=occ)
    if skipped:
        pytest.skip(f"board-2 spectra unreadable: {skipped[:2]}")
    return checks


def test_c_cell_stored_board2_groups_1_8_and_17_24_reproduce_spec(board2):
    g = cc.aggregate_groups(board2, board_id=2)
    assert g[(1, 8)].c_cell_F == pytest.approx(2.66e-10, rel=0.01)
    assert [ch for ch, _ in g[(1, 8)].channel_values] == [3]
    assert g[(17, 24)].c_cell_F == pytest.approx(3.08e-10, rel=0.01)
    assert [ch for ch, _ in g[(17, 24)].channel_values] == [20]
    assert (25, 32) not in g


def test_c_cell_stored_board2_group_9_16_diverges_from_spec_by_flatness_rule(board2):
    """Spec §2.2 states 1.82e-10 = median of ch 9/10/13/14, but 9, 10 and 14 fail the
    spec's own flatness rule (max/min 2.61 / 1.43 / 1.45 > 1.3) and ch11 passes every
    rule. The estimator reproduces the spec's four values; the selector, applied as
    written, gives 1.71e-10 from ch 11 and 13. Surfaced in the lane report, not repaired."""
    by_ch = {c.channel: c for c in board2}
    spec_set = [by_ch[ch].c_cell_F for ch in (9, 10, 13, 14)]
    assert np.median(spec_set) == pytest.approx(1.82e-10, rel=0.01)
    assert {ch for ch in (9, 10, 14) if "apparent_c_not_flat" in by_ch[ch].reasons} == {9, 10, 14}
    g = cc.aggregate_groups(board2, board_id=2)[(9, 16)]
    assert [ch for ch, _ in g.channel_values] == [11, 13]
    assert g.c_cell_F == pytest.approx(1.713e-10, rel=0.01)


def test_c_cell_stored_board2_route_entered_by_real_data(board2):
    """§3.2 counter: real data visits the source branch and every major refusal."""
    reasons = {r for c in board2 for r in c.reasons}
    assert sum(c.qualifies for c in board2) == 4
    assert {"film_arc_present", "regime_bad_data", "lf_dropped", "regime_a_film",
            "phase_not_capacitive", "apparent_c_not_flat"} <= reasons
