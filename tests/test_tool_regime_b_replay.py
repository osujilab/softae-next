"""``softae.tools.regime_b_replay``: the read-only before/after shadow table.

A temp project with the real schema (a temp ``DataStore``, never the real one), two
synthetic two-feature spectra on disk, and one ``cell_capacitance`` row through eis-acq's
own ``record_cell_capacitance``. Each spectrum runs through the real gated engine twice.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import re
import sqlite3
from pathlib import Path

import pytest

from softae.analysis.eis import regime_b as rb
from softae.tools import regime_b_replay as tool
from tests import eis_two_feature_synthetic as tf

RUN = "20260101T000000Z_replay"
BOARD = 7


def _engine_config(monkeypatch, engine: str) -> None:
    """``[eis] engine`` as the tool will read it (it never names one itself, [a23])."""
    from softae.analysis.eis import engine as engine_mod

    real = engine_mod.eis_settings
    monkeypatch.setattr(engine_mod, "eis_settings",
                        lambda: dataclasses.replace(real(), engine=engine))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Channel 3 (group 1-8, a store row) and channel 12 (no row): one film arc each."""
    from softae.analysis.eis_data import EISResult
    from softae.config import loader
    from softae.core.data_store import DataStore

    monkeypatch.setattr(loader, "data_db_filename", lambda: "softae.db")
    _engine_config(monkeypatch, "gated")
    store = DataStore(tmp_path)
    store.record_cell_capacitance(board_id=BOARD, group=(1, 8), c_cell_F=tf.C_CELL,
                                  spread=0.0, method="test", source_measurement_ids=[1],
                                  accepted_by="T")
    store.close()
    conn = sqlite3.connect(tmp_path / "db" / "softae.db")
    conn.execute("INSERT INTO experiments (run_id, started_at, workflow_name) "
                 "VALUES (?, '2026-01-01T00:00:00', 't')", (RUN,))
    for ch, seed in ((3, 1), (12, 2)):
        f, Z = tf.spectrum(tf.TwoFeature(), seed=seed)
        rel = f"runs/{RUN}/eis/ch{ch}.txt"
        EISResult.from_arrays(ch, f, Z.real, -Z.imag).save(tmp_path / rel)
        conn.execute("INSERT INTO measurements (run_id, channel, timestamp, eis_file_path) "
                     "VALUES (?, ?, '2026-01-01T01:00:00', ?)", (RUN, ch, rel))
        conn.execute("INSERT INTO electrode_occupancy (board_id, electrode, run_id, cast_at) "
                     "VALUES (?, ?, ?, '2026-01-01T00:30:00+00:00')", (BOARD, ch, RUN))
    conn.commit()
    conn.close()
    return tmp_path


#: SQLite's own WAL-mode side files. Closing the last connection may delete an EMPTY
#: ``-wal`` / ``-shm`` pair; that is housekeeping, not a write — so they are left out of
#: the digest, and :func:`_no_pending_wal` checks no data is sitting in a WAL instead.
_SQLITE_AUX = ("-wal", "-shm")


def _tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.name.endswith(_SQLITE_AUX)}


def _no_pending_wal(root: Path) -> bool:
    return all(p.stat().st_size == 0 for p in root.rglob("*-wal"))


def _row(out: str, ch: int) -> str:
    return next(line for line in out.splitlines()
                if re.match(rf"^[* ]\s+\d+\s+{ch}\s", line))


def test_regime_b_replay_store_c_cell_moves_row_and_never_writes(project, capsys):
    before = _tree_digest(project)
    code = tool.main(["--run", RUN, "--project", str(project), "--thickness-um", "150"])
    out = capsys.readouterr().out
    assert code == tool.EXIT_OK
    assert _tree_digest(project) == before                  # not one byte, not one file
    assert _no_pending_wal(project)
    ch3, ch12 = _row(out, 3), _row(out, 12)
    # ch3: the board (from electrode_occupancy) has a group row, so the route is live.
    assert ch3.startswith("*") and "store #1" in ch3
    assert "regime_b_film_arc" in ch3 and rb.RESOLVED_IN_BAND in ch3 and "film_arc" in ch3
    # ch12: no row covers it — unmeasured, never a default; nothing moves.
    assert ch12.startswith(" ") and "unmeasured" in ch12 and rb.C_CELL_UNMEASURED in ch12
    assert "2 spectra, 1 moved. Nothing was written." in out


def test_regime_b_replay_override_beats_store_and_channel_beats_group(project, capsys):
    code = tool.main(["--run", RUN, "--project", str(project), "--thickness-um", "150",
                      "--channels", "12", "--c-cell", "9-16=1e-12", "--c-cell", "12=2.5e-10"])
    out = capsys.readouterr().out
    assert code == tool.EXIT_OK
    ch12 = _row(out, 12)
    assert "override" in ch12 and "2.5e-10" in ch12 and "regime_b_film_arc" in ch12
    assert "1 spectra, 1 moved" in out


def test_regime_b_replay_legacy_engine_config_says_nothing_can_move(project, monkeypatch,
                                                                    capsys):
    """The tool leaves the engine to config; on legacy it says why no row moves."""
    _engine_config(monkeypatch, "legacy")
    assert tool.main(["--run", RUN, "--project", str(project), "--thickness-um", "150",
                      "--channels", "3"]) == tool.EXIT_OK
    out = capsys.readouterr().out
    assert "1 spectra, 0 moved" in out and "engine is not 'gated'" in out


def test_regime_b_replay_store_without_table_notes_and_reads_unmeasured(tmp_path):
    """An older store (no ``cell_capacitance`` table, like the rig's today) is a note,
    not a crash — and its C_cell is *unmeasured*, never a default."""
    db = tmp_path / "old.db"
    sqlite3.connect(db).close()
    conn = tool.connect_ro(db)
    try:
        rows, note = tool.store_c_cell_rows(conn, BOARD)
    finally:
        conn.close()
    assert rows == [] and "no cell_capacitance table" in note
    assert tool.resolve_c_cell(3, [], rows) == tool.CCell(None, "unmeasured")


def test_regime_b_replay_connection_refuses_writes(project):
    """The positive control for "never writes": the tool's own connection cannot."""
    conn = tool.connect_ro(project / "db" / "softae.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM cell_capacitance")
    finally:
        conn.close()


def test_regime_b_replay_board_from_occupancy_only_when_unique(project):
    conn = tool.connect_ro(project / "db" / "softae.db")
    try:
        assert tool.board_for_runs(conn, [RUN]) == BOARD
        assert tool.board_for_runs(conn, ["no_such_run"]) is None
        assert tool.board_for_runs(conn, []) is None
    finally:
        conn.close()


@pytest.mark.parametrize("text, parsed", [("1-8=2.66e-10", ((1, 8), 2.66e-10)),
                                          ("19=3e-10", ((19, 19), 3e-10))])
def test_regime_b_replay_parse_c_cell_group_and_channel(text, parsed):
    assert tool.parse_c_cell(text) == parsed


@pytest.mark.parametrize("text", ["1-8", "x=1e-10", "1-8=-1", "8-1=1e-10", "1-8=nan"])
def test_regime_b_replay_parse_c_cell_refuses_malformed(text):
    with pytest.raises(argparse.ArgumentTypeError):
        tool.parse_c_cell(text)


def test_regime_b_replay_nothing_selected_fails(capsys):
    assert tool.main([]) == tool.EXIT_FAILED
    assert "Nothing selected" in capsys.readouterr().err
