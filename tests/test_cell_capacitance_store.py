"""``tools/c_cell.py --write`` against the REAL ``DataStore`` (slice 2, L3 seam).

``tests/test_tool_c_cell.py`` (afl's) drives the write path through a fake store
modelling eis-acq's contract; this file closes the seam from the other side: the
same project fixture, the default store class, a typed ``yes``, and the reader
returning what was written. Its fixtures are imported rather than rebuilt so the
two files cannot drift onto different spectra.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from softae.analysis.eis import cell_capacitance as cc
from softae.core.data_store import DataStore
from softae.tools import c_cell as tool
from tests.test_tool_c_cell import RUN, project  # noqa: F401  (fixture)


def _write(project_dir) -> int:
    return tool.main(["--board", "5", "--run", RUN, "--project", str(project_dir),
                      "--write", "--accepted-by", "CO"],
                     input_fn=lambda prompt: "yes", store_cls=None)


def _rows(project_dir) -> list[dict]:
    conn = sqlite3.connect(project_dir / "db" / "softae.db")
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM cell_capacitance ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_c_cell_write_real_store_appends_one_row_per_group_and_reads_back(
    project,  # noqa: F811
) -> None:
    assert _write(project) == tool.EXIT_OK

    (row,) = _rows(project)                    # one qualifying group: 9-16
    assert (row["board_id"], row["channel"], row["group_lo"], row["group_hi"]) == (
        5, None, 9, 16)
    assert row["method"] == cc.METHOD and row["accepted_by"] == "CO"
    assert row["n_spectra"] == 2               # the film (ch11) is not a source
    with DataStore(project) as store:
        got = store.cell_capacitance(5, 12)
        assert store.cell_capacitance(5, 3) is None
    assert got["id"] == row["id"]
    assert got["c_cell_F"] == pytest.approx(np.median([2.0e-10, 3.0e-10]), rel=0.01)
    assert len(got["source_measurement_ids"]) == 2
    assert all(isinstance(i, int) for i in got["source_measurement_ids"])


def test_c_cell_write_real_store_twice_keeps_history_one_current(
    project,  # noqa: F811
) -> None:
    """A re-accepted group supersedes, never overwrites: the drift record."""
    assert _write(project) == tool.EXIT_OK
    assert _write(project) == tool.EXIT_OK

    first, second = _rows(project)
    assert first["superseded_at"] is not None and second["superseded_at"] is None
    with DataStore(project) as store:
        assert store.cell_capacitance(5, 12)["id"] == second["id"]
