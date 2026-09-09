"""Tests for tools/dbq.py - the read-only DataStore query tool.

`tools/` is not an importable package (the shipped package is `src/softae/tools`),
so the module is loaded by path, exactly as `tests/test_whose.py` does.

Every test runs against a temporary fixture database built by `incident_db`.
The real DataStore is never opened here.

The load-bearing group is `TestTheNumberThatWasNotShown`. This tool exists
because an agent read fifteen rows and reported them as thirty-two rows' worth
of conclusion, so the assertion that actually matters is not "it printed a
warning" but **"it printed the total"**. Those tests are paired with a
complete-result control on the same database, so that a banner which fires on
everything is distinguishable from one that discriminates - the SUBAGENT_RULES
3.1 shape, where the conservative answer and the broken answer look alike.
"""

from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DBQ_PATH = REPO_ROOT / "tools" / "dbq.py"


def _load_dbq():
    spec = importlib.util.spec_from_file_location("dbq_tool", DBQ_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dbq_tool"] = module
    spec.loader.exec_module(module)
    return module


dbq = _load_dbq()


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def incident_db(tmp_path: Path) -> Path:
    """A database reproducing the shape of the real incident.

    32 distinct channels carry `role='sample'` rows. Channels 1-16 carry many
    rows each and channels 17-32 carry few, so

        ... GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 15

    returns fifteen rows that are *all* drawn from 1-16 - which is precisely
    what let the original agent conclude "all sample measurements are on
    channels 1-16". The fixture is built so the wrong conclusion is available
    to be made; the assertions are about whether the tool prevents it.
    """
    path = tmp_path / "fixture.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE measurements (id INTEGER PRIMARY KEY, channel INT, role TEXT)")
    rows = []
    for channel in range(1, 17):
        rows += [(channel, "sample")] * (100 + channel)
    for channel in range(17, 33):
        rows += [(channel, "sample")] * 5
    rows += [(1, "blank")] * 7
    con.executemany("INSERT INTO measurements (channel, role) VALUES (?, ?)", rows)
    con.commit()
    con.close()
    return path


def run(db: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = dbq.main(["--db", str(db), *argv], out=out, err=err)
    return code, out.getvalue(), err.getvalue()


SAMPLE_BY_CHANNEL = (
    "SELECT channel, COUNT(*) FROM measurements WHERE role='sample' "
    "GROUP BY channel ORDER BY COUNT(*) DESC"
)


# --------------------------------------------------------------------------- #


class TestTheNumberThatWasNotShown:
    """The whole point: the result names the rows it did not show."""

    def test_limited_query_reports_shown_of_total(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, SAMPLE_BY_CHANNEL + " LIMIT 15")
        assert code == dbq.EXIT_OK
        assert "15 of 32" in out
        assert "TRUNCATED" in out
        assert "17 row(s) NOT SHOWN" in out

    def test_the_incident_conclusion_is_no_longer_available(self, incident_db: Path) -> None:
        """The fifteen visible channels really are all within 1-16 - and the
        output still makes "all channels are 1-16" unsayable, because 32 is on
        the page next to them."""
        _, out, _ = run(incident_db, SAMPLE_BY_CHANNEL + " LIMIT 15")
        table = [line for line in out.splitlines() if line and line[0].isdigit()]
        shown = {int(line.split()[0]) for line in table}
        assert len(shown) == 15 and max(shown) <= 16  # the trap is still set
        assert "of 32" in out  # and the tool disarms it

    def test_unlimited_query_states_completeness(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, SAMPLE_BY_CHANNEL)
        assert code == dbq.EXIT_OK
        assert "32 of 32" in out
        assert "COMPLETE" in out

    def test_a_complete_result_carries_no_truncation_banner(self, incident_db: Path) -> None:
        """The control. Without this, a banner printed unconditionally would
        satisfy every other test in this class."""
        _, out, _ = run(incident_db, SAMPLE_BY_CHANNEL)
        assert "TRUNCATED" not in out
        assert dbq.BANNER not in out

    def test_display_cap_is_reported_as_truncation(self, incident_db: Path) -> None:
        """A cap the tool imposes itself is no more forgivable than a LIMIT."""
        code, out, _ = run(incident_db, "--max-rows", "4", SAMPLE_BY_CHANNEL)
        assert code == dbq.EXIT_OK
        assert "4 of 32" in out
        assert "display cap" in out

    def test_strict_exits_nonzero_only_when_truncated(self, incident_db: Path) -> None:
        truncated, _, _ = run(incident_db, "--strict", SAMPLE_BY_CHANNEL + " LIMIT 3")
        complete, _, _ = run(incident_db, "--strict", SAMPLE_BY_CHANNEL)
        assert truncated == dbq.EXIT_TRUNCATED
        assert complete == dbq.EXIT_OK

    def test_total_counts_matching_rows_not_returned_rows(self, incident_db: Path) -> None:
        """`LIMIT 1` over the fixture's 1816 sample rows must say 1816, not 1.

        1816 = sum(100 + c for c in 1..16) + 16 * 5; the 7 `blank` rows are
        excluded by the WHERE, so the total is the population the *query*
        matches, not the table's row count.
        """
        _, out, _ = run(incident_db, "SELECT id FROM measurements WHERE role='sample' LIMIT 1")
        assert "1 of 1816" in out


class TestUnknownIsNotSpelledLikeComplete:
    """SUBAGENT_RULES 3.1(a): a total that could not be obtained must never
    render as a total that was obtained and found equal."""

    def test_limit_in_a_subquery_reports_unknown(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, "SELECT * FROM (SELECT channel FROM measurements LIMIT 3)")
        assert code == dbq.EXIT_OK
        assert "COMPLETENESS UNKNOWN" in out
        assert "COMPLETE," not in out

    def test_unknown_does_not_fall_back_to_the_returned_count(self, incident_db: Path) -> None:
        """The tempting bug: reporting "3 of 3" because 3 rows came back. That
        would print a truncated slice as the whole set."""
        _, out, _ = run(incident_db, "SELECT * FROM (SELECT channel FROM measurements LIMIT 3)")
        assert "3 of 3" not in out

    def test_unlimited_count_returns_none_rather_than_guessing(self, incident_db: Path) -> None:
        con = dbq.connect_ro(incident_db)
        try:
            total, reason = dbq.unlimited_count(con, "SELECT this is not sql")
            assert total is None
            assert reason
        finally:
            con.close()


class TestItWillNotWrite:
    def test_non_select_is_refused(self, incident_db: Path) -> None:
        code, _, err = run(incident_db, "DELETE FROM measurements")
        assert code == dbq.EXIT_ERROR
        assert "refused" in err

    @pytest.mark.parametrize(
        "sql",
        [
            "UPDATE measurements SET channel=99",
            "INSERT INTO measurements (channel) VALUES (1)",
            "DROP TABLE measurements",
            "CREATE TABLE t (x)",
            "ATTACH DATABASE 'other.db' AS o",
            "PRAGMA writable_schema=1",
        ],
    )
    def test_every_writing_statement_is_refused(self, incident_db: Path, sql: str) -> None:
        code, _, _ = run(incident_db, sql)
        assert code == dbq.EXIT_ERROR

    def test_a_chained_statement_is_refused(self, incident_db: Path) -> None:
        code, _, err = run(incident_db, "SELECT 1; DROP TABLE measurements")
        assert code == dbq.EXIT_ERROR
        assert "more than one statement" in err

    def test_the_connection_itself_rejects_writes(self, incident_db: Path) -> None:
        """The text gate above is a courtesy; this is the enforcement. If the
        gate were bypassed entirely, the connection must still refuse."""
        con = dbq.connect_ro(incident_db)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                con.execute("DELETE FROM measurements")
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                con.execute("CREATE TABLE scratch (x)")
        finally:
            con.close()

    def test_the_database_is_unchanged_after_a_refused_write(self, incident_db: Path) -> None:
        before = incident_db.read_bytes()
        run(incident_db, "DELETE FROM measurements")
        assert incident_db.read_bytes() == before

    def test_a_missing_database_is_not_an_empty_one(self, tmp_path: Path) -> None:
        code, _, err = run(tmp_path / "nope.db", "SELECT 1")
        assert code == dbq.EXIT_ERROR
        assert "not found" in err


class TestStatementParsing:
    def test_a_trailing_string_literal_survives_limit_stripping(self, incident_db: Path) -> None:
        """Regression: literals are masked to spaces, so trimming the *mask*
        cut `'sample'` off the end of the statement before wrapping it, and
        every query ending in a string reported UNKNOWN."""
        code, out, _ = run(
            incident_db,
            "SELECT COUNT(*) AS n FROM measurements WHERE role='sample'",
        )
        assert code == dbq.EXIT_OK
        assert "COMPLETE" in out
        assert "UNKNOWN" not in out

    def test_a_semicolon_inside_a_literal_is_not_a_chain(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, "SELECT ';' AS c")
        assert code == dbq.EXIT_OK
        assert "1 of 1" in out

    def test_the_word_limit_inside_a_literal_is_not_a_limit(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, "SELECT 'LIMIT 5' AS c")
        assert code == dbq.EXIT_OK
        assert "COMPLETE" in out

    def test_a_trailing_semicolon_does_not_defeat_the_count(self, incident_db: Path) -> None:
        _, out, _ = run(incident_db, SAMPLE_BY_CHANNEL + " LIMIT 3;")
        assert "3 of 32" in out

    def test_a_cte_is_counted(self, incident_db: Path) -> None:
        _, out, _ = run(
            incident_db,
            "WITH s AS (SELECT channel FROM measurements WHERE role='sample') "
            "SELECT DISTINCT channel FROM s LIMIT 5",
        )
        assert "5 of 32" in out

    def test_limit_with_offset_is_stripped(self, incident_db: Path) -> None:
        _, out, _ = run(incident_db, SAMPLE_BY_CHANNEL + " LIMIT 5 OFFSET 10")
        assert "5 of 32" in out

    @pytest.mark.parametrize(
        "sql, masked_free_of",
        [
            ("SELECT 'a;b'", ";"),
            ("SELECT 1 -- limit 5\n", "limit"),
            ("SELECT /* limit 5 */ 1", "limit"),
        ],
    )
    def test_mask_literals_blanks_strings_and_comments(self, sql: str, masked_free_of: str) -> None:
        mask = dbq.mask_literals(sql)
        assert len(mask) == len(sql)
        assert masked_free_of not in mask


class TestPathResolution:
    def test_the_configured_project_dir_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`loader.data_project_dir()` returns the raw config string, whose
        default is the unexpanded `~/softae_data`. A literal `~` directory is
        the bug this guards."""

        class FakeLoader:
            @staticmethod
            def data_project_dir() -> str:
                return "~/softae_data"

            @staticmethod
            def data_db_filename() -> str:
                return "softae.db"

        monkeypatch.setattr(dbq, "_import_loader", lambda: FakeLoader)
        resolved = dbq.resolve_db_path()
        assert "~" not in str(resolved)
        assert resolved == Path.home() / "softae_data" / "db" / "softae.db"

    def test_explicit_db_wins(self, tmp_path: Path) -> None:
        assert dbq.resolve_db_path(str(tmp_path / "x.db")) == tmp_path / "x.db"


class TestInspection:
    def test_tables_lists_every_table_and_says_so(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, "--tables")
        assert code == dbq.EXIT_OK
        assert "measurements" in out
        assert "COMPLETE" in out

    def test_schema_reports_columns(self, incident_db: Path) -> None:
        code, out, _ = run(incident_db, "--schema", "measurements")
        assert code == dbq.EXIT_OK
        assert "channel" in out and "role" in out

    def test_exactly_one_mode_is_required(self, incident_db: Path) -> None:
        both, _, err = run(incident_db, "--tables", "SELECT 1")
        neither, _, _ = run(incident_db)
        assert both == dbq.EXIT_ERROR and neither == dbq.EXIT_ERROR
        assert "exactly one" in err
