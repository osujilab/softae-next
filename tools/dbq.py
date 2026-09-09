#!/usr/bin/env python
"""dbq.py - read-only DataStore queries that report what they did NOT show.

NOTE the shebang: `python`, not `python3`. On this machine the Windows `py`
launcher honours the shebang, and `python3` resolves to the Store's Python
3.10, which has no `tomllib` -- so `py tools/whose.py` currently dies with
"needs Python 3.11+" for exactly that reason. `python` resolves to 3.11.3.

Sibling of `tools/whose.py`. Same house style: it fails loudly rather than
answering from a state it cannot trust.

The failure this exists to prevent is not "I forgot that I truncated". It is
**"I did not know 32 was not 15."** An agent ran

    SELECT channel, COUNT(*) FROM measurements
    WHERE role='sample' GROUP BY channel ORDER BY COUNT(*) DESC LIMIT 15

got fifteen rows that happened to be channels 1..16, and published *"all sample
measurements are on channels 1-16"* to three sessions and the operator. There
are 32. Its own `LIMIT` had become a finding, and nothing in the output said so.

So a banner reading "output was truncated" is the weak version of this tool.
The strong version is the one below: **every result prints the unlimited count
for the same query**, so the reader sees `15 of 32` and cannot mistake a slice
for the set. When that count cannot be obtained, the tool says UNKNOWN - never
"complete" - because an unknown spelled like a clean answer is the failure shape
`SUBAGENT_RULES.md` 3.1(a) is about.

Usage:

    dbq.py "SELECT ..."            # run a query
    dbq.py --tables                # list tables, with row counts
    dbq.py --schema measurements   # PRAGMA table_info
    dbq.py --db path/to.db "..."   # override the resolved DataStore

Exit codes:

    0   the query ran
    1   the query ran and the result is TRUNCATED or its completeness is
        UNKNOWN -- only with --strict; the banner is the default signal
    2   refused, or could not run: not a read-only statement, more than one
        statement, no database, or SQL error
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_TRUNCATED = 1
EXIT_ERROR = 2

#: Rows printed before the display itself truncates. A display cap is a
#: truncation like any other and is reported in the same provenance block.
DEFAULT_MAX_ROWS = 200
#: Characters printed per cell before the cell is elided.
CELL_WIDTH = 200

ALLOWED_PREFIXES = ("select", "with")
ALLOWED_PRAGMAS = ("table_info", "table_list", "index_list", "index_info", "database_list")


class QueryError(Exception):
    """The statement will not be run, or could not be."""


# --------------------------------------------------------------------------- #
# statement inspection
# --------------------------------------------------------------------------- #


def mask_literals(sql: str) -> str:
    """Blank out string/identifier literals and comments, preserving offsets.

    Every scan below (statement counting, LIMIT detection, paren balance) runs
    on the mask, so a `;` or the word `limit` inside a string literal cannot be
    mistaken for syntax.
    """
    out = list(sql)
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled = escaped
                        j += 2
                        continue
                    break
                j += 1
            for k in range(i, min(j + 1, n)):
                out[k] = " "
            i = j + 1
        elif ch == "[":
            j = sql.find("]", i)
            j = n - 1 if j < 0 else j
            for k in range(i, j + 1):
                out[k] = " "
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                out[k] = " "
            i = j
        else:
            i += 1
    return "".join(out)


def check_single_statement(sql: str, mask: str) -> None:
    if not sql.strip():
        raise QueryError("empty statement")
    body = mask.rstrip().rstrip(";")
    if ";" in body:
        raise QueryError(
            "more than one statement. dbq.py runs exactly one read-only "
            "statement; chaining is how a write hides behind a SELECT."
        )


def check_read_only(mask: str) -> str:
    """Return the statement kind, or refuse. Text gate only.

    The *enforcement* is the `mode=ro` connection; this exists so a rejected
    statement gets an explanation instead of a bare sqlite error.
    """
    head = mask.strip().lstrip("(").strip().lower()
    first = re.split(r"[^a-z_]", head, maxsplit=1)[0]
    if first in ALLOWED_PREFIXES:
        return first
    if first == "pragma":
        rest = head[len("pragma"):].strip()
        name = re.split(r"[^a-z_]", rest, maxsplit=1)[0]
        if name in ALLOWED_PRAGMAS:
            return "pragma"
        raise QueryError(
            f"PRAGMA {name or '?'} is not on the read-only allow-list "
            f"({', '.join(ALLOWED_PRAGMAS)})"
        )
    raise QueryError(
        f"refused: statement starts with '{first or '?'}'. Only SELECT, WITH "
        "and PRAGMA table_info/table_list are permitted. This tool never opens "
        "a writable connection to the DataStore."
    )


#: A trailing LIMIT clause whose argument is simple enough to strip safely.
_TAIL_LIMIT = re.compile(
    r"\blimit\b\s+(?:\d+|\?|:\w+)\s*(?:(?:,|\boffset\b)\s*(?:\d+|\?|:\w+)\s*)?$",
    re.IGNORECASE,
)
_ANY_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


def split_trailing_limit(sql: str, mask: str) -> tuple[str, str | None, bool]:
    """Split a top-level trailing LIMIT off the statement.

    Returns ``(base_sql, limit_text, strippable)``.

    ``strippable`` is False when the word LIMIT appears but this function is not
    confident it owns the outermost query - a LIMIT inside a subquery, a second
    LIMIT further in, or an expression argument. The caller must then report the
    total as UNKNOWN. Guessing here would print a wrong `n of m`, which is worse
    than admitting to not knowing.
    """
    # Trim trailing whitespace and real (unmasked) semicolons, measured on the
    # ORIGINAL text. Trimming the mask instead would eat a trailing string
    # literal -- `WHERE role='sample'` masks to eight spaces, and rstrip() then
    # cuts the literal out of the statement it is about to wrap.
    end = len(sql)
    while end > 0 and (sql[end - 1].isspace() or mask[end - 1] == ";"):
        end -= 1
    body_mask = mask[:end]
    body_sql = sql[:end]

    match = _TAIL_LIMIT.search(body_mask)
    if match is None:
        # No strippable trailing LIMIT. If the token appears anywhere else we
        # cannot claim the result is complete.
        return body_sql, None, not _ANY_LIMIT.search(body_mask)

    prefix = body_mask[: match.start()]
    if prefix.count("(") != prefix.count(")") or _ANY_LIMIT.search(prefix):
        return body_sql, sql[match.start(): end].strip(), False
    return sql[: match.start()].rstrip(), sql[match.start(): end].strip(), True


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #


def _import_loader():
    """Import `softae.config.loader`, adding the repo's `src/` if it is absent.

    The venv carries an editable-install `.pth` that puts `src/` on `sys.path`,
    but this tool is also run under the bare `py` launcher, where it is not.
    Falling back to the repo layout keeps `py tools/dbq.py "..."` working from
    any interpreter rather than sending the user to `--db`.
    """
    try:
        from softae.config import loader
    except ImportError:
        src = Path(__file__).resolve().parent.parent / "src"
        if not src.is_dir():
            raise
        sys.path.insert(0, str(src))
        from softae.config import loader
    return loader


def resolve_db_path(explicit: str | None = None) -> Path:
    """Locate the DataStore SQLite file. Never hardcoded.

    `loader.data_project_dir()` returns the raw config string, whose default is
    the *unexpanded* ``"~/softae_data"`` - still true at `config/loader.py:540`.
    Callers across the tree each re-`expanduser()` it; so does this one.
    `DataStore.__init__` is deliberately not used: it mkdirs and opens a
    writable connection.
    """
    if explicit:
        return Path(explicit).expanduser()
    try:
        loader = _import_loader()
    except ImportError as exc:  # pragma: no cover - environment failure
        raise QueryError(
            f"cannot import softae.config.loader to resolve the DataStore: {exc}\n"
            "  Pass --db <path> to query a database directly."
        ) from exc
    project = Path(loader.data_project_dir()).expanduser()
    return project / "db" / loader.data_db_filename()


def connect_ro(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise QueryError(
            f"database not found: {path}\n"
            "  Refusing to continue: a missing DataStore must not read as an "
            "empty one."
        )
    try:
        return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise QueryError(f"could not open read-only: {path}\n  {exc}") from exc


def unlimited_count(con: sqlite3.Connection, base_sql: str) -> tuple[int | None, str]:
    """Rows the query would match with no LIMIT.

    Wraps the LIMIT-stripped statement as ``SELECT COUNT(*) FROM (<base>)``.

    When that fails - an exotic statement SQLite will not accept as a subquery -
    this returns ``(None, reason)`` and the caller prints UNKNOWN. It does not
    fall back to the returned row count: that would report a truncated slice as
    the whole set, which is the exact defect this tool exists to prevent.
    """
    try:
        row = con.execute(f"SELECT COUNT(*) FROM ({base_sql})").fetchone()
    except sqlite3.Error as exc:
        return None, f"count wrap rejected by sqlite: {exc}"
    if not row or row[0] is None:
        return None, "count wrap returned no value"
    return int(row[0]), ""


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def render_cell(value: object) -> tuple[str, bool]:
    if value is None:
        return "NULL", False
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<blob {len(bytes(value))} bytes>", False
    text = str(value).replace("\n", "\\n").replace("\t", "\\t")
    if len(text) > CELL_WIDTH:
        return text[: CELL_WIDTH - 1] + "\u2026", True
    return text, False


def format_table(columns: list[str], rows: list[tuple], out) -> int:
    """Print an aligned table. Returns the number of elided cells."""
    if not columns:
        return 0
    cells: list[list[str]] = []
    elided = 0
    for row in rows:
        rendered = []
        for value in row:
            text, cut = render_cell(value)
            elided += int(cut)
            rendered.append(text)
        cells.append(rendered)
    widths = [len(c) for c in columns]
    for row_cells in cells:
        for i, text in enumerate(row_cells):
            if i < len(widths):
                widths[i] = max(widths[i], len(text))
    print("  ".join(c.ljust(widths[i]) for i, c in enumerate(columns)).rstrip(), file=out)
    print("  ".join("-" * w for w in widths), file=out)
    for row_cells in cells:
        print(
            "  ".join(
                text.ljust(widths[i]) if i < len(widths) else text
                for i, text in enumerate(row_cells)
            ).rstrip(),
            file=out,
        )
    return elided


BANNER = "!" * 74


def print_provenance(
    out,
    *,
    db_path: Path,
    shown: int,
    total: int | None,
    total_reason: str,
    limit_text: str | None,
    capped: bool,
    max_rows: int,
    elided: int,
) -> bool:
    """Print the provenance block. Returns True if the result is not complete.

    Complete and incomplete are rendered differently on purpose: a truncated
    result must not be mistakable for a full one by a reader skimming.
    """
    print(f"-- db: {db_path} (read-only)", file=out)

    causes: list[str] = []
    if limit_text:
        causes.append(f"LIMIT clause in the query: {limit_text}")
    if capped:
        causes.append(f"display cap --max-rows {max_rows}")

    if total is None:
        print(f"-- rows shown: {shown}", file=out)
        print(BANNER, file=out)
        print("!! COMPLETENESS UNKNOWN. The total for this query could NOT be", file=out)
        print(f"!! determined: {total_reason}", file=out)
        for cause in causes:
            print(f"!!   truncating construct: {cause}", file=out)
        print("!! Treat these rows as a SAMPLE, not as the set. Re-run without", file=out)
        print("!! the LIMIT, or as SELECT COUNT(*), before stating anything", file=out)
        print("!! about how many or which values exist.", file=out)
        print(BANNER, file=out)
        return True

    if shown >= total and not causes:
        print(f"-- rows: {shown} of {total} -- COMPLETE, no LIMIT or cap in effect.", file=out)
        if elided:
            print(
                f"-- note: {elided} cell(s) elided at {CELL_WIDTH} chars "
                "(the rows themselves are complete).",
                file=out,
            )
        return False

    hidden = total - shown
    print(f"-- rows: {shown} of {total}", file=out)
    print(BANNER, file=out)
    print(f"!! TRUNCATED -- {shown} of {total} rows shown; {hidden} row(s) NOT SHOWN.", file=out)
    for cause in causes:
        print(f"!!   cause: {cause}", file=out)
    print(f"!! The {shown} value(s) above are NOT the full set. Any claim about", file=out)
    print(f"!! how many, which, or 'all' must be made against {total}, not {shown}.", file=out)
    print(BANNER, file=out)
    if elided:
        print(f"-- note: {elided} cell(s) also elided at {CELL_WIDTH} chars.", file=out)
    return True


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #


def run_query(con: sqlite3.Connection, sql: str, max_rows: int, db_path: Path, out) -> int:
    mask = mask_literals(sql)
    check_single_statement(sql, mask)
    kind = check_read_only(mask)

    try:
        cursor = con.execute(sql)
        rows = cursor.fetchall()
        columns = [d[0] for d in (cursor.description or [])]
    except sqlite3.Error as exc:
        raise QueryError(f"sqlite error: {exc}") from exc

    returned = len(rows)
    capped = 0 < max_rows < returned
    shown_rows = rows[:max_rows] if capped else rows

    limit_text: str | None = None
    if kind == "pragma":
        total: int | None = returned
        reason = ""
    else:
        base_sql, limit_text, strippable = split_trailing_limit(sql, mask)
        if limit_text and not strippable:
            total, reason = None, (
                "a LIMIT is present that dbq.py cannot safely strip "
                "(subquery, compound, or expression argument)"
            )
        elif not strippable:
            total, reason = None, "the word LIMIT appears in a position dbq.py does not parse"
        else:
            total, reason = unlimited_count(con, base_sql)

    elided = format_table(columns, shown_rows, out) if columns else 0
    if not columns:
        print("-- statement returned no columns", file=out)
    incomplete = print_provenance(
        out,
        db_path=db_path,
        shown=len(shown_rows),
        total=total,
        total_reason=reason,
        limit_text=limit_text,
        capped=capped,
        max_rows=max_rows,
        elided=elided,
    )
    return EXIT_TRUNCATED if incomplete else EXIT_OK


def list_tables(con: sqlite3.Connection, db_path: Path, out) -> int:
    names = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    rows = [(name, con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]) for name in names]
    format_table(["table", "rows"], rows, out)
    print(f"-- db: {db_path} (read-only)", file=out)
    print(f"-- rows: {len(rows)} of {len(rows)} -- COMPLETE, every table listed.", file=out)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbq.py",
        description="Read-only DataStore query. Always reports the rows it did not show.",
    )
    parser.add_argument("sql", nargs="?", help="a single SELECT / WITH / PRAGMA statement")
    parser.add_argument("--db", help="database file (default: the configured DataStore)")
    parser.add_argument("--tables", action="store_true", help="list tables with row counts")
    parser.add_argument("--schema", metavar="TABLE", help="columns of TABLE (PRAGMA table_info)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="rows to print before the display truncates, reported as such "
        f"(default {DEFAULT_MAX_ROWS}; 0 = no cap)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when the result is truncated or its completeness is unknown",
    )
    return parser


def _use_utf8(stream) -> None:
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main(argv: list[str] | None = None, *, out=None, err=None) -> int:
    if out is None:
        _use_utf8(sys.stdout)
    if err is None:
        _use_utf8(sys.stderr)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(argv)

    if sum([bool(args.sql), args.tables, bool(args.schema)]) != 1:
        print("dbq.py: give exactly one of: a SQL string, --tables, --schema TABLE", file=err)
        return EXIT_ERROR

    con = None
    try:
        db_path = resolve_db_path(args.db)
        con = connect_ro(db_path)
        if args.tables:
            return list_tables(con, db_path, out)
        sql = args.sql or f"PRAGMA table_info({args.schema})"
        status = run_query(con, sql, max(0, args.max_rows), db_path, out)
        return status if args.strict else EXIT_OK
    except QueryError as exc:
        print(f"dbq.py: {exc}", file=err)
        return EXIT_ERROR
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
