"""The schema/data epoch ledger and the modality migration (Tier 2 component 3).

Two things land together here because they answer the same question from
opposite ends — *what does a stored row mean?*

``_migrate_modality`` makes a ``measurements`` row say what kind of measurement
it is and where its payload lives, so a future modality does not need a parallel
table (spec §4 component 3).

``schema_version`` exists because of a failure a version number could not have
caught. On 2026-08-07 ``deposit_area_mm2`` changed *derivation* — 4.0 → 18.704 mm²
on the 4-stripe board — so every thickness derived from it moved by 4.676× while
the column name, its units and its type stayed identical (SESSION_MAIL #5). No
schema migration happened, and nothing in a row said its numbers now meant
something else. The ledger records that as a first-class ``'data-epoch'`` entry,
which is what makes the deliberately un-backfilled NULLs on legacy rows
interpretable rather than merely missing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from softae.core.data_store import SCHEMA_EPOCHS, DataStore


@pytest.fixture()
def store(tmp_path: Path) -> DataStore:
    with DataStore(tmp_path / "epoch_project") as ds:
        yield ds


def _legacy_db(project: Path) -> Path:
    """A database predating both the ledger and the modality columns."""
    (project / "db").mkdir(parents=True)
    db_path = project / "db" / "softae.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE measurements (
               measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id         TEXT    NOT NULL,
               channel        INTEGER NOT NULL,
               timestamp      TEXT    NOT NULL,
               eis_file_path  TEXT
           )"""
    )
    conn.execute(
        "INSERT INTO measurements (run_id, channel, timestamp, eis_file_path)"
        " VALUES ('old_run', 4, '2026-01-01T00:00:00Z', 'runs/old_run/eis/a.txt')"
    )
    conn.commit()
    conn.close()
    return db_path


def _columns(store: DataStore, table: str) -> list[str]:
    return [r[1] for r in
            store._conn.execute(f"PRAGMA table_info({table})").fetchall()]


# ---------------------------------------------------------------------------
# The epoch ledger
# ---------------------------------------------------------------------------


class TestEpochLedger:
    def test_a_fresh_store_is_seeded_with_every_declared_epoch(self, store) -> None:
        rows = store.schema_epochs()
        assert [r["version"] for r in rows] == [v for v, _, _ in SCHEMA_EPOCHS]
        assert [r["kind"] for r in rows] == [k for _, k, _ in SCHEMA_EPOCHS]
        assert [r["note"] for r in rows] == [n for _, _, n in SCHEMA_EPOCHS]

    def test_the_ledger_carries_both_kinds_because_one_cannot_stand_for_the_other(
            self, store) -> None:
        """A schema row describes shape; a data-epoch row describes meaning.

        The 4.676x correction changed no shape at all, so a ledger holding only
        'schema' rows would have recorded nothing about the one event that made
        stored values incomparable.
        """
        kinds = {r["kind"] for r in store.schema_epochs()}
        assert kinds == {"schema", "data-epoch"}

    def test_the_data_epoch_row_names_the_correction_and_its_magnitude(
            self, store) -> None:
        """The note must be enough to act on without consulting the mail archive."""
        # Located by VERSION, never by kind-and-position. `kind == "data-epoch"` read
        # epoch 2 only for as long as epoch 2 was the *first* data-epoch and
        # `schema_epochs()` returned rows in version order — and epoch 5 is a
        # data-epoch too, so this already depended on the ordering rather than on
        # anything about epoch 2. It survives appends and breaks on any insertion.
        epoch = next(r for r in store.schema_epochs() if r["version"] == 2)
        assert "2026-08-07" in epoch["note"]
        assert "deposit_area_mm2" in epoch["note"]
        assert "18.704" in epoch["note"] and "4.676" in epoch["note"]
        # The sessile board's *absence* of an area is stated, not left to be
        # inferred from a NULL (SESSION_MAIL #6 commitment 3).
        assert "unavailable" in epoch["note"]

    def test_current_version_is_the_highest_recorded_epoch(self, store) -> None:
        assert store.current_schema_version() == max(v for v, _, _ in SCHEMA_EPOCHS)

    def test_an_unseeded_ledger_reports_zero_rather_than_guessing_a_version(
            self, store) -> None:
        """Zero cannot collide with a real epoch: numbering starts at 1."""
        store._conn.execute("DELETE FROM schema_version")
        assert store.current_schema_version() == 0

    def test_a_bogus_kind_is_refused_by_the_table_itself(self, store) -> None:
        """The vocabulary is a constraint, not a convention.

        A third kind invented at a call site would silently split the ledger into
        rows readers know to check and rows they do not.
        """
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(
                "INSERT INTO schema_version (version, applied_at, kind, note)"
                " VALUES (99, '2026-08-07T00:00:00Z', 'migration', 'x')"
            )


class TestLedgerIsAppendOnly:
    def test_reopening_neither_duplicates_nor_rewrites_a_row(
            self, tmp_path: Path) -> None:
        """Seeding is idempotent, and `applied_at` records first sight.

        Rewriting it on every open would destroy the only record of when this
        database learned an epoch — the timestamp would just track the last time
        anyone opened the file.
        """
        project = tmp_path / "reopen"
        with DataStore(project) as first:
            before = first.schema_epochs()

        with DataStore(project) as second:
            after = second.schema_epochs()

        assert after == before
        assert len(after) == len(SCHEMA_EPOCHS)

    def test_a_row_already_present_survives_a_changed_seed_definition(
            self, tmp_path: Path) -> None:
        """A ledger row is a statement about data written under it.

        Overwriting one would retroactively change what historical rows claim, so
        a differing seed must be added as a new version, never applied in place.
        """
        project = tmp_path / "frozen"
        with DataStore(project) as store:
            store._conn.execute(
                "UPDATE schema_version SET note = 'hand-edited' WHERE version = 1"
            )
            store._conn.commit()

        with DataStore(project) as reopened:
            v1 = next(r for r in reopened.schema_epochs() if r["version"] == 1)
            assert v1["note"] == "hand-edited"

    def test_a_legacy_database_gains_the_ledger_on_open(self, tmp_path: Path) -> None:
        project = tmp_path / "legacy_ledger"
        _legacy_db(project)

        with DataStore(project) as store:
            assert store.current_schema_version() == max(
                v for v, _, _ in SCHEMA_EPOCHS)
            # The pre-existing row is untouched — the ledger explains old rows,
            # it does not rewrite them.
            assert store._conn.execute(
                "SELECT COUNT(*) FROM measurements").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# The modality migration
# ---------------------------------------------------------------------------


class TestModalityMigration:
    def test_a_fresh_database_has_the_columns_from_the_ddl(self, store) -> None:
        assert {"modality", "payload_path", "payload_format", "sample_uuid"} <= set(
            _columns(store, "measurements"))

    def test_a_legacy_row_is_labelled_eis_because_it_provably_is_one(
            self, tmp_path: Path) -> None:
        """The one justified non-absence default in this table.

        Until this migration, `record_measurement` took an `EISResult` and nothing
        else could reach the table — so every pre-existing row is an EIS spectrum
        by construction. 'unknown' would be the false statement here, discarding a
        fact held with certainty. The defaults-record-absence convention exists to
        stop a default inventing a fact, not to stop one recording it.
        """
        project = tmp_path / "legacy_modality"
        _legacy_db(project)

        with DataStore(project) as store:
            row = store._conn.execute(
                "SELECT modality, payload_path, payload_format, sample_uuid "
                "FROM measurements WHERE run_id = 'old_run'"
            ).fetchone()
            assert row["modality"] == "eis"
            # The other three follow the convention unbent: nothing was written,
            # nothing is claimed.
            assert row["payload_path"] is None
            assert row["payload_format"] is None
            assert row["sample_uuid"] is None

    def test_migrating_twice_adds_no_duplicate_column(self, tmp_path: Path) -> None:
        project = tmp_path / "twice"
        _legacy_db(project)

        with DataStore(project):
            pass
        with DataStore(project) as store:
            cols = _columns(store, "measurements")
            for name in ("modality", "payload_path", "payload_format",
                         "sample_uuid"):
                assert cols.count(name) == 1

    def test_the_role_migration_is_untouched_and_still_applies(
            self, tmp_path: Path) -> None:
        """`_migrate_modality` is a separate function by agreement (MAIL #8).

        Both must run on the same legacy table; a merge into one function would
        have made either session's change the other's merge conflict.
        """
        project = tmp_path / "coexist"
        _legacy_db(project)

        with DataStore(project) as store:
            cols = set(_columns(store, "measurements"))
            assert {"role", "fixture_id", "electrode_mode"} <= cols
            assert {"modality", "payload_path"} <= cols
