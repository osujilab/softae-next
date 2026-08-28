"""Tests for the DataStore class (task d-ii).

Coverage:
- Construction and directory creation
- Run lifecycle (start, finish, query)
- Measurement recording
- Multi-stage conditions recording
- Fit result recording
- Formulation recording
- Query methods with filters
- Temperature-range query via conditions join
- Context manager
- Error handling (missing measurement_id)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from softae.core.data_store import FIT_QUALITY_COLUMNS, DataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eis_result(channel: int = 1, npts: int = 10) -> "EISResult":
    """Create a minimal EISResult for testing."""
    from softae.analysis.eis_data import EISResult

    freq = np.logspace(2, 5, npts)
    z_mag = np.full(npts, 1000.0)
    phase = np.full(npts, -10.0)
    z_real = np.full(npts, 980.0)
    z_imag = np.full(npts, 170.0)
    return EISResult(
        channel=channel,
        frequency=freq,
        z_magnitude=z_mag,
        phase=phase,
        z_real=z_real,
        z_imag_neg=z_imag,
        timestamp=datetime(2026, 3, 6, 12, 0, 0),
        measurement_time_s=5.0,
        eis_params={"npts": npts, "f_hi": 200_000},
    )


@dataclass
class _FakeFitResult:
    """Minimal stand-in for FitResult to avoid importing impedance pkg."""
    model_name: str = "simpleSalt"
    parameters: np.ndarray = field(default_factory=lambda: np.array([100.0, 1e-7, 0.7, 1000.0, 1e-10]))
    R0: float = 100.0
    R1: float = 1000.0
    success: bool = True
    error_msg: str = ""
    #: Which route produced ``R1``. The engine sets this as a plain attribute on the
    #: real ``FitResult`` (``engine.py``: ``fit.estimator = TWO_POINT``) rather than
    #: declaring it, and ``_engine_label`` reads it with ``getattr(..., None)``, so
    #: ``None`` here is the ordinary CPE fit — the same thing an unset attribute means.
    estimator: str | None = None
    #: Goodness-of-fit metrics, as ``compute_fit_quality`` leaves them on the real
    #: ``FitResult``. Empty by default because that is what a failed fit — or one
    #: with no ``z_fit`` to compare against — actually carries.
    quality: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> DataStore:
    """Create a DataStore backed by a temporary directory."""
    with DataStore(tmp_path / "test_project") as ds:
        yield ds


@pytest.fixture()
def store_with_run(store: DataStore) -> tuple[DataStore, str]:
    """DataStore with a single started run."""
    run_id = store.start_run(
        "ht_experiment",
        '{"instruments": {}}',
        mode="full",
        pcb_name="SoftAE_IDE_EIS",
        eis_preset="Standard",
    )
    return store, run_id


# ---------------------------------------------------------------------------
# Electrode occupancy (persistent, single-use wells)
# ---------------------------------------------------------------------------


class TestElectrodeOccupancy:
    def test_record_and_query_occupied(self, store: DataStore) -> None:
        store.record_electrode_cast(0, 3, run_id="r1", iteration=2)
        store.record_electrode_cast(0, 7)
        assert store.occupied_electrodes(0) == {3, 7}
        assert store.occupied_electrodes(1) == set()  # fresh board is empty

    def test_record_is_idempotent_per_well(self, store: DataStore) -> None:
        store.record_electrode_cast(0, 5)
        store.record_electrode_cast(0, 5)  # same well again → no duplicate
        assert store.occupied_electrodes(0) == {5}

    def test_current_board_id_tracks_max(self, store: DataStore) -> None:
        assert store.current_board_id() == 0  # nothing cast yet
        store.record_electrode_cast(0, 1)
        store.record_electrode_cast(2, 1)
        assert store.current_board_id() == 2

    def test_clear_board_forgets_occupancy(self, store: DataStore) -> None:
        store.record_electrode_cast(1, 1)
        store.record_electrode_cast(1, 2)
        store.clear_board(1)
        assert store.occupied_electrodes(1) == set()

    def test_occupancy_persists_across_reopen(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj") as ds:
            ds.record_electrode_cast(0, 4, run_id="r1", iteration=1)
        # Reopen the same project (simulates a GUI restart).
        with DataStore(tmp_path / "proj") as ds2:
            assert ds2.occupied_electrodes(0) == {4}
            assert ds2.current_board_id() == 0


class TestActiveBoardPointer:
    """The board pointer is durable, independent of any cast landing on it."""

    def test_set_active_board_moves_pointer(self, store: DataStore) -> None:
        store.record_electrode_cast(0, 1)
        store.set_active_board(1)
        assert store.current_board_id() == 1
        assert store.occupied_electrodes(1) == set()  # fresh plate, no casts

    def test_swap_survives_reopen_with_no_casts(self, tmp_path: Path) -> None:
        """The regression this table exists for: swap, shut down, reopen."""
        with DataStore(tmp_path / "proj") as ds:
            ds.record_electrode_cast(0, 1)
            ds.record_electrode_cast(0, 2)
            ds.set_active_board(1)          # board replaced; nothing cast yet
        with DataStore(tmp_path / "proj") as ds2:
            assert ds2.current_board_id() == 1        # swap remembered
            assert ds2.occupied_electrodes(1) == set()  # all wells free

    def test_pointer_is_monotonic(self, store: DataStore) -> None:
        store.set_active_board(3)
        store.set_active_board(1)  # never moves backwards
        assert store.current_board_id() == 3

    def test_falls_back_to_max_for_legacy_projects(self, store: DataStore) -> None:
        """A project predating board_state still resolves via MAX(board_id)."""
        store._conn.execute("DELETE FROM board_state")
        store._conn.commit()
        store.record_electrode_cast(2, 1)
        assert store.current_board_id() == 2

    def test_cast_on_higher_board_wins_over_stale_pointer(self, store: DataStore) -> None:
        store.set_active_board(1)
        store.record_electrode_cast(4, 1)  # cast recorded above the pointer
        assert store.current_board_id() == 4

    def test_clear_board_does_not_regress_pointer(self, store: DataStore) -> None:
        store.record_electrode_cast(2, 1)
        store.set_active_board(2)
        store.clear_board(2)  # records wiped, but we are still on plate 2
        assert store.current_board_id() == 2


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_project_subdirs(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj"):
            assert (tmp_path / "proj" / "db").is_dir()
            assert (tmp_path / "proj" / "runs").is_dir()
            assert (tmp_path / "proj" / "formulations").is_dir()

    def test_creates_db_file(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj"):
            assert (tmp_path / "proj" / "db" / "softae.db").is_file()

    def test_custom_db_filename(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj", db_filename="custom.db"):
            assert (tmp_path / "proj" / "db" / "custom.db").is_file()

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        with DataStore("~/softae_test_proj") as ds:
            assert ds.project_dir.is_dir()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_start_run_returns_id(self, store: DataStore) -> None:
        run_id = store.start_run("test_wf", "{}")
        assert "test_wf" in run_id
        assert len(run_id) > 10

    def test_start_run_creates_dirs(self, store: DataStore) -> None:
        run_id = store.start_run("test_wf", "{}")
        assert store.eis_dir(run_id).is_dir()
        assert (store.run_dir(run_id) / "images").is_dir()
        assert (store.run_dir(run_id) / "config_snapshot.toml").is_file()

    def test_start_run_inserts_row(self, store: DataStore) -> None:
        run_id = store.start_run("test_wf", "{}", campaign="q1", quality="explore")
        rows = store.query_runs()
        assert len(rows) == 1
        assert rows[0]["run_id"] == run_id
        assert rows[0]["status"] == "running"
        assert rows[0]["campaign"] == "q1"

    def test_finish_run(self, store_with_run) -> None:
        store, run_id = store_with_run
        store.finish_run(run_id, "done")
        rows = store.query_runs()
        assert rows[0]["status"] == "done"
        assert rows[0]["finished_at"] is not None

    def test_finish_run_aborted(self, store_with_run) -> None:
        store, run_id = store_with_run
        store.finish_run(run_id, "aborted")
        assert store.query_runs()[0]["status"] == "aborted"

    def test_query_runs_by_campaign(self, store: DataStore) -> None:
        store.start_run("a", "{}", campaign="alpha")
        store.start_run("b", "{}", campaign="beta")
        assert len(store.query_runs(campaign="alpha")) == 1


# ---------------------------------------------------------------------------
# Measurement recording
# ---------------------------------------------------------------------------


class TestMeasurements:
    def test_record_returns_id(self, store_with_run) -> None:
        store, run_id = store_with_run
        eis = _make_eis_result(channel=1)
        mid = store.record_measurement(run_id, eis)
        assert isinstance(mid, int) and mid > 0

    def test_query_by_channel(self, store_with_run) -> None:
        store, run_id = store_with_run
        for ch in [1, 2, 3]:
            store.record_measurement(run_id, _make_eis_result(channel=ch))
        assert len(store.query_measurements(channel=2)) == 1

    def test_query_by_since(self, store_with_run) -> None:
        store, run_id = store_with_run
        store.record_measurement(run_id, _make_eis_result())
        rows = store.query_measurements(since="2020-01-01T00:00:00")
        assert len(rows) == 1
        rows_future = store.query_measurements(since="2099-01-01T00:00:00")
        assert len(rows_future) == 0


# ---------------------------------------------------------------------------
# Multi-stage conditions
# ---------------------------------------------------------------------------


class TestConditions:
    def test_record_single_stage(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        cid = store.record_conditions(
            mid, "measurement", stage_temp_sp_C=50.0, chamber_air_C=49.8,
            rh_sp_pct=40.0, rh_pv_pct=41.2,
        )
        assert isinstance(cid, int) and cid > 0

    def test_record_stage_temp_pv_roundtrips(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_conditions(
            mid, "measurement",
            stage_temp_sp_C=50.0, chamber_air_C=49.8, stage_temp_pv_C=48.5,
            rh_sp_pct=40.0, rh_pv_pct=41.2,
        )
        row = store.query_conditions(measurement_id=mid)[0]
        assert row["stage_temp_sp_C"] == pytest.approx(50.0)
        assert row["chamber_air_C"] == pytest.approx(49.8)
        assert row["stage_temp_pv_C"] == pytest.approx(48.5)
        assert row["rh_sp_pct"] == pytest.approx(40.0)
        assert row["rh_pv_pct"] == pytest.approx(41.2)

    def test_query_conditions_by_stage(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_conditions(mid, "formulation", chamber_air_C=25.0)
        store.record_conditions(mid, "measurement", chamber_air_C=50.0)

        rows = store.query_conditions(stage="measurement")
        assert len(rows) == 1
        assert rows[0]["chamber_air_C"] == pytest.approx(50.0)

    def test_query_conditions_by_run_id(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_conditions(mid, "measurement", chamber_air_C=50.0)
        assert len(store.query_conditions(run_id=run_id)) == 1

    def test_missing_measurement_raises(self, store: DataStore) -> None:
        with pytest.raises(ValueError, match="No measurement"):
            store.record_conditions(99999, "measurement")


class TestConditionsResolvedTemperature:
    """Schema epoch 4: the resolver's answer is written, not re-derived.

    ``record_conditions`` is the only production writer, so these pin the whole
    contract: every row leaves here with a source label, and the number attached
    to it is the one :func:`resolve_temperature_C` chose.
    """

    def _record(self, store_with_run, **env) -> dict:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_conditions(mid, "measurement", **env)
        return store.query_conditions(measurement_id=mid)[0]

    def test_record_conditions_resolves_the_stage_pv_first(
            self, store_with_run) -> None:
        row = self._record(
            store_with_run,
            stage_temp_pv_C=85.0, stage_temp_sp_C=84.0, chamber_air_C=42.8)
        assert row["temperature_C"] == pytest.approx(85.0)
        assert row["temperature_source"] == "stage_pv"

    def test_record_conditions_falls_back_to_the_setpoint(
            self, store_with_run) -> None:
        row = self._record(
            store_with_run, stage_temp_sp_C=65.0, chamber_air_C=36.6)
        assert row["temperature_C"] == pytest.approx(65.0)
        assert row["temperature_source"] == "stage_sp"

    def test_record_conditions_falls_back_to_the_air_probe_and_says_so(
            self, store_with_run) -> None:
        row = self._record(store_with_run, chamber_air_C=29.1)
        assert row["temperature_C"] == pytest.approx(29.1)
        assert row["temperature_source"] == "chamber_air"

    def test_record_conditions_with_no_thermometer_is_null_not_a_number(
            self, store_with_run) -> None:
        """NULL + ``'unavailable'``: one absence, spelled one way.

        The number column is NULL rather than NaN so nothing that only checks
        ``is None`` mistakes it for a reading; the source column is populated
        rather than NULL so *recorded as absent* stays distinguishable from
        *never recorded* — the FORMULATION_THICKNESS_METHODS convention.
        """
        row = self._record(store_with_run, rh_pv_pct=30.0)
        assert row["temperature_C"] is None
        assert row["temperature_source"] == "unavailable"

    def test_record_conditions_rejects_an_impossible_reading_like_the_resolver(
            self, store_with_run) -> None:
        """Validity is the resolver's rule, and it is not re-implemented here."""
        row = self._record(
            store_with_run, stage_temp_pv_C=-300.0, stage_temp_sp_C=40.0)
        assert row["temperature_C"] == pytest.approx(40.0)
        assert row["temperature_source"] == "stage_sp"

    def test_the_source_columns_are_untouched_by_the_derivation(
            self, store_with_run) -> None:
        """Derived, not replacing: the three reads still say what they read."""
        row = self._record(
            store_with_run,
            stage_temp_pv_C=85.0, stage_temp_sp_C=84.0, chamber_air_C=42.8)
        assert row["stage_temp_pv_C"] == pytest.approx(85.0)
        assert row["stage_temp_sp_C"] == pytest.approx(84.0)
        assert row["chamber_air_C"] == pytest.approx(42.8)


# ---------------------------------------------------------------------------
# Conditions-table temperature migrations
#
# The fixtures below build databases in shapes that no longer exist, so their
# CREATE/INSERT statements deliberately spell the OLD column names — that is what
# a legacy database on disk actually says. Every *assertion* reads the live
# schema, under the new names. Keeping that split visible is the point: a legacy
# fixture that quietly acquired the new names would test nothing.
# ---------------------------------------------------------------------------

#: A `conditions` table as it stood before `stage_temp_pv_C` existed (the oldest
#: shape still reachable on disk). Old names, on purpose.
_LEGACY_CONDITIONS_DDL_PRE_STAGE_PV = """\
CREATE TABLE conditions (
    condition_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL,
    run_id         TEXT    NOT NULL,
    stage          TEXT    NOT NULL,
    timestamp      TEXT    NOT NULL,
    temp_sp_C      REAL,
    temp_pv_C      REAL,
    rh_sp_pct      REAL,
    rh_pv_pct      REAL,
    notes          TEXT    NOT NULL DEFAULT ''
)"""

#: The same table once `stage_temp_pv_C` had been added but before the rename.
_LEGACY_CONDITIONS_DDL_PRE_RENAME = _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV.replace(
    "    rh_sp_pct      REAL,",
    "    stage_temp_pv_C REAL,\n    rh_sp_pct      REAL,",
)


def _build_legacy_conditions_db(project: Path, ddl: str, insert: str) -> None:
    """Write a pre-migration database at *project* and close it."""
    import sqlite3

    (project / "db").mkdir(parents=True)
    conn = sqlite3.connect(str(project / "db" / "softae.db"))
    conn.execute(ddl)
    conn.execute(insert)
    conn.commit()
    conn.close()


def _conditions_columns(store: DataStore) -> set[str]:
    return {
        r[1]
        for r in store._conn.execute("PRAGMA table_info(conditions)").fetchall()
    }


def _conditions_indexes(store: DataStore) -> set[str]:
    return {
        r[0]
        for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'conditions'"
        ).fetchall()
    }


class TestStageTempMigration:
    def test_legacy_db_gains_stage_temp_pv_column(self, tmp_path: Path) -> None:
        """A pre-existing conditions table without stage_temp_pv_C is migrated."""
        project = tmp_path / "legacy_project"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp, temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z', 42.0)",
        )

        # Opening the store must add the missing column without losing old rows.
        with DataStore(project) as store:
            assert "stage_temp_pv_C" in _conditions_columns(store)
            old = store.query_conditions()
            assert len(old) == 1
            # Legacy INPUT wrote `temp_pv_C`; the live schema calls it what it is.
            assert old[0]["chamber_air_C"] == pytest.approx(42.0)
            assert old[0]["stage_temp_pv_C"] is None


class TestConditionsTempRenameMigration:
    """Schema epoch 3: temperature columns renamed after their instruments.

    The assertion these exist for is *data survival*. A rename that lost its
    values would be a drop-and-add, and would look identical from the schema.
    """

    def test_pre_rename_db_gains_instrument_named_columns_with_data_intact(
            self, tmp_path: Path) -> None:
        project = tmp_path / "pre_rename"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_RENAME,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
            " temp_sp_C, temp_pv_C, stage_temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z',"
            " 85.0, 42.0, 84.6)",
        )

        with DataStore(project) as store:
            cols = _conditions_columns(store)
            assert {"chamber_air_C", "stage_temp_sp_C", "stage_temp_pv_C"} <= cols
            assert "temp_pv_C" not in cols
            assert "temp_sp_C" not in cols

            rows = store.query_conditions()
            assert len(rows) == 1
            assert rows[0]["chamber_air_C"] == pytest.approx(42.0)
            assert rows[0]["stage_temp_sp_C"] == pytest.approx(85.0)
            assert rows[0]["stage_temp_pv_C"] == pytest.approx(84.6)

    def test_db_predating_stage_temp_pv_is_added_to_then_renamed(
            self, tmp_path: Path) -> None:
        """Both conditions migrations run, in order, on the oldest shape."""
        project = tmp_path / "oldest"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
            " temp_sp_C, temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z', 30.0, 42.0)",
        )

        with DataStore(project) as store:
            cols = _conditions_columns(store)
            assert {"chamber_air_C", "stage_temp_sp_C", "stage_temp_pv_C"} <= cols
            assert not {"temp_pv_C", "temp_sp_C"} & cols
            row = store.query_conditions()[0]
            assert row["chamber_air_C"] == pytest.approx(42.0)
            assert row["stage_temp_sp_C"] == pytest.approx(30.0)
            assert row["stage_temp_pv_C"] is None

    def test_reopening_a_migrated_db_is_a_no_op(self, tmp_path: Path) -> None:
        project = tmp_path / "twice"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_RENAME,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
            " temp_sp_C, temp_pv_C, stage_temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z',"
            " 85.0, 42.0, 84.6)",
        )

        with DataStore(project) as store:
            first = store.query_conditions()
        with DataStore(project) as store:
            assert store.query_conditions() == first
            assert "temp_pv_C" not in _conditions_columns(store)

    def test_a_fresh_db_never_carries_the_old_names(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "fresh") as store:
            cols = _conditions_columns(store)
            assert {"chamber_air_C", "stage_temp_sp_C", "stage_temp_pv_C"} <= cols
            assert not {"temp_pv_C", "temp_sp_C"} & cols

    def test_the_temperature_index_follows_the_column_it_indexes(
            self, tmp_path: Path) -> None:
        """The index named for the air probe is retired, not silently repointed.

        SQLite auto-repoints an index across ``RENAME COLUMN``, which is exactly
        the failure to guard: it would leave `idx_conditions_temp_pv` alive,
        indexing `chamber_air_C`, under a name naming neither.
        """
        project = tmp_path / "indexed"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_RENAME,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp, temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z', 42.0)",
        )
        import sqlite3

        conn = sqlite3.connect(str(project / "db" / "softae.db"))
        conn.execute("CREATE INDEX idx_conditions_temp_pv ON conditions(temp_pv_C)")
        conn.commit()
        conn.close()

        with DataStore(project) as store:
            indexes = _conditions_indexes(store)
            assert "idx_conditions_temp_pv" not in indexes
            assert "idx_conditions_stage_temp_pv" in indexes

    def test_the_epoch_is_recorded_in_the_ledger(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "epoch") as store:
            row = store._conn.execute(
                "SELECT kind, note FROM schema_version WHERE version = 3"
            ).fetchone()
        assert row is not None
        # 'schema', not 'data-epoch': the labels moved and the numbers did not.
        assert row[0] == "schema"
        assert "chamber_air_C" in row[1]


class TestConditionsResolvedTemperatureMigration:
    """Schema epoch 4: derived temperature columns, and the first backfill.

    These stack on the *oldest* reachable shape on purpose. The migration reads
    all three source columns by their settled names, so it can only work if
    ``_migrate_conditions_stage_temp`` (adds the stage PV) and
    ``_migrate_conditions_temp_names`` (renames the other two) have already run
    — ordering the assertions cannot see directly, but a wrong order produces a
    row resolved from columns that did not exist yet.
    """

    #: One legacy row per branch of the resolver's precedence, in the oldest
    #: shape — which has no stage-PV column at all, so 'stage_pv' is unreachable
    #: here and is covered from the pre-rename shape below.
    _OLDEST_ROWS = (
        "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
        " temp_sp_C, temp_pv_C) VALUES"
        " (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z', 65.0, 36.6),"
        " (2, 'old_run', 'measurement', '2026-01-01T00:01:00Z', NULL, 29.1),"
        " (3, 'old_run', 'measurement', '2026-01-01T00:02:00Z', NULL, NULL)"
    )

    def _resolved(self, store: DataStore) -> dict[int, tuple]:
        return {
            r[0]: (r[1], r[2])
            for r in store._conn.execute(
                "SELECT measurement_id, temperature_C, temperature_source "
                "FROM conditions ORDER BY measurement_id"
            ).fetchall()
        }

    def test_legacy_db_gains_both_derived_columns(self, tmp_path: Path) -> None:
        project = tmp_path / "derived"
        _build_legacy_conditions_db(
            project, _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV, self._OLDEST_ROWS)

        with DataStore(project) as store:
            assert {"temperature_C", "temperature_source"} <= _conditions_columns(store)

    def test_every_historical_row_is_backfilled_through_the_resolver(
            self, tmp_path: Path) -> None:
        """The first backfilling migration in this codebase — so, spot-checked.

        A derived column left NULL for history would drop every pre-epoch row
        out of temperature filtering, which is why this migration diverges from
        the NULL-for-historical choice made for ``sample_uuid`` and ``outcome``:
        those are facts the past failed to record, this is a function of the
        row's own columns.
        """
        project = tmp_path / "backfill"
        _build_legacy_conditions_db(
            project, _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV, self._OLDEST_ROWS)

        with DataStore(project) as store:
            rows = self._resolved(store)

        assert rows[1][0] == pytest.approx(65.0)
        assert rows[1][1] == "stage_sp"
        assert rows[2][0] == pytest.approx(29.1)
        assert rows[2][1] == "chamber_air"
        # Nothing to resolve is a *result*: NULL number, populated label.
        assert rows[3] == (None, "unavailable")

    def test_backfill_prefers_the_stage_pv_where_the_row_has_one(
            self, tmp_path: Path) -> None:
        project = tmp_path / "backfill_pv"
        _build_legacy_conditions_db(
            project,
            _LEGACY_CONDITIONS_DDL_PRE_RENAME,
            "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
            " temp_sp_C, temp_pv_C, stage_temp_pv_C)"
            " VALUES (1, 'old_run', 'measurement', '2026-01-01T00:00:00Z',"
            " 85.0, 42.8, 84.6)",
        )

        with DataStore(project) as store:
            assert self._resolved(store)[1] == (pytest.approx(84.6), "stage_pv")

    def test_reopening_a_backfilled_db_changes_nothing(
            self, tmp_path: Path) -> None:
        project = tmp_path / "twice_derived"
        _build_legacy_conditions_db(
            project, _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV, self._OLDEST_ROWS)

        with DataStore(project) as store:
            first = store.query_conditions()
        with DataStore(project) as store:
            assert store.query_conditions() == first

    def test_a_row_left_unresolved_by_a_stale_writer_is_repaired_on_open(
            self, tmp_path: Path) -> None:
        """The backfill targets ``temperature_source IS NULL``, not 'first open'.

        That set is every row after the ALTER, nothing on a settled database,
        and exactly the right rows after an open interrupted mid-backfill or a
        raw INSERT from a binary that predates the columns.
        """
        project = tmp_path / "stale_writer"
        with DataStore(project) as store:
            run_id = store.start_run("legacy_writer")
            mid = store.record_measurement(run_id, _make_eis_result())
            store._conn.execute(
                "INSERT INTO conditions (measurement_id, run_id, stage, timestamp,"
                " stage_temp_pv_C) VALUES (?, ?, 'measurement', ?, 77.0)",
                (mid, run_id, "2026-01-01T00:00:00Z"),
            )
            store._conn.commit()
            assert store.query_conditions(measurement_id=mid)[0][
                "temperature_source"] is None

        with DataStore(project) as store:
            row = store.query_conditions(measurement_id=mid)[0]
            assert row["temperature_C"] == pytest.approx(77.0)
            assert row["temperature_source"] == "stage_pv"

    def test_a_fresh_db_carries_the_derived_columns_and_the_index(
            self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "fresh_derived") as store:
            assert {"temperature_C", "temperature_source"} <= _conditions_columns(store)
            assert "idx_conditions_temperature_C" in _conditions_indexes(store)

    def test_the_derived_index_is_created_on_a_legacy_db_too(
            self, tmp_path: Path) -> None:
        """The filter's new target is indexed on every path, fresh or migrated.

        The DDL cannot declare it — on a legacy database the CREATE TABLE is a
        no-op and the column does not exist yet — so the migration owns it, the
        same division ``idx_conditions_stage_temp_pv`` already follows.
        """
        project = tmp_path / "legacy_index"
        _build_legacy_conditions_db(
            project, _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV, self._OLDEST_ROWS)

        with DataStore(project) as store:
            indexes = _conditions_indexes(store)
            assert "idx_conditions_temperature_C" in indexes
            # The stage-PV index survives: ad-hoc and analysis queries still
            # select on the raw column directly.
            assert "idx_conditions_stage_temp_pv" in indexes

    def test_the_epoch_is_recorded_as_schema_not_data_epoch(
            self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "epoch4") as store:
            row = store._conn.execute(
                "SELECT kind, note FROM schema_version WHERE version = 4"
            ).fetchone()
        assert row is not None
        # 'data-epoch-grade' in the resolver's note measures significance; the
        # kind column asks only whether stored numbers changed meaning.
        assert row[0] == "schema"
        assert "temperature_source" in row[1]

    def test_the_epoch_four_note_names_the_query_for_the_per_database_distribution(
            self, tmp_path: Path) -> None:
        """A count of rows per source is a fact about one FILE, not about the epoch.

        The literal ask was to bake the distribution into the note. It cannot be:
        ``SCHEMA_EPOCHS`` is a per-code constant seeded ``INSERT OR IGNORE``, so an
        amended note reaches only databases that have never been opened — never the
        ones that actually hold the backfilled rows — and one database's numbers in
        source would be false for every other. So the note names the query instead,
        and the migration logs the counts (below).
        """
        with DataStore(tmp_path / "epoch4_query") as store:
            note, = store._conn.execute(
                "SELECT note FROM schema_version WHERE version = 4").fetchone()
        assert "GROUP BY temperature_source" in note
        # The pre-existing pin survives the amendment, twice over.
        assert "temperature_source" in note

    def test_the_backfill_log_line_carries_a_count_per_temperature_source(
            self, tmp_path: Path) -> None:
        # Per-database by construction, and it lands in the run log beside the
        # migration that produced it. `temperature_source IS NULL` is the target,
        # so it fires once per database and counts exactly the rows it wrote.
        import structlog

        project = tmp_path / "backfill_log"
        _build_legacy_conditions_db(
            project, _LEGACY_CONDITIONS_DDL_PRE_STAGE_PV, self._OLDEST_ROWS)

        with structlog.testing.capture_logs() as logs:
            DataStore(project).close()

        events = [e for e in logs
                  if e.get("event") == "conditions_temperature_backfilled"]
        assert len(events) == 1
        assert events[0]["rows"] == 3
        assert events[0]["sources"] == {"stage_sp": 1, "chamber_air": 1,
                                        "unavailable": 1}

    def test_the_conditions_ddl_states_that_the_stored_pair_is_a_record_not_a_view(
            self) -> None:
        # The columns' epistemic status, pinned so it cannot be deleted silently:
        # editing TEMPERATURE_SOURCES changes what future writes conclude and moves
        # no stored row, which is exactly what a reader of the column needs to know
        # and what the DDL did not say before.
        from softae.core.data_store import _DDL

        assert "RECORD, NOT A VIEW" in _DDL
        assert "resolve_temperature_C" in _DDL


# ---------------------------------------------------------------------------
# Temperature-range query via conditions join
# ---------------------------------------------------------------------------


class TestTempRangeQuery:
    """The filter names one column — ``conditions.temperature_C`` (epoch 4).

    These scenarios are unchanged from when the filter restated the precedence
    as a COALESCE; what changed is *where* the precedence is applied. It now
    happens at ``record_conditions`` time, so these read as write-time-resolution
    tests observed through the query — which is the point: a filter that can only
    return the right row if the write resolved correctly is the stronger test.
    """

    def test_temp_range_filter(self, store_with_run) -> None:
        store, run_id = store_with_run
        m1 = store.record_measurement(run_id, _make_eis_result(channel=1))
        m2 = store.record_measurement(run_id, _make_eis_result(channel=2))
        store.record_conditions(m1, "measurement", chamber_air_C=30.0)
        store.record_conditions(m2, "measurement", chamber_air_C=50.0)

        rows = store.query_measurements(temp_range=(48.0, 52.0))
        assert len(rows) == 1
        assert rows[0]["channel"] == 2

    def test_temp_range_prefers_the_stage_over_the_air_probe(
            self, store_with_run) -> None:
        """The filter answers "measurements at temperature X" about the sample.

        Both rows below carry a chamber-air read near 43 °C — filtering on the
        air probe (as this query did until 2026-08-11) would select whichever row
        the *air* happened to match and confidently return the wrong one.
        """
        store, run_id = store_with_run
        m1 = store.record_measurement(run_id, _make_eis_result(channel=1))
        m2 = store.record_measurement(run_id, _make_eis_result(channel=2))
        store.record_conditions(
            m1, "measurement", stage_temp_pv_C=45.0, chamber_air_C=42.8)
        store.record_conditions(
            m2, "measurement", stage_temp_pv_C=85.0, chamber_air_C=42.9)

        rows = store.query_measurements(temp_range=(83.0, 87.0))
        assert [r["channel"] for r in rows] == [2]
        assert store.query_measurements(temp_range=(41.0, 44.0)) == []

    def test_temp_range_falls_back_to_the_setpoint_then_the_air(
            self, store_with_run) -> None:
        """The stored column carries the resolver's fallbacks: SP, then air."""
        store, run_id = store_with_run
        m1 = store.record_measurement(run_id, _make_eis_result(channel=1))
        m2 = store.record_measurement(run_id, _make_eis_result(channel=2))
        store.record_conditions(
            m1, "measurement", stage_temp_sp_C=65.0, chamber_air_C=36.6)
        store.record_conditions(m2, "measurement", chamber_air_C=29.1)

        assert [r["channel"] for r in
                store.query_measurements(temp_range=(64.0, 66.0))] == [1]
        assert [r["channel"] for r in
                store.query_measurements(temp_range=(28.0, 30.0))] == [2]

    def test_temp_range_excludes_rows_no_thermometer_spoke_for(
            self, store_with_run) -> None:
        """NULL is not 0 °C, and must not be swept into a range containing it."""
        store, run_id = store_with_run
        m1 = store.record_measurement(run_id, _make_eis_result(channel=1))
        m2 = store.record_measurement(run_id, _make_eis_result(channel=2))
        store.record_conditions(m1, "measurement", rh_pv_pct=30.0)
        store.record_conditions(m2, "measurement", stage_temp_pv_C=25.0)

        assert [r["channel"] for r in
                store.query_measurements(temp_range=(-50.0, 50.0))] == [2]

    def test_temp_range_custom_stage(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_conditions(mid, "formulation", chamber_air_C=25.0)
        store.record_conditions(mid, "measurement", chamber_air_C=50.0)

        # Filter by formulation-stage temperature
        rows = store.query_measurements(
            temp_range=(20.0, 30.0), condition_stage="formulation"
        )
        assert len(rows) == 1

        # Same range but measurement stage — should be empty
        rows = store.query_measurements(
            temp_range=(20.0, 30.0), condition_stage="measurement"
        )
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Fit results
# ---------------------------------------------------------------------------


class TestFitResults:
    def test_record_fit(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _FakeFitResult())
        assert isinstance(fid, int) and fid > 0

    def test_record_fit_with_geometry(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(
            mid, _FakeFitResult(R1=500.0),
            L_cm=0.2, t_cm=0.175, w_cm=0.2,
        )
        rows = store.query_fits(measurement_id=mid)
        assert len(rows) == 1
        sigma = rows[0]["sigma_S_per_cm"]
        expected = 0.2 / (500.0 * 0.175 * 0.2)
        assert sigma == pytest.approx(expected)

    def test_record_fit_failure(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fit = _FakeFitResult(success=False, error_msg="did not converge")
        store.record_fit(mid, fit)
        rows = store.query_fits(run_id=run_id)
        assert rows[0]["success"] == 0
        assert "converge" in rows[0]["error_msg"]

    def test_query_fits_by_model(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult(model_name="simpleSalt"))
        store.record_fit(mid, _FakeFitResult(model_name="flexSalt"))
        assert len(store.query_fits(model_name="flexSalt")) == 1


# ---------------------------------------------------------------------------
# fit_results.engine — the column that separates two incomparable populations
# ---------------------------------------------------------------------------


class TestTheEngineColumnIsAClaimAndNotADefault:
    """The same spectrum fitted by the two engines differs in R₁ by up to 55×.

    Measured on ch32_002: 3.758e7 Ω legacy against 6.802e5 Ω gated, and σ = K/R
    inverts it. ``fit_results.engine`` is the only per-row thing that separates those
    populations, so a row whose label is wrong is worse than a row with no label —
    only the second can be excluded from an analysis.

    ``record_fit`` learns the engine from the ``SpectrumReport`` it is handed and from
    nothing else. Three of its four call sites hand it none.
    """

    def test_a_fit_recorded_without_a_report_declares_no_engine(
            self, store_with_run) -> None:
        from softae.core.data_store import FIT_ENGINE_UNKNOWN

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult())
        # The LITERAL is what is pinned. Asserting the column against the constant
        # that wrote it is SUBAGENT_RULES §3's test that cannot fail: rebind the
        # constant to 'legacy' and both sides move together, green throughout.
        assert store.query_fits(measurement_id=mid)[0]["engine"] == "unknown"
        assert FIT_ENGINE_UNKNOWN == "unknown"

    def test_the_undeclared_value_is_not_legacy(self, store_with_run) -> None:
        """Named separately because 'not legacy' is the whole defect.

        ``'legacy'`` was correct only while ``[eis] engine`` was ``legacy``; the
        operator is setting it to ``gated``, at which point the literal states the
        opposite of what ran. A test that only pinned the new spelling would still
        pass if someone restored the old one under a renamed constant.
        """
        from softae.core.data_store import FIT_ENGINE_UNKNOWN

        assert FIT_ENGINE_UNKNOWN not in ("legacy", "gated")
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult())
        assert store.query_fits(measurement_id=mid)[0]["engine"] != "legacy"

    @pytest.mark.parametrize("declared", ["legacy", "gated"])
    def test_a_declared_engine_is_recorded_verbatim(
            self, store_with_run, declared: str) -> None:
        """Both directions, so the sentinel cannot be a blanket that swallows reports.

        SUBAGENT_RULES §3: the conservative answer here would be to stamp 'unknown' on
        everything, which would pass any test that only checked the undeclared path
        while destroying the labels the gated cutover depends on.
        """
        from softae.analysis.eis.report import SigmaReport, SpectrumReport

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult(),
                         report=SpectrumReport(engine=declared, sigma=SigmaReport()))
        assert store.query_fits(measurement_id=mid)[0]["engine"] == declared

    def test_a_report_that_declares_no_engine_is_also_unknown(
            self, store_with_run) -> None:
        """The ``getattr`` fallback, which a real SpectrumReport can never reach.

        ``SpectrumReport.engine`` has no default, so only a duck-typed stand-in gets
        here — and such an object has declared no engine either.
        """
        class _ReportWithoutAnEngine:
            sigma = None
            cell = None
            quality = None
            mask = None
            gate_log = ()

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult(), report=_ReportWithoutAnEngine())
        assert store.query_fits(measurement_id=mid)[0]["engine"] == "unknown"

    def test_the_column_is_never_null_so_no_reader_needs_a_null_branch(
            self, store_with_run) -> None:
        """``'unknown'`` and not NULL, and this is the reason it is a value.

        NULL would need the column to stop being NOT NULL — a twelve-step table
        rebuild in SQLite, on the table holding every fit the rig has recorded — and
        would collide with the meaning NULL already carries three columns to the right
        on ``arc_state``: *never annotated*. Every ``fit_results`` row WAS annotated.
        """
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult())
        assert store._conn.execute(
            "SELECT COUNT(*) FROM fit_results WHERE engine IS NULL"
        ).fetchone()[0] == 0

    def test_nothing_backfills_or_rewrites_an_already_labelled_row(self) -> None:
        """Rows stored as ``'legacy'`` were legacy; a sweep would destroy a true label.

        Asserted over the source rather than over a database because the claim is that
        no such code exists — a fixture can only show that one path did not run it.
        """
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src" / "softae" / "core"
                  / "data_store.py").read_text(encoding="utf-8")
        assert "UPDATE fit_results" not in source
        migrations = [n.name for n in ast.walk(ast.parse(source))
                      if isinstance(n, ast.FunctionDef) and n.name.startswith("_migrate")]
        assert not any("engine" in name for name in migrations)

    def test_the_setting_is_never_read_at_write_time(self) -> None:
        """"What the config said when the row was INSERTed" is a different claim.

        ``analyze_spectrum`` takes an ``engine=`` override AND a whole ``settings=``
        override (``tools/shadow_rehearse.py`` uses the second to run gated against a
        legacy config), and ``record_fit`` may be called for a fit produced minutes
        earlier by a path that never consulted the config. Resolving ``[eis] engine``
        here would be a fact about the process wearing a fact about the fit — and,
        unlike the sentinel, undetectable when wrong.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "src" / "softae" / "core"
                  / "data_store.py").read_text(encoding="utf-8")
        assert "eis_settings" not in source

    def test_the_estimator_half_survives_an_undeclared_engine(
            self, store_with_run) -> None:
        """``_engine_label`` composes, so the evidenced half is still recorded.

        The estimator is read off the FIT, so it is known even when the engine is not.
        Pre-gate diversion only happens on the gated path, so the old default spelled
        such a row bare ``'two_point'`` — i.e. *legacy* — which is precisely the
        unlabelled-biased-R₁ failure epoch 5 exists to prevent.
        """
        from softae.analysis.eis.engine import TWO_POINT

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _FakeFitResult(estimator=TWO_POINT))
        assert store.query_fits(
            measurement_id=mid)[0]["engine"] == "unknown_two_point"


# ---------------------------------------------------------------------------
# Formulations
# ---------------------------------------------------------------------------


class TestFormulations:
    def test_record_formulation(self, store_with_run) -> None:
        store, run_id = store_with_run
        fid = store.record_formulation(
            run_id, channel=1,
            pump0_uL=5.0, pump1_uL=3.0, total_uL=8.0,
            solution_name="EMIm-TFSI_30pct",
        )
        assert isinstance(fid, int) and fid > 0

    def test_query_formulations_by_channel(self, store_with_run) -> None:
        store, run_id = store_with_run
        store.record_formulation(run_id, channel=1, pump0_uL=5.0, total_uL=5.0)
        store.record_formulation(run_id, channel=2, pump0_uL=10.0, total_uL=10.0)
        rows = store.query_formulations(channel=1)
        assert len(rows) == 1
        assert rows[0]["pump0_uL"] == pytest.approx(5.0)

    def test_record_and_query_third_pump(self, store_with_run) -> None:
        store, run_id = store_with_run
        store.record_formulation(
            run_id, channel=3,
            pump0_uL=5.0, pump1_uL=3.0, pump2_uL=2.0, total_uL=10.0,
        )
        rows = store.query_formulations(channel=3)
        assert len(rows) == 1
        assert rows[0]["pump2_uL"] == pytest.approx(2.0)
        assert rows[0]["total_uL"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# End-to-end scenario
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_experiment_lifecycle(self, tmp_path: Path) -> None:
        """Simulate a complete HT experiment with formulation → EIS → fit."""
        with DataStore(tmp_path / "experiment") as ds:
            # Start run
            run_id = ds.start_run(
                "ht_experiment",
                '{"mock": true}',
                mode="full",
                pcb_name="SoftAE_IDE_EIS",
                campaign="q1_emim",
                quality="explore",
            )

            for ch in range(1, 5):
                # Record formulation
                ds.record_formulation(
                    run_id, channel=ch,
                    pump0_uL=5.0, pump1_uL=3.0, total_uL=8.0,
                    solution_name=f"sol_{ch}",
                )

                # Record measurement
                eis = _make_eis_result(channel=ch)
                mid = ds.record_measurement(run_id, eis, electrode_x_mm=float(ch * 10))

                # Record multi-stage conditions
                ds.record_conditions(mid, "formulation", chamber_air_C=25.0, rh_pv_pct=50.0)
                ds.record_conditions(mid, "processing", chamber_air_C=80.0, rh_pv_pct=30.0)
                ds.record_conditions(mid, "measurement", chamber_air_C=50.0, rh_pv_pct=40.0)

                # Record fit
                fit = _FakeFitResult(R1=500.0 + ch * 100)
                ds.record_fit(mid, fit, L_cm=0.2, t_cm=0.175, w_cm=0.2)

            # Finish run
            ds.finish_run(run_id, "done")

            # Verify
            runs = ds.query_runs(campaign="q1_emim")
            assert len(runs) == 1
            assert runs[0]["status"] == "done"

            measurements = ds.query_measurements(run_id=run_id)
            assert len(measurements) == 4

            conditions = ds.query_conditions(run_id=run_id)
            assert len(conditions) == 12  # 4 channels × 3 stages

            fits = ds.query_fits(run_id=run_id)
            assert len(fits) == 4

            formulations = ds.query_formulations(run_id=run_id)
            assert len(formulations) == 4

            # Temperature range query (measurement stage)
            hot_meas = ds.query_measurements(temp_range=(48.0, 52.0))
            assert len(hot_meas) == 4  # all channels measured at 50 °C


# ---------------------------------------------------------------------------
# E6 — overhaul §6's mandatory-at-ingest metadata
# ---------------------------------------------------------------------------


class TestRequiredMetadataE6:
    """§6's remaining fields, and the defaults that keep them honest.

    The test running through all of these: **a default must not look like an
    answer.** Every one of these columns exists to record something unrecoverable
    after the fact, so a default that resembles a real value destroys exactly the
    distinction the column was added to make.
    """

    def _row(self, store, mid):
        return store._conn.execute(
            "SELECT thermal_history, sweep_order, re_connection, re_contact_verified "
            "FROM measurements WHERE measurement_id = ?", (mid,)
        ).fetchone()

    def test_the_fields_round_trip(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(
            run_id, _make_eis_result(channel=4),
            thermal_history="anneal 80C/2h, cool 1h, equilibrate 40%RH 12h",
            sweep_order=3, re_connection="bridged_by_sample",
            re_contact_verified=True)
        hist, order, re_conn, verified = self._row(store, mid)
        assert hist.startswith("anneal 80C/2h")
        assert order == 3
        assert re_conn == "bridged_by_sample"
        assert verified == 1

    def test_an_unrecorded_sweep_position_is_null_not_zero(self, store_with_run) -> None:
        # Zero would make every legacy row claim to be the FIRST measurement of its
        # sweep -- and first-versus-last is precisely the drift signal this supports.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        assert self._row(store, mid)[1] is None

    def test_unsupplied_re_state_defaults_to_unverified_not_to_a_working_loop(
            self, store_with_run) -> None:
        # F13 is the dominant source of quadrant violations to date. Defaulting to a
        # closed loop would suppress the one diagnostic that catches it.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        _, _, re_conn, verified = self._row(store, mid)
        assert re_conn == "unverified"
        assert verified == 0

    def test_existing_callers_are_unchanged_because_every_field_is_keyword_defaulted(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        assert store.record_measurement(run_id, _make_eis_result()) > 0

    def test_electrode_configuration_is_not_duplicated_as_a_second_column(
            self, store_with_run) -> None:
        # §6 asks for `electrode_configuration`; E1.7's `electrode_mode` already IS
        # that field. A synonym column would be two records of one fact, free to
        # disagree -- and the disagreement would be silent.
        store, _ = store_with_run
        cols = {r[1] for r in
                store._conn.execute("PRAGMA table_info(measurements)").fetchall()}
        assert "electrode_mode" in cols
        assert "electrode_configuration" not in cols

    def test_cycles_per_point_is_deliberately_absent(self, store_with_run) -> None:
        # `eis_run_mscrbuild` exposes no averaging parameter and `meas_loop_eis`
        # passes none, so the column could only ever hold a number nobody commanded.
        # Recording what was commanded -- nothing -- is the honest option.
        store, _ = store_with_run
        cols = {r[1] for r in
                store._conn.execute("PRAGMA table_info(measurements)").fetchall()}
        assert "cycles_per_point" not in cols


class TestRunIdCollision:
    """`run_id` is a one-second timestamp — ample for a rig run, not for a batch.

    Found importing seven commissioning blanks in a loop: the first succeeded and the
    rest raised `UNIQUE constraint failed: experiments.run_id` *after* each had already
    written its EIS file to disk, leaving files with no row pointing at them.
    """

    def test_runs_started_in_the_same_second_get_distinct_ids(self, store) -> None:
        ids = [store.start_run("import_blank_open") for _ in range(7)]
        assert len(set(ids)) == 7

    def test_the_first_id_keeps_its_exact_historical_spelling(self, store) -> None:
        # Suffixing rather than widening the stamp: every run_id already on disk keeps
        # its name, and the format stays lexically sortable by time.
        first = store.start_run("ht_experiment")
        second = store.start_run("ht_experiment")
        assert not first.endswith("_2")
        assert second == f"{first}_2"

    def test_each_deduplicated_run_is_a_real_row(self, store) -> None:
        ids = [store.start_run("import_blank_open") for _ in range(3)]
        for rid in ids:
            row = store._conn.execute(
                "SELECT run_id FROM experiments WHERE run_id = ?", (rid,)).fetchone()
            assert row is not None


class TestFormulationAreaProvenance:
    """`predicted_thickness_um` stored a quotient with no denominator (P.7).

    The deposit area on the 4-stripe board moved from 4.0 mm² (the inter-electrode
    rectangle) to 18.704 mm² (the well) when P7.2 made the well authoritative, so
    rows written either side of that differ by a factor of 4.676 **in the same
    column, with the same units** — and a formulation row does not record which
    board it was cast on. Storing the area beside the thickness is the only thing
    that keeps rows from the two epochs comparable.
    """

    def _formulation_columns(self, store: DataStore) -> set[str]:
        return {r[1] for r in store._conn.execute(
            "PRAGMA table_info(formulations)").fetchall()}

    def _row(self, store: DataStore, fid: int) -> tuple:
        # tuple(): the store sets `row_factory = sqlite3.Row`, which does not
        # compare equal to a plain tuple.
        return tuple(store._conn.execute(
            "SELECT deposit_area_mm2, thickness_method FROM formulations "
            "WHERE formulation_id = ?", (fid,)).fetchone())

    def test_a_fresh_database_has_the_columns_from_the_ddl_not_only_from_the_migration(
            self, store: DataStore) -> None:
        # `_DDL` and the migration are two descriptions of one schema. Covering only
        # the migration lets `_DDL` drift, and then a fresh install and an upgraded
        # one disagree about what a formulation row contains -- silently, until a
        # query written against one is run against the other.
        assert {"deposit_area_mm2", "thickness_method"} <= self._formulation_columns(
            store)

    def test_a_recorded_area_is_the_one_the_thickness_was_divided_by(
            self, store_with_run) -> None:
        # The pair is only worth storing if it round-trips as a pair: thickness times
        # area must give back the cast volume that produced it. Recording an area
        # that did not divide this row's thickness would be worse than recording
        # none, because it reads as provenance.
        store, run_id = store_with_run
        fid = store.record_formulation(
            run_id, 3, total_uL=10.0, predicted_thickness_um=534.6,
            deposit_area_mm2=18.703786, thickness_method="predicted")
        area, method = self._row(store, fid)
        assert area == pytest.approx(18.703786)
        assert method == "predicted"
        thickness_um, = store._conn.execute(
            "SELECT predicted_thickness_um FROM formulations WHERE formulation_id = ?",
            (fid,)).fetchone()
        assert thickness_um * area / 1000.0 == pytest.approx(10.0, rel=1e-3)

    def test_a_legacy_row_keeps_null_so_never_recorded_stays_distinct_from_recorded_absent(
            self, store_with_run) -> None:
        # Three states, and the third is the whole justification for a column whose
        # value is otherwise near-constant. A writer with no twin (the HT tab) leaves
        # NULL = *never recorded*; the campaign path writes 'unavailable' = *asked,
        # and the answer was no*. Collapsing them loses exactly the distinction P.7
        # exists to create.
        store, run_id = store_with_run
        untouched = store.record_formulation(run_id, 1, total_uL=10.0)
        declined = store.record_formulation(
            run_id, 2, total_uL=10.0, thickness_method="unavailable")
        assert self._row(store, untouched) == (None, None)
        assert self._row(store, declined) == (None, "unavailable")

    def test_the_method_vocabulary_is_the_one_resolve_thickness_cm_already_uses(
            self) -> None:
        # `formulations.thickness_method` says which tier a row *offers*;
        # `fit_results.thickness_method` says which tier a fit *used*. Same axis, so
        # a second spelling of the same ladder would fork the vocabulary rather than
        # avoid a collision. Driving the resolver is what proves they agree -- a
        # tuple compared against a copy of itself would not.
        from softae.analysis.eis.geometry import resolve_thickness_cm
        from softae.core.data_store import FORMULATION_THICKNESS_METHODS

        reachable = {
            resolve_thickness_cm(profilometry_um=1.0)[1],
            resolve_thickness_cm(target_um=1.0)[1],
            resolve_thickness_cm(predicted_um=1.0)[1],
            resolve_thickness_cm(dispensed_um=1.0)[1],
            resolve_thickness_cm()[1],
        }
        assert reachable == {"profilometry", "target", "predicted", "dispensed",
                             "unavailable"}
        assert reachable <= set(FORMULATION_THICKNESS_METHODS)

    def test_the_record_reader_returns_the_thickness_with_the_area_it_was_divided_by(
            self, store_with_run) -> None:
        # `predicted_thickness_um` hands back a quotient and drops its denominator,
        # which is what made the two area epochs indistinguishable in the first
        # place. Reading them together is the whole point of P.7's write side: a
        # consumer can only judge whether a thickness means anything if the basis
        # arrives with it, in one read, from one row.
        store, run_id = store_with_run
        store.record_formulation(
            run_id, 3, total_uL=10.0, predicted_thickness_um=534.6,
            deposit_area_mm2=18.703786, thickness_method="predicted")

        record = store.predicted_thickness_record(run_id, 3)
        assert record is not None
        assert record.um == pytest.approx(534.6)
        assert record.area_mm2 == pytest.approx(18.703786)
        assert record.method == "predicted"

    def test_a_row_with_no_recorded_area_still_yields_a_record_with_a_null_area(
            self, store_with_run) -> None:
        # The reader must not collapse "no area recorded" into "no thickness
        # recorded". They are different facts and the caller acts on them
        # differently -- one is a row nobody could describe, the other is a row that
        # exists and whose basis is unknown. Returning a bare `None` here would hide
        # which occurred, and the withheld-thickness warning could not name a reason.
        store, run_id = store_with_run
        store.record_formulation(run_id, 3, total_uL=10.0,
                                 predicted_thickness_um=534.6)

        record = store.predicted_thickness_record(run_id, 3)
        assert record is not None
        assert record.um == pytest.approx(534.6)
        assert record.area_mm2 is None
        # Never 4.0, and never rescaled to 18.7: we cannot tell whether this row
        # predates the 2026-08-07 correction or ran on a board it never touched, and
        # a guess is indistinguishable from a record once it is in the column.
        assert record.method is None

    def test_the_record_reader_is_none_when_no_thickness_was_recorded_at_all(
            self, store_with_run) -> None:
        # Matching `predicted_thickness_um`'s absence exactly. A record whose `um`
        # were `None` would push the same "is there a number here?" branch onto every
        # caller, and the σ path already treats absent thickness as a single case.
        store, run_id = store_with_run
        store.record_formulation(run_id, 1, total_uL=10.0)
        assert store.predicted_thickness_record(run_id, 1) is None
        assert store.predicted_thickness_record(run_id, 99) is None

    def test_the_record_reader_takes_the_latest_cast_like_the_scalar_reader(
            self, store_with_run) -> None:
        # The two readers must agree on *which row* they describe, or a consumer
        # switching from one to the other silently changes which cast it is talking
        # about. Re-casting a well is ordinary, so this is not a corner case.
        store, run_id = store_with_run
        store.record_formulation(run_id, 3, predicted_thickness_um=1.0,
                                 deposit_area_mm2=4.0)
        store.record_formulation(run_id, 3, predicted_thickness_um=4.0,
                                 deposit_area_mm2=18.7038)

        record = store.predicted_thickness_record(run_id, 3)
        assert record.um == pytest.approx(store.predicted_thickness_um(run_id, 3))
        assert record.area_mm2 == pytest.approx(18.7038)

    def test_the_scalar_reader_is_unchanged_so_its_existing_consumers_do_not_break(
            self, store_with_run) -> None:
        # P.11 adds a reader rather than changing one. `tab_analysis` multiplies the
        # result by 1e-4 and several test stubs define it as returning a float, so a
        # widened return type would break the GUI outright and quietly keep the
        # unguarded path alive in the stubs.
        store, run_id = store_with_run
        store.record_formulation(run_id, 3, predicted_thickness_um=534.6)
        um = store.predicted_thickness_um(run_id, 3)
        assert isinstance(um, float)
        assert um == pytest.approx(534.6)


# ---------------------------------------------------------------------------
# Modality / payload contract on `measurements` (Tier 2 component 3)
# ---------------------------------------------------------------------------


class TestMeasurementPayloadColumns:
    """The storage half of the modality contract, at the DataStore's own surface.

    The router's end-to-end behaviour is pinned by
    ``tests/test_result_router_golden.py``; these cover what the store guarantees
    to *any* caller, including the future non-EIS ones the columns exist for.
    """

    def _row(self, store: DataStore, mid: int) -> dict:
        return dict(store._conn.execute(
            "SELECT modality, payload_path, payload_format, sample_uuid "
            "FROM measurements WHERE measurement_id = ?", (mid,)
        ).fetchone())

    def test_every_new_field_reaches_the_row(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(
            run_id, _make_eis_result(channel=1),
            modality="image", payload_path="runs/r/data/image/f.nc",
            payload_format="netcdf4", sample_uuid="sample-abc",
        )
        assert self._row(store, mid) == {
            "modality": "image",
            "payload_path": "runs/r/data/image/f.nc",
            "payload_format": "netcdf4",
            "sample_uuid": "sample-abc",
        }

    def test_existing_callers_are_unchanged_because_all_four_are_keyword_defaulted(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result(channel=2))
        assert self._row(store, mid) == {
            "modality": "eis", "payload_path": None,
            "payload_format": None, "sample_uuid": None,
        }

    def test_an_empty_payload_path_is_stored_as_null_not_as_a_path(
            self, store_with_run) -> None:
        """'' is a path that has been *recorded*, and joins build happily from it.

        NULL is the only spelling of "no payload was written", so an empty string
        must not become a second one that readers checking `is None` miss.
        """
        store, run_id = store_with_run
        mid = store.record_measurement(
            run_id, _make_eis_result(channel=3),
            payload_path="", payload_format="", sample_uuid="",
        )
        row = self._row(store, mid)
        assert row["payload_path"] is None
        assert row["payload_format"] is None
        assert row["sample_uuid"] is None

    def test_attaching_a_payload_stores_it_relative_to_the_project_dir(
            self, store_with_run) -> None:
        """A project directory must survive being copied to another machine.

        An absolute path recorded here would resolve to the writer's filesystem
        and nowhere else — the same reason `eis_file_path` is relative.
        """
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result(channel=4))
        absolute = store.payload_dir(run_id, "eis") / "m.nc"

        store.set_measurement_payload(mid, absolute)

        row = self._row(store, mid)
        assert row["payload_path"] == str(
            Path("runs") / run_id / "data" / "eis" / "m.nc")
        assert row["payload_format"] == "netcdf4"
        assert (store.project_dir / row["payload_path"]) == absolute

    def test_attaching_nothing_leaves_both_columns_null(self, store_with_run) -> None:
        """The failed-write path: a format with no file would claim one exists."""
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result(channel=5))

        store.set_measurement_payload(mid, None, "netcdf4")

        row = self._row(store, mid)
        assert row["payload_path"] is None
        assert row["payload_format"] is None

    def test_payloads_are_partitioned_by_modality_under_the_run(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        assert store.payload_dir(run_id, "eis") == (
            store.run_dir(run_id) / "data" / "eis")
        # A sibling of the transitional `eis/` .txt tree, never nested inside it —
        # retiring the .txt files must not become a payload migration.
        assert store.payload_dir(run_id, "image") != store.eis_dir(run_id)
        assert store.eis_dir(run_id) not in store.payload_dir(run_id, "eis").parents


# ---------------------------------------------------------------------------
# Arc-closure columns on `fit_results` (T7.7)
#
# The legacy fixture below spells the OLDEST reachable `fit_results` — before the
# E0/E1 gate columns and long before the arc ones — because that is the shape a
# migration actually meets on disk, and because it is the only shape that can prove
# the index trap of §3.2 is avoided.
# ---------------------------------------------------------------------------


#: `fit_results` as it stood before any gate or arc column existed.
_LEGACY_FIT_RESULTS_DDL = """\
CREATE TABLE fit_results (
    fit_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id      INTEGER NOT NULL,
    run_id              TEXT    NOT NULL,
    model_name          TEXT    NOT NULL,
    R0                  REAL,
    R1                  REAL,
    sigma_S_per_cm      REAL,
    electrode_L_cm      REAL,
    electrode_t_cm      REAL,
    electrode_w_cm      REAL,
    success             INTEGER NOT NULL DEFAULT 1,
    error_msg           TEXT    NOT NULL DEFAULT '',
    parameters_json     TEXT    NOT NULL DEFAULT '{}',
    fitted_at           TEXT    NOT NULL
)"""

_LEGACY_FIT_ROW = (
    "INSERT INTO fit_results (measurement_id, run_id, model_name, R0, R1, "
    "fitted_at) VALUES (1, 'old_run', 'simpleSalt', 50.0, 1000.0, "
    "'2026-07-01T00:00:00Z')"
)


def _build_legacy_fit_results_db(project: Path) -> None:
    """Write a pre-T7.7 database carrying one fit row, and close it."""
    import sqlite3

    (project / "db").mkdir(parents=True)
    conn = sqlite3.connect(str(project / "db" / "softae.db"))
    conn.execute(_LEGACY_FIT_RESULTS_DDL)
    conn.execute(_LEGACY_FIT_ROW)
    conn.commit()
    conn.close()


def _fit_results_columns(store: DataStore) -> set[str]:
    return {r[1] for r in store._conn.execute(
        "PRAGMA table_info(fit_results)").fetchall()}


def _fit_results_declarations(store: DataStore) -> dict[str, tuple]:
    """``{name: (type, notnull, default)}`` — the DDL and the ALTERs, compared."""
    return {r[1]: (r[2].upper(), r[3], r[4]) for r in store._conn.execute(
        "PRAGMA table_info(fit_results)").fetchall()}


def _fit_results_indexes(store: DataStore) -> set[str]:
    return {r[0] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' "
        "AND tbl_name = 'fit_results'").fetchall()}


ARC_COLUMNS = {"arc_state", "arc_f_peak_hz", "arc_f_low_hz", "arc_phase_low_deg"}


def _annotated(state: str = "open", f_peak: float = 20.0, f_low: float = 20.0,
               phase: float = -41.5):
    """A fit carrying an `ArcClosure`, as `annotate_arc_closure` leaves one."""
    from softae.analysis.eis.arc import ArcClosure

    fit = _FakeFitResult()
    fit.arc_closure = ArcClosure(state, f_peak, f_low, phase)
    return fit


class TestArcColumns:
    """Four columns instead of a JSON blob, so the verdict is queryable by SQL.

    The annotation used to reach the database only inside ``gate_log_json``, via a
    provenance shim, which meant every consumer had to ``json.loads`` a TEXT column
    to ask what the arc did. These columns made that shim redundant, and it is gone.
    """

    def _row(self, store: DataStore, fid: int) -> dict:
        return dict(store._conn.execute(
            "SELECT arc_state, arc_f_peak_hz, arc_f_low_hz, arc_phase_low_deg "
            "FROM fit_results WHERE fit_id = ?", (fid,)).fetchone())

    # ── Schema ──────────────────────────────────────────────────────────────

    def test_a_fresh_database_has_the_arc_columns_from_the_ddl_not_only_from_the_migration(
            self, store: DataStore) -> None:
        """`_DDL` and the migration are two descriptions of one schema.

        Covering only the migration lets the DDL drift until a fresh install and an
        upgraded one disagree about what a fit row contains.

        **The `_DDL` TEXT assertion is the load-bearing one, and the PRAGMA below
        cannot substitute for it.** `_migrate_fit_gate_columns` runs on EVERY open,
        including a brand-new database, so its ALTER loop supplies any column the
        DDL omitted and `PRAGMA table_info` reads back identically either way —
        green on a `_DDL` that declares none of these. The text is also what a
        future hand actually edits. Do not "simplify" this back to the PRAGMA.
        """
        from softae.core.data_store import _DDL

        for name in sorted(ARC_COLUMNS):
            assert f"\n    {name} " in _DDL, f"{name} missing from _DDL"
        # Kept alongside, not traded away: the text proves the DECLARATION exists,
        # this proves the column reached the opened database.
        assert ARC_COLUMNS <= _fit_results_columns(store)

    def test_a_legacy_database_gains_the_arc_columns_from_the_migration(
            self, tmp_path: Path) -> None:
        project = tmp_path / "legacy_arc"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as store:
            assert ARC_COLUMNS <= _fit_results_columns(store)

    def test_a_fresh_and_a_migrated_database_agree_on_the_fit_results_column_set(
            self, tmp_path: Path, store: DataStore) -> None:
        # Sets, not sequences: `query_fits` does `SELECT *` into a dict and
        # `record_fit` names every column, so nothing depends on ordering — but
        # everything depends on the two paths producing the same columns, with the
        # same declarations. The eighteen E0/E1 columns are in this comparison too:
        # they were migration-only until T7.7 declared them in `_DDL` as well.
        project = tmp_path / "legacy_parity"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as migrated:
            assert _fit_results_columns(migrated) == _fit_results_columns(store)
            assert _fit_results_declarations(migrated) == _fit_results_declarations(
                store)

    def test_reopening_a_store_twice_adds_no_column_and_raises_nothing(
            self, tmp_path: Path) -> None:
        # The DDL/migration pair is idempotent by its PRAGMA guard; a second open
        # must be a no-op rather than a duplicate-column error.
        project = tmp_path / "reopen"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as first:
            once = _fit_results_columns(first)
        with DataStore(project) as second:
            assert _fit_results_columns(second) == once

    def test_the_arc_state_index_exists_on_both_a_fresh_and_a_migrated_database(
            self, tmp_path: Path, store: DataStore) -> None:
        project = tmp_path / "legacy_index_arc"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as migrated:
            assert "idx_fit_results_arc_state" in _fit_results_indexes(migrated)
        # The migration runs on every open, so a fresh database gets it too even
        # though `_DDL` deliberately does not declare it.
        assert "idx_fit_results_arc_state" in _fit_results_indexes(store)

    def test_opening_a_pre_t7_7_database_does_not_fail_on_the_index(
            self, tmp_path: Path) -> None:
        """The trap: `_DDL` runs BEFORE the migrations, so it cannot index this.

        On a legacy database `CREATE TABLE IF NOT EXISTS fit_results` is a no-op and
        the table still lacks `arc_state` at that moment, so a `CREATE INDEX ... ON
        fit_results(arc_state)` in the DDL's index block raises `no such column`
        inside `executescript` and fails the open of every existing project. Opening
        the oldest shape without raising is the whole proof, so the assertion is
        that we get here at all — plus the DDL text itself, which is what a future
        hand would edit.
        """
        from softae.core.data_store import _DDL

        project = tmp_path / "pre_t77"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as store:               # must not raise
            assert "idx_fit_results_arc_state" in _fit_results_indexes(store)
        assert "idx_fit_results_arc_state" not in _DDL

    def test_a_row_written_before_the_columns_existed_reads_null_not_unknown(
            self, tmp_path: Path) -> None:
        # NULL means *never annotated*. 'unknown' is an answer the annotator gives
        # when it looked and could not tell, and stamping it on a July row would
        # manufacture an inspection that never happened.
        project = tmp_path / "legacy_null"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as store:
            row = self._row(store, 1)
        assert row == {"arc_state": None, "arc_f_peak_hz": None,
                       "arc_f_low_hz": None, "arc_phase_low_deg": None}

    # ── `record_fit` population ─────────────────────────────────────────────

    def test_record_fit_stores_the_arc_state_from_the_fit_when_no_report_is_passed(
            self, store_with_run) -> None:
        # The columns come from the FIT, so the legacy `report=None` call — which is
        # most of them — populates them just the same.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _annotated())
        row = self._row(store, fid)
        assert row["arc_state"] == "open"
        assert row["arc_f_peak_hz"] == pytest.approx(20.0)
        assert row["arc_phase_low_deg"] == pytest.approx(-41.5)

    def test_record_fit_stores_the_arc_columns_when_a_report_shaped_object_is_passed(
            self, store_with_run) -> None:
        """§3.3 made executable: a real report has NO arc entry in its gate log.

        `annotate_arc_closure` writes to the fit and nowhere else, so an
        implementation that scanned `report` would silently NULL these columns the
        day P.18 passes the genuine SpectrumReport. This is that day, in advance.
        """
        from types import SimpleNamespace

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        report = SimpleNamespace(
            sigma=SimpleNamespace(mode="split", is_bound=False),
            cell=SimpleNamespace(dead_height_cm=0.0),
            quality=SimpleNamespace(verdict="ok"),
            gate_log=[{"gate": "kk_residual", "severity": "warn", "passed": True,
                       "n_dropped": 0, "detail": "no arc entry here"}],
        )
        fid = store.record_fit(mid, _annotated(), report=report)

        row = self._row(store, fid)
        assert row["arc_state"] == "open"
        # ...and the gate log is still the report's own, untranslated.
        import json
        log, = store._conn.execute(
            "SELECT gate_log_json FROM fit_results WHERE fit_id = ?",
            (fid,)).fetchone()
        assert [e["gate"] for e in json.loads(log)] == ["kk_residual"]

    def test_record_fit_leaves_the_arc_columns_null_for_a_fit_without_an_annotation(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _FakeFitResult())
        assert self._row(store, fid) == {
            "arc_state": None, "arc_f_peak_hz": None,
            "arc_f_low_hz": None, "arc_phase_low_deg": None}

    def test_an_unknown_arc_state_is_stored_as_the_word_unknown_not_as_null(
            self, store_with_run) -> None:
        # UNKNOWN is an answer with a reason attached, and it is a different fact
        # from a row nobody annotated. The two must not collapse into one NULL.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _annotated(
            "unknown", f_peak=float("nan"), f_low=float("nan"),
            phase=float("nan")))
        row = self._row(store, fid)
        assert row["arc_state"] == "unknown"
        # Its numbers are absent, though — there was no peak to report.
        assert row["arc_f_peak_hz"] is None and row["arc_f_low_hz"] is None

    def test_a_nan_peak_frequency_is_stored_as_null_not_as_nan(
            self, store_with_run) -> None:
        """`_f_or_none` is the file's NaN -> NULL boundary and these columns use it.

        **The `_arc_columns` assertion is the load-bearing one, and the round trip
        below cannot substitute for it.** Measured on sqlite3 3.40.1: the driver
        binds a Python NaN to NULL by itself, `IS NULL` true, so a stored row reads
        NULL whether or not `_f_or_none` ever ran — an implementation that passed
        NaN straight through would pass a database assertion. The boundary earns its
        place on the way OUT, per its own docstring: a NaN that survives reads as a
        number to anything checking only for `None`, so "absent" has to be one state
        before the value reaches the driver at all. Do not "simplify" this back to
        the round trip.
        """
        from softae.core.data_store import _arc_columns

        fit = _annotated(phase=float("nan"))
        mapped = _arc_columns(fit)
        assert mapped["arc_phase_low_deg"] is None
        assert mapped["arc_f_peak_hz"] == pytest.approx(20.0)

        # Kept alongside, not traded away: this proves the mapped value survives the
        # INSERT and the SELECT as an absence.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        row = self._row(store, store.record_fit(mid, fit))
        assert row["arc_state"] == "open"
        assert row["arc_phase_low_deg"] is None

    def test_an_annotated_fit_leaves_the_gate_log_json_column_empty(
            self, store_with_run) -> None:
        # The arc record lives in the columns and only there. A `report=None` row —
        # which every caller now is — carries the literal it always carried, which is
        # what `shadow_db` distinguishes an unannotated pre-column row by.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _annotated())
        assert store._conn.execute(
            "SELECT gate_log_json FROM fit_results WHERE fit_id = ?",
            (fid,)).fetchone()[0] == "[]"


# ---------------------------------------------------------------------------
# Goodness of fit — five columns, sourced from the fit and not from the report
# ---------------------------------------------------------------------------


#: Measured on ch25_001 (1.5x from the numpy Kasa anchor) and ch32_004 (271x),
#: legacy fitter, run live on two stored spectra. The pair is used rather than
#: round numbers because the discrimination is the point: `residual_rms_pct`
#: separates these two by 34x where `r_squared` separates them by 1.8x.
_GOOD_FIT_QUALITY = {
    "chi2": 0.4147, "chi2_reduced": 0.0207, "r_squared": 0.948,
    "residual_rms_pct": 38.6, "residual_max_pct": 121.3,
}
_BAD_FIT_QUALITY = {
    "chi2": 84.11, "chi2_reduced": 4.2055, "r_squared": 0.521,
    "residual_rms_pct": 1297.4, "residual_max_pct": 4820.9,
}


def _with_quality(quality: dict | None):
    """A fit carrying `quality`, as `fit_circuit` leaves one."""
    fit = _FakeFitResult()
    fit.quality = quality
    return fit


class TestFitQualityColumns:
    """Five columns instead of nothing at all.

    ``FitResult.quality`` has been computed at fit time since P4.1 and discarded at
    this boundary ever since, so the database held 3619 fits and not one number
    saying whether any of them described its data. A non-converged fit still reports
    an ``R1``, and the sigma derived from it is indistinguishable from a good one
    without these.

    Both metrics are DIMENSIONLESS, which is why they can carry a threshold at all:
    this formulation space spans ~10 decades of conductivity, so no absolute bound on
    ``R1`` or ``sigma_S_per_cm`` is writable.
    """

    def _row(self, store: DataStore, fid: int) -> dict:
        return dict(store._conn.execute(
            "SELECT chi2, chi2_reduced, r_squared, residual_rms_pct, "
            "residual_max_pct FROM fit_results WHERE fit_id = ?",
            (fid,)).fetchone())

    # ── Schema ──────────────────────────────────────────────────────────────

    def test_the_columns_are_declared_in_the_ddl_and_not_only_in_the_migration(
            self, store: DataStore) -> None:
        """Asserted against the `_DDL` TEXT, and that is not pedantry.

        The migration runs on every open, including on a brand-new database, so
        `PRAGMA table_info` alone cannot tell a DDL-declared column from one the
        ALTER loop supplied — a fresh store looks identical either way. `_DDL` is
        what a future hand edits, and the two descriptions of one schema drift
        exactly where nothing checks the text.
        """
        from softae.core.data_store import _DDL

        assert set(FIT_QUALITY_COLUMNS) <= _fit_results_columns(store)
        for name in FIT_QUALITY_COLUMNS:
            assert f"\n    {name} " in _DDL, f"{name} missing from _DDL"

    def test_the_column_names_are_the_metric_names_untranslated(self) -> None:
        """The dict keys `compute_fit_quality` returns ARE the column names.

        A rename between the metric and its column is how a reader ends up querying
        a number that is no longer the one the fitter computed. Asserted against
        `quality.py`'s own return, not against a second list written here.
        """
        import inspect

        from softae.analysis import quality as quality_mod

        source = inspect.getsource(quality_mod.compute_fit_quality)
        for name in FIT_QUALITY_COLUMNS:
            assert f'"{name}"' in source

    def test_no_goodness_of_fit_column_carries_a_not_null_default(
            self, store: DataStore) -> None:
        """The `gate_log_json` shape, refused deliberately.

        `gate_log_json TEXT NOT NULL DEFAULT '[]'` makes absence structurally
        unobservable: 100% "coverage" by null-check, 99.4% empty in fact. NULL here
        must stay reachable, because a fit that recorded no metrics is a different
        row from one that scored badly.
        """
        decls = _fit_results_declarations(store)
        for name in FIT_QUALITY_COLUMNS:
            type_, notnull, default = decls[name]
            assert type_ == "REAL"
            assert notnull == 0, f"{name} is NOT NULL"
            assert default is None, f"{name} defaults to {default!r}"

    # ── `record_fit` population ─────────────────────────────────────────────

    def test_record_fit_stores_all_five_metrics_from_a_populated_quality_dict(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _with_quality(_GOOD_FIT_QUALITY))
        assert self._row(store, fid) == pytest.approx(_GOOD_FIT_QUALITY)

    def test_the_report_less_path_populates_the_columns(
            self, store_with_run) -> None:
        """The whole design constraint, made executable.

        `record_fit` receives a *fit* at all four call sites and a *report* at
        exactly one (`tools/eis_validate.py`). `analysis/eis/router.py`,
        `gui/tabs/tab_analysis.py` and `gui/tabs/tab_manual.py` pass none. An
        implementation that read `report.quality` would leave these columns as empty
        as `gate_verdict` is — 126 rows of 3619 — while every test that passed a
        report stayed green.
        """
        import inspect

        from softae.core import data_store as ds_mod

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        # No `report=`, exactly as the three call sites call it.
        fid = store.record_fit(mid, _with_quality(_BAD_FIT_QUALITY))
        assert self._row(store, fid) == pytest.approx(_BAD_FIT_QUALITY)
        # ...and the source pins WHERE it read them from, because a future edit that
        # moved the read into `_fit_report_columns` would still pass the assertion
        # above for as long as a report happened to be absent.
        assert "report" not in inspect.signature(
            ds_mod._fit_quality_columns).parameters

    def test_a_fit_with_no_quality_stores_null_and_null_is_not_zero(
            self, store_with_run) -> None:
        """NULL means *not recorded*; 0.0 is a real, and terrible, `r_squared`.

        `compute_fit_quality` returns exactly 0.0 for `r_squared` when the trace has
        no variance (`ss_tot == 0`), so the two states genuinely both occur and a
        sentinel would fuse them.
        """
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        absent = store.record_fit(mid, _with_quality({}))
        # A sibling row carrying REAL zeros, so the assertion below discriminates
        # rather than merely restating the fixture: a sentinel implementation would
        # make these two rows identical and only this pair notices.
        zeroed = store.record_fit(mid, _with_quality(
            dict.fromkeys(FIT_QUALITY_COLUMNS, 0.0)))

        assert self._row(store, absent) == dict.fromkeys(FIT_QUALITY_COLUMNS, None)
        assert self._row(store, zeroed) == dict.fromkeys(FIT_QUALITY_COLUMNS, 0.0)
        for name in FIT_QUALITY_COLUMNS:
            nulls = store._conn.execute(
                f"SELECT fit_id FROM fit_results WHERE {name} IS NULL").fetchall()
            zeros = store._conn.execute(
                f"SELECT fit_id FROM fit_results WHERE {name} = 0.0").fetchall()
            assert [r[0] for r in nulls] == [absent], name
            assert [r[0] for r in zeros] == [zeroed], name

    def test_a_recorded_zero_r_squared_is_stored_as_zero_and_not_as_null(
            self, store_with_run) -> None:
        # The other half of the distinction: a fit that scored 0.0 must be
        # distinguishable from one that scored nothing, or the column answers a
        # different question than the one asked of it.
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _with_quality(dict(_BAD_FIT_QUALITY,
                                                       r_squared=0.0)))
        assert self._row(store, fid)["r_squared"] == 0.0
        assert store._conn.execute(
            "SELECT COUNT(*) FROM fit_results WHERE r_squared IS NULL"
        ).fetchone()[0] == 0

    def test_a_fit_object_with_no_quality_attribute_at_all_stores_null(
            self, store_with_run) -> None:
        # Pre-P4.1 fits and duck-typed stand-ins have no `quality` attribute; the
        # honest record for them is the same NULL, not an AttributeError.
        class _NoQuality:
            model_name = "simpleSalt"
            parameters = [1.0]
            R0 = 100.0
            R1 = 1000.0
            success = True
            error_msg = ""

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _NoQuality())
        assert self._row(store, fid) == dict.fromkeys(FIT_QUALITY_COLUMNS, None)

    def test_a_nan_metric_becomes_none_at_the_boundary_not_at_the_driver(
            self, store_with_run) -> None:
        """Asserted on `_fit_quality_columns`, and the round trip CANNOT assert it.

        Measured: sqlite3 3.40.1 binds a Python NaN to NULL by itself, so a stored
        row reads NULL whether or not `_f_or_none` ran — an implementation that
        skipped the boundary entirely would pass a database assertion. (This is
        exactly the vacuity `test_a_nan_peak_frequency_is_stored_as_null_not_as_nan`
        above still has; reported, not fixed here.) `_f_or_none` earns its place on
        the way OUT, per its own docstring: a NaN that survives reads as a number to
        anything checking only for `None`, so "absent" must be one state before the
        value ever reaches the driver.
        """
        from softae.core.data_store import _fit_quality_columns

        fit = _with_quality(dict(_GOOD_FIT_QUALITY, residual_max_pct=float("nan")))
        mapped = _fit_quality_columns(fit)
        assert mapped["residual_max_pct"] is None
        assert mapped["r_squared"] == pytest.approx(0.948)

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        row = self._row(store, store.record_fit(mid, fit))
        assert row["residual_max_pct"] is None

    def test_an_unexpected_quality_key_is_ignored_rather_than_stored(
            self, store_with_run) -> None:
        # The mapping is over `FIT_QUALITY_COLUMNS`, not over the dict, so a new
        # metric appearing upstream cannot become an unbound INSERT parameter — it
        # is simply not persisted until a column exists for it.
        from softae.core.data_store import _fit_quality_columns

        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fit = _with_quality(dict(_GOOD_FIT_QUALITY, aic=17.0))
        # The KEY SET is what is pinned. Asserting only that the row came back right
        # would pass a pass-through implementation too, since `record_fit` names its
        # INSERT columns and would simply ignore the extra key — until the day
        # something iterates the mapping instead of indexing it.
        assert set(_fit_quality_columns(fit)) == set(FIT_QUALITY_COLUMNS)
        fid = store.record_fit(mid, fit)                 # must not raise
        assert self._row(store, fid) == pytest.approx(_GOOD_FIT_QUALITY)
        assert "aic" not in _fit_results_columns(store)

    # ── The T7.9 boundary ───────────────────────────────────────────────────

    def test_a_report_less_record_fit_writes_no_gate_verdict(
            self, store_with_run) -> None:
        """P.18 / T7.9 is deferred on the operator's bench shadow run.

        `gate_verdict` is derived from the *report* and there is no separate switch,
        so any call site that starts passing one starts persisting verdicts — from
        gates measured anti-correlated with accuracy, under thresholds that have no
        per-row record yet (`SHADOW_CAMPAIGN.md:420`). These columns must not smuggle
        that across: adding them changes what a report-less call stores about the
        FIT and nothing about what it stores about the GATES.
        """
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        fid = store.record_fit(mid, _with_quality(_GOOD_FIT_QUALITY))

        verdict, log = store._conn.execute(
            "SELECT gate_verdict, gate_log_json FROM fit_results WHERE fit_id = ?",
            (fid,)).fetchone()
        assert verdict is None
        assert log == "[]"
        assert store._conn.execute(
            "SELECT COUNT(*) FROM fit_results WHERE gate_verdict IS NOT NULL"
        ).fetchone()[0] == 0

    # ── Round trip and migration ────────────────────────────────────────────

    def test_query_fits_returns_the_metrics(self, store_with_run) -> None:
        store, run_id = store_with_run
        mid = store.record_measurement(run_id, _make_eis_result())
        store.record_fit(mid, _with_quality(_GOOD_FIT_QUALITY))

        row, = store.query_fits(measurement_id=mid)
        assert {k: row[k] for k in FIT_QUALITY_COLUMNS} == pytest.approx(
            _GOOD_FIT_QUALITY)

    def test_a_legacy_database_gains_the_columns_and_its_old_rows_read_null(
            self, tmp_path: Path) -> None:
        # No backfill: the residual needs the spectrum AND the fitted trace, and
        # `z_fit` was never stored, so nothing recoverable after the fact
        # reconstructs a metric for a 2026-07 row. NULL is the honest record — the
        # same argument `_migrate_experiment_skipped_channels` makes.
        project = tmp_path / "legacy_quality"
        _build_legacy_fit_results_db(project)

        with DataStore(project) as store:
            assert set(FIT_QUALITY_COLUMNS) <= _fit_results_columns(store)
            assert self._row(store, 1) == dict.fromkeys(FIT_QUALITY_COLUMNS, None)

    def test_the_addition_takes_no_new_schema_epoch_row(self) -> None:
        """A change of shape, not of meaning — T2.3's condition for skipping one.

        No stored value's interpretation moves: every pre-existing column keeps its
        value, and the new ones are NULL on every row written before them. Contrast
        epochs 2, 5 and 6, where a column held still and its meaning moved.
        """
        from softae.core.data_store import SCHEMA_EPOCHS

        assert not any(
            any(name in note for name in FIT_QUALITY_COLUMNS)
            for _, _, note in SCHEMA_EPOCHS
        )
        assert max(v for v, _, _ in SCHEMA_EPOCHS) == 6


# ---------------------------------------------------------------------------
# Skipped-channel provenance on the run row (halt/park Priority 4)
# ---------------------------------------------------------------------------


#: An `experiments` table as it stood before `skipped_channels` existed — i.e.
#: the shape of every database now on disk.
_LEGACY_EXPERIMENTS_DDL = """\
CREATE TABLE experiments (
    run_id               TEXT PRIMARY KEY,
    started_at           TEXT NOT NULL,
    finished_at          TEXT,
    workflow_name        TEXT NOT NULL,
    workflow_mode        TEXT NOT NULL DEFAULT 'unknown',
    campaign             TEXT NOT NULL DEFAULT 'dev',
    quality              TEXT NOT NULL DEFAULT 'explore',
    pcb_name             TEXT,
    eis_preset           TEXT,
    config_snapshot_json TEXT NOT NULL DEFAULT '{}',
    config_hash          TEXT NOT NULL DEFAULT '',
    annotation           TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'running'
)"""


class TestSkippedChannelProvenance:
    """Three states, because "nobody counted" is not "nothing was skipped"."""

    def test_finish_run_without_the_argument_leaves_the_column_null(
            self, store_with_run) -> None:
        # Every pre-existing caller (the campaign wiring, the Arrhenius tab) lands
        # here, and none of them counts skips. NULL is the only honest value.
        store, run_id = store_with_run
        store.finish_run(run_id, "done")
        assert store.run_skipped_channels(run_id) is None
        assert store.query_runs()[0]["skipped_channels"] is None

    def test_finish_run_with_an_empty_list_records_none_skipped(
            self, store_with_run) -> None:
        store, run_id = store_with_run
        store.finish_run(run_id, "done", skipped_channels=[])
        assert store.run_skipped_channels(run_id) == []

    def test_finish_run_records_the_skipped_channels_as_ints(
            self, store_with_run) -> None:
        # Steps carry the channel as a tag, so it arrives as "3"/"7"; a reader
        # asking which wells are real wants numbers.
        store, run_id = store_with_run
        store.finish_run(run_id, "partial", skipped_channels=["3", "7"])
        assert store.run_skipped_channels(run_id) == [3, 7]
        row = store.query_runs()[0]
        assert row["status"] == "partial"
        assert row["finished_at"] is not None

    def test_run_skipped_channels_is_none_for_an_unknown_run(
            self, store: DataStore) -> None:
        assert store.run_skipped_channels("no_such_run") is None

    def test_legacy_experiments_table_gains_the_column_with_rows_intact(
            self, tmp_path: Path) -> None:
        """The bare CREATE TABLE IF NOT EXISTS cannot add it — the migration must.

        And the historical row must come back NULL: it was written `done` whether
        it skipped nothing or half the plate, and nothing recovers which.
        """
        project = tmp_path / "legacy_experiments"
        _build_legacy_conditions_db(
            project,
            _LEGACY_EXPERIMENTS_DDL,
            "INSERT INTO experiments (run_id, started_at, finished_at, "
            "workflow_name, status) VALUES ('old_run', '2026-01-01T00:00:00Z', "
            "'2026-01-01T01:00:00Z', 'ht_experiment', 'done')",
        )

        with DataStore(project) as store:
            cols = {
                r[1] for r in
                store._conn.execute("PRAGMA table_info(experiments)").fetchall()
            }
            assert "skipped_channels" in cols
            rows = store.query_runs()
            assert len(rows) == 1
            assert rows[0]["run_id"] == "old_run"
            assert rows[0]["skipped_channels"] is None
            assert store.run_skipped_channels("old_run") is None


class TestCampaignCheckpointCounterMigration:
    """Both escalation counters must reach a store that already exists.

    ``campaign_checkpoints`` is created by a bare ``CREATE TABLE IF NOT EXISTS``,
    so the DDL alone is correct on a fresh store and silently broken on every
    store that already has the table — which is every real one. A missing column
    is not a quiet degradation here: the resume read names it, so the failure is
    ``no such column`` at the moment a parked overnight campaign is restarted.
    """

    #: The table as it stood before either counter column existed.
    _LEGACY_DDL = (
        "CREATE TABLE campaign_checkpoints ("
        " campaign TEXT PRIMARY KEY, run_id TEXT, iteration INTEGER NOT NULL,"
        " loop_state TEXT, board_id INTEGER, spec_json TEXT,"
        " optimizer_json TEXT, updated_at TEXT NOT NULL)"
    )

    def _legacy_store(self, project: Path) -> None:
        store = DataStore(project)
        store._conn.execute("DROP TABLE campaign_checkpoints")
        store._conn.execute(self._LEGACY_DDL)
        store._conn.execute(
            "INSERT INTO campaign_checkpoints (campaign, run_id, iteration,"
            " loop_state, updated_at) VALUES ('overnight', 'r1', 7, 'STOPPED',"
            " '2026-01-01T00:00:00Z')")
        store._conn.commit()
        store.close()

    def test_a_legacy_table_gains_both_counter_columns(self, tmp_path: Path) -> None:
        project = tmp_path / "legacy_checkpoints"
        self._legacy_store(project)

        with DataStore(project) as store:
            cols = {
                r[1] for r in store._conn.execute(
                    "PRAGMA table_info(campaign_checkpoints)").fetchall()
            }
            assert {"rh_ceiling_streak", "consecutive_failures"} <= cols

    def test_a_pre_existing_checkpoint_reads_back_with_zeroed_counters(
        self, tmp_path: Path
    ) -> None:
        """Zero is the honest value — it had never counted anything.

        The row itself must survive: a checkpoint is the only thing standing
        between an interrupted multi-day campaign and starting it again.
        """
        project = tmp_path / "legacy_checkpoints"
        self._legacy_store(project)

        with DataStore(project) as store:
            cp = store.campaign_checkpoint("overnight")
            assert cp is not None
            assert cp["iteration"] == 7                  # the row survived
            assert cp["rh_ceiling_streak"] == 0
            assert cp["consecutive_failures"] == 0

    def test_a_fresh_database_has_the_counters_from_the_ddl_not_only_the_migration(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the DDL and the migration drift and only one is ever tested."""
        with DataStore(tmp_path / "fresh") as store:
            cols = {
                r[1] for r in store._conn.execute(
                    "PRAGMA table_info(campaign_checkpoints)").fetchall()
            }
            assert {"rh_ceiling_streak", "consecutive_failures"} <= cols

    def test_the_failure_streak_round_trips_through_a_save(self, tmp_path: Path) -> None:
        """The write replaces the whole row, so an unpassed counter is a lost one."""
        with DataStore(tmp_path / "proj") as store:
            store.save_campaign_checkpoint(
                "overnight", iteration=3, consecutive_failures=2)
            assert store.campaign_checkpoint("overnight")["consecutive_failures"] == 2

            store.save_campaign_checkpoint("overnight", iteration=4)
            assert store.campaign_checkpoint("overnight")["consecutive_failures"] == 0


class TestRunOutcome:
    """``run_outcome`` — the read side of ``finish_run``.

    Its two fields answer different questions and the second is the one that is
    easy to omit: ``status`` says which exit path closed the row, ``finished``
    says whether anything closed it at all.
    """

    def test_a_finished_run_reports_its_status(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj") as store:
            run_id = store.start_run("wf")
            store.finish_run(run_id, "converged")
            assert store.run_outcome(run_id) == {
                "status": "converged", "finished": True}

    def test_a_live_run_is_reported_unfinished(self, tmp_path: Path) -> None:
        """The state a hard kill leaves: status still at its default, never closed.

        This is what stops a reader mistaking a killed run for a completed one
        before the next launch's recovery sweep rewrites the status.
        """
        with DataStore(tmp_path / "proj") as store:
            outcome = store.run_outcome(store.start_run("wf"))
            assert outcome == {"status": "running", "finished": False}

    def test_an_unknown_run_has_no_outcome(self, tmp_path: Path) -> None:
        with DataStore(tmp_path / "proj") as store:
            assert store.run_outcome("no-such-run") is None


# ---------------------------------------------------------------------------
# Non-finite floats never reach a JSON column
# ---------------------------------------------------------------------------


class TestNonFiniteFloatsNeverReachAJsonColumn:
    """``json.dumps`` emits bare ``NaN``; SQLite's JSON1 rejects the document.

    The failure is not row-local. The JSON1 predicate is evaluated per row, so a
    single NaN-bearing row anywhere in ``measurements`` makes every
    ``json_extract`` query over the table raise ``malformed JSON`` — including
    queries about entirely unrelated rows and unrelated keys. And it is reachable
    in ordinary operation: ``arc_closure`` returns NaN for an apex it did not
    find, which is the common case on an open-arc sweep, and the scout stamps that
    straight into ``eis_params``.

    These tests pin the *boundary*, not any one stamper: ``DataStore`` has exactly
    one INSERT into ``eis_params_json`` and no UPDATE, so scrubbing there is what
    makes the guarantee unforgeable.
    """

    def test_a_nan_parameter_is_stored_as_json_null(self, store_with_run) -> None:
        """``null`` — not the string ``"NaN"``, and not omitted.

        A string forces every reader to know to parse it; an omission is a
        different fact from a null for any key whose absence means "this run
        predates the field".
        """
        import json

        store, run_id = store_with_run
        eis = _make_eis_result()
        eis.eis_params["eis_scout_apex_hz"] = float("nan")
        store.record_measurement(run_id, eis)

        raw = store._conn.execute(
            "SELECT eis_params_json FROM measurements").fetchone()[0]
        assert "NaN" not in raw
        params = json.loads(raw)
        assert "eis_scout_apex_hz" in params          # present, not dropped
        assert params["eis_scout_apex_hz"] is None    # null, not the string "NaN"

    def test_infinity_is_stored_as_json_null_too(self, store_with_run) -> None:
        """``json.dumps`` renders ``inf`` as ``Infinity``, equally invalid JSON.

        ``_f_or_none``'s ``f == f`` idiom catches NaN and misses this one, which is
        why the JSON boundary tests for finiteness rather than for NaN.
        """
        import json

        store, run_id = store_with_run
        eis = _make_eis_result()
        eis.eis_params["eis_scout_apex_hz"] = float("inf")
        eis.eis_params["band_below_apex_decades"] = float("-inf")
        store.record_measurement(run_id, eis)

        raw = store._conn.execute(
            "SELECT eis_params_json FROM measurements").fetchone()[0]
        assert "Infinity" not in raw
        params = json.loads(raw)
        assert params["eis_scout_apex_hz"] is None
        assert params["band_below_apex_decades"] is None

    def test_finite_values_are_untouched(self, store_with_run) -> None:
        """The scrub must be invisible to every value that was already legal."""
        import json

        store, run_id = store_with_run
        eis = _make_eis_result()
        eis.eis_params.update({
            "eis_scout_apex_hz": 42.5,
            "eis_scout_verdict": "ok",
            "eis_scout_enabled": True,
            "eis_scout_note": None,
        })
        store.record_measurement(run_id, eis)

        params = json.loads(store._conn.execute(
            "SELECT eis_params_json FROM measurements").fetchone()[0])
        assert params["eis_scout_apex_hz"] == 42.5
        assert params["npts"] == 10
        assert params["eis_scout_verdict"] == "ok"
        assert params["eis_scout_enabled"] is True
        assert params["eis_scout_note"] is None

    def test_nested_and_array_valued_params_are_scrubbed(
        self, store_with_run
    ) -> None:
        """One NaN buried in a list or a sub-dict poisons the document identically."""
        import json

        import numpy as np

        store, run_id = store_with_run
        eis = _make_eis_result()
        eis.eis_params["segments"] = [
            {"f_hi": 1e5, "f_lo": float("nan")},
            {"f_hi": 1e3, "f_lo": 1.0},
        ]
        eis.eis_params["residuals"] = np.array([1.0, np.nan, 3.0])
        store.record_measurement(run_id, eis)

        raw = store._conn.execute(
            "SELECT eis_params_json FROM measurements").fetchone()[0]
        assert "NaN" not in raw
        params = json.loads(raw)
        assert params["segments"][0]["f_lo"] is None
        assert params["segments"][1]["f_lo"] == 1.0
        assert params["residuals"] == [1.0, None, 3.0]

    def test_a_nan_bearing_row_leaves_json_extract_working_for_every_row(
        self, store_with_run
    ) -> None:
        """The regression that matters: the whole TABLE stays queryable.

        Before the boundary scrub this raised ``malformed JSON`` and returned
        nothing at all — not merely a bad value for the offending row. Any reader
        forced to prefilter with ``LIKE`` instead of ``json_extract`` is paying
        this tax.
        """
        store, run_id = store_with_run

        open_arc = _make_eis_result(channel=1)
        open_arc.eis_params["eis_scout_apex_hz"] = float("nan")
        open_arc.eis_params["eis_scout_verdict"] = "open_no_apex"
        store.record_measurement(run_id, open_arc)

        closed_arc = _make_eis_result(channel=2)
        closed_arc.eis_params["eis_scout_apex_hz"] = 137.0
        closed_arc.eis_params["eis_scout_verdict"] = "ok"
        store.record_measurement(run_id, closed_arc)

        rows = store._conn.execute(
            "SELECT channel, "
            "       json_extract(eis_params_json, '$.eis_scout_verdict'), "
            "       json_extract(eis_params_json, '$.eis_scout_apex_hz') "
            "  FROM measurements ORDER BY channel"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            (1, "open_no_apex", None),
            (2, "ok", 137.0),
        ]

        # …and the row-selecting form works too: the predicate is what used to
        # raise, so a WHERE over the NaN row is the sharper version of the test.
        picked = store._conn.execute(
            "SELECT channel FROM measurements "
            " WHERE json_extract(eis_params_json, '$.eis_scout_verdict') = 'ok'"
        ).fetchall()
        assert [r[0] for r in picked] == [2]

    def test_the_scrub_covers_every_json_column_the_store_writes(
        self, store_with_run
    ) -> None:
        """``_safe_json`` is shared, so ``gate_log_json`` gets the same guarantee.

        ``arc_closure``'s own gate dict already nulls its non-finite fields, but the
        column must not depend on every producer remembering to.
        """
        from softae.core.data_store import _safe_json

        assert _safe_json({"f_peak_hz": float("nan")}) == '{"f_peak_hz": null}'
        assert _safe_json([float("inf"), 2.0]) == "[null, 2.0]"
        assert _safe_json(np.array([np.nan, 1.0])) == "[null, 1.0]"


class TestRunOwnerLiveness:
    """``experiments.owner_pid`` — what tells a *crashed* run from a *live* one.

    ``unfinished_runs()`` was a bare ``finished_at IS NULL``, and its three
    readers all go on to relabel what it returns ``interrupted`` and offer to
    park the rig. A live run's row and a crashed run's row are byte-for-byte
    identical, so the only thing separating a working headless tool from being
    parked out from under it was ``foreign_run_lock()`` — an external oracle
    several tools never populated. The row now carries its own answer.
    """

    #: ``experiments`` exactly as it stood before ``owner_pid`` was declared.
    #: The store's own DDL is a bare ``CREATE TABLE IF NOT EXISTS``, so a table
    #: that already exists in this shape is what every real database looks like
    #: on the first open after the upgrade.
    _LEGACY_DDL = (
        "CREATE TABLE experiments ("
        " run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,"
        " workflow_name TEXT NOT NULL,"
        " workflow_mode TEXT NOT NULL DEFAULT 'unknown',"
        " campaign TEXT NOT NULL DEFAULT 'dev',"
        " quality TEXT NOT NULL DEFAULT 'explore',"
        " pcb_name TEXT, eis_preset TEXT,"
        " config_snapshot_json TEXT NOT NULL DEFAULT '{}',"
        " config_hash TEXT NOT NULL DEFAULT '',"
        " annotation TEXT NOT NULL DEFAULT '',"
        " status TEXT NOT NULL DEFAULT 'running', skipped_channels TEXT)"
    )

    #: A run row the pre-column code wrote: in flight, and naming no owner
    #: because there was nowhere to name one.
    _LEGACY_RUN = "20260501T010203Z_ran_before_the_upgrade"

    def _pre_column_store(self, project: Path) -> str:
        """Build a database in the shape it had before this column existed.

        Written with raw sqlite3 rather than by mutating a fresh store, because
        ``ALTER TABLE ... DROP COLUMN`` re-parses the stored schema and this
        table's DDL is heavily commented. The point stands either way: the
        migration is tested against a database created *before* the change.
        """
        import sqlite3

        (project / "db").mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(project / "db" / "softae.db"))
        conn.execute(self._LEGACY_DDL)
        conn.execute(
            "INSERT INTO experiments (run_id, started_at, workflow_name)"
            " VALUES (?, '2026-05-01T01:02:03Z', 'ran_before_the_upgrade')",
            (self._LEGACY_RUN,))
        conn.commit()
        conn.close()
        return self._LEGACY_RUN

    @staticmethod
    def _owner_of(store: DataStore, run_id: str):
        return store._conn.execute(
            "SELECT owner_pid FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()[0]

    # ── The column ─────────────────────────────────────────────────────────

    def test_a_fresh_database_has_the_owner_column(self, tmp_path: Path) -> None:
        """The DDL and the migration must not drift; only one gets exercised."""
        with DataStore(tmp_path / "fresh") as store:
            cols = {
                r[1] for r in store._conn.execute(
                    "PRAGMA table_info(experiments)").fetchall()
            }
            assert "owner_pid" in cols

    def test_the_column_carries_no_default(self, tmp_path: Path) -> None:
        """An absent owner is a *fact*, not a value.

        A ``DEFAULT`` would stamp an owner onto rows nobody recorded one for.
        ``_pid_alive(0)`` is False so ``DEFAULT 0`` would happen to be harmless
        today — but the harm is not what the number does, it is the claim that
        the row said something it never said.
        """
        with DataStore(tmp_path / "fresh") as store:
            declared = {
                r[1]: r[4] for r in store._conn.execute(
                    "PRAGMA table_info(experiments)").fetchall()
            }
            assert declared["owner_pid"] is None

    def test_start_run_stamps_this_process_as_the_owner(self, store) -> None:
        run_id = store.start_run("wf")
        assert self._owner_of(store, run_id) == os.getpid()

    # ── The migration, against a database created before the change ────────

    def test_a_pre_column_database_gains_the_column_on_open(
        self, tmp_path: Path
    ) -> None:
        project = tmp_path / "legacy_experiments"
        self._pre_column_store(project)

        with DataStore(project) as store:
            cols = {
                r[1] for r in store._conn.execute(
                    "PRAGMA table_info(experiments)").fetchall()
            }
            assert "owner_pid" in cols

    def test_a_pre_column_row_survives_the_migration_with_no_owner(
        self, tmp_path: Path
    ) -> None:
        """No backfill. Nothing recoverable says which process wrote it."""
        project = tmp_path / "legacy_experiments"
        run_id = self._pre_column_store(project)

        with DataStore(project) as store:
            assert store.run_outcome(run_id) == {"status": "running",
                                                 "finished": False}
            assert self._owner_of(store, run_id) is None

    def test_reopening_a_migrated_database_is_a_noop(self, tmp_path: Path) -> None:
        """Idempotent: the PRAGMA guard, not the ALTER, is what runs twice."""
        project = tmp_path / "legacy_experiments"
        run_id = self._pre_column_store(project)

        with DataStore(project):
            pass
        with DataStore(project) as store:                # third open
            owners = store._conn.execute(
                "SELECT owner_pid FROM experiments").fetchall()
            assert [r[0] for r in owners] == [None]
            assert store.run_outcome(run_id)["status"] == "running"

    # ── What `unfinished_runs()` now means ─────────────────────────────────

    def test_a_row_with_no_owner_is_still_reported_unfinished(
        self, tmp_path: Path
    ) -> None:
        """Historical behaviour, preserved exactly.

        Unknown must never resolve to "alive": that would silently stop
        reporting the crashes this table is the only durable record of.
        """
        project = tmp_path / "legacy_experiments"
        run_id = self._pre_column_store(project)

        with DataStore(project) as store:
            assert [r["run_id"] for r in store.unfinished_runs()] == [run_id]

    def test_a_dead_owners_row_is_reported_unfinished(
        self, store, crashed_run
    ) -> None:
        run_id = crashed_run(store, "died_last_night")
        assert [r["run_id"] for r in store.unfinished_runs()] == [run_id]

    def test_a_live_owners_row_is_not_reported_unfinished(self, store) -> None:
        """The defect. This row's owner is the running pytest process."""
        store.start_run("still_going")
        assert store.unfinished_runs() == []

    def test_this_processs_own_row_is_excluded_like_any_other_live_one(
        self, store
    ) -> None:
        """Self-exclusion is deliberate, and it is the same rule, not an exception.

        Unlike ``foreign_run_lock`` — which asks *is someone else driving?* and
        so must ignore itself — this query asks *did this run die?*. Being alive
        is proof that it did not, and a long-lived GUI that starts a run and
        later re-runs the start-up check must not relabel its own live row.
        """
        mine = store.start_run("this_very_process")
        assert self._owner_of(store, mine) == os.getpid()
        assert store.unfinished_runs() == []

    def test_a_finished_row_is_never_reported_whatever_its_owner(
        self, store, crashed_run
    ) -> None:
        """``finished_at`` still decides first; liveness only filters."""
        run_id = crashed_run(store, "crashed_then_stamped")
        store.finish_run(run_id, "interrupted")
        assert store.unfinished_runs() == []

    def test_the_reported_row_carries_the_owner_it_was_judged_on(
        self, store, crashed_run
    ) -> None:
        """The dead PID is evidence an operator can act on, so it is returned."""
        run_id = crashed_run(store, "died_last_night")
        row = store.unfinished_runs()[0]
        assert row["run_id"] == run_id
        assert row["owner_pid"] is not None

    def test_an_owner_value_that_names_no_pid_is_treated_as_unknown(
        self, store
    ) -> None:
        """SQLite columns are dynamically typed, so the value may not be a PID.

        Unknown resolves the same way ``NULL`` does — reported — because the
        alternative is that one unreadable value silences crash reporting for
        the whole database: all three readers wrap this call in a bare
        ``except`` and return nothing at all if it raises.
        """
        run_id = store.start_run("wf")
        store._conn.execute(
            "UPDATE experiments SET owner_pid = 'not-a-pid' WHERE run_id = ?",
            (run_id,))
        store._conn.commit()

        assert [r["run_id"] for r in store.unfinished_runs()] == [run_id]

    def test_the_liveness_predicate_is_the_rig_locks_own(self) -> None:
        """One definition of "alive", shared with the rig lock.

        A second copy here would be free to disagree with the one
        ``foreign_run_lock`` uses, and both answers are read side by side by
        ``gui/widgets/unclean_shutdown`` and ``tools/campaign``.
        """
        from softae.core import data_store as ds_mod
        from softae.core import run_lock

        assert ds_mod._pid_alive is run_lock._pid_alive
