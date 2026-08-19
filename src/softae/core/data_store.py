"""Project-scoped data directory and SQLite backend.

Implements the ``DataStore`` class per the specification in
``docs/SubAgent docs/data_directory_spec.md`` — task d-ii.

The DataStore manages:

* A canonical project directory with ``db/``, ``runs/``, and
  ``formulations/`` subdirectories.
* A single SQLite database (WAL mode) containing six tables:
  ``experiments``, ``measurements``, ``conditions``, ``fit_results``,
  ``formulations``, and ``doe_parameters``.
* Run lifecycle (``start_run`` / ``finish_run``).
* Measurement, condition-snapshot, fit, and formulation persistence.
* Filtered queries for downstream analysis and GUI display.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from softae.analysis.conditions import resolve_temperature_C
from softae.analysis.eis.calibration import MEASUREMENT_ROLES
from softae.analysis.eis.geometry import THICKNESS_METHODS, CellConstant
from softae.analysis.eis_data import EISResult

logger = structlog.get_logger(__name__)

#: Vocabulary for ``formulations.thickness_method`` (P.7) — the analysis-side ladder
#: :data:`~softae.analysis.eis.geometry.THICKNESS_METHODS` **verbatim**, plus the
#: ``'unavailable'`` that :func:`~softae.analysis.eis.geometry.resolve_thickness_cm`
#: returns when no tier could speak. It is derived from that tuple rather than
#: re-spelled, so the tier a row *offers* and the tier a fit *used* cannot drift into
#: two vocabularies. ``'unavailable'`` is absent from the analysis tuple because it is
#: a result, not a tier.
#:
#: ``NULL`` is deliberately **not** a member: it means *never recorded* (a row written
#: before this column existed, or by a writer that has no twin), which is a different
#: fact from ``'unavailable'`` meaning *recorded as absent*.
FORMULATION_THICKNESS_METHODS = THICKNESS_METHODS + ("unavailable",)

#: Seed rows for the ``schema_version`` epoch ledger, ``(version, kind, note)``.
#:
#: Append-only and **never rewritten**: a row already in a database is a statement
#: about data that was written under it, so editing one would retroactively change
#: what historical rows claim. Add a new version instead.
#:
#: The ledger is seeded on every open (``INSERT OR IGNORE``), including into a
#: brand-new database, because it describes the epochs *this code* knows about —
#: not the update history of one file. A fresh database still needs the 2026-08-07
#: row to explain any pre-correction rows later imported into it.
SCHEMA_EPOCHS: tuple[tuple[int, str, str], ...] = (
    (1, "schema", "baseline: consolidates the pre-existing in-line migrations"),
    (2, "data-epoch",
     "2026-08-07 deposit_area_mm2 derivation corrected 4.0 -> 18.704 mm2 "
     "(4-stripe, 4.676x); thickness-derived values written before this date are "
     "comparable only among themselves; IDE board sessile, area unavailable"),
    (3, "schema",
     "2026-08-11 conditions temperature columns renamed for instrument "
     "provenance: temp_pv_C -> chamber_air_C (the I2C humidity sensor's chamber "
     "AIR, never the sample) and temp_sp_C -> stage_temp_sp_C (the Modbus stage "
     "setpoint). Values, units and row meaning are UNCHANGED - only the labels. "
     "Contrast version 2, a 'data-epoch': there the name held still and the "
     "numbers changed meaning; here the numbers held still and the names "
     "changed. That is what the kind column is for. Pre-rename readers that "
     "took temp_pv_C for 'the temperature' were reading air up to 42 C below "
     "the stage; see softae.analysis.conditions"),
    (4, "schema",
     "2026-08-12 conditions gains temperature_C + temperature_source: the "
     "answer resolve_temperature_C() would give for a row, computed once at "
     "record time and stored, instead of re-derived by every consumer (and "
     "restated as a COALESCE inside query_measurements(temp_range=...)). "
     "softae.analysis.conditions called this end state 'data-epoch-grade' - "
     "that phrase measures how significant the change is, not what kind it is. "
     "The kind column asks only whether stored numbers changed meaning, and "
     "none did: every source column keeps its value, and the new columns are a "
     "deterministic function of columns already in the same row. So: 'schema'. "
     "Backfilled across all historical rows, unlike the deliberately "
     "NULL-for-historical columns of T2.6 / T3.1b - see "
     "_migrate_conditions_resolved_temperature for why a derivation differs "
     "from a fact the past failed to record. temperature_source is never NULL "
     "after this epoch; 'unavailable' means no thermometer spoke. The "
     "per-database distribution is not recorded here - it is a fact about one "
     "file, not about the epoch, and this tuple is a per-code constant - so it "
     "is obtained with SELECT temperature_source, COUNT(*) FROM conditions "
     "GROUP BY temperature_source. The migration that wrote it logs the same "
     "counts once, per database, as it goes"),
    (5, "data-epoch",
     "2026-08-18 fit_results.R1 may be produced by the TWO-POINT DEBYE READ instead "
     "of the CPE circuit fit, on OPEN-ARC spectra only, when [eis.pregate] "
     "two_point_open is armed. Column, units and type are unchanged - R1 is still a "
     "REAL in ohms - and that is exactly why this is an epoch and not a schema note: "
     "it is version 2's situation, where deposit_area_mm2 changed DERIVATION while "
     "the column held still, and not version 3's, where the numbers held still and "
     "the names moved. Operator-authorized by name, on the rationale that the "
     "deliberate change is acceptable 'given that the raw data will be better "
     "represented': on truncated arcs the CPE fitter's median R_est/R_true is 2.752 "
     "(175.2% over) against the two-point read's 1.598 (60.9%), p16 = 0.031, and all "
     "80 fits reported success while none declined - so the cheaper estimator is the "
     "less biased one precisely where it is used. SCOPE, and it is narrow: arcs that "
     "CLOSED are untouched and provably so; only spectra whose arc did not close AND "
     "whose phase at the sweep floor is still essentially capacitive are diverted. "
     "COMPARABILITY: rows either side of this date are comparable only among "
     "themselves ON THE OPEN POPULATION; the closed population is continuous across "
     "it. NO BACKFILL - historical rows keep their CPE-fit values, because "
     "recomputing them would manufacture the false comparability this ledger exists "
     "to prevent, the same argument _migrate_experiment_skipped_channels makes for "
     "leaving NULL alone. A row does not have to be dated to be read: "
     "fit_results.engine carries 'gated_two_point' on exactly the diverted rows, so "
     "the population is SELECT-able rather than inferred from this timestamp - which "
     "is the whole point, since [a53] records that the failure mode on this rig is "
     "biased R1 flowing through UNLABELLED. Shipped DISABLED; the epoch begins for a "
     "given database when someone arms the flag, not when this row was seeded"),
)

# ---------------------------------------------------------------------------
# DDL — executed once at DataStore construction
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS experiments (
    run_id              TEXT    PRIMARY KEY,
    started_at          TEXT    NOT NULL,
    finished_at         TEXT,
    workflow_name       TEXT    NOT NULL,
    workflow_mode       TEXT    NOT NULL DEFAULT 'unknown',
    campaign            TEXT    NOT NULL DEFAULT 'dev',
    quality             TEXT    NOT NULL DEFAULT 'explore',
    pcb_name            TEXT,
    eis_preset          TEXT,
    config_snapshot_json TEXT   NOT NULL DEFAULT '{}',
    config_hash         TEXT    NOT NULL DEFAULT '',
    annotation          TEXT    NOT NULL DEFAULT '',
    status              TEXT    NOT NULL DEFAULT 'running',
    -- Which channels the graceful-recovery path abandoned, as a JSON array.
    -- Declared here as well as in `_migrate_experiment_skipped_channels`, with
    -- that migration's declaration verbatim, so a fresh install and an upgraded
    -- one hold the same table (see the `fit_results` block for the same pairing).
    --
    -- NO DEFAULT, deliberately, and the three states are the whole point:
    -- NULL means *never recorded* (the run predates this column, or its host
    -- does not track skips), '[]' means *recorded as none skipped*, and a
    -- populated array means *these wells are not real*. A DEFAULT '[]' would
    -- stamp "nothing was skipped" onto every historical row, which is the exact
    -- false claim this column exists to stop the store making.
    skipped_channels    TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_started_at ON experiments(started_at);
CREATE INDEX IF NOT EXISTS idx_experiments_campaign   ON experiments(campaign);
CREATE INDEX IF NOT EXISTS idx_experiments_quality    ON experiments(quality);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    channel             INTEGER NOT NULL,
    electrode_x_mm      REAL,
    electrode_y_mm      REAL,
    timestamp           TEXT    NOT NULL,
    npts                INTEGER,
    f_min_hz            REAL,
    f_max_hz            REAL,
    measurement_time_s  REAL,
    eis_file_path       TEXT,
    eis_params_json     TEXT    NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_measurements_run_id    ON measurements(run_id);
CREATE INDEX IF NOT EXISTS idx_measurements_channel   ON measurements(channel);
CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp);

CREATE TABLE IF NOT EXISTS conditions (
    condition_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id      INTEGER NOT NULL REFERENCES measurements(measurement_id) ON DELETE CASCADE,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    stage               TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    stage_temp_sp_C     REAL,
    chamber_air_C       REAL,
    stage_temp_pv_C     REAL,
    rh_sp_pct           REAL,
    rh_pv_pct           REAL,
    -- The temperature this row means, and which thermometer said so: what
    -- `softae.analysis.conditions.resolve_temperature_C` returns for the three
    -- columns above, written once at record time (schema epoch 4). NULL
    -- `temperature_C` means no thermometer spoke; `temperature_source` is then
    -- 'unavailable' and is never NULL on a row any writer since epoch 4 wrote.
    --
    -- The stored pair is a RECORD, NOT A VIEW: it is what `resolve_temperature_C`
    -- concluded from this row's source columns *at write time*. Editing
    -- `TEMPERATURE_SOURCES` changes what future writes conclude and does not move a
    -- single stored row. `softae.analysis.conditions` remains the authority for
    -- re-analysis -- a consumer that wants today's precedence over yesterday's rows
    -- must call the resolver on the source columns, which are all still here.
    temperature_C       REAL,
    temperature_source  TEXT,
    notes               TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_conditions_measurement_id ON conditions(measurement_id);
CREATE INDEX IF NOT EXISTS idx_conditions_run_id         ON conditions(run_id);
CREATE INDEX IF NOT EXISTS idx_conditions_stage          ON conditions(stage);
-- NO temperature index is declared here — neither on the source columns nor on
-- the derived `temperature_C` this DDL does declare. This script runs before the
-- migrations, so on a legacy database `conditions` still carries whichever
-- columns it was created with (the CREATE above is a no-op there), and indexing
-- a column this DDL has not been able to add yet fails the whole open.
-- `_migrate_conditions_temp_names` creates `idx_conditions_stage_temp_pv` once
-- the column names are settled; `_migrate_conditions_resolved_temperature`
-- creates `idx_conditions_temperature_C` once the derived column exists
-- everywhere. Migrations own every conditions temperature index, on both paths.

CREATE TABLE IF NOT EXISTS fit_results (
    fit_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id      INTEGER NOT NULL REFERENCES measurements(measurement_id) ON DELETE CASCADE,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
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
    fitted_at           TEXT    NOT NULL,
    -- The gated engine's provenance columns (E0/E1). Declared here as well as in
    -- `_migrate_fit_gate_columns`, in that migration's order and with its
    -- declarations verbatim, so a fresh install and an upgraded one hold the same
    -- table rather than two shapes that only a `SELECT *` reader can survive. The
    -- pair is idempotent by construction: on a fresh database the CREATE supplies
    -- these and every `if name not in cols` is false; on a legacy one the CREATE is
    -- a no-op and the ALTERs supply them.
    engine              TEXT    NOT NULL DEFAULT 'legacy',
    gate_verdict        TEXT,
    gate_log_json       TEXT    NOT NULL DEFAULT '[]',
    n_points_used       INTEGER,
    n_points_dropped    INTEGER,
    report_mode         TEXT    NOT NULL DEFAULT 'split',
    R_sum_ohm           REAL,
    R_sum_se_ohm        REAL,
    rho_series_bulk     REAL,
    sigma_is_bound      INTEGER NOT NULL DEFAULT 0,
    sigma_rel_unc       REAL,
    phase_headroom      REAL,
    model_free_R_ohm    REAL,
    K_per_cm            REAL,
    K_route             TEXT,
    dead_height_cm      REAL    NOT NULL DEFAULT 0.0,
    thickness_method    TEXT,
    thickness_unc_cm    REAL,
    -- Arc closure (T7.7): did the impedance semicircle peak inside the swept
    -- window, or is R1 reached by extrapolating off the high-frequency side?
    -- NOT ONE of these carries a NOT NULL DEFAULT, and that is binding rather than
    -- stylistic. `arc.UNKNOWN` means *the annotator looked and could not tell* --
    -- an answer, and one that carries a reason. A DEFAULT 'unknown' would stamp
    -- that answer onto every historical row nobody ever inspected. NULL here means
    -- never annotated, and on a row written before this column it is the only
    -- honest statement available.
    arc_state           TEXT,
    arc_f_peak_hz       REAL,
    arc_f_low_hz        REAL,
    arc_phase_low_deg   REAL
);

CREATE INDEX IF NOT EXISTS idx_fit_results_measurement_id ON fit_results(measurement_id);
CREATE INDEX IF NOT EXISTS idx_fit_results_run_id         ON fit_results(run_id);
CREATE INDEX IF NOT EXISTS idx_fit_results_model_name     ON fit_results(model_name);
-- NO `arc_state` index is declared here, for the reason the conditions block
-- above spells out: this script runs before the migrations, so on a legacy
-- database `fit_results` is still whatever it was created with (the CREATE above
-- is a no-op there), and `CREATE INDEX ... ON fit_results(arc_state)` would raise
-- `no such column` inside `executescript` and fail the open of every existing
-- project. `_migrate_fit_gate_columns` owns that index, on both paths.

CREATE TABLE IF NOT EXISTS formulations (
    formulation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    channel             INTEGER NOT NULL,
    pump0_uL            REAL    NOT NULL DEFAULT 0.0,
    pump1_uL            REAL    NOT NULL DEFAULT 0.0,
    pump2_uL            REAL    NOT NULL DEFAULT 0.0,
    total_uL            REAL    NOT NULL DEFAULT 0.0,
    solution_name       TEXT,
    dep_fraction        REAL,
    dispense_rate_uL_min REAL,
    predicted_thickness_um REAL,
    deposit_area_mm2    REAL,
    thickness_method    TEXT,
    sample_uuid         TEXT,
    notes               TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_formulations_run_id  ON formulations(run_id);
CREATE INDEX IF NOT EXISTS idx_formulations_channel ON formulations(channel);

CREATE TABLE IF NOT EXISTS doe_parameters (
    doe_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    channel             INTEGER NOT NULL,
    iteration           INTEGER NOT NULL DEFAULT 0,
    parameters_json     TEXT    NOT NULL DEFAULT '{}',
    objective_value     REAL,
    acquisition_fn      TEXT,
    outcome             TEXT,
    failure_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_doe_run_id ON doe_parameters(run_id);

CREATE TABLE IF NOT EXISTS arrhenius_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT    NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
    channel             INTEGER NOT NULL,
    model               TEXT    NOT NULL DEFAULT 'arrhenius',
    Ea_eV               REAL,
    Ea_kJ_per_mol       REAL,
    ln_A                REAL,
    A                   REAL,
    B                   REAL,
    T0_K                REAL,
    T0_C                REAL,
    R_squared           REAL,
    T_min_C             REAL,
    T_max_C             REAL,
    n_points            INTEGER,
    fit_success         INTEGER NOT NULL DEFAULT 0,
    error_msg           TEXT    NOT NULL DEFAULT '',
    conductivities_json TEXT    NOT NULL DEFAULT '[]',
    temperatures_json   TEXT    NOT NULL DEFAULT '[]',
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_arrhenius_run_id  ON arrhenius_results(run_id);
CREATE INDEX IF NOT EXISTS idx_arrhenius_channel ON arrhenius_results(channel);

-- Persistent electrode-well occupancy (drop-cast wells are single-use). Keyed
-- by a monotonic per-project board_id + electrode, so it survives GUI restarts:
-- a resumed campaign can detect that a well was already cast and warn before
-- re-casting. A physical board replacement advances board_id (fresh + empty).
CREATE TABLE IF NOT EXISTS electrode_occupancy (
    board_id    INTEGER NOT NULL,
    electrode   INTEGER NOT NULL,
    run_id      TEXT,
    iteration   INTEGER,
    cast_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    sample_uuid TEXT,
    PRIMARY KEY (board_id, electrode)
);
CREATE INDEX IF NOT EXISTS idx_occupancy_board ON electrode_occupancy(board_id);

-- The *active* board pointer, persisted independently of occupancy rows. A
-- confirmed board replacement advances it immediately, so a swap survives even
-- when no cast lands on the fresh plate before shutdown. Without this the id
-- could only be inferred as MAX(board_id) over occupancy, which silently
-- "forgets" a swap and makes a resumed campaign skip physically-free wells.
-- Single row (id = 1); absent in pre-existing projects, hence the MAX fallback
-- in current_board_id().
CREATE TABLE IF NOT EXISTS board_state (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    board_id   INTEGER NOT NULL,
    updated_at TEXT    NOT NULL
);

-- Small durable rig-level scalars that are properties of the *bench*, not of any
-- run: waste accumulated since the container was emptied, spare plates on hand.
-- Both outlive sessions and both cap how long a campaign can go unattended.
CREATE TABLE IF NOT EXISTS rig_state (
    key        TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    updated_at TEXT NOT NULL
);

-- Campaign resume checkpoint (P3.2). One row per campaign, replaced after every
-- completed iteration, so the newest row is always the resume point and the
-- table cannot grow without bound. Written AFTER the observation is told to the
-- optimizer: a crash between casting a well and checkpointing then costs one
-- data point, whereas the reverse ordering would claim an observation for a well
-- that was never actually cast. Losing data is recoverable; fabricating it is not.
-- `rh_ceiling_streak` and `consecutive_failures` are the per-campaign escalation
-- counters that must survive a restart: neither parks anything until it reaches
-- its limit, so a restart can land mid-streak for reasons unrelated to it, and
-- zeroing there would hand a chronic fault a fresh allowance of trials. They are
-- separate columns rather than one `{reason: streak}` blob because each is read
-- by exactly one owner at a different layer, and a blob would need parsing before
-- either could be queried. Legacy stores gain both through
-- `_migrate_campaign_checkpoint_counters` — this CREATE is a no-op on them, so
-- the DDL alone would be correct on a fresh store and silently broken on every
-- store that already exists.
CREATE TABLE IF NOT EXISTS campaign_checkpoints (
    campaign             TEXT PRIMARY KEY,
    run_id               TEXT,
    iteration            INTEGER NOT NULL,
    loop_state           TEXT,
    board_id             INTEGER,
    spec_json            TEXT,
    optimizer_json       TEXT,
    rh_ceiling_streak    INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT    NOT NULL
);

-- Durable operator-facing alerts (a parked campaign, a depleted reservoir, a
-- gate timeout). An unattended run's event stream dies with its process/GUI, so
-- without this the *reason* a campaign stopped overnight is unrecoverable in the
-- morning. Queryable so a future notifier can drain unsent alerts.
CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    raised_at  TEXT    NOT NULL,
    run_id     TEXT,
    kind       TEXT    NOT NULL,
    severity   TEXT    NOT NULL,
    message    TEXT    NOT NULL,
    details    TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id);

-- Remaining stock per pump. `res_vol` was only ever a per-call argument that was
-- never decremented, so nothing knew how much was actually left; running dry
-- drives the plunger into a mechanical stop, so this is a safety interlock and
-- must survive restarts like any other physical state.
CREATE TABLE IF NOT EXISTS reservoir_levels (
    pump_id      INTEGER PRIMARY KEY,
    remaining_uL REAL    NOT NULL,
    updated_at   TEXT    NOT NULL
);

-- Epoch ledger (Tier 2 component 3). Deliberately NOT a single version integer:
-- the event that forced this table into existence was not a schema change at all.
-- On 2026-08-07 `deposit_area_mm2` changed *derivation*, so stored thickness
-- values changed meaning by a factor of 4.676 while the column, its units and its
-- type all stayed identical. A version number records that the schema moved; only
-- a ledger row can record that the DATA did, and say what a reader must do about
-- it. Hence `kind`: 'schema' rows describe shape, 'data-epoch' rows describe
-- meaning, and a reader comparing rows across an epoch boundary needs the latter.
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT    NOT NULL,
    kind       TEXT    NOT NULL CHECK (kind IN ('schema', 'data-epoch')),
    note       TEXT    NOT NULL DEFAULT ''
);
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _as_channel(value: Any) -> Any:
    """Normalise a channel identifier to ``int`` where it truly is one.

    Workflow steps carry the channel as a *tag*, so it arrives as ``"7"``. A
    reader asking "which wells are real?" wants ``7``. Anything not purely
    numeric is passed through as a string rather than coerced or dropped — the
    column records what was skipped, and a name it does not recognise is still
    the honest answer.
    """
    text = str(value)
    return int(text) if text.isdigit() else text


def _json_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None``, in place of NaN.

    ``arc.py``'s boundary rule, applied to every JSON column this module writes:
    *JSON has no NaN, and a NaN read back as a number is worse than a null.*

    **The blast radius is the table, not the row.** ``json.dumps`` emits the bare
    token ``NaN`` (and ``Infinity``) for a non-finite float. Python's own loader
    accepts both; SQLite's JSON1 accepts neither, and it rejects the *whole
    document* rather than the offending key. Because the predicate is evaluated
    per row, **one** NaN-bearing row anywhere in ``measurements`` makes every
    ``json_extract`` query over the table raise ``malformed JSON`` — every row,
    every key, for every reader. That is reachable in ordinary operation:
    ``arc_closure`` returns NaN for an apex it did not find, which is the common
    case on an open-arc sweep, and the scout stamps it into ``eis_params``.

    Scrubbing here rather than at each stamper is deliberate. This is the single
    write boundary — there is exactly one INSERT into ``eis_params_json`` and no
    UPDATE — so no upstream module can defeat it by forgetting.

    ``None`` and not the string ``"NaN"``: a reader that gets a string back has to
    know to parse it, and one that does not silently treats "no apex" as a value.
    ``None`` and not omission: for keys whose absence means "this run predates the
    field", a missing key and a null are different facts.
    """
    if isinstance(obj, np.ndarray):
        return _json_finite(obj.tolist())
    if isinstance(obj, dict):
        return {k: _json_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if value == value and np.isfinite(value) else None
    return obj


def _safe_json(obj: Any) -> str:
    """Serialise *obj* to JSON, handling numpy arrays and other types.

    Non-finite floats become JSON ``null`` at this boundary — see
    :func:`_json_finite` for why one un-scrubbed row would cost the whole table.
    """
    if isinstance(obj, np.ndarray):
        return json.dumps(_json_finite(obj.tolist()))
    return json.dumps(_json_finite(obj), default=str)


def _f_or_none(value: Any) -> float | None:
    """A finite float, or ``None`` — so NaN never reaches a REAL column.

    SQLite stores NaN as NULL on the way in but not reliably on the way out through
    every driver, and a NaN that survives reads as a number to anything that only
    checks for ``None``. Normalising here keeps "absent" a single state.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _fit_report_columns(report: Any | None) -> dict[str, Any]:
    """Gate/covariance columns for one ``fit_results`` row.

    Returns the legacy defaults when *report* is ``None``, so a row written by the
    unchanged path means precisely what it has always meant.
    """
    defaults: dict[str, Any] = {
        "engine": "legacy",
        "gate_verdict": None,
        "gate_log_json": "[]",
        "n_points_used": None,
        "n_points_dropped": None,
        "report_mode": "split",
        "R_sum_ohm": None,
        "R_sum_se_ohm": None,
        "rho_series_bulk": None,
        "sigma_is_bound": 0,
        "sigma_rel_unc": None,
        "phase_headroom": None,
        "model_free_R_ohm": None,
        "K_per_cm": None,
        "K_route": None,
        "dead_height_cm": 0.0,
        "thickness_method": None,
        "thickness_unc_cm": None,
    }
    if report is None:
        return defaults

    sigma = getattr(report, "sigma", None)
    cell = getattr(report, "cell", None)
    quality = getattr(report, "quality", None)
    verdict = getattr(quality, "verdict", None)
    mask = getattr(report, "mask", None)

    n_used = int(np.asarray(mask, dtype=bool).sum()) if mask is not None else None
    n_dropped = getattr(report, "n_dropped", None)

    defaults.update(
        engine=str(getattr(report, "engine", "legacy")),
        gate_verdict=getattr(verdict, "value", None) or (
            str(verdict) if verdict is not None else None),
        gate_log_json=_safe_json(list(getattr(report, "gate_log", ()) or [])),
        n_points_used=n_used,
        n_points_dropped=int(n_dropped) if n_dropped is not None else None,
        report_mode=str(getattr(sigma, "mode", "split")),
        R_sum_ohm=(
            _f_or_none(getattr(sigma, "R_reported_ohm", None))
            if getattr(sigma, "R_basis", "") == "sum" else None
        ),
        R_sum_se_ohm=_f_or_none(getattr(sigma, "R_reported_se_ohm", None)),
        rho_series_bulk=_f_or_none(getattr(sigma, "rho", None)),
        sigma_is_bound=1 if getattr(sigma, "is_bound", False) else 0,
        sigma_rel_unc=_f_or_none(getattr(sigma, "rel_uncertainty", None)),
        phase_headroom=_f_or_none(getattr(sigma, "phase_headroom", None)),
        model_free_R_ohm=_f_or_none(getattr(sigma, "model_free_R_ohm", None)),
        K_per_cm=_f_or_none(getattr(sigma, "K_per_cm", None)),
        K_route=getattr(sigma, "K_route", None),
        dead_height_cm=float(getattr(cell, "dead_height_cm", 0.0) or 0.0),
        thickness_method=getattr(cell, "thickness_method", None),
        thickness_unc_cm=_f_or_none(getattr(cell, "thickness_unc_cm", None)),
    )
    return defaults


def _arc_columns(fit_result: Any) -> dict[str, Any]:
    """The arc-closure columns for one row — read off the FIT, not the report.

    **This deviates from :func:`_fit_report_columns`' report-only convention, and
    the deviation is the point.** A real
    :class:`~softae.analysis.eis.report.SpectrumReport` carries ``run_gates``' log,
    and that log has **no** ``arc_closure`` entry:
    :func:`~softae.analysis.eis.arc.annotate_arc_closure` writes to the *fit*, and
    nothing copies the record into a ``gate_log`` — the shim that once did was
    retired the moment these columns existed. A report-scanning implementation
    would therefore find nothing to read — a column that empties itself is worse
    than no column at all.

    Reading the fit also reaches further: ``annotate_arc_closure`` runs on every
    ``analyze_spectrum`` call on both engines, so ``fit.arc_closure`` is present for
    every fit that came through analysis, including the ``report=None`` callers —
    which, since the shim went, is all of them.

    All four are ``None`` when nothing annotated the fit, which is a different fact
    from ``'unknown'`` — see the DDL. The three REALs go through :func:`_f_or_none`,
    the file's NaN→NULL boundary, so an ``unknown`` state and a phase-less spectrum
    both arrive as NaN and both land as NULL with no special-casing here.
    """
    arc = getattr(fit_result, "arc_closure", None)
    if arc is None:
        return {"arc_state": None, "arc_f_peak_hz": None,
                "arc_f_low_hz": None, "arc_phase_low_deg": None}
    return {"arc_state": str(arc.state),
            "arc_f_peak_hz": _f_or_none(arc.f_peak_hz),
            "arc_f_low_hz": _f_or_none(arc.f_low_hz),
            "arc_phase_low_deg": _f_or_none(arc.phase_low_deg)}


def _engine_label(engine: str, fit_result: Any) -> str:
    """Name the estimator that produced this row's ``R1``, not just the engine.

    Schema epoch 5. ``R1`` stays a REAL in ohms whichever route produced it, so
    nothing about the row's *shape* says whether it came from the CPE circuit fit or
    from the two-point Debye read — and [a53] records that the failure mode on this
    rig is precisely a biased ``R1`` flowing through **unlabelled**. A date in the
    epoch ledger cannot answer it either: the flag is per-configuration, so a database
    can hold both kinds written the same afternoon.

    ``engine`` is refined rather than joined by a second column, because that column
    already answers *what produced this number* — ``'legacy'``, ``'gated'``, and now
    ``'gated_two_point'`` — and a parallel ``estimator`` column would be a second
    spelling of one fact, which is how two columns start disagreeing.

    Read off the **fit**, the same deviation :func:`_arc_columns` makes and for the
    same reason: a fit reaches this table from callers that pass no report at all, and
    a label that only survives on reported rows is a label that goes missing exactly
    where the epoch matters. Anything unlabelled is left exactly as it was.
    """
    from softae.analysis.eis.engine import TWO_POINT

    if getattr(fit_result, "estimator", None) != TWO_POINT:
        return engine
    return f"{engine}_two_point" if engine != "legacy" else "two_point"


@dataclass(frozen=True)
class PredictedThicknessRecord:
    """A recorded thickness **and the basis it was computed on** (P.11).

    What :meth:`DataStore.predicted_thickness_record` returns. A thickness is
    ``final_volume_uL / area_mm2 * 1000``, so the quotient alone does not say what
    it means: the same column holds rows divided by 4.0 mm² and by 18.704 mm².

    ``area_mm2 is None`` means the denominator was **never recorded** — not that it
    was wrong, and emphatically not that it was 4.0. Such a thickness has an unknown
    basis and must read as unavailable to anything that divides by it. It is never
    rescaled: we cannot tell whether the row predates the 2026-08-07 area correction
    or ran on a board the correction never touched, and a rescale would manufacture
    a number for the rows that never needed one.

    ``method`` is the row's :data:`FORMULATION_THICKNESS_METHODS` tag, or ``None``
    when the row predates the column.
    """

    um: float
    area_mm2: float | None
    method: str | None


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------


class DataStore:
    """Project-scoped SQLite data store.

    Parameters
    ----------
    project_dir : str or Path
        Root of the project data directory.  Created if it does not
        exist, along with ``db/``, ``runs/``, and ``formulations/``
        subdirectories.
    db_filename : str
        Name of the SQLite file inside ``project_dir/db/``.
    """

    def __init__(
        self,
        project_dir: str | Path,
        db_filename: str = "softae.db",
    ) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()

        # Ensure canonical subdirectories exist.
        (self.project_dir / "db").mkdir(parents=True, exist_ok=True)
        (self.project_dir / "runs").mkdir(exist_ok=True)
        (self.project_dir / "formulations").mkdir(exist_ok=True)

        db_path = self.project_dir / "db" / db_filename
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row

        # Create tables (idempotent).
        self._conn.executescript(_DDL)
        self._conn.commit()

        # Migrate existing databases: add config_hash column if missing.
        self._migrate_config_hash()
        # Migrate existing databases: add VFT/model columns to the thermal-fit table.
        self._migrate_thermal_columns()
        # Migrate existing databases: add annotation column if missing.
        self._migrate_annotation()
        # Migrate existing databases: record which channels a run abandoned.
        self._migrate_experiment_skipped_channels()
        # Migrate existing databases: add pump2_uL to formulations if missing.
        self._migrate_formulation_pump2()
        # Migrate existing databases: add stage_temp_pv_C to conditions if missing.
        self._migrate_conditions_stage_temp()
        # ...then rename the other two temperature columns after their instruments.
        # Must follow the line above: it can only rename a table that already has
        # all three columns, and it owns the temperature index for every database.
        self._migrate_conditions_temp_names()
        # ...and only then derive the resolved temperature from them: this one
        # reads all three source columns by their settled names, so it cannot run
        # before the rename above has given them those names.
        self._migrate_conditions_resolved_temperature()
        self._migrate_formulation_thickness()
        # Migrate existing databases: record the area a thickness was divided by.
        self._migrate_formulation_area()
        self._migrate_measurement_role()
        self._migrate_eis_calibrations()
        self._migrate_fixture_corrections()
        self._migrate_thickness()
        self._migrate_fit_gate_columns()
        # Tier 2 component 3: the modality/payload contract on `measurements`.
        self._migrate_modality()
        # Tier 2 component 6: the sample-identity spine's other two anchors.
        self._migrate_formulation_sample_uuid()
        self._migrate_doe_outcome()
        # The per-campaign escalation counters, which the resume path reads by name.
        self._migrate_campaign_checkpoint_counters()
        # LAST, always: the ledger records the epochs every migration above has
        # just finished establishing, so it must not claim them before they hold.
        self._migrate_schema_version()

        logger.info(
            "data_store_opened",
            project_dir=str(self.project_dir),
            db=str(db_path),
        )

    # ── Run lifecycle ───────────────────────────────────────────────────

    def _run_id_taken(self, run_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM experiments WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone() is not None

    def start_run(
        self,
        workflow_name: str,
        config_snapshot: str = "{}",
        *,
        mode: str = "full",
        pcb_name: str | None = None,
        eis_preset: str | None = None,
        campaign: str = "dev",
        quality: str = "explore",
        config_hash: str = "",
        annotation: str = "",
    ) -> str:
        """Begin a new experiment run.

        Returns the generated ``run_id`` string, which also serves as
        the filesystem directory name under ``runs/``.
        """
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = workflow_name.replace(" ", "_")
        run_id = f"{ts}_{safe_name}"

        # The stamp has one-second resolution, which is ample for a rig run and not
        # ample for a batch of imports: seven `softae-commission import` calls in a
        # loop collide and all but the first raise
        # `UNIQUE constraint failed: experiments.run_id`, mid-batch, having already
        # written their EIS file to disk. Suffix rather than widen the stamp, so every
        # historical run_id keeps its exact spelling and the format stays sortable.
        if self._run_id_taken(run_id):
            base = run_id
            for n in range(2, 1000):
                run_id = f"{base}_{n}"
                if not self._run_id_taken(run_id):
                    break
            else:
                raise RuntimeError(
                    f"could not allocate a run_id from {base} after 999 attempts")
            logger.debug("run_id_deduplicated", base=base, run_id=run_id)

        # Create run directories.
        run_dir = self.project_dir / "runs" / run_id
        (run_dir / "eis").mkdir(parents=True, exist_ok=True)
        (run_dir / "images").mkdir(exist_ok=True)

        # Persist config snapshot to disk.
        (run_dir / "config_snapshot.toml").write_text(
            config_snapshot, encoding="utf-8"
        )

        # Write annotation as a plain-text notes file for quick identification.
        if annotation:
            notes_path = run_dir / f"{run_id}_notes.txt"
            notes_path.write_text(annotation, encoding="utf-8")

        self._conn.execute(
            """INSERT INTO experiments
               (run_id, started_at, workflow_name, workflow_mode,
                campaign, quality, pcb_name, eis_preset,
                config_snapshot_json, config_hash, annotation, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
            (
                run_id,
                _now_iso(),
                workflow_name,
                mode,
                campaign,
                quality,
                pcb_name,
                eis_preset,
                config_snapshot,
                config_hash,
                annotation,
            ),
        )
        self._conn.commit()

        logger.info("run_started", run_id=run_id)
        return run_id

    def finish_run(
        self,
        run_id: str,
        status: str = "done",
        *,
        skipped_channels: Sequence[Any] | None = None,
    ) -> None:
        """Mark *run_id* as finished with the given *status*.

        ``skipped_channels`` is three-valued, and the caller decides which of the
        three it is entitled to:

        ``None`` (default)
            Leave the column alone. A caller that does not track skips must not
            claim there were none — every existing caller lands here.
        ``[]``
            Recorded as none skipped. Only a caller that *counted* may say this.
        ``[3, 7]``
            These channels were abandoned; the wells are not real.

        The status vocabulary in use is ``running`` / ``done`` / ``partial`` /
        ``aborted`` / ``error`` / ``interrupted`` (plus the campaign loop's
        ``converged`` / ``stopped``). ``partial`` is what a plate that finished
        every step it attempted but abandoned channels along the way is entitled
        to: ``done`` overstates it and ``error`` understates it, and the run
        genuinely produced usable wells.
        """
        if skipped_channels is None:
            self._conn.execute(
                "UPDATE experiments SET finished_at = ?, status = ? WHERE run_id = ?",
                (_now_iso(), status, run_id),
            )
        else:
            self._conn.execute(
                "UPDATE experiments SET finished_at = ?, status = ?, "
                "skipped_channels = ? WHERE run_id = ?",
                (_now_iso(), status,
                 json.dumps([_as_channel(c) for c in skipped_channels]), run_id),
            )
        self._conn.commit()
        logger.info("run_finished", run_id=run_id, status=status,
                    skipped_channels=skipped_channels)

    def run_skipped_channels(self, run_id: str) -> list[Any] | None:
        """Channels abandoned during *run_id* — ``None`` when never recorded.

        The read side of :meth:`finish_run`'s three states, kept here rather than
        left to every caller's own ``json.loads``: an empty list and a NULL mean
        different things, and a decoder written per-caller is where that
        distinction gets flattened.
        """
        row = self._conn.execute(
            "SELECT skipped_channels FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])

    def run_outcome(self, run_id: str) -> dict[str, Any] | None:
        """How *run_id* ended — ``None`` when there is no such row.

        Returns ``{"status": str, "finished": bool}``. Both halves are needed and
        neither substitutes for the other:

        ``status``
            The vocabulary :meth:`finish_run` documents. It says which *exit
            path* closed the row.
        ``finished``
            ``finished_at IS NOT NULL`` — whether anything closed the row **at
            all**. A hard kill leaves ``status`` at its ``'running'`` default
            with ``finished_at`` NULL, and the next launch's recovery sweep
            (:func:`softae.core.shutdown.record_unclean_shutdown`) only later
            rewrites it to ``'interrupted'``. A reader that asked for the status
            alone, before that sweep ran, would see ``'running'`` and have no way
            to tell "died mid-run" from "running right now".

        Read-only, and the read side of :meth:`finish_run` in the same way
        :meth:`run_skipped_channels` is: the ``finished_at``-NULL convention is
        stated once here rather than re-derived by every caller that needs it.
        """
        row = self._conn.execute(
            "SELECT status, finished_at FROM experiments WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {"status": row[0], "finished": row[1] is not None}

    # ── Path helpers ────────────────────────────────────────────────────

    def eis_dir(self, run_id: str) -> Path:
        """Return the EIS data directory for a run."""
        return self.project_dir / "runs" / run_id / "eis"

    def run_dir(self, run_id: str) -> Path:
        """Return the top-level directory for a run."""
        return self.project_dir / "runs" / run_id

    def payload_dir(self, run_id: str, modality: str) -> Path:
        """Directory for a modality's scientific payloads (spec §7a).

        ``runs/<run_id>/data/<modality>/`` — partitioned by modality rather than
        pooled, so a new modality adds a directory instead of colliding with EIS
        filenames, and so a reader can enumerate one modality without opening
        files to discover their kind. Deliberately a sibling of ``eis/`` rather
        than inside it: the ``.txt`` tree is transitional and will retire, and
        nesting the durable format under the doomed one would make that removal a
        migration.
        """
        return self.run_dir(run_id) / "data" / str(modality or "eis")

    # ── Measurement recording ───────────────────────────────────────────

    def record_measurement(
        self,
        run_id: str,
        eis_result: EISResult,
        *,
        electrode_x_mm: float | None = None,
        electrode_y_mm: float | None = None,
        role: str = "sample",
        fixture_id: str | None = None,
        nominal_value: float | None = None,
        electrode_mode: str = "unknown",
        thermal_history: str = "",
        sweep_order: int | None = None,
        re_connection: str = "unverified",
        re_contact_verified: bool = False,
        modality: str = "eis",
        payload_path: str | None = None,
        payload_format: str | None = None,
        sample_uuid: str | None = None,
    ) -> int:
        """Persist one EIS measurement row.

        The caller must save the EIS file to disk (via
        ``eis_result.save()``) **before** calling this method.
        Returns the new ``measurement_id``.

        *role* is what makes commissioning cost almost no new code (E2): a blank is
        an EIS spectrum with the same columns, the same file format and the same
        conditions row as a sample, so it takes this identical path and only the tag
        differs. ``sample`` is the default, so every existing caller is unchanged and
        every pre-existing row is already correct.

        *fixture_id* ties a commissioning artifact to the hardware it was taken on —
        a short blank measured before a board swap must not be silently applied after
        one (framework §8.5).

        *nominal_value* is the reference part's **marked** value. It is recorded here
        rather than asked for again at derivation time because it is not recoverable
        later: nobody remembers which resistor was in the socket weeks ago, and the
        marking disagreeing with the measurement is the check that catches an
        unusable part (overhaul §3.7).

        The last four are overhaul §6's remaining mandatory-at-ingest fields. All
        default to *unrecorded* rather than to a plausible value, for the same reason
        ``electrode_mode`` defaults to ``'unknown'``: a default that looks like an
        answer destroys the distinction the column exists to make. ``sweep_order`` in
        particular is ``None``, not ``0`` — zero would make every legacy row claim to
        be the first measurement of its sweep, which is exactly the drift signal it is
        meant to support.

        The last four are the Tier-2 modality contract (see :meth:`_migrate_modality`
        for why ``modality`` alone carries a non-absent default). ``payload_path`` and
        ``payload_format`` stay ``None`` here on every current call: the payload is
        written *after* this row exists, so that the file can name the row it belongs
        to, and :meth:`set_measurement_payload` attaches it. ``sample_uuid`` is
        accepted now and minted in T2.6.
        """
        role = str(role or "sample")
        if role not in MEASUREMENT_ROLES:
            logger.warning(
                "measurement_role_unknown", role=role, known=MEASUREMENT_ROLES,
                msg="recording as 'sample' — an unknown role must not silently "
                    "become a calibration artifact",
            )
            role = "sample"
        # Make file path relative to project_dir for portability.
        rel_path: str | None = None
        if eis_result.raw_file_path:
            try:
                rel_path = str(
                    Path(eis_result.raw_file_path).relative_to(self.project_dir)
                )
            except ValueError:
                rel_path = eis_result.raw_file_path

        freq = eis_result.frequency
        cur = self._conn.execute(
            """INSERT INTO measurements
               (run_id, channel, electrode_x_mm, electrode_y_mm,
                timestamp, npts, f_min_hz, f_max_hz,
                measurement_time_s, eis_file_path, eis_params_json,
                role, fixture_id, nominal_value, electrode_mode,
                thermal_history, sweep_order, re_connection, re_contact_verified,
                modality, payload_path, payload_format, sample_uuid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?)""",
            (
                run_id,
                eis_result.channel,
                electrode_x_mm,
                electrode_y_mm,
                eis_result.timestamp.isoformat()
                if isinstance(eis_result.timestamp, datetime)
                else str(eis_result.timestamp),
                eis_result.npts,
                float(freq.min()) if len(freq) else None,
                float(freq.max()) if len(freq) else None,
                eis_result.measurement_time_s,
                rel_path,
                _safe_json(eis_result.eis_params),
                role,
                fixture_id,
                float(nominal_value) if nominal_value is not None else None,
                str(electrode_mode or "unknown"),
                str(thermal_history or ""),
                int(sweep_order) if sweep_order is not None else None,
                str(re_connection or "unverified"),
                1 if re_contact_verified else 0,
                str(modality or "eis"),
                # Normalised to None, never '': an empty string would be a path
                # that has been recorded, and every reader that only checks for
                # None would then try to open it.
                payload_path or None,
                payload_format or None,
                sample_uuid or None,
            ),
        )
        self._conn.commit()
        mid = cur.lastrowid
        logger.debug("measurement_recorded", measurement_id=mid,
                     channel=eis_result.channel, role=role,
                     electrode_mode=electrode_mode, modality=modality)
        return mid  # type: ignore[return-value]

    def set_measurement_payload(
        self,
        measurement_id: int,
        payload_path: str | Path | None,
        payload_format: str | None = "netcdf4",
    ) -> None:
        """Attach a written payload file to an existing measurement row.

        Separate from :meth:`record_measurement` because of an ordering the
        payload itself imposes: the file's ``attrs`` carry ``measurement_id`` so a
        file on disk points back at its row, which means the row must exist first.
        The alternative — writing the file before the INSERT — costs either that
        backlink or a second write of the whole payload.

        The consequence is that the columns are only ever set for a file that is
        already on disk. A failed write simply never calls this, and the row keeps
        its NULLs, which is the honest record: **there is no payload.** A path
        written ahead of the file would be a row asserting a file that does not
        exist.

        *payload_path* is stored **relative to** ``project_dir`` when it lies
        inside it, matching ``eis_file_path`` — a project directory must survive
        being moved or copied to another machine.
        """
        rel: str | None = None
        if payload_path:
            rel = str(payload_path)
            try:
                rel = str(Path(payload_path).relative_to(self.project_dir))
            except ValueError:
                pass

        self._conn.execute(
            "UPDATE measurements SET payload_path = ?, payload_format = ? "
            "WHERE measurement_id = ?",
            (rel, (payload_format or None) if rel else None, int(measurement_id)),
        )
        self._conn.commit()
        logger.debug("measurement_payload_recorded",
                     measurement_id=measurement_id, payload_path=rel,
                     payload_format=payload_format)

    # ── Conditions recording ────────────────────────────────────────────

    def record_conditions(
        self,
        measurement_id: int,
        stage: str,
        *,
        stage_temp_sp_C: float | None = None,
        chamber_air_C: float | None = None,
        stage_temp_pv_C: float | None = None,
        rh_sp_pct: float | None = None,
        rh_pv_pct: float | None = None,
        notes: str = "",
    ) -> int:
        """Record an environmental-conditions snapshot for a measurement.

        Multiple snapshots per measurement are expected — one per
        experimental stage (``"formulation"``, ``"processing"``,
        ``"measurement"``, ``"anneal"``, etc.).

        **Three temperatures, two instruments.** Each column names the
        instrument that wrote it, because the earlier names (``temp_sp_C`` /
        ``temp_pv_C``) read as one controller's SP/PV pair and were not:

        ===================  ==================================  ================
        Column               Instrument                          What it is
        ===================  ==================================  ================
        ``stage_temp_sp_C``  temperature controller (Modbus)     stage setpoint
        ``stage_temp_pv_C``  temperature controller (Modbus)     stage PV — sample
        ``chamber_air_C``    humidity controller (I²C sensor)    chamber air
        ===================  ==================================  ================

        Only the first two are a pair. Nothing sets ``chamber_air_C``; it is air
        in the enclosure and ran up to 42 °C below the stage on run
        ``20260811T023757Z``.

        **The choice between them is made here, once** (schema epoch 4). Every
        row also stores what
        :func:`softae.analysis.conditions.resolve_temperature_C` says about it:
        ``temperature_C`` — the sample's temperature, ``NULL`` when no
        thermometer spoke — and ``temperature_source``, one of
        :data:`~softae.analysis.conditions.TEMPERATURE_SOURCES` or
        ``'unavailable'``, never ``NULL``. Consumers read those two columns
        rather than re-deriving the precedence; the resolver remains the only
        place the precedence is written down.

        Returns the new ``condition_id``.
        """
        run_id = self._run_id_for_measurement(measurement_id)
        temperature_C, temperature_source = resolve_temperature_C(
            stage_pv_C=stage_temp_pv_C,
            stage_sp_C=stage_temp_sp_C,
            chamber_air_C=chamber_air_C,
        )
        cur = self._conn.execute(
            """INSERT INTO conditions
               (measurement_id, run_id, stage, timestamp,
                stage_temp_sp_C, chamber_air_C, stage_temp_pv_C,
                temperature_C, temperature_source,
                rh_sp_pct, rh_pv_pct, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                measurement_id,
                run_id,
                stage,
                _now_iso(),
                stage_temp_sp_C,
                chamber_air_C,
                stage_temp_pv_C,
                # NaN never reaches the REAL column: 'no thermometer spoke' is
                # NULL + 'unavailable', one state with one spelling.
                _f_or_none(temperature_C),
                temperature_source,
                rh_sp_pct,
                rh_pv_pct,
                notes,
            ),
        )
        self._conn.commit()
        cid = cur.lastrowid
        logger.debug(
            "conditions_recorded",
            condition_id=cid,
            measurement_id=measurement_id,
            stage=stage,
        )
        return cid  # type: ignore[return-value]

    # ── Fit recording ───────────────────────────────────────────────────

    def record_fit(
        self,
        measurement_id: int,
        fit_result: Any,
        *,
        L_cm: float | None = None,
        t_cm: float | None = None,
        w_cm: float | None = None,
        report: Any | None = None,
    ) -> int:
        """Persist a circuit-fit result linked to a measurement.

        *fit_result* is expected to be a
        :class:`~softae.analysis.circuit_fitting.FitResult` instance.

        *report* is an optional
        :class:`~softae.analysis.eis.report.SpectrumReport` from the gated engine.
        When absent — which is every legacy-engine call — the gate columns take their
        defaults and the row means exactly what such a row has always meant.

        The four ``arc_*`` columns are the one exception to that split: they come
        from *fit_result*, never from *report*, so they populate whether or not a
        report is passed. :func:`_arc_columns` says why.

        Returns the new ``fit_id``.
        """
        sigma: float | None = None
        if (
            L_cm is not None
            and t_cm is not None
            and w_cm is not None
            and fit_result.R1
            and fit_result.R1 > 0
        ):
            # P.20 site 10. This was a *third* independent spelling of the σ formula
            # — not ``z_to_sigma``, not ``CellConstant.sigma``, living in the
            # persistence layer — so a correction to the physics could reach every
            # display and still miss the column readers actually query. The guards
            # above are unchanged; only the arithmetic moved onto the one
            # implementation intended to survive. ``K/R`` associates differently from
            # ``L/((R·t)·w)``, so a stored σ can differ from the old value in its
            # last bit and in no other way.
            sigma = CellConstant.from_legacy(L_cm, t_cm, w_cm).sigma(fit_result.R1)

        extra = _fit_report_columns(report)
        # DELIBERATE DEVIATION from the report-only convention one line above: the
        # arc columns are sourced from the *fit*, because the arc record is not in
        # any real report's gate log — `annotate_arc_closure` writes it to the fit
        # and nowhere else, so scanning `report` would silently NULL these columns.
        # See `_arc_columns`. `gate_log_json` is untouched by this and stays byte
        # for byte what `_fit_report_columns` produced; rows written since these
        # columns landed carry the literal "[]" there, and the older rows that
        # carried the record in the JSON are still read by `shadow_db`'s fallback.
        extra.update(_arc_columns(fit_result))
        extra["engine"] = _engine_label(extra["engine"], fit_result)
        # A bounded σ is not a value.  Storing it in ``sigma_S_per_cm`` would let any
        # reader that does not check ``sigma_is_bound`` treat a ceiling as a
        # measurement, so the column is cleared and the bound travels separately.
        if extra["sigma_is_bound"]:
            sigma = None

        run_id = self._run_id_for_measurement(measurement_id)
        cur = self._conn.execute(
            """INSERT INTO fit_results
               (measurement_id, run_id, model_name, R0, R1,
                sigma_S_per_cm, electrode_L_cm, electrode_t_cm, electrode_w_cm,
                success, error_msg, parameters_json, fitted_at,
                engine, gate_verdict, gate_log_json, n_points_used,
                n_points_dropped, report_mode, R_sum_ohm, R_sum_se_ohm,
                rho_series_bulk, sigma_is_bound, sigma_rel_unc, phase_headroom,
                model_free_R_ohm, K_per_cm, K_route, dead_height_cm,
                thickness_method, thickness_unc_cm,
                arc_state, arc_f_peak_hz, arc_f_low_hz, arc_phase_low_deg)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?)""",
            (
                measurement_id,
                run_id,
                fit_result.model_name,
                fit_result.R0,
                fit_result.R1,
                sigma,
                L_cm,
                t_cm,
                w_cm,
                1 if fit_result.success else 0,
                fit_result.error_msg,
                _safe_json(
                    fit_result.parameters.tolist()
                    if isinstance(fit_result.parameters, np.ndarray)
                    else fit_result.parameters
                ),
                _now_iso(),
                extra["engine"],
                extra["gate_verdict"],
                extra["gate_log_json"],
                extra["n_points_used"],
                extra["n_points_dropped"],
                extra["report_mode"],
                extra["R_sum_ohm"],
                extra["R_sum_se_ohm"],
                extra["rho_series_bulk"],
                extra["sigma_is_bound"],
                extra["sigma_rel_unc"],
                extra["phase_headroom"],
                extra["model_free_R_ohm"],
                extra["K_per_cm"],
                extra["K_route"],
                extra["dead_height_cm"],
                extra["thickness_method"],
                extra["thickness_unc_cm"],
                extra["arc_state"],
                extra["arc_f_peak_hz"],
                extra["arc_f_low_hz"],
                extra["arc_phase_low_deg"],
            ),
        )
        self._conn.commit()
        fid = cur.lastrowid
        logger.debug("fit_recorded", fit_id=fid, measurement_id=measurement_id)
        return fid  # type: ignore[return-value]

    # ── Formulation recording ───────────────────────────────────────────

    def record_formulation(
        self,
        run_id: str,
        channel: int,
        *,
        pump0_uL: float = 0.0,
        pump1_uL: float = 0.0,
        pump2_uL: float = 0.0,
        total_uL: float = 0.0,
        solution_name: str | None = None,
        dep_fraction: float | None = None,
        dispense_rate_uL_min: float | None = None,
        predicted_thickness_um: float | None = None,
        deposit_area_mm2: float | None = None,
        thickness_method: str | None = None,
        sample_uuid: str | None = None,
        notes: str = "",
    ) -> int:
        """Persist a formulation (dispense volumes) for one channel.

        ``predicted_thickness_um`` is the deposition twin's dry-film
        thickness (P7.6), or ``None`` when the twin could not speak. It is
        what lets the conductivity path use a *computed* ``t`` instead of a
        hand-typed one.

        ``deposit_area_mm2`` is the area that thickness was divided by, and
        ``thickness_method`` — one of :data:`FORMULATION_THICKNESS_METHODS` — which
        tier it came from (P.7). Both default to ``None``, so a caller with no twin
        writes ``NULL`` meaning *never recorded*, which stays distinct from an
        explicit ``'unavailable'`` meaning *recorded as absent*.

        ``sample_uuid`` (T2.6) is the identity of the physical sample this cast
        creates, minted by the caller at the moment a well is consumed. ``None``
        for every writer that has no well-consumption event to mint at — the row
        is then honestly anonymous rather than carrying an identity nothing else
        shares.
        """
        cur = self._conn.execute(
            """INSERT INTO formulations
               (run_id, channel, pump0_uL, pump1_uL, pump2_uL, total_uL,
                solution_name, dep_fraction, dispense_rate_uL_min,
                predicted_thickness_um, deposit_area_mm2, thickness_method,
                sample_uuid, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                channel,
                pump0_uL,
                pump1_uL,
                pump2_uL,
                total_uL,
                solution_name,
                dep_fraction,
                dispense_rate_uL_min,
                predicted_thickness_um,
                deposit_area_mm2,
                thickness_method,
                # Normalised to None, never '': an empty string is an identity
                # that has been recorded, and would join to every other row that
                # also failed to mint one.
                sample_uuid or None,
                notes,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def predicted_thickness_um(self, run_id: str, channel: int) -> float | None:
        """The twin's dry-film thickness for a channel's cast (µm), or ``None``.

        ``None`` means *not available* — no formulation recorded, or the twin
        could not speak for it — and must never be silently read as zero: the
        conductivity path divides by ``t``.

        The most recent row wins if a channel was cast more than once, which
        matches how re-casting a well works elsewhere in the system.
        """
        row = self._conn.execute(
            """SELECT predicted_thickness_um FROM formulations
               WHERE run_id = ? AND channel = ?
               ORDER BY formulation_id DESC LIMIT 1""",
            (str(run_id), int(channel)),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            return float(row[0])
        except (TypeError, ValueError):
            return None

    def predicted_thickness_record(
        self, run_id: str, channel: int
    ) -> PredictedThicknessRecord | None:
        """The twin's thickness **with the area it was divided by** (P.11).

        :meth:`predicted_thickness_um` returns a quotient with no denominator, and
        a caller cannot tell a row cast against 4.0 mm² from one cast against
        18.704 mm² — a factor of 4.676 in the same column with the same units.
        This returns the pair, so a consumer can decide whether the number means
        anything before dividing σ by it.

        ``None`` when no formulation row exists for the channel, or the row records
        no thickness — the same absence :meth:`predicted_thickness_um` reports. A
        row that *has* a thickness always yields a record, even when its
        ``area_mm2`` is ``None``: **"the basis was never recorded" is a fact the
        caller needs**, and collapsing it into a plain ``None`` here would hide
        which of the two absences occurred.

        The most recent row wins, matching :meth:`predicted_thickness_um`.
        """
        row = self._conn.execute(
            """SELECT predicted_thickness_um, deposit_area_mm2, thickness_method
               FROM formulations
               WHERE run_id = ? AND channel = ?
               ORDER BY formulation_id DESC LIMIT 1""",
            (str(run_id), int(channel)),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            um = float(row[0])
        except (TypeError, ValueError):
            return None
        try:
            area = None if row[1] is None else float(row[1])
        except (TypeError, ValueError):
            area = None
        method = None if row[2] is None else str(row[2])
        return PredictedThicknessRecord(um=um, area_mm2=area, method=method)

    # ── DOE parameter recording ─────────────────────────────────────────

    def record_doe_parameter(
        self,
        run_id: str,
        channel: int,
        iteration: int,
        parameters: dict[str, Any],
        *,
        objective_value: float | None = None,
        acquisition_fn: str | None = None,
    ) -> int:
        """Persist one DOE / optimizer observation row.

        Returns the new ``doe_id``.
        """
        cur = self._conn.execute(
            """INSERT INTO doe_parameters
               (run_id, channel, iteration, parameters_json,
                objective_value, acquisition_fn)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                channel,
                iteration,
                _safe_json(parameters),
                objective_value,
                acquisition_fn,
            ),
        )
        self._conn.commit()
        doe_id = cur.lastrowid
        logger.debug(
            "doe_parameter_recorded",
            doe_id=doe_id,
            run_id=run_id,
            iteration=iteration,
        )
        return doe_id  # type: ignore[return-value]

    def update_doe_objective(
        self, doe_id: int, objective_value: float
    ) -> None:
        """Set the objective value for an existing DOE row."""
        self._conn.execute(
            "UPDATE doe_parameters SET objective_value = ? WHERE doe_id = ?",
            (objective_value, doe_id),
        )
        self._conn.commit()

    def update_doe_outcome(
        self,
        *,
        outcome: str,
        run_id: str | None = None,
        channel: int | None = None,
        doe_id: int | None = None,
        failure_reason: str | None = None,
    ) -> int | None:
        """Record *why* a trial ended as it did (T3.1b). Returns the row updated.

        ``doe_id`` addresses a row exactly. Without one, the **most recent row for
        ``(run_id, channel)``** is updated — within a trial that is this trial's
        row, because the DOE row is written before the workflow runs.

        **The channel match falls back to the run's most recent row**, and that is
        not laziness: the three campaign paths do not agree on what ``channel``
        means here. The batched and placement paths tag each row with its real
        electrode, but the single-point path records ``channel=0`` — a trial-grain
        sentinel, because one suggestion cast across several replicate electrodes
        has no single channel. A strict channel match would therefore silently
        write nothing for every single-point campaign, which is the default one.
        The fallback keeps the precise case precise and the trial-grain case
        correct instead of quietly empty.

        Returns ``None`` — and writes nothing — when no row matches at all. A
        missing row means the caller is describing a trial this store never
        recorded, and inventing one would put a reason in the database with no
        experiment attached to it.
        """
        if doe_id is None:
            if run_id is None:
                return None
            row = None
            if channel is not None:
                row = self._conn.execute(
                    "SELECT doe_id FROM doe_parameters "
                    "WHERE run_id = ? AND channel = ? ORDER BY doe_id DESC LIMIT 1",
                    (run_id, int(channel)),
                ).fetchone()
            if row is None:
                row = self._conn.execute(
                    "SELECT doe_id FROM doe_parameters "
                    "WHERE run_id = ? ORDER BY doe_id DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
            if row is None:
                return None
            doe_id = int(row[0])

        cur = self._conn.execute(
            "UPDATE doe_parameters SET outcome = ?, failure_reason = ? "
            "WHERE doe_id = ?",
            (str(outcome), failure_reason, int(doe_id)),
        )
        self._conn.commit()
        return int(doe_id) if cur.rowcount else None

    # ── Electrode occupancy (single-use wells, persistent across sessions) ──

    def record_electrode_cast(
        self,
        board_id: int,
        electrode: int,
        *,
        run_id: str | None = None,
        iteration: int | None = None,
        sample_uuid: str | None = None,
    ) -> None:
        """Mark ``(board_id, electrode)`` as cast (idempotent per well).

        ``sample_uuid`` (T2.6) names the sample now occupying the well, tying it
        to the ``formulations`` row that describes the cast and the
        ``measurements`` rows taken off it. ``None`` for callers with nothing to
        mint — the well is still recorded as occupied, which is what this table
        exists to say; only the identity is absent.

        The ``INSERT OR REPLACE`` means a re-cast *replaces* the identity rather
        than accumulating one, which is correct: a re-cast well holds a new
        physical sample, and the previous one's rows keep the old uuid, so the
        two remain distinguishable in exactly the case ``(run_id, channel)``
        cannot separate them.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO electrode_occupancy "
            "(board_id, electrode, run_id, iteration, cast_at, sample_uuid) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(board_id), int(electrode), run_id, iteration, _now_iso(),
             sample_uuid or None),
        )
        self._conn.commit()

    def occupied_electrodes(self, board_id: int) -> set[int]:
        """Electrodes already cast on ``board_id`` (empty for a fresh board)."""
        rows = self._conn.execute(
            "SELECT electrode FROM electrode_occupancy WHERE board_id = ?",
            (int(board_id),),
        ).fetchall()
        return {int(r[0]) for r in rows}

    def set_active_board(self, board_id: int) -> None:
        """Persist the active board pointer (call on a confirmed replacement).

        Durable immediately, so a swap is not lost when the session ends before
        any cast reaches the fresh plate.  Never moves the pointer backwards —
        board ids are monotonic per project.
        """
        board_id = max(int(board_id), self.current_board_id())
        self._conn.execute(
            "INSERT OR REPLACE INTO board_state (id, board_id, updated_at) "
            "VALUES (1, ?, ?)",
            (board_id, _now_iso()),
        )
        self._conn.commit()

    # ── Waste + board inventory (physical limits; see core/consumables.py) ────

    def _kv_get(self, key: str) -> float | None:
        row = self._conn.execute(
            "SELECT value FROM rig_state WHERE key = ?", (str(key),)
        ).fetchone()
        return None if row is None else float(row[0])

    def _kv_set(self, key: str, value: float) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO rig_state (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (str(key), float(value), _now_iso()),
        )
        self._conn.commit()

    # Text-valued rig state. `rig_state.value` is declared REAL, but SQLite's
    # typing is per-value rather than per-column, so a TEXT value stores and
    # round-trips faithfully. Kept as a separate accessor pair rather than
    # loosening `_kv_get`, whose float contract every numeric caller relies on.
    def _kv_get_text(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM rig_state WHERE key = ?", (str(key),)
        ).fetchone()
        return None if row is None else str(row[0])

    def _kv_set_text(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO rig_state (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (str(key), str(value), _now_iso()),
        )
        self._conn.commit()

    def waste_level_uL(self) -> float | None:
        """Waste accumulated since the container was last emptied (µL)."""
        return self._kv_get("waste_level_uL")

    def set_waste_level(self, level_uL: float) -> None:
        self._kv_set("waste_level_uL", max(0.0, float(level_uL)))

    def spare_boards(self) -> int | None:
        """Fresh electrode plates on hand, or ``None`` when undeclared."""
        value = self._kv_get("spare_boards")
        return None if value is None else int(value)

    def set_spare_boards(self, n: int) -> None:
        self._kv_set("spare_boards", max(0, int(n)))

    # ── Campaign checkpoints (resume durability; see core/campaign_resume.py) ──

    def save_campaign_checkpoint(
        self,
        campaign: str,
        *,
        iteration: int,
        run_id: str | None = None,
        loop_state: str | None = None,
        board_id: int | None = None,
        spec_json: str | None = None,
        optimizer_json: str | None = None,
        rh_ceiling_streak: int = 0,
        consecutive_failures: int = 0,
    ) -> None:
        """Record (or replace) the resume point for *campaign*.

        A single ``INSERT OR REPLACE`` so the write is atomic: a crash mid-write
        leaves the previous checkpoint intact rather than a half-updated one.
        Call it **after** the iteration's observation has been told to the
        optimizer — see the schema note on ordering.

        ``rh_ceiling_streak`` (consecutive RH-decided equilibrate phases) and
        ``consecutive_failures`` (consecutive failed/unmeasured trials) are the
        two escalation counters. Both default to ``0`` and the write replaces the
        whole row, so a caller that tracks either must pass it on **every** call
        or the count is lost — which is the same contract every other column here
        already has.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO campaign_checkpoints "
            "(campaign, run_id, iteration, loop_state, board_id, spec_json, "
            " optimizer_json, rh_ceiling_streak, consecutive_failures, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(campaign), run_id, int(iteration), loop_state,
             None if board_id is None else int(board_id),
             spec_json, optimizer_json, max(0, int(rh_ceiling_streak)),
             max(0, int(consecutive_failures)), _now_iso()),
        )
        self._conn.commit()

    def campaign_checkpoint(self, campaign: str) -> dict[str, Any] | None:
        """The saved resume point for *campaign*, or ``None``."""
        row = self._conn.execute(
            "SELECT campaign, run_id, iteration, loop_state, board_id, "
            "spec_json, optimizer_json, rh_ceiling_streak, consecutive_failures, "
            "updated_at "
            "FROM campaign_checkpoints WHERE campaign = ?",
            (str(campaign),),
        ).fetchone()
        if row is None:
            return None
        keys = ("campaign", "run_id", "iteration", "loop_state", "board_id",
                "spec_json", "optimizer_json", "rh_ceiling_streak",
                "consecutive_failures", "updated_at")
        return dict(zip(keys, row))

    def campaign_checkpoints(self) -> list[dict[str, Any]]:
        """All saved resume points, newest first (for a resume picker)."""
        rows = self._conn.execute(
            "SELECT campaign, run_id, iteration, loop_state, board_id, updated_at "
            "FROM campaign_checkpoints ORDER BY updated_at DESC"
        ).fetchall()
        keys = ("campaign", "run_id", "iteration", "loop_state", "board_id",
                "updated_at")
        return [dict(zip(keys, r)) for r in rows]

    def clear_campaign_checkpoint(self, campaign: str) -> None:
        """Drop the resume point once a campaign ends on purpose.

        Only for *intentional* terminal states (converged, budget exhausted,
        operator stop). A parked or crashed campaign must keep its checkpoint —
        that is the whole point of having one.
        """
        self._conn.execute(
            "DELETE FROM campaign_checkpoints WHERE campaign = ?", (str(campaign),)
        )
        self._conn.commit()

    def advance_board(self) -> int:
        """Log a board replacement and return the new board id.

        Occupancy is keyed by board id, so moving the pointer forward is what
        "resets the electrode positions": the fresh board starts with nothing
        occupied while every prior board's casts stay queryable. **No occupancy
        row is ever deleted** — the reset is a new namespace, not erased
        history, which keeps past runs interpretable.
        """
        new_id = self.current_board_id() + 1
        self.set_active_board(new_id)
        return new_id

    def current_board_id(self) -> int:
        """The active board id (0 when nothing has been used yet).

        Takes the greater of the persisted pointer and the highest board with
        recorded casts.  The MAX fallback keeps projects created before
        ``board_state`` existed working, and the ``max`` keeps the id monotonic
        if a cast were ever recorded under a higher board than the pointer.
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(board_id), 0) FROM electrode_occupancy"
        ).fetchone()
        from_casts = int(row[0]) if row else 0
        row = self._conn.execute(
            "SELECT board_id FROM board_state WHERE id = 1"
        ).fetchone()
        persisted = int(row[0]) if row else 0
        return max(from_casts, persisted)

    # ── Reservoir levels (safety interlock; see core/reservoir.py) ──────

    def reservoir_level_uL(self, pump_id: int) -> float | None:
        """Remaining stock on *pump_id*, or ``None`` if the pump is unmanaged.

        ``None`` means *unknown*, never *empty* — an undeclared reservoir must
        not be mistaken for a depleted one.
        """
        row = self._conn.execute(
            "SELECT remaining_uL FROM reservoir_levels WHERE pump_id = ?",
            (int(pump_id),),
        ).fetchone()
        return float(row[0]) if row else None

    def set_reservoir_level(self, pump_id: int, remaining_uL: float) -> None:
        """Persist the remaining stock for *pump_id*."""
        self._conn.execute(
            "INSERT OR REPLACE INTO reservoir_levels (pump_id, remaining_uL, updated_at) "
            "VALUES (?, ?, ?)",
            (int(pump_id), float(remaining_uL), _now_iso()),
        )
        self._conn.commit()

    def reservoir_levels(self) -> dict[int, float]:
        """All managed pumps → remaining µL."""
        rows = self._conn.execute(
            "SELECT pump_id, remaining_uL FROM reservoir_levels"
        ).fetchall()
        return {int(r[0]): float(r[1]) for r in rows}

    # ── Unclean-shutdown detection ──────────────────────────────────────

    def unfinished_runs(self) -> list[dict]:
        """Runs still marked in-flight — evidence of an unclean stop.

        Every terminal path finalizes its run row (see
        ``run_autonomous_campaign._finalize_run``), so a row left with
        ``finished_at`` NULL means the process died without unwinding: a crash, a
        power cut, or an OS-forced restart.  On the next start-up this is the
        only durable signal that the rig may have been left mid-experiment.
        """
        rows = self._conn.execute(
            "SELECT run_id, workflow_name, started_at, status FROM experiments "
            "WHERE finished_at IS NULL ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Alerts (durable operator-facing notifications) ──────────────────

    def record_alert(
        self,
        kind: str,
        message: str,
        *,
        severity: str = "warning",
        run_id: str | None = None,
        details: dict | None = None,
    ) -> int:
        """Persist an operator-facing alert; returns its row id."""
        cur = self._conn.execute(
            "INSERT INTO alerts (raised_at, run_id, kind, severity, message, details) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                _now_iso(), run_id, str(kind), str(severity), str(message),
                json.dumps(details, default=str) if details else None,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query_alerts(
        self, *, run_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Most-recent alerts first, optionally scoped to one run."""
        if run_id is None:
            rows = self._conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE run_id = ? ORDER BY id DESC LIMIT ?",
                (run_id, int(limit)),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if d.get("details"):
                try:
                    d["details"] = json.loads(d["details"])
                except Exception:
                    pass
            out.append(d)
        return out

    def clear_board(self, board_id: int) -> None:
        """Forget occupancy for ``board_id`` (e.g. a board re-used as fresh)."""
        self._conn.execute(
            "DELETE FROM electrode_occupancy WHERE board_id = ?", (int(board_id),)
        )
        self._conn.commit()

    # ── Queries ─────────────────────────────────────────────────────────

    def query_measurements(
        self,
        *,
        run_id: str | None = None,
        channel: int | None = None,
        temp_range: tuple[float, float] | None = None,
        since: str | None = None,
        condition_stage: str = "measurement",
        limit: int | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve measurement rows with optional filters.

        When *temp_range* is provided the query joins the ``conditions``
        table filtered by *condition_stage* and applies a temperature
        range filter.
        """
        params: list[Any] = []
        clauses: list[str] = []

        if temp_range is not None:
            base = (
                "SELECT m.*, e.workflow_name, e.pcb_name "
                "FROM measurements m "
                "JOIN experiments e ON m.run_id = e.run_id "
                "JOIN conditions c ON c.measurement_id = m.measurement_id"
            )
            clauses.append("c.stage = ?")
            params.append(condition_stage)
            # The mirror of the source precedence that used to sit here — a
            # COALESCE restating `TEMPERATURE_SOURCES` because SQL cannot call
            # the resolver — is RETIRED as of schema epoch 4. `temperature_C`
            # *is* the resolver's answer, written at record time, so this filter
            # names one column and holds no copy of the precedence. Rows where
            # no thermometer spoke are NULL and drop out of the range, which is
            # correct: they have no temperature to be in range of.
            #
            # Accepted edge: a row written by a stale pre-epoch-4 process against
            # an already-migrated database lands with NULL and is invisible here
            # until the next open re-resolves it (the backfill targets exactly
            # those rows) — a single-machine, single-upgrade window.
            clauses.append("c.temperature_C BETWEEN ? AND ?")
            params.extend(temp_range)
        else:
            base = (
                "SELECT m.*, e.workflow_name, e.pcb_name "
                "FROM measurements m "
                "JOIN experiments e ON m.run_id = e.run_id"
            )

        if run_id is not None:
            clauses.append("m.run_id = ?")
            params.append(run_id)
        if channel is not None:
            clauses.append("m.channel = ?")
            params.append(channel)
        if since is not None:
            clauses.append("m.timestamp >= ?")
            params.append(since)

        sql = base
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY m.timestamp DESC" if descending else " ORDER BY m.timestamp"
        if limit is not None and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_conditions(
        self,
        *,
        measurement_id: int | None = None,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve condition snapshots with optional filters."""
        params: list[Any] = []
        clauses: list[str] = []

        if measurement_id is not None:
            clauses.append("measurement_id = ?")
            params.append(measurement_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)

        sql = "SELECT * FROM conditions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_fits(
        self,
        *,
        run_id: str | None = None,
        measurement_id: int | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve fit-result rows with optional filters."""
        params: list[Any] = []
        clauses: list[str] = []

        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if measurement_id is not None:
            clauses.append("measurement_id = ?")
            params.append(measurement_id)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)

        sql = "SELECT * FROM fit_results"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY fitted_at"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_formulations(
        self,
        *,
        run_id: str | None = None,
        channel: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve formulation rows with optional filters."""
        params: list[Any] = []
        clauses: list[str] = []

        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)

        sql = "SELECT * FROM formulations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY channel"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_doe_parameters(
        self,
        *,
        run_id: str | None = None,
        channel: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve DOE / optimizer observation rows with optional filters."""
        params: list[Any] = []
        clauses: list[str] = []

        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)

        sql = "SELECT * FROM doe_parameters"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY iteration"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_runs(
        self,
        *,
        campaign: str | None = None,
        quality: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve experiment-run rows with optional filters."""
        params: list[Any] = []
        clauses: list[str] = []

        if campaign is not None:
            clauses.append("campaign = ?")
            params.append(campaign)
        if quality is not None:
            clauses.append("quality = ?")
            params.append(quality)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM experiments"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC"

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── Cleanup ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Commit and close the database connection."""
        self._conn.commit()
        self._conn.close()
        logger.info("data_store_closed")

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── Internal helpers ────────────────────────────────────────────────

    def _migrate_config_hash(self) -> None:
        """Add ``config_hash`` column to ``experiments`` if it is missing."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "config_hash" not in cols:
            self._conn.execute(
                "ALTER TABLE experiments ADD COLUMN config_hash TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

    def _migrate_experiment_skipped_channels(self) -> None:
        """Add ``skipped_channels`` to ``experiments`` if it is missing (P4).

        The table's DDL is a bare ``CREATE TABLE IF NOT EXISTS``, so the operator's
        existing store — which is every store that already exists — never gains
        the column from it. Without this, :meth:`finish_run` would fail with
        ``no such column`` on the first HT plate that finished.

        **No backfill and no new ``SCHEMA_EPOCHS`` row**, following
        :meth:`_migrate_doe_outcome`. Historical rows keep ``NULL``, meaning *we
        do not know which channels on this plate were real* — which is the exact
        truth about them, and the reason this column exists. Every one of them was
        written ``status='done'`` whether it skipped 0 channels or 31, and nothing
        recoverable after the fact distinguishes the two. Writing ``'[]'`` into
        them would manufacture the certainty the fix is meant to stop
        manufacturing. This is a shape change, not a change of meaning, which is
        the condition T2.3 set for skipping an epoch row.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "skipped_channels" not in cols:
            self._conn.execute(
                "ALTER TABLE experiments ADD COLUMN skipped_channels TEXT"
            )
            self._conn.commit()

    def _migrate_annotation(self) -> None:
        """Add ``annotation`` column to ``experiments`` if it is missing."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if "annotation" not in cols:
            self._conn.execute(
                "ALTER TABLE experiments ADD COLUMN annotation TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

    def _migrate_formulation_pump2(self) -> None:
        """Add ``pump2_uL`` column to ``formulations`` if it is missing (legacy DBs).

        The HT tab dispenses from three pumps; older databases predate the third
        pump and default it to 0.0, which is correct for those 2-pump runs.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(formulations)").fetchall()
        }
        if "pump2_uL" not in cols:
            self._conn.execute(
                "ALTER TABLE formulations ADD COLUMN pump2_uL REAL NOT NULL DEFAULT 0.0"
            )
            self._conn.commit()

    def _migrate_formulation_thickness(self) -> None:
        """Add ``predicted_thickness_um`` to ``formulations`` (P7.6).

        The deposition twin's dry-film thickness for this channel's cast.
        Nullable on purpose: it is ``NULL`` whenever the twin could not speak —
        a raw per-pump spec with no composition, or a board declaring neither a
        deposit area nor a capacity. A default of 0.0 would be a *claim* that
        the film is zero thick, which the conductivity path would then divide
        by; absent must stay distinguishable from zero.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(formulations)").fetchall()
        }
        if "predicted_thickness_um" not in cols:
            self._conn.execute(
                "ALTER TABLE formulations ADD COLUMN predicted_thickness_um REAL"
            )
            self._conn.commit()

    def _migrate_formulation_area(self) -> None:
        """Add ``deposit_area_mm2`` and ``thickness_method`` to ``formulations`` (P.7).

        ``predicted_thickness_um`` stores a thickness but not the area it was divided
        by, and that area is board-dependent: on the 4-stripe board it moved from
        4.0 mm² (the inter-electrode rectangle) to 18.704 mm² (the well) when P7.2
        made the well authoritative. Rows written either side of that differ by a
        factor of 4.676 **in the same column, with the same units**, and a row does
        not record which board it was cast on — so the ambiguity is unrecoverable
        after the fact. Recording the denominator beside the quotient is the only
        thing that stops it deepening.

        ``thickness_method`` says which tier the thickness came from, in the
        vocabulary of :data:`FORMULATION_THICKNESS_METHODS`. On this table it is
        near-constant (``'predicted'`` when the twin spoke), and earns its place
        purely on the third state: ``NULL`` means *never recorded*, ``'unavailable'``
        means *recorded as absent*. Collapsing those two would lose exactly the
        distinction the column exists to make.

        **No backfill.** Legacy rows keep ``NULL`` on both columns. Writing 4.0 mm²
        into them would manufacture the false comparability this migration exists to
        prevent — we do not know which board each of them was cast on.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(formulations)").fetchall()
        }
        additions = {
            "deposit_area_mm2": "REAL",
            "thickness_method": "TEXT",
        }
        changed = False
        for name, decl in additions.items():
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE formulations ADD COLUMN {name} {decl}"
                )
                changed = True
        if changed:
            self._conn.commit()

    def _migrate_thermal_columns(self) -> None:
        """Add VFT/model columns to ``arrhenius_results`` if missing (legacy DBs).

        The table is the unified thermal-fit store; older databases predate VFT
        support and lack these columns.  Existing rows are Arrhenius fits, so the
        ``model`` default of ``'arrhenius'`` is correct for them.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(arrhenius_results)"
            ).fetchall()
        }
        additions = {
            "model": "TEXT NOT NULL DEFAULT 'arrhenius'",
            "A": "REAL",
            "B": "REAL",
            "T0_K": "REAL",
            "T0_C": "REAL",
        }
        changed = False
        for name, decl in additions.items():
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE arrhenius_results ADD COLUMN {name} {decl}"
                )
                changed = True
        if changed:
            self._conn.commit()

    def _migrate_conditions_stage_temp(self) -> None:
        """Add ``stage_temp_pv_C`` to ``conditions`` if missing (legacy DBs).

        Older databases have only a single temperature PV column — the one now
        called ``chamber_air_C``, which is the humidity sensor's air reading and
        not the sample's temperature at all.  The **stage** PV, read from the
        Modbus temperature controller, is stored separately so both can be
        surfaced; existing rows get ``NULL``.  (There is no thermocouple in this
        path: the NI-DAQ surface thermocouple is unwired on this rig — see
        :mod:`softae.core.conditions_capture`.)
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(conditions)"
            ).fetchall()
        }
        if "stage_temp_pv_C" not in cols:
            self._conn.execute(
                "ALTER TABLE conditions ADD COLUMN stage_temp_pv_C REAL"
            )
            self._conn.commit()

    #: Per-campaign escalation counters the resume path reads by name. Additive
    #: only, and each defaults to ``0`` — the honest value for a checkpoint
    #: written before the counter existed, which had never counted anything.
    _CHECKPOINT_COUNTER_COLUMNS: tuple[str, ...] = (
        "rh_ceiling_streak",
        "consecutive_failures",
    )

    def _migrate_campaign_checkpoint_counters(self) -> None:
        """Add the escalation-counter columns to ``campaign_checkpoints``.

        The table's DDL is a bare ``CREATE TABLE IF NOT EXISTS``, so an existing
        database never gains a column from it. Without this every real store —
        which is every store that already exists — would fail the resume read
        with ``no such column``.

        **No backfill and no new ``SCHEMA_EPOCHS`` row**: nothing already stored
        changes meaning, and ``0`` is what every pre-existing checkpoint honestly
        held.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(campaign_checkpoints)"
            ).fetchall()
        }
        missing = [c for c in self._CHECKPOINT_COUNTER_COLUMNS if c not in cols]
        for name in missing:
            self._conn.execute(
                f"ALTER TABLE campaign_checkpoints "
                f"ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
            )
        if missing:
            self._conn.commit()

    #: Legacy ``conditions`` temperature column → the instrument-named column it
    #: becomes.  Old names are quoted here **by design**: naming what it renames
    #: is the whole job of a rename guard.
    _CONDITIONS_TEMP_RENAMES: tuple[tuple[str, str], ...] = (
        ("temp_pv_C", "chamber_air_C"),
        ("temp_sp_C", "stage_temp_sp_C"),
    )

    def _migrate_conditions_temp_names(self) -> None:
        """Rename the ``conditions`` temperature columns after their instruments.

        ``temp_sp_C`` / ``temp_pv_C`` read as one controller's SP/PV pair. They
        were two different instruments — the Modbus stage controller and the
        humidity sensor's onboard air probe — with the stage's own PV sitting in
        a third column under a different prefix, so the obvious join selected the
        air. Schema epoch 3; **labels only**, every stored value is untouched.

        ``ALTER TABLE … RENAME COLUMN`` moves the data in place: no rebuild, no
        copy, no window in which a row exists twice. Each rename is guarded
        independently, so a database interrupted between the two finishes on the
        next open, and a database that never had the old names is a no-op.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(conditions)"
            ).fetchall()
        }
        pending = [
            (old, new)
            for old, new in self._CONDITIONS_TEMP_RENAMES
            if old in cols and new not in cols
        ]
        if pending:
            if sqlite3.sqlite_version_info < (3, 25):
                raise RuntimeError(
                    "conditions temperature rename needs SQLite >= 3.25 "
                    "(RENAME COLUMN); this interpreter bundles "
                    f"{sqlite3.sqlite_version}"
                )
            for old, new in pending:
                self._conn.execute(
                    f"ALTER TABLE conditions RENAME COLUMN {old} TO {new}"
                )

        # The index moves with the column, and moves *onto a different column*.
        # It existed only to serve `query_measurements(temp_range=…)`, which used
        # to filter the air probe; the sample's temperature is `stage_temp_pv_C`,
        # so an index on the air was making the wrong answer fast.
        #
        # Assessed and NOT chosen: an index on the COALESCE expression itself,
        # which is the only thing the filter could actually use. It would put a
        # third copy of the source precedence in the schema — after the resolver
        # and the SQL — and a copy inside an index definition is the one nobody
        # greps. `conditions` holds ~1.4k rows per characterization run; the scan
        # is not the problem here.
        #
        # Epoch 4 settled that differently, and better: the precedence is applied
        # once at record time and the filter now names `temperature_C`, which
        # `_migrate_conditions_resolved_temperature` indexes. This index no
        # longer serves that query — it stays because ad-hoc and analysis
        # queries do select on the stage PV directly, which is a different
        # question ("what did the stage read") from the one the filter asks
        # ("what temperature was this sample at").
        indexes = {
            row[0]
            for row in self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' AND tbl_name = 'conditions'"
            ).fetchall()
        }
        reindex = (
            "idx_conditions_temp_pv" in indexes
            or "idx_conditions_stage_temp_pv" not in indexes
        )
        if reindex:
            self._conn.execute("DROP INDEX IF EXISTS idx_conditions_temp_pv")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conditions_stage_temp_pv "
                "ON conditions(stage_temp_pv_C)"
            )
        if pending or reindex:
            self._conn.commit()

    def _migrate_conditions_resolved_temperature(self) -> None:
        """Add ``temperature_C`` / ``temperature_source`` to ``conditions`` (epoch 4).

        These are :func:`~softae.analysis.conditions.resolve_temperature_C`'s
        answer for a row, stored. The resolver stays the only authority for the
        precedence; what changes is that the answer is computed once, at record
        time, instead of being re-derived by every consumer — and, worse,
        restated as a ``COALESCE`` inside ``query_measurements(temp_range=…)``
        because SQL cannot call Python. That mirror is what this migration
        retires.

        **This is the first backfilling migration in this codebase, and the
        divergence is deliberate.** :meth:`_migrate_formulation_sample_uuid` and
        :meth:`_migrate_doe_outcome` both chose NULL-for-historical, and were
        right to: a sample uuid and a trial outcome are *facts the past failed to
        record*, and no amount of inspection recovers them — inventing values
        would fabricate exactly the fact the column exists to state honestly.
        A resolved temperature is not that. It is a deterministic function of
        three columns **already present in the same row**, so backfilling asserts
        nothing the row did not already say; it only spells the answer out. A
        NULL here would be the strictly worse choice, because it would mean every
        historical row silently drops out of temperature filtering.

        **SQL ``CASE`` was assessed and rejected**, on the same ground
        :meth:`_migrate_conditions_temp_names` rejected an index over the
        ``COALESCE`` expression: it would put a third copy of the source
        precedence into the schema — after the resolver and the query — and this
        one would additionally have to re-implement the resolver's *validity*
        rules (non-finite, at-or-below absolute zero) in SQL, where they would
        drift unnoticed. The rows go through the real function instead. There are
        a few thousand of them per database; a one-time Python pass is not the
        expensive part of an open.

        The backfill targets ``temperature_source IS NULL``, which is every row
        immediately after the ``ALTER``, nothing on a settled database, and
        exactly the right set after an open interrupted between the ``ALTER`` and
        the ``UPDATE``.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(conditions)"
            ).fetchall()
        }
        added = False
        for column, decl in (("temperature_C", "REAL"),
                             ("temperature_source", "TEXT")):
            if column not in cols:
                self._conn.execute(
                    f"ALTER TABLE conditions ADD COLUMN {column} {decl}"
                )
                added = True
        if added:
            self._conn.commit()

        stale = self._conn.execute(
            "SELECT condition_id, stage_temp_pv_C, stage_temp_sp_C, chamber_air_C "
            "FROM conditions WHERE temperature_source IS NULL"
        ).fetchall()
        if stale:
            resolved = []
            for condition_id, stage_pv, stage_sp, air in stale:
                celsius, source = resolve_temperature_C(
                    stage_pv_C=stage_pv, stage_sp_C=stage_sp, chamber_air_C=air
                )
                resolved.append((_f_or_none(celsius), source, condition_id))
            self._conn.executemany(
                "UPDATE conditions SET temperature_C = ?, temperature_source = ? "
                "WHERE condition_id = ?",
                resolved,
            )
            self._conn.commit()
            # The per-source distribution belongs here and not in the epoch note:
            # it is a fact about *this* file, and `SCHEMA_EPOCHS` is a per-code
            # constant seeded INSERT OR IGNORE, so a count baked into the note
            # would be false for every other database and would never reach the
            # ones that actually hold the backfilled rows. The backfill targets
            # `temperature_source IS NULL`, so this fires once per database and
            # counts exactly the rows it wrote -- not the whole table, which is
            # the wrong denominator.
            logger.info("conditions_temperature_backfilled", rows=len(resolved),
                        sources=dict(Counter(source for _, source, _ in resolved)))

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conditions_temperature_C "
            "ON conditions(temperature_C)"
        )
        self._conn.commit()

    def _migrate_measurement_role(self) -> None:
        """Add ``role`` and ``fixture_id`` to ``measurements`` (E2 calibration).

        A blank **is** an EIS spectrum — same columns, same file format, same
        conditions row — so it lives in ``measurements`` rather than a parallel table
        that would duplicate eleven columns and fork the loader.  ``role`` says which
        kind: ``sample`` (the default, so every existing row is already correct with
        no backfill), ``blank_open``, ``blank_short``, ``blank_load``,
        ``reference_cap``, ``reference_r``.

        ``fixture_id`` exists because commissioning data must be invalidated by a
        hardware change (framework §8.5): a short blank taken before a board swap
        must not be silently applied after one.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(measurements)"
            ).fetchall()
        }
        changed = False
        if "role" not in cols:
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN role TEXT NOT NULL "
                "DEFAULT 'sample'"
            )
            changed = True
        if "fixture_id" not in cols:
            self._conn.execute("ALTER TABLE measurements ADD COLUMN fixture_id TEXT")
            changed = True
        if "nominal_value" not in cols:
            # The reference part's *marked* value (ohms / farads), recorded at
            # acquisition. Without it the derivation cannot compute a load error or
            # check a capacitor against its code — and it is not recoverable later,
            # because nobody remembers which resistor was in the socket three weeks
            # ago. Supplying it at `run` time and needing it again at `derive` time
            # was a silent no-op that left the load validation permanently unmet.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN nominal_value REAL")
            changed = True
        if "electrode_mode" not in cols:
            # How the cell was *sensed* when this spectrum was taken (E1.7 / F17).
            #
            # Overhaul §3.10: three-electrode sensing of a load with no ionic path to
            # the reference stripe puts RE on a floating capacitive divider whose ratio
            # depends on the load — measured between 2.2 and 23 on one instrument in a
            # single session. Every two-terminal commissioning reference is such a load,
            # so a value obtained that way is not merely uncalibrated but
            # *uncalibratable*, and R24 forbids it entering the instrument envelope.
            #
            # Recording the mode is what makes that enforceable after the fact. Existing
            # rows default to 'unknown' rather than to a mode, because they were taken
            # before anyone was asked — and claiming they were two-electrode would be
            # inventing the one fact this column exists to establish.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN electrode_mode TEXT "
                "NOT NULL DEFAULT 'unknown'")
            changed = True

        # ── E6: overhaul §6's mandatory-at-ingest fields ─────────────────────
        #
        # Only the ones that are genuinely absent. §6 lists fourteen; most are already
        # here under other names, and adding a synonym would create two columns that
        # can disagree:
        #
        #   electrode_configuration -> `electrode_mode` (E1.7) is exactly this
        #   excitation_amplitude    -> `eis_params_json.eis_mv_ac`
        #   temperature / humidity  -> the `conditions` rows (5 values, SP and PV)
        #   thickness + method + unc-> `measured_thickness` / the twin's prediction
        #   fixture_blank_ids       -> `fixture_corrections` (E3)
        #   channel_id, fixture_id  -> already columns here
        #   cycles_per_point        -> deliberately absent: `eis_run_mscrbuild` exposes
        #                              no averaging parameter and `meas_loop_eis` passes
        #                              none, so a column would record a number nobody
        #                              commanded. Record what was commanded — nothing.
        #
        # What remains is unrecoverable after the fact, which is the test for whether
        # it belongs on the acquisition row at all.
        if "thermal_history" not in cols:
            # Overhaul §3.6: the as-cast and annealed states differ *systematically*,
            # so a spectrum without its pre-conditioning is not comparable to one with
            # a different history — and the difference is invisible in the spectrum.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN thermal_history TEXT "
                "NOT NULL DEFAULT ''")
            changed = True
        if "sweep_order" not in cols:
            # Position within the acquisition sequence, for drift detection. NULL
            # means unrecorded rather than first: a default of 0 would make every
            # legacy row claim to be the opening measurement of its sweep.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN sweep_order INTEGER")
            changed = True
        if "re_connection" not in cols:
            # F13 — the dominant source of quadrant violations to date. The analysis
            # path has consumed this since E0 and nothing ever stored it, so a stored
            # spectrum could not say whether its control loop was closed.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN re_connection TEXT "
                "NOT NULL DEFAULT 'unverified'")
            changed = True
        if "re_contact_verified" not in cols:
            # R26 — whether an ionic path to the reference stripe was confirmed for
            # this sample. Stored as 0/1 with 0 meaning *not verified*, which is the
            # honest reading of every row written before anyone was asked.
            #
            # This is what lets a stored σ explain its own scale. Without it the
            # K_config_factor applied to a measurement is only reconstructible from
            # whatever `[eis.cell]` happened to say at analysis time, which is not a
            # property of the measurement at all.
            self._conn.execute(
                "ALTER TABLE measurements ADD COLUMN re_contact_verified INTEGER "
                "NOT NULL DEFAULT 0")
            changed = True
        if changed:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_role "
                "ON measurements(role)"
            )
            self._conn.commit()

    def _migrate_eis_calibrations(self) -> None:
        """Append-only calibration history (E2).

        Two persistence layers, different jobs. The canonical set is a
        version-controlled ``calibration/eis/<fixture_id>.toml`` beside the code,
        because framework §8.5 requires commissioning data to travel with the
        software. *This* table is the history, and it appends rather than overwrites
        for a reason worth stating: **comparing successive calibrations of the same
        fixture is itself an instrument-drift measurement**, obtained from hardware
        repeats rather than from sample replicates. That is a large part of what P4.4
        wanted, for free, from work that has to happen anyway.
        """
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS eis_calibrations (
                   calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   fixture_id     TEXT NOT NULL,
                   hardware_hash  TEXT NOT NULL,
                   created_at     TEXT NOT NULL,
                   json           TEXT NOT NULL,
                   superseded_at  TEXT
               )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eis_calibrations_fixture "
            "ON eis_calibrations(fixture_id, created_at)"
        )
        self._conn.commit()

    def record_calibration(self, calibration: Any) -> int:
        """Append a calibration set to the history. Returns its ``calibration_id``.

        Supersedes any earlier live row for the same fixture — marking, not deleting,
        so the drift comparison stays possible.
        """
        import json as _json
        from datetime import datetime as _dt

        data = calibration.to_dict()
        now = _dt.now().isoformat(timespec="seconds")
        self._conn.execute(
            "UPDATE eis_calibrations SET superseded_at = ? "
            "WHERE fixture_id = ? AND superseded_at IS NULL",
            (now, data.get("fixture_id", "default")),
        )
        cur = self._conn.execute(
            "INSERT INTO eis_calibrations "
            "(fixture_id, hardware_hash, created_at, json) VALUES (?, ?, ?, ?)",
            (
                str(data.get("fixture_id", "default")),
                str(data.get("hardware_hash", "")),
                str(data.get("created_at", "") or now),
                _json.dumps(data, default=str),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def calibration_history(self, fixture_id: str = "default") -> list[dict[str, Any]]:
        """Every calibration recorded for *fixture_id*, oldest first.

        Successive rows are the drift record. Returned as raw dicts so a caller can
        rebuild :class:`CalibrationSet` objects without this module importing them.
        """
        import json as _json

        rows = self._conn.execute(
            "SELECT calibration_id, fixture_id, hardware_hash, created_at, json, "
            "superseded_at FROM eis_calibrations WHERE fixture_id = ? "
            "ORDER BY created_at, calibration_id",
            (fixture_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = _json.loads(row[4])
            except Exception:
                payload = {}
            out.append({
                "calibration_id": row[0], "fixture_id": row[1],
                "hardware_hash": row[2], "created_at": row[3],
                "superseded_at": row[5], "calibration": payload,
            })
        return out

    def _migrate_thickness(self) -> None:
        """Planned and measured film thickness (E5 harness).

        Two tables because they answer two questions asked at two different times.
        ``thickness_plans`` records what *should* be cast, before casting — that is the
        only moment at which overhaul F12's confounding can still be prevented.
        ``measured_thickness`` records what a profilometer later found, which is the
        top tier of :func:`~softae.analysis.eis.geometry.resolve_thickness_cm` and,
        until now, a tier nothing could reach.

        ``level_um`` is stored alongside the measured value on purpose: the planned
        level is what the design was balanced on, and the measured value is what
        actually happened. Keeping both is what makes "was this cast as planned?"
        answerable at analysis time rather than a matter of recollection.
        """
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS thickness_plans (
                   plan_id     TEXT PRIMARY KEY,
                   created_at  TEXT,
                   json        TEXT NOT NULL,
                   notes       TEXT
               )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS measured_thickness (
                   thickness_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                   plan_id        TEXT,
                   run_id         TEXT,
                   channel        INTEGER NOT NULL,
                   level_um       REAL,
                   thickness_um   REAL NOT NULL,
                   uncertainty_um REAL,
                   instrument     TEXT,
                   operator       TEXT,
                   measured_at    TEXT,
                   notes          TEXT
               )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measured_thickness_lookup "
            "ON measured_thickness(plan_id, channel)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_measured_thickness_run "
            "ON measured_thickness(run_id, channel)"
        )
        self._conn.commit()

    def record_thickness_plan(self, plan: Any) -> str:
        """Store a :class:`ThicknessPlan`. Returns its ``plan_id``.

        Replaces an existing plan with the same id — a plan is a statement of intent
        and re-planning before casting is legitimate. Once thicknesses are recorded
        against it, ``check`` compares the two, so a late edit cannot hide a deviation.
        """
        from datetime import datetime as _dt

        plan_id = str(getattr(plan, "plan_id", "") or "")
        if not plan_id:
            raise ValueError("a thickness plan needs a plan_id")
        self._conn.execute(
            "INSERT OR REPLACE INTO thickness_plans (plan_id, created_at, json, notes) "
            "VALUES (?, ?, ?, ?)",
            (plan_id,
             str(getattr(plan, "created_at", "") or _dt.now().isoformat(
                 timespec="seconds")),
             plan.to_json(),
             str(getattr(plan, "notes", "") or "") or None),
        )
        self._conn.commit()
        return plan_id

    def thickness_plan(self, plan_id: str) -> Any | None:
        """Load a stored plan, or ``None``."""
        from softae.core.thickness_series import ThicknessPlan

        row = self._conn.execute(
            "SELECT json FROM thickness_plans WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return ThicknessPlan.from_json(row[0])
        except Exception:
            logger.warning("thickness_plan_unreadable", plan_id=plan_id, exc_info=True)
            return None

    def thickness_plans(self) -> list[dict[str, Any]]:
        """Every stored plan, newest first."""
        rows = self._conn.execute(
            "SELECT plan_id, created_at, notes FROM thickness_plans "
            "ORDER BY created_at DESC, plan_id DESC"
        ).fetchall()
        return [{"plan_id": r[0], "created_at": r[1], "notes": r[2]} for r in rows]

    def record_thickness(
        self,
        channel: int,
        thickness_um: float,
        *,
        plan_id: str | None = None,
        run_id: str | None = None,
        level_um: float | None = None,
        uncertainty_um: float | None = None,
        instrument: str | None = None,
        operator: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Record one measured film thickness. Returns its row id.

        Appends rather than replaces: re-measuring a channel is a second observation,
        not a correction, and two measurements disagreeing is information worth keeping.
        :meth:`thickness_for` takes the most recent.
        """
        from datetime import datetime as _dt

        cur = self._conn.execute(
            "INSERT INTO measured_thickness "
            "(plan_id, run_id, channel, level_um, thickness_um, uncertainty_um, "
            " instrument, operator, measured_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (plan_id, run_id, int(channel), _f_or_none(level_um),
             float(thickness_um), _f_or_none(uncertainty_um),
             instrument, operator, _dt.now().isoformat(timespec="seconds"), notes),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def thickness_for(
        self, channel: int, *, plan_id: str | None = None, run_id: str | None = None
    ) -> float | None:
        """The most recent measured thickness (µm) for a channel, or ``None``.

        ``None`` is a legitimate answer and callers must treat it as *σ unavailable*
        rather than substituting a nominal — the same posture the rest of the thickness
        ladder takes.
        """
        sql = "SELECT thickness_um FROM measured_thickness WHERE channel = ?"
        args: list[Any] = [int(channel)]
        if plan_id:
            sql += " AND plan_id = ?"
            args.append(plan_id)
        if run_id:
            sql += " AND run_id = ?"
            args.append(run_id)
        sql += " ORDER BY thickness_id DESC LIMIT 1"
        row = self._conn.execute(sql, args).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def measured_thickness(
        self, *, plan_id: str | None = None, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Every recorded thickness, oldest first, optionally filtered."""
        sql = ("SELECT thickness_id, plan_id, run_id, channel, level_um, "
               "thickness_um, uncertainty_um, instrument, operator, measured_at, notes "
               "FROM measured_thickness")
        where, args = [], []
        if plan_id:
            where.append("plan_id = ?")
            args.append(plan_id)
        if run_id:
            where.append("run_id = ?")
            args.append(run_id)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY thickness_id"
        keys = ("thickness_id", "plan_id", "run_id", "channel", "level_um",
                "thickness_um", "uncertainty_um", "instrument", "operator",
                "measured_at", "notes")
        return [dict(zip(keys, r)) for r in self._conn.execute(sql, args).fetchall()]

    def _migrate_fixture_corrections(self) -> None:
        """Which fixture blanks were subtracted from which measurement (E3, R17/§7.5).

        **An absent row means uncorrected**, and that is the honest reading for every
        row written before this table existed — which is why the correction is recorded
        here rather than as a nullable column with a default. A default would have to
        claim something about the past; a missing row claims nothing.
        """
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS fixture_corrections (
                   correction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                   measurement_id INTEGER NOT NULL,
                   channel        INTEGER,
                   fixture_id     TEXT,
                   mode           TEXT NOT NULL,
                   R_short_ohm    REAL,
                   L_lead_H       REAL,
                   inherited      INTEGER NOT NULL DEFAULT 0,
                   declined       TEXT,
                   max_shift_pct  REAL,
                   induced_nonphysical INTEGER NOT NULL DEFAULT 0,
                   source_measurement_id INTEGER,
                   created_at     TEXT
               )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fixture_corrections_measurement "
            "ON fixture_corrections(measurement_id)"
        )
        self._conn.commit()

    def record_fixture_correction(
        self, measurement_id: int, correction: Any, outcome: Any = None
    ) -> int:
        """Record what was subtracted from *measurement_id*. Returns its row id.

        Records a declined correction too — ``mode = 'none'`` with the reason — because
        "no short blank existed yet" is a fact about the measurement worth keeping, and
        it is the difference between a spectrum nobody corrected and one nobody *could*.
        """
        from datetime import datetime as _dt

        cur = self._conn.execute(
            "INSERT INTO fixture_corrections "
            "(measurement_id, channel, fixture_id, mode, R_short_ohm, L_lead_H, "
            " inherited, declined, max_shift_pct, induced_nonphysical, "
            " source_measurement_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(measurement_id),
                int(getattr(correction, "channel", -1)),
                str(getattr(correction, "fixture_id", "") or ""),
                str(getattr(correction, "mode", "none")),
                _f_or_none(getattr(correction, "R_short_ohm", None)),
                _f_or_none(getattr(correction, "L_lead_H", None)),
                1 if getattr(correction, "inherited", False) else 0,
                str(getattr(correction, "declined", "") or "") or None,
                _f_or_none(getattr(outcome, "max_shift_pct", None)),
                int(getattr(outcome, "induced_nonphysical", 0) or 0),
                getattr(correction, "source_measurement_id", None),
                _dt.now().isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def fixture_correction_for(self, measurement_id: int) -> dict[str, Any] | None:
        """The correction applied to *measurement_id*, or ``None`` if uncorrected."""
        row = self._conn.execute(
            "SELECT channel, fixture_id, mode, R_short_ohm, L_lead_H, inherited, "
            "declined, max_shift_pct, induced_nonphysical, source_measurement_id, "
            "created_at FROM fixture_corrections WHERE measurement_id = ? "
            "ORDER BY correction_id DESC LIMIT 1",
            (int(measurement_id),),
        ).fetchone()
        if row is None:
            return None
        keys = ("channel", "fixture_id", "mode", "R_short_ohm", "L_lead_H",
                "inherited", "declined", "max_shift_pct", "induced_nonphysical",
                "source_measurement_id", "created_at")
        out = dict(zip(keys, row))
        out["inherited"] = bool(out["inherited"])
        return out

    def _migrate_fit_gate_columns(self) -> None:
        """Add the gated engine's provenance columns to ``fit_results`` (E0/E1).

        Every one is nullable or defaults to what the legacy path already does, so an
        existing row keeps meaning exactly what it meant: ``engine='legacy'``,
        ``report_mode='split'``, ``dead_height_cm=0.0``, ``sigma_is_bound=0``.

        ``gate_log_json`` holds ``run_gates``' log verbatim — R17's "named gate and
        reason" with no translation layer between the check and the record.

        **The four ``arc_*`` columns (T7.7) ride along here** rather than in a
        migration of their own: they are four more nullable additions to the same
        table, and a second PRAGMA pass over ``fit_results`` would buy nothing. They
        take no ``NOT NULL DEFAULT`` — ``'unknown'`` is an *answer* the annotator can
        give, so defaulting to it would put that answer in the mouth of every row
        written before anything looked. NULL means never annotated, and there is
        nothing to prove otherwise for a row written in July.

        **The ``arc_state`` index is created here and cannot be created in ``_DDL``.**
        That script runs at ``__init__`` before any migration, where a legacy
        ``fit_results`` still lacks the column and ``CREATE INDEX`` would raise
        ``no such column`` inside ``executescript``, failing the open of every
        existing project — the same trap the conditions DDL documents, resolved the
        same way. Creating it unconditionally after the ALTER loop covers both
        paths, because this migration runs on every open.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(fit_results)"
            ).fetchall()
        }
        additions = {
            "engine": "TEXT NOT NULL DEFAULT 'legacy'",
            "gate_verdict": "TEXT",
            "gate_log_json": "TEXT NOT NULL DEFAULT '[]'",
            "n_points_used": "INTEGER",
            "n_points_dropped": "INTEGER",
            "report_mode": "TEXT NOT NULL DEFAULT 'split'",
            "R_sum_ohm": "REAL",
            "R_sum_se_ohm": "REAL",
            "rho_series_bulk": "REAL",
            "sigma_is_bound": "INTEGER NOT NULL DEFAULT 0",
            "sigma_rel_unc": "REAL",
            "phase_headroom": "REAL",
            "model_free_R_ohm": "REAL",
            "K_per_cm": "REAL",
            "K_route": "TEXT",
            "dead_height_cm": "REAL NOT NULL DEFAULT 0.0",
            "thickness_method": "TEXT",
            "thickness_unc_cm": "REAL",
            "arc_state": "TEXT",
            "arc_f_peak_hz": "REAL",
            "arc_f_low_hz": "REAL",
            "arc_phase_low_deg": "REAL",
        }
        changed = False
        for name, decl in additions.items():
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE fit_results ADD COLUMN {name} {decl}"
                )
                changed = True
        if changed:
            self._conn.commit()

        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fit_results_arc_state "
            "ON fit_results(arc_state)"
        )
        self._conn.commit()

    def _migrate_modality(self) -> None:
        """Add the modality/payload contract to ``measurements`` (Tier 2 comp. 3).

        Four columns that let a row describe a measurement of *any* kind, so a
        camera frame or a profilometry trace attaches to the same table instead of
        forcing a parallel one. The existing EIS columns stay put as the de-facto
        EIS side table during the transition (spec §4 component 3).

        ``modality TEXT NOT NULL DEFAULT 'eis'`` **knowingly bends** this file's
        defaults-record-absence convention, which everywhere else insists a default
        must not look like an answer (``electrode_mode='unknown'``, ``sweep_order``
        NULL rather than 0). It is justified here, and only here, because the
        default is not a guess about an unrecorded fact — it is a *known* one.
        Until this migration runs, ``record_measurement`` takes an ``EISResult``
        and nothing else can reach the table: every pre-existing row is an EIS
        spectrum, provably, by construction. ``'unknown'`` would therefore be the
        false statement, discarding a fact we hold with certainty and forcing every
        reader to special-case a NULL that never meant anything. The convention
        exists to stop defaults inventing facts; it is not violated by a default
        that records one.

        The other three follow the convention unbent:

        * ``payload_path`` — NULL means *no payload was written* (the write failed,
          or the row predates payloads). Never ``''``: an empty string is a path
          that has been recorded, and ``os.path.join`` would happily build from it.
        * ``payload_format`` — NULL alongside a NULL path; ``'netcdf4'`` when one
          was written. Stored beside the path rather than inferred from the suffix
          so a re-encoding is a data change, not a filename convention.
        * ``sample_uuid`` — column only. Minting belongs to T2.6, which assigns it
          at cast time; every row written before then is honestly NULL rather than
          carrying a uuid invented at measurement time, which would attach distinct
          identities to what may be one physical sample.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(measurements)"
            ).fetchall()
        }
        additions = {
            "modality": "TEXT NOT NULL DEFAULT 'eis'",
            "payload_path": "TEXT",
            "payload_format": "TEXT",
            "sample_uuid": "TEXT",
        }
        changed = False
        for name, decl in additions.items():
            if name not in cols:
                self._conn.execute(
                    f"ALTER TABLE measurements ADD COLUMN {name} {decl}"
                )
                changed = True
        if changed:
            self._conn.commit()

    def _migrate_formulation_sample_uuid(self) -> None:
        """Add ``sample_uuid`` to ``formulations`` and ``electrode_occupancy`` (T2.6).

        :meth:`_migrate_modality` put the column on ``measurements`` only, which
        gives a spectrum an identity but nothing to join it *to*. The spine needs
        three anchors, because a physical sample is described by three rows that
        share nothing else reliable:

        * ``formulations`` — what was cast (volumes, thickness, the area it was
          divided by),
        * ``electrode_occupancy`` — which well it went into, on which board,
        * ``measurements`` — what was measured off it afterwards.

        ``(run_id, channel)`` almost joins them and is exactly wrong at the seam
        that matters: an electrode is board-relative and re-used across boards, a
        run casts the same channel numbers repeatedly, and a re-cast well produces
        a second formulation row that the first one's spectra would silently join
        to. A minted uuid is the only key that survives all three.

        Two separate tables in one migration because they are one fact. Splitting
        them would allow a database where a cast has an identity and the well it
        occupies does not — the halfway state this column exists to prevent.

        **No backfill**, following ``_migrate_formulation_area``: every historical
        row keeps ``NULL``, meaning *this row predates sample identity*. Inventing
        uuids for them would be worse than useless — a fresh uuid per row would
        assert that three rows describing one physical sample are three different
        samples, which is precisely the false statement the spine exists to
        refute. A resumed campaign therefore mints for new trials only.
        """
        for table in ("formulations", "electrode_occupancy"):
            cols = {
                row[1]
                for row in self._conn.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if "sample_uuid" not in cols:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN sample_uuid TEXT"
                )
        self._conn.commit()

    def _migrate_doe_outcome(self) -> None:
        """Add ``outcome`` + ``failure_reason`` to ``doe_parameters`` (T3.1b).

        Why this table rather than a new one: the DOE row already exists at
        exactly the right grain, is created *before* execution, and is already the
        row that carries the NULL objective. A second table would be a second
        definition of "a trial".

        Why it is needed at all: before this, only *one* failure signal was
        persisted at trial grain — ``objective_value`` being NULL — and it carried
        no reason. Every distinction the feasibility labels depend on (a rig
        timeout vs a film that never formed; an open circuit vs a short) existed
        **only in the log stream**, which dies with the process. So the reason had
        to become a column before a classifier could be trained on it.

        ``outcome`` vocabulary — ``'measured'`` | ``'infeasible'`` | ``'unknown'``,
        with NULL meaning *this row predates the feature*. There is deliberately
        **no ``'hardware_suspect'`` value**: a channel-reject pattern is a
        statement about a *channel across runs*, not about this trial, and this
        trial's own honest outcome is ``'unknown'``. Encoding a channel-level
        suspicion in a trial-level column would put two different subjects in one
        field — and a retraction would then have to rewrite history rather than
        simply stop counting the row.

        **No backfill and no new ``SCHEMA_EPOCHS`` row**, following
        :meth:`_migrate_formulation_sample_uuid`: historical rows keep NULL, which
        is *unknown*, never *empty*. Inventing an outcome for them would fabricate
        exactly the fact this column exists to record honestly — we cannot tell,
        after the event, whether a legacy NULL objective was a stage timeout or a
        film that never formed, and that distinction is the whole feature. This is
        a shape change rather than a change of meaning, which is the condition
        T2.3 set for skipping an epoch row.
        """
        cols = {
            row[1]
            for row in self._conn.execute(
                "PRAGMA table_info(doe_parameters)"
            ).fetchall()
        }
        for column in ("outcome", "failure_reason"):
            if column not in cols:
                self._conn.execute(
                    f"ALTER TABLE doe_parameters ADD COLUMN {column} TEXT"
                )
        self._conn.commit()

    def _migrate_schema_version(self) -> None:
        """Create and seed the ``schema_version`` epoch ledger (Tier 2 comp. 3).

        ``INSERT OR IGNORE`` on the primary key makes this idempotent *and*
        append-only in one stroke: re-opening a store is a no-op, and a row already
        present is never overwritten even if :data:`SCHEMA_EPOCHS` is later edited.
        That is deliberate — ``applied_at`` records when this database first learned
        of an epoch, which a rewriting seeder would destroy.

        The ``CREATE TABLE IF NOT EXISTS`` is repeated from the DDL so the function
        stands alone: a legacy database that has never run the DDL still gets the
        ledger, and the seed below has somewhere to land.
        """
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_version (
                   version    INTEGER PRIMARY KEY,
                   applied_at TEXT    NOT NULL,
                   kind       TEXT    NOT NULL
                              CHECK (kind IN ('schema', 'data-epoch')),
                   note       TEXT    NOT NULL DEFAULT ''
               )"""
        )
        now = _now_iso()
        for version, kind, note in SCHEMA_EPOCHS:
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_version "
                "(version, applied_at, kind, note) VALUES (?, ?, ?, ?)",
                (version, now, kind, note),
            )
        self._conn.commit()

    # ── Schema / data epochs ────────────────────────────────────────────

    def schema_epochs(self) -> list[dict[str, Any]]:
        """Every ledger row, oldest first.

        Both kinds are returned together because a reader interpreting a stored
        value needs them together: ``'schema'`` rows say what shape the row has,
        ``'data-epoch'`` rows say what its numbers mean. Filtering is the caller's
        business — splitting them here would invite reading only one.
        """
        return [
            dict(r) for r in self._conn.execute(
                "SELECT version, applied_at, kind, note FROM schema_version "
                "ORDER BY version"
            ).fetchall()
        ]

    def current_schema_version(self) -> int:
        """The highest ledger version, or ``0`` if the ledger is empty.

        ``0`` is unambiguous: version numbering starts at 1, so it can only mean
        *no epoch has been recorded* — a database opened by code that predates the
        ledger. It never collides with a real epoch.
        """
        row = self._conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    # ── Thermal fits (Arrhenius / VFT) ──────────────────────────────────

    def record_thermal_fit(self, run_id: str, result: Any) -> int:
        """Insert one thermal-fit row (Arrhenius **or** VFT) into the unified table.

        The model is taken from ``result.model``.  Parameters absent from a given
        model (e.g. ``Ea_eV`` for a VFT fit, or ``B``/``T0`` for Arrhenius) are
        stored as ``NULL`` via ``getattr(..., None)``.

        Parameters
        ----------
        run_id : str
            Experiment run identifier.
        result : ArrheniusResult | VftResult
            Fitted output for a single channel.

        Returns
        -------
        int
            The ``id`` of the newly inserted row.
        """
        import json as _json

        cur = self._conn.execute(
            """
            INSERT INTO arrhenius_results (
                run_id, channel, model,
                Ea_eV, Ea_kJ_per_mol, ln_A, A, B, T0_K, T0_C,
                R_squared, T_min_C, T_max_C, n_points,
                fit_success, error_msg,
                conductivities_json, temperatures_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result.channel,
                getattr(result, "model", "arrhenius"),
                getattr(result, "Ea_eV", None),
                getattr(result, "Ea_kJ_per_mol", None),
                getattr(result, "ln_A", None),
                getattr(result, "A", None),
                getattr(result, "B", None),
                getattr(result, "T0_K", None),
                getattr(result, "T0_C", None),
                result.R_squared,
                getattr(result, "T_min_C", None),
                getattr(result, "T_max_C", None),
                result.n_points,
                int(result.fit_success),
                result.error_msg,
                _json.dumps(result.conductivities),
                _json.dumps(result.temperatures_C),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_arrhenius(self, run_id: str, result: Any) -> int:
        """Back-compat alias for :meth:`record_thermal_fit`."""
        return self.record_thermal_fit(run_id, result)

    def query_arrhenius(
        self,
        *,
        run_id: str | None = None,
        channel: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve Arrhenius result rows with optional filters.

        Parameters
        ----------
        run_id : str or None
            Filter to a specific experiment.
        channel : int or None
            Filter to a specific channel.

        Returns
        -------
        list[dict]
            Each dict matches one ``arrhenius_results`` row; ``conductivities``
            and ``temperatures_C`` are decoded from JSON into Python lists.
        """
        import json as _json

        params: list[Any] = []
        clauses: list[str] = []

        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)

        sql = "SELECT * FROM arrhenius_results"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, channel"

        rows = self._conn.execute(sql, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["conductivities"] = _json.loads(d.pop("conductivities_json", "[]"))
            d["temperatures_C"] = _json.loads(d.pop("temperatures_json", "[]"))
            results.append(d)
        return results

    def query_thermal_fits(
        self, *, run_id: str | None = None, channel: int | None = None
    ) -> list[dict[str, Any]]:
        """Alias for :meth:`query_arrhenius` — returns Arrhenius and VFT rows.

        Each row includes ``model`` and the model-specific columns
        (``Ea_eV``/… for Arrhenius, ``A``/``B``/``T0_K``/``T0_C`` for VFT).
        """
        return self.query_arrhenius(run_id=run_id, channel=channel)

    def _run_id_for_measurement(self, measurement_id: int) -> str:
        """Look up the run_id that owns *measurement_id*."""
        row = self._conn.execute(
            "SELECT run_id FROM measurements WHERE measurement_id = ?",
            (measurement_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"No measurement with id {measurement_id}")
        return row["run_id"]
