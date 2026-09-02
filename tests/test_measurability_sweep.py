"""Tests for the offline S1/S2/S3 sweep (``softae.tools.measurability_sweep``).

**No live DataStore.** Every test builds its own sqlite file and its own spectra, so the
suite says nothing about whether the operator's store happens to be present — a test that
passes only on this machine is a test about this machine. The *corpus reachability*
question the sweep exists to answer is deliberately **not** asked here: it cannot be, since
a fixture reaches whatever branch the fixture was written to reach (``SUBAGENT_RULES`` §3,
"a fixture that is off the production manifold"). That answer comes from running the tool
over the real store and is reported alongside it, not asserted in pytest.

What *is* asserted here is the harness logic the engine cannot check for itself: path
resolution, the provenance rule, skip accounting, the phase-table fallback, CSV shape, and
that the store is opened read-only in fact rather than in intent.

**And, since 2026-08-28, the numbers themselves.** A positive-control mutation run by
afl-session — ``measurability.parallel_capacitance`` returning all NaN — killed nine tests
in the engine's own file and **zero** here, on a suite that imports the real engine and
mocks nothing. The diagnosis is theirs and is accepted: *an all-NaN engine produces a
well-formed frame full of NaN, and this file was checking well-formedness.* Every
assertion above was downstream of the plumbing and none was downstream of a value.

That is worse here than ordinary thin coverage, because this sweep is the instrument that
produces the distribution every S1/S2/S3 arming threshold will be chosen from: a silently
dead engine would hand an arming discussion a well-formed CSV and a green suite. So the
``analytic values`` section below constructs spectra whose S1/S2/S3 answers are known in
closed form and asserts the numbers the sweep emits, and the ``real blank`` section runs
the same path over a stored commissioning spectrum with bench-measured answers.

The ``--expect-engine-sha`` section covers the gap the fingerprint alone leaves. Two
sweeps nine minutes apart disagreed 25x on median S1 lift; the cause, established later,
was that the earlier one had run against a *deliberately mutated* engine during another
session's audit. ``engine_sha`` would have shown those two runs to be incommensurable —
but only because there were two. One sweep against a mutated engine is fingerprinted,
internally consistent, complete and wrong. So these tests assert the four states of the
expectation check, that a mismatch is loud, that it is confined to reporting, and that
"not checked" is stated rather than left blank.

**And, since 2026-08-31, that the digest names CONTENT rather than a checkout rendering.**
The fingerprint originally hashed the source file's working-tree bytes; with
``core.autocrlf=true`` against an LF repository that made it a property of the checkout, so
one commit measured ``70e94699b69f…`` here and ``a7d0c8fbacae4b99…`` in the index. It could
therefore both false-alarm on a correct engine and fail to pin content between machines —
the same failure family it exists to prevent. The section on line-ending invariance is the
test that would have caught it; the section after it covers the retired value being pasted,
which is a stale citation and not a wrong engine.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis.calibration import PhaseAccuracyTable
from softae.analysis.eis_data import EISResult
from softae.tools.measurability_sweep import (
    BYTE_DIGEST_RETIRED_ON,
    CSV_COLUMNS,
    ENGINE_SHA_BYTE_MATCH_LABEL,
    ENGINE_SHA_MATCH_LABELS,
    ENGINE_SHA_MIN_PREFIX,
    SKIP_REASONS,
    EngineShaCheck,
    ScalarColumnHealth,
    SweepResult,
    SweepRow,
    _expected_sha_arg,
    _open_readonly,
    classify_provenance,
    engine_byte_fingerprint_full,
    engine_fingerprint,
    engine_fingerprint_full,
    load_phase_tables,
    main,
    normalize_engine_sha,
    resolve_spectrum_path,
    scalar_column_health,
    select_phase_table,
    sweep,
    sweep_summary,
    write_csv,
)

# ── fixtures ─────────────────────────────────────────────────────────────────

_EXPERIMENTS_DDL = """
CREATE TABLE experiments (
    run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
    workflow_name TEXT NOT NULL, workflow_mode TEXT NOT NULL DEFAULT 'unknown',
    campaign TEXT NOT NULL DEFAULT 'dev')
"""
_MEASUREMENTS_DDL = """
CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    channel INTEGER NOT NULL, timestamp TEXT NOT NULL, eis_file_path TEXT,
    eis_params_json TEXT NOT NULL DEFAULT '{}', role TEXT NOT NULL DEFAULT 'sample',
    fixture_id TEXT, electrode_mode TEXT NOT NULL DEFAULT 'unknown',
    re_connection TEXT NOT NULL DEFAULT 'unverified',
    modality TEXT NOT NULL DEFAULT 'eis')
"""


#: The sweep grid the analytic fixtures are built on: 41 points, 0.125 decades apart.
F = np.logspace(0.0, 5.0, 41)


def _z_from_admittance(f, G, C) -> np.ndarray:
    """``Z = 1/(G + jωC)`` — constructed in the coordinates S1–S3 actually measure.

    Building in admittance rather than impedance makes ``Im(Y)/ω`` *exactly* the ``C``
    handed in, so a test can state the plateau, the lift and the loss tangent it expects
    instead of deriving them from the fixture it just built. The same construction the
    engine's own file uses, deliberately: a fixture whose expected value is computed by
    the code under test asserts nothing.
    """
    f = np.asarray(f, dtype=float)
    Y = np.broadcast_to(np.asarray(G, dtype=float), f.shape) + 1j * (
        2.0 * np.pi * f * np.broadcast_to(np.asarray(C, dtype=float), f.shape))
    return 1.0 / Y


def _write_spectrum(path, freq, Z, *, channel: int = 19) -> None:
    """Write *freq*/*Z* through the real ``EISResult.save``.

    Round-tripped through the production writer and reader rather than hand-rolled, so a
    change to the file format breaks these tests instead of leaving them passing against
    a format nothing writes. The round trip is exact to machine precision on this format,
    which is what lets the tests below assert closed-form values rather than tolerances
    chosen to survive a lossy writer.
    """
    Z = np.asarray(Z, dtype=complex)
    EISResult(
        channel=channel, frequency=np.asarray(freq, dtype=float), z_magnitude=np.abs(Z),
        phase=np.angle(Z, deg=True), z_real=Z.real, z_imag_neg=-Z.imag,
    ).save(path)


def _spectrum(path, *, channel: int = 19) -> None:
    """The default fixture spectrum: one parallel RC over 30 points."""
    freq = np.logspace(0, 5, 30)
    omega = 2.0 * np.pi * freq
    _write_spectrum(path, freq, 1.0 / (1.0 / 2.0e6 + 1j * omega * 4.0e-10),
                    channel=channel)


def _make_store(tmp_path, rows, *, spectrum=_spectrum):
    """Build a project dir with a store and the spectra *rows* reference.

    *rows* are ``(eis_file_path, params, fixture_id, mode, campaign)`` tuples; a
    ``eis_file_path`` of ``"@"`` means "write a real spectrum here and use its relative
    path", which keeps each test's table declarative. *spectrum* is the writer used for
    ``"@"`` rows, so a value test supplies a spectrum whose answers it knows.
    """
    project_dir = tmp_path / "proj"
    (project_dir / "db").mkdir(parents=True)
    db = project_dir / "db" / "softae.db"
    conn = sqlite3.connect(db)
    conn.executescript(_EXPERIMENTS_DDL + ";" + _MEASUREMENTS_DDL)
    for i, (rel, params, fixture, mode, campaign) in enumerate(rows):
        run = f"run{i}"
        conn.execute(
            "INSERT INTO experiments(run_id, started_at, workflow_name, workflow_mode,"
            " campaign) VALUES (?,?,?,?,?)",
            (run, "2026-01-01T00:00:00", "wf", mode, campaign))
        if rel == "@":
            rel = f"runs/{run}/eis/ch19.txt"
            spectrum(project_dir / rel)
        conn.execute(
            "INSERT INTO measurements(run_id, channel, timestamp, eis_file_path,"
            " eis_params_json, fixture_id) VALUES (?,?,?,?,?,?)",
            (run, 19, "2026-01-01T00:00:01", rel, json.dumps(params), fixture))
    conn.commit()
    conn.close()
    return db, project_dir


@pytest.fixture()
def calibration_root(tmp_path):
    """One committed-shaped calibration on disk, so the sole-fallback rule applies."""
    from softae.analysis.eis.calibration import CalibrationSet, save_calibration

    root = tmp_path / "cal"
    save_calibration(
        CalibrationSet(
            fixture_id="mux16", hardware_hash="abc", created_at="2026-01-01T00:00:00",
            phase_acc=PhaseAccuracyTable(
                z_ohm=(1.0e3, 1.0e5, 1.0e7), eps_deg=(1.0, 2.0, 3.0),
                load="capacitive", valid_decades=1.0)),
        root=root)
    return root


#: The phase table the S2 value test is computed against: three decades apart, ε rising,
#: so a log-interpolated answer at a known |Z| is a different number from every tabulated
#: point and from the envelope's lowest-|Z| entry.
S2_TABLE_Z_OHM = (1.0e6, 1.0e8, 1.0e10)
S2_TABLE_EPS_DEG = (0.4, 1.6, 4.0)


@pytest.fixture()
def s2_calibration_root(tmp_path):
    """A sole calibration whose table brackets the S2 fixture's minimum-tan δ impedance."""
    from softae.analysis.eis.calibration import CalibrationSet, save_calibration

    root = tmp_path / "cal_s2"
    save_calibration(
        CalibrationSet(
            fixture_id="mux16", hardware_hash="abc", created_at="2026-01-01T00:00:00",
            phase_acc=PhaseAccuracyTable(
                z_ohm=S2_TABLE_Z_OHM, eps_deg=S2_TABLE_EPS_DEG,
                load="capacitive", valid_decades=1.0)),
        root=root)
    return root


# ── path resolution ──────────────────────────────────────────────────────────

def test_resolve_spectrum_path_relative_joins_project_dir(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "a.txt").write_text("x", encoding="utf-8")
    path, reason = resolve_spectrum_path("runs/a.txt", tmp_path)
    assert reason is None
    assert path == tmp_path / "runs" / "a.txt"


def test_resolve_spectrum_path_absolute_ignores_project_dir(tmp_path):
    absolute = tmp_path / "elsewhere.txt"
    absolute.write_text("x", encoding="utf-8")
    other = tmp_path / "not_the_project"
    other.mkdir()
    path, reason = resolve_spectrum_path(str(absolute), other)
    assert reason is None
    assert path == absolute


def test_resolve_spectrum_path_dangling_relative_reports_file_missing(tmp_path):
    path, reason = resolve_spectrum_path("runs/gone.txt", tmp_path)
    assert path is None and reason == "file_missing"


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_resolve_spectrum_path_blank_reports_no_file_path(tmp_path, raw):
    path, reason = resolve_spectrum_path(raw, tmp_path)
    assert path is None and reason == "no_file_path"


# ── provenance ───────────────────────────────────────────────────────────────

def test_classify_provenance_mock_flag_true_is_mock_declared():
    assert classify_provenance('{"eis_validation_mock": true}', "validation") == \
        "mock_declared"


def test_classify_provenance_mock_flag_false_is_real_declared():
    assert classify_provenance('{"eis_validation_mock": false}', "validation") == \
        "real_declared"


def test_classify_provenance_simulation_mode_is_simulated_run():
    assert classify_provenance("{}", "simulation") == "simulated_run"


def test_classify_provenance_absent_flag_is_undeclared():
    """The label the whole real corpus carries. Named for what the store asserts —
    nothing — rather than for what is probably true."""
    assert classify_provenance('{"f_hi": 200000}', "full") == "undeclared"


def test_classify_provenance_mock_flag_outranks_simulation_mode():
    """Precedence is most-specific-first: the row's own declaration beats its run's mode,
    because the mode is a property of runs that produced no rows."""
    assert classify_provenance('{"eis_validation_mock": false}', "simulation") == \
        "real_declared"


@pytest.mark.parametrize("blob", [None, "", "not json", "[1,2,3]"])
def test_classify_provenance_unparseable_params_is_undeclared(blob):
    assert classify_provenance(blob, "full") == "undeclared"


# ── phase-table selection ────────────────────────────────────────────────────

def test_select_phase_table_named_fixture_uses_its_own_table():
    tables = {"mux16": PhaseAccuracyTable(z_ohm=(1.0,), eps_deg=(1.0,)),
              "other": PhaseAccuracyTable(z_ohm=(2.0,), eps_deg=(2.0,))}
    table, label = select_phase_table("mux16", tables)
    assert label == "mux16" and table is tables["mux16"]


def test_select_phase_table_null_fixture_sole_calibration_falls_back():
    """3727 of 3745 spectrum rows have ``fixture_id IS NULL``; without this rule they
    would all be provisional, and the fallback must be visible in the label."""
    tables = {"mux16": PhaseAccuracyTable(z_ohm=(1.0,), eps_deg=(1.0,))}
    table, label = select_phase_table(None, tables)
    assert table is tables["mux16"] and label == "mux16(sole-fallback)"


def test_select_phase_table_null_fixture_two_calibrations_declines():
    tables = {"a": PhaseAccuracyTable(z_ohm=(1.0,), eps_deg=(1.0,)),
              "b": PhaseAccuracyTable(z_ohm=(2.0,), eps_deg=(2.0,))}
    assert select_phase_table("", tables) == (None, "none")


def test_select_phase_table_unknown_fixture_declines_rather_than_falling_back():
    """A row naming a fixture we have no calibration for gets no table — inheriting the
    sole one would be applying another board's constants, which is the thing
    ``resolve_calibration`` refuses to do."""
    tables = {"mux16": PhaseAccuracyTable(z_ohm=(1.0,), eps_deg=(1.0,))}
    assert select_phase_table("mux99", tables) == (None, "none")


def test_load_phase_tables_calibration_without_phase_data_is_excluded(tmp_path):
    from softae.analysis.eis.calibration import CalibrationSet, save_calibration

    root = tmp_path / "cal"
    save_calibration(CalibrationSet(fixture_id="bare"), root=root)
    assert load_phase_tables(root) == {}


# ── the sweep ────────────────────────────────────────────────────────────────

def test_sweep_skip_accounting_covers_every_considered_row(tmp_path, calibration_root):
    db, project_dir = _make_store(tmp_path, [
        ("@", {}, None, "full", "manual"),
        (None, {"eis_validation_mock": True}, None, "validation", "eis_validate:mock"),
        ("runs/gone/eis/ch19.txt", {}, None, "full", "manual"),
    ])
    result = sweep(db, project_dir, calibration_root=calibration_root)

    assert result.n_considered == 3
    assert len(result.rows) + sum(result.skipped.values()) == result.n_considered
    assert result.skipped["no_file_path"] == 1
    assert result.skipped["file_missing"] == 1
    assert set(result.skipped_ids.values()) <= set(SKIP_REASONS)


def test_sweep_unreadable_spectrum_is_counted_not_raised(tmp_path, calibration_root):
    db, project_dir = _make_store(tmp_path, [("runs/bad.txt", {}, None, "full", "c")])
    (project_dir / "runs").mkdir(parents=True, exist_ok=True)
    (project_dir / "runs" / "bad.txt").write_bytes(b"\xff\xfe not a spectrum")

    result = sweep(db, project_dir, calibration_root=calibration_root)
    assert result.rows == []
    assert result.skipped["unreadable"] == 1


def test_sweep_row_carries_run_provenance_and_phase_table(tmp_path, calibration_root):
    db, project_dir = _make_store(
        tmp_path, [("@", {"f_hi": 2e5}, None, "full", "arrhenius")])
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.campaign == "arrhenius"
    assert row.workflow_mode == "full"
    assert row.provenance == "undeclared"
    assert row.phase_table == "mux16(sole-fallback)"
    assert row.s3_re_state == "unverified"
    assert row.n_points == 30


def test_sweep_without_calibration_marks_margin_provisional(tmp_path):
    """No table means NaN ε means NaN margin — provisional, never a pass."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    (row,) = sweep(db, project_dir, calibration_root=tmp_path / "no_cal").rows

    assert row.phase_table == "none"
    assert row.s2_eps_deg != row.s2_eps_deg
    assert row.s2_provisional is True


def test_sweep_row_records_the_engine_that_produced_it(tmp_path, calibration_root):
    """An S1 value is a claim about the engine that computed it, and this engine has no
    version — a sweep run inside a transient mutation of it put the corpus median lift 25x
    off. The fingerprint is what makes two CSVs commensurable."""
    from softae.tools.measurability_sweep import engine_fingerprint

    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.engine_sha == engine_fingerprint()
    assert row.engine_sha != "unknown" and len(row.engine_sha) == 12


def test_engine_fingerprint_tracks_the_measurability_source(monkeypatch, tmp_path):
    """It must digest the source actually imported, not a constant. Pointing the module
    at different bytes must change the answer, or the column is decoration."""
    from softae.analysis.eis import measurability
    from softae.tools.measurability_sweep import engine_fingerprint

    before = engine_fingerprint()
    other = tmp_path / "other.py"
    other.write_text("# not the engine\n", encoding="utf-8")
    monkeypatch.setattr(measurability, "__file__", str(other))
    assert engine_fingerprint() != before


# ── the digest names CONTENT, not a checkout rendering ───────────────────────
#
# Until 2026-08-31 this module hashed the source file's bytes as they sat in the working
# tree. This machine has ``core.autocrlf=true`` against an LF repository, so that digest was
# a property of the CHECKOUT: the identical commit measured 70e94699b69f... here and
# a7d0c8fbacae4b99... in the index, a clone with autocrlf=false would have been shouted at
# as a WRONG ENGINE while holding byte-identical content, and two machines could never agree
# on the digest of one commit. The fingerprint had exactly the defect it was built to
# prevent — an identifier that looks like it names content and names a rendering of it.

_ENGINE_SOURCE_LF = (
    b'"""A stand-in engine."""\n\n\ndef parallel_capacitance(freq, Z):\n'
    b"    return freq / Z\n")


def _point_engine_at(monkeypatch, path: Path) -> None:
    """Make the fingerprint functions digest *path* instead of the real engine."""
    from softae.analysis.eis import measurability

    monkeypatch.setattr(measurability, "__file__", str(path))


def test_engine_fingerprint_is_identical_for_crlf_and_lf_renderings(
        monkeypatch, tmp_path):
    """The test that would have caught the bug.

    Two files, byte-different and content-identical — which is precisely the relationship
    between this working tree and this repository's index. The content digest must not be
    able to tell them apart, because if it can, then "the engine that produced these
    numbers" is answered differently on two checkouts of one commit and the column cannot be
    the portable identity every CSV cites it as.
    """
    lf = tmp_path / "engine_lf.py"
    crlf = tmp_path / "engine_crlf.py"
    lf.write_bytes(_ENGINE_SOURCE_LF)
    crlf.write_bytes(_ENGINE_SOURCE_LF.replace(b"\n", b"\r\n"))
    # The fixture is only worth anything if the two really are byte-different: a writer that
    # silently normalised would make the assertion below true and empty.
    assert lf.read_bytes() != crlf.read_bytes()

    _point_engine_at(monkeypatch, lf)
    lf_content, lf_bytes = engine_fingerprint_full(), engine_byte_fingerprint_full()
    _point_engine_at(monkeypatch, crlf)
    crlf_content, crlf_bytes = engine_fingerprint_full(), engine_byte_fingerprint_full()

    assert lf_content == crlf_content
    # And the retired digest is shown doing the wrong thing, so the assertion above is
    # demonstrably about normalisation rather than about both digests being constants.
    assert lf_bytes != crlf_bytes
    assert lf_content == lf_bytes            # an LF file's two digests coincide
    assert crlf_content != crlf_bytes        # a CRLF checkout's do not


def test_engine_fingerprint_folds_lone_cr_as_well(monkeypatch, tmp_path):
    """Normalisation with a hole in it is a normalisation nobody can rely on."""
    cr = tmp_path / "engine_cr.py"
    cr.write_bytes(_ENGINE_SOURCE_LF.replace(b"\n", b"\r"))
    lf = tmp_path / "engine_lf.py"
    lf.write_bytes(_ENGINE_SOURCE_LF)

    _point_engine_at(monkeypatch, lf)
    expected = engine_fingerprint_full()
    _point_engine_at(monkeypatch, cr)
    assert engine_fingerprint_full() == expected


def test_engine_fingerprint_still_separates_genuinely_different_content(
        monkeypatch, tmp_path):
    """The falsifiable half: normalising must not have flattened the digest into a
    constant. A one-character edit to the engine still has to move it."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_bytes(_ENGINE_SOURCE_LF)
    b.write_bytes(_ENGINE_SOURCE_LF.replace(b"freq / Z", b"freq * Z"))

    _point_engine_at(monkeypatch, a)
    first = engine_fingerprint_full()
    _point_engine_at(monkeypatch, b)
    assert engine_fingerprint_full() != first


def test_engine_fingerprint_equals_the_digest_of_the_head_blob():
    """Content identity, checked against the one authority on this repository's content.

    ``git show HEAD:<path>`` is the stored form — LF — and an unmodified working tree is the
    same content in this machine's rendering. The two digests agreeing is the whole claim:
    a caller on any other checkout computes the same value, so the digest can be quoted
    between machines. Skipped rather than asserted when the engine is dirty, since a
    modified working tree is legitimately a different content and the test would then be
    reporting on another session's edit rather than on this function.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    rel = "src/softae/analysis/eis/measurability.py"
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=repo,
                               capture_output=True, timeout=60)
        blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=repo,
                              capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git on PATH
        pytest.skip("git unavailable")
    if dirty.returncode != 0 or blob.returncode != 0:  # pragma: no cover
        pytest.skip("engine not resolvable from HEAD")
    if dirty.stdout.strip():
        pytest.skip(f"{rel} is modified in the working tree; content differs from HEAD")

    import hashlib

    assert engine_fingerprint_full() == hashlib.sha256(blob.stdout).hexdigest()


def test_engine_fingerprint_full_unreadable_source_is_unknown_for_both_digests(
        monkeypatch, tmp_path):
    """Losing the fingerprint must be visible in both forms; ``"unknown"`` is not hex and
    so can never start-with its way past an expectation."""
    _point_engine_at(monkeypatch, tmp_path / "absent.py")
    assert engine_fingerprint_full() == "unknown"
    assert engine_byte_fingerprint_full() == "unknown"


# ── the retired byte digest: a stale citation is not a wrong engine ──────────
#
# ``70e94699b69f`` was published on the channel as the blessed value and is in the filename
# of a durable artifact, so it WILL be pasted. It is the byte digest of the same content, so
# reporting it as a bare MISMATCH would be a false alarm of exactly the kind normalising the
# digest was meant to remove — and a guard that cries wolf about a correct engine is a guard
# that stops being armed.


def _retired_byte_prefix() -> str:
    return engine_byte_fingerprint_full()[:12]


def test_engine_sha_check_retired_byte_digest_is_its_own_label_not_a_mismatch():
    check = EngineShaCheck(actual="a" * 64, expected="b" * 12, actual_bytes="b" * 64)

    assert check.matched is False
    assert check.matched_bytes_only is True
    assert check.label == ENGINE_SHA_BYTE_MATCH_LABEL
    assert check.label in ENGINE_SHA_MATCH_LABELS


def test_engine_sha_check_content_match_does_not_route_through_the_byte_branch():
    """On a checkout where the two digests coincide — ``core.autocrlf=false``, or a file
    that was always LF — the plain ``match`` must answer, or every such caller would be told
    they had pasted a retired value they never pasted."""
    same = "c" * 64
    check = EngineShaCheck(actual=same, expected=same[:12], actual_bytes=same)

    assert check.matched is True
    assert check.matched_bytes_only is False
    assert check.label == "match"


def test_engine_sha_check_unknown_byte_digest_cannot_manufacture_a_byte_match():
    lost = EngineShaCheck(actual="unknown", expected="70e94699b69f",
                          actual_bytes="unknown")
    assert lost.matched_bytes_only is False
    assert lost.label == "MISMATCH"


def test_sweep_retired_byte_digest_is_explained_rather_than_bannered(
        tmp_path, calibration_root):
    """The pasted pre-change value, end to end.

    The message has to do three things a bare MISMATCH does not: say the value is the
    *byte* digest of this same content, print the content digest to use instead, and refrain
    from claiming the numbers are incomparable — because they are not.
    """
    retired = _retired_byte_prefix()
    if retired == engine_fingerprint()[:12]:  # pragma: no cover - LF checkout
        pytest.skip("this checkout stores the engine LF; the two digests coincide")

    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha=retired)
    (row,) = result.rows
    text = sweep_summary(result, {})

    assert result.engine_sha_check.matched_bytes_only is True
    assert row.engine_sha_match == ENGINE_SHA_BYTE_MATCH_LABEL
    assert row.engine_sha_expected == retired
    # Region-scoped to the engine block: the module docstring is not under test and the
    # summary says "content digest" in its header line too, so a whole-text ``in`` would be
    # satisfied by a block that had lost its explanation entirely.
    block = text.split("=" * 78)[2].split("Skipped (every row")[0]
    assert "RETIRED BYTE DIGEST" in block
    assert f"pre-{BYTE_DIGEST_RETIRED_ON} BYTE digest of this same content" in block
    assert engine_fingerprint_full() in block
    # The thing it must NOT say: this is not a different engine and the numbers stand.
    assert "WRONG ENGINE" not in text
    assert "NOT comparable" not in text


def test_sweep_retired_byte_digest_changes_no_computed_value(
        tmp_path, calibration_root):
    """Reporting only, in the new branch as in the old one."""
    retired = _retired_byte_prefix()
    if retired == engine_fingerprint()[:12]:  # pragma: no cover - LF checkout
        pytest.skip("this checkout stores the engine LF; the two digests coincide")

    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    (plain,) = sweep(db, project_dir, calibration_root=calibration_root).rows
    (stale,) = sweep(db, project_dir, calibration_root=calibration_root,
                     expect_engine_sha=retired).rows

    scalars = [f for f in CSV_COLUMNS if not f.startswith("engine_sha_")]
    assert [getattr(stale, f) for f in scalars] == [getattr(plain, f) for f in scalars]


def test_sweep_genuine_mismatch_states_the_byte_digest_was_checked_too(
        tmp_path, calibration_root):
    """A real wrong engine must still be shouted at — and must say the retired form was
    ruled out, so a reader cannot wonder whether the banner simply had not heard of it."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha="0" * 40)
    text = sweep_summary(result, {})
    banner = text.split("*" * 78)[1]

    assert result.engine_sha_check.matched_bytes_only is False
    assert "WRONG ENGINE" in banner
    assert engine_byte_fingerprint_full() in banner
    assert "different engine, not a different way of writing the same one" in banner


def test_sweep_summary_header_says_the_digest_is_content_normalised(
        tmp_path, calibration_root):
    """Wherever the digest is printed, it says which digest it is — otherwise a reader
    compares it against the published 70e94699b69f and concludes the engine changed when
    only the way of naming it did."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    header = sweep_summary(sweep(db, project_dir,
                                 calibration_root=calibration_root), {}).split("=" * 78)[1]

    assert engine_fingerprint() in header
    assert "content digest, line-ending normalised" in header


def test_sweep_matching_content_digest_says_which_digest_matched(
        tmp_path, calibration_root):
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    text = sweep_summary(sweep(db, project_dir, calibration_root=calibration_root,
                               expect_engine_sha=engine_fingerprint_full()), {})
    block = text.split("=" * 78)[2].split("Skipped (every row")[0]

    assert "Expected engine sha: MATCH" in block
    assert "content digest (line-ending normalised)" in block


# ── --expect-engine-sha: the fingerprint is not enough on its own ────────────
#
# ``engine_sha`` detects that two runs are INCOMMENSURABLE. It cannot detect that ONE run
# is wrong: a sweep against a mutated engine is fingerprinted, internally consistent,
# complete and entirely wrong, carrying the digest of a file state that never legitimately
# existed. The 25x move in median S1 lift was caught only because the sweep happened to be
# run twice. These tests are about the case where it is run once.


def test_engine_fingerprint_full_is_the_digest_the_short_column_abbreviates():
    """The comparison is against the full 64 chars, so an expected prefix longer than the
    12-char column still discriminates rather than silently truncating to it."""
    full = engine_fingerprint_full()
    assert len(full) == 64
    assert set(full) <= set("0123456789abcdef")
    assert engine_fingerprint() == full[:12]


def test_sweep_matching_full_digest_reports_a_match(tmp_path, calibration_root):
    """The whole 64-char digest, which is what a scripted caller would pin."""
    full = engine_fingerprint_full()
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha=full)
    (row,) = result.rows

    assert result.engine_sha_check.matched is True
    assert row.engine_sha_match == "match"
    assert row.engine_sha_expected == full
    text = sweep_summary(result, {})
    assert "WRONG ENGINE" not in text
    assert "Expected engine sha: MATCH" in text


def test_sweep_matching_prefix_reports_a_match(tmp_path, calibration_root):
    """The channel cites the engine abbreviated — the value published there was
    ``70e94699b69f...`` — so a prefix has to be accepted, in the mixed case and whitespace
    it gets pasted in."""
    prefix = engine_fingerprint_full()[:ENGINE_SHA_MIN_PREFIX].upper()
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha=f"  {prefix}  ")
    (row,) = result.rows

    assert row.engine_sha_match == "match"
    assert row.engine_sha_expected == prefix.lower()
    assert "WRONG ENGINE" not in sweep_summary(result, {})


def test_sweep_mismatched_digest_banners_and_names_both_digests(
        tmp_path, calibration_root):
    """The banner has to say the words, above every count.

    Not a flag among other flags: the report below it is entirely normal — every column
    present, every provenance label right, the scalar-health block clean — because the
    numbers really were computed, just by the wrong code. So the banner names both digests
    and states the incomparability, and it does so ahead of the first count, since a reader
    who has already read the counts has already begun believing them.
    """
    wrong = "0" * 40
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha=wrong)
    (row,) = result.rows
    text = sweep_summary(result, {})

    # The corpus is deliberately healthy: every scalar column carries values, so nothing
    # below the banner looks wrong. That is the situation the banner exists for.
    assert "Scalar column health — every column carries values:" in text
    assert row.engine_sha_match == "MISMATCH"
    assert row.engine_sha_expected == wrong
    assert "WRONG ENGINE" in text
    assert wrong in text and engine_fingerprint_full() in text
    assert "NOT comparable" in text
    # Above every count, and above the scalar-health verdict — which is itself a verdict
    # about the engine, so it must not be read before the engine's identity is settled.
    assert text.index("WRONG ENGINE") < text.index("Scalar column health")
    assert text.index("WRONG ENGINE") < text.index("Skipped (every row accounted for")


def test_sweep_banner_justifies_itself_with_the_mutated_engine_not_an_edit(
        tmp_path, calibration_root):
    """The banner's *reason* is asserted, not just its headline.

    The banner is printed to every operator and saved into every summary, so whatever
    story it tells propagates by the very mechanism built to stop bad numbers spreading.
    The 90.8 was produced by a transiently mutated engine, **not** by an earlier version of
    this module, and the causal reading ("an edit to this engine moved it") is formally
    withdrawn. Asserting a stable substring of the justification — here and in the header
    line — is what stops it rotting back.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    text = sweep_summary(sweep(db, project_dir, calibration_root=calibration_root,
                               expect_engine_sha="0" * 40), {})
    # Region-scoped, because the two sites tell the same story and a whole-text ``in``
    # is satisfied by either — which is how the first version of this test passed a
    # mutation that had emptied the banner.
    header = text.split("=" * 78)[1]
    banner = text.split("*" * 78)[1]

    assert "transiently mutated" in header
    assert "transiently mutated engine" in banner
    assert "90.8" in banner and "3.62" in banner
    # Why a recorded fingerprint is not enough: one run cannot audit itself.
    assert "second run happened to disagree" in banner
    # The withdrawn causal claim, in the forms it was printed in.
    assert "an edit to this engine" not in text
    assert "an edit to it" not in text


def test_sweep_mismatched_digest_changes_no_computed_value(tmp_path, calibration_root):
    """The guard reports; it does not gate. A sweep against an unblessed engine is
    legitimate and sometimes the intent, so every scalar must be bit-identical to the same
    sweep run with no expectation at all — otherwise the guard would be quietly deciding
    which engines are allowed to produce numbers."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    (plain,) = sweep(db, project_dir, calibration_root=calibration_root).rows
    (guarded,) = sweep(db, project_dir, calibration_root=calibration_root,
                       expect_engine_sha="0" * 40).rows

    scalars = [f for f in CSV_COLUMNS if not f.startswith("engine_sha_")]
    assert [getattr(guarded, f) for f in scalars] == [getattr(plain, f) for f in scalars]
    assert guarded.engine_sha_match == "MISMATCH" and plain.engine_sha_match \
        == "not_checked"


def test_sweep_mismatched_digest_raises_nothing_and_main_still_exits_zero(
        tmp_path, calibration_root, capsys):
    """End to end through the CLI: the banner prints and the process still succeeds.

    An unblessed engine is not a failure, so a non-zero exit here would train the operator
    to pass ``--expect-engine-sha`` only when they already know it will match — which is
    every case except the one it exists for.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    code = main(["--project-dir", str(project_dir), "--db", str(db),
                 "--calibration-root", str(calibration_root),
                 "--out", str(tmp_path / "out.csv"),
                 "--expect-engine-sha", "0" * 40])

    assert code == 0
    assert "WRONG ENGINE" in capsys.readouterr().out


def test_sweep_omitted_expectation_states_it_was_not_checked(
        tmp_path, calibration_root):
    """Not-checked is stated, never implied.

    Same distinction this module draws everywhere else — measured-and-absent versus not
    measured at all. A guard that is silent when unarmed is indistinguishable from one that
    passed, and the CSV must carry that difference too, because it outlives the console it
    was printed to.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    result = sweep(db, project_dir, calibration_root=calibration_root)
    (row,) = result.rows
    text = sweep_summary(result, {})

    assert result.engine_sha_check.checked is False
    assert row.engine_sha_match == "not_checked"
    assert row.engine_sha_expected == ""
    assert "Expected engine sha: NOT CHECKED" in text
    assert "UNKNOWN, not confirmed" in text
    assert "WRONG ENGINE" not in text


@pytest.mark.parametrize("short", ["70e9", "7", "0" * 7])
def test_sweep_expected_prefix_below_the_minimum_is_rejected(
        tmp_path, calibration_root, short):
    """Refused at the door rather than matched loosely.

    ``70e9`` is a genuine prefix of the blessed digest, so accepting it would *pass* — and
    would collide with roughly one engine state in 65 536 by accident. A guard satisfiable
    by chance reports its own tolerance, not the engine's identity. This is the one place
    the feature refuses anything, and it refuses before any work is done.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    with pytest.raises(ValueError, match="hex chars"):
        sweep(db, project_dir, calibration_root=calibration_root,
              expect_engine_sha=short)


def test_normalize_engine_sha_short_prefix_of_the_real_digest_is_still_rejected():
    """The discriminating half of the rule: length, not content. The rejected value here
    is a true prefix of the engine actually imported, so a guard that compared first and
    validated second would accept it."""
    real = engine_fingerprint_full()
    assert normalize_engine_sha(real[:ENGINE_SHA_MIN_PREFIX]) == \
        real[:ENGINE_SHA_MIN_PREFIX]
    with pytest.raises(ValueError, match=str(ENGINE_SHA_MIN_PREFIX)):
        normalize_engine_sha(real[:ENGINE_SHA_MIN_PREFIX - 1])


@pytest.mark.parametrize("bad", ["70e94699b69f...", "70e94699b69g", "not hex at all",
                                 "", "   ", "a" * 65])
def test_normalize_engine_sha_malformed_input_is_rejected_with_a_reason(bad):
    """Including the ellipsis the digest is usually pasted with: a clear refusal beats
    silently stripping decoration off something the caller may have mistyped."""
    with pytest.raises(ValueError):
        normalize_engine_sha(bad)


def test_expected_sha_arg_reports_the_reason_argparse_would_otherwise_swallow():
    """argparse prints only ``invalid value`` for a ``ValueError``, which would reduce a
    stated minimum to a bare refusal."""
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="at least"):
        _expected_sha_arg("70e9")


def test_engine_sha_check_unreadable_engine_is_a_mismatch_not_a_pass():
    """``engine_fingerprint_full`` degrades to ``"unknown"`` when the source cannot be
    read. That must not start-with its way past an expectation: losing the fingerprint is
    losing the answer, and the answer's absence is not a match."""
    lost = EngineShaCheck(actual="unknown", expected="70e94699b69f")

    assert lost.short == "unknown"
    assert lost.matched is False
    assert lost.label == "MISMATCH"


def test_sweep_csv_carries_the_engine_sha_outcome_beside_the_digest(
        tmp_path, calibration_root):
    """The outcome is in the file, not only on stdout.

    A CSV read next month is detached from the console it was printed with, and the whole
    point of the guard is the run nobody thought to repeat. Columns sit beside
    ``engine_sha`` so the digest and the verdict on it cannot be separated by a column
    selection.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")])
    result = sweep(db, project_dir, calibration_root=calibration_root,
                   expect_engine_sha="0" * 40)
    out = write_csv(result.rows, tmp_path / "sweep.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        record = next(iter(csv.DictReader(fh)))

    assert record["engine_sha_expected"] == "0" * 40
    assert record["engine_sha_match"] == "MISMATCH"
    header = list(CSV_COLUMNS)
    assert header[header.index("engine_sha") + 1: header.index("engine_sha") + 3] == \
        ["engine_sha_expected", "engine_sha_match"]


def test_sweep_limit_stops_after_n_rows(tmp_path, calibration_root):
    db, project_dir = _make_store(
        tmp_path, [("@", {}, None, "full", "c") for _ in range(4)])
    assert len(sweep(db, project_dir,
                     calibration_root=calibration_root, limit=2).rows) == 2


def test_open_readonly_connection_refuses_a_write(tmp_path):
    """The read-only guarantee, asserted rather than intended. ``mode=ro`` is a URI flag,
    so a malformed scheme silently yields a writable connection."""
    db, _ = _make_store(tmp_path, [])
    conn = _open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO experiments(run_id, started_at, workflow_name)"
                         " VALUES ('x','y','z')")
    finally:
        conn.close()


def test_open_readonly_missing_store_raises_rather_than_creating_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        _open_readonly(tmp_path / "absent.db")


# ── analytic values: the numbers, not the frame around them ──────────────────
#
# Everything above this line survives an engine that computes nothing. Everything below
# it does not. Each fixture is constructed in admittance so its S1/S2/S3 answers are
# known in closed form before the sweep runs, and the assertions are on those answers.

#: The S1 fixture: ``Im(Y)/ω`` is 20x its plateau below 100 Hz and flat above it. The
#: plateau band spans 3.0 decades against the low band's 1.875, so the plateau search
#: selects the upper run and the lift is exactly the ratio of the two constants.
S1_PLATEAU_C_F = 1.0e-10
S1_LOW_C_F = 2.0e-9
S1_CORNER_HZ = 1.0e2

#: The S2 fixture: tan δ held at 0.5 across the band with one deliberately low-loss point,
#: so ``argmin(tan δ)`` — and therefore the impedance the table is queried at — is known.
S2_INDEX = 10
S2_MIN_TAND = 0.05
S2_BAND_TAND = 0.5
S2_C_F = 1.0e-10

#: The S3 fixture: four points given a negative real admittance, so ``Re Z < 0`` there.
S3_N_NEGATIVE = 4


def _excursion_spectrum(path, *, channel: int = 19) -> None:
    C = np.where(F < S1_CORNER_HZ, S1_LOW_C_F, S1_PLATEAU_C_F)
    _write_spectrum(path, F, _z_from_admittance(F, 1.0e-12, C), channel=channel)


def _flat_spectrum(path, *, channel: int = 19) -> None:
    _write_spectrum(path, F, _z_from_admittance(F, 1.0e-12, S1_PLATEAU_C_F),
                    channel=channel)


def _s2_spectrum(path, *, channel: int = 19) -> None:
    tand = np.full(F.size, S2_BAND_TAND)
    tand[S2_INDEX] = S2_MIN_TAND
    Z = _z_from_admittance(F, tand * 2.0 * np.pi * F * S2_C_F, S2_C_F)
    _write_spectrum(path, F, Z, channel=channel)


def _s3_spectrum(path, *, channel: int = 19) -> None:
    G = np.full(F.size, 1.0e-12)
    G[:S3_N_NEGATIVE] = -1.0e-12
    _write_spectrum(path, F, _z_from_admittance(F, G, S1_PLATEAU_C_F), channel=channel)


def _s2_expected() -> tuple[float, float, float]:
    """``(z_at_min, eps_deg, margin)`` for :func:`_s2_spectrum`, in closed form.

    Computed from the fixture's own construction constants and the table's own points —
    ``|Y| = ωC·sqrt(1 + tan²δ)`` and a log-interpolation between the two bracketing
    entries — rather than from anything the engine returns. Hand-deriving the denominator
    is the point: reading ``epsilon_deg`` here would let the table and the sweep agree
    with each other while both being wrong.
    """
    z = 1.0 / (2.0 * math.pi * F[S2_INDEX] * S2_C_F
               * math.sqrt(1.0 + S2_MIN_TAND ** 2))
    lo_z, hi_z = S2_TABLE_Z_OHM[0], S2_TABLE_Z_OHM[1]
    lo_e, hi_e = S2_TABLE_EPS_DEG[0], S2_TABLE_EPS_DEG[1]
    span = math.log10(hi_z) - math.log10(lo_z)
    eps = lo_e + (math.log10(z) - math.log10(lo_z)) / span * (hi_e - lo_e)
    return z, eps, S2_MIN_TAND / math.tan(math.radians(eps))


def test_sweep_s1_lift_is_the_constructed_excursion_factor(tmp_path, calibration_root):
    """The sweep must emit 20x, not merely a float-shaped column.

    ``Im(Y)/ω`` is built as exactly ``2 nF`` below the corner and ``100 pF`` above it, so
    S1's lift is the ratio of two constants this test wrote. An engine returning NaN, or
    one computing ``C_app = 1/(ω|Z''|)`` instead — which differs by ``1 + tan²δ`` — fails
    here rather than producing a well-formed frame of the wrong numbers.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.n_points == F.size
    assert row.s1_outcome == "excursion"
    assert row.s1_judgeable is True
    assert row.s1_lift == pytest.approx(S1_LOW_C_F / S1_PLATEAU_C_F, rel=1e-9)
    # The plateau is the lowest capacitance in the spectrum, so nothing dips below it.
    assert row.s1_below_plateau_depth == pytest.approx(0.0, abs=1e-9)


def test_sweep_plateau_columns_are_the_constructed_plateau(tmp_path, calibration_root):
    """S1's denominator and its band, asserted as values.

    The plateau columns are what make a lift interpretable — a 20x lift over a plateau of
    the wrong capacitance, or over a band the search picked from the low side, is a
    different claim. All four are known here because the fixture set them.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.plateau_C_F == pytest.approx(S1_PLATEAU_C_F, rel=1e-9)
    assert row.plateau_lo_hz == pytest.approx(S1_CORNER_HZ, rel=1e-9)
    assert row.plateau_hi_hz == pytest.approx(F[-1], rel=1e-9)
    assert row.plateau_decades == pytest.approx(3.0, abs=1e-9)
    assert row.plateau_wide_enough is True


def test_sweep_flat_spectrum_reports_no_low_band_rather_than_a_flat_verdict(
        tmp_path, calibration_root):
    """A wholly flat spectrum has no below-plateau region, so S1 was **not measured**.

    Distinct from the excursion case by *outcome*, not merely by a NaN: an engine that
    lost its numbers reports ``no_plateau`` here, and an engine that reported this as
    ``flat`` would be stating an absence it never looked for.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_flat_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.s1_outcome == "no_low_band"
    assert row.s1_judgeable is False
    assert row.s1_lift != row.s1_lift
    assert row.plateau_C_F == pytest.approx(S1_PLATEAU_C_F, rel=1e-9)
    assert row.plateau_decades == pytest.approx(5.0, abs=1e-9)


def test_sweep_s2_margin_is_the_analytic_ratio_at_the_analytic_impedance(
        tmp_path, s2_calibration_root):
    """S2's three reported numbers, each against a hand-derived value.

    ``f_at_min`` pins which point was chosen, ``z_at_min`` pins the impedance the table
    was queried at, and ``margin`` pins the ratio — so a sweep that queried the table at
    the wrong point, or collapsed it to its lowest-|Z| entry the way
    ``CalibrationSet.envelope()`` would, fails on a number rather than passing on a shape.
    """
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_s2_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=s2_calibration_root).rows
    z_expected, eps_expected, margin_expected = _s2_expected()

    assert row.phase_table == "mux16(sole-fallback)"
    assert row.s2_provisional is False
    assert row.eps_clamped is False
    assert row.s2_f_at_min_hz == pytest.approx(F[S2_INDEX], rel=1e-12)
    assert row.s2_z_at_min_ohm == pytest.approx(z_expected, rel=1e-9)
    assert row.s2_eps_deg == pytest.approx(eps_expected, rel=1e-9)
    assert row.s2_margin == pytest.approx(margin_expected, rel=1e-9)
    # Pinned literally as well, so a coherent-but-wrong rewrite of the closed form above
    # cannot move the expectation with the implementation.
    assert row.s2_margin == pytest.approx(1.8233586891, rel=1e-9)
    # And the number the lowest-|Z| envelope would have produced is a different one.
    assert row.s2_margin != pytest.approx(
        S2_MIN_TAND / math.tan(math.radians(S2_TABLE_EPS_DEG[0])), rel=1e-3)


def test_sweep_s3_counts_the_constructed_negative_points_exactly(
        tmp_path, calibration_root):
    """S3's count and fraction, against a spectrum built with a known number of them."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_s3_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.s3_n_negative == S3_N_NEGATIVE
    assert row.s3_frac == pytest.approx(S3_N_NEGATIVE / F.size, rel=1e-12)
    assert row.s3_re_state == "unverified"


def test_sweep_s3_is_zero_on_a_wholly_passive_spectrum(tmp_path, calibration_root):
    """The negative control for the test above: without it, an S3 that counted every
    point would satisfy neither and an S3 that counted none would satisfy the fraction
    only by coincidence."""
    db, project_dir = _make_store(tmp_path, [("@", {}, None, "full", "c")],
                                  spectrum=_excursion_spectrum)
    (row,) = sweep(db, project_dir, calibration_root=calibration_root).rows

    assert row.s3_n_negative == 0
    assert row.s3_frac == 0.0


# ── the all-NaN column guard ─────────────────────────────────────────────────

def test_sweep_summary_healthy_corpus_reports_no_all_nan_column(
        tmp_path, calibration_root):
    """The assertion that would have caught the all-NaN engine: over a corpus where all
    three scalars are computable, none of the three columns may be empty.

    This is the guard's falsifiable half. Without it the banner below could fire on
    everything and still pass its own test — the shape ``SUBAGENT_RULES`` §3 calls a check
    that cannot fail.
    """
    db, project_dir = _make_store(
        tmp_path, [("@", {}, None, "full", "c") for _ in range(3)],
        spectrum=_excursion_spectrum)
    result = sweep(db, project_dir, calibration_root=calibration_root)
    health = {h.column: h for h in scalar_column_health(result.rows)}

    assert len(result.rows) == 3
    assert health["s1_lift"].n_absent == 0
    assert health["s2_margin"].n_absent == 0
    assert health["s3_frac"].n_absent == 0
    assert not any(h.alarming for h in health.values())

    text = sweep_summary(result, {})
    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" not in text
    assert "Scalar column health — every column carries values:" in text


def test_sweep_summary_all_nan_scalar_column_is_announced_as_a_suspected_failure():
    """An all-NaN engine writes a perfect CSV, so the summary has to say the words.

    Not a NaN count among other counts: the banner names the columns and says the frame
    is suspect, because every other block in the report looks entirely normal when this
    happens.
    """
    dead = dict(s1_outcome="no_plateau", s1_judgeable=False, s1_lift=float("nan"),
                s2_margin=float("nan"), s2_provisional=True, s3_frac=float("nan"))
    result = SweepResult(
        rows=[_row(measurement_id=i, **dead) for i in range(1, 4)], n_considered=3)
    text = sweep_summary(result, {})

    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" in text
    for column in ("s1_lift", "s2_margin", "s3_frac"):
        assert f"{column:<12} absent in 3 of 3 rows (100.0%)" in text
    assert "measurement of this tool" in text


def test_sweep_summary_banner_fires_on_one_dead_column_among_healthy_ones():
    """Per column, not per frame: a dead S3 beside a live S1 and S2 is still dead."""
    result = SweepResult(
        rows=[_row(measurement_id=i, s3_frac=float("nan")) for i in range(1, 4)],
        n_considered=3)
    text = sweep_summary(result, {})

    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" in text
    assert "Above the 98% absence alarm: s3_frac" in text
    assert "s1_lift      absent in 0 of 3 rows" in text


def test_scalar_column_health_a_minority_of_absent_values_is_not_an_alarm():
    """Unjudgeable S1 rows are a normal population fact — both real commissioning blanks
    are judgeable, but a corpus containing flat spectra will carry some. The alarm is
    about a column with nothing in it, not about a column with gaps."""
    rows = [_row(measurement_id=1, s1_judgeable=False, s1_lift=float("nan"),
                 s1_outcome="no_plateau")] + [_row(measurement_id=i) for i in range(2, 6)]
    health = {h.column: h for h in scalar_column_health(rows)}

    assert health["s1_lift"].n_absent == 1
    assert health["s1_lift"].frac_absent == pytest.approx(0.2)
    assert not health["s1_lift"].alarming


def test_scalar_column_health_uncovered_calibration_explains_a_nan_margin_column():
    """Measured-and-absent, not not-measured-at-all.

    When no calibration matched any row, ``phase_table='none'`` is in the same frame and
    says why every margin is NaN. That is the calibration set's answer, not the engine's,
    so it is reported with its explanation and does not raise the banner — while S1 and S3
    in the same frame remain fully alarmable.
    """
    rows = [_row(measurement_id=i, phase_table="none", s2_margin=float("nan"),
                 s2_provisional=True, s2_eps_deg=float("nan")) for i in range(1, 4)]
    health = {h.column: h for h in scalar_column_health(rows)}

    assert health["s2_margin"].n_absent == 3
    assert health["s2_margin"].explanation
    assert not health["s2_margin"].alarming
    text = sweep_summary(SweepResult(rows=rows, n_considered=3), {})
    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" not in text
    assert "no calibration covered ANY swept row" in text


def test_scalar_column_health_a_partially_covered_corpus_gets_no_excuse():
    """The explanation is earned by the whole frame, not by a majority of it: one row that
    *did* find a table means the NaNs are no longer attributable to the calibration set."""
    rows = [_row(measurement_id=1, phase_table="mux16"),
            *[_row(measurement_id=i, phase_table="none", s2_margin=float("nan"),
                   s2_provisional=True) for i in range(2, 4)]]
    (_, s2, _) = scalar_column_health(rows)

    assert s2.explanation == ""
    assert s2.n_absent == 2 and not s2.alarming        # 67 %, under the alarm fraction


def test_sweep_summary_with_no_rows_says_nothing_was_swept():
    """A zero-row sweep must not read as three healthy columns. An empty frame passes
    every all-NaN check vacuously, which is the alarm silencing itself."""
    text = sweep_summary(SweepResult(rows=[], n_considered=7), {})

    assert "NOTHING WAS SWEPT" in text
    assert "absence of alarms" in text
    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" not in text


def test_scalar_column_health_of_an_empty_frame_alarms_on_nothing():
    columns = scalar_column_health([])
    assert [h.column for h in columns] == ["s1_lift", "s2_margin", "s3_frac"]
    for health in columns:
        assert isinstance(health, ScalarColumnHealth)
        assert health.n_rows == 0
        assert not health.alarming
        assert health.frac_absent != health.frac_absent


# ── the same path, over a real stored commissioning blank ────────────────────
#
# The synthetic fixtures above are exact but are still fixtures. This section runs the
# whole sweep — sqlite row, path resolution, the committed mux16 calibration, the engine —
# over a spectrum measured on this rig, against values measured from it. It is the half
# that would notice a fixture living off the production manifold.

COMMISSIONING_DIR = Path(
    r"C:\Users\Osuji\Documents\Users\Pavel\EIS_capacitance_commissioning_data")
CH25_BLANK = COMMISSIONING_DIR / "ch25_manual_open_PCB_blank_RECEcoupled.txt"
REPO_CALIBRATION = Path(__file__).resolve().parents[1] / "calibration" / "eis"

requires_ch25_blank = pytest.mark.skipif(
    not (CH25_BLANK.is_file() and (REPO_CALIBRATION / "mux16.toml").is_file()),
    reason=f"commissioning blank or committed mux16 calibration absent "
           f"({COMMISSIONING_DIR})",
)


@requires_ch25_blank
def test_sweep_over_the_real_ch25_blank_emits_its_measured_scalars(tmp_path):
    """A healthy RE/CE-tied open PCB blank from 2026-08-06 — the source spectrum for the
    committed ``mux16.toml`` — swept end to end.

    The values are the ones measured from that file and asserted in the engine's own
    tests: lift 1.42x over a 2.49-decade plateau at 50 pF, tan δ margin 0.593 against an
    interpolated ε at 3.86x10^7 Ω, and 6 of 35 points outside the passive quadrant. The
    row is inserted with an **absolute** path, which is also how the eight absolute-path
    rows in the live store resolve.
    """
    db, project_dir = _make_store(
        tmp_path, [(str(CH25_BLANK), {}, "mux16", "manual", "commissioning")])
    (row,) = sweep(db, project_dir, calibration_root=REPO_CALIBRATION).rows

    assert row.n_points == 35
    assert row.phase_table == "mux16"

    assert row.s1_outcome == "excursion"
    assert row.s1_lift == pytest.approx(1.4192, rel=1e-3)
    assert row.plateau_C_F == pytest.approx(5.0247e-11, rel=1e-3)
    assert row.plateau_decades == pytest.approx(2.4877, rel=1e-3)

    assert row.s2_provisional is False
    assert row.eps_clamped is False
    assert row.s2_z_at_min_ohm == pytest.approx(3.86003e7, rel=1e-4)
    assert row.s2_eps_deg == pytest.approx(1.35166, rel=1e-4)
    assert row.s2_margin == pytest.approx(0.59348, rel=1e-4)

    assert row.s3_n_negative == 6
    assert row.s3_frac == pytest.approx(6 / 35, rel=1e-9)


@requires_ch25_blank
def test_sweep_summary_over_a_real_blank_raises_no_all_nan_alarm(tmp_path):
    """The guard's negative control on real data: a spectrum the rig actually produced
    fills all three scalar columns, so the banner must stay silent."""
    db, project_dir = _make_store(
        tmp_path, [(str(CH25_BLANK), {}, "mux16", "manual", "commissioning")])
    result = sweep(db, project_dir, calibration_root=REPO_CALIBRATION)

    assert all(h.n_absent == 0 for h in scalar_column_health(result.rows))
    assert "SUSPECTED ENGINE OR PLUMBING FAILURE" not in sweep_summary(result, {})


# ── output ───────────────────────────────────────────────────────────────────

def _row(**overrides) -> SweepRow:
    base = dict(
        measurement_id=1, run_id="r", channel=19, role="sample", fixture_id="",
        modality="eis", re_connection="unverified", timestamp="t", campaign="c",
        workflow_mode="full", workflow_name="wf", provenance="undeclared",
        phase_table="mux16", engine_sha="deadbeef0000",
        engine_sha_expected="", engine_sha_match="not_checked",
        n_points=30, s1_outcome="excursion", s1_lift=90.0,
        s1_judgeable=True, s1_below_plateau_depth=0.3, plateau_C_F=4e-10,
        plateau_lo_hz=100.0, plateau_hi_hz=1e4, plateau_decades=2.0,
        plateau_wide_enough=True, s2_margin=2.5, s2_provisional=False,
        s2_f_at_min_hz=800.0, s2_z_at_min_ohm=1.7e6, s2_eps_deg=1.9,
        eps_clamped=False, s3_n_negative=0, s3_frac=0.0, s3_re_state="unverified")
    base.update(overrides)
    return SweepRow(**base)


def test_write_csv_header_matches_declared_columns(tmp_path):
    out = write_csv([_row(), _row(measurement_id=2)], tmp_path / "s.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert len(rows) == 3


def test_write_csv_row_carries_provenance_before_scalars(tmp_path):
    """Risk 6's requirement is that provenance is in the frame from the first row, so the
    columns are checked for presence, not merely the file for existence."""
    out = write_csv([_row(provenance="mock_declared", fixture_id="mux16")],
                    tmp_path / "s.csv")
    with out.open(encoding="utf-8", newline="") as fh:
        record = next(iter(csv.DictReader(fh)))
    for column in ("measurement_id", "run_id", "channel", "role", "fixture_id",
                   "modality", "re_connection", "timestamp", "campaign",
                   "workflow_mode", "workflow_name", "provenance", "phase_table",
                   "eps_clamped"):
        assert record[column] != ""
    assert record["provenance"] == "mock_declared"


def test_sweep_summary_single_provenance_label_declares_not_applicable():
    """The degenerate cross-tab must announce its own inapplicability. A uniform table
    printed silently reads as 'no confounding detected' when the truth is 'no comparison
    was possible'."""
    result = SweepResult(rows=[_row(), _row(measurement_id=2)], n_considered=2)
    text = sweep_summary(result, {"mock_declared": 116, "real_declared": 10})

    assert "NOT APPLICABLE" in text
    assert "0 of 2 swept spectra are mock-sourced" in text
    assert "116 mock_declared" in text and "10 real_declared" in text


def test_sweep_summary_mixed_provenance_omits_not_applicable():
    """The positive half: when both labels are actually swept, the cross-tab is real and
    the banner must not fire — otherwise the banner would be unfalsifiable."""
    result = SweepResult(
        rows=[_row(provenance="mock_declared"),
              _row(measurement_id=2, provenance="undeclared")], n_considered=2)
    text = sweep_summary(result, {"mock_declared": 1})

    assert "NOT APPLICABLE" not in text
    assert "mock_declared" in text and "undeclared" in text


def test_sweep_summary_reports_every_outcome_including_unreached():
    """A branch reached by zero spectra is a finding, so it is printed as a zero rather
    than omitted from the table."""
    result = SweepResult(rows=[_row(s1_outcome="excursion")], n_considered=1)
    text = sweep_summary(result, {})

    for outcome in ("no_plateau", "no_low_band", "flat", "excursion"):
        assert outcome in text
    assert "reached by NO spectrum" in text


def test_sweep_summary_skip_reasons_all_listed_even_at_zero():
    result = SweepResult(rows=[_row()], n_considered=3)
    result.record_skip(2, "file_missing")
    result.record_skip(3, "no_file_path")
    text = sweep_summary(result, {})

    for reason in SKIP_REASONS:
        assert reason in text
    assert "TOTAL SKIPPED        2" in text


def test_sweep_result_record_skip_unknown_reason_raises():
    """Skip reasons are a closed vocabulary; a typo would otherwise vanish from the
    accounting that is supposed to be exhaustive."""
    with pytest.raises(ValueError):
        SweepResult().record_skip(1, "because")
