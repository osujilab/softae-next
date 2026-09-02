"""Offline, read-only sweep of S1/S2/S3 over every stored EIS spectrum.

Stage 2a of ``docs/SubAgent docs/measurability_scalars.md``. This is the instrument that
produces the *arming evidence* for S1/S2/S3, and it is deliberately the only thing it is:
it opens the DataStore ``mode=ro``, never writes a row, never returns a
:class:`~softae.analysis.eis.gates.GateResult`, and cannot change any verdict anywhere.
The scalars themselves are **not** reimplemented here —
:mod:`softae.analysis.eis.measurability` is the engine and this module is its harness.

Why offline rather than in-line (spec §8, Risk 3): ``_log_spectrum_metrics`` is
gated-path-only and ``[eis] engine`` ships ``"legacy"``, so an in-line emission would
produce **zero** rows under the shipped configuration. A sweep over stored spectra runs
regardless of engine and covers the whole store backwards rather than one campaign
forwards.


Provenance, and why every row carries it
========================================

`[a109]` #11 and spec Risk 6: *P4's gate verdict was 100 % collinear with mock-vs-real
provenance; a separation that tracks provenance is not a separation.* So provenance is a
column from the first row rather than an analysis added later.

**The rule, derived from the live store rather than assumed** (counts as of 2026-08-28,
3871 measurement rows):

=========================  =========================================================
label                      rule, and what it was measured to be
=========================  =========================================================
``mock_declared``          ``eis_params_json["eis_validation_mock"]`` is **true**.
                           116 rows — ``eis_validate`` campaigns ``rehearsal`` (48),
                           ``rehearsal-e5`` (48), ``mocktest-mintreat1`` (10),
                           ``trendpreview`` (10).
``real_declared``          that key is present and **false**. 10 rows, all
                           ``eis_validate:probe-3ch-v3``.
``simulated_run``          ``experiments.workflow_mode == 'simulation'``. **34 runs,
                           ZERO measurement rows** — see the warning below.
``undeclared``             the key is absent. 3745 rows: every spectrum-bearing row in
                           the store.
=========================  =========================================================

**Three things about this rule are load-bearing and none of them is tidy.**

**1. ``workflow_mode == 'simulation'`` is inert and is kept only so that its absence is
measurable.** The 34 ``bo_campaign`` runs carrying that mode produced **no measurement
rows at all** — they reference already-collected data. A provenance rule resting on it
can never fire on any row, which is ``SUBAGENT_RULES`` §3's fixture-off-the-production-
manifold exactly: a branch that discriminates perfectly and is never visited. The clause
stays because a reader must be able to see the zero, and :func:`sweep_summary` prints it.

**2. ``undeclared`` is not a synonym for real.** No acquisition path outside the
``eis_validate`` harness records a mock flag, so a GUI run against
:class:`~softae.drivers.mock_espico.MockESPico` would land in ``undeclared`` beside real
bench data and nothing in the store would say so. The label is named for what the store
actually asserts — nothing — rather than for what is probably true. Reading it as "real"
is the reader's inference, not this module's claim.

**3. Consequently the Risk 6 cross-tab is DEGENERATE on the swept population, and
:func:`sweep_summary` says so in those words.** Every row this sweep can process needs a
resolvable ``eis_file_path``; all 126 declared rows (116 mock + 10 real) have
``eis_file_path IS NULL`` and carry no spectrum file, so **the declared population and the
swept population are exactly disjoint**. A cross-tab over one label value would read as
"no confounding detected" when the truth is "no comparison was possible" — the wrong
answer wearing the safe answer's clothes. It is printed as ``NOT APPLICABLE`` with the
counts that justify the sentence.

*Recorded because it corrects a briefing:* the mock/real split inside the validation path
is **not** a campaign-name convention. It is an explicit typed boolean in the measurement's
own ``eis_params_json``, and it separates the 116 from the 10 exactly. It lives inside a
JSON blob rather than in a column, which is why it is easy to miss, but nothing here parses
a name to find it.


Path resolution, and the rows that do not resolve
=================================================

``eis_file_path`` is stored **relative to the project directory** unless already absolute,
which is :func:`softae.tools.commission._load_role_spectra`'s convention. 8 rows hold an
absolute path; 20 rows do not resolve on disk and are **counted, never silently dropped**
— 19 under run directories that no longer exist, and measurement 3494 pointing into a dead
agent scratchpad, which is the row ``SUBAGENT_RULES`` §8 was written about.


The phase table, and the fallback that had to be stated
=======================================================

S2's denominator is ``PhaseAccuracyTable.epsilon_deg(|Z|)``, queried **per point at that
point's own |Z|** (spec §5, three hard requirements). Tables are loaded from the committed
``calibration/eis/*.toml`` by :func:`softae.analysis.eis.calibration.load_calibration`.

**The fallback rule, stated because 3727 of 3745 spectrum rows have ``fixture_id IS NULL``
and would otherwise silently get no table at all:** a row naming a fixture uses that
fixture's table; a row naming none inherits the *sole* calibration on disk when there is
exactly one, and gets no table when there are none or several. Which table qualified each
row is recorded per row in ``phase_table`` — ``mux16``, ``mux16(sole-fallback)`` or
``none`` — so a reviewer can separate the two populations after the fact rather than
having to trust the rule.

Staleness is deliberately **not** applied: :func:`resolve_calibration` would drop the table
when ``hardware_hash`` has moved, which is right for a gate and wrong for a descriptive
sweep — dropping it would replace a stated caveat with a NaN. The sweep reports; it does
not qualify.

``eps_clamped`` is carried alongside ``eps_deg`` because a finite ε is not proof the
impedance was characterised: inside ``valid_decades`` and past the table's last point,
``np.interp`` clamps to the endpoint (see
:func:`~softae.analysis.eis.measurability.eps_is_clamped`). NaN ε means the table refused;
the margin is NaN with it and the row is **provisional, never a pass**.


Every row names the engine that produced it
===========================================

``engine_sha`` is a digest of :mod:`softae.analysis.eis.measurability`'s own **content** —
the source bytes with line endings normalised to LF before hashing — and it is a column
rather than a footnote because **a distribution is a joint fact about the corpus and the
exact engine state that read it**. Two sweeps of the identical 3725 spectra
nine minutes apart returned ``flat`` 288 → 465, ``excursion`` 2933 → 2756 and median S1
lifts of **90.8 and 3.62** — a 25× move in the headline statistic without a single byte of
stored data changing. The cause, established afterwards, was that the earlier sweep had run
inside a *transient in-place mutation of the engine* during another session's audit. The
90.8 is therefore a fact about a mutated engine, **not** about this module at any point in
its own history, and it must never be cited as a version-to-version difference.

``SUBAGENT_RULES`` §8: *a stored number is a claim about the code that produced it.* An
S1 distribution cited in an arming discussion is worthless without the engine state it was
computed under, and two CSVs concatenated without this column are silently incommensurable.

**But a fingerprint that is merely recorded detects only that two runs are incommensurable;
it cannot detect that ONE run is wrong.** Nothing in the mutated run's output distinguished
it from a good one: the distribution was plausible, internally consistent and complete, and
the 25× discrepancy surfaced only because a second run happened to disagree. A single sweep
against a mutated engine yields a CSV that is fingerprinted, internally consistent,
complete and entirely wrong, carrying the digest of a file state that never legitimately
existed and with nothing to compare against. So the fingerprint has to be **checked against
a digest believed good**, not merely written down.

So ``--expect-engine-sha`` lets the caller name the digest it *believes* it is sweeping
under, and :class:`EngineShaCheck` reports whether that belief held. A prefix of at least
``ENGINE_SHA_MIN_PREFIX`` hex characters is accepted, because the digest is cited in
abbreviated form; anything shorter is **refused rather than matched loosely**, since a
4-char prefix matches roughly one engine state in 65 536 by accident and a guard that can
be satisfied by chance is not a guard.

The check **reports and never refuses**: no exception, no non-zero exit, no changed value.
Sweeping an unblessed engine is legitimate and is sometimes exactly the intent. The single
requirement is that it can never be *silently indistinguishable* from a blessed one — and
that the not-checked case is equally visible, which is why omitting the argument prints
``NOT CHECKED`` and writes ``not_checked`` to the CSV rather than leaving a blank. Same
distinction as everywhere else in this module: **measured-and-absent versus not measured at
all**, never a silence that reads as a pass.


Why the digest normalises line endings — and what the retired one was
====================================================================

Until 2026-08-31 this module hashed the source file's bytes **as they sat in the working
tree**, which on a machine with ``core.autocrlf=true`` is not the content: this repository
stores LF (see ``.gitattributes``) and checks CRLF out, so the same commit of the same
engine measured

===========================  ==========  ===============================================
form                         size        sha256
===========================  ==========  ===============================================
working tree (CRLF)          25 279 B    ``70e94699b69f65fd…``  <- the retired byte digest
HEAD blob / content (LF)     24 796 B    ``a7d0c8fbacae4b99…``  <- what is published now
===========================  ==========  ===============================================

**The retired digest named a local checkout rendering, not the content**, which is the
failure family the fingerprint exists to prevent — an identifier that looks like it names
content and names a rendering of it. It fails in both directions: a clone with
``core.autocrlf=false`` holds byte-identical *content* and would have been shouted at as a
WRONG ENGINE, and two machines could disagree about the digest of one commit, so the value
could never be the portable identity it was cited as. :func:`engine_fingerprint_full` now
normalises CRLF and lone CR to LF before hashing, which makes the digest equal on every
checkout of the same content — and, for an unmodified file, equal to the sha256 of the
``git show HEAD:…`` blob.

``70e94699b69f`` was published on the channel as the blessed value and is in the filename of
a durable artifact, so a caller **will** paste it. :func:`engine_byte_fingerprint_full`
computes the un-normalised digest as well, purely so that
:attr:`EngineShaCheck.matched_bytes_only` can recognise that specific paste and say what it
is — *the pre-2026-08-31 byte digest of this same content* — instead of reporting a bare
mismatch. A reader who cannot tell that case from a genuinely different engine will
distrust the guard, and a guard that is distrusted is not armed.


An all-NaN column is announced, not tallied
===========================================

A positive-control mutation — ``measurability.parallel_capacitance`` returning all NaN —
killed nine tests in the engine's own file and **none** here, because an all-NaN engine
produces a *well-formed frame full of NaN* and a shape check cannot see the difference.
The CSV has every column, every row, the right dtypes and the right provenance; only the
numbers are gone. That is the exact failure this sweep must not have, because the
distribution it produces is what every S1/S2/S3 arming threshold will be chosen from.

So :func:`scalar_column_health` asks, per scalar column, *how many rows had no value*, and
:func:`sweep_summary` prints a banner above everything else when a column is absent in
essentially all of them. The distinction it preserves is the one the whole package is
about: **measured-and-absent versus not measured at all.** ``UNJUDGEABLE_OUTCOMES`` already
draws that line for a single S1 row; this draws it for a whole column, where the answer is
never a property of the corpus. It reports — it raises nothing, gates nothing, and changes
no value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from softae.analysis.eis.measurability import (
    OUTCOMES,
    UNJUDGEABLE_OUTCOMES,
    conduction_lift,
    eps_is_clamped,
    negative_conductance_count,
    tand_margin,
)
from softae.analysis.eis.policy import RE_STATES

__all__ = [
    "BYTE_DIGEST_RETIRED_ON",
    "ENGINE_SHA_BYTE_MATCH_LABEL",
    "ENGINE_SHA_MATCH_LABELS",
    "ENGINE_SHA_MIN_PREFIX",
    "ENGINE_SHA_SHORT_LEN",
    "NAN_COLUMN_ALARM_FRAC",
    "PROVENANCE_LABELS",
    "SKIP_REASONS",
    "EngineShaCheck",
    "ScalarColumnHealth",
    "SweepRow",
    "SweepResult",
    "classify_provenance",
    "engine_byte_fingerprint_full",
    "engine_fingerprint",
    "engine_fingerprint_full",
    "normalize_engine_sha",
    "resolve_spectrum_path",
    "load_phase_tables",
    "scalar_column_health",
    "select_phase_table",
    "sweep",
    "sweep_summary",
    "write_csv",
    "main",
]

#: Provenance vocabulary, most specific first. See the module docstring for the evidence
#: behind each and for why ``undeclared`` is not a synonym for "real".
PROVENANCE_LABELS = ("mock_declared", "real_declared", "simulated_run", "undeclared")

#: Every reason a stored measurement is not swept. Each is counted and reported; none is
#: silent, because "the corpus was smaller than you think" is the failure that makes a
#: reachability count lie.
SKIP_REASONS = ("no_file_path", "file_missing", "unreadable", "no_finite_points")

#: The measurement's own mock declaration, inside ``eis_params_json``. A typed boolean,
#: not a naming convention — see the module docstring.
MOCK_FLAG_KEY = "eis_validation_mock"

#: Fraction of swept rows at or above which an absent scalar stops being a property of the
#: corpus and becomes a property of the code. Deliberately not 1.0: one degenerate but
#: readable spectrum in 3745 must not be able to silence the alarm for the other 3744.
NAN_COLUMN_ALARM_FRAC = 0.98

#: How much of the engine digest the ``engine_sha`` column carries. Unchanged from the
#: column's introduction; the *comparison* is against the full 64-char digest, so widening
#: an expected prefix past this stays meaningful.
ENGINE_SHA_SHORT_LEN = 12

#: Shortest expected prefix ``--expect-engine-sha`` will accept. Below this a prefix starts
#: matching engine states it was never meant to — 4 hex chars collide roughly once in
#: 65 536 — and a guard satisfiable by chance reports its own tolerance, not the engine.
ENGINE_SHA_MIN_PREFIX = 8

#: The day :func:`engine_fingerprint_full` stopped hashing working-tree bytes and started
#: hashing line-ending-normalised content. Named rather than inlined because it is quoted in
#: the one message a caller pasting the retired digest will read.
BYTE_DIGEST_RETIRED_ON = "2026-08-31"

#: The label for *matched the retired byte digest, not the content digest*. Written in the
#: ``mux16(sole-fallback)`` style already used by ``phase_table``: the answer, plus the
#: qualification that earned it. It is a **match** — a byte-digest hit proves the working
#: tree holds exactly those bytes, which is a stronger identity than the content digest, not
#: a weaker one — so shouting MISMATCH at it would be the false alarm this change removes;
#: but it is not spelled ``match`` either, because a CSV read next month, detached from the
#: console that explained it, would then show ``engine_sha`` and ``engine_sha_expected``
#: visibly disagreeing beside the word "match" and no way to tell why.
ENGINE_SHA_BYTE_MATCH_LABEL = "match(byte-digest)"

#: Closed vocabulary for the ``engine_sha_match`` column, mirroring ``SKIP_REASONS`` and
#: ``PROVENANCE_LABELS``. ``MISMATCH`` is shouted where the others are not, because
#: ``match``/``mismatch`` differ by two leading characters and this column is read by eye
#: down a 3745-row CSV; and because ``not_checked`` must never be mistaken for a pass.
ENGINE_SHA_MATCH_LABELS = (
    "not_checked", "match", ENGINE_SHA_BYTE_MATCH_LABEL, "MISMATCH")

_HEX_DIGITS = frozenset("0123456789abcdef")

_QUERY = """
SELECT m.measurement_id, m.run_id, m.channel, m.role, m.fixture_id, m.modality,
       m.re_connection, m.timestamp, m.eis_file_path, m.eis_params_json,
       e.campaign, e.workflow_mode, e.workflow_name
FROM measurements m
LEFT JOIN experiments e ON e.run_id = m.run_id
WHERE m.modality = 'eis'
ORDER BY m.measurement_id
"""


# ── row ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SweepRow:
    """One spectrum: its provenance, then its three scalars. Provenance comes first
    because it is the column Risk 6 is checked in, and a field order is a statement
    about what a reader is meant to look at."""

    measurement_id: int
    run_id: str
    channel: int
    role: str
    fixture_id: str
    modality: str
    re_connection: str
    timestamp: str
    campaign: str
    workflow_mode: str
    workflow_name: str
    provenance: str
    phase_table: str
    #: Abbreviated *content* digest of the engine — line endings normalised before hashing,
    #: so it is equal on every checkout of the same commit (module docstring).
    engine_sha: str
    #: The digest the caller said it expected, or ``""`` when none was supplied.
    engine_sha_expected: str
    #: One of :data:`ENGINE_SHA_MATCH_LABELS`. Written per row, not only to stdout, so a
    #: CSV read next month — detached from the console it was printed with — still carries
    #: the answer. A file whose provenance lives only in a terminal has none.
    engine_sha_match: str
    n_points: int

    s1_outcome: str
    s1_lift: float
    s1_judgeable: bool
    s1_below_plateau_depth: float
    plateau_C_F: float
    plateau_lo_hz: float
    plateau_hi_hz: float
    plateau_decades: float
    plateau_wide_enough: bool

    s2_margin: float
    s2_provisional: bool
    s2_f_at_min_hz: float
    s2_z_at_min_ohm: float
    s2_eps_deg: float
    eps_clamped: bool

    s3_n_negative: int
    s3_frac: float
    s3_re_state: str


CSV_COLUMNS: tuple[str, ...] = tuple(SweepRow.__dataclass_fields__)


@dataclass(frozen=True)
class EngineShaCheck:
    """Did this sweep run against the engine the caller expected?

    Four states, not two, and the ones past the second are the reason this is a class rather
    than a boolean: *matched*, *matched the retired byte digest of the same content*, *did
    not match*, and **nobody said** — see the module docstring. An unreadable engine source
    yields ``actual == "unknown"``, which cannot start with any hex prefix and therefore
    reports a mismatch; losing the fingerprint must not quietly become a pass.
    """

    #: sha256 of the engine source with line endings normalised to LF — the *content*
    #: identity, equal on every checkout of the same commit.
    actual: str
    #: Already normalised by :func:`normalize_engine_sha`; ``""`` means not checked.
    expected: str = ""
    #: sha256 of the same file's raw bytes on **this** checkout — the digest this module
    #: published until :data:`BYTE_DIGEST_RETIRED_ON`. Carried only so that a caller pasting
    #: the retired value is told what it is instead of being shouted at.
    actual_bytes: str = ""

    @property
    def checked(self) -> bool:
        return bool(self.expected)

    @property
    def matched(self) -> bool:
        return self.checked and self.actual.startswith(self.expected)

    @property
    def matched_bytes_only(self) -> bool:
        """The expectation is the *retired* byte digest of the very same file.

        Only true when the content digest did **not** match, so on a checkout where the two
        digests coincide — ``core.autocrlf=false``, or a file that was always LF — this is
        never reached and :attr:`matched` answers instead.
        """
        return (self.checked and not self.matched and bool(self.actual_bytes)
                and self.actual_bytes.startswith(self.expected))

    @property
    def label(self) -> str:
        """The value written to the ``engine_sha_match`` column."""
        if not self.checked:
            return "not_checked"
        if self.matched:
            return "match"
        if self.matched_bytes_only:
            return ENGINE_SHA_BYTE_MATCH_LABEL
        return "MISMATCH"

    @property
    def short(self) -> str:
        """What the ``engine_sha`` column carries: the leading
        :data:`ENGINE_SHA_SHORT_LEN` characters, or ``"unknown"`` unabridged."""
        return self.actual[:ENGINE_SHA_SHORT_LEN] if self.actual else "unknown"


@dataclass
class SweepResult:
    """Rows that were computed, plus a full account of the ones that were not."""

    rows: list[SweepRow] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    #: measurement_id -> reason, so an unresolved row can be chased rather than tallied.
    skipped_ids: dict[int, str] = field(default_factory=dict)
    n_considered: int = 0
    #: Carried on the result rather than passed separately to :func:`sweep_summary`, so
    #: the CSV's columns and the summary's banner can never disagree about the same run.
    engine_sha_check: EngineShaCheck = field(
        default_factory=lambda: EngineShaCheck(actual=""))

    def record_skip(self, measurement_id: int, reason: str) -> None:
        if reason not in SKIP_REASONS:
            raise ValueError(f"unknown skip reason {reason!r}")
        self.skipped[reason] += 1
        self.skipped_ids[int(measurement_id)] = reason


# ── the three rules the brief said to state explicitly ───────────────────────

def classify_provenance(
    eis_params_json: str | None, workflow_mode: str | None
) -> str:
    """The provenance label for one measurement row. Rule and evidence: module docstring.

    Precedence is most-specific-first: the row's own declaration outranks its run's mode,
    because the mode is a property of 34 runs that produced no rows and the declaration is
    a property of the measurement itself.
    """
    try:
        params = json.loads(eis_params_json or "{}")
    except (TypeError, ValueError):
        params = {}
    if isinstance(params, Mapping) and MOCK_FLAG_KEY in params:
        return "mock_declared" if params[MOCK_FLAG_KEY] else "real_declared"
    if (workflow_mode or "") == "simulation":
        return "simulated_run"
    return "undeclared"


def resolve_spectrum_path(
    raw: str | None, project_dir: Path
) -> tuple[Path | None, str | None]:
    """``(path, skip_reason)`` for a stored ``eis_file_path``.

    Relative to *project_dir* unless already absolute — the convention
    :func:`softae.tools.commission._load_role_spectra` uses. Returns a reason rather than
    raising, because a store with 20 dangling paths in 3745 is a fact to count, not an
    error to abort on.
    """
    if not raw or not str(raw).strip():
        return None, "no_file_path"
    candidate = Path(str(raw))
    full = candidate if candidate.is_absolute() else Path(project_dir) / candidate
    if not full.exists():
        return None, "file_missing"
    return full, None


def load_phase_tables(root: Path) -> dict[str, Any]:
    """``fixture_id -> PhaseAccuracyTable`` for every committed calibration under *root*.

    Loaded with :func:`~softae.analysis.eis.calibration.load_calibration` rather than
    ``resolve_calibration``: staleness-dropping is a gate's behaviour, and this is a
    descriptive sweep (module docstring).
    """
    from softae.analysis.eis.calibration import load_calibration

    tables: dict[str, Any] = {}
    for path in sorted(Path(root).glob("*.toml")):
        cal = load_calibration(path=path)
        if cal is not None and not cal.phase_acc.is_empty:
            tables[cal.fixture_id or path.stem] = cal.phase_acc
    return tables


def select_phase_table(
    fixture_id: str | None, tables: Mapping[str, Any]
) -> tuple[Any | None, str]:
    """``(table, label)`` for a row, applying the stated fallback.

    A row naming a fixture gets that fixture's table. A row naming none — 3727 of 3745 —
    inherits the sole calibration on disk *when there is exactly one*, labelled
    ``…(sole-fallback)`` so the inheritance is visible in the CSV rather than implied by
    this docstring. Anything else gets no table, ``eps`` is NaN, and the row is
    provisional.
    """
    key = (fixture_id or "").strip()
    if key:
        table = tables.get(key)
        return (table, key) if table is not None else (None, "none")
    if len(tables) == 1:
        only = next(iter(tables))
        return tables[only], f"{only}(sole-fallback)"
    return None, "none"


# ── the sweep ────────────────────────────────────────────────────────────────

def _engine_source_bytes() -> bytes | None:
    """The engine's own source bytes, or ``None`` when it cannot be read."""
    from softae.analysis.eis import measurability

    try:
        return Path(measurability.__file__).read_bytes()
    except (OSError, TypeError):
        return None


def _normalize_newlines(data: bytes) -> bytes:
    """CRLF and lone CR to LF, so a digest names content and not a checkout rendering.

    Lone CR is folded too, and not for tidiness: a file arriving through an old-Mac or a
    mangled-transfer path would otherwise digest differently from the identical content, and
    a normalisation with a hole in it is a normalisation nobody can rely on.
    """
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def engine_fingerprint_full() -> str:
    """The full 64-char **content** digest of the measurability engine, or ``"unknown"``.

    Digests the *source file* rather than reading a version string, because the engine has
    no version and the thing that actually moved the numbers was an unversioned edit. An
    unreadable source degrades to ``"unknown"``: losing the fingerprint must not lose the
    sweep, but it must be visible in the column that it was lost.

    Line endings are normalised to LF **before** hashing (module docstring). Without that
    the digest is a property of the checkout rather than of the engine: this machine has
    ``core.autocrlf=true`` against an LF repository, so the identical commit measured
    ``70e94699b69f…`` here and ``a7d0c8fbacae4b99…`` in the index. For an unmodified file
    the value returned now equals ``sha256(git show HEAD:<path>)``, which is the portable
    identity the column was always cited as carrying.

    The full digest is what ``--expect-engine-sha`` compares against, so an expected prefix
    longer than :data:`ENGINE_SHA_SHORT_LEN` still discriminates.
    """
    from hashlib import sha256

    source = _engine_source_bytes()
    if source is None:
        return "unknown"
    return sha256(_normalize_newlines(source)).hexdigest()


def engine_byte_fingerprint_full() -> str:
    """The engine's **un-normalised** byte digest — what this module published until
    :data:`BYTE_DIGEST_RETIRED_ON`, and nothing else.

    Not an identity: on this checkout it is ``70e94699b69f…`` and on one with
    ``core.autocrlf=false`` it is the content digest, for byte-identical content. It is
    computed solely so :attr:`EngineShaCheck.matched_bytes_only` can recognise a caller who
    pasted the retired value and answer *that is this same content* rather than *wrong
    engine* — which would be a false alarm of exactly the kind this change removed.
    """
    from hashlib import sha256

    source = _engine_source_bytes()
    if source is None:
        return "unknown"
    return sha256(source).hexdigest()


def engine_fingerprint() -> str:
    """The abbreviated content digest carried by the ``engine_sha`` column."""
    return engine_fingerprint_full()[:ENGINE_SHA_SHORT_LEN]


def normalize_engine_sha(value: str) -> str:
    """Lower-cased hex for an expected digest, or :class:`ValueError` with the reason.

    A prefix is accepted because the digest is cited abbreviated — the engine has been
    quoted on the channel as ``70e94699b69f…``, which is now the *retired byte* form and is
    recognised as such rather than refused. A *short* prefix is refused rather than
    matched loosely: the point of the guard is to say which engine ran, and a prefix short
    enough to collide by accident answers that question about itself instead. Refusing is
    safe here precisely because it happens at argument-parsing time; the check itself,
    once armed, never refuses anything (module docstring).
    """
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError(
            "expected engine sha is empty; omit --expect-engine-sha rather than passing "
            "a blank one, so that 'not checked' is stated instead of implied")
    if not set(text) <= _HEX_DIGITS:
        raise ValueError(
            f"expected engine sha {value!r} is not hexadecimal; pass the digest itself, "
            "without an ellipsis or any other decoration")
    if len(text) < ENGINE_SHA_MIN_PREFIX:
        raise ValueError(
            f"expected engine sha {value!r} is {len(text)} hex chars; at least "
            f"{ENGINE_SHA_MIN_PREFIX} are required. A shorter prefix matches engine "
            "states it was never meant to, which reports the guard's tolerance rather "
            "than the engine's identity")
    if len(text) > 64:
        raise ValueError(
            f"expected engine sha {value!r} is {len(text)} hex chars; a sha256 digest is "
            "at most 64")
    return text


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """The only way this module opens the store. ``mode=ro`` is a URI flag, so a typo in
    the scheme degrades to a *writable* file connection — hence one function, used
    everywhere, rather than the pattern repeated at three call sites."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"no DataStore at {db_path}")
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def _measure(
    freq: Any, Z: Any, table: Any, re_state: str
) -> tuple[Any, Any, Any, bool]:
    """S1, S2, S3 and ``eps_clamped`` for one spectrum. No masking, by design: S2's
    minimum is taken over the band *including* the points ``gate_quadrant`` would drop
    (spec §6), and dropping them first is half the shipped defect."""
    # ``tand_margin`` and ``eps_is_clamped`` both duck-type the table, so ``None`` needs
    # no branch here: it yields a NaN ``eps`` and therefore a provisional row, which is
    # the correct answer for a spectrum no calibration covers.
    s1 = conduction_lift(freq, Z)
    s2 = tand_margin(freq, Z, table)
    s3 = negative_conductance_count(Z, re_state=re_state)
    clamped = bool(
        table is not None
        and s2.z_at_min == s2.z_at_min
        and eps_is_clamped(table, s2.z_at_min)
    )
    return s1, s2, s3, clamped


def _build_row(
    rec: Sequence[Any], eis: Any, table: Any, label: str, engine: EngineShaCheck
) -> SweepRow:
    (mid, run_id, channel, role, fixture_id, modality, re_connection, timestamp,
     _path, params_json, campaign, workflow_mode, workflow_name) = rec

    freq = eis.frequency
    Z = eis.z_complex
    re_state = re_connection if re_connection in RE_STATES else "unverified"
    s1, s2, s3, clamped = _measure(freq, Z, table, re_state)

    return SweepRow(
        measurement_id=int(mid),
        run_id=str(run_id or ""),
        channel=int(channel),
        role=str(role or ""),
        fixture_id=str(fixture_id or ""),
        modality=str(modality or ""),
        re_connection=str(re_connection or ""),
        timestamp=str(timestamp or ""),
        campaign=str(campaign or ""),
        workflow_mode=str(workflow_mode or ""),
        workflow_name=str(workflow_name or ""),
        provenance=classify_provenance(params_json, workflow_mode),
        phase_table=label,
        engine_sha=engine.short,
        engine_sha_expected=engine.expected,
        engine_sha_match=engine.label,
        n_points=int(len(freq)),
        s1_outcome=s1.outcome,
        s1_lift=float(s1.lift),
        s1_judgeable=bool(s1.judgeable),
        s1_below_plateau_depth=float(s1.below_plateau_depth),
        plateau_C_F=float(s1.plateau.C_plateau),
        plateau_lo_hz=float(s1.plateau.lo_hz),
        plateau_hi_hz=float(s1.plateau.hi_hz),
        plateau_decades=float(s1.plateau.decades),
        plateau_wide_enough=bool(s1.plateau.wide_enough),
        s2_margin=float(s2.margin),
        s2_provisional=bool(s2.provisional),
        s2_f_at_min_hz=float(s2.f_at_min),
        s2_z_at_min_ohm=float(s2.z_at_min),
        s2_eps_deg=float(s2.eps_deg),
        eps_clamped=clamped,
        s3_n_negative=int(s3.n),
        s3_frac=float(s3.frac),
        s3_re_state=s3.re_state,
    )


def _iter_records(conn: sqlite3.Connection) -> Iterator[tuple]:
    yield from conn.execute(_QUERY)


def sweep(
    db_path: Path,
    project_dir: Path,
    *,
    calibration_root: Path | None = None,
    limit: int | None = None,
    expect_engine_sha: str | None = None,
) -> SweepResult:
    """Compute S1/S2/S3 over every stored EIS spectrum. Opens the store read-only.

    Every measurement row is *considered*; the ones that cannot be loaded are tallied by
    reason on :attr:`SweepResult.skipped` and their ids kept on
    :attr:`SweepResult.skipped_ids`.

    *expect_engine_sha* is the digest — full or a prefix of at least
    :data:`ENGINE_SHA_MIN_PREFIX` hex chars — the caller believes it is sweeping under. A
    malformed one raises :class:`ValueError` here, before any work; a **mismatching** one
    does not raise, does not change a single value, and is reported instead (module
    docstring).
    """
    from softae.analysis.eis_data import EISResult

    root = Path(calibration_root) if calibration_root else Path("calibration") / "eis"
    tables = load_phase_tables(root)
    engine = EngineShaCheck(
        actual=engine_fingerprint_full(),
        expected=normalize_engine_sha(expect_engine_sha) if expect_engine_sha else "",
        actual_bytes=engine_byte_fingerprint_full())
    result = SweepResult(engine_sha_check=engine)

    conn = _open_readonly(Path(db_path))
    try:
        for rec in _iter_records(conn):
            if limit is not None and len(result.rows) >= limit:
                break
            result.n_considered += 1
            mid = int(rec[0])
            path, reason = resolve_spectrum_path(rec[8], Path(project_dir))
            if reason is not None:
                result.record_skip(mid, reason)
                continue
            try:
                eis = EISResult.load(path)
            except Exception:
                result.record_skip(mid, "unreadable")
                continue
            if len(eis.frequency) == 0:
                result.record_skip(mid, "no_finite_points")
                continue
            table, label = select_phase_table(rec[4], tables)
            result.rows.append(_build_row(rec, eis, table, label, engine))
    finally:
        conn.close()
    return result


# ── reporting ────────────────────────────────────────────────────────────────

def write_csv(rows: Iterable[SweepRow], path: Path) -> Path:
    """One row per spectrum, all provenance columns then all scalar columns."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return target


def _finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if v == v and not math.isinf(v)]


def _median(values: Iterable[float]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _scalar_stats(rows: Sequence[SweepRow]) -> dict[str, Any]:
    """The S1/S2/S3 summary used both for the whole corpus and for each cross-tab cell."""
    judgeable = [r for r in rows if r.s1_judgeable]
    return {
        "n": len(rows),
        "s1_judgeable": len(judgeable),
        "s1_excursion": sum(1 for r in rows if r.s1_outcome == "excursion"),
        "s1_lift_median": _median(r.s1_lift for r in judgeable),
        "s2_measured": sum(1 for r in rows if not r.s2_provisional),
        "s2_margin_median": _median(r.s2_margin for r in rows),
        "s3_any_negative": sum(1 for r in rows if r.s3_n_negative > 0),
        "s3_frac_median": _median(r.s3_frac for r in rows),
    }


def _fmt_stats(label: str, s: Mapping[str, Any]) -> str:
    return (
        f"  {label:<28} n={s['n']:<6} "
        f"S1 judgeable {s['s1_judgeable']:<6} excursion {s['s1_excursion']:<6} "
        f"lift~{s['s1_lift_median']:.3g} | "
        f"S2 measured {s['s2_measured']:<6} margin~{s['s2_margin_median']:.3g} | "
        f"S3 n>0 {s['s3_any_negative']:<6} frac~{s['s3_frac_median']:.3g}"
    )


@dataclass(frozen=True)
class ScalarColumnHealth:
    """How often one scalar column had no value, and whether the frame explains it.

    ``explanation`` is where *measured-and-absent* is separated from *not measured at
    all*, and it is a column of this same frame rather than a judgement: an S2 margin is
    NaN **by design** on a row no calibration covered, which is a fact about the
    calibration set and not about the engine. An S1 lift NaN in every row is a fact about
    the engine. Nothing outside the frame can tell the two apart, so the explanation is
    derived from it or it is empty — and an empty one is what makes a column alarming.
    """

    column: str
    n_absent: int
    n_rows: int
    explanation: str = ""

    @property
    def frac_absent(self) -> float:
        return self.n_absent / self.n_rows if self.n_rows else float("nan")

    @property
    def alarming(self) -> bool:
        return (self.n_rows > 0 and not self.explanation
                and self.frac_absent >= NAN_COLUMN_ALARM_FRAC)


def scalar_column_health(rows: Sequence[SweepRow]) -> list[ScalarColumnHealth]:
    """One :class:`ScalarColumnHealth` per scalar column, in S1/S2/S3 order.

    S1's unjudgeable rows are counted as absent rather than excluded, which is the
    opposite of what :func:`_scalar_stats` does and is deliberate: the median must not be
    taken over rows that were never measured, while *the alarm is precisely about every
    row being unmeasured*. Filtering them out here would compute the statistic over an
    empty set and report NaN-as-usual — the alarm silencing itself on its own trigger.
    """
    n = len(rows)
    uncovered = n > 0 and all(r.phase_table == "none" for r in rows)
    return [
        ScalarColumnHealth(
            "s1_lift",
            sum(1 for r in rows if not r.s1_judgeable or r.s1_lift != r.s1_lift), n),
        ScalarColumnHealth(
            "s2_margin",
            sum(1 for r in rows if r.s2_provisional or r.s2_margin != r.s2_margin), n,
            "no calibration covered ANY swept row (phase_table='none' throughout), so a "
            "NaN margin is the stated answer here, not a missing one" if uncovered
            else ""),
        ScalarColumnHealth(
            "s3_frac", sum(1 for r in rows if r.s3_frac != r.s3_frac), n),
    ]


def _scalar_health_block(rows: Sequence[SweepRow]) -> list[str]:
    """The all-NaN-column banner, or the one-line statement that no column is empty.

    Printed **above** every other statistic because of what it detects: an engine
    returning all NaN yields a CSV with every column, every row and every label correct,
    so each block below this one would look entirely normal. A NaN count buried among
    other counts is not enough — a reader scanning a healthy-looking report is exactly
    the reader this is for.
    """
    if not rows:
        return [
            "*** NOTHING WAS SWEPT — no scalar column could be formed. ***",
            "  Read the skip table below as the whole result. The absence of alarms "
            "here is not health; nothing was measured to alarm about.",
            "",
        ]

    health = scalar_column_health(rows)
    detail = [
        f"  {h.column:<12} absent in {h.n_absent} of {h.n_rows} rows "
        f"({100.0 * h.frac_absent:.1f}%)"
        + (f"   <- {h.explanation}" if h.explanation else "")
        for h in health
    ]
    alarms = [h for h in health if h.alarming]
    if not alarms:
        return ["Scalar column health — every column carries values:", *detail, ""]
    return [
        "*" * 78,
        "*** SUSPECTED ENGINE OR PLUMBING FAILURE — a scalar column has NO values. ***",
        *detail,
        f"  Above the {100.0 * NAN_COLUMN_ALARM_FRAC:.0f}% absence alarm: "
        + ", ".join(h.column for h in alarms),
        "  A scalar absent for EVERY row is not a measurement of the corpus; it is a "
        "measurement of this tool. An all-NaN engine still writes a well-formed CSV "
        "with every column present, so nothing else in this report would look wrong.",
        "  Check the engine at the sha above, and this tool's inputs, BEFORE citing any "
        "S1/S2/S3 distribution from this run. Descriptive only: no value was changed, "
        "no row rejected, no verdict formed.",
        "*" * 78,
        "",
    ]


def _engine_sha_block(engine: EngineShaCheck) -> list[str]:
    """The wrong-engine banner, a match confirmation, the retired-digest explanation, or
    the not-checked statement.

    Printed **above** the scalar-health block and therefore above every count, for the same
    reason that block is: a sweep against a mutated engine produces a CSV that is complete,
    internally consistent, correctly fingerprinted and wrong, so no block below this one
    would look abnormal. It sits above scalar health specifically because scalar health is
    a verdict *about* the engine — if the wrong engine ran, its clean bill of health is
    also about the wrong engine.

    All four states print. A guard that is silent when unarmed is indistinguishable from
    one that passed.
    """
    if not engine.checked:
        return [
            "Expected engine sha: NOT CHECKED — no --expect-engine-sha was supplied.",
            "  Whether this run used the intended engine is UNKNOWN, not confirmed. A "
            "single sweep against a mutated engine is fingerprinted, internally "
            "consistent, complete and entirely wrong, with nothing to compare against; "
            "the sha above says only what ran, never whether it was the right thing.",
            "",
        ]
    if engine.matched:
        return [
            f"Expected engine sha: MATCH — actual content digest (line-ending normalised) "
            f"starts with the expected {engine.expected} (--expect-engine-sha).",
            "",
        ]
    if engine.matched_bytes_only:
        return [
            "Expected engine sha: MATCH via the RETIRED BYTE DIGEST — the engine is the "
            "expected one, and NOTHING here is wrong.",
            f"  You passed {engine.expected}. That is the pre-{BYTE_DIGEST_RETIRED_ON} "
            f"BYTE digest of this same content — this checkout's working-tree bytes hash "
            f"to it exactly — not a different engine.",
            f"  The content digest of that same content is: {engine.actual}",
            "  Until "
            f"{BYTE_DIGEST_RETIRED_ON} this tool hashed the source file's bytes as checked "
            "out. With core.autocrlf=true against an LF repository that named a local "
            "rendering rather than the content, so the identical commit measured one digest "
            "here and another in the index, and a clone with autocrlf=false would have been "
            "reported as the wrong engine while holding byte-identical content. The digest "
            "is now taken over LF-normalised bytes and is equal on every checkout.",
            # The shouted token stays out of this branch's prose on purpose: it is the
            # verdict one block below, and a reader scanning for it must not find it inside
            # the message that exists to say the verdict does not apply.
            "  Cite the content digest above from now on. This run's numbers are comparable "
            "with any other run against the same content, whatever digest form named it.",
            "",
        ]
    return [
        "*" * 78,
        "*** WRONG ENGINE — THIS SWEEP DID NOT RUN AGAINST THE EXPECTED ENGINE. ***",
        f"  expected (--expect-engine-sha): {engine.expected}",
        f"  actual   (analysis/eis/measurability.py, content digest, line-ending "
        f"normalised): {engine.actual}",
        "  Checked against this checkout's raw byte digest too, in case the expected value "
        f"were the pre-{BYTE_DIGEST_RETIRED_ON} form; it is not that either "
        f"({engine.actual_bytes or 'unknown'}). This is a different engine, not a "
        "different way of writing the same one.",
        "  Every S1/S2/S3 number below was computed by a DIFFERENT engine state and is "
        "NOT comparable with any run made against the expected digest. Do not "
        "concatenate this CSV with one, and do not cite these distributions as that "
        "engine's. A sweep run against a transiently mutated engine once returned a "
        "plausible, internally consistent, complete distribution whose median S1 lift "
        "was 90.8 where the clean engine gave 3.62 on the identical corpus — and nothing "
        "in its output distinguished it from a good run; it was caught only because a "
        "second run happened to disagree.",
        "  This is not an error and nothing was refused: sweeping an unblessed engine is "
        "legitimate and is sometimes the intent. Descriptive only — no value changed, no "
        "row rejected, no verdict formed, exit status unaffected. The requirement is "
        "only that such a run can never be silently mistaken for a blessed one.",
        "*" * 78,
        "",
    ]


def _risk6_block(rows: Sequence[SweepRow], declared: Mapping[str, int]) -> list[str]:
    """The Risk 6 cross-tab, or an unmissable statement that it could not be formed.

    A breakdown printed over a label with one value reads as *"no confounding detected"*
    when the truth is *"no comparison was possible"*. That is the failure this whole
    section exists to check for, so the degenerate case gets the loud branch and the
    counts that justify it, never a quietly uniform table.
    """
    by_label: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        by_label[row.provenance].append(row)

    mock_swept = len(by_label.get("mock_declared", ()))
    out = ["", "Risk 6 — S1/S2/S3 cross-tabulated by provenance"]
    if mock_swept == 0 or len(by_label) < 2:
        n_declared = declared.get("mock_declared", 0) + declared.get("real_declared", 0)
        out += [
            "  *** NOT APPLICABLE — no mock/real comparison was possible. ***",
            f"  {mock_swept} of {len(rows)} swept spectra are mock-sourced; the swept "
            f"population carries {len(by_label)} provenance label(s).",
            f"  The declared population ({n_declared} rows: "
            f"{declared.get('mock_declared', 0)} mock_declared + "
            f"{declared.get('real_declared', 0)} real_declared) has NO spectrum files "
            "and is unreachable by this sweep.",
            f"  Rows under workflow_mode='simulation': "
            f"{declared.get('simulated_run', 0)} (the 34 bo_campaign runs carry no "
            "measurement rows at all).",
            "  This is NOT evidence that provenance does not confound S1/S2/S3. It is "
            "the statement that this corpus cannot answer the question.",
        ]
    for label in PROVENANCE_LABELS:
        if by_label.get(label):
            out.append(_fmt_stats(label, _scalar_stats(by_label[label])))
    return out


def sweep_summary(result: SweepResult, declared: Mapping[str, int] | None = None) -> str:
    """Totals, the all-NaN-column guard, skip accounting, reachability, and Risk 6.

    The reachability pass is the deliverable that matters as much as the numbers
    (``SUBAGENT_RULES`` §3): a test proving the code *can* branch says nothing about
    whether real data *reaches* the branch, and only a pass over the real corpus does.
    **A branch reached by zero spectra is reported as a zero, never omitted.**

    The two identity guards sit immediately under the header, ahead of every count,
    because they are the things no other block can reveal — which engine ran
    (:func:`_engine_sha_block`) and whether it produced any values at all
    (:func:`_scalar_health_block`).
    """
    rows = result.rows
    engine = result.engine_sha_check
    lines = [
        "=" * 78,
        f"Measurability sweep — {len(rows)} spectra swept of "
        f"{result.n_considered} EIS measurement rows considered",
        f"engine (analysis/eis/measurability.py) sha: "
        f"{engine.short if engine.actual else engine_fingerprint()}"
        "  (content digest, line-ending normalised)"
        "   <- these numbers are a claim about THIS engine state; a transiently mutated "
        "engine once gave median S1 lift 90.8 where the clean one gave 3.62 on the "
        "identical corpus",
        "=" * 78,
        "",
    ]
    lines += _engine_sha_block(engine)
    lines += _scalar_health_block(rows)
    lines.append("Skipped (every row accounted for, none silently):")
    lines += [f"  {reason:<20} {result.skipped.get(reason, 0)}"
              for reason in SKIP_REASONS]
    lines.append(f"  {'TOTAL SKIPPED':<20} {sum(result.skipped.values())}")

    lines += ["", "CORPUS REACHABILITY — S1 outcome (all four, zeros included):"]
    outcomes = Counter(r.s1_outcome for r in rows)
    for outcome in OUTCOMES:
        n = outcomes.get(outcome, 0)
        mark = "   <- reached by NO spectrum" if n == 0 else ""
        tag = " [UNJUDGEABLE]" if outcome in UNJUDGEABLE_OUTCOMES else ""
        lines.append(f"  {outcome:<16}{tag:<15} {n}{mark}")
    unjudgeable = sum(outcomes.get(o, 0) for o in UNJUDGEABLE_OUTCOMES)
    lines += [
        f"  {'unjudgeable total':<31} {unjudgeable}  "
        "(S1 NOT MEASURED, as against measured-and-absent)",
        f"  {'plateau too narrow':<31} "
        f"{sum(1 for r in rows if r.plateau_decades > 0 and not r.plateau_wide_enough)}"
        "  (spec Risk 2)",
    ]

    n_nan = sum(1 for r in rows if r.s2_eps_deg != r.s2_eps_deg)
    lines += [
        "",
        "CORPUS REACHABILITY — S2 phase-table coverage:",
        f"  {'eps NaN (outside coverage)':<31} {n_nan}",
        f"  {'eps finite but CLAMPED':<31} "
        f"{sum(1 for r in rows if r.eps_clamped)}  (endpoint, NOT characterised)",
        f"  {'eps interpolated':<31} "
        f"{sum(1 for r in rows if r.s2_eps_deg == r.s2_eps_deg and not r.eps_clamped)}",
        f"  {'margin provisional (NaN)':<31} "
        f"{sum(1 for r in rows if r.s2_provisional)}  (never a pass)",
        "  phase table used:  " + ", ".join(
            f"{k}={v}" for k, v in sorted(Counter(r.phase_table for r in rows).items())),
    ]

    lines += ["", "CORPUS REACHABILITY — S3 negative conductance, by re_connection:"]
    per_state: dict[str, list[SweepRow]] = defaultdict(list)
    for row in rows:
        per_state[row.s3_re_state].append(row)
    for state in RE_STATES:
        group = per_state.get(state, [])
        n_pos = sum(1 for r in group if r.s3_n_negative > 0)
        mark = "   <- state reached by NO spectrum" if not group else ""
        lines.append(f"  {state:<20} n={len(group):<6} with n_negative>0: {n_pos:<6}"
                     f" frac~{_median(r.s3_frac for r in group):.3g}{mark}")

    lines += ["", "Whole corpus:", _fmt_stats("ALL", _scalar_stats(rows))]
    lines += _risk6_block(rows, declared or {})
    return "\n".join(lines)


def declared_provenance_counts(db_path: Path) -> dict[str, int]:
    """Provenance over **all** measurement rows, including the unsweepable ones.

    Without this the Risk 6 block could only say "we saw no mock rows", which is
    indistinguishable from "there are none". The counts here are what let it say *where
    the mock rows went* — they exist, they have no spectrum files, and that disjointness
    is the finding.
    """
    conn = _open_readonly(Path(db_path))
    try:
        counts: Counter = Counter()
        for params, mode in conn.execute(
                "SELECT m.eis_params_json, e.workflow_mode FROM measurements m "
                "LEFT JOIN experiments e ON e.run_id = m.run_id"):
            counts[classify_provenance(params, mode)] += 1
    finally:
        conn.close()
    return dict(counts)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _default_out() -> Path:
    """A scratch location — deliberately neither the repo nor the project dir, since a
    sweep is an observation and writing it beside the data would make it look like one."""
    return Path(tempfile.gettempdir()) / "measurability_sweep.csv"


def _expected_sha_arg(value: str) -> str:
    """argparse adapter for :func:`normalize_engine_sha`.

    Re-raised as :class:`argparse.ArgumentTypeError` because argparse swallows a
    ``ValueError``'s message and prints only ``invalid value``, which would turn a stated
    reason ("8 hex chars minimum, and why") into a bare refusal.
    """
    try:
        return normalize_engine_sha(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    # The summary prints Greek and typographic characters, and this rig's console is
    # cp1252: without this the tool dies on its own report rather than on the analysis.
    from softae.tools import use_utf8_console
    use_utf8_console()

    parser = argparse.ArgumentParser(
        prog="measurability-sweep",
        description="Read-only S1/S2/S3 sweep over stored EIS spectra. Writes no store.")
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="DataStore project dir (default: [data] project_dir)")
    parser.add_argument("--db", type=Path, default=None,
                        help="sqlite file (default: <project-dir>/db/<db_filename>)")
    parser.add_argument("--calibration-root", type=Path,
                        default=Path("calibration") / "eis")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"CSV path (default: {_default_out()})")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expect-engine-sha", type=_expected_sha_arg, default=None, metavar="HEX",
        help=f"content digest (line-ending normalised) of the measurability engine this "
             f"sweep is expected to run against; a prefix of >= {ENGINE_SHA_MIN_PREFIX} "
             f"hex chars is accepted. The pre-{BYTE_DIGEST_RETIRED_ON} byte digest of the "
             f"same content is recognised and explained rather than reported as a "
             f"mismatch. A real mismatch is reported loudly and changes nothing — it never "
             f"fails the run. Omitted, the summary states that no digest was checked.")
    args = parser.parse_args(argv)

    if args.project_dir is not None:
        project_dir = Path(args.project_dir).expanduser()
    else:
        from softae.config.loader import data_project_dir
        project_dir = Path(data_project_dir()).expanduser()

    if args.db is not None:
        db_path = Path(args.db)
    else:
        from softae.config.loader import data_db_filename
        db_path = project_dir / "db" / data_db_filename()

    result = sweep(db_path, project_dir,
                   calibration_root=args.calibration_root, limit=args.limit,
                   expect_engine_sha=args.expect_engine_sha)
    out = write_csv(result.rows, args.out or _default_out())
    print(sweep_summary(result, declared_provenance_counts(db_path)))
    print(f"\nCSV: {out}")
    print("  A scratch path. Copy it somewhere durable before citing it anywhere "
          "persistent (SUBAGENT_RULES §8).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
