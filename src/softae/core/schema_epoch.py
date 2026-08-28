"""Read the ``SCHEMA_EPOCHS`` ledger: which epoch governs a stored value.

:data:`softae.core.data_store.SCHEMA_EPOCHS` is six carefully written notes about
what stored numbers mean, and until this module it was **write-only** — every
reference in ``src/`` and ``tools/`` was the definition itself, the
``INSERT OR IGNORE`` seeder, a docstring explaining why some *other* change
needed no row, or one of two dump accessors. Nothing mapped a row to the epoch
that governs it, so the notes were addressed to a human reading the ledger and
to nothing that executes.

The question a reader actually has is not "what epochs exist" but
**"is this stored value comparable to one written today, and if not, why not?"**
:func:`crossed_since` answers exactly that.

Why a companion scope table
---------------------------
A ledger row is ``(version, kind, note)``. It carries **no structured statement**
of which table or column it affects, nor its effective date except inside the
prose — so a lookup needs scope metadata that does not exist in the tuple.

The tuples are not extended, for two reasons. The seeder unpacks exactly three
elements and lives in ``data_store.py``; and, more bindingly, the ledger's own
docstring says its rows are *"never rewritten"* — a row already in a database is
a statement about data written under it. So :data:`_SCOPES` here declares scope
*beside* the ledger rather than inside it, and :func:`_build` refuses to import
if the two drift apart. A companion structure is acceptable only because that
check exists: without it, appending epoch 7 and forgetting to scope it would
make every query about epoch 7's columns quietly answer "nothing happened".

Why the scopes read the prose rather than derive anything
---------------------------------------------------------
Each scope entry is a **reading** of one note, and the note is the specification.
Where a note names columns, those columns are the scope and nothing is inferred
past them. Epoch 2 is the case that shows the discipline: its note says
"thickness-derived values", and ``fit_results.sigma_S_per_cm`` is thickness-
derived in the physics — but its ``electrode_t_cm`` arrives from ``[eis.cell]``
config through the EIS step params, never from ``formulations.deposit_area_mm2``,
so the 4.676x correction does not reach it and it is deliberately out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum

from softae.core.data_store import SCHEMA_EPOCHS

__all__ = [
    "Certainty",
    "EpochAnswer",
    "EpochRecord",
    "EpochScopeError",
    "Uncertainty",
    "Undetermined",
    "crossed_since",
    "epoch",
    "epochs",
    "epochs_touching",
]


class EpochScopeError(RuntimeError):
    """The scope declarations and ``SCHEMA_EPOCHS`` disagree."""


class Certainty(Enum):
    """Whether an epoch's start can be stated as a date at all.

    ``DATED`` is the ordinary case: the change landed with the code, so every
    database crossed it on the same day.

    ``CONDITIONAL`` exists because epoch 5 says of itself *"Shipped DISABLED;
    the epoch begins for a given database when someone arms the flag, not when
    this row was seeded"*. There is no universal date to return, and a helper
    that returned one would be lying with a plausible number.

    ``BASELINE`` is version 1, which consolidates pre-existing in-line
    migrations. It is the floor of the ledger, not a change to anything, so it
    has neither a date in its note nor a column whose meaning it moved.
    """

    DATED = "dated"
    CONDITIONAL = "conditional"
    BASELINE = "baseline"


class Uncertainty(Enum):
    """Why an epoch could not be settled for a given row."""

    #: The epoch has no universal effective date (see :attr:`Certainty.CONDITIONAL`).
    CONDITIONAL = "conditional"
    #: The row was written on the epoch's effective date, and the ledger records
    #: days, not instants — so which side of the change it fell on is unrecorded.
    SAME_DAY = "same_day"


@dataclass(frozen=True)
class _Scope:
    """Declared scope for one ledger version. Private: build records, not these."""

    columns: frozenset[tuple[str, str]]
    certainty: Certainty
    effective_date: date | None = None
    condition: str | None = None
    row_discriminator: str | None = None


# ---------------------------------------------------------------------------
# The scope declarations — one entry per SCHEMA_EPOCHS version, enforced below
# ---------------------------------------------------------------------------

_SCOPES: dict[int, _Scope] = {
    1: _Scope(
        # Deliberately empty. A baseline is where the ledger starts, not a change
        # of meaning; there is no population on the other side of it to compare
        # against. Recording it as an empty scope is a declared fact, which is
        # what keeps the coverage check below meaningful for version 1 too.
        columns=frozenset(),
        certainty=Certainty.BASELINE,
    ),
    2: _Scope(
        # `predicted_thickness_um` is final_volume_uL / area * 1000, so the
        # 4.0 -> 18.704 correction moves it by the same 4.676x that moved the
        # denominator. Both live on `formulations`.
        columns=frozenset({
            ("formulations", "deposit_area_mm2"),
            ("formulations", "predicted_thickness_um"),
        }),
        certainty=Certainty.DATED,
        effective_date=date(2026, 8, 7),
    ),
    3: _Scope(
        # Both sides of the rename are in scope. A reader who asks about
        # `temp_pv_C` is precisely the reader who needs to be told it became
        # `chamber_air_C` and meant chamber AIR all along; answering "no epoch
        # touches that column" because the name no longer exists would be the
        # least useful true statement available.
        columns=frozenset({
            ("conditions", "chamber_air_C"),
            ("conditions", "stage_temp_sp_C"),
            ("conditions", "temp_pv_C"),
            ("conditions", "temp_sp_C"),
        }),
        certainty=Certainty.DATED,
        effective_date=date(2026, 8, 11),
    ),
    4: _Scope(
        columns=frozenset({
            ("conditions", "temperature_C"),
            ("conditions", "temperature_source"),
        }),
        certainty=Certainty.DATED,
        effective_date=date(2026, 8, 12),
    ),
    5: _Scope(
        columns=frozenset({("fit_results", "R1")}),
        certainty=Certainty.CONDITIONAL,
        condition=(
            "[eis.pregate] two_point_open was armed for this database; the epoch "
            "begins when someone armed the flag, not when the ledger row was "
            "seeded (2026-08-18), and it applies only to OPEN-ARC spectra"
        ),
        # The note's own point: you cannot date this population, but you can
        # SELECT it. That makes the conditional answer actionable instead of a
        # shrug.
        row_discriminator="fit_results.engine = 'gated_two_point'",
    ),
    6: _Scope(
        columns=frozenset({
            ("fit_results", "arc_state"),
            ("fit_results", "arc_f_peak_hz"),
        }),
        certainty=Certainty.DATED,
        effective_date=date(2026, 8, 26),
        # No `row_discriminator`, and the note says why in capitals: arc_state has
        # no column recording which rule produced it, so this date is the only
        # thing separating "genuinely unjudgeable" from "refused by the old rule".
        # That absence is the reason the row had to exist at all.
    ),
}


@dataclass(frozen=True)
class EpochRecord:
    """One ledger row joined to its declared scope.

    ``kind`` and ``note`` are verbatim from :data:`SCHEMA_EPOCHS`. The ``kind``
    distinction is the point of the ledger and survives the lookup intact: a
    ``'schema'`` row says the shape moved while the numbers held still (epoch 3
    renamed columns and changed no value), a ``'data-epoch'`` row says the
    numbers changed meaning while the name held still.
    """

    version: int
    kind: str
    note: str
    certainty: Certainty
    effective_date: date | None
    columns: frozenset[tuple[str, str]]
    condition: str | None
    row_discriminator: str | None

    @property
    def changes_meaning(self) -> bool:
        """Whether stored values crossing this epoch changed what they mean."""
        return self.kind == "data-epoch"


@dataclass(frozen=True)
class Undetermined:
    """An epoch that touches the column but cannot be settled for this row."""

    epoch: EpochRecord
    cause: Uncertainty
    detail: str


@dataclass(frozen=True)
class EpochAnswer:
    """Settled epochs and undeterminable ones, kept apart on purpose.

    Flattening the two would let a caller read "unknown" as "no" — which is the
    direction that silently answers *comparable* when the truth is *nobody
    recorded*. ``bool(answer)`` is false only when both are empty.
    """

    settled: tuple[EpochRecord, ...]
    undetermined: tuple[Undetermined, ...]

    def __bool__(self) -> bool:
        return bool(self.settled or self.undetermined)


def _build() -> tuple[EpochRecord, ...]:
    """Join the ledger to the scope table, refusing to import if they disagree.

    Raising here rather than returning a partial answer is the whole reason a
    companion table is acceptable. A missing scope entry would otherwise make
    every query about that epoch's columns report "nothing happened" — the wrong
    answer wearing the safe answer's clothes.
    """
    ledger = {version: (kind, note) for version, kind, note in SCHEMA_EPOCHS}

    unscoped = sorted(set(ledger) - set(_SCOPES))
    if unscoped:
        raise EpochScopeError(
            f"SCHEMA_EPOCHS version(s) {unscoped} have no entry in _SCOPES; "
            "declare the affected table(s), column(s) and effective date"
        )
    orphaned = sorted(set(_SCOPES) - set(ledger))
    if orphaned:
        raise EpochScopeError(
            f"_SCOPES declares version(s) {orphaned} absent from SCHEMA_EPOCHS"
        )

    records = []
    for version in sorted(ledger):
        kind, note = ledger[version]
        scope = _SCOPES[version]
        _check_scope_shape(version, note, scope)
        records.append(EpochRecord(
            version=version,
            kind=kind,
            note=note,
            certainty=scope.certainty,
            effective_date=scope.effective_date,
            columns=scope.columns,
            condition=scope.condition,
            row_discriminator=scope.row_discriminator,
        ))
    return tuple(records)


def _check_scope_shape(version: int, note: str, scope: _Scope) -> None:
    """Enforce the invariant each :class:`Certainty` promises."""
    if scope.certainty is Certainty.DATED:
        if scope.effective_date is None:
            raise EpochScopeError(f"epoch {version} is DATED but declares no date")
        if scope.condition is not None:
            raise EpochScopeError(f"epoch {version} is DATED but names a condition")
        # The declared date must be the date the note itself states, so the
        # declaration cannot drift away from the prose it claims to summarise.
        if scope.effective_date.isoformat() not in note:
            raise EpochScopeError(
                f"epoch {version} declares {scope.effective_date.isoformat()}, "
                "which does not appear in its ledger note"
            )
        return

    if scope.effective_date is not None:
        raise EpochScopeError(
            f"epoch {version} is {scope.certainty.name} and must carry no date"
        )
    if scope.certainty is Certainty.CONDITIONAL and not scope.condition:
        raise EpochScopeError(
            f"epoch {version} is CONDITIONAL and must name its condition"
        )
    if scope.certainty is Certainty.BASELINE and scope.columns:
        raise EpochScopeError(
            f"epoch {version} is BASELINE and must govern no column"
        )


EPOCHS: tuple[EpochRecord, ...] = _build()

_BY_VERSION: dict[int, EpochRecord] = {e.version: e for e in EPOCHS}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def epochs() -> tuple[EpochRecord, ...]:
    """Every ledger row with its scope, oldest first."""
    return EPOCHS


def epoch(version: int) -> EpochRecord:
    """One ledger row by version."""
    try:
        return _BY_VERSION[version]
    except KeyError:
        raise KeyError(f"no schema epoch with version {version}") from None


def epochs_touching(table: str, column: str) -> tuple[EpochRecord, ...]:
    """Every epoch whose declared scope covers ``table.column``, ignoring dates."""
    key = (table.casefold(), column.casefold())
    return tuple(
        e for e in EPOCHS
        if key in {(t.casefold(), c.casefold()) for t, c in e.columns}
    )


def governing_epochs(
    table: str, column: str, at: str | datetime | date
) -> EpochAnswer:
    """Which epochs were already in force for ``table.column`` at ``at``.

    This is the value *as written*: the rules a reader must know to interpret it
    on its own terms. For "may I compare it to a value written today", ask
    :func:`crossed_since` instead — that is the more useful question.
    """
    when = _as_utc(at)
    settled: list[EpochRecord] = []
    undetermined: list[Undetermined] = []

    for candidate in epochs_touching(table, column):
        if candidate.certainty is Certainty.CONDITIONAL:
            undetermined.append(_conditional(candidate))
            continue
        assert candidate.effective_date is not None  # DATED, per _check_scope_shape
        if candidate.effective_date == when.date():
            undetermined.append(_same_day(candidate))
        elif candidate.effective_date < when.date():
            settled.append(candidate)

    return EpochAnswer(tuple(settled), tuple(undetermined))


def crossed_since(
    table: str,
    column: str,
    written_at: str | datetime | date,
    *,
    until: str | datetime | date | None = None,
) -> EpochAnswer:
    """Which epochs fall between when a row was written and now.

    The reader's real question: *is this stored value comparable to one written
    today, and if not, why not?* An empty answer means directly comparable. A
    settled epoch 6 means it is not, and ``note`` says exactly why.

    ``until`` bounds the far end — pass the timestamp of the row being compared
    against — and applies only to dated epochs. A conditional epoch cannot be
    excluded by a bound, because it cannot be placed on the timeline at all.
    """
    written = _as_utc(written_at)
    ceiling = None if until is None else _as_utc(until).date()
    settled: list[EpochRecord] = []
    undetermined: list[Undetermined] = []

    for candidate in epochs_touching(table, column):
        if candidate.certainty is Certainty.CONDITIONAL:
            undetermined.append(_conditional(candidate))
            continue
        assert candidate.effective_date is not None  # DATED, per _check_scope_shape
        if ceiling is not None and candidate.effective_date > ceiling:
            continue
        if candidate.effective_date == written.date():
            undetermined.append(_same_day(candidate))
        elif candidate.effective_date > written.date():
            settled.append(candidate)

    return EpochAnswer(tuple(settled), tuple(undetermined))


def _conditional(record: EpochRecord) -> Undetermined:
    detail = f"no universal effective date: {record.condition}"
    if record.row_discriminator:
        detail += (
            f"; the affected rows are SELECT-able rather than datable: "
            f"{record.row_discriminator}"
        )
    return Undetermined(record, Uncertainty.CONDITIONAL, detail)


def _same_day(record: EpochRecord) -> Undetermined:
    return Undetermined(
        record,
        Uncertainty.SAME_DAY,
        f"the row was written on {record.effective_date.isoformat()}, the epoch's "
        "own effective date; the ledger records days, not instants, so which side "
        "of the change this row fell on is not recorded",
    )


def _as_utc(value: str | datetime | date) -> datetime:
    """Coerce a stored timestamp to an aware UTC datetime.

    Naive input is read as UTC: ``_now_iso`` has always written aware UTC, and a
    naive timestamp in this store is an older writer's UTC, never local time.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        moment = datetime.fromisoformat(str(value).strip())

    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
