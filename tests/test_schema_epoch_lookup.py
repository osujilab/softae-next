"""Reading the epoch ledger: which epoch governs a stored value.

``tests/test_schema_epoch.py`` covers the ledger as a *table* — that it is
seeded, append-only, and refuses a bogus kind. This file covers the ledger as a
*lookup*: given a table, a column and a row timestamp, which epochs stand
between that row and one written today.

Two properties carry most of the weight here. The **drift guard** — every
``SCHEMA_EPOCHS`` version has a scope entry and vice versa — is the only thing
that makes a companion scope table safe rather than a second place to forget.
And the **conditional/settled split**: epoch 5 has no universal date, so any
answer that quietly folded it into "nothing happened" would report *comparable*
where the truth is *nobody recorded*.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from softae.core.data_store import SCHEMA_EPOCHS
from softae.core.schema_epoch import (
    _SCOPES,
    Certainty,
    EpochScopeError,
    Uncertainty,
    _check_scope_shape,
    crossed_since,
    epoch,
    epochs,
    epochs_touching,
    governing_epochs,
)

# Anchored on epoch 6 (2026-08-26, data-epoch, fit_results.arc_state). Located by
# VERSION, never by position or by kind — epochs 2, 5 and 6 are all data-epochs,
# so anything positional here would silently start reading a different row on the
# next append.
BEFORE_EPOCH_6 = "2026-08-20T12:00:00+00:00"
AFTER_EPOCH_6 = "2026-08-27T12:00:00+00:00"
ON_EPOCH_6 = "2026-08-26T09:00:00+00:00"


def _versions(records) -> list[int]:
    return [r.version for r in records]


# ---------------------------------------------------------------------------
# The drift guard — the reason a companion scope table is acceptable at all
# ---------------------------------------------------------------------------


class TestScopeTracksTheLedger:
    def test_every_ledger_version_has_a_scope_entry(self) -> None:
        """Append epoch 7 without scoping it and this is what goes red.

        Without it, a query about epoch 7's columns would answer "no epoch
        touches this" — the wrong answer wearing the safe answer's clothes.
        """
        assert set(_SCOPES) >= {v for v, _, _ in SCHEMA_EPOCHS}

    def test_every_scope_entry_names_a_ledger_version(self) -> None:
        """The other direction: a scope for an epoch nobody seeded is fiction."""
        assert {v for v, _, _ in SCHEMA_EPOCHS} >= set(_SCOPES)

    def test_the_lookup_exposes_exactly_the_ledgers_versions(self) -> None:
        assert _versions(epochs()) == sorted(v for v, _, _ in SCHEMA_EPOCHS)

    def test_kind_and_note_are_verbatim_from_the_ledger(self) -> None:
        """The lookup joins; it does not paraphrase. The notes are the spec."""
        ledger = {v: (k, n) for v, k, n in SCHEMA_EPOCHS}
        for record in epochs():
            assert (record.kind, record.note) == ledger[record.version]

    def test_a_dated_epochs_declared_date_appears_in_its_own_note(self) -> None:
        """Binds the declaration to the prose it claims to summarise.

        The scope table states an effective date the tuple does not carry. If
        that date is not the one the note states, the scope has drifted from the
        ledger and the lookup is answering from a different document.
        """
        for record in epochs():
            if record.certainty is Certainty.DATED:
                assert record.effective_date.isoformat() in record.note

    def test_a_dated_scope_without_a_date_is_refused_at_build_time(self) -> None:
        """The invariant is enforced in production code, not only asserted here."""
        broken = _SCOPES[6].__class__(
            columns=frozenset({("fit_results", "arc_state")}),
            certainty=Certainty.DATED,
            effective_date=None,
        )
        with pytest.raises(EpochScopeError, match="declares no date"):
            _check_scope_shape(6, epoch(6).note, broken)

    def test_a_conditional_scope_carrying_a_date_is_refused_at_build_time(self) -> None:
        broken = _SCOPES[5].__class__(
            columns=frozenset({("fit_results", "R1")}),
            certainty=Certainty.CONDITIONAL,
            effective_date=date(2026, 8, 18),
            condition="armed",
        )
        with pytest.raises(EpochScopeError, match="must carry no date"):
            _check_scope_shape(5, epoch(5).note, broken)


# ---------------------------------------------------------------------------
# crossed_since — "is this value comparable to one written today?"
# ---------------------------------------------------------------------------


class TestCrossedSince:
    def test_a_row_predating_epoch_six_reports_epoch_six_crossed(self) -> None:
        answer = crossed_since("fit_results", "arc_state", BEFORE_EPOCH_6)
        assert _versions(answer.settled) == [6]
        assert answer.undetermined == ()
        assert answer

    def test_a_row_written_after_epoch_six_reports_nothing_crossed(self) -> None:
        answer = crossed_since("fit_results", "arc_state", AFTER_EPOCH_6)
        assert answer.settled == ()
        assert answer.undetermined == ()
        assert not answer

    def test_a_column_no_epoch_touches_reports_nothing_for_any_timestamp(self) -> None:
        """`measurements.npts` is named by no ledger note, at any date."""
        for when in ("1999-01-01T00:00:00+00:00", BEFORE_EPOCH_6, AFTER_EPOCH_6):
            answer = crossed_since("measurements", "npts", when)
            assert not answer, when

    def test_a_row_written_on_the_effective_date_is_undetermined_not_settled(
            self) -> None:
        """The ledger records days; the row records an instant.

        Reporting "not crossed" for a row written that morning would assert
        something the ledger never recorded — which side of the change it fell
        on. The whole point of this module is not to do that.
        """
        answer = crossed_since("fit_results", "arc_state", ON_EPOCH_6)
        assert answer.settled == ()
        assert [u.epoch.version for u in answer.undetermined] == [6]
        assert answer.undetermined[0].cause is Uncertainty.SAME_DAY
        assert answer

    def test_the_epoch_carries_its_note_so_a_caller_can_say_why(self) -> None:
        """Records, not formatted strings: the caller formats, the helper does not."""
        crossed = crossed_since("fit_results", "arc_state", BEFORE_EPOCH_6).settled[0]
        assert crossed.kind == "data-epoch"
        assert "arc_closure()" in crossed.note
        assert "THERE IS NO PER-ROW DISCRIMINATOR" in crossed.note

    def test_an_until_bound_excludes_an_epoch_the_comparison_row_also_predates(
            self) -> None:
        """Two rows both written before epoch 6 are comparable to each other."""
        answer = crossed_since(
            "fit_results", "arc_state", "2026-08-01T00:00:00+00:00",
            until="2026-08-20T00:00:00+00:00")
        assert not answer

    def test_an_offset_timestamp_is_placed_on_its_utc_day_not_its_local_one(
            self) -> None:
        """Chosen so the offset moves the row across the epoch's day boundary.

        `2026-08-25T22:00-05:00` is `2026-08-26T03:00Z` — the epoch's own day,
        so undetermined. The same wall clock read naively is the day before, so
        settled. A comparison that dropped the offset would collapse the two.
        """
        offset = crossed_since("fit_results", "arc_state", "2026-08-25T22:00:00-05:00")
        assert offset.settled == ()
        assert offset.undetermined[0].cause is Uncertainty.SAME_DAY

        naive = crossed_since("fit_results", "arc_state", "2026-08-25T22:00:00")
        assert _versions(naive.settled) == [6]

        aware = crossed_since(
            "fit_results", "arc_state",
            datetime(2026, 8, 26, 3, 0, tzinfo=timezone.utc))
        assert aware.settled == ()
        assert aware.undetermined[0].cause is Uncertainty.SAME_DAY


# ---------------------------------------------------------------------------
# Epoch 5 — the one that cannot be dated
# ---------------------------------------------------------------------------


class TestConditionalEpoch:
    def test_epoch_five_is_conditional_and_carries_no_date(self) -> None:
        """Its note: 'the epoch begins ... when someone arms the flag'.

        A helper returning a confident date for epoch 5 would be lying, so there
        is no date to read: `effective_date` is None, and a caller comparing it
        to a timestamp raises rather than silently succeeding.
        """
        five = epoch(5)
        assert five.certainty is Certainty.CONDITIONAL
        assert five.effective_date is None
        assert "two_point_open" in five.condition
        with pytest.raises(TypeError):
            _ = five.effective_date < date(2026, 8, 18)

    def test_epoch_five_never_appears_among_settled_crossings(self) -> None:
        """Not omitted either — omission would answer 'comparable' from ignorance."""
        for when in ("2026-01-01T00:00:00+00:00", AFTER_EPOCH_6):
            answer = crossed_since("fit_results", "R1", when)
            assert answer.settled == (), when
            assert [u.epoch.version for u in answer.undetermined] == [5], when
            assert answer.undetermined[0].cause is Uncertainty.CONDITIONAL
            assert answer, when

    def test_epoch_five_records_the_per_row_discriminator_its_note_names(self) -> None:
        """'You cannot date it, but you can SELECT it' is the useful answer."""
        assert epoch(5).row_discriminator == "fit_results.engine = 'gated_two_point'"
        detail = crossed_since("fit_results", "R1", BEFORE_EPOCH_6).undetermined[0].detail
        assert "gated_two_point" in detail

    def test_epoch_six_records_no_discriminator_because_it_has_none(self) -> None:
        """The asymmetry is the reason epoch 6 had to be a ledger row at all.

        Epoch 5 can point at `fit_results.engine`; `arc_state` has no column
        recording which rule produced it, so the date is the only separator.
        """
        assert epoch(6).row_discriminator is None


# ---------------------------------------------------------------------------
# The kind distinction, and scope membership
# ---------------------------------------------------------------------------


class TestKindSurvivesTheLookup:
    def test_a_schema_epoch_is_distinguished_from_a_data_epoch(self) -> None:
        """Epoch 3 renamed columns and moved no number; epoch 6 moved meaning.

        `kind` is the point of the ledger, and a lookup that flattened it would
        report the conditions rename as a comparability break — it is not one.
        """
        rename = epoch(3)
        assert rename.kind == "schema"
        assert rename.changes_meaning is False

        for version in (2, 5, 6):
            assert epoch(version).kind == "data-epoch"
            assert epoch(version).changes_meaning is True

    def test_a_row_predating_the_rename_reports_a_schema_epoch_not_a_break(
            self) -> None:
        answer = crossed_since("conditions", "chamber_air_C", "2026-08-01T00:00:00Z")
        assert _versions(answer.settled) == [3]
        assert answer.settled[0].changes_meaning is False

    def test_the_rename_is_findable_under_the_pre_rename_column_name(self) -> None:
        """A reader holding old code asks about `temp_pv_C`, not `chamber_air_C`.

        Answering 'no epoch touches that column' because the name no longer
        exists would be the least useful true statement available.
        """
        assert _versions(epochs_touching("conditions", "temp_pv_C")) == [3]
        assert _versions(epochs_touching("conditions", "temp_sp_C")) == [3]

    def test_column_matching_is_case_insensitive_as_sqlite_identifiers_are(
            self) -> None:
        assert _versions(epochs_touching("FIT_RESULTS", "r1")) == [5]

    def test_the_baseline_epoch_governs_no_column(self) -> None:
        """Version 1 is the floor of the ledger, not a change to anything.

        Scoping it database-wide would make every query on every old row report
        a crossing, which is noise that would train readers to ignore the answer.
        """
        one = epoch(1)
        assert one.certainty is Certainty.BASELINE
        assert one.columns == frozenset()
        assert all(1 not in _versions(epochs_touching(t, c))
                   for t, c in [("fit_results", "arc_state"),
                                ("conditions", "temperature_C"),
                                ("formulations", "deposit_area_mm2")])

    def test_the_deposit_area_epoch_scopes_the_thickness_it_moved(self) -> None:
        """`predicted_thickness_um` is volume/area, so it moved by the same 4.676x."""
        assert _versions(
            epochs_touching("formulations", "predicted_thickness_um")) == [2]

    def test_sigma_is_deliberately_outside_the_deposit_area_epoch(self) -> None:
        """`electrode_t_cm` arrives from `[eis.cell]` config, not from the cast area.

        Scoping it to epoch 2 would report a break the correction never caused.
        This is a *reading* of the note, recorded so a later reader can dispute
        it rather than rediscover it.
        """
        assert epochs_touching("fit_results", "sigma_S_per_cm") == ()
        assert epochs_touching("fit_results", "electrode_t_cm") == ()


# ---------------------------------------------------------------------------
# governing_epochs — the value as written
# ---------------------------------------------------------------------------


class TestGoverningEpochs:
    def test_an_epoch_not_yet_in_force_does_not_govern_the_row(self) -> None:
        answer = governing_epochs("fit_results", "arc_state", BEFORE_EPOCH_6)
        assert not answer

    def test_an_epoch_already_in_force_governs_the_row(self) -> None:
        answer = governing_epochs("fit_results", "arc_state", AFTER_EPOCH_6)
        assert _versions(answer.settled) == [6]

    def test_the_conditional_epoch_governs_undeterminably_at_every_date(self) -> None:
        answer = governing_epochs("fit_results", "R1", AFTER_EPOCH_6)
        assert answer.settled == ()
        assert [u.epoch.version for u in answer.undetermined] == [5]

    def test_governing_and_crossed_partition_the_epochs_on_a_column(self) -> None:
        """Every dated epoch on a column either already applied or came later."""
        when = BEFORE_EPOCH_6
        dated = [e for e in epochs_touching("conditions", "chamber_air_C")
                 if e.certainty is Certainty.DATED]
        already = governing_epochs("conditions", "chamber_air_C", when).settled
        later = crossed_since("conditions", "chamber_air_C", when).settled
        assert sorted(_versions(already) + _versions(later)) == _versions(dated)

    def test_an_unknown_version_is_a_loud_key_error(self) -> None:
        with pytest.raises(KeyError, match="no schema epoch with version 99"):
            epoch(99)
