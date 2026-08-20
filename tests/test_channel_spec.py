"""Tests for the channel-spec parser (softae.core.channel_spec)."""

from __future__ import annotations

import pytest

from softae.core.channel_spec import ChannelSpecError, parse_channel_spec


def test_single_channel():
    assert parse_channel_spec("5") == [5]


def test_comma_list():
    assert parse_channel_spec("2,4,7") == [2, 4, 7]


def test_range():
    assert parse_channel_spec("5-10") == [5, 6, 7, 8, 9, 10]


def test_mixed_list_and_ranges():
    assert parse_channel_spec("2,4,5-10") == [2, 4, 5, 6, 7, 8, 9, 10]


def test_whitespace_is_ignored():
    assert parse_channel_spec("  1 , 3 ,  5 - 7 ") == [1, 3, 5, 6, 7]


def test_result_is_sorted_and_deduped():
    assert parse_channel_spec("10,2,2,3-5,4") == [2, 3, 4, 5, 10]


def test_tolerates_trailing_and_double_commas():
    assert parse_channel_spec("2,,4,") == [2, 4]


def test_empty_raises():
    with pytest.raises(ChannelSpecError):
        parse_channel_spec("")
    with pytest.raises(ChannelSpecError):
        parse_channel_spec("   ")


def test_non_numeric_raises():
    with pytest.raises(ChannelSpecError):
        parse_channel_spec("2,x,4")


def test_reversed_range_raises():
    with pytest.raises(ChannelSpecError, match="reversed"):
        parse_channel_spec("10-5")


def test_out_of_range_raises():
    with pytest.raises(ChannelSpecError, match="out of range"):
        parse_channel_spec("0,5", min_ch=1, max_ch=32)
    with pytest.raises(ChannelSpecError, match="out of range"):
        parse_channel_spec("5-40", max_ch=32)


def test_custom_bounds():
    assert parse_channel_spec("1-4", min_ch=1, max_ch=4) == [1, 2, 3, 4]


# ── on_invalid="drop" — the forgiving free-text-field mode ────────────────────
#
# These six pin the Experiment tab's long-standing semantics, which the shared
# parser now owns.  They are subtler than "ignore bad input": a single channel
# out of bounds is dropped, but a *range* that straddles a bound is clamped.


def test_drop_empty_spec_returns_empty_list_instead_of_raising():
    assert parse_channel_spec("", on_invalid="drop") == []
    assert parse_channel_spec("   ", on_invalid="drop") == []


def test_drop_skips_non_numeric_tokens():
    assert parse_channel_spec("2,x,4", on_invalid="drop") == [2, 4]
    assert parse_channel_spec("1-x,5", on_invalid="drop") == [5]


def test_drop_skips_single_channel_out_of_bounds():
    # The Experiment tab's pinned case: "0,1,99" at max_ch=16 -> [1].
    assert parse_channel_spec("0,1,99", max_ch=16, on_invalid="drop") == [1]


def test_drop_clamps_a_range_that_straddles_the_bounds():
    # NOT dropped — clamped.  Contrast with the raise-mode test below.
    assert parse_channel_spec("0-99", max_ch=16, on_invalid="drop") == list(range(1, 17))
    assert parse_channel_spec("14-99", max_ch=16, on_invalid="drop") == [14, 15, 16]


def test_drop_and_raise_disagree_on_a_straddling_range():
    """The asymmetry is deliberate; assert both halves so neither drifts."""
    assert parse_channel_spec("0-99", max_ch=16, on_invalid="drop") == list(range(1, 17))
    with pytest.raises(ChannelSpecError):
        parse_channel_spec("0-99", max_ch=16)


def test_drop_treats_a_reversed_range_as_contributing_nothing():
    assert parse_channel_spec("8-5", on_invalid="drop") == []
    assert parse_channel_spec("8-5,2", on_invalid="drop") == [2]


def test_drop_result_is_sorted_and_deduped():
    assert parse_channel_spec("10,2,2,3-5,4", on_invalid="drop") == [2, 3, 4, 5, 10]


def test_drop_never_raises_even_when_everything_is_bad():
    assert parse_channel_spec("x,y,-,99", max_ch=16, on_invalid="drop") == []


# ── order="as-written" — where channel order is measurement order ─────────────


def test_as_written_preserves_entry_order():
    assert parse_channel_spec("10,2,3-5", order="as-written") == [10, 2, 3, 4, 5]


def test_as_written_dedups_first_wins():
    assert parse_channel_spec("3,1-3", order="as-written") == [3, 1, 2]


def test_as_written_still_enforces_bounds_and_emptiness():
    with pytest.raises(ChannelSpecError, match="out of range"):
        parse_channel_spec("40", max_ch=32, order="as-written")
    with pytest.raises(ChannelSpecError):
        parse_channel_spec("", order="as-written")


def test_as_written_composes_with_drop():
    assert parse_channel_spec("10,x,2,0", max_ch=16,
                              on_invalid="drop", order="as-written") == [10, 2]


def test_unknown_keyword_values_are_rejected():
    with pytest.raises(ValueError):
        parse_channel_spec("1", on_invalid="ignore")
    with pytest.raises(ValueError):
        parse_channel_spec("1", order="reverse")


def test_defaults_are_unchanged():
    """Regression guard: the two new keywords must not move the default path.

    Passing the defaults explicitly must be indistinguishable from omitting
    them, for every shape the original 11 tests cover — values and raises
    alike.  This is what protects the three existing consumers
    (``tools/equilibration.py``, ``tools/eis_validate.py``,
    ``gui/tabs/tab_manual.py``), which call the parser with no keywords.
    """
    for spec in ("5", "2,4,7", "5-10", "2,4,5-10", "  1 , 3 ,  5 - 7 ",
                 "10,2,2,3-5,4", "2,,4,"):
        assert (parse_channel_spec(spec)
                == parse_channel_spec(spec, on_invalid="raise", order="sorted"))

    for bad in ("", "   ", "2,x,4", "10-5", "0,5", "5-40"):
        with pytest.raises(ChannelSpecError):
            parse_channel_spec(bad)
        with pytest.raises(ChannelSpecError):
            parse_channel_spec(bad, on_invalid="raise", order="sorted")
