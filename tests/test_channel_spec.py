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
