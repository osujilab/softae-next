"""Tests for sequential electrode allocation + board exchange."""

from __future__ import annotations

import pytest

from softae.core.electrode_allocator import (
    DEFAULT_BOARD_CAPACITY,
    ElectrodeAllocator,
)


def test_default_capacity_is_32():
    assert DEFAULT_BOARD_CAPACITY == 32
    assert ElectrodeAllocator().capacity == 32


def test_sequential_allocation_advances_cursor():
    a = ElectrodeAllocator(capacity=32)
    assert a.allocate(4).channels == [1, 2, 3, 4]
    assert a.allocate(4).channels == [5, 6, 7, 8]
    assert a.remaining == 32 - 8


def test_allocate_reports_overflow_when_board_fills():
    a = ElectrodeAllocator(capacity=6)
    a.allocate(4)  # uses 1..4, 2 remain
    alloc = a.allocate(4)  # only 2 fit
    assert alloc.channels == [5, 6]
    assert alloc.overflow == 2
    assert a.board_full


def test_swap_board_resets_cursor_and_bumps_index():
    a = ElectrodeAllocator(capacity=4)
    a.allocate(4)
    assert a.board_full
    assert a.swap_board() == 1
    assert a.board_index == 1
    assert not a.board_full
    assert a.allocate(2).channels == [1, 2]  # fresh board, from the start


def test_full_board_allocates_nothing_until_swap():
    a = ElectrodeAllocator(capacity=3)
    a.allocate(3)
    alloc = a.allocate(2)
    assert alloc.channels == []
    assert alloc.overflow == 2


def test_custom_start_electrode():
    a = ElectrodeAllocator(capacity=32, start=29)
    alloc = a.allocate(6)
    assert alloc.channels == [29, 30, 31, 32]  # only 4 fit from electrode 29
    assert alloc.overflow == 2


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ElectrodeAllocator(capacity=0)
    with pytest.raises(ValueError):
        ElectrodeAllocator(capacity=8, start=9)
