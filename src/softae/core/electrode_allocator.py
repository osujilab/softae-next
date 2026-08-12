"""Sequential electrode allocation with board-capacity + exchange.

Drop-cast wells are **single-use**, so every deposited sample consumes a fresh
electrode.  A physical board/plate exposes a fixed number of addressable
electrodes (the "non-intervening" limit — default 32, matching the two-pico
routing of channels 1–16 / 17–32).  :class:`ElectrodeAllocator` hands out the
next free electrode(s) on the current board and reports when the board is full
so the campaign can prompt a board exchange and continue.

The allocator is pure book-keeping (no hardware); the loop drives the physical
swap + equilibration when :meth:`allocate` reports an overflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default electrodes per board (channels 1–16 on pico1, 17–32 on pico2).
DEFAULT_BOARD_CAPACITY = 32


@dataclass
class ElectrodeAllocation:
    """Electrodes granted on the current board for one request.

    ``channels`` are the (1-based) electrodes to use now; ``overflow`` is how
    many of the requested samples did **not** fit and require a fresh board.
    """

    channels: list[int]
    board_index: int
    overflow: int


@dataclass
class ElectrodeAllocator:
    """Hands out fresh electrodes sequentially, one board at a time.

    Parameters
    ----------
    capacity : int
        Electrodes per board (default :data:`DEFAULT_BOARD_CAPACITY`).
    start : int
        First electrode on a board (1-based, default 1).
    """

    capacity: int = DEFAULT_BOARD_CAPACITY
    start: int = 1
    board_index: int = 0
    #: Electrodes already cast on this board (1-based) — skipped, never reused.
    #: Populated on resume from the durable occupancy record. Allocation walks
    #: *past* these rather than starting after the highest one, so wells left
    #: free by a skipped channel are still used. With 32 wells a board and
    #: multi-hour anneals, silently abandoning free wells is expensive.
    occupied: frozenset[int] = frozenset()
    _cursor: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("board capacity must be >= 1")
        if not (1 <= self.start <= self.capacity):
            raise ValueError(f"start electrode must be in 1..{self.capacity}")
        self.occupied = frozenset(int(e) for e in self.occupied)
        self._cursor = self.start
        self._skip_occupied()

    def _skip_occupied(self) -> None:
        """Move the cursor to the next electrode that is not already cast."""
        while self._cursor <= self.capacity and self._cursor in self.occupied:
            self._cursor += 1

    @property
    def remaining(self) -> int:
        """Electrodes still free on the current board (gaps included)."""
        if self._cursor > self.capacity:
            return 0
        return sum(
            1 for e in range(self._cursor, self.capacity + 1)
            if e not in self.occupied
        )

    @property
    def board_full(self) -> bool:
        return self.remaining == 0

    def allocate(self, n: int) -> ElectrodeAllocation:
        """Grant up to *n* fresh electrodes on the current board.

        Skips any electrode recorded in :attr:`occupied`; ``overflow`` counts the
        requests that did not fit (they need a :meth:`swap_board` first).
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        channels: list[int] = []
        while len(channels) < n and self._cursor <= self.capacity:
            if self._cursor not in self.occupied:
                channels.append(self._cursor)
            self._cursor += 1
        self._skip_occupied()
        return ElectrodeAllocation(
            channels=channels, board_index=self.board_index,
            overflow=n - len(channels),
        )

    def swap_board(self) -> int:
        """Advance to a fresh board (cursor back to ``start``). Returns new index.

        A fresh plate has no history, so the occupancy set is cleared — carrying
        it over would blank out wells on a board that has never been cast into.
        """
        self.board_index += 1
        self.occupied = frozenset()
        self._cursor = self.start
        return self.board_index
