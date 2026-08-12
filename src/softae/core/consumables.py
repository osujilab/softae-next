"""Waste capacity and spare plates — the physical limits on unattended time (P5.4).

Stock (P5.1) answers "can we keep dispensing?". These answer the two questions
that bound a *long* run in the other direction:

* **Waste.** Everything flushed, preconditioned, or wicked ends up in a
  container with a finite volume. Once anti-clog purging exists (P8) that
  container fills on a *schedule* — roughly 3.8 mL/day at the specified rate —
  regardless of how many trials run, so waste becomes a wall-clock limit rather
  than a per-trial one. Overflowing it is a bench spill, not a soft failure.
* **Spare plates.** Drop-cast wells are single-use, so a long campaign consumes
  boards. When the last one is gone the campaign cannot continue no matter how
  much stock is left, and the board-exchange prompt would otherwise ask the
  operator to install a plate that does not exist.

Both mirror :class:`~softae.core.reservoir.ReservoirLedger`'s conventions
deliberately: **undeclared is unknown, never empty or full**, so adding these
cannot break a bench that has not declared them; thresholds warn before they
stop; and the stop raises :class:`SafetyError`, which the loop already treats as
park-immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from softae.errors import SafetyError

logger = structlog.get_logger(__name__)

#: Warn when the waste container passes this fraction of capacity.
DEFAULT_WASTE_WARN_FRACTION = 0.8
#: Warn when spare plates fall to this many.
DEFAULT_SPARE_BOARD_WARN = 1


class RigStateStore(Protocol):
    """The narrow persistence surface these need."""

    def waste_level_uL(self) -> float | None: ...
    def set_waste_level(self, level_uL: float) -> None: ...
    def spare_boards(self) -> int | None: ...
    def set_spare_boards(self, n: int) -> None: ...


@dataclass
class WasteStatus:
    """Result of adding volume to the waste container."""

    level_uL: float
    capacity_uL: float | None
    warned: bool = False

    @property
    def fraction(self) -> float | None:
        if not self.capacity_uL:
            return None
        return self.level_uL / self.capacity_uL

    @property
    def headroom_uL(self) -> float | None:
        if self.capacity_uL is None:
            return None
        return max(0.0, self.capacity_uL - self.level_uL)


class WasteLedger:
    """Tracks the waste container against a declared capacity.

    An **undeclared capacity means unmanaged**: volume is still accumulated (so
    the operator can read it), but nothing is ever refused. Declaring a capacity
    is what turns the record into an interlock.
    """

    def __init__(
        self,
        store: "RigStateStore | None" = None,
        *,
        capacity_uL: float | None = None,
        warn_fraction: float = DEFAULT_WASTE_WARN_FRACTION,
        on_warn: Any = None,
    ) -> None:
        if capacity_uL is not None and float(capacity_uL) <= 0:
            raise ValueError("waste capacity must be > 0")
        self._store = store
        self.capacity_uL = None if capacity_uL is None else float(capacity_uL)
        self.warn_fraction = float(warn_fraction)
        self._on_warn = on_warn
        self._level: float | None = None

    def level_uL(self) -> float:
        """Accumulated waste. Absent record reads as empty, which is the
        operator's implicit starting state for a fresh container."""
        if self._level is None:
            stored = None
            if self._store is not None:
                try:
                    stored = self._store.waste_level_uL()
                except Exception:
                    logger.warning("waste_read_failed", exc_info=True)
            self._level = float(stored or 0.0)
        return self._level

    def empty(self) -> None:
        """Operator emptied the container."""
        self._set(0.0)
        logger.info("waste_emptied")

    def _set(self, level: float) -> None:
        self._level = max(0.0, float(level))
        if self._store is not None:
            try:
                self._store.set_waste_level(self._level)
            except Exception:
                logger.warning("waste_persist_failed", exc_info=True)

    def check(self, volume_uL: float, *, instrument: str = "waste") -> None:
        """Refuse a transfer that would overflow the container."""
        if self.capacity_uL is None:
            return
        projected = self.level_uL() + max(0.0, float(volume_uL))
        if projected > self.capacity_uL:
            raise SafetyError(
                f"Waste container would overflow: {projected:.0f} µL exceeds the "
                f"{self.capacity_uL:.0f} µL capacity. Empty it before continuing.",
                instrument=instrument,
                requested=float(volume_uL),
                limit=max(0.0, self.capacity_uL - self.level_uL()),
            )

    def add(self, volume_uL: float) -> WasteStatus:
        """Record volume routed to waste. Call :meth:`check` first to enforce."""
        before = self.level_uL()
        after = before + max(0.0, float(volume_uL))
        self._set(after)

        warned = False
        if self.capacity_uL:
            threshold = self.capacity_uL * self.warn_fraction
            warned = before <= threshold < after      # fires once, on crossing
            if warned:
                logger.warning(
                    "waste_high", level_uL=after, capacity_uL=self.capacity_uL,
                    fraction=after / self.capacity_uL,
                )
                if self._on_warn is not None:
                    try:
                        self._on_warn(after, self.capacity_uL)
                    except Exception:
                        logger.warning("waste_warn_hook_failed", exc_info=True)

        return WasteStatus(after, self.capacity_uL, warned)

    def check_and_add(self, volume_uL: float, *, instrument: str = "waste") -> WasteStatus:
        self.check(volume_uL, instrument=instrument)
        return self.add(volume_uL)


class BoardInventory:
    """Fresh electrode plates on hand.

    ``None`` means undeclared — a board exchange proceeds as it always has. Once
    declared, the count decrements on each swap and reaching zero stops the
    campaign *before* the operator is asked to install a plate that does not
    exist.
    """

    def __init__(
        self,
        store: "RigStateStore | None" = None,
        *,
        warn_at: int = DEFAULT_SPARE_BOARD_WARN,
    ) -> None:
        self._store = store
        self.warn_at = int(warn_at)
        self._count: int | None = None
        self._loaded = False

    def remaining(self) -> int | None:
        if not self._loaded:
            self._loaded = True
            if self._store is not None:
                try:
                    self._count = self._store.spare_boards()
                except Exception:
                    logger.warning("spare_boards_read_failed", exc_info=True)
                    self._count = None
        return self._count

    def declare(self, n: int) -> None:
        """Operator states how many fresh plates are on hand."""
        self._count = max(0, int(n))
        self._loaded = True
        if self._store is not None:
            try:
                self._store.set_spare_boards(self._count)
            except Exception:
                logger.warning("spare_boards_persist_failed", exc_info=True)
        logger.info("spare_boards_declared", n=self._count)

    @property
    def is_managed(self) -> bool:
        return self.remaining() is not None

    def check(self, *, instrument: str = "board") -> None:
        """Raise if no fresh plate is available for an exchange."""
        n = self.remaining()
        if n is not None and n <= 0:
            raise SafetyError(
                "No spare electrode boards remain. The campaign cannot continue "
                "past a full board until fresh plates are loaded and declared.",
                instrument=instrument,
                requested=1,
                limit=0,
            )

    def consume(self) -> int | None:
        """Take one plate for an exchange. Returns what is left."""
        n = self.remaining()
        if n is None:
            return None
        self.declare(max(0, n - 1))
        left = self.remaining()
        if left is not None and left <= self.warn_at:
            logger.warning("spare_boards_low", remaining=left)
        return left


def attach_consumables(
    data_store: "Any", *, config: dict[str, Any] | None = None
) -> tuple[WasteLedger, BoardInventory]:
    """Build both trackers against a project store — one wiring path.

    Mirrors ``attach_reservoir_ledger``: a single helper the GUI and the headless
    CLI both call, so neither surface can end up with weaker limits than the
    other.
    """
    if config is None:
        try:
            from softae.config.loader import safety

            config = safety()
        except Exception:
            config = {}

    try:
        capacity = config.get("waste_capacity_uL")
        capacity = None if capacity is None else float(capacity)
    except (TypeError, ValueError):
        capacity = None
    try:
        warn_fraction = float(
            config.get("waste_warn_fraction", DEFAULT_WASTE_WARN_FRACTION))
    except (TypeError, ValueError):
        warn_fraction = DEFAULT_WASTE_WARN_FRACTION

    waste = WasteLedger(
        data_store, capacity_uL=capacity, warn_fraction=warn_fraction)
    boards = BoardInventory(data_store)
    logger.info(
        "consumables_attached", waste_capacity_uL=capacity,
        spare_boards=boards.remaining(),
    )
    return waste, boards
