"""Warn-before-recast guard for single-use electrode wells.

Drop-cast wells are single use: casting twice into one wastes the sample and
corrupts the measurement.  When a manual pump or an HT run is about to cast into
a well the :class:`~softae.core.data_store.DataStore` already records as
occupied, this asks the operator whether the board has been physically replaced
(a fresh plate resets occupancy), the cast should proceed anyway on the same
board (a deliberate re-cast/touch-up), or be cancelled.

The check runs against a ``board_id`` (the project's monotonic board counter);
answering *replaced* advances to a fresh, empty board id so subsequent casts do
not collide with the retired plate's record. Answering *cast anyway* keeps the
same board id — the well's occupancy record is simply re-affirmed.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox, QWidget

if TYPE_CHECKING:
    from softae.core.data_store import DataStore


class BoardReplacedDecision(Enum):
    """Operator's answer to the occupied-well prompt."""

    FRESH = "fresh"        # board physically replaced → cast on a fresh board id
    CAST_ANYWAY = "cast_anyway"  # same board, deliberate re-cast into the well
    CANCEL = "cancel"      # do not cast (occupancy record stands)


def occupied_conflicts(
    data_store: "DataStore | None", board_id: int, electrodes: "set[int] | list[int]"
) -> set[int]:
    """Electrodes among ``electrodes`` already recorded occupied on ``board_id``."""
    if data_store is None:
        return set()
    try:
        occupied = set(data_store.occupied_electrodes(board_id))
    except Exception:
        return set()
    return {int(e) for e in electrodes} & occupied


def prompt_board_replaced(
    parent: QWidget | None, board_id: int, conflicts: "set[int]"
) -> BoardReplacedDecision:
    """Modal: some wells are recorded occupied — how should this cast proceed?

    Returns :attr:`BoardReplacedDecision.FRESH` (replaced → proceed on a fresh
    board), :attr:`BoardReplacedDecision.CAST_ANYWAY` (same board, deliberate
    re-cast), or :attr:`BoardReplacedDecision.CANCEL` (dismissed or declined).
    """
    wells = ", ".join(f"E{e}" for e in sorted(conflicts))
    box = QMessageBox(parent)
    box.setWindowTitle("Occupied Well")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText(
        f"This cast targets {'a well' if len(conflicts) == 1 else 'wells'} already "
        f"recorded as occupied on board {board_id}: {wells}."
    )
    box.setInformativeText(
        "Drop-cast wells are single use. Has the electrode board been replaced "
        "with a fresh plate?\n\n"
        "• Board replaced — occupancy resets; cast on a fresh board.\n"
        "• Same board, cast anyway — deliberate re-cast into the existing well.\n"
        "• Cancel — do not cast (the existing record stands)."
    )
    fresh_btn = box.addButton("Board replaced", QMessageBox.ButtonRole.YesRole)
    anyway_btn = box.addButton("Same board, cast anyway", QMessageBox.ButtonRole.NoRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(box.button(QMessageBox.StandardButton.Cancel))
    box.exec()
    clicked = box.clickedButton()
    if clicked is fresh_btn:
        return BoardReplacedDecision.FRESH
    if clicked is anyway_btn:
        return BoardReplacedDecision.CAST_ANYWAY
    return BoardReplacedDecision.CANCEL


def prompt_log_board_swap(
    parent: QWidget | None, data_store: "DataStore | None"
) -> int | None:
    """Operator-initiated board swap: confirm, advance the board, reset positions.

    The counterpart to :func:`prompt_board_replaced`, which only fires *reactively*
    when a cast happens to collide with an occupied well. An operator who swaps a
    plate between runs has no such collision to trigger on, so without this the
    stale occupancy sits there until the next conflict — and the electrode map
    keeps showing the retired plate's wells as occupied.

    Returns the new board id, or ``None`` if there is no store or the operator
    declined. Confirmation is required because the pointer is **monotonic**: an
    accidental swap cannot be undone by swapping back, it just burns a board id
    and orphans the real plate's occupancy record.
    """
    if data_store is None:
        return None
    try:
        board_id = data_store.current_board_id()
        occupied = data_store.occupied_electrodes(board_id)
    except Exception:
        return None

    box = QMessageBox(parent)
    box.setWindowTitle("Log Board Swap")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText(
        f"Log a fresh electrode board?\n\nBoard {board_id} currently has "
        f"{len(occupied)} well(s) recorded as cast."
    )
    box.setInformativeText(
        f"This advances to board {board_id + 1} and clears the occupancy display, "
        "so every well is available again.\n\n"
        f"Board {board_id}'s records are kept — past runs stay interpretable. "
        "Only do this once the plate is physically replaced; the board counter "
        "cannot be moved back."
    )
    confirm = box.addButton("Log swap", QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(box.button(QMessageBox.StandardButton.Cancel))
    box.exec()
    if box.clickedButton() is not confirm:
        return None

    new_id = data_store.advance_board()
    # Durable provenance: an unattended campaign's data is only interpretable if
    # you can tell which physical plate a sample landed on.
    try:
        from softae.core.alerts import INFO, Alert, raise_alert

        raise_alert(
            Alert(
                kind="board_swap",
                severity=INFO,
                message=(
                    f"Board swap logged by operator: board {board_id} → {new_id} "
                    f"({len(occupied)} well(s) had been cast)."
                ),
                details={"previous_board": board_id, "new_board": new_id,
                         "previous_occupied": sorted(occupied)},
            ),
            data_store=data_store,
        )
    except Exception:
        pass
    return new_id
