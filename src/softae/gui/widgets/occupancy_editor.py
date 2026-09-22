"""Hand-edit which wells a board has cast, with a durable record of every override.

Occupancy is otherwise written only by the machine: a cast records a well, and
nothing short of a board swap ever clears one. That leaves no way to record a
cast made by hand at the bench, and no way to correct a mis-record — the
operator's only recourse was to burn a board id and lose the rest of the plate's
account of itself. This dialog is the missing per-well edit.

Two properties keep a hand-edited board as interpretable as a machine-cast one:

* **Every override writes an alert.** An ``occupancy_override`` row carries the
  well, the action, the row that was displaced and a free-text reason, so the
  one case where the database and the bench can diverge is also the one case
  with a human's note attached to it.
* **It refuses while a run holds the rig.** A running campaign reads occupancy
  once at launch and then holds a frozen snapshot, so an override underneath it
  changes nothing that run can see. An editor that appeared to work would be
  worse than one that refuses.

The board swap reachable from here is **staged like every other edit**. The
board pointer is monotonic and cannot be walked back, so the button only
re-renders a fresh, empty grid; the swap itself is written on OK, or not at all.
Cancel therefore means nothing happened.

The core above the dialog is Qt-free on purpose: what changed, what gets written
and what gets warned about is the part worth testing without a display.

Two store methods are **feature-detected, never assumed**: a per-well delete
(``release_electrode``) and a row reader (``electrode_occupancy_rows``). Without
the delete, recorded wells are read-only; without the reader, the
free-confirmation lists every freed well and says the identities could not be
read — "could not check" must not be spelled like "checked and found nothing".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from softae.core.alerts import INFO, Alert, raise_alert
from softae.core.geometry import electrode_count
from softae.core.run_lock import foreign_run_lock
from softae.gui.widgets.occupancy_guard import prompt_log_board_swap
from softae.gui.widgets.rig_owner import campaign_identity, owner_line

#: The two edit directions. Strings rather than an enum because they travel into
#: an alert's ``details`` and out again through JSON.
OCCUPY = "occupy"
FREE = "free"

#: Alert kind, shared with whatever reads the alert log back.
ALERT_KIND = "occupancy_override"

#: Said on every refusal. The asymmetry is the whole reason for the refusal and
#: is not obvious from the outside.
SNAPSHOT_NOTE = (
    "A campaign started after this edit reads occupancy at its own launch; one "
    "already running does not."
)

#: Checkboxes per row in the well grid.
_COLUMNS = 8

#: The swap button's two faces; it toggles rather than acting.
SWAP_STAGE_TEXT = "Log Board Swap…"
SWAP_UNDO_TEXT = "Undo pending swap"


# ── Qt-free core ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OccupancyEdit:
    """One staged change to one well, carrying the row it displaces."""

    electrode: int
    action: str
    previous_row: dict[str, Any] | None = None


def diff_occupancy(
    staged: Iterable[int],
    recorded: Iterable[int],
    rows: dict[int, dict[str, Any]] | None = None,
) -> list[OccupancyEdit]:
    """The edits turning *recorded* occupancy into *staged*, lowest well first.

    A well in both sets, or in neither, produces nothing: an OK pressed on an
    untouched grid must not write an alert claiming a human changed something.
    """
    staged_set = {int(e) for e in staged}
    recorded_set = {int(e) for e in recorded}
    by_electrode = rows or {}
    edits = [OccupancyEdit(e, OCCUPY) for e in staged_set - recorded_set]
    edits += [
        OccupancyEdit(e, FREE, by_electrode.get(e))
        for e in recorded_set - staged_set
    ]
    return sorted(edits, key=lambda edit: edit.electrode)


def grid_range(pcb_config: dict[str, Any] | None, recorded: Iterable[int]) -> range:
    """Well numbers to offer: the board's capacity, extended to cover any record.

    A row written under a larger layout would otherwise be stranded — shown
    nowhere, and so impossible to free.
    """
    highest = max([electrode_count(pcb_config or {})] + [int(e) for e in recorded])
    return range(1, highest + 1)


def effective_board(current_id: int, swap_pending: bool) -> int:
    """The board a staged edit targets: the next one when a swap is staged.

    The swap is not written until OK, so the store's pointer still reads the old
    board while the grid on screen already describes the fresh plate.
    """
    return int(current_id) + 1 if swap_pending else int(current_id)


def override_alert(board_id: int, edit: OccupancyEdit, reason: str = "") -> Alert:
    """The durable trace that a human, not a run, changed this well."""
    verb = "marked cast" if edit.action == OCCUPY else "freed"
    note = (reason or "").strip()
    message = f"Well E{edit.electrode} on board {board_id} {verb} by the operator."
    if note:
        message = f"{message} Reason: {note}"
    return Alert(
        kind=ALERT_KIND,
        severity=INFO,
        message=message,
        details={
            "board_id": int(board_id),
            "electrode": int(edit.electrode),
            "action": edit.action,
            # ``None``, never ``{}``: an empty dict would read back as a row
            # that existed and happened to carry no fields.
            "previous_row": dict(edit.previous_row) if edit.previous_row else None,
            "reason": note or None,
        },
    )


def identity_at_risk(
    edits: Sequence[OccupancyEdit], *, rows_readable: bool
) -> list[OccupancyEdit]:
    """Freed wells worth confirming, because they may carry a sample's identity.

    With no way to read the rows, every freed well qualifies: unknown must not
    be reported in the same words as checked-and-clean.
    """
    frees = [edit for edit in edits if edit.action == FREE]
    if not rows_readable:
        return frees
    return [edit for edit in frees if (edit.previous_row or {}).get("sample_uuid")]


def occupancy_edit_refusal(
    lock_reader: Callable[[], Any] | None = None,
    identify: Callable[[Any], tuple[str, str] | None] = campaign_identity,
) -> str | None:
    """Why editing is refused right now, or ``None`` when the rig is free.

    A reader that *raises* resolves to a refusal rather than to permission, the
    same swallow-into-the-conservative-answer the launch decision performs:
    deferring an edit costs a minute, editing underneath a live run costs that
    run's account of its own board. The reader is resolved at call time instead
    of bound as a default so a caller can substitute one.
    """
    reader = lock_reader if lock_reader is not None else foreign_run_lock
    try:
        lock = reader()
    except Exception as exc:
        return (
            "The rig lock could not be read, so this session assumes a run is "
            f"driving the rig ({exc.__class__.__name__}). Well occupancy is not "
            f"edited underneath one.\n\n{SNAPSHOT_NOTE}"
        )
    if lock is None:
        return None
    return (
        f"{_holder_phrase(lock, identify)} holds the rig, so well occupancy "
        f"cannot be edited from here.\n\n{SNAPSHOT_NOTE}"
    )


def _holder_phrase(lock: Any, identify: Callable[[Any], Any]) -> str:
    """Who holds the rig, named as precisely as the lock allows."""
    try:
        identity = identify(lock)
    except Exception:
        identity = None
    if identity is not None:
        name, run_id = identity
        return f"Campaign '{name}' (run {run_id})" if run_id else f"Campaign '{name}'"
    try:
        return f"Another process — {owner_line(lock)} —"
    except Exception:
        return "Another process"


def _safe(call: Callable[[], Any], default: Any) -> Any:
    """Best-effort read: a closed or absent store must not break the dialog."""
    try:
        return call()
    except Exception:
        return default


def _default_pcb_config() -> dict[str, Any]:
    """The board the rest of the GUI opens on; ``{}`` falls back to a 4×4."""
    try:
        from softae.config.loader import default_pcb_name, pcb_configs

        return dict(pcb_configs().get(default_pcb_name() or "", {}))
    except Exception:
        return {}


# ── The dialog ───────────────────────────────────────────────────────────────


class OccupancyEditorDialog(QDialog):
    """Per-well occupancy override for the board the project is currently on."""

    def __init__(
        self,
        parent: QWidget | None,
        store: Any,
        *,
        pcb_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._pcb_config = (
            pcb_config if pcb_config is not None else _default_pcb_config()
        )
        self._refusal = occupancy_edit_refusal()
        self._can_release = hasattr(store, "release_electrode")
        self._can_read_rows = hasattr(store, "electrode_occupancy_rows")

        self._boxes: dict[int, QCheckBox] = {}
        self._recorded: set[int] = set()
        self._rows: dict[int, dict[str, Any]] = {}
        self._board_id = 0
        #: A swap the operator has asked for but not yet committed. Nothing is
        #: written while this is set; OK turns it into the real swap.
        self._swap_pending = False

        #: The board actually written, or ``None`` after a refusal, a cancel or
        #: a no-op OK. The caller refreshes its map only when this is set.
        self.changed_board_id: int | None = None

        self.setWindowTitle("Edit Well Occupancy")
        self.setMinimumWidth(560)
        self._build_ui()
        self._reload()

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._lbl_intro = QLabel()
        self._lbl_intro.setWordWrap(True)
        layout.addWidget(self._lbl_intro)

        self._grid = QGridLayout()
        holder = QWidget()
        holder.setLayout(self._grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)

        self._lbl_degraded = QLabel()
        self._lbl_degraded.setWordWrap(True)
        self._lbl_degraded.setStyleSheet("color: gray;")
        layout.addWidget(self._lbl_degraded)

        reason_row = QHBoxLayout()
        reason_row.addWidget(QLabel("Reason (optional)"))
        self._reason = QLineEdit()
        self._reason.setPlaceholderText("e.g. cast by hand at the bench")
        reason_row.addWidget(self._reason, 1)
        layout.addLayout(reason_row)

        swap_row = QHBoxLayout()
        self._btn_swap = QPushButton(SWAP_STAGE_TEXT)
        self._btn_swap.setToolTip(
            "A fresh plate resets occupancy. Staged here and written on OK; the "
            "board counter advances only then, and cannot be moved back."
        )
        self._btn_swap.clicked.connect(self._on_toggle_swap)
        swap_row.addWidget(self._btn_swap)
        self._lbl_swap = QLabel()
        self._lbl_swap.setWordWrap(True)
        swap_row.addWidget(self._lbl_swap, 1)
        layout.addLayout(swap_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reload(self) -> None:
        """Re-read the board and rebuild the grid — also used when a swap is staged."""
        self._board_id = _safe(self._store.current_board_id, 0)
        target = effective_board(self._board_id, self._swap_pending)
        if self._swap_pending:
            # The fresh plate has nothing on it, and the retired board's rows
            # are stale detail about wells this grid no longer describes.
            self._recorded, self._rows = set(), {}
            board = f"board {self._board_id} → {target} (pending — applied on OK)"
            intro = (
                f"Board swap staged, {board}. The grid below is the fresh plate; "
                f"tick a well cast by hand and it is recorded against board "
                f"{target}. Cancel discards the swap along with the ticks."
            )
        else:
            self._recorded = set(
                _safe(lambda: self._store.occupied_electrodes(self._board_id), set())
            )
            self._rows = self._read_rows()
            board = f"board {self._board_id}"
            intro = (
                f"Board {self._board_id}: {len(self._recorded)} well(s) recorded as "
                "cast. Tick to record a manual cast, untick to free a well. Nothing "
                "is written until OK, and every change is logged as an alert."
            )
        self.setWindowTitle(f"Edit Well Occupancy — {board}")
        self._lbl_intro.setText(intro)
        self._btn_swap.setText(
            SWAP_UNDO_TEXT if self._swap_pending else SWAP_STAGE_TEXT)
        self._lbl_degraded.setText(self._degradation_note())
        self._rebuild_grid()

    def _read_rows(self) -> dict[int, dict[str, Any]]:
        if not self._can_read_rows:
            return {}
        rows = _safe(
            lambda: self._store.electrode_occupancy_rows(self._board_id), []
        )
        return {
            int(row["electrode"]): dict(row)
            for row in rows
            if row.get("electrode") is not None
        }

    def _rebuild_grid(self) -> None:
        while self._grid.count():
            widget = self._grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self._boxes.clear()
        for idx, electrode in enumerate(grid_range(self._pcb_config, self._recorded)):
            occupied = electrode in self._recorded
            box = QCheckBox(f"E{electrode}")
            box.setChecked(occupied)
            box.setToolTip(self._tooltip(electrode, occupied))
            # Nothing can free a well the store has no per-well delete for, so
            # the box must not offer it.
            box.setEnabled(self._can_release or not occupied)
            self._grid.addWidget(box, idx // _COLUMNS, idx % _COLUMNS)
            self._boxes[electrode] = box

    def _tooltip(self, electrode: int, occupied: bool) -> str:
        if occupied and not self._can_release:
            return (
                "This project's data store cannot release a single well yet, so "
                "recorded wells are read-only here."
            )
        if not occupied:
            return "Free — tick to record a manual cast."
        row = self._rows.get(electrode)
        if row is None:
            return "Recorded as cast; this store exposes no row detail for it."
        return (
            f"cast at {row.get('cast_at') or '—'}\n"
            f"run {row.get('run_id') or '—'}\n"
            f"sample {row.get('sample_uuid') or '—'}"
        )

    def _degradation_note(self) -> str:
        missing = []
        if not self._can_release:
            missing.append(
                "recorded wells cannot be freed — this store has no per-well "
                "release"
            )
        if not self._can_read_rows:
            missing.append(
                "well identities cannot be read, so freeing warns about every "
                "well rather than only those carrying a sample"
            )
        return ("Limited: " + "; ".join(missing) + ".") if missing else ""

    # ── Operator actions ─────────────────────────────────────────────────

    def exec(self) -> int:
        """Refuse before showing anything, so the caller needs no second check."""
        if self._refusal is None:
            return super().exec()
        box = QMessageBox(self.parentWidget())
        box.setWindowTitle("Edit Well Occupancy")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText("Well occupancy cannot be edited while a run holds the rig.")
        box.setInformativeText(self._refusal)
        box.exec()
        self.reject()
        return int(QDialog.DialogCode.Rejected)

    def _on_toggle_swap(self) -> None:
        """Stage a board swap, or withdraw one — neither writes anything.

        The pointer only moves on OK, so a swap reached by mis-click costs
        exactly as little as a mis-ticked well.
        """
        self._swap_pending = not self._swap_pending
        # ``_reload`` rebuilds the boxes from what is recorded, which is what
        # discards the staged edits; they meant a different board.
        self._reload()
        moved = "staged" if self._swap_pending else "withdrawn"
        self._lbl_swap.setText(
            f"Swap {moved}. Staged well edits discarded — a swap changes which "
            "board they would mean."
        )

    def _on_accept(self) -> None:
        staged = {e for e, box in self._boxes.items() if box.isChecked()}
        edits = diff_occupancy(staged, self._recorded, self._rows)
        if not edits and not self._swap_pending:
            self.accept()
            return
        # Before the swap, not after: a decline here must not leave a board id
        # burnt. Under a pending swap nothing is recorded, so nothing is freed
        # and this is a no-op.
        if not self._confirm_identity_loss(edits):
            return
        board_id = self._board_id
        if self._swap_pending:
            # That prompt still owns the confirmation and the swap alert. A
            # decline aborts the whole apply rather than landing these casts on
            # the plate the operator has just said is gone.
            new_id = prompt_log_board_swap(self, self._store)
            if new_id is None:
                return
            board_id = new_id
            self._board_id, self._swap_pending = new_id, False
        reason = self._reason.text().strip()
        for edit in edits:
            self._apply(board_id, edit, reason)
        self.changed_board_id = board_id
        self.accept()

    def _confirm_identity_loss(self, edits: Sequence[OccupancyEdit]) -> bool:
        """Last stop before the only irreversible half of this feature."""
        at_risk = identity_at_risk(edits, rows_readable=self._can_read_rows)
        if not at_risk:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Free Recorded Wells")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Free wells that may carry a sample identity?")
        box.setInformativeText(self._identity_warning(at_risk))
        confirm = box.addButton("Free them", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(box.button(QMessageBox.StandardButton.Cancel))
        box.exec()
        return box.clickedButton() is confirm

    def _identity_warning(self, at_risk: Sequence[OccupancyEdit]) -> str:
        wells = ", ".join(f"E{edit.electrode}" for edit in at_risk)
        if not self._can_read_rows:
            return (
                f"Freeing: {wells}. This store cannot report which of them carry "
                "a sample identity, so all are listed — the identities could not "
                "be read, not checked and found absent.\n\nFreeing a well with an "
                "identity orphans a real film's record and silences the "
                "warn-before-recast prompt."
            )
        detail = "\n".join(
            f"  E{edit.electrode} — sample "
            f"{(edit.previous_row or {}).get('sample_uuid')}"
            for edit in at_risk
        )
        return (
            "These wells carry a sample identity. Freeing them orphans a real "
            "film's record and silences the warn-before-recast prompt:\n\n"
            f"{detail}"
        )

    def _apply(self, board_id: int, edit: OccupancyEdit, reason: str) -> None:
        if edit.action == OCCUPY:
            # No run_id and no sample_uuid: a manual cast has no minted identity,
            # and a fabricated one would join the well to a run that never
            # touched it.
            self._store.record_electrode_cast(board_id, edit.electrode)
        else:
            removed = self._store.release_electrode(board_id, edit.electrode)
            if edit.previous_row is None and removed:
                edit = replace(edit, previous_row=dict(removed))
        raise_alert(override_alert(board_id, edit, reason), data_store=self._store)
