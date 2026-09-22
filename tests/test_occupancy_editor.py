"""Manual per-well occupancy override — the diff, the alert, and the refusal."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QMessageBox

from softae.core.alerts import INFO
from softae.core.data_store import DataStore
from softae.gui.widgets import occupancy_editor as oe
from softae.gui.widgets.occupancy_editor import (
    OccupancyEdit,
    OccupancyEditorDialog,
    diff_occupancy,
    effective_board,
    grid_range,
    identity_at_risk,
    occupancy_edit_refusal,
    override_alert,
)

#: Pinned so the grid's size does not depend on this machine's config file.
_PCB = {"channels": 16}


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


@pytest.fixture
def free_rig(monkeypatch):
    """No foreign lock, whatever this machine's real rig lock happens to say."""
    monkeypatch.setattr(oe, "foreign_run_lock", lambda: None)


def _modal(monkeypatch, *, accept: bool = False) -> list[str]:
    """Drive every modal without a display; returns the text each one showed."""
    seen: list[str] = []

    def fake_exec(self):
        seen.append(f"{self.text()}\n{self.informativeText()}")
        picked = None
        if accept:
            for btn in self.buttons():
                if self.buttonRole(btn) == QMessageBox.ButtonRole.AcceptRole:
                    picked = btn
                    break
        self._picked = picked or self.button(QMessageBox.StandardButton.Cancel)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec, raising=False)
    monkeypatch.setattr(
        QMessageBox, "clickedButton",
        lambda self: getattr(self, "_picked", None), raising=False,
    )
    return seen


def _trap_close(monkeypatch, dlg) -> list[str]:
    """Record — rather than perform — the dialog closing itself.

    "The dialog stays open" is otherwise indistinguishable from "it accepted",
    because an unshown dialog reports the same result either way.
    """
    closed: list[str] = []
    monkeypatch.setattr(dlg, "accept", lambda: closed.append("accept"))
    monkeypatch.setattr(dlg, "reject", lambda: closed.append("reject"))
    return closed


def _trap_dialog_loop(monkeypatch) -> list[bool]:
    """Record — rather than enter — the dialog's own modal loop.

    Without this a refusal that wrongly permits blocks the run forever instead
    of failing, and a check that can only hang is not a check.
    """
    entered: list[bool] = []
    monkeypatch.setattr(
        QDialog, "exec", lambda self: (entered.append(True), 1)[1], raising=False)
    return entered


class _StoreWithRelease:
    """A store that has the two per-well methods the editor feature-detects.

    Stands in until the real store grows them, so the free path is exercised
    now instead of only skipped.
    """

    def __init__(self, inner: DataStore) -> None:
        self._inner = inner
        self._identities: dict[tuple[int, int], dict] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def record_electrode_cast(self, board_id, electrode, *, run_id=None,
                              iteration=None, sample_uuid=None) -> None:
        self._inner.record_electrode_cast(
            board_id, electrode, run_id=run_id, iteration=iteration,
            sample_uuid=sample_uuid)
        self._identities[(int(board_id), int(electrode))] = {
            "board_id": int(board_id), "electrode": int(electrode),
            "run_id": run_id, "iteration": iteration,
            "cast_at": "2026-09-22T00:00:00+00:00", "sample_uuid": sample_uuid,
        }

    def electrode_occupancy_rows(self, board_id) -> list[dict]:
        return [dict(row) for (board, _e), row in self._identities.items()
                if board == int(board_id)]

    def release_electrode(self, board_id, electrode) -> dict | None:
        removed = self._identities.pop((int(board_id), int(electrode)), None)
        if removed is None:
            return None
        survivors = self.electrode_occupancy_rows(board_id)
        self._inner.clear_board(int(board_id))
        for row in survivors:
            self._inner.record_electrode_cast(
                board_id, row["electrode"], run_id=row["run_id"],
                iteration=row["iteration"], sample_uuid=row["sample_uuid"])
        return removed


# ── The Qt-free core ─────────────────────────────────────────────────────────


class TestDiff:
    def test_diff_occupancy_unchanged_wells_produce_no_edits(self):
        """An untouched grid must not write an alert claiming a human edited it."""
        assert diff_occupancy({1, 2}, {1, 2}) == []
        assert diff_occupancy(set(), set()) == []

    def test_diff_occupancy_newly_checked_well_is_an_occupy_edit(self):
        assert diff_occupancy({1, 3}, {1}) == [OccupancyEdit(3, "occupy", None)]

    def test_diff_occupancy_unchecked_well_is_a_free_edit_carrying_its_row(self):
        """The displaced row travels with the edit so the alert can record it."""
        row = {"electrode": 4, "sample_uuid": "u-4"}
        assert diff_occupancy(set(), {4}, {4: row}) == [
            OccupancyEdit(4, "free", row)]

    def test_diff_occupancy_mixed_edits_come_back_in_well_order(self):
        edits = diff_occupancy({2, 9}, {5, 9})
        assert [(e.electrode, e.action) for e in edits] == [
            (2, "occupy"), (5, "free")]


class TestGridRange:
    def test_grid_range_covers_the_board_capacity(self):
        assert grid_range({"channels": 32}, set()) == range(1, 33)

    def test_grid_range_extends_past_capacity_to_cover_a_recorded_well(self):
        """A well recorded under a larger layout must stay reachable, not stranded."""
        assert grid_range({"channels": 16}, {20}) == range(1, 21)


class TestEffectiveBoard:
    def test_effective_board_without_a_pending_swap_is_the_current_board(self):
        assert effective_board(4, False) == 4

    def test_effective_board_with_a_pending_swap_is_the_next_board(self):
        """A tick on the staged fresh grid belongs to the board the swap creates."""
        assert effective_board(4, True) == 5


class TestOverrideAlert:
    def test_override_alert_free_records_the_previous_row_and_reason(self):
        """The alert is the only record that a human, not a run, changed the well."""
        row = {"electrode": 9, "sample_uuid": "u-9"}
        alert = override_alert(3, OccupancyEdit(9, "free", row), "mis-record")

        assert (alert.kind, alert.severity) == ("occupancy_override", INFO)
        assert alert.details["board_id"] == 3
        assert alert.details["electrode"] == 9
        assert alert.details["action"] == "free"
        assert alert.details["previous_row"] == row
        assert alert.details["reason"] == "mis-record"
        assert "E9" in alert.message and "mis-record" in alert.message

    def test_override_alert_occupy_records_no_previous_row(self):
        """Nothing was displaced; an empty dict would read back as a blank row."""
        alert = override_alert(1, OccupancyEdit(2, "occupy"), "   ")
        assert alert.details["previous_row"] is None
        assert alert.details["reason"] is None


class TestIdentityAtRisk:
    def test_identity_at_risk_lists_only_wells_with_a_sample_uuid(self):
        edits = [
            OccupancyEdit(1, "free", {"sample_uuid": "u-1"}),
            OccupancyEdit(2, "free", {"sample_uuid": None}),
            OccupancyEdit(3, "occupy"),
        ]
        at_risk = identity_at_risk(edits, rows_readable=True)
        assert [e.electrode for e in at_risk] == [1]

    def test_identity_at_risk_without_a_rows_reader_lists_every_freed_well(self):
        """Unknown is not spelled like clean: an unreadable row still warns."""
        edits = [OccupancyEdit(1, "free"), OccupancyEdit(2, "free"),
                 OccupancyEdit(3, "occupy")]
        at_risk = identity_at_risk(edits, rows_readable=False)
        assert [e.electrode for e in at_risk] == [1, 2]


class TestRefusal:
    def test_occupancy_edit_refusal_names_the_campaign_holding_the_rig(self):
        lock = SimpleNamespace(what="campaign:phase-map:run-7", pid=99,
                               started_at="14:02", log_path="/runs/7")
        message = occupancy_edit_refusal(lambda: lock)
        assert message is not None
        assert "phase-map" in message and "run-7" in message

    def test_occupancy_edit_refusal_names_a_non_campaign_holder(self):
        lock = SimpleNamespace(what="workflow 'anneal'", pid=99,
                               started_at="14:02", log_path="")
        message = occupancy_edit_refusal(lambda: lock)
        assert message is not None and "99" in message

    def test_occupancy_edit_refusal_treats_an_unreadable_lock_as_held(self):
        """Unknown must refuse: an edit under a live run is invisible to that run."""
        def unreadable():
            raise OSError("the lock share is down")

        assert occupancy_edit_refusal(unreadable) is not None

    def test_occupancy_edit_refusal_is_none_when_the_rig_is_free(self):
        assert occupancy_edit_refusal(lambda: None) is None


# ── The dialog ───────────────────────────────────────────────────────────────


class TestDialogRefusal:
    def test_dialog_refuses_to_open_while_a_campaign_holds_the_rig(
        self, qapp, store, monkeypatch
    ):
        """A running campaign froze its occupancy at launch; an edit under it is unseen."""
        lock = SimpleNamespace(what="campaign:phase-map:run-7", pid=4321,
                               started_at="14:02", log_path="/runs/7")
        monkeypatch.setattr(oe, "foreign_run_lock", lambda: lock)
        shown = _modal(monkeypatch)
        entered = _trap_dialog_loop(monkeypatch)

        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        result = dlg.exec()

        assert entered == []
        assert result == int(QDialog.DialogCode.Rejected)
        assert dlg.changed_board_id is None
        assert any("phase-map" in text and "run-7" in text for text in shown)

    def test_dialog_refuses_to_open_when_the_lock_cannot_be_read(
        self, qapp, store, monkeypatch
    ):
        """A reader that raises must not resolve to permission."""
        def unreadable():
            raise OSError("the lock share is down")

        monkeypatch.setattr(oe, "foreign_run_lock", unreadable)
        shown = _modal(monkeypatch)
        entered = _trap_dialog_loop(monkeypatch)

        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        assert dlg.exec() == int(QDialog.DialogCode.Rejected)
        assert entered == []
        assert dlg.changed_board_id is None
        assert any("could not be read" in text for text in shown)

    def test_dialog_opens_when_the_rig_is_free(
        self, qapp, store, free_rig, monkeypatch
    ):
        """The ordinary case stays reachable — a refusal that always fires is no check."""
        entered = _trap_dialog_loop(monkeypatch)

        assert OccupancyEditorDialog(None, store, pcb_config=_PCB).exec() == 1
        assert entered == [True]


class TestDialogApply:
    def test_dialog_ok_writes_one_alert_per_changed_well(
        self, qapp, store, free_rig
    ):
        """Every override is its own audit event — a batched one names no well."""
        store.record_electrode_cast(1, 3)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        dlg._boxes[5].setChecked(True)
        dlg._boxes[6].setChecked(True)
        dlg._reason.setText("cast by hand")

        dlg._on_accept()

        overrides = [a for a in store.query_alerts()
                     if a["kind"] == "occupancy_override"]
        assert len(overrides) == 2
        assert {a["details"]["electrode"] for a in overrides} == {5, 6}
        assert all(a["details"]["reason"] == "cast by hand" for a in overrides)
        assert store.occupied_electrodes(1) == {3, 5, 6}
        assert dlg.changed_board_id == 1

    def test_dialog_ok_with_no_change_writes_nothing(self, qapp, store, free_rig):
        """A no-op OK must not claim an override happened."""
        store.record_electrode_cast(1, 3)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        dlg._on_accept()

        assert store.query_alerts() == []
        assert dlg.changed_board_id is None

    def test_dialog_cancel_writes_nothing(self, qapp, store, free_rig):
        """Staged ticks are not writes; a mis-click must cost nothing."""
        store.record_electrode_cast(1, 3)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        dlg._boxes[5].setChecked(True)

        dlg.reject()

        assert store.occupied_electrodes(1) == {3}
        assert store.query_alerts() == []
        assert dlg.changed_board_id is None

    @pytest.mark.skipif(
        not hasattr(DataStore, "electrode_occupancy_rows"),
        reason="the store cannot report a well's run_id yet",
    )
    def test_dialog_occupy_records_the_cast_with_no_run_id(
        self, qapp, store, free_rig
    ):
        """A manual cast has no minted identity; a fabricated one would be a lie."""
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        dlg._boxes[5].setChecked(True)
        dlg._on_accept()

        row = next(r for r in store.electrode_occupancy_rows(dlg._board_id)
                   if int(r["electrode"]) == 5)
        assert row["run_id"] is None
        assert row["sample_uuid"] is None


class TestDialogFree:
    """The free path, driven through a store that has the per-well methods."""

    @pytest.fixture
    def releasing(self, store):
        store.set_active_board(1)
        return _StoreWithRelease(store)

    def test_dialog_free_deletes_only_the_named_well(
        self, qapp, releasing, free_rig, monkeypatch
    ):
        releasing.record_electrode_cast(1, 3)
        releasing.record_electrode_cast(1, 4)
        _modal(monkeypatch, accept=True)
        dlg = OccupancyEditorDialog(None, releasing, pcb_config=_PCB)
        dlg._boxes[3].setChecked(False)

        dlg._on_accept()

        assert releasing.occupied_electrodes(1) == {4}
        overrides = [a for a in releasing.query_alerts()
                     if a["kind"] == "occupancy_override"]
        assert len(overrides) == 1
        assert overrides[0]["details"]["previous_row"]["electrode"] == 3

    def test_dialog_free_warns_before_detaching_a_sample_identity(
        self, qapp, releasing, free_rig, monkeypatch
    ):
        """Freeing a well with an identity orphans a real film's record."""
        releasing.record_electrode_cast(1, 3, sample_uuid="u-3")
        shown = _modal(monkeypatch, accept=True)
        dlg = OccupancyEditorDialog(None, releasing, pcb_config=_PCB)
        dlg._boxes[3].setChecked(False)

        dlg._on_accept()

        assert any("u-3" in text for text in shown)
        assert releasing.occupied_electrodes(1) == set()

    def test_dialog_free_declined_at_the_warning_writes_nothing(
        self, qapp, releasing, free_rig, monkeypatch
    ):
        releasing.record_electrode_cast(1, 3, sample_uuid="u-3")
        _modal(monkeypatch, accept=False)
        dlg = OccupancyEditorDialog(None, releasing, pcb_config=_PCB)
        dlg._boxes[3].setChecked(False)

        dlg._on_accept()

        assert releasing.occupied_electrodes(1) == {3}
        assert releasing.query_alerts() == []
        assert dlg.changed_board_id is None


class TestDialogDegradation:
    def test_dialog_without_release_electrode_disables_the_free_control(
        self, qapp, store, free_rig, monkeypatch
    ):
        """A missing capability must degrade visibly — a box that cannot free is a lie."""
        monkeypatch.delattr(DataStore, "release_electrode", raising=False)
        store.record_electrode_cast(1, 3)

        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        assert dlg._boxes[3].isEnabled() is False
        assert "release" in dlg._boxes[3].toolTip().lower()
        assert dlg._boxes[4].isEnabled() is True

    def test_dialog_without_a_rows_reader_says_identities_are_unread(
        self, qapp, store, free_rig, monkeypatch
    ):
        monkeypatch.delattr(DataStore, "electrode_occupancy_rows", raising=False)
        store.record_electrode_cast(1, 3)

        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        assert "identities cannot be read" in dlg._lbl_degraded.text()


def _advancing_prompt(monkeypatch) -> list[object]:
    """Stand in for the swap prompt, confirmed; returns the stores it was given."""
    seen: list[object] = []

    def fake_prompt(parent, data_store):
        seen.append(data_store)
        return data_store.advance_board()

    monkeypatch.setattr(oe, "prompt_log_board_swap", fake_prompt)
    return seen


def _never_prompt(monkeypatch) -> None:
    """Staging must not reach the swap prompt — and must not hang if it does.

    The real prompt opens a modal with nothing to dismiss it, so a regression
    here would wedge the run instead of failing it.
    """
    def refuse(parent, data_store):
        raise AssertionError("the swap prompt was reached while only staging")

    monkeypatch.setattr(oe, "prompt_log_board_swap", refuse)


class TestDialogBoardSwap:
    """The swap is staged like every other edit and written only on OK."""

    def test_dialog_swap_button_stages_without_moving_the_board(
        self, qapp, store, free_rig, monkeypatch
    ):
        """Pressing the button must not spend a board id the operator can cancel."""
        store.record_electrode_cast(1, 3)
        seen = _advancing_prompt(monkeypatch)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        dlg._btn_swap.click()

        assert seen == []
        assert store.current_board_id() == 1
        assert dlg.changed_board_id is None
        assert dlg._btn_swap.text() == oe.SWAP_UNDO_TEXT
        assert not any(box.isChecked() for box in dlg._boxes.values())

    def test_dialog_cancel_after_a_staged_swap_leaves_the_board_untouched(
        self, qapp, store, free_rig, monkeypatch
    ):
        """Cancel must mean nothing happened; the board pointer cannot be walked back."""
        store.record_electrode_cast(1, 3)
        _modal(monkeypatch, accept=True)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        dlg._btn_swap.click()
        dlg._boxes[5].setChecked(True)
        dlg.reject()

        assert store.current_board_id() == 1
        assert store.occupied_electrodes(1) == {3}
        assert store.query_alerts() == []
        assert dlg.changed_board_id is None

    def test_dialog_ok_after_a_staged_swap_delegates_then_casts_on_the_new_board(
        self, qapp, store, free_rig, monkeypatch
    ):
        """One confirmation of the monotonic pointer, and the ticks follow the plate."""
        store.record_electrode_cast(1, 3)
        seen = _advancing_prompt(monkeypatch)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)

        dlg._btn_swap.click()
        dlg._boxes[7].setChecked(True)
        dlg._on_accept()

        assert seen == [store]
        assert store.occupied_electrodes(2) == {7}
        assert store.occupied_electrodes(1) == {3}
        overrides = [a for a in store.query_alerts()
                     if a["kind"] == "occupancy_override"]
        assert [(a["details"]["board_id"], a["details"]["electrode"])
                for a in overrides] == [(2, 7)]
        assert dlg.changed_board_id == 2

    def test_dialog_ok_with_the_swap_declined_writes_nothing_and_stays_open(
        self, qapp, store, free_rig, monkeypatch
    ):
        """A declined swap must not land the fresh plate's casts on the old board."""
        store.record_electrode_cast(1, 3)
        monkeypatch.setattr(oe, "prompt_log_board_swap", lambda parent, s: None)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        dlg._btn_swap.click()
        dlg._boxes[7].setChecked(True)
        closed = _trap_close(monkeypatch, dlg)

        dlg._on_accept()

        assert closed == []
        assert dlg._swap_pending is True
        assert store.current_board_id() == 1
        assert store.occupied_electrodes(1) == {3}
        assert store.query_alerts() == []
        assert dlg.changed_board_id is None

    def test_dialog_swap_toggled_twice_restores_the_boards_own_grid(
        self, qapp, store, free_rig, monkeypatch
    ):
        """Staging shows a fresh plate; withdrawing it must show the real one again."""
        store.record_electrode_cast(1, 3)
        _never_prompt(monkeypatch)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        dlg._boxes[5].setChecked(True)

        dlg._btn_swap.click()
        staged_fresh = [e for e, b in dlg._boxes.items() if b.isChecked()]
        dlg._btn_swap.click()

        assert staged_fresh == []
        assert dlg._swap_pending is False
        assert dlg._btn_swap.text() == oe.SWAP_STAGE_TEXT
        assert dlg._boxes[3].isChecked() is True
        assert dlg._boxes[5].isChecked() is False

    def test_dialog_pending_swap_keeps_every_well_tickable(
        self, qapp, store, free_rig, monkeypatch
    ):
        """A fresh plate is hand-cast well by well, so the staged grid must stay live."""
        monkeypatch.delattr(DataStore, "release_electrode", raising=False)
        _never_prompt(monkeypatch)
        store.record_electrode_cast(1, 3)
        dlg = OccupancyEditorDialog(None, store, pcb_config=_PCB)
        assert dlg._boxes[3].isEnabled() is False

        dlg._btn_swap.click()

        assert all(box.isEnabled() for box in dlg._boxes.values())
