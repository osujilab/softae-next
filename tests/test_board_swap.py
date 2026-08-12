"""Operator-initiated board swap — log a fresh plate and reset the positions."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from softae.core.data_store import DataStore
from softae.gui.widgets import occupancy_guard as og
from softae.gui.widgets.occupancy_guard import (
    occupied_conflicts,
    prompt_log_board_swap,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def store(tmp_path: Path):
    ds = DataStore(tmp_path / "proj")
    yield ds
    ds.close()


def _answer(monkeypatch, *, confirm: bool):
    """Drive the modal by clicking the accept button (or cancelling)."""
    def fake_exec(self):
        if confirm:
            for btn in self.buttons():
                if self.buttonRole(btn) == QMessageBox.ButtonRole.AcceptRole:
                    self.setProperty("_clicked", btn)
                    self._picked = btn
                    return 0
        self._picked = self.button(QMessageBox.StandardButton.Cancel)
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec, raising=False)
    monkeypatch.setattr(
        QMessageBox, "clickedButton", lambda self: getattr(self, "_picked", None),
        raising=False,
    )


# ── The store operation ──────────────────────────────────────────────────────

class TestAdvanceBoard:
    def test_advance_moves_to_a_fresh_empty_board(self, store):
        store.record_electrode_cast(1, 5)
        store.record_electrode_cast(1, 6)
        assert store.occupied_electrodes(1) == {5, 6}

        new_id = store.advance_board()

        assert new_id == 2
        assert store.current_board_id() == 2
        assert store.occupied_electrodes(2) == set()

    def test_history_is_preserved_not_erased(self, store):
        """The reset is a new namespace — past runs must stay interpretable."""
        store.record_electrode_cast(1, 5)
        store.advance_board()
        assert store.occupied_electrodes(1) == {5}

    def test_advance_from_a_pristine_project(self, store):
        assert store.current_board_id() == 0
        assert store.advance_board() == 1

    def test_repeated_swaps_keep_climbing(self, store):
        assert [store.advance_board() for _ in range(3)] == [1, 2, 3]

    def test_conflicts_clear_after_a_swap(self, store):
        """The point of the feature: previously-cast wells become available."""
        store.record_electrode_cast(1, 5)
        assert occupied_conflicts(store, store.current_board_id(), [5]) == {5}

        new_id = store.advance_board()
        assert occupied_conflicts(store, new_id, [5]) == set()


# ── The operator prompt ──────────────────────────────────────────────────────

class TestPrompt:
    def test_confirming_advances_the_board(self, qapp, store, monkeypatch):
        store.record_electrode_cast(1, 5)
        _answer(monkeypatch, confirm=True)

        assert prompt_log_board_swap(None, store) == 2
        assert store.current_board_id() == 2

    def test_cancelling_changes_nothing(self, qapp, store, monkeypatch):
        store.record_electrode_cast(1, 5)
        _answer(monkeypatch, confirm=False)

        assert prompt_log_board_swap(None, store) is None
        assert store.current_board_id() == 1
        assert store.occupied_electrodes(1) == {5}

    def test_no_store_is_a_no_op(self, qapp):
        assert prompt_log_board_swap(None, None) is None

    def test_swap_is_recorded_as_a_durable_alert(self, qapp, store, monkeypatch):
        """Provenance: which physical plate a sample landed on must outlive the session."""
        store.record_electrode_cast(1, 5)
        _answer(monkeypatch, confirm=True)

        prompt_log_board_swap(None, store)

        swaps = [a for a in store.query_alerts() if a["kind"] == "board_swap"]
        assert len(swaps) == 1
        assert "1" in swaps[0]["message"] and "2" in swaps[0]["message"]
