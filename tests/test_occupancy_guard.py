"""Tests for the warn-before-recast occupancy guard."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from softae.core.data_store import DataStore
from softae.gui.widgets.occupancy_guard import (
    BoardReplacedDecision,
    occupied_conflicts,
    prompt_board_replaced,
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


class TestConflicts:
    def test_none_when_store_missing(self):
        assert occupied_conflicts(None, 0, [1, 2, 3]) == set()

    def test_intersection_of_selected_and_occupied(self, store):
        store.record_electrode_cast(0, 2)
        store.record_electrode_cast(0, 5)
        assert occupied_conflicts(store, 0, [1, 2, 3, 5]) == {2, 5}
        assert occupied_conflicts(store, 0, [1, 3, 4]) == set()
        assert occupied_conflicts(store, 1, [2, 5]) == set()  # different board


class TestPrompt:
    def _patch(self, monkeypatch, text):
        monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)

        def _clicked(self):
            if text is None:
                return None
            for b in self.buttons():
                if b.text().replace("&", "") == text:
                    return b
            return None

        monkeypatch.setattr(QMessageBox, "clickedButton", _clicked)

    def test_board_replaced(self, qapp, monkeypatch):
        self._patch(monkeypatch, "Board replaced")
        assert prompt_board_replaced(None, 0, {3}) is BoardReplacedDecision.FRESH

    def test_cast_anyway(self, qapp, monkeypatch):
        self._patch(monkeypatch, "Same board, cast anyway")
        assert prompt_board_replaced(None, 0, {3}) is BoardReplacedDecision.CAST_ANYWAY

    def test_cancel(self, qapp, monkeypatch):
        self._patch(monkeypatch, "Cancel")
        assert prompt_board_replaced(None, 0, {3}) is BoardReplacedDecision.CANCEL

    def test_dismissed_is_cancel(self, qapp, monkeypatch):
        self._patch(monkeypatch, None)
        assert prompt_board_replaced(None, 0, {3}) is BoardReplacedDecision.CANCEL
