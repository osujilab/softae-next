"""Manual-pump single-use occupancy: infer target well, gate on head-down, guard."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from softae.core.data_store import DataStore
from softae.core.deposition_steps import deposition_positions
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_manual import ManualControlTab
from softae.gui.widgets import occupancy_guard
from softae.gui.widgets.occupancy_guard import BoardReplacedDecision


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def env(qapp, tmp_path: Path):
    manager = create_mock_manager(config={})
    ds = DataStore(tmp_path / "proj")
    tab = ManualControlTab(manager, data_store=ds)
    yield tab, manager, ds
    pv = getattr(tab, "_pv_worker", None)
    if pv is not None and pv.isRunning():
        pv.stop_worker()
    pos_map = getattr(tab, "_pos_map", None)
    if pos_map is not None:
        w = getattr(pos_map, "_pos_worker", None)
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.wait(2000)
    tab.close()
    ds.close()


def _origin():
    return deposition_positions().origin


def _place_over_e1(manager, *, head_down: bool):
    """Head belief + stage position so the pump targets electrode 1."""
    manager.get("syringe").set_head_state(not head_down)  # is_up = not head_down
    ox, oy = _origin()
    manager.get("stage").live_position = lambda: (ox, oy, 0.0)


def _wait_until(cond, timeout=3.0):
    t0 = time.monotonic()
    while not cond() and (time.monotonic() - t0) < timeout:
        QApplication.processEvents()
        time.sleep(0.02)
    return cond()


# ── _pending_cast_target (which well would this pump occupy?) ───────────────


def test_head_up_is_not_a_cast(env):
    tab, manager, _ = env
    _place_over_e1(manager, head_down=False)
    assert tab._pending_cast_target() is None


def test_head_down_over_well_targets_it(env):
    tab, manager, _ = env
    _place_over_e1(manager, head_down=True)
    assert tab._pending_cast_target() == (0, 1)  # board 0, electrode 1


def test_head_down_away_from_wells_is_none(env):
    tab, manager, _ = env
    manager.get("syringe").set_head_state(False)
    ox, oy = _origin()
    manager.get("stage").live_position = lambda: (ox + 500.0, oy + 500.0, 0.0)
    assert tab._pending_cast_target() is None


# ── _on_infuse recording + guard ───────────────────────────────────────────


def test_infuse_records_occupancy_when_casting(env):
    tab, manager, ds = env
    _place_over_e1(manager, head_down=True)
    tab._on_infuse(0)
    assert _wait_until(lambda: ds.occupied_electrodes(0) == {1})


def test_infuse_head_up_records_nothing(env):
    tab, manager, ds = env
    _place_over_e1(manager, head_down=False)
    tab._on_infuse(0)
    # Let the pump worker finish, then confirm no well was marked.
    _wait_until(lambda: False, timeout=0.5)
    assert ds.occupied_electrodes(0) == set()


def test_infuse_into_occupied_cancel_does_not_pump(env, monkeypatch):
    tab, manager, ds = env
    ds.record_electrode_cast(0, 1)  # E1 already used
    _place_over_e1(manager, head_down=True)
    spy = MagicMock()
    manager.get("syringe").single_pump = spy
    monkeypatch.setattr(
        occupancy_guard, "prompt_board_replaced", lambda *a, **k: BoardReplacedDecision.CANCEL
    )
    tab._on_infuse(0)
    _wait_until(lambda: False, timeout=0.3)
    spy.assert_not_called()
    assert tab._pump_widgets[0]["btn"].isEnabled()  # never disabled


def test_infuse_into_occupied_fresh_records_on_new_board(env, monkeypatch):
    tab, manager, ds = env
    ds.record_electrode_cast(0, 1)
    _place_over_e1(manager, head_down=True)
    monkeypatch.setattr(
        occupancy_guard, "prompt_board_replaced", lambda *a, **k: BoardReplacedDecision.FRESH
    )
    tab._on_infuse(0)
    assert _wait_until(lambda: ds.occupied_electrodes(1) == {1})  # fresh board id
    assert ds.occupied_electrodes(0) == {1}  # retired board untouched
    assert ds.current_board_id() == 1        # pointer advanced durably


def test_infuse_fresh_board_pointer_persists_even_if_pump_fails(env, monkeypatch):
    """The swap is recorded up-front, so a failed dispense cannot lose it."""
    tab, manager, ds = env
    ds.record_electrode_cast(0, 1)
    _place_over_e1(manager, head_down=True)

    def boom(*a, **k):
        raise RuntimeError("pump offline")

    manager.get("syringe").single_pump = boom
    monkeypatch.setattr(
        occupancy_guard, "prompt_board_replaced", lambda *a, **k: BoardReplacedDecision.FRESH
    )
    # Swallow the error dialog the failed worker raises.
    monkeypatch.setattr(
        "softae.gui.tabs.tab_manual.QMessageBox.warning", lambda *a, **k: None
    )
    tab._on_infuse(0)
    assert _wait_until(lambda: ds.current_board_id() == 1)
    assert ds.occupied_electrodes(1) == set()  # nothing cast — pump failed


def test_infuse_into_occupied_cast_anyway_stays_on_same_board(env, monkeypatch):
    tab, manager, ds = env
    ds.record_electrode_cast(0, 1)
    _place_over_e1(manager, head_down=True)
    spy = MagicMock()
    manager.get("syringe").single_pump = spy
    monkeypatch.setattr(
        occupancy_guard, "prompt_board_replaced",
        lambda *a, **k: BoardReplacedDecision.CAST_ANYWAY,
    )
    tab._on_infuse(0)
    assert _wait_until(lambda: spy.called)
    assert ds.occupied_electrodes(0) == {1}  # same board, re-affirmed
    assert ds.occupied_electrodes(1) == set()  # no fresh board created
