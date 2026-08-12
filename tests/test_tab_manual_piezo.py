from __future__ import annotations

from unittest.mock import patch

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_manual import ManualControlTab


@pytest.fixture
def tab(qapp):
    manager = create_mock_manager(config={"piezo": {"enabled": True, "supports_cfg": True}})
    with patch("softae.gui.tabs.tab_manual.piezo_config", return_value={
        "enabled": True,
        "frequency_hz": 500,
        "sweep_on_s": 2.0,
        "sweep_rest_s": 3.0,
    }):
        widget = ManualControlTab(manager)
    yield widget
    pv_worker = getattr(widget, "_pv_worker", None)
    if pv_worker is not None and pv_worker.isRunning():
        pv_worker.stop_worker()
    widget.close()


def _wait_enabled(qtbot, button, timeout_ms: int = 2000):
    qtbot.waitUntil(lambda: button.isEnabled(), timeout=timeout_ms)


def test_piezo_apply_profile_non_blocking(tab, qtbot):
    tab._spin_piezo_freq.setValue(800)
    tab._spin_piezo_on_s.setValue(1.1)
    tab._spin_piezo_rest_s.setValue(2.2)

    tab._on_piezo_apply_settings()
    assert not tab._btn_piezo_apply.isEnabled()

    _wait_enabled(qtbot, tab._btn_piezo_apply)
    assert tab._btn_piezo_apply.isEnabled()
    assert "profile applied" in tab._lbl_piezo_status.text().lower()


def test_piezo_a_on_and_off_dispatch(tab, qtbot):
    tab._on_piezo_a_on()
    _wait_enabled(qtbot, tab._btn_piezo_a_on)
    assert "enabled" in tab._lbl_piezo_status.text().lower()

    tab._on_piezo_a_off()
    _wait_enabled(qtbot, tab._btn_piezo_a_off)
    assert "disabled" in tab._lbl_piezo_status.text().lower()


def test_piezo_disabled_by_config_disables_controls(qapp):
    manager = create_mock_manager(config={"piezo": {"enabled": False}})
    with patch("softae.gui.tabs.tab_manual.piezo_config", return_value={"enabled": False}):
        tab = ManualControlTab(manager)
    try:
        assert not tab._btn_piezo_a_on.isEnabled()
        assert not tab._btn_piezo_apply.isEnabled()
    finally:
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            pv_worker = getattr(tab, "_pv_worker", None)
            if pv_worker is not None and pv_worker.isRunning():
                pv_worker.stop_worker()
            tab.close()


def test_piezo_enabled_in_instruments_enables_controls(qapp):
    manager = create_mock_manager(config={"piezo": {"enabled": True, "supports_cfg": True}})
    with patch(
        "softae.gui.tabs.tab_manual.piezo_config",
        return_value={
            "enabled": True,
            "frequency_hz": 500,
            "sweep_on_s": 2.0,
            "sweep_rest_s": 3.0,
            "driver": "piezo",
            "port": "COM16",
            "baud": 115200,
        },
    ):
        tab = ManualControlTab(manager)
    try:
        assert tab._btn_piezo_a_on.isEnabled()
        assert tab._btn_piezo_apply.isEnabled()
    finally:
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            pv_worker = getattr(tab, "_pv_worker", None)
            if pv_worker is not None and pv_worker.isRunning():
                pv_worker.stop_worker()
            tab.close()


def test_piezo_legacy_mode_disables_profile_controls_but_keeps_ab(qapp):
    manager = create_mock_manager(
        config={"piezo": {"enabled": True, "supports_l2": False, "supports_cfg": False}}
    )
    with patch(
        "softae.gui.tabs.tab_manual.piezo_config",
        return_value={
            "enabled": True,
            "frequency_hz": 500,
            "sweep_on_s": 2.0,
            "sweep_rest_s": 3.0,
        },
    ):
        tab = ManualControlTab(manager)
    try:
        assert tab._btn_piezo_a_on.isEnabled()
        assert tab._btn_piezo_a_off.isEnabled()
        assert not tab._spin_piezo_freq.isEnabled()
        assert not tab._spin_piezo_on_s.isEnabled()
        assert not tab._spin_piezo_rest_s.isEnabled()
        assert not tab._btn_piezo_apply.isEnabled()
        assert "legacy mode" in tab._lbl_piezo_status.text().lower()
    finally:
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            pv_worker = getattr(tab, "_pv_worker", None)
            if pv_worker is not None and pv_worker.isRunning():
                pv_worker.stop_worker()
            tab.close()


def test_piezo_capability_not_latched_before_connect(qapp):
    class _FakePiezo:
        def __init__(self):
            self.connected = False
            self.supports_l2 = False
            self.supports_cfg = False

        def status(self):
            if not self.connected:
                return {"connected": False}
            return {
                "connected": True,
                "supports_l2": self.supports_l2,
                "supports_cfg": self.supports_cfg,
                "config_supported": self.supports_l2 or self.supports_cfg,
            }

    class _FakeManager:
        def __init__(self):
            self.piezo = _FakePiezo()

        def get(self, name):
            if name == "piezo":
                return self.piezo
            raise RuntimeError("missing")

    manager = _FakeManager()
    with patch(
        "softae.gui.tabs.tab_manual.piezo_config",
        return_value={
            "enabled": True,
            "frequency_hz": 500,
            "sweep_on_s": 2.0,
            "sweep_rest_s": 3.0,
        },
    ):
        tab = ManualControlTab(manager)

    try:
        assert not tab._btn_piezo_apply.isEnabled()
        assert "connecting" in tab._lbl_piezo_status.text().lower()

        manager.piezo.connected = True
        manager.piezo.supports_l2 = True
        tab._refresh_piezo_capability_status()

        assert tab._btn_piezo_apply.isEnabled()
        assert "lean config supported" in tab._lbl_piezo_status.text().lower()
    finally:
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            pv_worker = getattr(tab, "_pv_worker", None)
            if pv_worker is not None and pv_worker.isRunning():
                pv_worker.stop_worker()
            tab.close()


def test_piezo_capability_rechecks_after_initial_legacy_snapshot(qapp):
    class _FakePiezo:
        def __init__(self):
            self.connected = True
            self.supports_l2 = False

        def status(self):
            return {
                "connected": self.connected,
                "supports_l2": self.supports_l2,
                "supports_cfg": False,
                "config_supported": self.supports_l2,
            }

    class _FakeManager:
        def __init__(self):
            self.piezo = _FakePiezo()

        def get(self, name):
            if name == "piezo":
                return self.piezo
            raise RuntimeError("missing")

    manager = _FakeManager()
    with patch(
        "softae.gui.tabs.tab_manual.piezo_config",
        return_value={
            "enabled": True,
            "frequency_hz": 500,
            "sweep_on_s": 2.0,
            "sweep_rest_s": 3.0,
        },
    ):
        tab = ManualControlTab(manager)

    try:
        assert not tab._btn_piezo_apply.isEnabled()
        assert "legacy mode" in tab._lbl_piezo_status.text().lower()

        manager.piezo.supports_l2 = True
        tab._ensure_piezo_capability_status()

        assert tab._btn_piezo_apply.isEnabled()
        assert "lean config supported" in tab._lbl_piezo_status.text().lower()
    finally:
        with patch("softae.gui.tabs.tab_manual.QMessageBox.warning"):
            pv_worker = getattr(tab, "_pv_worker", None)
            if pv_worker is not None and pv_worker.isRunning():
                pv_worker.stop_worker()
            tab.close()
