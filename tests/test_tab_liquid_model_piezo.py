from __future__ import annotations

from unittest.mock import patch

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_liquid_model import LiquidModelTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


def test_piezo_event_settings_load_and_save_shape(qapp, manager):
    with patch("softae.config.loader.piezo_config", return_value={
        "enabled": False,
        "liquid_events": {
            "enabled": True,
            "settings_source": "liquid_event_profile",
            "channel_a": True,
            "frequency_hz": 900,
            "sweep_on_s": 1.5,
            "sweep_rest_s": 2.5,
        },
    }):
        tab = LiquidModelTab(manager)
    try:
        assert tab._chk_piezo_events_enabled.isChecked() is True
        assert tab._combo_piezo_source.currentText() == "liquid_event_profile"

        with patch("softae.config.loader.save_liquid_handling_config"), patch(
            "softae.config.loader.save_piezo_config"
        ) as save_piezo:
            tab._on_apply_liquid_model()

        payload = save_piezo.call_args[0][0]
        assert "liquid_events" in payload
        assert payload["liquid_events"]["settings_source"] in {
            "manual_profile",
            "liquid_event_profile",
        }
    finally:
        tab.close()


def test_piezo_profile_inputs_disabled_for_manual_source(qapp, manager):
    tab = LiquidModelTab(manager)
    try:
        tab._combo_piezo_source.setCurrentText("manual_profile")
        tab._refresh_piezo_event_inputs_enabled()
        assert not tab._spin_piezo_event_freq.isEnabled()
        assert not tab._spin_piezo_event_on_s.isEnabled()
        assert not tab._spin_piezo_event_rest_s.isEnabled()
    finally:
        tab.close()
