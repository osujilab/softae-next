from __future__ import annotations

from unittest.mock import patch

import pytest

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_experiment import ExperimentBuilderTab


@pytest.fixture
def tab(qapp):
    manager = create_mock_manager(config={})
    widget = ExperimentBuilderTab(manager)
    # Pin a deterministic engine recipe so piezo assertions are independent of the
    # operator's [dropcast].default_recipe.
    i = widget._combo_deposit_recipe.findData("single_drop")
    if i >= 0:
        widget._combo_deposit_recipe.setCurrentIndex(i)
    yield widget
    widget.close()


def _step_names(tab: ExperimentBuilderTab) -> list[str]:
    wf = tab._generate_workflow()
    return [s.name for s in wf.setup + wf.loop_steps + wf.teardown]


def test_no_piezo_steps_when_events_disabled(tab):
    with patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
        "enabled": False,
        "liquid_events": {"enabled": False},
    }):
        names = _step_names(tab)
    assert not any(name.startswith("piezo_") for name in names)


def test_piezo_steps_inserted_for_full_protocol(tab):
    # all_elution=False → the deposit-only scope (piezo brackets the deposit phase).
    with patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
        "enabled": True,
        "liquid_events": {
            "enabled": True,
            "settings_source": "manual_profile",
            "channel_a": True,
            "all_elution": False,
            "frequency_hz": 700,
            "sweep_on_s": 1.0,
            "sweep_rest_s": 2.0,
        },
    }):
        wf = tab._generate_workflow()

    names = [s.name for s in wf.setup + wf.teardown]
    # The engine wraps each channel's deposit with piezo on/off and returns to
    # standby in teardown.
    assert any(n.startswith("piezo_on_ch") for n in names)
    assert any(n.startswith("piezo_off_ch") for n in names)
    assert "piezo_standby" in names
    assert wf.metadata["piezo"] == "applied"
    # on precedes the deposit, off follows it.
    assert names.index("piezo_on_ch1") < names.index("deposit_ch1")
    assert names.index("deposit_ch1") < names.index("piezo_off_ch1")


def test_piezo_all_elution_wraps_every_elution_event(tab):
    # Default (no all_elution key) → all-elution scope: the piezo brackets the
    # startup flush, each channel's deposit, and the final flush.
    with patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
        "enabled": True,
        "liquid_events": {
            "enabled": True,
            "settings_source": "manual_profile",
            "channel_a": True,
            "frequency_hz": 700,
            "sweep_on_s": 1.0,
            "sweep_rest_s": 2.0,
        },
    }):
        wf = tab._generate_workflow()

    names = [s.name for s in wf.setup + wf.teardown]
    # Startup flush, deposit, and final flush are each individually bracketed.
    assert "piezo_on_startup_flush" in names and "piezo_off_startup_flush" in names
    assert "piezo_on_deposit_ch1" in names and "piezo_off_deposit_ch1" in names
    assert "piezo_on_final_flush" in names and "piezo_off_final_flush" in names
    assert "piezo_standby" in names
    # Piezo brackets the deposit; the (non-elution) EIS falls outside it.
    assert names.index("piezo_on_deposit_ch1") < names.index("deposit_ch1")
    assert names.index("deposit_ch1") < names.index("piezo_off_deposit_ch1")
    assert names.index("piezo_off_deposit_ch1") < names.index("measure_eis_ch1")


def test_event_profile_applied_once_when_selected(tab):
    with patch("softae.gui.tabs.tab_experiment.piezo_config", return_value={
        "enabled": True,
        "liquid_events": {
            "enabled": True,
            "settings_source": "liquid_event_profile",
            "channel_a": True,
            "frequency_hz": 850,
            "sweep_on_s": 1.4,
            "sweep_rest_s": 2.8,
        },
    }):
        wf = tab._generate_workflow()

    names = [s.name for s in wf.setup]
    assert names.count("piezo_event") == 1     # once, in setup
    step = next(s for s in wf.setup if s.name == "piezo_event")
    assert step.params["frequency_hz"] == 850
    assert step.params["on_s"] == pytest.approx(1.4)
    assert step.params["rest_s"] == pytest.approx(2.8)
