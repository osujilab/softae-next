from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_liquid_model import LiquidModelTab


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def manager():
    return create_mock_manager(config={})


class TestLiquidModelTab:
    def test_always_shows_three_lines_when_config_sparse(self, qapp, manager):
        sparse_cfg = {
            "enabled": True,
            "beta": 0.25,
            "eta_ref_mpas": 1.0,
            "alpha_growth_per_run": 0.0,
            "line": {
                "0": {
                    "cracking_kpa_per_valve": 8.0,
                    "compliance_uL_per_kpa": 0.55,
                    "alpha_base": 0.2,
                    "viscosity_mpas": 1.0,
                }
            },
        }

        with patch("softae.config.loader.liquid_handling_config", return_value=sparse_cfg):
            tab = LiquidModelTab(manager)
        try:
            assert sorted(tab._line_liq_widgets.keys()) == [0, 1, 2]
        finally:
            tab.close()

    def test_prime_labels_are_computed(self, qapp, manager):
        tab = LiquidModelTab(manager)
        try:
            for line_id in (0, 1, 2):
                text = tab._line_liq_widgets[line_id]["prime"].text()
                assert text.endswith("uL")
                assert text != "-- uL"
        finally:
            tab.close()

    def test_apply_saves_all_three_lines(self, qapp, manager):
        tab = LiquidModelTab(manager)
        try:
            with patch("softae.config.loader.save_liquid_handling_config") as save_mock, patch(
                "softae.config.loader.save_piezo_config"
            ):
                tab._on_apply_liquid_model()
            save_mock.assert_called_once()
            payload = save_mock.call_args[0][0]
            assert set(payload["line"].keys()) == {"0", "1", "2"}
        finally:
            tab.close()
