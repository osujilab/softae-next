"""Tests for the Autonomous placeholder tab (tab_autonomous.py).

Tab "7. Autonomous" is intentional scaffolding: layout and labels stay, but
its decorative action buttons must be disabled — an enabled "Approve &
Execute" that only emits a status string is an operator-trust hazard. Real
autonomous execution lives in the "Live BO Campaign" tab.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_autonomous import AutonomousTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


@pytest.fixture
def tab(qapp, manager):
    widget = AutonomousTab(manager)
    yield widget
    widget.close()


class TestPlaceholderButtonsDisabled:
    def test_approve_execute_button_disabled(self, tab: AutonomousTab):
        assert not tab._btn_approve.isEnabled()

    def test_stop_loop_button_disabled(self, tab: AutonomousTab):
        assert not tab._btn_stop_loop.isEnabled()

    def test_suggest_next_button_disabled(self, tab: AutonomousTab):
        assert not tab._btn_suggest.isEnabled()

    def test_disabled_buttons_carry_placeholder_tooltip(self, tab: AutonomousTab):
        for btn in (tab._btn_approve, tab._btn_stop_loop, tab._btn_suggest):
            tip = btn.toolTip()
            assert "Placeholder" in tip
            assert "Live BO Campaign" in tip  # points at the real surface

    def test_disabled_approve_does_not_emit_status(self, tab: AutonomousTab):
        """A disabled button must not fake a 'Running' workflow status."""
        emitted: list[str] = []
        tab.workflow_status_changed.connect(emitted.append)
        tab._btn_approve.click()      # click() on a disabled button is a no-op
        tab._btn_stop_loop.click()
        assert emitted == []

    def test_scaffolding_widgets_are_kept(self, tab: AutonomousTab):
        """The placeholder keeps its layout: selectors and notes stay present."""
        assert tab._combo_obj.count() > 0
        assert tab._combo_algo.count() > 0
        assert tab._spin_budget.value() == 50
        assert tab._te_annotation is not None
