"""A registered head position must survive connection.

The dispenser head is a motor flipper with **no position feedback**. The operator is
therefore asked at launch where it physically is, and that answer becomes the driver's
belief — which the stage-motion interlock then trusts.

Connection is backgrounded and runs *after* the prompt. ``connect()`` used to assert
``_is_up = True``, so a head confirmed as **Lowered** was silently re-registered as
raised. That is not a display bug: :func:`softae.drivers.contracts.check_head_clear_to_move`
reads the same flag, so the interlock guarding stage travel would have been satisfied
with the head down and the tip over an electrode.

These tests pin the flag, and the interlock that consumes it, rather than the label.
"""

from __future__ import annotations

import asyncio

import pytest

from softae.drivers.mock_syringe import MockSyringe
from softae.errors import SafetyError


def _syringe() -> MockSyringe:
    return MockSyringe("syringe", {})


class TestConnectDoesNotOverwriteTheOperatorsAnswer:
    def test_a_head_registered_as_lowered_is_still_lowered_after_connecting(self):
        syr = _syringe()
        syr.set_head_state(False)          # operator answered "Lowered"
        asyncio.run(syr.connect())
        assert syr.is_head_up() is False

    def test_a_head_registered_as_raised_is_still_raised_after_connecting(self):
        syr = _syringe()
        syr.set_head_state(True)
        asyncio.run(syr.connect())
        assert syr.is_head_up() is True

    def test_reconnecting_does_not_reset_the_belief_either(self):
        # Instruments can be reconnected mid-session after a comms fault; that must
        # not quietly become a claim about where the head is.
        syr = _syringe()
        syr.set_head_state(False)
        asyncio.run(syr.connect())
        asyncio.run(syr.disconnect())
        asyncio.run(syr.connect())
        assert syr.is_head_up() is False


class TestTheInterlockSeesTheCorrectedFlag:
    """The consequence that makes this more than cosmetic."""

    def test_stage_motion_is_refused_when_the_operator_reported_a_lowered_head(self):
        from softae.drivers.contracts import check_head_clear_to_move

        syr = _syringe()
        syr.set_head_state(False)
        asyncio.run(syr.connect())

        with pytest.raises(SafetyError):
            check_head_clear_to_move(syr, instrument="stage")

    def test_stage_motion_is_permitted_once_the_head_is_registered_raised(self):
        from softae.drivers.contracts import check_head_clear_to_move

        syr = _syringe()
        syr.set_head_state(True)
        asyncio.run(syr.connect())
        check_head_clear_to_move(syr, instrument="stage")   # must not raise


class TestManualTabLabelFollowsTheDriver:
    def test_the_label_reports_descended_for_a_lowered_head(self, qapp):
        pytest.importorskip("PySide6")
        from softae.gui.tabs.tab_manual import ManualControlTab
        from softae.server.manager import InstrumentManager

        manager = InstrumentManager()
        syr = _syringe()
        manager.register(syr)

        tab = ManualControlTab(manager)
        syr.set_head_state(False)
        tab.refresh_head_label()
        assert "Descended" in tab._lbl_head_status.text()

        syr.set_head_state(True)
        tab.refresh_head_label()
        assert "Retracted" in tab._lbl_head_status.text()
