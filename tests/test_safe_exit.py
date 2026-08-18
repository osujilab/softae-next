"""Safe Exit — parking on purpose, with a say over the head.

The behaviour worth protecting is narrow and easy to regress: **the head moves
only when a human has just said which way it is pointing.** An automatic park
adds no motion to an unknown; Safe Exit asks, and both of its answers are
legitimate. A test suite that only checked "safe_park retracts" would pass while
the choice silently did nothing — and, worse, would pin the old default in which
an automatic park flipped a head it could not sense.
"""

from __future__ import annotations

import pytest

from softae.core.safe_park import safe_park


class _Syringe:
    def __init__(self, up: bool = True, connected: bool = True):
        self.is_connected = connected
        self._up = up
        self.retracted = 0
        self.pumps: list[int] = []

    def is_head_up(self) -> bool:
        return self._up

    def head_retract(self) -> None:
        self.retracted += 1
        self._up = True

    def halt_pump(self, pump_id) -> None:
        self.pumps.append(int(pump_id))

    def single_pump(self, _speed, pump_id, _vol, _rate) -> None:
        # Present so a park that regressed to halting via a dispense is visible
        # rather than merely absent from ``pumps``.
        raise AssertionError("a park must not halt pumps with a dispense")


class _Manager:
    def __init__(self, syringe=None):
        self._syringe = syringe

    def get(self, name: str):
        if name == "syringe":
            return self._syringe
        raise KeyError(name)


# ── The park itself ──────────────────────────────────────────────────────────

class TestRetractHead:
    def test_the_default_adds_no_motion_to_an_unknown(self):
        """Reversed deliberately. The old default retracted "because nobody is
        there to decide" — sound about the *decision*, wrong about the
        *capability*: with no feedback, ``head_retract`` is a conditional flip on
        a belief, and on a stale belief it drives the head **down**."""
        syr = _Syringe(up=False)
        result = safe_park(_Manager(syr), reason="test")
        assert syr.retracted == 0
        assert syr.is_head_up() is False
        assert any("head" in u.lower() for u in result.unverifiable)

    def test_declining_leaves_the_head_where_it_is(self):
        syr = _Syringe(up=False)
        result = safe_park(_Manager(syr), reason="test", retract_head=False)
        assert syr.retracted == 0
        assert not syr.is_head_up()
        assert any("left lowered" in a for a in result.actions)

    def test_an_operator_instruction_still_retracts(self):
        """Never refusing the operator is the other half of the policy."""
        syr = _Syringe(up=False)
        result = safe_park(_Manager(syr), reason="test", retract_head=True)
        assert syr.retracted == 1
        assert any("head retract commanded" in a for a in result.actions)

    def test_the_choice_is_recorded_as_an_action_not_a_skip(self):
        """It is something the park *did*, and the log is the only durable trace of
        why the head is found down next session."""
        result = safe_park(_Manager(_Syringe(up=False)), retract_head=False)
        assert any("left lowered" in a for a in result.actions)
        assert not any("head" in s for s in result.skipped)
        assert result.ok

    def test_leaving_the_head_down_still_halts_the_pumps(self):
        """The head is one decision; the rest of the park is not up for negotiation.

        Fluid motion, thermal load and the lamp are unconditional — a stop that
        skipped them because the head stayed down would not be a park at all.
        """
        syr = _Syringe(up=False)
        safe_park(_Manager(syr), retract_head=False, pump_ids=(0, 1, 2))
        assert syr.pumps == [0, 1, 2]

    def test_a_failing_retract_is_reported_rather_than_raised(self):
        class Stubborn(_Syringe):
            def head_retract(self):
                raise RuntimeError("stage timeout")

        result = safe_park(_Manager(Stubborn(up=False)), retract_head=True)
        assert not result.ok
        assert any("stage timeout" in e for e in result.errors)

    def test_no_syringe_is_not_an_error_in_either_mode(self):
        assert safe_park(_Manager(None), retract_head=True).ok
        assert safe_park(_Manager(None), retract_head=False).ok


@pytest.mark.asyncio
async def test_the_async_wrapper_threads_the_choice_through():
    """The campaign loop's path must not silently retract when told not to."""
    from softae.core.safe_park import safe_park_async

    syr = _Syringe(up=False)
    await safe_park_async(_Manager(syr), retract_head=False)
    assert syr.retracted == 0


# ── Reading the head state ───────────────────────────────────────────────────

class TestHeadIsDown:
    def _detect(self, manager):
        from softae.gui.widgets.safe_exit import head_is_down

        return head_is_down(manager)

    def test_a_lowered_connected_head_is_detected(self):
        assert self._detect(_Manager(_Syringe(up=False)))

    def test_a_raised_head_is_not(self):
        assert not self._detect(_Manager(_Syringe(up=True)))

    def test_an_absent_syringe_asks_nothing(self):
        assert not self._detect(_Manager(None))

    def test_a_disconnected_syringe_asks_nothing(self):
        assert not self._detect(_Manager(_Syringe(up=False, connected=False)))

    def test_a_driver_that_does_not_track_the_head_asks_nothing(self):
        """"Do not invent a belief" — the same posture as ``check_head_clear``.

        The operator must never be asked a question about hardware whose state
        nothing actually knows.
        """

        class Untracked:
            is_connected = True

        assert not self._detect(_Manager(Untracked()))

    def test_an_unreadable_state_asks_nothing(self):
        class Broken(_Syringe):
            def is_head_up(self):
                raise OSError("port gone")

        assert not self._detect(_Manager(Broken()))


# ── The button ───────────────────────────────────────────────────────────────

pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture
def button(qtbot, monkeypatch):
    from softae.gui.widgets import safe_exit as mod

    syr = _Syringe(up=False)
    manager = _Manager(syr)
    parked: list[dict] = []

    def fake_park(mgr, *, reason="", retract_head=True, **kw):
        from softae.core.safe_park import SafeParkResult

        parked.append({"reason": reason, "retract_head": retract_head})
        if retract_head:
            syr.head_retract()
        return SafeParkResult(commanded=["parked"])

    monkeypatch.setattr("softae.core.safe_park.safe_park", fake_park)

    btn = mod.SafeExitButton(manager)
    qtbot.addWidget(btn)
    return btn, syr, parked


class TestTheButton:
    def test_it_is_amber_and_not_the_estop_red(self, button):
        """The colour carries the meaning: a planned stop, not an emergency."""
        btn, _, _ = button
        assert "#f57c00" in btn.styleSheet()
        assert "#d32f2f" not in btn.styleSheet()

    def test_cancelling_touches_nothing_at_all(self, qtbot, button, monkeypatch):
        btn, syr, parked = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: None)

        btn.click()
        qtbot.wait(50)

        assert parked == []
        assert syr.retracted == 0
        assert btn.isEnabled()

    def test_choosing_to_leave_it_down_reaches_safe_park(self, qtbot, button,
                                                        monkeypatch):
        btn, syr, parked = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: False)

        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert parked[0]["retract_head"] is False
        assert syr.retracted == 0

    def test_choosing_to_raise_it_reaches_safe_park(self, qtbot, button, monkeypatch):
        btn, syr, parked = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: True)

        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert parked[0]["retract_head"] is True
        assert syr.retracted == 1

    def test_a_raised_head_is_never_asked_about(self, qtbot, monkeypatch):
        """The prompt appears only when there is a decision to make."""
        from softae.gui.widgets import safe_exit as mod
        from softae.core.safe_park import SafeParkResult

        monkeypatch.setattr(
            "softae.core.safe_park.safe_park",
            lambda *a, **k: SafeParkResult(commanded=["parked"]))

        btn = mod.SafeExitButton(_Manager(_Syringe(up=True)))
        asked = []
        monkeypatch.setattr(btn, "_ask_head_choice",
                            lambda: asked.append(True) or True)

        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert asked == []

    def test_the_park_latch_fires_before_the_hardware_moves(self, qtbot, button,
                                                            monkeypatch):
        """Nothing may start actuating between the press and the park completing —
        the same contract the E-Stop's ``parked`` signal carries."""
        btn, _, _ = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: True)

        with qtbot.waitSignal(btn.parked, timeout=3000) as latch:
            btn.click()

        assert latch.args == ["operator safe exit"]

    def test_a_second_click_while_parking_is_ignored(self, qtbot, button, monkeypatch):
        btn, _, parked = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: True)

        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()
            btn.click()

        assert len(parked) == 1

    def test_a_partial_park_does_not_exit_without_confirmation(self, qtbot,
                                                               monkeypatch):
        """Closing the window removes the operator's easiest way to see what failed."""
        from softae.gui.widgets import safe_exit as mod
        from softae.core.safe_park import SafeParkResult
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(
            "softae.core.safe_park.safe_park",
            lambda *a, **k: SafeParkResult(errors=["lamp: no reply"]))
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel))

        btn = mod.SafeExitButton(_Manager(_Syringe(up=True)))
        qtbot.addWidget(btn)

        exited = []
        btn.exit_requested.connect(lambda: exited.append(True))
        btn.click()
        qtbot.wait(200)

        assert exited == []
        assert btn.isEnabled()
