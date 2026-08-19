"""The E-Stop button: what it commands, what it refuses to claim, and the head.

Two properties are the point of this file.

**The stop never moves the head.** Not on the way in (the park is run with the
default head policy, which issues no motion) and not on the way out unless the
operator has just said which way it is pointing. The previous behaviour flipped
the head on a belief with no feedback, so an emergency stop could drive it *down*
onto the board.

**The dialog no longer says "All instruments stopped / safe."** Nothing on this
rig reads back; that sentence was a claim about exceptions not being raised.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QMessageBox      # noqa: E402

from softae.core.safe_park import (                  # noqa: E402
    HEADLINE_COMMANDED,
    HEADLINE_NOTHING,
    HEADLINE_PARTIAL,
    SafeParkResult,
)
from softae.gui.widgets import emergency_stop as mod  # noqa: E402


class _Syringe:
    def __init__(self, up: bool = True, connected: bool = True):
        self.is_connected = connected
        self._up = up
        self.retracted = 0
        self.states: list[bool] = []
        self.pumps: list[int] = []

    def is_head_up(self) -> bool:
        return self._up

    def set_head_state(self, is_up: bool) -> None:
        self._up = bool(is_up)
        self.states.append(bool(is_up))

    def head_retract(self) -> None:
        self.retracted += 1
        self._up = True

    def halt_pump(self, pump_id) -> None:
        self.pumps.append(int(pump_id))


class _Manager:
    def __init__(self, syringe=None):
        self._syringe = syringe

    def get(self, name: str):
        if name == "syringe":
            return self._syringe
        raise KeyError(name)


@pytest.fixture
def button(qtbot):
    syr = _Syringe(up=False)
    btn = mod.EmergencyStopButton(_Manager(syr))
    qtbot.addWidget(btn)
    return btn, syr


def _silence_dialogs(monkeypatch, btn, *, head_answer=None, retract=False):
    """Replace both prompts. ``head_answer`` is ``True``/``False``/``None``."""
    shown: list[str] = []
    monkeypatch.setattr(btn, "_report", lambda result: shown.append(result.describe()))

    def fake_ask():
        if head_answer is None:
            return
        btn._manager.get("syringe").set_head_state(head_answer)
        if retract:
            btn._offer_retract(btn._manager.get("syringe"), head_answer)

    monkeypatch.setattr(btn, "_ask_head_state", fake_ask)
    return shown


# ── The stop itself ──────────────────────────────────────────────────────────

class TestTheStop:
    def test_the_stop_does_not_move_the_head(self, qtbot, button, monkeypatch):
        """Belief says DOWN; the old code flipped it, i.e. drove it down."""
        btn, syr = button
        _silence_dialogs(monkeypatch, btn)

        with qtbot.waitSignal(btn.parked, timeout=3000):
            btn._on_stop()
        qtbot.waitUntil(lambda: btn.isEnabled(), timeout=5000)

        assert syr.retracted == 0
        assert syr.is_head_up() is False

    def test_the_stop_still_halts_the_pumps(self, qtbot, button, monkeypatch):
        btn, syr = button
        _silence_dialogs(monkeypatch, btn)

        btn._on_stop()
        qtbot.waitUntil(lambda: syr.pumps == [0, 1, 2], timeout=5000)

    def test_the_latch_fires_before_the_sequence_runs(self, qtbot, button,
                                                      monkeypatch):
        btn, _ = button
        _silence_dialogs(monkeypatch, btn)

        with qtbot.waitSignal(btn.parked, timeout=3000) as latch:
            btn._on_stop()
        qtbot.waitUntil(lambda: btn.isEnabled(), timeout=5000)

        assert latch.args == ["operator emergency stop"]

    def test_a_failing_prompt_does_not_break_the_stop(self, qtbot, button,
                                                      monkeypatch):
        """A stop must not end in a traceback because a dialog misbehaved."""
        btn, _ = button
        monkeypatch.setattr(btn, "_report", lambda result: None)
        monkeypatch.setattr(btn, "_ask_head_state",
                            lambda: (_ for _ in ()).throw(RuntimeError("no display")))

        btn._on_stop()
        qtbot.waitUntil(lambda: btn.isEnabled(), timeout=5000)


# ── What the operator is told ────────────────────────────────────────────────

class TestTheReport:
    def test_it_no_longer_claims_everything_is_safe(self):
        text = SafeParkResult(commanded=["lamp off"]).describe()
        assert "safe" not in text.lower()
        assert "stopped / safe" not in text

    def test_it_separates_commanded_from_verified(self, qtbot, button, monkeypatch):
        btn, _ = button
        shown = _silence_dialogs(monkeypatch, btn)

        btn._on_stop()
        qtbot.waitUntil(lambda: bool(shown), timeout=5000)

        assert "Commanded" in shown[0]
        assert "Nothing was verified" in shown[0]
        assert "NOT verifiable" in shown[0]

    def test_the_head_is_named_as_unverifiable(self, qtbot, button, monkeypatch):
        btn, _ = button
        shown = _silence_dialogs(monkeypatch, btn)

        btn._on_stop()
        qtbot.waitUntil(lambda: bool(shown), timeout=5000)

        assert "head" in shown[0].lower()


# ── The headline: three grades, not two ──────────────────────────────────────

def _capture_report(monkeypatch, btn, result):
    """Run ``_report`` against a real QMessageBox and read back what it said."""
    seen: dict = {}

    def fake_exec(self):
        seen["text"] = self.text()
        seen["informative"] = self.informativeText()
        seen["icon"] = self.icon()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    btn._report(result)
    return seen


class TestTheHeadline:
    """``result.ok`` used to pick the words, and it means *nothing raised*.

    So a park against a rig this process is not connected to — the crisp case,
    and the one an attached GUI hits every time — skipped every instrument,
    raised nothing, and was headed *"Stop commands were issued."* under a benign
    blue icon with an empty ``commanded`` list. ``describe()`` was always honest;
    only the headline over-read.
    """

    def test_report_of_a_commanded_park_says_issued_with_the_information_icon(
        self, qtbot, button, monkeypatch
    ):
        btn, _ = button
        seen = _capture_report(monkeypatch, btn,
                               SafeParkResult(commanded=["lamp off"]))

        assert seen["text"] == HEADLINE_COMMANDED
        assert seen["icon"] == QMessageBox.Icon.Information

    def test_report_of_a_refusing_park_says_partial_with_the_warning_icon(
        self, qtbot, button, monkeypatch
    ):
        btn, _ = button
        seen = _capture_report(
            monkeypatch, btn,
            SafeParkResult(commanded=["lamp off"], errors=["pump 0: dead"]))

        assert seen["text"] == HEADLINE_PARTIAL
        assert seen["icon"] == QMessageBox.Icon.Warning

    def test_report_of_a_park_that_commanded_nothing_warns_and_says_so(
        self, qtbot, button, monkeypatch
    ):
        btn, _ = button
        seen = _capture_report(
            monkeypatch, btn,
            SafeParkResult(skipped=["lamp: not connected"]))

        assert seen["text"] == HEADLINE_NOTHING
        assert seen["icon"] == QMessageBox.Icon.Warning
        assert seen["text"] != HEADLINE_COMMANDED

    def test_pressing_stop_against_an_unconnected_rig_does_not_report_success(
        self, qtbot, monkeypatch
    ):
        """End to end through the real park: press the red button with nothing
        connected and the dialog must not say the stop was issued."""
        btn = mod.EmergencyStopButton(_Manager(None))
        qtbot.addWidget(btn)
        monkeypatch.setattr(btn, "_ask_head_state", lambda: None)

        seen: dict = {}
        monkeypatch.setattr(
            QMessageBox, "exec",
            lambda self: seen.update(text=self.text(), icon=self.icon()) or 0)

        btn._on_stop()
        qtbot.waitUntil(lambda: bool(seen), timeout=5000)
        qtbot.waitUntil(lambda: btn.isEnabled(), timeout=5000)

        assert seen["text"] == HEADLINE_NOTHING
        assert seen["icon"] == QMessageBox.Icon.Warning


# ── The operator is the sensor ───────────────────────────────────────────────

class TestTheHeadPrompt:
    """Driven through the two decision seams (``_ask_head_choice`` /
    ``_confirm_retract``) rather than through Qt button identity — what is worth
    pinning is which driver calls each answer produces."""

    def _answer(self, monkeypatch, btn, *, head, retract=False):
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: head)
        monkeypatch.setattr(btn, "_confirm_retract", lambda is_up: retract)

    def test_answering_records_the_position_without_moving_anything(
        self, qtbot, button, monkeypatch
    ):
        btn, syr = button
        self._answer(monkeypatch, btn, head=False, retract=False)

        btn._ask_head_state()

        assert syr.states == [False]
        assert syr.retracted == 0

    def test_a_retract_is_offered_and_honoured(self, qtbot, button, monkeypatch):
        btn, syr = button
        self._answer(monkeypatch, btn, head=False, retract=True)

        btn._ask_head_state()

        assert syr.states == [False]
        assert syr.retracted == 1

    def test_the_belief_is_written_before_any_motion_is_offered(
        self, qtbot, button, monkeypatch
    ):
        """Ordering is the whole mechanism: ``head_retract`` is belief-gated, so
        the answer must land first or the flip is the same coin toss as before."""
        order: list[str] = []
        btn, syr = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: False)
        monkeypatch.setattr(btn, "_confirm_retract",
                            lambda is_up: order.append("offer") or True)
        syr.set_head_state = lambda up: order.append(f"record:{up}")
        syr.head_retract = lambda: order.append("retract")

        btn._ask_head_state()

        assert order == ["record:False", "offer", "retract"]

    def test_not_sure_records_nothing_and_moves_nothing(self, qtbot, button,
                                                        monkeypatch):
        """A guess written into the belief is worse than no answer: later paths
        act on it."""
        btn, syr = button
        offered: list[bool] = []
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: None)
        monkeypatch.setattr(btn, "_confirm_retract",
                            lambda is_up: offered.append(True) or True)

        btn._ask_head_state()

        assert syr.states == []
        assert syr.retracted == 0
        assert syr.is_head_up() is False
        assert offered == []

    def test_the_retract_offer_is_unconditional_not_belief_gated(
        self, qtbot, button, monkeypatch
    ):
        """Offered in both branches. It is ``head_retract`` that is conditional —
        which is safe now, because the operator has just told it the truth."""
        btn, syr = button
        syr._up = True
        offered: list[bool] = []
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: True)
        monkeypatch.setattr(btn, "_confirm_retract",
                            lambda is_up: offered.append(is_up) or False)

        btn._ask_head_state()

        assert offered == [True]
        assert syr.states == [True]

    def test_the_prompt_offers_three_answers(self, qtbot, button, monkeypatch):
        """"Not sure" must be reachable — an operator forced to pick UP or DOWN
        will pick one, and a guess is what this whole path exists to avoid."""
        btn, _ = button
        seen: dict = {}

        def capture(self):
            seen["labels"] = [b.text() for b in self.buttons()]
            return 0

        monkeypatch.setattr(QMessageBox, "exec", capture)
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)

        assert btn._ask_head_choice() is None      # dismissed = not sure
        assert len(seen["labels"]) == 3
        assert any("Not sure" in t for t in seen["labels"])

    def test_no_syringe_is_asked_about_nothing(self, qtbot, monkeypatch):
        btn = mod.EmergencyStopButton(_Manager(None))
        qtbot.addWidget(btn)
        monkeypatch.setattr(
            btn, "_ask_head_choice",
            lambda: (_ for _ in ()).throw(AssertionError("prompted anyway")))

        btn._ask_head_state()      # must simply return

    def test_a_disconnected_syringe_is_asked_about_nothing(self, qtbot, monkeypatch):
        btn = mod.EmergencyStopButton(_Manager(_Syringe(connected=False)))
        qtbot.addWidget(btn)
        monkeypatch.setattr(
            btn, "_ask_head_choice",
            lambda: (_ for _ in ()).throw(AssertionError("prompted anyway")))

        btn._ask_head_state()

    def test_a_failing_retract_is_reported_not_raised(self, qtbot, button,
                                                      monkeypatch):
        btn, syr = button

        def boom():
            raise RuntimeError("air line")

        syr.head_retract = boom
        warned: list[str] = []
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, **k: warned.append(a[2])))
        monkeypatch.setattr(btn, "_confirm_retract", lambda is_up: True)

        btn._offer_retract(syr, False)

        assert warned and "air line" in warned[0]
