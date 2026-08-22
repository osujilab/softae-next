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


# ── The dry-purge boundary ───────────────────────────────────────────────────
#
# Asserted over the call sites themselves rather than over behaviour, because
# the guarantee is about code that does not exist: *no safety path opts in*. A
# behavioural test can only check the sites someone remembered to write one for,
# and the failure being guarded against is a future edit adding a fifth site or
# flipping one of these four. Reading the source is what makes that trip.

_PARK_FUNCS = {"safe_park", "safe_park_async"}

#: Orderly, operator-initiated exits: dry gas keeps flowing, and the *device*
#: decides for how long via its own ``ctrl_timeout``.
DRY_PURGE_SITES = {
    "src/softae/gui/main_window.py": "_safe_park_on_exit",
    "src/softae/gui/widgets/safe_exit.py": "run",
}

#: Safety paths. Every one of these zeroes the humidifier immediately, closing
#: both Aalborg PSVs — operator ruling: an emergency stop that leaves gas
#: flowing is not an emergency stop.
ZEROING_SITES = {
    "src/softae/gui/widgets/emergency_stop.py": "the E-Stop",
    "src/softae/core/autonomous_wiring.py": "the fault-class campaign park",
    # Both of this module's parks. ``ParkGuard.park`` is the shared body behind
    # a signal handler, the campaign CLI's teardown *and* the campaign's abort
    # catch-all, so an opt-in there could not be confined to the orderly half.
    "src/softae/core/shutdown.py": "crash and signal recovery",
    "src/softae/gui/widgets/unclean_shutdown.py": "unclean-shutdown recovery",
}


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _park_calls(rel_path: str):
    """``(enclosing function, line, Call)`` for every park call in a module.

    Deduplicated by position and attributed to the *innermost* enclosing
    function, so a call inside a closure is named once and named usefully.
    """
    import ast

    tree = ast.parse((_repo_root() / rel_path).read_text(encoding="utf-8"))
    calls: dict[tuple[int, int], ast.Call] = {}
    owner: dict[tuple[int, int], tuple[int, str]] = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(
                func, "id", None)
            if name not in _PARK_FUNCS:
                continue
            key = (node.lineno, node.col_offset)
            calls[key] = node
            if key not in owner or scope.lineno > owner[key][0]:
                owner[key] = (scope.lineno, scope.name)
    return [(owner[k][1], k[0], call) for k, call in sorted(calls.items())]


def _dry_purge_arg(call):
    """The ``rh_dry_purge`` argument node, ``None`` if the call omits it.

    A ``**kwargs`` splat returns the splat node: it cannot be shown to withhold
    the purge, and on a safety path *unproven* is treated as *failed*.
    """
    for kw in call.keywords:
        if kw.arg in (None, "rh_dry_purge"):
            return kw.value
    return None


def _is_true(node) -> bool:
    import ast

    return isinstance(node, ast.Constant) and node.value is True


class TestTheDryPurgeIsOrderlyExitsOnly:
    """The boundary, as a test rather than a promise."""

    @pytest.mark.parametrize("rel_path,what", sorted(ZEROING_SITES.items()))
    def test_no_safety_path_asks_for_a_dry_purge(self, rel_path, what):
        import ast

        calls = _park_calls(rel_path)
        # Anti-vacuity: a renamed import or a moved call must fail here rather
        # than silently reduce this to an assertion over an empty list.
        assert calls, f"no safe_park call found in {rel_path} — has {what} moved?"

        for func, line, call in calls:
            arg = _dry_purge_arg(call)
            proven_off = arg is None or (isinstance(arg, ast.Constant)
                                         and arg.value is False)
            assert proven_off, (
                f"{rel_path}:{line} ({func}) — {what} must zero the humidifier "
                "immediately. A dry purge leaves both Aalborg PSVs open until "
                "the Trinket's own deadman closes them; that is the right trade "
                "for a planned exit and the wrong one for a safety path."
            )

    @pytest.mark.parametrize("rel_path,func_name", sorted(DRY_PURGE_SITES.items()))
    def test_both_orderly_exits_ask_for_a_dry_purge(self, rel_path, func_name):
        """The other half, and the control on the half above.

        Without it the safety assertion would still pass with the feature
        deleted entirely — and it is what shows the reader that this file's
        detector can tell ``True`` from its absence.
        """
        sites = [(f, line, call) for f, line, call in _park_calls(rel_path)
                 if f == func_name]
        assert len(sites) == 1, (
            f"expected exactly one park call in {rel_path}::{func_name}, "
            f"found {len(sites)}"
        )
        _f, line, call = sites[0]
        assert _is_true(_dry_purge_arg(call)), (
            f"{rel_path}:{line} ({func_name}) — an orderly exit must pass "
            "rh_dry_purge=True. Duty 0 is the firmware's auto-shutoff, not its "
            "dry end: it closes both PSVs and lets room air back into the "
            "chamber."
        )

    def test_no_orderly_exit_encodes_the_purge_duration(self):
        """The window belongs to the device, and to no host-side constant.

        The Trinket's ``ctrl_timeout`` decides when the valves close. A host
        timer that re-zeroed after N seconds would agree with it today and
        silently truncate the purge the moment ``ctrl_timeout`` is raised — the
        wrong value wearing the safe value's clothes. So neither exit site may
        name a duration at all, whether as a literal or as an argument.
        """
        import ast

        for rel_path, func_name in sorted(DRY_PURGE_SITES.items()):
            for func, line, call in _park_calls(rel_path):
                if func != func_name:
                    continue
                timing = [kw.arg for kw in call.keywords
                          if kw.arg and any(t in kw.arg for t in
                                            ("duration", "timeout", "seconds",
                                             "deadman", "hold"))]
                assert not timing, (
                    f"{rel_path}:{line} passes {timing} — the purge window is "
                    "the Trinket's to decide, not this host's."
                )
                numbers = [n.value for n in ast.walk(call)
                           if isinstance(n, ast.Constant)
                           and isinstance(n.value, (int, float))
                           and not isinstance(n.value, bool)]
                assert not numbers, (
                    f"{rel_path}:{line} passes literal {numbers} into the park. "
                    "Numbers are barred from this call outright rather than by "
                    "name: the one that must never appear is a purge duration, "
                    "and it would arrive under whatever name its author chose. "
                    "If the number genuinely is not a duration, add it here "
                    "deliberately — the tripwire is asking for a decision, not "
                    "claiming your constant is wrong."
                )


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

        parked.append({"reason": reason, "retract_head": retract_head, **kw})
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

    def test_it_asks_for_the_rh_dry_purge(self, qtbot, button, monkeypatch):
        """Both orderly ways out of the GUI leave the chamber in the same state.

        Safe Exit and the window's own close are the same act with different
        buttons, so they park the humidifier the same way: dry gas still
        flowing, the Trinket's ``ctrl_timeout`` deadman closing the valves.
        Duty 0 is the firmware's auto-shutoff, not its dry end — it shuts both
        PSVs and admits room air.
        """
        btn, _syr, parked = button
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: True)

        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert parked[0]["rh_dry_purge"] is True
        # The head decision must survive the new argument, not be replaced by it.
        assert parked[0]["retract_head"] is True

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

    def _park_returning(self, qtbot, monkeypatch, result):
        """A Safe Exit whose park yields *result*; returns (button, exited, warned)."""
        from softae.gui.widgets import safe_exit as mod
        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr("softae.core.safe_park.safe_park",
                            lambda *a, **k: result)
        warned: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: warned.append(a[2])
                         or QMessageBox.StandardButton.Cancel))

        btn = mod.SafeExitButton(_Manager(_Syringe(up=True)))
        qtbot.addWidget(btn)
        exited: list[bool] = []
        btn.exit_requested.connect(lambda: exited.append(True))
        btn.click()
        qtbot.wait(200)
        return btn, exited, warned

    def test_a_partial_park_does_not_exit_without_confirmation(self, qtbot,
                                                               monkeypatch):
        """Closing the window removes the operator's easiest way to see what failed."""
        from softae.core.safe_park import HEADLINE_PARTIAL, SafeParkResult

        btn, exited, warned = self._park_returning(
            qtbot, monkeypatch, SafeParkResult(errors=["lamp: no reply"]))

        assert exited == []
        assert btn.isEnabled()
        assert warned and HEADLINE_PARTIAL in warned[0]

    def test_a_park_that_commanded_nothing_does_not_exit_without_confirmation(
        self, qtbot, monkeypatch
    ):
        """The exit path's version of the E-Stop's misreport, and the quieter one.

        Safe Exit never claimed success in words — it signalled it by *closing
        the window*. So a park that reached no instrument at all looked exactly
        like one that reached every instrument, and the worker discarded the
        evidence anyway by emitting only ``result.errors``.
        """
        from softae.core.safe_park import HEADLINE_NOTHING, SafeParkResult

        btn, exited, warned = self._park_returning(
            qtbot, monkeypatch, SafeParkResult(skipped=["lamp: not connected"]))

        assert exited == []
        assert btn.isEnabled()
        assert warned and HEADLINE_NOTHING in warned[0]

    def test_a_park_that_commanded_something_exits_without_a_dialog(
        self, qtbot, monkeypatch
    ):
        """The inverse, so the change cannot be written as "always confirm"."""
        from softae.core.safe_park import SafeParkResult

        _btn, exited, warned = self._park_returning(
            qtbot, monkeypatch, SafeParkResult(commanded=["lamp off"]))

        assert exited == [True]
        assert warned == []
