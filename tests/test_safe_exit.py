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


class _RH:
    """A humidity controller double, hand-written rather than a ``MagicMock``.

    Specced by hand on purpose. An unspecced mock grows ``safe_dry`` on demand,
    so a park that had stopped dry-purging would still look exactly like one
    that did — ``SUBAGENT_RULES`` §3's first shape, and the one already caught
    once on this code path.

    ``safe_dry`` and ``safe_off`` are counted separately because the whole
    question these doubles exist to answer is *which of the two* a park reached.
    """

    is_connected = True
    #: What the driver reports it actually held the duty at. Non-zero, because
    #: zero is the firmware's auto-shutoff rather than its dry end.
    last_safe_dry_duty = 0.05
    last_safe_dry_error = ""

    def __init__(self):
        self.dried = 0
        self.zeroed = 0

    def safe_dry(self) -> None:
        self.dried += 1

    def safe_off(self) -> None:
        self.zeroed += 1


class _Manager:
    def __init__(self, syringe=None, **instruments):
        self._instruments = {"syringe": syringe, **instruments}

    def get(self, name: str):
        if name in self._instruments:
            return self._instruments[name]
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


# ── The purge duration belongs to the device ─────────────────────────────────
#
# What used to live here was a *boundary*: an AST sweep proving that the safety
# paths did not opt into the dry purge while these two exits did. **Operator
# ruling, 2026-08-24 dissolved that boundary** — every park purges dry now, the
# E-Stop included, on the grounds that dry gas carries very little volatile
# species. With no boundary there is nothing for a call-site test to police: the
# end state is chosen inside ``safe_park``, where no reading of *our* call sites
# can observe it. Those tests are gone, replaced by behavioural ones over our
# two paths further down.
#
# What survives is the other constraint, which never depended on the boundary:
# **the purge window is the firmware's and must not be encoded host-side.** That
# one is still a claim about code that does not exist — no literal, no timer, no
# duration argument — so reading the source is still the only way to make it
# trip.

_PARK_FUNCS = {"safe_park", "safe_park_async"}

#: The two park call sites this file owns: the Safe Exit button's worker and the
#: window's exit park. Not a policy set any more — every park behaves alike —
#: simply the two places our code enters :func:`softae.core.safe_park.safe_park`.
OUR_PARK_SITES = {
    "src/softae/gui/main_window.py": "_safe_park_on_exit",
    "src/softae/gui/widgets/safe_exit.py": "run",
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


class TestThePurgeWindowIsTheDevices:
    """The one constraint the 2026-08-24 ruling left standing."""

    def test_no_orderly_exit_encodes_the_purge_duration(self):
        """The window belongs to the device, and to no host-side constant.

        The Trinket's ``ctrl_timeout`` decides when the valves close. A host
        timer that re-zeroed after N seconds would agree with it today and
        silently truncate the purge the moment ``ctrl_timeout`` is raised — the
        wrong value wearing the safe value's clothes. So neither exit site may
        name a duration at all, whether as a literal or as an argument.

        Unchanged by the ruling that dissolved the opt-in boundary above: it was
        never about *which* paths purge dry, only about who owns the clock.
        """
        import ast

        for rel_path, func_name in sorted(OUR_PARK_SITES.items()):
            sites = [(f, line, call) for f, line, call in _park_calls(rel_path)
                     if f == func_name]
            # Anti-vacuity, and it used to be supplied by the sibling test that
            # the ruling removed: a renamed or moved call must fail here rather
            # than quietly reduce the loop below to zero iterations.
            assert len(sites) == 1, (
                f"expected exactly one park call in {rel_path}::{func_name}, "
                f"found {len(sites)} — has the exit park moved?"
            )
            for func, line, call in sites:
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


# ── Both of our exits reach the dry purge ────────────────────────────────────
#
# Behavioural, and deliberately so. The predecessor tests read our call sites
# for ``rh_dry_purge=True``; after the 2026-08-24 ruling no call site chooses
# anything, so a reading of the source can no longer tell a path that purges
# dry from one that does not. These run the **real** ``safe_park`` against a
# hand-written driver and ask which method it reached.


def _spy_on_the_real_park(monkeypatch) -> list:
    """Let the genuine park run, and collect the results it produced.

    Patching in a *fake* park would only re-assert that our call sites call
    something; the guarantee at issue is what the humidifier is left doing, and
    only the real ``safe_park`` decides that.
    """
    from softae.core.safe_park import safe_park as real_safe_park

    results: list = []

    def spy(manager, **kwargs):
        result = real_safe_park(manager, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr("softae.core.safe_park.safe_park", spy)
    return results


def _dismiss_the_park_warning(monkeypatch) -> None:
    """Answer *Close* to the partial-park dialog, without a human.

    Not decoration. A park that failed to dry-purge lands in ``errors``, and
    ``SafeExitButton._on_done`` then opens a **modal** warning — so the very
    scenario these tests exist to catch would hang the suite rather than fail
    it. Found by mutating the park to zero instead of dry: the run blocked for
    ten minutes on a dialog nobody could see.
    """
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Close))


def _assert_dry_purged(rh: _RH, result) -> None:
    """The chamber was left purging dry, on the driver *and* in the report."""
    from softae.core.safe_park import DRY_PURGE_COMMANDED, RH_DEADMAN_S

    assert rh.dried == 1, "the park never asked the humidifier for a dry purge"
    assert rh.zeroed == 0, (
        "the park zeroed the humidifier — duty 0 is the firmware's auto-shutoff, "
        "so both PSVs close and room air comes back into the chamber"
    )
    # And it is *reported* as the deliberate standing command it is, under
    # ``commanded`` rather than as a failure. Built from the module's own
    # constant, so a rewording of the operator text propagates instead of
    # breaking this.
    assert DRY_PURGE_COMMANDED.format(duty=rh.last_safe_dry_duty,
                                      deadman=RH_DEADMAN_S) in result.commanded


class TestBothExitsLeaveTheChamberPurgingDry:
    """Safe Exit and the window's close are the same act with two buttons.

    They must not leave the chamber in two different humidity states — which is
    the half of the old boundary that survived the ruling, restated as
    behaviour. Neither test names a duration; the deadman is the device's.
    """

    def test_the_safe_exit_button_purges_dry(self, qtbot, monkeypatch):
        from softae.gui.widgets import safe_exit as mod

        rh = _RH()
        results = _spy_on_the_real_park(monkeypatch)
        _dismiss_the_park_warning(monkeypatch)

        btn = mod.SafeExitButton(_Manager(_Syringe(up=True), rh_controller=rh))
        qtbot.addWidget(btn)
        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert len(results) == 1
        _assert_dry_purged(rh, results[0])

    def test_the_windows_exit_park_purges_dry(self, monkeypatch):
        """Driven through the real method, with only the attributes it reads.

        Building a whole ``MainWindow`` would buy nothing here and drag the
        window's construction into a test about the humidifier.
        """
        from types import SimpleNamespace

        from softae.gui.main_window import MainWindow

        rh = _RH()
        results = _spy_on_the_real_park(monkeypatch)

        MainWindow._safe_park_on_exit(
            SimpleNamespace(_manager=_Manager(_Syringe(up=True),
                                              rh_controller=rh)))

        assert len(results) == 1, (
            "the exit park never reached safe_park — note it swallows every "
            "exception, so a broken call site is silent here"
        )
        _assert_dry_purged(rh, results[0])

    def test_the_head_decision_still_reaches_the_park(self, qtbot, monkeypatch):
        """The control on both tests above.

        They would pass with the head choice dropped on the floor, so the one
        thing our call sites genuinely still decide is pinned separately. This
        is what shows the reader that the humidity end state being caller-free
        is a *fact about the humidifier*, not this file having stopped looking
        at arguments.
        """
        from softae.gui.widgets import safe_exit as mod

        syr = _Syringe(up=False)
        rh = _RH()
        results = _spy_on_the_real_park(monkeypatch)
        _dismiss_the_park_warning(monkeypatch)

        btn = mod.SafeExitButton(_Manager(syr, rh_controller=rh))
        qtbot.addWidget(btn)
        monkeypatch.setattr(btn, "_ask_head_choice", lambda: False)
        with qtbot.waitSignal(btn.exit_requested, timeout=3000):
            btn.click()

        assert syr.retracted == 0
        assert any("left lowered" in a for a in results[0].actions)
        # ...and the chamber is dry either way. The two decisions are independent.
        _assert_dry_purged(rh, results[0])
