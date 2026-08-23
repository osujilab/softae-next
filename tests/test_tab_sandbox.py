"""Daemon-shutdown tests for the Sandbox tab (tab_sandbox.py)."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs.tab_sandbox import SandboxTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


@pytest.fixture
def tab(qapp, manager):
    widget = SandboxTab(manager)
    yield widget
    widget.close()


class _StubExecutor:
    """Executor stub whose abort() sets a threading.Event (the run's abort signal)."""

    def __init__(self) -> None:
        self.ev = threading.Event()

    def abort(self) -> None:
        self.ev.set()


def _spin_on_event(ev: threading.Event) -> threading.Thread:
    def run() -> None:
        while not ev.is_set():
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestDaemonShutdown:
    def test_sandbox_cleanup_aborts_running_thread(self, tab: SandboxTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        assert tab._run_thread.is_alive()
        tab.cleanup()
        assert ex.ev.is_set()
        assert not tab._run_thread.is_alive()

    def test_sandbox_cleanup_is_noop_when_idle(self, tab: SandboxTab):
        assert tab._executor is None
        assert tab._run_thread is None
        tab.cleanup()  # must not raise / block

    def test_sandbox_cleanup_is_idempotent(self, tab: SandboxTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        tab.cleanup()
        tab.cleanup()
        assert not tab._run_thread.is_alive()

    def test_sandbox_abort_run_signals_without_joining(self, tab: SandboxTab):
        ex = _StubExecutor()
        tab._executor = ex
        tab._run_thread = _spin_on_event(ex.ev)
        tab.abort_run()
        assert ex.ev.is_set()
        assert tab._run_thread.is_alive()  # signal-only: not joined
        tab.cleanup()  # teardown join


# ── Builder step-editor persistence (prefill/param clobber regression) ───────

def test_step_params_persist_across_selection_switch(tab):
    """Switching between steps must not overwrite a step's params.

    Regression for the reported glitch: bouncing selection between two steps of
    the same instrument but different methods corrupted the params of the step
    being selected (the method-combo signal fired mid-population and wrote a
    half-filled editor back onto the item).
    """
    from PySide6.QtCore import Qt

    role = Qt.ItemDataRole.UserRole
    cat = tab._task_catalog
    tab._insert_task(cat.get("single_drop_simul"), "loop")
    tab._insert_task(cat.get("precondition_flush"), "loop")
    a, b = tab._loop_root.child(0), tab._loop_root.child(1)

    params_a = dict(a.data(0, role)["params"])
    params_b = dict(b.data(0, role)["params"])
    assert params_a and params_b and params_a != params_b

    # Bounce the selection back and forth several times.
    for item in (a, b, a, b, a):
        tab._tree.setCurrentItem(item)

    assert a.data(0, role)["params"] == params_a
    assert b.data(0, role)["params"] == params_b
    assert a.data(0, role)["method"] == "single_drop_simul"
    assert b.data(0, role)["method"] == "precondition_flush"


def test_run_error_surfaced_in_status_and_preview(tab):
    """A workflow-level failure shows its detail (status truncated, preview full)."""
    tab._run_error = "Duplicate step name(s) ['head_descend'] — step names must be unique"
    tab._ui_done(1)   # exit_code 1 = failed
    assert "Duplicate step name" in tab._lbl_status.text()
    assert "Duplicate step name" in tab._txt_preview.toPlainText()


def test_run_success_shows_completed(tab):
    tab._ui_done(0)
    assert "Completed" in tab._lbl_status.text()


def test_step_error_surfaced_in_preview(tab):
    tab._ui_step_error("deposit_ch1", 2, 5, "boom: pump offline")
    assert "boom: pump offline" in tab._txt_preview.toPlainText()


def test_build_workflow_uniquifies_duplicate_names(tab):
    """A renamed-to-collide step must not produce a duplicate-named workflow.

    Duplicate names within a phase break execution (the DAG keys on names and
    would surface a false 'dependency cycle'); the builder resolves them.
    """
    from PySide6.QtCore import Qt

    role = Qt.ItemDataRole.UserRole
    a = tab._add_step("setup")
    b = tab._add_step("setup")
    for item in (a, b):
        data = dict(item.data(0, role))
        data["name"] = "dup"
        item.setData(0, role, data)

    wf = tab._build_workflow()
    names = [s.name for s in wf.setup]
    assert len(names) == len(set(names))          # all unique
    assert "dup" in names and "dup_2" in names


def test_add_step_reused_index_stays_unique(tab):
    """Remove-then-add can reuse an index; the generated name must stay unique."""
    tab._add_step("setup")               # setup_step_1
    s2 = tab._add_step("setup")          # setup_step_2
    tab._add_step("setup")               # setup_step_3
    tab._tree.setCurrentItem(s2)
    tab._remove_step()                   # drop setup_step_2 → children 1, 3
    tab._add_step("setup")               # childCount=2 → base 'setup_step_3' collides
    root = tab._setup_root
    names = [root.child(i).text(0) for i in range(root.childCount())]
    assert len(names) == len(set(names))


def test_sandbox_run_suspends_its_own_claim_while_the_executor_is_held(tab):
    """A held Sandbox run hands the rig back, exactly as an HT run does.

    Sandbox drives the same ``WorkflowExecutor`` through the same hold loops, so
    the operator's pause ruling has to reach it too — and it is the run kind most
    likely to be forgotten, since it claimed nothing at all until recently.

    The assertion is on the *registry*, not on the wiring: a spy that only
    recorded "something was assigned to on_pause_hold" would pass against a
    handle pointing at the wrong owner, which does not raise — it registers a
    second entry that never drains.
    """
    from softae.core.rig_activity import PURGE_INSTRUMENTS, RigActivity
    from softae.gui.rig_claim import RigRunClaim
    from softae.workflows.workflow_model import Workflow

    activity = RigActivity()
    owner = "sandbox:bench"
    activity.acquire(owner, None)
    observed: dict[str, object] = {}

    class _HeldExecutor:
        on_pause_hold = None

        async def run(self, wf):
            observed["driving"] = activity.conflicts(PURGE_INSTRUMENTS)
            self.on_pause_hold(True)
            observed["held"] = activity.conflicts(PURGE_INSTRUMENTS)
            observed["held_owner"] = activity.suspended_conflict(PURGE_INSTRUMENTS)
            self.on_pause_hold(False)
            observed["resumed"] = activity.conflicts(PURGE_INSTRUMENTS)

    # The tab is windowless, so `rig_run` yields the null handle; point it at a
    # real registry the way a hosted tab's `MainWindow.rig_run` would.
    monkey = RigRunClaim(activity, owner)
    import softae.gui.tabs.tab_sandbox as mod

    real_rig_run = mod.rig_run
    from contextlib import contextmanager

    @contextmanager
    def _hosted(host, owner_str, **kw):
        with real_rig_run(host, owner_str, **kw):
            yield monkey

    mod.rig_run = _hosted
    try:
        tab._executor = _HeldExecutor()
        t = threading.Thread(target=tab._run_thread_fn,
                             args=(Workflow(name="bench"),), daemon=True)
        t.start()
        t.join(timeout=20.0)
        assert not t.is_alive()
    finally:
        mod.rig_run = real_rig_run

    assert observed["driving"] == owner        # driving: manual refused
    assert observed["held"] is None            # held: manual permitted
    assert observed["held_owner"] == owner     # …and still not an idle rig
    assert observed["resumed"] == owner        # driving again


def test_user_param_edit_persists_after_switch(tab):
    """A user's param edit on one step survives selecting another and returning."""
    import json

    from PySide6.QtCore import Qt

    role = Qt.ItemDataRole.UserRole
    cat = tab._task_catalog
    tab._insert_task(cat.get("single_drop_simul"), "loop")
    tab._insert_task(cat.get("star_mix"), "loop")
    a, b = tab._loop_root.child(0), tab._loop_root.child(1)

    # Edit step A's params through the editor (as the user would).
    tab._tree.setCurrentItem(a)
    tab._edit_params.setText(json.dumps({"x": 12.5, "y": 7.0}))
    # Switch away and back.
    tab._tree.setCurrentItem(b)
    tab._tree.setCurrentItem(a)
    assert a.data(0, role)["params"] == {"x": 12.5, "y": 7.0}
