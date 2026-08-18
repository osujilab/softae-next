"""Headless park-on-exit and an honest exit code (halt_and_park Priority 1).

Four claims, each one a failure that was reachable before:

1. **A signal parks the rig.** There was no ``signal.signal`` anywhere in
   ``src/``, so ``SIGTERM`` killed the process outright and Ctrl-C printed a
   sentence — with the heater at setpoint and the lamp on either way.
2. **The park happens before the instruments disconnect.** ``safe_park`` skips
   anything that is not ``is_connected``, so the teardown that ran first made
   recovery structurally impossible rather than merely late.
3. **A parked campaign exits non-zero.** ``_park`` leaves the loop ``STOPPED``,
   the same state a clean stop leaves, and the CLI returned 0 for both — so a
   cron wrapper could not tell convergence from a 3 a.m. hard-fault park.
4. **A hard kill is caught at the next launch**, keyed on the unfinished run
   row. Not on the run lock, which unlinks itself when read.

No real signals are sent: the handler is fetched with ``signal.getsignal`` and
called, which is what the OS does to it and is safe to do under a test runner.
"""

from __future__ import annotations

import signal
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from softae.core import shutdown
from softae.core.autonomous_wiring import CampaignResult, CampaignSpec
from softae.core.data_store import DataStore
from softae.core.safe_park import SafeParkResult
from softae.core.shutdown import (
    ParkGuard,
    active_park_guard,
    detect_unfinished_runs,
    install_signal_park,
    park_on_shutdown,
    recover_from_unclean_shutdown,
    shutdown_signals,
)
from softae.tools import campaign as cli


def _manager(connected: bool = True) -> MagicMock:
    """A manager whose instruments answer ``is_connected`` like the real ones."""
    insts = {}
    for name in ("syringe", "temp_controller", "lamp"):
        inst = MagicMock()
        inst.is_connected = connected
        insts[name] = inst
    mgr = MagicMock()
    mgr.get.side_effect = lambda n: insts[n]
    mgr._insts = insts
    return mgr


@pytest.fixture
def spy_park(monkeypatch):
    """Record every park the shutdown module performs, with rig connectivity.

    Connectivity is captured *at park time* because that is the whole ordering
    claim: a park issued after ``disconnect_all`` records skips and moves nothing.
    """
    calls: list[dict] = []

    def _park(manager, *, reason="", **kwargs):
        try:
            connected = bool(getattr(manager.get("syringe"), "is_connected", False))
        except Exception:
            connected = False
        calls.append({"reason": reason, "connected": connected,
                      "manager": manager, "kwargs": kwargs})
        return SafeParkResult(commanded=["spied park"])

    monkeypatch.setattr(shutdown, "safe_park", _park)
    return calls


# ── ParkGuard: one park per shutdown ─────────────────────────────────────────

class TestParkGuard:
    def test_the_first_caller_parks(self, spy_park):
        guard = ParkGuard(_manager())
        result = guard.park("reason one")

        assert result is not None and result.ok
        assert guard.parked is True
        assert [c["reason"] for c in spy_park] == ["reason one"]

    def test_a_second_caller_is_told_it_already_happened(self, spy_park):
        """Three callers ask unconditionally; only one park may reach the rig."""
        guard = ParkGuard(_manager())
        guard.park("first")

        assert guard.park("second") is None
        assert len(spy_park) == 1
        assert guard.reason == "first"

    def test_the_claim_is_taken_before_the_first_driver_write(self, monkeypatch):
        """What makes a second Ctrl-C mid-park harmless: the flag is already set.

        Re-entering ``park`` from inside the park itself is exactly what a nested
        signal handler does, and it must find the guard claimed rather than add a
        second write sequence to a serial line that is mid-sequence.
        """
        seen: list[bool] = []
        reentered: list[object] = []
        guard = ParkGuard(_manager())

        def _park(manager, *, reason="", **kwargs):
            seen.append(guard.in_progress)
            reentered.append(guard.park("re-entrant"))
            return SafeParkResult()

        monkeypatch.setattr(shutdown, "safe_park", _park)
        guard.park("first")

        assert seen == [True]           # claimed before safe_park was entered
        assert reentered == [None]      # the nested ask was declined
        assert guard.in_progress is False

    def test_a_raising_safe_park_is_reported_not_propagated(self, monkeypatch):
        """``safe_park`` says it never raises; shutdown is a bad place to find out."""
        def _boom(*_a, **_k):
            raise RuntimeError("VISA is gone")

        monkeypatch.setattr(shutdown, "safe_park", _boom)
        guard = ParkGuard(_manager())
        result = guard.park("crash")

        assert result is not None
        assert result.ok is False
        assert "VISA is gone" in result.errors[0]
        assert "INCOMPLETE" in guard.describe()

    def test_describe_says_so_when_nothing_was_parked(self):
        assert "NOT parked" in ParkGuard(_manager()).describe()

    def test_park_on_shutdown_deduplicates_against_the_active_guard(self, spy_park):
        """Library code parks unconditionally; the guard decides if it lands."""
        mgr = _manager()
        guard = ParkGuard(mgr)

        with install_signal_park(guard, signals=()):
            assert park_on_shutdown(mgr, "first") is not None
            assert park_on_shutdown(mgr, "second") is None

        assert len(spy_park) == 1

    def test_a_different_rig_is_not_covered_by_someone_elses_guard(self, spy_park):
        guard = ParkGuard(_manager())
        other = _manager()

        with install_signal_park(guard, signals=()):
            assert park_on_shutdown(other, "other rig") is not None

        assert [c["manager"] for c in spy_park] == [other]

    def test_without_a_guard_it_still_parks(self, spy_park):
        """The wiring must be correct when nobody installed anything."""
        assert active_park_guard() is None
        assert park_on_shutdown(_manager(), "no guard here") is not None
        assert len(spy_park) == 1


# ── Signal handling ──────────────────────────────────────────────────────────

class TestSignalPark:
    def test_sigint_and_sigterm_are_covered_on_this_platform(self):
        names = {signal.Signals(s).name for s in shutdown_signals()}
        assert "SIGINT" in names
        assert "SIGTERM" in names
        if hasattr(signal, "SIGBREAK"):     # Windows Ctrl-Break
            assert "SIGBREAK" in names

    def test_the_handler_parks_then_raises_to_unwind(self, spy_park):
        """Raising is load-bearing: the exit must run the campaign's teardown."""
        guard = ParkGuard(_manager())

        with install_signal_park(guard, signals=(signal.SIGINT,)) as installed:
            assert installed == (signal.SIGINT,)
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)

        assert len(spy_park) == 1
        assert "SIGINT" in spy_park[0]["reason"]

    def test_the_previous_handler_is_restored_on_exit(self):
        before = signal.getsignal(signal.SIGINT)
        with install_signal_park(ParkGuard(_manager()), signals=(signal.SIGINT,)):
            assert signal.getsignal(signal.SIGINT) is not before
        assert signal.getsignal(signal.SIGINT) is before

    def test_a_second_signal_during_the_park_is_declined_not_obeyed(self,
                                                                    monkeypatch):
        """The operator hammering Ctrl-C must not corrupt an in-flight park."""
        nested: list[object] = []
        guard = ParkGuard(_manager())

        def _park(manager, *, reason="", **kwargs):
            handler = signal.getsignal(signal.SIGINT)
            # Python may run a handler *inside* a handler; this must simply
            # return, leaving the park that is already talking to the rig alone.
            nested.append(handler(signal.SIGINT, None))
            return SafeParkResult()

        monkeypatch.setattr(shutdown, "safe_park", _park)

        with install_signal_park(guard, signals=(signal.SIGINT,)):
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)

        assert nested == [None]         # the nested signal raised nothing

    def test_a_second_signal_after_the_park_reaches_the_default_handler(self,
                                                                        spy_park):
        """A handler that swallowed every later signal would trap the operator.

        Once the park is done there is nothing left to protect, so the handler
        uninstalls itself and a further Ctrl-C takes the ordinary route out.
        """
        before = signal.getsignal(signal.SIGINT)

        with install_signal_park(ParkGuard(_manager()), signals=(signal.SIGINT,)):
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            assert signal.getsignal(signal.SIGINT) is before

        assert len(spy_park) == 1

    def test_off_the_main_thread_the_guard_installs_without_handlers(self):
        """``signal.signal`` is main-thread only — a worker gets the guard anyway."""
        seen: dict = {}

        def _run():
            guard = ParkGuard(_manager())
            try:
                with install_signal_park(guard) as installed:
                    seen["installed"] = installed
                    seen["active"] = active_park_guard() is guard
            except Exception as exc:       # pragma: no cover - the bug it guards
                seen["error"] = exc

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join(timeout=10)

        assert "error" not in seen
        assert seen["installed"] == ()
        assert seen["active"] is True


# ── Next-launch recovery, keyed on the unfinished run row ────────────────────

class TestUncleanShutdownRecovery:
    def test_an_unfinished_row_is_what_marks_a_hard_kill(self, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("killed_campaign", mode="autonomous")

        unfinished = detect_unfinished_runs(store)

        assert unfinished is not None
        assert unfinished.run_ids == (run_id,)
        assert "lowered over an electrode" in unfinished.describe()
        store.close()

    def test_a_clean_history_detects_nothing(self, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        store.finish_run(store.start_run("clean", mode="autonomous"), "done")

        assert detect_unfinished_runs(store) is None
        store.close()

    def test_recovery_alerts_marks_and_parks(self, tmp_path: Path, spy_park):
        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("killed_campaign", mode="autonomous")
        lines: list[str] = []

        result = recover_from_unclean_shutdown(_manager(), store, report=lines.append)

        assert result is not None and result.ok
        assert spy_park[0]["reason"] == shutdown.RECOVERY_PARK_REASON
        assert spy_park[0]["connected"] is True
        alerts = store.query_alerts()
        assert [a["kind"] for a in alerts] == ["unclean_shutdown"]
        assert alerts[0]["details"]["head_position_unknown"] is True
        assert run_id in alerts[0]["details"]["runs"]
        assert any("did not finish cleanly" in line for line in lines)
        store.close()

    def test_it_is_reported_once_not_at_every_launch(self, tmp_path: Path,
                                                     spy_park):
        store = DataStore(tmp_path / "proj")
        store.start_run("killed_campaign", mode="autonomous")

        recover_from_unclean_shutdown(_manager(), store)

        assert detect_unfinished_runs(store) is None    # marked `interrupted`
        assert recover_from_unclean_shutdown(_manager(), store) is None
        assert len(spy_park) == 1
        store.close()

    def test_the_recovery_park_does_not_consume_the_shutdown_guards_claim(
            self, tmp_path: Path, spy_park):
        """The recovery park belongs to the *previous* run.

        Spending the guard's single claim on it would leave the run about to
        start with no park left for its own shutdown — the exact failure this
        priority exists to close, one launch later.
        """
        store = DataStore(tmp_path / "proj")
        store.start_run("killed_campaign", mode="autonomous")
        mgr = _manager()
        guard = ParkGuard(mgr)

        with install_signal_park(guard, signals=()):
            recover_from_unclean_shutdown(mgr, store)
            assert guard.parked is False
            assert park_on_shutdown(mgr, "this run's own shutdown") is not None

        assert len(spy_park) == 2
        store.close()

    def test_a_broken_store_never_stops_a_campaign_starting(self):
        store = MagicMock()
        store.unfinished_runs.side_effect = RuntimeError("db is locked")

        assert detect_unfinished_runs(store) is None
        assert detect_unfinished_runs(None) is None


# ── The CLI: ordering and the exit code ──────────────────────────────────────

DEMO = """
name = "shutdown_test"
channels = [21]
pcb_name = "SoftAE_EIS_4Stripe"
budget = 1
time_scale = 0.0

[parameter_space.vol_p0]
type = "float"
low = 5.0
high = 30.0
"""


def _spec_file(tmp_path: Path) -> str:
    path = tmp_path / "spec.toml"
    path.write_text(DEMO, encoding="utf-8")
    return str(path)


def _result(**over) -> CampaignResult:
    base = dict(run_id="r1", best_params=None, best_objective=None, n_trials=3,
                final_state="STOPPED", converged=False, history=[],
                park_reason=None)
    base.update(over)
    return CampaignResult(**base)


def _run_cli(tmp_path: Path, monkeypatch, fake_campaign) -> int:
    monkeypatch.setattr(
        "softae.core.autonomous_wiring.run_autonomous_campaign", fake_campaign)
    return cli.main(["run", _spec_file(tmp_path), "--mock", "--yes", "--head-up",
                     "--project", str(tmp_path / "proj")])


class TestExitCode:
    """Zero means *ended on purpose*. A park is not that."""

    @pytest.mark.parametrize("state,park_reason,expected", [
        ("CONVERGED", None, cli.EXIT_OK),
        ("STOPPED", None, cli.EXIT_OK),
        ("STOPPED", "reservoir at its hard stop", cli.EXIT_FAILED),
        ("CONVERGED", "gate timed out", cli.EXIT_FAILED),
        ("ERROR", None, cli.EXIT_FAILED),
    ])
    def test_the_exit_code_reports_whether_the_run_ended_on_purpose(
            self, tmp_path, monkeypatch, capsys, state, park_reason, expected):
        async def _fake(spec, **kwargs):
            return _result(final_state=state, park_reason=park_reason)

        assert _run_cli(tmp_path, monkeypatch, _fake) == expected

    def test_a_parked_campaign_says_why_on_the_terminal(self, tmp_path,
                                                        monkeypatch, capsys):
        async def _fake(spec, **kwargs):
            return _result(park_reason="reservoir at its hard stop")

        rc = _run_cli(tmp_path, monkeypatch, _fake)
        out = capsys.readouterr().out

        assert rc == cli.EXIT_FAILED
        assert "PARKED: reservoir at its hard stop" in out

    def test_park_reason_is_carried_on_the_result_the_cli_sees(self):
        """The loop always knew; until now only the event stream could see it."""
        assert _result().park_reason is None
        assert _result(park_reason="x").park_reason == "x"


class TestSignalOrdering:
    def test_sigint_parks_before_disconnect(self, tmp_path, monkeypatch,
                                            spy_park, capsys):
        """The ordering *is* the fix — a park after teardown parks nothing."""
        order: list[str] = []

        async def _fake(spec, *, manager, **kwargs):
            original = manager.disconnect_all

            async def _spy_disconnect():
                order.append("disconnect")
                return await original()

            manager.disconnect_all = _spy_disconnect
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler), "no SIGINT handler was installed"
            handler(signal.SIGINT, None)        # raises KeyboardInterrupt
            raise AssertionError("the handler must not return")

        def _record(manager, *, reason="", **kwargs):
            order.append("park")
            return SafeParkResult(commanded=["spied park"])

        monkeypatch.setattr(shutdown, "safe_park", _record)
        rc = _run_cli(tmp_path, monkeypatch, _fake)
        out = capsys.readouterr().out

        assert rc == cli.EXIT_FAILED
        assert order == ["park", "disconnect"]
        assert "PARKING" in out
        assert "resume with" in out

    def test_the_park_is_reported_in_the_interrupted_message(self, tmp_path,
                                                             monkeypatch,
                                                             spy_park, capsys):
        async def _fake(spec, *, manager, **kwargs):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

        rc = _run_cli(tmp_path, monkeypatch, _fake)
        out = capsys.readouterr().out

        assert rc == cli.EXIT_FAILED
        assert "Interrupted — parked" in out
        assert spy_park[0]["connected"] is True

    def test_an_unfinished_row_is_recovered_after_the_rig_connects(
            self, tmp_path, monkeypatch, spy_park, capsys):
        """Detection is early (so it is reported); the park waits for sessions."""
        store = DataStore(tmp_path / "proj")
        store.start_run("killed_campaign", mode="autonomous")
        store.close()

        async def _fake(spec, **kwargs):
            return _result(final_state="CONVERGED")

        rc = _run_cli(tmp_path, monkeypatch, _fake)
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        assert "PREVIOUS SESSION DID NOT FINISH" in out
        assert [c["reason"] for c in spy_park] == [shutdown.RECOVERY_PARK_REASON]
        assert spy_park[0]["connected"] is True


# ── The campaign's own catch-all ─────────────────────────────────────────────

SPACE = {"vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
         "vol_p1": {"type": "float", "low": 5.0, "high": 30.0}}


def _campaign_spec(**over) -> CampaignSpec:
    base = dict(name="shutdown_campaign", channels=(21, 22),
                pcb_name="SoftAE_EIS_4Stripe", parameter_space=SPACE,
                vol_params=("vol_p0", "vol_p1"), pump_ids=(0, 1),
                deadvols=(10.0, 30.0), time_scale=0.0, budget=6, seed=7)
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
async def connected():
    from softae.drivers.mock_factory import create_mock_manager

    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


@pytest.mark.asyncio
async def test_an_unexpected_exception_parks_the_rig(connected, tmp_path: Path,
                                                     spy_park):
    """A crash used to unwind straight into ``disconnect_all``, parking nothing."""
    from softae.core.autonomous_wiring import run_autonomous_campaign

    store = DataStore(tmp_path / "proj")

    def _boom(event):
        if event["type"] == "run_started":
            raise RuntimeError("the host is on fire")

    with pytest.raises(RuntimeError, match="on fire"):
        await run_autonomous_campaign(_campaign_spec(budget=1),
                                      manager=connected, data_store=store,
                                      on_event=_boom)

    assert len(spy_park) == 1
    assert spy_park[0]["connected"] is True
    assert "RuntimeError" in spy_park[0]["reason"]
    # The row is still closed, and after the park rather than before it.
    assert store.unfinished_runs() == []
    alerts = store.query_alerts()
    assert [a["kind"] for a in alerts] == ["park"]
    assert alerts[0]["severity"] == "critical"
    store.close()


@pytest.mark.asyncio
async def test_a_campaign_that_already_parked_does_not_park_twice(
        connected, tmp_path: Path, monkeypatch):
    """A campaign parks once: one fault, one safe_park, one CRITICAL alert."""
    import softae.core.autonomous_wiring as aw
    from softae.core.autonomous_wiring import run_autonomous_campaign

    parks: list[str] = []

    def _record(manager, *, reason="", **kwargs):
        parks.append(reason)
        return SafeParkResult(commanded=["spied park"])

    monkeypatch.setattr(aw, "safe_park", _record)          # the loop's park
    monkeypatch.setattr(shutdown, "safe_park", _record)    # the crash park
    # Nothing ever measures → consecutive failures → the loop parks.
    monkeypatch.setattr(aw, "eis_impedance_objective", lambda r, p: None)
    monkeypatch.setattr(aw, "eis_impedance_objective_for_channel", lambda r, c: None)

    store = DataStore(tmp_path / "proj")

    def _boom(event):
        # Raised after the loop has already parked, on the way out.
        if event["type"] == "run_finished":
            raise RuntimeError("the store went away")

    with pytest.raises(RuntimeError, match="store went away"):
        await run_autonomous_campaign(_campaign_spec(budget=8),
                                      manager=connected, data_store=store,
                                      on_event=_boom)

    assert len(parks) == 1
    store.close()


@pytest.mark.asyncio
async def test_a_parked_campaign_reports_its_reason_on_the_result(
        connected, tmp_path: Path, monkeypatch):
    """The field the exit code is computed from, on the real path."""
    import softae.core.autonomous_wiring as aw
    from softae.core.autonomous_wiring import run_autonomous_campaign

    monkeypatch.setattr(aw, "eis_impedance_objective", lambda r, p: None)
    monkeypatch.setattr(aw, "eis_impedance_objective_for_channel", lambda r, c: None)
    store = DataStore(tmp_path / "proj")

    result = await run_autonomous_campaign(_campaign_spec(budget=8),
                                           manager=connected, data_store=store)

    assert result.final_state == "STOPPED"      # indistinguishable from a clean stop
    assert result.park_reason                    # …except by this
    store.close()
