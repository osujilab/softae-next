"""The ``softae-env`` CLI surface — the measurement-free humidity hold.

``hold --execute`` actuates the humidifier unattended and **nothing in the
shipped configuration refuses it**: ``validate_rh_setpoint`` is a cap with no
floor, and ``assert_hardware_armed`` covers ``("stage", "syringe", "piezo")``
only, so no interlock here has anything to say about a humidifier. The
confirmation prompt and the ``--execute`` gate are the only barriers, and these
tests are what keep them ones.

Synthetic throughout: no serial port is opened, no Trinket is driven, and every
``--execute`` path either runs against ``--mock`` drivers or against the fakes
below.
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock

import pytest

from softae.core.run_lock import RunLock, RunLockHeld
from softae.drivers.contracts import RH_FAULT, RHHoldVerdict, rh_watchdog_config
from softae.errors import SafetyError
from softae.tools import env_hold
from softae.tools.env_hold import (
    CLAIM_KIND,
    EXIT_BUSY,
    EXIT_DECLINED,
    EXIT_FAILED,
    EXIT_OK,
    HoldSession,
    _cmd_hold,
    _run_hold,
    build_parser,
    confirm_hold,
    describe_duration,
    duration_s,
    hold_loop,
    install_handlers,
    main,
)

# ── Fixtures and fakes ───────────────────────────────────────────────────────

@pytest.fixture()
def project(monkeypatch, tmp_path):
    """Point the default store at a temporary project directory."""
    from softae.config import loader

    monkeypatch.setattr(loader, "data_project_dir", lambda: str(tmp_path / "real"))
    return tmp_path


@pytest.fixture()
def no_hardware(monkeypatch):
    """Make *any* attempt to build a driver manager an immediate, loud failure.

    The dry-run and declined paths are defined by what they do **not** open, and
    a test that only checks the exit code would pass against a tool that opened
    every port and then returned 0.
    """
    def _boom(*_a, **_k):
        raise AssertionError("create_manager must not be called on this path")

    monkeypatch.setattr("softae.drivers.factory.create_manager", _boom)


@pytest.fixture()
def no_claim(monkeypatch):
    """Make any rig claim an immediate failure, for the paths that take none."""
    def _boom(*_a, **_k):
        raise AssertionError("the rig must not be claimed on this path")

    monkeypatch.setattr("softae.core.rig_session.acquire_run_lock", _boom)


@pytest.fixture()
def rig_free(monkeypatch):
    """Report the rig as unheld.

    The lock's scope is **this machine**, not the temporary project directory, so
    an unpatched peek reads whatever really holds the rig right now — the
    operator's GUI, very possibly. Every test of a non-``--mock`` hold has to say
    which answer it is testing against.
    """
    monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *_a, **_k: None)


class FakeStore:
    """Enough ``DataStore`` for the run row and the alert rows, and no more."""

    def __init__(self, project_dir: str = "fake") -> None:
        self.project_dir = project_dir
        self.started: list[tuple] = []
        self.status: str | None = None
        self.alerts: list[tuple] = []
        self.closed = False

    def start_run(self, name, **kw) -> str:
        self.started.append((name, kw))
        return "20260819T000000Z_env_hold"

    def finish_run(self, run_id, status) -> None:
        # First writer wins, exactly as `run_finalizer`'s one-shot guarantees.
        if self.status is None:
            self.status = status

    def record_alert(self, kind, message, **kw) -> int:
        self.alerts.append((kind, message, kw))
        return len(self.alerts)

    def close(self) -> None:
        self.closed = True


class FakeRH:
    def __init__(self) -> None:
        self.setpoint: float | None = None
        self.running = False
        self.safe_off_calls = 0
        self.last_safe_off_error = ""
        self.set_setpoint_error: Exception | None = None

    def set_setpoint(self, val: float) -> None:
        if self.set_setpoint_error is not None:
            raise self.set_setpoint_error
        self.setpoint = float(val)

    def status(self) -> dict:
        return {"running": self.running}

    def start(self) -> None:
        self.running = True

    def get_TH(self) -> tuple[float, float]:
        return 22.5, 44.0

    def safe_off(self) -> None:
        self.safe_off_calls += 1
        self.running = False
        self.setpoint = 0.0


class FakeManager:
    def __init__(self, rh: FakeRH) -> None:
        self._rh = rh
        self.connected = False
        self.disconnected = False

    async def connect_all(self) -> None:
        self.connected = True

    async def disconnect_all(self) -> None:
        self.disconnected = True

    def get(self, _name: str):
        return self._rh


class FakeClock:
    """Monotonic time that only moves when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


def _args(*argv):
    return build_parser().parse_args(list(argv))


def _drive(args, rh=None, store=None, *, seconds=0.0, watch_cls=None,
           thresholds=None):
    """Run ``_run_hold`` against fakes. Returns ``(exit_code, rh, store)``."""
    rh = rh or FakeRH()
    store = store or FakeStore()
    finalize = env_hold.run_finalizer(store, "RUNID")
    watch_cls = watch_cls or RecordingWatch
    code = _run_hold(args, FakeManager(rh), store, "RUNID", finalize, seconds,
                     RHHoldWatch=watch_cls,
                     thresholds=thresholds or rh_watchdog_config(),
                     driver_errors=(SafetyError,))
    return code, rh, store


class RecordingWatch:
    """An ``RHHoldWatch`` stand-in that records how it was built and sampled."""

    last: "RecordingWatch | None" = None

    def __init__(self, reader, setpoint_pct, **kw) -> None:
        self.reader = reader
        self.setpoint_pct = setpoint_pct
        self.kwargs = kw
        self.samples = 0
        self.temperature_C = 22.5
        self.verdict = RHHoldVerdict("converging", setpoint_pct, pv_pct=44.0)
        RecordingWatch.last = self

    def sample(self) -> None:
        self.samples += 1


# ── The dry run ──────────────────────────────────────────────────────────────

class TestDryRun:
    def test_a_hold_without_execute_opens_nothing(self, project, no_hardware):
        assert main(["hold", "--rh", "45", "--duration-h", "4"]) == EXIT_OK

    def test_a_dry_run_starts_no_run_row(self, project, no_hardware, monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError("no run row on a dry run")

        monkeypatch.setattr(env_hold, "_open_store", _boom)
        assert main(["hold", "--rh", "45"]) == EXIT_OK

    def test_a_dry_run_takes_no_rig_claim(self, project, no_hardware, no_claim):
        assert main(["hold", "--rh", "45"]) == EXIT_OK

    def test_a_dry_run_says_it_opened_nothing(self, project, no_hardware, capsys):
        main(["hold", "--rh", "45", "--duration-h", "4"])
        out = capsys.readouterr().out
        assert "Dry run" in out
        assert "--execute" in out
        assert "45 %RH" in out


# ── The confirmation ─────────────────────────────────────────────────────────

class TestConfirmation:
    def test_a_non_tty_without_yes_declines(self, project, no_hardware, monkeypatch):
        """``input()`` on a non-TTY raises ``EOFError``. An unattended
        invocation that meant to run had ``--yes`` to say so."""
        def _eof(_prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        assert main(["hold", "--rh", "45", "--execute"]) == EXIT_DECLINED

    def test_the_wrong_word_declines(self, project, no_hardware, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _p: "y")
        assert main(["hold", "--rh", "45", "--execute"]) == EXIT_DECLINED

    def test_yes_proceeds_and_says_so(self, capsys):
        assert confirm_hold(45.0, 3600.0, assume_yes=True) is True
        assert "--yes given" in capsys.readouterr().out

    def test_the_confirmation_names_the_setpoint_and_says_unattended(self, capsys):
        confirm_hold(45.0, 3600.0, reader=lambda _p: "yes")
        out = capsys.readouterr().out
        assert "45 %RH" in out
        assert "UNATTENDED" in out
        assert "1h 00m" in out

    def test_the_literal_word_is_required(self):
        assert confirm_hold(45.0, None, reader=lambda _p: "yes") is True
        assert confirm_hold(45.0, None, reader=lambda _p: "yep") is False


# ── The argument surface ─────────────────────────────────────────────────────

class TestArguments:
    def test_rh_is_required(self):
        with pytest.raises(SystemExit):
            _args("hold", "--execute")

    def test_the_two_duration_flags_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            _args("hold", "--rh", "45", "--duration-s", "60", "--duration-h", "1")

    def test_both_durations_absent_means_until_signal(self):
        assert duration_s(_args("hold", "--rh", "45")) is None
        assert describe_duration(None) == "until interrupted"

    def test_plan_opens_nothing(self, project, no_hardware, no_claim, capsys):
        assert main(["plan", "--rh", "45", "--duration-h", "4"]) == EXIT_OK
        assert "4h 00m" in capsys.readouterr().out


# ── The rig claim ────────────────────────────────────────────────────────────

class TestRigClaim:
    def test_a_mock_hold_takes_no_claim(self, project, no_claim, monkeypatch,
                                        capsys):
        """``session_is_simulated`` short-circuits, so a dry run cannot become an
        outage for a real one — and the busy peek is skipped for the same reason:
        a hold that claims nothing must not be refused over a lock it never wants.
        """
        def _boom(*_a, **_k):
            raise AssertionError("a --mock hold must not consult the rig lock")

        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", _boom)
        code = main(["hold", "--rh", "45", "--duration-s", "0",
                     "--execute", "--yes", "--mock"])
        assert code == EXIT_OK

    def test_a_foreign_holder_refuses_with_exit_busy(self, project, monkeypatch,
                                                    capsys):
        """And writes **no run row**. An `aborted` row is byte-for-byte what a
        hold that started and failed looks like; a hold refused before it touched
        anything must not leave one."""
        rh = FakeRH()
        manager = FakeManager(rh)

        def _no_store(*_a, **_k):
            raise AssertionError("the store must not be opened on a busy rig")

        monkeypatch.setattr("softae.drivers.factory.create_manager",
                            lambda **_k: manager)
        monkeypatch.setattr(env_hold, "_open_store", _no_store)

        holder = RunLock(pid=99999, what="gui:desktop", started_at="2026-08-19T00:00:00Z")
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock",
                            lambda *_a, **_k: holder)

        code = _cmd_hold(_args("hold", "--rh", "45", "--execute", "--yes"))

        assert code == EXIT_BUSY
        assert "would drive the same rig" in capsys.readouterr().out
        assert manager.connected is False

    def test_a_holder_arriving_after_the_peek_aborts_the_row(self, project,
                                                             monkeypatch, capsys):
        """The residual race the peek deliberately does not close.

        A holder that appears between the peek and the claim still raises
        `RunLockHeld` with the row already open, and that path keeps its `aborted`
        finalization. The peek makes this rare, not impossible; `acquire_run_lock`'s
        exclusive create is what keeps the claim itself correct.
        """
        rh = FakeRH()
        manager = FakeManager(rh)
        store = FakeStore()
        monkeypatch.setattr("softae.drivers.factory.create_manager",
                            lambda **_k: manager)
        monkeypatch.setattr(env_hold, "_open_store", lambda _a: (store, "p"))
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock",
                            lambda *_a, **_k: None)      # free at the peek …

        holder = RunLock(pid=99999, what="gui:desktop", started_at="2026-08-19T00:00:00Z")
        def _held(*_a, **_k):
            raise RunLockHeld(holder)                     # … and taken by the claim

        monkeypatch.setattr("softae.core.rig_session.claim_rig_session", _held)

        code = _cmd_hold(_args("hold", "--rh", "45", "--execute", "--yes"))

        assert code == EXIT_BUSY
        assert "would drive the same rig" in capsys.readouterr().out
        assert manager.connected is False
        assert store.started, "the row was already open when the holder arrived"
        assert store.status == "aborted"
        assert store.closed is True

    def test_the_claim_names_the_run_id(self, project, rig_free, monkeypatch):
        rh = FakeRH()
        store = FakeStore()
        seen: dict = {}
        monkeypatch.setattr("softae.drivers.factory.create_manager",
                            lambda **_k: FakeManager(rh))
        monkeypatch.setattr(env_hold, "_open_store", lambda _a: (store, "p"))
        monkeypatch.setattr(env_hold, "_run_hold",
                            lambda *a, **k: EXIT_OK)

        def _claim(_manager, *, what, **_k):
            seen["what"] = what
            return None

        monkeypatch.setattr("softae.core.rig_session.claim_rig_session", _claim)

        _cmd_hold(_args("hold", "--rh", "45", "--duration-s", "0",
                        "--execute", "--yes"))

        assert seen["what"] == f"{CLAIM_KIND}:{store.start_run('x')}"
        assert seen["what"].count(":") == 2      # no trailing empty run-id field


# ── The signal path ──────────────────────────────────────────────────────────

class TestSignalPath:
    def _session(self, rh=None):
        rh = rh or FakeRH()
        return HoldSession(rh, FakeManager(rh), FakeStore(),
                           lambda _s: None, run_async=lambda c: c.close(),
                           setpoint_pct=45.0), rh

    def test_the_handler_zeroes_the_humidifier_once(self):
        session, rh = self._session()
        previous = signal.getsignal(signal.SIGINT)
        restore = install_handlers(session, signals=(signal.SIGINT,))
        try:
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        finally:
            restore()
            signal.signal(signal.SIGINT, previous)
        assert rh.safe_off_calls == 1

    def test_a_second_signal_during_teardown_does_not_zero_twice(self):
        """A second write sequence down the same serial line is declined, which
        is ``ParkGuard``'s discipline for the same reason."""
        rh = FakeRH()
        session, _ = self._session(rh)
        previous = signal.getsignal(signal.SIGINT)
        restore = install_handlers(session, signals=(signal.SIGINT,))
        handler = signal.getsignal(signal.SIGINT)

        original_safe_off = rh.safe_off

        def _reentrant():
            original_safe_off()
            handler(signal.SIGINT, None)     # must be declined, not re-entered

        rh.safe_off = _reentrant
        try:
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
        finally:
            restore()
            signal.signal(signal.SIGINT, previous)
        assert rh.safe_off_calls == 1

    def test_the_handler_restores_the_previous_disposition(self):
        """A second Ctrl-C must reach the default handler, or an operator
        watching a wedged teardown has only the task manager."""
        session, _ = self._session()
        previous = signal.getsignal(signal.SIGINT)
        restore = install_handlers(session, signals=(signal.SIGINT,))
        try:
            assert signal.getsignal(signal.SIGINT) is not previous
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGINT, None)
            assert signal.getsignal(signal.SIGINT) is previous
        finally:
            restore()
            signal.signal(signal.SIGINT, previous)


# ── The run row ──────────────────────────────────────────────────────────────

class TestRunRow:
    def test_a_bounded_hold_that_reached_its_duration_is_done(self):
        code, rh, store = _drive(_args("hold", "--rh", "45", "--duration-s", "0"),
                                 seconds=0.0)
        assert code == EXIT_OK
        assert store.status == "done"
        assert rh.safe_off_calls == 1
        assert store.closed is True

    def test_an_interrupted_bounded_hold_is_interrupted_and_fails(self, monkeypatch):
        def _interrupt(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(env_hold, "hold_loop", _interrupt)
        code, rh, store = _drive(_args("hold", "--rh", "45", "--duration-h", "4"),
                                 seconds=14400.0)
        assert code == EXIT_FAILED
        assert store.status == "interrupted"
        assert rh.safe_off_calls == 1

    def test_an_until_signal_hold_stopped_by_the_operator_succeeds(self, monkeypatch):
        """The status still records that nothing but a person decided the end;
        the exit code says the tool did what it was asked."""
        def _interrupt(*_a, **_k):
            raise KeyboardInterrupt

        monkeypatch.setattr(env_hold, "hold_loop", _interrupt)
        code, _, store = _drive(_args("hold", "--rh", "45"), seconds=None)
        assert code == EXIT_OK
        assert store.status == "interrupted"

    def test_a_refusing_driver_is_aborted(self):
        rh = FakeRH()
        rh.set_setpoint_error = SafetyError("RH setpoint 99% exceeds limit of 95%")
        code, rh, store = _drive(_args("hold", "--rh", "99", "--duration-s", "0"),
                                 rh=rh, seconds=0.0)
        assert code == EXIT_FAILED
        assert store.status == "aborted"
        assert rh.safe_off_calls == 1

    def test_an_unnamed_exception_is_recorded_as_error(self, monkeypatch):
        def _blow_up(*_a, **_k):
            raise RuntimeError("something no except names")

        monkeypatch.setattr(env_hold, "hold_loop", _blow_up)
        store = FakeStore()
        with pytest.raises(RuntimeError):
            _drive(_args("hold", "--rh", "45", "--duration-s", "0"), store=store,
                   seconds=0.0)
        assert store.status == "error"
        assert store.closed is True


# ── The watchdog ─────────────────────────────────────────────────────────────

class TestWatchdog:
    def test_the_watch_is_built_with_the_setpoint_and_the_configured_thresholds(self):
        thresholds = rh_watchdog_config()
        _drive(_args("hold", "--rh", "45", "--duration-s", "0"), seconds=0.0,
               thresholds=thresholds)
        watch = RecordingWatch.last
        assert watch.setpoint_pct == 45.0
        assert watch.kwargs["thresholds"] == thresholds
        assert watch.kwargs["run_id"] == "RUNID"
        assert watch.kwargs["data_store"] is not None

    def test_the_loop_samples_at_least_once_per_poll_interval(self):
        """``RHHoldWatch`` owns no clock, so the cadence is this loop's problem.
        Sampling *more* often than the interval is free; less often silently
        lengthens it."""
        clock = FakeClock()
        watch = RecordingWatch(lambda: (22.5, 44.0), 45.0)
        hold_loop(watch, 45.0, seconds=600.0, heartbeat_s=300.0, quiet=True,
                  now=clock.now, sleep=clock.sleep)
        assert watch.samples >= 600 / 60      # one per rh_poll_interval_s of held time

    def test_a_sub_ceiling_poll_interval_tightens_the_cadence(self):
        """A rig configured to poll faster than :data:`SAMPLE_CEILING_S` gets that
        cadence. The 60 s default cannot catch this: it is above the ceiling, so
        the ceiling alone satisfies it whether or not the setting is read at all.
        """
        clock = FakeClock()
        watch = RecordingWatch(lambda: (22.5, 44.0), 45.0)
        hold_loop(watch, 45.0, seconds=10.0, heartbeat_s=300.0, quiet=True,
                  poll_interval_s=1.0, now=clock.now, sleep=clock.sleep)
        assert watch.samples >= 10        # one per second, not one per 5 s

    def test_the_configured_poll_interval_reaches_the_loop(self, monkeypatch):
        """Threaded from the same thresholds the watch is built with, so the loop
        cannot sample slower than the watchdog was configured to be told about."""
        seen: dict = {}

        def _spy(*_a, **kw):
            seen.update(kw)

        monkeypatch.setattr(env_hold, "hold_loop", _spy)
        thresholds = dict(rh_watchdog_config(), poll_interval_s=2.0)
        _drive(_args("hold", "--rh", "45", "--duration-s", "0"), seconds=0.0,
               thresholds=thresholds)

        assert seen["poll_interval_s"] == 2.0

    def test_a_nonsense_poll_interval_falls_back_to_the_ceiling(self):
        """``sleep(0)`` in this loop is a hot loop on a serial line."""
        clock = FakeClock()
        watch = RecordingWatch(lambda: (22.5, 44.0), 45.0)
        hold_loop(watch, 45.0, seconds=60.0, heartbeat_s=300.0, quiet=True,
                  poll_interval_s=0.0, now=clock.now, sleep=clock.sleep)
        assert watch.samples <= 60 / env_hold.SAMPLE_CEILING_S + 1

    def test_a_sustained_excursion_records_an_alert_row(self):
        from softae.drivers.contracts import RHHoldWatch

        store = FakeStore()
        clock = FakeClock()
        watch = RHHoldWatch(lambda: (22.5, 30.0), 45.0,
                            thresholds={"warn_pct": 3.0, "fault_pct": 5.0,
                                        "grace_s": 60.0, "poll_interval_s": 10.0},
                            data_store=store, run_id="RUNID", now=clock.now)
        hold_loop(watch, 45.0, seconds=300.0, heartbeat_s=300.0, quiet=True,
                  now=clock.now, sleep=clock.sleep)

        assert store.alerts, "a sustained 15 %RH excursion must be announced"
        assert watch.verdict.is_fault

    def test_a_fault_verdict_does_not_stop_the_hold(self):
        """A hold whose humidity is off command is a fact the operator needs; it
        is not a reason for software to stop actuating a chamber a human asked
        to be held."""
        clock = FakeClock()
        watch = RecordingWatch(lambda: (22.5, 10.0), 45.0)
        watch.verdict = RHHoldVerdict(RH_FAULT, 45.0, pv_pct=10.0,
                                      reason="sustained more than 5 %RH BELOW")

        hold_loop(watch, 45.0, seconds=600.0, heartbeat_s=300.0, quiet=True,
                  now=clock.now, sleep=clock.sleep)

        assert clock.t >= 600.0        # it ran the whole commanded duration
        assert watch.verdict.is_fault


# ── Rendering ────────────────────────────────────────────────────────────────

def test_an_unreadable_pv_renders_as_dashes_never_as_zero():
    """``AsyncRHController`` deliberately NaNs a held reading past
    ``max_stale_s``; flattening that into a plausible figure would let a dead
    sensor read as a working one for four hours."""
    watch = MagicMock()
    watch.verdict = RHHoldVerdict("converging", 45.0, pv_pct=float("nan"))
    watch.temperature_C = float("nan")

    line = env_hold.heartbeat_line(900.0, watch, 45.0, 9900.0)

    assert "--%" in line
    assert "--C" in line
    assert "0.0%" not in line
    assert "[0:15:00]" in line
