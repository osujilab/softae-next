"""Linear setpoint ramp: interval-derived schedule, progress, cancel, and the °C/min anneal unit.

Spec: ``docs/SubAgent docs/manual_temp_ramp_progress_abort.md``. The runner is
synchronous, so the async ``conftest.FakeClock`` does not fit; a local sync fake
clock replaces ``softae.drivers.temp_ramp.time`` instead.
"""

from __future__ import annotations

import itertools
import threading
from pathlib import Path

import numpy as np
import pytest
from structlog.testing import capture_logs

from softae.drivers import temp_ramp
from softae.drivers.async_temp_controller import AsyncTempController
from softae.drivers.mock_temp_controller import MockTempController
from softae.drivers.temp_ramp import (
    MAX_UPDATE_INTERVAL_S,
    RampReport,
    RampStep,
    ramp_schedule,
    update_interval_s,
)
from softae.errors import ValidationError_


# ── fakes ────────────────────────────────────────────────────────────────────

class _SyncClock:
    """``monotonic``/``sleep`` stand-in; ``at(t, fn)`` fires *fn* once the clock reaches *t*."""

    def __init__(self) -> None:
        self.now = 1000.0
        self._hooks: list[tuple[float, object]] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:
        self.now += dt
        due = [h for h in self._hooks if h[0] <= self.now]
        self._hooks = [h for h in self._hooks if h[0] > self.now]
        for _, fn in due:
            fn()

    def at(self, t: float, fn) -> None:
        self._hooks.append((self.now + t, fn))


class _Registers:
    """Minimal ``minimalmodbus.Instrument`` stand-in recording every write."""

    def __init__(self, clock: _SyncClock, sp_C: float = 25.0) -> None:
        self._clock = clock
        self.regs = {0: int(sp_C * 10), 1: int(sp_C * 10)}
        self.writes: list[tuple[float, float]] = []   # (clock time, °C)
        self.reads = 0

    def read_register(self, reg):
        self.reads += 1
        return self.regs[reg]

    def write_register(self, reg, value):
        self.regs[reg] = value
        self.writes.append((self._clock.now, value / 10))


@pytest.fixture
def clock(monkeypatch) -> _SyncClock:
    c = _SyncClock()
    monkeypatch.setattr(temp_ramp, "time", c)
    return c


@pytest.fixture
def tc(clock) -> AsyncTempController:
    ctrl = AsyncTempController(config={"min_temp": 0, "max_temp": 150})
    ctrl._instrument = _Registers(clock)
    return ctrl


def _written(tc) -> list[float]:
    return [c for _, c in tc._instrument.writes]


# ── schedule: the interval rule ──────────────────────────────────────────────

def test_ramp_schedule_1C_per_min_over_70C_updates_every_6s_in_tenths():
    times, temps = ramp_schedule(25.0, 95.0, 70 * 60.0)
    assert update_interval_s(25.0, 95.0, 4200.0) == pytest.approx(6.0)
    assert len(times) == 701                        # t = 0 write + 700 steps
    assert np.diff(times) == pytest.approx(np.full(700, 6.0))
    assert np.diff(temps) == pytest.approx(np.full(700, 0.1))
    assert (temps[0], temps[-1], times[-1]) == (25.0, 95.0, 4200.0)


def test_ramp_schedule_slow_rate_interval_capped_at_60s():
    t_span = 2.0 / 0.05 * 60.0                      # 2 °C at 0.05 °C/min
    assert update_interval_s(25.0, 27.0, t_span) == MAX_UPDATE_INTERVAL_S
    times, temps = ramp_schedule(25.0, 27.0, t_span)
    assert np.diff(times).max() == pytest.approx(60.0)
    assert temps[-1] == 27.0


def test_ramp_schedule_shipped_anneal_rate_300C_per_min_is_14_one_second_steps():
    t_span = temp_ramp.ramp_span_s(25.0, 95.0, 300.0)
    assert t_span == pytest.approx(14.0)
    times, temps = ramp_schedule(25.0, 95.0, t_span)
    assert len(times) == 15
    assert np.diff(times) == pytest.approx(np.full(14, 1.0))
    assert list(temps) == [25.0 + 5 * i for i in range(15)]


@pytest.mark.parametrize("up_int", [None, 0.5, 30.0, 60.0, 61.0, 600.0, 3600.0])
def test_ramp_schedule_any_rate_and_span_never_exceeds_60s_between_writes(up_int):
    rates_C_per_min = [0.001, 0.01, 0.05, 0.1, 1.0, 5.0, 60.0, 300.0, 6000.0]
    deltas = [0.2, 0.5, 5.0, 70.0, -70.0, 175.0]
    for rate, delta in itertools.product(rates_C_per_min, deltas):
        start = 25.0
        t_span = abs(delta) / rate * 60.0
        times, temps = ramp_schedule(start, start + delta, t_span, up_int)
        case = (rate, delta, up_int)
        assert len(times) >= 2, case
        assert np.diff(times).max() <= MAX_UPDATE_INTERVAL_S + 1e-9, case
        assert times[0] == 0.0 and times[-1] == pytest.approx(t_span), case
        assert temps[0] == round(start, 1) and temps[-1] == round(start + delta, 1), case


def test_ramp_schedule_short_span_ends_at_target():
    # The old math (n = max(int(t_span/up_int), 1)) gave ONE point here: T_start only.
    times, temps = ramp_schedule(25.0, 30.0, 1.0, 1.0)
    assert list(temps) == [25.0, 30.0]
    # anneal at the shipped 300 °C/min over a 9 °C step (t_span 1.8 s)
    _, temps = ramp_schedule(25.0, 34.0, temp_ramp.ramp_span_s(25.0, 34.0, 300.0))
    assert temps[-1] == 34.0


def test_ramp_schedule_equal_rounded_endpoints_is_empty():
    times, temps = ramp_schedule(25.0, 25.04, 10.0)
    assert len(times) == len(temps) == 0


# ── real driver ──────────────────────────────────────────────────────────────

def test_ramp_linear_first_write_is_start_last_is_end(tc):
    report = tc.ramp_linear(25.0, 35.0, 100.0)
    assert _written(tc)[0] == 25.0 and _written(tc)[-1] == 35.0
    assert report == RampReport(total=report.total, steps_done=report.total, last_sp=35.0,
                                cancelled=False, zero_span=False,
                                elapsed_s=pytest.approx(100.0))


def test_ramp_linear_positional_legacy_call_still_binds(tc):
    report = tc.ramp_linear(25.0, 30.0, 10.0, 1.0, 0)   # tab's positional order
    assert report.total == 11 and _written(tc)[-1] == 30.0


def test_ramp_linear_on_step_called_once_per_step_with_fields(tc):
    steps: list[RampStep] = []
    report = tc.ramp_linear(25.0, 28.0, 30.0, on_step=steps.append)
    assert [s.index for s in steps] == list(range(1, report.total + 1))
    assert all(s.total == report.total for s in steps)
    assert [s.sp_C for s in steps] == _written(tc)
    assert steps[-1].next_in_s == 0.0 and steps[0].next_in_s > 0.0


def test_ramp_linear_writes_never_more_than_60s_apart(tc):
    tc.ramp_linear(25.0, 27.0, 2.0 / 0.05 * 60.0, print_flag=0)
    times = [t for t, _ in tc._instrument.writes]
    assert max(np.diff(times)) <= MAX_UPDATE_INTERVAL_S
    assert _written(tc)[-1] == 27.0


def test_ramp_linear_explicit_up_int_is_capped_at_60s(tc):
    tc.ramp_linear(25.0, 26.0, 600.0, 300.0, 0)
    times = [t for t, _ in tc._instrument.writes]
    assert max(np.diff(times)) <= MAX_UPDATE_INTERVAL_S


def test_ramp_linear_cancel_mid_ramp_holds_last_sp(tc, clock):
    cancel = threading.Event()
    cancel_at = 0.4 * 4200.0
    clock.at(cancel_at, cancel.set)
    t0 = clock.now
    report = tc.ramp_linear(25.0, 95.0, 4200.0, cancel=cancel)
    writes = tc._instrument.writes
    assert report.cancelled and 0 < report.steps_done < report.total
    assert all(t <= t0 + cancel_at for t, _ in writes)          # no write after the cancel
    assert report.last_sp == writes[-1][1] == report.last_sp
    assert report.steps_done == len(writes)
    assert clock.now - (t0 + cancel_at) <= temp_ramp.POLL_S     # latency ≤ one poll


def test_ramp_linear_cancel_before_start_writes_nothing(tc):
    cancel = threading.Event()
    cancel.set()
    report = tc.ramp_linear(25.0, 95.0, 100.0, cancel=cancel)
    assert report.cancelled and report.last_sp is None and report.steps_done == 0
    assert tc._instrument.writes == []


def test_ramp_linear_zero_span_reports_and_writes_nothing(tc):
    with capture_logs() as logs:
        report = tc.ramp_linear(25.0, 25.0, 100.0)
    assert report.zero_span and report.total == 0 and report.last_sp is None
    assert tc._instrument.writes == []
    assert any(e["event"] == "ramp_zero_span" for e in logs)


def test_ramp_linear_callback_error_does_not_stop_ramp(tc):
    def broken(step):
        raise RuntimeError("label gone")

    with capture_logs() as logs:
        report = tc.ramp_linear(25.0, 26.0, 10.0, on_step=broken)
    assert report.steps_done == report.total and _written(tc)[-1] == 26.0
    assert [e["event"] for e in logs].count("ramp_progress_callback_failed") == 1


def test_ramp_linear_logs_started_and_finished(tc):
    with capture_logs() as logs:
        tc.ramp_linear(25.0, 26.0, 10.0, print_flag=0)
    events = [e["event"] for e in logs if e["log_level"] == "info"]
    assert events == ["ramp_started", "ramp_finished"]


def test_ramp_linear_cancel_is_not_stop_wait(tc):
    cancel = threading.Event()
    tc.ramp_linear(25.0, 30.0, 10.0, on_step=lambda s: cancel.set(), cancel=cancel)
    assert not tc._stop_wait.is_set()
    assert tc.get_sp() == 25.0          # a following serial call still succeeds


# ── the park race: check-and-write under the SP-write lock ──────────────────
#
# An E-Stop sets the ramp's cancel on the GUI thread, then a park worker writes
# the safe SP. A ramp thread that passed its poll check before the cancel must
# not write after the park. These tests use real threads and real (short) time;
# every wait is bounded so a regression fails rather than hangs.

_BOUND_S = 5.0


class _CancelOnEnter:
    """A lock stand-in that sets *cancel* on its *at*-th entry — i.e. after the
    runner's poll check passed, exactly where a racing abort would land."""

    def __init__(self, cancel: threading.Event, at: int) -> None:
        self._cancel, self._at, self.entries = cancel, at, 0

    def __enter__(self):
        self.entries += 1
        if self.entries == self._at:
            self._cancel.set()
        return self

    def __exit__(self, *exc) -> None:
        return None


class _ObservedRLock:
    """``threading.RLock`` that announces when a named thread *starts* to acquire it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._armed: dict[str, threading.Event] = {}

    def expect(self, thread_name: str) -> threading.Event:
        event = threading.Event()
        self._armed[thread_name] = event
        return event

    def __enter__(self):
        event = self._armed.pop(threading.current_thread().name, None)
        if event is not None:
            event.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._lock.release()


class _GatedRegisters:
    """Thread-aware register fake: logs ``(thread, op, °C)`` in order, and can
    hold one named thread inside one op until released."""

    def __init__(self, sp_C: float = 25.0) -> None:
        self.regs = {0: int(sp_C * 10), 1: int(sp_C * 10)}
        self.log: list[tuple[str, str, float]] = []
        self._gates: dict[tuple[str, str], tuple[threading.Event, threading.Event]] = {}

    def gate(self, thread_name: str, op: str) -> tuple[threading.Event, threading.Event]:
        entered, release = threading.Event(), threading.Event()
        self._gates[(thread_name, op)] = (entered, release)
        return entered, release

    def _pass(self, op: str) -> None:
        gate = self._gates.pop((threading.current_thread().name, op), None)
        if gate is not None:
            gate[0].set()
            assert gate[1].wait(_BOUND_S), f"gate {op} never released"

    def read_register(self, reg):
        self._pass("read")
        self.log.append((threading.current_thread().name, "read", self.regs[reg] / 10))
        return self.regs[reg]

    def write_register(self, reg, value):
        self._pass("write")
        self.regs[reg] = value
        self.log.append((threading.current_thread().name, "write", value / 10))

    def writes(self) -> list[tuple[str, float]]:
        return [(who, c) for who, op, c in self.log if op == "write"]


@pytest.fixture
def raced_tc() -> AsyncTempController:
    ctrl = AsyncTempController(config={"min_temp": 0, "max_temp": 150})
    ctrl._instrument = _GatedRegisters()
    ctrl._sp_write_lock = _ObservedRLock()
    return ctrl


def _run(name: str, fn, out: dict) -> threading.Thread:
    def body():
        out[name] = fn()
    thread = threading.Thread(target=body, name=name, daemon=True)
    thread.start()
    return thread


def _join(*threads: threading.Thread) -> None:
    for thread in threads:
        thread.join(_BOUND_S)
        assert not thread.is_alive(), f"{thread.name} did not finish"


def test_run_ramp_cancel_between_poll_check_and_write_writes_nothing(clock):
    cancel = threading.Event()
    lock = _CancelOnEnter(cancel, at=2)           # step 2's check-and-write
    written: list[float] = []
    report = temp_ramp.run_ramp(written.append, 25.0, 26.0, 2.0, 1.0,
                                cancel=cancel, write_lock=lock)
    assert lock.entries == 2, "step 2 never reached its write point"
    assert written == [25.0]
    assert report.cancelled and report.steps_done == 1 and report.last_sp == 25.0


def test_ramp_linear_park_while_ramp_write_holds_lock_lands_after_it(raced_tc):
    regs, lock = raced_tc._instrument, raced_tc._sp_write_lock
    cancel, out = threading.Event(), {}
    ramp_in_write, release_ramp = regs.gate("ramp", "write")
    ramp = _run("ramp", lambda: raced_tc.ramp_linear(
        25.0, 26.0, 0.2, 0.1, print_flag=0, cancel=cancel), out)
    assert ramp_in_write.wait(_BOUND_S)
    park_waiting = lock.expect("park")
    park = _run("park", lambda: raced_tc.write_sp(20.0, print_flag=0), out)
    assert park_waiting.wait(_BOUND_S)
    cancel.set()                                   # E-Stop: abort, then park
    release_ramp.set()
    _join(ramp, park)

    assert regs.writes() == [("ramp", 25.0), ("park", 20.0)]
    assert regs.regs[0] == 200
    assert out["ramp"].cancelled and out["ramp"].last_sp == 25.0


def test_ramp_linear_step_after_cancel_and_park_writes_nothing(raced_tc):
    """The ramp passed its poll check (cancel clear) and is waiting on the lock
    the park holds; the park's cancel lands, the park writes, and the ramp —
    re-checking under the lock — writes nothing."""
    regs, lock = raced_tc._instrument, raced_tc._sp_write_lock
    cancel, out, staged = threading.Event(), {}, {}
    park_in_read, release_park = regs.gate("park", "read")
    park_started = threading.Event()

    def on_step(step: RampStep) -> None:        # runs on the ramp thread, lock free
        if step.index != 1:
            return
        staged["park"] = _run("park", lambda: raced_tc.write_sp(20.0, print_flag=0), out)
        assert park_in_read.wait(_BOUND_S)       # the park now holds the SP lock
        staged["ramp_waiting"] = lock.expect("ramp")
        park_started.set()

    ramp = _run("ramp", lambda: raced_tc.ramp_linear(
        25.0, 26.0, 0.2, 0.1, print_flag=0, on_step=on_step, cancel=cancel), out)
    assert park_started.wait(_BOUND_S)
    park = staged["park"]
    # Set only when the ramp starts to take the lock — after its poll check
    # passed with cancel still clear.
    assert staged["ramp_waiting"].wait(_BOUND_S), "ramp never reached step 2's lock"
    cancel.set()
    release_park.set()
    _join(ramp, park)

    assert regs.writes() == [("ramp", 25.0), ("park", 20.0)]
    assert out["ramp"].cancelled and out["ramp"].steps_done == 1


def test_write_sp_read_write_pair_not_interleaved_by_another_write_sp(raced_tc):
    regs, lock = raced_tc._instrument, raced_tc._sp_write_lock
    out = {}
    a_in_read, release_a = regs.gate("A", "read")
    a = _run("A", lambda: raced_tc.write_sp(40.0, print_flag=0), out)
    assert a_in_read.wait(_BOUND_S)
    b_waiting = lock.expect("B")
    b = _run("B", lambda: raced_tc.write_sp(30.0, print_flag=0), out)
    assert b_waiting.wait(_BOUND_S)
    release_a.set()
    _join(a, b)

    assert [(who, op) for who, op, _ in regs.log] == [
        ("A", "read"), ("A", "write"), ("B", "read"), ("B", "write")]
    assert regs.regs[0] == 300


# ── anneal: °C/min, and the retired °C/s key ─────────────────────────────────

@pytest.mark.parametrize("make", [
    lambda: AsyncTempController(config={"min_temp": 0, "max_temp": 150}),
    MockTempController,
])
def test_anneal_legacy_ramp_rate_refused_before_any_serial_call(make, monkeypatch):
    ctrl = make()
    touched = []
    monkeypatch.setattr(ctrl, "get_sp", lambda: touched.append("get_sp") or 25.0)
    monkeypatch.setattr(ctrl, "write_sp", lambda *a, **k: touched.append("write_sp"))
    with pytest.raises(ValidationError_, match=r"ramp_rate_C_per_min = 300"):
        ctrl.anneal(target_temp_C=85.0, hold_time_s=0, ramp_rate=5)
    assert touched == []


def test_anneal_unknown_kwarg_is_a_type_error(tc):
    with pytest.raises(TypeError, match="bogus"):
        tc.anneal(target_temp_C=85.0, hold_time_s=0, bogus=1)
    assert tc._instrument.writes == []


class _StopAfterRamp(Exception):
    pass


def test_anneal_rate_is_C_per_min_and_interval_is_derived(tc, monkeypatch):
    calls = []
    monkeypatch.setattr(tc, "ramp_linear", lambda *a, **k: calls.append((a, k)))

    def stop(**_):
        raise _StopAfterRamp

    monkeypatch.setattr(tc, "wait", stop)
    with pytest.raises(_StopAfterRamp):
        tc.anneal(target_temp_C=95.0, hold_time_s=0, ramp_rate_C_per_min=300)
    (args, kwargs), = calls
    assert args == (25.0, 95.0, pytest.approx(14.0), None)   # 70 °C at 5 °C/s
    assert _written(tc) == [25.0]                            # finally: restore SP


# ── mock ─────────────────────────────────────────────────────────────────────

def test_mock_ramp_linear_honours_on_step_and_cancel():
    mock = MockTempController()
    written: list[float] = []
    orig = mock.write_sp
    mock.write_sp = lambda sp, print_flag=1: (written.append(sp), orig(sp, print_flag))

    steps: list[RampStep] = []
    report = mock.ramp_linear(25.0, 95.0, 4200.0, on_step=steps.append)
    assert [s.index for s in steps] == [1, 701] and report.total == 701
    assert written == [25.0, 95.0] and report.last_sp == 95.0

    written.clear()
    cancel = threading.Event()
    report = mock.ramp_linear(95.0, 25.0, 4200.0,
                              on_step=lambda s: cancel.set(), cancel=cancel)
    assert written == [95.0]
    assert report.cancelled and report.last_sp == 95.0 and report.steps_done == 1


def test_mock_ramp_linear_cancel_rechecked_under_sp_lock_writes_nothing_after():
    """Same check-and-write rule as the real driver: entry 3 is the ramp's own
    step-2 check (1 = step 1 check, 2 = write_sp re-entering for step 1)."""
    mock = MockTempController()
    cancel = threading.Event()
    real_lock = mock._sp_write_lock
    probe = _CancelOnEnter(cancel, at=3)

    class _Both:
        def __enter__(self):
            probe.__enter__()
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    mock._sp_write_lock = _Both()
    report = mock.ramp_linear(25.0, 95.0, 4200.0, cancel=cancel)
    assert probe.entries == 3
    assert mock.get_sp() == 25.0
    assert report.cancelled and report.last_sp == 25.0 and report.steps_done == 1


def test_mock_ramp_linear_zero_span_writes_nothing():
    mock = MockTempController()
    report = mock.ramp_linear(25.0, 25.0, 10.0)
    assert report.zero_span and report.last_sp is None


# ── catalog: the retired key fails at load, not on the rig ───────────────────

def _anneal_task(**params):
    from softae.core.task_catalog import Task

    return Task(name="cure", instrument="temp_controller", method="anneal",
                timeout_s=29_400.0, params={"hold_time_s": 28_800, **params})


def test_validate_task_flags_legacy_ramp_rate_with_conversion():
    from softae.core.task_catalog import validate_task

    (problem,) = validate_task(_anneal_task(ramp_rate=5))
    assert "ramp_rate_C_per_min = 300" in problem
    assert validate_task(_anneal_task(ramp_rate_C_per_min=300)) == []


def test_catalog_load_rejects_legacy_ramp_rate_anneal(tmp_path):
    from softae.core.task_catalog import TaskCatalog

    path = tmp_path / "tasks.toml"
    path.write_text(
        "[tasks.old]\ninstrument = \"temp_controller\"\nmethod = \"anneal\"\n"
        "timeout_s = 29400.0\n[tasks.old.params]\nhold_time_s = 28800\nramp_rate = 5\n"
        "\n[tasks.new]\ninstrument = \"temp_controller\"\nmethod = \"anneal\"\n"
        "timeout_s = 29400.0\n[tasks.new.params]\nhold_time_s = 28800\n"
        "ramp_rate_C_per_min = 300\n",
        encoding="utf-8",
    )
    cat = TaskCatalog.load_toml(path)
    assert "old" not in cat and "new" in cat


def test_shipped_default_catalog_anneal_uses_C_per_min():
    from importlib.resources import files

    from softae.core.task_catalog import TaskCatalog

    path = files("softae.catalog_defaults").joinpath("tasks.toml")
    task = TaskCatalog.load_toml(Path(str(path))).get("anneal_85C_8h")
    assert task.params["ramp_rate_C_per_min"] == 300
    assert "ramp_rate" not in task.params
