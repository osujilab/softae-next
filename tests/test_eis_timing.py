"""The rig claim `eis_timing.py` now takes while it holds the ports.

Unlike `eis_validate.py`, this tool has no `DataStore` and no `run_id` -- it is a
pure benchmarking harness (see its own module docstring). Before this claim
existed, a GUI or another tool opened mid-sweep found the rig apparently free
and could connect onto the same serial ports underneath it. The pattern here is
the same one `tests/test_eis_validate.py`'s `TestRigClaim` pins for the sibling
tool, narrowed to what this tool actually has: no run row to finalize, no
narration stream, and an ad-hoc timestamp (`ET.CLAIM_KIND:<stamp>`) standing in
for the run id `tool:eis-validate:<run_id>` would otherwise carry.
"""

from __future__ import annotations

import pytest

from softae.tools import eis_timing as ET

#: `--yes` skips the actuation confirmation prompt this tool already has --
#: this suite is about the rig LOCK, not that prompt. One preset, one pass, no
#: warmup: the claim is what these pin, not the sweep loop's own behaviour.
ARGV = ["--channel", "1", "--presets", "Quick", "--repeats", "1", "--warmup", "0",
        "--yes"]


class _FakeManager:
    """Reads as REAL hardware to `session_is_simulated` / `probe_motion`.

    `keithley` is the one registered instrument, is outside `MOTION_INSTRUMENTS`,
    and is not an instance of any shipped mock class -- so `session_is_simulated`
    returns `False` (a claim is taken) while `assert_hardware_armed` stays the
    documented no-op it is for every EIS-only manager. `connect_all` /
    `disconnect_all` open and close nothing: no real port is touched by any test
    in this file.
    """

    def __init__(self):
        self.names = ["keithley"]
        self.connect_calls = 0
        self.disconnect_calls = 0

    def get(self, _name):
        return object()

    async def connect_all(self):
        self.connect_calls += 1

    async def disconnect_all(self):
        self.disconnect_calls += 1


def _boom(*_a, **_kw):
    raise AssertionError("this must not be reached")


async def _boom_async(*_a, **_kw):
    raise AssertionError("no port may be opened on this path")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A per-test lock scope -- a lock left behind by one test must not leak."""
    import softae.core.run_lock as RL

    monkeypatch.setattr(RL, "DEFAULT_SCOPE", tmp_path / "rig_scope")


@pytest.fixture(autouse=True)
def _fast_sweep(monkeypatch):
    """Stub the timed sweep itself: no `.mscr` build, no driver I/O.

    What these tests pin is the claim wrapped around `connect_all` through
    `disconnect_all`, not the sweep loop's own timing behaviour.
    """
    monkeypatch.setattr(ET, "_time_one_sweep", lambda *_a, **_kw: 0.01)


def _run(monkeypatch, *, manager=None, extra=()):
    import softae.drivers.factory as factory

    rig = manager if manager is not None else _FakeManager()
    monkeypatch.setattr(factory, "create_manager", lambda **_kw: rig)
    return ET.main(list(ARGV) + list(extra)), rig


class TestRigClaim:
    """`ET.CLAIM_KIND` is `"tool:eis-timing"`; the claim's `what` is
    `f"{ET.CLAIM_KIND}:{stamp}"`, mirroring `tools/eis_validate.py`'s
    `tool:eis-validate:<run_id>` with a launch timestamp standing in for the
    run id this tool has none of.
    """

    def test_claim_kind_is_the_declared_family_name(self):
        assert ET.CLAIM_KIND == "tool:eis-timing"

    # -- (a) --mock takes no claim at all ------------------------------------

    def test_a_mock_run_takes_no_rig_claim(self, monkeypatch):
        """`held_rig_session` is never even called under `--mock`.

        The gate lives on this tool's own `mock` flag (`contextlib.nullcontext()`
        in place of the claim), not on `held_rig_session`'s internal
        `session_is_simulated` exemption -- so this is checked by making the
        function itself explode if reached, the same shape as
        `tests/test_eis_validate.py`'s mock-exemption test.
        """
        import softae.core.rig_session as RS

        monkeypatch.setattr(RS, "held_rig_session", _boom)
        exit_code, _rig = _run(monkeypatch, extra=["--mock"])
        assert exit_code == 0

    def test_a_mock_run_leaves_no_lock_file_behind(self, monkeypatch):
        from softae.core.run_lock import read_run_lock

        exit_code, _rig = _run(monkeypatch, extra=["--mock"])
        assert exit_code == 0
        assert read_run_lock() is None

    # -- (b) a non-mock run DOES claim, shaped tool:eis-timing:<stamp> ------

    def test_a_real_run_claims_the_rig_naming_its_stamp(self, monkeypatch):
        from softae.core.run_lock import read_run_lock

        seen = {}

        def _peek(*_a, **_kw):
            seen["lock"] = read_run_lock()
            return 0.01

        monkeypatch.setattr(ET, "_time_one_sweep", _peek)
        exit_code, _rig = _run(monkeypatch)
        assert exit_code == 0
        assert seen["lock"] is not None, "the run held the ports and claimed nothing"
        what = seen["lock"].what
        assert what.startswith(f"{ET.CLAIM_KIND}:")
        stamp = what[len(ET.CLAIM_KIND) + 1:]
        assert stamp, "the third field must be filled, never a trailing bare colon"
        # main()'s "%Y%m%dT%H%M%SZ" -- 16 chars, ending in Z.
        assert len(stamp) == 16 and stamp.endswith("Z")

    def test_the_claim_is_taken_before_any_port_is_opened(self, monkeypatch):
        from softae.core.run_lock import read_run_lock

        order = []
        rig = _FakeManager()
        real_connect = rig.connect_all

        async def _watched_connect():
            order.append(("connect", read_run_lock() is not None))
            return await real_connect()

        rig.connect_all = _watched_connect
        exit_code, _rig = _run(monkeypatch, manager=rig)
        assert exit_code == 0
        assert order == [("connect", True)]

    def test_a_real_run_gives_the_rig_back(self, monkeypatch):
        from softae.core.run_lock import read_run_lock

        exit_code, _rig = _run(monkeypatch)
        assert exit_code == 0
        assert read_run_lock() is None

    def test_the_claim_outlives_the_disconnect(self, monkeypatch):
        """Acquired before `connect_all`, released after `disconnect_all`."""
        from softae.core.run_lock import read_run_lock

        order = []
        rig = _FakeManager()
        real_disconnect = rig.disconnect_all

        async def _watched_disconnect():
            order.append(("disconnect", read_run_lock() is not None))
            return await real_disconnect()

        rig.disconnect_all = _watched_disconnect
        exit_code, _rig = _run(monkeypatch, manager=rig)
        assert exit_code == 0
        assert order == [("disconnect", True)]
        assert read_run_lock() is None

    # -- (c) a foreign claim is surfaced, not silently bypassed --------------

    def test_a_foreign_claim_is_surfaced_not_silently_bypassed(self, monkeypatch,
                                                                capsys):
        from softae.core.run_lock import RunLock, RunLockHeld

        holder = RunLock(pid=424242, what="gui:desktop",
                         started_at="2026-08-19T14:02:00+00:00", host="bench")

        def _held(*_a, **_kw):
            raise RunLockHeld(holder)

        monkeypatch.setattr("softae.core.rig_session.claim_rig_session", _held)
        rig = _FakeManager()
        rig.connect_all = _boom_async
        exit_code, _rig = _run(monkeypatch, manager=rig)
        assert exit_code == 1
        out = capsys.readouterr().out
        assert "NOT STARTING" in out and "424242" in out

    def test_a_foreign_claim_never_opens_a_port(self, monkeypatch):
        """`connect_all` exploding if reached IS the assertion."""
        from softae.core.run_lock import RunLockHeld, RunLock

        def _held(*_a, **_kw):
            raise RunLockHeld(RunLock(pid=1, what="tool:eis-validate:x",
                                      started_at="2026-01-01T00:00:00+00:00",
                                      host="bench"))

        monkeypatch.setattr("softae.core.rig_session.claim_rig_session", _held)
        rig = _FakeManager()
        rig.connect_all = _boom_async
        exit_code, rig = _run(monkeypatch, manager=rig)
        assert exit_code == 1
        assert rig.connect_calls == 0
