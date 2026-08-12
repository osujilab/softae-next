"""Tests for the hardware-arming interlock (core/hardware_safety.py)."""

from __future__ import annotations

import pytest

from softae.core.dropcast import DropcastFormulation, run_dropcast_sweep
from softae.core.hardware_safety import (
    ARM_ENV_VAR,
    HardwareNotArmedError,
    arm_hardware,
    assert_hardware_armed,
    hardware_is_armed,
    real_motion_instruments,
)
from softae.drivers.mock_factory import create_mock_manager


class _FakeRealStage:
    """A non-``Mock*`` class → looks like real hardware to the interlock."""

    name = "stage"
    is_connected = False


class _FakeManager:
    def __init__(self, insts: dict):
        self._insts = insts

    @property
    def names(self):
        return list(self._insts)

    def get(self, name):
        return self._insts[name]


@pytest.fixture(autouse=True)
def _reset_arm(monkeypatch):
    """Every test starts disarmed and with the env var cleared."""
    monkeypatch.delenv(ARM_ENV_VAR, raising=False)
    arm_hardware(False)
    yield
    arm_hardware(False)


# ── Detection ────────────────────────────────────────────────────────────────

def test_mock_manager_has_no_real_motion_instruments():
    mgr = create_mock_manager(config={})
    assert real_motion_instruments(mgr) == []
    assert assert_hardware_armed(mgr) == []  # no-op, no raise


def test_detects_nonmock_motion_instrument():
    mgr = _FakeManager({"stage": _FakeRealStage()})
    assert real_motion_instruments(mgr) == ["stage"]


# ── Arming ───────────────────────────────────────────────────────────────────

def test_unarmed_real_hardware_raises():
    mgr = _FakeManager({"stage": _FakeRealStage()})
    with pytest.raises(HardwareNotArmedError):
        assert_hardware_armed(mgr, action="test move")


def test_env_var_arms(monkeypatch):
    monkeypatch.setenv(ARM_ENV_VAR, "1")
    assert hardware_is_armed()
    mgr = _FakeManager({"stage": _FakeRealStage()})
    assert assert_hardware_armed(mgr) == ["stage"]  # armed → no raise


def test_process_flag_arms():
    arm_hardware(True)
    assert hardware_is_armed()
    mgr = _FakeManager({"stage": _FakeRealStage()})
    assert assert_hardware_armed(mgr) == ["stage"]


def test_env_var_falsey_does_not_arm(monkeypatch):
    monkeypatch.setenv(ARM_ENV_VAR, "0")
    assert not hardware_is_armed()


# ── Integration: dropcast + executor refuse unarmed real motion ──────────────

@pytest.mark.asyncio
async def test_dropcast_blocks_unarmed_real_hardware():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    original = mgr.get("stage")
    # Make the manager present a "real" stage so the interlock engages.
    mgr._instruments["stage"] = _FakeRealStage()  # noqa: SLF001 (test)
    try:
        with pytest.raises(HardwareNotArmedError):
            await run_dropcast_sweep(
                (21, 22), DropcastFormulation(ids=(0, 1), vols=(0.1, 0.1), time_scale=0.0),
                manager=mgr, pcb_name="SoftAE_EIS_4Stripe",
            )
    finally:
        mgr._instruments["stage"] = original  # noqa: SLF001
        await mgr.disconnect_all()


@pytest.mark.asyncio
async def test_mock_dropcast_still_runs_without_arming():
    """The interlock must never block simulation."""
    result = await run_dropcast_sweep(
        (21,), DropcastFormulation(ids=(0,), vols=(0.1,), time_scale=0.0),
        pcb_name="SoftAE_EIS_4Stripe",
    )
    assert result.executed is True


# ── An unreadable probe must not read as "no hardware" ───────────────────────
#
# The hole this closes: `real_motion_instruments` caught every exception and
# continued, so a driver layer that raised returned []. An empty list is
# indistinguishable from "all mocks", so `assert_hardware_armed` evaluated
# `if real and not armed`, found `real` falsy, raised nothing, and let the stage move
# unarmed. `gui.app` used the same call to decide whether to arm, so it also declined —
# the two compounding into "nothing armed, nothing blocked, motion proceeds".


class _ExplodingManager:
    """A manager whose instrument listing raises — a broken driver layer."""

    @property
    def names(self):
        raise RuntimeError("driver layer is broken")

    def get(self, name):
        raise RuntimeError("driver layer is broken")


class _FakeMockSyringe:
    name = "syringe"
    is_connected = False


# `Mock*` prefix is how the interlock recognises a simulator.
_FakeMockSyringe.__name__ = "MockSyringe"


class _PartlyExplodingManager:
    """Enumerates fine, but one motion instrument cannot be fetched."""

    @property
    def names(self):
        return ["stage", "syringe"]

    def get(self, name):
        if name == "stage":
            raise RuntimeError("stage driver raised on access")
        return _FakeMockSyringe()


def test_an_unenumerable_manager_reports_unreadable_not_empty():
    from softae.core.hardware_safety import MOTION_INSTRUMENTS, probe_motion

    real, unreadable = probe_motion(_ExplodingManager())
    assert real == []
    assert set(unreadable) == set(MOTION_INSTRUMENTS)


def test_unarmed_motion_is_refused_when_the_probe_cannot_be_read():
    # The actual defect: this used to return [] and raise nothing.
    with pytest.raises(HardwareNotArmedError) as exc:
        assert_hardware_armed(_ExplodingManager(), action="move the stage")
    assert "Could not determine" in str(exc.value)


def test_arming_still_overrides_an_unreadable_probe(monkeypatch):
    # Arming is an explicit declaration of intent, so it outranks a failed check.
    monkeypatch.setenv(ARM_ENV_VAR, "1")
    assert assert_hardware_armed(_ExplodingManager()) == []


def test_one_broken_instrument_is_enough_to_refuse():
    # Partial knowledge is not safety: the stage could not be classified, so it is
    # treated as able to move even though its neighbour is a confirmed mock.
    from softae.core.hardware_safety import probe_motion

    real, unreadable = probe_motion(_PartlyExplodingManager())
    assert unreadable == ["stage"]
    with pytest.raises(HardwareNotArmedError):
        assert_hardware_armed(_PartlyExplodingManager())


def test_real_motion_instruments_keeps_its_old_contract():
    # Four callers and the GUI's auto-arm depend on this returning plain names.
    assert real_motion_instruments(_FakeManager({"stage": _FakeRealStage()})) == ["stage"]
    assert real_motion_instruments(_ExplodingManager()) == []


def test_a_healthy_mock_manager_is_still_a_no_op():
    # The regression guard: failing closed must not make simulation refuse.
    from softae.core.hardware_safety import probe_motion

    mgr = create_mock_manager(config={})
    real, unreadable = probe_motion(mgr)
    assert real == [] and unreadable == []
    assert assert_hardware_armed(mgr) == []
