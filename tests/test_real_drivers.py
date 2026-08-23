"""Tests for the real hardware driver wrappers.

These tests mock the underlying I/O libraries (``minimalmodbus``,
``serial``, ``hid``) so they run **without** physical hardware.
They verify:
  - config wiring (port, baud, address, registers)
  - safety limit enforcement
  - retry / reconnect logic via ``_with_retry``
  - PID control loop lifecycle (start/stop)
  - factory fallback (real → mock)
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import tomllib
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from softae.errors import CommunicationError, ConnectionError_, SafetyError
from softae.server.base_instrument import InstrumentState

REPO = Path(__file__).resolve().parents[1]

# Pre-import AsyncESPico at module level so it stays in sys.modules even
# when patch.dict() contexts inside fixtures restore the snapshot.
from softae.drivers.async_espico import AsyncESPico as _AsyncESPico  # noqa: F401


# ─── Helpers ─────────────────────────────────────────────────────────────────

def run(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncTempController
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncTempController:
    """Unit tests for :class:`AsyncTempController` with mocked Modbus I/O."""

    @pytest.fixture()
    def mock_minimalmodbus(self):
        """Provide a fake ``minimalmodbus`` module."""
        mm = types.ModuleType("minimalmodbus")
        mm.MODE_RTU = "rtu"
        mm.NoResponseError = type("NoResponseError", (IOError,), {})

        mock_inst = MagicMock()
        mock_inst.serial.baudrate = 115200
        mock_inst.close_port_after_each_call = False
        mm.Instrument = MagicMock(return_value=mock_inst)

        return mm, mock_inst

    @pytest.fixture()
    def controller(self, mock_minimalmodbus):
        mm_module, mock_inst = mock_minimalmodbus
        with patch.dict("sys.modules", {"minimalmodbus": mm_module}):
            from softae.drivers.async_temp_controller import AsyncTempController
            tc = AsyncTempController(
                name="tc_test",
                config={
                    "port": "COM99",
                    "baud": 9600,
                    "addr": 2,
                    "reg_sp": 10,
                    "reg_pv": 11,
                    "max_temp": 150.0,
                },
            )
            run(tc.connect())
        return tc, mock_inst

    def test_config_wiring(self, controller):
        tc, _ = controller
        assert tc._port == "COM99"
        assert tc._baud == 9600
        assert tc._addr == 2
        assert tc._reg_sp == 10
        assert tc._reg_pv == 11

    def test_get_sp_reads_register(self, controller):
        tc, mock_inst = controller
        mock_inst.read_register.return_value = 250  # 25.0 °C × 10
        assert tc.get_sp() == 25.0
        mock_inst.read_register.assert_called_with(10)

    def test_write_sp_within_limit(self, controller):
        tc, mock_inst = controller
        mock_inst.read_register.return_value = 200
        tc.write_sp(T_SP=80.0, print_flag=0)
        mock_inst.write_register.assert_called_with(10, 800)

    def test_write_sp_exceeds_limit_raises(self, controller):
        tc, mock_inst = controller
        mock_inst.read_register.return_value = 200
        with pytest.raises(SafetyError, match="exceeds max"):
            tc.write_sp(T_SP=200.0, print_flag=0)

    def test_get_pv_surf_no_channel(self, controller):
        tc, _ = controller
        # No daq_channel configured → NaN
        assert math.isnan(tc.get_pv_surf())

    def test_retry_raises_after_timeout(self, mock_minimalmodbus):
        mm_module, mock_inst = mock_minimalmodbus
        with patch.dict("sys.modules", {"minimalmodbus": mm_module}):
            from softae.drivers.async_temp_controller import AsyncTempController
            tc = AsyncTempController(name="tc_retry", config={"max_temp": 300})
            run(tc.connect())
            mock_inst.read_register.side_effect = TimeoutError("no response")
            with pytest.raises(CommunicationError, match="failed after"):
                tc._with_retry(mock_inst.read_register, 0, timeout=0.2, backoff_base=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncRHController
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncRHController:
    """Unit tests for :class:`AsyncRHController` with mocked serial I/O."""

    @pytest.fixture()
    def mock_serial(self):
        mm = types.ModuleType("serial")
        mock_ser = MagicMock()
        mock_ser.is_open = True
        mm.Serial = MagicMock(return_value=mock_ser)
        return mm, mock_ser

    @pytest.fixture()
    def mock_pid(self):
        mm = types.ModuleType("simple_pid")
        pid_inst = MagicMock()
        pid_inst.return_value = 0.5  # PID output
        mm.PID = MagicMock(return_value=pid_inst)
        return mm, pid_inst

    @pytest.fixture()
    def rh_ctrl(self, mock_serial, mock_pid):
        ser_mod, mock_ser = mock_serial
        pid_mod, pid_inst = mock_pid

        fake_rh = 55.0

        with patch.dict("sys.modules", {"serial": ser_mod, "simple_pid": pid_mod}):
            from softae.drivers.async_rh_controller import AsyncRHController
            ctrl = AsyncRHController(
                name="rh_test",
                config={"port": "COM42", "baud": 9600, "max_rh": 90.0},
                rh_reader=lambda: fake_rh,
            )
            run(ctrl.connect())
        return ctrl, mock_ser, pid_inst

    def test_set_setpoint_exceeds_limit(self, rh_ctrl):
        ctrl, _, _ = rh_ctrl
        with pytest.raises(SafetyError, match="exceeds limit"):
            ctrl.set_setpoint(95.0)

    def test_start_stop_lifecycle(self, rh_ctrl):
        ctrl, _, _ = rh_ctrl
        ctrl.start()
        assert ctrl._running is True
        assert ctrl._thread is not None
        ctrl.stop()
        assert ctrl._running is False

    def test_get_H_reads_sensor_when_loop_stopped(self, rh_ctrl):
        ctrl, _, _ = rh_ctrl
        # Now get_H() does a live read via _rh_reader when PID isn't running
        h = ctrl.get_H()
        assert h == 55.0  # the injected lambda always returns 55.0

    # ── safe_off: the entry point the park calls ─────────────────────────────
    #
    # The serial spy throughout is ``mock_ser.write``, whose call args are the
    # encoded duty strings the Trinket receives.

    ZERO = b"0.0000\n"

    def test_safe_off_sends_duty_zero_when_start_was_never_called(self, rh_ctrl):
        """The hole the whole entry point exists for.

        ``stop()`` returns immediately when ``_running`` is False, so a process
        that connected and died before ``start()`` sends nothing at all and
        leaves the Trinket at a previous session's duty.
        """
        ctrl, mock_ser, _ = rh_ctrl
        assert ctrl._running is False
        mock_ser.write.reset_mock()

        ctrl.safe_off()

        assert mock_ser.write.call_count == 1
        assert mock_ser.write.call_args[0][0] == self.ZERO
        assert ctrl.last_safe_off_error == ""

    def test_safe_off_zeroes_the_stored_setpoint(self, rh_ctrl):
        """A later bare ``start()`` must not resume the pre-park target."""
        ctrl, _, pid_inst = rh_ctrl
        ctrl.set_setpoint(45.0)

        ctrl.safe_off()

        assert ctrl._setpoint == 0.0
        assert pid_inst.setpoint == 0.0

    def test_safe_off_stops_a_running_loop(self, rh_ctrl):
        ctrl, _, _ = rh_ctrl
        ctrl.start()
        assert ctrl._running is True

        ctrl.safe_off()

        assert ctrl._running is False
        assert ctrl._thread is None

    def test_safe_off_after_a_running_loop_still_writes_zero_itself(self, rh_ctrl):
        """A duplicate zero from the exiting thread is explicitly tolerated.

        What matters is that the *last* thing on the wire is the zero this
        method wrote — the thread's own write cannot be relied on, because a
        wedged loop's ``join`` expires and ``stop()`` returns anyway.
        """
        ctrl, mock_ser, _ = rh_ctrl
        ctrl.start()
        time.sleep(0.05)                      # let the loop take a tick

        ctrl.safe_off()

        assert mock_ser.write.call_args[0][0] == self.ZERO
        assert ctrl.last_safe_off_error == ""

    def test_safe_off_does_not_raise_when_the_write_fails(self, rh_ctrl):
        """Never-raise is the park's contract; the failure travels in the attribute."""
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.side_effect = OSError("port went away")

        ctrl.safe_off()                        # must not raise

        assert ctrl.last_safe_off_error != ""
        assert "port went away" in ctrl.last_safe_off_error

    def test_safe_off_does_not_raise_when_the_port_is_none(self, rh_ctrl):
        """No transport still zeroes the setpoint, and still says so."""
        ctrl, _, _ = rh_ctrl
        ctrl.set_setpoint(45.0)
        ctrl._serial = None

        ctrl.safe_off()                        # must not raise

        assert ctrl._setpoint == 0.0
        assert ctrl.last_safe_off_error != ""

    def test_safe_off_clears_a_previous_error_on_a_later_success(self, rh_ctrl):
        """Per-call, not sticky — a stale error would fail the *next* park."""
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.side_effect = OSError("port went away")
        ctrl.safe_off()
        assert ctrl.last_safe_off_error != ""

        mock_ser.write.side_effect = None
        ctrl.safe_off()

        assert ctrl.last_safe_off_error == ""

    def test_safe_off_leaves_the_port_open(self, rh_ctrl):
        """It is a safe state, not a disconnect: the caller owns teardown."""
        ctrl, _, _ = rh_ctrl

        ctrl.safe_off()

        assert ctrl._serial is not None
        assert ctrl._state is InstrumentState.CONNECTED

    def test_safe_off_is_idempotent(self, rh_ctrl):
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.reset_mock()

        ctrl.safe_off()
        ctrl.safe_off()                        # must not raise

        assert mock_ser.write.call_count == 2
        assert all(c[0][0] == self.ZERO for c in mock_ser.write.call_args_list)
        assert ctrl._running is False
        assert ctrl._setpoint == 0.0

    # ── safe_dry: the dry-purge park ─────────────────────────────────────────
    #
    # `ctrl` near 0 is DRY air and `ctrl == 1` is fully humid (bench-verified
    # 2026-08-21). `ctrl == 0` EXACTLY is a firmware special case that shuts both
    # Aalborg PSVs, so `safe_off` leaves no flow at all and room air wins — which
    # is why a clean shutdown after a long low-RH hold collapses the chamber from
    # 10 %RH to ~50 %RH in tens of seconds. `safe_dry` commands `out_min` instead
    # and lets the Trinket's own ~25 s deadman close the valves.

    DRY = b"0.0100\n"

    @pytest.fixture()
    def rh_factory(self, mock_serial, mock_pid):
        """Build a connected controller over an arbitrary config section."""
        ser_mod, mock_ser = mock_serial
        pid_mod, pid_inst = mock_pid

        def _make(**config):
            section = {"port": "COM42", "baud": 9600, "max_rh": 90.0}
            section.update(config)
            with patch.dict("sys.modules",
                            {"serial": ser_mod, "simple_pid": pid_mod}):
                from softae.drivers.async_rh_controller import AsyncRHController
                ctrl = AsyncRHController(name="rh_test", config=section,
                                         rh_reader=lambda: 55.0)
                run(ctrl.connect())
            return ctrl, mock_ser, pid_inst

        return _make

    def test_safe_dry_writes_out_min_and_never_zero(self, rh_ctrl):
        """The whole point: duty 0 shuts both valves, `out_min` keeps dry air on."""
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        assert mock_ser.write.call_count == 1
        assert mock_ser.write.call_args[0][0] == self.DRY
        assert self.ZERO not in [c[0][0] for c in mock_ser.write.call_args_list]
        assert ctrl.last_safe_dry_error == ""
        assert ctrl.last_safe_dry_duty == pytest.approx(0.01)

    def test_safe_dry_uses_the_configured_out_min_not_the_code_default(
            self, rh_factory):
        """A hardcoded 0.01 would silently disagree with a retuned rig."""
        ctrl, mock_ser, _ = rh_factory(out_min=0.05)
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        assert mock_ser.write.call_args[0][0] == b"0.0500\n"
        assert ctrl.last_safe_dry_duty == pytest.approx(0.05)

    def test_safe_dry_suppresses_the_loops_exit_zero(self, rh_ctrl):
        """No zero may precede the dry duty, or the valves slam shut and reopen.

        The exiting PID thread used to write ``0.0`` unconditionally. The Trinket
        treats ``ctrl == 0`` as its own shutoff case, so that frame is visible to
        the firmware — not merely redundant.
        """
        ctrl, mock_ser, _ = rh_ctrl
        ctrl.start()
        time.sleep(0.05)                      # let the loop take a tick
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        written = [c[0][0] for c in mock_ser.write.call_args_list]
        assert self.ZERO not in written
        assert written[-1] == self.DRY
        assert ctrl._running is False
        assert ctrl._thread is None

    def test_safe_dry_writes_the_duty_itself_when_the_loop_never_ran(self, rh_ctrl):
        """The belt-and-braces case ``safe_off`` documents, and the one where it
        is not merely a duplicate but the *only* write.

        ``_running`` True with no thread reproduces what a wedged loop leaves
        behind: ``_stop_pid_loop`` returns cleanly having had nothing to join, so
        no thread ever reaches an exit write and the method's own write is all
        the Trinket will ever see.
        """
        ctrl, mock_ser, _ = rh_ctrl
        ctrl._running = True
        ctrl._thread = None
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        assert mock_ser.write.call_count == 1
        assert mock_ser.write.call_args[0][0] == self.DRY

    def test_safe_dry_zeroes_the_stored_setpoint(self, rh_ctrl):
        """Exactly as ``safe_off`` does — a later bare ``start()`` must not
        resume the pre-park target, which ``_pid_loop`` re-reads every tick."""
        ctrl, _, pid_inst = rh_ctrl
        ctrl.set_setpoint(45.0)

        ctrl.safe_dry()

        assert ctrl._setpoint == 0.0
        assert pid_inst.setpoint == 0.0

    def test_safe_dry_never_strands_the_last_pid_output(self, rh_factory):
        """An exit mid-approach to a WET setpoint must not leave that duty on.

        The PID here returns 0.9 — a humidifying duty. Parking to "the last
        output" would leave the chamber being actively wetted for the deadman's
        whole window, which is the failure the park exists to prevent reached by
        a different road.
        """
        ctrl, mock_ser, pid_inst = rh_factory()
        pid_inst.return_value = 0.9
        ctrl.set_setpoint(80.0)
        ctrl.start()
        time.sleep(0.05)
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        written = [c[0][0] for c in mock_ser.write.call_args_list]
        assert b"0.9000\n" not in written
        assert written[-1] == self.DRY

    def test_a_plain_stop_after_a_safe_dry_still_exits_on_zero(self, rh_ctrl):
        """The exit duty is per-stop, not sticky: a dry purge must not silently
        convert every later ``stop()`` into one."""
        ctrl, mock_ser, _ = rh_ctrl
        ctrl.start()
        ctrl.safe_dry()

        ctrl.start()
        time.sleep(0.05)
        mock_ser.write.reset_mock()
        ctrl.stop()

        assert mock_ser.write.call_args[0][0] == self.ZERO

    def test_safe_off_after_a_safe_dry_still_ends_on_zero(self, rh_ctrl):
        """The pinned safe state is untouched by the new sibling's existence."""
        ctrl, mock_ser, _ = rh_ctrl
        ctrl.start()
        ctrl.safe_dry()

        ctrl.start()
        time.sleep(0.05)
        mock_ser.write.reset_mock()
        ctrl.safe_off()

        assert mock_ser.write.call_args[0][0] == self.ZERO
        assert ctrl.last_safe_off_error == ""

    def test_disconnect_after_a_safe_dry_writes_nothing_over_the_dry_duty(
            self, rh_ctrl):
        """The mechanism ``eis-validate --end-state hold`` rests on entirely.

        ``disconnect()`` calls ``_stop_pid_loop()`` with the default exit duty
        ``0.0`` — the firmware's valve shutoff — so any exit that merely printed
        a different message would still hand the chamber a shutoff a line later.
        ``safe_dry`` stops the loop *itself*, so by the time ``disconnect`` runs,
        ``_stop_pid_loop`` finds ``_running`` already ``False`` and returns having
        written nothing. ``out_min`` is the last thing the Trinket saw, and the
        ~25 s deadman — not the host — is what closes the valves.

        Unlike the two tests above, nothing restarts the loop in between: that is
        the difference between "the exit duty is not sticky" and "the dry duty
        survives the port closing".
        """
        ctrl, mock_ser, _ = rh_ctrl
        ctrl.start()
        time.sleep(0.05)
        ctrl.safe_dry()
        assert mock_ser.write.call_args[0][0] == self.DRY
        mock_ser.write.reset_mock()

        run(ctrl.disconnect())

        assert mock_ser.write.call_count == 0
        assert ctrl._running is False

    def test_safe_dry_falls_back_to_safe_off_when_out_min_is_zero(self, rh_factory):
        """A "dry purge" at duty 0 is a valve shutoff wearing the wrong name.

        The fallback is the safe direction, so it is taken — and it is *reported*,
        because a config typo that silently disables the dry purge is met months
        later as unexplained RH collapses with nothing naming the cause.
        """
        ctrl, mock_ser, _ = rh_factory(out_min=0.0)
        ctrl.set_setpoint(45.0)
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        assert mock_ser.write.call_args[0][0] == self.ZERO
        assert ctrl._setpoint == 0.0
        assert ctrl.last_safe_dry_duty == 0.0
        assert "out_min" in ctrl.last_safe_dry_error
        assert "zeroed instead" in ctrl.last_safe_dry_error

    def test_safe_dry_falls_back_to_safe_off_when_out_min_is_negative(
            self, rh_factory):
        ctrl, mock_ser, _ = rh_factory(out_min=-0.5)
        mock_ser.write.reset_mock()

        ctrl.safe_dry()

        assert mock_ser.write.call_args[0][0] == self.ZERO
        assert "out_min" in ctrl.last_safe_dry_error

    def test_safe_dry_reports_a_fallback_that_also_failed(self, rh_factory):
        """Both halves of the bad news, not just the first."""
        ctrl, _, _ = rh_factory(out_min=0.0)
        ctrl._serial = None

        ctrl.safe_dry()

        assert "out_min" in ctrl.last_safe_dry_error
        assert "fallback to safe_off also failed" in ctrl.last_safe_dry_error

    def test_safe_dry_does_not_raise_when_the_write_fails(self, rh_ctrl):
        """Never-raise is the park's contract; the failure travels in the attribute."""
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.side_effect = OSError("port went away")

        ctrl.safe_dry()                        # must not raise

        assert "port went away" in ctrl.last_safe_dry_error
        assert ctrl.last_safe_dry_duty == 0.0

    def test_safe_dry_does_not_raise_when_the_port_is_none(self, rh_ctrl):
        ctrl, _, _ = rh_ctrl
        ctrl.set_setpoint(45.0)
        ctrl._serial = None

        ctrl.safe_dry()                        # must not raise

        assert ctrl._setpoint == 0.0
        assert "no serial transport" in ctrl.last_safe_dry_error

    def test_safe_dry_clears_a_previous_error_on_a_later_success(self, rh_ctrl):
        ctrl, mock_ser, _ = rh_ctrl
        mock_ser.write.side_effect = OSError("port went away")
        ctrl.safe_dry()
        assert ctrl.last_safe_dry_error != ""

        mock_ser.write.side_effect = None
        ctrl.safe_dry()

        assert ctrl.last_safe_dry_error == ""
        assert ctrl.last_safe_dry_duty == pytest.approx(0.01)

    def test_safe_dry_leaves_the_port_open(self, rh_ctrl):
        """A safe state, not a disconnect — and the port must stay open anyway,
        or a later float could not reset the firmware's deadman."""
        ctrl, _, _ = rh_ctrl

        ctrl.safe_dry()

        assert ctrl._serial is not None
        assert ctrl._state is InstrumentState.CONNECTED


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncRHController — config key spellings ([a69])
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncRHControllerConfigKeys:
    """The five dual-spelled ``[instruments.rh_controller]`` keys.

    ``softae_config.toml`` documents ``trinket_port`` / ``trinket_baud`` /
    ``pid_kp`` / ``pid_ki`` / ``pid_kd``; the driver historically read only
    ``port`` / ``baud`` / ``kp`` / ``ki`` / ``kd``, so **none of the five
    reached it** and the rig ran the code-default ``kp = 0.008`` rather than the
    operator's tuned ``pid_kp = 0.007``.  See SESSION_MAIL ``[a69]`` and
    ``rh_safe_state_and_hold_spec.md`` §9.2.
    """

    @pytest.fixture()
    def build(self):
        """``(build, pid_cls)`` — ``build(config, connect=…)`` → controller.

        Serial and ``simple_pid`` are faked exactly as in
        :class:`TestAsyncRHController`; ``pid_cls`` is the ``PID`` constructor
        spy, which is where the gains actually have to land.
        """
        ser_mod = types.ModuleType("serial")
        mock_ser = MagicMock()
        mock_ser.is_open = True
        ser_mod.Serial = MagicMock(return_value=mock_ser)

        pid_mod = types.ModuleType("simple_pid")
        pid_inst = MagicMock()
        pid_inst.return_value = 0.5
        pid_mod.PID = MagicMock(return_value=pid_inst)

        def _build(config, *, connect=False):
            with patch.dict("sys.modules", {"serial": ser_mod, "simple_pid": pid_mod}):
                from softae.drivers.async_rh_controller import AsyncRHController
                ctrl = AsyncRHController(
                    name="rh_cfg_test",
                    config=dict(config),
                    rh_reader=lambda: 55.0,
                )
                if connect:
                    run(ctrl.connect())
            return ctrl

        return _build, pid_mod.PID

    LONG = {
        "trinket_port": "COM77",
        "trinket_baud": 57600,
        "pid_kp": 0.007,
        "pid_ki": 0.002,
        "pid_kd": 0.04,
    }

    def test_rh_controller_long_spellings_reach_the_driver(self, build):
        """The defect itself: the TOML's own spellings must be read."""
        make, pid_cls = build
        ctrl = make(self.LONG, connect=True)

        assert ctrl._port == "COM77"
        assert ctrl._baud == 57600
        assert ctrl._kp == 0.007
        assert ctrl._ki == 0.002
        assert ctrl._kd == 0.04

        # …and reaching the driver is not enough — the PID must be built with them.
        kwargs = pid_cls.call_args.kwargs
        assert (kwargs["Kp"], kwargs["Ki"], kwargs["Kd"]) == (0.007, 0.002, 0.04)

    def test_rh_controller_short_names_still_reach_the_driver(self, build):
        """Every existing caller and fixture passes the short names."""
        make, _ = build
        ctrl = make({"port": "COM42", "baud": 9600, "kp": 0.01, "ki": 0.003, "kd": 0.06})

        assert ctrl._port == "COM42"
        assert ctrl._baud == 9600
        assert ctrl._kp == 0.01
        assert ctrl._ki == 0.003
        assert ctrl._kd == 0.06

    def test_rh_controller_long_spelling_wins_over_short_name(self, build):
        """Precedence is documented-long → short alias → code default."""
        make, _ = build
        ctrl = make({
            **self.LONG,
            "port": "COM1", "baud": 9600, "kp": 0.5, "ki": 0.5, "kd": 0.5,
        })

        assert ctrl._port == "COM77"
        assert ctrl._baud == 57600
        assert ctrl._kp == 0.007
        assert ctrl._ki == 0.002
        assert ctrl._kd == 0.04

    def test_rh_controller_empty_config_yields_code_defaults(self, build):
        """Neither spelling present → the values baked into ``__init__``."""
        make, _ = build
        ctrl = make({})

        assert ctrl._port == "COM11"
        assert ctrl._baud == 115200
        assert ctrl._kp == 0.008
        assert ctrl._ki == 0.0015
        assert ctrl._kd == 0.05

    def test_rh_controller_shipped_config_delivers_the_tuned_gain(self, build):
        """The regression this claim exists for, against the file on disk.

        ``factory.py`` passes the section raw, so the literal
        ``[instruments.rh_controller]`` table is what the driver receives on the
        rig.  ``_kp == 0.007`` is the whole point: before [a69] this asserted
        ``0.008``, silently, for as long as the TOML has been spelled that way.
        """
        make, _ = build
        section = tomllib.loads(
            (REPO / "softae_config.toml").read_text(encoding="utf-8")
        )["instruments"]["rh_controller"]
        ctrl = make(section)

        assert ctrl._kp == 0.007
        assert ctrl._ki == section["pid_ki"]
        assert ctrl._kd == section["pid_kd"]
        assert ctrl._port == section["trinket_port"]
        assert ctrl._baud == section["trinket_baud"]
        # The two keys that were already spelled correctly must not regress.
        assert ctrl._max_consecutive_failures == section["max_consecutive_failures"]
        assert ctrl._max_stale_s == section["max_stale_s"]


# ═══════════════════════════════════════════════════════════════════════════════
# MockRHController — safe_off parity
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockRHControllerSafeOff:
    """The mock is not a formality here.

    ``create_manager`` falls back to :class:`MockRHController` for
    ``rh_controller`` and ``safe_park`` cannot tell it from a real driver, so
    every mock-backed park and every ``--mock`` tool run is graded by this class.
    """

    @pytest.fixture()
    def mock_rh(self):
        from softae.drivers.mock_rh_controller import MockRHController
        ctrl = MockRHController(name="rh_mock")
        run(ctrl.connect())
        return ctrl

    def test_mock_safe_off_zeroes_duty_and_setpoint_and_running(self, mock_rh):
        mock_rh.set_setpoint(45.0)
        mock_rh.start()
        mock_rh.status()                       # let the sim raise the duty

        mock_rh.safe_off()

        assert mock_rh._running is False
        assert mock_rh._duty == 0.0
        assert mock_rh._setpoint == 0.0

    def test_mock_safe_off_is_a_superset_of_stop(self, mock_rh):
        """``stop()`` leaves the setpoint standing; ``safe_off()`` does not."""
        mock_rh.set_setpoint(45.0)
        mock_rh.start()
        mock_rh.stop()
        assert mock_rh._setpoint == 45.0

        mock_rh.set_setpoint(45.0)
        mock_rh.safe_off()
        assert mock_rh._setpoint == 0.0

    def test_mock_stays_at_zero_duty_after_safe_off(self, mock_rh):
        """``_update_sim`` gates on ``_running``, so nothing re-raises the duty."""
        mock_rh.set_setpoint(45.0)
        mock_rh.start()

        mock_rh.safe_off()

        assert mock_rh.status()["duty_cycle"] == 0.0
        assert mock_rh.status()["duty_cycle"] == 0.0

    def test_both_drivers_expose_safe_off_with_the_same_signature(self):
        """The parity rule, made mechanical rather than asserted in prose."""
        from softae.drivers.async_rh_controller import AsyncRHController
        from softae.drivers.mock_rh_controller import MockRHController

        assert (inspect.signature(AsyncRHController.safe_off)
                == inspect.signature(MockRHController.safe_off))
        assert AsyncRHController.last_safe_off_error == ""
        assert MockRHController.last_safe_off_error == ""

    def test_both_drivers_expose_safe_dry_with_the_same_signature(self):
        from softae.drivers.async_rh_controller import AsyncRHController
        from softae.drivers.mock_rh_controller import MockRHController

        assert (inspect.signature(AsyncRHController.safe_dry)
                == inspect.signature(MockRHController.safe_dry))
        for cls in (AsyncRHController, MockRHController):
            assert cls.last_safe_dry_error == ""
            assert cls.last_safe_dry_duty == 0.0

    def test_mock_safe_dry_holds_out_min_rather_than_zero(self, mock_rh):
        """The mock is not a formality: ``safe_park`` cannot tell it from a real
        driver, so a ``safe_dry`` that quietly behaved like ``safe_off`` would
        make every ``--mock`` park pass while proving the opposite of the point."""
        mock_rh.set_setpoint(45.0)
        mock_rh.start()
        mock_rh.status()                       # let the sim raise the duty

        mock_rh.safe_dry()

        assert mock_rh._running is False
        assert mock_rh._duty == pytest.approx(0.01)
        assert mock_rh._setpoint == 0.0
        assert mock_rh.last_safe_dry_duty == pytest.approx(0.01)
        assert mock_rh.last_safe_dry_error == ""

    def test_mock_safe_dry_is_distinguishable_from_safe_off(self, mock_rh):
        """The one assertion a no-op implementation could never satisfy."""
        mock_rh.start()
        mock_rh.safe_off()
        zeroed = mock_rh._duty

        mock_rh.start()
        mock_rh.safe_dry()

        assert zeroed == 0.0
        assert mock_rh._duty > zeroed

    def test_mock_stays_at_out_min_after_safe_dry(self, mock_rh):
        """``_update_sim`` gates on ``_running``, so nothing overwrites the duty."""
        mock_rh.set_setpoint(45.0)
        mock_rh.start()

        mock_rh.safe_dry()

        assert mock_rh.status()["duty_cycle"] == pytest.approx(0.01)
        assert mock_rh.status()["duty_cycle"] == pytest.approx(0.01)

    def test_mock_safe_dry_reads_the_configured_out_min(self):
        from softae.drivers.mock_rh_controller import MockRHController

        ctrl = MockRHController(name="rh_mock", config={"out_min": 0.07})
        run(ctrl.connect())

        ctrl.safe_dry()

        assert ctrl._duty == pytest.approx(0.07)

    def test_mock_safe_dry_falls_back_to_safe_off_when_out_min_is_zero(self):
        """Same decision, same words as the real driver — the mock has to grade
        a misconfigured rig the way the rig would."""
        from softae.drivers.mock_rh_controller import MockRHController

        ctrl = MockRHController(name="rh_mock", config={"out_min": 0.0})
        run(ctrl.connect())
        ctrl.set_setpoint(45.0)
        ctrl.start()

        ctrl.safe_dry()

        assert ctrl._duty == 0.0
        assert ctrl._running is False
        assert ctrl._setpoint == 0.0
        assert "out_min" in ctrl.last_safe_dry_error
        assert "zeroed instead" in ctrl.last_safe_dry_error


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncHTSensor
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncHTSensor:
    """Unit tests for :class:`AsyncHTSensor` with mocked HID/I²C."""

    def test_get_T_not_connected(self):
        from softae.drivers.async_ht_sensor import AsyncHTSensor
        ht = AsyncHTSensor(name="ht_test")
        with pytest.raises(CommunicationError, match="not connected"):
            ht.get_T()


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncCamera
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncCamera:
    """Unit tests for :class:`AsyncCamera` with mocked ThorLabs SDK."""

    @pytest.fixture()
    def mock_thorlabs(self):
        """Create fake ``thorlabs_tsi_sdk`` modules."""
        # -- tl_camera module --
        tl_camera = types.ModuleType("thorlabs_tsi_sdk.tl_camera")
        mock_sdk_inst = MagicMock()
        mock_sdk_inst.discover_available_cameras.return_value = ["SN123"]

        # Camera mock with context-manager support
        mock_cam = MagicMock()
        mock_cam.__enter__ = MagicMock(return_value=mock_cam)
        mock_cam.__exit__ = MagicMock(return_value=False)
        mock_cam.camera_sensor_type = "BAYER"
        mock_cam.color_filter_array_phase = 0
        mock_cam.get_color_correction_matrix.return_value = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        mock_cam.get_default_white_balance_matrix.return_value = [1, 1, 1]
        mock_cam.bit_depth = 8
        mock_cam.image_height_pixels = 480
        mock_cam.image_width_pixels = 640
        mock_sdk_inst.open_camera.return_value = mock_cam

        tl_camera.TLCameraSDK = MagicMock(return_value=mock_sdk_inst)
        tl_camera.OPERATION_MODE = MagicMock()

        # -- tl_mono_to_color_processor module --
        tl_color = types.ModuleType("thorlabs_tsi_sdk.tl_mono_to_color_processor")
        mock_processor = MagicMock()
        # Return a flat array matching (H*W*3)
        mock_processor.transform_to_24.return_value = np.zeros(480 * 640 * 3, dtype=np.uint8)
        mock_mtc_sdk = MagicMock()
        mock_mtc_sdk.__enter__ = MagicMock(return_value=mock_mtc_sdk)
        mock_mtc_sdk.__exit__ = MagicMock(return_value=False)
        mock_mtc_ctx = MagicMock()
        mock_mtc_ctx.__enter__ = MagicMock(return_value=mock_processor)
        mock_mtc_ctx.__exit__ = MagicMock(return_value=False)
        mock_mtc_sdk.create_mono_to_color_processor.return_value = mock_mtc_ctx
        tl_color.MonoToColorProcessorSDK = MagicMock(return_value=mock_mtc_sdk)

        # -- root package --
        tl_root = types.ModuleType("thorlabs_tsi_sdk")
        tl_root.tl_camera = tl_camera
        tl_root.tl_mono_to_color_processor = tl_color

        mods = {
            "thorlabs_tsi_sdk": tl_root,
            "thorlabs_tsi_sdk.tl_camera": tl_camera,
            "thorlabs_tsi_sdk.tl_mono_to_color_processor": tl_color,
        }
        return mods, mock_sdk_inst, mock_cam, mock_processor

    @pytest.fixture()
    def camera(self, mock_thorlabs):
        mods, mock_sdk_inst, mock_cam, _ = mock_thorlabs
        with patch.dict("sys.modules", mods), \
             patch("os.add_dll_directory"):
            from softae.drivers.async_camera import AsyncCamera
            cam = AsyncCamera(
                name="cam_test",
                config={"exposure": 0.05, "dll_path": r"C:\fake\dlls"},
            )
            run(cam.connect())
        return cam, mock_sdk_inst, mock_cam

    def test_connect_no_cameras_raises(self, mock_thorlabs):
        mods, mock_sdk_inst, _, _ = mock_thorlabs
        mock_sdk_inst.discover_available_cameras.return_value = []
        with patch.dict("sys.modules", mods), \
             patch("os.add_dll_directory"):
            from softae.drivers.async_camera import AsyncCamera
            cam = AsyncCamera(name="cam_empty", config={"dll_path": r"C:\fake"})
            with pytest.raises(ConnectionError_, match="No ThorLabs cameras"):
                run(cam.connect())

    def test_connect_import_error(self):
        """If ThorLabs SDK is unavailable, connect() raises ConnectionError_."""
        absent = {
            "thorlabs_tsi_sdk": None,
            "thorlabs_tsi_sdk.tl_camera": None,
            "thorlabs_tsi_sdk.tl_mono_to_color_processor": None,
        }
        with patch.dict("sys.modules", absent), \
             patch("os.add_dll_directory"):
            from softae.drivers.async_camera import AsyncCamera
            cam = AsyncCamera(name="cam_noimport", config={"dll_path": r"C:\fake"})
            with pytest.raises(ConnectionError_, match="TSI SDK not installed"):
                run(cam.connect())

    def test_open_close(self, camera):
        cam, mock_sdk_inst, mock_cam = camera
        cam.open()
        mock_sdk_inst.open_camera.assert_called_with("SN123")
        cam.close()
        mock_cam.dispose.assert_called()

    def test_acquire_frame(self, camera, mock_thorlabs):
        cam, _, mock_cam = camera
        mods, _, _, _ = mock_thorlabs
        cam.open()
        mock_frame = MagicMock()
        mock_frame.image_buffer = np.zeros(480 * 640, dtype=np.uint8)
        mock_cam.get_pending_frame_or_null.return_value = mock_frame
        with patch.dict("sys.modules", mods):
            arr = cam.acquire_n_frames(1, 0.05)
        assert arr.shape == (480, 640, 3)
        assert cam._frame_count == 1




# ═══════════════════════════════════════════════════════════════════════════════
# AsyncDACSwitch  (covers AsyncLamp alias too — both resolve to the same class)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncDACSwitch:
    """Unit tests for :class:`AsyncDACSwitch` with a mocked MCP4728 DAC.

    ``AsyncLamp`` is an alias for ``AsyncDACSwitch``; importing from either
    path reaches the same class.
    """

    def _connect_with_mock_dac(self, switch):
        """Connect *switch* against mocked board/busio/adafruit_mcp4728 modules.

        Returns ``(mock_dac, mock_mcp)`` for assertions.
        """
        mock_board = types.ModuleType("board")
        mock_board.SCL = object()
        mock_board.SDA = object()

        mock_busio = types.ModuleType("busio")
        mock_busio.I2C = MagicMock(return_value=MagicMock())

        mock_dac = MagicMock()
        mock_dac.channel_a = MagicMock()
        mock_dac.channel_b = MagicMock()
        mock_dac.channel_c = MagicMock()
        mock_dac.channel_d = MagicMock()

        mock_mcp = types.ModuleType("adafruit_mcp4728")
        mock_mcp.MCP4728 = MagicMock(return_value=mock_dac)

        modules = {
            "board": mock_board,
            "busio": mock_busio,
            "adafruit_mcp4728": mock_mcp,
        }
        with patch.dict("sys.modules", modules):
            run(switch.connect())
        return mock_dac, mock_mcp

    @pytest.fixture()
    def switch(self):
        """A generic DAC switch on channel B @ 0x61."""
        from softae.drivers.async_dac_switch import AsyncDACSwitch
        return AsyncDACSwitch(
            name="switch_test",
            config={
                "channel": "B",
                "address": "0x61",
                "v_psu": 5.0,
                "on_volt": 0.0,
                "off_volt": 5.0,
            },
        )

    @pytest.fixture()
    def lamp_alias(self):
        """AsyncLamp alias (imported from async_camera) resolves to AsyncDACSwitch."""
        from softae.drivers.async_camera import AsyncLamp
        return AsyncLamp(
            name="lamp_alias_test",
            config={"channel": "A", "address": "0x60"},
        )

    def test_connect_opens_dac_at_address(self, switch):
        mock_dac, mock_mcp = self._connect_with_mock_dac(switch)
        assert mock_mcp.MCP4728.call_args.kwargs["address"] == 0x61
        assert switch.state is InstrumentState.CONNECTED

    def test_alias_is_same_class(self, lamp_alias, switch):
        """AsyncLamp alias is the same class as AsyncDACSwitch."""
        assert type(lamp_alias) is type(switch)

    def test_on_writes_voltage(self, switch):
        mock_dac, _ = self._connect_with_mock_dac(switch)
        switch.on()
        # channel "B", on_volt = 0 V → 16-bit code 0
        assert mock_dac.channel_b.value == 0
        assert switch._is_on is True

    def test_off_writes_voltage(self, switch):
        mock_dac, _ = self._connect_with_mock_dac(switch)
        switch.off()
        # off_volt = 5 V at full-scale 5 V → 16-bit code 65535
        assert mock_dac.channel_b.value == 65535
        assert switch._is_on is False

    def test_on_no_dac(self, switch):
        """If connect() was never called, on() is a graceful no-op."""
        switch.on()
        assert switch._is_on is True

    def test_set_eeprom_defaults(self, switch):
        """set_eeprom_defaults() drives all four channels to 65535 and saves."""
        mock_dac, _ = self._connect_with_mock_dac(switch)
        switch.set_eeprom_defaults()
        assert mock_dac.channel_a.value == 65535
        assert mock_dac.channel_b.value == 65535
        assert mock_dac.channel_c.value == 65535
        assert mock_dac.channel_d.value == 65535
        mock_dac.save_settings.assert_called_once()

    def test_set_eeprom_defaults_no_dac(self, switch):
        """set_eeprom_defaults() raises CommunicationError if not connected."""
        with pytest.raises(CommunicationError):
            switch.set_eeprom_defaults()

    def test_status_includes_channel(self, switch):
        """status() reports the configured channel."""
        s = switch.status()
        assert s["channel"] == "B"


# ═══════════════════════════════════════════════════════════════════════════════
# Driver Factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriverFactory:
    """Verify the factory falls back to mocks when hardware is unavailable."""

    def test_mock_true_returns_all_mocks(self):
        from softae.drivers.factory import create_manager
        mgr = create_manager(mock=True)
        names = [s["name"] for s in mgr.list_instruments()]
        assert "temp_controller" in names
        assert "rh_controller" in names
        assert "ht_sensor" in names

    def test_auto_falls_back_to_mocks(self):
        """When real hardware imports fail (no serial port), auto-detect uses mocks."""
        from softae.drivers.factory import create_manager
        mgr = create_manager(mock=None, config={})
        names = [s["name"] for s in mgr.list_instruments()]
        # Should have all instruments even without hardware
        assert len(names) >= 10




# ═══════════════════════════════════════════════════════════════════════════════
# AsyncStage
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncStage:
    """Unit tests for :class:`AsyncStage` with mocked PyVISA I/O."""

    @pytest.fixture()
    def mock_pyvisa(self):
        """Provide a fake ``pyvisa`` module with a mock instrument handle."""
        pv = types.ModuleType("pyvisa")
        mock_inst = MagicMock()
        mock_inst.baud_rate = 921600
        mock_rm = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        pv.ResourceManager = MagicMock(return_value=mock_rm)
        return pv, mock_rm, mock_inst

    @pytest.fixture()
    def stage(self, mock_pyvisa):
        pv_mod, mock_rm, mock_inst = mock_pyvisa
        with patch.dict("sys.modules", {"pyvisa": pv_mod}):
            from softae.drivers.async_stage import AsyncStage
            s = AsyncStage(
                name="stage_test",
                config={
                    "port": "ASRL99::INSTR",
                    "baud": 115200,
                    "velocity": 5.0,
                },
            )
            run(s.connect())
        return s, mock_rm, mock_inst

    def test_live_position_queries_axes(self, stage):
        s, _, mock_inst = stage
        mock_inst.query_ascii_values.side_effect = [[12.5], [-3.2]]
        x, y = s.live_position()
        assert x == 12.5
        assert y == -3.2
        assert mock_inst.query_ascii_values.call_count == 2

    def test_move_to_sends_commands(self, stage):
        s, _, mock_inst = stage
        mock_inst.query.return_value = "1\r\n"  # MD? → motion done
        mock_inst.query_ascii_values.return_value = [50.0]
        s.move_to(50.0, 25.0)
        calls = [str(c) for c in mock_inst.write.call_args_list]
        assert any("1PA50.0" in c for c in calls)
        assert any("2PA25.0" in c for c in calls)

    def test_wait_until_idle_timeout(self, stage):
        s, _, mock_inst = stage
        mock_inst.query.return_value = "0\r\n"  # MD? → always moving
        # Should not raise, just warn and return
        s._wait_until_idle(timeout=0.5, poll_interval=0.1)

    def test_connect_applies_comms_settings(self, stage):
        """connect() must configure timeout/termination, not just baud."""
        s, _, mock_inst = stage
        assert mock_inst.timeout == 8000  # default visa_timeout_ms
        assert mock_inst.write_termination == "\r"
        assert mock_inst.read_termination == "\r"


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncSyringe
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncSyringe:
    """Unit tests for :class:`AsyncSyringe` with mocked PyVISA I/O."""

    @pytest.fixture()
    def mock_pyvisa(self):
        pv = types.ModuleType("pyvisa")
        mock_inst = MagicMock()
        mock_inst.baud_rate = 115200
        mock_rm = MagicMock()
        mock_rm.open_resource.return_value = mock_inst
        pv.ResourceManager = MagicMock(return_value=mock_rm)
        return pv, mock_rm, mock_inst

    @pytest.fixture()
    def syringe(self, mock_pyvisa):
        pv_mod, mock_rm, mock_inst = mock_pyvisa
        with patch.dict("sys.modules", {"pyvisa": pv_mod}):
            from softae.drivers.async_syringe import AsyncSyringe
            syr = AsyncSyringe(
                name="syr_test",
                config={
                    "port": "ASRL42::INSTR",
                    "baud": 115200,
                    "diameter": 12.0,
                    "max_rate": 1000.0,
                },
            )
            run(syr.connect())
        return syr, mock_rm, mock_inst

    def test_config_wiring(self, syringe):
        syr, _, _ = syringe
        assert syr._port == "ASRL42::INSTR"
        assert syr._diameter == 12.0
        assert syr._max_rate == 1000.0

    def test_single_pump_writes_commands(self, syringe):
        syr, _, mock_inst = syringe
        syr.single_pump(res_vol=1000, ID=0, rate=5.0, dispense_vol=10.0)
        assert mock_inst.write.call_count >= 5  # svolume + diameter + irate + tvolume + irun

    def test_single_pump_exceeds_rate_limit(self, syringe):
        syr, _, _ = syringe
        with pytest.raises(SafetyError, match="exceeds max"):
            syr.single_pump(res_vol=1000, ID=0, rate=2000.0, dispense_vol=10.0)

    def test_head_flip_toggles_state(self, syringe):
        syr, _, _ = syringe
        assert syr._is_up is True
        syr.head_flip()  # No NI-DAQ channel → toggles in software
        assert syr._is_up is False
        syr.head_flip()
        assert syr._is_up is True

    def test_head_check_with_callback(self, syringe):
        syr, _, _ = syringe
        syr._is_up = True
        syr.head_check(confirm_fn=lambda _: False)  # "no, not retracted" → flip
        assert syr._is_up is False


# ═══════════════════════════════════════════════════════════════════════════════
# AsyncESPico
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsyncESPico:
    """Unit tests for :class:`AsyncESPico` with mocked PalmSens SDK."""

    @pytest.fixture()
    def mock_palmsens(self):
        """Create fake palmsens modules."""
        ps_serial = types.ModuleType("palmsens.serial")
        ps_serial.Serial = MagicMock()
        ps_serial.auto_detect_port = MagicMock(return_value=["COM5"])

        ps_instrument = types.ModuleType("palmsens.instrument")
        mock_device = MagicMock()
        mock_device.get_device_type.return_value = "EmStat Pico"
        mock_device.readlines_until_end.return_value = ["fake_line_1\n", "fake_line_2\n"]
        ps_instrument.Instrument = MagicMock(return_value=mock_device)

        ps_mscript = types.ModuleType("palmsens.mscript")
        ps_mscript.parse_result_lines = MagicMock(return_value="parsed_raw")
        ps_mscript.get_values_by_column = MagicMock(
            side_effect=lambda data, col: [1.0, 2.0, 3.0]
        )

        ps_root = types.ModuleType("palmsens")
        ps_root.serial = ps_serial
        ps_root.instrument = ps_instrument
        ps_root.mscript = ps_mscript

        return {
            "palmsens": ps_root,
            "palmsens.serial": ps_serial,
            "palmsens.instrument": ps_instrument,
            "palmsens.mscript": ps_mscript,
        }, mock_device

    @pytest.fixture()
    def pico(self, mock_palmsens, tmp_path):
        from softae.drivers.async_espico import AsyncESPico
        mods, mock_device = mock_palmsens
        p = AsyncESPico(
            name="pico_test",
            config={"port": "auto", "output_dir": str(tmp_path)},
        )
        with patch.dict("sys.modules", mods):
            run(p.connect())
        return p, mock_device, mods

    def test_connect_auto_detect(self, pico):
        p, _, _ = pico
        assert p._state == InstrumentState.CONNECTED
        assert p._resolved_port == "COM5"

    def test_connect_explicit_port(self, mock_palmsens, tmp_path):
        from softae.drivers.async_espico import AsyncESPico
        mods, _ = mock_palmsens
        p = AsyncESPico(name="pico_explicit", config={"port": "COM8"})
        with patch.dict("sys.modules", mods):
            run(p.connect())
        assert p._resolved_port == "COM8"

    def test_sendscript_getdata(self, pico, tmp_path):
        p, mock_device, mods = pico
        # Set up the serial context manager
        mock_comm = MagicMock()
        serial_ctx = MagicMock()
        serial_ctx.__enter__ = MagicMock(return_value=mock_comm)
        serial_ctx.__exit__ = MagicMock(return_value=False)
        mods["palmsens.serial"].Serial = MagicMock(return_value=serial_ctx)
        mods["palmsens.instrument"].Instrument = MagicMock(return_value=mock_device)

        with patch.dict("sys.modules", mods):
            result = p.sendscript_getdata("test.mscr", str(tmp_path), 1)
        assert result == "parsed_raw"
        assert p._measuring is False

    def test_eis_extractdata(self, pico):
        p, _, mods = pico
        # Return different values per column
        mods["palmsens.mscript"].get_values_by_column = MagicMock(
            side_effect=[
                [100.0, 1000.0],  # f
                [50.0, 100.0],    # zreal
                [-30.0, -60.0],   # zimg
            ]
        )
        with patch.dict("sys.modules", mods):
            result = p.eis_extractdata("raw_data")
        assert len(result) == 5  # f, |Z|, phase, Z', -Z''
        assert len(result[0]) == 2

    def test_connect_no_devices_raises(self, mock_palmsens):
        from softae.drivers.async_espico import AsyncESPico
        mods, _ = mock_palmsens
        mods["palmsens.serial"].auto_detect_port.return_value = []
        p = AsyncESPico(name="pico_fail", config={"port": "auto"})
        with patch.dict("sys.modules", mods):
            with pytest.raises(ConnectionError_, match="No EmStat Pico"):
                run(p.connect())

    def test_connect_import_error(self):
        """If palmsens SDK is unavailable, connect() raises ConnectionError_."""
        from softae.drivers.async_espico import AsyncESPico
        p = AsyncESPico(name="pico_noimport", config={"port": "COM3"})
        # Simulate SDK being absent by patching the modules to None, which
        # causes ``import palmsens.*`` to raise ImportError.
        absent = {
            "palmsens": None,
            "palmsens.serial": None,
            "palmsens.instrument": None,
            "palmsens.mscript": None,
        }
        with patch.dict("sys.modules", absent):
            with pytest.raises(ConnectionError_, match="PalmSens SDK"):
                run(p.connect())


    def test_auto_detect_pico2_gets_second_port(self, mock_palmsens, tmp_path):
        """pico2 binds to ports[1], not ports[0], when two devices are detected."""
        from softae.drivers.async_espico import AsyncESPico
        mods, _ = mock_palmsens
        mods["palmsens.serial"].auto_detect_port.return_value = ["COM3", "COM4"]
        p = AsyncESPico(name="pico2", config={"port": "auto", "output_dir": str(tmp_path)})
        with patch.dict("sys.modules", mods):
            run(p.connect())
        assert p._resolved_port == "COM4"

    def test_auto_detect_pico2_raises_when_only_one_device(self, mock_palmsens, tmp_path):
        """pico2 raises ConnectionError_ when auto-detect finds only one device."""
        from softae.drivers.async_espico import AsyncESPico
        mods, _ = mock_palmsens
        mods["palmsens.serial"].auto_detect_port.return_value = ["COM3"]
        p = AsyncESPico(name="pico2", config={"port": "auto", "output_dir": str(tmp_path)})
        with patch.dict("sys.modules", mods):
            with pytest.raises(ConnectionError_, match="port index 1"):
                run(p.connect())
