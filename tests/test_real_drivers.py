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
import math
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from softae.errors import CommunicationError, ConnectionError_, SafetyError
from softae.server.base_instrument import InstrumentState

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
