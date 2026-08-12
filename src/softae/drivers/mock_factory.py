"""Factory for creating mock instrument suites.

Provides :func:`create_mock_manager` which returns an
:class:`~softae.server.manager.InstrumentManager` pre-populated with all
mock drivers — ready to develop and test the GUI without physical hardware.
"""

from __future__ import annotations

from softae.config.loader import instruments as _cfg_instruments
from softae.drivers.async_liquid_handler import AsyncLiquidHandler
from softae.drivers.mock_camera import MockCamera, MockDACSwitch, MockLamp
from softae.drivers.mock_espico import MockESPico
from softae.drivers.mock_ht_sensor import MockHTSensor
from softae.drivers.mock_keithley import MockKeithley
from softae.drivers.mock_piezo import MockPiezoController
from softae.drivers.mock_rh_controller import MockRHController
from softae.drivers.mock_stage import MockStage
from softae.drivers.mock_syringe import MockSyringe
from softae.drivers.mock_temp_controller import MockTempController
from softae.server.manager import InstrumentManager


def create_mock_manager(config: dict | None = None) -> InstrumentManager:
    """Build an :class:`InstrumentManager` with every mock instrument registered.

    Parameters
    ----------
    config : dict, optional
        If provided, used instead of the file-based config.
        Each key should match an instrument name.

    Returns
    -------
    InstrumentManager
        Ready to ``await mgr.connect_all()``.
    """
    if config is None:
        try:
            config = _cfg_instruments()
        except FileNotFoundError:
            config = {}

    mgr = InstrumentManager()

    mgr.register(MockStage("stage", config.get("stage", {})))
    mgr.register(MockSyringe("syringe", config.get("syringe", {})))
    mgr.register(MockTempController("temp_controller", config.get("temp_controller", {})))
    mgr.register(MockESPico("pico1", config.get("pico1", {})))
    mgr.register(MockESPico("pico2", config.get("pico2", {})))
    mgr.register(MockCamera("camera", config.get("camera", {})))
    mgr.register(MockLamp("lamp", config.get("lamp", {})))
    mgr.register(MockDACSwitch("dac_switch", config.get("dac_switch", {})))
    mgr.register(MockKeithley("keithley", config.get("keithley", {})))
    mgr.register(MockHTSensor("ht_sensor", config.get("ht_sensor", {})))
    mgr.register(MockRHController("rh_controller", config.get("rh_controller", {})))
    mgr.register(MockPiezoController("piezo", config.get("piezo", {})))

    # Coordinator instrument — drives the already-registered stage + syringe.
    # The same class serves mock and real (it only orchestrates sub-instruments).
    lh = AsyncLiquidHandler("liquid_handler", config.get("liquid_handler", {}))
    lh.manager = mgr
    mgr.register(lh)

    return mgr
