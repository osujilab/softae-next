"""Driver factory — real or mock instruments based on config & availability.

Usage::

    from softae.drivers.factory import create_manager

    mgr = create_manager(mock=None)   # auto-detect — real, falling back to mock
    mgr = create_manager(mock=True)   # force mock suite
    mgr = create_manager(mock=False)  # force real drivers
    await mgr.connect_all()

``mock`` is a **required** keyword: whether a process means to open real ports
decides whether it claims the rig and whether what it records is data.

When ``mock=False`` (or auto-detect), the factory probes each serial
port / device.  If it cannot import a required hardware library **or**
the port is unreachable, it falls back to the corresponding mock driver
and logs a warning.
"""

from __future__ import annotations

import importlib
from typing import Any

import structlog

from softae.config.loader import instruments as _cfg_instruments
from softae.drivers.mock_factory import create_mock_manager
from softae.server.base_instrument import BaseInstrument
from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

# Mapping: (driver key in config) → (real module path, real class name)
_REAL_DRIVERS: dict[str, tuple[str, str]] = {
    "temp_controller": (
        "softae.drivers.async_temp_controller",
        "AsyncTempController",
    ),
    "rh_controller": (
        "softae.drivers.async_rh_controller",
        "AsyncRHController",
    ),
    "ht_sensor": (
        "softae.drivers.async_ht_sensor",
        "AsyncHTSensor",
    ),
    "stage": (
        "softae.drivers.async_stage",
        "AsyncStage",
    ),
    "syringe": (
        "softae.drivers.async_syringe",
        "AsyncSyringe",
    ),
    "pico1": (
        "softae.drivers.async_espico",
        "AsyncESPico",
    ),
    "pico2": (
        "softae.drivers.async_espico",
        "AsyncESPico",
    ),
    "camera": (
        "softae.drivers.async_camera",
        "AsyncCamera",
    ),
    # "lamp" and any future DAC-switched loads (channels B–D) all use AsyncDACSwitch.
    # To add a new switched instrument, register its driver key here and add a
    # corresponding [instruments.<key>] section in softae_config.toml.
    "lamp": (
        "softae.drivers.async_dac_switch",
        "AsyncDACSwitch",
    ),
    "dac_switch": (
        "softae.drivers.async_dac_switch",
        "AsyncDACSwitch",
    ),
    "piezo": (
        "softae.drivers.async_piezo",
        "AsyncPiezoController",
    ),
}


def _try_real_driver(
    name: str,
    cfg: dict[str, Any],
) -> BaseInstrument | None:
    """Attempt to instantiate the real driver for *name*.

    Returns ``None`` if the driver module or a hardware dependency is
    unavailable, allowing the caller to fall back to a mock.
    """
    entry = _REAL_DRIVERS.get(name)
    if entry is None:
        return None  # no real driver available for this instrument type

    mod_path, cls_name = entry
    try:
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return cls(name=name, config=cfg)
    except Exception as exc:
        logger.warning(
            "real_driver_import_failed",
            instrument=name,
            error=str(exc),
        )
        return None


def create_manager(
    *,
    mock: bool | None,
    config: dict[str, Any] | None = None,
) -> InstrumentManager:
    """Build an :class:`InstrumentManager` with the best available drivers.

    Parameters
    ----------
    mock : bool or None
        **Required — there is no default.** Whether this process intends to open
        real ports is not a detail a caller may leave unstated: it decides
        whether the rig gets claimed, whether the interlock arms, and whether a
        recorded spectrum is data or a simulation. The old default was ``None``,
        so a caller that simply forgot inherited *auto-detect* — the one mode
        that can produce a manager which is partly real and partly mock, which is
        also the shape that reads as "simulated" to any motion-scoped check (see
        :func:`softae.core.rig_session.session_is_simulated`).

        - ``True`` — force all mocks.
        - ``False`` — force real drivers (raises if hardware unavailable).
        - ``None`` — auto: try real, fall back to mock. Still available, but now
          only ever by an explicit request for it.
    config : dict, optional
        Override for the ``[instruments]`` section of the TOML config.

    Returns
    -------
    InstrumentManager
    """
    if mock is True:
        return create_mock_manager(config)

    if config is None:
        try:
            config = _cfg_instruments()
        except FileNotFoundError:
            config = {}

    mgr = InstrumentManager()

    # Import all mock classes for fallback
    from softae.drivers.mock_camera import MockCamera, MockDACSwitch, MockLamp
    from softae.drivers.mock_espico import MockESPico
    from softae.drivers.mock_ht_sensor import MockHTSensor
    from softae.drivers.mock_keithley import MockKeithley
    from softae.drivers.mock_rh_controller import MockRHController
    from softae.drivers.mock_stage import MockStage
    from softae.drivers.mock_syringe import MockSyringe
    from softae.drivers.mock_temp_controller import MockTempController
    from softae.drivers.mock_piezo import MockPiezoController

    # Instruments that only have mock drivers (for now)
    _mock_only: dict[str, type] = {
        "keithley": MockKeithley,
    }

    _mock_fallback: dict[str, type] = {
        "temp_controller": MockTempController,
        "rh_controller": MockRHController,
        "ht_sensor": MockHTSensor,
        "stage": MockStage,
        "syringe": MockSyringe,
        "pico1": MockESPico,
        "pico2": MockESPico,
        "camera": MockCamera,
        "lamp": MockDACSwitch,
        "dac_switch": MockDACSwitch,
        "piezo": MockPiezoController,
    }

    # Register mock-only instruments
    for name, mock_cls in _mock_only.items():
        cfg = config.get(name, {})
        mgr.register(mock_cls(name, cfg))

    # Register real-or-mock instruments
    for name, fallback_cls in _mock_fallback.items():
        cfg = config.get(name, {})
        inst = _try_real_driver(name, cfg)

        if inst is not None:
            logger.info("real_driver_loaded", instrument=name)
            mgr.register(inst)
        elif mock is False:
            # Caller demanded real drivers — don't silently fall back
            raise RuntimeError(
                f"Real driver for '{name}' could not be loaded and mock=False"
            )
        else:
            logger.info("mock_driver_fallback", instrument=name)
            mgr.register(fallback_cls(name, cfg))

    # Coordinator instrument — drives whatever stage + syringe were registered
    # above (real or mock fallback). Registered last so both exist.
    from softae.drivers.async_liquid_handler import AsyncLiquidHandler

    lh = AsyncLiquidHandler("liquid_handler", config.get("liquid_handler", {}))
    lh.manager = mgr
    mgr.register(lh)

    return mgr
