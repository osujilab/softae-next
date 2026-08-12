"""Mock ThorLabs Zelux camera + lamp — runs without hardware.

Generates synthetic color images (gradient patterns with noise).
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


class MockCamera(BaseInstrument):
    """In-memory camera simulator producing synthetic images."""

    def __init__(self, name: str = "camera", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._exposure: float = self.config.get("exposure", 0.045)
        self._frame_count: int = 0

    async def connect(self) -> None:
        logger.info("mock_camera_connect")
        await asyncio.sleep(0.02)
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(exposure=self._exposure, frames_captured=self._frame_count)
        return s

    # --- Camera API -----------------------------------------------------------

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def acquire_n_frames(
        self,
        frames: int = 1,
        exp: float | None = None,
        gain: int = 1,
        poll_TO: int = 10000,
    ) -> np.ndarray:
        """Return a synthetic (H, W, 3) uint8 image.

        Signature mirrors :meth:`AsyncCamera.acquire_n_frames`; *gain* and
        *poll_TO* are accepted but ignored by the simulation.
        """
        exposure = exp if exp is not None else self._exposure
        logger.debug("mock_acquire", frames=frames, exposure=exposure, gain=gain)
        h, w = 480, 640
        # Generate a smooth gradient with noise
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (128 + 60 * np.sin(2 * np.pi * xv * 3) + np.random.randint(0, 20, (h, w))).clip(0, 255)
        img[:, :, 1] = (100 + 80 * yv + np.random.randint(0, 15, (h, w))).clip(0, 255)
        img[:, :, 2] = (80 + 40 * np.cos(2 * np.pi * (xv + yv) * 2) + np.random.randint(0, 10, (h, w))).clip(0, 255)
        self._frame_count += frames
        return img

    def snap(self, n: int = 1, exposure: float = 0.045, show: bool = False,
             save: bool = False, path: str = "", dpi: int = 150) -> np.ndarray:
        """Convenience: acquire + optionally display/save."""
        return self.acquire_n_frames(n, exposure)

    def save_image(self, array: np.ndarray, path: str, dpi: int = 150) -> None:
        """Save image (no-op in mock — would use matplotlib)."""
        logger.debug("mock_save_image", path=path)


class MockDACSwitch(BaseInstrument):
    """In-memory on/off switch — mock for :class:`~softae.drivers.async_dac_switch.AsyncDACSwitch`.

    Tracks ``is_on`` state only; no hardware interaction.  Also provides a
    no-op ``set_eeprom_defaults()`` so tests that call it don't need to patch.
    """

    def __init__(self, name: str = "dac_switch", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._channel: str = str((config or {}).get("channel", "A"))
        self._is_on: bool = False

    async def connect(self) -> None:
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(is_on=self._is_on, channel=self._channel)
        return s

    def on(self) -> None:
        self._is_on = True

    def off(self) -> None:
        self._is_on = False

    def set_eeprom_defaults(self) -> None:
        pass  # no-op in mock


# Backward-compatible alias — factory and any existing code using MockLamp still work.
MockLamp = MockDACSwitch
