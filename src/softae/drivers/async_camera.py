"""Real ThorLabs Zelux camera driver.

Wraps the blocking ThorLabs TSI SDK from the original
``ThorLabsCamera_class.py`` behind the :class:`BaseInstrument` ABC.

For lamp / relay switching see :mod:`softae.drivers.async_dac_switch`.
``AsyncLamp`` below is a backward-compatible alias for
:class:`~softae.drivers.async_dac_switch.AsyncDACSwitch`.

Hardware Requirements
---------------------
- ThorLabs Zelux camera with Scientific Camera SDK installed
- ``thorlabs_tsi_sdk`` Python package

Configuration (``softae_config.toml``)::

    [instruments.camera]
    driver   = "camera"
    exposure = 0.045           # seconds

Anti-patterns fixed vs. legacy
------------------------------
- ``os.add_dll_directory`` hard-coded path → config-driven ``dll_path``
- ``print("No cameras detected"); exit()`` → raises ``ConnectionError_``
- ``print("Unable to acquire frame.")`` → raises ``CommunicationError``
- Bare ``except:`` → explicit exception handling
- Global ``plt.show()`` side-effects → opt-in ``show`` parameter
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import structlog

from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

# Lamp switching lives in the generic DAC-switch module; this alias keeps
# any code that does ``from softae.drivers.async_camera import AsyncLamp`` working.
from softae.drivers.async_dac_switch import AsyncDACSwitch as AsyncLamp  # noqa: F401

logger = structlog.get_logger(__name__)


class AsyncCamera(BaseInstrument):
    """Async-wrapped ThorLabs Zelux camera.

    All blocking SDK I/O is dispatched to the shared
    :data:`~softae.server.base_instrument._io_pool` via :meth:`execute`.
    """

    def __init__(self, name: str = "camera", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._exposure: float = float(self.config.get("exposure", 0.045))
        self._frame_count: int = 0
        self._sdk = None
        self._camera = None
        self._dll_path: str = self.config.get(
            "dll_path",
            r"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
            r"\Scientific Camera Interfaces\SDK\Python Toolkit\dlls\64_lib",
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialise ThorLabs SDK, add DLL directory, and discover cameras."""
        try:
            os.add_dll_directory(self._dll_path)
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK  # noqa: F811

            self._sdk = TLCameraSDK()
            cam_list = self._sdk.discover_available_cameras()
            if len(cam_list) < 1:
                raise ConnectionError_(
                    "No ThorLabs cameras detected",
                    instrument=self.name,
                )
            self._cam_list = cam_list
            self._state = InstrumentState.CONNECTED
            logger.info(
                "camera_connected",
                cameras_found=len(cam_list),
                dll_path=self._dll_path,
            )
        except ImportError as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"ThorLabs TSI SDK not installed: {exc}",
                instrument=self.name,
            ) from exc
        except ConnectionError_:
            self._state = InstrumentState.ERROR
            raise
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to initialise camera SDK: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Dispose the ThorLabs SDK."""
        if self._sdk is not None:
            try:
                self._sdk.dispose()
            except Exception:
                pass
            self._sdk = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("camera_disconnected")

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            exposure=self._exposure,
            frames_captured=self._frame_count,
        )
        return s

    # ── Camera API ───────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the first discovered camera."""
        self._camera = self._sdk.open_camera(self._cam_list[0])

    def close(self) -> None:
        """Dispose the camera handle."""
        if self._camera is not None:
            try:
                self._camera.dispose()
            except Exception:
                pass
            self._camera = None

    def acquire_n_frames(
        self,
        frames: int = 1,
        exp: float | None = None,
        gain: int = 1,
        poll_TO: int = 10000,
    ) -> np.ndarray:
        """Acquire *frames* from the camera and return a (H, W, 3) uint8 array.

        Uses one retry attempt if the first frame poll returns ``None``.

        Parameters
        ----------
        frames : int
            Number of frames to request from the trigger.
        exp : float, optional
            Exposure time in **seconds**. Falls back to ``self._exposure``.
        gain : int
            Camera gain setting.
        poll_TO : int
            Image poll timeout in milliseconds.

        Returns
        -------
        np.ndarray
            Colour image array of shape ``(H, W, 3)`` with dtype ``uint8``.

        Raises
        ------
        CommunicationError
            If no frame can be acquired after a retry.
        """
        import thorlabs_tsi_sdk.tl_mono_to_color_processor as TLColor

        exposure = exp if exp is not None else self._exposure

        # NOTE: Do NOT use ``with self._camera as camera:`` here.
        # The ThorLabs SDK context-manager disposes the camera handle on
        # __exit__, which makes the camera unusable for subsequent calls.
        # The CameraWorker (and snap()) manage open/close externally.
        camera = self._camera
        camera.exposure_time_us = int(exposure * 1e6)
        camera.gain = gain
        camera.frames_per_trigger_zero_for_unlimited = frames
        camera.image_poll_timeout_ms = poll_TO

        cSensType = camera.camera_sensor_type
        cArrPhase = camera.color_filter_array_phase
        cCorrMat = camera.get_color_correction_matrix()
        cWBMat = camera.get_default_white_balance_matrix()
        cBitDepth = camera.bit_depth
        cHpx = camera.image_height_pixels
        cWpx = camera.image_width_pixels

        camera.arm(2)
        camera.issue_software_trigger()
        frame = camera.get_pending_frame_or_null()

        if frame is None:
            # One retry: disarm → re-arm → trigger again
            try:
                camera.disarm()
                camera.arm(2)
                camera.issue_software_trigger()
                frame = camera.get_pending_frame_or_null()
            except Exception as exc:
                raise CommunicationError(
                    f"Unable to acquire frame after retry: {exc}",
                    instrument=self.name,
                ) from exc

        if frame is None:
            camera.disarm()
            raise CommunicationError(
                "Unable to acquire frame — no pending frame after retry",
                instrument=self.name,
            )

        image_buffer_copy = np.copy(frame.image_buffer)
        with TLColor.MonoToColorProcessorSDK() as MTCsdk:
            with MTCsdk.create_mono_to_color_processor(
                cSensType, cArrPhase, cCorrMat, cWBMat, cBitDepth,
            ) as processor:
                nd_image_array = processor.transform_to_24(
                    image_buffer_copy, cWpx, cHpx,
                )

        image_3d = nd_image_array.reshape(cHpx, cWpx, 3)
        camera.disarm()

        self._frame_count += frames
        return image_3d

    def snap(
        self,
        n: int = 1,
        exposure: float = 0.045,
        show: bool = False,
        save: bool = False,
        path: str = "",
        dpi: int = 150,
    ) -> np.ndarray:
        """Convenience: open camera, acquire, close, optionally display/save."""
        self.open()
        try:
            arr = self.acquire_n_frames(n, exposure)
        finally:
            self.close()

        if show:
            from matplotlib import pyplot as plt

            fig = plt.figure(frameon=False)
            fig.set_size_inches(3 * 1440 / 1080, 3)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(arr, aspect="auto")
            plt.show()

        if save and path:
            self.save_image(arr, path, dpi)

        return arr

    def save_image(self, array: np.ndarray, path: str, dpi: int = 150) -> None:
        """Save *array* to disk using OpenCV."""
        import cv2

        cv2.imwrite(path, array)
        logger.debug("camera_save_image", path=path)


# AsyncLamp is imported at the top of this module as an alias for AsyncDACSwitch.
# The class definition has moved to softae.drivers.async_dac_switch.
