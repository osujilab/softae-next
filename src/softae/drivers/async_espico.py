"""Real PalmSens EmStat Pico potentiostat driver.

Wraps the PalmSens MethodSCRIPT SDK serial calls from ``ESPico_class.py``
behind the :class:`BaseInstrument` ABC.

The orchestration functions (``eis_multichannel_measure``,
``eis_single_channel_measure``) are **not** included here — they belong
in the workflow / experiment layer.  This driver exposes only the
per-instrument measurement primitives.

Hardware Requirements
---------------------
- PalmSens EmStat Pico on a serial port
- ``palmsens`` Python SDK (serial, instrument, mscript)
- MethodSCRIPT library for script generation

Configuration (``softae_config.toml``)::

    [instruments.pico1]
    driver = "espico"
    name   = "Pico1"
    port   = "auto"
    script_dir = "scripts"
    output_dir = "output"
"""

from __future__ import annotations

import datetime
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)

# Path to the drivers/ directory — contains the local palmsens/ SDK
# package.  Path injection is deferred to connect() so we don't disturb
# the import system before we need to.
_DRIVERS_DIR = str(Path(__file__).parent)


def _ensure_sdk_on_path() -> None:
    """Add the drivers/ directory to sys.path if it is not already there.

    Calling this before ``import palmsens`` makes the local copies of the
    PalmSens SDK packages visible to Python without requiring them to be
    installed as a site-package.
    """
    if _DRIVERS_DIR not in sys.path:
        sys.path.insert(0, _DRIVERS_DIR)


class AsyncESPico(BaseInstrument):
    """Async-wrapped PalmSens EmStat Pico potentiostat.

    Unlike the legacy driver which opens/closes a serial context per
    measurement, this driver can keep the port reference and reconnect
    on demand.  The heavy ``sendscript_getdata`` call is inherently
    blocking (waits for the full measurement) and is dispatched to the
    I/O thread pool via :meth:`execute`.
    """

    def __init__(self, name: str = "pico1", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str | None = self.config.get("port", "auto")
        self._script_dir: str = self.config.get("script_dir", "scripts")
        self._output_dir: str = self.config.get("output_dir", "output")
        self._measuring: bool = False
        self._resolved_port: str | None = None  # actual port after auto-detect

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Verify PalmSens SDK availability and resolve the serial port.

        The actual serial connection is opened per-measurement in
        :meth:`sendscript_getdata` (matching the SDK's context-manager
        pattern), but we validate that the SDK imports succeed and
        optionally auto-detect the port here.
        """
        try:
            _ensure_sdk_on_path()
            import palmsens.serial  # noqa: F401
            import palmsens.instrument  # noqa: F401
            import palmsens.mscript  # noqa: F401

            if self._port == "auto":
                ports = palmsens.serial.auto_detect_port()
                if not ports:
                    raise ConnectionError_(
                        "No EmStat Pico devices found during auto-detect",
                        instrument=self.name,
                    )
                # Derive port index from name suffix so that "pico1" gets
                # ports[0] and "pico2" gets ports[1], avoiding both instances
                # binding to the same physical device when port="auto".
                _m = re.search(r"\d+$", self.name)
                _port_idx = (int(_m.group()) - 1) if _m else 0
                if _port_idx >= len(ports):
                    raise ConnectionError_(
                        f"Auto-detect found {len(ports)} device(s), but "
                        f"'{self.name}' requires port index {_port_idx}. "
                        "Assign an explicit port in softae_config.toml.",
                        instrument=self.name,
                    )
                self._resolved_port = ports[_port_idx]
                logger.info(
                    "espico_auto_detected",
                    name=self.name,
                    port=self._resolved_port,
                    port_idx=_port_idx,
                )
            else:
                self._resolved_port = self._port

            self._state = InstrumentState.CONNECTED
            logger.info(
                "espico_connected",
                name=self.name,
                port=self._resolved_port,
            )
        except ImportError as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"PalmSens SDK not installed: {exc}",
                instrument=self.name,
            ) from exc
        except ConnectionError_:
            self._state = InstrumentState.ERROR
            raise
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect to {self.name}: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Mark the instrument as disconnected."""
        self._resolved_port = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("espico_disconnected", name=self.name)

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            port=self._resolved_port,
            measuring=self._measuring,
        )
        return s

    # ── Public API (mirrors ESPico_class.ESPico) ─────────────────────────

    def sendscript_getdata(self, mscrpath: str, outdir: str, chan: int) -> list:
        """Send a MethodSCRIPT file to the instrument and return parsed data.

        Opens the serial connection, sends the script, reads all result
        lines, saves the raw hex output, and parses it.

        Parameters
        ----------
        mscrpath : str
            Path to the ``.mscr`` MethodSCRIPT file.
        outdir : str
            Directory for raw result output files.
        chan : int
            Multiplexer channel number (for logging/naming).

        Returns
        -------
        list
            Parsed raw-data object from ``palmsens.mscript.parse_result_lines``.
        """
        import palmsens.serial
        import palmsens.instrument
        import palmsens.mscript

        self._measuring = True
        try:
            with palmsens.serial.Serial(self._resolved_port, 1) as comm:
                device = palmsens.instrument.Instrument(comm)
                device_type = device.get_device_type()
                logger.info(
                    "espico_send_script",
                    name=self.name,
                    device=str(device_type),
                    channel=chan,
                )

                device.send_script(mscrpath)
                result_lines = device.readlines_until_end()

            # Save raw output
            os.makedirs(outdir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            result_file = os.path.join(outdir, f"ms_plot_{ts}.txt")
            with open(result_file, "wt", encoding="ascii") as fh:
                fh.writelines(result_lines)

            rawdata = palmsens.mscript.parse_result_lines(result_lines)
            logger.info("espico_data_received", channel=chan, file=result_file)
            return rawdata

        except Exception as exc:
            raise CommunicationError(
                f"EIS measurement failed on {self.name} ch{chan}: {exc}",
                instrument=self.name,
            ) from exc
        finally:
            self._measuring = False

    @staticmethod
    def list_available_ports() -> list[str]:
        """Return COM ports hosting EmStat Pico devices.

        Returns an empty list if the PalmSens SDK is not installed or no
        devices are found, so callers can handle both cases gracefully.
        """
        try:
            _ensure_sdk_on_path()
            import palmsens.serial
            return palmsens.serial.auto_detect_port() or []
        except Exception:
            return []

    def reassign_port(self, port: str) -> None:
        """Update the port this instance will use on the next :meth:`connect`.

        Call this before re-connecting after a manual port swap in the GUI.
        """
        self._port = port
        self._resolved_port = port
        logger.info("espico_port_reassigned", name=self.name, port=port)

    def eis_extractdata(self, rawdata) -> list:
        """Extract frequency, impedance, and phase arrays from raw EIS data.

        Parameters
        ----------
        rawdata
            Parsed result from :meth:`sendscript_getdata`.

        Returns
        -------
        list
            ``[f, |Z|, phase, Z', -Z'']`` — each element a numpy array.
        """
        import palmsens.mscript

        f = palmsens.mscript.get_values_by_column(rawdata, 0)
        zreal = palmsens.mscript.get_values_by_column(rawdata, 1)
        zimg = palmsens.mscript.get_values_by_column(rawdata, 2)

        z_complex = np.array(zreal) + 1j * np.array(zimg)
        phase = np.angle(z_complex, deg=True)
        z = np.abs(z_complex)
        zimg_neg = -np.array(zimg)

        return [np.array(f), z, phase, np.array(zreal), zimg_neg]

    def eis_plotdata(
        self,
        data: list,
        ch: int,
        show: bool = True,
        savePath: str = "",
    ) -> None:
        """Plot Nyquist and Bode diagrams for an EIS measurement.

        Parameters
        ----------
        data : list
            Five-element list ``[f, |Z|, phase, Z', -Z'']`` from
            :meth:`eis_extractdata`.
        ch : int
            Channel number (used in plot titles).
        show : bool
            If True, display the plot interactively.
        savePath : str
            If non-empty, save the figure to this path.
        """
        import matplotlib.pyplot as plt

        f, z, phase, zreal, zimg = data[0], data[1], data[2], data[3], data[4]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Nyquist
        axs[0].plot(zreal, zimg)
        axs[0].set_title(f"Nyquist plot, CH{ch}")
        axs[0].set_xlabel("Z' (Ω)")
        axs[0].set_ylabel("-Z'' (Ω)")
        axs[0].set_aspect("equal", adjustable="datalim")
        axs[0].grid(True)

        # Bode
        ax1 = axs[1]
        ax2 = ax1.twinx()
        ax1.loglog(f, zreal, "r-", label="Z'")
        ax1.loglog(f, zimg, "g--", label="-Z''")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Z components (Ω)", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.grid(True, which="major", linestyle="--", alpha=0.5)

        ax2.plot(f, phase, "b-", label="Phase")
        ax2.set_ylabel("-Phase (°)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        ax1.set_title(f"Bode plot, CH{ch}")
        fig.tight_layout()

        if savePath:
            fig.savefig(savePath)
            logger.info("eis_plot_saved", path=savePath)
        if show:
            plt.show(block=False)
        else:
            plt.close(fig)
