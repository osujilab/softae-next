"""Mock PalmSens EmStat Pico potentiostat — runs without hardware.

Generates synthetic EIS, LSV, and OCP data using a physically realistic
R0 – CPE0 – p(R1, C0) (simpleSalt) circuit model.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from softae.server.base_instrument import BaseInstrument, InstrumentState

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Physical EIS generator  (simpleSalt circuit: R0 – CPE0 – p(R1, C0))
# ---------------------------------------------------------------------------
#
# Default parameters match typical polymer-salt thin-film measurements:
#   R0      ~ 4.81e4 Ω     (bulk resistance — log-uniform in [1e3, 1e5])
#   CPE_Q   = 1.00e-7 Ω⁻¹s^α
#   CPE_α   = 0.70
#   R1      ~ 1.84e6 Ω     (interfacial resistance — log-uniform in [1e6, 1e9])
#   C0      = 1.10e-10 F

_CPE_Q: float = 1.00e-7
_CPE_A: float = 0.70
_C0: float    = 1.10e-10
_NPTS: int    = 41
# Frequency sweep: 50 kHz → 1 Hz (high-to-low, matches instrument output)
_FREQ: np.ndarray = np.geomspace(5e4, 1.0, _NPTS)


def _synthetic_eis(
    npts: int = _NPTS,
    R0: float = 4.81e4,
    R1: float = 1.84e6,
    C: float = _C0,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a synthetic EIS spectrum for the R0 – CPE0 – p(R1, C0) circuit.

    Output columns: [f, |Z|, phase_deg, Z_real, -Z_imag]
    """
    freq = np.geomspace(5e4, 1.0, npts) if npts != _NPTS else _FREQ.copy()
    omega = 2.0 * np.pi * freq

    Z_cpe   = 1.0 / (_CPE_Q * (1j * omega) ** _CPE_A)
    Z_r1c0  = R1 / (1.0 + 1j * omega * R1 * C)
    Z_total = R0 + Z_cpe + Z_r1c0

    # ~0.5 % rms multiplicative noise
    rng = np.random.default_rng(seed)
    Z_total = Z_total * rng.normal(1.0, 0.005, npts)

    return np.column_stack([
        freq,
        np.abs(Z_total),
        np.degrees(np.angle(Z_total)),
        Z_total.real,
        -Z_total.imag,
    ])


class MockESPico(BaseInstrument):
    """In-memory EmStat Pico simulator.

    Parameters
    ----------
    name : str
        Label (e.g. ``"pico1"`` or ``"pico2"``).
    config : dict
        Optional keys: ``port`` (ignored in mock).
    """

    def __init__(self, name: str = "pico1", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = self.config.get("port", "SIM")
        self._resolved_port: str | None = None
        self._measuring: bool = False

    async def connect(self) -> None:
        logger.info("mock_espico_connect", name=self.name)
        await asyncio.sleep(0.02)
        self._resolved_port = self._port
        self._state = InstrumentState.CONNECTED

    async def disconnect(self) -> None:
        self._resolved_port = None
        self._state = InstrumentState.DISCONNECTED

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(port=self._resolved_port, measuring=self._measuring)
        return s

    # --- ESPico API (mirrors ESPico_class.ESPico) -----------------------------

    def sendscript_getdata(self, mscrpath: str, outdir: str, chan: int) -> list:
        """Simulate sending a MethodSCRIPT and return synthetic raw curves."""
        import time

        self._measuring = True
        logger.info("mock_eis_measure", channel=chan)
        time.sleep(0.1)  # short delay to simulate measurement
        self._measuring = False
        # Seed from channel so results are reproducible per-channel
        rng = np.random.default_rng(chan)
        R0 = 10.0 ** rng.uniform(3.0, 5.0)
        R1 = 10.0 ** rng.uniform(6.0, 9.0)
        return [_synthetic_eis(R0=R0, R1=R1, seed=chan)]

    @staticmethod
    def list_available_ports() -> list[str]:
        """Return simulated EmStat Pico ports.

        Mirrors :meth:`AsyncESPico.list_available_ports` (always a list;
        empty when no devices are found — the mock always "finds" two).
        """
        return ["SIM1", "SIM2"]

    def reassign_port(self, port: str) -> None:
        """Update the port this instance will use on the next :meth:`connect`.

        Mirrors :meth:`AsyncESPico.reassign_port`.
        """
        self._port = port
        self._resolved_port = port
        logger.info("mock_espico_port_reassigned", name=self.name, port=port)

    def eis_extractdata(self, curves: list) -> list:
        """Extract frequency, impedance, and phase arrays from raw EIS data.

        Returns
        -------
        list
            ``[f, |Z|, phase, Z', -Z'']`` — each element a 1-D numpy array,
            matching :meth:`AsyncESPico.eis_extractdata` exactly.
        """
        if curves and len(curves) > 0:
            arr = np.asarray(curves[0])
        else:
            arr = _synthetic_eis()
        return [np.asarray(arr[:, col]) for col in range(5)]

    def eis_plotdata(self, data: list, ch: int, show: bool = True, savePath: str = "") -> None:
        """No-op plot in mock — data is available for the GUI to plot."""
        logger.debug("mock_eis_plotdata", channel=ch, npts=len(data[0]))
