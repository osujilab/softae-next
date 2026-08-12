"""Real Harvard Apparatus syringe pump + pneumatic head driver.

Wraps the blocking VISA and NI-DAQ calls from the original
``syringe_class.py`` behind the :class:`BaseInstrument` ABC.

Hardware Requirements
---------------------
- Harvard Apparatus syringe pump on a VISA serial (ASRL) port
- (Optional) NI-DAQ digital-output channel for pneumatic dispenser head
- ``pyvisa`` and ``pyvisa-py`` (or NI-VISA backend)
- (Optional) ``nidaqmx`` for head control

Configuration (``softae_config.toml``)::

    [instruments.syringe]
    port     = "ASRL4::INSTR"
    baud     = 115_200
    diameter = 14.4

    [instruments.pneumatic_head]
    channel  = "Dev1/port0"

Anti-patterns fixed vs. legacy
------------------------------
- ``global com_syr`` / ``global is_up`` → instance attributes
- Duplicate VISA handle → single ``self._visa_inst``
- ``input()`` in ``head_check`` → injectable ``confirm_fn`` callback
- Bare ``except:`` → logged, explicit exception handling
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog

from softae.drivers.contracts import ParallelSyringeMixin
from softae.errors import CommunicationError, ConnectionError_
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class AsyncSyringe(ParallelSyringeMixin, BaseInstrument):
    """Async-wrapped Harvard Apparatus syringe pump + pneumatic head.

    All blocking VISA I/O is dispatched to the shared
    :data:`~softae.server.base_instrument._io_pool` via :meth:`execute`.
    """

    def __init__(self, name: str = "syringe", config: dict[str, Any] | None = None):
        super().__init__(name, config)
        self._port: str = self.config.get("port", "ASRL4::INSTR")
        self._baud: int = int(self.config.get("baud", 115_200))
        self._diameter: float = float(self.config.get("diameter", 14.4))
        self._head_channel: str | None = self.config.get("head_channel")
        self._max_rate: float = float(self.config.get("max_rate", 2120.0))
        self._min_rate: float = float(self.config.get("min_rate", 0.001))
        self._init_parallel_syringes(self.config)
        self._visa_inst = None
        self._rm = None
        self._is_up: bool = True  # head retracted by default

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the VISA serial connection to the syringe pump."""
        try:
            import pyvisa

            self._rm = pyvisa.ResourceManager()
            self._visa_inst = self._rm.open_resource(self._port)
            self._visa_inst.baud_rate = self._baud
            # Deliberately does NOT touch ``_is_up``.  Opening a serial port tells
            # you nothing about where a mechanical flipper is — which is the whole
            # reason the operator is asked at launch.  This line used to assert
            # "retracted" here, and because connection is backgrounded it ran
            # *after* the prompt and silently discarded the operator's answer: a
            # head confirmed as Lowered was believed raised by
            # ``check_head_clear_to_move``, which would then permit stage travel
            # with the head down.
            self._state = InstrumentState.CONNECTED
            logger.info(
                "syringe_connected",
                port=self._port,
                baud=self._baud,
                diameter=self._diameter,
            )
        except Exception as exc:
            self._state = InstrumentState.ERROR
            self._last_error = str(exc)
            raise ConnectionError_(
                f"Failed to connect to syringe pump on {self._port}: {exc}",
                instrument=self.name,
            ) from exc

    async def disconnect(self) -> None:
        """Close the VISA session."""
        if self._visa_inst is not None:
            try:
                self._visa_inst.close()
            except Exception:
                pass
            self._visa_inst = None
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
        self._state = InstrumentState.DISCONNECTED
        logger.info("syringe_disconnected", port=self._port)

    def status(self) -> dict[str, Any]:
        s = self._base_status()
        s.update(
            diameter=self._diameter,
            head_up=self._is_up,
            parallel_syringes=self._parallel_syringes,
            parallel_syringes_by_pump=dict(self._parallel_syringes_by_pump),
        )
        return s

    # ── Public API (mirrors syringe_class.syringe) ───────────────────────
    #
    # set_parallel_syringes() and effective_per_syringe_volume() are provided
    # by ParallelSyringeMixin (shared with MockSyringe).

    def single_pump(
        self,
        res_vol: float,
        ID: int,
        rate: float,
        dispense_vol: float,
    ) -> None:
        """Command syringe pump *ID* to dispense *dispense_vol* µL.

        Parameters
        ----------
        res_vol : float
            Syringe volume declared to the pump firmware (mL), sent as
            ``{ID} svolume``.  This is a firmware limit parameter, **not** a
            measure of stock on hand — by convention it is padded to exceed the
            command so the pump does not trip its own hardware stop mid-dispense.
            Remaining stock is tracked separately by
            :class:`~softae.core.reservoir.ReservoirLedger`; never derive one
            from the other.
        ID : int
            Pump address index (0, 1, or 2).
        rate : float
            Infuse rate (µL/min).
        dispense_vol : float
            Target dispense volume (µL).

        Raises
        ------
        SafetyError
            If *rate* exceeds ``max_rate``, is below ``min_rate``,
            *dispense_vol* exceeds the declared syringe volume *res_vol*, or an
            attached reservoir ledger reports insufficient stock.

        Notes
        -----
        A commanded ``dispense_vol`` of ``0`` (or less) means "leave this pump
        alone": the call returns immediately without validating or writing to the
        hardware, so a zeroed formulation component never trips ``min_rate``.
        """
        if self._is_noop_pump_command(dispense_vol):
            logger.info("syringe_pump_skip", pump_id=ID, reason="zero_volume")
            return

        self._validate_single_pump(res_vol, rate, dispense_vol, ID)

        rate = max(rate, 0.001)
        dispense_vol = max(dispense_vol, 0.001)

        hw_vol = self.effective_per_syringe_volume(dispense_vol, ID)

        self._visa_inst.write(f"{ID} svolume {res_vol} ml")
        time.sleep(0.1)
        self._visa_inst.write(f"{ID}diameter {float(self._diameter)}")
        time.sleep(0.1)
        self._visa_inst.write(f"{ID}irate {float(rate)} ul/min")
        time.sleep(0.1)
        self._visa_inst.write(f"{ID}tvolume {float(hw_vol)} ul")
        time.sleep(0.1)
        self._visa_inst.write(f"{ID}irun")
        logger.info(
            "syringe_pump",
            pump_id=ID,
            rate=rate,
            vol=dispense_vol,
            diameter=self._diameter,
        )

    def head_flip(self) -> None:
        """Toggle the pneumatic dispenser head (retract ↔ descend).

        Sends a brief digital pulse via the NI-DAQ output channel.
        Falls back to a no-op with a warning if ``nidaqmx`` is unavailable
        or no channel is configured.
        """
        channel = self._head_channel
        if channel is None:
            logger.warning("head_flip_no_channel", msg="No NI-DAQ channel configured for head")
            self._is_up = not self._is_up
            return

        try:
            import nidaqmx

            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(channel)
                for val in [0, 1, 0]:
                    task.write(val)
                    time.sleep(0.5)
            self._is_up = not self._is_up
            logger.info("head_flip", is_up=self._is_up)
        except ImportError:
            logger.warning("nidaqmx_not_available", msg="head_flip requires nidaqmx")
            self._is_up = not self._is_up
        except Exception as exc:
            raise CommunicationError(
                f"Head flip failed: {exc}",
                instrument=self.name,
            ) from exc

    def head_check(self, confirm_fn: Callable[[str], bool] | None = None) -> None:
        """Confirm and set the dispenser head position.

        Parameters
        ----------
        confirm_fn : callable, optional
            Callback ``(prompt: str) -> bool``.  If the callback returns
            ``False`` (i.e. the head is not retracted) the head is flipped.
            If ``None``, the head is assumed retracted (safe for headless
            and GUI usage).
        """
        if confirm_fn is None:
            self._is_up = True
            return
        if not confirm_fn("Is the head retracted? (y/n) "):
            self.head_flip()

    def is_head_up(self) -> bool:
        """Registered head position: ``True`` when retracted/raised (safe).

        This is the software's *belief*, not a sensed value — the pneumatic head
        has no position feedback.  Operators keep it truthful via
        :meth:`set_head_state` (see the GUI head-verification prompts).
        """
        return self._is_up

    def set_head_state(self, is_up: bool) -> None:
        """Register the head position **without** issuing any motion.

        Used by the operator-verification dialogs to sync the software belief to
        physical reality before automated sequences issue conditional head
        commands (:meth:`head_retract` / :meth:`head_descend` only flip when the
        belief disagrees, so a stale belief would cause one wrong flip).
        """
        self._is_up = bool(is_up)
        logger.info("head_state_registered", is_up=self._is_up)

    def head_retract(self) -> None:
        """Ensure the dispenser head is retracted (safe travel position)."""
        if not self._is_up:
            self.head_flip()

    def head_descend(self) -> None:
        """Ensure the dispenser head is lowered to the dispensing position."""
        if self._is_up:
            self.head_flip()

    def syr_end(self) -> None:
        """Close the VISA session (alias for API compatibility)."""
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None
