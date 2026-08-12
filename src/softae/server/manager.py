"""Centralised instrument registry and resource manager.

The :class:`InstrumentManager` is the single point of access for all
connected instruments.  It owns the lifecycle (connect / disconnect) and
ensures that callers acquire per‑instrument locks before issuing commands,
preventing two threads or tasks from talking to the same serial port
simultaneously.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import structlog

from softae.errors import ConnectionError_, InstrumentError
from softae.server.base_instrument import BaseInstrument, InstrumentState

logger = structlog.get_logger(__name__)


class InstrumentManager:
    """Singleton registry for all connected instruments.

    Usage
    -----
    >>> mgr = InstrumentManager()
    >>> mgr.register(stage_instance)
    >>> await mgr.connect_all()
    >>> async with mgr.acquire("stage") as stage:
    ...     stage.move_to(10, 20)
    """

    def __init__(self) -> None:
        self._instruments: dict[str, BaseInstrument] = {}

    # --- Registration ---------------------------------------------------------

    def register(self, instrument: BaseInstrument) -> None:
        """Add an instrument to the registry."""
        if instrument.name in self._instruments:
            logger.warning("instrument_already_registered", name=instrument.name)
        self._instruments[instrument.name] = instrument
        logger.info("instrument_registered", name=instrument.name)

    def unregister(self, name: str) -> None:
        """Remove an instrument from the registry (must be disconnected)."""
        inst = self._instruments.pop(name, None)
        if inst and inst.is_connected:
            logger.warning("unregistering_connected_instrument", name=name)

    # --- Lifecycle ------------------------------------------------------------

    async def connect(self, name: str) -> None:
        """Connect a single instrument by name."""
        inst = self._get(name)
        try:
            inst._state = InstrumentState.CONNECTING
            await inst.connect()
            logger.info("instrument_connected", name=name)
        except Exception as exc:
            inst._state = InstrumentState.ERROR
            inst._last_error = str(exc)
            logger.error("instrument_connect_failed", name=name, error=str(exc))
            raise

    async def disconnect(self, name: str) -> None:
        """Disconnect a single instrument by name."""
        inst = self._get(name)
        try:
            await inst.disconnect()
            inst._state = InstrumentState.DISCONNECTED
            logger.info("instrument_disconnected", name=name)
        except Exception as exc:
            inst._state = InstrumentState.ERROR
            inst._last_error = str(exc)
            logger.error("instrument_disconnect_failed", name=name, error=str(exc))

    async def connect_all(self) -> dict[str, bool]:
        """Connect every registered instrument.  Returns ``{name: success}``."""
        results: dict[str, bool] = {}
        for name in self._instruments:
            try:
                await self.connect(name)
                results[name] = True
            except Exception:
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect every connected instrument (best‑effort)."""
        for name, inst in self._instruments.items():
            if inst.is_connected:
                await self.disconnect(name)

    # --- Access ---------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self, name: str):
        """Context manager that acquires the instrument's lock.

        Yields the :class:`BaseInstrument` subclass instance.

        Example
        -------
        >>> async with mgr.acquire("stage") as stage:
        ...     stage.move_to(50, 50)
        """
        inst = self._get(name)
        async with inst._lock:
            yield inst

    @asynccontextmanager
    async def acquire_multiple(self, *names: str):
        """Context manager that acquires locks on multiple instruments.

        Locks are acquired in sorted order to prevent deadlocks.
        Yields a ``dict[str, BaseInstrument]`` mapping name → instrument.

        Example
        -------
        >>> async with mgr.acquire_multiple("stage", "syringe") as insts:
        ...     insts["stage"].move_to(10, 20)
        ...     insts["syringe"].single_pump(1.0, 0, 0.5, 0.1)
        """
        if not names:
            raise InstrumentError("At least one instrument name required")

        unique_names = list(dict.fromkeys(names))
        sorted_names = sorted(unique_names)

        for name in sorted_names:
            inst = self._get(name)
            if not inst.is_connected:
                raise InstrumentError(
                    f"Instrument '{name}' is not connected",
                    instrument=name,
                )

        async with AsyncExitStack() as stack:
            for name in sorted_names:
                await stack.enter_async_context(self._instruments[name]._lock)
            yield {name: self._instruments[name] for name in unique_names}

    def get(self, name: str) -> BaseInstrument:
        """Return the instrument instance (no lock)."""
        return self._get(name)

    # --- Query ----------------------------------------------------------------

    def list_instruments(self) -> list[dict[str, Any]]:
        """Return a list of ``{name, state, connected, error}`` dicts."""
        return [inst.status() for inst in self._instruments.values()]

    def status_all(self) -> dict[str, dict[str, Any]]:
        """Return ``{name: status_dict}`` for every registered instrument."""
        return {name: inst.status() for name, inst in self._instruments.items()}

    def reset_locks(self) -> None:
        """Re-create per-instrument ``asyncio.Lock`` objects.

        Call this at the start of a new event loop (e.g. a second sweep in a
        background thread) so that locks bound to a previous loop are replaced
        with fresh ones that will bind to the current running loop.

        Only ``asyncio.Lock`` instances are replaced — synchronous
        ``threading.Lock`` objects (e.g. on the RH controller) are left
        untouched.
        """
        for inst in self._instruments.values():
            if isinstance(getattr(inst, "_lock", None), asyncio.Lock):
                inst._lock = asyncio.Lock()

    @property
    def names(self) -> list[str]:
        """Names of all registered instruments."""
        return list(self._instruments.keys())

    # --- Internal -------------------------------------------------------------

    def _get(self, name: str) -> BaseInstrument:
        try:
            return self._instruments[name]
        except KeyError:
            raise InstrumentError(
                f"No instrument registered with name '{name}'",
                instrument=name,
            ) from None
