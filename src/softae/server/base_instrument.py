"""Abstract base class for all SoftAE instrument drivers.

Every hardware driver in the system subclasses :class:`BaseInstrument` so
that the :class:`~softae.server.manager.InstrumentManager` can manage
connections, locks, and status polling uniformly.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Any

import structlog

from softae.errors import CommunicationError, InstrumentError

logger = structlog.get_logger(__name__)

# Shared thread pool for blocking I/O (VISA, Modbus, serial).
# Drivers use ``await loop.run_in_executor(_io_pool, blocking_fn, ...)``
_io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="softae-io")


class InstrumentState(Enum):
    """Connection lifecycle states."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


class BaseInstrument(ABC):
    """Abstract base for all SoftAE instrument drivers.

    Subclasses **must** implement :meth:`connect`, :meth:`disconnect`, and
    :meth:`status`.  They **should not** acquire :attr:`_lock` themselves —
    the :class:`InstrumentManager` (or :meth:`execute`) does that.

    Parameters
    ----------
    name : str
        Human‑readable label (e.g. ``"stage"``).
    config : dict
        Instrument‑specific configuration from ``softae_config.toml``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}
        self._state = InstrumentState.DISCONNECTED
        self._lock = asyncio.Lock()
        self._last_error: str | None = None

    # --- Abstract interface ---------------------------------------------------

    @abstractmethod
    async def connect(self) -> None:
        """Open the physical connection to the instrument.

        Implementations should set ``self._state = InstrumentState.CONNECTED``
        on success or ``InstrumentState.ERROR`` on failure.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the physical connection and release resources."""

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Return a **non‑blocking** snapshot of instrument state.

        Minimum keys: ``name``, ``state``, ``error``.
        Subclasses should add instrument‑specific keys (e.g. position, temperature).
        """

    # --- Convenience ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._state is InstrumentState.CONNECTED

    @property
    def state(self) -> InstrumentState:
        return self._state

    def _base_status(self) -> dict[str, Any]:
        """Common status fields — call from subclass ``status()``."""
        return {
            "name": self.name,
            "state": self._state.name,
            "connected": self.is_connected,
            "error": self._last_error,
        }

    async def execute(self, method_name: str, **kwargs: Any) -> Any:
        """Thread‑safe dispatch: acquire lock, call the named method.

        If the target method is a *sync* function it is automatically
        run in the shared I/O thread pool so the event loop is never blocked.

        Parameters
        ----------
        method_name : str
            Name of the method to call on ``self``.
        **kwargs
            Forwarded to the method.

        Returns
        -------
        Any
            Whatever the called method returns.

        Raises
        ------
        AttributeError
            If *method_name* does not exist.
        RuntimeError
            If the instrument is not connected.
        """
        if not self.is_connected:
            raise InstrumentError(f"not connected", instrument=self.name)

        method = getattr(self, method_name)

        async with self._lock:
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(_io_pool, lambda: method(**kwargs))

    # --- Context manager (sync convenience) -----------------------------------

    def __enter__(self):
        # Sync connect — for notebook/script usage outside an event loop.
        import nest_asyncio

        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(self.connect())
        return self

    def __exit__(self, *exc: Any):
        asyncio.get_event_loop().run_until_complete(self.disconnect())
