"""Ask the OS not to shut down while an experiment is running (Windows).

An organisation-mandated update restart will otherwise kill a multi-day campaign
without warning.  Registering a *shutdown block reason* does not make the machine
un-shutdownable — Group Policy and ``shutdown /f`` override it — but it turns a
silent kill into a visible "SoftAE is running an experiment", which is usually
enough for a human to defer the restart.

Deliberately best-effort and cross-platform-safe: on any non-Windows host, or if
the Win32 call is unavailable, every function is a no-op that reports failure
rather than raising.  Blocking shutdown is a *nicety*; never let it break a run.

Reliability note: this is one layer of several.  It cannot be depended on, so it
sits alongside (a) hardware that fails safe when de-energised — **the thermal
axis only**; the dispenser head is a motor flipper that holds position and does
not self-retract — (b) parking on normal exit, and (c) unclean-shutdown detection
at the next start-up, which never races the OS and is consequently the *primary*
protection for the head.
"""

from __future__ import annotations

import sys

import structlog

logger = structlog.get_logger(__name__)

#: Windows caps the displayed reason; keep it short and specific.
_MAX_REASON_CHARS = 255


def _user32():
    """Return ``ctypes.windll.user32`` on Windows, else ``None``."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        return ctypes.windll.user32
    except Exception:  # pragma: no cover - platform/ctypes edge
        return None


def block_shutdown(hwnd: int, reason: str) -> bool:
    """Ask Windows to hold off shutting down; ``True`` if the block registered.

    ``hwnd`` is the top-level window handle (``int(window.winId())``).
    """
    u32 = _user32()
    if u32 is None or not hwnd:
        return False
    try:
        import ctypes

        ok = bool(
            u32.ShutdownBlockReasonCreate(
                ctypes.c_void_p(int(hwnd)),
                ctypes.c_wchar_p(reason[:_MAX_REASON_CHARS]),
            )
        )
        logger.info("shutdown_block_created", ok=ok, reason=reason)
        return ok
    except Exception:
        logger.warning("shutdown_block_failed", exc_info=True)
        return False


def unblock_shutdown(hwnd: int) -> bool:
    """Release a previously registered block. Safe to call when none is set."""
    u32 = _user32()
    if u32 is None or not hwnd:
        return False
    try:
        import ctypes

        ok = bool(u32.ShutdownBlockReasonDestroy(ctypes.c_void_p(int(hwnd))))
        logger.info("shutdown_block_destroyed", ok=ok)
        return ok
    except Exception:
        logger.warning("shutdown_unblock_failed", exc_info=True)
        return False


class ShutdownBlocker:
    """Context manager / explicit pair around :func:`block_shutdown`.

    Nesting is reference-counted so overlapping runs (a Live campaign and an HT
    run) do not release each other's block.
    """

    def __init__(self, hwnd: int, reason: str) -> None:
        self._hwnd = int(hwnd) if hwnd else 0
        self._reason = reason
        self._depth = 0

    @property
    def active(self) -> bool:
        return self._depth > 0

    def acquire(self) -> None:
        self._depth += 1
        if self._depth == 1:
            block_shutdown(self._hwnd, self._reason)

    def release(self) -> None:
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            unblock_shutdown(self._hwnd)

    def __enter__(self) -> "ShutdownBlocker":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
