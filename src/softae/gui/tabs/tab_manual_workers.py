"""One-shot command workers for ManualControlTab button handlers, and the ramp's text.

Kept in a separate module because tab_manual.py already exceeds 400 lines
(contains _ManualPollingWorker and _ManualEisWorker). The manual ramp's progress
line is spelled here as pure functions (spec: ``docs/SubAgent docs/
manual_temp_ramp_progress_abort.md`` §3.2) so the tab only decides *when*.
"""

from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from softae.drivers.temp_ramp import RampReport, RampStep


class _CommandWorker(QThread):
    """Run a single callable on a background thread.

    Emits ``completed(object)`` with the callable's return value on
    success, or ``failed(str)`` with the exception message on error.
    Both signals are delivered back to the main thread via the normal Qt
    queued-connection mechanism.
    """

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _RampWorker(_CommandWorker):
    """A :class:`_CommandWorker` for a manual temperature ramp, with progress and abort.

    A subclass, not a sibling: ``settle_qt`` and every test synchronisation
    join ``_CommandWorker`` children, and the one-shot ``completed``/``failed``
    contract is unchanged. ``progress`` carries each
    :class:`~softae.drivers.temp_ramp.RampStep`; ``cancel`` is the dedicated
    event the driver re-checks under its setpoint-write lock.
    """

    progress = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(fn, parent)
        self.cancel = threading.Event()

    def stop_worker(self, timeout_ms: int = 3000) -> bool:
        """Cancel the ramp and wait up to *timeout_ms*; ``True`` if the thread ended.

        The duck-typed name ``MainWindow.closeEvent`` already calls on every
        QThread child, so the close's belt-and-braces join aborts a ramp too.
        """
        self.cancel.set()
        return self.wait(timeout_ms)


# ── The manual ramp's progress line (spec §3.2) ─────────────────────────────

RAMP_IDLE_TEXT = "Ramp: idle"
RAMP_STARTING_TEXT = "Ramp: reading current setpoint…"
RAMP_ABORTING_TEXT = "Ramp: aborting…"
RAMP_REFUSED_TEXT = "Ramp not started — refused (see the status line)"
#: Appended to a downward ramp's running line: this stage has no active cooling.
RAMP_COOLING_NOTE = " · cooling is passive, PV may lag"


def _mmss(seconds: float) -> str:
    whole = max(int(round(seconds)), 0)
    return f"{whole // 60}:{whole % 60:02d}"


def ramp_running_text(step: RampStep, elapsed_s: float) -> str:
    """The running line for the last *step* seen, re-rendered at *elapsed_s*.

    ETA and "next step in" count down from the step's own schedule, so the line
    moves between writes that may be up to a minute apart.
    """
    eta = max(step.t_span - elapsed_s, 0.0)
    next_in = max(step.elapsed_s + step.next_in_s - elapsed_s, 0.0)
    text = (f"Step {step.index}/{step.total} · SP {step.sp_C:.1f} °C "
            f"→ {step.T_end:.1f} °C · elapsed {_mmss(elapsed_s)} "
            f"· ETA {_mmss(eta)} · next step in {_mmss(next_in)}")
    if step.T_end < step.T_start:
        text += RAMP_COOLING_NOTE
    return text


def ramp_outcome_text(result: Any, T_end: float) -> str:
    """The final line for a ramp's ``completed`` *result*.

    A result that is not a :class:`RampReport` (a stubbed driver returning
    ``None``) is read as complete at *T_end*: the call returned without raising,
    which is all a report-less driver can say.
    """
    if not isinstance(result, RampReport):
        return f"Ramp complete — SP {T_end:.1f} °C"
    if result.zero_span:
        return f"Already at {T_end:.1f} °C — nothing to ramp"
    if result.cancelled and result.last_sp is None:
        return "Ramp aborted before its first step — SP unchanged"
    if result.cancelled:
        return (f"Ramp aborted at step {result.steps_done}/{result.total} "
                f"— holding SP {result.last_sp:.1f} °C")
    return (f"Ramp complete — SP {result.last_sp:.1f} °C "
            f"(elapsed {_mmss(result.elapsed_s)})")


def ramp_failed_text(last_step: RampStep | None, message: str) -> str:
    """The line for a ramp whose driver raised *message*."""
    if last_step is None:
        return f"Ramp failed before its first step: {message}"
    return f"Ramp failed at step {last_step.index}/{last_step.total}: {message}"
