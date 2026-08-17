"""Operator-facing alerts — a durable record plus a pluggable notifier seam.

An unattended campaign that parks at 3 a.m. is useless if nobody learns why. The
in-process event stream (``on_event``) dies with the GUI or CLI that hosted it,
so the *reason* a run stopped has to outlive the process. :func:`raise_alert`
does two things:

1. **persists** the alert to the project's DataStore (durable, queryable), and
2. **dispatches** it to any registered sinks (best-effort, never fatal).

No transport is implemented here on purpose — this is the seam a webhook, email,
or Slack sink plugs into later. Persistence works with zero sinks registered, so
the record is never lost just because notification is unconfigured.

Sinks are process-global, mirroring how logging handlers are installed once at
start-up. They are called synchronously; a sink that blocks will stall the
caller, so a slow transport should hand off to its own thread or queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import structlog

if TYPE_CHECKING:
    from softae.core.data_store import DataStore

logger = structlog.get_logger(__name__)


# Severity levels, ordered.  "critical" means a human is needed -- whether or not
# the rig stopped.  A park is self-announcing: the operator finds a halted rig and
# goes looking for the reason.  A demoted RH fault is the opposite -- the hold runs
# to completion, the samples get measured, and the data enters the record looking
# ordinary -- so it needs the loud severity more than a park does, not less.
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """One operator-facing event worth surfacing outside the process."""

    kind: str                                   # "park" | "reservoir" | ...
    message: str
    severity: str = WARNING
    run_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


AlertSink = Callable[[Alert], Any]

_sinks: list[AlertSink] = []


def register_alert_sink(sink: AlertSink) -> None:
    """Add a notifier. Idempotent for the same callable object."""
    if sink not in _sinks:
        _sinks.append(sink)


def unregister_alert_sink(sink: AlertSink) -> None:
    if sink in _sinks:
        _sinks.remove(sink)


def clear_alert_sinks() -> None:
    """Drop every sink (used by tests to avoid cross-test leakage)."""
    _sinks.clear()


def raise_alert(
    alert: Alert,
    *,
    data_store: "DataStore | None" = None,
) -> int | None:
    """Persist *alert* and fan it out to the registered sinks.

    Returns the stored row id, or ``None`` when no store was supplied or the
    write failed. **Never raises** — an alert is a report about something that
    already went wrong, so it must not become a second failure. A failing sink
    is logged and the remaining sinks still run.
    """
    log = logger.error if alert.severity == CRITICAL else logger.warning
    log(
        "alert",
        kind=alert.kind, severity=alert.severity, message=alert.message,
        run_id=alert.run_id, **alert.details,
    )

    row_id: int | None = None
    if data_store is not None:
        try:
            row_id = data_store.record_alert(
                alert.kind, alert.message,
                severity=alert.severity, run_id=alert.run_id,
                details=dict(alert.details) or None,
            )
        except Exception:
            logger.warning("alert_persist_failed", kind=alert.kind, exc_info=True)

    for sink in list(_sinks):
        try:
            sink(alert)
        except Exception:
            logger.warning("alert_sink_failed", sink=repr(sink), exc_info=True)

    return row_id
