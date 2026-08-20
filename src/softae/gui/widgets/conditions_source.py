"""The rig as an attached window may see it: the campaign's ``conditions.json``.

A window that opened no instrument session must not read the instruments. That is
not a policy about tidiness — a read is a serial transaction on a bus another
process is mid-anneal on, and
:class:`~softae.drivers.async_temp_controller.AsyncTempController` holds its
``_serial_lock`` for a whole retry window. So the campaign publishes what *it* can
see, into a single-slot file beside its event stream
(:class:`~softae.core.campaign_events.ConditionsPublisher`), and this reads it.

**It is a source, not a second consumer path.** It fills exactly the three dicts
:class:`~softae.gui.widgets.instrument_poller.InstrumentPoller` already emits, so
the Monitoring tab, the sidebar and the status bar render an attached rig through
the slots they already have. None of the three is edited for this.

Three things it says that the live source cannot
------------------------------------------------
**Every instrument is the campaign's, not disconnected.** ``DISCONNECTED`` is
grey, and grey on a live rig running an eight-hour anneal is the wrong sentence
entirely — it reads as "the rig is off". The state published here is
:data:`~softae.gui.widgets.rig_owner.OCCUPIED`, the word the Init tab and the
Manual Control banner already use, and it is the one thing in this arc that
required a change outside this module: ``status_indicator`` had no colour for
"owned by someone else".

**Stale is rendered as unknown, never as current.** The publisher rewrites the
file on a *skipped* beat too, so file mtime advances even while a read is wedged
inside a driver retry; only ``completed_at`` dates the numbers. Past
:data:`CONDITIONS_STALE_AFTER_S` the values are dropped and every field renders
``--``. A number that is two minutes old displayed as though it were now is the
one lie a monitoring surface must not tell, and the widgets have nowhere to show
an age — the sidebar's owner line carries the campaign's phase age instead.

**What is absent stays absent.** The sidecar carries five environment fields and
no stage position and no head state, because the campaign publishes what it reads
for its own purposes. Those fields are simply not in the reading, so the widgets
show ``--`` for them rather than a guess.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from softae.core.campaign_events import DEFAULT_CONDITIONS_POLL_S, conditions_path
from softae.gui.widgets.instrument_poller import PollReading
from softae.gui.widgets.rig_owner import OCCUPIED

logger = structlog.get_logger(__name__)

#: How old the last *completed* read may be before its numbers stop being shown.
#:
#: Three conditions beats, the same three-beat discipline
#: :func:`~softae.core.campaign_events.liveness` applies to the heartbeat: one
#: missed beat is a busy event loop, three is a fact. It is deliberately measured
#: on ``completed_at`` and not on the file's mtime, because a skipped beat
#: republishes the payload — mtime would report a wedged read as fresh.
CONDITIONS_STALE_AFTER_S = 3 * DEFAULT_CONDITIONS_POLL_S


class ConditionsFileSource:
    """Fill the poller's three dicts from a campaign's ``conditions.json``.

    Parameters
    ----------
    run_dir
        The campaign's run directory (the rig lock's ``log_path``). ``None`` is
        allowed and means *occupied by something that publishes nothing* — the
        instruments are still reported as the holder's, and every value is
        unknown, which is exactly the truth in that case.
    manager
        Read for :attr:`~softae.server.manager.InstrumentManager.names` only, so
        the status row lists the same instruments it lists in owner mode. The
        property is a dict-key copy and touches no hardware; ``status_all()``
        deliberately is **not** called, because it calls ``status()`` on every
        instrument and the RH controller's ``status()`` reads the sensor.
    """

    def __init__(
        self,
        run_dir: str | Path | None,
        *,
        manager: Any = None,
        stale_after_s: float = CONDITIONS_STALE_AFTER_S,
        now: Any = time.time,
    ) -> None:
        self.run_dir = str(run_dir) if run_dir else None
        self.path = conditions_path(run_dir) if run_dir else None
        self._manager = manager
        self._stale_after_s = float(stale_after_s)
        self._now = now
        self._warned = False

    # ── The one method the poller calls ──────────────────────────────────────

    def read(self) -> PollReading:
        """One cycle: the holder's instruments, plus the numbers if they are fresh."""
        payload = self._payload()
        sidebar, monitor = self._readings(payload)
        return PollReading(statuses=self._statuses(), sidebar=sidebar, monitor=monitor)

    # ── Internals ────────────────────────────────────────────────────────────

    def _statuses(self) -> dict[str, dict[str, Any]]:
        """Every registered instrument, reported as the holder's.

        Not conditional on the sidecar: who owns the rig is a fact from the
        launch decision, and a sidecar that stopped being written does not hand
        the instruments back.
        """
        try:
            names = list(getattr(self._manager, "names", ()) or ())
        except Exception:
            names = []
        return {name: {"state": OCCUPIED} for name in names}

    def _payload(self) -> dict[str, Any] | None:
        """The sidecar, or ``None`` if it is absent, unreadable or stale.

        Opened and closed inside the call, like the event reader and for a
        milder version of the same reason: the publisher replaces this file with
        :func:`os.replace`, which a held handle can refuse on Windows.
        """
        if self.path is None:
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception:
            self._warn()
            return None
        if not isinstance(payload, dict):
            return None
        return None if self._is_stale(payload) else payload

    def _is_stale(self, payload: dict[str, Any]) -> bool:
        completed = _parse_stamp(payload.get("completed_at"))
        if completed is None:
            # Published, but no read has ever completed — the campaign is up and
            # the numbers do not exist yet. Unknown, not old.
            return True
        return (self._now() - completed) >= self._stale_after_s

    @staticmethod
    def _readings(
        payload: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The five published fields, in the two shapes the widgets already take."""
        sidebar: dict[str, Any] = {}
        monitor: dict[str, Any] = {}
        if payload is None:
            return sidebar, monitor
        env = payload.get("env")
        if not isinstance(env, dict):
            return sidebar, monitor

        stage_sp = _number(env.get("stage_temp_sp_C"))
        stage_pv = _number(env.get("stage_temp_pv_C"))
        # The sidebar renders the stage pair or neither (it formats both or
        # writes "--"); the Monitoring tab takes them independently.
        if stage_sp is not None and stage_pv is not None:
            sidebar["temp_sp"] = stage_sp
            sidebar["temp_pv"] = stage_pv
        if stage_sp is not None:
            monitor["temp_sp"] = stage_sp
        if stage_pv is not None:
            monitor["temp_pv"] = stage_pv

        chamber = _number(env.get("chamber_air_C"))
        if chamber is not None:
            sidebar["chamber_temp"] = chamber
            monitor["chamber_temp"] = chamber

        rh_sp = _number(env.get("rh_sp_pct"))
        rh_pv = _number(env.get("rh_pv_pct"))
        if rh_sp is not None or rh_pv is not None:
            # NaN is the sidebar's own "unreadable" for humidity, so one half of
            # the pair can be published without inventing the other.
            sidebar["rh_sp"] = rh_sp if rh_sp is not None else math.nan
            sidebar["rh_pv"] = rh_pv if rh_pv is not None else math.nan
        if rh_sp is not None:
            monitor["rh_sp"] = rh_sp
        if rh_pv is not None:
            monitor["rh"] = rh_pv

        return sidebar, monitor

    def _warn(self) -> None:
        """First failure is a warning, the rest debug — a poll runs every 2 s."""
        if self._warned:
            logger.debug("conditions_sidecar_unreadable", path=str(self.path))
            return
        self._warned = True
        logger.warning("conditions_sidecar_unreadable", path=str(self.path),
                       exc_info=True)


def _number(value: Any) -> float | None:
    """A real number, or ``None`` for anything that must not be displayed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _parse_stamp(value: Any) -> float | None:
    """Epoch seconds from the publisher's ISO stamp; ``None`` if it is not one.

    Naive stamps are read as UTC, matching what the publisher writes. Kept here
    rather than imported because the equivalent in ``campaign_events`` is private
    to that module's reader, and a display path reaching into it would make a
    private helper part of the GUI's contract.
    """
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()
