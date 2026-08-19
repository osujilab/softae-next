"""Pause and Abort for a campaign running in **another** process.

Stop controls are scoped to their container. Rig-scale E-Stop belongs on the main
toolbar because it is about the rig; Pause and Abort are campaign-scale, so they
live in the tab that surfaces the campaign, beside the run controls the operator
was already looking at.

**This widget drives nothing.** It writes one small JSON file into the run
directory and waits. The campaign that owns the rig reads it on its own poll and
acts inside its own process, where the instrument sessions are — which is the
whole reason the channel is a file. In particular Abort's **park is not this
widget's job**: :meth:`~softae.core.autonomous_loop.AutonomousLoop.abort` sets
flags and the park happens on the loop's own thread of control once the trial has
actually stopped. A GUI that "helpfully" parked as well would be commanding
sessions it never opened, which is the invariant this whole arc exists to keep.

**It does not shell out either.** ``softae-campaign control`` is a thin wrapper
over :func:`~softae.core.campaign_events.write_control_request`, which this
process can call directly; spawning a subprocess to write a file we can write
ourselves would add a process, a PATH assumption and a failure mode for nothing.
What *is* shared with the CLI is the part that must not diverge: discovery
(:func:`~softae.core.campaign_discovery.find_running_campaign`) and the latency
sentences, which are quoted here as tooltips rather than paraphrased.

**Feedback is the ``control_ack`` record, not the button's own enabled state.**
The naive shape — disable on press, re-enable on a timer — is indistinguishable
for the four answers that matter most:

======================  ===================================================
``ignored_stale``       the campaign already acted on a newer request
``unreadable``          the campaign could not read the request at all
``ignored_pre_existing``  it was treated as left over from before the run
``handler_failed``      the handler raised; the run is **unchanged**
======================  ===================================================

Each of those means the rig did *not* do what the operator asked, and a button
that greys out and comes back says the same thing as one that worked. So a press
enters a pending state that is resolved only by an ack carrying our ``seq``, and
every outcome is shown, refusals loudest.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import structlog
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from softae.core.campaign_discovery import (
    ABORT_LATENCY_NOTE,
    CONTROL_LATENCY_NOTES,
    CampaignTarget,
    find_running_campaign,
)
from softae.core.campaign_events import (
    CONTROL_UNREADABLE,
    EventCursor,
    read_events,
    write_control_request,
)

logger = structlog.get_logger(__name__)

#: How an outcome reads to the operator. Every value the two ack sources can
#: produce is here — :class:`~softae.core.campaign_events.ControlWatcher` for the
#: channel-level answers and
#: :class:`~softae.core.autonomous_loop.AutonomousLoop` for the loop's — because
#: an outcome with no gloss would be rendered as a bare identifier at the exact
#: moment the operator needs a sentence.
CONTROL_OUTCOME_NOTES: dict[str, str] = {
    "applied": "the campaign accepted it.",
    "already_paused": "the campaign was already paused — nothing changed.",
    "not_paused": "the campaign was not paused — nothing changed.",
    "ended": "the run had already ended, so there was nothing to stop.",
    "ignored_stale": (
        "IGNORED — the campaign had already acted on a newer request. "
        "The run is unchanged."
    ),
    CONTROL_UNREADABLE: (
        "REFUSED — the campaign could not read the request. The run is "
        "unchanged; use the toolbar E-Stop if the rig must stop now."
    ),
    "ignored_pre_existing": (
        "IGNORED — the campaign treated the request as left over from before it "
        "started. The run is unchanged."
    ),
    "handler_failed": (
        "FAILED — the campaign received the request and raised while acting on "
        "it. The run is unchanged; use the toolbar E-Stop if the rig must stop "
        "now."
    ),
}

#: Outcomes after which the campaign is holding, whichever way it got there.
_PAUSED_AFTER = {"pause": ("applied", "already_paused")}
#: …and after which it is not.
_RUNNING_AFTER = {"resume": ("applied", "not_paused")}


def outcome_note(outcome: str) -> str:
    """The gloss for *outcome*, or an honest fallback for one we do not know.

    An unknown outcome is reported as unknown rather than as success: the answers
    this channel invents in future will arrive at an installed GUI before its
    vocabulary is updated, and guessing in the reassuring direction is how an
    operator walks away from a rig that did not stop.
    """
    return CONTROL_OUTCOME_NOTES.get(
        outcome, f"the campaign answered '{outcome}', which this window does not recognise."
    )


class CampaignControlRequester:
    """Write one control request; watch the stream until it is answered.

    Headless on purpose — no Qt here, so the ack-matching rule is testable
    without a window, and a script that wants the same guarantee can reuse it.

    **The matching rule, and its one subtlety.** An ack normally carries the
    ``seq`` of the request it answers, so a pending request is resolved by seq
    and nothing else. The exception is
    :meth:`~softae.core.campaign_events.ControlWatcher._ack` called with no
    request at all — the ``unreadable`` case, where the file could not be parsed
    and so *has* no seq to quote. That ack is matched by outcome instead, and it
    is sound because the cursor is snapshotted at write time: there is one
    ``control.json`` per run directory, ours is the newest thing written to it,
    and an ``unreadable`` recorded after our write is about our write. Dropping
    it because it carries no seq would silently swallow exactly the answer the
    operator most needs.
    """

    def __init__(
        self,
        run_dir: str,
        *,
        requested_by: str | None = None,
        writer: Callable[..., Any] = write_control_request,
        reader: Callable[..., Any] = read_events,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_dir = str(run_dir)
        self._requested_by = requested_by or f"softae GUI (pid {os.getpid()})"
        self._writer = writer
        self._reader = reader
        self._clock = clock
        self._pending: Any | None = None
        self._pending_at: float = 0.0
        self._cursor: EventCursor | None = None

    @property
    def pending(self) -> Any | None:
        """The request still waiting for an ack, or ``None``."""
        return self._pending

    @property
    def pending_age_s(self) -> float | None:
        """How long the pending request has gone unanswered, in seconds."""
        if self._pending is None:
            return None
        return max(0.0, self._clock() - self._pending_at)

    def request(self, action: str, *, reason: str = "") -> Any:
        """Write the request and start waiting for its ack.

        The stream position is snapshotted **before** the write, so an ack that
        was already on disk — the previous operator's, or this run's answer to
        the CLI — can never be mistaken for the answer to this press.
        """
        _, self._cursor = self._reader(self.run_dir)
        request = self._writer(
            self.run_dir, action, reason=reason, requested_by=self._requested_by
        )
        self._pending = request
        self._pending_at = self._clock()
        return request

    def poll(self) -> dict[str, Any] | None:
        """The ack answering the pending request, or ``None`` if none yet.

        Never raises: a run directory that vanished mid-wait leaves the request
        pending and visible, which is the honest state, rather than taking the
        tab down.
        """
        if self._pending is None:
            return None
        try:
            events, self._cursor = self._reader(self.run_dir, cursor=self._cursor)
        except Exception:
            logger.warning("campaign_control_poll_failed", run_dir=self.run_dir)
            return None
        for record in events:
            if record.get("type") != "control_ack":
                continue
            if self._answers_pending(record):
                self._pending = None
                return record
        return None

    def _answers_pending(self, ack: dict[str, Any]) -> bool:
        seq = ack.get("seq")
        if seq is None:
            return ack.get("outcome") == CONTROL_UNREADABLE
        return seq == getattr(self._pending, "seq", None)


class CampaignControlBar(QGroupBox):
    """Pause / Resume / Abort for the campaign that currently holds the rig.

    Resume is here because a Pause the GUI can request but not lift would send an
    operator to a terminal to undo a button they just pressed. It shares the
    Pause button rather than adding a third: the label follows the campaign's own
    acknowledged state, so it can only get out of step by the campaign answering
    something this window does not understand — in which case it says so.
    """

    #: Emitted with each resolved ack, so the surfacing tab can put the answer in
    #: its own log beside the campaign's other events.
    acknowledged = Signal(dict)

    _POLL_MS = 2000

    def __init__(
        self,
        *,
        run_dir: str | None = None,
        discover: Callable[[], CampaignTarget] = find_running_campaign,
        requester_factory: Callable[..., CampaignControlRequester] = CampaignControlRequester,
        confirm: Callable[[str, str], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Campaign controls (this campaign, not the rig)", parent)
        self._discover = discover
        self._requester_factory = requester_factory
        self._confirm = confirm if confirm is not None else self._ask
        self._explicit_run_dir = run_dir
        self._target: CampaignTarget | None = None
        self._requester: CampaignControlRequester | None = None
        self._paused = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self._lbl_target = QLabel("—")
        self._lbl_target.setWordWrap(True)
        root.addWidget(self._lbl_target)

        row = QHBoxLayout()
        self._btn_pause = QPushButton("⏸  Pause Campaign")
        self._btn_pause.setFixedHeight(30)
        self._btn_pause.clicked.connect(self._on_pause)
        row.addWidget(self._btn_pause)

        self._btn_abort = QPushButton("⏹  Abort Campaign")
        self._btn_abort.setFixedHeight(30)
        self._btn_abort.setToolTip(ABORT_LATENCY_NOTE)
        self._btn_abort.clicked.connect(self._on_abort)
        row.addWidget(self._btn_abort)
        root.addLayout(row)

        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        root.addWidget(self._lbl_status)

        self._apply_pause_labelling()
        self.refresh()

        self._timer = QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    # ── State ────────────────────────────────────────────────────────────────

    def set_run_dir(self, run_dir: str | None) -> None:
        """Name the campaign explicitly instead of discovering it from the lock.

        Used by a surface that started the campaign itself and therefore already
        knows where it put it. Passing ``None`` returns to discovery.
        """
        self._explicit_run_dir = run_dir
        self._requester = None
        self.refresh()

    @property
    def target(self) -> CampaignTarget | None:
        """The campaign a press would reach, as of the last :meth:`refresh`."""
        return self._target

    @property
    def pending(self) -> Any | None:
        """The request waiting for an ack, or ``None``."""
        return None if self._requester is None else self._requester.pending

    def refresh(self) -> None:
        """Re-read who holds the rig, resolve any pending ack, re-label."""
        if self._requester is not None and self._requester.pending is not None:
            ack = self._requester.poll()
            if ack is not None:
                self._resolve(ack)
            else:
                self._show_pending()
                return
        self._target = self._resolve_target()
        self._apply_enabled()

    def _resolve_target(self) -> CampaignTarget:
        if self._explicit_run_dir:
            return CampaignTarget(self._explicit_run_dir, self._explicit_run_dir)
        try:
            return self._discover()
        except Exception as exc:
            # The swallow belongs here rather than in the reader: a GUI timer
            # that cannot read the lock must not take the tab down, and "I could
            # not tell" resolves to offering no control — never to offering one
            # that would write into a directory we failed to identify.
            logger.warning("campaign_discovery_failed", error=str(exc))
            return CampaignTarget(
                None, f"the rig lock could not be read ({exc.__class__.__name__})."
            )

    # ── Presses ──────────────────────────────────────────────────────────────

    def _on_pause(self) -> None:
        self._send("resume" if self._paused else "pause")

    def _on_abort(self) -> None:
        # Confirmed, because it ends a run that may be hours in — and the
        # confirmation quotes the same sentence as the tooltip rather than
        # softening it.
        if not self._confirm(
            "Abort this campaign?",
            f"{self._target_name()}\n\n{ABORT_LATENCY_NOTE}",
        ):
            return
        self._send("abort", reason="operator abort (GUI)")

    def _send(self, action: str, *, reason: str = "") -> None:
        target = self._resolve_target()
        self._target = target
        if not target.controllable:
            self._apply_enabled()
            self._lbl_status.setText(target.refusal)
            return
        if self._requester is None or self._requester.run_dir != target.run_dir:
            self._requester = self._requester_factory(target.run_dir)
        try:
            request = self._requester.request(action, reason=reason)
        except Exception as exc:
            # write_control_request raises rather than degrading, deliberately:
            # the write is the operator's only evidence their button did
            # anything, so a failure here is the one failure they must see.
            logger.warning("campaign_control_write_failed", action=action, error=str(exc))
            self._lbl_status.setText(
                f"{action} could NOT be written ({exc.__class__.__name__}: {exc}). "
                "The campaign has not been asked to stop."
            )
            self._apply_enabled()
            return
        logger.info("campaign_control_requested", action=action, seq=request.seq)
        self._show_pending()

    def _resolve(self, ack: dict[str, Any]) -> None:
        action = str(ack.get("action") or "")
        outcome = str(ack.get("outcome") or "")
        if outcome in _PAUSED_AFTER.get(action, ()):
            self._paused = True
        elif outcome in _RUNNING_AFTER.get(action, ()):
            self._paused = False
        elif action == "abort" and outcome == "applied":
            self._paused = False
        self._apply_pause_labelling()
        self._lbl_status.setText(f"{action or 'request'} → {outcome}: {outcome_note(outcome)}")
        self.acknowledged.emit(dict(ack))

    # ── Rendering ────────────────────────────────────────────────────────────

    def _show_pending(self) -> None:
        pending = self.pending
        if pending is None:
            return
        age = self._requester.pending_age_s or 0.0
        self._btn_pause.setEnabled(False)
        self._btn_abort.setEnabled(False)
        self._lbl_status.setText(
            f"{pending.action} requested (seq {pending.seq}) — waiting for the "
            f"campaign to acknowledge… {age:.0f}s"
        )

    def _apply_enabled(self) -> None:
        target = self._target
        live = bool(target is not None and target.controllable)
        self._btn_pause.setEnabled(live)
        self._btn_abort.setEnabled(live)
        self._lbl_target.setText(
            self._target_name() if live else (target.refusal if target else "—")
        )

    def _apply_pause_labelling(self) -> None:
        action = "resume" if self._paused else "pause"
        self._btn_pause.setText(
            "▶  Resume Campaign" if self._paused else "⏸  Pause Campaign"
        )
        self._btn_pause.setToolTip(CONTROL_LATENCY_NOTES[action])

    def _target_name(self) -> str:
        target = self._target
        return "the running campaign" if target is None else target.detail

    def _ask(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes
