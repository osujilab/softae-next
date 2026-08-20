"""The sidebar's purge line — paint only.

Every decision this widget renders was already made by
:func:`~softae.gui.widgets.purge_indicator.purge_indicator`. It has the same
one-argument-slot shape as :meth:`MonitorSidebar.update_rig_owner`, and for the
reason that slot's docstring gives: **the widget renders a decision, it does not
make one.**

**On the animation.** The sidebar is looked at all day, next to a person working
at a bench. A blink is unpleasant to sit beside and becomes something to cover
with a sticky note, so attention is a **slow pulse** — a two-second traverse
between amber and red on an eased curve — carrying most of its weight in the
colour shift rather than the motion. It runs only while a purge is both overdue
and unacknowledged, and stops on either exit.

Repainting is deliberately coarse (8 Hz, not a frame timer). Each step reparses
one stylesheet, and this label is on screen for the whole session; sixteen steps
per cycle is smooth enough for a colour blend and costs a fifth of what a
``QPropertyAnimation`` on a colour property would.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel

#: Seconds for one full base → peak → base traverse.
PULSE_PERIOD_S = 2.0
#: Repaint cadence. See the module docstring for why this is not 60 Hz.
_PULSE_STEP_MS = 125

#: Colour per state. Everything except ``overdue`` is deliberately drawn from
#: the palette the neighbouring status labels already use, so the purge line
#: does not read as louder than the rig-owner line sitting above it.
_COLOURS = {
    "not_ours": "#888888",
    "unconfigured": "#888888",
    "scheduled": "#666666",
    # Neutral, and that is the ruling: dry run is the shipped default, and a
    # permanent amber for the normal case trains the eye to skip this label.
    "dry_run": "#666666",
    "purged": "#2a8a5e",
    "near": "#2277bb",
    "overdue": "#c07830",
}
_DEFAULT_COLOUR = "#888888"

#: What ``overdue`` pulses *to*. The shift from amber is the attention-getting
#: part; the motion only stops it reading as a static warning.
OVERDUE_BASE = _COLOURS["overdue"]
OVERDUE_PEAK = "#c62828"


class PurgeBadge(QLabel):
    """The purge schedule, permanently on screen.

    Clicking it acknowledges an overdue purge — the operator's "I know" — which
    is one of the two ways attention ends. The other is the purge running.
    """

    #: The operator clicked. The window stamps the acknowledgement; this widget
    #: keeps no state about it, so a stale badge cannot silence a live problem.
    acknowledged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Purge: --", parent)
        font = QFont()
        font.setPointSize(8)
        self.setFont(font)
        self.setWordWrap(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = "scheduled"
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(_PULSE_STEP_MS)
        self._pulse_timer.timeout.connect(self._pulse_step)
        self._pulse_started_at = 0.0
        self._apply_colour(_COLOURS["scheduled"], bold=False)

    def update_purge(self, indicator) -> None:
        """Render one :class:`~softae.gui.widgets.purge_indicator.PurgeIndicator`."""
        self._state = indicator.state
        self.setText(indicator.headline)
        self.setToolTip(indicator.detail)
        if indicator.attention:
            self._start_pulse()
        else:
            self._stop_pulse()
            self._apply_colour(_COLOURS.get(indicator.state, _DEFAULT_COLOUR),
                               bold=indicator.state == "overdue")

    # ── Attention ────────────────────────────────────────────────────────────

    def _start_pulse(self) -> None:
        if self._pulse_timer.isActive():
            return
        self._pulse_started_at = time.monotonic()
        self._pulse_timer.start()
        self._pulse_step()

    def _stop_pulse(self) -> None:
        self._pulse_timer.stop()

    def _pulse_step(self) -> None:
        phase = (time.monotonic() - self._pulse_started_at) / PULSE_PERIOD_S
        # Raised cosine: eased at both ends, so the traverse reads as a breath
        # rather than a switch.
        weight = (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
        self._apply_colour(_blend(OVERDUE_BASE, OVERDUE_PEAK, weight), bold=True)

    def _apply_colour(self, colour: str, *, bold: bool) -> None:
        self.setStyleSheet(
            f"color: {colour}; font-size: 8pt;"
            f"{' font-weight: bold;' if bold else ''}"
        )

    # ── Acknowledgement ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # noqa: N802  (Qt override)
        self.acknowledged.emit()
        super().mousePressEvent(event)


def _blend(start: str, end: str, weight: float) -> str:
    """Linear ``#rrggbb`` blend, *weight* 0 → *start*, 1 → *end*."""
    weight = min(1.0, max(0.0, weight))
    channels = (
        round(int(start[i:i + 2], 16)
              + weight * (int(end[i:i + 2], 16) - int(start[i:i + 2], 16)))
        for i in (1, 3, 5)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)
