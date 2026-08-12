"""Page-by-page EIS plots for a multi-channel manual run.

A single embedded matplotlib canvas that flips **one page at a time** — driven by
a horizontal scrollbar (plus Prev/Next), mirroring the per-channel page-flip the
Arrhenius 3D pop-out uses.  Pages are: one Nyquist/Bode/residual fit plot per
channel (:func:`softae.analysis.circuit_fitting.plot_eis_fit`), then a final
"fitted conductivity vs channel" summary.  One persistent navigation toolbar
(home / pan / zoom / save) acts on whichever page is shown.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class EisSeriesPlotWidget(QWidget):
    """A page-flipping viewer of per-channel EIS plots + a σ summary page.

    Parameters
    ----------
    results :
        One dict per channel, each with ``"channel"``, ``"eis_result"`` and
        ``"fit_result"`` (the manual-EIS worker payloads).
    sigma_fn :
        ``fit_result -> float | None`` — the fitted conductivity for a channel
        (``None`` when it can't be computed); used for the summary page.
    """

    def __init__(
        self,
        results: Sequence[dict[str, Any]],
        *,
        sigma_fn: Callable[[Any], float | None],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._results = list(results)
        self._sigma_fn = sigma_fn
        self._n_pages = len(self._results) + 1  # one per channel + the σ summary

        self.setWindowTitle("Manual EIS — multi-channel results")
        self.resize(840, 740)

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # No figure-level tight_layout: the fit page uses its own gridspec, and the
        # Nyquist's equal-aspect axis is incompatible with auto tight_layout.
        self._fig = Figure()
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)  # persists across pages
        v.addWidget(self._toolbar)
        v.addWidget(self._canvas, stretch=1)

        # ── page-flip navigation (scrollbar + Prev/Next + label) ──────────────
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.clicked.connect(lambda: self._goto(self._page - 1))
        nav.addWidget(self._btn_prev)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumWidth(200)
        nav.addWidget(self._label, stretch=1)

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(lambda: self._goto(self._page + 1))
        nav.addWidget(self._btn_next)
        v.addLayout(nav)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setRange(0, self._n_pages - 1)
        self._scrollbar.setSingleStep(1)
        self._scrollbar.setPageStep(1)   # a trough click flips exactly one page
        self._scrollbar.valueChanged.connect(self._on_scroll)
        v.addWidget(self._scrollbar)

        # Debounce scrollbar drags: render only after the value settles (fit plots
        # are relatively expensive), matching the Arrhenius 3D pop-out.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._pending = 0
        self._debounce.timeout.connect(lambda: self._render_page(self._pending))

        self._page = 0
        self._render_page(0)

    # ── navigation ────────────────────────────────────────────────────────

    def _goto(self, idx: int) -> None:
        """Flip to page *idx* immediately (Prev/Next buttons)."""
        idx = max(0, min(self._n_pages - 1, idx))
        self._scrollbar.blockSignals(True)
        self._scrollbar.setValue(idx)
        self._scrollbar.blockSignals(False)
        self._render_page(idx)

    def _on_scroll(self, idx: int) -> None:
        self._pending = idx
        self._debounce.start()

    # ── rendering ─────────────────────────────────────────────────────────

    def _render_page(self, idx: int) -> None:
        idx = max(0, min(self._n_pages - 1, idx))
        self._page = idx
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < self._n_pages - 1)
        self._fig.clear()
        try:
            if idx < len(self._results):
                res = self._results[idx]
                self._label.setText(
                    f"Channel {res.get('channel')}   ({idx + 1}/{self._n_pages})")
                eis = res.get("eis_result")
                fit = res.get("fit_result")
                if fit is not None:
                    from softae.analysis.circuit_fitting import plot_eis_fit
                    plot_eis_fit(eis, fit, show=False, fig=self._fig)
                else:
                    _draw_raw(self._fig, eis)
            else:
                self._label.setText(
                    f"σ vs channel   ({self._n_pages}/{self._n_pages})")
                self._draw_sigma(self._fig)
        except Exception as exc:  # never let one bad page sink the viewer
            self._fig.clear()
            self._fig.add_subplot(111).text(
                0.5, 0.5, f"Plot error:\n{exc}", ha="center", va="center")
        self._canvas.draw()  # synchronous — no deferred draw left pending

    def _draw_sigma(self, fig: Figure) -> None:
        ax = fig.add_subplot(111)
        channels: list[int] = []
        sigmas: list[float] = []
        n_fit = 0
        for res in self._results:
            fit = res.get("fit_result")
            if fit is None:
                continue
            n_fit += 1
            sigma = self._sigma_fn(fit)
            if sigma is not None and sigma > 0:
                channels.append(int(res.get("channel", 0)))
                sigmas.append(float(sigma))
        if channels:
            ax.plot(channels, sigmas, "o-", color="#0072B2", markersize=6)
            ax.set_yscale("log")
            ax.set_xticks(channels)
        else:
            # Distinguish "no fits" from "fits present but σ not computable" so the
            # cause (usually unset electrode geometry / cell constant) is actionable
            # rather than reading as a fit failure.
            if n_fit == 0:
                msg = "No successful fits — nothing to compute σ from."
            else:
                msg = (
                    f"{n_fit} channel(s) fitted, but σ could not be computed.\n"
                    "Set the σ inputs in the Manual tab — electrode geometry\n"
                    "(L · t · w) or the cell constant K — then re-run."
                )
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
        ax.set_xlabel("Channel")
        ax.set_ylabel("σ (S/cm)")
        ax.set_title("Fitted conductivity vs channel")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        fig.tight_layout()


def _draw_raw(fig: Figure, eis: Any) -> None:
    """Nyquist + Bode for a channel with no fit (auto-fit off / failed).

    Delegates to the same renderer the fitted view uses, with no fit passed. This
    used to be a hand-rolled pair of axes that plotted ``|Z|`` where the fitted view
    plots ``Z′`` and ``−Z″`` separately, and dropped the phase axis entirely — so the
    same channel looked like a different measurement depending on a checkbox.
    """
    from softae.analysis.circuit_fitting import plot_eis_fit

    plot_eis_fit(eis, None, show=False, fig=fig)
