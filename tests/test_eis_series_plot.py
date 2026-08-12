"""Tests for embedded EIS fit plotting and the multi-channel series window."""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.circuit_fitting import FitResult, plot_eis_fit
from softae.analysis.eis_data import EISResult
from softae.gui.widgets.eis_series_plot import EisSeriesPlotWidget


def _eis(channel: int = 1) -> EISResult:
    freq = np.geomspace(1e4, 1.0, 21)
    omega = 2 * np.pi * freq
    z = 5e4 + 1 / (1e-7 * (1j * omega) ** 0.7) + 1e6 / (1 + 1j * omega * 1e6 * 1e-10)
    return EISResult.from_arrays(channel=channel, f=freq, z_real=z.real, z_imag_neg=-z.imag)


def _fit(success: bool = True) -> FitResult:
    return FitResult(
        model_name="simpleSalt",
        parameters=np.array([5e4, 1e-7, 0.7, 1e6, 1e-10]),
        R0=5e4, R1=1e6, R0_guess=5e4, R1_guess=1e6,
        z_indices=[0, 3], success=success,
    )


# ── plot_eis_fit embedding refactor ─────────────────────────────────────────

def test_plot_eis_fit_draws_into_provided_figure():
    """Passing fig draws into it (no pyplot window) and returns the same figure."""
    from matplotlib.figure import Figure

    fig = Figure()
    out = plot_eis_fit(_eis(), _fit(success=False), show=False, fig=fig)
    assert out is fig
    assert len(fig.axes) >= 2  # Nyquist + Bode (+ residual placeholders)


# ── multi-channel series window ─────────────────────────────────────────────

def _payloads(channels, *, fit):
    return [
        {"channel": ch, "eis_result": _eis(ch),
         "fit_result": (_fit() if fit else None)}
        for ch in channels
    ]


def test_pages_are_one_per_channel_plus_a_sigma_summary(qtbot):
    payloads = _payloads([2, 4, 5], fit=True)
    win = EisSeriesPlotWidget(payloads, sigma_fn=lambda f: 1e-4 if f else None)
    qtbot.addWidget(win)
    # A single flip-through: scrollbar spans channels + the σ page.
    assert win._n_pages == len(payloads) + 1
    assert win._scrollbar.maximum() == len(payloads)  # last index = σ page
    assert win._scrollbar.minimum() == 0


def test_flipping_renders_each_page_and_updates_label(qtbot):
    payloads = _payloads([2, 4, 5], fit=True)
    win = EisSeriesPlotWidget(payloads, sigma_fn=lambda f: 1e-4 if f else None)
    qtbot.addWidget(win)

    for i, ch in enumerate(p["channel"] for p in payloads):
        win._render_page(i)
        assert f"Channel {ch}" in win._label.text()
        assert f"({i + 1}/{win._n_pages})" in win._label.text()

    win._render_page(win._n_pages - 1)                 # the σ summary page
    assert "σ vs channel" in win._label.text()


def test_prev_next_enable_states_at_ends(qtbot):
    win = EisSeriesPlotWidget(_payloads([1, 2], fit=True), sigma_fn=lambda f: 1e-4)
    qtbot.addWidget(win)
    win._render_page(0)
    assert not win._btn_prev.isEnabled() and win._btn_next.isEnabled()
    win._render_page(win._n_pages - 1)
    assert win._btn_prev.isEnabled() and not win._btn_next.isEnabled()


def test_next_button_advances_one_page(qtbot):
    win = EisSeriesPlotWidget(_payloads([3, 6], fit=True), sigma_fn=lambda f: 1e-4)
    qtbot.addWidget(win)
    assert win._page == 0
    win._btn_next.click()
    assert win._page == 1 and win._scrollbar.value() == 1


def test_handles_channels_without_fits(qtbot):
    win = EisSeriesPlotWidget(_payloads([1, 2], fit=False), sigma_fn=lambda f: None)
    qtbot.addWidget(win)
    win._render_page(0)                                # raw-plot path, no exception
    win._render_page(win._n_pages - 1)                # σ page with placeholder
    assert "σ vs channel" in win._label.text()


def _sigma_page_text(win) -> str:
    return " ".join(t.get_text() for ax in win._fig.axes for t in ax.texts)


def test_sigma_page_blames_missing_inputs_not_the_fits(qtbot):
    """Good fits but σ uncomputable (geometry/K unset) -> actionable message."""
    win = EisSeriesPlotWidget(_payloads([2, 4, 5], fit=True), sigma_fn=lambda f: None)
    qtbot.addWidget(win)
    win._render_page(win._n_pages - 1)
    text = _sigma_page_text(win)
    assert "σ could not be computed" in text
    assert "geometry" in text or "cell constant" in text


def test_sigma_page_reports_no_fits_distinctly(qtbot):
    """No successful fits -> a different, fit-focused message."""
    win = EisSeriesPlotWidget(_payloads([1, 2], fit=False), sigma_fn=lambda f: None)
    qtbot.addWidget(win)
    win._render_page(win._n_pages - 1)
    assert "No successful fits" in _sigma_page_text(win)


# ── One plotting style, fit or no fit ───────────────────────────────────────


class TestUnfittedSpectraKeepTheSameStyle:
    """Turning auto-fit off changed what the plot *looked like*, not just its content.

    The single-channel path fell through to the pico driver's own ``eis_plotdata``,
    which shares none of this palette, axes or conventions; the series path drew a
    hand-rolled pair of axes plotting ``|Z|`` where the fitted view plots ``Z'`` and
    ``-Z''`` separately, with no phase axis at all. The same measurement looked like
    it came from a different instrument depending on a checkbox.

    Residuals of no model genuinely do not exist, so those panes are dropped — but
    everything that *is* shared must be identical.
    """

    def _panes(self):
        eis = _eis(7)
        return plot_eis_fit(eis, None, show=False), plot_eis_fit(eis, _fit(), show=False)

    def test_a_missing_fit_is_accepted_rather_than_demanded(self):
        fig = plot_eis_fit(_eis(), None, show=False)
        assert fig is not None

    def test_the_nyquist_pane_is_identical(self):
        raw, fitted = self._panes()
        a, b = raw.axes[0], fitted.axes[0]
        assert (a.get_xlabel(), a.get_ylabel(), a.get_title()) == (
            b.get_xlabel(), b.get_ylabel(), b.get_title())

    def test_the_bode_pane_keeps_its_log_axes_and_labels(self):
        raw, fitted = self._panes()
        a, b = raw.axes[1], fitted.axes[1]
        assert a.get_xscale() == b.get_xscale() == "log"
        assert a.get_yscale() == b.get_yscale() == "log"
        assert (a.get_xlabel(), a.get_ylabel()) == (b.get_xlabel(), b.get_ylabel())

    def test_the_phase_axis_survives_without_a_fit(self):
        # The old raw renderer dropped it entirely, which is the most useful single
        # trace for spotting a blocking tail or a floating reference.
        def twin(fig):
            bode = fig.axes[1]
            return [ax for ax in fig.axes
                    if ax is not bode and ax.bbox.bounds == bode.bbox.bounds]

        raw, fitted = self._panes()
        assert len(twin(raw)) == 1
        assert twin(raw)[0].get_ylabel() == twin(fitted)[0].get_ylabel()

    def test_measured_points_use_the_same_markers_and_colours(self):
        def colours(ax):
            return [tuple(np.round(c.get_facecolor()[0][:3], 4)) for c in ax.collections]

        raw, fitted = self._panes()
        assert colours(raw.axes[0]) == colours(fitted.axes[0])
        assert colours(raw.axes[1]) == colours(fitted.axes[1])

    def test_no_model_line_is_drawn_when_there_is_no_model(self):
        raw, fitted = self._panes()
        assert len(raw.axes[0].lines) == 0
        assert len(fitted.axes[0].lines) == 1

    def test_the_residual_row_is_dropped_rather_than_left_empty(self):
        raw, fitted = self._panes()
        assert len(raw.axes) < len(fitted.axes)

    def test_a_failed_fit_still_shows_its_residual_panes_saying_so(self):
        # Distinct from "no fit requested": asking for a fit and not getting one is
        # worth seeing, so that layout is kept and labelled.
        fig = plot_eis_fit(_eis(), _fit(success=False), show=False)
        assert len(fig.axes) > 3
        assert "fit failed" in fig._suptitle.get_text()

    def test_the_title_says_which_state_it_is_in(self):
        raw, fitted = self._panes()
        assert "auto-fit off" in raw._suptitle.get_text()
        assert "simpleSalt" in fitted._suptitle.get_text()
