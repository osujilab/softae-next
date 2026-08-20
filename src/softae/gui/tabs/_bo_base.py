"""Shared base for the two Bayesian-optimization tabs.

:class:`BOTabBase` factors the plumbing common to the offline **BO Simulator**
and the live **Live BO Campaign** tabs: a daemon worker thread, a two-axis
convergence canvas, a log pane, JSON config save/load, and run/abort button
management.  Everything that carries per-run state is an *instance* attribute so
a Simulator run and a Live run can execute **concurrently** in the background
without sharing mutable state.

Qt ``Signal`` objects are unavoidably class attributes (that is how Qt's meta
object system works), but they are immutable descriptors — each *instance* binds
its own emission and its own connected slots/state, so two live instances never
couple.  The hard rule this module keeps is: **no mutable class-level data**
(no lists, dicts, or run flags on the class); all such state lives on ``self``.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Callable

import structlog
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from softae.gui.daemon_runner import DaemonRunnerMixin

if TYPE_CHECKING:
    from softae.core.data_store import DataStore

logger = structlog.get_logger(__name__)


class BOTabBase(DaemonRunnerMixin, QWidget):
    """Shared base widget for the BO Simulator and Live BO Campaign tabs.

    Subclasses implement ``_build_ui``, ``_build_config``, ``_populate_from_config``,
    ``_config_from_json``, ``_on_run`` and their worker function, and may add
    domain-specific plots.  The base owns the daemon-thread plumbing, the
    convergence canvas, the log pane, config save/load and button-state helpers.
    """

    # ── Worker → GUI signals (class-level descriptors; per-instance emission) ──
    _sig_log = Signal(str)                       # one log line to append
    _sig_done = Signal(bool, str)                # (success, message)
    _sig_step = Signal(int, float, float)        # (iteration, primary, secondary)

    #: Title stem used by the config save/load dialogs (subclasses override).
    _CONFIG_TITLE = "Config"

    def __init__(
        self,
        *,
        data_store: "DataStore | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data_store = data_store

        # ── per-instance run state (never class-level) ──
        self._thread: threading.Thread | None = None
        self._abort_requested: bool = False
        self._result: Any = None
        self._runner: Any = None

        # ── per-instance convergence series buffers + redraw flag ──
        self._xs: list[float] = []
        self._primary_series: list[float] = []
        self._secondary_series: list[float] = []
        self._conv_dirty: bool = False

        # Axis-label state (applied on every reset / redraw).
        self._conv_primary_label: str = "primary"
        self._conv_secondary_label: str = "secondary"
        self._conv_xlabel: str = "iteration"

        # Marshal worker-thread events onto the GUI thread.
        self._sig_log.connect(self._log_line)
        self._sig_done.connect(self._on_done)
        self._sig_step.connect(self._on_step)

    # ── DaemonRunnerMixin hooks (implemented once here) ─────────────────────

    def _abort_run_impl(self) -> None:
        """Set the cooperative abort flag; nudge the runner if it exposes abort()."""
        self._abort_requested = True
        runner = getattr(self, "_runner", None)
        if runner is not None:
            abort = getattr(runner, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass

    def _runner_thread(self):  # -> threading.Thread | None
        return self._thread

    # ── Convergence canvas (two stacked axes) ───────────────────────────────

    def _make_convergence_canvas(
        self,
        *,
        primary_label: str,
        secondary_label: str,
        xlabel: str,
    ) -> QWidget:
        """Build the shared two-axis convergence canvas; store axes on ``self``.

        Returns the :class:`FigureCanvasQTAgg` widget (subclasses embed it and
        may add their own export button).
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._conv_primary_label = primary_label
        self._conv_secondary_label = secondary_label
        self._conv_xlabel = xlabel

        self._conv_fig = Figure(tight_layout=True)
        self._ax_primary = self._conv_fig.add_subplot(211)
        self._ax_primary.set_ylabel(primary_label)
        self._ax_secondary = self._conv_fig.add_subplot(212)
        self._ax_secondary.set_xlabel(xlabel)
        self._ax_secondary.set_ylabel(secondary_label)
        self._conv_canvas = FigureCanvasQTAgg(self._conv_fig)
        self._conv_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        return self._conv_canvas

    def _reset_convergence(self) -> None:
        """Clear the convergence buffers and redraw an empty canvas."""
        self._xs = []
        self._primary_series = []
        self._secondary_series = []
        self._conv_dirty = False
        for ax in (self._ax_primary, self._ax_secondary):
            ax.cla()
        self._ax_primary.set_ylabel(self._conv_primary_label)
        self._ax_secondary.set_xlabel(self._conv_xlabel)
        self._ax_secondary.set_ylabel(self._conv_secondary_label)
        self._conv_canvas.draw_idle()

    def _on_step(self, iteration: int, primary: float, secondary: float) -> None:
        """``_sig_step`` slot (GUI thread): append to buffers, coalesce redraw."""
        self._xs.append(float(iteration))
        self._primary_series.append(float(primary))
        self._secondary_series.append(float(secondary))
        if not self._conv_dirty:
            self._conv_dirty = True
            QTimer.singleShot(0, self._flush_convergence)

    def _flush_convergence(self) -> None:
        if not self._conv_dirty:
            return
        self._conv_dirty = False
        self._ax_primary.cla()
        self._ax_primary.set_ylabel(self._conv_primary_label)
        self._ax_primary.plot(self._xs, self._primary_series, "o-", color="#1f77b4", ms=3)
        self._ax_secondary.cla()
        self._ax_secondary.set_xlabel(self._conv_xlabel)
        self._ax_secondary.set_ylabel(self._conv_secondary_label)
        if any(math.isfinite(v) for v in self._secondary_series):
            self._ax_secondary.plot(
                self._xs, self._secondary_series, "s-", color="#2ca02c", ms=3
            )
        self._conv_canvas.draw_idle()

    # ── Log pane ─────────────────────────────────────────────────────────────

    def _make_log_pane(self, *, title: str = "Log") -> QWidget:
        """Build the shared read-only log group box; store the QTextEdit on self."""
        grp = QGroupBox(title)
        lay = QVBoxLayout(grp)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFontFamily("Courier New")
        lay.addWidget(self._log)
        return grp

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    # ── Control bar (run / abort / status / progress / config / export) ──────

    def _make_control_bar(
        self, *, run_label: str = "▶  Run", with_export: bool = True,
        with_abort: bool = True,
    ) -> QWidget:
        """Build the shared control bar; register buttons as ``self`` attributes.

        ``with_abort=False`` omits the in-process Abort. It exists for a surface
        whose run is **not** in this process: this button sets a cooperative flag
        a worker thread reads, so against a detached campaign it would grey
        itself out, log "stopping after current step", and reach nothing at all.
        A stop control that reports success and does nothing is worse than an
        absent one — the surface offers the request-based
        :class:`~softae.gui.widgets.campaign_control.CampaignControlBar` instead.
        """
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 2, 2, 2)

        ctrl = QHBoxLayout()
        self._btn_run = QPushButton(run_label)
        self._btn_run.setFixedHeight(34)
        self._btn_run.clicked.connect(self._on_run)
        ctrl.addWidget(self._btn_run)
        if with_abort:
            self._btn_abort = QPushButton("■  Abort")
            self._btn_abort.setFixedHeight(34)
            self._btn_abort.setEnabled(False)
            self._btn_abort.clicked.connect(self._on_abort)
            ctrl.addWidget(self._btn_abort)
        self._lbl_status = QLabel("Idle")
        self._lbl_status.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ctrl.addWidget(self._lbl_status)
        v.addLayout(ctrl)

        self._progress = QProgressBar()
        v.addWidget(self._progress)

        cfg_row = QHBoxLayout()
        b_save = QPushButton("Save Config…")
        b_save.clicked.connect(self._on_save_config)
        cfg_row.addWidget(b_save)
        b_load = QPushButton("Load Config…")
        b_load.clicked.connect(self._on_load_config)
        cfg_row.addWidget(b_load)
        if with_export:
            self._btn_export = QPushButton("Export Results…")
            self._btn_export.clicked.connect(self._on_export_results)
            self._btn_export.setEnabled(False)
            cfg_row.addWidget(self._btn_export)
        cfg_row.addStretch()
        v.addLayout(cfg_row)
        return w

    def _set_running(self, running: bool) -> None:
        """Toggle run/abort/export buttons for the run state (all optional)."""
        btn_run = getattr(self, "_btn_run", None)
        if btn_run is not None:
            btn_run.setEnabled(not running)
        btn_abort = getattr(self, "_btn_abort", None)
        if btn_abort is not None:
            btn_abort.setEnabled(running)
        btn_export = getattr(self, "_btn_export", None)
        if running and btn_export is not None:
            btn_export.setEnabled(False)

    def _start_worker(self, target: Callable, *args: Any, name: str) -> bool:
        """Spawn a daemon worker thread (guards against a double-start).

        Returns ``True`` when a worker was started, ``False`` if one is already
        running.  Clears the abort flag and flips the UI into the running state.
        """
        if self._thread is not None and self._thread.is_alive():
            return False
        self._abort_requested = False
        self._thread = threading.Thread(
            target=target, args=args, daemon=True, name=name
        )
        self._thread.start()
        self._set_running(True)
        return True

    # ── Run / abort / done ───────────────────────────────────────────────────

    def _on_abort(self) -> None:
        self._abort_requested = True
        lbl = getattr(self, "_lbl_status", None)
        if lbl is not None:
            lbl.setText("Aborting…")
        btn_abort = getattr(self, "_btn_abort", None)
        if btn_abort is not None:
            btn_abort.setEnabled(False)
        self._sig_log.emit("  ⚠ Abort requested — stopping after current step")

    def _on_done(self, success: bool, message: str) -> None:
        self._set_running(False)
        lbl = getattr(self, "_lbl_status", None)
        if lbl is not None:
            lbl.setText(message)
        btn_export = getattr(self, "_btn_export", None)
        if btn_export is not None:
            btn_export.setEnabled(success and self._result is not None)
        self._log.append(("\n✓ " if success else "\n✗ ") + message)
        self._on_done_extra(success)

    def _on_done_extra(self, success: bool) -> None:
        """Hook for subclass post-run redraws (default no-op)."""

    def _on_run(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ── Config save / load (generic; subclass supplies (de)serialisers) ──────

    def _on_save_config(self) -> None:
        try:
            cfg = self._build_config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {self._CONFIG_TITLE}", "",
            f"{self._CONFIG_TITLE} (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(cfg.to_json())
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {self._CONFIG_TITLE}", "",
            f"{self._CONFIG_TITLE} (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = self._config_from_json(fh.read())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load config:\n{exc}")
            return
        self._populate_from_config(cfg)

    def _build_config(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _populate_from_config(self, cfg: Any) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _config_from_json(self, text: str) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    # ── Exports (generic) ────────────────────────────────────────────────────

    def _export_fig(self, fig, default_name: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", default_name, "PNG (*.png);;All files (*)"
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_export_results(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "bo_result.json", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._result.to_json())
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Public API ───────────────────────────────────────────────────────────

    def set_data_store(self, data_store: "DataStore") -> None:
        self._data_store = data_store
