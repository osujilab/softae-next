"""Sweep status widget for display in the Monitor tab.

:class:`SweepStatusWidget` shows live progress of an Arrhenius temperature
sweep.  Attach it to a running :class:`~softae.workflows.workflow_executor.WorkflowExecutor`
via :meth:`attach_executor`; detach between runs.

Example wiring (inside MonitoringTab)::

    self._sweep_status = SweepStatusWidget()
    # ... added to layout ...

    # When a sweep is launched:
    self._sweep_status.attach_executor(executor, n_temperatures=7, channels=[1, 2])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from softae.workflows.workflow_executor import WorkflowExecutor
    from softae.workflows.workflow_model import WorkflowStep


class SweepStatusWidget(QGroupBox):
    """Compact status panel for a running temperature-stepped EIS sweep.

    Displays:

    * Temperature setpoint and current measured value.
    * Overall sweep progress bar.
    * A text status line (e.g. ``"Waiting for T = 55 °C…"``).

    The widget is initially idle.  Call :meth:`attach_executor` when a sweep
    starts and :meth:`detach_executor` when it finishes.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Arrhenius Sweep", parent)
        self._executor: "WorkflowExecutor | None" = None
        self._n_steps: int = 0
        self._steps_done: int = 0
        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        self._lbl_t_sp = QLabel("T setpoint: -- °C")
        self._lbl_t_pv = QLabel("T current:  -- °C")
        self._lbl_status = QLabel("Idle")
        self._lbl_status.setWordWrap(True)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m steps")

        for lbl in (self._lbl_t_sp, self._lbl_t_pv, self._lbl_status):
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(lbl)

        layout.addWidget(self._progress)

    # ── Public API ───────────────────────────────────────────────────────

    def attach_executor(
        self,
        executor: "WorkflowExecutor",
        *,
        n_temperatures: int = 0,
        channels: list[int] | None = None,
    ) -> None:
        """Wire the widget to *executor* callbacks.

        Parameters
        ----------
        executor : WorkflowExecutor
            The executor driving the sweep.
        n_temperatures : int
            Number of temperature setpoints.  Used to compute total steps
            (``n_temperatures × len(channels)`` EIS steps + temperature steps).
        channels : list[int] or None
            Channel list; used for step-count calculation.
        """
        self._executor = executor

        n_ch = len(channels) if channels else 1
        # Each temperature produces: 1 set_temp + 1 wait_temp + n_ch EIS steps
        self._n_steps = n_temperatures * (2 + n_ch)
        self._steps_done = 0

        self._progress.setRange(0, max(self._n_steps, 1))
        self._progress.setValue(0)
        self._progress.setFormat(f"%v / {self._n_steps} steps")

        # Register callbacks (non-destructive to existing ones)
        _prev_start = executor.on_step_start
        _prev_complete = executor.on_step_complete
        _prev_state = executor.on_state_change

        def _on_start(step: "WorkflowStep", idx: int, total: int) -> None:
            self._on_step_start(step)
            if _prev_start:
                _prev_start(step, idx, total)

        def _on_complete(step: "WorkflowStep", idx: int, total: int, result: Any) -> None:
            self._on_step_complete(step)
            if _prev_complete:
                _prev_complete(step, idx, total, result)

        def _on_state(old: Any, new: Any) -> None:
            self._on_state_change(new)
            if _prev_state:
                _prev_state(old, new)

        executor.on_step_start = _on_start
        executor.on_step_complete = _on_complete
        executor.on_state_change = _on_state

        self._lbl_status.setText("Sweep queued — waiting for executor…")
        self.setEnabled(True)

    def detach_executor(self) -> None:
        """Disconnect from the current executor and reset the display."""
        self._executor = None
        self.reset()

    def reset(self) -> None:
        """Return the widget to its idle state."""
        self._lbl_t_sp.setText("T setpoint: -- °C")
        self._lbl_t_pv.setText("T current:  -- °C")
        self._lbl_status.setText("Idle")
        self._progress.setValue(0)
        self._progress.setRange(0, 100)
        self._progress.setFormat("%v / %m steps")
        self._steps_done = 0
        self._n_steps = 0

    def update_temperature(self, t_sp: float | None, t_pv: float | None) -> None:
        """Push a live temperature reading into the widget.

        This may be called periodically from the Monitor tab's polling loop.

        Parameters
        ----------
        t_sp : float or None
            Current temperature setpoint (°C).
        t_pv : float or None
            Current measured temperature (°C).
        """
        sp_txt = f"{t_sp:.1f} °C" if t_sp is not None else "-- °C"
        pv_txt = f"{t_pv:.1f} °C" if t_pv is not None else "-- °C"
        self._lbl_t_sp.setText(f"T setpoint: {sp_txt}")
        self._lbl_t_pv.setText(f"T current:  {pv_txt}")

    # ── Executor callback handlers ───────────────────────────────────────

    @Slot()
    def _on_step_start(self, step: "WorkflowStep") -> None:
        """Update status label on step start."""
        name = step.name
        tags = step.tags or {}
        temp = tags.get("temperature", "")
        ch = tags.get("channel", "")

        if name.startswith("set_temp_"):
            sp = temp or step.params.get("val", "")
            self._lbl_status.setText(f"Setting T → {sp} °C")
        elif name.startswith("wait_temp_"):
            sp = temp or step.params.get("target", "")
            self._lbl_status.setText(f"Waiting for T = {sp} °C…")
        elif name.startswith("eis_ch"):
            self._lbl_status.setText(
                f"Measuring EIS — Ch {ch}, T = {temp} °C"
            )
        elif name == "restore_ambient":
            self._lbl_status.setText("Restoring ambient temperature…")
        else:
            self._lbl_status.setText(f"Step: {name}")

    @Slot()
    def _on_step_complete(self, step: "WorkflowStep") -> None:
        """Advance the progress bar on step completion."""
        self._steps_done += 1
        self._progress.setValue(self._steps_done)

    @Slot()
    def _on_state_change(self, new_state: Any) -> None:
        """Reflect executor lifecycle transitions in the status label."""
        try:
            from softae.workflows.workflow_executor import ExecutorState

            if new_state is ExecutorState.COMPLETED:
                self._lbl_status.setText("Sweep complete ✓")
                self._progress.setValue(self._n_steps)
            elif new_state is ExecutorState.ABORTED:
                self._lbl_status.setText("Sweep aborted")
            elif new_state is ExecutorState.ERROR:
                self._lbl_status.setText("Sweep error — see log")
        except ImportError:
            pass
