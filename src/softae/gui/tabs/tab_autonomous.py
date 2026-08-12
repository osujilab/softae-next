"""Tab 7: Autonomous Experiment Control.

Objective function, parameter space, optimization algorithm selector,
iteration budget, constraint editor, suggested next experiment,
approval toggle, convergence plot, and live results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class AutonomousTab(QWidget):
    """Closed-loop autonomous experiment control panel."""

    # Public signal: sidebar listens to this.
    workflow_status_changed = Signal(str)  # human-readable status text

    def __init__(self, manager: InstrumentManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        top = QHBoxLayout()

        # --- Objective ---
        obj_grp = QGroupBox("Objective Function")
        obj_layout = QVBoxLayout(obj_grp)
        self._combo_obj = QComboBox()
        self._combo_obj.addItems([
            "Maximize σ (conductivity)",
            "Minimize σ variance",
            "Custom expression...",
        ])
        obj_layout.addWidget(self._combo_obj)
        top.addWidget(obj_grp)

        # --- Algorithm ---
        algo_grp = QGroupBox("Optimization Algorithm")
        algo_layout = QVBoxLayout(algo_grp)
        self._combo_algo = QComboBox()
        self._combo_algo.addItems([
            "Bayesian (GP-UCB)",
            "Bayesian (EI)",
            "Grid Search",
            "Random",
        ])
        algo_layout.addWidget(self._combo_algo)
        top.addWidget(algo_grp)

        # --- Budget ---
        budget_grp = QGroupBox("Iteration Budget")
        budget_layout = QVBoxLayout(budget_grp)
        self._spin_budget = QSpinBox()
        self._spin_budget.setRange(1, 1000)
        self._spin_budget.setValue(50)
        budget_layout.addWidget(self._spin_budget)
        top.addWidget(budget_grp)

        layout.addLayout(top)

        # --- Parameter space ---
        param_grp = QGroupBox("Parameter Space")
        param_layout = QVBoxLayout(param_grp)
        param_layout.addWidget(QLabel("Table: variable name, type (continuous/categorical), bounds.\n"
                                       "Define the search space for the optimizer."))
        layout.addWidget(param_grp)

        # --- Control row ---
        ctrl_row = QHBoxLayout()
        self._chk_auto = QCheckBox("Fully Autonomous (no approval gate)")
        ctrl_row.addWidget(self._chk_auto)

        # Placeholder action buttons: kept for layout scaffolding but DISABLED.
        # An enabled "Approve & Execute" that only emits a status string is an
        # operator-trust hazard — it looks like it acts on the rig but does
        # nothing. Real autonomous execution lives in the Live BO Campaign tab.
        _placeholder_tip = (
            "Placeholder — autonomous execution currently lives in the "
            "Live BO Campaign tab"
        )

        self._btn_suggest = QPushButton("Suggest Next")
        self._btn_suggest.setEnabled(False)
        self._btn_suggest.setToolTip(_placeholder_tip)
        ctrl_row.addWidget(self._btn_suggest)

        self._btn_approve = QPushButton("Approve && Execute")
        self._btn_approve.setStyleSheet("background-color: #4CAF50; color: white;")
        self._btn_approve.setEnabled(False)
        self._btn_approve.setToolTip(_placeholder_tip)
        self._btn_approve.clicked.connect(lambda: self.workflow_status_changed.emit("Running"))
        ctrl_row.addWidget(self._btn_approve)

        self._btn_stop_loop = QPushButton("Stop Loop")
        self._btn_stop_loop.setStyleSheet("background-color: #f44336; color: white;")
        self._btn_stop_loop.setEnabled(False)
        self._btn_stop_loop.setToolTip(_placeholder_tip)
        self._btn_stop_loop.clicked.connect(lambda: self.workflow_status_changed.emit("Idle"))
        ctrl_row.addWidget(self._btn_stop_loop)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- Campaign annotation ---
        notes_grp = QGroupBox("Campaign Notes (stored with run)")
        notes_lay = QVBoxLayout(notes_grp)
        self._te_annotation = QTextEdit()
        self._te_annotation.setPlaceholderText(
            "Brief description of this autonomous campaign "
            "(objective, search space, initial conditions, etc.)\u2026"
        )
        self._te_annotation.setFixedHeight(56)
        notes_lay.addWidget(self._te_annotation)
        layout.addWidget(notes_grp)

        # --- Convergence / history placeholder ---
        hist_grp = QGroupBox("Convergence & History")
        hist_layout = QVBoxLayout(hist_grp)
        hist_layout.addWidget(QLabel("Scatter plot of objective vs. iteration will render here.\n"
                                      "Pareto front for multi-objective campaigns."))
        layout.addWidget(hist_grp)

        # --- Live results ---
        res_grp = QGroupBox("Campaign Results")
        res_layout = QVBoxLayout(res_grp)
        res_layout.addWidget(QLabel("Results table filtered to current autonomous campaign."))
        layout.addWidget(res_grp)
