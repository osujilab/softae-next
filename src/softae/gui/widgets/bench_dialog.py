"""Declare the bench's physical limits: waste level and spare plates (P5.4).

Neither is measurable by the software — a waste bottle has no level sensor and
nobody can count plates in a drawer over a serial link — so both are operator
assertions, like syringe stock. What the platform contributes is remembering
them across sessions and refusing to walk into a wall it was told about.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class BenchDialog(QDialog):
    """Waste-container level and spare-board count."""

    def __init__(self, waste, boards, parent: QWidget | None = None,
                 *, data_store=None):
        super().__init__(parent)
        self._waste = waste
        self._boards = boards
        self._data_store = data_store
        self.setWindowTitle("Bench Consumables")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Neither of these can be sensed, so the software relies on what "
                "you declare here. Both cap how long a campaign can run "
                "unattended."
            )
        )

        form = QFormLayout()

        # ── Waste ────────────────────────────────────────────────────────
        waste_row = QHBoxLayout()
        self._lbl_waste = QLabel()
        waste_row.addWidget(self._lbl_waste, 1)
        btn_empty = QPushButton("Mark emptied")
        btn_empty.setToolTip("Reset the accumulated waste volume to zero")
        btn_empty.clicked.connect(self._on_empty)
        waste_row.addWidget(btn_empty)
        form.addRow("Waste container", waste_row)

        # ── Spare boards ─────────────────────────────────────────────────
        board_row = QHBoxLayout()
        self._spin_boards = QSpinBox()
        self._spin_boards.setRange(0, 999)
        board_row.addWidget(self._spin_boards)
        btn_boards = QPushButton("Declare")
        btn_boards.clicked.connect(self._on_declare_boards)
        board_row.addWidget(btn_boards)
        self._lbl_boards = QLabel()
        board_row.addWidget(self._lbl_boards, 1)
        form.addRow("Spare electrode plates", board_row)

        layout.addLayout(form)

        # ── Anti-clog purge ──────────────────────────────────────────────
        if data_store is not None:
            from softae.core.purge import load_purge_settings

            self._purge = load_purge_settings(data_store)
            layout.addWidget(QLabel(
                "\nAnti-clog purge — particulate stock clogs its check valve "
                "when it sits still. During a run the purge is queued to a safe "
                "boundary, so the interval is a floor, not a deadline."
            ))

            purge_form = QFormLayout()
            self._spin_interval = QDoubleSpinBox()
            self._spin_interval.setRange(1.0, 1440.0)
            self._spin_interval.setDecimals(0)
            self._spin_interval.setSuffix(" min")
            self._spin_interval.setValue(self._purge.interval_s / 60.0)
            purge_form.addRow("Every", self._spin_interval)

            self._spin_particulate = QDoubleSpinBox()
            self._spin_particulate.setRange(0.0, 1000.0)
            self._spin_particulate.setDecimals(0)
            self._spin_particulate.setSuffix(" µL")
            self._spin_particulate.setValue(self._purge.particulate_uL)
            purge_form.addRow("Particulate line", self._spin_particulate)

            self._spin_other = QDoubleSpinBox()
            self._spin_other.setRange(0.0, 1000.0)
            self._spin_other.setDecimals(0)
            self._spin_other.setSuffix(" µL")
            self._spin_other.setValue(self._purge.other_uL)
            purge_form.addRow("Other lines", self._spin_other)
            layout.addLayout(purge_form)

            rate_row = QHBoxLayout()
            self._lbl_rate = QLabel()
            rate_row.addWidget(self._lbl_rate, 1)
            btn_save = QPushButton("Save purge settings")
            btn_save.clicked.connect(self._on_save_purge)
            rate_row.addWidget(btn_save)
            layout.addLayout(rate_row)

            for spin in (self._spin_interval, self._spin_particulate,
                         self._spin_other):
                spin.valueChanged.connect(self._refresh_purge_rate)
            self._refresh_purge_rate()
        else:
            self._purge = None

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()

    # ── Actions ──────────────────────────────────────────────────────────

    def _on_empty(self) -> None:
        self._waste.empty()
        self._refresh()

    def _on_declare_boards(self) -> None:
        self._boards.declare(self._spin_boards.value())
        self._refresh()

    def _edited_purge(self):
        """The settings as currently shown, without saving them."""
        from dataclasses import replace

        return replace(
            self._purge,
            interval_s=self._spin_interval.value() * 60.0,
            particulate_uL=self._spin_particulate.value(),
            other_uL=self._spin_other.value(),
        )

    def _refresh_purge_rate(self) -> None:
        """Show the consumption these settings imply, as they are edited.

        The daily rate is the number that actually decides how long a run can go
        unattended, and it is not obvious from an interval and two volumes.
        """
        rate = self._edited_purge().total_uL_per_day() / 1000.0
        self._lbl_rate.setText(f"≈ {rate:.1f} mL/day across all lines")
        self._lbl_rate.setStyleSheet(
            "color: #c47f1a; font-weight: bold;" if rate > 20.0 else "")

    def _on_save_purge(self) -> None:
        from softae.core.purge import save_purge_settings

        self._purge = self._edited_purge()
        save_purge_settings(self._data_store, self._purge)
        self._refresh_purge_rate()

    # ── Display ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        level = self._waste.level_uL()
        capacity = self._waste.capacity_uL
        if capacity is None:
            self._lbl_waste.setText(
                f"{level:.0f} µL accumulated — no capacity configured, so it is "
                f"tracked but not enforced."
            )
            self._lbl_waste.setStyleSheet("color: gray;")
        else:
            frac = level / capacity
            self._lbl_waste.setText(
                f"{level:.0f} of {capacity:.0f} µL ({frac * 100:.0f}% full)")
            if frac >= 1.0:
                self._lbl_waste.setStyleSheet("color: #c0392b; font-weight: bold;")
            elif frac >= self._waste.warn_fraction:
                self._lbl_waste.setStyleSheet("color: #c47f1a; font-weight: bold;")
            else:
                self._lbl_waste.setStyleSheet("")

        remaining = self._boards.remaining()
        if remaining is None:
            # Undeclared is unknown, not zero — saying "0 left" would be a claim
            # the software has no basis for.
            self._lbl_boards.setText("not declared")
            self._lbl_boards.setStyleSheet("color: gray;")
        else:
            self._spin_boards.setValue(remaining)
            self._lbl_boards.setText(f"{remaining} on hand")
            if remaining == 0:
                self._lbl_boards.setStyleSheet("color: #c0392b; font-weight: bold;")
            elif remaining <= self._boards.warn_at:
                self._lbl_boards.setStyleSheet("color: #c47f1a; font-weight: bold;")
            else:
                self._lbl_boards.setStyleSheet("")
