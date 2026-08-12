"""Operator-facing stock declaration for the reservoir interlock.

The ledger in :mod:`softae.core.reservoir` refuses a dispense that would drive a
plunger into its mechanical stop, but it can only do that for stocks it knows
about — an undeclared pump reads as *unknown*, never *empty*, and passes
through untouched. This dialog is how a pump comes under management: the
operator measures what they loaded and declares it.

**Declared volumes are an operator assertion, not a reading.** Nothing in the
software can measure the syringe, and in particular the ``res_vol`` argument on
``single_pump`` must never be used to fill this in — it is the syringe volume
declared to the pump firmware, conventionally padded past the command so the
pump does not trip its own limit, and it says nothing about contents.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from softae.drivers.contracts import N_PUMPS

#: Shown for a pump whose stock has not been declared.
_UNDECLARED = "— not declared —"


class ReservoirDialog(QDialog):
    """Show remaining stock per pump and let the operator declare a refill."""

    def __init__(self, ledger, parent: QWidget | None = None, *,
                 data_store=None, sol_catalog=None):
        super().__init__(parent)
        self._ledger = ledger
        self._data_store = data_store
        self._sol_catalog = sol_catalog
        self.setWindowTitle("Syringe Stock")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Declare the volume actually loaded into each syringe.\n"
                f"Dispensing is refused below {ledger.hard_stop_uL:.0f} µL — running "
                "dry drives the plunger into its mechanical stop."
            )
        )

        form = QFormLayout()
        self._spins: dict[int, QDoubleSpinBox] = {}
        self._labels: dict[int, QLabel] = {}
        self._stock_boxes: dict[int, QComboBox] = {}

        # Which solution sits on each line. Declaring it is what lets the
        # anti-clog purge derive its particulate lines from chemistry rather
        # than a hand-maintained config list — which was found wrong.
        #
        # The pump index leads every row because the IDs are physically
        # meaningful bench positions: the question an operator answers here
        # is "what is loaded in position 1", not "where did my silica go".
        self._loadout = _load_loadout(data_store)
        names = _solution_names(sol_catalog)
        for pump_id in range(N_PUMPS):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1_000_000.0)
            spin.setDecimals(0)
            spin.setSingleStep(500.0)
            spin.setSuffix(" µL")
            self._spins[pump_id] = spin

            remaining = QLabel()
            self._labels[pump_id] = remaining

            stock = QComboBox()
            stock.addItem(_UNDECLARED, None)
            for name in names:
                stock.addItem(name, name)
            current = self._loadout.solution_for(pump_id)
            if current:
                idx = stock.findData(current)
                if idx < 0:            # loadout names a solution the
                    stock.addItem(current, current)   # catalog has dropped
                    idx = stock.count() - 1
                stock.setCurrentIndex(idx)
            stock.currentIndexChanged.connect(
                lambda _idx, pid=pump_id: self._assign_stock(pid)
            )
            self._stock_boxes[pump_id] = stock

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(stock, 1)
            row_layout.addWidget(spin)
            row_layout.addWidget(remaining, 1)
            declare = QPushButton("Declare")
            declare.clicked.connect(
                lambda checked=False, pid=pump_id: self._declare(pid)
            )
            row_layout.addWidget(declare)
            form.addRow(f"Pump {pump_id}", row)
        layout.addLayout(form)

        self._particulate_note = QLabel()
        self._particulate_note.setWordWrap(True)
        self._particulate_note.setStyleSheet("color: gray;")
        layout.addWidget(self._particulate_note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()
        self._refresh_particulate_note()

    def _declare(self, pump_id: int) -> None:
        self._ledger.refill(pump_id, self._spins[pump_id].value())
        self._refresh()

    def _assign_stock(self, pump_id: int) -> None:
        """Bind (or clear) the solution loaded on this pump; persist it."""
        self._loadout.assign(pump_id, self._stock_boxes[pump_id].currentData())
        _save_loadout(self._data_store, self._loadout)
        self._refresh_particulate_note()

    def _refresh(self) -> None:
        for pump_id, label in self._labels.items():
            remaining = self._ledger.remaining_uL(pump_id)
            if remaining is None:
                label.setText("not tracked")
                label.setStyleSheet("color: gray;")
                continue
            label.setText(f"{remaining:.0f} µL left")
            if remaining <= self._ledger.hard_stop_uL:
                label.setStyleSheet("color: #c0392b; font-weight: bold;")
            elif remaining <= self._ledger.soft_warn_uL:
                label.setStyleSheet("color: #c47f1a; font-weight: bold;")
            else:
                label.setStyleSheet("")
            self._spins[pump_id].setValue(remaining)

    def _refresh_particulate_note(self) -> None:
        """Show which lines the purge treats as particulate, and why.

        Worth surfacing because the rule is not obvious: an *undeclared* pump
        counts as particulate. Partial declaration is the dangerous state — the
        operator has engaged with the mechanism, so silence about one line reads
        as an oversight rather than an assertion that the line is clean.
        """
        try:
            from softae.core.purge import load_purge_settings
            from softae.core.stock_assignment import derive_particulate_pumps

            settings = load_purge_settings(self._data_store)
            chem, sol = _catalogs(self._sol_catalog)
            derived = derive_particulate_pumps(
                self._loadout, chem_catalog=chem, sol_catalog=sol,
                pumps=tuple(settings.pumps),
                fallback=tuple(settings.particulate_pumps),
            )
        except Exception:
            self._particulate_note.setText("")
            return

        if self._loadout.is_empty():
            self._particulate_note.setText(
                f"No stocks declared — the anti-clog purge falls back to the "
                f"configured lines: {_pump_list(settings.particulate_pumps)}."
            )
            return

        undeclared = [p for p in settings.pumps
                      if not self._loadout.solution_for(p)]
        note = (
            f"Anti-clog purge treats {_pump_list(derived)} as particulate "
            f"({settings.particulate_uL:.0f} µL vs {settings.other_uL:.0f} µL "
            f"on the others)."
        )
        if undeclared:
            note += (
                f" {_pump_list(undeclared).capitalize()} undeclared, so counted "
                f"as particulate until declared."
            )
        self._particulate_note.setText(note)


def _pump_list(pumps) -> str:
    """Pump indices, always led by the word 'pump' — they are bench positions."""
    ids = sorted(int(p) for p in pumps)
    if not ids:
        return "no pumps"
    if len(ids) == 1:
        return f"pump {ids[0]}"
    return "pumps " + ", ".join(str(i) for i in ids)


def _solution_names(sol_catalog) -> list:
    try:
        return list(sol_catalog.list_names()) if sol_catalog is not None else []
    except Exception:
        return []


def _catalogs(sol_catalog):
    from softae.core.stock_assignment import catalogs_from_data_root

    chem, sol = catalogs_from_data_root()
    return chem, (sol_catalog if sol_catalog is not None else sol)


def _load_loadout(data_store):
    from softae.core.stock_assignment import load_loadout

    return load_loadout(data_store)


def _save_loadout(data_store, loadout) -> None:
    from softae.core.stock_assignment import save_loadout

    save_loadout(data_store, loadout)
