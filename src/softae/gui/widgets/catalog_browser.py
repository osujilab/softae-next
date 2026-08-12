"""Read-only live view of the current chemical + solution catalogs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.config import loader
from softae.core.formulation import ChemicalCatalog, Solution, SolutionCatalog


class CatalogBrowser(QWidget):
    """Read-only view of the current chemicals and solutions.

    Refreshed from ``data_root()`` (or from injected catalogs) via :meth:`reload`.
    The 'Edit Catalogs…' button emits :attr:`edit_requested`; the embedder opens
    the editor.  Construction is zero-IO when catalogs are injected.
    """

    edit_requested = Signal()  # 'Edit Catalogs…' clicked (MainWindow wires it)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        chem_catalog: ChemicalCatalog | None = None,
        sol_catalog: SolutionCatalog | None = None,
    ) -> None:
        super().__init__(parent)
        self._chem_catalog = chem_catalog
        self._sol_catalog = sol_catalog
        self._build_ui()
        # If no catalogs were injected, load from data_root now.
        if self._chem_catalog is None or self._sol_catalog is None:
            self.reload()  # re-read from data_root()
        else:
            self._repopulate()

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self._btn_edit = QPushButton("Edit Catalogs…")
        self._btn_edit.setStatusTip("Open the chemical/solution catalog editor")
        self._btn_edit.clicked.connect(self.edit_requested)
        header.addWidget(self._btn_edit)
        header.addStretch()
        self._lbl_status = QLabel("")
        header.addWidget(self._lbl_status)
        main_layout.addLayout(header)

        # === Chemicals ===
        chem_grp = QGroupBox("Chemicals")
        chem_lay = QVBoxLayout(chem_grp)
        self._chem_table = QTableWidget(0, 6)
        self._chem_table.setHorizontalHeaderLabels(
            ["Name", "Formula", "Density (g/mL)", "MW (g/mol)", "Viscosity", "Particulate"]
        )
        self._chem_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        chem_lay.addWidget(self._chem_table)
        main_layout.addWidget(chem_grp)

        # === Solutions ===
        sol_grp = QGroupBox("Solutions")
        sol_lay = QVBoxLayout(sol_grp)
        self._sol_table = QTableWidget(0, 3)
        self._sol_table.setHorizontalHeaderLabels(["Name", "Dep fraction", "Components"])
        self._sol_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        sol_lay.addWidget(self._sol_table)
        main_layout.addWidget(sol_grp)

    # -- Public API ------------------------------------------------------------

    def reload(
        self,
        chem_catalog: ChemicalCatalog | None = None,
        sol_catalog: SolutionCatalog | None = None,
    ) -> None:
        """Repopulate from the given catalogs, or re-read ``data_root()`` if omitted.

        Never raises: a failed load degrades to empty tables + a status note.
        """
        if chem_catalog is not None and sol_catalog is not None:
            self._chem_catalog, self._sol_catalog = chem_catalog, sol_catalog
        else:
            self._chem_catalog, self._sol_catalog = self._load_from_root()
        self._repopulate()

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _load_from_root() -> tuple[ChemicalCatalog, SolutionCatalog]:
        """Load catalogs from ``loader.data_root()``; degrade to empty on failure."""
        try:
            root = loader.data_root()
            return (
                ChemicalCatalog.load_csv(root / "chemicals.csv"),
                SolutionCatalog.load_csv(root / "solutions.csv"),
            )
        except Exception:
            return ChemicalCatalog(), SolutionCatalog()

    @staticmethod
    def _component_summary(solution: Solution) -> str:
        """Joined per-component summary, e.g. ``Fumed silica(1.0 g, dep), …``."""
        parts = [
            f"{c.chemical_name}({c.quantity} {c.unit}, {c.role})"
            for c in solution.components
        ]
        return ", ".join(parts)

    def _repopulate(self) -> None:
        """Fill both tables from the current catalogs; idempotent, never raises."""
        chem = self._chem_catalog if self._chem_catalog is not None else ChemicalCatalog()
        sol = self._sol_catalog if self._sol_catalog is not None else SolutionCatalog()

        self._chem_table.setRowCount(0)
        for name in chem.list_names():
            c = chem.get(name)
            row = self._chem_table.rowCount()
            self._chem_table.insertRow(row)
            visc = "" if c.viscosity_mPa_s is None else f"{c.viscosity_mPa_s}"
            cells = [
                name,
                c.formula,
                f"{c.density_g_per_mL}",
                f"{c.molar_mass_g_per_mol}",
                visc,
                "Yes" if c.is_particulate else "",
            ]
            for col, text in enumerate(cells):
                self._chem_table.setItem(row, col, QTableWidgetItem(text))

        self._sol_table.setRowCount(0)
        for name in sol.list_names():
            s = sol.get(name)
            row = self._sol_table.rowCount()
            self._sol_table.insertRow(row)
            try:
                dep = s.dep_fraction(chem)
            except Exception:
                dep = 0.0
            cells = [name, f"{dep:.2f}", self._component_summary(s)]
            for col, text in enumerate(cells):
                self._sol_table.setItem(row, col, QTableWidgetItem(text))

        self._lbl_status.setText(f"{len(chem)} chemicals, {len(sol)} solutions")
