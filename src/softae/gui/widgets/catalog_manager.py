"""Slim chemical + solution catalog editor (CRUD only, no calculator/pump UI)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from softae.core.formulation import ChemicalCatalog, SolutionCatalog
from softae.gui.widgets.catalog_editor_base import CatalogEditorMixin


class CatalogManager(CatalogEditorMixin, QDialog):
    """Focused chemical + solution CRUD editor.

    Reuses :class:`CatalogEditorMixin` for the 7-col chem table, 5-col component
    table (chemical + calc-mode combos), CRUD, save-time validation +
    highlighting, and the data-loss-safe round-trip.  Omits ``FormulationPanel``'s
    elution calculator, pump assignment, and volume-emit UI.  Because it has no
    pump rows it does NOT override ``_on_solution_set_changed`` (base no-op).
    """

    catalogs_changed = Signal()  # emitted after any successful save (mixin emits it)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        chem_catalog: ChemicalCatalog | None = None,
        sol_catalog: SolutionCatalog | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Catalog Manager")
        self.setMinimumWidth(720)
        self.setMinimumHeight(520)

        # Injected (caller owns them) or auto-loaded from data_root; either path
        # degrades to empty without a dialog.
        self._resolve_catalogs(chem_catalog, sol_catalog)

        layout = QVBoxLayout(self)
        layout.addWidget(self._build_chem_group())      # mixin
        layout.addWidget(self._build_solution_group())  # mixin

        btn_row = QHBoxLayout()
        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save_canonical)
        self._btn_save_as = QPushButton("Save As…")
        self._btn_save_as.clicked.connect(self._on_save_as)
        self._btn_load = QPushButton("Load From…")
        self._btn_load.clicked.connect(self._on_load)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        for b in (self._btn_save, self._btn_save_as, self._btn_load, self._btn_close):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._refresh_tables_from_catalogs()  # mixin (hook is a no-op here)
        self._wire_table_signals()            # mixin
