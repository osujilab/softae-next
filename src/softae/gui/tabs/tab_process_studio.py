"""Process Studio tab — the method/recipe library + lifecycle detail (slice 1).

Read-only surface for the method-maturity pipeline: a table of every catalogued
**method** and registered **recipe** with its maturity and effective maturity,
plus a detail panel showing provenance, evidence, version chain, and (for
recipes) the method dependencies with the one that caps effective maturity.

Later slices add lifecycle actions (test/promote/sign-off), the recipe builder,
and recipe-driven HT consumption — see docs/PROCESS_STUDIO_BUILD.md.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
from datetime import date
from html import escape
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from softae.config import loader
from softae.core import lifecycle as lc
from softae.core.deposition_recipe import (
    deposition_recipe_names,
    get_deposition_recipe,
)
from softae.core.lifecycle import GateError, Maturity
from softae.core.recipe_registry import Recipe, RecipeRegistry
from softae.core.task_catalog import TaskCatalog
from softae.workflows import workflow_parser
from softae.workflows.workflow_model import Workflow, WorkflowStep

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class _PytestWorker(QThread):
    """Run pytest node IDs off the UI thread; emit the return code when done."""

    finished_rc = Signal(int)

    def __init__(self, tests: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tests = tests

    def run(self) -> None:  # noqa: D401 (QThread override)
        try:
            rc = subprocess.run(
                [sys.executable, "-m", "pytest", *self._tests, "-q"]
            ).returncode
        except Exception:
            rc = 1
        self.finished_rc.emit(rc)

# Maturity → display colour (at-a-glance readout).
_MATURITY_COLOR = {
    "draft": "#9ca3af",      # gray
    "prototype": "#3b82f6",  # blue
    "tested": "#d97706",     # amber
    "validated": "#16a34a",  # green
}
_UserRole = Qt.ItemDataRole.UserRole


class ProcessStudioTab(QWidget):
    """Read-only library of methods + recipes with a lifecycle detail panel.

    Parameters
    ----------
    catalog, recipes :
        Optional injected registries (for tests / embedding).  When omitted, the
        tab loads from ``loader.tasks_toml_path()`` / ``loader.recipes_toml_path()``
        and re-reads them on :meth:`reload`.
    """

    def __init__(
        self,
        catalog: TaskCatalog | None = None,
        recipes: RecipeRegistry | None = None,
        data_store: Any = None,
        manager: "InstrumentManager | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._injected = catalog is not None or recipes is not None
        self._catalog = catalog or TaskCatalog()
        self._recipes = recipes or RecipeRegistry()
        self._data_store = data_store
        self._manager = manager
        self._sandbox = None  # embedded builder (only when a manager is present)
        self._test_worker: _PytestWorker | None = None
        self._build_ui()
        self.reload()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        self._views = QTabWidget()
        self._views.addTab(self._build_library_page(), "Library")
        if self._manager is not None:
            self._views.addTab(self._build_builder_page(), "Builder")
        outer.addWidget(self._views)

    def _build_library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("<b>Process Studio</b> — methods &amp; recipes"))
        toolbar.addStretch()
        toolbar.addWidget(QLabel("Show:"))
        self._filter = QComboBox()
        self._filter.addItems(["All", "Methods", "Recipes", "Deposition"])
        self._filter.currentTextChanged.connect(lambda _=None: self._populate_table())
        toolbar.addWidget(self._filter)
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self.reload)
        toolbar.addWidget(self._btn_refresh)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Kind", "Maturity", "Effective"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self._table)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self._detail = QTextBrowser()
        self._detail.setOpenExternalLinks(False)
        right_lay.addWidget(self._detail)

        # ── Lifecycle actions (slice 2) — operate on the selected entry ──
        actions = QHBoxLayout()
        self._btn_promote = QPushButton("Promote…")
        self._btn_promote.clicked.connect(self._on_promote)
        self._btn_test = QPushButton("Run tests")
        self._btn_test.clicked.connect(self._on_run_tests)
        self._btn_signoff = QPushButton("Sign-off…")
        self._btn_signoff.clicked.connect(self._on_signoff)
        self._btn_supersede = QPushButton("Supersede…")
        self._btn_supersede.clicked.connect(self._on_supersede)
        self._btn_edit = QPushButton("Edit…")
        self._btn_edit.setToolTip(
            "Edit a recipe: open a workflow-backed recipe in the Builder, or "
            "edit a builder-backed recipe's parameters.")
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_remove = QPushButton("Remove…")
        self._btn_remove.setToolTip(
            "Delete the selected method or recipe from the library. Removing a "
            "method that recipes depend on will break them.")
        self._btn_remove.clicked.connect(self._on_remove)
        self._btn_set_default = QPushButton("Set default…")
        self._btn_set_default.setToolTip(
            "Make the selected deposition recipe the HT tab's default "
            "([dropcast].default_recipe).")
        self._btn_set_default.clicked.connect(self._on_set_default)
        for b in (self._btn_promote, self._btn_test, self._btn_signoff,
                  self._btn_supersede, self._btn_edit, self._btn_remove,
                  self._btn_set_default):
            b.setEnabled(False)
            actions.addWidget(b)
        actions.addStretch()
        right_lay.addLayout(actions)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray;")
        right_lay.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ── Step-sequence preview (lower half) — the fully expanded steps the
        # selected entry runs, at the same granularity the Builder shows. For a
        # (code-generated) deposition recipe this is the only place to see it.
        steps_grp = QGroupBox("Step sequence")
        steps_lay = QVBoxLayout(steps_grp)
        steps_lay.setContentsMargins(4, 4, 4, 4)
        self._steps_preview = QPlainTextEdit()
        self._steps_preview.setReadOnly(True)
        self._steps_preview.setStyleSheet("font-family: monospace; font-size: 11px;")
        self._steps_preview.setPlaceholderText(
            "Select a method, recipe, or deposition recipe to see its expanded steps.")
        steps_lay.addWidget(self._steps_preview)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(splitter)
        vsplit.addWidget(steps_grp)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        layout.addWidget(vsplit)

        self._count = QLabel("")
        layout.addWidget(self._count)
        return page

    def _build_builder_page(self) -> QWidget:
        """Embed the recipe builder + a 'Register as recipe' bridge (slice 3)."""
        from softae.gui.tabs.tab_sandbox import SandboxTab

        page = QWidget()
        lay = QVBoxLayout(page)
        bar = QHBoxLayout()
        bar.addWidget(QLabel(
            "Build a recipe below, then register it into the pipeline as a draft:"))
        bar.addStretch()
        self._btn_register = QPushButton("Register as recipe (draft)…")
        self._btn_register.clicked.connect(self._on_register_recipe)
        bar.addWidget(self._btn_register)
        lay.addLayout(bar)

        self._sandbox = SandboxTab(self._manager)
        lay.addWidget(self._sandbox)
        return page

    # ── Data ───────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload the catalog + recipe registry (unless injected) and repopulate."""
        if not self._injected:
            try:
                self._catalog = TaskCatalog.load_toml(loader.tasks_toml_path())
                self._recipes = RecipeRegistry.load_toml(loader.recipes_toml_path())
            except Exception:
                self._catalog = self._catalog or TaskCatalog()
                self._recipes = self._recipes or RecipeRegistry()
        self._populate_table()

    def _rows(self) -> list[tuple[str, str, str, str]]:
        """(kind, name, maturity, effective) for every entry, filtered."""
        show = self._filter.currentText()
        out: list[tuple[str, str, str, str]] = []
        if show in ("All", "Methods"):
            for name in self._catalog.list_names():
                m = lc.method_maturity(name, self._catalog).label
                out.append(("method", name, m, m))
        if show in ("All", "Recipes"):
            for name in self._recipes.list_names():
                own = self._recipes.get(name).maturity
                eff = lc.effective_maturity(name, self._catalog, self._recipes).label
                out.append(("recipe", name, own, eff))
        if show in ("All", "Deposition"):
            for name in deposition_recipe_names():
                eff = self._deposition_recipe_maturity(name).label
                out.append(("deposition", name, eff, eff))
        return out

    def _populate_table(self) -> None:
        rows = self._rows()
        self._table.setRowCount(len(rows))
        for r, (kind, name, maturity, effective) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setData(_UserRole, (kind, name))
            self._table.setItem(r, 0, name_item)
            self._table.setItem(r, 1, QTableWidgetItem(kind))
            self._table.setItem(r, 2, self._maturity_item(maturity))
            self._table.setItem(r, 3, self._maturity_item(effective))
        n_methods = sum(1 for k, *_ in rows if k == "method")
        n_dep = sum(1 for k, *_ in rows if k == "deposition")
        n_recipes = len(rows) - n_methods - n_dep
        self._count.setText(
            f"{n_methods} methods · {n_recipes} recipes · {n_dep} deposition")
        self._detail.clear()

    @staticmethod
    def _maturity_item(label: str) -> QTableWidgetItem:
        item = QTableWidgetItem(label)
        color = _MATURITY_COLOR.get(label)
        if color:
            item.setForeground(QColor(color))
        return item

    # ── Detail panel ─────────────────────────────────────────────────────────

    def _on_select(self) -> None:
        sel = self._selected()
        if sel is None:
            for b in (self._btn_promote, self._btn_test, self._btn_signoff,
                      self._btn_supersede, self._btn_edit, self._btn_remove,
                      self._btn_set_default):
                b.setEnabled(False)
            self._steps_preview.clear()
            return
        kind, name = sel
        self._update_step_preview(kind, name)
        if kind == "deposition":
            self._detail.setHtml(self._deposition_html(name))
        elif kind == "recipe":
            self._detail.setHtml(self._recipe_html(name))
        else:
            self._detail.setHtml(self._method_html(name))
        running = self._test_worker is not None
        # Deposition recipes are built-in (code-defined): lifecycle actions and
        # removal don't apply — only 'Set default' does.
        is_dep = kind == "deposition"
        self._btn_promote.setEnabled(not running and not is_dep)
        self._btn_test.setEnabled(not running and not is_dep)
        self._btn_signoff.setEnabled(not running and not is_dep)
        self._btn_supersede.setEnabled(kind == "method" and not running)
        self._btn_edit.setEnabled(kind == "recipe" and not running)
        self._btn_remove.setEnabled(not running and not is_dep)
        self._btn_set_default.setEnabled(is_dep and not running)

    # ── Selection / registry helpers ─────────────────────────────────────────

    def _selected(self) -> tuple[str, str] | None:
        items = self._table.selectedItems()
        if not items:
            return None
        data = self._table.item(items[0].row(), 0).data(_UserRole)
        return tuple(data) if data else None

    def _registry(self, kind: str) -> Any:
        return self._recipes if kind == "recipe" else self._catalog

    def _persist(self, kind: str) -> None:
        """Save the mutated registry back to disk (skipped for injected/tests)."""
        if self._injected:
            return
        if kind == "recipe":
            self._recipes.save_toml(loader.recipes_toml_path())
        else:
            self._catalog.save_toml(loader.tasks_toml_path())

    def _select_by_name(self, name: str) -> None:
        for r in range(self._table.rowCount()):
            if self._table.item(r, 0).text() == name:
                self._table.selectRow(r)
                return

    def _after_mutation(self, name: str) -> None:
        self._populate_table()
        self._select_by_name(name)

    def _ensure_store(self) -> Any:
        if self._data_store is None:
            from softae.core.data_store import DataStore

            self._data_store = DataStore(
                project_dir=loader.data_project_dir(),
                db_filename=loader.data_db_filename(),
            )
        return self._data_store

    # ── Lifecycle actions (testable core) ────────────────────────────────────

    def promote_selected(self, stage: "Maturity | str") -> tuple[bool, str]:
        sel = self._selected()
        if sel is None:
            return False, "nothing selected"
        kind, name = sel
        try:
            lc.promote(name, stage, self._registry(kind))
        except GateError as exc:
            return False, str(exc)
        self._persist(kind)
        self._after_mutation(name)
        return True, f"'{name}' promoted to '{Maturity.parse(stage).label}'."

    def sign_off_selected(self, run_id: str, by: str, note: str = "") -> tuple[bool, str]:
        sel = self._selected()
        if sel is None:
            return False, "nothing selected"
        kind, name = sel
        try:
            lc.sign_off(name, run_id=run_id, by=by, catalog=self._registry(kind),
                        notes=note, data_store=self._ensure_store())
        except GateError as exc:
            return False, str(exc)
        self._persist(kind)
        self._after_mutation(name)
        return True, f"'{name}' signed off on {run_id} → validated."

    def supersede_selected(self) -> tuple[bool, str]:
        sel = self._selected()
        if sel is None:
            return False, "nothing selected"
        kind, name = sel
        if kind != "method":
            return False, "only methods can be superseded"
        clone = copy.deepcopy(self._catalog.get(name))
        clone.evidence = {}  # a new version is unproven — start with no evidence
        archived, _ = lc.supersede(name, clone, self._catalog)
        self._persist("method")
        self._after_mutation(name)
        return True, f"'{name}' superseded (previous archived as '{archived}')."

    # ── Deposition recipes (built-in; the HT engine's per-channel phase sequences) ──

    def _deposition_recipe_maturity(self, name: str) -> Maturity:
        """Effective maturity of a deposition recipe: min over its methods.

        A missing method caps the recipe at ``draft`` (it cannot run).
        """
        recipe = get_deposition_recipe(name)
        maturities = [
            lc.method_maturity(m, self._catalog) if m in self._catalog else Maturity.DRAFT
            for m in recipe.method_deps()
        ]
        return min(maturities) if maturities else Maturity.DRAFT

    def _deposition_html(self, name: str) -> str:
        recipe = get_deposition_recipe(name)
        eff = self._deposition_recipe_maturity(name)
        parts = [self._header(name, "deposition recipe", eff.label, eff.label)]
        if recipe.description:
            parts.append(f"<p>{escape(recipe.description)}</p>")
        phases = "".join(
            f"<li><b>{escape(p.key)}</b>: {self._code(p.method)}</li>" for p in recipe.phases
        )
        parts.append(f"<p><b>Phases</b> (per channel)</p><ol>{phases}</ol>")
        rows = []
        for m in recipe.method_deps():
            if m in self._catalog:
                mm = lc.method_maturity(m, self._catalog)
                caps = " &larr; caps effective" if mm == eff else ""
                rows.append(f"<li>{escape(m)}: {self._colored(mm.label)}{caps}</li>")
            else:
                rows.append(f"<li>{escape(m)}: <span style='color:#dc2626'>MISSING</span></li>")
        parts.append(f"<p><b>Methods</b></p><ul>{''.join(rows)}</ul>")
        parts.append(
            "<p><i>Built-in deposition recipe (code-defined). Use 'Set default…' to "
            "make it the HT tab's default.</i></p>")
        return "".join(parts)

    def set_default_selected(self) -> tuple[bool, str]:
        """Persist the selected deposition recipe as the HT default (testable core)."""
        sel = self._selected()
        if sel is None:
            return False, "nothing selected"
        kind, name = sel
        if kind != "deposition":
            return False, "only deposition recipes can be set as default"
        try:
            loader.set_dropcast_default_recipe(name)
        except Exception as exc:
            return False, f"could not write default: {exc}"
        return True, f"'{name}' is now the default deposition recipe."

    def _on_set_default(self) -> None:
        sel = self._selected()
        if sel is None or sel[0] != "deposition":
            return
        ok, msg = self.set_default_selected()
        self._status.setText(msg)
        if not ok:
            QMessageBox.warning(self, "Set default", msg)

    # ── Step-sequence preview (lower panel) ──────────────────────────────────

    def _update_step_preview(self, kind: str, name: str) -> None:
        """Render the selected entry's expanded step sequence into the preview."""
        try:
            if kind == "deposition":
                text = self._deposition_step_preview(name)
            elif kind == "recipe":
                text = self._recipe_step_preview(name)
            else:
                text = self._method_step_preview(name)
        except Exception as exc:  # a preview failure must never break selection
            text = f"(could not build step preview: {exc})"
        self._steps_preview.setPlainText(text)

    def _method_step_preview(self, name: str) -> str:
        wf = Workflow(name=name, setup=[self._catalog.get(name).to_step(name)])
        return self._workflow_step_lines(wf, header=f"Method '{name}' — one step:")

    def _recipe_step_preview(self, name: str) -> str:
        recipe = self._recipes.get(name)
        if recipe.workflow_path:
            wf = workflow_parser.parse_file(recipe.workflow_path)
            return self._workflow_step_lines(wf, header=f"Recipe '{name}':")
        if recipe.builder:
            values = {p["name"]: p.get("default") for p in (recipe.parameters or [])}
            wf, msg = self.build_recipe_preview(name, values)
            if wf is None:
                return f"Recipe '{name}' (builder-backed): {msg}"
            return self._workflow_step_lines(
                wf, header=f"Recipe '{name}' (built with default parameters):")
        return f"Recipe '{name}' has no workflow or builder to expand."

    def _deposition_step_preview(self, name: str) -> str:
        from softae.config.loader import pcb_configs, pico_for_channel
        from softae.core.deposition_recipe import (
            DepositionSettings, build_deposition_workflow, get_deposition_recipe,
        )

        recipe = get_deposition_recipe(name)
        missing = [m for m in recipe.method_deps() if m not in self._catalog]
        if missing:
            return (f"Deposition recipe '{name}': methods {missing} not in the "
                    f"catalog — cannot expand the sequence.")

        pcbs = pcb_configs()
        pcb = next(iter(pcbs.values())) if pcbs else {"grid": [4, 4], "spacing_mm": [15, 15]}
        pump_ids = [0, 1, 2]
        ch = 1
        # A representative equal-split formulation just to shape the sequence.
        formulation = {ch: [10.0] * len(pump_ids)}
        eis = {ch: WorkflowStep(
            name=f"measure_eis_ch{ch}", instrument=pico_for_channel(ch),
            method="sendscript_getdata", params={"chan": ch})}

        wf = build_deposition_workflow(
            recipe, [ch], formulation,
            settings=DepositionSettings.from_config(pcb, pump_ids=pump_ids),
            catalog=self._catalog,
            eis_step_by_channel=eis,
        )
        note = (
            f"Representative sequence — 1 sample channel, {len(pump_ids)} pumps; "
            f"rates/volumes from the [dropcast] config. Each composite "
            f"liquid_handler step drives stage+syringe internally "
            f"(move → descend → dispense → wait → wick).")
        return self._workflow_step_lines(
            wf, header=f"Deposition recipe '{name}' — {recipe.label}:", note=note)

    @staticmethod
    def _workflow_step_lines(wf: Workflow, *, header: str, note: str = "") -> str:
        lines = [header, ""]
        if note:
            lines += [note, ""]
        steps = wf.resolve_steps()
        lines.append(f"{len(steps)} step(s):")
        for i, s in enumerate(steps, 1):
            tagbits = [b for b in (
                s.tags.get("phase"),
                f"ch{s.tags['channel']}" if s.tags.get("channel") else None,
            ) if b]
            suffix = f"   [{', '.join(tagbits)}]" if tagbits else ""
            lines.append(f"  {i}. {s.name} -> {s.instrument}.{s.method}(){suffix}")
            for k, v in s.params.items():
                vs = str(v)
                if len(vs) > 60:
                    vs = vs[:57] + "…"
                lines.append(f"        {k} = {vs}")
        return "\n".join(lines)

    def _recipes_using_method(self, name: str) -> list[str]:
        """Registered recipes that list *name* among their method dependencies."""
        return [
            r for r in self._recipes.list_names()
            if name in (self._recipes.get(r).methods or [])
        ]

    def remove_selected(self) -> tuple[bool, str]:
        """Delete the selected method/recipe from its registry and persist.

        Testable core of the Library 'Remove…' button.  The registry raises
        ``KeyError`` if the entry vanished between selection and removal, which we
        report rather than surface as a crash.
        """
        sel = self._selected()
        if sel is None:
            return False, "nothing selected"
        kind, name = sel
        try:
            self._registry(kind).remove(name)
        except KeyError:
            return False, f"'{name}' not found"
        self._persist(kind)
        self._populate_table()
        for b in (self._btn_promote, self._btn_test, self._btn_signoff,
                  self._btn_supersede, self._btn_edit, self._btn_remove):
            b.setEnabled(False)
        return True, f"removed {kind} '{name}'."

    def _on_remove(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        kind, name = sel
        warn = ""
        if kind == "method":
            users = self._recipes_using_method(name)
            if users:
                warn = (
                    f"\n\nWARNING: {len(users)} recipe(s) depend on this method "
                    f"and will break: {', '.join(users)}.")
        if QMessageBox.question(
            self, "Remove",
            f"Remove {kind} '{name}' from the library?{warn}\n\nThis cannot be undone.",
        ) != QMessageBox.StandardButton.Yes:
            return
        ok, msg = self.remove_selected()
        self._status.setText(msg)
        if not ok:
            QMessageBox.warning(self, "Remove", msg)

    def register_recipe_from_workflow(self, name: str, wf: Any) -> tuple[bool, str]:
        """Save *wf* as a workflow YAML and register it as a draft recipe.

        Method dependencies are auto-detected by mapping the workflow's steps to
        catalogued methods (unique instrument.method match).  Testable core of
        the 'Register as recipe' bridge.
        """
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
        if not name:
            return False, "a valid recipe name is required"
        if getattr(wf, "total_steps", 0) == 0:
            return False, "the recipe has no steps — build it first"

        yaml_dir = loader.recipe_workflows_dir()
        yaml_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = yaml_dir / f"{name}.yaml"
        workflow_parser.dump_file(wf, yaml_path)

        methods = sorted(lc.workflow_method_maturities(wf, self._catalog).keys())
        self._recipes.add(Recipe(
            name=name, kind="workflow", workflow_path=str(yaml_path),
            methods=methods, maturity="draft",
            provenance={"source": "process_studio_builder",
                        "authored_on": date.today().isoformat()},
        ))
        self._persist("recipe")
        if hasattr(self, "_views"):
            self._views.setCurrentIndex(0)  # jump to Library
        self.reload()
        self._select_by_name(name)
        return True, f"registered recipe '{name}' ({len(methods)} method deps) as draft."

    def _on_register_recipe(self) -> None:
        if self._sandbox is None:
            return
        wf = self._sandbox._build_workflow()
        if getattr(wf, "total_steps", 0) == 0:
            QMessageBox.information(
                self, "Register recipe", "Add steps to the recipe before registering.")
            return
        name, ok = QInputDialog.getText(self, "Register recipe", "Recipe name:")
        if not ok or not name.strip():
            return
        if name.strip() in self._recipes and QMessageBox.question(
            self, "Register recipe", f"Recipe '{name.strip()}' exists. Overwrite?",
        ) != QMessageBox.StandardButton.Yes:
            return
        ok, msg = self.register_recipe_from_workflow(name, wf)
        self._status.setText(msg)
        (QMessageBox.information if ok else QMessageBox.warning)(
            self, "Register recipe", msg)

    # ── Recipe editing (slice 5) ─────────────────────────────────────────────

    @staticmethod
    def _resolve_builder(ref: str):
        """Resolve a ``"module:func"`` builder reference to a callable."""
        import importlib

        mod, _, func = ref.partition(":")
        return getattr(importlib.import_module(mod), func)

    @staticmethod
    def _coerce_param(spec: dict[str, Any], raw: Any) -> Any:
        t = str(spec.get("type", "str"))
        if t == "bool":
            return bool(raw)
        if t == "int":
            return int(raw)
        if t == "float":
            return float(raw)
        return str(raw)

    @staticmethod
    def _workflow_summary(wf: Any) -> str:
        lines = [f"{wf.name} — {wf.total_steps} steps"]
        for i, s in enumerate(wf.resolve_steps(), 1):
            ch = f"  [ch {s.tags['channel']}]" if s.tags.get("channel") else ""
            lines.append(f"  {i}. {s.name} → {s.instrument}.{s.method}(){ch}")
        return "\n".join(lines)

    def edit_recipe_in_builder(self, name: str) -> tuple[bool, str]:
        """Open a workflow-backed recipe's YAML in the embedded Builder."""
        if self._sandbox is None:
            return False, "the Builder is unavailable (no instrument manager)"
        recipe = self._recipes.get(name)
        if not recipe.workflow_path:
            return False, f"'{name}' is builder-backed — no editable workflow YAML"
        try:
            wf = workflow_parser.parse_file(recipe.workflow_path)
        except Exception as exc:
            return False, f"could not load recipe workflow: {exc}"
        self._sandbox.load_workflow(wf)
        for i in range(self._views.count()):
            if self._views.tabText(i) == "Builder":
                self._views.setCurrentIndex(i)
                break
        return True, f"opened '{name}' in the Builder"

    def build_recipe_preview(self, name: str, values: dict[str, Any]) -> tuple[Any, str]:
        """Call a builder-backed recipe's builder with *values*; return (wf|None, msg)."""
        recipe = self._recipes.get(name)
        if not recipe.builder:
            return None, "recipe has no builder to run"
        try:
            wf = self._resolve_builder(recipe.builder)(**values)
        except Exception as exc:
            return None, f"build failed: {exc}"
        return wf, f"built {wf.total_steps} steps"

    def save_recipe_params(self, name: str, values: dict[str, Any]) -> tuple[bool, str]:
        """Persist edited parameter defaults back onto the recipe entry."""
        recipe = self._recipes.get(name)
        for p in recipe.parameters:
            if p["name"] in values:
                p["default"] = values[p["name"]]
        self._persist("recipe")
        self._after_mutation(name)
        return True, f"saved default parameters for '{name}'."

    def _on_edit(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        kind, name = sel
        if kind != "recipe":
            return
        recipe = self._recipes.get(name)
        if recipe.workflow_path and self._sandbox is not None:
            ok, msg = self.edit_recipe_in_builder(name)
            self._status.setText(msg)
            if not ok:
                QMessageBox.warning(self, "Edit", msg)
        elif recipe.parameters:
            self._open_param_dialog(name)
        else:
            QMessageBox.information(
                self, "Edit",
                f"'{name}' is builder-backed with no editable parameters "
                f"or workflow YAML.")

    def _open_param_dialog(self, name: str) -> None:
        recipe = self._recipes.get(name)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit recipe parameters: {name}")
        outer = QVBoxLayout(dlg)
        form = QFormLayout()
        widgets: dict[str, tuple[dict, Any]] = {}
        for p in recipe.parameters:
            if str(p.get("type")) == "bool":
                w = QCheckBox()
                w.setChecked(bool(p.get("default", False)))
            else:
                w = QLineEdit(str(p.get("default", "")))
            widgets[p["name"]] = (p, w)
            form.addRow(p.get("label", p["name"]), w)
        outer.addLayout(form)

        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setPlaceholderText("Preview shows the generated workflow steps.")
        outer.addWidget(preview)

        def collect() -> dict[str, Any]:
            out: dict[str, Any] = {}
            for pname, (spec, w) in widgets.items():
                raw = w.isChecked() if isinstance(w, QCheckBox) else w.text()
                try:
                    out[pname] = self._coerce_param(spec, raw)
                except (TypeError, ValueError):
                    out[pname] = raw
            return out

        row = QHBoxLayout()
        b_preview = QPushButton("Preview")
        b_save = QPushButton("Save defaults")
        b_close = QPushButton("Close")
        row.addWidget(b_preview)
        row.addWidget(b_save)
        row.addStretch()
        row.addWidget(b_close)
        outer.addLayout(row)

        def do_preview():
            wf, msg = self.build_recipe_preview(name, collect())
            preview.setPlainText(self._workflow_summary(wf) if wf is not None else msg)

        def do_save():
            ok, msg = self.save_recipe_params(name, collect())
            self._status.setText(msg)

        b_preview.clicked.connect(do_preview)
        b_save.clicked.connect(do_save)
        b_close.clicked.connect(dlg.accept)
        do_preview()  # show an initial preview
        dlg.exec()

    def _on_tests_finished(self, name: str, kind: str, rc: int) -> None:
        self._test_worker = None
        if rc != 0:
            self._status.setText(f"Tests FAILED (exit {rc}); '{name}' not promoted.")
            self._on_select()  # re-enable buttons
            return
        obj = self._registry(kind).get(name)
        msg = f"Tests passed. '{name}' already at '{obj.maturity}'."
        if Maturity.parse(obj.maturity) < Maturity.TESTED:
            try:
                lc.promote(name, Maturity.TESTED, self._registry(kind))
                self._persist(kind)
                msg = f"Tests passed → '{name}' promoted to 'tested'."
            except GateError as exc:
                msg = f"Tests passed but promotion blocked: {exc}"
        self._after_mutation(name)
        self._status.setText(msg)

    # ── Action-button handlers (dialogs) ─────────────────────────────────────

    def _on_promote(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        _kind, name = sel
        obj = self._registry(_kind).get(name)
        current = Maturity.parse(obj.maturity)
        # Offer only the reachable non-hardware stages; 'validated' is via Sign-off.
        options = [m.label for m in (Maturity.PROTOTYPE, Maturity.TESTED) if m > current]
        if not options:
            QMessageBox.information(
                self, "Promote",
                f"'{name}' is at '{current.label}'. Use Sign-off to reach 'validated'.")
            return
        stage, ok = QInputDialog.getItem(
            self, "Promote", f"Promote '{name}' to:", options, 0, False)
        if not ok:
            return
        ok, msg = self.promote_selected(stage)
        self._status.setText(msg)
        if not ok:
            QMessageBox.warning(self, "Promote blocked", msg)

    def _on_run_tests(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        kind, name = sel
        tests = (self._registry(kind).get(name).evidence.get("tests") or [])
        if not tests:
            QMessageBox.information(self, "Run tests", f"'{name}' has no linked tests.")
            return
        self._status.setText(f"Running {len(tests)} test(s) for '{name}'…")
        for b in (self._btn_promote, self._btn_test, self._btn_signoff, self._btn_supersede):
            b.setEnabled(False)
        self._test_worker = _PytestWorker(list(tests), self)
        self._test_worker.finished_rc.connect(
            lambda rc, n=name, k=kind: self._on_tests_finished(n, k, rc))
        self._test_worker.start()

    def _on_signoff(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        _kind, name = sel
        result = self._prompt_signoff(name)
        if result is None:
            return
        run_id, by, note = result
        ok, msg = self.sign_off_selected(run_id, by, note)
        self._status.setText(msg)
        if not ok:
            QMessageBox.warning(self, "Sign-off blocked", msg)

    def _on_supersede(self) -> None:
        sel = self._selected()
        if sel is None:
            return
        _kind, name = sel
        if QMessageBox.question(
            self, "Supersede",
            f"Start a new draft version of '{name}'?\nThe current version is archived and "
            f"any role bound to it still resolves.",
        ) != QMessageBox.StandardButton.Yes:
            return
        ok, msg = self.supersede_selected()
        self._status.setText(msg)

    def _prompt_signoff(self, name: str) -> tuple[str, str, str] | None:
        """Modal dialog: pick a DataStore run + operator + note. None on cancel."""
        try:
            runs = self._ensure_store().query_runs()
        except Exception as exc:
            QMessageBox.warning(self, "Sign-off", f"Could not read runs: {exc}")
            return None
        if not runs:
            QMessageBox.information(
                self, "Sign-off", "No recorded runs in the DataStore to sign off against.")
            return None

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Sign-off '{name}'")
        form = QFormLayout(dlg)
        combo = QComboBox()
        for r in runs:
            rid = r.get("run_id", "")
            combo.addItem(f"{rid}  ({r.get('workflow_name', '')})", rid)
        form.addRow("Run:", combo)
        by_edit = QLineEdit()
        form.addRow("Operator:", by_edit)
        note_edit = QLineEdit()
        form.addRow("Note:", note_edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        by = by_edit.text().strip()
        if not by:
            QMessageBox.warning(self, "Sign-off", "An operator name is required.")
            return None
        return combo.currentData(), by, note_edit.text().strip()

    def _method_html(self, name: str) -> str:
        st = lc.status(name, self._catalog)
        parts = [self._header(name, "method", st.maturity.label, st.maturity.label)]
        parts.append(self._provenance_html(st.provenance))
        parts.append(self._evidence_html(st.evidence))
        chain = lc.version_chain(name, self._catalog)
        if len(chain) > 1:
            items = "".join(
                f"<li>{escape(k)}{' <i>(current)</i>' if k == name else ''}</li>"
                for k in chain
            )
            parts.append(f"<p><b>Version chain</b></p><ul>{items}</ul>")
        return "".join(parts)

    def _recipe_html(self, name: str) -> str:
        recipe = self._recipes.get(name)
        own = lc.Maturity.parse(recipe.maturity)
        eff = lc.effective_maturity(name, self._catalog, self._recipes)
        parts = [self._header(name, "recipe", own.label, eff.label)]

        source = recipe.builder or recipe.workflow_path or "(none)"
        parts.append(f"<p><b>Defined by</b>: {self._code(source)}</p>")

        # Method deps — flag the one(s) capping effective maturity.
        rows = []
        for m in recipe.methods:
            if m in self._catalog:
                mm = lc.method_maturity(m, self._catalog)
                caps = " &larr; caps effective" if mm == eff and mm < own else ""
                rows.append(
                    f"<li>{escape(m)}: {self._colored(mm.label)}{caps}</li>"
                )
            else:
                rows.append(f"<li>{escape(m)}: <span style='color:#dc2626'>MISSING</span></li>")
        parts.append(f"<p><b>Methods</b></p><ul>{''.join(rows) or '<li>(none)</li>'}</ul>")

        parts.append(self._provenance_html(recipe.provenance))
        parts.append(self._evidence_html(recipe.evidence))
        return "".join(parts)

    # ── HTML helpers ─────────────────────────────────────────────────────────

    def _header(self, name: str, kind: str, own: str, effective: str) -> str:
        eff_line = (
            f" &nbsp; effective: {self._colored(effective)}"
            if effective != own or kind == "recipe"
            else ""
        )
        return (
            f"<h3>{escape(name)}</h3>"
            f"<p>{kind} &nbsp; maturity: {self._colored(own)}{eff_line}</p>"
        )

    @staticmethod
    def _colored(label: str) -> str:
        color = _MATURITY_COLOR.get(label, "#000")
        return f"<b style='color:{color}'>{escape(label)}</b>"

    @staticmethod
    def _code(text: str) -> str:
        return f"<code>{escape(str(text))}</code>"

    def _provenance_html(self, prov: dict[str, Any]) -> str:
        if not prov:
            return ""
        rows = "".join(
            f"<li>{escape(k)}: {escape(str(v))}</li>" for k, v in prov.items() if v
        )
        return f"<p><b>Provenance</b></p><ul>{rows}</ul>" if rows else ""

    def _evidence_html(self, ev: dict[str, Any]) -> str:
        if not ev:
            return "<p><i>No evidence recorded.</i></p>"
        parts = ["<p><b>Evidence</b></p><ul>"]
        tests = ev.get("tests") or []
        if tests:
            parts.append(f"<li>tests: {len(tests)} linked</li>")
            for t in tests:
                parts.append(f"<li style='margin-left:1em'>{self._code(t)}</li>")
        for key in ("validated_run_id", "validated_by", "validated_on", "config_hash", "notes"):
            if ev.get(key):
                parts.append(f"<li>{escape(key)}: {escape(str(ev[key]))}</li>")
        parts.append("</ul>")
        return "".join(parts)
