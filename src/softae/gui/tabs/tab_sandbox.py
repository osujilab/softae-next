"""Process Configuration tab — visual process/recipe builder.

A tree-based Setup → Execution → Teardown sequencer for authoring, editing, and
saving step-based processes ("recipes").  Users can:

* Insert catalogued **tasks** (atomic instrument steps) from a task palette, or
  add steps by hand and save any step back to the task catalog
* Add / remove / drag-reorder steps within and between the three phases
* Pick any registered instrument and method (live-discovered where possible)
* Edit per-step parameters inline
* Save / load recipes as YAML in the canonical schema ``workflow_parser`` and the
  ``softae-run`` CLI consume (recipes round-trip through both)
* Run the assembled recipe with full WorkflowExecutor support

Hierarchy: a **task** (one atomic step, catalogued in ``tasks.toml``) → a
**process recipe** (a named Workflow = Setup/Execution/Teardown arrangement,
saved as YAML) → a **campaign** (``BOCampaignConfig``, which references a recipe).

The middle phase is labelled "Execution"; loops are optional (a repeat count of 1
runs it once).  Internally it maps to the workflow model's ``loop_steps`` field
and the canonical ``loop:`` YAML key, so nothing downstream changes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.config import loader
from softae.core.rig_activity import workflow_instruments
from softae.core.task_catalog import Task, TaskCatalog
from softae.gui.daemon_runner import DaemonRunnerMixin
from softae.gui.rig_claim import rig_run
from softae.gui.widgets import task_catalog_io as tio
from softae.workflows import workflow_parser
from softae.workflows.workflow_executor import WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

# Instrument methods available for workflow steps (generic set)
_DEFAULT_METHODS: dict[str, list[str]] = {
    "stage": ["move_to", "move_by", "home_stage", "live_position"],
    "syringe": ["single_pump", "head_flip", "head_retract", "head_descend"],
    "temp_controller": ["write_sp", "get_pv", "get_sp", "ramp_linear", "wait"],
    "rh_controller": ["set_setpoint", "start", "stop", "get_H", "wait"],
    "ht_sensor": ["get_T", "get_H"],
    "pico1": ["sendscript_getdata", "eis_extractdata"],
    "pico2": ["sendscript_getdata", "eis_extractdata"],
    "camera": ["snap", "save_image"],
    "lamp": ["on", "off"],
    "keithley": ["read_resistance"],
    "liquid_handler": [
        "startup_flush", "single_drop_simul", "precondition_flush", "star_mix",
    ],
    # Virtual instrument for hardware-agnostic control steps (executor built-in).
    "control": ["wait"],
}


def _json_safe(value: Any) -> bool:
    """True if ``value`` round-trips through ``json.dumps`` as-is."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _placeholder_for(annotation: Any) -> Any:
    """A JSON-fillable placeholder inferred from a parameter's type annotation."""
    return {int: 0, float: 0.0, bool: False, str: "", list: [], dict: {}}.get(
        annotation, None
    )


class _PhaseWorkflowTree(QTreeWidget):
    """Tree that constrains InternalMove drops to step reordering.

    The three phase roots (Setup / Execution / Teardown) are fixed containers:
    a dragged step may only reorder within a phase or move to another phase.
    Drops that would nest a step under another step, reorder the phase roots,
    or land at top level are rejected.  ``on_drop`` (if set) is invoked after a
    successful move so the owner can refresh the preview / expansion.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.on_drop = None  # optional callable set by the owner

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt override)
        target = self.itemAt(event.position().toPoint())
        pos = self.dropIndicatorPosition()
        Ind = QAbstractItemView.DropIndicatorPosition
        is_root = target is not None and target.parent() is None

        if target is None or pos == Ind.OnViewport:
            allowed = False  # top-level / empty-space drop → reject
        elif is_root:
            allowed = pos == Ind.OnItem  # onto a phase root → drop as its child
        else:
            allowed = pos in (Ind.AboveItem, Ind.BelowItem)  # reorder as a sibling

        if not allowed:
            event.ignore()
            return

        super().dropEvent(event)
        if callable(self.on_drop):
            self.on_drop()


class SandboxTab(DaemonRunnerMixin, QWidget):
    """Process Configuration tab — task/recipe builder with drag-drop steps."""

    _sig_step_start = Signal(str, int, int)
    _sig_step_complete = Signal(str, int, int, object)
    _sig_step_error = Signal(str, int, int, str)
    _sig_state_change = Signal(str, str)
    _sig_done = Signal(int)

    def __init__(self, manager: "InstrumentManager", parent: QWidget | None = None):
        super().__init__(parent)
        self._manager = manager
        self._executor: WorkflowExecutor | None = None
        self._run_thread: threading.Thread | None = None
        self._task_catalog = self._load_task_catalog()
        # True while the editor is being populated from a selected step, so the
        # live editor-change handler doesn't write half-populated state back onto
        # the newly selected item (see _on_tree_selection / _on_editor_changed).
        self._loading = False
        # Last run's failure message, surfaced by _ui_done (workflow-level errors —
        # e.g. a duplicate-name/validation failure raised before any step — never
        # reach on_step_error, so their detail would otherwise be lost).
        self._run_error = ""
        self._build_ui()
        self._connect_signals()

    @staticmethod
    def _load_task_catalog() -> TaskCatalog:
        """Load the task catalog from the canonical path; empty on any failure."""
        try:
            return TaskCatalog.load_toml(loader.tasks_toml_path())
        except Exception:
            logger.warning("task_catalog_load_failed", exc_info=True)
            return TaskCatalog()

    # ── UI ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Toolbar ---
        toolbar = QHBoxLayout()

        self._btn_add_setup = QPushButton("+ Setup Step")
        self._btn_add_setup.clicked.connect(lambda: self._add_step("setup"))
        toolbar.addWidget(self._btn_add_setup)

        self._btn_add_loop = QPushButton("+ Execution Step")
        self._btn_add_loop.clicked.connect(lambda: self._add_step("loop"))
        toolbar.addWidget(self._btn_add_loop)

        self._btn_add_teardown = QPushButton("+ Teardown Step")
        self._btn_add_teardown.clicked.connect(lambda: self._add_step("teardown"))
        toolbar.addWidget(self._btn_add_teardown)

        self._btn_remove = QPushButton("− Remove")
        self._btn_remove.clicked.connect(self._remove_step)
        toolbar.addWidget(self._btn_remove)

        self._btn_move_up = QPushButton("↑ Up")
        self._btn_move_up.clicked.connect(lambda: self._move_step(-1))
        toolbar.addWidget(self._btn_move_up)

        self._btn_move_down = QPushButton("↓ Down")
        self._btn_move_down.clicked.connect(lambda: self._move_step(1))
        toolbar.addWidget(self._btn_move_down)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Repeat execution:"))
        self._spin_iterations = QSpinBox()
        self._spin_iterations.setRange(1, 1000)
        self._spin_iterations.setValue(1)
        self._spin_iterations.setSuffix(" ×")
        self._spin_iterations.setToolTip(
            "How many times to repeat the Execution phase. 1 = run once (no loop)."
        )
        toolbar.addWidget(self._spin_iterations)

        layout.addLayout(toolbar)

        # --- Main splitter: task palette + tree + params + preview ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Far left: Task palette (catalogued atomic steps)
        splitter.addWidget(self._build_task_palette())

        # Left: Workflow tree
        tree_widget = QWidget()
        tree_lay = QVBoxLayout(tree_widget)
        tree_lay.setContentsMargins(0, 0, 0, 0)

        self._tree = _PhaseWorkflowTree()
        self._tree.on_drop = self._on_tree_drop
        self._tree.setHeaderLabels(["Phase / Step", "Instrument", "Method"])
        self._tree.setColumnCount(3)
        self._tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tree.setRootIsDecorated(True)
        self._tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        self._tree.currentItemChanged.connect(self._on_tree_selection)

        # Create phase root nodes
        self._setup_root = QTreeWidgetItem(self._tree, ["Setup", "", ""])
        self._setup_root.setFlags(
            self._setup_root.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
        )
        self._loop_root = QTreeWidgetItem(self._tree, ["Execution", "", ""])
        self._loop_root.setFlags(
            self._loop_root.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
        )
        self._teardown_root = QTreeWidgetItem(self._tree, ["Teardown", "", ""])
        self._teardown_root.setFlags(
            self._teardown_root.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
        )

        for root in (self._setup_root, self._loop_root, self._teardown_root):
            root.setExpanded(True)
            font = root.font(0)
            font.setBold(True)
            root.setFont(0, font)

        tree_lay.addWidget(self._tree)
        splitter.addWidget(tree_widget)

        # Centre: Step editor
        editor_widget = QWidget()
        editor_lay = QVBoxLayout(editor_widget)
        editor_lay.setContentsMargins(4, 0, 4, 0)

        ed_grp = QGroupBox("Step Editor")
        ed_inner = QVBoxLayout(ed_grp)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._edit_name = QLineEdit()
        self._edit_name.setPlaceholderText("step_name")
        self._edit_name.textChanged.connect(self._on_editor_changed)
        name_row.addWidget(self._edit_name)
        ed_inner.addLayout(name_row)

        inst_row = QHBoxLayout()
        inst_row.addWidget(QLabel("Instrument:"))
        self._combo_instrument = QComboBox()
        self._combo_instrument.addItems(self._instrument_names())
        self._combo_instrument.currentTextChanged.connect(self._on_instrument_changed)
        inst_row.addWidget(self._combo_instrument)

        inst_row.addWidget(QLabel("Method:"))
        self._combo_method = QComboBox()
        inst_row.addWidget(self._combo_method)
        self._combo_method.currentTextChanged.connect(self._on_editor_changed)
        self._combo_method.currentTextChanged.connect(self._update_signature_hint)
        ed_inner.addLayout(inst_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Params (JSON):"))
        self._edit_params = QLineEdit()
        self._edit_params.setPlaceholderText('{"key": "value"}')
        self._edit_params.textChanged.connect(self._on_editor_changed)
        param_row.addWidget(self._edit_params)
        self._btn_prefill = QPushButton("Prefill")
        self._btn_prefill.setToolTip(
            "Fill missing parameters from the selected method's signature, keeping "
            "any values already entered (live instruments only)."
        )
        self._btn_prefill.clicked.connect(self._on_prefill_params)
        param_row.addWidget(self._btn_prefill)
        ed_inner.addLayout(param_row)

        # Read-only signature hint, refreshed whenever the method changes.
        self._lbl_signature = QLabel("")
        self._lbl_signature.setWordWrap(True)
        self._lbl_signature.setStyleSheet(
            "color: gray; font-family: monospace; font-size: 10px;"
        )
        self._lbl_signature.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        ed_inner.addWidget(self._lbl_signature)

        timeout_row = QHBoxLayout()
        timeout_row.addWidget(QLabel("Timeout (s):"))
        self._spin_timeout = QDoubleSpinBox()
        self._spin_timeout.setRange(0, 9999)
        self._spin_timeout.setValue(300)
        self._spin_timeout.valueChanged.connect(self._on_editor_changed)
        timeout_row.addWidget(self._spin_timeout)

        timeout_row.addWidget(QLabel("Retries:"))
        self._spin_retry = QSpinBox()
        self._spin_retry.setRange(0, 10)
        self._spin_retry.setValue(0)
        self._spin_retry.valueChanged.connect(self._on_editor_changed)
        timeout_row.addWidget(self._spin_retry)
        timeout_row.addStretch()
        ed_inner.addLayout(timeout_row)

        editor_lay.addWidget(ed_grp)

        # Populate methods for initial instrument
        self._on_instrument_changed(self._combo_instrument.currentText())

        splitter.addWidget(editor_widget)

        # Right: Preview + run
        right_widget = QWidget()
        right_lay = QVBoxLayout(right_widget)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self._txt_preview = QPlainTextEdit()
        self._txt_preview.setReadOnly(True)
        self._txt_preview.setStyleSheet("font-family: monospace; font-size: 11px;")
        right_lay.addWidget(self._txt_preview)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)  # task palette
        splitter.setStretchFactor(1, 2)  # tree
        splitter.setStretchFactor(2, 2)  # step editor
        splitter.setStretchFactor(3, 3)  # preview
        layout.addWidget(splitter)

        # --- Bottom: save/load/run ---
        bottom = QHBoxLayout()

        self._btn_save = QPushButton("Save YAML…")
        self._btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self._btn_save)

        self._btn_load = QPushButton("Load YAML…")
        self._btn_load.clicked.connect(self._on_load)
        bottom.addWidget(self._btn_load)

        self._btn_preview = QPushButton("Preview")
        self._btn_preview.clicked.connect(self._on_preview)
        bottom.addWidget(self._btn_preview)

        bottom.addStretch()

        self._btn_run = QPushButton("▶  Run")
        self._btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; padding: 6px;"
        )
        self._btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self._btn_run)

        self._btn_abort = QPushButton("⏹  Abort")
        self._btn_abort.setStyleSheet("background-color: #f44336; color: white;")
        self._btn_abort.setEnabled(False)
        self._btn_abort.clicked.connect(self._on_abort)
        bottom.addWidget(self._btn_abort)

        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        bottom.addWidget(self._progress)

        self._lbl_status = QLabel("Idle")
        bottom.addWidget(self._lbl_status)

        layout.addLayout(bottom)

    def _connect_signals(self) -> None:
        self._sig_step_start.connect(self._ui_step_start)
        self._sig_step_complete.connect(self._ui_step_complete)
        self._sig_step_error.connect(self._ui_step_error)
        self._sig_state_change.connect(self._ui_state_change)
        self._sig_done.connect(self._ui_done)

    # ── Tree manipulation ────────────────────────────────────────────

    def _phase_root(self, phase: str) -> QTreeWidgetItem:
        return {
            "setup": self._setup_root,
            "loop": self._loop_root,
            "teardown": self._teardown_root,
        }[phase]

    def _add_step(self, phase: str) -> QTreeWidgetItem:
        """Add a new step under the specified phase."""
        root = self._phase_root(phase)
        idx = root.childCount()
        inst = self._combo_instrument.currentText()
        method = self._combo_method.currentText()
        # Unique within the phase — remove-then-add can reuse an index, so guard
        # against colliding with an existing step name (duplicate names break
        # execution: the DAG keys on them).
        name = self._unique_step_name(root, f"{phase}_step_{idx + 1}")

        item = QTreeWidgetItem(root, [name, inst, method])
        item.setData(0, Qt.ItemDataRole.UserRole, {
            "name": name,
            "instrument": inst,
            "method": method,
            "params": {},
            "timeout_s": 300,
            "retry": 0,
        })
        root.setExpanded(True)
        self._tree.setCurrentItem(item)
        return item

    def _remove_step(self) -> None:
        """Remove the currently selected step."""
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return  # don't remove phase roots
        parent = item.parent()
        parent.removeChild(item)

    def _move_step(self, direction: int) -> None:
        """Move the selected step up (−1) or down (+1) within its phase."""
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return
        parent = item.parent()
        idx = parent.indexOfChild(item)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= parent.childCount():
            return
        parent.removeChild(item)
        parent.insertChild(new_idx, item)
        self._tree.setCurrentItem(item)

    # ── Step editor ──────────────────────────────────────────────────

    def _on_tree_selection(self, current: QTreeWidgetItem | None,
                           previous: QTreeWidgetItem | None) -> None:
        """Populate the editor when a step is selected.

        Guarded by ``self._loading``: populating the editor programmatically
        fires the widgets' change signals, and ``_on_editor_changed`` would
        otherwise write the *half-populated* editor (e.g. the new method with the
        previous step's params) back onto the newly selected item — the source of
        the "params change when I switch steps" corruption.  The flag makes those
        intermediate signals no-ops so only the user's own edits persist.
        """
        if current is None or current.parent() is None:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        self._loading = True
        try:
            self._edit_name.setText(data.get("name", ""))
            inst = data.get("instrument", "")
            idx = self._combo_instrument.findText(inst)
            if idx >= 0:
                self._combo_instrument.setCurrentIndex(idx)
            self._on_instrument_changed(inst)
            midx = self._combo_method.findText(data.get("method", ""))
            if midx >= 0:
                self._combo_method.setCurrentIndex(midx)
            self._edit_params.setText(json.dumps(data.get("params", {})))
            self._spin_timeout.setValue(data.get("timeout_s", 300))
            self._spin_retry.setValue(data.get("retry", 0))
        finally:
            self._loading = False

        self._update_signature_hint()

    def _on_instrument_changed(self, inst: str) -> None:
        """Update method combo when instrument changes.

        Preserves the method combo's prior *blocked* state rather than
        unconditionally unblocking — otherwise, when called mid-population from
        :meth:`_on_tree_selection`, it would re-enable the combo's signals early.
        """
        prev = self._combo_method.blockSignals(True)
        self._combo_method.clear()
        self._combo_method.addItems(self._methods_for(inst))
        self._combo_method.blockSignals(prev)
        self._update_signature_hint()

    # ── Instrument / method discovery ────────────────────────────────

    def _instrument_names(self) -> list[str]:
        """Instrument names: live manager registry ∪ the static fallback set."""
        names = set(_DEFAULT_METHODS)
        try:
            names |= set(self._manager.names)
        except Exception:
            pass
        return sorted(names)

    def _methods_for(self, inst_name: str) -> list[str]:
        """Methods offered for an instrument.

        Starts from the curated ``_DEFAULT_METHODS`` list, then — when the
        instrument is live — merges in its driver-specific public methods
        (those declared on the concrete driver, not the ``BaseInstrument`` ABC),
        so real capabilities like ``temp_controller.anneal`` or
        ``piezo.apply_profile`` surface without hand-maintaining the dict.
        """
        methods = list(_DEFAULT_METHODS.get(inst_name, []))
        try:
            inst = self._manager.get(inst_name)
        except Exception:
            inst = None
        if inst is not None:
            try:
                from softae.server.base_instrument import BaseInstrument
                base_attrs = set(dir(BaseInstrument))
            except Exception:
                base_attrs = set()
            for m in dir(type(inst)):
                if m.startswith("_") or m in base_attrs or m in methods:
                    continue
                if callable(getattr(inst, m, None)):
                    methods.append(m)
        return methods

    # ── Method-signature introspection ───────────────────────────────

    def _selected_method_signature(self) -> tuple[str, dict[str, Any]] | None:
        """Introspect the currently selected ``instrument.method``.

        Returns ``(hint, template)`` where ``hint`` is a readable signature
        string (e.g. ``move_to(x, y, speed=10.0)``) and ``template`` is a
        JSON-fillable ``{param: default_or_placeholder}`` dict.  Returns ``None``
        when the instrument isn't live (a static ``_DEFAULT_METHODS`` entry
        carries no signature to read) or the method can't be introspected.
        """
        method_name = self._combo_method.currentText()
        if not method_name:
            return None
        try:
            inst = self._manager.get(self._combo_instrument.currentText())
        except Exception:
            inst = None
        fn = getattr(inst, method_name, None) if inst is not None else None
        if not callable(fn):
            return None
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return None

        parts: list[str] = []
        template: dict[str, Any] = {}
        for pname, p in sig.parameters.items():
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if p.default is inspect.Parameter.empty:
                parts.append(pname)
                template[pname] = _placeholder_for(p.annotation)
            else:
                parts.append(f"{pname}={p.default!r}")
                template[pname] = (
                    p.default if _json_safe(p.default) else _placeholder_for(p.annotation)
                )
        return f"{method_name}({', '.join(parts)})", template

    def _update_signature_hint(self) -> None:
        """Refresh the read-only signature hint below the Params field."""
        sig = self._selected_method_signature()
        if sig is None:
            method = self._combo_method.currentText()
            self._lbl_signature.setText(
                f"{method}(…) — signature unavailable (instrument offline)"
                if method
                else ""
            )
            self._btn_prefill.setEnabled(False)
            return
        hint, _template = sig
        self._lbl_signature.setText(hint)
        self._btn_prefill.setEnabled(True)

    def _on_prefill_params(self) -> None:
        """Fill missing params from the signature, preserving entered values.

        Non-destructive: starts from the method's default template and overlays
        whatever valid JSON is already in the field, so absent parameters get
        added without discarding anything you've typed.
        """
        sig = self._selected_method_signature()
        if sig is None:
            return
        _hint, template = sig
        try:
            current = json.loads(self._edit_params.text() or "{}")
        except json.JSONDecodeError:
            current = {}
        if not isinstance(current, dict):
            current = {}
        self._edit_params.setText(json.dumps({**template, **current}))

    # ── Task palette ─────────────────────────────────────────────────

    # Display label → internal phase token (model keeps the "loop" token).
    _PHASE_TOKENS = {"Setup": "setup", "Execution": "loop", "Teardown": "teardown"}

    def _build_task_palette(self) -> QWidget:
        """Left-most pane: the catalogued-task palette + CRUD controls."""
        palette = QWidget()
        lay = QVBoxLayout(palette)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.addWidget(QLabel("<b>Task Catalog</b>"))

        self._task_list = QListWidget()
        self._task_list.itemDoubleClicked.connect(self._on_task_double_clicked)
        lay.addWidget(self._task_list)

        ins_row = QHBoxLayout()
        ins_row.addWidget(QLabel("Insert into:"))
        self._combo_task_phase = QComboBox()
        self._combo_task_phase.addItems(list(self._PHASE_TOKENS))
        self._combo_task_phase.setCurrentText("Execution")
        ins_row.addWidget(self._combo_task_phase)
        lay.addLayout(ins_row)

        self._btn_insert_task = QPushButton("Insert as Step")
        self._btn_insert_task.clicked.connect(self._on_insert_task)
        lay.addWidget(self._btn_insert_task)

        self._btn_save_task = QPushButton("Save Current Step as Task…")
        self._btn_save_task.clicked.connect(self._on_save_step_as_task)
        lay.addWidget(self._btn_save_task)

        task_btn_row = QHBoxLayout()
        self._btn_remove_task = QPushButton("Remove Task")
        self._btn_remove_task.clicked.connect(self._on_remove_task)
        task_btn_row.addWidget(self._btn_remove_task)
        self._btn_reload_tasks = QPushButton("Reload")
        self._btn_reload_tasks.clicked.connect(self._on_reload_tasks)
        task_btn_row.addWidget(self._btn_reload_tasks)
        lay.addLayout(task_btn_row)

        self._refresh_task_palette()
        return palette

    def _refresh_task_palette(self) -> None:
        tio.populate_task_list(self._task_list, self._task_catalog)

    def _current_task_phase_token(self) -> str:
        return self._PHASE_TOKENS[self._combo_task_phase.currentText()]

    def _on_task_double_clicked(self, item) -> None:
        task = tio.task_from_item(item)
        if task is not None:
            self._insert_task(task, self._current_task_phase_token())

    def _on_insert_task(self) -> None:
        task = tio.task_from_item(self._task_list.currentItem())
        if task is None:
            QMessageBox.information(self, "No Task", "Select a task in the catalog first.")
            return
        self._insert_task(task, self._current_task_phase_token())

    def _insert_task(self, task: Task, phase: str) -> None:
        """Append a catalogued task as a step under ``phase`` (name de-duped)."""
        root = self._phase_root(phase)
        step = task.to_step(self._unique_step_name(root, task.name))
        item = QTreeWidgetItem(root, [step.name, step.instrument, step.method])
        item.setData(0, Qt.ItemDataRole.UserRole, {
            "name": step.name,
            "instrument": step.instrument,
            "method": step.method,
            "params": dict(step.params),
            "timeout_s": 300 if step.timeout_s is None else step.timeout_s,
            "retry": step.retry,
        })
        root.setExpanded(True)
        self._tree.setCurrentItem(item)
        self._on_preview()

    @staticmethod
    def _unique_step_name(root: QTreeWidgetItem, base: str) -> str:
        existing = {root.child(i).text(0) for i in range(root.childCount())}
        if base not in existing:
            return base
        n = 2
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def _on_save_step_as_task(self) -> None:
        """Save the currently selected step to the task catalog."""
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            QMessageBox.information(self, "No Step", "Select a step in the tree first.")
            return
        data = item.data(0, Qt.ItemDataRole.UserRole) or {}

        name, ok = QInputDialog.getText(
            self, "Save Step as Task", "Task name:", text=data.get("name", "task")
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._task_catalog and (
            QMessageBox.question(
                self, "Overwrite", f"Task '{name}' already exists. Overwrite?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        category, _ = QInputDialog.getText(
            self, "Save Step as Task", "Category (optional):"
        )
        description, _ = QInputDialog.getText(
            self, "Save Step as Task", "Description (optional):"
        )
        self._task_catalog.add(Task(
            name=name,
            instrument=data.get("instrument", ""),
            method=data.get("method", ""),
            params=dict(data.get("params", {})),
            timeout_s=data.get("timeout_s"),
            retry=int(data.get("retry", 0) or 0),
            description=description.strip(),
            category=category.strip(),
        ))
        if self._persist_task_catalog():
            self._refresh_task_palette()

    def _on_remove_task(self) -> None:
        task = tio.task_from_item(self._task_list.currentItem())
        if task is None:
            return
        if QMessageBox.question(
            self, "Remove Task", f"Remove task '{task.name}' from the catalog?"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._task_catalog.remove(task.name)
        except KeyError:
            pass
        if self._persist_task_catalog():
            self._refresh_task_palette()

    def _on_reload_tasks(self) -> None:
        self._task_catalog = self._load_task_catalog()
        self._refresh_task_palette()

    def _persist_task_catalog(self) -> bool:
        try:
            path = loader.tasks_toml_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._task_catalog.save_toml(path)
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Could not save task catalog:\n{exc}")
            return False

    def _on_tree_drop(self) -> None:
        """Post-drop fixup: keep phase roots expanded and refresh the preview."""
        for root in (self._setup_root, self._loop_root, self._teardown_root):
            root.setExpanded(True)
        self._on_preview()

    def _on_editor_changed(self) -> None:
        """Write editor values back to the selected tree item.

        No-op while :attr:`_loading` is set (the editor is being populated from a
        newly selected step), so programmatic population never overwrites a step
        with a half-filled editor.
        """
        if self._loading:
            return
        item = self._tree.currentItem()
        if item is None or item.parent() is None:
            return
        try:
            params = json.loads(self._edit_params.text() or "{}")
        except json.JSONDecodeError:
            params = {}

        data = {
            "name": self._edit_name.text(),
            "instrument": self._combo_instrument.currentText(),
            "method": self._combo_method.currentText(),
            "params": params,
            "timeout_s": self._spin_timeout.value(),
            "retry": self._spin_retry.value(),
        }
        item.setData(0, Qt.ItemDataRole.UserRole, data)
        item.setText(0, data["name"])
        item.setText(1, data["instrument"])
        item.setText(2, data["method"])

    # ── Workflow assembly ────────────────────────────────────────────

    def _steps_from_tree(self, root: QTreeWidgetItem) -> list[WorkflowStep]:
        """Extract WorkflowSteps from a phase's tree children."""
        steps: list[WorkflowStep] = []
        for i in range(root.childCount()):
            child = root.child(i)
            data = child.data(0, Qt.ItemDataRole.UserRole)
            if not data:
                continue
            steps.append(WorkflowStep(
                name=data["name"],
                instrument=data["instrument"],
                method=data["method"],
                params=data.get("params", {}),
                timeout_s=data.get("timeout_s", 300),
                retry=data.get("retry", 0),
            ))
        return steps

    @staticmethod
    def _uniquify_phase(steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """Return *steps* with names made unique within the phase.

        Step names are execution identifiers (the executor's dependency DAG and
        the result map key on them); duplicates within a phase silently drop a
        step and can fabricate a false dependency cycle.  A rename in the editor
        can introduce a collision, so we resolve them here at build time.
        """
        from dataclasses import replace

        seen: set[str] = set()
        out: list[WorkflowStep] = []
        for step in steps:
            name = step.name
            n = 2
            while name in seen:
                name = f"{step.name}_{n}"
                n += 1
            seen.add(name)
            out.append(step if name == step.name else replace(step, name=name))
        return out

    def _build_workflow(self) -> Workflow:
        """Assemble a Workflow from the current tree state."""
        setup = self._uniquify_phase(self._steps_from_tree(self._setup_root))
        loop = self._uniquify_phase(self._steps_from_tree(self._loop_root))
        teardown = self._uniquify_phase(self._steps_from_tree(self._teardown_root))
        iterations = self._spin_iterations.value()

        return Workflow(
            name=f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Process recipe",
            setup=setup,
            loop_steps=loop,
            teardown=teardown,
            iterations=iterations,
            metadata={"source": "sandbox"},
        )

    # ── Preview ──────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        """Show a YAML-like preview of the assembled workflow."""
        try:
            wf = self._build_workflow()
            lines = [
                f"name: {wf.name}",
                f"iterations: {wf.iterations}",
                f"total_steps: {wf.total_steps}",
                "",
                "Resolved steps:",
            ]
            for i, step in enumerate(wf.resolve_steps()):
                lines.append(
                    f"  {i + 1}. {step.name} → {step.instrument}.{step.method}()"
                )
                if step.params:
                    for k, v in step.params.items():
                        lines.append(f"       {k}: {v}")
            self._txt_preview.setPlainText("\n".join(lines))
        except Exception as exc:
            self._txt_preview.setPlainText(f"Error: {exc}")

    # ── Save / Load ──────────────────────────────────────────────────
    # Recipes use the canonical schema that ``workflow_parser`` and the
    # ``softae-run`` CLI consume, so they round-trip through both.

    def _on_save(self) -> None:
        """Save the current recipe as YAML (canonical ``softae-run`` schema)."""
        wf = self._build_workflow()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe", "process_recipe.yaml", "YAML (*.yaml *.yml);;JSON (*.json)"
        )
        if not path:
            return
        try:
            workflow_parser.dump_file(wf, path)
            self._lbl_status.setText(f"Saved: {Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _populate_step_item(self, phase: str, step: WorkflowStep) -> None:
        """Append one step to a phase root's tree (used by load)."""
        root = self._phase_root(phase)
        item = QTreeWidgetItem(root, [step.name, step.instrument, step.method])
        item.setData(0, Qt.ItemDataRole.UserRole, {
            "name": step.name,
            "instrument": step.instrument,
            "method": step.method,
            "params": dict(step.params),
            "timeout_s": 300 if step.timeout_s is None else step.timeout_s,
            "retry": step.retry,
        })
        root.setExpanded(True)

    def _on_load(self) -> None:
        """Load a recipe (any valid ``softae-run`` workflow) and populate the tree."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "", "Recipes (*.yaml *.yml *.json);;All (*)"
        )
        if not path:
            return
        try:
            wf = workflow_parser.parse_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return
        self.load_workflow(wf)
        self._lbl_status.setText(f"Loaded: {Path(path).name}")

    def load_workflow(self, wf: Workflow) -> None:
        """Populate the builder tree from a :class:`Workflow` (Setup/Loop/Teardown).

        The programmatic path behind Load… — also used by Process Studio's
        'Edit in Builder' to open a registered recipe for editing.
        """
        for root in (self._setup_root, self._loop_root, self._teardown_root):
            while root.childCount():
                root.removeChild(root.child(0))

        self._spin_iterations.setValue(max(1, wf.iterations))
        for step in wf.setup:
            self._populate_step_item("setup", step)
        for step in wf.loop_steps:
            self._populate_step_item("loop", step)
        for step in wf.teardown:
            self._populate_step_item("teardown", step)
        self._on_preview()

    # ── Run workflow ─────────────────────────────────────────────────

    def _on_run(self) -> None:
        """Build and run the sandbox workflow."""
        try:
            wf = self._build_workflow()
        except Exception as exc:
            QMessageBox.warning(self, "Workflow Error", str(exc))
            return

        if wf.total_steps == 0:
            QMessageBox.information(self, "Empty", "Add steps before running.")
            return

        self._executor = WorkflowExecutor(self._manager)
        self._executor.on_step_start = (
            lambda s, i, t: self._sig_step_start.emit(s.name, i, t)
        )
        self._executor.on_step_complete = (
            # Executor passes (step, index, total, result, elapsed); drop elapsed.
            lambda s, i, t, r, e: self._sig_step_complete.emit(s.name, i, t, r)
        )
        self._executor.on_step_error = (
            lambda s, i, t, e: self._sig_step_error.emit(s.name, i, t, str(e))
        )
        self._executor.on_state_change = (
            lambda o, n: self._sig_state_change.emit(o.name, n.name)
        )

        self._btn_run.setEnabled(False)
        self._btn_abort.setEnabled(True)
        self._progress.setRange(0, wf.total_steps)
        self._progress.setValue(0)
        self._lbl_status.setText("Running…")
        self._on_preview()

        self._run_thread = threading.Thread(
            target=self._run_thread_fn, args=(wf,), daemon=True
        )
        self._run_thread.start()

    def _run_thread_fn(self, wf: Workflow) -> None:
        # Rebind per-instrument asyncio.Lock objects to the fresh loop
        # asyncio.run() creates below. Locks bound to the GUI-startup loop (or a
        # previous run's now-closed loop) would otherwise deadlock the first
        # ``async with inst._lock`` in _dispatch(), hanging the run at step 1.
        try:
            self._manager.reset_locks()
        except Exception:
            logger.warning("sandbox_reset_locks_failed", exc_info=True)
        try:
            # Claim the rig for the run so the background purge timer defers
            # instead of travelling the stage to the flush basin mid-workflow.
            # Scoped to what the workflow's steps name; an empty or unreadable
            # union widens to the whole rig. Idle rest is deliberately left
            # alone — a sandbox workflow is the operator's own composition and
            # ends where they put it, and step B claims without moving anything
            # that did not move before.
            with rig_run(self, f"sandbox:{getattr(wf, 'name', 'workflow')}",
                         instruments=workflow_instruments(wf),
                         manage_rest=False):
                asyncio.run(self._executor.run(wf))
            self._run_error = ""
            self._sig_done.emit(0)
        except Exception as exc:
            # Capture the message so the GUI can surface it (see _ui_done). This
            # is the only place a workflow-level failure's detail is available.
            self._run_error = str(exc)
            logger.error("sandbox_workflow_error", error=str(exc))
            self._sig_done.emit(1)

    def _on_abort(self) -> None:
        if self._executor is not None:
            self._executor.abort()

    # ── Daemon shutdown seam (signal-first abort + bounded join) ─────────
    def _abort_run_impl(self) -> None:
        if self._executor is not None:
            self._executor.abort()

    def _runner_thread(self):
        return self._run_thread

    # ── UI update slots ──────────────────────────────────────────────

    def _ui_step_start(self, name: str, index: int, total: int) -> None:
        self._progress.setValue(index)
        self._lbl_status.setText(f"[{index + 1}/{total}] {name}")

    def _ui_step_complete(self, name: str, index: int, total: int, result: object) -> None:
        self._progress.setValue(index + 1)

    def _ui_step_error(self, name: str, index: int, total: int, error: str) -> None:
        self._lbl_status.setText(f"Error: {name} — {error[:60]}")
        # Full detail in the preview pane (status is truncated).
        self._txt_preview.setPlainText(f"Step '{name}' failed:\n{error}")

    def _ui_state_change(self, old: str, new: str) -> None:
        self._lbl_status.setText(f"State: {new}")

    def _ui_done(self, exit_code: int) -> None:
        self._btn_run.setEnabled(True)
        self._btn_abort.setEnabled(False)
        if exit_code == 0:
            self._progress.setValue(self._progress.maximum())
            self._lbl_status.setText("Completed ✓")
            return
        state = self._executor.state.name if self._executor else "ERROR"
        err = self._run_error
        if err:
            # Surface the failure detail: truncated in the status line, full text
            # in the preview pane (workflow-level errors have no step to attach to).
            self._lbl_status.setText(f"Failed ({state}): {err[:80]}")
            self._txt_preview.setPlainText(f"Run failed ({state}):\n{err}")
        else:
            self._lbl_status.setText(f"Failed ({state})")
