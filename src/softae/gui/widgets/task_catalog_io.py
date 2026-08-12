"""List<->catalog helpers for the Process Configuration task palette.

Mirrors the role of :mod:`softae.gui.widgets.formulation_io` for the
:class:`~softae.core.task_catalog.TaskCatalog`: small, pure helpers that move
:class:`Task` objects in and out of a ``QListWidget`` so the tab module stays
lean.  Each task row stores its :class:`Task` in ``Qt.UserRole``; category
header rows are non-selectable and carry no task.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from softae.core.task_catalog import Task, TaskCatalog

_UNCATEGORISED = "uncategorised"


def _make_header_item(label: str) -> QListWidgetItem:
    """A bold, non-selectable category divider row."""
    item = QListWidgetItem(f"— {label} —")
    item.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable, not enabled
    font = item.font()
    font.setBold(True)
    item.setFont(font)
    return item


def make_task_item(task: Task) -> QListWidgetItem:
    """A selectable palette row carrying ``task`` in ``Qt.UserRole``."""
    label = task.name
    subtitle = f"{task.instrument}.{task.method}"
    item = QListWidgetItem(f"{label}\n    {subtitle}")
    item.setData(Qt.ItemDataRole.UserRole, task)
    if task.description:
        item.setToolTip(task.description)
    return item


def populate_task_list(list_widget: QListWidget, catalog: TaskCatalog) -> None:
    """Clear and repopulate ``list_widget`` from ``catalog``, grouped by category."""
    list_widget.clear()
    for category, names in sorted(catalog.list_by_category().items()):
        list_widget.addItem(_make_header_item(category or _UNCATEGORISED))
        for name in names:
            list_widget.addItem(make_task_item(catalog.get(name)))


def task_from_item(item: QListWidgetItem | None) -> Task | None:
    """Return the :class:`Task` a row carries, or ``None`` for headers/empty."""
    if item is None:
        return None
    data = item.data(Qt.ItemDataRole.UserRole)
    return data if isinstance(data, Task) else None
