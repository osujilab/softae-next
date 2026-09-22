"""The placeholder catalog shipped inside the package.

Four files — ``tasks.toml``, ``recipes.toml``, ``chemicals.csv``,
``solutions.csv`` — that let a fresh clone compile the campaign examples under
``examples/`` and run the test suite before anyone has built a catalog.

**They are a demonstration, not a rig configuration.** ``data/`` is untracked by
design: every instance develops its own methods, recipes and chemistry for its
own hardware. ``softae-catalog init`` copies these files into ``data/`` as a
starting point, and the copy is what an instance edits. Nothing here is
guaranteed to work on any particular rig out of the box.

The tests read them through :func:`load_defaults`; the bootstrap CLI copies them
through :func:`default_file`. Both go through ``importlib.resources`` so they
resolve from an installed wheel as well as from a source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

#: The files this package ships, in the order the bootstrap reports them. Each
#: is copied to ``data_root()/<same name>``.
CATALOG_FILES: tuple[str, ...] = (
    "tasks.toml", "recipes.toml", "chemicals.csv", "solutions.csv",
)

#: One sentence, printed by the bootstrap and quoted in the docs, so the status
#: of these files is stated wherever they are handed to somebody.
PLACEHOLDER_NOTICE = (
    "These are PLACEHOLDER catalogs, not a rig configuration - copy and edit "
    "your own copy under data/. They are not guaranteed to work on your "
    "hardware as shipped."
)


@dataclass(frozen=True)
class DefaultCatalogs:
    """The three catalogs the campaign compiler resolves names against."""

    tasks: Any
    chemicals: Any
    solutions: Any


@contextmanager
def default_file(name: str) -> Iterator[Path]:
    """A real filesystem path for one shipped file, for the life of the block.

    ``importlib.resources`` extracts the file when the package is a zip, so the
    path is valid only inside the ``with``. Copy or read it there.
    """
    if name not in CATALOG_FILES:
        raise KeyError(f"{name!r} is not a shipped catalog file; have "
                       f"{list(CATALOG_FILES)}")
    with resources.as_file(resources.files(__name__).joinpath(name)) as path:
        yield path


def load_defaults() -> DefaultCatalogs:
    """The shipped catalogs, loaded and proved non-empty.

    Every loader below degrades a file it cannot find to an *empty* catalog, and
    an empty task catalog compiles a workflow with its flush steps silently
    absent. So "the package data did not ship" is refused loudly here rather
    than spelling itself the same way as "the catalogs are fine".
    """
    from softae.core.formulation import ChemicalCatalog, SolutionCatalog
    from softae.core.task_catalog import TaskCatalog

    with default_file("tasks.toml") as path:
        tasks = TaskCatalog.load_toml(path)
    with default_file("chemicals.csv") as path:
        chemicals = ChemicalCatalog.load_csv(path)
    with default_file("solutions.csv") as path:
        solutions = SolutionCatalog.load_csv(path)

    empty = [name for name, catalog in (("tasks.toml", tasks),
                                        ("chemicals.csv", chemicals),
                                        ("solutions.csv", solutions))
             if not len(catalog.list_names())]
    if empty:
        raise RuntimeError(
            f"the shipped placeholder catalog(s) {empty} loaded EMPTY from "
            f"{resources.files(__name__)}. Either the package data was not "
            f"installed with the package, or a file no longer parses - a "
            f"comment line at the top of a CSV does this, because the header "
            f"row is taken literally."
        )
    return DefaultCatalogs(tasks=tasks, chemicals=chemicals, solutions=solutions)
