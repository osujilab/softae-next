"""Recipe registry — named runnable procedures with lifecycle metadata.

A :class:`Recipe` is a composed, runnable procedure (a builder function or a
saved workflow YAML) that produces and characterizes **one sample** by stitching
together catalogued **methods** (and, optionally, sub-recipes).  It carries the
same lifecycle fields as a :class:`~softae.core.task_catalog.Task`, plus a
``methods`` dependency list.  Its *effective* maturity is capped by its least
mature dependency (computed in :mod:`softae.core.lifecycle`), so a recipe can
never present as more validated than the methods it runs.

This is the middle rung of the ``method → recipe → campaign`` hierarchy (a
*campaign* is the closed-loop exploration layer on top, distinguished by having
a policy + objective).  TOML-backed (``data_root()/recipes.toml``), mirroring
:class:`~softae.core.task_catalog.TaskCatalog` — same "missing file → empty
registry, never raise" load contract.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w


@dataclass
class Recipe:
    """One named, runnable recipe (single-sample procedure) with lifecycle metadata."""

    name: str
    maturity: str = "draft"  # own declared stage; effective = min(own, deps)
    version: int = 1
    kind: str = "builder"  # "builder" (dotted ref) | "workflow" (yaml path)
    builder: str = ""  # e.g. "softae.core.dropcast:build_dropcast_from_params"
    workflow_path: str = ""  # e.g. "workflows/deposit_anneal_eis.yaml"
    methods: list[str] = field(default_factory=list)  # catalog task (method) deps
    # Editable knobs for a builder-backed recipe (drives a parameter panel):
    # each item is {"name", "type" (str|int|float|bool), "default", optional "label"}.
    parameters: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a TOML-friendly table (omits empty/default fields)."""
        d: dict[str, Any] = {}
        if self.maturity and self.maturity != "draft":
            d["maturity"] = self.maturity
        if self.version and self.version != 1:
            d["version"] = self.version
        if self.kind and self.kind != "builder":
            d["kind"] = self.kind
        if self.builder:
            d["builder"] = self.builder
        if self.workflow_path:
            d["workflow_path"] = self.workflow_path
        if self.methods:
            d["methods"] = list(self.methods)
        if self.parameters:
            d["parameters"] = [dict(p) for p in self.parameters]
        if self.description:
            d["description"] = self.description
        if self.provenance:
            d["provenance"] = dict(self.provenance)
        if self.evidence:
            d["evidence"] = dict(self.evidence)
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Recipe":
        """Rebuild from a parsed TOML table; tolerant of missing keys."""
        methods = data.get("methods") or []
        parameters = data.get("parameters") or []
        provenance = data.get("provenance") or {}
        evidence = data.get("evidence") or {}
        return cls(
            name=name,
            maturity=str(data.get("maturity", "draft")),
            version=int(data.get("version", 1) or 1),
            kind=str(data.get("kind", "builder")),
            builder=str(data.get("builder", "")),
            workflow_path=str(data.get("workflow_path", "")),
            methods=[str(m) for m in methods] if isinstance(methods, list) else [],
            parameters=[dict(p) for p in parameters if isinstance(p, dict)]
            if isinstance(parameters, list) else [],
            description=str(data.get("description", "")),
            provenance=dict(provenance) if isinstance(provenance, dict) else {},
            evidence=dict(evidence) if isinstance(evidence, dict) else {},
        )


class RecipeRegistry:
    """In-memory dict of named :class:`Recipe` instances with TOML persistence."""

    def __init__(self) -> None:
        self._recipes: dict[str, Recipe] = {}

    def add(self, recipe: Recipe) -> None:
        self._recipes[recipe.name] = recipe

    def remove(self, name: str) -> None:
        del self._recipes[name]

    def get(self, name: str) -> Recipe:
        return self._recipes[name]

    def list_names(self) -> list[str]:
        return sorted(self._recipes.keys())

    def __len__(self) -> int:
        return len(self._recipes)

    def __contains__(self, name: str) -> bool:
        return name in self._recipes

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipes": {
                name: self._recipes[name].to_dict() for name in self.list_names()
            }
        }

    def save_toml(self, path: Path) -> None:
        with open(path, "wb") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def load_toml(cls, path: Path) -> "RecipeRegistry":
        """Load from ``path``; a missing file yields an EMPTY registry."""
        reg = cls()
        if not path.exists():
            return reg
        with open(path, "rb") as f:
            data = tomllib.load(f)
        recipes = data.get("recipes", {})
        if isinstance(recipes, dict):
            for name, table in recipes.items():
                if isinstance(table, dict):
                    reg.add(Recipe.from_dict(name, table))
        return reg
