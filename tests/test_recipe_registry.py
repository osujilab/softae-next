"""Tests for the recipe registry (Phase 2)."""

from __future__ import annotations

from pathlib import Path

from softae.core.recipe_registry import Recipe, RecipeRegistry


def test_recipe_round_trip():
    r = Recipe(
        name="dropcast_sweep",
        maturity="tested",
        builder="softae.core.dropcast:build_dropcast_sweep_workflow",
        methods=["startup_flush_full", "single_drop_simul"],
        provenance={"ported_by": "p"},
        evidence={"tests": ["t::a"]},
    )
    rebuilt = Recipe.from_dict("dropcast_sweep", r.to_dict())
    assert rebuilt.maturity == "tested"
    assert rebuilt.methods == ["startup_flush_full", "single_drop_simul"]
    assert rebuilt.builder.endswith(":build_dropcast_sweep_workflow")
    assert rebuilt.evidence["tests"] == ["t::a"]


def test_default_recipe_omits_defaults():
    d = Recipe(name="x").to_dict()
    assert "maturity" not in d and "version" not in d and "kind" not in d


def test_workflow_path_round_trips():
    r = Recipe(name="r", kind="workflow", workflow_path="workflows/deposit_anneal.yaml")
    rebuilt = Recipe.from_dict("r", r.to_dict())
    assert rebuilt.kind == "workflow"
    assert rebuilt.workflow_path == "workflows/deposit_anneal.yaml"


def test_registry_save_load(tmp_path: Path):
    reg = RecipeRegistry()
    reg.add(Recipe(name="a", maturity="prototype", methods=["m1"]))
    reg.add(Recipe(name="b", maturity="tested"))
    path = tmp_path / "recipes.toml"
    reg.save_toml(path)

    reloaded = RecipeRegistry.load_toml(path)
    assert reloaded.list_names() == ["a", "b"]
    assert reloaded.get("a").methods == ["m1"]
    assert "b" in reloaded


def test_missing_file_yields_empty_registry(tmp_path: Path):
    reg = RecipeRegistry.load_toml(tmp_path / "nope.toml")
    assert len(reg) == 0


def test_seeded_registry_loads():
    reg = RecipeRegistry.load_toml(Path("data/recipes.toml"))
    assert "dropcast_sweep" in reg
    assert set(reg.get("dropcast_sweep").methods) == {
        "startup_flush_full", "single_drop_simul"
    }
