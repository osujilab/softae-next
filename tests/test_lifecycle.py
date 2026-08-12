"""Tests for the method-maturity lifecycle (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.data_store import DataStore
from softae.core.lifecycle import (
    GateError,
    Maturity,
    can_promote,
    effective_maturity,
    link_tests,
    maturity_warnings,
    method_maturity,
    promote,
    sign_off,
    status,
    supersede,
    version_chain,
    workflow_method_maturities,
)
from softae.core.recipe_registry import Recipe, RecipeRegistry
from softae.core.task_catalog import Task, TaskCatalog
from softae.workflows.workflow_model import Workflow, WorkflowStep


def _catalog() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(name="m", instrument="liquid_handler", method="single_drop_simul"))
    return cat


# ── Maturity enum ────────────────────────────────────────────────────────────

def test_maturity_ordering():
    assert Maturity.DRAFT < Maturity.PROTOTYPE < Maturity.TESTED < Maturity.VALIDATED


def test_maturity_parse_variants():
    assert Maturity.parse("tested") is Maturity.TESTED
    assert Maturity.parse(3) is Maturity.VALIDATED
    assert Maturity.parse(Maturity.DRAFT) is Maturity.DRAFT
    with pytest.raises(ValueError):
        Maturity.parse("nonsense")


# ── Task round-trip (backward compatible) ────────────────────────────────────

def test_lifecycle_fields_round_trip():
    t = Task(
        name="m", instrument="i", method="mm",
        maturity="tested", version=2,
        provenance={"source": "legacy::x", "ported_by": "p"},
        evidence={"tests": ["a::b"]},
    )
    rebuilt = Task.from_dict("m", t.to_dict())
    assert rebuilt.maturity == "tested"
    assert rebuilt.version == 2
    assert rebuilt.provenance["source"] == "legacy::x"
    assert rebuilt.evidence["tests"] == ["a::b"]


def test_pre_lifecycle_task_defaults_to_draft():
    # A catalog entry with no lifecycle fields loads unchanged.
    t = Task.from_dict("old", {"instrument": "i", "method": "mm"})
    assert t.maturity == "draft"
    assert t.version == 1
    assert t.provenance == {} and t.evidence == {}


def test_to_dict_omits_default_lifecycle():
    d = Task(name="m", instrument="i", method="mm").to_dict()
    assert "maturity" not in d and "version" not in d
    assert "provenance" not in d and "evidence" not in d


# ── Gates ────────────────────────────────────────────────────────────────────

def test_promote_to_prototype_ok():
    cat = _catalog()
    ok, reasons = can_promote("m", "prototype", cat)
    assert ok, reasons


def test_tested_requires_linked_tests():
    cat = _catalog()
    promote("m", "prototype", cat)
    ok, reasons = can_promote("m", "tested", cat)
    assert not ok
    assert any("linked tests" in r for r in reasons)
    link_tests("m", ["tests/x.py::t"], cat)
    ok, _ = can_promote("m", "tested", cat)
    assert ok


def test_validated_requires_run_and_operator():
    cat = _catalog()
    link_tests("m", ["tests/x.py::t"], cat)
    promote("m", "tested", cat)
    ok, reasons = can_promote("m", "validated", cat)
    assert not ok
    assert any("validated_run_id" in r for r in reasons)


def test_backward_promotion_refused():
    cat = _catalog()
    promote("m", "prototype", cat)
    ok, reasons = can_promote("m", "draft", cat)
    assert not ok


def test_promote_raises_gate_error_with_reasons():
    cat = _catalog()
    with pytest.raises(GateError) as exc:
        promote("m", "tested", cat)  # skips prototype fine, but no tests
    assert exc.value.reasons


def test_promote_straight_to_validated_still_needs_tests():
    cat = _catalog()
    cat.get("m").evidence.update(validated_run_id="r1", validated_by="p")
    # Crossing the 'tested' gate still requires linked tests.
    ok, reasons = can_promote("m", "validated", cat)
    assert not ok
    assert any("linked tests" in r for r in reasons)


# ── Sign-off ─────────────────────────────────────────────────────────────────

def test_sign_off_records_evidence_and_validates(tmp_path: Path):
    cat = _catalog()
    link_tests("m", ["tests/x.py::t"], cat)
    promote("m", "tested", cat)

    ds = DataStore(tmp_path / "proj")
    run_id = ds.start_run("cast", config_hash="deadbeef")

    task = sign_off("m", run_id=run_id, by="pshaps", catalog=cat,
                    notes="wells 21-24", data_store=ds)

    assert task.maturity == "validated"
    assert method_maturity("m", cat) is Maturity.VALIDATED
    ev = task.evidence
    assert ev["validated_run_id"] == run_id
    assert ev["validated_by"] == "pshaps"
    assert ev["config_hash"] == "deadbeef"  # pulled from the DataStore run
    assert ev["notes"] == "wells 21-24"
    ds.close()


def test_sign_off_verifies_run_exists(tmp_path: Path):
    cat = _catalog()
    link_tests("m", ["tests/x.py::t"], cat)
    promote("m", "tested", cat)
    ds = DataStore(tmp_path / "proj2")
    with pytest.raises(GateError):
        sign_off("m", run_id="nonexistent_run", by="p", catalog=cat, data_store=ds)
    ds.close()


def test_status_render_smoke():
    cat = _catalog()
    cat.get("m").provenance = {"source": "legacy::x", "ported_by": "p", "ported_on": "2026-07-10"}
    link_tests("m", ["t::a"], cat)
    text = status("m", cat).render()
    assert "m" in text and "draft" in text and "provenance" in text


# ── Phase 2: recipes, effective maturity, warn-scan ────────────────────────

def _two_method_catalog() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(name="mature", instrument="liquid_handler", method="single_drop_simul",
                 maturity="validated"))
    cat.add(Task(name="green", instrument="stage", method="move_to", maturity="draft"))
    return cat


def test_effective_maturity_of_method_is_its_own():
    cat = _two_method_catalog()
    assert effective_maturity("mature", cat) is Maturity.VALIDATED


def test_effective_maturity_of_recipe_capped_by_least_mature_method():
    cat = _two_method_catalog()
    reg = RecipeRegistry()
    reg.add(Recipe(name="proto", maturity="validated", methods=["mature", "green"]))
    # own=validated, deps=[validated, draft] -> min = draft
    assert effective_maturity("proto", cat, reg) is Maturity.DRAFT


def test_effective_maturity_recipe_no_deps_is_own():
    cat = _two_method_catalog()
    reg = RecipeRegistry()
    reg.add(Recipe(name="p", maturity="tested", methods=[]))
    assert effective_maturity("p", cat, reg) is Maturity.TESTED


def test_workflow_maturity_scan_maps_unique_methods():
    cat = _two_method_catalog()
    wf = Workflow(name="w", setup=[
        WorkflowStep("s1", "liquid_handler", "single_drop_simul"),  # -> "mature"
        WorkflowStep("s2", "stage", "move_to"),                      # -> "green"
        WorkflowStep("s3", "pico1", "sendscript_getdata"),           # no catalog match
    ])
    mats = workflow_method_maturities(wf, cat)
    assert mats == {"mature": Maturity.VALIDATED, "green": Maturity.DRAFT}


def test_maturity_warnings_flags_below_expected():
    cat = _two_method_catalog()
    wf = Workflow(name="w", setup=[
        WorkflowStep("s1", "liquid_handler", "single_drop_simul"),
        WorkflowStep("s2", "stage", "move_to"),
    ])
    warns = maturity_warnings(wf, cat, expected="validated")
    assert len(warns) == 1
    assert warns[0]["method"] == "green"
    assert warns[0]["maturity"] == "draft"


def test_ambiguous_instrument_method_is_untracked():
    cat = TaskCatalog()
    cat.add(Task(name="a", instrument="syringe", method="single_pump"))
    cat.add(Task(name="b", instrument="syringe", method="single_pump"))
    wf = Workflow(name="w", setup=[WorkflowStep("s", "syringe", "single_pump")])
    # Two tasks share (syringe, single_pump) -> ambiguous -> skipped.
    assert workflow_method_maturities(wf, cat) == {}


# ── Phase 4: versioning + rollback ───────────────────────────────────────────

def test_supersede_archives_old_and_resets_new():
    cat = TaskCatalog()
    cat.add(Task(name="m", instrument="i", method="mm", maturity="validated",
                 provenance={"source": "legacy::x"}))

    new = Task(name="ignored", instrument="i", method="mm", params={"rate": 999})
    archived, result = supersede("m", new, cat)

    assert archived == "m@v1"
    # Old version retained, unchanged maturity, back-reference set.
    old = cat.get("m@v1")
    assert old.maturity == "validated"
    assert old.provenance["superseded_by"] == "m"
    # New version is canonical, bumped, reset to draft, forward-reference set.
    cur = cat.get("m")
    assert cur.version == 2
    assert cur.maturity == "draft"
    assert cur.provenance["supersedes"] == "m@v1"
    assert cur.params["rate"] == 999


def test_superseded_binding_still_resolves():
    # A role bound to the archived key must keep resolving (harmless rollback).
    cat = TaskCatalog()
    cat.add(Task(name="m", instrument="i", method="mm", version=1))
    supersede("m", Task(name="m", instrument="i", method="mm2"), cat)
    assert "m@v1" in cat  # old binding target survives
    assert cat.get("m@v1").method == "mm"
    assert cat.get("m").method == "mm2"


def test_version_chain_lists_all_versions():
    cat = TaskCatalog()
    cat.add(Task(name="m", instrument="i", method="mm"))
    supersede("m", Task(name="m", instrument="i", method="mm2"), cat)
    supersede("m", Task(name="m", instrument="i", method="mm3"), cat)
    assert version_chain("m", cat) == ["m", "m@v1", "m@v2"]
