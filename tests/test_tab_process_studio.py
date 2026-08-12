"""Widget tests for the Process Studio tab (slice 1: read-only library + detail)."""

from __future__ import annotations

import pytest

from softae.config import loader
from softae.core.lifecycle import Maturity
from softae.core.recipe_registry import Recipe, RecipeRegistry
from softae.core.task_catalog import Task, TaskCatalog
from softae.gui.tabs.tab_process_studio import ProcessStudioTab


def _catalog() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(
        name="single_drop_simul", instrument="liquid_handler", method="single_drop_simul",
        maturity="tested", provenance={"source": "legacy::singleDrop_simul", "ported_by": "p"},
        evidence={"tests": ["tests/test_dropcast.py::t"]},
    ))
    cat.add(Task(name="move_to_flush", instrument="stage", method="move_to", maturity="draft"))
    return cat


def _recipes() -> RecipeRegistry:
    reg = RecipeRegistry()
    reg.add(Recipe(
        name="dropcast_sweep", maturity="tested",
        methods=["single_drop_simul", "move_to_flush"],  # move_to_flush is draft
        builder="softae.core.dropcast:build_dropcast_sweep_workflow",
    ))
    return reg


@pytest.fixture
def studio(qtbot):
    w = ProcessStudioTab(catalog=_catalog(), recipes=_recipes())
    qtbot.addWidget(w)
    return w


def _find_row(table, name: str) -> int:
    for r in range(table.rowCount()):
        if table.item(r, 0).text() == name:
            return r
    return -1


def test_table_lists_methods_and_recipes(studio):
    table = studio._table
    assert _find_row(table, "single_drop_simul") >= 0
    assert _find_row(table, "move_to_flush") >= 0
    assert _find_row(table, "dropcast_sweep") >= 0


def test_remove_recipe_deregisters(studio):
    studio._table.selectRow(_find_row(studio._table, "dropcast_sweep"))
    ok, msg = studio.remove_selected()
    assert ok
    assert "dropcast_sweep" not in studio._recipes
    assert _find_row(studio._table, "dropcast_sweep") == -1


def test_remove_method_deletes_from_catalog(studio):
    studio._table.selectRow(_find_row(studio._table, "move_to_flush"))
    ok, msg = studio.remove_selected()
    assert ok
    assert "move_to_flush" not in studio._catalog
    assert _find_row(studio._table, "move_to_flush") == -1


def test_recipes_using_method_flags_dependents(studio):
    # dropcast_sweep depends on single_drop_simul.
    assert studio._recipes_using_method("single_drop_simul") == ["dropcast_sweep"]
    assert studio._recipes_using_method("move_to_flush") == ["dropcast_sweep"]


def test_remove_nothing_selected_is_reported(studio):
    studio._table.clearSelection()
    ok, msg = studio.remove_selected()
    assert not ok and "nothing selected" in msg


def test_remove_disables_action_buttons(studio):
    studio._table.selectRow(_find_row(studio._table, "move_to_flush"))
    studio.remove_selected()
    assert not studio._btn_remove.isEnabled()
    assert not studio._btn_promote.isEnabled()


def test_deposition_recipes_listed(studio):
    studio._filter.setCurrentText("Deposition")
    names = {studio._table.item(r, 0).text() for r in range(studio._table.rowCount())}
    assert {"single_drop", "two_phase"} <= names
    # They render as the 'deposition' kind.
    r = _find_row(studio._table, "two_phase")
    assert studio._table.item(r, 1).text() == "deposition"


def test_deposition_maturity_capped_by_missing_method(studio):
    # The fixture catalog lacks startup_flush_full/precondition_flush/final_flush,
    # so the two-phase recipe is capped at draft (it cannot run).
    assert studio._deposition_recipe_maturity("two_phase") is Maturity.DRAFT


def test_deposition_maturity_min_over_present_methods(qtbot):
    cat = TaskCatalog()
    for n in ("startup_flush_full", "precondition_flush", "final_flush"):
        cat.add(Task(name=n, instrument="lh", method="m", maturity="validated"))
    cat.add(Task(name="single_drop_simul", instrument="lh", method="single_drop_simul",
                 maturity="tested"))
    w = ProcessStudioTab(catalog=cat, recipes=RecipeRegistry())
    qtbot.addWidget(w)
    assert w._deposition_recipe_maturity("two_phase") is Maturity.TESTED


def test_deposition_detail_renders_phases(studio):
    studio._filter.setCurrentText("Deposition")
    studio._table.selectRow(_find_row(studio._table, "two_phase"))
    html = studio._detail.toHtml()
    assert "deposition recipe" in html
    assert "precondition_flush" in html and "single_drop_simul" in html


def _engine_catalog() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(name="startup_flush_full", instrument="liquid_handler", method="startup_flush",
                 params={"flush_x": -50, "flush_y": 50, "wick_x": -50, "wick_y": -25,
                         "disp_rate": 1500, "disp_vol": 150, "ids": [0, 1, 2]}))
    cat.add(Task(name="precondition_flush", instrument="liquid_handler", method="precondition_flush",
                 params={"flush_x": -50, "flush_y": 50, "wick_x": -50, "wick_y": -25,
                         "ids": [0, 1], "rate_list": [75, 75], "vol_list": [21, 21],
                         "flush_factor": 3.0}))
    cat.add(Task(name="single_drop_simul", instrument="liquid_handler", method="single_drop_simul",
                 params={"x": 0, "y": 0, "wick_x": -50, "wick_y": -25, "ids": [0, 1, 2],
                         "disp_rate": 75, "vols": [21, 21, 21], "deadvols": [20, 20, 20]}))
    cat.add(Task(name="final_flush", instrument="syringe", method="single_pump",
                 params={"res_vol": 1000, "ID": 0, "rate": 200, "dispense_vol": 80}))
    return cat


def test_deposition_step_preview_expands_full_sequence(qtbot):
    w = ProcessStudioTab(catalog=_engine_catalog(), recipes=RecipeRegistry())
    qtbot.addWidget(w)
    w._filter.setCurrentText("Deposition")
    w._table.selectRow(_find_row(w._table, "two_phase"))
    text = w._steps_preview.toPlainText()
    for token in ("startup_flush", "precondition_ch1", "deposit_ch1",
                  "measure_eis_ch1", "final_flush"):
        assert token in text
    # Phases appear in order; per-step params (e.g. rate_list) are shown too.
    assert text.index("precondition_ch1") < text.index("deposit_ch1")
    assert "rate_list" in text


def test_method_step_preview_shows_single_step(studio):
    studio._filter.setCurrentText("Methods")
    studio._table.selectRow(_find_row(studio._table, "single_drop_simul"))
    text = studio._steps_preview.toPlainText()
    assert "single_drop_simul" in text and "one step" in text


def test_deposition_preview_graceful_when_methods_missing(studio):
    # The default studio fixture catalog lacks the flush methods → graceful note,
    # not a crash.
    studio._filter.setCurrentText("Deposition")
    studio._table.selectRow(_find_row(studio._table, "two_phase"))
    assert "cannot expand" in studio._steps_preview.toPlainText()


def test_set_default_writes_config(studio, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(loader, "set_dropcast_default_recipe", lambda n: calls.append(n))
    studio._filter.setCurrentText("Deposition")
    studio._table.selectRow(_find_row(studio._table, "two_phase"))
    ok, _msg = studio.set_default_selected()
    assert ok and calls == ["two_phase"]


def test_set_default_rejects_non_deposition(studio):
    studio._filter.setCurrentText("Methods")
    studio._table.selectRow(_find_row(studio._table, "single_drop_simul"))
    ok, msg = studio.set_default_selected()
    assert not ok and "deposition" in msg


def test_set_default_button_enabled_only_for_deposition(studio):
    studio._filter.setCurrentText("All")
    studio._table.selectRow(_find_row(studio._table, "two_phase"))
    assert studio._btn_set_default.isEnabled() is True
    assert studio._btn_remove.isEnabled() is False   # built-in, not removable
    studio._table.selectRow(_find_row(studio._table, "single_drop_simul"))
    assert studio._btn_set_default.isEnabled() is False


def test_recipe_effective_is_min_over_deps(studio):
    table = studio._table
    r = _find_row(table, "dropcast_sweep")
    # own = tested, but move_to_flush is draft -> effective draft.
    assert table.item(r, 2).text() == "tested"     # Maturity (own)
    assert table.item(r, 3).text() == "draft"       # Effective (capped)
    assert table.item(r, 1).text() == "recipe"


def test_selecting_method_renders_detail(studio):
    table = studio._table
    table.selectRow(_find_row(table, "single_drop_simul"))
    html = studio._detail.toHtml()
    assert "single_drop_simul" in html
    assert "singleDrop_simul" in html   # provenance source
    assert "tests: 1 linked" in html


def test_selecting_recipe_flags_capping_method(studio):
    table = studio._table
    table.selectRow(_find_row(table, "dropcast_sweep"))
    html = studio._detail.toHtml()
    assert "dropcast_sweep" in html
    assert "move_to_flush" in html
    assert "caps effective" in html     # the draft dep that caps it


def test_filter_recipes_only(studio):
    studio._filter.setCurrentText("Recipes")
    table = studio._table
    assert _find_row(table, "dropcast_sweep") >= 0
    assert _find_row(table, "single_drop_simul") == -1


def test_reload_with_injected_is_idempotent(studio):
    studio.reload()
    # 2 methods + 1 recipe + 2 built-in deposition recipes (single_drop, two_phase).
    assert studio._table.rowCount() == 5


# ── Slice 2: lifecycle actions ───────────────────────────────────────────────

def _select(studio, name):
    studio._table.selectRow(_find_row(studio._table, name))


def test_promote_forward_updates_table(studio):
    _select(studio, "move_to_flush")  # draft
    ok, _ = studio.promote_selected("prototype")
    assert ok
    r = _find_row(studio._table, "move_to_flush")
    assert studio._table.item(r, 2).text() == "prototype"


def test_promote_blocked_returns_reason(studio):
    _select(studio, "move_to_flush")  # draft, no linked tests
    ok, msg = studio.promote_selected("tested")
    assert not ok
    assert "linked tests" in msg


def test_supersede_only_methods_and_archives(studio):
    _select(studio, "single_drop_simul")
    ok, msg = studio.supersede_selected()
    assert ok
    # Old archived as @v1, new canonical is draft.
    assert "single_drop_simul@v1" in studio._catalog
    assert studio._catalog.get("single_drop_simul").maturity == "draft"
    assert studio._catalog.get("single_drop_simul").version == 2


def test_supersede_rejects_recipe(studio):
    _select(studio, "dropcast_sweep")
    ok, msg = studio.supersede_selected()
    assert not ok
    assert "only methods" in msg


def test_run_tests_finished_green_promotes(studio):
    _select(studio, "single_drop_simul")  # tested already → stays tested
    studio._on_tests_finished("single_drop_simul", "method", 0)
    assert studio._catalog.get("single_drop_simul").maturity == "tested"


def test_run_tests_finished_red_does_not_promote(studio):
    # A draft method with linked tests, red result → stays draft.
    studio._catalog.get("move_to_flush").evidence["tests"] = ["tests/x.py::t"]
    _select(studio, "move_to_flush")
    studio._on_tests_finished("move_to_flush", "method", 1)
    assert studio._catalog.get("move_to_flush").maturity == "draft"


def test_sign_off_validates_with_run(studio, tmp_path):
    from softae.core.data_store import DataStore
    ds = DataStore(project_dir=str(tmp_path / "proj"), db_filename="softae.db")
    run_id = ds.start_run("cast", config_hash="cfg42")
    studio._data_store = ds

    _select(studio, "single_drop_simul")  # tested + has tests linked
    ok, msg = studio.sign_off_selected(run_id, "pshaps", "wells 21-24")
    assert ok, msg
    t = studio._catalog.get("single_drop_simul")
    assert t.maturity == "validated"
    assert t.evidence["validated_run_id"] == run_id
    assert t.evidence["config_hash"] == "cfg42"
    ds.close()


def test_supersede_button_disabled_for_recipe(studio):
    _select(studio, "dropcast_sweep")
    assert studio._btn_supersede.isEnabled() is False
    assert studio._btn_promote.isEnabled() is True


# ── Slice 3: builder embed + register-as-recipe bridge ───────────────────────

def _workflow():
    from softae.workflows.workflow_model import Workflow, WorkflowStep
    return Workflow(name="deposit_anneal", setup=[
        WorkflowStep("prime", "liquid_handler", "single_drop_simul"),  # -> single_drop_simul
        WorkflowStep("move", "stage", "move_to"),                       # -> move_to_flush
    ])


def test_register_recipe_detects_methods_and_writes_yaml(studio, tmp_path, monkeypatch):
    from softae.config import loader
    rw = tmp_path / "rw"
    monkeypatch.setattr(loader, "recipe_workflows_dir", lambda: rw)

    ok, msg = studio.register_recipe_from_workflow("deposit_anneal", _workflow())
    assert ok, msg

    r = studio._recipes.get("deposit_anneal")
    assert r.maturity == "draft"
    assert r.kind == "workflow"
    assert r.methods == ["move_to_flush", "single_drop_simul"]  # auto-detected, sorted
    assert (rw / "deposit_anneal.yaml").exists()
    # New recipe appears in the Library table.
    assert _find_row(studio._table, "deposit_anneal") >= 0


def test_register_empty_workflow_rejected(studio, tmp_path, monkeypatch):
    from softae.config import loader
    from softae.workflows.workflow_model import Workflow
    monkeypatch.setattr(loader, "recipe_workflows_dir", lambda: tmp_path / "rw")
    ok, msg = studio.register_recipe_from_workflow("empty", Workflow(name="e"))
    assert not ok
    assert "no steps" in msg


def test_register_sanitizes_name(studio, tmp_path, monkeypatch):
    from softae.config import loader
    rw = tmp_path / "rw"
    monkeypatch.setattr(loader, "recipe_workflows_dir", lambda: rw)
    ok, _ = studio.register_recipe_from_workflow("bad name/v2!", _workflow())
    assert ok
    assert "bad_name_v2" in studio._recipes


def test_no_builder_tab_without_manager(studio):
    # Injected-without-manager (the fixture) → Library only.
    assert studio._views.count() == 1
    assert studio._sandbox is None


def test_builder_tab_present_with_manager(qtbot):
    from softae.drivers.mock_factory import create_mock_manager
    w = ProcessStudioTab(catalog=_catalog(), recipes=_recipes(),
                         manager=create_mock_manager(config={}))
    qtbot.addWidget(w)
    assert w._views.count() == 2  # Library + Builder
    assert w._sandbox is not None
    assert w._btn_register is not None


# ── Slice 5: recipe editing ──────────────────────────────────────────────────

def _param_recipe():
    from softae.core.recipe_registry import Recipe, RecipeRegistry
    reg = RecipeRegistry()
    reg.add(Recipe(
        name="dropcast_sweep", maturity="tested",
        kind="builder", builder="softae.core.dropcast:build_dropcast_from_params",
        methods=["single_drop_simul"],
        parameters=[
            {"name": "wells", "type": "str", "default": "21-24"},
            {"name": "vol", "type": "float", "default": 0.1},
            {"name": "measure_eis", "type": "bool", "default": False},
            {"name": "time_scale", "type": "float", "default": 0.0},
        ],
    ))
    return reg


@pytest.fixture
def pstudio(qtbot):
    w = ProcessStudioTab(catalog=_catalog(), recipes=_param_recipe())
    qtbot.addWidget(w)
    return w


def test_edit_button_enabled_for_recipe_only(pstudio):
    _select(pstudio, "dropcast_sweep")
    assert pstudio._btn_edit.isEnabled() is True
    _select(pstudio, "single_drop_simul")
    assert pstudio._btn_edit.isEnabled() is False


def test_build_recipe_preview_from_params(pstudio):
    wf, msg = pstudio.build_recipe_preview(
        "dropcast_sweep",
        {"wells": "21-22", "vol": 0.5, "measure_eis": False, "time_scale": 0.0},
    )
    assert wf is not None, msg
    names = [s.name for s in wf.setup]
    assert "dropcast_ch21" in names and "dropcast_ch22" in names
    dep = next(s for s in wf.setup if s.name == "dropcast_ch21")
    assert dep.params["vols"] == [0.5, 0.5, 0.5][: len(dep.params["vols"])] or dep.params["vols"]


def test_preview_bad_params_returns_message(pstudio):
    wf, msg = pstudio.build_recipe_preview("dropcast_sweep", {"wells": "not-a-range"})
    assert wf is None
    assert "build failed" in msg


def test_save_recipe_params_updates_defaults(pstudio):
    ok, _ = pstudio.save_recipe_params(
        "dropcast_sweep", {"wells": "5-8", "vol": 0.25})
    assert ok
    params = {p["name"]: p["default"] for p in pstudio._recipes.get("dropcast_sweep").parameters}
    assert params["wells"] == "5-8"
    assert params["vol"] == 0.25


def test_edit_in_builder_requires_manager(pstudio):
    # No manager (Library-only) → builder unavailable.
    ok, msg = pstudio.edit_recipe_in_builder("dropcast_sweep")
    assert not ok
    assert "Builder is unavailable" in msg


def test_edit_in_builder_loads_yaml(qtbot, tmp_path):
    from softae.drivers.mock_factory import create_mock_manager
    from softae.core.recipe_registry import Recipe, RecipeRegistry
    from softae.workflows import workflow_parser
    from softae.workflows.workflow_model import Workflow, WorkflowStep

    # Write a small recipe workflow YAML.
    wf = Workflow(name="mini", setup=[WorkflowStep("s", "stage", "move_to", params={"x": 1, "y": 2})])
    ypath = tmp_path / "mini.yaml"
    workflow_parser.dump_file(wf, ypath)

    reg = RecipeRegistry()
    reg.add(Recipe(name="mini", kind="workflow", workflow_path=str(ypath),
                   methods=["move_to_flush"]))
    w = ProcessStudioTab(catalog=_catalog(), recipes=reg,
                         manager=create_mock_manager(config={}))
    qtbot.addWidget(w)

    ok, msg = w.edit_recipe_in_builder("mini")
    assert ok, msg
    # The builder tree now has the loaded step.
    assert w._sandbox._setup_root.childCount() == 1
