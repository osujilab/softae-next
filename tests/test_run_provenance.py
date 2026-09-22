"""``provenance.json`` — the record of what a campaign run compiled.

Covers the promised key set, the JSON round trip, byte-identical repeat writes,
the recorded catalog digest, and the contract that an I/O failure returns
``None`` while a bad argument still raises.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import pytest

from softae.core.catalog_digest import NONFINITE_KEY, catalog_digest
from softae.core.formulation import Chemical, ChemicalCatalog
from softae.core.run_provenance import (
    PROVENANCE_FILENAME,
    SCHEMA_VERSION,
    build_run_provenance,
    serialise_workflow,
    unrecorded_step_fields,
    write_run_provenance,
)
from softae.core.task_catalog import Task, TaskCatalog
from softae.workflows.workflow_model import Workflow, WorkflowStep

PROMISED_KEYS = {
    "schema_version", "run_id", "spec", "workflow",
    "catalog_digest", "catalog_sub_digests", "config_hash",
    "code_revision", "notes",
}


# ── In-memory inputs ─────────────────────────────────────────────────────────

def _spec():
    from softae.core.autonomous_wiring import CampaignSpec

    return CampaignSpec(
        name="provenance_probe",
        channels=(1, 2),
        parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
    )


def _tasks() -> TaskCatalog:
    cat = TaskCatalog()
    cat.add(Task(name="flush", instrument="pump", method="dispense",
                 params={"volume_uL": 5.0}))
    return cat


def _workflow() -> Workflow:
    setup = WorkflowStep(name="flush", instrument="pump", method="dispense",
                         params={"volume_uL": 5.0, "b": 1, "a": 2},
                         timeout_s=30.0, tags={"phase": "PRIME"})
    loop = WorkflowStep(name="cast", instrument="pump", method="dispense",
                        params={"volume_uL": 12.5}, depends_on=["flush"],
                        retry=1, tags={"phase": "CAST", "channel": "1"})
    teardown = WorkflowStep(name="park", instrument="stage", method="park")
    return Workflow(name="trial", description="one trial", setup=[setup],
                    loop_steps=[loop], teardown=[teardown], iterations=2,
                    iterate_over="channels",
                    metadata={"derived": object()})


def _write(run_dir: Path, **overrides) -> Path | None:
    kwargs = dict(spec=_spec(), workflow=_workflow(), tasks=_tasks())
    kwargs.update(overrides)
    return write_run_provenance(run_dir, **kwargs)


# ── Content ──────────────────────────────────────────────────────────────────

def test_build_run_provenance_document_has_every_promised_key() -> None:
    """A missing key is a silently incomplete record, so the set is asserted."""
    document = build_run_provenance(
        run_id="r1", spec=_spec(), workflow=_workflow(), tasks=_tasks())
    assert set(document) == PROMISED_KEYS


def test_build_run_provenance_records_the_schema_version() -> None:
    """A reader needs to know which rules produced the document it is holding."""
    document = build_run_provenance(
        run_id="r1", spec=_spec(), workflow=_workflow(), tasks=_tasks())
    assert document["schema_version"] == SCHEMA_VERSION


def test_write_run_provenance_written_file_parses_with_every_key(tmp_path) -> None:
    """What lands on disk carries the same key set the builder promised."""
    path = _write(tmp_path / "run")
    assert path is not None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == PROMISED_KEYS


def test_write_run_provenance_lands_beside_the_event_stream(tmp_path) -> None:
    """The file is named and sited where a reader of the run directory finds it."""
    run_dir = tmp_path / "20260921T000000Z_probe"
    path = _write(run_dir)
    assert path == run_dir / PROVENANCE_FILENAME


def test_write_run_provenance_run_id_defaults_to_directory_name(tmp_path) -> None:
    """A run's identity is its directory name; the caller need not repeat it."""
    run_dir = tmp_path / "20260921T000000Z_probe"
    path = _write(run_dir)
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == run_dir.name


def test_write_run_provenance_records_the_catalog_digest(tmp_path) -> None:
    """The recorded digest is the one the digest module computes for the inputs."""
    tasks = _tasks()
    path = _write(tmp_path / "run", tasks=tasks)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["catalog_digest"] == catalog_digest(tasks, None, None)


def test_write_run_provenance_catalog_change_moves_recorded_digest(tmp_path) -> None:
    """Two runs against different catalogs must not record the same digest."""
    first = _write(tmp_path / "a", tasks=_tasks())
    chems = ChemicalCatalog()
    chems.add(Chemical(name="water"))
    second = _write(tmp_path / "b", tasks=_tasks(), chemicals=chems)
    assert (json.loads(first.read_text(encoding="utf-8"))["catalog_digest"]
            != json.loads(second.read_text(encoding="utf-8"))["catalog_digest"])


def test_write_run_provenance_code_revision_is_null_with_a_reason(tmp_path) -> None:
    """An unrecorded revision says so in words; a bare null reads as an oversight."""
    loaded = json.loads(_write(tmp_path / "run").read_text(encoding="utf-8"))
    assert loaded["code_revision"] is None
    assert loaded["notes"]["code_revision"]


def test_write_run_provenance_file_ends_with_a_newline(tmp_path) -> None:
    """A trailing newline keeps the file well-formed for line-oriented tools."""
    assert _write(tmp_path / "run").read_text(encoding="utf-8").endswith("}\n")


# ── Workflow serialisation ───────────────────────────────────────────────────

def test_serialise_workflow_keeps_step_lists_separate() -> None:
    """A step moving between setup and teardown changes when it runs."""
    encoded = serialise_workflow(_workflow())
    assert [s["name"] for s in encoded["setup"]] == ["flush"]
    assert [s["name"] for s in encoded["loop_steps"]] == ["cast"]
    assert [s["name"] for s in encoded["teardown"]] == ["park"]


def test_serialise_workflow_records_dependencies_and_retry() -> None:
    """Both change what the executor does, so both belong in the record."""
    step = serialise_workflow(_workflow())["loop_steps"][0]
    assert step["depends_on"] == ["flush"]
    assert step["retry"] == 1


def test_serialise_step_covers_every_workflow_step_field() -> None:
    """A field added to ``WorkflowStep`` is recorded or explicitly excluded, so
    it cannot enter one enumeration of the step fields and not the other.
    """
    names = [f.name for f in dataclasses.fields(WorkflowStep)]
    assert unrecorded_step_fields(names) == set()


def test_unrecorded_step_fields_names_a_field_that_is_neither() -> None:
    """The control: an unlisted field is reported, so the guard above can fail."""
    assert unrecorded_step_fields(["name", "cooldown_s"]) == {"cooldown_s"}


def test_serialise_workflow_omits_derived_metadata() -> None:
    """Metadata restates facts the steps already carry, so it is left out."""
    assert "metadata" not in serialise_workflow(_workflow())


def test_serialise_workflow_nonfinite_param_becomes_tagged(tmp_path) -> None:
    """A non-finite timeout must be recorded, not crash the write."""
    wf = _workflow()
    wf.setup = [wf.setup[0].with_params(volume_uL=math.inf)]
    loaded = json.loads(
        _write(tmp_path / "run", workflow=wf).read_text(encoding="utf-8"))
    assert (loaded["workflow"]["setup"][0]["params"]["volume_uL"]
            == {NONFINITE_KEY: "inf"})


# ── Determinism ──────────────────────────────────────────────────────────────

def test_write_run_provenance_repeat_write_is_byte_identical(tmp_path) -> None:
    """Same inputs, same bytes — otherwise a diff cannot mean anything."""
    run_dir = tmp_path / "run"
    first = _write(run_dir).read_bytes()
    second = _write(run_dir).read_bytes()
    assert first == second


def test_write_run_provenance_leaves_no_temp_file(tmp_path) -> None:
    """The atomic write must not litter the run directory it writes into."""
    run_dir = tmp_path / "run"
    _write(run_dir)
    assert [p.name for p in run_dir.iterdir()] == [PROVENANCE_FILENAME]


# ── The two failure arms ─────────────────────────────────────────────────────

def test_write_run_provenance_unwritable_directory_returns_none(tmp_path) -> None:
    """An I/O problem must cost the record, never the running campaign."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert _write(blocker / "run") is None


def test_write_run_provenance_good_directory_returns_a_path(tmp_path) -> None:
    """The control for the arm above: the same call succeeds where it can."""
    assert isinstance(_write(tmp_path / "run"), Path)


def test_write_run_provenance_replace_failure_returns_none(tmp_path, monkeypatch) -> None:
    """A failure at the final rename is still I/O, so it is reported the same way."""
    def boom(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", boom)
    assert _write(tmp_path / "run") is None


def test_write_run_provenance_replace_failure_leaves_no_temp_file(
    tmp_path, monkeypatch
) -> None:
    """A half-written record must not survive as a file a reader could trust."""
    def boom(src, dst):
        raise OSError("replace refused")

    run_dir = tmp_path / "run"
    monkeypatch.setattr(os, "replace", boom)
    _write(run_dir)
    assert list(run_dir.iterdir()) == []


def test_write_run_provenance_missing_workflow_raises(tmp_path) -> None:
    """A wiring fault must not produce a record that looks complete."""
    with pytest.raises(AttributeError):
        _write(tmp_path / "run", workflow=None)
