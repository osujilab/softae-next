"""Record what a campaign run actually compiled, beside its event stream.

A run directory holds the spec's name, its events and its results, but nothing
that says which *compiled* workflow the executor was handed or which catalogs it
was resolved against. Reproducing a run therefore depends on the working tree
happening not to have moved.

``provenance.json`` closes that: schema version, the canonical spec dict, one
representative compiled workflow, the catalog digests, the config hash and the
code revision — written once per run, atomically, next to ``events.jsonl``.

Writing it must never cost a run: an I/O failure is logged and reported by a
``None`` return, while a bad argument still raises so a wiring mistake is not
swallowed. Callers that cannot tolerate either wrap the call, as the campaign's
other best-effort records do.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import structlog

import softae
from softae.core.catalog_digest import (
    canonical,
    catalog_digest,
    catalog_sub_digests,
)

logger = structlog.get_logger(__name__)

#: Bumped whenever the document's key set or meaning changes.
SCHEMA_VERSION = 1

#: Written into the run directory, beside ``events.jsonl``.
PROVENANCE_FILENAME = "provenance.json"

#: Said once, here, rather than left as a bare ``null`` a reader must guess at.
NO_CODE_REVISION_REASON = (
    "this package records no git revision; an installed copy has no work tree "
    "to read one from, and shelling out to git on a run's startup path was "
    "declined. The package version is recorded instead."
)

#: Per compiled step. ``depends_on`` and ``retry`` are in because both change
#: what the executor does with the step, not merely how it is labelled.
#: ``workflows.workflow_parser.step_to_dict`` hand-enumerates the same fields
#: for the YAML round trip; the two lists are siblings and drift apart silently,
#: which is what :func:`unrecorded_step_fields` exists to make visible here.
STEP_FIELDS = ("name", "instrument", "method", "params", "timeout_s", "retry",
               "depends_on", "tags")

#: Step fields deliberately left out of the record. Empty today: every
#: ``WorkflowStep`` field changes what the executor does, so none of them is
#: derived bookkeeping the way ``Workflow.metadata`` is. A field added later
#: belongs in one list or the other, and the test that pairs the two with
#: ``dataclasses.fields(WorkflowStep)`` fails until somebody chooses.
STEP_FIELDS_EXCLUDED: frozenset[str] = frozenset()

#: The three step lists a workflow carries. Kept apart because a step moving
#: between them changes when it runs.
STEP_LISTS = ("setup", "loop_steps", "teardown")


def unrecorded_step_fields(field_names: Iterable[str]) -> set[str]:
    """Names that are neither recorded nor deliberately excluded.

    Pure, so a test can hand it a field list and prove the guard can fail.
    """
    return set(field_names) - set(STEP_FIELDS) - STEP_FIELDS_EXCLUDED


def serialise_step(step: Any) -> dict[str, Any]:
    """One compiled step as a canonical mapping, in :data:`STEP_FIELDS` order."""
    return {
        "name": str(step.name),
        "instrument": str(step.instrument),
        "method": str(step.method),
        "params": canonical(dict(step.params)),
        "timeout_s": canonical(step.timeout_s),
        "retry": int(step.retry),
        "depends_on": canonical(list(step.depends_on)),
        "tags": canonical(dict(step.tags)),
    }


def serialise_workflow(workflow: Any) -> dict[str, Any]:
    """A compiled workflow as a deterministic mapping of its executable content.

    ``metadata`` is left out: it holds bookkeeping derived from the steps
    recorded here, so including it would record the same facts twice.
    """
    return {
        "name": str(workflow.name),
        "description": str(workflow.description),
        "iterate_over": canonical(workflow.iterate_over),
        "iterations": int(workflow.iterations),
        "total_steps": int(workflow.total_steps),
        **{key: [serialise_step(s) for s in getattr(workflow, key)]
           for key in STEP_LISTS},
    }


def _config_hash() -> tuple[str | None, str | None]:
    """``(hash, reason_it_is_absent)`` — exactly one of the two is ``None``."""
    from softae.config import loader

    try:
        return loader.config_hash(), None
    except Exception as exc:  # no config file, or an unreadable one
        return None, f"{type(exc).__name__}: {exc}"


def _spec_dict(spec: Any) -> Any:
    """The spec's canonical file-representable dict, via the one spec encoder."""
    from softae.core.campaign_spec_io import spec_to_dict

    return canonical(spec_to_dict(spec))


def build_run_provenance(
    *,
    run_id: str,
    spec: Any,
    workflow: Any,
    tasks: Any,
    chemicals: Any = None,
    solutions: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance document, as a plain dict, without touching the disk.

    Separate from the write so a caller can inspect or log the document, and so
    the tests can prove the content and the I/O contract independently.
    """
    config_hash, config_hash_reason = _config_hash()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id),
        "spec": _spec_dict(spec),
        "workflow": serialise_workflow(workflow),
        "catalog_digest": catalog_digest(tasks, chemicals, solutions),
        "catalog_sub_digests": catalog_sub_digests(tasks, chemicals, solutions),
        "config_hash": config_hash,
        "code_revision": None,
        "notes": {
            "config_hash": config_hash_reason,
            "code_revision": NO_CODE_REVISION_REASON,
            "package_version": softae.__version__,
        },
    }
    if extra:
        document["extra"] = canonical(extra)
    return document


def dumps(document: dict[str, Any]) -> str:
    """The one text form: sorted keys, indented, UTF-8-safe, trailing newline."""
    return json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* via a temp file in the same directory, then replace."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".provenance-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_run_provenance(
    run_dir: str | Path,
    *,
    spec: Any,
    workflow: Any,
    tasks: Any,
    chemicals: Any = None,
    solutions: Any = None,
    run_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write ``provenance.json`` into *run_dir*; return its path, or ``None``.

    ``None`` means the document could not reach the disk (the reason is logged).
    An argument the document cannot be built from still raises: that is a wiring
    fault, and a run started on the wrong spec should not look recorded.
    """
    directory = Path(run_dir)
    document = build_run_provenance(
        run_id=run_id if run_id is not None else directory.name,
        spec=spec, workflow=workflow,
        tasks=tasks, chemicals=chemicals, solutions=solutions,
        extra=extra,
    )
    text = dumps(document)
    path = directory / PROVENANCE_FILENAME
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, text)
    except OSError as exc:
        logger.warning("run_provenance_write_failed",
                       path=str(path), error=str(exc))
        return None
    logger.info("run_provenance_written", path=str(path),
                catalog_digest=document["catalog_digest"])
    return path
