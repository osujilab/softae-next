"""Workflow parser — YAML / JSON → :class:`Workflow` objects.

Responsibilities:

1. Read a ``.yaml`` or ``.json`` workflow file.
2. Validate the top-level schema (``name``, ``setup``, ``loop``, ``teardown``).
3. Interpolate ``$variable`` references inside step ``params`` using the
   workflow's ``variables`` section.
4. Determine loop iteration count from the variable referred to by
   ``iterate_over`` (e.g. if ``iterate_over: channels`` and
   ``variables.vol_master`` is a 32-element list → 32 iterations).
5. Return a :class:`Workflow` dataclass.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import structlog

from softae.errors import ValidationError_
from softae.workflows.workflow_model import Workflow, WorkflowStep

try:
    import yaml  # type: ignore[import-untyped]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Regex for ``$variable`` references in param values.
_VAR_RE = re.compile(r"\$(\w+)")


def _loop_block(data: dict[str, Any]) -> dict[str, Any]:
    """Return the middle-phase block, accepting ``loop:`` or ``execution:``.

    The Process Configuration tab presents this phase as "Execution"; recipes
    may name it either way on load.  Canonical serialization always emits
    ``loop:`` so files stay compatible with ``softae-run`` and the existing
    templates.
    """
    block = data.get("loop")
    if block is None:
        block = data.get("execution")
    return block if isinstance(block, dict) else {}


# ── Public API ──────────────────────────────────────────────────────────────


def parse_file(path: str | Path) -> Workflow:
    """Parse a YAML or JSON workflow file into a :class:`Workflow`.

    Parameters
    ----------
    path : str or Path
        Path to the workflow definition file.

    Returns
    -------
    Workflow

    Raises
    ------
    ValidationError_
        If the file is missing required fields or has an invalid structure.
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workflow file not found: {p}")

    raw = p.read_text(encoding="utf-8")

    if p.suffix in {".yaml", ".yml"}:
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to parse .yaml workflow files. "
                "Install it with: pip install pyyaml"
            )
        data = yaml.safe_load(raw)
    elif p.suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValidationError_(f"Unsupported workflow file extension: {p.suffix}")

    if not isinstance(data, dict):
        raise ValidationError_(f"Workflow file must be a YAML/JSON mapping, got {type(data).__name__}")

    logger.info("parsing_workflow", path=str(p), name=data.get("name"))
    return parse_dict(data)


def parse_dict(data: dict[str, Any]) -> Workflow:
    """Parse a workflow from an already-loaded dictionary.

    Parameters
    ----------
    data : dict
        Workflow definition (same schema as the YAML file).

    Returns
    -------
    Workflow
    """
    _validate_schema(data)

    variables = data.get("variables", {})
    loop_block = _loop_block(data)
    iterate_over = loop_block.get("iterate_over")
    iterations = _resolve_iterations(iterate_over, variables)

    setup = _parse_steps(data.get("setup", []), variables, section="setup")
    loop_steps = _parse_steps(
        loop_block.get("steps", []),
        variables,
        section="loop",
    )
    teardown = _parse_steps(data.get("teardown", []), variables, section="teardown")

    wf = Workflow(
        name=data["name"],
        description=data.get("description", ""),
        variables=variables,
        setup=setup,
        loop_steps=loop_steps,
        teardown=teardown,
        iterate_over=iterate_over,
        iterations=iterations,
        metadata=data.get("metadata", {}),
    )

    _validate_dependencies(wf)
    return wf


# ── Serialization (Workflow → dict / file) ──────────────────────────────────
# Canonical inverse of parse_dict.  Kept here so the parse and dump schemas live
# in one module (single source of truth) and recipes round-trip through
# parse_file / the ``softae-run`` CLI.


def step_to_dict(step: WorkflowStep) -> dict[str, Any]:
    """Serialize a :class:`WorkflowStep`, omitting empty/default fields."""
    d: dict[str, Any] = {
        "name": step.name,
        "instrument": step.instrument,
        "method": step.method,
    }
    if step.params:
        d["params"] = dict(step.params)
    if step.depends_on:
        d["depends_on"] = list(step.depends_on)
    if step.timeout_s is not None:
        d["timeout_s"] = step.timeout_s
    if step.retry:
        d["retry"] = step.retry
    if step.tags:
        d["tags"] = dict(step.tags)
    return d


def workflow_to_dict(wf: Workflow) -> dict[str, Any]:
    """Serialize a :class:`Workflow` to the canonical nested schema.

    The middle phase is emitted under the canonical ``loop:`` key with a nested
    ``steps`` list.  Because :func:`parse_dict` derives the iteration count from
    ``variables``/``iterate_over`` (not from a bare ``iterations`` key), the
    count is encoded there so it round-trips exactly:

    * an explicit ``iterate_over`` is emitted with its backing variable intact;
    * otherwise a repeat count > 1 is encoded via a synthesized
      ``iterate_over="_iterations"`` and ``variables["_iterations"] = N``
      (``_resolve_iterations`` returns that int verbatim);
    * a single run (``iterations == 1``) omits ``iterate_over`` entirely.
    """
    variables = dict(wf.variables)
    if wf.iterate_over:
        iterate_over: str | None = wf.iterate_over
    elif wf.loop_steps and wf.iterations > 1:
        iterate_over = "_iterations"
        variables["_iterations"] = wf.iterations
    else:
        iterate_over = None

    loop_block: dict[str, Any] = {}
    if iterate_over is not None:
        loop_block["iterate_over"] = iterate_over
    loop_block["steps"] = [step_to_dict(s) for s in wf.loop_steps]

    data: dict[str, Any] = {"name": wf.name}
    if wf.description:
        data["description"] = wf.description
    if variables:
        data["variables"] = variables
    data["setup"] = [step_to_dict(s) for s in wf.setup]
    data["loop"] = loop_block
    data["teardown"] = [step_to_dict(s) for s in wf.teardown]
    if wf.metadata:
        data["metadata"] = dict(wf.metadata)
    return data


def dump_file(wf: Workflow, path: str | Path) -> None:
    """Write ``wf`` to ``path`` as YAML (default) or JSON (``.json`` suffix)."""
    path = Path(path)
    data = workflow_to_dict(wf)
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to write YAML workflows")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# ── Internal Helpers ────────────────────────────────────────────────────────


def _validate_schema(data: dict[str, Any]) -> None:
    """Check that required top-level keys exist."""
    if "name" not in data:
        raise ValidationError_("Workflow definition must include a 'name' field")

    # At least one section must have steps
    has_setup = bool(data.get("setup"))
    has_loop = bool(_loop_block(data).get("steps"))
    has_teardown = bool(data.get("teardown"))

    if not (has_setup or has_loop or has_teardown):
        raise ValidationError_(
            "Workflow must have at least one non-empty section: setup, loop, or teardown"
        )


def _validate_dependencies(wf: Workflow) -> None:
    """Validate that ``depends_on`` references are resolvable and acyclic."""
    setup_names = {s.name for s in wf.setup}
    teardown_names = {s.name for s in wf.teardown}
    loop_template_names = {s.name for s in wf.loop_steps}

    # Check setup deps
    for step in wf.setup:
        for dep in step.depends_on:
            if dep not in setup_names:
                raise ValidationError_(
                    f"Setup step '{step.name}' depends on '{dep}', "
                    f"which is not a setup step"
                )

    # Check teardown deps
    for step in wf.teardown:
        for dep in step.depends_on:
            if dep not in teardown_names:
                raise ValidationError_(
                    f"Teardown step '{step.name}' depends on '{dep}', "
                    f"which is not a teardown step"
                )

    # Check loop step template deps (within same iteration)
    all_valid = setup_names | loop_template_names
    for step in wf.loop_steps:
        for dep in step.depends_on:
            if dep not in all_valid:
                raise ValidationError_(
                    f"Loop step '{step.name}' depends on '{dep}', "
                    f"which is not a setup or loop step"
                )

    # Cycle detection (DFS on template names)
    def _check_cycles(steps: list[WorkflowStep], phase: str) -> None:
        adj: dict[str, list[str]] = {s.name: list(s.depends_on) for s in steps}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in adj}

        def dfs(node: str) -> str | None:
            color[node] = GRAY
            for neighbour in adj.get(node, []):
                if neighbour not in color:
                    continue  # cross-phase dep, already validated
                if color[neighbour] == GRAY:
                    return f"{neighbour} -> {node}"
                if color[neighbour] == WHITE:
                    result = dfs(neighbour)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for name in adj:
            if color[name] == WHITE:
                cycle = dfs(name)
                if cycle:
                    raise ValidationError_(
                        f"Dependency cycle in {phase}: {cycle}"
                    )

    _check_cycles(wf.setup, "setup")
    _check_cycles(wf.loop_steps, "loop")
    _check_cycles(wf.teardown, "teardown")


def _parse_steps(
    raw_steps: list[dict[str, Any]],
    variables: dict[str, Any],
    *,
    section: str,
) -> list[WorkflowStep]:
    """Convert a list of raw step dicts into :class:`WorkflowStep` objects."""
    steps: list[WorkflowStep] = []
    for idx, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValidationError_(
                f"{section}[{idx}]: each step must be a mapping, got {type(raw).__name__}"
            )
        try:
            name = raw["name"]
            instrument = raw["instrument"]
            method = raw["method"]
        except KeyError as exc:
            raise ValidationError_(
                f"{section}[{idx}]: step is missing required field {exc}"
            ) from None

        params = _interpolate_params(raw.get("params", {}), variables)

        step = WorkflowStep(
            name=name,
            instrument=instrument,
            method=method,
            params=params,
            depends_on=raw.get("depends_on", []),
            timeout_s=raw.get("timeout_s"),
            retry=raw.get("retry", 0),
            tags=raw.get("tags", {}),
        )
        steps.append(step)
    return steps


def interpolate_params(
    params: dict[str, Any],
    variables: dict[str, Any],
) -> dict[str, Any]:
    """Public: replace ``"$var"`` refs in *params* using *variables*.

    Same substitution the parser applies at parse time, exposed so runtime
    callers that build a :class:`Workflow` in memory and execute it directly
    (e.g. the autonomous loop injecting optimizer-suggested values) can resolve
    ``$var`` placeholders without re-serialising through YAML.

    Handles exact matches (type-preserving), embedded string substitution, and
    nested dicts/lists; unresolved refs are left untouched.
    """
    return _interpolate_value(params, variables)  # type: ignore[return-value]


# Backwards-compatible private alias (existing call sites).
_interpolate_params = interpolate_params


def _interpolate_value(value: Any, variables: dict[str, Any]) -> Any:
    """Recursively interpolate ``$var`` references."""
    if isinstance(value, str):
        # Exact match (preserves type: int, list, …)
        m = _VAR_RE.fullmatch(value)
        if m:
            var_name = m.group(1)
            if var_name in variables:
                return copy.deepcopy(variables[var_name])
            return value  # leave unresolved — may be a runtime ref

        # Partial substitution (always returns str)
        def _sub(match: re.Match[str]) -> str:
            vname = match.group(1)
            return str(variables[vname]) if vname in variables else match.group(0)

        return _VAR_RE.sub(_sub, value)

    if isinstance(value, dict):
        return {k: _interpolate_value(v, variables) for k, v in value.items()}

    if isinstance(value, list):
        return [_interpolate_value(item, variables) for item in value]

    return value  # int, float, bool, None — pass through


def _resolve_iterations(
    iterate_over: str | None,
    variables: dict[str, Any],
) -> int:
    """Determine how many loop iterations to run.

    If ``iterate_over`` is ``"channels"`` and there is a list-type variable
    whose length defines the channel count (heuristic: the longest list in
    *variables*), use that length.  Otherwise default to 1.
    """
    if iterate_over is None:
        return 1

    # Direct variable reference
    if iterate_over in variables:
        ref = variables[iterate_over]
        if isinstance(ref, (list, tuple)):
            return len(ref)
        if isinstance(ref, int):
            return ref

    # Heuristic: longest list in variables (common pattern: vol_master is 32-elem list)
    max_len = 1
    for v in variables.values():
        if isinstance(v, (list, tuple)):
            max_len = max(max_len, len(v))
    return max_len
