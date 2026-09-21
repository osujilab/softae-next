"""How a :class:`~softae.core.run_plan.RunPlan` crosses the file boundary.

Until this module existed, ``campaign_spec_io._UNSUPPORTED`` refused ``run_plan``
outright with the stated reason *"a RunPlan object"* — no codec was written, not
a principled objection — and the consequence was structural: **every TOML
campaign ran pointwise formulate→measure with no anneal and no equilibrate phase
at all**, because a file had no way to say otherwise. A run plan is ordered
phases, and a phase is a kind, a scope, an optional cure task, an optional
commanded environment, an optional settle window and an optional measurement
override. All of those are primitives or small dataclasses of primitives, so the
refusal was a missing codec and this is it.

The file shape is one ``[run_plan]`` table carrying a ``phases`` array::

    [[run_plan.phases]]
    kind  = "anneal"
    scope = "per_batch"
    anneal_task = "anneal_85C_8h"
    hold_s      = 28800.0        # the cure; the task's own hold if omitted
      [run_plan.phases.conditions]
      name            = "anneal"
      temp_setpoint_C = 25.0     # what the chamber RESTORES to after the hold
      rh_setpoint_pct = 20.0

``hold_s`` is the file's only per-run override of a catalog anneal task, and it
is one typed float rather than a free table for the reason the decoder never
guesses: handed ``hold_time_sec = 3600`` a dict-shaped override cannot tell a
typo from a future key. ``anneal_params`` stays unwritable for exactly that
reason — an encoder meeting one answers ``UNREPRESENTABLE``. Note the
``conditions`` temperature above is the **restore** target, not the cure; the
deposition engine warns when the two are equal.

This module obeys :mod:`softae.core.campaign_spec_fields`'s asymmetry rule
verbatim — *"an encoder may answer* ``UNREPRESENTABLE`` *and be silently omitted,
but a decoder never guesses"* — and adds three refusals of its own, each for the
same reason the axis keys of a composition are all written out:

* **``scope`` is required on every phase.** No inference from ``batch`` or from
  the phase's kind. Per-sample anneals each well as it is cast; per-batch cures
  the whole plate at once. Those are different physical processes, and guessing
  between them would silently run the one nobody wrote down.
* **A key that cannot reach the instrument is refused, not ignored** — an
  ``anneal_task`` or a ``hold_s`` on a phase that does not anneal, a
  ``[measurement]`` block on a phase that does not measure (the last refused by
  :meth:`~softae.core.run_plan.RunPhase.__post_init__` itself, whose message is
  surfaced rather than restated).
* **A settle window states all three of its durations.** ``min_hold_s`` is the
  cure and belongs to the recipe, ``max_hold_s`` is the ceiling that makes the
  phase terminate, ``round_period_s`` is instrument time per sample; none can be
  invented.

**Absence and nothing are two different statements, and they are spelled
differently on purpose.** On a ``conditions`` axis, absence *is* nothing:
omitting ``rh_setpoint_pct`` means *do not drive humidity*, which is what
``None`` means on :class:`~softae.core.phase_setpoints.PhaseSetpoints`. Inside
``[settle]`` it is the opposite — ``rh_stability_pct`` defaults to a live gate,
so an omitted key switches the gate **on** and ``None`` switches it off — and
that is the case ``campaign_spec_io``'s ``explicit_none`` array already exists
for. The same spelling is used here, scoped to the settle table::

      [run_plan.phases.settle]
      round_period_s = 240.0
      min_hold_s     = 1500.0
      max_hold_s     = 14400.0
      criterion      = "rate"                  # which gate ROUTES
      rate_tol_dec_per_h = 0.05                # its band, in DECADES per hour
      explicit_none  = ["rh_stability_pct"]    # the RH stability gate is OFF

``rate_tol_dec_per_h`` is the operator's unit throughout — file, label and log —
and is converted to the tracker's ln-units at the tracker's own boundary. The two
keys travel together: :class:`~softae.core.run_plan.SettlePlan` refuses a rate
criterion with no band, because that pair computes no verdict at all and fails
looking exactly like a working run. ``rate_tol_dec_per_h`` is nullable on the
``rh_stability_pct`` precedent, with one difference worth knowing: there, an
omitted key switches the gate **on**, so ``None`` has to be spelled out; here
absence already means ``None``, so naming it in ``explicit_none`` is a permitted
statement of intent that the encoder will not write back.

**New settle capabilities are spelled here and only here.** The flat settle
fields on ``CampaignSpec`` are the legacy operator spelling; they do not grow, so
a file that writes ``criterion`` at the top level is refused by
``campaign_spec_io``'s unknown-field check rather than silently defaulted.

**Why this is a module of its own** rather than three more functions in
:mod:`softae.core.campaign_spec_fields`, where the other codecs live: a run plan
is four nested tables and the file would have grown by half. Registration into
``OBJECT_FIELDS`` is therefore one line *there* and everything else *here*::

    from softae.core.campaign_spec_run_plan import field_codec as _run_plan_codec
    # ... then inside the OBJECT_FIELDS literal:
        "run_plan": _run_plan_codec(),

``field_codec()`` imports :class:`~softae.core.campaign_spec_fields.FieldCodec`
**lazily**, and so does :func:`encode_run_plan` for ``UNREPRESENTABLE``. That is
not style: it keeps the dependency running one way at import time, so this
module can be imported before, after, or by the one that registers it, and the
pair's import order can never become load-bearing.
"""

from __future__ import annotations

from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from typing import Any

import structlog

from softae.core.measurement_spec import MeasurementSpec
from softae.core.phase_setpoints import PhaseSetpoints
from softae.core.run_plan import (
    DEFAULT_ANNEAL_TASK,
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "BASELINE_CONDITIONS_WHY_NOT",
    "CONDITION_SETS_WHY_NOT",
    "RUN_PLAN_WHY_NOT",
    "baseline_conditions_codec",
    "condition_sets_codec",
    "decode_baseline_conditions",
    "decode_condition_sets",
    "decode_run_plan",
    "encode_baseline_conditions",
    "encode_condition_sets",
    "encode_run_plan",
    "field_codec",
    "resolve_condition_references",
]

#: Why an encode can answer ``UNREPRESENTABLE``, in an operator's words. Read
#: back by ``spec_toml_completeness`` as "run_plan is {this}".
RUN_PLAN_WHY_NOT = (
    "a run plan carrying something the file shape cannot spell — per-run anneal "
    "parameter overrides, or an anneal task named on a phase that does not anneal"
)

#: The one key a ``[run_plan]`` table carries. The phases are the plan.
_PLAN_KEYS = frozenset({"phases"})

_PHASE_KEYS = frozenset(
    {"kind", "scope", "anneal_task", "hold_s", "conditions", "settle",
     "measurement"})

#: The two axes a phase may drive. Absent means ``None`` means *not driven*.
_CONDITION_AXES = ("temp_setpoint_C", "rh_setpoint_pct")
#: Approach bands and timeouts. Written only when they differ from
#: :mod:`softae.core.phase_setpoints`'s own documented defaults, for the reason
#: ``spec_to_dict`` writes only non-default spec fields: a file should show what
#: was chosen. Unlike a composition axis's ``basis``, these change how long an
#: approach is allowed to take, not which experiment runs.
_CONDITION_TUNING = ("tolerance_C", "rh_tolerance_pct",
                     "approach_timeout_s", "rh_approach_timeout_s")
_CONDITION_KEYS = frozenset(("name",) + _CONDITION_AXES + _CONDITION_TUNING)

#: The campaign-level ``[conditions]`` baseline (T11.28) is the same type and the
#: same axes — but **nothing waits for it**, so the two approach timeouts are
#: numbers that would do nothing. A number that does nothing is the trap this
#: codec exists to avoid, so they are refused there rather than accepted in
#: silence. The two *tolerances* are refused with them: a band is only ever read
#: by a wait, so it is exactly as inert.
_BASELINE_KEYS = frozenset(("name",) + _CONDITION_AXES)

#: Why an encode of the baseline can answer ``UNREPRESENTABLE``.
BASELINE_CONDITIONS_WHY_NOT = (
    "a baseline carrying an approach tolerance or timeout, which the campaign "
    "baseline waits for nothing and so cannot carry"
)

#: Why an encode of the named sets can answer ``UNREPRESENTABLE``. The one shape
#: a file cannot spell is a set whose key and whose ``PhaseSetpoints.name``
#: disagree: in a file there is only one place either is written — the TOML key
#: **is** the name — so a Python-built spec that separates them describes
#: something no file can say, and writing either half would be a guess.
CONDITION_SETS_WHY_NOT = (
    "a named condition set whose key and whose own name disagree, which a file "
    "cannot express because the key is the name"
)

#: The key a phase writes to reference a named set. Identical to the key an
#: inline table uses, deliberately: a phase says ``conditions`` and the *type* of
#: what follows says which form it is. Two keys would let a file carry both and
#: make the loader pick one.
CONDITION_REFERENCE_KEY = "conditions"

_SETTLE_REQUIRED = ("round_period_s", "min_hold_s", "max_hold_s")
_SETTLE_OPTIONAL = ("settle_tol_rel", "settle_n_rounds", "settle_min_channels",
                    "rh_stability_pct", "criterion", "rate_tol_dec_per_h")
#: The settle fields for which ``None`` is a value rather than an absence. They
#: are not the same statement: an omitted ``rh_stability_pct`` switches its gate
#: **on**, so ``None`` must be spellable; an omitted ``rate_tol_dec_per_h``
#: already *is* ``None``, so naming it here is a redundant — and permitted —
#: statement of intent that the encoder will not write back.
_SETTLE_NULLABLE = ("rh_stability_pct", "rate_tol_dec_per_h")

#: The same spelling :mod:`softae.core.campaign_spec_io` uses at the top level,
#: restated rather than imported because that module imports this one.
_EXPLICIT_NONE_KEY = "explicit_none"

_SETTLE_KEYS = frozenset(
    _SETTLE_REQUIRED + _SETTLE_OPTIONAL + (_EXPLICIT_NONE_KEY,))

_INT_SETTLE_KEYS = ("settle_n_rounds", "settle_min_channels")
#: Settle keys carrying a word rather than a number. The word itself is validated
#: by :class:`~softae.core.run_plan.SettlePlan` and deliberately not restated
#: here, so the legal set cannot drift from the one the tracker accepts.
_STR_SETTLE_KEYS = ("criterion",)


class _NotWritable(Exception):
    """Internal: this plan cannot be written. Never escapes the encoder."""


def _defaults(cls: type) -> dict[str, Any]:
    """Each declared default of *cls*, read from the dataclass itself.

    Read rather than restated so a retuned default there cannot leave a stale
    copy here deciding what is worth writing.
    """
    return {f.name: f.default for f in dataclass_fields(cls)
            if f.default is not MISSING}


# ── encode ───────────────────────────────────────────────────────────────────

def encode_run_plan(value: Any) -> Any:
    """The plan as ``{"phases": [table, …]}``, or ``UNREPRESENTABLE``.

    Never raises, per the module contract in
    :mod:`softae.core.campaign_spec_fields`: a plan this shape cannot carry is
    omitted from the written file and reported by ``spec_toml_completeness``,
    rather than breaking a write that is usually already a recovery path.
    """
    from softae.core.campaign_spec_fields import UNREPRESENTABLE

    if not isinstance(value, RunPlan):
        return UNREPRESENTABLE
    try:
        return {"phases": [_phase_to_table(p) for p in value.phases]}
    except (_NotWritable, AttributeError, TypeError, ValueError):
        logger.warning("run_plan_not_encodable", exc_info=True)
        return UNREPRESENTABLE


def _phase_to_table(phase: RunPhase) -> dict[str, Any]:
    if phase.anneal_params:
        raise _NotWritable(
            "anneal_params is a per-run override of a catalog task and the file "
            "shape has no key for it")
    table: dict[str, Any] = {"kind": phase.kind.value, "scope": phase.scope.value}
    if phase.kind is PhaseKind.ANNEAL:
        table["anneal_task"] = str(phase.anneal_task)
        if phase.hold_s is not None:
            table["hold_s"] = float(phase.hold_s)
    elif str(phase.anneal_task) != DEFAULT_ANNEAL_TASK:
        raise _NotWritable(
            f"{phase.kind.name} names anneal_task={phase.anneal_task!r}, which "
            f"the decoder refuses on a phase that does not anneal")
    # Scalars first, tables after: TOML puts a bare key under the most recently
    # opened table, so a nested table written before a scalar would capture it.
    if phase.conditions is not None:
        table["conditions"] = _conditions_to_table(phase.conditions)
    if phase.settle is not None:
        table["settle"] = _settle_to_table(phase.settle)
    if phase.measurement is not None:
        table["measurement"] = phase.measurement.as_dict()
    return table


def _conditions_to_table(setpoints: PhaseSetpoints) -> dict[str, Any]:
    table: dict[str, Any] = {"name": str(setpoints.name)}
    for axis in _CONDITION_AXES:
        value = getattr(setpoints, axis)
        if value is not None:
            table[axis] = float(value)      # absence is how `None` is spelled
    defaults = _defaults(PhaseSetpoints)
    for key in _CONDITION_TUNING:
        value = float(getattr(setpoints, key))
        if value != float(defaults[key]):
            table[key] = value
    return table


def _settle_to_table(settle: SettlePlan) -> dict[str, Any]:
    table: dict[str, Any] = {k: float(getattr(settle, k))
                             for k in _SETTLE_REQUIRED}
    defaults = _defaults(SettlePlan)
    nulls: list[str] = []
    for key in _SETTLE_OPTIONAL:
        value = getattr(settle, key)
        if value is None:
            # Only where an omitted key would switch the gate back ON — the exact
            # round-trip silence `explicit_none` exists to stop. A field whose own
            # default is None is already said by absence, and writing it would put
            # a redundant `explicit_none` in every file this encoder touches.
            if defaults[key] is not None:
                nulls.append(key)
        elif value != defaults[key]:
            table[key] = _settle_scalar(key, value)
    if nulls:
        table[_EXPLICIT_NONE_KEY] = sorted(nulls)
    return table


def _settle_scalar(key: str, value: Any) -> Any:
    """*value* in the type TOML should carry it for *key*."""
    if key in _STR_SETTLE_KEYS:
        return str(value)
    return int(value) if key in _INT_SETTLE_KEYS else float(value)


# ── decode ───────────────────────────────────────────────────────────────────

def decode_run_plan(value: Any) -> RunPlan:
    """A :class:`RunPlan` from the ``[run_plan]`` table, or ``ValueError``.

    Every refusal below is a refusal to run a different experiment from the one
    the file describes. :class:`RunPlan`'s own invariants — a plan needs a
    FORMULATE phase, FORMULATE is per-sample, EQUILIBRATE needs a settle window,
    ARRHENIUS is reserved — are left to its ``__post_init__`` and surface from
    here unchanged.
    """
    if not isinstance(value, dict):
        raise ValueError(
            "expected a [run_plan] table carrying a 'phases' array of "
            "[[run_plan.phases]] tables")
    unknown = sorted(set(value) - _PLAN_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown}; a run plan carries only 'phases'")
    raw = value.get("phases")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "'phases' must be a non-empty array of [[run_plan.phases]] tables — "
            "a run plan with no phases describes no run")
    return RunPlan(tuple(_phase_from_dict(row, i)
                         for i, row in enumerate(raw, start=1)))


def _phase_from_dict(row: Any, index: int) -> RunPhase:
    if not isinstance(row, dict):
        raise ValueError(f"phase #{index} is not a table")
    unknown = sorted(set(row) - _PHASE_KEYS)
    if unknown:
        raise ValueError(
            f"phase #{index} has unknown key(s) {unknown}; valid keys: "
            f"{sorted(_PHASE_KEYS)}")

    kind = _enum_from(PhaseKind, row.get("kind"), "kind", index)
    if "scope" not in row:
        raise ValueError(
            f"phase #{index} ({kind.value}) does not declare 'scope'. Every "
            f"phase states it: 'per_sample' anneals, equilibrates or measures "
            f"each well as it is cast, 'per_batch' does the whole plate at once, "
            f"and inferring one would silently run the physical process nobody "
            f"wrote down")
    scope = _enum_from(PhaseScope, row["scope"], "scope", index)

    kwargs: dict[str, Any] = {}
    if "anneal_task" in row:
        if kind is not PhaseKind.ANNEAL:
            raise ValueError(
                f"phase #{index} ({kind.value}) names an 'anneal_task'; only an "
                f"ANNEAL phase runs one, so it would be accepted here and never "
                f"reach the chamber")
        task = row["anneal_task"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                f"phase #{index}: 'anneal_task' must be the name of a catalog task")
        kwargs["anneal_task"] = task
    if "hold_s" in row:
        if kind is not PhaseKind.ANNEAL:
            raise ValueError(
                f"phase #{index} ({kind.value}) names a 'hold_s'; only an "
                f"ANNEAL phase holds at temperature for a stated duration, so "
                f"it would be accepted here and never reach the chamber")
        kwargs["hold_s"] = _number(row["hold_s"], f"phase #{index}: 'hold_s'")
    if "conditions" in row:
        kwargs["conditions"] = _conditions_from_dict(row["conditions"], index)
    if "settle" in row:
        kwargs["settle"] = _settle_from_dict(row["settle"], index)
    if "measurement" in row:
        kwargs["measurement"] = _measurement_from_dict(row["measurement"], index)

    try:
        return RunPhase(kind, scope, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"phase #{index} ({kind.value}): {exc}") from exc


def _enum_from(enum_cls: type, raw: Any, key: str, index: int) -> Any:
    """*raw* as a member of *enum_cls*. Absent and unknown are different errors."""
    if raw is None:
        raise ValueError(f"phase #{index} is missing '{key}'")
    try:
        return enum_cls(str(raw))
    except ValueError:
        legal = ", ".join(repr(m.value) for m in enum_cls)     # type: ignore[attr-defined]
        raise ValueError(
            f"phase #{index} has unknown {key} {raw!r}; legal values are "
            f"{legal}") from None


def _conditions_from_dict(raw: Any, index: int) -> PhaseSetpoints:
    return _conditions_from_table(raw, f"phase #{index}: [conditions]", tuning=True)


def _conditions_from_table(raw: Any, where: str, *, tuning: bool) -> PhaseSetpoints:
    """*raw* as :class:`PhaseSetpoints`. ``tuning=False`` is the baseline shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table")
    if not tuning:
        inert = sorted(set(raw) & set(_CONDITION_TUNING))
        if inert:
            raise ValueError(
                f"{where} names {inert}, and NOTHING WAITS HERE. The campaign "
                f"baseline is commanded at the top of the run and waited for by "
                f"nothing, so an approach band or timeout would be a number with "
                f"no reader — accepting it would read as a gate that does not "
                f"exist. Put them on the phase whose [conditions] actually gates, "
                f"or drop them.")
    legal = _CONDITION_KEYS if tuning else _BASELINE_KEYS
    unknown = sorted(set(raw) - legal)
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {unknown}; valid keys: {sorted(legal)}")
    if "name" not in raw:
        raise ValueError(
            f"{where} needs a 'name' — it is what an operator reads back in the "
            f"phase order")
    kwargs: dict[str, Any] = {"name": raw["name"]}
    # An omitted axis is the axis not being driven at all, which is exactly what
    # `None` means on PhaseSetpoints; TOML has no other way to say it. In
    # particular `rh_setpoint_pct = 0.0` is NOT absence — it is a commanded dry
    # purge (`drivers/async_rh_controller.safe_dry`, bench-verified 2026-08-21).
    for key in _CONDITION_AXES + (_CONDITION_TUNING if tuning else ()):
        if key in raw:
            kwargs[key] = _number(raw[key], f"{where} '{key}'")
    try:
        return PhaseSetpoints(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


# ── the campaign-level [conditions] baseline (T11.28) ────────────────────────

def encode_baseline_conditions(value: Any) -> Any:
    """The baseline as a ``[conditions]`` table, or :data:`UNREPRESENTABLE`.

    A baseline carrying a non-default tolerance or timeout answers
    ``UNREPRESENTABLE`` rather than writing keys this module's own decoder
    refuses: a file that cannot be read back is worse than a field reported
    missing by ``spec_toml_completeness``.
    """
    from softae.core.campaign_spec_fields import UNREPRESENTABLE

    if not isinstance(value, PhaseSetpoints):
        return UNREPRESENTABLE
    table = _conditions_to_table(value)
    if set(table) - _BASELINE_KEYS:
        logger.warning("baseline_conditions_not_encodable", name=value.name)
        return UNREPRESENTABLE
    return table


def decode_baseline_conditions(value: Any) -> PhaseSetpoints:
    """The campaign-level ``[conditions]`` table as :class:`PhaseSetpoints`."""
    return _conditions_from_table(value, "[conditions]", tuning=False)


# -- named condition sets (T11.28b) -------------------------------------------
#
# ``[condition_sets.<name>]`` at the TOP level of the file, beside
# ``parameter_space`` and ``budget`` (operator ruling 2026-09-20; spec
# ``t11_28b_one_config_level_for_humidity.md`` section 4.4.2a), referenced from a
# phase by name: ``conditions = "casting"``.
#
# Three properties decide the shape, and each one is a refusal somewhere below:
#
#   * **Omission still means NOT DRIVEN.** A phase acquires conditions only by
#     *naming* a set, so absence keeps the single meaning
#     :mod:`softae.core.phase_setpoints`'s module docstring gives it. This is why
#     the operator chose named sets over campaign-level inheritance: inheritance
#     re-spells absence as "take the campaign value", and ``SUBAGENT_RULES.md``
#     section 3.1(a) forbids one token meaning two things. Nothing here has an
#     ``else`` branch that supplies a default set.
#
#   * **A named set is only ever a PHASE's conditions, never the baseline.** So
#     :data:`_CONDITION_KEYS` applies unconditionally -- the approach bands and
#     timeouts are legal here, because a phase waits. That is the one asymmetry
#     with :func:`decode_baseline_conditions`, and it is why the baseline's
#     *"NOTHING WAITS HERE"* refusal is untouched by any of this.
#
#   * **The TOML key IS the name**, table-of-tables rather than array-of-tables.
#     TOML's own duplicate-key refusal gives uniqueness for free, and there is
#     exactly one place a name is written, so the reference and
#     ``PhaseSetpoints.name`` cannot disagree. An explicit ``name`` *inside* a set
#     is refused for that reason: two spellings of one thing, where preferring
#     either silently is the guess this module never makes.


def _named_set_table(name: Any, body: Any, where: str) -> dict[str, Any]:
    """One named set's body as the inline conditions table it is equivalent to.

    The TOML key supplies ``name``, which is why an explicit one in the body is
    refused rather than preferred: :func:`_conditions_from_table` requires a
    name, and a set carrying its own could name itself something the phase
    reference does not -- one thing spelled two ways, with no rule for which wins
    that is not a guess.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"{where}: a condition set's name must be a non-empty string -- it "
            f"is what a phase writes to reference it (got {name!r})")
    if not isinstance(body, dict):
        raise ValueError(
            f"[condition_sets.{name}] must be a table of setpoints "
            f"(got {type(body).__name__})")
    if "name" in body:
        raise ValueError(
            f"[condition_sets.{name}] sets its own 'name'. The table key IS the "
            f"name -- it is what a phase writes as conditions = \"{name}\" -- so a "
            f"second spelling could disagree with it and neither could be "
            f"preferred silently. Drop the 'name' key.")
    # Checked here rather than left to `_conditions_from_table`, which would
    # report `name` among the valid keys one line after refusing it: a message
    # that contradicts the refusal above sends an operator round the loop again.
    unknown = sorted(set(body) - (_CONDITION_KEYS - {"name"}))
    if unknown:
        raise ValueError(
            f"[condition_sets.{name}] has unknown key(s) {unknown}; valid keys: "
            f"{sorted(_CONDITION_KEYS - {'name'})} — 'name' is the table key")
    return {"name": name, **body}


def decode_condition_sets(value: Any) -> dict[str, PhaseSetpoints]:
    """The top-level ``[condition_sets]`` table as ``{name: PhaseSetpoints}``.

    Arbitrary N, names the operator chooses: the legal set of names is TOML's key
    syntax and nothing else. There is no vocabulary, no enum and no count
    anywhere in this function -- one set and nine take the identical path.
    """
    if not isinstance(value, dict):
        raise ValueError(
            "expected a [condition_sets] table of named tables, e.g. "
            "[condition_sets.casting]")
    out: dict[str, PhaseSetpoints] = {}
    for name, body in value.items():
        table = _named_set_table(name, body, "[condition_sets]")
        out[str(name)] = _conditions_from_table(
            table, f"[condition_sets.{name}]", tuning=True)
    return out


def encode_condition_sets(value: Any) -> Any:
    """``{name: PhaseSetpoints}`` as a ``[condition_sets]`` table, or
    :data:`UNREPRESENTABLE`.

    ``name`` is dropped from each body because the key carries it and the decoder
    refuses it inside -- writing both would produce a file this module's own
    decoder rejects, which is worse than a field reported missing by
    ``spec_toml_completeness``.
    """
    from softae.core.campaign_spec_fields import UNREPRESENTABLE

    if not isinstance(value, dict):
        return UNREPRESENTABLE
    out: dict[str, Any] = {}
    for name, setpoints in value.items():
        if not isinstance(name, str) or not isinstance(setpoints, PhaseSetpoints):
            logger.warning("condition_sets_not_encodable", key=str(name))
            return UNREPRESENTABLE
        if str(setpoints.name) != name:
            logger.warning("condition_sets_name_disagrees",
                           key=name, name=str(setpoints.name))
            return UNREPRESENTABLE
        table = _conditions_to_table(setpoints)
        table.pop("name")
        out[name] = table
    return out


def resolve_condition_references(
    run_plan: Any, condition_sets: dict[str, PhaseSetpoints],
) -> "tuple[Any, frozenset[str]]":
    """*run_plan* with every ``conditions = "<name>"`` replaced by that set's table.

    This is the cross-key pre-pass the top-level placement costs.
    ``OBJECT_FIELDS`` decodes each field as ``codec.decode(supplied[key])`` --
    **one argument** -- so the ``run_plan`` codec is handed the ``run_plan``
    sub-table and nothing else in the file. A *sibling* top-level
    ``[condition_sets]`` is invisible to it, and resolution therefore has to
    happen before it is called, in the one place that can see both keys.

    Substituting the **encoded table** rather than the object keeps one decode
    authority: :func:`_conditions_to_table` is the exact inverse of
    :func:`_conditions_from_table` with ``tuning=True``, so a resolved phase is
    indistinguishable from one that wrote the block inline -- which is what makes
    migrating a file to references a pure edit with no behaviour attached.

    Returns the rewritten plan and the names actually referenced, so the caller
    can say how many declared sets nothing points at. Anything this function does
    not recognise is passed through untouched for :func:`decode_run_plan` to
    refuse with its own message; refusing twice, in two vocabularies, for one
    defect is how a loader grows two disagreeing error surfaces.
    """
    if not isinstance(run_plan, dict):
        return run_plan, frozenset()
    phases = run_plan.get("phases")
    if not isinstance(phases, list):
        return run_plan, frozenset()

    referenced: set[str] = set()
    resolved: list[Any] = []
    for index, row in enumerate(phases, start=1):
        if not isinstance(row, dict) or CONDITION_REFERENCE_KEY not in row:
            resolved.append(row)
            continue
        raw = row[CONDITION_REFERENCE_KEY]
        if isinstance(raw, dict):
            resolved.append(row)            # inline, exactly as before
            continue
        if not isinstance(raw, str):
            raise ValueError(
                f"phase #{index}: 'conditions' must be either the NAME of a "
                f"declared condition set (conditions = \"casting\") or an inline "
                f"[run_plan.phases.conditions] table -- got "
                f"{type(raw).__name__}. Those are the only two forms, and a "
                f"phase carries one or the other, never both.")
        if raw not in condition_sets:
            declared = sorted(condition_sets)
            names = (str(declared) if declared
                     else "none -- the file declares no [condition_sets] at all")
            raise ValueError(
                f"phase #{index}: 'conditions' names {raw!r}, which no "
                f"[condition_sets.{raw}] declares. Declared set(s): {names}. "
                f"A name that resolves to nothing is refused at load rather than "
                f"at the bench: this refusal is the whole reason a name beats a "
                f"repeated number, because it turns a typo into a parse-time "
                f"failure that says what was probably meant.")
        referenced.add(raw)
        resolved.append({
            **row,
            CONDITION_REFERENCE_KEY: _conditions_to_table(condition_sets[raw]),
        })
    return {**run_plan, "phases": resolved}, frozenset(referenced)


def _settle_from_dict(raw: Any, index: int) -> SettlePlan:
    where = f"phase #{index}: [settle]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table")
    unknown = sorted(set(raw) - _SETTLE_KEYS)
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {unknown}; valid keys: "
            f"{sorted(_SETTLE_KEYS)}")

    nulls = _settle_nulls(raw, where)
    missing = [k for k in _SETTLE_REQUIRED if k not in raw]
    if missing:
        raise ValueError(
            f"{where} is missing {missing}. min_hold_s is the cure time and "
            f"belongs to the recipe, max_hold_s is the ceiling that guarantees "
            f"the phase terminates, and round_period_s is instrument time spent "
            f"per sample — none of the three has a safe default")

    kwargs: dict[str, Any] = {k: _number(raw[k], f"{where} '{k}'")
                              for k in _SETTLE_REQUIRED}
    for key in _SETTLE_OPTIONAL:
        if key not in raw:
            continue
        if key in _STR_SETTLE_KEYS:
            # Passed through unvalidated on purpose: SettlePlan owns the legal
            # set, and its refusal surfaces from the constructor below with this
            # table's prefix already on it. A non-string reaches the same
            # membership test and is refused as a ValueError, not a TypeError.
            kwargs[key] = raw[key]
        else:
            kwargs[key] = (_integer(raw[key], f"{where} '{key}'")
                           if key in _INT_SETTLE_KEYS
                           else _number(raw[key], f"{where} '{key}'"))
    for key in nulls:
        kwargs[key] = None

    try:
        return SettlePlan(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _settle_nulls(raw: dict, where: str) -> tuple[str, ...]:
    """The settle fields this table sets to **nothing**, validated.

    Scoped to the fields for which ``None`` is a value: naming any other one
    would be a request to drop a required duration, which is not an absence but
    a missing floor.
    """
    listed = raw.get(_EXPLICIT_NONE_KEY)
    if listed is None:
        return ()
    if not isinstance(listed, list) or not all(isinstance(n, str) for n in listed):
        raise ValueError(f"{where} '{_EXPLICIT_NONE_KEY}' must be an array of "
                         f"field names")
    illegal = sorted(set(listed) - set(_SETTLE_NULLABLE))
    if illegal:
        raise ValueError(
            f"{where} '{_EXPLICIT_NONE_KEY}' names {illegal}; only "
            f"{list(_SETTLE_NULLABLE)} can be set to nothing — the rest are "
            f"durations and thresholds a hold cannot run without")
    both = sorted(set(listed) & set(raw))
    if both:
        raise ValueError(
            f"{where} gives {both} a value *and* lists them in "
            f"'{_EXPLICIT_NONE_KEY}' — the file says two things about one field, "
            f"and neither can be preferred silently")
    return tuple(listed)


def _measurement_from_dict(raw: Any, index: int) -> MeasurementSpec:
    where = f"phase #{index}: [measurement]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table")
    try:
        return MeasurementSpec.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number (got {value!r})")
    return float(value)


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be a whole number (got {value!r})")
    return int(value)


# ── registration ─────────────────────────────────────────────────────────────

_CODEC: Any = None


def field_codec() -> Any:
    """This module's codec, in the shape ``campaign_spec_io`` consults.

    Built on first call rather than at import, because
    :class:`~softae.core.campaign_spec_fields.FieldCodec` lives in the module
    that will one day import *this* one — see the module docstring.
    """
    global _CODEC
    if _CODEC is None:
        from softae.core.campaign_spec_fields import FieldCodec

        _CODEC = FieldCodec(encode_run_plan, decode_run_plan, RUN_PLAN_WHY_NOT)
    return _CODEC


_BASELINE_CODEC: Any = None


def baseline_conditions_codec() -> Any:
    """The ``conditions`` codec, built on first call for ``field_codec``'s reason."""
    global _BASELINE_CODEC
    if _BASELINE_CODEC is None:
        from softae.core.campaign_spec_fields import FieldCodec

        _BASELINE_CODEC = FieldCodec(
            encode_baseline_conditions, decode_baseline_conditions,
            BASELINE_CONDITIONS_WHY_NOT)
    return _BASELINE_CODEC


_CONDITION_SETS_CODEC: Any = None


def condition_sets_codec() -> Any:
    """The ``condition_sets`` codec, built on first call for ``field_codec``'s reason.

    Registered like any other field even though the *decode* half is also
    reached by :func:`spec_from_dict`'s pre-pass: the registration is what gives
    ``spec_to_dict`` an **encoder**, and without one ``spec_toml_completeness``
    would hand ``tomli_w`` a dict of :class:`PhaseSetpoints` objects and report
    the whole representable part as invalid TOML.
    """
    global _CONDITION_SETS_CODEC
    if _CONDITION_SETS_CODEC is None:
        from softae.core.campaign_spec_fields import FieldCodec

        _CONDITION_SETS_CODEC = FieldCodec(
            encode_condition_sets, decode_condition_sets,
            CONDITION_SETS_WHY_NOT)
    return _CONDITION_SETS_CODEC
