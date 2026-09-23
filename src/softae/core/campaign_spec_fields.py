"""The spec fields that are live Python objects, and how a file carries them.

:mod:`softae.core.campaign_spec_io` refuses what it cannot represent faithfully.
That refusal is right, but for three fields it was refusing something that *is*
representable — and since a campaign is started from a spec file, "unrepresentable"
had become "unrunnable" for the Live BO tab's headline mode:

============================  ==================================================
``general_formulation``       held a ``build_targets`` **callable**, but the
                              panel builds that callable from declared
                              :class:`~softae.core.composition_axes.CompositionAxis`
                              rows. The axes are primitives; the stocks are
                              *names* the solution catalog resolves.
``prior_mean``                a callable, but chosen from a fixed combo — so it
                              is a **name** (:mod:`softae.optimizers.prior_means`).
``seed_observations``         ``[(params, value), …]`` — primitives already.
============================  ==================================================

Each field gets an encoder and a decoder here, and the split from
``campaign_spec_io`` is deliberate: that module owns *whether* a file is the whole
spec, this one owns *how* one field crosses the boundary. The asymmetry both must
preserve is the design's core — an encoder may answer :data:`UNREPRESENTABLE` and
be silently omitted, but a **decoder never guesses**: a file that sets one of these
to something it cannot mean raises, because a partially-loaded field would run a
different experiment from the one the file describes.

Encoders never raise. Decoders raise :class:`ValueError`, which the loader wraps
into a ``SpecLoadError`` naming the field and the source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import structlog

from softae.core.campaign_spec_run_plan import (
    baseline_conditions_codec as _baseline_conditions_codec,
)
from softae.core.campaign_spec_run_plan import (
    condition_sets_codec as _condition_sets_codec,
)
from softae.core.campaign_spec_run_plan import field_codec as _run_plan_codec

logger = structlog.get_logger(__name__)


class _Unrepresentable:
    """Sentinel: this value cannot be written, and no ``None`` means that."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unrepresentable>"


#: Returned by an encoder that cannot faithfully write the value it was given.
#: Distinct from ``None``, which is a value a spec field can legitimately hold.
UNREPRESENTABLE = _Unrepresentable()


@dataclass(frozen=True)
class FieldCodec:
    """How one object-valued spec field crosses the file boundary."""

    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    #: Why an encode can answer :data:`UNREPRESENTABLE`, in an operator's words.
    why_not: str


# ── prior_mean ───────────────────────────────────────────────────────────────

def encode_prior_mean(value: Any) -> Any:
    """The registry name of *value*, or :data:`UNREPRESENTABLE`."""
    from softae.optimizers.prior_means import prior_mean_name

    name = prior_mean_name(value)
    return name if name is not None else UNREPRESENTABLE


def decode_prior_mean(value: Any) -> Any:
    from softae.optimizers.prior_means import resolve_prior_mean

    if not isinstance(value, str):
        raise ValueError(
            "expected the name of a built-in prior mean (a string), got "
            f"{type(value).__name__}"
        )
    return resolve_prior_mean(value)


# ── seed_observations ────────────────────────────────────────────────────────

def encode_seed_observations(value: Any) -> Any:
    """``[{params = {...}, value = x}, …]`` — an array of tables, or the sentinel."""
    rows: list[dict[str, Any]] = []
    try:
        for params, objective in value:
            if not isinstance(params, dict) or not all(
                    isinstance(k, str) for k in params):
                return UNREPRESENTABLE
            rows.append({"params": dict(params), "value": float(objective)})
    except (TypeError, ValueError):
        return UNREPRESENTABLE
    return rows


def decode_seed_observations(value: Any) -> Any:
    if not isinstance(value, list):
        raise ValueError(
            "expected an array of [[seed_observations]] tables, each with a "
            "'params' table and a 'value'")
    out: list[tuple[dict[str, Any], float]] = []
    for i, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"observation #{i} is not a table")
        unknown = sorted(set(row) - {"params", "value"})
        if unknown:
            raise ValueError(f"observation #{i} has unknown key(s) {unknown}")
        params = row.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"observation #{i} needs a 'params' table")
        if "value" not in row:
            raise ValueError(f"observation #{i} needs a 'value'")
        try:
            objective = float(row["value"])
        except (TypeError, ValueError):
            raise ValueError(
                f"observation #{i} has a non-numeric 'value'") from None
        out.append((dict(params), objective))
    return tuple(out)


# ── general_formulation ──────────────────────────────────────────────────────

#: The axis keys a *file* must carry. Each is always written too: an axis with an
#: omitted key would silently take a default (``basis``, ``low``) and search a
#: different composition.
_AXIS_REQUIRED_KEYS = ("kind", "a", "b", "low", "high", "basis")

#: Every field of a :class:`~softae.core.composition_axes.CompositionAxis`, which
#: is what an axis table may *carry* — the same "permitted" role ``_GF_KEYS`` and
#: ``_PIEZO_KEYS`` play. ``scale`` is permitted but **not** required: it postdates
#: every spec on disk, and each of those describes the linear search its absence
#: already means, so requiring it would refuse a file for omitting a key it could
#: not have known about. The encoder writes it regardless, for the reason above —
#: the asymmetry is deliberate and narrow, and applies to no other axis key.
_AXIS_KEYS = _AXIS_REQUIRED_KEYS + ("scale",)

_GF_KEYS = frozenset(
    {"stocks", "pump_assignment", "target_deposition_uL", "axes",
     "budget_uL", "dried_frac", "thickness_um", "thickness_basis"})

#: The two keys that fix a formulation's scale. Exactly one may be declared:
#: both would let a spec state the scale twice and silently prefer one, and
#: neither leaves the solve under-determined. Conversion between them is
#: impossible here anyway — the area comes from ``spec.pcb_name``, which this
#: decoder never sees.
_SCALE_KEYS = ("target_deposition_uL", "thickness_um")


def catalogs() -> tuple[Any, Any]:
    """``(ChemicalCatalog, SolutionCatalog)`` a stock **name** resolves against.

    A seam, so a test — and only a test — can supply catalogs without a data
    root. Production has exactly one answer and it is the shared one: the same
    CSVs the deposition twin, the consumables ledger and the particulate-pump
    resolver read, because a campaign built from a second copy of the chemistry
    would be free to disagree with the rig.
    """
    from softae.core.stock_assignment import catalogs_from_data_root

    return catalogs_from_data_root()


def _encode_scale(value: Any) -> dict[str, Any]:
    """Whichever scale key the context declares — never both, never neither.

    ``thickness_basis`` is written whenever a thickness is, for the reason the
    axis keys are all written: an omitted key would silently take a default and
    cast a different film.
    """
    dt = getattr(value, "deposit_target", None)
    if dt is not None and getattr(dt, "kind", None) == "thickness":
        return {"thickness_um": float(dt.value),
                "thickness_basis": str(dt.basis)}
    if dt is not None:
        return {"target_deposition_uL": float(dt.value)}
    return {"target_deposition_uL": float(value.target_deposition_uL)}


def encode_general_formulation(value: Any) -> Any:
    """The composition context as primitives, or :data:`UNREPRESENTABLE`.

    Representable **only when the axes are the whole truth about its targets** —
    that is, when the context derived ``build_targets`` from them
    (``axes_define_targets``). ``build_targets`` alone is an arbitrary callable,
    and a file that wrote the stocks but not the targets would reload as a
    campaign searching the same volumes for a different composition — which is
    precisely the silent difference this refuses. A context given *both* is
    refused for the same reason and no weaker one: the callable may compute
    something the axes do not describe, and no check can compare two callables.
    """
    from softae.core.autonomous_wiring import GeneralFormulation

    if not isinstance(value, GeneralFormulation) or not getattr(
            value, "axes_define_targets", False):
        return UNREPRESENTABLE
    try:
        out: dict[str, Any] = {
            "stocks": sorted(str(name) for name in value.stocks),
            "pump_assignment": {str(k): int(v)
                                for k, v in value.pump_assignment.items()},
            **_encode_scale(value),
            "axes": [
                {"kind": str(ax.kind), "a": str(ax.a), "b": str(ax.b),
                 "low": float(ax.low), "high": float(ax.high),
                 "basis": str(ax.basis), "scale": str(ax.scale)}
                for ax in value.axes
            ],
        }
        if value.budget_uL is not None:
            out["budget_uL"] = float(value.budget_uL)
        if value.dried_frac is not None:
            out["dried_frac"] = {str(k): float(v)
                                 for k, v in value.dried_frac.items()}
    except (AttributeError, TypeError, ValueError):
        logger.warning("general_formulation_not_encodable", exc_info=True)
        return UNREPRESENTABLE
    return out


def _axis_from_dict(row: Any, index: int) -> Any:
    from softae.core.composition_axes import CompositionAxis

    if not isinstance(row, dict):
        raise ValueError(f"axis #{index} is not a table")
    unknown = sorted(set(row) - set(_AXIS_KEYS))
    if unknown:
        raise ValueError(f"axis #{index} has unknown key(s) {unknown}")
    missing = [k for k in _AXIS_REQUIRED_KEYS if k not in row]
    if missing:
        raise ValueError(
            f"axis #{index} is missing {missing} — every key is written so that "
            f"an omitted one cannot silently take a default and search a "
            f"different composition")
    # ``scale`` is passed only when the file declares it, so ``"linear"`` stays
    # written in exactly one place — ``CompositionAxis``'s own default — for the
    # reason ``decode_piezo`` omits a keyword rather than restating one. An
    # illegal value is refused by the dataclass and wrapped below, so this codec
    # adds no second opinion about which scales exist.
    optional = {"scale": str(row["scale"])} if "scale" in row else {}
    try:
        return CompositionAxis(
            kind=str(row["kind"]), a=str(row["a"]), b=str(row["b"]),
            low=float(row["low"]), high=float(row["high"]),
            basis=str(row["basis"]), **optional)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"axis #{index}: {exc}") from exc


def _decode_scale(value: dict) -> tuple[Any, float | None]:
    """``(DepositTargetSpec, target_deposition_uL)`` from the declared scale key.

    ``target_deposition_uL`` comes back ``None`` for a thickness spec: it has no
    volume until an area is resolved, and a placeholder would be a scale nobody
    declared travelling into a solve.
    """
    from softae.core.deposit_target import DepositTargetSpec

    declared = [k for k in _SCALE_KEYS if k in value]
    if len(declared) > 1:
        raise ValueError(
            "declares both 'target_deposition_uL' and 'thickness_um' — both fix "
            "the scale of the cast, so a spec carrying both states it twice and "
            "one of them would be silently ignored. Declare exactly one.")
    if not declared:
        raise ValueError(
            "declares neither 'target_deposition_uL' nor 'thickness_um' — a "
            "composition solve with no scale row is under-determined, and a "
            "substituted default would be a scale nobody declared.")

    if declared[0] == "target_deposition_uL":
        if "thickness_basis" in value:
            raise ValueError(
                "'thickness_basis' has no meaning beside 'target_deposition_uL'; "
                "it selects which volume a 'thickness_um' refers to")
        try:
            volume_uL = float(value["target_deposition_uL"])
        except (TypeError, ValueError):
            raise ValueError("'target_deposition_uL' must be a number") from None
        return DepositTargetSpec("volume", volume_uL), volume_uL

    try:
        thickness_um = float(value["thickness_um"])
    except (TypeError, ValueError):
        raise ValueError("'thickness_um' must be a number") from None
    basis = value.get("thickness_basis", "dry")
    if not isinstance(basis, str):
        raise ValueError("'thickness_basis' must be 'dry' or 'wet'")
    return DepositTargetSpec("thickness", thickness_um, basis), None


def decode_general_formulation(value: Any) -> Any:
    from dataclasses import fields as dataclass_fields

    from softae.core.autonomous_wiring import GeneralFormulation

    if not isinstance(value, dict):
        raise ValueError("expected a [general_formulation] table")
    unknown = sorted(set(value) - _GF_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}")
    for required in ("stocks", "pump_assignment", "axes"):
        if required not in value:
            raise ValueError(f"'{required}' is required")
    deposit_target, target_deposition_uL = _decode_scale(value)

    raw_axes = value["axes"]
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ValueError(
            "'axes' must be a non-empty array of [[general_formulation.axes]] "
            "tables — a composition campaign with no targets has nothing to solve")
    axes = tuple(_axis_from_dict(row, i)
                 for i, row in enumerate(raw_axes, start=1))

    raw_stocks = value["stocks"]
    if not isinstance(raw_stocks, list) or not raw_stocks:
        raise ValueError("'stocks' must be a non-empty array of solution names")
    chem_catalog, sol_catalog = catalogs()
    stocks: dict[str, Any] = {}
    for name in raw_stocks:
        try:
            stocks[str(name)] = sol_catalog.get(str(name))
        except (KeyError, AttributeError):
            raise ValueError(
                f"stock {str(name)!r} is not in the solution catalog, so the "
                f"solver cannot turn a composition target into pump volumes. "
                f"Declare it in the catalog, or write a spec naming stocks that "
                f"are in it."
            ) from None

    raw_pumps = value["pump_assignment"]
    if not isinstance(raw_pumps, dict):
        raise ValueError("'pump_assignment' must be a table of stock name → pump")
    try:
        pump_assignment = {str(k): int(v) for k, v in raw_pumps.items()}
    except (TypeError, ValueError):
        raise ValueError("'pump_assignment' values must be pump indices") from None
    unassigned = sorted(set(pump_assignment) - set(stocks))
    if unassigned:
        raise ValueError(
            f"pump_assignment names stock(s) {unassigned} that 'stocks' does not "
            f"list")

    dried = value.get("dried_frac")
    kwargs: dict[str, Any] = dict(
        stocks=stocks,
        catalog=chem_catalog,
        pump_assignment=pump_assignment,
        target_deposition_uL=target_deposition_uL,
        budget_uL=(None if value.get("budget_uL") is None
                   else float(value["budget_uL"])),
        dried_frac=(None if dried is None
                    else {str(k): float(v) for k, v in dried.items()}),
        axes=axes,
    )
    # The campaign context grows its ``deposit_target`` field in wave W3
    # (docs/SubAgent docs/thickness_target_all_paths.md). Until it does, a
    # thickness must be refused rather than dropped: a spec that decoded
    # without its scale would run a different experiment from the one the file
    # describes — the exact silence this module exists to prevent.
    if any(f.name == "deposit_target" for f in dataclass_fields(GeneralFormulation)):
        kwargs["deposit_target"] = deposit_target
    elif deposit_target.kind == "thickness":
        raise ValueError(
            "'thickness_um' cannot be carried yet: the campaign's composition "
            "context has no deposit target field, so the thickness would be "
            "dropped and the campaign would cast an undeclared volume. Declare "
            "'target_deposition_uL' until the campaign wiring lands.")
    return GeneralFormulation(**kwargs)


# ── piezo ────────────────────────────────────────────────────────────────────
#
# ``PiezoPlan`` was the last object-valued field ``campaign_spec_io._UNSUPPORTED``
# refused outright, and the refusal had the same shape ``run_plan``'s did: the
# plan is wired end-to-end in Python (``CampaignSpec.piezo`` →
# ``DepositionSettings.piezo`` → ``build_recipe_deposition_workflow``, since
# P2.1) and the HT tab builds one from config, so a *file* was the only surface
# that could not ask for it — ``autonomous_wiring.py``'s own field comment names
# the consequence, *"campaigns never actuated the piezo even with [piezo]
# configured"*. It is six flat primitives; the refusal was a missing codec.
#
# It lives here rather than in a module of its own, unlike ``run_plan``: a piezo
# plan is one flat table with no nested phase array, so there is nothing for a
# second module to hold.

#: Why an encode can answer :data:`UNREPRESENTABLE`, in an operator's words. Read
#: back by ``spec_toml_completeness`` as "piezo is {this}".
PIEZO_WHY_NOT = (
    "a piezo plan carrying event_params, a per-run override of a catalog task's "
    "parameters that the file shape has no key for"
)

#: Every key a ``[piezo]`` table may carry, each the name of a
#: :class:`~softae.core.deposition_recipe.PiezoPlan` field. Written out rather
#: than derived from the dataclass **on purpose**: a field added to ``PiezoPlan``
#: must be *refused* here until this codec learns to carry it, where a derived set
#: would accept the key and then drop it on the floor — "accepted" and "carried"
#: spelled with one token. ``test_campaign_spec_io`` pins the pair so the
#: addition goes red rather than silent.
_PIEZO_KEYS = ("enabled", "on_task", "off_task", "standby_task", "event_task",
               "elution_scope")

#: The keys naming a catalog task. An unresolvable name already raises
#: ``PlanCompileError`` at compile while the plan is enabled (``_require_task``),
#: so this codec checks only that a name was written at all and leaves *which*
#: names exist to the catalog that owns them.
_PIEZO_TASK_KEYS = ("on_task", "off_task", "standby_task", "event_task")

#: The two scopes the engine actually branches on (``deposition_recipe.py``:
#: ``elution_scope == "all_elution"``, else deposit-only).
#: ``PiezoPlan.elution_scope`` is a plain ``str`` with no ``__post_init__``, so a
#: typo is accepted there, misses the ``all_elution`` branch and silently runs the
#: *narrower* scope — actuating less than the file asked for, in the direction
#: nothing goes red. This codec owns the refusal the dataclass does not make.
_ELUTION_SCOPES = ("deposit", "all_elution")


def encode_piezo(value: Any) -> Any:
    """The plan as a ``[piezo]`` table, or :data:`UNREPRESENTABLE`.

    Only what was *chosen*: a field equal to ``PiezoPlan``'s own declared default
    is omitted, so an otherwise-default enabled plan writes ``{"enabled": True}``
    and nothing else. The defaults are read from the dataclass — a
    default-constructed instance — rather than restated, so a task name retuned in
    ``deposition_recipe`` cannot leave a stale copy here deciding what is worth
    writing.
    """
    from softae.core.deposition_recipe import PiezoPlan

    if not isinstance(value, PiezoPlan):
        return UNREPRESENTABLE
    if value.event_params:
        # The identical refusal `campaign_spec_run_plan` makes for
        # `anneal_params`, for the identical reason: a dict-shaped per-run
        # override of an arbitrary named task's parameters gives the decoder no
        # way to tell a typo from a future key, so the file shape has no slot for
        # it. Truthiness rather than `is not None` because the engine itself reads
        # it that way (`if piezo.event_params`) — an empty dict overrides nothing
        # and is therefore written as the absence it already is.
        logger.warning("piezo_not_encodable", reason="event_params")
        return UNREPRESENTABLE
    default = PiezoPlan()
    table: dict[str, Any] = {}
    for key in _PIEZO_KEYS:
        current = getattr(value, key)
        if current != getattr(default, key):
            table[key] = bool(current) if key == "enabled" else str(current)
    return table


def decode_piezo(value: Any) -> Any:
    """A :class:`~softae.core.deposition_recipe.PiezoPlan` from the ``[piezo]``
    table, or ``ValueError``.

    An omitted key is **not given a value here** — the keyword is simply not
    passed, so ``PiezoPlan``'s own dataclass default applies and each default is
    written in exactly one place. An empty ``[piezo]`` table is therefore legal
    and inert, describing the same plan a file omitting the table entirely leaves
    unset.
    """
    from softae.core.deposition_recipe import PiezoPlan

    if not isinstance(value, dict):
        raise ValueError(
            "expected a [piezo] table, e.g. a [piezo] block setting "
            "enabled = true")
    if "event_params" in value:
        raise ValueError(
            "'event_params' cannot be set from a file: it is a per-run override "
            "of the parameters of an arbitrary named catalog task, and a "
            "dict-shaped override gives this decoder no way to tell a typo from "
            "a future key. Name a catalog task carrying the parameters you want "
            "as 'event_task' instead — the same way a custom anneal duration "
            "gets its own task rather than a per-run override.")
    unknown = sorted(set(value) - set(_PIEZO_KEYS))
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown}; valid keys: {sorted(_PIEZO_KEYS)}")

    kwargs: dict[str, Any] = {}
    if "enabled" in value:
        if not isinstance(value["enabled"], bool):
            raise ValueError(
                f"'enabled' must be true or false (got {value['enabled']!r})")
        kwargs["enabled"] = value["enabled"]
    for key in _PIEZO_TASK_KEYS:
        if key not in value:
            continue
        name = value[key]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"'{key}' must be the name of a catalog task (got {name!r})")
        kwargs[key] = name
    if "elution_scope" in value:
        scope = value["elution_scope"]
        if scope not in _ELUTION_SCOPES:
            legal = ", ".join(repr(s) for s in _ELUTION_SCOPES)
            raise ValueError(
                f"unknown elution_scope {scope!r}; legal values are {legal} — "
                f"'deposit' brackets each channel's own deposit phase, "
                f"'all_elution' brackets every elution event including the "
                f"flushes. A scope this codec does not recognise would reach the "
                f"engine, miss the 'all_elution' branch and quietly actuate the "
                f"narrower one.")
        kwargs["elution_scope"] = scope
    return PiezoPlan(**kwargs)


#: The fields this module owns, by spec field name. :mod:`campaign_spec_io`
#: consults it in both directions, so adding a field here is the whole change.
OBJECT_FIELDS: dict[str, FieldCodec] = {
    # The one field whose codec is not written here. It is four nested tables
    # rather than one, so it has a module — :mod:`campaign_spec_run_plan` — and
    # registering it is this line. Built through ``field_codec()`` rather than
    # named directly so that module never has to import this one at module
    # level: the dependency runs one way, and the import order of the pair
    # cannot become load-bearing.
    "run_plan": _run_plan_codec(),
    # The campaign-level baseline (T11.28): the same `PhaseSetpoints` a phase
    # carries, decoded by the same module, minus the approach bands and timeouts
    # — nothing waits for a baseline, so a number that only a wait reads would
    # have no reader. Built through `baseline_conditions_codec()` for the same
    # import-order reason as `run_plan` above.
    "conditions": _baseline_conditions_codec(),
    # Named condition sets (T11.28b): a table of tables, decoded by the same
    # module, with the phase key set rather than the baseline's -- a named set is
    # only ever a PHASE's conditions, so an approach band or timeout on one has a
    # reader. Registered here for the ENCODE half above all: the decode is also
    # reached by `spec_from_dict`'s cross-key pre-pass (the top-level placement's
    # cost), but only this entry gives `spec_to_dict` a way to write the field
    # back out as TOML.
    "condition_sets": _condition_sets_codec(),
    "general_formulation": FieldCodec(
        encode_general_formulation, decode_general_formulation,
        "a composition context whose targets are a Python callable rather than "
        "declared composition axes",
    ),
    "prior_mean": FieldCodec(
        encode_prior_mean, decode_prior_mean,
        "a Python callable that is not one of the built-in prior means a file "
        "can name",
    ),
    "seed_observations": FieldCodec(
        encode_seed_observations, decode_seed_observations,
        "a list of (params, value) pairs this file cannot encode",
    ),
    # The last field `campaign_spec_io._UNSUPPORTED` refused outright. Six flat
    # primitives, so its codec is above rather than in a module of its own.
    "piezo": FieldCodec(encode_piezo, decode_piezo, PIEZO_WHY_NOT),
}
