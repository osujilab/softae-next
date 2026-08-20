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

#: Every field of a :class:`~softae.core.composition_axes.CompositionAxis`. All
#: six are always written: an axis with an omitted key would silently take a
#: default (``basis``, ``low``) and search a different composition.
_AXIS_KEYS = ("kind", "a", "b", "low", "high", "basis")

_GF_KEYS = frozenset(
    {"stocks", "pump_assignment", "target_deposition_uL", "axes",
     "budget_uL", "dried_frac"})


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
            "target_deposition_uL": float(value.target_deposition_uL),
            "axes": [
                {"kind": str(ax.kind), "a": str(ax.a), "b": str(ax.b),
                 "low": float(ax.low), "high": float(ax.high),
                 "basis": str(ax.basis)}
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
    missing = [k for k in _AXIS_KEYS if k not in row]
    if missing:
        raise ValueError(
            f"axis #{index} is missing {missing} — every key is written so that "
            f"an omitted one cannot silently take a default and search a "
            f"different composition")
    try:
        return CompositionAxis(
            kind=str(row["kind"]), a=str(row["a"]), b=str(row["b"]),
            low=float(row["low"]), high=float(row["high"]),
            basis=str(row["basis"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"axis #{index}: {exc}") from exc


def decode_general_formulation(value: Any) -> Any:
    from softae.core.autonomous_wiring import GeneralFormulation

    if not isinstance(value, dict):
        raise ValueError("expected a [general_formulation] table")
    unknown = sorted(set(value) - _GF_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}")
    for required in ("stocks", "pump_assignment", "target_deposition_uL", "axes"):
        if required not in value:
            raise ValueError(f"'{required}' is required")

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
    return GeneralFormulation(
        stocks=stocks,
        catalog=chem_catalog,
        pump_assignment=pump_assignment,
        target_deposition_uL=float(value["target_deposition_uL"]),
        budget_uL=(None if value.get("budget_uL") is None
                   else float(value["budget_uL"])),
        dried_frac=(None if dried is None
                    else {str(k): float(v) for k, v in dried.items()}),
        axes=axes,
    )


#: The fields this module owns, by spec field name. :mod:`campaign_spec_io`
#: consults it in both directions, so adding a field here is the whole change.
OBJECT_FIELDS: dict[str, FieldCodec] = {
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
}
