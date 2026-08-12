"""Load a :class:`CampaignSpec` from TOML (P6.1).

The spec has been Python-only, which meant a headless run had no way to say what
it was running. This reads one from a file — and, importantly, **refuses what it
cannot represent faithfully** rather than loading a partial spec that looks
complete.

That refusal is the whole design. A spec carries live Python objects
(``prior_mean`` is an arbitrary callable; ``formulation`` and ``run_plan`` are
rich objects), and a loader that quietly dropped them would hand back a spec that
runs a *different experiment* from the one the file describes — the same failure
the resume path refuses by fingerprint. An unknown key is an error for the same
reason: a typo'd field name would otherwise silently take its default.

Example::

    name = "peo_licl_scan"
    channels = [21, 22, 23, 24]
    pcb_name = "SoftAE_EIS_4Stripe"
    budget = 40
    optimizer = "bayesian"
    two_phase = true

    [parameter_space.vol_p0]
    type = "float"
    low  = 5.0
    high = 30.0

    [measurement]
    modality = "eis"
    preset   = "Quick"
    enabled  = true

    [measurement.overrides]
    npts = 41

The deprecated top-level ``eis_preset`` / ``eis_overrides`` / ``measure_eis``
keys still load (T2.4) and are folded into the same block. Supplying both
spellings with *different* values is an error, not a precedence question.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SpecLoadError(Exception):
    """The file does not describe a runnable campaign."""


#: Fields that cannot round-trip through a file. Naming them explicitly (rather
#: than skipping unknown types) means a file that sets one gets a clear error
#: instead of a spec that silently differs from what it says.
_UNSUPPORTED = {
    "prior_mean": "a Python callable",
    "formulation": "a FormulationContext object",
    "general_formulation": "a GeneralFormulation object",
    "run_plan": "a RunPlan object",
    "piezo": "a PiezoPlan object",
    "seed_observations": "a list of (params, value) pairs",
}

#: Fields the dataclass declares as tuples; TOML gives lists.
_TUPLE_FIELDS = {
    "channels", "vol_params", "pump_ids", "deadvols", "start_flush_uL",
}

#: DEPRECATED spellings of the ``[measurement]`` table (T2.4). Still accepted on
#: read — a file written before the block existed must keep loading — but never
#: *written*, since they are derivable from the canonical block and emitting
#: both would make every round-trip re-raise the deprecation warning.
_LEGACY_MEASUREMENT_FIELDS = ("eis_preset", "eis_overrides", "measure_eis")


def load_campaign_spec(path: "str | Path") -> Any:
    """Read a :class:`CampaignSpec` from a TOML file."""
    from softae.core.autonomous_wiring import CampaignSpec

    p = Path(path)
    if not p.exists():
        raise SpecLoadError(f"No such campaign file: {p}")

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore

    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        raise SpecLoadError(f"Could not parse {p}: {exc}") from exc

    return spec_from_dict(data, source=str(p))


def spec_from_dict(data: dict[str, Any], *, source: str = "<dict>") -> Any:
    """Build a spec from an already-parsed mapping."""
    from softae.core.autonomous_wiring import CampaignSpec

    if not isinstance(data, dict):
        raise SpecLoadError(f"{source}: expected a table at the top level")

    known = {f.name for f in dataclass_fields(CampaignSpec)}
    supplied = dict(data)

    for key, what in _UNSUPPORTED.items():
        if key in supplied:
            raise SpecLoadError(
                f"{source}: '{key}' cannot be set from a file — it is {what}. "
                f"Build the spec in Python, or omit it and accept the default; "
                f"loading it partially would run a different experiment from the "
                f"one this file describes."
            )

    unknown = sorted(set(supplied) - known)
    if unknown:
        raise SpecLoadError(
            f"{source}: unknown field(s) {unknown}. A misspelled field would "
            f"silently take its default, so this is refused rather than ignored. "
            f"Valid fields: {sorted(known)}"
        )

    if "name" not in supplied:
        raise SpecLoadError(f"{source}: 'name' is required")
    if not supplied.get("parameter_space"):
        raise SpecLoadError(
            f"{source}: 'parameter_space' is required — a campaign with nothing "
            f"to search has nothing to optimize."
        )

    for key in _TUPLE_FIELDS:
        if key in supplied and isinstance(supplied[key], list):
            supplied[key] = tuple(supplied[key])

    # `[measurement]` (T2.4). The legacy `eis_*` keys keep working alongside it;
    # the conflict rule lives in CampaignSpec.__post_init__ and surfaces here as
    # the ValueError branch below, so both spellings disagreeing is a spec error
    # rather than a silently-preferred one.
    if "measurement" in supplied:
        from softae.core.measurement_spec import MeasurementSpec

        try:
            supplied["measurement"] = MeasurementSpec.from_dict(
                supplied["measurement"])
        except (TypeError, ValueError) as exc:
            raise SpecLoadError(f"{source}: [measurement]: {exc}") from exc

    try:
        spec = CampaignSpec(**supplied)
    except TypeError as exc:
        raise SpecLoadError(f"{source}: {exc}") from exc
    except ValueError as exc:
        raise SpecLoadError(f"{source}: {exc}") from exc

    _validate_parameter_space(spec.parameter_space, source)
    # `campaign=` rather than `name=`: the key states what kind of identifier it
    # is, matching every other campaign log line in the codebase (T1.7).
    logger.info(
        "campaign_spec_loaded", source=source, campaign=spec.name,
        n_params=len(spec.parameter_space), budget=spec.budget,
        modality=spec.measurement.modality,
    )
    return spec


def _validate_parameter_space(space: dict[str, Any], source: str) -> None:
    """Fail here rather than inside the optimizer, where the error is opaque."""
    for pname, p in (space or {}).items():
        if not isinstance(p, dict) or "type" not in p:
            raise SpecLoadError(
                f"{source}: parameter '{pname}' needs a 'type' "
                f"('float', 'int', or 'categorical')")
        ptype = p["type"]
        if ptype in ("float", "int"):
            if "low" not in p or "high" not in p:
                raise SpecLoadError(
                    f"{source}: numeric parameter '{pname}' needs 'low' and 'high'")
            if float(p["low"]) >= float(p["high"]):
                raise SpecLoadError(
                    f"{source}: parameter '{pname}' has low >= high")
        elif ptype == "categorical":
            if not p.get("choices"):
                raise SpecLoadError(
                    f"{source}: categorical parameter '{pname}' needs 'choices'")
        else:
            raise SpecLoadError(
                f"{source}: parameter '{pname}' has unknown type '{ptype}'")


def spec_to_dict(spec: Any) -> dict[str, Any]:
    """Serialize the file-representable subset of a spec.

    Round-trips through :func:`spec_from_dict`. Fields that cannot be
    represented are **omitted**, so a written file is honest about being the
    representable part rather than pretending to be the whole spec.

    Measurement settings are written **only** as the ``[measurement]`` block
    (T2.4). The deprecated ``eis_*`` mirrors are skipped rather than written
    alongside it: they carry no information the block does not, and a file
    holding both would deprecation-warn on every reload of something this
    function itself produced.
    """
    from softae.core.autonomous_wiring import CampaignSpec
    from softae.core.measurement_spec import MeasurementSpec

    out: dict[str, Any] = {}
    defaults = CampaignSpec(name="_", parameter_space={"_": {"type": "float",
                                                            "low": 0, "high": 1}})
    for f in dataclass_fields(CampaignSpec):
        if f.name in _UNSUPPORTED or f.name in _LEGACY_MEASUREMENT_FIELDS:
            continue
        value = getattr(spec, f.name)
        current_default = getattr(defaults, f.name, object())
        if isinstance(value, MeasurementSpec):
            value = value.as_dict()
            current_default = (current_default.as_dict()
                               if isinstance(current_default, MeasurementSpec)
                               else current_default)
        if isinstance(value, tuple):
            value = list(value)
        if value is None:
            continue
        if isinstance(current_default, tuple):
            current_default = list(current_default)
        if f.name not in ("name", "parameter_space") and value == current_default:
            continue      # keep written files to what was actually chosen
        out[f.name] = value
    return out
