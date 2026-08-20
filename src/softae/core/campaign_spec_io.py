"""Load a :class:`CampaignSpec` from TOML (P6.1).

The spec has been Python-only, which meant a headless run had no way to say what
it was running. This reads one from a file — and, importantly, **refuses what it
cannot represent faithfully** rather than loading a partial spec that looks
complete.

That refusal is the whole design. A spec carries live Python objects
(``formulation`` and ``run_plan`` are rich objects), and a loader that quietly
dropped them would hand back a spec that runs a *different experiment* from the
one the file describes — the same failure the resume path refuses by fingerprint.
An unknown key is an error for the same reason: a typo'd field name would
otherwise silently take its default.

**Refusing is not free, though, and three fields were being refused wrongly.**
Since a campaign runs in a detached child started *from a file*, a field a file
cannot carry is a field that cannot be run at all — and ``general_formulation``,
``prior_mean`` and ``seed_observations`` are exactly what the Live BO tab's
composition mode and its Prior-informed group box set. Each of those turned out
to be representable once asked the right question (declared axes rather than a
callable; a registry *name* rather than a function object; primitives already),
so they now cross the boundary through :mod:`softae.core.campaign_spec_fields`.
Everything genuinely unrepresentable still raises.

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

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import structlog

from softae.core.campaign_spec_fields import OBJECT_FIELDS, UNREPRESENTABLE

logger = structlog.get_logger(__name__)


class SpecLoadError(Exception):
    """The file does not describe a runnable campaign."""


#: Fields that cannot round-trip through a file **at all**. Naming them
#: explicitly (rather than skipping unknown types) means a file that sets one
#: gets a clear error instead of a spec that silently differs from what it says.
#:
#: Contrast :data:`~softae.core.campaign_spec_fields.OBJECT_FIELDS`, which are
#: *conditionally* representable: those load, and are reported field-by-field by
#: :func:`spec_toml_completeness` when a particular value cannot be written.
_UNSUPPORTED = {
    "formulation": "a FormulationContext object",
    "run_plan": "a RunPlan object",
    "piezo": "a PiezoPlan object",
}

#: Top-level array naming fields the file sets to **nothing**.
#:
#: TOML has no null, and absence already means "take the default" — which for
#: ``rh_stability_pct`` is the *opposite* of what ``None`` means (``None``
#: switches the RH gate off; the default switches it back on) and for ``seed``
#: turns an unseeded campaign into a seeded one. Dropping a ``None`` on the way
#: out was therefore a gate silently re-enabling itself across a round trip, so
#: an explicit nothing is written down rather than omitted.
_EXPLICIT_NONE_KEY = "explicit_none"

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
    nulls = _pop_explicit_none(supplied, known, source)

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

    # Live objects rebuilt from what the file *names* (S5.K). A decode that
    # cannot mean what the file says raises here rather than substituting a
    # default, for the reason the unknown-key check exists one block above.
    for key, codec in OBJECT_FIELDS.items():
        if key in supplied:
            try:
                supplied[key] = codec.decode(supplied[key])
            except (TypeError, ValueError) as exc:
                raise SpecLoadError(f"{source}: '{key}': {exc}") from exc

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

    # Applied last: a name in `explicit_none` means the field is set to nothing,
    # so it must not then be handed to a decoder or a tuple coercion.
    for key in nulls:
        supplied[key] = None

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


def _pop_explicit_none(
    supplied: dict[str, Any], known: set[str], source: str
) -> tuple[str, ...]:
    """Take :data:`_EXPLICIT_NONE_KEY` off *supplied* and validate it.

    Popped before the unknown-field check, since it is a directive about fields
    rather than a field itself. A name that is both listed here and given a value
    is refused rather than resolved by precedence: the file says two things about
    one field, and picking either would be a guess.
    """
    raw = supplied.pop(_EXPLICIT_NONE_KEY, None)
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
        raise SpecLoadError(
            f"{source}: '{_EXPLICIT_NONE_KEY}' must be an array of field names")
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SpecLoadError(
            f"{source}: '{_EXPLICIT_NONE_KEY}' names unknown field(s) {unknown}")
    both = sorted(set(raw) & set(supplied))
    if both:
        raise SpecLoadError(
            f"{source}: {both} are given a value *and* listed in "
            f"'{_EXPLICIT_NONE_KEY}' — the file says two things about the same "
            f"field, and neither can be preferred silently")
    return tuple(raw)


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

    An **explicit** ``None`` on a field whose default is not ``None`` is written
    to :data:`_EXPLICIT_NONE_KEY` rather than dropped. Dropping it (which this
    used to do, by testing for ``None`` *before* comparing against the default)
    meant ``seed = None`` reloaded as ``42`` and a deliberately disabled RH gate
    reloaded enabled — a gate silently re-enabling itself across a round trip.
    """
    from softae.core.autonomous_wiring import CampaignSpec

    out: dict[str, Any] = {}
    nulls: list[str] = []
    defaults = _default_spec()
    for f in dataclass_fields(CampaignSpec):
        if f.name in _UNSUPPORTED or f.name in _LEGACY_MEASUREMENT_FIELDS:
            continue
        value = getattr(spec, f.name)
        if f.name in ("name", "parameter_space"):
            out[f.name] = _writable(value)      # always written
            continue
        if _is_default(value, getattr(defaults, f.name, object())):
            continue      # keep written files to what was actually chosen
        if value is None:
            nulls.append(f.name)                # and the default is not None
            continue
        encoded = _encode(f.name, value)
        if encoded is not UNREPRESENTABLE:
            out[f.name] = encoded
    if nulls:
        out[_EXPLICIT_NONE_KEY] = sorted(nulls)
    return out


# ── Is the written part the whole spec? ──────────────────────────────────────
#
# `spec_to_dict` is honest about being *a* part of the spec, and the loader is
# loud about what it refuses — but the two are asymmetric, and dangerously so:
# *reading* a file that sets `run_plan` raises by explicit design
# (`_UNSUPPORTED`), while *writing* a spec that has one omits it in silence. A
# caller that writes a file and then hands back the command to run it would hand
# back a command that runs a **different experiment** and raises nothing.
#
# The asymmetry survives `general_formulation` becoming representable, and is if
# anything sharper for it: a composition context is now written *when it declares
# its axes* and omitted when it carries only a callable — so "did the write keep
# it?" is a per-value question, not a per-field one, and only this check answers
# it. A spec whose axes went unwritten reloads with neither `general_formulation`
# nor `vol_params`, and `resolved_vol_params()` reads its axes as raw µL volumes.
#
# So a file may only stand in for a spec once something has *proved* it carries
# the whole thing. That proof is here, next to the writer it checks, rather than
# in each caller.


@dataclass(frozen=True)
class SpecCompleteness:
    """Whether a TOML file could stand in for a spec, and what it would drop."""

    complete: bool
    #: Fields this spec sets that a written file would not carry.
    missing: tuple[str, ...] = ()
    #: One operator-facing sentence per defect, naming the field and the reason.
    reasons: tuple[str, ...] = ()

    def explain(self) -> str:
        """The reasons as prose, or a statement that there are none."""
        if self.complete:
            return "Every setting this campaign uses can be written to a spec file."
        return "\n".join(f"  - {r}" for r in self.reasons)


def _default_spec() -> Any:
    """A spec holding nothing but the shipped defaults, to compare against."""
    from softae.core.autonomous_wiring import CampaignSpec

    return CampaignSpec(name="_", parameter_space={"_": {"type": "float",
                                                         "low": 0, "high": 1}})


def _writable(value: Any) -> Any:
    """*value* in the shape :func:`spec_to_dict` would write it.

    Normalising here rather than at each comparison is what makes "was it
    written?" and "is it non-default?" answers to the same question; comparing a
    tuple against the list the writer emits would report every tuple field as a
    silent loss.
    """
    from softae.core.measurement_spec import MeasurementSpec

    if isinstance(value, MeasurementSpec):
        return value.as_dict()
    if isinstance(value, tuple):
        return list(value)
    return value


def _encode(name: str, value: Any) -> Any:
    """*value* as the file carries it, or :data:`UNREPRESENTABLE`."""
    codec = OBJECT_FIELDS.get(name)
    return codec.encode(value) if codec is not None else _writable(value)


def _is_default(value: Any, default: Any) -> bool:
    """Whether the spec left this field alone. An unanswerable compare is *not*."""
    try:
        return bool(_writable(value) == _writable(default))
    except Exception:
        return False      # an answer we cannot get is not "unchanged"


def _chosen_fields(spec: Any) -> tuple[str, ...]:
    """Fields set to something other than the shipped default.

    ``name`` and ``parameter_space`` are excluded because the writer emits them
    unconditionally — there is nothing to prove about a field that is always
    written.
    """
    from softae.core.autonomous_wiring import CampaignSpec

    defaults = _default_spec()
    return tuple(
        f.name for f in dataclass_fields(CampaignSpec)
        if f.name not in _LEGACY_MEASUREMENT_FIELDS
        and f.name not in ("name", "parameter_space")
        and not _is_default(getattr(spec, f.name),
                            getattr(defaults, f.name, object()))
    )


def _written_fields(written: dict[str, Any]) -> set[str]:
    """Which spec fields *written* actually carries.

    A field set to an explicit nothing is carried by name inside
    :data:`_EXPLICIT_NONE_KEY` rather than as a key of its own, so reading the
    top-level keys alone would report every deliberate ``None`` as a silent loss
    — the very thing that key exists to stop being one.
    """
    names = set(written) - {_EXPLICIT_NONE_KEY}
    return names | set(written.get(_EXPLICIT_NONE_KEY, ()))


def _why_missing(spec: Any, name: str) -> str:
    """One sentence naming why a chosen field would not survive a write."""
    if name in _UNSUPPORTED:
        return (f"{name} is {_UNSUPPORTED[name]} and cannot be written to a "
                f"file at all")
    if name in OBJECT_FIELDS:
        return f"{name} is {OBJECT_FIELDS[name].why_not}"
    return f"{name} would not be written"


def spec_toml_completeness(spec: Any) -> SpecCompleteness:
    """Whether ``spec_to_dict(spec)`` is the *whole* spec rather than part of it.

    Three questions, and a file has to pass all three before anything may offer
    it as a stand-in for what is on screen:

    1. **Coverage** — is every field this spec chose actually written? This is
       what catches a value none of the codecs can name: a ``general_formulation``
       carrying a ``build_targets`` callable instead of declared axes, a
       ``prior_mean`` that is not a built-in, a ``run_plan``.
    2. **Encodability** — is the written part valid TOML? Nothing else checks
       this, so an unencodable value would surface as a traceback at write time.
    3. **Round trip** — does reloading the file write the same file back? That
       is the end-to-end statement, and the only one that survives a future
       change to either half.

    Never raises: every failure is a reason, because the caller is usually on a
    path where something has already gone wrong.
    """
    written = spec_to_dict(spec)
    carried = _written_fields(written)
    missing = tuple(f for f in _chosen_fields(spec) if f not in carried)
    reasons = [_why_missing(spec, f) for f in missing]

    text: str | None = None
    try:
        import tomli_w

        text = tomli_w.dumps(written)
    except Exception as exc:
        reasons.append(f"the representable part is not valid TOML: {exc}")

    if text is not None:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore
        try:
            if spec_to_dict(spec_from_dict(tomllib.loads(text))) != written:
                reasons.append("the file does not reload to what was written")
        except Exception as exc:
            reasons.append(f"the file would not reload: {exc}")

    return SpecCompleteness(complete=not reasons, missing=missing,
                            reasons=tuple(reasons))


def write_campaign_spec_toml(spec: Any, path: "str | Path") -> Path:
    """Write the representable part of *spec* to *path*.

    **Not a substitute for :func:`spec_toml_completeness`.** This writes what
    can be written; only the check knows whether that is everything. Callers
    that offer the file as a way to re-run the campaign must ask first.
    """
    import tomli_w

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        tomli_w.dump(spec_to_dict(spec), fh)
    logger.info("campaign_spec_written", path=str(p), campaign=spec.name)
    return p
