"""The modality-agnostic measurement block of a campaign spec (Tier 2, T2.4).

:class:`CampaignSpec` has said what to *measure* in three EIS-shaped fields —
``eis_preset``, ``eis_overrides``, ``measure_eis`` — which is the fourth of the
four EIS seams named in ``afl_comparison_and_restructuring_spec.md`` §3.2. This
module holds the replacement: one block that names a **modality** alongside its
preset and overrides, so a second modality needs no new spec fields.

**The three old fields are a Transitional shim, deprecated at birth.** They are
still accepted everywhere they were accepted before — constructor, TOML loader,
GUI — and are canonicalized *into* the block at construction time, so there is
exactly one authority at run time. Named removal condition, per the spec's shim
policy: **after one full campaign runs from a measurement-block spec.**

Canonicalization refuses rather than resolves. If a caller supplies both
spellings and they disagree, one of the two descriptions would have run a
different experiment from the one it asked for, and no precedence rule can pick
the right one — so :func:`canonicalize_measurement` raises.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_MODALITY",
    "DEFAULT_PRESET",
    "LEGACY_UNSET",
    "MeasurementSpec",
    "canonicalize_measurement",
    "measurement_identity",
]

#: The one modality that exists today. Also the value that contributes *nothing*
#: to the resume fingerprint — see :func:`measurement_identity`.
DEFAULT_MODALITY = "eis"

DEFAULT_PRESET = "Quick"


class _LegacyUnset:
    """Sentinel for "the caller did not supply this legacy field".

    ``None`` cannot serve: ``measure_eis=None`` and ``eis_overrides=None`` are
    values a caller might plausibly pass, and the conflict rule has to be able to
    tell "not supplied" from "supplied as empty".
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"

    def __bool__(self) -> bool:
        return False


LEGACY_UNSET = _LegacyUnset()


@dataclass(frozen=True)
class MeasurementSpec:
    """What a campaign measures, and how — independent of which instrument.

    ``modality`` is the plug-in point Tier 2 exists to create: the step
    builders, router, and objective extractors for a modality are looked up by
    this string (the registry itself is T2.5). ``preset`` and ``overrides`` are
    the modality's own settings vocabulary — for EIS, a ``[eis_presets.*]``
    section name and the ``EISParams`` keys that override it.

    ``enabled=False`` means *formulate and cast, but do not measure* — the old
    ``measure_eis=False``. It is a property of the campaign, not of the
    hardware, so it lives here rather than in a driver.
    """

    modality: str = DEFAULT_MODALITY
    preset: str = DEFAULT_PRESET
    overrides: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.modality or not isinstance(self.modality, str):
            raise ValueError("MeasurementSpec.modality must be a non-empty string")
        if not isinstance(self.preset, str):
            raise ValueError("MeasurementSpec.preset must be a string")
        if self.overrides is None:
            object.__setattr__(self, "overrides", {})
        elif not isinstance(self.overrides, dict):
            raise ValueError("MeasurementSpec.overrides must be a table/dict")
        else:
            # Copy: the block is frozen, and a caller holding the original dict
            # could otherwise mutate a "frozen" spec after it was fingerprinted.
            object.__setattr__(self, "overrides", dict(self.overrides))
        object.__setattr__(self, "enabled", bool(self.enabled))

    # ── Serialization ────────────────────────────────────────────────────────

    def as_dict(self) -> dict[str, Any]:
        """Plain-data form for TOML round-trip and the checkpoint snapshot."""
        return {
            "modality": self.modality,
            "preset": self.preset,
            "overrides": dict(self.overrides),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MeasurementSpec":
        """Build from a ``[measurement]`` table.

        An unknown key is an error for the same reason it is in the spec loader:
        a misspelled one would silently take its default, and the caller would
        run something other than what the file says.
        """
        if not isinstance(data, dict):
            raise ValueError("a measurement block must be a table")
        known = {"modality", "preset", "overrides", "enabled"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown measurement key(s) {unknown}; valid keys: {sorted(known)}"
            )
        return cls(**{k: v for k, v in data.items()})


def measurement_identity(measurement: "MeasurementSpec | None") -> dict[str, Any] | None:
    """The measurement block's contribution to the resume fingerprint.

    Returns ``None`` — *contribute nothing* — whenever the block is expressible
    in the retired ``eis_*`` fields, i.e. whenever ``modality`` is the default.
    That omission is the whole point, and it buys two properties at once:

    1. **A legacy spec and its new-form equivalent fingerprint identically**,
       because neither adds a key.
    2. **A checkpoint written before T2.4 still verifies against the same spec
       loaded after it.** The identity payload for every spec that could exist
       before this change is byte-identical to what it was, so no in-flight
       campaign becomes unresumable. This is why the contribution is omitted
       rather than defaulted: adding ``{"modality": "eis"}`` would change the
       hashed JSON of every spec in existence.

    Only ``modality`` is identity-bearing. ``preset`` / ``overrides`` /
    ``enabled`` were never part of the fingerprint when they were spelled
    ``eis_preset`` / ``eis_overrides`` / ``measure_eis`` — they are measurement
    *settings*, re-tunable between sessions like rates and dwells — and making
    them identity for a new modality but not for EIS would be incoherent.
    Changing the modality, by contrast, is a different experiment.
    """
    if measurement is None or measurement.modality == DEFAULT_MODALITY:
        return None
    return {"modality": measurement.modality}


def canonicalize_measurement(
    *,
    measurement: "MeasurementSpec | dict[str, Any] | None" = None,
    eis_preset: Any = LEGACY_UNSET,
    eis_overrides: Any = LEGACY_UNSET,
    measure_eis: Any = LEGACY_UNSET,
    owner: str = "CampaignSpec",
) -> MeasurementSpec:
    """Fold the legacy ``eis_*`` spelling and the new block into one block.

    Four cases:

    * **neither supplied** → the default block (today's behaviour exactly).
    * **only legacy** → the block those fields describe, plus a
      :class:`DeprecationWarning`.
    * **only new** → the block, untouched.
    * **both** → they must agree, field by field. Disagreement raises
      :class:`ValueError`. Silent precedence is refused deliberately: whichever
      spelling lost would have been a written instruction to run a different
      measurement, and neither the file nor the caller would show that it had
      been overruled.
    """
    if isinstance(measurement, dict):
        measurement = MeasurementSpec.from_dict(measurement)
    if measurement is not None and not isinstance(measurement, MeasurementSpec):
        raise ValueError(
            f"{owner}.measurement must be a MeasurementSpec (got "
            f"{type(measurement).__name__})"
        )

    legacy = {
        "preset": ("eis_preset", eis_preset),
        "overrides": ("eis_overrides", eis_overrides),
        "enabled": ("measure_eis", measure_eis),
    }
    supplied = {
        attr: (old_name, value)
        for attr, (old_name, value) in legacy.items()
        if not isinstance(value, _LegacyUnset)
    }

    if measurement is None:
        if not supplied:
            return MeasurementSpec()
        block = MeasurementSpec(
            modality=DEFAULT_MODALITY,
            preset=DEFAULT_PRESET if isinstance(eis_preset, _LegacyUnset) else eis_preset,
            overrides={} if isinstance(eis_overrides, _LegacyUnset) else eis_overrides,
            enabled=True if isinstance(measure_eis, _LegacyUnset) else measure_eis,
        )
        warnings.warn(
            f"{owner}: {sorted(name for name, _ in supplied.values())} are "
            f"deprecated; use the measurement block instead, e.g. "
            f"measurement=MeasurementSpec(preset={block.preset!r}). The old "
            f"fields still work and are canonicalized into the block; they are "
            f"removed after one full campaign runs from a measurement-block spec.",
            DeprecationWarning,
            # 4 frames up: here → CampaignSpec.__post_init__ → the dataclass's
            # synthesised __init__ → the code that actually wrote the spec, which
            # is the only frame the reader can do anything about.
            stacklevel=4,
        )
        return block

    disagree = []
    for attr, (old_name, value) in supplied.items():
        new_value = getattr(measurement, attr)
        if attr == "enabled":
            value = bool(value)
        elif attr == "overrides":
            value = dict(value or {})
        if value != new_value:
            disagree.append(f"{old_name}={value!r} vs measurement.{attr}={new_value!r}")
    if disagree:
        raise ValueError(
            f"{owner}: the legacy eis_* fields and the measurement block disagree "
            f"({'; '.join(disagree)}). Neither can be preferred silently — running "
            f"either one would contradict the other description of this campaign. "
            f"Supply one spelling, not both (use CampaignSpec.with_measurement() to "
            f"replace the block on an existing spec)."
        )
    return measurement
