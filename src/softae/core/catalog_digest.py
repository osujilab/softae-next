"""Stable digests over the task, chemical and solution catalogs.

A campaign's behaviour depends on the catalogs it compiled against as much as on
its spec, but nothing in a run record says which catalogs those were. These
digests give a run one hex string per catalog plus one over all three, so a
later reader can say whether the catalogs moved and, if they did, **which** one.

The canonical form is sorted-key JSON built from each catalog's own name-sorted
listing, so insertion order cannot move a digest while any parameter change
must. Values JSON cannot carry exactly (non-finite floats, unknown object
types) are replaced by explicitly tagged markers rather than by silence, so a
reader can tell a recorded value from an unrepresentable one.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from pathlib import PurePath
from typing import Any

#: Bumped whenever the canonical form changes, so two digests are comparable
#: only when they were produced by the same rules.
DIGEST_VERSION = 1

#: Ordered catalog names; also the key order of :func:`catalog_sub_digests`.
CATALOG_NAMES = ("tasks", "chemicals", "solutions")

#: Tag for a float JSON cannot carry (``nan``, ``inf``, ``-inf``).
NONFINITE_KEY = "__nonfinite__"

#: Tag for a value this module declines to represent; carries the type name
#: only, never a ``repr`` (default reprs embed a memory address and would make
#: the digest differ between two identical catalogs).
UNREPRESENTABLE_KEY = "__unrepresentable__"


def _nonfinite_tag(value: float) -> dict[str, str]:
    """Name a non-finite float unambiguously, sign included."""
    if math.isnan(value):
        return {NONFINITE_KEY: "nan"}
    return {NONFINITE_KEY: "inf" if value > 0 else "-inf"}


def _json_text(canonical_value: Any) -> str:
    """The one deterministic text form of an already-canonical value.

    Set members are ordered by this rather than by ``repr``: a default ``repr``
    embeds a memory address, so repr-ordering would make an unrepresentable
    member's position vary between two runs over identical catalogs.
    """
    return json.dumps(
        canonical_value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def canonical(value: Any) -> Any:
    """A JSON-ready copy of *value* that never raises and never varies by run.

    Mappings keep their keys as strings; unknown types become a tagged marker
    naming the type, because a digest that crashed would be worse than one that
    records what it could not read.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _nonfinite_tag(value)
    if isinstance(value, enum.Enum):
        return canonical(value.value)
    if isinstance(value, PurePath):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return canonical(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical(v) for v in value), key=_json_text)
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return {UNREPRESENTABLE_KEY: type(value).__name__}


def canonical_json(value: Any) -> str:
    """Canonicalise *value*, then dump it in the one deterministic text form."""
    return _json_text(canonical(value))


def digest_of(value: Any) -> str:
    """SHA-256 hex digest of *value*'s canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _entries(catalog: Any, attr_name: str) -> dict[str, Any]:
    """Name-sorted ``{name: entry}`` for a catalog exposing ``list_names``/``get``."""
    names = catalog.list_names()
    getter = getattr(catalog, attr_name)
    return {str(name): getter(name) for name in sorted(names, key=str)}


def tasks_canonical(tasks: Any) -> Any:
    """Canonical form of a task catalog, or ``None`` when none was supplied."""
    if tasks is None:
        return None
    if hasattr(tasks, "to_dict"):
        return canonical(tasks.to_dict())
    return canonical(_entries(tasks, "get"))


def chemicals_canonical(chemicals: Any) -> Any:
    """Canonical form of a chemical catalog, or ``None`` when none was supplied.

    Built here from the catalog's public listing: the catalog class offers no
    dict form of its own and this module does not edit it to add one.
    """
    if chemicals is None:
        return None
    return canonical(_entries(chemicals, "get"))


def solutions_canonical(solutions: Any) -> Any:
    """Canonical form of a solution catalog, or ``None`` when none was supplied.

    Same shape and same reason as :func:`chemicals_canonical`; a solution's
    nested components come along through the dataclass walk in :func:`canonical`.
    """
    if solutions is None:
        return None
    return canonical(_entries(solutions, "get"))


def catalogs_canonical(tasks: Any, chemicals: Any, solutions: Any) -> dict[str, Any]:
    """The three canonical catalog forms under :data:`CATALOG_NAMES`."""
    return {
        "tasks": tasks_canonical(tasks),
        "chemicals": chemicals_canonical(chemicals),
        "solutions": solutions_canonical(solutions),
    }


def catalog_sub_digests(tasks: Any, chemicals: Any, solutions: Any) -> dict[str, str]:
    """One digest per catalog, so a mismatch can name which catalog moved.

    An absent catalog digests as ``null`` and an empty one as ``{}``, which are
    different strings — "not supplied" must not read as "supplied and empty".
    """
    forms = catalogs_canonical(tasks, chemicals, solutions)
    return {name: digest_of(forms[name]) for name in CATALOG_NAMES}


def catalog_digest(tasks: Any, chemicals: Any, solutions: Any) -> str:
    """One digest over all three catalogs, derived from their sub-digests.

    Deriving it from the parts rather than from the whole means a catalog
    cannot move without moving both its own sub-digest and this one.
    """
    subs = catalog_sub_digests(tasks, chemicals, solutions)
    return digest_of({"version": DIGEST_VERSION, "catalogs": subs})
