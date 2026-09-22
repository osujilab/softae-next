"""Digests over the task, chemical and solution catalogs.

Covers order-independence, sensitivity to a single parameter, which sub-digest
names a change, absent-versus-empty chemistry, and the tagged forms used for
values JSON cannot carry.
"""

from __future__ import annotations

import dataclasses
import json
import math

from softae.core.catalog_digest import (
    CATALOG_NAMES,
    NONFINITE_KEY,
    UNREPRESENTABLE_KEY,
    canonical,
    canonical_json,
    catalog_digest,
    catalog_sub_digests,
    digest_of,
)
from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.core.task_catalog import Task, TaskCatalog


# ── In-memory catalogs (no data root, no files) ──────────────────────────────

def _task(name: str, **params: object) -> Task:
    return Task(name=name, instrument="pump", method="dispense",
                params=dict(params) or {"volume_uL": 5.0})


def _tasks(*names: str) -> TaskCatalog:
    cat = TaskCatalog()
    for name in names:
        cat.add(_task(name))
    return cat


def _chemicals() -> ChemicalCatalog:
    cat = ChemicalCatalog()
    cat.add(Chemical(name="water", density_g_per_mL=1.0))
    cat.add(Chemical(name="peo", density_g_per_mL=1.21,
                     molar_mass_g_per_mol=44.05))
    return cat


def _solutions() -> SolutionCatalog:
    cat = SolutionCatalog()
    cat.add(Solution(name="peo_stock", components=[
        SolutionComponent(chemical_name="peo", role="solute",
                          quantity=0.5, unit="g"),
        SolutionComponent(chemical_name="water", role="solvent",
                          quantity=10.0, unit="mL"),
    ]))
    return cat


# ── Order-independence ───────────────────────────────────────────────────────

def test_catalog_digest_task_insertion_order_same_digest() -> None:
    """Two catalogs holding the same tasks digest alike however they were filled."""
    forward = _tasks("a_flush", "b_cast", "c_anneal")
    reverse = _tasks("c_anneal", "b_cast", "a_flush")
    assert catalog_digest(forward, None, None) == catalog_digest(reverse, None, None)


def test_catalog_digest_chemical_insertion_order_same_digest() -> None:
    """Chemistry digests by name, not by the order rows arrived in."""
    first, second = ChemicalCatalog(), ChemicalCatalog()
    water = Chemical(name="water", density_g_per_mL=1.0)
    peo = Chemical(name="peo", density_g_per_mL=1.21)
    first.add(water)
    first.add(peo)
    second.add(peo)
    second.add(water)
    assert catalog_digest(None, first, None) == catalog_digest(None, second, None)


def test_canonical_dict_key_order_same_json() -> None:
    """Parameter dicts canonicalise by sorted key, so a re-ordered dict is equal."""
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


# ── Sensitivity, and which catalog moved ─────────────────────────────────────

def test_catalog_digest_task_param_change_moves_digest() -> None:
    """One parameter changing must move the digest, or it records nothing."""
    before = _tasks("cast")
    after = TaskCatalog()
    after.add(_task("cast", volume_uL=6.0))
    assert catalog_digest(before, None, None) != catalog_digest(after, None, None)


def test_catalog_sub_digests_task_change_moves_only_tasks() -> None:
    """A task edit names the task catalog and leaves the chemistry ones alone."""
    chems, sols = _chemicals(), _solutions()
    before = catalog_sub_digests(_tasks("cast"), chems, sols)
    changed = TaskCatalog()
    changed.add(_task("cast", volume_uL=6.0))
    after = catalog_sub_digests(changed, chems, sols)
    assert before["tasks"] != after["tasks"]
    assert before["chemicals"] == after["chemicals"]
    assert before["solutions"] == after["solutions"]


def test_catalog_sub_digests_chemical_change_moves_only_chemicals() -> None:
    """A density edit names the chemical catalog and nothing else."""
    tasks, sols = _tasks("cast"), _solutions()
    before = catalog_sub_digests(tasks, _chemicals(), sols)
    changed = _chemicals()
    changed.add(Chemical(name="water", density_g_per_mL=0.998))
    after = catalog_sub_digests(tasks, changed, sols)
    assert before["chemicals"] != after["chemicals"]
    assert before["tasks"] == after["tasks"]
    assert before["solutions"] == after["solutions"]


def test_catalog_sub_digests_solution_component_change_moves_only_solutions() -> None:
    """A component quantity is inside the solution catalog's digest, not beside it."""
    tasks, chems = _tasks("cast"), _chemicals()
    before = catalog_sub_digests(tasks, chems, _solutions())
    changed = _solutions()
    changed.get("peo_stock").components[0].quantity = 0.75
    after = catalog_sub_digests(tasks, chems, changed)
    assert before["solutions"] != after["solutions"]
    assert before["tasks"] == after["tasks"]
    assert before["chemicals"] == after["chemicals"]


def test_catalog_digest_moves_when_any_sub_digest_moves() -> None:
    """The combined digest is derived from the parts, so no part can move alone."""
    tasks, chems = _tasks("cast"), _chemicals()
    before = catalog_digest(tasks, chems, _solutions())
    changed = _solutions()
    changed.get("peo_stock").components[0].quantity = 0.75
    assert catalog_digest(tasks, chems, changed) != before


def test_catalog_sub_digests_keys_are_the_three_catalogs() -> None:
    """The sub-digest mapping names every catalog, so a reader can point at one."""
    subs = catalog_sub_digests(_tasks("cast"), _chemicals(), _solutions())
    assert tuple(subs) == CATALOG_NAMES


# ── Absent chemistry ─────────────────────────────────────────────────────────

def test_catalog_digest_absent_chemistry_is_accepted() -> None:
    """A spec with no formulation supplies no chemistry; that is not an error."""
    assert isinstance(catalog_digest(_tasks("cast"), None, None), str)


def test_catalog_sub_digests_absent_differs_from_empty() -> None:
    """"Not supplied" must not digest the same as "supplied and empty"."""
    absent = catalog_sub_digests(_tasks("cast"), None, None)
    empty = catalog_sub_digests(_tasks("cast"), ChemicalCatalog(), SolutionCatalog())
    assert absent["chemicals"] != empty["chemicals"]
    assert absent["solutions"] != empty["solutions"]
    assert absent["tasks"] == empty["tasks"]


# ── Values JSON cannot carry ─────────────────────────────────────────────────

def test_canonical_nan_becomes_tagged_marker() -> None:
    """A non-finite float is recorded as a named marker, never dropped."""
    assert canonical(float("nan")) == {NONFINITE_KEY: "nan"}


def test_canonical_infinities_keep_their_sign() -> None:
    """Positive and negative infinity are distinguishable in the record."""
    assert canonical(math.inf) == {NONFINITE_KEY: "inf"}
    assert canonical(-math.inf) == {NONFINITE_KEY: "-inf"}


def test_canonical_json_nonfinite_param_does_not_raise() -> None:
    """Strict JSON would refuse a bare NaN; the canonical form must not."""
    text = canonical_json({"timeout_s": math.inf, "ok": 1.5})
    assert json.loads(text)["timeout_s"] == {NONFINITE_KEY: "inf"}


def test_digest_of_nan_differs_from_digest_of_infinity() -> None:
    """The tagged forms are distinct, so the two do not collapse to one digest."""
    assert digest_of(float("nan")) != digest_of(math.inf)


def test_catalog_digest_nonfinite_task_param_does_not_raise() -> None:
    """A catalog carrying a non-finite parameter still digests."""
    cat = TaskCatalog()
    cat.add(_task("cast", volume_uL=math.inf))
    assert isinstance(catalog_digest(cat, None, None), str)


def test_canonical_unknown_type_becomes_type_named_marker() -> None:
    """An unrepresentable value records its type, never a repr with an address."""

    class Opaque:
        pass

    encoded = canonical({"driver": Opaque()})
    assert encoded == {"driver": {UNREPRESENTABLE_KEY: "Opaque"}}


def test_digest_of_unknown_type_is_stable_across_instances() -> None:
    """Two instances of one opaque type digest alike; a repr would not."""

    class Opaque:
        pass

    assert digest_of(Opaque()) == digest_of(Opaque())


def test_canonical_json_never_contains_a_memory_address() -> None:
    """Addresses vary per process and would make a run's digest unreproducible."""

    class Opaque:
        pass

    assert "0x" not in canonical_json({"driver": Opaque()})


# ── Sets ─────────────────────────────────────────────────────────────────────

def test_canonical_set_member_order_does_not_reach_the_json() -> None:
    """A set is unordered, so two fillings of one set must canonicalise alike."""
    forward = canonical_json({"tags": {"a", "b", "c", 2, 1}})
    reverse = canonical_json({"tags": {1, 2, "c", "b", "a"}})
    assert forward == reverse


def test_canonical_set_orders_by_what_is_recorded_not_by_repr() -> None:
    """Members sort by their canonical form, so an unrecorded ``repr`` cannot move
    them; a default ``repr`` carries an address and would reorder per run.
    """

    @dataclasses.dataclass(frozen=True)
    class Node:
        tag: str

        def __repr__(self) -> str:  # deliberately the opposite order
            return {"a": "zzz", "z": "aaa"}[self.tag]

    assert canonical({Node("a"), Node("z")}) == [{"tag": "a"}, {"tag": "z"}]


def test_canonical_set_of_an_unknown_type_uses_the_marker_not_a_repr() -> None:
    """An opaque member is recorded and ordered by its marker, address-free."""

    class Opaque:
        pass

    assert canonical({Opaque()}) == [{UNREPRESENTABLE_KEY: "Opaque"}]
    text = canonical_json({"drivers": {Opaque(), "aaa"}})
    assert "0x" not in text
    assert digest_of({Opaque(), "aaa"}) == digest_of({"aaa", Opaque()})
