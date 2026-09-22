"""Spec intent against bench reality: which record wins, and what differs.

Every spec here is built in memory against the shipped placeholder catalogs, so
nothing depends on this machine's ``data/`` directory.
"""

from __future__ import annotations

import json

from softae.core.stock_assignment import PumpLoadout
from softae.core.stock_resolution import (
    DECLARED,
    NONE,
    SPEC,
    UNRESOLVED_KEY,
    ResolvedStocks,
    resolve_stocks,
    spec_stock_by_pump,
    stocks_resolved_event,
)
from tests.support.fixture_catalog import use_default_chemistry

#: Three stocks the shipped solution catalog knows, on three pumps.
STOCKS = ["5wt% 20k PEO stock", "10 wt% silica solution", "LiCl 10M"]
PUMPS = {"5wt% 20k PEO stock": 0, "10 wt% silica solution": 1, "LiCl 10M": 2}


def make_spec(monkeypatch, **overrides):
    """A minimal three-stock campaign spec, decoded against the shipped catalogs."""
    from softae.core.campaign_spec_io import spec_from_dict

    use_default_chemistry(monkeypatch)
    data = {
        "name": "resolution_probe",
        "channels": [1, 2],
        "budget": 2,
        "optimizer": "grid",
        "pump_ids": [0, 1, 2],
        "parameter_space": {"replicate": {"type": "int", "low": 1, "high": 2}},
        "general_formulation": {
            "stocks": STOCKS,
            "pump_assignment": dict(PUMPS),
            "target_deposition_uL": 4.5,
            "axes": [
                {"kind": "molar_ratio", "a": "Ethylene oxide",
                 "b": "Lithium chloride", "low": 20.0, "high": 20.0,
                 "basis": "volume"},
            ],
        },
    }
    data.update(overrides)
    return spec_from_dict({k: v for k, v in data.items() if v is not None},
                          source="<test>")


def solutions():
    """The shipped solution catalog, via the seam a test is allowed to use."""
    from softae.catalog_defaults import load_defaults

    return load_defaults().solutions


# ── Which record wins ────────────────────────────────────────────────────────

def test_resolve_stocks_spec_assignment_wins_when_spec_names_stocks(monkeypatch):
    """A campaign is launched from a file, so the file's intent is the answer."""
    spec = make_spec(monkeypatch)
    loadout = PumpLoadout({0: "LiCl 1M", 1: "2 wt% silica solution"})
    resolved = resolve_stocks(spec, loadout, solutions())
    assert resolved.source == SPEC
    assert resolved.by_pump == {0: STOCKS[0], 1: STOCKS[1], 2: STOCKS[2]}


def test_resolve_stocks_without_formulation_is_source_none(monkeypatch):
    """No formulation names no stocks; that is not an empty spec assignment."""
    spec = make_spec(monkeypatch, general_formulation=None,
                     parameter_space={"vol": {"type": "float",
                                              "low": 1.0, "high": 2.0}})
    resolved = resolve_stocks(spec, PumpLoadout({0: "LiCl 1M"}), solutions())
    assert resolved.source == NONE
    assert resolved.by_pump == {} and resolved.by_name == {}
    assert resolved.is_empty


def test_resolve_stocks_empty_spec_assignment_falls_to_the_declared_loadout(
        monkeypatch):
    """A formulation naming no pump leaves the bench declaration as the answer."""
    spec = make_spec(monkeypatch)
    spec.general_formulation.pump_assignment = {}
    resolved = resolve_stocks(spec, PumpLoadout({0: "LiCl 1M"}), solutions())
    assert resolved.source == DECLARED
    assert resolved.by_pump == {0: "LiCl 1M"}
    assert resolved.by_name == {"LiCl 1M": 0}


# ── The three absences ───────────────────────────────────────────────────────

def test_resolve_stocks_loadout_disagreement_is_reported_not_merged(monkeypatch):
    """Neither side is evidence about the other, so both names are kept."""
    spec = make_spec(monkeypatch)
    loadout = PumpLoadout({0: STOCKS[0], 1: STOCKS[1], 2: "LiCl 1M"})
    resolved = resolve_stocks(spec, loadout, solutions())
    assert resolved.disagreements == {
        2: {"spec": "LiCl 10M", "declared": "LiCl 1M"}}
    assert resolved.by_pump[2] == "LiCl 10M"
    assert resolved.undeclared == () and resolved.unused_declared == ()


def test_resolve_stocks_undeclared_pump_is_distinct_from_disagreement(monkeypatch):
    """Nothing declared on a pump the spec uses is its own answer, not a clash."""
    spec = make_spec(monkeypatch)
    resolved = resolve_stocks(
        spec, PumpLoadout({0: STOCKS[0], 1: STOCKS[1]}), solutions())
    assert resolved.undeclared == (2,)
    assert resolved.disagreements == {}


def test_resolve_stocks_declared_pump_the_spec_ignores_is_its_own_answer(
        monkeypatch):
    """A syringe this campaign does not draw from is ordinary, not missing."""
    spec = make_spec(monkeypatch)
    loadout = PumpLoadout({**dict(zip((0, 1, 2), STOCKS)), 5: "LiCl 1M"})
    resolved = resolve_stocks(spec, loadout, solutions())
    assert resolved.unused_declared == (5,)
    assert resolved.undeclared == () and resolved.disagreements == {}


def test_resolve_stocks_without_a_loadout_reports_every_pump_undeclared(
        monkeypatch):
    """No store is not agreement: every assigned pump reads as undeclared."""
    resolved = resolve_stocks(make_spec(monkeypatch), None, solutions())
    assert resolved.undeclared == (0, 1, 2)
    assert resolved.source == SPEC


# ── The catalog snapshot ─────────────────────────────────────────────────────

def test_resolve_stocks_catalog_rows_snapshot_every_resolved_stock(monkeypatch):
    """The rows travel with the run so a moved catalog cannot rewrite history."""
    resolved = resolve_stocks(make_spec(monkeypatch), None, solutions())
    assert set(resolved.catalog_rows) == set(STOCKS)
    assert resolved.catalog_rows[STOCKS[0]]["components"]


def test_resolve_stocks_absent_catalog_marks_rows_unresolved(monkeypatch):
    """"Nothing to look it up in" must not be spelled as a clean empty row."""
    resolved = resolve_stocks(make_spec(monkeypatch), None, None)
    assert all(UNRESOLVED_KEY in row for row in resolved.catalog_rows.values())


def test_resolve_stocks_uncatalogued_stock_is_marked_not_omitted(monkeypatch):
    """A name the catalog does not carry is recorded as such, never dropped."""
    spec = make_spec(monkeypatch)
    spec.general_formulation.pump_assignment = {}
    resolved = resolve_stocks(spec, PumpLoadout({0: "Ghost solution"}),
                              solutions())
    assert resolved.catalog_rows["Ghost solution"][UNRESOLVED_KEY]


# ── The record ───────────────────────────────────────────────────────────────

def test_to_record_round_trips_through_json(monkeypatch):
    """Provenance is JSON on disk, so the record must survive a dump and load."""
    loadout = PumpLoadout({0: STOCKS[0], 2: "LiCl 1M", 5: "LiCl 1M"})
    record = resolve_stocks(make_spec(monkeypatch), loadout,
                            solutions()).to_record()
    reloaded = json.loads(json.dumps(record, allow_nan=False))
    assert reloaded["by_pump"] == {"0": STOCKS[0], "1": STOCKS[1], "2": STOCKS[2]}
    assert reloaded["by_name"][STOCKS[2]] == 2
    assert reloaded["disagreements"] == {
        "2": {"spec": "LiCl 10M", "declared": "LiCl 1M"}}
    assert reloaded["undeclared"] == [1] and reloaded["unused_declared"] == [5]


def test_to_record_keeps_both_directions_for_the_same_assignment(monkeypatch):
    """The pump index is the physical fact and the name its interpretation."""
    record = resolve_stocks(make_spec(monkeypatch), None, solutions()).to_record()
    assert record["by_pump"] == {"0": STOCKS[0], "1": STOCKS[1], "2": STOCKS[2]}
    assert record["by_name"] == {STOCKS[0]: 0, STOCKS[1]: 1, STOCKS[2]: 2}


# ── Two stocks on one pump ───────────────────────────────────────────────────

def test_spec_stock_by_pump_names_both_stocks_sharing_a_pump(monkeypatch):
    """``pump_assignment`` is name → pump, so a shared line must say both names."""
    spec = make_spec(monkeypatch)
    spec.general_formulation.pump_assignment = {STOCKS[0]: 0, STOCKS[1]: 0}
    inverted = spec_stock_by_pump(spec)
    assert set(inverted) == {0}
    assert STOCKS[0] in inverted[0] and STOCKS[1] in inverted[0]


# ── The event payload ────────────────────────────────────────────────────────

def test_stocks_resolved_event_drops_the_catalog_rows(monkeypatch):
    """The rows are bulky and static; provenance holds them beside the stream."""
    resolved = resolve_stocks(make_spec(monkeypatch), None, solutions())
    payload = stocks_resolved_event(resolved)
    assert "catalog_rows" not in payload
    assert payload["source"] == SPEC and payload["by_pump"]["0"] == STOCKS[0]


def test_stocks_resolved_event_is_json_serialisable():
    """The event stream is JSON lines, so the payload must dump without a coder."""
    payload = stocks_resolved_event(ResolvedStocks(source=NONE))
    assert json.loads(json.dumps(payload, allow_nan=False))["source"] == NONE
