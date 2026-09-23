"""The launch digest: does it say what the bench holds, and refuse the mismatch?

Every spec here is built in memory against the shipped placeholder catalogs, so
nothing depends on this machine's ``data/`` directory.
"""

from __future__ import annotations

import json

import pytest

from softae.core.final_check import (
    BLOCK,
    OK,
    WARN,
    FinalCheck,
    Finding,
    Section,
    build_final_check,
    confirm_final_check,
    main,
    render_final_check,
)
from tests.support import fixture_catalog
from tests.support.fixture_catalog import use_default_chemistry

#: Bound, not redefined: the fixture lives once in the helper module, and pytest
#: finds a fixture by name in whatever namespace it is bound to.
task_catalog = fixture_catalog.task_catalog

#: Three stocks the shipped solution catalog knows, on three pumps.
STOCKS = ["5wt% 20k PEO stock", "10 wt% silica solution", "LiCl 10M"]
PUMPS = {"5wt% 20k PEO stock": 0, "10 wt% silica solution": 1, "LiCl 10M": 2}


class FakeStore:
    """The four read-only surfaces the digest touches on a DataStore."""

    def __init__(self, *, loadout=None, board_id=0, occupied=(), levels=None):
        self._loadout = dict(loadout or {})
        self._board_id = int(board_id)
        self._occupied = set(int(e) for e in occupied)
        self._levels = dict(levels or {})

    def _kv_get_text(self, key):
        if key != "pump_loadout" or not self._loadout:
            return None
        return json.dumps({str(k): v for k, v in self._loadout.items()})

    def _kv_get(self, key):
        return None

    def current_board_id(self):
        return self._board_id

    def occupied_electrodes(self, board_id):
        return set(self._occupied) if int(board_id) == self._board_id else set()

    def reservoir_level_uL(self, pump_id):
        return self._levels.get(int(pump_id))


def make_spec(monkeypatch, **overrides):
    """A minimal three-stock campaign spec, decoded against the shipped catalogs."""
    from softae.core.campaign_spec_io import spec_from_dict

    use_default_chemistry(monkeypatch)
    data = {
        "name": "digest_probe",
        "channels": [1, 2],
        "budget": 2,
        "optimizer": "grid",
        "pump_ids": [0, 1, 2],
        "parameter_space": {"replicate": {"type": "int", "low": 1, "high": 2}},
        "general_formulation": {
            "stocks": STOCKS,
            "pump_assignment": dict(PUMPS),
            "target_deposition_uL": 4.5,
            "budget_uL": 118.0,
            "axes": [
                {"kind": "molar_ratio", "a": "Ethylene oxide",
                 "b": "Lithium chloride", "low": 20.0, "high": 20.0,
                 "basis": "volume"},
            ],
        },
        "run_plan": {"phases": [
            {"kind": "formulate", "scope": "per_sample",
             "conditions": {"name": "casting", "temp_setpoint_C": 25.0,
                            "rh_setpoint_pct": 22.0}},
            {"kind": "measure", "scope": "per_batch"},
        ]},
    }
    data.update(overrides)
    # A `None` override means "omit this block", not "carry a null": the object
    # decoders refuse a null rather than reading it as an absence.
    return spec_from_dict({k: v for k, v in data.items() if v is not None},
                          source="<test>")


def severities(digest, *, containing):
    """Severities of the findings whose text mentions *containing*."""
    return [f.severity for f in digest.findings if containing in f.text]


# ── Section 2: the spec's pump assignment against the bench's loadout ────────

def test_stock_spec_assigns_undeclared_pump_warns(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout={0: "5wt% 20k PEO stock",
                               1: "10 wt% silica solution"})
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="Pump 2") == [WARN]


def test_stock_declared_differs_from_spec_blocks(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout={0: "5wt% 20k PEO stock",
                               1: "10 wt% silica solution",
                               2: "LiCl 1M"})
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="Pump 2") == [BLOCK]
    assert digest.has_block


def test_stock_declared_matches_spec_is_ok(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout=dict(zip((0, 1, 2), STOCKS)))
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="Pump 2") == [OK]
    assert not digest.has_block


def test_stock_declared_but_unused_by_spec_is_ok(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout={**dict(zip((0, 1, 2), STOCKS)), 5: "LiCl 1M"})
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="Pump 5 carries") == [OK]


def test_stock_no_formulation_says_so_and_warns(monkeypatch):
    spec = make_spec(monkeypatch, general_formulation=None,
                     parameter_space={"vol": {"type": "float",
                                              "low": 1.0, "high": 2.0}})
    digest = build_final_check(spec, data_store=FakeStore())
    stock = next(s for s in digest.sections if s.title == "Stock on the pumps")
    assert len(stock.rows) == 1
    assert [f.severity for f in stock.findings] == [WARN]


def test_stock_reservoir_level_is_reported_per_pump(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout=dict(zip((0, 1, 2), STOCKS)),
                      levels={0: 9000.0, 1: 8000.0})
    digest = build_final_check(spec, data_store=store)
    rows = dict(next(s for s in digest.sections
                     if s.title == "Stock on the pumps").rows)
    assert "9,000 µL" in rows["pump 0"]
    assert "unknown" in rows["pump 2"]


# ── Section 3: the phase sequence and its conditions ────────────────────────

def test_sequence_no_run_plan_warns_about_the_legacy_layout(monkeypatch):
    spec = make_spec(monkeypatch, run_plan=None)
    digest = build_final_check(spec, data_store=FakeStore())
    sequence = next(s for s in digest.sections if s.title == "Sequence")
    assert [f.severity for f in sequence.findings] == [WARN]
    assert "legacy pointwise" in sequence.findings[0].text


def test_sequence_phase_conditions_name_both_axes(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore())
    body = "\n".join(v for _, v in
                     next(s for s in digest.sections
                          if s.title == "Sequence").rows)
    assert "T 25 °C" in body and "RH 22 %" in body


def test_sequence_undriven_axis_is_named_not_omitted(monkeypatch):
    spec = make_spec(monkeypatch, run_plan={"phases": [
        {"kind": "formulate", "scope": "per_sample",
         "conditions": {"name": "casting", "temp_setpoint_C": 25.0}},
        {"kind": "measure", "scope": "per_batch"},
    ]})
    digest = build_final_check(spec, data_store=FakeStore())
    body = "\n".join(v for _, v in
                     next(s for s in digest.sections
                          if s.title == "Sequence").rows)
    assert "RH not driven" in body


# ── Section 4: the board ────────────────────────────────────────────────────

def test_board_overlap_without_allocator_blocks(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(board_id=3, occupied=(2, 7))
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="already cast") == [BLOCK]


def test_board_overlap_with_allocator_warns(monkeypatch):
    spec = make_spec(monkeypatch, electrode_capacity=32)
    store = FakeStore(board_id=3, occupied=(2, 7))
    digest = build_final_check(spec, data_store=store)
    assert severities(digest, containing="already cast") == [WARN]


def test_board_no_overlap_is_ok(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore(board_id=3,
                                                          occupied=(7, 8)))
    assert severities(digest, containing="none of them") == [OK]


class PartialStore:
    """A store-like object missing ``current_board_id``.

    The digest is handed whatever object its caller holds — a GUI stand-in, a
    store from an older schema — and a missing read must degrade the section it
    feeds, never take the whole page down with it.
    """

    def _kv_get_text(self, key):
        return None

    def occupied_electrodes(self, board_id):
        return set()


def test_board_store_missing_a_read_degrades_instead_of_raising(monkeypatch):
    """One absent store method costs its own row, never the whole digest."""
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=PartialStore())
    board = next(s for s in digest.sections if s.title == "Board")
    assert dict(board.rows)["board id"] == "None"


def test_board_without_a_store_reports_unchecked_not_clean(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec)
    board = next(s for s in digest.sections if s.title == "Board")
    assert "NOT CHECKED" in dict(board.rows)["occupancy"]
    assert [f.severity for f in board.findings] == [WARN]


# ── Section 6: the projection ───────────────────────────────────────────────

def test_projection_without_a_catalog_says_it_did_not_run(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore())
    section = next(s for s in digest.sections if s.title == "Projection")
    assert "not projected" in dict(section.rows)["duration"]
    assert [f.severity for f in section.findings] == [WARN]


def test_projection_with_a_catalog_reports_duration_and_draw(
        monkeypatch, task_catalog):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore(),
                               task_catalog=task_catalog)
    rows = dict(next(s for s in digest.sections
                     if s.title == "Projection").rows)
    assert "per iteration" in rows and rows["per iteration"]
    assert "µL per iteration" in rows["stock draw"]


# ── Rendering ───────────────────────────────────────────────────────────────

def test_render_carries_every_section_title(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore())
    text = render_final_check(digest)
    for section in digest.sections:
        assert section.title in text
    assert "Verify before proceeding" in text


def test_render_stays_inside_the_line_width(monkeypatch):
    spec = make_spec(monkeypatch)
    store = FakeStore(loadout={0: "LiCl 1M"}, board_id=3, occupied=(1,))
    digest = build_final_check(spec, data_store=store)
    lines = render_final_check(digest, width=80).splitlines()
    # An empty page would satisfy the width check on its own, so the page has to
    # be shown to have content before its width means anything.
    assert len(lines) > 20
    assert [line for line in lines if len(line) > 80] == []


def test_render_counts_blocks_and_warnings_on_the_last_line(monkeypatch):
    spec = make_spec(monkeypatch)
    digest = build_final_check(spec, data_store=FakeStore(
        loadout={2: "LiCl 1M"}))
    last = render_final_check(digest).splitlines()[-1]
    assert last.startswith(f"{digest.n_block} blocking, {digest.n_warn} warning")


def test_render_empty_digest_says_nothing_to_verify():
    digest = FinalCheck("nothing", (Section("Campaign", (("a", "b"),)),))
    assert "nothing to verify" in render_final_check(digest)


def test_findings_collapse_identical_duplicates():
    twice = Finding(WARN, "the same sentence")
    digest = FinalCheck("dupes", (Section("A", (), (twice,)),
                                  Section("B", (), (twice,))))
    assert len(digest.findings) == 1


def test_findings_order_blocks_before_warnings_before_ok():
    digest = FinalCheck("order", (Section("A", (), (
        Finding(OK, "fine"), Finding(WARN, "hmm"), Finding(BLOCK, "no"))),))
    assert [f.severity for f in digest.findings] == [BLOCK, WARN, OK]


# ── The prompt ──────────────────────────────────────────────────────────────

CLEAN = FinalCheck("clean", (Section("Campaign", (("a", "b"),)),))
BLOCKED = FinalCheck("blocked", (Section("Campaign", (), (
    Finding(BLOCK, "intent and bench disagree"),)),))


@pytest.mark.parametrize("digest, assume_yes, answer, expected", [
    (CLEAN, True, None, True),
    (CLEAN, False, "y", True),
    (CLEAN, False, "yes", True),
    (CLEAN, False, "n", False),
    (CLEAN, False, "", False),
    (BLOCKED, False, "y", False),
    (BLOCKED, True, None, False),
])
def test_confirm_answer_and_blocks_decide_the_launch(
        digest, assume_yes, answer, expected):
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return answer

    assert confirm_final_check(
        digest, assume_yes=assume_yes, ask=ask) is expected
    assert bool(asked) is not (assume_yes or digest.has_block)


def test_confirm_block_with_assume_yes_is_refused_without_asking():
    """A block is never overridable: --yes answers warnings, not blocks."""
    def ask(prompt):  # pragma: no cover - must never be reached
        raise AssertionError("a block must not be put to the operator")

    assert confirm_final_check(BLOCKED, assume_yes=True, ask=ask) is False


def test_confirm_unanswerable_prompt_is_a_no():
    def ask(prompt):
        raise EOFError

    assert confirm_final_check(CLEAN, assume_yes=False, ask=ask) is False


def test_confirm_prints_nothing(capsys):
    confirm_final_check(CLEAN, assume_yes=False, ask=lambda _p: "y")
    assert capsys.readouterr().out == ""


# ── The module entry point ──────────────────────────────────────────────────

def write_spec(tmp_path, spec_dict):
    """Serialize a spec dict to TOML beside the test, via the spec's own codec."""
    import tomli_w

    path = tmp_path / "probe.toml"
    path.write_bytes(tomli_w.dumps(spec_dict).encode("utf-8"))
    return path


def test_main_exits_nonzero_on_a_block_without_yes(monkeypatch, tmp_path, capsys):
    spec = make_spec(monkeypatch)
    from softae.core.campaign_spec_io import spec_to_dict

    path = write_spec(tmp_path, spec_to_dict(spec))
    monkeypatch.setattr("sys.stdin", None)
    monkeypatch.setattr(
        "softae.core.final_check.build_final_check",
        lambda *a, **k: BLOCKED)

    assert main([str(path)]) == 3
    assert "intent and bench disagree" in capsys.readouterr().out


def test_main_with_yes_exits_zero(monkeypatch, tmp_path, capsys):
    spec = make_spec(monkeypatch)
    from softae.core.campaign_spec_io import spec_to_dict

    path = write_spec(tmp_path, spec_to_dict(spec))
    assert main([str(path), "--yes"]) == 0
    assert "FINAL CHECK" in capsys.readouterr().out


def test_main_without_a_terminal_says_to_pass_yes(monkeypatch, tmp_path, capsys):
    spec = make_spec(monkeypatch)
    from softae.core.campaign_spec_io import spec_to_dict

    path = write_spec(tmp_path, spec_to_dict(spec))
    monkeypatch.setattr("sys.stdin", None)
    assert main([str(path)]) == 3
    assert "no terminal" in capsys.readouterr().out
