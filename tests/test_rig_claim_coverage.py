"""Static-scan coverage check: every ``create_manager(`` call site in ``src/``
is accounted for, by name, as one of three things.

Written for TASKS.md item E.4 (`docs/SubAgent docs/rig_claim_enforcement_scan.md`,
"the enforcement scan"). That spec's dual read (orchestrator + an independent
research subagent, no divergence) found 11 real ``create_manager(`` call sites
under ``src/`` and classified each one. This module re-derives the call-site
list mechanically (so a 12th site added later is not silently missed) and
checks each one against that classification:

* **mock-only** — the call is ``create_manager(mock=True)`` with a literal
  ``True``, so it can never reach real hardware and needs no claim.
* **claimed** — a claim mechanism (``core.rig_session.held_rig_session`` /
  ``claim_rig_session``, or the inline ``run_lock`` acquisition inside
  ``WorkflowExecutor.run()``) appears, by a textual proxy described below, in
  a position that precedes the real port-open in the same code path.
* **exception** — a named, cited, two-entry list of real gaps this session
  does not own and is not fixing here (see ``KNOWN_GAPS`` below).

Anything that is none of the three is **unclassified**, and the test fails
loudly on it. That is the one property this file exists to guarantee: a new
``create_manager(`` call added anywhere under ``src/`` that is not mock-only,
not claimed, and not on the (append-only, no-expiry) exception list breaks
this test on the next run.

What this test does **not** prove (mirrors spec §4, "what this is not"):

* It is not a data-flow prover. "Claimed" is decided by a **textual proxy** —
  does a claim-mechanism marker string appear in the enclosing function's
  source text (and, for two sites, in one named callee's source text) — not
  by tracing execution order. A function that imports ``held_rig_session``
  in a docstring example and never calls it would false-positive as claimed;
  none of the 11 real sites do this today (checked by hand against the spec's
  §2 table when this file was written), but the proxy is a heuristic, not a
  proof, and is documented as one everywhere it is used below.
* It does not fix, or attempt to fix, either of the two named gaps. Both are
  owned by parallel-session and handed over, not patched, per the spec.
* It does not unify the three claim mechanisms (``rig_session``'s
  session-scoped lock, the inline ``run_lock`` in ``WorkflowExecutor.run()``,
  and the campaign-level lock in ``autonomous_wiring.py``) into one. All
  three are legitimate today; see spec §1.
* It never imports or instantiates a real driver. This is source inspection
  only — ``ast.parse`` over files on disk — and touches no hardware.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "softae"

# The exact call-site count the spec's dual read found under src/ (docs/SubAgent
# docs/rig_claim_enforcement_scan.md §2). If the AST scan below finds a different
# number, something about the codebase or the scan itself has changed enough that
# blind trust in either number is wrong -- investigate before trusting this test.
EXPECTED_SITE_COUNT = 11


# ---------------------------------------------------------------------------
# Known-gap exception list (spec §2 "Two genuine gaps"; this session's mail
# post [e82] hands both to parallel-session rather than patching them here).
# No expiry logic by design -- CLAUDE.md's shared-tree ownership rule treats a
# quietly-fixed note as unauditable; this list is removed from only by someone
# deleting the line once the gap is actually closed.
#
# Keyed by (path relative to src/softae, POSIX separators, enclosing function
# name) rather than line number -- SUBAGENT_RULES.md §2: "locate by symbol,
# never by line number", because anchors drift.
# ---------------------------------------------------------------------------
KNOWN_GAPS: dict[tuple[str, str], str] = {
    ("tools/equilibration.py", "_cmd_run"): (
        "No claim of any kind. EquilibrationRun.run() drives the real "
        "heater/RH controller for potentially hours with zero multi-process "
        "protection. Owned by parallel-session (p-tool-equilibration)."
    ),
    ("tools/campaign.py", "_cmd_run"): (
        "foreign_run_lock() at the site is a read-only peek, not a claim; "
        "connect_all() inside _go() runs before the eventual campaign-level "
        "claim in autonomous_wiring.py. Owned by parallel-session "
        "(p-attach-tranche)."
    ),
}


# ---------------------------------------------------------------------------
# Mock-only sites: the call is literally create_manager(mock=True), so no
# real port is ever opened on this path and no claim is required.
# ---------------------------------------------------------------------------
MOCK_ONLY_SITES: frozenset[tuple[str, str]] = frozenset(
    {
        ("workflows/__main__.py", "_validate_steps"),
        ("core/dropcast.py", "run_dropcast_sweep"),
        ("core/autonomous_wiring.py", "run_autonomous_campaign"),
    }
)


@dataclasses.dataclass(frozen=True)
class ClaimCheck:
    """A textual proxy for "a claim mechanism precedes the port-open".

    ``same_function_markers``: substrings that must all appear in the source
    text of the function directly enclosing the ``create_manager(`` call.

    ``callee``: optionally, the name of one other function in the same module
    that the enclosing function calls, whose own source text must contain
    ``callee_markers``. Used for the two sites where the claim is one hop
    away (a named helper), not inline.
    """

    same_function_markers: tuple[str, ...]
    callee: str | None = None
    callee_markers: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Claimed sites: one entry per non-exception, non-mock-only site (6 of the
# 11). Each check is the mechanical proxy used to confirm the spec's §2
# classification still holds -- see the module docstring for what "textual
# proxy" does and does not prove.
# ---------------------------------------------------------------------------
CLAIM_CHECKS: dict[tuple[str, str], ClaimCheck] = {
    # Mechanism #2 (inline run_lock in WorkflowExecutor.run()): main()
    # constructs a WorkflowExecutor and calls .run() on it in the same
    # function that opened the manager.
    ("workflows/__main__.py", "main"): ClaimCheck(
        same_function_markers=("WorkflowExecutor(", ".run(")
    ),
    # Mechanism #1, one hop: run_app() calls _begin_owner_session(), which
    # (in the same module) calls claim_rig_session() before scheduling
    # connect_all via _connect_and_refresh.
    ("gui/app.py", "run_app"): ClaimCheck(
        same_function_markers=("_begin_owner_session(",),
        callee="_begin_owner_session",
        callee_markers=("claim_rig_session(",),
    ),
    # Mechanism #2: _cmd_run builds a WorkflowExecutor and awaits .run() on
    # it inside the same function's nested _go() coroutine.
    ("tools/commission.py", "_cmd_run"): ClaimCheck(
        same_function_markers=("WorkflowExecutor(", ".run(")
    ),
    # Mechanism #1, one hop: cmd_run() calls the module's _rig_claim() helper,
    # which itself calls held_rig_session().
    ("tools/eis_validate.py", "cmd_run"): ClaimCheck(
        same_function_markers=("_rig_claim(",),
        callee="_rig_claim",
        callee_markers=("held_rig_session(",),
    ),
    # Mechanism #1, inline: run_timing() builds `claim` from held_rig_session()
    # directly and wraps connect_all() in `with claim:`.
    ("tools/eis_timing.py", "run_timing"): ClaimCheck(
        same_function_markers=("held_rig_session(",)
    ),
    # Mechanism #1, inline: _cmd_hold() builds `claim` from held_rig_session()
    # directly before handing off to _run_hold(), which opens the ports.
    ("tools/env_hold.py", "_cmd_hold"): ClaimCheck(
        same_function_markers=("held_rig_session(",)
    ),
}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CallSite:
    rel_path: str  # POSIX path relative to src/softae
    line: int
    func_name: str | None
    call_node: ast.Call
    func_source: str  # source text of the enclosing function (or "")
    module_source: str
    tree: ast.Module


def _callee_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _enclosing_function(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = getattr(current, "parent", None)
    return None


def _find_function_source(tree: ast.Module, source: str, name: str) -> str | None:
    """First function (any nesting depth) named ``name`` in ``tree``, as source text."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    return None


def discover_call_sites(src_root: Path) -> list[CallSite]:
    """AST-scan every ``.py`` file under ``src_root`` for ``create_manager(`` calls.

    AST (not regex/grep) so that a docstring or comment merely *mentioning*
    ``create_manager(`` -- as ``drivers/factory.py``'s module docstring does,
    three times, and its ``def create_manager(`` definition itself does once
    more -- is never mistaken for an invocation: a string literal's contents
    are not re-parsed as code, and a ``FunctionDef`` is not an ``ast.Call``.
    """
    sites: list[CallSite] = []
    for pyfile in sorted(src_root.rglob("*.py")):
        text = pyfile.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(pyfile))
        except SyntaxError:
            continue
        _attach_parents(tree)
        rel = pyfile.relative_to(src_root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee_name(node) == "create_manager":
                func_node = _enclosing_function(node)
                func_source = (
                    ast.get_source_segment(text, func_node) if func_node is not None else ""
                ) or ""
                sites.append(
                    CallSite(
                        rel_path=rel,
                        line=node.lineno,
                        func_name=func_node.name if func_node is not None else None,
                        call_node=node,
                        func_source=func_source,
                        module_source=text,
                        tree=tree,
                    )
                )
    return sites


def _is_literal_mock_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "mock":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def _claim_check_passes(site: CallSite, check: ClaimCheck) -> bool:
    if not all(marker in site.func_source for marker in check.same_function_markers):
        return False
    if check.callee is not None:
        callee_source = _find_function_source(site.tree, site.module_source, check.callee)
        if callee_source is None:
            return False
        if not all(marker in callee_source for marker in check.callee_markers):
            return False
    return True


def classify_site(
    site: CallSite, gaps: dict[tuple[str, str], str]
) -> tuple[str, str]:
    """Classify one call site. Returns ``(kind, detail)``.

    ``kind`` is one of ``"mock_only"``, ``"exception"``, ``"claimed"``, or
    ``"unclassified"``. ``gaps`` is passed in (rather than read as a module
    global) so the positive control (below) can call this with an empty dict
    and observe the two known gaps fall through to "unclassified".
    """
    key = (site.rel_path, site.func_name or "")

    if key in gaps:
        return "exception", gaps[key]

    if key in MOCK_ONLY_SITES:
        if _is_literal_mock_true(site.call_node):
            return "mock_only", "literal mock=True"
        return (
            "unclassified",
            f"{key} listed as mock-only but call is not literal mock=True "
            f"(line {site.line}) -- baseline assumption broke",
        )

    if key in CLAIM_CHECKS:
        check = CLAIM_CHECKS[key]
        if _claim_check_passes(site, check):
            return "claimed", "claim-mechanism marker(s) found"
        return (
            "unclassified",
            f"{key} listed as claimed but its claim-marker proxy did not match "
            f"(line {site.line}) -- either the code moved or the claim was lost",
        )

    return (
        "unclassified",
        f"{key} (line {site.line}) is not mock-only, not claimed, and not on "
        "the exception list -- a new create_manager( site with no accounted "
        "claim path. See docs/SubAgent docs/rig_claim_enforcement_scan.md.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discovers_exactly_the_known_call_site_count():
    """Guards the scanner itself: if this drifts, investigate before trusting
    either number (task instructions, and CLAUDE.md's "the check that could
    not check" caution) -- a scanner that silently finds fewer sites than
    exist would make every other assertion in this file vacuous.
    """
    sites = discover_call_sites(SRC_ROOT)
    found = sorted((s.rel_path, s.func_name, s.line) for s in sites)
    assert len(sites) == EXPECTED_SITE_COUNT, (
        f"Expected {EXPECTED_SITE_COUNT} create_manager( call sites under "
        f"src/softae (per rig_claim_enforcement_scan.md §2), found "
        f"{len(sites)}: {found}"
    )


def test_every_call_site_is_mock_only_claimed_or_a_named_exception():
    """The check this module exists to provide: no unclassified site.

    A 12th ``create_manager(`` call added anywhere under ``src/`` that is
    neither ``mock=True`` literal, nor behind one of the two proxied claim
    mechanisms, nor added by name to ``KNOWN_GAPS`` fails this test.
    """
    sites = discover_call_sites(SRC_ROOT)
    unclassified = []
    kinds: dict[str, int] = {}
    for site in sites:
        kind, detail = classify_site(site, KNOWN_GAPS)
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "unclassified":
            unclassified.append(f"{site.rel_path}:{site.func_name} (line {site.line}) -- {detail}")

    assert not unclassified, "Unclassified create_manager( site(s):\n" + "\n".join(unclassified)

    # Pin the known-good baseline shape too (3 mock-only + 6 claimed + 2
    # exception == 11), so a site silently sliding from "claimed" to
    # "mock-only" (or vice versa) -- which would not trip the assertion
    # above -- is still visible as a bucket-count change.
    assert kinds == {"mock_only": 3, "claimed": 6, "exception": 2}, kinds


@pytest.mark.parametrize("gap_key", sorted(KNOWN_GAPS))
def test_each_named_gap_actually_matches_its_site(gap_key: tuple[str, str]):
    """Every KNOWN_GAPS entry corresponds to a real, currently-existing site.

    Catches a stale exception entry (the gap was fixed, or the function was
    renamed) sitting in the list doing nothing -- silent staleness in the
    other direction from the one this file mainly guards against.
    """
    sites = discover_call_sites(SRC_ROOT)
    site_keys = {(s.rel_path, s.func_name or "") for s in sites}
    assert gap_key in site_keys, (
        f"{gap_key} is in KNOWN_GAPS but no longer matches any discovered "
        f"create_manager( call site -- update or remove the exception entry"
    )


def test_positive_control_empty_exception_list_fails_on_exactly_the_two_known_gaps():
    """Proves the check is not vacuously green (spec §5).

    Calls ``classify_site`` directly with an empty exception dict -- not a
    monkeypatch of module state, so this is a clean unit-level exercise of
    the classification function -- and asserts the *only* sites that fall to
    "unclassified" are exactly the two named gaps. Every other site must
    still resolve via mock-only or claimed, proving those two buckets do not
    depend on the exception list at all.
    """
    sites = discover_call_sites(SRC_ROOT)
    unclassified_keys = set()
    other_kinds = set()
    for site in sites:
        kind, _detail = classify_site(site, gaps={})
        key = (site.rel_path, site.func_name or "")
        if kind == "unclassified":
            unclassified_keys.add(key)
        else:
            other_kinds.add(kind)

    assert unclassified_keys == set(KNOWN_GAPS), (
        f"With an empty exception list, expected exactly {set(KNOWN_GAPS)} to "
        f"fall through as unclassified; got {unclassified_keys}"
    )
    # And nothing else was pulled down with them: mock-only and claimed
    # sites classify identically regardless of the exception list.
    assert other_kinds == {"mock_only", "claimed"}
