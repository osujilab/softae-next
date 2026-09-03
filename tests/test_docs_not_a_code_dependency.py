"""Working specs are untracked, so no shipped code may *read* one.

`docs/` is private by default: `.gitignore:56` ignores `docs/*` and then re-includes
a short public list (`api/`, `USER_GUIDE.md`, `architecture.md`, `index.md`). Everything
else under `docs/` — every working spec, every roadmap, `SESSION_MAIL.md`, `TASKS.md`,
`OWNERSHIP.toml` — exists on this machine and **does not exist in a fresh clone**.

That asymmetry is the whole reason this file exists. A test or tool that reads a spec
file passes here, where the file is on disk, and fails in CI or on a colleague's first
checkout, where it is not. Green here, red there — and the failure surfaces far from the
edit that caused it. Nobody has done this yet; the point of the guard is to move that
from "nobody has" to "nobody can".

**Citations are not reads.** Seven modules and a dozen test files name a spec path in a
prose docstring, which is good practice and must keep working. The guard is therefore
AST-based rather than textual: it looks for a filesystem read whose *path argument*
resolves to the untracked area, not for the string appearing somewhere in the file.

What the detector sees, and what it does not, is spelled out on `_path_segments` and
`_READ_SINK_METHODS` below. The honest summary: it follows literals, `Path(...)`,
`/` joining, `os.path.join`, f-strings and module-level constants into a read call. It
does **not** follow a path across a function boundary, out of a container, or in from a
config file or CLI argument.

`tools/whose.py` is exactly that blind spot, and is recorded in `ACCEPTED_READS` — see
the note there.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Trees whose Python is subject to the guard.
SCAN_ROOTS = ("src", "tests", "tools")

#: Directory names never descended into: virtualenvs, caches, build output. The
#: scratchpad lives outside the repo, so it is excluded by construction.
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site",
        "venv",
    }
)

# --------------------------------------------------------------------------- #
# what counts as "tracked"
# --------------------------------------------------------------------------- #

#: Files directly under `docs/` that `.gitignore` re-includes. Reading these is
#: legitimate: they are committed and present in every clone.
TRACKED_DOCS_FILES = frozenset({"USER_GUIDE.md", "architecture.md", "index.md"})

#: Directories under `docs/` that `.gitignore` re-includes, recursively.
TRACKED_DOCS_DIRS = frozenset({"api"})

#: Reads of the untracked area that are known, deliberate and accepted, mapped to why.
#:
#: This is not an amnesty list for convenience, and the admitting criterion is
#: **category, not behaviour**. "It fails loudly when the file is absent" is necessary
#: but nowhere near sufficient: a genuine violation can be made to fail loudly too, and
#: would then buy its way onto this list while still breaking a fresh clone of the thing
#: the guard protects.
#:
#: What actually admits an entry is what the file is *for*. Coordination tooling whose
#: **input is session state** — who is editing what, this hour, in this working tree —
#: has nothing to read on a fresh checkout because a fresh checkout has no sessions.
#: Absence of the map is not a degraded environment for such a tool; it is the correct
#: description of an empty one. Product code that drives the rig is the opposite
#: category: it must work on a fresh checkout, and that is the category this guard exists
#: to protect.
#:
#: An entry is not merely tolerated, either. Admission carries an obligation — the
#: refusal the category argument leans on is *asserted*, against the real file, further
#: down this module. See `test_whose_refuses_a_missing_map_rather_than_answering` and
#: its neighbours.
ACCEPTED_READS: dict[str, str] = {
    "tools/whose.py": (
        "Coordination tooling, not product code. Reading `docs/SubAgent docs/"
        "OWNERSHIP.toml` IS its purpose: the map is per-working-tree session state — who "
        "holds which paths right now — so it is correctly absent from a fresh clone, "
        "which has no sessions to describe. Nothing shipped imports it; it is run by hand "
        "and by the pre-commit gate, both of which only exist where sessions do. The "
        "category argument is backed by a tested contract below: with no map it refuses "
        "and names the cause rather than answering permissively. Note the entry is "
        "documentation rather than suppression today — whose.py routes DEFAULT_MAP "
        "through a function parameter, which the detector below cannot follow, so it "
        "would not be flagged in any case."
    ),
}


# --------------------------------------------------------------------------- #
# path-expression reconstruction
# --------------------------------------------------------------------------- #

#: Callables that build a path out of their arguments.
_PATH_CTORS = frozenset({"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath"})

#: Method calls that read the file their receiver names.
#:
#: These fire only when the *receiver* reconstructs to an untracked docs path, so
#: including a name as common as `open` costs nothing: `serial.open()` has no path
#: expression to taint.
_READ_SINK_METHODS = frozenset(
    {
        "open",
        "read",
        "read_bytes",
        "read_text",
    }
)

#: Functions taking a path as their first positional argument and reading it. Matched
#: on the trailing attribute name, so `json.load`, `pd.read_csv` and a bare imported
#: `read_csv` all land the same way. `json.load`/`tomllib.load` take a file object
#: rather than a path; they are listed because the wrapping `open()` may itself be
#: matched, and because a mistaken direct call should still trip the guard.
_READ_SINK_FUNCS = frozenset(
    {
        "load",
        "loadtxt",
        "genfromtxt",
        "read_csv",
        "read_excel",
        "read_json",
        "read_parquet",
        "read_table",
        "safe_load",
    }
)


def _split(text: str) -> list[str]:
    """Split a literal into path segments on either separator."""
    return [seg for seg in text.replace("\\", "/").split("/") if seg not in ("", ".")]


def _path_segments(
    node: ast.AST, bindings: dict[str, ast.AST], depth: int = 0
) -> list[str | None]:
    """Reconstruct a path expression as segments; `None` marks an unknown segment.

    Understood: string literals, `Path(...)`/`PurePath(...)`, `/` joining, `.joinpath()`,
    `os.path.join()`, f-strings, and plain `NAME` lookups resolved against module-level
    assignments in the same file.

    **Not** understood, and deliberately stated rather than glossed: a path arriving as a
    function parameter, read out of a list/dict/dataclass, returned by a helper, or
    supplied at runtime by a config file, environment variable or CLI argument. Those
    resolve to `None` and the expression is not flagged. This guard raises the cost of
    reaching an untracked spec by accident; it is not a proof that none can be reached.
    """
    if depth > 8:  # cheap cycle/recursion bound; `A = A / "x"` must not hang collection
        return [None]

    if isinstance(node, ast.Constant):
        return _split(node.value) if isinstance(node.value, str) else [None]

    if isinstance(node, ast.JoinedStr):  # f-string
        out: list[str | None] = []
        for part in node.values:
            out.extend(_path_segments(part, bindings, depth + 1))
        return out

    if isinstance(node, ast.FormattedValue):
        return [None]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [
            *_path_segments(node.left, bindings, depth + 1),
            *_path_segments(node.right, bindings, depth + 1),
        ]

    if isinstance(node, ast.Call):
        name = _callee_name(node.func)
        if name in _PATH_CTORS or name == "join" or name == "joinpath":
            out = []
            if name == "joinpath" and isinstance(node.func, ast.Attribute):
                out.extend(_path_segments(node.func.value, bindings, depth + 1))
            for arg in node.args:
                out.extend(_path_segments(arg, bindings, depth + 1))
            return out
        return [None]

    if isinstance(node, ast.Name):
        bound = bindings.get(node.id)
        if bound is not None:
            return _path_segments(bound, bindings, depth + 1)
        return [None]

    if isinstance(node, ast.Attribute):
        # `SOME.CONST` or `self.path` — opaque, but `.parent`/`.resolve()` style
        # navigation off a known root is common enough to follow through.
        if node.attr in ("parent", "resolve", "absolute"):
            return [*_path_segments(node.value, bindings, depth + 1), None]
        return [None]

    return [None]


def _callee_name(func: ast.AST) -> str | None:
    """Trailing name of a callee: `Path` -> 'Path', `os.path.join` -> 'join'."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _names_untracked_docs(segments: list[str | None]) -> bool:
    """True when the segments name something under `docs/` that is not re-included.

    An unknown segment immediately after `docs` (`docs/{name}/spec.md`) is treated as
    *not* a violation: it could be either, and a guard that fires on ambiguity gets
    disabled. The precision is spent on the reads that are unambiguous.
    """
    for index, segment in enumerate(segments):
        if segment is None or segment.lower() != "docs":
            continue
        rest = segments[index + 1 :]
        if not rest:
            return False  # the directory itself, not a file read
        head = rest[0]
        if head is None:
            return False
        if head in TRACKED_DOCS_DIRS or head in TRACKED_DOCS_FILES:
            return False
        return True
    return False


# --------------------------------------------------------------------------- #
# the scan
# --------------------------------------------------------------------------- #


class _Finding:
    __slots__ = ("path", "lineno", "expression")

    def __init__(self, path: str, lineno: int, expression: str) -> None:
        self.path = path
        self.lineno = lineno
        self.expression = expression

    def __repr__(self) -> str:  # pragma: no cover - only reached in failure text
        return f"{self.path}:{self.lineno}: {self.expression}"


class _DocsReadVisitor(ast.NodeVisitor):
    """Flag read calls whose path argument resolves into the untracked docs area."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.findings: list[_Finding] = []
        self.bindings: dict[str, ast.AST] = {}

    def collect_bindings(self, tree: ast.Module) -> None:
        """Record `NAME = <expr>` bindings so module constants resolve.

        First binding wins: a name reassigned in a branch is ambiguous, and the first
        is the one a reader takes as the definition.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bindings.setdefault(target.id, node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    self.bindings.setdefault(node.target.id, node.value)

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node.func)

        if isinstance(node.func, ast.Attribute) and name in _READ_SINK_METHODS:
            self._check(node.func.value, node)
        elif name is not None and (name == "open" or name in _READ_SINK_FUNCS):
            if node.args:
                self._check(node.args[0], node)

        self.generic_visit(node)

    def _check(self, path_expr: ast.AST, call: ast.Call) -> None:
        segments = _path_segments(path_expr, self.bindings)
        if _names_untracked_docs(segments):
            rendered = "/".join("<runtime>" if s is None else s for s in segments)
            self.findings.append(_Finding(self.relpath, call.lineno, rendered))


def scan_source(source: str, relpath: str) -> list[_Finding]:
    """Run the detector over one module's text. The single entry point for the scan.

    Both the repository sweep and the positive control call this, so the control
    exercises the production detector rather than a parallel reimplementation.
    """
    tree = ast.parse(source, filename=relpath)
    visitor = _DocsReadVisitor(relpath)
    visitor.collect_bindings(tree)
    visitor.visit(tree)
    return visitor.findings


def iter_python_files() -> list[Path]:
    """Every Python file under the scanned roots, caches and virtualenvs excluded."""
    found: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if SKIP_DIR_NAMES.intersection(path.parts):
                continue
            found.append(path)
    return sorted(found)


def _relpath(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _read_source(path: Path) -> str:
    """Read a module's text for parsing.

    `utf-8-sig`, not `utf-8`: one file in the tree (`gui/tabs/tab_monitor.py`) carries a
    UTF-8 BOM, and plain `utf-8` leaves the U+FEFF in the string, where `ast.parse`
    rejects it as a non-printable character. CPython strips the BOM itself when importing,
    so the module is perfectly valid — only a hand-rolled parse trips on it.
    """
    return path.read_text(encoding="utf-8-sig")


@pytest.fixture(scope="module")
def scan_results() -> tuple[list[Path], list[_Finding]]:
    """(files scanned, findings) — computed once; every assertion reads this one sweep."""
    files = iter_python_files()
    findings: list[_Finding] = []
    for path in files:
        relative = _relpath(path)
        if relative in ACCEPTED_READS:
            continue
        findings.extend(scan_source(_read_source(path), relative))
    return files, findings


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #


def test_no_python_source_reads_untracked_docs(scan_results) -> None:
    """No shipped Python may read a file that a fresh clone will not have."""
    _files, findings = scan_results
    if not findings:
        return

    offenders = "\n".join(f"  {f.path}:{f.lineno}  ->  {f.expression}" for f in findings)
    pytest.fail(
        "Python source reads a path under the untracked docs working area:\n\n"
        f"{offenders}\n\n"
        "Everything under `docs/` except `api/`, `USER_GUIDE.md`, `architecture.md` and\n"
        "`index.md` is gitignored (.gitignore:56). Those files exist on your machine and\n"
        "do NOT exist in a fresh clone or in CI, so this code passes here and fails there\n"
        "— and it fails a long way from the edit that caused it.\n\n"
        "Fix it one of these ways:\n"
        "  * Inline the data into the Python module or the test that needs it.\n"
        "  * Move it to a tracked location (`src/softae/**` data file, `tests/data/`) and\n"
        "    read it from there.\n"
        "  * If the file is genuinely committed, add its negation to .gitignore and to\n"
        "    TRACKED_DOCS_FILES / TRACKED_DOCS_DIRS in this module.\n"
        "  * If the file is coordination tooling whose input IS session state — so a\n"
        "    fresh checkout has nothing for it to read and needs none — add it to\n"
        "    ACCEPTED_READS above with that argument, and pin its refusal in a test.\n"
        "    'It fails loudly' on its own does not admit an entry: a real violation can\n"
        "    fail loudly too. The question is whether product code needs it to work.\n\n"
        "Citing a spec path in a docstring is fine and is not what tripped this; the guard\n"
        "flags a filesystem read, not a mention."
    )


def test_scan_reaches_the_whole_corpus(scan_results) -> None:
    """A walk that breaks scans nothing, and a scan of nothing passes silently.

    The repository holds ~472 Python files across the three roots. The floor is set well
    below that so ordinary growth or pruning does not trip it, but far enough above zero
    that a broken `rglob`, a wrong `REPO_ROOT`, or an over-eager skip list fails loudly
    instead of reporting a clean bill of health over an empty set.
    """
    files, _findings = scan_results
    assert len(files) >= 400, f"only {len(files)} Python files scanned — the walk is broken"

    roots_seen = {_relpath(p).split("/", 1)[0] for p in files}
    assert roots_seen == set(SCAN_ROOTS), f"missing roots: {set(SCAN_ROOTS) - roots_seen}"


# --------------------------------------------------------------------------- #
# anti-vacuity: the detector must discriminate
# --------------------------------------------------------------------------- #

def test_detector_flags_a_real_docs_read(tmp_path: Path) -> None:
    """Positive control: a module that reads a spec is flagged, through the real scanner.

    Without this the guard is indistinguishable from one that never fires.
    """
    source = (
        "import os\n"
        "import tomllib\n"
        "from pathlib import Path\n"
        "\n"
        'SPEC = Path("docs") / "SubAgent docs" / "OWNERSHIP.toml"\n'
        "\n"
        "def load_all(name):\n"
        '    a = open("docs/SubAgent docs/spec.md").read()\n'
        '    b = Path("docs/SubAgent docs/other.md").read_text()\n'
        '    c = open(os.path.join("docs", "SubAgent docs", "third.md"))\n'
        "    d = SPEC.read_text()\n"
        '    e = Path(f"docs/SubAgent docs/{name}.md").read_bytes()\n'
        '    f = Path("docs").joinpath("SubAgent docs", "sixth.md").read_text()\n'
        '    g = tomllib.load(open("docs/SubAgent docs/seventh.toml", "rb"))\n'
        "    return a, b, c, d, e, f, g\n"
    )
    module = tmp_path / "offender.py"
    module.write_text(source, encoding="utf-8")

    findings = scan_source(module.read_text(encoding="utf-8"), "offender.py")

    flagged_lines = {f.lineno for f in findings}
    # Seven distinct read shapes, on seven consecutive lines starting at the `open(...)`.
    expected = set(range(8, 15))
    assert expected <= flagged_lines, (
        f"detector missed shapes on lines {sorted(expected - flagged_lines)}; "
        f"flagged {sorted(flagged_lines)}"
    )


def test_detector_ignores_reads_of_tracked_docs(tmp_path: Path) -> None:
    """`docs/architecture.md` and friends are committed; reading them is legitimate."""
    source = (
        "from pathlib import Path\n"
        "\n"
        "def load():\n"
        '    a = Path("docs/architecture.md").read_text()\n'
        '    b = Path("docs/USER_GUIDE.md").read_text()\n'
        '    c = Path("docs/index.md").read_text()\n'
        '    d = Path("docs") / "api" / "errors.md"\n'
        "    return a, b, c, d.read_text()\n"
    )
    module = tmp_path / "tracked.py"
    module.write_text(source, encoding="utf-8")

    findings = scan_source(module.read_text(encoding="utf-8"), "tracked.py")
    assert findings == [], f"tracked docs must not be flagged, got {findings}"


#: Real modules whose docstrings cite a working spec. The discrimination that matters is
#: measured on these rather than on fixtures: prose citation must survive the guard.
CITING_SOURCES = (
    "src/softae/analysis/eis/measurability.py",
    "src/softae/analysis/eis/router.py",
    "src/softae/analysis/measurement_result.py",
    "src/softae/core/campaign_events.py",
    "src/softae/core/data_store.py",
    "src/softae/optimizers/feasibility.py",
    "src/softae/tools/measurability_sweep.py",
)


@pytest.mark.parametrize("relpath", CITING_SOURCES)
def test_docstring_citations_are_not_reads(relpath: str, scan_results) -> None:
    """A module naming its spec in prose is doing the right thing and must stay green.

    Two halves, and both are needed: that the citation is still *there* (otherwise the
    test proves nothing about discrimination — it would pass on any clean file), and that
    the file is not flagged.
    """
    files, findings = scan_results
    path = REPO_ROOT / relpath

    assert path in files, f"{relpath} was not reached by the scan"
    assert "docs/SubAgent docs/" in _read_source(path), (
        f"{relpath} no longer cites a spec path — this test no longer measures "
        "discrimination; repoint it at a module that does."
    )
    assert [f for f in findings if f.path == relpath] == [], (
        f"{relpath} cites a spec in prose and was wrongly flagged as reading it"
    )


def test_accepted_reads_are_not_stale() -> None:
    """An allowlist naming files that no longer exist is folklore, not a record."""
    for relpath in ACCEPTED_READS:
        assert (REPO_ROOT / relpath).is_file(), (
            f"ACCEPTED_READS names {relpath}, which does not exist — remove the entry"
        )


# --------------------------------------------------------------------------- #
# the obligation the ACCEPTED_READS entry carries
# --------------------------------------------------------------------------- #
#
# The entry above is admitted on category — coordination tooling reading session state.
# The category argument leans on one behaviour: on a checkout with no map, whose.py must
# REFUSE, not answer. If it ever answered, the exemption would flip from "correct by
# category" to a violation of exactly the kind this module exists to catch — and a
# quiet one, because "nothing is foreign" is the answer a pre-commit gate is happiest
# to hear. So the behaviour is asserted here rather than asserted in a comment.
#
# What is pinned is the *shape* of the refusal, not its prose: a non-zero exit that is
# not the "checked, found something" code, output naming the map it could not read, and
# the absence of any permissive answer. Message wording and line numbers are left free
# deliberately; a test that pins those tests the message, not the contract.

WHOSE_PY = REPO_ROOT / "tools" / "whose.py"

#: Fragments of whose.py's *successful* answers. None of these may appear when the map
#: is unusable: they are what "checked, and nothing is wrong" looks like on stdout, and
#: an unusable map has checked nothing. Compared case-folded.
_PERMISSIVE_ANSWERS = ("unclaimed", "nothing staged", "staged path(s), all")

#: whose.py's module docstring fixes 1 as "the gate found something" and reserves a
#: separate code for "the map cannot be trusted". The exact error code is not pinned —
#: only that a broken map is not reported as a clean-ish finding.
_EXIT_FINDINGS = 1


def _run_whose(
    args: list[str], *, script: Path = WHOSE_PY
) -> subprocess.CompletedProcess[str]:
    """Run whose.py the way the protocol runs it: a subprocess, not an import.

    Importing and monkeypatching would test a rearrangement of the module rather than
    the tool the pre-commit gate actually invokes — including its `main()` return path
    and its own stdout/stderr handling.

    `script` is a parameter only so an anti-vacuity run can point these same assertions
    at a deliberately broken copy and watch them go red; every test here uses the real
    file.
    """
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _assert_refused_without_answering(
    done: subprocess.CompletedProcess[str], map_path: Path
) -> None:
    """The contract: refused, said which file, and did not answer anyway."""
    combined = f"{done.stdout}\n{done.stderr}"
    context = f"\nexit={done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"

    assert done.returncode != 0, (
        "whose.py exited 0 with an unusable ownership map. A caller cannot distinguish "
        "that from a clean gate, which is precisely the failure the tool documents "
        "itself as avoiding — and the condition that would turn its ACCEPTED_READS "
        f"exemption into a violation.{context}"
    )
    assert done.returncode != _EXIT_FINDINGS, (
        "whose.py reported an unusable map with the exit code that means 'checked, "
        "found something'. Those must stay distinct: one is a coordination finding to "
        f"act on, the other is the tool declining to answer.{context}"
    )

    assert map_path.name in combined, (
        "whose.py refused but did not name the map it could not use, so the operator "
        f"cannot tell an absent map from a malformed one or a wrong --map.{context}"
    )

    lowered = combined.lower()
    for marker in _PERMISSIVE_ANSWERS:
        assert marker not in lowered, (
            f"whose.py printed {marker!r} — an answer — while holding an unusable map. "
            "The wrong answer wearing the safe answer's clothes is the whole hazard "
            f"here; refusing must not look like reporting nothing foreign.{context}"
        )

    assert "Traceback (most recent call last)" not in combined, (
        "whose.py crashed rather than refusing. A traceback is a loud failure, but not "
        "a diagnosed one, and it means the refusal path was not the path taken."
        f"{context}"
    )


def test_whose_refuses_a_missing_map_rather_than_answering(tmp_path: Path) -> None:
    """No map on disk: the fresh-checkout condition, reproduced without touching one.

    `--map` points at a path inside `tmp_path` that was never created. The live map at
    `docs/SubAgent docs/OWNERSHIP.toml` is coordination state for concurrent sessions
    and is never moved, renamed or perturbed to produce this condition.
    """
    absent = tmp_path / "no_such_ownership_map.toml"
    assert not absent.exists(), "fixture precondition: the map must not be there"

    done = _run_whose(["tools/whose.py", "--map", str(absent)])
    _assert_refused_without_answering(done, absent)


def test_whose_gate_refuses_a_missing_map(tmp_path: Path) -> None:
    """The same refusal on the mode that matters: the pre-commit staged gate.

    Lookup mode being safe would not save anything on its own — the gate is what stands
    between a session and staging another session's work.
    """
    absent = tmp_path / "no_such_ownership_map.toml"

    done = _run_whose(
        ["--staged", "--me", "eis-acq-session", "--map", str(absent)]
    )
    _assert_refused_without_answering(done, absent)


def test_whose_refuses_a_claimless_map(tmp_path: Path) -> None:
    """A map present but empty of claims is the permissive answer's best disguise.

    It parses, it has sessions, and every lookup against it comes back unowned — so a
    gate run on it reports nothing foreign about a tree full of foreign work. Absence of
    the file is loud; absence of its *contents* is the quiet version of the same thing,
    and must be refused just as hard.
    """
    claimless = tmp_path / "claimless_ownership_map.toml"
    claimless.write_text('sessions = ["probe-session"]\n', encoding="utf-8")

    done = _run_whose(["tools/whose.py", "--map", str(claimless)])
    _assert_refused_without_answering(done, claimless)


def test_whose_answers_normally_from_a_usable_map(tmp_path: Path) -> None:
    """Positive control for the three refusals above.

    Every one of those asserts a non-zero exit, which a subprocess yields for a wrong
    interpreter, a wrong script path, a wrong working directory or an import error just
    as readily as for the branch under test. This invocation differs from them in one
    respect — the map is usable — and must succeed. If it fails, the refusal tests are
    proving nothing about whose.py.
    """
    usable = tmp_path / "ownership_fixture.toml"
    usable.write_text(
        'sessions = ["probe-session"]\n'
        "\n"
        "[[claim]]\n"
        'id = "probe-claim"\n'
        'session = "probe-session"\n'
        'paths = ["tools/*.py"]\n'
        'note = "fixture claim for the positive control"\n'
        'claimed = "2026-01-01"\n',
        encoding="utf-8",
    )

    done = _run_whose(["tools/whose.py", "--map", str(usable)])

    assert done.returncode == 0, (
        "whose.py could not answer from a well-formed map, so the refusal tests above "
        f"may be passing for an unrelated reason.\nstdout:\n{done.stdout}\n"
        f"stderr:\n{done.stderr}"
    )
    assert "probe-session" in done.stdout, (
        f"expected the fixture's owner in the lookup output, got:\n{done.stdout}"
    )


# --------------------------------------------------------------------------- #
# the assumption underneath the guard
# --------------------------------------------------------------------------- #


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_working_specs_are_still_gitignored() -> None:
    """If the working area stops being ignored, the guard above stops meaning anything.

    Checked with `git check-ignore` rather than by parsing `.gitignore`, because the
    pattern semantics that matter here are git's own: `docs/*` plus a negation list is
    order-sensitive and directory-sensitive, and a reimplementation of that would be a
    second thing to get wrong. `check-ignore` exits 0 when a path is ignored and 1 when
    it is not, so it answers exactly the question asked. Paths are probed as strings and
    need not exist on disk.
    """
    probe = _git("check-ignore", "-q", "docs/SubAgent docs/some_working_spec.md")
    assert probe.returncode == 0, (
        "docs/SubAgent docs/ is NO LONGER gitignored. Either restore the `docs/*` rule "
        "in .gitignore, or — if working specs are now meant to be committed — delete "
        "this module, because the hazard it guards no longer exists.\n"
        f"git said: {probe.stderr.strip() or '(path is tracked or not ignored)'}"
    )


@pytest.mark.parametrize(
    "relpath",
    ["docs/index.md", "docs/USER_GUIDE.md", "docs/architecture.md", "docs/api/errors.md"],
)
def test_public_docs_are_still_negated(relpath: str) -> None:
    """The public surface must stay re-included, or a clone's site root is a 404.

    This is the other half of the same assumption: the guard permits reads of these
    paths, which is only correct while they are actually committed.
    """
    probe = _git("check-ignore", "-q", relpath)
    assert probe.returncode == 1, (
        f"{relpath} is now IGNORED. It is part of the public docs surface that "
        "TRACKED_DOCS_FILES / TRACKED_DOCS_DIRS permit code to read, so either restore "
        "its `!` negation in .gitignore or remove it from that permitted set."
    )


def test_public_docs_are_actually_tracked() -> None:
    """`not ignored` is weaker than `committed`; the guard's permission needs the latter."""
    listed = _git("ls-files", "docs")
    assert listed.returncode == 0, f"git ls-files failed: {listed.stderr}"
    tracked = {line.strip() for line in listed.stdout.splitlines() if line.strip()}

    for name in TRACKED_DOCS_FILES:
        assert f"docs/{name}" in tracked, (
            f"docs/{name} is permitted for reading but is not committed"
        )
    for name in TRACKED_DOCS_DIRS:
        assert any(p.startswith(f"docs/{name}/") for p in tracked), (
            f"docs/{name}/ is permitted for reading but nothing under it is committed"
        )
