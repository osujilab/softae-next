"""Tests for tools/whose.py — the ownership lookup over OWNERSHIP.toml.

`tools/` is not an importable package (the shipped package is `src/softae/tools`),
so the module is loaded by path.

The load-bearing group is `TestAMapThatCannotBeTrusted`. This tool's natural
failure is the SUBAGENT_RULES §3 shape: a map that is missing, unparseable or
empty would make every question answer "nothing is foreign", which reads exactly
like a clean gate. Those tests are paired with a good-map control on the same
repository so that "it errored" is distinguishable from "it always errors".
"""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WHOSE_PATH = REPO_ROOT / "tools" / "whose.py"


def _load_whose():
    spec = importlib.util.spec_from_file_location("whose_tool", WHOSE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["whose_tool"] = module
    spec.loader.exec_module(module)
    return module


whose = _load_whose()


MAP = """
sessions = ["parallel-session", "afl-session", "eis-acq-session"]

[[claim]]
id = "p-literal"
session = "parallel-session"
paths = ["src/pkg/one.py"]
note = "a literal path"
claimed = 2026-08-01

[[claim]]
id = "p-glob"
session = "parallel-session"
paths = ["src/analysis/eis/**", "src/pkg/*.cfg"]
except = ["src/analysis/eis/granted.py"]
note = "a glob with a carve-out"
claimed = 2026-08-02

[[claim]]
id = "e-granted"
session = "eis-acq-session"
paths = ["src/analysis/eis/granted.py"]
note = "the carve-out's grantee"
claimed = 2026-08-03

[[claim]]
id = "p-shared"
session = "parallel-session"
paths = ["src/shared/thing.py"]
note = "first of two claimants"
claimed = 2026-08-04

[[claim]]
id = "e-shared"
session = "eis-acq-session"
paths = ["src/shared/thing.py"]
note = "second of two claimants"
claimed = 2026-08-05

[[claim]]
id = "a-done"
session = "afl-session"
paths = ["src/pkg/finished.py"]
note = "a claim that has been released"
claimed = 2026-08-06

[[release]]
claim = "a-done"
at = 2026-08-07
note = "landed as deadbee"

[[advisory]]
paths = ["src/pkg/config.toml"]
note = "SHARED, unclaimed on purpose"
"""


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = whose.main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture()
def map_path(tmp_path: Path) -> Path:
    path = tmp_path / "OWNERSHIP.toml"
    path.write_text(MAP, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# glob translation
# --------------------------------------------------------------------------- #


class TestGlobTranslation:
    @pytest.mark.parametrize(
        "pattern,path,expected",
        [
            ("src/pkg/one.py", "src/pkg/one.py", True),
            ("src/pkg/one.py", "src/pkg/two.py", False),
            ("src/analysis/eis/**", "src/analysis/eis/arc.py", True),
            ("src/analysis/eis/**", "src/analysis/eis/deep/nested/x.py", True),
            ("src/analysis/eis/**", "src/analysis/other.py", False),
            ("src/analysis/eis/**", "src/analysis/eisx/arc.py", False),
            ("src/*.py", "src/a.py", True),
            ("src/*.py", "src/sub/a.py", False),
            ("tools/eis_validate*.py", "tools/eis_validate_hold.py", True),
            ("tools/eis_validate*.py", "tools/other.py", False),
        ],
    )
    def test_star_stops_at_separator_and_doublestar_crosses(self, pattern, path, expected):
        assert bool(whose.glob_to_regex(pattern).search(path)) is expected


# --------------------------------------------------------------------------- #
# lookup mode
# --------------------------------------------------------------------------- #


class TestLookup:
    def test_a_literal_path_reports_its_single_owner(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/pkg/one.py"])
        assert code == whose.EXIT_OK
        assert "parallel-session" in out
        assert "UNCLAIMED" not in out
        assert "CONFLICT" not in out

    def test_a_glob_claim_covers_a_nested_path(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/analysis/eis/deep/arc.py"])
        assert code == whose.EXIT_OK
        assert "parallel-session" in out
        assert "[p-glob]" in out

    def test_an_except_pattern_hands_the_path_to_its_grantee_alone(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/analysis/eis/granted.py"])
        assert code == whose.EXIT_OK
        assert "eis-acq-session" in out
        # the broad claim carved this out, so it must NOT read as a conflict
        assert "CONFLICT" not in out
        assert "[p-glob]" not in out

    def test_an_unclaimed_path_reports_unclaimed_not_an_owner(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/pkg/nobody.py"])
        assert code == whose.EXIT_OK
        assert "UNCLAIMED" in out
        for session in ("parallel-session", "afl-session", "eis-acq-session"):
            assert session not in out

    def test_two_claimants_report_as_conflict_with_both_named(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/shared/thing.py"])
        assert code == whose.EXIT_OK
        assert "CONFLICT" in out
        assert "parallel-session" in out
        assert "eis-acq-session" in out
        assert "[p-shared]" in out and "[e-shared]" in out

    def test_a_released_claim_is_not_an_owner_but_is_still_readable(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/pkg/finished.py"])
        assert code == whose.EXIT_OK
        assert "UNCLAIMED" in out
        assert "released [a-done] afl-session" in out
        assert "landed as deadbee" in out

    def test_an_advisory_is_printed_and_never_confers_ownership(self, map_path):
        code, out, _ = _run(["--map", str(map_path), "src/pkg/config.toml"])
        assert code == whose.EXIT_OK
        assert "UNCLAIMED" in out
        assert "advisory: SHARED, unclaimed on purpose" in out


# --------------------------------------------------------------------------- #
# --staged gate
# --------------------------------------------------------------------------- #


@pytest.fixture()
def staged_repo(tmp_path: Path, monkeypatch):
    """A throwaway git repo whose index we can load. Returns a stage() helper."""
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "seed.txt")
    git("commit", "-q", "-m", "seed")

    def stage(rel: str) -> None:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")
        git("add", "--", rel)

    monkeypatch.chdir(root)
    stage.root = root  # type: ignore[attr-defined]
    stage.git = git  # type: ignore[attr-defined]
    return stage


class TestStagedGate:
    def test_only_our_own_files_staged_exits_zero(self, map_path, staged_repo):
        staged_repo("src/pkg/one.py")
        staged_repo("src/analysis/eis/arc.py")
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_OK
        assert "all parallel-session's" in out
        assert "FOREIGN" not in out

    def test_a_foreign_staged_file_exits_non_zero_and_names_the_owner(
        self, map_path, staged_repo
    ):
        staged_repo("src/pkg/one.py")
        staged_repo("src/analysis/eis/granted.py")
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_FINDINGS
        assert "FOREIGN" in out
        assert "src/analysis/eis/granted.py" in out
        assert "eis-acq-session" in out
        assert "[e-granted]" in out
        assert "DO NOT COMMIT" in out
        # the file that IS ours must not be listed as a finding
        assert out.count("src/pkg/one.py") == 0

    def test_an_unclaimed_staged_file_is_a_finding_not_an_all_clear(
        self, map_path, staged_repo
    ):
        staged_repo("src/pkg/brand_new.py")
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_FINDINGS
        assert "UNCLAIMED" in out
        assert "src/pkg/brand_new.py" in out

    def test_allow_unclaimed_is_the_explicit_visible_override(self, map_path, staged_repo):
        staged_repo("src/pkg/brand_new.py")
        code, out, _ = _run(
            [
                "--map",
                str(map_path),
                "--staged",
                "--me",
                "parallel-session",
                "--allow-unclaimed",
            ]
        )
        assert code == whose.EXIT_OK
        assert "UNCLAIMED" not in out

    def test_a_doubly_claimed_staged_file_is_a_conflict_even_when_one_owner_is_us(
        self, map_path, staged_repo
    ):
        staged_repo("src/shared/thing.py")
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_FINDINGS
        assert "CONFLICT" in out
        assert "parallel-session" in out and "eis-acq-session" in out

    def test_a_staged_rename_reports_the_vacated_path_too(self, map_path, staged_repo):
        # src/pkg/one.py is parallel's; move it out from under eis-acq's identity
        staged_repo("src/pkg/one.py")
        staged_repo.git("commit", "-q", "-m", "add one")  # type: ignore[attr-defined]
        staged_repo.git("mv", "src/pkg/one.py", "src/pkg/renamed.py")  # type: ignore[attr-defined]
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "eis-acq-session"]
        )
        assert code == whose.EXIT_FINDINGS
        # the OLD path carries the owner and --name-only would hide it entirely;
        # the NEW path must survive the two-field parse rather than being eaten
        # as the next status token.
        assert "src/pkg/one.py" in out
        assert "src/pkg/renamed.py" in out

    def test_nothing_staged_exits_zero(self, map_path, staged_repo):
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_OK
        assert "nothing staged" in out

    def test_an_unknown_me_is_an_error_not_a_tree_full_of_foreign_files(
        self, map_path, staged_repo
    ):
        staged_repo("src/pkg/one.py")
        code, out, err = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel"]
        )
        assert code == whose.EXIT_ERROR
        assert "not a known session" in err
        assert "FOREIGN" not in out


# --------------------------------------------------------------------------- #
# the map must fail loudly, never quietly
# --------------------------------------------------------------------------- #


class TestAMapThatCannotBeTrusted:
    """Each case pairs with `test_control_*` below: the control proves the same
    repository and the same staged file DO produce a real answer under a good
    map, so exit 2 here means "the map is broken", not "this tool errors."""

    def test_control_a_good_map_produces_a_finding_on_this_repo(
        self, map_path, staged_repo
    ):
        staged_repo("src/analysis/eis/granted.py")
        code, out, _ = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_FINDINGS
        assert "FOREIGN" in out

    def test_a_missing_map_errors_rather_than_reporting_nothing_foreign(
        self, tmp_path, staged_repo
    ):
        staged_repo("src/analysis/eis/granted.py")
        missing = tmp_path / "no_such_map.toml"
        code, out, err = _run(
            ["--map", str(missing), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_ERROR
        assert "not found" in err
        assert out == ""

    def test_an_unparseable_map_errors(self, tmp_path, staged_repo):
        staged_repo("src/analysis/eis/granted.py")
        broken = tmp_path / "broken.toml"
        broken.write_text('sessions = ["a"\n[[claim]\n', encoding="utf-8")
        code, out, err = _run(
            ["--map", str(broken), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_ERROR
        assert "not valid TOML" in err
        assert out == ""

    def test_a_map_with_no_claims_errors_instead_of_blessing_everything(
        self, tmp_path, staged_repo
    ):
        """The one that matters most: a syntactically fine but empty map would
        otherwise answer 'nothing is foreign' to every question."""
        staged_repo("src/analysis/eis/granted.py")
        empty = tmp_path / "empty.toml"
        empty.write_text('sessions = ["parallel-session"]\n', encoding="utf-8")
        code, out, err = _run(
            ["--map", str(empty), "--staged", "--me", "parallel-session"]
        )
        assert code == whose.EXIT_ERROR
        assert "no [[claim]] entries" in err
        assert "all parallel-session's" not in out
        assert out == ""

    def test_a_duplicate_claim_id_errors(self, tmp_path):
        path = tmp_path / "dup.toml"
        path.write_text(
            'sessions = ["a"]\n'
            '[[claim]]\nid = "x"\nsession = "a"\npaths = ["p"]\nnote = "n"\nclaimed = 2026-01-01\n'
            '[[claim]]\nid = "x"\nsession = "a"\npaths = ["q"]\nnote = "n"\nclaimed = 2026-01-01\n',
            encoding="utf-8",
        )
        code, _, err = _run(["--map", str(path), "p"])
        assert code == whose.EXIT_ERROR
        assert "duplicate claim id" in err

    def test_a_release_of_an_unknown_claim_errors(self, tmp_path):
        path = tmp_path / "orphan.toml"
        path.write_text(
            'sessions = ["a"]\n'
            '[[claim]]\nid = "x"\nsession = "a"\npaths = ["p"]\nnote = "n"\nclaimed = 2026-01-01\n'
            '[[release]]\nclaim = "typo"\nat = 2026-01-02\n',
            encoding="utf-8",
        )
        code, _, err = _run(["--map", str(path), "p"])
        assert code == whose.EXIT_ERROR
        assert "unknown claim id" in err

    def test_a_claim_naming_an_undeclared_session_errors(self, tmp_path):
        path = tmp_path / "stranger.toml"
        path.write_text(
            'sessions = ["a"]\n'
            '[[claim]]\nid = "x"\nsession = "typo-session"\npaths = ["p"]\n'
            'note = "n"\nclaimed = 2026-01-01\n',
            encoding="utf-8",
        )
        code, _, err = _run(["--map", str(path), "p"])
        assert code == whose.EXIT_ERROR
        assert "is not in the top-level" in err

    @pytest.mark.parametrize(
        "body,fragment",
        [
            pytest.param(
                'sessions = ["a"]\n[[claim]]\nsession = "a"\npaths = ["p"]\n'
                'note = "n"\nclaimed = 2026-01-01\n',
                "'id'",
                id="no-id",
            ),
            pytest.param(
                'sessions = ["a"]\n[[claim]]\nid = "x"\nsession = "a"\npaths = []\n'
                'note = "n"\nclaimed = 2026-01-01\n',
                "must not be empty",
                id="no-paths",
            ),
            pytest.param(
                'sessions = ["a"]\n[[claim]]\nid = "x"\nsession = "a"\npaths = ["p"]\n'
                'note = "n"\n',
                "claimed",
                id="no-claimed-stamp",
            ),
        ],
    )
    def test_an_incomplete_claim_errors(self, tmp_path, body, fragment):
        path = tmp_path / "incomplete.toml"
        path.write_text(body, encoding="utf-8")
        code, _, err = _run(["--map", str(path), "p"])
        assert code == whose.EXIT_ERROR
        assert fragment in err


# --------------------------------------------------------------------------- #
# usage + the shipped map
# --------------------------------------------------------------------------- #


class TestUsage:
    def test_staged_without_me_is_a_usage_error(self, map_path, staged_repo):
        code, _, err = _run(["--map", str(map_path), "--staged"])
        assert code == whose.EXIT_ERROR
        assert "requires --me" in err

    def test_staged_with_paths_is_a_usage_error(self, map_path, staged_repo):
        code, _, err = _run(
            ["--map", str(map_path), "--staged", "--me", "parallel-session", "a.py"]
        )
        assert code == whose.EXIT_ERROR
        assert "takes no path arguments" in err

    def test_no_arguments_at_all_is_a_usage_error(self, map_path, staged_repo):
        code, _, err = _run(["--map", str(map_path)])
        assert code == whose.EXIT_ERROR


class TestTheShippedMap:
    """The real map is gitignored coordination state, so it may be absent on a
    fresh checkout. When it is present it must parse and validate — this is the
    check that catches a hand-written append breaking it for everyone."""

    def test_the_shipped_map_parses_and_validates(self):
        shipped = REPO_ROOT / whose.DEFAULT_MAP
        if not shipped.exists():
            pytest.skip("OWNERSHIP.toml absent (gitignored coordination state)")
        omap = whose.load_map(shipped)
        assert omap.claims
        assert set(omap.sessions) >= {
            "parallel-session",
            "afl-session",
            "eis-acq-session",
        }
        for claim in omap.claims:
            assert claim.note, claim.id
            assert claim.paths, claim.id
