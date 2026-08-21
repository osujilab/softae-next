#!/usr/bin/env python3
"""whose.py — answer "who owns this path?" from a declared map, not by inference.

Three sessions (`parallel-session`, `afl-session`, `eis-acq-session`) share one
working tree and one git index. Working out who owns a modified file used to be
done by grepping diffs for topic markers and guessing. This reads the declaration
instead: `docs/SubAgent docs/OWNERSHIP.toml`.

Two modes:

    whose.py <path>...                        # look up owners, informational
    whose.py --staged --me <session>          # the pre-commit gate

Exit codes:

    0   nothing to act on
    1   the gate found something: a foreign, conflicted or unclaimed staged path
    2   the map is missing, unparseable or self-inconsistent; or git failed;
        or a usage error

2 is deliberately distinct from 1. A broken map must never be mistaken for a
clean answer — an empty map would otherwise report "nothing is foreign" to every
question, which is the failure that looks exactly like success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the project targets 3.11+
    raise SystemExit("whose.py needs Python 3.11+ for tomllib")

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

DEFAULT_MAP = Path("docs") / "SubAgent docs" / "OWNERSHIP.toml"


class MapError(Exception):
    """The ownership map cannot be trusted to answer anything."""


# --------------------------------------------------------------------------- #
# glob matching
# --------------------------------------------------------------------------- #


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob to a regex.

    `*` and `?` stop at a path separator; `**` crosses them. `fnmatch` is not
    used because its `*` crosses separators, which would silently widen every
    claim in the map.
    """
    out = ["^"]
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**", i):
                i += 2
                while i < n and pattern[i] == "*":
                    i += 1
                if i < n and pattern[i] == "/":
                    out.append("(?:.*/)?")
                    i += 1
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
                i += 1
        elif char == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def _matches(patterns: list[str], path: str) -> bool:
    return any(glob_to_regex(p).search(path) for p in patterns)


# --------------------------------------------------------------------------- #
# the map
# --------------------------------------------------------------------------- #


@dataclass
class Release:
    claim: str
    at: str
    note: str = ""


@dataclass
class Claim:
    id: str
    session: str
    paths: list[str]
    note: str
    claimed: str
    evidence: str = ""
    excepts: list[str] = field(default_factory=list)
    release: Release | None = None

    @property
    def active(self) -> bool:
        return self.release is None

    def covers(self, path: str) -> bool:
        if self.excepts and _matches(self.excepts, path):
            return False
        return _matches(self.paths, path)


@dataclass
class Advisory:
    paths: list[str]
    note: str

    def covers(self, path: str) -> bool:
        return _matches(self.paths, path)


@dataclass
class OwnershipMap:
    sessions: list[str]
    claims: list[Claim]
    advisories: list[Advisory]

    def active_owners(self, path: str) -> list[Claim]:
        return [c for c in self.claims if c.active and c.covers(path)]

    def past_owners(self, path: str) -> list[Claim]:
        return [c for c in self.claims if not c.active and c.covers(path)]

    def advice(self, path: str) -> list[str]:
        return [a.note for a in self.advisories if a.covers(path)]


def _stamp(value: object, where: str) -> str:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise MapError(f"{where}: needs a non-empty date or string")


def _str_list(value: object, where: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise MapError(f"{where}: expected a list of strings")
    if not value and not allow_empty:
        raise MapError(f"{where}: must not be empty")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MapError(f"{where}: every entry must be a non-empty string")
    return [item.strip() for item in value]


def _required_str(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MapError(f"{where}: missing required non-empty string '{key}'")
    return value.strip()


def load_map(path: Path) -> OwnershipMap:
    """Read and validate the map. Every failure here raises, loudly."""
    if not path.exists():
        raise MapError(
            f"ownership map not found: {path}\n"
            "  This tool cannot answer anything without it. It is coordination "
            "state and is gitignored, so a fresh checkout will not have one."
        )
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise MapError(f"ownership map is not valid TOML: {path}\n  {exc}") from exc
    except OSError as exc:
        raise MapError(f"ownership map could not be read: {path}\n  {exc}") from exc

    sessions = _str_list(raw.get("sessions"), "sessions")

    claim_tables = raw.get("claim")
    if not isinstance(claim_tables, list) or not claim_tables:
        raise MapError(
            f"ownership map has no [[claim]] entries: {path}\n"
            "  A map with no claims would report every path unowned and every "
            "staged file safe. Refusing to answer from it."
        )

    claims: list[Claim] = []
    seen: dict[str, Claim] = {}
    for index, table in enumerate(claim_tables):
        where = f"[[claim]] #{index + 1}"
        if not isinstance(table, dict):
            raise MapError(f"{where}: not a table")
        claim = Claim(
            id=_required_str(table, "id", where),
            session=_required_str(table, "session", where),
            paths=_str_list(table.get("paths"), f"{where} paths"),
            note=_required_str(table, "note", where),
            claimed=_stamp(table.get("claimed"), f"{where} claimed"),
            evidence=str(table.get("evidence", "")).strip(),
            excepts=_str_list(table.get("except", []), f"{where} except", allow_empty=True),
        )
        if claim.id in seen:
            raise MapError(f"{where}: duplicate claim id '{claim.id}'")
        if claim.session not in sessions:
            raise MapError(
                f"{where}: session '{claim.session}' is not in the top-level "
                f"`sessions` list {sessions}"
            )
        seen[claim.id] = claim
        claims.append(claim)

    for index, table in enumerate(raw.get("release", []) or []):
        where = f"[[release]] #{index + 1}"
        if not isinstance(table, dict):
            raise MapError(f"{where}: not a table")
        target = _required_str(table, "claim", where)
        claim = seen.get(target)
        if claim is None:
            raise MapError(f"{where}: releases unknown claim id '{target}'")
        if claim.release is None:
            claim.release = Release(
                claim=target,
                at=_stamp(table.get("at"), f"{where} at"),
                note=str(table.get("note", "")).strip(),
            )

    advisories: list[Advisory] = []
    for index, table in enumerate(raw.get("advisory", []) or []):
        where = f"[[advisory]] #{index + 1}"
        if not isinstance(table, dict):
            raise MapError(f"{where}: not a table")
        advisories.append(
            Advisory(
                paths=_str_list(table.get("paths"), f"{where} paths"),
                note=_required_str(table, "note", where),
            )
        )

    return OwnershipMap(sessions=sessions, claims=claims, advisories=advisories)


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def repo_root(start: Path) -> Path:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MapError(f"not a git working tree (or git unavailable): {start}\n  {exc}") from exc
    return Path(done.stdout.strip())


def staged_paths(root: Path) -> list[str]:
    """Repo-relative POSIX paths currently in the index.

    `--name-status` rather than `--name-only` because a staged rename stages a
    deletion of the old path too, and `--name-only` hides it.
    """
    try:
        done = subprocess.run(
            ["git", "diff", "--cached", "--name-status", "-z"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MapError(f"could not read the git index: {exc}") from exc

    fields = [f for f in done.stdout.split("\0")]
    out: list[str] = []
    i = 0
    while i < len(fields):
        status = fields[i]
        i += 1
        if not status:
            continue
        if i >= len(fields):
            break
        out.append(fields[i])
        i += 1
        if status[0] in ("R", "C"):
            if i < len(fields) and fields[i]:
                out.append(fields[i])
            i += 1
    # stable, de-duplicated; an empty field would mean the -z parse desynced
    return [p for p in dict.fromkeys(out) if p]


def normalize(raw: str, root: Path) -> str:
    text = raw.strip().replace("\\", "/")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            text = candidate.resolve().relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            return text
    while text.startswith("./"):
        text = text[2:]
    return text


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def _sort_by_specificity(claims: list[Claim], path: str) -> list[Claim]:
    """Longest matching pattern first — reported as information, never used to
    discard a claim. See the module docstring of OWNERSHIP.toml."""

    def key(claim: Claim) -> int:
        widths = [len(p) for p in claim.paths if glob_to_regex(p).search(path)]
        return -max(widths, default=0)

    return sorted(claims, key=key)


def report_lookup(paths: list[str], omap: OwnershipMap, out) -> int:
    width = max((len(p) for p in paths), default=0)
    for path in paths:
        owners = _sort_by_specificity(omap.active_owners(path), path)
        sessions = list(dict.fromkeys(c.session for c in owners))
        if not owners:
            print(f"{path.ljust(width)}  UNCLAIMED", file=out)
        elif len(sessions) > 1:
            print(f"{path.ljust(width)}  CONFLICT  {', '.join(sessions)}", file=out)
        else:
            print(f"{path.ljust(width)}  {sessions[0]}", file=out)
        pad = " " * (width + 2)
        for claim in owners:
            print(
                f"{pad}  claim [{claim.id}] {claim.session} "
                f"({claim.claimed}) {claim.note}",
                file=out,
            )
            if claim.evidence:
                print(f"{pad}    evidence: {claim.evidence}", file=out)
        for claim in omap.past_owners(path):
            assert claim.release is not None
            tail = f" — {claim.release.note}" if claim.release.note else ""
            print(
                f"{pad}  released [{claim.id}] {claim.session} "
                f"({claim.claimed} -> {claim.release.at}){tail}",
                file=out,
            )
        for note in omap.advice(path):
            print(f"{pad}  advisory: {note}", file=out)
    return EXIT_OK


def report_staged(
    paths: list[str],
    omap: OwnershipMap,
    me: str,
    *,
    allow_unclaimed: bool,
    out,
) -> int:
    findings: list[tuple[str, str, str]] = []
    for path in paths:
        owners = _sort_by_specificity(omap.active_owners(path), path)
        sessions = list(dict.fromkeys(c.session for c in owners))
        advice = omap.advice(path)
        suffix = f"  ({'; '.join(advice)})" if advice else ""
        if not owners:
            if not allow_unclaimed:
                findings.append(("UNCLAIMED", path, f"nobody has declared it{suffix}"))
        elif len(sessions) > 1:
            ids = ", ".join(f"{c.session} [{c.id}]" for c in owners)
            findings.append(("CONFLICT", path, f"{ids}{suffix}"))
        elif sessions[0] != me:
            claim = owners[0]
            findings.append(("FOREIGN", path, f"{claim.session} [{claim.id}]{suffix}"))

    if not findings:
        if not paths:
            print(f"nothing staged. ({me})", file=out)
        else:
            print(f"{len(paths)} staged path(s), all {me}'s.", file=out)
        return EXIT_OK

    kind_w = max(len(k) for k, _, _ in findings)
    path_w = max(len(p) for _, p, _ in findings)
    for kind, path, detail in findings:
        print(f"{kind.ljust(kind_w)}  {path.ljust(path_w)}  {detail}", file=out)
    print(
        f"{len(findings)} of {len(paths)} staged path(s) not {me}'s alone. DO NOT COMMIT.",
        file=out,
    )
    return EXIT_FINDINGS


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whose.py",
        description="Who owns this path? Reads docs/SubAgent docs/OWNERSHIP.toml.",
    )
    parser.add_argument("paths", nargs="*", help="repo-relative or absolute paths to look up")
    parser.add_argument(
        "--staged",
        action="store_true",
        help="read the staged file list from git and report anything not --me's",
    )
    parser.add_argument(
        "--me",
        metavar="SESSION",
        help="this session's identity; required with --staged. It cannot be inferred: "
        "all sessions share one checkout.",
    )
    parser.add_argument("--map", dest="map_path", help="path to the ownership map")
    parser.add_argument(
        "--allow-unclaimed",
        action="store_true",
        help="with --staged, do not treat an unclaimed staged path as a finding",
    )
    return parser


def _use_utf8(stream) -> None:
    """The map carries section marks and em dashes lifted verbatim from the
    mailbox. On a cp1252 console those raise mid-report, which would turn a
    finding into a traceback."""
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def main(argv: list[str] | None = None, *, out=None, err=None) -> int:
    if out is None:
        _use_utf8(sys.stdout)
    if err is None:
        _use_utf8(sys.stderr)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(argv)

    if args.staged and args.paths:
        print("whose.py: --staged takes no path arguments", file=err)
        return EXIT_ERROR
    if args.staged and not args.me:
        print("whose.py: --staged requires --me <session>", file=err)
        return EXIT_ERROR
    if not args.staged and not args.paths:
        print("whose.py: give paths, or --staged --me <session>", file=err)
        return EXIT_ERROR

    try:
        root = repo_root(Path.cwd())
        map_path = Path(args.map_path) if args.map_path else root / DEFAULT_MAP
        omap = load_map(map_path)

        if args.staged:
            if args.me not in omap.sessions:
                raise MapError(
                    f"--me '{args.me}' is not a known session. "
                    f"The map declares: {', '.join(omap.sessions)}"
                )
            paths = staged_paths(root)
            return report_staged(
                paths, omap, args.me, allow_unclaimed=args.allow_unclaimed, out=out
            )

        return report_lookup([normalize(p, root) for p in args.paths], omap, out)
    except MapError as exc:
        print(f"whose.py: {exc}", file=err)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
