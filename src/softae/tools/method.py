"""``softae-method`` — lifecycle CLI for the method-maturity pipeline.

Drive a capability (catalog **method** or registered **recipe**) through the
maturity ladder and record hardware sign-off evidence.  See
``docs/METHOD_MATURITY_PIPELINE.md``.

    python -m softae.tools.method status [name]
    python -m softae.tools.method test <name>
    python -m softae.tools.method promote <name> <stage>
    python -m softae.tools.method sign-off <name> --run <run_id> --by <op> [--note "..."]

Exit codes: 0 ok · 1 test failure · 2 gate/usage error · 3 not found.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

from softae.config import loader
from softae.core import lifecycle as lc
from softae.core.lifecycle import GateError, Maturity
from softae.core.recipe_registry import RecipeRegistry
from softae.core.task_catalog import TaskCatalog
from softae.tools import use_utf8_console


# ── Registry access ──────────────────────────────────────────────────────────

def _load() -> tuple[TaskCatalog, RecipeRegistry]:
    return (
        TaskCatalog.load_toml(loader.tasks_toml_path()),
        RecipeRegistry.load_toml(loader.recipes_toml_path()),
    )


def _resolve(name: str, cat: TaskCatalog, reg: RecipeRegistry):
    """Return ``(kind, registry)`` for *name*, or ``(None, None)`` if unknown."""
    if name in reg:
        return "recipe", reg
    if name in cat:
        return "method", cat
    return None, None


def _save(kind: str, registry: Any) -> None:
    if kind == "recipe":
        registry.save_toml(loader.recipes_toml_path())
    else:
        registry.save_toml(loader.tasks_toml_path())


def _open_store():
    from softae.core.data_store import DataStore

    return DataStore(
        project_dir=loader.data_project_dir(),
        db_filename=loader.data_db_filename(),
    )


# ── status ───────────────────────────────────────────────────────────────────

def _cmd_status(args: argparse.Namespace) -> int:
    cat, reg = _load()

    if args.name:
        kind, registry = _resolve(args.name, cat, reg)
        if kind is None:
            print(f"unknown method/recipe '{args.name}'", file=sys.stderr)
            return 3
        if kind == "recipe":
            recipe = reg.get(args.name)
            eff = lc.effective_maturity(args.name, cat, reg)
            print(f"{args.name}  [recipe]  own={recipe.maturity}  effective={eff.label}")
            print(f"  methods: {', '.join(recipe.methods) or '(none)'}")
            for m in recipe.methods:
                mm = lc.method_maturity(m, cat).label if m in cat else "MISSING"
                print(f"    - {m}: {mm}")
            if recipe.evidence.get("tests"):
                print(f"  tests: {len(recipe.evidence['tests'])} linked")
        else:
            print(lc.status(args.name, cat).render())
        return 0

    # Table of everything.
    print(f"{'NAME':32} {'KIND':9} {'MATURITY':11} EFFECTIVE")
    for name in cat.list_names():
        m = lc.method_maturity(name, cat).label
        print(f"{name:32} {'method':9} {m:11} {m}")
    for name in reg.list_names():
        own = reg.get(name).maturity
        eff = lc.effective_maturity(name, cat, reg).label
        print(f"{name:32} {'recipe':9} {own:11} {eff}")
    return 0


# ── test ─────────────────────────────────────────────────────────────────────

def _cmd_test(args: argparse.Namespace) -> int:
    cat, reg = _load()
    kind, registry = _resolve(args.name, cat, reg)
    if kind is None:
        print(f"unknown method/recipe '{args.name}'", file=sys.stderr)
        return 3

    obj = registry.get(args.name)
    tests = obj.evidence.get("tests") or []
    if not tests:
        print(f"'{args.name}' has no linked tests (evidence.tests) to run", file=sys.stderr)
        return 2

    print(f"Running {len(tests)} linked test(s) for '{args.name}'…")
    proc = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"])
    if proc.returncode != 0:
        print(f"\nTests FAILED (exit {proc.returncode}); '{args.name}' not promoted.",
              file=sys.stderr)
        return 1

    current = Maturity.parse(obj.maturity)
    if current < Maturity.TESTED:
        try:
            lc.promote(args.name, Maturity.TESTED, registry)
            _save(kind, registry)
            print(f"\nTests passed → '{args.name}' promoted to 'tested'.")
        except GateError as exc:
            print(f"\nTests passed but promotion blocked:\n{exc}", file=sys.stderr)
            return 2
    else:
        print(f"\nTests passed. '{args.name}' already at '{current.label}'.")
    return 0


# ── versions ─────────────────────────────────────────────────────────────────

def _cmd_versions(args: argparse.Namespace) -> int:
    cat, _ = _load()
    chain = lc.version_chain(args.name, cat)
    if not chain:
        print(f"no versions found for '{args.name}'", file=sys.stderr)
        return 3
    for key in chain:
        t = cat.get(key)
        prov = t.provenance
        note = ""
        if prov.get("supersedes"):
            note = f"supersedes {prov['supersedes']}"
        elif prov.get("superseded_by"):
            note = f"superseded_by {prov['superseded_by']}"
        marker = "  <- current" if key == args.name else ""
        print(f"{key:30} v{t.version}  [{t.maturity}]  {note}{marker}")
    return 0


# ── promote ──────────────────────────────────────────────────────────────────

def _cmd_promote(args: argparse.Namespace) -> int:
    cat, reg = _load()
    kind, registry = _resolve(args.name, cat, reg)
    if kind is None:
        print(f"unknown method/recipe '{args.name}'", file=sys.stderr)
        return 3

    try:
        to = Maturity.parse(args.stage)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    store = _open_store() if to == Maturity.VALIDATED else None
    try:
        lc.promote(args.name, to, registry, data_store=store)
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()

    _save(kind, registry)
    print(f"'{args.name}' promoted to '{to.label}'.")
    return 0


# ── sign-off ─────────────────────────────────────────────────────────────────

def _cmd_signoff(args: argparse.Namespace) -> int:
    cat, reg = _load()
    kind, registry = _resolve(args.name, cat, reg)
    if kind is None:
        print(f"unknown method/recipe '{args.name}'", file=sys.stderr)
        return 3

    store = _open_store()
    try:
        lc.sign_off(
            args.name, run_id=args.run, by=args.by, catalog=registry,
            notes=args.note or "", data_store=store,
        )
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        store.close()

    _save(kind, registry)
    print(f"'{args.name}' signed off on run {args.run} by {args.by} → validated.")
    return 0


# ── Parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softae-method",
        description="Method-maturity lifecycle CLI (see docs/METHOD_MATURITY_PIPELINE.md).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="Show one capability, or a table of all.")
    s.add_argument("name", nargs="?", help="Method/recipe name (omit for a table).")
    s.set_defaults(func=_cmd_status)

    t = sub.add_parser("test", help="Run a capability's linked tests; promote to 'tested' on green.")
    t.add_argument("name")
    t.set_defaults(func=_cmd_test)

    v = sub.add_parser("versions", help="List a method's version chain (name + name@vN).")
    v.add_argument("name")
    v.set_defaults(func=_cmd_versions)

    pr = sub.add_parser("promote", help="Advance a capability to a maturity stage.")
    pr.add_argument("name")
    pr.add_argument("stage", help="draft | prototype | tested | validated")
    pr.set_defaults(func=_cmd_promote)

    so = sub.add_parser("sign-off", help="Record a hardware sign-off → validated.")
    so.add_argument("name")
    so.add_argument("--run", required=True, help="DataStore run_id proving the hardware run.")
    so.add_argument("--by", required=True, help="Operator recording the sign-off.")
    so.add_argument("--note", default="", help="Free-text sign-off note.")
    so.set_defaults(func=_cmd_signoff)

    return p


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
