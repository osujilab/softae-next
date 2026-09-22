"""``softae-catalog`` — seed a data root with the placeholder catalogs.

    softae-catalog init [--dest DIR]

Copies the four files in :mod:`softae.catalog_defaults` into the data root
(``data/`` by default) so a fresh clone has a catalog to compile against. The
copy is what an instance edits; the shipped files stay as they are.

It **refuses to overwrite**. A catalog under ``data/`` is an instance's own
work, developed against its own hardware, and silently replacing one with a
placeholder is the expensive mistake this command exists near. The refusal is
all-or-nothing: if any target file is already there, nothing at all is written.

So is the copy itself. The four files are staged in a temporary directory
beside the destination and only then moved in, and a failure partway removes
exactly the files this call had already placed — so a disk that fills on the
third file leaves a data root no more half-seeded than a refusal does.

Exit codes: 0 ok · 1 a copy failed and was rolled back · 2 refused (a target
already exists, or usage).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from softae.catalog_defaults import CATALOG_FILES, PLACEHOLDER_NOTICE, default_file
from softae.tools import use_utf8_console


def _dest_root(explicit: str | None) -> Path:
    """Where the copies go: ``--dest`` when given, else the configured root."""
    if explicit:
        return Path(explicit).expanduser()
    from softae.config import loader

    return loader.data_root()


def _discard(paths: Iterable[Path]) -> None:
    """Unlink exactly the named files, ignoring any already gone.

    Named files only, never a directory tree and never a glob: a rollback that
    recursed could take an instance's own catalog with it, which is the whole
    accident this module exists to avoid.
    """
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


def _discard_staging(staging: Path) -> None:
    """Remove the staged copies, then the (now empty) staging directory."""
    _discard(staging / name for name in CATALOG_FILES)
    try:
        staging.rmdir()
    except OSError:
        pass


def _stage(staging: Path) -> None:
    """Copy every shipped file into *staging*, naming the one that fails."""
    for name in CATALOG_FILES:
        with default_file(name) as source:
            try:
                shutil.copyfile(source, staging / name)
            except OSError as exc:
                raise OSError(f"could not stage {name}: {exc}") from exc


def _move_into_place(staging: Path, dest: Path, placed: list[Path]) -> None:
    """Move each staged file into *dest*, recording what landed in *placed*."""
    for name in CATALOG_FILES:
        try:
            os.replace(staging / name, dest / name)
        except OSError as exc:
            raise OSError(f"could not place {name} in {dest}: {exc}") from exc
        placed.append(dest / name)


def seed_catalog(dest: Path) -> list[Path]:
    """Copy every shipped catalog file into *dest*, or leave *dest* untouched.

    Returns the paths written, in :data:`CATALOG_FILES` order. The copies are
    staged beside *dest* and moved in, so a failure on any file rolls back the
    ones already placed rather than leaving a mixed data root.
    """
    present = [dest / name for name in CATALOG_FILES if (dest / name).exists()]
    if present:
        raise FileExistsError(
            "refusing to overwrite an existing catalog. Already in "
            f"{dest}:\n" + "".join(f"  {p.name}\n" for p in present) +
            "Nothing was written. Move or delete those files if you really "
            "want the placeholder catalogs here."
        )

    dest.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(dest.parent), prefix=".softae-catalog-"))
    placed: list[Path] = []
    try:
        _stage(staging)
        _move_into_place(staging, dest, placed)
    except BaseException:
        _discard(placed)
        raise
    finally:
        _discard_staging(staging)
    return placed


def _cmd_init(args: argparse.Namespace) -> int:
    dest = _dest_root(args.dest)
    try:
        written = seed_catalog(dest)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"{exc}\nRolled back: nothing was written to {dest}.",
              file=sys.stderr)
        return 1

    print(f"Wrote {len(written)} placeholder catalog file(s) to {dest}:")
    for path in written:
        print(f"  {path.name}")
    print()
    print(PLACEHOLDER_NOTICE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="softae-catalog",
        description="Seed a data root with the shipped placeholder catalogs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser(
        "init", help="Copy the placeholder catalogs into the data root.")
    init.add_argument(
        "--dest", default=None,
        help="Directory to write into (default: the configured data root).")
    init.set_defaults(func=_cmd_init)

    return p


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
