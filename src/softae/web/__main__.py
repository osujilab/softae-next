"""CLI entry point for the EIS web visualizer.

Usage::

    python -m softae.web [--db PATH] [--port PORT] [--no-browser] [--debug]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m softae.web",
        description="Launch the softae EIS web visualizer (Dash/Plotly).",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Path to softae DataStore .db file (auto-discover if omitted).",
    )
    try:
        from softae.config.loader import web_port as _web_port
        _default_port = _web_port()
    except Exception:
        _default_port = 8050

    parser.add_argument(
        "--port",
        type=int,
        default=_default_port,
        help=f"Port for Dash server (default: {_default_port} from softae_config.toml).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab on start.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug/hot-reload mode.",
    )

    args = parser.parse_args(argv)

    db_path: str | None = args.db
    if db_path is None:
        # Try auto-discovery via softae_config.toml in cwd or package root
        for candidate in [
            Path.cwd() / "softae_config.toml",
            Path(__file__).resolve().parents[4] / "softae_config.toml",
        ]:
            if candidate.exists():
                try:
                    import tomllib  # type: ignore
                    with open(candidate, "rb") as fh:
                        cfg = tomllib.load(fh)
                except ImportError:
                    try:
                        import tomli as tomllib  # type: ignore
                        with open(candidate, "rb") as fh:
                            cfg = tomllib.load(fh)
                    except ImportError:
                        cfg = {}
                project_dir = cfg.get("project_dir") or cfg.get("data", {}).get("project_dir")
                if project_dir:
                    db_candidate = Path(project_dir) / "db" / "softae.db"
                    if db_candidate.exists():
                        db_path = str(db_candidate)
                        break

    from softae.web import run_server

    print(f"Starting softae EIS Visualizer on http://localhost:{args.port}")
    if db_path:
        print(f"  DataStore: {db_path}")
    else:
        print("  DataStore: none (use sidebar to select a database)")

    run_server(
        db_path=db_path,
        port=args.port,
        debug=args.debug,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
