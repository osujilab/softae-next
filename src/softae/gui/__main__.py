"""SoftAE GUI entry point.

Launch with::

    python -m softae.gui
    # or
    softae-gui
"""

from __future__ import annotations

import asyncio
import sys

from softae.gui.app import run_app


def main() -> None:
    """CLI entry point for the SoftAE GUI."""
    sys.exit(run_app())


if __name__ == "__main__":
    main()
