"""softae.web — browser-based EIS visualizer powered by Dash + Plotly.

Launch with::

    python -m softae.web [--db PATH] [--port 8050]

or programmatically::

    from softae.web import run_server
    run_server(db_path="path/to/softae.db", port=8050)
"""

from __future__ import annotations

from softae.web.app import create_app

__all__ = ["create_app", "run_server"]


def run_server(
    *,
    db_path: str | None = None,
    port: int = 8050,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    """Create and start the Dash server (blocking call)."""
    import threading
    import webbrowser

    app = create_app(db_path=db_path)
    url = f"http://localhost:{port}"

    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    app.run(host="127.0.0.1", port=port, debug=debug)
