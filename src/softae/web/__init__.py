"""softae.web — browser-based EIS visualizer powered by Dash + Plotly.

Launch with::

    python -m softae.web [--db PATH] [--port 8050]

or programmatically::

    from softae.web import run_server
    run_server(db_path="path/to/softae.db", port=8050)
"""

from __future__ import annotations

__all__ = ["create_app", "run_server"]


def __getattr__(name: str):                      # PEP 562
    """Resolve ``create_app`` lazily so ``import softae.web`` needs no dash.

    dash, plotly and dash-bootstrap-components live in the optional ``[web]``
    extra, but ``create_app`` is documented public API of this package.  A
    module ``__getattr__`` keeps ``from softae.web import create_app`` working
    verbatim where the extra is installed, while leaving the bare package
    import — which ``__main__`` performs after argument parsing — free of it.
    """
    if name == "create_app":
        from softae.web.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # __getattr__ alone hides the name from dir() and tab-completion.
    return sorted(__all__)


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

    # Local, not module-scope: a module __getattr__ is not consulted for a
    # bare global name lookup inside this function's body.
    from softae.web.app import create_app

    app = create_app(db_path=db_path)
    url = f"http://localhost:{port}"

    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    app.run(host="127.0.0.1", port=port, debug=debug)
