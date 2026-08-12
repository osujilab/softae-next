"""Dash app factory and callback registration."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from softae.web.layout import build_layout


def create_app(*, db_path: str | None = None) -> dash.Dash:
    """Create and configure the Dash application.

    Parameters
    ----------
    db_path : str or None
        Pre-populate the DB path input in the sidebar.

    Returns
    -------
    dash.Dash
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="softae EIS Visualizer",
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(initial_db_path=db_path)

    # Register all callbacks (imported here to avoid circular imports)
    from softae.web import callbacks as _cb  # noqa: F401
    _cb.register(app)

    return app
