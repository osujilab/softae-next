"""Dash page layout for the softae EIS web visualizer."""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

_SIDEBAR_STYLE: dict = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "260px",
    "padding": "16px 14px",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    "borderRight": "1px solid #dee2e6",
    "zIndex": 1000,
}

_CONTENT_STYLE: dict = {
    "marginLeft": "268px",
    "padding": "16px 20px",
    "minHeight": "100vh",
    "backgroundColor": "#ffffff",
}


def build_layout(initial_db_path: str | None = None) -> html.Div:
    """Return the top-level Dash layout."""
    return html.Div(
        [
            # ── Live polling interval (5 s) ──────────────────────────────
            dcc.Interval(id="poll-interval", interval=5_000, n_intervals=0, disabled=True),

            # ── Client-side stores ───────────────────────────────────────
            dcc.Store(id="entries-store", data=[]),       # serialised EISEntry list (metadata only)
            dcc.Store(id="selected-entry-idx", data=0),
            dcc.Store(id="arrhenius-store", data=[]),

            # ── Sidebar ──────────────────────────────────────────────────
            html.Div(
                _build_sidebar(initial_db_path),
                style=_SIDEBAR_STYLE,
                id="sidebar",
            ),

            # ── Main content ─────────────────────────────────────────────
            html.Div(
                _build_content(),
                style=_CONTENT_STYLE,
                id="content",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _build_sidebar(initial_db_path: str | None) -> list:
    return [
        html.H5("softae EIS Viewer", className="mb-3", style={"fontWeight": "700", "color": "#333"}),

        # ── Data source selector ─────────────────────────────────────────
        html.Label("Data source", className="fw-semibold small text-muted"),
        dcc.RadioItems(
            id="source-radio",
            options=[
                {"label": " DataStore", "value": "db"},
                {"label": " Local files", "value": "files"},
                {"label": " Live (active run)", "value": "live"},
            ],
            value="db",
            labelStyle={"display": "block", "marginBottom": "4px"},
            className="mb-3",
        ),

        # ── DB path (shown for 'db' and 'live') ─────────────────────────
        html.Div(
            [
                html.Label("Database path", className="fw-semibold small text-muted"),
                dcc.Input(
                    id="db-path-input",
                    type="text",
                    value=initial_db_path or "",
                    placeholder="/path/to/softae.db",
                    debounce=True,
                    style={"width": "100%", "fontSize": "12px"},
                    className="mb-3",
                ),
            ],
            id="db-path-section",
        ),

        # ── File upload (shown for 'files') ──────────────────────────────
        html.Div(
            [
                html.Label("Upload EIS files", className="fw-semibold small text-muted"),
                dcc.Upload(
                    id="file-upload",
                    children=html.Div(
                        ["Drag & Drop or ", html.A("Select Files")],
                        style={"textAlign": "center", "padding": "8px", "fontSize": "12px"},
                    ),
                    style={
                        "width": "100%",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "4px",
                        "borderColor": "#aaa",
                        "marginBottom": "12px",
                    },
                    multiple=True,
                    accept=".txt,.csv",
                ),
                html.Div(id="upload-file-list", className="small text-muted mb-3"),
            ],
            id="file-upload-section",
            style={"display": "none"},
        ),

        # ── Run selector ─────────────────────────────────────────────────
        html.Div(
            [
                html.Label("Run(s)", className="fw-semibold small text-muted"),
                dcc.Dropdown(
                    id="run-selector",
                    multi=True,
                    placeholder="Select run(s)…",
                    style={"fontSize": "12px"},
                    className="mb-3",
                ),
            ],
            id="run-selector-section",
        ),

        # ── Channel checklist ────────────────────────────────────────────
        html.Label("Channels", className="fw-semibold small text-muted"),
        dcc.Checklist(
            id="channel-checklist",
            options=[{"label": f" Ch{c:02d}", "value": c} for c in range(1, 9)],
            value=list(range(1, 9)),
            labelStyle={"display": "inline-block", "marginRight": "6px", "fontSize": "12px"},
            className="mb-3",
        ),

        # ── Date range ───────────────────────────────────────────────────
        html.Div(
            [
                html.Label("Date range", className="fw-semibold small text-muted"),
                dcc.DatePickerRange(
                    id="date-range",
                    display_format="YYYY-MM-DD",
                    style={"fontSize": "11px", "width": "100%"},
                    className="mb-3",
                ),
            ],
            id="date-range-section",
        ),

        # ── Overview grid columns ────────────────────────────────────────
        html.Label("Overview columns", className="fw-semibold small text-muted"),
        dcc.Slider(
            id="grid-cols-slider",
            min=1, max=8, step=1, value=4,
            marks={i: str(i) for i in range(1, 9)},
            className="mb-3",
        ),

        html.Hr(style={"margin": "8px 0"}),

        # ── Action buttons ───────────────────────────────────────────────
        dbc.Button(
            "Refresh", id="refresh-btn", color="primary", size="sm",
            className="w-100 mb-2",
        ),
        dbc.ButtonGroup(
            [
                dbc.Button("PNG", id="export-png-btn", color="secondary", size="sm", outline=True),
                dbc.Button("SVG", id="export-svg-btn", color="secondary", size="sm", outline=True),
                dbc.Button("CSV", id="export-csv-btn", color="secondary", size="sm", outline=True),
            ],
            className="w-100 mb-3",
        ),
        dcc.Download(id="csv-download"),

        html.Hr(style={"margin": "8px 0"}),

        # ── Live status badge ────────────────────────────────────────────
        html.Div(
            [
                html.Span(id="live-badge", style={"fontSize": "11px"}),
            ],
            className="text-center",
        ),
        html.Div(
            id="status-text",
            className="small text-muted mt-1 text-center",
            style={"fontSize": "11px"},
        ),
    ]


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

def _build_content() -> list:
    return [
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Overview",
                    tab_id="tab-overview",
                    children=[
                        dcc.Loading(
                            dcc.Graph(
                                id="overview-graph",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                    "toImageButtonOptions": {"format": "png", "scale": 2},
                                },
                                style={"minHeight": "500px"},
                            ),
                            type="circle",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Inspection",
                    tab_id="tab-inspection",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Select entry", className="fw-semibold small text-muted"),
                                        dcc.Dropdown(
                                            id="inspection-entry-dropdown",
                                            placeholder="Select an entry…",
                                            style={"fontSize": "12px"},
                                            className="mb-2",
                                        ),
                                    ],
                                    width=5,
                                ),
                            ],
                            className="mb-2",
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="inspection-graph",
                                config={
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                    "toImageButtonOptions": {"format": "png", "scale": 2},
                                },
                                style={"minHeight": "640px"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="fit-metrics-table"),
                    ],
                ),
                dbc.Tab(
                    label="Conductivity",
                    tab_id="tab-conductivity",
                    children=[
                        dcc.Loading(
                            dcc.Graph(
                                id="conductivity-graph",
                                config={
                                    "displayModeBar": True,
                                    "toImageButtonOptions": {"format": "png", "scale": 2},
                                },
                                style={"minHeight": "480px"},
                            ),
                            type="circle",
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Arrhenius",
                    tab_id="tab-arrhenius",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Eₐ unit", className="fw-semibold small text-muted"),
                                        dcc.RadioItems(
                                            id="arrhenius-unit-radio",
                                            options=[
                                                {"label": " eV", "value": "eV"},
                                                {"label": " kJ/mol", "value": "kJ/mol"},
                                            ],
                                            value="eV",
                                            labelStyle={"display": "inline-block", "marginRight": "10px"},
                                            className="mb-2",
                                        ),
                                    ],
                                    width=4,
                                )
                            ]
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="arrhenius-graph",
                                config={
                                    "displayModeBar": True,
                                    "toImageButtonOptions": {"format": "png", "scale": 2},
                                },
                                style={"minHeight": "480px"},
                            ),
                            type="circle",
                        ),
                    ],
                ),
            ],
            id="main-tabs",
            active_tab="tab-overview",
            className="mt-0",
        ),
    ]
