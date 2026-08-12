"""Dash callbacks — all reactive logic for the EIS web visualizer.

Registered via ``register(app)`` from ``app.py``.
"""

from __future__ import annotations

import base64
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import dash
import pandas as pd
from dash import Input, Output, State, callback_context, dcc
from dash.exceptions import PreventUpdate

from softae.web.components import (
    build_arrhenius_figure,
    build_conductivity_figure,
    build_inspection_figure,
    build_overview_figure,
)
from softae.analysis.eis_entry import EISEntry


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entries_to_json(entries: list[EISEntry]) -> list[dict]:
    """Serialise entries to a JSON-safe list (arrays as lists)."""
    out = []
    for e in entries:
        out.append(
            {
                "label": e.label,
                "run_id": e.run_id,
                "sigma": e.sigma,
                "eis": e.eis.to_dict(),
                "fit": {
                    "model_name": e.fit.model_name,
                    "R0": float(e.fit.R0),
                    "R1": float(e.fit.R1),
                    "success": e.fit.success,
                    "error_msg": e.fit.error_msg,
                    "parameters": e.fit.parameters.tolist(),
                }
                if e.fit is not None
                else None,
            }
        )
    return out


def _entry_from_json(d: dict) -> EISEntry:
    """Reconstruct an EISEntry from serialised JSON dict."""
    import numpy as np
    from softae.analysis.circuit_fitting import FitResult
    from softae.analysis.eis_data import EISResult

    eis_d = d["eis"]
    eis = EISResult.from_arrays(
        channel=eis_d["channel"],
        f=np.array(eis_d["frequency"]),
        z_real=np.array(eis_d["z_real"]),
        z_imag_neg=np.array(eis_d["z_imag_neg"]),
    )

    fit = None
    if d.get("fit") is not None:
        fd = d["fit"]
        fit = FitResult(
            model_name=fd["model_name"],
            parameters=np.array(fd["parameters"]),
            R0=fd["R0"],
            R1=fd["R1"],
            R0_guess=fd["R0"],
            R1_guess=fd["R1"],
            z_indices=[0, 1],
            success=fd["success"],
            error_msg=fd.get("error_msg", ""),
        )
    return EISEntry(
        label=d["label"],
        eis=eis,
        fit=fit,
        sigma=d.get("sigma"),
        run_id=d.get("run_id"),
    )


def _load_entries(
    source: str,
    db_path: str,
    run_ids: list[str] | None,
    channels: list[int] | None,
    start_date: str | None,
    end_date: str | None,
    upload_contents: list | None,
    upload_names: list | None,
) -> list[EISEntry]:
    """Dispatch to the appropriate adapter and return entries."""
    if source == "db":
        if not db_path:
            return []
        from softae.web.data_adapter import DBAdapter
        adapter = DBAdapter(db_path)
        since = start_date  # ISO date string or None
        return adapter.get_entries(run_ids=run_ids or None, channels=channels or None, since=since)

    elif source == "files":
        if not upload_contents:
            return []
        from softae.web.data_adapter import FileAdapter
        paths: list[Path] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="softae_web_"))
        for content, fname in zip(upload_contents, upload_names or []):
            _, b64 = content.split(",", 1)
            data = base64.b64decode(b64)
            p = tmp_dir / fname
            p.write_bytes(data)
            paths.append(p)
        return FileAdapter(paths).get_entries()

    elif source == "live":
        if not db_path:
            return []
        from softae.web.data_adapter import LiveAdapter
        return LiveAdapter(db_path).get_entries()

    return []


# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------

def register(app: dash.Dash) -> None:
    """Attach all callbacks to *app*."""

    # ── 1. Show/hide sidebar sections based on source ────────────────────
    @app.callback(
        Output("db-path-section", "style"),
        Output("file-upload-section", "style"),
        Output("run-selector-section", "style"),
        Output("date-range-section", "style"),
        Output("poll-interval", "disabled"),
        Input("source-radio", "value"),
    )
    def toggle_sidebar_sections(source):
        db_style = {"display": "block"} if source in ("db", "live") else {"display": "none"}
        file_style = {"display": "block"} if source == "files" else {"display": "none"}
        run_style = {"display": "block"} if source == "db" else {"display": "none"}
        date_style = {"display": "block"} if source == "db" else {"display": "none"}
        poll_disabled = source != "live"
        return db_style, file_style, run_style, date_style, poll_disabled

    # ── 2. Populate run dropdown when db_path changes ────────────────────
    @app.callback(
        Output("run-selector", "options"),
        Output("run-selector", "value"),
        Input("db-path-input", "value"),
        Input("refresh-btn", "n_clicks"),
    )
    def update_run_options(db_path, _refresh):
        if not db_path:
            return [], []
        try:
            from softae.web.data_adapter import DBAdapter
            adapter = DBAdapter(db_path)
            runs = adapter.list_runs()
        except Exception:
            return [], []
        options = [
            {
                "label": f"{r['run_id'][:28]}  ({r['n_measurements']} meas.)",
                "value": r["run_id"],
            }
            for r in runs
        ]
        # Pre-select most recent run
        default = [runs[0]["run_id"]] if runs else []
        return options, default

    # ── 3. Main data load: entries-store + inspection dropdown ──────────
    @app.callback(
        Output("entries-store", "data"),
        Output("inspection-entry-dropdown", "options"),
        Output("inspection-entry-dropdown", "value"),
        Output("status-text", "children"),
        Input("refresh-btn", "n_clicks"),
        Input("poll-interval", "n_intervals"),
        State("source-radio", "value"),
        State("db-path-input", "value"),
        State("run-selector", "value"),
        State("channel-checklist", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("file-upload", "contents"),
        State("file-upload", "filename"),
        prevent_initial_call=True,
    )
    def load_data(
        _refresh, _intervals,
        source, db_path, run_ids, channels,
        start_date, end_date,
        upload_contents, upload_names,
    ):
        try:
            entries = _load_entries(
                source, db_path or "",
                run_ids, channels,
                start_date, end_date,
                upload_contents, upload_names,
            )
        except Exception as exc:
            return [], [], None, f"Error: {exc}"

        serialised = _entries_to_json(entries)
        dropdown_opts = [
            {"label": e.label, "value": i}
            for i, e in enumerate(entries)
        ]
        default_val = 0 if entries else None
        status = f"Loaded {len(entries)} entries"
        return serialised, dropdown_opts, default_val, status

    # ── 4. On click in Overview graph → switch tab + select entry ────────
    @app.callback(
        Output("selected-entry-idx", "data"),
        Output("main-tabs", "active_tab"),
        Output("inspection-entry-dropdown", "value", allow_duplicate=True),
        Input("overview-graph", "clickData"),
        Input("conductivity-graph", "clickData"),
        prevent_initial_call=True,
    )
    def on_graph_click(overview_click, cond_click):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        click_data = overview_click if trigger_id == "overview-graph" else cond_click
        if click_data is None:
            raise PreventUpdate
        try:
            idx = click_data["points"][0]["customdata"][0]
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate
        return idx, "tab-inspection", idx

    # ── 5. Overview figure ───────────────────────────────────────────────
    @app.callback(
        Output("overview-graph", "figure"),
        Input("entries-store", "data"),
        Input("grid-cols-slider", "value"),
    )
    def update_overview(data, cols):
        if not data:
            return build_overview_figure([], cols=cols or 4)
        entries = [_entry_from_json(d) for d in data]
        return build_overview_figure(entries, cols=cols or 4)

    # ── 6. Inspection figure ─────────────────────────────────────────────
    @app.callback(
        Output("inspection-graph", "figure"),
        Output("fit-metrics-table", "children"),
        Input("inspection-entry-dropdown", "value"),
        Input("entries-store", "data"),
    )
    def update_inspection(idx, data):
        if not data or idx is None:
            from softae.web.components.inspection import build_inspection_figure as _bif
            # Return empty placeholder
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.update_layout(height=640, paper_bgcolor="white",
                              annotations=[dict(text="Select an entry",
                                                xref="paper", yref="paper",
                                                x=0.5, y=0.5, showarrow=False,
                                                font=dict(size=16, color="#888"))])
            return fig, ""

        try:
            entry = _entry_from_json(data[int(idx)])
        except (IndexError, KeyError, ValueError):
            raise PreventUpdate

        fig = build_inspection_figure(entry)
        table = _build_fit_table(entry)
        return fig, table

    # ── 7. Conductivity figure ───────────────────────────────────────────
    @app.callback(
        Output("conductivity-graph", "figure"),
        Input("entries-store", "data"),
    )
    def update_conductivity(data):
        if not data:
            return build_conductivity_figure([])
        entries = [_entry_from_json(d) for d in data]
        return build_conductivity_figure(entries)

    # ── 8. Arrhenius figure ──────────────────────────────────────────────
    @app.callback(
        Output("arrhenius-graph", "figure"),
        Input("arrhenius-store", "data"),
        Input("arrhenius-unit-radio", "value"),
        Input("db-path-input", "value"),
        Input("run-selector", "value"),
        Input("refresh-btn", "n_clicks"),
    )
    def update_arrhenius(stored_data, unit, db_path, run_ids, _refresh):
        arrhenius_rows: list[dict] = stored_data or []
        if not arrhenius_rows and db_path:
            # Try to query from DataStore
            try:
                from softae.web.data_adapter import DBAdapter
                adapter = DBAdapter(db_path)
                conn = adapter._connect()
                try:
                    rows = conn.execute(
                        "SELECT * FROM arrhenius_results ORDER BY channel"
                    ).fetchall()
                    import sqlite3
                    for r in rows:
                        d = dict(r)
                        for key in ("temperatures_json", "conductivities_json"):
                            if d.get(key):
                                try:
                                    d[key.replace("_json", "_C") if "temp" in key
                                      else key.replace("_json", "_S_per_cm")] = json.loads(d[key])
                                except Exception:
                                    pass
                        arrhenius_rows.append(d)
                except Exception:
                    pass
                finally:
                    conn.close()
            except Exception:
                pass
        return build_arrhenius_figure(arrhenius_rows, unit=unit or "eV")

    # ── 9. Live status badge ─────────────────────────────────────────────
    @app.callback(
        Output("live-badge", "children"),
        Input("poll-interval", "n_intervals"),
        State("source-radio", "value"),
        State("db-path-input", "value"),
    )
    def update_live_badge(_, source, db_path):
        if source != "live" or not db_path:
            return ""
        try:
            from softae.web.data_adapter import LiveAdapter
            run_id = LiveAdapter(db_path).active_run_id()
            if run_id:
                return [
                    "● LIVE  ",
                    dash.html.Span(
                        run_id[:20],
                        style={"color": "#888", "fontSize": "10px"},
                    ),
                ]
        except Exception:
            pass
        return dash.html.Span("● No active run", style={"color": "#aaa"})

    # ── 10. CSV export ───────────────────────────────────────────────────
    @app.callback(
        Output("csv-download", "data"),
        Input("export-csv-btn", "n_clicks"),
        State("entries-store", "data"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks, data):
        if not data:
            raise PreventUpdate
        rows = []
        for d in data:
            rows.append(
                {
                    "label": d["label"],
                    "run_id": d.get("run_id"),
                    "channel": d["eis"]["channel"],
                    "sigma_S_per_cm": d.get("sigma"),
                    "R0_Ohm": d["fit"]["R0"] if d.get("fit") else None,
                    "R1_Ohm": d["fit"]["R1"] if d.get("fit") else None,
                    "fit_success": d["fit"]["success"] if d.get("fit") else None,
                }
            )
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "eis_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Fit metrics table builder
# ---------------------------------------------------------------------------

def _build_fit_table(entry: EISEntry):
    """Return a dbc.Table with fit metrics for the selected entry."""
    import dash_bootstrap_components as dbc
    from dash import html

    if entry.fit is None:
        return dbc.Alert("No fit available for this entry.", color="secondary", className="mt-2")

    rows_data = [
        ("Model", entry.fit.model_name),
        ("R₀ (Ω)", f"{entry.fit.R0:.6g}"),
        ("R₁ (Ω)", f"{entry.fit.R1:.6g}"),
        ("σ (S/cm)", f"{entry.sigma:.4e}" if entry.sigma is not None else "—"),
        ("Fit success", "✓" if entry.fit.success else "✗"),
        ("Error", entry.fit.error_msg or "—"),
    ]
    table_rows = [
        html.Tr([html.Td(html.Strong(k)), html.Td(v)])
        for k, v in rows_data
    ]
    return dbc.Table(
        [html.Tbody(table_rows)],
        bordered=False,
        striped=True,
        size="sm",
        className="mt-2",
        style={"maxWidth": "400px"},
    )
