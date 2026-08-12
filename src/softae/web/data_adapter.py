"""Data adapters: translate DataStore / files / live-poll → list[EISEntry].

All three adapters produce the same ``EISEntry`` dataclass that the Plotly
figure builders consume, keeping the figure code data-source agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from softae.analysis.circuit_fitting import FitResult
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant, cell_from_legacy_terms
from softae.analysis.eis_data import EISResult
from softae.analysis.eis_entry import EISEntry


def _cell_from_geometry(geo: dict | None) -> CellConstant | None:
    """The per-sample cell an ``{L_cm, t_cm, w_cm}`` dict implies, else ``None``.

    ``None`` for a missing, incomplete or degenerate geometry, which becomes
    ``sigma.mode == "unavailable"`` and then a blank σ — never a number built on a
    nominal thickness.

    This is now only the key extraction: the guard is
    :func:`~softae.analysis.eis.geometry.cell_from_legacy_terms`, shared with the GUI,
    the temperature sweep and the result router. It was previously spelled out here
    rather than imported from ``softae.gui.eis_sigma`` because a web adapter reaching
    into ``softae.gui`` would put the GUI package — and so Qt — on the headless import
    path. ``analysis.eis.geometry`` carries no such cost, which is why the shared guard
    lives there and not in the GUI module.
    """
    if not geo:
        return None
    try:
        return cell_from_legacy_terms(geo["L_cm"], geo["t_cm"], geo["w_cm"])
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# DBAdapter
# ---------------------------------------------------------------------------

class DBAdapter:
    """Read EIS measurements and fits from a softae DataStore (read-only).

    Parameters
    ----------
    db_path : str or Path
        Path to the ``softae.db`` file (or the project directory that
        contains ``db/softae.db``).
    """

    def __init__(self, db_path: str | Path) -> None:
        p = Path(db_path)
        # Accept either the .db file or the project root
        if p.is_dir():
            p = p / "db" / "softae.db"
        self._db_path = p

    def _connect(self):
        import sqlite3
        # Open read-only via URI
        uri = self._db_path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def list_runs(self) -> list[dict[str, Any]]:
        """Return [{run_id, started_at, workflow_name, n_measurements}] sorted newest-first."""
        if not self._db_path.exists():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT e.run_id, e.started_at, e.workflow_name,
                       COUNT(m.measurement_id) AS n_measurements
                FROM experiments e
                LEFT JOIN measurements m ON m.run_id = e.run_id
                GROUP BY e.run_id
                ORDER BY e.started_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def list_channels(self, run_ids: list[str]) -> list[int]:
        """Return sorted unique channel numbers present in the given runs."""
        if not self._db_path.exists() or not run_ids:
            return list(range(1, 9))
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(run_ids))
            rows = conn.execute(
                f"SELECT DISTINCT channel FROM measurements WHERE run_id IN ({placeholders}) ORDER BY channel",
                run_ids,
            ).fetchall()
            return [r["channel"] for r in rows]
        finally:
            conn.close()

    def get_entries(
        self,
        run_ids: list[str] | None = None,
        channels: list[int] | None = None,
        since: str | None = None,
    ) -> list[EISEntry]:
        """Load EISEntry objects matching filters from the DataStore."""
        if not self._db_path.exists():
            return []

        # Derive the project_dir as the parent of db/
        project_dir = self._db_path.parent.parent

        conn = self._connect()
        try:
            # Build measurement query
            params: list[Any] = []
            clauses: list[str] = []
            if run_ids:
                placeholders = ",".join("?" * len(run_ids))
                clauses.append(f"m.run_id IN ({placeholders})")
                params.extend(run_ids)
            if channels:
                placeholders = ",".join("?" * len(channels))
                clauses.append(f"m.channel IN ({placeholders})")
                params.extend(channels)
            if since:
                clauses.append("m.timestamp >= ?")
                params.append(since)

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            meas_rows = conn.execute(
                f"""
                SELECT m.*, e.workflow_name
                FROM measurements m
                JOIN experiments e ON e.run_id = m.run_id
                {where}
                ORDER BY m.timestamp
                """,
                params,
            ).fetchall()

            # Pre-fetch all relevant fit rows keyed by measurement_id
            fit_map: dict[int, list[dict]] = {}
            if meas_rows:
                all_mids = [r["measurement_id"] for r in meas_rows]
                ph = ",".join("?" * len(all_mids))
                fit_rows = conn.execute(
                    f"SELECT * FROM fit_results WHERE measurement_id IN ({ph}) ORDER BY fitted_at",
                    all_mids,
                ).fetchall()
                for fr in fit_rows:
                    fit_map.setdefault(fr["measurement_id"], []).append(dict(fr))
        finally:
            conn.close()

        entries: list[EISEntry] = []
        for row in meas_rows:
            row = dict(row)
            if row.get("eis_file_path") is None:
                continue
            p = Path(row["eis_file_path"])
            if not p.is_absolute():
                p = project_dir / p
            try:
                eis = EISResult.load(str(p))
            except Exception:
                continue

            fit: FitResult | None = None
            sigma: float | None = None
            fit_list = fit_map.get(row["measurement_id"], [])
            if fit_list:
                fr = fit_list[-1]
                raw_params = fr.get("parameters_json", "[]")
                try:
                    params_list = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
                    param_arr = np.array(params_list, dtype=float) if params_list else np.array([])
                except Exception:
                    param_arr = np.array([float(fr.get("R0") or 0), float(fr.get("R1") or 0)])

                fit = FitResult(
                    model_name=fr["model_name"],
                    parameters=param_arr,
                    R0=float(fr.get("R0") or 0),
                    R1=float(fr.get("R1") or 0),
                    R0_guess=float(fr.get("R0") or 0),
                    R1_guess=float(fr.get("R1") or 0),
                    z_indices=[0, 1],
                    success=bool(fr.get("success", 0)),
                    error_msg=str(fr.get("error_msg") or ""),
                )
                raw_sigma = fr.get("sigma_S_per_cm")
                sigma = float(raw_sigma) if raw_sigma is not None else None

            run_id_val: str = row["run_id"]
            label = f"Ch{row['channel']:02d} — {run_id_val[:16]}"
            entries.append(
                EISEntry(
                    label=label,
                    eis=eis,
                    fit=fit,
                    sigma=sigma,
                    run_id=run_id_val,
                )
            )
        return entries


# ---------------------------------------------------------------------------
# FileAdapter
# ---------------------------------------------------------------------------

class FileAdapter:
    """Load EIS entries from local text files.

    Parameters
    ----------
    paths : list of str or Path
        Paths to ``EISResult``-compatible text files.
    eis_model : str
        Circuit model to attempt fitting on load. Default ``"simpleSalt"``.
    electrode_geometry : dict or None
        Keys: ``L_cm``, ``t_cm``, ``w_cm`` — used to compute σ after fit.
    """

    def __init__(
        self,
        paths: list[str | Path],
        eis_model: str = "simpleSalt",
        electrode_geometry: dict | None = None,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        self._eis_model = eis_model
        self._geo = electrode_geometry

    def get_entries(self) -> list[EISEntry]:
        entries: list[EISEntry] = []
        for p in self._paths:
            if not p.exists():
                continue
            try:
                eis = EISResult.load(str(p))
            except Exception:
                continue
            fit: FitResult | None = None
            sigma: float | None = None
            try:
                # ``engine`` unset: ``[eis] engine`` decides here too, so the web
                # view and the GUI cannot end up reporting different physics.
                report = analyze_spectrum(
                    eis, cell=_cell_from_geometry(self._geo),
                    model_name=self._eis_model,
                )
                fit = report.fit
                if report.sigma.mode == "value":
                    sigma = float(report.sigma.value)
            except Exception:
                pass
            entries.append(
                EISEntry(
                    label=p.stem,
                    eis=eis,
                    fit=fit,
                    sigma=sigma,
                    run_id=None,
                )
            )
        return entries


# ---------------------------------------------------------------------------
# LiveAdapter
# ---------------------------------------------------------------------------

class LiveAdapter:
    """Returns entries from the most recently started run in the DataStore.

    Intended to be called on each Dash ``dcc.Interval`` tick.

    Parameters
    ----------
    db_path : str or Path
        Path to ``softae.db``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db = DBAdapter(db_path)

    def active_run_id(self) -> str | None:
        """Return the run_id of the most recent (possibly still running) run."""
        runs = self._db.list_runs()
        return runs[0]["run_id"] if runs else None

    def get_entries(self) -> list[EISEntry]:
        """Entries from the most recently started run."""
        run_id = self.active_run_id()
        if run_id is None:
            return []
        return self._db.get_entries(run_ids=[run_id])
