"""What the DataStore can honestly supply about a shadow run.

Split out of :mod:`softae.tools.shadow_review` so neither file carries two jobs: that
module parses a console log and renders a report, this one reads SQLite. The review
re-exports :func:`db_summary` so existing callers and tests are unaffected by the move.

Everything here is asked only for what it can answer without inventing. In particular
``fit_results.engine`` and ``fit_results.sigma_is_bound`` are reported as **stamped
defaults**, never as observations: ``analysis/eis/router.py`` does not pass ``report=``
to ``record_fit`` (P.18 is open), so a gated campaign still writes ``engine='legacy'``,
a NULL ``gate_verdict`` and an empty gate log. The structlog stream is the engine
evidence; this half supplies σ, R₁, and the two things the rows *do* record honestly —
which fits railed, and what the arc-closure annotator concluded.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


def db_summary(project: str, run_id: str | None) -> dict[str, Any]:
    """Per-channel measurement/σ rows for one run.  Read-only.

    Returns ``{"error": …}`` rather than raising: a missing or unreadable project must
    not cost the operator the log half of the review, which is the half that carries
    the verdicts.
    """
    from softae.core.data_store import DataStore

    store = None
    try:
        store = DataStore(project)
        if run_id is None:
            runs = store.query_runs()
            if not runs:
                return {"error": f"no runs recorded in {project}"}
            run_id = str(runs[0]["run_id"])
        fits = {int(f["measurement_id"]): f for f in store.query_fits(run_id=run_id)}
        rows = []
        engines = Counter()
        for m in store.query_measurements(run_id=run_id):
            fit = fits.get(int(m["measurement_id"]), {})
            engines[str(fit.get("engine") or "-")] += 1
            rows.append({
                "channel": int(m["channel"]),
                "measurement_id": int(m["measurement_id"]),
                "sigma": fit.get("sigma_S_per_cm"),
                "R1": fit.get("R1"),
                "engine": fit.get("engine"),
                "gate_verdict": fit.get("gate_verdict"),
            })
        return {"run_id": run_id, "rows": rows, "engines": engines}
    except Exception as exc:  # noqa: BLE001 - a CLI boundary over an optional input
        return {"error": str(exc)}
    finally:
        if store is not None:
            store.close()


#: Prefix ``engine._demote_if_railed`` writes into ``error_msg``.
RAILED_PREFIX = "railed fit:"


def railed_summary(project: str, run_id: str | None) -> dict[str, Any]:
    """Per-channel railed-fit counts, by **both** detectors.

    Two detectors because they see different eras, and neither alone sees the run.

    ``railed_new``
        Rows written since the railed-fit demotion landed carry ``success = 0``,
        ``R1 = NULL`` and an ``error_msg`` naming the bound. Unambiguous.
    ``railed_historical``
        Rows written before it carry ``success = 1``, ``R1`` sitting exactly on the
        model's floor and a σ computed from it, with **nothing** marking them —
        325 of 1440 rows in run ``20260811T023757Z``, every one reporting roughly
        seawater from a dry polymer film. Only the value betrays them.

    The bound is never restated here.
    :func:`~softae.analysis.equilibration.r1_lower_bound_ohms` reads it off
    ``circuit_fitting.CIRCUIT_MODELS``, so a bound edited in the registry moves this
    detector with it; a model declaring no bounds reports ``unknown``, never ``0``.
    """
    from softae.analysis.equilibration import is_railed, r1_lower_bound_ohms

    def _read(store: Any, run: str) -> dict[str, Any]:
        by_channel: dict[int, dict[str, Any]] = {}
        measurements = {int(m["measurement_id"]): m
                        for m in store.query_measurements(run_id=run)}
        for fit in store.query_fits(run_id=run):
            measurement = measurements.get(int(fit["measurement_id"]), {})
            channel = int(measurement.get("channel", -1))
            model = str(fit.get("model_name") or "")
            bound = r1_lower_bound_ohms(model)
            row = by_channel.setdefault(channel, {
                "channel": channel, "n_fits": 0, "n_success": 0, "railed_new": 0,
                "railed_historical": 0, "bound_ohm": "unknown",
                "sigmas": [], "R1s": [],
            })
            row["n_fits"] += 1
            if bound is not None:
                row["bound_ohm"] = bound
            success = bool(fit.get("success"))
            if success:
                row["n_success"] += 1
            if str(fit.get("error_msg") or "").startswith(RAILED_PREFIX):
                row["railed_new"] += 1
            elif success and is_railed(fit.get("R1"), bound):
                row["railed_historical"] += 1
            for key, column in (("sigmas", "sigma_S_per_cm"), ("R1s", "R1")):
                value = fit.get(column)
                if value is not None:
                    row[key].append(float(value))
        return {"run_id": run,
                "rows": [_finalise(r) for r in
                         sorted(by_channel.values(), key=lambda r: r["channel"])]}

    return _with_store(project, run_id, _read)


def _finalise(row: dict[str, Any]) -> dict[str, Any]:
    """Collapse the accumulated σ/R₁ lists to their medians."""
    import numpy as np

    for key, out in (("sigmas", "median_sigma"), ("R1s", "median_R1")):
        values = row.pop(key)
        row[out] = float(np.median(values)) if values else None
    return row


def arc_summary(project: str, run_id: str | None) -> dict[str, Any]:
    """Counts of ``arc_closure`` states across the run's stored gate logs.

    The auto-route path in ``analysis/eis/router.py`` passes ``arc.arc_provenance``
    unconditionally as ``report=``, so every row written by the current code carries
    exactly one real arc record — honest evidence, unlike the ``engine`` column beside
    it, because that shim exposes ``gate_log`` and nothing else and so cannot smuggle a
    stamped default in with it. Rows whose ``gate_log_json`` is ``[]`` predate the shim
    and are counted separately as ``no record``, never folded into an outcome they
    never reported.
    """
    def _read(store: Any, run: str) -> dict[str, Any]:
        states: Counter = Counter()
        no_record = 0
        for fit in store.query_fits(run_id=run):
            entries = _gate_log(fit.get("gate_log_json"))
            found = [e for e in entries
                     if isinstance(e, dict) and e.get("gate") == "arc_closure"]
            if not found:
                no_record += 1
                continue
            for entry in found:
                states[str(entry.get("state", "?"))] += 1
        return {"run_id": run, "states": states, "no_record": no_record,
                "sigma_is_bound": "stamped default — 0 on every row, not an "
                                  "observation. It becomes evidence once P.18 passes "
                                  "the real SpectrumReport to record_fit."}

    return _with_store(project, run_id, _read)


def _gate_log(raw: Any) -> list[Any]:
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _with_store(project: str, run_id: str | None, read: Any) -> dict[str, Any]:
    """Open, resolve the run, hand it to ``read``, close.  Errors become data.

    The same posture :func:`db_summary` takes and for the same reason: the DataStore is
    the *optional* half of this review, and losing it must never cost the log half.
    """
    from softae.core.data_store import DataStore

    store = None
    try:
        store = DataStore(project)
        if run_id is None:
            runs = store.query_runs()
            if not runs:
                return {"error": f"no runs recorded in {project}"}
            run_id = str(runs[0]["run_id"])
        return read(store, str(run_id))
    except Exception as exc:  # noqa: BLE001 - a CLI boundary over an optional input
        return {"error": str(exc)}
    finally:
        if store is not None:
            store.close()
