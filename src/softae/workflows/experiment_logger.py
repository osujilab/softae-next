"""Structured experiment logger — JSON-lines provenance trail.

Every instrument call executed through the :class:`WorkflowExecutor` is
recorded as a single JSON line, enabling:

* Full reproducibility / provenance of experiments.
* Post-hoc querying by the autonomous agent or analysis code.
* Replacement for scattered ``print()`` statements in the original codebase.

Log files are written to ``<output_dir>/<workflow_name>_<timestamp>.jsonl``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from softae.workflows.workflow_model import WorkflowStep

logger = structlog.get_logger(__name__)


class ExperimentLogger:
    """Appends structured records to a JSON-lines file.

    Parameters
    ----------
    output_dir : str or Path
        Directory where log files are written.  Created if it does not exist.
    workflow_name : str
        Used as the file-name prefix.
    """

    def __init__(self, output_dir: str | Path, workflow_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = self.output_dir / f"{workflow_name}_{ts}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115

        logger.info("experiment_log_opened", path=str(self.log_path))

    # ── Public API ──────────────────────────────────────────────────────

    def log_step(
        self,
        workflow: str,
        step: WorkflowStep,
        duration_s: float,
        result: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write one step record.

        Parameters
        ----------
        workflow : str
            Workflow name (for multi-workflow sessions).
        step : WorkflowStep
            The step that was executed.
        duration_s : float
            Wall-clock seconds the step took.
        result : str
            ``"ok"`` on success, or an error summary.
        extra : dict, optional
            Additional key-value pairs to include in the record.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "workflow": workflow,
            "step": step.name,
            "instrument": step.instrument,
            "method": step.method,
            "params": _safe_serialize(step.params),
            "tags": step.tags,
            "duration_s": round(duration_s, 4),
            "result": result,
        }
        if extra:
            record["extra"] = extra

        self._write(record)

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Write a free-form event record (e.g. pause, resume, abort).

        Parameters
        ----------
        event : str
            Short event label.
        **kwargs
            Arbitrary payload.
        """
        record = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event": event,
            **{k: _safe_serialize(v) for k, v in kwargs.items()},
        }
        self._write(record)

    def close(self) -> None:
        """Flush and close the log file."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("experiment_log_closed", path=str(self.log_path))

    # ── Context manager ─────────────────────────────────────────────────

    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Internal ────────────────────────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        self._file.write(line + "\n")
        self._file.flush()


def _safe_serialize(obj: Any) -> Any:
    """Convert common non-JSON-serialisable types to safe representations."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(item) for item in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    # numpy arrays, Paths, etc.
    return str(obj)
