"""Persist campaign results into the softae-next ``DataStore``.

Reuses the existing ``doe_parameters`` table (run_id, channel, iteration,
parameters_json, objective_value, acquisition_fn) — **no schema migration** — and
writes a JSON sidecar with the full per-step metrics under the run directory,
mirroring how Arrhenius sweeps persist a result file.

Channel is fixed at 0: a simulated campaign has no physical electrode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from softae.campaigns.config import BOCampaignConfig
from softae.campaigns.runner import CampaignResult

if TYPE_CHECKING:  # avoid importing hardware-coupled core at module load
    from softae.core.data_store import DataStore

_CAMPAIGN_CHANNEL = 0
_SIDECAR_NAME = "bo_campaign_result.json"


def record_campaign(
    data_store: "DataStore",
    result: CampaignResult,
    *,
    config: BOCampaignConfig | None = None,
    run_id: str | None = None,
    finish: bool = True,
) -> str:
    """Write *result* to the data store and return the ``run_id``.

    Creates a new run (``workflow_name="bo_campaign"``) unless an existing
    *run_id* is supplied.  Each campaign step becomes one ``doe_parameters`` row;
    the full result (including metrics) is saved as a JSON sidecar in the run
    directory.

    Parameters
    ----------
    data_store
        Open :class:`~softae.core.data_store.DataStore`.
    result
        The :class:`CampaignResult` to persist.
    config
        Optional config; its JSON becomes the run's ``config_snapshot`` and its
        acquisition name labels each DOE row.
    run_id
        Append to an existing run instead of starting a new one.
    finish
        Mark the run finished (only when this call created it).
    """
    acquisition = config.acquisition if config else "unknown"
    annotation = config.annotation if config else ""

    created = run_id is None
    if created:
        run_id = data_store.start_run(
            "bo_campaign",
            config_snapshot=config.to_json() if config else "{}",
            mode="simulation",
            campaign="bo_campaign",
            quality="explore",
            annotation=annotation,
        )

    for step in result.steps:
        data_store.record_doe_parameter(
            run_id,
            _CAMPAIGN_CHANNEL,
            int(step["iteration"]),
            dict(step["params"]),
            objective_value=float(step["sampled_value"]),
            acquisition_fn=acquisition,
        )

    # Full-fidelity sidecar (metrics, true optimum, convergence) next to the run.
    try:
        sidecar = data_store.run_dir(run_id) / _SIDECAR_NAME
        sidecar.write_text(result.to_json(), encoding="utf-8")
    except OSError:
        pass  # DB rows are the source of truth; sidecar is best-effort

    if created and finish:
        status = "done" if result.converged else "stopped"
        data_store.finish_run(run_id, status=status)

    return run_id
