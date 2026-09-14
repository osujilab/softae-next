"""The production read — one authoritative sweep, taken **after** settling.

What this exists to change
--------------------------
Today a settling campaign scores the **last settle round**. ``_equilibrate``
(``core/autonomous_wiring.py:3611-3614``) takes the raws
:func:`~softae.core.autonomous_wiring.drive_settle_phase` hands back and injects
them under the trial's own measure-step name::

    for channel, name in targets.items():
        raw = last_raws.get(channel)
        if raw is not None:
            step_results[name] = raw

Those raws came from a step tagged ``measurement="settle"`` — a tag
``settle_measure_step`` applies deliberately, *"so a round cannot enter the
objective by itself"*. The campaign then scores it anyway, by name, because the
driver chose it explicitly. That is a defensible choice and it has been the
behaviour of **every campaign score to date whenever settling was enabled**, but
it is not what :func:`~softae.core.autonomous_wiring.is_primary_measurement`
(``autonomous_wiring.py:2497``) describes, and it makes "the reading the campaign
recorded" a round whose whole purpose was to be evidence *about* readings.

:func:`take_production_read` replaces it with a real one: a fresh sweep taken
after ``drive_settle_phase`` returns, carrying the default ``measurement=
"primary"`` / ``role="sample"`` tags, which is exactly what
``is_primary_measurement`` selects and what makes "production" mean
*authoritative*.

Three properties, and each is load-bearing
------------------------------------------
* **It runs unconditionally.** With no ``measurement=`` override it uses the
  campaign's own ``spec.measurement``, so the only difference from today is that
  the recorded reading is a real primary sweep taken after settling rather than a
  recycled settle round. The read is defined by its **role**, not by its
  parameters — a denser preset only changes the acquisition.
* **An override prepares its own scripts first.** The step carries only a *path*
  to a ``.mscr``; a denser preset that nobody wrote scripts for would silently
  read the campaign's own sweep while recording the denser preset as provenance.
  So an override calls ``modality.prepare_run(...)`` here, before the step is
  built, under :mod:`~softae.core.eis_scripts`'s always-overwrite rule.
* **The step comes from the modality's own ``build_measure_step``.** Same route
  the settle rounds take, so the two cannot drift into different sweeps.

The call ``core/autonomous_wiring._equilibrate`` needs
------------------------------------------------------
Verbatim, replacing the injection loop quoted above (``phase`` being the
EQUILIBRATE/MEASURE phase whose ``measurement`` block may override the preset,
or ``None`` where no run plan carries one)::

    raws = await take_production_read(
        spec, round_channels, measurement=production_measurement,
        executor=WorkflowExecutor(manager, data_store=data_store, run_id=run_id),
        sample_uuid_by_channel=trial_sample_uuids)
    for channel, name in targets.items():
        raw = raws.get(channel)
        if raw is not None:
            step_results[name] = raw

A fresh :class:`~softae.workflows.workflow_executor.WorkflowExecutor` per read is
what ``_settle_round`` already does for each round, and for the same reason: this
function takes ownership of ``executor.on_step_complete`` to capture results (it
restores the previous value, but a shared executor is still a shared callback).

.. note::
   The trial's own pre-settle sweep is already tagged ``measurement="primary"``
   and has already fired ``on_trial_measured``, so a trial now carries **two**
   primary-tagged readings. That is a real consequence, surfaced rather than
   worked around here: the campaign scores by step *name*, and the name the loop
   injects under is still the trial's own. Nothing in this module can decide what
   the second row should mean.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import structlog

from softae.workflows.workflow_model import Workflow, WorkflowStep

if TYPE_CHECKING:  # annotation-only: this module sits below the campaign path
    from softae.core.autonomous_wiring import CampaignSpec
    from softae.core.measurement_spec import MeasurementSpec

logger = structlog.get_logger(__name__)

__all__ = [
    "PRODUCTION_STEP_PREFIX",
    "production_step_name",
    "build_production_read_workflow",
    "take_production_read",
]

#: Prefix distinguishing the production sweep's step from the trial's own.
#:
#: A second read of the same channel in the same run always gets its own step
#: name here — ``settle_eis_ch{N}_r{i}`` and ``confirm_eis_ch{N}_r{a}`` are the
#: two existing instances. The *tags* are what decide eligibility, so renaming
#: costs nothing and keeps the two readings separable in the record.
PRODUCTION_STEP_PREFIX = "production"


def production_step_name(measure_step_name: str) -> str:
    """``'production_measure_eis_ch3'`` — the modality's own step name, prefixed.

    Derived from the step rather than spelled out, so a modality whose steps are
    not called ``measure_eis_*`` needs no entry here.
    """
    return f"{PRODUCTION_STEP_PREFIX}_{measure_step_name}"


def _resolve(
    spec: "CampaignSpec", measurement: "MeasurementSpec | None"
) -> "MeasurementSpec":
    """The sweep the production read actually takes.

    An explicit *measurement* wins; otherwise the campaign's own block; otherwise
    the modality defaults — the same ``measurement or _MSpec()`` posture
    ``settle_measure_step`` and ``confirmation_measure_step`` already take, with
    ``spec.measurement`` (which is ``None`` on a campaign that never named one)
    in between.
    """
    from softae.core.measurement_spec import MeasurementSpec as _MSpec

    return measurement or getattr(spec, "measurement", None) or _MSpec()


def build_production_read_workflow(
    spec: "CampaignSpec",
    channels: Sequence[int],
    *,
    measurement: "MeasurementSpec | None" = None,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> "Workflow | None":
    """The one-round workflow :func:`take_production_read` runs, or ``None``.

    ``None`` when the campaign does not measure (``enabled=False`` — *formulate
    and cast, but do not measure*) or when the modality builds no per-electrode
    step. Both are the same statement: there is no production read to take, and a
    workflow that pretended otherwise would send a sweep the spec refused.

    **Preparing the scripts is part of building**, not a separate step a caller
    could forget: an override's ``.mscr`` must exist before the step that names
    it is built, because ``build_measure_step`` asks
    :func:`~softae.core.eis_scripts.variant_for` which file was written for this
    sweep and silently answers "the run's base" for one that was never prepared.
    """
    from softae.core.autonomous_wiring import _MEASUREMENT_TAGS_KEY
    from softae.core.modality_registry import get_modality

    channels = [int(ch) for ch in channels]
    production = _resolve(spec, measurement)
    if not production.enabled:
        logger.info("production_read_skipped", campaign=getattr(spec, "name", None),
                    detail="measurement is disabled for this campaign")
        return None

    modality = get_modality(production.modality)
    if measurement is not None:
        # `as_variant=True` is the whole difference between *"this is the run"*
        # and *"this is an extra sweep inside the run"*. The default
        # (`as_variant=False`) calls `eis_scripts.begin_run`, which declares
        # these parameters the run's BASE, clears every prepared variant and
        # writes the production sweep to the unsuffixed path — the path every
        # settle round reads. A later round would then measure the production
        # sweep while recording the campaign's own preset as provenance, which
        # is the exact substitution `eis_scripts` exists to prevent, inverted.
        modality.prepare_run(production, channels, as_variant=True)

    steps: list[WorkflowStep] = []
    for channel in channels:
        step = modality.build_measure_step(channel, production)
        if step is None:
            continue
        # Tags are *inherited*, never rewritten: `build_measure_step` already
        # emits `measurement="primary"` and leaves `role` at its "sample"
        # default, which is precisely what `is_primary_measurement` selects.
        # Re-stating them here would be a second place for the vocabulary to
        # drift from `autonomous_wiring.py:2497`.
        steps.append(replace(step, name=production_step_name(step.name),
                             params=dict(step.params), tags=dict(step.tags)))
    if not steps:
        logger.info("production_read_skipped", campaign=getattr(spec, "name", None),
                    detail=f"modality '{production.modality}' builds no per-channel step")
        return None

    wf = Workflow(
        name=f"{getattr(spec, 'name', 'campaign')}_production",
        description="Production read — the authoritative post-settle sweep",
        setup=steps,
        iterations=1,
        metadata={"source": "production_read",
                  "campaign": getattr(spec, "name", None),
                  _MEASUREMENT_TAGS_KEY: {s.name: dict(s.tags) for s in steps}},
    )
    if sample_uuid_by_channel:
        # One sample, several measurements: the production read re-reads the
        # films this trial cast, so it carries their identities exactly as the
        # settle rounds and the confirmation sweeps do.
        from softae.core.autonomous_wiring import _stamp_sample_uuids

        _stamp_sample_uuids(wf, sample_uuid_by_channel)
    return wf


async def take_production_read(
    spec: "CampaignSpec",
    channels: Sequence[int],
    *,
    executor: Any,
    measurement: "MeasurementSpec | None" = None,
    sample_uuid_by_channel: Mapping[int, str] | None = None,
) -> dict[int, Any]:
    """Take the authoritative post-settle sweep; return ``{channel: raw}``.

    Called **after** :func:`~softae.core.autonomous_wiring.drive_settle_phase`
    returns — the ordering is the point of the whole function, and it is the
    caller's to honour. See the module docstring for the exact call site.

    Returns one entry per channel, ``None`` where the step did not complete,
    mirroring ``_settle_round``: a shorter mapping would read downstream as a
    smaller board rather than as a missing reading. ``{}`` means no read was
    taken at all (measurement disabled, or a modality with no per-channel step).
    """
    channels = [int(ch) for ch in channels]
    wf = build_production_read_workflow(
        spec, channels, measurement=measurement,
        sample_uuid_by_channel=sample_uuid_by_channel)
    if wf is None:
        return {}

    captured: dict[str, Any] = {}
    previous = getattr(executor, "on_step_complete", None)
    executor.on_step_complete = (
        lambda step, idx, total, result, *_: captured.update({step.name: result}))
    try:
        await executor.run(wf)
    finally:
        executor.on_step_complete = previous

    by_channel = {int(s.tags["channel"]): s.name for s in wf.setup
                  if "channel" in s.tags}
    raws = {ch: captured.get(by_channel.get(ch, "")) for ch in channels}
    logger.info("production_read_taken", campaign=getattr(spec, "name", None),
                channels=channels,
                measured=sum(1 for v in raws.values() if v is not None),
                override=measurement is not None)
    return raws
