"""Result routers — the modality plug-in seam between executor and analysis.

This is Tier 1 of ``docs/SubAgent docs/afl_comparison_and_restructuring_spec.md``
(§4, "Result-router registry"): the ``_EIS_METHODS`` branch and the ~105-line
``_route_eis_to_datastore`` hook move out of :class:`WorkflowExecutor` intact, so
the generic execution layer no longer knows any EIS method name. The executor
iterates registered routers after each successful step; a future modality (image,
profilometry, …) registers a router here instead of editing the executor — the
seam spec Tier 2 component 5 (the modality registry) will land on.

Tier 2 component 2 upgraded the seam: ``handle`` returns a
:class:`~softae.analysis.measurement_result.MeasurementResult` (or ``None``)
alongside its persistence work. Component 3 made that payload durable — it is
written as netCDF beside the ``.txt`` and linked from the ``measurements`` row —
so the file, not just the live object, is the thing downstream analysis can read.

Both landed under the golden-run characterization test
(``tests/test_result_router_golden.py``): every column and file the EIS path
already produced is still produced identically, and the new payload is asserted
on top rather than in place of any of it.

Import direction: ``workflows/`` importing ``analysis/eis/`` follows the
existing dependency the routing already had (``data_store`` and the old executor
hook both import analysis-side types), never the reverse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant, cell_from_legacy_terms
from softae.analysis.eis_data import EISResult
from softae.analysis.measurement_result import MeasurementResult
from softae.core.conditions_capture import read_environment

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Annotation-only, and deliberately not a runtime import: `softae.workflows`
    # re-exports `WorkflowExecutor` from its `__init__`, and the executor imports
    # this module — so importing `workflow_model` here at runtime closes a cycle
    # that breaks whenever the router is imported *before* the executor. The
    # dependency is genuinely one-way (routers know steps; steps know nothing of
    # routers) and this keeps the import graph saying so.
    from softae.workflows.workflow_model import WorkflowStep

logger = structlog.get_logger(__name__)

#: Step methods this router claims. Moved verbatim from the executor's
#: ``_EIS_METHODS`` — the set is now the router's business, not the engine's.
EIS_METHODS: frozenset[str] = frozenset({"sendscript_getdata", "eis_extractdata"})

#: Electrode geometry a step may declare. These live on the *step*, never on
#: :class:`EISResult` — the instrument reports a spectrum, it has no idea what
#: it was measured across — so the router is the only place that sees both, and
#: therefore the only place that can attach geometry provenance to a payload.
ELECTRODE_GEOMETRY_PARAMS: frozenset[str] = frozenset({
    "electrode_L_cm", "electrode_t_cm", "electrode_w_cm",
    "electrode_x_mm", "electrode_y_mm",
})

#: Params consumed by the routing hook, not passed to instrument methods.
#: Declared by the router (via :attr:`ResultRouter.consumed_params`) so the
#: executor can filter them without hard-coding any modality's vocabulary.
ROUTING_PARAMS: frozenset[str] = frozenset({"circuit_model"}) | ELECTRODE_GEOMETRY_PARAMS

#: On-disk payload format, recorded in ``measurements.payload_format``. Stored as
#: data rather than inferred from the ``.nc`` suffix, so a future re-encoding is a
#: value change and not a filename convention nobody wrote down.
PAYLOAD_FORMAT = "netcdf4"

#: xarray engine for :meth:`xarray.Dataset.to_netcdf`. Named explicitly because
#: xarray's default engine depends on which optional backends happen to be
#: installed — a payload's format must be a decision, not an environment artifact.
PAYLOAD_ENGINE = "h5netcdf"


@dataclass
class SweepCounter:
    """Mutable acquisition-order counter, owned by the executor.

    Lives in the executor (one per run) and is *shared into* every
    :class:`RouterContext`, because ``sweep_order`` is counted at record time
    from what was actually recorded — a retry, a skipped channel, or a replay
    shifts every later position, and only the executor's single counter sees
    that sequence. A counter owned by the router would reset with the router's
    lifetime; a planner-supplied index would describe the *intended* order.
    """

    count: int = 0

    def next(self) -> int:
        """Advance and return the new position (1-based, like the old field)."""
        self.count += 1
        return self.count


@dataclass
class RouterContext:
    """Everything a router may need from the executor, passed per invocation.

    Carrying these explicitly (rather than handing routers the executor) keeps
    the dependency one-way: routers know this narrow contract, not the engine.
    """

    #: DataStore (or None). A missing store is a *skip*, not an error — routing
    #: is best-effort persistence beside a run that must not be blocked by it.
    data_store: Any | None = None
    run_id: str | None = None
    #: InstrumentManager, used only to snapshot temp/RH conditions at record
    #: time (:func:`read_environment` never raises and tolerates mocks).
    manager: Any | None = None
    #: Auto-fit toggle. True preserves the historical behavior (fit whenever a
    #: step declares ``circuit_model``); a host can switch fitting off wholesale
    #: without rewriting step params. Per-step model/geometry stay in
    #: ``step.params`` — they are the step author's declaration, not run state.
    auto_fit: bool = True
    #: Executor-owned acquisition counter (see :class:`SweepCounter`).
    sweep_counter: SweepCounter = field(default_factory=SweepCounter)


@runtime_checkable
class ResultRouter(Protocol):
    """A modality's claim on step results.

    ``matches`` must be cheap — it runs after every step. ``handle`` must never
    raise for persistence problems (a routing failure must not fail the step
    that physically succeeded); implementations own their own try/except, like
    the EIS router below.
    """

    #: Params the executor strips before calling the instrument, because they
    #: are addressed to this router rather than to the driver.
    consumed_params: frozenset[str]

    def matches(self, step: WorkflowStep) -> bool:
        """Does this router want *step*'s result?"""
        ...

    async def handle(self, step: WorkflowStep, raw_result: Any,
                     ctx: RouterContext) -> MeasurementResult | None:
        """Persist/route the raw result, returning it in contract form.

        ``None`` means "nothing to hand on" — the router declined, or its
        persistence failed. It is never an error signal: a router must still
        swallow its own exceptions, because a routing failure must not fail a
        step that physically succeeded.
        """
        ...


def _build_eis_result(step: WorkflowStep, raw_result: Any) -> EISResult:
    """Convert a raw EIS step result into a structured EISResult."""
    channel = int(step.params.get("chan", step.params.get("channel", 0)))
    return EISResult.from_raw(raw_result, channel=channel)


def _cell_from_params(step: WorkflowStep) -> CellConstant | None:
    """The per-sample cell this step's electrode geometry implies, else ``None``.

    Built from the same ``electrode_L/t/w_cm`` params ``record_fit`` already
    receives, so the geometry the engine analyses against and the geometry stored
    beside the row are one declaration and cannot disagree.

    ``None`` for a missing, incomplete or degenerate geometry — which the engine
    reports as ``sigma.mode == "unavailable"`` rather than a σ built on a nominal
    thickness. That guard was once spelled out here, in
    :func:`softae.gui.eis_sigma.gui_cell`, in
    :func:`softae.web.data_adapter._cell_from_geometry` and in the temperature
    sweep — four copies of the decision *whether a conductivity exists at all*. P.21
    collapsed them into
    :func:`~softae.analysis.eis.geometry.cell_from_legacy_terms`; what is left here is
    the key extraction, which is genuinely this module's own (step params, not a
    geometry dict).
    """
    try:
        return cell_from_legacy_terms(step.params["electrode_L_cm"],
                                      step.params["electrode_t_cm"],
                                      step.params["electrode_w_cm"])
    except (KeyError, TypeError):
        return None


def _sample_uuid(step: WorkflowStep) -> str | None:
    """The physical sample this step measures, or ``None`` if it declares none.

    Read from the step's tags rather than derived from anything here: the router
    sees a spectrum and the step that asked for it, and *neither* knows which
    well was consumed to make the sample. Minting is the wiring layer's job (it
    is the only layer that sees a trial being placed onto a channel), so this end
    of the spine only ever transcribes.

    Normalised to ``None`` for a missing **or empty** tag, so the "undeclared is
    unknown, never empty" convention holds at the one seam where a blank string
    would otherwise be written into a column and an ``attrs`` key.
    """
    tags = getattr(step, "tags", None) or {}
    value = tags.get("sample_uuid")
    return str(value) if value else None


def _to_measurement(step: WorkflowStep, eis_result: EISResult,
                    ctx: RouterContext, measurement_id: Any) -> MeasurementResult:
    """The recorded spectrum in contract form, with step-side provenance added.

    :meth:`EISResult.to_measurement` can only carry what the *result* holds.
    Two things it structurally cannot know are added here, because the router is
    the only place that sees both the spectrum and the step that requested it:

    1. **Electrode geometry** (SESSION_MAIL #6 commitment 3). Declared in
       ``step.params``, never on the result. Copied **present-only** — an absent
       key stays absent rather than becoming a zero or a guess, per the
       "undeclared is unknown, never empty" convention. A payload that omits
       ``electrode_L_cm`` is stating that the step did not declare one.
    2. **Row identity** — which run and which ``measurements`` row this payload
       belongs to, so a file on disk points back at its database row.

    Explicitly **out of scope**: ``deposit_area_mm2`` and the sessile
    "unavailable" case. Those resolve through ``core/geometry.py`` against board
    config, not through step params, and belong to T2.3 where that lookup and
    the schema epoch land together. Deriving an area here from
    ``electrode_L_cm × electrode_w_cm`` is precisely the 4.7×-wrong inference
    that P.5 corrected — those two are conduction geometry, not a footprint.

    3. **Sample identity** (T2.6). ``tags["sample_uuid"]``, minted where the well
       was consumed. Also present-only: a step carrying no tag yields a payload
       with **no** ``sample_uuid`` key at all, never an empty string — an absent
       identity and an anonymous one must not read alike, since the second would
       join to every other unidentified sample.
    """
    measurement = eis_result.to_measurement()

    geometry = {k: step.params[k] for k in sorted(ELECTRODE_GEOMETRY_PARAMS)
                if k in step.params}

    provenance: dict[str, Any] = {"step_name": step.name}
    if ctx.run_id is not None:
        provenance["run_id"] = ctx.run_id
    if measurement_id is not None:
        provenance["measurement_id"] = measurement_id
    sample_uuid = _sample_uuid(step)
    if sample_uuid is not None:
        provenance["sample_uuid"] = sample_uuid

    meta = {**measurement.meta, **provenance}
    if geometry:
        meta["electrode_geometry"] = geometry

    # `attrs` must stay netCDF-encodable, so only scalars go in; a geometry
    # value of an unexpected type is kept in `meta` (which has no such
    # constraint) rather than silently dropped or coerced.
    measurement.data.attrs.update(
        {k: v for k, v in {**provenance, **geometry}.items()
         if isinstance(v, (int, float, str))}
    )
    return replace(measurement, meta=meta)


def _write_payload(measurement: MeasurementResult, ctx: RouterContext,
                   file_stem: str, measurement_id: Any) -> None:
    """Write the payload beside the ``.txt`` and link it to its row (Tier 2 §3).

    The Dataset goes to ``runs/<run_id>/data/eis/<stem>.nc`` — same stem as the
    ``.txt``, so the transitional pair is obvious on disk and a reader can find
    either from the other without consulting the database.

    **Best-effort, and that is the whole point.** The measurement physically
    happened and its row is already committed; a full disk, a locked file or a
    missing HDF5 backend must not retract it. Every failure is swallowed here
    rather than by :meth:`EISResultRouter.handle`'s outer guard, because that one
    returns ``None`` — which would tell the caller the measurement failed, when in
    fact only its optional second copy did. The row simply keeps NULL payload
    columns, meaning exactly what NULL there means: no payload was written.

    The ``.txt`` is untouched and still authoritative during the transition (shim
    policy); it retires only once a full campaign has validated the round-trip.
    """
    try:
        payload_dir = ctx.data_store.payload_dir(ctx.run_id, measurement.modality)
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{file_stem}.nc"

        measurement.data.to_netcdf(payload_path, engine=PAYLOAD_ENGINE)
        ctx.data_store.set_measurement_payload(
            measurement_id, payload_path, PAYLOAD_FORMAT
        )
        logger.debug("eis_payload_written", measurement_id=measurement_id,
                     path=str(payload_path))
    except Exception:
        logger.warning("eis_payload_write_failed", stem=file_stem,
                       measurement_id=measurement_id, exc_info=True)


class EISResultRouter:
    """Routes EIS spectra into the DataStore — the old executor hook, moved.

    The body of :meth:`handle` is ``_route_eis_to_datastore`` verbatim (modulo
    ``self.``→``ctx.`` for executor state); behavioral changes here must go
    through the golden test first.
    """

    consumed_params: frozenset[str] = ROUTING_PARAMS

    def matches(self, step: WorkflowStep) -> bool:
        return step.method in EIS_METHODS

    async def handle(self, step: WorkflowStep, raw_result: Any,
                     ctx: RouterContext) -> MeasurementResult | None:
        """Persist EIS data to DataStore, and return it as a MeasurementResult.

        The legacy persistence half — the ``.txt`` file, the ``measurements``
        row's original columns, conditions and the auto-fit — is unchanged and
        must stay so; the golden test pins every one of them. Component 3 adds a
        netCDF payload beside the ``.txt`` (see :func:`_write_payload`), which
        cannot fail the measurement.

        Returns ``None`` when routing was skipped or failed, so a caller can
        never mistake a failed measurement for a recorded one. A *payload* that
        failed to write is not such a case: the measurement was recorded, so the
        result is returned and only the payload columns stay NULL.
        """
        if ctx.data_store is None or ctx.run_id is None:
            logger.debug("eis_autoroute_skip", reason="no datastore or run_id")
            # Deliberately still `None` rather than a store-less payload: the
            # early return happens *before* any parsing today, and building an
            # EISResult here would newly execute `from_raw` on a path that has
            # never run it — a behaviour change (it can raise) dressed as an
            # addition. Revisit in T2.3 if store-less runs need payloads.
            return None

        try:
            eis_result = _build_eis_result(step, raw_result)

            # Save EIS file to run directory
            eis_dir = ctx.data_store.eis_dir(ctx.run_id)
            eis_dir.mkdir(parents=True, exist_ok=True)
            # New step names already encode channel (eis_ch<N>_T<t>_RH<rh>);
            # legacy steps don't, so we keep the _ch<N> suffix for those.
            if re.match(r"^eis_ch\d+_T\d+_RH\d+$", step.name):
                file_stem = step.name
            else:
                file_stem = f"{step.name}_ch{eis_result.channel}"
            eis_path = eis_result.save(eis_dir / f"{file_stem}.txt")
            eis_result.raw_file_path = str(eis_path)

            # Persist measurement row. A commissioning sweep (E2) is an ordinary EIS
            # measurement carrying `role`/`fixture_id` on the step's tags — that tag
            # is the *only* thing distinguishing a blank from a sample, so it has to
            # be read here or the whole calibration lands in the database as sample
            # data and the commissioning pass is silently wasted.
            tags = getattr(step, "tags", None) or {}
            try:
                nominal = float(tags["nominal"]) if "nominal" in tags else None
            except (TypeError, ValueError):
                nominal = None
            electrode_x = step.params.get("electrode_x_mm")
            electrode_y = step.params.get("electrode_y_mm")

            # E6 §6 metadata. `sweep_order` is counted here rather than tagged: the
            # executor is the only thing that knows the acquisition sequence, and a
            # position supplied by the planner would describe the *intended* order,
            # which is the wrong one after a retry, a skip or a channel replay.
            sweep_order = ctx.sweep_counter.next()

            # `re_connection` is a genuine default rather than a placeholder for the
            # sample role. The RE is a stripe between CE and WE closed only by cast
            # material, so on a sample the loop is closed by the film -- a fact of the
            # workflow. Commissioning roles carry no film and must tag their own state
            # (a two-terminal reference is `tied_to_ce`), so they are not defaulted.
            role = str(tags.get("role", "sample"))
            default_re = "bridged_by_sample" if role == "sample" else "unverified"

            measurement_id = ctx.data_store.record_measurement(
                ctx.run_id,
                eis_result,
                electrode_x_mm=electrode_x,
                electrode_y_mm=electrode_y,
                role=role,
                fixture_id=tags.get("fixture_id"),
                electrode_mode=str(tags.get("electrode_mode", "unknown")),
                nominal_value=nominal,
                thermal_history=str(tags.get("thermal_history", "")),
                sweep_order=sweep_order,
                re_connection=str(tags.get("re_connection", default_re)),
                # Never defaulted true: cast material spanning the gap is not a
                # confirmation that it wets the reference stripe (R26).
                re_contact_verified=bool(tags.get("re_contact_verified", False)),
                # Tier 2 component 3. `modality` is known before the INSERT and
                # goes in it; the payload columns cannot, because the file names
                # the row it belongs to and so must be written after it exists.
                modality="eis",
                # Tier 2 component 6. Also in the INSERT: identity is known at
                # record time and must survive a failed payload write, which is
                # exactly when a row's provenance matters most.
                sample_uuid=_sample_uuid(step),
            )

            logger.info(
                "eis_autorouted",
                step=step.name,
                measurement_id=measurement_id,
                channel=eis_result.channel,
                role=str(tags.get("role", "sample")),
            )

            # Snapshot the chamber/stage/RH SP+PVs at measurement time.
            env = read_environment(ctx.manager)
            if any(v is not None for v in env.values()):
                ctx.data_store.record_conditions(
                    measurement_id, "measurement", **env
                )

            # Optional auto-fit
            circuit_model = step.params.get("circuit_model")
            if ctx.auto_fit and circuit_model:
                # P.20 site 9. ``engine`` is deliberately **unset**: mail [a23] is a
                # user ruling that one `[eis] engine` decides the physics at every
                # surface, and `engine.py`'s ``chosen = engine or cfg.engine`` is the
                # single place that reads it. Naming an engine here would make the
                # origin of nearly every stored ``fit_results`` row the one surface
                # that ignores the config.
                #
                # ``report=`` is deliberately **not** passed to ``record_fit``: that
                # is P.18, and it moves stored ``gate_verdict`` values.
                report = analyze_spectrum(
                    eis_result,
                    cell=_cell_from_params(step),
                    model_name=circuit_model,
                )
                # ``report.fit`` is ``None`` only on the gated engine's REJECT path
                # (R18 — an inadmissible spectrum is not handed to an optimiser), so
                # under the shipped ``engine = "legacy"`` this branch is never taken
                # and no row that exists today stops existing. When it is taken, no
                # fit row is the honest record: there was no fit. Writing one would
                # need a fabricated ``FitResult``, and raising would cost the
                # measurement its payload for a spectrum that physically happened.
                if report.fit is None:
                    logger.info(
                        "eis_fit_not_admitted",
                        step=step.name,
                        measurement_id=measurement_id,
                        model=circuit_model,
                    )
                else:
                    ctx.data_store.record_fit(
                        measurement_id,
                        report.fit,
                        L_cm=step.params.get("electrode_L_cm"),
                        t_cm=step.params.get("electrode_t_cm"),
                        w_cm=step.params.get("electrode_w_cm"),
                    )
                    logger.info(
                        "eis_fit_autorouted",
                        step=step.name,
                        measurement_id=measurement_id,
                        model=circuit_model,
                    )

            measurement = _to_measurement(step, eis_result, ctx, measurement_id)
            _write_payload(measurement, ctx, file_stem, measurement_id)
            return measurement

        except Exception:
            logger.warning("eis_autoroute_failed", step=step.name, exc_info=True)
            return None
