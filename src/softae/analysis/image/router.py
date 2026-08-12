"""Result router for the analysis-only ``image`` modality (Tier 2, T2.7).

The EIS router (``analysis/eis/router.py``) was extracted from the executor; this
one was never in it. That difference is the whole point of T2.7: a second data
stream reaches storage through the same seam **without a single edit to
``workflows/``, ``core/`` or ``analysis/eis/``**. Everything this module needs
from the engine arrives as arguments — the step, its raw result, and a
:class:`~softae.analysis.eis.router.RouterContext` — and everything it produces
leaves as a :class:`~softae.analysis.measurement_result.MeasurementResult` plus a
netCDF file beside it.

**This package imports nothing from ``analysis.eis``, not even for typing.** Two
modalities that import each other are not a plug-in system; they are one module
in two files. The consequence is visible in the signatures below: ``ctx`` is
annotated ``Any`` because the two types that describe the seam —
``ResultRouter`` (the Protocol) and ``RouterContext`` — currently live *inside*
``analysis/eis/router.py``. That is a pre-existing wart rather than a decision:
they are modality-neutral contracts sitting in a modality's package, so every
second modality must either import EIS or duck-type. This one duck-types, and
only ever reads ``ctx.data_store`` / ``ctx.run_id``. See the follow-up note in
TASKS.md T2.7 — moving them to a neutral module is a ``analysis/eis/`` edit,
which T2.7 was forbidden to make, and rightly: the point of the task is that the
new stream costs the existing one nothing.

Discovered seam limit — no ``measurements`` row is written
---------------------------------------------------------
An image payload lands on disk at the T2.3 location
(``runs/<run_id>/data/image/<stem>.nc``, via the public
:meth:`DataStore.payload_dir`) but **no database row is inserted**, and that is a
finding rather than an omission. ``DataStore.record_measurement`` takes
``eis_result: EISResult`` as its second positional parameter and reads six
EIS-specific attributes off it (``channel``, ``timestamp``, ``npts``,
``frequency``, ``measurement_time_s``, ``raw_file_path``, ``eis_params``) into
six EIS-specific columns. Nothing type-checks the argument, so a facade object
*would* get through — which is exactly why this is worth stating rather than
quietly doing:

1. Reaching the ``modality`` column that T2.3 added **for** non-EIS rows
   currently requires a non-EIS modality to impersonate an ``EISResult``.
2. Readers of the ``measurements`` table do not filter on ``modality``.
   ``DataStore.query_measurements`` selects ``m.*`` unconditionally (the GUI
   Analysis browser and ``web/data_adapter.py`` both consume it), so an image row
   would surface to every one of them as a spectrum with a NULL file path and
   NULL ``npts`` / ``f_min_hz`` / ``f_max_hz``. Writing it would make a
   *presentation* bug out of a *storage* gap.

The payload is therefore self-linking instead: ``attrs`` carry ``run_id``,
``step_name``, ``channel`` and the T2.6 ``sample_uuid``, so the file joins back to
the formulation and occupancy rows that describe the sample without a row of its
own. That is the T2.6 convention doing the work the missing row would have done.

Proposed follow-up (see TASKS.md T2.9): give ``record_measurement`` a
modality-agnostic entry point — the EIS-shaped columns become NULL-able optional
keywords and the readers filter on ``modality`` — after which this router adds an
INSERT and a :meth:`DataStore.set_measurement_payload` call, and nothing else here
changes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import xarray as xr

from softae.analysis.measurement_result import MeasurementResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Annotation-only, and deliberately not a runtime import: this is the same
    # latent cycle T2.2 designed out on the EIS side — `softae.workflows`
    # re-exports `WorkflowExecutor`, which imports routers.
    from softae.workflows.workflow_model import WorkflowStep

logger = structlog.get_logger(__name__)

#: Registry key. Matches ``MeasurementSpec.modality`` and the payload directory.
MODALITY = "image"

#: Instrument methods this router claims, taken from the camera driver's real
#: surface (:class:`softae.drivers.mock_camera.MockCamera` /
#: :class:`softae.drivers.async_camera.AsyncCamera`) rather than invented — a
#: router that matches a method no driver has is a modality that never fires.
IMAGE_METHODS: frozenset[str] = frozenset({"acquire_n_frames", "snap"})

#: On-disk payload format and xarray engine. Declared here rather than imported
#: from the EIS router: each modality chooses its own encoding (an image stream
#: could reasonably ship PNGs), and importing EIS's constants to get the same two
#: strings would couple the modalities for nothing.
PAYLOAD_FORMAT = "netcdf4"
PAYLOAD_ENGINE = "h5netcdf"

#: Colour-axis coordinate labels by channel count. 2 channels (gray+alpha) and
#: anything above 4 are refused rather than guessed — an unnamed colour axis is a
#: payload nobody can interpret later.
_COLOUR_LABELS: dict[int, tuple[str, ...]] = {
    3: ("r", "g", "b"),
    4: ("r", "g", "b", "a"),
}


def frame_to_dataset(frame: Any) -> xr.Dataset:
    """Wrap a captured frame as a labelled :class:`xarray.Dataset`.

    ``(y, x)`` for a grayscale frame, ``(y, x, rgb)`` for a colour one. The
    dimension names are the payload's contract: a bare ndarray on disk cannot say
    which axis is vertical, and a camera that changes orientation would silently
    transpose every stored image.

    A single-channel third axis is squeezed rather than kept as a length-1
    ``rgb`` dimension — ``(y, x, rgb=1)`` and ``(y, x)`` describe the same frame,
    and two spellings of one thing is how readers start branching.

    Raises :class:`ValueError` for a shape that cannot be labelled honestly;
    :meth:`ImageResultRouter.handle` catches it, because a payload we cannot
    describe must not be written under a name that claims we can.
    """
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 2:
        dims: tuple[str, ...] = ("y", "x")
        coords: dict[str, Any] = {}
    elif arr.ndim == 3 and arr.shape[2] in _COLOUR_LABELS:
        dims = ("y", "x", "rgb")
        coords = {"rgb": list(_COLOUR_LABELS[arr.shape[2]])}
    else:
        raise ValueError(
            f"cannot label an image of shape {arr.shape!r}: expected (y, x) or "
            f"(y, x, c) with c in {sorted(_COLOUR_LABELS)}"
        )

    coords["y"] = np.arange(arr.shape[0])
    coords["x"] = np.arange(arr.shape[1])
    return xr.Dataset({"frame": (dims, arr)}, coords=coords)


def _sample_uuid(step: "WorkflowStep") -> str | None:
    """The physical sample this step images, or ``None`` if it declares none.

    Same transcription-only rule as the EIS router's helper, and deliberately a
    separate two-line function rather than an import from it: a router reads the
    tag the wiring minted (T2.6) and never derives identity itself. A missing
    **or empty** tag normalises to ``None``, so an anonymous payload cannot join
    to every other unidentified sample.
    """
    tags = getattr(step, "tags", None) or {}
    value = tags.get("sample_uuid")
    return str(value) if value else None


def _write_payload(measurement: MeasurementResult, ctx: Any,
                   file_stem: str) -> Path | None:
    """Write the Dataset to the T2.3 payload location. Best-effort.

    ``runs/<run_id>/data/image/<stem>.nc`` — the same convention EIS uses, reached
    through the public :meth:`DataStore.payload_dir` so the layout stays one
    decision in one place.

    A store-less or run-less context is a **skip, not a failure**: an
    analysis-only modality is still meaningful in memory, and the caller gets its
    :class:`MeasurementResult` either way. Every write failure is swallowed here,
    for the reason the EIS router documents — the capture physically happened, and
    a full disk must not retract it.
    """
    store = getattr(ctx, "data_store", None)
    run_id = getattr(ctx, "run_id", None)
    if store is None or run_id is None:
        logger.debug("image_payload_skip", reason="no datastore or run_id")
        return None
    try:
        payload_dir = Path(store.payload_dir(run_id, measurement.modality))
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{file_stem}.nc"
        measurement.data.to_netcdf(payload_path, engine=PAYLOAD_ENGINE)
        measurement.meta["payload_path"] = str(payload_path)
        measurement.meta["payload_format"] = PAYLOAD_FORMAT
        logger.debug("image_payload_written", path=str(payload_path))
        return payload_path
    except Exception:
        logger.warning("image_payload_write_failed", stem=file_stem, exc_info=True)
        return None


class ImageResultRouter:
    """Routes captured frames into a netCDF payload + a ``MeasurementResult``.

    Claims a step two ways, because the two answer different questions.
    ``step.method in IMAGE_METHODS`` catches any step that drove the camera,
    including ones this modality did not build. ``tags["measurement"] == "image"``
    lets a step *declare* its modality independently of which driver call produced
    it, which is what a second camera method or a file-replay step would need.

    That tag value also does useful work at the other end of the system. T1.5's
    loop closure selects the campaign objective with
    ``tags.get("measurement", "primary") == "primary"``, so a step tagged
    ``image`` is automatically **not** a primary measurement — an analysis-only
    stream cannot accidentally be optimised against. The vocabulary already
    distinguished primary from secondary metrology; naming the modality here
    extends it rather than forking it.
    """

    #: Nothing. The camera's own kwargs (``frames``, ``exp``, ``gain``,
    #: ``poll_TO``) are addressed to the driver and must reach it, and this
    #: router's provenance comes from *tags*, which the executor never forwards.
    #: Declaring a param here would strip it from every other router's steps too,
    #: since the executor unions the sets.
    consumed_params: frozenset[str] = frozenset()

    def matches(self, step: "WorkflowStep") -> bool:
        if step.method in IMAGE_METHODS:
            return True
        tags = getattr(step, "tags", None) or {}
        return str(tags.get("measurement", "")) == MODALITY

    async def handle(self, step: "WorkflowStep", raw_result: Any,
                     ctx: Any) -> MeasurementResult | None:
        """Wrap the captured frame, write the payload, return the contract object.

        Returns ``None`` only when the frame could not be labelled — i.e. when
        there is nothing truthful to hand on. A failed *payload* write is not such
        a case: the capture succeeded, so the result is returned and only the file
        is missing.

        No ``measurements`` row is written; see the module docstring for the seam
        limit that decision records.
        """
        try:
            data = frame_to_dataset(raw_result)
        except Exception:
            logger.warning("image_route_failed", step=step.name, exc_info=True)
            return None

        provenance: dict[str, Any] = {
            "modality": MODALITY,
            "step_name": step.name,
            "captured_at": datetime.now().isoformat(),
        }
        run_id = getattr(ctx, "run_id", None)
        if run_id is not None:
            provenance["run_id"] = str(run_id)
        # Present-only, per "undeclared is unknown, never empty": an absent tag
        # omits the key entirely rather than writing a blank one.
        tags = getattr(step, "tags", None) or {}
        channel = tags.get("channel")
        if channel not in (None, ""):
            provenance["channel"] = str(channel)
        sample_uuid = _sample_uuid(step)
        if sample_uuid is not None:
            provenance["sample_uuid"] = sample_uuid

        # The capture settings the step asked for. Scalars only — `attrs` must
        # stay netCDF-encodable, and `meta` keeps the unfiltered original.
        capture = {f"capture_{k}": v for k, v in step.params.items()
                   if isinstance(v, (int, float, str)) and not isinstance(v, bool)}

        data.attrs.update(provenance)
        data.attrs.update(capture)

        frame = data["frame"].values
        summary = {
            "height": float(frame.shape[0]),
            "width": float(frame.shape[1]),
            "mean_intensity": float(frame.mean()) if frame.size else 0.0,
        }

        measurement = MeasurementResult(
            modality=MODALITY,
            data=data,
            meta={**provenance, "capture_params": dict(step.params)},
            summary=summary,
        )
        _write_payload(measurement, ctx, step.name)
        logger.info("image_routed", step=step.name, run_id=run_id,
                    shape=tuple(int(n) for n in frame.shape))
        return measurement
