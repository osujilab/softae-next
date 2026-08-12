"""The analysis-only ``image`` modality — Tier 2's plug-play proof (T2.7).

A modality that **measures nothing the optimizer can use**. ``objectives`` is
empty by design: an image stream records what a film looked like, and a campaign
still steers on EIS. That emptiness is the proof, not a gap — it shows the
registry's contract is genuinely about *data streams* rather than about
objectives, so adding a camera, a profilometer or a thermal log costs nothing in
the BO path.

What this package demonstrates, concretely: every file it took to add a second
measurement stream is **new and under ``analysis/``**. No edit to
``workflows/workflow_executor.py``, none to ``core/modality_registry.py``, none
to ``analysis/eis/``, none to any driver.
``tests/test_image_modality.py::test_no_core_or_workflow_module_imports_the_image_modality``
pins that, and the EIS golden test passes unmodified beside it.

Registration is an explicit call, never an import side effect
------------------------------------------------------------
:func:`register_image_modality` must be *called*; importing this package only
builds :data:`IMAGE_MODALITY` and mutates nothing. The registry's own docstring
rejects a filesystem scan because it "makes the set of available modalities
depend on what happens to be importable at the moment somebody looks" — and
registering at import time has exactly that shape one step removed, since the set
would then depend on which modules an unrelated caller happened to pull in.
``get_modality("image")`` raising until somebody asks for the image modality is
the honest answer, and it is the answer
``test_modality_registry.py::test_an_unknown_modality_names_the_ones_that_exist``
already depends on.

The consequence is one line of wiring left for whoever first ships an image
campaign: call :func:`register_image_modality` at application start, or add it to
``modality_registry._ensure_builtins`` once ``image`` stops being a proof and
becomes a shipped capability. Both are ``core/`` edits, which is why neither is
done here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from softae.analysis.image.router import (
    IMAGE_METHODS,
    MODALITY,
    PAYLOAD_FORMAT,
    ImageResultRouter,
    frame_to_dataset,
)
from softae.core.modality_registry import (
    Modality,
    ModalityDisplay,
    register_modality,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from softae.core.measurement_spec import MeasurementSpec
    from softae.workflows.workflow_model import WorkflowStep

__all__ = [
    "CAMERA_INSTRUMENT",
    "IMAGE_METHODS",
    "IMAGE_MODALITY",
    "MODALITY",
    "PAYLOAD_FORMAT",
    "ImageResultRouter",
    "build_image_step",
    "frame_to_dataset",
    "register_image_modality",
]

#: Manager key for the camera. Matches ``drivers.mock_factory.create_mock_manager``
#: and the ``[instruments.camera]`` config section, so the same step runs against
#: the mock and the real :class:`~softae.drivers.async_camera.AsyncCamera`.
CAMERA_INSTRUMENT = "camera"

#: Driver method the measure step calls. ``acquire_n_frames`` and not ``snap``:
#: ``snap`` is the convenience wrapper (it can display and save), and a workflow
#: step must not have a display side effect on a headless rig.
CAPTURE_METHOD = "acquire_n_frames"

#: Capture settings a spec may override, mapped to the driver's own kwarg names.
#: An allow-list rather than a splat of ``spec.overrides``: an unknown key would
#: reach ``acquire_n_frames`` as an unexpected kwarg and fail the step at run
#: time, on the rig, after the film was already cast.
_CAPTURE_OVERRIDES: dict[str, str] = {
    "frames": "frames",
    "exp": "exp",
    "exposure_s": "exp",
    "gain": "gain",
}


def build_image_step(channel: int, spec: "MeasurementSpec") -> "WorkflowStep | None":
    """The per-electrode capture step, or ``None`` when measurement is disabled.

    ``enabled=False`` means *formulate and cast, but do not measure* — the same
    contract EIS honours in ``prepare_run`` — so it returns no step rather than a
    step that captures and discards.

    Tags, not params, carry provenance: ``channel`` is what T1.5's loop closure
    and T2.6's ``_stamp_sample_uuids`` key on (the wiring stamps ``sample_uuid``
    onto any step already tagged with a channel, so this modality inherits the
    identity spine for free), and ``measurement="image"`` both routes the result
    and keeps the step out of the objective — see :class:`ImageResultRouter`.
    """
    if not spec.enabled:
        return None

    from softae.workflows.workflow_model import WorkflowStep

    params: dict[str, Any] = {"frames": 1}
    for key, value in spec.overrides.items():
        driver_kwarg = _CAPTURE_OVERRIDES.get(key)
        if driver_kwarg is not None and value is not None:
            params[driver_kwarg] = value

    return WorkflowStep(
        name=f"image_ch{int(channel)}",
        instrument=CAMERA_INSTRUMENT,
        method=CAPTURE_METHOD,
        params=params,
        tags={"channel": str(int(channel)), "measurement": MODALITY},
    )


def _prepare_run(spec: "MeasurementSpec", channels: Sequence[int], *,
                 temp_dir: str | None = None, emit: Any = None) -> None:
    """Nothing to prepare — and the no-op is the interesting part.

    EIS needs this hook because its steps carry only a ``.mscr`` path, so the
    scripts must exist before any step reads one. A camera step carries its own
    settings in ``params``, so there is no out-of-band state to stage and nothing
    to go stale between runs. A modality is free to do nothing here; the hook
    exists so the campaign path never has to know which kind it is holding.
    """


#: The modality itself. Built at import, registered only on request.
IMAGE_MODALITY = Modality(
    name=MODALITY,
    build_measure_step=build_image_step,
    router_factory=ImageResultRouter,
    objectives={},
    prepare_run=_prepare_run,
    display=ModalityDisplay(
        display_name="Camera image (analysis-only)",
        preset_names=(),
        # Empty on both counts, and consistent with `objectives`: a GUI rendering
        # registered modalities generically will show this one with no objective
        # picker, which is the truth about it.
        objective_units={},
        objective_labels={},
    ),
)


def register_image_modality() -> Modality:
    """Add :data:`IMAGE_MODALITY` to the registry and return it.

    Idempotent, because ``register_modality`` treats re-registering the *same*
    object as a no-op — so calling this from application start and again from a
    test costs nothing.
    """
    return register_modality(IMAGE_MODALITY)
