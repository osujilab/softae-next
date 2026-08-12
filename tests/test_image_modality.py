"""T2.7 — the plug-play proof: an analysis-only ``image`` modality.

Tier 2 exists so a second data stream can be added without editing the engine.
This file is where that claim is checked rather than asserted in a docstring:

* the registry returns the image modality once it is registered, and says it does
  not have it before (``softae.analysis.image`` is imported at module scope here,
  so a registration-on-import would show up as a failure in
  ``test_modality_registry.py`` — which is exactly why registration is a call);
* **the money test** —
  :func:`test_one_workflow_routes_an_image_step_and_an_eis_step_to_their_own_routers`
  runs ONE workflow containing both an EIS step and a camera step through one
  executor holding both routers, and asserts each result went to its own router;
* the payload lands at the T2.3 location and reopens with its dimensions and
  provenance intact;
* no module under ``core/`` or ``workflows/`` imports the image modality.

The EIS golden test (``test_result_router_golden.py``) is run alongside this file
unmodified. Nothing in T2.7 touched the EIS path, and that test is what proves it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from softae.analysis.eis.router import EISResultRouter
from softae.analysis.image import (
    CAMERA_INSTRUMENT,
    IMAGE_MODALITY,
    ImageResultRouter,
    build_image_step,
    frame_to_dataset,
    register_image_modality,
)
from softae.analysis.image.router import MODALITY
from softae.core.data_store import DataStore
from softae.core.measurement_spec import MeasurementSpec
from softae.core.modality_registry import get_modality, list_modalities
from softae.drivers.mock_factory import create_mock_manager
from softae.server.manager import InstrumentManager
from softae.workflows.workflow_executor import ExecutorState, WorkflowExecutor
from softae.workflows.workflow_model import Workflow, WorkflowStep

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def registered_image():
    """Register the image modality for one test, then evict it.

    The registry is process-global and ``test_modality_registry.py`` asserts that
    ``get_modality("image")`` raises. Leaking a registration out of this file
    would break a test this task is not allowed to edit — so the modality is
    added per test and removed in teardown, which runs even when the test fails.
    """
    modality = register_image_modality()
    try:
        yield modality
    finally:
        from softae.core.modality_registry import _MODALITIES

        _MODALITIES.pop(MODALITY, None)


@pytest.fixture
def manager() -> InstrumentManager:
    return create_mock_manager(config={})


@pytest.fixture
async def connected_manager(manager: InstrumentManager):
    await manager.connect_all()
    return manager


@pytest.fixture
def data_store(tmp_path: Path):
    store = DataStore(tmp_path / "project")
    yield store
    store.close()


def _image_step(name: str = "image_ch1", **tags: str) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        instrument=CAMERA_INSTRUMENT,
        method="acquire_n_frames",
        params={"frames": 1, "exp": 0.02},
        tags={"channel": "1", "measurement": MODALITY, **tags},
    )


def _eis_step(name: str = "measure", chan: int = 1) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        instrument="pico1",
        method="sendscript_getdata",
        params={"mscrpath": "f.mscr", "outdir": "out", "chan": chan},
    )


# ── Registration ────────────────────────────────────────────────────────────


class TestRegistration:

    def test_importing_the_package_does_not_register_the_modality(self):
        """Registration is a call, not an import side effect.

        The registry rejects a filesystem scan because it makes the available
        set depend on what happens to be importable; registering at import time
        has the same shape one step removed. This module imports
        ``softae.analysis.image`` at the top, so if that import registered, the
        assertion below — and ``test_modality_registry.py`` — would fail.
        """
        assert "softae.analysis.image" in sys.modules
        assert MODALITY not in list_modalities()

    def test_registry_lookup_returns_the_image_modality(self, registered_image):
        assert MODALITY in list_modalities()
        assert get_modality(MODALITY) is IMAGE_MODALITY
        assert get_modality(MODALITY).name == MODALITY

    def test_registering_twice_is_a_no_op(self, registered_image):
        assert register_image_modality() is IMAGE_MODALITY

    def test_an_analysis_only_modality_offers_no_objectives(self, registered_image):
        """The emptiness IS the proof.

        A modality that feeds the optimizer nothing must still be a first-class
        registry entry, or "plug-play" only ever meant "another way to compute an
        objective". Nothing in the campaign path is required to consume it.
        """
        modality = get_modality(MODALITY)
        assert modality.objectives == {}
        with pytest.raises(KeyError, match="offers no objective"):
            modality.objective("sigma")

    def test_display_metadata_is_present_and_objective_free(self, registered_image):
        """A GUI rendering registered modalities generically shows this one with
        no objective picker — which is the truth about it."""
        display = get_modality(MODALITY).display
        assert display.display_name
        assert display.preset_names == ()
        assert display.objective_units == {}
        assert display.objective_labels == {}


# ── The step builder and the lifecycle hook ─────────────────────────────────


class TestTheMeasureStep:

    def test_build_measure_step_drives_the_camera_capture_method(self):
        step = build_image_step(7, MeasurementSpec(modality=MODALITY))
        assert step is not None
        assert step.instrument == CAMERA_INSTRUMENT
        assert step.method == "acquire_n_frames"
        assert step.tags["channel"] == "7"
        assert step.tags["measurement"] == MODALITY

    def test_the_capture_method_exists_on_the_camera_driver(self):
        """The step names a real driver method, not an invented one."""
        from softae.drivers.mock_camera import MockCamera

        assert callable(getattr(MockCamera, "acquire_n_frames"))

    def test_an_image_step_is_not_a_primary_measurement(self):
        """T1.5's objective predicate must never select an analysis-only step.

        Loop closure selects on ``tags.get("measurement", "primary") ==
        "primary"``; tagging the modality name keeps image steps out of it
        without a second vocabulary.
        """
        step = build_image_step(3, MeasurementSpec(modality=MODALITY))
        assert step.tags.get("measurement", "primary") != "primary"

    def test_overrides_reach_the_driver_under_its_own_kwarg_names(self):
        spec = MeasurementSpec(modality=MODALITY,
                               overrides={"frames": 3, "exposure_s": 0.1,
                                          "not_a_camera_kwarg": 5})
        step = build_image_step(1, spec)
        assert step.params == {"frames": 3, "exp": 0.1}

    def test_disabled_measurement_builds_no_step(self):
        spec = MeasurementSpec(modality=MODALITY, enabled=False)
        assert build_image_step(1, spec) is None

    def test_prepare_run_is_a_no_op(self, registered_image):
        modality = get_modality(MODALITY)
        assert modality.prepare_run(MeasurementSpec(modality=MODALITY), [1, 2],
                                    temp_dir=None, emit=None) is None

    def test_the_router_factory_builds_an_image_router(self, registered_image):
        router = get_modality(MODALITY).router_factory()
        assert isinstance(router, ImageResultRouter)


# ── Frame → Dataset ─────────────────────────────────────────────────────────


class TestFrameLabelling:

    def test_a_colour_frame_gets_y_x_rgb_dimensions(self):
        ds = frame_to_dataset(np.zeros((4, 6, 3), dtype=np.uint8))
        assert ds["frame"].dims == ("y", "x", "rgb")
        assert list(ds["rgb"].values) == ["r", "g", "b"]
        assert ds.sizes["y"] == 4 and ds.sizes["x"] == 6

    def test_a_grayscale_frame_gets_y_x_dimensions(self):
        ds = frame_to_dataset(np.zeros((4, 6), dtype=np.uint8))
        assert ds["frame"].dims == ("y", "x")

    def test_a_single_channel_axis_is_squeezed_not_kept(self):
        """``(y, x, rgb=1)`` and ``(y, x)`` describe the same frame; two
        spellings of one thing is how readers start branching."""
        assert frame_to_dataset(np.zeros((4, 6, 1)))["frame"].dims == ("y", "x")

    def test_an_unlabelable_shape_refuses_rather_than_guesses(self):
        with pytest.raises(ValueError, match="cannot label"):
            frame_to_dataset(np.zeros((4, 6, 7)))


# ── The router in isolation ─────────────────────────────────────────────────


class TestTheRouter:

    def test_it_claims_camera_steps_and_declines_eis_steps(self):
        router = ImageResultRouter()
        assert router.matches(_image_step())
        assert not router.matches(_eis_step())

    def test_it_claims_a_step_that_declares_the_modality_by_tag(self):
        """Method-matching catches steps this modality built; the tag lets a
        step declare its stream independently of which driver call made it."""
        replay = WorkflowStep(name="replay", instrument="camera",
                              method="load_frame_from_disk",
                              tags={"measurement": MODALITY})
        assert ImageResultRouter().matches(replay)

    def test_the_eis_router_declines_an_image_step(self):
        assert not EISResultRouter().matches(_image_step())

    @pytest.mark.asyncio
    async def test_an_untagged_step_omits_sample_uuid_rather_than_writing_blank(self):
        from softae.analysis.eis.router import RouterContext

        step = WorkflowStep(name="anon", instrument="camera",
                            method="acquire_n_frames")
        result = await ImageResultRouter().handle(
            step, np.zeros((4, 6, 3), dtype=np.uint8), RouterContext())
        assert result is not None
        assert "sample_uuid" not in result.data.attrs
        assert "channel" not in result.data.attrs

    @pytest.mark.asyncio
    async def test_a_tagged_step_carries_its_sample_identity_into_the_payload(self):
        """T2.6's spine reaches a modality that was not written when it landed:
        the wiring stamps ``sample_uuid`` onto any step tagged with a channel."""
        from softae.analysis.eis.router import RouterContext

        step = _image_step(sample_uuid="abc-123")
        result = await ImageResultRouter().handle(
            step, np.zeros((4, 6, 3), dtype=np.uint8), RouterContext())
        assert result.data.attrs["sample_uuid"] == "abc-123"
        assert result.data.attrs["channel"] == "1"

    @pytest.mark.asyncio
    async def test_a_frame_that_cannot_be_labelled_is_declined(self):
        from softae.analysis.eis.router import RouterContext

        result = await ImageResultRouter().handle(
            _image_step(), np.zeros((4, 6, 7)), RouterContext())
        assert result is None


# ── The money test: one workflow, two modalities, two routers ───────────────


@pytest.mark.asyncio
async def test_one_workflow_routes_an_image_step_and_an_eis_step_to_their_own_routers(
    connected_manager, data_store
):
    """The plug-play proof, end to end through the public executor.

    One :class:`WorkflowExecutor`, two routers, one workflow carrying a camera
    step and an EIS step. The executor is unmodified — it knows neither modality
    and only iterates ``routers`` — so this passing is the statement that a second
    data stream needs no engine change.
    """
    run_id = data_store.start_run("mixed_modality")
    workflow = Workflow(name="mixed", setup=[_eis_step(), _image_step()])

    executor = WorkflowExecutor(
        connected_manager, data_store=data_store, run_id=run_id,
        routers=[EISResultRouter(), ImageResultRouter()],
    )
    await executor.run(workflow)
    assert executor.state == ExecutorState.COMPLETED

    by_modality = {m.modality: m for m in executor.measurement_results}
    assert sorted(by_modality) == ["eis", "image"]

    # Each result carries the step its own router handled — not the other's.
    assert by_modality["eis"].meta["step_name"] == "measure"
    assert by_modality["image"].meta["step_name"] == "image_ch1"

    # The image payload is a labelled frame; the EIS payload is a spectrum.
    assert by_modality["image"].data["frame"].dims == ("y", "x", "rgb")
    assert "frame" not in by_modality["eis"].data.data_vars

    # The EIS path is untouched: its row, its .txt and its .nc are all still there.
    eis_rows = data_store.query_measurements(run_id=run_id)
    assert [r["modality"] for r in eis_rows] == ["eis"]
    assert (data_store.eis_dir(run_id) / "measure_ch1.txt").exists()
    assert (data_store.payload_dir(run_id, "eis") / "measure_ch1.nc").exists()


@pytest.mark.asyncio
async def test_the_image_payload_is_written_beside_eis_and_reopens(
    connected_manager, data_store
):
    """T2.3's payload conventions, reached through the public API only.

    ``runs/<run_id>/data/image/<stem>.nc`` — a sibling of ``data/eis/``, so a new
    modality adds a directory instead of colliding with EIS filenames.
    """
    run_id = data_store.start_run("image_payload")
    step = _image_step(sample_uuid="sample-42")
    workflow = Workflow(name="image_only", setup=[step])

    executor = WorkflowExecutor(
        connected_manager, data_store=data_store, run_id=run_id,
        routers=[ImageResultRouter()],
    )
    await executor.run(workflow)
    assert executor.state == ExecutorState.COMPLETED

    payload = data_store.payload_dir(run_id, "image") / "image_ch1.nc"
    assert payload.exists()
    assert payload.parent.parent == data_store.payload_dir(run_id, "eis").parent

    with xr.open_dataset(payload, engine="h5netcdf") as reopened:
        assert reopened["frame"].dims == ("y", "x", "rgb")
        assert reopened.sizes["y"] == 480 and reopened.sizes["x"] == 640
        assert reopened.attrs["modality"] == "image"
        assert reopened.attrs["run_id"] == run_id
        assert reopened.attrs["sample_uuid"] == "sample-42"
        assert reopened.attrs["step_name"] == "image_ch1"
        assert reopened.attrs["capture_frames"] == 1


@pytest.mark.asyncio
async def test_no_measurements_row_is_written_for_an_image_capture(
    connected_manager, data_store
):
    """The discovered seam limit, pinned so it cannot be lost (T2.7 finding).

    ``DataStore.record_measurement`` takes ``eis_result: EISResult`` and writes
    six EIS-specific columns off it; ``query_measurements`` selects ``m.*`` with
    no ``modality`` filter, so an image row would surface to the Analysis browser
    and ``web/data_adapter`` as a spectrum with NULL ``npts`` and NULL file path.
    The payload is written and the row is skipped — the file self-links through
    its ``attrs`` instead. See TASKS.md T2.9 for the proposed follow-up; when it
    lands this test is the one that must change.
    """
    run_id = data_store.start_run("image_seam")
    workflow = Workflow(name="image_only", setup=[_image_step()])

    executor = WorkflowExecutor(
        connected_manager, data_store=data_store, run_id=run_id,
        routers=[ImageResultRouter()],
    )
    await executor.run(workflow)

    assert data_store.query_measurements(run_id=run_id) == []
    assert (data_store.payload_dir(run_id, "image") / "image_ch1.nc").exists()
    assert len(executor.measurement_results) == 1


# ── Import inspection: the engine must not know this modality exists ─────────


def test_no_core_or_workflow_module_imports_the_image_modality():
    """The proof T2.7 is actually about.

    Adding a data stream may not reach into the engine. Source is inspected
    rather than ``sys.modules``, because an import that only happens on some code
    path would still be an edit to ``core/`` or ``workflows/``.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "softae"
    offenders = []
    for package in ("core", "workflows"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "analysis.image" in text or "analysis import image" in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_the_image_modality_does_not_import_the_eis_one():
    """Two modalities that import each other are one module in two files."""
    root = Path(__file__).resolve().parents[1] / "src" / "softae" / "analysis" / "image"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "analysis.eis" not in stripped, f"{path.name}: {stripped}"
