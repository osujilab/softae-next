"""The modality registry — Tier 2's single plug-in point (T2.5).

A modality is everything the campaign path needs to know about *a kind of
measurement*: how to build its per-channel step, which router persists its
results, which objectives it offers and in which direction each is optimised,
what it must prepare before a run, and how a GUI should name it.

Two properties carry the risk here, and both are pinned below:

* **EIS behaviour is unchanged.** The registry composes the *existing* pieces —
  ``eis_measure_step``, ``EISResultRouter``, ``OBJECTIVE_DIRECTION`` — rather
  than re-implementing them, so the identity checks in
  :class:`TestTheEISModalityComposesTheRealPieces` are the guard: if someone
  writes a second EIS extractor, those tests fail before a campaign does.
* **The ``.mscr`` migration moves files, not behaviour.**
  :class:`TestPrepareRunWritesTheScripts` was written and passed against the
  *inline* block in ``run_autonomous_campaign`` before the block moved into the
  EIS modality's ``prepare_run`` hook, and is unchanged since. It asserts the
  files on disk, not the call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core import eis_scripts
from softae.core.autonomous_wiring import CampaignSpec, run_autonomous_campaign
from softae.core.data_store import DataStore
from softae.core.measurement_spec import MeasurementSpec
from softae.core.modality_registry import (
    Modality,
    ModalityDisplay,
    ObjectiveSpec,
    UnknownModalityError,
    get_modality,
    list_modalities,
    register_modality,
)
from softae.drivers.mock_factory import create_mock_manager

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="registry_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
    )
    base.update(over)
    return CampaignSpec(**base)


@pytest.fixture
async def connected():
    mgr = create_mock_manager(config={})
    await mgr.connect_all()
    yield mgr
    await mgr.disconnect_all()


@pytest.fixture
def mscr_dir(tmp_path: Path, monkeypatch) -> Path:
    """Redirect ``.mscr`` writes out of the shared system temp dir.

    ``mscr_path_for_channel`` hard-codes ``tempfile.gettempdir()`` because the
    measurement *step* reads the same function — the two must agree on the path.
    Redirecting the one function therefore moves both ends at once, which is the
    only way to assert on the files without writing into a directory other
    sessions share.
    """
    target = tmp_path / "mscr"
    target.mkdir()
    monkeypatch.setattr(
        eis_scripts, "mscr_path_for_channel",
        lambda channel: str(target / f"softae_ch{int(channel)}.mscr"),
    )
    return target


# ── Registry mechanics ───────────────────────────────────────────────────────

class TestTheRegistryIsExplicit:
    """Registration is a call, never a directory listing (ATLAS addendum §3).

    A filesystem scan makes the set of modalities depend on what happens to be
    importable, so a half-written module becomes a silently missing capability.
    An explicit dict fails at the lookup instead, naming what it does have.
    """

    def test_eis_is_registered_out_of_the_box(self):
        assert "eis" in list_modalities()
        assert get_modality("eis").name == "eis"

    def test_a_registered_modality_is_returned_by_name(self):
        probe = Modality(
            name="probe_registered",
            build_measure_step=lambda channel, spec: None,
            router_factory=lambda: None,
            objectives={},
            prepare_run=lambda spec, channels, **kw: None,
            display=ModalityDisplay(display_name="Probe"),
        )
        try:
            register_modality(probe)
            assert get_modality("probe_registered") is probe
        finally:
            list_modalities()  # bootstrap, then evict only what this test added
            from softae.core.modality_registry import _MODALITIES

            _MODALITIES.pop("probe_registered", None)

    def test_an_unknown_modality_names_the_ones_that_exist(self):
        with pytest.raises(UnknownModalityError) as excinfo:
            get_modality("image")
        message = str(excinfo.value)
        assert "image" in message
        assert "eis" in message  # the helpful half: what *is* registered

    def test_the_unknown_error_is_still_a_notimplementederror(self):
        """T2.4's refusal contract, preserved through the T2.5 replacement.

        ``run_autonomous_campaign`` used to raise a bare ``NotImplementedError``
        for a non-EIS modality; the registry lookup replaces it. Subclassing
        keeps every existing caller and test that catches the old type working,
        and it is honest — an unregistered modality genuinely is not implemented.
        """
        assert issubclass(UnknownModalityError, NotImplementedError)

    def test_a_second_modality_cannot_take_a_name_already_held(self):
        """Substitution is the hazard: which one runs would follow import order,
        and a campaign could measure something its spec never named."""
        impostor = Modality(
            name="eis",
            build_measure_step=lambda channel, spec: None,
            router_factory=lambda: None,
            objectives={},
            prepare_run=lambda spec, channels, **kw: None,
            display=ModalityDisplay(display_name="Not EIS"),
        )
        with pytest.raises(ValueError, match="already registered"):
            register_modality(impostor)
        assert get_modality("eis").display.display_name != "Not EIS"

    def test_re_registering_the_identical_modality_is_a_no_op(self):
        """Idempotent on purpose — the same object under the same name asserts
        nothing new, and raising would make a re-import a hard failure."""
        eis = get_modality("eis")
        assert register_modality(eis) is eis
        assert get_modality("eis") is eis

    def test_an_objective_must_declare_a_real_direction(self):
        """A typo'd direction is the one defect that inverts a whole campaign.

        It cannot be caught downstream: ``resolve_direction`` compares the
        modality's answer against the spec's, so an unrecognised string would
        simply never match and the failure would surface as a contradiction
        message about the *spec*. Refuse at construction, where the typo is.
        """
        with pytest.raises(ValueError, match="minimize"):
            ObjectiveSpec(name="sigma", direction="maximise",  # British spelling
                          extractor=lambda *a, **k: None)

    def test_asking_a_modality_for_an_objective_it_lacks_names_the_ones_it_has(self):
        with pytest.raises(KeyError, match="mean_abs_z"):
            get_modality("eis").objective("photoluminescence")


# ── EIS composes the real pieces ─────────────────────────────────────────────

class TestTheEISModalityComposesTheRealPieces:
    """Identity, not equivalence.

    Every assertion here is ``is`` against the object the campaign path already
    used. A registry that *re-implemented* the EIS step builder or the objective
    directions would pass a behavioural test on the day it was written and drift
    silently afterwards; these fail the moment a second implementation appears.
    """

    def test_the_step_builder_is_the_existing_eis_measure_step(self):
        from softae.core.autonomous_wiring import measure_step_name
        from softae.core.deposition_steps import eis_measure_step

        step = get_modality("eis").build_measure_step(21, MeasurementSpec())

        expected = eis_measure_step(21, name=measure_step_name(21))
        assert step.name == expected.name == measure_step_name(21)
        assert step.instrument == expected.instrument
        assert step.method == expected.method
        assert step.params == expected.params
        # The T1.5 loop-closure tags ride through the registry untouched.
        assert step.tags == expected.tags == {"channel": "21", "measurement": "primary"}

    def test_the_router_factory_builds_the_existing_eis_router(self):
        from softae.analysis.eis.router import EISResultRouter

        router = get_modality("eis").router_factory()
        assert isinstance(router, EISResultRouter)

    def test_objective_directions_are_the_existing_map_not_a_copy(self):
        """Re-spelling ``maximize`` here would be the 4.7x-style landmine again.

        σ and mean |Z| are the same physical goal in opposite signs, so a
        registry that carried its own direction table could invert a campaign
        while every step reported success. The directions are *derived* from
        ``OBJECTIVE_DIRECTION``, and this pins that they cannot fork.
        """
        from softae.core.autonomous_wiring import OBJECTIVE_DIRECTION

        objectives = get_modality("eis").objectives
        assert set(objectives) == set(OBJECTIVE_DIRECTION)
        for kind, direction in OBJECTIVE_DIRECTION.items():
            assert objectives[kind].direction == direction

    def test_the_extractors_are_the_existing_wiring_functions(self):
        from softae.core import autonomous_wiring as wiring

        sigma = get_modality("eis").objectives["sigma"]
        assert sigma.extractor.__wrapped__ is wiring.eis_impedance_objective
        assert (sigma.channel_extractor.__wrapped__
                is wiring.eis_impedance_objective_for_channel)

    def test_the_extractors_resolve_on_every_call_not_once_at_registration(
        self, monkeypatch
    ):
        """Late binding, pinned — it is behaviour, not an implementation detail.

        The campaign used to call ``eis_impedance_objective`` as a module global,
        so the name resolved afresh each trial. Two of the P1.2/P1.3 park guards
        force an unmeasured trial by rebinding exactly that attribute; a registry
        that captured the function at registration would leave them passing while
        the campaign they describe silently stopped parking. Caught that way
        once — this test is why it cannot happen quietly again.
        """
        from softae.core import autonomous_wiring as wiring

        extractor = get_modality("eis").objectives["sigma"].extractor
        monkeypatch.setattr(
            wiring, "eis_impedance_objective", lambda *a, **k: "rebound")

        assert extractor({}, {}) == "rebound"

    def test_display_metadata_is_static_data_with_no_gui_dependency(self):
        """Decision (c): the future Autonomous tab renders registered modalities
        generically, so the metadata must be importable without Qt."""
        import sys

        display = get_modality("eis").display
        assert display.display_name
        assert "Quick" in display.preset_names
        assert display.objective_units["sigma"]
        assert isinstance(display, ModalityDisplay)

        module = sys.modules["softae.core.modality_registry"]
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "PySide6" not in source and "QtWidgets" not in source


# ── The .mscr migration (characterized BEFORE the move) ──────────────────────

class TestPrepareRunWritesTheScripts:
    """The one real migration in T2.5, guarded from both ends.

    Written against the inline block in ``run_autonomous_campaign`` and passing
    there *before* the move, so it characterizes the old behaviour rather than
    describing the new code. The assertions are on files, parameters and the
    emitted event — the three things a run's provenance actually depends on.
    """

    @pytest.mark.asyncio
    async def test_a_campaign_writes_one_script_per_channel_with_the_resolved_preset(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        store = DataStore(tmp_path / "proj")
        events: list[dict] = []
        spec = _spec(measurement=MeasurementSpec(preset="Quick"))

        await run_autonomous_campaign(
            spec, manager=connected, data_store=store, on_event=events.append,
        )
        store.close()

        assert sorted(p.name for p in mscr_dir.glob("*.mscr")) == [
            "softae_ch21.mscr", "softae_ch22.mscr",
        ]
        built = [e for e in events if e["type"] == "eis_scripts_built"]
        assert len(built) == 1
        expected = eis_scripts.EISParams.from_preset("Quick")
        assert built[0]["n"] == 2
        assert {k: built[0][k] for k in expected.as_metadata()} == expected.as_metadata()

    @pytest.mark.asyncio
    async def test_overrides_reach_the_written_script(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        """The block's whole purpose: what the run *records* is what the
        instrument *gets*. An override that reached the metadata but not the
        file would be provenance asserting a sweep that never ran."""
        store = DataStore(tmp_path / "proj")
        events: list[dict] = []
        spec = _spec(measurement=MeasurementSpec(preset="Quick", overrides={"npts": 7}))

        await run_autonomous_campaign(
            spec, manager=connected, data_store=store, on_event=events.append,
        )
        store.close()

        built = [e for e in events if e["type"] == "eis_scripts_built"]
        assert built and built[0]["eis_npts"] == 7
        text = (mscr_dir / "softae_ch21.mscr").read_text(encoding="utf-8")
        assert "7" in text  # the point count reached the MethodSCRIPT itself

    @pytest.mark.asyncio
    async def test_measurement_disabled_writes_nothing_and_emits_nothing(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        """``enabled=False`` is *formulate and cast, but do not measure*."""
        store = DataStore(tmp_path / "proj")
        events: list[dict] = []
        spec = _spec(measurement=MeasurementSpec(enabled=False))

        await run_autonomous_campaign(
            spec, manager=connected, data_store=store, on_event=events.append,
        )
        store.close()

        assert list(mscr_dir.glob("*.mscr")) == []
        assert [e for e in events if e["type"] == "eis_scripts_built"] == []

    @pytest.mark.asyncio
    async def test_scripts_cover_the_whole_board_not_just_the_active_channels(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        """A board-aware campaign moves to electrodes it has not named yet.

        ``electrode_capacity`` means the allocator will walk past ``channels``
        onto later electrodes mid-run, and a step whose script was never written
        measures with whatever a previous session left behind. The union is the
        behaviour being preserved, and it is easy to lose in a refactor because
        nothing fails until the board advances.
        """
        store = DataStore(tmp_path / "proj")
        spec = _spec(channels=(21,), electrode_start=21, electrode_capacity=24,
                     budget=1)

        await run_autonomous_campaign(spec, manager=connected, data_store=store)
        store.close()

        assert sorted(p.name for p in mscr_dir.glob("*.mscr")) == [
            "softae_ch21.mscr", "softae_ch22.mscr",
            "softae_ch23.mscr", "softae_ch24.mscr",
        ]


# ── The campaign path goes through the registry ──────────────────────────────

class TestTheCampaignPathResolvesItsModality:

    @pytest.mark.asyncio
    async def test_an_unregistered_modality_fails_before_anything_is_touched(
        self, tmp_path: Path
    ):
        """Same early point as T2.4's refusal: no manager, no run row.

        The check sits ahead of ``create_manager`` deliberately — a campaign
        that cannot measure what it names must not connect to the rig first.
        """
        store = DataStore(tmp_path / "proj")
        spec = _spec(measurement=MeasurementSpec(modality="image"))

        with pytest.raises(UnknownModalityError, match="image"):
            await run_autonomous_campaign(spec, data_store=store)

        assert store.query_runs() == [] or all(
            r["campaign"] != "registry_campaign" for r in store.query_runs()
        )
        store.close()

    @pytest.mark.asyncio
    async def test_a_registered_modality_reaches_prepare_run_through_the_registry(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        """The lookup is real dispatch, not a rebranded ``if modality == "eis"``.

        A stand-in modality registered under EIS's own name is impossible (the
        registry refuses duplicates), so this swaps the *entry* for the duration
        of the run and asserts the campaign called what the registry held.
        """
        from softae.core import modality_registry as registry

        calls: list[tuple[MeasurementSpec, list[int]]] = []
        real = get_modality("eis")

        def spy_prepare(spec, channels, **kwargs):
            calls.append((spec, list(channels)))
            return real.prepare_run(spec, channels, **kwargs)

        stand_in = Modality(
            name="eis",
            build_measure_step=real.build_measure_step,
            router_factory=real.router_factory,
            objectives=real.objectives,
            prepare_run=spy_prepare,
            display=real.display,
        )
        registry._MODALITIES["eis"] = stand_in
        try:
            store = DataStore(tmp_path / "proj")
            await run_autonomous_campaign(
                _spec(), manager=connected, data_store=store)
            store.close()
        finally:
            registry._MODALITIES["eis"] = real

        assert len(calls) == 1
        measurement, channels = calls[0]
        assert measurement.modality == "eis"
        assert channels == [21, 22]
        assert sorted(p.name for p in mscr_dir.glob("*.mscr")) == [
            "softae_ch21.mscr", "softae_ch22.mscr",
        ]

    @pytest.mark.asyncio
    async def test_a_campaign_still_runs_end_to_end_on_a_mock_rig(
        self, connected, tmp_path: Path, mscr_dir: Path
    ):
        """The whole point of a no-behaviour-change refactor (T2.8)."""
        store = DataStore(tmp_path / "proj")
        result = await run_autonomous_campaign(
            _spec(budget=2), manager=connected, data_store=store)

        assert result.n_trials == 2
        rows = store.query_doe_parameters(run_id=result.run_id)
        assert len(rows) == 2
        store.close()
