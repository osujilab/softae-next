"""The production read — one authoritative sweep, taken after settling.

What these pin is the thing the campaign has never had: the reading it records
is a real ``measurement="primary"`` sweep taken *after* ``drive_settle_phase``
returns, not the last ``measurement="settle"`` round recycled under the trial's
step name (``autonomous_wiring.py:3611-3614``).

The three properties, one test family each:

* it runs **unconditionally**, and with no override it uses ``spec.measurement``
  and the run's own (unsuffixed) ``.mscr``;
* an **override** prepares its own scripts *first*, as a variant, and the step it
  builds points at that variant's file — a denser preset nobody wrote scripts for
  would otherwise read the campaign's own sweep and record the denser preset as
  provenance;
* the step keeps the default ``measurement="primary"`` / ``role="sample"`` tags,
  so :func:`~softae.core.autonomous_wiring.is_primary_measurement` selects it
  with no edit to that predicate.
"""

from __future__ import annotations

import asyncio

import pytest

from softae.core.autonomous_wiring import CampaignSpec, is_primary_measurement
from softae.core.measurement_spec import MeasurementSpec
from softae.core.production_read import (
    build_production_read_workflow,
    production_step_name,
    take_production_read,
)

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="production_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,
        budget=2,
        seed=7,
        measurement=MeasurementSpec(preset="Quick"),
    )
    base.update(over)
    return CampaignSpec(**base)


class FakeExecutor:
    """Runs a workflow's setup steps and hands each one a recognisable result.

    Mirrors the real executor's one relevant contract — ``on_step_complete(step,
    index, total, result, elapsed)`` — and records the order in a shared log so a
    test can pin *when* the read happened relative to the settle phase.
    """

    def __init__(self, log: list[str] | None = None) -> None:
        self.on_step_complete = None
        self.workflows: list = []
        self.log = log if log is not None else []

    async def run(self, workflow) -> None:
        self.workflows.append(workflow)
        self.log.append(f"run:{workflow.name}")
        steps = list(workflow.setup)
        for index, step in enumerate(steps):
            if self.on_step_complete:
                self.on_step_complete(step, index, len(steps),
                                      f"raw::{step.name}", 0.0)


@pytest.fixture(autouse=True)
def clean_variants():
    """A fresh sweep-variant registry either side of every test.

    ``eis_scripts`` keeps the run's base sweep in module state, so a test that
    declared one would otherwise decide the next test's variant keys.
    """
    from softae.core.eis_scripts import reset_run_variants

    reset_run_variants()
    yield
    reset_run_variants()


@pytest.fixture()
def no_script_writes(monkeypatch):
    """Record what ``build_eis_scripts`` was asked to write; write nothing.

    The variant *bookkeeping* is what these tests are about; actually emitting
    MethodSCRIPT files into the temp directory is ``test_eis_scripts.py``'s job.
    """
    from softae.core import eis_scripts

    calls: list[tuple[list[int], object, str | None]] = []

    def _record(channels, params, *, variant=None):
        calls.append(([int(c) for c in channels], params, variant))
        return []

    monkeypatch.setattr(eis_scripts, "build_eis_scripts", _record)
    return calls


# ── No override: the campaign's own block, the run's own scripts ─────────────

def test_take_production_read_without_override_uses_the_campaign_measurement(
    clean_variants, no_script_writes
):
    """`spec.measurement` is the sweep, and nothing is re-prepared for it.

    The campaign already wrote its base scripts at run start; re-preparing here
    would be work, and a *wrongly* re-prepared base would move the file every
    settle round reads.
    """
    from softae.core.eis_scripts import EISParams, begin_run, mscr_path_for_channel

    begin_run(EISParams.from_preset("Quick"))
    executor = FakeExecutor()
    raws = asyncio.run(take_production_read(_spec(), [21, 22], executor=executor))

    assert set(raws) == {21, 22}
    assert no_script_writes == [], "no override → no script preparation"
    step = executor.workflows[0].setup[0]
    assert step.params["mscrpath"] == mscr_path_for_channel(21)


def test_take_production_read_returns_the_raws_of_the_steps_it_ran():
    executor = FakeExecutor()
    raws = asyncio.run(take_production_read(_spec(), [21, 22], executor=executor))
    names = [s.name for s in executor.workflows[0].setup]
    assert raws == {21: f"raw::{names[0]}", 22: f"raw::{names[1]}"}


def test_take_production_read_step_is_tagged_primary_so_the_predicate_selects_it():
    executor = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21], executor=executor))
    step = executor.workflows[0].setup[0]
    assert step.tags["measurement"] == "primary"
    assert step.tags.get("role", "sample") == "sample"
    assert is_primary_measurement(step.tags)


def test_take_production_read_step_name_is_distinct_from_the_trials_own():
    """A second read of the same channel in one run gets its own step name.

    ``settle_eis_ch{N}_r{i}`` and ``confirm_eis_ch{N}_r{a}`` are the two existing
    instances of the rule; eligibility is decided by tags, never by the name.
    """
    from softae.core.autonomous_wiring import measure_step_name

    executor = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21], executor=executor))
    name = executor.workflows[0].setup[0].name
    assert name != measure_step_name(21)
    assert name == production_step_name(measure_step_name(21))


# ── Override: its own variant scripts, written before the step is built ──────

def test_take_production_read_with_override_prepares_the_variant_scripts(
    clean_variants, no_script_writes
):
    from softae.core.eis_scripts import EISParams, begin_run

    begin_run(EISParams.from_preset("Quick"))
    dense = MeasurementSpec(preset="Quick", overrides={"npts": 91})
    executor = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21, 22], measurement=dense,
                                     executor=executor))

    assert len(no_script_writes) == 1, "the override's scripts were never written"
    channels, params, variant = no_script_writes[0]
    assert channels == [21, 22]
    assert params.npts == 91
    assert variant is not None, (
        "prepared as the run's BASE, not as a variant — that clears the "
        "campaign's own variants and writes the production sweep to the path "
        "every settle round reads")


def test_take_production_read_with_override_step_points_at_the_variant_script(
    clean_variants, no_script_writes
):
    """The writer's path and the reader's path are the same path, or nothing works."""
    from softae.core.eis_scripts import (
        EISParams,
        begin_run,
        mscr_path_for_channel,
        variant_for,
    )

    begin_run(EISParams.from_preset("Quick"))
    dense = MeasurementSpec(preset="Quick", overrides={"npts": 91})
    executor = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21], measurement=dense,
                                     executor=executor))

    variant = variant_for(EISParams.from_preset("Quick", npts=91))
    assert variant is not None
    step = executor.workflows[0].setup[0]
    assert step.params["mscrpath"] == mscr_path_for_channel(21, variant=variant)
    assert step.params["mscrpath"] != mscr_path_for_channel(21)


def test_take_production_read_override_matching_the_base_stays_on_the_base_script(
    clean_variants, no_script_writes
):
    """An override that resolves to the run's own sweep is not a second sweep."""
    from softae.core.eis_scripts import EISParams, begin_run, mscr_path_for_channel

    begin_run(EISParams.from_preset("Quick"))
    executor = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21],
                                     measurement=MeasurementSpec(preset="Quick"),
                                     executor=executor))
    step = executor.workflows[0].setup[0]
    assert step.params["mscrpath"] == mscr_path_for_channel(21)
    assert no_script_writes[0][2] is None


# ── Ordering: after the settle phase, never beside it ────────────────────────

def test_take_production_read_runs_after_drive_settle_phase_returns():
    """The ordering is the entire point of the function.

    A read taken *during* the hold is another settle round; only one taken after
    the driver has returned can claim to be the reading closest to equilibrium.
    """
    log: list[str] = []

    async def fake_drive_settle_phase():
        for round_index in range(3):
            log.append(f"settle_round:{round_index}")
        return "ceiling"

    async def scenario():
        outcome = await fake_drive_settle_phase()
        raws = await take_production_read(
            _spec(), [21], executor=FakeExecutor(log))
        return outcome, raws

    outcome, raws = asyncio.run(scenario())
    assert outcome == "ceiling"
    assert log == ["settle_round:0", "settle_round:1", "settle_round:2",
                   "run:production_campaign_production"]
    assert raws[21] is not None


# ── Refusals and identity ────────────────────────────────────────────────────

def test_take_production_read_disabled_measurement_takes_no_read():
    """`enabled=False` is *formulate and cast, but do not measure*."""
    executor = FakeExecutor()
    spec = _spec(measurement=MeasurementSpec(preset="Quick", enabled=False))
    assert asyncio.run(take_production_read(spec, [21], executor=executor)) == {}
    assert executor.workflows == []


def test_take_production_read_stamps_the_sample_uuids_it_was_given():
    """One sample, several measurements — the read carries the film's identity."""
    executor = FakeExecutor()
    asyncio.run(take_production_read(
        _spec(), [21, 22], executor=executor,
        sample_uuid_by_channel={21: "uuid-a", 22: "uuid-b"}))
    tags = {s.tags["channel"]: s.tags.get("sample_uuid")
            for s in executor.workflows[0].setup}
    assert tags == {"21": "uuid-a", "22": "uuid-b"}


def test_take_production_read_stamps_the_settle_tags_it_was_given():
    """T11.33: the board word and this well's own verdict ride onto the row.

    The tags are the only route these four values have to ``record_measurement``
    — ``analysis/eis/router.py`` reads them off the step it is handed — so a step
    that does not carry them writes four NULLs.
    """
    executor = FakeExecutor()
    asyncio.run(take_production_read(
        _spec(), [21, 22], executor=executor,
        extra_tags_by_channel={
            21: {"certification": "settled", "well_verdict": "rate_quiet",
                 "rate_per_hour": "0.004", "upper_bound_per_hour": "0.019"},
            22: {"certification": "settled", "well_verdict": "rate_moving"},
        }))
    tags = {s.tags["channel"]: s.tags for s in executor.workflows[0].setup}
    assert tags["21"]["certification"] == "settled"
    assert tags["21"]["well_verdict"] == "rate_quiet"
    assert tags["21"]["rate_per_hour"] == "0.004"
    assert tags["21"]["upper_bound_per_hour"] == "0.019"
    # Present-only *within* a channel too: ch22 was judged but never reached a
    # rate estimate, and an absent slope must not become a zero.
    assert tags["22"]["well_verdict"] == "rate_moving"
    assert "rate_per_hour" not in tags["22"]


def test_take_production_read_leaves_a_channel_absent_from_the_settle_tags_alone():
    """Present-only: an unmapped channel gets no tag, not an empty one.

    ``tags.get("certification")`` downstream must distinguish *this read was not
    taken under a settle phase* from *it settled*, which is exactly the
    distinction a blank string would destroy.
    """
    executor = FakeExecutor()
    asyncio.run(take_production_read(
        _spec(), [21, 22], executor=executor,
        extra_tags_by_channel={21: {"certification": "ceiling"}, 22: {}}))
    tags = {s.tags["channel"]: s.tags for s in executor.workflows[0].setup}
    assert tags["21"]["certification"] == "ceiling"
    assert "certification" not in tags["22"]


def test_take_production_read_without_settle_tags_changes_no_tag():
    """The regression guard: omitting the parameter is byte-identical to before."""
    plain = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21, 22], executor=plain))
    explicit_none = FakeExecutor()
    asyncio.run(take_production_read(_spec(), [21, 22], executor=explicit_none,
                                     extra_tags_by_channel=None))

    assert ([dict(s.tags) for s in plain.workflows[0].setup]
            == [dict(s.tags) for s in explicit_none.workflows[0].setup])
    for step in plain.workflows[0].setup:
        assert not {"certification", "well_verdict", "rate_per_hour",
                    "upper_bound_per_hour"} & set(step.tags)


def test_take_production_read_settle_tags_do_not_disturb_the_sample_uuids():
    """Both stampers run on one workflow; neither may erase the other's tags."""
    executor = FakeExecutor()
    asyncio.run(take_production_read(
        _spec(), [21], executor=executor,
        sample_uuid_by_channel={21: "uuid-a"},
        extra_tags_by_channel={21: {"certification": "settled"}}))
    step = executor.workflows[0].setup[0]
    assert step.tags["sample_uuid"] == "uuid-a"
    assert step.tags["certification"] == "settled"
    # And the tags the modality itself set are still there.
    assert step.tags["measurement"] == "primary"


def test_take_production_read_restores_the_executors_previous_callback():
    """A caller may hand over a live executor; the capture hook is borrowed."""
    executor = FakeExecutor()
    sentinel = lambda *a: None  # noqa: E731
    executor.on_step_complete = sentinel
    asyncio.run(take_production_read(_spec(), [21], executor=executor))
    assert executor.on_step_complete is sentinel


def test_build_production_read_workflow_is_none_when_nothing_can_be_measured():
    """No per-channel step → no workflow, rather than an empty one that 'ran'."""
    from softae.core import production_read as module
    from softae.core.modality_registry import Modality, ModalityDisplay

    silent = Modality(
        name="quiet",
        build_measure_step=lambda channel, spec: None,
        router_factory=lambda: None,
        objectives={},
        prepare_run=lambda spec, channels, **kw: None,
        display=ModalityDisplay(display_name="Analysis only"),
    )
    original = module.__dict__.get("get_modality")
    assert original is None, "get_modality is imported inside the function"

    import softae.core.modality_registry as registry

    real = registry.get_modality
    registry.get_modality = lambda name: silent
    try:
        wf = build_production_read_workflow(
            _spec(measurement=MeasurementSpec(modality="quiet")), [21])
    finally:
        registry.get_modality = real
    assert wf is None
