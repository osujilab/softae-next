"""Rung 2 — the four-well mock rehearsal, with REAL fits.

Spec: ``docs/SubAgent docs/rung2_four_well_rehearsal.md`` (T11.3). Proves the
operator plan's Verification steps 2 and 6 together, on the mock rig:

* **step 2** — the emitted step order (cast conditions -> 4x deposit -> anneal
  conditions -> ``anneal_all`` -> equilibrate conditions -> measure), the settle
  driver running, ``settle.json`` carrying a verdict, and the recorded reading
  coming from a ``production*`` step taken *after* the settle rounds;
* **step 6** — the settle loop itself, against the mock rig, with the real gated
  fitter, proving ``settled`` / ``ceiling`` / ``not_evaluable`` are each
  reachable and distinguishable. *"The last attempt hung on EIS fitting"*, so
  nothing here scripts a fit: ``settle_round_fits`` runs for real.

Three things this module exists to say out loud, each of which cost a measured
run to find:

1. **``install_mock_picos`` must run BEFORE ``connect_all()``.** It puts freshly
   constructed (and therefore DISCONNECTED) picos into ``manager._instruments``;
   installed after connecting, every sweep dies ``[pico2] not connected``.
2. **Without a recorded thickness there is no sigma, and without sigma the only
   reachable verdict is ``not_evaluable``.** The campaign's own predicted tier
   needs a composition-mode spec; these raw-volume specs get ``twin = None`` and
   a NULL thickness. ``_record_thickness_on_run_start`` supplies the
   profilometry tier through the public writer, and it is a precondition of this
   module meaning anything — see its docstring.
3. **``max_hold_s`` is sized against ROUND cost, not against the intended
   hold.** A four-channel round costs 23-38 s here: ~2.6 s per gated fit plus
   roughly 17 s of executor and payload work that does not shrink with the
   channel count. At ``max_hold_s=3.0`` — the value the existing fast-settle
   fixture uses — the *first round's fits* overrun the ceiling and the phase
   reports ``not_evaluable`` after one round; at 45 s it reached three rounds
   on some runs and two on others. Both look exactly like a board that could
   not be judged, and one of them is. See :data:`CEILING_HOLD_S`, and note that
   every scenario asserts ``n_rounds >= 3`` before it asserts an outcome, so a
   future slowdown reports itself instead of changing a verdict.
4. **A railed R1 cannot be forced through this mock's apex knob, and no
   scenario here tries.** The spec drafted the unjudgeable well as a very high
   apex — a designed R1 below the ``simpleSalt`` floor. Measured, that does not
   work: across four apex values from 5e7 to 5e8 Hz, twelve fits each, the
   FITTED R1 ranged 0.0-918 ohm and railed **2/12 in every case**, with no
   dependence on the apex at all. The spectrum up there carries no resolvable
   arc, so the fitter is fitting noise and the rail is a ~1-in-6 coin flip per
   (channel, sweep). A test that leant on it would pass on the run it was
   written against and flake afterwards — it did, once, here. The exclusion
   these boards use instead is ``sigma_null``, withheld per channel and
   deterministic by construction: **no thickness for a well, no sigma for it,
   no vote from it.** The R1-bound threading itself stays covered where it is
   deterministic, in ``test_campaign_settle_phase.py`` and
   ``test_equilibration_analysis.py``, against fabricated fits.

**Sentinels.** One strict ``xfail`` remains — T11.8's ``certification`` field on
a measurement row, whose name parallel has confirmed. T11.7's sentinel has been
retired: the criterion pass-through landed, and the test that guarded it is live
below. It is worth knowing *why* it was replaced rather than simply un-marked —
it asserted on rendered prose that could never contain the token it looked for,
so it could not have flipped to XPASS when the feature arrived. A sentinel is
only a sentinel if the thing it waits for can actually trip it.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from softae.analysis.equilibration import (
    EXCLUDED_SIGMA_NULL,
    RATE_MOVING,
    SETTLE_CEILING,
    SETTLE_CRITERION_DEVIATION,
    SETTLE_CRITERION_RATE,
    SETTLE_NOT_EVALUABLE,
    SETTLE_SETTLED,
    RoundFit,
)
from softae.config import loader
from softae.core import autonomous_wiring as wiring
from softae.core.autonomous_wiring import (
    SETTLE_MEASUREMENT,
    CampaignSpec,
    build_settle_round_workflow,
    build_trial_workflow,
    drive_settle_phase,
    settle_r1_bound_ohms,
)
from softae.core.data_store import DataStore
from softae.core.eis_scripts import EISParams, variant_for
from softae.core.measurement_spec import MeasurementSpec
from softae.core.modality_registry import get_modality
from softae.core.phase_setpoints import PhaseSetpoints
from softae.core.production_read import (
    PRODUCTION_STEP_PREFIX,
    build_production_read_workflow,
)
from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)
from softae.core.task_catalog import TaskCatalog
from softae.drivers.mock_factory import create_mock_manager
from softae.tools.eis_validate_mock import (
    MockRig,
    install_fast_conditions,
    install_mock_picos,
)

CHANNELS = (21, 22, 23, 24)

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}

#: A quiet, well-resolved well: the arc closes inside the Quick preset's band
#: and the fitted R1 sits near 4.8e7 ohm, round to round, to within ~0.5 %.
QUIET_APEX_HZ = 30.0

#: The thickness handed to the profilometry tier. Any positive number works —
#: sigma is a ratio and the criterion is relative — but it must EXIST, and a
#: channel it is withheld from is a channel with no sigma at all. That is the
#: lever the ``not_evaluable`` board uses; see :data:`SCENARIOS`.
REHEARSAL_THICKNESS_UM = 50.0


@dataclass
class RecordingRig(MockRig):
    """A :class:`MockRig` that remembers which ``.mscr`` each sweep read.

    The script path is the only place the per-phase preset override is visible
    from outside: a variant sweep writes ``softae_ch{N}.{variant}.mscr`` and the
    base writes ``softae_ch{N}.mscr``. Recording is all this adds; the spectrum
    is the shipped one.
    """

    calls: list[tuple[int, str]] = field(default_factory=list)

    def measure(self, channel: int, script_path: str | Path) -> Any:
        self.calls.append((int(channel), os.path.basename(str(script_path))))
        return super().measure(channel, script_path)


@dataclass
class Rehearsal:
    """Everything one campaign run leaves behind, gathered once."""

    run_id: str
    events: list[dict[str, Any]]
    calls: list[tuple[int, str]]
    rows: list[dict[str, Any]]
    settle_records: list[dict[str, Any]]
    n_trials: int

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

    def stems(self) -> list[str]:
        """Each measurement row's file stem, in ``measurement_id`` order.

        Ordered by ``measurement_id`` and **not** by ``sweep_order``: the latter
        restarts at 1 for every workflow, so the trial's sweeps, each settle
        round and the production read all carry 1, 2, 3, 4.
        """
        ordered = sorted(self.rows, key=lambda r: int(r["measurement_id"]))
        return [os.path.basename(str(r.get("eis_file_path") or ""))
                for r in ordered]


def _settle_plan(max_hold_s: float, **over: Any) -> SettlePlan:
    base = dict(round_period_s=0.05, min_hold_s=0.0, max_hold_s=max_hold_s,
                settle_n_rounds=3, settle_min_channels=3,
                # Explicitly off. The gate is ON by default and judges the ROOM;
                # armed here, the mock chamber's spread would decide the verdict
                # instead of sigma, which is not what this module is about.
                rh_stability_pct=None)
    base.update(over)
    return SettlePlan(**base)


def _run_plan(max_hold_s: float, *, measurement: MeasurementSpec | None = None) -> RunPlan:
    return RunPlan(phases=(
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
                 conditions=PhaseSetpoints(name="cast", temp_setpoint_C=25.0)),
        # `conditions` on an ANNEAL phase is the temperature the chamber is
        # RESTORED to when the hold ends, not a second spelling of the cure.
        # 25 C, so no following phase reads a board still at 150 C.
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                 anneal_task="anneal_150C_5min",
                 conditions=PhaseSetpoints(name="rest", temp_setpoint_C=25.0)),
        RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                 settle=_settle_plan(max_hold_s),
                 conditions=PhaseSetpoints(name="equil", temp_setpoint_C=25.0)),
        RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                 measurement=measurement),
    ))


def _spec(max_hold_s: float, *, measurement: MeasurementSpec | None = None,
          **over: Any) -> CampaignSpec:
    """A four-well campaign that settles, said once — structurally.

    The settle parameters live on the EQUILIBRATE phase and **nowhere else**:
    ``CampaignSpec.settle_plan()`` refuses a spec that also sets the flat
    ``equilibration_method`` / ``round_period_s`` / ... fields, because picking
    one silently is how a campaign holds for a duration nobody wrote down.

    ``batch`` is left at its default. With ``batch=True`` and ``budget=1`` the
    q-batch asks for one suggestion and casts **one** well; the default casts
    the trial's formulation into all four, which is the board this is about.
    """
    base = dict(
        name="rung2_rehearsal",
        channels=CHANNELS,
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
        deadvols=(10.0, 30.0),
        time_scale=0.0,          # instant casting; this is not a timing test
        budget=1,
        seed=7,
        run_plan=_run_plan(max_hold_s, measurement=measurement),
    )
    base.update(over)
    return CampaignSpec(**base)


#: The ceiling for the two boards that never settle, and therefore always spend
#: it. Sized against **measured round cost, not against an intended hold**: the
#: deadline is tested after each round, so three rounds happen only if rounds
#: one and two both finish inside it. A four-channel round costs 23-38 s here —
#: roughly 17 s of executor and payload work plus ~2.6 s per gated fit — so 45 s
#: reached three rounds sometimes and two rounds otherwise, and a two-round run
#: reports ``not_evaluable`` with ``participating=[]`` because the tracker never
#: had a window to judge. That is the right word for the wrong reason, and it is
#: indistinguishable from the verdict scenario C exists to prove. 80 s clears two
#: worst-case rounds with margin; the tests assert ``n_rounds >= 3`` so a future
#: slowdown says *why* it failed rather than merely that it did.
CEILING_HOLD_S = 80.0

#: The three boards, the ceiling each needs, and which wells are given a
#: thickness. See the module docstring on why the ceilings are tens of seconds
#: rather than the fractions of a second the neighbouring fast-settle fixture
#: uses, and why the unjudgeable wells are withheld thicknesses rather than
#: driven to the R1 floor.
#:
#: Every well is quiet in all three: what separates the boards is the DRIFT
#: (does sigma move?) and the THICKNESS (does the well have a sigma at all?),
#: and both of those are deterministic.
SCENARIOS: dict[str, dict[str, Any]] = {
    # Four wells, all measurable, none moving -> the criterion says settled.
    "settled": dict(
        max_hold_s=90.0,
        drift_decades_per_hour=0.0,
        thickness_channels=CHANNELS,
    ),
    # Four wells, all measurable, all still drying -> held to the ceiling.
    "ceiling": dict(
        max_hold_s=CEILING_HOLD_S,
        drift_decades_per_hour=1.5,
        thickness_channels=CHANNELS,
    ),
    # Two wells carry no thickness, so they carry no sigma and cannot vote:
    # 2 participants against settle_min_channels=3.
    "not_evaluable": dict(
        max_hold_s=CEILING_HOLD_S,
        drift_decades_per_hour=0.0,
        thickness_channels=(21, 22),
    ),
}


async def _drive(kind: str, project_dir: Path) -> Rehearsal:
    scenario = SCENARIOS[kind]
    rig = RecordingRig(
        apex_hz={ch: QUIET_APEX_HZ for ch in CHANNELS},
        drift_decades_per_hour=float(scenario["drift_decades_per_hour"]),
        # Drift is per HOUR and a mock run finishes in seconds, so elapsed time
        # is counted in sweeps instead. Without this the drifting board would
        # show zero drift and `ceiling` would be a fake null.
        virtual_s_per_sweep=60.0,
    )
    manager = create_mock_manager(config={})
    install_fast_conditions(manager)
    # BEFORE connect_all: these picos are brand new and disconnected, and a
    # manager that has already connected will not connect them again.
    install_mock_picos(manager, rig)
    await manager.connect_all()

    store = DataStore(project_dir)
    events: list[dict[str, Any]] = []

    def on_event(event: dict[str, Any]) -> None:
        events.append(event)
        if event["type"] == "run_started":
            _record_thickness_on_run_start(
                store, str(event["run_id"]), scenario["thickness_channels"])

    try:
        result = await wiring.run_autonomous_campaign(
            _spec(float(scenario["max_hold_s"])), manager=manager,
            data_store=store, on_event=on_event)
        rows = store.query_measurements(run_id=result.run_id)
        sidecar = Path(store.run_dir(result.run_id)) / "settle.json"
        records: list[dict[str, Any]] = []
        if sidecar.exists():
            with open(str(sidecar), encoding="utf-8") as handle:
                records = json.load(handle)
        return Rehearsal(run_id=result.run_id, events=events,
                         calls=list(rig.calls), rows=rows,
                         settle_records=records, n_trials=result.n_trials)
    finally:
        store.close()
        await manager.disconnect_all()
        MockRig.clear_designed_r1()


def _record_thickness_on_run_start(
    store: DataStore, run_id: str, channels: Sequence[int]
) -> None:
    """Give the criterion a sigma to watch. **Not optional scaffolding.**

    ``settle_round_fits`` reports ``sigma=None`` for any spectrum whose report
    carries no conductivity, and conductivity needs a cell constant, which needs
    a thickness. The campaign's own ladder has two rungs: ``measured_thickness``
    (profilometry) and ``formulations.predicted_thickness_um`` (the deposition
    twin). The twin only speaks for a *composition-mode* spec; on the raw-volume
    axes used here it returns ``None``, so the row is NULL and withheld.

    Every channel would then be excluded ``sigma_null``, the board would be
    narrower than ``settle_min_channels``, and **every scenario would report
    ``not_evaluable``** — a module that looks like it is testing three outcomes
    while only one of them is reachable. So the profilometry rung is filled in
    here, through the public writer and the real lookup, at ``run_started``
    because that is the first moment the ``run_id`` exists.

    It is also the **per-channel** lever the ``not_evaluable`` board uses. A
    well with no thickness is excluded ``sigma_null`` every round, every time,
    with no dependence on a fit landing where a test hoped — which is exactly
    what the rail-by-apex route could not offer.
    """
    for channel in channels:
        store.record_thickness(int(channel), REHEARSAL_THICKNESS_UM,
                               run_id=run_id, instrument="rehearsal-fixture")


@pytest.fixture(scope="module")
def settled_run(tmp_path_factory) -> Rehearsal:
    return asyncio.run(_drive("settled", tmp_path_factory.mktemp("settled")))


@pytest.fixture(scope="module")
def ceiling_run(tmp_path_factory) -> Rehearsal:
    return asyncio.run(_drive("ceiling", tmp_path_factory.mktemp("ceiling")))


@pytest.fixture(scope="module")
def not_evaluable_run(tmp_path_factory) -> Rehearsal:
    return asyncio.run(
        _drive("not_evaluable", tmp_path_factory.mktemp("unevaluable")))


# ── Cheap surfaces: built, not run ───────────────────────────────────────────

def _trial_workflow():
    catalog = TaskCatalog.load_toml(loader.tasks_toml_path())
    return build_trial_workflow(_spec(90.0), {"vol_p0": 10.0, "vol_p1": 20.0},
                                catalog=catalog)


def test_the_trial_casts_then_cures_then_measures_in_that_order():
    """Plan step 2's step order, asserted by index rather than by list.

    A literal name list would break on any unrelated head-park or flush change
    and say nothing about the ordering it exists to protect. What matters is
    that the cure happens after the casting and before the reading, and that
    each phase's commanded conditions are written before the phase runs.
    """
    names = [step.name for step in _trial_workflow().setup]
    deposits = [i for i, n in enumerate(names) if n.startswith("deposit_ch")]
    measures = [i for i, n in enumerate(names) if n.startswith("measure_eis_ch")]

    assert len(deposits) == len(CHANNELS), names
    assert len(measures) == len(CHANNELS), names
    anneal = names.index("anneal_all")
    assert max(deposits) < anneal < min(measures)

    # Each phase's conditions block precedes the phase it conditions.
    assert names.index("conditions_cast_temp_sp_ch21") < min(deposits)
    assert names.index("conditions_rest_temp_wait_all") < anneal
    assert names.index("conditions_equil_temp_wait_all") < min(measures)


def test_a_settle_round_is_tagged_settle_and_a_production_read_is_tagged_primary():
    """The tag is what keeps a settle round out of the objective.

    Asserted on the workflows and not on the stored rows: ``measurements`` has
    no column for the measurement tag. What reaches the database is the step
    NAME, inside ``eis_file_path`` — which is what the provenance test below
    reads.
    """
    spec = _spec(90.0)
    rounds = build_settle_round_workflow(spec, CHANNELS, 0)
    assert rounds is not None
    assert {s.tags["measurement"] for s in rounds.setup} == {SETTLE_MEASUREMENT}

    production = build_production_read_workflow(spec, CHANNELS)
    assert production is not None
    assert {s.tags["measurement"] for s in production.setup} == {"primary"}
    assert all(s.name.startswith(f"{PRODUCTION_STEP_PREFIX}_")
               for s in production.setup)


def test_a_per_phase_preset_override_points_the_production_read_at_its_own_mscr():
    """The override's variant script, on the cheapest surface that carries it.

    ``build_production_read_workflow`` is where the override becomes a file
    path: it calls ``prepare_run(..., as_variant=True)`` itself, then builds the
    step through ``build_measure_step``, which asks ``variant_for`` which file
    was written for this sweep. Asserting here rather than on ``rig.calls``
    after a fourth campaign buys the same fact for ~80 s less rig time — the
    campaign path adds nothing to it, because the step it runs is this step.
    """
    spec = _spec(90.0)
    # Declare the campaign's own block the run's BASE first — the call
    # `run_autonomous_campaign` makes before any measurement step is built.
    # Without it the first sweep prepared *becomes* the base, so the override
    # below would take the unsuffixed path and `variant_for` would answer
    # `None`: the test would then be asserting that no variant exists.
    get_modality(spec.measurement.modality).prepare_run(spec.measurement, CHANNELS)

    base = build_production_read_workflow(spec, CHANNELS)
    assert base is not None
    assert all(s.params["mscrpath"].endswith(f"softae_ch{ch}.mscr")
               for s, ch in zip(base.setup, CHANNELS))

    override = MeasurementSpec(preset="Standard")
    varied = build_production_read_workflow(spec, CHANNELS, measurement=override)
    assert varied is not None
    key = variant_for(EISParams.from_preset(override.preset, **override.overrides))
    assert key is not None, "the override resolved to the run's base sweep"
    for step, channel in zip(varied.setup, CHANNELS):
        assert step.params["mscrpath"].endswith(f"softae_ch{channel}.{key}.mscr")


# ── The settled board ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_a_quiet_four_well_board_settles(settled_run: Rehearsal):
    """Step 6's first outcome, with real fits and nothing scripted.

    **Slow, and diagnosed rather than labelled**: one gated fit of a quiet mock
    well costs ~2.6 s, and this run does twelve of them across three settle
    rounds, plus the trial's own four and the production read's four. That cost
    *is* the thing under test — the last attempt at this hung on EIS fitting —
    so it is not something to trim away with a canned fit.
    """
    verdicts = settled_run.of_type("settle_verdict")
    assert len(verdicts) == 1
    # Diagnosis before assertion, as in the other two boards. A `settled` that
    # arrived on fewer than `settle_n_rounds` rounds would be a verdict the
    # tracker had no window to reach.
    assert verdicts[0]["n_rounds"] >= 3, (
        f"only {verdicts[0]['n_rounds']} round(s) fitted inside max_hold_s; this "
        f"verdict is the ceiling firing early, not the criterion speaking"
    )
    assert verdicts[0]["n_rounds"] == 3
    assert verdicts[0]["settle_outcome"] == SETTLE_SETTLED
    assert settled_run.n_trials == 1


@pytest.mark.slow
def test_every_well_that_could_be_judged_was_counted_as_evidence(
    settled_run: Rehearsal
):
    """All four wells vote, and the verdict says so rather than implying it.

    The complement of the ``not_evaluable`` board below: there, two wells are
    withheld a thickness and are excluded by name. Here nothing is withheld, so
    an exclusion of any kind would mean a quiet well silently stopped counting
    — which is how a board narrows without anybody noticing.
    """
    verdict = settled_run.of_type("settle_verdict")[0]
    assert sorted(verdict["participating"]) == list(CHANNELS)
    assert verdict["excluded"] == {}


@pytest.mark.slow
def test_each_round_is_narrated_and_says_when_it_was_not_yet_judged(
    settled_run: Rehearsal
):
    """*"Not judged yet"* must not be spelled the same way as *"judged, not settled"*."""
    rounds = settled_run.of_type("settle_round")
    assert [r["round"] for r in rounds] == [0, 1, 2]
    assert [r["judged"] for r in rounds] == [False, False, True]
    judged = rounds[-1]
    assert judged["settled"] is True and judged["evaluable"] is True
    assert judged["reason"]


@pytest.mark.slow
def test_the_verdict_survives_the_process_in_the_sidecar(settled_run: Rehearsal):
    """The event stream dies with the process; ``settle.json`` does not."""
    assert len(settled_run.settle_records) == 1
    record = settled_run.settle_records[0]
    assert record["settle_outcome"] == SETTLE_SETTLED
    assert record["n_rounds"] >= 1
    assert record["noise_floor_rel"] is not None
    assert record["channels"] == list(CHANNELS)


@pytest.mark.slow
def test_the_recorded_reading_is_a_production_sweep_taken_after_the_last_round(
    settled_run: Rehearsal
):
    """Plan step 2's assertion that nothing on the rig path made before.

    Every campaign score to date came from a ``measurement="settle"`` round
    whenever settling was enabled. This pins the replacement: four production
    steps exist, one per channel, and **every** one of them was recorded after
    **every** settle round — ordered by ``measurement_id``, because
    ``sweep_order`` restarts inside each workflow.
    """
    ordered = sorted(settled_run.rows, key=lambda r: int(r["measurement_id"]))
    production, settle = [], []
    for row in ordered:
        stem = os.path.basename(str(row.get("eis_file_path") or ""))
        assert stem.startswith(("measure_eis_", "settle_eis_",
                                f"{PRODUCTION_STEP_PREFIX}_")), stem
        if stem.startswith(f"{PRODUCTION_STEP_PREFIX}_"):
            production.append(row)
        elif stem.startswith("settle_eis_"):
            settle.append(row)

    assert sorted(int(r["channel"]) for r in production) == list(CHANNELS)
    assert settle, "no settle rounds were recorded at all"
    assert (min(int(r["measurement_id"]) for r in production)
            > max(int(r["measurement_id"]) for r in settle))


@pytest.mark.slow
def test_no_settle_round_was_filed_under_the_trials_own_measure_step(
    settled_run: Rehearsal
):
    """What a recycled settle round would look like in the record.

    A settle round written under ``measure_eis_ch{N}`` is indistinguishable
    downstream from the primary sweep, which is exactly the substitution the
    production read exists to stop.
    """
    stems = settled_run.stems()
    trial = [s for s in stems if s.startswith("measure_eis_ch")]
    assert len(trial) == len(CHANNELS), stems
    assert not any(s.startswith("settle_eis_") and "measure_eis" in s
                   for s in stems)


@pytest.mark.slow
def test_a_board_that_is_wide_enough_is_not_announced_as_narrow(
    settled_run: Rehearsal
):
    """Four channels clear ``settle_min_channels=3``, so the warning must be silent.

    The narrow-board announcement itself is covered by
    ``test_campaign_settle_phase.py``; what this board establishes is its
    absence, which is the half that would otherwise go unchecked.
    """
    assert not settled_run.of_type("settle_unevaluable_board")


@pytest.mark.slow
@pytest.mark.xfail(strict=True,
                   reason="T11.8 (parallel): no certification field exists on a "
                          "measurement row yet")
def test_the_production_row_carries_the_certification_it_was_taken_under(
    settled_run: Rehearsal
):
    """A sigma taken at ``ceiling`` is a weaker claim than one taken at ``settled``.

    Today the verdict lives only in the sidecar, so a reader holding a row
    cannot tell which. When T11.8 lands this flips to a loud XPASS. **If that
    column ships under a name other than ``certification``, this assertion moves
    with it** rather than being left to rot.
    """
    production = [r for r in settled_run.rows
                  if os.path.basename(str(r.get("eis_file_path") or ""))
                  .startswith(f"{PRODUCTION_STEP_PREFIX}_")]
    assert production
    assert production[0]["certification"] == SETTLE_SETTLED


# ── The other two outcomes ───────────────────────────────────────────────────

@pytest.mark.slow
def test_a_drifting_board_runs_to_the_ceiling_without_parking_the_campaign(
    ceiling_run: Rehearsal
):
    """``ceiling`` is an ordinary outcome for a slow film, not a fault.

    Parking an unattended run at 3 a.m. because one sample equilibrated slowly
    is the failure mode P0-P1 exists to prevent, so the campaign must proceed.
    """
    verdicts = ceiling_run.of_type("settle_verdict")
    assert len(verdicts) == 1
    # Diagnosis before assertion: a ceiling that fired before the tracker had a
    # full window is not a ceiling, it is a ceiling sized below the fit cost.
    assert verdicts[0]["n_rounds"] >= 3, (
        f"only {verdicts[0]['n_rounds']} round(s) fitted inside "
        f"max_hold_s={CEILING_HOLD_S:g}s; the ceiling is below the cost of three "
        f"four-channel rounds, so this scenario cannot reach a ceiling verdict"
    )
    assert verdicts[0]["settle_outcome"] == SETTLE_CEILING
    assert not ceiling_run.of_type("park")
    assert ceiling_run.n_trials == 1


@pytest.mark.slow
def test_two_wells_without_a_sigma_leave_the_criterion_unable_to_judge(
    not_evaluable_run: Rehearsal
):
    """*"Could not judge"* and *"judged, not settled"* want opposite actions.

    Two participants against ``settle_min_channels=3``: the phase runs to its
    ceiling and says it could not be evaluated, rather than reporting a drift it
    never measured. The two silent wells carry no thickness, so they carry no
    sigma — a real production state (an uncast or unmeasured well reaches the
    criterion the same way), and the only one of the exclusion reasons this rig
    can produce on demand.
    """
    verdicts = not_evaluable_run.of_type("settle_verdict")
    assert len(verdicts) == 1
    # Diagnosis before assertion. A phase that stopped before the tracker had a
    # full window also reports `not_evaluable` — with no participants and no
    # exclusions — and that is the *ceiling being too low*, not the board being
    # too narrow. The two are the same word for opposite facts.
    assert verdicts[0]["n_rounds"] >= 3, (
        f"only {verdicts[0]['n_rounds']} round(s) fitted inside "
        f"max_hold_s={CEILING_HOLD_S:g}s, so no window was ever judged; this "
        f"verdict is the ceiling firing early, not the criterion speaking"
    )
    assert verdicts[0]["settle_outcome"] == SETTLE_NOT_EVALUABLE
    assert sorted(verdicts[0]["participating"]) == [21, 22]
    # `settle.json` round-trips the mapping through JSON, so its keys are
    # strings; the event's are not. Compared as strings on both sides.
    excluded = {str(k): v for k, v in verdicts[0]["excluded"].items()}
    assert excluded == {"23": EXCLUDED_SIGMA_NULL, "24": EXCLUDED_SIGMA_NULL}
    # Still recorded, and still not a reason to stop.
    assert not_evaluable_run.settle_records
    assert not not_evaluable_run.of_type("park")


# ── T11.7: the criterion the driver routes on ──────────────────────────────
#
# The clock is `conftest.py`'s shared `FakeClock`, taken through its `fake_clock`
# fixture rather than imported: this module had a fifth private copy of a class
# that already existed four times, and CLAUDE.md §3.4 puts a pattern appearing in
# three or more files into `conftest`.

def _drive_fabricated(criterion: str, clock) -> tuple[Any, list[Any]]:
    """Run the settle driver over a script that rises 1.5x per round.

    No rig and no fits: ``measure_round`` / ``sleep`` / ``now`` are all injected,
    so this costs microseconds. A **rising** sigma is what makes the two criteria
    answer differently — a flat one settles under both and the test would not
    discriminate. Returns the outcome and every judged ``SettleCheck``.
    """
    channels = [21, 22, 23]
    checks: list[Any] = []
    seen = {"n": 0}

    async def measure_round(index: int) -> dict[int, Any]:
        return {ch: index for ch in channels}

    def fits_from(_raws) -> list[RoundFit]:
        seen["n"] += 1
        sigma = 1e-4 * (1.5 ** seen["n"])
        return [RoundFit(channel=ch, sigma=sigma, r1_ohms=4000.0)
                for ch in channels]

    def on_round(_index: int, check: Any) -> None:
        if check is not None:
            checks.append(check)

    async def go():
        return await drive_settle_phase(
            _settle_plan(9_000.0, round_period_s=600.0, criterion=criterion,
                         rate_tol_dec_per_h=0.05),
            channels=channels, measure_round=measure_round, fits_from=fits_from,
            r1_bound_ohms=settle_r1_bound_ohms(),
            sleep=clock.sleep, now=clock.now, on_round=on_round)

    outcome, _ = asyncio.run(go())
    return outcome, checks


def test_a_rate_plan_routes_on_the_rate_gate_and_a_deviation_plan_does_not(
    fake_clock,
):
    """T11.7: ``criterion`` must *route*, not merely be spelled in the plan.

    Live since T11.7 landed. It was a strict ``xfail`` and **the assertion could
    never have flipped it**: it looked for the literal tokens ``rate_moving`` /
    ``rate_undetectable`` in the rendered reason, and the ``RATE_MOVING`` branch
    renders prose (*"3 channel(s) still moving above 0.1151 ln/h: ch21 ch22
    ch23"*) while the ``tally`` that does embed tokens deliberately excludes
    ``RATE_MOVING``. A sentinel that cannot fire is worse than no sentinel, so
    everything below is asserted on **typed fields**, with the rendered text used
    for exactly one fact it is the only carrier of.

    **The deviation arm is the positive control.** Each typed fact the rate arm
    asserts is checked to be *different* under ``criterion="deviation"`` on the
    identical script — otherwise "rate routed" and "nothing routed" would look
    the same, which is the failure this test exists to rule out.

    Two surfaces, and the split is a finding rather than a convenience:
    ``as_settle_check`` flattens a :class:`RateCheck` into a
    :class:`SettleCheck`, so ``moving`` / ``quiet`` / ``by_channel`` do **not**
    reach ``on_round``. The per-channel classification is therefore asserted on
    the tracker, and the routing itself on the driver.
    """
    from softae.analysis.equilibration import SettleTracker, rate_tol_ln_per_hour

    channels = [21, 22, 23]

    # ── the driver: did the plan's criterion reach the tracker? ─────────────
    rate_outcome, rate_checks = _drive_fabricated(SETTLE_CRITERION_RATE, fake_clock)
    # The same clock, deliberately. `drive_settle_phase` reads `now()` only
    # relatively — `start = now()`, `deadline = start + max_hold_s`,
    # `t_s = now() - start` — so the second arm inheriting a clock that already
    # reads 9 000 s changes nothing, and one fixture instance serves both.
    dev_outcome, dev_checks = _drive_fabricated(SETTLE_CRITERION_DEVIATION,
                                                fake_clock)
    assert rate_checks and dev_checks
    rate_last, dev_last = rate_checks[-1], dev_checks[-1]

    # (a) The window each criterion judged. The rate gate reads
    # `max(n_rounds, min_fit_points) + 1` = 7 trailing rounds; the deviation gate
    # reads `settle_n_rounds` = 3. A typed integer, and the cheapest proof that
    # the plan's word changed which gate ran.
    assert rate_last.n_rounds == 7
    assert dev_last.n_rounds == 3

    # (a, cont.) `max_deviation_rel` is the deviation gate's own measurand, and
    # `as_settle_check` leaves it `None` because a rate verdict did not compute
    # one. `None` here means "a rate verdict", not "not judged": both checks are
    # `evaluable`.
    assert rate_last.evaluable and dev_last.evaluable
    assert rate_last.max_deviation_rel is None
    assert isinstance(dev_last.max_deviation_rel, float)

    # (c) The outcome, and the one prose fact only the rate gate can produce: its
    # tolerance is in ln/h, the deviation gate's is a percentage. Asserted after
    # the typed facts, never instead of them.
    assert rate_outcome.outcome == SETTLE_CEILING
    assert dev_outcome.outcome == SETTLE_CEILING
    assert "ln/h" in rate_last.reason
    assert "ln/h" not in dev_last.reason

    # ── the tracker: which cells were moving, by typed field ──────────────
    # (b) `RateCheck.moving` and `ChannelRate.refusal` are the classification
    # itself rather than a sentence about it. Driven here because the driver
    # flattens them away (see the docstring).
    tracker = SettleTracker(
        tol_rel=0.10, n_rounds=3, min_channels=3,
        r1_bound_ohms=settle_r1_bound_ohms(), rh_stability_pct=None,
        criterion=SETTLE_CRITERION_RATE,
        rate_tol_per_hour=rate_tol_ln_per_hour(0.05))
    for index in range(tracker.rate_window_rounds + 3):
        tracker.observe(
            [RoundFit(channel=ch, sigma=1e-4 * (1.5 ** (index + 1)),
                      r1_ohms=4000.0) for ch in channels],
            rh_median_pct=None, t_s=600.0 * index)

    rate = tracker.last_rate
    assert rate is not None, "the rate window never filled"
    assert rate.moving == channels
    assert rate.quiet == []
    assert {ch: rate.by_channel[ch].refusal for ch in channels} == {
        ch: RATE_MOVING for ch in channels}
