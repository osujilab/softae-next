"""T3.1 runtime seam + T3.1b storage — the campaign path actually derives labels.

Spec: ``docs/SubAgent docs/failure_informed_feasibility_spec.md`` §3.1-§3.3.

``tests/test_failure_labels.py`` pins the *decision rules* in isolation. This file
pins that a real campaign, driven through ``run_autonomous_campaign`` on a mock
rig, reaches those rules at all: that a gate REJECT triggers confirmation sweeps,
that the sweeps never touch the objective, and that the outcome lands in the
``doe_parameters`` columns T3.1b added.

The distinction matters because the whole feature was inert for one session — the
modules were complete and correct, and **nothing called them**. A rules test
cannot see that; only a campaign test can.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from softae.core.autonomous_loop import AutonomousLoop
from softae.core.autonomous_wiring import (
    CampaignSpec,
    DataStoreOutcomeSink,
    build_confirmation_workflow,
    confirmation_step_name,
    is_primary_measurement,
    run_autonomous_campaign,
)
from softae.core.data_store import DataStore
from softae.drivers.mock_factory import create_mock_manager
from softae.optimizers.failure_labels import (
    CONFIRM_MEASUREMENT,
    OUTCOME_MEASURED,
    OUTCOME_UNKNOWN,
)

SPACE = {
    "vol_p0": {"type": "float", "low": 5.0, "high": 30.0},
    "vol_p1": {"type": "float", "low": 5.0, "high": 30.0},
}

N = 12
_F = np.logspace(5, -1, N)


def _trace(magnitude: float) -> list:
    mag = magnitude * np.linspace(1.0, 1.6, N)
    return [np.column_stack([_F, mag * 0.8, mag * 0.6])]


OPEN = _trace(1e13)
GOOD = _trace(2.0e3)


def _spec(**over) -> CampaignSpec:
    base = dict(
        name="seam_campaign",
        channels=(21, 22),
        pcb_name="SoftAE_EIS_4Stripe",
        parameter_space=SPACE,
        vol_params=("vol_p0", "vol_p1"),
        pump_ids=(0, 1),
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


def _force_traces(manager, by_channel: dict[int, list]):
    """Make every pico answer with a scripted trace, per channel.

    Returns the list of ``(channel, mscrpath)`` calls actually made, which is how
    the confirmation sweeps become observable: a repeat is a *second* call on the
    same channel with the same parameters.
    """
    calls: list[tuple[int, str]] = []

    for name in list(manager.names):
        inst = manager.get(name)
        if not hasattr(inst, "sendscript_getdata"):
            continue

        def scripted(mscrpath, outdir, chan, _c=calls):
            _c.append((int(chan), mscrpath))
            return by_channel.get(int(chan), GOOD)

        inst.sendscript_getdata = scripted
    return calls


# ── The loop hook ────────────────────────────────────────────────────────────

class TestTheLoopExposesRawResultsBeforeScoring:
    @pytest.mark.asyncio
    async def test_the_hook_may_augment_the_results_it_is_given(self):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0

        async def handler(results):
            results["added"] = 42
            return results

        loop.on_trial_measured = handler
        assert await loop._post_measure({"a": 1}) == {"a": 1, "added": 42}

    @pytest.mark.asyncio
    async def test_a_synchronous_handler_is_accepted_too(self):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0
        loop.on_trial_measured = lambda results: {**results, "sync": True}
        assert (await loop._post_measure({"a": 1}))["sync"] is True

    @pytest.mark.asyncio
    async def test_a_raising_handler_never_costs_the_trial(self):
        """An observation seam must not be able to kill a campaign."""
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0

        async def broken(results):
            raise RuntimeError("handler bug")

        loop.on_trial_measured = broken
        assert await loop._post_measure({"a": 1}) == {"a": 1}

    @pytest.mark.asyncio
    async def test_a_handler_returning_nonsense_leaves_the_results_alone(self):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0
        loop.on_trial_measured = lambda results: "not a mapping"
        assert await loop._post_measure({"a": 1}) == {"a": 1}

    @pytest.mark.asyncio
    async def test_no_handler_is_the_default_and_returns_results_unchanged(self):
        loop = AutonomousLoop.__new__(AutonomousLoop)
        loop._iteration = 0
        loop.on_trial_measured = None
        payload = {"a": 1}
        assert await loop._post_measure(payload) is payload


# ── The confirmation workflow ────────────────────────────────────────────────

class TestTheConfirmationWorkflowIsAReRead:
    def test_it_carries_exactly_one_confirm_tagged_step(self):
        wf = build_confirmation_workflow(_spec(), 21, 1)
        assert len(wf.setup) == 1
        step = wf.setup[0]
        assert step.name == confirmation_step_name(21, 1)
        assert step.tags["measurement"] == CONFIRM_MEASUREMENT
        assert is_primary_measurement(step.tags) is False

    def test_it_publishes_its_tags_so_the_step_is_never_anonymous(self):
        from softae.core.autonomous_wiring import _MEASUREMENT_TAGS_KEY

        wf = build_confirmation_workflow(_spec(), 21, 2)
        tags = wf.metadata[_MEASUREMENT_TAGS_KEY]
        assert tags[confirmation_step_name(21, 2)]["channel"] == "21"

    def test_it_casts_nothing_and_consumes_no_well(self):
        """No re-cast, no new well — the whole economic argument for repeats."""
        wf = build_confirmation_workflow(_spec(), 21, 1)
        methods = {s.method for s in wf.setup}
        assert methods == {"sendscript_getdata"}
        assert wf.teardown == [] or not wf.teardown


# ── End to end on a mock rig ─────────────────────────────────────────────────

class TestACampaignDerivesLabelsFromItsOwnMeasurements:
    @pytest.mark.asyncio
    async def test_a_healthy_campaign_records_measured_outcomes_and_no_repeats(
            self, connected, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        calls = _force_traces(connected, {})

        result = await run_autonomous_campaign(
            _spec(), manager=connected, data_store=store)

        rows = store.query_doe_parameters(run_id=result.run_id)
        assert rows and all(r["outcome"] == OUTCOME_MEASURED for r in rows)
        assert all(r["failure_reason"] is None for r in rows)
        # Two channels, one sweep each per trial, and NO repeats: a healthy
        # campaign must cost exactly what it cost before this feature existed.
        assert len(calls) == 2 * len(rows), calls
        assert sorted({c for c, _ in calls}) == [21, 22]
        store.close()

    @pytest.mark.asyncio
    async def test_a_rejecting_channel_triggers_confirmation_sweeps(
            self, connected, tmp_path: Path):
        """The reject is re-read on the SAME channel, up to twice, no re-cast."""
        store = DataStore(tmp_path / "proj")
        calls = _force_traces(connected, {21: OPEN})
        events: list[dict] = []

        await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store,
            on_event=events.append)

        ch21 = [c for c, _ in calls if c == 21]
        ch22 = [c for c, _ in calls if c == 22]
        # ch21: one primary + two agreeing repeats. ch22 gated fine, so one only.
        assert len(ch21) == 3, calls
        assert len(ch22) == 1, calls

        sweeps = [e for e in events if e["type"] == "confirmation_sweep"]
        assert [s["attempt"] for s in sweeps] == [1, 2]
        assert all(s["matched"] for s in sweeps)
        assert all(s["channel"] == 21 for s in sweeps)
        store.close()

    @pytest.mark.asyncio
    async def test_agreeing_repeats_with_a_board_accept_record_an_infeasible_label(
            self, connected, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        _force_traces(connected, {21: OPEN})
        events: list[dict] = []

        await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store,
            on_event=events.append)

        labelled = [e for e in events if e["type"] == "infeasible_label_recorded"]
        assert len(labelled) == 1
        assert labelled[0]["channel"] == 21
        assert labelled[0]["signature"] == "open"
        assert labelled[0]["confirmations"] == 2
        store.close()

    @pytest.mark.asyncio
    async def test_the_repeats_never_reach_the_objective(
            self, connected, tmp_path: Path):
        """The objective averages the two PRIMARY channels — never the repeats.

        With ``[quality] enabled = false`` (the shipped state) the open-circuit
        *primary* is still used, and T3.1 deliberately does not change that: it
        reads the gate's report without granting the gate authority it lacks
        today. So the number to pin is not "small" — it is the mean over exactly
        two measurements rather than four.

        Leaking the two confirm reads would move it from 6.5e12 to 9.75e12, which
        is what makes this able to fail rather than merely able to pass.
        """
        store = DataStore(tmp_path / "proj")
        _force_traces(connected, {21: OPEN})

        result = await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store)

        told = [v for _, v in result.history]
        assert told, "the trial should still have produced an objective"

        def _mean_abs_z(trace):
            arr = np.asarray(trace[0], dtype=float)
            return float(np.mean(np.hypot(arr[:, -2], arr[:, -1])))

        two_primaries = (_mean_abs_z(OPEN) + _mean_abs_z(GOOD)) / 2
        four_with_repeats = (3 * _mean_abs_z(OPEN) + _mean_abs_z(GOOD)) / 4
        assert told[0] == pytest.approx(two_primaries, rel=1e-9)
        assert told[0] != pytest.approx(four_with_repeats, rel=1e-9)
        store.close()

    @pytest.mark.asyncio
    async def test_a_whole_board_of_rejects_labels_nothing(
            self, connected, tmp_path: Path):
        """2e, end to end: no ACCEPT anywhere means no bad chemistry proven."""
        store = DataStore(tmp_path / "proj")
        _force_traces(connected, {21: OPEN, 22: OPEN})
        events: list[dict] = []

        await run_autonomous_campaign(
            _spec(budget=1), manager=connected, data_store=store,
            on_event=events.append)

        assert not [e for e in events if e["type"] == "infeasible_label_recorded"]
        # ...and the sweeps still ran: the rule declined on corroboration, not
        # because the evidence was never gathered.
        assert [e for e in events if e["type"] == "confirmation_sweep"]
        store.close()


# ── T3.1b — the storage the interim interface now writes through ─────────────

class TestTheOutcomeColumnsBackTheInterimInterface:
    def test_the_columns_exist_and_default_to_null(self, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        cols = {
            r[1] for r in
            store._conn.execute("PRAGMA table_info(doe_parameters)").fetchall()
        }
        assert {"outcome", "failure_reason"} <= cols
        run_id = store.start_run("w")
        store.record_doe_parameter(run_id, 1, 0, {"x": 1.0})
        row = store.query_doe_parameters(run_id=run_id)[0]
        assert row["outcome"] is None and row["failure_reason"] is None
        store.close()

    def test_a_historical_row_is_never_backfilled(self, tmp_path: Path):
        """Undeclared is unknown, never empty.

        Re-opening the store runs every migration again; a legacy row must come
        back NULL rather than acquiring an invented outcome.
        """
        path = tmp_path / "proj"
        store = DataStore(path)
        run_id = store.start_run("w")
        store.record_doe_parameter(run_id, 1, 0, {"x": 1.0})
        store.close()

        reopened = DataStore(path)
        row = reopened.query_doe_parameters(run_id=run_id)[0]
        assert row["outcome"] is None
        reopened.close()

    def test_the_migration_is_idempotent(self, tmp_path: Path):
        path = tmp_path / "proj"
        for _ in range(3):
            store = DataStore(path)
            store._migrate_doe_outcome()       # explicitly, twice over
            store.close()
        store = DataStore(path)
        cols = [
            r[1] for r in
            store._conn.execute("PRAGMA table_info(doe_parameters)").fetchall()
        ]
        assert cols.count("outcome") == 1
        assert cols.count("failure_reason") == 1
        store.close()

    def test_the_sink_writes_the_reason_through_to_the_row(self, tmp_path: Path):
        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("w")
        store.record_doe_parameter(run_id, 7, 0, {"x": 1.0})

        DataStoreOutcomeSink(store).record_outcome(
            run_id=run_id, channel=7, params={"x": 1.0},
            outcome=OUTCOME_UNKNOWN,
            failure_reason="open x1 — withheld: no confirmation sweep",
        )
        row = store.query_doe_parameters(run_id=run_id)[0]
        assert row["outcome"] == OUTCOME_UNKNOWN
        assert "no confirmation sweep" in row["failure_reason"]
        store.close()

    def test_a_reason_for_a_trial_that_was_never_recorded_writes_nothing(
            self, tmp_path: Path):
        """A reason with no experiment attached to it is not a row worth having."""
        store = DataStore(tmp_path / "proj")
        run_id = store.start_run("w")
        assert store.update_doe_outcome(
            run_id=run_id, channel=99, outcome=OUTCOME_UNKNOWN) is None
        assert store.query_doe_parameters(run_id=run_id) == []
        store.close()

    def test_a_missing_store_degrades_quietly(self):
        """A campaign without a DataStore must still derive labels in memory."""
        DataStoreOutcomeSink(None).record_outcome(
            run_id=None, channel=1, params={}, outcome=OUTCOME_UNKNOWN,
            failure_reason="x")
