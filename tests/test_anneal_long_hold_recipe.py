"""An 8 h / 85 °C hold at a commanded RH — catalog, guard, and the RH gate.

Covers `docs/SubAgent docs/anneal_recipe_long_hold.md`: the long-hold anneal task,
the four RH tasks and their setup ordering, the catalog guard that makes a hold
longer than its own ceiling impossible, and the one correctness edge — an
``rh_wait`` that *fails* when the humidity never arrives.
"""

from __future__ import annotations

import asyncio

import pytest

from softae.config import loader
from softae.core.deposition_recipe import build_recipe_deposition_workflow, get_deposition_recipe
from softae.core.run_plan import PhaseKind, PhaseScope, RunPhase, RunPlan
from softae.core.task_catalog import (
    Task,
    TaskCatalog,
    TaskValidationError,
    validate_task,
)
from softae.drivers.async_rh_controller import AsyncRHController
from softae.drivers.mock_rh_controller import MockRHController
from softae.errors import InstrumentError
from softae.server.base_instrument import InstrumentState

PCB = {"grid": [8, 4], "spacing_mm": [10, 10]}
EIGHT_HOURS = 28_800.0

#: The RH tasks, in the order a campaign's setup must run them.
RH_SETUP_ORDER = ["rh_set_low", "rh_start", "rh_wait"]


@pytest.fixture(scope="module")
def catalog() -> TaskCatalog:
    """The *shipped* catalog — these tests pin ``data/tasks.toml``, not a stub."""
    return TaskCatalog.load_toml(loader.tasks_toml_path())


# ── 1. The long-hold anneal task ─────────────────────────────────────────────

class TestLongHoldAnnealTask:
    def test_anneal_85c_8h_resolves_with_the_full_hold(self, catalog):
        task = catalog.get("anneal_85C_8h")
        assert task.instrument == "temp_controller"
        assert task.method == "anneal"
        assert task.params["target_temp_C"] == 85
        assert float(task.params["hold_time_s"]) == EIGHT_HOURS

    def test_anneal_85c_8h_timeout_outlasts_its_own_hold(self, catalog):
        task = catalog.get("anneal_85C_8h")
        assert task.timeout_s is not None
        assert task.timeout_s > EIGHT_HOURS

    def test_shipped_catalog_has_no_unsound_task(self, catalog):
        assert catalog.validate() == {}


# ── 2. The catalog guard on hold-vs-timeout ──────────────────────────────────

def _anneal(timeout_s, hold_time_s=EIGHT_HOURS, method="anneal") -> Task:
    return Task(name="bad_anneal", instrument="temp_controller", method=method,
                params={"target_temp_C": 85, "hold_time_s": hold_time_s},
                timeout_s=timeout_s)


class TestAnnealTimeoutGuard:
    def test_timeout_below_hold_is_a_problem(self):
        problems = validate_task(_anneal(600.0))
        assert problems and "600" in problems[0] and "28800" in problems[0]

    def test_timeout_equal_to_hold_is_a_problem(self):
        """Equal leaves nothing for the ramp, settle and restore — still fatal."""
        assert validate_task(_anneal(EIGHT_HOURS))

    def test_missing_timeout_is_a_problem(self):
        assert validate_task(_anneal(None))

    def test_generous_timeout_is_sound(self):
        assert validate_task(_anneal(29_400.0)) == []

    def test_guard_applies_only_to_anneals(self):
        """A long ``wait`` param on another method is not this rule's business."""
        assert validate_task(_anneal(600.0, method="single_pump")) == []

    def test_no_declared_hold_is_not_flagged(self):
        assert validate_task(_anneal(600.0, hold_time_s=0)) == []
        assert validate_task(_anneal(600.0, hold_time_s=None)) == []

    def test_strict_add_refuses(self):
        with pytest.raises(TaskValidationError, match="bad_anneal"):
            TaskCatalog().add(_anneal(600.0), strict=True)

    def test_non_strict_add_stores_so_an_editor_never_crashes(self):
        cat = TaskCatalog()
        cat.add(_anneal(600.0))
        assert "bad_anneal" in cat
        assert "bad_anneal" in cat.validate()   # stored, but reported as unsound

    def test_load_rejects_a_malformed_anneal_but_keeps_the_rest(self, tmp_path):
        """The deliberately malformed fixture: 8 h hold under a 600 s ceiling."""
        path = tmp_path / "tasks.toml"
        path.write_text(
            "[tasks.anneal_bad]\n"
            'instrument = "temp_controller"\n'
            'method = "anneal"\n'
            "timeout_s = 600.0\n"
            "[tasks.anneal_bad.params]\n"
            "hold_time_s = 28800\n"
            "\n"
            "[tasks.anneal_good]\n"
            'instrument = "temp_controller"\n'
            'method = "anneal"\n'
            "timeout_s = 29400.0\n"
            "[tasks.anneal_good.params]\n"
            "hold_time_s = 28800\n",
            encoding="utf-8",
        )
        cat = TaskCatalog.load_toml(path)
        assert "anneal_bad" not in cat          # refused, not catalogued
        assert "anneal_good" in cat             # one bad table costs only itself


# ── 3. The phase carries the full timeout (the 600 s default is gone) ────────

def _build_with_anneal(catalog: TaskCatalog, anneal_task: str):
    plan = RunPlan((
        RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
        RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH, anneal_task=anneal_task),
    ))
    return build_recipe_deposition_workflow(
        get_deposition_recipe("single_drop"), [21], {21: [10.0, 30.0, 0.0]},
        catalog=catalog, pump_ids=[0, 1, 2], dispense_rate=100.0, flush_rate=500.0,
        flush_factor=3.0, settle_factor=2.0, settle_base_s=0.0,
        start_flush_uL=[80, 80, 80], pcb=PCB, origin_xy=(43.5, 50.0),
        eis_step_by_channel=None, run_plan=plan,
    )


class TestAnnealPhaseTimeout:
    def test_eight_hour_phase_step_outlives_the_hold(self, catalog):
        wf = _build_with_anneal(catalog, "anneal_85C_8h")
        step = next(s for s in wf.setup if s.name == "anneal_all")
        assert step.params["hold_time_s"] == EIGHT_HOURS
        assert step.timeout_s > EIGHT_HOURS

    def test_the_short_anneals_600s_ceiling_is_not_inherited(self, catalog):
        """Different task, different ceiling — the 5-minute default is not shadowed."""
        short = next(s for s in _build_with_anneal(catalog, "anneal_150C_5min").setup
                     if s.name == "anneal_all")
        long_ = next(s for s in _build_with_anneal(catalog, "anneal_85C_8h").setup
                     if s.name == "anneal_all")
        assert short.timeout_s < long_.timeout_s
        assert long_.timeout_s > EIGHT_HOURS


# ── 4. The RH tasks and their setup ordering ─────────────────────────────────

class TestRHTasks:
    @pytest.mark.parametrize("name,method", [
        ("rh_set_low", "set_setpoint"),
        ("rh_start", "start"),
        ("rh_wait", "wait"),
        ("rh_stop", "stop"),
    ])
    def test_task_resolves_to_an_rh_controller_step(self, catalog, name, method):
        step = catalog.get(name).to_step()
        assert step.instrument == "rh_controller"
        assert step.method == method

    def test_setpoint_is_a_parameter_above_the_hot_rh_floor(self, catalog):
        """20 %RH, not 15: the attainable floor rises with chamber temperature."""
        assert float(catalog.get("rh_set_low").params["val"]) >= 20.0

    def test_rh_wait_gates_rather_than_reports(self, catalog):
        assert catalog.get("rh_wait").params["raise_on_timeout"] is True

    def test_rh_wait_task_ceiling_outlasts_its_own_wait(self, catalog):
        """Otherwise the executor kills the step before the driver can fail it."""
        task = catalog.get("rh_wait")
        assert task.timeout_s > float(task.params["timeout"])

    def test_setup_order_is_setpoint_then_start_then_wait(self, catalog):
        steps = [catalog.get(n).to_step() for n in RH_SETUP_ORDER]
        assert [s.method for s in steps] == ["set_setpoint", "start", "wait"]

    def test_rh_setup_precedes_the_anneal_phase(self, catalog):
        """RH is an independent PID loop: commanded in setup, held during the cure."""
        wf = _build_with_anneal(catalog, "anneal_85C_8h")
        names = RH_SETUP_ORDER + [s.name for s in wf.setup]
        assert names.index("rh_wait") < names.index("anneal_all")
        assert names.index("rh_set_low") < names.index("rh_start") < names.index("rh_wait")


# ── 5. rh_wait must be able to fail ──────────────────────────────────────────

def _idle_controller(rh: float) -> AsyncRHController:
    """A controller whose reading never moves — no serial, no PID thread, no rig."""
    ctl = AsyncRHController(config={"poll_period": 0.01}, rh_reader=lambda: rh)
    ctl._current_rh = rh
    return ctl


class TestRHWaitFailure:
    def test_unreached_target_raises_when_gated(self):
        ctl = _idle_controller(60.0)
        with pytest.raises(InstrumentError, match="did not reach"):
            ctl.wait(target=20.0, tol=2.0, timeout=0.05, raise_on_timeout=True)

    def test_unreached_target_still_returns_by_default(self):
        """Default False — existing best-effort callers are untouched."""
        assert _idle_controller(60.0).wait(target=20.0, tol=2.0, timeout=0.05) is None

    def test_reached_target_never_raises(self):
        ctl = _idle_controller(20.5)
        assert ctl.wait(target=20.0, tol=2.0, timeout=5.0, raise_on_timeout=True) is None

    def test_mock_rh_reaches_the_commanded_low_setpoint(self, catalog):
        """The simulated path the campaign runs offline still settles at target."""
        mock = MockRHController()
        params = dict(catalog.get("rh_set_low").params)
        mock.set_setpoint(float(params["val"]))
        mock.start()
        mock.wait(tol=2.0)
        assert mock.get_H() == pytest.approx(float(params["val"]), abs=3.0)


# ── 6. The task-dispatch path (params forwarded verbatim by execute) ─────────

def _dispatch(instrument, catalog: TaskCatalog, names: list[str]):
    """Run catalogued tasks the way the executor does — ``execute(method, **params)``.

    This is the route that matters: :meth:`BaseInstrument.execute` forwards task
    params verbatim with no signature filtering, so a param the driver does not
    accept is a ``TypeError`` at run time.  Calling ``wait()`` directly bypasses
    exactly the coupling under test.
    """
    async def run():
        await instrument.connect()
        for name in names:
            task = catalog.get(name)
            await instrument.execute(task.method, **task.params)

    return asyncio.run(run())


class TestRHTaskDispatch:
    def test_the_rh_setup_sequence_dispatches_onto_the_mock(self, catalog):
        """rh_set_low → rh_start → rh_wait, params and all, against the simulator."""
        mock = MockRHController()
        _dispatch(mock, catalog, RH_SETUP_ORDER)
        assert mock.get_H() == pytest.approx(
            float(catalog.get("rh_set_low").params["val"]), abs=3.0)

    def test_rh_wait_dispatch_fails_when_the_mock_cannot_reach_target(self, catalog):
        """The flag is honoured through dispatch, not merely accepted."""
        mock = MockRHController()
        mock._wait_abort.set()        # the mock configured never to reach target
        mock._rh = 60.0
        with pytest.raises(InstrumentError, match="did not reach"):
            _dispatch(mock, catalog, ["rh_set_low", "rh_start", "rh_wait"])

    def test_rh_wait_dispatches_onto_the_real_driver_signature(self, catalog):
        """Same params, same call, no hardware — the signatures must not drift."""
        ctl = _idle_controller(60.0)
        ctl._state = InstrumentState.CONNECTED
        params = dict(catalog.get("rh_wait").params) | {"timeout": 0.05}
        with pytest.raises(InstrumentError, match="did not reach"):
            asyncio.run(ctl.execute("wait", **params))

    def test_rh_teardown_dispatches(self, catalog):
        mock = MockRHController()
        _dispatch(mock, catalog, RH_SETUP_ORDER + ["rh_stop"])
        assert mock.status()["running"] is False
