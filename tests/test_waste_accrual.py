"""Automatic waste accrual from executed steps (P5.4).

Waste is a **wall-clock** cap on unattended time once purging runs, so the level
has to move on its own. Before this it only changed when an operator edited it
by hand, which is exactly how a number drifts until nobody trusts it.

The classifier is shared with the preflight projection on purpose: two
classifiers over the same step shapes would be two things to keep in step with
the deposition engine.
"""

from __future__ import annotations

import asyncio

import pytest

from softae.core.preflight import (
    step_goes_to_waste,
    step_pump_volumes,
    step_waste_uL,
)
from softae.workflows.workflow_model import WorkflowStep


def _step(name, method, params, **tags):
    step = WorkflowStep(name=name, instrument="syringe", method=method,
                        params=params)
    return step.with_tags(**tags) if tags else step


class TestClassification:
    def test_a_cast_does_not_go_to_waste(self):
        """It is on the board — that is the entire point of the run."""
        step = _step("deposit_ch1", "single_drop_simul",
                     {"ids": [0, 1, 2], "vols": [10.0, 5.0, 5.0]},
                     phase="deposit", channel="1")
        assert not step_goes_to_waste(step)
        assert step_waste_uL(step) == 0.0

    def test_a_precondition_flush_does(self):
        step = _step("precondition_ch1", "precondition_flush",
                     {"ids": [0, 1, 2], "vol_list": [10.0, 5.0, 5.0],
                      "flush_factor": 3.0},
                     phase="precondition", channel="1")
        assert step_goes_to_waste(step)
        assert step_waste_uL(step) == pytest.approx(60.0)      # 20 × 3

    def test_a_teardown_flush_does(self):
        step = _step("final_flush", "single_pump",
                     {"ID": 0, "dispense_vol": 80.0})
        assert step_waste_uL(step) == 80.0

    def test_the_phase_tag_beats_the_method(self):
        """`single_pump` is ambiguous — the deposit_pumpN catalog tasks cast
        with it, and the teardown flush dispenses with it."""
        cast = _step("deposit_pump0", "single_pump",
                     {"ID": 0, "dispense_vol": 12.0}, phase="deposit")
        assert step_waste_uL(cast) == 0.0

    def test_an_untagged_cast_still_classifies_by_method(self):
        """Workflows built outside the recipe engine must still be sane."""
        step = _step("drop", "single_drop_simul",
                     {"ids": [0], "vols": [10.0]})
        assert not step_goes_to_waste(step)

    def test_a_non_dispensing_step_accrues_nothing(self):
        step = _step("anneal_ch1", "anneal", {"hold_time_s": 300},
                     phase="anneal")
        assert step_waste_uL(step) == 0.0

    def test_zero_volumes_accrue_nothing(self):
        step = _step("precondition", "precondition_flush",
                     {"ids": [0, 1, 2], "vol_list": [0.0, 0.0, 0.0],
                      "flush_factor": 3.0}, phase="precondition")
        assert step_waste_uL(step) == 0.0


class TestSharedTraversal:
    def test_draw_and_waste_read_the_same_volumes(self):
        """One traversal, so the projection and the meter cannot disagree."""
        step = _step("precondition_ch1", "precondition_flush",
                     {"ids": [0, 1, 2], "vol_list": [10.0, 5.0, 5.0],
                      "flush_factor": 2.0}, phase="precondition")
        assert step_pump_volumes(step) == {0: 20.0, 1: 10.0, 2: 10.0}
        assert step_waste_uL(step) == sum(step_pump_volumes(step).values())


class _Waste:
    def __init__(self):
        self.total = 0.0

    def add(self, volume_uL):
        self.total += float(volume_uL)


class _OkInstrument:
    async def execute(self, method_name, **kwargs):
        return {"ok": True}


class _OkManager:
    def get(self, name):
        return _OkInstrument()


class TestExecutorAccrual:
    def _executor(self, waste):
        from softae.workflows.workflow_executor import WorkflowExecutor

        ex = WorkflowExecutor(_OkManager())
        ex.waste_ledger = waste
        return ex

    def test_a_flush_step_books_itself(self):
        waste = _Waste()
        ex = self._executor(waste)
        step = _step("final_flush", "single_pump",
                     {"ID": 0, "dispense_vol": 80.0})

        asyncio.run(ex._run_step(step, 0, 1))

        assert waste.total == 80.0

    def test_a_cast_books_nothing(self):
        waste = _Waste()
        ex = self._executor(waste)
        step = _step("deposit_ch1", "single_drop_simul",
                     {"ids": [0], "vols": [10.0]}, phase="deposit")

        asyncio.run(ex._run_step(step, 0, 1))

        assert waste.total == 0.0

    def test_a_failed_step_books_nothing(self):
        """Accrual happens on the success path only — a step that never ran
        put nothing in the container."""
        class _Boom:
            def get(self, name):
                raise KeyError(name)

        from softae.workflows.workflow_executor import WorkflowExecutor

        waste = _Waste()
        ex = WorkflowExecutor(_Boom())
        ex.waste_ledger = waste
        step = _step("final_flush", "single_pump",
                     {"ID": 0, "dispense_vol": 80.0})

        with pytest.raises(Exception):
            asyncio.run(ex._run_step(step, 0, 1))
        assert waste.total == 0.0

    def test_no_ledger_is_not_an_error(self):
        from softae.workflows.workflow_executor import WorkflowExecutor

        ex = WorkflowExecutor(_OkManager())          # waste_ledger unset
        asyncio.run(ex._run_step(
            _step("final_flush", "single_pump",
                  {"ID": 0, "dispense_vol": 80.0}), 0, 1))

    def test_a_broken_ledger_does_not_fail_the_step(self):
        """Bookkeeping must never fail a step that physically succeeded."""
        class _Broken:
            def add(self, volume_uL):
                raise RuntimeError("db gone")

        ex = self._executor(_Broken())
        asyncio.run(ex._run_step(
            _step("final_flush", "single_pump",
                  {"ID": 0, "dispense_vol": 80.0}), 0, 1))

    def test_purges_are_not_double_counted(self):
        """Purges dispense straight through the syringe, not as executor steps,
        and book themselves — so the executor must never see them."""
        from softae.core.purge import PurgeScheduler, PurgeSettings
        from softae.core.purge_runner import IdleRestState, PurgeRunner

        class _Clock:
            def __init__(self):
                self.t = 0.0        # scheduler seeds its timers from here

            def __call__(self):
                return self.t

        class _Syringe:
            def is_head_up(self):
                return False

            def single_pump(self, **kw):
                pass

        class _Stage:
            def live_position(self):
                return (-50.0, 50.0)

        class _M:
            def get(self, name):
                return _Syringe() if name == "syringe" else _Stage()

        waste = _Waste()
        settings = PurgeSettings(enabled=True, actuate=True, interval_s=900.0,
                                 particulate_uL=20.0, other_uL=10.0,
                                 particulate_pumps=(1,), pumps=(0, 1, 2))
        clock = _Clock()
        sched = PurgeScheduler(settings, now=clock)
        clock.t = 1000.0                      # past the interval → due
        runner = PurgeRunner(_M(), sched, waste_ledger=waste,
                             idle_rest=IdleRestState(True),
                             flush_xy=(-50.0, 50.0))

        assert runner.maybe_purge().performed

        assert waste.total == 40.0        # booked exactly once, by the runner
