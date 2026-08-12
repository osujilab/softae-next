"""Long anneals must not be aborted by a task's short hand-set timeout (P1.6)."""

from __future__ import annotations

import pytest

from softae.core.deposition_recipe import (
    ANNEAL_RAMP_ALLOWANCE_S,
    ANNEAL_TIMEOUT_MARGIN,
    anneal_timeout_s,
)
from softae.workflows.workflow_model import WorkflowStep


class TestDerivedTimeout:
    def test_long_hold_raises_a_short_declared_ceiling(self):
        """The real scenario: a 4 h soft-material anneal on a 600 s task."""
        four_hours = 4 * 3600.0
        out = anneal_timeout_s({"hold_time_s": four_hours}, 600.0)
        assert out > four_hours                      # would have died at 600 s
        assert out == four_hours * ANNEAL_TIMEOUT_MARGIN + ANNEAL_RAMP_ALLOWANCE_S

    def test_generous_declared_ceiling_is_never_reduced(self):
        assert anneal_timeout_s({"hold_time_s": 60.0}, 99_999.0) == 99_999.0

    def test_builtin_five_minute_anneal_still_fits(self):
        """The shipped 300 s hold / 600 s task must stay valid."""
        assert anneal_timeout_s({"hold_time_s": 300.0}, 600.0) >= 300.0

    @pytest.mark.parametrize("params", [{}, {"hold_time_s": 0}, {"hold_time_s": None},
                                        {"hold_time_s": "abc"}])
    def test_no_usable_hold_leaves_the_declared_value(self, params):
        assert anneal_timeout_s(params, 600.0) == 600.0

    def test_none_declared_still_derives_a_floor(self):
        assert anneal_timeout_s({"hold_time_s": 100.0}, None) == (
            100.0 * ANNEAL_TIMEOUT_MARGIN + ANNEAL_RAMP_ALLOWANCE_S
        )


class TestWithTimeout:
    def test_with_timeout_preserves_everything_else(self):
        s = WorkflowStep(
            name="anneal_all", instrument="temp_controller", method="anneal",
            params={"hold_time_s": 10}, timeout_s=600.0, retry=2,
        ).with_tags(phase="anneal")
        out = s.with_timeout(1234.0)
        assert out.timeout_s == 1234.0
        assert out.name == s.name and out.params == s.params
        assert out.retry == s.retry and out.tags == s.tags
        assert s.timeout_s == 600.0          # original untouched (immutable-friendly)
