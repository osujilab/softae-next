"""Config keys that look like safety limits must actually be limits.

Both of these shipped as inert knobs: declared in ``[safety]``, read by nothing,
and — for the temperature pair — holding values identical to the live copy, so the
disconnection produced no visible symptom. An operator lowering a limit and seeing
no error would reasonably conclude it had taken effect.
"""

from __future__ import annotations

import pytest

from softae.drivers.contracts import temp_setpoint_limits, validate_temp_setpoint
from softae.errors import SafetyError


class TestTemperatureLimits:
    def test_the_stricter_safety_limit_wins(self):
        lo, hi = temp_setpoint_limits({"max_temp": 200.0, "min_temp": 5.0},
                                      {"temp_max_C": 80.0, "temp_min_C": 10.0})
        assert (lo, hi) == (10.0, 80.0)

    def test_a_stricter_instrument_limit_also_wins(self):
        """The direction that matters: a global limit must not *loosen* a local one.

        A thermal limit is an interlock, not a capability. Adding a second source
        may only ever tighten the window — otherwise declaring a rig-wide 200 °C
        ceiling would raise the ceiling of a controller rated to 150.
        """
        lo, hi = temp_setpoint_limits({"max_temp": 150.0, "min_temp": 8.0},
                                      {"temp_max_C": 200.0, "temp_min_C": 5.0})
        assert (lo, hi) == (8.0, 150.0)

    def test_either_source_alone_is_honoured(self):
        assert temp_setpoint_limits({"max_temp": 150.0, "min_temp": 8.0}, {}) == \
            (8.0, 150.0)
        assert temp_setpoint_limits({}, {"temp_max_C": 90.0, "temp_min_C": 4.0}) == \
            (4.0, 90.0)

    def test_both_absent_falls_back_to_the_shipped_envelope(self):
        assert temp_setpoint_limits({}, {}) == (5.0, 200.0)

    def test_a_lowered_safety_ceiling_actually_refuses_the_setpoint(self, monkeypatch):
        """The behaviour the whole finding was about: editing [safety] must bite.

        The instrument still declares a 200 °C ceiling. Before this, that is the one
        that applied and 120 °C sailed through.
        """
        import softae.config.loader as loader

        monkeypatch.setattr(loader, "safety",
                            lambda: {"temp_max_C": 80.0, "temp_min_C": 5.0})
        inst = {"max_temp": 200.0, "min_temp": 5.0}

        with pytest.raises(SafetyError):
            validate_temp_setpoint(120.0, inst, "temp_controller")

        validate_temp_setpoint(70.0, inst, "temp_controller")   # still permitted

    def test_a_non_numeric_limit_is_ignored_rather_than_crashing_a_run(self):
        lo, hi = temp_setpoint_limits({"max_temp": 150.0}, {"temp_max_C": "hot"})
        assert hi == 150.0


class TestStepTimeoutDefault:
    def _executor(self):
        from softae.workflows.workflow_executor import WorkflowExecutor

        return WorkflowExecutor.__new__(WorkflowExecutor)

    def test_the_configured_default_is_read(self, monkeypatch):
        import softae.config.loader as loader

        monkeypatch.setattr(loader, "safety", lambda: {"step_timeout_s": 42.0})
        ex = self._executor()
        assert ex._default_step_timeout_s() == 42.0

    def test_zero_means_unbounded_and_is_not_confused_with_absent(self, monkeypatch):
        """The escape hatch: a genuinely open-ended step says so, rather than the
        number being raised until it stops mattering."""
        import softae.config.loader as loader

        monkeypatch.setattr(loader, "safety", lambda: {"step_timeout_s": 0})
        ex = self._executor()
        assert ex._default_step_timeout_s() == 0

    def test_an_unreadable_config_still_bounds_the_step(self, monkeypatch):
        import softae.config.loader as loader

        def boom():
            raise RuntimeError("no config")

        monkeypatch.setattr(loader, "safety", boom)
        ex = self._executor()
        assert ex._default_step_timeout_s() == 900.0

    def test_it_is_cached_so_a_32_channel_run_does_not_reparse_per_step(self,
                                                                       monkeypatch):
        import softae.config.loader as loader

        calls = []
        monkeypatch.setattr(
            loader, "safety", lambda: calls.append(1) or {"step_timeout_s": 7.0})
        ex = self._executor()
        for _ in range(10):
            ex._default_step_timeout_s()
        assert len(calls) == 1
