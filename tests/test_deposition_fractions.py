"""Pure-Python tests for the deposition Σ-indicator state machine (no Qt)."""

from __future__ import annotations

import pytest

from softae.gui.widgets.deposition_fractions import (
    FractionRow,
    FractionSumState,
    normalize_to_one,
    resolve_sum_state,
)


def _row(name, *, checked=True, auto=False, value=0.0, dep_fraction=0.5) -> FractionRow:
    return FractionRow(name, checked, auto, value, dep_fraction)


class TestResolveSumState:
    def test_all_auto_reports_sum_one_and_auto_count(self):
        rows = [_row(n, auto=True) for n in ("a", "b", "c")]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "ok"
        assert state.message == "Σ = 1.00 (3 auto-balanced)"
        assert state.n_auto == 3

    def test_explicit_exact_one_no_auto_is_ok(self):
        rows = [_row("a", value=0.6), _row("b", value=0.4)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "ok"
        assert state.message == "Σ = 1.00"

    def test_explicit_sum_below_one_no_auto_warns_short(self):
        rows = [_row("a", value=0.3), _row("b", value=0.4)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "warn"
        assert "Σ = 0.70 < 1" in state.message
        assert "short by 6.00 µL" in state.message

    def test_explicit_sum_above_one_no_auto_warns_overshoot(self):
        rows = [_row("a", value=0.8), _row("b", value=0.4)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "warn"
        assert "Σ = 1.20 > 1" in state.message
        assert "overshoots target by 4.00 µL" in state.message

    def test_explicit_exceeds_one_with_auto_is_error(self):
        rows = [_row("a", value=1.2), _row("b", auto=True)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "error"
        assert "Σ_explicit = 1.20" in state.message

    def test_carrier_only_checked_excluded_from_sum(self):
        rows = [_row("a", value=0.6), _row("b", value=0.4),
                _row("water", value=0.9, dep_fraction=0.0)]
        state = resolve_sum_state(rows, 20.0)
        assert state.explicit_sum == 0.6 + 0.4  # carrier-only 0.9 not folded in
        assert state.n_auto == 0
        assert state.severity == "ok"
        assert state.message == "Σ = 1.00 (+1 carrier-only bulk share)"

    def test_carrier_only_explicit_nonzero_adds_bulk_suffix(self):
        rows = [_row("a", auto=True), _row("water", value=0.5, dep_fraction=0.0)]
        state = resolve_sum_state(rows, 20.0)
        assert state.carrier_bulk_count == 1
        assert state.message.endswith("(+1 carrier-only bulk share)")

    def test_no_dep_bearing_checked_stock_warns(self):
        rows = [_row("water", value=0.5, dep_fraction=0.0)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "warn"
        assert "No dep-bearing stock selected" in state.message

    def test_tolerance_boundary_sum_within_tol_is_ok(self):
        rows = [_row("a", value=1.0 + 5e-7)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "ok"
        assert state.message == "Σ = 1.00"

    def test_all_explicit_zero_no_auto_prompts_to_set(self):
        # OVERRIDE (explicit-first default): dep-bearing rows all at 0.00, no Auto.
        rows = [_row("a", value=0.0), _row("b", value=0.0)]
        state = resolve_sum_state(rows, 20.0)
        assert state.severity == "warn"
        assert state.message == "Σ = 0.00 — set fractions or enable Auto"
        assert isinstance(state, FractionSumState)


class TestNormalizeToOne:
    def test_normalize_to_one_scales_explicit_only_to_sum_one(self):
        out = normalize_to_one([0.30, 0.20])
        assert out == pytest.approx([0.60, 0.40])
        assert sum(out) == pytest.approx(1.00)

    def test_normalize_to_one_scales_down_when_over_one(self):
        out = normalize_to_one([0.80, 0.60])
        assert sum(out) == pytest.approx(1.00)
        assert out[0] < 0.80
        assert out[1] < 0.60

    def test_normalize_to_one_distributes_rounding_residual_to_largest(self):
        out = normalize_to_one([0.4, 0.4, 0.4])
        assert out == [0.34, 0.33, 0.33]
        assert sum(out) == pytest.approx(1.00)

    def test_normalize_to_one_zero_total_returns_input_unchanged(self):
        out = normalize_to_one([0.0, 0.0])
        assert out == [0.0, 0.0]

    def test_normalize_to_one_ignores_zero_rows_keeps_them_zero(self):
        out = normalize_to_one([0.0, 0.5])
        assert out == pytest.approx([0.0, 1.0])
