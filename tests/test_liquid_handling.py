from __future__ import annotations

import pytest

from softae.core.liquid_handling import (
    CorrectionInput,
    LinePhysicsConfig,
    LiquidHandlingCorrector,
    SystemPhysicsConfig,
)


def _line(viscosity: float = 1.0, alpha_base: float = 0.2) -> LinePhysicsConfig:
    return LinePhysicsConfig(
        line_id=0,
        cracking_kpa_per_valve=8.0,
        compliance_uL_per_kpa=0.55,
        alpha_base=alpha_base,
        viscosity_mpas=viscosity,
    )


def test_effective_cracking_scales_with_viscosity():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig(beta=0.30, eta_ref_mpas=1.0)
    low_eta = corrector.corrected_command(CorrectionInput(10.0, 0, 1), _line(viscosity=1.0), sys_cfg)
    high_eta = corrector.corrected_command(CorrectionInput(10.0, 0, 1), _line(viscosity=4.0), sys_cfg)
    assert high_eta.dead_uL > low_eta.dead_uL


def test_dead_volume_decreases_with_run_index_when_alpha_growth_positive():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig(alpha_growth_per_run=0.05)
    first = corrector.corrected_command(CorrectionInput(10.0, 0, 1), _line(), sys_cfg)
    later = corrector.corrected_command(CorrectionInput(10.0, 0, 5), _line(), sys_cfg)
    assert later.dead_uL < first.dead_uL


def test_commanded_equals_target_plus_dead():
    corrector = LiquidHandlingCorrector()
    out = corrector.corrected_command(CorrectionInput(12.0, 0, 1), _line(), SystemPhysicsConfig())
    assert out.commanded_uL == pytest.approx(out.target_uL + out.dead_uL)


def test_commanded_clamped_non_negative():
    corrector = LiquidHandlingCorrector()
    out = corrector.corrected_command(CorrectionInput(-3.0, 0, 1), _line(), SystemPhysicsConfig())
    assert out.target_uL == 0.0
    assert out.dead_uL == 0.0
    assert out.commanded_uL == 0.0


def test_corrected_pair_sums_to_total():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig()
    line_map = {0: _line(), 1: LinePhysicsConfig(1, 8.0, 0.55, 0.2, 1.0)}
    p0, p1, total = corrector.corrected_pair(8.0, 13.0, 2, line_map, sys_cfg)
    assert total == pytest.approx(p0 + p1)


def test_corrected_multi_three_pumps_sums_to_total():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig()
    line_map = {
        0: _line(),
        1: LinePhysicsConfig(1, 8.0, 0.55, 0.2, 1.0),
        2: LinePhysicsConfig(2, 8.0, 0.55, 0.2, 1.0),
    }
    commanded, total = corrector.corrected_multi([8.0, 13.0, 5.0], 2, line_map, sys_cfg)
    assert len(commanded) == 3
    assert total == pytest.approx(sum(commanded))
    # Each pump gets its target plus a positive dead volume.
    assert all(c > t for c, t in zip(commanded, [8.0, 13.0, 5.0]))


def test_corrected_multi_matches_corrected_pair():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig()
    line_map = {0: _line(), 1: LinePhysicsConfig(1, 8.0, 0.55, 0.2, 1.0)}
    (m0, m1), m_total = corrector.corrected_multi([8.0, 13.0], 2, line_map, sys_cfg)
    p0, p1, total = corrector.corrected_pair(8.0, 13.0, 2, line_map, sys_cfg)
    assert (m0, m1, m_total) == pytest.approx((p0, p1, total))


def test_prime_volume_is_positive_and_scales_with_margin():
    corrector = LiquidHandlingCorrector()
    sys_cfg = SystemPhysicsConfig()
    base = corrector.prime_volume(_line(), sys_cfg)
    larger = corrector.prime_volume(_line(), sys_cfg, margin=1.5)
    assert base > 0.0
    assert larger == pytest.approx(base * 1.5 / 1.2)
