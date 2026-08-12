from __future__ import annotations

import pytest

from softae.campaigns.doe import (
    ExperimentDesign,
    ParamScale,
    design_to_campaign,
)
from softae.errors import CampaignError


# ── ParamScale.values ──


class TestParamScale:
    def test_linear(self):
        assert ParamScale("x", "linear", 0.0, 1.0, 5).values() == pytest.approx(
            [0.0, 0.25, 0.5, 0.75, 1.0]
        )

    def test_linear_single_step(self):
        assert ParamScale("x", "linear", 0.3, 0.9, 1).values() == [0.3]

    def test_log(self):
        vals = ParamScale("f", "log", 1.0, 1000.0, 4).values()
        assert vals == pytest.approx([1.0, 10.0, 100.0, 1000.0])

    def test_pointwise(self):
        assert ParamScale("t", "pointwise", points=[25.0, 40.0, 10.0]).values() == [
            10.0, 25.0, 40.0
        ]

    def test_bad_scale_raises(self):
        with pytest.raises(CampaignError):
            ParamScale("x", "bogus").values()

    def test_linear_missing_bounds_raises(self):
        with pytest.raises(CampaignError):
            ParamScale("x", "linear").values()


# ── Composition pool enumeration ──


class TestCandidatePool:
    def test_binary_sums_to_one(self):
        design = ExperimentDesign(
            components=["A", "B"],
            param_scales=[ParamScale("x_A", "linear", 0.0, 1.0, 5)],
        )
        pool = design.candidate_pool()
        assert len(pool) == 5
        for pt in pool:
            assert pt["x_A"] + pt["x_B"] == pytest.approx(1.0)

    def test_ternary_feasibility_filter(self):
        # x_A and x_B each in {0, 0.5, 1.0}; drop points where x_A + x_B > 1
        design = ExperimentDesign(
            components=["A", "B", "C"],
            param_scales=[
                ParamScale("x_A", "linear", 0.0, 1.0, 3),
                ParamScale("x_B", "linear", 0.0, 1.0, 3),
            ],
        )
        pool = design.candidate_pool()
        # feasible (x_A, x_B): (0,0)(0,.5)(0,1)(.5,0)(.5,.5)(1,0) = 6
        assert len(pool) == 6
        for pt in pool:
            assert pt["x_A"] + pt["x_B"] + pt["x_C"] == pytest.approx(1.0)
            assert pt["x_C"] >= -1e-9

    def test_environment_axes_multiply(self):
        design = ExperimentDesign(
            components=["A", "B"],
            param_scales=[
                ParamScale("x_A", "linear", 0.0, 1.0, 3),
                ParamScale("temperature_C", "pointwise", points=[25.0, 40.0]),
            ],
        )
        pool = design.candidate_pool()
        assert len(pool) == 6  # 3 compositions x 2 temperatures
        assert {pt["temperature_C"] for pt in pool} == {25.0, 40.0}

    def test_case_insensitive_component_binding(self):
        # x_water axis should bind to the "Water" component
        design = ExperimentDesign(
            components=["Water", "Glycerol"],
            param_scales=[ParamScale("x_water", "linear", 0.0, 1.0, 3)],
        )
        pool = design.candidate_pool()
        assert all("x_Water" in pt and "x_Glycerol" in pt for pt in pool)


# ── Prototype Experiment-JSON compatibility ──


LYO_JSON = """
{
  "id": "20260211_190814_lyo_phase_exploration",
  "name": "Lyo phase exploration",
  "components": [
    {"solution_id": "Water", "role": "solvent"},
    {"solution_id": "50-50 glycerol-water stock", "role": "solute"}
  ],
  "param_scales": [
    {"name": "temperature_C", "scale_type": "pointwise", "points": [25.0]},
    {"name": "rh_pct", "scale_type": "pointwise", "points": [60.0]},
    {"name": "x_water", "scale_type": "linear", "start": 0.0, "stop": 1.0, "steps": 5}
  ],
  "target_deposition_uL": 20.0
}
"""


class TestExperimentJson:
    def test_from_json_roundtrip(self):
        design = ExperimentDesign.from_json(LYO_JSON)
        assert design.components == ["Water", "50-50 glycerol-water stock"]
        assert design.target_deposition_uL == 20.0
        assert len(design.composition_axes()) == 1
        assert len(design.environment_axes()) == 2
        # re-serialise and reload
        again = ExperimentDesign.from_json(design.to_json())
        assert again.components == design.components

    def test_lyo_pool(self):
        design = ExperimentDesign.from_json(LYO_JSON)
        pool = design.candidate_pool()
        assert len(pool) == 5  # 5 x_water points x 1 temp x 1 rh
        for pt in pool:
            assert pt["x_Water"] + pt["x_50-50 glycerol-water stock"] == pytest.approx(1.0)
            assert pt["temperature_C"] == 25.0
            assert pt["rh_pct"] == 60.0


# ── Adapter to BOCampaignConfig ──


class TestDesignToCampaign:
    def test_config_and_pool(self):
        design = ExperimentDesign(
            name="binary sweep",
            components=["A", "B"],
            param_scales=[ParamScale("x_A", "linear", 0.0, 1.0, 10)],
            meta={"seed": 7, "objective_direction": "maximize"},
        )
        config, pool = design_to_campaign(design)
        assert len(pool) == 10
        assert config.seed == 7
        assert config.annotation == "binary sweep"
        assert config.n_initial < len(pool)
        config.validate(pool_size=len(pool))  # must not raise
