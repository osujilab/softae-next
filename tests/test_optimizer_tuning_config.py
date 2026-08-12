"""T3.1 §8 — two sources, ONE resolution point; and the fingerprint guard.

Spec: ``docs/SubAgent docs/failure_informed_feasibility_spec.md`` §8, §10
tests 10 and 13.

Test 13 is pinned **end-to-end from ``load_campaign_spec``**, not from a
hand-built :class:`CampaignSpec`, because the bug user decision (iii) closes was
precisely that the knobs existed on ``BayesianOptimizer`` and *nothing carried a
value to them*. A test that constructs the spec in Python skips the entire
stretch of pipe that was missing.
"""

from __future__ import annotations

import textwrap

import pytest

from softae.core.autonomous_wiring import (
    SPEC_UNSET,
    CampaignSpec,
    build_optimizer,
    campaign_spec_fingerprint,
    optimizer_tuning_identity,
    resolve_optimizer_tuning,
)
from softae.core.campaign_spec_io import load_campaign_spec, spec_from_dict, spec_to_dict
from softae.errors import OptimizerError

SPACE = {"x": {"type": "float", "low": 0.0, "high": 1.0}}


def _spec(**kw) -> CampaignSpec:
    return CampaignSpec(name="t", parameter_space=dict(SPACE), **kw)


@pytest.fixture
def no_site_defaults(monkeypatch):
    """Silence the TOML layer so a test measures the spec field alone."""
    from softae.config import loader

    monkeypatch.setattr(loader, "optimizer_tuning", lambda: {})
    monkeypatch.setattr(loader, "feasibility_config", lambda: {})


def _site(monkeypatch, *, optimizer=None, feasibility=None):
    from softae.config import loader

    monkeypatch.setattr(loader, "optimizer_tuning", lambda: dict(optimizer or {}))
    monkeypatch.setattr(loader, "feasibility_config", lambda: dict(feasibility or {}))


# ── Test 10 — the fingerprint guard ──────────────────────────────────────────

class TestTheResumeFingerprintIsUnmovedByTheNewFields:
    #: Computed from the code **before** T3.1's edit and asserted after, the
    #: mechanism T2.4 established. If either moves, every in-flight checkpoint
    #: becomes unresumable and the campaign is blamed for a parameter space
    #: nobody touched.
    PINNED_DEFAULT = "0f8e00b5fa722fed"
    PINNED_RICH = "3da6f5e9cad42e9b"

    def test_a_spec_not_using_the_feature_hashes_to_its_pre_t3_1_value(self):
        spec = CampaignSpec(
            name="fp_probe",
            parameter_space={"x": {"type": "float", "low": 0.0, "high": 1.0}})
        assert campaign_spec_fingerprint(spec) == self.PINNED_DEFAULT

    def test_a_fully_specified_pre_t3_1_spec_also_hashes_unchanged(self):
        spec = CampaignSpec(
            name="fp_probe2",
            parameter_space={"a": {"type": "float", "low": 0.0, "high": 2.0},
                             "b": {"type": "int", "low": 1, "high": 5}},
            objective="minimize", optimizer="bayesian", acquisition="ei",
            kappa=3.0, batch=True, seed=7, pump_ids=(0, 1, 2),
            pcb_name="SoftAE_EIS_4Stripe")
        assert campaign_spec_fingerprint(spec) == self.PINNED_RICH

    def test_the_defaulted_fields_contribute_no_key_at_all(self):
        """Omission is the mechanism — seven defaulted keys would rehash everything."""
        assert optimizer_tuning_identity(_spec()) is None

    @pytest.mark.parametrize("field,value", [
        ("decision_rtol", 0.25),
        ("exclusion_radius", 0.1),
        ("learned_feasibility", True),
        ("feasibility_strategy", "fwa"),
        ("feasibility_min_filter", False),
        ("feasibility_min_infeasible", 5),
        ("feasibility_min_feasible", 5),
    ])
    def test_a_field_that_is_set_does_contribute(self, field, value):
        """A campaign that sets one really is searching differently."""
        identity = optimizer_tuning_identity(_spec(**{field: value}))
        assert identity == {field: value}
        assert campaign_spec_fingerprint(_spec(**{field: value})) != \
            campaign_spec_fingerprint(_spec())

    def test_setting_a_field_to_its_shipped_default_stays_distinct_from_unset(self):
        """The sentinel's whole purpose: 'unset' and 'chose the default' differ."""
        assert campaign_spec_fingerprint(_spec(decision_rtol=0.0)) != \
            campaign_spec_fingerprint(_spec())

    def test_a_pre_t3_1_checkpoint_still_resumes(self):
        """A spec that never heard of these fields verifies against its own hash."""
        spec = _spec()
        assert spec.decision_rtol is SPEC_UNSET
        assert spec.learned_feasibility is SPEC_UNSET
        assert campaign_spec_fingerprint(spec) == campaign_spec_fingerprint(
            CampaignSpec(name="t", parameter_space=dict(SPACE)))


# ── Test 13 — T1.3 reachability, end to end ──────────────────────────────────

def _write(tmp_path, body: str):
    path = tmp_path / "campaign.toml"
    path.write_text(textwrap.dedent(f"""
        name = "reach"
        budget = 6

        [parameter_space.x]
        type = "float"
        low = 0.0
        high = 1.0
        {body}
    """), encoding="utf-8")
    return path


class TestTheT13KnobsNowReachTheOptimizer:
    def test_both_absent_gives_todays_exact_defaults(self, tmp_path, no_site_defaults):
        spec = load_campaign_spec(_write(tmp_path, ""))
        opt = build_optimizer(spec)
        assert opt._decision_rtol == 0.0
        assert opt._exclusion_radius is None
        assert opt.feasibility.config.enabled is False

    def test_a_campaign_file_field_reaches_the_optimizer(self, tmp_path, no_site_defaults):
        path = _write(tmp_path, "")
        path.write_text(path.read_text(encoding="utf-8")
                        .replace('budget = 6',
                                 'budget = 6\ndecision_rtol = 0.25\n'
                                 'exclusion_radius = 0.05'),
                        encoding="utf-8")
        opt = build_optimizer(load_campaign_spec(path))
        assert opt._decision_rtol == 0.25
        assert opt._exclusion_radius == 0.05

    def test_a_toml_site_default_reaches_the_optimizer(self, monkeypatch, tmp_path):
        _site(monkeypatch, optimizer={"decision_rtol": 0.4,
                                      "exclusion_radius": 0.2})
        opt = build_optimizer(load_campaign_spec(_write(tmp_path, "")))
        assert opt._decision_rtol == 0.4
        assert opt._exclusion_radius == 0.2

    def test_the_spec_beats_the_site_default(self, monkeypatch, tmp_path):
        _site(monkeypatch, optimizer={"decision_rtol": 0.4})
        path = _write(tmp_path, "")
        path.write_text(path.read_text(encoding="utf-8")
                        .replace('budget = 6', 'budget = 6\ndecision_rtol = 0.9'),
                        encoding="utf-8")
        assert build_optimizer(load_campaign_spec(path))._decision_rtol == 0.9

    def test_a_spec_pinning_the_shipped_default_overrides_a_site_default(
            self, monkeypatch, tmp_path):
        """Sentinel, not value, is what 'explicitly set' means."""
        _site(monkeypatch, optimizer={"decision_rtol": 0.4})
        path = _write(tmp_path, "")
        path.write_text(path.read_text(encoding="utf-8")
                        .replace('budget = 6', 'budget = 6\ndecision_rtol = 0.0'),
                        encoding="utf-8")
        assert build_optimizer(load_campaign_spec(path))._decision_rtol == 0.0


class TestTheFeasibilityKnobsResolveTheSameWay:
    def test_the_shipped_config_leaves_the_feature_off(self, tmp_path):
        """The real softae_config.toml, unmocked: default is OFF everywhere."""
        opt = build_optimizer(load_campaign_spec(_write(tmp_path, "")))
        assert opt.feasibility.config.enabled is False

    def test_a_site_default_can_enable_it(self, monkeypatch, tmp_path):
        _site(monkeypatch, feasibility={"enabled": True, "min_infeasible": 5})
        cfg = build_optimizer(load_campaign_spec(_write(tmp_path, ""))).feasibility.config
        assert cfg.enabled is True
        assert cfg.min_infeasible == 5
        assert cfg.clamp == pytest.approx(0.05 ** (1 / 5))

    def test_a_campaign_field_overrides_the_site_default(self, monkeypatch, tmp_path):
        _site(monkeypatch, feasibility={"enabled": True})
        path = _write(tmp_path, "")
        path.write_text(path.read_text(encoding="utf-8")
                        .replace('budget = 6',
                                 'budget = 6\nlearned_feasibility = false'),
                        encoding="utf-8")
        opt = build_optimizer(load_campaign_spec(path))
        assert opt.feasibility.config.enabled is False

    def test_a_sub_floor_value_is_refused_where_it_is_written(self, no_site_defaults):
        with pytest.raises(OptimizerError, match="UPWARD only"):
            resolve_optimizer_tuning(_spec(feasibility_min_infeasible=2))

    def test_a_deferred_strategy_refuses_at_resolution_time(self, no_site_defaults):
        with pytest.raises(OptimizerError, match="DEFERRED"):
            resolve_optimizer_tuning(_spec(feasibility_strategy="fia"))

    def test_build_optimizer_is_the_only_reader(self):
        """No other module reads the two config sections (§8's coherence rule)."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "softae"
        readers = []
        for path in root.rglob("*.py"):
            if path.name in ("loader.py", "autonomous_wiring.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "optimizer_tuning(" in text or "feasibility_config(" in text:
                readers.append(str(path))
        assert readers == []


class TestTheNewFieldsRoundTripThroughASpecFile:
    def test_an_unset_field_is_not_written(self):
        assert "decision_rtol" not in spec_to_dict(_spec())
        assert "learned_feasibility" not in spec_to_dict(_spec())

    def test_a_set_field_round_trips(self):
        out = spec_to_dict(_spec(decision_rtol=0.3, learned_feasibility=True))
        assert out["decision_rtol"] == 0.3
        assert out["learned_feasibility"] is True
        back = spec_from_dict(out)
        assert back.decision_rtol == 0.3
        assert back.learned_feasibility is True

    def test_the_fields_are_accepted_by_name_rather_than_rejected_as_unknown(self):
        spec = spec_from_dict({
            "name": "t", "parameter_space": dict(SPACE),
            "feasibility_min_infeasible": 4,
        })
        assert spec.feasibility_min_infeasible == 4
