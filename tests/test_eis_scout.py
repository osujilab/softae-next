"""Where the next sweep should put its points — the decision, not the detection.

The central claim under test is a *negative* one: ``scout.py`` contains no extremum
search of its own. This tree already holds three of those over EIS arrays, and the
one that admitted its own window edge produced a σ wrong by more than 10×, twice,
into a published comparison. So the AST guard below is not stylistic — it is the
assertion that a fourth was not quietly added, and it is written the way
``test_eis_universal_fit_route.py`` writes its completeness guards.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from softae.analysis.eis import scout
from softae.analysis.eis.scout import (
    DEFAULT_APEX_PROMINENCE_MIN,
    DEFAULT_BAND_BELOW_APEX_MIN_DECADES,
    SCOUT_VERDICTS,
    ScoutSettings,
    SegmentLayout,
    plan_for,
    plan_segments,
    scout_decision,
    scout_settings,
)
from softae.drivers.mscr_library import resolve_segments

# The arc+tail generator lives beside the tests that pin `arc.py`'s new fields;
# reusing it here keeps one definition of "what a blocking tail looks like" rather
# than two that can drift apart.
from tests.test_eis_arc_closure import arc_with_blocking_tail, semicircle

SCOUT_PATH = Path(scout.__file__)


class TestTheScoutContainsNoDetector:
    """P1 — §5's central claim, asserted rather than asserted-in-a-docstring."""

    @staticmethod
    def _tree() -> ast.Module:
        # utf-8-sig: at least one module in this tree is BOM-prefixed and ast.parse
        # refuses those.
        return ast.parse(SCOUT_PATH.read_text(encoding="utf-8-sig"),
                         filename=str(SCOUT_PATH))

    def test_scout_module_imports_no_scipy(self):
        offences = []
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                offences += [a.name for a in node.names if a.name.startswith("scipy")]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "scipy"):
                offences.append(node.module)
        assert offences == [], f"scout.py grew a scipy dependency: {offences}"

    def test_scout_module_defines_no_extremum_search(self):
        forbidden = {"argmax", "argmin", "find_peaks", "peak_prominences",
                     "_peak_prominence", "_interior_apex"}
        offences = []
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Name) and node.id in forbidden:
                offences.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in forbidden:
                offences.append(node.attr)
            elif (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name in forbidden):
                offences.append(node.name)
        assert offences == [], (
            f"scout.py is searching for extrema itself: {sorted(set(offences))}. "
            "arc.py already does this, on sorted arrays, validated on 1440 spectra."
        )

    def test_the_scout_gets_its_apex_from_arc_closure(self):
        imported = {n.module for n in ast.walk(self._tree())
                    if isinstance(n, ast.ImportFrom)}
        assert "softae.analysis.eis.arc" in imported


class TestTheDecision:
    def test_a_closed_arc_with_band_below_it_is_plannable(self):
        decision = scout_decision(*arc_with_blocking_tail())
        assert decision.verdict == "ok"
        assert decision.f_apex_hz == pytest.approx(1.0e3, rel=0.15)
        assert decision.band_below_apex_decades > 1.0
        assert decision.plannable

    def test_an_open_arc_with_no_interior_peak_asks_for_more_low_end(self):
        decision = scout_decision(*semicircle(f_peak=2.0, f_lo=20.0))
        assert decision.verdict == "extend_low"
        assert decision.arc_state == "open"

    def test_an_apex_too_close_to_the_floor_asks_for_more_low_end(self):
        # The apex is observed and prominent; there is simply not enough band under
        # it to fit without extrapolating. 0.67 decades, against a 1.0 cut.
        decision = scout_decision(*semicircle(f_peak=30.0, f_lo=6.475))
        assert decision.verdict == "extend_low"
        assert 0.0 < decision.band_below_apex_decades < 1.0

    def test_a_weak_shoulder_is_not_an_arc(self):
        f = np.logspace(5, 1, 7)                       # descending, as the rig sweeps
        y = np.array([20.0, 4.0, 3.02, 3.05, 3.0, 2.0, 1.0])
        decision = scout_decision(f, y)
        assert decision.verdict == "no_arc"
        assert decision.apex_prominence_rel < DEFAULT_APEX_PROMINENCE_MIN

    def test_an_apex_at_the_top_of_the_band_asks_for_more_high_end(self):
        f = np.logspace(5, 1, 6)
        y = np.array([3.0, 9.0, 4.0, 3.0, 2.0, 1.0])   # peak one step below f_high
        assert scout_decision(f, y).verdict == "extend_high"

    def test_a_closed_verdict_with_no_interior_peak_reports_no_arc(self):
        # The lone-excursion rescue: `arc_closure` called it CLOSED off a shape test
        # at the sweep floor, so there is a verdict but no shape to plan around.
        f = np.logspace(5, 1, 5)
        y = np.array([4.0, 3.0, 2.0, 1.0, 10.0])       # the 10 is the floor spike
        decision = scout_decision(f, y)
        assert decision.arc_state == "closed"
        assert decision.verdict == "no_arc"

    def test_an_undecidable_spectrum_reports_no_data(self):
        f, y = semicircle(npts=4)
        decision = scout_decision(f, y)
        assert decision.verdict == "no_data"
        assert decision.arc_state == "unknown"

    def test_the_apex_is_withheld_wherever_none_was_observed(self):
        # Structural, not cosmetic: it makes planning a sweep around an
        # unobserved or unqualified apex impossible rather than merely discouraged.
        # An open arc with no interior maximum has nothing to carry.
        decision = scout_decision(*semicircle(f_peak=2.0, f_lo=20.0))
        assert decision.verdict == "extend_low"
        assert decision.f_apex_hz != decision.f_apex_hz          # NaN
        assert not decision.plannable
        assert plan_for(decision) == ()

    def test_the_apex_is_withheld_when_it_fails_the_prominence_cut(self):
        f = np.logspace(5, 1, 7)
        y = np.array([20.0, 4.0, 3.02, 3.05, 3.0, 2.0, 1.0])
        decision = scout_decision(f, y)
        assert decision.verdict == "no_arc"
        assert decision.f_apex_hz != decision.f_apex_hz          # NaN
        assert plan_for(decision) == ()

    def test_an_observed_apex_is_carried_even_when_the_band_is_short(self):
        # The one non-`ok` verdict that carries an apex, and the reason it may:
        # `arc_closure` measured this maximum and it cleared the prominence cut.
        # The sweep simply stopped too soon under it — which is the case a
        # piecewise grid answers better than any preset, because the band to
        # cover is known rather than guessed at.
        decision = scout_decision(*semicircle(f_peak=30.0, f_lo=6.475))
        assert decision.verdict == "extend_low"
        assert decision.f_apex_hz == pytest.approx(30.0, rel=0.3)
        assert decision.plannable
        assert len(plan_for(decision)) >= 2

    def test_a_planned_band_always_reaches_below_the_sweep_that_asked_for_it(self):
        # Why the previous test cannot widen into a *narrower* sweep by accident:
        # `extend_low` means the apex sits under one decade of band, and the plan
        # puts its floor a decade under the apex, so with the shipped pairing of
        # those two the plan's floor is always beneath the old one.
        f, y = semicircle(f_peak=30.0, f_lo=6.475)
        plan = plan_for(scout_decision(f, y))
        assert plan[-1][1] < float(np.min(f))

    def test_every_verdict_is_one_of_the_declared_five(self):
        cases = [arc_with_blocking_tail(), semicircle(), semicircle(npts=4),
                 semicircle(f_peak=2.0, f_lo=20.0), semicircle(f_peak=30.0, f_lo=6.475)]
        assert {scout_decision(f, y).verdict for f, y in cases} <= set(SCOUT_VERDICTS)

    def test_the_decision_never_raises(self, monkeypatch):
        """It is called from a measurement path. A planner that can crash a running
        sweep is worse than one that declines to plan."""
        monkeypatch.setattr(scout, "arc_closure",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert scout_decision(*semicircle()).verdict == "no_data"

    def test_rubbish_input_declines_rather_than_raising(self):
        assert scout_decision(None, None).verdict == "no_data"
        assert scout_decision([], []).verdict == "no_data"


class TestThePlan:
    APEX = 1.0e3

    def test_the_dense_segment_straddles_the_apex(self):
        arc = plan_segments(self.APEX)[1]
        assert arc[0] > self.APEX > arc[1]

    def test_the_plan_is_descending_and_free_of_shared_boundaries(self):
        plan = plan_segments(self.APEX)
        resolved = resolve_segments(plan)                # must not raise
        assert [s[0] for s in plan] == sorted((s[0] for s in plan), reverse=True)
        for previous, following in zip(resolved, resolved[1:]):
            assert following[0] < previous[1]

    def test_the_tail_is_omitted_unless_it_is_asked_for(self):
        assert len(plan_segments(self.APEX)) == 2
        assert len(plan_segments(
            self.APEX, SegmentLayout(tail_points=5, tail_decades=1.0))) == 3

    def test_points_scale_with_the_span_not_with_a_fixed_count(self):
        """A fixed arc count made a 65 Hz apex 1.7× dearer than the preset it
        replaced, by pushing points below 20 Hz at ~1.7 s each."""
        two = plan_segments(self.APEX, SegmentLayout())[1][2]
        one = plan_segments(self.APEX, SegmentLayout(arc_decades_below=0.0))[1][2]
        assert two == 24                                  # 12/decade over 2 decades
        assert one == pytest.approx(two / 2, abs=1)

    def test_the_plan_is_clamped_to_what_the_instrument_can_reach(self):
        layout = SegmentLayout(f_hi_hz=200_000.0, f_floor_hz=0.016)
        for apex in (0.05, 1.0e3, 1.0e5):
            for f_start, f_end, _ in plan_segments(apex, layout):
                assert layout.f_floor_hz <= f_end < f_start <= layout.f_hi_hz

    def test_a_band_clamped_out_of_existence_is_dropped_not_emitted_degenerate(self):
        # An apex at the top of the band leaves the HF limb nowhere to go.
        plan = plan_segments(200_000.0, SegmentLayout())
        assert len(plan) == 1
        assert plan[0][0] == 200_000.0

    def test_an_impossible_apex_produces_no_plan(self):
        assert plan_segments(float("nan")) == ()
        assert plan_segments(0.0) == ()

    def test_a_time_cap_trims_the_arc_band_and_says_so(self, monkeypatch):
        events = []
        monkeypatch.setattr(scout, "logger", _Recorder(events))
        capped = plan_segments(self.APEX, SegmentLayout(max_total_s=3.0))
        uncapped = plan_segments(self.APEX, SegmentLayout())
        assert capped[1][1] > uncapped[1][1]              # narrower at the low end
        trimmed = [kw for name, kw in events if name == "eis_scout_plan_trimmed"]
        assert trimmed, "a silently narrowed plan is the failure this logs against"
        assert trimmed[0]["requested_decades_below"] == 1.0
        assert trimmed[0]["granted_decades_below"] < 1.0

    def test_an_uncapped_plan_is_never_trimmed(self, monkeypatch):
        events = []
        monkeypatch.setattr(scout, "logger", _Recorder(events))
        plan_segments(self.APEX, SegmentLayout(max_total_s=0.0))
        assert events == []

    def test_planning_is_pure_and_reads_no_config(self, monkeypatch):
        import softae.config.loader as loader

        monkeypatch.setattr(loader, "load", _explode)
        layout = SegmentLayout()
        assert plan_segments(self.APEX, layout) == plan_segments(self.APEX, layout)
        assert layout == SegmentLayout()


class TestSettings:
    def test_the_scout_ships_inert(self):
        """All three switches ship off, so no call site can change a sweep."""
        settings = scout_settings()
        assert settings.enabled is False
        assert settings.actuate is False
        assert settings.actuate_manual is False

    def test_the_manual_default_is_its_own_key_and_not_the_global_one(self):
        # The Manual Control tab is where uncharacterised samples turn up, and
        # the planner assumes a single arc. A deployment-wide `actuate` must not
        # be able to start planning around whichever arc happens to be tallest.
        settings = scout_settings({"scout": {"actuate": True}})
        assert settings.actuate is True
        assert settings.actuate_manual is False

    def test_the_band_cut_is_one_decade_not_zero_point_four(self):
        # 0.4 decades is 2.5×, and the 60.9 % median R1 overestimate was measured at
        # 1.5× = 0.176 decades — so 0.4 sits inside the regime that measurement
        # condemns. The artifact's own selection rule already used 1.0.
        assert DEFAULT_BAND_BELOW_APEX_MIN_DECADES == 1.0
        assert scout_settings().band_below_apex_min_decades == 1.0

    def test_the_instrument_block_is_the_authority_on_the_band_edges(self):
        settings = scout_settings({"instrument": {"f_max_hz": 12_345.0,
                                                  "f_min_hz": 0.5}})
        assert settings.layout.f_hi_hz == 12_345.0
        assert settings.layout.f_floor_hz == 0.5

    def test_unparseable_values_fall_back_without_raising(self):
        settings = scout_settings({"scout": {"band_below_apex_min_decades": "wide",
                                             "hf_points": "lots",
                                             "apex_prominence_min": None}})
        assert settings.band_below_apex_min_decades == 1.0
        assert settings.apex_prominence_min == DEFAULT_APEX_PROMINENCE_MIN
        assert settings.layout.hf_points == 10

    def test_unknown_keys_are_ignored(self):
        assert scout_settings({"scout": {"enabled": True, "nonsense": 3}}).enabled

    def test_the_cut_is_the_settings_cut_and_not_a_constant(self):
        # Calibration (spec §10) moves this by replay, so a threshold baked past the
        # settings object would be a threshold nobody could recalibrate.
        f, y = semicircle(f_peak=30.0, f_lo=6.475)
        loose = ScoutSettings(band_below_apex_min_decades=0.1)
        assert scout_decision(f, y).verdict == "extend_low"
        assert scout_decision(f, y, settings=loose).verdict == "ok"

    def test_describe_says_which_of_the_three_states_is_live(self):
        assert "off" in ScoutSettings().describe()
        assert "observing" in ScoutSettings(enabled=True).describe()
        assert "planning" in ScoutSettings(enabled=True, actuate=True).describe()


class _Recorder:
    """A structlog stand-in that keeps the events rather than rendering them."""

    def __init__(self, events: list) -> None:
        self.events = events

    def warning(self, event: str, **kw) -> None:
        self.events.append((event, kw))

    info = debug = error = warning


def _explode(*_a, **_kw):
    raise AssertionError("plan_segments must not read config")
