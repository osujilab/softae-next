"""The ``[[run_plan.phases]]`` codec — :mod:`softae.core.campaign_spec_run_plan`.

**What this exists to stop.** ``run_plan`` was refused by the spec loader
outright, so no file-driven campaign could describe an anneal or an equilibrate
phase and every TOML campaign ran pointwise formulate→measure. The codec lifts
that, and the risk it introduces is the opposite one: a phase table that decodes
into *almost* the plan the file describes. Hence the shape of this file — one
round-trip class that proves nothing is lost, and one refusal class per thing the
decoder must never guess.

The decoder refusals are the substance. ``scope`` in particular is required on
every phase rather than inferred, because per-sample and per-batch name two
different physical processes: cure each well as it is cast, or cure the plate
once. A default there would silently run the one nobody wrote down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from softae.core.campaign_spec_io import (
    SpecLoadError,
    spec_from_dict,
    spec_to_dict,
    spec_toml_completeness,
)
from softae.core.campaign_spec_run_plan import (
    baseline_conditions_codec,
    decode_baseline_conditions,
    decode_run_plan,
    encode_baseline_conditions,
    encode_run_plan,
    field_codec,
)
from softae.core.measurement_spec import MeasurementSpec
from softae.core.phase_setpoints import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_RH_APPROACH_TIMEOUT_S,
    PhaseSetpoints,
)
from softae.core.run_plan import (
    PhaseKind,
    PhaseScope,
    RunPhase,
    RunPlan,
    SettlePlan,
)

BENCH_INSTANCE = (Path(__file__).resolve().parents[1]
                  / "examples" / "bench_instance.toml")

#: A legacy volume-mode spec carrying no run plan — the control for "nothing
#: changed for the common case".
MINIMAL = {
    "name": "c",
    "parameter_space": {"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
}

FORMULATE = {"kind": "formulate", "scope": "per_sample"}


def _phases(*extra: dict) -> dict:
    """A ``[run_plan]`` table: the mandatory FORMULATE phase, plus *extra*."""
    return {"phases": [dict(FORMULATE), *[dict(p) for p in extra]]}


def _spec(*extra: dict):
    """A loaded spec whose run plan is FORMULATE plus *extra*."""
    return spec_from_dict({**MINIMAL, "run_plan": _phases(*extra)})


SETTLE_TABLE = {"round_period_s": 240.0, "min_hold_s": 1500.0,
                "max_hold_s": 14400.0}
EQUILIBRATE = {"kind": "equilibrate", "scope": "per_batch",
               "settle": dict(SETTLE_TABLE)}


# ── Round trip ───────────────────────────────────────────────────────────────

class TestRoundTrip:
    """Write → read → the same plan. Anything lost here runs a different run."""

    def test_round_trip_four_phase_batch_plan_is_identical(self):
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
                     conditions=PhaseSetpoints("casting", 25.0, 40.0)),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h",
                     conditions=PhaseSetpoints("anneal", 85.0, 20.0,
                                               rh_approach_timeout_s=14400.0)),
            RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                     settle=SettlePlan(240.0, 1500.0, 14400.0),
                     conditions=PhaseSetpoints("equilibrate", 25.0, 50.0)),
            RunPhase(PhaseKind.MEASURE, PhaseScope.PER_BATCH,
                     measurement=MeasurementSpec(preset="Extended")),
        ))

        assert decode_run_plan(encode_run_plan(plan)) == plan

    def test_round_trip_undriven_axis_returns_none_not_a_setpoint(self):
        """THE ONE THAT MATTERS. Absence is how "do not drive it" is spelled."""
        plan = RunPlan((RunPhase(
            PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
            conditions=PhaseSetpoints("casting", temp_setpoint_C=25.0)),))

        table = encode_run_plan(plan)
        back = decode_run_plan(table)

        assert "rh_setpoint_pct" not in table["phases"][0]["conditions"]
        assert back.phases[0].conditions.rh_setpoint_pct is None
        assert back.phases[0].conditions.drives_humidity is False
        assert back == plan

    def test_round_trip_a_disabled_rh_stability_gate_stays_disabled(self):
        """``None`` switches the gate off; an omitted key switches it back ON."""
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                     settle=SettlePlan(240.0, 1500.0, 14400.0,
                                       rh_stability_pct=None)),
        ))

        table = encode_run_plan(plan)

        assert table["phases"][1]["settle"]["explicit_none"] == \
            ["rh_stability_pct"]
        assert decode_run_plan(table).phases[1].settle.rh_stability_pct is None

    def test_round_trip_carries_the_settle_criterion_and_band(self):
        """The two halves of a rate gate, across the file boundary."""
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                     settle=SettlePlan(240.0, 1500.0, 14400.0,
                                       criterion="rate",
                                       rate_tol_dec_per_h=0.05)),
        ))

        table = encode_run_plan(plan)
        settle = table["phases"][1]["settle"]

        assert settle["criterion"] == "rate"
        assert settle["rate_tol_dec_per_h"] == 0.05
        assert decode_run_plan(table) == plan

    def test_encoder_omits_a_default_criterion_and_absent_band(self):
        """A deviation plan writes neither key — and no ``explicit_none``.

        The band's own default is ``None``, so absence already says it. Listing
        it would put a redundant ``explicit_none`` in every file this encoder
        touches, which is the opposite of the rule that array exists for.
        """
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.EQUILIBRATE, PhaseScope.PER_BATCH,
                     settle=SettlePlan(240.0, 1500.0, 14400.0)),
        ))

        settle = encode_run_plan(plan)["phases"][1]["settle"]

        assert "criterion" not in settle
        assert "rate_tol_dec_per_h" not in settle
        assert "explicit_none" not in settle

    def test_round_trip_default_approach_timeouts_are_not_written(self):
        """A file shows what was chosen; the defaults live in the dataclass."""
        plan = RunPlan((RunPhase(
            PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE,
            conditions=PhaseSetpoints("casting", 25.0, 40.0)),))

        table = encode_run_plan(plan)["phases"][0]["conditions"]

        assert "rh_approach_timeout_s" not in table
        assert decode_run_plan(encode_run_plan(plan)).phases[0].conditions \
            .rh_approach_timeout_s == DEFAULT_RH_APPROACH_TIMEOUT_S

    def test_spec_toml_completeness_reports_a_run_plan_as_encodable(self):
        spec = _spec(EQUILIBRATE, {"kind": "measure", "scope": "per_batch"})

        result = spec_toml_completeness(spec)

        assert result.complete, result.explain()
        assert "run_plan" not in result.missing
        assert spec_from_dict(spec_to_dict(spec)).run_plan == spec.run_plan

    def test_spec_without_a_run_plan_writes_exactly_what_it_wrote_before(self):
        """The positive control: nothing changed for the common case."""
        spec = spec_from_dict(MINIMAL)

        written = spec_to_dict(spec)

        assert written == MINIMAL
        assert spec.run_plan is None
        assert spec_toml_completeness(spec).complete


# ── Encoder: it may refuse, and it never raises ──────────────────────────────

class TestEncoderRefusals:

    def test_encode_a_non_run_plan_is_unrepresentable_not_an_exception(self):
        from softae.core.campaign_spec_fields import UNREPRESENTABLE

        assert encode_run_plan(object()) is UNREPRESENTABLE

    def test_encode_anneal_params_is_unrepresentable_rather_than_dropped(self):
        """A per-run task override has no key in the file shape."""
        from softae.core.campaign_spec_fields import UNREPRESENTABLE

        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"hold_time_s": 600}),
        ))

        assert encode_run_plan(plan) is UNREPRESENTABLE

    def test_an_unencodable_run_plan_is_reported_by_the_completeness_check(self):
        spec = spec_from_dict(MINIMAL)
        spec.run_plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"hold_time_s": 600}),
        ))

        result = spec_toml_completeness(spec)

        assert result.missing == ("run_plan",)
        assert "anneal parameter overrides" in result.explain()


# ── Decoder: it never guesses ────────────────────────────────────────────────

class TestDecoderRefusals:

    def test_decode_a_phase_without_scope_is_refused(self):
        """THE DEFECT THIS GUARDS. Inferring it changes which process runs."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL,
                            "run_plan": _phases({"kind": "anneal"})})

        assert "does not declare 'scope'" in str(exc.value)
        assert "per_batch" in str(exc.value)

    def test_decode_an_unknown_kind_is_refused_with_the_legal_values(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "bake", "scope": "per_batch"})})

        assert "unknown kind 'bake'" in str(exc.value)
        assert "'anneal'" in str(exc.value)

    def test_decode_an_unknown_scope_is_refused(self):
        with pytest.raises(SpecLoadError, match="unknown scope 'per_plate'"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "anneal", "scope": "per_plate"})})

    def test_decode_a_measurement_block_on_a_non_measure_phase_is_refused(self):
        """``RunPhase`` already refuses it; the codec surfaces its words."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "anneal", "scope": "per_batch",
                 "measurement": {"preset": "Extended"}})})

        assert "only MEASURE acquires data" in str(exc.value)

    def test_decode_an_anneal_task_on_a_phase_that_does_not_anneal_is_refused(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "measure", "scope": "per_batch",
                 "anneal_task": "anneal_85C_8h"})})

        assert "never reach the chamber" in str(exc.value)

    def test_decode_an_unknown_phase_key_is_refused_not_ignored(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "anneal", "scope": "per_batch", "hold_time_s": 28800})})

        assert "unknown key(s) ['hold_time_s']" in str(exc.value)

    def test_decode_an_unknown_top_level_run_plan_key_is_refused(self):
        with pytest.raises(SpecLoadError, match="carries only 'phases'"):
            spec_from_dict({**MINIMAL,
                            "run_plan": {"phases": [dict(FORMULATE)],
                                         "scope": "per_batch"}})

    def test_decode_a_run_plan_that_is_not_a_table_is_refused(self):
        with pytest.raises(SpecLoadError, match="expected a \\[run_plan\\] table"):
            spec_from_dict({**MINIMAL, "run_plan": "whatever"})

    def test_decode_an_empty_phase_array_is_refused(self):
        with pytest.raises(SpecLoadError, match="describes no run"):
            spec_from_dict({**MINIMAL, "run_plan": {"phases": []}})

    def test_decode_an_unknown_conditions_key_is_refused(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": {"phases": [
                {**FORMULATE, "conditions": {"name": "casting",
                                             "temp_setpoint_c": 25.0}}]}})

        assert "[conditions] has unknown key(s) ['temp_setpoint_c']" in str(exc.value)

    def test_decode_conditions_without_a_name_is_refused(self):
        with pytest.raises(SpecLoadError, match="needs a 'name'"):
            spec_from_dict({**MINIMAL, "run_plan": {"phases": [
                {**FORMULATE, "conditions": {"temp_setpoint_C": 25.0}}]}})

    def test_decode_a_settle_window_missing_a_duration_is_refused(self):
        """None of the three has a safe default — min_hold_s is the cure."""
        table = {k: v for k, v in SETTLE_TABLE.items() if k != "min_hold_s"}

        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE, "settle": table})})

        assert "missing ['min_hold_s']" in str(exc.value)

    def test_decode_a_required_duration_listed_as_nothing_is_refused(self):
        """``explicit_none`` says "set to nothing"; a floor cannot be nothing."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE,
                            "explicit_none": ["min_hold_s"]}})})

        assert "can be set to nothing" in str(exc.value)
        assert "'rh_stability_pct'" in str(exc.value)

    def test_decode_a_settle_field_given_a_value_and_listed_as_nothing_is_refused(self):
        with pytest.raises(SpecLoadError, match="says two things about one field"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "rh_stability_pct": 1.5,
                            "explicit_none": ["rh_stability_pct"]}})})

    def test_decode_an_unknown_criterion_word_is_refused(self):
        """SettlePlan owns the legal set; the codec surfaces its refusal."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "criterion": "slope",
                            "rate_tol_dec_per_h": 0.05}})})

        assert "[settle]" in str(exc.value)
        assert "'slope' is not one of" in str(exc.value)

    def test_decode_a_rate_criterion_without_a_band_is_refused(self):
        """The pair that computes no verdict, refused where the file is read."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "criterion": "rate"}})})

        assert "[settle]" in str(exc.value)
        assert "needs a rate_tol_dec_per_h band" in str(exc.value)

    def test_decode_a_rate_band_set_to_nothing_round_trips_as_no_band(self):
        """Value-preserving, not byte-symmetric — and that is the intent.

        ``explicit_none`` on a field whose default is already ``None`` is a
        permitted statement of intent. It decodes to ``None`` and the encoder
        does not write it back, so the plan survives and the redundancy does not.
        """
        spec = spec_from_dict({**MINIMAL, "run_plan": _phases(
            {**EQUILIBRATE,
             "settle": {**SETTLE_TABLE,
                        "explicit_none": ["rate_tol_dec_per_h"]}})})
        settle = spec.run_plan.phases[1].settle

        assert settle.rate_tol_dec_per_h is None
        written = encode_run_plan(spec.run_plan)["phases"][1]["settle"]
        assert "explicit_none" not in written
        assert decode_run_plan(encode_run_plan(spec.run_plan)) == spec.run_plan

    def test_decode_a_rate_band_given_a_value_and_listed_as_nothing_is_refused(self):
        with pytest.raises(SpecLoadError, match="says two things about one field"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "rate_tol_dec_per_h": 0.05,
                            "explicit_none": ["rate_tol_dec_per_h"]}})})

    def test_decode_an_unknown_settle_key_lists_the_new_keys(self):
        """What proves the two names actually joined the key set."""
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "rate_tol_dec_per_hr": 0.05}})})

        message = str(exc.value)
        assert "unknown key(s) ['rate_tol_dec_per_hr']" in message
        assert "'criterion'" in message and "'rate_tol_dec_per_h'" in message

    def test_decode_a_non_numeric_duration_is_refused(self):
        with pytest.raises(SpecLoadError, match="must be a number"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {**EQUILIBRATE,
                 "settle": {**SETTLE_TABLE, "min_hold_s": "twenty"}})})

    def test_decode_an_unknown_measurement_key_is_refused(self):
        with pytest.raises(SpecLoadError, match="unknown measurement key"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "measure", "scope": "per_batch",
                 "measurement": {"presset": "Extended"}})})

    def test_decode_a_plan_with_no_formulate_phase_surfaces_run_plans_refusal(self):
        with pytest.raises(SpecLoadError, match="must contain a FORMULATE phase"):
            spec_from_dict({**MINIMAL, "run_plan": {"phases": [
                {"kind": "measure", "scope": "per_batch"}]}})

    def test_decode_a_per_batch_formulate_phase_surfaces_run_plans_refusal(self):
        with pytest.raises(SpecLoadError, match="FORMULATE must be per-sample"):
            spec_from_dict({**MINIMAL, "run_plan": {"phases": [
                {"kind": "formulate", "scope": "per_batch"}]}})

    def test_decode_a_reserved_arrhenius_phase_is_refused(self):
        with pytest.raises(SpecLoadError, match="ARRHENIUS phase is reserved"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "arrhenius", "scope": "per_batch"})})


# ── One authority for the settle window ──────────────────────────────────────

class TestSettleSaidTwice:
    """A file can now spell settle two ways. It still may not spell it both."""

    def test_a_file_carrying_both_spellings_of_settle_is_refused(self):
        spec = spec_from_dict({**MINIMAL, "equilibration_method": "settle",
                               "round_period_s": 120.0, "min_hold_s": 600.0,
                               "max_hold_s": 7200.0,
                               "run_plan": _phases(EQUILIBRATE)})

        with pytest.raises(ValueError, match="say it once"):
            spec.settle_plan()

    def test_a_file_carrying_only_the_phase_spelling_resolves_to_that_plan(self):
        """The control: the refusal must not fire on the one legal spelling."""
        spec = spec_from_dict({**MINIMAL, "run_plan": _phases(EQUILIBRATE)})

        assert spec.settle_plan() == SettlePlan(240.0, 1500.0, 14400.0)


# ── The worked example ───────────────────────────────────────────────────────

class TestBenchInstanceRunPlan:
    """``examples/bench_instance.toml`` is the file the bench run starts from.

    Loaded through the real loader on purpose — the example's whole job is to be
    the thing an operator actually runs.
    """

    @pytest.fixture(scope="class")
    def spec(self):
        from softae.core.campaign_spec_io import load_campaign_spec

        return load_campaign_spec(BENCH_INSTANCE)

    @pytest.fixture(scope="class")
    def plan(self, spec):
        return spec.run_plan

    def test_bench_instance_carries_the_four_phase_arc_in_order(self, plan):
        assert [p.kind for p in plan.phases] == [
            PhaseKind.FORMULATE, PhaseKind.ANNEAL,
            PhaseKind.EQUILIBRATE, PhaseKind.MEASURE]

    def test_bench_instance_casts_per_sample_and_cures_the_batch_once(self, plan):
        """Cast every well, then anneal the plate — not four cure cycles."""
        assert plan.phases[0].scope is PhaseScope.PER_SAMPLE
        assert all(p.scope is PhaseScope.PER_BATCH for p in plan.phases[1:])
        assert [scope for scope, _ in plan.segments()] == [
            PhaseScope.PER_SAMPLE, PhaseScope.PER_BATCH]

    def test_bench_instance_describes_every_phase_with_its_conditions(self, plan):
        line = plan.describe()

        assert line.index("Formulate") < line.index("Anneal") \
            < line.index("Equilibrate") < line.index("Measure")
        assert "casting (25 °C, 22 %RH) [per sample]" in line
        assert "anneal (25 °C, 22 %RH) [per batch]" in line
        assert "Measure EIS (Extended) [per batch]" in line

    def test_bench_instance_allows_four_hours_to_reach_the_anneal_humidity(
        self, plan
    ):
        """A CEILING, not a budget — and since the 2026-09-19/20 ruling put
        casting and the anneal rest state both at 22 %RH, no descent is
        commanded here at all. The old reading (1 800 s covers ~2.7 of an
        18 %RH descent) described a 40 → 20 step this file no longer takes;
        4 h stays because `rh_wait` raises on timeout and an enclosure that
        has drifted and cannot come back must not enter an 8 h cure.
        """
        assert plan.phases[1].conditions.rh_approach_timeout_s == 14400.0
        assert plan.phases[1].anneal_task == "anneal_85C_8h"

    def test_bench_instance_settle_window_is_the_campaigns_only_one(self, plan):
        assert plan.phases[2].settle == SettlePlan(240.0, 1500.0, 14400.0)

    def test_bench_instance_rests_below_the_cure_and_budgets_the_cool_down(self, plan):
        """Heater-only stage: the rest state is 25 °C and the descent gets its own ceiling."""
        anneal, equilibrate = plan.phases[1], plan.phases[2]
        assert anneal.conditions.temp_setpoint_C == 25.0
        assert equilibrate.conditions.temp_setpoint_C == anneal.conditions.temp_setpoint_C
        assert equilibrate.conditions.approach_timeout_s > DEFAULT_APPROACH_TIMEOUT_S

    def test_bench_instance_declares_a_pump_per_stock(self, spec):
        """`pump_ids` defaults to two and `deposition_settings` only TRIMS it.

        A third stock with no third id reaches the liquid handler as
        `ids/vols/deadvols/disp_rates length mismatch: 2/3/2/2` — at the first
        deposit, after the startup flush has already pushed fluid.
        """
        stocks = spec.general_formulation.stocks

        assert len(spec.pump_ids) == len(stocks)
        assert set(spec.pump_ids) == set(
            spec.general_formulation.pump_assignment.values())


# ── Registration ─────────────────────────────────────────────────────────────

def test_the_run_plan_codec_is_registered_in_the_shared_field_table():
    """``spec_from_dict`` reaches it through ``OBJECT_FIELDS``, not a merge."""
    from softae.core.campaign_spec_fields import OBJECT_FIELDS

    assert OBJECT_FIELDS["run_plan"] is field_codec()
    assert "run_plan" not in __import__(
        "softae.core.campaign_spec_io", fromlist=["_UNSUPPORTED"])._UNSUPPORTED


# ── The cure's duration crosses the boundary ─────────────────────────────────

class TestHoldS:
    """``hold_s`` is the file's one per-run override of a catalog anneal task."""

    def test_round_trip_hold_s_on_an_anneal_phase(self):
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_task="anneal_85C_8h", hold_s=28800.0),
        ))

        table = encode_run_plan(plan)

        assert table["phases"][1]["hold_s"] == 28800.0
        assert decode_run_plan(table) == plan

    def test_a_phase_without_a_hold_writes_no_key(self):
        """Absence is absence: the task's own hold stands."""
        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH),
        ))

        assert "hold_s" not in encode_run_plan(plan)["phases"][1]
        assert decode_run_plan(encode_run_plan(plan)).phases[1].hold_s is None

    def test_decode_a_hold_on_a_phase_that_does_not_anneal_is_refused(self):
        with pytest.raises(SpecLoadError) as exc:
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "measure", "scope": "per_batch", "hold_s": 28800.0})})

        assert "never reach the chamber" in str(exc.value)

    def test_decode_a_non_numeric_hold_is_refused(self):
        with pytest.raises(SpecLoadError, match="'hold_s' must be a number"):
            spec_from_dict({**MINIMAL, "run_plan": _phases(
                {"kind": "anneal", "scope": "per_batch", "hold_s": "8h"})})

    def test_decode_a_hold_said_twice_surfaces_run_phases_refusal(self):
        """The file can only spell it once, so the second spelling is Python's.

        A spec built in memory can still carry both; the codec surfaces
        ``RunPhase``'s words rather than restating them.
        """
        with pytest.raises(ValueError, match="say it once"):
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"hold_time_s": 600}, hold_s=28800.0)

    def test_anneal_params_stays_unrepresentable_beside_a_hold(self):
        """``hold_s`` is typed; the free dict it replaces is still unwritable."""
        from softae.core.campaign_spec_fields import UNREPRESENTABLE

        plan = RunPlan((
            RunPhase(PhaseKind.FORMULATE, PhaseScope.PER_SAMPLE),
            RunPhase(PhaseKind.ANNEAL, PhaseScope.PER_BATCH,
                     anneal_params={"target_temp_C": 85}, hold_s=28800.0),
        ))

        assert encode_run_plan(plan) is UNREPRESENTABLE

    def test_a_loaded_file_carries_the_hold_all_the_way_to_the_plan(self):
        """Through the real loader — the file shape is the point."""
        spec = spec_from_dict({**MINIMAL, "run_plan": _phases(
            {"kind": "anneal", "scope": "per_batch",
             "anneal_task": "anneal_85C_8h", "hold_s": 28800.0})})

        assert spec.run_plan.phases[1].hold_s == 28800.0


# ─────────────────────── the campaign-level [conditions] baseline (T11.28) ───────────────────────

class TestTheBaselineConditionsCodec:
    """A ``[conditions]`` table at the TOP level, not on a phase.

    Same type, same two axes, same "an omitted axis is not driven" rule. What it
    cannot carry is the approach bands and timeouts, and that is the point of
    the block: the baseline is commanded at the top of the run and waited for by
    NOTHING, so a tolerance or a timeout there would be a number with no reader.
    A number that does nothing reads as a gate that does not exist.
    """

    def test_baseline_conditions_round_trip_preserves_both_axes(self):
        table = {"name": "baseline", "temp_setpoint_C": 25.0,
                 "rh_setpoint_pct": 40.0}
        setpoints = decode_baseline_conditions(table)
        assert setpoints == PhaseSetpoints("baseline", temp_setpoint_C=25.0,
                                           rh_setpoint_pct=40.0)
        assert encode_baseline_conditions(setpoints) == table

    def test_baseline_conditions_an_omitted_axis_stays_undriven(self):
        setpoints = decode_baseline_conditions({"name": "damp",
                                                "rh_setpoint_pct": 22.0})
        assert setpoints.temp_setpoint_C is None
        assert setpoints.drives_temperature is False
        assert "temp_setpoint_C" not in encode_baseline_conditions(setpoints)

    def test_baseline_conditions_rh_zero_is_a_commanded_dry_purge_not_absence(self):
        """Section 4's direction rule, at the file boundary.

        ``ctrl`` near zero is dry air and ``ctrl == 0`` exactly shuts both
        Aalborg PSVs (bench-verified 2026-08-21), so ``0.0`` must survive the
        round trip as ``0.0`` and an omitted key as ``None``. A codec that
        collapsed them would turn "leave the axis alone" into a dry purge, which
        after a park is the state the chamber is already stuck in.
        """
        driven = decode_baseline_conditions({"name": "purge",
                                             "rh_setpoint_pct": 0.0})
        assert driven.rh_setpoint_pct == 0.0
        assert driven.drives_humidity is True
        assert encode_baseline_conditions(driven)["rh_setpoint_pct"] == 0.0

        quiet = decode_baseline_conditions({"name": "quiet"})
        assert quiet.rh_setpoint_pct is None
        assert quiet.drives_humidity is False

    @pytest.mark.parametrize("key, value", [
        ("approach_timeout_s", 10800.0),
        ("rh_approach_timeout_s", 14400.0),
        ("tolerance_C", 1.0),
        ("rh_tolerance_pct", 3.0),
    ])
    def test_baseline_conditions_refuses_an_approach_band_or_timeout(
        self, key, value
    ):
        with pytest.raises(ValueError) as exc:
            decode_baseline_conditions({"name": "baseline",
                                        "rh_setpoint_pct": 22.0, key: value})
        assert "NOTHING WAITS HERE" in str(exc.value)
        assert key in str(exc.value)

    def test_phase_conditions_still_accept_the_same_keys(self):
        """The control: the refusal is about WHERE the table sits, not the key.

        Without this the refusal above would pass just as well if the decoder
        had started rejecting the tuning keys everywhere.
        """
        plan = decode_run_plan({"phases": [
            {**FORMULATE, "conditions": {"name": "casting",
                                         "temp_setpoint_C": 25.0,
                                         "approach_timeout_s": 10800.0}}]})
        assert plan.phases[0].conditions.approach_timeout_s == 10800.0

    def test_baseline_conditions_without_a_name_is_refused(self):
        with pytest.raises(ValueError, match="name"):
            decode_baseline_conditions({"temp_setpoint_C": 25.0})

    def test_baseline_conditions_an_unknown_key_names_the_legal_ones(self):
        with pytest.raises(ValueError) as exc:
            decode_baseline_conditions({"name": "b", "temp_setpoint_c": 25.0})
        assert "unknown key(s) ['temp_setpoint_c']" in str(exc.value)

    def test_baseline_conditions_encode_refuses_a_tuned_setpoints(self):
        """Unrepresentable, rather than writing a file this decoder refuses.

        A ``PhaseSetpoints`` built in Python can carry a non-default timeout;
        the baseline block cannot spell one, so the encoder says so and
        ``spec_toml_completeness`` reports the field missing. Writing it would
        produce a file that raises on the next load.
        """
        from softae.core.campaign_spec_fields import UNREPRESENTABLE

        tuned = PhaseSetpoints("baseline", rh_setpoint_pct=22.0,
                               rh_approach_timeout_s=14400.0)
        assert encode_baseline_conditions(tuned) is UNREPRESENTABLE
        assert encode_baseline_conditions("not a PhaseSetpoints") is UNREPRESENTABLE

    def test_baseline_conditions_codec_is_the_registered_pair(self):
        codec = baseline_conditions_codec()
        assert codec is baseline_conditions_codec()
        assert (codec.encode, codec.decode) == (encode_baseline_conditions,
                                                decode_baseline_conditions)
        assert codec.why_not
