"""T3.1 §3 — which failures are evidence about a composition, and which are not.

Spec: ``docs/SubAgent docs/failure_informed_feasibility_spec.md`` §3.1-§3.3, §10
tests 2, 2a, 2b, 2c, 2d, 2e, 2f, 3.

Reports are built by driving the **real** quality gate over synthetic traces
rather than by hand-constructing ``QualityReport`` objects. The signature strings
(``"open circuit"``, ``"shorted or dead channel"``) are a coupling between
``analysis/quality.py`` and ``optimizers/failure_labels.py``, and a test that
spelled them itself would keep passing after the gate reworded them — which is
exactly when the labels would silently stop being issued.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.quality import Verdict, gate_raw_measurement, validate_raw_eis
from softae.core.autonomous_wiring import (
    confirmation_measure_step,
    confirmation_step_name,
    is_primary_measurement,
    measure_step_name,
)
from softae.drivers.mock_factory import create_mock_manager
from softae.optimizers.failure_labels import (
    CONFIRM_MEASUREMENT,
    OUTCOME_INFEASIBLE,
    OUTCOME_MEASURED,
    OUTCOME_UNKNOWN,
    FailureLabelEngine,
    LabelDecision,
    reject_signature,
)
from softae.optimizers.feasibility import (
    FEASIBLE,
    INFEASIBLE,
    FeasibilityConfig,
    FeasibilityModel,
)

N = 12
_F = np.logspace(5, -1, N)


def _trace(magnitude: float) -> np.ndarray:
    """A well-formed sweep whose |Z| sits at *magnitude* (varying, monotonic)."""
    mag = magnitude * np.linspace(1.0, 1.6, N)
    return np.column_stack([_F, mag * 0.8, mag * 0.6])


OPEN = _trace(1e13)      # |Z| median above max_abs_z -> "open circuit"
SHORT = _trace(1e-6)     # |Z| median below min_abs_z -> "shorted or dead channel"
GOOD = _trace(2.0e3)
EMPTY: np.ndarray = np.zeros((0, 3))


def _report(raw):
    """The gate's report — read as a *report*, never as a post-gate verdict."""
    return gate_raw_measurement(raw, config={"enabled": True})


def _engine(**kw):
    model = kw.pop("model", None)
    if model is None:
        model = FeasibilityModel(FeasibilityConfig(enabled=True))
    return FailureLabelEngine(model=model, **kw), model


PARAMS = {"x0": 0.3, "x1": 0.7}
OTHER = {"x0": 0.9, "x1": 0.1}


# ── Signature classification (§3 table) ──────────────────────────────────────

class TestOnlyOpenAndShortAreAboutTheFilm:
    def test_an_open_circuit_reject_is_recognised(self):
        assert reject_signature(_report(OPEN)) == "open"

    def test_a_shorted_channel_reject_is_recognised(self):
        assert reject_signature(_report(SHORT)) == "short"

    def test_a_healthy_trace_carries_no_signature(self):
        assert reject_signature(_report(GOOD)) is None

    @pytest.mark.parametrize("raw", [EMPTY, None, "not an array"])
    def test_instrument_side_rejects_carry_no_signature(self, raw):
        """Empty / unreadable says nothing about the film (quality.py:169-178)."""
        assert reject_signature(_report(raw)) is None

    def test_a_stuck_instrument_carries_no_signature(self):
        stuck = np.column_stack([_F, np.full(N, 500.0), np.full(N, 500.0)])
        report = validate_raw_eis(stuck)
        assert report.verdict is Verdict.REJECT
        assert reject_signature(report) is None

    def test_the_signature_is_read_from_the_report_not_the_verdict(self):
        """With [quality] enabled = false a REJECT arrives as SUSPECT.

        Keying on the verdict would make the richest label source inert exactly
        while the gate is being observed — spec §2 finding (2).
        """
        disabled = gate_raw_measurement(OPEN, config={"enabled": False})
        assert disabled.verdict is Verdict.SUSPECT
        assert "gate disabled" in disabled.issues
        assert reject_signature(disabled) == "open"


# ── Test 2 / 3 — one case per §3 row ─────────────────────────────────────────

class TestMostFailuresEarnNoLabel:
    def test_a_rig_exception_yields_no_label_and_an_unknown_outcome(self):
        engine, model = _engine()
        d = engine.record_rig_failure(
            params=PARAMS, what="execute: CommunicationError", channel=3)
        assert d.label is None and d.outcome == OUTCOME_UNKNOWN
        assert model.labels == []

    def test_a_run_of_pure_rig_faults_produces_zero_infeasible_labels(self):
        """3 — the NULL-objective discipline is untouched."""
        engine, model = _engine()
        for what in ("execute: StepTimeoutError", "execute: ConnectionError_",
                     "analyze: ValueError", "execute: VI_ERROR_TMO"):
            engine.record_rig_failure(params=PARAMS, what=what, channel=1)
        assert model.n_infeasible == 0
        assert model.n_feasible == 0

    def test_a_formulation_infeasible_error_is_labelled_but_excluded_from_training(self):
        engine, model = _engine()
        d = engine.record_formulation_infeasible(params=PARAMS, channel=1,
                                                 detail="over budget")
        assert d.outcome == OUTCOME_INFEASIBLE and d.label == INFEASIBLE
        assert d.train is False
        # Recorded for the audit trail; the hard feasibility_fn already enforces
        # this boundary exactly, so training on it is a wasted degree of freedom.
        assert model.labels == []

    def test_a_measured_trial_is_positive_evidence(self):
        engine, model = _engine()
        d = engine.record_measured(params=PARAMS, channel=1, board_id="b1",
                                   objective_value=1.2)
        assert d.outcome == OUTCOME_MEASURED and d.label == FEASIBLE
        assert model.n_feasible == 1

    def test_an_instrument_side_reject_yields_nothing(self):
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(EMPTY), confirmations=[_report(EMPTY)])
        assert d.label is None
        assert "not open-circuit or shorted" in d.failure_reason
        assert model.labels == []

    def test_there_is_no_hardware_suspect_outcome(self):
        """A channel-level suspicion is not a fact about this trial (§3)."""
        with pytest.raises(ValueError, match="not an outcome"):
            LabelDecision(outcome="hardware_suspect")


# ── Test 2a / 2b — the confirmation repeats ──────────────────────────────────

class TestAConfirmationSweepDecidesWhetherARejectIsEvidence:
    def test_a_repeat_that_disagrees_yields_no_label(self):
        """2a — a transient must cost nothing but one sweep."""
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN), confirmations=[_report(GOOD)])
        assert d.label is None and d.outcome == OUTCOME_UNKNOWN
        assert "did not reproduce" in d.failure_reason
        assert model.labels == []

    def test_a_repeat_with_a_different_signature_yields_no_label(self):
        """An open followed by a short did not reproduce — it changed."""
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN), confirmations=[_report(SHORT)])
        assert d.label is None
        assert model.labels == []

    def test_no_confirmation_at_all_yields_no_label(self):
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN))
        assert d.label is None
        assert "no confirmation sweep" in d.failure_reason

    def test_agreeing_repeats_with_a_board_accept_label_exactly_once(self):
        """2b — three rejects are one film, not three."""
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN),
            confirmations=[_report(OPEN), _report(OPEN)])
        assert d.outcome == OUTCOME_INFEASIBLE and d.label == INFEASIBLE
        assert d.train is True
        assert model.n_infeasible == 1
        assert "open x3 confirmed" in d.failure_reason

    def test_a_short_signature_labels_on_the_same_terms_as_an_open_one(self):
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b1")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(SHORT), confirmations=[_report(SHORT)])
        assert d.label == INFEASIBLE
        assert model.n_infeasible == 1

    def test_each_confirmation_is_reported_to_the_operator(self):
        events: list[tuple] = []
        engine, _ = _engine(emit=lambda name, **kw: events.append((name, kw)))
        engine.note_accept(channel=2, board_id="b1")
        engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN),
            confirmations=[_report(OPEN), _report(OPEN)])
        sweeps = [kw for name, kw in events if name == "confirmation_sweep"]
        assert [s["attempt"] for s in sweeps] == [1, 2]
        assert all(s["matched"] for s in sweeps)
        assert any(name == "infeasible_label_recorded" for name, _ in events)


# ── Test 2c — the repeats are never scored ───────────────────────────────────

class TestConfirmationStepsNeverEnterTheObjectivePath:
    def test_a_confirm_tagged_step_is_excluded_by_the_unmodified_predicate(self):
        """2c — T1.5's predicate does the work with no edit to it."""
        step = confirmation_measure_step(7, 1)
        assert step is not None
        assert step.tags["measurement"] == CONFIRM_MEASUREMENT
        assert is_primary_measurement(step.tags) is False

    def test_the_primary_step_it_repeats_is_still_selected(self):
        """The positive control: the predicate did not simply reject everything."""
        assert is_primary_measurement({"channel": "7", "measurement": "primary"})
        assert is_primary_measurement({"channel": "7"})

    def test_a_confirm_step_keeps_the_channel_tag_so_it_shares_the_sample_uuid(self):
        """One sample, several measurements — T2.6's grouping invariant."""
        step = confirmation_measure_step(7, 2)
        assert step.tags["channel"] == "7"
        assert step.tags["confirm_attempt"] == "2"

    def test_a_confirm_step_is_named_distinctly_from_the_primary(self):
        assert confirmation_step_name(7, 1) != measure_step_name(7)
        assert confirmation_measure_step(7, 1).name == confirmation_step_name(7, 1)

    def test_a_confirm_step_re_reads_the_same_well_with_the_same_parameters(self):
        """A repeat is a re-read, not a self-test: it changes no setting."""
        from softae.core.deposition_steps import eis_measure_step

        primary = eis_measure_step(7, name=measure_step_name(7))
        confirm = confirmation_measure_step(7, 1)
        assert confirm.params == primary.params
        assert confirm.instrument == primary.instrument
        assert confirm.method == primary.method

    def test_confirm_tagged_results_are_not_averaged_into_the_objective(self):
        from softae.core.autonomous_wiring import eis_impedance_objective

        step_results = {
            measure_step_name(7): [GOOD],
            confirmation_step_name(7, 1): [OPEN],
        }
        tags = {
            measure_step_name(7): {"channel": "7", "measurement": "primary"},
            confirmation_step_name(7, 1): {
                "channel": "7", "measurement": CONFIRM_MEASUREMENT},
        }
        both = eis_impedance_objective(step_results, PARAMS, step_tags=tags,
                                       kind="mean_abs_z")
        only = eis_impedance_objective(
            {measure_step_name(7): [GOOD]}, PARAMS,
            step_tags={measure_step_name(7): tags[measure_step_name(7)]},
            kind="mean_abs_z")
        assert both == only


# ── Test 2e — the board corroborator ─────────────────────────────────────────

class TestABoardOfRejectsIsNotABoardOfBadCompositions:
    def test_with_no_accept_anywhere_on_the_board_nothing_is_labelled(self):
        """2e — a run in which nothing measured must not read as bad chemistry."""
        engine, model = _engine()
        for ch in (1, 2, 3, 4):
            d = engine.record_gate_reject(
                params={"x0": 0.1 * ch, "x1": 0.5}, channel=ch, board_id="b1",
                primary_report=_report(OPEN), confirmations=[_report(OPEN)])
            assert d.label is None
            assert "no other channel on this board ACCEPTed" in d.failure_reason
        assert model.labels == []

    def test_a_channel_cannot_corroborate_itself(self):
        """With one measurement per film a failed channel has no ACCEPT of its own."""
        engine, _ = _engine()
        engine.note_accept(channel=1, board_id="b1")
        assert engine.board_has_accept("b1", excluding=1) is False
        engine.note_accept(channel=2, board_id="b1")
        assert engine.board_has_accept("b1", excluding=1) is True

    def test_an_accept_on_a_different_board_does_not_corroborate(self):
        engine, model = _engine()
        engine.note_accept(channel=2, board_id="b2")
        d = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        assert d.label is None
        assert model.labels == []


# ── Test 2d / 2f — channel-correlated rejects: report, retract, passive ──────

class TestAChannelCorrelatedRejectReportsAndRetracts:
    def _pattern(self, engine):
        """ch1 rejects on two boards under two unrelated compositions."""
        engine.note_accept(channel=2, board_id="b1")
        engine.note_accept(channel=2, board_id="b2")
        first = engine.record_gate_reject(
            params=PARAMS, channel=1, board_id="b1",
            primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        second = engine.record_gate_reject(
            params=OTHER, channel=1, board_id="b2",
            primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        return first, second

    def test_the_pattern_raises_one_alert_labels_nothing_further_and_retracts(self):
        """2d — including retroactively."""
        events: list[tuple] = []
        engine, model = _engine(emit=lambda name, **kw: events.append((name, kw)))
        first, second = self._pattern(engine)

        assert first.label == INFEASIBLE          # issued before the pattern emerged
        assert second.label is None               # withheld once it did
        # ...and the one already issued is dropped.
        assert model.n_infeasible == 0

        reports = [kw for name, kw in events if name == "channel_reject_pattern"]
        assert len(reports) == 1
        assert reports[0]["labels_retracted"] == 1
        assert reports[0]["channel"] == 1

    def test_the_report_is_worded_as_an_observation_not_a_verdict(self):
        """Naming a cause is the operator's call, not the campaign's (decision vi)."""
        events: list[tuple] = []
        engine, _ = _engine(emit=lambda name, **kw: events.append((name, kw)))
        self._pattern(engine)
        message = [kw for n, kw in events if n == "channel_reject_pattern"][0]["message"]
        assert "rejected on boards" in message
        assert "retracted" in message
        for verdict_word in ("faulty", "broken", "bad connector", "replace",
                             "failed hardware", "defective"):
            assert verdict_word not in message.lower()

    def test_one_board_alone_is_not_a_pattern(self):
        events: list[tuple] = []
        engine, model = _engine(emit=lambda name, **kw: events.append((name, kw)))
        engine.note_accept(channel=2, board_id="b1")
        for params in (PARAMS, OTHER):
            engine.record_gate_reject(
                params=params, channel=1, board_id="b1",
                primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        assert not [n for n, _ in events if n == "channel_reject_pattern"]
        assert model.n_infeasible == 2

    def test_the_same_composition_on_two_boards_is_not_unrelated(self):
        events: list[tuple] = []
        engine, _ = _engine(emit=lambda name, **kw: events.append((name, kw)))
        engine.note_accept(channel=2, board_id="b1")
        engine.note_accept(channel=2, board_id="b2")
        for board in ("b1", "b2"):
            engine.record_gate_reject(
                params=PARAMS, channel=1, board_id=board,
                primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        assert not [n for n, _ in events if n == "channel_reject_pattern"]

    def test_the_pattern_path_makes_no_instrument_call(self):
        """2f — passivity, asserted by spying on the manager.

        Spying catches an *added call*, which is the failure mode that matters:
        a diagnostic sweep, a re-measure beyond §3.2's repeats, or an allocator
        nudge would all show up here, where asserting on columns would not.
        """
        manager = create_mock_manager()
        seen: list[tuple] = []

        for name in list(manager.names):
            inst = manager.get(name)
            for attr in ("connect", "disconnect", "sendscript_getdata",
                         "move_to", "dispense", "measure", "read"):
                original = getattr(inst, attr, None)
                if original is None or not callable(original):
                    continue

                def spy(*a, _n=name, _a=attr, _o=original, **kw):
                    seen.append((_n, _a))
                    return _o(*a, **kw)

                try:
                    setattr(inst, attr, spy)
                except Exception:
                    pass

        engine, model = _engine()
        # The engine never receives the manager at all — passivity by
        # construction, not by discipline.
        assert not hasattr(engine, "_manager")
        self._pattern(engine)

        assert seen == [], f"the §3.3 path touched instruments: {seen}"
        assert model.n_infeasible == 0

    def test_a_flagged_channel_withholds_every_later_label(self):
        engine, model = _engine()
        self._pattern(engine)
        engine.note_accept(channel=2, board_id="b3")
        later = engine.record_gate_reject(
            params={"x0": 0.44, "x1": 0.44}, channel=1, board_id="b3",
            primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        assert later.label is None
        assert "channel-pattern report" in later.failure_reason
        assert model.labels == []

    def test_an_unflagged_channel_is_unaffected_by_its_neighbours_pattern(self):
        engine, model = _engine()
        self._pattern(engine)
        engine.note_accept(channel=2, board_id="b3")
        other = engine.record_gate_reject(
            params=OTHER, channel=5, board_id="b3",
            primary_report=_report(OPEN), confirmations=[_report(OPEN)])
        assert other.label == INFEASIBLE
        assert model.n_infeasible == 1


# ── The interim persistence seam (T3.1b) ─────────────────────────────────────

class TestOutcomesAreWrittenThroughAnInterfaceNotAColumn:
    def test_every_decision_reaches_the_sink(self):
        written: list[dict] = []

        class _Sink:
            def record_outcome(self, **kw):
                written.append(kw)

        engine, _ = _engine(sink=_Sink(), run_id="run-1")
        engine.record_measured(params=PARAMS, channel=1, board_id="b1")
        engine.record_rig_failure(params=OTHER, what="execute: X", channel=2)
        assert [w["outcome"] for w in written] == [OUTCOME_MEASURED, OUTCOME_UNKNOWN]
        assert {w["run_id"] for w in written} == {"run-1"}

    def test_a_failing_sink_never_costs_the_label(self):
        class _Broken:
            def record_outcome(self, **kw):
                raise RuntimeError("disk full")

        engine, model = _engine(sink=_Broken())
        d = engine.record_measured(params=PARAMS, channel=1, board_id="b1")
        assert d.label == FEASIBLE
        assert model.n_feasible == 1

    def test_feasible_labels_are_reconstructible_from_what_is_persisted_today(self):
        """A NOT NULL objective_value is a feasible label, with no new column.

        This is the interim contract: until T3.1b lands the outcome columns, the
        feasible class survives a restart because it is derivable from data the
        DataStore already holds, and the infeasible class does not — which the
        checkpoint's ``n_infeasible_labels`` makes visible rather than silent.
        """
        engine, model = _engine()
        rows = [({"x0": 0.1, "x1": 0.2}, 1.5), ({"x0": 0.3, "x1": 0.4}, None)]
        for params, objective in rows:
            if objective is not None:
                engine.record_measured(params=params, channel=1, board_id="b1",
                                       objective_value=objective)
        assert model.n_feasible == 1
        assert model.n_infeasible == 0
