"""The arc-closure verdict as a Front-2 flag: it must report, and it must not refuse.

Two things are pinned here, and the second is the one that would be expensive to get
wrong. The first is ordinary: the gate says ``open`` on a truncated arc and ``closed``
on a whole one. The second is the contract the whole design rests on — **this gate
cannot reject a spectrum**, whatever the arc does, because promoting the arc state to
a refusal was measured on the ``probe-3ch-v3`` corpus and rejected (it would have
refused the gated engine's single best row and admitted two of its four worst, at a
cost of 32.8 % of the corpus in new refusals). A flag that quietly gained the power to
refuse would reinstate exactly that, so it is asserted against ``reduce_gates``
itself rather than against the severity string alone.

The generators are imported from ``test_eis_arc_closure`` rather than rewritten: that
module's ``semicircle`` is the shape the arc rule's own tests are calibrated on, and a
second generator here would let the two files drift into testing different arcs.
``Z′`` is derived from it analytically rather than generated — for a Debye arc
``Z′ = (−Z″)·f_peak/f`` exactly — so the complex spectrum the gate consumes is the
same arc, not a lookalike.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis.arc import UNKNOWN, arc_closure
from softae.analysis.eis.arc_gate import GATE_NAME, gate_arc_closure
from softae.analysis.eis.engine_support import PregateSettings
from softae.analysis.eis.gates import (
    BLOCK_SPECTRUM,
    FLAG,
    SEVERITIES,
    GateResult,
)
from softae.analysis.eis.policy import reduce_gates
from softae.analysis.quality import Verdict
from tests.test_eis_arc_closure import semicircle

#: Fixed rather than inherited from ``softae_config.toml``. A gate whose expected
#: verdict moves with whatever the config says this hour is a gate whose tests prove
#: nothing about the gate.
PREGATE = PregateSettings(phase_low_max_deg=-60.0)
CTX = {"pregate": PREGATE}


def _debye_complex(f, y_neg, f_peak, r_series=0.0):
    """``Z`` in the physics convention (``Im Z < 0`` capacitive) for the same arc.

    ``semicircle`` emits ``−Z″ = R·x/(1+x²)`` with ``x = f/f_peak``; the matching real
    part of a Debye arc is ``R/(1+x²)``, which is ``(−Z″)/x``. Deriving it that way
    keeps this file from owning a second arc definition.

    *r_series* shifts the whole arc right along ``Z′`` without touching ``−Z″``, so it
    moves the **phase at the sweep floor** — the severity — while leaving the arc
    *state* exactly where it was. That is the one knob these tests need.
    """
    f = np.asarray(f, dtype=float)
    y_neg = np.asarray(y_neg, dtype=float)
    return (r_series + y_neg * f_peak / f) - 1j * y_neg


def _closed_spectrum():
    """A whole arc: the peak sits two decades above the sweep floor."""
    f, y = semicircle(f_peak=1.0e3, f_lo=20.0)
    return f, _debye_complex(f, y, 1.0e3)


def _open_spectrum(r_series=0.0):
    """The same arc with the sweep stopped an order of magnitude above its peak."""
    f, y = semicircle(f_peak=2.0, f_lo=20.0)
    return f, _debye_complex(f, y, 2.0, r_series=r_series)


def _unjudgeable_spectrum():
    """Four points — below ``arc_closure``'s ``min_points``, so it cannot judge."""
    f = np.array([2.0e5, 2.0e4, 2.0e3, 2.0e2])
    return f, _debye_complex(f, np.array([1.0, 5.0, 9.0, 4.0]), 2.0e3)


class TestTheGateReportsTheArcState:
    def test_gate_arc_closure_closed_arc_passes(self):
        f, Z = _closed_spectrum()
        r = gate_arc_closure(f, Z, CTX)
        assert r.passed is True
        assert r.checked is True
        assert "arc closed in band" in r.detail

    def test_gate_arc_closure_open_arc_flags(self):
        f, Z = _open_spectrum()
        r = gate_arc_closure(f, Z, CTX)
        assert r.passed is False
        assert r.checked is True
        assert "did not close in band" in r.detail

    def test_gate_arc_closure_unjudgeable_sweep_is_unchecked_not_failed(self):
        # `GateResult.__post_init__` forbids passed=False with checked=False, so the
        # only legal encoding of "could not judge" is fail-open plus checked=False.
        # It must not be spelled the same way as "checked and clean".
        f, Z = _unjudgeable_spectrum()
        r = gate_arc_closure(f, Z, CTX)
        assert r.checked is False
        assert r.passed is True
        assert arc_closure(f, -Z.imag).state == UNKNOWN

    def test_gate_arc_closure_agrees_with_the_arc_rule_it_wraps(self):
        # The gate must not become a second definition of closure. Same verdict as
        # `arc_closure` on the same spectrum, or the two have drifted apart.
        for build in (_closed_spectrum, _open_spectrum):
            f, Z = build()
            arc = arc_closure(f, -Z.imag, np.angle(Z, deg=True))
            assert gate_arc_closure(f, Z, CTX).passed is arc.closed

    def test_gate_arc_closure_name_matches_the_stored_record_spelling(self):
        # A query over historical `gate_log_json` and one over new rows must select
        # the same gate rather than two spellings of it.
        f, Z = _open_spectrum()
        assert gate_arc_closure(f, Z, CTX).name == GATE_NAME
        assert arc_closure(f, -Z.imag).as_record()["gate"] == GATE_NAME


class TestTheSeverityIsOneTheConsumersRecognise:
    def test_gate_arc_closure_severity_is_a_member_of_severities(self):
        # The defect this work exists to fix: `"annotate"` was in no enum, so every
        # consumer that switches on severity fell through it in silence.
        for build in (_closed_spectrum, _open_spectrum, _unjudgeable_spectrum):
            f, Z = build()
            r = gate_arc_closure(f, Z, CTX)
            assert r.severity in SEVERITIES
            assert r.severity == FLAG

    def test_arc_closure_record_severity_is_a_member_of_severities(self):
        # The same fix on the persisted record. `as_record` has no live writer, but
        # its shape is a read contract for rows written before the arc columns
        # landed, and a severity outside the enum is unreadable by anything.
        f, Z = _open_spectrum()
        assert arc_closure(f, -Z.imag).as_record()["severity"] in SEVERITIES

    def test_gate_arc_closure_removes_no_points_on_any_outcome(self):
        # A `flag` is advisory by definition. If the mask ever went partial the
        # severity would be lying about the consequence.
        for build in (_closed_spectrum, _open_spectrum, _unjudgeable_spectrum):
            f, Z = build()
            r = gate_arc_closure(f, Z, CTX)
            assert r.mask.all()
            assert r.n_dropped == 0


class TestTheFlagCannotRefuse:
    """The load-bearing contract, asserted against ``reduce_gates`` and not a string."""

    @pytest.mark.parametrize("build", [_closed_spectrum, _open_spectrum,
                                       _unjudgeable_spectrum])
    def test_reduce_gates_no_arc_outcome_can_produce_reject(self, build):
        f, Z = build()
        r = gate_arc_closure(f, Z, CTX)
        # `enabled=True` is the strongest form: with it False every REJECT is
        # downgraded to SUSPECT anyway, which would make this assertion vacuous.
        #
        # `n_surviving` is a fixed comfortable number rather than `f.size`, and that
        # is not a convenience. `reduce_gates` rejects independently when survivors
        # fall below `min_fit_pts`, so passing the four-point sweep's own size made
        # this test go red on a rule the gate has nothing to do with — the flag was
        # innocent and the assertion still failed. Isolating the gate is the point.
        report = reduce_gates([r], n_surviving=25, min_fit_pts=5, enabled=True)
        assert report.verdict is not Verdict.REJECT

    def test_reduce_gates_a_block_spectrum_result_does_reject(self):
        # POSITIVE CONTROL for the test above. Same call, same arguments, one field
        # different — if this did not reject, the assertion above would be passing
        # because `reduce_gates` never rejects anything here, not because a flag
        # cannot make it.
        f, Z = _open_spectrum()
        refusing = GateResult(GATE_NAME, BLOCK_SPECTRUM, False, "control",
                              np.ones(f.size, dtype=bool))
        report = reduce_gates([refusing], n_surviving=int(f.size), min_fit_pts=5,
                              enabled=True)
        assert report.verdict is Verdict.REJECT

    def test_reduce_gates_open_arc_is_suspect_rather_than_accept(self):
        # The flag is not inert either: it has to reach the verdict, or recording it
        # buys nothing. This is the whole operator-visible consequence of the change.
        f, Z = _open_spectrum()
        report = reduce_gates([gate_arc_closure(f, Z, CTX)],
                              n_surviving=int(f.size), min_fit_pts=5)
        assert report.verdict is Verdict.SUSPECT
        assert any(i.startswith(f"{GATE_NAME}:") for i in report.issues)

    def test_reduce_gates_closed_arc_leaves_the_verdict_at_accept(self):
        f, Z = _closed_spectrum()
        report = reduce_gates([gate_arc_closure(f, Z, CTX)],
                              n_surviving=int(f.size), min_fit_pts=5)
        assert report.verdict is Verdict.ACCEPT


class TestTheThresholdIsRecordedRatherThanApplied:
    """``passed`` turns on the state; ``phase_low_max_deg`` only labels the severity."""

    def test_gate_arc_closure_capacitive_open_arc_records_blocking_open(self):
        # Bare arc: at ten times the peak frequency the floor is essentially all
        # reactance, so the phase there is well below −60°.
        f, Z = _open_spectrum()
        r = gate_arc_closure(f, Z, CTX)
        assert r.metrics["arc_phase_low_deg"] < -60.0
        assert r.metrics["arc_blocking_open"] == 1.0

    def test_gate_arc_closure_resistive_open_arc_still_flags_but_is_not_severe(self):
        # A large series resistance moves the floor phase toward 0° without touching
        # −Z″, so the arc is equally open and the severity is not the same.
        f, Z = _open_spectrum(r_series=1.0e7)
        r = gate_arc_closure(f, Z, CTX)
        assert r.passed is False
        assert r.metrics["arc_phase_low_deg"] > -60.0
        assert r.metrics["arc_blocking_open"] == 0.0

    def test_gate_arc_closure_threshold_comes_from_the_pregate_settings(self):
        # The same spectrum, two thresholds, two labels — which is what proves the
        # number is read from `pregate_settings` and not hardcoded here.
        f, Z = _open_spectrum(r_series=1.0e7)
        phase = float(gate_arc_closure(f, Z, CTX).metrics["arc_phase_low_deg"])
        lenient = PregateSettings(phase_low_max_deg=phase - 1.0)
        strict = PregateSettings(phase_low_max_deg=phase + 1.0)
        assert gate_arc_closure(f, Z, {"pregate": lenient}
                                ).metrics["arc_blocking_open"] == 0.0
        assert gate_arc_closure(f, Z, {"pregate": strict}
                                ).metrics["arc_blocking_open"] == 1.0

    def test_gate_arc_closure_closed_arc_is_never_blocking_open(self):
        # `blocking_open` requires state == OPEN first, so no phase can make a closed
        # arc severe. Pins that the two fields cannot disagree.
        f, Z = _closed_spectrum()
        assert gate_arc_closure(f, Z, CTX).metrics["arc_blocking_open"] == 0.0


class TestTheGateNeedsNothingFromTheFit:
    def test_gate_arc_closure_empty_context_still_returns_a_verdict(self):
        # No `ctx["fit"]`, no covariance, no config override: the gate is a geometric
        # test on the raw sweep, which is why it cannot inherit the singularity
        # defect every other gate-accuracy number here was taken through.
        f, Z = _open_spectrum()
        r = gate_arc_closure(f, Z, {})
        assert r.severity == FLAG
        assert r.passed is False

    def test_gate_arc_closure_matches_the_cascade_call_signature(self):
        # `run_gates` calls `gate(f[idx], Z[idx], ctx)` positionally on a slice, so a
        # gate that needed a keyword or the whole array would fail only in a live run.
        f, Z = _closed_spectrum()
        idx = np.arange(f.size)[3:]
        r = gate_arc_closure(f[idx], Z[idx], CTX)
        assert r.mask.size == idx.size
