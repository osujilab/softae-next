"""``[eis.gates]`` as an operator reads it back — the one-line summary.

``GateSettings.describe()`` is the only place the armed thresholds are stated in words,
which makes it the only place they can quietly stop matching the gates they name — and it
did exactly that: the line went on describing a one-sided engine rule for five days after
``db0b9ae`` (2026-09-03) made the engine two-sided, with a test pinning the stale wording.
One config key, ``rho_degenerate``, drives two tests — ``gate_degeneracy`` and
``engine_support._resolve_reported_resistance`` — and since ``db0b9ae`` both are two-sided
on the magnitude (``|ρ| ≥ |rho_degenerate|``). The line has to carry both, attributed:
naming only one misdescribes the other.

It did it a second time, on the other branch. The observing-only line claimed "nothing is
removed" while ``run_gates`` was dropping points on two spectra in three, because only the
REJECT short-circuit ever read ``[eis.gates] enabled`` — and the test pinning that wording
was *named after the false claim*, so a passing suite read as confirmation. Both branches
are now asserted on substance, and the retracted phrase is asserted absent.

Nothing here reads config or touches the rig; these are assertions about a string.
"""

from __future__ import annotations

from softae.analysis.eis.settings import GateSettings


class TestGateSummary:
    def test_the_summary_says_points_are_still_dropped_while_the_gates_only_observe(self):
        # The old name and the old line both claimed "nothing is removed", and the
        # claim was false in the shipped configuration: `run_gates` applies every
        # `block_point` mask regardless of `[eis.gates] enabled`, and only the REJECT
        # short-circuit reads that flag. `enabled = false` withholds the *refusal of a
        # spectrum*, not the dropping of points — 66.2 % of 515 stored spectra lose at
        # least one point in exactly this mode. So the assertion pins both halves, and
        # names the retracted claim so it cannot quietly return.
        text = GateSettings().describe()
        assert "observe only" in text
        assert "failing points are still dropped" in text
        assert "recorded SUSPECT rather than refusing the spectrum" in text
        assert "nothing is removed" not in text

    def test_the_degeneracy_gate_is_summarised_two_sided_on_the_magnitude(self):
        assert "|ρ| ≥ 0.95" in GateSettings(enabled=True,
                                            rho_degenerate=-0.95).describe()

    def test_the_summary_attributes_the_two_sided_sum_only_rule_to_the_engine(self):
        # The attribution is still the assertion — an unattributed "sum-only ..." claims
        # the gate does what only the engine does. What changed is the *substance*: the
        # engine selects sum-vs-split on |ρ| since `db0b9ae`, not on a signed cutoff, so
        # the old "sum-only below ρ = -0.95" was false about ρ = +1.000 — a spectrum the
        # engine reports as a sum and the line said it would split.
        text = GateSettings(enabled=True, rho_degenerate=-0.95).describe()
        assert "engine reports sum-only when |ρ| ≥ 0.95" in text

    def test_a_positive_threshold_still_summarises_both_halves_on_the_magnitude(self):
        # Both the gate and the engine apply abs() to the configured value, so the
        # string does too: the sign of a config key must not silently invert what the
        # line claims about either half.
        text = GateSettings(enabled=True, rho_degenerate=0.95).describe()
        assert "|ρ| ≥ 0.95" in text
        assert "engine reports sum-only when |ρ| ≥ 0.95" in text
