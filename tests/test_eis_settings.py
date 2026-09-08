"""``[eis.gates]`` as an operator reads it back — the one-line summary.

``GateSettings.describe()`` is the only place the armed thresholds are stated in words,
which makes it the only place they can quietly stop matching the gates they name — and it
did exactly that: the line went on describing a one-sided engine rule for five days after
``db0b9ae`` (2026-09-03) made the engine two-sided, with a test pinning the stale wording.
One config key, ``rho_degenerate``, drives two tests — ``gate_degeneracy`` and
``engine_support._resolve_reported_resistance`` — and since ``db0b9ae`` both are two-sided
on the magnitude (``|ρ| ≥ |rho_degenerate|``). The line has to carry both, attributed:
naming only one misdescribes the other.

Nothing here reads config or touches the rig; these are assertions about a string.
"""

from __future__ import annotations

from softae.analysis.eis.settings import GateSettings


class TestGateSummary:
    def test_the_summary_says_nothing_is_removed_while_the_gates_only_observe(self):
        assert "observe only" in GateSettings().describe()

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
