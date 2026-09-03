"""``[eis.gates]`` as an operator reads it back — the one-line summary.

``GateSettings.describe()`` is the only place the armed thresholds are stated in words,
which makes it the only place they can quietly stop matching the gates they name. One
config key, ``rho_degenerate``, now drives two different tests: ``gate_degeneracy`` is
two-sided on the magnitude, while ``engine_support._resolve_reported_resistance`` keeps
the one-sided rule that chooses the reported resistance. The divergence is deliberate,
so the line has to carry both — naming only one misdescribes the other.

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

    def test_the_summary_keeps_the_engines_one_sided_rule_and_attributes_it(self):
        # The engine still selects sum-vs-split from the signed comparison, so the
        # summary must not read as though the gate's two-sided test replaced it. The
        # attribution is the assertion: an unattributed "sum-only below ρ = ..." was
        # exactly the old line, which claimed the gate did what only the engine does.
        text = GateSettings(enabled=True, rho_degenerate=-0.95).describe()
        assert "engine reports sum-only below ρ = -0.95" in text

    def test_a_positive_threshold_still_summarises_the_gate_on_the_magnitude(self):
        # `gate_degeneracy` applies abs() to the configured value, so the string does
        # too: the sign of a config key must not silently invert what the line claims.
        assert "|ρ| ≥ 0.95" in GateSettings(enabled=True,
                                            rho_degenerate=0.95).describe()
