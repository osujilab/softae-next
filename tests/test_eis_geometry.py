"""Per-sample cell constant (E0) — R12, without moving a single stored number.

Two things matter here and they pull in opposite directions. The rename must be
*arithmetically free*, so no historical σ changes and no database column is touched;
and a missing per-sample thickness must yield **no conductivity at all** rather than
one built on the 0.175 cm placeholder that has been the default since this pipeline
was written.
"""

from __future__ import annotations

import math

import pytest

from softae.analysis.circuit_fitting import z_to_sigma
from softae.analysis.eis.geometry import (
    DEFAULT_THICKNESS_MAX_CM,
    CellConstant,
    cell_constant_for_sample,
    resolve_thickness_cm,
)


class TestLegacyParity:
    def test_zero_dead_height_reproduces_the_legacy_sigma_exactly_so_the_rename_changes_no_number(self):
        """``z_to_sigma`` is deprecated and warns; it is called here *as the oracle*.

        The expectation is written out with ``pytest.warns`` rather than suppressed, so
        the parity proof keeps working while any accidental re-adoption in production
        still trips the completeness guard in ``test_eis_universal_fit_route.py``.
        """
        L, t, w, R = 0.2, 0.015, 0.2, 5.0e4
        cell = CellConstant(L_gap_cm=L, L_stripe_cm=w, thickness_cm=t)
        assert cell.dead_height_cm == 0.0
        with pytest.warns(DeprecationWarning, match="z_to_sigma is deprecated"):
            legacy = float(z_to_sigma(L, t, w, R))
        assert cell.sigma(R) == legacy

    def test_the_legacy_triple_round_trips_without_folding_in_the_dead_height(self):
        # The stored geometry columns record what was *measured*. Folding t-h into
        # them would make one column mean two things depending on when it was written.
        cell = CellConstant(0.2, 0.2, 0.015, dead_height_cm=0.0048)
        assert cell.as_legacy_triple() == (0.2, 0.015, 0.2)
        assert CellConstant.from_legacy(*cell.as_legacy_triple()).thickness_cm == 0.015

    def test_the_shipped_board_geometry_lands_in_the_expected_cell_constant_range(self):
        # Overhaul §1 puts this coplanar cell at 50-100 /cm for a real film.
        for t_um, expected in ((200, 50.0), (150, 66.67), (100, 100.0)):
            cell = CellConstant(0.2, 0.2, t_um * 1e-4)
            assert cell.K_per_cm == pytest.approx(expected, rel=1e-3)


class TestPlaceholderThickness:
    def test_the_historical_default_reads_as_implausible_because_it_is_ten_times_a_film(self):
        cell = CellConstant(0.2, 0.2, 0.175)      # the legacy DEFAULT_GEOMETRY value
        assert not cell.plausible
        assert cell.K_per_cm == pytest.approx(5.71, rel=1e-2)

    def test_an_implausible_thickness_still_computes_so_the_legacy_path_is_untouched(self):
        # Warn, never correct: changing this would move every number an operator has
        # already read off the analysis tab.
        cell = CellConstant(0.2, 0.2, 0.175)
        assert cell.sigma(1000.0) > 0

    def test_a_thickness_at_the_ceiling_is_plausible_and_just_above_it_is_not(self):
        assert CellConstant(0.2, 0.2, DEFAULT_THICKNESS_MAX_CM).plausible
        assert not CellConstant(0.2, 0.2, DEFAULT_THICKNESS_MAX_CM * 1.01).plausible


class TestDeadHeight:
    def test_dead_height_raises_the_cell_constant_because_less_film_carries_the_current(self):
        bare = CellConstant(0.2, 0.2, 0.015)
        corrected = CellConstant(0.2, 0.2, 0.015, dead_height_cm=0.0048)
        assert corrected.K_per_cm > bare.K_per_cm

    def test_a_film_thinner_than_the_dead_height_yields_no_conductivity(self):
        # Overhaul §9.1 calls this the sharp falsifiable test of whether h is real:
        # a film cast below h should show essentially no lateral conduction.
        cell = CellConstant(0.2, 0.2, 0.003, dead_height_cm=0.0048)
        assert cell.effective_thickness_cm == 0.0
        assert math.isnan(cell.K_per_cm)
        assert math.isnan(cell.sigma(1000.0))
        assert not cell.plausible


class TestThicknessProvenance:
    def test_profilometry_outranks_every_computed_thickness(self):
        t, method = resolve_thickness_cm(
            profilometry_um=120.0, target_um=150.0, predicted_um=140.0)
        assert method == "profilometry"
        assert t == pytest.approx(0.012)

    def test_an_autonomous_run_uses_its_thickness_target_as_the_nominal(self):
        t, method = resolve_thickness_cm(target_um=150.0, predicted_um=140.0)
        assert method == "target"
        assert t == pytest.approx(0.015)

    def test_a_high_throughput_run_uses_the_twins_predicted_thickness(self):
        t, method = resolve_thickness_cm(predicted_um=140.0, dispensed_um=200.0)
        assert method == "predicted"

    def test_a_missing_thickness_is_unavailable_rather_than_defaulted(self):
        t, method = resolve_thickness_cm()
        assert method == "unavailable"
        assert math.isnan(t)

    def test_a_non_positive_or_non_finite_thickness_falls_through_to_the_next_source(self):
        assert resolve_thickness_cm(target_um=0.0, predicted_um=140.0)[1] == "predicted"
        assert resolve_thickness_cm(target_um=float("nan"),
                                    predicted_um=140.0)[1] == "predicted"


class TestCellConstantForSample:
    def test_no_per_sample_thickness_yields_no_cell_rather_than_a_nominal_one(self):
        # Mirrors P7.2's posture for deposit area: absent is never guessed, because
        # conductivity divides by thickness and an invented one corrupts it silently.
        assert cell_constant_for_sample(config={}) is None

    def test_board_geometry_overrides_the_configured_defaults(self):
        cell = cell_constant_for_sample(
            predicted_um=150.0, config={},
            pcb_config={"electrode_L_cm": 0.3, "electrode_w_cm": 0.4})
        assert cell is not None
        assert (cell.L_gap_cm, cell.L_stripe_cm) == (0.3, 0.4)

    def test_the_resolved_method_is_carried_so_provenance_survives_to_the_database(self):
        cell = cell_constant_for_sample(target_um=150.0, config={})
        assert cell is not None and cell.thickness_method == "target"
        assert cell.measured_per_sample


class TestUncertainty:
    def test_thickness_uncertainty_combines_with_the_fit_in_quadrature(self):
        cell = CellConstant(0.2, 0.2, 0.015, thickness_unc_cm=0.003)  # 20 %
        assert cell.sigma_rel_uncertainty(0.0) == pytest.approx(0.2)
        assert cell.sigma_rel_uncertainty(0.15) == pytest.approx(
            math.hypot(0.15, 0.2))

    def test_an_unstated_thickness_uncertainty_contributes_nothing_rather_than_nan(self):
        cell = CellConstant(0.2, 0.2, 0.015)
        assert cell.sigma_rel_uncertainty(0.1) == pytest.approx(0.1)


class TestSigmaGuards:
    @pytest.mark.parametrize("bad_R", [0.0, -5.0, float("nan"), None, "x"])
    def test_a_non_positive_or_unreadable_resistance_yields_nan_not_an_exception(self, bad_R):
        assert math.isnan(CellConstant(0.2, 0.2, 0.015).sigma(bad_R))


class TestElectrodeConfigurationFactor:
    """`K_config_factor` (framework §1.1, R20, F16) — built, recorded, not yet armed.

    Three-electrode sensing measures only part of the current path. With WE and CE
    identical stripes and RE exactly mid-gap, mirror antisymmetry puts RE at the mean
    of the two electrode potentials, so ``Z_3el = ½·Z_2el`` at every frequency. The
    measured resistance is half the full-gap cell's, so the full-gap ``K_geom``
    over-reports σ by that factor — hence σ = K_geom / (factor · R).

    F16 is the nastiest class of error in the whole framework: a clean ~2× on the
    absolute number with the spectrum, the fit and the residuals all looking perfect.
    """

    def _cell(self, **over):
        from softae.analysis.eis.geometry import CellConstant

        base = dict(L_gap_cm=0.2, L_stripe_cm=0.2, thickness_cm=150e-4)
        base.update(over)
        return CellConstant(**base)

    def test_the_shipped_default_changes_no_number(self):
        # The doc's symmetry argument says 2.00; the only direct measurement on this
        # rig (overhaul §3.8) says 1.28x and 1.46x. Neither 2.00 nor noise — so the
        # term ships built and recorded at the value that moves nothing.
        cell = self._cell()
        assert cell.k_config_factor == 1.0
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm)

    def test_an_undeclared_configuration_leaves_absolute_sigma_unqualified(self):
        cell = self._cell()
        assert cell.electrode_config == "unverified"
        assert cell.config_declared is False
        assert cell.config_factor_verified is False

    def test_knowing_the_wiring_is_not_the_same_as_verifying_the_factor(self):
        # The rig is permanently 3-electrode, which is a fact anyone can read off the
        # board. The *factor* rests on stripe symmetry and RE centring, which nobody
        # has measured — and §3.8's own 1.28x/1.46x does not reproduce the predicted
        # 2.00. One flag for both would let the wiring fact silently arm a correction.
        cell = self._cell(electrode_config="3-electrode")
        assert cell.config_declared is True
        assert cell.config_factor_verified is False
        assert cell.k_config_factor == 1.0
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm)
        assert "unverified" in cell.describe()

    def test_verifying_the_factor_halves_the_effective_cell_constant(self):
        # R26 adds the third precondition: board symmetry alone no longer qualifies
        # the scale, because the symmetry argument assumes an ionic path to the RE.
        cell = self._cell(electrode_config="3-electrode", k_config_factor=2.0,
                          k_config_verified=True, re_contact_verified=True)
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm / 2.0)
        assert cell.config_factor_verified is True

    def test_two_electrode_needs_no_correction(self):
        cell = self._cell(electrode_config="2-electrode", k_config_factor=1.0,
                          k_config_verified=True)
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm)
        assert cell.config_factor_verified is True

    def test_the_factor_divides_so_sigma_falls_when_it_is_armed(self):
        # Direction matters: arming it must *reduce* the reported σ, because the
        # 3-electrode R is already half of what K_geom describes.
        plain = self._cell().sigma(1e5)
        armed = self._cell(electrode_config="3-electrode", k_config_factor=2.0,
                           k_config_verified=True).sigma(1e5)
        assert armed == pytest.approx(plain / 2.0)

    def test_a_nonsense_factor_falls_back_rather_than_producing_nan_sigma(self):
        from softae.analysis.eis.geometry import cell_config

        cfg = cell_config({"k_config_factor": 0.0})
        assert cfg["k_config_factor"] == 1.0

    def test_declaring_the_configuration_alone_does_not_arm_the_correction(self):
        from softae.analysis.eis.geometry import cell_config

        cfg = cell_config({"electrode_configuration": "3-electrode"})
        assert cfg["electrode_config"] == "3-electrode"   # recorded
        assert cfg["k_config_factor"] == 1.0              # but not applied
        assert cfg["k_config_verified"] is False

    def test_verifying_it_selects_the_configuration_factor_without_repeating_it(self):
        from softae.analysis.eis.geometry import cell_config

        assert cell_config({"electrode_configuration": "3-electrode",
                            "k_config_verified": True})["k_config_factor"] == 2.0
        assert cell_config({"electrode_configuration": "2-electrode",
                            "k_config_verified": True})["k_config_factor"] == 1.0

    def test_an_explicit_factor_overrides_the_configuration_default(self):
        from softae.analysis.eis.geometry import cell_config

        cfg = cell_config({"electrode_configuration": "3-electrode",
                           "k_config_factor": 1.35})
        assert cfg["k_config_factor"] == pytest.approx(1.35)

    def test_an_unknown_configuration_degrades_to_unverified(self):
        from softae.analysis.eis.geometry import cell_config

        cfg = cell_config({"electrode_configuration": "4-electrode"})
        assert cfg["electrode_config"] == "unverified"
        assert cfg["k_config_factor"] == 1.0

    def test_the_shipped_config_records_three_electrode_but_leaves_it_unarmed(self):
        # The operator's standing decision (2026-08-05): the rig stays 3-electrode for
        # the other techniques that need it. Recorded, so F16 is auditable; unarmed,
        # because the factor is still contested.
        from softae.analysis.eis.geometry import cell_config

        cfg = cell_config()
        assert cfg["electrode_config"] == "3-electrode"
        assert cfg["k_config_verified"] is False
        assert cfg["k_config_factor"] == 1.0

    def test_a_verified_board_still_does_not_qualify_a_sample_without_re_contact(self):
        # The precondition R26 adds. Board symmetry is checked once; contact is a
        # property of each film, and a dry or dewetted one has no ionic path.
        cell = self._cell(electrode_config="3-electrode", k_config_factor=2.0,
                          k_config_verified=True, re_contact_verified=False)
        assert cell.config_factor_verified is False
        assert "no verified RE contact" in cell.describe() or "RE contact" in cell.describe()

    def test_two_electrode_does_not_need_re_contact_because_no_re_is_in_the_path(self):
        # Requiring it here would relabel sound sigma as unqualified without changing
        # a single number: a two-electrode measurement has no RE in the sensing path,
        # so its factor of 1 is exact whether or not anything touches the stripe.
        cell = self._cell(electrode_config="2-electrode", k_config_verified=True,
                          re_contact_verified=False)
        assert cell.config_factor_verified is True

    def test_relative_trends_are_untouched_by_the_factor(self):
        # The reassurance that makes shipping it at 1.0 safe: a constant scale cannot
        # reorder a series, so every campaign ranking formulations is valid either way.
        cells = [self._cell(thickness_cm=t) for t in (100e-4, 150e-4, 200e-4)]
        armed = [self._cell(thickness_cm=t, electrode_config="3-electrode",
                            k_config_factor=2.0, k_config_verified=True)
                 for t in (100e-4, 150e-4, 200e-4)]
        plain_s = [c.sigma(1e5) for c in cells]
        armed_s = [c.sigma(1e5) for c in armed]
        ratios = [a / p for a, p in zip(armed_s, plain_s)]
        assert all(r == pytest.approx(ratios[0]) for r in ratios)


class TestReferenceElectrodeContactPrecondition:
    """R26 — `K_config_factor = 2` requires an ionic path to the reference stripe.

    The symmetry derivation behind the factor (§3.8) assumes RE senses a real potential
    in a conducting medium. §3.10 shows what happens when it does not: the RE floats
    onto a capacitive divider whose ratio depends on the load, measured at α = 2.2–23.8
    and *not reproducible even at fixed load* (9.85 and 4.96 for the same 1 nF part).
    So the failure is not "the factor is 2 but we cannot confirm it" — the factor is
    undefined, and nothing can be applied.

    These tests pin the precondition at the point where the *number* is chosen, not
    only where it is labelled: a check that only relabels would still divide by two.
    """

    def _cfg(self, **over):
        base = {"electrode_configuration": "3-electrode", "k_config_verified": True}
        base.update(over)
        return base

    def test_an_armed_board_still_ships_factor_one_without_contact(self):
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(predicted_um=150.0, config=self._cfg())
        assert cell is not None
        assert cell.k_config_factor == 1.0          # demoted, not merely unlabelled
        assert cell.config_factor_verified is False
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm)

    def test_verified_contact_on_an_armed_board_applies_the_factor(self):
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0, config=self._cfg(),
            re_contact_verified=True, re_connection="bridged_by_sample")
        assert cell.k_config_factor == 2.0
        assert cell.config_factor_verified is True
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm / 2.0)

    def test_tied_to_ce_closes_the_loop_but_is_not_ionic_contact(self):
        # The case that makes RE_IONIC_CONTACT a different set from RE_CLOSED_LOOP.
        # Jumpering RE to CE closes the control loop perfectly while making the RE read
        # the counter electrode -- the measurement IS two-electrode, so its factor is 1.
        # Applying 2 here would be F16 self-inflicted: a clean 2x with a perfect fit.
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0, config=self._cfg(),
            re_contact_verified=True, re_connection="tied_to_ce")
        assert cell.k_config_factor == 1.0
        assert cell.re_contact_verified is False

    def test_the_two_re_records_contradicting_fails_closed(self):
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0, config=self._cfg(),
            re_contact_verified=True, re_connection="open_by_geometry")
        assert cell.re_contact_verified is False
        assert cell.k_config_factor == 1.0

    def test_an_unrecorded_connection_is_not_a_contradiction(self):
        # "unverified" means nothing was written down, not that contact was denied.
        # Treating silence as a contradiction would make the explicit assertion useless
        # everywhere the acquisition path does not yet capture re_connection.
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0, config=self._cfg(), re_contact_verified=True)
        assert cell.re_contact_verified is True
        assert cell.k_config_factor == 2.0

    def test_contact_alone_does_not_arm_an_unverified_board(self):
        # The preconditions are independent: contact is per sample, symmetry per board.
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0,
            config={"electrode_configuration": "3-electrode",
                    "k_config_verified": False},
            re_contact_verified=True, re_connection="bridged_by_sample")
        assert cell.k_config_factor == 1.0
        assert cell.config_factor_verified is False

    def test_an_explicit_measured_factor_is_still_subject_to_the_precondition(self):
        # Pinning a board's measured ratio (say 1.37) does not exempt it: the ratio was
        # measured on films that had contact, and it means nothing on one that does not.
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(
            predicted_um=150.0,
            config={"electrode_configuration": "3-electrode", "k_config_factor": 1.37})
        assert cell.k_config_factor == 1.0

    def test_ionic_contact_is_a_strict_subset_of_a_closed_loop(self):
        from softae.analysis.eis.policy import RE_CLOSED_LOOP, RE_IONIC_CONTACT

        assert RE_IONIC_CONTACT < RE_CLOSED_LOOP
        assert "tied_to_ce" in RE_CLOSED_LOOP
        assert "tied_to_ce" not in RE_IONIC_CONTACT

    def test_the_shipped_config_is_unaffected_because_it_is_unarmed_anyway(self):
        # R26 changes no number today: k_config_verified already ships false. It is a
        # trap defused ahead of the bench work that would otherwise spring it.
        from softae.analysis.eis.geometry import cell_constant_for_sample

        cell = cell_constant_for_sample(predicted_um=150.0)
        assert cell.k_config_factor == 1.0
        assert cell.K_per_cm == pytest.approx(cell.K_geometric_per_cm)
