"""Composition axes — a twin target with a range, so a campaign can search it.

The Live BO tab could only search raw pump volumes. That is the easier mode and a
legitimate one, but it costs stock identity: without it the twin cannot elute, there
is no dry thickness, and the objective can only be mean |Z|. Searching composition
targets instead gives every trial a predicted thickness, which is exactly what makes
conductivity available — so this module is what lets the *same tab* ask either
question.

Pinning is the subtle part. An axis with ``low == high`` is a target held constant,
and it must never reach the optimizer: a zero-width dimension costs a GP dimension
and a scaling division by zero to learn nothing.
"""

from __future__ import annotations

import pytest

from softae.core.composition_axes import (
    AXIS_KINDS,
    CompositionAxis,
    axes_parameter_space,
    build_targets_from_axes,
    describe_axes,
    stocks_from_loadout,
    validate_axes,
)
from softae.core.formulation import (
    Basis,
    ConcentrationTarget,
    DriedFractionTarget,
    MolarRatioTarget,
)

RATIO = CompositionAxis("molar_ratio", "PEO", "LiCl", low=5.0, high=40.0)
FRAC = CompositionAxis("dried_fraction", "SiO2", low=0.0, high=0.2)
PINNED = CompositionAxis("dried_fraction", "SiO2", low=0.1, high=0.1)


class TestAxisValidation:
    def test_every_kind_the_twin_offers_is_expressible(self):
        assert set(AXIS_KINDS) == {"molar_ratio", "dried_fraction", "concentration"}

    def test_a_ratio_needs_both_species(self):
        with pytest.raises(ValueError, match="both A and B"):
            CompositionAxis("molar_ratio", "PEO", "", low=1.0, high=2.0)

    def test_an_axis_needs_a_subject(self):
        with pytest.raises(ValueError, match="species/component"):
            CompositionAxis("dried_fraction", "  ", low=0.0, high=1.0)

    def test_an_unknown_kind_is_refused_rather_than_silently_dropped(self):
        with pytest.raises(ValueError, match="unknown axis kind"):
            CompositionAxis("viscosity", "PEO", low=0.0, high=1.0)

    def test_inverted_bounds_are_refused(self):
        with pytest.raises(ValueError, match="below low"):
            CompositionAxis("concentration", "LiCl", low=5.0, high=1.0)


class TestParameterSpace:
    def test_a_searched_axis_becomes_a_float_dimension(self):
        space = axes_parameter_space([RATIO])
        assert space == {"ratio_PEO_LiCl": {"type": "float", "low": 5.0, "high": 40.0}}

    def test_a_pinned_axis_is_kept_out_of_the_optimizers_hands(self):
        # A zero-width dimension teaches the GP nothing and costs it a dimension —
        # and normalising by (high - low) divides by zero.
        assert axes_parameter_space([RATIO, PINNED]) == axes_parameter_space([RATIO])

    def test_names_are_distinct_per_subject_so_two_axes_cannot_collide(self):
        other = CompositionAxis("molar_ratio", "PEO", "SiO2", low=1.0, high=2.0)
        assert RATIO.name != other.name

    def test_names_survive_chemical_names_that_are_not_identifiers(self):
        axis = CompositionAxis("concentration", "Li+ (aq)", low=1.0, high=2.0)
        assert axis.name.isidentifier()


class TestBuildTargets:
    def test_a_suggestion_becomes_the_targets_the_solver_takes(self):
        targets = build_targets_from_axes([RATIO, FRAC])(
            {"ratio_PEO_LiCl": 20.0, "driedfrac_SiO2": 0.05})
        assert targets == [
            MolarRatioTarget("PEO", "LiCl", 20.0),
            DriedFractionTarget("SiO2", 0.05, Basis.VOLUME),
        ]

    def test_a_pinned_axis_contributes_its_constant_not_a_searched_value(self):
        targets = build_targets_from_axes([RATIO, PINNED])({"ratio_PEO_LiCl": 20.0})
        assert targets[1] == DriedFractionTarget("SiO2", 0.1, Basis.VOLUME)

    def test_concentration_axes_carry_molarity(self):
        axis = CompositionAxis("concentration", "LiCl", low=0.5, high=2.0)
        assert build_targets_from_axes([axis])({axis.name: 1.5}) == [
            ConcentrationTarget("LiCl", 1.5)]

    def test_the_basis_is_honoured(self):
        axis = CompositionAxis("dried_fraction", "SiO2", low=0.0, high=1.0, basis="mass")
        assert build_targets_from_axes([axis])({axis.name: 0.3})[0].basis is Basis.MASS

    def test_a_missing_axis_falls_back_instead_of_killing_the_round(self):
        # A campaign that dies mid-round on a KeyError loses the plate it was
        # casting; one that solves on a stale bound is recoverable and logged.
        targets = build_targets_from_axes([RATIO])({})
        assert targets == [MolarRatioTarget("PEO", "LiCl", 5.0)]


class TestDeterminacy:
    def test_no_axes_at_all_is_reported(self):
        assert validate_axes([], n_stocks=3)

    def test_all_pinned_is_reported_because_there_is_nothing_to_search(self):
        issues = validate_axes([PINNED], n_stocks=2)
        assert any("nothing to search" in i for i in issues)

    def test_a_duplicate_target_is_reported(self):
        issues = validate_axes([RATIO, RATIO], n_stocks=3)
        assert any("declared 2 times" in i for i in issues)

    def test_too_few_targets_for_the_stock_count_is_reported(self):
        # solve_formulation needs one target per stock; the deposition-µL box is one.
        issues = validate_axes([RATIO], n_stocks=4)
        assert any("under-determined" in i for i in issues)

    def test_a_determinate_set_reports_nothing(self):
        assert validate_axes([RATIO, FRAC], n_stocks=3) == []

    def test_the_description_names_what_is_searched_and_what_is_held(self):
        text = describe_axes([RATIO, PINNED])
        assert "1 searched" in text and "∈ [5, 40]" in text and "= 0.1" in text


class TestStocksFromLoadout:
    class _Loadout:
        def __init__(self, by_pump):
            self.by_pump = dict(by_pump)

        def declared_pumps(self):
            return tuple(sorted(self.by_pump))

        def solution_for(self, pid):
            return self.by_pump.get(pid)

    class _Catalog:
        def __init__(self, names):
            self._names = set(names)

        def get(self, name):
            if name not in self._names:
                raise KeyError(name)
            return f"solution:{name}"

    def test_the_pump_loadout_is_the_authority_on_what_is_loaded(self):
        stocks, pumps = stocks_from_loadout(
            self._Loadout({0: "PEO", 1: "LiCl"}), self._Catalog(["PEO", "LiCl"]))
        assert stocks == {"PEO": "solution:PEO", "LiCl": "solution:LiCl"}
        assert pumps == {"PEO": 0, "LiCl": 1}

    def test_a_stock_missing_from_the_catalog_is_skipped_not_guessed(self):
        # It cannot be solved for, and silently keeping it would change the
        # composition without saying so.
        stocks, pumps = stocks_from_loadout(
            self._Loadout({0: "PEO", 1: "Unobtainium"}), self._Catalog(["PEO"]))
        assert set(stocks) == {"PEO"} and set(pumps) == {"PEO"}

    def test_an_empty_loadout_yields_nothing_rather_than_raising(self):
        assert stocks_from_loadout(self._Loadout({}), self._Catalog([])) == ({}, {})
