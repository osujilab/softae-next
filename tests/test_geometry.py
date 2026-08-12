"""Electrode-grid geometry helpers, incl. the per-well volume capacity reader."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from softae.config import loader
from softae.core.geometry import (
    FILL_ABOVE_BRIM,
    FILL_OK,
    FILL_OVER_PERMITTED,
    FILL_UNKNOWN,
    bead_height_for_cap_mm,
    brim_cap_uL,
    classify_fill,
    deposit_area_mm2,
    electrode_count,
    electrode_xy_for_channel,
    elution_capacity_uL,
    hemisphere_cap_uL,
    nearest_electrode,
    permitted_overfill_mm,
    well_capacity_uL,
    well_void_uL,
)
from softae.core.overflow import sweep_overflow, well_overflow


class TestDepositArea:
    """The denominator of every thickness number (P7.2), resolved in three tiers."""

    def test_an_explicit_area_wins_over_everything(self):
        """The escape hatch: a masked or incompletely-wetted well."""
        pcb = {"deposit_area_mm2": 7.5, "well_diameter_mm": 4.88,
               "electrode_L_cm": 0.2, "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) == 7.5

    def test_a_declared_well_beats_the_electrode_rectangle(self):
        """`electrode_L_cm`/`electrode_w_cm` are the gap and the stripe length --
        conduction geometry. A drop covers the well, not the inter-electrode
        rectangle, so a board that declares a well must use it."""
        pcb = {"well_diameter_mm": 4.88, "electrode_L_cm": 0.2, "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) == pytest.approx(math.pi * 2.44 ** 2)
        assert deposit_area_mm2(pcb) == pytest.approx(18.7038, abs=1e-3)

    def test_the_rectangle_remains_the_fallback_so_silent_boards_do_not_move(self):
        pcb = {"electrode_L_cm": 0.2, "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) == pytest.approx(4.0)

    def test_a_board_declaring_nothing_is_unavailable_rather_than_guessed(self):
        assert deposit_area_mm2({"grid": [4, 4]}) is None

    def test_a_nonpositive_or_unparseable_well_falls_through_rather_than_raising(self):
        for bad in (0, -1, "oops", None):
            pcb = {"well_diameter_mm": bad, "electrode_L_cm": 0.2,
                   "electrode_w_cm": 0.2}
            assert deposit_area_mm2(pcb) == pytest.approx(4.0)

    def test_the_four_stripe_board_uses_its_measured_well(self):
        """4.88 mm measured on-site. This is 4.68x the rectangle it replaces, so
        thicknesses on this board are correspondingly smaller than before."""
        stripe = loader.pcb_configs()["SoftAE_EIS_4Stripe"]
        area = deposit_area_mm2(stripe)
        assert area == pytest.approx(18.7038, abs=1e-3)
        assert area / 4.0 == pytest.approx(4.6759, abs=1e-3)

    def test_a_sessile_board_refuses_the_rectangle_because_a_droplet_is_not_a_well(self):
        """No wells means the wetted area is volume-and-contact-angle, i.e. an
        observation. The electrode rectangle describes the gap between two stripes
        and has no relationship to where the droplet sat, so falling through to it
        would return a confident number about the wrong thing."""
        pcb = {"cast_confinement": "sessile", "electrode_L_cm": 0.2,
               "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) is None

    def test_a_sessile_board_accepts_a_measured_footprint(self):
        pcb = {"cast_confinement": "sessile", "deposit_area_mm2": 12.5,
               "electrode_L_cm": 0.2, "electrode_w_cm": 0.2}
        assert deposit_area_mm2(pcb) == 12.5

    def test_the_ide_board_is_sessile_so_thickness_is_unavailable_not_guessed(self):
        """It has no wells at all. Inventing a diameter, or keeping the 4.0 mm2
        rectangle, would corrupt every thickness on that board silently."""
        ide = loader.pcb_configs()["SoftAE_IDE_EIS"]
        assert "well_diameter_mm" not in ide
        assert deposit_area_mm2(ide) is None

    def test_the_measured_depth_confirms_the_area_and_the_rectangle_could_not(self):
        """The independent check, now closed by a second measurement.

        `WellGeometry.from_board` back-solves a cylinder from area and capacity.
        Against the previously-declared 120 uL, the measured well implies a 6.416 mm
        depth and the electrode rectangle a 30 mm one. The wall stack measures
        6.35 mm — so the well area agrees to ~1 % and the rectangle is out by 5x.
        """
        from softae.core.deposition import WellGeometry

        assert WellGeometry.from_board(4.0, 120.0).depth_mm == pytest.approx(30.0)
        implied = WellGeometry.from_board(18.7038, 120.0).depth_mm
        assert implied == pytest.approx(6.4158, abs=1e-3)
        assert abs(implied - 6.35) / 6.35 < 0.02


class TestWellCapacityFromGeometry:
    """A board that states its dimensions should not restate their product."""

    def test_capacity_is_derived_from_a_declared_well(self):
        pcb = {"well_diameter_mm": 4.88, "well_depth_mm": 6.35}
        assert well_capacity_uL(pcb) == pytest.approx(118.769, abs=1e-3)

    def test_an_explicit_capacity_still_wins_so_a_margin_below_the_brim_is_possible(self):
        """A deliberate working margin is a policy choice geometry cannot infer."""
        pcb = {"well_diameter_mm": 4.88, "well_depth_mm": 6.35,
               "well_capacity_uL": 100.0}
        assert well_capacity_uL(pcb) == 100.0

    def test_a_half_declared_well_derives_nothing_rather_than_assuming_a_dimension(self):
        assert well_capacity_uL({"well_diameter_mm": 4.88}) is None
        assert well_capacity_uL({"well_depth_mm": 6.35}) is None

    def test_the_four_stripe_capacity_no_longer_overdeclares_the_well(self):
        """It was hand-typed as 120 against a 118.769 uL brim volume — an overflow
        guard permitting 1 % more fluid than the well physically holds."""
        stripe = loader.pcb_configs()["SoftAE_EIS_4Stripe"]
        cap = well_capacity_uL(stripe)
        assert cap == pytest.approx(118.769, abs=1e-3)
        assert cap < 120.0


class TestWellCapacity:
    def test_reads_declared_capacity(self):
        assert well_capacity_uL({"well_capacity_uL": 50}) == 50.0
        assert well_capacity_uL({"well_capacity_uL": 120.0}) == 120.0

    def test_absent_is_none(self):
        assert well_capacity_uL({"grid": [4, 4]}) is None

    def test_nonpositive_or_bad_is_none(self):
        assert well_capacity_uL({"well_capacity_uL": 0}) is None
        assert well_capacity_uL({"well_capacity_uL": -5}) is None
        assert well_capacity_uL({"well_capacity_uL": "oops"}) is None

    def test_real_board_capacities_from_config(self):
        """The two shipped boards carry their physical per-electrode capacities.

        They get there differently: the IDE board has no wells, so its 50 uL is a
        declared sessile-droplet budget; the 4-stripe board derives its brim volume
        from the well it declares.
        """
        pcbs = loader.pcb_configs()
        ide = pcbs["SoftAE_IDE_EIS"]
        stripe = pcbs["SoftAE_EIS_4Stripe"]
        assert well_capacity_uL(ide) == 50.0
        assert electrode_count(ide) == 16
        assert well_capacity_uL(stripe) == pytest.approx(118.769, abs=1e-3)
        assert electrode_count(stripe) == 32


class TestBrimBead:
    """A PTFE brim pins the contact line, so fluid stands proud instead of spilling.

    That head is real capacity. The point of separating it from the void is that a
    volume above the brim stops being a refusal and becomes a warning.
    """

    R = 2.44          # 4-stripe well radius (mm)

    def test_a_bead_of_height_equal_to_the_radius_is_exactly_a_hemisphere(self):
        """Cross-checks the cap formula against a closed form it does not share."""
        assert brim_cap_uL(self.R, self.R) == pytest.approx(
            (2.0 / 3.0) * math.pi * self.R ** 3)

    def test_zero_and_negative_heights_carry_nothing(self):
        assert brim_cap_uL(self.R, 0.0) == 0.0
        assert brim_cap_uL(self.R, -1.0) == 0.0

    def test_the_height_inversion_round_trips_the_volume(self):
        for h in (0.1, 0.5, 1.0, 2.44):
            assert bead_height_for_cap_mm(brim_cap_uL(self.R, h), self.R) == (
                pytest.approx(h, abs=1e-6))

    def test_the_hemisphere_is_reported_as_the_ceiling_for_this_well(self):
        stripe = loader.pcb_configs()["SoftAE_EIS_4Stripe"]
        assert hemisphere_cap_uL(stripe) == pytest.approx(30.425, abs=1e-3)


class TestFillBands:
    PCB = {"well_diameter_mm": 4.88, "well_depth_mm": 6.35}
    VOID = 118.769

    def _with_overfill(self, mm):
        return {**self.PCB, "permitted_overfill_mm": mm}

    def test_no_permitted_overfill_means_the_brim_is_the_limit(self):
        """The default. Declaring nothing must behave exactly as before the key
        existed, so overfill is a permission that has to be granted."""
        assert elution_capacity_uL(self.PCB) == pytest.approx(self.VOID, abs=1e-3)
        assert classify_fill(self.VOID + 1, self.PCB).blocks

    def test_a_permitted_bead_raises_the_hard_stop_by_its_own_volume(self):
        pcb = self._with_overfill(1.0)
        assert elution_capacity_uL(pcb) == pytest.approx(128.645, abs=1e-3)

    def test_a_volume_inside_the_void_fits(self):
        v = classify_fill(100.0, self._with_overfill(1.0))
        assert v.band == FILL_OK
        assert not v.warns and not v.blocks

    def test_above_the_brim_but_within_the_permitted_bead_warns_not_stops(self):
        """The whole point: this used to be indistinguishable from an overflow."""
        v = classify_fill(125.0, self._with_overfill(1.0))
        assert v.band == FILL_ABOVE_BRIM
        assert v.warns and not v.blocks
        assert 0.0 < v.bead_height_mm < 1.0
        assert "ABOVE THE BRIM" in v.describe()

    def test_beyond_the_permitted_bead_stops(self):
        v = classify_fill(129.0, self._with_overfill(1.0))
        assert v.band == FILL_OVER_PERMITTED
        assert v.blocks and not v.warns
        assert v.headroom_uL < 0

    def test_the_boundary_volumes_land_on_the_permissive_side(self):
        """A volume exactly *at* a boundary is inside it, matching the convention
        `overflow.well_overflow` and the formulation solver already use."""
        pcb = self._with_overfill(1.0)
        assert classify_fill(well_void_uL(pcb), pcb).band == FILL_OK
        assert classify_fill(elution_capacity_uL(pcb), pcb).band == FILL_ABOVE_BRIM

    def test_the_reported_bead_height_is_the_one_the_volume_implies(self):
        pcb = self._with_overfill(2.0)
        v = classify_fill(self.VOID + brim_cap_uL(2.44, 1.25), pcb)
        assert v.bead_height_mm == pytest.approx(1.25, abs=1e-3)

    def test_a_board_with_no_well_never_blocks_because_absent_is_not_zero(self):
        """Same posture as `deposit_area_mm2`: nothing to check against is not a
        limit of zero."""
        v = classify_fill(500.0, {"cast_confinement": "sessile"})
        assert v.band == FILL_UNKNOWN
        assert not v.blocks and not v.warns
        assert "nothing to check it against" in v.describe()

    def test_the_board_key_outranks_the_rig_wide_default(self):
        """How far a bead stands proud depends on the wall material, which is a
        property of the board, not of the rig."""
        with patch("softae.config.loader.load",
                   return_value={"deposition": {"permitted_overfill_mm": 3.0}}):
            assert permitted_overfill_mm(self._with_overfill(0.5)) == 0.5
            assert permitted_overfill_mm(self.PCB) == 3.0

    def test_an_unreadable_config_permits_no_overfill_rather_than_assuming_some(self):
        with patch("softae.config.loader.load", side_effect=OSError("gone")):
            assert permitted_overfill_mm(self.PCB) == 0.0


class TestOverflowVerdictBands:
    """The shared guard gains the warn band without changing what `overflows` means."""

    def test_overflows_still_means_past_the_hard_stop(self):
        v = well_overflow(129.0, 128.645, void_uL=118.769)
        assert v.overflows and not v.above_brim

    def test_a_bead_within_the_permitted_head_is_flagged_but_does_not_overflow(self):
        v = well_overflow(125.0, 128.645, void_uL=118.769)
        assert not v.overflows
        assert v.above_brim
        assert v.bead_uL == pytest.approx(6.231, abs=1e-3)

    def test_omitting_the_void_leaves_the_verdict_exactly_as_it_was(self):
        """Every existing caller passes two arguments and must not shift."""
        v = well_overflow(125.0, 128.645)
        assert not v.overflows
        assert not v.above_brim
        assert v.bead_uL == 0.0

    def test_a_sweep_counts_above_brim_points_separately_from_overflows(self):
        points = [{"v": x} for x in (100.0, 125.0, 200.0)]
        result = sweep_overflow(points, lambda p: p["v"], 128.645, void_uL=118.769)
        assert result.n_overflow == 1
        assert result.n_above_brim == 1
        assert result.any_above_brim


class TestNearestElectrode:
    PCB = {"grid": [4, 4], "spacing_mm": [-10, -10]}
    ORIGIN = (43.5, 50.0)

    def test_exact_hit_round_trips(self):
        for ch in (1, 6, 11, 16):
            x, y = electrode_xy_for_channel(self.PCB, ch, *self.ORIGIN)
            assert nearest_electrode(self.PCB, x, y, *self.ORIGIN) == ch

    def test_within_tolerance_snaps(self):
        x, y = electrode_xy_for_channel(self.PCB, 6, *self.ORIGIN)
        # 2 mm off (pitch 10 → default tol 5 mm) still resolves to E6.
        assert nearest_electrode(self.PCB, x + 2.0, y - 1.0, *self.ORIGIN) == 6

    def test_beyond_tolerance_is_none(self):
        # E1 is the corner; move outward (+x,+y) away from every electrode so
        # ~11 mm exceeds the 5 mm half-pitch snap radius → not over a well.
        x, y = electrode_xy_for_channel(self.PCB, 1, *self.ORIGIN)
        assert nearest_electrode(self.PCB, x + 8.0, y + 8.0, *self.ORIGIN) is None

    def test_explicit_tolerance_override(self):
        # 8 mm off the E1 corner (outward): outside default tol, inside tol=10.
        x, y = electrode_xy_for_channel(self.PCB, 1, *self.ORIGIN)
        assert nearest_electrode(self.PCB, x + 8.0, y, *self.ORIGIN) is None
        assert nearest_electrode(
            self.PCB, x + 8.0, y, *self.ORIGIN, tolerance_mm=10.0
        ) == 1

    def test_empty_grid_is_none(self):
        assert nearest_electrode({"grid": [0, 0], "spacing_mm": [1, 1]}, 0.0, 0.0) is None
