"""Twin on the campaign path and persisted thickness (P7.4 / P7.5 / P7.6).

Before this the twin was reachable only from the Deposition panel, so a campaign
could neither predict the film it was about to cast nor record the thickness it
produced — and the conductivity path took a hand-typed `t`.
"""

from __future__ import annotations

import pytest

from softae.core.deposition import (
    DEFAULT_EVAPORATION_PCT,
    WellGeometry,
    evaporation_pct,
)


class TestEquivalentGeometry:
    """Boards declare an area and a capacity, not a cylinder."""

    def test_area_and_capacity_are_reproduced_exactly(self):
        w = WellGeometry.from_board(4.0, 120.0)
        assert w.area_mm2 == pytest.approx(4.0)
        assert w.capacity_uL == pytest.approx(120.0)

    def test_it_works_for_any_plausible_board(self):
        for area, cap in ((4.0, 50.0), (12.5, 120.0), (0.25, 3.0)):
            w = WellGeometry.from_board(area, cap)
            assert w.area_mm2 == pytest.approx(area)
            assert w.capacity_uL == pytest.approx(cap)

    def test_non_positive_inputs_are_refused(self):
        with pytest.raises(ValueError):
            WellGeometry.from_board(0.0, 120.0)
        with pytest.raises(ValueError):
            WellGeometry.from_board(4.0, 0.0)


class TestEvaporationConfig:
    def test_the_default_is_total_loss(self):
        """100 % keeps dry thickness deterministic without a solvent model —
        which is what lets a dry ThicknessTarget reduce to a volume target."""
        assert DEFAULT_EVAPORATION_PCT == 100.0

    def test_config_supplies_the_value(self):
        assert evaporation_pct({"evaporation_pct": 60.0}) == 60.0

    def test_an_out_of_range_value_falls_back(self):
        assert evaporation_pct({"evaporation_pct": 140.0}) == DEFAULT_EVAPORATION_PCT
        assert evaporation_pct({"evaporation_pct": -1.0}) == DEFAULT_EVAPORATION_PCT

    def test_junk_falls_back(self):
        assert evaporation_pct({"evaporation_pct": "lots"}) == DEFAULT_EVAPORATION_PCT

    def test_the_shipped_config_parses(self):
        assert 0.0 <= evaporation_pct() <= 100.0


class TestPersistedThickness:
    def _store(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        self.run_id = store.start_run("wf")
        return store

    def test_a_recorded_thickness_round_trips(self, tmp_path):
        store = self._store(tmp_path)
        store.record_formulation(self.run_id, 3, total_uL=10.0,
                                 predicted_thickness_um=2.5)
        assert store.predicted_thickness_um(self.run_id, 3) == pytest.approx(2.5)

    def test_absent_is_none_not_zero(self, tmp_path):
        """The conductivity path divides by t — absent must not read as zero."""
        store = self._store(tmp_path)
        store.record_formulation(self.run_id, 3, total_uL=10.0)
        assert store.predicted_thickness_um(self.run_id, 3) is None

    def test_an_unknown_channel_is_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store.predicted_thickness_um(self.run_id, 99) is None

    def test_the_latest_cast_wins(self, tmp_path):
        """Matches how re-casting a well behaves elsewhere in the system."""
        store = self._store(tmp_path)
        store.record_formulation(self.run_id, 3, predicted_thickness_um=1.0)
        store.record_formulation(self.run_id, 3, predicted_thickness_um=4.0)
        assert store.predicted_thickness_um(self.run_id, 3) == pytest.approx(4.0)

    def test_legacy_databases_are_migrated(self, tmp_path):
        """An older DB predates the column; opening it must add, not fail."""
        import sqlite3

        from softae.core.data_store import DataStore

        proj = tmp_path / "proj"
        (proj / "db").mkdir(parents=True)
        conn = sqlite3.connect(proj / "db" / "softae.db")
        conn.execute(
            "CREATE TABLE formulations (formulation_id INTEGER PRIMARY KEY, "
            "run_id TEXT, channel INTEGER)"
        )
        conn.commit()
        conn.close()

        store = DataStore(proj)
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(formulations)").fetchall()}
        assert "predicted_thickness_um" in cols


class TestTheTwinRunsOnBothCompositionContexts:
    """P7.5 wired the twin onto the campaign path — for one of the two contexts.

    A campaign carries either a fixed three-stock :class:`FormulationContext` (the
    PEO/LiCl/silica route the GUI builds) or a :class:`GeneralFormulation` with an
    arbitrary stock dict. ``_trial_stock_volumes`` handles both, but only the general
    branch was ever exercised: the fixed branch read ``ctx.stocks``, which did not
    exist, so **every fixed-context campaign raised ``AttributeError`` the moment it
    asked the twin to predict a thickness**.

    It was invisible because the one caller that mattered,
    ``_record_trial_formulations``, is deliberately best-effort — a bookkeeping
    failure must not stop a cast. So the cast proceeded, the exception was swallowed
    into ``trial_formulation_record_failed``, and the campaign recorded *no
    formulation row at all* — not merely a missing thickness. The analysis tab then
    joined on ``(run_id, channel)`` and found nothing, which is the same symptom as
    never having wired P7.6 in.
    """

    def _specs(self):
        from softae.core.autonomous_wiring import CampaignSpec
        from tests.test_autonomous_composition import SPACE, _context, _general

        common = dict(name="twin", channels=(21,), pcb_name="SoftAE_EIS_4Stripe",
                      parameter_space=SPACE, pump_ids=(0, 1, 2))
        return {
            "fixed": CampaignSpec(formulation=_context(), **common),
            "general": CampaignSpec(general_formulation=_general(), **common),
        }

    @pytest.mark.parametrize("context", ["fixed", "general"])
    def test_the_twin_predicts_a_thickness_for_a_composition_campaign(self, context):
        from softae.core.autonomous_wiring import simulate_trial
        from tests.test_autonomous_composition import POINT

        twin = simulate_trial(self._specs()[context], POINT)
        assert twin is not None, "the twin must speak to a composition campaign"
        assert twin.final_thickness_um > 0

    def test_both_contexts_agree_because_they_describe_the_same_stocks(self):
        # The two routes differ in how targets are declared, not in what is cast, so
        # a divergence here means one of them is reading the stocks wrongly.
        from softae.core.autonomous_wiring import simulate_trial
        from tests.test_autonomous_composition import POINT

        specs = self._specs()
        fixed = simulate_trial(specs["fixed"], POINT)
        general = simulate_trial(specs["general"], POINT)
        assert fixed.final_thickness_um == pytest.approx(
            general.final_thickness_um, rel=0.05)

    def test_a_volume_mode_campaign_still_gets_no_twin(self):
        # Not a failure: without stock identity there is nothing to elute, so the
        # twin has nothing to say beyond the total the overflow guard already checks.
        from softae.core.autonomous_wiring import CampaignSpec, simulate_trial

        spec = CampaignSpec(
            name="vol", channels=(21,), pcb_name="SoftAE_EIS_4Stripe",
            parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
            vol_params=("vol_p0",), pump_ids=(0,))
        assert simulate_trial(spec, {"vol_p0": 10.0}) is None

    def test_a_fixed_context_exposes_its_stocks_keyed_as_the_plan_reports_them(self):
        # per_stock_uL is keyed by Solution.name, so the accessor must match or the
        # elution lookup silently drops stocks it cannot find.
        from softae.core.formulation import plan_formulation
        from tests.test_autonomous_composition import POINT, _context

        ctx = _context()
        plan = plan_formulation(POINT, ctx)
        assert set(plan.per_stock_uL) <= set(ctx.stocks)


class TestTheSharedTwinSeam:
    """P.12 — the volumes→film half is one function, entered from both sides.

    ``simulate_trial`` was two halves welded together: a genuinely BO-specific
    suggestion→volumes solve, and a generic volumes→film tail. The HT tab needs
    only the tail, and re-implementing it would leave two places free to disagree
    about the deposit area, the capacity, the elution split or the drying
    assumption — the way three marshallers diverged before P2.2 collapsed them.
    """

    def _spec(self):
        from softae.core.autonomous_wiring import CampaignSpec
        from tests.test_autonomous_composition import SPACE, _general

        return CampaignSpec(name="twin", channels=(21,),
                            pcb_name="SoftAE_EIS_4Stripe", parameter_space=SPACE,
                            pump_ids=(0, 1, 2), general_formulation=_general())

    def test_simulate_cast_reproduces_simulate_trial_exactly_so_the_extraction_moves_no_number(
            self):
        """A pure refactor: every field of the result must be identical.

        Not just the thickness — the whole ``WellDepositionResult``, because the
        extraction touched the area, the capacity and the carrier keys as well,
        and a drift in any of them would show up first in a field nobody thought
        to assert on.
        """
        from softae.core.autonomous_wiring import (
            _trial_stock_volumes,
            campaign_well_capacity_uL,
            simulate_cast,
            simulate_trial,
        )
        from tests.test_autonomous_composition import POINT

        spec = self._spec()
        per_stock_uL, stocks, catalog = _trial_stock_volumes(spec, POINT)
        direct = simulate_cast(per_stock_uL, stocks, catalog,
                               pcb_name=spec.pcb_name,
                               capacity_uL=campaign_well_capacity_uL(spec))
        assert direct == simulate_trial(spec, POINT)
        assert direct.final_thickness_um > 0      # the comparison is not None == None

    def test_an_omitted_capacity_falls_back_to_the_board_not_to_no_budget(self):
        """``capacity_uL`` is an *override*, for a campaign whose gate enforces one.

        A caller with no budget of its own — the HT tab — must get the board's
        own well capacity, not a missing one, or the twin would decline on every
        board that declares a capacity in the ordinary way.
        """
        from softae.core.autonomous_wiring import _trial_stock_volumes, simulate_cast
        from softae.core.deposition_steps import resolve_pcb
        from softae.core.geometry import well_capacity_uL
        from tests.test_autonomous_composition import POINT

        per_stock_uL, stocks, catalog = _trial_stock_volumes(self._spec(), POINT)
        twin = simulate_cast(per_stock_uL, stocks, catalog,
                             pcb_name="SoftAE_EIS_4Stripe")
        assert twin is not None
        assert twin.well.capacity_uL == pytest.approx(
            well_capacity_uL(resolve_pcb("SoftAE_EIS_4Stripe")[1]))

    def test_a_board_with_no_deposit_area_still_declines(self):
        """The decline moved into ``simulate_cast``; it must still happen there."""
        from softae.core.autonomous_wiring import _trial_stock_volumes, simulate_cast
        from tests.test_autonomous_composition import POINT

        per_stock_uL, stocks, catalog = _trial_stock_volumes(self._spec(), POINT)
        assert simulate_cast(per_stock_uL, stocks, catalog,
                             pcb_name="SoftAE_IDE_EIS") is None


class TestTheAreaIsRecordedBesideTheThicknessItProduced:
    """P.7 — a thickness is a quotient, and the row stored only the quotient.

    The 4-stripe board's deposit area moved from 4.0 mm² to 18.704 mm² when P7.2
    made the well authoritative rather than the inter-electrode rectangle. Rows
    written either side of that differ by 4.676× in one column with one unit, and
    nothing in the row says which — a formulation row does not record its board.
    The ambiguity is unrecoverable after the fact and deepens with every campaign,
    so the denominator is now written beside the quotient at record time.
    """

    PCB = "SoftAE_EIS_4Stripe"

    def _store(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "proj")
        return store, store.start_run("campaign")

    def _row(self, store, run_id, channel):
        # tuple(): the store sets `row_factory = sqlite3.Row`, which does not
        # compare equal to a plain tuple.
        return tuple(store._conn.execute(
            "SELECT predicted_thickness_um, deposit_area_mm2, thickness_method "
            "FROM formulations WHERE run_id = ? AND channel = ?",
            (run_id, int(channel))).fetchone())

    def _legacy_db(self, tmp_path, *, with_row=False):
        """A project whose `formulations` table predates both new columns."""
        import sqlite3

        proj = tmp_path / "proj"
        (proj / "db").mkdir(parents=True)
        conn = sqlite3.connect(proj / "db" / "softae.db")
        conn.execute(
            "CREATE TABLE formulations (formulation_id INTEGER PRIMARY KEY, "
            "run_id TEXT, channel INTEGER, predicted_thickness_um REAL)"
        )
        if with_row:
            conn.execute(
                "INSERT INTO formulations (run_id, channel, predicted_thickness_um) "
                "VALUES ('old_run', 3, 534.6)"
            )
        conn.commit()
        conn.close()
        return proj

    def test_a_legacy_formulations_table_gains_both_columns_on_open(self, tmp_path):
        """An older DB predates both columns; opening it must add, not fail.

        `_DDL` alone would leave every existing project unable to write a
        provenance row at all, because `CREATE TABLE IF NOT EXISTS` is a no-op on
        a table that already exists.
        """
        from softae.core.data_store import DataStore

        store = DataStore(self._legacy_db(tmp_path))
        cols = {r[1] for r in store._conn.execute(
            "PRAGMA table_info(formulations)").fetchall()}
        assert {"deposit_area_mm2", "thickness_method"} <= cols

    def test_migrating_twice_is_a_no_op(self, tmp_path):
        """Every DataStore construction re-runs the migration.

        `ALTER TABLE ... ADD COLUMN` on an existing column is an error, so a
        migration that did not check `PRAGMA table_info` first would make the
        *second* open of any project fail — which is every open after the upgrade.
        """
        from softae.core.data_store import DataStore

        proj = self._legacy_db(tmp_path)
        DataStore(proj).close()
        store = DataStore(proj)
        cols = [r[1] for r in store._conn.execute(
            "PRAGMA table_info(formulations)").fetchall()]
        # A list, not a set: a second ADD COLUMN would have to be caught here, and a
        # set would silently absorb the duplicate it is meant to detect.
        assert cols.count("deposit_area_mm2") == 1
        assert cols.count("thickness_method") == 1

    def test_nothing_backfills_a_historical_row_with_a_guessed_area(self, tmp_path):
        """A pre-existing row must come out of the migration with NULL on both.

        Writing 4.0 mm² into it would manufacture exactly the false comparability
        this task exists to prevent: we do not know which board it was cast on, and
        a guess is indistinguishable from a record once it is in the column.
        """
        from softae.core.data_store import DataStore

        store = DataStore(self._legacy_db(tmp_path, with_row=True))
        assert self._row(store, "old_run", 3) == (534.6, None, None)

    def test_a_campaign_cast_records_the_area_the_twin_divided_by(self, tmp_path):
        """The stored area must be the twin's own denominator, not a lookalike.

        It is re-derived from config rather than read off `twin.well`, so this is
        what proves the two routes agree — an area that did *not* divide this row's
        thickness would be worse than none, because it reads as provenance.
        """
        from softae.core.autonomous_wiring import (
            CampaignSpec,
            _record_trial_formulations,
            simulate_trial,
        )
        from tests.test_autonomous_composition import POINT, SPACE, _general

        spec = CampaignSpec(name="twin", channels=(21,), pcb_name=self.PCB,
                            parameter_space=SPACE, pump_ids=(0, 1, 2),
                            general_formulation=_general())
        store, run_id = self._store(tmp_path)
        _record_trial_formulations(spec, [POINT], [21],
                                   data_store=store, run_id=run_id)

        twin = simulate_trial(spec, POINT)
        thickness_um, area_mm2, method = self._row(store, run_id, 21)
        assert method == "predicted"
        assert area_mm2 == pytest.approx(twin.well.area_mm2)
        assert thickness_um * area_mm2 / 1000.0 == pytest.approx(
            twin.final_volume_uL)

    def test_a_declined_twin_records_the_area_as_unavailable_rather_than_omitting_it(
            self, tmp_path):
        """A volume-mode campaign has no composition, so the twin declines.

        `'unavailable'`, not NULL: the twin *was* asked and had nothing to say, and
        that is a recorded fact distinct from a row nobody ever asked on behalf of.
        The area survives regardless — `simulate_trial` collapses "no area" and "no
        capacity" into one `None`, so re-deriving is what keeps a perfectly
        well-known area from being lost to a decline it had no part in.
        """
        from softae.core.autonomous_wiring import (
            CampaignSpec,
            _record_trial_formulations,
            simulate_trial,
        )

        spec = CampaignSpec(
            name="vol", channels=(21,), pcb_name=self.PCB,
            parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
            vol_params=("vol_p0",), pump_ids=(0,))
        params = {"vol_p0": 10.0}
        assert simulate_trial(spec, params) is None

        store, run_id = self._store(tmp_path)
        _record_trial_formulations(spec, [params], [21],
                                   data_store=store, run_id=run_id)

        thickness_um, area_mm2, method = self._row(store, run_id, 21)
        assert thickness_um is None
        assert method == "unavailable"
        assert area_mm2 == pytest.approx(18.7038, rel=1e-3)

    def test_a_sessile_board_records_a_null_area_not_the_string_unavailable(
            self, tmp_path):
        """The one case where the two absences have to be told apart on write.

        `SoftAE_IDE_EIS` declares `cast_confinement = "sessile"`: no wells, so the
        wetted footprint is set by volume and contact angle and *nothing on the
        board predicts it*. `deposit_area_mm2` therefore returns `None` rather than
        falling through to the inter-electrode rectangle, which would describe the
        gap between two stripes and bear no relation to where the droplet sat.

        So the two columns must disagree here, and only here. The twin declined for
        want of a capacity, which is a fact the campaign *asked about* -- that is
        `'unavailable'`. The area was never established at all, by anyone, ever --
        that is NULL. Writing 'unavailable' into the area column, or 0.0, or the
        4.0 mm² rectangle, would each turn "we have no idea" into a claim; P.11's
        read guard then keys off that column, so a fabricated value there would
        hand the objective a thickness with an invented basis.

        The class-level `PCB` is overridden inline: every other test here needs a
        board that *has* an area, and this one needs the board that has none.
        """
        from softae.core.autonomous_wiring import (
            CampaignSpec,
            _record_trial_formulations,
        )
        from softae.core.deposition_steps import resolve_pcb
        from softae.core.geometry import deposit_area_mm2

        assert deposit_area_mm2(resolve_pcb("SoftAE_IDE_EIS")[1]) is None, (
            "the premise: a sessile board declares no deposit area")

        spec = CampaignSpec(
            name="vol", channels=(21,), pcb_name="SoftAE_IDE_EIS",
            parameter_space={"vol_p0": {"type": "float", "low": 5.0, "high": 30.0}},
            vol_params=("vol_p0",), pump_ids=(0,))
        store, run_id = self._store(tmp_path)
        _record_trial_formulations(spec, [{"vol_p0": 10.0}], [21],
                                   data_store=store, run_id=run_id)

        thickness_um, area_mm2, method = self._row(store, run_id, 21)
        assert area_mm2 is None
        assert method == "unavailable"
        assert thickness_um is None
