"""Pump ↔ loaded-stock binding, and deriving particulate lines from it (P8).

The binding exists because `Chemical.is_particulate` was already fully wired but
nothing recorded *which solution sits on which pump*, so `[purge]
particulate_pumps` had to be hand-maintained in TOML — where it was found wrong.
"""

from __future__ import annotations

import pytest

from softae.core.formulation import (
    Chemical,
    ChemicalCatalog,
    Solution,
    SolutionCatalog,
    SolutionComponent,
)
from softae.core.stock_assignment import (
    PumpLoadout,
    derive_particulate_pumps,
    load_loadout,
    save_loadout,
    solution_is_particulate,
)


@pytest.fixture
def catalogs():
    chem = ChemicalCatalog()
    chem.add(Chemical("Water", "O", density_g_per_mL=1.0))
    chem.add(Chemical("Isopropanol", "CC(O)C", density_g_per_mL=0.786))
    chem.add(Chemical("Fumed silica", "O=[Si]=O", density_g_per_mL=2.65,
                      is_particulate=True))

    sol = SolutionCatalog()
    sol.add(Solution("Silica dispersion", [
        SolutionComponent("Fumed silica", "dep", 1.0, "g"),
        SolutionComponent("Isopropanol", "carrier", 9.0, "mL"),
    ]))
    sol.add(Solution("Clean IPA", [
        SolutionComponent("Isopropanol", "carrier", 10.0, "mL"),
    ]))
    sol.add(Solution("Water rinse", [
        SolutionComponent("Water", "carrier", 10.0, "mL"),
    ]))
    return chem, sol


def _derive(loadout, catalogs, *, pumps=(0, 1, 2), fallback=(1,)):
    chem, sol = catalogs
    return derive_particulate_pumps(
        loadout, chem_catalog=chem, sol_catalog=sol,
        pumps=pumps, fallback=fallback,
    )


class TestParticulateDetection:
    def test_a_solution_containing_a_particulate_is_flagged(self, catalogs):
        chem, sol = catalogs
        assert solution_is_particulate(sol.get("Silica dispersion"), chem)

    def test_a_clean_solution_is_not(self, catalogs):
        chem, sol = catalogs
        assert not solution_is_particulate(sol.get("Clean IPA"), chem)


class TestDerivation:
    def test_the_declared_particulate_line_is_derived(self, catalogs):
        loadout = PumpLoadout({0: "Water rinse", 1: "Silica dispersion",
                               2: "Clean IPA"})
        assert _derive(loadout, catalogs) == (1,)

    def test_multiple_particulate_lines_fall_out_for_free(self, catalogs):
        """`particulate_pumps` was always a list; nothing special is needed."""
        loadout = PumpLoadout({0: "Silica dispersion", 1: "Silica dispersion",
                               2: "Clean IPA"})
        assert _derive(loadout, catalogs) == (0, 1)

    def test_no_particulate_stock_means_no_particulate_lines(self, catalogs):
        loadout = PumpLoadout({0: "Water rinse", 1: "Clean IPA", 2: "Clean IPA"})
        assert _derive(loadout, catalogs) == ()

    def test_an_undeclared_pump_counts_as_particulate(self, catalogs):
        """Partial declaration is the dangerous state.

        The operator has engaged with the mechanism, so silence about one line
        reads as an oversight rather than an assertion that it is clean.
        """
        loadout = PumpLoadout({0: "Clean IPA", 2: "Clean IPA"})     # 1 missing
        assert _derive(loadout, catalogs) == (1,)

    def test_a_solution_missing_from_the_catalog_counts_as_particulate(self, catalogs):
        """The two records have drifted; purge more, not less."""
        loadout = PumpLoadout({0: "Clean IPA", 1: "Ghost solution",
                               2: "Clean IPA"})
        assert _derive(loadout, catalogs) == (1,)

    def test_an_empty_loadout_falls_back_to_config(self, catalogs):
        """Behaviour must not change under an operator who has not opted in.

        Treating every line as particulate would silently raise consumption by
        half again on a rig that was working fine.
        """
        assert _derive(PumpLoadout(), catalogs, fallback=(2,)) == (2,)

    def test_the_fallback_is_not_used_once_anything_is_declared(self, catalogs):
        loadout = PumpLoadout({0: "Clean IPA", 1: "Clean IPA", 2: "Clean IPA"})
        assert _derive(loadout, catalogs, fallback=(0, 1, 2)) == ()


class TestLoadout:
    def test_assigning_and_clearing(self):
        loadout = PumpLoadout()
        loadout.assign(1, "Silica dispersion")
        assert loadout.solution_for(1) == "Silica dispersion"
        loadout.assign(1, None)
        assert loadout.solution_for(1) is None
        assert loadout.is_empty()

    def test_describe_leads_with_the_pump_index(self):
        """Pump IDs are physical bench positions — they lead, always."""
        text = PumpLoadout({1: "Silica dispersion"}).describe()
        assert text.startswith("pump 1:")

    def test_describe_when_nothing_is_declared(self):
        assert "No stocks declared" in PumpLoadout().describe()


class TestPersistence:
    def test_a_loadout_round_trips(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "t.db")
        save_loadout(store, PumpLoadout({0: "Clean IPA", 1: "Silica dispersion"}))

        loaded = load_loadout(store)
        assert loaded.by_pump == {0: "Clean IPA", 1: "Silica dispersion"}

    def test_an_absent_record_reads_as_undeclared(self, tmp_path):
        from softae.core.data_store import DataStore

        assert load_loadout(DataStore(tmp_path / "t.db")).is_empty()

    def test_no_store_is_not_an_error(self):
        assert load_loadout(None).is_empty()
        save_loadout(None, PumpLoadout({0: "x"}))          # must not raise

    def test_a_corrupt_record_reads_as_undeclared(self, tmp_path):
        """Never guess from a damaged record — undeclared is the safe reading."""
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "t.db")
        store._kv_set_text("pump_loadout", "{not json")
        assert load_loadout(store).is_empty()

    def test_clearing_a_pump_persists(self, tmp_path):
        from softae.core.data_store import DataStore

        store = DataStore(tmp_path / "t.db")
        loadout = PumpLoadout({0: "Clean IPA", 1: "Silica dispersion"})
        save_loadout(store, loadout)
        loadout.assign(1, None)
        save_loadout(store, loadout)
        assert load_loadout(store).by_pump == {0: "Clean IPA"}


class TestSchedulerIntegration:
    def test_the_scheduler_uses_the_declared_loadout(self, tmp_path, catalogs):
        """End-to-end: declaring stock changes what actually gets purged."""
        from softae.core.data_store import DataStore
        from softae.core.purge import attach_purge_scheduler
        from softae.drivers.mock_factory import create_mock_manager

        chem, sol = catalogs
        store = DataStore(tmp_path / "t.db")
        # Pump 0 carries the particulate here, contradicting the config default.
        save_loadout(store, PumpLoadout({0: "Silica dispersion",
                                         1: "Clean IPA", 2: "Water rinse"}))

        import softae.core.stock_assignment as sa
        orig = sa.catalogs_from_data_root
        sa.catalogs_from_data_root = lambda: (chem, sol)
        try:
            scheduler = attach_purge_scheduler(
                create_mock_manager(config={}), data_store=store)
        finally:
            sa.catalogs_from_data_root = orig

        assert scheduler is not None
        assert scheduler.settings.particulate_pumps == (0,)
        # And the volumes follow the derivation, not the config.
        assert scheduler.settings.volume_for(0) == scheduler.settings.particulate_uL
        assert scheduler.settings.volume_for(1) == scheduler.settings.other_uL
