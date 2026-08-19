"""GUI tests for the Live BO Campaign tab (uses the session ``qapp`` fixture).

Kept fast: no full mock campaign runs in the default suite.  The worker wiring
is exercised by monkeypatching ``run_autonomous_campaign``; a real end-to-end
smoke is marked ``slow`` and kept tiny.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from softae.core.autonomous_wiring import CampaignResult, CampaignSpec
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs import tab_bo_live
from softae.gui.tabs._bo_base import BOTabBase
from softae.gui.tabs.tab_bo_live import LiveBOCampaignTab
from softae.gui.tabs.tab_bo_simulator import BOSimulatorTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


# ── Construction & config ──────────────────────────────────────────────────


def test_tab_constructs(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    assert tab is not None
    spec = tab._build_config()
    assert isinstance(spec, CampaignSpec)
    assert spec.optimizer == "bayesian"
    # default two-parameter composition space
    assert set(spec.parameter_space) == {"vol_p0", "vol_p1"}
    assert spec.parameter_space["vol_p0"]["type"] == "float"
    assert spec.vol_params == ("vol_p0", "vol_p1")
    assert spec.pump_ids == (0, 1)
    # no priors by default
    assert spec.prior_mean is None
    assert spec.seed_observations == ()


def test_build_config_carries_priors(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    tab._combo_prior.setCurrentText("linear (demo)")
    tab._seed_observations = [({"vol_p0": 10.0, "vol_p1": 20.0}, 0.7)]
    spec = tab._build_config()
    assert callable(spec.prior_mean)
    assert spec.seed_observations == (({"vol_p0": 10.0, "vol_p1": 20.0}, 0.7),)


def test_build_config_carries_acquisition_and_kappa(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    tab._combo_acq.setCurrentText("ei")
    tab._spin_kappa.setValue(3.5)
    spec = tab._build_config()
    assert spec.acquisition == "ei"
    assert spec.kappa == 3.5


def test_build_config_carries_batch_toggle(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    assert tab._build_config().batch is False  # default off
    tab._chk_batch.setChecked(True)
    assert tab._build_config().batch is True


def test_build_config_carries_batch_strategy(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    assert tab._build_config().batch_strategy == "constant_liar"  # default
    tab._chk_batch.setChecked(True)
    tab._combo_batch_strategy.setCurrentText("kriging_believer")
    assert tab._build_config().batch_strategy == "kriging_believer"
    # The "(planned)" annotation is stripped to the bare registry key.
    tab._combo_batch_strategy.setCurrentText("botorch_mc (planned)")
    assert tab._build_config().batch_strategy == "botorch_mc"


def test_batch_strategy_combo_enabled_with_batch(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    assert not tab._combo_batch_strategy.isEnabled()  # disabled until batch on
    tab._chk_batch.setChecked(True)
    assert tab._combo_batch_strategy.isEnabled()


def test_build_config_carries_electrode_board(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    # Board exchange is on by default (electrodes are single-use).
    spec = tab._build_config()
    assert spec.electrode_capacity == 32
    assert spec.equilibration_s == 60.0
    # Disabling it returns to unbounded fixed-channel behavior.
    tab._chk_board.setChecked(False)
    assert tab._build_config().electrode_capacity is None


def test_default_board_is_4stripe_and_capacity_is_derived(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    spec = tab._build_config()
    assert spec.pcb_name == "SoftAE_EIS_4Stripe"   # 4Stripe is the default board
    assert spec.electrode_capacity == 32           # derived from its electrode count


def test_switching_board_reseeds_capacity(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    # Select the 16-electrode IDE board → capacity re-seeds to 16.
    idx = tab._combo_pcb.findData("SoftAE_IDE_EIS")
    assert idx >= 0
    tab._combo_pcb.setCurrentIndex(idx)
    assert tab._spin_capacity.value() == 16
    spec = tab._build_config()
    assert spec.pcb_name == "SoftAE_IDE_EIS"
    assert spec.electrode_capacity == 16


def test_board_check_gate_marshals_and_returns_decision(qapp, manager):
    from softae.core.autonomous_loop import BoardCheck
    tab = LiveBOCampaignTab(manager)
    tab._sig_board_check.disconnect(tab._on_board_check_prompt)
    for want in (BoardCheck.FRESH, BoardCheck.RESUME, BoardCheck.CANCEL):
        conn = tab._sig_board_check.connect(
            lambda bid, occ, w=want: (setattr(tab, "_board_check_decision", w),
                                      tab._board_check_event.set())
        )
        assert tab._board_check_gate(0, {1, 2}) is want
        tab._sig_board_check.disconnect(conn)


def test_board_exchange_gate_marshals_and_returns_decision(qapp, manager, monkeypatch):
    from softae.core.autonomous_loop import BoardDecision
    tab = LiveBOCampaignTab(manager)
    # Simulate the GUI-thread modal answering "Yes" (proceed) by driving the slot
    # directly when the prompt signal fires.
    tab._sig_board_prompt.disconnect(tab._on_board_prompt)
    tab._sig_board_prompt.connect(
        lambda b: (setattr(tab, "_board_decision", BoardDecision.PROCEED),
                   tab._board_event.set())
    )
    assert tab._board_exchange_gate(1) is BoardDecision.PROCEED

    tab._sig_board_prompt.disconnect()
    tab._sig_board_prompt.connect(
        lambda b: (setattr(tab, "_board_decision", BoardDecision.CANCEL),
                   tab._board_event.set())
    )
    assert tab._board_exchange_gate(2) is BoardDecision.CANCEL


def test_linear_demo_prior_is_callable_and_numeric(qapp, manager):
    tab = LiveBOCampaignTab(manager)
    assert tab._selected_prior_mean() is None
    tab._combo_prior.setCurrentText("linear (demo)")
    pm = tab._selected_prior_mean()
    assert callable(pm)
    assert isinstance(pm({"vol_p0": 1.0, "vol_p1": 2.0}), float)


def test_load_seeds_json_populates_seed_observations(qapp, manager, tmp_path):
    seeds = [
        [{"vol_p0": 12.0, "vol_p1": 8.0}, 0.42],
        [{"vol_p0": 20.0, "vol_p1": 20.0}, 0.91],
    ]
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(seeds), encoding="utf-8")

    tab = LiveBOCampaignTab(manager)
    n = tab.load_seeds_from_file(str(path))
    assert n == 2
    spec = tab._build_config()
    assert len(spec.seed_observations) == 2
    assert spec.seed_observations[0][0] == {"vol_p0": 12.0, "vol_p1": 8.0}
    assert spec.seed_observations[1][1] == 0.91


def test_config_roundtrip_panel_state(qapp, manager):
    src = LiveBOCampaignTab(manager)
    src._le_name.setText("demo1")
    src._le_channels.setText("2, 4")
    src._spin_budget.setValue(15)
    src._combo_objdir.setCurrentText("minimize")
    src._combo_prior.setCurrentText("linear (demo)")
    state_json = json.dumps(src._panel_state())

    dst = LiveBOCampaignTab(manager)
    dst._populate_from_config(dst._config_from_json(state_json))
    assert dst._le_name.text() == "demo1"
    assert dst._spin_budget.value() == 15
    assert dst._combo_objdir.currentText() == "minimize"
    assert dst._combo_prior.currentText() == "linear (demo)"
    spec = dst._build_config()
    assert spec.channels == (2, 4)


# ── Worker wiring (monkeypatched — no real campaign) ───────────────────────


def test_run_campaign_wiring(qapp, manager, monkeypatch):
    events: list[dict] = []

    async def fake_run(spec, **kwargs):
        on_event = kwargs["on_event"]
        on_event({"type": "run_started", "run_id": "r1", "spec": spec.name})
        on_event({
            "type": "result", "iteration": 1,
            "params": {"vol_p0": 10.0, "vol_p1": 10.0}, "objective": 0.5,
        })
        events.append({"manager": kwargs.get("manager")})
        return CampaignResult(
            run_id="r1", best_params={"vol_p0": 10.0, "vol_p1": 10.0},
            best_objective=0.5, n_trials=1, final_state="CONVERGED",
            converged=True, history=[],
        )

    # Patched at the source module: since P2.4 the campaign is launched from
    # AutonomousRunMixin._execute_campaign, which imports this at call time.
    import softae.core.autonomous_wiring as wiring

    monkeypatch.setattr(wiring, "run_autonomous_campaign", fake_run)

    tab = LiveBOCampaignTab(manager)
    spec = tab._build_config()
    tab._run_campaign(spec)  # synchronous on the test thread

    assert events and events[0]["manager"] is manager
    assert tab._result is not None
    assert tab._result.best_objective == 0.5
    # convergence buffers received the result event
    assert tab._xs == [1.0]
    assert tab._primary_series == [0.5]


# ── Head-position start-gate ───────────────────────────────────────────────


def test_on_run_aborts_when_head_declined(qapp, manager, monkeypatch):
    tab = LiveBOCampaignTab(manager)
    monkeypatch.setattr(tab, "_verify_head_position", lambda: False)
    started: list = []
    monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: started.append(a))
    tab._on_run()
    assert started == []  # campaign never started


def test_verify_head_position_delegates(qapp, manager, monkeypatch):
    from softae.gui.widgets import head_check_dialog as hcd
    tab = LiveBOCampaignTab(manager)
    seen: dict = {}

    def fake(parent, mgr, *, context=""):
        seen["mgr"] = mgr
        seen["context"] = context
        return True

    monkeypatch.setattr(hcd, "verify_head_before_run", fake)
    assert tab._verify_head_position() is True
    assert seen["mgr"] is manager
    assert "campaign" in seen["context"]


# ── Single occupancy, and what a refusal costs (S5.I) ──────────────────────


class TestSingleOccupancyRefusal:
    """A second campaign is refused outright — and the refusal preserves the setup.

    What the tab adds over the harness (``test_autonomous_run_mixin.py``) is the
    thing that decides whether the refusal is *safe*: a composition campaign
    cannot be written to a spec file without silently becoming a raw-volume one,
    so it must be refused a relaunch command, and the panel state — which is
    lossless — must be what the operator is pointed at instead.
    """

    @staticmethod
    def _tab(qapp, manager, tmp_path):
        from types import SimpleNamespace

        return LiveBOCampaignTab(
            manager, data_store=SimpleNamespace(project_dir=tmp_path))

    @staticmethod
    def _hold_rig(monkeypatch) -> list[str]:
        from softae.core.run_lock import RunLock

        lock = RunLock(pid=4242, what="campaign:other:20260817T090000Z_other",
                       started_at="2026-08-17T09:00:00+00:00",
                       host="another-host", log_path=r"C:\proj\runs\x")
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *a: lock)
        monkeypatch.setattr("softae.core.run_lock.rig_is_simulated", lambda m: False)
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda parent, title, text, *a, **k: shown.append(text)))
        return shown

    def test_on_run_with_a_foreign_lock_starts_no_campaign(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        tab = self._tab(qapp, manager, tmp_path)
        self._hold_rig(monkeypatch)
        started: list = []
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: started.append(a))
        tab._on_run()
        assert started == []

    def test_on_run_with_a_foreign_lock_never_reaches_the_head_gate(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The head gate prompts, and may retract — a refused launch moves nothing."""
        tab = self._tab(qapp, manager, tmp_path)
        self._hold_rig(monkeypatch)
        asked: list[bool] = []
        monkeypatch.setattr(tab, "_verify_head_position",
                            lambda *a, **k: asked.append(True) or True)
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: None)
        tab._on_run()
        assert asked == []

    def test_a_refused_volume_campaign_is_offered_a_relaunch_command(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        tab = self._tab(qapp, manager, tmp_path)
        shown = self._hold_rig(monkeypatch)
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: None)
        tab._on_run()

        rejected = tmp_path / "rejected"
        assert len(list(rejected.glob("*.toml"))) == 1
        assert "softae-campaign run " in shown[0]

    def test_a_refused_composition_campaign_is_offered_no_relaunch_command(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """B-i: the written file would search composition axes as raw µL volumes."""
        tab = self._tab(qapp, manager, tmp_path)
        monkeypatch.setattr(
            tab, "_load_stocks",
            lambda: ({"PEO": object(), "LiCl": object()}, {"PEO": 0, "LiCl": 1},
                     object()))
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Molar ratio", a="PEO", b="LiCl", low="5", high="40")
        shown = self._hold_rig(monkeypatch)
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: None)
        tab._on_run()

        rejected = tmp_path / "rejected"
        assert list(rejected.glob("*.toml")) == []
        assert len(list(rejected.glob("*.json"))) == 1
        assert "softae-campaign run" not in shown[0]
        assert "general_formulation" in shown[0]

    def test_the_preserved_panel_state_restores_the_refused_campaign(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The whole point: a refusal costs a file, not the setup."""
        tab = self._tab(qapp, manager, tmp_path)
        monkeypatch.setattr(
            tab, "_load_stocks",
            lambda: ({"PEO": object(), "LiCl": object()}, {"PEO": 0, "LiCl": 1},
                     object()))
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Molar ratio", a="PEO", b="LiCl", low="5", high="40")
        tab._le_name.setText("phase_map")
        tab._spin_budget.setValue(23)
        self._hold_rig(monkeypatch)
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: None)
        tab._on_run()

        saved = next((tmp_path / "rejected").glob("*.json"))
        restored = LiveBOCampaignTab(manager)
        restored._populate_from_config(
            restored._config_from_json(saved.read_text(encoding="utf-8")))

        assert restored._le_name.text() == "phase_map"
        assert restored._spin_budget.value() == 23
        assert restored._search_mode() == "composition"
        axes = restored._axes_editor.axes()
        assert (axes[0].a, axes[0].b, axes[0].low, axes[0].high) == (
            "PEO", "LiCl", 5.0, 40.0)

    def test_a_free_rig_runs_the_campaign_as_before(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The refusal must be reachable only from a live foreign lock."""
        tab = self._tab(qapp, manager, tmp_path)
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *a: None)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        started: list = []
        monkeypatch.setattr(tab, "_start_worker", lambda *a, **k: started.append(a))
        tab._on_run()
        assert started and not (tmp_path / "rejected").exists()


# ── Concurrency invariant (Simulator + Live share a base, own their state) ──


def test_concurrency_state_is_per_instance(qapp, manager):
    sim = BOSimulatorTab()
    live = LiveBOCampaignTab(manager)

    # No run active on either.
    assert sim._thread is None and live._thread is None
    # Abort flags are independent objects/values.
    assert sim._abort_requested is False and live._abort_requested is False

    # Series buffers are distinct list objects (not shared class state).
    assert sim._xs is not live._xs
    assert sim._primary_series is not live._primary_series
    assert sim._secondary_series is not live._secondary_series

    # Setting state on one must not leak to the other.
    live._abort_requested = True
    assert sim._abort_requested is False
    sim._xs.append(1.0)
    assert live._xs == []


def test_base_has_no_mutable_class_level_data():
    """The shared base must hold NO mutable list/dict data attributes.

    Qt ``Signal`` descriptors and methods are fine; a class-level list/dict
    would couple concurrent instances, which is exactly what we forbid.
    """
    offenders = {
        name: type(val).__name__
        for name, val in vars(BOTabBase).items()
        if isinstance(val, (list, dict, set))
    }
    assert offenders == {}, f"mutable class-level data on BOTabBase: {offenders}"


@pytest.mark.slow
def test_real_mock_campaign_smoke(qapp, manager):
    """Tiny end-to-end run over the mock manager (budget=1, time_scale=0)."""
    import asyncio

    from softae.core.autonomous_wiring import run_autonomous_campaign

    asyncio.run(manager.connect_all())
    tab = LiveBOCampaignTab(manager)
    tab._spin_budget.setValue(1)
    tab._spin_timescale.setValue(0.0)
    spec = tab._build_config()
    result = asyncio.run(
        run_autonomous_campaign(
            spec, manager=manager, data_store=None,
            objective_extractor=None, on_event=tab._on_campaign_event,
        )
    )
    assert result.n_trials >= 1


# ── Objective direction is derived, not chosen ──────────────────────────────


class TestObjectiveDirectionControl:
    """The tab shipped a ``maximize``/``minimize`` combo defaulting to *maximize*.

    Its default was wrong for every campaign this tab can actually build: the tab
    casts by raw pump volumes, so there is no dry thickness, so the objective is
    mean |Z| — which must be **minimised**. Every live campaign launched from here
    was therefore steered toward the *most* resistive material on the board.

    The direction is not a preference, so the control now defaults to deriving it.
    """

    def test_the_default_defers_to_the_campaign_rather_than_guessing(self, qapp, manager):
        tab = LiveBOCampaignTab(manager)
        assert tab._combo_objdir.currentText() == "auto"
        assert tab._build_config().objective == "auto"

    def test_a_volume_campaign_from_this_tab_resolves_to_minimising_impedance(
        self, qapp, manager
    ):
        from softae.core.autonomous_wiring import resolve_direction

        tab = LiveBOCampaignTab(manager)
        direction, metric = resolve_direction(tab._build_config())
        assert (direction, metric) == ("minimize", "mean_abs_z")

    def test_the_old_default_is_now_refused_rather_than_silently_run(self, qapp, manager):
        from softae.core.autonomous_wiring import resolve_direction
        from softae.errors import CampaignError

        tab = LiveBOCampaignTab(manager)
        tab._combo_objdir.setCurrentText("maximize")
        with pytest.raises(CampaignError, match="contradicts"):
            resolve_direction(tab._build_config())

    def test_an_explicit_direction_that_agrees_is_still_allowed(self, qapp, manager):
        from softae.core.autonomous_wiring import resolve_direction

        tab = LiveBOCampaignTab(manager)
        tab._combo_objdir.setCurrentText("minimize")
        assert resolve_direction(tab._build_config())[0] == "minimize"

    def test_the_saved_state_of_an_older_session_restores_to_auto(self, qapp, manager):
        # A state file written before the combo had an "auto" entry must not pin a
        # direction the campaign never chose.
        tab = LiveBOCampaignTab(manager)
        tab._populate_from_config({})
        assert tab._combo_objdir.currentText() == "auto"


# ── Search mode: raw volumes ↔ composition targets ──────────────────────────


class TestSearchMode:
    """The tab can ask either of the two campaign questions.

    Raw volumes is the easier search — feasibility is native — but it has no stock
    identity, so no dry thickness and no conductivity. Composition targets cost an
    up-front solve per suggestion and buy a predicted thickness, which is what makes
    σ the objective. Both are legitimate; the tab now offers both.
    """

    def test_raw_volumes_stays_the_default(self, qapp, manager):
        tab = LiveBOCampaignTab(manager)
        assert tab._search_mode() == "volumes"
        assert tab._build_config().general_formulation is None

    def test_switching_mode_swaps_which_editor_is_shown(self, qapp, manager):
        tab = LiveBOCampaignTab(manager)
        tab._combo_search_mode.setCurrentIndex(1)
        assert tab._search_mode() == "composition"
        assert tab._axes_editor.isVisibleTo(tab)
        assert not tab._tbl_params.isVisibleTo(tab)

    def test_composition_mode_without_stocks_explains_itself_rather_than_crashing(
        self, qapp, manager, monkeypatch
    ):
        tab = LiveBOCampaignTab(manager)
        monkeypatch.setattr(tab, "_load_stocks", lambda: ({}, {}, None))
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Molar ratio", a="PEO", b="LiCl", low="5", high="40")
        with pytest.raises(ValueError, match="pump loadout"):
            tab._build_config()

    def test_composition_mode_builds_a_general_formulation_the_twin_can_solve(
        self, qapp, manager, monkeypatch
    ):
        from softae.core.formulation import MolarRatioTarget

        tab = LiveBOCampaignTab(manager)
        monkeypatch.setattr(
            tab, "_load_stocks",
            lambda: ({"PEO": object(), "LiCl": object()}, {"PEO": 0, "LiCl": 1}, object()),
        )
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Molar ratio", a="PEO", b="LiCl", low="5", high="40")

        spec = tab._build_config()
        gf = spec.general_formulation
        assert gf is not None
        assert spec.parameter_space == {
            "ratio_PEO_LiCl": {"type": "float", "low": 5.0, "high": 40.0}}
        # Volumes are solved, not searched — nothing may read an axis as a µL value.
        assert spec.vol_params == ()
        assert spec.pump_ids == (0, 1)
        assert gf.pump_assignment == {"PEO": 0, "LiCl": 1}
        assert gf.build_targets({"ratio_PEO_LiCl": 20.0}) == [
            MolarRatioTarget("PEO", "LiCl", 20.0)]

    def test_an_all_pinned_target_set_is_refused_with_a_reason(
        self, qapp, manager, monkeypatch
    ):
        tab = LiveBOCampaignTab(manager)
        monkeypatch.setattr(
            tab, "_load_stocks",
            lambda: ({"PEO": object()}, {"PEO": 0}, object()))
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Concentration", a="LiCl", low="1", high="1")
        with pytest.raises(ValueError, match="nothing to search"):
            tab._build_config()

    def test_the_mode_and_its_axes_survive_a_save_load_round_trip(self, qapp, manager):
        src = LiveBOCampaignTab(manager)
        src._combo_search_mode.setCurrentIndex(1)
        src._axes_editor.add_axis("Molar ratio", a="PEO", b="LiCl", low="5", high="40")
        src._spin_dep_uL.setValue(9.5)

        dst = LiveBOCampaignTab(manager)
        dst._populate_from_config(src._panel_state())
        assert dst._search_mode() == "composition"
        assert dst._spin_dep_uL.value() == pytest.approx(9.5)
        axes = dst._axes_editor.axes()
        assert len(axes) == 1
        assert (axes[0].a, axes[0].b, axes[0].low, axes[0].high) == ("PEO", "LiCl", 5.0, 40.0)
