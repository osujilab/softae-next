"""GUI tests for the BO Simulator tab (uses the session ``qapp`` fixture)."""

from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from softae.campaigns.config import BOCampaignConfig
from softae.gui.tabs.tab_bo_simulator import BOSimulatorTab


def test_tab_constructs(qapp):
    tab = BOSimulatorTab()
    assert tab is not None
    # default-built config validates structurally (no dataset path yet)
    cfg = tab._build_config()
    assert isinstance(cfg, BOCampaignConfig)
    assert cfg.transform == "log10_sigma"
    assert cfg.acquisition == "ucb"
    assert cfg.backend == "sklearn"


def test_tab_config_roundtrip(qapp):
    src = BOSimulatorTab()
    # change several widgets away from defaults
    src._combo_acq.setCurrentText("ei")
    src._combo_backend.setCurrentText("sklearn")
    src._combo_objdir.setCurrentText("minimize")
    src._spin_ninit.setValue(7)
    src._spin_seed.setValue(123)
    src._combo_channel.setCurrentText("acquisition_weight")
    src._chk_mean.setChecked(False)
    src._le_sources.setText("replicate")
    src._le_path.setText("data.txt")

    cfg = src._build_config()
    assert cfg.acquisition == "ei"
    assert cfg.objective_direction == "minimize"
    assert cfg.n_initial == 7
    assert cfg.seed == 123
    assert cfg.noise_channel == "acquisition_weight"
    assert cfg.target_is_mean is False
    assert cfg.noise_sources == ["replicate"]

    # round-trip through a fresh tab
    dst = BOSimulatorTab()
    dst._populate_from_config(cfg)
    cfg2 = dst._build_config()
    assert cfg2.acquisition == "ei"
    assert cfg2.objective_direction == "minimize"
    assert cfg2.n_initial == 7
    assert cfg2.noise_channel == "acquisition_weight"
    assert cfg2.target_is_mean is False
    assert cfg2.noise_sources == ["replicate"]


def test_tab_maxsteps_zero_means_pool(qapp):
    tab = BOSimulatorTab()
    tab._spin_maxsteps.setValue(0)
    assert tab._build_config().max_steps is None
    tab._spin_maxsteps.setValue(12)
    assert tab._build_config().max_steps == 12


def test_tab_rail_ceiling_zero_disables(qapp):
    tab = BOSimulatorTab()
    tab._spin_railsigma.setValue(0.0)
    assert tab._build_config().rail_sigma_ceiling is None


def test_campaign_mode_switches_acquisition_family(qapp):
    tab = BOSimulatorTab()
    # Optimize → optimization-family acquisitions + optimization stopping
    tab._combo_mode.setCurrentText("Optimize")
    acqs = {tab._combo_acq.itemText(i) for i in range(tab._combo_acq.count())}
    assert acqs == {"ucb", "ei"}
    assert tab._build_config().stopping_mode == "optimization"

    # Explore → active-learning acquisitions + model-accuracy stopping
    tab._combo_mode.setCurrentText("Explore")
    acqs = {tab._combo_acq.itemText(i) for i in range(tab._combo_acq.count())}
    assert acqs == {"max_variance", "integrated_variance", "uncertainty_weighted"}
    cfg = tab._build_config()
    assert cfg.acquisition in acqs
    assert cfg.stopping_mode == "model_accuracy"


def test_explore_config_roundtrips_mode(qapp):
    src = BOSimulatorTab()
    src._combo_mode.setCurrentText("Explore")
    src._combo_acq.setCurrentText("integrated_variance")
    cfg = src._build_config()
    assert cfg.acquisition == "integrated_variance"

    dst = BOSimulatorTab()
    dst._populate_from_config(cfg)
    # mode inferred from the acquisition's family
    assert dst._combo_mode.currentText() == "Explore"
    assert dst._build_config().acquisition == "integrated_variance"


# ── Daemon shutdown seam (abort_run / cleanup) ─────────────────────────────


def _spin_on_flag(tab: BOSimulatorTab) -> threading.Thread:
    """Real daemon thread spinning until the tab's cooperative abort flag is set."""

    def run() -> None:
        while not tab._abort_requested:
            time.sleep(0.02)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


class TestDaemonShutdown:
    def test_bo_campaign_cleanup_aborts_running_thread(self, qapp):
        tab = BOSimulatorTab()
        tab._abort_requested = False
        tab._thread = _spin_on_flag(tab)
        assert tab._thread.is_alive()
        tab.cleanup()
        assert tab._abort_requested is True
        assert not tab._thread.is_alive()

    def test_bo_campaign_cleanup_is_noop_when_idle(self, qapp):
        tab = BOSimulatorTab()
        assert tab._thread is None
        tab.cleanup()  # must not raise / block

    def test_bo_campaign_cleanup_is_idempotent(self, qapp):
        tab = BOSimulatorTab()
        tab._abort_requested = False
        tab._thread = _spin_on_flag(tab)
        tab.cleanup()
        tab.cleanup()
        assert not tab._thread.is_alive()

    def test_bo_campaign_abort_run_signals_without_joining(self, qapp):
        tab = BOSimulatorTab()
        tab._abort_requested = False
        tab._thread = _spin_on_flag(tab)
        tab.abort_run()
        assert tab._abort_requested is True
        assert tab._thread.is_alive()  # signal-only: not joined
        tab.cleanup()  # teardown join
