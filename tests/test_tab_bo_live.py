"""GUI tests for the Live BO Campaign tab (uses the session ``qapp`` fixture).

Kept fast: nothing here starts a campaign. Since S5.J the tab does not run one —
it writes a spec, hands the rig over and spawns a detached child — so the launch
is exercised by intercepting the spawn, and the *attached* half by fabricating a
run directory and letting the tab read it the way it would read a real one.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from softae.core.autonomous_wiring import CampaignSpec
from softae.core.run_lock import RunLock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.tabs._bo_base import BOTabBase
from softae.gui.tabs.tab_bo_live import LiveBOCampaignTab
from softae.gui.tabs.tab_bo_simulator import BOSimulatorTab


@pytest.fixture
def manager():
    return create_mock_manager(config={})


def _arm_handover(tab, monkeypatch, *, spawned=None, lock=None):
    """Let the handover run end-to-end without touching the rig or the OS.

    The scheduler needs the GUI's qasync loop in production, the lock read is the
    machine's real one, and the spawn starts a process — all three are replaced,
    and nothing else about the path is.
    """
    import asyncio

    import softae.gui.campaign_launch as launch

    spawned = [] if spawned is None else spawned
    monkeypatch.setattr(tab, "_schedule",
                        lambda coro, done: done(asyncio.run(coro) is None))
    monkeypatch.setattr("softae.core.rig_session.release_rig_session",
                        lambda *a, **k: True)
    monkeypatch.setattr("softae.core.run_lock.read_run_lock", lambda *a, **k: lock)
    monkeypatch.setattr(launch, "spawn_campaign",
                        lambda argv, *, log_file: spawned.append((argv, log_file))
                        or 24680)
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    return spawned


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


# ── Launching: the campaign leaves this process (S5.J) ─────────────────────


class TestTheCampaignShellsOut:
    """Run no longer means "start a thread"; it means "hand the rig over".

    ``test_run_campaign_wiring`` used to live here. It drove ``_run_campaign``,
    asserted the loop received *this window's* manager, and read the convergence
    buffers filled by the in-process ``on_event`` callback. All three describe
    the execution path this step removes, so the case is rewritten rather than
    deleted: the same three questions are asked of the new path — what was
    started, what it was given, and how the tab learns what it did.
    """

    @staticmethod
    def _tab(manager, tmp_path):
        from types import SimpleNamespace

        return LiveBOCampaignTab(
            manager, data_store=SimpleNamespace(project_dir=tmp_path))

    def test_on_run_spawns_a_child_instead_of_starting_a_worker_thread(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        tab = self._tab(manager, tmp_path)
        spawned = _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        threads: list = []
        monkeypatch.setattr(tab, "_start_worker",
                            lambda *a, **k: threads.append(a))

        tab._on_run()

        assert threads == []
        assert tab._thread is None
        assert len(spawned) == 1

    def test_on_run_writes_the_spec_the_child_is_started_from(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        from softae.core.campaign_spec_io import load_campaign_spec

        tab = self._tab(manager, tmp_path)
        spawned = _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        tab._le_name.setText("bench_run")
        tab._spin_budget.setValue(11)

        tab._on_run()

        written = sorted((tmp_path / "launched").glob("*.toml"))
        assert len(written) == 1
        assert str(written[0]) in spawned[0][0]
        # The child runs what is on screen, or the launch is refused (below).
        reloaded = load_campaign_spec(written[0])
        assert reloaded.name == "bench_run" and reloaded.budget == 11

    def test_on_run_declining_the_head_gate_starts_nothing(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        tab = self._tab(manager, tmp_path)
        spawned = _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: False)
        tab._on_run()
        assert spawned == []
        assert not (tmp_path / "launched").exists()

    @staticmethod
    def _composition_mode(tab, monkeypatch, *, stock_names=("PEO stock",
                                                            "LiCl stock")):
        """Put the tab in composition mode over a catalog made for this test.

        The catalog is patched at :func:`softae.core.campaign_spec_fields.catalogs`
        — the one seam the *loader* resolves stock names through — so the round
        trip the launch rests on is the real one rather than a stubbed answer.
        """
        import softae.core.campaign_spec_fields as fields
        from softae.core.formulation import (
            ChemicalCatalog,
            Solution,
            SolutionCatalog,
        )

        sol = SolutionCatalog()
        for name in ("PEO stock", "LiCl stock"):
            sol.add(Solution(name=name))
        chem = ChemicalCatalog()
        monkeypatch.setattr(fields, "catalogs", lambda: (chem, sol))
        monkeypatch.setattr(
            tab, "_load_stocks",
            lambda: ({n: Solution(name=n) for n in stock_names},
                     {n: i for i, n in enumerate(stock_names)}, chem))
        tab._combo_search_mode.setCurrentIndex(1)
        tab._axes_editor.add_axis("Molar ratio", a="EO", b="Li",
                                  low="5", high="40")

    def test_a_composition_campaign_launches_now_that_a_file_can_carry_it(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The tab's headline mode, and it was unlaunchable.

        ``general_formulation`` could not be written and the child is started
        *from* a spec file, so composition mode guaranteed a refusal with no way
        to run the campaign at all. The file carries declared axes and stock
        names now — and the launch still rests on the round trip proving it, not
        on a relaxed check.
        """
        from softae.core.campaign_spec_io import load_campaign_spec

        tab = self._tab(manager, tmp_path)
        spawned = _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        self._composition_mode(tab, monkeypatch)

        tab._on_run()

        assert len(spawned) == 1
        written = sorted((tmp_path / "launched").glob("*.toml"))
        gf = load_campaign_spec(written[0]).general_formulation
        assert gf is not None, "the child must run the campaign on screen"
        assert [ax.name for ax in gf.axes] == ["ratio_EO_Li"]
        assert sorted(gf.stocks) == ["LiCl stock", "PEO stock"]

    def test_a_composition_campaign_naming_an_unknown_stock_is_still_refused(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The refusal is not weakened, only narrowed to what is genuinely lost.

        A stock the catalog cannot resolve is a composition the solver cannot
        turn into volumes, so the file does not carry the campaign — refused on a
        free rig, with the panel state preserved, and **before the head gate**,
        which prompts the operator and can issue a safety retract.
        """
        tab = self._tab(manager, tmp_path)
        spawned = _arm_handover(tab, monkeypatch)
        asked: list = []
        monkeypatch.setattr(tab, "_verify_head_position",
                            lambda *a, **k: asked.append(True) or True)
        self._composition_mode(tab, monkeypatch, stock_names=("Unobtainium",))
        shown: list[str] = []
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda p, t, text, *a, **k: shown.append(text)))

        tab._on_run()

        assert spawned == []
        assert asked == []
        assert "general_formulation" in shown[0]
        assert len(list((tmp_path / "rejected").glob("*.json"))) == 1
        assert not (tmp_path / "launched").exists()

    def test_the_spawned_child_is_recorded_without_a_way_to_stop_it(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The filed ``self._runner = None`` defect.

        It was never reassigned, so ``_abort_run_impl``'s ``runner.abort()`` was
        dead for this tab. It is assigned now — to a handle that deliberately has
        no ``abort``, because the same code path runs from the window's
        ``closeEvent``.
        """
        from softae.gui.campaign_launch import DetachedCampaign

        tab = self._tab(manager, tmp_path)
        _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)

        tab._on_run()

        assert isinstance(tab._runner, DetachedCampaign)
        assert tab._runner.pid == 24680
        assert not hasattr(tab._runner, "abort")

    def test_closing_the_window_does_not_stop_the_detached_campaign(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """``MainWindow.closeEvent`` calls ``cleanup()`` on every runner tab.

        Against an in-process run that was a cooperative abort. Against a
        detached one it must be a no-op: the campaign is why the operator was
        allowed to close the window in the first place.
        """
        import softae.core.campaign_events as events

        tab = self._tab(manager, tmp_path)
        _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        tab._on_run()

        requests: list = []
        monkeypatch.setattr(events, "write_control_request",
                            lambda *a, **k: requests.append(a))

        tab.cleanup()          # what the window does on the way out

        assert requests == []          # no abort was asked for
        assert tab._runner.pid == 24680

    def test_the_tab_offers_no_in_process_abort_that_could_reach_nothing(
        self, qapp, manager, tmp_path
    ):
        """A stop control that greys itself out and does nothing is worse than none."""
        tab = self._tab(manager, tmp_path)
        assert not hasattr(tab, "_btn_abort")
        assert tab._campaign_controls is not None


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


# ── Attaching: the tab follows a campaign it does not own (S5.J) ───────────


class TestAttachedView:
    """The tab reaches its own child by the path every other attach uses.

    A channel that exists only for the GUI-started case is a second safety
    posture, and the drift between two postures is what this step exists to end.
    So there is nothing here that a colleague's terminal-started campaign would
    not also get: the rig lock names the run directory, and the run directory
    holds the transcript.
    """

    @staticmethod
    def _run_dir(tmp_path, records):
        """A run directory shaped exactly as the campaign narrator leaves one."""
        from softae.core.campaign_events import EVENTS_FILENAME

        run_dir = tmp_path / "runs" / "20260819T120000Z_demo"
        run_dir.mkdir(parents=True)
        (run_dir / EVENTS_FILENAME).write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return run_dir

    @staticmethod
    def _holding(monkeypatch, run_dir, *, what="campaign:demo:20260819T120000Z_demo"):
        """Make the rig lock say a campaign owns the rig, as a real child would."""
        lock = RunLock(pid=98765, what=what,
                       started_at="2026-08-19T12:00:00+00:00",
                       host="another-host", log_path=str(run_dir))
        monkeypatch.setattr("softae.core.campaign_discovery.read_run_lock",
                            lambda *a, **k: lock)
        return lock

    def test_the_tab_attaches_through_the_shared_discovery_helper(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """Patching the *shared* reader is what makes "one path" assertable."""
        run_dir = self._run_dir(tmp_path, [
            {"type": "run_started", "run_id": "20260819T120000Z_demo"},
        ])
        self._holding(monkeypatch, run_dir)
        tab = LiveBOCampaignTab(manager)

        tab._poll_campaign_stream()

        assert tab._event_run_dir == str(run_dir)
        assert any("run 20260819T120000Z_demo started" in line
                   for line in tab._log.toPlainText().splitlines())

    def test_the_tab_holds_no_private_pointer_at_the_campaign_it_started(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """It cannot: the run id is minted inside the child.

        So the "no privileged channel" rule is enforced by arithmetic rather than
        by discipline — this window does not know the run directory until the
        child publishes it on the lock, exactly like any other observer.
        """
        tab = LiveBOCampaignTab(manager,
                                data_store=type("S", (), {"project_dir": tmp_path})())
        _arm_handover(tab, monkeypatch)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        tab._on_run()

        assert tab._event_run_dir is None
        assert tab._campaign_controls._explicit_run_dir is None

    def test_a_result_record_from_the_stream_drives_the_convergence_trace(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        run_dir = self._run_dir(tmp_path, [
            {"type": "objective_resolved", "objective": "mean_abs_z",
             "direction": "minimize"},
            {"type": "result", "iteration": 1,
             "params": {"vol_p0": 10.0, "vol_p1": 10.0}, "objective": 0.5},
            {"type": "result", "iteration": 2,
             "params": {"vol_p0": 12.0, "vol_p1": 9.0}, "objective": 0.9},
        ])
        self._holding(monkeypatch, run_dir)
        tab = LiveBOCampaignTab(manager)

        tab._poll_campaign_stream()

        assert tab._xs == [1.0, 2.0]
        # Direction read from the run, not from this panel: 0.9 is worse than
        # 0.5 for a minimising campaign, so "best" must not move.
        assert tab._maximize is False
        assert tab._primary_series == [0.5, 0.5]

    def test_a_second_poll_reads_only_what_is_new(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        from softae.core.campaign_events import EVENTS_FILENAME

        run_dir = self._run_dir(tmp_path, [
            {"type": "result", "iteration": 1, "params": {"a": 1.0}, "objective": 1.0},
        ])
        self._holding(monkeypatch, run_dir)
        tab = LiveBOCampaignTab(manager)
        tab._poll_campaign_stream()

        with (run_dir / EVENTS_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"type": "result", "iteration": 2, "params": {"a": 2.0},
                 "objective": 2.0}) + "\n")
        tab._poll_campaign_stream()

        assert tab._xs == [1.0, 2.0]          # not [1, 1, 2]

    def test_a_lock_held_by_something_that_is_not_a_campaign_attaches_to_nothing(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """A bench sequence publishes no transcript and offers no control channel."""
        run_dir = self._run_dir(tmp_path, [{"type": "run_started", "run_id": "x"}])
        self._holding(monkeypatch, run_dir, what="gui:desktop")
        tab = LiveBOCampaignTab(manager)

        tab._poll_campaign_stream()

        assert tab._event_run_dir is None
        assert tab._btn_run.isEnabled()

    def test_a_live_campaign_disables_the_run_button_rather_than_racing_it(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        run_dir = self._run_dir(tmp_path, [{"type": "run_started", "run_id": "x"}])
        self._holding(monkeypatch, run_dir)
        tab = LiveBOCampaignTab(manager)

        tab._poll_campaign_stream()
        assert not tab._btn_run.isEnabled()

    def test_an_unreadable_lock_leaves_the_tab_standing(
        self, qapp, manager, monkeypatch
    ):
        def boom(*a, **k):
            raise OSError("the lock file is on a share that went away")

        monkeypatch.setattr("softae.core.campaign_discovery.read_run_lock", boom)
        tab = LiveBOCampaignTab(manager)
        tab._poll_campaign_stream()          # must not raise


class TestEveryRecordIsShown:
    """The filed missing-``else`` defect: ``park`` and ``safe_park`` were dropped.

    They are the two records that say the rig stopped itself, and they fell
    through a dispatcher whose last branch was ``run_finished``. So did every
    record added to the campaign after the dispatcher was written.
    """

    def test_a_park_record_reaches_the_campaign_log(self, qapp, manager):
        tab = LiveBOCampaignTab(manager)
        tab._on_campaign_event({"type": "park", "reason": "RH gate never settled"})
        assert "RH gate never settled" in tab._log.toPlainText()

    def test_a_failed_safe_park_is_not_reported_as_a_completed_one(
        self, qapp, manager
    ):
        tab = LiveBOCampaignTab(manager)
        tab._on_campaign_event({"type": "safe_park", "ok": False,
                                "errors": ["heater refused"]})
        text = tab._log.toPlainText()
        assert "INCOMPLETE" in text and "heater refused" in text

    def test_a_record_this_window_does_not_know_is_shown_rather_than_dropped(
        self, qapp, manager
    ):
        """An installed GUI meets records added after it shipped."""
        tab = LiveBOCampaignTab(manager)
        tab._on_campaign_event({"type": "settle_verdict", "ts": "t", "seq": 4,
                                "verdict": "suspect"})
        text = tab._log.toPlainText()
        assert "settle_verdict" in text and "suspect" in text
        assert "seq" not in text          # the envelope is not the message

    def test_a_heartbeat_goes_to_the_status_line_not_the_log(self, qapp, manager):
        """One every 30 s; a transcript that is mostly beats is one nobody reads."""
        tab = LiveBOCampaignTab(manager)
        before = tab._log.toPlainText()
        tab._on_campaign_event({"type": "heartbeat", "phase": "anneal",
                                "phase_age_s": 4210.0, "iteration": 3})
        assert tab._log.toPlainText() == before
        assert "anneal" in tab._lbl_status.text()


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
        handed: list = []
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: handed.append(a) or True)
        tab._on_run()
        assert handed == []

    def test_on_run_with_a_foreign_lock_never_reaches_the_head_gate(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The head gate prompts, and may retract — a refused launch moves nothing."""
        tab = self._tab(qapp, manager, tmp_path)
        self._hold_rig(monkeypatch)
        asked: list[bool] = []
        monkeypatch.setattr(tab, "_verify_head_position",
                            lambda *a, **k: asked.append(True) or True)
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: True)
        tab._on_run()
        assert asked == []

    def test_a_refused_volume_campaign_is_offered_a_relaunch_command(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        tab = self._tab(qapp, manager, tmp_path)
        shown = self._hold_rig(monkeypatch)
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: True)
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
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: True)
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
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: True)
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

    def test_a_free_rig_hands_the_campaign_over_rather_than_refusing_it(
        self, qapp, manager, monkeypatch, tmp_path
    ):
        """The refusal must be reachable only from a live foreign lock.

        Was ``..._runs_the_campaign_as_before``, which asserted a worker thread
        started. Nothing runs in this process any more, so the same guarantee is
        now stated against the handover.
        """
        tab = self._tab(qapp, manager, tmp_path)
        monkeypatch.setattr("softae.core.run_lock.foreign_run_lock", lambda *a: None)
        monkeypatch.setattr(tab, "_verify_head_position", lambda *a, **k: True)
        handed: list = []
        monkeypatch.setattr(tab, "_hand_over_to_a_detached_campaign",
                            lambda *a, **k: handed.append(a) or True)
        tab._on_run()
        assert handed and not (tmp_path / "rejected").exists()


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
