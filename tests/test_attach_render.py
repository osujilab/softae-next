"""What an attached window actually shows: the stream, the sidecar, the park.

Four surfaces, one rule — an attached window opens no session, so everything it
renders comes from files the campaign wrote:

* the **stream view** turns ``events.jsonl`` into the status strings;
* the **conditions source** turns ``conditions.json`` into the poller's dicts,
  and refuses to show numbers that have gone stale;
* the **poller** takes the source without any of its three consumer widgets
  changing — which is the test of whether the abstraction sits in the right
  layer;
* the **window** ticks all of it, including the park indicator, whose only
  periodic caller used to be the purge timer an attached window does not create.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from softae.config import loader
from softae.core.campaign_events import (
    CampaignNarrator,
    conditions_path,
    events_path,
)
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.campaign_stream import CampaignStreamView
from softae.gui.launch_mode import LaunchMode
from softae.gui.widgets.conditions_source import (
    CONDITIONS_STALE_AFTER_S,
    ConditionsFileSource,
)
from softae.gui.widgets.instrument_poller import InstrumentPoller, LiveInstrumentSource
from softae.gui.widgets.rig_owner import OCCUPIED
from softae.gui.widgets.status_indicator import _COLORS, InstrumentStatusBar


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def mock_manager():
    return create_mock_manager(config={})


def _write_events(run_dir: Path, *records: dict) -> None:
    """Append records with stamps, exactly as the narrator lays them out."""
    with events_path(run_dir).open("a", encoding="utf-8") as fh:
        for record in records:
            record.setdefault("ts", _stamp(0))
            fh.write(json.dumps(record) + "\n")


def _stamp(age_s: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()


def _write_conditions(run_dir: Path, *, age_s: float = 0.0, **env) -> None:
    payload = {
        "started_at": _stamp(age_s + 0.5),
        "completed_at": _stamp(age_s),
        "read_ms": 500,
        "env": {
            "stage_temp_sp_C": None, "chamber_air_C": None,
            "stage_temp_pv_C": None, "rh_sp_pct": None, "rh_pv_pct": None,
            **env,
        },
        "skipped_beats": 0,
    }
    conditions_path(run_dir).write_text(json.dumps(payload), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# The stream view
# ═════════════════════════════════════════════════════════════════════════════

class TestCampaignStreamView:

    def test_view_before_any_record_reports_nothing_read_yet(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        view.poll()
        assert "no records read yet" in view.auto_status()
        assert view.phase is None
        assert view.park_reason is None

    def test_view_missing_run_dir_does_not_raise(self, tmp_path):
        view = CampaignStreamView(str(tmp_path / "gone"), campaign=("shadow", "r1"))
        view.poll()          # must not raise
        assert view.park_reason is None

    def test_view_heartbeat_supplies_phase_and_age(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {
            "type": "heartbeat", "phase": "anneal", "phase_age_s": 42.0,
            "iteration": 7,
        })
        view.poll()
        assert view.phase == "anneal"
        assert view.phase_age_s == 42.0

    def test_view_status_names_state_iteration_and_liveness(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(
            tmp_path,
            {"type": "run_started", "run_id": "r1", "spec": "shadow"},
            {"type": "state", "old": "IDLE", "new": "RUNNING"},
            {"type": "result", "iteration": 3, "objective": 1.2},
        )
        view.poll()
        status = view.auto_status()
        assert "shadow" in status and "RUNNING" in status and "iter 3" in status
        assert "live" in status

    def test_view_silent_stream_reports_no_heartbeat_not_health(self, tmp_path):
        """Three beats of silence is the stream's own "wedged" rule, not ours."""
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {
            "type": "heartbeat", "phase": "anneal", "phase_age_s": 1.0,
            "ts": _stamp(600),
        })
        view.poll()
        assert "NO HEARTBEAT" in view.auto_status()

    def test_view_park_record_is_the_park_reason(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {"type": "park", "reason": "RH gate failed"})
        view.poll()
        assert view.park_reason == "RH gate failed"

    def test_view_park_stays_latched_after_later_records(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {"type": "park", "reason": "RH gate failed"})
        view.poll()
        _write_events(tmp_path, {"type": "safe_park", "ok": True},
                      {"type": "run_finished", "status": "STOPPED"})
        view.poll()
        assert view.park_reason == "RH gate failed"

    def test_view_finished_run_points_at_the_way_back_in(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {"type": "run_finished", "status": "CONVERGED"})
        view.poll()
        assert "CONVERGED" in view.auto_status()
        assert "Connect All" in view.ht_status()

    def test_view_ht_slot_says_why_it_is_idle(self, tmp_path):
        """"Idle" alone is the same word a free rig shows — useless here."""
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        view.poll()
        assert view.ht_status() == "idle — shadow holds the rig"

    def test_view_incremental_poll_does_not_lose_the_previous_phase(self, tmp_path):
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        _write_events(tmp_path, {"type": "heartbeat", "phase": "anneal",
                                 "phase_age_s": 5.0})
        view.poll()
        _write_events(tmp_path, {"type": "result", "iteration": 9})
        view.poll()
        assert view.phase == "anneal"      # no beat in that batch
        assert "iter 9" in view.auto_status()

    def test_view_reading_does_not_prevent_a_rotation(self, tmp_path):
        """A held handle makes os.replace fail on Windows — and the 32 MB cap
        exists because this process shares a disk with the DataStore."""
        narrator = CampaignNarrator(tmp_path, heartbeat_s=0, max_bytes=200)
        view = CampaignStreamView(str(tmp_path), campaign=("shadow", "r1"))
        try:
            for i in range(30):
                narrator.record("result", {"iteration": i, "padding": "x" * 40})
                view.poll()
        finally:
            narrator.close()
        assert (tmp_path / "events.1.jsonl").exists(), "rotation never happened"
        assert view.auto_status()          # and the view followed it


# ═════════════════════════════════════════════════════════════════════════════
# The conditions sidecar as a poller source
# ═════════════════════════════════════════════════════════════════════════════

class TestConditionsFileSource:

    def test_source_fresh_sidecar_fills_the_widget_dicts(self, tmp_path, mock_manager):
        _write_conditions(tmp_path, stage_temp_sp_C=60.0, stage_temp_pv_C=59.4,
                          chamber_air_C=23.1, rh_sp_pct=35.0, rh_pv_pct=34.2)
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar["temp_sp"] == 60.0
        assert reading.sidebar["temp_pv"] == 59.4
        assert reading.sidebar["chamber_temp"] == 23.1
        assert reading.sidebar["rh_pv"] == 34.2
        assert reading.monitor["rh"] == 34.2
        assert reading.monitor["temp_pv"] == 59.4

    def test_source_stale_sidecar_renders_unknown_not_current(
        self, tmp_path, mock_manager
    ):
        """A two-minute-old number shown as "now" is the one lie this must not
        tell, and the widgets have nowhere to display an age."""
        _write_conditions(tmp_path, age_s=CONDITIONS_STALE_AFTER_S + 5,
                          stage_temp_sp_C=60.0, stage_temp_pv_C=59.4)
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar == {}
        assert reading.monitor == {}
        # …but the rig is still the campaign's; a stale file does not hand it back.
        assert all(v == {"state": OCCUPIED} for v in reading.statuses.values())

    def test_source_publisher_with_no_completed_read_is_unknown_not_old(
        self, tmp_path, mock_manager
    ):
        """The publisher is up and its first read has not returned yet."""
        conditions_path(tmp_path).write_text(json.dumps({
            "started_at": _stamp(1), "completed_at": None, "read_ms": None,
            "env": {"stage_temp_sp_C": 60.0}, "skipped_beats": 1,
        }), encoding="utf-8")
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar == {}

    def test_source_missing_sidecar_is_unknown_not_an_error(
        self, tmp_path, mock_manager
    ):
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar == {} and reading.monitor == {}
        assert reading.statuses, "the instruments are still somebody's"

    def test_source_corrupt_sidecar_is_unknown_not_an_error(
        self, tmp_path, mock_manager
    ):
        conditions_path(tmp_path).write_text("{not json", encoding="utf-8")
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar == {}

    def test_source_reports_every_instrument_as_occupied_not_disconnected(
        self, tmp_path, mock_manager
    ):
        """DISCONNECTED is grey, and grey on a rig mid-anneal reads as "it is off"."""
        _write_conditions(tmp_path, stage_temp_sp_C=60.0)
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert set(reading.statuses) == set(mock_manager.names)
        assert all(s["state"] == OCCUPIED for s in reading.statuses.values())
        assert OCCUPIED in _COLORS, "a live rig would render grey"
        assert _COLORS[OCCUPIED] != _COLORS["DISCONNECTED"]

    def test_source_reads_no_instrument(self, tmp_path, mock_manager):
        """The invariant: an attached window opens no session, so it may not read.

        ``status_all()`` is the trap — it calls ``status()`` on every instrument
        and the RH controller's ``status()`` reads the sensor.
        """
        called: list[str] = []
        mock_manager.status_all = lambda: called.append("status_all") or {}
        original_get = mock_manager.get
        mock_manager.get = lambda name: called.append(name) or original_get(name)
        _write_conditions(tmp_path, stage_temp_sp_C=60.0)
        ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert called == []

    def test_source_half_published_humidity_does_not_invent_the_other_half(
        self, tmp_path, mock_manager
    ):
        import math
        _write_conditions(tmp_path, rh_pv_pct=34.2)
        reading = ConditionsFileSource(tmp_path, manager=mock_manager).read()
        assert reading.sidebar["rh_pv"] == 34.2
        assert math.isnan(reading.sidebar["rh_sp"])
        assert "rh_sp" not in reading.monitor

    def test_source_no_run_dir_is_occupied_with_nothing_to_show(self, mock_manager):
        """Held by something that is not a campaign: no stream, no sidecar."""
        reading = ConditionsFileSource(None, manager=mock_manager).read()
        assert reading.sidebar == {} and reading.monitor == {}
        assert all(s["state"] == OCCUPIED for s in reading.statuses.values())


# ═════════════════════════════════════════════════════════════════════════════
# The poller takes a source; the three consumer widgets do not change
# ═════════════════════════════════════════════════════════════════════════════

class TestPollerSource:

    def test_poller_defaults_to_the_live_instrument_source(self, mock_manager):
        poller = InstrumentPoller(mock_manager)
        assert isinstance(poller.source, LiveInstrumentSource)

    def test_poller_emits_what_the_injected_source_returns(
        self, qapp, tmp_path, mock_manager
    ):
        _write_conditions(tmp_path, stage_temp_sp_C=60.0, stage_temp_pv_C=59.4)
        poller = InstrumentPoller(
            mock_manager, source=ConditionsFileSource(tmp_path, manager=mock_manager)
        )
        seen: dict[str, dict] = {}
        poller.status_ready.connect(lambda d: seen.setdefault("status", d))
        poller.sidebar_ready.connect(lambda d: seen.setdefault("sidebar", d))
        poller.monitor_ready.connect(lambda d: seen.setdefault("monitor", d))
        poller._do_poll()                    # the loop body, without the thread
        assert seen["sidebar"]["temp_pv"] == 59.4
        assert seen["monitor"]["temp_sp"] == 60.0
        assert all(s["state"] == OCCUPIED for s in seen["status"].values())

    def test_poller_source_that_raises_does_not_end_the_thread(self, mock_manager):
        class _Boom:
            def read(self):
                raise RuntimeError("sidecar on a dead share")

        poller = InstrumentPoller(mock_manager, source=_Boom())
        poller._do_poll()                    # must not raise

    def test_status_bar_renders_the_attached_rig_through_its_existing_slot(
        self, qapp, tmp_path, mock_manager
    ):
        """The consumer widget is unchanged apart from one colour: it is handed
        the same dict shape and asked nothing new."""
        bar = InstrumentStatusBar(mock_manager, poller=InstrumentPoller(mock_manager))
        bar._apply_statuses({"stage": {"state": OCCUPIED}})
        assert _COLORS[OCCUPIED] in bar._indicators["stage"].text()

    def test_sidebar_renders_sidecar_values_through_its_existing_slot(
        self, qapp, tmp_path, mock_manager
    ):
        from softae.gui.widgets.monitor_sidebar import MonitorSidebar

        _write_conditions(tmp_path, stage_temp_sp_C=60.0, stage_temp_pv_C=59.4)
        poller = InstrumentPoller(
            mock_manager, source=ConditionsFileSource(tmp_path, manager=mock_manager)
        )
        sidebar = MonitorSidebar(mock_manager, poller=poller)
        try:
            poller._do_poll()
            assert "59.4" in sidebar._lbl_t_pv.text()
            # Not published by the campaign, so not invented here.
            assert "--" in sidebar._lbl_stage.text()
        finally:
            timer = getattr(sidebar, "_wc_frame_timer", None)
            if timer is not None:
                timer.stop()
            sidebar.close()


# ═════════════════════════════════════════════════════════════════════════════
# The window: the tick, the slots, the park indicator
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "runs" / "run-42"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def attached_window(qapp, qtbot, monkeypatch, mock_manager, run_dir):
    monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
    from softae.gui.main_window import MainWindow

    mode = LaunchMode(
        attached=True, campaign=("shadow-run", "run-42"), run_dir=str(run_dir),
        holder=None, reason="Campaign 'shadow-run' (run run-42) holds the rig.",
    )
    mw = MainWindow(mock_manager, launch_mode=mode)
    qtbot.addWidget(mw)
    yield mw
    mw.close()
    qapp.processEvents()
    mw.deleteLater()
    qapp.processEvents()


@pytest.fixture
def owner_window(qapp, qtbot, monkeypatch, mock_manager):
    monkeypatch.setattr(loader, "load", lambda: {"webcam": {"enabled": False}})
    from softae.gui.main_window import MainWindow

    mw = MainWindow(mock_manager)
    qtbot.addWidget(mw)
    yield mw
    mw.close()
    qapp.processEvents()
    mw.deleteLater()
    qapp.processEvents()


class TestAttachedWindowRender:

    def test_window_attached_drives_the_status_slots_from_the_stream(
        self, attached_window, run_dir
    ):
        _write_events(run_dir, {"type": "state", "old": "IDLE", "new": "RUNNING"},
                      {"type": "result", "iteration": 5})
        attached_window._on_campaign_tick()
        assert "iter 5" in attached_window._sidebar._lbl_wf_auto.text()
        assert "holds the rig" in attached_window._sidebar._lbl_wf_ht.text()

    def test_window_attached_leaves_the_in_process_signals_disconnected(
        self, attached_window
    ):
        """One source swapped, not two sources fighting over one label."""
        attached_window._on_campaign_tick()
        before = attached_window._sidebar._lbl_wf_auto.text()
        attached_window._tab_autonomous.workflow_status_changed.emit("Iteration 99")
        assert attached_window._sidebar._lbl_wf_auto.text() == before

    def test_window_owner_keeps_the_in_process_signal_sources(self, owner_window):
        owner_window._tab_autonomous.workflow_status_changed.emit("Iteration 99")
        assert "Iteration 99" in owner_window._sidebar._lbl_wf_auto.text()
        owner_window._tab_experiment.workflow_status_changed.emit("Step 3/8")
        assert "Step 3/8" in owner_window._sidebar._lbl_wf_ht.text()

    def test_window_attached_uses_the_conditions_source(self, attached_window):
        assert isinstance(attached_window._poller.source, ConditionsFileSource)

    def test_window_owner_uses_the_live_instrument_source(self, owner_window):
        assert isinstance(owner_window._poller.source, LiveInstrumentSource)

    def test_window_attached_renders_the_sidecar(self, attached_window, run_dir):
        """All three consumers, through the slots they already had."""
        _write_conditions(run_dir, stage_temp_sp_C=60.0, stage_temp_pv_C=59.4,
                          rh_pv_pct=34.2)
        attached_window._poller._do_poll()
        assert "59.4" in attached_window._sidebar._lbl_t_pv.text()
        assert "59.4" in attached_window._tab_monitor._lbl_temp_pv.text()
        assert "34.2" in attached_window._tab_monitor._lbl_rh_pv.text()

    def test_window_attached_stale_sidecar_stops_showing_the_old_number(
        self, attached_window, run_dir
    ):
        """The widgets have nowhere to show an age, so the only honest rendering
        of a stale value is no value."""
        _write_conditions(run_dir, stage_temp_pv_C=59.4, stage_temp_sp_C=60.0)
        attached_window._poller._do_poll()
        assert "59.4" in attached_window._sidebar._lbl_t_pv.text()

        _write_conditions(run_dir, age_s=CONDITIONS_STALE_AFTER_S + 5,
                          stage_temp_pv_C=59.4, stage_temp_sp_C=60.0)
        attached_window._poller._do_poll()
        assert "59.4" not in attached_window._sidebar._lbl_t_pv.text()
        assert "--" in attached_window._sidebar._lbl_t_pv.text()

    def test_window_attached_park_record_raises_the_clear_control(
        self, attached_window, run_dir
    ):
        """The park indicator's only periodic caller was the purge tick, and an
        attached window creates no purge timer — E supplies both."""
        assert not attached_window._clear_park_action.isVisible()
        _write_events(run_dir, {"type": "park", "reason": "RH gate failed"})
        attached_window._on_campaign_tick()
        assert attached_window._clear_park_action.isVisible()
        assert "RH gate failed" in attached_window._park_reason()

    def test_window_attached_has_a_tick_although_it_has_no_purge_timer(
        self, attached_window
    ):
        assert attached_window._purge_timer is None
        assert attached_window._campaign_timer.isActive()

    def test_window_attached_owner_line_names_the_campaign(
        self, attached_window, run_dir, monkeypatch
    ):
        import softae.gui.main_window as mw_mod

        class _Lock:
            what = "campaign:shadow-run:run-42"
            pid = 4242
            started_at = "2026-08-19T14:02:00"
            log_path = str(run_dir)

        monkeypatch.setattr(mw_mod, "foreign_rig_lock", lambda: _Lock())
        _write_events(run_dir, {"type": "heartbeat", "phase": "anneal",
                                "phase_age_s": 12.0})
        attached_window._on_campaign_tick()
        line = attached_window._sidebar._lbl_rig_owner.text()
        assert line == "shadow-run · anneal · 12s — See Monitoring tab"

    def test_window_non_campaign_lock_renders_occupied_with_nothing_to_attach_to(
        self, owner_window, monkeypatch
    ):
        import softae.gui.main_window as mw_mod

        class _Lock:
            what = "workflow 'ht_sequence'"
            pid = 99
            started_at = "2026-08-19T14:02:00"
            log_path = ""

        monkeypatch.setattr(mw_mod, "foreign_rig_lock", lambda: _Lock())
        owner_window._on_campaign_tick()
        line = owner_window._sidebar._lbl_rig_owner.text()
        assert line.startswith(OCCUPIED)
        assert "Monitoring tab" not in line

    def test_window_free_rig_hides_the_owner_line(self, owner_window, monkeypatch):
        import softae.gui.main_window as mw_mod

        monkeypatch.setattr(mw_mod, "foreign_rig_lock", lambda: None)
        owner_window._on_campaign_tick()
        assert owner_window._sidebar._lbl_rig_owner.isHidden()

    def test_window_tick_survives_an_unreadable_run_directory(
        self, attached_window, run_dir
    ):
        import shutil

        shutil.rmtree(run_dir)
        attached_window._on_campaign_tick()      # must not raise

    def test_window_closing_stops_the_campaign_tick(self, attached_window):
        attached_window.close()
        assert not attached_window._campaign_timer.isActive()
