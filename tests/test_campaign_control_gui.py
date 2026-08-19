"""Pause and Abort on a GUI surface — the request, and the answer to it.

Three things are pinned here, and each of them is a way the feature could look
finished while being useless:

**One discovery implementation.** The CLI and the buttons must not come to
disagree about "no campaign to control" — a terminal that refuses while a button
is live (or the reverse) leaves an operator with no way to tell which is right
about a rig mid-anneal. Every refusal branch is asserted through *both* surfaces
in the same test.

**Every ack outcome surfaces.** ``ignored_stale``, ``unreadable`` and
``handler_failed`` all mean the run is unchanged. A button that greys out on
press and comes back on a timer renders them identically to success, which is
the failure this widget exists to avoid.

**The GUI parks nothing.** Abort writes a file; the park happens in the process
that owns the sessions. A park issued from here would command instruments this
process never opened.

No rig, no lock file, no subprocess: discovery is driven by an injected reader
and the stream is fabricated on ``tmp_path``.
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from softae.core.campaign_discovery import (
    ABORT_LATENCY_NOTE,
    PAUSE_LATENCY_NOTE,
    RESUME_LATENCY_NOTE,
    find_running_campaign,
)
from softae.core.campaign_events import (
    events_path,
    read_control_request,
    write_control_request,
)
from softae.gui.widgets import campaign_control
from softae.gui.widgets.campaign_control import (
    CampaignControlBar,
    CampaignControlRequester,
    outcome_note,
)
from softae.tools import campaign as campaign_cli


# ── Fixtures and helpers ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeLock:
    """Only the three fields discovery reads, plus the line it renders."""

    what: str = "campaign:demo:run-7"
    log_path: str = "/runs/run-7"
    pid: int = 4321

    def describe(self) -> str:
        return f"PID {self.pid} — {self.what}"


def _reader(lock):
    return lambda: lock


def _append_ack(run_dir, **fields) -> None:
    """Append one ``control_ack`` as the narrator would have written it."""
    with events_path(run_dir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "control_ack", **fields}) + "\n")


def _yes(*_args) -> bool:
    return True


def _no(*_args) -> bool:
    return False


@pytest.fixture
def bar(qapp, tmp_path):
    """A bar aimed at ``tmp_path``, with Abort pre-confirmed."""
    return CampaignControlBar(run_dir=str(tmp_path), confirm=_yes)


# ── The shared discovery helper: one implementation, two surfaces ───────────


class TestSharedDiscovery:
    """Both surfaces must answer identically on all four branches."""

    @pytest.mark.parametrize(
        "lock, detail",
        [
            (None, "no process holds the rig."),
            (
                _FakeLock(what="workflow 'ht_sweep'", log_path=""),
                "the rig is held by something that is not a campaign — "
                "PID 4321 — workflow 'ht_sweep'",
            ),
            (
                _FakeLock(log_path=""),
                "the campaign did not publish a run directory — "
                "PID 4321 — campaign:demo:run-7",
            ),
        ],
    )
    def test_discovery_refusal_branch_reads_the_same_on_cli_and_gui(
        self, qapp, monkeypatch, lock, detail
    ):
        monkeypatch.setattr(
            "softae.core.campaign_discovery.read_run_lock", _reader(lock))

        target = find_running_campaign(lock_reader=_reader(lock))
        assert target.run_dir is None
        assert target.detail == detail

        # CLI path — the tuple `_cmd_control` reads.
        assert campaign_cli._running_campaign_run_dir() == (None, detail)

        # GUI path — the same clause, rendered on the widget.
        gui_bar = CampaignControlBar(
            discover=lambda: find_running_campaign(lock_reader=_reader(lock)))
        assert gui_bar.target.detail == detail
        assert gui_bar._lbl_target.text() == f"No campaign to control: {detail}"

    def test_discovery_live_campaign_names_the_same_run_dir_on_both_surfaces(
        self, qapp, monkeypatch
    ):
        lock = _FakeLock()
        monkeypatch.setattr(
            "softae.core.campaign_discovery.read_run_lock", _reader(lock))

        assert campaign_cli._running_campaign_run_dir() == (
            "/runs/run-7", "campaign:demo:run-7")

        gui_bar = CampaignControlBar(
            discover=lambda: find_running_campaign(lock_reader=_reader(lock)))
        assert gui_bar.target.run_dir == "/runs/run-7"
        assert gui_bar._btn_pause.isEnabled()
        assert gui_bar._btn_abort.isEnabled()

    def test_discovery_unreadable_lock_disables_the_buttons_rather_than_raising(
        self, qapp
    ):
        def explode():
            raise OSError("lock volume offline")

        gui_bar = CampaignControlBar(discover=explode)
        assert gui_bar.target.controllable is False
        assert not gui_bar._btn_abort.isEnabled()
        assert "could not be read" in gui_bar._lbl_target.text()


# ── The tooltips are the CLI's sentences, not a paraphrase ──────────────────


class TestLatencyTooltips:

    def test_tooltips_are_the_shared_latency_sentences_verbatim(self, bar):
        assert bar._btn_abort.toolTip() == ABORT_LATENCY_NOTE
        assert bar._btn_pause.toolTip() == PAUSE_LATENCY_NOTE

    @pytest.mark.parametrize(
        "action, note",
        [("abort", ABORT_LATENCY_NOTE),
         ("pause", PAUSE_LATENCY_NOTE),
         ("resume", RESUME_LATENCY_NOTE)],
    )
    def test_cli_control_prints_the_same_sentence_the_button_shows(
        self, tmp_path, capsys, action, note
    ):
        campaign_cli._cmd_control(
            Namespace(run_dir=str(tmp_path), action=action, reason=""))
        assert note in capsys.readouterr().out

    def test_abort_confirmation_quotes_the_latency_sentence_unsoftened(
        self, qapp, tmp_path
    ):
        seen: list[str] = []

        def capture(title, text):
            seen.append(text)
            return False

        gui_bar = CampaignControlBar(run_dir=str(tmp_path), confirm=capture)
        gui_bar._btn_abort.click()
        assert ABORT_LATENCY_NOTE in seen[0]


# ── Writing the request ─────────────────────────────────────────────────────


class TestControlWrite:

    def test_pause_press_writes_a_pause_request_at_seq_one(self, bar, tmp_path):
        bar._btn_pause.click()
        request = read_control_request(tmp_path)
        assert request.action == "pause"
        assert request.seq == 1
        assert request.reason == ""
        assert "softae GUI" in request.requested_by
        assert bar.pending.seq == 1

    def test_pause_press_after_an_existing_request_takes_the_next_seq(
        self, bar, tmp_path
    ):
        write_control_request(tmp_path, "abort", requested_by="someone else")
        bar._btn_pause.click()
        assert read_control_request(tmp_path).seq == 2
        assert bar.pending.seq == 2

    def test_abort_press_writes_an_abort_request_naming_the_operator(
        self, bar, tmp_path
    ):
        bar._btn_abort.click()
        request = read_control_request(tmp_path)
        assert request.action == "abort"
        assert "operator abort" in request.reason

    def test_abort_press_declined_at_the_prompt_writes_nothing(self, qapp, tmp_path):
        gui_bar = CampaignControlBar(run_dir=str(tmp_path), confirm=_no)
        gui_bar._btn_abort.click()
        assert read_control_request(tmp_path) is None
        assert gui_bar.pending is None

    def test_abort_press_declined_in_the_default_dialog_writes_nothing(
        self, qapp, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.No)
        gui_bar = CampaignControlBar(run_dir=str(tmp_path))
        gui_bar._btn_abort.click()
        assert read_control_request(tmp_path) is None

    def test_press_while_pending_is_refused_until_the_ack_arrives(self, bar, tmp_path):
        bar._btn_pause.click()
        assert not bar._btn_pause.isEnabled()
        assert not bar._btn_abort.isEnabled()
        assert read_control_request(tmp_path).seq == 1

    def test_press_after_the_campaign_vanished_writes_nothing_and_says_why(
        self, qapp, tmp_path
    ):
        """Discovery is re-run at press time, not trusted from the last refresh.

        The buttons are driven by a 2 s poll, so a campaign that ended in the
        interval leaves a live-looking Abort behind. ``_on_abort`` is called
        directly here because a disabled button swallows the click.
        """
        found = [find_running_campaign(lock_reader=_reader(_FakeLock(
            log_path=str(tmp_path))))]

        gui_bar = CampaignControlBar(discover=lambda: found[0], confirm=_yes)
        assert gui_bar._btn_abort.isEnabled()

        found[0] = find_running_campaign(lock_reader=_reader(None))
        gui_bar._on_abort()

        assert read_control_request(tmp_path) is None
        assert "no process holds the rig." in gui_bar._lbl_status.text()

    def test_a_write_that_fails_says_the_campaign_was_not_asked(self, qapp, tmp_path):
        def explode(*_a, **_k):
            raise PermissionError("read-only run directory")

        gui_bar = CampaignControlBar(
            run_dir=str(tmp_path),
            confirm=_yes,
            requester_factory=lambda run_dir: CampaignControlRequester(
                run_dir, writer=explode),
        )
        gui_bar._btn_abort.click()
        assert gui_bar.pending is None
        assert "has not been asked to stop" in gui_bar._lbl_status.text()


# ── Resolving the request: every outcome, including the three refusals ──────


class TestAckResolution:

    def test_matching_ack_resolves_the_pending_request(self, bar, tmp_path):
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=1, action="pause", outcome="applied")
        bar.refresh()
        assert bar.pending is None
        assert "applied" in bar._lbl_status.text()
        assert bar._btn_abort.isEnabled()

    @pytest.mark.parametrize(
        "outcome, expected",
        [
            ("ignored_stale", "IGNORED"),
            ("handler_failed", "FAILED"),
            ("unreadable", "REFUSED"),
        ],
    )
    def test_failure_ack_is_surfaced_rather_than_swallowed(
        self, bar, tmp_path, outcome, expected
    ):
        bar._btn_abort.click()
        _append_ack(tmp_path, seq=1, action="abort", outcome=outcome)
        bar.refresh()
        assert bar.pending is None
        text = bar._lbl_status.text()
        assert outcome in text
        assert expected in text
        assert "The run is unchanged" in text

    def test_seqless_unreadable_ack_resolves_the_pending_request(self, bar, tmp_path):
        """The watcher omits ``seq`` when the file could not be parsed at all.

        Matching on seq alone would leave that press pending forever — the one
        outcome where the operator most needs an answer.
        """
        bar._btn_abort.click()
        _append_ack(tmp_path, outcome="unreadable", path=str(tmp_path / "control.json"))
        bar.refresh()
        assert bar.pending is None
        assert "REFUSED" in bar._lbl_status.text()

    def test_ack_for_another_seq_leaves_the_request_pending(self, bar, tmp_path):
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=99, action="abort", outcome="applied")
        bar.refresh()
        assert bar.pending is not None
        assert bar.pending.seq == 1

    def test_ack_written_before_the_press_does_not_resolve_it(self, bar, tmp_path):
        _append_ack(tmp_path, seq=1, action="pause", outcome="applied")
        bar._btn_pause.click()
        bar.refresh()
        assert bar.pending is not None

    def test_pending_request_reports_how_long_it_has_gone_unanswered(
        self, bar, tmp_path
    ):
        bar._btn_pause.click()
        bar.refresh()
        assert "waiting for the campaign to acknowledge" in bar._lbl_status.text()

    def test_resolved_ack_is_emitted_for_the_surfacing_tab_to_log(self, bar, tmp_path):
        seen: list[dict] = []
        bar.acknowledged.connect(seen.append)
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=1, action="pause", outcome="applied")
        bar.refresh()
        assert seen and seen[0]["outcome"] == "applied"

    def test_unknown_outcome_is_reported_as_unrecognised_not_as_success(self):
        note = outcome_note("teleported")
        assert "does not recognise" in note

    def test_a_vanished_run_directory_leaves_the_request_pending(self, bar, tmp_path):
        bar._btn_pause.click()

        def explode(*_a, **_k):
            raise OSError("gone")

        bar._requester._reader = explode
        bar.refresh()
        assert bar.pending is not None


# ── Pause is leavable ───────────────────────────────────────────────────────


class TestPauseToggle:

    def test_acknowledged_pause_turns_the_button_into_resume(self, bar, tmp_path):
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=1, action="pause", outcome="applied")
        bar.refresh()
        assert "Resume" in bar._btn_pause.text()
        assert bar._btn_pause.toolTip() == RESUME_LATENCY_NOTE

    def test_resume_press_after_a_pause_requests_resume(self, bar, tmp_path):
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=1, action="pause", outcome="applied")
        bar.refresh()
        bar._btn_pause.click()
        assert read_control_request(tmp_path).action == "resume"

    def test_refused_pause_leaves_the_button_offering_pause(self, bar, tmp_path):
        bar._btn_pause.click()
        _append_ack(tmp_path, seq=1, action="pause", outcome="handler_failed")
        bar.refresh()
        assert "Pause" in bar._btn_pause.text()


# ── The GUI's job ends at the file ──────────────────────────────────────────


class TestNoParkFromTheGui:

    def test_abort_press_issues_no_park_from_the_gui(self, qapp, tmp_path, monkeypatch):
        """``loop.abort()`` parks in the owning process, where the sessions are."""
        import softae.core.safe_park as safe_park_module

        monkeypatch.setattr(
            safe_park_module, "safe_park",
            lambda *a, **k: pytest.fail("the GUI parked a rig it does not own"))

        gui_bar = CampaignControlBar(run_dir=str(tmp_path), confirm=_yes)
        gui_bar._btn_abort.click()
        _append_ack(tmp_path, seq=1, action="abort", outcome="applied")
        gui_bar.refresh()

        assert read_control_request(tmp_path).action == "abort"

    def test_the_control_widget_has_no_park_symbol_at_all(self):
        """Introspective, and deliberately so — it stops a later 'helpful' park."""
        assert not hasattr(campaign_control, "safe_park")
        assert not hasattr(campaign_control, "SafeParkResult")


# ── The tab that surfaces the campaign ──────────────────────────────────────


class TestLiveBoTabSurface:

    def test_live_bo_tab_carries_the_campaign_control_bar(self, qapp):
        from softae.drivers.mock_factory import create_mock_manager
        from softae.gui.tabs.tab_bo_live import LiveBOCampaignTab

        tab = LiveBOCampaignTab(create_mock_manager(config={}))
        assert isinstance(tab._campaign_controls, CampaignControlBar)
        assert tab._campaign_controls._btn_abort.toolTip() == ABORT_LATENCY_NOTE

    def test_live_bo_tab_logs_the_ack_outcome(self, qapp):
        from softae.drivers.mock_factory import create_mock_manager
        from softae.gui.tabs.tab_bo_live import LiveBOCampaignTab

        tab = LiveBOCampaignTab(create_mock_manager(config={}))
        tab._on_control_ack({"action": "abort", "outcome": "handler_failed"})
        assert "handler_failed" in tab._log.toPlainText()
