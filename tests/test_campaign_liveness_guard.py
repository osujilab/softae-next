"""Two campaigns, or a campaign and a GUI launch, must not destroy each other.

The collision this closes: ``DataStore.unfinished_runs`` is **project-wide** and a
*running* campaign's row is indistinguishable from a crashed one's. So the
unclean-shutdown recovery — on either surface — used to read a live campaign's own
row, mark it ``interrupted`` and park the rig underneath it. Starting a second
headless campaign killed the first; opening the GUI killed it from the other side.

The fix is an **ordering**: ask the rig lock (liveness) before asking the run rows
(recovery). Each mechanism is useless at the other's job, in opposite directions —
``read_run_lock`` unlinks a stale lock so it can never be a recovery marker, and a
run row never self-clears so it can never be a liveness check. These tests pin the
ordering, both surfaces, and the two things that must NOT change: a genuinely
crashed run is still recovered, and nothing here refuses manual control.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from softae.core import run_lock as rl
from softae.core.data_store import DataStore
from softae.core.run_lock import (
    acquire_run_lock,
    busy_rig_message,
    foreign_run_lock,
    lock_path,
)
from softae.tools import campaign as cli

# A one-trial campaign: this suite is about what happens *around* the run, so the
# run itself is made as small as the spec allows.
SPEC = """
name = "liveness_test"
channels = [21]
pcb_name = "SoftAE_EIS_4Stripe"
budget = 1
two_phase = true
vol_params = ["vol_p0", "vol_p1"]
pump_ids = [0, 1]
time_scale = 0.0

[parameter_space.vol_p0]
type = "float"
low = 5.0
high = 30.0

[parameter_space.vol_p1]
type = "float"
low = 5.0
high = 30.0
"""


@pytest.fixture(autouse=True)
def _no_sink_leakage():
    """Recovery raises real alerts; a sink bound to a closed store must not escape.

    Same guard as ``test_unclean_shutdown``, for the same reason — the tests here
    exercise the recovery path deliberately.
    """
    from softae.core.alerts import clear_alert_sinks

    clear_alert_sinks()
    yield
    clear_alert_sinks()


@pytest.fixture(autouse=True)
def _isolated_scope(tmp_path, monkeypatch):
    """Never read the operator's real ``~/.softae/rig.lock``.

    Autouse and unconditional: a test that consulted the real lock would pass or
    fail depending on whether the GUI happened to be mid-sequence, and could see a
    live rig claim belonging to an actual experiment.
    """
    scope = tmp_path / "scope"
    monkeypatch.setattr(rl, "DEFAULT_SCOPE", scope)
    return scope


def _write_foreign(scope, *, what="campaign:overnight:run-7"):
    """A claim held by a live process that is not this one.

    Another *host* rather than another PID, matching ``test_run_lock``: a foreign
    host always reads as alive by design, so the fixture does not depend on
    finding a real running PID it is allowed to inspect.
    """
    lock_path(scope).parent.mkdir(parents=True, exist_ok=True)
    lock_path(scope).write_text(json.dumps(
        {"pid": 4321, "what": what, "started_at": "2026-08-12T14:02:00+00:00",
         "host": "some-other-machine", "log_path": ""}), encoding="utf-8")


@pytest.fixture
def _real_rig(monkeypatch):
    """Force `campaign.py` down its real-rig branch without real drivers.

    The import on the first line is load-bearing, not tidiness.
    :mod:`softae.core.autonomous_wiring` binds ``rig_is_simulated`` at **module
    import time**, so patching :mod:`~softae.core.run_lock` before that module is
    first loaded leaves the campaign holding this lambda *permanently* —
    ``monkeypatch`` restores the attribute on ``run_lock`` and cannot reach the
    copy ``autonomous_wiring`` already took. The symptom is a later, unrelated
    test watching a ``--mock`` campaign try to claim the rig and abort with
    ``RunLockHeld``. Importing first binds the real function before the patch.
    """
    import softae.core.autonomous_wiring  # noqa: F401 — bind the real one first

    monkeypatch.setattr(rl, "rig_is_simulated", lambda _m: False)


def _spec_file(tmp_path: Path) -> str:
    p = tmp_path / "spec.toml"
    p.write_text(SPEC, encoding="utf-8")
    return str(p)


# ── The liveness predicate ───────────────────────────────────────────────────

class TestForeignRunLock:
    """Liveness composed from the predicates that already exist, not a new one."""

    def test_no_lock_reads_as_nobody_holding_the_rig(self, _isolated_scope):
        assert foreign_run_lock(_isolated_scope) is None

    def test_a_live_foreign_claim_is_returned_with_its_identity(self,
                                                                _isolated_scope):
        _write_foreign(_isolated_scope, what="campaign:overnight:run-7")
        holder = foreign_run_lock(_isolated_scope)

        assert holder is not None
        assert holder.what == "campaign:overnight:run-7"
        assert holder.started_at == "2026-08-12T14:02:00+00:00"

    def test_this_processes_own_claim_is_not_foreign(self, _isolated_scope):
        """A campaign re-entering its own claim is not a second owner.

        Reporting it as one would refuse the ordinary case — the GUI running its
        own sequence, or any re-entrant acquire.
        """
        acquire_run_lock(_isolated_scope, "my own run")
        assert foreign_run_lock(_isolated_scope) is None

    def test_a_crashed_holders_lock_does_not_read_as_live(self, _isolated_scope):
        """Staleness self-clears, which is exactly right for a liveness check.

        This is also why the lock cannot double as a recovery marker: the answer
        here is `None`, identical to "nobody ever ran", and the crashed run's
        evidence has just been deleted as a side effect.
        """
        lock_path(_isolated_scope).parent.mkdir(parents=True, exist_ok=True)
        lock_path(_isolated_scope).write_text(json.dumps(
            {"pid": 999_999_999, "what": "crashed run", "started_at": "",
             "host": socket.gethostname(), "log_path": ""}), encoding="utf-8")

        assert foreign_run_lock(_isolated_scope) is None

    def test_it_reuses_the_default_scope_so_one_machine_is_one_rig(self):
        """No scope argument must resolve to the machine-wide lock, not a project."""
        assert lock_path() == rl.DEFAULT_SCOPE / "rig.lock"


class TestBusyMessage:
    """A refusal an operator can act on, never a bare "busy"."""

    def test_it_names_the_pid_the_run_and_the_start_time(self, _isolated_scope):
        _write_foreign(_isolated_scope, what="campaign:overnight:run-7")
        text = busy_rig_message(foreign_run_lock(_isolated_scope),
                                action="This campaign")

        assert "4321" in text
        assert "campaign:overnight:run-7" in text
        assert "14:02" in text

    def test_it_offers_the_wedged_holder_an_existing_deliberate_override(
            self, _isolated_scope):
        """Staleness self-clears; a *hung* holder does not, and needs a way out.

        The way out already exists — `break_run_lock`, surfaced as the Calibration
        Launcher's "Take the rig?" confirmation — so the message points at it
        rather than this guard growing a force-takeover of its own.
        """
        _write_foreign(_isolated_scope)
        text = busy_rig_message(foreign_run_lock(_isolated_scope), action="X")

        assert "wedged" in text
        assert "Calibration Launcher" in text

    def test_it_says_the_refusal_does_not_extend_to_manual_control(
            self, _isolated_scope):
        """The standing no-lockout ruling is about manual control at the rig.

        Refusing a second *automated* run is a different act, and the message has
        to draw that line or it reads as the lockout the ruling forbids.
        """
        _write_foreign(_isolated_scope)
        text = busy_rig_message(foreign_run_lock(_isolated_scope), action="X")

        assert "Manual control" in text


# ── Headless surface ─────────────────────────────────────────────────────────

class TestHeadlessRefusesToJoinALiveRun:
    def test_a_real_run_is_refused_and_the_holder_is_named(
            self, tmp_path, capsys, monkeypatch, _isolated_scope, _real_rig):
        """The first half of the fix: never reach recovery while someone is live."""
        from softae.drivers import factory
        from softae.drivers.mock_factory import create_mock_manager

        monkeypatch.setattr(factory, "create_manager",
                            lambda *a, **k: create_mock_manager(config={}))
        _write_foreign(_isolated_scope, what="campaign:overnight:run-7")

        rc = cli.main(["run", _spec_file(tmp_path), "--yes", "--head-up",
                       "--project", str(tmp_path / "proj")])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_DECLINED
        assert "NOT STARTING" in out
        assert "campaign:overnight:run-7" in out      # who, not just "busy"
        assert "4321" in out

    def test_the_refusal_happens_before_the_head_prompt(
            self, tmp_path, capsys, monkeypatch, _isolated_scope, _real_rig):
        """Refuse before interrogating the operator about hardware state.

        `--head-up` is deliberately omitted here: reaching the head gate at all
        would mean the guard ran too late.
        """
        from softae.drivers import factory
        from softae.drivers.mock_factory import create_mock_manager

        monkeypatch.setattr(factory, "create_manager",
                            lambda *a, **k: create_mock_manager(config={}))
        monkeypatch.setattr(cli.sys, "stdin", None)
        _write_foreign(_isolated_scope)

        rc = cli.main(["run", _spec_file(tmp_path), "--yes",
                       "--project", str(tmp_path / "proj")])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_DECLINED
        assert "Head position unknown" not in out

    def test_a_simulated_run_is_not_refused_over_hardware_it_never_touches(
            self, tmp_path, capsys, _isolated_scope):
        """`--mock` claims no rig, so contention does not apply to it.

        Scoped by the same `rig_is_simulated` predicate `autonomous_wiring` uses to
        decide whether to claim the lock at all, so the two cannot disagree about
        what counts as contention.
        """
        _write_foreign(_isolated_scope)

        rc = cli.main(["run", _spec_file(tmp_path), "--mock", "--yes",
                       "--head-up", "--project", str(tmp_path / "proj")])

        assert rc == cli.EXIT_OK
        assert "NOT STARTING" not in capsys.readouterr().out


class TestHeadlessRecoveryOrdering:
    """A live run's row must never be consumed as a crashed one's."""

    def test_a_live_campaigns_row_is_not_marked_interrupted(
            self, tmp_path, capsys, _isolated_scope):
        """The destructive case, on the one path that still reaches recovery.

        `--mock` is exempt from the refusal because it touches no hardware — but it
        shares the DataStore, and marking the live campaign's row `interrupted` is
        just as corrupting from a simulated process as from a real one.
        """
        project = tmp_path / "proj"
        with DataStore(project) as ds:
            live_run = ds.start_run("the_campaign_that_is_running")

        _write_foreign(_isolated_scope, what=f"campaign:x:{live_run}")

        rc = cli.main(["run", _spec_file(tmp_path), "--mock", "--yes",
                       "--head-up", "--project", str(project)])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        with DataStore(project) as ds:
            still_open = [r["run_id"] for r in ds.unfinished_runs()]
        assert live_run in still_open, "the running campaign's row was consumed"
        assert "skipping unclean-shutdown recovery" in out

    def test_a_genuinely_crashed_run_is_still_recovered(
            self, tmp_path, capsys, _isolated_scope):
        """Positive control: the guard defers, it does not disable recovery.

        Without this the test above could be satisfied by breaking recovery
        outright, which is the failure Priority 1 exists to prevent.
        """
        project = tmp_path / "proj"
        with DataStore(project) as ds:
            crashed = ds.start_run("died_last_night")

        # No lock at all — nobody is running, so the row means what it says.
        rc = cli.main(["run", _spec_file(tmp_path), "--mock", "--yes",
                       "--head-up", "--project", str(project)])
        out = capsys.readouterr().out

        assert rc == cli.EXIT_OK
        with DataStore(project) as ds:
            still_open = [r["run_id"] for r in ds.unfinished_runs()]
            alerts = [a["kind"] for a in ds.query_alerts()]
        assert crashed not in still_open, "a real crash went unrecovered"
        assert "unclean_shutdown" in alerts
        assert "PREVIOUS SESSION DID NOT FINISH" in out


# ── GUI surface — the same collision from the other direction ────────────────

class TestGuiRecoveryOrdering:
    def test_opening_the_gui_does_not_park_a_running_headless_campaign(
            self, qapp, tmp_path, _isolated_scope):
        """The mirror case. The GUI cannot refuse to open, so it skips instead."""
        pytest.importorskip("PySide6")
        from unittest.mock import MagicMock

        from softae.gui.widgets.unclean_shutdown import check_unclean_shutdown

        with DataStore(tmp_path / "proj") as ds:
            live_run = ds.start_run("headless_campaign_in_flight")
            _write_foreign(_isolated_scope, what=f"campaign:x:{live_run}")
            manager = MagicMock()

            assert check_unclean_shutdown(None, manager, ds) is False
            # Neither half of the damage: the row survives and nothing was driven.
            assert [r["run_id"] for r in ds.unfinished_runs()] == [live_run]
            assert ds.query_alerts() == []
        manager.get.assert_not_called()

    def test_the_gui_still_recovers_when_nothing_holds_the_rig(
            self, qapp, tmp_path, monkeypatch, _isolated_scope):
        """Positive control for the GUI path, mirroring the headless one."""
        pytest.importorskip("PySide6")
        from unittest.mock import MagicMock

        from PySide6.QtWidgets import QMessageBox

        from softae.gui.widgets.unclean_shutdown import check_unclean_shutdown

        monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
        monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: None)

        with DataStore(tmp_path / "proj") as ds:
            ds.start_run("died_last_night")

            check_unclean_shutdown(None, MagicMock(), ds)

            assert ds.unfinished_runs() == []          # marked interrupted
            assert [a["kind"] for a in ds.query_alerts()] == ["unclean_shutdown"]

    def test_an_unreadable_lock_defers_rather_than_breaking_startup(
            self, qapp, tmp_path, monkeypatch, _isolated_scope):
        """Two rules meet here, and both are kept.

        This check must never stop the GUI opening, so it cannot raise; and it
        must not consume a live campaign's row on evidence it just failed to
        obtain, so it cannot assume the rig is free. Deferring satisfies both.
        """
        pytest.importorskip("PySide6")
        from unittest.mock import MagicMock

        import softae.gui.widgets.unclean_shutdown as gui_mod
        from softae.gui.widgets.unclean_shutdown import check_unclean_shutdown

        def _boom(*_a, **_k):
            raise OSError("lock directory is unreadable")

        monkeypatch.setattr(gui_mod, "foreign_run_lock", _boom)

        with DataStore(tmp_path / "proj") as ds:
            run = ds.start_run("might_be_live")

            assert check_unclean_shutdown(None, MagicMock(), ds) is False
            assert [r["run_id"] for r in ds.unfinished_runs()] == [run]

    def test_both_surfaces_describe_the_unknown_head_in_one_wording(self):
        """The alert text is defined once, in `core.shutdown`, and imported here.

        It was duplicated verbatim in both files; two copies of a sentence about
        the same physical unknown drift the moment one of them is edited.
        """
        pytest.importorskip("PySide6")
        import softae.gui.widgets.unclean_shutdown as gui_mod
        from softae.core.shutdown import UNCLEAN_SHUTDOWN_MESSAGE

        assert gui_mod.UNCLEAN_SHUTDOWN_MESSAGE is UNCLEAN_SHUTDOWN_MESSAGE
        assert "lowered" in UNCLEAN_SHUTDOWN_MESSAGE
