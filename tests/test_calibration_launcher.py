"""The launcher hands the rig over — or refuses to, and says why.

Every test here is about a way the handover can be wrong: launching into a running
sequence, launching while this process still holds the ports, offering a sequence that
does nothing, or reading the lock from a file nobody writes.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from softae.core.run_lock import RunLock
from softae.drivers.mock_factory import create_mock_manager
from softae.gui.widgets import calibration_launcher as cl


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def dlg(qapp, tmp_path):
    """A launcher whose coroutine scheduler runs the callback synchronously."""
    manager = create_mock_manager(config={})
    scheduled: list = []

    def _schedule(coro, done):
        coro.close()                 # never actually disconnect in a test
        scheduled.append(done)

    with patch.object(cl, "read_run_lock", return_value=None):
        d = cl.CalibrationLauncherDialog(
            manager, str(tmp_path), schedule=_schedule)
    d._scheduled = scheduled
    yield d
    d._timer.stop()
    d.close()


def _foreign_lock() -> RunLock:
    return RunLock(pid=os.getpid() + 1, what="commissioning blank_short",
                   started_at="2026-08-07T14:02:00+00:00", host="")


# ── Scope: the defect that made the busy check invisible ─────────────────────


class TestLockScope:
    def test_the_lock_is_read_at_the_machine_scope_not_the_project_directory(self, dlg):
        """`WorkflowExecutor.run()` acquires at ``~/.softae/rig.lock``.

        Reading a project-scoped path instead would find no file, report "Free" for the
        whole length of a running sequence, and let this dialog launch a second one
        onto the same ports. Passing *any* scope here is the bug.
        """
        with patch.object(cl, "read_run_lock", return_value=None) as read:
            dlg._refresh_state()
        read.assert_called_once_with()

    def test_taking_over_breaks_the_lock_at_the_same_scope_it_was_read_from(self, dlg):
        with patch.object(cl, "read_run_lock", return_value=_foreign_lock()), \
             patch.object(cl, "break_run_lock", return_value=_foreign_lock()) as brk, \
             patch.object(QMessageBox, "warning",
                          return_value=QMessageBox.StandardButton.Yes):
            dlg._on_break_lock()
        brk.assert_called_once_with()


# ── Refusals ─────────────────────────────────────────────────────────────────


class TestRefusals:
    def test_a_live_lock_disables_launching_rather_than_only_warning_on_click(self, dlg):
        with patch.object(cl, "read_run_lock", return_value=_foreign_lock()):
            dlg._refresh_state()
        assert not dlg._btn_launch.isEnabled()
        assert dlg._btn_break.isEnabled()
        assert "commissioning blank_short" in dlg._state_label.text()

    def test_launch_rechecks_the_lock_even_though_the_button_was_enabled(self, dlg):
        """The button reflects a 2 s poll; the check at click time is the real one."""
        dlg._combo.setCurrentText("Commissioning — open blank")
        dlg._channels.setText("1-8")
        with patch.object(cl, "read_run_lock", return_value=_foreign_lock()), \
             patch.object(QMessageBox, "warning") as warn, \
             patch.object(cl, "spawn_detached") as spawn:
            dlg._on_launch()
        assert spawn.call_count == 0
        assert warn.call_count == 1

    def test_a_sequence_needing_channels_refuses_an_empty_field(self, dlg):
        dlg._combo.setCurrentText("Commissioning — open blank")
        dlg._channels.setText("")
        with patch.object(QMessageBox, "warning"):
            assert dlg._argv() is None

    def test_a_reference_part_refuses_without_its_marked_value(self, dlg):
        """Marking-versus-measurement disagreement is the check that catches an
        unusable reference; it needs both numbers."""
        dlg._combo.setCurrentText("Commissioning — reference capacitor")
        dlg._channels.setText("1")
        dlg._nominal.setText("")
        with patch.object(QMessageBox, "warning"):
            assert dlg._argv() is None


# ── The handover ─────────────────────────────────────────────────────────────


class TestHandover:
    def test_instruments_are_released_before_the_child_is_spawned(self, dlg):
        dlg._combo.setCurrentText("Commissioning — open blank")
        dlg._channels.setText("1-8")
        with patch.object(cl, "read_run_lock", return_value=None), \
             patch.object(dlg._manager, "list_instruments",
                          return_value=[{"name": "stage", "connected": True}]), \
             patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Ok), \
             patch.object(cl, "spawn_detached") as spawn:
            dlg._on_launch()
            assert spawn.call_count == 0, "spawn must wait for the release callback"
            assert dlg._scheduled, "a disconnect must have been scheduled"

    def test_a_failed_release_aborts_the_launch_rather_than_racing_the_ports(self, dlg):
        with patch.object(dlg._manager, "list_instruments",
                          return_value=[{"name": "stage", "connected": True}]), \
             patch.object(cl, "read_run_lock", return_value=None), \
             patch.object(QMessageBox, "critical") as crit, \
             patch.object(cl, "spawn_detached") as spawn:
            dlg._after_release(True, ["-m", "softae.tools.commission"])
        assert spawn.call_count == 0, "stage is still connected"
        assert crit.call_count == 1

    def test_a_clean_release_proceeds_to_spawn(self, dlg):
        with patch.object(dlg._manager, "list_instruments", return_value=[]), \
             patch.object(cl, "read_run_lock", return_value=None), \
             patch.object(cl, "spawn_detached", return_value=4242) as spawn, \
             patch.object(QMessageBox, "information"):
            dlg._after_release(True, ["-m", "softae.tools.commission"])
        assert spawn.call_count == 1


# ── argv: the dialog and a terminal must run the same command ────────────────


class TestArgv:
    def test_argv_carries_the_project_and_skips_the_cli_prompt(self, dlg, tmp_path):
        dlg._combo.setCurrentText("Commissioning — open blank")
        dlg._channels.setText("1-8")
        argv = dlg._argv()
        assert argv[:4] == ["-m", "softae.tools.commission", "run", "blank_open"]
        assert argv[argv.index("--channels") + 1] == "1-8"
        assert argv[argv.index("--project") + 1] == str(tmp_path)
        assert "--yes" in argv, "the dialog already prompted"

    def test_every_sequence_maps_to_a_console_entry_point(self):
        """No GUI-only path that can drift from what a terminal run does."""
        for name, spec in cl.SEQUENCES.items():
            assert spec["argv"][0] == "-m", name
            assert spec["argv"][1].startswith("softae.tools."), name


class TestSequenceCatalogue:
    def test_the_geometry_series_cast_is_absent_while_its_execute_path_is_a_no_op(self):
        """`softae-thickness cast --execute` prints instructions and exits.

        Listed here, it would spawn a detached child, return, and have the dialog
        report "Started as PID nnnn" — an operator could wait at the rig for a cast
        that was never going to happen. An entry that looks like the others and does
        nothing is worse than an absent one.
        """
        assert not any("eometry" in name for name in cl.SEQUENCES)
        assert not any("thickness" in " ".join(s["argv"]) for s in cl.SEQUENCES.values())

    def test_the_lock_claim_this_dialog_rests_on_is_checkable_in_one_place(self):
        assert cl.CHILDREN_ACQUIRE_THE_LOCK is True


# ── Detachment ───────────────────────────────────────────────────────────────


class TestSpawnDetached:
    def test_the_child_is_detached_from_this_processs_session(self, tmp_path):
        """Without this the child joins this console group and dies with the GUI —
        the single behaviour the dialog exists to avoid."""
        log = tmp_path / "run.log"
        with patch.object(cl.subprocess, "Popen") as popen:
            popen.return_value = MagicMock(pid=1234)
            assert cl.spawn_detached(["-m", "x"], log_file=log) == 1234
        kwargs = popen.call_args.kwargs
        if os.name == "nt":
            assert kwargs["creationflags"] == 0x00000008 | 0x00000200
        else:
            assert kwargs["start_new_session"] is True

    def test_output_goes_to_a_file_not_a_pipe(self, tmp_path):
        """A pipe needs a reader for the child's whole life, which reintroduces the
        dependency on the GUI staying open, and a full buffer would block mid-sweep."""
        log = tmp_path / "run.log"
        with patch.object(cl.subprocess, "Popen") as popen:
            popen.return_value = MagicMock(pid=1)
            cl.spawn_detached(["-m", "x"], log_file=log)
        assert popen.call_args.kwargs["stdout"].name == str(log)
        assert popen.call_args.kwargs["stdin"] == cl.subprocess.DEVNULL
        assert "-m x" in log.read_text(encoding="utf-8")

    def test_the_log_directory_is_created_under_the_project(self, tmp_path):
        assert cl._log_dir(tmp_path) == Path(tmp_path) / "logs"
        assert (Path(tmp_path) / "logs").is_dir()
