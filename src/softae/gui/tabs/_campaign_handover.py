"""Giving the rig to a child process, and refusing to when it is not safe (S5.J).

The preparation a launch needs — head gate, board gates, pre-flights, the
single-occupancy refusal — lives in :mod:`softae.gui.tabs._autonomous_run`. This
is the other half: the moment the rig changes hands.

It is separate because it is a different kind of code. Everything in the
preparation half is a question put to the operator; everything here is an
irreversible act on the rig's ownership, sequenced so that no two processes can
believe they own it. The sequence is
:class:`~softae.gui.widgets.calibration_launcher.CalibrationLauncherDialog`'s,
which is the shipped precedent for this handover:

======  =====================================================================
1       prove a spec file can carry this campaign faithfully — a file that
        reloads as a *different* experiment must never start a child
2       write the spec beside the project
3       disconnect the instrument sessions, then release the rig claim
4       refuse to spawn if anything is still connected **or** still claimed
5       spawn detached, and let go
======  =====================================================================

Step 4 is the one that earns the split. A launcher that skipped it would start a
child that either collides on the ports or is refused the rig lock and dies into
a log file nobody is watching — and both failures look, from the GUI, exactly
like a campaign that started.

The mixin carries no state of its own. Hosts supply ``self._manager``, a
``_sig_log`` signal, ``_project_dir()`` and ``_preserve_rejected_launch()``.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from PySide6.QtWidgets import QMessageBox

logger = structlog.get_logger(__name__)


class CampaignHandoverMixin:
    """Write the spec, release the rig, spawn the child — or refuse, and say why."""

    def _hand_over_to_a_detached_campaign(self, spec) -> bool:
        """Write the spec, release the rig, spawn a child that outlives us.

        ``True`` means the handover was started — the spawn itself completes in
        :meth:`_spawn_after_release`, because ``disconnect_all()`` is a coroutine
        on the qasync loop and the child must not be started until it has
        genuinely finished. ``False`` means nothing was started and the operator
        has been told why.

        The order is :class:`~softae.gui.widgets.calibration_launcher.CalibrationLauncherDialog`'s,
        and every step of it is load-bearing:

        1. **Prove the file is the whole spec.** A composition campaign written
           to TOML reloads as a raw-volume one and raises nothing, so an
           unprovable spec is refused *before* anything is written or released.
        2. **Write it.** The child reads a file, exactly as a terminal run does.
        3. **Release**: disconnect the sessions, then give the rig claim back.
        4. **Refuse to spawn if anything is still held.** Two processes on one
           set of ports is the collision the whole lock exists to prevent, and a
           disconnect that failed is precisely when it would happen.
        """
        from softae.gui.campaign_launch import (
            campaign_run_argv,
            campaign_runs_on_mocks,
            connected_instruments,
            write_launch_spec,
        )

        # Normally already answered by `_on_run`, which asks before the head gate
        # so a doomed launch prompts nobody. Repeated here because it is the
        # invariant rather than the ordering: nothing may spawn a child from a
        # file that is not the campaign on screen, including a caller that
        # reached this method by another route.
        if self._refuse_if_spec_is_unwritable(spec):
            return False

        head_up = self._head_state_after_gate()
        if head_up is None:
            self._sig_log.emit(
                "✗ Not started — the dispenser-head belief could not be read.")
            QMessageBox.critical(
                self, "Head position unknown",
                "The head position could not be read back after the start-gate, "
                "so it cannot be passed to the campaign.\n\n"
                "The loop drives the head with conditional commands, so a guess "
                "costs one wrong flip. Nothing was started.",
            )
            return False

        try:
            spec_path = write_launch_spec(spec, self._project_dir())
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("campaign_spec_write_failed", exc_info=True)
            QMessageBox.critical(
                self, "Could not write the campaign spec",
                f"{exc}\n\nNothing was started and the rig was not released.")
            return False

        argv = campaign_run_argv(
            spec_path, self._project_dir(),
            mock=campaign_runs_on_mocks(self._manager), head_up=head_up)

        held = connected_instruments(self._manager)
        if held and not self._confirm_release(held):
            return False

        self._sig_log.emit(f"  spec written to {spec_path}")
        self._schedule(
            self._release_rig_for_handover(),
            lambda ok: self._spawn_after_release(ok, spec, argv, spec_path))
        return True

    def _confirm_release(self, held: list[str]) -> bool:
        """Ask before disconnecting. A button that silently drops the rig is worse.

        The same opinion :class:`CalibrationLauncherDialog` states in its own
        launch button: handing the instruments over is a side effect large enough
        that it gets said out loud rather than discovered later.
        """
        reply = QMessageBox.question(
            self, "Release the instruments?",
            "This window currently holds:\n  " + ", ".join(held)
            + "\n\nThey will be disconnected so the campaign can own them for "
              "its whole run. Two processes cannot drive the rig at once.\n\n"
              "The campaign then runs in its own process: it keeps going when "
              "this window closes, and its board prompts have nobody to ask, so "
              "a full board stops the run instead of asking for a fresh one.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Ok

    async def _release_rig_for_handover(self) -> None:
        """Close the sessions, then give the rig back — in that order.

        Mirrors :meth:`softae.gui.tabs.tab_init.InitCalibrationTab._disconnect_all_and_release`
        exactly, including the ``finally``: releasing first would advertise a free
        rig while these ports are still closing, and a claim that outlived a
        failed disconnect would refuse the child we are about to start.
        """
        from softae.core.rig_session import release_rig_session

        try:
            await self._manager.disconnect_all()
        finally:
            release_rig_session()

    def _schedule(self, coro, done) -> None:
        """Run *coro* on the GUI's qasync loop; call ``done(ok)`` when it lands.

        The GUI runs on **qasync**, so there is already a loop driving Qt and the
        instruments' ``asyncio.Lock``s are bound to it: making a second loop here
        would either raise "already running" or take those locks from the wrong
        one. Overridable so a test can drive the handover with no loop at all.
        """
        import asyncio

        task = asyncio.ensure_future(coro)

        def _done(t) -> None:
            try:
                t.result()
                done(True)
            except Exception:
                logger.warning("release_instruments_failed", exc_info=True)
                done(False)

        task.add_done_callback(_done)

    def _spawn_after_release(self, ok: bool, spec, argv: list[str],
                             spec_path: Path) -> None:
        """Start the child only once this process genuinely holds nothing.

        Two conditions, not one. No open session is the obvious half; **no rig
        claim** is the half that decides whether the child can start at all,
        because :func:`~softae.core.run_lock.acquire_run_lock` would refuse it
        outright and it would exit with ``EXIT_BUSY`` into a log file nobody is
        watching.
        """
        from softae.core.run_lock import read_run_lock
        from softae.gui.campaign_launch import connected_instruments

        still_held = connected_instruments(self._manager) if ok else []
        try:
            claim = read_run_lock()
        except Exception:
            logger.warning("run_lock_unreadable_before_spawn", exc_info=True)
            claim = "unreadable"

        if ok and not still_held and claim is None:
            self._spawn_campaign_child(spec, argv, spec_path)
            return

        detail = ", ".join(still_held) if still_held else ""
        self._sig_log.emit("✗ Not started — the rig was not released cleanly.")
        QMessageBox.critical(
            self, "Could not release the rig",
            "The campaign was NOT started.\n\n"
            + ("The instruments did not disconnect cleanly"
               + (f" ({detail} still connected)" if detail else "") + ".\n"
               if not ok or still_held else "")
            + (f"The rig is still claimed — {claim}.\n"
               if claim is not None else "")
            + "\nStarting anyway would put two processes on the same ports.",
        )

    def _spawn_campaign_child(self, spec, argv: list[str],
                              spec_path: Path) -> None:
        """Spawn, record the handle, and tell the host it happened."""
        from datetime import datetime, timezone

        from softae.gui.campaign_launch import (
            DetachedCampaign,
            launch_log_path,
            spawn_campaign,
        )

        project = self._project_dir()
        log_file = launch_log_path(project, spec.name)
        try:
            pid = spawn_campaign(argv, log_file=log_file)
        except Exception as exc:
            logger.warning("campaign_launch_failed", exc_info=True)
            self._sig_log.emit(f"✗ Not started — {exc}")
            QMessageBox.critical(
                self, "Launch failed",
                f"{exc}\n\nThe instruments were released but no campaign was "
                f"started. Reconnect them from the Init tab when you are ready.")
            return

        child = DetachedCampaign(
            pid=pid, name=str(getattr(spec, "name", "campaign")),
            spec_path=Path(spec_path), log_path=log_file, project_dir=Path(project),
            started_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        )
        logger.info("campaign_detached", pid=pid, campaign=child.name,
                    spec=str(child.spec_path), log=str(log_file))
        self._sig_log.emit(
            f"▶ '{child.name}' started as PID {pid} — it now owns the rig.")
        self._on_campaign_spawned(child)

    def _on_campaign_spawned(self, child) -> None:
        """Host hook: the child is running. Default is to say so and stop there."""
        QMessageBox.information(self, "Campaign running", child.describe())

    def _head_state_after_gate(self) -> bool | None:
        """The head belief this process just registered, or ``None`` if unreadable.

        Read back rather than assumed.
        :func:`~softae.gui.widgets.head_check_dialog.verify_head_before_run`
        leaves the head raised on both of its accepting paths — reported raised,
        or reported lowered and then safety-retracted — so the answer is almost
        always ``True``; but "almost always" is not what may be written into
        another process's belief about a head with no position feedback.
        """
        try:
            return bool(self._manager.get("syringe").is_head_up())
        except Exception:
            logger.warning("head_state_unreadable", exc_info=True)
            return None

    def _refuse_if_spec_is_unwritable(self, spec) -> bool:
        """``True`` when a spec file could not carry this campaign faithfully.

        This is the cost of the handover, stated where the operator meets it. The
        campaign runs in a child now, and a child is started from a file — so a
        campaign whose file would silently become a *different* campaign cannot
        be started at all. The panel state is still written, so nothing typed is
        lost, and it is the same artifact a rig-busy refusal leaves.

        Shaped like :meth:`_refuse_if_rig_busy`, and called from the same place
        in the launch sequence for the same reason: **before** the head gate.
        That gate prompts the operator and can issue a safety retract, and a
        launch that is going to be refused must ask for nothing and move nothing.

        **The wording is written here rather than taken from**
        :meth:`~softae.core.rejected_launch.PreservedLaunch.describe`, and that
        is not duplication. That paragraph was written for a *rig-busy* refusal,
        where waiting is the answer, and it says two things that this refusal
        makes false: "press Run again once the rig is free" (the rig has nothing
        to do with it — pressing Run will refuse again forever) and "relaunching
        it through this tab is the only way to run the campaign you configured"
        (this tab no longer runs anything). A dialog that tells an operator to
        retry something that cannot succeed is worse than one that says so.
        """
        from softae.core.campaign_spec_io import spec_toml_completeness

        completeness = spec_toml_completeness(spec)
        if completeness.complete:
            return False

        preserved = self._preserve_rejected_launch(spec)
        self._sig_log.emit(
            "✗ Not started — this campaign cannot be written to a spec file: "
            + "; ".join(completeness.missing))
        saved = (f"\n\nNothing you entered was lost — it is saved at:\n"
                 f"    {preserved.panel_state_path}\n"
                 f'Reload it with "Load Config…" in this tab.'
                 if preserved.panel_state_path is not None else "")
        QMessageBox.warning(
            self, "This campaign cannot be launched",
            "The campaign runs in its own process now, started from a spec "
            "file — and a spec file cannot carry all of this one:\n\n"
            + completeness.explain()
            + "\n\nStarting it anyway would run a different experiment from the "
              "one on screen, without raising anything, so it is refused.\n\n"
              "Waiting will not help: this is about what a spec file can say, "
              "not about who holds the rig. Until those settings can be written "
              "to a file, this campaign cannot be started from here or from a "
              "terminal."
            + saved,
        )
        return True
