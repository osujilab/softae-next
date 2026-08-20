"""Detect and report a previous session that died mid-experiment.

Every terminal campaign path finalizes its run row, so a run still carrying a
NULL ``finished_at`` at start-up is durable evidence that the process was killed
— a crash, a power cut, or an organisation-mandated update restart. This is the
most reliable layer of the shutdown story precisely because it does **not** race
the OS: it runs afterwards, when there is time to think.

**This is the primary protection for the head axis, not a belt-and-braces check.**
The dispenser head is a two-state *motor-driven flipper*: it **holds position when
de-energised** rather than returning to a safe one. So an unplanned stop can leave
the head **lowered over an electrode**, and nothing in the hardware will lift it.
Racing the OS to park was deliberately not attempted (a ~5 s budget against 2-5 s
of serial I/O, and `atexit` does not run under `TerminateProcess`), so reporting
the condition afterwards — when there is time to think — is how the operator
learns about it.

Only the thermal axis genuinely fails safe (heater output drops on comms loss),
and even that depends on the watchdog firing: a forced OS restart is **not** a
power loss, since the host reboots while the instruments stay powered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from PySide6.QtWidgets import QMessageBox, QWidget

from softae.core.run_lock import foreign_run_lock
from softae.core.shutdown import UNCLEAN_SHUTDOWN_MESSAGE

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)


def check_unclean_shutdown(
    parent: QWidget | None,
    manager: "InstrumentManager",
    data_store: "DataStore | None",
) -> bool:
    """Warn about unfinished runs and offer to park. ``True`` if a park ran.

    Best-effort: any failure here is logged and swallowed, because this runs
    during start-up and must never prevent the application from opening.
    """
    if data_store is None:
        return False

    # ── Liveness BEFORE recovery. This ordering is the fix — do not reorder. ──
    #
    # The mirror of the headless guard in `tools/campaign.py`, and it closes this
    # collision from the other direction: opening the GUI while a headless campaign
    # is running. `unfinished_runs()` is project-wide and a live run's row is
    # indistinguishable from a crashed one's, so asking it first would read the
    # *running* campaign's row, mark it `interrupted`, and offer to park the rig
    # out from under it — turning a start-up courtesy into the thing that kills an
    # unattended overnight run.
    #
    # Skipping defers rather than discards: the unfinished row is durable and never
    # clears itself, so a genuinely crashed run is still caught at the next launch
    # that is not racing a live campaign.
    #
    # An unreadable lock counts as "someone might be running". It is the only
    # conservative reading here: this function must never stop the GUI opening
    # (see the docstring), so raising is not an option, and guessing "free" would
    # go on to consume a live campaign's row on exactly the evidence we just
    # failed to obtain. Deferring costs a launch; guessing wrong costs the run.
    try:
        holder = foreign_run_lock()
    except Exception:
        logger.warning("rig_lock_unreadable_skipping_recovery", exc_info=True)
        return False

    if holder is not None:
        logger.info("unclean_shutdown_check_skipped",
                    holder_pid=holder.pid, holder_what=holder.what,
                    msg="another process holds the rig — a live run's row must "
                        "not be mistaken for a crashed one's")
        return False

    try:
        stale = data_store.unfinished_runs()
    except Exception:
        logger.warning("unfinished_runs_query_failed", exc_info=True)
        return False
    if not stale:
        return False

    names = ", ".join(r["run_id"] for r in stale[:3])
    more = f" (+{len(stale) - 3} more)" if len(stale) > 3 else ""
    logger.warning("unclean_shutdown_detected", count=len(stale), runs=names)

    # Durable record first — the operator may dismiss the dialog and forget.
    try:
        from softae.core.alerts import WARNING, Alert, raise_alert

        raise_alert(
            Alert(
                kind="unclean_shutdown",
                # One wording, defined in `core.shutdown`. This text was
                # duplicated verbatim there and here; two copies of a sentence
                # describing the same physical unknown is how the CLI and the
                # GUI come to describe it differently after one of them is edited.
                message=f"{len(stale)} {UNCLEAN_SHUTDOWN_MESSAGE}",
                severity=WARNING,
                run_id=stale[0]["run_id"],
                details={
                    "runs": [r["run_id"] for r in stale[:10]],
                    # The head does not self-retract; this is a real physical
                    # unknown, not boilerplate, so it belongs in the record.
                    "head_position_unknown": True,
                },
            ),
            data_store=data_store,
        )
    except Exception:
        logger.warning("unclean_shutdown_alert_failed", exc_info=True)

    box = QMessageBox(parent)
    box.setWindowTitle("Previous session ended unexpectedly")
    box.setIcon(QMessageBox.Icon.Warning)
    box.setText(
        f"{len(stale)} experiment run(s) did not finish cleanly:\n{names}{more}"
    )
    box.setInformativeText(
        "The previous session was killed before it could stop the rig — a crash, "
        "a power cut, or a forced restart.\n\n"
        "The dispenser head holds its position when power is lost, so it may have "
        "been left LOWERED over an electrode. Check the head before moving the "
        "stage, and treat any well it was over as suspect.\n\n"
        "The heater should have dropped on comms loss, but a forced OS restart "
        "does not cut instrument power, so that depends on its watchdog.\n\n"
        "Drive the rig to its safe state now?"
    )
    park_btn = box.addButton("Park now", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(park_btn)

    # ── Ask FIRST, stamp SECOND. This ordering is the fix — do not reorder. ──
    #
    # `finish_run` is an UPDATE with no unset, so relabelling a row `interrupted`
    # is permanent and dismissing the dialog cannot undo it. Stamping before the
    # dialog therefore rewrote the history of runs the operator was never shown,
    # and — while `tools/commission.py` and `tools/equilibration.py` still leaked
    # an unfinished row on their *success* path — it was relabelling completed
    # runs as interrupted. Both halves land together: those tools now finalize
    # their own rows, so a row that survives to here really is a crashed one, and
    # the relabelling now happens only after the operator has seen the list.
    #
    # A dialog that never appeared is not an ask. If `exec()` raises — no
    # display, no QApplication, a Qt teardown race at start-up — nothing is
    # stamped and the whole report returns intact at the next launch. That also
    # keeps the promise this function's docstring makes, which the bare `exec()`
    # did not: no failure here may stop the application opening.
    try:
        box.exec()
    except Exception:
        logger.warning("unclean_shutdown_dialog_failed", exc_info=True)
        return False

    # Stamped now that it has been reported, so it is reported once rather than
    # on every subsequent launch.
    #
    # Why the stamp is kept at all, rather than replaced by a suppression record
    # that leaves the run row untouched: "we already asked" and "this run was
    # interrupted" really are different facts, but `unfinished_runs()` is
    # project-wide and read by three surfaces — `core/shutdown.py`,
    # `tools/campaign.py` and here. A row left NULL forever re-arms all three,
    # so suppression living anywhere else (the alerts table, a sidecar) would
    # have to be taught to each of them separately. And `interrupted` is the
    # row's *true* terminal status: the write was mistimed, not untrue.
    #
    # Why closing via the window's X also stamps. Qt already resolves the close
    # box to Reject, which is the same answer as "Skip", and both mean the
    # operator was shown the rows and chose not to park. Re-prompting on X would
    # put this dialog in front of anyone who habitually dismisses it on every
    # launch — and a dialog seen every launch is one that gets dismissed unread,
    # which is a worse outcome than the bug on a prompt whose subject is a
    # possibly-lowered head. What carries the fact for an operator who X'd out is
    # the durable alert above, raised *before* the dialog for exactly that reason.
    for row in stale:
        try:
            data_store.finish_run(row["run_id"], "interrupted")
        except Exception:
            logger.warning("mark_interrupted_failed", run_id=row["run_id"])

    if box.clickedButton() is not park_btn:
        logger.info("unclean_shutdown_park_declined")
        return False

    try:
        from softae.core.safe_park import safe_park

        result = safe_park(manager, reason="recovery after unclean shutdown")
        # Two ways this recovery park can fail to make the rig safe, and the
        # second one used to be silent: something refused (``errors``), or
        # nothing was commanded at all because no instrument was connected —
        # which raises nothing and so passed the old ``result.ok`` check. That
        # is the likelier of the two here: this runs at start-up, and whether
        # the manager has connected yet is a start-up ordering question.
        # ``headline()`` is the one place that decides between them.
        text, severe = result.headline()
        if severe:
            QMessageBox.warning(parent, "Park incomplete",
                                text + "\n\n" + result.describe())
        return True
    except Exception:
        logger.warning("recovery_park_failed", exc_info=True)
        return False
