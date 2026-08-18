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

    # Mark them so this is reported once, not on every subsequent launch.
    for row in stale:
        try:
            data_store.finish_run(row["run_id"], "interrupted")
        except Exception:
            logger.warning("mark_interrupted_failed", run_id=row["run_id"])

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
    box.exec()

    if box.clickedButton() is not park_btn:
        logger.info("unclean_shutdown_park_declined")
        return False

    try:
        from softae.core.safe_park import safe_park

        result = safe_park(manager, reason="recovery after unclean shutdown")
        if not result.ok:
            QMessageBox.warning(
                parent, "Park incomplete",
                "Some subsystems did not go safe:\n" + "\n".join(result.errors),
            )
        return True
    except Exception:
        logger.warning("recovery_park_failed", exc_info=True)
        return False
