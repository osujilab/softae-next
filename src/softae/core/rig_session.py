"""The rig claim follows the **instrument session**, not the run that uses it.

:mod:`softae.core.run_lock` answers "who owns the rig"; every acquire site before
this one was a *run* — ``WorkflowExecutor.run``, ``run_autonomous_campaign``. So
the rig was claimed only while something was executing, and the process that had
actually opened the serial ports claimed nothing at all in between. The desktop
GUI is that process: in owner mode it arms the interlock and opens every port at
launch, and ``gui/app.py`` contained no reference to the lock. For the whole idle
life of an open window :func:`~softae.core.run_lock.foreign_run_lock` reported the
rig **free** while the GUI held every port, and a headless ``softae-campaign run``
started in that window passed its own guard and connected on top of them.

**The rule, stated once:** the operator already ruled that parking follows the
instrument session — whoever opened the ports is who parks them. Claiming is the
same rule read forwards. *Acquire when the ports open; release when they close.*
Connect All and Disconnect All then become the operator's visible hand-off of the
rig rather than two mechanisms free to disagree with the lock file.

This module **adds nothing to** :mod:`~softae.core.run_lock` and changes nothing
in it. It is a separate acquire site with a separate exemption rule, and it lives
in its own file so it can be reviewed as one.

Why the exemption rule here is not :func:`~softae.core.run_lock.rig_is_simulated`
-----------------------------------------------------------------------------
``rig_is_simulated`` delegates to
:func:`~softae.core.hardware_safety.probe_motion`, which is defined only over
``MOTION_INSTRUMENTS = ("stage", "syringe", "piezo")`` — the *arming interlock's*
vocabulary, and rightly so: that list decides what may move. But a session's claim
is not about motion, it is about **ports**. A manager with a real potentiostat and
a real heater but a mock stage reads as *simulated* to a motion-shaped question,
and a GUI that skipped the claim on that basis would hold two real instrument
sessions while telling every other process the rig was free — the exact defect
this module exists to close, reintroduced one instrument at a time.

:func:`session_is_simulated` is therefore a **superset** predicate, never a
competing one: everything ``rig_is_simulated`` calls real, this calls real too. It
only also counts the non-motion drivers, and it can only ever claim *more* often.
It does not widen ``MOTION_INSTRUMENTS`` and must not — changing that list changes
what gets armed.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import structlog

from softae.core.run_lock import (
    RunLock,
    acquire_run_lock,
    read_run_lock,
    release_run_lock,
)

logger = structlog.get_logger(__name__)

#: ``<kind>:`` for a claim held by an interactive **session** rather than a run.
#:
#: The ``what`` grammar in flight is ``<kind>:<name>:<run_id>``, whose shipped
#: instance is ``campaign:<name>:<run_id>``. A GUI session has no run id — it is
#: not a run, it is an open set of ports that may execute nothing at all for
#: hours — so the third field is **omitted rather than left empty**. A trailing
#: colon would assert "there is a run id and it is blank", which is what
#: :func:`softae.gui.widgets.rig_owner.campaign_identity` produces for a campaign
#: that failed to stamp its run; the two must not look alike in a lock file a
#: human reads to decide whether to take the rig.
SESSION_PREFIX = "gui:"

#: The desktop GUI's claim. Renders through
#: :meth:`~softae.core.run_lock.RunLock.describe` and
#: :func:`~softae.gui.widgets.rig_owner.owner_line` as
#: ``held by PID 8821 — gui:desktop, started 2026-08-19T14:02:00+00:00``, and
#: through :func:`~softae.core.run_lock.busy_rig_message` as a named holder a
#: person can walk over and look at. It carries no ``campaign:`` prefix, so
#: ``campaign_identity`` correctly finds no campaign to attach to: a GUI publishes
#: no event stream and offers no control channel, and offering either would be a
#: lie.
DESKTOP_SESSION = f"{SESSION_PREFIX}desktop"

#: Registered instruments that own **no port** of their own. ``liquid_handler``
#: is a coordinator over the stage and syringe, which are checked directly — the
#: same exclusion, for the same reason, as
#: :func:`~softae.core.hardware_safety.probe_motion`.
#:
#: **Still required after the move to** :func:`isinstance` **, and not an
#: instance of the same mistake.** The prefix rule needed this constant because
#: ``AsyncLiquidHandler`` is simulated-but-not-``Mock``-named, which made the
#: exception look like a patch over a naming heuristic. It is not:
#: ``AsyncLiquidHandler`` is the *real* coordinator too — the mock factory and
#: the real factory register the identical class — so no mock-detection rule of
#: any kind can classify it. It is excluded because its ports belong to the
#: stage and the syringe and are counted there, which is a fact about the
#: rig's topology rather than about anybody's class names.
PORTLESS_INSTRUMENTS = ("liquid_handler",)


@lru_cache(maxsize=1)
def _mock_driver_classes() -> tuple[type, ...]:
    """The shipped mock drivers, as **classes** — the thing a subclass keeps.

    Imported lazily so that reasoning about the rig lock does not drag the whole
    driver stack into a headless process that only wanted to read a lock file.

    A mock added to :mod:`softae.drivers` and forgotten here reads as *real*, so
    a fully simulated session claims the rig: one refused dry run. That is the
    direction this list is allowed to fail in, and it is the opposite of the
    prefix rule's, which failed by *widening* the set of things called mock
    the moment somebody named a subclass ``FastMockRHController``.
    """
    from softae.drivers.mock_camera import MockCamera, MockDACSwitch
    from softae.drivers.mock_espico import MockESPico
    from softae.drivers.mock_ht_sensor import MockHTSensor
    from softae.drivers.mock_keithley import MockKeithley
    from softae.drivers.mock_piezo import MockPiezoController
    from softae.drivers.mock_rh_controller import MockRHController
    from softae.drivers.mock_stage import MockStage
    from softae.drivers.mock_syringe import MockSyringe
    from softae.drivers.mock_temp_controller import MockTempController

    # ``MockLamp`` is an alias of ``MockDACSwitch``; listing it would add nothing.
    return (
        MockCamera,
        MockDACSwitch,
        MockESPico,
        MockHTSensor,
        MockKeithley,
        MockPiezoController,
        MockRHController,
        MockStage,
        MockSyringe,
        MockTempController,
    )


def session_is_simulated(manager: Any) -> bool:
    """Whether this session opens no real port at all, so the claim may be skipped.

    The single legitimate exemption, and it is narrow: two mock suites collide
    over nothing, and a mock run holding the rig turns a dry run into an outage
    for a real one.

    **Simulated means "is one of the shipped mock drivers", by type.** It used to
    mean "has a class name starting with ``Mock``", which is a statement about
    spelling: ``GridAwareMockPico(MockESPico)`` and ``FastMockRHController(
    MockRHController)`` — the fast-clock subclasses ``eis-validate --mock``
    installs — are wholly simulated and read as *real* under a prefix test. The
    cost was paid twice over: a ``--mock`` run at an operator's terminal took the
    machine-scope ``~/.softae/rig.lock`` and refused the live GUI, and
    :func:`softae.gui.campaign_launch.campaign_runs_on_mocks`, which asks this
    same question to decide whether the campaign child gets ``--mock``, would
    have launched that child *without* it — sending a simulated session's child
    at the real ports. :func:`isinstance` survives subclassing, which is the
    whole property the predicate needed and the one a name cannot have.

    **Every failure answers "real", and so does every driver this cannot place.**
    An unreadable enumeration, an unreadable driver, and a driver that is neither
    a known mock nor anything else recognisable all mean "I could not confirm
    this is simulated". Claiming a rig that turns out to be simulated costs at
    worst one refused dry run; skipping the claim on a rig that turns out to be
    real costs a port collision with whoever already had it open.
    """
    try:
        names = list(manager.names)
    except Exception:
        logger.warning(
            "rig_session_enumeration_failed", exc_info=True,
            msg="cannot list instruments — assuming this session opens real ports "
                "and claiming the rig")
        return False

    try:
        mock_classes = _mock_driver_classes()
    except Exception:
        logger.warning(
            "rig_session_mock_registry_unavailable", exc_info=True,
            msg="cannot import the mock drivers — nothing can be recognised as "
                "simulated, so this session claims the rig")
        return False

    for name in names:
        if name in PORTLESS_INSTRUMENTS:
            continue
        try:
            driver = manager.get(name)
        except Exception:
            logger.warning(
                "rig_session_probe_failed", instrument=name, exc_info=True,
                msg="cannot read this driver — assuming it is real")
            return False
        if not isinstance(driver, mock_classes):
            logger.debug("rig_session_real_instrument", instrument=name,
                         driver=type(driver).__name__)
            return False
    return True


def claim_rig_session(
    manager: Any,
    *,
    what: str = DESKTOP_SESSION,
    scope: str | Path | None = None,
    log_path: str = "",
) -> RunLock | None:
    """Claim the rig for the sessions this process is about to open.

    Returns the claim, or ``None`` when the rig is simulated and no claim was
    made. Raises :class:`~softae.core.run_lock.RunLockHeld` if another live
    process already owns the rig — the caller must then open nothing.

    Re-entrant, because :func:`~softae.core.run_lock.acquire_run_lock` is: a
    session that claims at launch and then runs a workflow gets its own claim
    handed back rather than a refusal.

    ``log_path`` defaults to empty **deliberately**. It is the field a campaign
    uses to publish its run directory, and pointing it at the GUI's project
    directory would offer a directory that may contain an *earlier* campaign's
    ``events.jsonl`` — a stream belonging to a run that is over, presented as the
    live holder's.
    """
    if session_is_simulated(manager):
        logger.info(
            "rig_session_claim_skipped", what=what,
            msg="every registered driver is a mock — nothing physical is at stake")
        return None

    lock = acquire_run_lock(scope, what, log_path=log_path)
    logger.info("rig_session_claimed", what=lock.what, pid=lock.pid)
    return lock


def release_rig_session(scope: str | Path | None = None) -> bool:
    """Give the rig back **if this process is what holds it**. Whether it did.

    A claim another process owns is left alone — an attached window's Disconnect
    All cannot free the campaign it is attached to. Ownership is checked here
    rather than left to :func:`~softae.core.run_lock.release_run_lock`'s own
    refusal, which logs a warning: closing a window that never held the rig is an
    ordinary act, and a warning on every attached exit is noise that teaches
    operators to ignore the line that matters.

    Checking ownership before releasing is not a race. The only claim this can
    remove is one this process holds, and no other process can take it away.
    """
    lock = read_run_lock(scope)
    if lock is None or not lock.is_mine():
        return False
    return release_run_lock(scope)


@contextmanager
def held_rig_session(
    manager: Any,
    *,
    what: str = DESKTOP_SESSION,
    scope: str | Path | None = None,
    log_path: str = "",
) -> Iterator[RunLock | None]:
    """Hold the rig for the duration of a block, releasing even on exception.

    For callers whose session has a lexical extent — a script, a tool, a test. The
    GUI cannot use this: its connect path is a coroutine scheduled onto the qasync
    loop and its release happens in a different callback entirely, so it uses the
    explicit :func:`claim_rig_session` / :func:`release_rig_session` pair.

    Releases only what *this* block created. Same ``mine_already`` discipline as
    :func:`~softae.core.run_lock.held_run_lock` and for the same reason: a nested
    block exiting must not hand away the rig its caller is still using.
    """
    before = read_run_lock(scope)
    mine_already = before is not None and before.is_mine()
    lock = claim_rig_session(manager, what=what, scope=scope, log_path=log_path)
    try:
        yield lock
    finally:
        if lock is not None and not mine_already:
            release_rig_session(scope)
