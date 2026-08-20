"""Launch a bench sequence headlessly, then walk away from it.

Commissioning sweeps and geometry-series casts are **bench tasks that outlive the
window that started them** — a four-artifact commissioning pass is an afternoon, a cast
series plus anneal is overnight. Running them inside the GUI's event loop means the
sequence dies with the window, so the operator has to leave a desktop session open for
hours and cannot restart the GUI to look at anything.

So this dialog does not run anything. It **hands the rig over**: releases the GUI's
instruments *and its rig claim*, spawns a detached child, and gets out of the way.
Closing the dialog — or the whole application — leaves the child running.

The handover is explicit, and that is the design's one opinion. Two processes cannot
share the rig (see :mod:`softae.core.run_lock`), so *something* has to give up the
instruments. Doing it silently on launch would mean a button that quietly disconnects
hardware, which is exactly the kind of side effect that gets discovered at 2 a.m., so
the button says what it does: **Release instruments and launch**.

"And its rig claim" is not a detail. Since :mod:`softae.core.rig_session` the GUI
claims ``gui:desktop`` for the whole connected life of the window, so this dialog
must hand back two things rather than one — and it must stop treating that claim as
somebody else's. Reading it as "the rig is busy" disabled the launch button
permanently; leaving it standing at spawn time would have the child refused its own
rig lock and exit into a log file while this dialog reported a PID.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

# Read at the DEFAULT scope -- no argument. Passing ``project_dir`` here was a bug:
# the rig is one physical object per machine, `WorkflowExecutor.run()` acquires at
# ``~/.softae/rig.lock``, and a project-scoped read would have looked in a different
# file, found nothing, and reported "Free" for the entire length of a running sequence.
#
# **Two questions, two readers, and they are not interchangeable.**
#
# ``foreign_run_lock`` asks *may this window act* — is the rig held by somebody
# **else**? Since :mod:`softae.core.rig_session`, this process claims ``gui:desktop``
# for the whole connected life of the window, so ``read_run_lock`` answers "yes,
# held" continuously and a refusal built on it disabled this dialog's launch button
# permanently. That is the same distinction
# :meth:`softae.gui.tabs.tab_init.InitCalibrationTab._refuse_if_rig_held` and
# :func:`softae.gui.launch_mode.decide_launch_mode` already make.
#
# ``read_run_lock`` asks *is the rig free for the child* — and there, **our own claim
# counts against us**: the child's ``acquire_run_lock`` is re-entrant only within one
# process, so a claim this window forgot to release refuses the child outright and it
# dies with ``EXIT_BUSY`` into a log nobody is watching. It stays load-bearing in
# `_after_release`, which is the last gate before the spawn.
from softae.core.rig_session import release_rig_session
from softae.core.run_lock import break_run_lock, foreign_run_lock, read_run_lock

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: Sequences this dialog can start. Each maps to a console entry point, so the dialog
#: and a terminal run the *same* command — no GUI-only path that can drift from the CLI.
SEQUENCES: dict[str, dict[str, Any]] = {
    "Commissioning — short blank": {
        "argv": ["-m", "softae.tools.commission", "run", "blank_short"],
        "needs_channels": True,
        "hint": "a jumpered channel (CE-WE shorted), RE tied to CE",
    },
    "Commissioning — load blank": {
        "argv": ["-m", "softae.tools.commission", "run", "blank_load"],
        "needs_channels": True, "needs_nominal": True,
        "hint": "a precision resistor across CE-WE, RE tied to CE",
    },
    "Commissioning — reference capacitor": {
        "argv": ["-m", "softae.tools.commission", "run", "reference_cap"],
        "needs_channels": True, "needs_nominal": True,
        "hint": "a low-loss C0G/NP0 capacitor, RE tied to CE",
    },
    "Commissioning — reference resistor": {
        "argv": ["-m", "softae.tools.commission", "run", "reference_r"],
        "needs_channels": True, "needs_nominal": True,
        "hint": "a reference resistor; repeat once per impedance decade",
    },
    "Commissioning — open blank": {
        "argv": ["-m", "softae.tools.commission", "run", "blank_open"],
        "needs_channels": True,
        "hint": "a bare uncast board, RE TIED TO CE (else the RE floats)",
    },
    # NOT LISTED YET: "Geometry series — cast a plan".
    #
    # `softae-thickness cast --execute` currently prints how to build the workflow and
    # exits; it does not cast. Offering it here would spawn a detached process that
    # writes instructions to a log and returns, while this dialog reports "Started as
    # PID nnnn" — an operator could stand at the rig waiting for a cast that was never
    # going to happen, and lose the session. A menu entry that looks like the others
    # and silently does nothing is worse than an absent one, so it stays out until the
    # execute path and the `[geometry_series]` spec land together.
}

#: Whether a launched child takes :mod:`softae.core.run_lock`.
#:
#: **True since 2026-08-07.** Every sequence here runs its workflow through
#: :meth:`~softae.workflows.workflow_executor.WorkflowExecutor.run`, which acquires the
#: rig lock and releases it after teardown. That is deliberately the *only* acquire
#: site: wiring each CLI separately would need every current and future entry point to
#: remember, and the one that forgets does not fail — it silently defeats the lock for
#: everyone else.
#:
#: Kept as a named constant rather than deleted, because the claim "the child owns the
#: rig" is the one this dialog rests on, and it should be checkable in one place if the
#: acquire site ever moves.
CHILDREN_ACQUIRE_THE_LOCK = True


def _log_dir(project_dir: str | Path) -> Path:
    d = Path(project_dir).expanduser() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spawn_detached(argv: list[str], *, log_file: Path, cwd: str | None = None) -> int:
    """Start a child that survives this process. Returns its PID.

    ``DETACHED_PROCESS`` on Windows and ``start_new_session`` elsewhere: without them
    the child joins this process's console/session group and dies with it, which is the
    single behaviour this dialog exists to avoid.

    stdout and stderr go to *log_file* rather than a pipe. A pipe would need a reader
    for the child's whole life — reintroducing the dependency on the GUI staying open —
    and a full pipe buffer would block the sequence mid-sweep.
    """
    kwargs: dict[str, Any] = {"cwd": cwd}
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(f"\n=== {datetime.now(tz=timezone.utc).isoformat()} "
                 f"{' '.join(argv)} ===\n")
        fh.flush()
        proc = subprocess.Popen(
            [sys.executable, *argv], stdout=fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **kwargs)
    logger.info("headless_sequence_launched", pid=proc.pid, argv=argv,
                log=str(log_file))
    return proc.pid


class CalibrationLauncherDialog(QDialog):
    """Pick a bench sequence, hand the rig over, walk away."""

    def __init__(self, manager: "InstrumentManager", project_dir: str,
                 parent: Any = None, *, schedule: Any = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._project = project_dir
        # How to run a coroutine. The GUI runs on **qasync**, so there is already a
        # loop driving Qt and the instruments' asyncio.Locks are bound to it: making a
        # second loop here and calling run_until_complete would either raise
        # "already running" or hand disconnect_all() locks from the wrong loop, which
        # is the failure `InstrumentManager.reset_locks` exists to repair. Injectable
        # so a test can drive it without a loop at all.
        self._schedule = schedule or self._default_schedule
        self.setWindowTitle("Calibration & bench sequences")
        self.setMinimumWidth(620)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<b>Runs headlessly.</b> The sequence keeps going when this dialog — or "
            "the whole application — closes."))
        if not CHILDREN_ACQUIRE_THE_LOCK:
            # Kept rather than deleted: if the acquire site ever moves and the constant
            # goes False, the gap should reappear in the product, not only in a diff.
            warn = QLabel(
                "⚠ The rig lock is not yet enforced by launched sequences. This "
                "window releases the instruments before launching, but nothing stops "
                "another process connecting mid-run — don't start one.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #b00020;")
            layout.addWidget(warn)

        # ── what to run ──────────────────────────────────────────────────────
        pick = QGroupBox("Sequence")
        pick_l = QVBoxLayout(pick)
        self._combo = QComboBox()
        self._combo.addItems(list(SEQUENCES))
        self._combo.currentTextChanged.connect(self._on_sequence_changed)
        pick_l.addWidget(self._combo)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #b26a00;")
        pick_l.addWidget(self._hint)

        row = QHBoxLayout()
        self._channels = QLineEdit()
        self._channels.setPlaceholderText("channels, e.g. 1-8")
        self._nominal = QLineEdit()
        self._nominal.setPlaceholderText("marked value (ohms / farads)")
        self._plan = QLineEdit()
        self._plan.setPlaceholderText("plan id, e.g. geo-1")
        for w in (self._channels, self._nominal, self._plan):
            row.addWidget(w)
        pick_l.addLayout(row)
        layout.addWidget(pick)

        # ── who owns the rig ─────────────────────────────────────────────────
        state = QGroupBox("Rig")
        state_l = QVBoxLayout(state)
        self._state_label = QLabel()
        self._state_label.setWordWrap(True)
        state_l.addWidget(self._state_label)
        self._btn_break = QPushButton("Take over (owner is gone)")
        self._btn_break.clicked.connect(self._on_break_lock)
        state_l.addWidget(self._btn_break)
        layout.addWidget(state)

        # ── launch ───────────────────────────────────────────────────────────
        self._btn_launch = QPushButton("Release instruments and launch")
        self._btn_launch.setStyleSheet(
            "QPushButton { background: #f57c00; color: white; font-weight: bold;"
            " padding: 8px; }")
        self._btn_launch.clicked.connect(self._on_launch)
        layout.addWidget(self._btn_launch)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        layout.addWidget(self._log)

        btns = QHBoxLayout()
        btns.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(close)
        layout.addLayout(btns)

        self._on_sequence_changed(self._combo.currentText())

        # The owner can change without this dialog doing anything — a child may exit,
        # or a terminal run may start — so the state is polled rather than read once.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_state)
        self._timer.start(2000)
        self._refresh_state()

    # ── state ────────────────────────────────────────────────────────────────

    def _on_sequence_changed(self, name: str) -> None:
        spec = SEQUENCES.get(name, {})
        self._hint.setText("Install: " + str(spec.get("hint", "")))
        self._channels.setVisible(bool(spec.get("needs_channels")))
        self._nominal.setVisible(bool(spec.get("needs_nominal")))
        self._plan.setVisible(bool(spec.get("needs_plan")))

    def _refresh_state(self) -> None:
        """Who owns the rig, and therefore whether this dialog may hand it over.

        A **foreign** claim is the only one that means busy. This window's own
        ``gui:desktop`` claim is not somebody else driving the rig — it is this
        session holding its ports, which is exactly what the launch button
        releases, so reporting it as busy disabled the button for the whole life
        of the window.
        """
        foreign = foreign_run_lock()
        mine = read_run_lock() if foreign is None else None
        held = 0
        try:
            held = sum(1 for s in self._manager.list_instruments()
                       if s.get("connected"))
        except Exception:
            pass

        if foreign is not None:
            self._state_label.setText(
                f"<b style='color:#b00020'>Busy.</b> {foreign.describe()}")
            self._btn_break.setEnabled(True)
            self._btn_launch.setEnabled(False)
            self._btn_launch.setText("Rig is busy")
            return

        self._btn_break.setEnabled(False)
        self._btn_launch.setEnabled(True)
        if held or mine is not None:
            what = (f"<b>{held}</b> instrument(s)" if held
                    else "the rig claim for this session")
            self._state_label.setText(
                f"Free. This window holds {what} — launching releases "
                f"{'them' if held else 'it'} first.")
            self._btn_launch.setText("Release instruments and launch")
        else:
            self._state_label.setText("Free. No instruments held here.")
            self._btn_launch.setText("Launch")

    def _on_break_lock(self) -> None:
        # Foreign, for the same reason: "take over (owner is gone)" is meaningless
        # against this window's own claim, and breaking it would leave the session
        # holding open ports while the lock file says the rig is free.
        lock = foreign_run_lock()
        if lock is None:
            return
        # PID reuse means a lock can read as live when its owner is long gone, so an
        # override must exist -- but it must be a person's decision after reading who
        # holds it, never something a code path can take.
        reply = QMessageBox.warning(
            self, "Take the rig?",
            f"{lock.describe()}\n\n"
            "Only do this if that process is genuinely gone. If it is still running, "
            "two processes will drive the rig at once.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            broken = break_run_lock()
            self._append(f"took the rig from PID {broken.pid if broken else '?'}")
            self._refresh_state()

    # ── launch ───────────────────────────────────────────────────────────────

    def _argv(self) -> list[str] | None:
        name = self._combo.currentText()
        spec = SEQUENCES.get(name)
        if not spec:
            return None
        argv = list(spec["argv"])
        if spec.get("needs_channels"):
            text = self._channels.text().strip()
            if not text:
                QMessageBox.warning(self, "Channels", "This sequence needs channels.")
                return None
            argv += ["--channels", text]
        if spec.get("needs_nominal"):
            text = self._nominal.text().strip()
            if not text:
                QMessageBox.warning(
                    self, "Marked value",
                    "This part's marked value is required. The marking and the "
                    "measurement disagreeing is the check that catches an unusable "
                    "reference — it needs both numbers.")
                return None
            argv += ["--nominal", text]
        if spec.get("needs_plan"):
            text = self._plan.text().strip()
            if not text:
                QMessageBox.warning(self, "Plan", "This sequence needs a plan id.")
                return None
            argv += ["--plan", text]
        argv += ["--project", str(self._project)]
        if "commission" in " ".join(argv):
            argv += ["--yes"]          # the dialog already prompted
        return argv

    def _on_launch(self) -> None:
        argv = self._argv()
        if argv is None:
            return

        # The button reflects a 2 s poll; this is the check that decides. Foreign
        # only — our own session claim is not a second owner, it is what we are
        # about to give back.
        lock = foreign_run_lock()
        if lock is not None:
            QMessageBox.warning(self, "Rig is busy", lock.describe())
            self._refresh_state()
            return

        try:
            held = [s.get("name") for s in self._manager.list_instruments()
                    if s.get("connected")]
        except Exception:
            held = []

        if held:
            reply = QMessageBox.question(
                self, "Release the instruments?",
                "This window currently holds:\n  "
                + ", ".join(str(h) for h in held)
                + "\n\nThey will be disconnected so the headless sequence can use "
                  "them. Two processes cannot drive the rig at once.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Ok:
                return
            # Disconnect is a coroutine on the running qasync loop, so the spawn has to
            # happen in its completion callback rather than after a blocking wait.
            self._btn_launch.setEnabled(False)
            self._btn_launch.setText("Releasing instruments…")
            self._schedule(self._release_rig_for_handover(),
                           lambda ok: self._after_release(ok, argv))
            return

        # Nothing open, but the claim may still be ours — the session claim
        # outlives the last disconnect only by mistake, and a claim that outlives
        # it here would refuse the child we are about to start.
        release_rig_session()
        self._after_release(True, argv)

    async def _release_rig_for_handover(self) -> None:
        """Close the sessions, then give the rig back — in that order.

        Mirrors ``InitCalibrationTab._disconnect_all_and_release`` and
        :class:`~softae.gui.tabs._campaign_handover.CampaignHandoverMixin`'s
        method of the same name exactly, including the ``finally``.

        **Releasing the claim is not optional here**, and it is the half this
        dialog was missing once the GUI began claiming ``gui:desktop`` for its
        connected life. ``WorkflowExecutor.run()`` in the child acquires the rig
        lock, re-entrancy is per-process, so a claim still standing when the child
        starts refuses it — and the failure is invisible: the child exits into a
        log file while this dialog reports "Started as PID nnnn".

        Order matters in the other direction too: releasing *first* would
        advertise a free rig while these ports are still closing.
        """
        try:
            await self._manager.disconnect_all()
        finally:
            release_rig_session()

    def _default_schedule(self, coro: Any, done: Any) -> None:
        import asyncio

        task = asyncio.ensure_future(coro)

        def _cb(t: Any) -> None:
            try:
                t.result()
                done(True)
            except Exception:
                logger.warning("release_instruments_failed", exc_info=True)
                done(False)

        task.add_done_callback(_cb)

    def _after_release(self, ok: bool, argv: list[str]) -> None:
        """Spawn only once this process genuinely holds nothing.

        Two conditions, not one. No open session is the obvious half; **no rig
        claim at all** — ours included, hence :func:`read_run_lock` rather than
        :func:`foreign_run_lock` — is the half that decides whether the child can
        start, because its own acquire would refuse it outright.
        """
        still_held = []
        try:
            still_held = [s.get("name") for s in self._manager.list_instruments()
                          if s.get("connected")]
        except Exception:
            ok = False

        try:
            claim = read_run_lock()
        except Exception:
            logger.warning("run_lock_unreadable_before_spawn", exc_info=True)
            claim = "unreadable"

        self._refresh_state()
        if not ok or still_held or claim is not None:
            QMessageBox.critical(
                self, "Could not release",
                "The sequence was NOT started.\n\n"
                + ("The instruments did not disconnect cleanly"
                   + (f" ({', '.join(str(h) for h in still_held)} still connected)"
                      if still_held else "") + ".\n"
                   if not ok or still_held else "")
                + (f"The rig is still claimed — {claim}.\n"
                   if claim is not None else "")
                + "\nLaunching anyway would put two processes on the same ports, "
                  "or hand the child a rig it will be refused.")
            return
        self._spawn(argv)

    def _spawn(self, argv: list[str]) -> None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_file = _log_dir(self._project) / f"{stamp}_{self._combo.currentText()[:24]}.log"
        log_file = Path(str(log_file).replace(" ", "_").replace("—", "-"))
        try:
            pid = spawn_detached(argv, log_file=log_file)
        except Exception as exc:
            logger.warning("headless_launch_failed", exc_info=True)
            QMessageBox.critical(self, "Launch failed", str(exc))
            return

        self._append(f"launched PID {pid}\n  log: {log_file}")
        QMessageBox.information(
            self, "Running",
            f"Started as PID {pid}.\n\nIt keeps running if you close this dialog or "
            f"the application.\n\nLog:\n{log_file}")
        self._refresh_state()

    def _append(self, text: str) -> None:
        self._log.appendPlainText(text)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt name
        # Explicitly does NOT stop anything: the child is detached, so closing this
        # window is meant to be free.
        #
        # The child keeps the rig because `WorkflowExecutor.run()` holds the rig lock
        # for the length of the sequence, not merely because this window disconnected —
        # so a third party that tries to start a run mid-sequence is refused rather
        # than colliding.
        self._timer.stop()
        super().closeEvent(event)
