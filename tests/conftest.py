from __future__ import annotations

import os

# Run Qt headless during the test suite so widgets that call ``.show()``
# (e.g. the EIS Browser pop-out in test_popout_creates_window) never flash a
# real window on screen. Must be set before any QApplication is created; pytest
# imports conftest before collecting test modules, so this runs first.
# ``setdefault`` lets a developer override it (e.g. to watch a test visually).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(scope="session", autouse=True)
def rig_lock_scope(tmp_path_factory):
    """No test may see — or delete — the operator's real ``~/.softae/rig.lock``.

    :func:`softae.core.run_lock.read_run_lock` **unlinks a stale lock as a side
    effect**, deliberately, so that every caller who asks the question also
    repairs the answer. That is right on the rig and wrong in a test suite: the
    GUI runs on this machine while the suite does, and any widget that merely
    *renders* ownership (the Manual tab's banner polls on a 2 s timer) reaches
    the default scope without anyone writing a line of lock code.

    Redirecting :data:`~softae.core.run_lock.DEFAULT_SCOPE` at a session tmp dir
    closes that whole class at the root rather than one fixture at a time.
    ``tests/test_run_lock.py`` binds ``DEFAULT_SCOPE`` by value at import — before
    any fixture runs — so its assertion about the real home path still holds, and
    tests that want their own scope keep overriding this with ``monkeypatch``.
    """
    from _pytest.monkeypatch import MonkeyPatch
    from softae.core import run_lock

    mp = MonkeyPatch()
    mp.setattr(run_lock, "DEFAULT_SCOPE", tmp_path_factory.mktemp("rig_lock_scope"))
    yield
    mp.undo()


@pytest.fixture(scope="session")
def dead_owner_pid() -> int:
    """A PID whose process is genuinely gone, obtained without killing anything.

    ``experiments.owner_pid`` is what tells :meth:`DataStore.unfinished_runs` a
    *crashed* run from a *live* one, so any test about crash recovery needs a
    dead PID. Producing one by signalling a process would be a
    ``taskkill``-shaped act against a number the suite does not own (CLAUDE.md
    §5); instead a trivial child is **spawned and reaped**, so the number was
    ours and its process is provably finished.

    The value is re-checked at each use, because a session lasts long enough
    for the OS to reissue the number.
    """
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", ""],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()                      # reaped: terminated, and by us
    return proc.pid


@pytest.fixture
def crashed_run(dead_owner_pid):
    """Start a run row and leave it in the state a hard kill leaves it in.

    ``start_run`` stamps ``os.getpid()``, so a row a test creates is owned by
    the *live* pytest process and is correctly **excluded** from
    ``unfinished_runs()``. A test about crash recovery therefore has to say that
    the owner is gone, and this is the one place that knows how.
    """
    from softae.core.run_lock import _pid_alive

    def _crashed(store, workflow_name: str = "wf", **kwargs) -> str:
        run_id = store.start_run(workflow_name, **kwargs)
        # A negative PID is refused by `_pid_alive` before it asks the OS
        # anything, so it is the safe fallback if the number came back round.
        pid = dead_owner_pid if not _pid_alive(dead_owner_pid) else -1
        store._conn.execute(
            "UPDATE experiments SET owner_pid = ? WHERE run_id = ?", (pid, run_id))
        store._conn.commit()
        return run_id

    return _crashed


@pytest.fixture
def settle_qt():
    """Join a widget's one-shot command workers, then deliver what they emitted.

    A GUI test that asserts a driver was called has to synchronise on the *thread*
    that calls it. Polling ``button.isEnabled()`` with ``time.sleep`` synchronises
    on the clock instead, which buys both a race and a test whose cost is its
    timeout rather than its work; ``QThread.wait`` is exact and costs only as long
    as the command.

    Draining the queued signals afterwards is not cosmetic. ``_on_infuse`` wires
    ``failed`` to ``QMessageBox.warning``; an undelivered ``failed`` outlives the
    test that started it and is delivered by pytest-qt's post-teardown
    ``processEvents`` — *after* ``monkeypatch`` has put the real modal back — where
    it opens a dialog no test can dismiss and wedges the run indefinitely.

    Only :class:`_CommandWorker` is joined. The tab's PV-polling worker runs until
    stopped, so waiting on every :class:`QThread` child would block for the full
    timeout by design.
    """
    from softae.gui.tabs.tab_manual_workers import _CommandWorker

    def _settle(widget, timeout_ms: int = 5000) -> None:
        for worker in widget.findChildren(_CommandWorker):
            worker.wait(timeout_ms)
        QApplication.processEvents()

    return _settle
