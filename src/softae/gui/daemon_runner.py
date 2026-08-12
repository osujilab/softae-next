"""Shared mixin for the daemon-runner tabs' shutdown contract.

``DaemonRunnerMixin`` codifies the ``abort_run()`` + ``cleanup()`` pair the four
daemon-runner tabs (Experiment, Sandbox, BO-Campaign, Arrhenius) previously
re-derived by hand.  Each tab's ``cleanup()`` body was byte-identical (abort
then bounded-join the stored ``threading.Thread``); only the abort call and the
thread attribute differed.  Those two differences become template hooks:

- ``_abort_run_impl()`` — fire the runner's existing cooperative-abort path.
- ``_runner_thread()`` — return the ``threading.Thread`` to join (or ``None``).

The mixin is a plain ``object`` subclass so it contributes only these two
methods with no Qt metaclass/parenting interaction; each tab declares it *first*
in the bases (``class Tab(DaemonRunnerMixin, QWidget)``) so ``abort_run`` and
``cleanup`` resolve to the shared implementation.
"""

from __future__ import annotations

from softae.gui._shutdown import DAEMON_JOIN_TIMEOUT


class DaemonRunnerMixin:
    """Codifies the daemon-runner shutdown contract for the four runner tabs.

    Provides signal-first ``abort_run()`` and bounded-join ``cleanup()``.
    Subclasses implement two hooks: how to signal their runner's existing
    cooperative abort, and which ``threading.Thread`` attribute to join.
    """

    def abort_run(self) -> None:
        """Signal the running daemon to abort (idempotent, no join)."""
        try:
            self._abort_run_impl()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Abort an in-progress run and join the daemon thread (bounded, idempotent)."""
        self.abort_run()
        t = self._runner_thread()
        if t is not None and t.is_alive():
            t.join(DAEMON_JOIN_TIMEOUT)

    # ── template hooks (subclass implements) ────────────────────────────
    def _abort_run_impl(self) -> None:
        """Fire the runner's existing cooperative-abort path (set flag / call abort)."""
        raise NotImplementedError

    def _runner_thread(self):  # -> threading.Thread | None
        """Return the daemon thread to join, or None when no run is active."""
        raise NotImplementedError
