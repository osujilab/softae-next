"""Shared instrument polling service.

Consolidates the three independent background polling workers
(``_PollingWorker`` in ``tab_monitor``, ``_SidebarPollWorker`` in
``monitor_sidebar``, ``_StatusWorker`` in ``status_indicator``) into a
single thread that polls all instruments once per 2-second cycle, eliminating
serial-port contention when multiple workers previously issued concurrent
requests to the same COM-port instruments.

Usage
-----
Create one :class:`InstrumentPoller` in :class:`~softae.gui.main_window.MainWindow`
and pass it as the ``poller=`` keyword argument to :class:`MonitoringTab`,
:class:`MonitorSidebar`, and :class:`InstrumentStatusBar`.  Each consumer
connects its existing slot to one of the three emitted signals; no slot
signature changes are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QMutex, QWaitCondition, Signal

from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.server.manager import InstrumentManager


class InstrumentPoller(StoppableWorker):
    """Single background thread that polls all live instruments every 2 s.

    Emits three signals whose dict shapes are identical to those emitted by
    the per-widget workers they replace, so existing consumer slots require
    no changes.

    Signals
    -------
    status_ready : dict
        ``{name: {state: ..., ...}}`` — compatible with ``_StatusWorker``.
    sidebar_ready : dict
        Stage / syringe / temp / RH readings — compatible with
        ``_SidebarPollWorker``.
    monitor_ready : dict
        Temp / RH / stage readings — compatible with ``_PollingWorker``.
    """

    status_ready  = Signal(dict)   # → InstrumentStatusBar._apply_statuses
    sidebar_ready = Signal(dict)   # → MonitorSidebar._on_poll_done
    monitor_ready = Signal(dict)   # → MonitoringTab._on_poll_done

    def __init__(self, manager: "InstrumentManager", parent=None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def poke(self) -> None:
        """Wake the timed sleep for an immediate poll (safe to call from any thread)."""
        self._condition.wakeOne()

    def _wake(self) -> None:
        """Wake the timed wait so stop is prompt (interruption family)."""
        self.poke()

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self._do_poll()
            self._mutex.lock()
            self._condition.wait(self._mutex, 2000)
            self._mutex.unlock()

    def _do_poll(self) -> None:  # noqa: C901
        import math

        # ── Status (InstrumentStatusBar) ──────────────────────────────────
        try:
            statuses = self._manager.status_all()
            self.status_ready.emit(statuses)
        except Exception:
            pass

        # ── Shared instrument readings — one call per instrument ───────────
        sidebar: dict = {}
        monitor: dict = {}

        try:
            stage = self._manager.get("stage")
            pos = stage.live_position()
            x, y = float(pos[0]), float(pos[1])
            sidebar["stage_pos"] = (x, y)
            monitor["pos_x"] = x
            monitor["pos_y"] = y
        except Exception:
            pass

        try:
            syringe = self._manager.get("syringe")
            is_up = getattr(syringe, "_is_up", None)
            if is_up is None and callable(getattr(syringe, "status", None)):
                is_up = syringe.status().get("is_up")
            sidebar["syringe_is_up"] = is_up
        except Exception:
            pass

        try:
            tc = self._manager.get("temp_controller")
            sp = tc.get_sp()
            pv = tc.get_pv()
            sidebar["temp_sp"] = sp
            sidebar["temp_pv"] = pv
            monitor["temp_sp"] = sp
            monitor["temp_pv"] = pv
        except Exception:
            pass

        # RH + chamber temperature — both come from the RH sensor (SHT31-D),
        # which the controller reads in a single I²C transaction.  Chamber T
        # is distinct from the stage temperature (``temp_controller``).
        try:
            rh_ctrl = self._manager.get("rh_controller")
            st = rh_ctrl.status()
            sp_rh = st.get("setpoint", float("nan"))
            pv_rh = st.get("current_rh", float("nan"))
            h = pv_rh
            ct = st.get("chamber_temp", float("nan"))
            if callable(getattr(rh_ctrl, "get_TH", None)):
                try:
                    t_raw, h_raw = rh_ctrl.get_TH()
                    if not math.isnan(h_raw):
                        h = h_raw
                    if not math.isnan(t_raw):
                        ct = t_raw
                except Exception:
                    pass
            elif callable(getattr(rh_ctrl, "get_H", None)):
                try:
                    h_raw = rh_ctrl.get_H()
                    if not math.isnan(h_raw):
                        h = h_raw
                except Exception:
                    pass
            sidebar["rh_sp"] = sp_rh
            sidebar["rh_pv"] = pv_rh
            monitor["rh"] = h
            monitor["rh_sp"] = sp_rh
            if ct is not None and not math.isnan(ct):
                sidebar["chamber_temp"] = ct
                monitor["chamber_temp"] = ct
        except Exception:
            pass

        self.sidebar_ready.emit(sidebar)
        self.monitor_ready.emit(monitor)
