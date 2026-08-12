"""Tab 1: Initialization & Calibration.

Instrument discovery table, connect/disconnect controls, configuration
editor, stage calibration, position map, and PCB selector.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QMutex, Qt, QWaitCondition, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

from softae.gui.widgets.position_map import PositionMapWidget
from softae.gui.widgets.worker_thread import StoppableWorker


class _TablePollWorker(StoppableWorker):
    """Background thread that polls list_instruments() and channel_routing()
    every 2 s, keeping all serial and TOML I/O off the GUI thread.

    Supports an immediate re-poll via :meth:`poke` (e.g. after
    connect/disconnect) using a QWaitCondition timed sleep.
    """

    # statuses, routing_labels, pico_ports, rig_lock (RunLock | None)
    poll_ready = Signal(list, dict, dict, object)

    def __init__(self, manager, parent=None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def poke(self) -> None:
        """Wake the sleep early for an immediate poll (call from GUI thread)."""
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

    def _do_poll(self) -> None:
        statuses: list = []
        routing_labels: dict = {}
        pico_ports: dict = {"pico1": "(not connected)", "pico2": "(not connected)"}
        try:
            statuses = self._manager.list_instruments()
            _annotate_busy(self._manager, statuses)
        except Exception:
            pass
        try:
            from softae.config.loader import channel_routing
            r = channel_routing()
            p1 = r.get("pico1_range", [1, 16])
            p2 = r.get("pico2_range", [17, 32])
            routing_labels = {
                "pico1": f"EmStat Pico  [ch {p1[0]}–{p1[1]}]",
                "pico2": f"EmStat Pico  [ch {p2[0]}–{p2[1]}]",
            }
        except Exception:
            pass
        try:
            for pico_name in ("pico1", "pico2"):
                pico = self._manager.get(pico_name)
                port = pico.status().get("port") or "(auto)"
                pico_ports[pico_name] = str(port)
        except Exception:
            pass
        self.poll_ready.emit(statuses, routing_labels, pico_ports, _read_rig_lock())


def _annotate_busy(manager, statuses: list) -> None:
    """Stamp ``busy`` onto each status dict from the instrument's own lock.

    Read here rather than added to every driver's ``status()`` because
    :class:`~softae.server.base_instrument.InstrumentState` is the *connection*
    lifecycle, and every driver, test and log line reads its four members. Busy is a
    different axis — a connected instrument is busy or not without changing state — so
    it travels beside the state rather than becoming a fifth member of it.

    ``locked()`` is a plain attribute read, safe from the poll thread. It is also a
    **snapshot**: a 2 s poll against sub-second lock holds under-reports badly, which
    is exactly why nothing gates on it (see :func:`_compose_state`).
    """
    for status in statuses:
        name = status.get("name")
        if not name:
            continue
        try:
            lock = getattr(manager.get(name), "_lock", None)
            status["busy"] = bool(lock is not None and lock.locked())
        except Exception:
            status["busy"] = False


def _read_rig_lock():
    """The cross-process rig lock, or ``None``. Never raises — this only decorates."""
    try:
        from softae.core.run_lock import read_run_lock

        return read_run_lock()
    except Exception:
        return None


def _owner_line(rig_lock) -> str:
    """One-line owner summary for a table cell — ``describe()`` is multi-line.

    Names the PID, the run and the start time rather than just "busy", because PID
    reuse means the lock can read as live when its owner is long gone (see
    :mod:`softae.core.run_lock`). A person can tell "commissioning blank_short, started
    14:02" from a stale number; a check cannot.
    """
    what = rig_lock.what or "unnamed run"
    when = rig_lock.started_at or "unknown time"
    return f"held by PID {rig_lock.pid} — {what}, started {when}"


def _compose_state(state: str, *, busy: bool = False, rig_lock=None) -> str:
    """The State cell's text: connection lifecycle, plus who is using the rig.

    Two different facts, deliberately not merged into one enum:

    ``OCCUPIED``
        A **different process** holds the cross-process rig lock. It is file-backed and
        machine-scoped, so it is the one signal here authoritative enough to gate on —
        and it dominates the row, because while a headless child owns the hardware this
        GUI's per-instrument view describes connections it is not free to use. It shows
        on every row whatever the state, since the constraint is rig-wide.

    ``CONNECTED · ACTIVE`` / ``CONNECTED · IDLE``
        When the rig is ours (no lock, or a lock this process owns), the per-instrument
        ``asyncio.Lock`` is the truth. ``IDLE`` is stated outright rather than left
        implied: a bare ``CONNECTED`` reads as "fine" whether or not something is
        running, and that ambiguity is what this replaces.
    """
    if rig_lock is not None and not rig_lock.is_mine():
        return "OCCUPIED"
    if state == "CONNECTED":
        return "CONNECTED · ACTIVE" if busy else "CONNECTED · IDLE"
    return state


def _load_pcb_configs() -> dict[str, dict[str, Any]]:
    """Load ``[pcb.*]`` sections from the TOML config file."""
    try:
        from softae.config.loader import load
        cfg = load()
        return cfg.get("pcb", {})
    except Exception:
        return {}


class InitCalibrationTab(QWidget):
    """Instrument connection and stage calibration interface."""

    #: Emitted when the active PCB changes; carries the total electrode/channel count.
    pcb_channel_count_changed = Signal(int)

    def __init__(
        self,
        manager: InstrumentManager,
        parent: QWidget | None = None,
        data_store: "DataStore | None" = None,
    ):
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._rig_lock = None          # last polled cross-process rig lock
        self._pcb_configs = _load_pcb_configs()
        self._build_ui()
        self._start_polling()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # ── Main vertical splitter (top section | bottom section) ────────────
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        root.addWidget(main_splitter)

        # ── TOP SECTION: Instrument Status (left) | Stage Cal + Pico (right) ─
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(top_splitter)

        # ── LEFT: Instrument Status ──────────────────────────────────────────
        grp = QGroupBox("Instrument Status")
        grp_layout = QVBoxLayout(grp)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Instrument", "Type", "State", "Details"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        grp_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_connect_all = QPushButton("Connect All")
        self._btn_connect_all.clicked.connect(self._on_connect_all)
        btn_row.addWidget(self._btn_connect_all)

        self._btn_disconnect_all = QPushButton("Disconnect All")
        self._btn_disconnect_all.clicked.connect(self._on_disconnect_all)
        btn_row.addWidget(self._btn_disconnect_all)

        self._btn_connect_sel = QPushButton("Connect Selected")
        self._btn_connect_sel.clicked.connect(self._on_connect_selected)
        btn_row.addWidget(self._btn_connect_sel)

        self._btn_disconnect_sel = QPushButton("Disconnect Selected")
        self._btn_disconnect_sel.clicked.connect(self._on_disconnect_selected)
        btn_row.addWidget(self._btn_disconnect_sel)

        btn_row.addStretch()

        # Bench sequences live here because they are configuration, not experiments:
        # a calibration set is a durable asset reused across campaigns. Kept behind a
        # dialog rather than inlined -- this tab is already dense, and the dialog can
        # be closed while the sequence it launched keeps running.
        self._btn_bench = QPushButton("Calibration && Bench Sequences…")
        self._btn_bench.setToolTip(
            "Commissioning sweeps that run headlessly and survive closing the app.")
        self._btn_bench.clicked.connect(self._on_bench_sequences)
        btn_row.addWidget(self._btn_bench)

        grp_layout.addLayout(btn_row)
        top_splitter.addWidget(grp)

        # ── RIGHT: vertical splitter (Stage Cal | Pico Port Assignment) ──────
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(right_splitter)

        # Stage Calibration
        cal_grp = QGroupBox("Stage Calibration")
        cal_layout = QGridLayout(cal_grp)

        # Load saved calibration defaults (falls back to (0,0)/(43.5,50) if absent)
        try:
            from softae.config.loader import stage_calibration as _stage_cal
            _cal = _stage_cal()
        except Exception:
            _cal = {
                "home_x": 0.0, "home_y": 0.0, "dep1_x": 43.5, "dep1_y": 50.0,
                "flush_x": 0.0, "flush_y": 0.0, "wick_x": 0.0, "wick_y": 0.0,
            }

        # One (X, Y) spinbox pair + Set-current button per named position.
        def _add_position_row(row: int, label: str, xkey: str, ykey: str, slot):
            cal_layout.addWidget(QLabel(f"{label} X:"), row, 0)
            spin_x = QDoubleSpinBox()
            spin_x.setRange(-200, 200)
            spin_x.setDecimals(3)
            spin_x.setValue(_cal.get(xkey, 0.0))
            cal_layout.addWidget(spin_x, row, 1)

            cal_layout.addWidget(QLabel(f"{label} Y:"), row, 2)
            spin_y = QDoubleSpinBox()
            spin_y.setRange(-200, 200)
            spin_y.setDecimals(3)
            spin_y.setValue(_cal.get(ykey, 0.0))
            cal_layout.addWidget(spin_y, row, 3)

            btn = QPushButton(f"Set Current → {label}")
            btn.clicked.connect(slot)
            cal_layout.addWidget(btn, row, 4)
            return spin_x, spin_y, btn

        self._spin_home_x, self._spin_home_y, self._btn_set_home = _add_position_row(
            0, "Home", "home_x", "home_y", self._on_set_home
        )
        self._spin_dep1_x, self._spin_dep1_y, self._btn_set_dep1 = _add_position_row(
            1, "Dep-1", "dep1_x", "dep1_y", self._on_set_dep1
        )
        self._spin_flush_x, self._spin_flush_y, self._btn_set_flush = _add_position_row(
            2, "Flush", "flush_x", "flush_y", self._on_set_flush
        )
        self._spin_wick_x, self._spin_wick_y, self._btn_set_wick = _add_position_row(
            3, "Wick", "wick_x", "wick_y", self._on_set_wick
        )

        self._btn_go_home = QPushButton("Go Home")
        self._btn_go_home.clicked.connect(self._on_go_home)
        cal_layout.addWidget(self._btn_go_home, 4, 0)

        self._btn_go_dep1 = QPushButton("Go Dep-1")
        self._btn_go_dep1.clicked.connect(self._on_go_dep1)
        cal_layout.addWidget(self._btn_go_dep1, 4, 1)

        self._btn_go_flush = QPushButton("Go Flush")
        self._btn_go_flush.clicked.connect(self._on_go_flush)
        cal_layout.addWidget(self._btn_go_flush, 4, 2)

        self._btn_go_wick = QPushButton("Go Wick")
        self._btn_go_wick.clicked.connect(self._on_go_wick)
        cal_layout.addWidget(self._btn_go_wick, 4, 3)

        self._lbl_stage_pos = QLabel("Current position: (-- , --)")
        cal_layout.addWidget(self._lbl_stage_pos, 4, 4)

        self._chk_autosave_cal = QCheckBox("Auto-save positions to config on change")
        self._chk_autosave_cal.setToolTip(
            "When checked, any edit to a position value is written to "
            "softae_config.toml immediately."
        )
        cal_layout.addWidget(self._chk_autosave_cal, 5, 0, 1, 5)

        self._btn_save_cal = QPushButton("Save Positions to Config")
        self._btn_save_cal.setToolTip(
            "Persist Home, Dep-1, Flush, and Wick coordinates to softae_config.toml"
        )
        self._btn_save_cal.clicked.connect(self._on_save_calibration)
        cal_layout.addWidget(self._btn_save_cal, 6, 0, 1, 5)

        self._spin_dep1_x.valueChanged.connect(self._on_dep1_changed)
        self._spin_dep1_y.valueChanged.connect(self._on_dep1_changed)

        # Auto-save hook: any position edit persists to TOML when enabled.
        for _spin in (
            self._spin_home_x, self._spin_home_y,
            self._spin_dep1_x, self._spin_dep1_y,
            self._spin_flush_x, self._spin_flush_y,
            self._spin_wick_x, self._spin_wick_y,
        ):
            _spin.valueChanged.connect(self._on_calibration_value_changed)

        right_splitter.addWidget(cal_grp)

        # Pico Port Assignment
        pico_grp = QGroupBox("Pico Port Assignment")
        pico_grid = QGridLayout(pico_grp)

        pico_grid.addWidget(QLabel("<b>Logical name</b>"), 0, 0)
        pico_grid.addWidget(QLabel("<b>Resolved port</b>"), 0, 1)
        pico_grid.addWidget(QLabel("<b>Assign to port</b>"), 0, 2)

        pico_grid.addWidget(QLabel("pico1  (ch 1–16 by default):"), 1, 0)
        self._lbl_pico1_port = QLabel("(not connected)")
        pico_grid.addWidget(self._lbl_pico1_port, 1, 1)
        self._combo_pico1 = QComboBox()
        self._combo_pico1.setMinimumWidth(120)
        pico_grid.addWidget(self._combo_pico1, 1, 2)

        pico_grid.addWidget(QLabel("pico2  (ch 17–32 by default):"), 2, 0)
        self._lbl_pico2_port = QLabel("(not connected)")
        pico_grid.addWidget(self._lbl_pico2_port, 2, 1)
        self._combo_pico2 = QComboBox()
        self._combo_pico2.setMinimumWidth(120)
        pico_grid.addWidget(self._combo_pico2, 2, 2)

        pico_btn_row = QHBoxLayout()
        self._btn_scan_pico = QPushButton("Scan for Pico Devices")
        self._btn_scan_pico.clicked.connect(self._on_scan_pico)
        pico_btn_row.addWidget(self._btn_scan_pico)

        self._btn_apply_pico = QPushButton("Apply & Save to Config")
        self._btn_apply_pico.clicked.connect(self._on_apply_pico)
        pico_btn_row.addWidget(self._btn_apply_pico)
        pico_btn_row.addStretch()
        pico_grid.addLayout(pico_btn_row, 3, 0, 1, 3)

        self._lbl_pico_status = QLabel("Click 'Scan' to detect available EmStat Pico devices.")
        self._lbl_pico_status.setWordWrap(True)
        pico_grid.addWidget(self._lbl_pico_status, 4, 0, 1, 3)

        right_splitter.addWidget(pico_grp)

        # Equal split between Stage Cal and Pico
        right_splitter.setSizes([300, 200])

        # Equal split between Instrument Status and right splitter
        top_splitter.setSizes([500, 500])

        # ── BOTTOM SECTION: Position Map (wide) | PCB Configuration (narrow) ─
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(bottom_splitter)

        # Position Map
        pos_grp = QGroupBox("Position Map")
        pos_layout = QVBoxLayout(pos_grp)
        from softae.config.loader import default_pcb_name
        self._default_pcb = default_pcb_name()
        _initial_name = (
            self._default_pcb
            if self._default_pcb in self._pcb_configs
            else (next(iter(self._pcb_configs), None))
        )
        initial_pcb = self._pcb_configs.get(_initial_name, {}) if _initial_name else {}
        self._pos_map = PositionMapWidget(
            self._manager,
            pcb_config=initial_pcb,
            home_pos=(self._spin_home_x.value(), self._spin_home_y.value()),
            dep1_pos=(self._spin_dep1_x.value(), self._spin_dep1_y.value()),
            parent=self,
            data_store=self._data_store,
        )
        pos_layout.addWidget(self._pos_map)
        bottom_splitter.addWidget(pos_grp)

        # PCB Configuration
        pcb_grp = QGroupBox("PCB Configuration")
        pcb_layout = QVBoxLayout(pcb_grp)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("PCB Layout:"))
        self._combo_pcb = QComboBox()
        pcb_names = list(self._pcb_configs.keys()) or ["(none loaded)"]
        self._combo_pcb.addItems(pcb_names)
        if self._default_pcb and self._default_pcb in pcb_names:
            self._combo_pcb.setCurrentText(self._default_pcb)
        self._combo_pcb.currentTextChanged.connect(self._on_pcb_selected)
        sel_row.addWidget(self._combo_pcb)
        sel_row.addStretch()
        pcb_layout.addLayout(sel_row)

        self._lbl_pcb_info = QLabel("")
        pcb_layout.addWidget(self._lbl_pcb_info)
        self._on_pcb_selected(self._combo_pcb.currentText())

        config_panel = QWidget()
        config_panel_layout = QVBoxLayout(config_panel)
        config_panel_layout.setContentsMargins(0, 0, 0, 0)
        config_panel_layout.setSpacing(8)

        # Syringe Configuration
        syr_grp = QGroupBox("Syringe Config")
        syr_layout = QVBoxLayout(syr_grp)
        self._spin_syr_parallel_by_pump: dict[int, QSpinBox] = {}
        try:
            from softae.config.loader import syringe_parallel_counts
            counts = syringe_parallel_counts()
        except Exception:
            counts = {0: 1, 1: 1, 2: 1}

        for pump_id in range(3):
            syr_row = QHBoxLayout()
            syr_row.addWidget(QLabel(f"Pump {pump_id}:"))
            syr_row.addWidget(QLabel("Loaded syringes:"))
            spin = QSpinBox()
            spin.setRange(1, 2)
            spin.setValue(int(counts.get(pump_id, 1)))
            self._spin_syr_parallel_by_pump[pump_id] = spin
            syr_row.addWidget(spin)
            syr_row.addStretch()
            syr_layout.addLayout(syr_row)

        self._lbl_syr_info = QLabel(
            "Each pump can be configured with its own loaded syringe count."
        )
        self._lbl_syr_info.setWordWrap(True)
        syr_layout.addWidget(self._lbl_syr_info)

        self._btn_apply_syr = QPushButton("Apply + Save")
        self._btn_apply_syr.clicked.connect(self._on_apply_syringe_parallel)
        syr_layout.addWidget(self._btn_apply_syr)

        self._lbl_syr_status = QLabel("")
        self._lbl_syr_status.setWordWrap(True)
        syr_layout.addWidget(self._lbl_syr_status)

        config_panel_layout.addWidget(syr_grp)
        config_panel_layout.addStretch()

        bottom_splitter.addWidget(config_panel)
        bottom_splitter.addWidget(pcb_grp)

        # Position Map wide, compact config editors on the right.
        bottom_splitter.setStretchFactor(0, 4)
        bottom_splitter.setStretchFactor(1, 2)
        bottom_splitter.setStretchFactor(2, 1)

        # Main splitter: top section ~55%, bottom section ~45%
        main_splitter.setSizes([550, 450])

    # --- Polling ---------------------------------------------------------------

    def _start_polling(self) -> None:
        self._poll_worker = _TablePollWorker(self._manager, parent=self)
        self._poll_worker.poll_ready.connect(self._refresh_table)
        self._poll_worker.start()

    def _refresh_table(
        self,
        statuses: list | None = None,
        routing_labels: dict | None = None,
        pico_ports: dict | None = None,
        rig_lock=None,
    ) -> None:
        if statuses is None or routing_labels is None or pico_ports is None:
            statuses = []
            routing_labels = {}
            pico_ports = {"pico1": "(not connected)", "pico2": "(not connected)"}
            rig_lock = _read_rig_lock()
            try:
                statuses = self._manager.list_instruments()
                _annotate_busy(self._manager, statuses)
            except Exception:
                pass
            try:
                from softae.config.loader import channel_routing
                r = channel_routing()
                p1 = r.get("pico1_range", [1, 16])
                p2 = r.get("pico2_range", [17, 32])
                routing_labels = {
                    "pico1": f"EmStat Pico  [ch {p1[0]}–{p1[1]}]",
                    "pico2": f"EmStat Pico  [ch {p2[0]}–{p2[1]}]",
                }
            except Exception:
                pass
            try:
                for pico_name in ("pico1", "pico2"):
                    pico = self._manager.get(pico_name)
                    port = pico.status().get("port") or "(auto)"
                    pico_ports[pico_name] = str(port)
            except Exception:
                pass

        self._rig_lock = rig_lock
        held_elsewhere = rig_lock is not None and not rig_lock.is_mine()
        owner = _owner_line(rig_lock) if held_elsewhere else ""

        self._table.setRowCount(len(statuses))

        for row, s in enumerate(statuses):
            name = s.get("name", "")
            self._table.setItem(row, 0, QTableWidgetItem(name))
            type_label = routing_labels.get(name, name.split("_")[0])
            self._table.setItem(row, 1, QTableWidgetItem(type_label))
            state = s.get("state", "UNKNOWN")
            busy = bool(s.get("busy"))
            item = QTableWidgetItem(
                _compose_state(state, busy=busy, rig_lock=rig_lock))
            if held_elsewhere:
                item.setForeground(Qt.GlobalColor.darkMagenta)
                item.setToolTip(rig_lock.describe())
            elif state == "CONNECTED":
                item.setForeground(
                    Qt.GlobalColor.darkYellow if busy else Qt.GlobalColor.darkGreen)
            elif state == "ERROR":
                item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(row, 2, item)
            details = {
                k: v for k, v in s.items()
                if k not in ("name", "state", "connected", "error", "busy")
            }
            # The owner replaces the details while another process holds the rig: those
            # readings came from a manager that is not the one driving the hardware.
            detail_text = owner if held_elsewhere else (str(details) if details else "")
            self._table.setItem(row, 3, QTableWidgetItem(detail_text))

        # Update pico port labels from worker-collected data
        for pico_name, lbl in (("pico1", self._lbl_pico1_port), ("pico2", self._lbl_pico2_port)):
            lbl.setText(pico_ports.get(pico_name, "(not connected)"))

    # --- Connection slots ------------------------------------------------------

    def _schedule_async(self, coro) -> None:
        """Schedule an async coroutine non-blockingly via qasync."""
        task = asyncio.ensure_future(coro)
        task.add_done_callback(self._on_async_done)

    def _on_async_done(self, task: asyncio.Task) -> None:
        """Handle completion of an async operation."""
        try:
            task.result()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Async operation failed: %s", exc)
        # Poke the background poll worker for a near-immediate refresh.
        if hasattr(self, "_poll_worker"):
            self._poll_worker.poke()

    def _refuse_if_rig_held(self, action: str) -> bool:
        """Whether *action* must not proceed because another process owns the rig.

        Read fresh rather than from the last poll: the table is up to 2 s stale, and a
        headless run that started in that window is exactly the case worth catching.

        Connecting is guarded and disconnecting is not, deliberately. Opening a port a
        running commissioning sweep is driving is the collision this exists to prevent;
        letting go of one never is.
        """
        lock = _read_rig_lock()
        if lock is None or lock.is_mine():
            return False
        QMessageBox.warning(
            self, "Rig in use",
            f"{action} was not attempted — {lock.describe()}\n\n"
            "Connecting now would give the hardware two owners. Wait for the run to "
            "finish, or take the rig from it deliberately if you are certain that "
            "process is gone.",
        )
        return True

    def _on_connect_all(self) -> None:
        if self._refuse_if_rig_held("Connect All"):
            return
        self._schedule_async(self._manager.connect_all())

    def _on_disconnect_all(self) -> None:
        self._schedule_async(self._manager.disconnect_all())

    def _project_dir(self) -> str:
        """Where a launched sequence should write. The store's directory wins.

        Falling back to the configured default rather than the current working
        directory: a child launched from a shortcut inherits whatever cwd the shell
        had, and artifacts landing outside the project are how a calibration goes
        missing.
        """
        store_dir = getattr(self._data_store, "project_dir", None)
        if store_dir:
            return str(store_dir)
        try:
            from softae.config.loader import data_project_dir
            return data_project_dir()
        except Exception:
            return "."

    def _on_bench_sequences(self) -> None:
        from softae.gui.widgets.calibration_launcher import CalibrationLauncherDialog

        dlg = CalibrationLauncherDialog(self._manager, self._project_dir(), parent=self)
        dlg.exec()
        # The child owns the rig now, so this tab's view of it is stale by definition.
        if hasattr(self, "_poll_worker"):
            self._poll_worker.poke()

    def _on_connect_selected(self) -> None:
        name = self._selected_instrument()
        if name and not self._refuse_if_rig_held(f"Connect '{name}'"):
            self._schedule_async(self._manager.connect(name))

    def _on_disconnect_selected(self) -> None:
        name = self._selected_instrument()
        if name:
            inst = self._manager.get(name)
            self._schedule_async(inst.disconnect())

    def _selected_instrument(self) -> str | None:
        row = self._table.currentRow()
        if row < 0:
            QMessageBox.information(self, "No selection", "Select an instrument row first.")
            return None
        item = self._table.item(row, 0)
        return item.text() if item else None

    # --- Stage calibration slots -----------------------------------------------

    def _on_set_home(self) -> None:
        self._btn_set_home.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                pos = await asyncio.to_thread(stage.live_position)
                self._spin_home_x.setValue(float(pos[0]))
                self._spin_home_y.setValue(float(pos[1]))
                self._lbl_stage_pos.setText(f"Home set: ({float(pos[0]):.3f}, {float(pos[1]):.3f})")
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_set_home.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_set_dep1(self) -> None:
        self._btn_set_dep1.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                pos = await asyncio.to_thread(stage.live_position)
                self._spin_dep1_x.setValue(float(pos[0]))
                self._spin_dep1_y.setValue(float(pos[1]))
                self._lbl_stage_pos.setText(f"Dep-1 set: ({float(pos[0]):.3f}, {float(pos[1]):.3f})")
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_set_dep1.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_set_flush(self) -> None:
        self._btn_set_flush.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                pos = await asyncio.to_thread(stage.live_position)
                self._spin_flush_x.setValue(float(pos[0]))
                self._spin_flush_y.setValue(float(pos[1]))
                self._lbl_stage_pos.setText(
                    f"Flush set: ({float(pos[0]):.3f}, {float(pos[1]):.3f})"
                )
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_set_flush.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_set_wick(self) -> None:
        self._btn_set_wick.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                pos = await asyncio.to_thread(stage.live_position)
                self._spin_wick_x.setValue(float(pos[0]))
                self._spin_wick_y.setValue(float(pos[1]))
                self._lbl_stage_pos.setText(
                    f"Wick set: ({float(pos[0]):.3f}, {float(pos[1]):.3f})"
                )
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_set_wick.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_go_home(self) -> None:
        x, y = self._spin_home_x.value(), self._spin_home_y.value()
        self._btn_go_home.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                await asyncio.to_thread(stage.move_to, x, y)
                pos = await asyncio.to_thread(stage.live_position)
                self._lbl_stage_pos.setText(f"At home: ({float(pos[0]):.3f}, {float(pos[1]):.3f})")
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_go_home.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_go_dep1(self) -> None:
        x, y = self._spin_dep1_x.value(), self._spin_dep1_y.value()
        self._btn_go_dep1.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                await asyncio.to_thread(stage.move_to, x, y)
                pos = await asyncio.to_thread(stage.live_position)
                self._lbl_stage_pos.setText(f"At dep-1: ({float(pos[0]):.3f}, {float(pos[1]):.3f})")
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_go_dep1.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_go_flush(self) -> None:
        x, y = self._spin_flush_x.value(), self._spin_flush_y.value()
        self._btn_go_flush.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                await asyncio.to_thread(stage.move_to, x, y)
                pos = await asyncio.to_thread(stage.live_position)
                self._lbl_stage_pos.setText(
                    f"At flush: ({float(pos[0]):.3f}, {float(pos[1]):.3f})"
                )
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_go_flush.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_go_wick(self) -> None:
        x, y = self._spin_wick_x.value(), self._spin_wick_y.value()
        self._btn_go_wick.setEnabled(False)

        async def _work():
            try:
                stage = self._manager.get("stage")
                await asyncio.to_thread(stage.move_to, x, y)
                pos = await asyncio.to_thread(stage.live_position)
                self._lbl_stage_pos.setText(
                    f"At wick: ({float(pos[0]):.3f}, {float(pos[1]):.3f})"
                )
            except Exception as exc:
                self._lbl_stage_pos.setText(f"Error: {exc}")
            finally:
                self._btn_go_wick.setEnabled(True)

        asyncio.ensure_future(_work())

    def _persist_calibration(self) -> None:
        """Write all four calibration positions to softae_config.toml."""
        from softae.config.loader import save_stage_calibration
        save_stage_calibration(
            self._spin_home_x.value(), self._spin_home_y.value(),
            self._spin_dep1_x.value(), self._spin_dep1_y.value(),
            self._spin_flush_x.value(), self._spin_flush_y.value(),
            self._spin_wick_x.value(), self._spin_wick_y.value(),
        )

    def _on_calibration_value_changed(self, _value: float = 0.0) -> None:
        """Auto-persist position edits to config when auto-save is enabled."""
        if not self._chk_autosave_cal.isChecked():
            return
        try:
            self._persist_calibration()
            self._lbl_stage_pos.setText("Positions auto-saved to config.")
        except Exception as exc:
            # Non-fatal: surface once, don't spam a modal on every keystroke.
            self._lbl_stage_pos.setText(f"Auto-save error: {exc}")

    def _on_save_calibration(self) -> None:
        """Save Home, Dep-1, Flush, and Wick spinbox values to softae_config.toml."""
        try:
            self._persist_calibration()
            self._lbl_stage_pos.setText("Calibration saved to config.")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Could not save calibration:\n{exc}")

    # --- Pico port assignment --------------------------------------------------

    def _refresh_pico_ports(self) -> None:
        """Update resolved-port labels from live instrument status.

        .. deprecated::
            Port data is now collected in :class:`_TablePollWorker` and
            delivered to :meth:`_refresh_table` via the ``poll_ready`` signal.
            This method is kept for call sites inside async coroutines that
            run before the worker has had a chance to emit; call
            ``self._poll_worker.poke()`` instead wherever possible.
        """
        if hasattr(self, "_poll_worker"):
            self._poll_worker.poke()

    def _on_scan_pico(self) -> None:
        """Detect available EmStat Pico COM ports and populate the dropdowns."""
        self._btn_scan_pico.setEnabled(False)
        self._lbl_pico_status.setText("Scanning...")

        def _scan():
            from softae.drivers.async_espico import AsyncESPico
            return AsyncESPico.list_available_ports()

        async def _work():
            try:
                ports = await asyncio.to_thread(_scan)
                for combo in (self._combo_pico1, self._combo_pico2):
                    prev = combo.currentText()
                    combo.clear()
                    if ports:
                        combo.addItems(ports)
                        idx = combo.findText(prev)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                    else:
                        combo.addItem("(none found)")
                if ports:
                    self._lbl_pico_status.setText(
                        f"Found {len(ports)} device(s): {', '.join(ports)}. "
                        "Select ports for pico1 and pico2, then click 'Apply & Save'."
                    )
                else:
                    self._lbl_pico_status.setText(
                        "No EmStat Pico devices detected. Check USB connections and PalmSens SDK."
                    )
            except Exception as exc:
                self._lbl_pico_status.setText(f"Scan error: {exc}")
            finally:
                self._btn_scan_pico.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_apply_pico(self) -> None:
        """Save the selected port assignments to config and reconnect both picos."""
        p1 = self._combo_pico1.currentText()
        p2 = self._combo_pico2.currentText()

        if not p1 or not p2 or p1.startswith("(") or p2.startswith("("):
            self._lbl_pico_status.setText("Scan for devices first, then select ports for both picos.")
            return
        if p1 == p2:
            self._lbl_pico_status.setText(
                "Error: pico1 and pico2 must be assigned to different physical ports."
            )
            return

        self._btn_apply_pico.setEnabled(False)
        self._lbl_pico_status.setText(f"Saving config and reconnecting…")

        async def _work():
            try:
                from softae.config import loader as cfg_loader
                cfg_loader.save_pico_ports(p1, p2)

                for name, port in (("pico1", p1), ("pico2", p2)):
                    try:
                        pico = self._manager.get(name)
                        await pico.disconnect()
                        pico.reassign_port(port)
                        await pico.connect()
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning(
                            "pico_reconnect_failed: %s → %s", name, exc
                        )

                self._lbl_pico_status.setText(
                    f"Saved.  pico1 → {p1}   pico2 → {p2}   (persisted to softae_config.toml)"
                )
                self._refresh_pico_ports()
                self._refresh_table()
            except Exception as exc:
                self._lbl_pico_status.setText(f"Error: {exc}")
            finally:
                self._btn_apply_pico.setEnabled(True)

        asyncio.ensure_future(_work())

    def _on_apply_syringe_parallel(self) -> None:
        """Persist and apply per-pump parallel-syringe counts."""
        counts = {pump_id: int(spin.value()) for pump_id, spin in self._spin_syr_parallel_by_pump.items()}
        self._btn_apply_syr.setEnabled(False)

        try:
            from softae.config.loader import save_syringe_parallel_counts

            save_syringe_parallel_counts(counts)

            live_applied = False
            live_error: str | None = None
            try:
                syr = self._manager.get("syringe")
                setter = getattr(syr, "set_parallel_syringes", None)
                if callable(setter):
                    for pump_id, count in counts.items():
                        try:
                            setter(count, pump_id=pump_id)
                        except TypeError:
                            setter(count)
                    live_applied = True
            except Exception as exc:
                live_error = str(exc)

            if live_applied:
                self._lbl_syr_status.setText(
                    f"Saved per-pump syringe counts {counts} and applied to active syringe driver."
                )
            elif live_error:
                self._lbl_syr_status.setText(
                    f"Saved per-pump syringe counts {counts}. Live apply failed: {live_error}"
                )
            else:
                self._lbl_syr_status.setText(
                    f"Saved per-pump syringe counts {counts}. Reconnect syringe if live value does not update."
                )

            if hasattr(self, "_poll_worker"):
                self._poll_worker.poke()
        except Exception as exc:
            QMessageBox.warning(self, "Syringe Config Error", str(exc))
        finally:
            self._btn_apply_syr.setEnabled(True)

    def refresh_occupancy(self, board_id: int | None = None) -> None:
        """Re-read board occupancy into the electrode map (no-op if not built)."""
        pos_map = getattr(self, "_pos_map", None)
        if pos_map is not None:
            pos_map.refresh_occupancy(board_id)

    # --- Cleanup ----------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop this tab's worker threads (idempotent)."""
        worker = getattr(self, "_poll_worker", None)
        if worker is not None:
            worker.stop_worker()           # no-op if already stopped
        pos_map = getattr(self, "_pos_map", None)   # embedded PositionMapWidget
        if pos_map is not None and hasattr(pos_map, "cleanup"):
            pos_map.cleanup()

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)

    # --- PCB selector ----------------------------------------------------------

    def _on_pcb_selected(self, name: str) -> None:
        info = self._pcb_configs.get(name, {})
        if not info:
            self._lbl_pcb_info.setText("No PCB configuration data available.")
            return
        lines = [
            f"Channels: {info.get('channels', '?')}",
            f"Grid: {info.get('grid', '?')}",
            f"Spacing (mm): {info.get('spacing_mm', '?')}",
            f"Electrode L (cm): {info.get('electrode_L_cm', '?')}",
            f"Electrode w (cm): {info.get('electrode_w_cm', '?')}",
        ]
        self._lbl_pcb_info.setText("\n".join(lines))
        n_channels = info.get("channels", 0) or (
            info.get("grid", [0, 0])[0] * info.get("grid", [0, 0])[1]
        )
        if n_channels > 0:
            self.pcb_channel_count_changed.emit(int(n_channels))
        self._pos_map.set_pcb_config(
            info,
            home_pos=(self._spin_home_x.value(), self._spin_home_y.value()),
            dep1_pos=(self._spin_dep1_x.value(), self._spin_dep1_y.value()),
        )

    def _on_dep1_changed(self) -> None:
        """Redraw the position map whenever Dep-1 coordinates are edited."""
        if not hasattr(self, "_pos_map"):
            return
        self._pos_map.set_pcb_config(
            self._pcb_configs.get(self._combo_pcb.currentText(), {}),
            dep1_pos=(self._spin_dep1_x.value(), self._spin_dep1_y.value()),
        )
