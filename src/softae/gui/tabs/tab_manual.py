"""Tab 2: Manual Control.

Direct jog/go-to for the stage, per-pump syringe controls, dispenser
head toggle, temperature setpoint, RH setpoint, camera snap, and EIS
quick-run.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import TYPE_CHECKING

from .tab_manual_workers import _CommandWorker

from PySide6.QtCore import (
    QMutex,
    QObject,
    QThread,
    Qt,
    QTimer,
    QWaitCondition,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from softae.config.loader import (
    liquid_handling_config,
    piezo_config,
    syringe_parallel_count,
)
from softae.gui.widgets.worker_thread import StoppableWorker

if TYPE_CHECKING:
    from softae.analysis.circuit_fitting import FitResult
    from softae.analysis.eis_data import EISResult
    from softae.core.data_store import DataStore
    from softae.core.eis_scout_scripts import ScoutPlanner
    from softae.core.run_lock import RunLock
    from softae.gui.widgets.camera_worker import CameraWorker
    from softae.server.manager import InstrumentManager


def _manual_scout_default() -> bool:
    """Where the tab's scout checkbox starts — ``[eis.scout] actuate_manual``.

    A default, not the control: the box is authoritative once the window is up.
    Falls back to *off* if the config cannot be read, because the safe answer for
    an uncharacterised sample is the sweep the operator chose, run once.
    """
    try:
        from softae.analysis.eis.scout import scout_settings

        return bool(scout_settings().actuate_manual)
    except Exception:
        return False


class _ManualEisWorker(QObject):
    """Background worker for manual EIS acquisition + optional fit/save.

    Measures one *or more* channels in series.  All channels of a run are grouped
    under a single ``manual_eis`` run in the DataStore, and ``finished`` emits a
    **list** of per-channel payloads (one per measured channel, in order).

    This is the one call site that is genuinely **per measurement**: the ``.mscr``
    is written inside :meth:`_measure_one`, immediately before the sweep it
    describes. So it is the site where a channel's own spectrum can decide, then
    and there, that one more sweep is worth taking — see
    :class:`~softae.core.eis_scout_scripts.ScoutPlanner`. In the common case it
    is not, and nothing extra is measured.
    """

    progress = Signal(str)
    finished = Signal(object)   # list[dict] — one payload per channel
    error = Signal(str)

    def __init__(
        self,
        manager: InstrumentManager,
        data_store: DataStore | None,
        *,
        channels: list[int],
        eis_params: dict,
        preset_label: str = "custom",
        auto_fit: bool,
        fit_model: str,
        auto_save: bool,
        scout: "ScoutPlanner | None" = None,
    ):
        super().__init__()
        self._manager = manager
        self._data_store = data_store
        self._channels = list(channels)
        self._eis_params = eis_params
        self._preset_label = preset_label
        self._auto_fit = auto_fit
        self._fit_model = fit_model
        self._auto_save = auto_save
        # Owned by the tab, not by the worker: a plan is only worth anything to
        # the *next* run on that channel, and a worker lives for one run.
        self._scout = scout

    @Slot()
    def run(self) -> None:
        from softae.config.loader import pico_for_channel

        try:
            run_id: str | None = None
            if self._auto_save and self._data_store is not None:
                run_id = self._data_store.start_run(
                    workflow_name="manual_eis", campaign="manual", quality="explore",
                )
            payloads: list[dict] = []
            n = len(self._channels)
            for i, ch in enumerate(self._channels, start=1):
                suffix = f" ({i}/{n})" if n > 1 else ""
                self.progress.emit(
                    f"Measuring ch {ch}{suffix} ({self._preset_label})...")
                payloads.append(
                    self._measure_one(ch, pico_for_channel(ch), run_id))
            if run_id is not None and self._data_store is not None:
                self._data_store.finish_run(run_id)
            self.finished.emit(payloads)
        except Exception as exc:
            self.error.emit(str(exc))

    def _measure_one(self, channel: int, pico_name: str, run_id: str | None) -> dict:
        """Acquire, (optionally) fit, and record one channel; return its payload."""
        import time

        from softae.analysis.eis_data import EISResult
        from softae.drivers.mscr_library import eis_run_mscrbuild

        pico = self._manager.get(pico_name)
        p = self._eis_params

        script_path = os.path.join(tempfile.gettempdir(), "softae_testing.mscr")

        def _acquire(params: dict) -> "EISResult":
            """Send whatever is at *script_path* and parse what comes back.

            *params* is what the row records, so it must describe the script that
            is about to run — never the one the operator selected if a different
            one was written over it.
            """
            t_start = time.monotonic()
            raw = pico.sendscript_getdata(script_path, pico._output_dir, channel)
            return EISResult.from_raw(
                raw, channel=channel,
                measurement_time_s=time.monotonic() - t_start, eis_params=params,
            )

        eis_run_mscrbuild(
            script_path,
            mux_ch=channel,
            mVac=p.get("mv_ac", 10),
            f_hi=p.get("f_hi", 200_000),
            f_lo=p.get("f_lo_mHz", 100),
            npts=p.get("npts", 20),
            mVdc=p.get("mv_dc", 0),
        )
        # A copy only when the scout will stamp its verdict on the row: `p` is one
        # dict shared by every channel of this run, so stamping the shared one
        # would leave channel 3's saved result claiming channel 7's verdict. With
        # the scout off nothing writes to it and the caller's own dict is passed
        # through, exactly as before.
        observing = self._scout is not None and self._scout.observing
        eis_result = _acquire(dict(p) if observing else p)

        if self._scout is not None:
            # The sweep just taken IS the scout sweep, and in the common case it
            # is also the measurement: `build_follow_up` returns None — writing
            # nothing — unless it can name a strictly wider or denser sweep. So
            # an adequate spectrum is never re-measured, and a follow-up can
            # never be narrower than what the operator asked for.
            decision = self._scout.observe(channel, eis_result)
            follow_up = self._scout.build_follow_up(script_path, channel, p, decision)
            if follow_up is not None:
                scout_sweep_s = eis_result.measurement_time_s
                eis_result = _acquire(follow_up)
                # The superseded sweep is not stored, so its cost would otherwise
                # vanish from the record; the acquisition really did take both.
                eis_result.eis_params["eis_scout_sweep_s"] = float(scout_sweep_s)

        fit_result: FitResult | None = None
        fit_error: str | None = None
        if self._auto_fit:
            from softae.analysis.eis.engine import analyze_spectrum

            try:
                # ``engine`` unset — ``[eis] engine`` governs the manual tab too.
                # No ``cell``: the σ-mode radio and its geometry spins live on the
                # GUI thread, and ``_conductivity_from_fit`` reads them there. What
                # this buys is that the R it divides is standard-suite-derived.
                fit_result = analyze_spectrum(
                    eis_result, model_name=self._fit_model).fit
            except Exception as exc:  # keep measurement usable even if fit fails
                fit_error = str(exc)

        # Pre-compute residual channels before optional autosave so they can
        # be persisted as extra columns in the saved EIS text file.
        if fit_result is not None and fit_result.success:
            try:
                from softae.analysis.circuit_fitting import (
                    compute_fit_residuals,
                    predict_fit_curve,
                )

                z_fit_complex = predict_fit_curve(fit_result, eis_result.frequency)
                if z_fit_complex is not None:
                    resid_real, resid_imag = compute_fit_residuals(
                        eis_result, z_fit_complex
                    )
                    eis_result.residual_real_pct = resid_real
                    eis_result.residual_imag_pct = resid_imag
            except Exception:
                # Keep autosave robust even if residual reconstruction fails.
                pass

        measurement_id: int | None = None
        if run_id is not None and self._data_store is not None:
            eis_dir = self._data_store.eis_dir(run_id)
            save_path = eis_dir / f"ch{channel:02d}_manual.txt"
            eis_result.raw_file_path = str(save_path)
            eis_result.save(save_path)
            measurement_id = self._data_store.record_measurement(run_id, eis_result)
            if fit_result is not None:
                self._data_store.record_fit(measurement_id, fit_result)
            from softae.core.conditions_capture import read_environment

            env = read_environment(self._manager)
            if any(v is not None for v in env.values()):
                self._data_store.record_conditions(
                    measurement_id, "measurement", **env
                )

        return {
            "pico_name": pico_name,
            "channel": channel,
            "auto_fit": self._auto_fit,
            "fit_model": self._fit_model,
            "fit_error": fit_error,
            "fit_result": fit_result,
            "run_id": run_id,
            "measurement_id": measurement_id,
            "db_missing": self._auto_save and self._data_store is None,
            "eis_result": eis_result,
        }


class _ManualPollingWorker(StoppableWorker):
    """Background thread for PV polling in the Manual Control tab.

    Keeps driver calls off the main thread to prevent GUI hangs when
    instruments are slow or retrying after a communication error.
    """

    poll_done = Signal(dict)

    _default_stop_timeout_ms = 5000

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._stop = False
        self._mutex = QMutex()
        self._condition = QWaitCondition()

    def run(self) -> None:
        self._stop = False
        while not self._stop:
            self.poll_done.emit(self.poll_once())
            # A *wakeable* wait rather than ``msleep``: an msleep cannot be
            # interrupted, so a stop arriving just after a poll blocked the
            # **caller** — the main thread, inside ``cleanup()`` — for the rest of
            # the interval. Closing the tab or the window froze the GUI for up to
            # 2 s for no reason. The flag is re-checked under the mutex so a stop
            # landing between the poll and the wait cannot lose its wake-up.
            self._mutex.lock()
            if not self._stop:
                self._condition.wait(self._mutex, 2000)
            self._mutex.unlock()

    def poll_once(self) -> dict:
        """One sweep of the instruments, as the payload ``poll_done`` carries.

        Separate from :meth:`run` so the reading can be asserted without starting
        a thread or standing in for a sleep: what a poll *reports* and how often
        it repeats are two different questions, and a test of the first should not
        have to answer the second.
        """
        import math

        out: dict = {}
        try:
            tc = self._manager.get("temp_controller")
            out["temp_sp"] = tc.get_sp()
            out["temp_pv"] = tc.get_pv()
        except Exception:
            pass
        try:
            rh = self._manager.get("rh_controller")
            h = rh.get_H()
            if not math.isnan(h):
                out["rh"] = h
        except Exception:
            pass
        try:
            stage = self._manager.get("stage")
            out["pos"] = stage.live_position()
        except Exception:
            pass
        try:
            syr = self._manager.get("syringe")
            status = syr.status()
            out["parallel_syringes"] = int(status.get("parallel_syringes", 1))
            counts_by_pump = status.get("parallel_syringes_by_pump")
            if isinstance(counts_by_pump, dict):
                out["parallel_syringes_by_pump"] = dict(counts_by_pump)
        except Exception:
            pass
        return out

    def _request_stop(self) -> None:
        self._mutex.lock()
        self._stop = True
        self._condition.wakeAll()
        self._mutex.unlock()


class ManualControlTab(QWidget):
    """Hands-on instrument control panel."""

    def __init__(
        self,
        manager: InstrumentManager,
        parent: QWidget | None = None,
        *,
        data_store: DataStore | None = None,
    ):
        super().__init__(parent)
        self._manager = manager
        self._data_store = data_store
        self._cam_worker: CameraWorker | None = None
        self._eis_thread: QThread | None = None
        self._eis_worker: _ManualEisWorker | None = None
        #: Rebuilt on every Run press — see :meth:`_on_eis_run`. Inert while
        #: ``[eis.scout] actuate`` is off, which is how it ships.
        self._eis_scout: ScoutPlanner | None = None
        self._eis_series_window: QWidget | None = None  # multi-channel plot window
        self._pv_worker: _ManualPollingWorker | None = None
        self._manual_dispense_count_by_pump: dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._build_ui()
        self._start_pv_polling()
        # A campaign can start (or end) in another process while this tab sits
        # open, so ownership is polled rather than read once. 2 s matches the Init
        # tab's cadence, so the two views cannot be more than one tick apart.
        self._rig_owner_timer = QTimer(self)
        self._rig_owner_timer.setInterval(2000)
        self._rig_owner_timer.timeout.connect(self.refresh_rig_owner)
        self._rig_owner_timer.start()
        self.refresh_rig_owner()
        # Reflect the driver's registered head belief (may already have been set
        # by the launch-time verification prompt before this tab is shown).
        self.refresh_head_label()
        self.refresh_stock_labels()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Who owns the rig, stated before any control that could reach it.  Hidden
        # while the rig is free or ours, so the ordinary session is unchanged.
        self._lbl_rig_owner = QLabel("")
        self._lbl_rig_owner.setWordWrap(True)
        self._lbl_rig_owner.setVisible(False)
        layout.addWidget(self._lbl_rig_owner)

        top = QHBoxLayout()

        # --- Stage section ---
        stage_grp = QGroupBox("Stage")
        stage_layout = QVBoxLayout(stage_grp)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("X:"))
        self._spin_x = QDoubleSpinBox()
        self._spin_x.setRange(-100, 100)
        self._spin_x.setDecimals(2)
        pos_row.addWidget(self._spin_x)
        pos_row.addWidget(QLabel("Y:"))
        self._spin_y = QDoubleSpinBox()
        self._spin_y.setRange(-100, 100)
        self._spin_y.setDecimals(2)
        pos_row.addWidget(self._spin_y)
        self._btn_goto = QPushButton("Go To")
        self._btn_goto.clicked.connect(self._on_goto)
        pos_row.addWidget(self._btn_goto)
        stage_layout.addLayout(pos_row)

        jog_row = QHBoxLayout()
        jog_row.addWidget(QLabel("Step:"))
        self._spin_jog_step = QDoubleSpinBox()
        self._spin_jog_step.setRange(0.01, 50.0)
        self._spin_jog_step.setValue(1.0)
        self._spin_jog_step.setDecimals(2)
        self._spin_jog_step.setSuffix(" mm")
        jog_row.addWidget(self._spin_jog_step)
        self._jog_buttons: list[QPushButton] = []
        for label, dx, dy in [("Left", -1, 0), ("Right", 1, 0), ("Up", 0, 1), ("Down", 0, -1)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked=False, _dx=dx, _dy=dy: self._on_jog(_dx, _dy))
            self._jog_buttons.append(btn)
            jog_row.addWidget(btn)
        stage_layout.addLayout(jog_row)

        self._lbl_pos = QLabel("Position: (?, ?)")
        stage_layout.addWidget(self._lbl_pos)
        top.addWidget(stage_grp)

        # --- Temperature section ---
        temp_grp = QGroupBox("Temperature")
        temp_layout = QVBoxLayout(temp_grp)
        sp_row = QHBoxLayout()
        sp_row.addWidget(QLabel("Setpoint (°C):"))
        self._spin_temp = QDoubleSpinBox()
        self._spin_temp.setRange(5, 200)
        self._spin_temp.setValue(25.0)
        sp_row.addWidget(self._spin_temp)
        self._btn_set_temp = QPushButton("Set")
        self._btn_set_temp.clicked.connect(self._on_set_temp)
        sp_row.addWidget(self._btn_set_temp)
        temp_layout.addLayout(sp_row)

        self._lbl_temp_sp = QLabel("SP: -- °C")
        temp_layout.addWidget(self._lbl_temp_sp)
        self._lbl_temp_pv = QLabel("PV: -- °C")
        temp_layout.addWidget(self._lbl_temp_pv)

        ramp_row = QHBoxLayout()
        ramp_row.addWidget(QLabel("Ramp to (°C):"))
        self._spin_ramp_end = QDoubleSpinBox()
        self._spin_ramp_end.setRange(5, 200)
        self._spin_ramp_end.setValue(25.0)
        ramp_row.addWidget(self._spin_ramp_end)
        ramp_row.addWidget(QLabel("Rate (°C/min):"))
        self._spin_ramp_rate = QDoubleSpinBox()
        self._spin_ramp_rate.setRange(0.1, 10.0)
        self._spin_ramp_rate.setValue(1.0)
        ramp_row.addWidget(self._spin_ramp_rate)
        self._btn_ramp = QPushButton("Start Ramp")
        self._btn_ramp.clicked.connect(self._on_ramp)
        ramp_row.addWidget(self._btn_ramp)
        temp_layout.addLayout(ramp_row)

        top.addWidget(temp_grp)

        # --- RH section ---
        rh_grp = QGroupBox("Relative Humidity")
        rh_layout = QVBoxLayout(rh_grp)

        rh_sp_row = QHBoxLayout()
        rh_sp_row.addWidget(QLabel("Setpoint (%):"))
        self._spin_rh = QDoubleSpinBox()
        self._spin_rh.setRange(0, 95)
        self._spin_rh.setValue(50.0)
        rh_sp_row.addWidget(self._spin_rh)
        self._btn_set_rh = QPushButton("Set")
        self._btn_set_rh.clicked.connect(self._on_set_rh)
        rh_sp_row.addWidget(self._btn_set_rh)
        rh_layout.addLayout(rh_sp_row)

        self._lbl_rh = QLabel("RH: -- %")
        rh_layout.addWidget(self._lbl_rh)

        rh_ctrl_row = QHBoxLayout()
        self._btn_rh_start = QPushButton("Start PID")
        self._btn_rh_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self._btn_rh_start.clicked.connect(self._on_rh_start)
        rh_ctrl_row.addWidget(self._btn_rh_start)

        self._btn_rh_stop = QPushButton("Stop PID")
        self._btn_rh_stop.setStyleSheet("background-color: #f44336; color: white;")
        self._btn_rh_stop.clicked.connect(self._on_rh_stop)
        rh_ctrl_row.addWidget(self._btn_rh_stop)
        rh_layout.addLayout(rh_ctrl_row)

        top.addWidget(rh_grp)

        layout.addLayout(top)

        mid = QHBoxLayout()

        # --- Syringe section ---
        syr_grp = QGroupBox("Syringe Pumps")
        syr_layout = QVBoxLayout(syr_grp)

        # Per-pump controls (3 pumps) — one spinbox per row to keep visible when narrow
        self._pump_widgets: list[dict] = []
        for pump_id in range(3):
            rate_row = QHBoxLayout()
            rate_row.addWidget(QLabel(f"Pump {pump_id}  Rate:"))
            spin_rate = QDoubleSpinBox()
            spin_rate.setRange(0.01, 2120.0)
            spin_rate.setValue(5.0)
            spin_rate.setSuffix(" µL/min")
            rate_row.addWidget(spin_rate)
            rate_row.addStretch()
            syr_layout.addLayout(rate_row)

            vol_row = QHBoxLayout()
            vol_row.addWidget(QLabel(f"       Vol:"))
            spin_vol = QDoubleSpinBox()
            spin_vol.setRange(0.01, 5000.0)
            spin_vol.setValue(10.0)
            spin_vol.setSuffix(" µL")
            vol_row.addWidget(spin_vol)
            vol_row.addStretch()
            syr_layout.addLayout(vol_row)

            action_row = QHBoxLayout()
            btn_infuse = QPushButton("Infuse")
            btn_infuse.clicked.connect(
                lambda checked=False, pid=pump_id: self._on_infuse(pid)
            )
            action_row.addWidget(btn_infuse)
            lbl_count = QLabel(f"Syringes loaded: {syringe_parallel_count(pump_id)}")
            action_row.addWidget(lbl_count)
            lbl_stock = QLabel()
            action_row.addWidget(lbl_stock)
            action_row.addStretch()
            syr_layout.addLayout(action_row)

            self._pump_widgets.append({
                "rate": spin_rate, "vol": spin_vol, "btn": btn_infuse,
                "count_lbl": lbl_count, "stock_lbl": lbl_stock,
            })

        # Stock readout + declaration, sited here rather than only in the menu
        # because this is where the operator stands when they refill a syringe.
        stock_row = QHBoxLayout()
        self._btn_report_stock = QPushButton("Report Stock…")
        self._btn_report_stock.setToolTip(
            "Declare the volume loaded into each syringe. Dispensing is refused "
            "below the hard stop — running dry drives the plunger into its "
            "mechanical stop."
        )
        self._btn_report_stock.clicked.connect(self._on_report_stock)
        stock_row.addWidget(self._btn_report_stock)
        stock_row.addStretch()
        syr_layout.addLayout(stock_row)

        self._chk_apply_correction = QCheckBox("Apply liquid correction")
        self._chk_apply_correction.setChecked(bool(liquid_handling_config().get("enabled", False)))
        syr_layout.addWidget(self._chk_apply_correction)

        self._lbl_last_command = QLabel("Last command: none")
        self._lbl_last_command.setWordWrap(True)
        syr_layout.addWidget(self._lbl_last_command)

        # Head controls
        head_row = QHBoxLayout()
        self._btn_head_retract = QPushButton("Retract")
        self._btn_head_retract.clicked.connect(self._on_head_retract)
        head_row.addWidget(self._btn_head_retract)

        self._btn_head_descend = QPushButton("Descend")
        self._btn_head_descend.clicked.connect(self._on_head_descend)
        head_row.addWidget(self._btn_head_descend)

        self._lbl_head_status = QLabel("Head: Retracted")
        self._lbl_head_status.setStyleSheet("font-weight: bold; color: #4CAF50;")
        head_row.addWidget(self._lbl_head_status)
        syr_layout.addLayout(head_row)

        mid.addWidget(syr_grp)

        # --- Piezo section ---
        piezo_grp = QGroupBox("Piezo (Channel A)")
        piezo_layout = QVBoxLayout(piezo_grp)

        piezo_btn_row = QHBoxLayout()
        self._btn_piezo_a_on = QPushButton("Channel A ON")
        self._btn_piezo_a_on.clicked.connect(self._on_piezo_a_on)
        piezo_btn_row.addWidget(self._btn_piezo_a_on)

        self._btn_piezo_a_off = QPushButton("Channel A OFF")
        self._btn_piezo_a_off.clicked.connect(self._on_piezo_a_off)
        piezo_btn_row.addWidget(self._btn_piezo_a_off)
        piezo_layout.addLayout(piezo_btn_row)

        piezo_cfg = piezo_config()

        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("Freq (Hz):"))
        self._spin_piezo_freq = QSpinBox()
        self._spin_piezo_freq.setRange(10, 5000)
        self._spin_piezo_freq.setValue(int(piezo_cfg.get("frequency_hz", 500)))
        freq_row.addWidget(self._spin_piezo_freq)
        freq_row.addStretch()
        piezo_layout.addLayout(freq_row)

        on_row = QHBoxLayout()
        on_row.addWidget(QLabel("ON (s):"))
        self._spin_piezo_on_s = QDoubleSpinBox()
        self._spin_piezo_on_s.setRange(0.01, 120.0)
        self._spin_piezo_on_s.setDecimals(3)
        self._spin_piezo_on_s.setSingleStep(0.1)
        self._spin_piezo_on_s.setValue(float(piezo_cfg.get("sweep_on_s", 2.0)))
        on_row.addWidget(self._spin_piezo_on_s)
        on_row.addStretch()
        piezo_layout.addLayout(on_row)

        rest_row = QHBoxLayout()
        rest_row.addWidget(QLabel("REST (s):"))
        self._spin_piezo_rest_s = QDoubleSpinBox()
        self._spin_piezo_rest_s.setRange(0.01, 120.0)
        self._spin_piezo_rest_s.setDecimals(3)
        self._spin_piezo_rest_s.setSingleStep(0.1)
        self._spin_piezo_rest_s.setValue(float(piezo_cfg.get("sweep_rest_s", 3.0)))
        rest_row.addWidget(self._spin_piezo_rest_s)
        rest_row.addStretch()
        piezo_layout.addLayout(rest_row)

        self._btn_piezo_apply = QPushButton("Apply Settings")
        self._btn_piezo_apply.clicked.connect(self._on_piezo_apply_settings)
        piezo_layout.addWidget(self._btn_piezo_apply)

        self._lbl_piezo_status = QLabel("Piezo ready")
        self._lbl_piezo_status.setWordWrap(True)
        piezo_layout.addWidget(self._lbl_piezo_status)

        self._piezo_controls = [
            self._btn_piezo_a_on,
            self._btn_piezo_a_off,
            self._spin_piezo_freq,
            self._spin_piezo_on_s,
            self._spin_piezo_rest_s,
            self._btn_piezo_apply,
        ]
        self._piezo_profile_controls = [
            self._spin_piezo_freq,
            self._spin_piezo_on_s,
            self._spin_piezo_rest_s,
            self._btn_piezo_apply,
        ]
        self._piezo_channel_controls = [
            self._btn_piezo_a_on,
            self._btn_piezo_a_off,
        ]
        self._piezo_enabled_cfg = bool(piezo_cfg.get("enabled", False))
        self._piezo_caps_checked = False
        self._piezo_config_supported = True
        if not self._piezo_enabled_cfg:
            for widget in self._piezo_controls:
                widget.setEnabled(False)
            self._lbl_piezo_status.setText("Piezo disabled in config.")
        else:
            self._refresh_piezo_capability_status()

        mid.addWidget(piezo_grp)

        # --- Camera section ---
        cam_grp = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_grp)

        # Exposure control row
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure (s):"))
        self._spin_cam_exp = QDoubleSpinBox()
        self._spin_cam_exp.setRange(0.001, 10.0)
        self._spin_cam_exp.setValue(0.045)
        self._spin_cam_exp.setDecimals(3)
        self._spin_cam_exp.setSuffix(" s")
        exp_row.addWidget(self._spin_cam_exp)
        cam_layout.addLayout(exp_row)

        # Button row
        btn_row = QHBoxLayout()
        self._btn_snap = QPushButton("Snap")
        self._btn_snap.clicked.connect(self._on_snap)
        btn_row.addWidget(self._btn_snap)

        self._btn_live = QPushButton("Live Preview")
        self._btn_live.setCheckable(True)
        self._btn_live.toggled.connect(self._on_live_toggle)
        btn_row.addWidget(self._btn_live)

        self._btn_lamp_on = QPushButton("Lamp On")
        self._btn_lamp_on.clicked.connect(self._on_lamp_on)
        btn_row.addWidget(self._btn_lamp_on)

        self._btn_lamp_off = QPushButton("Lamp Off")
        self._btn_lamp_off.clicked.connect(self._on_lamp_off)
        btn_row.addWidget(self._btn_lamp_off)
        cam_layout.addLayout(btn_row)

        # Image display
        self._lbl_cam_image = QLabel("No image")
        self._lbl_cam_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cam_image.setMinimumSize(320, 240)
        self._lbl_cam_image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._lbl_cam_image.setStyleSheet("border: 1px solid #ccc; background: #1a1a1a; color: #666;")
        cam_layout.addWidget(self._lbl_cam_image)

        self._cam_timer = None

        mid.addWidget(cam_grp)

        layout.addLayout(mid)

        # --- EIS quick run ---
        eis_grp = QGroupBox("EIS Quick Run")
        eis_layout = QVBoxLayout(eis_grp)

        eis_top = QHBoxLayout()
        eis_top.addWidget(QLabel("Channel(s):"))
        self._edit_eis_ch = QLineEdit("1")
        self._edit_eis_ch.setMaximumWidth(130)
        self._edit_eis_ch.setPlaceholderText("e.g. 2,4,5-10")
        self._edit_eis_ch.setToolTip(
            "One or more channels, measured in series: comma list and hyphen "
            "ranges, e.g. 2,4,5-10")
        eis_top.addWidget(self._edit_eis_ch)

        eis_top.addWidget(QLabel("Preset:"))
        self._combo_eis_preset = QComboBox()
        self._combo_eis_preset.addItems(["Standard", "Quick", "Extended", "Longest"])
        eis_top.addWidget(self._combo_eis_preset)

        eis_top.addWidget(QLabel("\u2192 Pico:"))
        self._lbl_eis_pico = QLabel("pico1")
        self._lbl_eis_pico.setStyleSheet("font-weight: bold;")
        eis_top.addWidget(self._lbl_eis_pico)
        self._edit_eis_ch.textChanged.connect(self._update_eis_pico_label)

        self._btn_eis_run = QPushButton("Run EIS")
        self._btn_eis_run.setStyleSheet("background-color: #2196F3; color: white;")
        self._btn_eis_run.clicked.connect(self._on_eis_run)
        eis_top.addWidget(self._btn_eis_run)

        eis_layout.addLayout(eis_top)

        # Editable EIS parameters — pre-populated from preset, user-adjustable
        eis_params_row = QHBoxLayout()
        eis_params_row.addWidget(QLabel("f_hi (Hz):"))
        self._spin_eis_f_hi = QSpinBox()
        self._spin_eis_f_hi.setRange(1, 200_000)
        self._spin_eis_f_hi.setSingleStep(10_000)
        eis_params_row.addWidget(self._spin_eis_f_hi)
        eis_params_row.addWidget(QLabel("f_lo (mHz):"))
        self._spin_eis_f_lo = QSpinBox()
        self._spin_eis_f_lo.setRange(1, 200_000)
        eis_params_row.addWidget(self._spin_eis_f_lo)
        eis_params_row.addWidget(QLabel("npts:"))
        self._spin_eis_npts = QSpinBox()
        self._spin_eis_npts.setRange(5, 100)
        eis_params_row.addWidget(self._spin_eis_npts)
        eis_params_row.addWidget(QLabel("mVac:"))
        self._spin_eis_mv_ac = QSpinBox()
        self._spin_eis_mv_ac.setRange(1, 200)
        eis_params_row.addWidget(self._spin_eis_mv_ac)
        eis_params_row.addWidget(QLabel("mVdc:"))
        self._spin_eis_mv_dc = QSpinBox()
        self._spin_eis_mv_dc.setRange(-500, 500)
        eis_params_row.addWidget(self._spin_eis_mv_dc)
        eis_params_row.addStretch()
        self._combo_eis_preset.currentTextChanged.connect(self._on_eis_preset_changed)
        self._on_eis_preset_changed(self._combo_eis_preset.currentText())
        eis_layout.addLayout(eis_params_row)

        # Auto-fit row
        fit_row = QHBoxLayout()
        self._chk_autofit = QCheckBox("Auto-fit")
        self._chk_autofit.setChecked(True)
        fit_row.addWidget(self._chk_autofit)
        fit_row.addWidget(QLabel("Model:"))
        self._combo_fit_model = QComboBox()
        self._combo_fit_model.addItems(["simpleSalt", "flexSalt", "simpleSaltMembrane"])
        fit_row.addWidget(self._combo_fit_model)
        # Off by default and deliberately so: this tab is where non-standard
        # samples get measured — two arcs, a stack, something nobody has
        # characterised — and the scout picks ONE apex to plan around. Whether
        # that is the right thing to do is a per-sample judgement made here, at
        # the rig, so it is a checkbox rather than a deployment setting. A global
        # `[eis.scout] actuate` cannot switch this on; only this box can.
        self._chk_eis_scout = QCheckBox("Scout sweep")
        self._chk_eis_scout.setChecked(_manual_scout_default())
        self._chk_eis_scout.setToolTip(
            "Measure once, then take a second, wider or denser sweep only if the "
            "first one's arc did not close with enough band below its apex.\n\n"
            "Leave OFF for a non-standard sample: the planner assumes a single "
            "arc and will pick the tallest one if there are several. With it off "
            "the preset above runs exactly once and that is the measurement.")
        fit_row.addWidget(self._chk_eis_scout)
        self._chk_autosave = QCheckBox("Auto-save to DB")
        self._chk_autosave.setChecked(True)
        fit_row.addStretch()
        fit_row.addWidget(self._chk_autosave)
        eis_layout.addLayout(fit_row)

        # Conductivity row: geometry (L, t, w) OR empirical cell constant K
        sigma_row = QHBoxLayout()
        self._rb_geom = QRadioButton("Geometry:")
        self._rb_geom.setChecked(True)
        sigma_row.addWidget(self._rb_geom)

        sigma_row.addWidget(QLabel("L (cm):"))
        self._spin_eis_L = QDoubleSpinBox()
        self._spin_eis_L.setRange(0.0, 10.0)
        self._spin_eis_L.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_eis_L.setSingleStep(0.01)
        sigma_row.addWidget(self._spin_eis_L)

        sigma_row.addWidget(QLabel("t (cm):"))
        self._spin_eis_t = QDoubleSpinBox()
        self._spin_eis_t.setRange(0.0, 10.0)
        self._spin_eis_t.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_eis_t.setSingleStep(0.01)
        sigma_row.addWidget(self._spin_eis_t)

        sigma_row.addWidget(QLabel("w (cm):"))
        self._spin_eis_w = QDoubleSpinBox()
        self._spin_eis_w.setRange(0.0, 10.0)
        self._spin_eis_w.setDecimals(6)  # single-micron (0.0001 cm) resolution
        self._spin_eis_w.setSingleStep(0.01)
        sigma_row.addWidget(self._spin_eis_w)

        self._rb_K = QRadioButton("Cell constant K (cm⁻¹):")
        sigma_row.addSpacing(12)
        sigma_row.addWidget(self._rb_K)
        self._spin_eis_K = QDoubleSpinBox()
        self._spin_eis_K.setRange(0.0, 1000.0)
        self._spin_eis_K.setDecimals(4)
        self._spin_eis_K.setSingleStep(0.1)
        self._spin_eis_K.setEnabled(False)
        sigma_row.addWidget(self._spin_eis_K)
        sigma_row.addStretch()

        # Toggle enabled state when radio buttons switch
        self._rb_geom.toggled.connect(self._on_sigma_mode_toggled)
        eis_layout.addLayout(sigma_row)

        self._lbl_eis_status = QLabel("Ready")
        eis_layout.addWidget(self._lbl_eis_status)

        layout.addWidget(eis_grp)

    def _refresh_piezo_capability_status(self) -> None:
        if not self._piezo_enabled_cfg:
            return
        try:
            piezo = self._manager.get("piezo")
            status = piezo.status()
        except Exception:
            return
        if not isinstance(status, dict):
            return

        # The manual tab is built before async instrument connect completes.
        # Avoid latching a false legacy state from a pre-connect status snapshot.
        connected = status.get("connected")
        has_capability_fields = (
            "supports_l2" in status
            or "supports_cfg" in status
            or "config_supported" in status
        )
        if connected is False and not has_capability_fields:
            self._piezo_caps_checked = False
            self._piezo_config_supported = True
            for widget in self._piezo_channel_controls:
                widget.setEnabled(True)
            for widget in self._piezo_profile_controls:
                widget.setEnabled(False)
            self._lbl_piezo_status.setText("Piezo connecting: profile settings pending capability probe.")
            return

        supports_l2 = bool(status.get("supports_l2", False))
        supports_cfg = bool(status.get("supports_cfg", False))
        config_supported = bool(
            status.get("config_supported", supports_l2 or supports_cfg)
        )

        self._piezo_caps_checked = True
        self._piezo_config_supported = config_supported
        for widget in self._piezo_channel_controls:
            widget.setEnabled(True)
        for widget in self._piezo_profile_controls:
            widget.setEnabled(config_supported)

        if config_supported:
            if supports_l2:
                self._lbl_piezo_status.setText("Piezo connected (lean config supported).")
            else:
                self._lbl_piezo_status.setText("Piezo connected (CFG config supported).")
        else:
            self._lbl_piezo_status.setText("Piezo connected (legacy mode: profile settings disabled).")

    def _ensure_piezo_capability_status(self) -> None:
        if not self._piezo_caps_checked or not self._piezo_config_supported:
            self._refresh_piezo_capability_status()

    # --- PV polling -----------------------------------------------------------

    def _start_pv_polling(self) -> None:
        """Start background PV polling thread (keeps driver calls off main thread)."""
        self._pv_worker = _ManualPollingWorker(self._manager, parent=self)
        self._pv_worker.poll_done.connect(self._on_pv_poll_done)
        self._pv_worker.start()

    def _on_pv_poll_done(self, data: dict) -> None:
        """Update PV labels from background poll result (runs on main thread via signal)."""
        if self._piezo_enabled_cfg and (
            (not self._piezo_caps_checked) or (not self._piezo_config_supported)
        ):
            self._refresh_piezo_capability_status()

        if "temp_sp" in data:
            self._lbl_temp_sp.setText(f"SP: {data['temp_sp']:.1f} °C")
        if "temp_pv" in data:
            self._lbl_temp_pv.setText(f"PV: {data['temp_pv']:.1f} °C")
        if "rh" in data:
            self._lbl_rh.setText(f"RH: {data['rh']:.1f} %")
        if "pos" in data:
            pos = data["pos"]
            self._lbl_pos.setText(f"Position: ({float(pos[0]):.2f}, {float(pos[1]):.2f})")
        counts_by_pump = data.get("parallel_syringes_by_pump")
        if isinstance(counts_by_pump, dict) and counts_by_pump:
            for pump_id, pump in enumerate(self._pump_widgets):
                count = counts_by_pump.get(
                    pump_id,
                    counts_by_pump.get(str(pump_id), data.get("parallel_syringes", 1)),
                )
                pump["count_lbl"].setText(f"Syringes loaded: {int(count)}")
        elif "parallel_syringes" in data:
            count = int(data["parallel_syringes"])
            for pump in self._pump_widgets:
                pump["count_lbl"].setText(f"Syringes loaded: {count}")

    def _manual_commanded_volume(self, pump_id: int, target_uL: float) -> float:
        """Delivered → commanded for one manual dispense.

        Shares :class:`DeadVolumeCorrection` with the HT and campaign paths, so
        a manual dispense and an automated one of the same nominal volume put
        the same amount through the line. ``enabled=True`` is forced because the
        caller only reaches here when the tab's own checkbox is ticked.
        """
        from softae.core.liquid_handling import DeadVolumeCorrection

        correction = DeadVolumeCorrection.from_config([pump_id], enabled=True)
        run_index = self._manual_dispense_count_by_pump.get(pump_id, 0) + 1
        return float(
            correction.commanded([max(0.0, target_uL)], run_index=run_index)[0]
        )

    # --- Rig ownership: awareness, never enforcement --------------------------

    def refresh_rig_owner(self) -> "RunLock | None":
        """Show who owns the rig, if it is not this process. Returns that lock.

        The banner is the *whole* of this tab's response to a foreign owner. It
        does not disable a control, refuse a command, or offer to take the rig
        over, and that is a decision rather than an omission: the operator at the
        bench reaches for manual control precisely when something has gone wrong,
        and a lockout at that moment is the failure, not the protection.

        Stopping a run is a designated control with a scope of its own — rig-scale
        (the E-Stop already on the main toolbar), campaign-scale and terminal
        (Abort), campaign-scale and resumable (Pause). Each belongs to the
        container that surfaces the run, so this banner **routes to them rather
        than duplicating them**: the owner named here is another *process*, and a
        Pause button wired to nothing would be worse than no button. It reports
        and points; it does not promise what it cannot honour.
        """
        from softae.gui.widgets.rig_owner import (
            OCCUPIED,
            campaign_identity,
            foreign_rig_lock,
            owner_line,
        )

        lock = foreign_rig_lock()
        if lock is None:
            self._lbl_rig_owner.setVisible(False)
            self._lbl_rig_owner.setText("")
            self._lbl_rig_owner.setToolTip("")
            return None

        identity = campaign_identity(lock)
        subject = (
            f"Campaign '{identity[0]}' (run {identity[1]})" if identity
            else "Another process"
        )
        self._lbl_rig_owner.setText(
            f"<b style='color:#8e24aa'>{OCCUPIED}</b> — {subject} is driving this "
            f"rig: {owner_line(lock)}.<br>"
            f"Manual commands here still work and are <b>not</b> blocked — they "
            f"will interleave with that run. To quiet the rig first: pause or "
            f"abort the campaign from the process that owns it, or use the "
            f"E-Stop on the main toolbar to park the whole rig."
        )
        self._lbl_rig_owner.setToolTip(lock.describe())
        self._lbl_rig_owner.setVisible(True)
        return lock

    def _note_manual_actuation(self, action: str) -> "RunLock | None":
        """Record — and never refuse — a manual *action* issued over a live run.

        Returns the foreign lock when there was one, so a caller can annotate its
        own status line. **The caller proceeds either way.** This exists so that a
        collision, if one happens, is in the log with a timestamp and an owner
        beside it rather than being reconstructed afterwards from a ruined board.
        """
        lock = self.refresh_rig_owner()
        if lock is None:
            return None
        import structlog

        structlog.get_logger(__name__).warning(
            "manual_actuation_during_foreign_run",
            action=action, owner_pid=lock.pid, owner_what=lock.what,
            owner_started_at=lock.started_at,
            msg="manual control was used while another process held the rig — "
                "allowed deliberately; the operator at the bench decides",
        )
        self._lbl_last_command.setText(
            f"Last command: {action} — issued while {lock.what or 'another run'} "
            f"(PID {lock.pid}) holds the rig"
        )
        return lock

    # --- Slots ----------------------------------------------------------------

    def _safe_run(self, fn, error_title: str = "Error") -> None:
        """Run *fn* and show a message box on failure."""
        try:
            fn()
        except Exception as exc:
            QMessageBox.warning(self, error_title, str(exc))

    def _on_goto(self) -> None:
        self._note_manual_actuation("stage go-to")
        x, y = self._spin_x.value(), self._spin_y.value()
        self._btn_goto.setEnabled(False)

        def _do():
            stage = self._manager.get("stage")
            stage.move_to(x, y)
            return stage.live_position()

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda pos: self._lbl_pos.setText(f"Position: ({float(pos[0]):.2f}, {float(pos[1]):.2f})"))
        w.completed.connect(lambda _: self._btn_goto.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Stage Error", e))
        w.failed.connect(lambda _: self._btn_goto.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_jog(self, dx: float, dy: float) -> None:
        self._note_manual_actuation("stage jog")
        step = self._spin_jog_step.value()
        for b in self._jog_buttons:
            b.setEnabled(False)

        def _do():
            stage = self._manager.get("stage")
            stage.move_by(dx * step, dy * step)
            return stage.live_position()

        def _restore():
            for b in self._jog_buttons:
                b.setEnabled(True)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda pos: self._lbl_pos.setText(f"Position: ({float(pos[0]):.2f}, {float(pos[1]):.2f})"))
        w.completed.connect(lambda _: _restore())
        w.failed.connect(lambda e: QMessageBox.warning(self, "Stage Error", e))
        w.failed.connect(lambda _: _restore())
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_set_temp(self) -> None:
        self._note_manual_actuation("temperature setpoint")
        sp = self._spin_temp.value()
        self._btn_set_temp.setEnabled(False)

        def _do():
            tc = self._manager.get("temp_controller")
            tc.write_sp(T_SP=sp, print_flag=0)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._lbl_temp_sp.setText(f"SP: {sp:.1f} °C"))
        w.completed.connect(lambda _: self._btn_set_temp.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Temperature Error", e))
        w.failed.connect(lambda _: self._btn_set_temp.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_ramp(self) -> None:
        self._note_manual_actuation("temperature ramp")
        end = self._spin_ramp_end.value()
        rate = self._spin_ramp_rate.value()
        self._btn_ramp.setEnabled(False)

        def _do():
            tc = self._manager.get("temp_controller")
            start = tc.get_sp()
            # Spinbox rate is °C/min; the driver takes a total duration in
            # seconds plus a setpoint-update interval (see
            # AsyncTempController.ramp_linear / .anneal for the granularity).
            t_span = abs(end - start) / rate * 60.0
            up_int = max(t_span / 100.0, 1.0)
            tc.ramp_linear(T_start=start, T_end=end, t_span=t_span, up_int=up_int, print_flag=0)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._btn_ramp.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Ramp Error", e))
        w.failed.connect(lambda _: self._btn_ramp.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_set_rh(self) -> None:
        self._note_manual_actuation("humidity setpoint")
        sp = self._spin_rh.value()
        self._btn_set_rh.setEnabled(False)

        def _do():
            rh = self._manager.get("rh_controller")
            rh.set_setpoint(sp)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._btn_set_rh.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "RH Error", e))
        w.failed.connect(lambda _: self._btn_set_rh.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_rh_start(self) -> None:
        self._note_manual_actuation("humidity control start")
        sp = self._spin_rh.value()
        self._btn_rh_start.setEnabled(False)

        def _do():
            rh = self._manager.get("rh_controller")
            rh.set_setpoint(sp)
            rh.start()

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._btn_rh_start.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "RH Error", e))
        w.failed.connect(lambda _: self._btn_rh_start.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_rh_stop(self) -> None:
        self._note_manual_actuation("humidity control stop")
        self._btn_rh_stop.setEnabled(False)

        def _do():
            rh = self._manager.get("rh_controller")
            rh.stop()

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._btn_rh_stop.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "RH Error", e))
        w.failed.connect(lambda _: self._btn_rh_stop.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_head_retract(self) -> None:
        self._note_manual_actuation("dispenser head retract")
        self._btn_head_retract.setEnabled(False)

        def _do():
            syr = self._manager.get("syringe")
            syr.head_retract()

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._lbl_head_status.setText("Head: Retracted"))
        w.completed.connect(lambda _: self._lbl_head_status.setStyleSheet("font-weight: bold; color: #4CAF50;"))
        w.completed.connect(lambda _: self._btn_head_retract.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Syringe Error", e))
        w.failed.connect(lambda _: self._btn_head_retract.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_head_descend(self) -> None:
        self._note_manual_actuation("dispenser head descend")
        self._btn_head_descend.setEnabled(False)

        def _do():
            syr = self._manager.get("syringe")
            syr.head_descend()

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._lbl_head_status.setText("Head: Descended"))
        w.completed.connect(lambda _: self._lbl_head_status.setStyleSheet("font-weight: bold; color: #FF9800;"))
        w.completed.connect(lambda _: self._btn_head_descend.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Syringe Error", e))
        w.failed.connect(lambda _: self._btn_head_descend.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def refresh_head_label(self) -> None:
        """Sync the head-status label to the driver's registered belief.

        The single source of truth is ``syringe.is_head_up()``; this reflects
        state changed elsewhere (launch/start-gate verification prompts) so the
        label never drifts from the driver.  Best-effort — never raises.
        """
        try:
            syr = self._manager.get("syringe")
            is_up = bool(getattr(syr, "is_head_up")())
        except Exception:
            return
        if is_up:
            self._lbl_head_status.setText("Head: Retracted")
            self._lbl_head_status.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self._lbl_head_status.setText("Head: Descended")
            self._lbl_head_status.setStyleSheet("font-weight: bold; color: #FF9800;")

    # --- Manual-cast occupancy --------------------------------------------------

    def _read_stage_xy(self) -> tuple[float, float] | None:
        """Current stage ``(x, y)`` in mm, or ``None`` if unavailable."""
        try:
            stage = self._manager.get("stage")
            pos = stage.live_position()
            return float(pos[0]), float(pos[1])
        except Exception:
            return None

    def _pending_cast_target(self) -> tuple[int, int] | None:
        """``(board_id, electrode)`` a pump would occupy now, or ``None``.

        A manual pump only *casts* into a well when the head is DOWN and the
        stage sits over an electrode; a head-up pump (or one away from any well)
        is a hardware test and marks nothing.  Best-effort — any missing piece
        (no store, head up, stage offline, not over a well) yields ``None``.
        """
        if self._data_store is None:
            return None
        try:
            syr = self._manager.get("syringe")
            if bool(getattr(syr, "is_head_up")()):
                return None  # head raised → hardware test, not a cast
        except Exception:
            return None
        xy = self._read_stage_xy()
        if xy is None:
            return None
        try:
            from softae.core.deposition_steps import deposition_positions, resolve_pcb
            from softae.core.geometry import nearest_electrode

            _, pcb = resolve_pcb(None)
            ox, oy = deposition_positions().origin
            electrode = nearest_electrode(pcb, xy[0], xy[1], ox, oy)
            if electrode is None:
                return None
            return int(self._data_store.current_board_id()), int(electrode)
        except Exception:
            return None

    # --- Syringe stock (consumables interlock) --------------------------------

    def _reservoir_ledger(self):
        """The ledger attached to the syringe at startup, or ``None``.

        Read off the instrument rather than threaded through the constructor so
        this tab stays usable on a rig where no ledger was attached.
        """
        try:
            return getattr(self._manager.get("syringe"), "reservoir_ledger", None)
        except Exception:
            return None

    def _on_report_stock(self) -> None:
        """Open the stock declaration dialog, then refresh the readouts."""
        ledger = self._reservoir_ledger()
        if ledger is None:
            QMessageBox.information(
                self, "Syringe Stock",
                "No reservoir ledger is attached, so stock is not being tracked.")
            return
        from softae.gui.widgets.reservoir_dialog import ReservoirDialog

        ReservoirDialog(ledger, parent=self).exec()
        self.refresh_stock_labels()

    def refresh_stock_labels(self) -> None:
        """Show remaining stock per pump, flagging low and depleted distinctly."""
        ledger = self._reservoir_ledger()
        for pump_id, pw in enumerate(self._pump_widgets):
            label = pw.get("stock_lbl")
            if label is None:
                continue
            remaining = None if ledger is None else ledger.remaining_uL(pump_id)
            if ledger is None or remaining is None:
                # Undeclared is "unknown", never "empty" — say so plainly rather
                # than showing a number the operator might trust.
                label.setText("Stock: not tracked")
                label.setStyleSheet("color: gray;")
                continue
            label.setText(f"Stock: {remaining:.0f} µL")
            if remaining <= ledger.hard_stop_uL:
                label.setStyleSheet("color: #c0392b; font-weight: bold;")
            elif remaining <= ledger.soft_warn_uL:
                label.setStyleSheet("color: #c47f1a; font-weight: bold;")
            else:
                label.setStyleSheet("")

    def _on_infuse(self, pump_id: int) -> None:
        pw = self._pump_widgets[pump_id]
        rate = pw["rate"].value()
        target_vol = pw["vol"].value()
        correction_on = self._chk_apply_correction.isChecked()
        commanded_vol = (
            self._manual_commanded_volume(pump_id, target_vol)
            if correction_on
            else target_vol
        )
        btn = pw["btn"]

        # Occupancy: a head-down pump over a well casts into it.  Warn before
        # re-casting into a recorded-occupied well; record it after a successful
        # dispense.  Head-up pumps resolve to None and never mark occupancy.
        cast_target = self._pending_cast_target()
        if cast_target is not None:
            board_id, electrode = cast_target
            try:
                occupied = self._data_store.occupied_electrodes(board_id)
            except Exception:
                occupied = set()
            if electrode in occupied:
                from softae.gui.widgets.occupancy_guard import (
                    BoardReplacedDecision,
                    prompt_board_replaced,
                )
                decision = prompt_board_replaced(self, board_id, {electrode})
                if decision is BoardReplacedDecision.CANCEL:
                    return  # do not cast into an occupied well
                if decision is BoardReplacedDecision.FRESH:
                    cast_target = (board_id + 1, electrode)  # fresh board
                    # Persist immediately: the swap must survive even if this
                    # dispense fails or the session ends before it completes.
                    try:
                        self._data_store.set_active_board(cast_target[0])
                    except Exception:
                        import structlog
                        structlog.get_logger(__name__).warning(
                            "active_board_persist_error", board_id=cast_target[0]
                        )
                # CAST_ANYWAY: keep (board_id, electrode) — deliberate re-cast.

        # Noted here rather than at the top of the slot: everything above can
        # still decline, and the log line should mean "fluid was commanded", not
        # "a button was pressed".
        self._note_manual_actuation(f"pump {pump_id} dispense")
        btn.setEnabled(False)

        def _do():
            syr = self._manager.get("syringe")
            syr.single_pump(res_vol=1000, ID=pump_id, rate=rate, dispense_vol=commanded_vol)

        w = _CommandWorker(_do, parent=self)
        def _on_done(_):
            btn.setEnabled(True)
            self._manual_dispense_count_by_pump[pump_id] = self._manual_dispense_count_by_pump.get(pump_id, 0) + 1
            mode = "on" if correction_on else "off"
            self._lbl_last_command.setText(
                f"Last command: Pump {pump_id} target {target_vol:.2f} uL -> commanded {commanded_vol:.2f} uL (correction {mode})"
            )
            if cast_target is not None:
                try:
                    self._data_store.record_electrode_cast(cast_target[0], cast_target[1])
                except Exception:
                    import structlog
                    structlog.get_logger(__name__).warning(
                        "manual_occupancy_record_error",
                        board_id=cast_target[0], electrode=cast_target[1],
                    )
            self.refresh_stock_labels()

        w.completed.connect(_on_done)
        w.failed.connect(lambda e: QMessageBox.warning(self, "Syringe Error", e))
        w.failed.connect(lambda _: btn.setEnabled(True))
        # The ledger debits on *command*, so a failed dispense may still have
        # moved fluid — the readout has to follow that, not just successes.
        w.failed.connect(lambda _: self.refresh_stock_labels())
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_piezo_a_on(self) -> None:
        if not self._piezo_enabled_cfg:
            return
        self._note_manual_actuation("piezo channel A on")
        self._ensure_piezo_capability_status()
        self._btn_piezo_a_on.setEnabled(False)
        self._btn_piezo_a_off.setEnabled(False)

        def _do():
            piezo = self._manager.get("piezo")
            return piezo.set_channel("A", True)

        def _restore(_=None):
            self._btn_piezo_a_on.setEnabled(True)
            self._btn_piezo_a_off.setEnabled(True)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._lbl_piezo_status.setText("Piezo channel A enabled."))
        w.completed.connect(_restore)
        w.failed.connect(lambda e: QMessageBox.warning(self, "Piezo Error", e))
        w.failed.connect(_restore)
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_piezo_a_off(self) -> None:
        if not self._piezo_enabled_cfg:
            return
        self._note_manual_actuation("piezo channel A off")
        self._ensure_piezo_capability_status()
        self._btn_piezo_a_on.setEnabled(False)
        self._btn_piezo_a_off.setEnabled(False)

        def _do():
            piezo = self._manager.get("piezo")
            return piezo.set_channel("A", False)

        def _restore(_=None):
            self._btn_piezo_a_on.setEnabled(True)
            self._btn_piezo_a_off.setEnabled(True)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(lambda _: self._lbl_piezo_status.setText("Piezo channel A disabled."))
        w.completed.connect(_restore)
        w.failed.connect(lambda e: QMessageBox.warning(self, "Piezo Error", e))
        w.failed.connect(_restore)
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_piezo_apply_settings(self) -> None:
        if not self._piezo_enabled_cfg:
            return
        self._note_manual_actuation("piezo profile")
        self._ensure_piezo_capability_status()
        if not self._piezo_config_supported:
            self._lbl_piezo_status.setText("Profile settings unavailable on legacy piezo firmware.")
            return
        freq = self._spin_piezo_freq.value()
        on_s = self._spin_piezo_on_s.value()
        rest_s = self._spin_piezo_rest_s.value()
        self._btn_piezo_apply.setEnabled(False)

        def _do():
            piezo = self._manager.get("piezo")
            return piezo.apply_profile(freq, on_s, rest_s)

        w = _CommandWorker(_do, parent=self)
        w.completed.connect(
            lambda _: self._lbl_piezo_status.setText(
                f"Piezo profile applied: {freq} Hz, on={on_s:.3f}s, rest={rest_s:.3f}s"
            )
        )
        w.completed.connect(lambda _: self._btn_piezo_apply.setEnabled(True))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Piezo Error", e))
        w.failed.connect(lambda _: self._btn_piezo_apply.setEnabled(True))
        w.finished.connect(w.deleteLater)
        w.start()

    def _update_eis_pico_label(self, *_args) -> None:
        """Update the pico label from the Channel(s) field.

        Shows the routed pico when every parsed channel maps to the same one,
        ``"mixed"`` when a range spans both picos, or ``"—"`` when the field is
        empty/invalid.
        """
        from softae.config.loader import pico_for_channel
        from softae.core.channel_spec import ChannelSpecError, parse_channel_spec

        try:
            channels = parse_channel_spec(self._edit_eis_ch.text())
            picos = {pico_for_channel(ch) for ch in channels}
        except (ChannelSpecError, ValueError):
            self._lbl_eis_pico.setText("—")
            return
        self._lbl_eis_pico.setText(next(iter(picos)) if len(picos) == 1 else "mixed")

    def _on_eis_preset_changed(self, name: str) -> None:
        """Populate EIS parameter spinboxes from the selected preset."""
        from softae.config.loader import eis_presets
        p = eis_presets().get(name, {})
        if not p:
            return
        self._spin_eis_f_hi.setValue(p.get("f_hi", 200_000))
        self._spin_eis_f_lo.setValue(p.get("f_lo_mHz", 4_000))
        self._spin_eis_npts.setValue(p.get("npts", 35))
        self._spin_eis_mv_ac.setValue(p.get("mv_ac", 10))
        self._spin_eis_mv_dc.setValue(p.get("mv_dc", 0))

    def _on_sigma_mode_toggled(self, geom_checked: bool) -> None:
        """Enable geometry or cell-constant inputs based on radio selection."""
        self._spin_eis_L.setEnabled(geom_checked)
        self._spin_eis_t.setEnabled(geom_checked)
        self._spin_eis_w.setEnabled(geom_checked)
        self._spin_eis_K.setEnabled(not geom_checked)

    def _on_eis_run(self) -> None:
        if self._eis_thread is not None and self._eis_thread.isRunning():
            self._lbl_eis_status.setText("EIS already running...")
            return

        from softae.config.loader import pico_for_channel
        from softae.core.channel_spec import ChannelSpecError, parse_channel_spec

        try:
            channels = parse_channel_spec(self._edit_eis_ch.text())
        except ChannelSpecError as exc:
            self._lbl_eis_status.setText(f"Channel error: {exc}")
            return
        try:
            for ch in channels:
                pico_for_channel(ch)  # validate routing up-front for every channel
        except ValueError as exc:
            self._lbl_eis_status.setText(f"Routing error: {exc}")
            return
        self._note_manual_actuation("manual EIS")
        preset = self._combo_eis_preset.currentText()
        auto_fit = self._chk_autofit.isChecked()
        fit_model = self._combo_fit_model.currentText()
        auto_save = self._chk_autosave.isChecked()

        self._btn_eis_run.setEnabled(False)
        n = len(channels)
        self._lbl_eis_status.setText(
            f"Queueing EIS on {n} channels ({preset})..." if n > 1
            else f"Queueing EIS on ch {channels[0]} ({preset})...")

        # A fresh planner per Run press, reading the checkbox as it stands now.
        # Nothing an earlier press concluded may govern this one: the operator may
        # have changed RH, temperature, the sample — or this very setting — in
        # between, and the apex was measured moving ~100x across an RH change.
        from softae.core.eis_scout_scripts import ScoutPlanner as _ScoutPlanner

        self._eis_scout = _ScoutPlanner(
            site="manual_tab", actuate=self._chk_eis_scout.isChecked())

        self._eis_thread = QThread(self)
        self._eis_worker = _ManualEisWorker(
            self._manager,
            self._data_store,
            channels=channels,
            eis_params={
                "f_hi": self._spin_eis_f_hi.value(),
                "f_lo_mHz": self._spin_eis_f_lo.value(),
                "npts": self._spin_eis_npts.value(),
                "mv_ac": self._spin_eis_mv_ac.value(),
                "mv_dc": self._spin_eis_mv_dc.value(),
            },
            preset_label=preset,
            auto_fit=auto_fit,
            fit_model=fit_model,
            auto_save=auto_save,
            scout=self._eis_scout,
        )
        self._eis_worker.moveToThread(self._eis_thread)

        self._eis_thread.started.connect(self._eis_worker.run)
        self._eis_worker.progress.connect(self._on_eis_progress)
        self._eis_worker.finished.connect(self._on_eis_finished)
        self._eis_worker.error.connect(self._on_eis_error)

        self._eis_worker.finished.connect(self._eis_thread.quit)
        self._eis_worker.error.connect(self._eis_thread.quit)
        self._eis_thread.finished.connect(self._cleanup_eis_worker)
        self._eis_thread.start()

    def _on_eis_progress(self, message: str) -> None:
        self._lbl_eis_status.setText(message)

    def _conductivity_from_fit(self, fit_result: "FitResult | None") -> float | None:
        """Fitted ionic conductivity (S/cm) from a fit's R1, per the σ-mode inputs.

        Uses electrode geometry (L·t·w) or the empirical cell constant K depending
        on the selected radio button; returns ``None`` when unavailable.
        """
        if fit_result is None:
            return None
        r1 = fit_result.R1
        if not r1 or r1 <= 0:
            return None
        try:
            if self._rb_geom.isChecked():
                from softae.gui.eis_sigma import cell_sigma, gui_cell

                cell = gui_cell(self._spin_eis_L.value(), self._spin_eis_t.value(),
                                self._spin_eis_w.value())
                return cell_sigma(cell, r1)
            else:
                # The empirical-K route is *not* a cell-constant route: K is typed in
                # by the operator, so there is no geometry to build a cell from and
                # nothing here to migrate.
                K = self._spin_eis_K.value()
                if K > 0:
                    return float(K / r1)
        except Exception:
            return None
        return None

    def _on_eis_finished(self, payloads: object) -> None:
        # The worker emits a list (one payload per channel); tolerate a bare dict.
        if isinstance(payloads, dict):
            items = [payloads]
        elif isinstance(payloads, (list, tuple)):
            items = list(payloads)
        else:
            items = []
        self._btn_eis_run.setEnabled(True)
        if not items:
            self._lbl_eis_status.setText("No channels measured.")
            return
        if len(items) == 1:
            self._show_single_eis_result(items[0])
            return
        self._show_series_eis_results(items)

    def _show_single_eis_result(self, payload: dict) -> None:
        """Single-channel result: pop the fit/raw plot and report σ in the status."""
        import matplotlib.pyplot as plt

        eis_result: EISResult = payload["eis_result"]
        fit_result: FitResult | None = payload["fit_result"]
        ch = payload["channel"]
        model = payload["fit_model"]

        status_parts = [f"Done - ch {ch}"]

        fitted = bool(payload["auto_fit"]) and fit_result is not None
        if fitted:
            r0 = fit_result.R0 or float("nan")
            r1 = fit_result.R1 or float("nan")
            status_parts.append(f"R0={r0:.1f} Ohm  R1={r1:.1f} Ohm [{model}]")
            sigma = self._conductivity_from_fit(fit_result)
            if sigma is not None:
                extra = "" if self._rb_geom.isChecked() else f"  (K={self._spin_eis_K.value():.4g})"
                status_parts.append(f"σ={sigma:.4g} S/cm{extra}")

        # One renderer either way. Turning auto-fit off changes what is *known* about
        # the spectrum, not how it is drawn — it used to drop to the driver's own
        # plotting routine, which shares none of this palette, axes or conventions,
        # so the same measurement looked like it came from a different instrument.
        from softae.analysis.circuit_fitting import plot_eis_fit

        plot_eis_fit(eis_result, fit_result if fitted else None, show=False)
        plt.show(block=False)

        if payload.get("fit_error"):
            status_parts.append(f"Fit failed: {payload['fit_error']}")
        if payload.get("run_id"):
            status_parts.append(f"Saved run {payload['run_id']}")
        elif payload.get("db_missing"):
            status_parts.append("(DB not available)")

        self._lbl_eis_status.setText("  |  ".join(status_parts))

    def _show_series_eis_results(self, payloads: list[dict]) -> None:
        """Multi-channel result: one scrollable window; details go to the Analysis tab."""
        from softae.gui.widgets.eis_series_plot import EisSeriesPlotWidget

        window = EisSeriesPlotWidget(
            payloads, sigma_fn=self._conductivity_from_fit, parent=self)
        window.setWindowFlag(Qt.WindowType.Window, True)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        window.show()
        self._eis_series_window = window  # keep a reference so it isn't GC'd
        self._lbl_eis_status.setText(
            "Multiple channels run: see Analysis tab for measurement details.")

    def _on_eis_error(self, message: str) -> None:
        self._btn_eis_run.setEnabled(True)
        self._lbl_eis_status.setText(f"EIS Error: {message}")
        QMessageBox.warning(self, "EIS Error", message)

    def _cleanup_eis_worker(self) -> None:
        self._eis_worker = None
        self._eis_thread = None

    def _on_snap(self) -> None:
        """Capture a single frame (non-blocking via CameraWorker)."""
        self._ensure_cam_worker_running()
        if self._cam_worker and self._cam_worker.isRunning():
            self._cam_worker.request_frame(self._spin_cam_exp.value())

    def _on_live_toggle(self, checked: bool) -> None:
        """Start or stop the live preview timer."""
        if checked:
            self._ensure_cam_worker_running()
            self._cam_timer = QTimer(self)
            self._cam_timer.timeout.connect(self._grab_frame)
            self._cam_timer.start(1000)  # 1 FPS
        else:
            if hasattr(self, '_cam_timer') and self._cam_timer is not None:
                self._cam_timer.stop()
                self._cam_timer = None

    def _grab_frame(self) -> None:
        """Called by live preview timer — request one frame from the worker."""
        if self._cam_worker and self._cam_worker.isRunning():
            self._cam_worker.request_frame(self._spin_cam_exp.value())

    def _display_frame(self, arr) -> None:
        """Convert a numpy (H,W,3) uint8 array to QPixmap and show it."""
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap
        arr = np.ascontiguousarray(arr)          # ensure C-contiguous, fixes real camera views
        h, w, ch = arr.shape
        bytes_per_line = ch * w
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        qimg = qimg.copy()                       # detach QImage from numpy buffer ownership
        pixmap = QPixmap.fromImage(qimg)
        label_size = self._lbl_cam_image.size()
        if label_size.width() > 0 and label_size.height() > 0:
            # Scale to fit the label while keeping aspect ratio
            scaled = pixmap.scaled(
                label_size,
                aspectMode=Qt.AspectRatioMode.KeepAspectRatio,
                mode=Qt.TransformationMode.SmoothTransformation,
            )
            self._lbl_cam_image.setPixmap(scaled)
        else:
            # Widget not yet painted — use native size (will be resized on next paint)
            self._lbl_cam_image.setPixmap(pixmap)

    def _on_lamp_on(self) -> None:
        self._note_manual_actuation("lamp on")

        def go():
            lamp = self._manager.get("lamp")
            lamp.on()
        self._safe_run(go, "Lamp Error")

    def _on_lamp_off(self) -> None:
        self._note_manual_actuation("lamp off")

        def go():
            lamp = self._manager.get("lamp")
            lamp.off()
        self._safe_run(go, "Lamp Error")

    def cleanup(self) -> None:
        """Stop this tab's worker threads (idempotent)."""
        timer = getattr(self, "_rig_owner_timer", None)
        if timer is not None:
            timer.stop()
        if self._pv_worker is not None and self._pv_worker.isRunning():
            self._pv_worker.stop_worker()
        eis = getattr(self, "_eis_thread", None)
        if eis is not None and eis.isRunning():
            eis.quit()
            eis.wait(2000)

    def closeEvent(self, event) -> None:
        """Stop background resources when the tab is being destroyed."""

        if hasattr(self, "_cam_timer") and self._cam_timer is not None:
            self._cam_timer.stop()
            self._cam_timer = None

        self.cleanup()

        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        if self._pv_worker is not None and self._pv_worker.isRunning():
            self._pv_worker.stop_worker()
        self._rig_owner_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        if self._pv_worker is not None and not self._pv_worker.isRunning():
            self._pv_worker.start()
        # Who owns the rig may have changed entirely while this tab was hidden.
        self._rig_owner_timer.start()
        self.refresh_rig_owner()
        # A start-gate on another tab may have changed the head belief while
        # this tab was hidden; re-sync the label when it becomes visible.
        self.refresh_head_label()
        # Likewise for stock — an HT run or campaign may have drawn it down.
        self.refresh_stock_labels()
        super().showEvent(event)

    # --- Camera worker integration --------------------------------------------

    def set_camera_worker(self, worker: CameraWorker) -> None:
        """Receive the shared :class:`CameraWorker` from *MainWindow*."""
        self._cam_worker = worker
        worker.frame_ready.connect(self._display_frame)
        worker.error_occurred.connect(self._on_cam_error)

    def _ensure_cam_worker_running(self) -> None:
        """Start the camera worker thread if it is not already running."""
        if self._cam_worker is not None and not self._cam_worker.isRunning():
            self._cam_worker.start()

    def _on_cam_error(self, msg: str) -> None:
        """Log (and optionally display) camera worker errors."""
        import structlog
        structlog.get_logger(__name__).warning("camera_worker_error", error=msg)
