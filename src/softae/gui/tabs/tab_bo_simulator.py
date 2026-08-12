"""Tab: BO Simulator.

Offline, hardware-free panel for the :mod:`softae.campaigns` suite.  Loads a
dataset (treated as an imperfect ground-truth oracle), runs a pool-based
Bayesian-optimization campaign in a background thread, and visualises progress
across several linked views:

* **Convergence** — simple regret / best-so-far and surrogate RMSE per step.
* **Composition map** — a live conductivity heatmap over two composition axes
  that fills in as points are *uncovered*, with the sampling **trajectory**
  overlaid (numbered arrows) at a selectable RH/T slice.
* **Surrogate 3D** — the GP posterior-mean surface over the two composition axes
  after the run.

A **Campaign Mode** selector switches between *Optimize* (UCB/EI toward the
conductivity optimum) and *Explore* (active-learning acquisitions that reduce
surrogate uncertainty, with a model-accuracy stopping rule).

Shares its daemon-worker plumbing, convergence canvas, log pane, config
save/load and button-state helpers with the Live BO Campaign tab via
:class:`~softae.gui.tabs._bo_base.BOTabBase`.  Needs no ``InstrumentManager``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from softae.campaigns.acquisitions import ACQUISITIONS
from softae.campaigns.adapters import DataStoreAdapter
from softae.campaigns.config import BOCampaignConfig
from softae.campaigns.derived import DERIVED_OBJECTIVES
from softae.campaigns.noise import NOISE_MODELS
from softae.campaigns.objectives import OBJECTIVES
from softae.errors import CampaignError
from softae.gui.tabs._bo_base import BOTabBase
from softae.optimizers.surrogates import BACKENDS

if TYPE_CHECKING:
    from softae.core.data_store import DataStore

logger = structlog.get_logger(__name__)

# Acquisitions grouped by family, for the Campaign Mode selector.
_OPTIMIZE_ACQS = sorted(n for n, c in ACQUISITIONS.items() if c.family == "optimization")
_EXPLORE_ACQS = sorted(n for n, c in ACQUISITIONS.items() if c.family == "active_learning")
_ENV_AXES = ("rh_pct", "temp_C")


class BOSimulatorTab(BOTabBase):
    """Control panel for simulated Bayesian-optimization campaigns."""

    _CONFIG_TITLE = "BO Campaign Config"

    # Subclass-specific signal (composition-map / trajectory point).
    _sig_point = Signal(int, object)  # iter, params dict

    def __init__(
        self,
        *,
        data_store: "DataStore | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(data_store=data_store, parent=parent)
        self._heatmap_dirty = False

        # composition-map state (set at run start)
        self._comp_axes: list[str] = []
        self._x_levels: list[float] = []
        self._y_levels: list[float] = []
        self._env_axes: list[str] = []
        self._slice_map: dict[str, tuple] = {}           # combo label -> env tuple
        self._value_by_pid: dict[str, float] = {}        # heatmap value per candidate
        self._map_is_sigma: bool = True                  # σ map vs derived-objective map
        self._uncovered: dict[tuple, float] = {}         # (x, y, env) -> value
        self._trajectory: list[tuple] = []               # ordered (x, y, env)

        self._sig_point.connect(self._on_point)

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(6)
        h_split.setChildrenCollapsible(False)

        h_split.addWidget(self._build_left())
        h_split.addWidget(self._build_right())
        h_split.setStretchFactor(0, 2)
        h_split.setStretchFactor(1, 3)
        root.addWidget(h_split)

    def _build_left(self) -> QWidget:
        """Scrollable parameter column with a pinned control bar at the bottom."""
        params = QWidget()
        col = QVBoxLayout(params)
        col.setContentsMargins(2, 2, 2, 2)
        for grp in (
            self._grp_dataset(),
            self._grp_optimizer(),
            self._grp_noise(),
            self._grp_stopping(),
            self._grp_rails(),
            self._grp_notes(),
        ):
            col.addWidget(grp)
        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(params)

        wrap = QWidget()
        wrap_v = QVBoxLayout(wrap)
        wrap_v.setContentsMargins(0, 0, 0, 0)
        wrap_v.addWidget(scroll, stretch=1)
        # Pinned, always-visible control bar (shared with the Live tab).
        wrap_v.addWidget(self._make_control_bar(run_label="▶  Run Campaign"))
        return wrap

    def _grp_dataset(self) -> QGroupBox:
        grp = QGroupBox("Dataset (ground-truth oracle)")
        form = QFormLayout(grp)
        self._combo_adapter = QComboBox()
        self._combo_adapter.addItems(["aggregated_txt", "datastore"])
        form.addRow("Adapter:", self._combo_adapter)
        path_row = QHBoxLayout()
        self._le_path = QLineEdit()
        self._le_path.setPlaceholderText("path to aggregated conductivity .txt")
        path_row.addWidget(self._le_path)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._on_browse)
        path_row.addWidget(btn)
        path_w = QWidget()
        path_w.setLayout(path_row)
        form.addRow("Path:", path_w)
        return grp

    def _grp_optimizer(self) -> QGroupBox:
        grp = QGroupBox("Optimizer && Objective")
        form = QFormLayout(grp)

        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Optimize", "Explore"])
        self._combo_mode.setToolTip(
            "Optimize: drive toward max/min conductivity (UCB/EI).\n"
            "Explore: reduce surrogate uncertainty (active learning); "
            "uses a model-accuracy stopping rule."
        )
        self._combo_mode.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Campaign mode:", self._combo_mode)

        self._combo_objdir = QComboBox()
        self._combo_objdir.addItems(["maximize", "minimize"])
        form.addRow("Direction:", self._combo_objdir)
        self._combo_transform = QComboBox()
        self._combo_transform.addItems(sorted(OBJECTIVES))
        self._combo_transform.setCurrentText("log10_sigma")
        form.addRow("Transform:", self._combo_transform)
        self._combo_tempobj = QComboBox()
        self._combo_tempobj.addItems(["none", *sorted(DERIVED_OBJECTIVES)])
        self._combo_tempobj.setToolTip(
            "Fold σ(T) into an Arrhenius/VFT parameter as the objective "
            "(requires multiple temperatures per composition)."
        )
        form.addRow("Temp. objective:", self._combo_tempobj)
        self._spin_targetT = QDoubleSpinBox()
        self._spin_targetT.setRange(-50.0, 300.0)
        self._spin_targetT.setValue(25.0)
        self._spin_targetT.setSuffix(" °C")
        self._spin_targetT.setToolTip("Target T for *_sigma_at_T objectives")
        form.addRow("Target T:", self._spin_targetT)
        self._combo_acq = QComboBox()
        self._combo_acq.addItems(_OPTIMIZE_ACQS)
        self._combo_acq.setCurrentText("ucb")
        form.addRow("Acquisition:", self._combo_acq)
        self._combo_backend = QComboBox()
        self._combo_backend.addItems(sorted(BACKENDS))
        self._combo_backend.setCurrentText("sklearn")
        form.addRow("Backend:", self._combo_backend)
        self._spin_ninit = QSpinBox()
        self._spin_ninit.setRange(1, 1000)
        self._spin_ninit.setValue(5)
        form.addRow("Initial (warm-up):", self._spin_ninit)
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 1_000_000)
        form.addRow("Seed:", self._spin_seed)
        self._spin_kappa = QDoubleSpinBox()
        self._spin_kappa.setRange(0.0, 20.0)
        self._spin_kappa.setSingleStep(0.5)
        self._spin_kappa.setValue(2.0)
        form.addRow("UCB kappa:", self._spin_kappa)
        self._chk_noiseless = QCheckBox("Noiseless oracle (reveal true means)")
        self._chk_noiseless.setChecked(True)
        form.addRow(self._chk_noiseless)
        return grp

    def _grp_noise(self) -> QGroupBox:
        grp = QGroupBox("Noise Model")
        form = QFormLayout(grp)
        self._combo_noise = QComboBox()
        self._combo_noise.addItems(sorted(NOISE_MODELS))
        self._combo_noise.setCurrentText("composite")
        form.addRow("Model:", self._combo_noise)
        self._le_sources = QLineEdit("replicate, fit_quality")
        self._le_sources.setToolTip("Composite sources, comma-separated")
        form.addRow("Sources:", self._le_sources)
        self._combo_combine = QComboBox()
        self._combo_combine.addItems(["sum", "max"])
        form.addRow("Combine:", self._combo_combine)
        self._combo_channel = QComboBox()
        self._combo_channel.addItems(["alpha", "acquisition_weight", "both"])
        form.addRow("Channel:", self._combo_channel)
        self._chk_mean = QCheckBox("Target is replicate mean (SEM²)")
        self._chk_mean.setChecked(True)
        form.addRow(self._chk_mean)
        self._combo_shrink = QComboBox()
        self._combo_shrink.addItems(["pooled", "none"])
        form.addRow("Replicate shrinkage:", self._combo_shrink)
        self._spin_kfit = QDoubleSpinBox()
        self._spin_kfit.setRange(0.0, 100.0)
        self._spin_kfit.setSingleStep(0.5)
        self._spin_kfit.setValue(1.0)
        form.addRow("k_fit:", self._spin_kfit)
        return grp

    def _grp_stopping(self) -> QGroupBox:
        grp = QGroupBox("Stopping Rule")
        form = QFormLayout(grp)
        self._combo_stop = QComboBox()
        self._combo_stop.addItems(["optimization", "model_accuracy"])
        form.addRow("Mode:", self._combo_stop)
        self._spin_reltol = QDoubleSpinBox()
        self._spin_reltol.setRange(0.0, 1.0)
        self._spin_reltol.setDecimals(4)
        self._spin_reltol.setSingleStep(0.01)
        self._spin_reltol.setValue(0.01)
        form.addRow("rel_tol:", self._spin_reltol)
        self._spin_patience = QSpinBox()
        self._spin_patience.setRange(1, 100)
        self._spin_patience.setValue(5)
        form.addRow("patience:", self._spin_patience)
        self._spin_rmsetol = QDoubleSpinBox()
        self._spin_rmsetol.setRange(0.0, 100.0)
        self._spin_rmsetol.setDecimals(3)
        self._spin_rmsetol.setValue(0.25)
        form.addRow("rmse_tol:", self._spin_rmsetol)
        self._spin_covtol = QDoubleSpinBox()
        self._spin_covtol.setRange(0.0, 1.0)
        self._spin_covtol.setSingleStep(0.05)
        self._spin_covtol.setValue(0.9)
        form.addRow("coverage_tol:", self._spin_covtol)
        self._spin_maxsteps = QSpinBox()
        self._spin_maxsteps.setRange(0, 100_000)
        self._spin_maxsteps.setToolTip("0 = whole pool")
        form.addRow("max_steps (0=pool):", self._spin_maxsteps)
        return grp

    def _grp_rails(self) -> QGroupBox:
        grp = QGroupBox("Rail Handling")
        form = QFormLayout(grp)
        self._spin_railsigma = QDoubleSpinBox()
        self._spin_railsigma.setRange(0.0, 10.0)
        self._spin_railsigma.setDecimals(4)
        self._spin_railsigma.setSingleStep(0.01)
        self._spin_railsigma.setValue(0.05)
        self._spin_railsigma.setToolTip(
            "σ ceiling above which a low-impedance point is a rail (0 disables)"
        )
        form.addRow("rail σ ceiling:", self._spin_railsigma)
        self._chk_exclude_rails = QCheckBox("Exclude rails from optimum")
        self._chk_exclude_rails.setChecked(True)
        form.addRow(self._chk_exclude_rails)
        return grp

    def _grp_notes(self) -> QGroupBox:
        grp = QGroupBox("Campaign Notes (stored with run)")
        lay = QVBoxLayout(grp)
        self._te_annotation = QTextEdit()
        self._te_annotation.setPlaceholderText("Material, dataset, purpose…")
        self._te_annotation.setFixedHeight(48)
        lay.addWidget(self._te_annotation)
        return grp

    def _build_right(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(5)
        right.setChildrenCollapsible(False)

        tabs = QTabWidget()

        # ── Convergence page (shared canvas from the base) ───────────────────
        conv = QWidget()
        conv_v = QVBoxLayout(conv)
        conv_v.addWidget(
            self._make_convergence_canvas(
                primary_label="simple regret",
                secondary_label="surrogate RMSE",
                xlabel="step (formulations uncovered)",
            )
        )
        b_exp = QPushButton("Export Plot…")
        b_exp.clicked.connect(lambda: self._export_fig(self._conv_fig, "bo_convergence.png"))
        conv_v.addWidget(b_exp)
        tabs.addTab(conv, "Convergence")

        # ── Composition map page ─────────────────────────────────────────────
        hm = QWidget()
        hm_v = QVBoxLayout(hm)
        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Slice:"))
        self._combo_slice = QComboBox()
        self._combo_slice.currentTextChanged.connect(lambda _: self._draw_heatmap())
        slice_row.addWidget(self._combo_slice)
        self._chk_traj = QCheckBox("Show trajectory")
        self._chk_traj.setChecked(True)
        self._chk_traj.toggled.connect(lambda _: self._draw_heatmap())
        slice_row.addWidget(self._chk_traj)
        slice_row.addStretch()
        hm_v.addLayout(slice_row)
        self._fig_hm = Figure(tight_layout=True)
        self._ax_hm = self._fig_hm.add_subplot(111)
        self._canvas_hm = FigureCanvasQTAgg(self._fig_hm)
        self._canvas_hm.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hm_v.addWidget(self._canvas_hm)
        b_exp_hm = QPushButton("Export Plot…")
        b_exp_hm.clicked.connect(lambda: self._export_fig(self._fig_hm, "bo_composition_map.png"))
        hm_v.addWidget(b_exp_hm)
        tabs.addTab(hm, "Composition map")

        # ── Surrogate 3D page ────────────────────────────────────────────────
        s3 = QWidget()
        s3_v = QVBoxLayout(s3)
        s3_row = QHBoxLayout()
        s3_row.addWidget(QLabel("Slice:"))
        self._combo_slice3d = QComboBox()
        self._combo_slice3d.currentTextChanged.connect(lambda _: self._draw_surrogate3d())
        s3_row.addWidget(self._combo_slice3d)
        b_refresh = QPushButton("Refresh surrogate")
        b_refresh.clicked.connect(self._draw_surrogate3d)
        s3_row.addWidget(b_refresh)
        s3_row.addStretch()
        s3_v.addLayout(s3_row)
        self._fig_s3 = Figure(tight_layout=True)
        self._ax_s3 = self._fig_s3.add_subplot(111, projection="3d")
        self._canvas_s3 = FigureCanvasQTAgg(self._fig_s3)
        self._canvas_s3.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        s3_v.addWidget(self._canvas_s3)
        b_exp_s3 = QPushButton("Export Plot…")
        b_exp_s3.clicked.connect(lambda: self._export_fig(self._fig_s3, "bo_surrogate3d.png"))
        s3_v.addWidget(b_exp_s3)
        tabs.addTab(s3, "Surrogate 3D")

        right.addWidget(tabs)

        # ── Log (shared pane from the base) ──────────────────────────────────
        right.addWidget(self._make_log_pane(title="Campaign Log"))
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)
        return right

    # ── Campaign mode ────────────────────────────────────────────────────────

    def _on_mode_changed(self, mode: str) -> None:
        """Swap acquisition family and stopping rule to match the mode."""
        explore = mode == "Explore"
        acqs = _EXPLORE_ACQS if explore else _OPTIMIZE_ACQS
        current = self._combo_acq.currentText()
        self._combo_acq.blockSignals(True)
        self._combo_acq.clear()
        self._combo_acq.addItems(acqs)
        if current in acqs:
            self._combo_acq.setCurrentText(current)
        elif explore:
            self._combo_acq.setCurrentText("max_variance")
        else:
            self._combo_acq.setCurrentText("ucb")
        self._combo_acq.blockSignals(False)
        self._combo_stop.setCurrentText("model_accuracy" if explore else "optimization")

    # ── config <-> UI ──────────────────────────────────────────────────────

    def _build_config(self) -> BOCampaignConfig:
        sources = [s.strip() for s in self._le_sources.text().split(",") if s.strip()]
        return BOCampaignConfig(
            dataset_adapter=self._combo_adapter.currentText(),
            dataset_path=self._le_path.text().strip(),
            backend=self._combo_backend.currentText(),
            acquisition=self._combo_acq.currentText(),
            objective_direction=self._combo_objdir.currentText(),
            transform=self._combo_transform.currentText(),
            n_initial=self._spin_ninit.value(),
            seed=self._spin_seed.value(),
            kappa=self._spin_kappa.value(),
            temperature_objective=self._combo_tempobj.currentText(),
            target_temp_C=self._spin_targetT.value(),
            noise_model=self._combo_noise.currentText(),
            noise_sources=sources,
            noise_combine=self._combo_combine.currentText(),
            noise_channel=self._combo_channel.currentText(),
            target_is_mean=self._chk_mean.isChecked(),
            replicate_shrinkage=self._combo_shrink.currentText(),
            k_fit=self._spin_kfit.value(),
            stopping_mode=self._combo_stop.currentText(),
            rel_tol=self._spin_reltol.value(),
            patience=self._spin_patience.value(),
            rmse_tol=self._spin_rmsetol.value(),
            coverage_tol=self._spin_covtol.value(),
            max_steps=self._spin_maxsteps.value() or None,
            rail_sigma_ceiling=self._spin_railsigma.value() or None,
            exclude_rails_from_optimum=self._chk_exclude_rails.isChecked(),
            noiseless_oracle=self._chk_noiseless.isChecked(),
            annotation=self._te_annotation.toPlainText().strip(),
        )

    def _config_from_json(self, text: str) -> BOCampaignConfig:
        return BOCampaignConfig.from_json(text)

    def _populate_from_config(self, cfg: BOCampaignConfig) -> None:
        # Set mode first so the acquisition list matches the saved acquisition.
        acq_cls = ACQUISITIONS.get(cfg.acquisition)
        mode = "Explore" if (acq_cls and acq_cls.family == "active_learning") else "Optimize"
        self._combo_mode.setCurrentText(mode)
        self._on_mode_changed(mode)
        self._combo_adapter.setCurrentText(cfg.dataset_adapter)
        self._le_path.setText(cfg.dataset_path)
        self._combo_backend.setCurrentText(cfg.backend)
        self._combo_acq.setCurrentText(cfg.acquisition)
        self._combo_objdir.setCurrentText(cfg.objective_direction)
        self._combo_transform.setCurrentText(cfg.transform)
        self._spin_ninit.setValue(cfg.n_initial)
        self._spin_seed.setValue(cfg.seed)
        self._spin_kappa.setValue(cfg.kappa)
        self._combo_tempobj.setCurrentText(cfg.temperature_objective)
        self._spin_targetT.setValue(cfg.target_temp_C)
        self._combo_noise.setCurrentText(cfg.noise_model)
        self._le_sources.setText(", ".join(cfg.noise_sources))
        self._combo_combine.setCurrentText(cfg.noise_combine)
        self._combo_channel.setCurrentText(cfg.noise_channel)
        self._chk_mean.setChecked(cfg.target_is_mean)
        self._combo_shrink.setCurrentText(cfg.replicate_shrinkage)
        self._spin_kfit.setValue(cfg.k_fit)
        self._combo_stop.setCurrentText(cfg.stopping_mode)
        self._spin_reltol.setValue(cfg.rel_tol)
        self._spin_patience.setValue(cfg.patience)
        self._spin_rmsetol.setValue(cfg.rmse_tol)
        self._spin_covtol.setValue(cfg.coverage_tol)
        self._spin_maxsteps.setValue(cfg.max_steps or 0)
        self._spin_railsigma.setValue(cfg.rail_sigma_ceiling or 0.0)
        self._chk_exclude_rails.setChecked(cfg.exclude_rails_from_optimum)
        self._chk_noiseless.setChecked(cfg.noiseless_oracle)
        self._te_annotation.setPlainText(cfg.annotation)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select dataset", "", "Text/CSV (*.txt *.csv);;All files (*)"
        )
        if path:
            self._le_path.setText(path)

    # ── run / abort ─────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        from softae.campaigns.runner import build_campaign

        try:
            cfg = self._build_config()
            # The datastore adapter needs the live store; the path field holds
            # the run_id in that case.
            adapter = None
            if cfg.dataset_adapter == "datastore":
                if self._data_store is None:
                    raise CampaignError("no DataStore is attached to this tab")
                adapter = DataStoreAdapter(self._data_store, cfg.dataset_path)
            runner = build_campaign(
                cfg,
                on_step=self._emit_step,
                should_abort=lambda: self._abort_requested,
                adapter=adapter,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Campaign Error", str(exc))
            return

        self._abort_requested = False
        self._result = None
        self._runner = runner
        self._cfg = cfg
        self._reset_views(runner)

        self._progress.setRange(0, runner.max_steps)
        self._progress.setValue(0)
        self._lbl_status.setText(f"Running — pool {runner.pool_size}")
        opt_params, opt_value = runner.dataset.true_optimum(maximize=cfg.maximize)
        self._sig_log.emit(
            f"Pool {runner.pool_size} candidates; true optimum {opt_value:.3f} at {opt_params}"
        )

        self._start_worker(self._run_thread, runner, name="bo-campaign")

    def _reset_views(self, runner) -> None:
        """Clear plots and (re)build the composition-map grid for this dataset."""
        self._reset_convergence()
        self._heatmap_dirty = False
        self._uncovered = {}
        self._trajectory = []
        # Rebuild the heatmap figure from scratch (drops any prior colorbar).
        self._fig_hm.clear()
        self._ax_hm = self._fig_hm.add_subplot(111)
        self._canvas_hm.draw_idle()
        self._ax_s3.cla()
        self._canvas_s3.draw_idle()
        self._log.clear()
        self._build_grid_metadata(runner.dataset)

    def _build_grid_metadata(self, dataset) -> None:
        """Derive composition axes, levels, env slices, and value lookup from the pool."""
        import numpy as np

        cells = dataset.cells
        # Heatmap value = σ when available, else the derived objective (e.g. Ea).
        self._map_is_sigma = any(np.isfinite(c.sigma_mean) for c in cells)
        self._value_by_pid = {
            c.point_id: (c.sigma_mean if np.isfinite(c.sigma_mean) else c.y)
            for c in cells
        }
        comp = [c for c in dataset.param_columns if c not in _ENV_AXES]
        self._comp_axes = comp[:2]
        self._env_axes = [c for c in dataset.param_columns if c in _ENV_AXES]

        if len(self._comp_axes) >= 2:
            xa, ya = self._comp_axes
            self._x_levels = sorted({c.params[xa] for c in cells})
            self._y_levels = sorted({c.params[ya] for c in cells})
        else:
            self._x_levels, self._y_levels = [], []

        # Distinct environment slices (e.g. RH/T combinations).
        slices: dict[str, tuple] = {}
        for c in cells:
            env = tuple(round(float(c.params[a]), 6) for a in self._env_axes)
            label = ", ".join(
                f"{a.replace('_pct', '').replace('_C', '')}={c.params[a]:g}"
                for a in self._env_axes
            ) or "all"
            slices[label] = env
        self._slice_map = slices

        # Default to the slice containing the true optimum.
        try:
            opt_params, _ = dataset.true_optimum(maximize=getattr(self, "_cfg").maximize)
            opt_env = tuple(round(float(opt_params[a]), 6) for a in self._env_axes)
            default_label = next((k for k, v in slices.items() if v == opt_env), None)
        except Exception:
            default_label = None

        for combo in (self._combo_slice, self._combo_slice3d):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(slices.keys()))
            if default_label:
                combo.setCurrentText(default_label)
            combo.blockSignals(False)

    def _emit_step(self, iteration: int, params: dict, metrics: Any) -> None:
        """on_step callback (worker thread) → marshal to GUI via signals.

        The base convergence signal is ``(iteration, primary, secondary)``:
        primary = simple regret, secondary = surrogate RMSE.
        """
        self._sig_step.emit(
            iteration + 1,
            float(metrics.simple_regret),
            float(metrics.surrogate_rmse),
        )
        self._sig_point.emit(iteration, dict(params))

    def _on_step(self, iteration: int, primary: float, secondary: float) -> None:
        # Base handles the convergence buffers + coalesced redraw; the Simulator
        # additionally advances its progress bar (iteration is 1-based here).
        super()._on_step(iteration, primary, secondary)
        self._progress.setValue(int(iteration))

    def _run_thread(self, runner) -> None:
        try:
            result = runner.run()
            self._result = result
            if self._data_store is not None:
                try:
                    from softae.campaigns.persistence import record_campaign

                    rid = record_campaign(self._data_store, result, config=self._cfg)
                    self._sig_log.emit(f"  saved to data store: run {rid}")
                except Exception as exc:
                    self._sig_log.emit(f"  ⚠ data-store save failed: {exc}")
            status = "converged" if result.converged else "stopped"
            msg = (
                f"{status} after {result.n_steps} steps; best {result.best_value:.3f} "
                f"(optimum {result.true_optimum_value:.3f})"
            )
            self._sig_done.emit(True, msg)
        except Exception as exc:
            logger.exception("bo_campaign_thread_error", error=str(exc))
            self._sig_done.emit(False, str(exc))

    # ── GUI-thread slots: composition map ────────────────────────────────────

    def _on_point(self, iteration: int, params: dict) -> None:
        if len(self._comp_axes) < 2:
            return
        xa, ya = self._comp_axes
        env = tuple(round(float(params[a]), 6) for a in self._env_axes)
        key = (round(float(params[xa]), 6), round(float(params[ya]), 6), env)
        # value lookup via the dataset point_id for this params dict.
        pid = self._runner.dataset.point_id_for(params) if self._runner else None
        value = self._value_by_pid.get(pid, float("nan")) if pid else float("nan")
        self._uncovered[key] = value
        self._trajectory.append(key)
        if not self._heatmap_dirty:
            self._heatmap_dirty = True
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self._flush_heatmap)

    def _flush_heatmap(self) -> None:
        if not self._heatmap_dirty:
            return
        self._heatmap_dirty = False
        self._draw_heatmap()

    def _draw_heatmap(self) -> None:
        """Draw the uncovered-conductivity heatmap + trajectory for the slice.

        The whole figure is cleared and rebuilt on every call so the colorbar
        never accumulates (each ``fig.colorbar`` steals axes space that
        ``.remove()`` does not give back) — the plot always conforms to the full
        tab dimensions and shows only the current run.
        """
        self._fig_hm.clear()
        self._ax_hm = self._fig_hm.add_subplot(111)

        if len(self._comp_axes) < 2 or not self._x_levels:
            self._ax_hm.text(0.5, 0.5, "Needs ≥ 2 composition axes",
                             ha="center", va="center", transform=self._ax_hm.transAxes)
            self._canvas_hm.draw_idle()
            return
        import numpy as np
        from matplotlib.colors import LogNorm

        env = self._slice_map.get(self._combo_slice.currentText())
        xa, ya = self._comp_axes
        nx, ny = len(self._x_levels), len(self._y_levels)
        grid = np.full((ny, nx), np.nan)
        for (xv, yv, ev), value in self._uncovered.items():
            if ev != env or not np.isfinite(value):
                continue
            if self._map_is_sigma and value <= 0:
                continue
            xi = self._nearest_index(self._x_levels, xv)
            yi = self._nearest_index(self._y_levels, yv)
            if xi is not None and yi is not None:
                grid[yi, xi] = value

        finite = grid[np.isfinite(grid)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            # Log scale only for strictly-positive σ maps; derived objectives
            # (which may be negative, e.g. log10 σ) use a linear scale.
            use_log = self._map_is_sigma and lo > 0 and hi > lo
            norm = LogNorm(vmin=lo, vmax=hi) if use_log else None
            cmap = self._cmap_with_bad()
            im = self._ax_hm.imshow(
                grid, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
            )
            label = (
                "Conductivity (S/cm) — log scale" if self._map_is_sigma
                else f"objective ({self._combo_tempobj.currentText()})"
            )
            self._fig_hm.colorbar(im, ax=self._ax_hm, label=label)
        else:
            self._ax_hm.text(0.5, 0.5, "No points uncovered yet",
                             ha="center", va="center", transform=self._ax_hm.transAxes)

        self._ax_hm.set_xticks(range(nx))
        self._ax_hm.set_xticklabels([f"{v:g}" for v in self._x_levels])
        self._ax_hm.set_yticks(range(ny))
        self._ax_hm.set_yticklabels([f"{v:g}" for v in self._y_levels])
        self._ax_hm.set_xlabel(xa)
        self._ax_hm.set_ylabel(ya)
        kind = "conductivity" if self._map_is_sigma else "objective"
        self._ax_hm.set_title(f"Uncovered {kind} — {self._combo_slice.currentText()}")

        if self._chk_traj.isChecked():
            self._overlay_trajectory(env)
        self._canvas_hm.draw_idle()

    def _overlay_trajectory(self, env) -> None:
        """Numbered arrows over the heatmap showing visit order within this slice."""
        pts = []
        for (xv, yv, ev) in self._trajectory:
            if ev != env:
                continue
            xi = self._nearest_index(self._x_levels, xv)
            yi = self._nearest_index(self._y_levels, yv)
            if xi is not None and yi is not None:
                pts.append((xi, yi))
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            self._ax_hm.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0, alpha=0.7),
            )
        for order, (xi, yi) in enumerate(pts):
            self._ax_hm.text(
                xi, yi, str(order + 1), color="black", fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec="black", alpha=0.7),
            )

    # ── GUI-thread slots: surrogate 3D ───────────────────────────────────────

    def _draw_surrogate3d(self) -> None:
        """GP posterior-mean surface over the two composition axes (after run)."""
        self._ax_s3.cla()
        if len(self._comp_axes) < 2:
            self._ax_s3.text2D(0.5, 0.5, "Needs ≥ 2 composition axes",
                               ha="center", va="center", transform=self._ax_s3.transAxes)
            self._canvas_s3.draw_idle()
            return
        opt = getattr(self._runner, "optimizer", None) if self._runner else None
        if opt is None or opt.n_trials < 1:
            self._ax_s3.text2D(0.5, 0.5, "Run a campaign first",
                               ha="center", va="center", transform=self._ax_s3.transAxes)
            self._canvas_s3.draw_idle()
            return
        import numpy as np

        try:
            xa, ya = self._comp_axes
            env = self._slice_map.get(self._combo_slice3d.currentText())
            base = self._slice_base_params(env)
            gx = np.linspace(min(self._x_levels), max(self._x_levels), 24)
            gy = np.linspace(min(self._y_levels), max(self._y_levels), 24)
            GX, GY = np.meshgrid(gx, gy)
            grid_params = []
            for yi in range(GY.shape[0]):
                for xi in range(GX.shape[1]):
                    p = dict(base)
                    p[xa] = float(GX[yi, xi])
                    p[ya] = float(GY[yi, xi])
                    grid_params.append(p)
            X = np.array([opt._encode(p) for p in grid_params], dtype=float)
            mu, _ = opt.backend.predict(X)
            Z = mu.reshape(GX.shape)
        except Exception as exc:
            self._ax_s3.text2D(0.5, 0.5, f"Surrogate unavailable:\n{exc}",
                               ha="center", va="center", transform=self._ax_s3.transAxes)
            self._canvas_s3.draw_idle()
            return

        self._ax_s3.plot_surface(GX, GY, Z, cmap="viridis", alpha=0.55, linewidth=0)

        # Overlay the real data points *of this slice* at their true values.
        # Iterate the pool cells filtered to the selected env so each candidate
        # is plotted exactly once (no duplication) and its z matches THIS slice's
        # RH/T — not the first RH's, which the old composition-only match used.
        uncovered = {
            (round(xv, 6), round(yv, 6))
            for (xv, yv, ev) in self._uncovered
            if ev == env
        }
        order: dict[tuple, int] = {}
        for (xv, yv, ev) in self._trajectory:
            if ev != env:
                continue
            k = (round(float(xv), 6), round(float(yv), 6))
            order.setdefault(k, len(order) + 1)

        samp: list[tuple] = []
        unsamp: list[tuple] = []
        for c in self._runner.dataset.cells:
            cenv = tuple(round(float(c.params[a]), 6) for a in self._env_axes)
            if cenv != env:
                continue
            kx, ky = round(float(c.params[xa]), 6), round(float(c.params[ya]), 6)
            pt = (float(c.params[xa]), float(c.params[ya]), float(c.y))
            (samp if (kx, ky) in uncovered else unsamp).append(pt + ((kx, ky),))

        if unsamp:
            self._ax_s3.scatter(
                [p[0] for p in unsamp], [p[1] for p in unsamp], [p[2] for p in unsamp],
                color="0.6", s=16, alpha=0.5, depthshade=False, label="pool (not sampled)",
            )
        if samp:
            self._ax_s3.scatter(
                [p[0] for p in samp], [p[1] for p in samp], [p[2] for p in samp],
                color="crimson", s=42, depthshade=True, label="sampled",
            )
            for x, y, z, key in samp:
                n = order.get(key)
                if n is not None:
                    self._ax_s3.text(x, y, z, f"  {n}", color="black", fontsize=7)

        self._ax_s3.set_xlabel(xa)
        self._ax_s3.set_ylabel(ya)
        self._ax_s3.set_zlabel(self._combo_transform.currentText())
        self._ax_s3.set_title(
            f"GP surrogate mean — {self._combo_slice3d.currentText()}  "
            f"({len(samp)} sampled / {len(samp) + len(unsamp)} in slice)"
        )
        if samp or unsamp:
            self._ax_s3.legend(loc="upper left", fontsize=7)
        self._canvas_s3.draw_idle()

    def _slice_base_params(self, env) -> dict:
        """A representative full param dict for a slice (fixes env + extra axes)."""
        for c in self._runner.dataset.cells:
            cenv = tuple(round(float(c.params[a]), 6) for a in self._env_axes)
            if cenv == env:
                return dict(c.params)
        return dict(self._runner.dataset.cells[0].params)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_index(levels: list[float], value: float) -> int | None:
        for i, lv in enumerate(levels):
            if abs(lv - value) < 1e-6:
                return i
        return None

    @staticmethod
    def _cmap_with_bad():
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps["RdBu_r"].copy()  # matplotlib >= 3.6
        except (AttributeError, KeyError):  # pragma: no cover - old matplotlib
            import matplotlib.cm as cm
            cmap = cm.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
        return cmap

    def _on_done_extra(self, success: bool) -> None:
        if success and self._result is not None:
            self._log_trajectory_descriptor()
            self._draw_heatmap()
            self._draw_surrogate3d()

    def _log_trajectory_descriptor(self) -> None:
        """Summarise the sampling trajectory: coverage + normalised path length."""
        import numpy as np

        n = len(self._trajectory)
        pool = self._runner.dataset.size if self._runner else 0
        if n == 0 or not self._comp_axes:
            return
        # Normalised path length across composition axes (0..1 per axis span).
        spans = {}
        for ax_name, levels in zip(self._comp_axes, (self._x_levels, self._y_levels)):
            lo, hi = min(levels), max(levels)
            spans[ax_name] = (hi - lo) or 1.0
        xa, ya = self._comp_axes
        path = 0.0
        for i in range(1, n):
            (x0, y0, _), (x1, y1, _) = self._trajectory[i - 1], self._trajectory[i]
            dx = (x1 - x0) / spans[xa]
            dy = (y1 - y0) / spans[ya]
            path += float(np.hypot(dx, dy))
        coverage = 100.0 * n / pool if pool else 0.0
        self._log.append(
            f"  trajectory: {n} steps, {coverage:.0f}% of pool, "
            f"normalised path length {path:.2f}"
        )
