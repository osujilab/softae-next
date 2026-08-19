"""Tab: Live BO Campaign.

Drives a **real** closed autonomous loop (``suggest → execute → analyze → tell``)
via :func:`softae.core.autonomous_wiring.run_autonomous_campaign` over the
:class:`~softae.server.manager.InstrumentManager`.  With the default mock
manager every instrument is simulated, so the tab runs a fully functional
closed loop out of the box; swapping in a real manager is a later config change.

Shares its daemon-worker plumbing, convergence canvas, log pane and
button-state helpers with the offline BO Simulator via
:class:`~softae.gui.tabs._bo_base.BOTabBase`.  All run state is per-instance, so
a Live run and a Simulator run can execute concurrently in the background.

Two prior-informed BO hooks are exposed on the panel and wired into the
:class:`~softae.core.autonomous_wiring.CampaignSpec`:

* **Seed observations** — prior ``(params, value)`` points fed to the optimizer
  via ``tell`` before the loop (warm-start), loaded from a JSON file.
* **Prior mean** — an optional physics model ``m(params) -> float`` the GP models
  the residual from (a simple built-in "linear (demo)" stand-in is provided).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from softae.config.loader import default_pcb_name, pcb_configs
from softae.core.autonomous_wiring import CampaignSpec, resolve_direction
from softae.core.geometry import electrode_count
from softae.errors import CampaignError
from softae.gui.tabs._autonomous_run import AutonomousRunMixin
from softae.gui.tabs._bo_base import BOTabBase
from softae.gui.widgets.campaign_control import CampaignControlBar, outcome_note
from softae.gui.widgets.composition_axes_editor import CompositionAxesEditor

if TYPE_CHECKING:
    from softae.core.data_store import DataStore
    from softae.server.manager import InstrumentManager

logger = structlog.get_logger(__name__)

#: What the optimizer searches. Order matters — index 1 is composition mode.
#:
#: These are the two campaign *modes*, not a good option and a fallback. Raw volumes
#: make exploration easy and feasibility native, at the cost of stock identity: with
#: no elution there is no dry thickness, so the objective can only be mean |Z|.
#: Composition targets cost an up-front solve and give every trial a predicted
#: thickness, which is what makes conductivity available. ``[eis] objective = "auto"``
#: follows whichever is chosen — see :func:`softae.core.autonomous_wiring.resolve_objective`.
SEARCH_MODES = ["Raw volumes", "Composition targets"]


class _LiveAborted(Exception):
    """Raised from the event stream to unwind the loop on a user abort."""


def _linear_demo_prior(params: dict[str, Any]) -> float:
    """A stand-in physics model: a deterministic weighted sum of the params.

    Weights ascend with the sorted parameter name so the value is reproducible
    regardless of dict ordering.  This demonstrates the residual-GP path (the GP
    learns only the correction to this trend).
    """
    return float(
        sum((i + 1) * float(v) for i, (_, v) in enumerate(sorted(params.items())))
    )


class LiveBOCampaignTab(AutonomousRunMixin, BOTabBase):
    """Control panel for a live, hardware-in-the-loop BO campaign.

    The BO-rich instance of an autonomous run (P2.4): everything about *driving
    the rig* — head gate, board gates, shutdown blocking, pre-flight, execution —
    comes from :class:`AutonomousRunMixin`, so this tab owns only the Bayesian
    parts. The general Autonomous façade mixes in the same harness rather than
    re-implementing gates that carry safety meaning.
    """

    _CONFIG_TITLE = "Live BO Campaign Config"

    # Subclass-specific signal: a suggested/evaluated point for the scatter.
    _sig_point = Signal(object, float)  # params dict, objective

    def __init__(
        self,
        manager: "InstrumentManager",
        *,
        data_store: "DataStore | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(data_store=data_store, parent=parent)
        self._manager = manager

        # Prior hooks (per-instance).
        self._seed_observations: list[tuple[dict[str, Any], float]] = []

        # Head gate, board gates, shutdown blocking, pre-flight, execution.
        self._init_autonomous_run()

        # Run-time convergence bookkeeping (per-instance).
        self._maximize: bool = True
        self._best_obj: float | None = None

        # Scatter buffers (per-instance).
        self._scatter_axes: tuple[str, ...] = ()
        self._scatter_x: list[float] = []
        self._scatter_y: list[float] = []
        self._scatter_c: list[float] = []
        self._scatter_dirty: bool = False

        self._sig_point.connect(self._on_live_point)

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
        col = QVBoxLayout()
        params = QWidget()
        params.setLayout(col)
        col.setContentsMargins(2, 2, 2, 2)
        col.addWidget(self._grp_space())
        col.addWidget(self._grp_optimizer())
        col.addWidget(self._grp_board())
        col.addWidget(self._grp_priors())
        col.addStretch()

        wrap = QWidget()
        wrap_v = QVBoxLayout(wrap)
        wrap_v.setContentsMargins(0, 0, 0, 0)
        wrap_v.addWidget(params, stretch=1)
        # Live campaigns have no exportable JSON result object → no export button.
        wrap_v.addWidget(
            self._make_control_bar(run_label="▶  Run Live Campaign", with_export=False)
        )
        # Beside the in-process Abort, and deliberately not merged with it: that
        # one is a flag this thread reads, and reaches nothing when the campaign
        # is running in another process. This one is a request written to the
        # run directory, which is the only thing that reaches a campaign we do
        # not own. Campaign-scoped, so it belongs in the tab that surfaces the
        # campaign; the rig-scale stop stays on the toolbar.
        self._campaign_controls = CampaignControlBar(parent=wrap)
        self._campaign_controls.acknowledged.connect(self._on_control_ack)
        wrap_v.addWidget(self._campaign_controls)
        return wrap

    def _on_control_ack(self, ack: dict[str, Any]) -> None:
        """Put the campaign's answer in the log beside its other events."""
        self._log_line(
            f"  ⇄ {ack.get('action', 'control')} → {ack.get('outcome')}: "
            f"{outcome_note(str(ack.get('outcome') or ''))}"
        )

    def _grp_space(self) -> QGroupBox:
        grp = QGroupBox("Parameter Space (continuous)")
        lay = QVBoxLayout(grp)

        name_row = QFormLayout()
        self._le_name = QLineEdit("live_bo")
        name_row.addRow("Campaign name:", self._le_name)
        self._le_channels = QLineEdit("1")
        self._le_channels.setToolTip("Electrodes to deposit on, e.g. \"1, 3-6\"")
        name_row.addRow("Channels:", self._le_channels)

        self._combo_search_mode = QComboBox()
        self._combo_search_mode.addItems(SEARCH_MODES)
        self._combo_search_mode.setToolTip(
            "What the optimizer searches.\n\n"
            "Raw volumes — per-pump µL directly. Exploration is easy and feasibility\n"
            "  is native (a volume limit is just a bound), but the twin has no stock\n"
            "  identity, so there is no dry thickness and the objective can only be\n"
            "  mean |Z|, minimised. Composition is worked out afterwards.\n\n"
            "Composition targets — the same target vocabulary as the deposition twin\n"
            "  (molar ratio / dried fraction / concentration), each searched between\n"
            "  a Low and a High. The solver turns every suggestion into pump volumes,\n"
            "  so each trial has a predicted thickness and the objective becomes\n"
            "  conductivity, maximised."
        )
        self._combo_search_mode.currentIndexChanged.connect(self._on_search_mode_changed)
        name_row.addRow("Search over:", self._combo_search_mode)
        lay.addLayout(name_row)

        self._tbl_params = QTableWidget(0, 3)
        self._tbl_params.setHorizontalHeaderLabels(["name", "low", "high"])
        self._tbl_params.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._tbl_params.verticalHeader().setVisible(False)
        self._tbl_params.setMinimumHeight(120)
        lay.addWidget(self._tbl_params)
        # Sensible default two-parameter composition space.
        self._add_param_row("vol_p0", 5.0, 30.0)
        self._add_param_row("vol_p1", 5.0, 30.0)

        btn_row = QHBoxLayout()
        b_add = QPushButton("Add")
        b_add.clicked.connect(lambda: self._add_param_row("vol_pN", 5.0, 30.0))
        btn_row.addWidget(b_add)
        b_del = QPushButton("Remove")
        b_del.clicked.connect(self._remove_param_row)
        btn_row.addWidget(b_del)
        btn_row.addStretch()
        self._wid_vol_buttons = QWidget()
        self._wid_vol_buttons.setLayout(btn_row)
        lay.addWidget(self._wid_vol_buttons)

        # ── Composition mode: the twin's targets, each given a range ──────────
        self._axes_editor = CompositionAxesEditor()
        self._axes_editor.setVisible(False)
        lay.addWidget(self._axes_editor)

        comp_form = QFormLayout()
        self._spin_dep_uL = QDoubleSpinBox()
        self._spin_dep_uL.setRange(0.01, 1000.0)
        self._spin_dep_uL.setDecimals(2)
        self._spin_dep_uL.setValue(6.0)
        self._spin_dep_uL.setToolTip(
            "Dried film volume per electrode — the scale target the solver adds "
            "alongside the composition targets. This is what sets how much is cast; "
            "the axes only set the ratios."
        )
        comp_form.addRow("Deposit (µL dried):", self._spin_dep_uL)
        self._lbl_stocks = QLabel("")
        self._lbl_stocks.setWordWrap(True)
        comp_form.addRow("Stocks:", self._lbl_stocks)
        self._wid_comp_form = QWidget()
        self._wid_comp_form.setLayout(comp_form)
        self._wid_comp_form.setVisible(False)
        lay.addWidget(self._wid_comp_form)
        return grp

    # ── Search mode (raw volumes ↔ composition targets) ──────────────────────

    def _search_mode(self) -> str:
        """``"volumes"`` or ``"composition"`` — see :data:`SEARCH_MODES`."""
        return ("composition" if self._combo_search_mode.currentIndex() == 1
                else "volumes")

    def _on_search_mode_changed(self, _index: int = 0) -> None:
        composition = self._search_mode() == "composition"
        self._tbl_params.setVisible(not composition)
        self._wid_vol_buttons.setVisible(not composition)
        self._axes_editor.setVisible(composition)
        self._wid_comp_form.setVisible(composition)
        if composition:
            self._refresh_stock_choices()

    def _load_stocks(self):
        """``(stocks, pump_assignment, chem_catalog)`` from the rig's pump loadout.

        Reads the same shared catalogs and the same persisted loadout the deposition
        twin and the consumables ledger use, so a composition campaign is built from
        what is actually on the pumps rather than from a second list typed here that
        could disagree with the rig.
        """
        from softae.core.composition_axes import stocks_from_loadout
        from softae.core.stock_assignment import load_loadout

        try:
            chem_catalog, sol_catalog = self._load_catalogs()
            loadout = load_loadout(getattr(self, "_data_store", None))
            stocks, pumps = stocks_from_loadout(loadout, sol_catalog)
            return stocks, pumps, chem_catalog
        except Exception:
            logger.warning("live_bo_stocks_unavailable", exc_info=True)
            return {}, {}, None

    @staticmethod
    def _load_catalogs():
        from softae.config import loader
        from softae.core.formulation import ChemicalCatalog, SolutionCatalog

        root = loader.data_root()
        return (ChemicalCatalog.load_csv(root / "chemicals.csv"),
                SolutionCatalog.load_csv(root / "solutions.csv"))

    def _refresh_stock_choices(self) -> None:
        """Feed the axes editor the names available from the loaded stocks."""
        from softae.core.formulation import (
            deposited_component_names,
            species_concentration,
        )

        stocks, pumps, chem_catalog = self._load_stocks()
        if not stocks:
            self._lbl_stocks.setText(
                "No stocks declared on the pumps — set the pump loadout in the "
                "Formulation Manager before searching compositions."
            )
            self._axes_editor.set_available([], [], 0)
            return

        species: set[str] = set()
        components: set[str] = set()
        for sol in stocks.values():
            # A stock naming an unknown chemical, or malformed in the CSV, drops out
            # of the *name suggestions* only — it must not take the editor with it.
            # The names are a convenience; the combos stay editable either way.
            try:
                species.update(species_concentration(sol, chem_catalog).keys())
                components.update(deposited_component_names(sol))
            except Exception:
                logger.debug("stock_names_unavailable", exc_info=True)
                continue
        self._axes_editor.set_available(
            sorted(species), sorted(components), len(stocks))
        self._lbl_stocks.setText(
            "; ".join(f"{name} → pump {pumps[name]}" for name in sorted(stocks)))

    def _add_param_row(self, name: str, low: float, high: float) -> None:
        r = self._tbl_params.rowCount()
        self._tbl_params.insertRow(r)
        self._tbl_params.setItem(r, 0, QTableWidgetItem(str(name)))
        self._tbl_params.setItem(r, 1, QTableWidgetItem(f"{low:g}"))
        self._tbl_params.setItem(r, 2, QTableWidgetItem(f"{high:g}"))

    def _remove_param_row(self) -> None:
        r = self._tbl_params.currentRow()
        if r < 0:
            r = self._tbl_params.rowCount() - 1
        if r >= 0:
            self._tbl_params.removeRow(r)

    def _grp_optimizer(self) -> QGroupBox:
        grp = QGroupBox("Optimizer")
        form = QFormLayout(grp)
        self._combo_objdir = QComboBox()
        self._combo_objdir.addItems(["auto", "maximize", "minimize"])
        self._combo_objdir.setToolTip(
            "The direction is fixed by the metric, not chosen alongside it:\n"
            "conductivity is maximised, mean |Z| is minimised — they are the same\n"
            "goal expressed two ways.\n\n"
            "auto  — derive both from what this campaign can measure. A campaign\n"
            "        with a composition is steered on σ; a volume-only campaign has\n"
            "        no dry thickness, so it is steered on mean |Z|.\n"
            "Explicit — honoured only if it agrees with the resolved metric; a\n"
            "        contradiction is refused rather than run, because the campaign\n"
            "        would spend its whole budget finding the worst conductor on\n"
            "        the board while every step reported progress."
        )
        form.addRow("Direction:", self._combo_objdir)
        self._combo_acq = QComboBox()
        self._combo_acq.addItems(["ucb", "ei"])
        form.addRow("Acquisition:", self._combo_acq)
        self._spin_budget = QSpinBox()
        self._spin_budget.setRange(1, 1000)
        self._spin_budget.setValue(8)
        form.addRow("Budget (trials):", self._spin_budget)
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 1_000_000)
        self._spin_seed.setValue(42)
        form.addRow("Seed:", self._spin_seed)
        self._spin_kappa = QDoubleSpinBox()
        self._spin_kappa.setRange(0.0, 20.0)
        self._spin_kappa.setSingleStep(0.5)
        self._spin_kappa.setValue(2.0)
        form.addRow("UCB kappa:", self._spin_kappa)
        self._spin_timescale = QDoubleSpinBox()
        self._spin_timescale.setRange(0.0, 5.0)
        self._spin_timescale.setSingleStep(0.05)
        self._spin_timescale.setDecimals(3)
        self._spin_timescale.setValue(0.05)  # low → fast mock/demo runs
        self._spin_timescale.setToolTip(
            "Scales all routine dwells; keep low (e.g. 0–0.1) for fast demo runs."
        )
        form.addRow("Time scale:", self._spin_timescale)
        self._chk_batch = QCheckBox("Batch across channels (q = #channels)")
        self._chk_batch.setToolTip(
            "q-batch BO: each round proposes q distinct suggestions and casts one "
            "per electrode in a single run, scoring each against its own channel. "
            "Off → one suggestion replicated across all channels. Rounds are "
            "atomic, so budget rounds up to a multiple of q."
        )
        self._chk_batch.toggled.connect(self._combo_batch_strategy_set_enabled)
        form.addRow("Parallelism:", self._chk_batch)
        self._combo_batch_strategy = QComboBox()
        self._combo_batch_strategy.addItems(
            ["constant_liar", "kriging_believer", "botorch_mc (planned)"]
        )
        self._combo_batch_strategy.setEnabled(False)
        self._combo_batch_strategy.setToolTip(
            "How the q batch points are diversified: constant-liar (robust "
            "default), Kriging-believer (uses the GP's own mean), or a planned "
            "BoTorch Monte-Carlo (qEI/qNEI) integration."
        )
        form.addRow("Batch strategy:", self._combo_batch_strategy)
        return grp

    def _combo_batch_strategy_set_enabled(self, on: bool) -> None:
        self._combo_batch_strategy.setEnabled(on)

    def _batch_strategy_name(self) -> str:
        # Strip the "(planned)" annotation to the bare registry key.
        return self._combo_batch_strategy.currentText().split(" ", 1)[0]

    def _grp_priors(self) -> QGroupBox:
        grp = QGroupBox("Prior-informed BO")
        form = QFormLayout(grp)

        self._combo_prior = QComboBox()
        self._combo_prior.addItems(["none", "linear (demo)"])
        self._combo_prior.setToolTip(
            "Prior mean m(params) → objective; the GP models the residual from it."
        )
        form.addRow("Prior mean:", self._combo_prior)

        seed_row = QHBoxLayout()
        b_seed = QPushButton("Load seeds…")
        b_seed.clicked.connect(self._on_load_seeds)
        seed_row.addWidget(b_seed)
        self._lbl_seeds = QLabel("0 seed observations")
        seed_row.addWidget(self._lbl_seeds)
        seed_row.addStretch()
        seed_w = QWidget()
        seed_w.setLayout(seed_row)
        form.addRow("Warm-start:", seed_w)
        return grp

    def _grp_board(self) -> QGroupBox:
        grp = QGroupBox("Electrode board")
        form = QFormLayout(grp)

        # PCB selector — the single source of truth for board electrode count
        # (and campaign provenance). Defaults to the configured default board.
        self._pcbs = pcb_configs()
        self._combo_pcb = QComboBox()
        for name in sorted(self._pcbs):
            self._combo_pcb.addItem(
                f"{name}  ({electrode_count(self._pcbs[name])} electrodes)",
                userData=name,
            )
        default = default_pcb_name()
        if default:
            idx = self._combo_pcb.findData(default)
            if idx >= 0:
                self._combo_pcb.setCurrentIndex(idx)
        self._combo_pcb.setToolTip("Board layout; its electrode count seeds capacity.")
        self._combo_pcb.currentIndexChanged.connect(self._on_board_pcb_changed)
        form.addRow("Board (PCB):", self._combo_pcb)

        self._chk_board = QCheckBox("Single-use electrodes (board exchange)")
        self._chk_board.setChecked(True)  # electrodes are single-use → on by default
        self._chk_board.setToolTip(
            "Drop-cast wells are single-use: each sample consumes a fresh "
            "electrode. When a board fills you are prompted to insert a new plate "
            "(or cancel the run); the fresh board is equilibrated before the "
            "campaign continues."
        )
        self._chk_board.toggled.connect(self._on_board_toggle)
        form.addRow(self._chk_board)
        self._spin_capacity = QSpinBox()
        self._spin_capacity.setRange(1, 256)
        self._spin_capacity.setValue(self._pcb_electrode_count())  # from the PCB
        self._spin_capacity.setToolTip(
            "Electrodes per board — seeded from the selected PCB; override if needed."
        )
        form.addRow("Board capacity:", self._spin_capacity)
        self._spin_equil = QDoubleSpinBox()
        self._spin_equil.setRange(0.0, 3600.0)
        self._spin_equil.setValue(60.0)
        self._spin_equil.setSuffix(" s")
        self._spin_equil.setToolTip("Equilibration dwell after a board swap.")
        form.addRow("Equilibration:", self._spin_equil)
        return grp

    def _pcb_electrode_count(self) -> int:
        name = self._combo_pcb.currentData()
        if name and name in self._pcbs:
            return electrode_count(self._pcbs[name])
        return 32

    def _on_board_pcb_changed(self, *_: Any) -> None:
        # Re-seed capacity from the newly selected board's electrode count.
        self._spin_capacity.setValue(self._pcb_electrode_count())

    def _on_board_toggle(self, on: bool) -> None:
        self._spin_capacity.setEnabled(on)
        self._spin_equil.setEnabled(on)

    # Board gates live in AutonomousRunMixin — they carry safety meaning
    # (resume protection, bounded waits) and must not be per-tab.

    def _on_abort(self) -> None:
        # Release any pending board prompts so the worker unwinds on abort.
        self._release_board_gates()
        super()._on_abort()

    def _build_right(self) -> QWidget:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        right = QSplitter(Qt.Orientation.Vertical)
        right.setHandleWidth(5)
        right.setChildrenCollapsible(False)

        from PySide6.QtWidgets import QTabWidget

        tabs = QTabWidget()

        # Convergence page (shared canvas from the base).
        conv = QWidget()
        conv_v = QVBoxLayout(conv)
        conv_v.addWidget(
            self._make_convergence_canvas(
                primary_label="best objective",
                secondary_label="latest objective",
                xlabel="trial",
            )
        )
        b_exp = QPushButton("Export Plot…")
        b_exp.clicked.connect(lambda: self._export_fig(self._conv_fig, "live_bo_convergence.png"))
        conv_v.addWidget(b_exp)
        tabs.addTab(conv, "Convergence")

        # Suggested-points scatter (first two params, colored by objective).
        sc = QWidget()
        sc_v = QVBoxLayout(sc)
        self._fig_sc = Figure(tight_layout=True)
        self._ax_sc = self._fig_sc.add_subplot(111)
        self._canvas_sc = FigureCanvasQTAgg(self._fig_sc)
        self._canvas_sc.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sc_v.addWidget(self._canvas_sc)
        b_exp_sc = QPushButton("Export Plot…")
        b_exp_sc.clicked.connect(lambda: self._export_fig(self._fig_sc, "live_bo_points.png"))
        sc_v.addWidget(b_exp_sc)
        tabs.addTab(sc, "Suggested points")

        right.addWidget(tabs)
        right.addWidget(self._make_log_pane(title="Campaign Log"))
        right.setStretchFactor(0, 4)
        right.setStretchFactor(1, 1)
        return right

    # ── Channels parsing ─────────────────────────────────────────────────────

    def _parse_channels(self) -> tuple[int, ...]:
        raw = self._le_channels.text()
        out: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
                if lo > hi:
                    raise ValueError(f"range start must be <= end: '{part}'")
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
        seen: set[int] = set()
        result = [c for c in out if not (c in seen or seen.add(c))]
        if not result:
            raise ValueError("at least one channel must be specified")
        return tuple(result)

    # ── config <-> UI ──────────────────────────────────────────────────────

    def _selected_prior_mean(self):
        return _linear_demo_prior if self._combo_prior.currentText() == "linear (demo)" else None

    def _read_parameter_space(self) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
        space: dict[str, dict[str, Any]] = {}
        names: list[str] = []
        for row in range(self._tbl_params.rowCount()):
            n_item = self._tbl_params.item(row, 0)
            if n_item is None or not n_item.text().strip():
                continue
            name = n_item.text().strip()
            try:
                low = float(self._tbl_params.item(row, 1).text())
                high = float(self._tbl_params.item(row, 2).text())
            except (AttributeError, ValueError):
                raise ValueError(f"parameter '{name}' has an invalid low/high")
            if high <= low:
                raise ValueError(f"parameter '{name}': high must exceed low")
            space[name] = {"type": "float", "low": low, "high": high}
            names.append(name)
        if not space:
            raise ValueError("at least one parameter is required")
        return space, tuple(names)

    def _read_composition_space(self):
        """``(parameter_space, general_formulation)`` for composition mode.

        Raises ``ValueError`` with the operator's own vocabulary — this is called
        from the Run button, where the message becomes the dialog.
        """
        from softae.core.autonomous_wiring import GeneralFormulation

        axes = self._axes_editor.axes()
        issues = self._axes_editor.issues()
        if issues:
            raise ValueError(" ".join(issues))

        stocks, pump_assignment, chem_catalog = self._load_stocks()
        if not stocks:
            raise ValueError(
                "Composition mode needs stocks on the pumps. Declare the pump "
                "loadout in the Formulation Manager, or switch to raw volumes."
            )

        space = self._axes_editor.parameter_space()
        if not space:
            raise ValueError("Every composition target is pinned — nothing to search.")

        return space, GeneralFormulation(
            stocks=stocks,
            catalog=chem_catalog,
            pump_assignment=pump_assignment,
            target_deposition_uL=self._spin_dep_uL.value(),
            build_targets=self._axes_editor.build_targets_fn(),
            # budget_uL is filled from the board's well capacity by
            # run_autonomous_campaign — the board is the authority on what fits.
        )

    def _build_config(self) -> CampaignSpec:
        composition = self._search_mode() == "composition"
        if composition:
            space, general = self._read_composition_space()
            # Volumes are *solved*, not searched: pump ids come from the loadout, and
            # vol_params stays empty so nothing treats an axis as a µL value.
            names: tuple[str, ...] = ()
            pump_ids = tuple(sorted(general.pump_assignment.values()))
        else:
            space, names = self._read_parameter_space()
            general = None
            pump_ids = tuple(range(len(names)))

        return CampaignSpec(
            name=self._le_name.text().strip() or "live_bo",
            channels=self._parse_channels(),
            pcb_name=self._combo_pcb.currentData(),
            parameter_space=space,
            vol_params=names,
            general_formulation=general,
            pump_ids=pump_ids,
            objective=self._combo_objdir.currentText(),
            optimizer="bayesian",
            acquisition=self._combo_acq.currentText(),
            kappa=self._spin_kappa.value(),
            batch=self._chk_batch.isChecked(),
            batch_strategy=self._batch_strategy_name(),
            budget=self._spin_budget.value(),
            seed=self._spin_seed.value(),
            time_scale=self._spin_timescale.value(),
            prior_mean=self._selected_prior_mean(),
            seed_observations=tuple(self._seed_observations),
            electrode_capacity=(
                self._spin_capacity.value() if self._chk_board.isChecked() else None
            ),
            equilibration_s=self._spin_equil.value(),
        )

    # ── Config save / load (UI-state JSON; CampaignSpec holds a non-serialisable
    #    prior_mean callable, so we persist the panel state rather than the spec) ─

    def _panel_state(self) -> dict[str, Any]:
        rows = []
        for row in range(self._tbl_params.rowCount()):
            n_item = self._tbl_params.item(row, 0)
            if n_item is None or not n_item.text().strip():
                continue
            rows.append([
                n_item.text().strip(),
                self._tbl_params.item(row, 1).text(),
                self._tbl_params.item(row, 2).text(),
            ])
        return {
            "name": self._le_name.text().strip(),
            "channels": self._le_channels.text().strip(),
            "parameters": rows,
            "search_mode": self._search_mode(),
            "composition_axes": self._axes_editor.to_state(),
            "deposit_uL": self._spin_dep_uL.value(),
            "objective": self._combo_objdir.currentText(),
            "acquisition": self._combo_acq.currentText(),
            "batch": self._chk_batch.isChecked(),
            "batch_strategy": self._batch_strategy_name(),
            "budget": self._spin_budget.value(),
            "seed": self._spin_seed.value(),
            "kappa": self._spin_kappa.value(),
            "time_scale": self._spin_timescale.value(),
            "pcb": self._combo_pcb.currentData(),
            "board_exchange": self._chk_board.isChecked(),
            "board_capacity": self._spin_capacity.value(),
            "equilibration_s": self._spin_equil.value(),
            "prior_mean": self._combo_prior.currentText(),
            "seed_observations": [[dict(p), float(v)] for p, v in self._seed_observations],
        }

    def _on_save_config(self) -> None:
        try:
            state = self._panel_state()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Config Error", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {self._CONFIG_TITLE}", "",
            f"{self._CONFIG_TITLE} (*.json);;All files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _config_from_json(self, text: str) -> dict[str, Any]:
        return json.loads(text)

    def _populate_from_config(self, state: dict[str, Any]) -> None:
        self._le_name.setText(str(state.get("name", "live_bo")))
        self._le_channels.setText(str(state.get("channels", "1")))
        rows = state.get("parameters", [])
        self._tbl_params.setRowCount(0)
        for row in rows:
            try:
                self._add_param_row(str(row[0]), float(row[1]), float(row[2]))
            except (IndexError, ValueError, TypeError):
                continue
        self._axes_editor.from_state(state.get("composition_axes"))
        try:
            self._spin_dep_uL.setValue(float(state.get("deposit_uL", 6.0)))
        except (TypeError, ValueError):
            pass
        self._combo_search_mode.setCurrentIndex(
            1 if str(state.get("search_mode", "volumes")) == "composition" else 0)
        self._on_search_mode_changed()
        self._combo_objdir.setCurrentText(str(state.get("objective", "auto")))
        self._combo_acq.setCurrentText(str(state.get("acquisition", "ucb")))
        self._chk_batch.setChecked(bool(state.get("batch", False)))
        saved_strategy = str(state.get("batch_strategy", "constant_liar"))
        for i in range(self._combo_batch_strategy.count()):
            if self._combo_batch_strategy.itemText(i).startswith(saved_strategy):
                self._combo_batch_strategy.setCurrentIndex(i)
                break
        self._combo_batch_strategy.setEnabled(self._chk_batch.isChecked())
        self._spin_budget.setValue(int(state.get("budget", 8)))
        self._spin_seed.setValue(int(state.get("seed", 42)))
        self._spin_kappa.setValue(float(state.get("kappa", 2.0)))
        self._spin_timescale.setValue(float(state.get("time_scale", 0.05)))
        saved_pcb = state.get("pcb")
        if saved_pcb:
            _pi = self._combo_pcb.findData(saved_pcb)
            if _pi >= 0:
                self._combo_pcb.setCurrentIndex(_pi)
        self._chk_board.setChecked(bool(state.get("board_exchange", True)))
        self._spin_capacity.setValue(int(state.get("board_capacity", self._pcb_electrode_count())))
        self._spin_equil.setValue(float(state.get("equilibration_s", 60.0)))
        self._on_board_toggle(self._chk_board.isChecked())
        self._combo_prior.setCurrentText(str(state.get("prior_mean", "none")))
        seeds = state.get("seed_observations", [])
        self._seed_observations = [
            (dict(p), float(v)) for p, v in seeds
        ]
        self._lbl_seeds.setText(f"{len(self._seed_observations)} seed observations")

    # ── Seed observations ────────────────────────────────────────────────────

    def _on_load_seeds(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Seed Observations", "", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            seeds = [(dict(p), float(v)) for p, v in raw]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.critical(self, "Load Error", f"Could not load seeds:\n{exc}")
            return
        self._seed_observations = seeds
        self._lbl_seeds.setText(f"{len(seeds)} seed observations")

    def load_seeds_from_file(self, path: str) -> int:
        """Load seed observations from *path* (testable, no dialog). Returns count."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._seed_observations = [(dict(p), float(v)) for p, v in raw]
        self._lbl_seeds.setText(f"{len(self._seed_observations)} seed observations")
        return len(self._seed_observations)

    # ── run ──────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        # The spec is built first so a refused launch has something to preserve,
        # and so a malformed panel is reported without first asking the operator
        # about the head.
        try:
            spec = self._build_config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Campaign Error", str(exc))
            return

        # Single occupancy, checked **before** the head gate: that gate prompts
        # the operator and can issue a safety retract, and a launch that is
        # about to be refused must ask for nothing and move nothing.
        if self._refuse_if_rig_busy(spec):
            return

        # Head-position start-gate (the autonomous loop drives the head via
        # conditional commands, so its belief must match reality first).
        if not self._verify_head_position():
            return

        if not self._preflight_overflow_ok(spec):
            return

        # Projected duration + stock runway (P5.2). Runs after the overflow scan
        # because an infeasible search space makes the projection moot.
        if not self._preflight_projection_ok(spec):
            return

        # Resolve the metric and its direction here, where a contradiction can be
        # shown to the operator, rather than letting it surface as a worker-thread
        # traceback. `spec.objective` is "auto" by default, so reading it directly
        # would silently track the best trial as if the campaign were minimising.
        try:
            direction, metric = resolve_direction(spec)
        except CampaignError as exc:
            QMessageBox.warning(self, "Objective Direction", str(exc))
            return

        self._result = None
        self._runner = None
        self._maximize = direction == "maximize"
        self._best_obj = None
        self._reset_convergence()
        self._reset_scatter(list(spec.resolved_vol_params())[:2])
        self._log.clear()
        self._progress.setRange(0, spec.budget)
        self._progress.setValue(0)
        self._lbl_status.setText(f"Running — budget {spec.budget}")
        self._sig_log.emit(
            f"Live campaign '{spec.name}' — {len(spec.parameter_space)} params, "
            f"channels {spec.channels}, budget {spec.budget}"
            + (f", {len(spec.seed_observations)} seed obs" if spec.seed_observations else "")
            + (", prior=linear(demo)" if spec.prior_mean is not None else "")
        )
        self._sig_log.emit(
            f"  objective: {metric} ({direction})"
            + ("" if spec.objective in ("", "auto") else "  [pinned]")
        )
        self._start_worker(self._run_campaign, spec, name="live-bo")

    def _run_campaign(self, spec: CampaignSpec) -> None:
        """Worker thread: run the campaign through the shared harness."""
        self._result = self._execute_campaign(
            spec, on_event=self._on_campaign_event, aborted_exc=_LiveAborted,
        )

    def _on_campaign_event(self, evt: dict[str, Any]) -> None:
        """``on_event`` callback — runs on the WORKER thread.

        Marshals to the GUI thread via the base signals only; never touches
        widgets directly.  A cooperative abort unwinds the loop at the next
        suggestion boundary.
        """
        etype = evt.get("type")
        if etype == "suggestion":
            if self._abort_requested:
                raise _LiveAborted()
            self._sig_log.emit(
                f"  [{evt.get('iteration')}] suggest {self._fmt_params(evt.get('params', {}))}"
            )
        elif etype == "result":
            it = int(evt.get("iteration", 0))
            params = evt.get("params", {})
            obj = float(evt.get("objective", 0.0))
            if self._best_obj is None or (
                obj > self._best_obj if self._maximize else obj < self._best_obj
            ):
                self._best_obj = obj
            self._sig_step.emit(it, float(self._best_obj), obj)
            self._sig_point.emit(dict(params), obj)
            self._sig_log.emit(f"  [{it}] objective {obj:.4g}")
        elif etype == "run_started":
            self._sig_log.emit(f"▶ run {evt.get('run_id')} started")
        elif etype == "warm_start":
            self._sig_log.emit(f"  warm-start: {evt.get('n_seed')} seed observation(s)")
        elif etype == "batch_mode":
            self._sig_log.emit(
                f"  batch mode: q={evt.get('q')} across channels {evt.get('channels')}"
            )
        elif etype == "board_check":
            self._sig_log.emit(
                f"  board #{evt.get('board_id')} occupancy: "
                f"{len(evt.get('occupied', []))} used → {evt.get('decision')}"
            )
        elif etype == "electrode_mode":
            self._sig_log.emit(
                f"  electrodes: board #{evt.get('board_id')} from #{evt.get('start')} "
                f"(capacity {evt.get('capacity')})"
            )
        elif etype == "maturity_warning":
            self._sig_log.emit(
                f"  ⚠ maturity: {evt.get('method', '?')} below {evt.get('expected', '?')}"
            )
        elif etype == "state":
            self._sig_log.emit(f"  state {evt.get('old')} → {evt.get('new')}")
        elif etype == "converged":
            self._sig_log.emit(
                f"  ✓ converged at trial {evt.get('iteration')} (best {evt.get('best')})"
            )
        elif etype == "run_finished":
            self._sig_log.emit(f"■ run finished after {evt.get('n_trials')} trials")

    @staticmethod
    def _fmt_params(params: dict[str, Any]) -> str:
        return ", ".join(f"{k}={float(v):g}" for k, v in params.items())

    # ── Suggested-points scatter ─────────────────────────────────────────────

    def _reset_scatter(self, axes: list[str]) -> None:
        self._scatter_axes = tuple(axes)
        self._scatter_x = []
        self._scatter_y = []
        self._scatter_c = []
        self._scatter_dirty = False
        self._ax_sc.cla()
        self._canvas_sc.draw_idle()

    def _on_live_point(self, params: dict[str, Any], obj: float) -> None:
        if len(self._scatter_axes) < 2:
            return
        xa, ya = self._scatter_axes[0], self._scatter_axes[1]
        self._scatter_x.append(float(params.get(xa, float("nan"))))
        self._scatter_y.append(float(params.get(ya, float("nan"))))
        self._scatter_c.append(float(obj))
        if not self._scatter_dirty:
            self._scatter_dirty = True
            QTimer.singleShot(0, self._flush_scatter)

    def _flush_scatter(self) -> None:
        if not self._scatter_dirty:
            return
        self._scatter_dirty = False
        if len(self._scatter_axes) < 2:
            return
        # Rebuild the whole figure so the colorbar never accumulates.
        self._fig_sc.clear()
        self._ax_sc = self._fig_sc.add_subplot(111)
        sc = self._ax_sc.scatter(
            self._scatter_x, self._scatter_y, c=self._scatter_c,
            cmap="viridis", s=48, edgecolors="black", linewidths=0.5,
        )
        self._ax_sc.set_xlabel(self._scatter_axes[0])
        self._ax_sc.set_ylabel(self._scatter_axes[1])
        self._ax_sc.set_title("Evaluated points (color = objective)")
        try:
            self._fig_sc.colorbar(sc, ax=self._ax_sc, label="objective")
        except Exception:
            pass
        self._canvas_sc.draw_idle()
