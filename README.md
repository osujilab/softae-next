# SoftAE — Soft-matter Autonomous Experimentation
Next-generation benchtop platform for autonomous (self-driving) soft materials science experiments.

Authored by: 
Pavel Shapturenka,
Christopher W. Johnson,
Yvonne Zagzag,
Justin Hughes,
Minki Lee,
 *In collaboration and with assistance from frontier agentic large-language models.*

**Osuji lab**, Department of Chemical and Biomolecular Engineering, University of Pennsylvania

SoftAE functionality integrates stage motion, liquid dispensing, environmental control (T/RH%), polarized optical microscopy, and conductivity measurements to afford high-throughput soft material formulation and inspection. High-throughput methods are in turn extended to autonomous experimentation via in-loop implementation of algorithms such as Bayesian optimization and Gaussian process-informed parameter phase space exploration.

Core system functionality is accessible through a central graphical interface and is operable with various degrees of autonomy.
The design intent is to be transparent in an instructive and functional manner. The codebase addresses many facets of a fully functional self-driving laboratory: orchestration, data analysis, algorithmic machinery, safety, reproducibility, telemetry, and data provenance.

![Project Screenshot](images/SoftAE_schema.png)

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell

# Install in editable mode (no hardware dependencies)
pip install -e ".[dev]"

# Launch the GUI (runs with mock instruments by default)
softae-gui
```

This repository ships **no process catalog**: `data/` — `tasks.toml`, `recipes.toml` and the
chemicals/solutions CSVs — is gitignored by policy, because each new instance of this system
needs users to develop their own processes, recipes and workflows that best reflect the
integrated hardware. A fresh checkout therefore names no tasks and is not green for the
catalog-loading tests by construction — see the User Guide's *Before you start*.

`softae-catalog init` writes a **placeholder** catalog into `data/` so a clone has something to
compile the specs under `examples/` against, and so each file's shape is visible. It is a
demonstration, not a configuration: generic chemistry and illustrative numbers, no ports, no
calibrated volumes, no cure recipe belonging to any rig, and **no guarantee it works on your
hardware as shipped**. It refuses to overwrite an existing catalog. Copy it, then edit your own
copy under `data/`.

Build or edit a catalog in the GUI's *Process Studio* (Tab 13, the TOMLs) and *Catalogs*
(Tab 11, the CSVs), or by hand against the schemas in `src/softae/core/task_catalog.py`,
`recipe_registry.py` and `formulation.py`. A missing catalog file loads as an *empty* catalog
rather than raising, so the absence surfaces at the first task name a run asks for — the User
Guide's *Before you start* names the routes and the test that demonstrates it.

## Project Structure

```
softae-next/
├── pyproject.toml              # Package metadata & dependencies
├── softae_config.toml          # Hardware addresses, paths, defaults
├── data/                       # Process catalog (tasks, recipes) — GITIGNORED, per-instance
├── src/
│   └── softae/
│       ├── config/             # Configuration loader
│       ├── drivers/            # Instrument driver wrappers (Phase 0 refactored)
│       ├── server/             # InstrumentManager, BaseInstrument ABC
│       ├── workflows/          # Workflow engine (Phase 2)
│       ├── analysis/           # EIS analysis, fitting (Phase 2)
│       ├── optimizers/         # Bayesian, grid search (Phase 4)
│       └── gui/                # PySide6 multi-tab GUI (Phase 3)
│           ├── tabs/           # One module per tab
│           └── widgets/        # Reusable UI components
├── tests/                      # pytest test suite
├── workflows/                  # YAML workflow templates
└── docs/                       # Documentation
```

## Development Phases

- **Phase 0**: Code hygiene — eliminate globals, fix imports, add context managers
- **Phase 1**: Instrument server — BaseInstrument ABC, InstrumentManager, async drivers
- **Phase 2**: Workflow engine — YAML-defined experiments, structured logging
- **Phase 3**: Multi-tab GUI — PySide6 desktop application
- **Phase 4**: Autonomous loop — Bayesian optimization, closed-loop experiments
