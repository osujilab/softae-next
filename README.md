# SoftAE — Soft-matter Autonomous Experimentation

Next-generation platform for autonomous (self-driving) materials science experiments, with automated/combinatorial modalities as a manual exploratory fallback.

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

## Project Structure

```
softae-next/
├── pyproject.toml              # Package metadata & dependencies
├── softae_config.toml          # Hardware addresses, paths, defaults
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
