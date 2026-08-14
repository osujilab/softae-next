# SoftAE — Soft-matter Autonomous Experimentation

Welcome to the SoftAE documentation.

SoftAE is a Python-based platform for autonomous high-throughput experimentation on soft-matter systems. It provides:

- **Instrument orchestration** — unified control of stage, syringe pumps, temperature/humidity controllers, EIS potentiostats, and cameras
- **Workflow engine** — YAML-defined experiment protocols with DAG-based parallel execution
- **PySide6 GUI** — 13-tab interface: initialization, liquid modeling, manual control, monitoring, HT experiment design, Arrhenius sweep, analysis, BO simulator, live BO campaigns, catalogs, the deposition twin, and Process Studio
- **Autonomous campaigns** — closed-loop suggest → cast → measure → tell, with crash/resume checkpoints, single-use electrode tracking, board-exchange handling, consumables projection and fault parking
- **EIS analysis pipeline** — circuit fitting, conductivity extraction, admission gates, per-sample cell constants, and SQLite storage
- **Fixture commissioning** — measured instrument envelope and fixture constants, persisted and reused across campaigns
- **Piezo integration** — protocol-backed piezo driver (`piezo` instrument), Manual tab controls, and optional HT liquid-event workflow hooks

## Quick Start

```bash
pip install -e ".[dev]"
softae-gui          # Launch GUI
softae-run --mock workflows/standard_eis_sweep.yaml  # Run workflow in mock mode
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [User Guide](USER_GUIDE.md) | Installation, configuration, tab reference, CLI usage |
| [Architecture](architecture.md) | Package layout, module dependencies, and the seams between them |
| [API Reference](api/core/formulation.md) | Auto-generated Python API documentation |

Working documents — the roadmap, the session-by-session progress log, the development
fronts and the per-task specs — are deliberately **not** published. They live under
`docs/` in the working tree and are excluded by `.gitignore`, because they record
in-flight decisions rather than the state of the system. The three pages above are the
committed documentation surface, and `mkdocs.yml`'s nav is limited to the same set.

Runnable examples ship as files rather than pages: workflow YAML under `workflows/`
(including `workflows/examples/piezo_assisted_dispense.yaml` for piezo-assisted
dispense), and Python demos under `examples/`.
