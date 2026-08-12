# Architecture

This page provides a compact structural view of `softae-next` module relationships and workflow execution control flow.

## Module Dependency Diagram

```mermaid
graph TD
    CFG[config.loader]
    ERR[errors]
    SRV[server.manager]
    DRV[drivers.*]
    PIEZO_DRV[drivers.async_piezo / drivers.mock_piezo]
    PIEZO_PROTO[core.piezo_protocol]
    CORE[core.data_store]
    FORM[core.formulation]
    DEP[core.deposition]
    WF_MODEL[workflows.workflow_model]
    WF_PARSE[workflows.workflow_parser]
    WF_EXEC[workflows.workflow_executor]
    LOOP[core.autonomous_loop]
    OPT[optimizers.*]
    ANL_EIS[analysis.eis_data]
    ANL_FIT[analysis.circuit_fitting]
    GUI_APP[gui.app]
    GUI_MAIN[gui.main_window]
    GUI_TABS[gui.tabs.*]
    GUI_WIDGETS[gui.widgets.*]
    DEP_APP[gui.deposition_app]
    DEP_PANEL[gui.widgets.deposition_panel]
    DEP_FRAC[gui.widgets.deposition_fractions]
    FORM_PANEL[gui.widgets.formulation_panel]
    FORM_IO[gui.widgets.formulation_io]
    CAT_BASE[gui.widgets.catalog_editor_base]
    CAT_MGR[gui.widgets.catalog_manager]
    CAT_BROWSE[gui.widgets.catalog_browser]
    WORKER[gui.widgets.worker_thread]
    DAEMON[gui.daemon_runner]
    DATA_CSV[(data/*.csv catalogs)]

    CFG --> DRV
    CFG --> PIEZO_DRV
    CFG --> GUI_APP
    ERR --> SRV
    ERR --> DRV
    PIEZO_PROTO --> PIEZO_DRV
    ERR --> WF_EXEC
    SRV --> DRV
    SRV --> PIEZO_DRV

    WF_PARSE --> WF_MODEL
    WF_EXEC --> WF_MODEL
    WF_EXEC --> SRV
    WF_EXEC --> CORE
    WF_EXEC --> PIEZO_DRV
    WF_EXEC --> ANL_EIS
    WF_EXEC --> ANL_FIT

    FORM -- ElutionResult --> DEP

    LOOP --> OPT
    LOOP --> WF_EXEC
    LOOP --> CORE

    GUI_APP --> SRV
    GUI_APP --> GUI_MAIN
    GUI_MAIN --> GUI_TABS
    GUI_MAIN --> GUI_WIDGETS
    GUI_TABS --> SRV
    GUI_TABS --> CORE
    GUI_TABS --> WF_EXEC
    GUI_TABS --> ANL_EIS
    GUI_TABS --> ANL_FIT
    GUI_WIDGETS --> FORM

    DEP_APP --> DEP_PANEL
    DEP_FRAC -. helper .-> DEP_PANEL
    DEP_PANEL --> FORM
    DEP_PANEL --> DEP

    CFG -- data_root --> FORM_PANEL
    GUI_TABS -- catalog edit --> FORM_PANEL
    FORM_IO -. helper .-> FORM_PANEL
    CAT_BASE -- mixin --> FORM_PANEL
    CAT_BASE -- mixin --> CAT_MGR
    FORM_IO -. helper .-> CAT_BASE
    CAT_MGR --> FORM
    CAT_MGR -- save --> DATA_CSV
    FORM_PANEL --> FORM
    FORM_PANEL -- save --> DATA_CSV
    CFG -- data_root --> DEP_APP
    DATA_CSV -- load --> DEP_APP

    GUI_MAIN -- "11. Catalogs tab" --> CAT_BROWSE
    GUI_MAIN -- "12. Deposition tab" --> DEP_PANEL
    GUI_MAIN -- "Catalogs menu / Edit" --> CAT_MGR
    CAT_BROWSE -- reload --> DATA_CSV
    CAT_MGR -- catalogs_changed --> GUI_MAIN

    WORKER -- base of pollers --> GUI_WIDGETS
    DAEMON -- mixin --> GUI_TABS
```

## Workflow Execution Flowchart (DAG-Tier Executor)

```mermaid
flowchart TD
    A[Load Workflow YAML or JSON] --> B[Parse into Workflow model]
    B --> C[Build DAG from steps and dependencies]
    C --> D[Topologically group into tiers]
    D --> E[For each tier: dispatch eligible steps]
    E --> F{Step success?}
    F -->|Yes| G[Record result and provenance]
    F -->|No| H{Retries left?}
    H -->|Yes| I[Retry step with timeout]
    I --> F
    H -->|No| J[Mark failed and apply failure policy]
    G --> K{More tiers?}
    J --> K
    K -->|Yes| E
    K -->|No| L[Run teardown and finalize run]
    L --> M[Optional EIS auto-route to DataStore]
    M --> N[Emit completion state]
```

## Notes

- The workflow executor is async and tier-based for parallelizable steps.
- GUI tabs trigger execution through shared manager/executor plumbing and consume signal-based updates.
- Autonomous logic composes optimizer suggestions with workflow execution and DataStore feedback.
- Piezo integration uses `config.loader.piezo_config()` defaults merged from `[piezo]` and `[piezo.liquid_events]`.
- Manual tab sends direct piezo calls through the manager (`set_channel`, `apply_profile`, `standby`) via non-blocking command workers.
- HT Experiment injects optional piezo workflow steps around liquid handling when piezo liquid events are enabled; `settings_source` selects manual profile reuse vs explicit event profile apply.
- Worker-thread lifecycle: every polling worker QThread exposes an idempotent `stop_worker()`, each thread-owning tab an idempotent `cleanup()`, and `MainWindow.closeEvent` stops all workers (tab `cleanup()`s → own `_poller`/`_cam_worker`/`_webcam_worker` → defensive `findChildren(QThread)` sweep) so closing the window leaves no running QThread. The `stop_worker()` contract is now codified in a shared `StoppableWorker(QThread)` base (`gui/widgets/worker_thread.py`): a template `stop_worker(timeout_ms)` (isRunning guard → `_request_stop()` → `wait()`) that the 9 polling workers inherit, replacing their bespoke copies (per-worker timeouts preserved). The transient analysis fit workers (`_FitAllWorker`/`_ArrhFitWorker`) are parented to the Analysis tab so the `closeEvent` sweep reaches them too.
- Daemon-runner cooperative abort: the tabs that run long operations on `threading.Thread(daemon=True)` (`ExperimentBuilderTab`, Process Studio's embedded builder, `BOSimulatorTab`, `LiveBOCampaignTab`, `ArrheniusTab` — the standalone `SandboxTab` having been retired into Process Studio, and the single BO tab having split into an offline simulator and a live campaign) each add an idempotent `abort_run()` (signal-only, reusing the tab's Stop-button abort path) + `cleanup()` (abort-then-bounded-join with shared `DAEMON_JOIN_TIMEOUT` in `gui/_shutdown.py`). This contract is codified in a `DaemonRunnerMixin` (`gui/daemon_runner.py`, plain-object mixin) that implements `abort_run()`/`cleanup()` over per-tab hooks `_abort_run_impl()`/`_runner_thread()`. `MainWindow.closeEvent` signals `abort_run()` on all four **first** (so hardware sweeps stop issuing commands ASAP and wind down in parallel), then joins each via `cleanup()` — so closing mid-run cooperatively aborts the in-progress experiment/campaign/sweep instead of leaving it issuing commands.
