# SoftAE User Guide

> Soft-matter Autonomous Experimentation Platform  
> Version 0.1.0 · Python ≥ 3.11 · PySide6

---

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Launching the GUI](#3-launching-the-gui)
4. [Tab Reference](#4-tab-reference)
5. [CLI Workflow Runner](#5-cli-workflow-runner)
6. [Writing Workflow YAML Files](#6-writing-workflow-yaml-files)
7. [EIS Analysis Pipeline](#7-eis-analysis-pipeline)
8. [Emergency Stop & Safe Exit](#8-emergency-stop--safe-exit)
9. [Error Reference](#9-error-reference)
10. [Instruments Reference](#10-instruments-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Documentation Site](#12-documentation-site)
13. [Data Persistence (DataStore)](#13-data-persistence-datastore)
14. [Deposition Digital Twin](#14-deposition-digital-twin)
15. [Autonomous Campaigns](#15-autonomous-campaigns)
16. [EIS Commissioning & Calibration](#16-eis-commissioning--calibration)
17. [EIS Analysis Engine & Gates](#17-eis-analysis-engine--gates)
18. [Unattended Operation & Safety](#18-unattended-operation--safety)
19. [Extending: a New Measurement Modality](#19-extending-a-new-measurement-modality)
20. [Shadow Campaign Review](#20-shadow-campaign-review)
21. [Thickness Series](#21-thickness-series)
22. [Equilibration Characterization](#22-equilibration-characterization)

---

## 1. Installation

### Prerequisites
- Python 3.11+
- Git
- (Optional) NI-DAQmx runtime, ThorLabs TSI SDK, PalmSens SDK for hardware

### Install from source

```powershell
cd softae-next
python -m venv .venv
.venv\Scripts\activate

pip install -e .              # core dependencies
pip install -e ".[dev]"       # + test/lint tools
pip install -e ".[hardware]"  # + NI-DAQ, Blinka, HID for real instruments
```

### Verify installation

```powershell
softae-gui --help     # GUI entry point
softae-run --help     # CLI workflow runner
pytest tests/ -v      # run test suite
```

---

## 2. Configuration

All instrument addresses, safety limits, EIS presets, and PCB layouts are defined in `softae_config.toml` at the repository root.

### Config Lookup Chain

The config loader searches in this order:
1. Explicit path passed to `config.load(path=...)`
2. `SOFTAE_CONFIG` environment variable
3. `softae_config.toml` in the current working directory
4. `softae_config.toml` in the package install root

### Key Sections

| Section | Purpose | Example Keys |
|---|---|---|
| `[paths]` | SDK / DLL locations | `thorlabs_dll`, `data_root` |
| `[instruments.*]` | Per-instrument connection details | port, baud, address |
| `[pcb.*]` | Printed circuit board layouts | channels, grid, spacing, electrode_dims |
| `[eis_presets.*]` | EIS measurement presets | npts, f_hi, f_lo, mv_ac |
| `[channel_routing]` | Channel → potentiostat mapping | `pico1_range = [1, 16]` |
| `[piezo]` | Piezo driver defaults and manual profile | `enabled`, `frequency_hz`, `sweep_on_s` |
| `[piezo.liquid_events]` | Optional event-driven piezo behavior for HT workflows | `enabled`, `settings_source`, `channel_a` |
| `[safety]` | Operational limits, gate timeouts, anneal watchdog bands | `temp_max_C`, `pump_rate_max`, `step_timeout_s`, `anneal_temp_band_C` |
| `[deposition]` | Drop-cast engine defaults | `evaporation_pct`, recipe defaults |
| `[dropcast]` | Two-phase cast rates and dwell | precondition flush, proportional rate |
| `[liquid_handling]` | Optional per-line volume correction | `enabled`, `beta`, `eta_ref_mpas` |
| `[quality]` | Measurement accept/suspect/reject grading | `enabled`, `max_residual_pct`, `max_abs_z` |
| `[purge]` | Anti-clog purge harness | `actuate`, cadence, particulate pump |
| `[eis]` | Analysis engine + campaign objective | `engine`, `objective` |
| `[eis.gates]` | Admission-gate thresholds | `enabled`, `min_fit_pts`, `rho_degenerate` |
| `[eis.instrument]` | **Measured** instrument envelope | `phase_noise_deg`, `z_max_ohm`, `max_amplitude_mV` |
| `[eis.cell]` | Cell geometry + electrode configuration | `L_gap_cm`, `dead_height_um`, `k_config_verified` |
| `[eis.fixture]` | Fixture correction (series-only) | `mode`, `fixture_id`, `load_tolerance_pct` |
| `[stage_calibration]` | Saved stage origin / skew | persisted by Tab 1 |
| `[logging]`, `[web]`, `[webcam]` | Log level and optional web/camera services | `level`, `port` |

Three of these ship **deliberately disabled** — `[quality] enabled`, `[purge] actuate`, and
`[eis.gates] enabled`. Each governs a mechanism that removes or reshapes data, and each ships
inert until its thresholds have been reviewed against real runs on *this* rig. See
[§17](#17-eis-analysis-engine--gates) and [§18](#18-unattended-operation--safety).

### Instruments Configured

| Config Key | Instrument | Default Port |
|---|---|---|
| `instruments.stage` | Newport ESP301 linear stage | `ASRL7::INSTR` |
| `instruments.syringe` | Harvard Apparatus syringe pump | `ASRL4::INSTR` |
| `instruments.temp_controller` | Novus N1040 temperature controller | `com6` |
| `instruments.pico1` / `pico2` | PalmSens EmStat Pico (×2) | `auto` |
| `instruments.piezo` | Trinket piezo controller | `COM16` |
| `instruments.camera` | ThorLabs Zelux camera | (SDK discovery) |

> **Hardware note — dual Pico port assignment:**  
> When `port = "auto"`, `pico1` binds to the first enumerated EmStat Pico (`ports[0]`) and `pico2` binds to the second (`ports[1]`). This relies on the OS enumerating COM ports in stable ascending order between reboots. If your two Picos are assigned COM port numbers inconsistently, channel routing will be wrong (channels 17–32 will run on the wrong device). To make the mapping deterministic, assign explicit ports in `softae_config.toml`:
>
> ```toml
> [instruments.pico1]
> port = "COM5"
>
> [instruments.pico2]
> port = "COM7"
> ```
| `instruments.lamp` | MCP4728 quad-DAC lamp (I²C via MCP2221) | channel `A` @ `0x60` |
| `instruments.keithley` | Keithley 2700 DMM | USB VISA address |
| `instruments.ht_sensor` | SHT31-D humidity/temp sensor | MCP2221 HID |
| `instruments.rh_controller` | Trinket M0 RH PID controller | `COM11` |

### EIS Presets

| Preset | Points | Freq Range | AC Amplitude |
|---|---|---|---|
| Standard | 35 | 200 kHz – 4 Hz | 10 mV |
| Quick | 25 | 200 kHz – 20 Hz | 10 mV |
| Extended | 45 | 200 kHz – 1.2 Hz | 10 mV |
| Longest | 35 | 200 kHz – 200 mHz | 10 mV |

### Safety Limits

| Parameter | Limit |
|---|---|
| Temperature | 5 – 200 °C |
| Pump rate | 0.05 – 2120 µL/min |
| Reservoir warning | 500 µL |
| Step timeout | 900 s |

### Piezo Configuration (optional)

Piezo control is disabled by default. Enable it explicitly in `softae_config.toml`.

```toml
[instruments.piezo]
driver  = "piezo"
port    = "COM16"
baud    = 115200
enabled = false

[piezo]
enabled      = false
channel      = "A"
frequency_hz = 500
sweep_on_s   = 2.0
sweep_rest_s = 3.0

[piezo.liquid_events]
enabled         = false
settings_source = "manual_profile"  # manual_profile | liquid_event_profile
channel_a       = true
frequency_hz    = 500
sweep_on_s      = 2.0
sweep_rest_s    = 3.0
```

Notes:
- Event profile values are validated to match protocol limits: frequency `10..5000` Hz and sweep timings `0.01..120.0` s.
- CFG commands require firmware capability support (`CAPS PIEZO_CFG_V1`); legacy firmware still supports channel on/off.
- Real hardware requires serial dependencies (`pyserial`) and a reachable configured COM port.

---

## 3. Launching the GUI

```powershell
# Standard (auto-detects real hardware, falls back to mocks)
softae-gui

# Force mock instruments (no hardware needed)
python -m softae.gui    # equivalent
```

The GUI opens a window titled **"SoftAE — Soft-matter Autonomous Experimentation"** (1200×800 minimum) with 9 tabs and a persistent emergency stop button in the toolbar.

### Mock vs Real Mode

| Mode | Behavior |
|---|---|
| Auto-detect (default) | Tries each real driver; falls back to mock per instrument |
| Mock (`mock=True`) | All 10 instruments use simulated drivers |
| Real (`mock=False`) | Demands real hardware; raises error if unavailable |

---

## 4. Tab Reference

### Tab 1: Init & Calibration

**Purpose:** Connect instruments, calibrate the stage, select the PCB layout.

| Section | What You Can Do |
|---|---|
| Instrument Table | View all 10 instruments with name, type, state (green/red), and details. Auto-refreshes every 2 s. |
| Connect / Disconnect | "Connect All", "Disconnect All", or select a row and use "Connect Selected" / "Disconnect Selected" |
| Stage Calibration | Set Home and Dep-1 (deposition start) coordinates. "Set Current →" captures live position. "Go Home" / "Go Dep-1" moves the stage (dispatched to a background thread; button disabled until complete). |
| Syringe Config | **Syringe count controls only.** Set the per-pump `parallel_syringes` count (1 or 2) and click **Apply + Save**. This persists to `softae_config.toml`, is applied to the active syringe driver, and propagates to Manual Control readouts via polling (~2 s interval). |
| PCB Selector | Dropdown of PCB layouts from config. Shows channel count, grid, spacing, electrode geometry. |
| Position Map | Interactive scatter plot showing electrode positions from the active PCB layout. Click an electrode to move the stage to that position. Current position is polled in a background thread and highlighted in real time. |

### Tab 2: Liquid Model

**Purpose:** View and refine liquid-handling correction parameters in a dedicated workspace.

**Display & Controls:**
- **System Parameters:** Toggle correction on/off and edit shared model values (`beta`, `eta_ref_mpas`, `alpha_growth_per_run`)
- **Three Line Panels (0/1/2):** Each line has editable `cracking_kpa_per_valve`, `compliance_uL_per_kpa`, `alpha_base`, and `viscosity_mpas`
- **Prime Estimates:** Live estimated prime volume (`uL`) shown for each line as parameters change
- **Apply + Save:** Persists edits to `softae_config.toml` (`[liquid_handling]` and `[liquid_handling.line.<id>]`)

**Piezo Event Settings (same tab):**
- **Enable piezo during liquid-handling events:** maps to `[piezo.liquid_events].enabled`
- **Settings source:** `manual_profile` or `liquid_event_profile`
  - `manual_profile`: HT workflow events use the Manual tab profile already active on the device
  - `liquid_event_profile`: HT workflow injects one `piezo.apply_profile(...)` step before channel events
- **Use channel A for events:** maps to `[piezo.liquid_events].channel_a` (current workflow integration targets channel A)
- **Event freq / ON / REST:** enabled only when source is `liquid_event_profile`
- **Apply + Save:** also persists `[piezo.liquid_events]` via `save_piezo_config(...)`

The tab always renders lines 0, 1, and 2 even if config was previously sparse, so the UI remains aligned with the three syringe lines used by Manual Control.

### Tab 3: Manual Control

**Purpose:** Hands-on control of every instrument.

**Stage:**
- Enter X/Y coordinates and click "Go To", or use arrow buttons for jogging; commands dispatch to a background thread (button disabled during execution)
- Adjustable step size (0.01–50 mm)
- Live position display (polled every 2 s off the main thread)

**Temperature:**
- Set a target temperature (5–200 °C) → "Set" button
- Or configure a ramp: target + rate (°C/min) → "Start Ramp"
- **Anneal**: target temp + hold time (s), optional ramp rate and tolerance → "Anneal". Ramps (or jumps) to the target, holds for the specified duration, then automatically restores the original setpoint (guaranteed via `finally`).
- Live SP and PV readout

**Relative Humidity:**
- Set target RH (0–95%) → "Set" / "Start PID" / "Stop PID"
- Live RH readout

**Syringe Pumps:**
- 3 pump rows (Pump 0/1/2): set rate (µL/min) and volume (µL) → "Infuse"
- Per-row readout shows `Syringes loaded: N` from live per-pump syringe status (polled every ~2 s and auto-updated when Init counts change)
- `Apply liquid correction` toggle controls whether entered volume is corrected before dispatch
- Each pump’s command is divided by its own loaded syringe count before dispatch
- Last-command feedback shows target→commanded volume and correction state (`on`/`off`)
- Retract / Descend buttons for the pneumatic head
- Head status indicator (green = retracted, orange = descended)

**Piezo (Channel A):**
- `Channel A ON` / `Channel A OFF` sends `piezo.set_channel(channel="A", enabled=...)`
- Profile controls: `Freq (Hz)`, `ON (s)`, `REST (s)`
- `Apply Settings` sends `piezo.apply_profile(frequency_hz, on_s, rest_s)`
- If piezo is disabled in config (`[piezo].enabled = false`), controls are read-only and status indicates config-disabled state
- CFG-dependent profile updates require compatible firmware; channel on/off continues to work with legacy listener behavior

**Camera:**
- Exposure control (0.001–10 s)
- "Snap" for a single frame, "Live Preview" toggle for 1 FPS feed
- "Lamp On" / "Lamp Off" buttons
- 320×240 image display

**EIS Quick Run:**
- Select channel (1–32); pico is auto-routed from config (channels 1–16 → pico1, 17–32 → pico2)
- Choose a **preset** (Standard, Quick, Extended, Longest) to pre-populate the five editable parameter fields
- Adjust **f_hi (Hz)**, **f_lo (mHz)**, **npts**, **mVac**, and **mVdc** directly before running — changes are not persisted to the preset, they apply only to the current run
- "Run EIS" launches measurement on a background thread
- Optional auto-save to the project DataStore run directory and optional circuit fit overlay
- When fit succeeds, residual channels are computed and included in saved EIS text output columns
- Displays Nyquist + Bode popup on completion

### Tab 4: Monitoring

**Purpose:** Real-time dashboard for ongoing experiments.

| Widget | What It Shows |
|---|---|
| Temperature plot | Rolling 10-min time series (PV + SP overlay) |
| Humidity plot | Rolling 10-min time series (RH + SP overlay) |
| Numeric readouts | Temp PV, Temp SP, RH, RH SP, Stage X/Y |
| Camera feed | Passive 1 FPS display from shared camera worker |
| Webcam feed | USB webcam panel (OpenCV `VideoCapture`): exposure slider (-1 to -9), live timestamp overlay, click-drag zoom rectangle, single-click to reset zoom. Visually distinct from ThorLabs feed (separate QGroupBox, different border). |
| Workflow progress | Progress bar + step label (updated by Experiment/Sandbox tabs) |
| Instrument log | Scrolling text log (last 500 lines) |

### Tab 5: HT Experiment

**Purpose:** Build and run high-throughput multi-channel EIS experiments.

**Workflow:**
1. Select a **Workflow Mode**: "Full Protocol" (flush + deposit + EIS) or "Measure Only" (EIS only)
2. Choose a **PCB layout**; select an **EIS preset** to pre-populate the five editable EIS parameter fields
3. Optionally adjust **f_hi (Hz)**, **f_lo (mHz)**, **npts**, **mVac**, and **mVdc** for this run
4. Fill the **Formulation Matrix** — per-channel volumes for each pump
   - Click **"Formulation Manager..."** to open the formulation dialog: define stock solutions (solute, concentration, solvent), compute per-channel dispense volumes, and manage the chemical/solution catalogs.
4. Use checkboxes or the channel spec field (`1,3,5-8`) to select active channels
5. Click **Generate Workflow** to preview the step list
6. Click **▶ Start** to execute
7. Use **⏸ Pause** / **⏹ Abort** during execution
8. Results appear in the table as steps complete (Channel and Duration columns populated)
9. **Save CSV**, **Save EIS Data**, **Fit All EIS**, or **Save PDF Report** to export

Automatic pico routing: channels 1–16 → pico1, channels 17–32 → pico2 (configurable in `[channel_routing]`).

**Formulator + liquid-handling integration:**
- Volumes applied from **Formulation Manager** feed per-channel dispense steps in generated HT workflows (Pump 0 and Pump 1 are channel-specific; no fixed per-channel dispense constants).
- Physical liquid-handling correction is optional and controlled by `[liquid_handling]` with `enabled = true|false`.
- Pump-to-line mapping is configured in `[liquid_handling.pump_line]`; per-line physics are configured under `[liquid_handling.line.<id>]` (for example: `cracking_kpa_per_valve`, `compliance_uL_per_kpa`, `alpha_base`, `viscosity_mpas`) in `softae_config.toml`.
- Dispense panel shows `Liquid correction: Enabled|Disabled` for immediate correction-state visibility.
- The correction-model panel is editable, including prime-estimate values used to refine the correction fit.
- Workflow preview includes `liquid_correction: enabled|disabled`, prime-estimate lines, and per-channel target vs commanded dispense readouts (`p0`/`p1`) for selected channels.
- When correction is disabled, preview states `correction disabled: commanded == target`.

**Optional piezo liquid events during Full Protocol runs:**
- Event generation requires all of the following:
  - `[piezo].enabled = true`
  - `[piezo.liquid_events].enabled = true`
  - `[piezo.liquid_events].channel_a = true`
- Per selected channel, workflow inserts:
  - `piezo_on_chN` (`piezo.set_channel(A, true)`) before dispense/EIS block
  - `piezo_off_chN` (`piezo.set_channel(A, false)`) after channel block
  - `piezo_standby` (`piezo.standby()`) in teardown
- Profile source behavior:
  - `settings_source = "manual_profile"`: no event profile step is injected; device uses currently active manual profile
  - `settings_source = "liquid_event_profile"`: setup includes one `piezo_apply_event_profile` step calling `piezo.apply_profile(...)`
- Measure-only mode does not add piezo liquid-event steps.

### Tab 6: Arrhenius Sweep

**Purpose:** Automated temperature-dependent EIS to extract ionic conductivity σ(T) and compute Arrhenius activation energy.

**Workflow:**
1. Set **Temperature Profile** — T start, T stop, T step (°C) and dwell time after equilibration
2. Enter **channels** to measure (comma-separated, e.g. `1, 2, 4`)
3. Set **instrument names** for the EIS device and temperature controller (`pico1`, `temp_controller`)
4. Set **Electrode Geometry** (L, t, w in cm) for conductivity calculation — leaving blank will produce NaN σ values
5. Choose an **EIS preset** to pre-populate **f_hi**, **f_lo**, **npts**, **mVac**, **mVdc**; adjust as needed
6. Click **▶ Start Sweep** — per-channel `.mscr` files are built before execution
7. Live log shows each step with elapsed time; progress bar advances per step
8. After completion, the **Arrhenius Plot** panel shows ln(σ) vs 1/T with a linear fit
9. Extracted E_a (activation energy) and pre-exponential σ₀ are displayed
10. **Export CSV** to save per-temperature σ data and fit parameters

Automatic pico routing follows the same channel-mapping rules as the HT tab.

### Tab 7: Autonomous

**Purpose:** Placeholder scaffolding, kept deliberately.

> **Status:** This tab is **intentional scaffolding, not the working autonomous path.**
> Closed-loop campaigns run from **Tab 10: Live BO Campaign** (interactive) or
> `softae-campaign` (headless) — see [§15](#15-autonomous-campaigns). The tab is retained
> as a placeholder for a future multi-objective / Pareto front-end; it is not dead code and
> should not be removed.

### Tab 8: Analysis

**Purpose:** Post-experiment EIS data analysis and circuit fitting.  
The tab has two sub-tabs: **Fit & Export** (existing workflow) and **EIS Browser** (interactive visualizer).

#### Sub-tab: Fit & Export

**Workflow:**
1. Click **Load File(s)…** to open EIS data files (`.txt`, `.csv`, `.dat`)
2. Data appears on Nyquist + Bode plots
3. Select a **circuit model** and click **Fit All**
4. Set electrode geometry (L, t, w in cm) for conductivity calculation
5. Results table shows R₀, R₁, σ per channel
6. **Save to Database** (SQLite), **Browse Database**, or **Export CSV**

**Available Circuit Models:**

| Model | Circuit | Use For |
|---|---|---|
| `simpleSalt` | R₀-CPE₀-p(R₁,C₀) | Simple ionic conductors |
| `flexSalt` | R₀-CPE₀-p(R₁,C₀) with fixed C₀ | Salt solutions with known stray capacitance |
| `simpleSaltMembrane` | R₀-CPE₀-p(R₁-Wo₁,C₀) | Membranes / diffusion-limited systems |

#### Sub-tab: EIS Browser

Three-pane interactive viewer (Overview / Inspection / Conductivity) for browsing stored EIS measurements.

| Control | Action |
|---|---|
| **↻ Reload from DataStore** | Opens a DataStore browser dialog (run/channel/limit filters), then loads only selected rows into Browser or imports selected rows into Fit & Export |
| **⤢ Pop Out Window** | Detaches the viewer into a resizable standalone window |

Reload/import does **not** auto-fit selected spectra. Fitting remains explicit via **Fit All** in the **Fit & Export** sub-tab.

The viewer can also be launched independently from a notebook or script — see [SoftAE_ClassTests.ipynb](../../SoftAE_ClassTests.ipynb) (EIS Visualizer section) for runnable examples with synthetic, DataStore, and live-poll modes.

**Standalone usage (script / notebook):**
```python
from softae.gui.widgets.eis_visualizer_widget import EISVisualizerWindow, ListEISSource
EISVisualizerWindow.open(ListEISSource(entries))   # blocks until window is closed
```

### Tab 9: BO Simulator

**Purpose:** Offline Bayesian-optimization sandbox — no hardware, no instruments.

Runs a campaign against a **simulated** conductivity landscape so you can size a budget,
compare acquisition functions and batch strategies, and see how a prior changes convergence
before spending electrodes. Needs no `InstrumentManager` at all.

- Acquisition (`ucb` / `ei`) and κ, batch size and strategy, seed, budget
- Optional temperature axis, folded into an Arrhenius/VFT parameter as the objective
- σ map vs derived-objective map, convergence trace, suggested-point scatter
- Results export to JSON

### Tab 10: Live BO Campaign

**Purpose:** The hardware-in-the-loop optimizer — this is the working autonomous path.

**Search over** (the mode switch, top of the parameter panel):

| Mode | What is searched | Objective |
|---|---|---|
| **Raw volumes** | per-pump µL directly | mean \|Z\|, minimised |
| **Composition targets** | molar ratio / dried fraction / concentration, each Low→High | σ, maximised |

Raw volumes is the easier search — feasibility is native, since a volume limit is just a
bound — but the twin has no stock identity, so there is no dry thickness and therefore no
conductivity. Composition targets use the deposition twin's own target vocabulary and give
every trial a predicted thickness, which is what makes σ available. Stocks and pump
assignment come from the persisted **pump loadout**, so declare it in the Formulation
Manager first. A target row with `Low == High` is *pinned*: held constant and kept out of
the optimizer entirely.

**Direction** defaults to `auto` and should normally stay there — it is derived from the
metric, not chosen alongside it (see [§15](#15-autonomous-campaigns)).

Also on this tab: board-exchange controls and electrode capacity, seed observations
(warm start), an optional prior mean, a pre-run overflow scan, and a projected
duration + stock-runway preflight.

### Tab 11: Catalogs

Read-only browser over the chemical and solution catalogs, with an **Edit** button opening
the slim Catalog Manager. See [§14](#14-deposition-digital-twin) for the full catalog story.

### Tab 12: Deposition

The deposition digital twin embedded in the main window — "what ends up in the well?".
Fully documented in [§14](#14-deposition-digital-twin).

### Tab 13: Process Studio

**Purpose:** Method and recipe library plus an embedded workflow builder.

Supersedes the former standalone **Sandbox** tab, which was retired as a strict subset of
this one. Browse the task catalog and deposition recipes from `tasks.toml` / `recipes.toml`,
see each method's **maturity** level, and build/preview/run workflows against connected
instruments.

> **Maturity** is a warn-and-proceed guard, not a block. A campaign that would run a method
> below its expected maturity emits `method_below_maturity` and continues — surfacing the
> risk without stopping an experiment on a judgement call.

---

## 5. CLI Workflow Runner

### The command set

| Command | Purpose | Documented in |
|---|---|---|
| `softae-gui` | Launch the desktop application | [§3](#3-launching-the-gui) |
| `softae-run` | Execute a workflow YAML headlessly | this section |
| `softae-campaign` | Run / resume an autonomous campaign | [§15](#15-autonomous-campaigns) |
| `softae-commission` | Acquire and derive the EIS fixture calibration | [§16](#16-eis-commissioning--calibration) |
| `softae-deposition` | Standalone deposition-twin GUI | [§14](#14-deposition-digital-twin) |
| `softae-method` | Method-maturity lifecycle (`status`, `test`, `promote`, `sign-off`, `versions`) | `docs/METHOD_MATURITY_PIPELINE.md` |
| `softae-web` | EIS web visualizer over the DataStore | `python -m softae.web --help` |
| `softae-shadow` | Arm and review a shadow campaign | [§20](#20-shadow-campaign-review) |
| `softae-thickness` | Plan / record an unconfounded thickness series | [§21](#21-thickness-series) |
| `softae-equilibration` | Measure σ(t) and derive the conditioning hold | [§22](#22-equilibration-characterization) |

> **New commands need a reinstall.** Console scripts are generated at install time, so a
> newly added entry point resolves only after `pip install -e .`. A "command not recognized"
> error on a documented command is almost always this — re-run the editable install, or fall
> back to `python -m softae.tools.<name>`, which resolves whether or not a script was
> generated.

**All ten resolve in this venv, verified 2026-08-11.** `softae-shadow`, `softae-thickness` and
`softae-equilibration` were added after the previous editable install and were module-only
until it was refreshed; the refresh generated all three `.exe`s in `.venv/Scripts/`. Every
section below names the console script first, with the module form given as the exact
equivalent:

```bash
softae-shadow --help          # equivalently: python -m softae.tools.shadow_review
softae-thickness --help       #               python -m softae.tools.thickness
softae-equilibration --help   #               python -m softae.tools.equilibration
```

The module behind `softae-shadow` is **`shadow_review`**, not `shadow` — the one substitution
that is not mechanical. The arguments are identical either way.

Anything that drives **real motion hardware** additionally requires the interlock — see
[§18](#18-unattended-operation--safety).

### `softae-run`

The `softae-run` command executes workflow YAML files from the command line — useful for scripted experiments, CI testing, and headless operation.

### Usage

```
softae-run <workflow.yaml> [OPTIONS]
```

### Options

| Flag | Description |
|---|---|
| `--mock` | Force mock instruments (no hardware) |
| `--real` | Require real instruments (fail if unavailable) |
| `--dry-run` | Parse and validate only — prints resolved steps |
| `--validate` | Check all instrument/method names exist against the driver registry (exit 3 on failure) |
| `--log-dir DIR` | Directory for JSON-lines logs (default: `./logs`) |
| `--verbose` / `-v` | Print step-by-step progress to stdout |

`--mock` and `--real` are mutually exclusive. Omitting both uses auto-detect.

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Workflow error / instrument failure |
| 2 | Bad arguments / parse error |
| 3 | Validation failure (`--validate` found errors) |
| 130 | Interrupted (Ctrl+C) |

### Examples

```powershell
# Validate a workflow without executing
softae-run workflows/examples/01_hello_stage.yaml --mock --dry-run

# Check that all instrument/method names are valid
softae-run workflows/standard_eis_sweep.yaml --validate

# Run a simple stage test
softae-run workflows/examples/01_hello_stage.yaml --mock --verbose

# Run a 3-channel EIS sweep with logging
softae-run workflows/examples/03_three_channel_eis.yaml --mock -v --log-dir ./my_logs

# Run against real instruments
softae-run workflows/standard_eis_sweep.yaml --real -v
```

### Log Output

Each run produces a JSON-lines file in the log directory:

```
logs/
  hello_stage_20260305T214242Z.jsonl
  three_channel_eis_20260305T214255Z.jsonl
```

Each line is a JSON object:
```json
{
  "timestamp": "2026-03-05T21:42:55.821Z",
  "workflow": "three_channel_eis",
  "step": "startup_flush",
  "instrument": "syringe",
  "method": "single_pump",
  "params": {"res_vol": 1000.0, "ID": 0, "rate": 200.0, "dispense_vol": 100.0},
  "tags": {},
  "duration_s": 0.109,
  "result": "ok"
}
```

---

## 6. Writing Workflow YAML Files

### Minimal Example

```yaml
name: my_experiment

setup:
  - name: set_temp
    instrument: temp_controller
    method: write_sp
    params:
      T_SP: 40.0
      print_flag: 0
```

### Full Schema

```yaml
name: "experiment_name"              # REQUIRED
description: "What this does"        # optional
variables:                           # optional — $var references
  my_list: [1, 2, 3]
  my_value: 42
metadata: {}                         # optional — logged for provenance

setup:                               # list of steps (at least one section needed)
  - name: step_name                  # REQUIRED
    instrument: instrument_name      # REQUIRED (must match InstrumentManager)
    method: method_name              # REQUIRED (must be a callable on the driver)
    params:                          # optional — kwargs passed to method
      key: "$my_value"               #   $var interpolation
    timeout_s: 120                   # optional — max seconds
    retry: 1                         # optional — retry attempts (default 0)
    depends_on: []                   # optional — Step names that must complete first. Omit for sequential order; use `[]` for explicit parallelism.
    tags: {}                         # optional — metadata

loop:                                # optional — dict, NOT a list
  iterate_over: my_list              # optional — variable name (list→len, int→count)
  steps:
    - name: measure
      instrument: pico1
      method: sendscript_getdata
      params:
        mscrpath: scripts/example_eis.mscr
        outdir: ./output
        chan: 1

teardown:                            # optional — always runs (even on error/abort)
  - name: cleanup
    instrument: temp_controller
    method: write_sp
    params:
      T_SP: 10.0
      print_flag: 0
```

### Variable Interpolation

| Syntax | Behavior | Example |
|---|---|---|
| `"$var"` (exact match) | Replaced with variable value, **type preserved** | `"$my_value"` → `42` (int) |
| `"prefix_$var_suffix"` | String substitution only | `"out_$name.csv"` → `"out_test.csv"` |
| Nested | Works inside dicts and lists in params | `{"x": "$val"}` |

### Loop Behavior

- `iterate_over` references a variable name
- If the variable is a **list**: iterations = `len(list)`
- If the variable is an **int**: iterations = that value
- If omitted: 1 iteration
- Loop steps get `__iter0`, `__iter1`, … suffixes and `{"iteration": "0"}` tags

### Parallel Step Execution (`depends_on`)

Steps can declare explicit dependencies to enable parallel execution within a phase:

```yaml
setup:
  - name: set_temp
    instrument: temp_controller
    method: write_sp
    params: { T_SP: 40.0, print_flag: 0 }

  - name: init_stage
    instrument: stage
    method: stage_init
    params: {}
    depends_on: []          # ← explicit empty: no deps, can run in parallel with set_temp

  - name: wait_for_temp
    instrument: temp_controller
    method: wait
    params: { within: 2.0 }
    depends_on: ["set_temp"]  # waits for set_temp to finish
```

**How it works:**

| Scenario | Behavior |
|---|---|
| No `depends_on` field at all | Step implicitly depends on the previous step → **sequential** (backward-compatible) |
| `depends_on: []` (explicit empty) | No dependencies → can run in parallel with other independent steps |
| `depends_on: ["step_a", "step_b"]` | Waits until both `step_a` and `step_b` complete before starting |

The executor groups steps into **tiers** using topological sorting. Steps in the same tier run concurrently via `asyncio.gather`. Steps in later tiers wait for all their dependencies to complete.

**Validation:**
- Circular dependencies (A→B→A) are detected at parse time and raise an error
- References to non-existent step names are rejected at parse time
- Dependencies must reference steps within the same phase (setup, loop, or teardown)

**Loop steps:** Inside a loop, dependency names resolve within the same iteration. If loop step `"measure"` depends on `"deposit"`, then `measure__iter2` waits for `deposit__iter2`.

**If a dependency fails:** All steps that depend on the failed step are skipped. Other independent steps in the same tier continue to completion.

### Important: Param Names Must Match Driver Signatures

YAML `params` keys are passed directly as `**kwargs` to the driver method. They must match the Python method's parameter names exactly:

| Instrument | Method | Required Params |
|---|---|---|
| `stage` | `move_to` | `x`, `y` |
| `stage` | `move_by` | `dx`, `dy` |
| `syringe` | `single_pump` | `res_vol`, `ID`, `rate`, `dispense_vol` |
| `piezo` | `set_channel` | `channel`, `enabled` |
| `piezo` | `apply_profile` | `frequency_hz`, `on_s`, `rest_s` |
| `piezo` | `standby` | (none) |
| `temp_controller` | `write_sp` | `T_SP`, `print_flag` |
| `temp_controller` | `wait` | `within`, `equilibration_time`, `timeout` |
| `temp_controller` | `ramp_linear` | `start`, `end`, `rate`, `step` |
| `temp_controller` | `anneal` | `target_temp_C`, `hold_time_s`, `ramp_rate` (opt), `tolerance` (opt, default 1.0) |
| `pico1`/`pico2` | `sendscript_getdata` | `mscrpath`, `outdir`, `chan` |
| `rh_controller` | `set_setpoint` | `sp` |
| `camera` | `snap` | (none) |
| `lamp` | `on` / `off` | (none) |

### Bundled Templates

| File | Description | Instruments Used |
|---|---|---|
| `workflows/standard_eis_sweep.yaml` | 16-channel EIS sweep with formulation | temp, syringe, pico1 |
| `workflows/single_drop_and_measure.yaml` | Single-channel deposit + EIS | temp, syringe, pico1 |
| `workflows/temp_ramp_eis.yaml` | Temperature ramp with EIS at each point | temp, syringe, pico1 |
| `workflows/examples/piezo_assisted_dispense.yaml` | Piezo on/off around dispense with optional profile apply | piezo, syringe |
| `workflows/examples/01_hello_stage.yaml` | Stage movement test | stage |
| `workflows/examples/02_temp_setpoint.yaml` | Temperature set/read | temp_controller |
| `workflows/examples/03_three_channel_eis.yaml` | 3-channel deposit + EIS loop | stage, syringe, pico1 |

---

## 7. EIS Analysis Pipeline

### Data Container: `EISResult`

```python
from softae.analysis.eis_data import EISResult

# Load from file
result = EISResult.load("path/to/eisdata.txt")

# Create from arrays
result = EISResult.from_arrays(
    channel=1,
    f=frequencies,
    z_real=z_prime,
    z_imag_neg=neg_z_double_prime
)

# Save
result.save("output/E1_eisdata.txt", study_name="my_study")
```

**File format:** 5-column text with `#`-prefixed metadata header:
```
# channel: 1
# timestamp: 2026-03-05T14:30:00
# eis_params: {"preset": "Standard", ...}
f(Hz)	Z_total(Ohm)	phase(deg)	Z'(Ohm)	-Z''(Ohm)
200000.0	105.2	-5.3	104.8	9.7
...
```

### Circuit Fitting

```python
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant

# Per-sample geometry in cm: electrode gap L, film thickness t, stripe length w.
cell = CellConstant.from_legacy(0.2, 0.175, 0.2)

# `engine` is deliberately not passed — `[eis] engine` in softae_config.toml
# decides whether the legacy or the gated engine runs, for every call site at once.
report = analyze_spectrum(eis_result, cell=cell, model_name="simpleSalt")

fit = report.fit
print(f"R0 = {fit.R0:.1f} Ω, R1 = {fit.R1:.1f} Ω, success = {fit.success}")

# Conductivity is reported, not merely computed: it can be a value, an upper
# bound, or nothing at all. Always check the mode before reading `.value`.
if report.sigma.mode == "value":
    print(f"σ = {report.sigma.value:.3e} S/cm")
elif report.sigma.mode in ("bound", "bound_unqualified"):
    print(f"σ ≤ {report.sigma.upper_bound:.3e} S/cm  (instrument-limited)")
else:                                  # "unavailable"
    print("σ unavailable — no per-sample thickness, or the spectrum was rejected")
```

> **Deprecated:** `circuit_fitting.z_to_sigma(L, t, w, R1)` and `FitResult.sigma(L, t, w)`
> both emit a `DeprecationWarning` and have no callers left in the system. They divide by a
> geometry with no provenance and bypass `[eis] engine` entirely, so a number taken from them
> cannot be told apart from one the standard suite produced. They are kept only as the
> independent oracle the parity tests check `CellConstant.sigma` against.

### Available Models

| Model | Circuit | Parameters | Use Case |
|---|---|---|---|
| `simpleSalt` | R₀-CPE₀-p(R₁,C₀) | 5 free | General ionic conductors |
| `flexSalt` | R₀-CPE₀-p(R₁,C₀) | 4 free + fixed C₀ | Known stray capacitance |
| `simpleSaltMembrane` | R₀-CPE₀-p(R₁-Wo₁,C₀) | 7 free | Membrane / diffusion systems |

---

## 8. Emergency Stop & Safe Exit

Two stop controls bracket the toolbar, visible on every tab. They are placed at opposite
ends deliberately: pressing the wrong one of two adjacent buttons is exactly the mistake to
design out, and an emergency stop is not something to hit on the way to closing the app.

| | Control | When |
|---|---|---|
| **left** | red **⛔ EMERGENCY STOP** | something is going wrong, now |
| **right** | amber **⏻ SAFE EXIT** | you are finished and leaving |

Both drive the *same* park sequence — there is deliberately no second path to the hardware:

1. Retract dispenser head
2. Stop all syringe pumps (0, 1, 2)
3. Set temperature to 10 °C
4. Turn lamp off

Each step is attempted even if others fail. A dialog reports success or lists any errors.

### Safe Exit and the dispenser head

Safe Exit parks the rig and then closes the window. **If the head is down when you press it,
you are asked** whether to raise it or leave it lowered:

- **Raise head, then exit** — the default, and what Enter selects.
- **Leave head down, then exit** — for a head that is *holding a position*: an anneal hold in
  the flush basin, a paused cast, a drop it is sitting in. Raising it would pull the tip clear.
- **Cancel** — nothing is touched and the window stays open. Discovering the head is down is
  sometimes itself the reason not to exit.

The question is asked only when there is a decision to make. A raised head, an absent or
disconnected syringe, or a driver that does not track head state all exit without prompting —
you are never asked about hardware whose state nothing actually knows.

> **Every other route out raises the head, including the window's X button.** That is not an
> oversight to be fixed by making X ask too. Closing a window is an unattended act — nobody is
> left to decide — so the safe default applies. Safe Exit is the deliberate path, and being
> asked is what earns the right to leave the head down.

Only the head is negotiable. Pumps, temperature and lamp are parked unconditionally in both
modes; a stop that skipped them would not be a park.

If any subsystem fails to park, Safe Exit reports what failed and asks before closing —
closing the window would remove your easiest way to see it.

**Closing the window mid-run:** if you close the application while an experiment, BO campaign, or Arrhenius temperature/EIS sweep is running, the in-progress run is cooperatively aborted (the same effect as its **Abort/Stop** button) before the window finishes closing, so no run keeps issuing instrument commands after the GUI is gone. Close may take up to a few seconds while the run winds down.

---

## 9. Error Reference

```
SoftAEError (base)
├── InstrumentError (message, instrument)
│   ├── ConnectionError_     — failed to open port
│   ├── CommunicationError   — timeout / no response
│   └── SafetyError          — value exceeds configured limit
│         (requested, limit)
├── WorkflowError
│   ├── StepTimeoutError     — step exceeded timeout_s
│   ├── AbortedError         — user or agent aborted
│   └── ValidationError_     — bad YAML / missing field
└── AnalysisError            — fit failure / data mismatch
```

All instrument errors include the instrument name. `SafetyError` includes the requested value and the limit. `StepTimeoutError` includes the step name and timeout duration.

---

## 10. Instruments Reference

### Registered Instrument Names

| Name | Driver | Mock Available | Methods |
|---|---|---|---|
| `stage` | Newport ESP301 | ✅ | `stage_init`, `move_to`, `move_by`, `home_stage`, `live_position`, `stage_end` |
| `syringe` | Harvard Apparatus | ✅ | `single_pump`, `head_flip`, `head_retract`, `head_descend`, `head_check`, `syr_end` |
| `temp_controller` | Novus N1040 | ✅ | `write_sp`, `get_sp`, `get_pv`, `get_pv_surf`, `wait`, `ramp_linear`, `anneal` |
| `pico1` / `pico2` | PalmSens EmStat Pico | ✅ | `sendscript_getdata`, `eis_extractdata`, `eis_plotdata` |
| `camera` | ThorLabs Zelux | ✅ | `snap`, `acquire_n_frames`, `save_image` |
| `lamp` | MCP4728 quad-DAC ch A (`AsyncDACSwitch`) | ✅ | `on`, `off`, `set_eeprom_defaults` |
| `keithley` | Keithley 2700 | ✅ | `read_resistance` |
| `ht_sensor` | SHT31-D | ✅ | `get_T`, `get_H` |
| `rh_controller` | Trinket M0 PID | ✅ | `set_setpoint`, `start`, `stop`, `get_H`, `wait` |
| `piezo` | Trinket piezo controller | ✅ | `set_channel`, `set_frequency`, `set_sweep`, `apply_profile`, `standby`, `reset_config` |

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `softae-gui` not found | Not installed in editable mode | `pip install -e .` from `softae-next/` |
| All instruments show "DISCONNECTED" | Hardware not connected, or auto-detect fell back | Check cables/ports; or run with `--mock` |
| `SafetyError: requested 250.0 exceeds limit 200.0` | Temperature setpoint above config max | Edit `[safety] temp_max_C` in `softae_config.toml` |
| Camera preview freezes GUI | Not using CameraWorker thread | Report as bug — SDK must run on dedicated thread |
| RH shows "nan %" | PID loop not started and no sensor connected | Start PID loop or connect SHT31-D sensor |
| Workflow step fails with `TypeError` | YAML param names don't match driver method signature | Check [Section 6 param table](#important-param-names-must-match-driver-signatures) |
| `No module named 'hid'` | Hardware optional deps not installed | `pip install -e ".[hardware]"` |
| Config not found | TOML not in CWD or env var | Set `SOFTAE_CONFIG` env var or copy `softae_config.toml` to CWD |
| EIS fit returns `success=False` | Bad initial guess or wrong circuit model | Try a different model; check data quality |

---

---

## 12. Documentation Site

The project includes a **mkdocs-material** documentation site with auto-generated API reference.

### Building Locally

```powershell
pip install -e ".[docs]"    # installs mkdocs-material + mkdocstrings
mkdocs serve                 # live preview at http://127.0.0.1:8000
mkdocs build                 # static site in site/
```

API reference pages are auto-generated from docstrings by `mkdocstrings[python]`. Adding or renaming a public module requires a corresponding stub in `docs/api/`.

---

## 13. Data Persistence (DataStore)

Every experiment run is automatically persisted in a project-scoped SQLite database.

### Configuration

Set the project directory and database filename in `softae_config.toml`:

```toml
[data]
project_dir = "~/SoftAE_Data"     # auto-created on first use
db_filename = "softae_data.db"    # default
auto_save_eis = true               # auto-save EIS data files
```

### What Gets Stored

| Table | Content |
|---|---|
| `experiments` | Run lifecycle — name, start/end time, status, config hash, notes |
| `measurements` | Per-channel raw-data file paths + timestamps, plus `modality`, `payload_path` / `payload_format`, and `sample_uuid` |
| `conditions` | Multi-stage environmental snapshots (formulation, processing, measurement, anneal) |
| `fit_results` | Circuit fit parameters (R₀, R₁, model, σ) per measurement |
| `formulations` | Dispense volumes per channel, deposit area + thickness provenance, `sample_uuid` |
| `electrode_occupancy` | Which wells are spent, per `(board_id, electrode)` — plus `sample_uuid` |
| `schema_version` | Append-only **epoch ledger** — records both schema shape and changes of *meaning* (e.g. the 2026-08-07 deposit-area correction, which moved stored thicknesses without moving a column) |
| `doe_parameters` | *(reserved for Phase 6 optimizer)* |

### Measurement payloads (netCDF)

Alongside the transitional `.txt` spectrum, every routed measurement writes a **self-describing
netCDF payload**:

```
runs/<run_id>/data/eis/<stem>.txt     # transitional raw text, unchanged
runs/<run_id>/data/eis/<stem>.nc      # payload — xarray Dataset
```

Payloads live in a **sibling** tree per modality (`data/<modality>/`), so retiring the `.txt`
files later is not also a payload migration. The `.nc` file reconstructs the measurement on
its own: its `attrs` carry `run_id`, `measurement_id`, `channel`, the electrode geometry and
the `sample_uuid`, so a file found on disk with no database beside it still says what it is
and what it was measured across.

Writing a payload is **best-effort**. If it fails, the measurement row is still written, and
`payload_path` / `payload_format` stay NULL — a NULL path means *no payload*, never a path to
a file that is not there.

### `sample_uuid` — one identity per cast well

A `sample_uuid` is minted when a well is consumed — **one per (trial, channel)** — and stamped
into the `formulations`, `electrode_occupancy` and `measurements` rows for that well, into the
workflow step tags, and into the payload `attrs`. It is what joins *what was cast* to *where it
was cast* to *what was measured*.

- It is a **grouping key, not a unique one.** One sample carries arbitrarily many measurements
  — a three-temperature sweep off one film is one identity and three independent measurement
  rows, each with its own timestamp, conditions snapshot and payload.
- A batch round of q wells mints **q distinct identities**, because four wells cast together
  dry differently, are measured separately, and can be discarded independently.
- **Rows recorded before 2026-08-08 carry NULL** and are not backfilled. Inventing an identity
  per historical row would assert that three rows describing one physical sample are three
  samples. A resumed campaign mints for new trials only.

### Automatic Integration

- **HT Experiment tab:** Calls `start_run()` on workflow start, `record_measurement()` on each EIS step, `finish_run()` on completion
- **Manual tab:** Records EIS snapshots to a daily pseudo-run (`manual_YYYYMMDD`)
- **CLI runner:** Logs config hash as first provenance event

### Programmatic Access

```python
from softae.core.data_store import DataStore

ds = DataStore("~/SoftAE_Data")
runs = ds.list_runs()              # all experiment runs
meas = ds.get_measurements(run_id) # EIS measurements for a run
conds = ds.get_conditions(run_id)  # environmental snapshots
fits = ds.get_fit_results(run_id)  # circuit fit parameters
```

The database uses WAL mode for concurrent reads during active experiments.

---

## 14. Deposition Digital Twin

The formulation core (see the **Formulation Manager** in [Tab 5](#tab-5-ht-experiment)) answers *"what do I elute?"*. The deposition twin (`softae.core.deposition`) answers *"what ends up in the well?"* — it casts an `ElutionResult` into cylindrical wells, evaporates the carrier (solvent) at a tunable percentage while retaining all dep (solute) volume, and reports flat-disc film thickness plus a full mass balance. Pure math, no hardware.

```python
from softae.core.formulation import (
    Chemical, ChemicalCatalog, Solution, SolutionComponent, compute_elution_volumes,
)
from softae.core.deposition import WellGeometry, simulate_plate_deposition

catalog = ChemicalCatalog()
catalog.add(Chemical("PEO", density_g_per_mL=1.2))
catalog.add(Chemical("Water", density_g_per_mL=1.0))

solutions = {"stock": Solution("stock", [
    SolutionComponent("PEO", "dep", 1.0, "mL"),      # solute — retained
    SolutionComponent("Water", "carrier", 3.0, "mL"), # solvent — evaporates
])}

elution = compute_elution_volumes(solutions, catalog, target_deposition_uL=20.0)

well = WellGeometry(diameter_mm=5.0, depth_mm=2.0)   # capacity_uL ≈ 39.3 (1 µL = 1 mm³)
summary = simulate_plate_deposition(
    elution, well, evaporation_pct=95.0, n_wells=4,  # dispense_uL=None → equal split
)

w = summary.wells[0]
print(w.wet_thickness_um, w.final_thickness_um)      # flat-disc film thickness
print(w.overflows, summary.any_overflow)             # wet volume > capacity (flag, not error)
print(summary.total_eluted_uL, summary.total_dispensed_uL,
      summary.undeposited_uL, summary.total_evaporated_uL, summary.total_final_uL)
print("\n".join(summary.summary_lines()))            # human-readable mass balance
```

- `dispense_uL` may be `None` (equal split of `grand_total_uL`), a single float per well, or a per-well list — total dispensed can never exceed total eluted; the remainder is tracked as `undeposited_uL`.
- `simulate_well_deposition(...)` is the single-well equivalent, returning one `WellDepositionResult`.
- Pass `carrier_keys=carrier_component_keys(solutions)` to either function for per-component final-volume breakdowns (`component_final_uL`).

### Standalone GUI

The twin is also available as an interactive standalone app — no instruments, no qasync, just the pure math behind a live panel:

```powershell
softae-deposition                        # console script
python -m softae.gui.deposition_app      # equivalent module launch
```

> The `softae-deposition` command is registered via `[project.scripts]`; if it is not found after pulling this feature, run `pip install -e .` once to refresh the entry points.

**Inputs** (every change recomputes live via `compute_elution_volumes` → `simulate_plate_deposition`):

- **Stocks** — check the solutions to include, then set each one's share of the deposition. The stock table has seven columns: **Use**, **Auto**, **Solution**, **Fraction**, and the read-only outputs **Eluted µL / Dep µL / Carrier µL**.
  - **Auto toggle (per row).** *Auto ON* means "absorb the remainder so all fractions sum to 1" — the row's Fraction is greyed and, after each recompute, shows the resolved share the core assigned it. *Auto OFF* lets you type an exact share; `0` is honoured as a literal zero (elute none of this stock's dep-share). **The default is Auto OFF with Fraction 0.00** (explicit-first) — set fractions or enable Auto to deposit.
  - **Auto-balance all** — one click sets Auto ON for every checked stock, restoring the equal split that sums to 1.
  - **Normalize** — rescales your *explicit* (Auto-off) fractions so they sum to exactly 1.00 (the residual from rounding lands on the largest row). This is the one-click fix for the amber `Σ < 1` / `Σ > 1` states; it's enabled only when there is an explicit sum to rescale and the split isn't already balanced (disabled when Auto already covers the remainder or Σ is already 1).
  - **Σ-fractions indicator** (below the table) is a live, non-blocking sanity check — results always compute; the label only warns. **Green** = Σ = 1 (or auto-balanced). **Amber** = Σ < 1 (deposits short by N µL), Σ > 1 (overshoots the target), or all-zero ("set fractions or enable Auto"). **Red** = explicit fractions already exceed 1 while Auto rows are present (the auto rows get clamped to 0). Carrier-only stocks (`dep_fraction` 0) cannot carry a dep-share, so they are excluded from Σ and flagged separately.
  - **Eluted / Dep / Carrier µL** columns fill after each recompute. Equal *dep*-share does **not** mean equal *eluted* volume: because `eluted = dep / dep_fraction`, five stocks each carrying an equal `0.20` dep-share elute wildly different totals (the seeded stocks span ~8 → 99 µL). These columns make that spread visible instead of surprising.
  - **Show component breakdown** (optional toggle) reveals a per-(solution, component) table with each chemical's **Role** (dep or carrier) and its eluted µL — e.g. it traces Silica's ~99 µL total to its ~95 µL of isopropanol carrier.
- **Target deposition (µL)** — the dep (solute) volume goal passed to the formulation core
- **Well geometry** — diameter and depth (mm) of the cylindrical well
- **Wells & dispense mode** — number of wells; equal split of the eluted total, or a fixed µL per well
- **Evaporation** — slider (0–100 %, 0.5 % steps) twinned with a spinbox

**Outputs:**

- Per-well results table (dispensed / wet / final volumes, film thicknesses, fill fraction)
- Mass-balance strip: eluted / dispensed / undeposited / evaporated / final
- Red banner when any well's wet volume exceeds capacity (overflow)
- `WellSketch` cross-section drawing that animates as you drag the evaporation slider
- Invalid inputs surface as an inline error label (core `ValueError`s are caught — the panel never crashes)
- **Export CSV…** — writes the current computed result to a file you choose: a sectioned CSV with `#CONFIG` (inputs), `#STOCKS` (per-stock elution), `#MASS_BALANCE`, and `#WELLS` (per-well) blocks. The button is enabled only once a valid result is cached; a write error (`OSError`) is reported on the status label rather than crashing.

**Catalogs:** chemicals and solutions load from `chemicals.csv` / `solutions.csv` under `[paths] data_root` in `softae_config.toml`, falling back to the repo root, then to empty catalogs with a status message — the app always starts.

### Managing catalogs

softae-next is now the **canonical editor** for the chemical/solution catalogs. They live at `chemicals.csv` / `solutions.csv` under `[paths] data_root` in `softae_config.toml`. The `data_root` path is resolved **relative to the config file's directory** (not the working directory), honoring absolute paths and `~`; when no config is present it falls back to `./data`. The catalogs ship seeded with the legacy bench entries (Water, Glycerol, AETDAB, Isopropanol, Fumed silica, Lithium chloride, PEO 20 kDa and their stocks).

In the main app the catalogs are reachable directly as two tabs: **11. Catalogs** (a read-only browser listing the chemicals and solutions, with an **Edit** button) and **12. Deposition** (the deposition twin embedded in the main window). Editing the catalogs anywhere refreshes both of these **live** — the browser re-lists and the embedded deposition panel re-reads its stocks the moment you save.

Reach the catalog editor several ways: the main window's **Catalogs → Edit Catalogs…** menu (or its toolbar button), the **Edit** button on the Catalogs tab, or the deposition panel's **Manage Catalogs…** button all open a **slim Catalog Manager** dialog (catalog CRUD only — no formulation calculator). The full **Formulation Manager** (catalog editing *plus* the elution calculator and pump controls) still opens from Tab 5. Both share the same editing, validation, and canonical-Save behavior described below.

- The editor auto-loads the current catalogs from `data_root` (no folder dialog needed). The primary **Save** writes back to the canonical `data_root` location (creating the directory if missing); **Save As…** / **Load From…** remain available for ad-hoc files.
- **Reload catalogs** (deposition app) re-reads the CSVs from disk into the stock list.
- Edits made in the editor appear **live** in the deposition panel's stock list the moment you save — no restart, and surviving stocks keep their checked state and fractions.

Each solution component's **Chemical** field is a **dropdown** of the current catalog chemicals (including any you just added but haven't saved yet), so a component can't reference a mistyped chemical name; a legacy/unknown reference loaded from disk is preserved and still selectable. **Renaming a chemical cascades**: every solution component that referenced the old name is updated automatically (both in memory and in the on-screen dropdowns). A rename that would collide with another chemical's name is blocked and reverted with a warning; clearing a name is not cascaded (the now-orphaned reference is flagged by validation instead).

Both **Save** and **Calculate** run a validation pass first. If it finds a blank/non-positive **density**, a blank/non-positive component **quantity**, an **unknown-chemical** reference, or a data-bearing row with a blank **name**, it lists the issues in a **Proceed/Cancel** prompt (Cancel aborts — no write and no compute; Proceed continues with the documented defaults) and tints the offending cells light red until you fix them. Blank molar mass or viscosity are legitimate and are not flagged. Validating on Calculate pre-empts the hard error an unknown-chemical reference would otherwise raise mid-computation.

The editable field set is full-fidelity: every chemical field round-trips (including **viscosity** and the **particulate** flag), as does each solution component's **calc mode**. (An earlier version silently dropped these on save; that data-loss bug is fixed.)

---

## 15. Autonomous Campaigns

A campaign is a closed loop: **suggest → cast → measure → tell**. Run it interactively from
[Tab 10](#tab-10-live-bo-campaign), or headless with `softae-campaign`.

```bash
softae-campaign check   my_campaign.toml                       # validate the spec, run no hardware
softae-campaign run     my_campaign.toml --yes --head-up       # execute
softae-campaign resume  my_campaign.toml --yes --head-up       # continue a saved checkpoint
softae-campaign run     my_campaign.toml --yes --mock          # mock instruments, no rig
softae-campaign run     my_campaign.toml --project ./runs/aug  # DataStore + checkpoint location
```

`--project` overrides `[data] project_dir`, and it also decides **where the resume checkpoint
lives** — a `resume` pointed at a different project directory will not find the run you mean.
`--mock` swaps in mock instruments for a full dry rehearsal of the loop.

### The head-state gate: `--head-up` / `--head-down`

This is the headless counterpart of the GUI's operator head-position verification, and it is
the one flag on this command worth reading twice. **The campaign never assumes the dispenser
head's state.** The loop drives the head with *conditional* commands — raise it if it is down,
lower it if it is up — so a wrong belief does not merely mis-report, it costs one wrong flip:
the head goes down when it should have come up, or drives into a board it should have cleared.

```bash
softae-campaign run my_campaign.toml --head-up      # head is currently RAISED
softae-campaign run my_campaign.toml --head-down    # head is currently LOWERED
```

The two are mutually exclusive. **Omit both and the CLI prompts on the terminal** — which is
fine when you are sitting there, and a hang when you are not. A genuinely unattended launch
(cron, `nohup`, a scheduler) must state the head position on the command line, alongside
`--yes`; that pairing is what makes the run answerable without a human.

> Look at the rig, do not recall it. The flag records what is true right now, not what the
> last run left behind — an aborted run, a manual jog or a power cycle all break that
> inference. Every other headless gate defaults to the *safe* answer (board exchange cancels,
> board freshness resumes past used wells, a stock shortfall stops the run); head position has
> no safe default to fall back on, which is exactly why it is asked.

### Campaign spec (TOML)

```toml
name      = "peo_licl_scan"
channels  = [21, 22, 23, 24]
pcb_name  = "SoftAE_EIS_4Stripe"
budget    = 40
optimizer = "bayesian"
two_phase = true

[parameter_space.vol_p0]
type = "float"
low  = 5.0
high = 30.0
```

The loader **refuses what it cannot represent faithfully**. An unknown key is an error, not
a silently-defaulted field, and fields carrying live Python objects (`prior_mean`,
`formulation`, `run_plan`, `seed_observations`) cannot be set from a file at all — a spec
that silently ran a different experiment from the one the file describes is the failure this
prevents. Those campaigns are built in Python or from Tab 10.

### What to measure — the `[measurement]` block

What a campaign measures is one block, named by **modality**, so a second kind of data needs
no new spec fields:

```toml
[measurement]
modality = "eis"      # the only modality a campaign can run today
preset   = "Quick"    # an [eis_presets.*] section name
enabled  = true       # false = formulate and cast, but do not measure

[measurement.overrides]
n_points = 40         # modality settings layered over the preset
```

The three EIS-shaped fields it replaces still work and still mean the same thing:

| Legacy spelling *(deprecated)* | Block spelling |
|---|---|
| `eis_preset = "Quick"` | `[measurement]` → `preset = "Quick"` |
| `eis_overrides = { n_points = 40 }` | `[measurement.overrides]` → `n_points = 40` |
| `measure_eis = false` | `[measurement]` → `enabled = false` |

A file written before the block still loads; the legacy fields raise a `DeprecationWarning`
and are folded *into* the block, so there is exactly one authority at run time. Files the
system writes carry the block only. The old fields are removed once one full campaign has
run from a measurement-block spec.

- **Both spellings together are allowed only when they agree.** A disagreement is refused,
  not resolved by precedence: whichever spelling lost was a written instruction to measure
  something else, and neither the file nor the caller would show it had been overruled.
  `softae-campaign check` surfaces this before any hardware moves.
- **Naming a modality that is not built refuses to start** — before an instrument connects
  or a run row is written, with the registered modalities listed in the error.
- **Resume is unaffected.** Only `modality` is identity-bearing, and the default `"eis"`
  contributes nothing to the checkpoint fingerprint, so a campaign checkpointed before the
  block existed still resumes. `preset` / `overrides` / `enabled` are settings, re-tunable
  between sessions, exactly as they were under the old names.

### The objective is derived, not chosen

`[eis] objective = "auto"` (the default) resolves the metric from what the campaign can
actually measure, and the **direction follows the metric**:

| Campaign mode | Twin can predict thickness? | Metric | Direction |
|---|---|---|---|
| **composition** — carries a formulation | yes | σ | maximise |
| **volume** — raw `vol_params` | no: no stock identity ⇒ no elution ⇒ no dry thickness | mean \|Z\| | minimise |

Neither is a fallback for the other. Volume mode is a legitimate exploration mode where σ is
*impossible*, not merely missing. Setting `CampaignSpec.objective` to an explicit
`maximize`/`minimize` is honoured only when it agrees with the resolved metric and
**refused** when it does not — minimising \|Z\| and maximising σ are the same goal, so a
contradiction would spend the whole budget finding the worst material on the board while
every step reported progress.

### One σ everywhere

The conductivity the GUI displays and the conductivity the autonomous objective optimises
against are **the same number**, produced by the engine named in one place — `[eis] engine`
in `softae_config.toml`. No surface names its own engine, so flipping that key moves the
whole system at once.

This closed a live divergence. Until 2026-08-09 the campaign objective forced the gated
engine while every GUI surface followed the config key; on one synthetic spectrum
(R_bulk = 2000 Ω, t = 150 µm) the objective reported σ = 3.34e-2 S/cm and the GUI
7.36e-2 S/cm — **a factor of 2.2 for the same film**, with nothing on screen saying so. Both
now read 7.36e-2 under the shipped `engine = "legacy"` and 3.34e-2 under `engine = "gated"`.
Nothing about the casting or the measurement differs between the two surfaces, so nothing
about the reported σ should either.

`[eis] objective` and `[eis] engine` remain different keys: the first chooses **which
metric** (σ or mean |Z|, per the table above), the second **which physics computes it**.

### Rounds, boards and budget

A round is sized *before* anything is suggested, to the smallest of the requested batch
size, the electrodes still free on the current plate, and the budget still unspent.

- **A full board is exchanged before the round**, so a round is never split across a plate
  swap. Half a cast batch held through an operator prompt of unbounded duration is worse
  than a narrower round that completes.
- **The final round narrows to the budget** rather than rounding up to the next multiple of
  q. A budget of 5 with q=4 spends exactly 5 electrodes.

Board exchange prompts the operator and can be **cancelled** — which stops the run while
keeping everything already measured. With **no handler** (a fully headless run), an exchange
request stops cleanly rather than assuming a fresh plate: casting onto a board that is still
full would destroy occupied single-use wells.

### Occupancy, resume and checkpoints

Drop-cast wells are single-use, so occupancy is persisted per `(board_id, electrode)` and
survives a restart. Resuming a campaign that finds recorded occupancy asks whether the plate
is **fresh**, a **resume**, or to **cancel**.

`--resume` continues a saved checkpoint rather than restarting. It is **off by default**:
silently resuming would make a re-run mean something different from what was typed. The
checkpoint is fingerprinted against the spec — a changed parameter space, objective or
optimizer setting is refused rather than continued into.

---

## 16. EIS Commissioning & Calibration

Commissioning measures the **fixture** rather than a sample: what the leads and multiplexer
contribute, and what the instrument can actually resolve. Because fixture electronics drift
is minimal, a calibration is a **durable asset reused across campaigns**, not a per-run
chore.

```bash
softae-commission status                                    # what is calibrated, what is next
softae-commission run blank_short --channels 1-32 --fixture mux16 --yes
softae-commission derive --fixture mux16                    # spectra → calibration → TOML
softae-commission history --fixture mux16                   # successive sets = drift
```

`run` acquires and tags; `derive` reads the tagged spectra back from the database. They are
separate because artifacts arrive over several sessions as parts turn up, and each `derive`
produces the best calibration the artifacts so far support.

Four flags are routine on every subcommand and need no ceremony: **`--fixture`** names the
fixture (defaults to the configured one) and **`--project`** the project directory holding the
DataStore — both accepted by `run`, `import`, `derive` and `history`, and both must match
across the three or `derive` will look for spectra where none were written. On `run` only,
**`--yes` / `-y`** skips the "is the hardware in place?" prompt and **`--mock`** measures a
simulated fixture, which is how you rehearse the sequence away from the bench.

### The artifacts, in order of value per hour of bench time

| Artifact | Install | Gives you |
|---|---|---|
| **`blank_short`** | jumpered channel (CE–WE shorted) | `R_fixture`, `L_lead` → unlocks series correction |
| **`blank_load`** | precision resistor, `--nominal <ohms>` | end-to-end correction error |
| **`reference_cap`** | low-loss C0G/NP0, `--nominal <farads>` | **measured** phase floor → qualified upper bounds |
| **`reference_r`** | reference resistor, ≥1 per decade | true \|Z\| window for *this* fixture |
| **`blank_open`** | bare, uncast board | whether OSL correction is legitimate at all |

`status` names **the next artifact** rather than listing every absence, and `derive` reports
which single artifact would unblock each remaining capability.

### Two-electrode mode is mandatory for every reference

**Tie RE to CE at the connector before measuring any two-terminal reference** — short,
load resistor, capacitor, or bare board. This is not a refinement; a value taken any
other way is refused at derive time.

A two-terminal load gives the reference stripe no ionic path, so in three-electrode mode
RE floats onto a **capacitive divider between WE and CE** and the instrument reports only
a fraction of the true impedance. The fraction is not a constant. Measured on this rig:

| configuration | apparent / true |
|---|---|
| ch25 blank | 2.24× |
| ch25 + 100 pF | 9.58× |
| ch17 blank (multichannel) | 23.8× |
| **same 1 nF part, two trials** | **9.85× and 4.96×** |

That last row is the decisive one: the ratio is not even reproducible at fixed load, so
there is no correction factor to apply and none may be fitted. Three-electrode
measurement of a two-terminal load is **uncalibratable in principle**, not merely
uncalibrated.

This also resolves a long run of "impossible" results — parts reading 8–9× their markings
and blanks at 120–250 pF. Re-measured with RE tied to CE, a part marked "101" gave 96.4 pF
and one marked "102" gave 974.6 pF, both within 4% of their EIA codes. **The components
were correct all along.**

> **A sample is different, and the distinction matters.** A conductive film in contact
> with the reference stripe *does* establish an ionic path. That is exactly when
> three-electrode sensing is valid and `K_config_factor = 2` is exact — which is why
> verified RE *contact* is a precondition for applying the factor, not a footnote.

Three ways to record the mode:

```bash
softae-commission run   reference_cap --electrode-mode two ...   # prompted at the bench
softae-commission import reference_cap --file <path> --electrode-mode two ...
softae-commission derive --fixture mux16 --declare-electrode-mode two
```

`import` registers an existing spectrum file without re-measuring — useful when the data
already exists from an earlier session or a bench instrument. `--file` is required and
`--electrode-mode` is too (*declaring it is the point of importing*); `--channel` names the
channel when the file itself does not record one, and `--nominal` carries the marked value
exactly as it does on `run`. `derive --declare-...` asserts a mode for spectra already stored,
and **persists it**, so it is asked once. It only touches rows recorded as `unknown`: a
spectrum explicitly recorded as three-electrode stays refused, because that is not missing
information but information saying the value is unusable.

### What closed the loop: `import --re-connection`

Electrode mode says how the cell was *sensed*. `--re-connection` says what physically **closed
the potentiostat's control loop** (overhaul R19) — a different question, and the one that
decides whether a quadrant violation is an instrument artifact or a structural result. It
defaults to `unverified`, which is honest but unlocks nothing, so declare it on import:

| `--re-connection` | Means | Loop closed? |
|---|---|---|
| `unverified` | nobody recorded it *(default)* | unknown |
| `tied_to_ce` | RE jumpered to CE at the connector | yes |
| `bridged_by_sample` | the cast film spans RE to the electrodes | yes |
| `open_by_geometry` | nothing spans the gap to the RE stripe | no |
| `connected` | the wire is on — says nothing about what spans the gap | yes |

> **Use `tied_to_ce` for any two-terminal reference and for open blanks.** It is the state you
> created by hand when you followed the tie-RE-to-CE rule above, and it is the honest label for
> a bare board where nothing bridges anything. It closes the loop *perfectly* while making RE
> read the counter electrode — so the measurement is two-electrode by construction and its
> configuration factor is 1, not 2. Recording it as `connected` or `bridged_by_sample` instead
> would earn the spectrum a K = 2 it has not established, a clean 2× error on the absolute
> number with the fit and residuals all looking perfect.

```bash
softae-commission import blank_short --file ch1_short.csv \
    --electrode-mode two --re-connection tied_to_ce --channel 1 --fixture mux16
```

> **Supply `--nominal` for any part with a marked value.** It is recorded with the
> measurement, because it is not recoverable later — nobody remembers which resistor was in
> the socket weeks ago — and the marking *disagreeing* with the measurement is exactly the
> check that catches an unusable part.

`--nominal` is always in **base SI units** — farads, not picofarads; ohms, not megohms — and
takes scientific notation. There is no unit suffix:

```bash
--nominal 470e-12    # 470 pF reference capacitor
--nominal 1e-9       # 1 nF
--nominal 1e6        # 1 MΩ load resistor
```

A mis-keyed exponent does not pass quietly: `derive` reports the measured-to-marked ratio, so
a 1000× discrepancy is visible in the same place a genuinely bad part would be.

### Deriving when most channels were never measured

Measuring 32 channels of short blank is hours of jumpering. Measuring 7 is one sitting. The
`derive` flag pair that bridges the gap is **`--representative` with `--channels`**, and it is
the most consequential thing on this command:

```bash
softae-commission derive --fixture mux16 \
    --channels 1-32 \          # the FULL channel set the calibration must cover
    --representative 3 \       # the MEASURED channel whose constants the rest inherit
    --nominal-load 1e6         # the marked value of the load resistor
```

- **`--representative <N>`** names one measured channel whose fixture constants (`R_short`,
  `L_lead`, `C_stray`) the **unmeasured channels inherit**. Pick a channel you actually
  measured and that is unremarkable — a representative with an outlying `C_stray` exports its
  own defect to every channel that inherits from it.
- **`--channels`** is the *full* set the calibration must cover, not the set you measured. It
  is what lets `derive` compute the difference — measured versus covered — and it is the only
  way the assumed channels get named at all. Give `--representative` without it and there is no
  set to inherit *into*.

**The inheritance is recorded, not silently applied**, and that provenance travels. Every
inheriting channel is marked `channels_assumed` in the `CalibrationSet`; every later correction
built for one carries `inherited = True`; and each time such a channel is corrected the system
logs `eis_calibration_channel_assumed` with the measured channel-to-channel spread attached. So
a σ derived through an assumed channel is traceable as such months later, from the log line
alone.

> **Assumed is a real uncertainty, not a formality.** Measured on this fixture, `C_stray`
> spanned **10.2–24.7 pF across 7 identical stripes — a 2.4× spread**. Stripes that are
> identical by layout are not identical by measurement, and a channel inheriting from one of
> them inherits a number that was never within a factor of two of correct for some of its
> peers. Derive with as many measured channels as the bench time allows, treat the assumed ones
> as the weaker evidence they are, and measure a channel outright before resting a headline
> result on it.

If you skip both flags, a channel with no constant of its own simply gets **no correction** —
`derive` declines it and says so, naming this pair as the remedy. Declined is safe; assumed is
useful; measured is best.

The rest of `derive`'s surface declares the marked part values at derive time, for artifacts
imported or measured without a `--nominal`:

| `derive` flag | Purpose |
|---|---|
| `--channels` | full channel set the calibration covers, so assumed channels are known |
| `--representative N` | measured channel whose constants unmeasured channels inherit |
| `--nominal-load` | load resistor's marked value, **ohms** |
| `--nominal-cap` | reference capacitor's marked value, **farads** |
| `--nominal-r` | reference resistor's marked value, **ohms** |
| `--declare-electrode-mode` | assert the mode for spectra stored as `unknown` (never overrides a recorded one) |
| `--fixture` / `--project` | which fixture, which DataStore |

Same base-SI rule as `--nominal`: `--nominal-cap 470e-12`, not `470`.

### Two things worth knowing before the bench

**An unusable open blank is a result, not a failure.** An open circuit's impedance can exceed
the instrument ceiling across most of the band, in which case the record is noise — and that
is itself the evidence that shunt admittance is negligible, which is precisely when
short-only series correction is *exact*.

**An open cell floats the reference electrode.** On a three-electrode fixture a bare-board
open measures inter-stripe geometry, not the fixture. A genuine fixture open needs RE tied to
CE at the connector, which the commissioning board should have designed in.

### Where it lands

| Destination | Contents |
|---|---|
| `measurements` table, `role != 'sample'` | raw spectra, queryable like any other |
| `calibration/eis/<fixture_id>.toml` | derived constants — **commit this file** |
| `eis_calibrations` table | append-only history; successive sets *are* a drift measurement |

Staleness is by **hardware identity, not by clock**: a `hardware_hash` over the board,
channel-routing and instrument-envelope config. A mismatch means the constants are
**dropped, not applied** — a short blank from a different board silently correcting today's
spectra is the failure this prevents. There is no expiry.

---

## 17. EIS Analysis Engine & Gates

Two analysis engines run side by side, selected by `[eis] engine`:

| Engine | What it does |
|---|---|
| **`legacy`** (default) | fit `R0-CPE0-p(R1,C0)`, take `R1`, divide. What the rig has always done. |
| **`gated`** | admission gates, covariance, per-sample cell constant, upper bounds where the measurement is resolution-limited |

Both return the same report shape, so flipping the key is the whole cutover and it is
reversible per run.

`engine` and `[eis.gates] enabled` are **deliberately separate**. `engine = "gated"` with
`enabled = false` runs every check and logs every verdict *while removing nothing* — which
is how you review a campaign's worth of would-reject decisions before giving thresholds
authority over data. Every gate threshold currently shipped is an engineering default from
the specification, chosen without reference to this rig's spectra — and values for these
thresholds *can* be derived from the spectra a shadow run produces, which is what the review
tool's section 7 does ([§20](#20-shadow-campaign-review)).

### What the gates catch

Admission gates run **before** any fit, because the expensive failure is not a fit that
fails — it is a fit that *succeeds* on a spectrum containing none of the physics being
extracted, and hands the number to a campaign. Every point removed is recorded with a named
gate and a reason; nothing is masked silently.

Two are worth knowing by name:

- **Valley feature** — `R_sol` must come from the interior local minimum of `−Z″`, never the
  `|Z|` minimum (which is the high-frequency intercept ≈ `R_series`). The two can differ by
  more than 10× on the same spectrum, and taking the wrong one has *no other symptom*: the
  fit and residuals both look fine.
- **Cross-spectrum duplicates** — bitwise-identical `|Z|` between independently measured
  spectra is impossible for distinct samples and therefore proves an instrument rail. The
  remedy is a **higher current range, not a lower amplitude**: saturation scales with current
  and bites hardest at the impedance minimum.

### Kramers–Kronig: what the K–K gate actually tests

The Kramers–Kronig relations hold for any response that is **linear, causal, stable and
finite**. They are the one check available that assumes nothing about the circuit — so a
K–K violation says the data could not have come from *any* such system, whatever model you
were planning to fit.

**The catch is that the K–K transform integrates over 0 → ∞.** You measured four decades.
Applying it directly means extrapolating outside the band, and that extrapolation is an
assumption about the very physics under test — you can make most spectra pass or fail
depending on how you close the integral.

So the test is inverted. Rather than transforming the data, fit a basis that is
**K–K-compliant by construction** and read the residuals. Nothing outside the band is ever
referenced. The basis is a **Voigt ladder** — a series of parallel `R‖C` elements with
log-spaced time constants, plus explicit `R`, `L` and (for a blocking cell) `C` terms.

Four properties make that the right basis:

| Property | Why it matters |
|---|---|
| Each `R‖C` is analytically K–K compliant, and compliance is additive | Whatever the fit returns is guaranteed legal — the test can only fail on the *data* |
| A log-spaced ladder discretises the distribution of relaxation times | It approaches *any* physically realisable relaxation response, so it is a basis, not a circuit — this is what makes the test model-free |
| With the time constants fixed, it is **linear in the fitted resistances** | Convex least squares: no initial guess, no local minima, no convergence failure |
| Residuals localise in frequency | A violation says not just *that* the spectrum is inadmissible but *where* |

The linearity is the practical keystone. With a nonlinear fit, a large residual is
ambiguous — non-K–K data, or a stalled optimiser? — and the test becomes uninterpretable
exactly when it matters. You cannot build a falsification test on something that can itself
fail to converge.

**Order selection.** Too few ladder elements and real features read as violations; too many
and the ladder fits the noise, everything passes, and the test is vacuous. The μ-criterion
(`kk_c`, default 0.85) exploits a structural signature: past the optimum, the unconstrained
solution starts producing large **negative** resistances that oscillate to chase noise. The
fit deliberately leaves them unconstrained — physically meaningless, but keeping the problem
linear, and their emergence is the overfitting detector.

**Why `add_cap` follows `[eis.cell] blocking`.** A finite RC ladder cannot produce
`Z → ∞` as `ω → 0`. A blocking electrode's low-frequency capacitive divergence is therefore
*structurally* outside its span, and without the series capacitance term the residuals blow
up at low frequency on every well-behaved blocking cell — which is precisely the region the
gate is allowed to truncate. It would manufacture the evidence it then acts on.

**Three limits worth knowing:**

- **K–K is necessary, not sufficient.** A response that is causal but physically wrong
  passes — a pure series RC and a dispersive dielectric both do. Those are failures of the
  *model*, not of causality, and the topology triad owns them.
- **It cannot separate drift from nonlinearity.** Both break the preconditions. Truncating
  only the low-frequency end is a *physical* argument — that is where the sweep is slow
  enough for the sample to change — bolted onto a mathematical test, not a conclusion the
  mathematics reaches on its own.
- **The residual is global.** Least squares over a shared basis redistributes a local
  perturbation across every element: perturbing 5 of 41 points drove 40 past a 1% residual
  in testing. That is inherent to the method, and it is why `kk_max_truncate_frac` exists —
  past that fraction the spectrum is rejected rather than cut, since the licence to cut
  rests on the cut staying clear of the arc.

### Cell constant and `K_config_factor`

`K = L_gap / ((t − h) · L_stripe)`, computed **per sample** from that sample's own thickness.
A single nominal thickness applied across a series is a defect, not a simplification.

`[eis.cell] electrode_configuration` records how the rig is wired — this one is
**3-electrode**. Three-electrode sensing measures only part of the current path, so in
principle σ = K_geom / (`k_config_factor` · R) with a factor of 2.

> **The factor ships unarmed (`k_config_verified = false`, factor 1.0), which changes no
> number.** The symmetry argument predicts exactly 2.00, but the only direct measurement on
> this rig gives 1.28× and 1.46×, and the derivation is contingent on stripe symmetry and RE
> centring — neither yet verified on this board. Until it is armed, absolute σ reports as
> *scale unqualified*; **relative trends are unaffected**, since a constant factor cannot
> reorder a series. Campaigns ranking formulations are valid now.

#### Arming it takes two checks, and one of them is per sample

Setting `k_config_verified = true` is **necessary but not sufficient** (overhaul R26):

| Check | Scope | How it is recorded |
|---|---|---|
| Stripe symmetry + RE centring | per board, once | `[eis.cell] k_config_verified = true` |
| An ionic path from the film to the RE stripe | **per sample** | `re_contact_verified=True` passed to `cell_constant_for_sample` |

There is deliberately **no `re_contact_verified` config key**. The symmetry derivation assumes
the reference electrode senses a real potential *in a conducting medium*; a dry, dewetted or
non-wetting film does not provide one, and a board-level key would assert contact for exactly
the samples where it fails. Unsupplied means unverified, which holds the factor at 1.0.

What goes wrong without the path is worse than an unknown factor. §3.10 measures the RE
floating onto a **load-dependent capacitive divider**, α = 2.2 to 23.8 — and not reproducible
even at a fixed load (9.85 and 4.96 for the same 1 nF part). The ratio is not merely
unmeasured; it is undefined, so there is nothing to apply.

> **`tied_to_ce` is the trap.** Jumpering RE to CE closes the control loop perfectly — the
> quadrant gate will correctly call violations instrument-side — while making the RE read the
> counter electrode. The measurement *is* two-electrode, so its factor is 1. This is why
> `RE_IONIC_CONTACT` is a strictly smaller set than `RE_CLOSED_LOOP`: "the loop is closed" and
> "ions reach the reference" are different questions, and confusing them costs a clean 2× with
> a perfect-looking fit.

If the two records contradict each other — contact asserted alongside `open_by_geometry` — the
resolution **fails closed and logs**. One of them is wrong and guessing which is not safe.

### Fixture correction

A measurement made through a mux, a ribbon and a PCB trace records the cell *and the path to
it*. `[eis.fixture]` subtracts the path. Gated engine only — correcting the legacy path would
break the parity that makes the two engines comparable.

| `mode` | Behaviour |
|---|---|
| **`auto`** (default) | `none` until this fixture has a short blank, `series` the moment it does |
| `series` | subtract `R_short + jωL_lead`; refuses, with a reason, if no short blank exists |
| `none` | subtract nothing |

`auto` is the point of the design: correction switches on by itself after commissioning, and
switches back off by itself after a board swap, because a `hardware_hash` mismatch has already
dropped the constants. Nobody has to remember either direction.

> **Series-only, on purpose — OSL is not implemented.** Open/short/load correction is the
> obvious richer option and this rig has evidence against it: overhaul F6 records it corrupting
> *every* spectrum on this fixture, mean error 32%, with one channel reading 1.26 MΩ against a
> true ~840 Ω. The open blank is still worth measuring, because an **unusable** open is the
> positive evidence that shunt admittance is negligible — which is exactly when short-only
> correction is *exact*. The open selects the fallback; it is never applied. `auto` therefore
> declines OSL even when the artifacts would license it, and says so in the log.

**Where it runs in the pipeline** matters and is not obvious. Framework §6 places the
correction at step 4, *between* the admission gates and the rest:

| Runs on the raw instrument record | Runs on corrected data |
|---|---|
| finiteness, monotonic-f, quadrant, magnitude window, phase noise, stuck instrument | HF inductive truncation, min-points, **topology triad**, valley feature, and the fit |

Both directions are load-bearing. The admission gates ask *"did this measure anything
real?"* — a railed point stays railed however much lead you subtract, so a correction must
never be able to rescue a failed measurement. Everything downstream asks *"does this
spectrum contain the physics being extracted?"*, which is only answerable once the
fixture's own contribution is gone: a fixture `R_short` is a **series parasitic**, and a
series parasitic is exactly what the loss-tangent slope test reads to decide a spectrum has
no parallel conduction. Judging the triad on uncorrected data lets the fixture masquerade
as the sample's own physics.

A spectrum rejected at admission is never corrected at all — every verdict downstream would
otherwise describe a measurement already known to be inadmissible.

**A wrong constant announces itself.** A subtraction can't be checked by looking at its output
— a wrong `R_short` yields a merely shifted spectrum, and shifted spectra look fine. So two
things happen instead. Any point the correction drives to `Re Z ≤ 0` that was physical
beforehand marks the spectrum SUSPECT with a stated reason, since a small series impedance
cannot do that. And `validate_load_blank` pushes a resistor of independently known value
through the correction end to end — the only real check available, because re-measuring the
short proves nothing about constants the short itself produced.

Pick that load to be *comparable to the fixture*. A 6 Ω correction validated against a 1 MΩ
reference passes no matter what: the error is below the noise. Against 100 Ω the same 6 Ω
shows up as a 6% miss.

Every analysed spectrum gets a `fixture_corrections` row — including a declined one, with its
reason. **An absent row means uncorrected**, which is the honest reading of every measurement
taken before this existed; a nullable column with a default would have had to claim something
about the past.

Set `fixture_id` to match what you commission. `softae-commission` takes its `--fixture`
default from this same key, so the two cannot drift into commissioning one fixture while the
engine looks for another.

---

## 18. Unattended Operation & Safety

### Hardware interlock

Headless commands **cannot arm real motion hardware on their own**. With real stage, syringe
or piezo instruments present, a workflow refuses to execute unless the operator has armed the
rig deliberately:

```bash
export SOFTAE_ALLOW_HARDWARE=1        # bash
$env:SOFTAE_ALLOW_HARDWARE = "1"      # PowerShell
```

It is session-scoped and conscious by design. Launching the desktop GUI arms the process
itself, since that is already a human-driven act. Mock managers never trip the interlock.

### Dispenser head state

The head's up/down state cannot be read back from the hardware, so it is **asked at startup**
and gates stage motion. The answer is authoritative — it is *not* overwritten by connecting
or reconnecting instruments — and both `softae-campaign` and the GUI re-confirm it before a
run that will move the stage.

### Fault handling: retry, then park

A failing trial is retried; a fault that looks systematic **parks** the run rather than
continuing to consume electrodes. A parked run keeps its checkpoint — being able to resume
after a park is exactly why the checkpoint exists.

An unmeasured trial is *never* told to the optimizer as a number. `None` means "not
measured", and a fabricated `0.0` would make the surrogate confident about a point nobody
observed.

### Consumables and preflight

Stock levels are tracked in a ledger and projected before a run: a campaign that cannot
finish on the declared stock says so up front, alongside its projected duration and waste
accrual. **Undeclared is "unknown", never "empty"** — an undeclared reservoir will not be
silently treated as full or as exhausted.

### Anti-clog purge

`[purge]` schedules purge windows against particulate lines. It ships with
`actuate = false`: the harness plans and logs every window it *would* run without moving a
pump, so the cadence can be observed against real runs before it is armed at the bench.

### Alerts

Long-running campaigns record alerts (parks, gate timeouts, board exchanges, stock warnings)
to the DataStore so an unattended run leaves an audit trail rather than only a log file.

---

## 19. Extending: a New Measurement Modality

A **modality** is a kind of measurement — EIS today, a camera or any other stream tomorrow.
The campaign path performs exactly one lookup (`get_modality(spec.measurement.modality)`), so
a new stream registers in one place instead of being threaded through the step builder, the
router, the objective table and the run-preparation block separately.

```python
from softae.core.modality_registry import Modality, ModalityDisplay, register_modality

register_modality(Modality(
    name="my_stream",
    build_measure_step=...,  # (channel, spec) -> the per-channel step, or None
    router_factory=...,      # () -> the router that persists what comes back
    objectives={},           # what the optimizer may be told; may be empty
    prepare_run=...,         # runs once before any measure step (EIS writes .mscr here)
    display=ModalityDisplay(display_name="My Stream"),  # static, GUI-facing metadata
))
```

Three points that are easy to get wrong:

- **Registration is an explicit call, never an import side effect.** A modality that appeared
  merely because a module happened to be imported would make the set of available modalities
  depend on import order, and a half-written module would read as a *missing capability*
  rather than an error.
- **An analysis-only stream is a first-class modality.** `objectives = {}` is legitimate: a
  stream that feeds the optimizer nothing still gets stored, tagged and joined to its sample.
  Tagging its steps `measurement = "image"` (anything but `primary`) also keeps it out of the
  loop-closure predicate, so it cannot accidentally be optimised against.
- **The sample spine comes for free.** Any step carrying `tags["channel"]` inherits its
  `sample_uuid`, so a new modality's payloads join to the formulation and occupancy rows
  without any wiring of its own.

A worked example ships in `src/softae/analysis/image/` — a camera-backed, analysis-only
modality built with **no edit to `core/`, `workflows/`, `analysis/eis/` or `drivers/`**. One
workflow can carry an image step and an EIS step and each result lands in its own router.

*(coming soon)* Three edges of this seam are built but not yet reachable from a shipped
campaign:

| Not yet | Why |
|---|---|
| Running the `image` modality | `register_image_modality()` is deliberately not called at startup — one line of `core/` wiring is left open for whoever ships the first image campaign |
| A `measurements` **row** for a non-EIS capture | the write path is still EIS-typed and the readers do not filter on `modality`, so an image row would surface to the Analysis browser as a spectrum with NULL fields. The payload self-links via its `attrs` in the meantime |
| A modality-neutral router contract | `ResultRouter` / `RouterContext` still live inside `analysis/eis/`; a second modality duck-types rather than importing the EIS package |

---

## 20. Shadow Campaign Review

A **shadow campaign** runs the gated physics engine with every data-quality gate *observing
rather than enforcing*: nothing is rejected, but everything that would have been is recorded.
It is how the gated engine and the quality gates earn their cutover — on real spectra, at no
risk to a run. `softae-shadow` sits on either side of that run — and, with `rehearse`, well
before it.

```bash
softae-shadow status                                                 # is the rig armed?
softae-shadow rehearse --dry-run                                     # what will a run cost?
softae-shadow review shadow_run.log \
    --project ./runs/aug --run-id run_20260810T1400Z                 # what did it see?
softae-shadow review shadow_run.log --project ./runs/aug \
    --emit-toml proposed_thresholds.toml                             # where would they sit?
```

> Equivalently `python -m softae.tools.shadow_review …` — note the module is **`shadow_review`**,
> not `shadow`. The console script exists here as of the 2026-08-11 editable install; the module
> form resolves regardless ([§5](#5-cli-workflow-runner)).

**`status`** is read-only and answers one question — *is the config armed for a shadow run?* —
which you ask **twice**: before the run ("did the flip take?") and after the revert ("is the rig
back to shipped?"). It prints the three keys that matter and one verdict, and the exit code
carries the same verdict for scripting:

| Verdict | Config state | Exit |
|---|---|---|
| **ARMED FOR A SHADOW RUN** | `[eis] engine = "gated"`, `[eis.gates] enabled = false`, `[quality] enabled = false` | 0 |
| **GATED AND ENFORCING** | engine gated but a gate is enforcing — this is a *cutover*, not a shadow run | 1 |
| **NOT ARMED** | the shipped legacy engine | 2 |

> **`status` also sizes the run.** When the config is armed, the screen ends with a wall-time
> advisory, because observe-only is the **slowest** analysis setting the rig has and the one that
> reads as the cheapest. A spectrum the gates would have rejected pre-fit still reaches the
> fitter, and **a fit with no arc to find takes the long way to failing** — the cost is set by
> **arc closure**, not by the gate verdict and not by the engine: an open arc has no in-band
> feature for the fitter to converge onto. The screen quotes measurements, not estimates. Over
> **192 real spectra (2026-08-14)**: open-arc median **~38 s**, max **~58 s**, against closed-arc
> **~0.16 s**. **Size the run by the clock, not by the well count** — and run `rehearse` first,
> because the open-arc *mix* is what sets the total and it is a property of the material.

### `rehearse` — a dress rehearsal on spectra you already have

A bench shadow run is single-shot: it spends half a board, and the two things you most want to
know beforehand — *what will this cost in wall time?* and *what will the review actually say?* —
are only answerable afterwards. **`rehearse` answers both in advance**, by replaying stored
spectra through the very same gated observe-only engine.

```bash
softae-shadow rehearse --dry-run                            # the plan and the projected duration
softae-shadow rehearse                                      # → logs/rehearsal_<UTC>.log
softae-shadow review logs/rehearsal_<UTC>.log --project ~/softae_data
```

It is a **replay, not a simulation**: the same `analyze_spectrum`, the same gates, the same
structlog stream, so the log it writes is one `softae-shadow review` reads with no special case
at all. Selection is **stratified and deterministic** — a cell is `(leg, setpoint, channel)`, and
the default takes 2 rounds from each of them, spaced across the round axis. A convenience slice
would sample one block and report the fast mode as the whole distribution; stratifying makes the
open-arc mix a *measured* quantity.

**Three read-only guarantees, structural rather than promised:**

| Guarantee | How |
|---|---|
| No database write | The corpus is opened `sqlite3.connect("file:…?mode=ro", uri=True)`. `DataStore` is never constructed, so its `mkdir`/DDL/migrate/commit path never runs, and `record_fit` is never called |
| No config edit | The gated engine is chosen by a `settings=` **argument**. `[eis] engine` is never read and never written — `softae-shadow status` says the same thing after a rehearsal as before |
| No rig | Analysis modules only. No instrument is opened, no pose read, no stage moved |

| Flag | Default | Behaviour |
|---|---|---|
| `--project DIR` | `[data] project_dir` from the config loader | Where the corpus lives |
| `--run-id ID` | most recent run with spectra | Which run to replay; the plan line names it, so a wrong default shows in the first line rather than in the totals |
| `--rounds N` | `2` | Rounds per cell |
| `--all` | — | Every spectrum in the run |
| `--limit N` | *(none)* | Hard cap applied **after** stratification, so a cut is a prefix of a balanced plan; the summary reports the cells it dropped |
| `--seed S` | *(deterministic)* | Randomise the round picks for a sensitivity check. Without it two rehearsals of one corpus compare line by line |
| `--out PATH` | `logs/rehearsal_<UTC>.log` | The log. Refuses to overwrite, like `--emit-toml` |
| `--tee` | off | Mirror to stdout for a watched run |
| `--model NAME` | the fit row's `model_name` | Override |
| `--enforced` | off | Replay with the gates **enforcing**, to measure what observing costs |
| `--dry-run` | off | Print the plan and the projected duration; analyse nothing |

One consequence of `--enforced` worth carrying into campaign design: under enforcing gates a
rejected spectrum never reaches the fit that would annotate its arc, so an enforcing campaign
**cannot report arc closure for the spectra it rejected** — their `arc_state` stays NULL — which
matters to any analysis that selects on closed arcs.

**Two outputs.** The **log** carries the engine's own events (`eis_spectrum_metrics`,
`eis_gate_would_reject`, `eis_gate_points_dropped`) interleaved with the rehearsal's own
(`rehearsal_started`, `rehearsal_spectrum_done`, `rehearsal_summary`), so an hours-long run is
observable while it runs. The tool **owns the file handle** rather than relying on shell
redirection — opened `utf-8`/`errors="replace"`, because a gate detail containing `tan δ` will
otherwise kill the run on a cp1252 console, on its first interesting spectrum. Beside it sits a
**timing CSV** (`<out>.timing.csv`), one row per analysed spectrum with `seconds`, `verdict`,
`arc_state`, `sigma_mode` and provenance. It is written **incrementally**, so a rehearsal
interrupted at spectrum 140 still leaves 139 rows of evidence.

> **When to run it.** Before any bench shadow run, and again after a recalibration — a new
> calibration set moves the envelope every gate is measured against. The 2026-08-14 rehearsal
> measured **192 spectra in 36 m 42 s**, median **0.46 s** but P90 **39.45 s**: the cost is
> **bimodal**, closed arcs at 0.16 s against open arcs at 38 s, and the *mix* sets the total. In
> analysis alone that brackets a 16-well bench run at **2.5 s → 4 min → 10 min** and a 32-well
> run at **5 s → 8 min → 20 min** (all-closed floor → measured mix → all-open ceiling). Read the
> brackets: the mix is a property of the material, and the bench campaign casts something else.
> Full figures and caveats in `docs/SHADOW_CAMPAIGN.md` §5.

> **A rehearsal's section 7 is evidence about the recommender, not thresholds for the rig.**
> Replaying an equilibration corpus tells you whether the rules behave on a real distribution —
> the first such run found two that did not. It does not tell you where *this* campaign's gates
> belong. Nothing from a rehearsal is pasted into `softae_config.toml`.

**`review`** summarizes a run's would-reject verdicts: how many spectra would have been
rejected, by which gate, and on which channel. `--project` adds the DataStore half —
measurements per channel, the stored σ, and the two columns `fit_results` records honestly
(§5b below) — and `--run-id` picks the run (default: the most recent in that project).

| Flag | Default | What it adds |
|---|---|---|
| `--project DIR` | *(log only)* | The DataStore half: section 5 (per-channel σ) and section 5b (railed fits, arc closure) |
| `--run-id ID` | most recent run | Which run in that project to read |
| `--min-evidence N` | `20` | Spectra a metric must be observed on before section 7 may propose a value for it |
| `--emit-toml PATH` | *(not written)* | Write the paste-ready `[eis.gates]` / `[quality]` block to a **new** file |

> **The log file is the artifact, and it exists only if you made it.** Gate verdicts are
> **not persisted**: a gated campaign still writes `gate_verdict = NULL` to `fit_results`, so
> the only record of a would-reject is the **structlog stream on the console**. Redirect it or
> lose it — `... | tee shadow_run.log` is a required step of the run, not a convenience. Pass
> `-` as the log argument to read stdin instead of a file.

Two attribution limits the report states rather than smooths: per-*gate* counts are exact (the
gate name heads every issue string), while per-*channel* counts are **positional** and are sound
only for verdicts emitted during the workflow's auto-fit. Verdicts emitted during objective
extraction land on whichever channel was routed last; the summary counts those separately as
unattributed rather than misattributing them. That limit is the *verdict* event's, and section 4
is where it lives: the `eis_spectrum_metrics` event behind section 7 carries `channel` and a
content fingerprint outright, so its population is attributed and de-duplicated rather than
inferred.

### Section 5b — what `fit_results` records honestly

Section 5 has to caveat almost every column it prints, because the router stamps defaults into
most of them (`engine='legacy'`, `gate_verdict = NULL`). **Section 5b prints the two things the
rows do record honestly**, and it appears whenever `--project` is given:

- **Railed fits, per channel, across both eras.** A fit that came to rest on the model's own R₁
  floor rather than on the data is detectable from the stored numbers whatever wrote them —
  but by *two* detectors, because the eras differ. Rows written since the railed-fit demotion
  landed carry `success = 0` and an `error_msg` naming the bound (**railed (new)**); rows
  written before it carry `success = 1` with `R1` sitting exactly on the bound and nothing
  marking them (**railed (historical)**) — a σ of roughly seawater from a dry film, wearing a
  success flag. The bound is read from `CIRCUIT_MODELS`, never restated; a model declaring none
  reports `unknown` rather than `0`.
- **Arc-closure state counts.** Since T7.7 the verdict is a real column — `record_fit` writes
  `arc_state` with `arc_f_peak_hz` / `arc_f_low_hz` / `arc_phase_low_deg` beside it, from the
  fit itself. They are NULLable with **no default**, because NULL ("never annotated") and
  `'unknown'` ("looked, and could not tell") are different facts. Older rows carried the same
  verdict as a `gate_log_json` entry from the router's arc-provenance shim, so section 5b reads
  **column first, JSON fallback**, counting each row exactly once. Rows predating the shim are
  counted as **no record** rather than folded into an outcome they never reported.

`sigma_is_bound` keeps section 5's posture and is labelled a **stamped default** — 0 on every
row, not an observation — until P.18 passes the real report to `record_fit`.

### Section 7 — recommended thresholds

Whether to arm a gate is a scientific claim, and no arithmetic establishes it. But **where** a
threshold would sit, given the decision to arm, is a percentile of a distribution — and doing
that by eye across fourteen keys and hundreds of spectra is exactly the work that gets skipped.
Section 7 computes the values and never decides — for what each key *means* before you move it,
see [§17](#17-eis-analysis-engine--gates):

```
7. RECOMMENDED THRESHOLDS
   Evidence: 61 spectra (122 events, deduplicated by content fingerprint).
   ! = changing this key changes a stored NUMBER, not only a verdict.

   section     key                default  recommended  rule         n   fired@def  rej@rec  status
   ----------  -----------------  -------  -----------  -----------  --  ---------  -------  -----------------
   eis.gates    tand_slope_max       -0.3            -  gap          61          0        0  hold (unimodal)
   eis.gates    cap_flatness_max     0.15         0.41  upper-fence  61         22        3  recommended
   eis.gates   !kk_resid_pct            1          2.6  upper-fence  58         31        2  recommended
   eis.gates    min_fit_pts              8            -  count       61          0        0  hold (unexercised)
   quality      min_r_squared        0.95        0.905  complement   47          9        2  recommended

   REFUSED (evidence insufficient — the default stands):
     eis.gates.max_rel_se
       only 4 spectra carry rel_se_measurand (need 20); below 20 the P95 is a single observation
   ...
   ⚠ ARMING IS NOT RECOMMENDED HERE. These are values, not a decision.
   Applying every 'recommended' value above (3 key(s)) would reject 7 of 61 spectra (11%).
   Per-key counts do not add: one spectrum routinely fails several gates.
```

Four things to read off it:

| Column / marker | Means |
|---|---|
| **`!`** before a key | **Behavioural** — changing it moves a stored *number*, not only a verdict. Re-fit before trusting any σ produced under it |
| `fired@def` / `rej@rec` | How many spectra the gate fired on at its shipped default, and how many the proposed value would reject |
| `hold (unexercised)` | 0 spectra failed this gate at its default. *Untested is not validated* — the default stands |
| `hold (unimodal)` | A gap rule found no two populations to separate, so the theory-anchored default stands |
| `recommended (measures-the-rig)` | The gate fired on ≈every spectrum (≥ 90 %): at its default it is measuring the rig, not the sample |

The final **joint** count is the number to act on. Per-key counts do not add — one bad spectrum
routinely fails several gates, so summing the column overstates the cost, sometimes by more than
the population size. Two further blocks follow the table and are stated rather than omitted:
**not recommendable from a shadow run** (`kk_c`, `bound_tol`, the `blank_*` and `geom_*` keys,
one reason each) and **observed but unconfigurable** — hardcoded constants such as
`cross_check_pct` and `runs_z` that fire in practice but have no config line to move, reported
with their P50/P95 and fire count as evidence *for* a future key.

> **A pre-T7.1 log recommends nothing, and says so.** The pass-side distribution comes from the
> `eis_spectrum_metrics` event, which the gated engine emits once per spectrum on **both**
> verdict paths. Older logs recorded `metrics=` only where a spectrum *failed*, so every key
> refuses with that one-sided-evidence reason. Sections 1–6 of such a log render exactly as they
> always did.

### The `--emit-toml` workflow

Four steps, and the tool performs exactly the first three:

1. `softae-shadow review shadow_run.log --project <dir>` — read sections 1–6 gate by gate, as
   `docs/SHADOW_CAMPAIGN.md` §8 asks.
2. Read **section 7** for where each threshold would sit, and what it would cost.
3. `--emit-toml proposed_thresholds.toml` — writes the paste-ready block.
4. **You** paste what you accept into `softae_config.toml`, and **you** decide arming.

The emitted block is auditable rather than merely pasteable: every value carries its rule, `n`,
and its fired → rejected counts as a trailing comment, and **refused and held keys are emitted
commented out with their reason**, so pasting the block can never silently apply a
non-recommendation. `enabled` is written `false` in both sections and is never written otherwise.

> **Two refusals, both absolute.** `--emit-toml` will not write to the **live config** — arming
> is a decision taken by reading the would-reject table, not by running a command that happens
> to write a file — and it will not **overwrite** an existing path, because the one file an
> operator would aim it at twice is the one holding the previous run's proposal. Either refusal
> prints the reason and exits 1.

### The evidence floor: a 16-well run recommends nothing

`--min-evidence` defaults to **20** spectra per metric, and that number is chosen rather than
inherited: at *n* = 20 the empirical P95 is the second-largest observation, so a fence rests on
two points instead of entirely on the single worst spectrum of the run. Below the floor, a key
is **refused by name with a reason** rather than given a value.

The packaged shadow spec (`examples/shadow_campaign.toml`) ships `budget = 16`, which sits
**deliberately below** the floor. A 16-well run therefore recommends nothing — that is the
design, not a failure. **Raise `budget` to 32** if you want the run to produce thresholds.
Lowering `--min-evidence` instead is possible and visible in the output, but it buys a number
the sample does not support.

The full bench procedure — which keys to flip, in what order, and how to revert — is
`docs/SHADOW_CAMPAIGN.md`. `status`, `rehearse` and `review` are the whole invocation surface.

---

## 21. Thickness Series

Film thickness is a variable like any other, and like any other it can be **confounded**. If
levels are assigned in channel order at cast time — CH27/28 at 200 µm, CH29/30 at 150, CH31/32
at 100 — then a channel artifact and a thickness effect become mathematically indistinguishable,
and no later analysis can separate them. `softae-thickness` exists to plan the assignment
*before* casting, because a tool that only recorded measurements would have recorded that series
faithfully and said nothing.

**Plan before you cast. Confounding cannot be undone afterwards.**

```bash
softae-thickness plan --levels 100,150,200,250 --channels 1-32
softae-thickness cast --plan geo-2026-08-06                 # DRY RUN
softae-thickness cast --plan geo-2026-08-06 --execute       # drives hardware
softae-thickness record --channel 7 --um 148.2 --uncertainty 3.0
softae-thickness check --plan geo-2026-08-06                # still unconfounded?
softae-thickness fit  --plan geo-2026-08-06                 # sigma from the slope
softae-thickness list --plan geo-2026-08-06
```

> Equivalently `python -m softae.tools.thickness …`. The console script exists here as of the
> 2026-08-11 editable install; the module form resolves regardless
> ([§5](#5-cli-workflow-runner)).

Six subcommands, in the order they are used. `--project` (project directory, default
`[data] project_dir`) is accepted by every one of them.

| Subcommand | Flags | When you reach for it |
|---|---|---|
| **`plan`** | `--levels` *(req)*, `--channels` *(req)*, `--id`, `--seed`, `--max-correlation`, `--notes` | Before casting. Assigns levels to channels so level and channel index are uncorrelated |
| **`cast`** | `--plan` *(req)*, `--board`, `--execute`, `--no-drift-control` | At the rig. Resolves a plan into a cast order and checks the board |
| **`record`** | `--channel` *(req)*, `--um` *(req)*, `--uncertainty`, `--plan`, `--run`, `--level`, `--instrument`, `--operator`, `--notes` | At the profilometer, one channel at a time |
| **`check`** | `--plan`, `--run`, `--max-correlation` | After casting. Compares what was cast against what was planned |
| **`fit`** | `--plan`, `--run`, `--fixture` | Geometry-series fit: σ from the slope, and *h* if `G_fixture` exists |
| **`list`** | `--plan`, `--run`, `--plans` | Show measurements, or `--plans` to list the plans themselves |

Three flags worth singling out:

> **`cast --execute` drives real hardware.** Without it, `cast` is a **dry run** that resolves
> the order and checks the board and touches nothing. Add `--execute` only when you mean to
> actually cast — and see [§18](#18-unattended-operation--safety) for the interlock that any
> real-motion command additionally requires.

- **`--no-drift-control`** omits the end-of-session repeat cast. That repeat is what separates a
  genuine thickness trend from session drift, so dropping it trades away the ability to tell
  those apart. Reach for it only when bench time or board area genuinely will not stretch.
- **`--seed`** (default 0) makes the assignment reproducible — the same seed and inputs give the
  same plan, so a plan can be regenerated and audited rather than merely trusted.
- **`--max-correlation`** is the |r| ceiling between level and channel index that `plan` designs
  under and `check` tests against. A confounded series exits **3**, distinct from a plain
  failure (1), so a script can branch on it.

`check` is not optional ceremony: a sound plan followed inattentively produces exactly the
dataset the plan existed to prevent. Run it after casting, while a re-cast is still cheap.

---

## 22. Equilibration Characterization

How long must a film be held at a setpoint before its conductivity is *the* conductivity, and
not a number still relaxing toward one? `softae.tools.equilibration` answers that empirically:
it records σ(t) while the chamber is brought to condition, fits the relaxation, and derives the
conditioning hold time from the fit. This is an overnight bench run the operator starts and
walks away from.

### Invocation

```bash
softae-equilibration plan --save plan.toml
softae-equilibration run --from-plan plan.toml --channels 1-16 --execute
softae-equilibration fit    --run <run_id>
softae-equilibration report --run <run_id> --tol-rel 0.02
```

> **`python -m softae.tools.equilibration …` is the exact equivalent**, and is what the tool
> itself prints in its suggested commands: a console script is generated only by an install,
> so the module form resolves whether or not one was. `softae-equilibration` was module-only
> until the **2026-08-11** editable install generated it; it resolves here now (verified), and
> the arguments are identical either way.
>
> The tool's `--help` **epilog is stale on this one point** — it still says the console script
> "is not installed in this venv", a sentence written before that install. Its recommendation
> (use the module form) is unaffected and remains correct.

### `plan` and `run` share no state — this has already cost a run

`plan` and `run` are separate process invocations. **Every design flag not repeated on `run`
silently reverts to its default.** On 2026-08-10 that cost ~40 minutes of rig time and the whole
scientific result: `--preset` fell back to `Standard` (40.7 s/channel measured, against
`Quick`'s 10.47) and the electrode geometry was dropped whole, so `sigma_S_per_cm` came back
NULL for all 41 fits — while every log line reported success.

Two things answer that, and both are worth using:

- **`plan --save plan.toml` writes the fully resolved design** — every value the run will use,
  defaults included — which **`run --from-plan plan.toml`** then executes verbatim. A flag typed
  alongside `--from-plan` still wins, but only as a **printed diff** against the file and
  repeated in the thermal confirmation. Silent override was the original defect; a loud one is
  fine.
- **`run --channels` is MANDATORY.** It has no default, deliberately: a defaulted channel set
  would energise exactly the channels a subset was chosen to exclude. `--from-plan` supplies it,
  so the flag is only required when you are not using a plan file.

### The flag surface, by purpose

The surface is large; group it rather than memorise it. **Design** flags appear on both `plan`
and `run` (that is what makes a plan file replayable); **execution** flags are `run`-only;
**analysis** flags are `fit`/`report`-only. `--project` and `--mock` are available throughout.

| Group | Flags | What it decides |
|---|---|---|
| **Design** *(`plan` + `run`)* | `--channels`, `--temperatures`, `--legs`, `--rh`, `--rounds`, `--preset`, `--round-period-s`, `--measured-per-channel-s`, `--circuit-model`, `--electrode-l-cm` / `-t-cm` / `-w-cm`, `--thickness-method`, `--fixture` *(plan only)* | Which channels, which setpoints, how the spectra are taken and turned into σ |
| **Settling** *(`plan` + `run`)* | `--settle on\|off`, `--settle-tol-rel`, `--settle-n-rounds`, `--settle-min-channels`, `--min-hold-first-s`, `--min-hold-s` | When a setpoint has been held long enough to stop |
| **Execution** *(`run`)* | `--from-plan`, `--execute`, `--yes` / `-y`, `--quiet`, `--telemetry-interval-s` | Whether hardware moves, and how loudly the run reports |
| **Analysis** *(`fit`, `report`)* | `--run` *(req)*, `--relaxation-model`, `--tol-rel`, `--n-settle` | How σ(t) is fitted offline and which tolerance the verdict uses |

Reading the groups:

- **`--rounds` is a ceiling, not a count.** A setpoint stops as soon as σ has settled, the hold
  floor has elapsed, and the fitter's minimum number of rounds has run. It reaches `--rounds`
  only when it has not settled.
- **`--settle-tol-rel` must exceed the run's own noise floor**, or no hold length can satisfy it
  and every setpoint runs to its ceiling. Measured here: 5.98 % median over 96 series, with 22
  of them above 20 %. The run says so, per setpoint, when the criterion is unsatisfiable.
- **`--settle on|off`** rather than `--no-settle`, because a `store_true` cannot be written into
  a plan file and retyped from it. `off` restores fixed-count behaviour exactly.
- **`--execute` is what makes anything real.** Without it `run` opens nothing — the default is a
  dry run. `--yes` skips the thermal confirmation prompt, and `--quiet` drops only the live
  status line (milestones, hold verdicts and telemetry still print, and everything still reaches
  structlog).
- **`--circuit-model` vs `--relaxation-model`.** Two different vocabularies that once shared the
  spelling `--model`. The circuit model (e.g. `simpleSalt`) is fitted to each *spectrum* on
  `plan`/`run`; the relaxation model (e.g. `exponential`, or `none` for t_tol only) is fitted to
  *σ(t)* on `fit`/`report`. `--model` survives as a working alias on both, meaning whichever is
  right for that subcommand.
- **`-v` / `--verbose`** works on the top-level parser *and* on every subcommand, so
  `... -v run ...` and `... run -v ...` both take. It is genuinely noisy — the RH controller logs
  a duty cycle on every update.

---

*Generated for SoftAE v0.1.0 — last revised August 2026*
