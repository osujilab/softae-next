# SoftAE User Guide

Soft-matter Autonomous Experimentation Platform · v0.1.0 · Python ≥ 3.11 · PySide6

**Before you start.** The repository ships no process catalog: `data/` (`tasks.toml`,
`recipes.toml`, `chemicals.csv`, `solutions.csv`) is gitignored by policy. Build one in Process
Studio (Tab 13, writes the TOMLs) and Catalogs (Tab 11, writes the CSVs), or by hand against
`core/task_catalog.py`, `core/recipe_registry.py` and `core/formulation.py`. A missing catalog
file loads as an empty catalog rather than raising, so catalog-loading tests are red on a fresh
clone by construction.

`softae-catalog init` seeds `data/` with a **placeholder** catalog — the four files, generic
chemistry, illustrative numbers — so the specs under `examples/` compile and each file's shape
is visible. It is a demonstration, **not a configuration for any rig**: no ports, no calibrated
volumes, no cure recipe of anyone's, and no guarantee it runs on your hardware as shipped. It
refuses to overwrite an existing catalog, naming the file and writing nothing. Copy it, then
edit your own copy under `data/`. `--dest DIR` seeds somewhere other than the configured data
root.

## Contents

1 [Installation](#1-installation) · 2 [Configuration](#2-configuration) · 3 [Launching the GUI](#3-launching-the-gui) ·
4 [Tab reference](#4-tab-reference) · 5 [Command-line tools](#5-command-line-tools) ·
6 [`softae-run` and workflow YAML](#6-softae-run-and-workflow-yaml) · 7 [EIS analysis API](#7-eis-analysis-api) ·
8 [Stopping, safe exit, interlock, head state](#8-stopping-safe-exit-interlock-head-state) ·
9 [Errors and troubleshooting](#9-errors-and-troubleshooting) · 10 [DataStore](#10-datastore) ·
11 [Deposition twin and catalogs](#11-deposition-twin-and-catalogs) · 12 [`softae-campaign`](#12-softae-campaign) ·
13 [`softae-commission`](#13-softae-commission) · 14 [EIS engine, gates, cell constant, fixture correction](#14-eis-engine-gates-cell-constant-fixture-correction) ·
15 [`softae-shadow`](#15-softae-shadow) · 16 [`softae-thickness`](#16-softae-thickness) ·
17 [`softae-equilibration`](#17-softae-equilibration) · 18 [`softae-env`](#18-softae-env) ·
19 [`softae-eis-validate`](#19-softae-eis-validate) · 20 [Adding a measurement modality](#20-adding-a-measurement-modality) ·
21 [Documentation site](#21-documentation-site)

---

## 1. Installation

Python 3.11+, Git. Optional for hardware: NI-DAQmx runtime, ThorLabs TSI SDK, PalmSens SDK.

```powershell
cd softae-next
python -m venv .venv
.venv\Scripts\activate

pip install -e .              # core: the 13 headless console scripts + analysis + DataStore
pip install -e ".[gui]"       # + PySide6, opencv, qasync — required for the desktop GUI
pip install -e ".[dev]"       # + test/lint tools (self-references [gui])
pip install -e ".[hardware]"  # + NI-DAQ, Blinka, HID for real instruments
pip install -e ".[web]"       # + Dash/Plotly for softae-web
pip install -e ".[docs]"      # + mkdocs, mkdocs-material, mkdocstrings[python]
```

Only `softae-gui`, `softae-deposition` and `softae` (the `gui_scripts` launcher) need `[gui]`. An
existing venv keeps its PySide6 until it is rebuilt, so verify the headless path in a new venv.

Verify with `softae-gui --help` (needs `[gui]`), `softae-run --help`, `pytest tests/ -v` (needs
`[dev]`).

---

## 2. Configuration

Everything configurable lives in `softae_config.toml` at the repository root. The loader searches:

1. Explicit path passed to `config.load(path=...)`
2. `SOFTAE_CONFIG` environment variable
3. `softae_config.toml` in the current working directory
4. `softae_config.toml` in the package install root — under an editable install, the repo root

### Sections

| Section | Purpose | Keys |
|---|---|---|
| `[paths]` | SDK/DLL locations, catalog root | `thorlabs_dll`, `palmsens_sdk`, `data_root` |
| `[data]` | Project store | `project_dir`, `db_filename`, `auto_save_eis` |
| `[instruments.*]` | Per-instrument connection | `driver`, `port`, `baud`, address |
| `[pcb.*]` | Board layouts | `channels`, grid, spacing, electrode dims |
| `[eis_presets.*]` | EIS sweep presets | `npts`, `f_hi`, `f_lo_mHz`, `mv_ac` |
| `[channel_routing]` | Channel → potentiostat | `pico1_range = [1,16]`, `pico2_range = [17,32]` |
| `[piezo]`, `[piezo.liquid_events]` | Piezo defaults, event profile | `enabled`, `frequency_hz`, `sweep_on_s`, `sweep_rest_s`, `channel_a`, `settings_source` |
| `[safety]` | Limits, timeouts, watchdogs | see below |
| `[deposition]`, `[dropcast]` | Cast engine defaults | `evaporation_pct`, flush + proportional rate |
| `[liquid_handling]`, `[liquid_handling.line.<id>]` | Volume correction | `enabled`, `beta`, `eta_ref_mpas`, `cracking_kpa_per_valve`, `compliance_uL_per_kpa`, `alpha_base`, `viscosity_mpas` |
| `[quality]` | Measurement grading | `enabled`, `max_residual_pct`, `max_abs_z`, `min_r_squared` |
| `[purge]` | Anti-clog purge | `actuate`, cadence, particulate pump |
| `[eis]` | Analysis engine + campaign objective | `engine`, `objective` |
| `[eis.gates]` | Admission-gate thresholds | `enabled`, `min_fit_pts`, `kk_resid_pct`, `kk_c`, `rho_degenerate` |
| `[eis.instrument]` | Measured instrument envelope | `phase_noise_deg`, `z_max_ohm`, `max_amplitude_mV` |
| `[eis.cell]` | Geometry + electrode config | `L_gap_cm`, `dead_height_um`, `electrode_configuration`, `k_config_verified`, `blocking` |
| `[eis.fixture]` | Fixture correction | `mode`, `fixture_id`, `load_tolerance_pct` |
| `[stage_calibration]` | Stage origin / skew | persisted by Tab 1 |
| `[logging]`, `[web]`, `[webcam]` | Log level, optional services | `level`, `port` |

`[quality] enabled`, `[purge] actuate` and `[eis.gates] enabled` all ship **false**: each removes
or reshapes data and waits on thresholds reviewed against this rig's runs.

### Instruments

| Config key | Instrument | Default port | Methods |
|---|---|---|---|
| `instruments.stage` | Newport ESP301 stage | `ASRL7::INSTR` | `stage_init`, `move_to`, `move_by`, `home_stage`, `live_position`, `stage_end` |
| `instruments.syringe` | Harvard Apparatus pump | `ASRL4::INSTR` | `single_pump`, `head_flip`, `head_retract`, `head_descend`, `head_check`, `syr_end` |
| `instruments.temp_controller` | Novus N1040 | `com6` | `write_sp`, `get_sp`, `get_pv`, `get_pv_surf`, `wait`, `ramp_linear`, `anneal` |
| `instruments.pico1` / `pico2` | PalmSens EmStat Pico (×2) | `auto` | `sendscript_getdata`, `eis_extractdata`, `eis_plotdata` |
| `instruments.piezo` | Trinket M0 piezo controller | `COM16` | `set_channel`, `set_frequency`, `set_sweep`, `apply_profile`, `standby`, `reset_config` |
| `instruments.camera` | ThorLabs Zelux | SDK discovery | `snap`, `acquire_n_frames`, `save_image` |
| `instruments.lamp` | MCP4728 quad-DAC via MCP2221 | ch `A` @ `0x60` | `on`, `off`, `set_eeprom_defaults` |
| `instruments.keithley` | Keithley DAQ6510 — **mock driver only** | USB VISA | `singleCh_measure(ch, nplc)`, `multi_measure(ch_start, ch_end, nplc)` |
| `instruments.ht_sensor` | SHT31-D | MCP2221 HID | `get_T`, `get_H` |
| `instruments.rh_controller` | Trinket M0 RH PID | `COM11` | `set_setpoint`, `start`, `stop`, `get_H`, `wait` |

With `port = "auto"`, `pico1` binds to the first enumerated EmStat Pico and `pico2` to the
second, so unstable COM enumeration puts channels 17–32 on the wrong device. Pin them with
explicit `port = "COM5"` / `"COM7"` entries.

### EIS presets

| Preset | Points | f_hi | f_lo | AC amplitude |
|---|---|---|---|---|
| Standard | 34 | 200 kHz | 3.912 Hz | 10 mV |
| Quick (`DEFAULT_PRESET`) | 27 | 200 kHz | 6.475 Hz | 10 mV |
| Extended | 53 | 200 kHz | 1.351 Hz | 10 mV |
| Longest | 39 | 200 kHz | 228 mHz | 10 mV |

`f_lo` is a conductivity floor: the −Z″ apex sits at `f = 1/(2πRC_cell)`, so a preset that does
not reach the apex extrapolates `R1` instead of measuring it. Editing a preset retires its
stopwatch anchor in `core/preflight.py`, and durations then report as extrapolated. **Preset names
are looked up case-sensitively.**

### `[safety]`

| Key | Value | Governs |
|---|---|---|
| `temp_min_C` / `temp_max_C` | 5.0 / 200.0 °C | Heater setpoint band |
| `pump_rate_min` / `pump_rate_max` | 0.05 / 2120.0 µL/min | Syringe rate (14.4 mm syringe) |
| `reservoir_soft_warn_uL` | 1000.0 | Alert; run continues |
| `reservoir_hard_stop_uL` | 250.0 | Dispense refused |
| `step_timeout_s` | 900 | Ceiling for a step declaring no `timeout_s`; `0` = unbounded |
| `stage_x_min_mm` / `x_max` / `y_min` / `y_max` | −100 / 100 / −50 / 50 mm | Stage travel |
| `anneal_deviation_warn_C` | 2.0 | Anneal watchdog: report |
| `anneal_deviation_fault_C` | 5.0 | Anneal watchdog: abort after the grace period |
| `anneal_deviation_grace_s` | 120.0 | Continuous fault time before abort |
| `anneal_poll_interval_s` | 30.0 | Watchdog poll cadence |

Reservoir levels are declared by the operator through Refill, never inferred from `single_pump`'s
`res_vol` (that is syringe volume declared to the pump firmware, not stock).

### Piezo

Disabled by default; enable explicitly.

```toml
[instruments.piezo]
driver = "piezo"
port = "COM16"
baud = 115200
enabled = false

[piezo]
enabled = false
channel = "A"
frequency_hz = 500
sweep_on_s = 2.0
sweep_rest_s = 3.0

[piezo.liquid_events]
enabled = false
settings_source = "manual_profile"   # manual_profile | liquid_event_profile
channel_a = true
frequency_hz = 500
sweep_on_s = 2.0
sweep_rest_s = 3.0
```

Event profile values are validated against protocol limits: frequency `10..5000` Hz, sweep timings
`0.01..120.0` s. CFG commands need firmware capability `CAPS PIEZO_CFG_V1`; legacy firmware still
supports channel on/off. Real hardware needs `pyserial` and a reachable COM port.

---

## 3. Launching the GUI

```powershell
softae-gui              # auto-detects real hardware, falls back to mocks per instrument
python -m softae.gui    # equivalent
```

The window is titled "SoftAE — Soft-matter Autonomous Experimentation" (1200×800 minimum),
carries **13 tabs**, and has emergency-stop and safe-exit buttons in the toolbar.

| Mode | Behaviour |
|---|---|
| Auto-detect (default) | Tries each real driver; falls back to mock per instrument |
| Mock (`mock=True`) | Every configured instrument uses a simulated driver |
| Real (`mock=False`) | Demands real hardware; raises if unavailable |

---

## 4. Tab reference

**Tab 1 — Init & Calibration.** Instrument table (name, type, state, details; 2 s refresh).
Connect/Disconnect All or Selected. Stage calibration: Home and Dep-1, "Set Current →" captures
live position, "Go Home"/"Go Dep-1" move on a background thread. Syringe config: per-pump
`parallel_syringes` (1 or 2), **Apply + Save** persists to `softae_config.toml`. PCB selector.
Position map: click an electrode to move there.

**Edit Well Occupancy…** (under the position map) pops out a per-well editor for the current board:
mark a well cast or free by hand, or swap to a fresh board from the same place. Nothing is written
until **OK** — Cancel means nothing happened — and every override records an `occupancy_override`
alert carrying the well, the direction, whatever row it displaced and your free-text reason. The
editor **refuses to open while a campaign holds the rig**, because a running campaign froze its copy
of occupancy at launch and would not see the edit. If the store cannot delete a well, recorded wells
are read-only rather than silently unchanged; if it cannot read row detail, the free-confirmation
says the sample identities could not be read rather than reporting none.

**Tab 2 — Liquid Model.** System parameters (`beta`, `eta_ref_mpas`, `alpha_growth_per_run`) and
three line panels (0/1/2) with per-line physics and a live prime-volume estimate. **Apply + Save**
writes `[liquid_handling]`, `[liquid_handling.line.<id>]` and `[piezo.liquid_events]`. Piezo event
settings: enable, source (`manual_profile` uses the Manual tab's active profile;
`liquid_event_profile` injects one `piezo.apply_profile(...)` step), channel A, freq/ON/REST
(editable only under `liquid_event_profile`).

**Tab 3 — Manual Control.**
- *Stage*: X/Y "Go To" or jog arrows, step 0.01–50 mm, position polled every 2 s.
- *Temperature*: setpoint (5–200 °C); ramp (target + °C/min); **Anneal** (target + hold seconds,
  optional ramp rate and tolerance) restores the original setpoint via `finally`.
- *Relative humidity*: target 0–95 %, Set / Start PID / Stop PID.
- *Syringe pumps*: three rows, rate µL/min + volume µL → Infuse. Each row shows
  `Syringes loaded: N` and divides its command by that count. `Apply liquid correction` toggles
  correction. Retract/Descend with a head indicator (green = retracted, orange = descended).
- *Piezo (channel A)*: ON/OFF via `set_channel`; Freq/ON/REST + **Apply Settings** via
  `apply_profile`. Read-only when `[piezo] enabled = false`.
- *Camera*: exposure 0.001–10 s, Snap, Live Preview (1 FPS), Lamp On/Off, 320×240.
- *EIS quick run*: channel 1–32, pico auto-routed. A preset populates `f_hi`, `f_lo` (mHz),
  `npts`, `mVac`, `mVdc`; edits apply to this run only and are never written back to the preset.
  Optional auto-save to the DataStore run directory and fit overlay; Nyquist + Bode popup.

**Tab 4 — Monitoring.** Rolling 10-min temperature (PV+SP) and humidity (RH+SP) plots; numeric
readouts (temp PV/SP, RH, RH SP, stage X/Y); ThorLabs camera feed at 1 FPS; USB webcam panel
(exposure slider −1..−9, timestamp overlay, click-drag zoom, single click resets); workflow
progress bar; instrument log (last 500 lines).

**Tab 5 — HT Experiment.** Mode (**Full Protocol** = flush + deposit + EIS, or **Measure Only**) →
PCB layout and EIS preset → formulation matrix, or **Formulation Manager…** to define stocks and
compute per-channel volumes → channel selection → **Generate Workflow** to preview → **▶ Start**,
**⏸ Pause**, **⏹ Abort** → **Save CSV** / **Save EIS Data** / **Fit All EIS** / **Save PDF
Report**. The preview reports `liquid_correction: enabled|disabled`, prime estimates, and
per-channel target vs commanded dispense for `p0`/`p1`.

Channel specs everywhere (HT, Arrhenius, Live BO) parse through `core/channel_spec.py`:
comma-separated channels, `lo-hi` inclusive ranges, whitespace ignored. The HT tab drops a bad
token silently; the others raise, and Live BO also rejects a channel beyond the selected board's
electrode count.

Piezo liquid events during Full Protocol need all of `[piezo] enabled`,
`[piezo.liquid_events] enabled` and `channel_a = true`; the workflow then inserts `piezo_on_chN`
before each channel's dispense/EIS block, `piezo_off_chN` after it, and `piezo_standby` in
teardown. Measure-only adds none.

**Tab 6 — Arrhenius Sweep.** Temperature profile (T start, T stop, T step °C, dwell) → channels →
instrument names (`pico1`, `temp_controller`) → electrode geometry L, t, w in cm (blank gives NaN
σ) → EIS preset → **▶ Start Sweep** (builds per-channel `.mscr` files first). The Arrhenius panel
plots ln(σ) vs 1/T with a linear fit and reports E_a and σ₀; **Export CSV** saves per-temperature
σ and fit parameters. Pico routing as Tab 5.

**Tab 7 — Autonomous.** Placeholder scaffolding, retained deliberately for a future
multi-objective front end. Closed-loop campaigns run from Tab 10 or `softae-campaign`. Do not
remove it.

**Tab 8 — Analysis.** *Fit & Export*: **Load File(s)…** (`.txt`, `.csv`, `.dat`) → Nyquist + Bode
→ circuit model → **Fit All** → L, t, w in cm → results table of R₀, R₁, σ per channel →
**Save to Database** / **Browse Database** / **Export CSV**. *EIS Browser*: three-pane viewer
(Overview / Inspection / Conductivity); **↻ Reload from DataStore** opens a dialog with
run/channel/limit filters and loads or imports selected rows; **⤢ Pop Out Window** detaches it.
Reload and import never auto-fit. Standalone:

```python
from softae.gui.widgets.eis_visualizer_widget import EISVisualizerWindow, ListEISSource
EISVisualizerWindow.open(ListEISSource(entries))   # blocks until the window closes
```

**Tab 9 — BO Simulator.** Offline Bayesian-optimization sandbox against a simulated conductivity
landscape; needs no `InstrumentManager`. Acquisition (`ucb`/`ei`) and κ, batch size and strategy,
seed, budget; optional temperature axis folded into an Arrhenius/VFT parameter; σ map vs
derived-objective map, convergence trace, suggested-point scatter; JSON export.

**Tab 10 — Live BO Campaign.** The hardware-in-the-loop optimizer.

| Search-over mode | Searched | Objective |
|---|---|---|
| Raw volumes | Per-pump µL directly | mean \|Z\|, minimised |
| Composition targets | Molar ratio / dried fraction / concentration, each Low→High | σ, maximised |

Raw volumes has no stock identity, hence no dry thickness and no σ; composition targets give every
trial a predicted thickness. Stocks and pump assignment come from the persisted pump loadout, so
declare it in **Instruments → Syringe Stock…** first ([§12](#12-softae-campaign)). A target row with
`Low == High` is pinned: held constant, kept out of the optimizer. **Direction** defaults to `auto`
and should stay there ([§12](#12-softae-campaign)). Also here: board-exchange controls and electrode
capacity, seed observations, an optional prior mean, a pre-run overflow scan, and a projected
duration plus stock-runway preflight. **Starting a campaign shows the Final-Check digest as a dialog
before anything is spawned**; on a block, **Proceed** is disabled ([§12](#12-softae-campaign)).

**Tab 11 — Catalogs.** Read-only browser over the chemical and solution catalogs with an **Edit**
button opening the Catalog Manager ([§11](#11-deposition-twin-and-catalogs)).

**Tab 12 — Deposition.** The deposition twin embedded in the main window
([§11](#11-deposition-twin-and-catalogs)).

**Tab 13 — Process Studio.** Browse the task catalog and deposition recipes from `tasks.toml` /
`recipes.toml`, see each method's maturity level, and build, preview and run workflows against
connected instruments. Maturity is warn-and-proceed: a campaign running a method below its
expected maturity emits `method_below_maturity` and continues.

---

## 5. Command-line tools

| Command | Purpose | Section |
|---|---|---|
| `softae-gui` | Launch the desktop application | [§3](#3-launching-the-gui) |
| `softae-run` | Execute a workflow YAML headlessly | [§6](#6-softae-run-and-workflow-yaml) |
| `softae-campaign` | Run / resume / control an autonomous campaign | [§12](#12-softae-campaign) |
| `softae-commission` | Acquire and derive the EIS fixture calibration | [§13](#13-softae-commission) |
| `softae-deposition` | Standalone deposition-twin GUI | [§11](#11-deposition-twin-and-catalogs) |
| `softae-method` | Method maturity (`status`, `test`, `promote`, `sign-off`, `versions`) | `docs/METHOD_MATURITY_PIPELINE.md` |
| `softae-catalog` | `init` — seed `data/` with the placeholder catalog, never overwriting | *Before you start*, above |
| `softae-web` | EIS web visualizer over the DataStore; needs `[web]` | `python -m softae.web --help` |
| `softae-shadow` | Arm, rehearse and review a shadow campaign | [§15](#15-softae-shadow) |
| `softae-thickness` | Plan and record an unconfounded thickness series | [§16](#16-softae-thickness) |
| `softae-equilibration` | Measure σ(t) and derive the conditioning hold | [§17](#17-softae-equilibration) |
| `softae-env` | Hold the chamber at a humidity and measure nothing | [§18](#18-softae-env) |
| `softae-eis-timing` | Time EIS acquisition grids | `python -m softae.tools.eis_timing --help` |
| `softae-eis-validate` | Validate the adaptive-acquisition path | [§19](#19-softae-eis-validate) |

`softae` is an additional `gui_scripts` launcher for the GUI.

**Install state in this venv (2026-09-21).** `softae-env`, `softae-eis-timing`,
`softae-eis-validate` and `softae-catalog` are registered in `pyproject.toml` but have **no
generated `.exe` here**; they were added after the last editable install. Re-run
`pip install -e .`, or use the module form, which resolves either way:

```bash
python -m softae.tools.env_hold --help        # softae-env
python -m softae.tools.eis_timing --help      # softae-eis-timing
python -m softae.tools.eis_validate --help    # softae-eis-validate
python -m softae.tools.catalog_init --help    # softae-catalog
python -m softae.tools.shadow_review --help   # softae-shadow — module is shadow_review, not shadow
```

Console scripts are generated at install time, so "command not recognized" on a documented command
is almost always a stale install. A missing **extra** is a different failure: `softae-web` without
`[web]` names the missing package, prints `pip install "softae[web]"` and exits **1** (argparse
owns 2 for usage errors); `softae-web --help` still works without it.

**Interlock.** Any command that drives real motion hardware also requires
`SOFTAE_ALLOW_HARDWARE=1` — see [§8](#8-stopping-safe-exit-interlock-head-state).

---

## 6. `softae-run` and workflow YAML

```
softae-run <workflow.yaml> [OPTIONS]
```

| Flag | Description |
|---|---|
| `--mock` | Force mock instruments |
| `--real` | Require real instruments; fail if unavailable |
| `--dry-run` | Parse and validate only; print resolved steps |
| `--validate` | Check instrument/method names against the driver registry (exit 3 on failure) |
| `--log-dir DIR` | JSON-lines log directory (default `./logs`) |
| `--verbose` / `-v` | Step-by-step progress to stdout |

`--mock` and `--real` are mutually exclusive; omit both for auto-detect.

| Exit | Meaning |
|---|---|
| 0 | Success |
| 1 | Workflow error / instrument failure |
| 2 | Bad arguments / parse error |
| 3 | `--validate` found errors |
| 130 | Interrupted (Ctrl-C) |

Each run writes `logs/<name>_<UTC>.jsonl`, one JSON object per step with `timestamp`, `workflow`,
`step`, `instrument`, `method`, `params`, `tags`, `duration_s` and `result`.

### Schema

```yaml
name: "experiment_name"              # REQUIRED
description: "What this does"        # optional
variables:                           # optional — $var references
  my_list: [1, 2, 3]
  my_value: 42
metadata: {}                         # optional — logged for provenance

setup:                               # at least one phase section is needed
  - name: step_name                  # REQUIRED
    instrument: instrument_name      # REQUIRED — must match InstrumentManager
    method: method_name              # REQUIRED — must be callable on the driver
    params: { key: "$my_value" }     # optional — kwargs passed to the method
    timeout_s: 120                   # optional
    retry: 1                         # optional (default 0)
    depends_on: []                   # optional — see below
    tags: {}                         # optional

loop:                                # optional — a dict, not a list
  iterate_over: my_list              # list → len(list); int → that count; omitted → 1
  steps: [...]

teardown:                            # optional — always runs, even on error or abort
  - ...
```

| Interpolation | Behaviour |
|---|---|
| `"$var"` (exact match) | Replaced with the value, type preserved (`"$my_value"` → `42`) |
| `"prefix_$var_suffix"` | String substitution only |
| Nested | Works inside dicts and lists in `params` |

Loop steps get `__iter0`, `__iter1`, … suffixes and `{"iteration": "N"}` tags.

| `depends_on` | Behaviour |
|---|---|
| Field absent | Implicitly depends on the previous step — sequential |
| `depends_on: []` | No dependencies; may run in parallel with other independent steps |
| `depends_on: ["a", "b"]` | Waits for both |

The executor topologically sorts steps into tiers and runs each tier concurrently via
`asyncio.gather`. Circular dependencies and references to non-existent step names are rejected at
parse time; dependencies must stay inside one phase. Inside a loop, names resolve within the same
iteration (`measure__iter2` waits for `deposit__iter2`). If a dependency fails, its dependents are
skipped and other steps in the tier still complete.

### Driver parameter names

`params` keys are passed as `**kwargs`, so they must match the driver method's signature exactly.
Check the driver signature before relying on a row.

| Instrument | Method | Params |
|---|---|---|
| `stage` | `move_to` / `move_by` | `x, y` / `dx, dy` |
| `syringe` | `single_pump` | `res_vol`, `ID`, `rate`, `dispense_vol` |
| `piezo` | `set_channel` / `apply_profile` / `standby` | `channel, enabled` / `frequency_hz, on_s, rest_s` / — |
| `temp_controller` | `write_sp` | `T_SP`, `print_flag` |
| `temp_controller` | `wait` | `within`, `equilibration_time`, `timeout` |
| `temp_controller` | `ramp_linear` | `start`, `end`, `rate`, `step` |
| `temp_controller` | `anneal` | `target_temp_C`, `hold_time_s`, `ramp_rate` (opt), `tolerance` (opt, 1.0) |
| `pico1` / `pico2` | `sendscript_getdata` | `mscrpath`, `outdir`, `chan` |
| `rh_controller` | `set_setpoint` | `sp` |
| `camera` | `snap` | — |
| `lamp` | `on` / `off` | — |

### Bundled templates

| File | Description |
|---|---|
| `workflows/standard_eis_sweep.yaml` | 16-channel EIS sweep with formulation |
| `workflows/single_drop_and_measure.yaml` | Single-channel deposit + EIS |
| `workflows/temp_ramp_eis.yaml` | Temperature ramp with EIS at each point |
| `workflows/examples/piezo_assisted_dispense.yaml` | Piezo around dispense, optional profile apply |
| `workflows/examples/01_hello_stage.yaml` | Stage movement test |
| `workflows/examples/02_temp_setpoint.yaml` | Temperature set/read |
| `workflows/examples/03_three_channel_eis.yaml` | 3-channel deposit + EIS loop |

---

## 7. EIS analysis API

```python
from softae.analysis.eis_data import EISResult

result = EISResult.load("path/to/eisdata.txt")
result = EISResult.from_arrays(channel=1, f=freqs, z_real=z_prime, z_imag_neg=neg_z_dd)
result.save("output/E1_eisdata.txt", study_name="my_study")
```

File format is 5-column text with a `#`-prefixed metadata header
(`f(Hz) Z_total(Ohm) phase(deg) Z'(Ohm) -Z''(Ohm)`).

```python
from softae.analysis.eis.engine import analyze_spectrum
from softae.analysis.eis.geometry import CellConstant

# Per-sample geometry in cm: electrode gap L, film thickness t, stripe length w.
cell = CellConstant.from_legacy(0.2, 0.175, 0.2)

# `engine` is deliberately not passed — `[eis] engine` decides, for every call site at once.
report = analyze_spectrum(eis_result, cell=cell, model_name="simpleSalt")
print(report.fit.R0, report.fit.R1, report.fit.success)

if report.sigma.mode == "value":
    print(report.sigma.value)                     # S/cm
elif report.sigma.mode in ("bound", "bound_unqualified"):
    print(report.sigma.upper_bound)               # instrument-limited upper bound
else:                                             # "unavailable"
    print("no per-sample thickness, or the spectrum was rejected")
```

Always check `report.sigma.mode` before reading `.value`.

| Registry | Engine | Models |
|---|---|---|
| `analysis/circuit_fitting.CIRCUIT_MODELS` | `legacy` | `simpleSalt` (R₀-CPE₀-p(R₁,C₀), 5 free), `flexSalt` (same with fixed C₀, 4 free) |
| `analysis/eis/models.EIS_CIRCUITS` | `gated` | `blocking_coplanar` (R0-CPE0-p(R1,C0)), `blocking_coplanar_L` (adds L0; ships unused — L must be pinned from a short blank, not fitted) |

The two key spaces deliberately do not collide, so a name cannot be resolved by the wrong engine.
`simpleSaltMembrane` is retired and `fit_circuit` raises on the name; use `simpleSalt` for membrane
samples.

`circuit_fitting.z_to_sigma(L, t, w, R1)` and `FitResult.sigma(L, t, w)` are **deprecated**, emit a
`DeprecationWarning`, and have no callers: they divide by a geometry with no provenance and bypass
`[eis] engine`. They survive only as the oracle the parity tests check `CellConstant.sigma` against.

---

## 8. Stopping, safe exit, interlock, head state

| Toolbar position | Control | When |
|---|---|---|
| Left | red **⛔ EMERGENCY STOP** | Something is going wrong now |
| Right | amber **⏻ SAFE EXIT** | You are finished and leaving |

Both drive the same park sequence; there is no second path to the hardware.

1. Retract dispenser head
2. Stop syringe pumps 0, 1, 2
3. Set temperature to 10 °C
4. Leave the humidifier **purging dry** — PID stopped, setpoint 0, duty held at `out_min` (0.01)
5. Lamp off

Each step is attempted even if others fail, and a dialog reports successes and errors. The
humidifier step is graded *commanded, never verified*: an absent or disconnected RH controller is
skipped, while a driver exposing no `safe_dry()` or a failed duty write is reported as an error.

**After an emergency stop the two Aalborg PSVs stay open with dry gas flowing for about 25 s**,
closing when the Trinket's deadman fires. Heater, pumps and lamp act immediately; only the RH axis
has this window. The dialog reports `DRY-PURGED … Leaving it commanded is DELIBERATE` — a success,
not a softened failure. Every route behaves this way: E-Stop, Safe Exit, the window's X, a
fault-class campaign park, crash and signal recovery (Ctrl-C, `SIGTERM`), and the unclean-shutdown
recovery park at next launch. No caller can ask for the other end state.

**Safe Exit and the head.** If the head is down, Safe Exit asks: **Raise head, then exit** (the
default, selected by Enter); **Leave head down, then exit** (for a head holding a position — an
anneal hold in the flush basin, a paused cast, a drop it is sitting in); or **Cancel** (nothing is
touched, the window stays open). A raised head, an absent or disconnected syringe, or a driver that
does not track head state all exit without the prompt. Every other route out, including the
window's X, raises the head. Pumps, temperature, humidifier and lamp are parked unconditionally in
both modes. If any subsystem fails to park, Safe Exit reports what failed and asks before closing.

**Closing mid-run** cooperatively aborts a running experiment, BO campaign or Arrhenius sweep —
the same effect as its Abort/Stop button — before the window finishes closing.

**Hardware interlock.** With a real stage, syringe or piezo present, a headless workflow refuses to
execute unless the rig is armed for the session:

```bash
export SOFTAE_ALLOW_HARDWARE=1        # bash
$env:SOFTAE_ALLOW_HARDWARE = "1"      # PowerShell
```

Launching the GUI arms its own process. Mock managers never trip the interlock.

**Dispenser head state.** The up/down state cannot be read back from hardware, so it is asked at
startup and gates stage motion. The answer is authoritative and is not overwritten by connecting or
reconnecting instruments; both `softae-campaign` and the GUI re-confirm it before a run that moves
the stage.

**Faults, consumables, purge, alerts.** A failing trial is retried; a systematic-looking fault parks
the run, keeping its checkpoint. An unmeasured trial is told to the optimizer as `None`, never
`0.0`. Stock levels are tracked in a ledger and projected before a run alongside projected duration
and waste accrual; undeclared stock is "unknown", never "empty". `[purge]` ships `actuate = false`,
so windows are planned and logged without moving a pump. Parks, gate timeouts, board exchanges and
stock warnings are written to the DataStore as alerts.

---

## 9. Errors and troubleshooting

Error classes live in `softae.errors`:

```
SoftAEError
├── InstrumentError (message, instrument)
│   ├── ConnectionError_      — failed to open port
│   ├── CommunicationError    — timeout / no response
│   └── SafetyError           — value exceeds a configured limit (requested, limit)
├── WorkflowError
│   ├── StepTimeoutError      — step exceeded timeout_s (step name, timeout)
│   ├── AbortedError          — user or agent aborted
│   └── ValidationError_      — bad YAML / missing field
├── AnalysisError             — fit failure / data mismatch
├── OptimizerError
└── CampaignError
```

| Symptom | Cause | Fix |
|---|---|---|
| `softae-gui` not found | Not installed editable | `pip install -e .` from `softae-next/` |
| A documented command not found | Entry point added after the last install | `pip install -e .`, or `python -m softae.tools.<module>` |
| All instruments DISCONNECTED | Hardware absent, or auto-detect fell back | Check cables and ports, or run `--mock` |
| `SafetyError: requested 250.0 exceeds limit 200.0` | Setpoint above the configured max | Edit `[safety] temp_max_C` |
| Camera preview freezes the GUI | Not using `CameraWorker` | Report as a bug — the SDK needs its own thread |
| RH shows `nan %` | PID not started and no sensor connected | Start the PID loop or connect the SHT31-D |
| Workflow step raises `TypeError` | YAML param names do not match the driver signature | See the table in [§6](#6-softae-run-and-workflow-yaml) |
| `No module named 'hid'` | Hardware extras absent | `pip install -e ".[hardware]"` |
| Config not found | TOML not in CWD and no env var | Set `SOFTAE_CONFIG`, or copy `softae_config.toml` to CWD |
| EIS fit returns `success=False` | Bad initial guess or wrong circuit model | Try another model; check data quality |
| `ModuleNotFoundError` from `softae-web` | `[web]` extra absent | `pip install "softae[web]"` |

### Device firmware

`rh_controller` and `piezo` each talk to an Adafruit Trinket M0 running its own CircuitPython
program, versioned at `scripts/trinket_firmware/` with SHA-256 per file and each device's protocol.

| Directory | Volume | Role | Firmware deadman |
|---|---|---|---|
| `dac0_rh/` | `DAC0` | RH controller — two Aalborg PSV valves, wet/dry mix | ≈ 25 s, self-recovering |
| `pwm0_piezo/` | `PWM0` | Piezo driver — two duty-cycled channels | 600 s, needs a fresh command to clear |

The device is the truth; the directory is a record. Nothing there is imported, executed or synced,
and the two boards run different CircuitPython builds (`DAC0` 9.2.7, `PWM0` 10.2.1). If the repo
copy and the `CIRCUITPY` volume disagree, the volume is right — re-copy and re-hash.

**Redeploying firmware is hardware actuation**: an operator copies `code.py` onto the volume and
CircuitPython restarts the program mid-loop, dropping valve or piezo drive at an arbitrary moment.
Never redeploy while a campaign, hold or experiment is running.

---

## 10. DataStore

Every run is persisted in a project-scoped SQLite database in WAL mode.

```toml
[data]
project_dir   = "~/softae_data"   # default; auto-created on first use
db_filename   = "softae.db"       # default; the SQLite file sits in project_dir/db/
auto_save_eis = true
```

| Table | Content |
|---|---|
| `experiments` | Run lifecycle — name, start/end, status, config hash, notes |
| `measurements` | Per-channel raw-data paths, `modality`, `payload_path`/`payload_format`, `sample_uuid` |
| `conditions` | Environmental snapshots (formulation, processing, measurement, anneal) |
| `fit_results` | Fit parameters, plus `arc_state`, `arc_f_peak_hz`, `arc_f_low_hz`, `arc_phase_low_deg` |
| `formulations` | Dispense volumes, deposit area + thickness provenance, `sample_uuid` |
| `electrode_occupancy` | Spent wells per `(board_id, electrode)`, `sample_uuid` |
| `arrhenius_results` | Per-series E_a and σ₀ |
| `board_state`, `rig_state` | Board and rig-claim state |
| `campaign_checkpoints` | Resume checkpoints |
| `alerts` | Parks, gate timeouts, exchanges, stock warnings |
| `reservoir_levels` | Declared stock ledger |
| `eis_calibrations` | Append-only calibration history |
| `fixture_corrections` | One row per analysed spectrum, including declined ones |
| `thickness_plans`, `measured_thickness` | Thickness plans and profilometer readings |
| `schema_version` | Append-only epoch ledger — schema shape and changes of meaning |
| `doe_parameters` | Reserved |

**Payloads.** Alongside the transitional `runs/<run_id>/data/eis/<stem>.txt`, every routed
measurement writes `runs/<run_id>/data/<modality>/<stem>.nc` — a self-describing xarray Dataset
whose `attrs` carry `run_id`, `measurement_id`, `channel`, electrode geometry and `sample_uuid`.
The sibling tree per modality means retiring the `.txt` files is not a payload migration. Writing a
payload is best-effort: on failure the measurement row is still written and `payload_path` /
`payload_format` stay NULL. A NULL path means *no payload*, never a path to a missing file.

**`sample_uuid`** is minted when a well is consumed, one per `(trial, channel)`, and stamped into
`formulations`, `electrode_occupancy` and `measurements`, into workflow step tags, and into the
payload `attrs`. It is a **grouping key, not a unique one**: a three-temperature sweep off one film
is one identity and three rows. A batch round of q wells mints q identities. Rows recorded before
2026-08-08 carry NULL and are not backfilled; a resumed campaign mints for new trials only.

The HT tab calls `start_run()` / `record_measurement()` / `finish_run()`; the Manual tab records EIS
snapshots to a daily pseudo-run (`manual_YYYYMMDD`); the CLI runner logs the config hash as its
first provenance event.

```python
from softae.core.data_store import DataStore

ds = DataStore("~/softae_data")
runs  = ds.query_runs()
meas  = ds.query_measurements(run_id)
conds = ds.query_conditions(run_id)
fits  = ds.query_fits(run_id)
```

---

## 11. Deposition twin and catalogs

The deposition twin (`softae.core.deposition`) casts an `ElutionResult` into cylindrical wells,
evaporates the carrier at a tunable percentage while retaining all dep volume, and reports
flat-disc thickness plus a mass balance. Pure math, no hardware.

```python
from softae.core.formulation import (
    Chemical, ChemicalCatalog, Solution, SolutionComponent, compute_elution_volumes,
)
from softae.core.deposition import WellGeometry, simulate_plate_deposition

catalog = ChemicalCatalog()
catalog.add(Chemical("PEO", density_g_per_mL=1.2))
catalog.add(Chemical("Water", density_g_per_mL=1.0))

solutions = {"stock": Solution("stock", [
    SolutionComponent("PEO", "dep", 1.0, "mL"),       # solute — retained
    SolutionComponent("Water", "carrier", 3.0, "mL"), # solvent — evaporates
])}

elution = compute_elution_volumes(solutions, catalog, target_deposition_uL=20.0)
well    = WellGeometry(diameter_mm=5.0, depth_mm=2.0)   # capacity_uL ~= 39.3
summary = simulate_plate_deposition(elution, well, evaporation_pct=95.0, n_wells=4)

w = summary.wells[0]
print(w.wet_thickness_um, w.final_thickness_um, w.overflows)
print(summary.total_eluted_uL, summary.total_dispensed_uL, summary.undeposited_uL,
      summary.total_evaporated_uL, summary.total_final_uL)
print("\n".join(summary.summary_lines()))
```

- `dispense_uL` may be `None` (equal split of `grand_total_uL`), one float per well, or a per-well
  list. Total dispensed can never exceed total eluted; the remainder is `undeposited_uL`.
- `simulate_well_deposition(...)` is the single-well equivalent, returning one
  `WellDepositionResult`.
- `carrier_keys=carrier_component_keys(solutions)` adds per-component final volumes
  (`component_final_uL`).
- Overflow (wet volume above capacity) is a flag, not an error.

**Standalone GUI:** `softae-deposition`, or `python -m softae.gui.deposition_app`.

Stock table columns: **Use**, **Auto**, **Solution**, **Fraction**, and read-only **Eluted µL /
Dep µL / Carrier µL**. **Auto ON** absorbs the remainder so fractions sum to 1; **Auto OFF** takes
an exact share, with `0` honoured literally. The default is Auto OFF at Fraction 0.00.
**Auto-balance all** sets Auto ON for every checked stock; **Normalize** rescales explicit
fractions to sum to 1.00. The Σ-fractions indicator warns without blocking: green Σ = 1; amber
Σ < 1, Σ > 1 or all-zero; red when explicit fractions exceed 1 while Auto rows are present (those
get clamped to 0). Carrier-only stocks (`dep_fraction` 0) are excluded from Σ. Equal *dep* share is
not equal *eluted* volume, because `eluted = dep / dep_fraction`; **Show component breakdown**
traces each chemical's role and eluted µL.

Other inputs: target deposition µL; well diameter and depth (mm); well count and dispense mode;
evaporation slider (0–100 %, 0.5 % steps). Outputs: per-well table, mass-balance strip, red
overflow banner, an animated `WellSketch` cross-section, and **Export CSV…** writing `#CONFIG`,
`#STOCKS`, `#MASS_BALANCE` and `#WELLS` blocks.

**Catalogs.** `chemicals.csv` and `solutions.csv` load from `[paths] data_root`, resolved
**relative to the config file's directory** (absolute paths and `~` honoured); with no config it
falls back to `./data`, then to empty catalogs with a status message.

Routes to the slim Catalog Manager (catalog CRUD only): the main window's **Catalogs → Edit
Catalogs…** menu or toolbar button, the **Edit** button on Tab 11, or **Manage Catalogs…** in the
deposition panel. The full Formulation Manager (catalog editing plus the elution calculator and
pump controls) opens from Tab 5. Editing anywhere refreshes Tabs 11 and 12 live.

- **Save** writes back to the canonical `data_root` location; **Save As… / Load From…** handle
  ad-hoc files. **Reload catalogs** re-reads from disk.
- A component's **Chemical** field is a dropdown of current catalog chemicals; an unknown reference
  loaded from disk is preserved and selectable.
- **Renaming a chemical cascades** to every referencing component. A colliding rename is blocked
  and reverted; clearing a name is flagged by validation instead.
- **Save** and **Calculate** both validate first: a blank or non-positive density, a blank or
  non-positive component quantity, an unknown-chemical reference, or a data-bearing row with a
  blank name produces a Proceed/Cancel prompt and tints the offending cells. Blank molar mass or
  viscosity are legitimate and are not flagged.
- Every chemical field round-trips, including viscosity and the particulate flag, as does each
  component's calc mode.

---

## 12. `softae-campaign`

A campaign is a closed loop: suggest → cast → measure → tell. Run it from Tab 10 or headlessly.

```bash
softae-campaign check   my_campaign.toml                       # validate; no hardware
softae-campaign run     my_campaign.toml --yes --head-up
softae-campaign resume  my_campaign.toml --yes --head-up       # alias for run --resume
softae-campaign run     my_campaign.toml --yes --mock
softae-campaign run     my_campaign.toml --project ./runs/aug
softae-campaign control pause|resume|abort [--run-dir DIR] [--reason TEXT]
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--yes` / `-y` | run, resume | Answers the launch prompts, Final Check included. **Never overrides a block** |
| `--resume` | run | Continue a saved checkpoint; off by default |
| `--mock` | run, resume | Mock instruments for a full dry rehearsal |
| `--project DIR` | all | Overrides `[data] project_dir`, and decides where the resume checkpoint lives |
| `--head-up` / `--head-down` | run, resume | Mutually exclusive; prompted on the terminal if omitted |

`control` reaches a campaign already running; `--run-dir` defaults to reading the rig lock, and
`--reason` is recorded verbatim in the transcript and, for `abort`, in the park alert.

| Exit | Meaning |
|---|---|
| 0 | OK |
| 1 | Campaign parked or failed |
| 2 | Usage or spec error |
| 3 | Declined — a gate answered no, a projected shortfall not accepted, head state not stated |
| 4 | Rig busy — another process holds it; retrying later is correct |

`check` prints name, parameters, budget, channels, the preflight projection, the EIS calibration
advisory, any checkpoint summary, and the Final-Check digest below — the same page `run` shows, with
no prompt, since `check` connects nothing and moves nothing. It does **not** print the resolved
run-plan phase order, and it does **not** emit the cure-temperature warning — that is raised during
`run`, after hardware connect, when the deposition recipe is built.

**Head state.** The loop drives the head with conditional commands (raise if down, lower if up), so
a wrong belief costs one wrong flip. The flag records what is true *now*; an aborted run, a manual
jog or a power cycle all break inference from the last run. Omitting both prompts on the terminal,
which hangs an unattended launch, so cron and scheduler invocations must state it alongside
`--yes`. Every other headless gate has a safe default; head position has none.

### Final Check

Every launch is preceded by a digest that puts what the file intends beside what the bench holds:
the stocks on each pump, the measurement block, the run-plan sequence, board occupancy, EIS
calibration coverage, purge state, and the duration and stock projection. Each finding is `ok`, a
`warn` or a **`BLOCK`**.

| Surface | Behaviour |
|---|---|
| `softae-campaign run` / `resume` | Prints the digest, then asks `Proceed? [y/N]`. Declining exits **3** |
| Tab 10 | Shows the digest as a dialog before the campaign is spawned; on a block, **Proceed** is disabled |
| `softae-campaign check` | Prints the digest and stops — no prompt, and today still exit 0 |
| `py -m softae.core.final_check <spec.toml> [--project DIR] [--yes]` | The same digest, standalone, against any spec |

**A block is not overridable.** `--yes` answers warnings; it cannot answer a block, and neither can a
typed `y`. A block means the spec and the bench declaration contradict each other — a pump loaded
with one solution and assigned another in the file — which is precisely the case that used to load
clean and cast the wrong chemistry. Fix the spec or re-declare the loadout.

Without `--project` there is no store, so the digest reports board occupancy as **unchecked** rather
than clean. *Unknown* is never printed as *fine*.

### Declaring stock, and what a run records

Declare what is physically loaded in **Instruments → Syringe Stock…**. Each pump's solution is picked
from the project's solution catalog. Opened without a project store, the stock combos are **disabled**
with a visible notice — a pick that could not be persisted is worse than no pick — so open a project
first. Every re-declaration writes an INFO `stock_declared` alert, giving the loadout a history
instead of a silent overwrite.

Each run's `provenance.json` then records `stocks` **both ways**: `by_pump` (the physical fact) and
`by_name` (its interpretation), with `source` saying which record won — `spec` when the file's
`pump_assignment` was authoritative, `declared` when the bench loadout was — plus the catalog rows
and any disagreement. A wrong name is therefore checkable afterwards against the pump the volume
actually left, from the run directory alone.

### Spec file

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

An unknown key is an error. Fields carrying live Python objects (`prior_mean`, `formulation`,
`piezo`, `seed_observations`) cannot be set from a file at all; those campaigns are built in Python
or from Tab 10. `general_formulation` is loadable (only the `formulation` key is refused), and so
is `run_plan` in the `[[run_plan.phases]]` form below.

**Casting one fixed recipe N times.** Pin a composition by writing it as a `[general_formulation]`
axis with `low == high`; a pinned axis is left out of the optimizer's parameter space. Add one
`int` axis to `[parameter_space]` (the shipped example calls it `replicate`) with `low = 1`,
`high = N`, set `optimizer = "grid"` and `budget = N`. Write every axis out in full — all six keys
(`kind`, `a`, `b`, `low`, `high`, `basis`), including `b = ""` on a `dried_fraction` axis — because
an omitted key takes a default and casts a different composition. Any axis left searched
(`low != high`) must also be named in `[parameter_space]`, or `check` refuses the spec.
`examples/bench_instance.toml` is the worked example.

### `[measurement]`

```toml
[measurement]
modality = "eis"      # the only modality a campaign can run today
preset   = "Quick"    # an [eis_presets.*] section name
enabled  = true       # false = formulate and cast, but do not measure

[measurement.overrides]
n_points = 40         # modality settings layered over the preset
```

| Legacy spelling (deprecated) | Block spelling |
|---|---|
| `eis_preset = "Quick"` | `[measurement] preset` |
| `eis_overrides = { n_points = 40 }` | `[measurement.overrides]` |
| `measure_eis = false` | `[measurement] enabled = false` |

Legacy fields still load, raise a `DeprecationWarning` and fold into the block. Both spellings
together are allowed **only when they agree**; a disagreement is refused rather than resolved by
precedence, and `check` surfaces it before hardware moves. Naming an unbuilt modality refuses to
start, listing the registered ones, before an instrument connects or a run row is written. Resume
is unaffected: only `modality` is identity-bearing.

### `[[run_plan.phases]]`

Array order is execution order.

```toml
[[run_plan.phases]]
kind  = "formulate"
scope = "per_sample"
  [run_plan.phases.conditions]
  name            = "casting"
  temp_setpoint_C = 25.0
  rh_setpoint_pct = 22.0

[[run_plan.phases]]
kind        = "anneal"
scope       = "per_batch"
anneal_task = "anneal_85C_8h"    # the task owns the cure: its temperature AND its hold
  [run_plan.phases.conditions]
  name            = "anneal"
  temp_setpoint_C = 25.0         # the RESTORE target after the hold, not the cure temperature
  rh_setpoint_pct = 22.0

[[run_plan.phases]]
kind  = "equilibrate"
scope = "per_batch"
  [run_plan.phases.settle]
  round_period_s = 240.0
  min_hold_s     = 1500.0
  max_hold_s     = 14400.0
  [run_plan.phases.conditions]
  name            = "equilibrate"
  temp_setpoint_C = 25.0
  rh_setpoint_pct = 22.0

[[run_plan.phases]]
kind  = "measure"
scope = "per_batch"
  [run_plan.phases.measurement]
  preset = "Extended"
```

| Key | Where | Required | Meaning |
|---|---|---|---|
| `kind` | every phase | yes | `formulate` · `anneal` · `equilibrate` · `measure` |
| `scope` | every phase | yes | `per_sample` (once per well) or `per_batch` (once for the round) |
| `anneal_task` | anneal | no | Task name from `data/tasks.toml`; this sets the cure |
| `hold_s` | anneal | no | Cure duration in seconds, overriding the task's `hold_time_s`. Refused if `anneal_params` also spells `hold_time_s` |
| `[…conditions]` | any phase | no | Chamber setpoints established at the phase boundary |
| `[…settle]` | equilibrate | no | Settle loop timing; the three keys are required together |
| `[…measurement]` | measure only | no | A `[measurement]`-shaped block for a denser read |

- **`scope` is never inferred.** `per_sample` casts and anneals each well on its own; `per_batch`
  casts all of them and cures once. A batch instance is
  `formulate ×N → anneal (all) → equilibrate (all) → measure (all)`.
- **`conditions` drives only the axes you name.** An omitted axis is not driven — leaving out
  `rh_setpoint_pct` does not mean 0 %RH. Tolerances and approach timeouts default to the
  *ascending* allowance (1 800 s); set them explicitly wherever the approach is passive, since this
  stage has no active cooling. `check` reports that as an advisory, never a refusal.
- **`conditions` on an ANNEAL phase is the rest state after the hold**, not the cure. The
  conditions setpoint is written first, the anneal ramps to the task's target, and on the way out
  it restores what conditions left. **Equal values are the trap**: `temp_setpoint_C = 85.0` beside
  `anneal_85C_8h` leaves the chamber at 85 °C after the cure, and a MEASURE phase with no
  conditions of its own then reads a hot board. Give the phase after an anneal its own
  `conditions`.
- **`settle` is all-or-nothing**; a partial block is refused rather than half-defaulted.
- **`measurement` is legal only on a MEASURE phase**, and optional there. The production read after
  settling is authoritative because of its role; this block only makes it denser.

### Objective and engine

`[eis] objective` and `[eis] engine` are different keys: the first chooses which metric, the second
which physics computes it. With `objective = "auto"` (the default) the metric resolves from what
the campaign can measure, and the direction follows the metric:

| Campaign mode | Twin predicts thickness? | Metric | Direction |
|---|---|---|---|
| composition — carries a formulation | yes | σ | maximise |
| volume — raw `vol_params` | no (no stock identity ⇒ no elution ⇒ no dry thickness) | mean \|Z\| | minimise |

Neither is a fallback for the other. An explicit `maximize`/`minimize` on `CampaignSpec.objective`
is honoured only when it agrees with the resolved metric and refused when it does not. The
conductivity the GUI shows and the conductivity the objective optimises are the same number,
produced by the engine named in `[eis] engine`; no surface names its own engine.

### Rounds, boards, budget, resume

A round is sized before anything is suggested, to the smallest of the requested batch size, the
electrodes still free on the plate, and the unspent budget.

- A full board is exchanged **before** the round, so no round is split across a plate swap.
- The final round narrows to the budget: budget 5 with q=4 spends exactly 5 electrodes.
- Board exchange prompts the operator and can be cancelled, stopping the run while keeping
  everything already measured. With no handler (fully headless) an exchange request stops cleanly
  rather than assuming a fresh plate.
- Drop-cast wells are single-use, so occupancy is persisted per `(board_id, electrode)` and
  survives a restart. Resuming into recorded occupancy asks fresh / resume / cancel. Wells can also
  be set by hand between runs, from Tab 1 ([§4](#4-tab-reference)).
- `--resume` is off by default. The checkpoint is fingerprinted against the spec; a changed
  parameter space, objective or optimizer setting is refused rather than continued into.

### `SOFTAE_SEED_DATASET`

Points at a historical aggregated-conductivity **file** (not a directory), used as a stand-in
oracle so a BO run can be exercised off the rig.

```bash
export SOFTAE_SEED_DATASET=/path/to/aggregated_conductivity.txt
python examples/bo_campaign_demo.py          # or pass the path as argv[1]
```

`examples/bo_campaign_demo.py` prints what to set and exits 1 when it is unset or missing (an
explicit `argv[1]` wins over the variable); the seeded tests in `tests/campaign_helpers.py` skip.
Neither invents data — the dataset is a lab record and is not in the repository.

---

## 13. `softae-commission`

Commissioning measures the **fixture** rather than a sample: what the leads and multiplexer
contribute, and what the instrument can resolve. Fixture drift is minimal, so a calibration is a
durable asset reused across campaigns.

```bash
softae-commission status
softae-commission run blank_short --channels 1-32 --fixture mux16 --electrode-mode two --yes
softae-commission derive --fixture mux16
softae-commission history --fixture mux16
```

`run` acquires and tags; `derive` reads the tagged spectra back from the database. Each `derive`
produces the best calibration the artifacts so far support.

| Subcommand | Flags |
|---|---|
| `status` | `--fixture`, `--project` |
| `run` | role positional, `--channels`, `--fixture`, `--project`, `--nominal`, `--electrode-mode {two,three}`, `--yes`/`-y`, `--mock` |
| `import` | role positional, `--file` (req), `--electrode-mode` (req), `--channel`, `--nominal`, `--re-connection {unverified,tied_to_ce,bridged_by_sample,open_by_geometry,connected}`, `--fixture`, `--project` |
| `derive` | `--channels`, `--representative N`, `--declare-electrode-mode`, `--nominal-load`, `--nominal-cap`, `--nominal-r`, `--fixture`, `--project` |
| `history` | `--fixture`, `--project` |

`--fixture` and `--project` must match across `run`, `import` and `derive`, or `derive` looks for
spectra where none were written. On `run`, `--yes` skips the hardware-in-place prompt and `--mock`
measures a simulated fixture.

| Artifact | Install | Gives you |
|---|---|---|
| `blank_short` | Jumpered channel (CE–WE shorted) | `R_fixture`, `L_lead` → unlocks series correction |
| `blank_load` | Precision resistor, `--nominal <ohms>` | End-to-end correction error |
| `reference_cap` | Low-loss C0G/NP0, `--nominal <farads>` | Measured phase floor → qualified upper bounds |
| `reference_r` | Reference resistor, ≥1 per decade | True \|Z\| window for this fixture |
| `blank_open` | Bare, uncast board | Whether OSL correction is legitimate at all |

`status` names the next artifact; `derive` reports which single artifact would unblock each
remaining capability.

**Two-electrode mode is mandatory for every reference.** Tie RE to CE at the connector before
measuring any two-terminal reference — short, load resistor, capacitor or bare board — because a
two-terminal load gives the reference stripe no ionic path and RE then floats onto a capacitive
divider whose ratio is not reproducible even at fixed load. A value taken any other way is refused
at derive time. `derive --declare-electrode-mode` persists a mode and touches only rows recorded as
`unknown`; a spectrum explicitly recorded as three-electrode stays refused.

**`--re-connection`** says what physically closed the potentiostat's control loop, which is a
different question from how the cell was sensed.

| Value | Means | Loop closed? |
|---|---|---|
| `unverified` | Nobody recorded it (default) | unknown |
| `tied_to_ce` | RE jumpered to CE at the connector | yes |
| `bridged_by_sample` | The cast film spans RE to the electrodes | yes |
| `open_by_geometry` | Nothing spans the gap to the RE stripe | no |
| `connected` | The wire is on; says nothing about what spans the gap | yes |

Use `tied_to_ce` for any two-terminal reference and for open blanks: it closes the loop while
making RE read the counter electrode, so the measurement is two-electrode by construction and its
configuration factor is 1, not 2. Recording it as `connected` or `bridged_by_sample` earns the
spectrum a K = 2 it has not established — a clean 2× error with a perfect-looking fit.

```bash
softae-commission import blank_short --file ch1_short.csv \
    --electrode-mode two --re-connection tied_to_ce --channel 1 --fixture mux16
```

**`--nominal`** is always in **base SI units** with no suffix: `470e-12` for 470 pF, `1e-9` for
1 nF, `1e6` for 1 MΩ. Supply it for any marked part; it is recorded with the measurement and is not
recoverable later. `derive` reports the measured-to-marked ratio, so a mis-keyed exponent is
visible where a bad part would be. Same rule for `--nominal-cap` / `--nominal-load` / `--nominal-r`.

**Deriving from a partial channel set.**

```bash
softae-commission derive --fixture mux16 \
    --channels 1-32 \        # the FULL channel set the calibration must cover
    --representative 3 \     # the MEASURED channel whose constants the rest inherit
    --nominal-load 1e6
```

`--representative N` names one measured channel whose `R_short`, `L_lead` and `C_stray` the
unmeasured channels inherit; pick an unremarkable one, since an outlier exports its defect to every
inheritor. `--channels` is the full covered set, not the measured set — without it there is no set
to inherit into. Inheritance is recorded: inheriting channels are marked `channels_assumed`, later
corrections carry `inherited = True`, and each logs `eis_calibration_channel_assumed` with the
measured channel-to-channel spread. Skip both flags and a channel with no constant of its own gets
**no correction**; `derive` declines it and names this pair as the remedy.

| Destination | Contents |
|---|---|
| `measurements` table, `role != 'sample'` | Raw spectra, queryable like any other |
| `calibration/eis/<fixture_id>.toml` | Derived constants — **commit this file** |
| `eis_calibrations` table | Append-only history; successive sets are a drift measurement |

Staleness is by **hardware identity, not clock**: a `hardware_hash` over the board, channel-routing
and instrument-envelope config. A mismatch means the constants are dropped, not applied. There is
no expiry.

---

## 14. EIS engine, gates, cell constant, fixture correction

| `[eis] engine` | What it does |
|---|---|
| `gated` (**shipped**) | Admission gates, covariance, per-sample cell constant, upper bounds where the measurement is resolution-limited |
| `legacy` | Fit `R0-CPE0-p(R1,C0)`, take `R1`, divide |

Both return the same report shape, so flipping the key is the whole cutover and it is reversible
per run.

`engine` and `[eis.gates] enabled` are deliberately separate. `engine = "gated"` with
`enabled = false` runs every check and logs every verdict, and **failing points are still
dropped**: the flag withholds the refusal, not the removal, so a would-be REJECT is recorded as
SUSPECT. Shipped gate thresholds are engineering defaults from the specification, chosen without
reference to this rig's spectra; `softae-shadow review` section 7 derives candidates from real
spectra ([§15](#15-softae-shadow)).

**What the gates catch.** Admission gates run before any fit, because the expensive failure is a
fit that *succeeds* on a spectrum containing none of the physics being extracted. Every removed
point is recorded with a named gate and a reason.

- *Valley feature* — `R_sol` must come from the interior local minimum of `−Z″`, never the `|Z|`
  minimum (the high-frequency intercept ≈ `R_series`). The two can differ by more than 10× on one
  spectrum, and taking the wrong one has no other symptom.
- *Cross-spectrum duplicates* — bitwise-identical `|Z|` between independently measured spectra
  proves an instrument rail. The remedy is a **higher current range, not a lower amplitude**.
- *Kramers–Kronig* — tests whether the response could have come from any linear, causal, stable,
  finite system, without assuming a circuit. It fits a K–K-compliant Voigt ladder (log-spaced
  `R‖C` elements plus explicit `R`, `L` and, for a blocking cell, `C`) and reads the residuals, so
  nothing outside the measured band is referenced. `add_cap` follows `[eis.cell] blocking`. K–K is
  necessary, not sufficient, and cannot separate drift from nonlinearity.

| K–K key | Shipped | Meaning |
|---|---|---|
| `kk_resid_pct` | `3.0` | Residual percentage above which a point is inadmissible; tuned on this rig, not an engineering default |
| `kk_c` | `0.30` | μ-criterion for ladder order selection; tuned on this rig, not an engineering default |
| `kk_max_M` | `50` | Ceiling on the Voigt ladder's order. Lower it to buy back fitter time, at the price of ladder flexibility |
| `kk_max_truncate_frac` | `0.5` | Fraction of the band the low-f truncation may remove before the spectrum is rejected instead of cut |

**Cell constant.** `K = L_gap / ((t − h) · L_stripe)`, computed per sample from that sample's own
thickness. A single nominal thickness applied across a series is a defect.
`[eis.cell] electrode_configuration` records the wiring; this rig is 3-electrode, so in principle
σ = K_geom / (`k_config_factor` · R) with a factor of 2. **The factor ships unarmed
(`k_config_verified = false`, factor 1.0), which changes no number.** Arming needs two checks:

| Check | Scope | Recorded as |
|---|---|---|
| Stripe symmetry + RE centring | Per board, once | `[eis.cell] k_config_verified = true` |
| An ionic path from film to RE stripe | **Per sample** | `re_contact_verified=True` passed to `cell_constant_for_sample` |

There is deliberately no `re_contact_verified` config key: a board-level key would assert contact
for exactly the samples where it fails. Unsupplied means unverified, holding the factor at 1.0.
Until armed, absolute σ reports as *scale unqualified*; relative trends are unaffected, so
campaigns ranking formulations are valid now. `RE_IONIC_CONTACT = {bridged_by_sample}` is strictly
smaller than `RE_CLOSED_LOOP = {bridged_by_sample, tied_to_ce, connected}`. If contact is asserted
alongside `open_by_geometry`, resolution fails closed and logs.

**`[eis.fixture]`** subtracts the mux, ribbon and trace from the measurement. Gated engine only.

| `mode` | Behaviour |
|---|---|
| `auto` (default) | `none` until this fixture has a short blank, `series` the moment it does |
| `series` | Subtract `R_short + jωL_lead`; refuses with a reason if no short blank exists |
| `none` | Subtract nothing |

`auto` switches correction on after commissioning and off after a board swap, because a
`hardware_hash` mismatch has already dropped the constants. **OSL is not implemented**, and `auto`
declines it even when the artifacts would license it, saying so in the log. The open blank is still
worth measuring: an *unusable* open is positive evidence that shunt admittance is negligible, which
is exactly when short-only correction is exact.

| Gates on the raw instrument record | Gates on corrected data |
|---|---|
| Finiteness, monotonic-f, quadrant, magnitude window, phase noise, stuck instrument | HF inductive truncation, min-points, topology triad, valley feature, and the fit |

A correction must never rescue a failed measurement, so admission runs first; a spectrum rejected
at admission is never corrected at all.

A wrong constant announces itself two ways: any point the correction drives to `Re Z ≤ 0` that was
physical beforehand marks the spectrum SUSPECT, and `validate_load_blank` pushes a resistor of
known value through the correction end to end — pick that load comparable to the fixture, since a
6 Ω correction validated against 1 MΩ passes no matter what. Every analysed spectrum gets a
`fixture_corrections` row, including a declined one with its reason; an absent row means
uncorrected. Set `fixture_id` to match what you commission, since `softae-commission` takes its
`--fixture` default from the same key.

---

## 15. `softae-shadow`

A shadow campaign runs the gated engine with every data-quality gate observing rather than
enforcing: nothing is rejected, everything that would have been is recorded.

```bash
softae-shadow status
softae-shadow rehearse --dry-run --run-id <film-run>
softae-shadow rehearse --run-id <film-run>                  # → logs/rehearsal_<UTC>.log
softae-shadow review shadow_run.log --project ./runs/aug --run-id run_20260810T1400Z
softae-shadow review shadow_run.log --project ./runs/aug --emit-toml proposed.toml
```

Module form: `python -m softae.tools.shadow_review`.

### `status`

Read-only. Prints five keys — `[eis] engine`, `[eis] objective`, `[eis.gates] enabled`,
`[quality] enabled`, `[eis.fixture] mode` (with its `fixture_id`) — plus one verdict.

| Verdict | Config state | Exit |
|---|---|---|
| ARMED FOR A SHADOW RUN | `engine = "gated"`, `gates.enabled = false`, `quality.enabled = false` | 0 |
| GATED AND ENFORCING | Engine gated but a gate is enforcing — a cutover, not a shadow run | 1 |
| NOT ARMED | The legacy engine | 2 |

Ask it twice: before the run and after the revert. When armed it also prints a wall-time advisory,
because observe-only is the slowest analysis setting the rig has — a spectrum the gates would have
rejected pre-fit still reaches the fitter, and a fit with no arc to find takes the long way to
failing. Cost is set by **arc closure**, not by the gate verdict, so size the run by the clock
rather than the well count and run `rehearse` first.

### `rehearse`

Replays stored spectra through the same gated observe-only engine, so its log is one `review` reads
with no special case.

| Flag | Default | Behaviour |
|---|---|---|
| `--project DIR` | `[data] project_dir` | Where the corpus lives |
| `--run-id ID` | Most recent run with spectra | Which run to replay; the plan line names it |
| `--rounds N` | `2` | Rounds per `(leg, setpoint, channel)` cell |
| `--all` | off | Every spectrum in the run |
| `--limit N` | none | Hard cap applied **after** stratification; the summary reports dropped cells |
| `--seed S` | deterministic | Randomise the round picks for a sensitivity check |
| `--out PATH` | `logs/rehearsal_<UTC>.log` | The log; refuses to overwrite |
| `--tee` | off | Mirror to stdout |
| `--model NAME` | The fit row's `model_name` | Override |
| `--enforced` | off | Replay with gates enforcing, to measure what observing costs |
| `--dry-run` | off | Print the plan and projected duration; analyse nothing |

**Pass `--run-id`**, or the pre-flight rehearses whatever landed last — often a one-spectrum
reference-resistor import, which returns in seconds and tells you nothing. Read the plan line and
the spectrum count before the duration beside them. Selection is stratified and deterministic, so
the open-arc mix is a measured quantity.

| Guarantee | How |
|---|---|
| No database write | Corpus opened `sqlite3.connect("file:…?mode=ro", uri=True)`; `DataStore` is never constructed and `record_fit` never called |
| No config edit | The gated engine is chosen by a `settings=` argument; `[eis] engine` is never read or written |
| No rig | Analysis modules only; no instrument opened, no pose read, no stage moved |

Under `--enforced` a rejected spectrum never reaches the fit that would annotate its arc, so
`arc_state` stays NULL for what it rejects.

Two outputs: the **log** (engine events interleaved with `rehearsal_started`,
`rehearsal_spectrum_done`, `rehearsal_summary`; opened `utf-8`/`errors="replace"` by the tool
itself, so a gate detail containing `tan δ` does not kill the run on a cp1252 console) and a
**timing CSV** at `<out>.timing.csv` with `seconds`, `verdict`, `arc_state`, `sigma_mode` and
provenance per spectrum, written incrementally.

Run it before any bench shadow run and again after a recalibration. A rehearsal's section 7 is
evidence about the recommender, not thresholds for the rig — nothing from a rehearsal is pasted
into `softae_config.toml`.

### `review`

| Flag | Default | What it adds |
|---|---|---|
| `log` (positional) | required | The redirected run log, or `-` for stdin |
| `--project DIR` | log only | The DataStore half: section 5 (per-channel σ) and 5b |
| `--run-id ID` | Most recent run | Which run in that project to read |
| `--min-evidence N` | `20` | Spectra a metric must be observed on before section 7 may propose a value |
| `--emit-toml PATH` | not written | Write the paste-ready `[eis.gates]` / `[quality]` block to a **new** file |

**Gate verdicts are not persisted.** A gated campaign writes `gate_verdict = NULL` to
`fit_results`, so the only record of a would-reject is the structlog stream on the console:
`... | tee shadow_run.log` is a required step of the run.

Per-*gate* counts are exact (the gate name heads every issue string). Per-*channel* counts are
positional and sound only for verdicts emitted during the workflow's auto-fit; verdicts from
objective extraction land on whichever channel was routed last and are counted separately as
unattributed. The `eis_spectrum_metrics` event behind section 7 carries `channel` and a content
fingerprint outright, so its population is attributed and de-duplicated.

**Section 5b** (with `--project`) prints railed fits per channel — rows since the railed-fit
demotion carry `success = 0` and an `error_msg` naming the bound, earlier rows carry `success = 1`
with `R1` exactly on it — and arc-closure state counts, read column first with a `gate_log_json`
fallback. `sigma_is_bound` is labelled a stamped default (0 on every row, not an observation).

**Section 7** computes where each threshold would sit given the decision to arm, and never decides.

| Column / marker | Means |
|---|---|
| `!` before a key | Behavioural: changing it moves a stored *number*, not only a verdict. Re-fit before trusting any σ produced under it |
| `fired@def` / `rej@rec` | Spectra the gate fired on at its default, and how many the proposal would reject |
| `hold (unexercised)` | 0 spectra failed at the default; untested is not validated, so the default stands |
| `hold (unimodal)` | A gap rule found no two populations to separate |
| `recommended (measures-the-rig)` | The gate fired on ≥90 % of spectra: at its default it is measuring the rig, not the sample |

Act on the final **joint** count; per-key counts do not add, because one bad spectrum routinely
fails several gates. Two further blocks follow: *not recommendable from a shadow run* (`kk_c`,
`bound_tol`, the `blank_*` and `geom_*` keys) and *observed but unconfigurable* (hardcoded
constants such as `cross_check_pct` and `runs_z`). A proposed value admitting many open-arc
spectra deserves suspicion, since `R1` recovered past the apex is systematically biased high.

### `--emit-toml`

Four steps; the tool performs the first three.

1. `softae-shadow review shadow_run.log --project <dir>` — read sections 1–6 gate by gate.
2. Read section 7 for where each threshold would sit and what it would cost.
3. `--emit-toml proposed_thresholds.toml` writes the paste-ready block.
4. **You** paste what you accept into `softae_config.toml`, and **you** decide arming.

Every emitted value carries its rule, `n`, and fired → rejected counts as a trailing comment;
refused and held keys are emitted commented out with their reason. `enabled` is written `false` in
both sections and is never written otherwise. Two absolute refusals: `--emit-toml` will not write
to the live config, and will not overwrite an existing path. Either prints the reason and exits 1.

**Evidence floor.** `--min-evidence` defaults to 20 spectra per metric: at n = 20 the empirical P95
is the second-largest observation, so a fence rests on two points rather than the single worst
spectrum. Below the floor a key is refused by name with a reason. The packaged shadow spec
(`examples/shadow_campaign.toml`) ships `budget = 16`, deliberately below the floor, so a 16-well
run recommends nothing. **Raise `budget` to 32** if you want thresholds. Lowering `--min-evidence`
is possible and visible in the output, but buys a number the sample does not support.

The full bench procedure — which keys to flip, in what order, how to revert — is
`docs/SHADOW_CAMPAIGN.md`.

---

## 16. `softae-thickness`

Film thickness can be confounded: assign levels in channel order at cast time and a channel
artifact becomes mathematically indistinguishable from a thickness effect. **Plan before you cast;
confounding cannot be undone afterwards.**

```bash
softae-thickness plan --levels 100,150,200,250 --channels 1-32
softae-thickness cast --plan geo-2026-08-06                 # DRY RUN
softae-thickness cast --plan geo-2026-08-06 --execute       # drives hardware
softae-thickness record --channel 7 --um 148.2 --uncertainty 3.0
softae-thickness check --plan geo-2026-08-06
softae-thickness fit  --plan geo-2026-08-06
softae-thickness list --plan geo-2026-08-06
```

Module form: `python -m softae.tools.thickness`. `--project` (default `[data] project_dir`) is
accepted by every subcommand.

| Subcommand | Flags | When |
|---|---|---|
| `plan` | `--levels` (req), `--channels` (req), `--id`, `--seed`, `--max-correlation`, `--notes` | Before casting; assigns levels so level and channel index are uncorrelated |
| `cast` | `--plan` (req), `--board`, `--execute`, `--no-drift-control` | At the rig; resolves a plan into a cast order and checks the board |
| `record` | `--channel` (req), `--um` (req), `--uncertainty`, `--plan`, `--run`, `--level`, `--instrument`, `--operator`, `--notes` | At the profilometer, one channel at a time |
| `check` | `--plan`, `--run`, `--max-correlation` | After casting; compares cast against planned |
| `fit` | `--plan`, `--run`, `--fixture` | σ from the slope, and *h* if `G_fixture` exists |
| `list` | `--plan`, `--run`, `--plans` | Show measurements, or list the plans |

- **`cast --execute` drives real hardware.** Without it `cast` is a dry run that resolves the
  order, checks the board and touches nothing. See
  [§8](#8-stopping-safe-exit-interlock-head-state) for the interlock every real-motion command also
  needs.
- **`--no-drift-control`** omits the end-of-session repeat cast, which is what separates a genuine
  thickness trend from session drift.
- **`--seed`** (default 0) makes the assignment reproducible.
- **`--max-correlation`** is the |r| ceiling between level and channel index that `plan` designs
  under and `check` tests against. A confounded series exits **3**, distinct from a plain failure
  (1), so a script can branch on it.

Run `check` after casting, while a re-cast is still cheap.

---

## 17. `softae-equilibration`

Records σ(t) while the chamber is brought to condition, fits the relaxation, and derives the
conditioning hold time from the fit. This is an overnight bench run.

```bash
softae-equilibration plan --save plan.toml
softae-equilibration run --from-plan plan.toml --channels 1-16 --execute
softae-equilibration fit    --run <run_id>
softae-equilibration report --run <run_id> --tol-rel 0.02
```

`python -m softae.tools.equilibration` is the exact equivalent and is what the tool prints in its
own suggested commands. (Its `--help` epilog still says the console script is not installed in this
venv; it is. The recommendation to use the module form is unaffected.)

**`plan` and `run` share no state**, so **every design flag not repeated on `run` reverts to its
default.** Two answers: `plan --save plan.toml` writes the fully resolved design, defaults
included, and `run --from-plan plan.toml` executes it verbatim (a flag typed alongside
`--from-plan` still wins, but as a printed diff against the file, repeated in the thermal
confirmation); and `run --channels` is **mandatory** with no default, because a defaulted channel
set would energise exactly the channels a subset was chosen to exclude. `--from-plan` supplies it.

Design flags appear on both `plan` and `run`, which is what makes a plan file replayable.
`--project` and `--mock` are available throughout.

| Group | Flags | Decides |
|---|---|---|
| Design (`plan` + `run`) | `--channels`, `--temperatures`, `--legs`, `--rh`, `--rounds`, `--preset`, `--round-period-s`, `--measured-per-channel-s`, `--circuit-model`, `--electrode-l-cm` / `-t-cm` / `-w-cm`, `--thickness-method`, `--fixture` (plan only) | Which channels, which setpoints, how spectra become σ |
| Settling (`plan` + `run`) | `--settle on\|off`, `--settle-tol-rel`, `--settle-n-rounds`, `--settle-min-channels`, `--min-hold-first-s`, `--min-hold-s` | When a setpoint has been held long enough to stop |
| Execution (`run`) | `--from-plan`, `--execute`, `--yes`/`-y`, `--quiet`, `--telemetry-interval-s` | Whether hardware moves, and how loudly |
| Analysis (`fit`, `report`) | `--run` (req), `--relaxation-model`, `--tol-rel`, `--n-settle` | How σ(t) is fitted offline and which tolerance the verdict uses |

- **`--rounds` is a ceiling, not a count.** A setpoint stops as soon as σ has settled, the hold
  floor has elapsed, and the fitter's minimum rounds have run.
- **`--settle-tol-rel` must exceed the run's own noise floor**, or no hold length satisfies it and
  every setpoint runs to its ceiling. The run says so, per setpoint, when the criterion is
  unsatisfiable.
- **`--settle on|off`** rather than `--no-settle`, because a `store_true` cannot be written into a
  plan file and retyped from it. `off` restores fixed-count behaviour exactly.
- **`--execute` is what makes anything real.** Without it `run` opens nothing. `--yes` skips the
  thermal confirmation; `--quiet` drops only the live status line.
- **`--circuit-model` vs `--relaxation-model`.** The circuit model (e.g. `simpleSalt`) is fitted to
  each *spectrum* on `plan`/`run`; the relaxation model (e.g. `exponential`, or `none` for t_tol
  only) is fitted to *σ(t)* on `fit`/`report`. `--model` is a working alias on both, meaning
  whichever is right for that subcommand.
- **`-v` / `--verbose`** works on the top-level parser and on every subcommand, so `... -v run …`
  and `... run -v …` both take. It is genuinely noisy.

---

## 18. `softae-env`

A chamber hold that measures nothing: conditions a board at one humidity for a stated duration,
with a rig claim, a watchdog and a defined teardown. **One axis, deliberately** — there is no
`--temp` and no heater is driven; `softae-equilibration run` owns the temperature hold. Chamber
temperature is reported beside the humidity because one `get_TH` transaction returns both.

```bash
softae-env plan --rh 45 --duration-h 4                        # prints; opens nothing
softae-env hold --rh 45 --duration-h 4 --execute
softae-env hold --rh 45 --execute --yes --quiet > hold.log    # unattended, until signalled
softae-env hold --rh 45 --duration-s 600 --execute --mock
```

`python -m softae.tools.env_hold` is the exact equivalent and is currently the form that resolves
here — see the install-state note in [§5](#5-command-line-tools). `plan` and `hold` take the same
flags, so a plan becomes a run by changing one word and adding `--execute`.

| Flag | Default | Decides |
|---|---|---|
| `--rh PCT` | **required** | The setpoint; validated against the driver's own cap at `set_setpoint` time |
| `--duration-s S` / `--duration-h H` | Mutually exclusive; **both omitted ⇒ hold until signalled** | How long |
| `--execute` | off | **Without it nothing is opened** |
| `--yes` / `-y` | off | Skips the typed confirmation, with a printed acknowledgement |
| `--mock` | off | Simulated drivers recording to an isolated `<project>/mock` store; claims no rig |
| `--project PATH` | `[data] project_dir` | Where the run is recorded |
| `--quiet` | off | Drops the heartbeat line only; plan, milestones and verdict still print |
| `--heartbeat-s S` | 300 | Seconds between heartbeat lines; watchdog sampling is independent of this |

**Three separate acts before anything actuates.** `hold` is a dry run unless `--execute` (without
it the tool prints the plan, says so, opens no instrument and creates no run row); then a typed
confirmation — the literal word `yes`, not `y` — stating the setpoint, the duration (or "until
interrupted") and that the humidifier will actuate unattended; and `--yes` for scripts, since on a
non-TTY the prompt reads end-of-input as a **decline** and exits 2.

**Rig claim.** A real hold claims the rig as `tool:env-hold:<run_id>`. If someone else holds it,
the tool refuses **before it opens anything** — the lock is checked before the store exists, so a
refusal leaves no run row behind — prints who holds it, and exits **4**. The usual holder is the
GUI, which claims the rig for its entire connected life, not only while a run executes: disconnect
in the GUI (or close it), then run the tool. `--mock` claims nothing and can neither be refused nor
lock out a real run.

**Watchdog.** The RH watchdog from `[safety]` is attached with the same thresholds every other
RH-watched run uses and writes durable alert rows to the DataStore. A sustained excursion, or a
sensor unreadable for the whole window, raises at CRITICAL on the console. **It does not stop the
hold.** Unreadable values render as `--` on the heartbeat line, never as `0.0` and never as a stale
number.

### Stopping a hold

Ctrl-C (and `SIGBREAK`) triggers the teardown:

1. The PID loop is stopped and the humidifier is left **purging dry** — duty at `out_min`, the same
   end state as the GUI park, so the chamber keeps its dry state over the ≈25 s until the Trinket's
   deadman shuts both valves. If the driver exposes no `safe_dry()`, the fallback zeroes the duty
   instead and says so.
2. The verdict prints **to stderr**, so `--quiet > hold.log` still shows on the terminal whether
   the humidifier came off:
   - `Humidifier DRY-PURGED: PID stopped, setpoint 0, duty held at <d> = dry air.` — success; gas
     is still flowing and that is deliberate.
   - `!! NO DRY PURGE -- THE HUMIDIFIER WAS ZEROED INSTEAD:` — hardware safe (PID stopped, duty 0,
     both valves shut), but the chamber collapses to room RH within tens of seconds.
   - `!!!! NO DRY PURGE WAS CONFIRMED, AND THE HUMIDIFIER'S STATE IS UNKNOWN:` — it may still be
     driving the setpoint. Check it at the rig.
3. The run row is finalized, the rig claim released, the instruments disconnected — in that order,
   because a disconnected driver can no longer be written to.

A **second** Ctrl-C during teardown reaches the default handler, because the handler uninstalls
itself.

| Exit | Meaning |
|---|---|
| 0 | A bounded hold reached its duration, or an until-signal hold was signalled |
| 1 | A bounded hold was interrupted early; or the driver refused (setpoint cap, comms, instrument error) |
| 2 | The confirmation was declined, including end-of-input on a non-TTY without `--yes` |
| 4 | The rig is held by someone else; nothing was opened |

Every path finalizes the run row as `done`, `interrupted`, `aborted` or `error`. An interrupted
until-signal hold exits 0 and still records `interrupted`. A hard-killed host (`SIGKILL`, End Task,
power cut, blue screen) runs no teardown, so the humidifier latches its last duty for the ≈25 s
deadman, not for hours.

---

## 19. `softae-eis-validate`

Adaptive EIS acquisition ships inert behind `[eis.scout] actuate`. It works scout-then-measure: the
configured baseline sweep runs first and *is* the measurement whenever the verdict is `ok`; only an
inadequate spectrum earns a second, wider sweep. This tool takes both arms on the same cell seconds
apart at one held condition, and is the only shipped route that can say GO or NO-GO on the adaptive
path.

```bash
softae-eis-validate run \
  --channels 18-32 \
  --rh-setpoint-pct 20 --temp-setpoint-c 25 \
  --validation-name adaptive-2026-09 \
  --settle-criterion both --settle-rate-tol-dec-per-h 0.05

softae-eis-validate report --validation-name adaptive-2026-09
```

`--channels`, `--rh-setpoint-pct`, `--temp-setpoint-c` and `--validation-name` are **required with
no defaults**. `python -m softae.tools.eis_validate` is the exact equivalent — see the install-state
note in [§5](#5-command-line-tools). `--dry-run` prints the plan and projection and runs nothing;
`--mock` uses a grid-aware synthetic backend that never prompts, never arms and never emits a GO.

**Resolving window.** `Extended` reaches 1.351 Hz, so its arc closes only for an apex above
~13.5 Hz; below that its `R1` is an extrapolation. The baseline grid (6.475 Hz on the shipped
`Quick` preset) returns `ok` for any apex above 64.75 Hz, and on `ok` the two arms are
byte-identical. So the experiment resolves **only apexes in 13.5–65 Hz**: above it a cell is
CONTROL, below it UNRESOLVED, and neither carries information about the decision. The **setpoint**
is the lever that decides how many cells land in that window; the channel count is not. The run
prints its own apex histogram before every refusal.

**End state and unheated running.** A park drives the heater to its safe setpoint and suspends
anti-clog purging, so `--resume` **always** re-runs the full approach and settle gate before a
single sweep; no flag skips it. `--end-state hold` holds temperature and **cannot** hold humidity
(the Trinket wants a continuous heartbeat with a ≈25 s deadman), so the run prints the exact
`softae-env` command that takes the axis over, and it must be started inside that window.
**A `--temp-setpoint-c` at or below 10 °C is a condition, not a target**: this rig has a heater and
no chiller, so 10 °C means *stop heating* — the temperature approach is skipped and the hold watch
grades against where the board started. Do **not** widen `--tolerance-c` to get past an unreachable
setpoint; that loosens the over-temperature gate for the whole run.

| Group | Flags | Decides |
|---|---|---|
| Condition | `--rh-setpoint-pct`, `--temp-setpoint-c`, `--rh-tolerance-pct`, `--tolerance-c`, `--rh-approach-timeout-s`, `--temp-approach-timeout-s`, `--approach-dwell-s` | Where the chamber goes, and when it counts as arrived |
| Settling | `--settle`, `--settle-tol-rel`, `--settle-criterion`, `--settle-rate-tol-dec-per-h`, `--settle-max-rounds`, `--settle-max-hold-s`, `--rh-stability-pct`, `--soak-h` | When the board has stopped moving, and what to do if it has not |
| Survivors | `--survivors`, `--min-treatment`, `--max-consecutive-failures` | Whether a partial board may proceed, and what it may conclude |
| Measurement | `--channels`, `--baseline`, `--reference-preset`, `--order`, `--max-follow-ups`, `--drift-check`, `--retries` | Which cells, which grids, how pairs are sequenced |
| End state | `--end-state`, `--resume`, `--yes`, `--project`, `--out`, `--dry-run`, `--mock` | Whether hardware moves, and what is left behind |

| Flag | Default | Note |
|---|---|---|
| `--settle-criterion` | `deviation` | `both` routes on deviation and reports the rate beside it |
| `--settle-tol-rel` | `0.10` | Relative deviation of σ from its own window mean; dimensionless, **not** %RH. The run refuses above `0.50` |
| `--settle-rate-tol-dec-per-h` | `0.05` | Decades of σ per hour. Required by `--settle-criterion rate` and `both`; the run refuses above `0.5` |
| `--settle-max-rounds` | `14` | The ceiling in **rounds**, the unit every criterion reads |
| `--settle-max-hold-s` | derived | From `--settle-max-rounds` × the **achieved** round period (sweep block plus sleep). A typed value wins and is announced as an override |
| `--rh-stability-pct` | `1.5` | How far per-round RH **medians** may span across the judged window. `0` or `off` disables the gate. Not `--rh-tolerance-pct`, which judges only the approach |
| `--survivors` | `off` | Needs `--settle-criterion rate` or `both` |
| `--max-consecutive-failures` | derived | `3` under `--survivors off`; the board size under `--survivors on`, where a dropped cell's failures stop counting |
| `--approach-dwell-s` | `600` | Seconds each axis must stay in band before it counts as arrived; `0` restores first-in-band-poll behaviour |
| `--rh-approach-timeout-s` / `--temp-approach-timeout-s` | `5400` / `1800` | Ceilings, not expected durations |
| `--soak-h` | `0` | **Hours**, not seconds. The settle gate proves the rig stopped moving; the soak is the sample's own equilibration. Settle time counts against it |
| `--reference-preset` | `Extended` | Pass **`Longest`** (capital L) to widen the resolving window. Preset lookup is case-sensitive: `longest` logs `eis_preset_unknown` and silently falls back to built-in defaults |
| `--baseline` | the modality's configured preset | Not a literal |
| `--order` | `ref-first` | `alternate` is a diagnostic, not a design element |
| `--max-follow-ups` | `1` | 1 validates what ships; >1 is exploratory |
| `--min-treatment` / `--drift-check` | `6` / `3` | Cells that must land in the resolving window; cells re-measured at the end of the block |
| `--end-state` | `park` | `hold` keeps temperature only |

**`--survivors on` changes what the run may conclude**, not how long it waits. At the ceiling it
partitions instead of refusing: cells the rate criterion certified quiet proceed, cells that could
not be *judged* are dropped with a recorded reason, and cells **proven** to be moving still refuse.
Every number a survivor run produces is conditional on settling — fine for "does the scout resolve
the arc?", wrong for any hold-time or objective number. Two details trip people up: the RH preroll
runs under `--settle-criterion rate` and under `both --survivors on`, but **not** under plain
`both`; and under `both --survivors on` the detection-floor refusal fires only when *no* cell
certified quiet.

**Watching a run.** It publishes `events.jsonl` and `conditions.json` in
`<project>/runs/<run_id>/`, and the rig claim's `log_path` names that directory. The per-round
`settle_round` record carries per-channel deviations, participating cells, the achieved RH spread,
and under a rate criterion the per-channel standard error, relative residual, upper bound and the
window's detection floor. **σ itself is deliberately absent from the stream.** Raw settle sweeps
are persisted under `eis/` tagged as the `settle` arm and excluded from every reader unless asked
for.

**Reading the report.** `report --validation-name <name>` re-evaluates the pre-registered rule over
whatever has been persisted. Every measurement row carries a `hold_certified` stamp:

| Value | Meaning |
|---|---|
| `settled` | The whole board was certified quiet before the arms ran |
| `survivors` | A partitioned settle phase certified this cell quiet; others were dropped |
| `dropped_moving` | Dropped: proven to be still moving |
| `dropped_unevaluable` | Dropped: could not be judged either way |
| `disabled` | `--settle off` — no gate ran |
| `pre_settle` | A settle sweep taken before the gate spoke; never an experiment arm |

Only a board-level `settled` licenses a verdict. A survivor run is not `settled` at board level.
Under `--settle off` every row is stamped `disabled` and the outcome is **WITHHELD**, announced in
the projection before the run starts; a `--mock` run is WITHHELD for the same reason. Uncertified
rows keep their numbers in every accuracy table but lose their anonymity: marked `(uncertified)`
wherever they print, counted beside every criterion computed over them, and partitionable offline
off `payload["cells"][*]["stillness_certified"]`.

**A run that ends in `ceiling` has measured nothing.** Read the per-round table and the apex
histogram before changing any flag; both print on every refusal and both are built from sweeps that
were taken anyway.

---

## 20. Adding a measurement modality

The campaign path performs exactly one lookup (`get_modality(spec.measurement.modality)`), so a new
stream registers in one place.

```python
from softae.core.modality_registry import Modality, ModalityDisplay, register_modality

register_modality(Modality(
    name="my_stream",
    build_measure_step=...,  # (channel, spec) -> the per-channel step, or None
    router_factory=...,      # () -> the router that persists what comes back
    objectives={},           # what the optimizer may be told; may be empty
    prepare_run=...,         # runs once before any measure step (EIS writes .mscr here)
    display=ModalityDisplay(display_name="My Stream"),
))
```

- **Registration is an explicit call, never an import side effect**, so the set of available
  modalities does not depend on import order.
- **An analysis-only stream is first-class.** `objectives = {}` is legitimate. Tagging its steps
  `measurement = "image"` (anything but `primary`) keeps it out of the loop-closure predicate.
- **The sample spine comes for free.** Any step carrying `tags["channel"]` inherits its
  `sample_uuid`.

A worked example ships in `src/softae/analysis/image/`, built with no edit to `core/`,
`workflows/`, `analysis/eis/` or `drivers/`.

*(coming soon)* Three edges are built but not reachable from a shipped campaign:

| Not yet | Why |
|---|---|
| Running the `image` modality | `register_image_modality()` is deliberately not called at startup; one line of `core/` wiring is left open |
| A `measurements` row for a non-EIS capture | The write path is still EIS-typed and readers do not filter on `modality`, so an image row would surface to the Analysis browser as a spectrum with NULL fields. The payload self-links via its `attrs` meanwhile |
| A modality-neutral router contract | `ResultRouter` / `RouterContext` still live inside `analysis/eis/`; a second modality duck-types rather than importing the EIS package |

---

## 21. Documentation site

```powershell
pip install -e ".[docs]"    # mkdocs-material + mkdocstrings
mkdocs serve                # live preview at http://127.0.0.1:8000
mkdocs build                # static site in site/
```

API reference pages are auto-generated from docstrings by `mkdocstrings[python]`. Adding or
renaming a public module requires a corresponding stub in `docs/api/`.

---

*Revised 2026-09-21 against the working tree at 7e445f5 (audit: docs/SubAgent docs/user_guide_rev1_audit.md).*
