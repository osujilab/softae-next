# Master guiding document for developing softae-next (Soft Matter Autonomous Experimentation system in the Osuji Lab)

We are in the process of building & upgrading an autonomous experimentation system (softae-next) described in README.md and index.md under ".\docs". 
This is a master guiding document for researchers and AI agents to read for rapid, general context evaluation.

Absolute path to project home directory:
`C:\Users\Osuji\AppData\Local\Programs\Python\Python311\Scripts\softae-next`

## Project documentation and associated purpose

1. Introductory document: README.md.
1. Adhering to the clean-break migration from old system architecture, with active roadmap tracking in `ACTION_PLAN.md`.
1. Adhering to ACTION_PLAN.md for next-highest priority tasks and alignment with AND/OR transcendence of the migration.
1. Detailing progress through PROGRESS.md.
1. User-facing USER_GUIDE.md for up-to-date visibility on system functionality.

## Master guiding principle:

### Orchestrator Workflow (Mandatory)

File reads are **dual-path**: the orchestrator reads directly *and* spawns a subagent to read the
same material in parallel, then compares the two interpretations. Edits and implementations remain
delegated to subagents.

1. **Dual read** → the orchestrator reads the target files directly while a research subagent reads
   them independently and in parallel. Neither is given the other's conclusions beforehand.
1. **Compare interpretations** → the orchestrator diffs its own reading against the subagent's.
   Where they agree, work proceeds. **Divergences are surfaced to the user** — what each concluded
   and where they part — rather than silently reconciled in favour of either one.
1. **Spawns a research subagent** → produces a spec doc at `docs/SubAgent docs/<topic>.md`
1. **Reviews** the spec (or asks the user to review)
1. **Spawns an implementation subagent** → writes code + tests
1. **Runs tiered tests** in the terminal → neighbourhood after each edit, package before accepting the task, full serial suite before a commit (see "Tiered Test Execution")
1. **Spawns update subagents** → updates PROGRESS.md, USER_GUIDE.md, ACTION_PLAN.md and other ".\docs" files as needed
1. **Recurrent workflow reminders** → occasionally reminds the system to retain this workflow structure (roughly every three prompts).

The orchestrator **never** edits source files directly — edits stay delegated. Reading directly is
required, not merely permitted, because the comparison in step 2 depends on the orchestrator having
formed its own independent view.

#### Subagent capabilities

Every spawned subagent is granted **full command-line access** (Bash and PowerShell), not just
file-read and edit tools. Subagents run tests, invoke git, execute scripts, and inspect the
environment themselves rather than reporting findings back for the orchestrator to act on. A
subagent that needs to run a command runs it — with the four exceptions below.

##### Exception — commands that return to the orchestrator

Four classes sit **outside** the grant. A subagent that needs one stops, reports the exact command
it intends to run and why, and waits for authorisation:

1. **Destructive filesystem operations** — recursive deletes, overwriting or truncating files
   outside the subagent's own task scope, moving or clobbering anything under the DataStore.
2. **Git history rewrites** — `reset --hard`, `push --force`, `rebase`, `clean -fd`, branch or tag
   deletion. Ordinary `add` / `commit` / `status` / `diff` / `log` stay inside the grant.
3. **Hardware actuation** — any command that drives real rig equipment: pump dispense, stage
   motion, piezo firing, potentiostat measurement, temperature or humidity setpoint changes. This
   system controls physical instruments in the Osuji Lab. A stage move into an occupied well, or a
   dry pump fired at speed, has no undo and can destroy a board or the hardware itself. Dry-run and
   simulation paths (`actuate=false`, simulated backends, offline BO) remain inside the grant.
4. **Terminating a process the subagent did not launch** — killing a process you started, by PID,
   stays inside the grant; every other process escalates. **Killing by image name is never inside the
   grant** — `taskkill /IM`, `pkill`, `killall`, `Get-Process <name> | Stop-Process` — because on this
   machine the operator's GUI, the VS Code extension hosts and pytest all run the same interpreter
   from the same virtualenv, so no name- or path-based filter can separate them. See "5. Process
   Hygiene" for how to identify a process and what to capture before terminating it.

The orchestrator may authorise class 2 on its own judgement, and class 4 when it can establish the
process's provenance — it knows what it spawned and the subagent does not. **Classes 1 and 3 go to
the user** — the orchestrator does not self-authorise a destructive write or any motion of real
equipment — **and so does any class-4 kill whose target the orchestrator cannot identify**: an
unidentified process may be the operator's GUI, whose consequence class is class 3's, so it gets
class 3's treatment.

## Additional Guiding Principles

### 2. Visibility at Every Step
Every development task must be **visible** — to the developer, to reviewers, and to future selves.
This means:

- Each task produces a **spec document** (in `docs/SubAgent docs/`) before implementation begins.
- Each implementation is accompanied by **tests** that pass before the task is marked done.
- **PROGRESS.md** is updated after each task with: what changed, what files were touched, and the new test count.
- **Session entries** in PROGRESS.md provide a timestamped audit trail.
- Higher-level overviews of development tasks are verified against the real-time state of the codebase and the project architecture (`architecture.md`) plus roadmap state in `ACTION_PLAN.md`, and updated/revised accordingly. This is meant to be a living document that can evolve based on novel constraints and targets.

### 3. Elegance and Conciseness

After each development task, the system should perform cross-cutting reviews of the codebase and associated tests for elegance and conciseness of the documentation and implementation. It is important to prune and condense summaries to readable formats, and implement visualization where structures and hierarchy are apparent but not visible.

1. **Post-task review sweep** — After every implementation subagent completes and tests pass, a dedicated review subagent inspects the touched files *and* their immediate neighbors (imports, callers, tests) for:
   - Dead code or unused imports introduced by refactoring.
   - Duplicated logic that could be consolidated into an existing utility or base class.
   - Overly defensive error handling that duplicates guarantees already provided by the framework or parent class.

2. **Documentation density rule** — No single document should grow unboundedly.
   - **PROGRESS.md**: Once a phase is fully complete, its session-level entries are collapsed into a single "Phase N Summary" paragraph, and the granular session log is archived to `docs/archive/progress_phaseN.md`.
   - **Spec documents**: After implementation is merged and tests pass, the spec moves from `docs/SubAgent docs/` to `docs/SubAgent docs/old_specs/` with a one-line status note appended at the top.
   - **USER_GUIDE.md**: Reviewed for stale content after any feature addition or removal; sections for incomplete features are marked `*(coming soon)*` rather than left half-written.

3. **Structural visualization** — Where hierarchy, data flow, or state transitions exist but are not immediately visible in prose, the system should produce or update:
   - A **module dependency diagram** (Mermaid in `docs/architecture.md`) showing `src/softae/` package relationships — updated whenever a new module is added or an import path changes.
   - A **workflow execution flowchart** (Mermaid) for the DAG-tier executor, kept in sync with `workflow_executor.py`.
   - **Table-format summaries** (rather than bulleted lists) for any catalogue of items > 5 entries (instruments, config keys, error types, CLI flags).

4. **Test suite hygiene** — Tests are reviewed alongside implementation for:
   - **Redundant coverage**: If two tests assert the same invariant via different surface APIs, the less direct one is removed.
   - **Fixture reuse**: Shared setup patterns appearing in ≥ 3 test files are extracted into `tests/conftest.py` fixtures.
   - **Naming clarity**: Test names follow `test_<unit>_<scenario>_<expected>` (e.g., `test_manager_acquire_multiple_sorted_order`). Freeform names are renamed during review.

5. **Code conciseness standards** — The review subagent enforces (within reasonable tolerances of +/- ~50% given the specific implementation case):
   - **Single Responsibility**: No function exceeds ~50 statements; no module exceeds ~400 lines. Violations trigger a split-and-refactor task.
   - **Flat over nested**: Prefer early returns / guard clauses over deeply nested `if/else` trees (max 3 levels of indentation in logic paths).
   - **Naming as documentation**: Variable and function names should make inline comments unnecessary for *what* the code does; comments are reserved for *why*.

6. **Cross-cutting review cadence** — In addition to per-task reviews:
   - Every **5 development sessions**, a full-codebase review subagent is spawned to audit import graphs, docstring completeness, and config-key coverage between `softae_config.toml` and the loader.
   - Every **new phase milestone**, the orchestrator spawns a subagent to reconcile `ACTION_PLAN.md` against the actual file tree and test count, pruning completed items and surfacing drift.

### 4. Tiered Test Execution

Regression confidence is bought by **selection**, not by parallelism. Collection alone costs ~8 s on
this rig, so a narrowed run is essentially free: a typical task's real regression surface is ~225
tests in ~25 s, whereas the full 3552-test suite spends ~17 min largely re-proving subsystems the
edit cannot reach. Three tiers, each with its own owner and its own gate:

| Tier | Scope | Who runs it | When | Typical cost |
|---|---|---|---|---|
| 1. Neighbourhood | The touched module's own test file plus its direct neighbours — importers, callers, and the tests of modules it imports | Implementation subagent | Immediately after each edit | ~25 s |
| 2. Package | The package glob covering the touched area (e.g. `tests/test_eis_*.py`) | Orchestrator | Before accepting a task as done | 1–3 min |
| 3. Full serial suite | Everything, serial | Orchestrator | Before a commit, and at the end of a task arc | ~17 min, run backgrounded |

- **The full suite is the commit gate, not the per-edit gate.** A green tier-1 run is what licenses
  the next edit; a green tier-3 run is what licenses a commit. Tier 3 is the tier that actually
  catches cross-cutting regressions, so it is never traded away before a commit — it is backgrounded
  so it does not block.
- **Test time is a budgeted resource, and the budget is planned before the work starts.** Tier 3 now
  costs ~28 min. On a multi-step arc, one tier-3 run per step is the default that quietly turns a
  day of work into three: eleven steps is over five hours of gating, most of it re-proving the same
  untouched subsystems. **Batch tier 3 to the wave boundary, not the step boundary** — steps whose
  files do not collide are developed in parallel, each gated by tier 1 and tier 2, and one tier-3 run
  licenses the whole wave's commits. This is a scheduling decision made when the arc is sequenced,
  and the sequencing plan states explicitly where the tier-3 runs fall.
- **Parallelism comes from file disjointness, not from dependency order.** Before fanning out, map
  which files each step touches and cluster the collisions; steps in one cluster serialize, clusters
  run concurrently. A lettered dependency list routinely hides five independent roots.
- **This economy is every agent's concern, not only the orchestrator's.** A subagent that finds
  itself about to run the full suite to validate a two-file edit runs the neighbourhood instead and
  says what it ran. Spending an agent — or a suite run — must be worth more than what it costs.
- **A slow test file is a defect to investigate, not merely a label.** When a file becomes the
  bottleneck of a tier, it is diagnosed rather than annotated: a single test costing 77+ s is usually
  reporting something real about the code under test, not about the test.
- **The `slow` marker** is declared in `pyproject.toml` and currently used exactly once across 3552
  tests. It remains available for genuinely irreducible long-running tests, and `-m "not slow"` may
  then trim a tier-2 run — but only *after* the underlying cost has been investigated per the rule
  above.

### 5. Process Hygiene

**Path and image-name filtering cannot distinguish these processes.** The operator's instrument-control
GUI, the VS Code extension hosts, and pytest all run the same interpreter out of the same virtualenv:

```
...softae-next\.venv\Scripts\python.exe -m pytest tests/test_...   <- a test run
...softae-next\.venv\Scripts\python.exe "C:\Users\Osuji\Ap...      <- the operator's GUI
...softae-next\.venv\Scripts\python.exe c:\Users\Osuji\.vsco...    <- VS Code extension host
```

Only the **command line** separates them. A rule phrased as "kill Python from the project venv" is
therefore still lethal, which is why the weaker formulations of this section failed twice: two
implementation subagents each ran `taskkill /F /IM python.exe /T` to clear a wedged pytest and took
down the operator's live GUI, the extension hosts, and unrelated test runs with it. A force-killed
GUI runs no `closeEvent`, so no `_safe_park_on_exit` — heater still at setpoint, lamp still on,
dispenser head wherever it was. That is a hardware-safety event, not an inconvenience.

| Rule | Practice |
|---|---|
| Never kill by image name | `taskkill /F /IM python.exe`, `pkill python`, `killall python`, `Get-Process python \| Stop-Process` and equivalents are **forbidden without exception** |
| Kill only PIDs you started | Terminate by PID, and only a process you launched. Unclear provenance → stop and report to the orchestrator |
| Verify before terminating | Read the command line first; a test process carries `-m pytest` |
| Wedged ≠ slow | Decide by CPU measurement, not by wall clock |
| Capture before you kill | Stack dump first — a killed process is not diagnosable |
| Bound and reap | Timeout the run, then confirm *your* child is gone, by PID |

- **Verify before terminating.**
  `Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select ProcessId, CommandLine`.
  If the command line does not clearly identify the process as your own test run, do not kill it.
- **Distinguish wedged from slow, by measurement.** Sample CPU twice a few seconds apart: a frozen
  counter is a hang, a rising one is work. Wall-clock elapsed alone proves nothing — a GUI test
  suite is legitimately slow. This is the check that actually diagnosed the incident:
  ```powershell
  $p = Get-Process -Id <pid>; $before = $p.CPU
  Start-Sleep -Seconds 6
  $after = (Get-Process -Id <pid>).CPU
  # delta ~0 over 6 s => wedged, not slow
  ```
- **Capture before you kill.** A wedged process is diagnosable and a killed one is not. Take a stack
  dump first — `py-spy dump --pid <id>` if present, otherwise `faulthandler` — because a traceback
  names the blocking call directly, which is the thing actually needed. Two rounds of
  file-combination bisecting produced less information than one faulthandler trace.
- **Bound test runs, and reap by PID.** Tiered runs (§4) are the processes this most often applies
  to: give the run a timeout, then confirm afterwards that the specific child you started is gone —
  by PID, never by name. A shell timeout on Windows frequently kills the wrapper and leaves the
  Python child orphaned, which is the state that tempts an image-name kill.
- **Assume the operator's GUI is live.** It usually is, it may be mid-experiment, and it owns the
  instruments. Editing source is safe — a running process already imported it. Killing processes and
  opening instrument sessions are not.

---