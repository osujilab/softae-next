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
1. **Runs tests** in the terminal to confirm zero regressions
1. **Spawns update subagents** → updates PROGRESS.md, USER_GUIDE.md, ACTION_PLAN.md and other ".\docs" files as needed
1. **Recurrent workflow reminders** → occasionally reminds the system to retain this workflow structure (roughly every three prompts).

The orchestrator **never** edits source files directly — edits stay delegated. Reading directly is
required, not merely permitted, because the comparison in step 2 depends on the orchestrator having
formed its own independent view.

#### Subagent capabilities

Every spawned subagent is granted **full command-line access** (Bash and PowerShell), not just
file-read and edit tools. Subagents run tests, invoke git, execute scripts, and inspect the
environment themselves rather than reporting findings back for the orchestrator to act on. A
subagent that needs to run a command runs it.

##### Exception — commands that return to the orchestrator

Three classes sit **outside** the grant. A subagent that needs one stops, reports the exact command
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

The orchestrator may authorise class 2 on its own judgement. **Classes 1 and 3 go to the user** —
the orchestrator does not self-authorise a destructive write or any motion of real equipment.

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

---