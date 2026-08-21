# Standing rules for spawned agents

Required reading for every subagent, alongside `CLAUDE.md`. A brief cites this file
instead of restating it.

`CLAUDE.md` says what the project's workflow *is*. This file says what has actually
gone wrong, repeatedly, and what to do instead. Every rule below was bought.

---

## 1. Safety — these have no exceptions

**Never kill a process by image name.** `taskkill /IM`, `pkill`, `killall`,
`Get-Process <name> | Stop-Process` and every equivalent are forbidden. On this machine
the operator's GUI, the VS Code extension hosts and pytest all run the same interpreter
from the same virtualenv, so no name- or path-based filter can separate them. Two agents
have already taken down the operator's live GUI this way. Terminate only a PID you
launched yourself, and confirm it by command line first.

**Assume the operator's GUI is live and holds the rig.** It usually is, it may be
mid-experiment, and it owns the instruments. Editing source is safe — a running process
already imported it. Killing processes and opening instrument sessions are not.

**Never actuate hardware.** No pump dispense, stage motion, piezo firing, potentiostat
measurement, or temperature/humidity setpoint change. Dry-run and simulated paths are
fine. This rig has no undo: a stage move into an occupied well or a dry pump fired at
speed destroys a board or the hardware.

**Wedged is decided by measurement, not by clock.** Sample a process's CPU twice about
six seconds apart. A frozen counter is a hang; a rising one is work. Capture a stack dump
before terminating anything — a killed process is not diagnosable.

---

## 2. Verify the brief before you build on it

**The brief's structural claims are not evidence.** Every step in the recent arcs found
at least one wrong premise in its own brief, several of which would have defeated the
step entirely. Treat "X has one caller", "Y is fed by Z", "the flow is A then B" as
*claims to check*, and check them before writing code.

**If the premise is wrong, stop and report. Do not implement it anyway**, and do not
silently repair it either — the orchestrator needs to know, because the same premise is
usually in a spec and in two other briefs.

**Locate by symbol, never by line number.** Anchors in specs and briefs drift constantly;
one module moved 278 lines in a single step. Grep for the symbol.

---

## 3. Two failure shapes that look like success

These are the ones that cost the most, because nothing goes red.

**A test that cannot fail.** A test injected a manager whose `connect_all` *raises* — but
the production manager catches everything and returns a dict, so it can never raise. The
test passed forever and the guarantee it named was never checked. **Before trusting a
negative test, ask whether production can actually produce the shape you injected.**

**A fallback that hides a no-op.** A helper was specced to union instruments over
`workflow.steps`. That attribute does not exist, so the union would have been empty, the
result would have fallen back to "whole rig" — the *safe* direction — and no test would
have failed while the feature did nothing. **When a helper has a safe fallback, prove the
non-fallback path is reachable.**

The general form: when the wrong answer is also the conservative answer, only an explicit
test distinguishes "working" from "not running at all".

---

## 4. Ownership — three sessions share one working tree

**`git add` on a shared file stages the other session's work into our commit.** Check
`git status` before touching anything, and treat any file you did not modify as theirs.

**Never `git add -A` or `git commit -a`.** Never revert or unstage another session's work.

**Do not run `ruff --fix` as a sweep.** Lint files you already edited. A gratuitous
import reordering in a shared 4,000-line file is pure conflict risk for another session
and buys nothing.

**Do not commit.** The orchestrator commits, by explicit file list, after the gate.

---

## 5. Test spend

**Tier 1 only: the touched modules' own tests plus direct neighbours**, found by grep
rather than guessed by name.

**Never run the full suite.** It costs ~26 minutes and the orchestrator batches one run
at the wave boundary. A subagent about to run everything to validate a two-file edit runs
the neighbourhood instead.

**Report exactly what you ran and what it returned** — the selection, the counts, the
duration. "Tests pass" is not a report.

Bound every run with a timeout, and confirm your own child is gone **by PID** afterwards.
Note `pytest-timeout` is not installed; use the shell's timeout.

---

## 6. What a good report contains

- What changed, and why that shape rather than the obvious one.
- **Anything the brief or spec got wrong** — this is the most valuable part of the report,
  not a discourtesy.
- What you ran, and its result.
- Decisions you made that the brief left open, with the reasoning.
- What you deliberately did **not** do, and why.
- Anything you noticed that is out of scope but real. Say it; do not fix it silently.

---

## 7. Scope

**Do not silently widen or silently narrow.** If the task's edges are unclear, do the
part that is clearly in scope, and report the rest as a finding.

**Prefer a new file to a contested edit.** A new module that imports an in-flight file
costs that file zero edits and is separately reviewable.

**Say when a change is operator-visible.** A moved button, a changed refusal, a lost
capability, a new prompt — these are contract changes and the operator needs to hear
about them in words, not discover them at the bench.
