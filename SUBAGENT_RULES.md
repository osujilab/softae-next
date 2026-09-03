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

## 3. Failure shapes that look like success

These cost the most, because nothing goes red. **Three shapes, and the first has five faces.**
There is deliberately no count in the heading: a new fault is an invitation to hold it against
the boundaries below and to widen one of them if it does not fit — not to append another bullet.

### 3.1 — The self-measuring instrument

**Anything built to answer a question can quietly answer a *different* question and return the
shape of a pass.** A test, a gate, a harness, a lookup, a fingerprint. When the wrong answer is
also the conservative answer, only an explicit test distinguishes *working* from *not running at
all*. **The rule applies to your instruments, not only to your code: check that the check can
fail.** Five faces, each found separately here.

**(a) It could not check, and said fine.** The most common and the hardest to see, because the
absence is asserted rather than hidden. `gate_log_json` is `TEXT NOT NULL DEFAULT '[]'`, so it is
structurally incapable of reporting its own absence, and a null-check reports 100 % coverage over
a 99.4 %-empty column. A gate whose covariance is `None` returns `passed=True` with the detail *"no covariance
available"* — indistinguishable, to every consumer, from a gate that checked and found nothing. An
envelope field named `phase_noise_measured` **defaults to `True`**, so the guard that exists to
force a provisional result when the floor was never measured can never fire. **"Unknown" must not
be spelled with the same token as "checked and clean."**

**(b) It checked, and the answer was discarded.** Worse than (a), because the work was done. One
report path computes whether the phase floor is even applicable at the sample's impedance —
`False` for every real film on this rig, the anchor being three decades away — and spends that
result on a log label while the decision two lines below gates on something else.

**(c) It ran downstream of the thing it governs.** A tail-read check placed in the *same* shell
invocation as the append it was meant to gate: the output scrolls past after the bytes are on
disk. A `||` branch that fired on a *previous* command's exit status. **A check that runs
concurrently with what it governs is a log line, not a check** — and the version that happens to
report correctly is the harder one to find.

**(d) It checked the wrong surface.** A mutation audit mutated the *caller*, watched it die, and
pronounced the *callee* tested. A suite imported the real engine and asserted only that its output
was well-formed, so an all-NaN engine produced a well-formed frame and 34 green tests. **A
mutation audit certifies exactly the surface it mutates.**

**(e) It could not have failed.** A test injected a manager whose `connect_all` *raises* — but
the production manager catches everything and returns a dict, so it can never raise. The
test passed forever and the guarantee it named was never checked. **Before trusting a
negative test, ask whether production can actually produce the shape you injected.**

And the same face again, in a fallback: a helper was specced to union instruments over
`workflow.steps`. That attribute does not exist, so the union would have been empty, the
result would have fallen back to "whole rig" — the *safe* direction — and no test would
have failed while the feature did nothing. **When a helper has a safe fallback, prove the
non-fallback path is reachable.**

And once more, in the harness itself — the case worth the most detail, because it invalidates
every result taken through it. §5 forbids mutating the shared tree, so the standing method
is to mutate a *scratch copy*. On this repo that has a silent no-op:
`.venv/Lib/site-packages/_editable_impl_softae.pth` appends the real `src/` to `sys.path`,
so a copy anywhere else is **never imported**. The mutation lands on a file nothing loads,
the test runs against unmutated code, and the check comes back **green** — a harness that
cannot fail, verifying a test that may also not fail. This was hit for real: ten mutations
in a row returned green before the agent thought to doubt the harness rather than the tests.

The asymmetry is what makes it survivable, and it is worth knowing precisely:

- **A shadowed mutation cannot produce a red.** So a reported *red* is itself proof the
  mutation reached the code, and past results that reported reds are sound.
- **A reported green proves nothing at all.** *"The mutation did not fire, so I strengthened
  the test"* is the reasoning to distrust — the test may have been fine and the harness dead.

**So: every mutation run carries a positive control** — one mutation you are certain must go
red, such as breaking the assertion's own subject. If the control comes back green, the
harness is shadowed, not the test weak. Set `PYTHONPATH` to the scratch copy and confirm with
`print(module.__file__)` before trusting a single result.

**A mutation that silently does not mutate returns green, and that green is indistinguishable
from a vacuous test.** Assert the edit landed — read the file back, or diff it — before scoring
the run. Otherwise face (e) is what you will conclude, and the harness is what was broken.

### 3.2 — A check that is sound, and never reached

The test discriminates, the mutation goes
red, the harness is sound — and the behaviour under test is **never reached by real data.** This
is the one mutation testing cannot see, because mutation testing asks *can the assertion fail*
and this asks *does the population ever visit the branch*. Two faces, found separately before
anyone noticed they were the same thing:

- **A fixture from a code state that never existed.** `gate_pegged_parameters` has tests that
  discriminate perfectly, on a covariance whose bounds the production path never constructs. The
  gate cannot fire on any resistance or capacitance the rig produces, and its test file scored
  55/55 in a mutation audit.
- **Data from a code state that no longer exists.** 407 bound-pinned rows were reported as a live
  defect and the guard had shipped twelve days before the newest of them. **A defect found in
  historical data is a claim about the code as it was when the data was written.**
A third instance used to sit here — a `NOT NULL DEFAULT` standing in for absence — and it belongs
to **3.1(a)** instead: nothing about it concerns which branches real data visits. **Filing a fault
under the wrong shape is how a boundary silently widens.**

**The check is not another mutation.** It is running the thing against a sample of the real
corpus and confirming the branch is entered at all — a counter and one pass over stored data.
Ask what the rig actually produces, not only whether the assertion can fail.

### 3.3 — A rule that is correct in one regime, applied uniformly

Conservatism has a *direction*,
and the direction depends on which side of a ratio you are standing on. `derive_phase_table`
refuses to use the minimum loss angle and is right: the instrument's floor must not be
understated, so the denominator takes a per-decade **median**. Generalising that into a house rule
puts a median in the **numerator** too — which is the defect shipped at `report.py:232`, where the
sample's own margin should be its **minimum**. To any reader skimming for the rule: **there is no
single rule.** Median for the instrument, minimum for the sample, and the two are the same
sentence pointed in opposite directions.

**Placing a new fault.** Ask, in order: did the instrument answer a different question than the
one asked (**3.1**); did it answer correctly about a branch real data never visits (**3.2**); or
is it a sound rule carried across a boundary it does not hold over (**3.3**)? If a fault fits
none of the three, that is the interesting case — **say so, and propose widening a boundary
rather than appending a fourth shape.** Amendments to this file go to the operator for approval;
the channel is where a rule is found and argued, not where it is authorised.

---

## 4. Ownership — three sessions share one working tree

**`git add` on a shared file stages the other session's work into our commit.** Check
`git status` before touching anything, and treat any file you did not modify as theirs.

**Never `git add -A` or `git commit -a`.** Never revert or unstage another session's work.

**Do not run `ruff --fix` as a sweep.** Lint files you already edited. A gratuitous
import reordering in a shared 4,000-line file is pure conflict risk for another session
and buys nothing.

**Do not commit.** The orchestrator commits, by explicit file list, after the gate.

**The map answers who may WRITE a file. It does not answer whether the file is currently
TRUSTWORTHY.** That gap is the source of every ownership surprise here so far, and it has two
faces. **A claim protects against a conflicting write; nothing in the protocol protects a
reader.**

**(a) A paused edit is indistinguishable from an abandoned one unless the map says which.** A
file held mid-task and a file someone walked away from look identical in `git status`, so the
map has to carry the difference. One session asked the channel four times who owned a dirty file
that was its own paused work, because it ran an ownership lookup — which answers `UNCLAIMED` for
anything unclaimed — instead of running `git diff` on its own tree. **A map's silence is not a
fact about the world.**

**(b) A file under an in-place mutation harness is not merely claimed — it is INVALID, and no
ownership lookup can reveal that.** A mutation run that backs up, restores after every mutation
and verifies the hash at the end protects the *end state*; for the twenty minutes in between, the
file is about as likely to be wrong as right. Another session swept the corpus during exactly
such a window, read a deliberately broken engine, and published its number as a finding about
working code — a 25× shift in a headline statistic, on identical stored data. **Mutate a copy and
point the run at it; if you must mutate in place, announce the window on the channel with its
start and end.** A gate window makes the tree *busy*; a mutation window makes it *lying*, and
only one of those has a convention.

**A killed agent's untracked output survives a revert of tracked files.** Reverting what you
edited does not remove what you created. One halt left three artifacts and one was audited: the
other two were a new test file that reddened every session's suite for two days, and a dirty
module nobody could attribute. **After stopping work mid-flight, audit the whole tree, not the
files you remember touching.**

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

---

## 8. What you write outlives you

**A path written into the DataStore must outlive the session that wrote it.** Every agent here
has a scratchpad directory and it is temporary *by construction*. A row pointing into one is a
dangling reference the moment that session ends, and nothing fails at the time — it fails when
somebody tries to reproduce the result, which is exactly when they most need it.

This is not a hypothetical either. `mux16.toml` is a committed calibration whose `blank_open`
source row pointed at a scratchpad path that no longer existed: **real bench data, correct
numbers, and not reproducible**, discovered only when a spec proposed re-deriving it to answer
whether the phase table was trustworthy. One row in 3745, and it was the one row that mattered.

**So: before writing a path into any persistent store, ask whether that location will exist next
week.** Copy the artifact somewhere durable and record *that* path. The same goes for anything
persisted that names a location — run directories, exports, figures cited in a spec.

**And a stored number is a claim about the code that produced it.** A defect found in historical
data may have been fixed since; a threshold in a config may have been calibrated against a
statistic that has since moved. Check when the data was written against when the code changed,
before reporting either as live.
