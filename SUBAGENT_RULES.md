# Standing rules for spawned agents

Required reading for every subagent, alongside `CLAUDE.md`. A brief cites this file instead of
restating it. `CLAUDE.md` says what the workflow *is*; this file says what a spawned agent **must
do and must never do** inside it. Every rule was bought by an incident, and the incidents live in
the casebook (`docs/SubAgent docs/SUBAGENT_RULES_CASEBOOK.md` — internal, untracked), cited inline
as **(casebook §N)** so a rule stays short while its evidence stays findable.

## 1. Safety — these have no exceptions

**Never kill a process by image name.** One interpreter in one virtualenv serves the operator's GUI,
the editor's hosts and the test runner alike (`CLAUDE.md` §5), so no name- or path-based filter
separates them. Kill only a PID **you** launched, after reading its command line. (casebook §1a)

**Assume the operator's GUI is live and holds the instruments.** Editing source is safe — a running
process already imported it. Killing processes and opening instrument sessions are not.

**Never actuate hardware.** `CLAUDE.md`'s class-3 list is the boundary; dry-run and simulated paths
stay inside the grant. This rig has no undo: one wrong motion costs a board or the hardware.

**Wedged is decided by measurement, not by the clock.** Sample CPU twice a few seconds apart — a
frozen counter is a hang, a rising one is work — and **capture a stack dump before terminating
anything**, because a killed process is not diagnosable. (casebook §1b)

## 2. Verify the brief before you build on it

**The brief's structural claims are not evidence.** "X has one caller", "Y is fed by Z", "the flow
is A then B" are *claims to check*; recent steps each found at least one wrong premise in their own
brief, several of them fatal to the step. (casebook §2a)

**If a premise is wrong, stop and report — do not implement it anyway, and do not silently repair
it.** The same premise is usually in a spec and two other briefs, so the orchestrator needs the
correction more than you need the unblocking.

**Locate by symbol, never by line number.** Anchors drift constantly, and a stale one points
confidently at the *wrong* function rather than at nothing. (casebook §2b, §2c)

## 3. Failure shapes that look like success

These cost the most, because nothing goes red. **Three shapes, and the first has five faces.** There
is deliberately no count in the heading: a new fault is an invitation to hold it against the
boundaries below and widen one if it does not fit — not to append another bullet.

### 3.1 — The self-measuring instrument

**Anything built to answer a question can quietly answer a *different* question and return the shape
of a pass** — a test, a gate, a harness, a lookup, a fingerprint. When the wrong answer is also the
conservative answer, only an explicit test distinguishes *working* from *not running at all*, so
**the rule applies to your instruments and not only to your code: check that the check can fail.**

**(a) It could not check, and said fine.** The commonest and hardest to see, because the absence is
asserted rather than hidden: a column that cannot hold a null, a "was this measured?" flag that
defaults to true, a gate returning *passed* with the detail *"nothing available to check"*.
**"Unknown" must not be spelled with the same token as "checked and clean."** (casebook §3.1a)

**(b) It checked, and the answer was discarded.** Worse than (a), because the work was done: a path
computes the right applicability test, then spends the result on a log label while the decision
below it gates on something else. (casebook §3.1b)

**(c) It ran downstream of the thing it governs.** A check in the same invocation as the write it
gates; a branch firing on a *previous* command's exit status. **A check that runs concurrently with
what it governs is a log line, not a check.** (casebook §3.1c)

**(d) It checked the wrong surface.** A mutation audit that mutates the *caller* and pronounces the
*callee* tested; a suite asserting only that the real engine's output is well-formed, so an all-NaN
engine stays green. **A mutation audit certifies exactly the surface it mutates.** (casebook §3.1d)

**(e) It could not have failed.** A negative test injects a shape production cannot produce — an
exception from a function that catches everything — and passes forever without checking the
guarantee it names. **Ask whether production can produce the shape you injected**, and **where a
helper has a safe fallback, prove the non-fallback path is reachable**: falling back in the *safe*
direction is exactly what no test catches. (casebook §3.1e)

**And once more in the harness — the case worth the most detail, because it invalidates every result
taken through it.** §4 forbids mutating the shared tree, so the method is to mutate a scratch copy;
an editable install can put the real source ahead of that copy on the import path, so the mutation
lands on a file nothing loads and the run comes back **green**. The asymmetry is survivable and
worth knowing exactly: a shadowed mutation **cannot** produce a red, so past reds are sound, while
**a reported green proves nothing at all** — *"the mutation did not fire, so I strengthened the
test"* is the reasoning to distrust, because the test may have been fine and the harness dead.

**So every mutation or harness run carries a positive control** — one change you are certain must go
red, such as breaking the assertion's own subject; a green control means the harness is shadowed,
not the test weak. **Assert the edit landed** too, by reading the file back: a mutation that
silently does not mutate returns the same green as a vacuous test. (casebook §3.1 harness)

### 3.2 — A check that is sound, and never reached

The test discriminates, the mutation goes red, the harness is sound — and **the behaviour under test
is never reached by real data.** Mutation testing cannot see this: it asks *can the assertion fail*,
this asks *does the population ever visit the branch*. Two faces:

- **A fixture from a code state that never existed** — perfectly discriminating tests over an input
  shape the production path never constructs: an immaculate mutation score on a branch the rig
  cannot enter. (casebook §3.2a)
- **Data from a code state that no longer exists** — a defect found in stored data is a claim about
  the code as it was when that data was written. **§8 owns this rule**; go there for it.
  (casebook §3.2b)

**The check is not another mutation** but a pass over a sample of the real corpus, confirming with a
counter that the branch is entered at all. And note that a "could not check, and said fine" default
belongs under **3.1(a)**, not here, because nothing about it concerns which branches real data
visits: **filing a fault under the wrong shape is how a boundary silently widens.** (casebook §3.2c)

### 3.3 — A rule that is correct in one regime, applied uniformly

**Conservatism has a *direction*, and the direction depends on which side of a ratio you stand on.**
An instrument's floor must not be understated, so its side takes a robust central estimator;
generalising that into a house rule puts the same estimator on the *sample's* side, where the
sample's margin should be its worst case. **There is no single rule** — median for the instrument,
minimum for the sample, the same sentence pointed in opposite directions. (casebook §3.3a)

**Each side must also be ROBUST in its own direction, or the statistic measures noise instead of the
quantity.** A single extreme point is a noise statistic, not a conservative one: it says nothing
about the sample and will flip a verdict repeatedly while the physical quantity holds still. A
**windowed** extreme — the most pessimistic *region*, not the luckiest single reading — is the
quantity; direction is conservatism, robustness is what makes the direction mean anything.
(casebook §3.3b)

**A hard cut on a continuous covariate is a cliff, not a fix.** Where the suspect readings shade
continuously into the good ones, that covariate is **provenance to record, never a filter to
apply** — filtering removes some artefact and some signal, and cannot say which. (casebook §3.3c)

**Placing a new fault.** Ask, in order: did the instrument answer a different question than the one
asked (**3.1**); answer correctly about a branch real data never visits (**3.2**); or carry a sound
rule across a boundary it does not hold over (**3.3**)? A fault fitting none of the three is the
interesting case — **say so, and propose widening a boundary rather than appending a fourth shape.**
Amendments go to the operator for approval; the channel is where a rule is found and argued, not
where it is authorised.

## 4. Ownership on a shared tree

Several sessions share one working tree and one index; `CLAUDE.md` §6 defines the declaration map
and its lookup tool.

**`git add` on a shared file stages another session's work into our commit.** Check `git status`
first, treat any file you did not modify as theirs, and **never `git add -A` or `git commit -a`**.
Never revert or unstage another session's work, and **do not commit at all** — the orchestrator
commits, by explicit file list, after the gate.

**Do not run a lint autofixer as a sweep.** Lint the files you already edited; a gratuitous import
reordering in a large shared file is pure conflict risk and buys nothing. (casebook §4d)

**The map answers who may WRITE a file, not whether the file is currently TRUSTWORTHY.** That gap is
the source of every ownership surprise here, and it has two faces: **a claim protects against a
conflicting write; nothing in the protocol protects a reader.**

**(a) A paused edit is indistinguishable from an abandoned one unless the map says which.** Both
look identical in `git status`, and **a map's silence is not a fact about the world** — a lookup
answers *unclaimed*, which is neither *free* nor *mine, paused*. Diff your own tree before asking
the channel. (casebook §4a)

**(b) A file under an in-place mutation harness is not merely claimed — it is INVALID, and no
ownership lookup can reveal that.** Backing up and restoring per mutation protects the *end state*;
in between, the file is about as likely to be wrong as right. **Mutate a copy and point the run at
it; if you must mutate in place, announce the window with its start and end.** A gate window makes
the tree *busy*; a mutation window makes it *lying*, and only one has a convention. (casebook §4b)

**A killed agent's untracked output survives a revert of tracked files.** Reverting what you edited
does not remove what you created, so **after stopping mid-flight audit the whole tree, not the files
you remember touching.** (casebook §4c)

**Check the seam your change creates, not only the halves you wrote.** A field added in one
session's file and read in another's is green in every working tree and broken in a fresh checkout,
so when a symbol crosses a file boundary the halves commit together — say so in your report.
(casebook §4e)

## 5. Test spend

**Tier 1 only: the touched modules' own tests plus direct neighbours**, found by grep rather than
guessed by name. `CLAUDE.md` §4 defines the tiers and who owns each.

**Never run the full suite.** It costs tens of minutes and the orchestrator batches one run at the
wave boundary; validating a two-file edit with the neighbourhood instead is two orders of magnitude
cheaper. (casebook §5a)

**Report exactly what you ran and what it returned** — the selection, the counts, the duration.
"Tests pass" is not a report.

**Bound every run with a timeout, and reap your own child by PID** under §1's rule, never by name.
Where the test runner offers no timeout of its own, use the shell's. (casebook §5b)

## 6. What a good report contains

- What changed, and why that shape rather than the obvious one.
- **Anything the brief or spec got wrong** — the most valuable part of a report, not a discourtesy.
- What you ran, and its result.
- Decisions you made that the brief left open, with the reasoning.
- What you deliberately did **not** do, and why.
- Anything you noticed that is out of scope but real. Say it; do not fix it silently.

## 7. Scope

**Do not silently widen or silently narrow.** If the task's edges are unclear, do the part that is
clearly in scope and report the rest as a finding.

**Prefer a new file to a contested edit.** A new module that imports an in-flight file costs that
file zero edits and is separately reviewable.

**Say when a change is operator-visible.** A moved control, a changed refusal, a lost capability, a
new prompt — these are contract changes, and the operator needs them in words rather than at the
bench.

## 8. What you write outlives you

**A path written into a persistent store must outlive the session that wrote it.** Every agent has a
scratchpad directory that is temporary *by construction*, so a stored row pointing into one is a
dangling reference the moment that session ends — and nothing fails until somebody tries to
reproduce the result, which is when they most need it. (casebook §8a)

**Before writing a path into any persistent store, ask whether that location will exist next week.**
Copy the artifact somewhere durable and record *that* path — likewise for anything persisted that
names a location: run directories, exports, figures cited in a spec.

**And a stored number is a claim about the code that produced it.** This is that rule's single home,
pointed to from §3.2: a defect in historical data may have been fixed since, and a config threshold
may have been calibrated against a statistic that has since moved. **Check when the data was written
against when the code changed, before reporting either as live.** (casebook §8b)
