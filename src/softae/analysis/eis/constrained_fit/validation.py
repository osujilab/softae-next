"""Hold one out, score it against a known answer, and refuse on the result.

The fourth refusal, and the only one with an answer key behind it: the three in
:func:`~softae.analysis.eis.constrained_fit.fit.fit_shared` are self-consistency
tests a set of standards is not required for, while this one asks whether the number
is *right*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .admissibility import MIN_ARC_RESOLVED
from .fit import CONSTRAINED_FIT_TOL, SharedArtifact, fit_frozen, fit_shared
from .model import ConstrainedSpectrum
from .refusals import InadmissibleFitSet, PoorGeneralisation

#: Refusal threshold on :attr:`HoldoutReport.mean_abs_error_pct`, in percent.
#:
#: :func:`holdout_report` computes the only genuine generalisation number this codebase
#: has — fit N artifacts each missing one spectrum, predict the one that was held out —
#: and until 2026-09-08 **nothing refused on it.** Measured populations, four NIST
#: standards, both preparations:
#:
#: ==========================================  ==============  ==============
#: artifact                                     in-sample mean  leave-one-out
#: ==========================================  ==============  ==============
#: shipped grid, gate at ``MIN_ARC_RESOLVED``   3.54 / 3.73 %   **3.04 / 2.44 %**
#: (worst single fold)                          4.89 / 5.59 %   6.37 / 4.36 %
#: **grid restricted to the poisoned basin**    **7669 / 5870%** **39.97 / 40.96 %**
#: (worst single fold)                          18 123 %        82.43 / 84.06 %
#: ==========================================  ==============  ==============
#:
#: (thin / full-conditioned, the same two preparations throughout this module.)
#:
#: **The leave-one-out gap is 13×, not the four orders of magnitude the in-sample
#: column suggests, and that difference is itself the finding.** Two of the four
#: poisoned folds escape the basin — their ``Qg`` comes back at 8.3e-10 — so the mean
#: they contribute to is dragged back toward the good population. A gap that narrow
#: cannot be split by "put the constant in the middle" the way
#: :data:`ORDER_SPREAD_REFUSE_PCT`'s four-order gap could.
#:
#: **So the number is anchored on utility instead, and that is the honest basis:** a
#: 25 % error on ``R_sol`` propagates to a 25 % error on σ, which is already useless for
#: the campaign this feeds, so there is no reason to buy margin above it. It sits ~8×
#: above the worst good mean, above the worst single good fold (6.37 %), and below the
#: only bad population measured (39.97 %) — but by 1.6×, which is thin, and a corpus
#: whose honest generalisation is worse than 25 % would be refused. That is the
#: intended reading: this gate says *"not good enough to use"*, not *"detectably
#: poisoned"*.
#:
#: **The gate is on the mean, and** :attr:`HoldoutReport.worst_abs_error_pct` **is
#: reported but not gated.** With four standards a per-fold extreme is one number from
#: one fit; thresholding it would be calibrating on n = 1. That is the part to revisit
#: when the corpus grows, and it is a real hole: a single catastrophic fold among many
#: good ones is diluted by the mean — exactly what the 39.97 % row above is.
HOLDOUT_REFUSE_PCT = 25.0


# ── Validation: hold one out, report the error ───────────────────────────────

@dataclass(frozen=True)
class HoldoutResult:
    """One known answer, what was measured for it, and the deviation.

    The shape ``derive_reference_r``'s ``error_pct`` and ``blank_load``'s
    ``load_error_pct`` are both already instances of, named once. Run something whose
    answer is known, report the deviation. (Those two are not migrated onto this class
    here — they live in files this task does not own.)
    """

    label: str
    reference: float
    measured: float

    @property
    def error_pct(self) -> float:
        if not (self.reference == self.reference and self.reference):
            return float("nan")
        return 100.0 * (self.measured - self.reference) / self.reference


@dataclass(frozen=True)
class HoldoutReport:
    """``kind`` is load-bearing: an in-sample error is not a generalisation claim."""

    kind: str
    results: tuple[HoldoutResult, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    artifacts: tuple[SharedArtifact, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> list[float]:
        return [abs(r.error_pct) for r in self.results if r.error_pct == r.error_pct]

    @property
    def mean_abs_error_pct(self) -> float:
        e = self.errors
        return float(np.mean(e)) if e else float("nan")

    @property
    def worst_abs_error_pct(self) -> float:
        e = self.errors
        return float(max(e)) if e else float("nan")

    def describe(self) -> str:
        refused = f", {len(self.refused)} fold(s) refused" if self.refused else ""
        return (f"{self.kind}: mean |err| {self.mean_abs_error_pct:.2f}%, worst "
                f"{self.worst_abs_error_pct:.2f}% over {len(self.results)} case(s)"
                f"{refused}")



def refuse_unless_generalises(
    report: HoldoutReport, *, max_mean_abs_error_pct: float = HOLDOUT_REFUSE_PCT
) -> HoldoutReport:
    """Raise :class:`PoorGeneralisation` unless *report*'s mean error is under the limit.

    Returns *report* unchanged when it passes, so it composes as
    ``refuse_unless_generalises(holdout_report(specs))`` and the caller keeps the
    report. :func:`holdout_report` applies it by default; this is the seam for a caller
    holding a report it obtained some other way.

    **NaN refuses.** An empty ``results`` — every fold refused, or no spectrum carrying
    a reference — makes the mean NaN, and NaN is *"could not judge"*, which must not be
    spelled the same way as *"checked and clean"* (``SUBAGENT_RULES`` §3.1(a)). The
    comparison is written on the passing side for that reason, as in :func:`fit_shared`.

    An ``in_sample`` report is refused on the same threshold and the message says so.
    Passing it proves strictly less than passing a leave-one-out one — that is what
    :attr:`HoldoutReport.kind` is load-bearing *for* — but an artifact that cannot
    reproduce the standards it was fitted on has failed something real, so the number is
    worth refusing on rather than ignoring.
    """
    if not math.isfinite(max_mean_abs_error_pct):
        return report
    mean = report.mean_abs_error_pct
    if mean != mean:
        raise PoorGeneralisation(
            f"no holdout case could be scored, so generalisation is unjudged rather "
            f"than acceptable ({report.describe()})")
    if not mean <= max_mean_abs_error_pct:
        claim = ("generalisation" if report.kind == "leave_one_out"
                 else "in-sample agreement, which is a weaker claim still")
        raise PoorGeneralisation(
            f"mean |error| {mean:.4g}% against the known references exceeds "
            f"{max_mean_abs_error_pct:g}% — the artifact fails on {claim} "
            f"({report.describe()})")
    return report


def holdout_report(
    spectra: list[ConstrainedSpectrum],
    *,
    kind: str = "leave_one_out",
    tol: float = CONSTRAINED_FIT_TOL,
    min_arc_resolved: int = MIN_ARC_RESOLVED,
    max_mean_abs_error_pct: float = HOLDOUT_REFUSE_PCT,
    **fit_kwargs: Any,
) -> HoldoutReport:
    """Fit, score against :attr:`ConstrainedSpectrum.reference_ohm`, **and refuse**.

    ``kind="in_sample"`` fits one artifact on everything and scores everything against
    it. ``kind="leave_one_out"`` fits N artifacts, each missing one spectrum, and scores
    only the held-out one — the number that is actually a generalisation claim.

    **A refused fold is recorded, not skipped.** A report that silently averaged over
    "the folds that worked" would hide the module's own headline limitation, so a fold
    refused by any :class:`InadmissibleFitSet` — including
    :class:`UnstableFitSurface` and :class:`ImplausibleArtifact`, which is why both
    subclass it — is recorded with its reason.

    **The report is also a gate**, via :func:`refuse_unless_generalises` and
    :data:`HOLDOUT_REFUSE_PCT`; ``max_mean_abs_error_pct=math.inf`` is the one explicit
    way to ask for the report regardless. It was purely a report until 2026-09-08:
    it computed the only genuine generalisation number here and nothing acted on it,
    which is ``SUBAGENT_RULES`` §3.1(b) — the work was done and the answer discarded.

    *fit_kwargs* reach :func:`fit_shared` unchanged — ``seed_grid``,
    ``max_order_spread_pct``, ``max_qg``, ``max_nfev``.
    """
    if kind == "in_sample":
        artifact = fit_shared(spectra, tol=tol, min_arc_resolved=min_arc_resolved,
                              **fit_kwargs)
        results = tuple(
            HoldoutResult(s.label, s.reference_ohm,
                          fit_frozen(s, artifact, tol=tol).R_sol)
            for s in spectra if s.reference_ohm == s.reference_ohm)
        return refuse_unless_generalises(
            HoldoutReport("in_sample", results, (), (artifact,)),
            max_mean_abs_error_pct=max_mean_abs_error_pct)

    if kind != "leave_one_out":
        raise ValueError(f"unknown holdout kind {kind!r}")

    results: list[HoldoutResult] = []
    refused: list[tuple[str, str]] = []
    artifacts: list[SharedArtifact] = []
    for held in spectra:
        rest = [s for s in spectra if s is not held]
        try:
            artifact = fit_shared(rest, tol=tol, min_arc_resolved=min_arc_resolved,
                                  **fit_kwargs)
        except InadmissibleFitSet as exc:
            refused.append((held.label, str(exc)))
            continue
        artifacts.append(artifact)
        if held.reference_ohm == held.reference_ohm:
            results.append(HoldoutResult(held.label, held.reference_ohm,
                                         fit_frozen(held, artifact, tol=tol).R_sol))
    return refuse_unless_generalises(
        HoldoutReport("leave_one_out", tuple(results), tuple(refused),
                      tuple(artifacts)),
        max_mean_abs_error_pct=max_mean_abs_error_pct)
