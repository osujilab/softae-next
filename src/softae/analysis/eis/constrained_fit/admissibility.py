"""Two questions asked *about* a fit rather than by it: can this set identify the
shared three, and is the artifact it produced physical?

:func:`check_admissible` runs before any fitting and is model-free on purpose.
:func:`implausible_shared` runs after one and looks only at what the artifact says,
which is why it can also be applied to a stored artifact without refitting. Both
carry their own threshold, with the populations it was derived from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .model import (
    PER_SPECTRUM_PARAMS,
    SHARED_PARAMS,
    ConstrainedSpectrum,
    arc_is_resolved,
)

if TYPE_CHECKING:  # pragma: no cover - annotation only; fit imports this module
    from .fit import SharedArtifact

#: How many spectra in the fitting set must show a **resolved** geometric arc before
#: the shared parameters are identifiable. See :func:`check_admissible`, which carries
#: the measurement this number was re-derived from — it was ``2`` until 2026-09-08 and
#: the ``2`` was calibrated against seeding that no longer exists.
MIN_ARC_RESOLVED = 1

#: Ceiling on a plausible fitted ``Qg``, in F·s^(n−1). **A refusal on the artifact's
#: own value**, and the only one here that does not ask the fit to grade itself.
#:
#: Two measured populations, both on the four NIST KCl standards, both on the current
#: module. "Returned" means what :func:`fit_shared` hands back — the winner on cost —
#: over every one of the eleven subsets of the four standards with ≥2 members, on two
#: independent preparations (thin, and the offline harness's full conditioning chain):
#:
#: ====================================  ====================  ==================
#: population                             ``Qg``                ``R_sol`` MAE
#: ====================================  ====================  ==================
#: **returned** by the 10 subsets ×       4.76e-10 .. 1.05e-09  3.00 .. 4.39 %
#: 2 preparations that resolve an arc
#: losing-but-physical grid starts        1.14e-10 .. 3.13e-09  3.1 .. 59 %
#: (never returned; recorded for margin)
#: **returned by the subset that          **1.14e-07 ..         **2338 / 2401 %**
#: resolves NO arc** (1413 + 4500)        1.32e-07**
#: **returned with the grid restricted    **8.42e-08 ..         **5944 .. 8251 %**
#: to the poisoned basin**, 4 routes      1.47e-07**
#: ====================================  ====================  ==================
#:
#: Widest returned good value 1.05e-09 against lowest poisoned 8.42e-08 is **80×, with
#: nothing in between**, and ``1e-8`` is very nearly the geometric centre of that gap:
#: 9.5× above the widest good artifact ever returned, 8.4× below the lowest poisoned
#: one. Even counting the losing starts — descents that never win and so are never
#: handed back — the widest physical-basin ``Qg`` is 3.13e-09, still 3.2× below the
#: ceiling. The gap is the evidence; ``1e-8`` is a readable number inside it.
#:
#: **Two warnings that belong on this constant rather than in a spec.**
#:
#: 1. ``Qg`` is a *geometric* CPE and this ceiling is therefore **corpus-specific in a
#:    way** :data:`ORDER_SPREAD_REFUSE_PCT` **is not**: it is a dimensional quantity
#:    scaling with electrode area and gap, measured on AMP_v1's KCl cell. The separation
#:    here is 80× between the two populations; a fixture whose geometry differs by more
#:    than about a decade from that cell can move a *good* ``Qg`` into this band. It is a
#:    keyword on :func:`fit_shared` for exactly that reason, and re-deriving it is part
#:    of commissioning a new fixture, not an optional refinement.
#: 2. **There is deliberately no band on** ``nd``, and the reason is a measurement that
#:    contradicts ``eis_estimator_pipeline.md`` §2.2(b). That spec records good ``nd``
#:    0.75–0.77 against poisoned 0.34–0.52, "non-overlapping". Re-measured here: the
#:    poisoned *basin* does sit at ``nd`` 0.33–0.46, but the poisoned artifact produced
#:    by the arc-blind subset sits at ``nd`` **0.7622 / 0.7660 — inside the spec's good
#:    window** — while legitimate consensus starts reach 0.8499. So an ``nd`` band would
#:    miss a real poisoning and false-fire on real good fits, and every case it does
#:    catch is already caught by ``Qg``. ``SUBAGENT_RULES`` §3.1: a second condition that
#:    only ever fires when the first does is not a second opinion.
QG_PLAUSIBLE_MAX = 1e-8


@dataclass(frozen=True)
class Admissibility:
    """Whether a set of spectra can identify ``Qg, ng, nd`` — and if not, why not."""

    admissible: bool
    reason: str
    n_spectra: int = 0
    n_arc_resolved: int = 0
    arc_resolved: tuple[str, ...] = ()
    n_free: int = 0
    n_residuals: int = 0

    def describe(self) -> str:
        return (f"{'admissible' if self.admissible else 'INADMISSIBLE'}: {self.reason} "
                f"({self.n_spectra} spectra, {self.n_arc_resolved} arc-resolved, "
                f"{self.n_free} free parameters against {self.n_residuals} residuals)")


def check_admissible(
    spectra: list[ConstrainedSpectrum], *, min_arc_resolved: int = MIN_ARC_RESOLVED
) -> Admissibility:
    """Can this set identify the shared three? **Kept, and recalibrated from 2 to 1.**

    The reasoning was always this: below the geometric arc's crossover the sweep sees
    only the blocking electrode, so nothing constrains ``Qg`` or ``ng``, and a set with
    no arc-resolved member is identifying them from nothing. **The reasoning survived
    re-measurement and the threshold did not.** ``MIN_ARC_RESOLVED`` was ``2``, and the
    ``2`` came from `[a178]` §3 — dropping an arc-resolved standard broke the fold at
    −84.06 % / −74.38 % against the *continuation* seeding :data:`SHARED_SEED_GRID`
    replaced. Re-measured on the multi-start those same two folds read **−6.44 % /
    −2.48 %** with their ``Qg`` in the good band: the gate was refusing two of four
    leave-one-out folds for no measured benefit.

    So the question was re-asked from scratch rather than the old answer defended. All
    eleven subsets of the four NIST standards with ≥2 members, this gate bypassed, each
    artifact scored against AMP_v1's ``R_sol`` on **all four** standards — in-sample for
    members, genuine holdout for non-members — on two independent preparations:

    ==================  ===  =============================  ======================
    arc-resolved in set   n   ``R_sol`` MAE over all four    ``Qg``
    ==================  ===  =============================  ======================
    2                     4   3.00 .. 4.28 %                 4.79e-10 .. 7.13e-10
    **1**                 6   **3.10 .. 4.12 %**             5.52e-10 .. 7.43e-10
    **0**                 1   **2337.9 %**                   **1.324e-07**
    ==================  ===  =============================  ======================

    (thin preparation; the full conditioning chain gives 3.22–4.11 / 3.06–4.39 / **2401.1
    %** and ``Qg`` **1.14e-07** for the same three groups.)

    **One arc-resolved spectrum is indistinguishable from two — 3.10–4.12 % against
    3.00–4.28 %, interleaved — and zero is 560× worse on both preparations.** The
    threshold belongs between 0 and 1, so it is 1. The single narc = 0 case is
    ``kcl_1413uS + kcl_4500uS``, the only subset of this corpus with no arc-resolved
    member, and its artifact lands at ``Qg`` 1.3e-7 — inside the poisoned band of
    :data:`QG_PLAUSIBLE_MAX`, which is the mechanism the reasoning above predicts.

    **The honest limit on this recalibration: n = 1 on the refusing side.** Four
    standards admit exactly one arc-blind subset, so "0 is catastrophic" rests on a
    single measurement (repeated on two preparations, which is not the same as two
    populations). The passing side is n = 10 and interleaved, which is the half that
    matters for *not* refusing correct work — a gate that cries wolf is the one that
    gets overridden. Widening the corpus would strengthen the refusing side; nothing
    here waits on it.
    """
    n = len(spectra)
    resolved = tuple(s.label for s in spectra if arc_is_resolved(s.freq_hz, s.z))
    n_free = len(SHARED_PARAMS) + n * len(PER_SPECTRUM_PARAMS)
    n_resid = 2 * sum(int(np.asarray(s.freq_hz).size) for s in spectra)
    base = dict(n_spectra=n, n_arc_resolved=len(resolved), arc_resolved=resolved,
                n_free=n_free, n_residuals=n_resid)

    if n < 2:
        return Admissibility(False, "a shared parameter needs at least two spectra to "
                                    "be shared across", **base)
    if len(resolved) < min_arc_resolved:
        return Admissibility(
            False,
            f"only {len(resolved)} of {n} spectra resolve the geometric arc in band "
            f"(need {min_arc_resolved}); Qg and ng have nothing to be identified from "
            f"and the artifact will be confidently wrong",
            **base)
    if n_resid <= n_free:
        return Admissibility(False, f"{n_free} free parameters against {n_resid} "
                                    f"residuals — underdetermined", **base)
    return Admissibility(True, "shared parameters are identifiable from this set", **base)


def implausible_shared(
    artifact: SharedArtifact, *, max_qg: float = QG_PLAUSIBLE_MAX
) -> str:
    """Why *artifact*'s shared parameters are non-physical, or ``""`` if they are not.

    Split out of :func:`fit_shared` so it can be applied to an artifact that was already
    fitted — a stored one, or one obtained with the refusal explicitly disabled — and so
    the band can be tested without running a fit. See :data:`QG_PLAUSIBLE_MAX` for the
    populations behind the ceiling, and for why ``nd`` is deliberately not checked.
    """
    qg = artifact.Qg
    if qg != qg:
        return "Qg is NaN — the fit did not return a geometric element at all"
    if math.isfinite(max_qg) and qg > max_qg:
        return (f"Qg {qg:.4g} is above the plausible ceiling {max_qg:.4g}: the "
                f"geometric element has absorbed something that is not geometry, and "
                f"it is shared, so every spectrum in the set carries it")
    return ""
