"""How much two nominally identical samples disagree on this rig.

The sibling of :data:`~softae.analysis.equilibration.DEFAULT_SETTLE_TOL_REL`, which is the
**temporal** noise floor — one well watched over time. This module holds the **spatial**
one: two wells cast from the same stock, in the same session, on the same board, measured
minutes apart. They are different quantities and neither bounds the other.

**What this number is for.** An optimiser steering a campaign on σ needs to know what
difference between two formulations is real. Below the replicate floor, it is looking at
casting variation and calling it signal.

.. warning::
   **This is a zero-day prior, not a specification, and it will never become one.** It is
   one board, one formulation, one session. The operator's ruling (2026-09-03) is that the
   diversity of samples this system will run makes a single enshrined value wrong by
   construction — a polymer electrolyte, a lyotropic surfactant and a silica dispersion have
   no reason to share a casting reproducibility. It ships so that an optimiser has somewhere
   to start rather than nowhere, and **any per-material measurement supersedes it without
   argument.**

**Why it is not a config key.** A config key is something an operator decides; this is
something the rig does. Restating it in TOML would separate the number from the provenance
that qualifies it — which run, which estimator, and which level was excluded and why — and
that provenance is most of its value. The pattern here follows
:data:`~softae.analysis.equilibration.DEFAULT_SETTLE_TOL_REL`: the measured constant lives
with its reasoning and consumers import it rather than restating it.

**The rule is more durable than the number**, and outlives any revision of it:

    No threshold that compares one sample against another may be tighter than the
    replicate reproducibility of the rig that produced them.

A threshold below the floor does not discriminate — it partitions noise, reproducibly and
meaninglessly. The rule applies only to **across-sample** comparisons: a criterion watching
a single well against its own past (settling, drift) is a different question and is not in
scope, which is why the settle tolerance can legitimately be 10 % while this figure is a
factor of two.
"""

from __future__ import annotations

import math

__all__ = [
    "REPLICATE_SIGMA_SPREAD",
    "RANGE_TO_SD_N3",
    "replicate_log_sd",
    "resolvable",
]

#: Ratio of largest to smallest σ across replicate wells of one formulation at one
#: nominal thickness. **Measured 1.83× (median over the 110/165/221 µm levels, three
#: replicates each) on the 2026-08-17 thickness board, measured 2026-08-18.**
#:
#: **Shipped rounded to 2.0 deliberately.** The measurement supports "about a factor of
#: two"; writing 1.83 would claim three significant figures from three wells on one board.
#: Rounding up is also the conservative direction for a floor.
#:
#: The 55 µm level is **excluded and is not noise**: it spread 142.72×, and two of its
#: three wells returned negative Re Z. That is failed casting — 21 µL into a well whose
#: void is 118.77 µL — and it is excluded on casting evidence, not because it was
#: inconvenient. Estimator is the single-point resistive-plateau read, so no circuit-fit
#: error is folded in. Full derivation: MAIL ``[a163]`` §1.
REPLICATE_SIGMA_SPREAD = 2.0

#: Expected range of three standard normal samples, in units of their standard deviation
#: (``E[range] ≈ 1.693 σ`` for ``n = 3``). Used only by :func:`replicate_log_sd`, and named
#: rather than inlined because it is the assumption that conversion rests on.
RANGE_TO_SD_N3 = 1.693


def replicate_log_sd(spread: float = REPLICATE_SIGMA_SPREAD,
                     n: int = 3) -> float:
    """A log-space standard deviation for an optimiser's noise term, from a *range*.

    :data:`REPLICATE_SIGMA_SPREAD` is a max/min **ratio over three wells**, which is not a
    standard deviation and must not be handed to a Gaussian likelihood as one. This does the
    conversion and **states the assumption rather than burying it**: for ``n`` normal samples
    the expected range is a known multiple of σ, and at ``n = 3`` that multiple is
    :data:`RANGE_TO_SD_N3`.

    Returns σ of ``ln σ_measured`` — appropriate because conductivity is positive and varies
    multiplicatively, so the natural error model is log-normal. At the shipped 2.0× this is
    ≈ 0.41, i.e. **roughly 40 % relative**.

    .. warning::
       Three samples estimate a range badly. The conversion is assumption-laden in two
       places — normality, and that the range of three is representative — and it inherits
       every caveat on the constant itself. Treat the output as an order of magnitude for a
       prior, never as a calibrated uncertainty.
    """
    if not (spread == spread) or spread <= 0.0:
        return float("nan")
    if n != 3:
        raise ValueError(
            f"only n=3 is calibrated here (RANGE_TO_SD_N3); got n={n}. Add the "
            f"tabulated range-to-sd factor for that n rather than reusing this one."
        )
    return math.log(spread) / RANGE_TO_SD_N3


def resolvable(a: float, b: float,
               spread: float = REPLICATE_SIGMA_SPREAD) -> bool:
    """Is the difference between two σ values larger than replicate variation?

    The rule in the module docstring, made executable so it can be tested rather than
    merely believed. ``False`` means the two samples are **indistinguishable on this rig** —
    not that they are equal, which is a different and unmeasured claim.

    Compared as a ratio rather than a difference because conductivity spans decades and
    varies multiplicatively; an absolute tolerance would be far too strict at the low end and
    meaningless at the high end.

    **Nothing consumes this yet.** It is offered for the optimiser's noise model and for
    reviewing existing across-sample thresholds against the floor; it is recorded here as
    unwired rather than presented as an integrated check.
    """
    if not (a == a and b == b) or a <= 0.0 or b <= 0.0:
        return False
    ratio = max(a, b) / min(a, b)
    return bool(ratio > spread)
