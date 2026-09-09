"""Constrained global fit: three parameters shared across a set of spectra.

**SHIPS INERT.** Nothing in ``src/`` calls this module, and that is deliberate. Wiring
it into a live path is a separate, separately-reviewable step.

Why it is a module of its own rather than a branch inside
:func:`~softae.analysis.eis.engine.analyze_spectrum`: a global fit is a **batch**
operation. ``analyze_spectrum(one_spectrum) -> SpectrumReport`` is per-spectrum by
construction and cannot express a parameter shared across N spectra. The batch logic
therefore lives here, and inference against a frozen artifact — which *is* per-spectrum
— is :func:`fit_frozen`.

The measured case for constraining at all (``constrained_fit_chain.md`` §1, and the
offline harness at ``docs/SubAgent docs/constrained_fit_harness/``): scoring ``R_sol →
K → σ`` against the four NIST KCl standards, an unconstrained per-spectrum fit of the
same circuit, with the same cleaning and the same optimiser, gives mean |σ error| in
the **thousands of percent**, while sharing three parameters across the set gives
~1.4 % in-sample. **The accuracy is bought by the constraint structure, not by the
circuit.**

The split, and why each line is where it is::

    PINNED    L                 CalibrationSet.L_lead_H -- per-channel, commissioned,
                                hardware_hash-keyed. Never fitted: overhaul F5 recorded
                                fitted inductances of 400-500 uH against a short blank's
                                true 4.18 uH.
    KNOWN     G_fixture(f)      CalibrationSet.G_fixture, consumed POINT BY POINT.
                                Never collapsed to a scalar, and never fitted alongside
                                a free leak -- see `fixture_admittance`.
    SHARED    Qg, ng, nd        fitted once across the set
    PER WELL  R0, R1, Qd        fitted per spectrum
    REPORTED  R_sol = R0 + R1   unconditionally

Four refusals sit around that fit, and **each asks a question the others cannot**::

    check_admissible          can this SET identify the shared three?   InadmissibleFitSet
    order_spread_pct          was the minimum REPRODUCIBLE?             UnstableFitSurface
    Qg plausibility band      is the minimum PHYSICAL?                  ImplausibleArtifact
    holdout_report            does it PREDICT a spectrum it has not     PoorGeneralisation
                              seen? (needs an answer key)

The middle two are the pair most easily confused, and the measurement that separates
them is in :class:`ImplausibleArtifact`: two seed grids clustered inside the poisoned
basin reproduce each other to 0.5 % and 2 % while being wrong by 7689 % and 8251 %.
**Reproducibility is not correctness**, and only the third row asks about the second.

**All three shared parameters are load-bearing, and this was measured.** Sharing only
``Qg`` and ``nd`` — pinning the geometric element to an ideal capacitor, ``ng = 1`` —
does not degrade gracefully: it gives ``+4.3e8 %`` on the most conductive standard.
Sharing two of the three gives 4.93 % / 7.38 % against 1.55 %.

Topology. This is :data:`~softae.analysis.eis.models.EIS_CIRCUITS`'s
``blocking_coplanar_L`` (``L0-R0-CPE0-p(R1,C0)``) with **two deliberate differences**,
and it is defined in closed form here rather than reused from that registry:

1. ``C0`` becomes a CPE with a *fitted, shared* exponent. The registry's pure capacitor
   is the ``ng = 1`` case the paragraph above measures as catastrophic.
2. A **known** shunt branch ``Y_shunt(f)`` sits across the path. The registry's circuit
   strings are fitted through ``impedance.py``'s ``wrapCircuit``, which has no way to
   express either a frequency-resolved constant or a parameter shared across spectra.

So the registry model is not reusable *as a registry model* here. Its docstring
(*"SHIPS UNUSED — L must be pinned from a short blank"*) names exactly this design, and
the pinning it asks for is what :attr:`ConstrainedSpectrum.pinned_l_H` carries.

**HF-inductive truncation is worth having, and it is no longer a precondition — that
claim did not survive the multi-start.** Spectra should reach this module with the
contiguous ``Im Z > 0`` run at the top of the band removed
(:func:`~softae.analysis.eis.gates.gate_hf_inductive`'s rule, which already ships).
Measured against the four NIST standards, same thin preparation both times:

======================  =======================  =====================
what was measured        R_sol MAE (in-sample)    σ MAE vs NIST
======================  =======================  =====================
before the multi-start   2.9 % → **6295 %**       —
after the multi-start    3.54 % → **3.50 %**      0.98 % → **2.41 %**
======================  =======================  =====================

So the three orders of magnitude were a property of the *seeding*, not of the inductive
points: leaving them in perturbed the continuation's chain enough to send it into the
poisoned basin, and any ordering could do the same to a clean set. What survives is a
factor of ~2.5 on σ, which is a real effect and a real reason to truncate, but it is not
a precondition and this module no longer behaves catastrophically without it. The
mechanism named for the old figure still holds directionally — a blocking cell has no
inductance of its own, so an inductive point is the fixture and the optimiser parks it
in ``Qg``, which is *shared*, so a handful of bad points at one end of one spectrum
degrades the artifact for the whole set. Sharing parameters shares contamination too.

Layout. This is a package only because the single module reached 1340 lines; every
name below is importable from ``softae.analysis.eis.constrained_fit`` exactly as it
was before, and each constant still sits beside the measurement that fixed it::

    refusals.py       the exception hierarchy, and only that
    model.py          parameter split, bounds, the known shunt, Z(f), one spectrum
    admissibility.py  can this set identify the shared three; is the artifact physical
    fit.py            tolerances, seed grids, fit_shared, fit_frozen
    validation.py     leave-one-out against a known answer, and the gate on it
"""

from __future__ import annotations

from .admissibility import (
    MIN_ARC_RESOLVED,
    QG_PLAUSIBLE_MAX,
    Admissibility,
    check_admissible,
    implausible_shared,
)
from .fit import (
    CONSENSUS_COST_FACTOR,
    CONSTRAINED_FIT_TOL,
    EXACT_COST_PER_RESIDUAL,
    MAX_NFEV_GLOBAL,
    MAX_NFEV_SINGLE,
    ORDER_SPREAD_REFUSE_PCT,
    R_SOL_SEED_MULTIPLIERS,
    SHARED_SEED_GRID,
    SharedArtifact,
    WellFit,
    fit_frozen,
    fit_shared,
)
from .model import (
    BEYOND_COVERAGE_POLICIES,
    BOUNDS,
    PARAM_NAMES,
    PER_SPECTRUM_PARAMS,
    SHARED_PARAMS,
    TWO_PI,
    ConstrainedSpectrum,
    ShuntTable,
    arc_is_resolved,
    conductance_r_sol,
    fixture_admittance,
    logger,
    model_impedance,
)
from .refusals import (
    ImplausibleArtifact,
    InadmissibleFitSet,
    PoorGeneralisation,
    UnmeasuredFixtureShunt,
    UnstableFitSurface,
)
from .validation import (
    HOLDOUT_REFUSE_PCT,
    HoldoutReport,
    HoldoutResult,
    holdout_report,
    refuse_unless_generalises,
)

__all__ = [
    "BEYOND_COVERAGE_POLICIES",
    "BOUNDS",
    "CONSENSUS_COST_FACTOR",
    "CONSTRAINED_FIT_TOL",
    "EXACT_COST_PER_RESIDUAL",
    "HOLDOUT_REFUSE_PCT",
    "MAX_NFEV_GLOBAL",
    "MAX_NFEV_SINGLE",
    "MIN_ARC_RESOLVED",
    "ORDER_SPREAD_REFUSE_PCT",
    "PARAM_NAMES",
    "PER_SPECTRUM_PARAMS",
    "QG_PLAUSIBLE_MAX",
    "R_SOL_SEED_MULTIPLIERS",
    "SHARED_PARAMS",
    "SHARED_SEED_GRID",
    "TWO_PI",
    "Admissibility",
    "ConstrainedSpectrum",
    "HoldoutReport",
    "HoldoutResult",
    "ImplausibleArtifact",
    "InadmissibleFitSet",
    "PoorGeneralisation",
    "SharedArtifact",
    "ShuntTable",
    "UnmeasuredFixtureShunt",
    "UnstableFitSurface",
    "WellFit",
    "arc_is_resolved",
    "check_admissible",
    "conductance_r_sol",
    "fit_frozen",
    "fit_shared",
    "fixture_admittance",
    "holdout_report",
    "implausible_shared",
    "logger",
    "model_impedance",
    "refuse_unless_generalises",
]
