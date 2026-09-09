"""The refusals, and nothing else: one exception per question the fit can fail.

Four refusals sit around the fit and **each asks a question the others cannot** —
the table is in this package's ``__init__``. They live together, apart from their
raise sites, because the thing consumers depend on is the *hierarchy*:
:func:`~softae.analysis.eis.constrained_fit.validation.holdout_report` records a
refused fold by catching :class:`InadmissibleFitSet`, so every refusal a fit can
raise has to be a subclass of it and a base separated from its subclasses is a
contract separated from itself.

This module imports nothing from its siblings, which is what lets
:mod:`~softae.analysis.eis.constrained_fit.model` raise
:class:`UnmeasuredFixtureShunt` without an import cycle.
"""

from __future__ import annotations


class InadmissibleFitSet(ValueError):
    """The fitting set cannot identify the shared parameters. A refusal, not a failure."""


class UnmeasuredFixtureShunt(InadmissibleFitSet):
    """No ``G_fixture`` for this channel, and nobody said the shunt was negligible.

    Raised by :func:`~softae.analysis.eis.constrained_fit.model.fixture_admittance`,
    in a different module of this package. **No refusal here sits beside its raiser**,
    and that is the point of this file: the hierarchy below is what
    :func:`~softae.analysis.eis.constrained_fit.validation.holdout_report` catches on,
    so splitting a base class from its subclasses would be splitting the contract. Each
    raise site names its own exception.

    **Why a refusal and not a zero.** Until 2026-09-08 an absent conductance table left
    ``y`` at its initialised zeros and the function returned normally, reporting
    ``n_held = 0`` — which is :attr:`ShuntTable.held_fraction` of 0.0, the *same* value a
    fully covered spectrum reports. "Never measured" was spelled with the token for
    "measured everywhere" (``SUBAGENT_RULES`` §3.1(a)), and the resulting fit is not
    merely uncorrected: on a blocking cell the low-frequency tail is largely the shunt,
    so the optimiser puts the missing admittance somewhere — and ``Qg`` is *shared*, so
    it lands in the artifact for the whole set.

    **The population makes this urgent rather than tidy.** The live calibration
    (``eis_calibrations`` id 23, ``mux16``, 2026-09-03) carries ``G_fixture`` — and
    ``C_stray_F`` — for channels **17-23 and 25 only**, while 3 538 of the corpus's
    3 872 ``role='sample'`` measurements (91 %) sit on channels that have neither.
    Wired as it stood, nine of every ten samples would have been fitted against a shunt
    of exactly zero **and nothing in the result would have said so**, while the
    remaining 334 got a real one — a systematic split inside a single campaign, in the
    parameter the design shares.

    The escape is :func:`fixture_admittance`'s ``beyond_coverage="zero"``, which is the
    same declaration the policy already carried for *partial* coverage: the open blank
    was judged unusable, which is itself positive evidence the shunt is negligible.
    """


class UnstableFitSurface(InadmissibleFitSet):
    """The starts that reached the same minimum did not reach the same answer.

    A subclass of :class:`InadmissibleFitSet` because it is the same kind of event —
    this set cannot support an artifact — and because every caller that already records
    a refusal (:func:`holdout_report`) should record this one too, rather than crashing
    on a new exception type.
    """


class ImplausibleArtifact(InadmissibleFitSet):
    """The descents agreed with each other, and agreed on a non-physical geometry.

    The one refusal here that looks at *what the artifact says* rather than at how the
    optimiser got there — and the measurement that makes it necessary is that
    :class:`UnstableFitSurface` provably does not close this door. Two independent
    seed grids aimed at the poisoned basin, on the four NIST standards:

    ==========================  ============  ==============  =================
    seed grid                    ``Qg``        order spread    ``R_sol`` MAE
    ==========================  ============  ==============  =================
    shipped ``POISONED_STARTS``  9.38e-08      1.96e5 %        refused already
    Qg 4e-8 + 8e-8, nd 0.85      9.38e-08      **2.06 %**      **7689 %**
    Qg 1e-7 + 1.2e-7, nd 0.45    9.48e-08      **0.51 %**      **8251 %**
    ==========================  ============  ==============  =================

    **Two descents into the same wrong basin agree with each other perfectly**, and
    agreement is the entire content of ``order_spread_pct``. :meth:`SharedArtifact.railed`
    fires on none of these either — ``nd`` 0.334 is not quite on its 0.30 bound. Without
    this class the bottom two rows are returned, converged, self-consistent and wrong by
    four orders of magnitude. ``SUBAGENT_RULES`` §3.1: the consensus statistic answers
    *"was this minimum reproducible"*, which is a different question from *"is this
    minimum the right one"*, and it returns the shape of a pass for both.

    A subclass of :class:`InadmissibleFitSet` for the same reason
    :class:`UnstableFitSurface` is: :func:`holdout_report` records it as a refused fold
    rather than crashing on an unfamiliar type.
    """


class PoorGeneralisation(InadmissibleFitSet):
    """The artifact was fitted, was plausible, and does not predict a spectrum it
    has not seen.

    The last of the refusals, and the only one with an answer key behind it: the three
    in :func:`fit_shared` are all self-consistency tests that a set of standards is not
    required for, while this one asks whether the number is *right*. It is therefore
    also the one that cannot run on a campaign well — nothing there carries a
    :attr:`ConstrainedSpectrum.reference_ohm` — which is exactly why the other three
    exist and why this is not a substitute for them.
    """
