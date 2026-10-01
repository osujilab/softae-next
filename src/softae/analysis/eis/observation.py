"""One σ observation per spectrum: the number the engine stated, and what kind of claim it is.

The single reading of a :class:`~softae.analysis.eis.report.SpectrumReport` that the store
(``fit_results`` / ``doe_parameters``) and the campaign loop both key on, so the mapping
from ``SigmaReport`` to *value / lower bound / upper bound* exists exactly once
(``[e150]`` §3, with R3's amendments and R7's basis in
``docs/SubAgent docs/censored_tell_path_recommendations.md``).

=================================  ===============  ========================  ====================
Engine state                       ``kind``         ``reported``              admitted
=================================  ===============  ========================  ====================
``value``                          ``value``        ``value``                 unless bad-data U
``bound`` / ``bound_unqualified``  ``upper_bound``  ``upper_bound``           route A only
``unavailable``, active route,     ``lower_bound``  ``regime_sigma_lower``    yes
finite ``regime_sigma_lower``
anything else                      ``None``         ``None``                  no
=================================  ===============  ========================  ====================

Every row is also gated on ``report.ok``: a gate-rejected number is recorded, never
admitted, and its *classified* status is unaffected.

**Admitted** means the number may become the campaign objective. An old-route upper bound
is recorded but never admitted: on regime-A films the legacy loss ceiling sits 0.5–3.5
decades below the film's σ, so it is a statement about the instrument, not the sample.
With ``[eis] regime_aware`` off, ``regime_active`` is never set, so no lower bound ever
appears and no bound is ever admitted.

**Regime U splits by its reason** (operator ruling 2026-10-01, narrowing arming review F3):

=====================  ==================================  ================  ==============
U family               ``regime_reason``                   value admitted    classified
=====================  ==================================  ================  ==============
bad data               ``incoherent``,                     no                no
                       ``too_many_nonphysical``,
                       ``too_few_points``,
                       ``a_too_many_nonphysical``,
                       ``phase_incoherent``, *any other*
unrecognised shape     ``non_monotone``,                   yes               yes
                       ``floor_phase_ambiguous``,
                       ``no_cpe_drop``
=====================  ==================================  ================  ==============

*Bad data* means the spectrum itself is not a measurement (an unclosed reference electrode,
a jagged phase trace — Q5): a number today's route still extracts from it is recorded and
never told, and the read counts toward a park. *Unrecognised shape* means the data are
coherent but fit none of A/B/C: treated exactly like B and C, so today's value is told and
anything else is a measured, untold read. ``""`` (the classifier did not run: legacy
engine, early return) is not U and keeps the value admitted.

**Classified** is independent of admission: a spectrum labelled A, B, C or
unrecognised-shape U was acquired and screened, which is what the park counter means by
*measured* (R1), whether or not its σ is admissible.

No I/O and no config reads; never raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from softae.analysis.eis.regime import A, B, C, U
from softae.analysis.eis.report import REGIME_A_FOOT_LOWER, REGIME_A_PASSIVE

VALUE = "value"
LOWER_BOUND = "lower_bound"
UPPER_BOUND = "upper_bound"
KINDS = (VALUE, LOWER_BOUND, UPPER_BOUND)

#: Regime labels that are always *acquired, screened and classified*. ``U`` is classified
#: only for an unrecognised-shape reason (:data:`U_SHAPE_REASONS`); ``""`` means the
#: classifier did not run, which is never a classification.
CLASSIFIED_REGIMES = (A, B, C)
#: ``regime_reason`` tokens of a U whose data are coherent but whose shape is none of
#: A/B/C — measured, treated like B/C. Retyped from ``regime.classify_regime``, which has
#: no constants for its reasons; ``tests/test_eis_observation.py`` pins both families
#: against that source so a new token cannot slip in unfiled.
U_SHAPE_REASONS = ("non_monotone", "floor_phase_ambiguous", "no_cpe_drop")
#: ``regime_reason`` tokens of a U whose data are not a measurement. Documentation and
#: test anchor only: the code never consults it, because **any U reason outside
#: :data:`U_SHAPE_REASONS` -- these, or a new/unknown token -- is bad data.** Fail safe: an
#: unfiled reason must cost a park, never tell a number from an unscreened spectrum.
U_BAD_DATA_REASONS = ("incoherent", "too_many_nonphysical", "too_few_points",
                      "a_too_many_nonphysical", "phase_incoherent")
#: Upper-bound bases the regime-A route produces. Only these may be admitted.
ROUTE_A_BOUND_BASES = (REGIME_A_PASSIVE,)

__all__ = ["VALUE", "LOWER_BOUND", "UPPER_BOUND", "KINDS", "CLASSIFIED_REGIMES",
           "U_SHAPE_REASONS", "U_BAD_DATA_REASONS", "SigmaObservation", "is_classified",
           "sigma_observation", "unmeasured"]


@dataclass(frozen=True)
class SigmaObservation:
    """What one spectrum says about σ, in the store's and the optimizer's vocabulary."""

    kind: str | None        # one of KINDS; None = nothing stateable
    reported: float | None  # the σ (S/cm) the engine stated: the value, or the bound
    lower: float | None     # interval's other side when kind == UPPER_BOUND (K/R_foot)
    regime: str             # classifier label (A/B/C/U); "" if the classifier did not run
    #: Where :attr:`reported` came from: ``R_basis`` for a value, ``upper_bound_basis``
    #: for an upper bound (the ceiling's source, e.g. ``loss_at_numerator`` or
    #: ``regime_a_passive``), ``regime_a_foot_lower`` for a lower bound.
    basis: str
    R_film_ohm: float | None
    admitted: bool          # may become the campaign objective
    classified: bool        # see :func:`is_classified`
    #: The basis of :attr:`R_film_ohm` itself (``SigmaReport.R_basis``). Differs from
    #: :attr:`basis` on an old-route upper bound, where the reported number is a loss
    #: ceiling and the resistance is the fit's — pairing them would name the wrong source.
    R_film_basis: str = ""
    #: The classifier's reason (``SigmaReport.regime_reason``); what splits regime U.
    regime_reason: str = ""

    @property
    def counts_as_measured(self) -> bool:
        """R1: the park counter's *measured* — an admitted number, or a classified spectrum."""
        return (self.admitted and self.kind is not None) or self.classified


def is_classified(regime: str, reason: str = "") -> bool:
    """R1's *measured*: A, B, C, or a U whose reason is an unrecognised shape."""
    return regime in CLASSIFIED_REGIMES or (regime == U and reason in U_SHAPE_REASONS)


def unmeasured(regime: str = "", reason: str = "") -> SigmaObservation:
    """Nothing stateable — for acquisition failures and unreadable reports.

    A ``"U"`` with no *reason* is bad data (unclassified), by the fail-safe rule above.
    """
    return SigmaObservation(kind=None, reported=None, lower=None, regime=regime, basis="",
                            R_film_ohm=None, admitted=False,
                            classified=is_classified(regime, reason),
                            regime_reason=reason)


def _positive(x: Any) -> float | None:
    """``float(x)`` when finite and > 0, else ``None``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _label(report: Any, field: str) -> str:
    try:
        label = getattr(report.sigma, field)
    except Exception:  # noqa: BLE001 - never raises, by contract
        return ""
    return label if isinstance(label, str) else ""


def sigma_observation(report: Any) -> SigmaObservation:
    """Map a ``SpectrumReport`` to its :class:`SigmaObservation`. Never raises."""
    regime, reason = _label(report, "regime"), _label(report, "regime_reason")
    try:
        return _observe(report.sigma, bool(report.ok), regime, reason)
    except Exception:  # noqa: BLE001 - a malformed report is unmeasured, not a crash
        return unmeasured(regime, reason)


def _observe(s: Any, ok: bool, regime: str, reason: str) -> SigmaObservation:
    active = bool(s.regime_active)
    classified = is_classified(regime, reason)
    R_film = _positive(s.R_reported_ohm)
    if s.mode == "value":
        kind, reported, lower, basis = VALUE, _positive(s.value), None, s.R_basis
        admitted = regime != U or classified          # bad-data U never tells (F3)
    elif s.mode in ("bound", "bound_unqualified"):
        kind, reported, basis = UPPER_BOUND, _positive(s.upper_bound), s.upper_bound_basis
        lower = _positive(s.regime_sigma_lower) if active else None
        admitted = active and basis in ROUTE_A_BOUND_BASES
    elif s.mode == "unavailable" and active and _positive(s.regime_sigma_lower) is not None:
        kind, reported, lower = LOWER_BOUND, _positive(s.regime_sigma_lower), None
        basis, R_film = REGIME_A_FOOT_LOWER, _positive(s.regime_R_foot_ohm)
        admitted = True
    else:
        return unmeasured(regime, reason)
    if reported is None:
        return unmeasured(regime, reason)
    return SigmaObservation(
        kind=kind, reported=reported, lower=lower, regime=regime, basis=str(basis),
        R_film_ohm=R_film, admitted=admitted and ok, classified=classified,
        R_film_basis=(REGIME_A_FOOT_LOWER if kind == LOWER_BOUND else str(s.R_basis)),
        regime_reason=reason,
    )
