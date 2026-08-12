"""Turning gate results into the verdict the rest of the system already speaks.

Two vocabularies exist and **both are correct**, because they measure different
things:

``GateResult.severity`` ∈ ``{block_point, block_spectrum, flag}``
    The *scope of the consequence* — a property of the **check**.
``Verdict`` ∈ ``{ACCEPT, SUSPECT, REJECT}``
    The *outcome for the consumer* — a property of the **measurement**.

So they compose rather than translate: gates produce, and this module reduces. The
reduction targets :class:`~softae.analysis.quality.QualityReport` deliberately —
:func:`softae.core.autonomous_wiring._scalar_from_eis_raw` already branches on
``report.ok``, so the campaign's consumer contract does not change shape at all when
the gated engine arrives behind it.

Imports run one way only: ``policy → quality``, never the reverse. ``quality.py``
stays the stable outward-facing layer and this package stays the new physics layer
depending on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from softae.analysis.eis.gates import (
    BLOCK_POINT,
    BLOCK_SESSION,
    BLOCK_SPECTRUM,
    FLAG,
    GateResult,
    gate_metrics,
)
from softae.analysis.quality import QualityReport, Verdict

logger = structlog.get_logger(__name__)

#: Reporting modes that mean "this is a bound, not a value".
BOUND_MODES = ("bound", "bound_unqualified")


def reduce_gates(
    results: Sequence[GateResult],
    *,
    n_surviving: int,
    min_fit_pts: int,
    report_mode: str = "split",
    enabled: bool = True,
) -> QualityReport:
    """Reduce a gate log to one accept / suspect / reject verdict.

    ==================================  =========  ==========================
    Condition                           Verdict    Why
    ==================================  =========  ==========================
    any ``block_spectrum`` failed       REJECT     the physics is absent
    survivors < ``min_fit_pts``         REJECT     no support for a fit
    ``report_mode`` is a bound          SUSPECT    a result, but not a value
    any ``block_point`` dropped > 0     SUSPECT    spectrum survived, reduced
    any ``flag`` failed                 SUSPECT    degraded, not disqualifying
    otherwise                           ACCEPT     —
    ==================================  =========  ==========================

    **Bound-mode ⇒ SUSPECT is the load-bearing choice.** An upper bound is a
    legitimate scientific output (framework §4.8) but it is emphatically not a value,
    and nothing that consumes σ as a number should silently take one. ``SUSPECT`` is
    exactly the existing "use it, but flag" rung, so a bound stays visible and stays
    distinguishable; the objective extractor refuses bounds separately and explicitly,
    where that decision belongs.

    ``enabled=False`` mirrors :func:`softae.analysis.quality.gate_raw_measurement`:
    every check still runs and a would-be rejection is logged, but the verdict is
    downgraded to SUSPECT so nothing is discarded while the thresholds are still
    being observed against real runs.
    """
    issues: list[str] = []
    metrics = gate_metrics(results)
    verdict = Verdict.ACCEPT

    for r in results:
        if r.passed:
            continue
        if r.severity in (BLOCK_SPECTRUM, BLOCK_SESSION):
            issues.append(f"{r.name}: {r.detail}")
            verdict = Verdict.REJECT
        elif r.severity == FLAG:
            issues.append(f"{r.name}: {r.detail}")
            if verdict is Verdict.ACCEPT:
                verdict = Verdict.SUSPECT

    for r in results:
        if r.severity == BLOCK_POINT and r.n_dropped:
            issues.append(f"{r.name} dropped {r.n_dropped} points")
            if verdict is Verdict.ACCEPT:
                verdict = Verdict.SUSPECT

    metrics["n_surviving"] = float(n_surviving)
    if int(n_surviving) < int(min_fit_pts):
        issues.append(
            f"only {int(n_surviving)} points survived gating "
            f"(need {int(min_fit_pts)})"
        )
        verdict = Verdict.REJECT

    if report_mode in BOUND_MODES:
        issues.append("resolution-limited — σ reported as an upper bound")
        if verdict is Verdict.ACCEPT:
            verdict = Verdict.SUSPECT

    if verdict is Verdict.REJECT and not enabled:
        logger.warning(
            "eis_gate_would_reject", issues=issues, metrics=metrics,
            msg="gates observing only — spectrum used despite failing checks",
        )
        return QualityReport(Verdict.SUSPECT, issues + ["gates observing only"],
                             metrics)

    if verdict is Verdict.REJECT:
        logger.warning("eis_gate_reject", issues=issues, metrics=metrics)
    elif issues:
        logger.info("eis_gate_suspect", issues=issues)

    return QualityReport(verdict, issues, metrics)


#: Reference-electrode states, named for *this board's* geometry (F13/R19).
#:
#: The RE is a **stripe lying between CE and WE** with no direct electrical connection
#: to either. It reaches the cell only through whatever spans the coplanar gap. So the
#: control loop is closed by the *sample itself*, and RE integrity is a property of
#: what is on the board rather than of the wiring:
#:
#: ``bridged_by_sample``
#:     Cast material spans the stripes. The loop is closed and the spectrum is valid.
#:     This is the normal case for every film measurement.
#: ``tied_to_ce``
#:     RE jumpered to CE at the connector — closes the loop electrically without
#:     needing anything on the board. Required for two-terminal reference components,
#:     which sit across CE/WE and do not touch the RE stripe.
#: ``open_by_geometry``
#:     Nothing bridges the stripes — bare board, air gap. The RE floats **by
#:     construction**, not by fault. Quadrant violations here are expected and
#:     structural; there is no wiring to repair.
#: ``unverified``
#:     Unrecorded. The honest default, since nothing in the acquisition path captures
#:     this yet and assuming a closed loop would suppress the diagnostic that matters.
RE_STATES = (
    "unverified", "bridged_by_sample", "tied_to_ce", "open_by_geometry", "connected",
)

#: States in which the potentiostat's control loop is actually closed.
RE_CLOSED_LOOP = frozenset({"bridged_by_sample", "tied_to_ce", "connected"})

#: States in which the RE senses a real potential **in the sample's own conducting
#: medium** — overhaul R26's precondition for ``K_config_factor = 2``.
#:
#: Deliberately *not* :data:`RE_CLOSED_LOOP`, though it is a subset of it. The two
#: answer different questions and the difference is a factor of two:
#:
#: - ``RE_CLOSED_LOOP`` asks *is the control loop closed?* — which is what decides
#:   whether a quadrant violation is instrument-side or structural (§3.7, F13).
#: - ``RE_IONIC_CONTACT`` asks *do ions reach the reference stripe?* — which is what
#:   the §3.8 symmetry derivation assumes when it puts RE at the mean of the two
#:   electrode potentials.
#:
#: ``tied_to_ce`` separates them. Jumpering RE to CE closes the loop perfectly while
#: making the RE read the counter electrode, so the measurement *is* two-electrode by
#: construction and its factor is 1. Applying 2 there would be F16 self-inflicted:
#: a clean 2× on the absolute number with the spectrum, fit and residuals all perfect.
#: ``connected`` is likewise excluded — it records that the wire is on, which says
#: nothing about whether anything spans the gap.
#:
#: F17 is the general statement: with no ionic path the RE floats onto a capacitive
#: divider whose ratio depends on the load (α measured 2.2–23.8, and not reproducible
#: even at fixed load — 9.85 and 4.96 for the same 1 nF part). There is no factor to
#: apply, verified or otherwise.
RE_IONIC_CONTACT = frozenset({"bridged_by_sample"})


def build_context(
    *,
    envelope: Any = None,
    gates: Any = None,
    cell: Any = None,
    blocking: bool = True,
    re_connection: str = "unverified",
    re_contact_verified: bool = False,
) -> dict[str, Any]:
    """Assemble the ``ctx`` mapping every gate reads.

    Kept in one place because gates reach into it by ``(section, key)`` and a typo in
    a section name would silently fall back to a default rather than fail — which is
    the correct behaviour for a gate, and therefore a bad place to build the dict
    ad hoc at each call site.
    """
    from softae.analysis.eis.envelope import instrument_envelope
    from softae.analysis.eis.settings import eis_settings

    return {
        "envelope": envelope if envelope is not None else instrument_envelope(),
        "gates": gates if gates is not None else eis_settings().gates,
        "cell": {"blocking": bool(blocking), "constant": cell},
        "meta": {
            "re_connection": str(re_connection or "unverified"),
            "re_contact_verified": bool(re_contact_verified),
        },
    }
