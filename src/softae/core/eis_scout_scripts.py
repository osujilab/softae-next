"""Scout first, then measure again only if the scout sweep was not enough.

:mod:`softae.analysis.eis.scout` decides *where the next sweep should put its
points*; :mod:`softae.drivers.mscr_library` renders a segment list as
MethodSCRIPT. Between them sits the acquisition shape itself, which is the thing
every call site needs and no call site should own twice.

**The sweep the operator asked for runs first, and is usually the measurement.**
It doubles as the scout: it is a real spectrum, so if the arc closes with enough
band below its apex there is nothing to gain by taking another one, and the
follow-up is skipped. Measured on the source artifact's wells, 11 of 15 ambient
and 12 of 12 humid were adequate on ``Quick`` alone — so in the common case this
costs nothing at all, and only the wells that genuinely need a wider or denser
sweep pay for a second one.

**A follow-up can only ever be wider or denser, never narrower.** Two mechanisms
hold that, and both are worth stating because they cover different cases. A
*preset* follow-up steps one rung down a ladder ordered by floor, so it reaches
lower by construction. A *segmented* follow-up is only ever built on the verdict
``extend_low``, which means the apex sits less than
``band_below_apex_min_decades`` above the sweep floor — while the plan's own floor
is ``arc_decades_below`` beneath that apex. With the shipped pairing of those two
(1.0 and 1.0) the plan's floor is therefore always below the floor of the sweep it
follows, and it carries roughly twice the points across the arc.

**Nothing survives an acquisition unit.** There is no plan cache here to go
stale: a decision is consumed by the measurement that produced it and then
discarded. Between a sweep and the next sweep — or between one Run press and the
next — the operator may have changed RH, temperature, or the sample, and the apex
was measured moving ~100x across an RH change. A planner that cannot remember
cannot remember wrongly.

**Provenance says which sweep produced the row.** ``eis_sweep_role`` is
``"scout"`` on an accepted first sweep and ``"follow_up"`` on a second one, and a
follow-up carries the verdict that triggered it. That matters here for the same
reason :mod:`softae.core.eis_scripts`' docstring exists: the defect being guarded
against is a row whose stored parameters name a sweep that never ran, and a
follow-up differs from the operator's selected preset *by design*.

**Inert when ``[eis.scout] enabled = false``**, which is how it ships:
:meth:`ScoutPlanner.observe` and :meth:`ScoutPlanner.build_follow_up` both return
``None`` immediately, so a call site takes the same single-sweep path, with the
same bytes, that it took before this module existed. Both call sites pin that
with a characterization test rather than trusting this paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from softae.analysis.eis.scout import (
    ScoutDecision,
    ScoutSettings,
    plan_for,
    scout_decision,
    scout_settings,
)

logger = structlog.get_logger(__name__)

#: ``eis_sweep_role``: which sweep of an acquisition produced the stored row.
SWEEP_ROLE_SCOUT = "scout"
SWEEP_ROLE_FOLLOW_UP = "follow_up"

#: ``eis_sweep``: what kind of grid ran. Absent means the configured preset.
SEGMENTED_SWEEP_TAG = "segmented"
WIDER_PRESET_SWEEP_TAG = "wider_preset"

#: A segmented grid is always modelled, never stopwatched: it is generated per
#: sample, so no anchor in ``preflight.EIS_ANCHOR_GRIDS`` can ever match it. That
#: is *structural* extrapolation, which is a different thing from "extrapolated
#: because somebody edited a preset and retired its stopwatch" — and the reason
#: string is what lets an operator tell the two apart.
SEGMENTED_DURATION_BASIS = "extrapolated"
SEGMENTED_DURATION_BASIS_REASON = (
    "this grid is generated per sample, so no timing anchor can ever match it"
)

#: Why the sweep just taken was accepted as the measurement. Logged on every
#: verdict that does not earn a follow-up: "undeclared is unknown, never empty" —
#: the scout declining to ask for another sweep is a fact and is recorded as one.
DEGRADE_REASON: dict[str, str] = {
    "ok": "arc closed with enough band below the apex; this sweep is the measurement",
    "no_arc": "no interior apex clears the prominence cut; nothing to widen toward",
    "no_data": "spectrum unreadable by the arc guard; nothing to widen toward",
    "extend_low": (
        "apex at or below the sweep floor; a wider sweep is the answer, not a "
        "band extrapolated below an apex nobody has seen"
    ),
    "extend_high": "apex at the top of the sweep and 200 kHz is the ceiling",
}


def segmented_params(
    base_params: dict[str, Any] | None,
    segments: tuple[tuple[float, float, int], ...],
    *,
    decision: ScoutDecision | None = None,
) -> dict[str, Any]:
    """The ``eis_params`` a segmented sweep is honestly described by.

    *segments* must already be resolved (quantized, boundary-nudged), because
    these numbers are what gets stored and a stored grid differing from the
    emitted one by a nudge is the same class of lie as a stale preset name.
    """
    params = dict(base_params or {})
    params.update(
        {
            # Aggregates over bands, not one log grid — `eis_sweep` says so.
            "f_hi": float(segments[0][0]),
            "f_lo_mHz": float(segments[-1][1]) * 1000.0,
            "npts": sum(int(npts) for _, _, npts in segments),
            "eis_sweep": SEGMENTED_SWEEP_TAG,
            "eis_sweep_role": SWEEP_ROLE_FOLLOW_UP,
            "eis_segments": [[float(a), float(b), int(n)] for a, b, n in segments],
            "eis_duration_basis": SEGMENTED_DURATION_BASIS,
            "eis_duration_basis_reason": SEGMENTED_DURATION_BASIS_REASON,
        }
    )
    if decision is not None:
        params["eis_scout_trigger_verdict"] = decision.verdict
        params["eis_scout_apex_hz"] = float(decision.f_apex_hz)
    return params


@dataclass
class ScoutPlanner:
    """Decides whether one more sweep is worth taking, and writes it if so.

    Deliberately **stateless beyond its settings**: see the module docstring on
    why no plan is allowed to outlive the measurement that produced it.
    """

    settings: ScoutSettings = field(default_factory=scout_settings)
    #: Which call site this planner serves; carried on every log line.
    site: str = ""
    #: A call site's **own** control over acquiring, overriding ``[eis.scout]
    #: actuate`` for this planner alone. The Manual Control tab passes its
    #: checkbox here, so a global ``actuate = true`` cannot switch that tab on:
    #: manual measurements are where non-standard samples turn up — a two-arc
    #: system, a stack, something nobody has characterised — and whether this one
    #: is standard enough to plan a sweep around is a judgement made by a person
    #: at the rig. ``None`` defers to the global flag.
    actuate: bool | None = None

    @property
    def observing(self) -> bool:
        """Does the decision run at all?

        A site that may acquire always observes, because acting on a verdict
        requires having one. Deferring this to the global ``enabled`` alone would
        leave a call site's own control a dead switch whenever ``enabled`` was
        off — and ``enabled`` ships off.
        """
        return bool(self.settings.enabled or self.planning)

    @property
    def planning(self) -> bool:
        """May a decision actually call for a second sweep?"""
        if self.actuate is not None:
            return bool(self.actuate)
        return bool(self.settings.enabled and self.settings.actuate)

    def observe(self, channel: int, eis_result: Any) -> ScoutDecision | None:
        """Decide from the sweep just taken, and stamp the verdict onto its row.

        Returns ``None`` — changing nothing — whenever the scout is off. **Never
        raises**: it is called from a measurement path, and a planner that can
        abort a running sweep is worse than one that declines to plan.
        """
        if not self.observing:
            return None
        ch = int(channel)
        try:
            decision = scout_decision(
                eis_result.frequency,
                eis_result.z_imag_neg,
                getattr(eis_result, "phase", None),
                settings=self.settings,
            )
        except Exception as exc:
            logger.warning("eis_scout_observe_failed", site=self.site,
                           channel=ch, error=str(exc))
            return None

        logger.info(
            "eis_scout_verdict",
            site=self.site,
            channel=ch,
            verdict=decision.verdict,
            arc_state=decision.arc_state,
            f_apex_hz=float(decision.f_apex_hz),
            band_below_apex_decades=float(decision.band_below_apex_decades),
            apex_prominence_rel=float(decision.apex_prominence_rel),
            actuate=self.planning,
            reason=DEGRADE_REASON.get(decision.verdict, decision.verdict),
        )
        try:
            eis_result.eis_params["eis_scout_verdict"] = decision.verdict
            eis_result.eis_params["eis_scout_apex_hz"] = float(decision.f_apex_hz)
            eis_result.eis_params["eis_sweep_role"] = SWEEP_ROLE_SCOUT
        except Exception:                                   # pragma: no cover
            pass
        return decision

    def build_follow_up(
        self,
        path: str,
        channel: int,
        base_params: dict[str, Any] | None,
        decision: ScoutDecision | None,
    ) -> dict[str, Any] | None:
        """Write a script for a **second** sweep, or ``None`` to accept the first.

        ``None`` is the common answer and the one that keeps the economics
        positive: the sweep already taken is a real measurement, and re-measuring
        an adequate spectrum buys nothing. Nothing is written to *path* on that
        path, so the script the first sweep ran is left exactly as it was.

        Returns the ``eis_params`` the caller must record for the follow-up.
        """
        if not self.planning or decision is None:
            return None
        ch = int(channel)

        if decision.verdict != "extend_low":
            logger.info("eis_scout_sweep_accepted", site=self.site, channel=ch,
                        verdict=decision.verdict,
                        reason=DEGRADE_REASON.get(decision.verdict,
                                                  decision.verdict))
            return None

        # Two ways to widen, and which one applies turns on a single question:
        # is there a measured apex in hand? `plan_for` answers it — non-empty
        # when `arc_closure` found an interior maximum that cleared the
        # prominence cut, empty when the arc is still below the floor and nobody
        # has seen it. The apex is never re-derived here; a module that computes
        # its own is how this tree came by three extremum finders and one
        # published sigma wrong by 10x.
        plan = plan_for(decision, self.settings)
        if plan:
            written = self._write_segmented(path, ch, base_params, plan, decision)
            if written is not None:
                return written
        return self._write_wider_preset(path, ch, base_params, decision)

    # ── Internals ────────────────────────────────────────────────────────────

    def _write_segmented(
        self, path: str, channel: int, base_params: dict[str, Any] | None,
        plan: tuple[tuple[float, float, int], ...], decision: ScoutDecision,
    ) -> dict[str, Any] | None:
        from softae.drivers.mscr_library import (
            eis_segmented_mscrbuild,
            resolve_segments,
        )

        try:
            # Resolved here rather than only inside the emitter, because these
            # nudged bounds are what the provenance records and the two must be
            # the same numbers. `resolve_segments` is idempotent, so the emitter
            # resolving them again changes nothing.
            resolved = resolve_segments(plan)
            eis_segmented_mscrbuild(
                path,
                mux_ch=channel,
                segments=resolved,
                mVac=int((base_params or {}).get("mv_ac", 10)),
                mVdc=int((base_params or {}).get("mv_dc", 0)),
            )
        except Exception as exc:
            logger.warning("eis_scout_script_failed", site=self.site,
                           channel=channel, path=str(path), error=str(exc),
                           reason="falling back to the next wider preset")
            return None

        params = segmented_params(base_params, resolved, decision=decision)
        logger.info("eis_scout_follow_up", site=self.site, channel=channel,
                    kind=SEGMENTED_SWEEP_TAG, n_segments=len(resolved),
                    npts=params["npts"], f_lo_hz=params["f_lo_mHz"] / 1000.0,
                    duration_basis=SEGMENTED_DURATION_BASIS,
                    duration_basis_reason=SEGMENTED_DURATION_BASIS_REASON)
        return params

    def _write_wider_preset(
        self, path: str, channel: int, base_params: dict[str, Any] | None,
        decision: ScoutDecision,
    ) -> dict[str, Any] | None:
        """Step one rung down the preset ladder and re-measure there.

        **The blind path**, and the right one whenever no apex is in hand: the arc
        is somewhere below the floor and nothing has measured where. A segmented
        grid needs a centre to lay itself around, so widening by a known amount is
        all that can honestly be done — and the follow-up gets its own verdict, so
        finding the apex on the next rung turns the next attempt into a planned
        one.

        One rung, not straight to the widest: ``Quick -> Longest`` is +499 s on a
        spectrum that may well close at ``Standard``'s floor.
        """
        step = self._next_wider_preset(base_params)
        if step is None:
            logger.info("eis_scout_sweep_accepted", site=self.site,
                        channel=channel, verdict=decision.verdict,
                        reason="already at the widest configured preset")
            return None
        name, grid = step

        from softae.core.preflight import eis_duration_basis
        from softae.drivers.mscr_library import eis_run_mscrbuild

        params = dict(base_params or {})
        # Grid only. `mv_ac` / `mv_dc` stay the operator's: a follow-up widens the
        # band, it does not quietly re-specify the excitation.
        params.update(
            {
                "f_hi": grid.f_hi,
                "f_lo_mHz": grid.f_lo_mHz,
                "npts": grid.npts,
                "eis_sweep": WIDER_PRESET_SWEEP_TAG,
                "eis_sweep_role": SWEEP_ROLE_FOLLOW_UP,
                "eis_preset": name,
                "eis_scout_trigger_verdict": decision.verdict,
                "eis_duration_basis": eis_duration_basis(name),
            }
        )
        try:
            eis_run_mscrbuild(
                path, mux_ch=channel,
                mVac=params.get("mv_ac", 10), f_hi=grid.f_hi,
                f_lo=grid.f_lo_mHz, npts=grid.npts, mVdc=params.get("mv_dc", 0),
            )
        except Exception as exc:
            logger.warning("eis_scout_script_failed", site=self.site,
                           channel=channel, path=str(path), error=str(exc),
                           reason="no follow-up; the sweep just taken stands")
            return None

        logger.info("eis_scout_follow_up", site=self.site, channel=channel,
                    kind=WIDER_PRESET_SWEEP_TAG, preset=name,
                    f_lo_mHz=grid.f_lo_mHz, npts=grid.npts,
                    duration_basis=params["eis_duration_basis"])
        return params

    def _next_wider_preset(self, base_params: dict[str, Any] | None):
        """The configured preset with the highest floor still below this one's.

        ``None`` when nothing reaches lower — at which point the honest move is
        to keep the sweep already taken rather than to invent a grid.
        """
        from softae.config.loader import eis_presets
        from softae.core.eis_scripts import EISParams

        try:
            current = float((base_params or {}).get("f_lo_mHz", 0.0))
            candidates = [
                (name, EISParams.from_preset(name))
                for name in (eis_presets() or {})
            ]
        except Exception as exc:
            logger.warning("eis_scout_preset_ladder_unreadable",
                           site=self.site, error=str(exc))
            return None

        wider = [(n, g) for n, g in candidates if 0 < g.f_lo_mHz < current]
        return max(wider, key=lambda pair: pair[1].f_lo_mHz) if wider else None


__all__ = [
    "DEGRADE_REASON",
    "SEGMENTED_DURATION_BASIS",
    "SEGMENTED_DURATION_BASIS_REASON",
    "SEGMENTED_SWEEP_TAG",
    "SWEEP_ROLE_FOLLOW_UP",
    "SWEEP_ROLE_SCOUT",
    "WIDER_PRESET_SWEEP_TAG",
    "ScoutPlanner",
    "segmented_params",
]
