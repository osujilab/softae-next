"""Cast a planned thickness series — the execution half of E5 (framework §5.6).

A geometry series is a **designed experiment, not a search**, and that distinction is
the reason this module exists rather than a flag on the campaign path.

``ElectrodeAllocator`` answers *"give me the next free well"*: it skips occupied
electrodes, clears occupancy on a board swap, and narrows a round to what is left. Every
one of those behaviours is right for a Bayesian campaign, which does not know what it
will cast next. A thickness plan answers a different question — *channel 7 gets 150 µm* —
where the channel identity **is** the design. Route a plan through the allocator and one
of them must yield; the allocator skips an occupied well, the plan loses that channel,
and the level balance breaks with nothing raising a hand. That is overhaul F12 returning
by a different route, made worse by the plan's existence making it look prevented.

So here the plan is authoritative. The channels come from it, occupancy is checked
**before** anything is cast, and a conflict is a hard stop naming the channel — re-plan
rather than silently substitute.

Two orderings, both load-bearing
--------------------------------
The plan fixes *which* channel gets *which* level. The order of visiting them is free,
and this module round-robins the levels for two independent reasons that happen to agree:

*Interruption.* A run that dies halfway leaves a prefix. A 7-of-8 subset of a plan
scoring ``r = −0.098`` was measured at ``r = −0.55`` — a sound design degrades into a
confounded one when truncated in channel order. Round-robin keeps every prefix balanced.

*Drift.* §5.6 asks for "interleaved or randomized order so that drift appears as scatter
rather than a false slope".

Drift control
-------------
One member is measured again at the end of the session and tagged
:data:`~softae.analysis.eis.calibration.DRIFT_REPEAT_ROLE`, so its own first measurement
can be differenced against it. This catches *session* drift — the films continuing to
equilibrate over the ~11 minutes a 16-channel sweep takes — which is the one error the
geometry route is not otherwise immune to. Casting drift is a different experiment and
deliberately out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import structlog

logger = structlog.get_logger(__name__)


class GeometrySeriesError(RuntimeError):
    """A series that cannot be cast as planned — naming what to fix.

    Raised rather than degraded. Every failure this covers (an occupied well, an
    unsolvable level, a plan with no channels) invalidates the *design*, and a design
    that has quietly lost a channel produces data indistinguishable from a sound run.
    """


def round_robin_by_level(assignment: Mapping[int, float]) -> list[int]:
    """Channel order that visits every level once before repeating any (§5.6).

    Two distinct properties are wanted from a cast order, and it is worth separating
    them because the obvious algorithm delivers only the first:

    **Balance** — every prefix holds roughly equal numbers of each level. A plain
    round-robin over the level buckets guarantees this, and it is what makes an
    interrupted run salvageable rather than lopsided.

    **Decorrelation** — within a prefix, channel index must not track level. Round-robin
    does *not* guarantee this: taking each level's lowest unused channel first pairs low
    channels with low levels, and the opening block of a real 16-channel plan measured
    ``|r| = 0.61`` that way — worse than the 0.2 the planner worked to achieve for the
    series as a whole. Balance and decorrelation are different constraints and only the
    second is F12.

    So the *level* sequence is round-robin (balance, guaranteed), while *which* channel
    is drawn from each level's bucket is chosen greedily to keep the running correlation
    small.

    **What that buys, measured** — worst case over 30 planned 16-channel/4-level series,
    across every prefix at least *n* long:

    ======  ==============  ===================
    n       channel order   greedy round-robin
    ======  ==============  ===================
    4       0.949           0.899
    6       0.852           0.566
    8       0.754           **0.382**
    12      0.736           0.246
    ======  ==============  ===================

    So an interrupted run is salvageable **once roughly half the series is down**, and
    below that it is not — with four to six points there are too few arrangements for
    any ordering to decorrelate them. That limit is stated rather than smoothed over: a
    quarter-finished series is not a small version of a good design, and treating it as
    one is how F12 happened in the first place.
    """
    from softae.core.thickness_series import correlation

    by_level: dict[float, list[int]] = {}
    for ch, lv in sorted(assignment.items()):
        by_level.setdefault(float(lv), []).append(int(ch))

    levels = sorted(by_level)
    pool = {lv: list(chans) for lv, chans in by_level.items()}
    order: list[int] = []
    lvls: list[float] = []

    while any(pool[lv] for lv in levels):
        for lv in levels:
            remaining = pool[lv]
            if not remaining:
                continue
            # Pick the channel from this level whose addition leaves the prefix least
            # correlated. Ties break on the lowest channel, so the result is stable and
            # a re-run of the same plan produces the same order.
            best = min(
                remaining,
                key=lambda c, _lv=lv: (
                    round(abs(correlation(order + [c], lvls + [_lv])), 12)
                    if len(order) >= 1 else 0.0,
                    c,
                ),
            )
            remaining.remove(best)
            order.append(best)
            lvls.append(lv)
    return order


def choose_drift_channel(assignment: Mapping[int, float]) -> int | None:
    """The member to re-measure at the end of the session.

    A **mid-level** member, not an extreme. The drift control's job is to represent the
    typical sample, and the thinnest and thickest films are the two most likely to
    behave atypically — the thinnest sits closest to any dead-height effect, the
    thickest dries slowest and is therefore the one still changing.

    ``None`` when the series has fewer than two levels, where a drift control would
    measure nothing useful.
    """
    if not assignment:
        return None
    levels = sorted({float(v) for v in assignment.values()})
    if len(levels) < 2:
        return None
    mid = levels[len(levels) // 2] if len(levels) % 2 else levels[len(levels) // 2 - 1]
    candidates = sorted(ch for ch, v in assignment.items() if float(v) == mid)
    return int(candidates[0]) if candidates else None


def verify_channels_free(
    channels: Sequence[int], occupied: Sequence[int] | set[int]
) -> None:
    """Raise unless every planned channel is free. **Never substitutes.**

    The allocator's instinct is to skip an occupied well and hand out the next one.
    Doing that here would silently swap a planned channel for an unplanned one and
    break the level balance the plan was constructed to guarantee — invisibly, because
    the cast would still complete and the data would still look like a series.
    """
    taken = sorted(set(int(c) for c in occupied) & set(int(c) for c in channels))
    if taken:
        raise GeometrySeriesError(
            "planned channel(s) already cast: "
            + ", ".join(str(c) for c in taken)
            + ". A geometry series cannot substitute a free well for a planned one — "
              "the channel assignment IS the design, and swapping one breaks the level "
              "balance silently. Re-plan onto the free channels:\n"
              "  softae-thickness plan --levels <...> --channels <free ones>"
        )


def volumes_for_levels(
    levels_um: Sequence[float],
    *,
    stocks: Mapping[str, Any],
    catalog: Any,
    composition_targets: Sequence[Any] = (),
    deposit_area_mm2: float,
    pump_assignment: Mapping[str, int] | None = None,
    budget_uL: float | None = None,
) -> dict[float, list[float]]:
    """Solve per-stock cast volumes for each thickness level.

    One formulation, N scales. The composition targets fix the *relative* volumes and
    are identical at every level; :class:`~softae.core.formulation.ThicknessTarget`
    fixes the **scale** and is the only thing that differs between them. That is what
    makes this a geometry series rather than a composition series — the chemistry is
    held constant by construction, not by intention.

    Uses the same ``solve_formulation`` the campaign path uses, so a level cast here
    and a thickness target cast by a campaign go through identical arithmetic.
    """
    from softae.core.formulation import ThicknessTarget, solve_formulation

    out: dict[float, list[float]] = {}
    for level in sorted({float(v) for v in levels_um}):
        targets = list(composition_targets) + [
            ThicknessTarget(value_um=level, area_mm2=float(deposit_area_mm2),
                            basis="dry"),
        ]
        plan = solve_formulation(
            dict(stocks), catalog, targets,
            pump_assignment=dict(pump_assignment or {}),
            budget_uL=budget_uL,
        )
        vols = [float(v) for v in plan.per_pump_uL]
        if any(v < 0 for v in vols) or not plan.feasible:
            # Reported, not repaired. Projecting a negative volume onto zero casts a
            # different film than the one requested, and the series would still look
            # like a clean four-level design afterwards.
            raise GeometrySeriesError(
                f"level {level:g} um is infeasible for these stocks "
                f"(volumes {vols}, headroom {plan.headroom_uL:.3g} uL)"
                + ("; " + "; ".join(plan.notes) if plan.notes else "")
            )
        out[level] = vols
        logger.info("geometry_series_level_solved", level_um=level,
                    volumes_uL=vols, notes=list(plan.notes or ()))
    return out


@dataclass(frozen=True)
class GeometrySeriesRun:
    """Everything decided before a drop is dispensed."""

    plan_id: str = ""
    cast_order: tuple[int, ...] = ()
    measure_order: tuple[int, ...] = ()
    drift_channel: int | None = None
    volumes_by_channel: dict[int, list[float]] = field(default_factory=dict)
    levels_by_channel: dict[int, float] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.cast_order)

    def describe(self) -> str:
        levels = sorted({v for v in self.levels_by_channel.values()})
        drift = (f", drift control on ch{self.drift_channel}"
                 if self.drift_channel is not None else ", NO drift control")
        return (
            f"geometry series '{self.plan_id}': {self.n_channels} channels over "
            f"{len(levels)} levels ({', '.join(f'{v:g}' for v in levels)} um)"
            f"{drift}; cast order "
            + " ".join(str(c) for c in self.cast_order[:8])
            + (" ..." if self.n_channels > 8 else "")
        )


def plan_geometry_series_run(
    plan: Any,
    *,
    volumes_by_level: Mapping[float, Sequence[float]],
    occupied: Sequence[int] | set[int] = (),
    drift_control: bool = True,
) -> GeometrySeriesRun:
    """Resolve a :class:`~softae.core.thickness_series.ThicknessPlan` into an order.

    Raises :class:`GeometrySeriesError` rather than degrading: an occupied channel or a
    level with no solved volume both invalidate the design.
    """
    assignment = {int(k): float(v) for k, v in (getattr(plan, "assignment", {}) or {}).items()}
    if not assignment:
        raise GeometrySeriesError("the plan assigns no channels")

    verify_channels_free(sorted(assignment), occupied)

    missing = sorted({v for v in assignment.values()} - set(volumes_by_level))
    if missing:
        raise GeometrySeriesError(
            "no solved volume for level(s) "
            + ", ".join(f"{v:g}" for v in missing)
            + " — solve every level before casting any of them, so an infeasible "
              "target is found before the board is spent, not halfway through it."
        )

    order = round_robin_by_level(assignment)
    drift = choose_drift_channel(assignment) if drift_control else None
    return GeometrySeriesRun(
        plan_id=str(getattr(plan, "plan_id", "") or ""),
        cast_order=tuple(order),
        measure_order=tuple(order),
        drift_channel=drift,
        volumes_by_channel={ch: list(volumes_by_level[assignment[ch]]) for ch in order},
        levels_by_channel=assignment,
    )


def build_geometry_series_workflow(
    run: GeometrySeriesRun,
    spec: Any,
    *,
    catalog: Any,
    name: str | None = None,
) -> Any:
    """Cast and measure the whole series in one workflow.

    Goes through the **same** ``_build_deposition_workflow`` the campaign and HT paths
    use — P2's one-engine rule. A geometry series differs from a campaign round only in
    where its channels and volumes come from, so a second deposition path would be two
    implementations of one contract, free to diverge in exactly the way that produced
    three marshallers before P2.2 collapsed them.
    """
    from softae.core.autonomous_wiring import _build_deposition_workflow
    from softae.core.deposition_steps import eis_measure_step

    channels = list(run.cast_order)
    wf = _build_deposition_workflow(
        spec, dict(run.volumes_by_channel), catalog=catalog, channels=channels,
    )

    # The drift control: the same channel measured once more, last. Tagged with a
    # distinct role so the geometry fit's `role = 'sample'` filter excludes it —
    # otherwise, being the most recent spectrum for that channel, it would REPLACE the
    # in-sequence measurement and move that member to the wrong point in time.
    if run.drift_channel is not None:
        from softae.analysis.eis.calibration import DRIFT_REPEAT_ROLE

        step = eis_measure_step(
            run.drift_channel, name=f"geom_drift_repeat_ch{run.drift_channel}")
        wf.setup = list(wf.setup) + [
            type(step)(
                name=step.name, instrument=step.instrument, method=step.method,
                params=dict(step.params), timeout_s=step.timeout_s, retry=step.retry,
                tags={**step.tags, "role": DRIFT_REPEAT_ROLE,
                      "geometry_series": run.plan_id},
            )
        ]

    wf.name = name or f"geometry_series_{run.plan_id or 'unnamed'}"
    wf.metadata = {
        **(wf.metadata or {}),
        "source": "geometry_series",
        "plan_id": run.plan_id,
        "cast_order": list(run.cast_order),
        "drift_channel": run.drift_channel,
        "levels_by_channel": {str(k): v for k, v in run.levels_by_channel.items()},
    }
    logger.info("geometry_series_workflow_built", summary=run.describe())
    return wf
