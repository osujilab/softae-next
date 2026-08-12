"""Planning and checking an *unconfounded* thickness series.

Overhaul F12 is the failure this module exists to prevent, and it is worth stating
precisely because it is not a data-handling mistake — it is a design mistake that no
amount of later analysis can undo. A geometry series was cast as::

    CH27, CH28 -> 200 um     CH29, CH30 -> 150 um     CH31, CH32 -> 100 um

Thickness and channel index move together. Every channel has its own mux path, its own
trace length, its own contact — so a systematic difference *between channels* and a
genuine *thickness effect* produce the identical signature. The two are not merely hard
to separate; they are mathematically indistinguishable in that dataset. The series
cannot answer the question it was cast to answer.

**The confounding happened at cast time, not at measurement time.** A harness that only
recorded measured thicknesses would faithfully record a second confounded series and
report nothing wrong. So the assignment of level to channel is *planned here, before
casting*, and what was actually cast is checked against what was planned.

Two properties are enforced, and they are different requirements:

``balance``
    Every level gets the same number of channels (within one). Without it the
    lightly-sampled levels dominate the slope's uncertainty.
``decorrelation``
    ``|corr(channel index, level)|`` below a threshold. This is the F12 property
    specifically, and randomisation alone does **not** guarantee it — a uniformly random
    permutation can land on a strongly ordered arrangement by chance, and with 32
    channels and 4 levels that is not rare enough to ignore. So candidate assignments
    are drawn until one satisfies the bound, and the achieved correlation is recorded
    with the plan rather than assumed.

Systematic interleaving (``L1 L2 L3 L4 L1 L2 …``) would give exactly zero linear
correlation and is deliberately *not* used: it trades one confound for another, aligning
level with any periodic spatial effect across the board. Randomisation under a
correlation constraint keeps the guarantee without importing a new structure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import structlog

logger = structlog.get_logger(__name__)

#: Largest |Pearson r| between channel index and thickness level an assignment may have.
#:
#: 0.2 is loose enough to be reachable by rejection sampling with a handful of channels
#: and tight enough that a channel-linear artifact cannot masquerade as a thickness
#: effect. For reference, the F12 assignment above scores **r = -0.956**; a balanced
#: 4-level plan across 32 channels typically reaches |r| < 0.05 on the first draw.
DEFAULT_MAX_CORRELATION = 0.2

#: Attempts before giving up on the correlation bound. With realistic channel counts a
#: valid draw usually appears within a few tries; the ceiling exists so a degenerate
#: request (two channels, two levels — where every arrangement is perfectly correlated)
#: fails with an explanation instead of spinning.
DEFAULT_MAX_DRAWS = 2000

#: Fewest channels per level. One channel per level leaves no within-level variance, so
#: the slope gets no error bar and an outlier cannot be identified as one.
MIN_REPLICATES = 2


class ThicknessPlanError(ValueError):
    """A series that cannot be planned as requested — with the reason."""


def correlation(channels: Sequence[int], levels: Sequence[float]) -> float:
    """Pearson ``r`` between channel index and assigned level.

    This single number is the F12 detector. ``|r| → 1`` means level and channel vary
    together and the design cannot separate them.
    """
    n = len(channels)
    if n < 2 or len(levels) != n:
        return float("nan")
    cx = [float(c) for c in channels]
    cy = [float(v) for v in levels]
    mx, my = sum(cx) / n, sum(cy) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(cx, cy))
    sxx = sum((a - mx) ** 2 for a in cx)
    syy = sum((b - my) ** 2 for b in cy)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class ThicknessPlan:
    """A level→channel assignment, with the evidence that it is unconfounded."""

    plan_id: str = ""
    created_at: str = ""
    levels_um: tuple[float, ...] = ()
    #: channel → target thickness (µm). The order casting should follow.
    assignment: dict[int, float] = field(default_factory=dict)
    seed: int = 0
    achieved_correlation: float = float("nan")
    max_correlation: float = DEFAULT_MAX_CORRELATION
    draws: int = 0
    notes: str = ""

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(sorted(self.assignment))

    @property
    def replicates(self) -> dict[float, int]:
        out: dict[float, int] = {}
        for level in self.assignment.values():
            out[level] = out.get(level, 0) + 1
        return out

    def is_adequate_for_geometry_series(self) -> tuple[bool, str]:
        """Whether E5 could actually be answered from this design.

        Separate from "is it unconfounded". A perfectly decorrelated two-level series
        is still not a geometry series — §5.6 wants ≥4 levels spanning ≥2×, because a
        slope through two points has no residual and therefore no way to show that the
        linear model is wrong.
        """
        issues: list[str] = []
        n_levels = len(set(self.assignment.values()))
        if n_levels < 4:
            issues.append(f"{n_levels} levels (need >= 4)")
        if self.levels_um:
            span = max(self.levels_um) / min(self.levels_um)
            if span < 2.0:
                issues.append(f"span {span:.2f}x (need >= 2x)")
        thin = [lv for lv, n in self.replicates.items() if n < MIN_REPLICATES]
        if thin:
            issues.append(f"levels with < {MIN_REPLICATES} replicates: {sorted(thin)}")
        if not issues:
            return True, "adequate for a geometry series"
        return False, "; ".join(issues)

    def describe(self) -> str:
        ok, why = self.is_adequate_for_geometry_series()
        return (
            f"plan {self.plan_id or '(unsaved)'}: {len(self.assignment)} channel(s), "
            f"{len(set(self.assignment.values()))} level(s), "
            f"|r| = {abs(self.achieved_correlation):.3f} "
            f"(bound {self.max_correlation:g}, {self.draws} draw(s)) — {why}"
        )

    def as_rows(self) -> list[tuple[int, float]]:
        """``(channel, level_um)`` in channel order — the cast list."""
        return [(c, self.assignment[c]) for c in self.channels]

    def to_json(self) -> str:
        return json.dumps({
            "plan_id": self.plan_id, "created_at": self.created_at,
            "levels_um": list(self.levels_um), "seed": self.seed,
            "assignment": {str(k): v for k, v in self.assignment.items()},
            "achieved_correlation": self.achieved_correlation,
            "max_correlation": self.max_correlation,
            "draws": self.draws, "notes": self.notes,
        }, default=str)

    @classmethod
    def from_json(cls, text: str) -> "ThicknessPlan":
        d = json.loads(text)
        return cls(
            plan_id=str(d.get("plan_id", "")),
            created_at=str(d.get("created_at", "")),
            levels_um=tuple(float(x) for x in d.get("levels_um", [])),
            assignment={int(k): float(v)
                        for k, v in (d.get("assignment") or {}).items()},
            seed=int(d.get("seed", 0)),
            achieved_correlation=float(d.get("achieved_correlation", float("nan"))),
            max_correlation=float(d.get("max_correlation", DEFAULT_MAX_CORRELATION)),
            draws=int(d.get("draws", 0)),
            notes=str(d.get("notes", "")),
        )


def plan_series(
    levels_um: Sequence[float],
    channels: Sequence[int],
    *,
    seed: int = 0,
    max_correlation: float = DEFAULT_MAX_CORRELATION,
    max_draws: int = DEFAULT_MAX_DRAWS,
    created_at: str = "",
    plan_id: str = "",
    notes: str = "",
) -> ThicknessPlan:
    """Assign thickness levels to channels so the two are not confounded.

    Balanced by construction (levels are dealt round-robin, then shuffled), then
    rejection-sampled until ``|corr(channel, level)| <= max_correlation``.

    Raises :class:`ThicknessPlanError` rather than returning a confounded plan. A plan
    that quietly failed its own constraint would be worse than no plan, because the
    whole point is to be able to say afterwards that the design was sound.
    """
    import random

    levels = [float(x) for x in levels_um]
    chans = [int(c) for c in channels]
    if len(set(levels)) < 2:
        raise ThicknessPlanError("need at least two distinct thickness levels")
    if len(chans) < len(set(levels)):
        raise ThicknessPlanError(
            f"{len(chans)} channel(s) cannot carry {len(set(levels))} level(s)")
    if len(set(chans)) != len(chans):
        raise ThicknessPlanError("duplicate channels in the request")

    # Balanced multiset: deal levels round-robin so counts differ by at most one.
    deck = [levels[i % len(levels)] for i in range(len(chans))]

    rng = random.Random(seed)
    ordered = sorted(chans)
    best: tuple[float, list[float]] | None = None

    for attempt in range(1, int(max_draws) + 1):
        rng.shuffle(deck)
        r = correlation(ordered, deck)
        score = abs(r) if r == r else 1.0
        if best is None or score < best[0]:
            best = (score, list(deck))
        if score <= max_correlation:
            plan = ThicknessPlan(
                plan_id=plan_id, created_at=created_at,
                levels_um=tuple(sorted(set(levels))),
                assignment=dict(zip(ordered, deck)),
                seed=seed, achieved_correlation=r,
                max_correlation=max_correlation, draws=attempt, notes=notes,
            )
            logger.info("thickness_plan_created", channels=len(ordered),
                        levels=len(set(levels)), correlation=r, draws=attempt)
            return plan

    achieved = best[0] if best else float("nan")
    raise ThicknessPlanError(
        f"no assignment of {len(set(levels))} level(s) across {len(chans)} channel(s) "
        f"reached |r| <= {max_correlation:g} in {max_draws} draws (best {achieved:.3f}). "
        f"With few channels every arrangement is correlated — add channels, or relax "
        f"--max-correlation deliberately and record that you did."
    )


#: Fewest cast channels before a correlation means anything. Below this the statistic
#: is dominated by which handful happened to be measured first.
MIN_CHANNELS_TO_JUDGE = 4


@dataclass(frozen=True)
class ConfoundReport:
    """Whether an *as-cast* series can answer a thickness question.

    Three verdicts, not two. ``indeterminate`` is separate from ``confounded`` because
    a series being cast one channel at a time passes through a stage where too little
    exists to judge — and reporting that as a design failure would train an operator to
    ignore the one message that must never be ignored.
    """

    n_channels: int = 0
    n_levels: int = 0
    correlation: float = float("nan")
    max_correlation: float = DEFAULT_MAX_CORRELATION
    replicates: dict[float, int] = field(default_factory=dict)
    matches_plan: bool | None = None
    #: Channels cast at a level other than the planned one — the real F12 risk.
    deviations: tuple[str, ...] = ()
    #: Channels in the plan with nothing recorded yet. Normal mid-cast, not a fault.
    pending: tuple[int, ...] = ()

    @property
    def verdict(self) -> str:
        """``"ok"`` | ``"indeterminate"`` | ``"confounded"``."""
        if self.n_channels < MIN_CHANNELS_TO_JUDGE or self.n_levels < 2:
            return "indeterminate"
        r = self.correlation
        if not (r == r):
            return "indeterminate"
        return "confounded" if abs(r) > self.max_correlation else "ok"

    @property
    def confounded(self) -> bool:
        """Strictly the bad case — *judged, and found correlated*."""
        return self.verdict == "confounded"

    @property
    def certified(self) -> bool:
        """Only ``ok`` certifies. Indeterminate is not a pass."""
        return self.verdict == "ok"

    def describe(self) -> str:
        r = self.correlation
        head = f"{self.n_channels} channel(s), {self.n_levels} level(s)"
        if r == r:
            head += f", |r| = {abs(r):.3f}"

        verdict = self.verdict
        if verdict == "confounded":
            head += (f" — CONFOUNDED (bound {self.max_correlation:g}): a channel-linear "
                     f"artifact and a thickness effect are indistinguishable here")
        elif verdict == "indeterminate":
            head += (f" — too little cast to judge yet (need >= "
                     f"{MIN_CHANNELS_TO_JUDGE} channels across >= 2 levels)")
        else:
            head += f" — within the {self.max_correlation:g} bound"

        if self.deviations:
            head += f"; {len(self.deviations)} deviation(s) from plan"
        if self.pending:
            head += f"; {len(self.pending)} channel(s) not yet cast"
        return head


def detect_confounding(
    channels: Sequence[int],
    levels_um: Sequence[float],
    *,
    max_correlation: float = DEFAULT_MAX_CORRELATION,
    plan: ThicknessPlan | None = None,
) -> ConfoundReport:
    """Check an as-cast (or as-measured) series for the F12 defect.

    Supply *plan* to also verify that what was cast is what was planned — a sound plan
    followed inattentively produces exactly the dataset the plan existed to prevent.
    """
    chans = [int(c) for c in channels]
    levels = [float(v) for v in levels_um]
    reps: dict[float, int] = {}
    for v in levels:
        reps[v] = reps.get(v, 0) + 1

    deviations: list[str] = []
    pending: list[int] = []
    matches: bool | None = None
    if plan is not None:
        matches = True
        for ch, lv in zip(chans, levels):
            want = plan.assignment.get(ch)
            if want is None:
                deviations.append(f"ch{ch} was cast but is not in the plan")
                matches = False
            elif abs(want - lv) > 1e-9:
                deviations.append(f"ch{ch}: planned {want:g} um, cast {lv:g} um")
                matches = False
        # A channel with nothing recorded is *pending*, not deviant. A series is cast
        # and measured one channel at a time, so treating "not yet" as a fault would
        # make the report cry wolf through the whole middle of every campaign.
        pending = sorted(ch for ch in plan.assignment if ch not in chans)

    report = ConfoundReport(
        n_channels=len(chans), n_levels=len(set(levels)),
        correlation=correlation(chans, levels),
        max_correlation=max_correlation, replicates=reps,
        matches_plan=matches, deviations=tuple(deviations),
        pending=tuple(pending),
    )
    if report.confounded:
        logger.warning("thickness_series_confounded",
                       correlation=report.correlation,
                       n_channels=report.n_channels, n_levels=report.n_levels,
                       msg="level tracks channel — see overhaul F12")
    return report
