"""Where should the next sweep put its points?

Every EIS sweep on this rig is one log-spaced ``meas_loop_eis`` block chosen from
four named presets, and the bottom of that grid is where all the time goes: for
``Longest`` (39 points, 0.228 Hz, 516 s) the single lowest point is 30 % of the
sweep and the lowest five are 84 % of it. A preset that reaches low enough for a dry
sample spends most of that time in a decade where nothing is being measured.

**This module contains no detector.** :mod:`softae.analysis.eis.arc` already locates
the arc — it sorts internally, reports the apex, discriminates an interior peak from
a floor-pinned one with a shape test, and is validated on 1440 spectra. A second
search over the same arrays would be the fourth in this tree, and the third one
(``circuit_fitting._local_minima``, whose window comparison admits its own edge) is
how the ``|Z|`` minimum came to be read as the interior ``−Z″`` valley — wrong by
more than an order of magnitude, twice, and into a published comparison. So what
lives here is the *acquisition decision* and the *segment layout*: policy over
numbers somebody else measured.

**The cut is this module's own, and it points the other way from the guard's.**
``arc_closure`` is biased toward CLOSED because a missed warning is cheaper than a
false one. Planning reverses that: a sweep planned one preset too wide costs
19.7–102.9 s of instrument time, while one planned too narrow costs a well *and* an
R₁ biased 60.9 % high at the two-point read and 175.2 % with the full CPE fit —
biased, not scattered, so it flows into a campaign objective looking like data.
Hence :data:`DEFAULT_BAND_BELOW_APEX_MIN_DECADES` sits well toward OPEN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from softae.analysis.eis.arc import CLOSED, UNKNOWN, arc_closure

logger = structlog.get_logger(__name__)

#: What the scout can conclude about one spectrum.
SCOUT_VERDICTS = ("ok", "extend_low", "extend_high", "no_arc", "no_data")

#: Decades of band that must sit below the apex before a plan is built on it.
#:
#: **1.0, not the source artifact's 0.4.** The 60.9 % median overestimate was
#: measured at only 1.5× past the apex — which is 0.176 decades — so 0.4 decades is
#: barely outside the regime the measurement condemns. The artifact's own selection
#: rule already uses 1.0.
DEFAULT_BAND_BELOW_APEX_MIN_DECADES = 1.0

#: Prominence, relative to the candidate's own height, below which a rise is read as
#: a shoulder rather than an arc. Scaled to the candidate and never to the
#: spectrum's global maximum, which a blocking tail owns: on a 15-well humid series
#: global scaling found 1 arc of 15, relative scaling found 12.
DEFAULT_APEX_PROMINENCE_MIN = 0.10


@dataclass(frozen=True)
class SegmentLayout:
    """The shape of a piecewise sweep, in decades relative to the apex.

    ``arc_points_per_decade`` rather than a fixed arc point count: points per decade
    is what the circle fit cares about, it keeps cost proportional to the band being
    covered, and it lets ``max_total_s`` shrink a plan smoothly instead of by
    chopping the band. A fixed count made the 65 Hz case 1.7× *dearer* than the
    preset it replaced, by putting several points below 20 Hz at ~1.7 s each.
    """

    f_hi_hz: float = 200_000.0
    f_floor_hz: float = 0.016
    hf_points: int = 10
    arc_points_per_decade: float = 12.0
    arc_decades_above: float = 1.0
    arc_decades_below: float = 1.0
    tail_points: int = 0
    tail_decades: float = 0.0
    #: 0 = uncapped. Otherwise the modelled per-channel cost the plan must fit in.
    max_total_s: float = 0.0


@dataclass(frozen=True)
class ScoutSettings:
    """``[eis.scout]``, plus the two bounds that resolve from ``[eis.instrument]``.

    Two flags, the house pattern: ``enabled`` runs the decision and logs the plan it
    *would* have used; ``actuate`` lets that plan drive an acquisition. They are
    separate so verdicts can accumulate against real runs at zero hardware cost,
    exactly as ``[eis.gates] enabled = false`` accumulates gate verdicts.
    """

    enabled: bool = False
    actuate: bool = False
    #: Default for the Manual Control tab's own checkbox, and *only* its default —
    #: that tab's control is authoritative there, so ``actuate`` alone can never
    #: switch it on. Manual measurements are where non-standard samples turn up:
    #: a two-arc system, a stack, something nobody has characterised. Whether the
    #: thing on the board is standard enough to plan a sweep around is a judgement
    #: made by a person at the rig, not a deployment setting, which is why the
    #: checkbox is the control and this is only what it starts at.
    actuate_manual: bool = False
    band_below_apex_min_decades: float = DEFAULT_BAND_BELOW_APEX_MIN_DECADES
    apex_prominence_min: float = DEFAULT_APEX_PROMINENCE_MIN
    layout: SegmentLayout = field(default_factory=SegmentLayout)

    def describe(self) -> str:
        """One line an operator can sanity-check the settings against."""
        if not self.enabled:
            return "EIS scout off — every sweep runs on its preset."
        if not self.actuate:
            return (
                "EIS scout observing — the plan is logged, the preset still runs."
            )
        return (
            f"EIS scout planning: ≥ {self.band_below_apex_min_decades:.1f} decades "
            f"below the apex, prominence ≥ {self.apex_prominence_min:.2f}."
        )


@dataclass(frozen=True)
class ScoutDecision:
    """What the scout concluded, and the three numbers it concluded it from.

    ``f_apex_hz`` is carried **only when an interior apex was actually measured by
    ``arc_closure`` and cleared the prominence cut** — on ``ok``, and on the one
    ``extend_low`` where the arc was found but too little band sits under it. It is
    NaN on ``no_arc``, ``no_data``, ``extend_high``, on a failed prominence cut, and
    on the ``extend_low`` where no interior apex exists at all.

    That withholding is the load-bearing part: it makes it structurally impossible
    to lay a sweep around an apex that was never observed or never qualified. Note
    what the two carrying cases have in common and the withheld ones do not — the
    number came from a measurement. It is never reconstructed here, and a caller
    that cannot get it from this field must widen blindly rather than infer it.
    """

    verdict: str
    f_apex_hz: float = float("nan")
    band_below_apex_decades: float = float("nan")
    apex_prominence_rel: float = float("nan")
    #: The guard's own verdict, passed through unmodified.
    arc_state: str = UNKNOWN

    @property
    def plannable(self) -> bool:
        """Is there an observed apex to lay a sweep around?

        Defined by the *apex*, not by the verdict, because that is the question a
        planner is actually asking. The verdict answers a different one — whether
        the sweep already taken was good enough — and the two part company on
        exactly one case: an arc that was found, but with too little band under it.
        """
        return bool(np.isfinite(self.f_apex_hz))


def scout_settings(config: dict[str, Any] | None = None) -> ScoutSettings:
    """Read ``[eis.scout]`` — the single parse point, in ``eis_settings``' posture.

    *config* is the ``[eis]`` table. Unknown keys are ignored and an unparseable
    value falls back to its default with a warning rather than raising: a typo in a
    config file must not stop a run that would otherwise have proceeded on presets.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("eis", {}) or {}
        except Exception:
            config = {}

    scout_cfg = config.get("scout", {}) or {}
    instrument_cfg = config.get("instrument", {}) or {}

    def _num(cfg: dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(cfg.get(key, default))
        except (TypeError, ValueError):
            logger.warning("eis_scout_config_unparseable", key=key, default=default)
            return default
        return value if np.isfinite(value) else default

    def _int(cfg: dict[str, Any], key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            logger.warning("eis_scout_config_unparseable", key=key, default=default)
            return default

    defaults = SegmentLayout()
    return ScoutSettings(
        enabled=bool(scout_cfg.get("enabled", False)),
        actuate=bool(scout_cfg.get("actuate", False)),
        actuate_manual=bool(scout_cfg.get("actuate_manual", False)),
        band_below_apex_min_decades=_num(
            scout_cfg, "band_below_apex_min_decades",
            DEFAULT_BAND_BELOW_APEX_MIN_DECADES),
        apex_prominence_min=_num(
            scout_cfg, "apex_prominence_min", DEFAULT_APEX_PROMINENCE_MIN),
        layout=SegmentLayout(
            # Not duplicated under [eis.scout]: [eis.instrument] is the authority
            # on what this instrument can reach.
            f_hi_hz=_num(instrument_cfg, "f_max_hz", defaults.f_hi_hz),
            f_floor_hz=_num(instrument_cfg, "f_min_hz", defaults.f_floor_hz),
            hf_points=_int(scout_cfg, "hf_points", defaults.hf_points),
            arc_points_per_decade=_num(
                scout_cfg, "arc_points_per_decade", defaults.arc_points_per_decade),
            arc_decades_above=_num(
                scout_cfg, "arc_decades_above", defaults.arc_decades_above),
            arc_decades_below=_num(
                scout_cfg, "arc_decades_below", defaults.arc_decades_below),
            tail_points=_int(scout_cfg, "tail_points", defaults.tail_points),
            tail_decades=_num(scout_cfg, "tail_decades", defaults.tail_decades),
            max_total_s=_num(scout_cfg, "max_total_s", defaults.max_total_s),
        ),
    )


def scout_decision(
    frequency: Any,
    z_imag_neg: Any,
    phase: Any = None,
    *,
    settings: ScoutSettings | None = None,
) -> ScoutDecision:
    """Decide what the *next* sweep on this channel should reach for.

    **Never raises.** It is called from a measurement path, and an acquisition
    planner that can crash a running sweep is worse than one that declines to plan.
    """
    settings = settings or ScoutSettings()
    try:
        f = np.asarray(frequency, dtype=float).ravel()
        arc = arc_closure(f, z_imag_neg, phase)

        if arc.state == UNKNOWN:
            return ScoutDecision("no_data", arc_state=arc.state)

        apex = float(arc.f_apex_interior_hz)
        band = float(arc.band_below_apex_decades)
        prominence = float(arc.apex_prominence_rel)
        common = dict(band_below_apex_decades=band, apex_prominence_rel=prominence,
                      arc_state=arc.state)

        if not np.isfinite(apex):
            # No interior peak at all. OPEN means the arc is below the floor and a
            # wider sweep would find it; CLOSED here is the lone-excursion rescue,
            # where the reported peak came from the shape test rather than a shape.
            return ScoutDecision(
                "no_arc" if arc.state == CLOSED else "extend_low", **common)
        if not np.isfinite(prominence) or prominence < settings.apex_prominence_min:
            return ScoutDecision("no_arc", **common)
        if _is_at_the_band_top(apex, f):
            return ScoutDecision("extend_high", **common)
        if not np.isfinite(band) or band < settings.band_below_apex_min_decades:
            # The apex IS carried here, and this is the only non-``ok`` verdict
            # that carries one. By this line it has survived both guards above:
            # ``arc_closure`` measured an interior maximum, and that maximum
            # cleared the prominence cut. Nothing about it is unobserved — the
            # sweep simply stopped too soon under it, which is precisely the case
            # a piecewise grid answers better than any preset can, because the
            # band to cover is known rather than guessed at.
            return ScoutDecision("extend_low", apex, band, prominence, arc.state)
        return ScoutDecision("ok", apex, band, prominence, arc.state)
    except Exception as exc:                                   # pragma: no cover
        logger.warning("eis_scout_decision_failed", error=str(exc))
        return ScoutDecision("no_data")


def plan_segments(
    f_apex_hz: float, layout: SegmentLayout | None = None,
) -> tuple[tuple[float, float, int], ...]:
    """Lay a piecewise sweep around *f_apex_hz*. Pure, and reads no config.

    Three bands, descending, adjacent — an HF limb where every point is at the
    instrument's per-point floor, the arc itself, and an optional tail. Adjacent
    segments deliberately *share* their boundary here;
    :func:`~softae.drivers.mscr_library.resolve_segments` is what nudges them apart,
    so the duplicate-point rule lives in one place rather than two.

    A band clamped out of existence by the instrument's limits is dropped rather
    than emitted degenerate, so the result can be shorter than three segments.
    """
    layout = layout or SegmentLayout()
    apex = float(f_apex_hz)
    if not np.isfinite(apex) or apex <= 0.0:
        return ()

    below = float(layout.arc_decades_below)
    plan = _lay_out(apex, layout, below)
    cap = float(layout.max_total_s)
    if cap > 0.0 and plan:
        while _plan_cost_s(plan) > cap and below > 0.1:
            below = round(below - 0.1, 6)
            plan = _lay_out(apex, layout, below)
        if below != layout.arc_decades_below:
            logger.warning(
                "eis_scout_plan_trimmed",
                requested_decades_below=float(layout.arc_decades_below),
                granted_decades_below=below,
                max_total_s=cap,
                modelled_s=_plan_cost_s(plan),
                msg="arc band narrowed to fit the time cap",
            )
    return plan


def plan_for(
    decision: ScoutDecision, settings: ScoutSettings | None = None,
) -> tuple[tuple[float, float, int], ...]:
    """The plan *decision*'s apex licenses — empty when no apex was observed.

    Empty is the answer for ``no_arc``, ``no_data``, ``extend_high`` and for the
    ``extend_low`` with no interior maximum at all: a band laid below an apex
    nobody has seen is exactly the guess this module refuses to make, and widening
    blindly by one preset step is the caller's answer there. It is *not* the answer
    for an arc that was found with too little room under it — that apex is a
    measurement, and planning around it is the point of the exercise.

    Whether the caller should *act* on the plan is a separate question this does
    not answer: an ``ok`` verdict is plannable and yet means the sweep in hand is
    already good enough to keep.
    """
    settings = settings or ScoutSettings()
    if not decision.plannable:
        return ()
    return plan_segments(decision.f_apex_hz, settings.layout)


# ── Internals ─────────────────────────────────────────────────────────────────

def _is_at_the_band_top(apex_hz: float, f: np.ndarray) -> bool:
    """Is the apex within one grid step of the top of what was swept?

    Then the real peak may sit above the sweep and 200 kHz is the hardware ceiling,
    so the honest verdict is ``extend_high`` rather than a plan whose HF limb has
    nowhere to go.
    """
    if f.size < 2:
        return False
    f_high, f_low = float(np.max(f)), float(np.min(f))
    if not (f_high > 0.0 and f_low > 0.0 and f_high > f_low):
        return False
    step = (np.log10(f_high) - np.log10(f_low)) / (f.size - 1)
    return bool(np.log10(f_high / apex_hz) <= step * (1.0 + 1e-9))


def _lay_out(
    apex: float, layout: SegmentLayout, decades_below: float,
) -> tuple[tuple[float, float, int], ...]:
    """The three bands at one trial value of *decades_below*."""
    from softae.drivers.mscr_library import quantize_hz

    def _clamp(f_hz: float) -> float:
        return quantize_hz(
            min(max(f_hz, float(layout.f_floor_hz)), float(layout.f_hi_hz)))

    top = _clamp(layout.f_hi_hz)
    arc_top = _clamp(apex * 10.0 ** float(layout.arc_decades_above))
    arc_bottom = _clamp(apex / 10.0 ** decades_below)
    tail_bottom = _clamp(arc_bottom / 10.0 ** float(layout.tail_decades))

    per_decade = float(layout.arc_points_per_decade)
    arc_points = max(2, int(round(per_decade * _decades(arc_top, arc_bottom))))

    candidates = [
        (top, arc_top, int(layout.hf_points)),
        (arc_top, arc_bottom, arc_points),
    ]
    if int(layout.tail_points) > 0 and float(layout.tail_decades) > 0.0:
        candidates.append((arc_bottom, tail_bottom, int(layout.tail_points)))

    return tuple((a, b, n) for a, b, n in candidates if a > b and n >= 1)


def _decades(f_start: float, f_end: float) -> float:
    return float(np.log10(f_start / f_end)) if f_start > f_end > 0.0 else 0.0


def _plan_cost_s(segments: tuple[tuple[float, float, int], ...]) -> float:
    """Modelled per-channel cost, priced by the shipped model and nothing else.

    ``core/preflight.py`` owns the cost model — 34.5 cycles per point over a 0.131 s
    floor, refitted against four stopwatched presets. The source artifact carries its
    own pair (35.5 / 0.010 s), which puts the knee at ~3.5 kHz instead of this rig's
    ~264 Hz and would make most of an HF limb look free. No second model is
    introduced here and those constants are transcribed nowhere.
    """
    from softae.core.preflight import model_eis_duration

    return sum(
        model_eis_duration(_GridShim(f_hi=f_start, f_lo_mHz=f_end * 1000.0, npts=npts))
        for f_start, f_end, npts in segments
    )


@dataclass(frozen=True)
class _GridShim:
    """One segment in the shape ``model_eis_duration`` reads."""

    f_hi: float
    f_lo_mHz: float
    npts: int


__all__ = [
    "SCOUT_VERDICTS",
    "ScoutDecision",
    "ScoutSettings",
    "SegmentLayout",
    "plan_for",
    "plan_segments",
    "scout_decision",
    "scout_settings",
]
