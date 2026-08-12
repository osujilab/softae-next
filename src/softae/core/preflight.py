"""Can this campaign finish — on the stock on hand, and in what time? (P5.2)

Two questions an operator has to answer before walking away, both of which the
platform could previously only answer by running the campaign and finding out.

**Stock.** Every dispense now debits a ledger and a depleted reservoir parks the
run (P5.1), which is correct but late: discovering at iteration 40 that there was
never enough stock wastes a board and a night. Projecting the per-iteration draw
against declared levels answers it up front.

**Duration.** Reported as a **rate, not an ETA**. A Bayesian campaign stops on a
convergence criterion, not a known iteration count, so "this will finish at
14:20" would be a fabrication. What *is* well-determined is the per-iteration
wall-clock — deposit, settle, anneal, measure are all specified — so the honest
presentation is: time per iteration, time to the configured budget framed as an
**upper** bound, and the stock/waste runway in the same units. With purging
active (P8) the runway usually binds long before the budget does.

Every estimate here is a **lower bound on duration and a lower bound on draw**:
it counts the dwells the workflow declares and ignores comms overhead, stage
travel, and ramp time. Stated plainly rather than padded with a fudge factor,
because an operator can reason about "at least this long" and cannot reason about
an unexplained multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Periods measured per frequency point in an EIS sweep. The low-frequency end
#: dominates the sweep time; this is the multiplier on 1/f for each point.
EIS_CYCLES_PER_POINT = 3.0
#: Floor on per-point time, covering instrument overhead at high frequency.
EIS_MIN_POINT_S = 0.05


@dataclass
class DurationEstimate:
    """How long one workflow takes, and how much of that is actually known."""

    total_s: float = 0.0
    n_steps: int = 0
    n_unknown: int = 0
    by_step: dict[str, float] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True when every step contributed an estimate."""
        return self.n_unknown == 0


def _f(params: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def _extrusion_s(vols, rates) -> float:
    """Time for a proportional multi-pump extrusion (µL / (µL/min) → s).

    Rates are split so all components extrude for the *same* duration, so the
    slowest pump sets the time; ``max`` rather than ``sum`` is the correct
    reduction.
    """
    try:
        pairs = [
            (float(v), float(r))
            for v, r in zip(list(vols or []), list(rates or []))
            if float(r) > 0 and float(v) > 0
        ]
    except (TypeError, ValueError):
        return 0.0
    return max((v / r * 60.0 for v, r in pairs), default=0.0)


def estimate_eis_duration(eis_params: Any = None) -> float:
    """Estimated sweep time from the EIS parameters.

    The low-frequency end dominates: a point at 0.1 Hz costs ~30 s while one at
    100 kHz is instrument-limited. Points are assumed log-spaced between
    ``f_hi`` and ``f_lo``, each costing ``EIS_CYCLES_PER_POINT`` periods.
    """
    if eis_params is None:
        from softae.core.eis_scripts import EISParams

        eis_params = EISParams()

    try:
        f_hi = float(getattr(eis_params, "f_hi"))
        f_lo_hz = float(getattr(eis_params, "f_lo_mHz")) / 1000.0
        npts = int(getattr(eis_params, "npts"))
    except Exception:
        return 0.0
    if npts <= 0 or f_hi <= 0 or f_lo_hz <= 0:
        return 0.0

    import math

    if npts == 1:
        freqs = [f_lo_hz]
    else:
        step = (math.log10(f_hi) - math.log10(f_lo_hz)) / (npts - 1)
        freqs = [10 ** (math.log10(f_lo_hz) + i * step) for i in range(npts)]

    return sum(max(EIS_MIN_POINT_S, EIS_CYCLES_PER_POINT / f) for f in freqs)


def estimate_step_duration(step: Any, *, eis_params: Any = None) -> float | None:
    """Estimated wall-clock for one step, or ``None`` when it cannot be modelled.

    ``None`` is distinct from ``0.0``: an unmodelled step is *unknown* time, and
    counting it as free would understate the projection in a way that looks like
    precision.
    """
    params = dict(getattr(step, "params", {}) or {})
    method = str(getattr(step, "method", ""))
    scale = _f(params, "time_scale", 1.0)

    if method == "sendscript_getdata":
        return estimate_eis_duration(eis_params)

    if method == "single_pump":
        rate = _f(params, "rate")
        vol = _f(params, "dispense_vol")
        return (vol / rate * 60.0) if rate > 0 and vol > 0 else 0.0

    if method in ("startup_flush", "startup_flush_full", "final_flush"):
        vols = params.get("disp_vols") or [_f(params, "disp_vol")]
        rate = _f(params, "disp_rate")
        extrude = _extrusion_s(vols, [rate] * len(list(vols))) if rate > 0 else 0.0
        dwell = (_f(params, "post_flush_dwell_s") + _f(params, "wick_dwell_s")) * scale
        return extrude + dwell

    if method == "precondition_flush":
        # The preload is flush_factor × the deposit volume.
        vols = [v * _f(params, "flush_factor", 1.0) for v in params.get("vol_list") or []]
        extrude = _extrusion_s(vols, params.get("rate_list"))
        plug_rate, plug_vol = _f(params, "plug_rate"), _f(params, "plug_vol")
        plug = (plug_vol / plug_rate * 60.0) if plug_rate > 0 else 0.0
        return extrude + plug + _f(params, "wick_dwell_s") * scale

    if method in ("single_drop_simul", "alt_drop") or "drop" in method:
        extrude = _extrusion_s(params.get("vols"), params.get("disp_rates"))
        if extrude == 0.0:
            rate = _f(params, "disp_rate")
            vols = params.get("vols") or []
            extrude = _extrusion_s(vols, [rate] * len(list(vols))) if rate > 0 else 0.0
        dwell = (_f(params, "elution_wait_s") + _f(params, "wick_dwell_s")) * scale
        return extrude + dwell

    if method == "anneal":
        # The hold dominates; the ramp is not modelled, so this is a lower bound.
        return _f(params, "hold_time_s") * scale

    if method == "wait":
        return (_f(params, "duration_s") or _f(params, "seconds")) * scale

    return None


def estimate_workflow_duration(wf: Any, *, eis_params: Any = None) -> DurationEstimate:
    """Sum the per-step estimates for one trial workflow."""
    est = DurationEstimate()
    try:
        steps = list(wf.resolve_steps())
    except Exception:
        return est

    for step in steps:
        est.n_steps += 1
        value = estimate_step_duration(step, eis_params=eis_params)
        if value is None:
            est.n_unknown += 1
            continue
        name = str(getattr(step, "name", "step"))
        est.by_step[name] = float(value)
        est.total_s += float(value)
    return est


#: Methods that put fluid onto the board rather than into the waste container.
#: Everything else that dispenses does so at the flush basin or the wick.
_CAST_METHODS = frozenset({"single_drop_simul", "alt_drop"})


def step_pump_volumes(step: Any) -> dict[int, float]:
    """Per-pump volume one step commands (µL), keyed by pump id.

    The single traversal shared by the pre-run projection and the at-run waste
    accrual. Two classifiers over the same step shapes would be two things to
    keep in step with the engine; this way a new dispensing method is taught
    once.
    """
    volumes: dict[int, float] = {}

    def _add(pump_id: Any, volume: Any) -> None:
        try:
            pid, vol = int(pump_id), float(volume)
        except (TypeError, ValueError):
            return
        if vol > 0:
            volumes[pid] = volumes.get(pid, 0.0) + vol

    params = dict(getattr(step, "params", {}) or {})
    method = str(getattr(step, "method", ""))

    if method == "single_pump":
        _add(params.get("ID"), params.get("dispense_vol"))
        return volumes

    ids = params.get("ids")
    if not ids:
        return volumes
    factor = (
        _f(params, "flush_factor", 1.0) if method == "precondition_flush" else 1.0
    )
    vols = params.get("vols") or params.get("vol_list") or params.get("disp_vols")
    if vols:
        for pid, vol in zip(list(ids), list(vols)):
            _add(pid, _f({"v": vol}, "v") * factor)
    elif "disp_vol" in params:
        for pid in list(ids):
            _add(pid, _f(params, "disp_vol"))
    return volumes


def step_goes_to_waste(step: Any) -> bool:
    """Whether this step's fluid ends up in the waste container.

    The **phase tag is the primary signal**, not the method: ``single_pump`` is
    genuinely ambiguous — it is how the teardown flush dispenses *and* how the
    ``deposit_pumpN`` catalog tasks cast onto a board. Classifying by method
    alone would book every hand-built deposit as waste.

    Untagged steps fall back to the method, so workflows built outside the
    recipe engine still classify sensibly.
    """
    phase = (getattr(step, "tags", None) or {}).get("phase")
    if phase == "deposit":
        return False
    if phase:                      # precondition, flush, piezo, anneal…
        return True
    method = str(getattr(step, "method", ""))
    return not (method in _CAST_METHODS or "drop" in method)


def step_waste_uL(step: Any) -> float:
    """Volume this step sends to waste (0 for a cast)."""
    if not step_goes_to_waste(step):
        return 0.0
    return sum(step_pump_volumes(step).values())


def per_iteration_draw(wf: Any) -> dict[int, float]:
    """Per-pump stock consumed by one trial (µL), keyed by pump id.

    Read off the *built* workflow rather than recomputed, so it reflects the
    volumes the hardware will actually be commanded — including dead-volume
    correction, which is applied at the marshaller (P2.2).
    """
    draw: dict[int, float] = {}
    try:
        steps = list(wf.resolve_steps())
    except Exception:
        return draw

    for step in steps:
        for pid, vol in step_pump_volumes(step).items():
            draw[pid] = draw.get(pid, 0.0) + vol
    return draw


def waste_per_iteration_uL(wf: Any) -> float:
    """Waste one trial sends to the container (µL) — the projection's view."""
    try:
        steps = list(wf.resolve_steps())
    except Exception:
        return 0.0
    return sum(step_waste_uL(s) for s in steps)


@dataclass
class CampaignProjection:
    """What a campaign will cost in time and stock, with its limits named."""

    per_iteration_s: float
    per_iteration_draw_uL: dict[int, float]
    budget: int
    duration_complete: bool = True
    #: Declared stock per pump at projection time (``None`` = unmanaged).
    stock_uL: dict[int, float | None] = field(default_factory=dict)
    #: Idle/in-run purge consumption, once P8 exists. ``0`` until then.
    purge_uL_per_day: dict[int, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def time_to_budget_s(self) -> float:
        """Upper bound: convergence may stop the run sooner."""
        return self.per_iteration_s * max(0, int(self.budget))

    def iterations_supported(self) -> int | None:
        """Iterations the declared stock supports, or ``None`` if unmanaged.

        Takes the **minimum across pumps** — the first stock to run out stops the
        campaign, so an average would flatter it.
        """
        limits: list[int] = []
        for pid, per_iter in self.per_iteration_draw_uL.items():
            have = self.stock_uL.get(pid)
            if have is None or per_iter <= 0:
                continue
            purge_per_iter = (
                self.purge_uL_per_day.get(pid, 0.0)
                * (self.per_iteration_s / 86400.0)
            )
            limits.append(int(have // max(1e-9, per_iter + purge_per_iter)))
        return min(limits) if limits else None

    @property
    def stock_sufficient(self) -> bool | None:
        """``None`` when no stock is declared — unknown, not insufficient."""
        supported = self.iterations_supported()
        return None if supported is None else supported >= self.budget

    def describe(self) -> str:
        """Operator-facing summary: a rate with bounds, never a single ETA."""
        lines: list[str] = []
        per_it = _fmt_duration(self.per_iteration_s)
        lines.append(f"About {per_it} per iteration.")
        if not self.duration_complete:
            lines.append(
                "  (Some steps could not be timed, so this is a lower bound.)")

        lines.append(
            f"At most {_fmt_duration(self.time_to_budget_s)} to reach the "
            f"{self.budget}-iteration budget — convergence may stop it sooner."
        )

        draw_total = sum(self.per_iteration_draw_uL.values())
        if draw_total > 0:
            per_pump = ", ".join(
                f"pump {p} {v:.0f} µL"
                for p, v in sorted(self.per_iteration_draw_uL.items())
            )
            lines.append(f"Stock per iteration: {draw_total:.0f} µL ({per_pump}).")

        supported = self.iterations_supported()
        if supported is None:
            lines.append(
                "Stock runway unknown — no reservoir levels declared. "
                "Declare them in Syringe Stock to project it.")
        else:
            runway_s = supported * self.per_iteration_s
            verdict = "enough" if supported >= self.budget else "NOT enough"
            lines.append(
                f"Declared stock supports about {supported} iteration(s) "
                f"(~{_fmt_duration(runway_s)}) — {verdict} for the full budget."
            )

        lines.extend(f"Note: {w}" for w in self.warnings)
        return "\n".join(lines)


def _fmt_duration(seconds: float) -> str:
    """Human units — an operator plans in hours and days, not seconds."""
    s = max(0.0, float(seconds))
    if s < 90:
        return f"{s:.0f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 172800:          # switch at an hour, not 90 min: "60 min" reads
        return f"{s / 3600:.1f} h"   # worse than "1.0 h" at exactly one hour
    return f"{s / 86400:.1f} days"


def project_campaign(
    spec: Any,
    *,
    catalog: Any,
    ledger: Any = None,
    purge_uL_per_day: dict[int, float] | None = None,
) -> CampaignProjection:
    """Project one campaign's per-iteration time and stock draw.

    Builds a **representative trial** at the midpoint of the parameter space —
    a single trial is what a projection can honestly be based on, since the
    optimizer chooses the rest.
    """
    from softae.core.autonomous_wiring import build_trial_workflow
    from softae.core.eis_scripts import EISParams

    warnings: list[str] = []

    midpoint: dict[str, Any] = {}
    for name, p in (getattr(spec, "parameter_space", {}) or {}).items():
        try:
            if p.get("type") in ("float", "int"):
                midpoint[name] = (float(p["low"]) + float(p["high"])) / 2.0
            else:
                midpoint[name] = (p.get("choices") or [None])[0]
        except Exception:
            midpoint[name] = 0.0

    try:
        wf = build_trial_workflow(spec, midpoint, catalog=catalog)
    except Exception as exc:
        logger.warning("projection_build_failed", error=str(exc))
        return CampaignProjection(
            per_iteration_s=0.0, per_iteration_draw_uL={},
            budget=int(getattr(spec, "budget", 0)), duration_complete=False,
            warnings=[f"Could not build a representative trial: {exc}"],
        )

    eis = EISParams.from_preset(
        getattr(spec, "eis_preset", None),
        **(getattr(spec, "eis_overrides", None) or {}),
    )
    est = estimate_workflow_duration(wf, eis_params=eis)
    draw = per_iteration_draw(wf)

    if not est.is_complete:
        warnings.append(
            f"{est.n_unknown} of {est.n_steps} steps could not be timed; the "
            f"duration is a lower bound.")

    stock: dict[int, float | None] = {}
    if ledger is not None:
        for pid in draw:
            try:
                stock[pid] = ledger.remaining_uL(pid)
            except Exception:
                stock[pid] = None

    purge = dict(purge_uL_per_day or {})
    if purge:
        warnings.append(
            "Projection includes anti-clog purge consumption, which accrues with "
            "elapsed time rather than with iterations.")

        # Can a purge actually happen *during* this campaign? The background
        # timer defers for as long as the run holds the rig claim, so the only
        # in-run opportunity is an anneal hold. A trial with no anneal means the
        # lines stagnate untouched for the whole run — knowable now, rather than
        # discovered afterwards from a clogged check valve.
        if not any(
            (s.tags or {}).get("phase") == "anneal"
            for s in (list(wf.setup) + list(getattr(wf, "teardown", []) or []))
        ):
            per_it_min = est.total_s / 60.0
            warnings.append(
                f"No anneal phase, so no in-run purge opportunity: the "
                f"background purge defers for as long as a run holds the rig. "
                f"Lines will not be purged until the campaign ends "
                f"(~{per_it_min:.0f} min per iteration). Lines that the trials "
                f"themselves do not draw from — typically the particulate "
                f"line, which a zeroed component skips entirely — may clog."
            )

    projection = CampaignProjection(
        per_iteration_s=est.total_s,
        per_iteration_draw_uL=draw,
        budget=int(getattr(spec, "budget", 0)),
        duration_complete=est.is_complete,
        stock_uL=stock,
        purge_uL_per_day=purge,
        warnings=warnings,
    )

    if projection.stock_sufficient is False:
        supported = projection.iterations_supported()
        projection.warnings.insert(
            0,
            f"Declared stock supports only ~{supported} of {projection.budget} "
            f"iterations. The campaign will hard-stop before the budget.",
        )

    logger.info(
        "campaign_projected", campaign=getattr(spec, "name", "?"),
        per_iteration_s=round(est.total_s, 1),
        draw_uL=round(sum(draw.values()), 1),
        iterations_supported=projection.iterations_supported(),
    )
    return projection
