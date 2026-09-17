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

The **timed work** here is a lower bound on duration and a lower bound on draw:
it counts the dwells the workflow declares and ignores comms overhead, stage
travel, and ramp time. Stated plainly rather than padded with a fudge factor,
because an operator can reason about "at least this long" and cannot reason about
an unexplained multiplier.

**Two windows are the exception, and they are billed at their ceilings.** A run
plan carrying commanded conditions and an equilibrate phase spends time that is
bounded from *above* rather than from below, and there is no lower bound worth
quoting for either:

* a **condition approach** ends when the chamber arrives, and all the phase
  declares is the ``approach_timeout_s`` it will wait before giving up. Billing
  the ceiling overstates a normal approach; billing anything else understates a
  saturated one. Before this it was billed **zero** — a
  ``wait(within=…, timeout=…)`` step matched neither dwell key, so an 8 h cure's
  approach entered the projection as free *and* was not counted unknown either;
* an **equilibrate phase** terminates on evidence, so only ``max_hold_s`` is
  knowable in advance. It is not in the built workflow at all (the phase emits
  no steps — see :mod:`softae.core.run_plan`), so it is added from the spec's
  :class:`~softae.core.run_plan.SettlePlan` and named as a ceiling that may stop
  sooner.

:meth:`CampaignProjection.describe` says which of the two directions each part
of the number came from, because "at least 8 h" and "at most 12 h" are different
sentences and an operator planning a night needs to know which one they have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # annotation-only: `core` must not import `analysis` to run.
    from pathlib import Path

    from softae.analysis.rh_floor import TemperatureBin

import structlog

logger = structlog.get_logger(__name__)

#: Per-channel sweep cost, timed on the rig. **These are the ground truth**; the
#: model below exists only for grids that are not in this table.
#:
#: Measured 2026-08-17 on channel 1 by ``softae-eis-timing``, one session, all
#: four presets interleaved. The raw run is committed alongside the other rig
#: calibration artifacts at ``calibration/eis/timing_20260817.json`` — ``docs/``
#: is gitignored, so a provenance file there would not have survived. Each preset
#: was swept twice — a discarded warmup and a timed pass — and the two agreed to
#: better than 0.2 %, which is what licenses single-pass numbers: the instrument
#: is executing a fixed script, so replicate scatter is milliseconds, not
#: seconds. The timed regions accounted for 99.86 % of the session's wall clock,
#: so there is no meaningful unmeasured cost hiding between sweeps.
EIS_MEASURED_S_PER_CHANNEL: dict[str, float] = {
    "Quick": 17.50,
    "Standard": 37.19,
    "Extended": 120.42,
    "Longest": 516.44,
}

#: The grid each anchor above was timed at. **This is the staleness interlock.**
#:
#: An anchor is only meaningful for the sweep it measured, and presets get edited
#: — the 2026-08-17 mains-notch retune moved all four at once, instantly making
#: every then-current anchor a stopwatch on a grid that no longer existed. That
#: failure was survivable only because someone noticed. Keying the anchors to
#: their grids makes noticing unnecessary: :func:`measured_duration_s` compares
#: these against the live preset and returns ``None`` on any mismatch, so a
#: preset edit silently downgrades its own provenance to ``"extrapolated"``
#: instead of silently keeping a stopwatch's authority.
#:
#: ``mv_ac``/``mv_dc`` are deliberately absent: amplitude does not change how long
#: a sweep takes, and including it would retire good anchors for no reason.
EIS_ANCHOR_GRIDS: dict[str, dict[str, int]] = {
    "Quick":    {"npts": 27, "f_hi": 200_000, "f_lo_mHz": 6_475},
    "Standard": {"npts": 34, "f_hi": 200_000, "f_lo_mHz": 3_912},
    "Extended": {"npts": 53, "f_hi": 200_000, "f_lo_mHz": 1_351},
    "Longest":  {"npts": 39, "f_hi": 200_000, "f_lo_mHz": 228},
}

#: Periods measured per frequency point in an EIS sweep. The low-frequency end
#: dominates the sweep time; this is the multiplier on 1/f for each point.
#:
#: Refitted 2026-08-17 against all four measured presets above, replacing the
#: (33.2, 0.342) pair fitted to three anchors of which one had since been
#: retired. Against the new measurements that old pair ran +22.8 % on ``Quick``.
#:
#: **This form cannot fit this rig, and the residual says so.** The best two
#: constants reproduce the four anchors only to ~9 %, and the pattern is
#: structured rather than noisy — ``Standard`` under-predicted, ``Extended``
#: over-predicted, regardless of how the constants are chosen. Adding a third
#: term (fixed per-sweep overhead) moves the worst case 9.0 % -> 7.9 %, which is
#: not worth a parameter. Something frequency-dependent that this shape does not
#: express is going on, and finding it needs sweeps at grids the four presets do
#: not cover, not more refitting.
#:
#: That ~9 % is tolerable **only because it is now the fallback and not the
#: answer**: every shipped preset resolves through
#: :data:`EIS_MEASURED_S_PER_CHANNEL`, and the model governs custom sweeps alone.
EIS_CYCLES_PER_POINT = 34.5
#: Floor on per-point time, covering instrument overhead at high frequency. The
#: knee moved to ``34.5 / 0.131 ≈ 264 Hz`` in the 2026-08-17 refit, from ~97 Hz.
#: Treat the pair as fitted parameters rather than as physics: they are the least
#: bad two numbers for a shape that does not match the instrument.
EIS_MIN_POINT_S = 0.131


def _grid_key(eis_params: Any) -> tuple[int, int, int] | None:
    """``(npts, f_hi, f_lo_mHz)`` for *eis_params*, or ``None`` if unreadable."""
    try:
        return (
            int(getattr(eis_params, "npts")),
            int(getattr(eis_params, "f_hi")),
            int(getattr(eis_params, "f_lo_mHz")),
        )
    except Exception:
        return None


def measured_duration_s(eis_params: Any) -> float | None:
    """The timed cost for this exact grid, or ``None`` if nothing measured it.

    Matches on the grid rather than the preset name, so an overridden ``f_lo``
    or a hand-built sweep correctly finds no anchor even when it started life as
    a named preset. This is the check that keeps a stale stopwatch from being
    quoted at a sweep it never timed.
    """
    key = _grid_key(eis_params)
    if key is None:
        return None
    for name, grid in EIS_ANCHOR_GRIDS.items():
        if (grid["npts"], grid["f_hi"], grid["f_lo_mHz"]) == key:
            return EIS_MEASURED_S_PER_CHANNEL.get(name)
    return None


def measured_s_for_preset(preset: str | None) -> float | None:
    """Timed cost for *preset* as it is configured **right now**, else ``None``.

    Resolves the preset through config first, so a preset whose grid has been
    edited since it was timed reports no measurement rather than its old one.
    """
    if not preset:
        return None
    from softae.core.eis_scripts import EISParams

    return measured_duration_s(EISParams.from_preset(preset))


def eis_duration_basis(preset: str | None) -> str:
    """``"measured"`` for a preset still on its timed grid, else ``"extrapolated"``.

    A one-word provenance tag, not a confidence score. It answers ``"measured"``
    only while the preset's live grid matches the one in
    :data:`EIS_ANCHOR_GRIDS` — edit the preset and this reverts on its own,
    which is the whole point of keying anchors to grids.
    """
    return "measured" if measured_s_for_preset(preset) is not None else "extrapolated"


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
    """Best available sweep time for these parameters, in seconds per channel.

    **Prefers measurement over model.** If the grid matches one this rig has
    actually timed, that stopwatch reading is returned unmodified; the model is
    only consulted for grids nothing has measured. Before 2026-08-17 this always
    modelled, which meant projections carried a 22.8 % error on ``Quick`` while
    the true number sat unused two constants away.

    Use :func:`model_eis_duration` when you specifically need the model's own
    answer — calibrating the model against its anchors, for instance, which this
    function would otherwise short-circuit into a tautology.
    """
    if eis_params is None:
        from softae.core.eis_scripts import EISParams

        eis_params = EISParams()

    measured = measured_duration_s(eis_params)
    if measured is not None:
        return float(measured)
    return model_eis_duration(eis_params)


def model_eis_duration(eis_params: Any = None) -> float:
    """Modelled sweep time, in seconds per channel — the model's own answer.

    The low-frequency end dominates: a point at 0.2 Hz costs ~165 s while one at
    100 kHz is instrument-limited. Points are assumed log-spaced between
    ``f_hi`` and ``f_lo``, each costing ``EIS_CYCLES_PER_POINT`` periods or
    ``EIS_MIN_POINT_S``, whichever is longer.

    Reproduces the four measured presets to ~9 % — see
    :data:`EIS_CYCLES_PER_POINT` for why that is a property of the functional
    form and not of the constants. Callers surfacing this for a grid with no
    anchor should say the number is modelled.
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
        # Two different things share this method name, and they bound the
        # duration from opposite sides.
        #
        # `duration_s` / `seconds` are a **commanded dwell**: the step takes
        # exactly that long, and it is the number to bill.
        #
        # `timeout` is an **approach ceiling**: the driver returns as soon as the
        # chamber is inside tolerance, so the real cost is anywhere from seconds
        # to the whole timeout. The ceiling is the only defensible figure —
        # understating it is how an 8 h cure's approach becomes invisible — and
        # it is what `PhaseSetpoints.establish_steps` emits for both axes.
        #
        # Until this branch read `timeout`, such a step returned **0.0**: not
        # `None`, so it was not even counted as unknown. That is the failure
        # shape where "not measured" wears "measured and free"'s clothes, and
        # `workflows/equilibration.py` routes its own projection around this
        # function specifically because of it.
        dwell = _f(params, "duration_s") or _f(params, "seconds")
        return (dwell or _f(params, "timeout")) * scale

    return None


#: ``tags["phase"]`` on the steps that establish a phase's commanded conditions.
#: Mirrors :data:`softae.core.phase_setpoints.CONDITIONS_PHASE_TAG`; restated
#: rather than imported so this module stays free of the workflow-model layer
#: that ``phase_setpoints`` pulls in. ``test_preflight_projection.py`` pins the
#: two equal.
CONDITIONS_PHASE = "conditions"


def approach_ceiling_s(wf: Any) -> float:
    """Total time *wf* may spend waiting for commanded conditions to arrive.

    The upper-bound half of :func:`estimate_workflow_duration`'s total, split out
    so :meth:`CampaignProjection.describe` can say which part of the number is a
    floor and which is a cap. Reads the phase **tag**, not the step name: the
    names carry an operator-chosen condition label and a disambiguating suffix,
    and neither is a contract.
    """
    total = 0.0
    try:
        steps = list(wf.resolve_steps())
    except Exception:
        return 0.0
    for step in steps:
        if (getattr(step, "tags", None) or {}).get("phase") != CONDITIONS_PHASE:
            continue
        if str(getattr(step, "method", "")) != "wait":
            continue
        total += estimate_step_duration(step) or 0.0
    return total


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
    """What a campaign will cost in time and stock, with its limits named.

    :attr:`per_iteration_s` is the one number every consumer already reads, and
    it stays the whole cost — now including the two ceiling-bounded windows the
    module docstring describes. The four trailing fields decompose it so the
    summary can say which direction each part is bounded from; all four are
    defaulted, so a campaign with no run plan constructs and describes exactly
    as it did before.
    """

    per_iteration_s: float
    per_iteration_draw_uL: dict[int, float]
    budget: int
    duration_complete: bool = True
    #: Declared stock per pump at projection time (``None`` = unmanaged).
    stock_uL: dict[int, float | None] = field(default_factory=dict)
    #: Idle/in-run purge consumption, once P8 exists. ``0`` until then.
    purge_uL_per_day: dict[int, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: Of :attr:`per_iteration_s`, how much the **built workflow** declares —
    #: casts, dwells, the anneal hold, the sweeps. A lower bound.
    workflow_s: float = 0.0
    #: Of :attr:`workflow_s`, how much is condition-approach **ceiling** rather
    #: than timed work. Included in the total, reported separately because it is
    #: bounded from the other side.
    approach_ceiling_s: float = 0.0
    #: The equilibrate phase's ``max_hold_s``, which is **not** in the workflow —
    #: the phase emits no steps and terminates on evidence. ``0`` when the
    #: campaign does not settle.
    settle_ceiling_s: float = 0.0
    #: That phase's ``min_hold_s`` — the floor it cannot stop before.
    settle_floor_s: float = 0.0

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
        if self.approach_ceiling_s > 0:
            lines.append(
                f"  Includes up to {_fmt_duration(self.approach_ceiling_s)} "
                f"approaching commanded conditions — a ceiling, not a dwell: "
                f"each wait ends when the chamber arrives."
            )
        if self.settle_ceiling_s > 0:
            lines.append(
                f"  Plus an equilibrate window of at most "
                f"{_fmt_duration(self.settle_ceiling_s)} (at least "
                f"{_fmt_duration(self.settle_floor_s)}) — the phase stops when "
                f"the measurement stops moving, so it may end sooner."
            )

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


# ── The RH-floor advisory ────────────────────────────────────────────────────

def commanded_conditions(run_plan: Any, baseline: Any = None) -> list[Any]:
    """Every distinct :class:`PhaseSetpoints` a campaign commands, in run order.

    Deduplicated on the object itself, because the deposition engine emits a
    condition's steps **only when the setpoint changes** — two consecutive
    phases carrying the same conditions establish them once, and advising twice
    about one approach would misrepresent what the run does.

    *baseline* is the campaign-level ``[conditions]`` floor (T11.28), and it
    belongs at the **head** because that is where the engine commands it: before
    the piezo step and the startup flush, ahead of every phase. Without it, the
    one setpoint driven before the first cast is the only commanded humidity the
    RH-floor advisory cannot see. It defaults to ``None`` so every existing
    caller is unchanged, and it dedups against the first phase exactly as the
    engine does — a first phase equal to the baseline re-establishes nothing.
    """
    seen: list[Any] = []
    if baseline is not None:
        seen.append(baseline)
    for phase in getattr(run_plan, "phases", ()) or ():
        conditions = getattr(phase, "conditions", None)
        if conditions is not None and conditions not in seen:
            seen.append(conditions)
    return seen


def _bin_for(
    bins: "Sequence[TemperatureBin] | None", temperature_C: float, width: float
) -> "TemperatureBin | None":
    """The observed bin covering *temperature_C*, or ``None`` if none does.

    ``None`` rather than the nearest bin: a floor observed at 25 °C says nothing
    about 85 °C, and quoting it would be the advisory answering a question
    nobody asked.
    """
    for candidate in bins or ():
        if abs(float(candidate.temperature_C) - float(temperature_C)) <= width / 2.0:
            return candidate
    return None


def rh_floor_advisories(
    run_plan: Any,
    bins: "Sequence[TemperatureBin] | None",
    *,
    bin_width_C: float | None = None,
    baseline: Any = None,
) -> list[str]:
    """Advisories for commanded humidities below what this chamber has reached.

    **Advisory, never a refusal, and the wording is not ours.** The attainable
    minimum %RH is a property of the chamber's *state* — basin fill is
    uninstrumented, and :mod:`softae.analysis.rh_floor`'s own warning forbids
    fitting an absolute threshold to these numbers — so no setpoint can be
    rejected on this evidence. Each line therefore quotes
    :meth:`~softae.analysis.rh_floor.TemperatureBin.describe` verbatim, which
    already distinguishes the two cases that matter: a bin that was *asked for
    less than it delivered* ("do not command below this") from one whose lowest
    command was simply met ("floor not probed").

    **Three outcomes per condition, and they are three different strings.** The
    setpoint is below an observed floor; it is at or above one (silence); or the
    chamber has never been watched near that temperature at all — which is *not*
    the same as clean, and so is said rather than passed over.

    A phase that drives humidity but **not** temperature gets nothing: the bins
    are keyed on chamber temperature, and without one there is no bin to compare
    against and no honest sentence to write.
    """
    from softae.analysis.rh_floor import DEFAULT_BIN_WIDTH_C

    width = float(DEFAULT_BIN_WIDTH_C if bin_width_C is None else bin_width_C)
    lines: list[str] = []
    for conditions in commanded_conditions(run_plan, baseline):
        rh = getattr(conditions, "rh_setpoint_pct", None)
        temp = getattr(conditions, "temp_setpoint_C", None)
        name = getattr(conditions, "name", "?")
        if rh is None or temp is None:
            continue
        observed = _bin_for(bins, float(temp), width)
        if observed is None:
            lines.append(
                f"condition '{name}' commands {float(rh):g} %RH at "
                f"{float(temp):g} °C, and no conditions rows have been recorded "
                f"near that temperature — the floor there has never been "
                f"observed on this project, so nothing here says whether the "
                f"setpoint is reachable."
            )
            continue
        if float(rh) >= float(observed.rh_floor_pct):
            continue
        lines.append(
            f"condition '{name}' commands {float(rh):g} %RH at {float(temp):g} °C, "
            f"below the driest this chamber has been observed at that "
            f"temperature — {observed.describe()}. Advisory only: what the "
            f"chamber can reach depends on its state (the basin's fill is not "
            f"instrumented), so this is an observation, not a limit."
        )
    return lines


def _rh_floor_bins(
    project_dir: "Path | str | None",
) -> "list[TemperatureBin] | None":
    """The observed floor bins for *project_dir*, or ``None`` if not consulted.

    ``None`` and ``[]`` are deliberately different answers: ``None`` means
    nobody looked, ``[]`` means the history was read and holds nothing. A single
    token for both is how "unknown" comes to wear "checked and clean"'s clothes.
    """
    if project_dir is None:
        return None
    try:
        from softae.analysis.rh_floor import rh_floor_by_temperature

        return rh_floor_by_temperature(project_dir)
    except Exception as exc:  # a read-only history is never a reason to refuse
        logger.warning("rh_floor_unavailable", error=str(exc))
        return None


def _settle_window(spec: Any, warnings: list[str]) -> tuple[float, float]:
    """``(max_hold_s, min_hold_s)`` for this campaign's equilibrate phase.

    ``(0.0, 0.0)`` when it does not settle. The phase emits **no steps** — it
    terminates on evidence, which is a loop and not a sequence — so this is the
    one part of a run plan's cost that cannot be read off the built workflow and
    has to come from the spec.

    :meth:`CampaignSpec.settle_plan` refuses a spec that names settle twice; that
    refusal must not become a reason a projection cannot run, so it is caught and
    reported as a warning here. A campaign in that state will fail at launch on
    the same message, from the authority that owns it.
    """
    try:
        plan = spec.settle_plan()
    except AttributeError:
        return 0.0, 0.0
    except Exception as exc:
        warnings.append(f"Equilibrate window not projected: {exc}")
        return 0.0, 0.0
    if plan is None:
        return 0.0, 0.0
    return float(plan.max_hold_s), float(plan.min_hold_s)


def _rh_floor_warnings(spec: Any, project_dir: "Path | str | None") -> list[str]:
    """RH-floor advisories for *spec*'s run plan, or a note that none were read.

    The second half is the point. A plan that commands humidity and a projection
    that consulted no history produce the same *silence* as a plan checked and
    found clean, and silence is what lets an unreachable setpoint walk into an
    8 h cure. So the absence is stated.
    """
    run_plan = getattr(spec, "run_plan", None)
    baseline = getattr(spec, "conditions", None)
    if not any(
        getattr(c, "rh_setpoint_pct", None) is not None
        for c in commanded_conditions(run_plan, baseline)
    ):
        return []

    bins = _rh_floor_bins(project_dir)
    if bins is None:
        return ["This plan commands humidity, but no RH-floor history was "
                "consulted (no project directory was supplied), so a setpoint "
                "below what this chamber can reach would not be flagged here."]
    # An empty history needs no separate branch: with no bins, every commanded
    # humidity falls in no bin, and `rh_floor_advisories` says so per condition —
    # which is the more specific sentence, naming the temperature that was never
    # watched rather than only that the project has no rows at all.
    return rh_floor_advisories(run_plan, bins, baseline=baseline)


#: The A5 advisory (T11.28 §10). Not a refusal: an advisory commands no
#: humidifier, and the fix is a ``[conditions]`` block the operator writes.
UNCONDITIONED_START_WARNING = (
    "This plan's first phase drives neither temperature nor humidity, and no "
    "campaign-level [conditions] baseline is declared: a campaign following a "
    "park is unconditioned until the first phase that carries conditions, so "
    "the chamber will drift to room RH during the startup flush and the first "
    "casts. Declare a top-level [conditions] block to drive both axes from the "
    "top of the run (it waits for nothing)."
)


def unconditioned_start_warnings(spec: Any) -> list[str]:
    """The A5 advisory for *spec*, or ``[]``.

    Gated on a plan actually existing. A campaign with **no** ``run_plan`` runs
    the engine's legacy pointwise layout, which has never commanded the chamber
    at all — that is a different and older situation than a written plan whose
    first phase happens to stay silent, and saying this sentence about it would
    put a note on every legacy campaign that projects today.
    """
    from softae.core.phase_setpoints import first_phase_axes

    run_plan = getattr(spec, "run_plan", None)
    if run_plan is None or getattr(spec, "conditions", None) is not None:
        return []
    if not (getattr(run_plan, "phases", ()) or ()):
        return []
    return [] if first_phase_axes(run_plan) else [UNCONDITIONED_START_WARNING]


def project_campaign(
    spec: Any,
    *,
    catalog: Any,
    ledger: Any = None,
    purge_uL_per_day: dict[int, float] | None = None,
    project_dir: "Path | str | None" = None,
) -> CampaignProjection:
    """Project one campaign's per-iteration time and stock draw.

    Builds a **representative trial** at the midpoint of the parameter space —
    a single trial is what a projection can honestly be based on, since the
    optimizer chooses the rest.

    *project_dir* is the DataStore project directory the RH-floor history is
    read from (read-only; :mod:`softae.analysis.rh_floor` opens SQLite in
    ``mode=ro``). It is optional and defaulted so every existing caller is
    unchanged — but a run plan that commands humidity with no directory supplied
    says so in the warnings rather than reporting silence as a clean bill.
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

    preset = getattr(spec, "eis_preset", None)
    overrides = getattr(spec, "eis_overrides", None) or {}
    eis = EISParams.from_preset(preset, **overrides)
    est = estimate_workflow_duration(wf, eis_params=eis)
    draw = per_iteration_draw(wf)

    # An operator reading "at most 14 h" has no way to tell a projection anchored
    # on a stopwatch from one the model invented past its last anchor, and the two
    # deserve different amounts of trust. Overrides count as leaving the anchors:
    # a Standard preset with f_lo pushed to 0.2 Hz is not the Standard that was timed.
    if overrides or eis_duration_basis(preset) != "measured":
        warnings.append(
            f"EIS duration for preset '{preset or 'default'}' is EXTRAPOLATED — "
            f"the sweep model is calibrated only against "
            f"{', '.join(sorted(EIS_MEASURED_S_PER_CHANNEL))}, and this is not one "
            f"of them."
        )

    if not est.is_complete:
        warnings.append(
            f"{est.n_unknown} of {est.n_steps} steps could not be timed; the "
            f"duration is a lower bound.")

    # ── The two windows the built workflow cannot price ──────────────────────
    approach_s = approach_ceiling_s(wf)
    settle_ceiling, settle_floor = _settle_window(spec, warnings)
    per_iteration_s = est.total_s + settle_ceiling

    warnings.extend(unconditioned_start_warnings(spec))
    warnings.extend(_rh_floor_warnings(spec, project_dir))

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
            per_it_min = per_iteration_s / 60.0
            warnings.append(
                f"No anneal phase, so no in-run purge opportunity: the "
                f"background purge defers for as long as a run holds the rig. "
                f"Lines will not be purged until the campaign ends "
                f"(~{per_it_min:.0f} min per iteration). Lines that the trials "
                f"themselves do not draw from — typically the particulate "
                f"line, which a zeroed component skips entirely — may clog."
            )

    projection = CampaignProjection(
        per_iteration_s=per_iteration_s,
        per_iteration_draw_uL=draw,
        budget=int(getattr(spec, "budget", 0)),
        duration_complete=est.is_complete,
        stock_uL=stock,
        purge_uL_per_day=purge,
        warnings=warnings,
        workflow_s=est.total_s,
        approach_ceiling_s=approach_s,
        settle_ceiling_s=settle_ceiling,
        settle_floor_s=settle_floor,
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
        per_iteration_s=round(per_iteration_s, 1),
        workflow_s=round(est.total_s, 1),
        approach_ceiling_s=round(approach_s, 1),
        settle_ceiling_s=round(settle_ceiling, 1),
        draw_uL=round(sum(draw.values()), 1),
        iterations_supported=projection.iterations_supported(),
    )
    return projection
