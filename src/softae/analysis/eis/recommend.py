"""Where a threshold *would* sit, if the operator decides to arm it.

``docs/SHADOW_CAMPAIGN.md`` §8 ends its procedure with a judgement the tooling refuses
to make: *"Nothing here is a number the tool can decide for you."* That is half right,
and the half it gets right is the important half. Whether to arm a gate is a scientific
claim — it asserts that the samples the gate would have discarded deserved discarding,
and no arithmetic establishes that. But **where** a threshold sits, given the decision
to arm, is a percentile of a distribution, and doing that by eye across fourteen keys
and hundreds of spectra is exactly the kind of work that gets skipped.

So this module computes values and never decides. Every function here is pure: no
DataStore, no config writes, no argparse, no I/O. :mod:`softae.tools.shadow_review`
reads the log, builds the records and renders; this module owns the **metric →
config-key map**, which is a fact about the gates rather than about the review tool. A
second copy of that map under ``tools/`` would drift the first time a gate renames a
metric.

Three modules, in one direction:
:mod:`~softae.analysis.eis.recommend_rules` (distribution math, knows no keys) ←
**this one** (the map and the verdict) ←
:mod:`~softae.analysis.eis.recommend_report` (the catalogues of what cannot be
recommended, and the paste-ready TOML block).

**The refusals are the point.** A recommendation is withheld — by name, with a reason —
whenever the evidence cannot support a number: a metric never observed, a sample below
the evidence floor, a constant masquerading as a distribution, a gate no spectrum ever
challenged, or a pre-T7.1 log in which ``metrics=`` exists only for the failing tail.
A fence placed on a sample of rejects is worse than no fence at all, because it looks
like one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from softae.analysis.eis.recommend_rules import (
    complement_fence,
    count_minimum,
    decade_margin,
    gap_split,
    lower_fence,
    physical_point_floor,
    upper_fence,
)

#: The event Part A added. Its absence in a log is itself a finding (§6.3).
METRICS_EVENT = "eis_spectrum_metrics"

#: Spectra a metric must be observed on before a fence may be placed against it.
#:
#: Chosen rather than inherited: at n = 20 the empirical P95 is the second-largest
#: observation, so the fence is still supported by two points instead of resting
#: entirely on the single worst spectrum of the run. The packaged 16-well shadow spec
#: (``examples/shadow_campaign.toml``, ``budget = 16``) sits deliberately *below* it —
#: the honest output there is "run 32 wells", not a threshold invented from 16 samples.
DEFAULT_MIN_EVIDENCE = 20

#: Metrics that exist only once a fit with covariance ran. When a log mixes observing
#: and enforcing regimes the enforcing spectra are a *strict sample of the admitted
#: population* — R18 skips the fit for anything it rejected — so their Front-2 values
#: are excluded rather than pooled (§6.2).
FRONT2_METRICS = frozenset({
    "rel_se_measurand", "rho", "residual_rms_pct", "r_squared", "residual_max_pct",
    "chi2", "chi2_reduced", "n_pegged", "runs_z", "cross_check_pct",
})

#: The two reasons a key may be held rather than refused or recommended.  Named so the
#: renderer selects on a value rather than re-reading the prose that explains it.
HOLD_UNIMODAL = "unimodal"
HOLD_UNEXERCISED = "unexercised"


# ── The record one spectrum contributes ──────────────────────────────────────

@dataclass(frozen=True)
class SpectrumRecord:
    """One ``eis_spectrum_metrics`` event, parsed.

    ``key`` is the content fingerprint :func:`softae.analysis.eis.engine.spectrum_key`
    produced, and it is what makes the sample sizes below honest — see
    :func:`deduplicate`.
    """

    key: str
    channel: int | None = None
    verdict: str = ""
    enforced: bool = False
    report_mode: str = ""
    fit_ok: bool = False
    n_surviving: int = 0
    n_dropped: int = 0
    gates_run: tuple[str, ...] = ()
    gates_failed: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_event(cls, fields: dict[str, Any]) -> "SpectrumRecord | None":
        """Build a record from a parsed log line, or ``None`` if it is not one.

        Tolerant of a half-rendered line: a truncated log is common and a record
        missing its optional fields still carries usable metrics. Only ``metrics``
        being unparseable disqualifies the line, because a record with no metrics
        contributes to no distribution.
        """
        if str(fields.get("event", "")) != METRICS_EVENT:
            return None
        raw = fields.get("metrics")
        if not isinstance(raw, dict):
            return None

        metrics = _finite(raw)
        n_surviving = _as_int(fields.get("n_surviving")) or 0
        n_dropped = _as_int(fields.get("n_dropped")) or 0
        metrics.update(_derived(metrics, n_surviving + n_dropped))
        return cls(
            key=str(fields.get("spectrum_key", "")),
            channel=_as_int(fields.get("channel")),
            verdict=str(fields.get("verdict", "")),
            enforced=bool(fields.get("enforced", False)),
            report_mode=str(fields.get("report_mode", "")),
            fit_ok=bool(fields.get("fit_ok", False)),
            n_surviving=n_surviving,
            n_dropped=n_dropped,
            gates_run=tuple(str(g) for g in _as_list(fields.get("gates_run"))),
            gates_failed=tuple(str(g) for g in _as_list(fields.get("gates_failed"))),
            metrics=metrics,
        )


def _derived(metrics: dict[str, float], n_band: int) -> dict[str, float]:
    """Metrics the gates imply but do not store.

    ``kk_max_truncate_frac`` is compared inside ``gate_kk_truncation`` against
    ``n_run / n`` — a ratio computed and discarded, with only the numerator
    (``kk_truncated``) surviving into the metrics dict, and only when at least one
    point failed. Reconstructing it needs the band size, which the event carries as
    ``n_surviving + n_dropped``.

    A K–K-compliant spectrum truncated nothing, so its fraction is **0**, not missing —
    and the difference decides whether the key reads as unexercised (correct) or as
    having four observations (wrong). ``kk_max_resid_pct`` is the marker that the ladder
    ran at all; without it the gate produced no observation either way.
    """
    if "kk_max_resid_pct" not in metrics or n_band <= 0:
        return {}
    return {"kk_truncate_frac": metrics.get("kk_truncated", 0.0) / float(n_band)}


def deduplicate(records: Iterable[SpectrumRecord]) -> list[SpectrumRecord]:
    """Collapse repeat analyses of one physical spectrum, keeping the richest.

    An autonomous campaign analyses every spectrum twice — the auto-route fit and the
    objective extraction — so a naive population is 2× the true sample size and every
    evidence floor passes at half the evidence it demands. The two calls see the same
    arrays but not the same context: the objective call may carry no ``cell`` and so a
    different ``report_mode``, and a call whose fit raised carries no covariance
    metrics. Keeping the **larger metrics superset** takes the analysis that saw most.
    """
    best: dict[str, SpectrumRecord] = {}
    for record in records:
        seen = best.get(record.key)
        if seen is None or len(record.metrics) > len(seen.metrics):
            best[record.key] = record
    return list(best.values())


# ── The recommendation ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Recommendation:
    """A proposed value for one config key, or a named refusal to propose one."""

    section: str                 # "eis.gates" | "quality"
    key: str                     # "cap_flatness_max"
    metric: str                  # "cap_slope"
    default: float
    value: float | None          # None ⇒ refused or held; the default stands
    rule: str                    # upper-fence | lower-fence | gap | complement | count
    status: str                  # "recommended" | "hold" | "refused"
    direction: str               # "loosens" | "tightens" | "unchanged" | "-"
    reason: str                  # always populated
    n: int                       # spectra carrying a finite value for this metric
    fired_at_default: int
    would_reject_at_value: int
    exercise: str                # unexercised | exercised | measures-the-rig
    behavioural: bool            # does changing this move a NUMBER, not just a verdict?
    #: Which of the two holds this is, when ``status == "hold"``; ``""`` otherwise.
    #: Carried explicitly rather than inferred from :attr:`reason`, because the two
    #: are decided in different places for different reasons and ``exercise`` cannot
    #: separate them — a gap rule that found no split is a unimodal hold even when the
    #: gate also never fired.
    hold_kind: str = ""

    @property
    def applied(self) -> bool:
        """True when this recommendation carries a value an operator could paste."""
        return self.status == "recommended" and self.value is not None


@dataclass(frozen=True)
class _Key:
    """One recommendable config key and the distribution rule that fits it."""

    section: str
    key: str
    metric: str
    rule: str
    #: ``upper``: the gate passes while ``metric <= key``. ``lower``: while
    #: ``metric >= key``. ``lower_eq``: the condition *fires* at ``metric <= key`` —
    #: ``rho_degenerate`` selects a reporting basis rather than passing a spectrum.
    sense: str
    behavioural: bool = False
    absolute: bool = False       # graded on |metric|, as cap_flatness is
    geometric: bool = False      # decade margins rather than a percentage
    min_gap: float = 0.0
    span: tuple[float, float] = (-math.inf, math.inf)
    floor: float | None = None   # a value may never be proposed below this
    note: str = ""


#: The map. Metric names are exactly the keys ``gates.gate_metrics()`` produces; the
#: ``[quality]`` metrics arrive from the ``fit_report.metrics`` merge inside
#: ``analyze_spectrum`` and from ``gate_phase_noise_extrapolated``.
METRIC_KEYS: tuple[_Key, ...] = (
    _Key("eis.gates", "tand_slope_max", "tand_slope", "gap", "upper",
         min_gap=0.25, span=(-1.2, 0.5),
         note="−1 is ideal parallel conduction, +1 an ideal series parasitic: the "
              "physics separates two populations rather than grading one"),
    _Key("eis.gates", "cap_flatness_max", "cap_slope", "upper-fence", "upper",
         absolute=True, floor=0.0),
    _Key("eis.gates", "kk_resid_pct", "kk_max_resid_pct", "upper-fence", "upper",
         behavioural=True, floor=0.0),
    _Key("eis.gates", "kk_max_truncate_frac", "kk_truncate_frac", "upper-fence",
         "upper", behavioural=True, floor=0.0),
    _Key("eis.gates", "plateau_min_decades", "plateau_decades", "lower-fence",
         "lower", floor=0.0),
    _Key("eis.gates", "min_fit_pts", "n_surviving", "count", "lower"),
    _Key("eis.gates", "max_rel_se", "rel_se_measurand", "upper-fence", "upper",
         floor=0.0,
         note="conditioned on a fit with covariance having existed"),
    _Key("eis.gates", "rho_degenerate", "rho", "gap", "lower_eq",
         behavioural=True, min_gap=0.03, span=(-1.0, 1.0)),
    _Key("eis.gates", "residual_hard_pct", "residual_rms_pct", "upper-fence",
         "upper", floor=0.0),
    _Key("quality", "min_r_squared", "r_squared", "complement", "lower"),
    _Key("quality", "max_residual_pct", "residual_rms_pct", "upper-fence", "upper",
         floor=0.0,
         note="the same distribution as eis.gates.residual_hard_pct, graded twice — "
              "one catches catastrophe, the other grades"),
    _Key("quality", "min_points", "n_surviving", "count", "lower"),
    _Key("quality", "min_abs_z", "z_median", "lower-fence", "lower", geometric=True),
    _Key("quality", "max_abs_z", "z_median", "upper-fence", "upper", geometric=True),
)

# ── Rule dispatch ────────────────────────────────────────────────────────────

def _propose(spec: _Key, values: Sequence[float], floor_points: int) -> float | None:
    """Dispatch to the rule this key's metric calls for.

    The rules themselves live in :mod:`softae.analysis.eis.recommend_rules` and know
    nothing about config keys — this is the only place the two meet.
    """
    if spec.rule == "gap":
        return gap_split(values, min_gap=spec.min_gap, span=spec.span)
    if spec.rule == "count":
        return count_minimum(values, floor_points)
    if spec.rule == "complement":
        return complement_fence(values)
    if spec.geometric:
        return decade_margin(values, upper=spec.sense == "upper")
    proposed = upper_fence(values) if spec.sense == "upper" else lower_fence(values)
    # A fence may land below a physical minimum on a very clean population — a
    # negative `cap_flatness_max` rejects every spectrum, a negative
    # `plateau_min_decades` accepts every one. Neither is what the distribution said.
    return proposed if spec.floor is None else max(proposed, spec.floor)


# ── Assembly ─────────────────────────────────────────────────────────────────

def recommend_all(
    records: Sequence[SpectrumRecord],
    *,
    settings: Any = None,
    quality_cfg: dict[str, float] | None = None,
    min_evidence: int = DEFAULT_MIN_EVIDENCE,
    model_name: str = "simpleSalt",
) -> list[Recommendation]:
    """One :class:`Recommendation` per key in :data:`METRIC_KEYS`, in that order.

    ``records`` are expected already deduplicated by :func:`deduplicate`; passing raw
    events doubles every ``n`` and halves the effective evidence floor.
    """
    if settings is None:
        from softae.analysis.eis.settings import GateSettings

        settings = GateSettings()
    if quality_cfg is None:
        from softae.analysis.quality import quality_config

        quality_cfg = quality_config({})

    records = list(records)
    floor_points = physical_point_floor(model_name)
    mixed_regimes = len({r.enforced for r in records}) > 1
    return [_recommend_one(spec, records, settings=settings, quality_cfg=quality_cfg,
                           min_evidence=int(min_evidence), floor_points=floor_points,
                           mixed_regimes=mixed_regimes)
            for spec in METRIC_KEYS]


def _recommend_one(spec: _Key, records: Sequence[SpectrumRecord], *, settings: Any,
                   quality_cfg: dict[str, float], min_evidence: int,
                   floor_points: int, mixed_regimes: bool) -> Recommendation:
    default = _default_for(spec, settings, quality_cfg)
    values = _observations(spec, records, mixed_regimes)
    n = len(values)
    fired = sum(1 for v in values if _fires(spec, v, default))
    exercise = _exercise(fired, n)

    def _out(status: str, reason: str, value: float | None = None,
             rejected: int = 0, hold_kind: str = "") -> Recommendation:
        return Recommendation(
            section=spec.section, key=spec.key, metric=spec.metric, default=default,
            value=value, rule=spec.rule, status=status,
            direction=_direction(spec, value, default), reason=reason, n=n,
            fired_at_default=fired, would_reject_at_value=rejected,
            exercise=exercise, behavioural=spec.behavioural, hold_kind=hold_kind,
        )

    if not records:
        return _out("refused",
                    "pre-T7.1 log: metrics exist only for the failing tail, so there "
                    "is no pass-side distribution to place a fence against")
    if n == 0:
        return _out("refused",
                    f"never observed — {spec.metric} was not produced by any "
                    f"spectrum, so the gate did not run or produced no metric")
    if n < min_evidence:
        return _out("refused",
                    f"only {n} spectra carry {spec.metric} (need {min_evidence}); "
                    f"below {min_evidence} the P95 is a single observation")
    distinct = len({round(v, 12) for v in values})
    if distinct < 3:
        return _out("refused",
                    f"{n} spectra but only {distinct} distinct value(s) of "
                    f"{spec.metric} — a constant is not a distribution")

    # Order matters, and the two holds are not interchangeable. A gap rule that found
    # no split is a *unimodal* hold even on a gate that also never fired: the evidence
    # question ("are there two populations?") is answered before the exercise question
    # ("has anything challenged the default?"), so the reason reported is the one that
    # actually stopped the recommendation.
    proposed = _propose(spec, values, floor_points)
    if proposed is None:
        return _out("hold", hold_kind=HOLD_UNIMODAL, reason=(
            f"unimodal at n={n} — no two populations to separate, so the "
            f"theory-anchored default stands"))
    if fired == 0:
        return _out("hold", hold_kind=HOLD_UNEXERCISED, reason=(
            f"unexercised: 0/{n} spectra failed this gate at its default. "
            f"Untested is not validated — arming it on this run would give "
            f"authority to a number no spectrum has challenged"))

    rejected = sum(1 for v in values if _fires(spec, v, proposed))
    reason = f"{spec.rule} on {n} spectra; {fired} fired at the default → {rejected}"
    if exercise == "measures-the-rig":
        reason += (". This gate fired on ≈every spectrum: at its default it is "
                   "measuring the rig, not the sample. The proposed value is what "
                   "the rig's own population supports")
    if spec.note:
        reason += f". {spec.note}"
    return _out("recommended", reason, value=proposed, rejected=rejected)


def joint_would_reject(recs: Sequence[Recommendation],
                       records: Sequence[SpectrumRecord]) -> int:
    """Spectra failing **at least one** recommended key at its proposed value.

    The number the operator actually acts on, and never the sum of the per-key counts:
    one bad spectrum routinely fails several gates, so adding the columns overstates
    the cost of applying the block — sometimes by more than the population size.
    """
    by_key = {spec.key: spec for spec in METRIC_KEYS}
    applied = [r for r in recs if r.applied]
    hit = 0
    for record in records:
        for rec in applied:
            value = _value_of(by_key[rec.key], record)
            if value is not None and _fires(by_key[rec.key], value, rec.value):
                hit += 1
                break
    return hit


# ── Small shared helpers ─────────────────────────────────────────────────────

def _default_for(spec: _Key, settings: Any, quality_cfg: dict[str, float]) -> float:
    if spec.section == "quality":
        return float(quality_cfg.get(spec.key, float("nan")))
    return float(getattr(settings, spec.key, float("nan")))


def _value_of(spec: _Key, record: SpectrumRecord) -> float | None:
    """This key's observation on one spectrum, or ``None`` if it carries none."""
    raw = record.metrics.get(spec.metric)
    if raw is None or not math.isfinite(raw):
        return None
    return abs(raw) if spec.absolute else float(raw)


def _observations(spec: _Key, records: Sequence[SpectrumRecord],
                  mixed_regimes: bool) -> list[float]:
    skip_enforced = mixed_regimes and spec.metric in FRONT2_METRICS
    values = []
    for record in records:
        if skip_enforced and record.enforced:
            continue
        value = _value_of(spec, record)
        if value is not None:
            values.append(value)
    return values


def _fires(spec: _Key, value: float, threshold: float) -> bool:
    """Would this key act on this observation, at this threshold?"""
    if not math.isfinite(threshold):
        return False
    if spec.sense == "upper":
        return value > threshold
    if spec.sense == "lower_eq":
        return value <= threshold
    return value < threshold


def _direction(spec: _Key, value: float | None, default: float) -> str:
    """Does the proposal loosen the gate or tighten it?

    Tightening is permitted, and marked. It is only ever reachable for a gate that
    fired at least once — a gate no spectrum challenged is held as ``unexercised``
    before any value is computed — so no threshold is tightened on silence.
    """
    if value is None or not math.isfinite(default):
        return "-"
    if value == default:
        return "unchanged"
    looser = (value > default) if spec.sense == "upper" else (value < default)
    return "loosens" if looser else "tightens"


def _exercise(fired: int, n: int) -> str:
    """§8's prose failure shapes, made numeric.

    0.90 rather than 1.0 because a gate firing on 58 of 60 is already measuring the
    rig, and a rule keyed to "all" would be defeated by two outliers.
    """
    if n == 0 or fired == 0:
        return "unexercised"
    return "measures-the-rig" if fired / n >= 0.90 else "exercised"


def _finite(raw: dict[Any, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[str(name)] = number
    return out


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []
