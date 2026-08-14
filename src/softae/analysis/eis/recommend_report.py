"""What a shadow run *cannot* propose, and the artifact it hands the operator.

Two halves, both of them statements rather than proposals:

* the catalogues — keys deliberately out of scope, and hardcoded constants that fire in
  practice but have no config line to move. Stated in the report rather than silently
  omitted, because a key missing from a recommendation table reads as a key with
  nothing to say about it.
* :func:`as_toml_block` — the paste-ready block, in which refused and held keys are
  emitted **commented out with their reason** so pasting can never silently apply a
  non-recommendation.

Imports run one way: this module depends on :mod:`softae.analysis.eis.recommend`,
never the reverse.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from softae.analysis.eis.gates import MIN_PLATEAU_POINTS, RE_SUSPICION_FRAC
from softae.analysis.eis.recommend import Recommendation, SpectrumRecord
from softae.analysis.eis.recommend_rules import pct

#: ``gate_model_free_crosscheck`` and ``gate_residual_structure`` compare against these
#: numbers inline — there is no module constant to import, which is itself part of what
#: §5.2 reports. Restated here because there is nothing to reference, not by preference.
_CROSS_CHECK_PCT = 25.0
_RUNS_Z_LIMIT = 3.0

#: Keys stated as out of scope rather than silently omitted, one reason each.
OUT_OF_SCOPE: tuple[tuple[str, str], ...] = (
    ("enabled", "never auto-set — that is §8's decision, not a distribution's"),
    ("kk_c", "a numerical parameter of the Lin-KK ladder, not a data-quality limit"),
    ("kk_max_M", "a runtime bound on ladder order, not a data-quality limit"),
    ("bound_tol", "a distance to a box constraint; the log carries n_pegged, a count, "
                  "so there is no distance distribution to fence"),
    ("plateau_tol_pct", "defines what 'flat' means *inside* plateau_decades — "
                        "recommending both from one distribution would be circular"),
    ("tand_headroom_mult", "decides value-vs-bound, which is not behind `enabled`; "
                           "needs the phase-floor calibration, not a shadow run"),
    ("blank_over_frac / blank_flip_frac / blank_repro_pct",
     "blank-qualification gates — no blank is measured in a shadow campaign"),
    ("geom_sigma_spread_tol / geom_slope_exponent_tol",
     "the E5 geometry-series route, which a shadow campaign does not exercise"),
)

#: Hardcoded constants that fire in practice. They have no config line, so their
#: distributions are reported as evidence *for* a future key and never as a proposal:
#: proposing a value without also proposing the line it lives on would be inventing
#: config, which is not this tool's job.
UNCONFIGURABLE: tuple[tuple[str, str, float | None, str], ...] = (
    ("frac_quadrant_violation", "gates.RE_SUSPICION_FRAC", RE_SUSPICION_FRAC, "upper"),
    ("cross_check_pct", "gate_model_free_crosscheck", _CROSS_CHECK_PCT, "upper"),
    ("runs_z", "gate_residual_structure |z|", _RUNS_Z_LIMIT, "abs-upper"),
    ("plateau_n_points", "gates.MIN_PLATEAU_POINTS", float(MIN_PLATEAU_POINTS), "lower"),
    ("zreal_slope", "gate_series_rc slope pair", None, "-"),
    ("zimag_slope", "gate_series_rc slope pair", None, "-"),
)


def observed_only(records: Sequence[SpectrumRecord]) -> list[dict[str, Any]]:
    """Distributions of the hardcoded constants, with how often each one fired.

    A constant no spectrum measured is omitted rather than reported with a zero, which
    would read as "measured, never fired" — the opposite of what happened.
    """
    out: list[dict[str, Any]] = []
    for metric, owner, threshold, sense in UNCONFIGURABLE:
        values = [r.metrics[metric] for r in records if metric in r.metrics]
        if not values:
            continue
        fired = 0 if threshold is None else sum(
            1 for v in values
            if (abs(v) > threshold if sense == "abs-upper"
                else v > threshold if sense == "upper" else v < threshold))
        out.append({"metric": metric, "owner": owner, "threshold": threshold,
                    "n": len(values), "p50": pct(values, 50), "p95": pct(values, 95),
                    "fired": fired})
    return out


def as_toml_block(recs: Sequence[Recommendation], *, source: str, n_spectra: int,
                  when: str) -> str:
    """A ``[eis.gates]`` / ``[quality]`` block an operator can paste, and audit.

    ``enabled`` is written ``false`` in both sections and is never written otherwise.
    Arming stays §8's decision, and a tool that emitted ``true`` — even once, even
    correctly — would have made it for them.
    """
    lines = [
        f"# ── RECOMMENDED by `softae-shadow review {source}` — {when} " + "─" * 12,
        f"# Evidence: {n_spectra} spectra. Paste into softae_config.toml.",
        "# Keys marked (!) change stored NUMBERS, not only verdicts — re-fit before",
        "# trusting any σ produced under them.",
    ]
    for section in ("eis.gates", "quality"):
        lines.append(f"[{section}]")
        lines.append("# ARMING IS YOURS. This block never sets `enabled` true; see")
        lines.append("# docs/SHADOW_CAMPAIGN.md §8 and read the would-reject table "
                     "gate by gate first.")
        lines.append(f"{'enabled':<20} = false")
        lines.extend(_toml_line(r) for r in recs if r.section == section)
    return "\n".join(lines)


def _toml_line(rec: Recommendation) -> str:
    mark = " (!)" if rec.behavioural else ""
    if not rec.applied:
        label = "HELD" if rec.status == "hold" else "REFUSED"
        return f"# {rec.key}{mark} — {label} at {_fmt(rec.default)}: {rec.reason}."
    return (f"{rec.key:<20} = {_fmt(rec.value)}"
            f"  #{mark} {rec.rule}; n={rec.n}; "
            f"{rec.fired_at_default} → {rec.would_reject_at_value} ({rec.direction})")


def _fmt(value: float) -> str:
    if value != value:
        return "nan"
    if float(value).is_integer() and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.6g}"
