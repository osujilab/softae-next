"""Which EIS analysis engine runs, and with what thresholds.

Two engines exist side by side. ``legacy`` is what the rig has always done:
fit ``R0-CPE0-p(R1,C0)`` through ``impedance.py``, take ``R1``, divide. ``gated``
is the framework in ``docs/EIS_GATE_FRAMEWORK_new.md`` — admission gates, covariance,
per-sample cell constant, upper bounds when the measurement is resolution-limited.

**Legacy is the shipped default and stays so until the gated path is validated on
the bench.** That is not caution for its own sake: a gate calibrated against no real
spectra is a worse failure than the one it fixes, because it discards good samples
silently and biases a campaign toward the data the checker expected. The same
lesson is why ``[quality] enabled`` and ``[purge] actuate`` ship false.

``enabled`` is deliberately separate from ``engine`` — the same two-flag design
:class:`~softae.core.purge.PurgeSettings` uses. ``engine = "gated"`` with
``enabled = false`` runs every gate and logs every verdict while removing nothing,
so the thresholds can be observed against real runs before they are given authority
over data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


#: Analysis engines. ``legacy`` is bit-for-bit what the rig did before this work.
ENGINES = ("legacy", "gated")

#: What a campaign optimises against.
#:
#: ``auto`` (default)
#:     Conductivity when the campaign can produce one, impedance otherwise. This
#:     tracks the two legitimate campaign *modes* rather than overriding them —
#:     see below.
#: ``sigma``
#:     Conductivity, **maximised**. Requires a per-sample thickness.
#: ``mean_abs_z``
#:     Mean |Z| across the sweep, **minimised**.
#:
#: **The two modes, and why neither is a fallback for the other.**
#:
#: A *composition* campaign plans formulations and maps them to volumes, so the twin
#: knows what was cast and can predict a thickness — conductivity is available and is
#: the quantity actually wanted.
#:
#: A *volume* campaign explores raw pump volumes directly. Exploration is easier and
#: feasibility is native (a volume limit is just a bound), with composition resolved
#: post hoc. But without stock identity there is no elution and therefore no dry
#: thickness, so σ is *structurally* unavailable — not missing, impossible. Mean |Z|
#: is the honest objective there, and a perfectly reasonable one for exploration.
#:
#: ``auto`` picks the right one per campaign and logs which and why. Pinning either
#: value forces it, and a campaign that cannot honour the pin is refused rather than
#: quietly switched — the two have **opposite optimisation directions**, so silently
#: swapping them would invert the search.
#:
#: A different question from ``engine``, and no longer an independent one. This
#: paragraph used to say the σ extractor "forces the gated physics internally"; T2.6b
#: (2026-08-09) retired that — the campaign objective now calls
#: :func:`~softae.analysis.eis.engine.analyze_spectrum` with ``engine`` left unset,
#: exactly as every GUI fit site does. So ``engine`` chooses **which physics** computes
#: σ, for the objective and the analysis tab and the auto-route alike; ``objective``
#: chooses **which metric** a campaign is scored on. Two keys, one σ.
OBJECTIVES = ("auto", "sigma", "mean_abs_z")

#: ``d log tanδ / d log f`` must be at least this negative for a spectrum to
#: contain parallel conduction at all. Positive slope ⇒ series parasitic ⇒ the
#: measurand is absent at every frequency (framework §3.5.1).
DEFAULT_TAND_SLOPE_MAX = -0.3
#: Allowed ``|d log C_app / d log f|`` before the dielectric reads as dispersive.
DEFAULT_CAP_FLATNESS_MAX = 0.15
#: Lin-KK residual threshold (%), above which a point is not K–K representable.
#: 3.0, not the specification's 1.0: this rig's per-point noise floor is ~3 %,
#: measured without any K–K basis at all (robust-scaled 3-point second difference of
#: log|Z| and phase on the log-f grid — 1.25–5.31 % over ten real spectra), and
#: reproduced spectrum by spectrum by the lin-KK median. A 1 % ceiling asks the gate
#: to resolve drift three times finer than the measurement resolves anything.
DEFAULT_KK_RESID_PCT = 3.0
#: μ ceiling for Lin-KK ladder order selection. See
#: :data:`softae.analysis.eis.kk.DEFAULT_KK_C`, which owns the reasoning.
DEFAULT_KK_C = 0.30
#: Runtime bound on Lin-KK ladder order. Conditioning is enforced by the μ floor in
#: :mod:`softae.analysis.eis.kk`, not by this number.
DEFAULT_KK_MAX_M = 50
#: Largest fraction of the band the low-f K–K truncation may remove before the
#: spectrum is rejected instead. See :mod:`softae.analysis.eis.kk` — the ladder fit is
#: global, so a small drift makes almost every point "fail", and unbounded truncation
#: would delete the spectrum while reporting that it had tidied a tail.
DEFAULT_KK_MAX_TRUNCATE_FRAC = 0.5
#: How close ``Re Z`` must sit to the model-free ``R`` to count as on the plateau (%).
DEFAULT_PLATEAU_TOL_PCT = 10.0
#: Fewer surviving points than this cannot support a 5-parameter fit.
DEFAULT_MIN_FIT_PTS = 8
#: Parameter relative standard error above which a value is not determined.
DEFAULT_MAX_REL_SE = 0.10
#: ``ρ(R_series, R_bulk)`` at or below this means the split is unidentifiable and
#: only the sum may be reported (framework §1.6).
DEFAULT_RHO_DEGENERATE = -0.95
#: Fractional distance to a box constraint counted as "pegged".
DEFAULT_BOUND_TOL = 1e-3
#: A resistive plateau narrower than this (decades) is an extrapolation.
DEFAULT_PLATEAU_MIN_DECADES = 0.5
#: Require ``tanδ ≥ this × phase-floor`` before σ is reported as a value.
DEFAULT_TAND_HEADROOM_MULT = 3.0
#: Fraction of blank points over range above which the open is unusable.
DEFAULT_BLANK_OVER_FRAC = 0.25
#: Density of Im-sign flips above which a blank is noise, not a measurement.
DEFAULT_BLANK_FLIP_FRAC = 0.30
#: Blank repeatability tolerance (%), used from E2 onward.
DEFAULT_BLANK_REPRO_PCT = 5.0
#: RMS residual (%) above which the model plainly does not describe the data. Far
#: above ``[quality] max_residual_pct``, which grades; this one catches catastrophe
#: (overhaul F11: fits with 10²–10³ % residuals still reporting a usable-looking R1).
DEFAULT_RESIDUAL_HARD_PCT = 100.0

#: Geometry-series slope validation (E5, framework §5.6.3). Imported from
#: :mod:`softae.analysis.eis.geometry_series`, which owns the reasoning; they are
#: re-exported here only so `[eis.gates]` stays the single parse point for
#: thresholds and no second config reader appears.
DEFAULT_GEOM_SIGMA_SPREAD_TOL = 0.25
DEFAULT_GEOM_SLOPE_EXPONENT_TOL = 0.15


@dataclass(frozen=True)
class GateSettings:
    """Thresholds for the admission and post-fit gates.

    Every one of these is an *engineering default from the specification*, chosen
    without reference to this rig's spectra. They are shipped so the gates can run
    and log; they are not shipped with authority to discard data. Review a
    campaign's worth of ``eis_gate_would_reject`` lines before enabling.
    """

    enabled: bool = False
    tand_slope_max: float = DEFAULT_TAND_SLOPE_MAX
    cap_flatness_max: float = DEFAULT_CAP_FLATNESS_MAX
    kk_resid_pct: float = DEFAULT_KK_RESID_PCT
    kk_c: float = DEFAULT_KK_C
    kk_max_M: int = DEFAULT_KK_MAX_M
    kk_max_truncate_frac: float = DEFAULT_KK_MAX_TRUNCATE_FRAC
    plateau_tol_pct: float = DEFAULT_PLATEAU_TOL_PCT
    min_fit_pts: int = DEFAULT_MIN_FIT_PTS
    max_rel_se: float = DEFAULT_MAX_REL_SE
    rho_degenerate: float = DEFAULT_RHO_DEGENERATE
    bound_tol: float = DEFAULT_BOUND_TOL
    plateau_min_decades: float = DEFAULT_PLATEAU_MIN_DECADES
    tand_headroom_mult: float = DEFAULT_TAND_HEADROOM_MULT
    blank_over_frac: float = DEFAULT_BLANK_OVER_FRAC
    blank_flip_frac: float = DEFAULT_BLANK_FLIP_FRAC
    blank_repro_pct: float = DEFAULT_BLANK_REPRO_PCT
    residual_hard_pct: float = DEFAULT_RESIDUAL_HARD_PCT
    geom_sigma_spread_tol: float = DEFAULT_GEOM_SIGMA_SPREAD_TOL
    geom_slope_exponent_tol: float = DEFAULT_GEOM_SLOPE_EXPONENT_TOL

    def describe(self) -> str:
        """One line an operator can sanity-check the settings against.

        ``rho_degenerate`` drives two different tests and the line says both, because
        naming only one would misdescribe whichever is omitted. ``gate_degeneracy`` is
        **two-sided on the magnitude** (``|ρ| ≥ |rho_degenerate|`` is degenerate), while
        ``engine_support._resolve_reported_resistance`` keeps the **one-sided** rule that
        chooses the reported resistance — a deliberate divergence documented in
        ``gate_degeneracy``. The gate's half reads ``abs()`` so the string cannot disagree
        with the gate if an operator writes the threshold positive.
        """
        if not self.enabled:
            return (
                "EIS gates observe only — every check runs and logs, nothing is "
                "removed."
            )
        return (
            f"EIS gates enforcing: tanδ slope ≤ {self.tand_slope_max:+.2f}, "
            f"≥ {self.min_fit_pts} surviving points, "
            f"degenerate at |ρ| ≥ {abs(self.rho_degenerate):.2f} "
            f"(engine reports sum-only below ρ = {self.rho_degenerate:+.2f})."
        )


@dataclass(frozen=True)
class FixtureSettings:
    """How much of the measured impedance belongs to the path rather than the cell.

    ``mode`` defaults to ``auto`` and that is the whole design: correction resolves
    from :meth:`CalibrationSet.capabilities`, so it is ``none`` until a short blank
    exists and ``series`` the moment one does. Nobody has to remember to flip a key
    after commissioning — and, more to the point, nobody can forget to flip it *back*
    after replacing a board, because staleness already drops the constants.

    ``auto`` never resolves to OSL. See :mod:`softae.analysis.eis.fixture`.
    """

    mode: str = "auto"
    fixture_id: str = "default"
    load_tolerance_pct: float = 5.0

    def describe(self) -> str:
        if self.mode == "none":
            return "fixture correction: off by configuration."
        if self.mode == "auto":
            return (f"fixture correction: auto — series once fixture "
                    f"'{self.fixture_id}' has a short blank, none until then.")
        return f"fixture correction: {self.mode} on fixture '{self.fixture_id}'."


@dataclass(frozen=True)
class EISSettings:
    """Engine selection plus its gate thresholds."""

    engine: str = "legacy"
    objective: str = "auto"
    gates: GateSettings = GateSettings()
    fixture: FixtureSettings = FixtureSettings()

    @property
    def is_gated(self) -> bool:
        """True when the new physics layer decides anything at all."""
        return self.engine == "gated"

    def describe(self) -> str:
        obj = {"sigma": "maximising conductivity",
               "mean_abs_z": "minimising mean |Z|",
               "auto": "objective chosen per campaign mode"}[self.objective]
        if not self.is_gated:
            return (f"EIS analysis: legacy engine (R1 only, no gates, no "
                    f"covariance, no fixture correction); campaigns {obj}.")
        return (f"EIS analysis: gated engine, campaigns {obj}. "
                f"{self.gates.describe()} {self.fixture.describe()}")


def eis_settings(config: dict[str, Any] | None = None) -> EISSettings:
    """Read ``[eis]`` and ``[eis.gates]`` — the single parse point for both.

    An unknown engine name falls back to ``legacy`` with a warning rather than
    raising: a typo in a config file must not stop a campaign that would otherwise
    have run exactly as it always has.
    """
    if config is None:
        try:
            from softae.config import loader

            config = loader.load().get("eis", {}) or {}
        except Exception:
            config = {}

    gates_cfg = config.get("gates", {}) or {}
    fixture_cfg = config.get("fixture", {}) or {}

    def _f(key: str, default: float) -> float:
        try:
            return float(gates_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    def _i(key: str, default: int) -> int:
        try:
            return int(gates_cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    engine = str(config.get("engine", "legacy")).strip().lower()
    if engine not in ENGINES:
        logger.warning(
            "eis_engine_unknown", engine=engine, known=ENGINES,
            msg="falling back to the legacy engine",
        )
        engine = "legacy"

    objective = str(config.get("objective", "auto")).strip().lower()
    if objective not in OBJECTIVES:
        logger.warning(
            "eis_objective_unknown", objective=objective, known=OBJECTIVES,
            msg="falling back to automatic selection",
        )
        objective = "auto"

    from softae.analysis.eis.fixture import CORRECTION_MODES

    fixture_mode = str(fixture_cfg.get("mode", "auto")).strip().lower()
    if fixture_mode not in CORRECTION_MODES:
        logger.warning(
            "eis_fixture_mode_unknown", mode=fixture_mode, known=CORRECTION_MODES,
            msg="falling back to automatic selection",
        )
        fixture_mode = "auto"

    try:
        load_tol = float(fixture_cfg.get("load_tolerance_pct", 5.0))
    except (TypeError, ValueError):
        load_tol = 5.0

    return EISSettings(
        engine=engine,
        objective=objective,
        fixture=FixtureSettings(
            mode=fixture_mode,
            fixture_id=str(fixture_cfg.get("fixture_id", "default")).strip() or "default",
            load_tolerance_pct=load_tol,
        ),
        gates=GateSettings(
            enabled=bool(gates_cfg.get("enabled", False)),
            tand_slope_max=_f("tand_slope_max", DEFAULT_TAND_SLOPE_MAX),
            cap_flatness_max=_f("cap_flatness_max", DEFAULT_CAP_FLATNESS_MAX),
            kk_resid_pct=_f("kk_resid_pct", DEFAULT_KK_RESID_PCT),
            kk_c=_f("kk_c", DEFAULT_KK_C),
            kk_max_M=_i("kk_max_M", DEFAULT_KK_MAX_M),
            kk_max_truncate_frac=_f(
                "kk_max_truncate_frac", DEFAULT_KK_MAX_TRUNCATE_FRAC),
            plateau_tol_pct=_f("plateau_tol_pct", DEFAULT_PLATEAU_TOL_PCT),
            min_fit_pts=_i("min_fit_pts", DEFAULT_MIN_FIT_PTS),
            max_rel_se=_f("max_rel_se", DEFAULT_MAX_REL_SE),
            rho_degenerate=_f("rho_degenerate", DEFAULT_RHO_DEGENERATE),
            bound_tol=_f("bound_tol", DEFAULT_BOUND_TOL),
            plateau_min_decades=_f(
                "plateau_min_decades", DEFAULT_PLATEAU_MIN_DECADES),
            tand_headroom_mult=_f(
                "tand_headroom_mult", DEFAULT_TAND_HEADROOM_MULT),
            blank_over_frac=_f("blank_over_frac", DEFAULT_BLANK_OVER_FRAC),
            blank_flip_frac=_f("blank_flip_frac", DEFAULT_BLANK_FLIP_FRAC),
            blank_repro_pct=_f("blank_repro_pct", DEFAULT_BLANK_REPRO_PCT),
            residual_hard_pct=_f("residual_hard_pct", DEFAULT_RESIDUAL_HARD_PCT),
            geom_sigma_spread_tol=_f(
                "geom_sigma_spread_tol", DEFAULT_GEOM_SIGMA_SPREAD_TOL),
            geom_slope_exponent_tol=_f(
                "geom_slope_exponent_tol", DEFAULT_GEOM_SLOPE_EXPONENT_TOL),
        ),
    )
