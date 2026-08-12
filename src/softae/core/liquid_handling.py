"""Liquid-handling correction helpers.

This module is pure Python and intentionally independent from GUI/executor code
so the same correction logic can be reused by multiple entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math


@dataclass(frozen=True)
class LinePhysicsConfig:
    line_id: int
    cracking_kpa_per_valve: float
    compliance_uL_per_kpa: float
    alpha_base: float
    viscosity_mpas: float = 1.0


@dataclass(frozen=True)
class SystemPhysicsConfig:
    valves_in_series: int = 2
    beta: float = 0.30
    eta_ref_mpas: float = 1.0
    alpha_growth_per_run: float = 0.0


@dataclass(frozen=True)
class CorrectionInput:
    target_uL: float
    line_id: int
    run_index: int


@dataclass(frozen=True)
class CorrectionResult:
    target_uL: float
    dead_uL: float
    commanded_uL: float


class LiquidHandlingCorrector:
    """Physics-inspired dead-volume correction model."""

    @staticmethod
    def _validate_finite(name: str, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    def _effective_cracking_pressure(
        self,
        line_cfg: LinePhysicsConfig,
        sys_cfg: SystemPhysicsConfig,
    ) -> float:
        eta_ref = max(sys_cfg.eta_ref_mpas, 1e-9)
        viscosity = max(line_cfg.viscosity_mpas, 1e-9)
        eta_ratio = viscosity / eta_ref
        return max(line_cfg.cracking_kpa_per_valve, 0.0) * sys_cfg.valves_in_series * (eta_ratio ** sys_cfg.beta)

    def _alpha_n(self, line_cfg: LinePhysicsConfig, sys_cfg: SystemPhysicsConfig, run_index: int) -> float:
        alpha = line_cfg.alpha_base + sys_cfg.alpha_growth_per_run * (run_index - 1)
        return min(max(alpha, 0.0), 1.0)

    def _dead_volume(self, line_cfg: LinePhysicsConfig, sys_cfg: SystemPhysicsConfig, run_index: int) -> float:
        cracking_eff = self._effective_cracking_pressure(line_cfg, sys_cfg)
        alpha = self._alpha_n(line_cfg, sys_cfg, run_index)
        return max(0.0, max(line_cfg.compliance_uL_per_kpa, 0.0) * cracking_eff * (1.0 - alpha))

    def corrected_command(
        self,
        inp: CorrectionInput,
        line_cfg: LinePhysicsConfig,
        sys_cfg: SystemPhysicsConfig,
    ) -> CorrectionResult:
        self._validate_finite("target_uL", inp.target_uL)
        self._validate_finite("cracking_kpa_per_valve", line_cfg.cracking_kpa_per_valve)
        self._validate_finite("compliance_uL_per_kpa", line_cfg.compliance_uL_per_kpa)
        self._validate_finite("alpha_base", line_cfg.alpha_base)
        self._validate_finite("viscosity_mpas", line_cfg.viscosity_mpas)
        self._validate_finite("beta", sys_cfg.beta)
        self._validate_finite("eta_ref_mpas", sys_cfg.eta_ref_mpas)
        self._validate_finite("alpha_growth_per_run", sys_cfg.alpha_growth_per_run)

        if inp.run_index < 1:
            raise ValueError("run_index must be >= 1")
        if sys_cfg.valves_in_series < 1:
            raise ValueError("valves_in_series must be >= 1")

        target = max(0.0, float(inp.target_uL))
        if target <= 0.0:
            return CorrectionResult(target_uL=0.0, dead_uL=0.0, commanded_uL=0.0)

        dead_uL = self._dead_volume(line_cfg, sys_cfg, inp.run_index)

        commanded = max(0.0, target + dead_uL)
        return CorrectionResult(target_uL=target, dead_uL=dead_uL, commanded_uL=commanded)

    def prime_volume(
        self,
        line_cfg: LinePhysicsConfig,
        sys_cfg: SystemPhysicsConfig,
        margin: float = 1.2,
    ) -> float:
        """Return a recommended prime volume for a line starting from rest."""
        self._validate_finite("margin", margin)
        if margin < 1.0:
            raise ValueError("margin must be >= 1.0")
        # Prime from rest assumes no residual pressure, i.e. alpha=0.
        cracking_eff = self._effective_cracking_pressure(line_cfg, sys_cfg)
        base = max(0.0, max(line_cfg.compliance_uL_per_kpa, 0.0) * cracking_eff)
        return base * margin

    def corrected_multi(
        self,
        targets_uL: list[float],
        run_index: int,
        line_cfg_by_pump: dict[int, LinePhysicsConfig],
        sys_cfg: SystemPhysicsConfig,
    ) -> tuple[list[float], float]:
        """Correct an arbitrary number of per-pump targets in one call.

        ``targets_uL[i]`` is corrected against ``line_cfg_by_pump[i]``.  Returns
        the per-pump commanded volumes (same order/length as ``targets_uL``) and
        their sum.  This is the N-pump generalisation of :meth:`corrected_pair`.
        """
        commanded: list[float] = []
        for pump_id, target in enumerate(targets_uL):
            line = line_cfg_by_pump[pump_id]
            commanded.append(
                self.corrected_command(
                    CorrectionInput(
                        target_uL=target, line_id=line.line_id, run_index=run_index
                    ),
                    line,
                    sys_cfg,
                ).commanded_uL
            )
        return commanded, sum(commanded)

    def corrected_pair(
        self,
        pump0_target_uL: float,
        pump1_target_uL: float,
        run_index: int,
        line_cfg_by_pump: dict[int, LinePhysicsConfig],
        sys_cfg: SystemPhysicsConfig,
    ) -> tuple[float, float, float]:
        (p0, p1), total = self.corrected_multi(
            [pump0_target_uL, pump1_target_uL],
            run_index,
            line_cfg_by_pump,
            sys_cfg,
        )
        return p0, p1, total


# ── Config-gated correction at the hardware boundary (P2.2) ──────────────────

@dataclass(frozen=True)
class DeadVolumeCorrection:
    """Config → physics marshalling plus the delivered→commanded conversion.

    Three surfaces each rebuilt this from ``[liquid_handling]`` by hand (the HT
    tab, the manual-control tab, and — by omission — the campaign path, which
    did not correct at all). That meant **the same nominal formulation could
    reach the hardware as different physical volumes depending on which surface
    launched it.** Dormant in practice, since correction is off by default and
    has never been used on the rig, but a trap worth closing while unifying.

    Layering (decided): the solver stays in composition space and emits *desired
    delivered* volumes — what actually lands in the well. This converts those to
    *commanded* volumes at the hardware boundary, and nowhere else. Keeping the
    conversion this late matters for correctness, not tidiness: the deposition
    twin, well-capacity/overflow checks, and thickness targets must all reason
    about delivered volume. Folding dead volume in earlier would compare
    commanded volume against well capacity (over-conservative) and overestimate
    thickness.
    """

    enabled: bool = False
    sys_cfg: SystemPhysicsConfig = SystemPhysicsConfig()
    line_cfg_by_pump: dict[int, LinePhysicsConfig] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.line_cfg_by_pump is None:
            object.__setattr__(self, "line_cfg_by_pump", {})

    @classmethod
    def from_config(
        cls,
        pump_ids,
        cfg: dict | None = None,
        *,
        enabled: bool | None = None,
    ) -> "DeadVolumeCorrection":
        """Build from ``[liquid_handling]`` — the single place this is parsed.

        *enabled* overrides the config flag (the manual tab has a per-command
        checkbox); ``None`` honours the config.
        """
        from softae.config.loader import liquid_handling_config, liquid_line_for_pump

        cfg = liquid_handling_config() if cfg is None else cfg
        sys_cfg = SystemPhysicsConfig(
            valves_in_series=int(cfg.get("valves_in_series", 2)),
            beta=float(cfg.get("beta", 0.30)),
            eta_ref_mpas=float(cfg.get("eta_ref_mpas", 1.0)),
            alpha_growth_per_run=float(cfg.get("alpha_growth_per_run", 0.0)),
        )
        lines = cfg.get("line", {})

        def _line(line_id: int) -> LinePhysicsConfig:
            lc = lines.get(str(line_id), lines.get(line_id, {}))
            return LinePhysicsConfig(
                line_id=line_id,
                cracking_kpa_per_valve=float(lc.get("cracking_kpa_per_valve", 8.0)),
                compliance_uL_per_kpa=float(lc.get("compliance_uL_per_kpa", 0.55)),
                alpha_base=float(lc.get("alpha_base", 0.20)),
                viscosity_mpas=float(lc.get("viscosity_mpas", 1.0)),
            )

        return cls(
            enabled=bool(cfg.get("enabled", False)) if enabled is None else bool(enabled),
            sys_cfg=sys_cfg,
            line_cfg_by_pump={
                int(p): _line(liquid_line_for_pump(int(p))) for p in pump_ids
            },
        )

    def commanded(self, targets_uL, run_index: int = 1) -> list[float]:
        """Delivered → commanded for one dispense. Disabled → returned unchanged."""
        targets = [max(0.0, float(v)) for v in targets_uL]
        if not self.enabled or not self.line_cfg_by_pump:
            return targets
        commanded, _ = LiquidHandlingCorrector().corrected_multi(
            targets, max(1, int(run_index)), self.line_cfg_by_pump, self.sys_cfg,
        )
        return commanded

    def apply_by_channel(
        self, volumes_by_channel: "Mapping[int, Sequence[float]]", channels
    ) -> dict[int, list[float]]:
        """Correct a whole per-channel formulation map.

        ``run_index`` is the channel's 1-based position in *channels*, which is
        how dead volume grows across successive dispenses within one run — the
        same convention the HT path used before this moved.
        """
        out: dict[int, list[float]] = {}
        for i, ch in enumerate(channels, start=1):
            vols = volumes_by_channel.get(int(ch))
            if vols is None:
                continue
            out[int(ch)] = self.commanded(vols, run_index=i)
        return out
