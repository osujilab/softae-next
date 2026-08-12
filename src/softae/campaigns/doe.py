"""Design-of-experiments / composition-sweep generator.

Ported forward (softae idiom) from the ``experiment_manager`` prototype's
``ParamScale`` + ``Experiment`` model.  This is softae-next's phase-diagram
sampler: it turns a small, **user-authored** set of named parameter axes into an
enumerated candidate pool for the pooled BO optimizer.

Two axis kinds are recognised by name:

* **Composition axes** ``x_<component>`` — fractions of each stock in the
  mixture.  For an ``n``-component system the design supplies ``n-1`` free
  ``x_`` axes and the final component takes the complement (sum-to-one simplex),
  so only feasible compositions (all fractions in ``[0, 1]``) are emitted.
* **Environment axes** — anything else (``temperature_C``, ``rh_pct``, …),
  swept as an independent Cartesian factor.

The axis set is deliberately open: add a new ``x_<component>`` or environment
axis and it flows through without code changes.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

from softae.campaigns.config import BOCampaignConfig
from softae.core.formulation import simplex_fractions
from softae.errors import CampaignError

_SCALE_TYPES = ("linear", "log", "pointwise", "custom")
_COMPOSITION_PREFIX = "x_"


@dataclass
class ParamScale:
    """One named axis of a design sweep.

    ``scale_type`` is ``linear`` / ``log`` (need ``start``, ``stop``, ``steps``)
    or ``pointwise`` / ``custom`` (need explicit ``points``).
    """

    name: str
    scale_type: str = "linear"
    start: float | None = None
    stop: float | None = None
    steps: int | None = None
    points: list[float] | None = None

    def values(self) -> list[float]:
        if self.scale_type == "linear":
            if self.start is None or self.stop is None or not self.steps:
                raise CampaignError(f"linear axis '{self.name}' needs start, stop, steps")
            n = int(self.steps)
            if n == 1:
                return [float(self.start)]
            step = (self.stop - self.start) / (n - 1)
            return [float(self.start + i * step) for i in range(n)]
        if self.scale_type == "log":
            if self.start is None or self.stop is None or not self.steps:
                raise CampaignError(f"log axis '{self.name}' needs start, stop (>0), steps")
            if self.start <= 0 or self.stop <= 0:
                raise CampaignError(f"log axis '{self.name}' requires positive bounds")
            n = int(self.steps)
            lo, hi = math.log10(self.start), math.log10(self.stop)
            if n == 1:
                return [float(self.start)]
            step = (hi - lo) / (n - 1)
            return [float(10 ** (lo + i * step)) for i in range(n)]
        if self.scale_type in ("pointwise", "custom"):
            if not self.points:
                return []
            return sorted(float(x) for x in self.points)
        raise CampaignError(f"unknown scale_type '{self.scale_type}' for axis '{self.name}'")

    @property
    def is_composition(self) -> bool:
        return self.name.startswith(_COMPOSITION_PREFIX)

    @property
    def component(self) -> str:
        """Component name a composition axis refers to (``x_water`` -> ``water``)."""
        return self.name[len(_COMPOSITION_PREFIX):] if self.is_composition else self.name


@dataclass
class ExperimentDesign:
    """A user-authored composition/environment sweep over a stock set."""

    name: str = ""
    components: list[str] = field(default_factory=list)  # ordered stock/solution ids
    param_scales: list[ParamScale] = field(default_factory=list)
    target_deposition_uL: float = 0.0
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ── serialisation (prototype-compatible Experiment JSON) ──────────────

    @classmethod
    def from_json(cls, text: str) -> "ExperimentDesign":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentDesign":
        scales = [ParamScale(**{k: v for k, v in p.items() if k in _SCALE_FIELDS})
                  for p in d.get("param_scales", [])]
        # ``components`` in the prototype JSON is a list of {solution_id, role}.
        comps: list[str] = []
        for c in d.get("components", []):
            if isinstance(c, dict) and "solution_id" in c:
                comps.append(c["solution_id"])
            elif isinstance(c, str):
                comps.append(c)
        return cls(
            name=d.get("name", d.get("id", "")),
            components=comps,
            param_scales=scales,
            target_deposition_uL=float(d.get("target_deposition_uL", 0.0)),
            notes=d.get("notes", ""),
            meta=d.get("meta", {}) or {},
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "components": [{"solution_id": c} for c in self.components],
                "param_scales": [dataclasses.asdict(p) for p in self.param_scales],
                "target_deposition_uL": self.target_deposition_uL,
                "notes": self.notes,
                "meta": self.meta,
            },
            indent=2,
        )

    # ── axis views ────────────────────────────────────────────────────────

    def composition_axes(self) -> list[ParamScale]:
        return [p for p in self.param_scales if p.is_composition]

    def environment_axes(self) -> list[ParamScale]:
        return [p for p in self.param_scales if not p.is_composition]

    def resolve_component(self, hint: str) -> str:
        """Map a composition-axis suffix to the actual component name.

        Case-insensitive exact match first (``x_water`` -> ``Water``), then a
        prefix match; falls back to the hint unchanged.
        """
        for c in self.components:
            if c.lower() == hint.lower():
                return c
        for c in self.components:
            if c.lower().startswith(hint.lower()):
                return c
        return hint

    def complement_component(self) -> str | None:
        """The component whose fraction is not a free axis (takes the complement)."""
        free = {self.resolve_component(p.component) for p in self.composition_axes()}
        remaining = [c for c in self.components if c not in free]
        return remaining[-1] if remaining else None

    # ── pool enumeration ──────────────────────────────────────────────────

    def candidate_pool(self) -> list[dict[str, float]]:
        """Enumerate the feasible candidate pool as a list of parameter dicts.

        Each dict carries ``x_<component>`` fractions summing to 1.0 (one per
        listed component) plus every environment axis value.  Cartesian points
        whose free composition fractions fall outside the simplex are dropped.
        """
        comp_axes = self.composition_axes()
        env_axes = self.environment_axes()

        # bind each composition axis to an actual component name
        comp_names = [self.resolve_component(p.component) for p in comp_axes]
        comp_value_lists = [p.values() for p in comp_axes]

        env_names = [p.name for p in env_axes]
        env_value_lists = [p.values() for p in env_axes]

        pool: list[dict[str, float]] = []
        comp_product = itertools.product(*comp_value_lists) if comp_value_lists else [()]
        env_product = list(itertools.product(*env_value_lists)) if env_value_lists else [()]

        for comp_combo in comp_product:
            free = dict(zip(comp_names, comp_combo))
            # feasibility: every free fraction in [0,1] and their sum <= 1
            if any(v < 0.0 or v > 1.0 for v in free.values()):
                continue
            if sum(free.values()) > 1.0 + 1e-9:
                continue
            if self.components:
                fractions = simplex_fractions(self.components, free)
            else:
                fractions = free
            for env_combo in env_product:
                point: dict[str, float] = {
                    f"{_COMPOSITION_PREFIX}{name}": val
                    for name, val in fractions.items()
                }
                point.update(dict(zip(env_names, env_combo)))
                pool.append(point)
        return pool


_SCALE_FIELDS = {f.name for f in dataclasses.fields(ParamScale)}


# ---------------------------------------------------------------------------
# Adapter: ExperimentDesign -> (BOCampaignConfig, candidate pool)
# ---------------------------------------------------------------------------

def design_to_campaign(
    design: ExperimentDesign,
) -> tuple[BOCampaignConfig, list[dict[str, float]]]:
    """Translate an authored design into a campaign config + candidate pool.

    Keeps the two stacks coexisting rather than merged: the Experiment JSON stays
    the user-facing authoring format, and this adapter emits softae-next's
    :class:`BOCampaignConfig` (optimizer/objective/stopping knobs, read from
    ``design.meta`` with sensible defaults) alongside the enumerated pool that a
    real closed-loop campaign (WS2) will search.
    """
    pool = design.candidate_pool()
    pool_size = len(pool)
    meta = design.meta or {}

    # n_initial must stay strictly below the pool size.
    requested_initial = int(meta.get("n_initial", 5))
    n_initial = max(1, min(requested_initial, pool_size - 1)) if pool_size > 1 else 1

    config = BOCampaignConfig(
        backend=meta.get("backend", "sklearn"),
        acquisition=meta.get("acquisition", "ucb"),
        objective_direction=meta.get("objective_direction", "maximize"),
        transform=meta.get("transform", "log10_sigma"),
        temperature_objective=meta.get("temperature_objective", "none"),
        n_initial=n_initial,
        seed=int(meta.get("seed", 0)),
        max_steps=int(meta.get("max_steps", pool_size)) if pool_size else None,
        annotation=design.name or design.notes,
    )
    if pool_size:
        config.validate(pool_size=pool_size)
    return config, pool

