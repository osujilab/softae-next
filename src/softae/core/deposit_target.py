"""What fixes a formulation's *scale*: a deposit volume, or a film thickness.

Every composition solve needs exactly one row that says how big the cast is.
Until now that row was always ``TotalDepositTarget(target_deposition_uL)`` — a
volume the operator worked out by hand from a thickness they actually wanted.
This module makes the thickness itself declarable, on every path, and keeps the
three things a thickness needs in one place so the surfaces cannot disagree:

**the area** — :func:`resolve_area_mm2` is the single authority, always
``deposit_area_mm2(resolve_pcb(name)[1])``.  A board that declares no usable
area (a sessile board, whose wetted footprint is an *observation*) gets
:class:`ThicknessUnresolvable`, never a guess.

**the drying model** — the dry basis is coupled to ``[deposition]
evaporation_pct`` through :func:`softae.core.deposition.evaporation_pct`, the
same single parse point the deposition twin reads.  Agreement is by
construction rather than by a check that could rot.  Writing ``r = 1 − e/100``
for the retained carrier fraction, the twin's own arithmetic expands in the
per-stock volumes ``v_i`` as::

    final_volume = Σ v_i · [ depf_i + r·(1 − depf_i) ]

so a dry thickness target is one linear row with coefficient
``a_i = depf_i + r·(1 − depf_i)``.  It reduces to ``depf_i`` at ``e = 100``
(today's row, unchanged) and to ``1.0`` at ``e = 0`` (the wet row — nothing
evaporates, so dry *is* wet).

**what a thickness is worth** — a *nominal dense-film geometric estimate*.
Volumes are additive; no dry-film density or porosity is modelled, so a porous
real film is thicker.  Carrier loss is one scalar applied uniformly to every
stock's carrier — there is no per-solvent volatility term.  Sound as a control
target and as a relative measure across a campaign; not a metrology claim.

Spec: ``docs/SubAgent docs/thickness_target_all_paths.md`` (W1).  Standing rules:
``SUBAGENT_RULES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from softae.core.formulation import (
        ChemicalCatalog,
        FormulationTarget,
        Solution,
    )

#: What a declaration may fix.
KINDS = ("volume", "thickness")
#: Which volume a thickness refers to.
BASES = ("dry", "wet")


class ThicknessUnresolvable(ValueError):
    """A thickness was asked for where it cannot be turned into a volume.

    Two causes, both of which must refuse rather than substitute a number:

    * **no deposit area** — the board declares none (``cast_confinement =
      "sessile"``, or nothing usable at all), and an invented area silently
      corrupts every thickness derived from it;
    * **no composition solve** — the legacy-identity path searches per-pump
      volumes directly, so there is no stock identity and no ``dep_fraction``.
      Every coefficient the dry row needs is undefined, and the wet row
      (``1.0``) would "work" while quietly ignoring the parameters being
      searched.
    """


@dataclass(frozen=True)
class DepositTargetSpec:
    """What a spec file or a panel *declares* about the scale of a cast.

    Deliberately inert: it carries no area and no drying assumption, so it can
    be written to a file, round-tripped, and compared.  :func:`scale_target`
    is what turns it into a solver row, and that is where the area and the
    evaporation setting are resolved — one place, one answer.
    """

    kind: str            # "volume" | "thickness"
    value: float         # µL, or µm
    basis: str = "dry"   # thickness only: "dry" | "wet"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"DepositTargetSpec kind must be one of {KINDS}, got {self.kind!r}"
            )
        if self.basis not in BASES:
            raise ValueError(
                f"DepositTargetSpec basis must be one of {BASES}, got {self.basis!r}"
            )
        if not float(self.value) > 0.0:
            raise ValueError(
                f"DepositTargetSpec value must be positive, got {self.value!r} — "
                f"a zero or negative scale is not a cast"
            )

    @property
    def is_thickness(self) -> bool:
        return self.kind == "thickness"


def _configured_evaporation_pct() -> float:
    """The one parse point, reached by attribute so a test can move both users."""
    from softae.core import deposition

    return deposition.evaporation_pct()


def resolve_area_mm2(pcb_name: str | None) -> float | None:
    """The deposit area a cast on *pcb_name* covers (mm²), or ``None``.

    THE area authority for every thickness on every path.  ``None`` means the
    board declares nothing usable and the caller must treat a thickness as
    unavailable — see :func:`softae.core.geometry.deposit_area_mm2` for why a
    sessile board legitimately answers that.
    """
    from softae.core.deposition_steps import resolve_pcb
    from softae.core.geometry import deposit_area_mm2

    _name, pcb = resolve_pcb(pcb_name)
    return deposit_area_mm2(pcb)


def scale_target(
    dt: DepositTargetSpec | None,
    *,
    fallback_uL: float | None,
    pcb_name: str | None = None,
    evaporation_pct: float | None = None,
    dep_fractions: Sequence[float] | None = None,
) -> "FormulationTarget":
    """The one row that fixes a formulation's scale.

    ``dt is None`` reproduces today's behaviour exactly —
    ``TotalDepositTarget(fallback_uL)`` — so every existing caller is
    unchanged by construction.

    ``evaporation_pct`` pins the drying assumption for this solve.  Left
    ``None`` it is read from :func:`softae.core.deposition.evaporation_pct`,
    the same call the twin makes; a campaign that wants trial 1 and trial 12 to
    cast the same volume for the same composition captures it once at run start
    and passes it here.

    ``dep_fractions`` is the per-stock dried fraction vector the solve will
    use, when the caller has one.  It is not needed to build the row (the
    solver recomputes it) and exists to make two refusals possible: an *empty*
    vector says the caller has no composition solve at all (D4), and an
    all-carrier vector says no dried film can be produced.  ``None`` means the
    caller did not offer it and no such check is made.

    Raises :class:`ThicknessUnresolvable` rather than guessing an area or a
    dried fraction.
    """
    from softae.core.formulation import ThicknessTarget, TotalDepositTarget

    if dt is None:
        if fallback_uL is None:
            raise ValueError(
                "no deposit target and no fallback volume — a formulation with "
                "no scale row is under-determined, and a substituted default "
                "would be a scale nobody declared"
            )
        return TotalDepositTarget(float(fallback_uL))

    if not dt.is_thickness:
        return TotalDepositTarget(float(dt.value))

    if dep_fractions is not None and not len(dep_fractions):
        raise ThicknessUnresolvable(
            "no composition solve, so no dried fraction, so no thickness "
            "inverse: this path searches per-pump volumes directly and has no "
            "stock identity. Declare a deposit volume instead, or give the "
            "campaign a composition context."
        )

    e = float(evaporation_pct if evaporation_pct is not None
              else _configured_evaporation_pct())

    area = resolve_area_mm2(pcb_name)
    if not area:
        raise ThicknessUnresolvable(
            f"board {pcb_name!r} declares no deposit area, so a thickness "
            f"cannot be turned into a volume. A sessile board's wetted "
            f"footprint is an observation, not a geometry — declare "
            f"'deposit_area_mm2' from a measured footprint, or target a volume."
        )

    if dt.basis == "dry" and dep_fractions is not None:
        r = 1.0 - e / 100.0
        if not any(f + r * (1.0 - f) > 0.0 for f in (float(x) for x in dep_fractions)):
            raise ThicknessUnresolvable(
                f"every stock is pure carrier and {e:g}% of it is lost, so no "
                f"volume leaves a dried film — a dry thickness target has no "
                f"finite solution here"
            )

    return ThicknessTarget(
        float(dt.value), area_mm2=float(area), basis=dt.basis, evaporation_pct=e
    )


def scale_for_thickness(
    per_stock_uL: dict[str, float],
    stocks: dict[str, "Solution"],
    catalog: "ChemicalCatalog",
    *,
    target_um: float,
    basis: str = "dry",
    pcb_name: str | None = None,
    evaporation_pct: float | None = None,
) -> dict[str, float]:
    """Rescale an existing volume vector so its film hits ``target_um``.

    The HT path's inverse: the operator already has a volume per stock (typed,
    pasted, or pre-filled) and wants the same *composition* at a different
    thickness.

    **Closed form, not iterative.**  ``elution_from_stock_volumes`` is linear in
    the per-stock volumes and every term of ``simulate_well_deposition`` scales
    with the dispensed volume, so ``thickness(k·v) == k·thickness(v)`` exactly.
    One forward twin call gives ``k = target_um / thickness(v)``, and the twin's
    own refusals are reused verbatim rather than reimplemented.
    """
    from softae.core.deposition import WellGeometry, simulate_well_deposition
    from softae.core.deposition_steps import resolve_pcb
    from softae.core.formulation import elution_from_stock_volumes
    from softae.core.geometry import well_capacity_uL

    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    if not float(target_um) > 0.0:
        raise ValueError(f"target_um must be positive, got {target_um!r}")

    area = resolve_area_mm2(pcb_name)
    if not area:
        raise ThicknessUnresolvable(
            f"board {pcb_name!r} declares no deposit area, so these volumes "
            f"cannot be rescaled to a thickness"
        )

    elution = elution_from_stock_volumes(per_stock_uL, stocks, catalog)
    if elution.grand_total_uL <= 0:
        raise ThicknessUnresolvable(
            "nothing is dispensed, so no rescale reaches any thickness"
        )

    e = float(evaporation_pct if evaporation_pct is not None
              else _configured_evaporation_pct())
    _name, pcb = resolve_pcb(pcb_name)
    # Capacity does not enter a thickness (it only scales the fill fractions we
    # discard here), so a board that declares none is given the volume already
    # in hand rather than an invented brim.
    capacity = well_capacity_uL(pcb) or elution.grand_total_uL
    twin = simulate_well_deposition(
        elution, WellGeometry.from_board(float(area), float(capacity)), e
    )

    current = twin.final_thickness_um if basis == "dry" else twin.wet_thickness_um
    if current <= 0:
        raise ThicknessUnresolvable(
            f"these stocks leave a {basis} film of zero thickness at {e:g}% "
            f"carrier loss, so no finite rescale reaches {target_um} µm — an "
            f"all-carrier mixture has no dried film to thicken"
        )

    k = float(target_um) / current
    return {name: float(vol) * k for name, vol in per_stock_uL.items()}


__all__ = [
    "BASES",
    "KINDS",
    "DepositTargetSpec",
    "ThicknessUnresolvable",
    "resolve_area_mm2",
    "scale_for_thickness",
    "scale_target",
]
