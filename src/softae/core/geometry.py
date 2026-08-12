"""Electrode-grid geometry — the channel → stage-position mapping.

Pure, GUI-free functions that turn a PCB layout (grid + spacing) plus a
deposition origin into per-electrode stage coordinates.  This is the single
source of truth for "where is channel N on the stage" and is shared by the HT
Experiment tab (which injects the per-channel position into deposition steps at
run time) and the position-map widget (which draws it).

Electrode 1 (channel 1) sits at the origin — the upper-left, most-positive
corner; labels advance row-major with both X and Y *decreasing* as the index
grows, matching the legacy ``dropArray_generate`` ordering and the physical
board layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Slack (µL) below which a total counts as *at* a boundary, not past it — the
#: same tolerance :mod:`softae.core.overflow` and the formulation solver use, so
#: the boundary convention agrees everywhere it is asked about.
_VOL_TOL_UL = 1e-9


def electrode_count(pcb_config: dict[str, Any]) -> int:
    """Number of addressable electrodes on a PCB.

    Uses the explicit ``channels`` key when present, else the ``grid``
    product (``rows * cols``); defaults to a 4×4 board (16).  This is the single
    source of truth for a board's electrode capacity.
    """
    n = pcb_config.get("channels")
    if n is not None:
        return int(n)
    rows, cols = pcb_config.get("grid", [4, 4])
    return int(rows) * int(cols)


def well_void_uL(pcb_config: dict[str, Any]) -> float | None:
    """The well's **void volume alone** — brim-full, nothing above it.

    ``π(d/2)² × depth``, with 1 mm³ = 1 µL exactly. ``None`` when the board does
    not declare both dimensions (or has no well at all).

    Kept separate from :func:`elution_capacity_uL` because they answer different
    questions and only one of them is a hard limit. The void is a *geometric*
    fact about the walls; what a well can actually hold depends on the wetting
    behaviour of the material those walls are made from.
    """
    area = _well_area_mm2(pcb_config)
    depth = _positive_float(pcb_config.get("well_depth_mm"))
    if area is None or depth is None:
        return None
    return area * depth


def brim_cap_uL(radius_mm: float, height_mm: float) -> float:
    """Volume of a spherical cap of *height_mm* standing on a circle of *radius_mm*.

    ``V = πh(3r² + h²)/6`` — the meniscus a non-wetting brim can support above
    a full well. A PTFE wall pins the contact line at the sharp brim edge (Gibbs
    pinning), so the *apparent* angle grows well past the material's intrinsic
    ~110° as fluid is added, and the bead sits proud of the rim instead of
    spilling. That head of fluid is real capacity and treating it as overflow
    needlessly narrows the range of formulations a board can take.

    At ``h == r`` the cap is exactly a hemisphere — the geometric ceiling, the
    largest bead a pinned circular contact line can hold in any material. See
    :func:`hemisphere_cap_uL`.
    """
    r = float(radius_mm)
    h = max(0.0, float(height_mm))
    return math.pi * h * (3.0 * r * r + h * h) / 6.0


def hemisphere_cap_uL(pcb_config: dict[str, Any]) -> float | None:
    """The largest bead this well's brim could ever hold — ``h = r``, a hemisphere.

    A ceiling, **not** a target. Reaching it needs a 180° apparent angle; a real
    bead depins and spills well before that, and this says nothing about how a
    board moves, how level it is, or what the dispenser does on approach. It is
    here so a configured ``permitted_overfill_mm`` can be read against the
    physical maximum rather than against nothing.
    """
    d = _positive_float(pcb_config.get("well_diameter_mm"))
    return None if d is None else brim_cap_uL(d / 2.0, d / 2.0)


def permitted_overfill_mm(pcb_config: dict[str, Any]) -> float:
    """Bead height above the brim this board is allowed to carry (mm).

    Board key first, then ``[deposition] permitted_overfill_mm``, then **0.0**.

    Zero is the deliberate default: it makes the brim the limit, which is what
    every board did before this existed, so declaring nothing changes nothing.
    Overfill is permission to exceed a physical boundary and has to be granted
    explicitly, per board — it depends on the wall material, and only the person
    who built the wells knows what it is made of.
    """
    board = _positive_float(pcb_config.get("permitted_overfill_mm"))
    if board is not None:
        return board
    try:
        from softae.config.loader import load

        cfg = load().get("deposition", {})
        return _positive_float(cfg.get("permitted_overfill_mm")) or 0.0
    except Exception:
        return 0.0


def elution_capacity_uL(pcb_config: dict[str, Any]) -> float | None:
    """The most fluid this electrode may receive — void **plus** the permitted bead.

    This is the hard stop. :func:`well_void_uL` is the softer, purely geometric
    boundary; a volume between the two is above the brim but attainable, and
    :func:`classify_fill` reports it as a warning rather than a refusal.
    """
    void = well_void_uL(pcb_config)
    if void is None:
        return None
    h = permitted_overfill_mm(pcb_config)
    if h <= 0:
        return void
    d = _positive_float(pcb_config.get("well_diameter_mm"))
    return void if d is None else void + brim_cap_uL(d / 2.0, h)


def well_capacity_uL(pcb_config: dict[str, Any]) -> float | None:
    """Per-electrode max eluted volume (µL) for a PCB, or ``None`` if unset.

    The budget every formulation-bearing surface enforces.  ``None`` means the
    board declares no cap, which downstream
    (:func:`softae.core.formulation.plan_formulation`) treats as "no budget
    enforced".  The volume analogue of :func:`electrode_count`.

    Resolution:

    1. an explicit ``well_capacity_uL`` — still wins, because a deliberate
       working margin is a policy choice this cannot infer;
    2. otherwise :func:`elution_capacity_uL` — the well's own geometry plus
       whatever bead the board is permitted to carry above its brim.

    Tier 2 exists because a hand-typed capacity and a declared geometry are two
    numbers that can disagree, and the failure is silent and in the unsafe
    direction: the 4-stripe board carried 120 µL against a void of 118.77 µL, so
    its overflow guard permitted 1 % more than the walls hold. A board that
    states its dimensions should not also have to state their product.
    """
    cap = pcb_config.get("well_capacity_uL")
    if cap is not None:
        try:
            cap = float(cap)
        except (TypeError, ValueError):
            return None
        return cap if cap > 0 else None

    return elution_capacity_uL(pcb_config)


#: Fill bands, in increasing severity. ``above_brim`` is the point of the split:
#: it is a *warning*, because the bead is physically supportable, where before
#: this it was indistinguishable from a refusal.
FILL_OK = "within_well"
FILL_ABOVE_BRIM = "above_brim"
FILL_OVER_PERMITTED = "over_permitted"
FILL_UNKNOWN = "unknown"


@dataclass(frozen=True)
class FillVerdict:
    """Where a cast volume sits against a well's void and its permitted bead."""

    total_uL: float
    band: str
    void_uL: float | None = None
    permitted_uL: float | None = None
    hemisphere_uL: float | None = None
    #: Bead height the volume actually implies (mm); 0.0 at or below the brim.
    bead_height_mm: float = 0.0
    #: Permitted minus total (negative when over). ``inf`` when no limit is known.
    headroom_uL: float = float("inf")

    @property
    def blocks(self) -> bool:
        """Whether this must stop a cast. Only the permitted limit does."""
        return self.band == FILL_OVER_PERMITTED

    @property
    def warns(self) -> bool:
        return self.band == FILL_ABOVE_BRIM

    def describe(self) -> str:
        if self.band == FILL_UNKNOWN:
            return (f"{self.total_uL:.1f} uL — the board declares no well, so there "
                    f"is nothing to check it against")
        if self.band == FILL_OK:
            return (f"{self.total_uL:.1f} uL fits the well "
                    f"({self.void_uL:.1f} uL void, {self.headroom_uL:.1f} uL spare)")
        over = self.total_uL - (self.void_uL or 0.0)
        if self.band == FILL_ABOVE_BRIM:
            return (f"{self.total_uL:.1f} uL sits {over:.1f} uL ABOVE THE BRIM as a "
                    f"{self.bead_height_mm:.2f} mm bead — attainable on a non-wetting "
                    f"wall, but the well itself is full at {self.void_uL:.1f} uL")
        return (f"{self.total_uL:.1f} uL exceeds the permitted "
                f"{self.permitted_uL:.1f} uL by {-self.headroom_uL:.1f} uL — it would "
                f"need a {self.bead_height_mm:.2f} mm bead")


def bead_height_for_cap_mm(volume_uL: float, radius_mm: float) -> float:
    """Invert :func:`brim_cap_uL`: the bead height *volume_uL* implies (mm).

    Solved by bisection rather than Cardano. The cap volume is strictly
    increasing in ``h`` for ``h > 0``, so bisection is unconditionally correct
    here and stays obviously correct on inspection, which the cubic's sign
    handling does not. Nothing calls this in a loop.
    """
    v = float(volume_uL)
    r = float(radius_mm)
    if v <= 0 or r <= 0:
        return 0.0
    lo, hi = 0.0, max(r, 1.0)
    while brim_cap_uL(r, hi) < v:      # a bead taller than the hemisphere is
        hi *= 2.0                      # unphysical, but still worth reporting
        if hi > 1e4:
            return hi
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if brim_cap_uL(r, mid) < v:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def classify_fill(total_uL: float, pcb_config: dict[str, Any]) -> FillVerdict:
    """Three-band verdict on a cast volume: fits / above the brim / over the limit.

    The middle band is the reason this exists. A volume between the void and the
    permitted bead was previously an overflow — a hard stop — when in fact it is a
    perfectly castable bead on a non-wetting wall. Demoting it to a warning widens
    the range of formulations a board can take without weakening the actual limit,
    which is now stated explicitly by ``permitted_overfill_mm`` rather than being
    an accident of where the walls happen to end.

    A board with no declared well returns :data:`FILL_UNKNOWN` and never blocks —
    the same posture as :func:`deposit_area_mm2`: absent is not zero.
    """
    total = float(total_uL)
    void = well_void_uL(pcb_config)
    permitted = elution_capacity_uL(pcb_config)
    if void is None or permitted is None:
        return FillVerdict(total_uL=total, band=FILL_UNKNOWN)

    d = _positive_float(pcb_config.get("well_diameter_mm")) or 0.0
    above = max(0.0, total - void)
    bead = bead_height_for_cap_mm(above, d / 2.0) if above > 0 else 0.0

    if total <= void + _VOL_TOL_UL:
        band = FILL_OK
    elif total <= permitted + _VOL_TOL_UL:
        band = FILL_ABOVE_BRIM
    else:
        band = FILL_OVER_PERMITTED

    return FillVerdict(
        total_uL=total, band=band, void_uL=void, permitted_uL=permitted,
        hemisphere_uL=hemisphere_cap_uL(pcb_config), bead_height_mm=bead,
        headroom_uL=permitted - total,
    )


def _positive_float(value: Any) -> float | None:
    """A strictly-positive float, or ``None`` for absent / zero / unparseable."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _well_area_mm2(pcb_config: dict[str, Any]) -> float | None:
    """Mouth area of a declared circular well, or ``None`` if the board has none."""
    d = _positive_float(pcb_config.get("well_diameter_mm"))
    return None if d is None else math.pi * (d / 2.0) ** 2


def deposit_area_mm2(pcb_config: dict[str, Any]) -> float | None:
    """Area the cast film covers on one electrode (mm²), or ``None`` if unset.

    **The authoritative deposit area (P7.2).** Thickness is
    ``final_volume_uL / area_mm2 * 1000``, but the area behind that number came
    only from GUI spin boxes defaulting to a 5 mm disc — so a headless campaign
    had no area at all, and a GUI one silently used a default that no board
    declared. A thickness target is meaningless without a board-tied area.

    Resolution order, most specific first:

    1. an explicit ``deposit_area_mm2`` on the board — the escape hatch for a
       wetted area that is neither the well nor the electrode rectangle (a
       hydrophobic mask, an observed spread that does not fill the well);
    2. ``well_diameter_mm`` — the circular well that physically confines the
       cast. **This is what a drop actually covers**, and on a board that
       declares one it is the right denominator;
    3. otherwise ``electrode_L_cm × electrode_w_cm``, converted from cm² to mm²
       (×100).

    Tier 3 is a **weak** fallback and is kept only so boards that declare nothing
    behave as they always have. Those two config keys are, per ``[eis.cell]``,
    the electrode *gap* and the stripe *length* — conduction geometry, not a
    wetted footprint — so their product is the area of the inter-electrode
    rectangle and has no reason to equal the area a drop covers. On the 4-stripe
    board it is 4.0 mm² against a 4.88 mm well's 18.7 mm², a factor of 4.7 in
    every thickness computed from it. Declare ``well_diameter_mm``.

    **A board with no wells does not reach tier 3 at all.** ``cast_confinement =
    "sessile"`` marks a board cast as free droplets on a flat surface, where the
    wetted area is set by volume and contact angle — an *observation*, not a
    geometry. Nothing on the board predicts it, so such a board must declare
    ``deposit_area_mm2`` from a measured footprint or get ``None``. Letting it
    fall through to the electrode rectangle would return a number that describes
    the gap between two stripes and has no relationship whatever to where the
    droplet actually sat.

    ``None`` means the board declares nothing usable, and callers must treat a
    computed thickness as unavailable rather than substituting a guess — the whole
    point of this function is that an invented area silently corrupts every
    thickness downstream of it.
    """
    explicit = _positive_float(pcb_config.get("deposit_area_mm2"))
    if explicit is not None:
        return explicit

    if str(pcb_config.get("cast_confinement", "")).strip().lower() == "sessile":
        return None

    well = _well_area_mm2(pcb_config)
    if well is not None:
        return well

    try:
        length_cm = float(pcb_config.get("electrode_L_cm", 0.0))
        width_cm = float(pcb_config.get("electrode_w_cm", 0.0))
    except (TypeError, ValueError):
        return None
    area_cm2 = length_cm * width_cm
    return area_cm2 * 100.0 if area_cm2 > 0 else None


def thickness_um(volume_uL: float, area_mm2: float) -> float:
    """Film thickness (µm) from a deposited volume and its area.

    1 µL over 1 mm² is 1 mm, i.e. 1000 µm — hence the factor.

    **This is a nominal, dense-film geometric estimate.** Volumes are treated as
    additive and no dry-film density or porosity is modelled, so a porous real
    film is thicker than this number says. Sound as a control target and as a
    consistent relative measure across a campaign; not a metrology claim.
    """
    if area_mm2 <= 0:
        raise ValueError("deposit area must be > 0 mm²")
    return float(volume_uL) / float(area_mm2) * 1000.0


def volume_for_thickness_uL(thickness_um_value: float, area_mm2: float) -> float:
    """Volume (µL) needed to reach *thickness_um_value* over *area_mm2*.

    The inverse of :func:`thickness_um`, and the arithmetic behind a
    :class:`~softae.core.formulation.ThicknessTarget`.
    """
    if area_mm2 <= 0:
        raise ValueError("deposit area must be > 0 mm²")
    return float(thickness_um_value) * float(area_mm2) / 1000.0


def electrode_positions(
    pcb_config: dict[str, Any],
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)`` electrode coordinate arrays for a PCB config.

    Parameters
    ----------
    pcb_config : dict
        Must contain ``"grid"`` (``[rows, cols]``) and ``"spacing_mm"``
        (``[step_x, step_y]``, centre-to-centre distance between *adjacent*
        electrodes).  Defaults to a 4×4 grid with zero spacing.
    origin_x, origin_y : float
        Stage position of electrode 1 (upper-left corner) in mm.  For SoftAE
        this is ``[stage_calibration].dep1_x``/``dep1_y``.

    Returns
    -------
    xs, ys : np.ndarray
        Row-major coordinate arrays (electrode 1 first).
    """
    rows, cols = pcb_config.get("grid", [4, 4])
    step_x = abs(pcb_config.get("spacing_mm", [0, 0])[0])
    step_y = abs(pcb_config.get("spacing_mm", [0, 0])[1])

    xs: list[float] = []
    ys: list[float] = []
    for r in range(rows):
        for c in range(cols):
            xs.append(origin_x - c * step_x)
            ys.append(origin_y - r * step_y)

    return np.array(xs), np.array(ys)


def nearest_electrode(
    pcb_config: dict[str, Any],
    x: float,
    y: float,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    *,
    tolerance_mm: float | None = None,
) -> int | None:
    """Inverse of :func:`electrode_xy_for_channel`: stage ``(x, y)`` → channel.

    Returns the 1-based channel of the electrode closest to ``(x, y)`` when that
    electrode is within ``tolerance_mm``, else ``None`` (the stage is not over a
    well).  This is how a *manual* pump learns which well it is casting into.

    ``tolerance_mm`` defaults to half the smaller electrode pitch (the snap
    radius that partitions the plane into per-electrode cells); when the board
    declares no spacing it falls back to 1 mm so an exact hit still registers.
    """
    xs, ys = electrode_positions(pcb_config, origin_x, origin_y)
    if len(xs) == 0:
        return None

    if tolerance_mm is None:
        step_x = abs(pcb_config.get("spacing_mm", [0, 0])[0])
        step_y = abs(pcb_config.get("spacing_mm", [0, 0])[1])
        pitches = [s for s in (step_x, step_y) if s > 0]
        tolerance_mm = (min(pitches) / 2.0) if pitches else 1.0

    d2 = (xs - float(x)) ** 2 + (ys - float(y)) ** 2
    idx = int(np.argmin(d2))
    if float(d2[idx]) ** 0.5 > float(tolerance_mm):
        return None
    return idx + 1


def electrode_xy_for_channel(
    pcb_config: dict[str, Any],
    channel_1based: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> tuple[float, float]:
    """Return the ``(x, y)`` stage position for a 1-based channel number.

    Raises
    ------
    ValueError
        If ``channel_1based`` is outside the grid (``1 .. rows*cols``).
    """
    xs, ys = electrode_positions(pcb_config, origin_x, origin_y)
    idx = int(channel_1based) - 1
    if idx < 0 or idx >= len(xs):
        raise ValueError(
            f"channel {channel_1based} outside grid of {len(xs)} electrodes"
        )
    return float(xs[idx]), float(ys[idx])
