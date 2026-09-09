"""The forward model, its parameter split, and the known quantities it consumes.

Everything here is *per spectrum* and model-side: what a spectrum is
(:class:`ConstrainedSpectrum`), what is known about the fixture
(:func:`fixture_admittance`), what ``Z(f)`` is (:func:`model_impedance`), and the
model-free observable admissibility is decided on (:func:`arc_is_resolved`).
Nothing here fits anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from .refusals import UnmeasuredFixtureShunt

#: One logger for the whole package, named for the package rather than for this
#: submodule so that the split did not rename a single log event. Imported by
#: :mod:`~softae.analysis.eis.constrained_fit.fit` rather than re-created there.
logger = structlog.get_logger("softae.analysis.eis.constrained_fit")

TWO_PI = 2.0 * math.pi

#: Fitted parameters, in vector order. ``L`` is *not* here: it is pinned, and a pinned
#: quantity carried in the parameter vector is one bounds change away from being fitted.
PARAM_NAMES: tuple[str, ...] = ("R0", "R1", "Qg", "ng", "Qd", "nd")
SHARED_PARAMS: tuple[str, ...] = ("Qg", "ng", "nd")
PER_SPECTRUM_PARAMS: tuple[str, ...] = ("R0", "R1", "Qd")

#: ``name -> (lower, upper)``. Physical, not tuned: ``ng``/``nd`` are CPE exponents in
#: [0, 1] narrowed to the range a geometric or a blocking element can occupy.
BOUNDS: dict[str, tuple[float, float]] = {
    "R0": (1e-3, 1e11),
    "R1": (1e-3, 1e11),
    "Qg": (1e-15, 1e-3),
    "ng": (0.50, 1.00),
    "Qd": (1e-14, 1e-1),
    "nd": (0.30, 1.00),
}


# ── The known fixture shunt ──────────────────────────────────────────────────

#: The policies :func:`fixture_admittance` accepts for points ``G_fixture`` does not
#: cover — and, since 2026-09-08, for the case where it covers *nothing* because there
#: is no table at all. ``"zero"`` is the only one that means anything there, and saying
#: it is how a caller declares the shunt negligible rather than merely unmeasured.
BEYOND_COVERAGE_POLICIES = ("hold", "drop", "zero")


@dataclass(frozen=True)
class ShuntTable:
    """``Y_shunt(f)`` evaluated on one spectrum's frequencies, with its own accounting.

    ``n_held`` is not decoration. :meth:`FixtureConductance.at
    <softae.analysis.eis.calibration.FixtureConductance.at>` returns NaN outside the
    band it was measured in, correctly and on purpose, and this module has to *decide*
    what to do with that rather than inherit whichever answer happens to fall out.
    Holding the endpoint is a choice with a cost, so the count of points it was applied
    to travels with the values.

    ``n_held == n_points`` — a :attr:`held_fraction` of exactly 1.0 — is the shape of a
    spectrum with **no** measured shunt at all, reachable only by declaring
    ``beyond_coverage="zero"``; see :class:`UnmeasuredFixtureShunt` for why that has to
    be declared rather than inferred.
    """

    y: np.ndarray
    n_points: int
    n_held: int = 0
    coverage_hz: tuple[float, float] = (float("nan"), float("nan"))
    beyond_coverage: str = "hold"

    @property
    def held_fraction(self) -> float:
        return self.n_held / self.n_points if self.n_points else float("nan")


def fixture_admittance(
    freq_hz: np.ndarray,
    conductance: Any,
    c_stray_F: float = 0.0,
    *,
    beyond_coverage: str = "hold",
) -> ShuntTable:
    """``Y_shunt(f) = G_fixture(f) + jωC_stray``, evaluated per frequency point.

    **Consumed as a table, never collapsed to a scalar.** On this fixture the shunt's
    real part is dielectric loss — seven tied open blanks give ``d ln G/d ln f`` between
    +0.87 and +1.04 — so a single number describes the fixture at exactly one frequency
    and is wrong everywhere else.

    **And never fitted while also being supplied.** Handing the optimiser a free leak
    resistance *and* the measured shunt is degenerate: measured on the 10 kΩ rung the
    combination gives ``+1.6e6 %`` with the fitted ``R_leak`` running to 2.6e11 Ω. That
    is why there is no ``Rleak`` in :data:`PARAM_NAMES`. AMP_v1 fits a scalar leak
    because it has no open blank; we have one, and a measurement beats a free parameter.

    *beyond_coverage* decides the points the conductance table does not cover — a real
    case, not a corner one: ``G_fixture`` on ch25 spans 14.3 Hz–200 kHz while the
    reference-resistor sweeps start near 1.2 Hz, so 10–11 points of each fall below it.

    ``"hold"``
        Clamp to the nearest measured endpoint. **The default, and measured against
        ``"drop"`` on the three reference resistors** — mean |error| 1.03 % (worst
        1.70 %) holding, against 2.15 % (worst 5.24 %) dropping, and leave-one-out
        0.54 % against 4.88 %. The dropped points are the *bottom* decade of the sweep,
        which is where a blocking cell's information about ``R_sol`` lives, so dropping
        them costs more than the extrapolation does. It is also defensible in direction:
        below the table ``G ∝ ω`` is still falling, so holding the lowest measured ``G``
        *overstates* the shunt there. The count is reported via
        :attr:`ShuntTable.n_held` and logged, so it can never read as coverage that was
        actually measured.
    ``"drop"``
        Report the uncovered points via ``n_held`` and emit NaN for them, leaving the
        caller to drop them. The honest option when the excursion is large — and the
        one to reach for rather than widening ``"hold"``'s reach silently.
    ``"zero"``
        Treat the uncovered points as having no fixture shunt. **Only** legitimate when
        the open blank was judged unusable, which is itself positive evidence that the
        shunt is negligible. **It is also the whole-table case**: with no ``G_fixture``
        at all every point is uncovered, so this is the one policy that still means
        something, and choosing it is how the absence gets *declared*.

    :raises UnmeasuredFixtureShunt: when *conductance* carries no table and
        *beyond_coverage* is not ``"zero"``. That combination used to return a shunt of
        exactly zero reporting ``n_held = 0``, which is indistinguishable from full
        coverage — see the exception's own docstring for the population this would run
        silently wrong on.
    """
    if beyond_coverage not in BEYOND_COVERAGE_POLICIES:
        # Hoisted out of the interpolation branch: a policy this function does not
        # understand is an error whether or not there is a table to apply it to.
        raise ValueError(f"unknown beyond_coverage policy {beyond_coverage!r}")

    f = np.asarray(freq_hz, dtype=float)
    y = np.zeros(f.shape, dtype=complex)
    n_held = 0
    lo = hi = float("nan")

    gf = np.asarray(getattr(conductance, "freq_hz", ()) or (), dtype=float)
    gs = np.asarray(getattr(conductance, "G_S", ()) or (), dtype=float)
    if gf.size and gf.size == gs.size:
        order = np.argsort(gf)
        gf, gs = gf[order], gs[order]
        lo, hi = float(gf[0]), float(gf[-1])
        outside = (f < lo) | (f > hi)
        n_held = int(np.count_nonzero(outside))
        g = np.interp(np.log10(f), np.log10(gf), gs, left=gs[0], right=gs[-1])
        if beyond_coverage == "drop":
            g = np.where(outside, np.nan, g)
        elif beyond_coverage == "zero":
            g = np.where(outside, 0.0, g)
        y = y + g
    elif gf.size:
        raise ValueError("G_fixture freq_hz and G_S differ in length")
    elif beyond_coverage == "zero":
        # Declared absent. Every point is beyond a coverage of nothing, so the whole
        # spectrum is counted as held — `held_fraction` reads 1.0 and the log below
        # fires, neither of which a genuinely measured shunt can produce.
        n_held = int(f.size)
    else:
        raise UnmeasuredFixtureShunt(
            "no G_fixture: the fixture shunt would be exactly zero at every frequency "
            "and the ShuntTable would report full coverage of it. On a blocking cell "
            "the low-frequency tail is largely the shunt, and Qg is shared across the "
            "set, so one uncorrected spectrum moves the artifact for all of them. "
            "Supply a FixtureConductance for this channel, or declare the absence with "
            "beyond_coverage='zero' — legitimate only when the open blank was judged "
            "unusable, which is itself evidence the shunt is negligible."
        )

    stray = float(c_stray_F)
    if stray == stray and stray:
        y = y + 1j * TWO_PI * f * stray

    if n_held:
        logger.info(
            "eis_constrained_fit_shunt_beyond_coverage",
            n_held=n_held, n_points=int(f.size), policy=beyond_coverage,
            coverage_hz=(lo, hi),
            msg=("G_fixture has no coverage for these points; the policy named here "
                 "is what was applied to them — not a measurement"),
        )
    return ShuntTable(y=y, n_points=int(f.size), n_held=n_held,
                      coverage_hz=(lo, hi), beyond_coverage=beyond_coverage)


# ── The model ────────────────────────────────────────────────────────────────

def _z_cpe(omega: np.ndarray, q: float, n: float) -> np.ndarray:
    return 1.0 / (q * (1j * omega) ** n)


def model_impedance(
    freq_hz: np.ndarray,
    params: dict[str, float],
    y_shunt: np.ndarray,
    pinned_l_H: float,
) -> np.ndarray:
    """``Z(f) = jωL + [ (R0 + (R1 ∥ CPE_g) + CPE_d) ∥ Y_shunt ]``.

    ``L`` and ``y_shunt`` are inputs, not unknowns — the whole point of the design.
    """
    omega = TWO_PI * np.asarray(freq_hz, dtype=float)
    z_geo = _z_cpe(omega, params["Qg"], params["ng"])
    z_path = (params["R0"]
              + 1.0 / (1.0 / params["R1"] + 1.0 / z_geo)
              + _z_cpe(omega, params["Qd"], params["nd"]))
    y_total = 1.0 / z_path + np.asarray(y_shunt, dtype=complex)
    return 1j * omega * float(pinned_l_H) + 1.0 / y_total


def conductance_r_sol(freq_hz: np.ndarray, z: np.ndarray) -> float:
    """``1 / max Re(Y)`` — the seed estimate, and a model-free sanity value."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(1.0 / np.nanmax((1.0 / np.asarray(z, dtype=complex)).real))


# ── One spectrum, ready to fit ───────────────────────────────────────────────

@dataclass(frozen=True)
class ConstrainedSpectrum:
    """One cleaned spectrum with everything the fit treats as known already attached.

    The spectrum is expected in the **corrected** domain — whatever
    :func:`~softae.analysis.eis.fixture.apply_series_correction` was going to do has
    been done. ``pinned_l_H`` is then whatever inductance remains to be modelled, which
    is ``0.0`` when the series correction already consumed ``L_lead_H`` and
    ``L_lead_H`` itself when it did not. Both are "L is pinned from commissioning";
    only the arithmetic differs, and ``l_source`` records which.
    """

    label: str
    freq_hz: np.ndarray
    z: np.ndarray
    y_shunt: np.ndarray
    pinned_l_H: float = 0.0
    channel: int | None = None
    l_source: str = ""
    #: Known answer, when there is one (a standard, a reference resistor). Never used
    #: by the fit — only by :func:`holdout_report`.
    reference_ohm: float = float("nan")
    shunt: ShuntTable | None = None

    def __post_init__(self) -> None:
        n = int(np.asarray(self.freq_hz).size)
        for name in ("z", "y_shunt"):
            if int(np.asarray(getattr(self, name)).size) != n:
                raise ValueError(
                    f"{self.label}: {name} has "
                    f"{np.asarray(getattr(self, name)).size} points against "
                    f"{n} frequencies")


def arc_is_resolved(freq_hz: np.ndarray, z: np.ndarray) -> bool:
    """Whether the geometric arc's **low-frequency flank** is inside the measured band.

    The observable is an interior local minimum in ``-Im Z`` — the valley where the
    falling blocking tail crosses the rising bulk arc. Below that crossover the sweep
    sees only the blocking electrode and nothing constrains ``Qg`` or ``ng``. The
    crossover moves **up** in frequency with conductivity, so on a conductivity ladder
    the *dilute* members are the ones that show it: on the four NIST standards
    ``kcl_45uS`` and ``kcl_84uS`` do (valleys at 124 kHz, near the top of their sweeps)
    and ``kcl_1413uS``/``kcl_4500uS`` do not.

    Model-free on purpose. An admissibility test that had to run the fit first would be
    answering the question it exists to gate.

    **What this is not.** It is a *necessary* condition validated against one corpus,
    not a proof of identifiability. The stricter reading — require the arc's **apex** in
    band — was tried and rejected on measurement: the apex is out of band for all four
    NIST standards, including the two whose presence makes the artifact work, so that
    criterion refuses the very fit that scores 1.43 %. Conversely a synthetic spectrum
    can show the crossover with its apex still a decade above the sweep. The validation
    that this reading is the useful one is in :func:`check_admissible`.
    """
    f = np.asarray(freq_hz, dtype=float)
    im = -np.asarray(z, dtype=complex).imag
    ok = np.isfinite(im) & np.isfinite(f)
    if int(np.count_nonzero(ok)) < 3:
        return False
    y = im[ok][np.argsort(f[ok])]
    return bool(np.any((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])))
