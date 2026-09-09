"""The two fits — the stacked one that produces an artifact, and the frozen one that
spends it — with the constants that govern them.

:func:`fit_shared` fits ``Qg, ng, nd`` once across a set; :func:`fit_frozen` fits
``R0, R1, Qd`` for one spectrum against a frozen artifact. The constants at the top
of this file are measurement records, not preferences: each one carries the
experiment that fixed its value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .admissibility import (
    MIN_ARC_RESOLVED,
    QG_PLAUSIBLE_MAX,
    Admissibility,
    check_admissible,
    implausible_shared,
)
from .model import (
    BOUNDS,
    PER_SPECTRUM_PARAMS,
    SHARED_PARAMS,
    ConstrainedSpectrum,
    conductance_r_sol,
    logger,
    model_impedance,
)
from .refusals import (
    ImplausibleArtifact,
    InadmissibleFitSet,
    UnstableFitSurface,
)

#: ``xtol``/``ftol``/``gtol`` for every fit in this module.
#:
#: **Not a preference, and scipy's default is a wrong answer that looks converged.**
#: Measured on the stacked four-standard fit, everything else held fixed:
#:
#: ======  ========  ========  ======  ================================
#: tol     cost      status    nfev    mean |R_sol error| vs AMP
#: ======  ========  ========  ======  ================================
#: 1e-8    3.60      3         63      **8014 %**
#: 1e-10   0.337     3         39      1.55 %
#: 1e-12   0.332     3         44      1.56 %
#: 1e-14   0.332     3         47      1.56 %
#: ======  ========  ========  ======  ================================
#:
#: ``status = 3`` is *"ftol termination"* — the optimiser's own report of a converged
#: fit — in **every** row, the 8014 % one included, and no parameter is railed there
#: either. That is ``SUBAGENT_RULES`` §3.1 in its purest form: an instrument returning
#: the shape of a pass for a different question. The knee is between 1e-8 and 1e-10 and
#: the extra decades are free (47 ``nfev`` against 63), so the constant sits at the far
#: side of it rather than on it.
#:
#: The tolerance is therefore a module constant rather than a keyword with a permissive
#: default, and :func:`fit_shared` logs a warning when a caller loosens it.
#:
#: Distinct from :data:`~softae.analysis.eis.fitter.DEFAULT_FIT_TOL` (1e-10) because the
#: problems are different: that one fits 5 parameters against one spectrum, this one
#: fits ``3 + 3N`` against N stacked spectra, where the shared parameters are informed
#: only by the *differences* between spectra and the gradient in them is correspondingly
#: shallow.
CONSTRAINED_FIT_TOL = 1e-14

#: Iteration ceilings. Generous for the same reason ``fitter.DEFAULT_MAX_NFEV`` is:
#: a failure to converge should be reported, not manufactured by too small a budget.
#:
#: **``MAX_NFEV_GLOBAL`` was 120 000 and is now 20 000 — and the reason it was left at
#: 120 000 in Wave 1A does not survive re-measurement, so it is worth stating what
#: actually happens rather than repeating it.** ``[a184]`` §4(b) reported that the two
#: ``Qg = 4e-8`` starts "spend the full 120 000". **They do not.** Per start, four NIST
#: standards, full grid, each start timed alone, three preparations:
#:
#: ===================  ==================  ============  ==================
#: start (seed ``Qg``)   ``nfev``            status        cost reached
#: ===================  ==================  ============  ==================
#: 4e-12 .. 4e-9         34 .. 734           ftol / xtol   0.235 .. 6.73
#: **4e-8 (both)**       **2 125 .. 31 444** ftol / xtol   **1.9 .. 5.4**
#: ===================  ==================  ============  ==================
#:
#: The ceiling was never binding: every start terminates on tolerance, the worst at 26 %
#: of it. What it was buying was *permission* — the poisoned starts take 42–108 s each
#: to converge to a cost an order of magnitude worse than the winner's, and on the four
#: standards **those two starts are 91 s of a 96 s call (thin) and 115 s of 123 s (full
#: conditioning): 93–95 % of the wall time spent on two descents that neither win nor
#: join the consensus set.**
#:
#: **20 000 is accuracy-neutral over every fit this module has been measured on**, which
#: is the standard Wave 1A said the change had to meet and could not meet from inside a
#: defect fix. Bitwise-identical ``(Qg, ng, nd)``, cost, consensus count and order
#: spread against 120 000 for: the four standards on both preparations at ceilings
#: 40 000 / 20 000 / 8 000 / 4 000 / 2 000 / 1 000 / 500 — every one of them — and all
#: eleven ≥2-member subsets of the four standards on both preparations at
#: 6 000 — 22 further independent fits. On the full-conditioned four standards the whole
#: call is **140.7 s at 120 000 against 75.6 s at 20 000, same artifact to the last
#: digit.** It is 29× the largest *winning* start measured here (688 ``nfev``), 27× the
#: largest start that reached consensus (734), and 8× the largest winner `[a184]`
#: reported (~2 400) — so it truncates losers only.
#:
#: **It does not go lower than that, and the reason is the direction of the risk.** The
#: ceiling is not accuracy-neutral by nature — it is accuracy-neutral *on this corpus*,
#: where the winner converges in 41-48 ``nfev``. Truncating a *loser* costs nothing;
#: truncating a *winner* manufactures a non-convergence that reports as a fit, which is
#: the failure this constant exists to prevent. The films, which have no answer key,
#: could easily need more iterations than four clean liquids do. 20 000 keeps more than
#: an order of magnitude of headroom over anything observed; 500 keeps none.
#:
#: **The ceiling is not where the cost is, and the honest lever is somewhere this change
#: deliberately does not touch.** Lowering it to 20 000 roughly halves the call and the
#: two poisoned starts still run — most of what remains is theirs. Deleting them from
#: :data:`SHARED_SEED_GRID` returns a bitwise-identical artifact and would remove ~93 %
#: of the runtime — **and it is refused anyway**, because those two entries are what
#: demonstrate that the good basin is *preferred* rather than merely reachable, and they
#: are the only starts that can show the poisoned basin exists at all. A grid that
#: cannot reach the failure cannot evidence the refusal.
#:
#: With one start the ceiling was the only safety net; with ten the other nine are, so a
#: start that exhausts its budget reports ``status = 0`` and loses on cost.
#: :func:`fit_shared` still takes ``max_nfev`` for a caller who knows its starts are bad.
MAX_NFEV_SINGLE = 40_000
MAX_NFEV_GLOBAL = 20_000

#: Multipliers on the conductance estimate of ``R_sol`` used to seed :func:`fit_frozen`.
#:
#: **Multi-start is required, and its necessity is diagnostic.** Once the artifact is
#: good the frozen fit is seed-insensitive and every multiplier lands in the same
#: basin; when the artifact is *not* good the seed decides the answer outright. So a
#: wide spread across multipliers is itself evidence about the artifact, which is why
#: :attr:`WellFit.seed_spread_pct` is reported rather than discarded.
R_SOL_SEED_MULTIPLIERS: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: Starting points for the shared three in :func:`fit_shared`'s multi-start, one
#: stacked fit per entry, lowest cost kept. **The same principle as
#: :data:`R_SOL_SEED_MULTIPLIERS`, applied one level up**: decades either side of the
#: default, wide enough that no single descent decides the answer.
#:
#: **This replaced a continuation, and the continuation was order-dependent.** The
#: previous construction walked the set seeding each spectrum from the *previous*
#: spectrum's fit, resetting only ``R0``/``R1``, and took the median of that chain as
#: ``x0``. Reordering the input therefore reordered the chain and landed the stacked fit
#: in a different basin: **24 permutations of the same four NIST standards spanned
#: ``R_sol`` MAE 1.561 % to 1.065e9 %**, with :func:`check_admissible` returning True on
#: all 24 and :meth:`SharedArtifact.railed` firing on none. A clean *duplicate* at
#: position 0 gave 91 870 %; the dirtiest spectrum at position 4 gave 2.16 %. "Poisons
#: the artifact" was a property of a spectrum's *position*, not of the spectrum.
#:
#: The continuation was itself chosen on measurement and the measurement was real:
#: re-seeding ``Qd`` per spectrum *while the shared three were also free from their
#: defaults* converges at cost 3.53 / 28 395 % error against cost 0.33 / 1.4 % for the
#: continuation. That result is not a case for the continuation, only against one naive
#: alternative. :func:`_stacked_start` keeps what the continuation was buying — a
#: per-spectrum ``Qd`` that has already been fitted rather than guessed — by fitting
#: each spectrum's ``R0``/``R1``/``Qd`` **with the candidate shared triple frozen**,
#: exactly as :func:`fit_frozen` does. Each spectrum's start then depends only on
#: itself, so the construction is order-free by shape rather than by luck.
#:
#: ``Qg`` steps by decades because it is the parameter the two known basins separate on
#: (good ``Qg`` ~1e-9, poisoned ~1e-7); ``nd`` carries two entries because the poisoned
#: basin also sits low in ``nd`` (~0.39) and a grid that cannot start there cannot
#: demonstrate that the good basin is preferred. On the four standards the grid reaches
#: cost 0.2537 — below the continuation's own best of 0.33 — from three separate entries.
SHARED_SEED_GRID: tuple[dict[str, float], ...] = tuple(
    {"Qg": qg, "ng": 0.95, "nd": nd}
    for qg in (4e-12, 4e-11, 4e-10, 4e-9, 4e-8)
    for nd in (0.85, 0.55)
)

#: A start counts as agreeing with the winner when its cost is within this factor of
#: the winner's. Starts that stopped somewhere worse are describing a *different*
#: minimum and their disagreement is expected, not evidence; the question
#: :attr:`SharedArtifact.order_spread_pct` asks is whether the descents that reached
#: the **same** minimum also reached the same answer.
CONSENSUS_COST_FACTOR = 1.5

#: Floor under :data:`CONSENSUS_COST_FACTOR`'s window, expressed per residual.
#:
#: **A purely relative window asks the wrong question when the fit is exact**, and this
#: was found the hard way: on a spectrum synthesised from :func:`model_impedance` the
#: winning start reaches cost 1.3e-29, so a window of 1.5x admits nothing but itself,
#: the spread is NaN, and a *perfect* recovery is refused. ``1e-12`` per residual is a
#: modulus-weighted residual of about a part per million per point — orders below any
#: real measurement, so on measured data (best cost ~0.25 over ~320 residuals) the floor
#: never binds and the relative window decides. It exists only to keep "several starts
#: recovered the answer exactly" from reading as "no start agreed with the winner".
EXACT_COST_PER_RESIDUAL = 1e-12

#: Refusal threshold on :attr:`SharedArtifact.order_spread_pct`, in percent.
#:
#: **Derived from the separation between two measured populations, not chosen round.**
#: Running :data:`SHARED_SEED_GRID` over fitting sets whose quality is known
#: independently — ``R_sol`` against AMP_v1's own constrained artifact, and σ against
#: NIST after the same cell-constant calibration AMP applies to its own:
#:
#: ==================================  ==========  ==========  ==============
#: fitting set                          R_sol MAE   σ MAE       order spread
#: ==================================  ==========  ==========  ==============
#: 4 NIST standards, conditioned         3.73 %      1.13 %      **5.04 %**
#: LOO fold dropping ``kcl_1413uS``      4.02 %      —           **3.20 %**
#: LOO fold dropping ``kcl_4500uS``      3.82 %      —           **5.19 %**
#: 4 standards synthesised from
#: :func:`model_impedance` (exact)       ~0 %        —           **1.5e-9 %**
#: same 4 standards, grid restricted
#: to the poisoned basin                 5894 %      102 476 %   **1.35e5 %**
#: ==================================  ==========  ==========  ==============
#:
#: Worst good case 5.19 %, only poisoned case 1.35e5 % — four orders of magnitude apart
#: with nothing in between. The constant sits a factor of ~20 above the worst good case
#: rather than on it, and is still three orders below the poisoned one, so it is
#: insensitive to where in that gap it is placed. **The gap is the evidence; 100 is only
#: a readable number inside it.** See :func:`fit_shared` for what happens above it.
ORDER_SPREAD_REFUSE_PCT = 100.0


# ── Packing ──────────────────────────────────────────────────────────────────

def _pack_bounds(n: int) -> tuple[np.ndarray, np.ndarray]:
    lo = [BOUNDS[p][0] for p in SHARED_PARAMS] + [BOUNDS[p][0] for p in PER_SPECTRUM_PARAMS] * n
    hi = [BOUNDS[p][1] for p in SHARED_PARAMS] + [BOUNDS[p][1] for p in PER_SPECTRUM_PARAMS] * n
    return np.array(lo), np.array(hi)


def _unpack(x: np.ndarray, i: int) -> dict[str, float]:
    p = {name: float(x[j]) for j, name in enumerate(SHARED_PARAMS)}
    base = len(SHARED_PARAMS) + i * len(PER_SPECTRUM_PARAMS)
    for j, name in enumerate(PER_SPECTRUM_PARAMS):
        p[name] = float(x[base + j])
    return p


def _residual(spec: ConstrainedSpectrum, params: dict[str, float]) -> np.ndarray:
    """Modulus-weighted real/imaginary residual, stacked.

    Weighting matches :mod:`~softae.analysis.eis.fitter`'s ``weight_by_modulus=True``:
    unweighted least squares over decades of ``|Z|`` is dominated by the low-frequency
    end and cannot see a small series term against a large tail.
    """
    z_model = model_impedance(spec.freq_hz, params, spec.y_shunt, spec.pinned_l_H)
    w = 1.0 / np.abs(spec.z)
    return np.concatenate([(z_model.real - spec.z.real) * w,
                           (z_model.imag - spec.z.imag) * w])


def _seed(spec: ConstrainedSpectrum, multiplier: float = 1.0) -> dict[str, float]:
    r = conductance_r_sol(spec.freq_hz, spec.z) * multiplier
    return {"R0": 0.15 * r, "R1": 0.85 * r,
            "Qg": 4e-11, "ng": 0.95, "Qd": 2e-6, "nd": 0.85}


def _spread_pct(values: list[float]) -> float:
    """``100·(max − min)/min`` over the finite positive entries; NaN below two.

    NaN is the honest answer for "one value" and it must stay distinguishable from
    ``0.0``, which is the answer for "several values that agree" — ``SUBAGENT_RULES``
    §3.1(a). Both :attr:`WellFit.seed_spread_pct` and
    :attr:`SharedArtifact.order_spread_pct` are this quantity.
    """
    finite = [v for v in values if v == v and v > 0]
    if len(finite) < 2:
        return float("nan")
    return 100.0 * (max(finite) - min(finite)) / min(finite)


def _fit_free(
    spec: ConstrainedSpectrum, start: dict[str, float], free: tuple[str, ...], tol: float
) -> tuple[dict[str, float], Any]:
    lo = [BOUNDS[p][0] for p in free]
    hi = [BOUNDS[p][1] for p in free]

    def residual(q: np.ndarray) -> np.ndarray:
        p = dict(start)
        p.update({n: float(v) for n, v in zip(free, q)})
        return _residual(spec, p)

    q0 = np.clip([start[p] for p in free], lo, hi)
    res = least_squares(residual, q0, bounds=(lo, hi), method="trf",
                        max_nfev=MAX_NFEV_SINGLE, x_scale="jac",
                        xtol=tol, ftol=tol, gtol=tol)
    out = dict(start)
    out.update({n: float(v) for n, v in zip(free, res.x)})
    return out, res


# ── The two fits ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SharedArtifact:
    """The frozen shared three, and everything needed to judge them later."""

    Qg: float
    ng: float
    nd: float
    labels: tuple[str, ...] = ()
    cost: float = float("nan")
    status: int = 0
    nfev: int = 0
    n_free: int = 0
    n_residuals: int = 0
    tol: float = CONSTRAINED_FIT_TOL
    #: Points across the set where the shunt table had no coverage. Carried on the
    #: artifact because it is a property of the artifact, not of one spectrum.
    n_shunt_held: int = 0
    admissibility: Admissibility | None = None
    #: Entries in :data:`SHARED_SEED_GRID` that were run, and how many of them landed
    #: within :data:`CONSENSUS_COST_FACTOR` of the winner's cost.
    n_starts: int = 0
    n_consensus: int = 0
    #: The grid entry the winning descent started from.
    seed: dict[str, float] = field(default_factory=dict)
    #: **The counterpart to :attr:`WellFit.seed_spread_pct`, one level up.** Worst
    #: disagreement, in percent, between the consensus starts about any one spectrum's
    #: ``R_sol``. Independent descents that reached the *same* minimum should also have
    #: reached the same answer; when they did not, the minimum is a valley floor and the
    #: number that comes out of it is arbitrary. NaN when fewer than two starts reached
    #: consensus — "could not judge", which :func:`fit_shared` refuses on rather than
    #: passing through as if it were agreement.
    order_spread_pct: float = float("nan")

    @property
    def shared(self) -> dict[str, float]:
        return {"Qg": self.Qg, "ng": self.ng, "nd": self.nd}

    def railed(self) -> tuple[str, ...]:
        """Shared parameters resting on a bound — the artifact's own alarm.

        **Reported, and measured not to be sufficient.** Every poisoned artifact this
        module has produced — the arc-blind subset, and four seed grids aimed at the
        poisoned basin — returns ``()`` here: the closest any of them comes to a bound
        is ``nd`` 0.334 against a floor of 0.30. That is why
        :data:`QG_PLAUSIBLE_MAX` exists and why this stayed a report.
        """
        out = []
        for name, v in self.shared.items():
            lo, hi = BOUNDS[name]
            if abs(v - lo) <= 1e-6 * max(abs(v), abs(lo)) or \
                    abs(v - hi) <= 1e-6 * max(abs(v), abs(hi)):
                out.append(name)
        return tuple(out)

    def describe(self) -> str:
        rail = f", RAILED: {','.join(self.railed())}" if self.railed() else ""
        return (f"artifact from {len(self.labels)} spectra: Qg={self.Qg:.4g} "
                f"ng={self.ng:.4f} nd={self.nd:.4f} (cost {self.cost:.4g}, "
                f"status {self.status}, {self.nfev} nfev, order spread "
                f"{self.order_spread_pct:.3g}% over {self.n_consensus}/{self.n_starts} "
                f"starts{rail})")


@dataclass(frozen=True)
class WellFit:
    """One spectrum against a frozen artifact. ``R_sol`` is the reported observable."""

    label: str
    R0: float
    R1: float
    Qd: float
    cost: float = float("nan")
    ok: bool = False
    seed_multiplier: float = float("nan")
    seed_spread_pct: float = float("nan")

    @property
    def R_sol(self) -> float:
        """``R0 + R1``, **unconditionally**.

        Not split-vs-sum by ρ. The topology puts the two resistances at two places in
        frequency and the corner between them leaves the band as conductivity rises;
        once it has, the optimiser trades between them at near-zero cost and only the
        sum is observable. Reporting ``R1`` alone silently drops a σ-dependent fraction
        of the true resistance — the origin of the apparent non-constant cell constant
        in the KCl campaign.
        """
        return self.R0 + self.R1


def _canonical_order(
    spectra: list[ConstrainedSpectrum],
) -> list[ConstrainedSpectrum]:
    """The set's own order, so the stacked problem does not depend on the caller's.

    **The second half of order invariance, and it is not the seeding.** A fixed seed
    grid makes the *starting point* a function of the set; the residual vector is still
    stacked in the caller's order, so :func:`~scipy.optimize.least_squares` sees permuted
    Jacobian rows and ``trf`` takes a different path through the same surface. Measured
    on the four NIST standards from identical starting points: three orderings gave
    ``Qg`` 1.168e-9 / 1.261e-9 / 1.299e-9 — an 11.3 % spread, with the accuracy barely
    moving (``R_sol`` MAE 2.107 / 2.112 / 2.171 %). Small, and the same kind of thing,
    and there is no reason to keep it.

    Ties are spectra this module cannot tell apart; ``sorted`` is stable, so a repeated
    label still gives a deterministic result for any given call.
    """
    return sorted(spectra, key=lambda s: s.label)


def _stacked_start(
    spectra: list[ConstrainedSpectrum], shared: dict[str, float], tol: float
) -> list[float]:
    """One starting vector for the stacked fit, from one grid entry.

    **Order-free by shape.** Each spectrum's ``R0``/``R1``/``Qd`` are fitted against
    *itself* with the candidate shared triple frozen — :func:`fit_frozen`'s inner loop —
    so the vector this returns is a function of the set, not of the sequence. Nothing
    is carried between spectra.
    """
    x0 = [shared[p] for p in SHARED_PARAMS]
    for spec in spectra:
        start = _seed(spec)
        start.update(shared)
        fitted, _ = _fit_free(spec, start, PER_SPECTRUM_PARAMS, tol)
        x0 += [fitted[p] for p in PER_SPECTRUM_PARAMS]
    return x0


def fit_shared(
    spectra: list[ConstrainedSpectrum],
    *,
    tol: float = CONSTRAINED_FIT_TOL,
    min_arc_resolved: int = MIN_ARC_RESOLVED,
    seed_grid: tuple[dict[str, float], ...] = SHARED_SEED_GRID,
    max_order_spread_pct: float = ORDER_SPREAD_REFUSE_PCT,
    max_qg: float = QG_PLAUSIBLE_MAX,
    max_nfev: int = MAX_NFEV_GLOBAL,
) -> SharedArtifact:
    """Fit ``Qg, ng, nd`` once across *spectra*, with ``R0, R1, Qd`` free per spectrum.

    **Multi-start over a fixed grid, lowest cost kept** — :func:`fit_frozen`'s pattern,
    applied to the stacked problem. The answer does not depend on the order of
    *spectra*, which was the defect this replaced: see :data:`SHARED_SEED_GRID`.

    Three refusals, all raising :class:`InadmissibleFitSet` or a subclass, because a
    refusal is the deliverable for a set that cannot support an artifact. **They ask
    three different questions and the order they run in is the order of increasing
    cost, not of importance:**

    * *can this set identify the shared three at all* — :func:`check_admissible`. Cheap,
      model-free, and it runs before any fitting.
    * *was the minimum reproducible* — the consensus starts disagree by more than
      *max_order_spread_pct* about some spectrum's ``R_sol``,
      :class:`UnstableFitSurface`. This also fires when fewer than two starts reached
      consensus, because a single descent cannot show that its minimum is reproducible
      and "could not judge" must not be spelled the same way as "checked and clean".
    * *is the minimum physical* — ``Qg`` above *max_qg*, :class:`ImplausibleArtifact`.
      **This one is not redundant with the one above and that was measured**: seed grids
      clustered inside the poisoned basin reproduce each other to 0.51 % and 2.06 %
      while being wrong by 7689 % and 8251 %. Reproducibility and correctness are
      different properties and only this refusal asks about the second.

    *max_nfev* bounds **each start**, not the call: ten starts at the default can cost
    ten times it. With one start the budget was the safety net; with ten the other nine
    are, so a start that exhausts its budget reports ``status = 0`` and loses on cost.
    Observed winning starts use 34-734 ``nfev``, so the default is not a tuning knob —
    see :data:`MAX_NFEV_GLOBAL` for what it is and what it is measured to cost.
    """
    admissible = check_admissible(spectra, min_arc_resolved=min_arc_resolved)
    if not admissible.admissible:
        raise InadmissibleFitSet(admissible.describe())
    if not seed_grid:
        raise ValueError("seed_grid is empty — there is nothing to start from")
    if len({tuple(sorted(e.items())) for e in seed_grid}) != len(seed_grid):
        # Two identical starts descend identically, so they would report perfect
        # agreement while constituting one opinion — a clean-looking spread that was
        # never measured. SUBAGENT_RULES §3.1(a).
        raise ValueError("seed_grid has duplicate entries; a repeated start is not a "
                         "second opinion about the minimum")
    if tol > CONSTRAINED_FIT_TOL:
        logger.warning(
            "eis_constrained_fit_loose_tolerance", tol=tol,
            recommended=CONSTRAINED_FIT_TOL,
            msg=("a stacked fit at scipy's default tolerance terminates early and "
                 "returns a converged-looking wrong answer — measured at 17.5% high "
                 "on R_sol with status=3"),
        )

    spectra = _canonical_order(spectra)
    n = len(spectra)
    lo, hi = _pack_bounds(n)

    def stacked(x: np.ndarray) -> np.ndarray:
        return np.concatenate([_residual(spectra[i], _unpack(x, i)) for i in range(n)])

    runs: list[tuple[dict[str, float], Any]] = []
    for candidate in seed_grid:
        x0 = np.clip(np.asarray(_stacked_start(spectra, candidate, tol), dtype=float),
                     lo, hi)
        runs.append((candidate, least_squares(
            stacked, x0, bounds=(lo, hi), method="trf", max_nfev=max_nfev,
            x_scale="jac", xtol=tol, ftol=tol, gtol=tol)))

    seed, res = min(runs, key=lambda r: r[1].cost)
    window = max(res.cost * CONSENSUS_COST_FACTOR,
                 EXACT_COST_PER_RESIDUAL * res.fun.size)
    consensus = [r for _, r in runs if r.cost <= window]

    def r_sol_of(run: Any, i: int) -> float:
        p = _unpack(run.x, i)
        return p["R0"] + p["R1"]

    per_spectrum = [_spread_pct([r_sol_of(r, i) for r in consensus]) for i in range(n)]
    # One unjudgeable spectrum makes the artifact unjudgeable — ``max`` over a list
    # containing NaN is order-dependent, so the NaN is propagated deliberately.
    spread = (float("nan") if any(s != s for s in per_spectrum)
              else max(per_spectrum))

    held = sum(s.shunt.n_held for s in spectra if s.shunt is not None)
    artifact = SharedArtifact(
        Qg=float(res.x[0]), ng=float(res.x[1]), nd=float(res.x[2]),
        labels=tuple(s.label for s in spectra),
        cost=float(res.cost), status=int(res.status), nfev=int(res.nfev),
        n_free=int(res.x.size), n_residuals=int(res.fun.size), tol=float(tol),
        n_shunt_held=held, admissibility=admissible,
        n_starts=len(runs), n_consensus=len(consensus), seed=dict(seed),
        order_spread_pct=spread,
    )
    # Not ``spread > limit``: NaN fails every comparison, so the test is written on the
    # passing side and NaN — "could not judge" — falls through to the refusal. An
    # infinite limit is the one explicit way to ask for the artifact regardless; it is
    # never the default, and ``order_spread_pct`` still reports the NaN.
    if math.isfinite(max_order_spread_pct) and not spread <= max_order_spread_pct:
        raise UnstableFitSurface(
            f"order spread {spread:.4g}% over {len(consensus)}/{len(runs)} consensus "
            f"starts exceeds {max_order_spread_pct:g}% — the stacked minimum is not "
            f"reproducible from independent starting points, so its R_sol is arbitrary "
            f"({artifact.describe()})")
    # After the spread refusal, not before it. Both fire on the shipped poisoned grid,
    # and leaving that case reported as UnstableFitSurface keeps the Wave-1 contract
    # intact; what this adds is the case the spread refusal *passes*.
    reason = implausible_shared(artifact, max_qg=max_qg)
    if reason:
        raise ImplausibleArtifact(f"{reason} ({artifact.describe()})")
    return artifact


def fit_frozen(
    spectrum: ConstrainedSpectrum,
    artifact: SharedArtifact,
    *,
    tol: float = CONSTRAINED_FIT_TOL,
    multipliers: tuple[float, ...] = R_SOL_SEED_MULTIPLIERS,
) -> WellFit:
    """Fit ``R0, R1, Qd`` with the artifact's shared three **frozen**. Multi-start.

    This is the per-spectrum half, and the one an engine could eventually call: it
    takes one spectrum and one frozen artifact and returns a deterministic answer,
    which is the invariant ``constrained_fit_chain.md`` §5 requires of σ.
    """
    best: tuple[dict[str, float], Any, float] | None = None
    sums: list[float] = []
    for m in multipliers:
        start = _seed(spectrum, m)
        start.update(artifact.shared)
        fitted, res = _fit_free(spectrum, start, PER_SPECTRUM_PARAMS, tol)
        sums.append(fitted["R0"] + fitted["R1"])
        if best is None or res.cost < best[1].cost:
            best = (fitted, res, m)
    assert best is not None

    fitted, res, m = best
    return WellFit(label=spectrum.label, R0=fitted["R0"], R1=fitted["R1"],
                   Qd=fitted["Qd"], cost=float(res.cost), ok=bool(res.status > 0),
                   seed_multiplier=float(m), seed_spread_pct=_spread_pct(sums))
