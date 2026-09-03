"""Circuit topologies, with their parameters addressed by role rather than by index.

The legacy registry identifies the two resistances positionally, via ``z_indices =
[0, 3]``. That is fragile in a way that has already bitten: ``simpleSaltMembrane``
carried seven parameters and pinned ``R0``/``R1`` as constants, so ``[0, 3]`` indexed
the wrong slots for it, and ``eis_visualizer_widget`` rebuilds fits with ``[0, 1]``
which matches nothing in the registry at all.

**``simpleSaltMembrane`` was retired on 2026-09-02** and no longer exists in either
registry — it never produced a usable fit in the whole stored history, and it raised
before the optimiser on 53 of 54 spectra in the most recent sweep. It is named here
anyway because it is *why this module exists*: retiring the model does not unmake the
defect that role-addressing was built to prevent, and a reader who greps for the name
should find that reason rather than a dangling reference.

Here a model declares which *element names* play ``R_series`` and ``R_bulk``, and the
parameter vector is addressed through those names. Renaming or reordering a circuit
string then cannot silently move which resistance the conductivity is computed from.

The reference topology in framework §1.1 is richer than what ships::

    Z(ω) = jωL_lead + [ R_leak ∥ ( R_series + (R_bulk ∥ C_par) + Z_CPE ) ]

Two of its elements are deliberately absent:

``L_lead``
    Overhaul F5 records fitted inductances of 400–500 µH against a short blank's
    measured 4.18 µH. Lead inductance must be **pinned from the short blank**
    (framework §4.10.3), not fitted — and there is no short blank yet. Until E2
    provides one, the HF-inductive truncation gate handles the artefact instead.
``R_leak``
    The dry fixture showed > 10⁹ Ω, i.e. no DC leakage floor was observed. A free
    parameter with nothing to constrain it would simply absorb other artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from softae.analysis.eis.fitter import COLLAPSED_AT_ZERO as _COLLAPSED_AT_ZERO


@dataclass(frozen=True)
class CircuitModel:
    """A fittable topology and the roles its elements play."""

    name: str
    circuit: str
    roles: dict[str, str]                      # role -> element name
    constants: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def element_for(self, role: str) -> str | None:
        """The element name playing *role*, e.g. ``"R_bulk" -> "R1"``."""
        return self.roles.get(role)


#: Models available to the gated engine. Keys deliberately do not collide with
#: :data:`softae.analysis.circuit_fitting.CIRCUIT_MODELS` — the legacy registry stays
#: exactly as it is, and a name cannot be resolved by the wrong engine by accident.
EIS_CIRCUITS: dict[str, CircuitModel] = {
    "blocking_coplanar": CircuitModel(
        name="blocking_coplanar",
        circuit="R0-CPE0-p(R1,C0)",
        roles={"R_series": "R0", "R_bulk": "R1", "C_par": "C0", "cpe": "CPE0"},
        description=(
            "Blocking coplanar cell: series resistance, blocking-electrode CPE, and "
            "the bulk arc. Topologically the reference core already."
        ),
    ),
    "blocking_coplanar_L": CircuitModel(
        name="blocking_coplanar_L",
        circuit="L0-R0-CPE0-p(R1,C0)",
        roles={"R_series": "R0", "R_bulk": "R1", "C_par": "C0", "cpe": "CPE0",
               "L_lead": "L0"},
        description=(
            "As above with explicit lead inductance. SHIPS UNUSED — L must be pinned "
            "from a short blank, not fitted (overhaul F5)."
        ),
    ),
}

#: What the legacy registry's models map onto, so a stored ``model_name`` from before
#: this work can still be interpreted by role.
LEGACY_ROLE_MAP: dict[str, dict[str, str]] = {
    "simpleSalt": {"R_series": "R0", "R_bulk": "R1"},
    "flexSalt": {"R_series": "R0", "R_bulk": "R1"},
}
# ``simpleSaltMembrane`` was removed here on 2026-09-02 with the model itself. Its entry
# named the same roles this map defaults to (``R_series``/``R0``, ``R_bulk``/``R1``), and
# :func:`fitter.fit_spectrum` rejects an unknown model before it ever reaches
# :func:`roles_for` — so the one stored row from 2026-03-07 (a failed fit, ``R1`` NULL)
# resolves exactly as it did. The deletion is a no-op, which is why it is safe to make.


def roles_for(model_name: str) -> dict[str, str]:
    """Role → element map for either registry, empty when the model is unknown."""
    model = EIS_CIRCUITS.get(model_name)
    if model is not None:
        return dict(model.roles)
    return dict(LEGACY_ROLE_MAP.get(model_name, {}))


#: Magnitude at or below which a parameter has **collapsed** onto a lower bound of
#: exactly zero — re-exported from :mod:`softae.analysis.eis.fitter`, where the
#: bounds arithmetic lives and where the number is justified in full.
#:
#: It is imported rather than restated because the two paths must not be able to
#: disagree about what "collapsed" means. The **gated** path asks
#: :meth:`~softae.analysis.eis.fitter.FitCovariance.pegged`, which owns this test
#: outright. The **legacy** path carries no ``FitCovariance`` at all — its bounds sit
#: in :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS` and nothing on the fit
#: itself knows them — so it has to apply the same rule by hand, in
#: :func:`_collapsed_at_zero` below. One arithmetic per *path*, and one number across
#: both: until 2026-08-31 the constant was written down here as well, and the two
#: readings had already drifted three decades apart.
COLLAPSED_AT_ZERO = _COLLAPSED_AT_ZERO


def _finite(value: Any) -> float:
    """*value* as a float, or NaN when it is not one."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _collapsed_at_zero(
    names: Any, values: Any, lower: Any
) -> tuple[tuple[str, float], ...]:
    """``(name, value)`` for each parameter collapsed onto a **zero** lower bound.

    Separate from the near-a-finite-bound test because the two are different
    geometries, not two tolerances on one geometry — see :data:`COLLAPSED_AT_ZERO`.
    """
    out: list[tuple[str, float]] = []
    for i, name in enumerate(names):
        if i >= len(lower) or i >= len(values):
            break
        if _finite(lower[i]) != 0.0:
            continue
        v = _finite(values[i])
        if v == v and abs(v) <= COLLAPSED_AT_ZERO:
            out.append((str(name), v))
    return tuple(out)


def _rests_on_finite_bound(
    names: Any, values: Any, lower: Any, upper: Any, tol: float
) -> tuple[tuple[str, float, float], ...]:
    """``(name, value, bound)`` for each parameter resting near a finite, non-zero bound.

    :meth:`~softae.analysis.eis.fitter.FitCovariance.pegged`'s rule, spelled here for
    the **legacy** registry only — that path carries no ``FitCovariance``, so its
    bounds live in :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS` and nothing
    on the fit itself knows them. The gated path keeps asking ``pegged`` rather than
    this, so the bound the optimiser actually fitted against stays the one consulted.
    """
    out: list[tuple[str, float, float]] = []
    for i, name in enumerate(names):
        if i >= len(lower) or i >= len(upper) or i >= len(values):
            break
        v = _finite(values[i])
        if v != v:
            continue
        for bound in (_finite(lower[i]), _finite(upper[i])):
            if bound != bound or bound in (float("inf"), float("-inf")) or bound == 0.0:
                continue
            if abs(v - bound) <= tol * max(abs(v), abs(bound)):
                out.append((str(name), v, bound))
                break
    return tuple(out)


def _legacy_parameter_bounds(model_name: str) -> tuple[Any, Any, Any] | None:
    """``(names, lower, upper)`` for a legacy model, or ``None`` when it has no bounds.

    ``None`` is also the answer when the bounds vector and the fitted vector disagree
    in length: an off-by-one there would report the wrong *parameter* as railed, which
    is worse than reporting none, and the R₁ check above has already run either way.
    """
    from softae.analysis.circuit_fitting import CIRCUIT_MODELS

    try:
        spec = CIRCUIT_MODELS[str(model_name)]
        lower, upper = spec["bounds"]
        lower, upper = list(lower), list(upper)
    except (KeyError, TypeError, ValueError):
        return None
    if len(lower) != len(upper):
        return None

    names = parameter_names(str(spec.get("circuit", "")), spec.get("constants") or {})
    if len(names) != len(lower):
        # ``impedance`` absent, or a circuit string the registry and the library read
        # differently. Positional labels still name *which* slot railed.
        names = tuple(f"p{i}" for i in range(len(lower)))
    return names, lower, upper


def _nearest_bounds(cov: Any, names: tuple[str, ...]) -> tuple[tuple[str, float, float], ...]:
    """``(name, value, bound)`` for parameters ``pegged`` already judged to be on one.

    ``pegged`` reports *that* a parameter rests on a constraint, not *which* side, so
    the closer of the two is recovered here purely to put the number in the message.
    Nothing is re-decided: the membership test stays ``pegged``'s.
    """
    lower, upper = cov.bounds if getattr(cov, "bounds", None) is not None else ((), ())
    out: list[tuple[str, float, float]] = []
    for name in names:
        i = cov.index(name)
        v = cov.value(name)
        candidates = [_finite(b[i]) for b in (lower, upper)
                      if i is not None and i < len(b)]
        candidates = [b for b in candidates if b == b and abs(b) != float("inf")]
        bound = min(candidates, key=lambda b: abs(v - b)) if candidates else float("nan")
        out.append((str(name), v, bound))
    return tuple(out)


def _describe(
    collapsed: tuple[tuple[str, float], ...],
    resting: tuple[tuple[str, float, float], ...],
) -> str:
    """One sentence naming every unidentified parameter, with the value and the bound.

    The value is carried because a stored ``error_msg`` is the only place a reader
    later finds out *how* a parameter railed — ``R0`` at 4.6e-62 and ``C0`` at its
    1e-11 floor are both rails and are not the same diagnosis.
    """
    parts = [f"{n} collapsed to {v:g} against a zero lower bound" for n, v in collapsed]
    # ``b != b`` is a NaN bound: ``pegged`` said this parameter rests on one but the
    # side could not be recovered. Say so rather than rendering the word "nan".
    parts += [f"{n} rests on a fitted bound of {b:g} (at {v:g})" if b == b
              else f"{n} rests on a fitted bound (at {v:g})"
              for n, v, b in resting]
    return "; ".join(parts)


def railed_measurand(fit: Any) -> str:
    """Why this fit's measurand is not a measurement, or ``""`` when it is.

    σ is ``K/R_bulk``, so ``R_bulk`` is the measurand and a ``R_bulk`` that came to
    rest on the optimiser's box constraint is not one. 335 of the 1440 fits in run
    ``20260811T023757Z_equilibration_characterization`` (23.3 %) sat on the
    ``simpleSalt`` R₁ floor of 100 Ω and reported ``success = 1`` with
    σ = 0.5 S/cm — roughly seawater, from a dry polymer film. The population is
    unambiguous: 335 rows inside ``[100.000, 100.030]`` Ω and then nothing at all
    until 226.9 Ω, so the tolerance below has an order of magnitude of room.

    **Both resistances are asked, not only the measurand.** Until 2026-08-27 this
    function put exactly one question — *is R_bulk on its bound?* — and discarded the
    answer to every other, including on the gated path where ``pegged()`` had already
    computed them. 449 of 3 618 stored ``simpleSalt`` fits carry an ``R0`` below
    1e-30 Ω, and 222 of those sit beside an R₁ that is nowhere near its floor, so they
    reported ``success = 1`` and stored a σ. That σ is not merely uncertain, it is
    mis-attributed: ``R_series`` and ``R_bulk`` are in series and the optimiser trades
    freely between them, so an ``R_series`` driven to zero means ``R_bulk`` has
    absorbed it and ``R1`` is now ``R_series + R_bulk`` wearing the split's name. That
    is precisely the degenerate split
    :func:`~softae.analysis.eis.engine_support._resolve_reported_resistance` refuses to
    report as ``split_bulk`` on the gated path — and the legacy path has no ρ with
    which to notice it.

    **The watch is the two resistances, and deliberately not "every fitted
    parameter".** Widening it that far was tried first and is wrong, empirically and
    then in principle. Empirically: it turned 14 settle-phase tests in
    ``test_eis_validate.py`` red, because the blocking-electrode CPE runs to its
    constraint on ordinary spectra - ``CPE0_0`` at its 9e-6 ceiling and ``CPE0_1`` at
    0.9 - so every channel on the board was demoted, sigma went null, the survivor set
    fell below ``DEFAULT_SETTLE_MIN_CHANNELS`` and the verdict went from ``ceiling``
    to ``not_evaluable``. A gate that fires on the normal condition certifies nothing.
    On the stored corpus the same widening demotes 539 of 3 618 fits (14.9 %) against
    the 222 that are the actual defect. In principle: sigma is ``K/R``, and ``R`` is
    built from ``R_series`` and ``R_bulk`` alone. The CPE and ``C_par`` terms are
    nuisance parameters, and this module already draws that line -
    :meth:`~softae.analysis.eis.fitter.FitCovariance.rel_se` exists because "a nuisance
    parameter may legitimately be poorly determined, but the measurand may not". A
    pegged CPE says the electrode response is unidentified, which is worth knowing and
    is not the same claim as *this conductivity is not a measurement*. If that should
    become a gate, it deserves its own threshold and its own evidence rather than
    arriving as a side effect of this one.

    The bound is never written down here. It is read from whichever registry
    actually fitted the spectrum:

    * the **gated** path carries the bounds it fitted against on
      :class:`~softae.analysis.eis.fitter.FitCovariance`, whose
      :meth:`~softae.analysis.eis.fitter.FitCovariance.pegged` names every parameter
      resting on one — so that is asked rather than re-derived, and it is asked for
      the zero-bound case too. Between ``443c948`` and 2026-08-31 a second,
      absolute test was supplied alongside it, because ``pegged`` was then a purely
      relative rule and a relative rule cannot express a bound of exactly zero. That
      supplement is **retired**: ``pegged`` now expresses it, and a second
      implementation on the same path is how two answers to one question start to
      disagree;
    * the **legacy** path declares them in
      :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS`, read through
      :func:`~softae.analysis.equilibration.r1_lower_bound_ohms` — the one
      existing spelling of "``bounds[0][z_indices[1]]``", reused rather than
      restated so a bound edited in the registry moves both readers at once.

    Detection is *near*, not *at*, the bound (``RAILED_R1_TOL_REL``): bounded
    least squares stops within its own step size of a constraint, so an equality
    test would miss most railed fits.

    **R₁'s own message is unchanged, deliberately.** It names the model and the bound's
    value, and downstream readers tell a railed row from a non-converged one by that
    string, so the R₁ case is answered first on both paths and returns exactly what it
    always returned. The parameters added here can only ever produce a reason where
    there was none before; none of them can change the reason R₁ gives.
    """
    from softae.analysis.equilibration import (
        RAILED_R1_TOL_REL,
        is_railed,
        r1_lower_bound_ohms,
    )

    model_name = str(getattr(fit, "model_name", "") or "")
    roles = roles_for(model_name)
    bulk = roles.get("R_bulk", "R1")
    watched = {roles.get("R_series", "R0"), bulk}

    cov = getattr(fit, "covariance", None)
    if cov is not None:
        pegged = tuple(n for n in cov.pegged() if n in watched)
        if bulk in pegged:
            return f"{bulk} rests on a fitted bound"
        if not pegged:
            return ""
        # Membership is entirely ``pegged``'s. The bound is recovered here only to
        # choose which sentence describes it: resting *on zero* is a collapse and
        # reads as one, which is a different diagnosis from resting on a fitted
        # floor. Nothing is re-decided, so the two cannot disagree.
        nearest = _nearest_bounds(cov, pegged)
        collapsed = tuple((n, v) for n, v, b in nearest if b == 0.0)
        return _describe(collapsed, tuple(r for r in nearest if r[2] != 0.0))

    bound = r1_lower_bound_ohms(model_name)
    if is_railed(getattr(fit, "R1", None), bound):
        return f"R1 rests on the '{model_name}' lower bound of {bound:g} ohm"

    declared = _legacy_parameter_bounds(model_name)
    if declared is None:
        return ""
    names, lower, upper = declared
    # ``list(...)`` and never ``... or ()``: ``parameters`` is an ndarray on this path,
    # and ``or`` on one raises rather than falling back.
    try:
        values = list(getattr(fit, "parameters", ()))
    except TypeError:
        return ""
    if len(values) != len(lower):
        return ""
    collapsed = tuple(c for c in _collapsed_at_zero(names, values, lower)
                      if c[0] in watched)
    resting = tuple(
        r for r in _rests_on_finite_bound(names, values, lower, upper, RAILED_R1_TOL_REL)
        if r[0] in watched and not any(r[0] == c for c, _ in collapsed))
    if not collapsed and not resting:
        return ""
    return _describe(collapsed, resting)


def parameter_names(circuit: str, constants: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Fitted parameter names in vector order, e.g. ``("R0","CPE0_0","CPE0_1","R1","C0")``.

    Delegates to ``impedance.py`` so the naming can never drift from the library
    actually doing the fitting. Returns ``()`` when the backend is unavailable —
    callers fall back to positional access, which is what they did before.
    """
    try:
        from impedance.models.circuits.fitting import (  # type: ignore
            calculateCircuitLength,
            extract_circuit_elements,
        )
    except Exception:
        return ()

    held = set(constants or {})
    names: list[str] = []
    try:
        for element in extract_circuit_elements(circuit):
            n_params = calculateCircuitLength(element)
            if element in held:
                continue
            if n_params == 1:
                names.append(element)
            else:
                names.extend(f"{element}_{i}" for i in range(n_params))
    except Exception:
        return ()
    return tuple(names)
