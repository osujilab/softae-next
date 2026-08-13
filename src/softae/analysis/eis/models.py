"""Circuit topologies, with their parameters addressed by role rather than by index.

The legacy registry identifies the two resistances positionally, via ``z_indices =
[0, 3]``. That is fragile in a way that has already bitten: ``simpleSaltMembrane``
carries seven parameters and pins ``R0``/``R1`` as constants, so ``[0, 3]`` indexes
the wrong slots for it, and ``eis_visualizer_widget`` rebuilds fits with ``[0, 1]``
which matches nothing in the registry at all.

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
    "simpleSaltMembrane": {"R_series": "R0", "R_bulk": "R1"},
}


def roles_for(model_name: str) -> dict[str, str]:
    """Role → element map for either registry, empty when the model is unknown."""
    model = EIS_CIRCUITS.get(model_name)
    if model is not None:
        return dict(model.roles)
    return dict(LEGACY_ROLE_MAP.get(model_name, {}))


def railed_measurand(fit: Any) -> str:
    """Why this fit's measurand is not a measurement, or ``""`` when it is.

    σ is ``K/R_bulk``, so ``R_bulk`` is the measurand and a ``R_bulk`` that came to
    rest on the optimiser's box constraint is not one. 335 of the 1440 fits in run
    ``20260811T023757Z_equilibration_characterization`` (23.3 %) sat on the
    ``simpleSalt`` R₁ floor of 100 Ω and reported ``success = 1`` with
    σ = 0.5 S/cm — roughly seawater, from a dry polymer film. The population is
    unambiguous: 335 rows inside ``[100.000, 100.030]`` Ω and then nothing at all
    until 226.9 Ω, so the tolerance below has an order of magnitude of room.

    The bound is never written down here. It is read from whichever registry
    actually fitted the spectrum:

    * the **gated** path carries the bounds it fitted against on
      :class:`~softae.analysis.eis.fitter.FitCovariance`, whose
      :meth:`~softae.analysis.eis.fitter.FitCovariance.pegged` already names every
      parameter resting on one;
    * the **legacy** path declares them in
      :data:`~softae.analysis.circuit_fitting.CIRCUIT_MODELS`, read through
      :func:`~softae.analysis.equilibration.r1_lower_bound_ohms` — the one
      existing spelling of "``bounds[0][z_indices[1]]``", reused rather than
      restated so a bound edited in the registry moves both readers at once.

    Detection is *near*, not *at*, the bound (``RAILED_R1_TOL_REL``): bounded
    least squares stops within its own step size of a constraint, so an equality
    test would miss most railed fits.
    """
    from softae.analysis.equilibration import is_railed, r1_lower_bound_ohms

    model_name = str(getattr(fit, "model_name", "") or "")
    cov = getattr(fit, "covariance", None)
    if cov is not None:
        bulk = roles_for(model_name).get("R_bulk", "R1")
        if bulk in cov.pegged():
            return f"{bulk} rests on a fitted bound"
        return ""

    bound = r1_lower_bound_ohms(model_name)
    if not is_railed(getattr(fit, "R1", None), bound):
        return ""
    return f"R1 rests on the '{model_name}' lower bound of {bound:g} ohm"


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
