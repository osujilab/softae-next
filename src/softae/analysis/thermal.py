"""Thermal-model selection: a unified entry point for Arrhenius / VFT fitting.

The temperature-sweep pipeline fits conductivity vs temperature with one of two
models, chosen explicitly by the user:

* ``"arrhenius"`` — :class:`~softae.analysis.arrhenius.ArrheniusFitter`
  (σ = A·exp(−Eₐ/k_BT); a straight line in 1/T).
* ``"vft"`` — :class:`~softae.analysis.vft.VftFitter`
  (σ = A·exp(−B/(T−T₀)); curved, for glassy/segmental-motion transport).

Both fitters share the ``fit(temperatures_C, conductivities, *, channel, run_id)``
signature and return a result carrying ``model``, ``R_squared``, ``n_points``,
``fit_success`` and ``error_msg`` — so callers can stay model-agnostic and the
data store can persist either in one unified table.
"""

from __future__ import annotations

from softae.analysis.arrhenius import ArrheniusFitter, ArrheniusResult
from softae.analysis.vft import VftFitter, VftResult
from softae.errors import AnalysisError

#: A fit from either model.
ThermalResult = ArrheniusResult | VftResult

#: Registry mapping model name → fitter class.
THERMAL_MODELS: dict[str, type] = {
    "arrhenius": ArrheniusFitter,
    "vft": VftFitter,
}


def make_fitter(model: str):
    """Return a fitter instance for *model* (``"arrhenius"`` or ``"vft"``)."""
    try:
        return THERMAL_MODELS[model]()
    except KeyError:
        raise AnalysisError(
            f"unknown thermal model '{model}'; available: {sorted(THERMAL_MODELS)}"
        ) from None


def fit_thermal(
    model: str,
    temperatures_C: list[float],
    conductivities: list[float],
    *,
    channel: int = 0,
    run_id: str = "",
) -> ThermalResult:
    """Fit a σ(T) series with the selected thermal *model*.

    Minimum points: Arrhenius needs ≥ 2, VFT needs ≥ 3 (it has a third parameter,
    T₀).  An under-determined fit returns a result with ``fit_success=False``
    rather than raising.
    """
    return make_fitter(model).fit(
        temperatures_C, conductivities, channel=channel, run_id=run_id
    )
