"""The modality-agnostic measurement contract.

Tier 2 component 1 of ``docs/SubAgent docs/afl_comparison_and_restructuring_spec.md``
(§4): the single type every modality's router returns, so the generic layers
(executor, storage, campaign spec) can carry measurement payloads without
knowing what an EIS spectrum, a camera frame or a profilometry trace *is*.

**This module must never import a modality.** No ``EISResult``, no
``analysis.eis``, no instrument types — the spec names "``EISResult``'s shape
leaking into ``MeasurementResult``" as the feared failure mode of this tier, and
an import here is how that leak would start. Conversion lives on the modality
side as a bridge (:meth:`softae.analysis.eis_data.EISResult.to_measurement` /
:meth:`~softae.analysis.eis_data.EISResult.from_measurement`), which is why the
dependency points *inward* to this module and never back out.
``tests/test_measurement_result.py`` asserts the absence of such imports.

The payload lingua franca is :class:`xarray.Dataset`: labelled dimensions and
``attrs`` mean a payload self-describes on disk (Tier 2 component 3 writes these
as netCDF), which a bare ndarray cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import xarray as xr


@dataclass(frozen=True, eq=False)
class MeasurementResult:
    """One measurement, of any modality, as data + provenance.

    Parameters
    ----------
    modality : str
        Registry key for the kind of measurement — ``"eis"``, ``"image"``, … .
        The string (not a Python type) is what storage rows and campaign specs
        carry, so a modality can be added without importing its package.
    data : xarray.Dataset
        The payload. Dimension/coordinate names are the modality's business;
        ``attrs`` must stay netCDF-encodable (str/number/array — no nested
        dicts, no ``None``) so the Dataset alone is enough to reconstruct the
        modality object, with no companion metadata needed. That self-
        sufficiency is what makes the written file, not just the live object,
        the source of truth.
    meta : dict[str, Any]
        Provenance that is *not* constrained by netCDF encoding — conditions,
        settings, timestamps, and context the router knows but the instrument
        result does not (e.g. electrode geometry declared on the workflow
        step). Free-form by design: this is where a caller may put a nested
        dict that ``attrs`` could not hold.
    summary : dict[str, float] | None
        Optional scalar digest for quick indexing/display. Always *derived* —
        never the only home for a value, so dropping it loses nothing.

    Notes
    -----
    **Equality is by identity** (``eq=False``). The generated dataclass
    ``__eq__`` would compare ``self.data == other.data``, and
    ``xarray.Dataset.__eq__`` returns a *Dataset* of elementwise results whose
    truth value is ambiguous — so ``==`` would raise ``ValueError`` rather than
    answer. Compare payloads explicitly with ``Dataset.equals`` / ``identical``,
    which state which kind of equality is meant. Identity equality also keeps
    instances hashable, so they can go in sets and dict keys.

    Frozen guards the *binding*, not the payload: ``Dataset`` and ``dict`` are
    themselves mutable. The freeze is there so a result cannot be silently
    re-pointed at different data after routing, not to make the payload
    immutable — xarray has no frozen mode.
    """

    modality: str
    data: xr.Dataset
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, float] | None = None

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        dims = ", ".join(f"{k}={v}" for k, v in self.data.sizes.items())
        return (
            f"MeasurementResult(modality={self.modality!r}, "
            f"dims=({dims}), vars={list(self.data.data_vars)})"
        )
