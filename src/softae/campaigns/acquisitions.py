"""Backward-compat shim: acquisitions moved to :mod:`softae.optimizers.acquisitions`.

The acquisition module was relocated down a layer (``campaigns`` → ``optimizers``)
to remove an upward import (``optimizers.pooled_bayesian`` importing ``campaigns``).
This shim re-exports every public name so existing
``from softae.campaigns.acquisitions import X`` sites keep working.
"""

from __future__ import annotations

from softae.optimizers.acquisitions import *  # noqa: F401,F403
from softae.optimizers.acquisitions import (
    ACQUISITIONS,
    AcquisitionStrategy,
    EiAcquisition,
    IntegratedVarianceAcquisition,
    MaxVarianceAcquisition,
    UcbAcquisition,
    UncertaintyWeightedAcquisition,
    make_acquisition,
)

__all__ = [
    "ACQUISITIONS",
    "AcquisitionStrategy",
    "EiAcquisition",
    "IntegratedVarianceAcquisition",
    "MaxVarianceAcquisition",
    "UcbAcquisition",
    "UncertaintyWeightedAcquisition",
    "make_acquisition",
]
