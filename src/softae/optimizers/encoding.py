"""Shared parameter-space encoding for optimizers.

Floats/ints pass through as a single numeric axis; categoricals are one-hot
encoded (one axis per choice).  Both
:class:`~softae.optimizers.bayesian.BayesianOptimizer` and
:class:`~softae.optimizers.pooled_bayesian.PooledBayesianOptimizer` encode the
same way so their surrogates see an identical feature layout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class OneHotEncoder:
    """Encode/decode parameter dicts against a fixed parameter space.

    Parameters
    ----------
    parameter_space
        The optimizer's search-space definition (see :class:`BaseOptimizer`).
    """

    def __init__(self, parameter_space: dict[str, dict[str, Any]]) -> None:
        self._parameter_space = parameter_space
        #: name → ordered list of categorical choices.
        self.cat_maps: dict[str, list[Any]] = {
            name: list(spec["choices"])
            for name, spec in parameter_space.items()
            if spec["type"] == "categorical"
        }

    def encode(self, params: dict[str, Any]) -> list[float]:
        """Encode a parameter dict into a flat numeric vector."""
        vec: list[float] = []
        for name, spec in self._parameter_space.items():
            if spec["type"] in ("float", "int"):
                vec.append(float(params[name]))
            else:  # categorical → one-hot
                for c in self.cat_maps[name]:
                    vec.append(1.0 if params[name] == c else 0.0)
        return vec

    def key(self, params: dict[str, Any]) -> tuple:
        """Float-robust hashable identity for a point (rounded encode)."""
        return tuple(round(v, 9) for v in self.encode(params))

    def decode(self, vec: np.ndarray) -> dict[str, Any]:
        """Decode a numeric vector back to a parameter dict."""
        params: dict[str, Any] = {}
        idx = 0
        for name, spec in self._parameter_space.items():
            if spec["type"] == "float":
                params[name] = float(vec[idx])
                idx += 1
            elif spec["type"] == "int":
                params[name] = int(round(vec[idx]))
                idx += 1
            else:  # categorical
                choices = self.cat_maps[name]
                one_hot = vec[idx : idx + len(choices)]
                params[name] = choices[int(np.argmax(one_hot))]
                idx += len(choices)
        return params
