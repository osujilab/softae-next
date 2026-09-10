"""Compile a circuit expression **once** instead of on every residual evaluation.

``impedance.py`` evaluates a circuit by *generating Python source and* ``eval``-*ing it*.
:func:`impedance.models.circuits.fitting.wrapCircuit` returns a closure whose body is::

    x = eval(buildCircuit(circuit, frequencies, *parameters, ...)[0], circuit_elements)

and ``buildCircuit`` splices the **numeric literals** — every parameter, and the whole
frequency list once per element — into that source. Both halves therefore rerun on
*every* residual call: the recursive parse, and the compile of a string that for a
41-point five-element circuit runs to several kilobytes.

Measured on one gated ``analyze_spectrum`` (41 points, shipped config), 3.946 s total:

======================== ============ ===============
call                      calls        tottime
======================== ============ ===============
``builtins.eval``          16,791         1.746 s
``buildCircuit``            6,678         0.961 s
``eval_linKK``                 84         0.259 s
======================== ============ ===============

**The circuit string is constant for the duration of a fit; only the numbers move.** So
the parse and the compile are pure repetition, and this module removes them: the
expression is built once with *symbols* where the numbers go, compiled to a code object,
and evaluated per residual with the actual values bound as names. Measured per-residual
cost on the shipped topologies falls from ~0.71-0.85 ms to ~0.07-0.09 ms — a factor of 10.

Why the template is built by ``buildCircuit`` itself
----------------------------------------------------
The obvious implementation reimplements ``buildCircuit``'s recursive parser to emit names
instead of literals — and then owns a second, silently drifting copy of a third-party
grammar. This module instead calls the *real* ``buildCircuit`` and passes it objects whose
``repr`` is the name we want::

    buildCircuit("R0-p(CPE0,R1)", _Symbol("_f_"), *symbols, constants={})
    -> 's([R([_p_[0]],_f_),p([CPE([_p_[1], _p_[2]],_f_),R([_p_[3]],_f_)])])'

``buildCircuit`` reaches its arguments only through ``np.array(...).tolist()`` and
``str``/``repr``, so a symbol survives the round trip untouched and the parse, the element
lookup, the constant substitution and the parameter indexing all remain the library's own.
Held constants keep their literal spelling, exactly as before, because they do not move.

Bitwise identity, not "close enough"
------------------------------------
``str`` of a Python float is its shortest round-tripping repr, so the literal
``buildCircuit`` splices in is *the same double* as the value it came from. Binding that
value to a name instead of spelling it into source therefore hands the element functions
identical arguments, and the arithmetic below is untouched — same element callables, same
``circuit_elements`` namespace, same ``np.hstack([real, imag])``. The output is bitwise
equal, and that is asserted rather than assumed:

* :func:`wrap_circuit` verifies its first call against ``wrapCircuit`` and falls back
  permanently if they differ, so the fast path cannot silently ship a different number;
* ``tests/test_eis_circuit_expr.py`` covers every shipped topology and includes a
  *positive control* — a deliberately corrupted template must be caught by that check.

Non-finite parameters raise, deliberately
-----------------------------------------
``buildCircuit`` spells a NaN parameter as the bare token ``nan``, which is not a name in
``circuit_elements``, so the reference path raises ``NameError`` and the fit is reported as
a failure. Binding values instead of spelling them would *tolerate* NaN and return a NaN
residual — a different outcome, on a population the caller currently treats as failed.
:func:`wrap_circuit` therefore reproduces the raise rather than the tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["wrap_circuit", "circuit_template"]

#: Symbols offered to ``buildCircuit``. It consumes exactly as many as the topology needs
#: and ignores the rest, so this only has to be an upper bound — the shipped circuits use
#: 4-6. Over-supplying is what lets the parameter count be *read back* from the library's
#: own index rather than derived by a second implementation of its counting rules.
_MAX_PARAMS = 64


class _Symbol:
    """A number's stand-in inside ``buildCircuit``'s generated source.

    ``buildCircuit`` renders parameters through ``repr`` (they end up as list elements)
    and frequencies through ``str`` (``str(frequencies)``), so both must yield the name.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    def __repr__(self) -> str:
        return self._text

    __str__ = __repr__


def _key(constants: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(k), v) for k, v in dict(constants or {}).items()))


@lru_cache(maxsize=64)
def _compiled(circuit: str, constants_key: tuple[tuple[str, float], ...]):
    """``(code, n_params, namespace, source)`` for one topology, built once per process."""
    from impedance.models.circuits.fitting import (  # type: ignore
        buildCircuit,
        circuit_elements,
    )

    symbols = tuple(_Symbol(f"_p_[{i}]") for i in range(_MAX_PARAMS))
    source, n_params = buildCircuit(
        circuit, _Symbol("_f_"), *symbols,
        constants=dict(constants_key), eval_string="", index=0,
    )
    if not isinstance(source, str) or "_f_" not in source:
        raise ValueError(f"symbolic build produced no template for {circuit!r}")
    # A copy: the library's ``circuit_elements`` is a module global, and ``eval`` writes
    # ``__builtins__`` into whatever mapping it is handed.
    namespace = dict(circuit_elements)
    return compile(source, f"<circuit {circuit}>", "eval"), int(n_params), namespace, source


def circuit_template(circuit: str, constants: Mapping[str, float] | None = None) -> str:
    """The generated source for *circuit*, for tests and diagnostics."""
    return _compiled(circuit, _key(constants))[3]


def wrap_circuit(
    circuit: str, constants: Mapping[str, float] | None = None
) -> Callable[..., np.ndarray]:
    """A drop-in for ``impedance``'s ``wrapCircuit`` that parses the circuit once.

    Returns ``f(frequencies, *parameters)`` giving ``hstack([Re Z, Im Z])`` — the same
    signature, the same convention, and the same numbers. On the first call the result is
    checked against ``wrapCircuit``; a mismatch (or any failure to build the template)
    demotes the closure to the reference implementation for good and logs it, so an
    ``impedance`` upgrade that broke the symbolic build would cost speed, not correctness.
    """
    from impedance.models.circuits.fitting import wrapCircuit  # type: ignore

    held = dict(constants or {})
    reference = wrapCircuit(circuit, held)

    try:
        code, _n_params, namespace, _source = _compiled(circuit, _key(held))
    except Exception:
        logger.warning("eis_circuit_template_unavailable", circuit=circuit, exc_info=True)
        return reference

    state = {"verified": False, "fallback": False}

    def wrapped(frequencies: Any, *parameters: float) -> np.ndarray:
        values = np.array(parameters).tolist()
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            # See the module docstring: the reference path raises on these, and matching
            # it is the point. Spelled as NameError because that is what ``eval`` of the
            # token ``nan`` raises, and no caller keys off anything finer than the type.
            raise NameError("non-finite circuit parameter")
        x = eval(code, namespace, {"_p_": values, "_f_": np.array(frequencies).tolist()})
        fast = np.hstack([np.real(x), np.imag(x)])

        if not state["verified"]:
            ref = reference(frequencies, *parameters)
            if not (ref.shape == fast.shape and np.array_equal(ref, fast, equal_nan=True)):
                logger.warning(
                    "eis_circuit_template_mismatch", circuit=circuit,
                    msg="compiled circuit disagreed with impedance.wrapCircuit; "
                        "falling back to the reference evaluator",
                )
                state["fallback"] = True
                return ref
            state["verified"] = True
        return fast

    def guarded(frequencies: Any, *parameters: float) -> np.ndarray:
        if state["fallback"]:
            return reference(frequencies, *parameters)
        return wrapped(frequencies, *parameters)

    return guarded
